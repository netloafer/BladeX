"""2026-09-03 拍板 e（MQ-CA6）：装配只对 `BLADEX_ASSEMBLY_AGENTS` 清单内的 agent 开。

钉四件事：
① 默认清单 = `claude-code`（flags.py 单一真相源，与 `.env.example` 对账）；
② 不在清单的 agent 走 `assembler=None` **逐字相同**的原样返回（消息与 cap_info 都不变）；
③ 清单内 agent（含 base 匹配）行为不变——止血不许顺手改 CC；
④ 不在清单的 agent 仍打 `assembly_done`（evidence_degraded=0、chars 前后相等）——仪器不关。
"""

from __future__ import annotations

import re
from pathlib import Path

import structlog
from bladex_core.flags import MEMORY_TEXT_DEFAULTS, flag_csv, flag_text
from bladex_proxy.assembly import AssemblyConfig, ContextAssembler, assembly_enabled_for
from bladex_proxy.inject import _apply_cap

_ENV_EXAMPLE = Path(__file__).resolve().parents[3] / "config" / ".env.example"


def test_text_defaults_table_and_env_example_agree():
    assert MEMORY_TEXT_DEFAULTS == {"BLADEX_ASSEMBLY_AGENTS": "claude-code"}
    text = _ENV_EXAMPLE.read_text(encoding="utf-8")
    for key, default in MEMORY_TEXT_DEFAULTS.items():
        m = re.search(rf'^\s*export\s+{key}="?([^"\n]*)"?\s*$', text, re.MULTILINE)
        assert m, f"config/.env.example 未以 `export {key}=\"...\"` 形态声明"
        assert m.group(1).strip() == default, \
            f"{key}: .env.example 写 {m.group(1)!r}，flags.py 默认 {default!r}——两侧必须一致"


def test_flag_text_and_csv(monkeypatch):
    monkeypatch.delenv("BLADEX_ASSEMBLY_AGENTS", raising=False)
    assert flag_text("BLADEX_ASSEMBLY_AGENTS") == "claude-code"
    assert flag_csv("BLADEX_ASSEMBLY_AGENTS") == frozenset({"claude-code"})
    monkeypatch.setenv("BLADEX_ASSEMBLY_AGENTS", " claude-code, hermes ,, ")
    assert flag_csv("BLADEX_ASSEMBLY_AGENTS") == frozenset({"claude-code", "hermes"})


def test_enabled_for_matrix(monkeypatch):
    monkeypatch.delenv("BLADEX_ASSEMBLY_AGENTS", raising=False)
    assert assembly_enabled_for("claude-code") is True
    assert assembly_enabled_for("hermes:default") is False
    assert assembly_enabled_for("codex") is False
    assert assembly_enabled_for("unknown-9d9bdd3e") is False
    assert assembly_enabled_for("") is True, "无身份的直调（测试/脚本）无对象可 gate ⇒ 开"
    monkeypatch.setenv("BLADEX_ASSEMBLY_AGENTS", "hermes")
    assert assembly_enabled_for("hermes:default") is True, "base 匹配"
    assert assembly_enabled_for("claude-code") is False
    monkeypatch.setenv("BLADEX_ASSEMBLY_AGENTS", "*")
    assert assembly_enabled_for("anything") is True
    monkeypatch.setenv("BLADEX_ASSEMBLY_AGENTS", "")
    assert assembly_enabled_for("claude-code") is False, "空清单 = 全关"


def _long_messages(n: int) -> list[dict]:
    msgs = [{"role": "system", "content": "You are helpful."}]
    for i in range(n):
        msgs.append({"role": "user", "content": f"q{i}"})
        msgs.append({"role": "assistant", "content": None,
                     "tool_calls": [{"id": f"c{i}", "type": "function",
                                     "function": {"name": "read", "arguments": "{}"}}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": "x" * 3000})
        msgs.append({"role": "assistant", "content": f"a{i}"})
    return msgs


def _asm() -> ContextAssembler:
    return ContextAssembler(AssemblyConfig(enabled=True, msg_threshold=10,
                                           evidence_min_chars=500, preserve_units=2))


def test_not_listed_agent_is_verbatim_assembler_none_path(monkeypatch):
    monkeypatch.delenv("BLADEX_ASSEMBLY_AGENTS", raising=False)
    msgs = _long_messages(8)
    none_out, none_info = _apply_cap([dict(m) for m in msgs], None, [], False,
                                     "s1", agent_id="hermes:default")
    out, info = _apply_cap([dict(m) for m in msgs], _asm(), [], False,
                           "s1", agent_id="hermes:default")
    assert out == msgs and out == none_out
    assert info == none_info, "不在清单 ⇒ 与 assembler=None 路径逐字相同（含 cap_info）"


def test_listed_agent_still_assembles(monkeypatch):
    monkeypatch.delenv("BLADEX_ASSEMBLY_AGENTS", raising=False)
    msgs = _long_messages(8)
    out, info = _apply_cap([dict(m) for m in msgs], _asm(), [], False,
                           "s2", agent_id="claude-code")
    assert info["triggered"] is True and info["evidence_degraded"] > 0, \
        "CC 行为不变：closed 单元巨型 tool 结果照常降解"
    assert info["chars_after"] < info["chars_before"]


def test_not_listed_agent_still_emits_assembly_done(monkeypatch):
    monkeypatch.delenv("BLADEX_ASSEMBLY_AGENTS", raising=False)
    events: list[dict] = []

    def _sink(logger, method, event_dict):
        events.append(dict(event_dict))
        raise structlog.DropEvent

    structlog.configure(processors=[_sink])
    try:
        _apply_cap(_long_messages(3), _asm(), [], False, "s3", agent_id="codex")
    finally:
        structlog.reset_defaults()
    done = [e for e in events if e.get("event") == "assembly_done"]
    assert len(done) == 1, "仪器不关：不在清单的 agent 也要有 assembly_done 读数"
    e = done[0]
    assert e["agent_id"] == "codex" and e["evidence_degraded"] == 0
    assert e["chars_before"] == e["chars_after"] > 0
    assert e.get("skipped") == "agent_not_enabled"
