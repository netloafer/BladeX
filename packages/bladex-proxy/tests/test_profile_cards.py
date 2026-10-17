"""U10（ADR-0026 R5）：画像**写侧捕获**与工具习惯。

2026-09-18 批 S（拍板 0.3.0-(5)，ADR-0034 §3.2）：①平面画像卡 / 习惯卡的**注入渲染**
（`get_profile_cards`）删除——注入面 09-03 起不读它，零生产调用点，live 484 轮 `planes=False`。
写侧捕获（规则文件 / 工具计数）与投影数据源（`get_profile_md` / `get_tool_stats` /
`get_rule_files`）保留：它们喂 USER.md / AGENT.md 投影与 `/admin/profiles`。本文件改守这一面：
规则文件被动捕获（确定性）/ 工具计数器聚合 / per-agent 隔离 / 重建等价 / 默认关零回归 /
渲染入口不许回来。
"""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest
from bladex_core.profile import detect_rule_files
from bladex_proxy.models import Identity, ToolEvent, Turn, TurnStatus
from bladex_proxy.storage.memory_hub import MemoryHub
from bladex_proxy.storage.memory_index import MemoryIndex


class MockEmbedder:
    def embed(self, texts):
        results = []
        for t in texts:
            h = hash(t) % 100
            vec = [0.0] * 64
            vec[h % 64] = 1.0
            results.append(vec)
        return results

    @property
    def available(self):
        return True


_RULE_TEXT = ("Contents of CLAUDE.md (project instructions):\n"
              "# BladeX — AI 协作开发指南\n跑测试只用 .venv/bin/python -m pytest。")


def test_detect_rule_files_deterministic():
    got = detect_rule_files(_RULE_TEXT)
    assert got and got[0][0] == "CLAUDE.md"
    assert got == detect_rule_files(_RULE_TEXT)  # 确定性
    assert detect_rule_files("普通消息没有规则文件") == []


def _put_turn(ledger: MemoryHub, *, agent: str, idx: int, system: str = "",
              tools: list[str] | None = None) -> str:
    identity = Identity(user_id="u1", agent_id=agent, session_id=f"s-{agent}",
                        turn_index=idx)
    events = [ToolEvent(tool_name=t, direction="call") for t in (tools or [])]
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user",
                 "content": f"这是第 {idx} 条足够长的普通用户消息，用于消费。"})
    turn = Turn(identity=identity, model="m", request_messages=msgs,
                response_text="ok", tool_events=events, status=TurnStatus.OK,
                ts=datetime(2026, 8, 1, 12, 0, idx, tzinfo=UTC))
    key = identity.storage_key(f"1719900{idx}-0")
    ledger.put(key, turn)
    return key


def _make_hub(tmpdir: str) -> MemoryHub:
    ledger = MemoryHub(Path(tmpdir) / "rocksdb")
    ledger.open()
    _put_turn(ledger, agent="codex", idx=0, system=_RULE_TEXT,
              tools=["bash", "bash", "read_file"])
    _put_turn(ledger, agent="codex", idx=1, tools=["bash"])
    _put_turn(ledger, agent="hermes:default", idx=2, tools=["web_search"])
    return ledger


def _stats(index: MemoryIndex, base: str) -> dict[str, int]:
    """U10 工具计数器（`profile/agent/<base>` 的 tools 表）；E7.4 结果维度统计另有 `get_tool_stats`。"""
    tools = index._profile_get(f"profile/agent/{base}").get("tools", {})  # noqa: SLF001
    return {k: int(v) for k, v in tools.items()}


def test_profile_capture_writes_rulefile_and_tool_stats(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BLADEX_PROFILE_CARDS", "1")
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger = _make_hub(tmpdir)
        index = MemoryIndex(Path(tmpdir) / "index", embedder=MockEmbedder(), read_only=False)
        index.open()
        index.rebuild_from_hub(ledger)
        assert "claude.md" in [n.lower() for n in index.get_rule_files("codex")]
        assert _stats(index, "codex") == {"bash": 3, "read_file": 1}
        index.close()
        ledger.close()


def test_tool_stats_are_per_agent(monkeypatch: pytest.MonkeyPatch):
    """codex 的工具习惯不混进 hermes（audience 隔离在写侧就成立）。"""
    monkeypatch.setenv("BLADEX_PROFILE_CARDS", "1")
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger = _make_hub(tmpdir)
        index = MemoryIndex(Path(tmpdir) / "index", embedder=MockEmbedder(), read_only=False)
        index.open()
        index.rebuild_from_hub(ledger)
        assert _stats(index, "hermes") == {"web_search": 1}
        index.close()
        ledger.close()


def test_profile_rebuild_equivalence(monkeypatch: pytest.MonkeyPatch):
    """两个独立 Memory Index 重放同一 Memory Hub → 投影数据源逐字一致（G6）。"""
    monkeypatch.setenv("BLADEX_PROFILE_CARDS", "1")
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger = _make_hub(tmpdir)
        rendered = []
        for name in ("p2a", "p2b"):
            index = MemoryIndex(Path(tmpdir) / name, embedder=MockEmbedder(), read_only=False)
            index.open()
            index.rebuild_from_hub(ledger)
            rendered.append((index.get_rule_files("codex"), _stats(index, "codex"),
                             index.get_profile_md("u1", "codex")))
            index.close()
        assert rendered[0] == rendered[1] and rendered[0][1]
        ledger.close()


def test_profile_explicitly_off(monkeypatch: pytest.MonkeyPatch):
    """显式关闭：写侧不捕获（回退保证；ADR-0027 §5.4 起默认开，回滚通道不变）。"""
    monkeypatch.setenv("BLADEX_PROFILE_CARDS", "0")
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger = _make_hub(tmpdir)
        index = MemoryIndex(Path(tmpdir) / "index", embedder=MockEmbedder(), read_only=False)
        index.open()
        index.rebuild_from_hub(ledger)
        assert _stats(index, "codex") == {}
        assert index.get_rule_files("codex") == []
        index.close()
        ledger.close()


def test_injection_renderers_do_not_come_back():
    """废除清单的可 grep 形态（ADR-0034 §5）：三张常驻卡的注入渲染入口不许回来。"""
    assert not hasattr(MemoryIndex, "get_profile_cards")
    assert not hasattr(MemoryIndex, "project_card_for")
