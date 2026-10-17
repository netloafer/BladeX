"""批 U · U1.4 —— ADR-0033 §2.6 裁法 + §2.7 末段 reconstruction（取代 ADR-0019 L1 按位置降解）。

钉的事：
① 裁法：被判 trimmed 的**已闭合**单元，tool 结果 → 可见标记；user 原文 / conclusion / tool_calls /
   消息条数与顺序原样；**open 单元永不裁**（即使判定表里有它）；kept 单元原样。
② tool_call_id 配对不动（反向变异 5：裁剪时丢掉 tool_call_id ⇒ `test_trim_keeps_tool_call_pairing` 红）。
③ 标记是原文的纯函数：同一条消息同一单元逐字相同（保 cache；复原靠它）。
④ `degrade_plan` 档位 = kept | trimmed，附 unit_key 与判定来源；`replay_trim` 从 Hub 原文逐字复原。
⑤ 冷轮判定冻结、热轮原样重放（不新增裁剪，前缀逐字稳定）。
⑥ 判据⑤：两级都关 ⇒ `assembly_done triggered=1 reviewed=0` 且正文逐字不变（反向变异 3：都关仍裁 ⇒ 红）。
⑦ 信息守恒：请求原样入库——裁掉的证据原文仍在 Hub（`Turn.request_messages` 是 agent 原文）。
⑧ 端到端：`do_inject_async` 冷轮 → reconstruction 带 verdicts + degrade_plan → ReconstructionRecord 可存。
"""

from __future__ import annotations

import asyncio

import pytest
from bladex_core.task_unit import build_task_units
from bladex_proxy import assembly_review as ar
from bladex_proxy import inject
from bladex_proxy.assembly import (
    EVIDENCE_TRIMMED_OPEN,
    AssemblyConfig,
    ContextAssembler,
    evidence_marker,
    replay_trim,
)
from bladex_proxy.cache_state import registry
from bladex_proxy.config import ProxyConfig
from bladex_proxy.models import ReconstructionRecord
from structlog.testing import capture_logs

_VEC = {"alpha": [1.0, 0.0], "beta": [0.0, 1.0]}


class _Embedder:
    def embed_passage(self, texts):
        return [self.embed_query_vec(t) for t in texts]

    def embed_query_vec(self, text):
        for k, v in _VEC.items():
            if k in text:
                return list(v)
        return [0.5, 0.5]


class _Source(inject.InjectionSource):
    def __init__(self) -> None:
        super().__init__(index=_Embedder())


def _turn(word: str, i: int) -> list[dict]:
    return [{"role": "user", "content": f"{word} task"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": f"c{i}", "type": "function", "function": {"name": "read", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": f"c{i}", "name": "read", "content": f"{word}-EVIDENCE " * 100},
            {"role": "assistant", "content": f"{word} done"}]


def _msgs(*words: str) -> list[dict]:
    m: list[dict] = [{"role": "system", "content": "sys"}]
    for i, w in enumerate(words):
        m += _turn(w, i)
    # open 单元：带一次工具往返（它的证据永不许裁）
    m += [{"role": "user", "content": "alpha again"},
          {"role": "assistant", "content": "", "tool_calls": [
              {"id": "cx", "type": "function", "function": {"name": "read", "arguments": "{}"}}]},
          {"role": "tool", "tool_call_id": "cx", "name": "read", "content": "OPEN-EVIDENCE " * 100}]
    return m


def _keys(msgs):
    return [u.unit_key for u in build_task_units(msgs)]


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    registry().clear()
    ar.cache().clear()
    monkeypatch.setenv("BLADEX_ASSEMBLY_AGENTS", "claude-code")
    monkeypatch.setenv("BLADEX_ASSEMBLY_MIN_UNITS", "1")
    monkeypatch.delenv("BLADEX_ASSEMBLY_REVIEW_MODEL", raising=False)
    monkeypatch.delenv("BLADEX_ASSEMBLY_REVIEW_EMBED", raising=False)
    yield
    registry().clear()
    ar.cache().clear()


# ── ① ② ③ 裁法 ─────────────────────────────────────────────────────────


def test_trim_replaces_only_tool_content_of_trimmed_closed_units():
    msgs = _msgs("alpha", "beta")
    k_alpha, k_beta, k_open = _keys(msgs)
    out, plan, n = ContextAssembler(AssemblyConfig()).trim(
        msgs, {k_alpha: ("kept", "embed"), k_beta: ("trimmed", "embed"), k_open: ("trimmed", "embed")})
    assert n == 1 and len(out) == len(msgs)
    beta_tool = 7
    assert out[beta_tool]["content"].startswith(f"{EVIDENCE_TRIMMED_OPEN} tool=read chars=")
    assert f"unit={k_beta}>" in out[beta_tool]["content"]
    for i, (a, b) in enumerate(zip(msgs, out, strict=True)):
        if i != beta_tool:
            assert a == b, f"只许动被裁单元的 tool 结果，消息 {i} 变了"
    assert out[-1]["content"].startswith("OPEN-EVIDENCE"), "open 单元永不裁（§2.6）"
    assert plan[3] == {"tier": "kept", "unit": k_alpha, "source": "embed"}
    assert plan[beta_tool] == {"tier": "trimmed", "unit": k_beta, "source": "embed"}
    assert len(msgs) - 1 not in plan, "open 单元不进 degrade_plan"


def test_trim_keeps_tool_call_pairing():
    """反向变异 5 的判别用例：裁完后每个 tool_calls.id 仍有一条同 id 的 tool 消息，反之亦然。"""
    msgs = _msgs("alpha", "beta")
    ks = _keys(msgs)
    out, _plan, n = ContextAssembler(AssemblyConfig()).trim(msgs, {k: ("trimmed", "model") for k in ks})
    assert n == 2
    call_ids = [tc["id"] for m in out if m.get("role") == "assistant" for tc in (m.get("tool_calls") or [])]
    tool_ids = [m.get("tool_call_id") for m in out if m.get("role") == "tool"]
    assert call_ids == tool_ids == ["c0", "c1", "cx"]
    assert [m["role"] for m in out] == [m["role"] for m in msgs]


def test_marker_is_pure_and_idempotent():
    msgs = _msgs("beta")
    k = _keys(msgs)[0]
    asm = ContextAssembler(AssemblyConfig())
    a, _, _ = asm.trim(msgs, {k: ("trimmed", "embed")})
    b, _, _ = asm.trim(msgs, {k: ("trimmed", "embed")})
    assert a == b, "同输入同输出（热轮重放前缀逐字稳定的前提）"
    c, _, n2 = asm.trim(a, {k: ("trimmed", "embed")})
    assert c == a and n2 == 0, "已裁的不再二次处理"
    assert evidence_marker(msgs[3], k) == a[3]["content"]


# ── ④ 复原 ────────────────────────────────────────────────────────────────


def test_replay_trim_reproduces_assembly_from_hub_original():
    msgs = _msgs("alpha", "beta")
    k_alpha, k_beta, _ = _keys(msgs)
    out, plan, _ = ContextAssembler(AssemblyConfig()).trim(
        msgs, {k_alpha: ("kept", "embed"), k_beta: ("trimmed", "model")})
    rec = ReconstructionRecord.model_validate({"layer": "A", "degrade_plan": plan})
    stored = rec.model_dump()["degrade_plan"]            # 走一遍 schema（Hub 存的就是这个形态）
    assert replay_trim(msgs, stored) == out


def test_replay_skips_historical_string_tiers():
    """历史 Hub 行（L1 `excerpt` / B 层 `minimal`）是裸 str：读得回，但摘录正文没落 Hub ⇒ 不可复原、原样跳过。"""
    msgs = _msgs("beta")
    rec = ReconstructionRecord.model_validate({"layer": "AB", "degrade_plan": {3: "minimal"}})
    assert rec.degrade_plan == {3: "minimal"}
    assert replay_trim(msgs, rec.degrade_plan) == msgs


# ── ⑤ ⑥ ⑧ 端到端（异步入口）──────────────────────────────────────────────


def _inject(msgs, *, session="s1", assembler=None):
    return asyncio.run(inject.do_inject_async(
        [dict(m) for m in msgs], "q", ProxyConfig(), _Source(),
        assembler=assembler or ContextAssembler(AssemblyConfig()),
        session_id=session, agent_id="claude-code"))


def test_cold_round_trims_by_verdict_and_records_reconstruction():
    msgs = _msgs("alpha", "beta")
    with capture_logs() as logs:
        out, _txt, _ms, _facts, cap = _inject(msgs)
    done = [e for e in logs if e.get("event") == "assembly_done"][-1]
    assert done["triggered"] == 1 and done["reviewed"] == 2 and done["trimmed"] == 1
    assert cap["triggered"] is True and cap["evidence_degraded"] == 1
    recon = ReconstructionRecord.model_validate(cap["reconstruction"])
    by_unit = {v.unit_key: v for v in recon.verdicts}
    k_alpha, k_beta, _ = _keys(msgs)
    assert by_unit[k_alpha].verdict == "kept" and by_unit[k_beta].verdict == "trimmed"
    assert recon.review["mode"] == "embed" and recon.review["trigger"]["triggered"] is True
    beta_tool = 7
    assert any(isinstance(m.get("content"), str) and m["content"].startswith(EVIDENCE_TRIMMED_OPEN)
               and f"unit={k_beta}" in m["content"] for m in out)
    # 复原：Hub 原文（剥注入）+ degrade_plan ⇒ 装配结果逐字相同（再注入硬规则 = 模型看到的全文）
    replayed = replay_trim(inject.strip_previous_injection(msgs), recon.model_dump()["degrade_plan"])
    assert replayed[beta_tool] == [m for m in out if m.get("role") != "system" or "bladex-memory" not in
                                   str(m.get("content"))][beta_tool]


def test_warm_round_replays_frozen_plan_without_new_trims():
    msgs = _msgs("alpha", "beta")
    cold_out, *_ = _inject(msgs)
    registry().note_cache_read("s1", 90_000)                     # 上游这轮命中了 ⇒ 下一轮热
    with capture_logs() as logs:
        warm_out, _t, _m, _f, cap = _inject(msgs)
    done = [e for e in logs if e.get("event") == "assembly_done"][-1]
    assert done["triggered"] == 0 and done["reviewed"] == 0 and done["trimmed"] == 1
    assert warm_out == cold_out, "热轮原样重放冷轮判定 ⇒ 送上游的正文逐字相同"
    assert "review" not in cap and all(v["source"] == "frozen" for v in cap["degrade_plan"].values())


def test_both_levels_off_is_observe_only_verbatim(monkeypatch):
    """判据⑤：两级都关 ⇒ triggered=1 reviewed=0 且正文逐字不变。反向变异 3：都关仍裁 ⇒ 这里红。"""
    monkeypatch.setenv("BLADEX_ASSEMBLY_REVIEW_EMBED", "0")
    msgs = _msgs("alpha", "beta")
    baseline, *_ = asyncio.run(inject.do_inject_async(
        [dict(m) for m in msgs], "q", ProxyConfig(), _Source(), assembler=None,
        session_id="s1", agent_id="claude-code"))
    with capture_logs() as logs:
        out, _t, _m, _f, cap = _inject(msgs)
    done = [e for e in logs if e.get("event") == "assembly_done"][-1]
    assert done["triggered"] == 1 and done["reviewed"] == 0 and done["trimmed"] == 0
    assert done["review_mode"] == "observe"
    assert out == baseline, "只观测：送上游的正文与装配关闭时逐字相同"
    assert cap["triggered"] is False


def test_hot_path_never_calls_review(monkeypatch):
    """热轮不进复核路径（200 ms 热路径预算不动）。"""
    called: list[int] = []

    async def _boom(*a, **k):
        called.append(1)
        raise AssertionError("热轮不该进复核")

    monkeypatch.setattr(inject, "_review_cold_round", _boom)
    registry().note_cache_read("s-hot", 5_000)
    _inject(_msgs("alpha", "beta"), session="s-hot")
    assert called == []


# ── ⑦ 信息守恒 ────────────────────────────────────────────────────────────


def test_information_conservation_original_stays_for_hub():
    """装配只改送上游的那份；调用方拿去入 Hub 的原文（`messages` 入参）一个字都不动。"""
    msgs = _msgs("alpha", "beta")
    original = [dict(m) for m in msgs]
    snapshot = [dict(m) for m in original]
    out, *_ = asyncio.run(inject.do_inject_async(
        original, "q", ProxyConfig(), _Source(), assembler=ContextAssembler(AssemblyConfig()),
        session_id="s-cons", agent_id="claude-code"))
    assert original == snapshot, "入参原文被改了 ⇒ Hub 存到的就不是 agent 原文"
    assert any("beta-EVIDENCE" in str(m.get("content")) for m in original)
    assert not any("beta-EVIDENCE" in str(m.get("content")) for m in out)
