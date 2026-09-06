"""MQ-L34 · 零工具 / 孤立子调用只挡账本族，记忆族照注 + `aux_reason` 子条件可读
（2026-09-02 Jason 拍板「仪器先行 + 修」）。

病灶：`server._apply_agency_surfaces` 把四个账本域子条件合成的 `ledgerless_aux`
原样传给 `augment_tools(auxiliary=…)` ⇒ `inject_tools` 两族一起不注。08-30 拍板是
「记忆族无条件注入」（注入面砍到只剩硬规则后，工具面是模型取记忆的唯一通道），
零工具的聊天类请求因此连 `bladex_memory_search` 都没有。
仪器缺口：`agency_toolface_decision reason=aux` 四个子条件一个字不分，
08-31 hermes:accept 18 轮读不出是哪一个。

判据取 A 的原文（`identity.AUX_REASON_*` / `LEDGER_ONLY_AUX_REASONS`），不取 live 样本
（engineering-conventions §3.2b 第 2 条）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bladex_proxy.identity import (
    AUX_REASON_ISOLATED,
    AUX_REASON_NO_TOOLS,
    AUX_REASON_SOURCE,
    AUX_REASON_SUBAGENT,
    LEDGER_ONLY_AUX_REASONS,
)
from bladex_proxy.toolface import FAMILY_MEMORY, tool_family

_AGENT = "hermes:accept"


def _on(monkeypatch, *names):
    for n in names:
        monkeypatch.setenv(f"BLADEX_MODULE_{n.upper()}", "1")


def _names(tools) -> list[str]:
    return [(t.get("function") or t).get("name", "") for t in (tools or [])]


def _families(tools) -> set[str]:
    return {tool_family(n) for n in _names(tools) if n.startswith("bladex_")}


@pytest.fixture
def runtime(monkeypatch):
    from bladex_proxy.agency import AgencyRuntime
    _on(monkeypatch, "toolface", "ledger")
    monkeypatch.setenv("BLADEX_LEDGER_TIERS", "strong,medium")
    return AgencyRuntime()


class TestContract:
    def test_ledger_only_set_is_exactly_the_two_structural_reasons(self):
        """契约表本体：哪些子条件只挡账本族。改这个集合 = 改拍板，得同步改台账。"""
        assert LEDGER_ONLY_AUX_REASONS == {AUX_REASON_NO_TOOLS, AUX_REASON_ISOLATED}
        assert AUX_REASON_SOURCE not in LEDGER_ONLY_AUX_REASONS
        assert AUX_REASON_SUBAGENT not in LEDGER_ONLY_AUX_REASONS


class TestAugmentToolsFamilySplit:
    def test_ledgerless_only_injects_memory_family(self, runtime):
        """零工具轮（ledgerless=True, auxiliary=False）：记忆族在、账本族不在。"""
        tools, injected = runtime.augment_tools(
            [], agent_id=_AGENT, auxiliary=False, ledgerless=True,
            aux_reason=AUX_REASON_NO_TOOLS, tier="strong")
        assert injected is True
        assert _families(tools) == {FAMILY_MEMORY}
        assert "bladex_memory_search" in _names(tools)
        assert not any(n.startswith("bladex_ledger_") for n in _names(tools))

    def test_auxiliary_still_blocks_both_families(self, runtime):
        """真·aux（aux_source / subagent）行为不变：两族都不注。"""
        tools, injected = runtime.augment_tools(
            [], agent_id=_AGENT, auxiliary=True, ledgerless=True,
            aux_reason=AUX_REASON_SOURCE, tier="strong")
        assert injected is False
        assert _families(tools) == set()

    def test_ledgerless_defaults_to_auxiliary_for_old_callers(self, runtime):
        """`ledgerless=None` ⇒ 沿用 `auxiliary`（既有调用点/测试零改动）。"""
        t1, i1 = runtime.augment_tools([], agent_id=_AGENT, auxiliary=True,
                                       tier="strong")
        assert i1 is False and _families(t1) == set()
        t2, i2 = runtime.augment_tools([], agent_id=_AGENT, auxiliary=False,
                                       tier="strong")
        assert i2 is True and "bladex_ledger_switch" in _names(t2)

    def test_ledgerless_overrides_session_owns_ledger(self, runtime, monkeypatch):
        """🔴 `ledgerless` 压过账本族的第二个放行口（本会话绑着账本）：零工具轮
        正是 MQ-L21 里建噪声账本的那类请求，绑着账本也不给它切换工具。"""
        monkeypatch.setattr(runtime, "session_owns_active_ledger",
                            lambda *a, **k: True)
        tools, injected = runtime.augment_tools(
            [], agent_id=_AGENT, auxiliary=False, ledgerless=True,
            aux_reason=AUX_REASON_ISOLATED, tier="weak", session_id="s1")
        assert injected is True
        assert _families(tools) == {FAMILY_MEMORY}
        assert runtime.last_ledger_face is False

    def test_last_ledger_face_false_when_memory_only(self, runtime):
        """`Turn.ledger_face` 语义（models.py）：False = 明确没给账本面。
        记忆族注了不等于账本面给了——不许把 memory-only 记成 True。"""
        runtime.augment_tools([], agent_id=_AGENT, auxiliary=False,
                              ledgerless=True, aux_reason=AUX_REASON_NO_TOOLS,
                              tier="strong")
        assert runtime.last_ledger_face is False
        runtime.augment_tools([], agent_id=_AGENT, auxiliary=False,
                              ledgerless=False, tier="strong")
        assert runtime.last_ledger_face is True


class TestDecisionLogCarriesSubReason:
    def _rows(self, monkeypatch, runtime, **kw):
        import structlog.testing
        with structlog.testing.capture_logs() as cap:
            runtime.augment_tools([], agent_id=_AGENT, tier="strong", **kw)
        return [e for e in cap if e.get("event") == "agency_toolface_decision"]

    def test_reason_carries_sub_condition_when_blocked(self, monkeypatch, runtime):
        rows = self._rows(monkeypatch, runtime, auxiliary=True, ledgerless=True,
                          aux_reason=AUX_REASON_SUBAGENT)
        assert rows[-1]["reason"] == f"aux:{AUX_REASON_SUBAGENT}"
        assert rows[-1]["aux_reason"] == AUX_REASON_SUBAGENT
        assert rows[-1]["ledgerless"] is True

    def test_aux_reason_logged_even_when_memory_injected(self, monkeypatch, runtime):
        """注了记忆族 `reason=""`，子条件只能从 `aux_reason` 读——这一列必须在。"""
        rows = self._rows(monkeypatch, runtime, auxiliary=False, ledgerless=True,
                          aux_reason=AUX_REASON_NO_TOOLS)
        assert rows[-1]["injected"] is True
        assert rows[-1]["reason"] == ""
        assert rows[-1]["aux_reason"] == AUX_REASON_NO_TOOLS
        assert rows[-1]["ledgerless"] is True
        assert rows[-1]["ledger_face"] is False

    def test_plain_aux_without_reason_keeps_legacy_label(self, monkeypatch, runtime):
        """旧调用形态（不传 aux_reason）仍是 `reason=aux`——历史 grep 不断。"""
        rows = self._rows(monkeypatch, runtime, auxiliary=True)
        assert rows[-1]["reason"] == "aux"
        assert rows[-1]["aux_reason"] == ""


class TestServerWiring:
    """接线守卫（按路径读源码，MQ-V9：inspect.getsource 会静默取错函数）。"""

    @staticmethod
    def _src() -> str:
        return (Path(__file__).resolve().parents[1]
                / "bladex_proxy" / "server.py").read_text(encoding="utf-8")

    def test_surfaces_pass_split_knobs(self):
        src = self._src()
        assert "auxiliary=memory_blocked, ledgerless=ledgerless_aux," in src
        assert "aux_reason=aux_reason" in src
        assert ("memory_blocked = ledgerless_aux and aux_reason not in "
                "LEDGER_ONLY_AUX_REASONS") in src

    def test_ledger_block_still_gated_by_full_ledgerless(self):
        """账本块注入（正文）照旧吃完整的 `ledgerless_aux`——本卡只动工具面的族拆分。"""
        src = self._src()
        assert "auxiliary=ledgerless_aux, with_instruction=tf_injected" in src

    def test_all_four_sub_conditions_named_in_order(self):
        """判定顺序 = 日志里子条件名的优先级：aux_source > subagent > no_tools > isolated。"""
        src = self._src()
        seq = [src.index(f"AUX_REASON_{k} if ")
               for k in ("SOURCE", "SUBAGENT", "NO_TOOLS", "ISOLATED")]
        assert seq == sorted(seq)
