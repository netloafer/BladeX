"""V-L6 账本锚层验收剧本（core 侧纯函数；memory_index 接线由 live 复验签收）。"""

from __future__ import annotations

from datetime import UTC, datetime

from bladex_core.attribution import AttributionSource
from bladex_core.ledger import (
    EVENT_LEDGER_SWITCH,
    ledger_active_at,
    replay_ledger_events,
    switch_timeline,
)
from bladex_core.ledger_runtime import (
    activation_scope,
    agent_of_turn_key,
    anchor_decision,
    bind_matter,
    ledger_anchor_matter_id,
    session_of_turn_key,
    turn_key_ts_ms,
)
from bladex_core.ledger import new_ledger
from bladex_core.matter import Matter


def _switch(agent: str, lid: str, ts_ms: int, project: str = "") -> dict:
    return {"type": EVENT_LEDGER_SWITCH,
            "ts": datetime.fromtimestamp(ts_ms / 1000, UTC),
            "payload": {"agent_id": agent, "project_id": project,
                        "session_id": "fp:whatever", "to_ledger_id": lid}}


class TestAnchorMatterId:
    def test_deterministic_from_ledger_id(self):
        # 重建等价：同一账本 → 永远同一 Matter id
        a = ledger_anchor_matter_id("ldg-aaba6472df9d")
        assert a == ledger_anchor_matter_id("ldg-aaba6472df9d")
        assert a.startswith("m-") and len(a) == 14

    def test_distinct_ledgers_distinct_matters(self):
        assert ledger_anchor_matter_id("ldg-a") != ledger_anchor_matter_id("ldg-b")

    def test_empty_is_empty(self):
        assert ledger_anchor_matter_id("") == ""


class TestSessionOfTurnKey:
    def test_extracts_session_segment(self):
        assert session_of_turn_key(
            "local/hermes:default/fp:955be74ada9f/1787643914863-0"
        ) == "fp:955be74ada9f"

    def test_malformed_returns_empty(self):
        assert session_of_turn_key("") == ""
        assert session_of_turn_key("a/b") == ""


class TestAgentOfTurnKey:
    """MQ-L14 修法依据：turn key 的 agent 段与 LEDGER_SWITCH.agent_id 同形。"""

    def test_extracts_agent_segment(self):
        assert agent_of_turn_key(
            "24345848/hermes:default/fp:955be74ada9f/1787643914863-0"
        ) == "hermes:default"

    def test_malformed_returns_empty(self):
        assert agent_of_turn_key("") == ""
        assert agent_of_turn_key("a/b") == ""

    def test_ts_ms_from_entry_id(self):
        assert turn_key_ts_ms(
            "24345848/claude-code/fp:320d06072b76/1786342400933-0") == 1786342400933.0
        assert turn_key_ts_ms("a/b/c/not-a-number") == 0.0
        assert turn_key_ts_ms("") == 0.0

    def test_scope_key_matches_replay_binding_key(self):
        """🔴 阳性对照：查找键必须与 `replay_ledger_events` 的绑定键逐字相同。

        MQ-L14 就死在这两者不同构（一个 scope、一个 session），而两边各自
        都"看起来对"。这条测试把它们钉在一起。
        """
        lk = "24345848/codex/fp:3aba77b92604/1787651990350-0"
        _pool, bindings = replay_ledger_events(
            [_switch("codex", "ldg-aaba6472df9d", 1787651990000)])
        assert activation_scope(agent_of_turn_key(lk), "") in bindings


class TestSwitchTimeline:
    """MQ-L15：终态绑定会把先后做的几件事合成一个 Matter（误合并红线）。"""

    def _three(self) -> list[dict]:
        return [_switch("hermes:default", "ldg-A", 1000),
                _switch("hermes:default", "ldg-B", 2000),
                _switch("hermes:default", "ldg-C", 3000)]

    def test_point_in_time_lookup_per_segment(self):
        tl = switch_timeline(self._three())
        scope = activation_scope("hermes:default", "")
        assert ledger_active_at(tl, scope, 1500) == "ldg-A"
        assert ledger_active_at(tl, scope, 2500) == "ldg-B"
        assert ledger_active_at(tl, scope, 9999) == "ldg-C"

    def test_terminal_binding_would_have_merged_all_three(self):
        """反证：同一批事件走终态绑定，三段全归 ldg-C —— 那就是误合并。"""
        _pool, bindings = replay_ledger_events(self._three())
        assert bindings[activation_scope("hermes:default", "")] == "ldg-C"

    def test_before_first_switch_is_empty_not_first_ledger(self):
        # 建账本之前的历史不得被塞进锚定 Matter（反向误合并）
        tl = switch_timeline(self._three())
        assert ledger_active_at(tl, activation_scope("hermes:default", ""), 500) == ""

    def test_unknown_scope_empty(self):
        tl = switch_timeline(self._three())
        assert ledger_active_at(tl, activation_scope("codex", ""), 2500) == ""

    def test_events_out_of_order_are_sorted(self):
        tl = switch_timeline([_switch("a", "ldg-B", 2000), _switch("a", "ldg-A", 1000)])
        assert ledger_active_at(tl, activation_scope("a", ""), 1500) == "ldg-A"

    def test_scopes_are_independent(self):
        tl = switch_timeline([_switch("a", "ldg-A", 1000), _switch("b", "ldg-B", 1000)])
        assert ledger_active_at(tl, activation_scope("a", ""), 5000) == "ldg-A"
        assert ledger_active_at(tl, activation_scope("b", ""), 5000) == "ldg-B"

    def test_ignores_non_switch_events(self):
        assert switch_timeline([{"type": "ledger_create", "payload": {}}]) == {}

    def test_missing_ts_sorts_first(self):
        tl = switch_timeline([{"type": EVENT_LEDGER_SWITCH,
                               "payload": {"agent_id": "a", "to_ledger_id": "ldg-X"}}])
        assert ledger_active_at(tl, activation_scope("a", ""), 1) == "ldg-X"


class TestAnchorDecision:
    def test_unbound_ledger_creates(self):
        led = new_ledger(title="t")
        assert anchor_decision(led).create_matter is True

    def test_bound_ledger_returns_its_matter(self):
        led = bind_matter(new_ledger(title="t"), "m-aaa")
        assert anchor_decision(led).matter_id == "m-aaa"

    def test_second_matter_warns_and_keeps_ledger_binding(self):
        """🔴 ADR-0032 §4.5：告警人工复核，**绝不静默换绑**。"""
        led = bind_matter(new_ledger(title="t"), "m-aaa")
        dec = anchor_decision(led, proposed_matter_id="m-bbb")
        assert dec.warn_multi is True
        assert dec.matter_id == "m-aaa"

    def test_same_matter_no_warning(self):
        led = bind_matter(new_ledger(title="t"), "m-aaa")
        assert anchor_decision(led, proposed_matter_id="m-aaa").warn_multi is False

    def test_no_ledger_is_fallback(self):
        """🔴 阴性对照：无账本流量走 D1/DPL 兜底，锚层不表态。"""
        dec = anchor_decision(None)
        assert dec.matter_id == "" and dec.create_matter is False

    def test_rebinding_refused(self):
        led = bind_matter(new_ledger(title="t"), "m-aaa")
        try:
            bind_matter(led, "m-bbb")
        except Exception as e:
            assert "manual review" in str(e)
        else:
            raise AssertionError("re-binding must be refused")

    def test_bind_is_idempotent(self):
        led = bind_matter(new_ledger(title="t"), "m-aaa")
        assert bind_matter(led, "m-aaa").matter_id == "m-aaa"


class TestSchema:
    def test_matter_ledger_id_field_defaults_empty(self):
        m = Matter(matter_id="m-x", title="t")
        assert m.ledger_id == ""          # 历史数据零迁移

    def test_attribution_source_has_ledger_anchor(self):
        assert AttributionSource.LEDGER_ANCHOR.value == "ledger_anchor"
