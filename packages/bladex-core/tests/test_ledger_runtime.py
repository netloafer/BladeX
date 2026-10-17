"""V-L2–L5 验收剧本：切换（Y1c 起：举证、无防抖）/ 更新通道端到端（含 Goal 双门）/ 陈旧标记 / Matter 锚点。"""

from __future__ import annotations

import pytest
from bladex_core.ledger import GoalWriteViolation, LedgerError, new_ledger
from bladex_core.ledger_runtime import (
    ActivationTable,
    AnchorDecision,
    anchor_decision,
    apply_tool_update,
    bind_matter,
    staleness_marker,
    user_turn_index,
)


def _ledger(**kw):
    return new_ledger(ledger_id="ldg-t", title="测试任务", **kw)


# ── L2 · 激活与切换（批 Y · Y1c：轮数防抖已删，MQ-L91）──────────────────────

class TestActivation:
    def test_first_bind(self):
        t = ActivationTable()
        v = t.request_switch("s1", "ldg-a", agent_id="hermes", turn_index=1)
        assert v.allowed and t.active("s1") == "ldg-a"
        assert v.event_payload["to_ledger_id"] == "ldg-a"

    def test_rapid_reswitch_lands(self):
        """Y1c 判据 3 的纯函数面：同 session 隔 1 轮、同一轮内再切，全部落定（原防抖在这里拒）。"""
        t = ActivationTable()
        t.request_switch("s1", "ldg-a", agent_id="h", turn_index=1, session_id="sess")
        for turn, lid in ((2, "ldg-b"), (2, "ldg-a"), (2, "ldg-b")):
            v = t.request_switch("s1", lid, agent_id="h", turn_index=turn, session_id="sess")
            assert v.allowed and v.reason == "" and t.active("s1") == lid

    def test_only_refusal_is_noop(self):
        """反向变异（判据 6）：任何轮数/豁免判据加回来 ⇒ 这张网格里必有一格被拒。"""
        for sess_a, sess_b in (("s", "s"), ("s", "t"), ("", "s"), ("s", "")):
            for ta, tb in ((5, 5), (5, 6), (5, 7), (18, 2), (0, 0)):
                t = ActivationTable()
                t.request_switch("sc", "ldg-a", agent_id="pi", turn_index=ta, session_id=sess_a)
                v = t.request_switch("sc", "ldg-b", agent_id="pi", turn_index=tb, session_id=sess_b)
                assert v.allowed, (sess_a, sess_b, ta, tb, v.reason)

    def test_no_debounce_surface_left(self):
        """删干净（判据 11 的代码面）：构造参数、豁免参数、基点读数都不存在了。"""
        import inspect

        import bladex_core.ledger as ledger_mod
        sig = inspect.signature(ActivationTable.request_switch)
        assert "newly_created" not in sig.parameters and "user_requested" not in sig.parameters
        assert "switch_turns" not in inspect.signature(ActivationTable.restore).parameters
        assert "debounce_turns" not in inspect.signature(ActivationTable.__init__).parameters
        for gone in ("last_switch_turn", "last_switch_turn_session", "debounce_turns"):
            assert not hasattr(ActivationTable, gone), gone
        assert not hasattr(ledger_mod, "binding_switch_turns")

    def test_observation_fields_in_turn_and_back_to_prev(self):
        """「一轮里 A→B→A」放行但看得见：本轮第几次 / 距上次几轮 / 是否切回上一本。"""
        t = ActivationTable()
        v = t.request_switch("s1", "ldg-a", agent_id="h", turn_index=3, session_id="sess")
        assert (v.switch_in_turn, v.delta_turns, v.back_to_prev) == (1, None, False)
        v = t.request_switch("s1", "ldg-b", agent_id="h", turn_index=3, session_id="sess")
        assert (v.switch_in_turn, v.delta_turns, v.back_to_prev) == (2, 0, False)
        v = t.request_switch("s1", "ldg-a", agent_id="h", turn_index=3, session_id="sess")
        assert (v.switch_in_turn, v.delta_turns, v.back_to_prev) == (3, 0, True)
        v = t.request_switch("s1", "ldg-c", agent_id="h", turn_index=5, session_id="sess")
        assert (v.switch_in_turn, v.delta_turns, v.back_to_prev) == (1, 2, False)
        # 跨 session：不相减（不同尺子），轮内计数重开
        v = t.request_switch("s1", "ldg-a", agent_id="h", turn_index=1, session_id="other")
        assert (v.switch_in_turn, v.delta_turns) == (1, None)

    def test_observation_trace_not_restored_from_hub(self):
        """判据 9：重启（restore）后没有任何轮位基点——首次切换的 Δ轮次为空，不做跨口径相减。"""
        t = ActivationTable()
        t.restore({"s1": "ldg-a"}, sessions={"s1": "sess"})
        assert t.active("s1") == "ldg-a" and t.last_session("s1") == "sess"
        v = t.request_switch("s1", "ldg-b", agent_id="pi", turn_index=11, session_id="sess")
        assert v.allowed and v.delta_turns is None and v.switch_in_turn == 1

    def test_payload_still_carries_turn_index(self):
        """Hub 原始数据不因防抖删除而少一格（刚性原则 13 无损）。"""
        t = ActivationTable()
        v = t.request_switch("s1", "ldg-a", agent_id="cc", turn_index=8, session_id="sess")
        assert v.event_payload["turn_index"] == 8 and v.event_payload["session_id"] == "sess"

    def test_user_turn_index_counts_user_messages_only(self):
        msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
                {"role": "assistant", "content": "a"}, {"role": "tool", "content": "t"},
                {"role": "tool", "content": "t"}, {"role": "assistant", "content": "a"},
                {"role": "user", "content": "u"}]
        assert user_turn_index(msgs) == 2
        # 纯工具追加不动这个数（工具循环里每个调用一次请求，修前每次 +1 轮）
        more = msgs + [{"role": "assistant", "content": None, "tool_calls": []},
                       {"role": "tool", "content": "x"}] * 5
        assert user_turn_index(more) == 2
        assert user_turn_index([]) == 0 and user_turn_index(None) == 0

        class _M:
            def __init__(self, role): self.role = role
        assert user_turn_index([_M("user"), _M("assistant"), _M("user")]) == 2

    def test_same_ledger_is_noop_no_event(self):
        t = ActivationTable()
        t.request_switch("s1", "ldg-a", agent_id="h", turn_index=1)
        v = t.request_switch("s1", "ldg-a", agent_id="h", turn_index=99)
        assert not v.allowed and v.reason == "noop" and v.event_payload == {}

    def test_restore_projection(self):
        t = ActivationTable()
        t.restore({"s9": "ldg-z"})
        assert t.active("s9") == "ldg-z"

    def test_sessions_independent(self):
        t = ActivationTable()
        t.request_switch("s1", "ldg-a", agent_id="h", turn_index=1)
        v = t.request_switch("s2", "ldg-b", agent_id="h", turn_index=1)
        assert v.allowed


class TestSwitchEvidence:
    """`verify_switch_evidence` 纯函数（Y1c 判据 4 的内核）。"""

    def test_none_when_not_given(self):
        from bladex_core.ledger_runtime import verify_switch_evidence
        assert verify_switch_evidence("", "任何话") is None
        assert verify_switch_evidence("   ", "任何话") is None

    def test_verbatim_fragment_ok_and_fabricated_false(self):
        from bladex_core.ledger_runtime import verify_switch_evidence
        user = "小黑，排查一下最近Hermes飞书网关的连接情况，看看还有没有超时的问题存在。"
        assert verify_switch_evidence("排查一下最近Hermes飞书网关", user) is True
        assert verify_switch_evidence("“Hermes飞书网关的连接情况”", user) is True    # 引号不计
        assert verify_switch_evidence("hermes飞书网关", user) is True                # 大小写不计
        assert verify_switch_evidence("用户要求切换到飞书排查账本", user) is False     # 转述 = 编造
        assert verify_switch_evidence("排", user) is False                           # 太短不成证据

    def test_whitespace_folding(self):
        from bladex_core.ledger_runtime import verify_switch_evidence
        assert verify_switch_evidence("go back to\nthe  router  bug", "ok, go back to the router bug now") is True


# ── L3 · 更新通道 ───────────────────────────────────────────────────────────

class TestToolUpdate:
    def test_add_with_ref_is_tool_source(self):
        led, msg = apply_tool_update(_ledger(), {
            "section": "verified", "op": "add",
            "text": "pytest 全绿", "ref": "gate#1"})
        e = led.entries("verified")[0]
        assert (e.source, e.ref) == ("tool", "gate#1")
        assert "verified" in msg

    def test_add_without_ref_is_model_source(self):
        led, _ = apply_tool_update(_ledger(), {
            "section": "next", "op": "add", "text": "跑基准"})
        assert led.entries("next")[0].source == "model"

    def test_remove_roundtrip(self):
        led, _ = apply_tool_update(_ledger(), {
            "section": "open", "op": "add", "text": "悬而未决"})
        led2, msg = apply_tool_update(led, {"section": "open", "op": "remove",
                                            "index": 0})
        assert led2.entries("open") == [] and "悬而未决" in msg

    def test_goal_refused_second_gate(self):
        # schema 层 enum 已挡 goal；此处验证模型伪造参数绕 schema 时第二道门生效
        with pytest.raises((GoalWriteViolation, LedgerError)):
            apply_tool_update(_ledger(), {"section": "goal", "op": "add",
                                          "text": "偷改目标"})

    def test_bad_args(self):
        with pytest.raises(LedgerError):
            apply_tool_update(_ledger(), {"section": "next", "op": "add", "text": ""})
        with pytest.raises(LedgerError):
            apply_tool_update(_ledger(), {"section": "next", "op": "remove"})
        with pytest.raises(LedgerError):
            apply_tool_update(_ledger(), {"section": "next", "op": "replace"})


# ── L4 · 陈旧标记 ───────────────────────────────────────────────────────────

class TestStaleness:
    def test_fresh_no_marker(self):
        assert staleness_marker(turns_since_update=5, threshold_turns=20) == ""

    def test_exactly_at_threshold_fires(self):
        # 边界属于"陈旧"侧：`>=` 而非 `>`
        assert staleness_marker(turns_since_update=20, threshold_turns=20) != ""
        assert staleness_marker(turns_since_update=19, threshold_turns=20) == ""

    def test_stale_states_the_count_and_names_the_tool(self):
        m = staleness_marker(turns_since_update=30, threshold_turns=20)
        assert "30 turns" in m
        # 只陈述事实 + 指路，不替模型写内容（三红线：BladeX 不做作者）
        assert "bladex_ledger_update" in m

    def test_zero_threshold_disables(self):
        assert staleness_marker(turns_since_update=999, threshold_turns=0) == ""

    def test_signature_refuses_the_two_clock_form(self):
        """🔴 旧签名 `(ledger_turn, current_turn)` 邀请调用方去减 `turn_index`——

        而 `turn_index` 跨会话不单调（MQ-L13 同款陷阱），账本又天生跨会话。
        改成单一计数后喂错时钟在类型上就做不到；这条守住它别被"顺手加回来"。
        """
        import pytest
        with pytest.raises(TypeError):
            staleness_marker(ledger_turn=10, current_turn=40)  # type: ignore[call-arg]


# ── L5 · Matter 锚点 ────────────────────────────────────────────────────────

class TestAnchor:
    def test_no_ledger_falls_back(self):
        assert anchor_decision(None) == AnchorDecision()

    def test_unbound_ledger_creates_matter(self):
        d = anchor_decision(_ledger())
        assert d.create_matter and not d.matter_id

    def test_bound_ledger_anchors(self):
        led = bind_matter(_ledger(), "m-abc")
        d = anchor_decision(led)
        assert d.matter_id == "m-abc" and not d.warn_multi

    def test_second_matter_warns_never_rebinds(self):
        led = bind_matter(_ledger(), "m-abc")
        d = anchor_decision(led, proposed_matter_id="m-xyz")
        assert d.matter_id == "m-abc" and d.warn_multi   # 归属仍走锚，只告警

    def test_rebind_refused(self):
        led = bind_matter(_ledger(), "m-abc")
        with pytest.raises(LedgerError, match="manual review"):
            bind_matter(led, "m-xyz")

    def test_bind_same_is_idempotent(self):
        led = bind_matter(_ledger(), "m-abc")
        assert bind_matter(led, "m-abc").matter_id == "m-abc"
