"""V-L2–L5 验收剧本：切换防抖 / 更新通道端到端（含 Goal 双门）/ 陈旧标记 / Matter 锚点。"""

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


# ── L2 · 防抖 ───────────────────────────────────────────────────────────────

class TestActivation:
    def test_first_bind_not_debounced(self):
        t = ActivationTable(debounce_turns=3)
        v = t.request_switch("s1", "ldg-a", agent_id="hermes", turn_index=1)
        assert v.allowed and t.active("s1") == "ldg-a"
        assert v.event_payload["to_ledger_id"] == "ldg-a"

    def test_rapid_reswitch_refused(self):
        t = ActivationTable(debounce_turns=3)
        t.request_switch("s1", "ldg-a", agent_id="h", turn_index=1, session_id="sess")
        v = t.request_switch("s1", "ldg-b", agent_id="h", turn_index=2, session_id="sess")
        assert not v.allowed and v.reason == "debounce"
        assert t.active("s1") == "ldg-a"     # 抖动不生效

    def test_newly_created_exempt_from_debounce(self):
        """🔴 2026-08-25 live 回归：CC 4 秒内建两本（第二本是自我纠正），
        防抖若拦住新建的切换 ⇒ 会话绑在差的那本、好的成孤儿。"""
        t = ActivationTable(debounce_turns=3)
        t.request_switch("s1", "ldg-a", agent_id="cc", turn_index=1)
        v = t.request_switch("s1", "ldg-b", agent_id="cc", turn_index=1,
                             newly_created=True)
        assert v.allowed and t.active("s1") == "ldg-b"

    def test_existing_ledger_still_debounced(self):
        """豁免只给"新建"——在已有账本间横跳仍要防抖（这才是抖动本体）。"""
        t = ActivationTable(debounce_turns=3)
        t.request_switch("s1", "ldg-a", agent_id="cc", turn_index=1, session_id="sess")
        v = t.request_switch("s1", "ldg-b", agent_id="cc", turn_index=2, session_id="sess",
                             newly_created=False)
        assert not v.allowed and v.reason == "debounce"

    def test_user_requested_exempt_from_debounce(self):
        """用户显式指定的切换永远算数（与 Goal 只有用户能改同一条原则）。"""
        t = ActivationTable(debounce_turns=3)
        t.request_switch("s1", "ldg-a", agent_id="cc", turn_index=1)
        v = t.request_switch("s1", "ldg-b", agent_id="cc", turn_index=2,
                             user_requested=True)
        assert v.allowed and t.active("s1") == "ldg-b"

    def test_switch_after_debounce_window(self):
        t = ActivationTable(debounce_turns=3)
        t.request_switch("s1", "ldg-a", agent_id="h", turn_index=1)
        v = t.request_switch("s1", "ldg-b", agent_id="h", turn_index=5)
        assert v.allowed and v.event_payload["from_ledger_id"] == "ldg-a"

    def test_lower_turn_index_is_a_new_session_not_a_bounce(self):
        """🔴 MQ-L44（2026-09-05 live）：`turn_index` 会话内单调、新会话归零，
        而 `last_switch_turn` 按 `(agent, project)` 跨会话持久 —— 两把尺子相减无意义。

        修前：`2 - 18 = -16 < 3` ⇒ 拦。live 代价（`proxy-20260904-210657.log` 13:06:09）：
        Pi 新会话读完目标账本、**正确地**调了 switch 被拒，此后新会话在攒到约
        `(last+3)×2` 条消息前完全不能切账本。

        修前的代码注释还把这件事论证成"方向安全"（"差值为负 ⇒ 判定更宽松"）——
        判据是"小于就拦"，负值必然被拦，那段推理写反了（刚性原则 12 升级形态）。
        """
        t = ActivationTable(debounce_turns=3)
        t.request_switch("s1", "ldg-a", agent_id="pi", turn_index=18)
        v = t.request_switch("s1", "ldg-b", agent_id="pi", turn_index=2)
        assert v.allowed, "turn_index 低于 last_switch_turn = 跨会话，不是抖动"
        assert t.active("s1") == "ldg-b"

    def test_same_session_bounce_still_refused_after_l44(self):
        """判别力对照：L44 只放行**负差值**（跨会话），同会话内的真抖动照拦。
        少了这条，把 `turn_index >= last` 写成恒真也能让上一条绿。"""
        t = ActivationTable(debounce_turns=3)
        t.request_switch("s1", "ldg-a", agent_id="pi", turn_index=18, session_id="sess")
        v = t.request_switch("s1", "ldg-b", agent_id="pi", turn_index=19, session_id="sess")
        assert not v.allowed and v.reason == "debounce"
        assert t.active("s1") == "ldg-a"

    def test_equal_turn_index_still_debounced(self):
        """边界：差值 0（同一轮内连切两本）仍是抖动本体，不许被 `>=` 放过。"""
        t = ActivationTable(debounce_turns=3)
        t.request_switch("s1", "ldg-a", agent_id="pi", turn_index=7, session_id="sess")
        v = t.request_switch("s1", "ldg-b", agent_id="pi", turn_index=7, session_id="sess")
        assert not v.allowed and v.reason == "debounce"

    # ── E0.2（MQ-L44 根治 + MQ-L48）：参照系 = (session_id, 用户轮) ─────────

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

    def test_same_session_one_user_turn_apart_refused_three_allowed(self):
        t = ActivationTable(debounce_turns=3)
        t.request_switch("s1", "ldg-a", agent_id="cc", turn_index=5, session_id="sess")
        v = t.request_switch("s1", "ldg-b", agent_id="cc", turn_index=6, session_id="sess")
        assert not v.allowed and v.reason == "debounce"
        v = t.request_switch("s1", "ldg-b", agent_id="cc", turn_index=8, session_id="sess")
        assert v.allowed and t.active("s1") == "ldg-b"
        assert v.event_payload["turn_index"] == 8

    def test_cross_session_any_delta_allowed(self):
        """不同 session 的两个 user_turn 不是同一把尺子量的：差为正/零/负都不判。"""
        for other in (6, 5, 2, 40):
            t = ActivationTable(debounce_turns=3)
            t.request_switch("s1", "ldg-a", agent_id="pi", turn_index=5, session_id="A")
            v = t.request_switch("s1", "ldg-b", agent_id="pi", turn_index=other, session_id="B")
            assert v.allowed, f"跨 session Δ={other - 5} 被判成抖动"

    def test_missing_session_never_debounces(self):
        """任一方缺 session ⇒ 没有参照系可比 ⇒ 不判（宽松方向只漏不误）。"""
        t = ActivationTable(debounce_turns=3)
        t.request_switch("s1", "ldg-a", agent_id="pi", turn_index=5, session_id="A")
        assert t.request_switch("s1", "ldg-b", agent_id="pi", turn_index=6).allowed
        t2 = ActivationTable(debounce_turns=3)
        t2.request_switch("s1", "ldg-a", agent_id="pi", turn_index=5)
        assert t2.request_switch("s1", "ldg-b", agent_id="pi", turn_index=6, session_id="A").allowed

    def test_l44_negative_delta_implied_by_same_session_rule(self):
        """L44 最小修（负差不判）被新判据蕴含：同 session 压缩归零 ⇒ Δ<0 ⇒ 不判。"""
        t = ActivationTable(debounce_turns=3)
        t.request_switch("s1", "ldg-a", agent_id="pi", turn_index=18, session_id="sess")
        assert t.request_switch("s1", "ldg-b", agent_id="pi", turn_index=2, session_id="sess").allowed

    def test_restore_switch_turns_reinstates_debounce_baseline(self):
        """重启恢复 (session, turn)：同 session 隔 1 轮被拒；无字段 ⇒ 放行。"""
        t = ActivationTable(debounce_turns=3)
        t.restore({"s1": "ldg-a"}, switch_turns={"s1": ("sess", 10)})
        assert t.last_switch_turn("s1") == 10 and t.last_switch_turn_session("s1") == "sess"
        v = t.request_switch("s1", "ldg-b", agent_id="pi", turn_index=11, session_id="sess")
        assert not v.allowed and v.reason == "debounce"
        assert t.request_switch("s1", "ldg-b", agent_id="pi", turn_index=11, session_id="new").allowed
        t2 = ActivationTable(debounce_turns=3)
        t2.restore({"s1": "ldg-a"})                       # 历史事件无 turn_index
        assert t2.last_switch_turn("s1") is None
        assert t2.request_switch("s1", "ldg-b", agent_id="pi", turn_index=1, session_id="sess").allowed

    def test_binding_switch_turns_projection(self):
        from bladex_core.ledger import binding_switch_turns
        from bladex_core.ledger_runtime import activation_scope
        sc = activation_scope("pi", "")
        events = [
            {"type": "ledger_switch", "payload": {"agent_id": "pi", "session_id": "old",
                                                 "to_ledger_id": "ldg-a"}},            # 历史：无字段
        ]
        assert binding_switch_turns(events) == {}
        events.append({"type": "ledger_switch", "payload": {
            "agent_id": "pi", "session_id": "sess", "to_ledger_id": "ldg-b", "turn_index": 7}})
        assert binding_switch_turns(events) == {sc: ("sess", 7)}
        # last-wins；切到空（墓碑改写）清基点
        events.append({"type": "ledger_switch", "payload": {
            "agent_id": "pi", "session_id": "sess", "to_ledger_id": "", "turn_index": 9}})
        assert binding_switch_turns(events) == {}

    def test_same_ledger_is_noop_no_event(self):
        t = ActivationTable(debounce_turns=3)
        t.request_switch("s1", "ldg-a", agent_id="h", turn_index=1)
        v = t.request_switch("s1", "ldg-a", agent_id="h", turn_index=99)
        assert not v.allowed and v.reason == "noop" and v.event_payload == {}

    def test_restore_projection(self):
        t = ActivationTable()
        t.restore({"s9": "ldg-z"})
        assert t.active("s9") == "ldg-z"

    def test_sessions_independent(self):
        t = ActivationTable(debounce_turns=3)
        t.request_switch("s1", "ldg-a", agent_id="h", turn_index=1)
        v = t.request_switch("s2", "ldg-b", agent_id="h", turn_index=1)
        assert v.allowed


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
