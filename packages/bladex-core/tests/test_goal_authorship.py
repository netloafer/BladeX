"""Goal 执笔与改动规则（Jason 2026-08-26 拍板，取代 ADR-0031「取」不是「炼」）。

拍板三条：
  ① Goal 由 **agent 侧模型**在建账本时提炼（简洁、可验证的完成条件）；
  ② 过程中模型**不得自行修改**；用户发 prompt 调整了目标时可以改，
     但必须锚定到**本轮用户原话**；
  ③ dashboard 改 Goal 留到 0.2.0。

原立论证伪来自 live：jydesignhk 那 5 本账本的 Goal 全部以 agent 自己的
`Fully describe and explain everything about this image…` 开头，用户真正说的话
被压在 90 字符之后、5 个 Goal 前缀逐字相同。**原文 ≠ 用户意图。**

🔴 红线没有松动，只是把「谁执笔」与「谁能改」拆开——红线防的一直是
"模型中途把做不到的事改成做得到的事然后宣告完成"，那是中途篡改。
"""

from __future__ import annotations

import pytest
from bladex_core.ledger import (
    ACTOR_DASHBOARD,
    ACTOR_MODEL,
    ACTOR_USER,
    GoalWriteViolation,
    new_ledger,
    parse_ledger_md,
    render_ledger_md,
    revise_goal,
    update_goal,
)
from bladex_core.ledger_runtime import apply_tool_update

_USER = "帮我把路由回归修掉，另外顺便看一下账本机制"


def _led(**kw):
    return new_ledger(ledger_id="ldg-goal00000001", title="t",
                      goal="Route regression is fixed and covered by a test.",
                      goal_source="model", goal_verbatim=_USER,
                      created_at="2026-08-26T10:00:00Z", **kw)


class TestCreationAuthorship:
    def test_model_written_goal_is_marked_model(self):
        led = _led()
        assert led.goal_source == "model"
        assert led.goal_verbatim == _USER

    def test_verbatim_is_stored_but_never_rendered(self):
        """🔴 只存不注：`render_ledger_md` 的产物就是注入块。

        原话本来就在消息主体里，注入面再放一份是白花 token（Jason 拍板）。
        """
        md = render_ledger_md(_led())
        assert "Route regression is fixed" in md
        assert _USER not in md

    def test_md_roundtrip_needs_base_to_keep_verbatim(self):
        """MD 不携带 verbatim/revisions ⇒ 不给 base 会静默清空。"""
        led = _led()
        assert parse_ledger_md(render_ledger_md(led)).goal_verbatim == ""
        back = parse_ledger_md(render_ledger_md(led), base=led)
        assert back.goal_verbatim == _USER


class TestModelRevision:
    def test_quote_from_this_turn_is_accepted(self):
        led = _led()
        new = revise_goal(led, "Also audit the ledger mechanism end to end.",
                          quote="顺便看一下账本机制", user_text=_USER)
        assert new.goal.startswith("Also audit")
        assert new.goal_source == "model_revised"
        assert new.goal_revisions == 1

    def test_fabricated_quote_is_refused(self):
        """🔴 这道门的**唯一**硬保证：凭空编造的依据过不去。"""
        with pytest.raises(GoalWriteViolation, match="not found"):
            revise_goal(_led(), "Ship it without tests.",
                        quote="用户说不用写测试了", user_text=_USER)

    def test_missing_quote_is_refused(self):
        with pytest.raises(GoalWriteViolation, match="goal_change_quote"):
            revise_goal(_led(), "Something else", quote="", user_text=_USER)

    def test_empty_goal_is_refused(self):
        with pytest.raises(GoalWriteViolation):
            revise_goal(_led(), "  ", quote="路由回归", user_text=_USER)

    def test_punctuation_and_spacing_differences_are_tolerated(self):
        """模型复述时几乎必然改标点；卡死在标点上只会逼它猜格式。"""
        new = revise_goal(_led(), "New goal",
                          quote="顺便看一下，账本机制。", user_text=_USER)
        assert new.goal == "New goal"

    def test_weak_quote_still_passes_this_is_documented_not_prevented(self):
        """🔴 诚实记账：用户只说了"继续"、模型硬引用"继续"来改目标 —— **过得去**。

        这道门锚定责任、挡住伪造，但**证明不了"用户真的要求了"**。
        真正的保护是第二层：`goal_revisions` 会涨，改动留管理事件，改了能被数出来。
        这条测试存在的意义是**别让人以为已经防住了**。
        """
        new = revise_goal(_led(), "Whatever I want now",
                          quote="继续", user_text="继续")
        assert new.goal_revisions == 1   # ← 唯一的护栏：它涨了，所以看得见


class TestUserOwnership:
    def test_user_and_dashboard_may_write_directly(self):
        for actor in (ACTOR_USER, ACTOR_DASHBOARD):
            new = update_goal(_led(), "User's own goal", actor=actor)
            assert new.goal == "User's own goal" and new.goal_source == ACTOR_USER

    def test_model_may_not_use_the_user_path(self):
        """模型走 `revise_goal` 的锚定门，不能借 `update_goal` 绕过。"""
        with pytest.raises(GoalWriteViolation):
            update_goal(_led(), "sneaky", actor=ACTOR_MODEL)

    def test_revisions_count_accumulates_across_both_paths(self):
        led = revise_goal(_led(), "a", quote="路由回归", user_text=_USER)
        led = update_goal(led, "b", actor=ACTOR_USER)
        assert led.goal_revisions == 2


class TestToolUpdateRouting:
    def test_goal_arg_routes_to_the_anchored_path(self):
        new, msg = apply_tool_update(
            _led(), {"goal": "New goal", "goal_change_quote": "账本机制"},
            user_text=_USER)
        assert new.goal == "New goal" and "revision 1" in msg

    def test_goal_without_user_text_cannot_pass(self):
        """user_text 拿不到时**拒绝**，不是放行——fail-closed。"""
        with pytest.raises(GoalWriteViolation):
            apply_tool_update(_led(), {"goal": "x", "goal_change_quote": "账本机制"},
                              user_text="")

    def test_neither_section_nor_goal_is_a_clear_error(self):
        with pytest.raises(Exception, match="section"):
            apply_tool_update(_led(), {}, user_text=_USER)

    def test_entry_path_is_unchanged(self):
        new, msg = apply_tool_update(
            _led(), {"section": "verified", "op": "add", "text": "确认了一件事"},
            user_text=_USER)
        assert new.goal_revisions == 0
        assert "verified" in msg
