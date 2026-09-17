"""信封指纹的**位置门**与账本面的**第三套 aux 语义**（MQ-A19 / MQ-L20）。

🔴 事故（2026-08-27 live）：Claude Code 每轮把 `<system-reminder>` + CLAUDE.md 全文
塞进 user 消息，而 CLAUDE.md 里写着 "`<transcript>` 安全检查子 agent 转录一项就占
439/507"——`<transcript>` 落在第 **33917** 字符，指纹是**裸子串**匹配 ⇒
113/114 轮判 aux ⇒ 工具面零注入、账本零建、蒸馏全跳。

**我们自己写的规则描述，把读这份文档的 agent 判成了内部调用。**

变量名叫 `prefix`、注释写"模板前缀"，实现却是 `in`——命名与实现分歧无人发现，
直到自指触发（刚性原则 12 的形状）。

隔离对照（归档全库，旧实现 vs 新实现同批数据）：
  claude-code   181 → 27 aux（-154，全是自指）
  hermes:default / codex / Pi / dsh   **零变化**
"""

from __future__ import annotations

import pytest
from bladex_proxy.identity import (
    _ENVELOPE_MAX_OFFSET,
    CONTINUATION_AUX_RULES,
    DISTILL_ONLY_AUX_RULES,
    classify_auxiliary,
    is_cheap_tier_auxiliary,
    is_ledgerless_auxiliary,
)


def _turn(user_text: str) -> list[dict]:
    return [{"role": "system", "content": "you are claude code"},
            {"role": "user", "content": user_text}]


class TestEnvelopeOffsetGate:
    def test_real_envelope_at_position_zero(self):
        """阳性对照：真信封开头就是标记。"""
        ok, rule = classify_auxiliary(_turn("<transcript>\n\n{\"user\":\"...\"}"))
        assert ok and rule == "claude_code_safety_transcript"

    def test_real_envelope_with_small_preamble(self):
        """实测形态：`[Request interrupted by user]` 之类的小前言，标记在 278 字符处。"""
        text = ("[Request interrupted by user]\n\n<local-command-caveat>Caveat: "
                + "x" * 200 + "</local-command-caveat>\n<command-name>/foo</command-name>")
        ok, rule = classify_auxiliary(_turn(text))
        assert ok and rule == "claude_code_slash_command"

    def test_marker_buried_deep_is_not_an_envelope(self):
        """🔴 自指陷阱：正文里**提到**信封标记，不代表这一轮是信封。"""
        text = ("请按任务卡执行开发。\n\n" + "背景说明。" * 4000
                + "\n实测 `<transcript>` 安全检查子 agent 转录一项就占 439/507。")
        assert text.index("<transcript>") > _ENVELOPE_MAX_OFFSET
        ok, rule = classify_auxiliary(_turn(text))
        assert not ok and rule == "", f"自指被判成信封: {rule}"

    def test_the_real_claude_md_shape(self):
        """把事故形态原样钉住：CLAUDE.md 全文 + 深处的标记。"""
        text = ("<system-reminder>\nAs you answer the user's questions, you can use "
                "the following context:\n# claudeMd\n" + "文档正文。" * 5000
                + "（`<transcript>` 安全检查子 agent 转录一项就占 439/507）")
        ok, _ = classify_auxiliary(_turn(text))
        assert not ok

    def test_earliest_match_wins_not_table_order(self):
        """实测病例：开头是 summarization prompt，深处才有 task-list 标记。

        旧实现按**表序**取，而 `task_list_preserved` 排在
        `summarization_checkpoint` 之前 ⇒ 贴错标签。
        """
        text = ("You are a summarization agent creating a context checkpoint. "
                + "对话内容。" * 100
                + "[Your active task list was preserved across context compression]")
        ok, rule = classify_auxiliary(_turn(text))
        assert ok and rule == "summarization_checkpoint", rule

    def test_threshold_is_a_calibrated_constant(self):
        """阈值是标定出来的，不是圆整数——改它要重跑分布（见 identity.py 注释）。

        实测真信封最深 278，自指最浅 >20000；2000 落在双峰之间。
        """
        assert 278 < _ENVELOPE_MAX_OFFSET < 20000

    def test_system_keyword_path_unaffected(self):
        """第二条道（system prompt 关键词）不受位置门影响。"""
        ok, rule = classify_auxiliary(
            [{"role": "system", "content": "You are naming a coding session"},
             {"role": "user", "content": "随便什么"}])
        assert ok and rule.startswith("system:")


class TestThreeAuxSemantics:
    """一个 `auxiliary` 旋钮，三个消费方，三套语义（MQ-L20）。"""

    def test_the_three_sets_are_deliberately_different(self):
        """🔴 若两套集合完全相同，说明有人把它们合并了——那正是事故的形状。"""
        assert DISTILL_ONLY_AUX_RULES != CONTINUATION_AUX_RULES
        assert CONTINUATION_AUX_RULES < DISTILL_ONLY_AUX_RULES  # 真子集

    @pytest.mark.parametrize("rule", sorted(CONTINUATION_AUX_RULES))
    def test_continuation_rules_keep_the_ledger_face(self, rule):
        """续写/重试/恢复：assistant 侧接着做同一件正事 ⇒ 该带账本。"""
        assert is_ledgerless_auxiliary(rule) is False

    @pytest.mark.parametrize("rule", [
        "async_delegation_batch", "web_content_summary", "background_fanout",
        "summarization_checkpoint", "hermes_memory_save",
        "claude_code_safety_transcript", "claude_code_task_notification",
    ])
    def test_standalone_internal_calls_lose_the_ledger_face(self, rule):
        """真·独立内部调用不给账本面——错给会开出噪声账本（jydesignhk 五卡）。"""
        assert is_ledgerless_auxiliary(rule) is True

    def test_not_auxiliary_is_never_blocked(self):
        assert is_ledgerless_auxiliary("") is False

    def test_ledger_and_routing_disagree_on_purpose(self):
        """同一条规则，路由与账本给出**不同**答案——这正是拆开的意义。

        `claude_code_safety_transcript`：路由不降档（DISTILL_ONLY），
        账本面要挡（真·安全检查子 agent，给了工具就会开噪声账本）。
        """
        r = "claude_code_safety_transcript"
        assert is_cheap_tier_auxiliary(r) is False
        assert is_ledgerless_auxiliary(r) is True
