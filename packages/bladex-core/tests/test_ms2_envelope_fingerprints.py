"""MS-2：信封指纹补齐（codex / cowork 脚手架 / hermes 转录壳）。

对应 `docs/planning/memory-core-dev-tasks-supplement-20260806.md` MS-2。

三组指纹的共同点：**它们长度达标、语义自洽**，所以既躲过长度过滤又能蒸出
像模像样的"事实/偏好"——库里那批 `## Goal` 假 preference 就是这么来的。
判据都保守（要么成对标签、要么长产品口吻短语、要么双条件结构匹配），
因为这一层的误伤代价是**真实用户意图被整条丢掉**。

X6 纪律：本文件的样本不出现在实现代码里；实现里写的是**指纹**（真实数据的形状），
测试里写的是**边界**（哪些不该命中）。
"""

from __future__ import annotations

import pytest

from bladex_core.envelope import (
    is_pure_envelope,
    prepare_distill_inputs,
    strip_envelopes,
)


# ── ① codex：<environment_context> ────────────────────────────────────────


def test_codex_environment_context_is_stripped():
    text = (
        "<environment_context>\n"
        "  <cwd>/Users/x/dev/BladeX</cwd>\n"
        "  <sandbox_mode>workspace-write</sandbox_mode>\n"
        "  <approval_policy>on-request</approval_policy>\n"
        "</environment_context>\n"
        "帮我把检索融合那段的秩合并逻辑理一下"
    )
    clean, kinds = strip_envelopes(text)
    assert "codex_environment_context" in kinds
    assert clean == "帮我把检索融合那段的秩合并逻辑理一下"
    assert "sandbox_mode" not in clean


def test_codex_environment_context_truncated_form():
    """未闭合形态（流被截断，只发了开标签）→ 从该处截断，不把半个块当意图。"""
    text = "先看下这个问题\n\n<environment_context>\n  <cwd>/Users/x/dev"
    clean, kinds = strip_envelopes(text)
    assert "unclosed_envelope" in kinds
    assert clean == "先看下这个问题"


def test_pure_environment_context_is_pure_envelope():
    assert is_pure_envelope("<environment_context><cwd>/a/b</cwd></environment_context>")


# ── ② cowork / codex 建议脚手架：整条剥除 ─────────────────────────────────


@pytest.mark.parametrize("text", [
    "# Overview\n\nGenerate 0 to 3 hyperpersonalized suggestions for what the user "
    "might want to do next, based on their recent activity.",
    "Get an understanding of the user's intent and goals, then propose next steps.",
    "The user stepped away and is coming back. Recap what happened while they were gone.",
])
def test_suggestion_scaffolding_is_stripped_entirely(text):
    """这类消息里没有任何一部分属于用户意图 → 整条剥除。"""
    clean, kinds = strip_envelopes(text)
    assert clean == ""
    assert kinds and kinds[0].endswith("_scaffold")
    assert is_pure_envelope(text)


def test_scaffolding_marker_anywhere_in_message_still_strips():
    """脚手架前面被套了别的壳时也要认出来（宿主 app 常在前面加标题行）。"""
    text = ("## Task\n\nGenerate 0 to 3 hyperpersonalized suggestions "
            "based on the conversation so far.\n\n## Output format\nJSON.")
    assert is_pure_envelope(text)


@pytest.mark.parametrize("text", [
    # 用户真的在谈"建议"功能 —— 不得误伤
    "帮我看看建议功能为什么只出了一条，预期应该有三条",
    "Generate a summary of the failing tests",          # 短、无产品口吻
    "The user guide needs a recap section",             # 含 recap 但不是脚手架
    "我想了解一下用户的真实意图应该怎么建模",
])
def test_scaffolding_does_not_hit_real_messages(text):
    clean, kinds = strip_envelopes(text)
    assert clean == text.strip()
    assert not any(k.endswith("_scaffold") for k in kinds)


# ── ③ hermes 转录壳：剥壳保留首段问句 ─────────────────────────────────────


def test_hermes_transcript_keeps_the_real_question():
    """🔴 泰山那条线的起点就是被这个形态整条 miss 的。"""
    text = (
        "User: 帮我核实一下泰山啤酒破产案里那个 3000 万共益债公告是不是真的\n\n"
        "Assistant: 好的，我查一下公开渠道。\n\n"
        "User: 谢谢"
    )
    clean, kinds = strip_envelopes(text)
    assert "hermes_transcript" in kinds
    assert clean == "帮我核实一下泰山啤酒破产案里那个 3000 万共益债公告是不是真的"
    assert "Assistant" not in clean


def test_hermes_transcript_needs_both_conditions():
    """双条件才动手：只以 `User:` 起头、没有 `\\n\\nAssistant:` → 原样保留。

    "User: 是我们数据库里的一张表" 是**合法的用户输入**，剥它就是丢意图。
    """
    text = "User: 是我们数据库里的一张表，帮我加个索引"
    clean, kinds = strip_envelopes(text)
    assert clean == text
    assert "hermes_transcript" not in kinds


def test_assistant_marker_without_user_prefix_is_untouched():
    text = "我贴一段日志：\n\nAssistant: 已完成\n\n这里为什么会这样？"
    clean, kinds = strip_envelopes(text)
    assert clean == text.strip()
    assert "hermes_transcript" not in kinds


def test_transcript_shell_goes_through_the_distill_window():
    """剥壳后的问句要能过 20–500 窗口进入蒸馏候选（整条 miss 的正面修复）。"""
    text = (
        "User: 帮我核实一下泰山啤酒破产案里那个 3000 万共益债公告是不是真的\n\n"
        "Assistant: " + "好的我这就去查。" * 200
    )
    texts, kinds = prepare_distill_inputs(text, 20, 500)
    assert "hermes_transcript" in kinds
    assert texts == ["帮我核实一下泰山啤酒破产案里那个 3000 万共益债公告是不是真的"]


# ── 🔴 零误伤总闸：聊天式 agent 的普通消息一条都不许动 ─────────────────────


@pytest.mark.parametrize("text", [
    "今天天气怎么样",
    "帮我把这个函数改成异步的，注意异常要往上抛",
    "美国和伊朗最近的局势有什么新进展吗",
    "泰山啤酒那个案子后来怎么样了",
    "Please refactor the retrieval module to use the new index.",
])
def test_ordinary_messages_are_untouched(text):
    clean, kinds = strip_envelopes(text)
    assert clean == text
    assert kinds == []
