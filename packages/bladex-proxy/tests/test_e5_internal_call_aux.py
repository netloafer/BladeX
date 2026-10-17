"""批 E · E5（拍板 (41)）：三条内部子调用漏判补规则——MQ-A84 / A85 / A86。

每条：Hub 原文（脱敏、截短）作病例夹具判 aux；同一 agent 的正常轮逐字不受影响；
Index 重建重判（`classify_turn_disposition`，与 A82 的 TestRebuildRejudge 同口径）能认出；
反向变异（删规则 ⇒ 对应用例红）在结算稿里记读数。
"""

from __future__ import annotations

import pytest
from bladex_core.task_goal import TURN_DROPPED, TURN_REAL
from bladex_proxy.identity import (
    CONTINUATION_AUX_RULES,
    DISTILL_ONLY_AUX_RULES,
    classify_auxiliary,
    classify_turn_disposition,
    is_cheap_tier_auxiliary,
    is_ledgerless_auxiliary,
    resolve_identity,
)
from bladex_proxy.models import ChatCompletionRequest

_CC_SYS = ("You are Claude Code, Anthropic's official CLI for Claude.\n\n"
           "You are an interactive agent that helps users with software engineering tasks.")
_CODEX_SYS = ("You are Codex, a coding agent. You and the user share the same workspace "
              "and collaborate to achieve the user's goals.")
_DSH_SYS = ("You are an AI agent powered by DeepSeek Harness.  You are a coding agent "
            "operating in the user's terminal.")
_DSH_TITLE_SYS = ("Create a concise title for an AI coding-assistant session from the supplied "
                  "human messages.\nReturn only the title on one line, **in plain text of natural "
                  "language**, with no quotes, prefix, explanation, Markdown, XML, or terminal "
                  "control codes.")

# ── Hub 原文（08-09 18:38 / 09-28 15:57 / 08-22 00:45，截短）──
_CC_SUGGESTION = ("[SUGGESTION MODE: Suggest what the user might naturally type next into Claude "
                  "Code.]\n\nFIRST: Look at the user's recent messages and original request.\n\n"
                  "Your job is to predict what THEY would type - not what you think they should do.")
_CODEX_CHECKPOINT = ("You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff summary "
                     "for another LLM that will resume the task.\n\nInclude:\n- Current progress and "
                     "key decisions made\n- Important context, constraints, or user preferences")
_DSH_TITLE_USER = ('Generate the session title from this JSON array of human messages:\n'
                   '[{"seq":7,"text":"只回复两个字：收到"}]')

_TOOL = {"type": "function", "function": {"name": "Bash"}}


def _cc_suggestion() -> list[dict]:
    return [{"role": "system", "content": _CC_SYS},
            {"role": "user", "content": "帮我把这个测试修绿"},
            {"role": "assistant", "content": "改好了。"},
            {"role": "user", "content": [{"type": "text", "text": _CC_SUGGESTION}]}]


def _codex_checkpoint() -> list[dict]:
    return [{"role": "system", "content": _CODEX_SYS},
            {"role": "developer", "content": "<skills_instructions>…</skills_instructions>"},
            {"role": "user", "content": "## 任务 1 · MQ-I33：全量重建被守卫中止时，不许报\"完成\""},
            {"role": "user", "content": _CODEX_CHECKPOINT}]


def _dsh_title() -> list[dict]:
    return [{"role": "system", "content": _DSH_TITLE_SYS},
            {"role": "user", "content": _DSH_TITLE_USER}]


# ── MQ-A92（2026-10-15）：Claude Code 活动摘要子调用 ──────────────────────────
# Hub 病例：local/claude-code/ce10048d…/1792026421690-0。
# 真实请求里它是 33 条消息的尾部：带完整 CC system prompt、有工具往返；
# 末条 user 是活动摘要信封。初判 auxiliary=False，ledger_face=True（病例全文见
# docs/planning/report-batch-HO-20261015.md §3.3；此处截短脱敏，保留判据前缀与形态）。
_CC_ACTIVITY_SUMMARY = (
    "Describe your most recent action in 3-5 words using present tense (-ing). "
    "Name the file or function, not the branch. Do not use tools.\n\n"
    'Previous: "Reading memory_index.py error strings." — say something NEW.\n\n'
    'Good: "Reading runAgent.ts"\n'
    'Good: "Fixing null check in validate.ts"\n'
    'Good: "Running auth module tests"\n'
    'Bad (too vague): "Investigating the issue"\n'
    'Bad (branch name): "Analyzed adam/background-summary branch diff"'
)


def _cc_activity_summary() -> list[dict]:
    return [
        {"role": "system", "content": _CC_SYS},
        {"role": "user", "content": "帮我把这个测试修绿"},
        {"role": "assistant", "content": "改好了。"},
        {"role": "user", "content": [{"type": "text", "text": _CC_ACTIVITY_SUMMARY}]},
    ]


# ── MQ-A93（2026-10-16）：dsh 上下文压缩引擎请求 ────────────────────────────
# Hub 病例：local/dsh/fp:4e24cc3b0a81/1787902902426-0（全库 8 条真轮之一）。
# 真实请求里它是 275 条消息的尾部：带完整 dsh system prompt、有工具往返；
# 末条 user 是压缩引擎模板（公共前缀 1802 字符，全部 8 条逐字相同）。此前
# auxiliary=False、ledger_face=True（活账 MQ-A93），整段压缩摘要被当事实提炼。
# 与 A84（codex CONTEXT CHECKPOINT COMPACTION）同一形态；此处截短脱敏，
# 保留判据前缀与结构（模板正文是 harness 生成的通用指令，不含私有信息）。
_DSH_COMPACTION = (
    "You are now acting as a compaction engine for this AI coding assistant. "
    "Condense the conversation ABOVE into a structured checkpoint that lets another "
    "model resume the work with no loss of essential context.\n\n"
    "Output EXACTLY the Markdown structure below: keep every section, in order. Use "
    'terse bullets, not prose paragraphs. Write "(none)" for an empty section — '
    "never drop a section.\n\n"
    "## Primary Request and Intent\n"
    "- [the user's original and evolving goals; quote verbatim where the exact wording matters]\n\n"
    "## Key Technical Concepts\n"
    "- [technologies, frameworks, patterns, and conventions in play]"
)


def _dsh_compaction() -> list[dict]:
    return [{"role": "system", "content": _DSH_SYS},
            {"role": "user", "content": "我用 Pi Agent 改进了一些代码，帮我评估一下还有什么缺陷"},
            {"role": "assistant", "content": "我先看一下仓库结构。"},
            {"role": "user", "content": _DSH_COMPACTION}]


_CASES = {
    "A85_cc_suggestion": (_cc_suggestion, "claude_code_suggestion_mode"),
    "A84_codex_checkpoint": (_codex_checkpoint, "codex_context_checkpoint"),
    # 规则名 = "system:" + 关键词前 40 字（`classify_auxiliary` 的定义，不手抄）
    "A86_dsh_title": (_dsh_title, "system:" + "create a concise title for an ai coding-assistant session"[:40]),
    "A92_cc_activity_summary": (_cc_activity_summary, "claude_code_activity_summary"),
    "A93_dsh_compaction": (_dsh_compaction, "dsh_compaction_engine"),
}

_NORMAL = {
    "claude-code": [{"role": "system", "content": _CC_SYS},
                    {"role": "user", "content": "看一下 git status，然后告诉我哪些文件改了"}],
    "codex": [{"role": "system", "content": _CODEX_SYS},
              {"role": "user", "content": "## 任务 2 · MQ-A78：陈旧 pidfile 让 bladex start 误报"}],
    "dsh": [{"role": "system", "content": _DSH_SYS},
            {"role": "user", "content": "这个分析有些简单，要深度全面分析一下。"}],
}


@pytest.mark.parametrize("name", sorted(_CASES))
def test_case_is_full_aux(name):
    build, rule = _CASES[name]
    is_aux, hit = classify_auxiliary(build())
    assert (is_aux, hit) == (True, rule)
    # 完整 aux：不在 DISTILL_ONLY（蒸馏跳过）、不在 CONTINUATION（不给账本面）、可走 aux 降档
    assert hit not in DISTILL_ONLY_AUX_RULES and hit not in CONTINUATION_AUX_RULES
    assert is_ledgerless_auxiliary(hit) is True and is_cheap_tier_auxiliary(hit) is True


@pytest.mark.parametrize("name", sorted(_CASES))
def test_case_identity_and_rebuild_rejudge(name):
    build, rule = _CASES[name]
    msgs = build()
    ident, _ = resolve_identity({"authorization": "Bearer k", "user-agent": "x"},
                                ChatCompletionRequest(model="m", messages=msgs, tools=[_TOOL]))
    assert ident.auxiliary is True and ident.aux_source == rule
    assert classify_turn_disposition(msgs) == (TURN_DROPPED, rule)


@pytest.mark.parametrize("agent", sorted(_NORMAL))
def test_normal_turn_of_each_agent_is_untouched(agent):
    msgs = _NORMAL[agent]
    assert classify_auxiliary(msgs) == (False, "")
    assert classify_turn_disposition(msgs) == (TURN_REAL, "")
    ident, _ = resolve_identity({"authorization": "Bearer k", "user-agent": "x"},
                                ChatCompletionRequest(model="m", messages=msgs, tools=[_TOOL]))
    assert ident.auxiliary is False and ident.aux_source == ""


def test_suggestion_prefix_only_counts_at_the_head():
    """前缀锚定在信封偏移内：用户在正文深处引用这句话不算内部调用。"""
    deep = "我想问一下，" + "x" * 3000 + " [SUGGESTION MODE: 这个词是我引用的]"
    assert classify_auxiliary([{"role": "system", "content": _CC_SYS},
                               {"role": "user", "content": deep}]) == (False, "")


def test_dsh_quoting_compaction_prefix_is_not_aux():
    """自引守护：我们的任务消息在正文里引用压缩模板前缀（off=38，MQ-A93 取证）
    不算内部调用——marker 取到 'Condense the conversation ABOVE' 正是为了越过
    'assistant.' 截止点：真实压缩轮有它（off=0），自引只有前缀到 'assistant.'。"""
    quoted = ("再接一件小活：dsh 自己在压缩上下文时发的那个请求"
              "（最后一条用户消息以 `You are now acting as a compaction engine "
              "for this AI coding assistant.` 开头）……")
    assert classify_auxiliary([{"role": "system", "content": _DSH_SYS},
                               {"role": "user", "content": quoted}]) == (False, "")
