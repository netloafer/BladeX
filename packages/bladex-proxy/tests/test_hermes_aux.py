"""U3：hermes 侧三类内部调用 auxiliary 指纹（ADR-0024 遗留卡）。

三类此前 classify_auxiliary 未命中，既污染归属又让蒸馏做无用功：
  - hermes_memory_save          "Review the conversation above and consider saving to memory…"
  - hermes_empty_response_retry "You just executed tool calls but returned an empty response…"
  - hermes_transcript_replay    "User: …\n\nAssistant: …" 转录形态（结构化匹配）

语义拆分：三条都归 DISTILL_ONLY —— 跳蒸馏（没记忆价值），但**不**降档路由
（避免 T3 式"用户配置模型被旁路"回归）。

🔴 零误伤：对 claude-code / codex / 真实用户消息命中 = 0。
"""

from __future__ import annotations

import pytest
from bladex_proxy.identity import (
    DISTILL_ONLY_AUX_RULES,
    classify_auxiliary,
    is_cheap_tier_auxiliary,
)


def _msgs(user: str, system: str = "You are a helpful assistant.") -> list[dict]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def test_hermes_memory_save_is_aux_distill_only() -> None:
    is_aux, rule = classify_auxiliary(_msgs(
        "Review the conversation above and consider saving to memory any durable facts."
    ))
    assert is_aux is True
    assert rule == "hermes_memory_save"
    assert rule in DISTILL_ONLY_AUX_RULES
    assert is_cheap_tier_auxiliary(rule) is False  # 跳蒸馏但不降档


def test_hermes_empty_response_retry_is_aux_distill_only() -> None:
    is_aux, rule = classify_auxiliary(_msgs(
        "You just executed tool calls but returned an empty response. Please try again."
    ))
    assert is_aux is True
    assert rule == "hermes_empty_response_retry"
    assert rule in DISTILL_ONLY_AUX_RULES
    assert is_cheap_tier_auxiliary(rule) is False


def test_hermes_transcript_replay_structural_match() -> None:
    is_aux, rule = classify_auxiliary(_msgs(
        "User: 帮我查下天气\n\nAssistant: 好的，今天晴。\n\nUser: 谢谢"
    ))
    assert is_aux is True
    assert rule == "hermes_transcript_replay"
    assert rule in DISTILL_ONLY_AUX_RULES
    assert is_cheap_tier_auxiliary(rule) is False


# ── 🔴 零误伤 ──


def test_normal_user_message_not_aux() -> None:
    is_aux, rule = classify_auxiliary(_msgs("帮我把这个函数改成异步的"))
    assert is_aux is False
    assert rule == ""


def test_user_message_starting_with_User_but_not_transcript_not_aux() -> None:
    """真实消息恰好以 'User:' 开头但无 '\\n\\nAssistant:' 结构 → 不误判。"""
    is_aux, rule = classify_auxiliary(_msgs("User: 是我们数据库里的一张表，帮我加索引"))
    assert is_aux is False


def test_claude_code_and_codex_traffic_not_hit_by_hermes_rules() -> None:
    """claude-code / codex 常见内容不命中 hermes 三条新指纹（命中的话是别的 claude 规则）。"""
    # claude-code 真实任务描述（长、含代码），不该命中 hermes_* 规则
    cc = classify_auxiliary(_msgs(
        "Please refactor the retrieval module. Here is the current code:\n"
        "def search(): ...\nMake it use the new index."
    ))
    assert cc[1] not in {"hermes_memory_save", "hermes_empty_response_retry", "hermes_transcript_replay"}
    # codex 风格
    cx = classify_auxiliary(_msgs("Fix the failing test in test_route.py"))
    assert cx[1] not in {"hermes_memory_save", "hermes_empty_response_retry", "hermes_transcript_replay"}


def test_hermes_rules_registered_in_distill_only_set() -> None:
    """三条 hermes 规则都在 DISTILL_ONLY 集合（制度性保证不降档）。"""
    for r in ("hermes_memory_save", "hermes_empty_response_retry", "hermes_transcript_replay"):
        assert r in DISTILL_ONLY_AUX_RULES


# ── MS-2 信封普查发现（2026-08-06，真实 Memory Hub 全量）──────────────────


@pytest.mark.parametrize("dash", ["—", "-", "–", "−"])
def test_context_compaction_matches_every_dash_variant(dash: str) -> None:
    """🔴 一个字符的差别让 811 轮从未命中。

    指纹表写的是 ASCII 连字符，真实流量发的是 em dash——`context_compaction_handoff`
    因此对生产流量**恒不命中**，那 811 轮全被当成真实用户轮（污染归属 + 蒸馏白烧）。
    指纹表看起来是对的、日志里也没有异常，只有把真实数据按形态聚类才看得见。
    修法是归一破折号而不是改那一行：agent 换个 Unicode 变体就复发。
    """
    is_aux, rule = classify_auxiliary(_msgs(
        f"[CONTEXT COMPACTION {dash} REFERENCE ONLY] Earlier turns were compacted."
    ))
    assert is_aux is True
    assert rule == "context_compaction_handoff"
    # 原生 aux（整轮内部通信）→ 仍享受路由降档，T9 收益不变
    assert is_cheap_tier_auxiliary(rule) is True


def test_truncated_response_continue_is_distill_only() -> None:
    """输出被截断后的续写提示（实测 364 次，此前无指纹）。

    user 那侧只有一句系统提示，但 assistant 续写的是**真实工作** ——
    所以归 DISTILL_ONLY：跳用户侧蒸馏、产出照收、不降路由档。
    """
    is_aux, rule = classify_auxiliary(_msgs(
        "[System: Your previous response was truncated by the output limit. "
        "Please continue from where you left off.]"
    ))
    assert is_aux is True
    assert rule == "truncated_response_continue"
    assert rule in DISTILL_ONLY_AUX_RULES
    assert is_cheap_tier_auxiliary(rule) is False


@pytest.mark.parametrize("text", [
    "这段文字里有个破折号 —— 但它不是压缩交接",
    "帮我看看 previous response 为什么被截断了",
    "小黑，帮我核实一下虎彩集团破产进入程序了吗？",
])
def test_new_fingerprints_do_not_hit_real_messages(text: str) -> None:
    """归一破折号之后不得放宽到误伤真实消息。"""
    is_aux, _rule = classify_auxiliary(_msgs(text))
    assert is_aux is False


# ── MQ-S12：三条脚手架模板（2026-08-18，实测在污染归属）──────────────────────


@pytest.mark.parametrize(("text", "expected_rule"), [
    (
        "Review the conversation above and update the skill library. Be ACTIVE — most "
        "sessions produce at least one skill update, even if small.",
        "hermes_skill_library_update",
    ),
    (
        "The following is the Codex agent history added since the last checkpoint:\n"
        "- ran tests\n- fixed lint",
        "codex_history_injection",
    ),
    (
        "The user stepped away and is coming back. Recap in a few sentences what was "
        "being worked on.",
        "session_resume_recap",
    ),
])
def test_mq_s12_scaffold_templates_are_aux_distill_only(text: str, expected_rule: str) -> None:
    """三条脚手架模板命中，且归 DISTILL_ONLY（跳蒸馏、不降路由档）。

    比"白跑蒸馏"严重的地方：这些轮 user 侧是脚手架、内容与任务无关，assistant 侧却在
    做真实工作 → 蒸出的 fact 带着上一件事的上下文落到当前卡上，形成跨话题错边
    （实证：`local-llm-inference-macos` ⇔ `bladex 正式 CLI 注册方案`）。
    """
    is_aux, rule = classify_auxiliary(_msgs(text))
    assert is_aux is True
    assert rule == expected_rule
    assert rule in DISTILL_ONLY_AUX_RULES
    assert is_cheap_tier_auxiliary(rule) is False


def test_skill_library_update_not_confused_with_memory_save() -> None:
    """🔴 两条共享前缀 "Review the conversation above and"，判别段必须留住。

    取短了会把两条并成一条、失去可区分性（G4.1 明确警告过）。
    """
    save = classify_auxiliary(_msgs(
        "Review the conversation above and consider saving to memory any durable facts."
    ))
    update = classify_auxiliary(_msgs(
        "Review the conversation above and update the skill library."
    ))
    assert save[1] == "hermes_memory_save"
    assert update[1] == "hermes_skill_library_update"
    assert save[1] != update[1]


@pytest.mark.parametrize("text", [
    # 讨论这些机制本身的真实用户消息 —— 措辞相近但不是模板
    "帮我看看为什么 Codex agent history 没有被识别成 aux",
    "我想让助手在我离开后回来时给个摘要，这个功能怎么做？",
    "复盘一下上面的对话，看看技能库要不要改",
    "review the code above and update the docstring",
    "The skill library needs a new entry for the retrieval module",
    "用户走开了这件事要不要记进记忆？",
])
def test_mq_s12_fingerprints_zero_false_positive_on_real_messages(text: str) -> None:
    """🔴 零误伤：真实用户消息（含同题材讨论）不得命中三条新指纹。

    匹配是子串而非严格前缀，所以判别段必须长到不会在自然语句里偶然出现。
    """
    _is_aux, rule = classify_auxiliary(_msgs(text))
    assert rule not in {
        "hermes_skill_library_update",
        "codex_history_injection",
        "session_resume_recap",
    }


def test_mq_s12_rules_registered_in_distill_only_set() -> None:
    """制度性保证：三条都在 DISTILL_ONLY 集合，assistant 侧产出照收、路由不降档。"""
    for r in ("hermes_skill_library_update", "codex_history_injection", "session_resume_recap"):
        assert r in DISTILL_ONLY_AUX_RULES


# ── Hermes 会话标题生成（2026-08-20 真实流量发现）──────────────────────────
#
# 形态：system = "You name chat sessions…"、user = **用户真实提问原文**、无 tools。
# 此前完全没有指纹：`aux_source='' auxiliary=False`，被当真实用户轮进蒸馏，
# 而它零记忆价值（内容就是"给这段对话起个标题"）。

_TITLE_SYSTEM = (
    "You name chat sessions. Given the user's opening message, write a title "
    "that lets them find this conversation again in a list.\n\nRules:\n"
    "- 3 to 7 words, sentence case"
)
#: live 那一轮的真实 user 内容（逐字抄自 Memory Hub）
_TITLE_USER = "帮我再查询一下，看看qwen3.8-27B发布了没有"


def _title_msgs() -> list[dict]:
    return [{"role": "system", "content": _TITLE_SYSTEM},
            {"role": "user", "content": _TITLE_USER}]


def test_title_generation_is_auxiliary() -> None:
    """标题生成轮必须判为 auxiliary（rebuild 侧重判这一条）。"""
    is_aux, rule = classify_auxiliary(_title_msgs())
    assert is_aux is True
    assert "you name chat sessions" in rule


def test_title_generation_fingerprint_lives_on_the_system_side() -> None:
    """🔴 指纹必须落在 system 侧 —— user 那条是**用户真实提问**，拿它做判据会误伤。

    这也是它进 `_AUX_SYSTEM_KEYWORDS` 而不是 `_AUX_USER_PATTERNS` 的原因：
    同一句用户提问，配上 Hermes 正常的 system prompt 就是真实用户轮，
    必须判 False。
    """
    normal = [{"role": "system", "content": "You are a Hermes agent built by Nous Research."},
              {"role": "user", "content": _TITLE_USER}]
    assert classify_auxiliary(normal) == (False, "")


def test_title_generation_is_cheap_tier_not_distill_only() -> None:
    """归**完整 aux**：既没记忆价值，也确实是廉价内部调用，降档合理。

    与同在 `_AUX_SYSTEM_KEYWORDS` 的 MoA reference / 压缩检查点同口径；
    U3 那三条是另一类（assistant 侧在做真实工作，故只跳蒸馏不降档）。
    """
    _is_aux, rule = classify_auxiliary(_title_msgs())
    assert rule not in DISTILL_ONLY_AUX_RULES
    assert is_cheap_tier_auxiliary(rule) is True


def test_title_generation_is_recognised_as_hermes_not_unknown() -> None:
    """🔴 规则表里也要有，否则冷启动窗口会把它落进 `unknown-<hash>` 桶。

    实测（2026-08-20 12:18:29）：proxy 重启后第一条请求恰好是它，同源继承的注册表
    还是空的 → 无处可继承 → `agent_source=fallback agent_id=unknown-0eaa8faf`。
    三条识别信号当时全空：无 Hermes system 关键词、无工具、UA 是 OpenAI SDK 默认值。
    加规则后它直接被指纹识别，不再依赖继承，冷启动窗口对它失效。
    """
    from bladex_proxy.identity import _fingerprint_agent

    agent_id, trigger, aux = _fingerprint_agent(
        _title_msgs(), None, {"user-agent": "OpenAI/Python 2.24.0"})
    assert agent_id == "hermes"
    assert aux is True
    assert trigger.startswith("system_prompt:")


def test_title_rule_precedes_the_hermes_base_rule() -> None:
    """aux 规则必须排在 base 之前，否则被 base 先命中就丢了 auxiliary 标记。"""
    from bladex_proxy.agent_rules import load_agent_rules

    names = [r.name for r in load_agent_rules() if r.agent_id == "hermes"]
    assert names.index("hermes-title-generation") < names.index("hermes")
