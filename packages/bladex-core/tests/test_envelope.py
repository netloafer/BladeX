"""信封剥离验收剧本（ADR-0024 T3）。

覆盖：真实信封形状剥离 / 自反馈回声阻断 / 聊天式 agent 零回归 /
分段边界 / 蒸馏管线接入（结论通道不受影响）。
"""

from __future__ import annotations

import pytest
from bladex_core.envelope import (
    is_pure_envelope,
    prepare_distill_inputs,
    segment_long_text,
    strip_envelopes,
)

MIN_C, MAX_C = 20, 500


# ── 真实信封形状（取自归档 Memory Hub 的实际样本结构）──


def test_strip_safety_transcript():
    """Claude Code 安全检查子 agent：整段会话转录 + 判定指令包着一句真实意图。"""
    text = (
        "<transcript>\n"
        '{"user":"<command-name>/model</command-name>"}\n'
        '{"Bash":"ps aux | grep proxy"}\n' * 50
        + "</transcript>\n\n"
        "Err on the side of blocking. Your ENTIRE response MUST begin with <block>."
    )
    clean, kinds = strip_envelopes(text)
    assert "transcript" in kinds
    assert "ps aux" not in clean
    assert "Err on the side of blocking" in clean
    assert len(clean) < len(text) / 10


def test_strip_slash_command_and_local_output():
    text = (
        "<command-name>/model</command-name>\n"
        "<command-message>model</command-message>\n"
        "<command-args></command-args>\n"
        "<local-command-stdout>Set model to glm-5.2</local-command-stdout>"
    )
    clean, kinds = strip_envelopes(text)
    assert set(kinds) == {"slash_command", "local_command"}
    assert clean == ""
    assert is_pure_envelope(text)


def test_strip_system_reminder_keeps_user_intent():
    text = (
        "<system-reminder>\nCLAUDE.md 内容……\n" + "x" * 5000 + "\n</system-reminder>\n"
        "帮我把 T3 的剥离规则实现一下"
    )
    clean, kinds = strip_envelopes(text)
    assert kinds == ["system_reminder"]
    assert clean == "帮我把 T3 的剥离规则实现一下"


def test_strip_task_notification():
    text = (
        "<task-notification><task-id>t1</task-id><status>done</status>"
        "<summary>构建完成</summary></task-notification>\n继续下一步"
    )
    clean, kinds = strip_envelopes(text)
    assert kinds == ["task_notification"]
    assert clean == "继续下一步"


def test_multiple_envelopes_in_one_message():
    text = (
        "<system-reminder>env</system-reminder>"
        "<transcript>log</transcript>"
        "<user_claude_md>md</user_claude_md>"
        "真实问题在这里"
    )
    clean, kinds = strip_envelopes(text)
    assert set(kinds) == {"transcript", "system_reminder", "user_claude_md"}
    assert clean == "真实问题在这里"


# ── 自反馈闭环阻断（ADR-0025 §6）──


def test_bladex_memory_echo_is_stripped():
    """真实数据里有 24 条 user 消息回带了我们自己的注入块。

    不剥 = 自己注入的记忆被蒸馏回 Memory Index = 自反馈闭环。
    """
    text = (
        "<bladex-memory>\n- [MUST/NEVER] 用中文回答\n- 用户在做 BladeX 项目\n"
        "</bladex-memory>\n实际问题：proxy 502 怎么排查"
    )
    clean, kinds = strip_envelopes(text)
    assert kinds == ["bladex_memory_echo"]
    assert "MUST/NEVER" not in clean
    assert "用户在做 BladeX 项目" not in clean
    assert clean == "实际问题：proxy 502 怎么排查"


def test_echo_blocked_end_to_end_in_distill_inputs():
    text = "<bladex-memory>\n- 用户偏好 glm-5.2\n</bladex-memory>\n改用 opus 试试"
    texts, kinds = prepare_distill_inputs(text, MIN_C, MAX_C)
    assert "bladex_memory_echo" in kinds
    assert all("glm-5.2" not in t for t in texts)


# ── 未闭合信封（流截断）──


def test_unclosed_envelope_truncates():
    text = "先看这个问题\n<transcript>\n" + '{"Bash":"x"}\n' * 100
    clean, kinds = strip_envelopes(text)
    assert "unclosed_envelope" in kinds
    assert clean == "先看这个问题"


# ── 聊天式 agent 零回归（hermes 实测 0 命中、残留 100%）──


@pytest.mark.parametrize("text", [
    "帮我查一下北京今天的天气",
    "这是一段普通的长消息。" * 60,
    "用 <> 尖括号写点东西，比如 a < b > c",
    "```python\nprint(1)\n```",
])
def test_plain_chat_untouched(text):
    clean, kinds = strip_envelopes(text)
    assert kinds == []
    assert clean == text.strip()


def test_empty_input():
    assert strip_envelopes("") == ("", [])
    assert prepare_distill_inputs("", MIN_C, MAX_C) == ([], [])


# ── 长度判定发生在剥离之后（T3 的核心修复）──


def test_length_check_happens_after_stripping():
    """旧逻辑：整条 >500 直接丢弃。新逻辑：剥完只剩 30 字符 → 正常进蒸馏。"""
    intent = "把 ADR-0024 T3 的验收剧本补齐，顺便跑一遍归档回放"
    assert MIN_C <= len(intent) <= MAX_C
    text = "<transcript>" + "x" * 50000 + "</transcript>\n" + intent
    assert len(text) > MAX_C          # 旧逻辑在这里就 continue 了
    texts, kinds = prepare_distill_inputs(text, MIN_C, MAX_C)
    assert texts == [intent]
    assert "transcript" in kinds


def test_pure_envelope_yields_no_candidate():
    text = "<command-name>/clear</command-name>"
    texts, kinds = prepare_distill_inputs(text, MIN_C, MAX_C)
    assert texts == []
    assert kinds == ["slash_command"]


def test_too_short_after_stripping_dropped():
    texts, _ = prepare_distill_inputs("<transcript>x</transcript>\n好的", MIN_C, MAX_C)
    assert texts == []


# ── 分段（剥离后仍超长的 9%）──


def test_segment_splits_on_blank_lines():
    text = "\n\n".join(["段落" * 100] * 4)     # 每段 200 字符
    segs = segment_long_text(text, max_chars=500)
    assert len(segs) > 1
    assert all(len(s) <= 500 for s in segs)


def test_segment_hard_cuts_oversized_paragraph():
    segs = segment_long_text("x" * 2000, max_chars=500)
    assert len(segs) == 4
    assert all(len(s) == 500 for s in segs)


def test_segment_respects_max_segments_cap():
    """28 万字符的粘贴不该产出上百个候选打爆蒸馏预算。"""
    segs = segment_long_text("y" * 280000, max_chars=500, max_segments=6)
    assert len(segs) == 6


def test_segment_short_text_passthrough():
    assert segment_long_text("短", max_chars=500) == ["短"]


def test_prepare_segments_share_nothing_but_are_ordered():
    text = "<system-reminder>env</system-reminder>\n" + "\n\n".join(["块" * 300] * 3)
    texts, kinds = prepare_distill_inputs(text, MIN_C, MAX_C)
    assert kinds == ["system_reminder"]
    assert len(texts) >= 2
    assert all(MIN_C <= len(t) <= MAX_C for t in texts)


# ── 确定性（重建等价性前提）──


def test_deterministic():
    text = "<transcript>a</transcript>\n" + "z" * 3000
    assert prepare_distill_inputs(text, MIN_C, MAX_C) == prepare_distill_inputs(text, MIN_C, MAX_C)


# ── 蒸馏管线接入 ──


def test_consolidator_strips_envelope_and_keeps_conclusion():
    """管线级：信封被剥、真实意图进蒸馏，且结论通道不受影响。

    🔴 结论通道扛着 66% 的 Fact 产出——纯信封轮次也必须照收结论。
    """
    from bladex_core.consolidation_proxy import ProxyConsolidator
    from bladex_core.distillation import DistillFact, DistillOutput
    from bladex_core.fact import ConversationTurn

    seen: list[str] = []

    class _Distiller:
        model_name = "test-model"

        def distill(self, text: str) -> DistillOutput:
            seen.append(text)
            return DistillOutput(
                facts=[DistillFact(content=f"fact::{text[:20]}", kind="event", entities=["e"])],
                matter_proposals=[], model_name="test-model",
            )

        def distill_conclusion(self, text: str) -> DistillOutput:
            return DistillOutput(
                facts=[DistillFact(content="conclusion-fact", kind="decision", entities=["e"])],
                matter_proposals=[], model_name="test-model",
            )

    intent = "请帮我修复 proxy 的 502 错误，这已经是今天第三次出现了"
    assert MIN_C <= len(intent) <= MAX_C
    turns = [
        ConversationTurn(                        # 真实意图裹在 transcript 里
            session_id="s", user_id="u", ledger_key="k1",
            user_messages=["<transcript>" + "L" * 60000 + "</transcript>\n" + intent],
            assistant_response="已修复",
            assistant_conclusion="根因是上游配额耗尽，已切换到备用模型池并加了熔断。" * 3,
        ),
        ConversationTurn(                        # 纯信封轮：无意图，但有结论
            session_id="s", user_id="u", ledger_key="k2",
            user_messages=["<command-name>/model</command-name>"],
            assistant_response="ok",
            assistant_conclusion="模型已切换为 glm-5.2，并保存为新会话的默认模型。" * 3,
        ),
    ]

    c = ProxyConsolidator(distiller=_Distiller())
    cands = c._collect_candidates(turns)

    # 信封被剥掉，只有真实意图进了蒸馏器
    assert seen == [intent]
    # 两轮的结论都收到了（纯信封轮也不例外），且每轮只收一次
    conclusions = [x for x in cands if x["content"] == "conclusion-fact"]
    assert len(conclusions) == 2
    assert {x["ledger_key"] for x in conclusions} == {"k1", "k2"}


def test_consolidator_conclusion_collected_once_per_turn():
    """一轮多条 user 消息时，结论不得按消息数重复收。"""
    from bladex_core.consolidation_proxy import ProxyConsolidator
    from bladex_core.distillation import DistillFact, DistillOutput
    from bladex_core.fact import ConversationTurn

    class _Distiller:
        model_name = "m"

        def distill(self, text: str) -> DistillOutput:
            return DistillOutput(facts=[DistillFact(content=f"f::{text}", kind="event")],
                                 matter_proposals=[], model_name="m")

        def distill_conclusion(self, text: str) -> DistillOutput:
            return DistillOutput(facts=[DistillFact(content="conclusion-fact", kind="decision")],
                                 matter_proposals=[], model_name="m")

    turn = ConversationTurn(
        session_id="s", user_id="u", ledger_key="k",
        user_messages=[
            "第一个问题：请检查 proxy 的路由配置是否正确加载了模型池",
            "第二个问题：consolidator 的蒸馏台账命中率现在是多少",
        ],
        assistant_response="r",
        assistant_conclusion="这是一条足够长的结论文本，用于验证结论通道每轮只收一次。" * 3,
    )
    cands = ProxyConsolidator(distiller=_Distiller())._collect_candidates([turn])
    assert len([x for x in cands if x["content"] == "conclusion-fact"]) == 1
    # 两条 user 消息各自进了蒸馏（信封剥离不影响正常消息）
    assert len([x for x in cands if x["content"].startswith("f::")]) == 2
