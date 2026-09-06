"""G12.3 `task_goal` 捕获的纯函数验收（判据 `g12-3-goal-criteria-20260822.md` §3）。

用例编号与判据文档一一对应：N* = 阴性，P* = 阳性对照。
🔴 **阴性单独存在没有意义**——全拒也能让 N1–N10 全绿，故 P1/P2 是这组的必要一半。
"""

from __future__ import annotations

import pytest
from bladex_core.task_goal import (
    GOAL_ABSENT_DROPPED,
    GOAL_ABSENT_EMPTY,
    GOAL_ABSENT_NO_CONTEXT,
    GOAL_ABSENT_NO_KEY,
    GOAL_ABSENT_REASONS,
    GOAL_ABSENT_SCAFFOLD,
    GOAL_SOURCE_FIRST_TURN_USER,
    GOAL_SOURCES,
    TURN_CLASSES,
    TURN_DROPPED,
    TURN_REAL,
    TURN_SCAFFOLD,
    capture_first_turn_goal,
)

LK = "local/hermes:default/sess-1/0000000001"

# 判据文档 §3 P1：真实用户目标，含两个限定条件（"重点看…"/"不用管…"）——
# 提炼成「泰山啤酒破产重整尽调」正是 ADR-0031 §4.1 举的反例。
P1_TEXT = "帮我搞定泰山啤酒的破产重整尽调，重点看股权变动，不用管财务报表"


def _cap(text: str, *, turn_class: str = TURN_REAL, ledger_key: str = LK):
    return capture_first_turn_goal(
        ledger_key=ledger_key, user_text=text, turn_class=turn_class)


# ── 阳性对照 ────────────────────────────────────────────────────────────

def test_p1_goal_is_verbatim_including_qualifiers():
    """P1：goal 与用户原话**逐字**一致（不是相似），限定条件一个都不能少。"""
    cap = _cap(P1_TEXT)
    assert cap.text == P1_TEXT
    assert "不用管财务报表" in cap.text and "重点看股权变动" in cap.text
    assert cap.source == GOAL_SOURCE_FIRST_TURN_USER
    assert cap.reason == ""
    assert cap.ledger_key == LK
    assert cap.captured


def test_p2_long_real_goal_is_not_truncated():
    """P2：565 字符量级的真实多行任务陈述，**存不截断**。

    🔴 这条钉的是判据 §2.C：注入体积由渲染层管，捕获层一个字都不许少——
    截断会让"用户一眼认出是自己说的"失效，而那是 goal 的全部价值。
    """
    body = (
        "深度调研 LiteLLM proxy 的核心架构和请求处理机制，输出一份结构清晰的技术分析报告。\n\n"
        "需要覆盖以下维度：\n"
    ) + "".join(f"{i}. 维度{i}：请展开说明其设计取舍与实现要点。\n" for i in range(1, 30))
    assert len(body) > 600          # 前提：确实超过建议的注入上限
    cap = _cap(body)
    assert cap.text == body.strip()  # 逐字（strip_envelopes 只去首尾空白）
    assert len(cap.text) > 600


def test_p2b_multiline_goal_keeps_all_paragraphs():
    """定位 = 剥离后全文，**不挑段**：任何"只取第一段"的实现都会红在这里。"""
    text = "第一段：把 BladeX 的注入面接上。\n\n第二段：注意别动 assembly 的确定性。"
    cap = _cap(text)
    assert cap.text == text
    assert "第二段" in cap.text


# ── 阴性（判据 §3 N1–N10）────────────────────────────────────────────────

@pytest.mark.parametrize("turn_class,reason", [
    (TURN_SCAFFOLD, GOAL_ABSENT_SCAFFOLD),
    (TURN_DROPPED, GOAL_ABSENT_DROPPED),
])
def test_n1_scaffold_and_dropped_turns_never_supply_goal(turn_class, reason):
    """N1：脚手架轮/整轮丢的轮次即使带着像样的文本，也不许当目标。"""
    cap = _cap("<command-name>/compact</command-name> 顺便把注入面修一下",
               turn_class=turn_class)
    assert cap.text == ""
    assert cap.reason == reason
    assert cap.source == ""


def test_n2_bladex_memory_echo_is_refused():
    """N2：`<bladex-memory>` 以 user 角色回流（真实数据 24 条）——剥完为空。"""
    cap = _cap("<bladex-memory>用户偏好：喜欢简洁回答</bladex-memory>")
    assert cap.text == ""
    assert cap.reason == GOAL_ABSENT_EMPTY


def test_n3_transcript_in_user_is_refused_by_a1_not_by_stripping():
    """N3：转录形态里的用户原话属**历史轮**，不是本轮目标。

    🔴 这条同时解释了 A1 为什么必须排在 A2 前面：单看剥离，
    `User: …\\n\\nAssistant: …` 会被解成"首段真实问句"并**通过** A2——
    挡住它的是 aux 判定（`hermes_transcript_replay` ⇒ SCAFFOLD），不是剥离。
    模型产的文本以 user role 回流，正是"只有用户能改 goal"的绕过路径。
    """
    transcript = "User: 帮我查一下泰山啤酒的债权人\n\nAssistant: 好的，我来查。"
    # 生产路径（A1 先判）：拒
    assert _cap(transcript, turn_class=TURN_SCAFFOLD).reason == GOAL_ABSENT_SCAFFOLD
    # 若绕过 A1，剥离层**拦不住**——这就是为什么 A1 不能省
    assert _cap(transcript, turn_class=TURN_REAL).text != ""


def test_n10_missing_ledger_key_is_refused_and_reported():
    """N10：合成 fact / 存量回放没有 ledger key ⇒ 留空且不 crash。"""
    cap = _cap(P1_TEXT, ledger_key="")
    assert cap.text == ""
    assert cap.reason == GOAL_ABSENT_NO_KEY


def test_unknown_turn_class_fails_closed():
    """闭集外的类别 = 不知道这轮是什么 ⇒ 不猜（fail-closed）。"""
    cap = _cap(P1_TEXT, turn_class="whatever")
    assert cap.text == ""
    assert cap.reason == GOAL_ABSENT_NO_CONTEXT


def test_empty_input_is_refused():
    assert _cap("").reason == GOAL_ABSENT_EMPTY


# ── N9 重建等价性 ───────────────────────────────────────────────────────

def test_n9_capture_is_a_pure_function():
    """N9：同一条 Memory Hub 重放两次 ⇒ goal 逐字相同（重建等价性）。"""
    a = _cap(P1_TEXT)
    b = _cap(P1_TEXT)
    assert (a.text, a.source, a.reason, a.ledger_key) == (
        b.text, b.source, b.reason, b.ledger_key)


# ── 闭集守卫（硬约束 8：新加一档先让这里红一次）────────────────────────

def test_closed_sets_are_declared_once():
    """闭集只在生产模块声明一处；这里断言的是**成员**，不是另抄一份定义。"""
    assert set(TURN_CLASSES) == {TURN_REAL, TURN_SCAFFOLD, TURN_DROPPED}
    assert set(GOAL_ABSENT_REASONS) == {
        GOAL_ABSENT_SCAFFOLD, GOAL_ABSENT_DROPPED, GOAL_ABSENT_EMPTY,
        GOAL_ABSENT_NO_KEY, GOAL_ABSENT_NO_CONTEXT,
    }
    assert len(set(GOAL_ABSENT_REASONS)) == len(GOAL_ABSENT_REASONS)
    assert len(set(GOAL_SOURCES)) == len(GOAL_SOURCES)


def test_every_produced_reason_is_in_the_closed_set():
    """穷举本函数能产出的 reason，逐个必须在闭集里（枚举取值对账，H1 第四类）。"""
    produced = {
        _cap("x", turn_class=TURN_SCAFFOLD).reason,
        _cap("x", turn_class=TURN_DROPPED).reason,
        _cap("", turn_class=TURN_REAL).reason,
        _cap(P1_TEXT, ledger_key="").reason,
        _cap(P1_TEXT, turn_class="???").reason,
    }
    assert produced <= set(GOAL_ABSENT_REASONS)
    assert produced == set(GOAL_ABSENT_REASONS), "有一档 reason 永远产不出来 = 死枚举"


def test_captured_result_carries_source_and_no_reason():
    """text 非空 ⇔ reason 为空（"空着比编造好"的另一半：取到了就必须标来源）。"""
    cap = _cap(P1_TEXT)
    assert bool(cap.text) is not bool(cap.reason)
    assert cap.source in GOAL_SOURCES
