"""v6 三改合一（MQ-D1 标题身份化 / MQ-D2 时间锚 / MQ-D3 事件不进 profile_obs）。

这些是 **prompt 条款**，产出正确与否只能靠真实模型验（T3 定向重放 + LongMemEval）。
本文件钉的是"条款还在、语义没被后来的改动悄悄抹掉"——三条都是被真实错题反推出来
的（证据出处见 `distillation.TURN_PROMPT_VER` 上方注释块），删掉任何一条都会让
对应的错题类型无声回归，而单测里看不出来。锚词测试的先例：M1-4 语言钉死条款。

解析侧不变是本轮的**边界**：item_kind 六值枚举、schema、字段名一个都没动，
所以 v5 的解析测试全绿即证明"只改了措辞、没改契约"。
"""

from __future__ import annotations

from bladex_proxy.distillation import (
    _DISTILL_CONCLUSION_SYSTEM,
    _DISTILL_SYSTEM,
    _DISTILL_TURN_SYSTEM,
    _VALID_ITEM_KINDS,
    TURN_PROMPT_VER,
)


# ── MQ-D1：提案标题 = 主题身份，不是任务句 ────────────────────────────────


def test_d1_title_asks_for_identity_not_action():
    """标题是"这件事叫什么"，不是"这轮干了什么"。

    机制上为什么要求逐字相同：`attribution` 的 L3 走 proposal_title 与 Matter
    原生键的**全串规范化相等**——标题一稳定，同一件事的后续轮直接精确命中；
    判据本身没有放松，误合并=0 红线不受影响。
    """
    assert "IDENTITY" in _DISTILL_TURN_SYSTEM
    assert "SAME title verbatim" in _DISTILL_TURN_SYSTEM
    # 反例保留：动词切面各开一张卡是 qwen 会话 8 卡的实证形态
    assert "拉取并验证 qwen3.8:27b" in _DISTILL_TURN_SYSTEM
    assert "qwen3.8:27b 本地部署评估" in _DISTILL_TURN_SYSTEM


def test_d1_title_forbids_turn_local_wording():
    """标题里不许出现本轮的步骤/状态/结果——那正是切面化的入口。"""
    assert "not this turn's action, step, status or outcome" in _DISTILL_TURN_SYSTEM


# ── MQ-D2：时间锚三义务 ────────────────────────────────────────────────────


def test_d2_time_anchor_has_three_obligations():
    """RESOLVE / AS-OF / IN CONTENT 三条缺一不可。

    缺 RESOLVE → "六周前"锚到回放日；缺 AS-OF → "最近在用 X"无法判定是否还成立
    （T1 错题 gpt4_2f56ae70）；缺 IN CONTENT → 日期只落 attribute/entities，
    向量与正文都看不见它（C 类 BBQ / 地毯两题的形状）。
    """
    for anchor in ("RESOLVE:", "AS-OF:", "IN CONTENT:"):
        assert anchor in _DISTILL_TURN_SYSTEM, anchor
    # 唯一时钟条款（v5 就有）必须保留：三条义务都以它为前提
    assert "it is the ONLY clock" in _DISTILL_TURN_SYSTEM
    assert "never use your own idea of" in _DISTILL_TURN_SYSTEM


def test_d2_as_of_example_is_concrete():
    """as-of 给的是"截至 DATE"的成品句，不是抽象要求——抽象要求模型会绕过。"""
    assert "截至 2026-08-14" in _DISTILL_TURN_SYSTEM
    assert "用户最近在用 HBO add-on" in _DISTILL_TURN_SYSTEM  # 反例原样保留


# ── MQ-D3：带日期的一次性事件 = assertion ─────────────────────────────────


def test_d3_kind_rule_present_with_both_polarities():
    """A 类 11 题的写入侧半边：入群/开始/购买日期落 profile_obs = 死数据。

    正反例都要在：只说"不要 profile_obs"会让模型把标准画像观察也改判 assertion，
    OK 例（neovim）守的是这一侧。
    """
    assert "KIND RULE" in _DISTILL_TURN_SYSTEM
    assert "用户于 2026-08-07 加入 Page Turners 读书会" in _DISTILL_TURN_SYSTEM
    assert "用户的日常编辑器是 neovim" in _DISTILL_TURN_SYSTEM


def test_d3_profile_obs_meaning_narrowed_to_standing_traits():
    assert "STANDING trait" in _DISTILL_TURN_SYSTEM
    assert "Never used for something that happened on a date." in _DISTILL_TURN_SYSTEM


def test_d3_does_not_change_the_kind_enum():
    """边界：只改分类**指引**，六值封闭枚举不动（改枚举 = 改 schema = 另一件事）。"""
    assert _VALID_ITEM_KINDS == {
        "assertion", "preference", "procedure", "lesson", "file_ref", "profile_obs"}


# ── MQ-D8：提案数量收敛（prompt 侧半边）────────────────────────────────────


def test_d8_proposals_converge_to_one():
    """一个 turn 正常只产 1 个提案。

    N 个 payload × 0-3 个提案、跨 payload 零收敛，而 L5 只取 `proposal_titles[0]`
    开卡 —— 提案越多，开出无关卡的面越大。跨 payload 收敛在归属层，不在这里。
    """
    assert "normally exactly 1" in _DISTILL_TURN_SYSTEM
    assert "Never pad the list" in _DISTILL_TURN_SYSTEM


# ── MQ-D9：三段自洽（topic ↔ facts，entities ⊆ keywords）──────────────────


def test_d9_topic_must_match_facts():
    assert "CONSISTENCY:" in _DISTILL_TURN_SYSTEM
    assert 'must name what the emitted "facts" are about' in _DISTILL_TURN_SYSTEM


def test_d9_entities_must_be_subset_of_keywords():
    """原文是"where applicable"的软要求且无校验 —— 用户观察到的"fact 跟关键词无关"。"""
    assert 'Every entity MUST also appear' in _DISTILL_TURN_SYSTEM
    assert "where applicable" not in _DISTILL_TURN_SYSTEM, (
        "软措辞回潮：entities ⊆ keywords 必须是硬要求"
    )


# ── 边界：只动 turn 路 ─────────────────────────────────────────────────────


def test_only_turn_prompt_touched():
    """v3 / conclusion 两个 prompt 生产零流量（T1a 台账实测），本轮一个字不动。

    它们的完整文本 hash 由 `test_distill_three_segment.py` 冻结；这里只钉
    "新条款没有溢出到它们身上"，好让 grep 全部消费点这件事有测试背书。
    """
    for other in (_DISTILL_SYSTEM, _DISTILL_CONCLUSION_SYSTEM):
        assert "KIND RULE" not in other
        assert "AS-OF:" not in other
        assert "IDENTITY" not in other


def test_v6_clauses_survived_the_v7_bump():
    """本文件的职责是"v6 六条条款还在"，**不是**"版本号等于 v6"。

    2026-08-19 v7（G9.1 九款）把 turn prompt 又改了一版，原来那句
    `assert TURN_PROMPT_VER == "v6-identity-time-001"` 于是有两种改法：
    跟着改成 v7（=两个文件都断言版本号，下次 bump 要改两处），或者交出去。
    交出去更对：**版本号的断言归"当前版本"那个文件**
    （`test_distill_v7_granularity.py::test_version_bumped_exactly_once_this_card`），
    历史卡的文件只守自己那几条条款没被后来的改动抹掉——上面每个 test 都是。
    这里只钉一条：版本确实往前走了，v6 不再是当前版本。

    🔴 第一版这里还写了 `assert TURN_PROMPT_VER.startswith("v7-")` —— 犯的正是
    本条要避免的那个错（钉死一个会随下一次 bump 过期的值），v8 一来就红。
    已删。历史卡的文件不该知道当前版本叫什么。
    """
    assert TURN_PROMPT_VER != "v6-identity-time-001"
