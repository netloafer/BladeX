"""M1-1 / M1-3 / M1-4 / M1-5：蒸馏 v4 一次改版的验收剧本。

八件事一次改版（复核汇总章第 1 层）：
  ① 内容分流（M1-2，另测）        ② 带上下文抽取 + 通道合一
  ③ 语言钉死                      ④ 时间锚
  ⑤ importance 1–10               ⑥ corrects 修正自声明槽
  ⑦ valid_until 时效槽            ⑧ 指派/打回轮的应产出物

X6 纪律：本文件的样本词汇**不得出现在实现代码里**。
实现侧只认结构（槽位、语言、通道），不认任何具体词。
"""

from __future__ import annotations

import pytest

from bladex_core.distillation import (
    DistillFact,
    DistillOutput,
    DistillTurnInput,
    build_context_digest,
)
from bladex_core.fact import Provenance
from bladex_proxy import distillation as D


# ── ③ 语言钉死（X2：一处病灶三层发病，蒸馏侧一条规则治三层）────────────────


def test_output_language_is_configuration_not_input(monkeypatch):
    """产出语言由配置决定，**不跟随输入**。

    跟随输入是病灶本身：同一轮 user 中文、assistant 英文 → 同一事实两份，
    ZH/EN 同义对 cosine 0.8639 < 判重阈值 → 都入库，而中文 query 只捞回一份。
    """
    monkeypatch.delenv("BLADEX_DISTILL_LANG", raising=False)
    assert D.distill_lang() == "zh", "默认中文（拍板 2）"

    monkeypatch.setenv("BLADEX_DISTILL_LANG", "en")
    assert D.distill_lang() == "en"

    monkeypatch.setenv("BLADEX_DISTILL_LANG", "klingon")
    assert D.distill_lang() == "zh", "未知值回落默认，不该悄悄换一种语言"


def test_language_placeholder_is_substituted(monkeypatch):
    """占位符必须被替换掉——漏替换 = 模型收到字面量 `__OUTPUT_LANG__`。"""
    monkeypatch.setenv("BLADEX_DISTILL_LANG", "en")
    rendered = D._DISTILL_TURN_SYSTEM.replace(
        D._LANG_PLACEHOLDER, D._LANG_NAMES[D.distill_lang()])
    assert D._LANG_PLACEHOLDER not in rendered
    assert "English" in rendered


def test_prompt_is_not_format_based():
    """prompt 里有大量 JSON 花括号 —— 用 str.format 就得整段转义，漏一个就 KeyError。

    钉死"用占位符替换、不用 format"这个选择，避免下一个改 prompt 的人踩回去
    （本卡开发时已经踩过一次）。
    """
    src = (D.__file__ and open(D.__file__, encoding="utf-8").read()) or ""
    assert "_DISTILL_TURN_SYSTEM.format(" not in src
    assert "_DISTILL_TURN_SYSTEM.replace(" in src


def test_fidelity_language_pins_to_config():
    """保真层判"漂没漂"要按配置语言判，不按输入语言判（M1-4）。"""
    from bladex_core.distill_fidelity import check_language

    facts = [DistillFact(content="这是中文产出")]
    # 输入是英文：v3 会认为"期望 en、实得 zh"→ 判漂移并重蒸（把对的改错）
    want_default, got = check_language("this is an english source", facts)
    assert (want_default, got) == ("en", "zh")

    # M1-4：钉配置语言 zh → 不判漂移
    want_pinned, got2 = check_language("this is an english source", facts, want_lang="zh")
    assert want_pinned == got2 == "zh"


# ── ② 带上下文 + 通道合一 ────────────────────────────────────────────────


def test_context_digest_is_deterministic():
    """确定性是重建等价性的前提——同输入必须恒得同 digest。"""
    prior = ["第一轮说的话", "第二轮说的话", "第三轮说的话"]
    a = build_context_digest(prior)
    b = build_context_digest(list(prior))
    assert a == b and a


def test_context_digest_prefers_matter_summary():
    """Matter 卡 summary 已是"这件事是什么"的浓缩，比几段原始对话更能定位。"""
    got = build_context_digest(["无关的历史"], matter_summary="某件事的摘要")
    assert got == "某件事的摘要"


def test_context_digest_is_bounded_and_takes_the_tail():
    """上限 500 字符；超过 K 轮时取**最近**几轮（近的更相关）。"""
    prior = [f"第{i}轮" * 100 for i in range(10)]
    got = build_context_digest(prior)
    assert len(got) <= 500
    assert "第9轮" in got, "应取尾部最近几轮"
    assert "第0轮" not in got


def test_turn_payload_packs_user_and_assistant_together():
    """一轮一次统一调用：user 与 assistant 同包，模型同时看见"问了什么/做了什么"。"""
    p = DistillTurnInput(user_text="用户这轮的话",
                         assistant_text="模型这轮的产出", assistant_role="progress")
    body = D.build_turn_prompt(p)
    assert "用户这轮的话" in body and "模型这轮的产出" in body
    assert "progress" in body, "通道角色要让模型看见（它决定怎么读 assistant 半边）"


def test_source_field_maps_to_provenance():
    """通道合一后来源信息只能由产出侧标注，否则合流那一刻就丢了。"""
    out = D._parse_distill_json(
        '{"facts":[{"content":"甲","source":"user"},'
        '{"content":"乙","source":"assistant"}],"matter_proposals":[]}', "m")
    assert out.facts[0].provenance == Provenance.USER_DIRECT.value
    assert out.facts[1].provenance == Provenance.CONCLUSION.value


def test_assistant_role_is_decided_by_caller_not_model():
    """conclusion 还是 progress 是**调用方**才知道的事实（本轮有没有 tool call）。

    让模型猜等于把一个确定量交给概率。
    """
    stub = DistillOutput(facts=[
        DistillFact(content="甲", provenance=Provenance.CONCLUSION.value),
        DistillFact(content="乙", provenance=Provenance.USER_DIRECT.value),
    ])
    # 复刻 _distill_with 里的改写逻辑（assistant_role="progress"）
    for f in stub.facts:
        if f.provenance == Provenance.CONCLUSION.value:
            f.provenance = "progress"
    assert [f.provenance for f in stub.facts] == ["progress", Provenance.USER_DIRECT.value], (
        "只改 assistant 侧，user 侧不动"
    )


# ── Q3：输入预算按部分分配（不是把一个数调大）────────────────────────────


def test_each_part_has_its_own_budget():
    """assistant 正文不许被 user 正文挤掉。

    这是本次改版最容易犯的错：沿用"一条输入 2000 字符"的总额上限，
    三段拼一起后排在后面的 assistant 会被整段截掉 —— 等于用新 prompt 复现 D2
    （结论通道静默截断），而 assistant 通道扛着 66% 的 Fact 产出。
    """
    p = DistillTurnInput(
        context_digest="甲" * 5000,
        user_text="乙" * 5000,
        assistant_text="丙" * 5000,
        assistant_role="conclusion",
    )
    body = D.build_turn_prompt(p)
    # 用 CJK 填充：标签本身是 ASCII，不会把标签里的字母算进来
    # （首版用 "C"/"U"/"A" 填充，`CONTEXT:` 标签自带一个 C，多数了 1 个）
    assert body.count("甲") == D._MAX_CONTEXT_CHARS
    assert body.count("乙") == D._MAX_USER_CHARS
    assert body.count("丙") == D._MAX_ASSISTANT_CHARS, "assistant 有自己的预算，没被挤掉"


def test_assistant_budget_is_not_smaller_than_user():
    """assistant 半边是"做了什么"的载体，也是 D3 实测欠采的那一侧。"""
    assert D._MAX_ASSISTANT_CHARS >= D._MAX_USER_CHARS


# ── ⑤⑥⑦ 三个新槽位 + v3 兼容 ────────────────────────────────────────────


def test_new_slots_are_parsed():
    out = D._parse_distill_json(
        '{"facts":[{"content":"甲","item_kind":"assertion","source":"user",'
        '"importance":9,"valid_until":"2026-09-01","corrects":"旧的说法"}],'
        '"matter_proposals":[],"event":"completed"}', "m")
    f = out.facts[0]
    assert (f.importance, f.valid_until, f.corrects, f.event) == (
        9, "2026-09-01", "旧的说法", "completed")


@pytest.mark.parametrize("raw,want", [
    ("9", 9), (9, 9), (9.0, 9), (1, 1), (10, 10),
    (0, 0), (11, 0), (-1, 0), ("high", 0), (None, 0), ("", 0),
])
def test_rating_parsing_is_lenient_but_bounded(raw, want):
    """越界/解析不出 → 0（未评分），**不是**默认 5 分。

    默认中间值会把"未评分"伪装成"中等重要"，历史数据会被整体拉平。
    """
    assert D._parse_rating(raw) == want


def test_v3_output_degrades_to_defaults():
    """v3 的 JSON 里没有这些键 —— 必须全部落默认值，解析零回归。"""
    out = D._parse_distill_json(
        '{"facts":[{"content":"甲","item_kind":"assertion"}],"matter_proposals":[]}', "m")
    f = out.facts[0]
    assert (f.importance, f.valid_until, f.corrects, f.event, f.provenance) == (
        0, "", "", "", "")


def test_invalid_event_is_dropped():
    """event 是封闭枚举——词表外的值一律归空，不让模型自造分类。"""
    out = D._parse_distill_json(
        '{"facts":[{"content":"甲"}],"matter_proposals":[],"event":"随便编的"}', "m")
    assert out.facts[0].event == ""


def test_event_is_turn_level_not_per_fact():
    """一轮里"被打回"只发生一次；每条 fact 各带一个会让同一事件按 fact 数重复折叠。"""
    out = D._parse_distill_json(
        '{"facts":[{"content":"甲"},{"content":"乙"},{"content":"丙"}],'
        '"matter_proposals":[],"event":"reworked"}', "m")
    assert {f.event for f in out.facts} == {"reworked"}, "同一轮的 fact 共享同一个 event"


def test_event_lives_on_facts_not_on_output():
    """🔴 event 不许挂在 DistillOutput 上。

    台账 `DistillRecord` 只持久化 facts + matter_proposals；挂在 output 上的字段
    在**台账命中那条路径上会丢** → 首次蒸馏与缓存复用产出不一致 → 重建等价性破掉，
    而且只在命中时发作。
    """
    from bladex_proxy.models import DistillRecord

    persisted = set(DistillRecord.model_fields)
    assert "facts" in persisted and "matter_proposals" in persisted
    assert "event" in DistillFact.model_fields
    assert "turn_event" not in DistillOutput.model_fields, (
        "轮级字段挂 output 上会在台账往返中丢失"
    )


# ── ⑤ importance 接入（M1-5）────────────────────────────────────────────


def test_rating_overrides_kind_baseline():
    from bladex_core.importance import compute_importance

    assert compute_importance("assertion", rating=9) == pytest.approx(0.9)
    assert compute_importance("assertion", rating=2) == pytest.approx(0.2)


def test_rating_zero_falls_back_to_kind_baseline():
    """未评分（含全部历史数据）→ 退回 kind 基线 = v3 行为逐字不变。"""
    from bladex_core.importance import compute_importance

    assert compute_importance("assertion", rating=0) == compute_importance("assertion")
    assert compute_importance("preference", rating=0) == pytest.approx(0.9)


def test_rating_still_multiplies_the_dynamic_factors():
    """rating 换的是**基线**，不是整个公式——强度/引用/衰减照常作用其上。"""
    from bladex_core.importance import compute_importance

    base = compute_importance("assertion", rating=5)
    stronger = compute_importance("assertion", rating=5, strength=4)
    assert stronger > base


# ── M1-3：台账 key = hash(上下文 + 消息)（拍板 1）───────────────────────


def test_ledger_key_includes_context():
    """同一句话在不同脉络下该蒸出不同产物 → 必须是不同的 key。"""
    from bladex_proxy.storage.memory_index import _ledger_source_text_hash

    a = _ledger_source_text_hash("继续", "在聊甲事")
    b = _ledger_source_text_hash("继续", "在聊乙事")
    assert a != b


def test_ledger_key_without_context_is_unchanged():
    """空 context 走旧算法逐字不变 —— v3 台账不因本次改版整代作废。"""
    import hashlib

    from bladex_proxy.storage.memory_index import _ledger_source_text_hash

    text = "某条消息"
    assert _ledger_source_text_hash(text) == hashlib.sha256(
        text.encode()).hexdigest()[:16]
    assert _ledger_source_text_hash(text, "") == _ledger_source_text_hash(text)


def test_ledger_key_is_injective_on_the_split():
    """分隔符必须让 (digest, text) → key 单射。

    用普通拼接的话 `("ab","c")` 与 `("a","bc")` 会撞同一个 key，
    两条不同语境的记忆互相顶掉缓存。
    """
    from bladex_proxy.storage.memory_index import _ledger_source_text_hash

    assert _ledger_source_text_hash("c", "ab") != _ledger_source_text_hash("bc", "a")


def test_same_context_and_text_is_stable():
    """确定性：同 (digest, text) 恒得同 key —— rebuild 复跑零现蒸的前提。"""
    from bladex_proxy.storage.memory_index import _ledger_source_text_hash

    assert (_ledger_source_text_hash("消息", "脉络")
            == _ledger_source_text_hash("消息", "脉络"))


def test_prompt_version_bumped_once():
    """一卡一 bump——不许出现 "-002" 这种"再 bump 一次"。

    每 bump 一代就废掉一代台账（全量重蒸的钱要重烧一遍）。
    历史：M1 定 v4-turn-001；三段式卡 T1（2026-08-10，授权 bump）定
    v5-three-seg-001——那卡的纪律同款：全卡只许一次，见
    docs/planning/distill-three-segment-plan-20260809.md；
    记忆质量攻坚 T1（2026-08-15，授权 bump）定 v6-identity-time-001——
    MQ-D1/D2/D3 三改合一付一次重蒸成本，见
    docs/planning/memory-quality-problem-ledger-20260815.md；
    G9.1（2026-08-19，授权 bump）定 v7-title-noun-001——M1–M9 九款合一
    （当天 M6 补 `MINIMAL NOUN PHRASE` 后串名从 `v7-granularity-subject-001` 改过一次，
    仍只付一次钱，理由见 `distillation.TURN_PROMPT_VER` 上方）。

    🔴 **2026-08-19 改判据（不是"测试过时"，是这条判据从来没守住它自己那句话）**：
    原文是 `assert D.TURN_PROMPT_VER == "v6-identity-time-001"`。docstring 说的规矩是
    「不许出现 -002」，但钉死具体值**守不住那条规矩**——它守的是"版本号别变"，
    而版本号本来就会一卡一变。副作用是同一个断言散在三个文件里
    （本文件 / `test_distill_three_segment.py` / 当版本卡自己的文件），
    v7 一改**同时红三处**，而三处红都不代表规矩被破。
    现在断言**规矩本身**：版本串永远以 `-001` 结尾。
    一卡一 bump = 每张卡起一个新名字并停在 `-001`；出现 `-002` 就是同一张卡里 bump 了两次。
    "当前值具体是多少"归当前版本那张卡自己的文件管，此处不重复
    （2026-08-19 起 = `test_distill_v7_granularity.py`）。
    """
    assert D.TURN_PROMPT_VER.endswith("-001"), (
        f"{D.TURN_PROMPT_VER}：一卡只许 bump 一次。同一张卡里再改 prompt 就要再付一次"
        f"全量重蒸的钱——要改就停下来重排卡，而不是把版本号 +1。"
    )
    assert D.PROMPT_VER == "v3-items-001", "v3 常量保留供旧台账读取"
