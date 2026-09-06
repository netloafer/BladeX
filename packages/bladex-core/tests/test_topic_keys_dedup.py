"""主题键机制验收（matter-dedup-keywords-20260814 卡，拍板 A/B/D）。

事故复现驱动：同一件事（泰安仁信入股泰山啤酒）当天被开出 4 个 Matter——
标题只差动词（核实/查询/确认/查明），L3 全串匹配永不相等 → L4 判 none →
L5 反复新开。本文件用**真实事故的标题**做正向剧本，用跨主题对做阴性对照
（误合并率=0 是第一红线，宁可漏合不可错合）。
"""

from __future__ import annotations

from bladex_core.attribution import (
    AttributionPipeline,
    AttributionSource,
)
from bladex_core.fact import Fact
from bladex_core.topic_keys import (
    accumulate_topic_keys,
    clean_topic_keys,
    derive_topic_keys,
    is_strong_topic_match,
    is_valid_topic_key,
    normalize_key,
)

# ── 事故现场的真实数据形态 ──────────────────────────────────────────────────

_VERB_VARIANT_TITLES = [
    "核实泰安仁信入股泰山啤酒的时间与方式（增资扩股或受让老股）并更新股权报告",
    "查询泰安仁信文化投资合伙企业（有限合伙）入股泰山啤酒的方式（增资扩股或受让老股）并更新HTML报告",
    "确认泰安仁信入股泰山啤酒的方式（增资扩股或受让老股）并查明華太山控股（香港）的进入路径",
    "查明泰安仁信文化投资合伙企业（有限合伙）入股泰山啤酒的方式（增资扩股还是受让老股）",
]
_SHARED_KEYWORDS = ["泰安仁信", "泰山啤酒", "增资扩股", "受让老股"]
_POISON_ENTITIES = ["opencli", "Chrome", "16%股份转让", "张开利62.4",
                    "2025.10审计负债率106.6", "企信宝"]


def _fact(i: int, *, title: str, keywords: list[str] | None = None,
          entities: list[str] | None = None) -> Fact:
    return Fact(id=f"f{i}", content=f"第 {i} 条：{title[:40]} 的调查进展。",
                source_user_id="u1", source_session=f"s{i}",
                proposal_titles=[title],
                keywords=list(keywords or []),
                entities=list(entities or []))


# ── 毒词过滤（拍板 D）────────────────────────────────────────────────────────


def test_poison_words_rejected():
    for bad in _POISON_ENTITIES + ["16%", "2022-11-21", "vs", "的", "P3", "e.g"]:
        assert not is_valid_topic_key(normalize_key(bad)), f"毒词漏网：{bad!r}"


def test_legit_keys_kept():
    for good in ["泰安仁信", "泰山啤酒", "增资扩股", "華太山控股", "rocksdict-repair"]:
        assert is_valid_topic_key(normalize_key(good)), f"合法键被误杀：{good!r}"


def test_clean_topic_keys_order_and_dedup():
    got = clean_topic_keys(["泰山啤酒", "opencli", "泰山啤酒", "泰安仁信", "16%"])
    assert got == ["泰山啤酒", "泰安仁信"]


def test_derive_falls_back_to_entities_and_title_when_no_keywords():
    """存量台账无 keywords → 从 entities + 标题分词确定性派生（G6 安全）。"""
    keys = derive_topic_keys(None, "", [_VERB_VARIANT_TITLES[0]],
                             ["泰安仁信文化投资合伙企业（有限合伙）", "泰山啤酒", "opencli"])
    assert "泰山啤酒" in keys
    assert "opencli" not in keys
    # 派生是确定性的（同输入同输出）
    assert keys == derive_topic_keys(None, "", [_VERB_VARIANT_TITLES[0]],
                                     ["泰安仁信文化投资合伙企业（有限合伙）", "泰山啤酒", "opencli"])


# ── 匹配门槛（高精度：单键永不放行）─────────────────────────────────────────


def test_single_key_never_matches():
    assert not is_strong_topic_match({"泰山啤酒"}, {"泰山啤酒": 5, "别的": 1})


def test_three_keys_match_and_two_plus_jaccard():
    assert is_strong_topic_match({"泰安仁信", "泰山啤酒", "增资扩股", "无关"},
                                 {"泰安仁信": 1, "泰山啤酒": 2, "增资扩股": 1})
    # 2 键且覆盖小集合 100%
    assert is_strong_topic_match({"泰安仁信", "泰山啤酒"},
                                 {"泰安仁信": 1, "泰山啤酒": 1, "股权报告": 1})
    # 2 键但小集合覆盖不足 60%
    assert not is_strong_topic_match({"泰安仁信", "泰山啤酒", "a1键", "b2键", "c3键"},
                                     {"泰安仁信": 1, "泰山啤酒": 1, "x键": 1, "y键": 1, "z键": 1})


def test_accumulate_caps_deterministically():
    tk: dict[str, int] = {}
    for i in range(40):
        tk = accumulate_topic_keys(tk, [f"键{i:02d}", "泰山啤酒"])
    assert len(tk) == 30
    assert tk["泰山啤酒"] == 40          # 高频键永远留下
    assert accumulate_topic_keys(tk, []) == tk  # 空输入幂等


# ── 事故复现主剧本（拍板 A+B）：四个动词变体 → 一个 Matter ────────────────────


def test_verb_variant_proposals_converge_to_one_matter():
    """L5 只开第一次；后续动词变体经 L3 主题键子层挂进同一个 Matter。

    无 L4 judge（纯规则路径）——事故当天 L4 判了 none，修复不能依赖 L4 改判。
    """
    pipeline = AttributionPipeline(link_judge=None)
    matters = []
    decided = []
    for i, title in enumerate(_VERB_VARIANT_TITLES):
        # 刻意不给 entities：全串必 miss、实体交集必 miss → 只有主题键子层能接住
        fact = _fact(i, title=title, keywords=_SHARED_KEYWORDS)
        d = pipeline.attribute(fact, candidate_matters=list(matters))
        decided.append(d)
        if d.is_new_matter:
            matters.append(pipeline.create_matter_for_decision(d))

    assert decided[0].source == AttributionSource.PROVISIONAL, "首条应 L5 新开"
    assert len(matters) == 1, (
        f"动词变体开出了 {len(matters)} 个 Matter——四开事故未修复："
        f"{[d.decision.get('layer') for d in decided]}")
    for d in decided[1:]:
        assert d.matter_id == matters[0].matter_id
        assert d.source == AttributionSource.ALIAS
        assert d.signal == "alias:topic_keys_overlap"


def test_verb_variants_converge_without_keywords_via_derivation():
    """存量形态（keywords 空）：靠 entities/标题派生兜底仍应收敛。"""
    pipeline = AttributionPipeline(link_judge=None)
    matters = []
    shared_entities = ["泰安仁信文化投资合伙企业（有限合伙）", "泰山啤酒", "增资扩股"]
    for i, title in enumerate(_VERB_VARIANT_TITLES):
        d = pipeline.attribute(_fact(i, title=title, entities=shared_entities),
                               candidate_matters=list(matters))
        if d.is_new_matter:
            matters.append(pipeline.create_matter_for_decision(d))
    assert len(matters) == 1


def test_cross_topic_negative_control():
    """🔴 阴性对照（误合并红线）：不同主题绝不因子层被吸走。"""
    pipeline = AttributionPipeline(link_judge=None)
    d1 = pipeline.attribute(
        _fact(0, title=_VERB_VARIANT_TITLES[0], keywords=_SHARED_KEYWORDS),
        candidate_matters=[])
    m1 = pipeline.create_matter_for_decision(d1)
    d2 = pipeline.attribute(
        _fact(1, title="阿根廷 vs 佛得角友谊赛赔率分析",
              keywords=["阿根廷", "佛得角", "赔率"]),
        candidate_matters=[m1])
    assert d2.matter_id != m1.matter_id, "跨主题 fact 被主题键子层误吸"
    # 单个共享泛词也不行
    d3 = pipeline.attribute(
        _fact(2, title="泰山啤酒品牌历史科普", keywords=["泰山啤酒", "品牌历史"]),
        candidate_matters=[m1])
    assert d3.matter_id != m1.matter_id, "单键+不足额重叠不得命中（宁分勿合）"


# ── alias 卫生（拍板 D）＋ 种子（拍板 B）───────────────────────────────────


def test_l5_alias_seed_filters_poison_keeps_title():
    pipeline = AttributionPipeline(link_judge=None)
    d = pipeline.attribute(
        _fact(0, title=_VERB_VARIANT_TITLES[0], keywords=_SHARED_KEYWORDS,
              entities=["泰山啤酒", *_POISON_ENTITIES]),
        candidate_matters=[])
    assert d.is_new_matter
    assert _VERB_VARIANT_TITLES[0] in d.new_matter_aliases, "标题必须保留"
    assert "泰山啤酒" in d.new_matter_aliases
    for bad in _POISON_ENTITIES:
        assert bad not in d.new_matter_aliases, f"毒词入 alias：{bad!r}"


def test_new_matter_carries_topic_key_seed():
    """没有种子，同批下一条动词变体照样开第二个（B 的存在理由）。"""
    pipeline = AttributionPipeline(link_judge=None)
    d = pipeline.attribute(
        _fact(0, title=_VERB_VARIANT_TITLES[0], keywords=_SHARED_KEYWORDS),
        candidate_matters=[])
    m = pipeline.create_matter_for_decision(d)
    assert set(_SHARED_KEYWORDS) <= set(m.topic_keys)
    assert all(v == 1 for v in m.topic_keys.values())
