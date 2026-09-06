"""T4b（三段式卡，2026-08-10）：topic 接归属信号 + 注入③平面按 topic 分组。

钉住的边界（卡原文）：topic **只作为证据输入**——不改判定门槛、不改
provisional 转正规则、不改宁分勿合；分组只影响呈现顺序与标题行，
不改选取集合（选哪些条仍由检索/预算决定）。
"""

from __future__ import annotations

from bladex_core.attribution import (
    _ALIAS_CONFIDENCE,
    _DEFAULT_L3_ENTITY_INTERSECTION,
    AttributionPipeline,
    AttributionSource,
    LinkJudgeItem,
    LinkJudgeResult,
    LinkVerdict,
)
from bladex_core.fact import Fact, HardRule
from bladex_core.matter import Matter, MatterOrigin, MatterStatus


def _fact(fid: str, content: str, *, topic: str = "", entities=None,
          proposals=None) -> Fact:
    return Fact(id=fid, content=content, embedding=[1.0, 0.0], topic=topic,
                entities=entities or [], proposal_titles=proposals or [])


def _matter(mid: str, title: str, aliases: list[str]) -> Matter:
    return Matter(matter_id=mid, title=title, aliases=aliases,
                  status=MatterStatus.ACTIVE, origin=MatterOrigin.AUTO)


# ── L3：topic 进标题对照面 ────────────────────────────────────────────────


def test_l3_topic_matches_native_alias():
    """fact.topic 与 Matter 原生键全串相等 -> ALIAS link（新证据通道）。"""
    pipeline = AttributionPipeline()
    m = _matter("m-1", "泰山啤酒破产重整分析", ["泰山啤酒破产重整分析"])
    f = _fact("f-1", "方案C核心资产估值约4.5亿元",
              topic="泰山啤酒破产重整分析",
              proposals=["别的提案标题"])       # proposal 不命中，靠 topic
    d = pipeline.attribute(f, [m])
    assert d.source == AttributionSource.ALIAS
    assert d.matter_id == "m-1"
    assert "title_alias_match" in d.signal


def test_l3_without_topic_behavior_unchanged():
    """同一 fact 去掉 topic -> 不命中 L3（回归钉死：topic 是增量证据）。"""
    pipeline = AttributionPipeline()
    m = _matter("m-1", "泰山啤酒破产重整分析", ["泰山啤酒破产重整分析"])
    f = _fact("f-1", "方案C核心资产估值约4.5亿元", proposals=["别的提案标题"])
    d = pipeline.attribute(f, [m])
    assert d.source != AttributionSource.ALIAS


def test_l3_topic_partial_overlap_does_not_link():
    """宁分勿合：topic 与原生键只是子串相含（非全串相等）-> L3 不命中。"""
    pipeline = AttributionPipeline()
    m = _matter("m-1", "泰山啤酒破产重整分析", ["泰山啤酒破产重整分析"])
    f = _fact("f-1", "内容", topic="泰山啤酒")   # 子串，非全串
    d = pipeline.attribute(f, [m])
    assert d.source != AttributionSource.ALIAS


def test_thresholds_untouched():
    """门槛零改动断言：置信度 / 实体交集门槛 / 转正规则常量不动。"""
    assert _ALIAS_CONFIDENCE == 0.9
    assert _DEFAULT_L3_ENTITY_INTERSECTION == 2


# ── L4：裁决输入带 topic 行 ──────────────────────────────────────────────


class _CapturingJudge:
    model_name = "mock-judge"

    def __init__(self) -> None:
        self.last_items: list[LinkJudgeItem] = []

    def judge_links(self, items):
        self.last_items = list(items)
        return [LinkJudgeResult(fact_id=it.fact_id, verdict=LinkVerdict.UNCERTAIN)
                for it in items]


def test_l4_item_carries_topic():
    """管线装配的 LinkJudgeItem 带 fact.topic（L1-L3 未命中才进 L4）。"""
    judge = _CapturingJudge()
    pipeline = AttributionPipeline(link_judge=judge)
    m = _matter("m-x", "完全无关的事", ["完全无关的事"])
    f = _fact("f-t", "内容", topic="泰山啤酒破产重整", proposals=["提案"],
              entities=["方案C"])
    pipeline.attribute(f, [m])
    assert judge.last_items and judge.last_items[0].topic == "泰山啤酒破产重整"


def test_l4_prompt_renders_topic_line():
    """proxy 侧 prompt 装配：item 有 topic -> 'topic:' 一行；无 -> 不出现。"""
    from bladex_proxy.linking import _build_link_prompt

    with_topic = LinkJudgeItem(fact_id="f-1", content="c", topic="泰山啤酒破产重整")
    without = LinkJudgeItem(fact_id="f-2", content="c")
    text = _build_link_prompt([with_topic, without])
    assert "topic: 泰山啤酒破产重整" in text
    assert text.count("topic:") == 1


def test_l4_system_prompt_untouched():
    """_LINK_SYSTEM 与 LINK_PROMPT_VER 不动（只加证据行，不改裁决语义）。"""
    from bladex_proxy import linking

    assert linking.LINK_PROMPT_VER == "v1-link-001"
    assert "uncertain" in linking._LINK_SYSTEM   # 宁分勿合条款仍在


# ── 注入③平面：topic 分组渲染（呈现层）──────────────────────────────────


class _Retriever:
    """最小 FactRetriever：返回固定命中集。"""

    def __init__(self, facts: list[Fact]) -> None:
        self._facts = facts

    def search(self, query, top_k=10, user_id=None, visibility=None):
        return list(self._facts[:top_k])

    def list_hard_rules(self) -> list[HardRule]:
        return []


