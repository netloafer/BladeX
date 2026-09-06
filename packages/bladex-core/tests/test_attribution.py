"""五层链接归属管线单元测试（ADR-0018 §3.2，DPL）。

L1 manual / L2 explicit / L3 alias-link / L4 LLM-link(批量+台账缓存) / L5 provisional|unassigned。
cosine 阈值判定已退役（ADR-0018 §2）-- e5 只剩 L4 召回 top-k。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from bladex_core.attribution import (
    UNASSIGNED_MATTER_ID,
    AttributionDecision,
    AttributionPipeline,
    AttributionSource,
    ExplicitSignalDetector,
    LinkJudgeItem,
    LinkJudgeResult,
    LinkVerdict,
    _deterministic_matter_id,
    _normalize,
)
from bladex_core.fact import Fact
from bladex_core.matter import (
    EdgeProvenance,
    Matter,
    MatterOrigin,
    MatterStatus,
)


def _make_fact(
    content: str = "some content",
    *,
    fact_id: str = "f-001",
    embedding: list[float] | None = None,
    entities: list[str] | None = None,
    proposal_titles: list[str] | None = None,
) -> Fact:
    return Fact(
        id=fact_id, content=content, embedding=embedding,
        entities=entities or [], proposal_titles=proposal_titles or [],
    )


def _make_matter(
    matter_id: str = "m-001",
    *,
    title: str = "test matter",
    aliases: list[str] | None = None,
    entities: list[str] | None = None,
    status: MatterStatus = MatterStatus.ACTIVE,
    centroid: list[float] | None = None,
    updated_at: datetime | None = None,
) -> Matter:
    m = Matter(
        matter_id=matter_id, title=title,
        aliases=aliases or ([title] if title else []),
        entities=entities or [],
        status=status, centroid=centroid, origin=MatterOrigin.AUTO,
    )
    if updated_at is not None:
        m.updated_at = updated_at
    return m


class MockLinkJudge:
    """mock LinkJudge：按预设 fact_id -> verdict 返回。"""

    def __init__(self, verdicts: dict[str, LinkJudgeResult], model: str = "mock-judge") -> None:
        self._verdicts = verdicts
        self._model = model
        self.calls = 0
        self.last_items: list[LinkJudgeItem] = []

    @property
    def model_name(self) -> str:
        return self._model

    def judge_links(self, items: list[LinkJudgeItem]) -> list[LinkJudgeResult]:
        self.calls += 1
        self.last_items = list(items)
        return [
            self._verdicts.get(it.fact_id,
                               LinkJudgeResult(fact_id=it.fact_id, verdict=LinkVerdict.UNCERTAIN))
            for it in items
        ]


class MockJudgmentJournal:
    """mock JudgmentLedger：内存存储，可预设命中。"""

    def __init__(self, cached: dict[str, LinkJudgeResult] | None = None) -> None:
        self._cached = dict(cached) if cached else {}
        self.store: dict[str, list[tuple[list[dict], LinkJudgeResult]]] = {}
        self.get_calls = 0
        self.put_calls = 0

    def get_latest_judgment(self, fact_id: str) -> LinkJudgeResult | None:
        self.get_calls += 1
        return self._cached.get(fact_id)

    def put_judgment(self, fact_id: str, candidates: list[dict], result: LinkJudgeResult) -> None:
        self.put_calls += 1
        self.store.setdefault(fact_id, []).append((candidates, result))


# ── L1 manual ──


def test_l1_manual_overrides_everything():
    """L1: 手动指定最高优先，压过 L2-L4。"""
    judge = MockLinkJudge({})
    pipeline = AttributionPipeline(link_judge=judge)

    fact = _make_fact("content", embedding=[1.0, 0.0], proposal_titles=["anything"])
    cands = [_make_matter("m-sem", aliases=["anything"])]

    d = pipeline.attribute(fact, cands, manual_matter_id="m-manual")
    assert d.source == AttributionSource.MANUAL
    assert d.matter_id == "m-manual"
    assert d.confidence == 1.0
    assert judge.calls == 0  # manual 不进 L4


# ── L2 explicit ──


def test_l2_explicit_title_mention():
    """L2: 直接点名 Matter 标题 -> EXPLICIT。"""
    pipeline = AttributionPipeline()
    matter = _make_matter("m-msg", title="给 BladeX 加 /v1/messages 支持")
    fact = _make_fact("unrelated content", embedding=[1.0, 0.0], fact_id="f-tm")
    d = pipeline.attribute(fact, [matter], user_message="关于 给 BladeX 加 /v1/messages 支持 这个事")
    assert d.source == AttributionSource.EXPLICIT
    assert d.matter_id == "m-msg"
    assert "title_mention" in d.signal


def test_l2_explicit_continuation():
    """L2: 续接语 + 标题关键词 -> EXPLICIT。"""
    pipeline = AttributionPipeline()
    matter = _make_matter("m-bug", title="修复 P3 key 覆盖 bug")
    fact = _make_fact("content", embedding=[1.0, 0.0], fact_id="f-exp")
    d = pipeline.attribute(fact, [matter], user_message="接着修复 P3 key 覆盖 bug 的问题")
    assert d.source == AttributionSource.EXPLICIT
    assert d.matter_id == "m-bug"


def test_l2_explicit_does_not_merge_unrelated():
    """守宁分勿合：续接语指向 A，不误归 B。"""
    pipeline = AttributionPipeline()
    a = _make_matter("m-a", title="给 BladeX 加 /v1/messages 支持")
    b = _make_matter("m-b", title="修复 P3 key 覆盖 bug")
    fact = _make_fact("content", embedding=[0.9, 0.1], fact_id="f-nm")
    d = pipeline.attribute(fact, [a, b], user_message="继续做 v1/messages 的支持")
    assert d.source == AttributionSource.EXPLICIT
    assert d.matter_id == "m-a"
    assert d.matter_id != "m-b"


def test_l2_continuation_no_cross_skill_via_polluted_entities():
    """ADR-0020 后续：continuation 不用 Matter entities 判相容，防污染正反馈放大跨技能。

    m-41bdcd982d05 案例：Matter entities 累积多技能实体（local-llm/dexter/不良资产），
    conclusion fact entities 与污染 entities 交集 -> 旧逻辑放行跨技能链入。
    修复后 continuation 只用 title/aliases，fact 与 title/aliases 无交集 -> 阻断。
    """
    pipeline = AttributionPipeline()
    matter = _make_matter(
        "m-polluted", title="更新 research-report-generation 技能规则",
        aliases=["更新 research-report-generation 技能规则", "research-report-generation"],
        entities=["local-llm-inference-macos", "dexter", "不良资产尽调", "SKILL.md",
                  "MiroThinker 1.7-mini", "deep-research"],
    )
    fact = _make_fact(
        "更新了 local-llm-inference-macos/SKILL.md",
        embedding=[1.0, 0.0], fact_id="f-cross",
        entities=["local-llm-inference-macos", "SKILL.md", "MiroThinker 1.7-mini"],
        proposal_titles=["local-llm 技能更新"],
    )
    d = pipeline.attribute(fact, [matter], user_message="继续更新技能规则")
    # continuation 不应靠污染 entities 跨技能链入（修复后只用 title/aliases，无交集->阻断）
    assert d.source != AttributionSource.EXPLICIT or d.matter_id != "m-polluted"


def test_l2_keyword_vs_filtered():
    """A: 'vs' 不再当关键词（真机回溯：致阿根廷 Matter 吞跨主题 fact）。"""
    det = ExplicitSignalDetector()
    kws = det._extract_keywords("阿根廷 vs 佛得角比赛赔率查询与预测")
    assert "vs" not in kws
    assert "阿根廷" in kws
    # 纯 ASCII 短片段（P3/v1）也过滤；CJK 2 字（修复）保留
    kws2 = det._extract_keywords("修复 P3 key 覆盖 bug")
    assert "P3" not in kws2
    assert "修复" in kws2


def test_l2_keyword_ascii_word_boundary():
    """A: ASCII 关键词词边界匹配，'Open' 不命中 'opencode'。"""
    det = ExplicitSignalDetector()
    assert det._kw_in_msg("Open", "我在用opencode写代码") is False
    assert det._kw_in_msg("Open", "继续看 open claw 对比") is True


def test_l2_gate_blocks_cross_topic_fact():
    """B 门控：turn 续接阿根廷，但 fact 是 DNS 主题 -> L2 不误链到阿根廷 Matter。

    真机回溯核心回归：一轮多主题时，turn 命中 X 不能把该轮无关 fact 链给 X。
    fact 有主题信号（entities/proposal_titles）且与阿根廷无重叠 -> 门控阻断，
    落到 L3/L5（不进 L2 EXPLICIT）。
    """
    pipeline = AttributionPipeline()
    argentina = _make_matter(
        "m-arg", title="阿根廷 vs 佛得角比赛赔率查询与预测",
        aliases=["阿根廷", "佛得角"],
    )
    fact = _make_fact(
        "用户设置了1.1.1.2的DNS服务器", embedding=[0.1, 0.9], fact_id="f-dns",
        entities=["1.1.1.2", "DNS"], proposal_titles=["设置DNS服务器"],
    )
    # turn 同时提了阿根廷 + 续接语
    d = pipeline.attribute(
        fact, [argentina],
        user_message="继续看阿根廷那场赔率，另外我把DNS设成1.1.1.2了",
    )
    assert d.source != AttributionSource.EXPLICIT
    assert d.matter_id != "m-arg"


def test_l2_gate_allows_relevant_fact():
    """B 门控：fact 主题与 Matter 重叠时，turn 续接信号正常链（门控放行）。"""
    pipeline = AttributionPipeline()
    argentina = _make_matter(
        "m-arg", title="阿根廷 vs 佛得角比赛赔率查询与预测",
        aliases=["阿根廷", "佛得角"],
    )
    fact = _make_fact(
        "用户询问阿根廷 vs 佛得角的市场赔率", embedding=[0.9, 0.1], fact_id="f-arg",
        entities=["阿根廷", "佛得角"], proposal_titles=["查询阿根廷佛得角赔率"],
    )
    d = pipeline.attribute(fact, [argentina], user_message="继续看阿根廷那场赔率")
    assert d.source == AttributionSource.EXPLICIT
    assert d.matter_id == "m-arg"


# ── L3 alias-link ──


def test_l3_alias_title_match():
    """L3: proposal 标题规范化 == Matter alias -> ALIAS link。"""
    pipeline = AttributionPipeline()
    matter = _make_matter("m-1", title="嵌入模型选型", aliases=["嵌入模型选型", "e5"])
    # fact 的 proposal 标题与 alias 规范化相等（大小写/空白差异）
    fact = _make_fact(
        "用户讨论 嵌入 模型 选型", embedding=[1.0, 0.0], fact_id="f-alias",
        proposal_titles=["  嵌入模型选型  "],
    )
    d = pipeline.attribute(fact, [matter])
    assert d.source == AttributionSource.ALIAS
    assert d.matter_id == "m-1"
    assert d.confidence == 0.9


def test_l3_alias_entity_intersection():
    """L3: entities ∩ 原生键(title+aliases) >= 门槛 -> ALIAS link。

    MS-16：对照面 = 创建时 aliases（含首批成员实体，
    `new_matter_aliases = [title]+entities`），不再是累积 `matter.entities`。
    """
    pipeline = AttributionPipeline(l3_entity_intersection=2)
    matter = _make_matter("m-1", title="嵌入选型",
                          aliases=["嵌入选型", "e5", "fastembed", "proxy"])
    fact = _make_fact(
        "讨论 e5 和 fastembed", embedding=[1.0, 0.0], fact_id="f-ent",
        entities=["e5", "fastembed", "router"],
        proposal_titles=["别的提案"],  # 标题不匹配，靠实体交集
    )
    d = pipeline.attribute(fact, [matter])
    assert d.source == AttributionSource.ALIAS
    assert d.matter_id == "m-1"


def test_l3_accumulated_entities_no_longer_match():
    """MS-16 雪球回归（T4a-2）：累积 matter.entities 不再是 L3 匹配面。

    永安←泰山形态：外题实体（泰山啤酒/破产重整）被一条错边并进
    matter.entities 后，旧逻辑让共享这些实体的所有后续 fact 以 conf=0.9
    持续 L3 链入。修后：entities 交集再大也不命中，落 L4/L5。
    """
    pipeline = AttributionPipeline(l3_entity_intersection=2)
    yongan = _make_matter(
        "m-yongan", title="永安林业近三年一期经营情况分析",
        aliases=["永安林业近三年一期经营情况分析", "永安林业", "000663.SZ"],
        entities=["永安林业", "000663.SZ", "泰山啤酒", "破产重整", "方案C"],  # 已被污染的累积并集
    )
    taishan_fact = _make_fact(
        "泰山啤酒方案C核心资产合计估值约4.5亿元", embedding=[0.1, 0.9],
        fact_id="f-taishan", entities=["泰山啤酒", "方案C", "破产重整"],
        proposal_titles=["泰山啤酒破产重整投资分析"],
    )
    d = pipeline.attribute(taishan_fact, [yongan])
    assert d.source != AttributionSource.ALIAS
    assert d.matter_id != "m-yongan"


def test_l3_skips_closed_and_unassigned():
    """L3: closed Matter 与未归属池不参与链接。"""
    pipeline = AttributionPipeline()
    closed = _make_matter("m-closed", title="closed topic", aliases=["closed topic"],
                          status=MatterStatus.CLOSED)
    fact = _make_fact("closed topic content", embedding=[1.0, 0.0],
                      proposal_titles=["closed topic"])
    # 只有 closed 候选 -> L3 不命中 -> 无 judge -> L5（无 proposal? 有 proposal -> provisional）
    d = pipeline.attribute(fact, [closed])
    assert d.source == AttributionSource.PROVISIONAL  # closed 被排除，走 L5 新开


# ── L4 LLM-link ──


def test_l4_link_verdict():
    """L4: judge 裁决 link:<id> -> LLM_LINK。"""
    judge = MockLinkJudge({
        "f-l4": LinkJudgeResult(fact_id="f-l4", verdict=LinkVerdict.LINK,
                                matter_id="m-1", summary_rewrite="新摘要"),
    })
    journal = MockJudgmentJournal()
    pipeline = AttributionPipeline(link_judge=judge, judgment_journal=journal)

    # matter aliases 与 fact proposal 不重合 -> L3 不命中 -> 进 L4
    matter = _make_matter("m-1", title="topic", aliases=["topic"])
    fact = _make_fact("content about topic", embedding=[1.0, 0.0], fact_id="f-l4",
                      proposal_titles=["unrelated-proposal"])
    d = pipeline.attribute(fact, [matter])
    assert d.source == AttributionSource.LLM_LINK
    assert d.matter_id == "m-1"
    assert d.summary_rewrite == "新摘要"
    assert judge.calls == 1
    assert journal.put_calls == 1  # miss 后写台账


def test_l4_ledger_hit_skips_llm():
    """L4: 台账命中 -> 0 LLM 调用。"""
    cached = LinkJudgeResult(fact_id="f-c", verdict=LinkVerdict.LINK, matter_id="m-1")
    judge = MockLinkJudge({})
    journal = MockJudgmentJournal(cached={"f-c": cached})
    pipeline = AttributionPipeline(link_judge=judge, judgment_journal=journal)

    matter = _make_matter("m-1", title="topic", aliases=["topic"])
    fact = _make_fact("content", embedding=[1.0, 0.0], fact_id="f-c",
                      proposal_titles=["unrelated-proposal"])
    d = pipeline.attribute(fact, [matter])
    assert d.source == AttributionSource.LLM_LINK
    assert d.matter_id == "m-1"
    assert judge.calls == 0  # 台账命中，未调 LLM
    assert journal.put_calls == 0


def test_l4_link_id_not_in_candidates_degrades_to_uncertain():
    """守宁分勿合：judge 幻觉不存在的 matter_id -> uncertain -> 留池。"""
    judge = MockLinkJudge({
        "f-h": LinkJudgeResult(fact_id="f-h", verdict=LinkVerdict.LINK, matter_id="m-ghost"),
    })
    pipeline = AttributionPipeline(link_judge=judge, judgment_journal=MockJudgmentJournal())
    matter = _make_matter("m-real", title="topic", aliases=["topic"])
    fact = _make_fact("content", embedding=[1.0, 0.0], fact_id="f-h",
                      proposal_titles=["unrelated-proposal"])
    d = pipeline.attribute(fact, [matter])
    # m-ghost 不在候选 -> uncertain -> 留池 pending
    assert d.source == AttributionSource.UNASSIGNED
    assert d.pending_judgment is True


def test_l4_judge_unavailable_all_uncertain():
    """降级（ADR-0018 §3.7）：无 judge -> L4 全 uncertain -> 留池 pending。"""
    pipeline = AttributionPipeline()  # 无 link_judge
    matter = _make_matter("m-1", title="topic", aliases=["topic"])
    fact = _make_fact("content", embedding=[1.0, 0.0], fact_id="f-u",
                      proposal_titles=["unrelated-proposal"])
    d = pipeline.attribute(fact, [matter])
    # 有候选但无 judge -> _run_l4_batch 返回 uncertain -> L5 uncertain -> 留池 pending
    assert d.source == AttributionSource.UNASSIGNED
    assert d.pending_judgment is True


def test_l4_batch_single_call_for_multiple_facts():
    """L4 批量：多条 fact 一次 LLM 调用（ADR-0018 §3.2 可批量）。"""
    judge = MockLinkJudge({
        "f-1": LinkJudgeResult(fact_id="f-1", verdict=LinkVerdict.LINK, matter_id="m-a"),
        "f-2": LinkJudgeResult(fact_id="f-2", verdict=LinkVerdict.LINK, matter_id="m-b"),
    })
    pipeline = AttributionPipeline(link_judge=judge, judgment_journal=MockJudgmentJournal())
    # aliases 与 proposals 不重合 -> 两条都进 L4
    ma = _make_matter("m-a", title="alpha", aliases=["alpha"])
    mb = _make_matter("m-b", title="beta", aliases=["beta"])
    facts = [
        _make_fact("alpha content", embedding=[1.0, 0.0], fact_id="f-1",
                   proposal_titles=["alpha-thing"]),
        _make_fact("beta content", embedding=[0.0, 1.0], fact_id="f-2",
                   proposal_titles=["beta-thing"]),
    ]
    decisions = pipeline.attribute_batch(
        facts, candidates_by_fact={"f-1": [ma], "f-2": [mb]},
    )
    assert judge.calls == 1  # 一次调用裁决两条
    assert decisions[0].matter_id == "m-a"
    assert decisions[1].matter_id == "m-b"


# ── L5 provisional / unassigned ──


def test_l5_provisional_from_proposal():
    """L5: verdict none + 有 proposal -> 新开 provisional Matter（确定性 id）。"""
    judge = MockLinkJudge({
        "f-n": LinkJudgeResult(fact_id="f-n", verdict=LinkVerdict.NONE),
    })
    pipeline = AttributionPipeline(link_judge=judge, judgment_journal=MockJudgmentJournal())
    # 2026-08-14 拍板 D：alias 种子过毒词——'x'/'y' 这类短碎片不再入 alias
    # （matter-dedup-keywords 卡；合法实体照常保留）。
    fact = _make_fact("全新话题", embedding=[1.0, 0.0], fact_id="f-n",
                      entities=["x", "全新话题实体"], proposal_titles=["全新话题提案"])
    d = pipeline.attribute(fact, [])  # 无候选 -> 直接 L5
    assert d.source == AttributionSource.PROVISIONAL
    assert d.is_new_matter is True
    assert d.new_matter_title == "全新话题提案"
    assert "全新话题提案" in d.new_matter_aliases
    assert "全新话题实体" in d.new_matter_aliases
    assert "x" not in d.new_matter_aliases, "短碎片实体不得入 alias（拍板 D）"
    # 确定性 matter_id
    assert d.matter_id == _deterministic_matter_id("全新话题提案")


def test_l5_no_proposal_goes_unassigned():
    """L5: 无 proposal -> 未归属池（不新开）。"""
    pipeline = AttributionPipeline()
    fact = _make_fact("orphan content", embedding=[1.0, 0.0], fact_id="f-orphan")
    d = pipeline.attribute(fact, [])
    assert d.source == AttributionSource.UNASSIGNED
    assert d.matter_id == UNASSIGNED_MATTER_ID


def test_l5_uncertain_goes_pending():
    """L5: uncertain -> 未归属池 + pending_judgment（T9 重判用）。"""
    judge = MockLinkJudge({
        "f-unc": LinkJudgeResult(fact_id="f-unc", verdict=LinkVerdict.UNCERTAIN),
    })
    pipeline = AttributionPipeline(link_judge=judge, judgment_journal=MockJudgmentJournal())
    matter = _make_matter("m-1", title="topic", aliases=["topic"])
    fact = _make_fact("content", embedding=[1.0, 0.0], fact_id="f-unc",
                      proposal_titles=["unrelated-proposal"])
    d = pipeline.attribute(fact, [matter])
    assert d.source == AttributionSource.UNASSIGNED
    assert d.pending_judgment is True


def test_l5_deterministic_id_merges_same_proposal():
    """同一 proposal 的两条 fact -> 同一确定性 matter_id（合并）。"""
    pipeline = AttributionPipeline()
    f1 = _make_fact("a", embedding=[1.0, 0.0], fact_id="f-1", proposal_titles=["共享提案"])
    f2 = _make_fact("b", embedding=[0.0, 1.0], fact_id="f-2", proposal_titles=["共享提案"])
    d1 = pipeline.attribute(f1, [])
    d2 = pipeline.attribute(f2, [])
    assert d1.source == AttributionSource.PROVISIONAL
    assert d1.matter_id == d2.matter_id  # 同一确定性 id -> 合并


# ── Matter / Edge 构造 ──


def test_create_matter_is_provisional_with_aliases():
    """create_matter_for_decision: provisional 状态 + aliases + 确定性 id。"""
    pipeline = AttributionPipeline()
    dec = AttributionDecision(
        fact_id="f", matter_id="", confidence=0.0,
        source=AttributionSource.PROVISIONAL, is_new_matter=True,
        new_matter_title="提案标题", new_matter_aliases=["提案标题", "e5"],
    )
    matter = pipeline.create_matter_for_decision(dec, centroid=[1.0, 0.0])
    assert matter.status == MatterStatus.PROVISIONAL
    assert matter.origin == MatterOrigin.AUTO
    assert matter.title == "提案标题"
    assert matter.aliases == ["提案标题", "e5"]
    assert matter.centroid == [1.0, 0.0]
    assert matter.matter_id == _deterministic_matter_id("提案标题")


def test_create_edge_carries_decision_record():
    """create_edge_for_decision: 边带 decision 记录（ADR-0018 §3.5）。"""
    pipeline = AttributionPipeline()
    fact = _make_fact("c", fact_id="f-dec")
    dec = AttributionDecision(
        fact_id="f-dec", matter_id="m-1", confidence=0.8,
        source=AttributionSource.LLM_LINK,
        decision={"layer": "L4", "verdict": "link:m-1"},
    )
    edge = pipeline.create_edge_for_decision(dec, fact)
    assert edge.matter_id == "m-1"
    assert edge.target_key == "f-dec"
    assert edge.provenance == EdgeProvenance.AUTO
    assert edge.confidence == 0.8
    assert edge.decision.get("layer") == "L4"

    # MANUAL 决策 -> MANUAL provenance
    dec_man = AttributionDecision(
        fact_id="f-dec", matter_id="m-man", confidence=1.0,
        source=AttributionSource.MANUAL, decision={"layer": "L1"},
    )
    edge_man = pipeline.create_edge_for_decision(dec_man, fact)
    assert edge_man.provenance == EdgeProvenance.MANUAL


# ── 生命周期 ──


def test_lifecycle_dormant_downgrade():
    """active 超时 -> dormant。"""
    pipeline = AttributionPipeline(dormant_timeout_s=3600)
    matter = _make_matter("m-1", status=MatterStatus.ACTIVE,
                          updated_at=datetime.now(UTC) - timedelta(hours=2))
    changes = pipeline.update_lifecycles([matter])
    assert len(changes) == 1
    assert changes[0][2] == MatterStatus.DORMANT


def test_lifecycle_provisional_not_downgraded():
    """provisional 不自动降级（等成员，ADR-0018 §3.3）。"""
    pipeline = AttributionPipeline(dormant_timeout_s=1)
    matter = _make_matter("m-1", status=MatterStatus.PROVISIONAL,
                          updated_at=datetime.now(UTC) - timedelta(days=30))
    changes = pipeline.update_lifecycles([matter])
    assert len(changes) == 0
    assert matter.status == MatterStatus.PROVISIONAL


def test_reactivate_dormant_on_hit():
    """dormant 被命中 -> active。"""
    pipeline = AttributionPipeline()
    matter = _make_matter("m-1", status=MatterStatus.DORMANT)
    assert pipeline.reactivate_on_hit(matter) is True
    assert matter.status == MatterStatus.ACTIVE
    assert pipeline.reactivate_on_hit(matter) is False  # 已 active


# ── 规范化工具 ──


def test_normalize_fullwidth_and_case():
    assert _normalize("ＡＢＣ　Ｔｅｓｔ") == "abc test"
    assert _normalize("  Hello   World  ") == "hello world"
