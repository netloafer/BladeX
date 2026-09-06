"""M5-2 判据③修复：裁决邻居源与判重对象拆分（2026-08-09 拍板：修法 2 + 两模式统一）。

根因（HANDOFF-20260809 §四）：`memory_index.py` 全量重建时
`novelty_checker = None if clear_db else ...` —— 对**判重**正确（库马上 clear），
但 `_neighbors_for` 的跨批分支复用同一个对象 → 裁决邻居被整段关掉。
同一对象、两个消费方、语义相反（与 2026-07-28 auxiliary 事故同型）。

第二个洞（本次读码新发现）：`_adjudicate` 循环①里候选一旦有邻居就转 pending，
而 pending 要到循环②才进 `running`、邻居全在循环①算完 ——
**任何有邻居的候选对它之后的所有候选整体不可见**。密集簇里第一条进 running
后其余全部转 pending 消失（「第19条」簇 [0]×[4] cosine 0.9637 互不可见的完整机制）。

这组剧本钉死：
  1. pending 候选对后续候选可见（密集簇不再变黑洞）；
  2. 全量重建形态（checker=None）下裁决邻居照常工作；
  3. 增量形态下内存集与库内视图合并、按 id 去重；
  4. 阈值先筛、后取 top-k（顺带发现 #3 的正确形状）；
  5. NOOP 丢弃的候选从邻居源摘除；悬空 target 不炸（apply_verdict 已容忍）；
  6. 判重路径零接触：novelty_checker 仍只服务判重。
"""

from __future__ import annotations

import math

from bladex_core.adjudication import AdjudicationOp, AdjudicationVerdict
from bladex_core.consolidation_proxy import InRunNeighborSource, ProxyConsolidator
from bladex_core.fact import Fact, ItemKind


def _vec(deg: float) -> list[float]:
    """单位圆上的二维向量：夹角即余弦可控（cos12° ≈ 0.978，cos60° = 0.5）。"""
    r = math.radians(deg)
    return [math.cos(r), math.sin(r)]


def _fact(fid: str, content: str, deg: float, *, subject: str = "",
          attribute: str = "") -> Fact:
    return Fact(id=fid, content=content, subject=subject, attribute=attribute,
                item_kind=ItemKind.ASSERTION, embedding=_vec(deg),
                scope="personal:u1")


class _CaptureAdjudicator:
    """裁决器替身：记下收到的输入包，按预设 verdict 回答（缺省全 ADD）。"""

    def __init__(self, verdicts: dict[str, AdjudicationVerdict] | None = None):
        self.packages: list = []
        self._verdicts = verdicts or {}

    def adjudicate(self, packages: list) -> list[AdjudicationVerdict]:
        self.packages.extend(packages)
        out = []
        for p in packages:
            out.append(self._verdicts.get(p.candidate_id) or AdjudicationVerdict(
                candidate_id=p.candidate_id, op=AdjudicationOp.ADD, reason="stub"))
        return out

    def package_for(self, candidate_id: str):
        for p in self.packages:
            if p.candidate_id == candidate_id:
                return p
        return None


def _consolidator(adjudicator, *, novelty_checker=None) -> ProxyConsolidator:
    return ProxyConsolidator(embedder=None, novelty_checker=novelty_checker,
                             adjudicator=adjudicator)


# ── 1+2. 密集簇：pending 候选对后续候选可见（全量重建形态 checker=None）──


def test_pending_candidates_are_visible_to_later_ones():
    """「第19条」簇的结构缩影：A 进 running 后，B/C 转 pending——

    修复前 D 的裁决包里只有 A（B/C 在循环②之前不在任何视野里）；
    修复后 D 必须看见 A、B、C 三条。checker=None = 全量重建形态。
    """
    adj = _CaptureAdjudicator()
    c = _consolidator(adj)
    facts = [
        _fact("fA", "该事项进行了两次招募", 0),
        _fact("fB", "该事项针对同一类对象进行了两次招募", 4),
        _fact("fC", "第二次招募面向另一类对象", 8),
        _fact("fD", "招募只进行过一次，此前把另一类招募误算成第二次", 12),
    ]

    kept = c._adjudicate(facts, [])

    pkg = adj.package_for("fD")
    assert pkg is not None, "fD 有邻居，必须进裁决"
    ids = {n.fact_id for n in pkg.neighbors}
    assert ids == {"fA", "fB", "fC"}, (
        f"密集簇必须互相可见（修复前这里只有 fA）：{ids}")
    # 全 ADD 时四条都入库
    assert {f.id for f in kept} == {"fA", "fB", "fC", "fD"}


def test_unrelated_candidate_still_goes_free():
    """远离簇的候选（cos60°=0.5 < 0.90）不该被拖进裁决——add_no_neighbor 照旧。"""
    adj = _CaptureAdjudicator()
    c = _consolidator(adj)
    kept = c._adjudicate([_fact("f1", "簇里的", 0), _fact("f2", "无关的", 60)], [])
    assert adj.package_for("f2") is None, "无邻居不花 LLM 钱"
    assert {f.id for f in kept} == {"f1", "f2"}


# ── 3. 增量形态：内存集 + 库内视图合并、按 id 去重 ──────────────────────


class _StubChecker:
    """库内视图替身：neighbors() 返回 LanceDB 行形状（dict + 相似度）。"""

    def __init__(self, rows: list[tuple[dict, float]]):
        self._rows = rows

    def neighbors(self, embedding, *, k=10, min_similarity=0.90):
        return [(r, s) for r, s in self._rows if s >= min_similarity][:k]


def test_incremental_merges_store_view_with_in_run_facts():
    checker = _StubChecker([
        ({"fact_id": "lib1", "content": "库里的旧说法", "subject": "", "attribute": ""},
         0.93),
    ])
    adj = _CaptureAdjudicator()
    c = _consolidator(adj, novelty_checker=checker)

    c._adjudicate([_fact("fA", "簇首", 0), _fact("fB", "簇二", 4),
                   _fact("fC", "簇三", 8)], [])

    pkg = adj.package_for("fC")
    assert pkg is not None
    ids = {n.fact_id for n in pkg.neighbors}
    assert "lib1" in ids, "库内视图必须还在（增量语义不回归）"
    assert "fB" in ids, "本轮 pending 也必须在（两模式统一）"


def test_store_view_dedups_by_id_against_in_run():
    """同一条既在内存集又被库返回 → 只出现一次（内存版优先，字段全）。"""
    src = InRunNeighborSource(checker=_StubChecker([
        ({"fact_id": "dup", "content": "库里的影子"}, 0.95),
    ]))
    src.add(_fact("dup", "内存里的本体", 2))
    out = src.neighbors(_vec(0), k=10, min_similarity=0.90)
    ids = [f.id for f, _ in out]
    assert ids.count("dup") == 1
    assert out[0][0].content == "内存里的本体"


# ── 4. 阈值先筛、后取 top-k（顺带发现 #3 的正确形状）────────────────────


def test_threshold_filters_before_topk():
    """12 条够格邻居、k=10 → 返回按相似度最高的 10 条，全部 ≥ 阈值。

    对照 LanceDB 侧的旧形状（limit(k) 先于阈值筛：密集簇里够格的邻居
    可能在筛之前就被 top-k 砍掉）——新对象不许重演。
    """
    src = InRunNeighborSource()
    for i in range(12):
        src.add(_fact(f"f{i}", f"第{i}条", i * 1.0))  # 0°..11°，与 0° 全部 ≥ cos11°≈0.982
    out = src.neighbors(_vec(0), k=10, min_similarity=0.90)
    assert len(out) == 10
    sims = [s for _, s in out]
    assert sims == sorted(sims, reverse=True)
    assert all(s >= 0.90 for s in sims)
    # 被挤掉的必须是最不像的两条（10°/11°），不是任意两条
    kept_ids = {f.id for f, _ in out}
    assert kept_ids == {f"f{i}" for i in range(10)}


def test_below_threshold_never_returned():
    src = InRunNeighborSource()
    src.add(_fact("far", "远处的", 60))  # cos60° = 0.5
    assert src.neighbors(_vec(0), k=10, min_similarity=0.90) == []


# ── 5. NOOP 摘除 + 悬空 target 容忍 ──────────────────────────────────────


def test_noop_dropped_fact_is_discarded_from_source():
    adj = _CaptureAdjudicator({
        "fB": AdjudicationVerdict(candidate_id="fB", op=AdjudicationOp.NOOP,
                                  target_fact_ids=["fA"], reason="restated"),
    })
    c = _consolidator(adj)
    kept = c._adjudicate([_fact("fA", "原话", 0), _fact("fB", "重述", 4)], [])

    assert {f.id for f in kept} == {"fA"}, "NOOP 的新条不入库"
    ids = [f.id for f, _ in c._neighbor_source.neighbors(_vec(4), k=10,
                                                         min_similarity=0.90)]
    assert "fB" not in ids, "被丢弃的候选不得作为幽灵邻居跨批存活"


def test_dangling_target_does_not_crash():
    """后见候选取代一条随后被 NOOP 丢弃的 pending 邻居 → 落盘侧安全跳过。

    这是文档化的悬空 target 边界：apply_verdict 按 by_id 查不到即跳过
    （目标本来就没进库），judgment 台账读侧需容忍悬空 id。
    """
    adj = _CaptureAdjudicator({
        "fB": AdjudicationVerdict(candidate_id="fB", op=AdjudicationOp.NOOP,
                                  target_fact_ids=["fA"], reason="restated"),
        "fC": AdjudicationVerdict(candidate_id="fC", op=AdjudicationOp.UPDATE,
                                  target_fact_ids=["fB"], reason="supersedes fB"),
    })
    c = _consolidator(adj)
    kept = c._adjudicate([_fact("fA", "原话", 0), _fact("fB", "重述", 4),
                          _fact("fC", "修正", 8)], [])

    assert {f.id for f in kept} == {"fA", "fC"}, "fC 降级 ADD 入库（宁多勿丢）"
    rec = next(r for r in c.adjudication_records if r["candidate_id"] == "fC")
    assert rec["targets"] == ["fB"] and rec["changed"] == [], (
        "台账保留悬空 target 供审计，但无旧条被改动")


# ── 6. 病根不复发：两个消费方各自的对象 ──────────────────────────────────


def test_full_mode_checker_none_still_has_neighbor_source():
    """全量重建形态：novelty_checker=None（判重语义）≠ 裁决没有邻居。"""
    c = _consolidator(_CaptureAdjudicator(), novelty_checker=None)
    assert c._novelty_checker is None, "判重语义不变：full 时仍 None"
    assert isinstance(c._neighbor_source, InRunNeighborSource)
    # 邻居源可用
    c._neighbor_source.add(_fact("x", "一条", 0))
    assert c._neighbor_source.neighbors(_vec(2), k=5, min_similarity=0.90)


def test_neighbors_for_does_not_touch_novelty_checker():
    """守卫：`_neighbors_for` 不得再摸 `_novelty_checker`（病根接线）。

    用一个只要被摸就炸的替身当 novelty_checker——邻居查询若走回旧接线，
    这条测试当场红。判重路径（is_novel）不在本调用链上，不受影响。
    """
    class _Tripwire:
        def __getattr__(self, name):
            raise AssertionError(
                f"_neighbors_for 摸了 novelty_checker.{name} —— 判据③病根复发")

    c = ProxyConsolidator(embedder=None, adjudicator=_CaptureAdjudicator(),
                          novelty_checker=None,
                          neighbor_source=InRunNeighborSource())
    c._novelty_checker = _Tripwire()  # 判重对象换成绊线
    out = c._neighbors_for(_fact("q", "查询", 0), [_fact("r", "批内", 4)])
    assert [f.id for f, _ in out] == ["r"], "批内视图照常，且没碰绊线"
