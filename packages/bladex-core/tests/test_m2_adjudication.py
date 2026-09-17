"""M2：写入时裁决的验收剧本（复核第 2 层 · 附录 C 机制 2）。

核心剧本来自复核 3.9 的**真实裁决输入包**（live 库实测，无一虚构）：
一条自带修正宣告的新事实 + 5 条近邻，其中两条持相反主张。
现状下这五条以现行事实身份共存，检测器只吐两个巧合告警然后被丢弃。

X6 纪律：这些词汇不得出现在实现代码里。裁决器只认结构
（候选 / 邻居 / corrects 槽），不认任何领域词。
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from bladex_core.adjudication import (
    AdjudicationOp,
    AdjudicationVerdict,
    apply_verdict,
    build_input,
    fast_path_verdict,
)
from bladex_core.fact import Fact, ItemKind


def _fact(fid: str, content: str, *, subject: str = "", attribute: str = "",
          kind: ItemKind = ItemKind.ASSERTION, emb: list[float] | None = None,
          entities: list[str] | None = None) -> Fact:
    return Fact(id=fid, content=content, subject=subject, attribute=attribute,
                item_kind=kind, embedding=emb, entities=entities or [],
                scope="personal:u1")


# ── 三种 verdict 的应用语义（**没有 DELETE**）────────────────────────────


def test_update_is_non_destructive():
    """UPDATE = 旧条置 t_invalid + superseded_by，**旧条保留**。

    删除会毁掉重建等价性（G6），而且"哪条被取代了"本身就是有价值的记忆。
    """
    now = datetime.now(UTC)
    old = _fact("f_old", "旧的说法")
    new = _fact("f_new", "修正后的说法")
    v = AdjudicationVerdict(candidate_id="f_new", op=AdjudicationOp.UPDATE,
                            target_fact_ids=["f_old"])

    keep, changed = apply_verdict(v, new, {"f_old": old}, now=now)

    assert keep is True, "新条要入库"
    assert changed == [old]
    assert old.t_invalid == now and old.superseded_by == "f_new"
    assert old.content == "旧的说法", "旧条内容不许被改写——as-of 查询要看得到它"
    assert old.updated_at == now, "被取代是最要紧的一种变更，导出增量必须看得见"


def test_noop_strengthens_and_drops_the_new_one():
    """NOOP（MS-7）：同义重述 → 旧条 strength+1，新条丢弃、库条数不变。

    这恢复了"重复陈述 = 强化"的信号链（原 merge 机制的唯一遗产），
    也是 importance 公式里 strength 项唯一的生产者（E5：它此前永远是 1）。
    """
    now = datetime.now(UTC)
    old = _fact("f_old", "同一件事")
    assert old.strength == 1
    v = AdjudicationVerdict(candidate_id="f_new", op=AdjudicationOp.NOOP,
                            target_fact_ids=["f_old"])

    keep, changed = apply_verdict(v, _fact("f_new", "同一件事换个说法"),
                                  {"f_old": old}, now=now)

    assert keep is False, "新条不入库"
    assert old.strength == 2 and old.updated_at == now
    assert old.t_invalid is None, "强化不是失效"


def test_noop_does_not_touch_the_relevance_clock():
    """🔴 MS-7：NOOP **不动 `last_hit_at`** —— 重复陈述是"强化"，不是"命中"。

    相关钟只该被检索命中推动。掺进写入侧信号会让三时钟的判据失真：
    一条从没被检索到、但用户反复提起的记忆会看起来"很有相关性"。
    """
    now = datetime.now(UTC)
    old = _fact("f_old", "某条事实")
    old.last_hit_at = None
    v = AdjudicationVerdict(candidate_id="f_new", op=AdjudicationOp.NOOP,
                            target_fact_ids=["f_old"])

    apply_verdict(v, _fact("f_new", "某条事实的重述"), {"f_old": old}, now=now)

    assert old.last_hit_at is None, "写入侧不许推动相关钟"


def test_add_changes_nothing_else():
    now = datetime.now(UTC)
    old = _fact("f_old", "别的事")
    keep, changed = apply_verdict(
        AdjudicationVerdict(candidate_id="f_new", op=AdjudicationOp.ADD),
        _fact("f_new", "新的事"), {"f_old": old}, now=now)
    assert keep is True and changed == []
    assert old.t_invalid is None and old.strength == 1


def test_missing_target_degrades_to_add():
    """目标已被别的裁决取代 / 不在库里 → 降级 ADD（宁多勿丢）。

    反过来（丢弃新条）会**静默丢记忆**，那是不可接受的一侧。
    """
    now = datetime.now(UTC)
    v = AdjudicationVerdict(candidate_id="f_new", op=AdjudicationOp.UPDATE,
                            target_fact_ids=["gone"])
    keep, changed = apply_verdict(v, _fact("f_new", "新的事"), {}, now=now)
    assert keep is True and changed == []


def test_update_skips_already_invalidated_targets():
    """已经失效的旧条不重复取代（否则 superseded_by 会被后来者改写）。"""
    now = datetime.now(UTC)
    old = _fact("f_old", "更早就被取代了")
    old.t_invalid = now
    old.superseded_by = "f_mid"
    v = AdjudicationVerdict(candidate_id="f_new", op=AdjudicationOp.UPDATE,
                            target_fact_ids=["f_old"])
    apply_verdict(v, _fact("f_new", "又一条"), {"f_old": old}, now=now)
    assert old.superseded_by == "f_mid", "取代链不许被后来者改写"


# ── 快路径：精确取代键命中，零 LLM ────────────────────────────────────────


def test_fast_path_same_key_new_value_is_update():
    cand = _fact("f2", "值乙", subject="某主题", attribute="某槽位")
    old = _fact("f1", "值甲", subject="某主题", attribute="某槽位")
    v = fast_path_verdict(cand, [old])
    assert v is not None and v.op is AdjudicationOp.UPDATE
    assert v.target_fact_ids == ["f1"] and v.reason.startswith("fast_path")


def test_fast_path_same_key_restated_is_noop():
    cand = _fact("f2", "值甲", subject="某主题", attribute="某槽位")
    old = _fact("f1", "值甲", subject="某主题", attribute="某槽位")
    v = fast_path_verdict(cand, [old])
    assert v is not None and v.op is AdjudicationOp.NOOP


def test_fast_path_returns_none_without_exact_key():
    """没有精确键就别猜——交给裁决器看文本。

    E3 已经说明这条路在真实蒸馏产出上区分度 ≈ 0（subject/attribute 很少精确对齐），
    命中率低是预期的；它的价值是"命中时 100% 正确且零成本"。
    """
    cand = _fact("f2", "内容乙", subject="", attribute="")
    assert fast_path_verdict(cand, [_fact("f1", "内容甲")]) is None


# ── 裁决包装配 ──────────────────────────────────────────────────────────


def test_build_input_keeps_top_k_by_similarity():
    cand = _fact("c", "候选")
    neighbors = [(_fact(f"n{i}", f"邻居{i}"), 0.90 + i * 0.001) for i in range(15)]
    pkg = build_input(cand, neighbors, top_k=10)
    assert len(pkg.neighbors) == 10
    sims = [n.similarity for n in pkg.neighbors]
    assert sims == sorted(sims, reverse=True), "按相似度降序，最像的排前面"


def test_build_input_carries_the_correction_note():
    """🔴 `corrects` 是复核 F4 说的"最高精度信号"——它必须进裁决包。

    新事实自带的修正自声明是**明文证据**，比任何相似度都硬。
    此前它没有结构化出口（X4：楼建好了没人搬进去）。
    """
    cand = _fact("c", "修正后的陈述")
    cand.corrects = "先前那个说法"
    pkg = build_input(cand, [(_fact("n1", "先前那个说法的原文"), 0.95)])
    assert pkg.corrects_hint == "先前那个说法"


def test_build_input_topk_default_is_wide_enough():
    """k=10 不是随手取的：跨语言冲突对实测排在 top-8，k=5 会整个漏掉。"""
    from bladex_core.flags import MEMORY_NUMERIC_DEFAULTS

    assert MEMORY_NUMERIC_DEFAULTS["BLADEX_ADJUDICATE_TOPK"] >= 8


# ── 复核 3.9 的真实裁决输入包（本卡的主验收剧本）────────────────────────


@pytest.fixture
def knot_package():
    """一条带修正宣告的新事实 + 5 条近邻，其中 N1/N3 与之直接矛盾。

    结构照搬复核 3.9 的实测输入包：对手方就在 top-1（相似度 0.956），
    e5 已经把冲突双方送到了同一张桌子上。
    """
    cand = _fact("c_new", "该事项的招募只进行过一次，此前把另一类招募误算成了第二次",
                 subject="某事项招募", attribute="次数")
    cand.corrects = "认为进行过两次招募"
    neighbors = [
        (_fact("n1", "该事项进行了两次招募：第一次在较早时间，第二次在稍后",
               subject="某事项招募", attribute="次数"), 0.956),
        (_fact("n2", "第二次招募面向的是另一类对象，发布于稍后的日期"), 0.954),
        (_fact("n3", "该事项针对同一类对象进行了两次招募", subject="某事项招募",
               attribute="次数"), 0.947),
        (_fact("n4", "该事项在更早的时间被裁定受理"), 0.943),
        (_fact("n5", "第一次招募面向的是另一类对象"), 0.941),
    ]
    return cand, neighbors


def test_knot_package_puts_the_opponents_on_the_table(knot_package):
    """裁决的**有效性前提**：对手方必须出现在输入包里。

    3.9 的结论是"e5 已经把冲突双方送到同一张桌子上，只是桌边没坐判定者"。
    这条测试守的是"桌子"，不是"判定者"（后者是模型的事）。
    """
    cand, neighbors = knot_package
    pkg = build_input(cand, neighbors, top_k=10)
    ids = [n.fact_id for n in pkg.neighbors]
    assert "n1" in ids and "n3" in ids, "两条矛盾方都必须在裁决输入里"
    assert pkg.corrects_hint, "修正宣告是明文证据，不能丢"


def test_knot_fast_path_catches_the_exact_key_conflict(knot_package):
    """这个包里 N1/N3 与候选**精确同键但值不同** → 快路径直接 UPDATE，零 LLM。"""
    cand, neighbors = knot_package
    v = fast_path_verdict(cand, [n for n, _ in neighbors])
    assert v is not None and v.op is AdjudicationOp.UPDATE
    assert v.target_fact_ids[0] in {"n1", "n3"}


def test_knot_converges_to_a_consistent_set(knot_package):
    """回放整个 knot：矛盾方带取代链退出 current，兼容方保留。

    验收判据（卡内）：「10+ 变体收敛为带取代链的一致集」。
    这里用 5 条的结构缩影，断言的是**收敛形状**而不是条数。
    """
    cand, neighbors = knot_package
    now = datetime.now(UTC)
    by_id = {f.id: f for f, _ in neighbors}

    # 模型判定：N1/N3 被取代，N2/N4/N5 兼容共存
    verdict = AdjudicationVerdict(candidate_id="c_new", op=AdjudicationOp.UPDATE,
                                  target_fact_ids=["n1", "n3"])
    keep, changed = apply_verdict(verdict, cand, by_id, now=now)

    assert keep is True
    assert {f.id for f in changed} == {"n1", "n3"}
    # 取代链完整：旧条失效 + 指向新条
    for f in changed:
        assert f.t_invalid == now and f.superseded_by == "c_new"
    # 兼容方不受影响（误伤它们就是把仍然成立的记忆当成矛盾清掉）
    for fid in ("n2", "n4", "n5"):
        assert by_id[fid].t_invalid is None, f"{fid} 与新事实兼容，不该被取代"


def test_related_but_compatible_is_not_a_target(knot_package):
    """反向红线：同话题 ≠ 矛盾。

    这是误合并红线在裁决层的对应物——把"相关"当成"取代"会静默删掉正确记忆，
    而它在库里看起来只是"少了一条"，没有任何告警。
    """
    cand, neighbors = knot_package
    now = datetime.now(UTC)
    by_id = {f.id: f for f, _ in neighbors}
    # 模型只指认真正矛盾的那条
    apply_verdict(AdjudicationVerdict(candidate_id="c_new",
                                      op=AdjudicationOp.UPDATE,
                                      target_fact_ids=["n1"]),
                  cand, by_id, now=now)
    assert by_id["n2"].t_invalid is None
    assert by_id["n4"].t_invalid is None


# ── profile 阈值不得静默压过 flags（2026-08-07，M5-2 前夜实测发现）──


def test_profile_novelty_matches_flags():
    """每个 embedding profile 的 `novelty` 必须等于 flags 里的默认值。

    背景：`effective_thresholds` 的优先级是「显式 env > per-model profile > flags」，
    所以 profile 里留一个旧值 = **静默改写 flags**，而且不打任何告警。

    实际发作过：M2 把 novelty 从 0.95（去重线）改成 0.98（只拦完全重发）并把默认值
    收进 flags，但 profile 表里 e5-large 那条仍是 07-26 按旧语义标定的 0.95、
    还标着 calibrated=True。于是生产机上 M2 的 0.98 从未生效，裁决漏斗在
    「候选 → 判重」这一段就被掐断，而日志只会平静地打 `novelty_threshold=0.95`。

    novelty 在新语义下不是 per-model 标定量（"完全重发"跟模型无关），
    所以这里要求全表一致。**确需某模型不同，就改这条断言** ——
    让分歧刻意且可见，是这条测试存在的全部意义。
    """
    from bladex_core.flags import MEMORY_NUMERIC_DEFAULTS
    from bladex_proxy.embedding import MODEL_PROFILES

    want = MEMORY_NUMERIC_DEFAULTS["BLADEX_NOVELTY_THRESHOLD"]
    bad = {m: p["novelty"] for m, p in MODEL_PROFILES.items()
           if "novelty" in p and p["novelty"] != want}
    assert not bad, (
        f"profile novelty 与 flags 默认（{want}）不一致：{bad}\n"
        "  profile 会静默压过 flags —— 要么改回一致，要么改这条断言说明为什么该不同")
