"""G9.3 S4：融合分数分解（`fuse_and_rerank(explain=True)`）的验收剧本。

**为什么这个功能值得有测试**：2026-08-20 那次"注入丢记忆"的归因结论
（72.6% 死在同义折叠、`rank` 那档只占 0.8%）是靠仪器**在生产外面把整条流水线
重算一遍**得出的。外挂重算的尺子迟早与生产脱钩——`FOLD_COSINE` 从 0.92 调到 0.97
那天，外挂那份就得跟着改，改漏了没有任何人会发现。所以分解必须由**生产自己产出**。

两条红线：
  ① `explain=False` 时输出**逐字不变**（它是热路径，多算一分钱都不行）；
  ② `dropped_at` 必须记**被淘汰的候选**——返回值里只有幸存者，
     而"为什么没进来"恰恰是归因要问的那一半。
"""

from __future__ import annotations

from bladex_core.fusion import DROPPED_AT_VALUES, Channel, fuse_and_rerank


class _F:
    """最小 fact 替身（fusion 只吃 id/content/entities/importance/item_kind）。"""

    def __init__(self, fid: str, content: str = "", *, importance: float = 0.0,
                 kind: str = "assertion", entities: list[str] | None = None):
        self.id = fid
        self.content = content or fid
        self.importance = importance
        self.item_kind = kind
        self.entities = entities or []
        self._score = 0.0


def _mk(n: int) -> dict[str, _F]:
    return {f"f{i}": _F(f"f{i}", f"内容各不相同的第 {i} 条事实") for i in range(n)}


# ── ① explain=False 逐字不变 ──────────────────────────────────────────────


def test_explain_off_output_is_byte_identical():
    facts = _mk(6)
    ch = [Channel("dense", list(facts)), Channel("lexical", list(reversed(facts)))]
    out_a, info_a = fuse_and_rerank(facts, ch, top_k=4)
    out_b, info_b = fuse_and_rerank(_mk(6), ch, top_k=4, explain=False)
    assert [f.id for f in out_a] == [f.id for f in out_b]
    assert info_a == info_b


def test_explain_off_does_not_add_the_key():
    facts = _mk(4)
    _out, info = fuse_and_rerank(facts, [Channel("dense", list(facts))], top_k=2)
    assert "explain" in info or True          # 不强制存在
    assert info.get("explain") is None        # 关时必须没有


def test_explain_on_keeps_the_same_ranking():
    """开 explain 只是**多记一份账**，不得改变排序——否则归因看的是另一条流水线。"""
    facts = _mk(6)
    ch = [Channel("dense", list(facts))]
    out_plain, _ = fuse_and_rerank(facts, ch, top_k=3)
    out_expl, _info = fuse_and_rerank(_mk(6), ch, top_k=3, explain=True)
    assert [f.id for f in out_plain] == [f.id for f in out_expl]


# ── ② 分解本身 ────────────────────────────────────────────────────────────


def test_breakdown_components_sum_to_final():
    facts = _mk(5)
    facts["f0"].importance = 1.2
    _out, info = fuse_and_rerank(facts, [Channel("dense", list(facts))],
                                 top_k=3, explain=True)
    row = info["explain"]["f0"]
    parts = row["base"] + row["importance_term"] + row["recency_term"] + row["audience_bonus"]
    assert abs(parts * row["discount"] - row["final"]) < 1e-5


def test_breakdown_covers_every_scored_candidate_not_just_survivors():
    """分解要覆盖**全部参与打分的候选**，不是只覆盖 top_k。"""
    facts = _mk(8)
    out, info = fuse_and_rerank(facts, [Channel("dense", list(facts))],
                                top_k=2, explain=True)
    assert len(out) == 2
    assert len(info["explain"]) == 8


def test_dropped_at_marks_survivors_as_none():
    facts = _mk(5)
    out, info = fuse_and_rerank(facts, [Channel("dense", list(facts))],
                                top_k=5, explain=True)
    for f in out:
        assert info["explain"][f.id]["dropped_at"] is None


def test_dropped_at_records_mmr_stage_for_squeezed_out_candidates():
    """被 top_k 挤掉的候选要标出死在哪一段——这是归因的主入口。"""
    facts = _mk(9)
    _out, info = fuse_and_rerank(facts, [Channel("dense", list(facts))],
                                 top_k=3, explain=True)
    dropped = [r["dropped_at"] for r in info["explain"].values() if r["dropped_at"]]
    assert len(dropped) == 6
    assert set(dropped) <= set(DROPPED_AT_VALUES)


def test_fold_stage_is_attributed_to_fold_not_to_ranking():
    """近重复候选死在折叠，不能记成"排序没排上"——两者修法完全相反。

    钉的就是 2026-08-20 那次归因的结论形态（fold 72.6% vs rank 0.8%）。
    """
    # 🔴 折叠的第二条件是**实体 Jaccard ≥ 0.6**，而 `_entity_jaccard` 对空集合返回
    # 0.0 —— 不给 entities 的话折叠永远不触发，这条测试会变成一个恒真的空壳
    # （写测试先验证前提：初稿正是这么错的，跑出来 folded=0 才发现）。
    facts = {"a": _F("a", "同一句话", entities=["泰山啤酒"]),
             "b": _F("b", "同一句话", entities=["泰山啤酒"]),
             "c": _F("c", "另一件事", entities=["虎彩集团"])}
    vectors = {"a": [1.0, 0.0], "b": [1.0, 0.0], "c": [0.0, 1.0]}
    _out, info = fuse_and_rerank(facts, [Channel("dense", ["a", "b", "c"])],
                                 top_k=3, vectors=vectors, explain=True,
                                 fold_cosine=0.9)
    assert info["explain"]["b"]["dropped_at"] == "fold"
    assert info["folded"] >= 1


# ── ③ 配额那一段的三件事要分开（2026-08-20d，仪器改吃 explain 时补）─────────


def test_rank_cut_is_not_reported_as_a_quota_kill():
    """`apply_type_quota(ranked, 2×top_k)` 同时干着「类型配额」和「截断线」两件事。

    只是名次落在 2×top_k 之外的候选记 `rank_cut`，**不能**记成 `quota_*`——
    两者修法完全相反（前者调融合权重/召回宽度，后者调配额）。
    仪器此前正是在生产外面自己拆的这一刀（`exit_stage` 按 `after_fold.index`
    比截断线），拆法一旦与生产不同就是两把尺子。
    """
    facts = _mk(9)                       # top_k=3 ⇒ 截断线 = 6，9−6 = 3 条纯名次不够
    _out, info = fuse_and_rerank(facts, [Channel("dense", list(facts))],
                                 top_k=3, explain=True)
    stages = [r["dropped_at"] for r in info["explain"].values()]
    assert stages.count("rank_cut") == 3
    assert stages.count("quota_type") == 0          # 全是 assertion，没有类型上限触发
    assert stages.count("mmr") == 3                 # 截断线内 6 → top_k 3


def test_type_quota_kill_is_labelled_quota_type():
    """画像原料撞上限（默认 `QUOTA_MAX_PROFILE_OBS = 0`）走的是另一条路。"""
    facts = {"a": _F("a", "断言"), "p": _F("p", "画像原料", kind="profile_obs")}
    _out, info = fuse_and_rerank(facts, [Channel("dense", ["a", "p"])],
                                 top_k=5, explain=True)
    assert info["explain"]["p"]["dropped_at"] == "quota_type"
    assert info["explain"]["a"]["dropped_at"] is None


def test_dropped_at_stays_inside_the_declared_closed_set():
    """闭集只有一处定义（`DROPPED_AT_VALUES`）——仪器照它读，不再自己默写一份。

    钉这条是因为本仓栽过两次「闭集按记忆默写」（`lane` 漏一个取值 ⇒
    447 条边静默消失）。新增出局原因时这条测试会先红。
    """
    facts = _mk(9)
    facts["f8"] = _F("f8", "画像原料", kind="profile_obs")
    _out, info = fuse_and_rerank(facts, [Channel("dense", list(facts))],
                                 top_k=2, explain=True)
    for row in info["explain"].values():
        assert row["dropped_at"] is None or row["dropped_at"] in DROPPED_AT_VALUES


def test_fold_pair_comes_from_inside_fold_synonyms():
    """折叠掉它的是谁、cos/jaccard 多少——生产给出，仪器不回找。

    🔴 外挂回找那份用的是模块常量 `FOLD_COSINE`（0.92），而生产早已由
    `BLADEX_FOLD_COSINE`（0.97）接线：同一个"折叠"在两把尺子下判据不同。
    """
    facts = {"a": _F("a", "同一句话", entities=["泰山啤酒"]),
             "b": _F("b", "同一句话", entities=["泰山啤酒"])}
    vectors = {"a": [1.0, 0.0], "b": [1.0, 0.0]}
    _out, info = fuse_and_rerank(facts, [Channel("dense", ["a", "b"])],
                                 top_k=3, vectors=vectors, explain=True,
                                 fold_cosine=0.9)
    row = info["explain"]["b"]
    assert row["dropped_at"] == "fold"
    assert row["folded_by"] == "a"
    assert row["fold_cos"] == 1.0 and row["fold_jaccard"] == 1.0
    assert row["fold_why"] == "cosine"


def test_fold_pair_records_the_supersede_key_path_separately():
    """同取代键折叠与 cosine 折叠是两种机制，`fold_why` 分开——
    前者是"这两条本就是同一个属性的新旧值"，后者才是相似度判断。"""
    import bladex_core.fusion as _f

    facts = {"a": _F("a", "住址 = 北京"), "b": _F("b", "住址 = 上海")}
    orig = _f._supersede_key
    _f._supersede_key = lambda fact: ("personal:u", "用户住址", "城市")
    try:
        _out, info = fuse_and_rerank(facts, [Channel("dense", ["a", "b"])],
                                     top_k=3, explain=True)
    finally:
        _f._supersede_key = orig
    row = info["explain"]["b"]
    assert row["dropped_at"] == "fold" and row["fold_why"] == "supersede_key"
    # 走取代键那条路时没算过 cosine，落 0.0 —— 读的人不能把它当"相似度为 0"
    assert row["fold_cos"] == 0.0


def test_explain_off_does_not_pay_for_the_pair_bookkeeping():
    """关 explain 时 `fold_synonyms` / `apply_type_quota` 的 out-param 都是 None，
    输出仍逐字不变（热路径不为归因付一分钱）。"""
    facts = {"a": _F("a", "同一句话", entities=["泰山啤酒"]),
             "b": _F("b", "同一句话", entities=["泰山啤酒"]),
             "c": _F("c", "另一件事", entities=["虎彩集团"])}
    vectors = {"a": [1.0, 0.0], "b": [1.0, 0.0], "c": [0.0, 1.0]}
    ch = [Channel("dense", ["a", "b", "c"])]
    out_off, info_off = fuse_and_rerank(dict(facts), ch, top_k=2,
                                        vectors=vectors, fold_cosine=0.9)
    out_on, info_on = fuse_and_rerank(dict(facts), ch, top_k=2, vectors=vectors,
                                      fold_cosine=0.9, explain=True)
    assert [f.id for f in out_off] == [f.id for f in out_on]
    assert info_off["folded"] == info_on["folded"]
    assert "explain" not in info_off


def test_entity_key_defaults_to_zero_when_channel_absent():
    """没有 entity 通道时 `entity_key` 是 0.0 而不是缺字段——仪器不该做判空分支。"""
    facts = _mk(3)
    _out, info = fuse_and_rerank(facts, [Channel("dense", list(facts))],
                                 top_k=2, explain=True)
    assert all(r["entity_key"] == 0.0 for r in info["explain"].values())
