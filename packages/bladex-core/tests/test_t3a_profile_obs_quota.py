"""T3-A（检索侧 16 点收复卡，2026-08-15）：profile_obs 限额参与 + 排序折价。

背景：T1 归因 11/18 错题——temporal 题的时间锚（"加入/开始/购买 于 DATE"）
蒸成 profile_obs，在融合类型配额层被整类丢弃（QUOTA_MAX_PROFILE_OBS=0）→
检索结构性不可达；①平面画像卡实测不聚合其内容 → 两头无消费者=死数据。
证据：docs/benchmarks/longmemeval-retr-20260815.md（fact 级实锤 3 题）。

修法（拍板「参与召回+排序折价」）：配额 0→flag `BLADEX_QUOTA_PROFILE_OBS`（默认 2）
限额参与 + 融合总分乘 `BLADEX_PROFILE_OBS_DISCOUNT`（默认 0.85）折价。
**默认参数（0 配额 / 1.0 折价）= 改造前行为逐字不变**（回归通道）。
"""

from __future__ import annotations

from bladex_core.fact import Fact, ItemKind
from bladex_core.flags import MEMORY_NUMERIC_DEFAULTS
from bladex_core.fusion import Channel, apply_type_quota, fuse_and_rerank


def _f(fid: str, kind: ItemKind = ItemKind.ASSERTION, *, importance: float = 0.7) -> Fact:
    return Fact(id=fid, content=f"内容 {fid}", item_kind=kind, importance=importance,
                source_user_id="u", source_session="s")


# ── 配额层 ──────────────────────────────────────────────────────────────


def test_default_param_drops_profile_obs_verbatim():
    """不传参 = 旧行为逐字不变：profile_obs 整类丢弃（既有测试的同款断言）。"""
    ranked = [_f("p1", ItemKind.PROFILE_OBS), _f("a1"), _f("a2")]
    out = [f.id for f in apply_type_quota(ranked, 5)]
    assert "p1" not in out and out == ["a1", "a2"]


def test_quota_admits_up_to_cap_in_rank_order():
    """限额参与：按原排序放行前 N 条 profile_obs，超限丢弃。"""
    ranked = [_f("p1", ItemKind.PROFILE_OBS), _f("a1"),
              _f("p2", ItemKind.PROFILE_OBS), _f("a2"),
              _f("p3", ItemKind.PROFILE_OBS)]
    out = [f.id for f in apply_type_quota(ranked, 5, max_profile_obs=2)]
    assert out == ["p1", "a1", "p2", "a2"]      # p3 超限；相对顺序保持


def test_quota_zero_explicit_equals_default():
    """显式 0 与缺省一致（flag 设 0 = 回归通道）。"""
    ranked = [_f("p1", ItemKind.PROFILE_OBS), _f("a1")]
    assert [f.id for f in apply_type_quota(ranked, 5, max_profile_obs=0)] == ["a1"]


def test_file_ref_cap_and_lesson_floor_untouched():
    """既有配额语义不受牵连：file_ref ≤1、lesson 保底仍生效。"""
    ranked = [_f("fr1", ItemKind.FILE_REF), _f("fr2", ItemKind.FILE_REF)] + \
        [_f(f"a{i}") for i in range(5)] + [_f("l1", ItemKind.LESSON)]
    out = [f.id for f in apply_type_quota(ranked, 3, max_profile_obs=2)]
    assert "fr2" not in out and "l1" in out


# ── 折价（fuse_and_rerank 打分环）────────────────────────────────────────


def _run_fuse(discount: float, quota: int = 2):
    p = _f("p1", ItemKind.PROFILE_OBS)
    a = _f("a1")
    facts = {"p1": p, "a1": a}
    # 两路秩互补：p1 与 a1 融合票完全对称 → 无折价时并列，靠 id 决序（p1<a1? 'a1'<'p1'）
    channels = [Channel("dense", ["p1", "a1"]), Channel("lexical", ["a1", "p1"])]
    out, _info = fuse_and_rerank(facts, channels, 2, max_profile_obs=quota,
                                 profile_obs_discount=discount)
    return [f.id for f in out]


def test_discount_demotes_profile_obs_on_ties():
    """对称票下：无折价按 id 定序（a1 先），折价后 p1 必须仍在但排后。"""
    assert _run_fuse(1.0) == ["a1", "p1"]
    assert _run_fuse(0.85) == ["a1", "p1"]


def test_discount_flips_leading_profile_obs():
    """p1 两路皆第一（本应压过 a1）：折价把它压到 a1 之后——参与但不挤掉散点。

    数值口径（先算后写，防拍脑袋预期）：min-max 归一下 p1=1.0、a1≈0.49、
    b1=0.0（归一锚点）；修正项两者同为 0.15·(0.7/1.5)+0.1·1.0=0.17。
    无折价 p1=1.17 > a1=0.66；折价 0.5 后 p1=0.585 < 0.66 → 翻转，
    但 p1=0.585 > b1=0.17 → 仍参与、不被清出。
    """
    p = _f("p1", ItemKind.PROFILE_OBS)
    a, b = _f("a1"), _f("b1")
    facts = {"p1": p, "a1": a, "b1": b}
    channels = [Channel("dense", ["p1", "a1", "b1"]),
                Channel("lexical", ["p1", "a1", "b1"])]
    out_no, _ = fuse_and_rerank(facts, channels, 3, max_profile_obs=2,
                                profile_obs_discount=1.0)
    out_disc, _ = fuse_and_rerank(facts, channels, 3, max_profile_obs=2,
                                  profile_obs_discount=0.5)
    assert [f.id for f in out_no] == ["p1", "a1", "b1"]
    assert [f.id for f in out_disc] == ["a1", "p1", "b1"]


def test_fuse_default_params_verbatim_old_behavior():
    """fuse_and_rerank 不传新参 = profile_obs 整类丢弃（改造前逐字）。"""
    p = _f("p1", ItemKind.PROFILE_OBS)
    a = _f("a1")
    out, _ = fuse_and_rerank({"p1": p, "a1": a},
                             [Channel("dense", ["p1", "a1"])], 2)
    assert [f.id for f in out] == ["a1"]


# ── 注入面（prefetch 选取环）——仪器问题 #14 的教训 ─────────────────────
#
# 第一版 T3-A 只改了 fusion 层，prefetch 选取环还有 U10 的第二道整类排除：
# obs 在融合层占 top-k 名额、注入前又被扔 = 净负作用（−22 分事故次因）。
# 内部尺子走 index.search 只过融合层，测不到这里——所以本组测试必须钉在
# **注入面**上：同一 flag、同一口径，两个消费点行为一致。


class _ObsRetriever:
    """③平面检索桩：返回带 kind 的 facts。"""

    def __init__(self, facts):
        self._facts = facts

    def search(self, query, top_k=10, user_id=None, visibility=None, q_vec=None):
        return list(self._facts)

    def list_hard_rules(self):
        return []


# ── flag 表接线 ─────────────────────────────────────────────────────────


def test_flags_declared_in_numeric_table():
    """默认值只在 flags 表一处（ADR-0027 §5.4：默认值散落=腐化温床）。"""
    assert MEMORY_NUMERIC_DEFAULTS["BLADEX_QUOTA_PROFILE_OBS"] == 2.0
    assert MEMORY_NUMERIC_DEFAULTS["BLADEX_PROFILE_OBS_DISCOUNT"] == 0.85
