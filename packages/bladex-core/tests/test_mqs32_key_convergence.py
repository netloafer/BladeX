"""MQ-S32 病 B 的两条确定性修复（2026-08-31，`--batch` 全库读数驱动）。

读数（`docs/benchmarks/supersede-why-batch-20260831.md`，400 键 / 1124 条并存）：

- `update_landed` **55 条** = 取代**执行成功了、键仍并存** ⇒ 根因链第 4 环
  「一次只取代一条」从推断变成实证 → `fast_path_verdict` 改为一次收敛整个键。
- `adjudicator_error` **46 条** = 上游断线时裁决静默降级成一整块 ADD，
  而 ADD 是合法判决、下游看不出没被判过 → 新增裁决中断守卫
  （`adjudicate_outage`，与蒸馏中断守卫同型）。

两条都做**判别力断言**：把旧行为写成对照用例，确保测试不是在描述一个恒真的事。
"""

from __future__ import annotations

from datetime import UTC, datetime

from bladex_core.adjudication import (
    AdjudicationOp,
    adjudicate_outage,
    apply_verdict,
    fast_path_verdict,
)
from bladex_core.fact import Fact


def _fact(fid: str, content: str, *, subject: str = "某主题",
          attribute: str = "某槽位") -> Fact:
    return Fact(id=fid, content=content, subject=subject, attribute=attribute,
                scope="personal:local", item_kind="assertion")


# ── ① 一次收敛整个键（第 4 环）────────────────────────────────────────────


def test_update_targets_every_live_same_key_sibling() -> None:
    """N 条同键并存 ⇒ 一次判决把 N 条全部取代，不是只取代第一条。

    旧行为：`for old in neighbors: ... return` ⇒ targets 恒为 1 条。
    """
    cand = _fact("f_new", "最新的值")
    olds = [_fact(f"f_old{i}", f"旧值{i}") for i in range(4)]
    v = fast_path_verdict(cand, olds)
    assert v is not None and v.op is AdjudicationOp.UPDATE
    assert set(v.target_fact_ids) == {o.id for o in olds}, "必须一次收敛整个键"
    # 判别力：旧实现在同一输入上只会给出 1 个 target。
    assert len(v.target_fact_ids) > 1


def test_convergence_actually_invalidates_all_of_them() -> None:
    """判决只是决策——连着 `apply_verdict` 一起验，确保落盘那侧也收敛。"""
    now = datetime.now(UTC)
    cand = _fact("f_new", "最新的值")
    olds = [_fact(f"f_old{i}", f"旧值{i}") for i in range(3)]
    v = fast_path_verdict(cand, olds)
    keep, changed = apply_verdict(v, cand, {o.id: o for o in olds}, now=now)
    assert keep is True
    assert len(changed) == 3
    assert all(o.t_invalid is not None and o.superseded_by == "f_new" for o in olds)


def test_restatement_among_stale_ones_does_not_flip_to_noop() -> None:
    """🔴 邻居顺序不该决定结果。

    `[旧值, 与候选相同的重述]`：若"有一条重述就 NOOP"，那条**旧值**会原地留 current
    ——比旧行为还差。规则是"全部都是重述才 NOOP"。
    """
    cand = _fact("f_new", "值乙")
    stale = _fact("f_stale", "值甲")
    restated = _fact("f_same", "值乙")
    v = fast_path_verdict(cand, [stale, restated])
    assert v is not None and v.op is AdjudicationOp.UPDATE
    assert set(v.target_fact_ids) == {"f_stale", "f_same"}


def test_all_restatements_still_noop() -> None:
    """全是重述 ⇒ 候选丢弃（旧行为保持不变，这是零回归对照）。"""
    cand = _fact("f_new", "值甲")
    v = fast_path_verdict(cand, [_fact("f_a", "值甲"), _fact("f_b", "值甲")])
    assert v is not None and v.op is AdjudicationOp.NOOP
    assert len(v.target_fact_ids) == 1


def test_invalidated_and_self_are_excluded() -> None:
    """已失效的旧条、候选自己，都不进 targets（取代链不许被改写）。"""
    now = datetime.now(UTC)
    dead = _fact("f_dead", "早就没用了")
    dead.t_invalid = now
    live = _fact("f_live", "还活着的旧值")
    cand = _fact("f_new", "新值")
    v = fast_path_verdict(cand, [dead, live, cand])
    assert v is not None and v.target_fact_ids == ["f_live"]


def test_no_key_returns_none() -> None:
    cand = _fact("f_new", "内容", subject="", attribute="")
    assert fast_path_verdict(cand, [_fact("f_old", "内容")]) is None


# ── ② 裁决中断守卫 ───────────────────────────────────────────────────────


def test_outage_aborts_when_most_candidates_degraded() -> None:
    before = {"items": 100, "degraded_items": 0}
    after = {"items": 140, "degraded_items": 40}      # 本轮 40/40 全降级
    abort, items, degraded, ratio = adjudicate_outage(before, after, 0.5)
    assert abort is True
    assert (items, degraded, ratio) == (40, 40, 1.0)


def test_outage_is_measured_per_candidate_not_per_call() -> None:
    """🔴 分母是候选条数：分块后一次调用判 25 条，按调用数会把整块降级稀释掉。"""
    before = {"items": 0, "degraded_items": 0, "calls": 0, "fails": 0}
    after = {"items": 50, "degraded_items": 25, "calls": 2, "fails": 1}
    abort, items, degraded, ratio = adjudicate_outage(before, after, 0.5)
    assert abort is True and (items, degraded) == (50, 25) and ratio == 0.5


def test_sporadic_degradation_does_not_abort_but_is_reported() -> None:
    before = {"items": 0, "degraded_items": 0}
    after = {"items": 100, "degraded_items": 3}
    abort, items, degraded, ratio = adjudicate_outage(before, after, 0.5)
    assert abort is False
    assert degraded == 3, "零星降级也要报数——调用方据此打 partial_degrade 告警"
    assert 0 < ratio < 0.5


def test_threshold_zero_disables_the_guard() -> None:
    after = {"items": 10, "degraded_items": 10}
    assert adjudicate_outage({"items": 0, "degraded_items": 0}, after, 0.0)[0] is False


def test_missing_counters_report_unknown_not_healthy() -> None:
    """拿不到计数器 = 无从判断，返回全零让调用方**不打日志**，而不是打"守卫通过"。"""
    assert adjudicate_outage(None, {"items": 9, "degraded_items": 9}, 0.5) \
        == (False, 0, 0, 0.0)
    assert adjudicate_outage({"items": 0, "degraded_items": 0}, None, 0.5) \
        == (False, 0, 0, 0.0)


def test_zero_candidates_is_not_a_zero_ratio() -> None:
    """本轮零候选 ⇒ 不是 0% 降级，是没有分母。"""
    same = {"items": 7, "degraded_items": 2}
    assert adjudicate_outage(same, same, 0.5) == (False, 0, 0, 0.0)
