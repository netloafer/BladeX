"""重放取代判决的计划器（MQ-S32 病 B 第五环，2026-08-31）。

设计卡 `docs/planning/supersede-replay-and-basket-20260831.md`。
读数依据：1124 条同键并存里 **237 条（21.1%）** 是"判过取代、点名了兄弟、
兄弟今天仍 current"——因为 `fact.id` 确定性、取代关系只活在 Fact 对象上、
而 `consolidation:*` 判决在生产里零消费者。

本文件钉死四条安全阀 + 两条语义（先到先得、ts 来自记录），
每条都带**判别力对照**（把规则去掉会得到什么）。
"""

from __future__ import annotations

from datetime import UTC, datetime

from bladex_core.adjudication import plan_supersede_replay

T1 = datetime(2026, 8, 1, tzinfo=UTC)
T2 = datetime(2026, 8, 2, tzinfo=UTC)


def _rec(seq: str, cand: str, targets: list[str], ts: datetime = T1):
    return (seq, cand, targets, ts)


def test_replays_a_simple_supersession() -> None:
    plan, skipped = plan_supersede_replay([_rec("001", "f_new", ["f_old"])],
                                          {"f_new", "f_old"})
    assert plan == {"f_old": ("f_new", T1)}
    assert skipped == {}


def test_ts_comes_from_the_record_not_now() -> None:
    """🔴 用 now() 会让两次重建产出不同的 t_invalid，重建等价性就测不出来了。"""
    plan, _ = plan_supersede_replay([_rec("001", "f_new", ["f_old"], T2)],
                                    {"f_new", "f_old"})
    assert plan["f_old"][1] == T2


def test_first_judgment_wins_matching_apply_verdict() -> None:
    """先到先得 —— 与 `apply_verdict` 跳过 `t_invalid is not None` 逐条对齐。

    倒过来取"最后一条"会重放出一个**生产从未产生过**的状态。
    """
    plan, skipped = plan_supersede_replay(
        [_rec("002", "f_late", ["f_old"], T2), _rec("001", "f_early", ["f_old"], T1)],
        {"f_early", "f_late", "f_old"})
    assert plan == {"f_old": ("f_early", T1)}
    assert skipped["already_superseded"] == 1


def test_sorting_is_by_seq_not_input_order() -> None:
    """输入顺序不该决定结果——台账扫描顺序是 dict 迭代序，不保证有序。"""
    a = [_rec("001", "f_early", ["f_old"]), _rec("002", "f_late", ["f_old"])]
    assert plan_supersede_replay(a, {"f_early", "f_late", "f_old"})[0] \
        == plan_supersede_replay(list(reversed(a)), {"f_early", "f_late", "f_old"})[0]


def test_absent_candidate_is_skipped() -> None:
    """安全阀 1：不能让一条**不存在**的 fact 去取代别人（内容变了 ⇒ id 变了）。"""
    plan, skipped = plan_supersede_replay([_rec("001", "f_gone", ["f_old"])],
                                          {"f_old"})
    assert plan == {} and skipped == {"candidate_absent": 1}


def test_absent_target_is_skipped_which_is_how_tombstones_win() -> None:
    """安全阀 2 —— 墓碑走的就是这条：墓碑过滤在重放之前，被 forget 的 fact
    不在 `present` 里 ⇒ 不会因重放复活（ADR-0012 §3.6 减项最后生效）。"""
    plan, skipped = plan_supersede_replay([_rec("001", "f_new", ["f_forgotten"])],
                                          {"f_new"})
    assert plan == {} and skipped == {"target_absent": 1}


def test_self_target_is_skipped_not_silently_applied() -> None:
    """安全阀 4：脏数据护栏。自己取代自己不该静默通过（会让一条 fact 当场失效）。"""
    plan, skipped = plan_supersede_replay([_rec("001", "f_a", ["f_a", "f_b"])],
                                          {"f_a", "f_b"})
    assert plan == {"f_b": ("f_a", T1)}
    assert skipped == {"self_target": 1}


def test_one_judgment_can_converge_several_targets() -> None:
    """配合 08-31 的「一次收敛整个键」：UPDATE 的 targets 可以是整组。"""
    plan, _ = plan_supersede_replay([_rec("001", "f_new", ["f1", "f2", "f3"])],
                                    {"f_new", "f1", "f2", "f3"})
    assert set(plan) == {"f1", "f2", "f3"}
    assert all(v[0] == "f_new" for v in plan.values())


def test_chain_across_two_judgments() -> None:
    """A 取代 B、C 取代 A：两条都保留，链条不被压平。"""
    plan, skipped = plan_supersede_replay(
        [_rec("001", "A", ["B"]), _rec("002", "C", ["A"])], {"A", "B", "C"})
    assert plan == {"B": ("A", T1), "A": ("C", T1)}
    assert skipped == {}


def test_empty_inputs_are_safe() -> None:
    assert plan_supersede_replay([], set()) == ({}, {})
    assert plan_supersede_replay([_rec("001", "f", [])], {"f"}) == ({}, {})
