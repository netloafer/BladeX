"""批 BB · 2a · D7 成本：每条 fact 判定次数上限 + 日上限；超限停判、告警、pending 不丢、次日续。"""

from __future__ import annotations

import logging

import pytest
from _bb_support import FakeJudge, FakeLedger, FakePending, load_mutant, make_detector, mk_fact
from bladex_core import consistency as c


def _olds(n: int) -> list[tuple]:
    return [(mk_fact(f"old{i:02d}", f"旧内容{i}"), round(0.95 - i * 0.005, 4)) for i in range(n)]


def _run_per_fact(mod=c):
    judge = FakeJudge("不矛盾")
    det, parts = make_detector(mod, judge=judge, budget=mod.JudgeBudget(per_fact=6, daily_input_tokens=10**9))
    res = det.process(mk_fact("new", "新内容", day=9), _olds(12))
    return judge, res


def test_per_fact_cap_limits_judgments() -> None:
    judge, res = _run_per_fact()
    assert res.candidates == 12 and judge.calls == res.judged == 6
    assert res.truncated == 6 and not res.budget_exhausted


def test_per_fact_cap_mutant_is_red() -> None:
    m = load_mutant(("if self.store.fact_calls(fact_id) >= self.per_fact:", "if False:"))
    judge, _ = _run_per_fact(m)
    with pytest.raises(AssertionError):
        assert judge.calls <= 6, "去掉每条 fact 次数上限，必须红"


def _run_daily(mod=c, tokens_per_call: int = 1000):
    day = {"d": "2026-11-14"}
    store = mod.MemoryBudgetStore()
    budget = mod.JudgeBudget(per_fact=6, daily_input_tokens=3000, store=store, today=lambda: day["d"])
    judge, pending, ledger = FakeJudge("不矛盾", tokens=tokens_per_call), FakePending(), FakeLedger()
    det, _ = make_detector(mod, judge=judge, budget=budget, pending=pending, ledger=ledger)
    new = mk_fact("new", "新内容", day=9)
    return det, judge, pending, ledger, day, new


def test_daily_cap_stops_alerts_keeps_pending_and_resumes_next_day(caplog: pytest.LogCaptureFixture) -> None:
    det, judge, pending, ledger, day, new = _run_daily()
    olds = _olds(6)
    with caplog.at_level(logging.WARNING, logger="bladex_core.consistency"):
        r1 = det.process(new, olds)
    assert r1.budget_exhausted and r1.pending and judge.calls == 3          # 3000 / 1000
    assert pending.ids() == ["new"]                                           # 不丢
    assert any("contradiction_budget_exhausted" in rec.message and "pending=1" in rec.message for rec in caplog.records)
    assert len(ledger.records) == 3                                           # 已判的留痕

    # 同一天再续：仍超限，不多花
    det.resume(lambda fid: (new, olds))
    assert judge.calls == 3

    day["d"] = "2026-11-15"                                                   # 次日续
    out = det.resume(lambda fid: (new, olds))
    assert judge.calls == 6 and pending.ids() == [] and out[0].cached_skips == 3
    assert len(ledger.records) == 6                                           # 同一对不重判


def test_daily_cap_mutant_dropping_pending_is_red() -> None:
    m = load_mutant(("        if res.pending:\n            self._pending.add(fact.id)",
                     "        if False:\n            self._pending.add(fact.id)"))
    det, judge, pending, _ledger, _day, new = _run_daily(m)
    det.process(new, _olds(6))
    with pytest.raises(AssertionError):
        assert pending.ids() == ["new"], "超限后 pending 丢了，必须红"


def test_daily_cap_mutant_without_daily_check_is_red() -> None:
    m = load_mutant(("if self.store.day_input(self._today()) >= self.daily_input_tokens:", "if False:"))
    det, judge, *_ = _run_daily(m)
    det.process(mk_fact("new", "新内容", day=9), _olds(6))
    with pytest.raises(AssertionError):
        assert judge.calls <= 3, "去掉日上限，必须红"


def test_budget_survives_restart_through_store() -> None:
    """日计数落 store（2b 落 Index meta）：新建 JudgeBudget 复用同一 store 不清零。"""
    store = c.MemoryBudgetStore()
    b1 = c.JudgeBudget(per_fact=6, daily_input_tokens=1000, store=store, today=lambda: "d")
    b1.charge("f", 1000)
    b2 = c.JudgeBudget(per_fact=6, daily_input_tokens=1000, store=store, today=lambda: "d")
    assert b2.admit("g") is c.Admission.DAILY_CAP


def test_budget_defaults_come_from_flags() -> None:
    b = c.JudgeBudget()
    assert (b.per_fact, b.daily_input_tokens) == (6, 400000)


def test_upstream_error_keeps_pending_and_does_not_charge() -> None:
    def boom(_p: str):
        raise c.JudgeUpstreamError("429")

    det, parts = make_detector(judge=boom)
    r = det.process(mk_fact("new", "新", day=9), _olds(3))
    assert r.pending and r.judged == 0 and parts["pending"].ids() == ["new"]


def test_parse_failure_retries_once_then_gives_up_without_negative_ledger_entry() -> None:
    judge = FakeJudge(text="not json")
    det, parts = make_detector(judge=judge)
    new, olds = mk_fact("new", "新", day=9), _olds(1)
    r1 = det.process(new, olds)
    assert r1.parse_failed == 1 and r1.pending and not parts["ledger"].records
    r2 = det.process(new, olds)
    assert r2.parse_failed == 1 and judge.calls == 2
    r3 = det.process(new, olds)
    assert judge.calls == 2 and not parts["ledger"].records      # 共 2 次尝试，不再判，也不进否定台账
    assert r3.judged == 0
