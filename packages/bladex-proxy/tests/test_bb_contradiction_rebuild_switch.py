"""批 BB · 2a · D8 重建零判定 + 开关（关着：判定 0 / Hub 事件 0 / Index 写入 0）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from _bb_support import (
    FakeJudge,
    FakeLedger,
    FakePending,
    FakeSink,
    load_mutant,
    make_detector,
    mk_fact,
)
from bladex_core import consistency as c
from bladex_proxy.models import AdminEventType
from bladex_proxy.storage.memory_hub import MemoryHub
from bladex_proxy.storage.memory_index import MemoryIndex


def _olds(n: int) -> list[tuple]:
    return [(mk_fact(f"old{i:02d}", f"旧内容{i}"), round(0.95 - i * 0.005, 4)) for i in range(n)]


def _counts(judge: FakeJudge, parts: dict) -> tuple[int, int, int]:
    return judge.calls, len(parts["sink"].events), parts["ledger"].writes + parts["pending"].writes


# ── 开关 ─────────────────────────────────────────────────────────────────


def test_switch_default_is_off_and_zero_side_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(c.ENABLED_FLAG, raising=False)
    judge = FakeJudge()
    det = c.ContradictionDetector(  # 不传 enabled ⇒ 走真实开关解析
        judge=judge, budget=c.JudgeBudget(), ledger=(led := FakeLedger()), sink=(snk := FakeSink()),
        pending=(pnd := FakePending()), settled=lambda: c.SettledPairs({}))
    r = det.process(mk_fact("new", "新", day=9), _olds(5))
    assert r.skipped == "disabled"
    assert (judge.calls, len(snk.events), led.writes + pnd.writes) == (0, 0, 0)


def test_switch_on_has_side_effects_positive_control(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(c.ENABLED_FLAG, "1")
    judge = FakeJudge()
    det = c.ContradictionDetector(
        judge=judge, budget=c.JudgeBudget(), ledger=(led := FakeLedger()), sink=(snk := FakeSink()),
        pending=FakePending(), settled=lambda: c.SettledPairs({}))
    det.process(mk_fact("new", "新", day=9), _olds(5))
    assert judge.calls > 0 and snk.events and led.writes > 0


def test_switch_off_mutant_without_check_is_red() -> None:
    m = load_mutant(("        if not self._enabled():  # 关着：零判定、零 Hub 事件、零 Index 写入", "        if False:"))
    judge = FakeJudge()
    det, parts = make_detector(m, judge=judge, enabled=False)
    det.process(mk_fact("new", "新", day=9), _olds(5))
    with pytest.raises(AssertionError):
        assert _counts(judge, parts) == (0, 0, 0), "去掉开关判断，关着也有副作用，必须红"


# ── D8 ───────────────────────────────────────────────────────────────────


def _hub_with_flags(tmp_path: Path) -> tuple[MemoryHub, list]:
    hub = MemoryHub(str(tmp_path / "hub"))

    class HubSink:
        def append_flag(self, pair_key: str, payload: dict) -> None:
            hub.append_admin_event(AdminEventType.AUDIENCE_SET, "", pair_key, **payload)

    facts = [mk_fact(f"f{i}", f"内容{i}", day=2 + i) for i in range(4)]
    det, _ = make_detector(judge=FakeJudge("矛盾"), sink=HubSink())
    r = det.process(facts[-1], [(f, 0.95 - i * 0.01) for i, f in enumerate(facts[:-1])])
    lo, hi = r.flagged[0].split("|")
    sha = {f.id: c.content_sha(f) for f in facts}
    hub.append_admin_event(AdminEventType.AUDIENCE_SET, "", r.flagged[0], **c.build_verdict_payload(
        (lo, hi), (sha[lo], sha[hi]), verdict="confirmed", origin="dashboard"))
    return hub, facts


def _state_json(mod, hub: MemoryHub) -> str:
    return json.dumps(mod.replay(e for _, e in hub.scan_admin_events()).snapshot(), sort_keys=True, ensure_ascii=False)


def test_rebuild_makes_zero_judgments_and_flag_set_is_identical(tmp_path: Path) -> None:
    hub, facts = _hub_with_flags(tmp_path)
    before = _state_json(c, hub)
    assert json.loads(before)["flags"] and json.loads(before)["verdicts"], "场景必须真有标记与裁定"

    ix = MemoryIndex(str(tmp_path / "ix_rebuild"))
    ix.open()
    for f in facts:
        ix.add_fact(f)
    ix._replay_admin_events(hub)                      # 全量重建期的重放
    judge, pending = FakeJudge("矛盾"), FakePending()
    pending.add("f3")
    det, parts = make_detector(judge=judge, pending=pending)
    for f in facts:
        det.process(f, [(g, 0.9) for g in facts if g.id != f.id], rebuilding=True)
    det.resume(lambda fid: (facts[-1], [(g, 0.9) for g in facts[:-1]]), rebuilding=True)
    assert _counts(judge, parts) == (0, 0, parts["pending"].writes) and judge.calls == 0
    assert _state_json(c, hub) == before              # 重建后标记集合逐字相等


def test_rebuild_guard_mutant_is_red(tmp_path: Path) -> None:
    m = load_mutant(("        if rebuilding:  # D8：全量重建期不调模型，标记来自 Hub 重放", "        if False:"))
    judge = FakeJudge("矛盾")
    det, _ = make_detector(m, judge=judge)
    det.process(mk_fact("new", "新", day=9), _olds(5), rebuilding=True)
    with pytest.raises(AssertionError):
        assert judge.calls == 0, "重建期调了模型，必须红"


def test_replay_dispatch_mutant_is_red(tmp_path: Path) -> None:
    """去掉重放里对标记的分派 ⇒ 标记集合与重建前不等。"""
    hub, _ = _hub_with_flags(tmp_path)
    m = load_mutant(("        state.flags[rec.key] = rec\n", "        pass\n"))
    with pytest.raises(AssertionError):
        assert _state_json(m, hub) == _state_json(c, hub), "丢了标记重放，必须红"


def test_settled_binds_content_sha(tmp_path: Path) -> None:
    hub, facts = _hub_with_flags(tmp_path)
    st = c.settled_pairs(e for _, e in hub.scan_admin_events())
    assert len(st) == 1
    flagged = next(iter(c.replay(e for _, e in hub.scan_admin_events()).verdicts.values()))
    a, b = (next(f for f in facts if f.id == i) for i in flagged.pair)
    assert st.covers(a, b)
    edited = a.model_copy(update={"content": a.content + "（改）"})
    assert not st.covers(edited, b), "任一侧文字变了，旧裁定不再覆盖"


def test_settled_pair_is_not_rejudged() -> None:
    a, b = mk_fact("new", "新", day=9), mk_fact("old00", "旧内容0")
    pk = c.pair_key(a.id, b.id)
    sha = {a.id: c.content_sha(a), b.id: c.content_sha(b)}
    lo, hi = pk.split("|")
    ev = type("E", (), {"event_type": AdminEventType.AUDIENCE_SET, "payload": c.build_verdict_payload(
        (lo, hi), (sha[lo], sha[hi]), verdict="not_same_thing", origin="cli")})()
    judge = FakeJudge()
    det, _ = make_detector(judge=judge, settled=lambda: c.settled_pairs([ev]))
    r = det.process(a, [(b, 0.95)])
    assert judge.calls == 0 and r.settled_skips == 1
