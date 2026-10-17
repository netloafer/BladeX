"""批 Z · Z3 —— 账本 close / supersede / reopen 原语 + 完结即停锚（Y1b）。

判据编号对 `prompt-batch-Z3-20260923.md` §2：
- ① 生命周期事件重放后状态保持；supersede 链可读（阴性：删重放分支 ⇒ 红）；
- ③ 关闭激活账本 ⇒ 该 scope 激活解除，之后的轮次不锚到它；
- ④ 机判「做完」⇒ 停锚（阴性：拆掉停锚 ⇒ 仍锚）；恢复两条路（Open/Next 重新非空 · 显式切回）；
- ⑥ close 不改 Goal。
（② 在 `test_ledger_relevant_candidates.py`，与相关性候选的其余用例同处。）
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from bladex_core.ledger import (
    EVENT_LEDGER_CLOSE,
    EVENT_LEDGER_CREATE,
    EVENT_LEDGER_REOPEN,
    EVENT_LEDGER_SUPERSEDE,
    EVENT_LEDGER_SWITCH,
    EVENT_LEDGER_UPDATE,
    LEDGER_EVENT_TYPES,
    LedgerEntry,
    SwitchTimeline,
    add_entry,
    apply_ledger_tombstones,
    binding_sessions,
    close_ledger,
    ledger_is_done,
    new_ledger,
    parse_ledger_md,
    render_ledger_md,
    reopen_ledger,
    replay_ledger_events,
    switch_timeline,
)
from bladex_core.ledger_runtime import (
    GATE_OTHER_LEDGER,
    GATE_PASS,
    activation_scope,
    ledger_done_paused,
    ledger_gate,
    turn_active_ledger,
)


@pytest.fixture(autouse=True)
def _done_pause_on(monkeypatch):
    """停锚默认关（读数见 flags 注释）；本文件钉的是**打开后**的机制，逐条开。"""
    monkeypatch.setenv("BLADEX_LEDGER_DONE_PAUSE", "1")


AGENT = "dsh"
PROJECT = "p-e479f97c661a"
SCOPE = activation_scope(AGENT, PROJECT)


def _ms(h: int, m: int, s: int = 0) -> float:
    return datetime(2026, 9, 19, h, m, s, tzinfo=UTC).timestamp() * 1000.0


def _ts(h: int, m: int, s: int = 0) -> datetime:
    return datetime(2026, 9, 19, h, m, s, tzinfo=UTC)


def _key(session: str, h: int, m: int, s: int = 0) -> str:
    return f"local/{AGENT}/{session}/{int(_ms(h, m, s))}-0"


def _led(lid: str = "ldg-w15", *, verified: int = 0, open_: int = 0, next_: int = 0):
    led = new_ledger(ledger_id=lid, title="MQ-W15 守卫测试", goal="守卫测试接线",
                     goal_source="model", created_at="2026-09-19T07:46:24+00:00")
    for sec, n in (("verified", verified), ("open", open_), ("next", next_)):
        for i in range(n):
            led = add_entry(led, sec, LedgerEntry(text=f"{sec} {i}", source="model"),
                            actor="model")
    return led


def _ev(etype: str, payload: dict, ts: datetime) -> dict:
    return {"type": etype, "payload": payload, "ts": ts}


def _switch(to: str, ts: datetime, *, frm: str = "", session: str = "s-a",
            project: str = PROJECT) -> dict:
    return _ev(EVENT_LEDGER_SWITCH, {"agent_id": AGENT, "project_id": project,
                                     "session_id": session, "from_ledger_id": frm,
                                     "to_ledger_id": to}, ts)


def _update(led, ts: datetime, **writer) -> dict:
    return _ev(EVENT_LEDGER_UPDATE, {"ledger": led.model_dump(), **writer}, ts)


# ── W15 形状（Y1a 抽检 15 条锚错的原样时序，时间取 UTC）──────────────────────
#   07:46:24 建本 + 切入（s-a）→ 07:49:47 写（仍有 Open/Next）→ 07:50:58 写：Open 0 · Next 0 · Verified 5
#   → 08:27 起另一会话 s-b 问 ADR（锚层修后被锚进 W15，全是错边）


def _w15_events(*, writer: dict | None = None) -> list[dict]:
    return [
        _ev(EVENT_LEDGER_CREATE, {"ledger": _led(open_=1, next_=3).model_dump()}, _ts(7, 46, 24)),
        _switch("ldg-w15", _ts(7, 46, 24)),
        _update(_led(verified=3, open_=1, next_=2), _ts(7, 49, 47)),
        _update(_led(verified=5), _ts(7, 50, 58), **(writer or {})),
    ]


def test_done_shape_predicate():
    assert ledger_is_done(_led(verified=1))
    assert not ledger_is_done(_led(verified=1, open_=1))
    assert not ledger_is_done(_led(verified=1, next_=1))
    assert not ledger_is_done(_led())                 # 空账本不是做完
    assert ledger_is_done(_led(verified=2).model_dump())   # 事件载荷形态同一判据


def test_w15_turns_after_done_do_not_anchor():
    """判据 ④ 的机制半：完结之后另一会话的轮次**不再锚**到它；完结之前照锚。"""
    tl = switch_timeline(_w15_events())
    assert isinstance(tl, SwitchTimeline)
    assert turn_active_ledger(tl, _key("s-a", 7, 48), PROJECT) == "ldg-w15"
    assert turn_active_ledger(tl, _key("s-a", 7, 50, 10), PROJECT) == "ldg-w15"
    for h, m in ((8, 27), (10, 0), (14, 1)):
        assert turn_active_ledger(tl, _key("s-b", h, m), PROJECT) == "", (h, m)
        assert ledger_done_paused(tl, "ldg-w15", _key("s-b", h, m))


def test_default_off_is_zero_behavior(monkeypatch):
    """开关默认关：不设 env ⇒ 没有完结区间 ⇒ 逐字等于 Z3 之前（完结之后照锚）。"""
    monkeypatch.delenv("BLADEX_LEDGER_DONE_PAUSE", raising=False)
    tl = switch_timeline(_w15_events())
    assert tl.done_spans == {}
    assert turn_active_ledger(tl, _key("s-b", 8, 27), PROJECT) == "ldg-w15"


def test_reverse_mutation_plain_dict_timeline_keeps_anchoring():
    """判据 ④ 阴性对照：拆掉停锚（时间轴退成普通 dict = 修前口径）⇒ 同一批轮次照锚。"""
    tl = dict(switch_timeline(_w15_events()))
    assert turn_active_ledger(tl, _key("s-b", 8, 27), PROJECT) == "ldg-w15"


def test_live_binding_is_not_released_by_done():
    """停锚只影响锚定：绑定终态照旧（解除绑定是 close 的事）。"""
    _pool, bindings = replay_ledger_events(_w15_events())
    assert bindings[SCOPE] == "ldg-w15"
    assert switch_timeline(_w15_events())[SCOPE][-1] == (_ms(7, 46, 24), "ldg-w15")


def test_gate_label_semantics_keep_other_ledger_cards_closed():
    """判据 ⑤ 不换受害者：停锚只摘 ANCHOR，不摘标签。DPL 闸② 读标签口径
    （`apply_done_pause=False`）⇒ 被停锚的轮次进不了**别的**账本锚卡，回自己这本交给内容。
    阴性对照：闸② 若读停锚口径（标签为空）⇒ 别人的锚卡也放行——那就是换受害者。"""
    tl = switch_timeline(_w15_events())
    k = _key("s-b", 8, 27)
    label = turn_active_ledger(tl, k, PROJECT, apply_done_pause=False)
    assert label == "ldg-w15"

    def gate(turn_ledger: str, target: str) -> str:
        return ledger_gate(turn_ledger_id=turn_ledger, turn_ms=_ms(8, 27),
                           target_ledger_id=target, target_created_ms=_ms(1, 0))

    assert gate(label, "ldg-other") == GATE_OTHER_LEDGER
    assert gate(label, "ldg-w15") == GATE_PASS
    paused = turn_active_ledger(tl, k, PROJECT)
    assert paused == "" and gate(paused, "ldg-other") == GATE_PASS   # 阴性对照


def test_writer_user_turn_keeps_anchoring():
    """完结写入所在的那个用户轮照常锚（收尾轮属于它）；同会话的下一个用户轮停锚。"""
    tl = switch_timeline(_w15_events(writer={"session_id": "s-a", "turn_index": 4}))
    k = _key("s-a", 7, 51, 13)
    assert turn_active_ledger(tl, k, PROJECT, turn_index=4) == "ldg-w15"
    assert turn_active_ledger(tl, k, PROJECT, turn_index=lambda: 4) == "ldg-w15"
    assert turn_active_ledger(tl, k, PROJECT, turn_index=5) == ""
    assert turn_active_ledger(tl, k, PROJECT) == ""   # 不给 turn_index ⇒ 不豁免
    assert turn_active_ledger(tl, _key("s-b", 8, 27), PROJECT, turn_index=4) == ""


def test_writer_lookup_is_lazy():
    """调用方给的是可调用对象时，只有命中区间且会话对得上才取（绝大多数轮次零 Hub 读）。"""
    tl = switch_timeline(_w15_events(writer={"session_id": "s-a", "turn_index": 4}))
    calls: list[int] = []

    def ti() -> int:
        calls.append(1)
        return 4

    turn_active_ledger(tl, _key("s-a", 7, 48), PROJECT, turn_index=ti)    # 区间前
    turn_active_ledger(tl, _key("s-b", 9, 0), PROJECT, turn_index=ti)     # 别的会话
    assert calls == []


def test_resume_when_open_or_next_refilled():
    ev = _w15_events() + [_update(_led(verified=5, next_=1), _ts(9, 0))]
    tl = switch_timeline(ev)
    assert turn_active_ledger(tl, _key("s-b", 8, 27), PROJECT) == ""
    assert turn_active_ledger(tl, _key("s-b", 9, 5), PROJECT) == "ldg-w15"


def test_resume_on_explicit_switch_back():
    """模型显式切回 ⇒ 恢复（原则 15：不拿启发式否决显式动作）。"""
    ev = _w15_events() + [
        _switch("ldg-other", _ts(8, 0), frm="ldg-w15"),
        _switch("ldg-w15", _ts(9, 0), frm="ldg-other"),
    ]
    tl = switch_timeline(ev)
    assert turn_active_ledger(tl, _key("s-b", 8, 30), PROJECT) == "ldg-other"
    assert turn_active_ledger(tl, _key("s-b", 9, 5), PROJECT) == "ldg-w15"


def test_maintenance_write_without_content_change_does_not_rearm():
    """`bind-legacy-matters` 这类整本 dump、段不变的维护写不是"模型说做完了"。"""
    ev = _w15_events() + [
        _switch("ldg-other", _ts(8, 0), frm="ldg-w15"),
        _switch("ldg-w15", _ts(9, 0), frm="ldg-other"),
        _update(_led(verified=5).model_copy(update={"matter_id": "m-x"}), _ts(9, 30)),
    ]
    tl = switch_timeline(ev)
    assert turn_active_ledger(tl, _key("s-b", 10, 0), PROJECT) == "ldg-w15"


def test_verified_write_after_resume_rearms():
    """拍板文字「某次 ledger_update 之后 Open、Next 皆空 ⇒ 从这一刻起」：恢复后再写一次完结形态 ⇒ 再停。"""
    ev = _w15_events() + [
        _switch("ldg-other", _ts(8, 0), frm="ldg-w15"),
        _switch("ldg-w15", _ts(9, 0), frm="ldg-other"),
        _update(_led(verified=6), _ts(9, 30)),
    ]
    tl = switch_timeline(ev)
    assert turn_active_ledger(tl, _key("s-b", 9, 10), PROJECT) == "ldg-w15"
    assert turn_active_ledger(tl, _key("s-b", 10, 0), PROJECT) == ""


# ── ① close / supersede / reopen 重放 ─────────────────────────────────────


def _lifecycle(etype: str, lid: str, ts: datetime, **extra) -> dict:
    return _ev(etype, {"ledger_id": lid, "origin": "model", **extra}, ts)


def _two_ledgers() -> list[dict]:
    return [
        _ev(EVENT_LEDGER_CREATE, {"ledger": _led("ldg-a", next_=1).model_dump()}, _ts(1, 0)),
        _switch("ldg-a", _ts(1, 0)),
        _ev(EVENT_LEDGER_CREATE, {"ledger": _led("ldg-b", next_=1).model_dump()}, _ts(1, 5)),
    ]


def test_close_replays_status_and_releases_binding():
    ev = _two_ledgers() + [_lifecycle(EVENT_LEDGER_CLOSE, "ldg-a", _ts(2, 0))]
    pool, bindings = replay_ledger_events(ev)
    assert pool["ldg-a"].status == "closed"
    assert SCOPE not in bindings
    assert SCOPE not in binding_sessions(ev)


def test_close_ends_window_on_timeline():
    """判据 ③：关闭激活账本 ⇒ 之后的轮次不锚到它；之前照锚。"""
    ev = _two_ledgers() + [_lifecycle(EVENT_LEDGER_CLOSE, "ldg-a", _ts(2, 0))]
    tl = switch_timeline(ev)
    assert turn_active_ledger(tl, _key("s-a", 1, 30), PROJECT) == "ldg-a"
    assert turn_active_ledger(tl, _key("s-a", 2, 30), PROJECT) == ""


def test_close_does_not_cut_a_scope_already_on_another_ledger():
    ev = _two_ledgers() + [
        _switch("ldg-b", _ts(1, 30), frm="ldg-a"),
        _lifecycle(EVENT_LEDGER_CLOSE, "ldg-a", _ts(2, 0)),
    ]
    tl = switch_timeline(ev)
    assert turn_active_ledger(tl, _key("s-a", 2, 30), PROJECT) == "ldg-b"
    assert replay_ledger_events(ev)[1][SCOPE] == "ldg-b"


def test_supersede_chain_readable_and_reopen_does_not_activate():
    ev = _two_ledgers() + [
        _lifecycle(EVENT_LEDGER_SUPERSEDE, "ldg-a", _ts(2, 0), superseded_by="ldg-b"),
    ]
    pool, _ = replay_ledger_events(ev)
    assert pool["ldg-a"].status == "closed"
    assert pool["ldg-a"].superseded_by == "ldg-b"
    md = render_ledger_md(pool["ldg-a"])
    assert "- superseded by: `ldg-b`" in md
    assert parse_ledger_md(md).superseded_by == "ldg-b"     # 投影往返不丢链
    ev.append(_lifecycle(EVENT_LEDGER_REOPEN, "ldg-a", _ts(3, 0)))
    pool, bindings = replay_ledger_events(ev)
    assert pool["ldg-a"].status == "active"
    assert pool["ldg-a"].superseded_by == ""
    assert SCOPE not in bindings                                 # 重开 ≠ 激活
    assert turn_active_ledger(switch_timeline(ev), _key("s-a", 3, 30), PROJECT) == ""


def test_lifecycle_survives_full_replay_from_events_only():
    """判据 ①：状态只来自事件流——同一串事件重放两次得同一状态（重建等价）。"""
    ev = _two_ledgers() + [
        _lifecycle(EVENT_LEDGER_SUPERSEDE, "ldg-a", _ts(2, 0), superseded_by="ldg-b"),
        _lifecycle(EVENT_LEDGER_CLOSE, "ldg-b", _ts(2, 5)),
    ]
    p1, b1 = replay_ledger_events(ev)
    p2, b2 = replay_ledger_events(list(ev))
    assert {k: v.model_dump() for k, v in p1.items()} == {k: v.model_dump() for k, v in p2.items()}
    assert b1 == b2
    assert (p1["ldg-a"].status, p1["ldg-b"].status) == ("closed", "closed")


def test_lifecycle_events_are_ledger_events():
    assert {EVENT_LEDGER_CLOSE, EVENT_LEDGER_REOPEN, EVENT_LEDGER_SUPERSEDE} <= LEDGER_EVENT_TYPES


def test_close_before_tombstone_is_kept_after_is_dropped():
    """墓碑前的关本比墓碑更早结束窗口 ⇒ 保留；墓碑时刻未知 ⇒ 丢（切入已改写成空）。"""
    ev = _two_ledgers() + [_lifecycle(EVENT_LEDGER_CLOSE, "ldg-a", _ts(2, 0))]
    kept = apply_ledger_tombstones(ev, {"ldg-a"}, tombstone_ts_ms={"ldg-a": _ms(3, 0)})
    assert any(e["type"] == EVENT_LEDGER_CLOSE for e in kept)
    tl = switch_timeline(kept)
    assert turn_active_ledger(tl, _key("s-a", 2, 30), PROJECT) == ""
    dropped = apply_ledger_tombstones(ev, {"ldg-a"})
    assert not any(e["type"] == EVENT_LEDGER_CLOSE for e in dropped)


# ── ⑥ close 不改 Goal ────────────────────────────────────────────────────


def test_close_never_touches_goal_or_sections():
    led = _led(verified=2, next_=1)
    for out in (close_ledger(led), close_ledger(led, superseded_by="ldg-b"),
                reopen_ledger(close_ledger(led))):
        assert out.goal == led.goal
        assert out.goal_source == led.goal_source
        assert out.goal_revisions == led.goal_revisions
        assert out.sections == led.sections
