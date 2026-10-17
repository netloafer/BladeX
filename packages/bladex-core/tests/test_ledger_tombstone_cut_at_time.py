"""批 S · S2.1（MQ-I19）：账本墓碑**从发生时刻起生效**，不倒填到切入时刻。

旧语义把"切到被墓碑账本"的 switch 改写成"切到空"——账本存活期整段从时间轴消失，
窗口内的轮次在重建 / 滞后蒸馏时查不到激活账本（批 S 实测 live 232 条 fact 落内容卡，
Q3 真跑 S3「换卡长回来」同一成因）。新语义：已知墓碑时刻 ⇒ 切入保留 + 墓碑时刻合成
「切到空」；未知时刻 ⇒ 回落旧语义。每条带判别力对照（去掉 `tombstone_ts_ms` 就是旧行为）。
"""

from __future__ import annotations

from bladex_core.ledger import (
    EVENT_LEDGER_CREATE,
    EVENT_LEDGER_SWITCH,
    apply_ledger_tombstones,
    ledger_activation_windows,
    ledger_active_at,
    replay_ledger_events,
    switch_timeline,
)
from bladex_core.ledger_runtime import activation_scope

AG = "claude-code"
SCOPE = activation_scope(AG, "")
L = "ldg-tombstone0001"
M = "ldg-other0000002"


def _create(lid: str, ts: float) -> dict:
    return {"type": EVENT_LEDGER_CREATE, "ts": ts,
            "payload": {"ledger": {"ledger_id": lid, "title": f"title-{lid}",
                                   "goal": "g", "created_at": "2026-09-01T00:00:00+00:00",
                                   "updated_at": "2026-09-01T00:00:00+00:00"}}}


def _switch(lid: str, ts: float, agent: str = AG) -> dict:
    return {"type": EVENT_LEDGER_SWITCH, "ts": ts,
            "payload": {"agent_id": agent, "project_id": "", "session_id": "s", "to_ledger_id": lid}}


#: 事件 ts 按秒给（`_event_ts_ms` <1e11 按秒转毫秒），查询按毫秒。
def _ms(sec: float) -> float:
    return sec * 1000.0


def test_tombstone_keeps_window_before_and_empties_after() -> None:
    """切入 100s，墓碑 200s ⇒ 150s 在场、250s 无账本；终态绑定为空。"""
    events = [_create(L, 90), _switch(L, 100)]
    out = apply_ledger_tombstones(events, {L}, tombstone_ts_ms={L: _ms(200)})
    tl = switch_timeline(out)
    assert ledger_active_at(tl, SCOPE, _ms(150)) == L
    assert ledger_active_at(tl, SCOPE, _ms(250)) == ""
    pool, bindings = replay_ledger_events(out)
    assert L not in pool, "池仍不含被墓碑账本（agency/flash 零变化）"
    assert SCOPE not in bindings, "墓碑那一刻仍激活 ⇒ 合成的空切换把绑定 pop 掉"
    # 合成事件可 grep：reason 与 from_ledger_id 都在
    syn = [e for e in out if e["type"] == EVENT_LEDGER_SWITCH and e["payload"].get("to_ledger_id") == ""]
    assert len(syn) == 1 and syn[0]["payload"]["reason"] == "ledger_tombstone"
    assert syn[0]["payload"]["from_ledger_id"] == L


def test_negative_control_without_timestamp_falls_back_to_old_semantics() -> None:
    """判别力对照：不传时刻 ⇒ 旧语义（切入改写为空，窗口整段消失）。"""
    events = [_create(L, 90), _switch(L, 100)]
    out = apply_ledger_tombstones(events, {L})
    tl = switch_timeline(out)
    assert ledger_active_at(tl, SCOPE, _ms(150)) == ""
    out2 = apply_ledger_tombstones(events, {L}, tombstone_ts_ms={L: 0.0})
    assert ledger_active_at(switch_timeline(out2), SCOPE, _ms(150)) == "", "时刻 0.0 = 未知 ⇒ 旧语义"


def test_scope_that_moved_on_before_tombstone_is_untouched() -> None:
    """L(100s) → M(150s)，L 于 200s 墓碑：M 的窗口不得被合成事件切断（MQ-L15 同型）。"""
    events = [_create(L, 90), _create(M, 91), _switch(L, 100), _switch(M, 150)]
    out = apply_ledger_tombstones(events, {L}, tombstone_ts_ms={L: _ms(200)})
    tl = switch_timeline(out)
    assert ledger_active_at(tl, SCOPE, _ms(120)) == L
    assert ledger_active_at(tl, SCOPE, _ms(250)) == M
    _pool, bindings = replay_ledger_events(out)
    assert bindings[SCOPE] == M
    assert not [e for e in out if e["payload"].get("reason") == "ledger_tombstone"], "无需合成"


def test_switch_after_tombstone_treated_as_switch_to_empty() -> None:
    """切入发生在墓碑之后（陈旧/错序事件）⇒ 仍按旧语义改写为空，不给死账本开窗口。"""
    events = [_create(L, 90), _switch(L, 300)]
    out = apply_ledger_tombstones(events, {L}, tombstone_ts_ms={L: _ms(200)})
    tl = switch_timeline(out)
    assert ledger_active_at(tl, SCOPE, _ms(350)) == ""


def test_later_real_switch_after_tombstone_survives_replay_order() -> None:
    """L(100s)，墓碑 200s，之后真切到 M(300s)：合成事件必须插在 300s 之前，M 的绑定不能被 pop。"""
    events = [_create(L, 90), _create(M, 91), _switch(L, 100), _switch(M, 300)]
    out = apply_ledger_tombstones(events, {L}, tombstone_ts_ms={L: _ms(200)})
    _pool, bindings = replay_ledger_events(out)
    assert bindings[SCOPE] == M
    tl = switch_timeline(out)
    assert ledger_active_at(tl, SCOPE, _ms(250)) == ""
    assert ledger_active_at(tl, SCOPE, _ms(350)) == M


def test_two_scopes_only_active_one_gets_synthesized_switch() -> None:
    """两个 agent 都切过 L；墓碑时只有一个仍在 L 上 ⇒ 只给那个 scope 合成。"""
    other = "codex"
    events = [_create(L, 90), _create(M, 91),
              _switch(L, 100), _switch(L, 100, agent=other), _switch(M, 150, agent=other)]
    out = apply_ledger_tombstones(events, {L}, tombstone_ts_ms={L: _ms(200)})
    syn = [e for e in out if e["payload"].get("reason") == "ledger_tombstone"]
    assert [e["payload"]["agent_id"] for e in syn] == [AG]


def test_activation_windows_follow_the_same_timeline() -> None:
    """窗口 = 切入 → 下一次切换（含合成的空切换）；仍激活的段 end=None / until。"""
    events = [_create(L, 90), _create(M, 91), _switch(L, 100), _switch(M, 150), _switch(L, 300)]
    tl = switch_timeline(apply_ledger_tombstones(events, {L}, tombstone_ts_ms={L: _ms(400)}))
    win = ledger_activation_windows(tl, L)
    assert win == {SCOPE: [(_ms(100), _ms(150)), (_ms(300), _ms(400))]}
    tl_live = switch_timeline(events)
    assert ledger_activation_windows(tl_live, L, until_ms=_ms(999)) == {
        SCOPE: [(_ms(100), _ms(150)), (_ms(300), _ms(999))]}
    assert ledger_activation_windows(tl_live, L)[SCOPE][-1] == (_ms(300), None)
    assert ledger_activation_windows(tl_live, "ldg-never") == {}
