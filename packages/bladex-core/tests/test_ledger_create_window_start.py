"""批 T0（2026-09-22，MQ-I22 连带）：「建本即激活」也是 `switch_timeline` 的窗口起点。

# 为什么需要它

`ledger_create` 之后紧跟一条切到新本的 switch（Y1c 前靠 newly_created 豁免防抖；全新 id 又不可能
noop），所以两个起点**今天恒等**——但那是实现巧合，不是结构保证。switch 一旦发不出来
（`agency_switch_unknown_ledger` / MQ-L80 那一族形态），账本就只剩一条 create 事件、
窗口整段消失，`--with-memory` 因此打 0 条却报成功（MQ-I22 同族）。

# 🔴 只认载荷里带 scope 的 create

存量 85/85 条 `ledger_create` 的载荷只有 `{"ledger": …}`，**没有 agent_id / project_id**
（`last_writer` 也 85/85 为空，MQ-L85）。没有 scope 就定不了在哪条轴上开窗——硬猜正是
原则 13 层 4 要防的"值不来自它声称的来源"。故老事件一律跳过（live 实测：时间轴逐字不变、
20,097 轮逐轮归属零差异），生产侧自本批起在 create 载荷补 scope。
"""

from __future__ import annotations

from bladex_core.ledger import (
    EVENT_LEDGER_CREATE,
    EVENT_LEDGER_SWITCH,
    apply_ledger_tombstones,
    ledger_activation_windows,
    ledger_active_at,
    switch_timeline,
)
from bladex_core.ledger_runtime import activation_scope

_A = "claude-code"
_P = "p-e479f97c661a"
_L = "ldg-t0create01"
SCOPE = activation_scope(_A, _P)


def _ms(sec: float) -> float:
    """事件 `ts` 传 epoch 秒（`_event_ts_ms` 按量级归一），判据里用毫秒对照。"""
    return sec * 1000.0


def _create(lid: str, sec: float, *, agent: str = "", project: str = "") -> dict:
    payload: dict = {"ledger": {"ledger_id": lid, "title": "t"}}
    if agent:
        payload["agent_id"] = agent
        payload["project_id"] = project
    return {"type": EVENT_LEDGER_CREATE, "ts": sec, "payload": payload}


def _switch(lid: str, sec: float, *, agent: str = _A, project: str = _P) -> dict:
    return {"type": EVENT_LEDGER_SWITCH, "ts": sec,
            "payload": {"agent_id": agent, "project_id": project, "session_id": "s1",
                        "turn_index": 1, "to_ledger_id": lid}}


def test_1_create_with_scope_opens_window_without_any_switch() -> None:
    """🔴 T0.2 本体：只有 create、没有 switch ⇒ 窗口仍非空。"""
    tl = switch_timeline([_create(_L, 100, agent=_A, project=_P)])
    assert tl == {SCOPE: [(_ms(100), _L)]}
    assert ledger_activation_windows(tl, _L, until_ms=_ms(500)) == {SCOPE: [(_ms(100), _ms(500))]}


def test_2_negative_control_create_without_scope_is_invisible() -> None:
    """阴性对照 = 存量形态（85/85）：载荷无 agent_id ⇒ 时间轴零变化。"""
    assert switch_timeline([_create(_L, 100)]) == {}
    assert ledger_activation_windows(switch_timeline([_create(_L, 100)]), _L) == {}


def test_3_create_plus_switch_is_idempotent_on_the_axis() -> None:
    """建本 + 紧随其后的 switch（今天的常态）⇒ 窗口起点仍是建本那一刻，归属不变。"""
    evs = [_create(_L, 100, agent=_A, project=_P), _switch(_L, 100.001)]
    tl = switch_timeline(evs)
    assert ledger_activation_windows(tl, _L, until_ms=_ms(500)) == {SCOPE: [(_ms(100), _ms(500))]}
    assert ledger_active_at(tl, SCOPE, _ms(100)) == _L
    assert ledger_active_at(tl, SCOPE, _ms(99)) == "", "建本之前仍是无账本（不回填历史）"


def test_4_create_scope_is_the_real_scope_not_global() -> None:
    """判别力：project 写进 scope，不是落 Global。"""
    tl = switch_timeline([_create(_L, 100, agent=_A, project=_P)])
    assert activation_scope(_A, "") not in tl
    assert SCOPE in tl


def test_5_tombstoned_ledger_create_does_not_reopen_the_window() -> None:
    """🔴 墓碑语义优先：被墓碑账本的 create 被 `apply_ledger_tombstones` 丢掉，窗口不复活。"""
    evs = [_create(_L, 100, agent=_A, project=_P), _switch(_L, 100.001)]
    out = apply_ledger_tombstones(evs, {_L}, tombstone_ts_ms={_L: _ms(300)})
    tl = switch_timeline(out)
    assert ledger_activation_windows(tl, _L, until_ms=_ms(900)) == {SCOPE: [(_ms(100.001), _ms(300))]}, \
        "窗口在墓碑时刻闭合（MQ-I19），起点是保留下来的切入事件"
    # 墓碑时刻未知 ⇒ 旧语义（切入被改写成"切到空"）⇒ 窗口必须整段消失，create 不得把它顶回来
    out2 = apply_ledger_tombstones(evs, {_L})
    assert ledger_activation_windows(switch_timeline(out2), _L, until_ms=_ms(900)) == {}


def test_6_create_for_another_ledger_does_not_steal_the_window() -> None:
    """建 B 却切到 C（假想形态）：B 只得到一个零长窗口，C 的窗口不被吃。"""
    evs = [_switch("ldg-A", 50), _create("ldg-B", 100, agent=_A, project=_P),
           _switch("ldg-C", 100.002)]
    tl = switch_timeline(evs)
    assert ledger_activation_windows(tl, "ldg-A", until_ms=_ms(900))[SCOPE] == [(_ms(50), _ms(100))]
    assert ledger_activation_windows(tl, "ldg-C", until_ms=_ms(900))[SCOPE] == [(_ms(100.002), _ms(900))]
    assert ledger_active_at(tl, SCOPE, _ms(500)) == "ldg-C"
