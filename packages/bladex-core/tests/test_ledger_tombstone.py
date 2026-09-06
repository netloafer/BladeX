"""账本墓碑（`task-ledger-tombstone-cleanup-20260901.md` §2.3）。

墓碑本身不难，难的是**switch 事件怎么处理**：直接丢弃会让前一本账本的激活窗口
向后延伸、吃掉本属于被墓碑账本的那段轮次——换个受害者，问题没解决
（MQ-L15「按终态给历史轮次归属 = 机制制造的误合并」的同一形状）。

故语义是 **`to_ledger_id=""` = 切到无账本**，`replay_ledger_events` 与
`switch_timeline` 都要认。本文件把"不许延伸"钉死，每条带判别力对照。
"""

from __future__ import annotations

from bladex_core.ledger import (
    EVENT_LEDGER_CREATE,
    EVENT_LEDGER_SWITCH,
    apply_ledger_tombstones,
    ledger_active_at,
    replay_ledger_events,
    switch_timeline,
)
from bladex_core.ledger_runtime import ActivationTable, activation_scope

AG = "claude-code"
SCOPE = activation_scope(AG, "")


def _create(lid: str, title: str = "t", ts: float = 0) -> dict:
    return {"type": EVENT_LEDGER_CREATE, "ts": ts,
            "payload": {"ledger": {"ledger_id": lid, "title": title,
                                   "goal": "g", "created_at": "2026-08-25T00:00:00+00:00",
                                   "updated_at": "2026-08-25T00:00:00+00:00"}}}


#: 🔴 事件 ts 按**秒**给（`_event_ts_ms` 对 <1e11 的值按 epoch 秒转毫秒），
#: 查询则按**毫秒**。混用会让断言在一个与被测逻辑无关的地方失败——
#: 首版就栽在这里（查 150ms，而事件在 100000ms）。
def _ms(sec: float) -> float:
    return sec * 1000.0


def _switch(lid: str, ts: float) -> dict:
    return {"type": EVENT_LEDGER_SWITCH, "ts": ts,
            "payload": {"agent_id": AG, "project_id": "", "session_id": "s",
                        "from_ledger_id": "", "to_ledger_id": lid}}


# ── 事件过滤 ──────────────────────────────────────────────────────────────


def test_create_of_tombstoned_ledger_is_dropped() -> None:
    evs = [_create("ldg-a"), _create("ldg-b")]
    out = apply_ledger_tombstones(evs, {"ldg-a"})
    assert [e["payload"]["ledger"]["ledger_id"] for e in out] == ["ldg-b"]


def test_switch_to_tombstoned_is_rewritten_not_dropped() -> None:
    """🔴 本卡的核心：改写成"切到空"，**不是**丢掉整条事件。"""
    out = apply_ledger_tombstones([_switch("ldg-a", 100)], {"ldg-a"})
    assert len(out) == 1, "事件被丢掉了 —— 前一本会延伸，正是要防的那件事"
    assert out[0]["payload"]["to_ledger_id"] == ""
    assert out[0]["payload"]["agent_id"] == AG, "scope 信息必须保留，否则定位不到"
    assert out[0]["ts"] == 100, "ts 必须保留，否则时间轴位置就丢了"


def test_input_is_not_mutated() -> None:
    evs = [_switch("ldg-a", 1)]
    apply_ledger_tombstones(evs, {"ldg-a"})
    assert evs[0]["payload"]["to_ledger_id"] == "ldg-a"


def test_empty_tombstone_set_is_identity() -> None:
    evs = [_create("ldg-a"), _switch("ldg-a", 1)]
    assert apply_ledger_tombstones(evs, set()) == evs


# ── 🔴 硬点：不许让前一本延伸 ─────────────────────────────────────────────


def test_previous_ledger_must_not_extend_into_the_tombstoned_window() -> None:
    """A 活跃 → 切到 B（被墓碑）→ 切到 C。

    B 的那段窗口必须变成"无账本"，**不能让 A 继续生效**——那只是把
    105 条错锚从 B 挪到 A。
    """
    evs = [_switch("ldg-A", 100), _switch("ldg-B", 200), _switch("ldg-C", 300)]
    tl = switch_timeline(apply_ledger_tombstones(evs, {"ldg-B"}))
    assert ledger_active_at(tl, SCOPE, _ms(150)) == "ldg-A"      # B 之前仍是 A
    assert ledger_active_at(tl, SCOPE, _ms(250)) == "", "🔴 B 的窗口被 A 吃掉了"
    assert ledger_active_at(tl, SCOPE, _ms(350)) == "ldg-C"      # C 之后不受影响


def test_discriminating_power_dropping_the_event_would_extend_A() -> None:
    """判别力对照：若把 switch 整条丢掉（旧的天真做法），250 时刻会读成 A。"""
    evs = [_switch("ldg-A", 100), _switch("ldg-B", 200), _switch("ldg-C", 300)]
    naive = [e for e in evs if e["payload"]["to_ledger_id"] != "ldg-B"]
    assert ledger_active_at(switch_timeline(naive), SCOPE, _ms(250)) == "ldg-A"


def test_timeline_keeps_the_empty_marker() -> None:
    """`switch_timeline` 此前 `if not to: continue` 会把标记吞掉。"""
    tl = switch_timeline(apply_ledger_tombstones([_switch("ldg-B", 200)], {"ldg-B"}))
    assert tl.get(SCOPE) == [(_ms(200), "")]


def test_timeline_still_skips_events_without_agent() -> None:
    """agent 缺失仍要跳过——那条事件定不了 scope，保留它没有意义。"""
    ev = {"type": EVENT_LEDGER_SWITCH, "ts": 1,
          "payload": {"to_ledger_id": "", "session_id": "s"}}
    assert switch_timeline([ev]) == {}


# ── 终态绑定 ─────────────────────────────────────────────────────────────


def test_binding_is_cleared_not_left_at_the_previous_ledger() -> None:
    """注入侧同理：被墓碑账本原本是激活的 ⇒ 该 scope 变成**无激活账本**。"""
    evs = [_create("ldg-A"), _create("ldg-B"),
           _switch("ldg-A", 100), _switch("ldg-B", 200)]
    _pool, bindings = replay_ledger_events(apply_ledger_tombstones(evs, {"ldg-B"}))
    assert bindings.get(SCOPE, "") == "", "🔴 绑定退回了 A —— 前一本又活了"


def test_pool_drops_the_tombstoned_ledger() -> None:
    evs = [_create("ldg-A"), _create("ldg-B")]
    pool, _ = replay_ledger_events(apply_ledger_tombstones(evs, {"ldg-B"}))
    assert set(pool) == {"ldg-A"}


# ── 活着的进程内表 ────────────────────────────────────────────────────────


def test_forget_ledger_clears_every_scope_holding_it() -> None:
    t = ActivationTable(debounce_turns=0)
    t.restore({activation_scope("a", ""): "ldg-X",
               activation_scope("b", ""): "ldg-X",
               activation_scope("c", ""): "ldg-Y"})
    freed = t.forget_ledger("ldg-X")
    assert sorted(freed) == sorted([activation_scope("a", ""), activation_scope("b", "")])
    assert t.active(activation_scope("a", "")) == ""
    assert t.active(activation_scope("c", "")) == "ldg-Y", "别的账本不受影响"


def test_forget_ledger_is_a_noop_for_unknown_id() -> None:
    t = ActivationTable(debounce_turns=0)
    t.restore({SCOPE: "ldg-X"})
    assert t.forget_ledger("ldg-nope") == []
    assert t.active(SCOPE) == "ldg-X"
