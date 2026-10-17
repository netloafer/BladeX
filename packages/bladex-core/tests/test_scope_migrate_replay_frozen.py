"""批 Y · Y1d 判据②：含历史 `ledger_scope_migrate` 事件的时间轴重放**逐字等于改前**。

拍板 (30)（2026-09-26）停掉惰性作用域迁移的**生产**（`agency/handlers._maybe_migrate_binding`
已删），但 Hub 里已写入的 4 条迁移事件是当时运行时的真实行为——ADR-0009 §8b.3
「重放 = 运行时」⇒ `ledger.py` 的重放分支一字不动。

期望值是 **改前 HEAD（`646f8cb`）上跑同一夹具得到的输出**（原样冻结，不手抄推理），
夹具按 T1 §7a 的两种真实形状造：

  A  Global→同项目：codex@Global 绑着 BladeX 本，首次进 `p-bladex` 被搬；
  B  跨项目：codex@Global 绑着 BladeX 本，首次进 g3 bench（`p-g3`）被搬走，69 s 后
     模型在 g3 切到自己的本，Global 上随后建新本；另有一个更早的 `p-other` 切换
     （让 project 维「已生效」，见 `project_scope_since`）。

阴性对照：删掉 `switch_timeline` 里 `EVENT_LEDGER_SCOPE_MIGRATE` 那一支 ⇒ A 的
`t+20000|p-bladex` 与 B 的 `t+20000|p-g3` 变空、Global 侧不再清空 ⇒ 红。
"""

from __future__ import annotations

from bladex_core.ledger import (
    EVENT_LEDGER_SCOPE_MIGRATE,
    EVENT_LEDGER_SWITCH,
    binding_sessions,
    replay_ledger_events,
    switch_timeline,
)
from bladex_core.ledger_runtime import turn_active_ledger

T0 = 1_790_000_000_000.0
G = "codex\x1fGlobal"


def _sw(ts: float, to: str, project: str = "", session: str = "s0") -> dict:
    return {"type": EVENT_LEDGER_SWITCH, "ts": ts,
            "payload": {"agent_id": "codex", "project_id": project,
                        "session_id": session, "to_ledger_id": to}}


def _mig(ts: float, lid: str, frm: str, to: str) -> dict:
    return {"type": EVENT_LEDGER_SCOPE_MIGRATE, "ts": ts,
            "payload": {"agent_id": "codex", "from_project": frm,
                        "to_project": to, "ledger_id": lid}}


def _key(ts: float) -> str:
    return f"u1/codex/s1/{int(ts)}-0"


FIXTURE_A = [_sw(T0, "ldg-bladex"), _mig(T0 + 10_000, "ldg-bladex", "", "p-bladex")]
FIXTURE_B = [_sw(T0 - 60_000, "ldg-other", "p-other"),
             _sw(T0, "ldg-bladex"),
             _mig(T0 + 10_000, "ldg-bladex", "", "p-g3"),
             _sw(T0 + 40_000, "ldg-g3own", "p-g3", session="s1"),
             _sw(T0 + 50_000, "ldg-bladex2", "", session="s2")]

_PROBES = (-30_000, 5_000, 20_000, 45_000, 55_000)
_PROJECTS = ("", "p-bladex", "p-g3", "p-other")

# ── 改前 HEAD 646f8cb 的原样输出（脚本 `baseline_replay.py`，2026-09-28）──
EXPECTED_A = {
    "timeline": {G: [(T0, "ldg-bladex"), (T0 + 10_000, "")],
                 "codex\x1fp-bladex": [(T0 + 10_000, "ldg-bladex")]},
    "turn_active_ledger": {
        "t-30000|Global": "", "t-30000|p-bladex": "", "t-30000|p-g3": "", "t-30000|p-other": "",
        "t+5000|Global": "ldg-bladex", "t+5000|p-bladex": "ldg-bladex",
        "t+5000|p-g3": "ldg-bladex", "t+5000|p-other": "ldg-bladex",
        "t+20000|Global": "", "t+20000|p-bladex": "ldg-bladex", "t+20000|p-g3": "", "t+20000|p-other": "",
        "t+45000|Global": "", "t+45000|p-bladex": "ldg-bladex", "t+45000|p-g3": "", "t+45000|p-other": "",
        "t+55000|Global": "", "t+55000|p-bladex": "ldg-bladex", "t+55000|p-g3": "", "t+55000|p-other": "",
    },
    "bindings": {"codex\x1fp-bladex": "ldg-bladex"},
    "binding_sessions": {},
}
EXPECTED_B = {
    "timeline": {G: [(T0, "ldg-bladex"), (T0 + 10_000, ""), (T0 + 50_000, "ldg-bladex2")],
                 "codex\x1fp-g3": [(T0 + 10_000, "ldg-bladex"), (T0 + 40_000, "ldg-g3own")],
                 "codex\x1fp-other": [(T0 - 60_000, "ldg-other")]},
    "turn_active_ledger": {
        "t-30000|Global": "", "t-30000|p-bladex": "", "t-30000|p-g3": "", "t-30000|p-other": "ldg-other",
        "t+5000|Global": "ldg-bladex", "t+5000|p-bladex": "", "t+5000|p-g3": "", "t+5000|p-other": "ldg-other",
        "t+20000|Global": "", "t+20000|p-bladex": "", "t+20000|p-g3": "ldg-bladex", "t+20000|p-other": "ldg-other",
        "t+45000|Global": "", "t+45000|p-bladex": "", "t+45000|p-g3": "ldg-g3own", "t+45000|p-other": "ldg-other",
        "t+55000|Global": "ldg-bladex2", "t+55000|p-bladex": "", "t+55000|p-g3": "ldg-g3own",
        "t+55000|p-other": "ldg-other",
    },
    "bindings": {"codex\x1fp-other": "ldg-other", "codex\x1fp-g3": "ldg-g3own", G: "ldg-bladex2"},
    "binding_sessions": {"codex\x1fp-other": "s0", "codex\x1fp-g3": "s1", G: "s2"},
}


def _probe(events: list[dict]) -> dict:
    tl = switch_timeline(events)
    q = {f"t{dt:+d}|{proj or 'Global'}": turn_active_ledger(tl, _key(T0 + dt), proj)
         for dt in _PROBES for proj in _PROJECTS}
    _, bindings = replay_ledger_events(events)
    return {"timeline": {k: [tuple(x) for x in v] for k, v in tl.items()},
            "turn_active_ledger": q, "bindings": bindings,
            "binding_sessions": binding_sessions(events)}


def test_global_to_project_migration_replays_exactly_as_before() -> None:
    assert _probe(FIXTURE_A) == EXPECTED_A


def test_cross_project_migration_replays_exactly_as_before() -> None:
    assert _probe(FIXTURE_B) == EXPECTED_B


def test_migrate_event_type_still_registered() -> None:
    """事件类型登记不动：重放要认得它，`LEDGER_EVENT_TYPES` 里必须还在。"""
    from bladex_core.ledger import LEDGER_EVENT_TYPES
    assert EVENT_LEDGER_SCOPE_MIGRATE in LEDGER_EVENT_TYPES
