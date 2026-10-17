"""MQ-L85（批 Q3A）：`ledger_create` 事件带写者四格（agent_id / project_id / session_id / turn_index）。

归档账 L17123：85/85 条 create 载荷没有作者；批 T0 补了 `agent_id` / `project_id`（半修），
`session_id` / `turn_index` 仍缺——而 `ledger_update` 从 Z3.2 起就带齐四格（`_writer_fields`）。
修后 create 与 update 走同一个 `_writer_fields(context)`，同源同值。存量不回填：老事件
（只有 `{"ledger": …}`，或只有 agent/project）重放不报错、读数不变。
"""

from __future__ import annotations

import asyncio

import pytest
from bladex_core.ledger import (
    EVENT_LEDGER_CREATE,
    EVENT_LEDGER_SWITCH,
    new_ledger,
    replay_ledger_events,
    switch_timeline,
)
from bladex_core.ledger_runtime import activation_scope
from bladex_proxy.agency import AgencyRuntime

AGENT = "claude-code"
PROJECT = "p-l85test00001"


@pytest.fixture(autouse=True)
def _modules_on(monkeypatch):
    for m in ("TOOLFACE", "LEDGER", "INTERCEPTION"):
        monkeypatch.setenv(f"BLADEX_MODULE_{m}", "1")


def _create(turn_index: int = 5) -> list[tuple[str, dict]]:
    ag = AgencyRuntime(hub=None)
    events: list[tuple[str, dict]] = []
    ag._emit = lambda et, p: events.append((getattr(et, "value", str(et)), p))  # noqa: SLF001
    ctx = {"session_id": "sess-l85", "agent_id": AGENT, "project_id": PROJECT,
           "turn_index": turn_index, "user_query": "帮我整理泰山啤酒重整进展"}
    out = asyncio.run(ag._h_ledger_switch({"title": "泰山啤酒重整进展", "goal": "梳理到可核对"},
                                          allowed_exposure="public", context=ctx))
    assert not out.startswith("Error"), out
    return events


def test_create_payload_carries_the_four_writer_cells():
    events = _create(turn_index=5)
    (create,) = [p for t, p in events if t == EVENT_LEDGER_CREATE]
    assert (create["agent_id"], create["project_id"], create["session_id"],
            create["turn_index"]) == (AGENT, PROJECT, "sess-l85", 5)


def test_create_and_following_switch_share_the_same_source():
    """同源同值：create 的写者与紧随其后的 switch 事件出自同一个 tool_context。"""
    events = _create(turn_index=9)
    (create,) = [p for t, p in events if t == EVENT_LEDGER_CREATE]
    (switch,) = [p for t, p in events if t == EVENT_LEDGER_SWITCH]
    for k in ("agent_id", "session_id", "turn_index"):
        assert create[k] == switch[k], k


@pytest.mark.parametrize("payload_extra", [
    {},                                              # 批 T0 之前：只有 ledger
    {"agent_id": AGENT, "project_id": PROJECT},      # 批 T0 之后、本批之前：半修
], ids=["pre_t0", "t0_half"])
def test_old_create_events_replay_without_error(payload_extra):
    led = new_ledger(ledger_id="ldg-l85old000001", title="老本", goal="g")
    ev = {"type": EVENT_LEDGER_CREATE, "ts": "2026-09-01T00:00:00+00:00",
          "payload": {"ledger": led.model_dump(mode="json"), **payload_extra}}
    pool, _act = replay_ledger_events([ev])[:2]
    assert "ldg-l85old000001" in pool
    tl = switch_timeline([ev])
    if payload_extra:
        assert tl[activation_scope(AGENT, PROJECT)] and tl.done_spans == {}
    else:
        assert dict(tl) == {}
