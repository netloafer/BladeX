"""批 PW · 步骤 2（拍板 (98)，MQ-L121 / L122 / L123）· 子本回写父本不必 switch。

`bladex_ledger_update(ledger_id=<父>)`：只允许写「激活本 Next 里 `[sub-task]` 条目的 ref」指向的那一本，
激活位不动；rev 锁 / Goal 保护按目标本；父本已关写入后仍 closed、不重开。设计与读数见
`docs/planning/report-batch-PW-20261112.md` §3–§4、§6。

反向变异（每条对应一个用例，实测结果记在结算稿 §6）：见各用例 docstring。
"""

from __future__ import annotations

import asyncio
import json

import pytest
import structlog.testing
from bladex_core.ledger_runtime import activation_scope
from bladex_proxy import toolface
from bladex_proxy.agency import AgencyRuntime
from bladex_proxy.agency.handlers import SUBTASK_TAG

AGENT = "dsh"
PROJECT = "p-pwtest000001"
SCOPE = activation_scope(AGENT, PROJECT)


@pytest.fixture(autouse=True)
def _modules_on(monkeypatch):
    for m in ("TOOLFACE", "LEDGER", "INTERCEPTION"):
        monkeypatch.setenv(f"BLADEX_MODULE_{m}", "1")


class _Rec:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    def __call__(self, etype, payload):
        self.events.append((getattr(etype, "value", str(etype)), payload))

    def kinds(self, start: int = 0) -> list[str]:
        return [t for t, _ in self.events[start:]]


def _ctx(session: str, turn_index: int, user: str) -> dict:
    return {"session_id": session, "agent_id": AGENT, "project_id": PROJECT,
            "turn_index": turn_index, "user_query": user}


def _tool(ag: AgencyRuntime, name: str, args: dict, ctx: dict) -> str:
    fn = {"switch": ag._h_ledger_switch, "update": ag._h_ledger_update,   # noqa: SLF001
          "close": ag._h_ledger_close}[name]                              # noqa: SLF001
    return asyncio.run(fn(args, allowed_exposure="public", context=ctx))


def _setup(close_parent: bool = False) -> tuple[AgencyRuntime, _Rec, str, str]:
    """父本（会话 s1 建）→ 子本（会话 s2 显式挂父）。激活 = 子本。返回 (ag, rec, parent, child)。"""
    ag = AgencyRuntime(hub=None)
    rec = _Rec()
    ag._emit = rec  # noqa: SLF001 —— 只替换事件出口，判定路径原样
    _tool(ag, "switch", {"ledger_id": "", "title": "父任务", "goal": "修 A92",
                         "core": ["A92 的根因在压缩引擎"]}, _ctx("s1", 1, "修 A92"))
    parent = ag.activation.active(SCOPE)
    if close_parent:
        assert "closed" in _tool(ag, "close", {}, _ctx("s1", 2, "A92 做完了"))
    _tool(ag, "switch", {"ledger_id": "", "title": "子任务", "goal": "修 A93",
                         "parent_ledger_id": parent}, _ctx("s2", 1, "顺手把 A93 修了"))
    child = ag.activation.active(SCOPE)
    assert child not in ("", parent)
    return ag, rec, parent, child


def _wb_args(child: str, parent: str) -> dict:
    return {"ledger_id": parent, "entries": [
        {"section": "verified", "op": "add", "text": f"[sub-task] {child} done: A93 规则已落"}]}


# ── 1 回写落在父本、激活位不动、无 switch ───────────────────────────────────


def test_1_writeback_via_update_writes_parent_and_keeps_activation():
    """变异：把 handler 的目标改回 `_ledger_for(context)` ⇒ 条目写进子本，本用例红。"""
    ag, rec, parent, child = _setup()
    mark = len(rec.events)
    rev0, crev0 = ag.pool[parent].rev, ag.pool[child].rev
    out = _tool(ag, "update", _wb_args(child, parent), _ctx("s2", 2, "做完了"))
    assert not out.startswith("Error"), out
    p = ag.pool[parent]
    assert p.rev == rev0 + 1
    assert [e for e in p.entries("verified") if child in e.text]
    assert ag.activation.active(SCOPE) == child                  # 激活位不动
    assert ag.pool[child].rev == crev0                           # 子本没被写
    kinds = rec.kinds(mark)
    assert "ledger_switch" not in kinds and kinds.count("ledger_update") == 1
    upd = [pl for t, pl in rec.events[mark:] if t == "ledger_update"][0]
    assert upd["ledger"]["ledger_id"] == parent


# ── 2 非子本义务指向的本：拒绝、零写入 ───────────────────────────────────────


def test_2_non_parent_ledger_id_refused_without_write():
    """变异：去掉 ref 校验 ⇒ 写入成功，本用例红。"""
    ag, rec, parent, child = _setup()
    _tool(ag, "switch", {"ledger_id": "", "title": "无关本", "goal": "别的事"}, _ctx("s3", 1, "别的事"))
    other = ag.activation.active(SCOPE)
    _tool(ag, "switch", {"ledger_id": child}, _ctx("s2", 3, "回到 A93"))
    mark = len(rec.events)
    revs = {k: v.rev for k, v in ag.pool.items()}
    with structlog.testing.capture_logs() as cap:
        out = _tool(ag, "update", _wb_args(child, other), _ctx("s2", 3, "回到 A93"))
    assert out.startswith("Error") and "[sub-task]" in out
    assert {k: v.rev for k, v in ag.pool.items()} == revs
    assert rec.kinds(mark) == []
    ref = [e for e in cap if e["event"] == "agency_ledger_update_refused"]
    assert len(ref) == 1 and ref[0]["reason"] == "not_subtask_parent" and ref[0]["target"] == other
    # 无激活本时给 ledger_id ⇒ 照旧 no_active_ledger（不绕过）
    ag2 = AgencyRuntime(hub=None)
    assert _tool(ag2, "update", {"ledger_id": parent, "section": "next", "op": "add", "text": "x"},
                 _ctx("s9", 1, "x")).startswith("Error: no active ledger")


# ── 3 父本已关：写入后仍 closed、不重开 ──────────────────────────────────────


def test_3_closed_parent_stays_closed_and_not_reopened():
    """变异：在 ledger_id 写入路径加 reopen ⇒ 本用例红。"""
    ag, rec, parent, child = _setup(close_parent=True)
    orig = ag.pool[parent]
    orig_info = (orig.status, orig.closed_at, orig.closed_by_agent, orig.closed_by_session)
    mark = len(rec.events)
    out = _tool(ag, "update", _wb_args(child, parent), _ctx("s2", 2, "做完了"))
    assert not out.startswith("Error"), out
    p = ag.pool[parent]
    assert (p.status, p.closed_at, p.closed_by_agent, p.closed_by_session) == orig_info
    assert [e for e in p.entries("verified") if child in e.text]
    kinds = rec.kinds(mark)
    assert "ledger_reopen" not in kinds and "ledger_switch" not in kinds
    assert ag.activation.active(SCOPE) == child


# ── 4 rev 锁按目标本 ────────────────────────────────────────────────────────


def test_4_rev_lock_is_judged_on_target_ledger():
    """变异：rev 比对改回激活本 ⇒ 落后 rev 的 remove 不再被拒（或错拒），本用例红。"""
    ag, _rec, parent, child = _setup()
    # 父本先被别人写过一次，rev 前进
    _tool(ag, "update", _wb_args(child, parent), _ctx("s2", 2, "做完了"))
    cur = ag.pool[parent].rev
    assert cur != ag.pool[child].rev                      # 两本 rev 不同，才分得出按谁判
    stale = {"ledger_id": parent, "rev": cur - 1,
             "entries": [{"op": "remove", "match": f"[sub-task] {child} done"}]}
    out = _tool(ag, "update", stale, _ctx("s2", 3, "x"))
    assert out.startswith("Error: ledger changed since you read it")
    assert f"current {cur}" in out and "父任务" in out        # 返回的是父本最新正文
    assert ag.pool[parent].rev == cur
    # 带对 rev 的 remove 通过
    ok = {"ledger_id": parent, "rev": cur,
          "entries": [{"op": "remove", "match": f"[sub-task] {child} done"}]}
    assert not _tool(ag, "update", ok, _ctx("s2", 3, "x")).startswith("Error")


# ── 5 Goal 保护照旧按目标本 ──────────────────────────────────────────────────


def test_5_goal_protection_applies_on_target():
    """变异：对 ledger_id 路径跳过 goal 检查 ⇒ 无依据的 Goal 改写成功，本用例红。"""
    ag, _rec, parent, child = _setup()
    goal0 = ag.pool[parent].goal
    out = _tool(ag, "update", {"ledger_id": parent, "goal": "擅自改的目标",
                               "goal_change_quote": "用户从没说过这句"}, _ctx("s2", 2, "做完了"))
    assert out.startswith("Error")
    assert ag.pool[parent].goal == goal0


# ── 6 老义务文字的子本：switch 回写路径仍在 ──────────────────────────────────


def test_6_legacy_switch_writeback_path_still_recorded():
    """变异：删 `_child_ref` 判据 ⇒ `child_writeback` 不再记，本用例（及 L100 三件）红。"""
    ag, rec, parent, child = _setup()
    # 把子本 Next 里的义务换成**老文字**（存量子本的形态）
    led = ag.pool[child]
    old_text = (f"{SUBTASK_TAG} on completion: report status and results back to parent "
                f"ledger {parent} (bladex_ledger_switch to it, then bladex_ledger_update).")
    nxt = [e.model_copy(update={"text": old_text}) if getattr(e, "ref", "") == parent else e
           for e in led.entries("next")]
    sections = dict(led.sections)
    sections["next"] = nxt
    ag.pool[child] = led.model_copy(update={"sections": sections})
    mark = len(rec.events)
    _tool(ag, "switch", {"ledger_id": parent}, _ctx("s2", 2, "做完了"))
    sw = [pl for t, pl in rec.events[mark:] if t == "ledger_switch"]
    assert len(sw) == 1 and sw[0]["evidence_source"] == "child_writeback"
    assert ag.activation.active(SCOPE) == parent


# ── 7 文案守卫 ──────────────────────────────────────────────────────────────


def test_7_wording_guard_obligation_and_descriptions_and_budget():
    """变异：把义务文字还原为 "bladex_ledger_switch to it…" ⇒ 本用例红。"""
    ag, _rec, parent, child = _setup()
    ob = [e for e in ag.pool[child].entries("next") if getattr(e, "ref", "") == parent]
    assert len(ob) == 1
    assert ob[0].text == (f"{SUBTASK_TAG} on completion: report status and results back to "
                          f"parent ledger {parent} with bladex_ledger_update(ledger_id={parent}); "
                          "do not switch.")
    assert "bladex_ledger_switch" not in ob[0].text
    by = {t["function"]["name"]: t["function"] for t in toolface.TOOL_SCHEMAS}
    pdesc = by["bladex_ledger_switch"]["parameters"]["properties"]["parent_ledger_id"]["description"]
    assert "NOT a sub-task" in pdesc and len(pdesc) == 232
    udesc = by["bladex_ledger_update"]["parameters"]["properties"]["ledger_id"]["description"]
    assert len(udesc) == 145 and "do not switch" in udesc
    assert "ledger_id" not in by["bladex_ledger_update"]["parameters"].get("required", [])
    total = len(json.dumps(toolface.TOOL_SCHEMAS, ensure_ascii=False))
    assert total <= toolface.TOOL_FACE_SCHEMA_BUDGET_CHARS


# ── 8 日志 ──────────────────────────────────────────────────────────────────


def test_8_child_writeback_update_logged_once():
    """变异：去掉 `agency_child_writeback_update` 的 logger ⇒ 本用例红；普通 update 不打这条。"""
    ag, _rec, parent, child = _setup()
    with structlog.testing.capture_logs() as cap:
        _tool(ag, "update", _wb_args(child, parent), _ctx("s2", 2, "做完了"))
        _tool(ag, "update", {"section": "next", "op": "add", "text": "子本自己的一步"},
              _ctx("s2", 2, "做完了"))
    wb = [e for e in cap if e["event"] == "agency_child_writeback_update"]
    assert len(wb) == 1 and wb[0]["child"] == child and wb[0]["parent"] == parent
    assert wb[0]["parent_status"] == "active"
