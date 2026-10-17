"""批 L100（MQ-L100，2026-10-16）· 子任务回写不改变父本的开 / 关状态。

病例（10-16 dsh）：为 MQ-A93 新建子本、父 = **已关**的 A92 账本；做完按子本 Next 里的回写义务
切到父本 ⇒ switch 先 reopen，写一条、切回子本、关子本 ⇒ 父本从此停在 active。

本文件钉：

- **修得对**：A 建本 → 关本 → B 以它为父建子本 → 按义务回写 → 切回、关子本 ⇒ 父本最终 closed，
  回写条目在父本里可见，`closed_at` / 关闭者保持原值，B 段 `ledger_create` 只 1 次、`ledger_reopen` 0 次；
  经真 Hub 落盘重放后同样成立。
- **建子本那一刻**：已关父本不记 `awaiting`（`agency_subtask_parent_mark_skipped`），父子关系仍在
  子本 `parent_ledger_id` 与子本 Next 的回写义务里。
- **不误伤**：父本开着时的同一剧本，事件序列与修前**逐条相同**（金样 = 修前代码跑出的序列，
  `fixtures/l100_open_parent_events.json`）。
- 显式重开不受影响：不是从它的子本切过去（用户说回到那件事）⇒ 照旧 reopen。
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import re

import pytest
import structlog.testing
from bladex_core.ledger_runtime import activation_scope
from bladex_proxy.agency import AgencyRuntime

AGENT = "dsh"
PROJECT = "p-l100test0001"
SCOPE = activation_scope(AGENT, PROJECT)
_GOLDEN = pathlib.Path(__file__).parent / "fixtures" / "l100_open_parent_events.json"


@pytest.fixture(autouse=True)
def _modules_on(monkeypatch):
    for m in ("TOOLFACE", "LEDGER", "INTERCEPTION"):
        monkeypatch.setenv(f"BLADEX_MODULE_{m}", "1")


class _Rec:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    def __call__(self, etype, payload):
        self.events.append((getattr(etype, "value", str(etype)), payload))

    def of(self, etype: str) -> list[dict]:
        return [p for t, p in self.events if t == etype]

    def kinds(self, start: int = 0) -> list[str]:
        return [t for t, _ in self.events[start:]]


def _ctx(session: str, turn_index: int, user: str) -> dict:
    return {"session_id": session, "agent_id": AGENT, "project_id": PROJECT,
            "turn_index": turn_index, "user_query": user}


def _tool(ag: AgencyRuntime, name: str, args: dict, ctx: dict) -> str:
    fn = {"switch": ag._h_ledger_switch, "update": ag._h_ledger_update,   # noqa: SLF001
          "close": ag._h_ledger_close}[name]                              # noqa: SLF001
    return asyncio.run(fn(args, allowed_exposure="public", context=ctx))


def _create_parent(ag: AgencyRuntime) -> str:
    """A 段：会话 s1 建本、写一条 Core。返回父本 id。"""
    ctx = _ctx("s1", 1, "修 MQ-A92")
    out = _tool(ag, "switch", {"ledger_id": "", "title": "MQ-A92 修复",
                               "goal": "修 MQ-A92", "core": ["A92 的根因在压缩引擎"]}, ctx)
    assert out.startswith("Active ledger is now ")
    return ag.activation.active(SCOPE)


def _child_round_trip(ag: AgencyRuntime, parent: str) -> tuple[str, list[str]]:
    """B 段：会话 s2 以 parent 为父建子本 → 按义务回写 → 切回 → 关子本。返回 (子本 id, 各步回文)。"""
    outs = [_tool(ag, "switch", {"ledger_id": "", "title": "MQ-A93 修复", "goal": "修 MQ-A93",
                                 "parent_ledger_id": parent}, _ctx("s2", 1, "顺手把 A93 修了"))]
    child = ag.activation.active(SCOPE)
    assert child not in ("", parent)
    outs.append(_tool(ag, "switch", {"ledger_id": parent}, _ctx("s2", 2, "做完了")))
    outs.append(_tool(ag, "update", {"section": "verified", "op": "add",
                                     "text": f"[sub-task] {child} done: A93 规则已落"},
                      _ctx("s2", 2, "做完了")))
    outs.append(_tool(ag, "switch", {"ledger_id": child}, _ctx("s2", 2, "做完了")))
    outs.append(_tool(ag, "close", {}, _ctx("s2", 2, "做完了")))
    return child, outs


def _runtime(hub=None) -> tuple[AgencyRuntime, _Rec]:
    ag = AgencyRuntime(hub=hub)
    rec = _Rec()
    if hub is None:
        ag._emit = rec          # noqa: SLF001 —— 只替换事件出口，判定路径原样
    return ag, rec


# ── 修得对：已关父本 ─────────────────────────────────────────────────────────


def test_writeback_to_closed_parent_keeps_it_closed_with_original_close_info():
    ag, rec = _runtime()
    parent = _create_parent(ag)
    assert "closed" in _tool(ag, "close", {}, _ctx("s1", 2, "A92 做完了"))
    closed = ag.pool[parent]
    orig = (closed.status, closed.closed_at, closed.closed_by_agent, closed.closed_by_session)
    assert orig[0] == "closed" and orig[1] and orig[2:] == (AGENT, "s1")

    mark = len(rec.events)
    with structlog.testing.capture_logs() as cap:
        child, outs = _child_round_trip(ag, parent)

    # 回写那一次：激活了父本，但没有重开；回文如实说明
    assert "stays closed" in outs[1] and "reopened" not in outs[1]
    # 父本最终 closed，关闭信息逐字是原值（不被回写时刻覆盖）
    p = ag.pool[parent]
    assert (p.status, p.closed_at, p.closed_by_agent, p.closed_by_session) == orig
    # 回写条目在父本里可见，写者可指认
    wb = [e for e in p.entries("verified") if child in e.text]
    assert len(wb) == 1 and wb[0].writer == AGENT
    # 已关父本不记 awaiting
    assert not [e for e in p.entries("next") if getattr(e, "ref", "") == child]
    # 子本已关、scope 无激活
    assert ag.pool[child].status == "closed" and ag.activation.active(SCOPE) == ""
    # B 段事件：create 1 次、reopen 0 次；回写 update 的载荷是父本整本 dump（状态 closed）
    kinds = rec.kinds(mark)
    assert kinds.count("ledger_create") == 1 and "ledger_reopen" not in kinds
    upd = [pl for t, pl in rec.events[mark:] if t == "ledger_update"
           and pl["ledger"]["ledger_id"] == parent]
    assert len(upd) == 1 and upd[0]["ledger"]["status"] == "closed"
    assert upd[0]["ledger"]["closed_at"] == orig[1]
    sw = [pl for t, pl in rec.events[mark:] if t == "ledger_switch"
          and pl["to_ledger_id"] == parent]
    assert len(sw) == 1 and sw[0]["evidence_source"] == "child_writeback"
    assert sw[0]["kept_closed"] is True
    # 日志可 grep
    kept = [e for e in cap if e["event"] == "agency_ledger_writeback_kept_closed"]
    assert len(kept) == 1 and kept[0]["ledger"] == parent and kept[0]["child"] == child
    skipped = [e for e in cap if e["event"] == "agency_subtask_parent_mark_skipped"]
    assert len(skipped) == 1 and skipped[0]["reason"] == "parent_closed"
    assert not [e for e in cap if e["event"] == "agency_ledger_reopened"]


def test_child_keeps_the_parent_link_when_parent_is_closed():
    """父子关系不因不记 awaiting 而丢：子本 `parent_ledger_id` + 子本 Next 的回写义务都在。"""
    ag, rec = _runtime()
    parent = _create_parent(ag)
    _tool(ag, "close", {}, _ctx("s1", 2, "A92 做完了"))
    _tool(ag, "switch", {"ledger_id": "", "title": "MQ-A93 修复", "goal": "修 MQ-A93",
                         "parent_ledger_id": parent}, _ctx("s2", 1, "顺手把 A93 修了"))
    child = ag.pool[ag.activation.active(SCOPE)]
    assert child.parent_ledger_id == parent
    duty = [e for e in child.entries("next") if getattr(e, "ref", "") == parent]
    assert len(duty) == 1 and "[sub-task] on completion" in duty[0].text
    created = rec.of("ledger_create")[-1]["ledger"]
    assert created["parent_ledger_id"] == parent
    # 建子本没往父本写任何事件（修前这里有一条 awaiting 的 ledger_update）
    assert not [pl for t, pl in rec.events if t == "ledger_update"
                and pl["ledger"]["ledger_id"] == parent]


def test_closed_parent_survives_restart_through_a_real_hub(tmp_path):
    """真 Hub 落盘 → 新进程重放：父本 closed、关闭信息原值、回写条目在、无激活。"""
    from bladex_proxy.storage.memory_hub import MemoryHub

    hub = MemoryHub(tmp_path / "hub")
    hub.open()
    try:
        ag, _ = _runtime(hub=hub)
        parent = _create_parent(ag)
        _tool(ag, "close", {}, _ctx("s1", 2, "A92 做完了"))
        orig = ag.pool[parent].model_copy()
        child, _ = _child_round_trip(ag, parent)
    finally:
        hub.close()
    hub2 = MemoryHub(tmp_path / "hub")
    hub2.open()
    try:
        ag2 = AgencyRuntime(hub=hub2)
        ag2._load_pool()          # noqa: SLF001 —— 与 lifespan 同一装载入口
    finally:
        hub2.close()
    p = ag2.pool[parent]
    assert (p.status, p.closed_at, p.closed_by_agent, p.closed_by_session) == \
        (orig.status, orig.closed_at, orig.closed_by_agent, orig.closed_by_session)
    assert [e.text for e in p.entries("verified") if child in e.text]
    assert ag2.pool[child].status == "closed"
    assert ag2.activation.active(SCOPE) == ""


def test_explicit_switch_to_closed_ledger_still_reopens():
    """不是从它的子本切过去 ⇒ 照旧重开（Z3 / HO 的「切回去即重开」不变）。"""
    ag, rec = _runtime()
    parent = _create_parent(ag)
    _tool(ag, "close", {}, _ctx("s1", 2, "A92 做完了"))
    out = _tool(ag, "switch", {"ledger_id": parent}, _ctx("s3", 1, "A92 还有个尾巴"))
    assert "reopened" in out and ag.pool[parent].status == "active"
    assert rec.kinds()[-2:] == ["ledger_reopen", "ledger_switch"]
    assert "kept_closed" not in rec.of("ledger_switch")[-1]


# ── 不误伤：开着的父本，事件序列与修前逐条相同 ────────────────────────────────


_ISO = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[+-]\d{2}:\d{2}|Z)?")
_LID = re.compile(r"ldg-[0-9a-f]{12}")
_MID = re.compile(r"m-[0-9a-f]{12}")


def _normalize(events: list[tuple[str, dict]]) -> list:
    """时刻 → `<ts>`；账本 id / Matter id 按首次出现序 → `<L0>` `<M0>`…（id 随机，Matter id 由账本 id 派生）。"""
    raw = json.dumps(events, ensure_ascii=False, sort_keys=True, default=str)
    raw = _ISO.sub("<ts>", raw)
    ids: dict[str, str] = {}
    raw = _LID.sub(lambda m: ids.setdefault(m.group(0), f"<L{len(ids)}>"), raw)
    mids: dict[str, str] = {}
    raw = _MID.sub(lambda m: mids.setdefault(m.group(0), f"<M{len(mids)}>"), raw)
    return json.loads(raw)


def open_parent_events() -> list:
    """父本开着的同一剧本（不关 A 段）→ 归一化事件序列。金样由修前代码跑出。"""
    ag, rec = _runtime()
    parent = _create_parent(ag)
    _child_round_trip(ag, parent)
    return _normalize(rec.events)


def test_open_parent_event_sequence_is_identical_to_before():
    golden = json.loads(_GOLDEN.read_text(encoding="utf-8"))
    got = open_parent_events()
    assert [e[0] for e in got] == [e[0] for e in golden["events"]]
    for i, (a, b) in enumerate(zip(got, golden["events"], strict=True)):
        assert a == b, f"事件 #{i}（{a[0]}）与修前不同"


def test_open_parent_golden_has_the_awaiting_mark_and_no_reopen():
    """金样自检：它确实是「开父本」形状——有 awaiting 标记、没有 reopen / close 父本。"""
    golden = json.loads(_GOLDEN.read_text(encoding="utf-8"))["events"]
    kinds = [e[0] for e in golden]
    assert "ledger_reopen" not in kinds
    assert any(e[0] == "ledger_update" and "awaiting sub-task" in json.dumps(e[1], ensure_ascii=False)
               for e in golden)
    assert all(e[1].get("ledger_id") != "<L0>" for e in golden if e[0] == "ledger_close")
