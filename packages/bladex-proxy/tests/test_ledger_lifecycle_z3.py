"""批 Z · Z3 —— 账本 close / supersede / reopen 的三个入口（工具 · admin 端点 · CLI）。

判据编号对 `prompt-batch-Z3-20260923.md` §2（core 侧的重放 / 时间轴 / 停锚见
`packages/bladex-core/tests/test_ledger_lifecycle.py`）：

- ① 事件经**真 Hub** 落盘，新进程从 Hub 重放后状态保持（关 / 取代 / 重开）；链可读；
- ② 已关的本不进 recency top-5 与相关性候选（阴性：没关的照进）；
- ③ 关闭激活账本 ⇒ 该 scope 激活解除（进程内与重放后都成立）；
- ⑥ 模型调用 close 不改 Goal；
- Z3.2 生产侧：`ledger_update` 载荷带写者用户轮（停锚豁免的数据来源）。
"""

from __future__ import annotations

import asyncio

import pytest
from bladex_core.ledger import new_ledger
from bladex_core.ledger_runtime import activation_scope
from bladex_proxy.agency import AgencyRuntime

AGENT = "claude-code"
PROJECT = "p-z3test00001"
SCOPE = activation_scope(AGENT, PROJECT)
A, B, C = "ldg-aaaa0000z3a1", "ldg-bbbb0000z3b2", "ldg-cccc0000z3c3"


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


def _ctx(turn_index: int = 3, session: str = "sess-1", user: str = "这件事做完了") -> dict:
    return {"session_id": session, "agent_id": AGENT, "project_id": PROJECT,
            "turn_index": turn_index, "user_query": user}


def _runtime(hub=None) -> tuple[AgencyRuntime, _Rec]:
    ag = AgencyRuntime(hub=hub)
    rec = _Rec()
    if hub is None:
        ag._emit = rec          # noqa: SLF001 —— 只替换事件出口
    for lid, title in ((A, "泰山啤酒破产重整进展梳理"), (B, "泰山啤酒重整第二轮跟进"),
                       (C, "Mac Studio 选购")):
        ag.pool[lid] = new_ledger(ledger_id=lid, title=title, goal=f"{title}：做完可核对",
                                  goal_source="model", created_at="2026-09-20T00:00:00+00:00")
    return ag, rec


def _run(coro):
    return asyncio.run(coro)


def _close(ag, args: dict, **ctx):
    return _run(ag._h_ledger_close(args, allowed_exposure="public", context=_ctx(**ctx)))


def _switch(ag, args: dict, **ctx):
    return _run(ag._h_ledger_switch(args, allowed_exposure="public", context=_ctx(**ctx)))


# ── 工具面 ────────────────────────────────────────────────────────────────


def test_tool_close_releases_active_scope_and_keeps_goal():
    """判据 ③ + ⑥：关激活本 ⇒ scope 无激活；Goal / 正文一字不动；事件 origin=model 带写者。"""
    ag, rec = _runtime()
    _switch(ag, {"ledger_id": A})
    before = ag.pool[A].model_copy()
    out = _close(ag, {})                                  # 缺省 = 关激活本
    assert f"Ledger {A} closed." in out
    assert ag.activation.active(SCOPE) == ""
    after = ag.pool[A]
    assert after.status == "closed"
    assert (after.goal, after.goal_source, after.goal_revisions, after.sections) == \
        (before.goal, before.goal_source, before.goal_revisions, before.sections)
    (ev,) = rec.of("ledger_close")
    assert ev["ledger_id"] == A and ev["origin"] == "model"
    assert (ev["agent_id"], ev["project_id"], ev["session_id"], ev["turn_index"]) == \
        (AGENT, PROJECT, "sess-1", 3)
    assert "Error" in _close(ag, {})                      # 已无激活本


def test_tool_close_is_idempotent_and_validates_successor():
    ag, rec = _runtime()
    assert "Error: unknown ledger 'ldg-nope'" in _close(ag, {"ledger_id": A,
                                                             "superseded_by": "ldg-nope"})
    assert "cannot supersede itself" in _close(ag, {"ledger_id": A, "superseded_by": A})
    assert rec.events == [] and ag.pool[A].status == "active"
    assert "superseded by" in _close(ag, {"ledger_id": A, "superseded_by": B})
    assert ag.pool[A].superseded_by == B
    assert "already closed" in _close(ag, {"ledger_id": A, "superseded_by": B})
    assert len(rec.of("ledger_supersede")) == 1


def test_switch_to_closed_ledger_reopens_then_switches():
    """reopen 不单开工具：模型显式切到已关的本 = 先 reopen 再 switch（两条事件，reopen 在前）。"""
    ag, rec = _runtime()
    _close(ag, {"ledger_id": A})
    out = _switch(ag, {"ledger_id": A, "evidence": "做完了"})
    assert "reopened" in out
    assert ag.pool[A].status == "active" and ag.activation.active(SCOPE) == A
    kinds = [t for t, _ in rec.events]
    assert kinds[-2:] == ["ledger_reopen", "ledger_switch"]
    assert rec.of("ledger_reopen")[0]["origin"] == "model"


def test_update_payload_carries_writer_user_turn():
    """Z3.2 数据来源：完结写入发生在哪个用户轮（停锚豁免只认它）。"""
    ag, rec = _runtime()
    _switch(ag, {"ledger_id": A})
    _run(ag._h_ledger_update({"section": "verified", "op": "add", "text": "查到了", "ref": "t1"},
                             allowed_exposure="public", context=_ctx(turn_index=7)))
    (ev,) = rec.of("ledger_update")
    assert ev["ledger"]["ledger_id"] == A
    assert (ev["session_id"], ev["turn_index"], ev["agent_id"], ev["project_id"]) == \
        ("sess-1", 7, AGENT, PROJECT)


def test_close_tool_is_in_the_ledger_family_and_registered():
    from bladex_proxy.toolface import FAMILY_LEDGER, TOOL_NAMES, tool_family
    ag, _ = _runtime()
    assert "bladex_ledger_close" in TOOL_NAMES
    assert tool_family("bladex_ledger_close") == FAMILY_LEDGER
    assert "bladex_ledger_close" in ag.toolface.registered


# ── ② 候选：已关的本不进 top-5；相关性段按内容命中时照列并标 closed（批 HO，MQ-L99）──
#
# 🔴 原判据「已关的本不进 top-5 与相关性两段」自批 HO（2026-10-15，拍板 (43)）起只对 recency 段成立：
# 交接病例里交出方关了本，接手方的首句按内容唯一命中它，却因相关性段整本排除而只能新建。


def test_closed_ledger_leaves_recency_and_stays_in_relevance_marked_closed():
    ag, _ = _runtime()
    user = "泰山啤酒破产重整最新进展"
    msg, _bd = ag._build_ledger_block(AGENT, project_id=PROJECT, user_text=user,
                                      with_instruction=True, face="user")
    text0 = msg["content"]
    assert A in text0 and B in text0 and C in text0, "夹具前提：三本都在候选里"
    ag.lifecycle_close(A, origin="user")
    msg, _bd = ag._build_ledger_block(AGENT, project_id=PROJECT, user_text=user,
                                      with_instruction=True, face="user")
    text1 = msg["content"]
    rows_a = [ln for ln in text1.splitlines() if ln.startswith(f"- {A}:")]
    assert len(rows_a) == 1 and "[closed " in rows_a[0], rows_a   # 只在相关性段、带标记
    recent = text1[text1.index("Other recent ledgers"):] if "Other recent ledgers" in text1 else ""
    assert A not in recent                                # recency 段仍不列已关本
    assert B in text1 and C in text1                     # 阴性对照：没关的照进


# ── ① 真 Hub：落盘 → 新进程重放 ──────────────────────────────────────────


def test_lifecycle_survives_restart_through_a_real_hub(tmp_path):
    from bladex_proxy.storage.memory_hub import MemoryHub

    hub = MemoryHub(tmp_path / "hub")
    hub.open()
    try:
        ag, _ = _runtime(hub=hub)
        for lid in (A, B, C):     # 池里的本也要进 Hub（建本事件），否则重放没有它
            from bladex_proxy.models import AdminEventType
            hub.append_admin_event(AdminEventType.LEDGER_CREATE, "",
                                   ledger=ag.pool[lid].model_dump())
        _switch(ag, {"ledger_id": A})
        _close(ag, {"ledger_id": A, "superseded_by": B})
        _close(ag, {"ledger_id": C})
        ag.lifecycle_reopen(C, origin="cli", reason="回来了")
    finally:
        hub.close()
    hub2 = MemoryHub(tmp_path / "hub")
    hub2.open()
    try:
        ag2 = AgencyRuntime(hub=hub2)
        ag2._load_pool()          # noqa: SLF001 —— 与 lifespan 同一装载入口
    finally:
        hub2.close()
    assert ag2.pool[A].status == "closed" and ag2.pool[A].superseded_by == B
    assert ag2.pool[C].status == "active" and ag2.pool[C].superseded_by == ""
    assert ag2.activation.active(SCOPE) == ""             # 关本解除的激活重放后仍解除


# ── admin 端点（CLI 走同一端点）──────────────────────────────────────────


@pytest.fixture
def client(monkeypatch):
    from bladex_proxy.config import ProxyConfig
    from bladex_proxy.server import create_app
    from fastapi.testclient import TestClient

    monkeypatch.setenv("BLADEX_AUTH_ENABLED", "false")
    ag, rec = _runtime()
    _switch(ag, {"ledger_id": A})
    with TestClient(create_app(ProxyConfig())) as c:
        c.app.state.agency = ag
        yield c, ag, rec


def test_admin_close_supersede_reopen(client):
    c, ag, rec = client
    r = c.post(f"/admin/ledgers/{A}/supersede", json={"origin": "cli", "superseded_by": B})
    assert r.status_code == 200 and r.json()["status"] == "superseded"
    assert r.json()["freed_scopes"] and ag.activation.active(SCOPE) == ""
    show = c.get(f"/admin/ledgers/{A}").json()
    assert [x["ledger_id"] for x in show["superseded_chain"]] == [B]
    assert c.get(f"/admin/ledgers/{B}").json()["supersedes"] == [A]
    r = c.post(f"/admin/ledgers/{A}/reopen", json={"origin": "cli"})
    assert r.json()["status"] == "reopened" and ag.pool[A].status == "active"
    assert ag.activation.active(SCOPE) == ""              # 重开 ≠ 激活
    r = c.post(f"/admin/ledgers/{C}/close", json={"reason": "done"})
    assert r.json()["status"] == "closed"
    origins = [p["origin"] for t, p in rec.events if t.startswith("ledger_") and "origin" in p]
    assert origins == ["cli", "cli", "user"]


def test_admin_rejects_model_origin_and_unknown_ledger(client):
    c, ag, rec = client
    n = len(rec.events)
    r = c.post(f"/admin/ledgers/{A}/close", json={"origin": "model"})
    assert r.status_code == 400
    assert c.post("/admin/ledgers/ldg-nope/close", json={}).status_code == 404
    assert c.post(f"/admin/ledgers/{A}/supersede", json={}).status_code == 400
    assert len(rec.events) == n and ag.pool[A].status == "active"


def test_cli_commands_send_cli_origin(client, monkeypatch, capsys):
    from bladex_proxy import cli
    from bladex_proxy.cli import ledger_cmds as _ldg

    c, ag, _rec = client

    def transport(method, path, body=None, params=None):
        r = c.request(method, path, json=body)
        return r.status_code, r.json()

    monkeypatch.setattr(_ldg, "_admin_call", transport)
    assert cli.main(["ledger", "supersede", A, "--by", B, "--reason", "dup"]) == 0
    assert cli.main(["ledger", "show", A]) == 0
    out = capsys.readouterr().out
    assert f"superseded by {B}" in out and f"-> {B}" in out
    assert cli.main(["ledger", "reopen", A]) == 0
    assert cli.main(["ledger", "close", A]) == 0
    assert ag.pool[A].status == "closed"
