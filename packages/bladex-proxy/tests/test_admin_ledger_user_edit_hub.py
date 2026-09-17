"""MQ-L51（批 F0.3，2026-09-06）：用户直编采纳 = 进池 **+ 落 Hub 一条 `LEDGER_USER_EDIT`**。

修前 `admin_ledger_user_edit` 只 `agency.pool[...] = led`；`EVENT_LEDGER_USER_EDIT` /
`replay_ledger_events` 状态式重放 / `AdminEventType.LEDGER_USER_EDIT` 早已存在却没有生产者。
后果：模型此后不写这本 ⇒ Hub 停在采纳前 ⇒ 每次 proxy 重启池回旧版 ⇒ daemon 对账重采 ⇒
一条假 `clobber_risk`（09-06 11:06 / 11:33 两次重启各重采 2 本、各 2 条假 clobber）。

钉：① 采纳后 Hub `append_admin_event` 恰一次、type=`ledger_user_edit`、payload rev = 采纳 rev；
    ② `replay_ledger_events([create, user_edit])` ⇒ 池版 = 用户版（rev = 用户提交 rev+1）；
    ③ 判别力对照：`_emit` 不落 Hub（= 修前形态）⇒ 事件流重建池后 daemon `_reconcile_stale_adoption`
       判 `readopted`；落 Hub ⇒ pending 当轮清、重启后无对账项、零重发。
"""

from __future__ import annotations

import json
import os

import structlog.testing
from fastapi.testclient import TestClient

from bladex_core.ledger import (
    EVENT_LEDGER_CREATE,
    EVENT_LEDGER_USER_EDIT,
    Ledger,
    ledger_md_path,
    new_ledger,
    replay_ledger_events,
)
from bladex_proxy.agency import AgencyRuntime
from bladex_proxy.config import ProxyConfig
from bladex_proxy.flash_daemon import FlashDaemon
from bladex_proxy.models import AdminEventType
from bladex_proxy.server import create_app


class _Ev:
    def __init__(self, etype: str, payload: dict):
        self.event_type = etype
        self.payload = payload
        self.ts = None


class _Hub:
    """最小 Hub：追加即入事件流；`scan_admin_events` 供池重建（= 模拟 proxy 重启）。"""

    def __init__(self):
        self.events: list[_Ev] = []
        self.appended: list[tuple[str, str, dict]] = []

    def append_admin_event(self, etype, matter_id, **payload):
        et = getattr(etype, "value", str(etype))
        self.appended.append((et, matter_id, payload))
        self.events.append(_Ev(et, payload))

    def scan_admin_events(self):
        yield from (("k", e) for e in self.events)

    def scan_tombstones(self):
        return iter(())

    def catch_up(self):
        return None


def _seed() -> Ledger:
    return new_ledger(ledger_id="ldg-l51a000001", title="L51 采纳落 Hub",
                      goal="直编后重启不回退", goal_source="user",
                      created_at="2026-09-06T10:00:00Z").model_copy(update={"rev": 11})


def _client(monkeypatch, hub: _Hub) -> TestClient:
    monkeypatch.setenv("BLADEX_AUTH_ENABLED", "false")
    app = create_app(ProxyConfig())
    c = TestClient(app)
    c.__enter__()
    ag = AgencyRuntime(hub=hub)
    c.app.state.agency = ag
    return c


# ── ① 采纳落 Hub 恰一次 ────────────────────────────────────────────────────

def test_adoption_appends_one_user_edit_event_with_adopted_rev(monkeypatch):
    hub = _Hub()
    seed = _seed()
    hub.append_admin_event(AdminEventType.LEDGER_CREATE, "", ledger=seed.model_dump())
    c = _client(monkeypatch, hub)
    try:
        assert c.app.state.agency.pool["ldg-l51a000001"].rev == 11, "池从事件流装载"
        hub.appended.clear()
        user_version = seed.model_copy(update={"title": "L51（用户改名）"})
        r = c.post("/admin/ledgers/user-edit", json={"ledger": user_version.model_dump(mode="json")})
        assert r.status_code == 200, r.text
        assert r.json()["rev"] == 12
        assert len(hub.appended) == 1, hub.appended
        et, matter_id, payload = hub.appended[0]
        assert et == "ledger_user_edit" and matter_id == "" , "MQ-A7 护栏：matter_id 留空"
        assert payload["ledger"]["rev"] == 12 and payload["ledger"]["last_writer"] == "user"
        assert payload["ledger"]["title"] == "L51（用户改名）"
    finally:
        c.__exit__(None, None, None)


# ── ② 重放等价 ────────────────────────────────────────────────────────────

def test_replay_treats_user_edit_as_full_state():
    seed = _seed()
    adopted = seed.model_copy(update={"title": "用户版", "rev": seed.rev + 1, "last_writer": "user"})
    pool, _ = replay_ledger_events([
        {"type": EVENT_LEDGER_CREATE, "payload": {"ledger": seed.model_dump()}},
        {"type": EVENT_LEDGER_USER_EDIT, "payload": {"ledger": adopted.model_dump()}},
    ])
    assert pool["ldg-l51a000001"].rev == 12 and pool["ldg-l51a000001"].title == "用户版"


# ── ③ 判别力对照：修前形态（不落 Hub）⇒ 重启后 readopted；修后 ⇒ pool_wins ──────

def _daemon_round_trip(tmp_path, monkeypatch, *, emit_to_hub: bool) -> dict:
    """直编 → 采纳 → 模拟 proxy 重启（池 = 事件流重放）→ daemon 启动对账。"""
    hub = _Hub()
    seed = _seed()
    hub.append_admin_event(AdminEventType.LEDGER_CREATE, "", ledger=seed.model_dump())
    c = _client(monkeypatch, hub)
    try:
        ag: AgencyRuntime = c.app.state.agency
        if not emit_to_hub:
            monkeypatch.setattr(ag, "_emit", lambda *_a, **_k: None)   # = 修前：只进池

        def hub_source():
            return replay_ledger_events(
                [{"type": e.event_type, "payload": e.payload} for e in hub.events])

        proxy_calls: list = []

        def emit(etype, payload):
            proxy_calls.append(etype)
            r = c.post("/admin/ledgers/user-edit", json=payload)
            return r.json() if r.status_code == 200 else None

        d = FlashDaemon(root=str(tmp_path), principal="u1", ledger_source=hub_source,
                        emit_admin_event=emit)
        d.run_once()
        path = ledger_md_path(str(tmp_path), "u1", "ldg-l51a000001")
        content = open(path, encoding="utf-8").read()
        edited = content.replace("## Open\n\n_(empty)_", "## Open\n\n- 用户手记的一个疑点")
        assert edited != content
        open(path, "w", encoding="utf-8").write(edited)
        d.run_once()                                   # 采纳（proxy 池 rev 12）
        assert proxy_calls == ["ledger_user_edit"]
        assert ag.pool["ldg-l51a000001"].rev == 12
        state = json.load(open(os.path.join(str(tmp_path), ".flash_state.json")))
        out = {"pending_after_adopt": path in d._pending_edits,
               "adopted_rev_persisted": state.get("adopted_rev", {})}

        # 模拟 proxy 重启：新 AgencyRuntime 从同一 Hub 事件流重建池
        ag2 = AgencyRuntime(hub=hub)
        c.app.state.agency = ag2
        out["pool_rev_after_restart"] = ag2.pool["ldg-l51a000001"].rev
        proxy_calls.clear()
        d2 = FlashDaemon(root=str(tmp_path), principal="u1", ledger_source=hub_source,
                         emit_admin_event=emit)
        with structlog.testing.capture_logs() as cap:
            d2.startup_check()
        out["modes"] = [e["mode"] for e in cap
                        if e.get("event") == "flash_daemon_stale_adoption_reconciled"]
        out["resends"] = len(proxy_calls)
        out["disk_has_edit"] = "用户手记的一个疑点" in open(path, encoding="utf-8").read()
        return out
    finally:
        c.__exit__(None, None, None)


def test_with_hub_event_restart_keeps_adoption(tmp_path, monkeypatch):
    """采纳进了 Hub ⇒ 当轮 Hub 投影 rev 12 ≥ 采纳 rev ⇒ pending 当轮清；重启后池 = 采纳版，
    startup 无对账项、零重采。"""
    r = _daemon_round_trip(tmp_path, monkeypatch, emit_to_hub=True)
    assert r["pending_after_adopt"] is False and r["adopted_rev_persisted"] == {}
    assert r["pool_rev_after_restart"] == 12
    assert r["modes"] == [] and r["resends"] == 0 and r["disk_has_edit"]


def test_without_hub_event_restart_readopts_old_form(tmp_path, monkeypatch):
    """判别力对照：去掉那一行（`_emit` 不落 Hub）⇒ 池回 rev 11 ⇒ daemon 启动判 `readopted`、重采一次。"""
    r = _daemon_round_trip(tmp_path, monkeypatch, emit_to_hub=False)
    assert r["pending_after_adopt"] is True and list(r["adopted_rev_persisted"].values()) == [12]
    assert r["pool_rev_after_restart"] == 11, "修前形态：重启回旧版"
    assert r["modes"] == ["readopted"] and r["resends"] == 1
