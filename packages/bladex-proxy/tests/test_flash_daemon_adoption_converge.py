"""MQ-L47 · Flash 直编采纳后**永不收敛**，daemon 每轮重发旧版盖掉模型写入（批 E E0.1，2026-09-06）。

修前机制：采纳把池内 rev +1（server.py），daemon 的收敛判据是 `render(pool) == on_disk` 逐字
相等——磁盘 `rev: 11` vs 渲染 `rev: 12` ⇒ 等式在采纳那一刻起恒假 ⇒ 每轮 ≥30s 重发磁盘旧版，
成功无日志、失败 `except: pass`。修前基线（09-02～05 全部 proxy 日志）：proxy 侧
`admin_ledger_user_edit_adopted` 177 vs daemon 侧 `flash_daemon_user_edit_adopted` 2；
`changed=['verified']` 11 条里 10 条 = 模型 `ledger_update` 后被重发盖回。

修法（原则 12：修实现，不把"永不收敛"写成规格）：proxy 响应带采纳后的池版 `ledger`，daemon
**当轮写回磁盘**并改按 rev 判收敛（Hub 投影 rev ≥ 采纳 rev）；新路径不重发；旧路径重发有上限。

钉：① 采纳 → 同轮物化：磁盘 rev = 池 rev、`_adopted_sha` 空、下一轮 0 emit
    ② 判别力对照：去掉回写（响应不带 ledger = 旧 proxy）⇒ 第二轮 emit 必发生
    ③ 重发上限：三轮后停发、warning 恰一条
    ④ 旧 proxy 兼容：响应无 `ledger` ⇒ pending 路径
    ⑤ startup 捞旧：`.flash_state.json` 带 adopted sha + 池 rev 更高 ⇒ 磁盘 rev 追平 + 日志一条
    ⑥ server：响应 `ledger.rev == prev.rev + 1`；`same_as_last_adoption` 两态
    ⑦ Hub 追上（模型采纳后写过）⇒ 按 rev 收敛、池版（含模型新写）落盘
"""

from __future__ import annotations

import json
import os

import structlog.testing
from bladex_core.ledger import (
    ACTOR_MODEL,
    Ledger,
    LedgerEntry,
    add_entry,
    ledger_md_path,
    new_ledger,
    parse_ledger_md,
    render_ledger_md,
)
from bladex_proxy import flash_daemon as fd
from bladex_proxy.flash_daemon import FlashDaemon


def _pool_v1():
    led = new_ledger(ledger_id="ldg-a1", title="任务甲",
                     goal="把批二做完", goal_source="user",
                     created_at="2026-08-25T10:00:00Z")
    led = add_entry(led, "next", LedgerEntry(text="写测试", source="model"),
                    actor=ACTOR_MODEL)
    return {"ldg-a1": led.model_copy(update={"rev": 11})}


class _Proxy:
    """模拟 proxy 端：采纳 = 池整本覆盖 + rev+1；响应带池版（新）或不带（旧）。"""

    def __init__(self, pool: dict, *, new_protocol: bool = True):
        self.pool = pool
        self.new_protocol = new_protocol
        self.calls: list = []

    def __call__(self, etype: str, payload: dict):
        self.calls.append((etype, payload))
        led = Ledger.model_validate(payload["ledger"])
        prev = self.pool.get(led.ledger_id)
        if prev is not None:
            led = led.model_copy(update={"rev": prev.rev + 1, "last_writer": "user"})
        self.pool[led.ledger_id] = led
        if not self.new_protocol:
            return None
        return {"status": "ok", "ledger_id": led.ledger_id, "rev": led.rev,
                "ledger": led.model_dump(mode="json")}


def _mk(tmp_path, hub_holder, emit):
    return FlashDaemon(root=str(tmp_path), principal="u1",
                       ledger_source=lambda: (hub_holder["pool"], {}),
                       emit_admin_event=emit)


def _edit(path: str) -> str:
    content = open(path, encoding="utf-8").read()
    edited = content.replace("## Open\n\n_(empty)_", "## Open\n\n- 用户手记的一个疑点")
    assert edited != content
    open(path, "w", encoding="utf-8").write(edited)
    return edited


def _disk_rev(path: str) -> int:
    return fd.rendered_rev(open(path, encoding="utf-8").read())


# ── ① 采纳当轮收敛 ─────────────────────────────────────────────────────────

def test_adoption_writes_pool_version_back_same_round(tmp_path, monkeypatch):
    monkeypatch.setattr(fd.time, "time", lambda: 1_000_000.0)
    hub = {"pool": _pool_v1()}
    proxy = _Proxy(dict(hub["pool"]))          # proxy 内存池（与 Hub 投影分家）
    d = _mk(tmp_path, hub, proxy)
    d.run_once()
    path = ledger_md_path(str(tmp_path), "u1", "ldg-a1")
    assert _disk_rev(path) == 11
    _edit(path)

    with structlog.testing.capture_logs() as cap:
        out = d.run_once()
    assert out["user_edits"] == 1 and len(proxy.calls) == 1
    on_disk = open(path, encoding="utf-8").read()
    # 磁盘 = 池版（rev 12、含用户条目），不是用户原版（rev 11）
    assert _disk_rev(path) == 12 == proxy.pool["ldg-a1"].rev
    assert "用户手记的一个疑点" in on_disk
    assert on_disk == render_ledger_md(proxy.pool["ldg-a1"])
    assert d._adopted_sha == {}, "新路径不该留旧式 pending 标记（它驱动重发）"
    assert d._adopted_rev[path] == 12
    adopted = [e for e in cap if e.get("event") == "flash_daemon_user_edit_adopted"]
    assert len(adopted) == 1 and adopted[0]["rev"] == 12

    # 下一轮（含 30s 之后）：Hub 还没追上 ⇒ 不覆盖、不重发、不重报
    monkeypatch.setattr(fd.time, "time", lambda: 1_000_100.0)
    d.push()
    assert d.run_once()["user_edits"] == 0
    assert len(proxy.calls) == 1, "采纳成功后不许再 emit（修前每轮 ≥30s 重发一次）"
    assert open(path, encoding="utf-8").read() == on_disk


# ── ② 判别力对照 + ④ 旧 proxy 兼容 ────────────────────────────────────────

def test_without_writeback_second_round_reemits(tmp_path, monkeypatch):
    """去掉回写（= 响应不带 ledger 的旧 proxy）⇒ 30s 后第二轮 emit 必发生。
    这条红了才证明 ① 的"0 emit"是回写带来的，而不是重发通道本身坏了。"""
    t = {"now": 1_000_000.0}
    monkeypatch.setattr(fd.time, "time", lambda: t["now"])
    hub = {"pool": _pool_v1()}
    proxy = _Proxy(dict(hub["pool"]), new_protocol=False)
    d = _mk(tmp_path, hub, proxy)
    d.run_once()
    path = ledger_md_path(str(tmp_path), "u1", "ldg-a1")
    edited = _edit(path)
    d.run_once()
    assert len(proxy.calls) == 1 and path in d._pending_edits and path in d._adopted_sha
    assert open(path, encoding="utf-8").read() == edited, "旧路径：用户版不被覆盖"
    t["now"] += 31
    d.push()
    with structlog.testing.capture_logs() as cap:
        d.run_once()
    assert len(proxy.calls) == 2, "旧 proxy 路径的重发通道必须仍在工作"
    assert [e for e in cap if e.get("event") == "flash_daemon_user_edit_reemitted"]


# ── ③ 重发上限 ────────────────────────────────────────────────────────────

def test_reemit_capped_after_three_attempts(tmp_path, monkeypatch):
    t = {"now": 1_000_000.0}
    monkeypatch.setattr(fd.time, "time", lambda: t["now"])
    hub = {"pool": _pool_v1()}
    proxy = _Proxy(dict(hub["pool"]), new_protocol=False)
    d = _mk(tmp_path, hub, proxy)
    d.run_once()
    path = ledger_md_path(str(tmp_path), "u1", "ldg-a1")
    edited = _edit(path)
    d.run_once()                                   # 采纳（1 次 emit）
    with structlog.testing.capture_logs() as cap:
        for _ in range(6):
            t["now"] += 31
            d.push()
            d.run_once()
    assert len(proxy.calls) == 1 + fd.REEMIT_CAP, "上限之后不许再发"
    capped = [e for e in cap if e.get("event") == "flash_daemon_user_edit_reemit_capped"]
    assert len(capped) == 1, "告警恰一条（每轮都吵等于没告警）"
    assert path in d._pending_edits
    assert open(path, encoding="utf-8").read() == edited, "停发 ≠ 放弃红线 2：仍不覆盖"


def test_reemit_failure_is_logged_not_swallowed(tmp_path, monkeypatch):
    t = {"now": 1_000_000.0}
    monkeypatch.setattr(fd.time, "time", lambda: t["now"])
    hub = {"pool": _pool_v1()}
    state = {"fail": False, "n": 0}

    def emit(_t, _p):
        state["n"] += 1
        if state["fail"]:
            raise RuntimeError("proxy down")
        return None                                # 旧 proxy 形态

    d = _mk(tmp_path, hub, emit)
    d.run_once()
    _edit(ledger_md_path(str(tmp_path), "u1", "ldg-a1"))
    d.run_once()
    state["fail"] = True
    t["now"] += 31
    d.push()
    with structlog.testing.capture_logs() as cap:
        d.run_once()
    assert state["n"] == 2
    assert [e for e in cap if e.get("event") == "flash_daemon_reemit_failed"], \
        "修前失败 `except: pass`——daemon 侧零痕迹是 L47 参照系脱钩的成因"


# ── ⑤ startup 捞旧 ────────────────────────────────────────────────────────

def test_startup_reconciles_stale_adoption_when_hub_moved_past(tmp_path):
    """`.flash_state.json` 带 adopted sha（修前形态的存量 pending）+ 池 rev 更高
    ⇒ 不重发（重发 = 再盖一次模型写入），磁盘 rev 追平池 rev，日志一条。"""
    hub = {"pool": _pool_v1()}
    calls: list = []
    d = _mk(tmp_path, hub, lambda t, p: calls.append(t))
    d.run_once()
    path = ledger_md_path(str(tmp_path), "u1", "ldg-a1")
    edited = _edit(path)                            # 用户版（rev 行仍是 11）
    # 伪造修前 daemon 留下的状态：采纳过、未收敛
    sha = FlashDaemon._sha(edited)
    with open(os.path.join(str(tmp_path), ".flash_state.json"), "w", encoding="utf-8") as f:
        json.dump({"baseline": {path: sha}, "adopted": {path: sha}}, f)
    # Hub 在采纳后被模型写过：rev 13 + Verified 条目（含用户那条——模型 update 带整本）
    moved = Ledger.model_validate(parse_ledger_md(edited, base=hub["pool"]["ldg-a1"]).model_dump())
    moved = add_entry(moved, "verified", LedgerEntry(text="gate 全绿", source="tool", ref="gate#9"),
                      actor=ACTOR_MODEL).model_copy(update={"rev": 13})
    hub["pool"] = {"ldg-a1": moved}

    d2 = _mk(tmp_path, hub, lambda t, p: calls.append(t))
    with structlog.testing.capture_logs() as cap:
        d2.startup_check()
    rows = [e for e in cap if e.get("event") == "flash_daemon_stale_adoption_reconciled"]
    assert len(rows) == 1 and rows[0]["mode"] == "pool_wins"
    assert calls == [], "池已越过采纳 ⇒ 不重发"
    final = open(path, encoding="utf-8").read()
    assert _disk_rev(path) == 13 and "gate 全绿" in final and "用户手记的一个疑点" in final
    assert path not in d2._pending_edits and d2._adopted_sha == {}


def test_startup_readopts_when_hub_never_got_it(tmp_path):
    """对偶：池 rev 没越过采纳（proxy 重启丢了内存池、模型没写过）⇒ 走一次新采纳路径。"""
    hub = {"pool": _pool_v1()}
    d = _mk(tmp_path, hub, lambda t, p: None)
    d.run_once()
    path = ledger_md_path(str(tmp_path), "u1", "ldg-a1")
    edited = _edit(path)
    sha = FlashDaemon._sha(edited)
    with open(os.path.join(str(tmp_path), ".flash_state.json"), "w", encoding="utf-8") as f:
        json.dump({"baseline": {path: sha}, "adopted": {path: sha}}, f)
    proxy = _Proxy(dict(hub["pool"]))              # 新 proxy，池 = Hub（rev 11）
    d2 = _mk(tmp_path, hub, proxy)
    with structlog.testing.capture_logs() as cap:
        d2.startup_check()
    rows = [e for e in cap if e.get("event") == "flash_daemon_stale_adoption_reconciled"]
    assert len(rows) == 1 and rows[0]["mode"] == "readopted"
    assert len(proxy.calls) == 1
    assert _disk_rev(path) == 12 and "用户手记的一个疑点" in open(path, encoding="utf-8").read()


def test_startup_readopts_new_path_adoption_after_proxy_restart(tmp_path):
    """新路径的采纳（`adopted_rev` 持久化）跨 daemon 重启：Hub 没追上（proxy 单独
    重启丢了内存池）⇒ 补采一次；追上了 ⇒ 池版胜。两态各一。"""
    hub = {"pool": _pool_v1()}
    proxy = _Proxy(dict(hub["pool"]))
    d = _mk(tmp_path, hub, proxy)
    d.run_once()
    path = ledger_md_path(str(tmp_path), "u1", "ldg-a1")
    _edit(path)
    d.run_once()                                   # 新路径采纳，磁盘 rev 12，状态落盘
    assert json.load(open(os.path.join(str(tmp_path), ".flash_state.json")))["adopted_rev"] == {path: 12}

    # A. proxy 重启丢池（新 proxy 池 = Hub，rev 11）⇒ daemon 重启补采一次
    proxy2 = _Proxy(dict(hub["pool"]))
    d2 = _mk(tmp_path, hub, proxy2)
    with structlog.testing.capture_logs() as cap:
        d2.startup_check()
    rows = [e for e in cap if e.get("event") == "flash_daemon_stale_adoption_reconciled"]
    assert len(rows) == 1 and rows[0]["mode"] == "readopted" and len(proxy2.calls) == 1
    assert proxy2.pool["ldg-a1"].rev == 12 and "用户手记的一个疑点" in open(path, encoding="utf-8").read()

    # B. Hub 追上（rev 12 = 采纳 rev）⇒ 池版胜、不重发
    hub["pool"] = {"ldg-a1": proxy2.pool["ldg-a1"]}
    proxy3 = _Proxy(dict(hub["pool"]))
    d3 = _mk(tmp_path, hub, proxy3)
    with structlog.testing.capture_logs() as cap:
        d3.startup_check()
    rows = [e for e in cap if e.get("event") == "flash_daemon_stale_adoption_reconciled"]
    assert len(rows) == 1 and rows[0]["mode"] == "pool_wins" and proxy3.calls == []
    assert path not in d3._pending_edits and d3._adopted_rev == {}


def test_pending_reemit_runs_without_a_wake(tmp_path, monkeypatch):
    """E0.1b（live 09-06 11:06）：启动补采撞上 proxy 还没起（Connection refused）落回 pending 后，
    主循环是唤醒驱动的——没有流量 `run_once` 不跑、重发永远等不到第二次。`pending_wait_s()`
    给主循环一个重发窗；`run_once` 在不 dirty 时也处理 pending；上限到了回到 None（零轮询）。"""
    t = {"now": 1_000_000.0}
    monkeypatch.setattr(fd.time, "time", lambda: t["now"])
    hub = {"pool": _pool_v1()}
    state = {"up": False, "n": 0}

    def emit(_t, _p):
        state["n"] += 1
        if not state["up"]:
            raise RuntimeError("Connection refused")
        return None                                  # 旧 proxy 形态（走 pending）

    d = _mk(tmp_path, hub, emit)
    d.run_once()
    path = ledger_md_path(str(tmp_path), "u1", "ldg-a1")
    edited = _edit(path)
    sha = FlashDaemon._sha(edited)
    with open(os.path.join(str(tmp_path), ".flash_state.json"), "w", encoding="utf-8") as f:
        json.dump({"baseline": {path: sha}, "adopted": {path: sha}}, f)
    d2 = _mk(tmp_path, hub, emit)
    d2.startup_check()                               # 补采失败 → pending；同轮重发 attempt=1 也失败
    assert path in d2._pending_edits and state["n"] == 2
    assert d2.pending_wait_s() is not None
    state["up"] = True
    t["now"] += 31
    # 🔴 不 push（无门铃）也要重发
    assert d2.pending_wait_s() == 0.0
    d2.run_once()
    assert state["n"] == 3, "无唤醒时 pending 重发没跑——468d 会一直等到下一个请求"
    # 达上限后回到零轮询
    for _ in range(3):
        t["now"] += 31
        d2.run_once()
    # 启动补采 1 次 + 重发恰 REEMIT_CAP 次（成功的重发也计次，否则旧 proxy 下永不停）
    assert state["n"] == 1 + fd.REEMIT_CAP and d2.pending_wait_s() is None


def test_startup_readopt_refused_downgrades_to_reemit_channel(tmp_path, monkeypatch):
    """E0.1c（live 09-06 11:23）：新路径 pending（`adopted_rev`）在 daemon 重启对账时补采撞
    Connection refused ⇒ 必须降级进有上限的重发通道，否则新路径不重发、卡到下次重启再撞一次。"""
    t = {"now": 1_000_000.0}
    monkeypatch.setattr(fd.time, "time", lambda: t["now"])
    hub = {"pool": _pool_v1()}
    proxy = _Proxy(dict(hub["pool"]))
    d = _mk(tmp_path, hub, proxy)
    d.run_once()
    path = ledger_md_path(str(tmp_path), "u1", "ldg-a1")
    _edit(path)
    d.run_once()                                    # 新路径采纳：磁盘 rev 12，adopted_rev=12
    written = open(path, encoding="utf-8").read()
    state = {"up": False}
    proxy2 = _Proxy(dict(hub["pool"]))              # 新 proxy：池 = Hub（rev 11），丢了采纳

    def emit(et, pl):
        if not state["up"]:
            raise RuntimeError("Connection refused")
        return proxy2(et, pl)

    d2 = _mk(tmp_path, hub, emit)
    d2.startup_check()                              # 补采 refused
    assert path in d2._pending_edits and path in d2._adopted_sha and path not in d2._adopted_rev
    assert d2.pending_wait_s() is not None, "降级后要能被重发窗叫醒"
    assert open(path, encoding="utf-8").read() == written, "失败期间不覆盖"
    state["up"] = True
    t["now"] += 31
    d2.run_once()                                   # 无门铃：重发 → 新 proxy 采纳 → 池版回写
    assert len(proxy2.calls) == 1 and proxy2.pool["ldg-a1"].rev == 12
    assert "用户手记的一个疑点" in open(path, encoding="utf-8").read()
    assert d2._adopted_rev.get(path) == 12 and path not in d2._adopted_sha
    assert d2.pending_wait_s() is None


# ── ⑦ Hub 追上 ⇒ 按 rev 收敛 ──────────────────────────────────────────────

def test_converges_by_rev_when_model_writes_after_adoption(tmp_path):
    hub = {"pool": _pool_v1()}
    proxy = _Proxy(dict(hub["pool"]))
    d = _mk(tmp_path, hub, proxy)
    d.run_once()
    path = ledger_md_path(str(tmp_path), "u1", "ldg-a1")
    _edit(path)
    d.run_once()                                   # 采纳 → 磁盘 rev 12
    # 模型在池版上 update ⇒ Hub 事件带整本（rev 13、含用户条目 + Verified）
    led = add_entry(proxy.pool["ldg-a1"], "verified",
                    LedgerEntry(text="测试全绿", source="tool", ref="gate#2"),
                    actor=ACTOR_MODEL).model_copy(update={"rev": 13})
    hub["pool"] = {"ldg-a1": led}
    d.push()
    with structlog.testing.capture_logs() as cap:
        d.run_once()
    conv = [e for e in cap if e.get("event") == "flash_daemon_user_edit_converged"]
    assert len(conv) == 1 and conv[0]["by"] == "rev"
    final = open(path, encoding="utf-8").read()
    assert "测试全绿" in final and "用户手记的一个疑点" in final and _disk_rev(path) == 13
    assert path not in d._pending_edits and len(proxy.calls) == 1


# ── ⑥ server 端 ───────────────────────────────────────────────────────────

def test_server_returns_adopted_pool_version_and_flags_resend(monkeypatch):
    from bladex_proxy.config import ProxyConfig
    from bladex_proxy.server import create_app
    from fastapi.testclient import TestClient

    monkeypatch.setenv("BLADEX_AUTH_ENABLED", "false")
    app = create_app(ProxyConfig())

    class _Agency:
        def __init__(self):
            self.pool = {}

    prev = _pool_v1()["ldg-a1"].model_copy(update={"last_writer": "hermes:default"})
    with TestClient(app) as c:
        c.app.state.agency = _Agency()
        c.app.state.agency.pool = {"ldg-a1": prev}
        user_version = prev.model_copy(update={"title": "任务甲（用户改名）"})
        with structlog.testing.capture_logs() as cap:
            r = c.post("/admin/ledgers/user-edit", json={"ledger": user_version.model_dump(mode="json")})
        assert r.status_code == 200
        body = r.json()
        assert body["ledger"]["rev"] == prev.rev + 1 == body["rev"]
        assert body["ledger"]["title"] == "任务甲（用户改名）"
        assert body["ledger"]["last_writer"] == "user"
        assert body["same_as_last_adoption"] is False
        risk = [e for e in cap if e.get("event") == "admin_ledger_user_edit_clobber_risk"]
        assert len(risk) == 1 and risk[0]["same_as_last_adoption"] is False

        # 同一份用户版再发一次（= 修前 daemon 的重发形态）⇒ 标为重发，但不拒绝（兼容旧 daemon）
        c.app.state.agency.pool["ldg-a1"] = Ledger.model_validate(body["ledger"]).model_copy(
            update={"last_writer": "hermes:default"})
        with structlog.testing.capture_logs() as cap2:
            r2 = c.post("/admin/ledgers/user-edit", json={"ledger": user_version.model_dump(mode="json")})
        assert r2.status_code == 200 and r2.json()["same_as_last_adoption"] is True
        risk2 = [e for e in cap2 if e.get("event") == "admin_ledger_user_edit_clobber_risk"]
        assert len(risk2) == 1 and risk2[0]["same_as_last_adoption"] is True
