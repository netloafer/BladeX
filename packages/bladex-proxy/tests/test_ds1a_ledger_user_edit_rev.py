"""批 DS1a（DS-04 前置）：`POST /admin/ledgers/user-edit` 的可选乐观锁 `expected_rev`。

- 带了且与池里当前 rev 不符 ⇒ 409 + 当前 rev 与当前池版；**池与 Hub 一字不动**（没发生的编辑不落事件）。
- 带了且相符 / 不带 ⇒ 响应、池、Hub 事件与修前逐字相同（flash daemon 不带，不受影响）。
"""

from __future__ import annotations

import copy

from bladex_core.ledger import Ledger, new_ledger
from bladex_proxy.agency import AgencyRuntime
from bladex_proxy.config import ProxyConfig
from bladex_proxy.server import create_app
from fastapi.testclient import TestClient

_LID = "ldg-ds1a000001"


class _Ev:
    def __init__(self, etype: str, payload: dict) -> None:
        self.event_type = etype
        self.payload = payload
        self.ts = None


class _Hub:
    def __init__(self) -> None:
        self.appended: list[tuple[str, str, dict]] = []

    def append_admin_event(self, etype, matter_id, **payload):  # noqa: ANN001, ANN003
        self.appended.append((getattr(etype, "value", str(etype)), matter_id, payload))

    def scan_admin_events(self):
        return iter(())

    def scan_tombstones(self):
        return iter(())

    def catch_up(self) -> None:
        return None


def _seed() -> Ledger:
    return new_ledger(ledger_id=_LID, title="DS1a 乐观锁", goal="改账本不盖掉别人",
                      goal_source="user",
                      created_at="2026-10-25T10:00:00Z").model_copy(update={"rev": 11})


def _client(monkeypatch) -> tuple[TestClient, AgencyRuntime, _Hub]:
    monkeypatch.setenv("BLADEX_AUTH_ENABLED", "false")
    c = TestClient(create_app(ProxyConfig()))
    c.__enter__()
    hub = _Hub()
    ag = AgencyRuntime(hub=hub)
    ag.pool[_LID] = _seed()
    c.app.state.agency = ag
    return c, ag, hub


def _edited() -> dict:
    led = _seed().model_copy(update={"goal": "用户改过的目标"})
    return led.model_dump(mode="json")


def test_stale_expected_rev_is_409_and_writes_nothing(monkeypatch) -> None:
    c, ag, hub = _client(monkeypatch)
    try:
        before = copy.deepcopy(ag.pool[_LID].model_dump(mode="json"))
        r = c.post("/admin/ledgers/user-edit", json={"ledger": _edited(), "expected_rev": 10})
        assert r.status_code == 409, r.text
        body = r.json()
        assert body["status"] == "conflict"
        assert body["current_rev"] == 11 and body["expected_rev"] == 10
        assert body["ledger"]["goal"] == "改账本不盖掉别人"
        assert ag.pool[_LID].model_dump(mode="json") == before
        assert hub.appended == []
        assert not getattr(ag, "_last_user_edit_dump", {})
    finally:
        c.__exit__(None, None, None)


def test_expected_rev_for_ledger_not_in_pool_is_409(monkeypatch) -> None:
    c, ag, hub = _client(monkeypatch)
    try:
        led = _seed().model_copy(update={"ledger_id": "ldg-ds1a-notinpool"})
        r = c.post("/admin/ledgers/user-edit",
                   json={"ledger": led.model_dump(mode="json"), "expected_rev": 0})
        assert r.status_code == 409 and r.json()["current_rev"] is None, r.text
        assert "ldg-ds1a-notinpool" not in ag.pool
        assert hub.appended == []
    finally:
        c.__exit__(None, None, None)


def test_non_integer_expected_rev_is_400(monkeypatch) -> None:
    c, ag, hub = _client(monkeypatch)
    try:
        for bad in ("11", 11.0, True):
            r = c.post("/admin/ledgers/user-edit", json={"ledger": _edited(), "expected_rev": bad})
            assert r.status_code == 400, (bad, r.text)
        assert hub.appended == [] and ag.pool[_LID].rev == 11
    finally:
        c.__exit__(None, None, None)


def _run(monkeypatch, payload: dict) -> tuple[int, dict, dict, list]:
    c, ag, hub = _client(monkeypatch)
    try:
        r = c.post("/admin/ledgers/user-edit", json=payload)
        return r.status_code, r.json(), ag.pool[_LID].model_dump(mode="json"), hub.appended
    finally:
        c.__exit__(None, None, None)


def test_matching_or_absent_expected_rev_behaves_exactly_as_before(monkeypatch, fresh_redis_stream) -> None:
    without = _run(monkeypatch, {"ledger": _edited()})
    # 第二个 app 是独立对照、不是在测重启：共用 stream 会白等 4 s 判活窗口（批 GT #5）
    monkeypatch.setenv("BLADEX_REDIS_STREAM", fresh_redis_stream())
    matching = _run(monkeypatch, {"ledger": _edited(), "expected_rev": 11})
    assert without[0] == 200 and without[1]["rev"] == 12
    assert len(without[3]) == 1 and without[3][0][0] == "ledger_user_edit"
    assert matching == without
