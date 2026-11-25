"""批 BB · 2b · 端点 / CLI / status：origin 闭集、只追加事件、关着时照常可读、`status` 一行。"""

from __future__ import annotations

import json
import tempfile
from typing import Any

import pytest
from _bb_support import mk_fact
from bladex_core import consistency as c
from bladex_proxy.cli import lifecycle, memory_cmds
from bladex_proxy.config import ProxyConfig
from bladex_proxy.models import AdminEvent, AdminEventType
from bladex_proxy.server import create_app
from bladex_proxy.storage.memory_index import MemoryIndex
from fastapi.testclient import TestClient
from typer.testing import CliRunner

FA, FB, FC = mk_fact("fa", "端口是 6379", day=2), mk_fact("fb", "端口是 6380", day=3), mk_fact("fc", "无关", day=4)


def _config() -> ProxyConfig:
    d = tempfile.mkdtemp(prefix="bb2_ep_")
    ix = MemoryIndex(f"{d}/index", embedder=None, read_only=False)
    ix.open()
    for f in (FA, FB, FC):
        ix.add_fact(f)
    ev = AdminEvent(event_type=AdminEventType.AUDIENCE_SET, matter_id="", target_key="fa|fb",
                    payload=c.build_flag_payload(FB, FA, judge_model="m", reason="理由", rank=2, cos=0.93))
    ix._apply_admin_event(ev)
    ix.contradiction_kv_put("contradiction_judged/fa|fb/" + "-".join(
        [c.content_sha(FA), c.content_sha(FB)]), {"record": {"verdict": "矛盾", "reason": "全文理由"}})
    ix.close()
    return ProxyConfig(upstream_model="openai/test", upstream_api_key="sk-fake",
                       rocksdb_path=f"{d}/rocksdb", index_path=f"{d}/index")


@pytest.fixture()
def client():
    app = create_app(_config())
    with TestClient(app) as cl:
        yield cl


def _events(cl: TestClient) -> list[AdminEvent]:
    return [e for _k, e in cl.app.state.hub.scan_admin_events()]


def _flag_body(new=FB, old=FA) -> dict[str, Any]:
    p = c.build_flag_payload(new, old, judge_model="m", reason="r", rank=1, cos=0.9)
    return {"pair_key": c.pair_key(new.id, old.id), "payload": p}


# ── GET：关着也能读 ───────────────────────────────────────────────────────


def test_get_lists_flags_even_when_switch_is_off(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(c.ENABLED_FLAG, raising=False)
    r = client.get("/admin/contradictions")
    assert r.status_code == 200
    d = r.json()
    assert d["total"] == 1 and d["items"][0]["pair_id"] == "fa|fb" and d["items"][0]["status"] == "unsettled"
    assert d["items"][0]["reason"] == "全文理由"


def test_get_filters_by_status_and_rejects_unknown_status(client: TestClient) -> None:
    assert client.get("/admin/contradictions", params={"status": "confirmed"}).json()["total"] == 0
    assert client.get("/admin/contradictions", params={"status": "unsettled"}).json()["total"] == 1
    assert client.get("/admin/contradictions", params={"status": "bogus"}).status_code == 400


# ── POST flag ─────────────────────────────────────────────────────────────


def test_post_flag_appends_exactly_one_audience_set_event(client: TestClient) -> None:
    n0 = len(_events(client))
    r = client.post("/admin/contradictions", json=_flag_body(FC, FA))
    assert r.status_code == 200, r.text
    new = _events(client)[n0:]
    assert len(new) == 1 and new[0].event_type == AdminEventType.AUDIENCE_SET and new[0].matter_id == ""
    assert new[0].payload["kind"] == c.KIND_FLAG and not (c.FORBIDDEN_PAYLOAD_KEYS & set(new[0].payload))


@pytest.mark.parametrize("mutate", ["fact_id", "audience", "unsorted", "key_mismatch", "bad_kind", "verdict_payload"])
def test_post_flag_rejects_bad_payloads_and_writes_nothing(client: TestClient, mutate: str) -> None:
    body = _flag_body()
    p = body["payload"]
    if mutate in ("fact_id", "audience"):
        p[mutate] = "fa" if mutate == "fact_id" else "all"
    elif mutate == "unsorted":
        p["pair"] = ["fb", "fa"]
    elif mutate == "key_mismatch":
        body["pair_key"] = "fa|fc"
    elif mutate == "bad_kind":
        p["kind"] = "something_else"
    else:
        body["payload"] = c.build_verdict_payload(("fa", "fb"), ("1", "2"), verdict="confirmed", origin="cli")
    n0 = len(_events(client))
    assert client.post("/admin/contradictions", json=body).status_code == 400
    assert len(_events(client)) == n0


def test_post_flag_rejects_unknown_body_key(client: TestClient) -> None:
    body = {**_flag_body(), "extra": 1}
    assert client.post("/admin/contradictions", json=body).status_code == 400


# ── POST settle ───────────────────────────────────────────────────────────


def test_settle_appends_verdict_and_never_touches_facts(client: TestClient) -> None:
    n0 = len(_events(client))
    r = client.post("/admin/contradictions/fa%7Cfb/settle",
                    json={"verdict": "not_same_thing", "origin": "cli", "note": "不是一回事"})
    assert r.status_code == 200, r.text
    (ev,) = _events(client)[n0:]
    assert ev.event_type == AdminEventType.AUDIENCE_SET and ev.payload["kind"] == c.KIND_VERDICT
    assert ev.payload["origin"] == "cli" and ev.payload["pair"] == ["fa", "fb"]
    assert ev.payload["content_sha"] == [c.content_sha(FA), c.content_sha(FB)]
    assert not list(client.app.state.hub.scan_tombstones())


@pytest.mark.parametrize("origin", ["model", "", "api", "Dashboard"])
def test_settle_origin_is_a_closed_set_model_is_rejected(client: TestClient, origin: str) -> None:
    n0 = len(_events(client))
    r = client.post("/admin/contradictions/fa%7Cfb/settle", json={"verdict": "confirmed", "origin": origin})
    assert r.status_code == 400 and len(_events(client)) == n0


def test_settle_dashboard_origin_is_accepted(client: TestClient) -> None:
    assert client.post("/admin/contradictions/fa%7Cfb/settle",
                       json={"verdict": "confirmed", "origin": "dashboard"}).status_code == 200


def test_settle_validation_and_not_found(client: TestClient) -> None:
    post = client.post
    assert post("/admin/contradictions/fa%7Cfb/settle", json={"verdict": "delete", "origin": "cli"}).status_code == 400
    assert post("/admin/contradictions/fa%7Cfc/settle", json={"verdict": "confirmed", "origin": "cli"}).status_code == 404
    assert post("/admin/contradictions/onlyone/settle", json={"verdict": "confirmed", "origin": "cli"}).status_code == 400
    assert post("/admin/contradictions/fa%7Cfb/settle",
                json={"verdict": "confirmed", "origin": "cli", "oops": 1}).status_code == 400


# ── status / history ──────────────────────────────────────────────────────


def _hb(monkeypatch: pytest.MonkeyPatch, hb: dict | None) -> None:
    async def _fake(_pipeline: Any) -> dict | None:  # noqa: ANN401
        return hb

    monkeypatch.setattr("bladex_proxy.admin_read._read_consolidator_heartbeat", _fake)


def test_status_block_follows_the_consolidator_report_not_the_proxy_env(
        client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """批 BB · 2c：判定在 consolidator 里跑，`/admin/status` 以它心跳报的值为准。"""
    # proxy 进程开关关、consolidator 开 ⇒ 出现
    monkeypatch.delenv(c.ENABLED_FLAG, raising=False)
    _hb(monkeypatch, {"contradiction_enabled": "1", "fresh": True})
    blk = client.get("/admin/status").json()["contradiction"]
    assert blk["enabled"] is True and blk["daily_input_tokens_cap"] == 400000 and blk["per_fact_cap"] == 6
    assert blk["pending"] == 0 and blk["daily_input_tokens_used"] == 0
    # 反过来：proxy 开、consolidator 关 ⇒ 不出现
    monkeypatch.setenv(c.ENABLED_FLAG, "1")
    _hb(monkeypatch, {"contradiction_enabled": "0", "fresh": True})
    assert "contradiction" not in client.get("/admin/status").json()


def test_status_has_no_block_when_consolidator_never_reported(
        client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(c.ENABLED_FLAG, "1")  # proxy 自己的 env 开着也不算数
    for hb in (None, {}, {"state": "idle", "fresh": True}):  # 没起 / 心跳过期 / 旧版本不带这个字段
        _hb(monkeypatch, hb)
        assert "contradiction" not in client.get("/admin/status").json()


def test_status_source_never_reads_the_proxy_env_flag() -> None:
    """反向变异的静态面：把 `_contradiction_status` 改回读 `flag_enabled(ENABLED_FLAG)` ⇒ 这条红。"""
    import inspect

    from bladex_proxy import admin_read

    src = inspect.getsource(admin_read)
    block = src[src.index("def _contradiction_status"):src.index("contradiction_info = await")]
    assert "flag_enabled" not in block and "ENABLED_FLAG" not in block


def test_heartbeat_carries_the_switch_on_every_beat() -> None:
    from bladex_proxy.sync_control import HEARTBEAT_CONTRADICTION_FIELD, SyncControl

    class _R:
        def __init__(self) -> None:
            self.h: dict[str, str] = {}

        def hset(self, _k: str, mapping: dict) -> None:
            self.h.update(mapping)

        def expire(self, *_a: Any) -> None: ...

    r = _R()
    ctl = SyncControl(r)
    ctl.set_beat_base(**{HEARTBEAT_CONTRADICTION_FIELD: "1"})
    ctl.beat(state="running", phase="scan")
    assert r.h[HEARTBEAT_CONTRADICTION_FIELD] == "1" and r.h["phase"] == "scan"
    r.h.clear()
    ctl.beat(state="idle", phase="sleeping")  # 另一种心跳也带
    assert r.h[HEARTBEAT_CONTRADICTION_FIELD] == "1"


def test_status_line_only_when_enabled() -> None:
    assert lifecycle._contradiction_status_line({}) is None
    assert lifecycle._contradiction_status_line(None) is None
    line = lifecycle._contradiction_status_line({"contradiction": {
        "enabled": True, "pending": 3, "daily_input_tokens_used": 12000, "daily_input_tokens_cap": 400000}})
    assert line is not None and "pending 3" in line and "12,000 / 400,000" in line


def test_status_command_prints_the_line_through_the_helper() -> None:
    import inspect

    src = inspect.getsource(lifecycle.status)
    assert "_contradiction_status_line(admin)" in src


def test_fact_history_lists_the_contradiction_events(client: TestClient) -> None:
    client.post("/admin/contradictions/fa%7Cfb/settle", json={"verdict": "confirmed", "origin": "cli"})
    d = client.get("/admin/facts/fa/history").json()
    kinds = [e.get("kind") for e in d["events"] if e["type"] == "audience_set"]
    assert kinds == [c.KIND_VERDICT]
    assert d["events"][0]["other_fact_id"] == "fb"


# ── CLI ───────────────────────────────────────────────────────────────────


class _Rec:
    def __init__(self, resp: dict[tuple[str, str], tuple[int, dict]] | None = None) -> None:
        self.calls: list[tuple[str, str, dict | None, dict | None]] = []
        self.resp = resp or {}

    def __call__(self, method: str, path: str, body: dict | None = None, params: dict | None = None):
        self.calls.append((method, path, body, params))
        return self.resp.get((method, path), (200, {"status": "ok"}))


@pytest.fixture()
def rec(monkeypatch: pytest.MonkeyPatch) -> _Rec:
    r = _Rec()
    monkeypatch.setattr(memory_cmds, "_admin_call", r)
    return r


_LIST = {"total": 1, "items": [{
    "pair_id": "fa|fb", "status": "unsettled", "flagged_at": "2026-11-16T00:00:00+00:00", "judge_model": "m",
    "facts": [{"id": "fa", "content": "端口是 6379"}, {"id": "fb", "content": "端口是 6380"}], "reason": "两个端口不同"}]}


def _run(args: list[str]):
    return CliRunner().invoke(memory_cmds.memory_app, args)


def test_cli_conflicts_maps_to_the_endpoint_and_prints_its_fields(rec: _Rec) -> None:
    rec.resp[("GET", "/admin/contradictions")] = (200, _LIST)
    res = _run(["conflicts", "--status", "unsettled", "--limit", "5"])
    assert res.exit_code == 0, res.output
    assert rec.calls == [("GET", "/admin/contradictions", None, {"status": "unsettled", "limit": 5, "offset": 0})]
    for needle in ("fa|fb", "[unsettled]", "端口是 6379", "端口是 6380", "两个端口不同"):
        assert needle in res.output


def test_cli_conflicts_json_is_the_endpoints_json_verbatim(rec: _Rec) -> None:
    rec.resp[("GET", "/admin/contradictions")] = (200, _LIST)
    res = _run(["conflicts", "--json"])
    assert json.loads(res.output) == _LIST


def test_cli_settle_posts_origin_cli_and_encodes_the_pair(rec: _Rec) -> None:
    rec.resp[("POST", "/admin/contradictions/fa%7Cfb/settle")] = (
        200, {"status": "ok", "pair_id": "fa|fb", "verdict": "confirmed", "origin": "cli"})
    res = _run(["settle", "fa|fb", "--verdict", "confirmed", "--note", "确认"])
    assert res.exit_code == 0, res.output
    assert rec.calls == [("POST", "/admin/contradictions/fa%7Cfb/settle", {
        "verdict": "confirmed", "origin": "cli", "note": "确认"}, None)]
    assert "fa|fb" in res.output and "confirmed" in res.output


def test_cli_has_no_flag_to_choose_origin_model(rec: _Rec) -> None:
    res = _run(["settle", "fa|fb", "--verdict", "confirmed", "--origin", "model"])
    assert res.exit_code != 0 and rec.calls == []


def test_cli_surfaces_server_errors(rec: _Rec) -> None:
    rec.resp[("POST", "/admin/contradictions/fa%7Cfx/settle")] = (
        404, {"error": {"message": "no contradiction flagged for this pair", "type": "not_found"}})
    res = _run(["settle", "fa|fx", "--verdict", "confirmed"])
    assert "no contradiction flagged" in res.output


def test_consolidator_wires_the_switch_into_its_heartbeat() -> None:
    """接线的静态面：consolidator 必须把开关（∧ 检测器装配成功）交给 `set_beat_base`，否则状态行永远「没报过」。"""
    from pathlib import Path

    from bladex_proxy import consolidator

    src = Path(consolidator.__file__).read_text(encoding="utf-8")
    i = src.index("control.set_beat_base(")
    assert "HEARTBEAT_CONTRADICTION_FIELD" in src[i:i + 200] and "contradiction_detector is not None" in src[i:i + 300]
