"""批 DS2 第三步 · 步骤 D：fact 更正收 `origin=dashboard`（G4）+ 修改史读面（G5①②）。

页面行为与「页面 ↔ 回流通道」事件对账在 `test_dashboard.py`（要 node）。
"""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime

import pytest
from bladex_core.fact import Fact
from bladex_proxy.cli import memory_cmds
from bladex_proxy.config import ProxyConfig
from bladex_proxy.models import (
    BACKFLOW_ORIGINS,
    FACT_CORRECT_ORIGINS,
    AdminEventType,
)
from bladex_proxy.server import create_app
from bladex_proxy.storage.memory_index import MemoryIndex
from fastapi.testclient import TestClient
from typer.testing import CliRunner

FID, OTHER = "f-history-1", "f-history-2"


def _config() -> ProxyConfig:
    d = tempfile.mkdtemp(prefix="ds2_hist_")
    idx = MemoryIndex(f"{d}/index", embedder=None, read_only=False)
    idx.open()
    for fid, text in ((FID, "原文字"), (OTHER, "另一条")):
        idx.add_fact(Fact(id=fid, content=text, source_user_id="u1", source_session="s1",
                          created_at=datetime.now(UTC)))
    idx.close()
    return ProxyConfig(upstream_model="openai/test", upstream_api_key="sk-fake",
                       rocksdb_path=f"{d}/rocksdb", index_path=f"{d}/index")


@pytest.fixture()
def client():
    app = create_app(_config())
    with TestClient(app) as c:
        yield c


def _correct(c, fid=FID, content="新文字", origin="dashboard", submitter=""):
    return c.post(f"/admin/facts/{fid}/correct",
                  json={"content": content, "origin": origin, "submitter": submitter})


def _events(c) -> list:
    return [e for _k, e in c.app.state.hub.scan_admin_events()
            if e.event_type == AdminEventType.FACT_CORRECT]


# ── G4：origin=dashboard ─────────────────────────────────────────────────────

def test_correct_records_dashboard_as_the_origin(client) -> None:
    r = _correct(client)
    assert r.status_code in (200, 202), r.text
    (ev,) = _events(client)
    assert ev.payload["origin"] == "dashboard" and ev.payload["submitter"] == ""
    assert ev.payload["previous_content"] == "原文字"
    assert ev.payload["entity"]["content"] == "新文字"


def test_dashboard_origin_refuses_a_free_text_submitter(client) -> None:
    r = _correct(client, submitter="someone")
    assert r.status_code == 400 and "submitter" in r.json()["error"]["message"]
    assert _events(client) == []


@pytest.mark.parametrize("origin", ["", "dash", "cli", "user", "Dashboard"])
def test_correct_rejects_any_other_origin_and_writes_nothing(client, origin) -> None:
    assert _correct(client, origin=origin).status_code == 400
    assert _events(client) == []


def test_backflow_origins_still_correct(client) -> None:
    assert _correct(client, origin="obsidian", submitter="/vault").status_code in (200, 202)
    assert _events(client)[0].payload["submitter"] == "/vault"


def test_opening_correct_does_not_widen_the_other_write_routes(client) -> None:
    assert "dashboard" not in BACKFLOW_ORIGINS and "dashboard" in FACT_CORRECT_ORIGINS
    assert client.delete(f"/admin/facts/{FID}", params={"origin": "dashboard"}).status_code == 400
    assert client.post("/admin/hard-rules/set", json={"rules": ["MUST x"], "origin": "dashboard"}).status_code == 403
    assert client.post("/admin/matters/m-1/assign", json={
        "target_type": "fact", "target_key": FID, "origin": "dashboard"}).status_code == 400


# ── G5①：修改史 ──────────────────────────────────────────────────────────────

def _history(c, fid=FID):
    return c.get(f"/admin/facts/{fid}/history")


def test_history_of_an_untouched_fact(client) -> None:
    d = _history(client).json()
    assert d["fact_state"] == "present" and d["content"] == "原文字"
    assert d["events"] == [] and d["total"] == 0
    assert d["source"]["source_session"] == "s1" and d["source"]["source_user_id"] == "u1"
    assert set(d["supersession"]) == {"superseded_by", "t_invalid", "corrects"}


def test_history_lists_corrections_and_the_tombstone_in_time_order(client) -> None:
    assert _correct(client, content="第二版").status_code in (200, 202)
    assert _correct(client, OTHER, content="别的 fact").status_code in (200, 202)
    assert client.delete(f"/admin/facts/{FID}").status_code in (200, 202)
    r = _history(client)
    assert r.status_code == 200, r.text
    d = r.json()
    assert [e["type"] for e in d["events"]] == ["fact_correct", "tombstone"]
    assert d["total"] == 2
    correct, tomb = d["events"]
    assert correct["origin"] == "dashboard" and correct["content"] == "第二版"
    assert correct["previous_content"] == "原文字"
    assert tomb["source"] == "user" and tomb["origin"] == ""
    assert tomb["at"] >= correct["at"]


def test_history_survives_deletion_and_says_the_fact_is_gone(client) -> None:
    assert client.delete(f"/admin/facts/{FID}").status_code in (200, 202)
    # proxy 的 Index 读句柄追新有节流：直接问写端点也已落的 Hub，fact_state 可能还是 present
    client.app.state.index = _AbsentIndex()
    d = _history(client).json()
    assert d["fact_state"] == "absent" and d["content"] is None and d["source"] is None
    assert [e["type"] for e in d["events"]] == ["tombstone"]


class _AbsentIndex:
    def get_fact(self, _fid):  # noqa: ANN001, ANN201
        return None

    def close(self) -> None:
        pass


class _BrokenIndex:
    def get_fact(self, _fid):  # noqa: ANN001, ANN201
        raise RuntimeError("index down")

    def close(self) -> None:
        pass


def test_history_unknown_fact_is_404(client) -> None:
    client.app.state.index = _AbsentIndex()
    assert _history(client, "f-nope").status_code == 404


@pytest.mark.parametrize("index", [None, _BrokenIndex()])
def test_history_without_an_index_is_no_reading_not_not_found(client, index) -> None:
    client.app.state.index = index
    r = _history(client, "f-nope")
    assert r.status_code == 200
    assert r.json()["fact_state"] == "no_reading" and r.json()["content"] is None


def test_history_is_open_to_new_event_types(client, monkeypatch) -> None:
    """BB 步骤 2 的 audience_set + kind：加一项登记，响应结构不变（拍板 (67)-c）。"""
    from bladex_proxy import admin_read

    def _row(key, ev, fact_id):  # noqa: ANN001, ANN202
        if ev.payload.get("fact_id") != fact_id:
            return None
        return {"type": "audience_set", "at": ev.ts.isoformat(), "key": key,
                "origin": "", "kind": ev.payload.get("kind")}

    monkeypatch.setitem(admin_read._FACT_HISTORY_EVENTS, AdminEventType.AUDIENCE_SET, _row)
    client.app.state.hub.append_admin_event(AdminEventType.AUDIENCE_SET, "", FID,
                                            fact_id=FID, kind="contradiction_mark")
    d = _history(client).json()
    assert [(e["type"], e["kind"]) for e in d["events"]] == [("audience_set", "contradiction_mark")]


# ── G5②：bladex memory show --history ───────────────────────────────────────

@pytest.fixture()
def cli_on_server(client, monkeypatch):
    calls = []

    def fake(method, path, body=None, params=None, timeout=15.0):  # noqa: ANN001
        r = client.request(method, path, json=body)
        calls.append((method, path, r.text))
        return r.status_code, r.json()

    monkeypatch.setattr(memory_cmds, "_admin_call", fake)
    return calls


def _run(*argv: str):
    from bladex_proxy.cli import app as cli_app

    return CliRunner().invoke(cli_app, list(argv))


def test_cli_history_json_is_the_endpoint_response(client, cli_on_server) -> None:
    assert _correct(client).status_code in (200, 202)
    res = _run("memory", "show", FID, "--history", "--json")
    assert res.exit_code == 0, res.output
    assert cli_on_server[-1][1] == f"/admin/facts/{FID}/history"
    printed = [ln for ln in res.stdout.splitlines() if ln.startswith('{"fact_id"')]
    assert printed == [json.dumps(json.loads(cli_on_server[-1][2]), ensure_ascii=False)]
    assert json.loads(printed[0]) == _history(client).json()


def test_cli_history_text_shows_who_and_the_old_text(client, cli_on_server) -> None:
    assert _correct(client).status_code in (200, 202)
    res = _run("memory", "show", FID, "--history")
    assert res.exit_code == 0, res.output
    assert "fact_correct  origin=dashboard" in res.stdout
    assert "previous_content: 原文字" in res.stdout and "content: 新文字" in res.stdout


def test_cli_history_says_when_there_is_nothing(client, cli_on_server) -> None:
    res = _run("memory", "show", FID, "--history")
    assert "no corrections or deletion recorded" in res.stdout


def test_cli_lines_keep_unknown_event_types() -> None:
    lines = memory_cmds._fact_history_lines({
        "fact_id": "f", "fact_state": "no_reading", "content": None, "source": None, "supersession": None,
        "events": [{"type": "audience_set", "at": "t", "key": "k", "origin": "", "kind": "mark"}]})
    text = "\n".join(lines)
    assert "not readable" in text and "audience_set" in text and "kind: mark" in text


def test_cli_json_without_history_is_an_error(monkeypatch, capsys) -> None:
    from bladex_proxy import cli

    called = []
    monkeypatch.setattr(memory_cmds, "_admin_call", lambda *a, **k: called.append(a) or (200, {}))
    assert cli.main(["memory", "show", FID, "--json"]) == 2
    assert called == [] and "--history" in capsys.readouterr().err


def test_plain_memory_show_is_unchanged(client, cli_on_server) -> None:
    res = _run("memory", "show", FID)
    assert res.exit_code == 0
    assert cli_on_server[-1][1] == f"/admin/facts/{FID}"
