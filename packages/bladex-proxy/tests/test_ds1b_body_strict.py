"""批 DS1b（MQ-R16）：admin 写端点的 body 全部严格——一个机制、守卫、调用方回放。

- **全部** `/admin` 写端点（POST/PUT/PATCH/DELETE）挂 `_strict_body(...)`：未知 body 键 ⇒ 400 并点名，
  形状与 `_strict_query` 的 400 一致；什么都不写。端点清单从 app 路由表取，不在测试里另抄一份。
- 守卫：新加一个不经 `_strict_body` 的写端点 ⇒ 红；有 `*Body` 模型的端点，允许名单必须**等于**模型字段。
- `user-edit`：`expected_rev` 拼错（`expectedRev`）⇒ 400、池与 Hub 一字不动（修前：静默忽略 ⇒ 整本覆盖）。
- `DELETE /admin/turns/{key}`：挂 `_strict_query()`（空集）。
- 回放：CLI / MCP / flash daemon / `export_backflow` / dashboard 的真实请求（截获 URL 与 body）原样打给
  服务端，都不是 400。
"""

from __future__ import annotations

import ast
import copy
import re
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from bladex_core.ledger import new_ledger
from bladex_proxy.admin_contract import ADMIN_CONTRACT_VERSION
from bladex_proxy.config import ProxyConfig
from bladex_proxy.server import create_app
from bladex_proxy.server.admin_api import _strict_body
from fastapi import Depends, FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from pydantic import BaseModel

_PKG = Path(__file__).resolve().parents[1] / "bladex_proxy"
_WRITE = frozenset({"POST", "PUT", "PATCH", "DELETE"})
_LID = "ldg-ds1b000001"


def _config() -> ProxyConfig:
    tmpdir = tempfile.mkdtemp(prefix="ds1b_strict_")
    return ProxyConfig(
        upstream_model="openai/test", upstream_api_key="sk-fake",
        rocksdb_path=f"{tmpdir}/rocksdb", index_path=f"{tmpdir}/index",
    )


@pytest.fixture()
def client():
    app = create_app(_config())
    with TestClient(app) as c:
        yield c


# ── 守卫 ──────────────────────────────────────────────────────────────────


def _write_routes(app: FastAPI) -> list[APIRoute]:
    return [r for r in app.routes if isinstance(r, APIRoute)
            and r.path.startswith("/admin") and set(r.methods) & _WRITE]


def _strict_allowed_of(route: APIRoute) -> frozenset[str] | None:
    """该路由依赖里 `_strict_body(...)` 的允许名单；没挂 ⇒ None。"""
    for dep in route.dependant.dependencies:
        allowed = getattr(dep.call, "strict_body_allowed", None)
        if allowed is not None:
            return allowed
    return None


def _model_fields_of(route: APIRoute) -> frozenset[str] | None:
    """有 pydantic body 模型的路由 ⇒ 模型字段（含别名）；否则 None。"""
    for p in route.dependant.body_params:
        t = p.field_info.annotation
        if isinstance(t, type) and issubclass(t, BaseModel):
            return frozenset(f.alias or n for n, f in t.model_fields.items())
    return None


def strict_body_violations(app: FastAPI) -> list[str]:
    """守卫本体：写端点没挂 `_strict_body` / 允许名单与 body 模型字段不符 ⇒ 列出来。"""
    out: list[str] = []
    for r in _write_routes(app):
        name = f"{sorted(r.methods)} {r.path}"
        allowed = _strict_allowed_of(r)
        if allowed is None:
            out.append(f"{name}: no _strict_body dependency")
            continue
        fields = _model_fields_of(r)
        if fields is not None and allowed != fields:
            out.append(f"{name}: _strict_body allows {sorted(allowed)} but the body model "
                       f"has {sorted(fields)}")
    return out


def test_every_admin_write_route_goes_through_strict_body() -> None:
    app = create_app(_config())
    routes = _write_routes(app)
    assert len(routes) >= 27, f"scanned too few /admin write routes ({len(routes)})"
    assert strict_body_violations(app) == []


class _B(BaseModel):
    a: str = ""


def test_guard_flags_a_new_write_route_without_strict_body() -> None:
    """反向验证守卫不是摆设：新加一条不经机制的写端点 / 名单与模型漂移 ⇒ 被点名。"""
    app = FastAPI()

    @app.post("/admin/new-thing")
    async def new_thing(body: _B) -> dict:
        return {}

    @app.post("/admin/drifted")
    async def drifted(body: _B, _k: None = Depends(_strict_body("a", "b"))) -> dict:
        return {}

    @app.post("/admin/ok")
    async def ok(body: _B, _k: None = Depends(_strict_body(_B))) -> dict:
        return {}

    @app.delete("/admin/raw-ok")
    async def raw_ok(_k: None = Depends(_strict_body())) -> dict:
        return {}

    bad = strict_body_violations(app)
    assert len(bad) == 2, bad
    assert any("/admin/new-thing" in b and "no _strict_body" in b for b in bad)
    assert any("/admin/drifted" in b for b in bad)


def test_tombstone_by_project_has_no_second_mechanism() -> None:
    """`TombstoneByProjectBody` 收进同一机制：不再 `extra="forbid"`、端点里不再手拆 ValidationError。"""
    from bladex_proxy.server.admin_api import TombstoneByProjectBody

    assert TombstoneByProjectBody.model_config.get("extra") != "forbid"
    tree = ast.parse((_PKG / "server" / "admin_api.py").read_text(encoding="utf-8"))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    consts = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)}
    assert "ValidationError" not in names and "extra_forbidden" not in consts


# ── 拼错 ⇒ 400，什么都不写 ────────────────────────────────────────────────


def _concrete(path: str) -> str:
    return re.sub(r"\{([a-z_]+)(:path)?\}", lambda m: f"x-{m.group(1)}", path)


def test_unknown_body_key_is_400_on_every_write_route(client) -> None:
    hub = client.app.state.hub
    before = (len(list(hub.scan_tombstones())), len(list(hub.scan_admin_events())))
    seen = 0
    for r in _write_routes(client.app):
        for method in sorted(set(r.methods) & _WRITE):
            resp = client.request(method, _concrete(r.path), json={"zz_unknown_key": 1})
            assert resp.status_code == 400, (method, r.path, resp.text)
            assert "zz_unknown_key" in resp.text and "unknown body key" in resp.text
            seen += 1
    assert seen >= 27
    assert (len(list(hub.scan_tombstones())), len(list(hub.scan_admin_events()))) == before


def test_misspelled_key_names_the_key_and_lists_accepted(client) -> None:
    r = client.post("/admin/matters/m-1/merge", json={"source_matterid": "m-2"})
    assert r.status_code == 400
    d = r.json()["detail"]
    assert "source_matterid" in d and "source_matter_id" in d
    assert list(client.app.state.hub.scan_admin_events()) == []


def test_known_keys_and_empty_bodies_still_pass(client) -> None:
    """拼对的键、`{}`、不带 body：都不是 400（下游照旧 404/503/200）。"""
    assert client.post("/admin/matters/m-1/close", json={}).status_code != 400
    assert client.post("/admin/matters/m-1/close").status_code != 400
    assert client.request("DELETE", "/admin/turns/u/a/s/000001").status_code != 400
    assert client.post("/admin/matters/m-1/rename", json={"title": "t"}).status_code != 400


def test_delete_turn_rejects_unknown_query(client) -> None:
    r = client.request("DELETE", "/admin/turns/u/a/s/000001?dry_run=true")
    assert r.status_code == 400, r.text
    assert "dry_run" in r.text and "unknown query parameter" in r.text
    assert list(client.app.state.hub.scan_tombstones()) == []


# ── user-edit：`expected_rev` 拼错 ⇒ 400，不覆盖 ──────────────────────────────


class _Hub:
    def __init__(self) -> None:
        self.appended: list = []

    def append_admin_event(self, etype, matter_id, **payload):  # noqa: ANN001, ANN003
        self.appended.append((etype, matter_id, payload))

    def scan_admin_events(self):
        return iter(())

    def scan_tombstones(self):
        return iter(())


def _seed():
    return new_ledger(ledger_id=_LID, title="DS1b", goal="原目标", goal_source="user",
                      created_at="2026-10-25T10:00:00Z").model_copy(update={"rev": 11})


@pytest.fixture()
def agency_client(client):
    from bladex_proxy.agency import AgencyRuntime

    hub = _Hub()
    ag = AgencyRuntime(hub=hub)
    ag.pool[_LID] = _seed()
    client.app.state.agency = ag
    return client, ag, hub


@pytest.mark.parametrize("typo", ["expectedRev", "expected_revision", "rev"])
def test_user_edit_misspelled_expected_rev_is_400_and_writes_nothing(agency_client, typo) -> None:
    c, ag, hub = agency_client
    before = copy.deepcopy(ag.pool[_LID].model_dump(mode="json"))
    edited = _seed().model_copy(update={"goal": "用户改过的目标"}).model_dump(mode="json")
    r = c.post("/admin/ledgers/user-edit", json={"ledger": edited, typo: 10})
    assert r.status_code == 400, r.text
    assert typo in r.json()["detail"] and "expected_rev" in r.json()["detail"]
    assert ag.pool[_LID].model_dump(mode="json") == before
    assert hub.appended == []


def test_user_edit_ds1_shape_and_daemon_shape_pass(agency_client) -> None:
    """DS1 将发 `{"ledger", "expected_rev"}`；flash daemon 发 `{"ledger"}`——都不是 400。"""
    c, ag, _hub = agency_client
    edited = _seed().model_copy(update={"goal": "改一"}).model_dump(mode="json")
    r = c.post("/admin/ledgers/user-edit", json={"ledger": edited, "expected_rev": 11})
    assert r.status_code == 200, r.text
    led = r.json()["ledger"]
    r = c.post("/admin/ledgers/user-edit", json={"ledger": led})
    assert r.status_code == 200, r.text


# ── 调用方回放 ─────────────────────────────────────────────────────────────


class _Stop(Exception):
    pass


def _capture_cli(monkeypatch, invoke) -> tuple[str, str, dict | None]:
    import bladex_proxy.cli as cli

    seen: dict = {}

    def fake_http_request(method, url, *, key=None, body=None, timeout=15.0, **_kw):  # noqa: ANN001, ANN003
        seen.update(method=method, url=url, body=body)
        raise _Stop

    monkeypatch.setattr(cli, "_http_request", fake_http_request)
    for mod in ("memory_cmds", "ledger_cmds"):
        monkeypatch.setattr(f"bladex_proxy.cli.{mod}._require_server_contract",
                            lambda what: ADMIN_CONTRACT_VERSION, raising=False)
    with pytest.raises(_Stop):
        invoke()
    return seen["method"], seen["url"], seen["body"]


def _replay(client, method: str, url: str, body: dict | None):
    parts = urlsplit(url)
    target = parts.path + (f"?{parts.query}" if parts.query else "")
    return client.request(method, target, json=body)


def _cli_cases():
    from bladex_proxy.cli import ledger_cmds as lc
    from bladex_proxy.cli import memory_cmds as mc
    from bladex_proxy.cli import ops_cmds as oc

    return {
        "memory forget": lambda: mc.memory_forget("f-1", yes=True),
        "tombstone-project dry": lambda: mc.memory_tombstone_project(["p-1"], dry_run=True, yes=True),
        "matter create": lambda: mc.matter_create("t", summary="s"),
        "matter assign": lambda: mc.matter_assign("m-1", "f-1", target_type="fact"),
        "matter merge": lambda: mc.matter_merge("m-1", "m-2"),
        "matter detach edge": lambda: mc.matter_detach("m-1", edge_id="e-1", target_key="",
                                                       target_type="fact"),
        "matter detach target": lambda: mc.matter_detach("m-1", edge_id="", target_key="f-1",
                                                         target_type="fact"),
        "matter split": lambda: mc.matter_split("m-1", new_title="n", edge_ids=["e-1"]),
        "matter close": lambda: mc.matter_close("m-1"),
        "matter rename": lambda: mc.matter_rename("m-1", "t2"),
        "ledger bind dry": lambda: lc._ledger_bind_legacy_matters(write=False, json_output=False),
        "ledger close": lambda: lc.ledger_close(_LID, reason="r"),
        "ledger reopen": lambda: lc.ledger_reopen(_LID, reason="r"),
        "ledger supersede": lambda: lc.ledger_supersede(_LID, by="ldg-other", reason="r"),
        "ledger tombstone dry": lambda: lc.ledger_tombstone(_LID, with_memory=True, dry_run=True,
                                                            yes=True),
        "ledger tombstone": lambda: lc.ledger_tombstone(_LID, with_memory=False, dry_run=False,
                                                        yes=True),
        "ledger tombstone-memory dry": lambda: lc.ledger_tombstone_memory(_LID, dry_run=True,
                                                                          yes=True),
        "queue flush": lambda: oc.queue_flush(apply=False, max_batches=1, sample=0),
        "sticky flush": lambda: oc.sticky_flush(agent="hermes"),
    }


@pytest.mark.parametrize("case", sorted(_cli_cases()))
def test_cli_requests_replay_without_400(client, monkeypatch, case) -> None:
    method, url, body = _capture_cli(monkeypatch, _cli_cases()[case])
    assert urlsplit(url).path.startswith("/admin/")
    r = _replay(client, method, url, body)
    assert r.status_code != 400, (case, body, r.text)


def test_mcp_remember_replays_without_400(client, monkeypatch) -> None:
    import bladex_mcp.server as mcp

    seen: dict = {}

    def fake(method, path, params=None, body=None):  # noqa: ANN001
        seen.update(method=method, path=path, body=body)
        raise _Stop

    monkeypatch.setattr(mcp, "_http_json", fake)
    fn = getattr(mcp.remember, "fn", mcp.remember)
    with pytest.raises(_Stop):
        fn("用户偏好中文回答", scope="")
    r = client.request(seen["method"], seen["path"], json=seen["body"])
    assert r.status_code != 400, r.text


def test_export_backflow_requests_replay_without_400(client) -> None:
    from bladex_proxy.export_backflow import AdminBackflowClient, BackflowEdit

    sent: list = []

    def requester(method, path, body):  # noqa: ANN001
        r = client.request(method, path, json=body)
        sent.append((method, path, body, r.status_code, r.text))
        return r.status_code, (r.json() if r.content else {})

    bf = AdminBackflowClient(requester)
    for edit in (
        BackflowEdit(kind="correct", origin="obsidian", submitter="v", fact_id="f-1", content="x"),
        BackflowEdit(kind="delete", origin="postgres", submitter="s", fact_id="f-1"),
        BackflowEdit(kind="assign", origin="obsidian", submitter="v", fact_id="f-1",
                     matter_id="m-2", from_matter_id="m-1"),
        BackflowEdit(kind="detach", origin="obsidian", submitter="v", fact_id="f-1",
                     from_matter_id="m-1"),
        BackflowEdit(kind="hard_rules", origin="obsidian", submitter="v",
                     rules=["MUST a"], previous=["MUST b"]),
    ):
        bf.apply(edit)
    assert len(sent) >= 6
    assert all(code != 400 for *_x, code, _t in sent), sent


def test_flash_daemon_user_edit_replays_without_400(agency_client, monkeypatch) -> None:
    """daemon 发的键从源码读（每个 `_emit("ledger_user_edit", {...})` 的字面量键），再经
    `admin_emit` 真正发出的字节回放。"""
    import urllib.request

    from bladex_proxy import flash_daemon

    tree = ast.parse((_PKG / "flash_daemon.py").read_text(encoding="utf-8"))
    keysets = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "_emit" and len(node.args) == 2
                and isinstance(node.args[1], ast.Dict)):
            keysets.add(tuple(sorted(k.value for k in node.args[1].keys)))
    assert keysets == {("ledger",)}, keysets

    c, ag, _hub = agency_client
    seen: dict = {}

    def fake_urlopen(req, timeout=5):  # noqa: ANN001
        seen.update(url=req.full_url, data=req.data)
        raise _Stop

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    emit = flash_daemon.admin_emit("http://t")
    with pytest.raises(_Stop):
        emit("ledger_user_edit", {"ledger": ag.pool[_LID].model_dump()})
    r = c.post(urlsplit(seen["url"]).path, content=seen["data"],
               headers={"Content-Type": "application/json"})
    assert r.status_code == 200, r.text


#: dashboard.html 现有 JS 发的 body（`api("POST", …, {...})` 字面量；本批不改 dashboard，
#: 下面逐个核对键名确实出现在页面源码里，防止这张表与页面漂移）。
_DASHBOARD_BODIES: list[tuple[str, str, dict]] = [
    ("POST", "/admin/facts", {"content": "x", "scope": ""}),
    ("POST", "/admin/scope/promote", {"target_type": "fact", "target_key": "f-1",
                                      "new_scope": "team:t"}),
    ("POST", "/admin/matters/m-1/close", {}),
    ("POST", "/admin/matters/m-1/rename", {"title": "t"}),
    ("POST", "/admin/matters/m-1/assign", {"target_key": "f-1", "target_type": "fact"}),
    ("POST", "/admin/matters/m-1/merge", {"source_matter_id": "m-2"}),
    ("POST", "/admin/matters/m-1/detach", {"edge_id": "e-1"}),
    ("POST", "/admin/matters/m-1/detach", {"target_key": "f-1", "target_type": "fact"}),
    ("POST", "/admin/matters/m-1/split", {"new_title": "n", "edge_ids": ["e-1"]}),
    ("POST", "/admin/matters", {"title": "t", "summary": ""}),
    ("POST", "/admin/agents/rules", {"rule": {"name": "a", "agent_id": "a"}}),
    ("POST", "/admin/agents/claim", {"from_agent_ids": ["unknown-1"], "to_agent_id": "a",
                                     "rule": {"name": "a", "agent_id": "a"},
                                     "confirm_merge": True}),
    ("POST", "/admin/agents/reattribute", {}),
    ("DELETE", "/admin/facts/f-1", None),
    ("DELETE", "/admin/turns/u/a/s/000001", None),
]


def test_dashboard_bodies_replay_without_400(client) -> None:
    html = (_PKG / "dashboard.html").read_text(encoding="utf-8")
    for method, path, body in _DASHBOARD_BODIES:
        for k in body or {}:
            assert re.search(rf"\b{k}\s*:", html) or f".{k}=" in html, (path, k)
        r = client.request(method, path, json=body)
        assert r.status_code != 400, (path, body, r.text)
