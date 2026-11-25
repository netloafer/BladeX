"""批 DS1a（MQ-R15）：三个管理写端点拼错参数 ⇒ 400；CLI 照旧可用。

- `POST /admin/sticky/flush` / `POST /admin/queue/flush`：挂 `_strict_query`，未知 query ⇒ 400。
  修前 `?agent=hermes`（应为 `agent_id`）被静默丢掉 ⇒ 退化成**全局**释放并回 200。
- `POST /admin/turns/tombstone-by-project`：body 改显式模型 `extra="forbid"`，未知 body 键 ⇒ 400。
- CLI 三条命令发出的请求**原样回放**给服务端，必须不是 400；参数名逐个核对在允许名单里
  （允许名单从服务端源码 / 模型里取，不在测试里另抄一份）。
"""

from __future__ import annotations

import ast
import tempfile
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

import pytest
from bladex_proxy.admin_contract import ADMIN_CONTRACT_VERSION
from bladex_proxy.config import ProxyConfig
from bladex_proxy.server import create_app
from bladex_proxy.server.admin_api import TombstoneByProjectBody
from fastapi.testclient import TestClient

_ADMIN_API = Path(__file__).resolve().parents[1] / "bladex_proxy" / "server" / "admin_api.py"


def _config() -> ProxyConfig:
    tmpdir = tempfile.mkdtemp(prefix="ds1a_strict_")
    return ProxyConfig(
        upstream_model="openai/test", upstream_api_key="sk-fake",
        rocksdb_path=f"{tmpdir}/rocksdb", index_path=f"{tmpdir}/index",
    )


@pytest.fixture()
def client():
    app = create_app(_config())
    with TestClient(app) as c:
        yield c


def _strict_allowed(fn_name: str) -> set[str]:
    """服务端源码里该路由 `_strict_query(...)` 的参数——允许名单的唯一来源。"""
    tree = ast.parse(_ADMIN_API.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == fn_name:
            for sub in ast.walk(node):
                if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
                        and sub.func.id == "_strict_query"):
                    return {str(a.value) for a in sub.args}
            raise AssertionError(f"{fn_name} has no _strict_query")
    raise AssertionError(f"{fn_name} not found")


# ── 拼错 ⇒ 400 ──────────────────────────────────────────────────────────


def test_sticky_flush_rejects_misspelled_agent_id(client) -> None:
    r = client.post("/admin/sticky/flush?agent=hermes", json={})
    assert r.status_code == 400, r.text
    assert "agent" in r.text and "unknown query parameter" in r.text


def test_queue_flush_rejects_misspelled_apply(client) -> None:
    r = client.post("/admin/queue/flush?aply=true", json={})
    assert r.status_code == 400, r.text
    assert "aply" in r.text


def test_tombstone_by_project_rejects_unknown_body_key(client) -> None:
    r = client.post("/admin/turns/tombstone-by-project",
                    json={"project_ids": ["p-abc"], "dryrun": True})
    assert r.status_code == 400, r.text
    assert "dryrun" in r.text
    # 什么都没写：hub 里没有任何墓碑
    assert list(client.app.state.hub.scan_tombstones()) == []


def test_known_params_still_accepted(client) -> None:
    """拼对的参数不受影响（未开路由 / 没有 pipeline worker 时回 503，但不是 400）。"""
    assert client.post("/admin/sticky/flush?agent_id=hermes", json={}).status_code != 400
    assert client.post("/admin/queue/flush?apply=false&max_batches=1&sample=0",
                       json={}).status_code != 400
    r = client.post("/admin/turns/tombstone-by-project",
                    json={"project_ids": ["p-abc"], "dry_run": True})
    assert r.status_code == 200 and r.json()["status"] == "dry_run", r.text


def test_tombstone_body_semantics_unchanged_for_known_keys(client) -> None:
    """模型只多做一件事（拒未知键）：缺省 dry_run 仍是 True，project_ids 非 list 仍 400。"""
    r = client.post("/admin/turns/tombstone-by-project", json={"project_ids": ["p-abc"]})
    assert r.status_code == 200 and r.json()["dry_run"] is True
    r = client.post("/admin/turns/tombstone-by-project", json={"project_ids": "p-abc"})
    assert r.status_code == 400 and "must be a list" in r.text


# ── CLI 照旧可用：真实发出的请求回放 + 参数名核对 ────────────────────────────


class _Stop(Exception):
    pass


def _capture_cli(monkeypatch, invoke) -> tuple[str, str, dict | None]:
    import bladex_proxy.cli as cli

    seen: dict = {}

    def fake_http_request(method, url, *, key=None, body=None, timeout=15.0, **_kw):  # noqa: ANN001, ANN003
        seen.update(method=method, url=url, body=body)
        raise _Stop

    monkeypatch.setattr(cli, "_http_request", fake_http_request)
    monkeypatch.setattr("bladex_proxy.cli.memory_cmds._require_server_contract",
                        lambda what: ADMIN_CONTRACT_VERSION)
    with pytest.raises(_Stop):
        invoke()
    return seen["method"], seen["url"], seen["body"]


def _replay(client, method: str, url: str, body: dict | None):
    parts = urlsplit(url)
    target = parts.path + (f"?{parts.query}" if parts.query else "")
    return client.request(method, target, json=body), dict(parse_qsl(parts.query))


@pytest.mark.parametrize("variant", ["defaults", "all_flags"])
def test_cli_queue_flush_params_are_all_known(client, monkeypatch, variant) -> None:
    from bladex_proxy.cli.ops_cmds import queue_flush

    kw = ({"apply": False, "max_batches": 200, "sample": 0} if variant == "defaults"
          else {"apply": True, "max_batches": 7, "sample": 3})
    method, url, body = _capture_cli(monkeypatch, lambda: queue_flush(**kw))
    r, qs = _replay(client, method, url, body)
    assert set(qs) <= _strict_allowed("flush_queue"), qs
    assert r.status_code != 400, r.text


@pytest.mark.parametrize("agent", ["", "hermes"])
def test_cli_sticky_flush_params_are_all_known(client, monkeypatch, agent) -> None:
    from bladex_proxy.cli.ops_cmds import sticky_flush

    method, url, body = _capture_cli(monkeypatch, lambda: sticky_flush(agent=agent))
    r, qs = _replay(client, method, url, body)
    assert set(qs) <= _strict_allowed("flush_sticky"), qs
    if agent:
        assert qs == {"agent_id": agent}
    assert r.status_code != 400, r.text


@pytest.mark.parametrize("dry_run", [True, False])
def test_cli_tombstone_project_body_keys_are_all_known(client, monkeypatch, dry_run) -> None:
    from bladex_proxy.cli.memory_cmds import memory_tombstone_project

    method, url, body = _capture_cli(
        monkeypatch, lambda: memory_tombstone_project(["p-abc"], dry_run=dry_run, yes=True))
    assert set(body or {}) <= set(TombstoneByProjectBody.model_fields), body
    assert not urlsplit(url).query
    if dry_run:  # 真跑不回放（会起后台 job），键已逐个核对
        r, _ = _replay(client, method, url, body)
        assert r.status_code == 200 and r.json()["status"] == "dry_run", r.text


def test_admin_contract_bumped_for_ds1a() -> None:
    """闸 B：未知 body 键改报 400 + user-edit 新增 expected_rev ⇒ 协议 +1。"""
    assert ADMIN_CONTRACT_VERSION == 3
