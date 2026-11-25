"""批 AC2 步骤 2：`GET /admin/usage` + `bladex usage`。

- 端点输出**就是** `usage.aggregate(hub.usage_rows(), …)`，不再加工；未知参数 400（`_strict_query`）；
- 整段经 `run_blocking`：DS1a 的 AST 守卫自动扫到新路由（`test_ds1a_admin_blocking_guard.py`），
  这里另加事件循环用例——行缓存换成 `time.sleep` 的桩，跑着的时候 `/ready` 1 s 内回来；
- `bladex usage --json` 打印的就是端点返回的那份 JSON，逐字相等；CLI 不自己算（AST 守卫 + 运行时：
  CLI 的输出只能来自这次 HTTP 响应）。
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import json
import tempfile
import textwrap
import time
from datetime import UTC, datetime, timedelta, timezone
from urllib.parse import urlencode

import httpx
import pytest
from bladex_proxy.config import ProxyConfig
from bladex_proxy.models import Identity, ResponseMeta, TombstoneTargetType, Turn
from bladex_proxy.server import create_app
from bladex_proxy.usage import aggregate
from fastapi.testclient import TestClient
from typer.testing import CliRunner

CST = timezone(timedelta(hours=8))
_BASE = datetime(2026, 10, 20, 1, 0, tzinfo=UTC)


def _config() -> ProxyConfig:
    tmpdir = tempfile.mkdtemp(prefix="ac2_usage_")
    return ProxyConfig(
        upstream_model="openai/test", upstream_api_key="sk-fake",
        rocksdb_path=f"{tmpdir}/rocksdb", index_path=f"{tmpdir}/index",
    )


def _seed(hub, n: int = 30) -> list[str]:  # noqa: ANN001
    keys = []
    for i in range(n):
        agent = ("claude-code", "codex")[i % 2]
        key = f"local/{agent}/s{i % 3}/{i:06d}"
        hub.put(key, Turn(
            identity=Identity(user_id="local", agent_id=agent, session_id=f"s{i % 3}"),
            model=("openai/glm-5.3", "anthropic/glm-5.3-flash")[i % 2],
            ts=_BASE + timedelta(hours=7 * i),
            response_meta=ResponseMeta(
                usage={"prompt": 100 + i, "completion": 5, "total": 105 + i},
                usage_raw={"prompt_tokens": 100 + i, "completion_tokens": 5,
                           "total_tokens": 105 + i,
                           "prompt_tokens_details": {"cached_tokens": 64}} if i % 4 else {}),
        ))
        keys.append(key)
    return keys


@pytest.fixture()
def client():
    app = create_app(_config())
    with TestClient(app) as c:
        yield c


def test_endpoint_is_aggregate_of_the_row_cache(client) -> None:  # noqa: ANN001
    hub = client.app.state.hub
    keys = _seed(hub)
    hub.append_tombstone(TombstoneTargetType.TURN, keys[3])
    since, until = _BASE + timedelta(days=1), _BASE + timedelta(days=7)
    r = client.get("/admin/usage", params={
        "since": since.isoformat(), "until": until.isoformat(), "tz": "+08:00",
        "agent": "codex"})
    assert r.status_code == 200, r.text
    expected = aggregate(((x.key, x) for x in hub.usage_rows()), since=since, until=until,
                         agent="codex", tz=CST)
    assert r.json() == json.loads(json.dumps(expected))
    assert keys[3] not in json.dumps(r.json())
    assert r.json()["totals"]["all"]["turns"] > 0


def test_naive_since_is_read_in_tz_and_iana_names_work(client) -> None:  # noqa: ANN001
    hub = client.app.state.hub
    _seed(hub)
    a = client.get("/admin/usage", params={"since": "2026-10-21T09:00:00",
                                           "tz": "Asia/Shanghai"}).json()
    b = client.get("/admin/usage", params={"since": "2026-10-21T01:00:00+00:00",
                                           "tz": "+08:00"}).json()
    assert a["totals"] == b["totals"] and a["groups"] == b["groups"]


@pytest.mark.parametrize("params", [
    {"agnet": "codex"}, {"day": "3"}, {"days": "1", "since": "2026-10-01"},
    {"days": "0"}, {"tz": "Mars/Base"}, {"since": "yesterday"}])
def test_bad_or_unknown_params_are_400(client, params) -> None:  # noqa: ANN001
    assert client.get("/admin/usage", params=params).status_code == 400


def test_days_window(client) -> None:  # noqa: ANN001
    hub = client.app.state.hub
    now = datetime.now(UTC)
    for i, age in enumerate((1, 5, 40)):
        hub.put(f"local/a/s/{i:06d}", Turn(identity=Identity(user_id="local", agent_id="a",
                                                               session_id="s"),
                                           ts=now - timedelta(days=age),
                                           response_meta=ResponseMeta(usage={"prompt": 1})))
    assert client.get("/admin/usage", params={"days": 30}).json()["totals"]["all"]["turns"] == 2
    assert client.get("/admin/usage").json()["totals"]["all"]["turns"] == 3


# ── CLI ──────────────────────────────────────────────────────────────────────


def test_cli_json_is_the_endpoint_response_verbatim(client, monkeypatch) -> None:  # noqa: ANN001
    from bladex_proxy.cli import app as cli_app
    from bladex_proxy.cli import ops_cmds

    _seed(client.app.state.hub)
    calls: list[tuple[str, str, str]] = []

    def _fake_admin_call(method, path, body=None, params=None, timeout=15.0):  # noqa: ANN001, ANN202
        url = path
        if params:
            url += "?" + urlencode({k: v for k, v in params.items() if v not in ("", None)})
        r = client.request(method, url)
        calls.append((method, url, r.text))
        return r.status_code, r.json()

    monkeypatch.setattr(ops_cmds, "_admin_call", _fake_admin_call)
    res = CliRunner().invoke(cli_app, ["usage", "--json", "--days", "30", "--agent", "codex"])
    assert res.exit_code == 0, res.output
    assert len(calls) == 1 and calls[0][1].startswith("/admin/usage?")
    assert "days=30" in calls[0][1] and "agent=codex" in calls[0][1]
    # 进程内的服务端 structlog 也写 stdout（同一个 CliRunner 捕获），取 CLI 打印的那一行 JSON
    printed_lines = [ln for ln in res.stdout.splitlines() if ln.startswith('{"window"')]
    assert len(printed_lines) == 1, res.stdout
    printed = printed_lines[0]
    assert json.loads(printed) == json.loads(calls[0][2])
    assert printed == json.dumps(json.loads(calls[0][2]), ensure_ascii=False)

    # 表格形态：同一份响应排版，不报错、带上合计
    res = CliRunner().invoke(cli_app, ["usage", "--days", "30"])
    assert res.exit_code == 0, res.output
    assert "Totals" in res.stdout and "codex" in res.stdout


def test_cli_does_not_compute_usage_itself() -> None:
    """AST 守卫：`bladex usage` 只经 `_admin_call` 取 `/admin/usage`，不碰 `usage` / Hub。"""
    from bladex_proxy.cli import ops_cmds

    tree = ast.parse(textwrap.dedent(inspect.getsource(ops_cmds.usage_cmd)))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert "_admin_call" in names
    forbidden = {"aggregate", "iter_hub_turns", "usage_rows", "usage_row", "MemoryHub"}
    assert not (forbidden & (names | attrs)), forbidden & (names | attrs)
    assert not [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    consts = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)}
    assert "/admin/usage" in consts


def test_cli_params_are_endpoint_params() -> None:
    from bladex_proxy.admin_read import _USAGE_PARAMS
    from bladex_proxy.cli import ops_cmds

    src = inspect.getsource(ops_cmds.usage_cmd)
    for p in ("days", "agent", "model", "session"):
        assert p in _USAGE_PARAMS and f'"{p}": {p}' in src


# ── 事件循环 ─────────────────────────────────────────────────────────────────

SLOW_S = 2.0


async def _scenario() -> dict:
    app = create_app(_config())
    out: dict = {}
    async with app.router.lifespan_context(app):
        hub = app.state.hub

        def _slow_rows() -> list:
            time.sleep(SLOW_S)  # 模拟首次请求建缓存的那一趟全扫
            return []

        hub.usage_rows = _slow_rows
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t",
                                     timeout=30) as c:
            t_launch = time.monotonic()
            slow = asyncio.create_task(c.get("/admin/usage", params={"days": 30}))
            await asyncio.sleep(0.3)
            lat = []
            for _ in range(5):
                r = await c.get("/ready")
                lat.append((r.status_code, time.monotonic() - t_launch - 0.3))
            out["ready"] = lat
            r = await slow
            out["slow"] = (r.status_code, time.monotonic() - t_launch)
    return out


def test_usage_endpoint_does_not_block_event_loop() -> None:
    out = asyncio.run(_scenario())
    assert out["slow"][0] == 200 and out["slow"][1] >= SLOW_S - 0.1, out
    assert all(code in (200, 503) for code, _ in out["ready"]), out
    assert max(s for _, s in out["ready"]) < 1.0, out
