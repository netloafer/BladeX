"""批 DS1b（MQ-W25）：三条全扫写路由执行期间，事件循环不被占住。

`DELETE /admin/ledgers/{id}`（真跑与 dry-run）、`DELETE /admin/ledgers/{id}/memory`、
`POST /admin/agents/claim`：把各自的全扫函数换成 `time.sleep(3)` 的桩，慢请求跑着的时候发
`/ready` 与一个 `/v1` 桩请求，两者都必须在 1 s 内回来（计时口径同 DS1a 用例：从发出慢请求 + 0.3 s 起算）。
修前（路由体里直接调）事件循环被 sleep 占住，两者要等满 3 s。
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

import httpx
import pytest
from bladex_core.ledger import new_ledger
from bladex_proxy.agency import AgencyRuntime
from bladex_proxy.server import create_app
from test_ds1a_event_loop import _V1, _config, _fake_upstream, _timed

SLOW_S = 3.0
_LID = "ldg-ds1b-slow01"


def _slow_window(hub, ledger_id, *, now_ms=None):  # noqa: ANN001
    time.sleep(SLOW_S)
    return {"turn_keys": [], "scopes": [], "known": True, "title": "t"}


def _slow_claim_map(hub):  # noqa: ANN001
    time.sleep(SLOW_S)
    return {}


_CASES = {
    "delete_ledger": ("DELETE", f"/admin/ledgers/{_LID}", None,
                      "bladex_proxy.ledger_memory.ledger_window_turn_keys", _slow_window),
    "delete_ledger_dry_run": ("DELETE", f"/admin/ledgers/{_LID}?dry_run=true", None,
                              "bladex_proxy.ledger_memory.ledger_window_turn_keys", _slow_window),
    "delete_ledger_memory": ("DELETE", f"/admin/ledgers/{_LID}/memory", None,
                             "bladex_proxy.ledger_memory.ledger_window_turn_keys", _slow_window),
    "claim_agent": ("POST", "/admin/agents/claim",
                    {"from_agent_ids": ["unknown-ds1b0001"], "to_agent_id": "ds1b-new"},
                    "bladex_proxy.server.admin_api.build_agent_claim_map", _slow_claim_map),
}


async def _scenario(method: str, url: str, body: dict | None) -> dict:
    app = create_app(_config())
    out: dict = {}
    async with app.router.lifespan_context(app):
        if app.state.agency is None:  # 账本模块默认装配；没装上就补一个真的 runtime
            app.state.agency = AgencyRuntime(hub=app.state.hub)
        app.state.agency.pool[_LID] = new_ledger(
            ledger_id=_LID, title="t", goal="g", goal_source="user",
            created_at="2026-10-25T10:00:00Z")
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t",
                                     timeout=30) as client:
            await client.request("POST", "/v1/chat/completions", **_V1)  # 预热
            t_launch = time.monotonic()
            slow_task = asyncio.create_task(_timed(client, method, url,
                                                   **({"json": body} if body else {})))
            await asyncio.sleep(0.3)
            code, _ = await _timed(client, "GET", "/ready")
            out["ready"] = (code, time.monotonic() - t_launch - 0.3)
            code, _ = await _timed(client, "POST", "/v1/chat/completions", **_V1)
            out["v1"] = (code, time.monotonic() - t_launch - 0.3)
            out["slow"] = await slow_task
    return out


@pytest.mark.parametrize("case", sorted(_CASES))
def test_full_scan_write_route_does_not_block_ready_or_v1(case) -> None:
    method, url, body, target, stub = _CASES[case]
    with patch("bladex_proxy.route.router_sdk.acompletion", new=_fake_upstream()), \
            patch(target, new=stub):
        out = asyncio.run(_scenario(method, url, body))
    ready_code, ready_s = out["ready"]
    v1_code, v1_s = out["v1"]
    slow_code, slow_s = out["slow"]
    assert slow_code == 200 and slow_s >= SLOW_S - 0.1, out
    assert ready_code in (200, 503) and ready_s < 1.0, out
    assert v1_code == 200 and v1_s < 1.0, out
