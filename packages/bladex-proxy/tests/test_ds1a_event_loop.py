"""批 DS1a（MQ-W24）：慢 admin 读路由执行期间，事件循环不被占住。

做法：把 `/admin/matters/merge_candidates` 背后的同步函数换成 `time.sleep(3)` 的桩；用
`httpx.AsyncClient`（同一个事件循环）先发它，在它跑着的时候发 `/ready` 和一个 `/v1` 桩请求，
两者都必须在 1 s 内回来（从发出慢请求 + 0.3 s 起算，见场景里的注释）。修前（路由体里直接调）事件循环被 sleep 占住，两者要等满 3 s。

另钉关停：lifespan 关 Hub / Index 前等在途的线程读跑完（`drain_blocking`），不在半截关句柄。
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from bladex_proxy.config import ProxyConfig
from bladex_proxy.server import create_app

SLOW_S = 3.0


def _config() -> ProxyConfig:
    tmpdir = tempfile.mkdtemp(prefix="ds1a_loop_")
    return ProxyConfig(
        hard_rules=["MUST be polite"],
        upstream_model="openai/test",
        upstream_api_key="sk-fake",
        rocksdb_path=f"{tmpdir}/rocksdb",
        index_path=f"{tmpdir}/index",
    )


class _SlowIndex:
    """只实现 merge_candidates：同步 sleep，模拟 live 33 s 的全扫。"""

    def __init__(self) -> None:
        self.finished_at: float | None = None

    def matter_merge_candidates(self, max_pairs: int = 50) -> list:
        time.sleep(SLOW_S)
        self.finished_at = time.monotonic()
        return []

    def close(self) -> None:
        pass


class _Resp:
    """普通对象作上游响应（MagicMock 会让 usage 捕获无限递归、单次 /v1 慢到 1.5 s）。"""

    def __init__(self) -> None:
        msg = SimpleNamespace(role="assistant", content="hi", tool_calls=None)
        self.choices = [SimpleNamespace(message=msg, finish_reason="stop", index=0)]
        self.usage = None
        self.model = "openai/test"

    def model_dump(self, **_kw) -> dict:  # noqa: ANN003
        return {"id": "x", "model": "openai/test",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "hi"}}]}


def _fake_upstream():
    async def fake_acompletion(model, messages, stream, **kwargs):  # noqa: ANN001, ANN003
        return _Resp()

    return fake_acompletion


async def _timed(client: httpx.AsyncClient, method: str, url: str, **kw) -> tuple[int, float]:
    t0 = time.monotonic()
    r = await client.request(method, url, **kw)
    return r.status_code, time.monotonic() - t0


_V1 = {"json": {"model": "gpt-4o", "stream": False,
                "messages": [{"role": "user", "content": "Say hello there"}]},
       "headers": {"Authorization": "Bearer sk-test", "X-Agent-ID": "ds1a-agent"}}


async def _scenario() -> dict:
    app = create_app(_config())
    out: dict = {}
    async with app.router.lifespan_context(app):
        started = app.state.index
        slow = _SlowIndex()
        app.state.index = slow
        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t",
                                         timeout=30) as client:
                await client.request("POST", "/v1/chat/completions", **_V1)  # 预热
                # 计时起点 = 发出慢请求的那一刻。不能从"发 /ready 时"算：事件循环若被占住，
                # 下面这句 sleep(0.3) 本身就会拖到慢读结束，之后再发的 /ready 当然很快。
                t_launch = time.monotonic()
                slow_task = asyncio.create_task(
                    _timed(client, "GET", "/admin/matters/merge_candidates"))
                await asyncio.sleep(0.3)  # 让慢请求先进到路由里
                code, _ = await _timed(client, "GET", "/ready")
                out["ready"] = (code, time.monotonic() - t_launch - 0.3)
                code, _ = await _timed(client, "POST", "/v1/chat/completions", **_V1)
                out["v1"] = (code, time.monotonic() - t_launch - 0.3)
                out["slow"] = await slow_task
        finally:
            app.state.index = started
    return out


def test_slow_admin_read_does_not_block_ready_or_v1() -> None:
    with patch("bladex_proxy.route.router_sdk.acompletion", new=_fake_upstream()):
        out = asyncio.run(_scenario())
    ready_code, ready_s = out["ready"]
    v1_code, v1_s = out["v1"]
    slow_code, slow_s = out["slow"]
    assert slow_code == 200 and slow_s >= SLOW_S - 0.1, out
    assert ready_code in (200, 503) and ready_s < 1.0, out
    assert v1_code == 200 and v1_s < 1.0, out


async def _shutdown_scenario() -> dict:
    app = create_app(_config())
    slow = _SlowIndex()
    closed_at: dict = {}
    async with app.router.lifespan_context(app):
        started = app.state.index
        if started is not None:
            started.close()
        app.state.index = slow
        hub = app.state.hub
        real_close = hub.close

        def _close() -> None:
            closed_at["hub"] = time.monotonic()
            real_close()

        hub.close = _close
        transport = httpx.ASGITransport(app=app)
        client = httpx.AsyncClient(transport=transport, base_url="http://t", timeout=30)
        task = asyncio.create_task(client.get("/admin/matters/merge_candidates"))
        await asyncio.sleep(0.3)
    # 退出 lifespan 时慢读仍在线程里：必须先等它完成，再关 Hub
    r = await task
    await client.aclose()
    return {"status": r.status_code, "finished": slow.finished_at, "hub_closed": closed_at.get("hub")}


def test_lifespan_waits_for_inflight_reads_before_closing_hub() -> None:
    out = asyncio.run(_shutdown_scenario())
    assert out["finished"] is not None and out["hub_closed"] is not None, out
    assert out["finished"] <= out["hub_closed"], out
