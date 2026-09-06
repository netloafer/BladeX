#!/usr/bin/env python3
"""ADR-0021 T7 多 principal 并发压测（规模参数首次实测）。

启动 in-process proxy（上游转发 mock 为即时响应，隔离上游延迟，测 proxy 热路径本身），
N 个 principal 各发 C 并发请求，统计吞吐 + 延迟分位。

用法：
  .venv/bin/python scripts/selfhost_load_test.py
  .venv/bin/python scripts/selfhost_load_test.py --principals 10 --concurrency 5 --requests 200

注：本测的是 proxy 热路径（认身份 + 注入 + 路由 + 捕获 + 存储）开销，不含上游 LLM 延迟。
真实上游延迟需对接部署实例另测。结果进 benchmark 记录。
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import sys
import tempfile
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in ("packages/bladex-core", "packages/bladex-proxy"):
    _full = os.path.join(_ROOT, _p)
    if _full not in sys.path:
        sys.path.insert(0, _full)


async def _run(args) -> dict:
    from unittest.mock import MagicMock, patch

    from bladex_proxy.config import ProxyConfig
    from bladex_proxy.server import create_app
    from httpx import ASGITransport, AsyncClient

    tmpdir = tempfile.mkdtemp()
    cfg = ProxyConfig(
        upstream_model="openai/test-mock",
        upstream_api_key="sk-fake",
        rocksdb_path=f"{tmpdir}/rocksdb",
        index_path=f"{tmpdir}/index",
        overflow_dir=f"{tmpdir}/overflow",
        fastembed_cache_path=f"{tmpdir}/fe",
        auth_enabled=False,
        route_enabled=False,
    )

    fake_resp = MagicMock()
    fake_resp.choices = [MagicMock()]
    fake_resp.choices[0].message.content = "ok"
    fake_resp.model_dump.return_value = {
        "id": "x", "choices": [{"message": {"role": "assistant", "content": "ok"}}],
    }

    async def _fake_acompletion(model, messages, stream, **kwargs):
        return fake_resp

    app = create_app(cfg)

    latencies: list[float] = []
    total = args.principals * args.requests

    async def _one(client, principal_id, req_count):
        for _ in range(req_count):
            t0 = time.perf_counter()
            r = await client.post(
                "/v1/chat/completions",
                json={"model": "m", "stream": False,
                      "messages": [{"role": "user", "content": f"principal {principal_id} query"}]},
                headers={"X-Agent-ID": f"agent-{principal_id % 3}"},
            )
            latencies.append((time.perf_counter() - t0) * 1000)
            assert r.status_code == 200, r.text

    with patch("bladex_proxy.route.router_sdk") as mock_router:
        mock_router.acompletion = _fake_acompletion
        mock_router.retryable_error_types.return_value = ()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # lifespan 启动
            async with app.router.lifespan_context(app):
                t0 = time.perf_counter()
                tasks = []
                for pid in range(args.principals):
                    for _ in range(args.concurrency):
                        tasks.append(_one(client, pid, args.requests))
                await asyncio.gather(*tasks)
                elapsed = time.perf_counter() - t0

    latencies.sort()

    def pct(p):
        idx = max(0, min(len(latencies) - 1, int(len(latencies) * p) - 1))
        return latencies[idx]

    return {
        "principals": args.principals,
        "concurrency": args.concurrency,
        "requests": total,
        "elapsed_s": round(elapsed, 3),
        "throughput_rps": round(total / elapsed, 1),
        "latency_mean_ms": round(statistics.mean(latencies), 2),
        "latency_p50_ms": round(pct(0.5), 2),
        "latency_p95_ms": round(pct(0.95), 2),
        "latency_p99_ms": round(pct(0.99), 2),
        "latency_max_ms": round(max(latencies), 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="BladeX selfhost multi-principal load test")
    ap.add_argument("--principals", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--requests", type=int, default=20)
    args = ap.parse_args()

    print("=== ADR-0021 T7 多 principal 并发压测（in-process proxy, mock 上游）===")
    print(f"principals={args.principals} concurrency_per_principal={args.concurrency} "
          f"requests_per_principal={args.requests}")
    res = asyncio.run(_run(args))
    print("--- 结果 ---")
    for k, v in res.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
