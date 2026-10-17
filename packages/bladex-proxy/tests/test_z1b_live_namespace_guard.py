"""批 Z1b · Z1b.1 —— 测试进程里不许用 live 的 Redis 命名空间（MQ-A77 的结构性守卫）。

背景：`test_cli_ledger.py::_isolate_env` 删光 `BLADEX_*`（连 conftest 的 Redis 隔离值）⇒
`create_app(ProxyConfig())` 回落 live 的 `bladex:turns` 并起 PipelineWorker，约 25 天没人发现
（批 Z1 §3.6）。MQ-A38 的实例检测只在 live 正在跑时拦得住；本闸不看对方在不在。
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
from bladex_proxy.config import DEFAULT_REDIS_STREAM, ProxyConfig
from bladex_proxy.storage import pipeline_redis as pr

_PKG = Path(__file__).resolve().parents[1] / "bladex_proxy"


def test_guard_is_armed_in_the_test_process():
    """闸由根 conftest 打开；导入失败会静默不开——这条让"没开"变成红。"""
    assert pr.TEST_PROCESS_GUARD is True


def test_connect_to_live_default_stream_raises_and_names_the_test():
    p = pr.PipelineRedis(redis_url="redis://127.0.0.1:1/0", stream=DEFAULT_REDIS_STREAM,
                         group="g", consumer="c")
    with pytest.raises(pr.LiveNamespaceInTestError) as ei:
        asyncio.run(p.connect())
    msg = str(ei.value)
    assert "test_connect_to_live_default_stream_raises_and_names_the_test" in msg
    assert repr(DEFAULT_REDIS_STREAM) in msg and "MQ-A77" in msg


def test_negative_control_isolated_stream_passes_the_guard():
    """conftest 的隔离值照常可用（闸不误伤）。"""
    cfg = ProxyConfig()
    assert cfg.redis_stream != DEFAULT_REDIS_STREAM
    pr.refuse_live_namespace_in_tests(cfg.redis_stream, cfg.redis_group, cfg.redis_consumer)


def test_production_process_is_unaffected(monkeypatch):
    """闸未开（= 生产进程）⇒ 连 live 默认 stream 不抛：生产路径零行为变化。"""
    monkeypatch.setattr(pr, "TEST_PROCESS_GUARD", False)
    pr.refuse_live_namespace_in_tests(DEFAULT_REDIS_STREAM, "bladex-workers", "worker-1")


def test_defect_shape_env_scrub_then_create_app_is_red(tmp_path, monkeypatch):
    """复现原缺陷形状：删光 BLADEX_* 后建 app 跑 lifespan ⇒ 闸抛错、不被"Redis 连不上就降级"吞掉。"""
    from bladex_proxy.server import create_app
    from fastapi.testclient import TestClient
    monkeypatch.chdir(tmp_path)
    for k in list(os.environ):
        if k.startswith("BLADEX_"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("BLADEX_ROCKSDB_PATH", str(tmp_path / "hub"))
    app = create_app(ProxyConfig())
    assert app.state.config.redis_stream == DEFAULT_REDIS_STREAM
    with pytest.raises(pr.LiveNamespaceInTestError), TestClient(app):
        pytest.fail("lifespan 不该走到 yield")


def test_both_connect_call_sites_let_the_guard_error_through():
    """lifespan 与懒恢复都把 connect 包在 except Exception 里——两处都要先放行本闸的错误。"""
    for rel, head in (("server/app_factory.py", "async def lifespan("),
                      ("server/orchestration.py", "async def _try_recover_redis(")):
        body = (_PKG / rel).read_text(encoding="utf-8").split(head, 1)[1]
        first = body.index("await pipeline.connect()")
        seg = body[first:first + 400]
        assert seg.index("except LiveNamespaceInTestError:") < seg.index("except Exception"), rel


def test_guard_defaults_come_from_single_source():
    """默认值只在 config.DEFAULT_REDIS_* 定义一次，闸与 ProxyConfig 都读它（不抄字面量）。"""
    src = (_PKG / "storage" / "pipeline_redis.py").read_text(encoding="utf-8")
    assert '"bladex:turns"' not in src
    cfg_src = (_PKG / "config.py").read_text(encoding="utf-8")
    assert cfg_src.count('"bladex:turns"') == 1
