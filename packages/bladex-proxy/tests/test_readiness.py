"""T27 服务状态面拆分 liveness + readiness 验收剧本（B11.3）。

/health 保持免鉴权轻量 liveness；/ready 免鉴权探子系统就绪。
ready = redis ∧ ledger（写路径必需）；index/upstream 为 informational 不 gate。
只暴露布尔就绪位、不泄运行数据。
"""

from __future__ import annotations

import tempfile
from unittest.mock import MagicMock

from bladex_proxy.config import ProxyConfig
from bladex_proxy.model_health import ModelHealth
from bladex_proxy.server import create_app
from fastapi.testclient import TestClient


class _FakePipeline:
    """fake Pipeline：redis.ping() 成功（hermetic，不依赖真 Redis）。"""

    class _Redis:
        async def ping(self) -> bool:
            return True

    @property
    def redis(self) -> _FakePipeline._Redis:
        return self._Redis()

    overflow_count = 0
    spill_count = 0

    async def close(self) -> None:
        pass


def _make_config(auth: bool = False) -> ProxyConfig:
    tmpdir = tempfile.mkdtemp()
    return ProxyConfig(
        auth_enabled=auth,
        client_keys_raw="bladex-valid||test" if auth else "",
        hard_rules=["MUST be polite"],
        upstream_model="openai/test",
        upstream_api_key="sk-fake",
        rocksdb_path=f"{tmpdir}/rocksdb",
    )


def test_health_still_liveness():
    """/health 保持 liveness：进程活即 200。"""
    app = create_app(_make_config())
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


def test_ready_all_ready_200():
    """redis + ledger 就绪 -> 200 + 布尔映射。"""
    app = create_app(_make_config())
    with TestClient(app) as client:
        orig_pipeline = app.state.pipeline
        app.state.pipeline = _FakePipeline()  # hermetic：不依赖真 Redis
        try:
            resp = client.get("/ready")
            assert resp.status_code == 200
            data = resp.json()
            assert data["ready"] is True
            assert data["checks"]["redis"] is True
            assert data["checks"]["hub"] is True
        finally:
            app.state.pipeline = orig_pipeline


def test_ready_redis_missing_503():
    """redis 缺失（pipeline=None）-> 503。"""
    app = create_app(_make_config())
    with TestClient(app) as client:
        orig_pipeline = app.state.pipeline
        app.state.pipeline = None
        try:
            resp = client.get("/ready")
            assert resp.status_code == 503
            data = resp.json()
            assert data["ready"] is False
            assert data["checks"]["redis"] is False
            assert data["checks"]["hub"] is True  # ledger 仍在，但 ready 被 redis 拖低
        finally:
            app.state.pipeline = orig_pipeline


def test_ready_ledger_missing_503():
    """ledger 缺失 -> 503（即使 redis 就绪）。"""
    app = create_app(_make_config())
    with TestClient(app) as client:
        orig_pipeline, orig_ledger = app.state.pipeline, app.state.hub
        app.state.pipeline = _FakePipeline()
        app.state.hub = None
        try:
            resp = client.get("/ready")
            assert resp.status_code == 503
            assert resp.json()["checks"]["hub"] is False
        finally:
            app.state.pipeline, app.state.hub = orig_pipeline, orig_ledger


def test_ready_no_auth_accessible():
    """auth 开启时 /ready 仍免鉴权可访问（同 /health）。"""
    app = create_app(_make_config(auth=True))
    with TestClient(app) as client:
        orig_pipeline = app.state.pipeline
        app.state.pipeline = _FakePipeline()
        try:
            resp = client.get("/ready")  # 无 key
            assert resp.status_code == 200
        finally:
            app.state.pipeline = orig_pipeline


def test_ready_no_operational_data_leaked():
    """只暴露布尔就绪位、不含运行数据字段（计数/水位/延迟）。"""
    app = create_app(_make_config())
    with TestClient(app) as client:
        orig_pipeline = app.state.pipeline
        app.state.pipeline = _FakePipeline()
        try:
            resp = client.get("/ready")
            data = resp.json()
            # 顶层仅 ready + checks
            assert set(data.keys()) == {"ready", "checks"}
            # checks 仅四个布尔位
            assert set(data["checks"].keys()) == {"redis", "hub", "index", "upstream"}
            assert all(isinstance(v, bool) for v in data["checks"].values())
        finally:
            app.state.pipeline = orig_pipeline


def test_ready_index_informational_not_gating():
    """index=None（Memory Index 未初始化）时 ready 仍可 200（index 不 gate，fresh install 不 503）。"""
    app = create_app(_make_config())
    with TestClient(app) as client:
        orig_pipeline, orig_index = app.state.pipeline, app.state.index
        app.state.pipeline = _FakePipeline()
        app.state.index = None
        try:
            resp = client.get("/ready")
            assert resp.status_code == 200  # index 不 gate
            assert resp.json()["checks"]["index"] is False
        finally:
            app.state.pipeline, app.state.index = orig_pipeline, orig_index


def test_ready_upstream_unhealthy_informational():
    """单模型熔断 -> upstream=false（informational），但 ready 仍 200（不 gate）。"""
    app = create_app(_make_config())
    with TestClient(app) as client:
        orig_pipeline = app.state.pipeline
        app.state.pipeline = _FakePipeline()
        # router 开 + 标一个模型 unhealthy
        app.state.router = MagicMock()  # truthy -> 进 upstream 判定
        mh = ModelHealth(cooldown_s=30.0)
        mh.mark_unhealthy("openai/test")
        app.state.model_health = mh
        try:
            resp = client.get("/ready")
            assert resp.status_code == 200  # upstream 不 gate
            assert resp.json()["checks"]["upstream"] is False
        finally:
            app.state.pipeline = orig_pipeline
