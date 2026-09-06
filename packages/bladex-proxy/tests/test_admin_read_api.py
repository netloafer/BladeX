"""Beta T9 只读 admin API 验收剧本。

覆盖：/admin/status /admin/facts /admin/matters(+detail) /admin/hard_rules
/admin/sessions /admin/turns/{key}；鉴权；Memory Index/Memory Hub 缺失降级；分页与过滤。

Memory Index 用测试自建可写句柄（embedder=None → 语义搜索退化为空，列表路径不受影响），
不依赖 lifespan 里的只读句柄是否 init 成功（hermetic）。
"""

from __future__ import annotations

import asyncio
import tempfile
import time

from bladex_core.fact import Fact
from bladex_core.matter import (
    EdgeProvenance,
    EdgeTargetType,
    Matter,
    MatterEdge,
)
from bladex_proxy.config import ProxyConfig
from bladex_proxy.metrics import Metrics
from bladex_proxy.model_health import ModelHealth
from bladex_proxy.models import Identity, Turn
from bladex_proxy.server import create_app
from bladex_proxy.storage.memory_index import MemoryIndex
from fastapi.testclient import TestClient


def _make_config(auth: bool = False) -> ProxyConfig:
    tmpdir = tempfile.mkdtemp()
    return ProxyConfig(
        auth_enabled=auth,
        client_keys_raw="bladex-valid||test" if auth else "",
        hard_rules=["MUST be polite", "NEVER leak secrets"],
        upstream_model="openai/test",
        upstream_api_key="sk-fake",
        rocksdb_path=f"{tmpdir}/rocksdb",
        index_path=f"{tmpdir}/index",
    )


def _make_index() -> MemoryIndex:
    tmpdir = tempfile.mkdtemp()
    index = MemoryIndex(f"{tmpdir}/p2w", embedder=None, read_only=False)
    index.open()
    return index


def _seed_facts(index: MemoryIndex) -> None:
    index.add_fact(Fact(id="f1", content="用户偏好中文回复", kind="preference",
                     source_user_id="u1", tags="origin:conclusion"))
    index.add_fact(Fact(id="f2", content="proxy 502 已修复", kind="event",
                     source_user_id="u1"))
    index.add_fact(Fact(id="f3", content="别人的事实", kind="general",
                     source_user_id="u2"))


# ── /admin/status ──────────────────────────────────────────────────


def test_status_returns_layers_and_routing():
    app = create_app(_make_config())
    with TestClient(app) as client:
        app.state.metrics = Metrics()
        app.state.metrics.inc("bladex_route_source_total", labels={"source": "agent"})
        app.state.metrics.inc("bladex_route_source_total", labels={"source": "agent"})
        health = ModelHealth(cooldown_s=60.0)
        health.mark_unhealthy("openai/broken")
        app.state.model_health = health

        resp = client.get("/admin/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "pipeline" in data and "hub" in data and "index" in data
        assert data["hub"]["available"] is True
        assert isinstance(data["hub"]["turns"], int)
        assert data["routing"]["source_counts"].get("agent") == 2.0
        assert data["upstream"]["unhealthy_models"] == ["openai/broken"]
        assert "version" in data


def test_status_index_lag_turns():
    """lag_turns = Memory Hub turn 数 - Memory Index 已消费数（consolidator 追新滞后）。"""
    app = create_app(_make_config())
    with TestClient(app) as client:
        index = _make_index()
        app.state.index = index
        try:
            # Memory Hub 写 2 轮、Memory Index 消费 0 → lag = 2
            ledger = app.state.hub
            for i in range(2):
                ledger.put(f"u1/agentA/s1/{i:04d}",
                       Turn(identity=Identity(user_id="u1", agent_id="agentA",
                                              session_id="s1")))
            resp = client.get("/admin/status")
            data = resp.json()
            assert data["index"]["available"] is True
            assert data["index"]["lag_turns"] == 2
            assert data["index"]["facts"] == 0
        finally:
            index.close()


# ── 鉴权 ──────────────────────────────────────────────────────────


def test_admin_read_requires_key_when_auth_enabled():
    app = create_app(_make_config(auth=True))
    with TestClient(app) as client:
        assert client.get("/admin/status").status_code == 401
        assert client.get("/admin/facts").status_code == 401
        ok = client.get("/admin/status",
                        headers={"Authorization": "Bearer bladex-valid"})
        assert ok.status_code == 200


# ── /admin/hard_rules ─────────────────────────────────────────────


def test_hard_rules_listing():
    app = create_app(_make_config())
    with TestClient(app) as client:
        resp = client.get("/admin/hard_rules")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        assert "MUST be polite" in data["hard_rules"]


# ── /admin/facts ──────────────────────────────────────────────────


def test_facts_list_filters_and_pagination():
    app = create_app(_make_config())
    with TestClient(app) as client:
        index = _make_index()
        _seed_facts(index)
        app.state.index = index
        try:
            # 全量
            data = client.get("/admin/facts").json()
            assert data["total"] == 3
            assert data["mode"] == "list"
            assert all("embedding" not in f for f in data["facts"])
            # user_id 过滤
            data = client.get("/admin/facts", params={"user_id": "u1"}).json()
            assert data["total"] == 2
            # origin 过滤（tags 含 origin:conclusion）
            data = client.get("/admin/facts", params={"origin": "conclusion"}).json()
            assert data["total"] == 1
            assert data["facts"][0]["id"] == "f1"
            # kind 过滤
            data = client.get("/admin/facts", params={"kind": "event"}).json()
            assert data["total"] == 1
            # 分页
            data = client.get("/admin/facts", params={"limit": 2}).json()
            assert len(data["facts"]) == 2 and data["total"] == 3
            data = client.get("/admin/facts", params={"limit": 2, "offset": 2}).json()
            assert len(data["facts"]) == 1
        finally:
            index.close()


def test_facts_503_when_index_missing():
    app = create_app(_make_config())
    with TestClient(app) as client:
        app.state.index = None
        assert client.get("/admin/facts").status_code == 503
        assert client.get("/admin/matters").status_code == 503


def test_facts_search_mode_without_embedder_returns_empty():
    """q 非空但 embedder 缺失 → 空结果（退化，不 500）。"""
    app = create_app(_make_config())
    with TestClient(app) as client:
        index = _make_index()
        _seed_facts(index)
        app.state.index = index
        try:
            data = client.get("/admin/facts", params={"q": "proxy 502"}).json()
            assert data["mode"] == "search"
            assert data["facts"] == []
        finally:
            index.close()


# ── /admin/matters ────────────────────────────────────────────────


def test_matters_list_and_detail():
    app = create_app(_make_config())
    with TestClient(app) as client:
        index = _make_index()
        _seed_facts(index)
        index.add_matter(Matter(matter_id="m1", title="Beta 发布", summary="发布准备"))
        index.add_edge(MatterEdge(edge_id="e1", matter_id="m1",
                               target_type=EdgeTargetType.FACT, target_key="f1",
                               provenance=EdgeProvenance.MANUAL))
        index.add_edge(MatterEdge(edge_id="e2", matter_id="m1",
                               target_type=EdgeTargetType.SESSION,
                               target_key="u1/agentA/s1/"))
        app.state.index = index
        try:
            data = client.get("/admin/matters").json()
            assert data["total"] == 1
            m = data["matters"][0]
            assert m["matter_id"] == "m1"
            assert "centroid" not in m and "embedding" not in m

            detail = client.get("/admin/matters/m1").json()
            assert detail["matter"]["title"] == "Beta 发布"
            assert detail["edge_count"] == 2
            assert detail["manual_edge_count"] == 1
            assert detail["session_keys"] == ["u1/agentA/s1/"]

            assert client.get("/admin/matters/nope").status_code == 404
        finally:
            index.close()


# ── /admin/sessions + /admin/turns ────────────────────────────────


def test_sessions_aggregation_and_turn_detail():
    app = create_app(_make_config())
    with TestClient(app) as client:
        ledger = app.state.hub
        for i in range(3):
            ledger.put(f"u1/agentA/s1/{i:04d}",
                   Turn(identity=Identity(user_id="u1", agent_id="agentA",
                                          session_id="s1"),
                        model="openai/test",
                        request_messages=[{"role": "user", "content": f"hi {i}"}],
                        response_text=f"hello {i}",
                        injected_memory="<bladex-memory>rules</bladex-memory>"))
        ledger.put("u1/agentB/s9/0000",
               Turn(identity=Identity(user_id="u1", agent_id="agentB",
                                      session_id="s9")))

        data = client.get("/admin/sessions").json()
        assert data["total"] == 2
        by_sess = {s["session"]: s for s in data["sessions"]}
        assert by_sess["u1/agentA/s1/"]["turns"] == 3
        assert by_sess["u1/agentA/s1/"]["agent_id"] == "agentA"

        # agent 过滤
        data = client.get("/admin/sessions", params={"agent": "agentB"}).json()
        assert data["total"] == 1

        # turn 详情：默认不带 messages
        turn = client.get("/admin/turns/u1/agentA/s1/0001").json()
        assert turn["key"] == "u1/agentA/s1/0001"
        assert "request_messages" not in turn
        assert "response_text" not in turn
        assert turn["injected_memory"].startswith("<bladex-memory>")
        assert "decision_meta" in turn

        # include_messages=true 带上
        turn = client.get("/admin/turns/u1/agentA/s1/0001",
                          params={"include_messages": "true"}).json()
        assert turn["request_messages"][0]["content"] == "hi 1"
        assert turn["response_text"] == "hello 1"

        assert client.get("/admin/turns/u1/agentA/s1/9999").status_code == 404


# ── Metrics.snapshot / ModelHealth.unhealthy_models 单元 ──────────


def test_metrics_snapshot_roundtrip():
    m = Metrics()
    m.inc("bladex_requests_total", labels={"agent": "hermes", "sensitivity": "normal"})
    m.inc("bladex_requests_total", labels={"agent": "hermes", "sensitivity": "normal"})
    m.observe("bladex_request_latency_ms", 10.0, labels={"agent": "hermes"})
    m.observe("bladex_request_latency_ms", 20.0, labels={"agent": "hermes"})
    m.set_gauge("bladex_pipeline_queue_depth", 5.0)
    snap = m.snapshot()
    counters = snap["counters"]["bladex_requests_total"]
    assert counters[0][0] == {"agent": "hermes", "sensitivity": "normal"}
    assert counters[0][1] == 2.0
    hist = snap["histograms"]["bladex_request_latency_ms"][0][1]
    assert hist["count"] == 2
    assert snap["gauges"]["bladex_pipeline_queue_depth"][0][1] == 5.0


# ── /admin/queue（2026-08-06 队列卡死事故）──────────────────────────


class _StubPipeline:
    """只提供 inspect 的桩——端点层要验的是接线与降级，不是 Redis 语义。"""

    overflow_count = 0
    spill_count = 0

    def __init__(self, detail: dict | None = None, fail: bool = False) -> None:
        self._detail = detail or {"total": 487, "pending": 0, "undelivered": 2,
                                  "orphan": 485, "last_delivered_id": "9-0"}
        self._fail = fail

    async def inspect(self) -> dict:
        if self._fail:
            raise RuntimeError("redis gone")
        return dict(self._detail)

    async def close(self) -> None:  # lifespan 关停会调
        pass


class _StubWorker:
    def __init__(self) -> None:
        self.calls: list[tuple[bool, int]] = []

    async def stop(self) -> None:  # lifespan 关停会调
        pass

    async def force_drain(self, *, apply: bool = False, max_batches: int = 200,
                          sample: int = 0) -> dict:
        self.calls.append((apply, max_batches))
        self.sample = sample
        return {"mode": "applied" if apply else "dry-run", "delivered": 0,
                "reclaimed": 0, "failed": 0, "orphan_total": 485,
                "orphan_in_hub": 485, "orphan_missing": 0,
                "orphan_recovered": 0, "orphan_deleted": 485 if apply else 0,
                "truncated": False, "before": {"total": 487}, "after": {"total": 2}}


def test_queue_endpoint_reports_the_split():
    app = create_app(_make_config())
    with TestClient(app) as client:
        app.state.pipeline = _StubPipeline()
        data = client.get("/admin/queue").json()
        assert data["total"] == 487
        assert data["orphan"] == 485
        assert data["undelivered"] == 2


def test_queue_endpoint_503_when_pipeline_is_down():
    """Redis 掉线时不许假装队列是空的——那正是轮次在往磁盘落的时候。"""
    app = create_app(_make_config())
    with TestClient(app) as client:
        app.state.pipeline = None
        assert client.get("/admin/queue").status_code == 503
        app.state.pipeline = _StubPipeline(fail=True)
        assert client.get("/admin/queue").status_code == 503


def test_queue_flush_defaults_to_dry_run():
    """默认必须是预览：孤儿清理是不可逆的 XDEL。"""
    app = create_app(_make_config())
    with TestClient(app) as client:
        worker = _StubWorker()
        app.state.pipeline_worker = worker

        data = client.post("/admin/queue/flush").json()
        assert data["status"] == "dry-run"
        assert worker.calls[-1][0] is False

        data = client.post("/admin/queue/flush?apply=true&max_batches=5").json()
        assert data["status"] == "applied"
        assert worker.calls[-1] == (True, 5)

        # 取样是逐条扫 Ledger 的，必须有上限，不能由调用方随便开
        client.post("/admin/queue/flush?sample=999")
        assert worker.sample == 20


def test_heartbeat_marks_stale_beats_as_not_fresh():
    """心跳的年龄必须算出来——只看键在不在，会把一个卡住的进程报成健康。"""
    import time as _t

    from bladex_proxy.admin_read import _read_consolidator_heartbeat

    class _Pipe:
        def __init__(self, ts: float) -> None:
            self._ts = ts

        @property
        def redis(self):  # noqa: ANN202
            outer = self

            class _R:
                async def hgetall(self, _key):  # noqa: ANN202
                    return {b"state": b"running", b"phase": b"distill",
                            b"done": b"71", b"total": b"305", b"batch": b"3",
                            b"ts": str(outer._ts).encode()}
            return _R()

    fresh = asyncio.run(_read_consolidator_heartbeat(_Pipe(time.time())))
    assert fresh["fresh"] is True
    assert fresh["done"] == 71 and fresh["total"] == 305 and fresh["batch"] == 3

    stale = asyncio.run(_read_consolidator_heartbeat(_Pipe(_t.time() - 3600)))
    assert stale["fresh"] is False
    assert stale["last_seen_s"] > 3000


def test_heartbeat_absent_is_none_not_an_error():
    """没心跳要能和"心跳说它停了"区分开，所以返回 None 而不是空 dict。"""
    from bladex_proxy.admin_read import _read_consolidator_heartbeat

    class _Empty:
        @property
        def redis(self):  # noqa: ANN202
            class _R:
                async def hgetall(self, _key):  # noqa: ANN202
                    return {}
            return _R()

    assert asyncio.run(_read_consolidator_heartbeat(_Empty())) is None
    assert asyncio.run(_read_consolidator_heartbeat(None)) is None


def test_queue_flush_503_without_worker():
    app = create_app(_make_config())
    with TestClient(app) as client:
        app.state.pipeline_worker = None
        assert client.post("/admin/queue/flush").status_code == 503


def test_unhealthy_models_lazy_expiry():
    h = ModelHealth(cooldown_s=0.0)
    h.mark_unhealthy("m1")
    # cooldown 0 → 立即过期，惰性清除
    assert h.unhealthy_models() == []
    h2 = ModelHealth(cooldown_s=60.0)
    h2.mark_unhealthy("m2")
    assert h2.unhealthy_models() == ["m2"]
