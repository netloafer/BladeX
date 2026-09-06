"""ADR-0021 T7 metrics 验收：Metrics 采集器 + /metrics 端点。"""

import tempfile
from unittest.mock import MagicMock, patch

from bladex_proxy.config import ProxyConfig
from bladex_proxy.metrics import Metrics
from bladex_proxy.server import create_app
from fastapi.testclient import TestClient


def _make_config() -> ProxyConfig:
    tmpdir = tempfile.mkdtemp()
    return ProxyConfig(
        hard_rules=["MUST be polite"],
        upstream_model="openai/test",
        upstream_api_key="sk-fake",
        rocksdb_path=f"{tmpdir}/rocksdb",
    )


# ── Metrics 采集器单元 ──


def test_metrics_counter_and_render():
    m = Metrics()
    m.inc("bladex_requests_total", labels={"agent": "hermes", "sensitivity": "normal"})
    m.inc("bladex_requests_total", labels={"agent": "hermes", "sensitivity": "normal"})
    m.inc("bladex_requests_total", labels={"agent": "codex", "sensitivity": "sensitive"})
    out = m.render()
    assert "# TYPE bladex_requests_total counter" in out
    assert 'bladex_requests_total{agent="hermes",sensitivity="normal"} 2' in out
    assert 'bladex_requests_total{agent="codex",sensitivity="sensitive"} 1' in out


def test_metrics_histogram_quantiles():
    m = Metrics()
    for v in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        m.observe("bladex_request_latency_ms", float(v), labels={"agent": "a"})
    out = m.render()
    assert "# TYPE bladex_request_latency_ms histogram" in out
    assert 'quantile="0.5"' in out
    assert 'quantile="0.95"' in out
    assert 'quantile="0.99"' in out
    assert "bladex_request_latency_ms_count" in out
    # p50 应在中间附近
    assert "50" in out


def test_metrics_gauge():
    m = Metrics()
    m.set_gauge("bladex_pipeline_queue_depth", 42.0)
    out = m.render()
    assert "# TYPE bladex_pipeline_queue_depth gauge" in out
    assert "bladex_pipeline_queue_depth 42" in out


# ── /metrics 端点可 curl 抓取 + 维度齐全 ──


def test_metrics_endpoint_curlable_with_dimensions():
    """发一条请求后 /metrics 返回 Prometheus 文本，含请求计数/路由 source/注入指标维度。"""
    app = create_app(_make_config())
    fake_response = MagicMock()
    fake_response.choices = [MagicMock()]
    fake_response.choices[0].message.content = "hi"
    fake_response.model_dump.return_value = {
        "id": "x", "choices": [{"message": {"role": "assistant", "content": "hi"}}],
    }

    async def fake_acompletion(model, messages, stream, **kwargs):
        return fake_response

    with patch("bladex_proxy.route.router_sdk.acompletion", new=fake_acompletion):
        with TestClient(app) as client:
            client.post(
                "/v1/chat/completions",
                json={"model": "gpt-4o",
                      "messages": [{"role": "user", "content": "Say hello there"}],
                      "stream": False},
                headers={"Authorization": "Bearer sk-test", "X-Agent-ID": "test-agent"},
            )
            resp = client.get("/metrics")

    assert resp.status_code == 200
    text = resp.text
    # 维度齐全
    assert "bladex_requests_total" in text
    assert "bladex_request_latency_ms" in text
    assert "bladex_route_source_total" in text
    assert "bladex_inject_facts_injected" in text
    # 请求被计数（agent 维度）
    assert 'agent="test-agent"' in text
