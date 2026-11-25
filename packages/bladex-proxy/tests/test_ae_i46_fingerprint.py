"""批 AE · MQ-I46：`backend=proxy` 的 `embedcache/` 键含 proxy 端回报的指纹，不再按 `unversioned` 分桶。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from bladex_proxy.config import ProxyConfig
from bladex_proxy.embedding import ProxyEmbedAdapter
from bladex_proxy.storage.memory_index import MemoryIndex


class _Remote:
    """假的 `/v1/embeddings`：回报什么 identity / 指纹由用例改。"""

    def __init__(self, identity: str = "local:m1", fingerprint: str | None = "fastembed=1") -> None:
        self.identity, self.fingerprint = identity, fingerprint
        self.calls = 0

    def post(self, _url: str, *, json: dict, **_kw: object) -> SimpleNamespace:
        self.calls += 1
        body = {"model": "m", "bladex_model_identity": self.identity,
                "data": [{"index": i, "embedding": [0.5]} for i in range(len(json["input"]))]}
        if self.fingerprint is not None:
            body["bladex_vector_fingerprint"] = self.fingerprint
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: body)


@pytest.fixture
def remote(monkeypatch):
    import httpx

    r = _Remote()
    monkeypatch.setattr(httpx, "post", r.post)
    return r


def _cache_key(adapter: ProxyEmbedAdapter, text: str = "t") -> bytes:
    fake_index = SimpleNamespace(_embedder=adapter)
    scope = MemoryIndex._embed_cache_scope(fake_index, adapter.model_identity)
    return MemoryIndex._embed_cache_key(scope, text)


def test_i46_scope_uses_the_fingerprint_the_proxy_reports(remote):
    a = ProxyEmbedAdapter("http://x")
    scope = MemoryIndex._embed_cache_scope(SimpleNamespace(_embedder=a), a.model_identity)
    assert scope == "local:m1\x00fastembed=1"
    assert "unversioned" not in scope


def test_i46_cache_key_changes_when_upstream_model_changes(remote):
    a = ProxyEmbedAdapter("http://x")
    before = _cache_key(a)
    remote.identity = "local:m2"
    a.embed(["miss"])
    assert _cache_key(a) != before


def test_i46_cache_key_changes_when_upstream_library_changes(remote):
    a = ProxyEmbedAdapter("http://x")
    before = _cache_key(a)
    remote.fingerprint = "fastembed=2"
    a.embed(["miss"])
    assert _cache_key(a) != before


def test_i46_old_proxy_bypasses_the_cache_and_probes_at_a_limited_rate(remote):
    """旧版 proxy 不回报 ⇒ 这次不读不写（同 IPC 档），也不能每次都打一发真实嵌入。"""
    remote.fingerprint = None
    a = ProxyEmbedAdapter("http://x")
    assert MemoryIndex._embed_cache_scope(SimpleNamespace(_embedder=a), "id") is None
    calls = remote.calls
    for _ in range(5):
        assert a.vector_fingerprint == ""
    assert remote.calls == calls


def test_i46_unreachable_proxy_answers_empty_not_raises(monkeypatch):
    import httpx

    def boom(*_a: object, **_k: object) -> None:
        raise httpx.ConnectError("down")

    monkeypatch.setattr(httpx, "post", boom)
    assert ProxyEmbedAdapter("http://x").vector_fingerprint == ""


def _embeddings_client(embedder):
    from bladex_proxy.server import create_app
    from starlette.testclient import TestClient

    cfg = ProxyConfig()
    cfg.auth_enabled = False
    app = create_app(cfg)
    client = TestClient(app)
    app.state.config = cfg
    app.state.embedder = embedder
    app.state.embed_call_log = SimpleNamespace(record=lambda **_k: None)
    return client


class _FpEmbedder:
    model_name, model_identity, vector_fingerprint = "m", "local:m", "fastembed=9;ort=9;dtype=float32"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] for _ in texts]


class _NoFpEmbedder:
    model_name, model_identity = "x/y", "api:x/y"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] for _ in texts]


@pytest.mark.parametrize(("embedder", "expected"), [
    (_FpEmbedder(), "fastembed=9;ort=9;dtype=float32"),
    (_NoFpEmbedder(), "api:x/y"),  # api 档：版本在供应商那边，能知道的只有模型身份
])
def test_i46_endpoint_reports_fingerprint(embedder, expected):
    r = _embeddings_client(embedder).post("/v1/embeddings", json={"input": ["a"]})
    assert r.status_code == 200
    assert r.json()["bladex_vector_fingerprint"] == expected
