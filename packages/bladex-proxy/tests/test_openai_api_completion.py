"""T25 OpenAI 侧 API 补全验收剧本（B11.1）。

覆盖：
  - GET /v1/models/{model} retrieve：展示名命中 / 全名命中（T5b 语义）/ 无命中 404
  - POST /v1/embeddings：单串 / 批量 / 维度稳定 / 与 Memory Index 同源（同 embedder.embed）
    / 空 input 400 / 鉴权 / embedder 不可用 503
  - 统一错误信封 {"error":{"message","type"}}
"""

from __future__ import annotations

import tempfile
from typing import Any

from bladex_proxy.config import ProxyConfig
from bladex_proxy.server import create_app
from fastapi.testclient import TestClient


class _FakeEmbedder:
    """确定性 fake embedder（不加载真 e5，单测保速）。

    与 FastEmbedAdapter 同接口：embed(texts) -> list[list[float]]。
    同源断言靠"端点向量 == embedder.embed([text])[0]"证明端点走同一 embedder。
    """

    model_name = "intfloat/multilingual-e5-large"

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            # 确定性：文本 hash -> 向量；同文本同向量
            h = sum(ord(c) for c in t)
            out.append([float((h + i) % 97) for i in range(self._dim)])
        return out


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


_SENTINEL = object()


def _client_with_embedder(cfg: ProxyConfig, embedder: Any = _SENTINEL):
    if embedder is _SENTINEL:
        embedder = _FakeEmbedder()
    app = create_app(cfg)
    client = TestClient(app)
    ctx = client.__enter__()
    # 覆盖 lifespan 设置的 embedder（避免单测加载真 e5）
    app.state.embedder = embedder
    return client, app, ctx


# ── GET /v1/models/{model} retrieve ──

def test_retrieve_model_by_display_name():
    """展示名命中 -> 200 OpenAI list-item 形态。"""
    client, app, _ = _client_with_embedder(_make_config())
    try:
        resp = client.get("/v1/models/test")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "test"
        assert data["object"] == "model"
        assert data["owned_by"] == "bladex"
    finally:
        client.__exit__(None, None, None)


def test_retrieve_model_by_full_name():
    """全名（含 provider 前缀）命中 -> 200，返回展示名（T5b 全名 > 展示名语义）。"""
    client, app, _ = _client_with_embedder(_make_config())
    try:
        resp = client.get("/v1/models/openai/test")
        assert resp.status_code == 200
        assert resp.json()["id"] == "test"  # 展示名去前缀
    finally:
        client.__exit__(None, None, None)


def test_retrieve_model_not_found_404():
    """无匹配 -> 404 OpenAI 错误信封。"""
    client, app, _ = _client_with_embedder(_make_config())
    try:
        resp = client.get("/v1/models/does-not-exist")
        assert resp.status_code == 404
        body = resp.json()
        assert body["error"]["type"] == "not_found"
        assert "does-not-exist" in body["error"]["message"]
    finally:
        client.__exit__(None, None, None)


# ── POST /v1/embeddings ──

def test_embeddings_single_string():
    """单串 -> 200 OpenAI 形态 + embedding 向量 + usage。"""
    client, app, _ = _client_with_embedder(_make_config())
    try:
        resp = client.post("/v1/embeddings", json={"model": "anything", "input": "hello world"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["object"] == "embedding"
        assert data["data"][0]["index"] == 0
        assert isinstance(data["data"][0]["embedding"], list)
        assert len(data["data"][0]["embedding"]) == 8
        # model 字段恒走本地 e5（无论请求传什么）
        assert data["model"] == _FakeEmbedder.model_name
        assert "prompt_tokens" in data["usage"]
        assert "total_tokens" in data["usage"]
    finally:
        client.__exit__(None, None, None)


def test_embeddings_batch_list():
    """批量 list[str] -> 多条 data，index 连续。"""
    client, app, _ = _client_with_embedder(_make_config())
    try:
        resp = client.post("/v1/embeddings", json={"input": ["aaa", "bbb", "ccc"]})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["data"]) == 3
        assert [d["index"] for d in data["data"]] == [0, 1, 2]
        # 不同文本 -> 不同向量
        assert data["data"][0]["embedding"] != data["data"][1]["embedding"]
    finally:
        client.__exit__(None, None, None)


def test_embeddings_deterministic_and_same_source():
    """同文本两次 -> 同向量；端点向量 == embedder.embed([text])[0]（同源可比对）。"""
    fake = _FakeEmbedder()
    client, app, _ = _client_with_embedder(_make_config(), embedder=fake)
    try:
        r1 = client.post("/v1/embeddings", json={"input": "同源文本"}).json()
        r2 = client.post("/v1/embeddings", json={"input": "同源文本"}).json()
        assert r1["data"][0]["embedding"] == r2["data"][0]["embedding"]
        # 同源：端点走的就是 app.state.embedder.embed
        direct = fake.embed(["同源文本"])[0]
        assert r1["data"][0]["embedding"] == direct
    finally:
        client.__exit__(None, None, None)


def test_embeddings_empty_input_400():
    """空 input（"" / [] / 缺失）-> 400。"""
    client, app, _ = _client_with_embedder(_make_config())
    try:
        for body in ({"input": ""}, {"input": []}, {"model": "x"}):
            resp = client.post("/v1/embeddings", json=body)
            assert resp.status_code == 400, body
            assert resp.json()["error"]["type"] == "invalid_request_error"
    finally:
        client.__exit__(None, None, None)


def test_embeddings_auth_rejected_401():
    """auth 开 + 无/错 key -> 401 OpenAI 信封。"""
    client, app, _ = _client_with_embedder(_make_config(auth=True))
    try:
        resp = client.post("/v1/embeddings", json={"input": "hi"})
        assert resp.status_code == 401
        assert resp.json()["error"]["type"] == "authentication_error"
        resp = client.post("/v1/embeddings", json={"input": "hi"},
                           headers={"Authorization": "Bearer wrong"})
        assert resp.status_code == 401
    finally:
        client.__exit__(None, None, None)


def test_embeddings_auth_valid_key_passes():
    """auth 开 + 有效 key -> 200。"""
    client, app, _ = _client_with_embedder(_make_config(auth=True))
    try:
        resp = client.post("/v1/embeddings", json={"input": "hi"},
                           headers={"Authorization": "Bearer bladex-valid"})
        assert resp.status_code == 200
    finally:
        client.__exit__(None, None, None)


def test_embeddings_embedder_unavailable_503():
    """embedder=None（Memory Index 未初始化）-> 503。"""
    client, app, _ = _client_with_embedder(_make_config(), embedder=None)
    try:
        resp = client.post("/v1/embeddings", json={"input": "hi"})
        assert resp.status_code == 503
        assert resp.json()["error"]["type"] == "service_unavailable"
    finally:
        client.__exit__(None, None, None)


def test_embeddings_error_envelope_shape():
    """错误信封统一 {"error":{"message","type"}}（无多余/缺字段）。"""
    client, app, _ = _client_with_embedder(_make_config())
    try:
        resp = client.post("/v1/embeddings", json={"input": ""})
        body = resp.json()
        assert set(body.keys()) == {"error"}
        assert set(body["error"].keys()) == {"message", "type"}
    finally:
        client.__exit__(None, None, None)
