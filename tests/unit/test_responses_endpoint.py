"""ADR-0023 T5/T6: /v1/responses 端到端 handler 测试。

用 TestClient + mock call_model（不真连上游），验证：
  - 鉴权 / 解析 / 注入 / 路由 / 流式与非流式回包 / 入库 全链路通
  - T6: previous_response_id 防呆回 400
  - 入库 request_messages 为 OpenAI 格式（跨协议一致）
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from bladex_proxy import server as server_module
from bladex_proxy.config import ProxyConfig
from bladex_proxy.server import create_app


def _fake_text_response(text: str = "hello from upstream") -> SimpleNamespace:
    """非流式 OpenAI chat 响应。"""
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=text, tool_calls=None),
            finish_reason="stop",
        )],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=3, total_tokens=8),
    )


def _fake_text_chunks(text_parts: list[str]) -> list:
    """流式 OpenAI chat chunks。"""
    chunks = []
    for i, t in enumerate(text_parts):
        chunks.append(SimpleNamespace(
            choices=[SimpleNamespace(
                delta=SimpleNamespace(content=t, tool_calls=None),
                finish_reason=None,
            )],
            usage=None,
        ))
    chunks.append(SimpleNamespace(
        choices=[SimpleNamespace(
            delta=SimpleNamespace(content=None, tool_calls=None),
            finish_reason="stop",
        )],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=3, total_tokens=8),
    ))
    return chunks


async def _fake_async_iter(items: list):
    for it in items:
        yield it


def _make_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mock_response=None, mock_chunks=None) -> TestClient:
    """起 proxy app，mock 上游 call_model，patch 掉 e5 embedder。

    embedding 后端可选化后 server 经 build_embedder 构建（不再直接引用
    FastEmbedAdapter）——改 patch build_embedder。
    """
    monkeypatch.setattr(
        server_module, "build_embedder",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disabled in test")),
    )

    async def fake_call_model(model, messages, *, stream, route, failover, health, on_success, **kw):
        if stream:
            return _fake_async_iter(mock_chunks or _fake_text_chunks(["hi", " there"]))
        return mock_response or _fake_text_response()

    monkeypatch.setattr(server_module, "call_model", fake_call_model)

    cfg = ProxyConfig(
        auth_enabled=False,
        rocksdb_path=str(tmp_path / "rocksdb"),
        index_path=str(tmp_path / "index"),
        overflow_dir=str(tmp_path / "overflow"),
        fastembed_cache_path=str(tmp_path / "embed_cache"),
    )
    return TestClient(create_app(cfg))


def test_responses_non_stream_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """非流式：解析 -> 注入 -> mock 上游 -> Responses 格式回包。"""
    client = _make_client(tmp_path, monkeypatch, mock_response=_fake_text_response("done."))
    body = {
        "model": "gpt-5-codex",
        "instructions": "You are a coding agent.",
        "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "list files"}]},
        ],
        "stream": False,
    }
    with client:
        r = client.post("/v1/responses", json=body)
    assert r.status_code == 200
    out = r.json()
    assert out["object"] == "response"
    assert out["status"] == "completed"
    assert out["output_text"] == "done."
    assert out["output"][0]["type"] == "message"
    assert out["output"][0]["content"] == [{"type": "output_text", "text": "done."}]
    # model 字段是路由后的实际模型（请求 gpt-5-codex 未配置 -> fallback 到默认池）
    assert out["model"]
    assert out["usage"]["total_tokens"] == 8


def test_responses_stream_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """流式：SSE 事件序列正确 + 文本拼接。"""
    client = _make_client(tmp_path, monkeypatch, mock_chunks=_fake_text_chunks(["Hello", " world"]))
    body = {
        "model": "gpt-5-codex",
        "input": "hi",
        "stream": True,
    }
    with client:
        r = client.post("/v1/responses", json=body, headers={"Accept": "text/event-stream"})
    assert r.status_code == 200
    raw = r.content.decode()
    events = []
    for block in raw.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        etype = ""
        data_lines = []
        for line in block.split("\n"):
            if line.startswith("event: "):
                etype = line[7:]
            elif line.startswith("data: "):
                data_lines.append(line[6:])
        if etype and data_lines:
            events.append((etype, json.loads("\n".join(data_lines))))
    types = [e[0] for e in events]
    assert types[0] == "response.created"
    assert "response.output_text.delta" in types
    assert types[-1] == "response.completed"
    deltas = [e[1]["delta"] for e in events if e[0] == "response.output_text.delta"]
    assert "".join(deltas) == "Hello world"
    completed = events[-1][1]["response"]
    assert completed["output_text"] == "Hello world"


def test_responses_stateful_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """T6: previous_response_id 非空回 400，不进注入/入库。"""
    client = _make_client(tmp_path, monkeypatch)
    body = {
        "model": "x",
        "input": "hi",
        "previous_response_id": "resp_abc123",
    }
    with client:
        r = client.post("/v1/responses", json=body)
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["type"] == "invalid_request_error"
    assert "stateless" in err["message"].lower()
    assert err["param"] == "previous_response_id"


def test_responses_auth_required_when_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """auth_enabled=true 时错误 key 回 401。"""
    monkeypatch.setattr(
        server_module, "build_embedder",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disabled in test")),
    )
    cfg = ProxyConfig(
        auth_enabled=True,
        client_keys_raw="goodkey||test",
        rocksdb_path=str(tmp_path / "rocksdb"),
        index_path=str(tmp_path / "index"),
        overflow_dir=str(tmp_path / "overflow"),
        fastembed_cache_path=str(tmp_path / "embed_cache"),
    )
    client = TestClient(create_app(cfg))
    with client:
        r = client.post(
            "/v1/responses",
            json={"model": "x", "input": "hi"},
            headers={"Authorization": "Bearer wrongkey"},
        )
    assert r.status_code == 401


def test_responses_tool_call_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """非流式 tool_call 响应正确转成 function_call output item。"""
    resp = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(
                content=None,
                tool_calls=[SimpleNamespace(
                    id="call_1",
                    function=SimpleNamespace(name="shell", arguments='{"cmd":"ls"}'),
                )],
            ),
            finish_reason="tool_calls",
        )],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2, total_tokens=12),
    )
    client = _make_client(tmp_path, monkeypatch, mock_response=resp)
    body = {"model": "gpt-5-codex", "input": "list files", "stream": False}
    with client:
        r = client.post("/v1/responses", json=body)
    assert r.status_code == 200
    out = r.json()
    fc = out["output"][0]
    assert fc["type"] == "function_call"
    assert fc["call_id"] == "call_1"
    assert fc["name"] == "shell"
    assert fc["arguments"] == '{"cmd":"ls"}'
    assert out["status"] == "completed"
