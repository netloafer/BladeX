"""T26 Anthropic 侧 API 补全验收剧本（B11.2，T3 Claude Code 真实流量前置）。

覆盖：
  - POST /v1/messages/count_tokens：返 input_tokens、归一后计数（system 折进）、
    CJK 计数、tools 计入、鉴权、缺 messages 400、错误信封
  - GET /v1/models + /v1/models/{model} 按 anthropic-version header 分流两形态
  - Anthropic 错误信封 {"type":"error","error":{"type","message"}} 404/400
  - 模拟 Claude Code 请求序列（先 count_tokens 后 messages）不 404
"""

from __future__ import annotations

import tempfile
from unittest.mock import MagicMock, patch

from bladex_proxy.anthropic import approx_count_tokens
from bladex_proxy.config import ProxyConfig
from bladex_proxy.server import create_app
from fastapi.testclient import TestClient

_ANTHROPIC_HEADERS = {"anthropic-version": "2023-06-01"}


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


def _non_stream_response_mock():
    mock_response = MagicMock()
    mock_msg = MagicMock()
    mock_msg.content = "Hello from Claude"
    mock_msg.tool_calls = None
    mock_choice = MagicMock()
    mock_choice.message = mock_msg
    mock_choice.finish_reason = "stop"
    mock_response.choices = [mock_choice]
    mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
    return mock_response


# ── approx_count_tokens 单元 ──

def test_approx_count_tokens_formula():
    """锁近似公式：base 3 + 每消息 3 + 文本（CJK 1/字、latin 1/4字）。"""
    # 单条 "Hello world"（11 latin chars -> 2）: 3 + 3 + 2 = 8
    assert approx_count_tokens([{"role": "user", "content": "Hello world"}]) == 8
    # CJK 4 字 -> 4: 3 + 3 + 4 = 10
    assert approx_count_tokens([{"role": "user", "content": "你好世界"}]) == 10


def test_approx_count_tokens_system_folded():
    """system 折进 messages[0] 后计入（parse_anthropic_request 归一）。"""
    from bladex_proxy.anthropic import parse_anthropic_request
    body = {"system": "You are helpful.", "messages": [{"role": "user", "content": "Hi"}]}
    messages, _ = parse_anthropic_request(body)
    n_with_system = approx_count_tokens(messages)
    n_without = approx_count_tokens([{"role": "user", "content": "Hi"}])
    assert n_with_system > n_without  # system 计入


def test_approx_count_tokens_tools_counted():
    """tools 声明计入 token。"""
    base = approx_count_tokens([{"role": "user", "content": "Hi"}])
    tools = [{"name": "web_search", "description": "Search the web",
              "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}}}]
    with_tools = approx_count_tokens([{"role": "user", "content": "Hi"}], tools=tools)
    assert with_tools > base


# ── POST /v1/messages/count_tokens 端点 ──

def test_count_tokens_returns_input_tokens():
    """端点返 {"input_tokens": N}。"""
    app = create_app(_make_config())
    with TestClient(app) as client:
        resp = client.post("/v1/messages/count_tokens",
                           json={"messages": [{"role": "user", "content": "Hello world"}]},
                           headers=_ANTHROPIC_HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert data["input_tokens"] == 8
        assert isinstance(data["input_tokens"], int)


def test_count_tokens_system_counted_via_normalization():
    """带 system 的请求计数 > 不带 system（归一后 system 折进 messages）。"""
    app = create_app(_make_config())
    with TestClient(app) as client:
        r_with = client.post("/v1/messages/count_tokens",
                             json={"system": "You are helpful.",
                                   "messages": [{"role": "user", "content": "Hi"}]},
                             headers=_ANTHROPIC_HEADERS)
        r_without = client.post("/v1/messages/count_tokens",
                                json={"messages": [{"role": "user", "content": "Hi"}]},
                                headers=_ANTHROPIC_HEADERS)
        assert r_with.json()["input_tokens"] > r_without.json()["input_tokens"]


def test_count_tokens_cjk_counts_higher():
    """同长 CJK 文本 token 数高于 latin（CJK ~1/字 vs latin ~1/4字）。"""
    app = create_app(_make_config())
    with TestClient(app) as client:
        r_latin = client.post("/v1/messages/count_tokens",
                              json={"messages": [{"role": "user", "content": "abcdefgh"}]},
                              headers=_ANTHROPIC_HEADERS)
        r_cjk = client.post("/v1/messages/count_tokens",
                            json={"messages": [{"role": "user", "content": "你好世界测试"}]},
                            headers=_ANTHROPIC_HEADERS)
        # 8 latin -> 2；6 cjk -> 6
        assert r_cjk.json()["input_tokens"] > r_latin.json()["input_tokens"]


def test_count_tokens_missing_messages_400():
    """缺 messages -> 400 Anthropic 信封。"""
    app = create_app(_make_config())
    with TestClient(app) as client:
        resp = client.post("/v1/messages/count_tokens", json={"system": "x"},
                           headers=_ANTHROPIC_HEADERS)
        assert resp.status_code == 400
        body = resp.json()
        assert body["type"] == "error"
        assert body["error"]["type"] == "invalid_request_error"


def test_count_tokens_auth_rejected_401():
    """auth 开 + 无 key -> 401 Anthropic 信封。"""
    app = create_app(_make_config(auth=True))
    with TestClient(app) as client:
        resp = client.post("/v1/messages/count_tokens",
                           json={"messages": [{"role": "user", "content": "Hi"}]},
                           headers=_ANTHROPIC_HEADERS)
        assert resp.status_code == 401
        assert resp.json()["error"]["type"] == "authentication_error"


def test_count_tokens_auth_valid_via_x_api_key():
    """auth 开 + x-api-key（Claude Code 默认头）-> 200。"""
    app = create_app(_make_config(auth=True))
    with TestClient(app) as client:
        resp = client.post("/v1/messages/count_tokens",
                           json={"messages": [{"role": "user", "content": "Hi"}]},
                           headers={**_ANTHROPIC_HEADERS, "x-api-key": "bladex-valid"})
        assert resp.status_code == 200


# ── models 按 anthropic-version header 分流 ──

def test_models_list_anthropic_shape_with_header():
    """带 anthropic-version -> Anthropic 形态。"""
    app = create_app(_make_config())
    with TestClient(app) as client:
        resp = client.get("/v1/models", headers=_ANTHROPIC_HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert data["data"][0]["type"] == "model"
        assert "display_name" in data["data"][0]
        assert data["has_more"] is False
        assert "first_id" in data and "last_id" in data


def test_models_list_openai_shape_without_header():
    """不带 header -> OpenAI 形态（零回归）。"""
    app = create_app(_make_config())
    with TestClient(app) as client:
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert data["data"][0]["object"] == "model"


def test_retrieve_model_anthropic_shape_with_header():
    """retrieve 带 header -> Anthropic 单模型形态。"""
    app = create_app(_make_config())
    with TestClient(app) as client:
        resp = client.get("/v1/models/test", headers=_ANTHROPIC_HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "model"
        assert data["id"] == "test"
        assert data["display_name"] == "test"


def test_retrieve_model_anthropic_404_envelope():
    """retrieve 无匹配 + header -> 404 Anthropic 信封。"""
    app = create_app(_make_config())
    with TestClient(app) as client:
        resp = client.get("/v1/models/nope", headers=_ANTHROPIC_HEADERS)
        assert resp.status_code == 404
        body = resp.json()
        assert body["type"] == "error"
        assert body["error"]["type"] == "not_found_error"


def test_anthropic_error_envelope_shape():
    """Anthropic 错误信封统一 {"type":"error","error":{"type","message"}}。"""
    app = create_app(_make_config())
    with TestClient(app) as client:
        resp = client.post("/v1/messages/count_tokens", json={"system": "x"},
                           headers=_ANTHROPIC_HEADERS)
        body = resp.json()
        assert set(body.keys()) == {"type", "error"}
        assert set(body["error"].keys()) == {"type", "message"}


# ── 模拟 Claude Code 请求序列（T3 前置验收）──

def test_claude_code_sequence_count_tokens_then_messages():
    """Claude Code 每轮先 count_tokens 再 messages -- 两步都不 404。"""
    app = create_app(_make_config())
    mock_response = _non_stream_response_mock()

    async def fake_acompletion(model, messages, stream, **kwargs):
        return mock_response

    with patch("bladex_proxy.route.router_sdk.acompletion", new=fake_acompletion):
        with TestClient(app) as client:
            # 1. count_tokens（Claude Code 每轮请求前调用）
            r1 = client.post("/v1/messages/count_tokens",
                             json={"model": "claude-3-5-sonnet-20241022",
                                   "system": "You are helpful.",
                                   "messages": [{"role": "user", "content": "Hello"}],
                                   "max_tokens": 100},
                             headers=_ANTHROPIC_HEADERS)
            assert r1.status_code == 200
            assert "input_tokens" in r1.json()

            # 2. messages
            r2 = client.post("/v1/messages",
                             json={"model": "claude-3-5-sonnet-20241022",
                                   "system": "You are helpful.",
                                   "messages": [{"role": "user", "content": "Hello"}],
                                   "max_tokens": 100},
                             headers=_ANTHROPIC_HEADERS)
            assert r2.status_code == 200
            assert r2.json()["type"] == "message"
