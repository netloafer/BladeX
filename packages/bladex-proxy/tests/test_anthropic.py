"""Anthropic /v1/messages 端点单元测试（T4）。

测试格式解析、SSE 流式拼接、注入位置、指纹识别。
"""

import tempfile
from unittest.mock import MagicMock, patch

import pytest
from bladex_proxy.anthropic import (
    anthropic_stream_generator,
    format_anthropic_response,
    parse_anthropic_request,
)
from bladex_proxy.capture import CaptureResult
from bladex_proxy.config import ProxyConfig
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


# ── parse_anthropic_request ──

def test_parse_anthropic_system_field():
    """Anthropic system 独立字段 → 首条 system 消息。"""
    body = {
        "model": "claude-3-5-sonnet-20241022",
        "system": "You are helpful.",
        "messages": [
            {"role": "user", "content": "Hello"},
        ],
        "max_tokens": 1024,
    }
    messages, extra = parse_anthropic_request(body)
    assert messages[0] == {"role": "system", "content": "You are helpful."}
    assert messages[1] == {"role": "user", "content": "Hello"}
    assert extra["max_tokens"] == 1024


def test_parse_anthropic_system_as_blocks():
    """system 字段为 content blocks → 拼成纯文本。"""
    body = {
        "system": [{"type": "text", "text": "Part 1"}, {"type": "text", "text": "Part 2"}],
        "messages": [{"role": "user", "content": "Hi"}],
    }
    messages, _ = parse_anthropic_request(body)
    assert messages[0]["role"] == "system"
    assert "Part 1" in messages[0]["content"]
    assert "Part 2" in messages[0]["content"]


def test_parse_anthropic_content_blocks():
    """messages content 为 block 数组 → 归一成纯文本。"""
    body = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Hello world"}]},
        ],
    }
    messages, _ = parse_anthropic_request(body)
    assert messages[0] == {"role": "user", "content": "Hello world"}


def test_parse_anthropic_tools():
    """Anthropic tools → OpenAI function calling 格式。"""
    body = {
        "messages": [{"role": "user", "content": "search"}],
        "tools": [{
            "name": "web_search",
            "description": "Search the web",
            "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
        }],
    }
    _, extra = parse_anthropic_request(body)
    assert "tools" in extra
    assert extra["tools"][0]["type"] == "function"
    assert extra["tools"][0]["function"]["name"] == "web_search"


def test_parse_anthropic_stop_sequences():
    """stop_sequences → stop 参数名映射。"""
    body = {
        "messages": [{"role": "user", "content": "hi"}],
        "stop_sequences": ["END"],
    }
    _, extra = parse_anthropic_request(body)
    assert extra["stop"] == ["END"]


# ── format_anthropic_response ──

def test_format_anthropic_response_text():
    """OpenAI 文本响应 → Anthropic content blocks。"""
    mock_msg = MagicMock()
    mock_msg.content = "Hello back"
    mock_msg.tool_calls = None
    mock_choice = MagicMock()
    mock_choice.message = mock_msg
    mock_choice.finish_reason = "stop"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

    result = format_anthropic_response(mock_response, "claude-3")
    assert result["type"] == "message"
    assert result["content"][0]["type"] == "text"
    assert result["content"][0]["text"] == "Hello back"
    assert result["stop_reason"] == "end_turn"
    assert result["usage"]["input_tokens"] == 10
    assert result["usage"]["output_tokens"] == 5


def test_format_anthropic_response_tool_use():
    """OpenAI tool_calls → Anthropic tool_use blocks。"""
    mock_func = MagicMock()
    mock_func.name = "search"
    mock_func.arguments = '{"q": "hello"}'
    mock_tc = MagicMock()
    mock_tc.id = "call_123"
    mock_tc.function = mock_func
    mock_msg = MagicMock()
    mock_msg.content = ""
    mock_msg.tool_calls = [mock_tc]
    mock_choice = MagicMock()
    mock_choice.message = mock_msg
    mock_choice.finish_reason = "tool_calls"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

    result = format_anthropic_response(mock_response, "claude-3")
    assert result["stop_reason"] == "tool_use"
    tool_block = [b for b in result["content"] if b["type"] == "tool_use"][0]
    assert tool_block["name"] == "search"
    assert tool_block["input"] == {"q": "hello"}


# ── SSE streaming ──

class _FakeDelta:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _FakeChoice:
    def __init__(self, delta, finish_reason=None):
        self.delta = delta
        self.finish_reason = finish_reason


class _FakeChunk:
    def __init__(self, choices, usage=None):
        self.choices = choices
        self.usage = usage


class _FakeStream:
    """模拟上游 async stream。"""
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        self._iter = iter(self._chunks)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration from None


@pytest.mark.asyncio
async def test_anthropic_stream_capture():
    """流式：SSE event 序列正确、full_text 拼全、tool_use 聚合。"""
    chunks = [
        _FakeChunk([_FakeChoice(_FakeDelta(content="Hello"))]),
        _FakeChunk([_FakeChoice(_FakeDelta(content=" world"))]),
        _FakeChunk([_FakeChoice(_FakeDelta(), finish_reason="stop")]),
    ]
    stream = _FakeStream(chunks)
    result = CaptureResult()

    events = []
    async for sse_bytes in anthropic_stream_generator(
        stream, result, "claude-3", input_tokens_estimate=11
    ):
        events.append(sse_bytes.decode())

    # 验证事件序列
    event_types = []
    for e in events:
        for line in e.strip().split("\n"):
            if line.startswith("event: "):
                event_types.append(line[7:])

    assert "message_start" in event_types
    assert "content_block_start" in event_types
    assert "content_block_delta" in event_types
    assert "content_block_stop" in event_types
    assert "message_delta" in event_types
    assert "message_stop" in event_types

    # 验证 full_text 拼全
    assert result.full_text == "Hello world"
    assert result.done is True
    assert result.tool_events == []


@pytest.mark.asyncio
async def test_anthropic_stream_tool_use():
    """流式：tool_use block 正确聚合。"""
    mock_func1 = MagicMock()
    mock_func1.name = "search"
    mock_func1.arguments = '{"q":"test"}'
    mock_tc1 = MagicMock()
    mock_tc1.id = "call_1"
    mock_tc1.index = 0
    mock_tc1.function = mock_func1

    chunks = [
        _FakeChunk([_FakeChoice(_FakeDelta(content="Let me search"))]),
        _FakeChunk([_FakeChoice(_FakeDelta(tool_calls=[mock_tc1]))]),
        _FakeChunk([_FakeChoice(_FakeDelta(), finish_reason="tool_calls")]),
    ]
    stream = _FakeStream(chunks)
    result = CaptureResult()

    events = []
    async for sse_bytes in anthropic_stream_generator(
        stream, result, "claude-3", input_tokens_estimate=11
    ):
        events.append(sse_bytes.decode())

    assert result.full_text == "Let me search"
    assert len(result.tool_events) == 1
    assert result.tool_events[0].tool_name == "search"
    assert result.tool_events[0].arguments == {"q": "test"}


# ── 端点集成测试 ──

def test_anthropic_non_stream():
    """POST /v1/messages 非流式：Claude Code 风格请求 → 正常回复 + 完整入库。"""
    config = _make_config()
    app = create_app(config)

    mock_response = MagicMock()
    mock_msg = MagicMock()
    mock_msg.content = "Hello from Claude"
    mock_msg.tool_calls = None
    mock_choice = MagicMock()
    mock_choice.message = mock_msg
    mock_choice.finish_reason = "stop"
    mock_response.choices = [mock_choice]
    mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

    async def fake_acompletion(model, messages, stream, **kwargs):
        return mock_response

    with patch("bladex_proxy.route.router_sdk.acompletion", new=fake_acompletion):
        with TestClient(app) as client:
            response = client.post("/v1/messages", json={
                "model": "claude-3-5-sonnet-20241022",
                "system": "You are helpful.",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 100,
            })

    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "message"
    assert data["role"] == "assistant"
    assert data["content"][0]["type"] == "text"
    assert "Hello from Claude" in data["content"][0]["text"]


def test_anthropic_inject_position():
    """注入在 Anthropic 路径生效且不破坏 system 字段。"""
    config = _make_config()
    config.hard_rules = ["NEVER use emojis."]
    app = create_app(config)

    captured_messages: list[dict] = []

    async def fake_acompletion(model, messages, stream, **kwargs):
        captured_messages.extend(messages)
        mock_response = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = "OK"
        mock_msg.tool_calls = None
        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_choice.finish_reason = "stop"
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
        return mock_response

    with patch("bladex_proxy.route.router_sdk.acompletion", new=fake_acompletion):
        with TestClient(app) as client:
            response = client.post("/v1/messages", json={
                "model": "claude-3",
                "system": "You are helpful.",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 100,
            })

    assert response.status_code == 200
    # system 消息应在 messages 中，且注入块应在最后一条 user 前
    system_msgs = [m for m in captured_messages if m.get("role") == "system"]
    assert len(system_msgs) >= 1
    assert "You are helpful." in system_msgs[0]["content"]

    # 注入的 system 消息应在最后一条 user 之前
    inject_msgs = [m for m in captured_messages if "<bladex-memory>" in str(m.get("content", ""))]
    assert len(inject_msgs) >= 1

    # 找注入消息和最后一条 user 的位置
    last_user_idx = -1
    inject_idx = -1
    for i, m in enumerate(captured_messages):
        if m.get("role") == "user":
            last_user_idx = i
        if "<bladex-memory>" in str(m.get("content", "")):
            inject_idx = i

    assert inject_idx < last_user_idx, "注入应在最后一条 user 之前"


def test_anthropic_agent_fingerprint():
    """claude-code 指纹在 Anthropic 路径命中。"""
    from bladex_proxy.identity import resolve_identity
    from bladex_proxy.models import ChatCompletionRequest

    messages = [
        {"role": "system", "content": "You are Claude Code, Anthropic's official CLI for Claude."},
        {"role": "user", "content": "Read this file"},
    ]
    req = ChatCompletionRequest(messages=messages, tools=[
        {"type": "function", "function": {"name": "Read"}},
        {"type": "function", "function": {"name": "Write"}},
    ])

    headers = {"user-agent": "claude-code/1.0"}
    identity, source = resolve_identity(headers, req)
    assert identity.agent_id == "claude-code"


def test_anthropic_x_api_key_auth():
    """x-api-key 头（Claude Code 默认发此头）在 auth 开时也能通过 /v1/messages。

    Claude Code 走 Anthropic 风格 x-api-key，非 Authorization: Bearer。
    BladeX 适配（刚性原则）：两者都接受；错 key 401。
    """
    tmpdir = tempfile.mkdtemp()
    config = ProxyConfig(
        hard_rules=[], upstream_model="openai/test", upstream_api_key="sk-fake",
        rocksdb_path=f"{tmpdir}/rocksdb",
        auth_enabled=True, client_keys_raw="bladex-test-key||claude-code",
    )
    app = create_app(config)

    mock_response = MagicMock()
    mock_msg = MagicMock()
    mock_msg.content = "ok"
    mock_msg.tool_calls = None
    mock_choice = MagicMock()
    mock_choice.message = mock_msg
    mock_choice.finish_reason = "stop"
    mock_response.choices = [mock_choice]
    mock_response.usage = MagicMock(prompt_tokens=1, completion_tokens=1)

    async def fake_acompletion(model, messages, stream, **kwargs):
        return mock_response

    body = {
        "model": "claude-3-5-sonnet-20241022",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 10,
    }

    with patch("bladex_proxy.route.router_sdk.acompletion", new=fake_acompletion):
        with TestClient(app) as client:
            # x-api-key（Claude Code 风格）-> 200
            r1 = client.post("/v1/messages", json=body, headers={"x-api-key": "bladex-test-key"})
            assert r1.status_code == 200, r1.text
            # Authorization: Bearer 仍然兼容 -> 200
            r2 = client.post("/v1/messages", json=body,
                             headers={"Authorization": "Bearer bladex-test-key"})
            assert r2.status_code == 200, r2.text
            # 错误 key -> 401
            r3 = client.post("/v1/messages", json=body, headers={"x-api-key": "wrong-key"})
            assert r3.status_code == 401
            # 无 key -> 401
            r4 = client.post("/v1/messages", json=body)
            assert r4.status_code == 401
