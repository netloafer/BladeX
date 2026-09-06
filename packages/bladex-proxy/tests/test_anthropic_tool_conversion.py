"""Anthropic tool_result/tool_use -> OpenAI 格式转换回归测试。

背景：parse_anthropic_request 原把 Anthropic 的 tool_result block 原样塞进 user 消息
content list、tool_use block 原样塞进 assistant content，导致 OpenAI 兼容上游报
"Invalid user message at index 5" -> 502（Claude Code agentic 第二轮回传工具结果时触发）。
修复后正确转成独立的 role:tool 消息 / tool_calls 字段，丢弃 cache_control 等 Anthropic 专有字段。
"""

from __future__ import annotations

import tempfile
from unittest.mock import MagicMock, patch

from bladex_proxy.anthropic import approx_count_tokens, parse_anthropic_request
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


def _has_anthropic_block(messages: list[dict]) -> bool:
    """是否残留 Anthropic 专有 block（tool_result/tool_use）在 content list 里。"""
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            for b in c:
                if isinstance(b, dict) and b.get("type") in ("tool_result", "tool_use"):
                    return True
    return False


# ── parse_anthropic_request: tool_result -> role:tool ──

def test_tool_result_converts_to_tool_role_message():
    """user 消息含 tool_result block -> 独立 role:tool 消息（非 user content list）。"""
    body = {"messages": [
        {"role": "user", "content": "查一下"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "好的"},
            {"type": "tool_use", "id": "toolu_abc", "name": "search", "input": {"q": "x"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_abc", "content": "结果文本"},
        ]},
    ]}
    messages, _ = parse_anthropic_request(body)
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["tool_call_id"] == "toolu_abc"
    assert tool_msgs[0]["content"] == "结果文本"
    assert not _has_anthropic_block(messages)
    # 原始 user 文本保留
    assert any(m.get("role") == "user" and m.get("content") == "查一下" for m in messages)


def test_tool_result_content_as_blocks_extracted():
    """tool_result.content 是 list[text block] -> 提取纯文本。"""
    body = {"messages": [{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t1",
         "content": [{"type": "text", "text": "行1"}, {"type": "text", "text": "行2"}]},
    ]}]}
    messages, _ = parse_anthropic_request(body)
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert "行1" in tool_msgs[0]["content"]
    assert "行2" in tool_msgs[0]["content"]


def test_tool_result_cache_control_dropped():
    """tool_result 的 cache_control 字段丢弃（OpenAI 上游不认）。"""
    body = {"messages": [{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": "x",
         "cache_control": {"type": "ephemeral"}},
    ]}]}
    messages, _ = parse_anthropic_request(body)
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert "cache_control" not in tool_msgs[0]
    assert tool_msgs[0]["content"] == "x"


def test_multiple_tool_results_become_multiple_tool_messages():
    """一个 user 消息含多个 tool_result -> 多条独立 role:tool 消息。"""
    body = {"messages": [{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": "r1"},
        {"type": "tool_result", "tool_use_id": "t2", "content": "r2"},
    ]}]}
    messages, _ = parse_anthropic_request(body)
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 2
    assert {m["tool_call_id"] for m in tool_msgs} == {"t1", "t2"}


def test_tool_result_with_text_block_splits():
    """user 消息含 tool_result + text -> tool 消息 + 独立 user 文本消息。

    OpenAI 要求 tool 结果紧跟 assistant tool_calls；用户追加文本另起 user 消息。
    """
    body = {"messages": [{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": "结果"},
        {"type": "text", "text": "继续"},
    ]}]}
    messages, _ = parse_anthropic_request(body)
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    user_text = [m for m in messages if m.get("role") == "user" and m.get("content") == "继续"]
    assert len(user_text) == 1
    # tool 消息在 user 文本消息之前（紧跟 assistant）
    assert messages.index(tool_msgs[0]) < messages.index(user_text[0])


# ── parse_anthropic_request: tool_use -> tool_calls ──

def test_assistant_tool_use_converts_to_tool_calls():
    """assistant 含 tool_use block -> tool_calls 字段（arguments JSON 字符串）。"""
    body = {"messages": [{"role": "assistant", "content": [
        {"type": "text", "text": "我查一下"},
        {"type": "tool_use", "id": "call_1", "name": "search", "input": {"q": "hello"}},
    ]}]}
    messages, _ = parse_anthropic_request(body)
    assert len(messages) == 1
    m = messages[0]
    assert m["role"] == "assistant"
    assert m["content"] == "我查一下"
    assert "tool_calls" in m
    tc = m["tool_calls"][0]
    assert tc["id"] == "call_1"
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "search"
    assert isinstance(tc["function"]["arguments"], str)
    assert '"q"' in tc["function"]["arguments"]


def test_assistant_tool_use_only_no_text_content_none():
    """assistant 只含 tool_use 无 text -> content=None（OpenAI 标准）。"""
    body = {"messages": [{"role": "assistant", "content": [
        {"type": "tool_use", "id": "call_1", "name": "run", "input": {}},
    ]}]}
    messages, _ = parse_anthropic_request(body)
    assert len(messages) == 1
    assert messages[0]["content"] is None
    assert len(messages[0]["tool_calls"]) == 1


def test_assistant_plain_text_no_tool_calls():
    """assistant 纯文本 -> 无 tool_calls 字段（零回归）。"""
    body = {"messages": [{"role": "assistant", "content": [
        {"type": "text", "text": "你好"},
    ]}]}
    messages, _ = parse_anthropic_request(body)
    assert messages[0]["role"] == "assistant"
    assert messages[0]["content"] == "你好"
    assert "tool_calls" not in messages[0]


# ── Claude Code agentic 序列端到端 ──

def test_claude_code_agentic_sequence_normalizes_to_valid_openai():
    """Claude Code 完整 agentic 序列：user -> assistant(tool_use) -> user(tool_result)。

    转换后应是无 Anthropic 专有 block 的合法 OpenAI messages，且 assistant tool_calls
    的 id 与 tool 消息 tool_call_id 对应（上游配对校验能过）。
    """
    body = {
        "system": "You are Claude Code.",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "读这个文件"}]},
            {"role": "assistant", "content": [
                {"type": "text", "text": "好的"},
                {"type": "tool_use", "id": "toolu_01", "name": "Read", "input": {"path": "/a/b.py"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_01", "content": "file contents here"},
            ]},
        ],
    }
    messages, _ = parse_anthropic_request(body)
    assert messages[0]["role"] == "system"
    assert "tool" in [m["role"] for m in messages]
    assert not _has_anthropic_block(messages)
    asst = [m for m in messages if m["role"] == "assistant"][0]
    tool_msg = [m for m in messages if m["role"] == "tool"][0]
    assert asst["tool_calls"][0]["id"] == tool_msg["tool_call_id"]


def test_v1_messages_endpoint_with_tool_result_no_502():
    """端到端：POST /v1/messages 带 tool_result 的请求不再 502（上游收到 role:tool）。

    回归 502 根因：修复前 tool_result 原样塞 user content -> 上游 Invalid user message。
    """
    config = _make_config()
    app = create_app(config)

    captured: dict = {}

    async def fake_acompletion(model, messages, stream, **kwargs):
        captured["messages"] = messages
        mock_response = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = "done"
        mock_msg.tool_calls = None
        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_choice.finish_reason = "stop"
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
        return mock_response

    with patch("bladex_proxy.route.router_sdk.acompletion", new=fake_acompletion):
        with TestClient(app) as client:
            resp = client.post("/v1/messages", json={
                "model": "claude-3-5-sonnet-20241022",
                "system": "You are helpful.",
                "messages": [
                    {"role": "user", "content": "读文件"},
                    {"role": "assistant", "content": [
                        {"type": "text", "text": "好"},
                        {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"path": "x"}},
                    ]},
                    {"role": "user", "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "内容"},
                    ]},
                ],
                "max_tokens": 100,
            })

    assert resp.status_code == 200, resp.text
    upstream_msgs = captured["messages"]
    # 上游收到 role:tool 消息（不是 user 带 tool_result block）
    tool_msgs = [m for m in upstream_msgs if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["tool_call_id"] == "toolu_1"
    # assistant 带 tool_calls
    assert any("tool_calls" in m for m in upstream_msgs if m.get("role") == "assistant")
    # 无 Anthropic 专有 block 残留
    assert not _has_anthropic_block(upstream_msgs)


# ── count_tokens 对含 tool_result 的请求不报错（归一后计数）──

def test_count_tokens_normalizes_tool_result_without_error():
    """含 tool_result 的请求经归一后能计数（role:tool content 是 str）。"""
    body = {"messages": [{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": "结果文本"},
    ]}]}
    messages, _ = parse_anthropic_request(body)
    # 不抛异常 + 计入 tool 消息文本
    n = approx_count_tokens(messages)
    assert n > 0
