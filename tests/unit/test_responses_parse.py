"""ADR-0023 T2: parse_responses_request 单元测试。

覆盖 OpenAI Responses API 请求 -> 内部 OpenAI chat 格式的归一：
  - instructions -> system 消息
  - input string / list of items
  - message item (user/assistant) + content blocks
  - function_call / function_call_output items
  - tools 扁平 -> 嵌套
  - max_output_tokens / temperature / tool_choice 透传
  - reasoning.effort -> reasoning_effort
"""
from __future__ import annotations

from bladex_proxy.responses import (
    format_responses_response,
    parse_responses_request,
)

# ── parse: instructions + input ───────────────────────────────────────────


def test_parse_instructions_become_system_message() -> None:
    body = {
        "model": "gpt-5-codex",
        "instructions": "You are a coding agent.",
        "input": "hello",
    }
    msgs, extra = parse_responses_request(body)
    assert msgs[0] == {"role": "system", "content": "You are a coding agent."}
    assert msgs[1] == {"role": "user", "content": "hello"}


def test_parse_input_string_becomes_user_message() -> None:
    msgs, _ = parse_responses_request({"input": "just a string"})
    assert msgs == [{"role": "user", "content": "just a string"}]


def test_parse_input_message_item_with_input_text_blocks() -> None:
    body = {
        "input": [
            {"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "line one"},
                {"type": "input_text", "text": "line two"},
            ]},
        ]
    }
    msgs, _ = parse_responses_request(body)
    assert msgs == [{"role": "user", "content": "line one\nline two"}]


def test_parse_input_assistant_message_with_output_text_and_function_call() -> None:
    """assistant 历史 message：output_text -> content，function_call -> tool_calls。"""
    body = {
        "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "list files"}]},
            {"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": "I'll run a command."},
                {"type": "function_call", "call_id": "call_1", "name": "shell", "arguments": "{\"cmd\":\"ls\"}"},
            ]},
            {"type": "message", "role": "user", "content": [
                {"type": "function_call_output", "call_id": "call_1", "output": "file_a\nfile_b"},
            ]},
        ]
    }
    msgs, _ = parse_responses_request(body)
    assert msgs[0] == {"role": "user", "content": "list files"}
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"] == "I'll run a command."
    assert msgs[1]["tool_calls"] == [{
        "id": "call_1", "type": "function",
        "function": {"name": "shell", "arguments": "{\"cmd\":\"ls\"}"},
    }]
    assert msgs[2] == {"role": "tool", "tool_call_id": "call_1", "content": "file_a\nfile_b"}


def test_parse_top_level_function_call_item() -> None:
    """顶层 function_call item（不在 message content 里）-> assistant tool_calls。"""
    body = {
        "input": [
            {"type": "function_call", "call_id": "call_9", "name": "shell", "arguments": "{\"cmd\":\"pwd\"}"},
        ]
    }
    msgs, _ = parse_responses_request(body)
    assert len(msgs) == 1
    assert msgs[0]["role"] == "assistant"
    assert msgs[0]["tool_calls"][0]["id"] == "call_9"
    assert msgs[0]["tool_calls"][0]["function"]["name"] == "shell"


def test_parse_top_level_function_call_output_item() -> None:
    body = {"input": [
        {"type": "function_call_output", "call_id": "c1", "output": "/home/user"},
    ]}
    msgs, _ = parse_responses_request(body)
    assert msgs == [{"role": "tool", "tool_call_id": "c1", "content": "/home/user"}]


def test_parse_unknown_input_item_type_warning_skipped() -> None:
    """未知 item 类型不阻断，跳过（warning）。"""
    body = {"input": [
        {"type": "message", "role": "user", "content": "keep me"},
        {"type": "mystery_item", "foo": "bar"},
    ]}
    msgs, _ = parse_responses_request(body)
    assert msgs == [{"role": "user", "content": "keep me"}]


# ── parse: tools / tool_choice / kwargs ──────────────────────────────────


def test_parse_tools_flat_to_nested_function() -> None:
    """Responses 扁平 tools -> OpenAI 嵌套 {type:function, function:{...}}。"""
    body = {
        "input": "hi",
        "tools": [
            {"type": "function", "name": "shell", "description": "run shell", "parameters": {"type": "object"}},
        ],
    }
    _, extra = parse_responses_request(body)
    assert extra["tools"] == [{
        "type": "function",
        "function": {
            "name": "shell", "description": "run shell",
            "parameters": {"type": "object"},
        },
    }]


def test_parse_tools_already_nested_passthrough() -> None:
    """已经是嵌套结构的 tools 原样透传（兼容客户端直接发 OpenAI 格式）。"""
    nested = {"type": "function", "function": {"name": "shell", "parameters": {}}}
    _, extra = parse_responses_request({"input": "hi", "tools": [nested]})
    assert extra["tools"] == [nested]


def test_parse_max_output_tokens_to_max_tokens() -> None:
    _, extra = parse_responses_request({"input": "hi", "max_output_tokens": 4096})
    assert extra["max_tokens"] == 4096


def test_parse_temperature_top_p_passthrough() -> None:
    _, extra = parse_responses_request({"input": "hi", "temperature": 0.2, "top_p": 0.9})
    assert extra["temperature"] == 0.2
    assert extra["top_p"] == 0.9


def test_parse_tool_choice_passthrough() -> None:
    _, extra = parse_responses_request({"input": "hi", "tool_choice": "auto"})
    assert extra["tool_choice"] == "auto"


def test_parse_reasoning_effort_to_extra() -> None:
    _, extra = parse_responses_request({"input": "hi", "reasoning": {"effort": "high"}})
    assert extra["reasoning_effort"] == "high"


def test_parse_no_instructions_no_system_message() -> None:
    """无 instructions 时不开头塞空 system 消息。"""
    msgs, _ = parse_responses_request({"input": "hi"})
    assert all(m["role"] != "system" for m in msgs)


def test_parse_instructions_as_list_blocks() -> None:
    """instructions 也可能是 block 数组（提取文本）。"""
    body = {
        "instructions": [{"type": "input_text", "text": "be helpful"}],
        "input": "hi",
    }
    msgs, _ = parse_responses_request(body)
    assert msgs[0] == {"role": "system", "content": "be helpful"}


# ── format: 非流式响应转换 ───────────────────────────────────────────────


class _Func:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class _ToolCall:
    def __init__(self, id_: str, func: _Func) -> None:
        self.id = id_
        self.function = func


class _Message:
    def __init__(self, content: str | None, tool_calls: list | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls


class _Choice:
    def __init__(self, message: _Message, finish_reason: str) -> None:
        self.message = message
        self.finish_reason = finish_reason


class _Usage:
    def __init__(self, p: int, c: int) -> None:
        self.prompt_tokens = p
        self.completion_tokens = c
        self.total_tokens = p + c


class _Resp:
    def __init__(self, choices: list, usage: _Usage | None = None) -> None:
        self.choices = choices
        self.usage = usage


def test_format_text_only_response() -> None:
    resp = _Resp([_Choice(_Message("hello world"), "stop")], _Usage(10, 5))
    out = format_responses_response(resp, "gpt-5-codex")
    assert out["object"] == "response"
    assert out["status"] == "completed"
    assert out["output_text"] == "hello world"
    assert out["output"][0]["type"] == "message"
    assert out["output"][0]["content"] == [{"type": "output_text", "text": "hello world"}]
    assert out["usage"] == {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}


def test_format_tool_call_response() -> None:
    resp = _Resp([_Choice(
        _Message(None, [_ToolCall("call_1", _Func("shell", "{\"cmd\":\"ls\"}"))]),
        "tool_calls",
    )], _Usage(8, 2))
    out = format_responses_response(resp, "gpt-5-codex")
    assert out["status"] == "completed"
    fc = out["output"][0]
    assert fc["type"] == "function_call"
    assert fc["call_id"] == "call_1"
    assert fc["name"] == "shell"
    assert fc["arguments"] == "{\"cmd\":\"ls\"}"


def test_format_length_finish_maps_to_incomplete() -> None:
    resp = _Resp([_Choice(_Message("..."), "length")])
    out = format_responses_response(resp, "m")
    assert out["status"] == "incomplete"
