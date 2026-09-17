"""ADR-0023 T4: responses_stream_generator 单元测试。

用 fake 上游 stream（async iterator of chunks）验证：
  - 事件序列正确（response.created -> output_item.added -> content_part.added
    -> output_text.delta(×N) -> content_part.done -> output_item.done -> response.completed）
  - 文本拼接完整
  - tool_call delta 聚合成 function_call item
  - CaptureResult 填充正确（full_text / tool_events / done）
"""
from __future__ import annotations

import json
from typing import Any

import pytest
from bladex_proxy.capture import CaptureResult
from bladex_proxy.responses import responses_stream_generator


class _Delta:
    def __init__(self, content: str | None = None, tool_calls: list | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls


class _ToolCallFunc:
    def __init__(self, name: str = "", arguments: str = "") -> None:
        self.name = name
        self.arguments = arguments


class _ToolCallDelta:
    def __init__(self, index: int, id_: str | None, func: _ToolCallFunc | None) -> None:
        self.index = index
        self.id = id_
        self.function = func


class _Choice:
    def __init__(self, delta: _Delta, finish_reason: str | None = None) -> None:
        self.delta = delta
        self.finish_reason = finish_reason


class _Chunk:
    def __init__(self, choices: list, usage: Any = None) -> None:
        self.choices = choices
        self.usage = usage


class _Usage:
    def __init__(self, p: int, c: int) -> None:
        self.prompt_tokens = p
        self.completion_tokens = c
        self.total_tokens = p + c


async def _astream(chunks: list) -> Any:
    for c in chunks:
        yield c


def _parse_sse(raw: bytes) -> list[tuple[str, dict]]:
    """把 SSE 字节流解析成 (event_type, data) 列表。"""
    events: list[tuple[str, dict]] = []
    for block in raw.decode().split("\n\n"):
        block = block.strip()
        if not block:
            continue
        etype = ""
        data_lines: list[str] = []
        for line in block.split("\n"):
            if line.startswith("event: "):
                etype = line[len("event: "):]
            elif line.startswith("data: "):
                data_lines.append(line[len("data: "):])
        if etype and data_lines:
            events.append((etype, json.loads("\n".join(data_lines))))
    return events


@pytest.mark.asyncio
async def test_stream_text_only_event_sequence() -> None:
    chunks = [
        _Chunk([_Choice(_Delta(content="Hello"))]),
        _Chunk([_Choice(_Delta(content=" world"))]),
        _Chunk([_Choice(_Delta(), finish_reason="stop")], _Usage(10, 5)),
    ]
    result = CaptureResult()
    raw = b"".join([c async for c in responses_stream_generator(_astream(chunks), result, "gpt-5-codex")])
    events = _parse_sse(raw)
    types = [e[0] for e in events]
    # 顺序：created -> output_item.added -> content_part.added -> 2×delta -> content_part.done -> output_item.done -> completed
    assert types[0] == "response.created"
    assert "response.output_item.added" in types
    assert "response.content_part.added" in types
    assert types.count("response.output_text.delta") == 2
    assert "response.content_part.done" in types
    assert "response.output_item.done" in types
    assert types[-1] == "response.completed"

    # 拼接
    deltas = [e[1]["delta"] for e in events if e[0] == "response.output_text.delta"]
    assert "".join(deltas) == "Hello world"

    # CaptureResult
    assert result.done is True
    assert result.full_text == "Hello world"
    assert result.error == ""

    # completed 事件携带最终 output
    completed = events[-1][1]["response"]
    assert completed["status"] == "completed"
    assert completed["output_text"] == "Hello world"
    assert completed["usage"] == {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}


@pytest.mark.asyncio
async def test_stream_tool_call_aggregates_to_function_call_item() -> None:
    chunks = [
        _Chunk([_Choice(_Delta(content="I'll run it"))]),
        _Chunk([_Choice(_Delta(tool_calls=[
            _ToolCallDelta(0, "call_1", _ToolCallFunc(name="shell")),
        ]))]),
        _Chunk([_Choice(_Delta(tool_calls=[
            _ToolCallDelta(0, None, _ToolCallFunc(arguments="{\"cmd\":")),
        ]))]),
        _Chunk([_Choice(_Delta(tool_calls=[
            _ToolCallDelta(0, None, _ToolCallFunc(arguments="\"ls\"}")),
        ]))]),
        _Chunk([_Choice(_Delta(), finish_reason="tool_calls")]),
    ]
    result = CaptureResult()
    raw = b"".join([c async for c in responses_stream_generator(_astream(chunks), result, "m")])
    events = _parse_sse(raw)
    types = [e[0] for e in events]

    # 文本 item 先关闭，再开 function_call item
    assert "response.function_call_arguments.delta" in types
    assert types[-1] == "response.completed"

    # function_call arguments 拼接完整
    arg_deltas = [e[1]["delta"] for e in events if e[0] == "response.function_call_arguments.delta"]
    assert "".join(arg_deltas) == "{\"cmd\":\"ls\"}"

    # 最终 output 含 message + function_call 两个 item
    completed = events[-1][1]["response"]
    fc_items = [o for o in completed["output"] if o["type"] == "function_call"]
    assert len(fc_items) == 1
    assert fc_items[0]["call_id"] == "call_1"
    assert fc_items[0]["name"] == "shell"
    assert fc_items[0]["arguments"] == "{\"cmd\":\"ls\"}"

    # CaptureResult tool_events
    assert len(result.tool_events) == 1
    assert result.tool_events[0].tool_name == "shell"
    assert result.tool_events[0].arguments == {"cmd": "ls"}


@pytest.mark.asyncio
async def test_stream_empty_content_no_message_item() -> None:
    """没有任何文本 delta 时不产出空的 message item（只有 function_call）。"""
    chunks = [
        _Chunk([_Choice(_Delta(tool_calls=[
            _ToolCallDelta(0, "call_1", _ToolCallFunc(name="shell", arguments="{}")),
        ]))]),
        _Chunk([_Choice(_Delta(), finish_reason="tool_calls")]),
    ]
    result = CaptureResult()
    raw = b"".join([c async for c in responses_stream_generator(_astream(chunks), result, "m")])
    events = _parse_sse(raw)
    completed = events[-1][1]["response"]
    msg_items = [o for o in completed["output"] if o["type"] == "message"]
    fc_items = [o for o in completed["output"] if o["type"] == "function_call"]
    assert msg_items == []
    assert len(fc_items) == 1


@pytest.mark.asyncio
async def test_stream_interrupted_sets_error_and_completes() -> None:
    """stream 抛异常时，result.error 被设置，仍发 response.completed 收尾。"""
    async def boom() -> Any:
        yield _Chunk([_Choice(_Delta(content="partial"))])
        raise RuntimeError("upstream dead")

    result = CaptureResult()
    raw = b"".join([c async for c in responses_stream_generator(boom(), result, "m")])
    events = _parse_sse(raw)
    assert result.error == "upstream dead"
    assert result.done is False
    assert events[-1][0] == "response.completed"  # 仍收尾
