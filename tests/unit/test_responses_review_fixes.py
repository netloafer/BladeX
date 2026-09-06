"""ADR-0023 二轮复核修复回归测试（2026-07-25）。

验收复核发现三处遗漏，本文件覆盖其中两处的逻辑修复：

  #1 非流式 tool_events：`tool_events_from_output` 统一产出 dict arguments +
     tool_call_id（与流式一致，跨轮 result 匹配不断链）。
  #2 output_index 空洞：纯工具回复（无正文）时首个 function_call 拿 index 0，
     与 response.completed 快照一致；有正文时 message 占 0、工具从 1 起。
"""
from __future__ import annotations

import json

import pytest
from bladex_proxy.capture import CaptureResult
from bladex_proxy.responses import responses_stream_generator, tool_events_from_output

# ── 流式 SSE 解析辅助 ────────────────────────────────────────────────────


def _parse_sse(raw: bytes) -> tuple[str, dict]:
    text = raw.decode()
    event_line, data_line = text.strip().split("\n", 1)
    event = event_line.removeprefix("event: ").strip()
    data = json.loads(data_line.removeprefix("data: ").strip())
    return event, data


class _Func:
    def __init__(self, name: str = "", arguments: str = "") -> None:
        self.name = name
        self.arguments = arguments


class _TCDelta:
    def __init__(self, index: int, id_: str | None, func: _Func | None) -> None:
        self.index = index
        self.id = id_
        self.function = func


class _Delta:
    def __init__(self, content: str | None = None, tool_calls: list | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls


class _Choice:
    def __init__(self, delta: _Delta, finish_reason: str | None = None) -> None:
        self.delta = delta
        self.finish_reason = finish_reason


class _Chunk:
    def __init__(self, choices: list) -> None:
        self.choices = choices


async def _astream(chunks: list):
    for c in chunks:
        yield c


async def _collect(chunks: list) -> list[tuple[str, dict]]:
    result = CaptureResult()
    events: list[tuple[str, dict]] = []
    async for raw in responses_stream_generator(_astream(chunks), result, "m"):
        events.append(_parse_sse(raw))
    return events


# ── #2 output_index 空洞 ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_tool_only_response_first_function_call_index_zero() -> None:
    """纯工具回复（无正文）：首个 function_call 的 output_index 必须是 0（不留空洞）。"""
    chunks = [
        _Chunk([_Choice(_Delta(tool_calls=[
            _TCDelta(0, "call_a", _Func(name="shell", arguments='{"cmd":"ls"}')),
        ]))]),
        _Chunk([_Choice(_Delta(), finish_reason="tool_calls")]),
    ]
    events = await _collect(chunks)

    added = [d for e, d in events if e == "response.output_item.added"]
    assert len(added) == 1
    assert added[0]["item"]["type"] == "function_call"
    assert added[0]["output_index"] == 0  # 修复前为 1，index 0 空缺

    completed = [d for e, d in events if e == "response.completed"][0]
    out = completed["response"]["output"]
    assert out[0]["type"] == "function_call"  # 快照 output[0] 与流式 index 0 一致


@pytest.mark.asyncio
async def test_text_then_tool_indices_contiguous() -> None:
    """有正文 + 工具：message 占 0、function_call 占 1（连续无洞）。"""
    chunks = [
        _Chunk([_Choice(_Delta(content="thinking..."))]),
        _Chunk([_Choice(_Delta(tool_calls=[
            _TCDelta(0, "call_b", _Func(name="apply_patch", arguments="{}")),
        ]))]),
        _Chunk([_Choice(_Delta(), finish_reason="tool_calls")]),
    ]
    events = await _collect(chunks)

    added = [d for e, d in events if e == "response.output_item.added"]
    msg = [a for a in added if a["item"]["type"] == "message"][0]
    fc = [a for a in added if a["item"]["type"] == "function_call"][0]
    assert msg["output_index"] == 0
    assert fc["output_index"] == 1

    completed = [d for e, d in events if e == "response.completed"][0]
    out = completed["response"]["output"]
    assert out[0]["type"] == "message"
    assert out[1]["type"] == "function_call"


@pytest.mark.asyncio
async def test_two_tool_calls_indices_zero_one() -> None:
    """纯工具、两个并行调用：output_index 依次 0、1。"""
    chunks = [
        _Chunk([_Choice(_Delta(tool_calls=[
            _TCDelta(0, "call_0", _Func(name="shell", arguments="{}")),
            _TCDelta(1, "call_1", _Func(name="read", arguments="{}")),
        ]))]),
        _Chunk([_Choice(_Delta(), finish_reason="tool_calls")]),
    ]
    events = await _collect(chunks)
    added = [d for e, d in events if e == "response.output_item.added"]
    idxs = sorted(a["output_index"] for a in added)
    assert idxs == [0, 1]


# ── #1 非流式 tool_events_from_output ────────────────────────────────────


def test_tool_events_from_output_parses_args_and_call_id() -> None:
    """非流式路径：arguments 还原成 dict，且带 tool_call_id（跨轮匹配）。"""
    output = [
        {"type": "message", "content": [{"type": "output_text", "text": "hi"}]},
        {
            "type": "function_call",
            "call_id": "call_xyz",
            "name": "exec_command",
            "arguments": '{"cmd":"ls -la"}',
        },
    ]
    events = tool_events_from_output(output)
    assert len(events) == 1
    te = events[0]
    assert te.tool_name == "exec_command"
    assert te.direction == "call"
    assert te.tool_call_id == "call_xyz"  # 修复前为空
    assert te.arguments == {"cmd": "ls -la"}  # 修复前是原始字符串


def test_tool_events_from_output_bad_json_keeps_raw() -> None:
    """arguments 非合法 JSON 时保留原串，不崩。"""
    output = [{"type": "function_call", "call_id": "c1", "name": "x", "arguments": "{not json"}]
    events = tool_events_from_output(output)
    assert events[0].arguments == "{not json"
    assert events[0].tool_call_id == "c1"
