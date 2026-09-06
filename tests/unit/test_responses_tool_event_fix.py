"""ADR-0023 复核修复：tool_event 跨轮 name 补全。

真实 Codex 流量发现两个问题（2026-07-25 复核）：
  Bug A: responses_stream_generator 的 call 方向 ToolEvent 没填 tool_call_id
         -> 上一轮 call id 丢失，下一轮 result 匹配不上
  Bug B: pipeline.enrich_tool_results 的 result name 只查当前轮 call_map，
         上一轮的工具返回（在 request_messages 历史里）name 永远空

修复：
  A: generator 填 tool_call_id=acc["call_id"]
  B: enrich_tool_results 从 request_messages 的 assistant tool_calls 建补充
     name 映射，作为 _extract_tool_name 第 4 层 fallback（通用，惠及所有 agent）
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from bladex_proxy.capture import CaptureResult
from bladex_proxy.models import Identity, ToolEvent, Turn
from bladex_proxy.responses import responses_stream_generator
from bladex_proxy.storage.pipeline_worker import enrich_tool_results


# ── Bug A: generator 填 tool_call_id ─────────────────────────────────────


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
    def __init__(self, tool_calls: list | None = None) -> None:
        self.content = None
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


@pytest.mark.asyncio
async def test_generator_fills_tool_call_id() -> None:
    """Bug A 修复：call 方向 ToolEvent 必须带 tool_call_id（供下一轮 result 匹配）。"""
    chunks = [
        _Chunk([_Choice(_Delta(tool_calls=[
            _TCDelta(0, "call_abc123", _Func(name="shell")),
        ]))]),
        _Chunk([_Choice(_Delta(tool_calls=[
            _TCDelta(0, None, _Func(arguments='{"cmd":"ls"}')),
        ]))]),
        _Chunk([_Choice(_Delta(), finish_reason="tool_calls")]),
    ]
    result = CaptureResult()
    async for _ in responses_stream_generator(_astream(chunks), result, "m"):
        pass
    assert len(result.tool_events) == 1
    te = result.tool_events[0]
    assert te.direction == "call"
    assert te.tool_name == "shell"
    assert te.tool_call_id == "call_abc123"  # Bug A: 修复前为空


# ── Bug B: 跨轮 tool_result name 补全 ────────────────────────────────────


def _make_turn(request_messages: list, tool_events: list) -> Turn:
    return Turn(
        identity=Identity(user_id="u1", agent_id="codex", session_id="s1"),
        request_messages=request_messages,
        tool_events=tool_events,
    )


def test_enrich_cross_turn_tool_result_name_from_request_history() -> None:
    """Bug B 修复：上一轮工具返回的 name 从 request_messages 的 assistant tool_calls 补全。

    典型 Codex 工具循环：
      turn N request = [..., assistant(tool_call id=X name=exec_command), tool(id=X result)]
      turn N response = 新的 tool_call（id=Y）-- 与 result id=X 无关

    修复前：result id=X 在当前轮 call_map（只有 id=Y）里找不到 -> name 空。
    修复后：从 request_messages 的 assistant tool_calls 建 id->name 映射 -> name=exec_command。
    """
    request_messages = [
        {"role": "user", "content": "list files"},
        # 上一轮 assistant 的 tool_call（在历史消息里，带 name 和 id）
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_X987", "type": "function",
            "function": {"name": "exec_command", "arguments": '{"cmd":"ls"}'},
        }]},
        # 上一轮工具返回（role=tool，只有 tool_call_id，无 name）
        {"role": "tool", "tool_call_id": "call_X987", "content": "file_a\nfile_b"},
        # 本轮 user 新提问
        {"role": "user", "content": "now delete file_a"},
    ]
    # 当前轮响应的 tool_call（新 id，与历史 result 无关）
    turn = _make_turn(request_messages, tool_events=[
        ToolEvent(tool_name="exec_command", arguments={"cmd": "rm file_a"},
                  direction="call", tool_call_id="call_NEW"),
    ])
    enrich_tool_results(turn)

    # 修复后：历史 result 应补上 name=exec_command（从 assistant tool_calls）
    results = [e for e in turn.tool_events if e.direction == "result"]
    assert len(results) == 1
    assert results[0].tool_name == "exec_command"  # Bug B: 修复前为空
    assert results[0].tool_call_id == "call_X987"
    assert results[0].result == "file_a\nfile_b"


def test_enrich_current_round_result_still_matches_call_map() -> None:
    """同轮 result 仍优先匹配当前轮 call_map（回归保护）。"""
    request_messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_SAME", "type": "function",
            "function": {"name": "shell", "arguments": "{}"},
        }]},
        {"role": "tool", "tool_call_id": "call_SAME", "content": "ok"},
    ]
    turn = _make_turn(request_messages, tool_events=[
        ToolEvent(tool_name="shell", arguments={}, direction="call", tool_call_id="call_SAME"),
    ])
    enrich_tool_results(turn)
    # call 方向保留，result 回填到同一条
    calls = [e for e in turn.tool_events if e.direction == "call"]
    assert len(calls) == 1
    assert calls[0].result == "ok"  # 回填成功
    # 不新增空 result
    results = [e for e in turn.tool_events if e.direction == "result"]
    assert len(results) == 0


def test_enrich_unknown_result_name_still_empty_when_no_history() -> None:
    """无历史 assistant tool_call 时，name 仍为空（不臆造）。"""
    request_messages = [
        {"role": "user", "content": "hi"},
        {"role": "tool", "tool_call_id": "call_ORPHAN", "content": "mystery"},
    ]
    turn = _make_turn(request_messages, tool_events=[])
    enrich_tool_results(turn)
    results = [e for e in turn.tool_events if e.direction == "result"]
    assert len(results) == 1
    assert results[0].tool_name == ""  # 无来源，仍空（不臆造）
    assert results[0].result == "mystery"
