"""批 EH · MQ-P46：流式混合轮判据看结构，不看字符串。

`_intercept_protocol_stream` 原先用 `"tool_use" in json.dumps(out)` 判「有没有 agent 自己的调用」：
正文或回程 reasoning 里出现「tool_use」字样，纯 bladex 轮就错走混合分支（剥离-拼接）。
三协议各：正文含该字样 / reasoning 含该字样 ⇒ 纯 bladex 仍走内循环；真混合轮仍走剥离-拼接
（真混合另见 `test_mixed_splice_matrix.py`，这里各留一条对照）。
"""
from __future__ import annotations

import asyncio
import json

import pytest
from bladex_proxy.agency import (
    AgencyRuntime,
    intercept_anthropic_stream,
    intercept_chat_stream,
    intercept_responses_stream,
)

BX = {"id": "call_bx1", "type": "function",
      "function": {"name": "bladex_memory_search", "arguments": '{"query":"deploy"}'}}
AGENT = {"id": "call_sh1", "type": "function",
         "function": {"name": "shell", "arguments": '{"cmd":"ls"}'}}
WORD = "the tool_use block and function_call word"
PREFIX = "u/codex/s/"


@pytest.fixture(autouse=True)
def _on(monkeypatch):
    for k in ("BLADEX_MODULE_TOOLFACE", "BLADEX_MODULE_INTERCEPTION", "BLADEX_MODULE_LEDGER"):
        monkeypatch.setenv(k, "1")
    monkeypatch.setenv("BLADEX_STREAM_KEEPALIVE_S", "0")
    monkeypatch.setenv("BLADEX_NARRATE_INTERCEPT", "0")


class _Cap:
    full_text = ""
    reasoning_text = ""
    usage: dict = {}
    finish_reason = None

    def __init__(self):
        self.tool_events: list = []


def _sse(name, obj) -> bytes:
    return (f"event: {name}\ndata: {json.dumps(obj, ensure_ascii=False)}\n\n").encode()


def _chat_sse(obj) -> str:
    return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"


def _calls(mixed: bool):
    return (BX, AGENT) if mixed else (BX,)


async def _anthropic(where: str, mixed: bool):
    yield _sse("message_start", {"type": "message_start", "message": {
        "id": "m1", "type": "message", "role": "assistant", "model": "m", "content": [],
        "stop_reason": None, "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1}}})
    i = 0
    if where == "text":
        yield _sse("content_block_start", {"type": "content_block_start", "index": i,
                   "content_block": {"type": "text", "text": ""}})
        yield _sse("content_block_delta", {"type": "content_block_delta", "index": i,
                   "delta": {"type": "text_delta", "text": WORD}})
        yield _sse("content_block_stop", {"type": "content_block_stop", "index": i})
        i += 1
    elif where == "reasoning":
        yield _sse("content_block_start", {"type": "content_block_start", "index": i,
                   "content_block": {"type": "thinking", "thinking": ""}})
        yield _sse("content_block_delta", {"type": "content_block_delta", "index": i,
                   "delta": {"type": "thinking_delta", "thinking": WORD}})
        yield _sse("content_block_stop", {"type": "content_block_stop", "index": i})
        i += 1
    for tc in _calls(mixed):
        yield _sse("content_block_start", {"type": "content_block_start", "index": i,
                   "content_block": {"type": "tool_use", "id": tc["id"],
                                     "name": tc["function"]["name"], "input": {}}})
        yield _sse("content_block_delta", {"type": "content_block_delta", "index": i,
                   "delta": {"type": "input_json_delta",
                             "partial_json": tc["function"]["arguments"]}})
        yield _sse("content_block_stop", {"type": "content_block_stop", "index": i})
        i += 1
    yield _sse("message_delta", {"type": "message_delta", "delta": {"stop_reason": "tool_use"},
                                 "usage": {"output_tokens": 2}})
    yield _sse("message_stop", {"type": "message_stop"})


async def _responses(where: str, mixed: bool):
    def resp(out, status):
        return {"id": "r1", "object": "response", "created_at": 0, "model": "m",
                "status": status, "output": out, "parallel_tool_calls": False,
                "tool_choice": "auto", "tools": []}
    items: list = []
    yield _sse("response.created", {"type": "response.created", "response": resp([], "in_progress")})
    i = 0
    if where == "text":
        yield _sse("response.output_text.delta", {
            "type": "response.output_text.delta", "item_id": "msg0", "output_index": i,
            "content_index": 0, "delta": WORD})
        i += 1
    elif where == "reasoning":
        yield _sse("response.reasoning_summary_text.delta", {
            "type": "response.reasoning_summary_text.delta", "item_id": "rs0",
            "output_index": i, "summary_index": 0, "delta": WORD})
        i += 1
    for tc in _calls(mixed):
        item = {"type": "function_call", "id": f"fc{i}", "call_id": tc["id"],
                "name": tc["function"]["name"], "arguments": tc["function"]["arguments"],
                "status": "completed"}
        items.append(item)
        yield _sse("response.output_item.added", {
            "type": "response.output_item.added", "output_index": i,
            "item": {**item, "arguments": "", "status": "in_progress"}})
        yield _sse("response.function_call_arguments.delta", {
            "type": "response.function_call_arguments.delta", "item_id": f"fc{i}",
            "output_index": i, "delta": tc["function"]["arguments"]})
        yield _sse("response.output_item.done", {
            "type": "response.output_item.done", "output_index": i, "item": item})
        i += 1
    yield _sse("response.completed", {"type": "response.completed", "response": resp(items, "completed")})


async def _chat(where: str, mixed: bool):
    base = {"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": "m"}
    if where == "text":
        yield _chat_sse({**base, "choices": [{"index": 0, "delta": {"content": WORD},
                                              "finish_reason": None}]})
    elif where == "reasoning":
        yield _chat_sse({**base, "choices": [{"index": 0, "delta": {"reasoning_content": WORD},
                                              "finish_reason": None}]})
    for i, tc in enumerate(_calls(mixed)):
        yield _chat_sse({**base, "choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": i, "id": tc["id"], "type": "function",
             "function": {"name": tc["function"]["name"],
                          "arguments": tc["function"]["arguments"]}}]},
            "finish_reason": None}]})
    yield _chat_sse({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})
    yield "data: [DONE]\n\n"


PROTOS = [
    pytest.param(intercept_anthropic_stream, _anthropic, id="messages"),
    pytest.param(intercept_responses_stream, _responses, id="responses"),
    pytest.param(intercept_chat_stream, _chat, id="chat"),
]


def _run(fn, source, where: str, mixed: bool):
    ag = AgencyRuntime()
    state = {"llm": 0}

    async def dispatch(_n, _a, **_kw):
        return "result text"
    ag.toolface.dispatch = dispatch          # type: ignore[method-assign]

    async def call_llm(_msgs):
        state["llm"] += 1
        return {"role": "assistant", "content": "done"}

    async def go():
        return [c async for c in fn(
            source(where, mixed), agency=ag, capture_result=_Cap(), session_prefix=PREFIX,
            allowed_exposure="local", session_id="s", agent_id="codex", call_llm=call_llm,
            upstream_messages=[{"role": "user", "content": "干活"}])]
    asyncio.run(go())
    return state["llm"], ag.splice.new_in_turn(PREFIX)


@pytest.mark.parametrize("fn,source", PROTOS)
@pytest.mark.parametrize("where", ["text", "reasoning"])
def test_pure_bladex_with_word_in_prose_runs_inner_loop(fn, source, where) -> None:
    llm_calls, splice = _run(fn, source, where, mixed=False)
    assert llm_calls == 1, "纯 bladex 轮必须走内循环"
    assert not splice, "纯 bladex 轮不该走剥离-拼接（不该记拼接台账）"


@pytest.mark.parametrize("fn,source", PROTOS)
def test_real_mixed_still_strips_and_splices(fn, source) -> None:
    llm_calls, splice = _run(fn, source, "none", mixed=True)
    assert llm_calls == 0, "混合轮不走内循环"
    assert splice and [c["function"]["name"] for c in splice[0].calls] == ["bladex_memory_search"]
