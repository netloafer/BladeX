#!/usr/bin/env python3
"""V-R1 D2/D4 受控实验的 mock 上游（`agent-compat-survey-20260825.md` §5）。

把 agent **直接**指到本服务（绕开 proxy——测的是 agent 的容忍度，不是 proxy），
观察两件事：

  D2  --mode tool   ：响应里带一个 agent 不认识的工具调用 `bladex_probe_nonexistent`
                      → 看 agent 报错/重试/忽略/崩溃（= 剥离失败时的爆炸半径）
  D4  --mode delay  ：先干等 --delay 秒、期间每 --interval 秒发一次 keepalive
                      （OpenAI 流式 = SSE 注释行；Anthropic 流式 = ping 事件）
                      → 看 agent 撑不撑得过（= 内循环的时长预算上界）
  D7  --mode narrate：--delay 期间不发空 keepalive，改发 reasoning/thinking 流播报
                      （"[BladeX] searching memory… step k"）→ 看 agent 是否实时渲染、
                      是否报协议错；播报文本带指纹 BLX-NARRATE-PROBE，若后续请求里
                      出现该指纹 = 播报被 agent 存档回带（transcript 被污染，此路不通）

支持 /v1/chat/completions（流式+非流式）与 /v1/messages（流式）。
用法：
    .venv/bin/python scripts/agent_probe_upstream.py --mode tool --port 8099
    .venv/bin/python scripts/agent_probe_upstream.py --mode delay --delay 60 --port 8099

零状态、零存储，Ctrl-C 即停。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI(title="bladex-agent-probe")
ARGS: argparse.Namespace = argparse.Namespace(mode="tool", delay=60, interval=10)

_MODEL = "probe-model"
_TOOL_NAME = "bladex_probe_nonexistent"
_NARRATE_MARK = "BLX-NARRATE-PROBE"


def _narrate_line(step: int, total: int) -> str:
    return (f"[BladeX] {_NARRATE_MARK} searching memory for 'taishan beer'… "
            f"step {step}/{total}")


def _check_narrate_echo(body: dict) -> None:
    """D7 回带检测：播报指纹出现在入站请求里 = 播报进了 agent 历史（此路不通）。"""
    blob = json.dumps(body, ensure_ascii=False)
    if _NARRATE_MARK in blob:
        print("\n[probe][D7] 🔴 播报文本被 agent 存档回带（transcript 污染）\n")


@app.get("/v1/models")
async def models() -> JSONResponse:
    return JSONResponse({"object": "list",
                         "data": [{"id": _MODEL, "object": "model", "owned_by": "probe"}]})


# Hermes 对 localhost 上游先做服务器类型探测（实测 2026-08-25：依次 GET
# /api/v1/models → /api/tags → /v1/props → /props → /version，全 404 就**静默
# 回落原 provider**——请求根本不会到达）。补 llama.cpp 风格端点让识别通过，
# 之后它就按 OpenAI 兼容走 /v1/chat/completions。ollama 端点有意不补
# （补了会被当 ollama、改走 /api/chat）。

@app.get("/version")
async def version() -> JSONResponse:
    return JSONResponse({"version": "0.0.0-bladex-probe"})


@app.get("/props")
@app.get("/v1/props")
async def props() -> JSONResponse:
    return JSONResponse({"total_slots": 1, "default_generation_settings": {},
                         "model_path": _MODEL, "chat_template": ""})


# ── OpenAI /v1/chat/completions ─────────────────────────────────────────────

def _chat_nonstream_tool() -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}", "object": "chat.completion",
        "created": int(time.time()), "model": _MODEL,
        "choices": [{"index": 0, "finish_reason": "tool_calls",
                     "message": {"role": "assistant", "content": None,
                                 "tool_calls": [{"id": "call_probe1", "type": "function",
                                                 "function": {"name": _TOOL_NAME,
                                                              "arguments": "{}"}}]}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def _chat_chunk(delta: dict, finish: str | None = None) -> dict:
    return {"id": "chatcmpl-probe", "object": "chat.completion.chunk",
            "created": int(time.time()), "model": _MODEL,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


async def _chat_stream_tool():
    yield _sse(_chat_chunk({"role": "assistant"}))
    yield _sse(_chat_chunk({"tool_calls": [{"index": 0, "id": "call_probe1",
                                            "type": "function",
                                            "function": {"name": _TOOL_NAME,
                                                         "arguments": ""}}]}))
    yield _sse(_chat_chunk({"tool_calls": [{"index": 0,
                                            "function": {"arguments": "{}"}}]}))
    yield _sse(_chat_chunk({}, finish="tool_calls"))
    yield "data: [DONE]\n\n"


async def _chat_stream_delay():
    start = time.monotonic()
    while time.monotonic() - start < ARGS.delay:
        yield ": keepalive\n\n"          # SSE 注释行（OpenAI 侧 D4 形态）
        await asyncio.sleep(min(ARGS.interval, ARGS.delay))
    yield _sse(_chat_chunk({"role": "assistant"}))
    yield _sse(_chat_chunk({"content": f"probe ok after {ARGS.delay}s of comment keepalives"}))
    yield _sse(_chat_chunk({}, finish="stop"))
    yield "data: [DONE]\n\n"


async def _chat_stream_text(text: str):
    yield _sse(_chat_chunk({"role": "assistant"}))
    yield _sse(_chat_chunk({"content": text}))
    yield _sse(_chat_chunk({}, finish="stop"))
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    _check_narrate_echo(body)
    stream = bool(body.get("stream"))
    if ARGS.mode == "narrate" and stream:
        return StreamingResponse(_chat_stream_narrate(), media_type="text/event-stream")
    tool_result = _find_tool_result(body)
    if tool_result is not None and ARGS.mode == "tool":
        print(f"\n[probe][D2] agent 对 unknown tool 的回带结果 >>> {tool_result!r}\n")
        text = "probe complete — unknown-tool result received"
        if stream:
            return StreamingResponse(_chat_stream_text(text),
                                     media_type="text/event-stream")
        return JSONResponse({
            "id": "chatcmpl-probe", "object": "chat.completion",
            "created": int(time.time()), "model": _MODEL,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})
    if ARGS.mode == "tool":
        if stream:
            return StreamingResponse(_chat_stream_tool(), media_type="text/event-stream")
        return JSONResponse(_chat_nonstream_tool())
    if stream:
        return StreamingResponse(_chat_stream_delay(), media_type="text/event-stream")
    await asyncio.sleep(ARGS.delay)      # 非流式 delay：纯测客户端 HTTP 超时
    return JSONResponse({
        "id": "chatcmpl-probe", "object": "chat.completion",
        "created": int(time.time()), "model": _MODEL,
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant",
                                 "content": f"probe ok after {ARGS.delay}s silence"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})


async def _chat_stream_narrate():
    """D7（chat 协议）：reasoning_content delta 播报。不进 message.content。"""
    yield _sse(_chat_chunk({"role": "assistant"}))
    start = time.monotonic()
    total = max(1, ARGS.delay // ARGS.interval)
    step = 0
    while time.monotonic() - start < ARGS.delay:
        step += 1
        yield _sse(_chat_chunk({"reasoning_content": _narrate_line(step, total) + "\n"}))
        await asyncio.sleep(min(ARGS.interval, ARGS.delay))
    yield _sse(_chat_chunk({"content": f"probe ok after {ARGS.delay}s of narrated reasoning"}))
    yield _sse(_chat_chunk({}, finish="stop"))
    yield "data: [DONE]\n\n"


# ── Anthropic /v1/messages（流式）────────────────────────────────────────────

def _ev(name: str, obj: dict) -> str:
    return f"event: {name}\ndata: {json.dumps(obj, ensure_ascii=False)}\n\n"


async def _messages_stream_tool():
    mid = f"msg_{uuid.uuid4().hex[:12]}"
    yield _ev("message_start", {"type": "message_start", "message": {
        "id": mid, "type": "message", "role": "assistant", "model": _MODEL,
        "content": [], "stop_reason": None,
        "usage": {"input_tokens": 1, "output_tokens": 1}}})
    yield _ev("content_block_start", {"type": "content_block_start", "index": 0,
              "content_block": {"type": "tool_use", "id": "toolu_probe1",
                                "name": _TOOL_NAME, "input": {}}})
    yield _ev("content_block_delta", {"type": "content_block_delta", "index": 0,
              "delta": {"type": "input_json_delta", "partial_json": "{}"}})
    yield _ev("content_block_stop", {"type": "content_block_stop", "index": 0})
    yield _ev("message_delta", {"type": "message_delta",
              "delta": {"stop_reason": "tool_use", "stop_sequence": None},
              "usage": {"output_tokens": 1}})
    yield _ev("message_stop", {"type": "message_stop"})


async def _messages_stream_delay():
    mid = f"msg_{uuid.uuid4().hex[:12]}"
    yield _ev("message_start", {"type": "message_start", "message": {
        "id": mid, "type": "message", "role": "assistant", "model": _MODEL,
        "content": [], "stop_reason": None,
        "usage": {"input_tokens": 1, "output_tokens": 1}}})
    start = time.monotonic()
    while time.monotonic() - start < ARGS.delay:
        yield _ev("ping", {"type": "ping"})   # Anthropic 侧 D4 形态
        await asyncio.sleep(min(ARGS.interval, ARGS.delay))
    yield _ev("content_block_start", {"type": "content_block_start", "index": 0,
              "content_block": {"type": "text", "text": ""}})
    yield _ev("content_block_delta", {"type": "content_block_delta", "index": 0,
              "delta": {"type": "text_delta",
                        "text": f"probe ok after {ARGS.delay}s of ping keepalives"}})
    yield _ev("content_block_stop", {"type": "content_block_stop", "index": 0})
    yield _ev("message_delta", {"type": "message_delta",
              "delta": {"stop_reason": "end_turn", "stop_sequence": None},
              "usage": {"output_tokens": 1}})
    yield _ev("message_stop", {"type": "message_stop"})


async def _messages_stream_text(text: str):
    mid = f"msg_{uuid.uuid4().hex[:12]}"
    yield _ev("message_start", {"type": "message_start", "message": {
        "id": mid, "type": "message", "role": "assistant", "model": _MODEL,
        "content": [], "stop_reason": None,
        "usage": {"input_tokens": 1, "output_tokens": 1}}})
    yield _ev("content_block_start", {"type": "content_block_start", "index": 0,
              "content_block": {"type": "text", "text": ""}})
    yield _ev("content_block_delta", {"type": "content_block_delta", "index": 0,
              "delta": {"type": "text_delta", "text": text}})
    yield _ev("content_block_stop", {"type": "content_block_stop", "index": 0})
    yield _ev("message_delta", {"type": "message_delta",
              "delta": {"stop_reason": "end_turn", "stop_sequence": None},
              "usage": {"output_tokens": 1}})
    yield _ev("message_stop", {"type": "message_stop"})


async def _messages_stream_narrate():
    """D7（messages 协议）：thinking block 播报，随后正常 text block。"""
    mid = f"msg_{uuid.uuid4().hex[:12]}"
    yield _ev("message_start", {"type": "message_start", "message": {
        "id": mid, "type": "message", "role": "assistant", "model": _MODEL,
        "content": [], "stop_reason": None,
        "usage": {"input_tokens": 1, "output_tokens": 1}}})
    yield _ev("content_block_start", {"type": "content_block_start", "index": 0,
              "content_block": {"type": "thinking", "thinking": ""}})
    start = time.monotonic()
    total = max(1, ARGS.delay // ARGS.interval)
    step = 0
    while time.monotonic() - start < ARGS.delay:
        step += 1
        yield _ev("content_block_delta", {"type": "content_block_delta", "index": 0,
                  "delta": {"type": "thinking_delta",
                            "thinking": _narrate_line(step, total) + "\n"}})
        await asyncio.sleep(min(ARGS.interval, ARGS.delay))
    yield _ev("content_block_stop", {"type": "content_block_stop", "index": 0})
    yield _ev("content_block_start", {"type": "content_block_start", "index": 1,
              "content_block": {"type": "text", "text": ""}})
    yield _ev("content_block_delta", {"type": "content_block_delta", "index": 1,
              "delta": {"type": "text_delta",
                        "text": f"probe ok after {ARGS.delay}s of narrated thinking"}})
    yield _ev("content_block_stop", {"type": "content_block_stop", "index": 1})
    yield _ev("message_delta", {"type": "message_delta",
              "delta": {"stop_reason": "end_turn", "stop_sequence": None},
              "usage": {"output_tokens": 1}})
    yield _ev("message_stop", {"type": "message_stop"})


async def _responses_stream_narrate():
    """D7（responses 协议）：reasoning summary 事件播报（事件名按 OpenAI Responses
    规范 response.reasoning_summary_text.delta；Codex 不消费/报错本身就是 D7 读数）。"""
    yield _ev("response.created", {"type": "response.created", "response": {
        "id": "resp_probe", "object": "response", "created_at": int(time.time()),
        "model": _MODEL, "status": "in_progress", "output": []}})
    yield _ev("response.output_item.added", {
        "type": "response.output_item.added", "output_index": 0,
        "item": {"type": "reasoning", "id": "rs_probe1", "summary": []}})
    start = time.monotonic()
    total = max(1, ARGS.delay // ARGS.interval)
    step = 0
    while time.monotonic() - start < ARGS.delay:
        step += 1
        yield _ev("response.reasoning_summary_text.delta", {
            "type": "response.reasoning_summary_text.delta", "item_id": "rs_probe1",
            "output_index": 0, "summary_index": 0,
            "delta": _narrate_line(step, total) + "\n"})
        await asyncio.sleep(min(ARGS.interval, ARGS.delay))
    yield _ev("response.output_item.done", {
        "type": "response.output_item.done", "output_index": 0,
        "item": {"type": "reasoning", "id": "rs_probe1", "summary": [
            {"type": "summary_text", "text": "narrate probe"}]}})
    text = f"probe ok after {ARGS.delay}s of narrated reasoning"
    item = _msg_item(text)
    yield _ev("response.output_item.added", {
        "type": "response.output_item.added", "output_index": 1,
        "item": {**item, "status": "in_progress", "content": []}})
    yield _ev("response.output_text.delta", {
        "type": "response.output_text.delta", "item_id": "msg_probe1",
        "output_index": 1, "content_index": 0, "delta": text})
    yield _ev("response.output_item.done", {
        "type": "response.output_item.done", "output_index": 1, "item": item})
    yield _ev("response.completed", {"type": "response.completed",
                                     "response": _resp_snapshot(item)})


@app.post("/v1/messages")
async def messages(request: Request):
    body = await request.json()
    _check_narrate_echo(body)
    tool_result = _find_tool_result(body)
    if tool_result is not None and ARGS.mode == "tool":
        print(f"\n[probe][D2] agent 对 unknown tool 的回带结果 >>> {tool_result!r}\n")
        return StreamingResponse(
            _messages_stream_text("probe complete — unknown-tool result received"),
            media_type="text/event-stream")
    if ARGS.mode == "narrate":
        gen = _messages_stream_narrate()
    elif ARGS.mode == "tool":
        gen = _messages_stream_tool()
    else:
        gen = _messages_stream_delay()
    return StreamingResponse(gen, media_type="text/event-stream")


# ── OpenAI /v1/responses（Codex 原生入站；事件序对照 bladex_proxy/responses.py）──

def _resp_snapshot(item: dict, status: str = "completed") -> dict:
    return {"id": "resp_probe", "object": "response", "created_at": int(time.time()),
            "model": _MODEL, "status": status, "output": [item],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}}


_FC_ITEM = {"type": "function_call", "id": "fc_probe1", "call_id": "call_probe1",
            "name": _TOOL_NAME, "arguments": "{}", "status": "completed"}


def _msg_item(text: str) -> dict:
    return {"type": "message", "id": "msg_probe1", "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}]}


async def _responses_stream_tool():
    yield _ev("response.created", {"type": "response.created", "response": {
        "id": "resp_probe", "object": "response", "created_at": int(time.time()),
        "model": _MODEL, "status": "in_progress", "output": []}})
    yield _ev("response.output_item.added", {
        "type": "response.output_item.added", "output_index": 0,
        "item": {**_FC_ITEM, "arguments": "", "status": "in_progress"}})
    yield _ev("response.function_call_arguments.delta", {
        "type": "response.function_call_arguments.delta", "item_id": "fc_probe1",
        "output_index": 0, "delta": "{}"})
    yield _ev("response.function_call_arguments.done", {
        "type": "response.function_call_arguments.done", "item_id": "fc_probe1",
        "output_index": 0, "arguments": "{}"})
    yield _ev("response.output_item.done", {
        "type": "response.output_item.done", "output_index": 0, "item": _FC_ITEM})
    yield _ev("response.completed", {"type": "response.completed",
                                     "response": _resp_snapshot(_FC_ITEM)})


async def _responses_stream_delay():
    yield _ev("response.created", {"type": "response.created", "response": {
        "id": "resp_probe", "object": "response", "created_at": int(time.time()),
        "model": _MODEL, "status": "in_progress", "output": []}})
    start = time.monotonic()
    while time.monotonic() - start < ARGS.delay:
        yield ": keepalive\n\n"
        await asyncio.sleep(min(ARGS.interval, ARGS.delay))
    text = f"probe ok after {ARGS.delay}s of comment keepalives"
    item = _msg_item(text)
    yield _ev("response.output_item.added", {
        "type": "response.output_item.added", "output_index": 0,
        "item": {**item, "status": "in_progress",
                 "content": []}})
    yield _ev("response.content_part.added", {
        "type": "response.content_part.added", "item_id": "msg_probe1",
        "output_index": 0, "content_index": 0,
        "part": {"type": "output_text", "text": "", "annotations": []}})
    yield _ev("response.output_text.delta", {
        "type": "response.output_text.delta", "item_id": "msg_probe1",
        "output_index": 0, "content_index": 0, "delta": text})
    yield _ev("response.content_part.done", {
        "type": "response.content_part.done", "item_id": "msg_probe1",
        "output_index": 0, "content_index": 0,
        "part": {"type": "output_text", "text": text, "annotations": []}})
    yield _ev("response.output_item.done", {
        "type": "response.output_item.done", "output_index": 0, "item": item})
    yield _ev("response.completed", {"type": "response.completed",
                                     "response": _resp_snapshot(item)})


def _find_tool_result(body: dict) -> str | None:
    """请求里带回的探针工具结果（None = 首轮）。三协议各一种形态。

    找到就打印出来——**agent 对 unknown tool 回了什么错误文案**正是 D2 的另一半答案。
    找到后 mock 收口回正常文本，终止工具循环（否则 mock 每轮都回工具调用 = 无限循环，
    那是 mock 造出来的，不是 agent 行为）。
    """
    # 🔴 只认我们自己的 call id（call_probe1/toolu_probe1）——带历史的会话里
    # 第一条 tool 消息可能是 agent 以前某次真实调用的旧输出（dsh 实测踩过：
    # 抓回一条 '      87\n'，差点当成它对 unknown tool 的应答）。
    _OURS = {"call_probe1", "toolu_probe1", "fc_probe1"}
    # Responses：input 列表里的 function_call_output
    for item in (body.get("input") if isinstance(body.get("input"), list) else []) or []:
        if (isinstance(item, dict) and item.get("type") == "function_call_output"
                and item.get("call_id") in _OURS):
            return str(item.get("output", ""))
    # chat.completions：role=tool 消息
    for m in body.get("messages", []) or []:
        if isinstance(m, dict):
            if m.get("role") == "tool" and m.get("tool_call_id") in _OURS:
                return str(m.get("content", ""))
            # Anthropic：user 消息里的 tool_result block
            c = m.get("content")
            if isinstance(c, list):
                for blk in c:
                    if (isinstance(blk, dict) and blk.get("type") == "tool_result"
                            and blk.get("tool_use_id") in _OURS):
                        return json.dumps(blk.get("content", ""), ensure_ascii=False)
    return None


async def _responses_stream_text(text: str):
    item = _msg_item(text)
    yield _ev("response.created", {"type": "response.created", "response": {
        "id": "resp_probe", "object": "response", "created_at": int(time.time()),
        "model": _MODEL, "status": "in_progress", "output": []}})
    yield _ev("response.output_item.added", {
        "type": "response.output_item.added", "output_index": 0,
        "item": {**item, "status": "in_progress", "content": []}})
    yield _ev("response.output_text.delta", {
        "type": "response.output_text.delta", "item_id": "msg_probe1",
        "output_index": 0, "content_index": 0, "delta": text})
    yield _ev("response.output_item.done", {
        "type": "response.output_item.done", "output_index": 0, "item": item})
    yield _ev("response.completed", {"type": "response.completed",
                                     "response": _resp_snapshot(item)})


@app.post("/v1/responses")
async def responses(request: Request):
    body = await request.json()
    _check_narrate_echo(body)
    if ARGS.mode == "narrate":
        return StreamingResponse(_responses_stream_narrate(),
                                 media_type="text/event-stream")
    tool_result = _find_tool_result(body)
    if tool_result is not None and ARGS.mode == "tool":
        # 🔴 mode 判断不可省：delay 模式下会话历史仍带着上一轮的工具结果，
        # 不判 mode 会把 D4 短路成即时文本（2026-08-25 实测踩过）。
        print(f"\n[probe][D2] agent 对 unknown tool 的回带结果 >>> {tool_result!r}\n")
        return StreamingResponse(
            _responses_stream_text("probe complete — unknown-tool result received"),
            media_type="text/event-stream")
    stream = bool(body.get("stream", True))   # Codex 默认流式
    if not stream:
        if ARGS.mode == "delay":
            await asyncio.sleep(ARGS.delay)
            return JSONResponse(_resp_snapshot(_msg_item("probe ok (nonstream)")))
        return JSONResponse(_resp_snapshot(_FC_ITEM))
    gen = _responses_stream_tool() if ARGS.mode == "tool" else _responses_stream_delay()
    return StreamingResponse(gen, media_type="text/event-stream")


def main() -> int:
    import uvicorn
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=("tool", "delay", "narrate"), default="tool")
    ap.add_argument("--delay", type=int, default=60, help="delay 模式的干等秒数")
    ap.add_argument("--interval", type=int, default=10, help="keepalive 间隔秒数")
    ap.add_argument("--port", type=int, default=8099)
    global ARGS
    ARGS = ap.parse_args()
    print(f"[probe] mode={ARGS.mode} delay={ARGS.delay}s interval={ARGS.interval}s "
          f"port={ARGS.port} tool={_TOOL_NAME}")
    uvicorn.run(app, host="127.0.0.1", port=ARGS.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
