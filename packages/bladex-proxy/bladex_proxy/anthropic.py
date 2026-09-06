"""Anthropic /v1/messages 格式适配（T4，ADR-0008 §6.6）。

Claude Code 原生走 Anthropic /v1/messages，格式与 OpenAI /v1/chat/completions 不同：
  - system 是独立字段（不在 messages 数组里）
  - content 是 block 数组（[{type: "text", text: "..."}]）
  - SSE event 类型不同（message_start / content_block_delta / message_stop）

本模块负责：
  1. 解析 Anthropic 请求 → 归一成内部 OpenAI 格式 messages（复用全流程）
  2. 流式：自己迭代上游 stream、按 Anthropic SSE 格式回传 + 拼完整回复
  3. 非流式：把 OpenAI 格式响应转成 Anthropic 格式
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import structlog

from bladex_proxy.capture import CaptureResult

logger = structlog.get_logger()

# OpenAI finish_reason → Anthropic stop_reason 映射
_STOP_REASON_MAP: dict[str, str] = {
    "stop": "end_turn",
    "length": "max_tokens",  # 上游按输出上限截断（≠ BladeX 的 L1 证据降解，MQ-CA1）
    "tool_calls": "tool_use",
    "content_filter": "end_turn",
}


def _extract_tool_result_text(value: Any) -> str:
    """Anthropic tool_result 的 content 字段 -> 纯文本。

    content 可能是 str / list[text block] / 其他结构 -> 提取文本。
    """
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [
            b.get("text", "") for b in value
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        return "\n".join(p for p in parts if p)
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False)


def _normalize_user_content(content: list) -> list[dict[str, Any]]:
    """Anthropic user content blocks -> OpenAI 消息列表（可能多条）。

    - tool_result block -> 独立 {"role":"tool","tool_call_id":...,"content":...} 消息
      （OpenAI 要求 tool 结果是独立 tool 角色消息，不能塞进 user content；
       否则上游报 "Invalid user message" -> 502）
    - text block -> 合并为一条 user 消息（纯文本 str）
    - image 等多模态 block -> 保留在 user 消息 list content（OpenAI image_url 映射另论）
    cache_control 等 Anthropic 专有字段丢弃（OpenAI 上游不认）。
    """
    tool_msgs: list[dict[str, Any]] = []
    text_parts: list[str] = []
    other_blocks: list[dict] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        bt = block.get("type")
        if bt == "tool_result":
            tool_msgs.append({
                "role": "tool",
                "tool_call_id": block.get("tool_use_id", "") or "",
                "content": _extract_tool_result_text(block.get("content", "")),
            })
        elif bt == "text":
            text_parts.append(block.get("text", ""))
        else:
            other_blocks.append(block)
    out: list[dict[str, Any]] = list(tool_msgs)
    if other_blocks:
        blocks = list(other_blocks)
        if text_parts:
            blocks.insert(0, {"type": "text", "text": "\n".join(text_parts)})
        out.append({"role": "user", "content": blocks})
    elif text_parts:
        out.append({"role": "user", "content": "\n".join(text_parts)})
    return out


def _normalize_assistant_content(content: list) -> dict[str, Any]:
    """Anthropic assistant content blocks -> OpenAI assistant 消息。

    - tool_use block -> tool_calls 字段（input -> arguments JSON 字符串）
    - text block -> content（拼接，空则 None）
    OpenAI 要求 assistant 工具调用走 tool_calls 字段，不能塞 content block list。
    """
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        bt = block.get("type")
        if bt == "text":
            text_parts.append(block.get("text", ""))
        elif bt == "tool_use":
            input_val = block.get("input", {})
            if isinstance(input_val, str):
                args_str = input_val
            else:
                args_str = json.dumps(input_val, ensure_ascii=False)
            tool_calls.append({
                "id": block.get("id", "") or f"call_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {
                    "name": block.get("name", "") or "",
                    "arguments": args_str,
                },
            })
    msg: dict[str, Any] = {"role": "assistant"}
    joined = "\n".join(text_parts)
    msg["content"] = joined if joined else None
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def parse_anthropic_request(body: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """把 Anthropic /v1/messages 请求归一成内部 OpenAI 格式。

    返回 (messages, extra_kwargs)：
      messages: OpenAI 格式消息列表（system 字段 → 首条 system 消息）
      extra_kwargs: 透传给 Router 的额外参数（max_tokens, temperature, tools, ...）
    """
    messages: list[dict[str, Any]] = []

    # system 字段 → 首条 system 消息
    system = body.get("system")
    if system:
        if isinstance(system, str):
            messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            # content blocks → 拼成纯文本
            parts = [
                b.get("text", "") for b in system
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            if parts:
                messages.append({"role": "system", "content": "\n".join(parts)})

    # messages -> 归一 content blocks（按 OpenAI 格式）
    # tool_result/tool_use 等 Anthropic 专有 block 必须转成 OpenAI 对应结构，
    # 否则上游报 "Invalid user message"（tool_result 塞 user content）等 -> 502。
    for msg in body.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content")

        if isinstance(content, str):
            messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            if role == "user":
                messages.extend(_normalize_user_content(content))
            elif role == "assistant":
                messages.append(_normalize_assistant_content(content))
            else:
                # system 等：拼纯文本（cache_control 等 Anthropic 专有字段丢弃）
                text_parts = [
                    b.get("text", "") for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                ]
                messages.append({"role": role, "content": "\n".join(text_parts)})
        else:
            messages.append({"role": role, "content": content or ""})

    # 透传给 Router 的额外参数
    extra: dict[str, Any] = {}
    for key in ("max_tokens", "temperature", "top_p", "stop_sequences", "metadata"):
        val = body.get(key)
        if val is not None:
            # stop_sequences → stop（OpenAI 参数名）
            if key == "stop_sequences":
                extra["stop"] = val
            else:
                extra[key] = val

    # tools: Anthropic 格式 → OpenAI function calling 格式
    anthropic_tools = body.get("tools")
    if anthropic_tools:
        openai_tools = []
        for tool in anthropic_tools:
            if isinstance(tool, dict) and tool.get("name"):
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("input_schema", {"type": "object"}),
                    },
                })
        if openai_tools:
            extra["tools"] = openai_tools

    # tool_choice
    tc = body.get("tool_choice")
    if tc:
        if isinstance(tc, dict):
            if tc.get("type") == "auto":
                extra["tool_choice"] = "auto"
            elif tc.get("type") == "any":
                extra["tool_choice"] = "required"
            elif tc.get("type") == "tool" and tc.get("name"):
                extra["tool_choice"] = {
                    "type": "function",
                    "function": {"name": tc["name"]},
                }
        elif isinstance(tc, str):
            extra["tool_choice"] = tc

    return messages, extra


def format_anthropic_response(
    response: Any,
    model: str,
) -> dict[str, Any]:
    """把 OpenAI 格式非流式响应转成 Anthropic /v1/messages 格式。"""
    try:
        message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason or "stop"
    except (AttributeError, IndexError) as e:
        logger.error("anthropic_response_parse_failed", error=str(e))
        return _error_response(str(e))

    content_blocks: list[dict[str, Any]] = []

    # 文本内容
    text = getattr(message, "content", None) or ""
    if text:
        content_blocks.append({"type": "text", "text": text})

    # 工具调用
    tool_calls = getattr(message, "tool_calls", None) or []
    for tc in tool_calls:
        func = getattr(tc, "function", None)
        name = getattr(func, "name", "") if func else ""
        args_str = getattr(func, "arguments", "") if func else ""
        try:
            args = json.loads(args_str) if args_str else {}
        except json.JSONDecodeError:
            args = {}
        content_blocks.append({
            "type": "tool_use",
            "id": getattr(tc, "id", "") or f"toolu_{uuid.uuid4().hex[:12]}",
            "name": name,
            "input": args,
        })

    stop_reason = _STOP_REASON_MAP.get(finish_reason, "end_turn")

    # usage
    usage_raw = getattr(response, "usage", None)
    usage = {
        "input_tokens": getattr(usage_raw, "prompt_tokens", 0) if usage_raw else 0,
        "output_tokens": getattr(usage_raw, "completion_tokens", 0) if usage_raw else 0,
    }

    return {
        "id": f"msg_{uuid.uuid4().hex[:12]}",
        "type": "message",
        "role": "assistant",
        "content": content_blocks if content_blocks else [{"type": "text", "text": ""}],
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage,
    }


def _error_response(error: str) -> dict[str, Any]:
    return {
        "type": "error",
        "error": {"type": "internal_error", "message": error},
    }


# T26: 本地近似 token 计数（count_tokens 用）─────────────────────────────────────

# CJK / 韩日文区间（近似 1 token/字；BPE 实际 1-2，取下界近似）
_CJK_RANGES = (
    (0x4E00, 0x9FFF),    # CJK 统一表意
    (0x3000, 0x30FF),    # CJK 标点 + 假名
    (0xAC00, 0xD7AF),    # 韩文音节
    (0xFF00, 0xFFEF),    # 全角字符
)


def _is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def _text_token_estimate(s: str) -> int:
    """单段文本近似 token 数：CJK ~1 token/字，latin/digit ~4 字/token。"""
    if not s:
        return 0
    cjk = sum(1 for c in s if _is_cjk(c))
    other = len(s) - cjk
    return cjk + other // 4


def approx_count_tokens(messages: list[dict[str, Any]], tools: list[dict] | None = None) -> int:
    """本地近似 input_tokens 计数（T26，stdlib 无重依赖；标注 approximate）。

    Anthropic count_tokens 含 system + messages + tools。这里在归一化后的 messages
    上计数（parse_anthropic_request 已把 system 折进 messages[0]、content block 折成文本）。
    启发式：CJK ~1 token/字、latin ~4 字/token；每条消息 +3 开销；基础 +3。
    与真实 Anthropic 计数有偏差，仅供 Claude Code count_tokens 占位（上游常 ARK/glm 无真计数）。
    """
    total = 3  # 基础开销
    for msg in messages:
        total += 3  # 每条消息的 role/边界开销
        content = msg.get("content", "")
        if isinstance(content, str):
            total += _text_token_estimate(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    total += _text_token_estimate(part.get("text", ""))
                    # tool_result / image 等非文本块按其 text 字段（若有）计，无则 0
    if tools:
        for tool in tools:
            # 计 tool 声明 JSON（name/description/input_schema）的近似 token
            total += _text_token_estimate(json.dumps(tool, ensure_ascii=False))
    return total


def _cache_read_tokens(usage: Any) -> int | None:  # noqa: ANN401  上游 SDK 的 usage 对象
    """上游 prompt cache 命中的 token 数；上游没报则 None（MQ-CA3）。

    两家写法都认（BladeX 后面挂什么上游都可能）：
      - OpenAI 兼容：`usage.prompt_tokens_details.cached_tokens`
      - Anthropic 原生：`usage.cache_read_input_tokens`

    🔴 **None 不是 0**。None = 这个上游根本不报；0 = 报了、本轮没命中。
    把前者写成 0 会让"上游不支持"伪装成"cache 一直没命中"，
    而 V-C2b 恰恰要靠这个字段判断 L1 常开有没有在保 cache。
    """
    def _get(obj: Any, key: str) -> Any:  # noqa: ANN401
        """属性与 dict 两种载体一视同仁。

        分开写过一版，dict 形态的**嵌套** `prompt_tokens_details` 直接漏掉了
        （`getattr(dict, ...)` 恒为 None，而 dict 分支只查了顶层键）——
        单测当场抓住。载体差异不该在每一层各判一次。
        """
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    if usage is None:
        return None
    direct = _get(usage, "cache_read_input_tokens")
    if isinstance(direct, int):
        return direct
    details = _get(usage, "prompt_tokens_details")
    if details is not None:
        cached = _get(details, "cached_tokens")
        if isinstance(cached, int):
            return cached
    cached_flat = _get(usage, "cached_tokens")
    if isinstance(cached_flat, int):
        return cached_flat
    return None


async def anthropic_stream_generator(
    stream: Any,
    result: CaptureResult,
    model: str,
) -> AsyncIterator[bytes]:
    """迭代上游 stream，按 Anthropic SSE 格式回传 + 拼完整回复。

    事件序列：
      message_start → content_block_start/delta/stop (×N blocks) → message_delta → message_stop
    """
    msg_id = f"msg_{uuid.uuid4().hex[:12]}"
    full_text_parts: list[str] = []
    tool_calls_acc: dict[int, dict[str, Any]] = {}
    t0 = time.perf_counter()
    current_block_idx = -1
    text_block_open = False
    stop_reason = "end_turn"
    input_tokens = 0
    output_tokens = 0
    # MQ-CA3：初值 None 而非 0 —— "上游没报"与"报了且零命中"是两件事，
    # 混成 0 会让 cache 读数在上游不支持时看起来像"一直没命中"。
    cache_read_tokens: int | None = None

    # message_start
    yield _sse_event("message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    })

    try:
        async for chunk in stream:
            result.chunk_count += 1
            try:
                choice = chunk.choices[0]
                delta = choice.delta

                # 文本 delta
                if delta.content:
                    if not text_block_open:
                        current_block_idx += 1
                        text_block_open = True
                        yield _sse_event("content_block_start", {
                            "type": "content_block_start",
                            "index": current_block_idx,
                            "content_block": {"type": "text", "text": ""},
                        })
                    full_text_parts.append(delta.content)
                    yield _sse_event("content_block_delta", {
                        "type": "content_block_delta",
                        "index": current_block_idx,
                        "delta": {"type": "text_delta", "text": delta.content},
                    })

                # 工具调用 delta
                if delta.tool_calls:
                    # 关闭当前文本 block（如果有）
                    if text_block_open:
                        yield _sse_event("content_block_stop", {
                            "type": "content_block_stop",
                            "index": current_block_idx,
                        })
                        text_block_open = False

                    for tc in delta.tool_calls:
                        idx = tc.index if tc.index is not None else 0
                        if idx not in tool_calls_acc:
                            # 新工具调用 → content_block_start
                            current_block_idx += 1
                            tool_calls_acc[idx] = {
                                "block_idx": current_block_idx,
                                "id": "",
                                "name": "",
                                "arguments_parts": [],
                            }
                            acc = tool_calls_acc[idx]
                            if tc.id:
                                acc["id"] = tc.id
                            if tc.function and tc.function.name:
                                acc["name"] = tc.function.name
                            if tc.function and tc.function.arguments:
                                acc["arguments_parts"].append(tc.function.arguments)
                            yield _sse_event("content_block_start", {
                                "type": "content_block_start",
                                "index": current_block_idx,
                                "content_block": {
                                    "type": "tool_use",
                                    "id": acc["id"],
                                    "name": acc["name"],
                                    "input": {},
                                },
                            })
                        else:
                            acc = tool_calls_acc[idx]
                            if tc.id:
                                acc["id"] = tc.id
                            if tc.function:
                                if tc.function.name:
                                    acc["name"] = tc.function.name
                                if tc.function.arguments:
                                    acc["arguments_parts"].append(tc.function.arguments)

                        # 发 input_json_delta
                        if tc.function and tc.function.arguments:
                            yield _sse_event("content_block_delta", {
                                "type": "content_block_delta",
                                "index": acc["block_idx"],
                                "delta": {
                                    "type": "input_json_delta",
                                    "partial_json": tc.function.arguments,
                                },
                            })

                # finish_reason
                if choice.finish_reason:
                    stop_reason = _STOP_REASON_MAP.get(choice.finish_reason, "end_turn")

                # usage（某些 chunk 带 usage）
                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage:
                    input_tokens = getattr(chunk_usage, "prompt_tokens", input_tokens)
                    output_tokens = getattr(chunk_usage, "completion_tokens", output_tokens)
                    _cr = _cache_read_tokens(chunk_usage)
                    if _cr is not None:
                        cache_read_tokens = _cr

            except (AttributeError, IndexError, KeyError) as e:
                logger.debug("anthropic_chunk_skip", error=str(e))

        result.done = True

    except Exception as e:
        result.error = str(e)
        logger.error("anthropic_stream_interrupted", error=str(e), chunks=result.chunk_count)

    # 关闭未关闭的文本 block
    if text_block_open:
        yield _sse_event("content_block_stop", {
            "type": "content_block_stop",
            "index": current_block_idx,
        })

    # 关闭工具调用 blocks
    for idx in sorted(tool_calls_acc.keys()):
        acc = tool_calls_acc[idx]
        yield _sse_event("content_block_stop", {
            "type": "content_block_stop",
            "index": acc["block_idx"],
        })

    # 填充 result
    result.full_text = "".join(full_text_parts)
    result.ms = (time.perf_counter() - t0) * 1000

    for idx in sorted(tool_calls_acc.keys()):
        acc = tool_calls_acc[idx]
        args_str = "".join(acc["arguments_parts"])
        try:
            args = json.loads(args_str) if args_str else None
        except json.JSONDecodeError:
            args = args_str
        from bladex_proxy.models import ToolEvent
        result.tool_events.append(ToolEvent(
            tool_name=acc["name"], arguments=args, direction="call",
        ))

    logger.info(
        "anthropic_capture_done",
        chunks=result.chunk_count,
        text_len=len(result.full_text),
        tool_events=len(result.tool_events),
        elapsed_ms=round(result.ms, 2),
        done=result.done,
        error=result.error,
        # MQ-CA3：上游 prompt cache 命中读数。ADR-0019 的核心论证之一是
        # 「L1 内容纯函数 → 工具循环内前缀逐字稳定 → 保上游 cache」，
        # 而这条至今**零读数**——而且 open 单元的滑动降解每轮都在改写前缀
        # （MQ-CA1），论证在 open 路径上自相矛盾。没有这个字段，
        # "L1 常开是为了保 cache" 既无法证实也无法证伪。
        # 🔴 None ≠ 0：None = 上游没报这个字段；0 = 报了且没命中。两者不可混。
        cache_read_tokens=cache_read_tokens,
    )

    # message_delta
    yield _sse_event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    })

    # message_stop
    yield _sse_event("message_stop", {"type": "message_stop"})


def _sse_event(event_type: str, data: dict[str, Any]) -> bytes:
    """格式化一个 Anthropic SSE event。"""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode()
