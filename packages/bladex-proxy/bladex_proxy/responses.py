"""OpenAI Responses API (/v1/responses) 格式适配（ADR-0023）。

Codex CLI 原生走 OpenAI Responses API，格式与 /v1/chat/completions 不同：
  - instructions 是顶层字段（不在 input 数组里）
  - input 是 string 或 item 数组（message / function_call / function_call_output）
  - content block 类型用 input_text / output_text（非 text）
  - tools 是扁平结构（{type:"function", name, parameters}，非嵌套 function:{}）
  - SSE event 类型不同（response.created / response.output_text.delta / response.completed）

本模块负责：
  1. 解析 Responses 请求 -> 归一成内部 OpenAI chat 格式 messages（复用全流程）
  2. 流式：自己迭代上游 stream、按 Responses SSE 格式回传 + 拼完整回复
  3. 非流式：把 OpenAI chat 格式响应转成 Responses 格式

无状态假设（ADR-0023 §2.3）：Codex 每轮发完整 input，BladeX 当无状态 proxy 处理。
previous_response_id 非空由 server handler 拦截回 400（T6），不进本模块。
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import structlog

from bladex_proxy.capture import (
    CaptureResult, cache_read_tokens, extract_usage, record_first_chunk, warn_bare_reasoning,
    warn_inline_think,
)
from bladex_proxy.models import ToolEvent

logger = structlog.get_logger()

# OpenAI chat finish_reason -> Responses status 映射
_STATUS_MAP: dict[str, str] = {
    "stop": "completed",
    "length": "incomplete",
    "tool_calls": "completed",
    "content_filter": "incomplete",
}


def _extract_content_text(content: Any) -> str:
    """Responses input item 的 content 字段 -> 纯文本。

    content 可能是 str / list[input_text|output_text block] / 其他结构 -> 提取文本。
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for b in content:
            if isinstance(b, dict):
                # input_text / output_text 都取 text 字段
                if b.get("type") in ("input_text", "output_text", "text"):
                    parts.append(b.get("text", ""))
        return "\n".join(p for p in parts if p)
    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False)


def _normalize_input_item(item: dict[str, Any]) -> list[dict[str, Any]]:
    """单个 Responses input item -> OpenAI 消息列表（可能多条）。

    - message item (role=user): content blocks -> 可能拆出 tool 消息 + user 消息
      （function_call_output block -> 独立 tool 角色消息，同 Anthropic tool_result）
    - message item (role=assistant): content blocks -> assistant 消息（含 tool_calls）
      （function_call block -> tool_calls 字段）
    - 顶层 function_call item -> assistant 消息的 tool_calls
    - 顶层 function_call_output item -> 独立 tool 消息
    """
    itype = item.get("type")

    # 顶层 function_call item（不在 message 里，独立 item）
    if itype == "function_call":
        return [{
            "role": "assistant",
            # 🔴 MQ-P17（2026-09-08，Codex 首次真实流量当场撞到）：**空串不是 None**。
            # OpenAI 官方接受 assistant + tool_calls 且 `content: null`；**Ark 的 OpenAI
            # 兼容端点不接受**，报 `missing messages.content parameter`。live 读数：
            # `/v1/responses` 63 次里 200 只有 10 次，且那 10 次**全在 Codex 第一次工具
            # 调用之前** —— 第一个带 `function_call` 的回合起 53 次全 502，两边同一个上游
            # 模型 ⇒ 差异只能来自消息内容。
            # 不改成"删掉 content 键"：那是另一种形态的缺数，且 OpenAI 侧对
            # assistant+tool_calls 要求键存在。
            "content": "",
            "tool_calls": [{
                "id": item.get("call_id", "") or f"call_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {
                    "name": item.get("name", "") or "",
                    "arguments": item.get("arguments", "") or "",
                },
            }],
        }]

    # 顶层 function_call_output item -> 独立 tool 消息
    if itype == "function_call_output":
        return [{
            "role": "tool",
            "tool_call_id": item.get("call_id", "") or "",
            "content": _extract_content_text(item.get("output")),
        }]

    # message item
    if itype == "message":
        role = item.get("role", "user")
        content = item.get("content")
        if isinstance(content, str):
            return [{"role": role, "content": content}]
        if isinstance(content, list):
            if role == "user":
                return _normalize_user_message_content(content)
            if role == "assistant":
                return [_normalize_assistant_message_content(content)]
            # system 等：拼纯文本
            return [{"role": role, "content": _extract_content_text(content)}]
        return [{"role": role, "content": content or ""}]

    # 未知 item 类型：warning 跳过（不阻断，让请求继续）
    logger.warning("responses_unknown_input_item_type", item_type=itype, item_keys=list(item.keys()))
    return []


def _normalize_user_message_content(content: list) -> list[dict[str, Any]]:
    """user message 的 content blocks -> OpenAI 消息列表。

    - input_text block -> 合并为一条 user 消息（纯文本 str）
    - function_call_output block（Codex 偶尔塞在 user message content 里）-> 独立 tool 消息
    - image 等多模态 block -> 保留在 user 消息 list content
    """
    tool_msgs: list[dict[str, Any]] = []
    text_parts: list[str] = []
    other_blocks: list[dict] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        bt = block.get("type")
        if bt == "function_call_output":
            tool_msgs.append({
                "role": "tool",
                "tool_call_id": block.get("call_id", "") or "",
                "content": _extract_content_text(block.get("output")),
            })
        elif bt in ("input_text", "output_text", "text"):
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


def _normalize_assistant_message_content(content: list) -> dict[str, Any]:
    """assistant message 的 content blocks -> OpenAI assistant 消息。

    - output_text block -> content（拼接，空则**空串**——MQ-P17，不是 None）
    - function_call block（塞在 assistant content 里）-> tool_calls 字段
    """
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        bt = block.get("type")
        if bt in ("output_text", "input_text", "text"):
            text_parts.append(block.get("text", ""))
        elif bt == "function_call":
            tool_calls.append({
                "id": block.get("call_id", "") or f"call_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {
                    "name": block.get("name", "") or "",
                    "arguments": block.get("arguments", "") or "",
                },
            })
    msg: dict[str, Any] = {"role": "assistant"}
    # MQ-P17 同族第二处：正文为空时同样给 **空串**而不是 None（理由见上）。
    msg["content"] = "\n".join(text_parts)
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def parse_responses_request(body: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """把 OpenAI /v1/responses 请求归一成内部 OpenAI chat 格式。

    返回 (messages, extra_kwargs)：
      messages: OpenAI chat 格式消息列表（instructions -> 首条 system 消息）
      extra_kwargs: 透传给 Router 的额外参数（max_tokens, temperature, tools, ...）

    无状态假设：不处理 previous_response_id（由 handler 拦截）。
    """
    messages: list[dict[str, Any]] = []

    # instructions（顶层）-> 首条 system 消息（同 Anthropic system 字段处理）
    instructions = body.get("instructions")
    if instructions:
        if isinstance(instructions, str):
            messages.append({"role": "system", "content": instructions})
        elif isinstance(instructions, list):
            text = _extract_content_text(instructions)
            if text:
                messages.append({"role": "system", "content": text})

    # input: string -> 单条 user 消息；list -> 逐项归一
    input_field = body.get("input")
    if isinstance(input_field, str):
        messages.append({"role": "user", "content": input_field})
    elif isinstance(input_field, list):
        for item in input_field:
            if isinstance(item, dict):
                messages.extend(_normalize_input_item(item))
            elif isinstance(item, str):
                # input 数组里也可能直接放纯文本（OpenAI 允许）
                messages.append({"role": "user", "content": item})

    # 透传给 Router 的额外参数
    extra: dict[str, Any] = {}
    # max_output_tokens -> max_tokens（OpenAI chat 参数名）
    if body.get("max_output_tokens") is not None:
        extra["max_tokens"] = body["max_output_tokens"]
    for key in ("temperature", "top_p", "metadata"):
        val = body.get(key)
        if val is not None:
            extra[key] = val
    # parallel_tool_calls / strict 等 tool 语义参数透传
    if body.get("parallel_tool_calls") is not None:
        extra["parallel_tool_calls"] = body["parallel_tool_calls"]

    # tools: Responses 扁平结构 -> OpenAI 嵌套 function calling 结构
    # Responses: [{type:"function", name, description, parameters}]
    # OpenAI:    [{type:"function", function:{name, description, parameters}}]
    responses_tools = body.get("tools")
    if responses_tools:
        openai_tools = []
        for tool in responses_tools:
            if not isinstance(tool, dict):
                continue
            # 已经是嵌套结构（兼容客户端直接发 OpenAI 格式）
            if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
                openai_tools.append(tool)
                continue
            if tool.get("name") or tool.get("type") == "function":
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.get("name", "") or "",
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {"type": "object"}),
                        **({"strict": tool["strict"]} if "strict" in tool else {}),
                    },
                })
        if openai_tools:
            extra["tools"] = openai_tools

    # tool_choice: Responses 与 OpenAI chat 基本一致（auto/required/{type:function,function:{name}})
    tc = body.get("tool_choice")
    if tc is not None:
        extra["tool_choice"] = tc

    # reasoning.effort -> 顶层 `reasoning_effort`（ADR-0023 §2.2：尽力而为，不阻塞 MVP）。
    # 🔴 MQ-P18 读数（2026-09-09，`scripts/probe_reasoning_forwarding.py` + Hub 155 份 codex 请求体）：
    #   入站 `reasoning.effort` 155/155 都在 ⇒ 这个条件**每轮命中**、下面这行每轮都设上；
    #   出站 `router_sdk.py` 的 `litellm.drop_params=True` 对 ARK 这一族（supported_params 无它）
    #   **每轮都把它丢掉**，`extra_body` 为 `{}`（没藏在里面）⇒ **看着在转发、对 ARK 其实不出站**。
    # 处置（台账「出站那一格已取到」三条）：不动条件、不删这段——它对支持该参数的上游（o 系 / gpt-5 系）
    # 仍有效；不走 extra_body 绕过 drop_params（开它的原因就是 ARK 对此参数回 502）；
    # P18 的修法在响应侧（`reasoning_content` → thinking 块已落，MQ-P26；裸散文泄漏由 MQ-P29 守卫计数）。
    reasoning = body.get("reasoning")
    if isinstance(reasoning, dict) and reasoning.get("effort"):
        extra["reasoning_effort"] = reasoning["effort"]

    return messages, extra


def format_responses_response(
    response: Any,
    model: str,
) -> dict[str, Any]:
    """把 OpenAI chat 格式非流式响应转成 Responses /v1/responses 格式。"""
    try:
        message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason or "stop"
    except (AttributeError, IndexError) as e:
        logger.error("responses_response_parse_failed", error=str(e))
        return _error_response(str(e))

    output: list[dict[str, Any]] = []
    output_text_parts: list[str] = []

    # 文本内容 -> message item + output_text
    text = getattr(message, "content", None) or ""
    if text:
        output_text_parts.append(text)
        output.append({
            "type": "message",
            "id": f"msg_{uuid.uuid4().hex[:24]}",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text}],
        })

    # reasoning_content（glm/ARK 等返回；T7：累积但暂不作为独立 output item，
    # 待 T1 确认 Codex 消费 reasoning item 后再补）
    rc = getattr(message, "reasoning_content", None) or getattr(message, "reasoning_text", None)
    if rc:
        # 不加入 output（避免 Codex 因未预期的 reasoning item 报错），但保留在 metadata 供审计
        pass

    # 工具调用 -> function_call items
    tool_calls = getattr(message, "tool_calls", None) or []
    for tc in tool_calls:
        func = getattr(tc, "function", None)
        name = getattr(func, "name", "") if func else ""
        args_str = getattr(func, "arguments", "") if func else ""
        output.append({
            "type": "function_call",
            "id": f"fc_{uuid.uuid4().hex[:24]}",
            "call_id": getattr(tc, "id", "") or f"call_{uuid.uuid4().hex[:12]}",
            "name": name,
            "arguments": args_str,
            "status": "completed",
        })

    status = _STATUS_MAP.get(finish_reason, "completed")

    # usage
    usage_raw = getattr(response, "usage", None)
    usage = {
        "input_tokens": getattr(usage_raw, "prompt_tokens", 0) if usage_raw else 0,
        "output_tokens": getattr(usage_raw, "completion_tokens", 0) if usage_raw else 0,
        "total_tokens": getattr(usage_raw, "total_tokens", 0) if usage_raw else 0,
    }

    resp_id = f"resp_{uuid.uuid4().hex[:24]}"
    return {
        "id": resp_id,
        "object": "response",
        "created_at": int(time.time()),
        "model": model,
        "status": status,
        "output": output,
        "output_text": "\n".join(output_text_parts),
        "usage": usage,
        # 无状态 proxy：不返 previous_response_id 关联字段
        "metadata": {},
    }


def tool_events_from_output(output: list[dict[str, Any]]) -> list[ToolEvent]:
    """从 Responses output items 提取 call 方向 ToolEvent（非流式入库用）。

    与流式路径（responses_stream_generator）保持一致：
      - arguments 解析成 dict（非流式 format 里是 JSON 字符串，需还原）
      - 带 tool_call_id（供跨轮 result 匹配，Bug A 同源修复延伸到非流式路径）
    """
    events: list[ToolEvent] = []
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "function_call":
            continue
        raw_args = item.get("arguments")
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) and raw_args else raw_args
        except json.JSONDecodeError:
            args = raw_args
        events.append(ToolEvent(
            tool_name=item.get("name", "") or "",
            arguments=args,
            direction="call",
            tool_call_id=item.get("call_id", "") or "",
        ))
    return events


def _error_response(error: str) -> dict[str, Any]:
    return {
        "id": f"resp_{uuid.uuid4().hex[:24]}",
        "object": "response",
        "created_at": int(time.time()),
        "status": "failed",
        "error": {"code": "internal_error", "message": error},
        "output": [],
        "output_text": "",
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        "metadata": {},
    }


async def responses_stream_generator(
    stream: Any,
    result: CaptureResult,
    model: str,
) -> AsyncIterator[bytes]:
    """迭代上游 stream，按 Responses SSE 格式回传 + 拼完整回复。

    事件序列（OpenAI Responses streaming）：
      response.created
      -> response.output_item.added (message)
      -> response.content_part.added (output_text)
      -> response.output_text.delta (×N)
      -> response.content_part.done
      -> response.output_item.done (message)
      -> [function_call: response.output_item.added/done + response.function_call_arguments.delta]
      -> response.completed
    """
    resp_id = f"resp_{uuid.uuid4().hex[:24]}"
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    full_text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls_acc: dict[int, dict[str, Any]] = {}
    t0 = time.perf_counter()
    text_part_open = False
    message_item_added = False
    # output_index 单调计数：message item（若有正文）占 0，工具项顺序递增。
    # 纯工具回复（无正文，agentic coding 常见）时 message 从不出现，
    # 首个 function_call 拿 index 0，不再留空洞（与 response.completed 快照一致）。
    next_output_idx = 0
    status = "completed"
    input_tokens = 0
    output_tokens = 0
    # 🔴 MQ-P31：None ≠ 0（同 `capture.cache_read_tokens` 的口径）。
    # 上游不报是 None，报了且没命中是 0；混成 0 会让"这条路径测不了缓存"
    # 伪装成"缓存一直没命中"。
    cache_read: int | None = None
    created_at = int(time.time())

    # response.created
    yield _sse_event("response.created", {
        "type": "response.created",
        "response": {
            "id": resp_id,
            "object": "response",
            "created_at": created_at,
            "model": model,
            "status": "in_progress",
            "output": [],
        },
    })

    try:
        async for chunk in stream:
            result.chunk_count += 1
            # MQ-P19：首字延迟。**这条路径此前没有这一行**，而 codex 只走它。
            # 共用 `capture.record_first_chunk`（三条入站路径唯一实现点）。
            record_first_chunk(result, t0)
            try:
                choice = chunk.choices[0]
                delta = choice.delta

                # reasoning_content 累积（ADR-0023 T7：请求侧 reasoning_effort 已透传；
                # 响应侧 reasoning item 待 T1 确认 Codex 消费后再发事件，
                # 此处先累积到 result 供入库/Memory Index 提炼，不阻塞 MVP）
                rc = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
                if rc:
                    reasoning_parts.append(str(rc))

                # 文本 delta
                if delta.content:
                    if not message_item_added:
                        yield _sse_event("response.output_item.added", {
                            "type": "response.output_item.added",
                            "output_index": 0,
                            "item": {
                                "type": "message",
                                "id": msg_id,
                                "status": "in_progress",
                                "role": "assistant",
                                "content": [],
                            },
                        })
                        message_item_added = True
                        # message 保留 output_index 0（幂等，不重复占位）
                        if next_output_idx == 0:
                            next_output_idx = 1
                    if not text_part_open:
                        yield _sse_event("response.content_part.added", {
                            "type": "response.content_part.added",
                            "item_id": msg_id,
                            "output_index": 0,
                            "content_index": 0,
                            "part": {"type": "output_text", "text": ""},
                        })
                        text_part_open = True
                    full_text_parts.append(delta.content)
                    yield _sse_event("response.output_text.delta", {
                        "type": "response.output_text.delta",
                        "item_id": msg_id,
                        "output_index": 0,
                        "content_index": 0,
                        "delta": delta.content,
                    })

                # 工具调用 delta
                if delta.tool_calls:
                    # 关闭当前文本 part / message item（如果有）
                    if text_part_open:
                        yield _sse_event("response.content_part.done", {
                            "type": "response.content_part.done",
                            "item_id": msg_id,
                            "output_index": 0,
                            "content_index": 0,
                            "part": {"type": "output_text", "text": "".join(full_text_parts)},
                        })
                        text_part_open = False
                    if message_item_added:
                        yield _sse_event("response.output_item.done", {
                            "type": "response.output_item.done",
                            "output_index": 0,
                            "item": {
                                "type": "message",
                                "id": msg_id,
                                "status": "completed",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": "".join(full_text_parts)}] if full_text_parts else [],
                            },
                        })
                        message_item_added = False

                    for tc in delta.tool_calls:
                        idx = tc.index if tc.index is not None else 0
                        if idx not in tool_calls_acc:
                            # 单调计数：有正文时 message 已占 0（next=1）；纯工具时从 0 起
                            current_output_idx = next_output_idx
                            next_output_idx += 1
                            fc_id = f"fc_{uuid.uuid4().hex[:24]}"
                            tool_calls_acc[idx] = {
                                "output_idx": current_output_idx,
                                "fc_id": fc_id,
                                "call_id": "",
                                "name": "",
                                "arguments_parts": [],
                            }
                            acc = tool_calls_acc[idx]
                            if tc.id:
                                acc["call_id"] = tc.id
                            if tc.function and tc.function.name:
                                acc["name"] = tc.function.name
                            yield _sse_event("response.output_item.added", {
                                "type": "response.output_item.added",
                                "output_index": current_output_idx,
                                "item": {
                                    "type": "function_call",
                                    "id": fc_id,
                                    "call_id": acc["call_id"],
                                    "name": acc["name"],
                                    "arguments": "",
                                    "status": "in_progress",
                                },
                            })
                        else:
                            acc = tool_calls_acc[idx]
                            if tc.id:
                                acc["call_id"] = tc.id
                            if tc.function:
                                if tc.function.name:
                                    acc["name"] = tc.function.name

                        # 发 function_call_arguments.delta
                        if tc.function and tc.function.arguments:
                            acc["arguments_parts"].append(tc.function.arguments)
                            yield _sse_event("response.function_call_arguments.delta", {
                                "type": "response.function_call_arguments.delta",
                                "item_id": acc["fc_id"],
                                "output_index": acc["output_idx"],
                                "delta": tc.function.arguments,
                            })

                # finish_reason
                if choice.finish_reason:
                    # 🔴 MQ-P22：落上游原值（不是映射后的 Responses status）。
                    # 此前无生产者 ⇒ codex 的流式轮 `finish_reason` 恒空
                    # ⇒ MQ-A34 截断告警对 codex 结构性失明（实测 393 轮里 7% 有值，
                    # 那 7% 全是非流式轮）。
                    result.finish_reason = str(choice.finish_reason)
                    status = _STATUS_MAP.get(choice.finish_reason, "completed")

                # usage（某些 chunk 带 usage）
                # 🔴 MQ-P21：`getattr(obj, k, 默认)` 在**字段存在且为 None** 时返回 None
                # （上游报一个空 usage 是常见形态）⇒ 这两个变量会变成 None ⇒
                # 收尾处 `input_tokens + output_tokens` **TypeError**，而那一行在
                # try 之外、在 `response.completed` 之前 ⇒ codex 收不到终止事件。
                # 与 anthropic.py 同一处修法（同一件事，两条路径都要做对）。
                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage:
                    _pt = getattr(chunk_usage, "prompt_tokens", None)
                    if isinstance(_pt, int):
                        input_tokens = _pt
                    _ct = getattr(chunk_usage, "completion_tokens", None)
                    if isinstance(_ct, int):
                        output_tokens = _ct
                    # MQ-P22：同上，此前这条路径不灌 `result.usage`。
                    result.usage = extract_usage(chunk_usage)
                    # MQ-P31：与 anthropic 路径同名同义、同一实现点。
                    _cr = cache_read_tokens(chunk_usage)
                    if _cr is not None:
                        cache_read = _cr

            except (AttributeError, IndexError, KeyError) as e:
                logger.debug("responses_chunk_skip", error=str(e))

        result.done = True

    except Exception as e:
        result.error = str(e)
        logger.error("responses_stream_interrupted", error=str(e), chunks=result.chunk_count)

    # 收尾：关闭未关闭的文本 part + message item
    if text_part_open:
        yield _sse_event("response.content_part.done", {
            "type": "response.content_part.done",
            "item_id": msg_id,
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": "".join(full_text_parts)},
        })
    if message_item_added:
        yield _sse_event("response.output_item.done", {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "message",
                "id": msg_id,
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "".join(full_text_parts)}] if full_text_parts else [],
            },
        })

    # 关闭工具调用 items
    for idx in sorted(tool_calls_acc.keys()):
        acc = tool_calls_acc[idx]
        yield _sse_event("response.output_item.done", {
            "type": "response.output_item.done",
            "output_index": acc["output_idx"],
            "item": {
                "type": "function_call",
                "id": acc["fc_id"],
                "call_id": acc["call_id"],
                "name": acc["name"],
                "arguments": "".join(acc["arguments_parts"]),
                "status": "completed",
            },
        })

    # 填充 result
    result.full_text = "".join(full_text_parts)
    result.reasoning_text = "".join(reasoning_parts)
    result.ms = (time.perf_counter() - t0) * 1000

    for idx in sorted(tool_calls_acc.keys()):
        acc = tool_calls_acc[idx]
        args_str = "".join(acc["arguments_parts"])
        try:
            args = json.loads(args_str) if args_str else None
        except json.JSONDecodeError:
            args = args_str
        result.tool_events.append(ToolEvent(
            tool_name=acc["name"], arguments=args, direction="call",
            tool_call_id=acc["call_id"],
        ))

    logger.info(
        "responses_capture_done",
        chunks=result.chunk_count,
        text_len=len(result.full_text),
        tool_events=len(result.tool_events),
        elapsed_ms=round(result.ms, 2),
        # MQ-P19：与另两条路径同名同义（见 `capture.record_first_chunk`）。
        ms_first_chunk=round(result.ms_first_chunk, 2),
        # MQ-P18：结构化思考流走没走上来。`reasoning_len=0` 且下一行告警命中
        # ⇒ 上游把 reasoning **内联进了正文**，修法在响应侧剥离而不在请求侧转发。
        reasoning_len=len(result.reasoning_text),
        # 🔴 MQ-P31（2026-09-10 立，codex 双侧对照驱动）：这三格此前**一个都没有**，
        # 而 `capture_done`（chat）记 `usage`、`anthropic_capture_done` 记
        # `input_tokens_reported` / `cache_read_tokens` ⇒ **同一个读数三条路径三种拼法，
        # 其中一条压根没有**。代价：codex 经 BladeX 的 178 轮里 token 用量、上下文峰值、
        # 缓存命中率**全部不可测**，于是"BladeX 是否把上下文推过了 codex 的压缩阈值"
        # ——当前最大的开放问题——无法判定（裸跑侧 tap 有 usage，两侧比不了）。
        # ⚠️ 本批**只加不改**：另两条路径的字段名一个不动。改名会把正在跑的 bench
        #    序列断掉（尺子与被测对象不许分两次动）；三路字段统一归后续卡。
        # 🔴 `usage` 与这两格并列是故意的：`result.usage` 为空而 `input_tokens_reported`
        #    非零 ⇒ 是 `extract_usage` 的载体判断漏了，不是上游没报。两者分辨得开。
        usage=result.usage,
        input_tokens_reported=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read,
        done=result.done,
        error=result.error,
    )
    # G17.0 守卫（MQ-P18）：codex 侧 `apply_patch` 持续 abort 的直接症状就是
    # `</think>` 漏进正文，而生产代码此前对它零处置、零告警。
    # 🔴 **两处都查**：agentic coding 里纯工具回复很常见（`full_text` 为空），
    # 而真正会让 `apply_patch` abort 的是**工具参数**被污染 —— 只查正文会漏掉
    # 恰好是最贵的那一种（"同一件事只在一个地方做对了"的又一形态）。
    warn_inline_think(result.full_text, "responses")
    warn_inline_think(
        "".join("".join(acc["arguments_parts"]) for acc in tool_calls_acc.values()),
        "responses:tool_args",
    )
    # MQ-P29 第二步（批 L）：认行为不认标记 —— 正文长 ∧ 同轮 reasoning 为 0。
    warn_bare_reasoning(result.full_text, len(result.reasoning_text), "responses")

    # 构建最终 output（用于 response.completed）
    final_output: list[dict[str, Any]] = []
    if full_text_parts:
        final_output.append({
            "type": "message",
            "id": msg_id,
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "".join(full_text_parts)}],
        })
    for idx in sorted(tool_calls_acc.keys()):
        acc = tool_calls_acc[idx]
        final_output.append({
            "type": "function_call",
            "id": acc["fc_id"],
            "call_id": acc["call_id"],
            "name": acc["name"],
            "arguments": "".join(acc["arguments_parts"]),
            "status": "completed",
        })

    # response.completed
    yield _sse_event("response.completed", {
        "type": "response.completed",
        "response": {
            "id": resp_id,
            "object": "response",
            "created_at": created_at,
            "model": model,
            "status": status,
            "output": final_output,
            "output_text": "".join(full_text_parts),
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
            "metadata": {},
        },
    })


def _sse_event(event_type: str, data: dict[str, Any]) -> bytes:
    """格式化一个 Responses SSE event。"""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode()
