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

from bladex_proxy.capture import (
    CaptureResult, cache_read_tokens, extract_usage, record_first_chunk, warn_bare_reasoning,
    warn_inline_think,
)

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


# 🔴 MQ-P31（2026-09-10）：本体已搬到 `capture.py`，三条入站路径共用**单一实现点**。
# 这里保留原名别名 —— 既有测试与 monkeypatch 目标（`anthropic._cache_read_tokens`）
# 不动，且本文件内的调用点一行未改。V-C2b 仍靠这个字段判 L1 常开有没有在保 cache。
_cache_read_tokens = cache_read_tokens


def _anthropic_usage(
    prompt_tokens: int, output_tokens: int, cache_read: int | None
) -> dict[str, int]:
    """回给客户端的 `usage` 块 —— **`input_tokens` 按 Anthropic 语义 = 未缓存部分**（MQ-P27）。

    ## 同一个名字，两个端点上语义相反（09-10 实测，差点据此得出错误结论）

    | 上游 | `prompt_tokens` / `input_tokens` |
    |---|---|
    | Ark `/v3`（OpenAI 兼容，我们的上游） | **含缓存的总数**，`cached_tokens` 是它的子集 |
    | Anthropic 原生（客户端按这个读） | **未缓存部分**，`cache_read_input_tokens` 是它之外的另一份 |

    此前把上游的总数原样填进 `input_tokens` 且**不发** `cache_read_input_tokens`：
    两个错互相抵消，`input + cache_read` 的总量恰好是对的（客户端加了个 0）——
    所以 MQ-P20 的压缩阈值没被它污染（实测三跑 168,164 / 167,956 / 167,224）。
    🔴 **但这是一颗引信**：谁日后补上 `cache_read_input_tokens` 而不动 `input_tokens`，
    同一份 token 会被数两遍，上下文规模当场翻倍、压缩提前一半触发。
    **两处必须同一次改**，就是这个函数存在的理由（单一实现点）。

    **不变量（测试钉死）**：`input_tokens + cache_read_input_tokens == prompt_tokens`。
    总量不变 ⇒ 已兑现的 P20 那一格不回退。

    上游没报缓存（`cache_read is None`）⇒ 不发这个字段，行为与改动前逐字一致 ——
    「上游不报」与「报了且零命中」是两件事（同 `_cache_read_tokens` 的 None ≠ 0）。
    """
    if cache_read is None:
        return {"input_tokens": prompt_tokens, "output_tokens": output_tokens}
    # 防御：上游若把 cached 报得比总数还大（见过 usage 字段互相不自洽的上游），
    # 宁可让 input_tokens 归零也不发负数 —— 负 token 会让客户端的算术整段崩掉。
    uncached = prompt_tokens - cache_read
    if uncached < 0:
        logger.warning("anthropic_usage_cache_exceeds_prompt",
                       prompt_tokens=prompt_tokens, cache_read=cache_read)
        uncached = 0
    return {
        "input_tokens": uncached,
        "cache_read_input_tokens": cache_read,
        "output_tokens": output_tokens,
    }


async def anthropic_stream_generator(
    stream: Any,
    result: CaptureResult,
    model: str,
    *,
    input_tokens_estimate: int,
) -> AsyncIterator[bytes]:
    """迭代上游 stream，按 Anthropic SSE 格式回传 + 拼完整回复。

    事件序列：
      message_start → content_block_start/delta/stop (×N blocks) → message_delta → message_stop

    ## `input_tokens_estimate`（MQ-P20，2026-09-09）

    `message_start.usage.input_tokens` 此前**硬编码 0**，而 claude-code 只走流式
    ⇒ 它拿不到输入用量 ⇒ **永不触发上下文压缩**（实证：同一份默认配置，裸跑第 20
    分钟压缩并跑完，经 BladeX 全程零压缩、上下文涨到 39 万字符）。
    同一文件的非流式路径（`format_anthropic_response`）一直填对着 —— 同一字段、
    同一文件、两条路径一对一错（刚性原则 12）。

    🔴 **关键字参数且无默认值**：本条缺陷的形状就是"这个数悄悄是 0"，
    给个 `= 0` 的默认值等于把它原样保留给下一个忘记传的调用点
    （原则 12：同一参数在两条调用路径上各有一个默认值 = 缺陷）。
    调用方必须回答"这一轮的输入有多大"。

    上游若在 chunk 里报了真值（`usage.prompt_tokens`），**以上游为准**覆盖估计值，
    并在 `message_delta` 里回给客户端做校正；上游不报时估计值原样留着。
    `usage` 块的 cache 拆分见 `_anthropic_usage`（**MQ-P27**）。

    ## MQ-P28（2026-09-10）：工具块必须**严格串行**

    Anthropic 流式协议里内容块不允许嵌套：`start(i)` → deltas → `stop(i)` → `start(i+1)`。
    此前每来一个新的上游 tool index 就直接发 `content_block_start`、**不关上一个**，
    全部 `stop` 堆在收尾 ⇒ 多工具轮的块整个嵌套。
    实测（H4-P28，112 轮）：**多工具轮 25/112，块嵌套轮 25/112，两个集合逐轮重合**；
    单工具 85 轮结构全对。用户侧症状 = `● Update(...)` 状态点**全轮保持绿色** ——
    CC 按 `content_block_stop` 落定每个块的状态机，stop 拖到轮末它就一直认为块在流。

    ### 三个候选与为什么选第三个

    | 方案 | 问题 |
    |---|---|
    | ① 来新 index 就关上一个，全部工具都增量流式 | 上游**交错**发（OpenAI 语义允许按 index 并行）时，落在已关闭块上的 delta **无处可发** ⇒ 丢参数 ⇒ 工具 JSON 不完整。比原缺陷更糟 |
    | ② 全部工具缓冲到收尾统一发 | 永远正确，但**76% 的单工具轮**也失去增量流式 —— 为少数形态改多数路径的行为 |
    | **③ 第一个工具增量流式，其余缓冲到收尾按序发** | **选它** |

    ③ 的性质：**按构造无损**（缓冲的参数一个不丢）、**对任何到达顺序都不嵌套**
    （交错时后来的 delta 仍落在那个仍然开着的块上）、单工具轮行为**逐字不变**。
    代价：多工具轮里第二个及以后的工具块在流末才出现 —— 而它们的参数本来也是那时才收齐。

    ⚠️ 实测 `input_json_delta` 交错 **0/25**，所以 ① 在当前上游上也能work ——
    **但那是上游的偶然行为，不是协议保证**（刚性原则 10 的反面：不能把观察到的
    上游行为当契约）。③ 不依赖这个观察，所以换上游不会复发。
    """
    msg_id = f"msg_{uuid.uuid4().hex[:12]}"
    full_text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls_acc: dict[int, dict[str, Any]] = {}
    t0 = time.perf_counter()
    current_block_idx = -1
    text_block_open = False
    thinking_block_open = False
    # MQ-P28：正在增量流式的那个工具的**上游 index**（None = 还没开过工具块）。
    tool_stream_idx: int | None = None
    stop_reason = "end_turn"
    # MQ-P20：起点是本地估计值，不是 0。上游报了真值就覆盖（见下面的 chunk_usage 分支）。
    input_tokens = input_tokens_estimate
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
            # MQ-P20：此处曾是硬编码 0 —— claude-code 读的就是这一格。
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        },
    })

    try:
        async for chunk in stream:
            result.chunk_count += 1
            # MQ-P19：首字延迟。**这条路径此前没有这一行**，而 claude-code 只走它
            # ⇒ 分不开"等在首字节"与"等在吐字慢"（`ms_first_chunk` 是 A59 / MQ-A57
            # 那条"下行回传"判据的硬前置）。共用 `capture.record_first_chunk`。
            record_first_chunk(result, t0)
            try:
                choice = chunk.choices[0]
                delta = choice.delta

                # 🔴 MQ-P18/P26：结构化思考 → Anthropic `thinking` 块。
                # **CC 自己要的就是这个**：live 请求体 113/152 带
                # `thinking={"type":"adaptive"}` + `context_management.clear_thinking_*`，
                # 而我们此前把 `delta.reasoning_content` 一行不读、整段丢弃。
                # 上游默认就给它（直连实测：不带任何参数也有 144 字符 reasoning_content）。
                # 块形态照抄 `agency/streams.py::_AnthropicNarrator`——那条路径 08-25 D7
                # 已在真实 CC 上验证过（折叠成计时条）⇒ 不需要新的协议验证。
                rc = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
                if rc:
                    rc_text = "".join(str(x) for x in rc) if isinstance(rc, list) else str(rc)
                    reasoning_parts.append(rc_text)
                    # thinking 必须排在 text/tool 之前（Anthropic 块序）；正文一开就不再补发。
                    if not text_block_open and not tool_calls_acc:
                        if not thinking_block_open:
                            current_block_idx += 1
                            thinking_block_open = True
                            yield _sse_event("content_block_start", {
                                "type": "content_block_start",
                                "index": current_block_idx,
                                "content_block": {"type": "thinking", "thinking": ""},
                            })
                        yield _sse_event("content_block_delta", {
                            "type": "content_block_delta",
                            "index": current_block_idx,
                            "delta": {"type": "thinking_delta", "thinking": rc_text},
                        })

                # 文本 delta
                if delta.content:
                    if thinking_block_open:
                        yield _sse_event("content_block_stop", {
                            "type": "content_block_stop",
                            "index": current_block_idx,
                        })
                        thinking_block_open = False
                    if not text_block_open:
                        # 🔴 未观测形态的具名守卫（MQ-P28 同族）：正文在工具块开着时到达
                        # ⇒ 又会嵌套。112 轮真实流量里 0 例（上游发完 tool_calls 就
                        # `finish_reason=tool_calls` 收尾），所以**不为它写推测性修法**，
                        # 但也不让它静默 —— 真发生了要能 grep 到（原则 12/13）。
                        if tool_stream_idx is not None:
                            logger.warning("anthropic_text_after_tool_block",
                                           tool_index=tool_stream_idx,
                                           block_idx=current_block_idx)
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
                    # 纯工具回复：thinking 块开着就先关掉，再开 tool_use。
                    if thinking_block_open:
                        yield _sse_event("content_block_stop", {
                            "type": "content_block_stop",
                            "index": current_block_idx,
                        })
                        thinking_block_open = False
                    # 关闭当前文本 block（如果有）
                    if text_block_open:
                        yield _sse_event("content_block_stop", {
                            "type": "content_block_stop",
                            "index": current_block_idx,
                        })
                        text_block_open = False

                    for tc in delta.tool_calls:
                        idx = tc.index if tc.index is not None else 0
                        acc = tool_calls_acc.get(idx)
                        if acc is None:
                            acc = tool_calls_acc[idx] = {
                                "block_idx": None,      # None = 尚未发过 content_block_start
                                "id": "",
                                "name": "",
                                "arguments_parts": [],
                            }
                        if tc.id:
                            acc["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                acc["name"] = tc.function.name
                            if tc.function.arguments:
                                acc["arguments_parts"].append(tc.function.arguments)

                        # 🔴 MQ-P28：**第一个**工具走增量流式，其余缓冲到收尾按序发。
                        # 见函数 docstring「MQ-P28」一节的三个候选与取舍。
                        if tool_stream_idx is None:
                            tool_stream_idx = idx
                            current_block_idx += 1
                            acc["block_idx"] = current_block_idx
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
                        if (idx == tool_stream_idx and tc.function
                                and tc.function.arguments):
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
                    # 🔴 MQ-P22：落**上游原值**（不是映射后的 Anthropic 词）——
                    # `capture.OUTPUT_TRUNCATED_REASONS` 与 `_warn_if_output_truncated`
                    # 认的是 `length`；此前这条路径根本没有生产者，
                    # `response_meta.finish_reason` 在 claude-code 的流式轮上恒空
                    # ⇒ MQ-A34 的截断告警对 CC 结构性失明。
                    result.finish_reason = str(choice.finish_reason)
                    stop_reason = _STOP_REASON_MAP.get(choice.finish_reason, "end_turn")

                # usage（某些 chunk 带 usage）
                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage:
                    # MQ-P20：上游真值优先，但**只在它真的是个正整数时**才覆盖估计值——
                    # `getattr(obj, "prompt_tokens", 默认)` 在字段存在且为 None 时返回 None，
                    # 会把估计值抹成 None ⇒ 又变回"客户端读不到用量"。
                    _pt = getattr(chunk_usage, "prompt_tokens", None)
                    if isinstance(_pt, int) and _pt > 0:
                        input_tokens = _pt
                    _ct = getattr(chunk_usage, "completion_tokens", None)
                    if isinstance(_ct, int):
                        output_tokens = _ct
                    # MQ-P22：同上，此前这条路径不灌 `result.usage`。
                    result.usage = extract_usage(chunk_usage)
                    _cr = _cache_read_tokens(chunk_usage)
                    if _cr is not None:
                        cache_read_tokens = _cr

            except (AttributeError, IndexError, KeyError) as e:
                logger.debug("anthropic_chunk_skip", error=str(e))

        result.done = True

    except Exception as e:
        result.error = str(e)
        logger.error("anthropic_stream_interrupted", error=str(e), chunks=result.chunk_count)

    # 关闭未关闭的 thinking block（纯思考、没有正文也没有工具调用的轮次）
    if thinking_block_open:
        yield _sse_event("content_block_stop", {
            "type": "content_block_stop",
            "index": current_block_idx,
        })

    # 关闭未关闭的文本 block
    if text_block_open:
        yield _sse_event("content_block_stop", {
            "type": "content_block_stop",
            "index": current_block_idx,
        })

    # 🔴 MQ-P28：工具块**严格串行**——先关掉正在流式的那个，再把缓冲的逐个
    # start → 一次 input_json_delta → stop。此前是"全部 start 先发、全部 stop
    # 堆在这里"，产生嵌套块（实测多工具轮 25/112 全中，单工具轮 85 全对）。
    if tool_stream_idx is not None:
        yield _sse_event("content_block_stop", {
            "type": "content_block_stop",
            "index": tool_calls_acc[tool_stream_idx]["block_idx"],
        })
    for idx in sorted(tool_calls_acc.keys()):
        if idx == tool_stream_idx:
            continue
        acc = tool_calls_acc[idx]
        current_block_idx += 1
        acc["block_idx"] = current_block_idx
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
        _args = "".join(acc["arguments_parts"])
        if _args:
            yield _sse_event("content_block_delta", {
                "type": "content_block_delta",
                "index": current_block_idx,
                "delta": {"type": "input_json_delta", "partial_json": _args},
            })
        yield _sse_event("content_block_stop", {
            "type": "content_block_stop",
            "index": current_block_idx,
        })

    # 填充 result（`full_text` = 客户端看到的正文；思考另存 `reasoning_text`）
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
        # MQ-P19：与 `capture_done` 同名同义，跨三条路径可比（`elapsed_ms − ms_first_chunk`
        # 才是"吐字段"，两者处置相反 —— MQ-V23）。
        ms_first_chunk=round(result.ms_first_chunk, 2),
        # MQ-P20 的读数对：估计值 vs 上游真值。两个都记，才能判"我们报给 CC 的那个数
        # 离真值有多远"——以及上游到底报不报（相等 ⇒ 上游没报，用的是估计值）。
        input_tokens_estimate=input_tokens_estimate,
        input_tokens_reported=input_tokens,
        done=result.done,
        error=result.error,
        # MQ-CA3：上游 prompt cache 命中读数。ADR-0019 的核心论证之一是
        # 「L1 内容纯函数 → 工具循环内前缀逐字稳定 → 保上游 cache」，
        # 而这条至今**零读数**——而且 open 单元的滑动降解每轮都在改写前缀
        # （MQ-CA1），论证在 open 路径上自相矛盾。没有这个字段，
        # "L1 常开是为了保 cache" 既无法证实也无法证伪。
        # 🔴 None ≠ 0：None = 上游没报这个字段；0 = 报了且没命中。两者不可混。
        cache_read_tokens=cache_read_tokens,
        # 🔴 MQ-P29 的**第一步仪器**（09-10）：没有这一格就分不开两件事——
        # ① 上游本来就把推理放进 `content`（模型行为）；② 上游放在
        # `reasoning_content` 但**在正文之后才到**，被上面那个
        # "正文一开就不再补发" 的分支吞了（只进 `reasoning_parts`，不上线）。
        # 两种情况**屏幕上和日志里此前长得一模一样**，而修法方向相反。
        # 判读：`text_len` 大 + `thinking_len` 为 0 ⇒ ①；两者都非零 ⇒ ②。
        reasoning_len=len(result.reasoning_text),
    )
    warn_inline_think(result.full_text, "messages")
    # MQ-P29 第二步（批 L）：标记守卫对裸散文推理失明，判据换成行为（正文长 ∧ 同轮 reasoning 为 0）。
    warn_bare_reasoning(result.full_text, len(result.reasoning_text), "messages")

    # message_delta
    # MQ-P20：`message_delta` 带上 `input_tokens` 做**校正**——上游报了真值就是真值，
    # 没报就是 `message_start` 里那个估计值（两处必须一致，否则客户端两次读数打架）。
    yield _sse_event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": _anthropic_usage(input_tokens, output_tokens, cache_read_tokens),
    })

    # message_stop
    yield _sse_event("message_stop", {"type": "message_stop"})


def _sse_event(event_type: str, data: dict[str, Any]) -> bytes:
    """格式化一个 Anthropic SSE event。"""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode()
