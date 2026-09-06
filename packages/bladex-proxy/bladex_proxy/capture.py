"""流式捕获 - 自己迭代模型返回的流、边回传边拼完整（T7，ADR-0008 §6.3）。

不依赖上游 SDK 的 streaming hook。
自己迭代 chunk，一边 yield 给客户端、一边缓冲拼出完整回复。

用 CaptureResult dataclass 收集结果，不靠 setattr 副作用（ISSUE-3 修复）。

T4(ADR-0018 §4.4)：额外累积 reasoning_content（C1）、usage（C2）、finish_reason（C6）、
首字延迟 ms_first_chunk（C6）--这些事后不可回补，捕获时必须落。
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import structlog

from bladex_proxy.models import ToolEvent

logger = structlog.get_logger()

_SSE_DATA_PREFIX = "data: "
_SSE_DONE = "data: [DONE]"

# MQ-A34 ③（2026-09-03 拍板：0.1.0 只做告警，不改行为）——"上游因输出预算耗尽而截断"的
# finish_reason 等价值集合。Router（LiteLLM）把三家上游归一到 OpenAI 形态 ⇒ 主值 `length`；
# 三条入站协议各自的原生词（Anthropic `stop_reason=max_tokens` / Responses
# `incomplete_details.reason=max_output_tokens`）一并收进——直通或替身路径会原样带回，
# 少收一个就是一条看不见的截断（尺子量不到，与 MQ-A33 同族）。
OUTPUT_TRUNCATED_REASONS: frozenset[str] = frozenset({"length", "max_tokens", "max_output_tokens"})


def output_truncated(finish_reason: str | None) -> bool:
    """上游是否因输出预算耗尽而截断（`finish_reason` 可能是逗号拼接的多 choice 串）。"""
    if not finish_reason:
        return False
    return any(part.strip() in OUTPUT_TRUNCATED_REASONS for part in str(finish_reason).split(","))


@dataclass
class CaptureResult:
    """流式捕获的结果（不靠 setattr，调用方直接持有引用）。"""
    full_text: str = ""
    tool_events: list[ToolEvent] = field(default_factory=list)
    ms: float = 0.0
    chunk_count: int = 0
    done: bool = False  # 流是否正常读完
    error: str = ""    # 非空 = 中途出错/断开
    # T4(ADR-0018 §4.4 C1/C2/C6): 响应侧元数据
    reasoning_text: str = ""                   # C1: delta.reasoning_content 累积
    usage: dict = field(default_factory=dict)  # C2: {prompt, completion, total}
    finish_reason: str = ""                    # C6: length/tool_calls/stop
    ms_first_chunk: float = 0.0                # C6: 首字延迟


async def capture_stream(
    stream: Any,
    result: CaptureResult,
) -> AsyncIterator[str]:
    """迭代上游 stream，yield SSE 给客户端，同时写入 result。

    即使中途出错/断开，result 里也有已捕获的部分（供入库打 partial 标记）。
    T4: 额外累积 reasoning_content（C1）、usage（C2）、finish_reason（C6）、首字延迟（C6）。
    """
    full_text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls_acc: dict[int, dict[str, Any]] = {}
    t0 = time.perf_counter()
    saw_done = False
    first_chunk_recorded = False

    try:
        async for chunk in stream:
            result.chunk_count += 1
            if not first_chunk_recorded:
                result.ms_first_chunk = (time.perf_counter() - t0) * 1000
                first_chunk_recorded = True

            sse_line = _chunk_to_sse(chunk)
            # 检查是否是 [DONE]（避免重复 ISSUE-10）
            if _SSE_DONE in sse_line:
                saw_done = True
            yield sse_line

            # 缓冲
            try:
                choice = chunk.choices[0]
                delta = choice.delta
                if delta.content:
                    full_text_parts.append(delta.content)
                # C1: 累积思考流（GLM/DeepSeek/doubao 的 reasoning_content / reasoning）
                # ADR-0020 T4: 兼容 reasoning 字段名 + list -> str（防 "".join 失败）
                rc = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
                if rc:
                    if isinstance(rc, list):
                        rc = "".join(str(x) for x in rc)
                    reasoning_parts.append(str(rc))
                # C6: finish_reason（通常在最后一个 chunk）
                fr = getattr(choice, "finish_reason", None)
                if fr:
                    result.finish_reason = fr
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index if tc.index is not None else 0
                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {"id": "", "name": "", "arguments_parts": []}
                        acc = tool_calls_acc[idx]
                        if tc.id:
                            acc["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                acc["name"] = tc.function.name
                            if tc.function.arguments:
                                acc["arguments_parts"].append(tc.function.arguments)
            except (AttributeError, IndexError, KeyError) as e:
                logger.debug("capture_chunk_skip", error=str(e))

            # C2: usage（上游在最终 chunk 带 usage，字段在 chunk 顶层）
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                result.usage = _extract_usage(usage)

        result.done = True

    except Exception as e:
        result.error = str(e)
        logger.error("capture_interrupted", error=str(e), chunks=result.chunk_count)

    # 即使中断也填充已捕获的部分
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
            tool_call_id=acc.get("id", ""),
        ))

    logger.info(
        "capture_done",
        chunks=result.chunk_count,
        text_len=len(result.full_text),
        reasoning_len=len(result.reasoning_text),
        tool_events=len(result.tool_events),
        elapsed_ms=round(result.ms, 2),
        ms_first_chunk=round(result.ms_first_chunk, 2),
        finish_reason=result.finish_reason,
        usage=result.usage,
        done=result.done,
        error=result.error,
    )

    # 只在上游没发 [DONE] 时补（ISSUE-10 修复）
    if not saw_done:
        yield _SSE_DONE + "\n\n"


def _extract_usage(usage: Any) -> dict:
    """从上游 usage 对象提取 token 计数（C2）。

    上游 usage 可能是 pydantic 对象（model_dump）或 dict。
    取 prompt_tokens / completion_tokens / total_tokens。
    """
    try:
        if hasattr(usage, "model_dump"):
            data = usage.model_dump()
        elif isinstance(usage, dict):
            data = usage
        else:
            data = dict(usage)
        return {
            "prompt": int(data.get("prompt_tokens", 0)),
            "completion": int(data.get("completion_tokens", 0)),
            "total": int(data.get("total_tokens", 0)),
        }
    except Exception as e:
        logger.debug("capture_usage_extract_failed", error=str(e))
        return {}


def _chunk_to_sse(chunk: Any) -> str:
    """把上游 chunk 转成 OpenAI SSE 格式。原样透传（含 tool_calls）。"""
    try:
        if hasattr(chunk, "model_dump"):
            data = chunk.model_dump()
        elif hasattr(chunk, "json"):
            data = json.loads(chunk.json())
        else:
            data = {"choices": []}
        return f"{_SSE_DATA_PREFIX}{json.dumps(data)}\n\n"
    except Exception as e:
        logger.warning("chunk_to_sse_failed", error=str(e))
        return f"{_SSE_DATA_PREFIX}{{}}\n\n"
