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


# G17.0 / MQ-P19：内联思考标记。上游若把 reasoning 内联进 `delta.content`（而不是走
# 结构化的 `reasoning_content`），这两个串会原样透传给客户端 ⇒ 工具参数被污染
# （MQ-P18 在 codex 上的症状）。生产代码此前全仓 `grep "think>"` = 0 —— 一个
# 会毁掉工具调用的形态，连告警都没有。
INLINE_THINK_MARKERS: tuple[str, ...] = ("<think>", "</think>")


def inline_think_leaked(text: str) -> bool:
    """正文里是否混进了内联思考标记（纯函数，单测钉）。"""
    if not text:
        return False
    return any(marker in text for marker in INLINE_THINK_MARKERS)


def warn_inline_think(text: str, path: str) -> bool:
    """守卫：正文出现 `<think>`/`</think>` 即告警，返回是否命中（**唯一实现点**）。

    只告警、不改正文 —— 剥离与否要等 MQ-P18 的两份抓包（出站请求体 / 上游响应体）
    定了修法再动；在那之前**先让它可见**，否则下一次 `apply_patch` 全程 abort
    依旧只能靠人眼在 codex 终端里发现。
    """
    if not inline_think_leaked(text):
        return False
    logger.warning(
        "inline_think_leaked",
        path=path,
        text_len=len(text),
        markers=[m for m in INLINE_THINK_MARKERS if m in text],
    )
    return True


#: MQ-P29 的判据阈值：正文超过这个字符数、且同轮结构化 reasoning 为 0 ⇒ 可疑形态。
#: 400 是台账登记的读数口径（H4 112 轮 `text>400 ∧ thinking=0` = 11 轮 / 10%，H3 6%，裸跑 2.2%），
#: 改它 = 改判据，先改台账 MQ-P29。它不是 env 旋钮：一把尺子只有一个刻度，与 bench 读数对得上才有意义。
BARE_REASONING_MIN_TEXT = 400


def bare_reasoning_suspected(text_len: int, reasoning_len: int,
                             *, min_text: int = BARE_REASONING_MIN_TEXT) -> bool:
    """MQ-P29 判据（纯函数，单测钉）：**认行为不认标记**。

    `warn_inline_think` 按 `</think>` 标记判，而 H4 实测 112 轮里 11 轮推理以**裸散文**走了
    `content` 通道、零标记 ⇒ 那个守卫对它**结构性失明**（零告警，而推理确实到了用户屏幕上）。
    签名干净：有 thinking 的轮正文都短（60–190 字符）、正文长的轮 thinking 全是 0。
    """
    return text_len > min_text and reasoning_len == 0


def warn_bare_reasoning(text: str, reasoning_len: int, path: str) -> bool:
    """守卫：正文长且同轮 `reasoning_len == 0` 即告警 `bare_reasoning_suspected`，返回是否命中（**唯一实现点**）。

    只告警、不改正文（与 `warn_inline_think` 同一纪律）。它是**读数**，不是定论：
    上游本来就会把推理放进 `content`（裸跑 2.2%），经我们 8–10% —— 两个因素都在、权重未定，
    这条告警的用途是让两侧的比例**可持续测**，不是每响一次就去修。
    ⚠️ 已知假阳性面：不产 reasoning 的模型每条长回复都会命中 —— 按 `path` + 同轮 `model` 汇总时要分模型看。
    """
    if not bare_reasoning_suspected(len(text), reasoning_len):
        return False
    logger.warning(
        "bare_reasoning_suspected",
        path=path,
        text_len=len(text),
        reasoning_len=reasoning_len,
        min_text=BARE_REASONING_MIN_TEXT,
    )
    return True


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
    # 🔴 MQ-P25：**发起上游调用那一刻**的时钟（由端点在 `await call_model(...)` 之前写）。
    # 没有它，`ms_first_chunk` 量的是"生成器启动 → 第一个 chunk"——async generator
    # 创建时不执行函数体，等到第一次 `__anext__`，上游调用早已返回，
    # **整段 TTFB 落在测量窗口之外**。实证：全 agent p50 1.1–1.4ms，
    # 而同任务的裸跑真 TTFB p50 **1,958ms**（tap 录音），差三个数量级。
    t_upstream: float = 0.0
    # ── 批 R · R1（2026-09-18）：本轮归属与请求侧体量，由端点经 `bind_round()` 写入，
    #    三条路径的 `*_capture_done` 经 `round_fields()` 一并打出 ──
    # 🔴 归属（MQ-P29 卡登记的仪器缺口）：此前**三条**路径的 capture 行都不带
    # agent / model / session_id / path（批 M 结算稿写的"另两条路径带"是读错了——
    # 09-17 日志逐行看过，`anthropic_capture_done` 也没有），探针只能按行序就近挂到上一条
    # `route_decision`，并发两个 agent 时会错位。落进行里就不需要 join。
    agent_id: str = ""
    model: str = ""
    session_id: str = ""
    path: str = ""
    # MQ-CA13 ②：本轮请求里**工具结果**的体量（字符数 / 条数）。`probe_cache_gap.residual`
    # 只减了上一轮 output、没减工具结果，而工具结果各跑之间差 4× ⇒ 分母未受控。
    tool_result_chars: int = 0
    tool_result_n: int = 0
    # 发起上游调用的墙钟（epoch 秒）。与 `t_upstream`（perf_counter，只算差值）不同：
    # 这个要和上一轮 `t_done_epoch` 相减算 **gap_a（真空闲）**，必须是可跨行比较的绝对时钟。
    t_call_wall: float = 0.0
    # 批 U · U1.2（ADR-0033 §2.3）：本轮上游回报的 cache 命中读数（None = 没报）。
    # 由 `note_cache_reading()` 写入并登记进 `cache_state`，下一轮装配触发门读它。
    cache_read_tokens: int | None = None


def tool_result_volume(messages: list[dict]) -> tuple[int, int]:
    """请求里工具结果的 (字符数, 条数)。纯函数（MQ-CA13 ②）。

    两种载体都认：协议归一化后的 `role == "tool"` 消息（三条入站路径进 `do_inject`
    前都已归一化）；以及 Anthropic 风格残留在 user content 里的 `tool_result` block
    （`task_unit._has_tool_result_block` 同款，T0 实测 2/4251 轮）。
    content 为 list 时只数其 text 块。

    🔴 口径纪律（MQ-CA13）：报本条任何数字必须**同时给 `gap_a`（真空闲）**——本函数
    给的是分母的一部分，不是成因；成因判决表在 `task-ca12-cause-separation-20260915.md`。
    """
    chars = 0
    n = 0

    def _text(c: object) -> str:
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            return "".join(str(b.get("text") or b.get("content") or "")
                           for b in c if isinstance(b, dict))
        return ""

    for m in messages or ():
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        c = m.get("content")
        if role == "tool":
            chars += len(_text(c))
            n += 1
        elif role == "user" and isinstance(c, list):
            for b in c:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    chars += len(_text(b.get("content")))
                    n += 1
    return chars, n


def bind_round(result: CaptureResult, *, agent_id: str, model: str, session_id: str,
               path: str, messages: list[dict]) -> None:
    """端点在**发起上游调用之前**调一次：把本轮归属 + 工具结果体量 + 起跑墙钟写进 result。

    三条入站路径各调一次（与 `t_upstream` 同款接线；守卫 `test_r1_capture_round_fields.py`
    钉三处调用点都在）。`messages` 是**真正发给上游的那份**（含本轮注入与内循环拼接的
    工具结果）——量的是上游看到的分母。
    """
    result.agent_id = agent_id or ""
    result.model = model or ""
    result.session_id = session_id or ""
    result.path = path
    result.tool_result_chars, result.tool_result_n = tool_result_volume(messages)
    result.t_call_wall = time.time()


def round_fields(result: CaptureResult) -> dict[str, Any]:
    """`*_capture_done` 三条路径共用的归属/体量/时钟字段（MQ-CA5 / CA13 / P29 卡）。

    `t_call_epoch` / `t_done_epoch` 都是 `time.time()`：`probe_cache_ttl.py` 用
    本轮 `t_call_epoch − 上轮 t_done_epoch` 算 gap_a，与 `probe_cache_gap.py` 的
    "`route_calling` 行时刻 − 上轮 capture 行时刻"**同一定义**（起跑点都在 `call_model`
    之前几行内），只是时钟从日志行的秒级换成本行自带的浮点秒——不是第二把尺子。
    `t_call_wall=0` = 端点没绑（旧调用方 / 测试），两个 epoch 都报 None 而不是 0。
    """
    bound = result.t_call_wall > 0
    return {
        "agent": result.agent_id, "model": result.model,
        "session_id": result.session_id, "path": result.path,
        "tool_result_chars": result.tool_result_chars,
        "tool_result_n": result.tool_result_n,
        "t_call_epoch": round(result.t_call_wall, 3) if bound else None,
        "t_done_epoch": round(time.time(), 3) if bound else None,
    }


def note_cache_reading(result: CaptureResult, cache_read: int | None) -> None:
    """本轮 cache 命中读数的**唯一登记点**（批 U · U1.2，ADR-0033 §2.3 / MQ-CA17）。

    三条流式 capture 路径在 `*_capture_done` 之后各调一次：写进 `result`，并按
    `result.session_id`（`bind_round` 绑的）登记到 `cache_state`——下一轮的装配触发门
    （`inject._assembly_trigger`）据此判冷：上一轮 `cache_read_tokens == 0` ⇒ 冷。
    None（上游没报）也照登记：它让下一轮按"无读数"判冷，而不是沿用更早的旧值。
    守卫 `test_u12_trigger_gate.py` 钉三处调用点都在。
    """
    result.cache_read_tokens = cache_read
    if result.session_id:
        from bladex_proxy.cache_state import registry

        registry().note_cache_read(result.session_id, cache_read)


def record_first_chunk(result: CaptureResult, t0: float) -> None:
    """首字延迟落数（MQ-P19 + MQ-P25）—— **三条入站路径的唯一实现点**。

    `/v1/chat/completions` 一直有这个赋值，而 `/v1/messages`（claude-code）与
    `/v1/responses`（codex）各缺一处 ⇒ Hub 里 3982 个 turn 的
    `response_meta.ms_first_chunk` **全 0**：字段一路接进 Hub、管道全通、只是没往里灌
    （刚性原则 13 层 3 点名的形态）。**同一件事只在一个端点做对了** —— 收成一个函数，
    三条路径各调一次，别再各写一遍。

    🔴 **参照系（MQ-P25，2026-09-09 改）**：优先用 `result.t_upstream`
    （端点在发起上游调用**之前**记的），退回 `t0`（生成器启动时刻）只为兼容
    没设这个字段的调用方（测试）。两者差的正是**整段上游 TTFB**——
    量错的那一版读出 0.32ms，而真值是 2 秒量级。**退回路径要能看出来**：
    `t_upstream=0` 的轮次在 Hub 里表现为"旧口径"，不要与新口径混算。

    幂等：`ms_first_chunk == 0.0` 是"还没落"的哨兵（与 dataclass 默认值同义；
    `perf_counter` 的差值不可能真为 0），只有第一个 chunk 写得进去。
    首字延迟事后不可回补（见模块 docstring），必须在捕获时落。
    """
    if result.ms_first_chunk == 0.0:
        base = result.t_upstream or t0
        result.ms_first_chunk = (time.perf_counter() - base) * 1000


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
    # MQ-CA5：上游 prompt cache 命中读数。None = 上游没报；0 = 报了没命中（两者不可混）。
    cache_read: int | None = None

    try:
        async for chunk in stream:
            result.chunk_count += 1
            record_first_chunk(result, t0)

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
                result.usage = extract_usage(usage)
                _cr = cache_read_tokens(usage)
                if _cr is not None:
                    cache_read = _cr

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
        # 🔴 MQ-CA5（批 R · R1，2026-09-18）：此前三条路径里**只有这条**不记
        # `cache_read_tokens`——而它是 hermes / dsh / opencode 等全部 OpenAI 兼容 agent
        # 走的路，占 capture 总数的大头（09-15：110/267）。字段名与另两条路径逐字相同，
        # `probe_cache_gap` / `probe_cache_ttl` 按键名取、三协议同一把尺子。
        # None ≠ 0：上游不报就是 None（NO-DATA），不写成 0。
        cache_read_tokens=cache_read,
        # 与 `responses_capture_done` 同名同义（P31 那里是 `input_tokens_reported` /
        # `output_tokens`）；`usage` 为空 ⇒ None，不报 0。
        input_tokens_reported=(result.usage.get("prompt") if result.usage else None),
        output_tokens=(result.usage.get("completion") if result.usage else None),
        done=result.done,
        error=result.error,
        **round_fields(result),
    )
    note_cache_reading(result, cache_read)
    warn_inline_think(result.full_text, "chat_completions")
    warn_bare_reasoning(result.full_text, len(result.reasoning_text), "chat_completions")

    # 只在上游没发 [DONE] 时补（ISSUE-10 修复）
    if not saw_done:
        yield _SSE_DONE + "\n\n"


def cache_read_tokens(usage: Any) -> int | None:  # noqa: ANN401  上游 SDK 的 usage 对象
    """上游 prompt cache 命中的 token 数；上游没报则 None（MQ-CA3）。

    两家写法都认（BladeX 后面挂什么上游都可能）：
      - OpenAI 兼容：`usage.prompt_tokens_details.cached_tokens`
      - Anthropic 原生：`usage.cache_read_input_tokens`

    🔴 **None 不是 0**。None = 这个上游根本不报；0 = 报了、本轮没命中。
    把前者写成 0 会让"上游不支持"伪装成"cache 一直没命中"。

    🔴 **单一实现点（MQ-P31，2026-09-10）**：原来只在 `anthropic.py` 里，
    于是 `/v1/responses` 想记这个读数就得抄一份 —— 而"同一件事只在一条路径上
    做对了"正是本仓最高发的缺陷形态（MQ-A18 / P7 / P22 / P23 都是它）。
    搬到三条路径共用的 `capture.py`，`anthropic._cache_read_tokens` 保留为别名
    （既有测试与 monkeypatch 目标不动）。
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


def extract_usage(usage: Any) -> dict:
    """从上游 usage 对象提取 token 计数（C2）。

    🔴 **公开的**（MQ-P22，2026-09-09）：三条入站路径共用同一口径。
    此前只有 `/v1/chat/completions` 灌 `result.usage`，另两条压根没有生产者
    ⇒ `response_meta.usage` 在 claude-code / codex 上恒空。

    上游 usage 可能是 pydantic 对象（model_dump）、dict、或**只有属性的普通对象**。
    取 prompt_tokens / completion_tokens / total_tokens。

    🔴 **三种载体一视同仁**（2026-09-09 补第三种）：紧挨着的几行是
    `getattr(chunk_usage, "prompt_tokens", …)`，即那里假定属性载体，而本函数原来
    只认 `model_dump` / dict，`dict(普通对象)` 抛 TypeError ⇒ 静默返回 `{}`
    （只留一条 debug 行）。**同一个对象、相邻两行、两种载体假设**——
    `_cache_read_tokens` 的 docstring 里写着"载体差异不该在每一层各判一次"，
    这里就是那句话的第二个实例。加这一路只会多拿到数，不会少拿。
    """
    try:
        if hasattr(usage, "model_dump"):
            data = usage.model_dump()
        elif isinstance(usage, dict):
            data = usage
        else:
            try:
                data = dict(usage)
            except TypeError:
                data = {k: getattr(usage, k, None)
                        for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
        return {
            "prompt": int(data.get("prompt_tokens") or 0),
            "completion": int(data.get("completion_tokens") or 0),
            "total": int(data.get("total_tokens") or 0),
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
