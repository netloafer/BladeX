"""内循环驱动 —— 纯 bladex 调用的执行-回喂-循环（ADR-0032 §3.2；批二卡 V-P3）。

依赖全部注入（`call_llm` / `dispatch` / `on_progress` / 时钟），零 litellm import
——Router 网关纪律由调用方（server 接线层）持有，本模块可在沙盒全测。

# 预算（D4 实测标定，survey §5d）

- 墙钟 `BLADEX_INNER_LOOP_BUDGET_S`（默认 120s = 五 agent 全体无感区间）：
  预算内才发起下一次 LLM 往返；超了走降级。
- 轮数 `BLADEX_INNER_LOOP_MAX_ROUNDS`（默认 6）：与墙钟双保险。
- **降级 = 合成可转发响应**：把已有的工具结果以文本形态拼成 assistant 消息返回
  ——绝不把空响应或悬空 tool_call 发给 agent（hermes 空响应重试指纹 / D2 报错形态）。

# 播报（D7 采纳，gate=入站剥离已落地 V-P4）

`on_progress(text)` 每个动作前后各一报，文本以 `splice.NARRATE_MARK` 开头
——入站剥离按它识别回带。不开播报时调用方传 None，行为退化为纯 keepalive
（keepalive 本身由流层发，不在本模块）。

# 入账（§3.2-6，MQ-W 对账）—— 2026-09-02 落地（MQ-P9）

每轮产出 `LoopRound` 记录（调用名 / 耗时 / usage / 本轮消息 / LLM 回复）。
三个调用点（`agency.process_message` 非流式、`intercept_chat_stream`、
`_intercept_protocol_stream`）把 `InnerLoopResult` 交给
`agency.loop_ledger.record(session_prefix, result)`；`server._enqueue_turn`
在主轮入库时 `drain()` 同会话的待入账轮次，每轮以 **aux 轮**
（`auxiliary=True, aux_source=INNER_LOOP_AUX_SOURCE`）经既有 P0 管线入 Hub
——形态与落库见 `bladex_proxy.loopledger`。

历史（2026-08-31 标记时的读数）：`transcript`/`rounds` 曾有生产者零消费者，
08-30 那 56 轮 `route_calling` 82 vs `turn_enqueued` 56 ⇒ 31.7% 上游调用不在
Hub 里。对账口径改为 `route_calling ≈ turn_enqueued + inner_loop_turn_enqueued`。
"""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import structlog
from bladex_core.flags import flag_number

from bladex_proxy.interception import MODE_PURE, classify_message
from bladex_proxy.models import INNER_LOOP_AUX_SOURCE as _INNER_LOOP_AUX_SOURCE
from bladex_proxy.splice import NARRATE_MARK

#: 工具名 → 给**用户**看的人话（ADR-0032 §3.2 #8 举的例子就是 "searching memory…"）。
#: 🔴 **播报里绝不出现 `bladex_*` 原名**：agent 看到的流里不该有任何 bladex 标识符
#: （既有守卫 `test_anthropic_stream_strips_bladex_tool_use` 等钉的就是这条，
#: 2026-08-27 我第一版直接播报函数名，当场把它们打红——那三条测试是对的）。
#: 未登记的工具退化成通用措辞，同样不漏名字。
_NARRATE_PHRASES: dict[str, str] = {
    "bladex_memory_search": "searching memory",
    "bladex_ledger_read": "reading the task ledger",
    "bladex_ledger_update": "updating the task ledger",
    "bladex_ledger_switch": "opening the task ledger",
}


def _phrase(name: str) -> str:
    return _NARRATE_PHRASES.get(name, "working on memory")

logger = structlog.get_logger()

#: call_llm 契约：async (messages: list[dict]) -> dict（OpenAI assistant message 形态，
#: 可含 tool_calls；调用方负责路由/Router 网关/参数）。
CallLLM = Callable[[list[dict]], Awaitable[dict]]
#: dispatch 契约：async (name, arguments) -> str（toolface.dispatch 的偏应用）。
Dispatch = Callable[[str, str], Awaitable[str]]
#: 播报契约：async (text) -> None。None = 不播报。
Progress = Callable[[str], Awaitable[None]] | None


#: 内循环 aux 轮的 `aux_source`（真身在 `models.INNER_LOOP_AUX_SOURCE`，这里再导出）。
#: 不进 `identity.DISTILL_ONLY_AUX_RULES` ⇒ rebuild 三分为 `dropped`（整轮不蒸馏）
#: ——内循环的工具往返没有记忆价值，只是调用预算的支出凭证。
INNER_LOOP_AUX_SOURCE = _INNER_LOOP_AUX_SOURCE
#: aux 轮首条 system 消息的开头标记。`identity.classify_auxiliary` 据此**重判**
#: （ADR-0018 R2：重建不信冻结的 `turn.auxiliary`，判据必须能从消息本身算出）。
INNER_LOOP_MARKER = "[BladeX inner-loop round]"


@dataclass
class LoopRound:
    """一轮内循环的入账记录，经 `loopledger` 以 aux 轮入 Hub（MQ-P9）。

    `messages` = 本轮 assistant(tool_calls) + tool 结果消息；`reply` = 本轮
    LLM 回复（预算门在 LLM 之前触发时为空 dict）。两者合起来就是这一轮
    "模型看到了什么、答了什么"，重建可重放。
    """

    round_index: int
    tool_names: list[str] = field(default_factory=list)
    tool_ms: float = 0.0
    llm_ms: float = 0.0
    usage: dict = field(default_factory=dict)
    messages: list[dict] = field(default_factory=list)
    reply: dict = field(default_factory=dict)


@dataclass
class InnerLoopResult:
    final_message: dict
    rounds: list[LoopRound] = field(default_factory=list)
    degraded: bool = False          # 预算/轮数耗尽走了合成降级
    degrade_reason: str = ""        # budget | max_rounds
    elapsed_s: float = 0.0
    #: 循环期间累计的 (assistant(tool_calls), tool results…) 补充消息
    #: （= 各 `rounds[i].messages` 的拼接）。入 Hub 走 `rounds`，本字段留给
    #: 需要整段视图的调用方。
    transcript: list[dict] = field(default_factory=list)


def _fn(tc: dict) -> tuple[str, str]:
    f = tc.get("function") or {}
    return f.get("name", ""), f.get("arguments", "") or "{}"


def synthesize_results_message(calls_results: list[tuple[str, str]]) -> dict:
    """降级形态：把已有工具结果拼成一条**可转发**的 assistant 文本。"""
    lines = ["[BladeX] memory lookup (partial — budget reached):"]
    lines += [f"- {name}: {result[:500]}" for name, result in calls_results]
    return {"role": "assistant", "content": "\n".join(lines)}


async def run_inner_loop(
    *,
    messages: list[dict],
    initial_calls: list[dict],
    call_llm: CallLLM,
    dispatch: Dispatch,
    on_progress: Progress = None,
    initial_content: str = "",
    budget_s: float | None = None,
    max_rounds: int | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> InnerLoopResult:
    """执行纯 bladex 分流的内循环，返回一条可转发的最终消息。

    `messages` = 已注入/装配好的上游请求消息（不改入参）；`initial_calls` =
    分流器剥出的 bladex tool_calls。循环体：执行调用 → 拼 assistant+tool 消息 →
    call_llm → 若仍是纯 bladex 继续，否则返回。
    """
    budget = budget_s if budget_s is not None else flag_number("BLADEX_INNER_LOOP_BUDGET_S")
    rounds_cap = int(max_rounds if max_rounds is not None
                     else flag_number("BLADEX_INNER_LOOP_MAX_ROUNDS"))
    start = clock()
    working = list(messages)
    transcript: list[dict] = []
    rounds: list[LoopRound] = []
    calls = list(initial_calls)
    all_results: list[tuple[str, str]] = []

    for round_index in range(1, rounds_cap + 1):
        rec = LoopRound(round_index=round_index)
        # ── 执行本轮 bladex 调用 ──
        # 🔴 首轮带上模型**已经说出口的前言**（2026-08-26）：流式形态下那段正文
        # 已经发给 agent 了（用户已经看见"先把证据归档，再给结论"），
        # 内循环里若把它丢掉，模型看不到自己刚说过什么，续写就会重复或跑偏。
        assistant = {"role": "assistant",
                     "content": (initial_content or None) if round_index == 1 else None,
                     "tool_calls": calls}
        working.append(assistant)
        transcript.append(assistant)
        rec.messages.append(assistant)
        t0 = clock()
        for tc in calls:
            name, args = _fn(tc)
            rec.tool_names.append(name)
            if on_progress is not None:
                await on_progress(f"{NARRATE_MARK} {_phrase(name)}…")
            result = await dispatch(name, args)
            all_results.append((name, result))
            tool_msg = {"role": "tool", "tool_call_id": tc.get("id", ""),
                        "content": result}
            working.append(tool_msg)
            transcript.append(tool_msg)
            rec.messages.append(tool_msg)
            if on_progress is not None:
                await on_progress(f"{NARRATE_MARK} {_phrase(name)} — done")
        rec.tool_ms = (clock() - t0) * 1000

        # ── 预算门在下一次 LLM 往返**之前**（工具本地执行便宜，LLM 往返贵）──
        elapsed = clock() - start
        if elapsed >= budget:
            rounds.append(rec)
            logger.warning("innerloop_budget_degrade", elapsed_s=round(elapsed, 1),
                           budget_s=budget, rounds=round_index)
            return InnerLoopResult(
                final_message=synthesize_results_message(all_results),
                rounds=rounds, degraded=True, degrade_reason="budget",
                elapsed_s=elapsed, transcript=transcript)

        t1 = clock()
        reply = await call_llm(working)
        rec.llm_ms = (clock() - t1) * 1000
        rec.usage = dict(reply.get("usage") or {}) if isinstance(reply, dict) else {}
        rec.reply = dict(reply) if isinstance(reply, dict) else {}
        rounds.append(rec)
        # `usage` 是 server 侧 `_loop_llm` 为入账挂上的（消息本体没有这个字段）；
        # 入账已取走，**不许随最终消息外发给 agent**——它不是 wire 形态的一部分。
        if isinstance(reply, dict) and "usage" in reply:
            reply = {k: v for k, v in reply.items() if k != "usage"}

        disp = classify_message(reply)
        if disp.mode != MODE_PURE:
            # none / mixed 都返回——mixed 的剥离-拼接归调用方（V-P2/P4 路径）。
            return InnerLoopResult(final_message=reply, rounds=rounds,
                                   elapsed_s=clock() - start, transcript=transcript)
        calls = disp.bladex_calls

    logger.warning("innerloop_rounds_degrade", rounds=rounds_cap)
    return InnerLoopResult(
        final_message=synthesize_results_message(all_results),
        rounds=rounds, degraded=True, degrade_reason="max_rounds",
        elapsed_s=clock() - start, transcript=transcript)
