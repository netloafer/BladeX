"""三条流式拦截路径（chat / anthropic / responses）+ 内循环播报（`_Narrator` 族）+ 合成事件。

09-06 F0.1 自 `agency.py` 拆出，零行为（逐字搬家）。门面 `bladex_proxy.agency` re-export。
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import structlog
from bladex_core.flags import flag_number

from bladex_proxy.innerloop import InnerLoopResult, run_inner_loop
from bladex_proxy.splice import SpliceRecord, anchor_key
from bladex_proxy.agency.runtime import AgencyRuntime, _call_names, _loop_cost, tool_context

logger = structlog.get_logger()


# ── 流式拦截（chat 协议；V-P5a 只做 chat，anthropic/responses 归 V-P5b）────────



async def intercept_chat_stream(
    sse_iter: Any, *, agency: AgencyRuntime, capture_result: Any,
    upstream_messages: list[dict], session_prefix: str, allowed_exposure: str,
    call_llm: Any, session_id: str, agent_id: str, project_id: str = "",
):
    """包装 capture_stream 的 SSE 流：拦 bladex_* 调用、透传其余。

    行为矩阵（与非流式 process_message 同语义）：
      无 bladex 调用   → 逐字透传（含终止 chunk 与 [DONE]）
      mixed           → 剥离转发；流末执行己方工具 + 记拼接；终止 chunk 原样放行
      纯 bladex       → 全部吞下；流末跑内循环（期间发 SSE 注释 keepalive，
                        D4 实测五 agent 全部接受），把最终消息合成为 chunk 发出
                        ——绝不给 agent 一个空流（hermes 重试指纹）。
    """
    import asyncio
    import json as _json

    from bladex_proxy.interception import ChatStreamStripper

    stripper = ChatStreamStripper()
    held_terminal: list[str] = []      # finish chunk 与 [DONE]，流末按分流决定
    #: 🔴 两个信号必须分开（2026-08-25 live 事故）：`finish_reason=tool_calls`
    #: 的合法性只取决于**转发出去的调用数**，与有没有正文无关。首版用一个
    #: `forwarded_substance`（正文或调用任一为真）判 mixed，于是"有正文 +
    #: 调用全被拦"那轮把 `finish_reason=tool_calls` 原样放行，agent 收到
    #: 「indicated a tool call but none was included」并重试（实测 4 轮）。
    forwarded_calls = False           # 有 agent 自己的调用被转发出去
    forwarded_text = False            # 有正文被转发出去
    last_meta = {"id": "bladex-intercept", "model": ""}
    _last_out = [time.monotonic()]   # V-P3：上次给客户端发字节的时刻

    def _reser(chunk: dict) -> str:
        return f"data: {_json.dumps(chunk, ensure_ascii=False)}\n\n"

    async for sse in sse_iter:
        data = sse.strip()
        if not data.startswith("data: "):
            yield sse
            continue
        payload = data[len("data: "):]
        if payload == "[DONE]":
            held_terminal.append(sse)
            continue
        try:
            chunk = _json.loads(payload)
        except ValueError:
            yield sse
            continue
        last_meta["id"] = chunk.get("id", last_meta["id"])
        last_meta["model"] = chunk.get("model", last_meta["model"])
        choices = chunk.get("choices") or []
        finish = choices[0].get("finish_reason") if choices else None
        delta = (choices[0].get("delta") or {}) if choices else {}
        if delta.get("content"):
            forwarded_text = True
        if finish:
            held_terminal.append(sse)     # 终止 chunk 押后：分流未定
            continue
        _emitted = False
        for out in stripper.feed(chunk):
            if (out.get("choices") or [{}])[0].get("delta", {}).get("tool_calls"):
                forwarded_calls = True
            _emitted = True
            _last_out[0] = time.monotonic()
            yield _reser(out)
        # 剥空 ⇒ 这个 chunk 对客户端不可见。久无输出就发一行 SSE 注释（V-P3）。
        if not _emitted and _idle_too_long(_last_out):
            _last_out[0] = time.monotonic()
            yield _SSE_KEEPALIVE.decode()

    removed = stripper.removed_calls
    if not removed:
        for sse in held_terminal:
            yield sse
        return

    ctx = tool_context(session_id=session_id, agent_id=agent_id,
                       project_id=project_id, upstream_messages=upstream_messages)
    # 🔴 2026-08-26：判据只看 `forwarded_calls`，**不看 forwarded_text**
    # （与 `classify_message` 同一修正；病例见那里的 docstring）。
    # 正文已经流给 agent 了，但"说了一段前言"不代表这一轮结束——模型明说
    # "先归档，再给结论"，得让它拿着工具结果继续。已流出的正文不丢：作为
    # `initial_content` 进内循环的 assistant 消息，模型看得到自己刚说过什么，
    # 续写才连贯；用户侧则是同一条 SSE 流里前言之后接上结论。
    if forwarded_calls:
        # MIXED：agent 侧仍有自己的调用要执行，流后补拼接。
        # 🔴 终止 chunk 必须**按转发出去的内容重写** finish_reason：
        # 上游给的是 `tool_calls`（它算上了被我们拦掉的 bladex 调用），
        # 而 agent 只看到剩下的部分——转发调用为 0 时必须改成 `stop`，
        # 否则 agent 判"说了有工具调用却没有"并重试（live 实测 4 轮）。
        for sse in held_terminal:
            if not forwarded_calls and sse.startswith("data: ") and "[DONE]" not in sse:
                try:
                    _c = _json.loads(sse[len("data: "):])
                    _ch = (_c.get("choices") or [{}])[0]
                    if _ch.get("finish_reason") == "tool_calls":
                        _ch["finish_reason"] = "stop"
                        logger.info("agency_finish_reason_rewritten",
                                    session=session_id, removed=len(removed))
                        yield _reser(_c)
                        continue
                except ValueError:
                    pass
            yield sse
        # MQ-P8：tool_calls 取剥离器的**已发射形态**（含铸 id），不取 capture——
        # capture 看到的是上游原始 delta（deepseek 实测可无 id），agent 回传的
        # 是 wire 上的形态；两者不同源 = 锚哈希级永不相等（拼接恢复 0%）。
        stripped_assistant = {"role": "assistant",
                              "content": capture_result.full_text or "",
                              "tool_calls": list(stripper.forwarded_calls)}
        if not stripped_assistant["tool_calls"]:
            stripped_assistant.pop("tool_calls")
        results = []
        for tc in removed:
            fn = tc.get("function") or {}
            text = await agency.toolface.dispatch(
                fn.get("name", ""), fn.get("arguments", "{}"),
                allowed_exposure=allowed_exposure, context=ctx)
            results.append({"role": "tool", "tool_call_id": tc.get("id", ""),
                            "content": text})
        agency.splice.add(SpliceRecord(
            session_prefix=session_prefix, anchor=anchor_key(stripped_assistant),
            calls=removed, results=results, created_ms=int(time.time() * 1000)))
        logger.info("agency_stream_mixed_stripped", removed=len(removed),
                    names=_call_names(removed))
        return

    # PURE：内循环 + keepalive，最终消息合成为 chunk。
    logger.info("agency_stream_pure_loop", calls=len(removed),
                preamble_chars=len(capture_result.full_text or ""))
    _q: asyncio.Queue = asyncio.Queue()
    loop_task = asyncio.ensure_future(run_inner_loop(
        messages=upstream_messages, initial_calls=removed, call_llm=call_llm,
        initial_content=capture_result.full_text or "", on_progress=_q.put,
        dispatch=lambda n, a: agency.toolface.dispatch(
            n, a, allowed_exposure=allowed_exposure, context=ctx)))
    # V-P3 硬点 8：播报即 keepalive；没有播报的间隙才发注释行。
    _chunks = 0
    _narrator = _ChatNarrator() if _narrate_on() else None
    async for _out in _narrate_until_done(loop_task, _q, narrator=_narrator,
                                          keepalive=": bladex-keepalive\n\n"):
        _chunks += 1
        _last_out[0] = time.monotonic()
        yield _out
    if _chunks:
        logger.info("agency_stream_narrated", phase="inner_loop", chunks=_chunks)
    result: InnerLoopResult = loop_task.result()
    agency.loop_ledger.record(session_prefix, result)
    # 🔴 MQ-L45（2026-09-05）：内循环的最终回复可能是 MIXED（模型先 read、再同轮
    # switch + 自己的工具）。此前这里直接 `final.get("tool_calls")` 全量转发，
    # `bladex_*` 因此被 agent 当自己的工具执行 ⇒ `Tool bladex_ledger_switch not found`
    # （Hub 取证 09-05 12:12 / 13:29；agent 由此判定"BladeX 工具不可用"并转去翻文件系统）。
    # 剥离 + 就地执行 + 记拼接，与首条回复的 MIXED 处置**同一实现点**。
    final, _leaked, _extra = await agency._strip_execute_splice(
        result.final_message, ctx=ctx, session_prefix=session_prefix,
        allowed_exposure=allowed_exposure)
    text = str(final.get("content") or "")
    base = {"id": last_meta["id"], "object": "chat.completion.chunk",
            "created": int(time.time()), "model": last_meta["model"]}
    yield _reser({**base, "choices": [{"index": 0, "delta": {"content": text},
                                       "finish_reason": None}]})
    # 剥离之后剩下的才是 agent 自己的调用——以聚合形态一次性发出（少见但合法）。
    agent_calls = [tc for tc in final.get("tool_calls") or []]
    if agent_calls:
        yield _reser({**base, "choices": [{"index": 0,
                                           "delta": {"tool_calls": [
                                               {**tc, "index": i}
                                               for i, tc in enumerate(agent_calls)]},
                                           "finish_reason": None}]})
    yield _reser({**base, "choices": [{"index": 0, "delta": {},
                                       "finish_reason": "tool_calls" if agent_calls
                                       else "stop"}]})
    yield "data: [DONE]\n\n"
    # 🔴 `agent` / `final_tool_calls` 是 MQ-L23 判据字段（2026-08-31 补齐）。
    #: 转发本身 chat 路径一直是对的（MQ-L23 修的是另外两条协议），但**读数器
    #: 只认 `agency_protocol_inner_loop_done`** ⇒ 走 chat 的内循环
    #: （08-30 实测 10 次）在签收里一次都没被量过。缺的不是能力是**可观测性**，
    #: 与 MQ-P3「每个零件都对，装起来漏一环没人发现」同族：
    #: 这次是「三条路都对，只有两条被量了」。
    # MQ-L45：`final_bladex_stripped` / `final_call_names` 是本条的判据字段——
    # 修前 chat 路只记数量不记名字，16 次 `final_tool_calls>0` 里哪几次是泄漏读不出来。
    if _leaked:
        logger.info("agency_inner_loop_final_stripped",
                    agent=agent_id, count=len(_leaked), names=_call_names(_leaked))
    logger.info("agency_stream_loop_done", rounds=len(result.rounds),
                agent=agent_id, final_tool_calls=len(agent_calls),
                final_bladex_stripped=len(_leaked),
                final_call_names=_call_names(agent_calls),
                final_text_len=len(text),
                degraded=result.degraded, elapsed_s=round(result.elapsed_s, 1),
                **_loop_cost(result))


# ── V-P3：内循环 keepalive（ADR-0032 §3.2）───────────────────────────────────
#
# 🔴 2026-08-27 live 事故：Codex 首轮就调 `bladex_ledger_switch` + `bladex_memory_search`
# ⇒ 纯 bladex ⇒ 走内循环。而**两段都是静默的**：
#   ① 上游流的 603 个 chunk 全是 bladex_* 调用的 delta，被 stripper 剥掉不转发
#      —— 19.4 秒零字节；
#   ② 内循环本身 3.7 秒，生成器挂在 `await` 上什么也不 yield。
# 合计 23 秒无输出，`ms_total=25300` ⇒ Codex CLI 判超时断开
# （日志 `enqueue_shielded_on_disconnect`）。**内循环跑成功了，人没等到。**
#
# 为什么用 **SSE 注释行**而不是协议心跳事件（`ping` / `response.in_progress`）：
# 注释行在三个协议里都是合法且**语义惰性**的——任何符合规范的客户端读到它只会
# 重置空闲计时器，不会进状态机。心跳事件则各协议一套、且有污染客户端状态的风险。
# 一行注释买到"不静默"，代价是零。
#
# 为什么之前没撞到：Hermes 多数轮是混合调用（剥离-拼接，客户端一直有输出）；
# CC 此前 100% 被误判 aux（MQ-A19）根本拿不到工具面，走不到内循环。
# Codex 是第一个"首轮纯 bladex + 长上游流"的组合。MQ-A19 修完后 CC 也会走这条路。

_SSE_KEEPALIVE = b": bladex-keepalive\n\n"


# ── 硬点 8：拦截播报（ADR-0032 §3.2 #8；实测矩阵 survey §5c）────────────────
#
# 内循环期间不只发惰性字节，而是**播报 BladeX 在做什么**（"calling
# bladex_memory_search…"），走 reasoning/thinking 流。D7 实测各 agent 的收益分档：
#   codex 过程可见+收尾折叠（最理想）｜ Pi / dsh 全量渲染｜
#   CC 折叠成 "Thought for Xs" 计时条（半值）｜ hermes 不渲染（退化为 keepalive，零值）
#
# 🔴 **gate：入站剥离先行**（已由 `splice.strip_inbound_echoes` 满足）。
# CC/Pi/dsh 会把播报**回带**（messages 协议 thinking 块与 DeepSeek 系
# reasoning_content 本就要求客户端回传）。播报文本一律以 `NARRATE_MARK` 开头，
# 入站按标记剥净 ⇒ **播报只存在于 BladeX↔agent 之间，LLM 与真上游永远看不到**
# （三红线之 3：模型不调用时零行为差异）。
# 无剥离不许开播报——所以下面这三个发射器都不带"要不要剥"的开关，
# 剥离是 `strip_inbound_echoes` 的无条件行为，不是可选项。


#: 🔴 播报的正确形态**仓库里早就有**：`scripts/agent_probe_upstream.py`
#: （V-R1 D7 的 mock 上游，2026-08-25 对五个 agent 实测过）。
#: 初版我只抄了事件名、把序列和字段全丢了——发的是**不属于任何 item 的裸 delta**：
#:
#:   我写的： {"type": "response.reasoning_summary_text.delta", "delta": …}
#:   验过的： output_item.added(reasoning item) → delta(item_id/output_index/
#:            summary_index 齐全) → output_item.done
#:
#: 同一天第三次栽在"没先找现成的"上。D7 实测读数（survey §5c）：
#:   codex 过程实时可见+收尾折叠（最理想）｜ Pi/dsh 全量渲染 ｜
#:   CC 折叠成计时条 ｜ hermes 不渲染（自然退化为 keepalive）
#:
#: 播报是**有状态**的：开一次 item/block、多次 delta、收一次尾。
#: 故用类而不是纯函数——裸函数发不出"开/收"，那正是初版坏掉的根因。


class _Narrator:
    """协议对应的播报发射器。`open()/delta()/close()` 三段，调用方按序发。

    `index` 由调用方给（避开上游已用的编号），`closed` 幂等——
    内循环可能异常退出，收尾必须能安全重复调用。
    """

    def __init__(self, index: int = 90) -> None:
        self.index = index
        self._opened = False
        self._closed = False

    def open(self) -> list:      # noqa: D102 —— 子类实现
        return []

    def delta(self, text: str) -> list:  # noqa: D102
        return []

    def close(self) -> list:     # noqa: D102
        return []

    def items_used(self) -> int:
        """本轮播报**实际占用**了几个 output item 编号（没开口就是 0）。

        🔴 后续合成消息必须跳过这些编号。2026-08-27 实测：播报与合成都用
        `max_index + 1` ⇒ 官方解析器在 `output_text.delta` 处
        `assert output.type == "message"` 失败（那个下标上坐着 reasoning item）
        ⇒ 断连。与 08-25 `output_index=99` 事故同型：**编号是契约，不是装饰**。
        """
        return 1 if self._opened else 0

    def snapshot_item(self) -> dict | None:
        """交给 `response.completed` 快照的 item（没开口 ⇒ None）。

        流里发过的 item 必须在终止快照里出现，否则客户端对账失败——
        08-25 Codex `client disconnected` 就是这条。
        """
        return None


def _ev(name: str, obj: dict) -> bytes:
    import json as _json
    return (f"event: {name}\ndata: " + _json.dumps(obj, ensure_ascii=False)
            + "\n\n").encode()


class _ResponsesNarrator(_Narrator):
    """responses：reasoning item + summary delta（codex 的理想形态）。"""

    _ITEM_ID = "rs_bladex"

    def open(self):
        self._opened = True
        return [_ev("response.output_item.added", {
            "type": "response.output_item.added", "output_index": self.index,
            "item": {"type": "reasoning", "id": self._ITEM_ID, "summary": []}})]

    def delta(self, text: str):
        if not self._opened:
            return []
        return [_ev("response.reasoning_summary_text.delta", {
            "type": "response.reasoning_summary_text.delta",
            "item_id": self._ITEM_ID, "output_index": self.index,
            "summary_index": 0, "delta": text + "\n"})]

    def close(self):
        if not self._opened or self._closed:
            return []
        self._closed = True
        return [_ev("response.output_item.done", {
            "type": "response.output_item.done", "output_index": self.index,
            "item": self._item()})]

    def _item(self) -> dict:
        return {"type": "reasoning", "id": self._ITEM_ID,
                "summary": [{"type": "summary_text", "text": "BladeX memory work"}]}

    def snapshot_item(self) -> dict | None:
        return self._item() if self._opened else None


class _AnthropicNarrator(_Narrator):
    """messages：thinking block（CC 折叠成计时条，仍可感知）。"""

    def open(self):
        self._opened = True
        return [_ev("content_block_start", {
            "type": "content_block_start", "index": self.index,
            "content_block": {"type": "thinking", "thinking": ""}})]

    def delta(self, text: str):
        if not self._opened:
            return []
        return [_ev("content_block_delta", {
            "type": "content_block_delta", "index": self.index,
            "delta": {"type": "thinking_delta", "thinking": text + "\n"}})]

    def close(self):
        if not self._opened or self._closed:
            return []
        self._closed = True
        return [_ev("content_block_stop", {
            "type": "content_block_stop", "index": self.index})]


class _ChatNarrator(_Narrator):
    """chat/completions：reasoning_content delta（DeepSeek 系形态）。

    OpenAI chunk 无 item 概念，open/close 为空——**这正是我初版误以为
    三协议都这样的来源**。
    """

    def delta(self, text: str):
        import json as _json
        self._opened = True
        return ["data: " + _json.dumps(
            {"id": "bladex-narrate", "object": "chat.completion.chunk",
             "choices": [{"index": 0, "delta": {"reasoning_content": text + "\n"},
                          "finish_reason": None}]},
            ensure_ascii=False) + "\n\n"]


def _idle_too_long(last_out: list) -> bool:
    """距上次给客户端发字节是否已超过心跳间隔。`last_out` 是单元素可变槽。"""
    interval = _keepalive_interval_s()
    return interval > 0 and (time.monotonic() - last_out[0]) >= interval


def _narrate_on() -> bool:
    """播报开关。关掉 = 只发 SSE 注释行心跳（2026-08-27 已跑过真实流量的形态）。"""
    from bladex_core.flags import flag_enabled
    return flag_enabled("BLADEX_NARRATE_INTERCEPT")


def _keepalive_interval_s() -> float:
    """心跳间隔（秒）。0 = 关（回归通道）。"""
    return flag_number("BLADEX_STREAM_KEEPALIVE_S")


async def _narrate_until_done(task: Any, q: Any, *, narrator: Any, keepalive: Any):
    """驱动内循环，边等边产出播报；久无播报时补心跳。异步生成器，调用方 `async for`。

    🔴 **播报是有状态的**：先 `open()` 声明 item/block，之后 delta 才有归属，
    结束必须 `close()`。初版发裸 delta（不属于任何 item），协议上是非法的——
    正确形态见 `scripts/agent_probe_upstream.py`（V-R1 D7 实测过的 mock 上游）。

    `narrator=None` ⇒ 只发心跳（`BLADEX_NARRATE_INTERCEPT=0` 的形态，
    那一版当天跑过真实流量）。

    三条道同时等：**有播报** / **循环结束** / **一拍超时**。播报即 keepalive
    （ADR-0032 §3.2 #8"兼作 keepalive"），心跳只在两次播报间隔过长时兜底。
    """
    interval = _keepalive_interval_s()
    opened = False
    while True:
        getter = asyncio.ensure_future(q.get())
        done, _ = await asyncio.wait({task, getter}, timeout=interval or None,
                                     return_when=asyncio.FIRST_COMPLETED)
        if getter in done:
            text = getter.result()
            if narrator is not None:
                if not opened:
                    opened = True
                    for chunk in narrator.open():
                        yield chunk
                for chunk in narrator.delta(text):
                    yield chunk
            elif keepalive is not None:
                yield keepalive          # 播报关：退回心跳
            continue
        getter.cancel()
        if task in done:
            while not q.empty():          # 收尾：还没取走的播报
                text = q.get_nowait()
                if narrator is not None and opened:
                    for chunk in narrator.delta(text):
                        yield chunk
                elif narrator is None and keepalive is not None:
                    yield keepalive
            # 🔴 收尾在**正常返回路径**上，不能放 `finally`：
            # 消费方 `aclose()`（客户端断连）会在 yield 点抛 `GeneratorExit`，
            # 而 `finally` 里再 yield 就是 `RuntimeError: async generator
            # ignored GeneratorExit` —— 初版就是这么写的，测试当场抓出来。
            # 而且那种情况**本来也不需要收尾**：客户端都断了，发给谁？
            if narrator is not None and opened:
                for chunk in narrator.close():
                    yield chunk
            return
        if keepalive is not None:
            yield keepalive


# ── V-P5b：anthropic / responses 流式拦截（复用 V-P2 的两个剥离器）──────────
#
# 两协议的生成器都产出 `event: X\ndata: {json}\n\n` 的 bytes，故一套 parse-feed-
# reserialize 通吃；差别只在剥离器与"终止事件"的判据，用参数注入。
#
# 🔴 与 chat 端点的语义差（不是简化，是协议事实）：
# - anthropic/responses 的终止事件（message_delta / response.completed）携带
#   stop_reason，同样要按**转发出去的调用数**改写（chat 端点 live 事故同型）；
# - 纯 bladex 调用（剥完空流）在这两个协议里同样不可发空流，走内循环合成。
#   V-P5b 首版：内循环仅在 chat 端点启用（call_llm 需按协议构造），
#   两协议先只做**剥离-拼接**（mixed），纯 bladex 走"合成文本事件"降级——
#   这保证不会给 agent 一个空流，代价是那轮不做多轮内循环（记 V-P5c）。


async def _intercept_protocol_stream(
    sse_iter: Any, *, agency: AgencyRuntime, stripper: Any, capture_result: Any,
    session_prefix: str, allowed_exposure: str, session_id: str, agent_id: str,
    upstream_messages: list[dict], terminal_types: tuple[str, ...],
    project_id: str = "",
    rewrite_stop: Any, synth_text_events: Any, narrate: Any, call_llm: Any = None,
    synth_tool_call_events: Any = None,
):
    """anthropic / responses 共用的流式拦截。`stripper` = 对应协议的剥离器。"""
    import json as _json

    held: list[bytes] = []
    forwarded_calls = False
    forwarded_text = False
    _last_out = [time.monotonic()]   # V-P3：上次给客户端发字节的时刻
    #: 合成 item 的落点与 index（responses 协议：与终止快照同源，见 _synth_* 注释）
    synth_items: list = []
    max_index = -1

    async for raw in sse_iter:
        chunk = raw if isinstance(raw, bytes) else str(raw).encode()
        text = chunk.decode("utf-8", "replace")
        etype = ""
        payload: dict = {}
        for line in text.splitlines():
            if line.startswith("event: "):
                etype = line[len("event: "):].strip()
            elif line.startswith("data: "):
                try:
                    payload = _json.loads(line[len("data: "):])
                except ValueError:
                    payload = {}
        if not etype or not payload:
            yield chunk
            continue
        if etype in terminal_types:
            held.append(chunk)          # 终止事件押后：分流未定
            continue
        _emitted = False
        for out in stripper.feed({**payload, "type": payload.get("type", etype)}):
            if isinstance(out.get("output_index"), int):
                max_index = max(max_index, out["output_index"])
            t = out.get("type", "")
            if "text" in t or t.endswith("output_text.delta"):
                forwarded_text = True
            if "tool_use" in _json.dumps(out) or "function_call" in t:
                forwarded_calls = True
            _emitted = True
            _last_out[0] = time.monotonic()
            yield f"event: {t or etype}\ndata: {_json.dumps(out, ensure_ascii=False)}\n\n".encode()
        # 剥空 ⇒ 客户端看不到这个 chunk。久无输出就发心跳（V-P3；live 事故里
        # 603 个 chunk 全被剥掉 = 19.4 秒零字节）。
        if not _emitted and _idle_too_long(_last_out):
            _last_out[0] = time.monotonic()
            yield _SSE_KEEPALIVE

    removed = stripper.removed_calls
    if not removed:
        for h in held:
            yield h
        return

    ctx = tool_context(session_id=session_id, agent_id=agent_id,
                       project_id=project_id, upstream_messages=upstream_messages)

    # 同上：正文不参与"这一轮完没完"的判定（2026-08-26 修正）。
    if not forwarded_calls and call_llm is not None:
        # ── V-P5c：纯 bladex 走**真正的内循环**（CC live 实证：首轮就调 bladex
        # 工具是主路径，合成降级会把模型的思路打断在第一步——它再也走不到
        # `bladex_ledger_switch`，账本机制在 CC 上永远起不来）。
        # V-P3：内循环期间生成器本来挂在 await 上什么也不 yield ⇒ 客户端静默
        # （live 事故：+3.7 秒）。改成边等边 yield 心跳。
        # V-P3 硬点 8：内循环期间播报 BladeX 在做什么（走 reasoning/thinking 流），
        # 没有播报的间隙用心跳兜底。两者都必须**边等边出**，否则等于没有。
        _q: asyncio.Queue = asyncio.Queue()
        _task = asyncio.ensure_future(run_inner_loop(
            messages=upstream_messages, initial_calls=removed, call_llm=call_llm,
            on_progress=_q.put,
            dispatch=lambda n, a: agency.toolface.dispatch(
                n, a, allowed_exposure=allowed_exposure, context=ctx)))
        _beats = 0
        # index 避开上游已用的编号（max_index 一路在跟踪）。
        _narrator = narrate(max_index + 1) if _narrate_on() else None
        async for _out in _narrate_until_done(_task, _q, narrator=_narrator,
                                              keepalive=_SSE_KEEPALIVE):
            _beats += 1
            _last_out[0] = time.monotonic()
            yield _out
        if _beats:
            logger.info("agency_stream_narrated", phase="inner_loop", chunks=_beats)
        result = _task.result()
        agency.loop_ledger.record(session_prefix, result)
        # 🔴 MQ-L45（2026-09-05）：下面那段注释里"先量再修"的那件事，现在修了——
        # 内循环的最终回复若是 MIXED，`bladex_*` 必须先剥离执行，不能随 agent 调用一起转发。
        final, _leaked, _extra = await agency._strip_execute_splice(
            result.final_message, ctx=ctx, session_prefix=session_prefix,
            allowed_exposure=allowed_exposure)
        if _leaked:
            logger.info("agency_inner_loop_final_stripped",
                        agent=agent_id, count=len(_leaked), names=_call_names(_leaked))
        text = str(final.get("content") or "")
        # 🔴 记账缺口（同批修）：内循环产出的文本模型看得见 ⇒ 必须进 capture，
        # 否则 Hub 那轮 response_text 为空 =「model-visible ⟺ logged」破了。
        try:
            capture_result.full_text = (capture_result.full_text or "") + text
        except Exception:  # noqa: BLE001,S110
            pass
        # 🔴 播报若开了 item，它占掉一个编号，合成消息必须往后让。
        #: 并把播报 item 送进终止快照——流里发过的 item 不在 `response.completed`
        #: 的 output 里，客户端对账失败即断连（08-25 事故）。
        _used = _narrator.items_used() if _narrator else 0
        if _narrator is not None:
            _snap = _narrator.snapshot_item()
            if _snap is not None and synth_items is not None:
                synth_items.append(_snap)
        for ev in synth_text_events([{"content": text}], raw=True,
                                    index=max_index + 1 + _used, sink=synth_items):
            yield ev
        # 🔴 判据埋点（2026-08-27 MQ-L23）：内循环的最终回复若带 agent 自己的
        #: tool_calls，我们**只转发了 content**，调用被丢弃 ⇒ 客户端看到一个
        #: 无工具调用的助手消息 = 这回合已完成 ⇒ 收工。
        #: ~~`innerloop` 的契约明写「mixed 的剥离-拼接归调用方」，而这条路径没做。
        #: 先量再修~~ —— **2026-09-05 MQ-L45 已修**：`final` 在上面已过
        #: `_strip_execute_splice`，到这里只剩 agent 自己的调用。留着这段是为了记住
        #: 教训：「先量再修」当时把仪器装在了**这条**路上，而实际漏的是 chat 流式那条
        #: （那里连调用名都不记）。09-05 靠 agent 侧的 "Tool not found" 才浮出来——
        #: 仪器装错了路径，等于没装。
        _final_tcs = final.get("tool_calls") or []
        if _final_tcs and synth_tool_call_events is not None:
            for ev in synth_tool_call_events(
                    _final_tcs, index=max_index + 2 + _used, sink=synth_items):
                yield ev
            logger.info("agency_inner_loop_final_calls_forwarded",
                        agent=agent_id, count=len(_final_tcs),
                        names=[(t.get("function") or {}).get("name", "?")
                               for t in _final_tcs][:6])
        elif _final_tcs:
            logger.warning("agency_inner_loop_final_calls_dropped",
                           agent=agent_id, count=len(_final_tcs))
        logger.info("agency_protocol_inner_loop_done", removed=len(removed),
                    agent=agent_id, rounds=len(result.rounds),
                    degraded=result.degraded, final_tool_calls=len(_final_tcs),
                    final_text_len=len(text),
                    elapsed_s=round(result.elapsed_s, 1), **_loop_cost(result))
        # 🔴 转发了 agent 调用 ⇒ 终止事件必须保留 "还有工具要执行" 的语义
        #: （anthropic 保 `stop_reason=tool_use`，responses 保快照里的
        #: function_call item）。写死 False 会把它改成"回合结束"。
        for h in held:
            yield rewrite_stop(h, bool(_final_tcs), synth_items, stripper=stripper)
        return

    # 执行己方工具（mixed：剥离-拼接；无 call_llm 时的纯 bladex：合成降级）
    results = []
    for tc in removed:
        fn = tc.get("function") or {}
        out_text = await agency.toolface.dispatch(
            fn.get("name", ""), fn.get("arguments", "{}"),
            allowed_exposure=allowed_exposure, context=ctx)
        results.append({"role": "tool", "tool_call_id": tc.get("id", ""),
                        "content": out_text})

    if not forwarded_calls:
        # 无 call_llm 的兜底：绝不发空流——把结果合成为文本事件。
        for ev in synth_text_events(results, index=max_index + 1,
                                    sink=synth_items):
            yield ev
        logger.info("agency_protocol_pure_synthesized", removed=len(removed),
                    agent=agent_id)
    else:
        # MQ-P8：同 chat 侧——tool_calls 取剥离器已发射形态（含铸 id）。
        stripped_assistant = {"role": "assistant",
                              "content": capture_result.full_text or "",
                              "tool_calls": list(stripper.forwarded_calls)}
        if not stripped_assistant["tool_calls"]:
            stripped_assistant.pop("tool_calls")
        agency.splice.add(SpliceRecord(
            session_prefix=session_prefix, anchor=anchor_key(stripped_assistant),
            calls=removed, results=results, created_ms=int(time.time() * 1000)))
        logger.info("agency_protocol_mixed_stripped", removed=len(removed),
                    agent=agent_id, names=_call_names(removed))

    for h in held:
        yield rewrite_stop(h, forwarded_calls, synth_items, stripper=stripper)


def _rewrite_stop_anthropic(chunk: bytes, forwarded_calls: bool,
                            synth_items: list | None = None, stripper: Any = None) -> bytes:
    """message_delta 的 stop_reason：转发调用为 0 时 tool_use → end_turn。"""
    import json as _json
    if forwarded_calls:
        return chunk
    text = chunk.decode("utf-8", "replace")
    if "tool_use" not in text:
        return chunk
    try:
        head, _, data = text.partition("data: ")
        obj = _json.loads(data)
    except ValueError:
        return chunk
    d = obj.get("delta") or {}
    if d.get("stop_reason") == "tool_use":
        d["stop_reason"] = "end_turn"
        obj["delta"] = d
        logger.info("agency_stop_reason_rewritten", protocol="anthropic")
        return (head + "data: " + _json.dumps(obj, ensure_ascii=False) + "\n\n").encode()
    return chunk


def _rewrite_stop_responses(chunk: bytes, forwarded_calls: bool,
                            synth_items: list | None = None,
                            stripper: Any = None) -> bytes:
    """responses 的 `response.completed` 携带**整份 output 快照**——里面还有
    被我们拦掉的 function_call item（2026-08-25 测试当场抓到的泄漏）。
    终止事件必须同样过滤：剔除 bladex_ 开头的 item 并重排 output_index。"""
    import json as _json
    text = chunk.decode("utf-8", "replace")
    _has_minted = bool(getattr(stripper, "_minted_items", None))
    if "bladex_" not in text and not synth_items and not _has_minted:
        return chunk
    try:
        head, _, data = text.partition("data: ")
        obj = _json.loads(data)
    except ValueError:
        return chunk
    resp = obj.get("response") or {}
    items = resp.get("output") or []
    kept = [it for it in items
            if not str((it or {}).get("name", "")).startswith("bladex_")]
    dropped = len(items) - len(kept)
    # 🔴 合成 item 必须补进快照（否则流内有、快照无 ⇒ Codex 对账失败断连）。
    added = 0
    for it in synth_items or []:
        if not any((k or {}).get("id") == it.get("id") for k in kept):
            kept.append(it)
            added += 1
    # MQ-P8：流内 function_call item 带的是铸的 call_id，快照必须同源
    # （流/快照分叉 = Codex 对账断连，08-25 事故同族）。
    stamped = (stripper.stamp_snapshot_items(kept)
               if hasattr(stripper, "stamp_snapshot_items") else 0)
    if not dropped and not added and not stamped:
        return chunk
    resp["output"] = kept
    obj["response"] = resp
    logger.info("agency_responses_snapshot_filtered", dropped=dropped, added=added,
                stamped=stamped)
    return (head + "data: " + _json.dumps(obj, ensure_ascii=False) + "\n\n").encode()


#: 🔴 MQ-L23（2026-08-27）：内循环的最终回复可能带 **agent 自己的** tool_calls
#: （`innerloop` 契约明写「mixed 的剥离-拼接归调用方」）。chat 协议做了
#: （`intercept_chat_stream` 里 agent_calls 那段），**responses / anthropic 没做**
#: ⇒ 调用被丢弃 ⇒ 客户端收到一个无工具调用的助手消息 = 这回合完成 ⇒ 收工。
#: 症状：Codex 每次内循环之后零后续请求，2026-08-27 一天五次。
#: 与 MQ-A18 同型：**同一件事只在一个端点上做对了**。
#: 形态照抄 `scripts/agent_probe_upstream.py:355-369`（真协议那份）。


def _synth_responses_tool_call_events(tool_calls: list[dict], *, index: int = 0,
                                      sink: list | None = None):
    """agent tool_calls → Responses `function_call` 事件（每个调用占一个 item）。"""
    import json as _json

    def ev(name, obj):
        return f"event: {name}\ndata: {_json.dumps(obj, ensure_ascii=False)}\n\n".encode()

    for off, tc in enumerate(tool_calls):
        fn = tc.get("function") or {}
        call_id = str(tc.get("id") or f"call_bladex_{off}")
        item_id = f"fc_bladex_{off}"
        args = str(fn.get("arguments") or "{}")
        item = {"type": "function_call", "id": item_id, "call_id": call_id,
                "name": str(fn.get("name") or ""), "arguments": args,
                "status": "completed"}
        if sink is not None:
            sink.append(item)
        idx = index + off
        yield ev("response.output_item.added", {
            "type": "response.output_item.added", "output_index": idx,
            "item": {**item, "arguments": "", "status": "in_progress"}})
        yield ev("response.function_call_arguments.delta", {
            "type": "response.function_call_arguments.delta",
            "item_id": item_id, "output_index": idx, "delta": args})
        yield ev("response.function_call_arguments.done", {
            "type": "response.function_call_arguments.done",
            "item_id": item_id, "output_index": idx, "arguments": args})
        yield ev("response.output_item.done", {
            "type": "response.output_item.done", "output_index": idx, "item": item})


def _synth_anthropic_tool_call_events(tool_calls: list[dict], *, index: int = 1,
                                      sink: list | None = None):
    """agent tool_calls → Anthropic `tool_use` content blocks。

    `index` 从 1 起（0 被合成文本块占）。参数化而非硬编码，因为这里
    **块序号确实要连续**——与文本块那处不同，见该函数注释。
    """
    import json as _json

    def ev(name, obj):
        return f"event: {name}\ndata: {_json.dumps(obj, ensure_ascii=False)}\n\n".encode()

    for off, tc in enumerate(tool_calls):
        fn = tc.get("function") or {}
        idx = index + off
        block = {"type": "tool_use", "id": str(tc.get("id") or f"toolu_bladex_{off}"),
                 "name": str(fn.get("name") or ""), "input": {}}
        if sink is not None:
            sink.append(block)
        yield ev("content_block_start", {"type": "content_block_start",
                                         "index": idx, "content_block": block})
        yield ev("content_block_delta", {
            "type": "content_block_delta", "index": idx,
            "delta": {"type": "input_json_delta",
                      "partial_json": str(fn.get("arguments") or "{}")}})
        yield ev("content_block_stop", {"type": "content_block_stop", "index": idx})


def _synth_anthropic_text_events(results: list[dict], *, raw: bool = False,
                                 index: int = 0, sink: list | None = None):
    import json as _json
    import uuid as _uuid
    text = (str(results[0].get("content", "")) if raw and results
            else "[BladeX] " + " ".join(str(r.get("content", ""))[:400] for r in results))
    mid = f"msg_{_uuid.uuid4().hex[:12]}"

    def ev(name, obj):
        return f"event: {name}\ndata: {_json.dumps(obj, ensure_ascii=False)}\n\n".encode()

    # `index` 参数在本协议**故意不用**（不是漏了）：Anthropic 官方累加器
    # `accumulate_event` 对 `content_block_start` 做的是 `content.append(...)`，
    # 按到达顺序排，不按 index 定位——播报@0 之后合成再发 0 也会正确落到第 1 块。
    # 2026-08-27 用真 SDK 累加器实测三种形态全部通过。
    # 🔴 与 Responses 相反：那边 `content_part.added` 用 `output.content[i]` 定位，
    #    编号错一位就 IndexError。**同一件事在两个协议里的判据不同，别互相套用。**
    yield ev("content_block_start", {"type": "content_block_start", "index": 0,
                                     "content_block": {"type": "text", "text": ""}})
    yield ev("content_block_delta", {"type": "content_block_delta", "index": 0,
                                     "delta": {"type": "text_delta", "text": text}})
    yield ev("content_block_stop", {"type": "content_block_stop", "index": 0})
    _ = mid


#: 合成 message item 与终止快照**必须同源**（2026-08-25 Codex live 事故：
#: 合成事件用 `output_index=99` 占位、且 `response.completed` 的 output 快照
#: 里没有这条 item ⇒ Codex 拿流内 item 与最终快照对账、对不上就断开连接
#: `client disconnected`）。用模块级容器把这一轮合成的 item 交给 rewrite_stop。
_SYNTH_ITEM_ID = "msg_bladex"


def _synth_responses_text_events(results: list[dict], *, raw: bool = False,
                                 index: int = 0, sink: list | None = None):
    import json as _json
    text = (str(results[0].get("content", "")) if raw and results
            else "[BladeX] " + " ".join(str(r.get("content", ""))[:400] for r in results))

    def ev(name, obj):
        return f"event: {name}\ndata: {_json.dumps(obj, ensure_ascii=False)}\n\n".encode()

    item = {"type": "message", "id": _SYNTH_ITEM_ID, "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}]}
    if sink is not None:
        sink.append(item)          # 交给 rewrite_stop 补进 response.completed 快照
    part_done = {"type": "output_text", "text": text, "annotations": []}
    # 🔴 `content_part.added` 不是可选装饰——它是 delta 的**前置条件**。
    #: 官方解析器 `accumulate_event`：content_part.added 做
    #: `output.content.append(part)`，output_text.delta 做
    #: `output.content[content_index]`。少了前者，delta 落在空 list 上
    #: ⇒ IndexError ⇒ 流解析崩 ⇒ 客户端断连。
    #: 2026-08-27 实测（openai SDK 真解析器喂生产字节）：缺 → IndexError，
    #: 补 → 通过。四次「内循环之后零后续请求」的根因就是这一条。
    #: 形态照抄 `scripts/agent_probe_upstream.py:383-399`（Codex 已验通过那份），
    #: 含 `status="in_progress"`——`added` 时这条 item 还没写完。
    yield ev("response.output_item.added", {"type": "response.output_item.added",
                                            "output_index": index,
                                            "item": {**item, "status": "in_progress",
                                                     "content": []}})
    yield ev("response.content_part.added", {"type": "response.content_part.added",
                                             "item_id": _SYNTH_ITEM_ID,
                                             "output_index": index, "content_index": 0,
                                             "part": {"type": "output_text", "text": "",
                                                      "annotations": []}})
    yield ev("response.output_text.delta", {"type": "response.output_text.delta",
                                            "item_id": _SYNTH_ITEM_ID,
                                            "output_index": index, "content_index": 0,
                                            "delta": text})
    yield ev("response.content_part.done", {"type": "response.content_part.done",
                                            "item_id": _SYNTH_ITEM_ID,
                                            "output_index": index, "content_index": 0,
                                            "part": part_done})
    yield ev("response.output_item.done", {"type": "response.output_item.done",
                                           "output_index": index, "item": item})


def intercept_anthropic_stream(sse_iter, **kw):
    from bladex_proxy.interception import AnthropicStreamStripper
    return _intercept_protocol_stream(
        sse_iter, stripper=AnthropicStreamStripper(),
        terminal_types=("message_delta", "message_stop"),
        rewrite_stop=_rewrite_stop_anthropic,
        synth_text_events=_synth_anthropic_text_events,
        synth_tool_call_events=_synth_anthropic_tool_call_events,
        narrate=_AnthropicNarrator, **kw)


def intercept_responses_stream(sse_iter, **kw):
    from bladex_proxy.interception import ResponsesStreamStripper
    return _intercept_protocol_stream(
        sse_iter, stripper=ResponsesStreamStripper(),
        terminal_types=("response.completed", "response.incomplete"),
        rewrite_stop=_rewrite_stop_responses,
        synth_text_events=_synth_responses_text_events,
        synth_tool_call_events=_synth_responses_tool_call_events,
        narrate=_ResponsesNarrator, **kw)
