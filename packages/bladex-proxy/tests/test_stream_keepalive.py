"""V-P3 内循环 keepalive：纯 bladex 轮次期间不静默（ADR-0032 §3.2）。

🔴 事故（2026-08-27 live）：Codex 首轮就调 `bladex_ledger_switch` +
`bladex_memory_search` ⇒ 纯 bladex ⇒ 走内循环。**两段都是静默的**：
  ① 上游流 603 个 chunk 全是 bladex_* 调用的 delta，被 stripper 剥掉不转发 —— 19.4s；
  ② 内循环本身 3.7s，生成器挂在 await 上什么也不 yield。
合计 23 秒零字节，`ms_total=25300` ⇒ Codex CLI 判超时断开
（`enqueue_shielded_on_disconnect`）。**内循环跑成功了，人没等到。**

排查两处教训写在这里，免得重犯：
- 我第一次说"keepalive 全仓零实现"是**错的**——chat 端点早有，是 `head -8`
  截断了 grep 结果。缺的是 responses/anthropic 两条协议路径 + 上游剥空阶段。
- 初版把心跳攒进队列、等 await 回来再一起 yield ——**那等于没有心跳**。
  异步生成器不能在别的协程里 yield，但可以在自己的帧里边等边 yield。
"""

from __future__ import annotations

import asyncio

import pytest


class TestKeepaliveShape:
    def test_marker_is_an_inert_sse_comment(self):
        """三协议通吃、语义惰性：注释行不进任何客户端状态机。"""
        from bladex_proxy.agency import _SSE_KEEPALIVE

        assert _SSE_KEEPALIVE.startswith(b":"), "必须是 SSE 注释行"
        assert _SSE_KEEPALIVE.endswith(b"\n\n"), "SSE 帧要以空行结束"
        for bad in (b"event:", b"data:", b"{"):
            assert bad not in _SSE_KEEPALIVE, f"心跳不该携带 {bad!r}"

    def test_interval_comes_from_the_single_source(self):
        """🔴 同一参数两个默认值 = 缺陷（刚性原则 12）。

        原来 agency.py 有个硬编码 `_KEEPALIVE_INTERVAL_S = 10.0`，
        新接协议路径时差点再拍一个 —— 收进 flags 单一真相源。
        """
        import bladex_proxy.agency as ag
        from bladex_core.flags import MEMORY_NUMERIC_DEFAULTS

        assert not hasattr(ag, "_KEEPALIVE_INTERVAL_S"), "硬编码常量又回来了"
        assert MEMORY_NUMERIC_DEFAULTS["BLADEX_STREAM_KEEPALIVE_S"] == 10.0

    def test_zero_disables(self, monkeypatch):
        from bladex_proxy.agency import _idle_too_long, _keepalive_interval_s

        monkeypatch.setenv("BLADEX_STREAM_KEEPALIVE_S", "0")
        assert _keepalive_interval_s() == 0
        assert _idle_too_long([0.0]) is False, "关掉之后不该再发心跳"


class TestIdleGate:
    def test_fires_only_after_the_interval(self, monkeypatch):
        import time

        from bladex_proxy.agency import _idle_too_long

        monkeypatch.setenv("BLADEX_STREAM_KEEPALIVE_S", "10")
        now = time.monotonic()
        assert _idle_too_long([now]) is False          # 刚发过
        assert _idle_too_long([now - 11]) is True      # 静默超过一拍


# ── 硬点 8：拦截播报通道（ADR-0032 §3.2 #8）────────────────────────────────

class TestNarrationCarriers:
    """三协议的播报形态必须与 **V-R1 D7 实测过的 mock 上游**一致。

    🔴 参考实现就在仓库里：`scripts/agent_probe_upstream.py`
    （2026-08-25 对五个 agent 实测：codex 过程实时可见+收尾折叠、Pi/dsh 全量渲染、
    CC 折叠计时条、hermes 不渲染）。**初版我只抄了事件名，把序列和字段全丢了**
    ——发的是不属于任何 item 的裸 delta，协议上非法。
    这几条测试就是把"与参考实现同形"钉住。
    """

    def test_responses_declares_item_before_delta(self):
        from bladex_proxy.agency import _ResponsesNarrator

        n = _ResponsesNarrator(index=7)
        head = b"".join(n.open()).decode()
        assert "response.output_item.added" in head
        assert '"type": "reasoning"' in head
        assert '"output_index": 7' in head

    def test_responses_delta_carries_all_required_fields(self):
        """🔴 初版缺的正是这三个字段——delta 不属于任何 item 就是非法事件。"""
        from bladex_proxy.agency import _ResponsesNarrator

        n = _ResponsesNarrator(index=7)
        n.open()
        d = b"".join(n.delta("[bladex-live] searching memory")).decode()
        for field in ('"item_id"', '"output_index"', '"summary_index"', '"delta"'):
            assert field in d, f"缺 {field}（参考实现里有）"

    def test_responses_closes_the_item(self):
        from bladex_proxy.agency import _ResponsesNarrator

        n = _ResponsesNarrator(index=7)
        n.open()
        assert "response.output_item.done" in b"".join(n.close()).decode()

    def test_anthropic_opens_and_closes_a_thinking_block(self):
        from bladex_proxy.agency import _AnthropicNarrator

        n = _AnthropicNarrator(index=5)
        assert '"type": "thinking"' in b"".join(n.open()).decode()
        assert "thinking_delta" in b"".join(n.delta("x")).decode()
        assert "content_block_stop" in b"".join(n.close()).decode()

    def test_chat_needs_no_item_lifecycle(self):
        """OpenAI chunk 无 item 概念——**初版误以为三协议都这样**。"""
        from bladex_proxy.agency import _ChatNarrator

        n = _ChatNarrator()
        assert n.open() == [] and n.close() == []
        assert "reasoning_content" in n.delta("x")[0]

    def test_delta_before_open_emits_nothing(self):
        """没开 item 就发 delta = 初版那个非法形态。这条让它发不出来。"""
        from bladex_proxy.agency import _AnthropicNarrator, _ResponsesNarrator

        for cls in (_ResponsesNarrator, _AnthropicNarrator):
            assert cls(index=3).delta("x") == []

    def test_close_is_idempotent_and_needs_open(self):
        """内循环可能异常退出，收尾要能安全重复调用。"""
        from bladex_proxy.agency import _ResponsesNarrator

        n = _ResponsesNarrator()
        assert n.close() == []          # 没开过
        n.open()
        assert n.close() != []
        assert n.close() == []          # 幂等

    def test_every_carrier_keeps_the_mark(self):
        """🔴 标记是入站剥离的唯一依据——载体不得把它吃掉。"""
        from bladex_proxy.agency import (
            _AnthropicNarrator,
            _ChatNarrator,
            _ResponsesNarrator,
        )
        from bladex_proxy.splice import NARRATE_MARK

        text = f"{NARRATE_MARK} searching memory"
        for cls in (_ResponsesNarrator, _AnthropicNarrator):
            n = cls(index=9)
            n.open()
            assert NARRATE_MARK in b"".join(n.delta(text)).decode()
        assert NARRATE_MARK in _ChatNarrator().delta(text)[0]


class TestNarrationRoundTrip:
    """🔴 gate：播报回带必须被剥净——**LLM 与真上游永远看不到播报**。

    CC / Pi / dsh 实测会把播报存档回带（messages 协议 thinking 块与 DeepSeek 系
    reasoning_content 本就要求客户端回传）。剥不掉就等于污染上游上下文，
    破坏三红线之 3（模型不调用时零行为差异）。
    """

    def test_chat_echo_is_stripped(self):
        from bladex_proxy.splice import NARRATE_MARK, strip_inbound_echoes

        msgs = [{"role": "user", "content": "干活"},
                {"role": "assistant", "content": "好的",
                 "reasoning_content": f"{NARRATE_MARK} calling bladex_memory_search…"}]
        out, n = strip_inbound_echoes(msgs)
        assert n >= 1
        assert not any(NARRATE_MARK in str(m.get("reasoning_content", "")) for m in out)

    def test_anthropic_thinking_block_echo_is_stripped(self):
        from bladex_proxy.splice import NARRATE_MARK, strip_inbound_echoes

        msgs = [{"role": "assistant", "content": [
            {"type": "thinking", "thinking": f"{NARRATE_MARK} step 1"},
            {"type": "text", "text": "模型自己说的话"}]}]
        out, n = strip_inbound_echoes(msgs)
        assert n >= 1
        blocks = out[0]["content"]
        assert not any(b.get("type") == "thinking" for b in blocks)
        assert any(b.get("text") == "模型自己说的话" for b in blocks), "误伤了模型正文"

    def test_agents_own_thinking_is_not_touched(self):
        """阴性对照：没有标记的思考块是 agent/模型自己的，一个字都不许动。"""
        from bladex_proxy.splice import strip_inbound_echoes

        msgs = [{"role": "assistant", "content": [
            {"type": "thinking", "thinking": "我先看一下目录结构"}]}]
        out, n = strip_inbound_echoes(msgs)
        assert n == 0 and out == msgs


class _EchoNarrator:
    """驱动逻辑测试专用：原样回显文本，open/close 不产出。

    载体格式由 `TestNarrationCarriers` 单独钉；这里只关心**边等边出**。
    """

    def open(self):
        self._opened = True
        return []

    def delta(self, text):
        return [text]

    def close(self):
        return []


class TestNarrationDrive:
    @pytest.mark.asyncio
    async def test_narration_arrives_while_the_loop_runs(self, monkeypatch):
        """🔴 播报必须**边等边出**，不是循环结束一起吐。"""
        import time

        from bladex_proxy.agency import _narrate_until_done

        monkeypatch.setenv("BLADEX_STREAM_KEEPALIVE_S", "10")   # 心跳不该抢戏
        q: asyncio.Queue = asyncio.Queue()
        done_at: list[float] = []

        async def loop():
            await q.put("[bladex-live] step 1")
            await asyncio.sleep(0.15)
            await q.put("[bladex-live] step 2")
            await asyncio.sleep(0.15)
            done_at.append(time.monotonic())
            return "ok"

        task = asyncio.ensure_future(loop())
        first_at = None
        got = []
        async for out in _narrate_until_done(task, q, narrator=_EchoNarrator(),
                                             keepalive=None):
            got.append(out)
            first_at = first_at or time.monotonic()
        assert got == ["[bladex-live] step 1", "[bladex-live] step 2"], got
        assert first_at < done_at[0], "播报等到循环结束才出 = 等于没有"

    @pytest.mark.asyncio
    async def test_keepalive_only_fills_the_gaps(self, monkeypatch):
        """播报即 keepalive；心跳只在两次播报间隔过长时兜底。"""
        from bladex_proxy.agency import _narrate_until_done

        monkeypatch.setenv("BLADEX_STREAM_KEEPALIVE_S", "0.05")
        q: asyncio.Queue = asyncio.Queue()

        async def loop():
            await q.put("N1")
            await asyncio.sleep(0.3)      # 长间隙 ⇒ 该有心跳
            return "ok"

        task = asyncio.ensure_future(loop())
        got = [o async for o in _narrate_until_done(task, q,
                                                    narrator=_EchoNarrator(),
                                                    keepalive="KA")]
        assert got[0] == "N1"
        assert "KA" in got[1:], f"长间隙没兜底: {got}"

    @pytest.mark.asyncio
    async def test_trailing_narration_is_not_dropped(self, monkeypatch):
        """循环结束那一刻队列里还没取走的播报要收尾吐出，不能丢。"""
        from bladex_proxy.agency import _narrate_until_done

        monkeypatch.setenv("BLADEX_STREAM_KEEPALIVE_S", "5")
        q: asyncio.Queue = asyncio.Queue()

        async def loop():
            await q.put("first")
            await q.put("last")           # 紧接着就返回，来不及被取
            return "ok"

        task = asyncio.ensure_future(loop())
        got = [o async for o in _narrate_until_done(task, q,
                                                    narrator=_EchoNarrator(),
                                                    keepalive=None)]
        assert "last" in got, f"收尾播报被丢了: {got}"


class TestNonStreamingHasNoNarration:
    def test_non_streaming_path_passes_no_on_progress(self):
        """非流式没有增量通道，播报无处可去 —— 那一处保持 None 是对的。

        钉住它，免得有人"为了一致"顺手加上去，结果播报进了非流式响应正文。

        🔴 2026-08-28 改用 `source_of`（AST 取源）而非 `inspect.getsource`：
        后者按 `co_firstlineno` 从磁盘文件定位，字节码与文件不同步时会**安静地
        返回另一个函数的源码**。本条当天就这么红过一次——在
        `ledger_injection_message` 之前插了 50 行新方法，`process_message` 的旧
        行号落进了那个方法的范围，于是断言对着完全无关的源码跑。
        红了还算走运：取到相邻函数而断言恰好通过 = 静默假绿。见 `_source_probe`。
        """
        from bladex_proxy.agency import AgencyRuntime

        from _source_probe import source_of

        src = source_of(AgencyRuntime, "process_message")
        assert "run_inner_loop(" in src
        assert "on_progress" not in src


class TestNarrationProducerMarks:
    """🔴 生产侧必须**主动**带标记 —— 这是整个 gate 的依据。

    初版只测了"载体不吃掉标记"（自己传带标记的文本进去再检查），
    **没测文本确实带标记**。判别力实验当场证伪：把 `innerloop` 里的
    `NARRATE_MARK` 去掉，15 条测试**一条都不红**。
    测载体不等于测生产者——播报不带标记 ⇒ 入站剥不掉 ⇒ 污染上游上下文。
    """

    @pytest.mark.asyncio
    async def test_inner_loop_prefixes_every_narration(self):
        from bladex_proxy.innerloop import run_inner_loop
        from bladex_proxy.splice import NARRATE_MARK

        said: list[str] = []

        async def call_llm(msgs):
            return {"role": "assistant", "content": "done", "tool_calls": []}

        async def dispatch(name, args):
            return "result"

        await run_inner_loop(
            messages=[{"role": "user", "content": "hi"}],
            initial_calls=[{"id": "c1", "type": "function",
                            "function": {"name": "bladex_memory_search",
                                         "arguments": "{}"}}],
            call_llm=call_llm, dispatch=dispatch,
            on_progress=lambda t: said.append(t) or asyncio.sleep(0),
        )
        assert said, "内循环一条播报都没发"
        for t in said:
            assert t.startswith(NARRATE_MARK), f"播报没带标记，入站剥不掉: {t!r}"

    @pytest.mark.asyncio
    async def test_narration_survives_the_full_round_trip(self):
        """端到端：生产 → 载体 → agent 回带 → 入站剥净。"""
        from bladex_proxy.agency import _ChatNarrator
        from bladex_proxy.innerloop import run_inner_loop
        from bladex_proxy.splice import NARRATE_MARK, strip_inbound_echoes

        said: list[str] = []

        async def call_llm(msgs):
            return {"role": "assistant", "content": "done", "tool_calls": []}

        await run_inner_loop(
            messages=[{"role": "user", "content": "hi"}],
            initial_calls=[{"id": "c1", "type": "function",
                            "function": {"name": "bladex_ledger_read",
                                         "arguments": "{}"}}],
            call_llm=call_llm, dispatch=lambda n, a: asyncio.sleep(0, "ok"),
            on_progress=lambda t: said.append(t) or asyncio.sleep(0),
        )
        wire = _ChatNarrator().delta(said[0])[0]
        assert NARRATE_MARK in wire                      # 上线时带着
        echoed = [{"role": "assistant", "content": "x", "reasoning_content": said[0]}]
        out, n = strip_inbound_echoes(echoed)            # agent 回带后剥净
        assert n >= 1
        assert NARRATE_MARK not in str(out[0].get("reasoning_content", ""))


class TestNarrationNeverLeaksToolNames:
    """🔴 agent 看到的流里不得出现任何 `bladex_*` 标识符。

    2026-08-27 第一版播报直接发函数名（"calling bladex_ledger_update…"），
    当场打红三条既有守卫（`test_anthropic_stream_strips_bladex_tool_use` 等）——
    **那三条测试是对的**：agent 侧历史必须自洽、不出现它没见过的调用。
    ADR-0032 §3.2 #8 举的例子本来就是人话（"searching memory… step k"）。
    """

    @pytest.mark.parametrize("tool", [
        "bladex_memory_search", "bladex_ledger_read",
        "bladex_ledger_update", "bladex_ledger_switch",
        "bladex_something_new",          # 未登记 ⇒ 退化措辞，同样不漏名
    ])
    def test_phrase_contains_no_identifier(self, tool):
        from bladex_proxy.innerloop import _phrase

        p = _phrase(tool)
        assert "bladex" not in p.lower(), f"{tool} 的播报措辞漏了标识符: {p!r}"
        assert "_" not in p, f"{tool} 的措辞看着像函数名: {p!r}"

    @pytest.mark.asyncio
    async def test_end_to_end_narration_has_no_bladex_identifier(self):
        from bladex_proxy.innerloop import run_inner_loop

        said: list[str] = []

        async def call_llm(msgs):
            return {"role": "assistant", "content": "done", "tool_calls": []}

        await run_inner_loop(
            messages=[{"role": "user", "content": "hi"}],
            initial_calls=[{"id": "c1", "type": "function",
                            "function": {"name": "bladex_ledger_update",
                                         "arguments": "{}"}}],
            call_llm=call_llm, dispatch=lambda n, a: asyncio.sleep(0, "ok"),
            on_progress=lambda t: said.append(t) or asyncio.sleep(0),
        )
        assert said
        for t in said:
            body = t.replace("[bladex-live]", "")     # 标记本身不算标识符泄漏
            assert "bladex_" not in body, f"播报漏了工具原名: {t!r}"


class TestNarrationIsGatedOff:
    """🔴 播报默认关 —— 2026-08-27 事故的直接对偶。

    第一版把一个**全新、未经真实客户端验证**的机制直接接线上线：没有开关、
    没有实测。发出去的是不合协议的裸 delta（responses 缺 item 声明、
    anthropic 缺 content_block_start 且 index 与上游撞号），Codex 一次请求即断、
    无后续。**这违反的是本仓自己写了无数遍的红线**（flags.py 里 ITEM_BOUNDARY /
    LEDGER_ANCHOR / ATTRIB_L0 三处都在论证"未验证的机制默认关"）。

    协议事件序列**在沙盒里证伪不了**——这正是它能上线的原因。所以这里能钉的
    只有"默认关"与"关掉时退回已验证形态"，真正的验收必须在真机上做。
    """

    def test_default_is_off(self, monkeypatch):
        from bladex_core.flags import MEMORY_FLAG_DEFAULTS
        from bladex_proxy.agency import _narrate_on

        monkeypatch.delenv("BLADEX_NARRATE_INTERCEPT", raising=False)
        assert MEMORY_FLAG_DEFAULTS["BLADEX_NARRATE_INTERCEPT"] is False
        assert _narrate_on() is False

    def test_can_be_turned_on(self, monkeypatch):
        from bladex_proxy.agency import _narrate_on

        monkeypatch.setenv("BLADEX_NARRATE_INTERCEPT", "1")
        assert _narrate_on() is True

    @pytest.mark.asyncio
    async def test_off_degrades_to_the_verified_keepalive(self, monkeypatch):
        """关掉时产出的必须是那个**跑过真实流量**的注释行，不是半个播报。"""
        from bladex_proxy.agency import _SSE_KEEPALIVE, _narrate_on, _narrate_until_done

        monkeypatch.delenv("BLADEX_NARRATE_INTERCEPT", raising=False)
        monkeypatch.setenv("BLADEX_STREAM_KEEPALIVE_S", "5")
        assert _narrate_on() is False
        q: asyncio.Queue = asyncio.Queue()

        async def loop():
            await q.put("[bladex-live] searching memory…")
            return "ok"

        task = asyncio.ensure_future(loop())
        got = [o async for o in _narrate_until_done(task, q, narrator=None,
                                                    keepalive=_SSE_KEEPALIVE)]
        assert got and all(o == _SSE_KEEPALIVE for o in got), got


class TestNarrationAlwaysCloses:
    """🔴 开了 item/block 就**必须**收尾——否则后面的合成响应挂在未闭合的块里。

    判别力实验抓出来的缺口：把 `finally` 里的 `close()` 删掉，
    上面 30 条测试**一条都不红**。测了"开"和"delta"，没测"一定会收"。
    """

    @pytest.mark.asyncio
    async def test_close_is_emitted_after_the_loop(self, monkeypatch):
        from bladex_proxy.agency import _narrate_until_done

        monkeypatch.setenv("BLADEX_STREAM_KEEPALIVE_S", "5")
        q: asyncio.Queue = asyncio.Queue()

        async def loop():
            await q.put("[bladex-live] step 1")
            return "ok"

        events: list[str] = []

        class _Rec:
            def open(self):
                events.append("open"); return [b"OPEN"]

            def delta(self, t):
                events.append("delta"); return [b"D"]

            def close(self):
                events.append("close"); return [b"CLOSE"]

            _opened = False

        task = asyncio.ensure_future(loop())
        out = [c async for c in _narrate_until_done(task, q, narrator=_Rec(),
                                                    keepalive=None)]
        assert events[0] == "open" and events[-1] == "close", events
        assert out[-1] == b"CLOSE", out

    @pytest.mark.asyncio
    async def test_consumer_abort_does_not_raise(self, monkeypatch):
        """🔴 客户端断连时**不收尾、也不抛**。

        初版把 `close()` 放在 `finally` 里想"确保一定收尾"，结果消费方
        `aclose()` 会在 yield 点抛 `GeneratorExit`，`finally` 里再 yield
        就是 `RuntimeError: async generator ignored GeneratorExit`
        —— 一个为了"完备"写出来的、物理上不可能且有害的行为：
        **客户端都断了，收尾事件发给谁？**
        """
        from bladex_proxy.agency import _narrate_until_done

        monkeypatch.setenv("BLADEX_STREAM_KEEPALIVE_S", "0.05")
        q: asyncio.Queue = asyncio.Queue()

        async def loop():
            await q.put("a")
            await asyncio.sleep(5)
            return "ok"

        class _Rec:
            def open(self):
                return [b"OPEN"]

            def delta(self, t):
                return [b"D"]

            def close(self):
                return [b"CLOSE"]

        task = asyncio.ensure_future(loop())
        gen = _narrate_until_done(task, q, narrator=_Rec(), keepalive=None)
        async for _ in gen:
            break
        await gen.aclose()        # 不该抛
        task.cancel()

    @pytest.mark.asyncio
    async def test_never_opened_never_closes(self, monkeypatch):
        """一条播报都没有时不该凭空发一个收尾事件。"""
        from bladex_proxy.agency import _narrate_until_done

        monkeypatch.setenv("BLADEX_STREAM_KEEPALIVE_S", "5")
        q: asyncio.Queue = asyncio.Queue()
        seen: list[str] = []

        class _Rec:
            def open(self):
                seen.append("open"); return []

            def delta(self, t):
                return []

            def close(self):
                seen.append("close"); return [b"CLOSE"]

        async def loop():
            return "ok"

        task = asyncio.ensure_future(loop())
        out = [c async for c in _narrate_until_done(task, q, narrator=_Rec(),
                                                    keepalive=None)]
        assert seen == [] and out == []


class TestMatchesTheProbeReference:
    """🔴 与 `scripts/agent_probe_upstream.py`（V-R1 D7 实测过的 mock 上游）对账。

    2026-08-27 的教训：**参考实现就在仓库里，我却按协议名从头编了一遍**，
    结果发出不属于任何 item 的裸 delta。同一天第三次栽在"没先找现成的"上
    （另两次：`head -8` 截断误判 keepalive 零实现、`RequestParams.tools` 读了
    不存在的字段）。

    这组测试把两边钉在一起：参考实现改了而生产没跟上，就会红。
    """

    @staticmethod
    def _probe_src() -> str:
        from pathlib import Path
        p = Path(__file__).resolve().parents[3] / "scripts" / "agent_probe_upstream.py"
        assert p.is_file(), f"参考实现不见了：{p}（V-R1 的产物，不该被删）"
        return p.read_text(encoding="utf-8")

    def test_responses_event_names_match_the_probe(self):
        import inspect

        from bladex_proxy import agency

        probe = self._probe_src()
        src = inspect.getsource(agency._ResponsesNarrator)
        for name in ("response.output_item.added",
                     "response.reasoning_summary_text.delta",
                     "response.output_item.done"):
            assert name in probe, f"参考实现里没有 {name}（本测试的前提变了）"
            assert name in src, f"生产实现缺 {name}"

    def test_responses_delta_fields_match_the_probe(self):
        import inspect

        from bladex_proxy import agency

        probe = self._probe_src()
        src = inspect.getsource(agency._ResponsesNarrator)
        for field in ("item_id", "output_index", "summary_index"):
            assert f'"{field}"' in probe, f"参考实现里没有 {field}"
            assert f'"{field}"' in src, f"生产实现的 delta 缺 {field}"

    def test_anthropic_shape_matches_the_probe(self):
        import inspect

        from bladex_proxy import agency

        probe = self._probe_src()
        src = inspect.getsource(agency._AnthropicNarrator)
        for token in ("content_block_start", "thinking_delta", "content_block_stop"):
            assert token in probe and token in src, token
