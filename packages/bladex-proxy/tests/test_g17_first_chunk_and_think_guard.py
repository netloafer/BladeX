"""G17.0 埋点（MQ-P19 / MQ-P18）：三条入站路径的首字延迟 + 内联思考守卫。

## 为什么这张表要**逐条路径**各断言一次

`ms_first_chunk` 的缺陷形态不是"函数写错了"，是**同一件事只在一个端点做对了**：
`/v1/chat/completions` 一直有赋值，`/v1/messages`（claude-code）与
`/v1/responses`（codex）各缺一处 ⇒ Hub 里 3982 个 turn 全 0（刚性原则 13 层 3）。
只测 helper 本身抓不到这一类 —— **必须一条路径一条断言**，删掉任一处 `record_first_chunk`
调用都要有一条测试变红（阴性对照见 `test_negative_control_*`）。

守卫同理：`warn_inline_think` 是纯函数好测，但坏掉的是**接线**。
"""

from __future__ import annotations

import time

import pytest
import structlog.testing
from bladex_proxy.anthropic import anthropic_stream_generator
from bladex_proxy.capture import (
    CaptureResult,
    capture_stream,
    inline_think_leaked,
    record_first_chunk,
    warn_inline_think,
)
from bladex_proxy.responses import responses_stream_generator


class _FakeDelta:
    def __init__(self, content=None, tool_calls=None, reasoning_content=None):
        self.content = content
        self.tool_calls = tool_calls
        self.reasoning_content = reasoning_content


class _FakeChoice:
    def __init__(self, delta, finish_reason=None):
        self.delta = delta
        self.finish_reason = finish_reason


class _FakeChunk:
    def __init__(self, content=None, tool_calls=None, finish_reason=None, usage=None):
        self.choices = [_FakeChoice(_FakeDelta(content, tool_calls), finish_reason)]
        self.usage = usage

    def model_dump(self) -> dict:
        return {"choices": [{"delta": {"content": self.choices[0].delta.content}}]}


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        self._iter = iter(self._chunks)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration from None


def _chunks(*texts: str) -> list[_FakeChunk]:
    return [*[_FakeChunk(content=t) for t in texts], _FakeChunk(finish_reason="stop")]


async def _drain_messages(stream, result, model="claude-3"):
    """`/v1/messages` 生成器的调用垫（MQ-P20 之后它多了一个必填关键字参数）。"""
    async for _ in anthropic_stream_generator(stream, result, model, input_tokens_estimate=7):
        pass


async def _drain_responses(stream, result, model="gpt-5-codex"):
    async for _ in responses_stream_generator(stream, result, model):
        pass


# ── ① 首字延迟：三条路径各一条（MQ-P19） ──


@pytest.mark.asyncio
async def test_chat_completions_records_first_chunk() -> None:
    result = CaptureResult()
    async for _ in capture_stream(_FakeStream(_chunks("Hello", " world")), result):
        pass
    assert result.ms_first_chunk > 0


@pytest.mark.asyncio
async def test_messages_records_first_chunk() -> None:
    """/v1/messages —— claude-code 只走这条，此前恒 0。"""
    result = CaptureResult()
    await _drain_messages(_FakeStream(_chunks("Hello", " world")), result)
    assert result.ms_first_chunk > 0


@pytest.mark.asyncio
async def test_responses_records_first_chunk() -> None:
    """/v1/responses —— codex 只走这条，此前恒 0。"""
    result = CaptureResult()
    await _drain_responses(_FakeStream(_chunks("Hello", " world")), result)
    assert result.ms_first_chunk > 0


@pytest.mark.asyncio
async def test_first_chunk_is_first_not_last() -> None:
    """落的是**首字**不是末字：`ms_first_chunk` 必须显著早于 `ms`（幂等，只写一次）。"""
    result = CaptureResult()
    await _drain_messages(_FakeStream(_chunks(*[f"c{i}" for i in range(30)])), result)
    assert 0 < result.ms_first_chunk <= result.ms


def test_first_chunk_measures_from_upstream_call_not_generator_start() -> None:
    """🔴 MQ-P25：参照系必须是**发起上游调用那一刻**，不是生成器启动那一刻。

    async generator 创建时不执行函数体 —— 等它跑起来，`await call_model` 早已返回，
    **整段上游 TTFB 落在窗口之外**。实证差距：量错的那版全 agent p50 1.1–1.4ms，
    而同任务裸跑（tap 录音）真 TTFB p50 **1,958ms**，三个数量级。

    判别力：`t_upstream` 比 `t0` 早 50ms ⇒ 读数必须**至少**包含那 50ms。
    """
    result = CaptureResult()
    now = time.perf_counter()
    result.t_upstream = now - 0.050          # 上游调用 50ms 前发起
    record_first_chunk(result, now)          # 生成器"刚刚"才启动
    assert result.ms_first_chunk >= 50, "没把上游那段算进来 = 参照系又错了"


def test_first_chunk_falls_back_to_generator_clock_when_unset() -> None:
    """`t_upstream` 没设（测试 / 老调用方）⇒ 退回 t0，**但那是旧口径**。

    退回路径要能看出来：Hub 里 `t_upstream=0` 的轮次不可与新口径混算。
    """
    result = CaptureResult()
    record_first_chunk(result, time.perf_counter() - 0.010)
    assert 5 < result.ms_first_chunk < 5000


def test_record_first_chunk_is_idempotent() -> None:
    """哨兵语义：已落过就不再覆盖（否则"首字"会被最后一个 chunk 改写）。"""
    result = CaptureResult()
    record_first_chunk(result, 0.0)
    first = result.ms_first_chunk
    record_first_chunk(result, 0.0)
    assert result.ms_first_chunk == first


# ── ② 内联思考守卫：纯函数 + 接线（MQ-P18） ──


def test_inline_think_predicate() -> None:
    assert inline_think_leaked("<think>weighing options</think>done") is True
    assert inline_think_leaked("</think>tail only") is True
    assert inline_think_leaked("no markers here") is False
    assert inline_think_leaked("") is False


def test_warn_inline_think_logs_once_with_path() -> None:
    with structlog.testing.capture_logs() as cap:
        hit = warn_inline_think("a<think>b</think>c", "responses")
    assert hit is True
    events = [e for e in cap if e["event"] == "inline_think_leaked"]
    assert len(events) == 1
    assert events[0]["path"] == "responses"


def test_warn_inline_think_silent_on_clean_text() -> None:
    """阴性对照：干净正文一条都不许告警（否则守卫变噪声、没人再看它）。"""
    with structlog.testing.capture_logs() as cap:
        hit = warn_inline_think("clean output", "responses")
    assert hit is False
    assert [e for e in cap if e["event"] == "inline_think_leaked"] == []


@pytest.mark.asyncio
async def test_responses_stream_warns_on_leaked_think_in_text() -> None:
    """接线：`/v1/responses` 正文漏 `<think>` ⇒ `responses_capture_done` 之后有告警。"""
    result = CaptureResult()
    with structlog.testing.capture_logs() as cap:
        async for _ in responses_stream_generator(
            _FakeStream(_chunks("<think>plan", "</think>patch")), result, "gpt-5-codex"
        ):
            pass
    paths = [e["path"] for e in cap if e["event"] == "inline_think_leaked"]
    assert "responses" in paths


@pytest.mark.asyncio
async def test_responses_stream_warns_on_leaked_think_in_tool_args() -> None:
    """接线：纯工具回复（正文为空）时，污染在**工具参数**里 —— 这一格才是 `apply_patch`
    abort 的形态，只查正文会漏掉。"""

    class _Fn:
        name = "apply_patch"
        arguments = '{"patch":"</think>*** Begin Patch"}'

    class _TC:
        id = "call_1"
        index = 0
        function = _Fn()

    result = CaptureResult()
    with structlog.testing.capture_logs() as cap:
        async for _ in responses_stream_generator(
            _FakeStream([_FakeChunk(tool_calls=[_TC()]), _FakeChunk(finish_reason="tool_calls")]),
            result,
            "gpt-5-codex",
        ):
            pass
    paths = [e["path"] for e in cap if e["event"] == "inline_think_leaked"]
    assert "responses:tool_args" in paths
    assert result.full_text == ""


@pytest.mark.asyncio
async def test_clean_responses_stream_has_no_warning() -> None:
    """阴性对照（接线侧）。"""
    result = CaptureResult()
    with structlog.testing.capture_logs() as cap:
        async for _ in responses_stream_generator(
            _FakeStream(_chunks("plain answer")), result, "gpt-5-codex"
        ):
            pass
    assert [e for e in cap if e["event"] == "inline_think_leaked"] == []


# ── ③ 阴性对照：删掉赋值必须红 ──


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("module_name", "drain"),
    [
        ("bladex_proxy.anthropic", _drain_messages),
        ("bladex_proxy.responses", _drain_responses),
    ],
)
async def test_negative_control_first_chunk_wiring(monkeypatch, module_name, drain) -> None:
    """把该模块里的 `record_first_chunk` 换成 no-op ⇒ 读数必须掉回 0。

    这条不是重复上面的正向断言，而是钉住**这两个模块各自持有一个调用点**：
    monkeypatch 打的是**消费方子模块**的名字（`bladex_proxy.anthropic.record_first_chunk`），
    打门面不生效 —— 三包拆分后的共同纪律。
    """
    monkeypatch.setattr(module_name + ".record_first_chunk", lambda _r, _t: None)
    result = CaptureResult()
    await drain(_FakeStream(_chunks("x", "y")), result)
    assert result.ms_first_chunk == 0.0


# ── ④ MQ-P29：认行为不认标记（批 L，2026-09-17） ──


def test_bare_reasoning_predicate_boundary() -> None:
    """阈值取台账登记口径 `text>400`：401 命中、400 不命中；同轮有 reasoning 就不命中。"""
    from bladex_proxy.capture import BARE_REASONING_MIN_TEXT, bare_reasoning_suspected
    assert BARE_REASONING_MIN_TEXT == 400, "改阈值 = 改判据，先改台账 MQ-P29"
    assert bare_reasoning_suspected(401, 0) is True
    assert bare_reasoning_suspected(400, 0) is False
    assert bare_reasoning_suspected(5000, 1) is False
    assert bare_reasoning_suspected(0, 0) is False


def test_warn_bare_reasoning_logs_with_path_and_lengths() -> None:
    from bladex_proxy.capture import warn_bare_reasoning
    with structlog.testing.capture_logs() as cap:
        hit = warn_bare_reasoning("x" * 401, 0, "messages")
    assert hit is True
    ev = [e for e in cap if e["event"] == "bare_reasoning_suspected"]
    assert len(ev) == 1 and ev[0]["path"] == "messages" and ev[0]["text_len"] == 401 and ev[0]["reasoning_len"] == 0


def test_warn_bare_reasoning_silent_when_reasoning_present_or_text_short() -> None:
    """阴性对照：有 thinking 的长正文、无 thinking 的短正文都不许响。"""
    from bladex_proxy.capture import warn_bare_reasoning
    with structlog.testing.capture_logs() as cap:
        assert warn_bare_reasoning("x" * 5000, 120, "messages") is False
        assert warn_bare_reasoning("x" * 100, 0, "messages") is False
    assert [e for e in cap if e["event"] == "bare_reasoning_suspected"] == []


def _long_text_no_reasoning() -> list[_FakeChunk]:
    """H4 那 11 轮的形状：正文 >400 字符、`reasoning_content` 全程为空、**无任何 think 标记**。"""
    return _chunks(*["step %d of the plan is to inspect the module and " % i for i in range(12)])


def _long_text_with_reasoning() -> list[_FakeChunk]:
    first = _FakeChunk()
    first.choices[0].delta.reasoning_content = "weighing options"
    return [first, *_long_text_no_reasoning()]


@pytest.mark.asyncio
async def test_messages_stream_warns_on_bare_reasoning_without_markers() -> None:
    """接线（/v1/messages）：这正是 P18 守卫**看不见**的那一格 —— 零标记也要响。"""
    result = CaptureResult()
    with structlog.testing.capture_logs() as cap:
        await _drain_messages(_FakeStream(_long_text_no_reasoning()), result)
    assert len(result.full_text) > 400 and result.reasoning_text == ""
    assert [e for e in cap if e["event"] == "inline_think_leaked"] == [], "标记守卫对它本来就失明——这是它存在的理由"
    assert [e["path"] for e in cap if e["event"] == "bare_reasoning_suspected"] == ["messages"]


@pytest.mark.asyncio
async def test_responses_stream_warns_on_bare_reasoning_without_markers() -> None:
    result = CaptureResult()
    with structlog.testing.capture_logs() as cap:
        await _drain_responses(_FakeStream(_long_text_no_reasoning()), result)
    assert [e["path"] for e in cap if e["event"] == "bare_reasoning_suspected"] == ["responses"]


@pytest.mark.asyncio
async def test_chat_stream_warns_on_bare_reasoning_without_markers() -> None:
    result = CaptureResult()
    with structlog.testing.capture_logs() as cap:
        async for _ in capture_stream(_FakeStream(_long_text_no_reasoning()), result):
            pass
    assert [e["path"] for e in cap if e["event"] == "bare_reasoning_suspected"] == ["chat_completions"]


@pytest.mark.asyncio
@pytest.mark.parametrize("drain", [_drain_messages, _drain_responses])
async def test_long_text_with_structured_reasoning_is_not_flagged(drain) -> None:
    """阴性对照（接线侧）：同轮走了 `reasoning_content` 的长正文是正常形态，一条都不许响。"""
    result = CaptureResult()
    with structlog.testing.capture_logs() as cap:
        await drain(_FakeStream(_long_text_with_reasoning()), result)
    assert result.reasoning_text
    assert [e for e in cap if e["event"] == "bare_reasoning_suspected"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("module_name", "drain"),
    [("bladex_proxy.anthropic", _drain_messages), ("bladex_proxy.responses", _drain_responses)],
)
async def test_negative_control_bare_reasoning_wiring(monkeypatch, module_name, drain) -> None:
    """删掉该模块的调用点 ⇒ 告警必须消失：钉住每条路径各自持有一个接线点（打消费方子模块，打门面不生效）。"""
    monkeypatch.setattr(module_name + ".warn_bare_reasoning", lambda *_a, **_k: False)
    result = CaptureResult()
    with structlog.testing.capture_logs() as cap:
        await drain(_FakeStream(_long_text_no_reasoning()), result)
    assert [e for e in cap if e["event"] == "bare_reasoning_suspected"] == []
