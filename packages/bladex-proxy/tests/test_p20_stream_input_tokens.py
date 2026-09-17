"""MQ-P20：流式 Anthropic 回程的 `input_tokens` 不再恒为 0。

## 这条缺陷的形状（决定了下面每一条断言的判别力）

`message_start.usage.input_tokens` **硬编码 0**，而 claude-code **只走流式** ⇒
它拿不到输入用量 ⇒ **永不触发上下文压缩**。实证：同一份默认配置、同一张任务卡，
裸跑第 20 分钟压缩会话并跑完；经 BladeX 全程零压缩、上下文涨到 39 万字符
（`msg_count 3→167` 单调递增、零回落）。
而**同一文件的非流式路径一直填对着** —— 同一字段、同一文件、两条路径一对一错
（刚性原则 12）。所以测试必须钉在**流式**这一侧。

🔴 **本文件证明不了"CC 因此会压缩"**。那一步只有 live bench 能证
（判据 = 同一张任务卡上 `msg_count` 是否出现回落；不回落 ⇒ P20 的立论撤回）。
这里钉的是它的**必要条件**：我们报出去的那个数不再是 0，且与上游真值一致。
"""

from __future__ import annotations

import json

import pytest
from bladex_proxy.anthropic import anthropic_stream_generator
from bladex_proxy.capture import CaptureResult


class _FakeDelta:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _FakeChoice:
    def __init__(self, delta, finish_reason=None):
        self.delta = delta
        self.finish_reason = finish_reason


class _FakeUsage:
    def __init__(self, prompt_tokens, completion_tokens=0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _FakeChunk:
    def __init__(self, content=None, finish_reason=None, usage=None):
        self.choices = [_FakeChoice(_FakeDelta(content), finish_reason)]
        self.usage = usage


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


async def _events(chunks, *, estimate: int) -> list[dict]:
    """跑一遍生成器，把 SSE 还原成 event dict 列表。"""
    result = CaptureResult()
    raw = b""
    async for part in anthropic_stream_generator(
        _FakeStream(chunks), result, "claude-3", input_tokens_estimate=estimate
    ):
        raw += part
    out: list[dict] = []
    for block in raw.decode().split("\n\n"):
        for line in block.split("\n"):
            if line.startswith("data: "):
                out.append(json.loads(line[6:]))
    return out


def _one(events: list[dict], etype: str) -> dict:
    hits = [e for e in events if e.get("type") == etype]
    assert len(hits) == 1, f"{etype} 应恰好一条，实得 {len(hits)}"
    return hits[0]


@pytest.mark.asyncio
async def test_message_start_reports_estimate_not_zero() -> None:
    """🔴 本卡的核心：`message_start.usage.input_tokens` 不再是硬编码 0。"""
    events = await _events([_FakeChunk(content="hi"), _FakeChunk(finish_reason="stop")],
                           estimate=12345)
    usage = _one(events, "message_start")["message"]["usage"]
    assert usage["input_tokens"] == 12345
    assert usage["output_tokens"] == 0


@pytest.mark.asyncio
async def test_upstream_truth_overrides_estimate_in_message_delta() -> None:
    """上游报了真值就以真值为准（`message_start` 先发，只能带估计值）。"""
    events = await _events(
        [_FakeChunk(content="hi"),
         _FakeChunk(finish_reason="stop", usage=_FakeUsage(9000, 42))],
        estimate=12345,
    )
    assert _one(events, "message_start")["message"]["usage"]["input_tokens"] == 12345
    delta_usage = _one(events, "message_delta")["usage"]
    assert delta_usage["input_tokens"] == 9000, "上游真值必须覆盖估计值"
    assert delta_usage["output_tokens"] == 42


@pytest.mark.asyncio
async def test_upstream_none_does_not_wipe_the_estimate() -> None:
    """🔴 回归钉：`usage.prompt_tokens = None` 不许把估计值抹回 0/None。

    `getattr(obj, "prompt_tokens", 默认)` 在**字段存在且为 None** 时返回 None ——
    这正是"上游报了一个空 usage"的常见形态（Router 归一后各家都可能这样）。
    照原写法一路赋回去，客户端又读不到用量了，缺陷原样复活。
    """
    events = await _events(
        [_FakeChunk(content="hi"),
         _FakeChunk(finish_reason="stop", usage=_FakeUsage(None, None))],
        estimate=777,
    )
    assert _one(events, "message_delta")["usage"]["input_tokens"] == 777


@pytest.mark.asyncio
async def test_message_delta_carries_input_tokens_when_upstream_silent() -> None:
    """上游整轮不报 usage 时，两处读数必须一致（否则客户端两次读数打架）。"""
    events = await _events([_FakeChunk(content="hi"), _FakeChunk(finish_reason="stop")],
                           estimate=555)
    assert _one(events, "message_start")["message"]["usage"]["input_tokens"] == 555
    assert _one(events, "message_delta")["usage"]["input_tokens"] == 555


@pytest.mark.asyncio
async def test_p21_responses_none_usage_does_not_crash_the_tail() -> None:
    """🔴 MQ-P21（同族，另一条路径）：`/v1/responses` 上游报空 usage 不许炸尾。

    `getattr(usage, "prompt_tokens", 默认)` 在字段存在且为 None 时返回 None ⇒
    收尾处 `input_tokens + output_tokens` **TypeError**，而那一行在 try 之外、
    在 `response.completed` 之前 ⇒ **codex 收不到终止事件**。
    判别力：把 isinstance 守卫去掉，这条必红（TypeError 而不是断言失败）。
    """
    from bladex_proxy.responses import responses_stream_generator

    result = CaptureResult()
    raw = b""
    async for part in responses_stream_generator(
        _FakeStream([_FakeChunk(content="hi"),
                     _FakeChunk(finish_reason="stop", usage=_FakeUsage(None, None))]),
        result, "gpt-5-codex",
    ):
        raw += part
    assert "response.completed" in raw.decode()


@pytest.mark.asyncio
async def test_p22_messages_path_fills_finish_reason_and_usage() -> None:
    """🔴 MQ-P22：两条流式路径此前**没有** `finish_reason` / `usage` 的生产者。

    后果不是"少两个字段"，是 **MQ-A34 的截断告警结构性失明**：
    `_warn_if_output_truncated` 读的就是 `response_meta.finish_reason`，
    而它在 claude-code 的流式轮上恒空 ⇒ 「上游把 CC 的输出截断了」这件事
    **在日志里永远不会出现**。live 实测（09-01 起）：
    claude-code 899 轮 27% 有值、codex 393 轮 7% 有值，那点非零全是非流式轮；
    同期 hermes 97% / Pi 100%（它们走 `/v1/chat/completions`，一直有生产者）。

    落**上游原值**而不是映射后的 Anthropic 词，因为 `OUTPUT_TRUNCATED_REASONS`
    认的是 `length`。
    """
    result = CaptureResult()
    async for _ in anthropic_stream_generator(
        _FakeStream([_FakeChunk(content="hi"),
                     _FakeChunk(finish_reason="length", usage=_FakeUsage(120, 34))]),
        result, "claude-3", input_tokens_estimate=1,
    ):
        pass
    from bladex_proxy.capture import output_truncated

    assert result.finish_reason == "length", "必须是上游原值，不是 max_tokens/end_turn"
    assert output_truncated(result.finish_reason) is True, "A34 告警的判据必须能命中"
    assert result.usage == {"prompt": 120, "completion": 34, "total": 0}


@pytest.mark.asyncio
async def test_p22_responses_path_fills_finish_reason_and_usage() -> None:
    """同上，`/v1/responses`（codex 只走这条）。"""
    from bladex_proxy.responses import responses_stream_generator

    result = CaptureResult()
    async for _ in responses_stream_generator(
        _FakeStream([_FakeChunk(content="hi"),
                     _FakeChunk(finish_reason="length", usage=_FakeUsage(7, 8))]),
        result, "gpt-5-codex",
    ):
        pass
    assert result.finish_reason == "length"
    assert result.usage["prompt"] == 7 and result.usage["completion"] == 8


def test_input_tokens_estimate_is_required_keyword() -> None:
    """🔴 无默认值：缺陷的形状就是"这个数悄悄是 0"，给默认值等于留给下一个调用点。

    （原则 12：同一参数在两条调用路径上各有一个默认值 = 缺陷。）
    """
    import inspect

    sig = inspect.signature(anthropic_stream_generator)
    p = sig.parameters["input_tokens_estimate"]
    assert p.kind is inspect.Parameter.KEYWORD_ONLY
    assert p.default is inspect.Parameter.empty


# ── MQ-P23：流式 usage 要主动开（否则上游根本不发那个 chunk）──


def test_p23_stream_options_included_in_upstream_call():
    """🔴 OpenAI 兼容的流式响应，**只有请求带 `stream_options.include_usage` 才回 usage**。

    全仓此前没有任何地方设它 ⇒ `chunk.usage` 恒 falsy ⇒ `extract_usage` 从未被调用
    ⇒ `response_meta.usage` 在 claude-code / codex 上恒空（hermes 有 97% 是因为
    **客户端自己带**——那两条路径的上游请求是 BladeX 自己拼的）。

    落在 `_do_call` 是因为它是**唯一的转发点**（三条入站路径共用）。
    判别力：把那两行删掉，本条立刻红。
    """
    import inspect

    from bladex_proxy import route

    src = inspect.getsource(route._do_call)
    assert 'call_kwargs["stream_options"] = {"include_usage": True}' in src
    assert 'if stream and "stream_options" not in call_kwargs:' in src, \
        "必须只在流式加、且调用方显式给了就尊重"


def test_p23_is_the_precondition_for_cross_side_comparison():
    """把"为什么非做不可"钉在测试里：它不是可选的埋点。

    两边协议不同（Anthropic 事件 vs OpenAI chunk）⇒ `ms/chunk` 不可比；
    协议无关、两边都能算的只有 **TTFB**（MQ-P25）与 **output_tokens/s**（本条）。
    ⇒ 少了它，"BladeX vs 裸跑"的耗时对照**根本立不起来**（MQ-V25 同族）。
    """
    from bladex_core.flags import MEMORY_NUMERIC_DEFAULTS

    # 它是无条件行为、不是可调旋钮——别有人日后把它做成默认关的开关。
    assert not any("STREAM_OPTIONS" in k or "INCLUDE_USAGE" in k
                   for k in MEMORY_NUMERIC_DEFAULTS)
