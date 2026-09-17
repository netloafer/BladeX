"""MQ-P31：`/v1/responses` 的 usage / 缓存读数埋点（2026-09-10 立）。

## 这张表守的是"仪器缺口"，不是"函数写错了"

立卡的读数：codex 同一张任务卡、同一模型、同一 SHA、两侧都单跑，
**裸跑 69 轮 / 19分01秒 / 零压缩**，**经 BladeX 178 轮 / 29分55秒 / 两次压缩**，
而 **Σ 上游几乎相同（12.6 vs 12.7 分钟）** ⇒ 多出来的 11 分钟不是"上游变慢"，
是"多跑了 109 轮"，且那 109 轮里夹着两次压缩。

要判"BladeX 是否把上下文推过了 codex 的压缩阈值"，需要 token 数。
裸跑侧 tap 有（`input_tokens` / `input_tokens_details.cached_tokens`），
**BladeX 侧一个都没有** —— 因为 `responses_capture_done` 压根不记这几格。
于是当前最大的开放问题**不可判**。

🔴 **注意这里差一点犯的错**：最初我把"`responses_capture_done` 的字段表里没有 usage"
读成了"usage 是空的"。那是两件事 —— 前者是**尺子上没有这一格**，后者是**读数为零**。
`result.usage` 其实一直被 MQ-P22 灌着。**任何桶读数为 0，先证明那个桶可达。**

## 为什么阴性对照是这张表的主体

缺陷形态是**接线**（日志少几个 kwarg），不是纯函数。测 `cache_read_tokens()` 本身
一条都抓不到它。所以每一格都必须有一条"把那个 kwarg 删掉就变红"的断言。

## 单一实现点

`cache_read_tokens` 从 `anthropic.py` 搬进 `capture.py`（三条路径共用）。
`anthropic._cache_read_tokens` 保留为别名 —— 既有测试与 monkeypatch 目标不动。
这一条也各有一测：搬完之后两个名字必须是**同一个对象**，不是两份实现。
"""

from __future__ import annotations

import pytest
import structlog.testing
from bladex_proxy import anthropic as anthropic_mod
from bladex_proxy.capture import CaptureResult, cache_read_tokens
from bladex_proxy.responses import responses_stream_generator

# ── 假上游（与 test_g17_* 同款，故意不共享：那张表在守别的东西，
#    共享 fixture 会让"改一处红两张表"变成噪声）──


class _FakeDelta:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls
        self.reasoning_content = None


class _FakeChoice:
    def __init__(self, delta, finish_reason=None):
        self.delta = delta
        self.finish_reason = finish_reason


class _FakeChunk:
    def __init__(self, content=None, finish_reason=None, usage=None):
        self.choices = [_FakeChoice(_FakeDelta(content), finish_reason)]
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


class _Usage:
    """属性载体（litellm 的常见形态）。"""

    def __init__(self, prompt=0, completion=0, cached=None, flavour="openai"):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = prompt + completion
        if cached is not None:
            if flavour == "openai":
                self.prompt_tokens_details = type("D", (), {"cached_tokens": cached})()
            else:
                self.cache_read_input_tokens = cached


async def _run(chunks) -> list[dict]:
    """跑一遍流式生成器，返回 `responses_capture_done` 那一条日志。"""
    result = CaptureResult()
    with structlog.testing.capture_logs() as cap:
        async for _ in responses_stream_generator(_FakeStream(chunks), result, "gpt-5-codex"):
            pass
    return [e for e in cap if e["event"] == "responses_capture_done"]


# ── ① 四格必须在（阴性对照：删掉任一 kwarg 这里就红）──


@pytest.mark.asyncio
async def test_capture_done_carries_all_four_usage_fields() -> None:
    events = await _run([
        _FakeChunk(content="hi"),
        _FakeChunk(finish_reason="stop", usage=_Usage(prompt=1200, completion=34, cached=1100)),
    ])
    assert len(events) == 1
    ev = events[0]
    for field in ("usage", "input_tokens_reported", "output_tokens", "cache_read_tokens"):
        assert field in ev, f"MQ-P31：`{field}` 不在 responses_capture_done 里"


@pytest.mark.asyncio
async def test_reported_tokens_match_upstream() -> None:
    ev = (await _run([
        _FakeChunk(content="x"),
        _FakeChunk(finish_reason="stop", usage=_Usage(prompt=1200, completion=34, cached=1100)),
    ]))[0]
    assert ev["input_tokens_reported"] == 1200
    assert ev["output_tokens"] == 34
    assert ev["cache_read_tokens"] == 1100


@pytest.mark.asyncio
async def test_usage_dict_and_reported_are_independent_readouts() -> None:
    """两格并列是故意的：一格空另一格非零 ⇒ 是提取器漏了，不是上游没报。"""
    ev = (await _run([
        _FakeChunk(content="x"),
        _FakeChunk(finish_reason="stop", usage=_Usage(prompt=9, completion=2)),
    ]))[0]
    assert ev["usage"] == {"prompt": 9, "completion": 2, "total": 11}
    assert ev["input_tokens_reported"] == 9


# ── ② None ≠ 0（这一条错了，"这条路径测不了缓存"会伪装成"缓存一直没命中"）──


@pytest.mark.asyncio
async def test_cache_read_is_none_when_upstream_does_not_report() -> None:
    ev = (await _run([
        _FakeChunk(content="x"),
        _FakeChunk(finish_reason="stop", usage=_Usage(prompt=9, completion=2)),
    ]))[0]
    assert ev["cache_read_tokens"] is None, "上游不报必须是 None，不能是 0"


@pytest.mark.asyncio
async def test_cache_read_zero_is_kept_as_zero() -> None:
    ev = (await _run([
        _FakeChunk(content="x"),
        _FakeChunk(finish_reason="stop", usage=_Usage(prompt=9, completion=2, cached=0)),
    ]))[0]
    assert ev["cache_read_tokens"] == 0, "报了且零命中必须是 0，不能塌成 None"


@pytest.mark.asyncio
async def test_no_usage_chunk_at_all_leaves_fields_at_defaults() -> None:
    """上游全程不发 usage（没带 stream_options 的形态）：不许抛，读数如实为空。"""
    ev = (await _run([_FakeChunk(content="x"), _FakeChunk(finish_reason="stop")]))[0]
    assert ev["input_tokens_reported"] == 0
    assert ev["output_tokens"] == 0
    assert ev["cache_read_tokens"] is None
    assert ev["usage"] == {}


# ── ③ 后到的 usage 不许被后续空 usage 抹掉 ──


@pytest.mark.asyncio
async def test_later_empty_usage_does_not_erase_cache_read() -> None:
    ev = (await _run([
        _FakeChunk(content="x"),
        _FakeChunk(usage=_Usage(prompt=100, completion=5, cached=80)),
        _FakeChunk(finish_reason="stop"),
    ]))[0]
    assert ev["cache_read_tokens"] == 80


# ── ④ 两家写法都认（BladeX 后面挂什么上游都可能）──


def test_cache_read_reads_anthropic_native_shape() -> None:
    assert cache_read_tokens(_Usage(prompt=10, completion=1, cached=7, flavour="anthropic")) == 7


def test_cache_read_reads_dict_carrier_with_nested_details() -> None:
    assert cache_read_tokens({"prompt_tokens_details": {"cached_tokens": 42}}) == 42


def test_cache_read_reads_flat_cached_tokens() -> None:
    assert cache_read_tokens({"cached_tokens": 5}) == 5


def test_cache_read_none_usage_is_none() -> None:
    assert cache_read_tokens(None) is None


# ── ⑤ 单一实现点：搬家后两个名字必须是同一个对象 ──


def test_anthropic_alias_is_the_same_object_not_a_second_copy() -> None:
    assert anthropic_mod._cache_read_tokens is cache_read_tokens
