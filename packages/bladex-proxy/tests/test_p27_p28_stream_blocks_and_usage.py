"""MQ-P27（`usage` 语义）与 MQ-P28（工具块串行）—— 流式 Anthropic 回程。

## 两条缺陷的形状（决定了每条断言的判别力）

**MQ-P28**：Anthropic 流式协议里内容块不允许嵌套（`start(i)` → deltas → `stop(i)`
→ `start(i+1)`）。此前每来一个新的上游 tool index 就直接发 `content_block_start`、
不关上一个，全部 `stop` 堆在收尾。实测 H4-P28 112 轮：**多工具轮 25，块嵌套轮 25，
两个集合逐轮重合**；单工具 85 轮结构全对。

🔴 **本条是协议缺陷，不要给它挂用户侧症状**（2026-09-10 更正）：曾把
「CC 终端 `Update` 不折叠」当成它的症状写在这里，**那个归因是错的** ——
逐字转发的 tap 代理（内容零改动）同样复现不折叠，⇒ 症状与内容转换无关。
修它的理由**只是**「Anthropic 协议禁止块嵌套」，不需要也不应该借用户现象背书。

⇒ **本文件的主断言是"任何时刻至多一个块开着"**，不是"发了几个 stop"：
数 stop 的个数对嵌套完全失明（旧实现的 stop 个数是对的，位置是错的）。

**MQ-P27**：同一个字段名在两个端点上语义相反 —— Ark `/v3` 的 `prompt_tokens` 是
**含缓存的总数**，Anthropic 协议的 `input_tokens` 是**未缓存部分**。此前把总数原样填进
`input_tokens` 且不发 `cache_read_input_tokens`，两个错互相抵消，总量恰好是对的
⇒ **P20 的压缩阈值没被污染**（三跑 168,164 / 167,956 / 167,224）。
🔴 所以本文件必须钉的是**不变量**：`input + cache_read == prompt_tokens`。
只断言"cache_read 出现了"会放过"总量翻倍"这个真正的风险。
"""

from __future__ import annotations

import json

import pytest
from bladex_proxy.anthropic import anthropic_stream_generator
from bladex_proxy.capture import CaptureResult


class _Fn:
    def __init__(self, name=None, arguments=None):
        self.name = name
        self.arguments = arguments


class _TC:
    def __init__(self, index, id=None, name=None, arguments=None):  # noqa: A002
        self.index = index
        self.id = id
        self.function = _Fn(name, arguments)


class _Delta:
    def __init__(self, content=None, tool_calls=None, reasoning_content=None):
        self.content = content
        self.tool_calls = tool_calls
        self.reasoning_content = reasoning_content


class _Choice:
    def __init__(self, delta, finish_reason=None):
        self.delta = delta
        self.finish_reason = finish_reason


class _Usage:
    def __init__(self, prompt_tokens, completion_tokens=0, cache_read=None):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        if cache_read is not None:
            self.cache_read_input_tokens = cache_read


class _Chunk:
    def __init__(self, *, content=None, tool_calls=None, reasoning_content=None,
                 finish_reason=None, usage=None):
        self.choices = [_Choice(_Delta(content, tool_calls, reasoning_content),
                                finish_reason)]
        self.usage = usage


class _Stream:
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


async def _events(chunks, *, estimate: int = 100) -> list[dict]:
    result = CaptureResult()
    raw = b""
    async for part in anthropic_stream_generator(
        _Stream(chunks), result, "claude-3", input_tokens_estimate=estimate
    ):
        raw += part
    out: list[dict] = []
    for block in raw.decode().split("\n\n"):
        for line in block.split("\n"):
            if line.startswith("data: "):
                out.append(json.loads(line[6:]))
    return out


def _assert_blocks_strictly_serial(events: list[dict]) -> list[str]:
    """🔴 主断言：任何时刻至多一个块开着，且每个 stop 对应当前开着的那个 index。

    返回块类型的出现顺序，便于调用方再断言内容。
    **不数 stop 的个数** —— 旧实现 stop 个数是对的、位置是错的，计数对它失明。
    """
    open_idx: int | None = None
    order: list[str] = []
    for e in events:
        if e.get("type") == "content_block_start":
            assert open_idx is None, (
                f"块嵌套：index={e['index']} 在 index={open_idx} 未 stop 时就 start —— "
                "这正是 MQ-P28 的缺陷形状")
            open_idx = e["index"]
            order.append(e["content_block"]["type"])
        elif e.get("type") == "content_block_stop":
            assert open_idx == e["index"], (
                f"stop 的 index={e['index']} 与当前开着的 index={open_idx} 不符")
            open_idx = None
        elif e.get("type") == "content_block_delta":
            assert open_idx == e["index"], (
                f"delta 落在未开启的块 index={e['index']}（当前开着 {open_idx}）")
    assert open_idx is None, f"块 index={open_idx} 从未 stop"
    return order


# ── MQ-P28 ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_two_tool_calls_do_not_nest_blocks() -> None:
    """本条是 MQ-P28 的正面复现：两个工具，旧实现在这里嵌套。"""
    events = await _events([
        _Chunk(tool_calls=[_TC(0, id="c0", name="Read", arguments='{"p":')]),
        _Chunk(tool_calls=[_TC(0, arguments='"a.py"}')]),
        _Chunk(tool_calls=[_TC(1, id="c1", name="Edit", arguments='{"p":"b.py"}')]),
        _Chunk(finish_reason="tool_calls"),
    ])
    assert _assert_blocks_strictly_serial(events) == ["tool_use", "tool_use"]


@pytest.mark.asyncio
async def test_four_tool_calls_stay_serial() -> None:
    """实测分布里出现过一轮 4 个工具块（`{0:2, 1:85, 2:22, 3:2, 4:1}`）。"""
    events = await _events([
        _Chunk(tool_calls=[_TC(i, id=f"c{i}", name=f"T{i}", arguments="{}")])
        for i in range(4)
    ] + [_Chunk(finish_reason="tool_calls")])
    assert _assert_blocks_strictly_serial(events) == ["tool_use"] * 4


@pytest.mark.asyncio
async def test_single_tool_call_still_streams_incrementally() -> None:
    """🔴 单工具轮（实测 85/112 = 76%）行为**逐字不变**：参数仍逐段 `input_json_delta`。

    这条钉的是修法的**边界**：不许为了修多工具轮而把多数路径改成"收尾一次性发"
    （那是被否掉的候选②）。
    """
    events = await _events([
        _Chunk(tool_calls=[_TC(0, id="c0", name="Read", arguments='{"p":')]),
        _Chunk(tool_calls=[_TC(0, arguments='"a"')]),
        _Chunk(tool_calls=[_TC(0, arguments="}")]),
        _Chunk(finish_reason="tool_calls"),
    ])
    _assert_blocks_strictly_serial(events)
    deltas = [e for e in events if e.get("type") == "content_block_delta"]
    assert len(deltas) == 3, "单工具必须保持逐段流式，不许合并成一发"
    assert "".join(d["delta"]["partial_json"] for d in deltas) == '{"p":"a"}'


@pytest.mark.asyncio
async def test_buffered_tool_arguments_are_not_lost() -> None:
    """🔴 无损：第二个工具的参数分多段到达，收尾必须一段不少地发出去。

    候选①（来新 index 就关上一个）在这里会丢参数 —— 落在已关闭块上的 delta 无处可发。
    """
    events = await _events([
        _Chunk(tool_calls=[_TC(0, id="c0", name="Read", arguments="{}")]),
        _Chunk(tool_calls=[_TC(1, id="c1", name="Edit", arguments='{"a":1,')]),
        _Chunk(tool_calls=[_TC(1, arguments='"b":2,')]),
        _Chunk(tool_calls=[_TC(1, arguments='"c":3}')]),
        _Chunk(finish_reason="tool_calls"),
    ])
    _assert_blocks_strictly_serial(events)
    starts = [e for e in events if e.get("type") == "content_block_start"]
    second = starts[1]["index"]
    got = "".join(e["delta"]["partial_json"] for e in events
                  if e.get("type") == "content_block_delta" and e["index"] == second)
    assert json.loads(got) == {"a": 1, "b": 2, "c": 3}
    assert starts[1]["content_block"]["name"] == "Edit"
    assert starts[1]["content_block"]["id"] == "c1"


@pytest.mark.asyncio
async def test_interleaved_tool_arguments_stay_serial_and_lossless() -> None:
    """🔴 交错到达 —— 实测 0/25，但 OpenAI 语义允许，**修法不能依赖那个观察**。

    候选①在这里既嵌套又丢参数；本实现两样都不发生。
    """
    events = await _events([
        _Chunk(tool_calls=[_TC(0, id="c0", name="Read", arguments='{"x":')]),
        _Chunk(tool_calls=[_TC(1, id="c1", name="Edit", arguments='{"y":')]),
        _Chunk(tool_calls=[_TC(0, arguments="1}")]),
        _Chunk(tool_calls=[_TC(1, arguments="2}")]),
        _Chunk(finish_reason="tool_calls"),
    ])
    _assert_blocks_strictly_serial(events)
    starts = [e for e in events if e.get("type") == "content_block_start"]
    per_block: dict[int, str] = {}
    for e in events:
        if e.get("type") == "content_block_delta":
            per_block[e["index"]] = per_block.get(e["index"], "") + \
                e["delta"]["partial_json"]
    assert json.loads(per_block[starts[0]["index"]]) == {"x": 1}
    assert json.loads(per_block[starts[1]["index"]]) == {"y": 2}


@pytest.mark.asyncio
async def test_thinking_then_text_then_tools_stay_serial() -> None:
    """三种块混排（实测最常见形态 `['thinking','text','tool_use','tool_use']`）。"""
    events = await _events([
        _Chunk(reasoning_content="想"),
        _Chunk(content="答"),
        _Chunk(tool_calls=[_TC(0, id="c0", name="Read", arguments="{}")]),
        _Chunk(tool_calls=[_TC(1, id="c1", name="Edit", arguments="{}")]),
        _Chunk(finish_reason="tool_calls"),
    ])
    assert _assert_blocks_strictly_serial(events) == [
        "thinking", "text", "tool_use", "tool_use"]


# ── MQ-P27 ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_usage_splits_cached_from_uncached_and_total_is_unchanged() -> None:
    """🔴 不变量：`input_tokens + cache_read_input_tokens == prompt_tokens`。

    总量不变 ⇒ 已兑现的 MQ-P20 压缩阈值不回退。只断言"cache_read 出现了"
    会放过"总量翻倍"这个真正的风险（那正是 P27 的引信）。
    """
    events = await _events(
        [_Chunk(content="hi"),
         _Chunk(finish_reason="stop", usage=_Usage(167224, 42, cache_read=163840))],
        estimate=150000,
    )
    usage = [e for e in events if e.get("type") == "message_delta"][0]["usage"]
    assert usage["cache_read_input_tokens"] == 163840
    assert usage["input_tokens"] == 167224 - 163840
    assert usage["input_tokens"] + usage["cache_read_input_tokens"] == 167224
    assert usage["output_tokens"] == 42


@pytest.mark.asyncio
async def test_usage_omits_cache_field_when_upstream_does_not_report_it() -> None:
    """上游不报缓存 ⇒ 不发这个字段，行为与改动前**逐字一致**。

    「上游不报」与「报了且零命中」是两件事 —— 发一个 0 会让前者伪装成后者。
    """
    events = await _events(
        [_Chunk(content="hi"), _Chunk(finish_reason="stop", usage=_Usage(9000, 7))],
        estimate=100,
    )
    usage = [e for e in events if e.get("type") == "message_delta"][0]["usage"]
    assert usage == {"input_tokens": 9000, "output_tokens": 7}
    assert "cache_read_input_tokens" not in usage


@pytest.mark.asyncio
async def test_usage_zero_cache_read_is_reported_as_zero_not_omitted() -> None:
    """报了且零命中 ⇒ 字段在、值为 0（与"不报"必须能分开）。"""
    events = await _events(
        [_Chunk(finish_reason="stop", usage=_Usage(500, 1, cache_read=0))],
        estimate=100,
    )
    usage = [e for e in events if e.get("type") == "message_delta"][0]["usage"]
    assert usage["cache_read_input_tokens"] == 0
    assert usage["input_tokens"] == 500


@pytest.mark.asyncio
async def test_cache_read_exceeding_prompt_tokens_clamps_and_never_goes_negative() -> None:
    """上游 usage 自相矛盾时宁可 input_tokens=0，也不发负数（会让客户端算术崩掉）。"""
    events = await _events(
        [_Chunk(finish_reason="stop", usage=_Usage(100, 1, cache_read=999))],
        estimate=100,
    )
    usage = [e for e in events if e.get("type") == "message_delta"][0]["usage"]
    assert usage["input_tokens"] == 0
    assert usage["cache_read_input_tokens"] == 999


@pytest.mark.asyncio
async def test_message_start_carries_no_cache_field() -> None:
    """`message_start` 先于任何 usage chunk 发出 ⇒ 只能带估计值，且不谈缓存。"""
    events = await _events(
        [_Chunk(finish_reason="stop", usage=_Usage(9000, 1, cache_read=8000))],
        estimate=8888,
    )
    usage = [e for e in events if e.get("type") == "message_start"][0]["message"]["usage"]
    assert usage == {"input_tokens": 8888, "output_tokens": 0}


# ── 阴性对照：守卫必须能在**旧形态**上炸 ────────────────────────────────────


def test_serial_guard_rejects_the_old_nested_shape() -> None:
    """🔴 没有这条，`_assert_blocks_strictly_serial` 只是在陪跑。

    这里逐字复刻 H4-P28 录音里抓到的旧轨迹
    （`content_block_start:tool_use → input_json_delta×45 →
    content_block_start:tool_use → … → content_block_stop×2`），
    守卫必须拒绝它。**注意 stop 的个数是对的** —— 所以"数 stop"那种写法
    对这个缺陷完全失明，这也是本文件不数 stop 的理由。
    """
    old_shape = [
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "tool_use"}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "input_json_delta", "partial_json": "{}"}},
        {"type": "content_block_start", "index": 1,
         "content_block": {"type": "tool_use"}},
        {"type": "content_block_delta", "index": 1,
         "delta": {"type": "input_json_delta", "partial_json": "{}"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_stop", "index": 1},
    ]
    assert sum(1 for e in old_shape if e["type"] == "content_block_start") == \
        sum(1 for e in old_shape if e["type"] == "content_block_stop"), \
        "旧形态的 start/stop 个数是配平的 —— 计数式断言抓不到它"
    with pytest.raises(AssertionError, match="块嵌套"):
        _assert_blocks_strictly_serial(old_shape)


def test_serial_guard_rejects_a_never_closed_block() -> None:
    with pytest.raises(AssertionError, match="从未 stop"):
        _assert_blocks_strictly_serial([
            {"type": "content_block_start", "index": 0,
             "content_block": {"type": "text"}}])
