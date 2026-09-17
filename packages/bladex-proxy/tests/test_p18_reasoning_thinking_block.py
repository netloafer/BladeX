"""MQ-P26：上游 `reasoning_content` → Anthropic `thinking` 块（`/v1/messages`）。

## 为什么是这个修法，而不是在正文里剥 `</think>`

三条读数把落点定死了：

1. **CC 自己要的就是结构化思考**：本次 bench 的 152 份原始请求体里 **113 份**带
   `thinking={"type":"adaptive"}`，还带 `context_management.clear_thinking_20251015`
   （它连思考块的上下文管理策略都声明了）。而 `parse_anthropic_request` 对
   `thinking` / `context_management` / `output_config` **一个字都没处理**。
2. **上游默认就给**：直连 `/api/coding/v3` 实测，**不带任何参数**时
   `reasoning_content` 也有 144 字符 —— 我们此前在这条路径上一行不读，整段丢弃。
   （live 佐证：claude-code `reasoning_text` 非空 29%，而那 29% 与 `finish_reason`
   的 27% 是同一批**非流式**轮；流式轮一条都没有。）
3. **裸跑是这么干的**：Ark 的 Anthropic 原生端点回
   `content: [{type: thinking, len 163}, {type: text, len 122}]`、正文零标记
   —— CC 的"规整"来自这里。

块形态照抄 `agency/streams.py::_AnthropicNarrator`（V-P3 播报，D7 五 agent 实测
"CC 折叠成计时条"）。**不是新协议，是把已有的形态接到主路径上。**

## 🔴 这条修法**不**解决内联泄漏（MQ-P18），别把两件事混起来

leak 轮的 `reasoning_content` 是**空的**（34 leak / 19 structured，两组交集 0）——
上游在这两种投递形态之间切换。本文件修的是"结构化那份被我们丢了"；
内联那份的成因**仍未定位**，守卫 `inline_think_leaked` 继续报它。
"""

from __future__ import annotations

import json

import pytest
from bladex_proxy.anthropic import anthropic_stream_generator
from bladex_proxy.capture import CaptureResult


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
    def __init__(self, content=None, tool_calls=None, finish_reason=None,
                 reasoning_content=None):
        self.choices = [_FakeChoice(
            _FakeDelta(content, tool_calls, reasoning_content), finish_reason)]
        self.usage = None


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


class _Fn:
    name = "Bash"
    arguments = '{"command":"ls"}'


class _TC:
    id = "call_1"
    index = 0
    function = _Fn()


async def _events(chunks) -> tuple[list[dict], CaptureResult]:
    result = CaptureResult()
    raw = b""
    async for part in anthropic_stream_generator(
        _FakeStream(chunks), result, "claude-3", input_tokens_estimate=1
    ):
        raw += part
    out = [json.loads(ln[6:]) for block in raw.decode().split("\n\n")
           for ln in block.split("\n") if ln.startswith("data: ")]
    return out, result


def _blocks(events: list[dict]) -> list[tuple[int, str]]:
    """(index, 块类型) 序列，按 content_block_start 出现序。"""
    return [(e["index"], e["content_block"]["type"]) for e in events
            if e.get("type") == "content_block_start"]


@pytest.mark.asyncio
async def test_reasoning_becomes_a_thinking_block() -> None:
    """结构化思考走 `thinking` 块，正文走 `text` 块，两者不混。"""
    events, result = await _events([
        _FakeChunk(reasoning_content="先数一下样本空间"),
        _FakeChunk(reasoning_content="36 种"),
        _FakeChunk(content="概率是 1/6。"),
        _FakeChunk(finish_reason="stop"),
    ])
    assert _blocks(events) == [(0, "thinking"), (1, "text")]
    thinking = "".join(e["delta"]["thinking"] for e in events
                       if e.get("type") == "content_block_delta"
                       and e["delta"]["type"] == "thinking_delta")
    text = "".join(e["delta"]["text"] for e in events
                   if e.get("type") == "content_block_delta"
                   and e["delta"]["type"] == "text_delta")
    assert thinking == "先数一下样本空间36 种"
    assert text == "概率是 1/6。"
    # Hub 侧：正文与思考分开存，都不丢
    assert result.full_text == "概率是 1/6。"
    assert result.reasoning_text == "先数一下样本空间36 种"


@pytest.mark.asyncio
async def test_thinking_block_is_closed_before_text_opens() -> None:
    """块序：thinking 必须先 stop 再开 text（Anthropic 不允许交错）。"""
    events, _ = await _events([
        _FakeChunk(reasoning_content="想"),
        _FakeChunk(content="答"),
        _FakeChunk(finish_reason="stop"),
    ])
    seq = [(e.get("type"), e.get("index")) for e in events
           if e.get("type") in ("content_block_start", "content_block_stop")]
    assert seq == [("content_block_start", 0), ("content_block_stop", 0),
                   ("content_block_start", 1), ("content_block_stop", 1)]


@pytest.mark.asyncio
async def test_thinking_then_tool_call_only() -> None:
    """纯工具回复（思考 + 工具、无正文）：thinking 关掉后才开 tool_use，序号不留洞。"""
    events, result = await _events([
        _FakeChunk(reasoning_content="该看文件了"),
        _FakeChunk(tool_calls=[_TC()]),
        _FakeChunk(finish_reason="tool_calls"),
    ])
    assert _blocks(events) == [(0, "thinking"), (1, "tool_use")]
    stops = [e["index"] for e in events if e.get("type") == "content_block_stop"]
    assert sorted(stops) == [0, 1], "开了的块都要关"
    assert result.reasoning_text == "该看文件了"
    assert len(result.tool_events) == 1


@pytest.mark.asyncio
async def test_thinking_only_turn_closes_its_block() -> None:
    """只有思考、没有正文也没有工具：块仍要收尾（否则客户端等一个不来的 stop）。"""
    events, result = await _events([
        _FakeChunk(reasoning_content="嗯"),
        _FakeChunk(finish_reason="stop"),
    ])
    assert _blocks(events) == [(0, "thinking")]
    assert [e["index"] for e in events if e.get("type") == "content_block_stop"] == [0]
    assert result.full_text == ""
    assert result.reasoning_text == "嗯"


@pytest.mark.asyncio
async def test_no_reasoning_means_no_thinking_block() -> None:
    """🔴 阴性对照：上游没给思考就不许凭空造块——`text` 仍占 index 0，逐字不变。"""
    events, result = await _events([
        _FakeChunk(content="Hello"),
        _FakeChunk(content=" world"),
        _FakeChunk(finish_reason="stop"),
    ])
    assert _blocks(events) == [(0, "text")]
    assert result.full_text == "Hello world"
    assert result.reasoning_text == ""


@pytest.mark.asyncio
async def test_late_reasoning_after_text_is_kept_but_not_emitted() -> None:
    """正文已开之后才来的 reasoning：**只入库不发块**。

    Anthropic 的块不能交错，正文开了再插 thinking 会让客户端解析错乱
    （MQ-L22 的形态：事件流不合规 ⇒ 官方解析器炸 ⇒ 断连）。
    但它是采集到的原始信息，**必须入 Hub**（刚性原则 13）。
    """
    events, result = await _events([
        _FakeChunk(content="答案"),
        _FakeChunk(reasoning_content="补一句思考"),
        _FakeChunk(finish_reason="stop"),
    ])
    assert _blocks(events) == [(0, "text")], "不许在正文之后再开 thinking 块"
    assert result.reasoning_text == "补一句思考", "不发不等于丢"


@pytest.mark.asyncio
async def test_reasoning_list_form_is_coerced() -> None:
    """ADR-0020 T4 同款：某些 Router 解析下 reasoning 是 list，强制 str 再发。"""
    events, result = await _events([
        _FakeChunk(reasoning_content=["a", "b"]),
        _FakeChunk(content="x"),
        _FakeChunk(finish_reason="stop"),
    ])
    assert result.reasoning_text == "ab"
    assert _blocks(events)[0] == (0, "thinking")
