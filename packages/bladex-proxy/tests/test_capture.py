"""流式捕获单元测试（T7，ISSUE-3 修复后用 CaptureResult）。"""

import json

import pytest
from bladex_proxy.capture import CaptureResult, capture_stream


class _FakeDelta:
    def __init__(self, content: str | None = None, tool_calls=None,
                 reasoning_content: str | None = None):
        self.content = content
        self.tool_calls = tool_calls
        self.reasoning_content = reasoning_content


class _FakeChoice:
    def __init__(self, delta: _FakeDelta, finish_reason: str | None = None):
        self.delta = delta
        self.finish_reason = finish_reason


class _FakeChunk:
    def __init__(self, content: str | None = None, tool_calls=None,
                 reasoning_content: str | None = None,
                 finish_reason: str | None = None,
                 usage: dict | None = None):
        self.choices = [_FakeChoice(
            _FakeDelta(content, tool_calls, reasoning_content), finish_reason)]
        self.usage = usage

    def model_dump(self) -> dict:
        return {"choices": [{"delta": {"content": self.choices[0].delta.content}}]}


class _FakeStream:
    def __init__(self, chunks: list[_FakeChunk]):
        self._chunks = chunks

    def __aiter__(self):
        self._iter = iter(self._chunks)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration from None


@pytest.mark.asyncio
async def test_capture_yields_sse_and_buffers_text():
    """回传 SSE 给客户端，同时缓冲拼出完整回复。"""
    chunks = [_FakeChunk(content="Hello"), _FakeChunk(content=" world"), _FakeChunk(content="!")]
    stream = _FakeStream(chunks)
    result = CaptureResult()

    sse_parts = []
    async for sse in capture_stream(stream, result):
        sse_parts.append(sse)

    data_lines = [s for s in sse_parts if s.startswith("data: ") and "[DONE]" not in s]
    assert len(data_lines) == 3
    assert result.full_text == "Hello world!"
    assert result.done is True
    assert result.error == ""
    assert sse_parts[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_capture_empty_stream():
    """空流不报错。"""
    stream = _FakeStream([])
    result = CaptureResult()

    sse_parts = []
    async for sse in capture_stream(stream, result):
        sse_parts.append(sse)

    assert result.full_text == ""
    assert result.done is True
    assert sse_parts[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_capture_preserves_chunk_content():
    """客户端看到的流 == 模型原始流的 content。"""
    chunks = [_FakeChunk(content="A"), _FakeChunk(content="B"), _FakeChunk(content="C")]
    stream = _FakeStream(chunks)
    result = CaptureResult()

    received_content = []
    async for sse in capture_stream(stream, result):
        if sse.startswith("data: ") and "[DONE]" not in sse:
            data = json.loads(sse[6:])
            content = data["choices"][0]["delta"]["content"]
            received_content.append(content)

    assert received_content == ["A", "B", "C"]


@pytest.mark.asyncio
async def test_capture_interrupted_still_has_partial():
    """ISSUE-3: 中途断开/出错时，result 仍有已捕获的部分。"""
    class _InterruptingStream:
        def __aiter__(self):
            self._chunks = iter([_FakeChunk(content="partial"), _FakeChunk(content=" text")])
            self._count = 0
            return self

        async def __anext__(self):
            self._count += 1
            if self._count > 2:
                raise RuntimeError("connection lost")
            return next(self._chunks)

    stream = _InterruptingStream()
    result = CaptureResult()

    sse_parts = []
    async for sse in capture_stream(stream, result):
        sse_parts.append(sse)

    assert result.done is False
    assert result.error != ""
    assert result.full_text == "partial text"  # 已捕获的部分在


# ── T4(ADR-0018 §4.4)验收①: reasoning/usage/finish_reason/ms_first_chunk 入 CaptureResult ──


@pytest.mark.asyncio
async def test_capture_collects_reasoning_usage_finish_first_chunk():
    """T4 验收①(C1/C2/C6): reasoning_content 累积、usage 提取、finish_reason、首字延迟全部入 CaptureResult。

    这些字段事后不可回补（流过即逝），捕获时必须落。
    """
    chunks = [
        _FakeChunk(content="Hello", reasoning_content="Let me think"),
        _FakeChunk(content=" world", reasoning_content=" about this", finish_reason="stop"),
        # 最终 chunk 带 usage（上游在末 chunk 顶层带 usage），无 content
        _FakeChunk(usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}),
    ]
    stream = _FakeStream(chunks)
    result = CaptureResult()

    async for _ in capture_stream(stream, result):
        pass

    assert result.full_text == "Hello world"
    assert result.reasoning_text == "Let me think about this"
    assert result.finish_reason == "stop"
    assert result.usage == {"prompt": 10, "completion": 20, "total": 30}
    assert result.ms_first_chunk > 0
    assert result.done is True
    assert result.chunk_count == 3


@pytest.mark.asyncio
async def test_capture_usage_missing_when_no_usage_chunk():
    """无 usage chunk 时 usage 保持空 dict（不报错）。"""
    chunks = [_FakeChunk(content="hi", finish_reason="stop")]
    stream = _FakeStream(chunks)
    result = CaptureResult()

    async for _ in capture_stream(stream, result):
        pass

    assert result.usage == {}
    assert result.finish_reason == "stop"
