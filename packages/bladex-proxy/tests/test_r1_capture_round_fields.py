"""批 R · R1（2026-09-18）：MQ-CA5 三协议 cache 埋点 ‖ MQ-CA13 ①② 两个字段 ‖ P29 卡的归属字段。

## 这张表守的是"仪器接线"，不是纯函数（与 test_p31 同一形态）

缺陷形态：`capture_done`（chat）是三条路径里**唯一**不记 `cache_read_tokens` 的，
而它承接全部 OpenAI 兼容 agent（09-15：110/267 条 capture）⇒ 6.7% 覆盖。
补法不是再抄一份，而是三条路径各自的 `*_capture_done` 行都有同一把尺子的字段——
**覆盖率断言 = 三条路径的 capture 事件都带该键**，缺 / None ⇒ NO-DATA，不是 0。

同批合并（拍板 0.3.0-(3)：分开做就是第二把尺子）：
- CA13 ①：`agency_ledger_block_injected.block_hash` / `agency_about_injected.about_hash`（内容哈希，
  取代拿字数代理"逐字相同"）；
- CA13 ②：`tool_result_chars` / `tool_result_n`（工具结果体量，residual 的分母此前未受控）；
- P29 卡：`agent` / `model` / `session_id` / `path` 落进三条 capture 行（此前**三条都没有**，
  批 M 结算稿说"另两条带"是读错了——本表把它钉成事实）；
- `t_call_epoch` / `t_done_epoch`：行内自带起止墙钟，`probe_cache_ttl` 不再按行序 join。

每一格都有"把 kwarg 删掉就红"的形状：断言键**存在**（`in`），不是断言值。
"""

from __future__ import annotations

import os

import pytest
import structlog.testing
from bladex_proxy.anthropic import anthropic_stream_generator
from bladex_proxy.capture import (
    CaptureResult,
    bind_round,
    capture_stream,
    round_fields,
    tool_result_volume,
)
from bladex_proxy.responses import responses_stream_generator

# ── 假上游（与 test_g17 / test_p31 同款，故意不共享）──


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


class _Details:
    def __init__(self, cached):
        self.cached_tokens = cached


class _Usage:
    def __init__(self, prompt=100, completion=7, cached=None):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = prompt + completion
        self.prompt_tokens_details = _Details(cached) if cached is not None else None

    def model_dump(self) -> dict:
        return {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens}


def _chunks(*texts: str, usage=None) -> list[_FakeChunk]:
    return [*[_FakeChunk(content=t) for t in texts], _FakeChunk(finish_reason="stop", usage=usage)]


_ROUND_KEYS = ("agent", "model", "session_id", "path", "tool_result_chars", "tool_result_n",
               "t_call_epoch", "t_done_epoch")
_MSGS = [{"role": "user", "content": "hi"},
         {"role": "assistant", "content": None, "tool_calls": [{"id": "t1"}]},
         {"role": "tool", "tool_call_id": "t1", "content": "x" * 40}]


def _bound(path: str) -> CaptureResult:
    r = CaptureResult()
    bind_round(r, agent_id="ag", model="m1", session_id="s1", path=path, messages=_MSGS)
    return r


def _row(cap, event):
    rows = [e for e in cap if e.get("event") == event]
    assert len(rows) == 1, f"{event} 应恰好打一次，实得 {len(rows)}"
    return rows[0]


# ── ① 三协议覆盖率：每条 capture 行都带 cache_read_tokens 键 + 归属/体量/时钟 ──


@pytest.mark.asyncio
async def test_chat_capture_done_carries_cache_and_round_fields() -> None:
    r = _bound("/v1/chat/completions")
    with structlog.testing.capture_logs() as cap:
        async for _ in capture_stream(_FakeStream(_chunks("a", usage=_Usage(cached=5))), r):
            pass
    row = _row(cap, "capture_done")
    assert "cache_read_tokens" in row and row["cache_read_tokens"] == 5
    assert row["input_tokens_reported"] == 100 and row["output_tokens"] == 7
    for k in _ROUND_KEYS:
        assert k in row, f"capture_done 缺 {k}"
    assert row["agent"] == "ag" and row["model"] == "m1" and row["session_id"] == "s1"
    assert row["path"] == "/v1/chat/completions"
    assert row["tool_result_chars"] == 40 and row["tool_result_n"] == 1
    assert row["t_call_epoch"] is not None and row["t_done_epoch"] >= row["t_call_epoch"]


@pytest.mark.asyncio
async def test_chat_capture_done_none_is_not_zero() -> None:
    """上游不报 cached_tokens ⇒ None（NO-DATA）；根本没 usage ⇒ 三格都 None，不写 0。"""
    r = _bound("/v1/chat/completions")
    with structlog.testing.capture_logs() as cap:
        async for _ in capture_stream(_FakeStream(_chunks("a", usage=_Usage())), r):
            pass
    row = _row(cap, "capture_done")
    assert "cache_read_tokens" in row and row["cache_read_tokens"] is None
    assert row["input_tokens_reported"] == 100
    r2 = _bound("/v1/chat/completions")
    with structlog.testing.capture_logs() as cap2:
        async for _ in capture_stream(_FakeStream(_chunks("a")), r2):
            pass
    row2 = _row(cap2, "capture_done")
    assert row2["cache_read_tokens"] is None
    assert row2["input_tokens_reported"] is None and row2["output_tokens"] is None


@pytest.mark.asyncio
async def test_messages_capture_done_carries_round_fields() -> None:
    r = _bound("/v1/messages")
    with structlog.testing.capture_logs() as cap:
        async for _ in anthropic_stream_generator(_FakeStream(_chunks("a", usage=_Usage(cached=3))),
                                                  r, "claude-3", input_tokens_estimate=7):
            pass
    row = _row(cap, "anthropic_capture_done")
    assert "cache_read_tokens" in row and row["cache_read_tokens"] == 3
    for k in _ROUND_KEYS:
        assert k in row, f"anthropic_capture_done 缺 {k}"
    assert row["path"] == "/v1/messages" and row["tool_result_chars"] == 40


@pytest.mark.asyncio
async def test_responses_capture_done_carries_round_fields() -> None:
    r = _bound("/v1/responses")
    with structlog.testing.capture_logs() as cap:
        async for _ in responses_stream_generator(_FakeStream(_chunks("a", usage=_Usage(cached=2))),
                                                  r, "gpt-5-codex"):
            pass
    row = _row(cap, "responses_capture_done")
    assert "cache_read_tokens" in row and row["cache_read_tokens"] == 2
    for k in _ROUND_KEYS:
        assert k in row, f"responses_capture_done 缺 {k}"
    assert row["path"] == "/v1/responses" and row["tool_result_n"] == 1


# ── ② 端点接线：三条入站路径各调一次 bind_round（与 t_upstream 同款守卫）──


def test_all_three_endpoints_bind_round() -> None:
    from bladex_proxy.server import endpoints_anthropic, endpoints_chat, endpoints_responses
    for mod, path in ((endpoints_chat, "/v1/chat/completions"),
                      (endpoints_anthropic, "/v1/messages"),
                      (endpoints_responses, "/v1/responses")):
        src = open(mod.__file__, encoding="utf-8").read()
        assert src.count("bind_round(") == 1, f"{os.path.basename(mod.__file__)} 应恰有一处 bind_round"
        assert f'path="{path}"' in src, f"{os.path.basename(mod.__file__)} 绑错了 path"
        # bind 在发起上游调用之前：源码顺序 bind_round < t_upstream 赋值
        assert src.index("bind_round(") < src.index("result.t_upstream = time.perf_counter()")


# ── ③ 纯函数：工具结果体量 / 未绑定时 epoch 报 None ──


def test_tool_result_volume_counts_tool_role_and_tool_result_blocks() -> None:
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "q"},
        {"role": "tool", "content": "abc"},
        {"role": "tool", "content": [{"type": "text", "text": "de"}]},
        {"role": "user", "content": [{"type": "tool_result", "content": "fgh"},
                                     {"type": "text", "text": "ignored"}]},
        {"role": "assistant", "content": "not counted"},
    ]
    assert tool_result_volume(msgs) == (8, 3)
    assert tool_result_volume([]) == (0, 0)
    assert tool_result_volume([{"role": "user", "content": "x"}]) == (0, 0)


def test_round_fields_unbound_reports_none_not_zero() -> None:
    f = round_fields(CaptureResult())
    assert f["t_call_epoch"] is None and f["t_done_epoch"] is None
    assert f["agent"] == "" and f["tool_result_chars"] == 0


# ── ④ CA13 ①：块内容哈希 ──


def test_ledger_block_and_about_log_content_hash(monkeypatch) -> None:
    import hashlib

    from bladex_proxy.agency import AgencyRuntime
    from test_agency_enablement import _SYS, _mk_ledger, _on, _tc
    _on(monkeypatch, "toolface", "ledger")
    ag = AgencyRuntime()
    _mk_ledger(ag, "test-agent", goal="某任务")
    msgs = [_SYS, {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "tool_calls": [_tc("x", "t1")]},
            {"role": "tool", "tool_call_id": "t1", "content": "r"}]
    with structlog.testing.capture_logs() as cap:
        out = ag.insert_ledger_block(msgs, "test-agent")
    row = _row(cap, "agency_ledger_block_injected")
    assert "block_hash" in row and len(row["block_hash"]) == 16
    assert row["block_hash"] == hashlib.sha256(out[-1]["content"].encode("utf-8")).hexdigest()[:16]
    # 同内容同哈希、改一个字就变：哈希是内容的函数，不是字数的
    with structlog.testing.capture_logs() as cap2:
        ag.insert_ledger_block(msgs, "test-agent")
    assert _row(cap2, "agency_ledger_block_injected")["block_hash"] == row["block_hash"]
