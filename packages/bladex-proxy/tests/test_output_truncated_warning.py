"""MQ-A34 ③ · 上游输出截断告警 `upstream_output_truncated`（2026-09-03，拍板：0.1.0 只做告警）。

## 病例

本机 32K 窗口模型（M4 Pro 48G 跑 35B-A3B，硬件常量）上，注入面把输出预算挤到 30–164 token，
账本面在 hermes:accept / ornith 上 0/28。"模型不调 switch"与"模型调不出 switch"是两件事，
没有告警分不开（`task-hermes-accept-ledger-gap-20260903.md` §5/§6）。

## 钉什么

1. 三条入站协议（chat / messages / responses）各一条：上游 `finish_reason=length` ⇒ 恰好一条
   `upstream_output_truncated`，字段齐（agent / model / injected_chars / prompt_chars / inject_ratio）。
   告警点在 `_enqueue_turn`（三协议 × 流式/非流式的唯一汇合点）——所以三协议**分别**打到那里
   才算钉住，不是钉一个 helper。
2. 流式一条（chat）：finish_reason 在最后一个 chunk 上，capture 侧要拿得到。
3. 阴性对照：`finish_reason=stop` 零告警——告警不许变成每轮都响的噪声。
4. 等价值：`output_truncated` 收 `length` / `max_tokens` / `max_output_tokens`（三协议原生词），
   不收 `stop` / `tool_calls` / 空。
5. 零行为差异：有没有这条告警，响应体逐字相同（只记不改）。
6. 分子口径（2026-09-04 修正）：`injected_chars` = BladeX 加进正文的**全部**字符
   （硬规则 + about/名册 + 账本块 + 候选段），不是只算 inject 阶段的硬规则——
   挤爆 32K 输出预算的正是账本面那几块。

日志用 `structlog.testing.capture_logs`（proxy 侧走 structlog，pytest `caplog` 恒空）。
"""
from __future__ import annotations

import tempfile
from unittest.mock import patch

import pytest
import structlog.testing

from bladex_proxy.capture import OUTPUT_TRUNCATED_REASONS, output_truncated
from bladex_proxy.config import ProxyConfig
from bladex_proxy.server import create_app
from fastapi.testclient import TestClient

HARD_RULE = "MUST answer in Chinese"


def _cfg() -> ProxyConfig:
    return ProxyConfig(hard_rules=[HARD_RULE], upstream_model="openai/test",
                       upstream_api_key="sk-fake",
                       rocksdb_path=f"{tempfile.mkdtemp()}/rocksdb")


# ── 上游替身（OpenAI 形态；Router 把三家上游归一到这个形态）──────────────

class _Usage:
    prompt_tokens, completion_tokens, total_tokens = 30000, 42, 30042

    def model_dump(self):
        return {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens}


class _Msg:
    def __init__(self, content: str):
        self.content = content
        self.tool_calls = None

    def model_dump(self):
        return {"role": "assistant", "content": self.content}


class _Resp:
    def __init__(self, content: str, finish_reason: str):
        self.choices = [type("C", (), {"message": _Msg(content),
                                       "finish_reason": finish_reason})()]
        self.usage = _Usage()
        self._fr = finish_reason
        self._c = content

    def model_dump(self):
        return {"id": "chatcmpl-x", "choices": [
            {"index": 0, "message": {"role": "assistant", "content": self._c},
             "finish_reason": self._fr}],
            "usage": self.usage.model_dump()}


class _SChunk:
    def __init__(self, content=None, finish_reason=None, usage=None):
        self.choices = [type("C", (), {
            "delta": type("D", (), {"content": content, "tool_calls": None})(),
            "finish_reason": finish_reason})()]
        self.usage = usage

    def model_dump(self):
        d0 = self.choices[0]
        delta = {"content": d0.delta.content} if d0.delta.content else {}
        return {"id": "chatcmpl-x", "object": "chat.completion.chunk", "created": 0,
                "model": "m", "choices": [{"index": 0, "delta": delta,
                                           "finish_reason": d0.finish_reason}]}


class _SStream:
    def __init__(self, chunks):
        self._c = chunks

    def __aiter__(self):
        self._it = iter(self._c)
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration from None


def _fake_upstream(finish_reason: str):
    async def fake(model, messages, stream, **kw):
        if stream:
            return _SStream([_SChunk(content="截断前的半"),
                             _SChunk(finish_reason=finish_reason, usage=_Usage())])
        return _Resp("截断前的半", finish_reason)
    return fake


def _post(client, proto: str, stream: bool):
    hdr = {"X-Agent-ID": "probe-trunc", "X-Session-ID": "s-trunc"}
    if proto == "chat":
        return client.post("/v1/chat/completions", json={
            "model": "gpt-4o", "stream": stream, "max_tokens": 64,
            "messages": [{"role": "system", "content": "You are a test agent."},
                         {"role": "user", "content": "写一篇长文"}]},
            headers={"Authorization": "Bearer sk", **hdr})
    if proto == "messages":
        return client.post("/v1/messages", json={
            "model": "claude-x", "stream": stream, "max_tokens": 64,
            "system": "You are a test agent.",
            "messages": [{"role": "user", "content": "写一篇长文"}]},
            headers={"x-api-key": "sk", **hdr})
    return client.post("/v1/responses", json={
        "model": "gpt-x", "stream": stream, "max_output_tokens": 64,
        "instructions": "You are a test agent.", "input": "写一篇长文"},
        headers={"Authorization": "Bearer sk", **hdr})


def _run(proto: str, stream: bool, finish_reason: str):
    app = create_app(_cfg())
    with patch("bladex_proxy.route.router_sdk.acompletion", new=_fake_upstream(finish_reason)):
        with structlog.testing.capture_logs() as cap:
            with TestClient(app) as client:
                resp = _post(client, proto, stream)
                # 入库是 shield 过的后台 task；退出 TestClient 上下文即等 lifespan 收尾
    rows = [e for e in cap if e.get("event") == "upstream_output_truncated"]
    return resp, rows


# ── 1. 三协议各一条（非流式，finish_reason=length）────────────────────────

@pytest.mark.parametrize("proto", ["chat", "messages", "responses"])
def test_length_is_warned_once_per_protocol(proto):
    resp, rows = _run(proto, stream=False, finish_reason="length")
    assert resp.status_code == 200, resp.text
    assert len(rows) == 1, f"{proto}: 期望恰好一条告警，得到 {rows}"
    row = rows[0]
    assert row["agent"] == "probe-trunc" and row["finish_reason"] == "length"
    assert row["model"], "model 字段空——没法按模型 grep 截断率"
    # 注入占比读数：硬规则注入了 ⇒ injected_chars > 0；prompt_chars 是 agent 原始消息
    assert row["injected_chars"] >= len(HARD_RULE)
    assert row["prompt_chars"] > 0
    assert 0 < row["inject_ratio"] < 1
    assert row["completion_tokens"] == 42 and row["prompt_tokens"] == 30000
    assert row["max_tokens"] == 64, "请求侧 max_tokens 要带上——三协议各自的字段都要映到这一格"


# ── 2. 流式：finish_reason 在末 chunk，capture 侧拿得到 ────────────────────

def test_stream_length_is_warned():
    resp, rows = _run("chat", stream=True, finish_reason="length")
    assert resp.status_code == 200
    assert len(rows) == 1 and rows[0]["finish_reason"] == "length"
    assert rows[0]["completion_tokens"] == 42, "流式 usage 在末 chunk 顶层，也要进读数"


# ── 3. 阴性对照：stop 零告警 ───────────────────────────────────────────────

@pytest.mark.parametrize("proto", ["chat", "messages", "responses"])
def test_stop_is_silent(proto):
    resp, rows = _run(proto, stream=False, finish_reason="stop")
    assert resp.status_code == 200
    assert rows == [], f"{proto}: 正常结束不许告警，否则这条告警就是噪声"


# ── 4. 等价值集合 ─────────────────────────────────────────────────────────

def test_equivalents_cover_three_protocol_words():
    assert OUTPUT_TRUNCATED_REASONS == {"length", "max_tokens", "max_output_tokens"}
    for w in ("length", "max_tokens", "max_output_tokens", "stop,length"):
        assert output_truncated(w), w
    for w in ("stop", "tool_calls", "end_turn", "", None, "content_filter"):
        assert not output_truncated(w), w


# ── 5. 零行为差异：告警只记不改 ───────────────────────────────────────────

def test_injected_chars_counts_every_bladex_surface_not_only_hard_rules():
    """🔴 分子口径（2026-09-04，MQ-A34 补记）：`injected_chars` = BladeX 加进正文的**全部**。

    修前分子 = `len(prep.injected_text)`，只有 inject 阶段的硬规则正文。about / 名册 /
    账本块 / 候选段在 `_apply_agency_surfaces` 才进 `injected_messages`——**而 32K 病例里
    把输出预算挤到 30–164 token 的恰是这几块**。尺子量不到最大的那一段，与 MQ-A33 同族。

    判别力：本轮带工具 ⇒ 不是零工具 aux ⇒ 账本块进请求，它比硬规则长一个量级。
    只算硬规则的老口径在这条断言上必红。

    可达性前置在断言之前：先证明账本块真的进了上游请求，否则本条是空跑
    （"读数为 0 先证明那个桶可达"）。
    """
    from bladex_proxy.agency import LEDGER_BLOCK_OPEN

    sent: list[list[dict]] = []

    async def fake(model, messages, stream, **kw):
        sent.append(messages)
        return _Resp("截断前的半", "length")

    app = create_app(_cfg())
    with patch("bladex_proxy.route.router_sdk.acompletion", new=fake):
        with structlog.testing.capture_logs() as cap:
            with TestClient(app) as client:
                resp = client.post("/v1/chat/completions", json={
                    "model": "gpt-4o", "stream": False, "max_tokens": 64,
                    # 带工具 = 不落 `brings_no_tools` 的账本域 aux 判据（MQ-L21）
                    "tools": [{"type": "function", "function": {
                        "name": "read_file", "description": "read a file",
                        "parameters": {"type": "object", "properties": {}}}}],
                    "messages": [{"role": "system", "content": "You are a test agent."},
                                 {"role": "user", "content": "写一篇长文"}]},
                    headers={"Authorization": "Bearer sk",
                             "X-Agent-ID": "probe-trunc", "X-Session-ID": "s-trunc-tools"})
    assert resp.status_code == 200, resp.text

    ledger_chars = sum(len(m["content"]) for m in sent[0]
                       if isinstance(m.get("content"), str)
                       and LEDGER_BLOCK_OPEN in m["content"])
    assert ledger_chars > 0, "账本块没进上游请求 —— 本条空跑，先修可达性再看读数"
    assert ledger_chars > len(HARD_RULE), "账本块比硬规则还短 —— 判别力对照失效"

    rows = [e for e in cap if e.get("event") == "upstream_output_truncated"]
    assert len(rows) == 1
    assert rows[0]["injected_chars"] >= ledger_chars + len(HARD_RULE), (
        f"分子只有 {rows[0]['injected_chars']}，账本块本身就 {ledger_chars} —— "
        "分子退回了硬规则口径")
    assert 0 < rows[0]["inject_ratio"] < 1


def test_warning_does_not_change_response_body():
    a, _ = _run("chat", stream=False, finish_reason="length")
    b, _ = _run("chat", stream=False, finish_reason="stop")
    ja, jb = a.json(), b.json()
    assert ja["choices"][0]["message"]["content"] == jb["choices"][0]["message"]["content"]
    assert ja["choices"][0]["finish_reason"] == "length", "finish_reason 原样透传，不被告警吃掉"
    assert jb["choices"][0]["finish_reason"] == "stop"
