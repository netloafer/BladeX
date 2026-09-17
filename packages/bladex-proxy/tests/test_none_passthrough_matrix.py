"""V-P5 · `none` 分流零行为差异（三红线之 3）· 端点级 × 三协议 × 流式/非流式。

## 判据：同一请求跑两遍，逐字比对

三红线之 3 是「**模型不调用时零行为差异**」。表达它最直接的方式不是
"检查响应里没有 bladex 字样"，而是：

    拦截关 → 响应 A
    拦截开 → 响应 B
    normalize(A) == normalize(B)

只归一化天然随机的字段（id / 时间戳），其余**逐字**相同。
这样任何形态上的改动——多一个事件、少一个字段、编号变了——都会被抓住，
而不需要我预先想到"可能坏在哪"。今天已经有三次是"我没想到会坏在那里"。

## 为什么此前没有

`test_agency_enablement.py::test_bladex_call_passes_through_when_off` 测的是
**模块关闭**——那是另一件事（拦截压根没跑）。
**模块开着、模型只调自己的工具**，才是红线说的那个场景。

十八形态对账（2026-08-27）里，`none` 那一列三协议 × 两形式**全是空的**。

## 🔴 判别力的真实边界（实测，别高估这份文件）

`none` 模式下**代码本来就几乎不跑**，所以"零行为差异"测试天生弱。
复核方逐个注入实测：

| 注入 | 抓住？ |
|---|---|
| `if mode == "none":` 短路去掉（走替身） | ❌ 替身忠实复制，输出无差 |
| `MODE_NONE` 误判成 `MODE_MIXED` | ❌ 没有 bladex 调用可剥，输出无差 |
| 替身丢掉 `tool_calls` / 改写 `finish_reason` | ❌ `none` 根本到不了那段代码 |
| **透传时多塞一个字段**（chat 流式） | ✅ 2 红 |
| **`none` 路径吞掉终止事件**（三协议） | ✅ 各 1 红 |

⇒ 它能抓的是**「透传其实不是逐字透传」**这一类：多一个事件、少一个事件、
重新序列化改了字节。它抓不住"拦截逻辑内部写错"——那类由
`test_nonstream_interception.py`（判据取 `is`）与
`test_mixed_splice_matrix.py` 负责。

⚠️ 途中一次自查值得记：前两轮注入"全部照过"，我差点据此判本文件零判别力——
实查才发现**注入的字符串根本没匹配上、压根没落地**。
**没落地的注入什么都不证明**，先验证注入生效再读结论。
"""
from __future__ import annotations

import re
import tempfile
from unittest.mock import patch

import pytest
from bladex_proxy.config import ProxyConfig
from bladex_proxy.server import create_app
from fastapi.testclient import TestClient

AGENT_TOOL = {"type": "function",
              "function": {"name": "exec_command", "description": "run",
                           "parameters": {"type": "object", "properties": {}}}}
AGENT_CALL = {"id": "call_agent1", "type": "function",
              "function": {"name": "exec_command", "arguments": '{"cmd":"ls"}'}}


def _cfg() -> ProxyConfig:
    return ProxyConfig(hard_rules=["MUST be polite"], upstream_model="openai/test",
                       upstream_api_key="sk-fake",
                       rocksdb_path=f"{tempfile.mkdtemp()}/rocksdb")


# ── 上游 fake（OpenAI 形态，三端点共用——各自 generator 负责转协议）──

class _NsTc:
    """🔴 必须是**属性对象**不是 dict。

    chat 路径读 `message.model_dump()`，而 `format_anthropic_response` /
    `format_responses_response` 读 `getattr(tc, "function").name`——
    真实 litellm 响应两者都满足。fixture 只给 dict 时，
    非流式 messages/responses 会渲染出 `"name": ""`，
    而这**不是产品缺陷，是替身造得不像**（2026-08-28 被阳性对照当场抓住）。
    """

    def __init__(self, d: dict):
        self.id = d["id"]
        self.type = d.get("type", "function")
        fn = d["function"]
        self.function = type("F", (), {"name": fn["name"],
                                       "arguments": fn["arguments"]})()
        self._d = d

    def model_dump(self):
        return dict(self._d)


class _Msg:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self._raw = list(tool_calls or [])
        self.tool_calls = [_NsTc(t) for t in self._raw] or None

    def model_dump(self):
        d = {"role": "assistant", "content": self.content}
        if self._raw:
            d["tool_calls"] = list(self._raw)
        return d


class _Resp:
    def __init__(self, msg):
        self.choices = [type("C", (), {"message": msg,
                                       "finish_reason": "tool_calls"})()]
        self.usage = None
        self._m = msg

    def model_dump(self):
        return {"id": "chatcmpl-x", "choices": [
            {"index": 0, "message": self._m.model_dump(),
             "finish_reason": "tool_calls"}]}


class _STc:
    def __init__(self, index, cid, name, args):
        self.index, self.id = index, cid
        self.function = type("F", (), {"name": name, "arguments": args})()

    def dump(self):
        return {"index": self.index, "id": self.id, "type": "function",
                "function": {"name": self.function.name,
                             "arguments": self.function.arguments}}


class _SChunk:
    def __init__(self, tool_calls=None, finish_reason=None):
        self.choices = [type("C", (), {
            "delta": type("D", (), {"content": None,
                                    "tool_calls": tool_calls})(),
            "finish_reason": finish_reason})()]
        self.usage = None

    def model_dump(self):
        d0 = self.choices[0]
        delta = {}
        if d0.delta.tool_calls:
            delta["tool_calls"] = [tc.dump() for tc in d0.delta.tool_calls]
        return {"id": "chatcmpl-x", "object": "chat.completion.chunk",
                "created": 0, "model": "m",
                "choices": [{"index": 0, "delta": delta,
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


def _fake_upstream(stream_mode: bool):
    """模型**只调 agent 自己的工具** —— 这就是 `none` 分流。"""
    async def fake(model, messages, stream, **kw):
        if stream:
            return _SStream([
                _SChunk(tool_calls=[_STc(0, "call_agent1", "exec_command", "")]),
                _SChunk(tool_calls=[_STc(0, None, None, '{"cmd":"ls"}')]),
                _SChunk(finish_reason="tool_calls"),
            ])
        return _Resp(_Msg(tool_calls=[AGENT_CALL]))
    return fake


# ── 三端点的请求构造 ────────────────────────────────────────────────────

def _post(client, proto: str, stream: bool):
    if proto == "chat":
        return client.post("/v1/chat/completions", json={
            "model": "gpt-4o", "stream": stream, "tools": [AGENT_TOOL],
            "messages": [{"role": "system", "content": "You are a test agent."},
                         {"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer sk", "X-Agent-ID": "test-agent",
                     "X-Session-ID": "s-none"})
    if proto == "messages":
        return client.post("/v1/messages", json={
            "model": "claude-x", "stream": stream, "max_tokens": 100,
            "system": "You are a test agent.",
            "tools": [{"name": "exec_command", "description": "run",
                       "input_schema": {"type": "object", "properties": {}}}],
            "messages": [{"role": "user", "content": "hi"}]},
            headers={"x-api-key": "sk", "X-Agent-ID": "test-agent",
                     "X-Session-ID": "s-none"})
    return client.post("/v1/responses", json={
        "model": "gpt-x", "stream": stream, "instructions": "You are a test agent.",
        "tools": [{"type": "function", "name": "exec_command", "description": "run",
                   "parameters": {"type": "object", "properties": {}}}],
        "input": "hi"},
        headers={"Authorization": "Bearer sk", "X-Agent-ID": "test-agent",
                 "X-Session-ID": "s-none"})


_NOISE = [
    (re.compile(r'"(id|response_id|item_id)"\s*:\s*"[^"]*"'), '"\\1":"<id>"'),
    (re.compile(r'"(created|created_at)"\s*:\s*\d+'), '"\\1":0'),
    (re.compile(r'\bmsg_[0-9a-f]+\b'), "msg_<x>"),
    (re.compile(r'\btoolu_[0-9a-f]+\b'), "toolu_<x>"),
    (re.compile(r'"sequence_number"\s*:\s*\d+'), '"sequence_number":0'),
    (re.compile(r'"call_id"\s*:\s*"[^"]*"'), '"call_id":"<id>"'),
    (re.compile(r'\bfc_[0-9a-f]+\b'), "fc_<x>"),
    (re.compile(r'\bresp_[0-9a-f]+\b'), "resp_<x>"),
    (re.compile(r'\bcall_[0-9a-f]{8,}\b'), "call_<x>"),
]


def _normalize(text: str) -> str:
    """只抹掉天生随机的字段，其余逐字保留。"""
    for pat, rep in _NOISE:
        text = pat.sub(rep, text)
    return text


def _run(monkeypatch, proto: str, stream: bool, interception: bool) -> str:
    monkeypatch.setenv("BLADEX_MODULE_TOOLFACE", "1")
    monkeypatch.setenv("BLADEX_MODULE_LEDGER", "1")
    monkeypatch.setenv("BLADEX_MODULE_INTERCEPTION", "1" if interception else "0")
    app = create_app(_cfg())
    with patch("bladex_proxy.route.router_sdk.acompletion",
               new=_fake_upstream(stream)):
        with TestClient(app) as client:
            return _post(client, proto, stream).text


MATRIX = [(p, s) for p in ("chat", "messages", "responses") for s in (True, False)]
IDS = [f"{p}·{'流式' if s else '非流式'}" for p, s in MATRIX]


@pytest.mark.parametrize("proto,stream", MATRIX, ids=IDS)
class TestNoneIsZeroDiff:

    def test_identical_with_and_without_interception(self, monkeypatch,
                                                     proto, stream):
        """🔴 三红线之 3 的直接表达：开关拦截，响应逐字相同。"""
        off = _normalize(_run(monkeypatch, proto, stream, interception=False))
        on = _normalize(_run(monkeypatch, proto, stream, interception=True))
        assert on == off, (
            f"{proto}/{'stream' if stream else 'nonstream'}：拦截开启改变了 "
            f"`none` 轮次的响应\n--- 关 ---\n{off[:600]}\n--- 开 ---\n{on[:600]}")

    def test_agent_call_survives(self, monkeypatch, proto, stream):
        """阳性对照：agent 自己的调用确实在里面。

        没有这条，上一条测试在"两边都返回空响应"时也会通过——
        08-20 的教训：作废一份读数前先问它还能回答什么，
        这里则是**先确认这份读数不是空的**。
        """
        body = _run(monkeypatch, proto, stream, interception=True)
        assert "exec_command" in body, f"{proto}: agent 的调用没到达"

    def test_no_bladex_leak(self, monkeypatch, proto, stream):
        body = _run(monkeypatch, proto, stream, interception=True)
        assert "bladex_" not in body, f"{proto}: bladex 工具名泄漏进响应"


def test_toolface_was_actually_injected(monkeypatch):
    """前置断言：本文件测的是"工具面发了、模型没用"，不是"压根没发工具面"。

    没有这条，整个文件可能在一个**工具面从未注入**的形态上全绿——
    那测的就不是三红线之 3，而是一个不存在的场景（08-22 同款教训）。
    """
    import structlog
    seen: list = []
    monkeypatch.setattr(structlog.get_logger().__class__, "info",
                        lambda _s, ev, **kw: seen.append((ev, kw)), raising=False)
    _run(monkeypatch, "chat", False, interception=True)
    rows = [kw for ev, kw in seen if ev == "agency_toolface_decision"]
    assert rows, "没有工具面决策日志"
    assert any(kw.get("injected") for kw in rows), (
        f"工具面从未注入，本文件测的场景不成立：{rows}")
