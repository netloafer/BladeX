"""V-P4 · 剥离-拼接形态矩阵：三协议 × 流式/非流式（ADR-0032 §3.2）。

## 为什么要这一份

既有测试在**单元级**是齐的——三个 stripper 各自的重编号、`SpliceLedger`
的老化与锚失、`disposition()` 的三分判定，都有覆盖。缺的是**端到端往返**：

    上游混合回复 → 剥离下发 agent → 记台账 → 下一轮入站拼回去

2026-08-27 勘察实测：全仓唯一跑 `intercept_*_stream` 端到端的文件是
`test_synth_stream_shape.py`，而它走的是**纯 bladex** 路径。
**混合路径三协议零端到端覆盖。**

这正是 MQ-L23 能藏住的原因：那条缺陷（内循环最终回复的 agent tool_calls
被丢弃）在单元级看不出来——每个零件都对，装起来漏了一环。

## 判据

每个形态钉四件事，缺一不可：

1. **剥离**：agent 收不到 `bladex_*` 调用；它自己的调用**原样保留**（id 不变）
2. **记账**：`SpliceRecord` 真的被记下（否则下一轮无从拼回）
3. **拼接**：下一轮入站时 bladex 调用与结果被恢复进消息流
4. **锚失即弃**：agent 压缩掉了那条 assistant 消息 ⇒ 静默丢弃，不报错

⚠️ 本文件只覆盖 **mixed**。V-P5 的十八形态矩阵（×三分流）另立。
"""
from __future__ import annotations

import asyncio
import json

import pytest

from bladex_proxy.agency import (
    AgencyRuntime,
    intercept_anthropic_stream,
    intercept_chat_stream,
    intercept_responses_stream,
)

BLADEX_CALL = {"id": "call_bx1", "type": "function",
               "function": {"name": "bladex_memory_search",
                            "arguments": '{"query":"deploy"}'}}
AGENT_CALL = {"id": "call_shell1", "type": "function",
              "function": {"name": "shell", "arguments": '{"cmd":"ls"}'}}
SESSION_PREFIX = "u/codex/s/"


@pytest.fixture(autouse=True)
def _modules_on(monkeypatch):
    for k in ("BLADEX_MODULE_TOOLFACE", "BLADEX_MODULE_INTERCEPTION",
              "BLADEX_MODULE_LEDGER"):
        monkeypatch.setenv(k, "1")
    monkeypatch.setenv("BLADEX_STREAM_KEEPALIVE_S", "0")
    monkeypatch.setenv("BLADEX_NARRATE_INTERCEPT", "0")


class _Cap:
    full_text = ""
    reasoning_text = ""
    usage: dict = {}
    finish_reason = None

    def __init__(self):
        self.tool_events: list = []


def _sse(name: str | None, obj: dict) -> bytes:
    head = f"event: {name}\n" if name else ""
    return (head + "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode()


def _sse_text(obj: dict) -> str:
    """chat 协议吃 `str`，另两条吃 `bytes` —— 生产就是这个不对称
    （`server.py` chat 路径在末尾才 `.encode()`）。测试按真实形态给，
    不去"统一"它：统一是另一件事，混进本卡会让判据不纯。"""
    return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"


# ── 三协议的「混合上游流」构造器 ────────────────────────────────────────


async def _chat_mixed():
    """chat：一个 bladex 调用 + 一个 agent 调用，tool_calls delta 带 index。"""
    base = {"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": "m"}
    for i, tc in enumerate((BLADEX_CALL, AGENT_CALL)):
        yield _sse_text({**base, "choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": i, "id": tc["id"], "type": "function",
             "function": {"name": tc["function"]["name"], "arguments": ""}}]},
            "finish_reason": None}]})
        yield _sse_text({**base, "choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": i, "function": {"arguments": tc["function"]["arguments"]}}]},
            "finish_reason": None}]})
    yield _sse_text({**base, "choices": [{"index": 0, "delta": {},
                                          "finish_reason": "tool_calls"}]})
    yield "data: [DONE]\n\n"


async def _anthropic_mixed():
    yield _sse("message_start", {"type": "message_start", "message": {
        "id": "m1", "type": "message", "role": "assistant", "model": "m",
        "content": [], "stop_reason": None, "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1}}})
    for i, tc in enumerate((BLADEX_CALL, AGENT_CALL)):
        yield _sse("content_block_start", {
            "type": "content_block_start", "index": i,
            "content_block": {"type": "tool_use", "id": tc["id"],
                              "name": tc["function"]["name"], "input": {}}})
        yield _sse("content_block_delta", {
            "type": "content_block_delta", "index": i,
            "delta": {"type": "input_json_delta",
                      "partial_json": tc["function"]["arguments"]}})
        yield _sse("content_block_stop", {"type": "content_block_stop", "index": i})
    yield _sse("message_delta", {"type": "message_delta",
                                 "delta": {"stop_reason": "tool_use"},
                                 "usage": {"output_tokens": 2}})
    yield _sse("message_stop", {"type": "message_stop"})


async def _responses_mixed():
    def resp(out, status):
        return {"id": "r1", "object": "response", "created_at": 0, "model": "m",
                "status": status, "output": out, "parallel_tool_calls": False,
                "tool_choice": "auto", "tools": []}

    items = []
    yield _sse("response.created",
               {"type": "response.created", "response": resp([], "in_progress")})
    for i, tc in enumerate((BLADEX_CALL, AGENT_CALL)):
        item = {"type": "function_call", "id": f"fc{i}", "call_id": tc["id"],
                "name": tc["function"]["name"],
                "arguments": tc["function"]["arguments"], "status": "completed"}
        items.append(item)
        yield _sse("response.output_item.added", {
            "type": "response.output_item.added", "output_index": i,
            "item": {**item, "arguments": "", "status": "in_progress"}})
        yield _sse("response.function_call_arguments.delta", {
            "type": "response.function_call_arguments.delta",
            "item_id": f"fc{i}", "output_index": i,
            "delta": tc["function"]["arguments"]})
        yield _sse("response.output_item.done", {
            "type": "response.output_item.done", "output_index": i, "item": item})
    yield _sse("response.completed",
               {"type": "response.completed", "response": resp(items, "completed")})


# ── 收流 + 解析出「agent 实际看到的工具调用」 ──────────────────────────


def _collect(agen) -> list[dict]:
    async def run():
        out = []
        async for c in agen:
            out.append(c if isinstance(c, bytes) else str(c).encode())
        return out
    out = []
    for chunk in asyncio.run(run()):
        for ln in chunk.decode("utf-8", "replace").splitlines():
            if ln.startswith("data: ") and ln[6:].strip() != "[DONE]":
                out.append(json.loads(ln[6:]))
    return out


def _tools_seen_chat(evs) -> list[tuple[int, str]]:
    seen = []
    for e in evs:
        for tc in ((e.get("choices") or [{}])[0].get("delta") or {}).get("tool_calls") or []:
            if tc.get("function", {}).get("name"):
                seen.append((tc["index"], tc["function"]["name"]))
    return seen


def _tools_seen_anthropic(evs) -> list[tuple[int, str]]:
    return [(e["index"], e["content_block"]["name"]) for e in evs
            if e.get("type") == "content_block_start"
            and (e.get("content_block") or {}).get("type") == "tool_use"]


def _tools_seen_responses(evs) -> list[tuple[int, str]]:
    return [(e["output_index"], e["item"]["name"]) for e in evs
            if e.get("type") == "response.output_item.added"
            and (e.get("item") or {}).get("type") == "function_call"]


MATRIX = [
    pytest.param("chat", intercept_chat_stream, _chat_mixed, _tools_seen_chat,
                 id="chat·流式"),
    pytest.param("anthropic", intercept_anthropic_stream, _anthropic_mixed,
                 _tools_seen_anthropic, id="messages·流式"),
    pytest.param("responses", intercept_responses_stream, _responses_mixed,
                 _tools_seen_responses, id="responses·流式"),
]


async def _never_called(_msgs):
    """mixed 不该走内循环 —— 走到这里说明分流判错了。"""
    raise AssertionError("mixed 形态不应触发内循环 call_llm")


def _run_stream(fn, source, agency):
    return _collect(fn(
        source(), agency=agency, capture_result=_Cap(),
        session_prefix=SESSION_PREFIX, allowed_exposure="local",
        session_id="s", agent_id="codex", call_llm=_never_called,
        upstream_messages=[{"role": "user", "content": "干活"}]))


@pytest.mark.parametrize("proto,fn,source,tools_of", MATRIX)
class TestStreamingMixed:
    """流式 × 三协议：剥离 + 记账。"""

    def test_bladex_call_stripped_agent_call_kept(self, proto, fn, source, tools_of):
        evs = _run_stream(fn, source, AgencyRuntime())
        seen = tools_of(evs)
        names = [n for _i, n in seen]
        assert not any(n.startswith("bladex_") for n in names), (
            f"{proto}: agent 看到了 bladex 调用 —— 它历史里会留下悬空调用：{names}")
        assert "shell" in names, f"{proto}: agent 自己的调用被误删了：{names}"

    def test_kept_call_renumbered_from_zero(self, proto, fn, source, tools_of):
        """🔴 剥掉 index 0 之后，留下的必须重编号成 0。

        不重编号 = agent 侧解析器按 index 聚合时留一个空洞，
        轻则参数拼不上，重则整条消息解析失败。
        """
        seen = tools_of(_run_stream(fn, source, AgencyRuntime()))
        idxs = [i for i, _n in seen]
        assert idxs == list(range(len(idxs))), f"{proto}: 编号不连续 {idxs}"

    def test_splice_record_written(self, proto, fn, source, tools_of):
        """记账：不记就没法在下一轮拼回去 —— LLM 白查一次。"""
        ag = AgencyRuntime()
        _run_stream(fn, source, ag)
        recs = ag.splice.new_in_turn(SESSION_PREFIX)
        assert recs, f"{proto}: 剥了却没记台账"
        assert [c["function"]["name"] for c in recs[0].calls] == ["bladex_memory_search"]


class TestNonStreamingMixed:
    """非流式：`process_message` 是三协议共用的分流出口。"""

    def _run(self, agency, message):
        async def dispatch(_n, _a, **_kw):
            return "result text"
        agency.toolface.dispatch = dispatch          # type: ignore[method-assign]
        return asyncio.run(agency.process_message(
            message, upstream_messages=[{"role": "user", "content": "干活"}],
            session_prefix=SESSION_PREFIX, allowed_exposure="local",
            call_llm=None, session_id="s", agent_id="codex"))

    def _mixed_message(self):
        return {"role": "assistant", "content": "先查一下",
                "tool_calls": [dict(BLADEX_CALL), dict(AGENT_CALL)]}

    def test_strips_bladex_keeps_agent(self):
        out, _transcript, mode = self._run(AgencyRuntime(), self._mixed_message())
        names = [tc["function"]["name"] for tc in out.get("tool_calls") or []]
        assert names == ["shell"], f"剥离结果不对：{names}"
        assert mode == "mixed"

    def test_agent_call_id_preserved(self):
        """id 必须逐字保留 —— agent 用它匹配自己的 tool 结果。"""
        out, _t, _m = self._run(AgencyRuntime(), self._mixed_message())
        assert out["tool_calls"][0]["id"] == "call_shell1"

    def test_splice_record_written(self):
        ag = AgencyRuntime()
        self._run(ag, self._mixed_message())
        recs = ag.splice.new_in_turn(SESSION_PREFIX)
        assert recs and recs[0].results, "非流式路径没记结果"


class TestRoundTrip:
    """完整往返：剥离 → 记账 → 下一轮拼回去。这是本文件的核心判据。"""

    def _stripped_and_ledger(self):
        ag = AgencyRuntime()

        async def dispatch(_n, _a, **_kw):
            return "deploy notes: use run_proxy.sh"
        ag.toolface.dispatch = dispatch              # type: ignore[method-assign]
        out, _t, _m = asyncio.run(ag.process_message(
            {"role": "assistant", "content": "先查一下",
             "tool_calls": [dict(BLADEX_CALL), dict(AGENT_CALL)]},
            upstream_messages=[{"role": "user", "content": "干活"}],
            session_prefix=SESSION_PREFIX, allowed_exposure="local",
            call_llm=None, session_id="s", agent_id="codex"))
        return ag, out

    def test_next_turn_splices_back(self):
        ag, stripped = self._stripped_and_ledger()
        # agent 回来了：带上它自己的工具结果，历史里是**剥离后**那条 assistant
        inbound = [
            {"role": "user", "content": "干活"},
            stripped,
            {"role": "tool", "tool_call_id": "call_shell1", "content": "a.py b.py"},
        ]
        out = ag.prepare_inbound(inbound, SESSION_PREFIX)
        names = [tc["function"]["name"]
                 for m in out if m.get("role") == "assistant"
                 for tc in m.get("tool_calls") or []]
        assert "bladex_memory_search" in names, "bladex 调用没被拼回去"
        tool_contents = [m.get("content") for m in out if m.get("role") == "tool"]
        assert any("deploy notes" in str(c) for c in tool_contents), (
            "bladex 的结果没送到下一轮 LLM —— 这一次查询白花了")

    def test_anchor_lost_is_dropped_silently(self):
        """agent 自行压缩掉那条 assistant ⇒ 静默丢弃（ADR §3.2 硬点 2）。"""
        ag, _stripped = self._stripped_and_ledger()
        inbound = [{"role": "user", "content": "干活"},
                   {"role": "assistant", "content": "（被 agent 压缩过的别的内容）"}]
        out = ag.prepare_inbound(inbound, SESSION_PREFIX)
        names = [tc["function"]["name"]
                 for m in out if m.get("role") == "assistant"
                 for tc in m.get("tool_calls") or []]
        assert not names, "锚没了还硬拼 = 往 agent 历史里塞悬空调用"

    def test_agent_history_stays_self_consistent(self):
        """🔴 优雅降级的实质判据：拼接失败时 agent 侧历史仍然自洽。

        「自洽」= 每个 tool 结果都有对应的调用，没有悬空的一方。
        ADR §3.2 的整个降级论证都建立在这条上。
        """
        ag, stripped = self._stripped_and_ledger()
        ag.splice._by_session.clear()                 # 模拟拼接状态全丢
        inbound = [{"role": "user", "content": "干活"}, stripped,
                   {"role": "tool", "tool_call_id": "call_shell1", "content": "a.py"}]
        out = ag.prepare_inbound(inbound, SESSION_PREFIX)
        call_ids = {tc["id"] for m in out if m.get("role") == "assistant"
                    for tc in m.get("tool_calls") or []}
        result_ids = {m["tool_call_id"] for m in out if m.get("role") == "tool"}
        assert result_ids <= call_ids, f"有结果没调用：{result_ids - call_ids}"
        assert call_ids <= result_ids, f"有调用没结果：{call_ids - result_ids}"
