"""合成响应的 SSE 形态守卫（2026-08-27 Codex 断连事故驱动）。

事故：内循环跑完 → 合成响应发出 → **Codex 会话直接结束，零后续请求**，
一天四次（12:04 / 15:31 / 16:47 / 19:57）。排查中先后归因给播报形态、
keepalive、`GeneratorExit`，三次都错。

根因：`_synth_responses_text_events` 发 `output_item.added(content=[])`
之后**直接**发 `output_text.delta`，缺 `response.content_part.added`。
官方解析器 `accumulate_event`：

    content_part.added  → output.content.append(part)
    output_text.delta   → content = output.content[content_index]   ← IndexError

⇒ 客户端流解析崩 ⇒ 断连。

🔴 本文件的判据**不是我写的规则**，是 OpenAI SDK 真解析器的状态机。
装了 `openai` 就用真解析器跑；没装则退化成结构断言（CI 最小环境）。
两种模式都必须能抓住缺陷——见 `TestDiscriminatingPower`。

⚠️ Anthropic 侧**没有**同型缺陷：它的累加器对 `content_block_start` 做
`append`，不按 index 定位。排查中我曾用**自己假设的取值**（播报@1）
构造测试、判它有缺陷——生产真值是播报@0。**尺子必须读生产的真实取值。**
"""
from __future__ import annotations

import json

import pytest

from bladex_proxy.agency import (
    _synth_anthropic_text_events,
    _synth_responses_text_events,
)

try:  # 真解析器可用则用真的
    import inspect as _inspect

    from openai._models import construct_type_unchecked as _construct
    from openai.lib.streaming.responses._responses import ResponseStreamState
    from openai.types.responses import ResponseStreamEvent

    _OPENAI_KW = {
        k: None
        for k in _inspect.signature(ResponseStreamState.__init__).parameters
        if k != "self"
    }
    if "input_tools" in _OPENAI_KW:
        _OPENAI_KW["input_tools"] = []
    HAVE_OPENAI = True
except Exception:  # noqa: BLE001 — 最小环境没有 openai 是允许的
    HAVE_OPENAI = False


def _parse(chunks) -> list[dict]:
    out = []
    for c in chunks:
        for line in c.decode("utf-8").splitlines():
            if line.startswith("data: "):
                out.append(json.loads(line[6:]))
    return out


def _feed_real_parser(events: list[dict]) -> None:
    """喂进 OpenAI 官方流解析器。抛异常 = 客户端会断连。

    `response.created` 由上游发、不属本函数产物，这里补一个最小的让状态机起步。
    """
    created = {
        "type": "response.created", "sequence_number": 0,
        "response": {"id": "r1", "object": "response", "created_at": 0, "model": "m",
                     "status": "in_progress", "output": [], "parallel_tool_calls": False,
                     "tool_choice": "auto", "tools": []},
    }
    st = ResponseStreamState(**_OPENAI_KW)
    for i, ev in enumerate([created, *events]):
        # `response.completed` 走 parse_response，需要 text_format，
        # 本 harness 给不了 ⇒ 教科书形态也会报 TypeError（已实测），故跳过。
        if ev.get("type") == "response.completed":
            continue
        st.handle_event(_construct(type_=ResponseStreamEvent,
                                   value={**ev, "sequence_number": i}))


class TestResponsesSynthShape:
    """Responses 合成消息必须是完整的五段。"""

    def test_emits_content_part_before_delta(self):
        evs = _parse(_synth_responses_text_events(
            [{"content": "账本已切换"}], raw=True, index=0, sink=[]))
        types = [e["type"] for e in evs]
        assert types == [
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.content_part.done",
            "response.output_item.done",
        ], f"合成形态与 probe 参考不一致：{types}"

    def test_added_item_is_in_progress_with_empty_content(self):
        evs = _parse(_synth_responses_text_events(
            [{"content": "x"}], raw=True, index=0, sink=[]))
        item = evs[0]["item"]
        assert item["status"] == "in_progress"
        assert item["content"] == []
        # done 时才是 completed，且带正文
        done = evs[-1]["item"]
        assert done["status"] == "completed"
        assert done["content"][0]["text"] == "x"

    def test_indices_consistent(self):
        evs = _parse(_synth_responses_text_events(
            [{"content": "x"}], raw=True, index=3, sink=[]))
        assert {e.get("output_index") for e in evs} == {3}
        assert {e["content_index"] for e in evs if "content_index" in e} == {0}

    @pytest.mark.skipif(not HAVE_OPENAI, reason="openai SDK 不可用")
    def test_real_parser_accepts(self):
        """判据来自官方状态机，不是我写的规则。"""
        _feed_real_parser(_parse(_synth_responses_text_events(
            [{"content": "账本已切换"}], raw=True, index=0, sink=[])))


class TestAnthropicSynthShape:
    """Anthropic 侧是完整的三段，且 index 硬编码 0 是**对的**（累加器 append）。"""

    def test_three_segment_block(self):
        types = [e["type"] for e in _parse(
            _synth_anthropic_text_events([{"content": "x"}], raw=True, index=7, sink=[]))]
        assert types == ["content_block_start", "content_block_delta",
                         "content_block_stop"]

    def test_index_param_deliberately_ignored(self):
        """传 7 也发 0——这是经真累加器验证的正确行为，不是 bug。

        改成"尊重 index"会让这条红，届时请先读 `_synth_anthropic_text_events`
        里那段注释，再决定是不是真要改。
        """
        evs = _parse(_synth_anthropic_text_events(
            [{"content": "x"}], raw=True, index=7, sink=[]))
        assert {e["index"] for e in evs} == {0}


def _upstream_pure_bladex():
    """事故当天的真实形态：上游首轮只回一个 `bladex_*` 调用 ⇒ 走内循环。"""
    def ev(n, o):
        return (f"event: {n}\ndata: {json.dumps(o)}\n\n").encode()

    def resp(out, status):
        return {"id": "r1", "object": "response", "created_at": 0, "model": "m",
                "status": status, "output": out, "parallel_tool_calls": False,
                "tool_choice": "auto", "tools": []}

    fc = {"type": "function_call", "id": "fc1", "call_id": "c1",
          "name": "bladex_ledger_switch", "arguments": '{"ledger_id":""}'}

    async def gen():
        yield ev("response.created",
                 {"type": "response.created", "response": resp([], "in_progress")})
        yield ev("response.output_item.added",
                 {"type": "response.output_item.added", "output_index": 0,
                  "item": {**fc, "arguments": ""}})
        yield ev("response.output_item.done",
                 {"type": "response.output_item.done", "output_index": 0, "item": fc})
        yield ev("response.completed",
                 {"type": "response.completed", "response": resp([fc], "completed")})
    return gen()


class _Cap:
    full_text = ""
    reasoning_text = ""
    usage: dict = {}
    finish_reason = None

    def __init__(self):
        self.tool_events: list = []


@pytest.mark.skipif(not HAVE_OPENAI, reason="openai SDK 不可用")
@pytest.mark.parametrize("narrate", [False, True], ids=["播报关", "播报开"])
def test_end_to_end_stream_parses(monkeypatch, narrate):
    """整条拦截流喂进官方解析器——两种播报开关都必须解析通过。

    🔴 播报开时，reasoning item 占掉一个 `output_index`，合成消息必须往后让；
    且 reasoning item 必须出现在 `response.completed` 快照里（08-25 对账事故）。
    """
    import asyncio

    from bladex_proxy.agency import AgencyRuntime, intercept_responses_stream

    for k in ("BLADEX_MODULE_TOOLFACE", "BLADEX_MODULE_INTERCEPTION",
              "BLADEX_MODULE_LEDGER"):
        monkeypatch.setenv(k, "1")
    monkeypatch.setenv("BLADEX_STREAM_KEEPALIVE_S", "0")
    monkeypatch.setenv("BLADEX_NARRATE_INTERCEPT", "1" if narrate else "0")

    async def call_llm(_messages):
        return {"role": "assistant", "content": "账本已切换。", "tool_calls": []}

    async def collect():
        buf = b""
        async for c in intercept_responses_stream(
                _upstream_pure_bladex(), agency=AgencyRuntime(), capture_result=_Cap(),
                session_prefix="u/codex/s/", allowed_exposure="local", session_id="s",
                agent_id="codex",
                upstream_messages=[{"role": "user", "content": "干活"}],
                call_llm=call_llm):
            buf += c if isinstance(c, bytes) else str(c).encode()
        return buf

    evs = _parse([asyncio.run(collect())])
    _feed_real_parser(evs)          # 抛异常 = 客户端会断连

    final = [e for e in evs if e.get("type") == "response.completed"]
    assert final, "缺终止事件"
    kinds = [i.get("type") for i in final[-1]["response"]["output"]]
    assert kinds == (["reasoning", "message"] if narrate else ["message"]), (
        f"终止快照与流内 item 对不上：{kinds}")


@pytest.mark.skipif(not HAVE_OPENAI, reason="openai SDK 不可用")
def test_inner_loop_forwards_agent_tool_calls(monkeypatch):
    """🔴 MQ-L23：内循环最终回复里 agent 自己的 tool_calls 必须转发出去。

    丢掉它们 ⇒ 客户端收到一个"没有工具调用的助手消息" = 这回合完成 ⇒ 收工。
    2026-08-27 Codex 一天五次「内循环之后零后续请求」就是这条。
    """
    import asyncio

    from bladex_proxy.agency import AgencyRuntime, intercept_responses_stream

    for k in ("BLADEX_MODULE_TOOLFACE", "BLADEX_MODULE_INTERCEPTION",
              "BLADEX_MODULE_LEDGER"):
        monkeypatch.setenv(k, "1")
    monkeypatch.setenv("BLADEX_STREAM_KEEPALIVE_S", "0")
    monkeypatch.setenv("BLADEX_NARRATE_INTERCEPT", "0")

    async def call_llm(_messages):
        """模型切完账本后要真干活——调 agent 自己的 shell。"""
        return {"role": "assistant", "content": "账本已切换，开始干活。",
                "tool_calls": [{"id": "call_shell_1", "type": "function",
                                "function": {"name": "shell",
                                             "arguments": '{"cmd":"ls"}'}}]}

    async def collect():
        buf = b""
        async for c in intercept_responses_stream(
                _upstream_pure_bladex(), agency=AgencyRuntime(), capture_result=_Cap(),
                session_prefix="u/codex/s/", allowed_exposure="local", session_id="s",
                agent_id="codex",
                upstream_messages=[{"role": "user", "content": "干活"}],
                call_llm=call_llm):
            buf += c if isinstance(c, bytes) else str(c).encode()
        return buf

    evs = _parse([asyncio.run(collect())])
    _feed_real_parser(evs)

    # ① function_call 三段事件真的发了
    fc_added = [e for e in evs if e.get("type") == "response.output_item.added"
                and (e.get("item") or {}).get("type") == "function_call"]
    assert len(fc_added) == 1, "agent 的 tool_call 没有被转发"
    assert fc_added[0]["item"]["name"] == "shell"
    assert any(e.get("type") == "response.function_call_arguments.done"
               and e["arguments"] == '{"cmd":"ls"}' for e in evs)

    # ② 编号不与合成消息撞
    msg_idx = [e["output_index"] for e in evs
               if e.get("type") == "response.output_item.added"
               and (e.get("item") or {}).get("type") == "message"]
    assert fc_added[0]["output_index"] not in msg_idx

    # ③ 终止快照里两者都在，且 bladex_ 自己的调用已被剔除
    final = [e for e in evs if e.get("type") == "response.completed"][-1]
    out = final["response"]["output"]
    assert [i.get("type") for i in out] == ["message", "function_call"]
    assert not any(str(i.get("name", "")).startswith("bladex_") for i in out)


class TestDiscriminatingPower:
    """证明上面的测试**抓得住**缺陷，而不只是"绿着"。

    2026-08-27 教训：判别力与通过率是两件事。这里把事故当天的坏形态
    直接构造出来，验证它确实过不了。
    """

    @pytest.mark.skipif(not HAVE_OPENAI, reason="openai SDK 不可用")
    def test_missing_content_part_is_rejected_by_real_parser(self):
        """事故当天的形态：跳过 content_part.added。"""
        item = {"type": "message", "id": "msg_bladex", "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "x", "annotations": []}]}
        broken = [
            {"type": "response.output_item.added", "output_index": 0,
             "item": {**item, "content": []}},
            {"type": "response.output_text.delta", "item_id": "msg_bladex",
             "output_index": 0, "content_index": 0, "delta": "x", "logprobs": []},
        ]
        with pytest.raises(IndexError):
            _feed_real_parser(broken)

    @pytest.mark.skipif(not HAVE_OPENAI, reason="openai SDK 不可用")
    def test_index_collision_is_rejected_by_real_parser(self):
        """播报 reasoning item 与合成消息共用 output_index 的形态。"""
        broken = [
            {"type": "response.output_item.added", "output_index": 0,
             "item": {"type": "reasoning", "id": "rs_bladex", "summary": []}},
            {"type": "response.content_part.added", "item_id": "msg_bladex",
             "output_index": 0, "content_index": 0,
             "part": {"type": "output_text", "text": "", "annotations": []}},
            {"type": "response.output_text.delta", "item_id": "msg_bladex",
             "output_index": 0, "content_index": 0, "delta": "x", "logprobs": []},
        ]
        with pytest.raises((AssertionError, IndexError)):
            _feed_real_parser(broken)
