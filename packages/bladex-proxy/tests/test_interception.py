"""V-P2 验收剧本：三分流 / 非流式剥离 / 三协议流式剥离与重编号。"""

from __future__ import annotations

from bladex_proxy.interception import (
    MODE_MIXED,
    MODE_NONE,
    MODE_PURE,
    AnthropicStreamStripper,
    ChatStreamStripper,
    ResponsesStreamStripper,
    classify_message,
    strip_bladex_calls,
)


def _tc(name, cid, args="{}"):
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": args}}


class TestClassify:
    def test_none(self):
        d = classify_message({"role": "assistant",
                              "tool_calls": [_tc("web_search", "c1")]})
        assert d.mode == MODE_NONE

    def test_pure(self):
        d = classify_message({"role": "assistant", "content": None,
                              "tool_calls": [_tc("bladex_memory_search", "c1")]})
        assert d.mode == MODE_PURE
        assert len(d.bladex_calls) == 1

    def test_mixed_by_agent_call(self):
        d = classify_message({"role": "assistant", "content": "",
                              "tool_calls": [_tc("bladex_memory_search", "c1"),
                                             _tc("web_search", "c2")]})
        assert d.mode == MODE_MIXED

    def test_bladex_call_with_preamble_text_is_pure(self):
        """🔴 2026-08-26 行为变更（live 病例：Hermes"看起来卡死"）。

        **旧行为**：有正文 ⇒ MIXED（当时的理由写在这里："剥离后仍可转发，
        是 hermes 空响应重试指纹的反面"）。那个顾虑本身没错，但判据错了——
        它把"还剩能转发的东西"当成了"这一轮该结束了"。

        实测：模型返回 `"…先把证据归档到任务账本，再给结论。"` + 两个
        `bladex_ledger_update`、零 agent 调用 ⇒ 判 MIXED ⇒ 剥掉我方调用后
        转发的只剩这段前言，且转发调用为 0 ⇒ `finish_reason` 改写成 `stop`
        ⇒ **Hermes 认为回合结束**，用户看到"说要给结论然后没了"。

        **前言不是回答。** 只要 agent 自己没有调用，这一轮就不可能完整。
        改判 PURE 走内循环——空响应的顾虑由内循环兜底（它一定合成一条可转发消息，
        见 `test_pure_never_yields_empty_message`），两头都保住。
        """
        d = classify_message({"role": "assistant",
                              "content": "老板，我先把证据归档到任务账本，再给结论。",
                              "tool_calls": [_tc("bladex_memory_search", "c1")]})
        assert d.mode == MODE_PURE
        assert len(d.bladex_calls) == 1 and not d.agent_calls

    def test_agent_call_still_makes_it_mixed_even_without_text(self):
        """阳性对照：agent 自己有调用时仍是 MIXED——否则上面那条只是
        "永远 PURE"，什么也没证明。"""
        d = classify_message({"role": "assistant", "content": "",
                              "tool_calls": [_tc("bladex_memory_search", "c1"),
                                             _tc("write_file", "a1")]})
        assert d.mode == MODE_MIXED and len(d.agent_calls) == 1

    def test_plain_text(self):
        assert classify_message({"role": "assistant", "content": "hi"}).mode == MODE_NONE


class TestNonStreamStrip:
    def test_strip_keeps_agent_calls_order_and_ids(self):
        msg = {"role": "assistant", "content": "x",
               "tool_calls": [_tc("web_search", "a1"),
                              _tc("bladex_memory_search", "b1"),
                              _tc("write_file", "a2")]}
        out, removed = strip_bladex_calls(msg)
        assert [tc["id"] for tc in out["tool_calls"]] == ["a1", "a2"]
        assert [tc["id"] for tc in removed] == ["b1"]
        assert msg["tool_calls"][1]["id"] == "b1"  # 入参不改

    def test_strip_all_removes_field(self):
        out, removed = strip_bladex_calls(
            {"role": "assistant", "tool_calls": [_tc("bladex_ledger_read", "b1")]})
        assert "tool_calls" not in out and len(removed) == 1


def _chat_chunk(tcs):
    return {"choices": [{"index": 0, "delta": {"tool_calls": tcs},
                         "finish_reason": None}]}


class TestChatStreamStrip:
    def test_reindex_and_capture(self):
        s = ChatStreamStripper()
        # index0=agent, index1=bladex, index2=agent
        out = []
        out += s.feed(_chat_chunk([{"index": 0, "id": "a1",
                                    "function": {"name": "web_search", "arguments": ""}}]))
        out += s.feed(_chat_chunk([{"index": 1, "id": "b1",
                                    "function": {"name": "bladex_memory_search",
                                                 "arguments": ""}}]))
        out += s.feed(_chat_chunk([{"index": 1,
                                    "function": {"arguments": "{\"query\":\"x\"}"}}]))
        out += s.feed(_chat_chunk([{"index": 2, "id": "a2",
                                    "function": {"name": "write_file", "arguments": "{}"}}]))
        emitted = [tc for ch in out for tc in ch["choices"][0]["delta"]["tool_calls"]]
        assert [(tc["index"], tc.get("id")) for tc in emitted] == [(0, "a1"), (1, "a2")]
        assert s.removed_calls == [{"id": "b1", "type": "function",
                                    "function": {"name": "bladex_memory_search",
                                                 "arguments": "{\"query\":\"x\"}"}}]

    def test_pure_bladex_chunks_all_swallowed(self):
        s = ChatStreamStripper()
        out = s.feed(_chat_chunk([{"index": 0, "id": "b1",
                                   "function": {"name": "bladex_ledger_read",
                                                "arguments": "{}"}}]))
        assert out == []
        assert len(s.removed_calls) == 1

    def test_non_tool_chunks_pass_through(self):
        s = ChatStreamStripper()
        chunk = {"choices": [{"index": 0, "delta": {"content": "hi"},
                              "finish_reason": None}]}
        assert s.feed(chunk) == [chunk]


class TestAnthropicStreamStrip:
    def test_block_swallow_and_reindex(self):
        s = AnthropicStreamStripper()
        out = []
        out += s.feed({"type": "content_block_start", "index": 0,
                       "content_block": {"type": "text", "text": ""}})
        out += s.feed({"type": "content_block_start", "index": 1,
                       "content_block": {"type": "tool_use", "id": "b1",
                                         "name": "bladex_memory_search", "input": {}}})
        out += s.feed({"type": "content_block_delta", "index": 1,
                       "delta": {"type": "input_json_delta", "partial_json": "{}"}})
        out += s.feed({"type": "content_block_stop", "index": 1})
        out += s.feed({"type": "content_block_start", "index": 2,
                       "content_block": {"type": "tool_use", "id": "a1",
                                         "name": "web_search", "input": {}}})
        starts = [e for e in out if e["type"] == "content_block_start"]
        assert [(e["index"], e["content_block"].get("name")) for e in starts] == \
            [(0, None), (1, "web_search")]
        assert s.removed_calls[0]["function"]["name"] == "bladex_memory_search"

    def test_other_events_pass(self):
        s = AnthropicStreamStripper()
        ev = {"type": "message_start", "message": {}}
        assert s.feed(ev) == [ev]


class TestResponsesIndexConsistency:
    """🔴 2026-08-25 Codex live 事故回归：**所有**带 output_index 的事件都必须
    跟随同一张重编号表。首版只重编号 item/arguments 两族，`output_text.delta`
    与 `content_part.*` 原样放行 ⇒ item 被改到 index 0、文本增量仍说 index 1
    ⇒ Codex 对不上即断连（用户侧：输出一段话就中断）。"""

    def test_text_events_follow_renumbering(self):
        s = ResponsesStreamStripper()
        out = []
        # index0 = bladex 调用（被拦），index1 = message（重编号到 0）
        out += s.feed({"type": "response.output_item.added", "output_index": 0,
                       "item": {"type": "function_call", "call_id": "b1",
                                "name": "bladex_ledger_switch", "arguments": ""}})
        out += s.feed({"type": "response.output_item.added", "output_index": 1,
                       "item": {"type": "message", "id": "m0", "content": []}})
        out += s.feed({"type": "response.content_part.added", "output_index": 1,
                       "item_id": "m0", "content_index": 0,
                       "part": {"type": "output_text", "text": ""}})
        out += s.feed({"type": "response.output_text.delta", "output_index": 1,
                       "item_id": "m0", "content_index": 0, "delta": "答案"})
        out += s.feed({"type": "response.content_part.done", "output_index": 1,
                       "item_id": "m0", "content_index": 0,
                       "part": {"type": "output_text", "text": "答案"}})
        idxs = {e["type"]: e["output_index"] for e in out}
        assert idxs == {
            "response.output_item.added": 0,
            "response.content_part.added": 0,
            "response.output_text.delta": 0,
            "response.content_part.done": 0,
        }, f"index 不一致 ⇒ Codex 会断连: {idxs}"

    def test_text_events_of_intercepted_item_are_swallowed(self):
        s = ResponsesStreamStripper()
        s.feed({"type": "response.output_item.added", "output_index": 0,
                "item": {"type": "function_call", "call_id": "b1",
                         "name": "bladex_memory_search", "arguments": ""}})
        # 假如被拦 item 也带文本类事件，必须一并吞掉
        assert s.feed({"type": "response.output_text.delta", "output_index": 0,
                       "item_id": "x", "delta": "leak"}) == []

    def test_response_level_events_pass_through(self):
        s = ResponsesStreamStripper()
        for t in ("response.created", "response.completed"):
            ev = {"type": t, "response": {"id": "r"}}
            assert s.feed(ev) == [ev]


class TestResponsesStreamStrip:
    def test_item_swallow_and_reindex(self):
        s = ResponsesStreamStripper()
        out = []
        out += s.feed({"type": "response.output_item.added", "output_index": 0,
                       "item": {"type": "function_call", "call_id": "b1",
                                "name": "bladex_ledger_read", "arguments": ""}})
        out += s.feed({"type": "response.function_call_arguments.delta",
                       "output_index": 0, "delta": "{}"})
        out += s.feed({"type": "response.output_item.done", "output_index": 0,
                       "item": {"type": "function_call", "call_id": "b1",
                                "name": "bladex_ledger_read", "arguments": "{}"}})
        out += s.feed({"type": "response.output_item.added", "output_index": 1,
                       "item": {"type": "message", "content": []}})
        assert [e["type"] for e in out] == ["response.output_item.added"]
        assert out[0]["output_index"] == 0  # 重编号：1 → 0
        assert s.removed_calls[0]["function"]["arguments"] == "{}"
