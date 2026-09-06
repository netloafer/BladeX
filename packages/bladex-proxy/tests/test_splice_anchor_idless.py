"""MQ-P8：上游流式 delta 不带 tool_call id 时的拼接锚对称性。

真实病例（2026-08-28，CC × deepseek-v4-flash，哈希级取证）：
capture 看到的上游形态 id 为空 ⇒ 记录侧锚走"无 id 退化哈希"；
我们把空 id 发给 agent，agent（CC）自己铸一个回传 ⇒ 回传侧锚走 id 路径
⇒ 结构性永不相等，anthropic/responses 拼接恢复率 0%（`spliced=0
anchors_dropped=1` 每轮重演）。

修法两条（本文件逐条钉住）：
① **永不外发空 id**——发射权威（三个剥离器）在首个 fragment/block/item 铸；
② 锚从剥离器的**已发射形态**取（forwarded_calls），不从 capture 取。
"""

from __future__ import annotations

from bladex_proxy.interception import (
    AnthropicStreamStripper,
    ChatStreamStripper,
    ResponsesStreamStripper,
    ensure_call_ids,
    mint_call_id,
)
from bladex_proxy.splice import anchor_key


def _chat_chunk(index: int, name: str = "", args: str = "", call_id: str | None = None):
    tc: dict = {"index": index, "function": {}}
    if name:
        tc["function"]["name"] = name
    if args:
        tc["function"]["arguments"] = args
    if call_id is not None:
        tc["id"] = call_id
    return {"choices": [{"delta": {"tool_calls": [tc]}}]}


class TestChatStripperMintsIds:
    def test_idless_kept_call_gets_minted_id_on_wire(self):
        s = ChatStreamStripper()
        out1 = s.feed(_chat_chunk(0, name="Bash", args='{"c'))       # 无 id（deepseek 形态）
        out2 = s.feed(_chat_chunk(0, args='md":"ls"}'))
        s.feed(_chat_chunk(1, name="bladex_memory_search", args="{}"))  # 被拦

        emitted = [tc for o in out1 + out2
                   for tc in o["choices"][0]["delta"]["tool_calls"]]
        assert emitted, "kept 调用必须被转发"
        first_id = emitted[0].get("id")
        assert first_id and first_id.startswith("call_"), \
            f"首 fragment 永不外发空 id（MQ-P8）；实发 {first_id!r}"
        # 最小干预：后续 fragment 不强行加 id，但**不许出现分叉的 id**
        forked = {tc.get("id") for tc in emitted[1:] if tc.get("id")} - {first_id}
        assert not forked, f"同一调用的 fragment 出现分叉 id: {forked}"

        fwd = s.forwarded_calls
        assert len(fwd) == 1 and fwd[0]["id"] == first_id, \
            "记录侧（forwarded_calls）与 wire 必须同源"
        assert fwd[0]["function"]["name"] == "Bash"
        assert fwd[0]["function"]["arguments"] == '{"cmd":"ls"}'

    def test_upstream_id_passthrough_byte_identical(self):
        """none 零差异红线的单元级镜像：上游带 id 的流，kept fragment 除重编号外
        一概不碰（gate 全量里的 test_none_passthrough_matrix 曾抓到首版把后续
        fragment 的 id:null 改写掉——透传不再逐字）。"""
        s = ChatStreamStripper()
        c1 = _chat_chunk(0, name="Bash", call_id="call_upstream1")
        out1 = s.feed(c1)
        assert out1 == [c1], "上游带 id：首 fragment 必须原样"
        c2 = _chat_chunk(0, args='{"cmd":"ls"}')
        c2["choices"][0]["delta"]["tool_calls"][0]["id"] = None   # 上游惯常形态
        out2 = s.feed(c2)
        assert out2 == [c2], "后续 fragment 的 id:null 必须原样透传"
        assert s.forwarded_calls[0]["id"] == "call_upstream1"

    def test_late_upstream_id_cannot_fork_the_wire(self):
        """首 fragment 定终身：迟到的上游 id 不许推翻已发射的铸 id。"""
        s = ChatStreamStripper()
        out1 = s.feed(_chat_chunk(0, name="Bash"))
        minted = out1[0]["choices"][0]["delta"]["tool_calls"][0]["id"]
        out2 = s.feed(_chat_chunk(0, args="{}", call_id="call_late999"))
        assert out2[0]["choices"][0]["delta"]["tool_calls"][0]["id"] == minted

    def test_anchor_symmetry_record_vs_echo(self):
        """锚对称性 = 本修复的全部意义：record 形态 == agent 回传形态。"""
        s = ChatStreamStripper()
        out = s.feed(_chat_chunk(0, name="Bash", args="{}"))
        s.feed(_chat_chunk(1, name="bladex_memory_search", args="{}"))
        wire_id = out[0]["choices"][0]["delta"]["tool_calls"][0]["id"]

        record = {"role": "assistant", "content": "",
                  "tool_calls": list(s.forwarded_calls)}
        echo = {"role": "assistant", "content": None,
                "tool_calls": [{"id": wire_id, "type": "function",
                                "function": {"name": "Bash", "arguments": "{}"}}]}
        assert anchor_key(record) == anchor_key(echo)

        # 判别力：修前的记录形态（capture 无 id + arguments 写死 ""）≠ 回传形态
        old_record = {"role": "assistant", "content": "",
                      "tool_calls": [{"id": "", "type": "function",
                                      "function": {"name": "Bash", "arguments": ""}}]}
        assert anchor_key(old_record) != anchor_key(echo), \
            "修前形态若能锚上，说明本测试没在测真问题"


class TestAnthropicStripperMintsIds:
    def _start(self, idx, btype, name="", bid=""):
        return {"type": "content_block_start", "index": idx,
                "content_block": {"type": btype, "id": bid, "name": name, "input": {}}}

    def test_idless_tool_use_gets_minted_id(self):
        s = AnthropicStreamStripper()
        out = s.feed(self._start(0, "tool_use", name="Bash", bid=""))
        assert out and out[0]["content_block"]["id"].startswith("call_")
        assert s.forwarded_calls[0]["id"] == out[0]["content_block"]["id"]
        # 被拦块照吞
        assert s.feed(self._start(1, "tool_use", name="bladex_ledger_read")) == []
        assert len(s.removed_calls) == 1

    def test_kept_arguments_aggregate_into_forwarded(self):
        s = AnthropicStreamStripper()
        s.feed(self._start(0, "tool_use", name="Bash", bid=""))
        s.feed({"type": "content_block_delta", "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '{"a":1}'}})
        assert s.forwarded_calls[0]["function"]["arguments"] == '{"a":1}'

    def test_existing_id_untouched(self):
        s = AnthropicStreamStripper()
        out = s.feed(self._start(0, "tool_use", name="Bash", bid="call_real1"))
        assert out[0]["content_block"]["id"] == "call_real1"


class TestResponsesStripperMintsIds:
    def _added(self, idx, name, call_id="", item_id="fc_1"):
        return {"type": "response.output_item.added", "output_index": idx,
                "item": {"type": "function_call", "id": item_id,
                         "call_id": call_id, "name": name, "arguments": ""}}

    def test_idless_function_call_minted_and_done_consistent(self):
        s = ResponsesStreamStripper()
        out = s.feed(self._added(0, "Bash", call_id="", item_id="fc_x"))
        minted = out[0]["item"]["call_id"]
        assert minted.startswith("call_")
        done = s.feed({"type": "response.output_item.done", "output_index": 0,
                       "item": {"type": "function_call", "id": "fc_x",
                                "call_id": "", "name": "Bash",
                                "arguments": '{"k":1}'}})
        assert done[0]["item"]["call_id"] == minted, "done 快照必须与 added 同源"
        assert s.forwarded_calls[0]["id"] == minted
        assert s.forwarded_calls[0]["function"]["arguments"] == '{"k":1}'

    def test_snapshot_stamping_by_item_id(self):
        s = ResponsesStreamStripper()
        out = s.feed(self._added(0, "Bash", call_id="", item_id="fc_y"))
        minted = out[0]["item"]["call_id"]
        items = [{"type": "function_call", "id": "fc_y", "call_id": "",
                  "name": "Bash", "arguments": "{}"},
                 {"type": "message", "id": "msg_1"}]
        assert s.stamp_snapshot_items(items) == 1
        assert items[0]["call_id"] == minted

    def test_arg_delta_aggregates_into_forwarded(self):
        s = ResponsesStreamStripper()
        s.feed(self._added(0, "Bash", call_id=""))
        s.feed({"type": "response.function_call_arguments.delta",
                "output_index": 0, "delta": '{"x"'})
        s.feed({"type": "response.function_call_arguments.delta",
                "output_index": 0, "delta": ':2}'})
        assert s.forwarded_calls[0]["function"]["arguments"] == '{"x":2}'


def test_ensure_call_ids_mints_only_missing():
    calls = [{"id": "", "function": {"name": "a"}},
             {"id": "call_keep", "function": {"name": "b"}},
             {"function": {"name": "c"}}]
    ensure_call_ids(calls)
    assert calls[0]["id"].startswith("call_")
    assert calls[1]["id"] == "call_keep"
    assert calls[2]["id"].startswith("call_")


def test_mint_call_id_shape():
    a, b = mint_call_id(), mint_call_id()
    assert a != b and a.startswith("call_") and len(a) > 10
