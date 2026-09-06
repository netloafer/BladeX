"""V-P4 验收剧本：拼接-恢复往返 / 锚点消失即弃 / 入站剥离逐字节 / 重发去重 / 老化。"""

from __future__ import annotations

from bladex_proxy.splice import (
    NARRATE_MARK,
    RecentRequests,
    SpliceLedger,
    SpliceRecord,
    anchor_key,
    request_fingerprint,
    splice_into_messages,
    strip_inbound_echoes,
)


def _tc(name, cid, args="{}"):
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": args}}


def _mixed_scenario():
    """混合调用一轮后的入站请求形态（agent 侧历史 = 剥离后的消息 + 自家工具结果）。"""
    stripped_assistant = {"role": "assistant", "content": "",
                          "tool_calls": [_tc("web_search", "a1")]}
    rec = SpliceRecord(
        session_prefix="u/agent/s1/",
        anchor=anchor_key(stripped_assistant),
        calls=[_tc("bladex_memory_search", "b1", '{"query":"x"}')],
        results=[{"role": "tool", "tool_call_id": "b1", "content": "3 hits"}],
    )
    inbound = [
        {"role": "user", "content": "查一下"},
        stripped_assistant,
        {"role": "tool", "tool_call_id": "a1", "content": "search result"},
        {"role": "user", "content": "继续"},
    ]
    return inbound, rec


class TestSplice:
    def test_restore_roundtrip(self):
        inbound, rec = _mixed_scenario()
        out, applied, dropped = splice_into_messages(inbound, [rec])
        assert (applied, dropped) == (1, 0)
        asst = out[1]
        assert [tc["id"] for tc in asst["tool_calls"]] == ["a1", "b1"]
        # 我们的结果插在 agent 工具结果之后、下一条 user 之前
        assert out[2]["tool_call_id"] == "a1"
        assert out[3]["tool_call_id"] == "b1"
        assert out[4]["role"] == "user"
        assert inbound[1].get("tool_calls") == [_tc("web_search", "a1")]  # 入参不改

    def test_anchor_survives_trailing_whitespace_rstrip(self):
        """2026-08-25 真实流量回归：流式捕获尾带 \\n\\n、hermes 回传 rstrip
        ——锚必须仍然命中（首轮 live 实测 anchors_dropped=1 的病例）。"""
        stripped = {"role": "assistant",
                    "content": "小黑帮你翻翻之前的记录～ (•̀·̀•̀)\n\n",
                    "tool_calls": [_tc("session_search", "Tyr838")]}
        rec = SpliceRecord(session_prefix="u/a/s/", anchor=anchor_key(stripped),
                           calls=[_tc("bladex_memory_search", "93pP")],
                           results=[{"role": "tool", "tool_call_id": "93pP",
                                     "content": "Memory hits"}])
        inbound = [
            {"role": "user", "content": "问题"},
            {"role": "assistant",
             "content": "小黑帮你翻翻之前的记录～ (•̀·̀•̀)",   # rstrip 后的形态
             "tool_calls": [_tc("session_search", "Tyr838")]},
            {"role": "tool", "tool_call_id": "Tyr838", "content": "24KB result"},
        ]
        out, applied, dropped = splice_into_messages(inbound, [rec])
        assert (applied, dropped) == (1, 0)
        assert out[-1]["tool_call_id"] == "93pP"

    def test_anchor_missing_dropped_silently(self):
        inbound, rec = _mixed_scenario()
        # agent 压缩改写了 assistant 正文 → 锚不匹配
        inbound[1] = {"role": "assistant", "content": "[compacted]"}
        out, applied, dropped = splice_into_messages(inbound, [rec])
        assert (applied, dropped) == (0, 1)
        assert out[1] == {"role": "assistant", "content": "[compacted]"}

    def test_idempotent_on_resend(self):
        inbound, rec = _mixed_scenario()
        once, _, _ = splice_into_messages(inbound, [rec])
        twice, applied, _ = splice_into_messages(once, [rec])
        assert twice == once or applied == 0  # 已含 b1 → 不重复拼

    def test_ledger_projection_restore(self):
        _, rec = _mixed_scenario()
        led = SpliceLedger()
        led.restore([rec])
        assert led.records("u/agent/s1/")[0].anchor == rec.anchor

    def test_aging(self):
        _, rec = _mixed_scenario()
        led = SpliceLedger(max_age_turns=2)
        led.add(rec)
        assert led.tick_and_prune("u/agent/s1/") == 0
        assert led.tick_and_prune("u/agent/s1/") == 0
        assert led.tick_and_prune("u/agent/s1/") == 1
        assert led.records("u/agent/s1/") == []

    def test_no_aging_by_default(self):
        _, rec = _mixed_scenario()
        led = SpliceLedger()
        led.add(rec)
        for _ in range(10):
            led.tick_and_prune("u/agent/s1/")
        assert len(led.records("u/agent/s1/")) == 1


class TestInboundStrip:
    def test_memory_echo_stripped_from_user_str(self):
        msgs = [{"role": "user",
                 "content": "<bladex-memory>old inject</bladex-memory>真正的问题"}]
        out, n = strip_inbound_echoes(msgs)
        assert n == 1
        assert out[0]["content"] == "真正的问题"

    def test_memory_echo_stripped_from_user_list(self):
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": "<bladex-memory>x</bladex-memory>"},
            {"type": "text", "text": "问题本体"}]}]
        out, n = strip_inbound_echoes(msgs)
        assert n == 1
        assert [b["text"] for b in out[0]["content"]] == ["问题本体"]

    def test_memory_echo_stripped_from_tool_role(self):
        """2026-08-25 live 实测：hermes 把含注入的历史塞进 tool 结果回传
        （`marker_in_non_system_message role=tool ×4`）——tool 角色也必须剥。"""
        msgs = [{"role": "tool", "tool_call_id": "t1",
                 "content": "<bladex-memory>旧注入</bladex-memory>真实工具结果"}]
        out, n = strip_inbound_echoes(msgs)
        assert n == 1 and out[0]["content"] == "真实工具结果"
        assert out[0]["tool_call_id"] == "t1"      # 其余字段不动

    def test_ledger_block_echo_stripped_from_user(self):
        msgs = [{"role": "user",
                 "content": "<bladex-ledger>Active task ledger…</bladex-ledger>继续"}]
        out, n = strip_inbound_echoes(msgs)
        assert n == 1 and out[0]["content"] == "继续"

    def test_narrate_reasoning_field_stripped(self):
        msgs = [{"role": "assistant", "content": "答案",
                 "reasoning_content": f"{NARRATE_MARK} searching… step 1"}]
        out, n = strip_inbound_echoes(msgs)
        assert n == 1 and "reasoning_content" not in out[0]
        assert out[0]["content"] == "答案"

    def test_narrate_thinking_block_stripped(self):
        msgs = [{"role": "assistant", "content": [
            {"type": "thinking", "thinking": f"{NARRATE_MARK} step 1"},
            {"type": "text", "text": "答案"}]}]
        out, n = strip_inbound_echoes(msgs)
        assert n == 1
        assert out[0]["content"] == [{"type": "text", "text": "答案"}]

    def test_agent_own_content_untouched(self):
        # 红线：只剥带标记的块——agent 自己的 thinking/正文一个字不碰
        msgs = [{"role": "assistant", "content": [
            {"type": "thinking", "thinking": "模型自己的思考"},
            {"type": "text", "text": "答案"}]},
                {"role": "user", "content": "提到 bladex-memory 这个词但没有标记块"}]
        out, n = strip_inbound_echoes(msgs)
        assert n == 0 and out == msgs


class TestResendDedup:
    def test_same_body_detected(self):
        msgs = [{"role": "user", "content": "hi"}]
        rr = RecentRequests()
        fp = request_fingerprint(msgs)
        assert rr.seen(fp) is False
        assert rr.seen(fp) is True     # codex 重发形态

    def test_window_expiry(self):
        rr = RecentRequests(window_s=1.0)
        assert rr.seen("f1", now=0.0) is False
        assert rr.seen("f1", now=2.5) is False  # 窗口外重置

    def test_capacity_bounded(self):
        rr = RecentRequests(capacity=2, window_s=999)
        rr.seen("a", now=1)
        rr.seen("b", now=2)
        rr.seen("c", now=3)   # 驱逐 a
        assert rr.seen("a", now=4) is False
