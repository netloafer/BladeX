"""V-P3 验收剧本：多轮循环 / 墙钟降级 / 轮数降级 / 播报 / 入账 / 可转发保证。"""

from __future__ import annotations

import asyncio

from bladex_proxy.innerloop import run_inner_loop, synthesize_results_message
from bladex_proxy.splice import NARRATE_MARK


def _tc(name, cid, args="{}"):
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": args}}


def _run(**kw):
    return asyncio.run(run_inner_loop(**kw))


async def _dispatch(name, args):
    return f"result-of-{name}"


class TestLoop:
    def test_two_round_loop_then_text(self):
        replies = iter([
            {"role": "assistant", "content": None,
             "tool_calls": [_tc("bladex_ledger_read", "b2")], "usage": {"total_tokens": 5}},
            {"role": "assistant", "content": "最终回答", "usage": {"total_tokens": 7}},
        ])

        async def llm(messages):
            return next(replies)

        res = _run(messages=[{"role": "user", "content": "q"}],
                   initial_calls=[_tc("bladex_memory_search", "b1")],
                   call_llm=llm, dispatch=_dispatch)
        assert res.final_message["content"] == "最终回答"
        assert not res.degraded
        assert [r.tool_names for r in res.rounds] == \
            [["bladex_memory_search"], ["bladex_ledger_read"]]
        assert res.rounds[0].usage == {"total_tokens": 5}
        # transcript：assistant+tool 成对、供入 Hub 重放
        roles = [m["role"] for m in res.transcript]
        assert roles == ["assistant", "tool", "assistant", "tool"]

    def test_mixed_reply_returned_to_caller(self):
        async def llm(messages):
            return {"role": "assistant", "content": "",
                    "tool_calls": [_tc("web_search", "a1"),
                                   _tc("bladex_ledger_read", "b2")]}

        res = _run(messages=[], initial_calls=[_tc("bladex_memory_search", "b1")],
                   call_llm=llm, dispatch=_dispatch)
        assert not res.degraded
        assert res.final_message["tool_calls"][0]["id"] == "a1"  # mixed 原样交调用方

    def test_budget_degrade_is_forwardable(self):
        t = {"v": 0.0}

        def clock():
            t["v"] += 100.0     # 每次看钟走 100s → 第一轮后超 120s 预算
            return t["v"]

        async def llm(messages):
            raise AssertionError("预算耗尽后不许再发 LLM 往返")

        res = _run(messages=[], initial_calls=[_tc("bladex_memory_search", "b1")],
                   call_llm=llm, dispatch=_dispatch, budget_s=120.0, clock=clock)
        assert res.degraded and res.degrade_reason == "budget"
        # 降级消息必须可转发：有正文、无悬空 tool_calls
        assert res.final_message["content"].strip()
        assert "tool_calls" not in res.final_message
        assert "result-of-bladex_memory_search" in res.final_message["content"]

    def test_max_rounds_degrade(self):
        async def llm(messages):
            return {"role": "assistant", "content": None,
                    "tool_calls": [_tc("bladex_ledger_read", "bx")]}

        res = _run(messages=[], initial_calls=[_tc("bladex_memory_search", "b1")],
                   call_llm=llm, dispatch=_dispatch, budget_s=9999, max_rounds=3)
        assert res.degraded and res.degrade_reason == "max_rounds"
        assert len(res.rounds) == 3

    def test_progress_narration_marked(self):
        seen: list[str] = []

        async def progress(text):
            seen.append(text)

        async def llm(messages):
            return {"role": "assistant", "content": "ok"}

        _run(messages=[], initial_calls=[_tc("bladex_memory_search", "b1")],
             call_llm=llm, dispatch=_dispatch, on_progress=progress)
        assert seen and all(s.startswith(NARRATE_MARK) for s in seen)

    def test_input_messages_not_mutated(self):
        msgs = [{"role": "user", "content": "q"}]

        async def llm(messages):
            return {"role": "assistant", "content": "ok"}

        _run(messages=msgs, initial_calls=[_tc("bladex_memory_search", "b1")],
             call_llm=llm, dispatch=_dispatch)
        assert msgs == [{"role": "user", "content": "q"}]


class TestSynthesize:
    def test_long_results_truncated(self):
        msg = synthesize_results_message([("bladex_memory_search", "x" * 2000)])
        assert len(msg["content"]) < 700
        assert msg["role"] == "assistant"
