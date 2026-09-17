"""MQ-P9 验收：内循环轮次以 aux 轮入 Hub（ADR-0032 §3.2 硬点 6）。

2026-08-31 标记时的读数：`route_calling` 82 vs `turn_enqueued` 56，差 26 =
十次内循环 `rounds=` 之和逐条吻合 ⇒ 31.7% 上游调用不在 Hub 里。

本文件钉四件事：
1. rounds≥2 的内循环 → 缓冲里恰好对应条数，每条构成的 Turn **字段齐**
   （发起方 = 同 session_prefix / aux 标记 / 轮序 / usage / 耗时 / 工具名）。
2. 重建侧**从消息本身**能重判出 aux 且三分为 dropped（ADR-0018 R2：不信冻结字段）
   ——阴性对照：user 消息里**谈论**标记串不命中（MQ-A19 自指陷阱的形状）。
3. 模块关 ⇒ 零登记、零写入。
4. 生产接线：三个调用点都 `record()`；`_enqueue_turn` 真的 `drain()` 并入队；
   五处 `_loop_llm` 都经 `_loop_reply` 挂 usage（此前 usage 恒空，`tokens` 恒 0）。
"""
from __future__ import annotations

import asyncio
import inspect
import os

import pytest

from bladex_proxy.innerloop import (
    INNER_LOOP_AUX_SOURCE,
    INNER_LOOP_MARKER,
    run_inner_loop,
)
from bladex_proxy.loopledger import (
    InnerLoopLedger,
    build_inner_loop_turn,
    normalize_usage,
)
from bladex_proxy.models import Identity, Turn

_PKG = os.path.dirname(inspect.getsourcefile(run_inner_loop))


from _source_probe import package_source


def _src(name: str) -> str:
    with open(os.path.join(_PKG, name), encoding="utf-8") as f:
        return f.read()


def _tc(name, cid, args="{}"):
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": args}}


async def _dispatch(name, args):
    return f"result-of-{name}"


def _two_round_result():
    replies = iter([
        {"role": "assistant", "content": None,
         "tool_calls": [_tc("bladex_ledger_read", "b2")],
         "usage": {"prompt": 100, "completion": 5, "total": 105}},
        {"role": "assistant", "content": "最终回答",
         "usage": {"prompt_tokens": 120, "completion_tokens": 7, "total_tokens": 127}},
    ])

    async def llm(messages):
        return next(replies)

    clock = {"t": 0.0}

    def tick():
        clock["t"] += 0.25       # 每次取钟 +250ms ⇒ tool_ms / llm_ms 都非零
        return clock["t"]

    return asyncio.run(run_inner_loop(
        messages=[{"role": "user", "content": "q"}],
        initial_calls=[_tc("bladex_memory_search", "b1", '{"query":"x"}')],
        call_llm=llm, dispatch=_dispatch, clock=tick))


_IDENTITY = Identity(user_id="u", agent_id="claude-code", session_id="s1",
                     sensitivity="internal")


class TestRoundsBecomeAuxTurns:
    def test_each_round_is_one_turn_with_fields(self):
        res = _two_round_result()
        assert len(res.rounds) == 2
        led = InnerLoopLedger()
        assert led.record(_IDENTITY.session_prefix(), res) == 2
        pend = led.drain(_IDENTITY.session_prefix())
        assert len(pend) == 2
        assert led.drain(_IDENTITY.session_prefix()) == [], "drain 必须清空"

        turns = [build_inner_loop_turn(_IDENTITY, "m", p, trigger_ts="T0") for p in pend]
        for t in turns:
            assert isinstance(t, Turn)
            # 发起方：同一 session_prefix（agent/session 直接落在 key 上）
            assert t.identity.session_prefix() == _IDENTITY.session_prefix()
            assert t.auxiliary is True and t.identity.auxiliary is True
            assert t.aux_source == INNER_LOOP_AUX_SOURCE
            assert t.identity.aux_source == INNER_LOOP_AUX_SOURCE
            assert t.sensitivity == "internal", "敏感度随触发轮，不能回落 normal"
            assert t.request_messages[0]["role"] == "system"
            assert t.request_messages[0]["content"].startswith(INNER_LOOP_MARKER)
            assert "trigger_ts=T0" in t.request_messages[0]["content"]
            assert t.ms_total > 0
        t1, t2 = turns
        assert (t1.roundtrip, t2.roundtrip) == (1, 2)
        # usage：两种输入形态都归一到 {prompt, completion, total}
        assert t1.response_meta.usage == {"prompt": 100, "completion": 5, "total": 105}
        assert t2.response_meta.usage == {"prompt": 120, "completion": 7, "total": 127}
        # 工具名：每轮调用的 bladex 工具，call + result 成对
        calls1 = [e for e in t1.tool_events if e.direction == "call"]
        assert [e.tool_name for e in calls1] == ["bladex_memory_search"]
        assert calls1[0].arguments == '{"query":"x"}'
        assert [e.result for e in t1.tool_events if e.direction == "result"] == \
            ["result-of-bladex_memory_search"]
        assert [e.tool_name for e in t2.tool_events if e.direction == "call"] == \
            ["bladex_ledger_read"]
        # 回复：第 1 轮是 tool_calls（finish=tool_calls）、第 2 轮是正文
        assert t1.response_meta.finish_reason == "tool_calls"
        assert t2.response_text == "最终回答"
        assert t2.response_meta.finish_reason == "stop"

    def test_usage_stripped_from_outgoing_final_message(self):
        """usage 是入账附件，不是 wire 形态——不许随最终消息外发给 agent。"""
        res = _two_round_result()
        assert "usage" not in res.final_message
        assert res.rounds[1].usage["total_tokens"] == 127, "入账侧仍拿得到"

    def test_turn_roundtrips_through_pydantic(self):
        res = _two_round_result()
        led = InnerLoopLedger()
        led.record("p/", res)
        t = build_inner_loop_turn(_IDENTITY, "m", led.drain("p/")[0])
        back = Turn.model_validate(t.model_dump(mode="json"))
        assert back.aux_source == INNER_LOOP_AUX_SOURCE
        assert back.request_messages[1]["tool_calls"][0]["function"]["name"] == \
            "bladex_memory_search"

    def test_budget_degraded_round_has_empty_reply(self):
        """预算门在 LLM 之前触发的那一轮没有回复：usage 空、正文空，但轮次仍入账。"""
        async def llm(messages):
            raise AssertionError("预算已耗尽，不该再调 LLM")

        res = asyncio.run(run_inner_loop(
            messages=[], initial_calls=[_tc("bladex_memory_search", "b1")],
            call_llm=llm, dispatch=_dispatch, budget_s=0.0))
        assert res.degraded and len(res.rounds) == 1
        led = InnerLoopLedger()
        led.record("p/", res)
        t = build_inner_loop_turn(_IDENTITY, "m", led.drain("p/")[0])
        assert t.response_text == "" and t.response_meta.usage == {}
        assert "degraded=true(budget)" in t.request_messages[0]["content"]
        assert [e.tool_name for e in t.tool_events if e.direction == "call"] == \
            ["bladex_memory_search"]


class TestRebuildRejudgesFromMessages:
    def test_classified_aux_and_dropped(self):
        from bladex_core.task_goal import TURN_DROPPED
        from bladex_proxy.identity import (
            DISTILL_ONLY_AUX_RULES,
            classify_auxiliary,
            classify_turn_disposition,
        )
        res = _two_round_result()
        led = InnerLoopLedger()
        led.record("p/", res)
        t = build_inner_loop_turn(_IDENTITY, "m", led.drain("p/")[0])
        assert classify_auxiliary(t.request_messages) == (True, INNER_LOOP_AUX_SOURCE)
        assert INNER_LOOP_AUX_SOURCE not in DISTILL_ONLY_AUX_RULES, \
            "进了 DISTILL_ONLY 就成 scaffold，assistant 侧会被蒸馏"
        assert classify_turn_disposition(t.request_messages)[0] == TURN_DROPPED

    def test_talking_about_marker_in_user_is_not_aux(self):
        """阴性对照（MQ-A19 自指陷阱）：user 消息里出现标记串 ≠ 内循环轮。"""
        from bladex_proxy.identity import classify_auxiliary
        msgs = [{"role": "system", "content": "You are a coding assistant."},
                {"role": "user", "content": f"文档里提到 {INNER_LOOP_MARKER} 这个标记"}]
        assert classify_auxiliary(msgs) == (False, "")
        # 标记不在首条 / 首条不是 system 也不命中
        msgs2 = [{"role": "user", "content": "q"},
                 {"role": "system", "content": f"{INNER_LOOP_MARKER} round 1/1"}]
        assert classify_auxiliary(msgs2) == (False, "")

    def test_cheap_tier_semantics_unchanged(self):
        """内循环 aux 不进 DISTILL_ONLY ⇒ 按现有语义归"完整 aux"。它永远不会被路由
        （是记录不是请求），这里只钉语义没被顺手改掉。"""
        from bladex_proxy.identity import is_cheap_tier_auxiliary
        assert is_cheap_tier_auxiliary(INNER_LOOP_AUX_SOURCE) is True


class TestModuleOff:
    def test_process_message_records_nothing_when_off(self, monkeypatch):
        monkeypatch.setenv("BLADEX_MODULE_INTERCEPTION", "0")  # 2026-09-02 翻默认后显式关
        from bladex_proxy.agency import AgencyRuntime
        ag = AgencyRuntime()
        assert not ag.interception_on

        async def llm(messages):
            raise AssertionError("模块关时不该调 LLM")

        msg = {"role": "assistant", "content": None,
               "tool_calls": [_tc("bladex_memory_search", "b1")]}
        out = asyncio.run(ag.process_message(
            msg, upstream_messages=[{"role": "user", "content": "q"}],
            session_prefix="u/a/s/", allowed_exposure="public", call_llm=llm,
            session_id="s", agent_id="a"))
        assert out[0] is msg and out[2] == "none"
        assert ag.loop_ledger.pending_sessions() == 0
        assert ag.loop_ledger.drain("u/a/s/") == []

    def test_enqueue_turn_gates_drain_on_interception(self):
        """`_enqueue_turn` 只在 `interception_on` 时 drain（与 splice 同门）。"""
        src = package_source("server")   # F0.1 拆包：server.py → server/ 包，按包拼接读源码
        i = src.find("_ag.loop_ledger.drain(")
        assert i > 0, "找不到 drain 接线"
        gate = src.rfind('getattr(_ag, "interception_on", False)', 0, i)
        assert gate > 0 and i - gate < 600, "drain 必须在 interception_on 门内"


class TestProductionWiring:
    """接线守卫：机制写完接线断，是本仓最高发形态（MQ-P3 / MQ-P9 自己就是）。"""

    def test_all_three_loop_sites_record(self):
        from bladex_proxy import agency as ag_mod
        from bladex_proxy.agency import AgencyRuntime
        pm = inspect.getsource(AgencyRuntime.process_message)
        assert "self.loop_ledger.record(session_prefix, result)" in pm
        cs = inspect.getsource(ag_mod.intercept_chat_stream)
        assert "agency.loop_ledger.record(session_prefix, result)" in cs
        ps = inspect.getsource(ag_mod._intercept_protocol_stream)
        assert "agency.loop_ledger.record(session_prefix, result)" in ps

    def test_enqueue_turn_drains_and_enqueues(self):
        src = package_source("server")   # F0.1 拆包：server.py → server/ 包，按包拼接读源码
        head = src.find("async def _enqueue_turn(")
        tail = src.find("\nasync def ", head + 10)
        body = src[head:tail if tail > 0 else None]
        assert "_ag.loop_ledger.drain(identity.session_prefix())" in body
        assert "build_inner_loop_turn(" in body
        assert "_enqueue_inner_loop_turns(request, pipeline, identity, _loop_turns)" in body
        # 主轮落盘兜底时 aux 轮也落盘，不能只丢 aux
        assert "for _lt in _loop_turns:" in body and "disk_spill.spill(_lt)" in body
        # 对账日志名（MQ-W 第二项）
        assert 'logger.info("inner_loop_turn_enqueued"' in src

    def test_every_loop_llm_attaches_usage(self):
        """五处 `_loop_llm` 都必须经 `_loop_reply`——直接 `message.model_dump()`
        就是 usage 恒空的老形态。"""
        src = package_source("server")   # F0.1 拆包：server.py → server/ 包，按包拼接读源码
        n_defs = src.count("async def _loop_llm(")
        assert n_defs >= 5, f"调用点数变了（{n_defs}），核对本守卫"
        assert src.count("return _loop_reply(r2)") == n_defs
        # 老形态（各处自写 `model_dump() if hasattr else dict(m)`）一处都不许再有——
        # 2026-09-02 收成 `_message_as_dict` 唯一实现点；`_loop_reply` 也经它
        assert src.count('if hasattr(m2, "model_dump") else dict(m2)') == 0
        assert src.count('if hasattr(message, "model_dump")') == 0
        i0 = src.find("def _loop_reply(")
        assert "_message_as_dict(m2)" in src[i0:i0 + 1200]
        # 直接证：_loop_reply 挂 usage
        i = src.find("def _loop_reply(")
        assert '"usage"' in src[i:i + 1200] and "_extract_usage_dict(usage)" in src[i:i + 1200]

    def test_loop_cost_reads_normalized_total(self):
        from bladex_proxy.agency import _loop_cost

        class R:
            def __init__(self, usage):
                self.tool_ms, self.llm_ms, self.tool_names, self.usage = 1.0, 2.0, ["x"], usage

        class Res:
            rounds = [R({"total_tokens": 10}), R({"total": 5}), R({})]

        assert _loop_cost(Res())["tokens"] == 15


class TestNormalizeUsage:
    @pytest.mark.parametrize("raw,exp", [
        ({}, {}),
        (None, {}),
        ({"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
         {"prompt": 1, "completion": 2, "total": 3}),
        ({"prompt": 1, "completion": 2}, {"prompt": 1, "completion": 2, "total": 3}),
        ({"input_tokens": 4, "output_tokens": 6}, {"prompt": 4, "completion": 6, "total": 10}),
        ({"total_tokens": "x"}, {}),
    ])
    def test_shapes(self, raw, exp):
        assert normalize_usage(raw) == exp


class TestLedgerBounds:
    def test_eviction_is_bounded_and_logged(self):
        res = _two_round_result()
        led = InnerLoopLedger(max_sessions=2)
        led.record("a/", res)
        led.record("b/", res)
        led.record("c/", res)
        assert led.pending_sessions() == 2
        assert led.drain("a/") == [], "最旧会话被淘汰"
        assert len(led.drain("c/")) == 2
