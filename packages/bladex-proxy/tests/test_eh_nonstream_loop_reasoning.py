"""批 EH · MQ-P45：非流式内循环最终回复的思考前缀不丢（原则 13 无损）。

`orchestration._intercept_non_stream` 的 `_loop_llm` 此前没调 `strip_response_inline_reasoning`：
内联 `</think>` 前缀既不进 reasoning、也不进 Hub。修法 = 走与 responses 流式内循环同一个单一实现点，
并让 aux 轮（`loopledger.build_inner_loop_turn`）把回复里的 reasoning 带进 `ResponseMeta.reasoning_text`。

顺带回答出处的未决问题「流式内循环前缀是否进 Hub 的 aux 轮」：见 `test_aux_turn_carries_reasoning_*`。
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from bladex_proxy import server as srv
from bladex_proxy.innerloop import InnerLoopResult, LoopRound
from bladex_proxy.loopledger import InnerLoopLedger, build_inner_loop_turn
from bladex_proxy.models import Identity

PREFIX = "先想一下该怎么答"
RAW = f"{PREFIX}</think>最终答案"


def _resp(content: str):
    msg = SimpleNamespace(role="assistant", content=content, tool_calls=None,
                          reasoning_content=None)
    msg.model_dump = lambda: {"role": "assistant", "content": msg.content,
                              "reasoning_content": msg.reasoning_content,
                              "tool_calls": None}
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")],
                           usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1,
                                                 total_tokens=2))


class _Agency:
    interception_on = True

    def __init__(self) -> None:
        self.reply: dict | None = None

    async def process_message(self, message, *, call_llm, **_kw):
        self.reply = await call_llm([{"role": "user", "content": "x"}])
        return {"role": "assistant", "content": self.reply.get("content")}, [], "pure_bladex"


def _run(monkeypatch, agency):
    async def fake_call_model(*_a, **_kw):
        return _resp(RAW)
    monkeypatch.setattr("bladex_proxy.server.orchestration.call_model", fake_call_model)
    first = _resp(None)
    req = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(agency=agency)))
    asyncio.run(srv._intercept_non_stream(
        req, first, model="m", route=None, failover=[],
        messages=[{"role": "user", "content": "x"}],
        identity=Identity(user_id="u", agent_id="codex", session_id="s"),
        allowed_exposure="local", extra_kwargs={}, health=None, on_used=None))


def test_loop_reply_body_clean_and_prefix_in_reasoning(monkeypatch) -> None:
    ag = _Agency()
    _run(monkeypatch, ag)
    assert ag.reply is not None
    assert "</think>" not in ag.reply["content"]
    assert ag.reply["content"] == "最终答案"
    assert PREFIX in ag.reply["reasoning_content"]


def test_prefix_reaches_hub_aux_turn(monkeypatch) -> None:
    ag = _Agency()
    _run(monkeypatch, ag)
    ledger = InnerLoopLedger()
    ledger.record("sp", InnerLoopResult(
        final_message=ag.reply or {},
        rounds=[LoopRound(round_index=1, reply=dict(ag.reply or {}))]))
    pending = ledger.drain("sp")
    turn = build_inner_loop_turn(Identity(user_id="u", agent_id="codex", session_id="s"),
                                 "m", pending[0])
    assert turn.response_text == "最终答案"
    assert PREFIX in turn.response_meta.reasoning_text


def test_aux_turn_carries_reasoning_from_any_loop_reply() -> None:
    """流式内循环（responses 的 `_loop_llm` 已 strip）的 reply 同形：reasoning 进 aux 轮。
    此前 `build_inner_loop_turn` 根本不读 reasoning ⇒ 无论哪条路径前缀都进不了 Hub。"""
    ledger = InnerLoopLedger()
    ledger.record("sp", InnerLoopResult(
        final_message={}, rounds=[LoopRound(
            round_index=1, reply={"role": "assistant", "content": "答",
                                  "reasoning_content": PREFIX})]))
    t = build_inner_loop_turn(Identity(user_id="u", agent_id="a", session_id="s"),
                              "m", ledger.drain("sp")[0])
    assert t.response_meta.reasoning_text == PREFIX
    # list 形态也并成串
    t2 = build_inner_loop_turn(
        Identity(user_id="u", agent_id="a", session_id="s"), "m",
        _pending({"content": "答", "reasoning": ["a", "b"]}))
    assert t2.response_meta.reasoning_text == "ab"


def _pending(reply: dict):
    from bladex_proxy.loopledger import PendingRound
    return PendingRound(round_index=1, rounds_total=1, reply=reply)
