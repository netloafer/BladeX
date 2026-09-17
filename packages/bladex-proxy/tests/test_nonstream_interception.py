"""V-P5 · 非流式拦截 + none 分流透传（ADR-0032 §3.2）。

## 缺口来历（2026-08-27 V-P5 十八形态对账）

`_handle_anthropic_non_stream` / `_handle_responses_non_stream` 此前是
`call_model → format_* → JSONResponse`，**拦截零接线**。而 `augment_tools`
在流式/非流式分支**之前**执行 ⇒ 这两条路径**照发 `bladex_*` 工具面却不拦截**：
模型一调，调用直接透传给 agent，而 agent 没有这个工具的处理器。

chat 协议早有这段（`_handle_non_stream` 里的 V-P5a 注释）——**同一件事只在一个
端点上做对了**，与 MQ-A18（tools_hash）、MQ-L23（内循环 tool_calls）同型，
本仓一天之内第三次。

⇒ 自查项：新增一个跨协议的能力时，数一数"有几个端点"，逐个确认，
   **不要因为 chat 上跑通了就以为都通了**。

## 三红线之 3：模型不调用时零行为差异

`none` 分流（模型只调自己的工具 / 只回文本）必须**逐字透传**。
既有 `test_bladex_call_passes_through_when_off` 测的是**模块关闭**——
那是另一件事。模块开着、模型没碰我们的工具，才是红线说的那个场景。
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from bladex_proxy import server as srv

BLADEX_CALL = {"id": "call_bx1", "type": "function",
               "function": {"name": "bladex_ledger_read", "arguments": "{}"}}
AGENT_CALL = {"id": "call_shell1", "type": "function",
              "function": {"name": "shell", "arguments": '{"cmd":"ls"}'}}


@pytest.fixture(autouse=True)
def _modules_on(monkeypatch):
    for k in ("BLADEX_MODULE_TOOLFACE", "BLADEX_MODULE_INTERCEPTION",
              "BLADEX_MODULE_LEDGER"):
        monkeypatch.setenv(k, "1")


def _upstream(content: str | None, tool_calls: list[dict] | None,
              finish_reason: str = "tool_calls"):
    """litellm 非流式响应的最小同形替身（formatter 全走 getattr）。"""
    def _tc(d):
        fn = d["function"]
        return SimpleNamespace(id=d["id"], type="function",
                               function=SimpleNamespace(name=fn["name"],
                                                        arguments=fn["arguments"]))
    msg = SimpleNamespace(role="assistant", content=content,
                          tool_calls=[_tc(t) for t in tool_calls or []] or None)
    msg.model_dump = lambda: {                      # noqa: E731 —— 替身够用即可
        "role": "assistant", "content": content,
        "tool_calls": list(tool_calls or [])}
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason=finish_reason)],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2))


class _Agency:
    """只实现 `_intercept_non_stream` 用到的两件事。"""

    def __init__(self, mode: str, processed: dict):
        self._mode, self._processed = mode, processed
        self.calls = 0
        self.interception_on = True

    async def process_message(self, message, **_kw):
        self.calls += 1
        return self._processed, [], self._mode


class _OffAgency:
    interception_on = False

    async def process_message(self, *_a, **_kw):        # pragma: no cover
        raise AssertionError("拦截关闭时不该走处置")


def _req(agency):
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(agency=agency)))


def _identity():
    from bladex_proxy.models import Identity
    return Identity(user_id="u", agent_id="codex", session_id="s")


def _run(agency, response):
    return asyncio.run(srv._intercept_non_stream(
        _req(agency), response, model="m", route=None, failover=[],
        messages=[{"role": "user", "content": "干活"}], identity=_identity(),
        allowed_exposure="local", extra_kwargs={}, health=None, on_used=None))


class TestNoneIsBytewiseUntouched:
    """🔴 三红线之 3：模型没碰我们的工具 ⇒ 响应对象**原样返回**。

    判据是 `is` —— 返回同一个对象，连替身都不套。
    只要走了 `_shim_processed_response`，即便字段看着一样，
    也已经丢了 formatter 可能读到的其它属性。
    """

    def test_agent_only_call_returns_same_object(self):
        resp = _upstream(None, [AGENT_CALL])
        ag = _Agency("none", {"role": "assistant", "tool_calls": [AGENT_CALL]})
        assert _run(ag, resp) is resp

    def test_plain_text_returns_same_object(self):
        resp = _upstream("就是一段文本", None, finish_reason="stop")
        ag = _Agency("none", {"role": "assistant", "content": "就是一段文本"})
        assert _run(ag, resp) is resp

    def test_interception_off_never_calls_process(self):
        """模块关 = 零行为差异，连 `process_message` 都不该被调到。"""
        resp = _upstream(None, [BLADEX_CALL])
        assert _run(_OffAgency(), resp) is resp


class TestMixedAndPureAreProcessed:
    def test_mixed_strips_bladex_call(self):
        resp = _upstream("先查一下", [BLADEX_CALL, AGENT_CALL])
        ag = _Agency("mixed", {"role": "assistant", "content": "先查一下",
                               "tool_calls": [AGENT_CALL]})
        out = _run(ag, resp)
        assert out is not resp, "mixed 必须处置"
        names = [tc.function.name for tc in out.choices[0].message.tool_calls]
        assert names == ["shell"]

    def test_pure_rewrites_finish_reason_to_stop(self):
        """调用全被拦下且无剩余调用 ⇒ `stop`。

        留着 `tool_calls` 会让 agent 收到「说要调工具却一个都没有」并重试
        （chat 侧 2026-08-25 实测过 4 轮）。
        """
        resp = _upstream(None, [BLADEX_CALL])
        ag = _Agency("pure_bladex", {"role": "assistant", "content": "查到了"})
        out = _run(ag, resp)
        assert out.choices[0].finish_reason == "stop"
        assert not out.choices[0].message.tool_calls

    def test_pure_with_leftover_calls_keeps_tool_calls(self):
        resp = _upstream(None, [BLADEX_CALL])
        ag = _Agency("pure_bladex", {"role": "assistant", "tool_calls": [AGENT_CALL]})
        assert _run(ag, resp).choices[0].finish_reason == "tool_calls"

    def test_bad_response_shape_is_survived(self):
        """上游形状异常时降级返回原对象，不炸在热路径上。"""
        broken = SimpleNamespace(choices=[])
        assert _run(_Agency("mixed", {}), broken) is broken


class TestShimFeedsBothFormatters:
    """替身必须被两个 formatter 认得——它们靠 getattr 读属性，不是 dict。"""

    @pytest.mark.parametrize("fmt_name", ["anthropic", "responses"])
    def test_formatter_reads_processed_tool_calls(self, fmt_name):
        if fmt_name == "anthropic":
            from bladex_proxy.anthropic import format_anthropic_response as fmt
        else:
            from bladex_proxy.responses import format_responses_response as fmt
        shim = srv._shim_processed_response(
            _upstream(None, [BLADEX_CALL, AGENT_CALL]),
            {"role": "assistant", "content": "先查一下", "tool_calls": [AGENT_CALL]})
        out = fmt(shim, "m")
        blob = json.dumps(out, ensure_ascii=False)
        assert "shell" in blob, f"{fmt_name}: 处置后的 agent 调用没进最终响应"
        assert "bladex_" not in blob, f"{fmt_name}: bladex 调用泄漏给了 agent"
        assert "先查一下" in blob

    def test_shim_does_not_mutate_the_original(self):
        """🔴 不许就地改 litellm 的响应对象。

        `_build_response_meta_from_response` 与入库口径都还要读原对象——
        改了它，Hub 里存的就不再是"完整真相"。
        """
        resp = _upstream("原文", [BLADEX_CALL, AGENT_CALL])
        srv._shim_processed_response(resp, {"role": "assistant", "content": "改过的",
                                            "tool_calls": [AGENT_CALL]})
        assert resp.choices[0].message.content == "原文"
        assert len(resp.choices[0].message.tool_calls) == 2


class TestStreamMarkerIsLogged:
    """V-P5③：入站流式/非流式必须可 grep。

    2026-08-27 实测：日志里**没有任何**区分入站形态的信号，
    于是"非流式没接拦截"这个缺口发现后，想量它有没有被真实踩到都做不到
    （`route_calling stream=False` 是内循环自己的上游调用，当天差点据此误判）。
    """

    @pytest.mark.parametrize("stream", [True, False])
    def test_toolface_decision_carries_stream(self, stream, monkeypatch):
        import structlog
        from bladex_proxy.agency import AgencyRuntime
        seen: list = []
        monkeypatch.setattr(
            structlog.get_logger().__class__, "info",
            lambda _s, ev, **kw: seen.append((ev, kw)), raising=False)
        ag = AgencyRuntime()
        ag.augment_tools([{"type": "function", "function": {"name": "shell"}}],
                         agent_id="codex", auxiliary=False, tier="strong",
                         stream=stream)
        rows = [kw for ev, kw in seen if ev == "agency_toolface_decision"]
        assert rows, "toolface 决策没打日志"
        assert rows[-1].get("stream") is stream, (
            f"日志里没有入站形态标记：{rows[-1]}")
