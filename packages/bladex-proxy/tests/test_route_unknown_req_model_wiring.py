"""批 W W1（拍板 (12)，2026-09-22）：`[strategies.request]` 开着 + model 名不在
`[[models]]` ⇒ **报错给客户端、零上游调用**，而不是悄悄换一个模型。

## 本文件与 core 侧那份的分工

`packages/bladex-core/tests/test_static_route_priority.py` 钉的是**函数行为**
（`router.route()` 抛 `UnknownRequestedModelError`）。本文件钉的是**接线**——
MQ-W13 的教训：「函数有测试」≠「接线有守卫」。这里有两道接线会把新异常吃掉：

| 接线 | 断在这里的后果 |
|---|---|
| `route.py` 的 `except Exception`（L~327） | 异常被吞成 `route_decision_failed` → **回落单一上游 = 仍然替换**，且比改之前更难发现（MQ-CA15 同款：宽 except 静默降级） |
| `orchestration._prepare_round` 的 `except` 分档 | 异常冒到端点 → 500 而不是结构化 4xx，措辞指不到配置 |

所以本文件的核心不是"抛没抛"，而是：
**客户端真的收到了那条指向 `routing.toml` 的错误，且那一轮上游被调用 0 次。**

## 三协议都测

`/v1/chat/completions` · `/v1/messages` · `/v1/responses` —— 本仓的老毛病正是
"同一件事只在一个端点做对了"（MQ-A18 / P7）。

## 阴性对照

策略**关**那一格照 dsh 的 live 形态写（09-20 日志 9/9 轮
`route_request_model_ignored_disabled agent=dsh req_model=deepseek-v4-flash`）：
**不抛错、照常调上游**。拍板 (12) 明写这一格不改。
"""
from __future__ import annotations

import os
import tempfile
from unittest.mock import patch

import pytest
from bladex_core.routing import (
    AgentStrategyData,
    FilterStrategyData,
    MemoryAwareRouter,
    ModelCandidate,
    RequestStrategyData,
    StickyStrategyData,
    UnknownRequestedModelError,
)
from bladex_proxy.config import ProxyConfig
from bladex_proxy.server import create_app
from fastapi.testclient import TestClient

UNKNOWN = "no-such-model-xyz"
KNOWN = "openai/known-medium"


def _cfg() -> ProxyConfig:
    """🔴 与生产 `data/` 隔离：Hub 进临时目录，Redis 用**私有 stream**。

    生产默认 stream 是 `bladex:turns` —— 本地 Redis 起着时，端点级用例会往生产
    队列里塞测试轮次、并用同一个消费组去消费生产消息。私有 stream 由根 conftest
    按**用例**分配（`bladex:test:t<n>:<run>`，用例结束删掉）。

    批 GS（2026-09-29）：这里原先写死 `redis_stream="bladex:test:w1"` —— 本文件
    8 个用例、以及往后每一次 run 都共用这一条流，Z1.3 的实例判活每次都看见"上一个
    用例 4 s 内刚动过的消费者"⇒ 每例白等 4 s（本文件 37.5 s），流也从不回收。
    """
    return ProxyConfig(hard_rules=[], upstream_model="openai/test",
                       upstream_api_key="sk-fake",
                       rocksdb_path=f"{tempfile.mkdtemp()}/rocksdb")


def _router(*, request_enabled: bool) -> MemoryAwareRouter:
    models = {
        KNOWN: ModelCandidate(model=KNOWN, tier="medium", capabilities=["text", "code"]),
        "openai/known-weak": ModelCandidate(
            model="openai/known-weak", tier="weak", capabilities=["text", "code"]),
    }
    return MemoryAwareRouter(
        models=models,
        upstream=["openai/known-weak"],
        agent=AgentStrategyData(enabled=False),
        request_strategy=RequestStrategyData(enabled=request_enabled),
        filter_strategy=FilterStrategyData(enabled=False),
        sticky=StickyStrategyData(enabled=False),
    )


# ── 上游替身（同时充当"被调了几次"的计数器）────────────────────────────────

class _Msg:
    content = "ok"
    tool_calls = None

    def model_dump(self):
        return {"role": "assistant", "content": "ok"}


class _Resp:
    def __init__(self):
        self.choices = [type("C", (), {"message": _Msg(), "finish_reason": "stop"})()]
        self.usage = None

    def model_dump(self):
        return {"id": "chatcmpl-x", "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "ok"},
             "finish_reason": "stop"}]}


def _counting_upstream(calls: list[str]):
    async def fake(model, messages, stream, **kw):  # noqa: ANN001
        calls.append(model)
        return _Resp()
    return fake


def _post(client, proto: str, model: str):
    if proto == "chat":
        return client.post("/v1/chat/completions", json={
            "model": model, "stream": False,
            "messages": [{"role": "system", "content": "You are a test agent."},
                         {"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer sk", "X-Agent-ID": "w1-agent",
                     "X-Session-ID": "s-w1"})
    if proto == "messages":
        return client.post("/v1/messages", json={
            "model": model, "stream": False, "max_tokens": 64,
            "system": "You are a test agent.",
            "messages": [{"role": "user", "content": "hi"}]},
            headers={"x-api-key": "sk", "X-Agent-ID": "w1-agent",
                     "X-Session-ID": "s-w1"})
    return client.post("/v1/responses", json={
        "model": model, "stream": False,
        "instructions": "You are a test agent.", "input": "hi"},
        headers={"Authorization": "Bearer sk", "X-Agent-ID": "w1-agent",
                 "X-Session-ID": "s-w1"})


def _run(proto: str, model: str, *, request_enabled: bool):
    """返回 (response, 上游被调用的模型名列表)。"""
    calls: list[str] = []
    # agent 注册缓存也要隔离：不然测试 agent 会被写进真实的
    # `data/agent_registry.json`（`agent_registry.cache_path` 的注释里记着这条；
    # 本卡开发时**实测被写进去过两条 w1-agent**，已清）。用完还原，
    # 免得把隔离面漏给同进程里的其它用例。
    _prev = os.environ.get("BLADEX_AGENT_REGISTRY_CACHE")
    os.environ["BLADEX_AGENT_REGISTRY_CACHE"] = f"{tempfile.mkdtemp()}/agent_registry.json"
    try:
        app = create_app(_cfg())
        with patch("bladex_proxy.route.router_sdk.acompletion",
                   new=_counting_upstream(calls)):
            with TestClient(app) as client:
                app.state.router = _router(request_enabled=request_enabled)
                resp = _post(client, proto, model)
    finally:
        if _prev is None:
            os.environ.pop("BLADEX_AGENT_REGISTRY_CACHE", None)
        else:
            os.environ["BLADEX_AGENT_REGISTRY_CACHE"] = _prev
    return resp, calls


PROTOS = ["chat", "messages", "responses"]


# ── 判据①③：三协议各一条经真端点的用例 ────────────────────────────────────

@pytest.mark.parametrize("proto", PROTOS)
def test_unknown_req_model_returns_client_error_and_calls_no_upstream(proto):
    """判据①+③：明确错误到客户端 + **那一轮零上游调用**（这是本卡的实质）。"""
    resp, calls = _run(proto, UNKNOWN, request_enabled=True)

    assert resp.status_code == 422, resp.text
    body = resp.json()
    assert body["error"]["type"] == "unknown_requested_model"
    msg = body["error"]["message"]
    assert UNKNOWN in msg                       # 点名用户写的那个原名
    assert "routing.toml" in msg                # 指向配置文件
    assert "[[models]]" in msg                  # 指向要加的那一节
    assert "agent_overrides" in msg             # 指向"我不想要报错"的那条出路
    # 🔴 零替换 = 零上游调用。只断"状态码对"不够——替换发生在上游那一跳。
    assert calls == [], f"{proto}: 报错了却仍然调了上游 {calls}"


@pytest.mark.parametrize("proto", PROTOS)
def test_known_req_model_still_forwards(proto):
    """判据⑥（回归，CC 的 live 形态）：名字**在** [[models]] 内 ⇒ 正常转发到它。

    live 读数：claude-code 99/99 轮 `source=requested`
    （发的 `openai/deepseek-v4-flash` 在 [[models]] 内），本改动必须不碰它。
    """
    resp, calls = _run(proto, KNOWN, request_enabled=True)
    assert resp.status_code == 200, resp.text
    assert calls == [KNOWN], f"{proto}: 期望转发到 {KNOWN}，实际 {calls}"


@pytest.mark.parametrize("proto", PROTOS)
def test_strategy_disabled_negative_control(proto):
    """判据②（阴性对照，dsh 的 live 形态）：策略关 ⇒ 不报错，照常走动态路由。

    dsh 每轮都落在这一格（09-20 日志 9/9）。这里连"名字未知"一并覆盖：
    策略关时任意名都只告警不报错——把这一格也改成报错会打死 dsh 全部流量。
    """
    resp, calls = _run(proto, UNKNOWN, request_enabled=False)
    assert resp.status_code == 200, resp.text
    assert calls and calls[0] != UNKNOWN, f"{proto}: 应回落动态路由，实际 {calls}"


# ── 判据④：接线守卫（拆掉 re-raise 那一档 ⇒ 必须红）────────────────────────

class _RaisingRouter:
    """只做一件事：route() 抛 UnknownRequestedModelError。"""

    def __init__(self) -> None:
        self.calls = 0

    async def route(self, query, **kw):  # noqa: ANN001
        self.calls += 1
        raise UnknownRequestedModelError(UNKNOWN, kw.get("agent_id"))


@pytest.mark.asyncio
async def test_resolve_route_reraises_instead_of_degrading_to_default_upstream():
    """🔴 判据④：`route.py` 的 `except Exception` 不许吞掉这个异常。

    这条就是那一档 re-raise 的守卫：把
    `except (NoCapableCandidateError, UnknownRequestedModelError): raise`
    改回只留 `NoCapableCandidateError`，宽 except 会接住新异常、
    `route_decision_failed` + 回落 `config.upstream_model` —— 于是**仍然替换**，
    而且静默。届时本条必红（拿到的是一个 tuple 而不是异常）。
    """
    from bladex_proxy.route import resolve_route

    router = _RaisingRouter()
    with pytest.raises(UnknownRequestedModelError):
        await resolve_route(UNKNOWN, _cfg(), router, "hi", agent_id="w1-agent")
    assert router.calls == 1


def test_error_response_has_single_construction_point():
    """判据（陷阱 2）：两种成因共用**一个** JSONResponse 构造点，只有措辞与 type 不同。

    MQ-A18 / P7：同一件事不许两处各做一次；MQ-W16：两种成因不许合并成一句话。
    """
    from bladex_proxy.server import orchestration

    src = orchestration.__file__
    with open(src, encoding="utf-8") as f:
        text = f.read()
    assert text.count("JSONResponse(") == 1, (
        "orchestration 里出现了第二个 JSONResponse 构造点——"
        "路由快返错误必须只有 _route_error_response 一个出口")

    cap = orchestration._capability_error_response("a", ["vision"])
    unk = orchestration._unknown_req_model_error_response("a", UNKNOWN)
    assert cap.status_code == unk.status_code == 422
    import json
    cap_msg = json.loads(cap.body)["error"]["message"]
    unk_msg = json.loads(unk.body)["error"]["message"]
    # 措辞必须分开：能力不足那句不许出现在"名字未知"的回复里（指错方向 = 没报）
    assert "能力" in cap_msg and "能力" not in unk_msg
    assert "[[models]]" in unk_msg and "[[models]]" not in cap_msg
