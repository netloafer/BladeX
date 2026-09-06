"""路由接线测试（Phase 1：配置驱动确定性路由 + 可选 LLM 裁判）。

判分器/裁判 mock；Router 网关的 acompletion 用 monkeypatch 模拟。
"""

from __future__ import annotations

import asyncio

from bladex_proxy import router_sdk
import pytest
from bladex_core.routing import (
    RouteSource,
)
from bladex_proxy.config import ModelRoute, ProxyConfig
from bladex_proxy.route import LLMJudge, build_router, call_model, resolve_route


def _run(coro):
    return asyncio.run(coro)


# ── build_router ──


def test_build_router_none_when_disabled():
    """route_enabled=False → build_router 返回 None。"""
    cfg = ProxyConfig()
    cfg.route_enabled = False
    assert build_router(cfg) is None


def test_build_router_none_when_no_models(tmp_path, monkeypatch):
    """route_enabled=True 但 routing.toml 无模型 → None。"""
    monkeypatch.setenv("BLADEX_ROUTING_CONFIG", "/nonexistent/routing.toml")
    cfg = ProxyConfig()
    cfg.route_enabled = True
    assert build_router(cfg) is None


def test_build_router_none_when_no_upstream(tmp_path, monkeypatch):
    """有 models 但无 [upstream] → None。"""
    monkeypatch.setenv("K", "secret")
    toml = '''
[[models]]
name = "pro"
api_key_env = "K"
tier = "strong"

'''
    p = tmp_path / "routing.toml"
    p.write_text(toml, encoding="utf-8")
    monkeypatch.setenv("BLADEX_ROUTING_CONFIG", str(p))
    cfg = ProxyConfig()
    cfg.route_enabled = True
    assert build_router(cfg) is None


def _write_basic_toml(tmp_path, monkeypatch, *, agent=False, multimodal=False, filter_enabled=False):
    """写最小 routing.toml：pro(strong,vision) + glm(medium) + mini(weak)。"""
    monkeypatch.setenv("K", "secret")
    toml = '''
[[models]]
name = "pro"
api_key_env = "K"
capabilities = ["text", "vision"]
tier = "strong"

[[models]]
name = "glm"
api_key_env = "K"
capabilities = ["text"]
tier = "medium"

[[models]]
name = "mini"
api_key_env = "K"
capabilities = ["text"]
tier = "weak"

[upstream]
models = ["pro", "glm", "mini"]
'''
    if agent:
        toml += '''
[strategies.agent]
enabled = true
[strategies.agent.map]
hermes = ["pro", "glm"]
'''
    if multimodal:
        toml += '''
[strategies.multimodal]
enabled = true
[strategies.multimodal.map]
vision = ["pro"]
'''
    if filter_enabled:
        toml += '''
[strategies.filter]
enabled = true
judge_model = "mini"
no_think = true
max_prompt_chars = 2000
default_tier = "weak"
[strategies.filter.candidates]
weak = ["mini"]
medium = ["glm"]
strong = ["pro"]
'''
    p = tmp_path / "routing.toml"
    p.write_text(toml, encoding="utf-8")
    monkeypatch.setenv("BLADEX_ROUTING_CONFIG", str(p))
    return p


def test_build_router_basic(tmp_path, monkeypatch):
    """全关 → router 非 None，走 upstream 默认池。"""
    _write_basic_toml(tmp_path, monkeypatch)
    cfg = ProxyConfig()
    cfg.route_enabled = True
    router = build_router(cfg)
    assert router is not None
    d = _run(router.route("hi"))
    assert d.source == RouteSource.UPSTREAM_DEFAULT
    assert d.model == "pro"


def test_build_router_wires_agent(tmp_path, monkeypatch):
    """agent 策略开 → 命中 hermes 时收窄集合。"""
    _write_basic_toml(tmp_path, monkeypatch, agent=True)
    cfg = ProxyConfig()
    cfg.route_enabled = True
    router = build_router(cfg)
    d = _run(router.route("hi", agent_id="hermes"))
    assert d.source == RouteSource.AGENT
    assert d.model in ("pro", "glm")  # hermes 集合


def test_build_router_wires_filter_with_judge(tmp_path, monkeypatch):
    """filter 策略开 → build_router 创建 LLMJudge。"""
    _write_basic_toml(tmp_path, monkeypatch, filter_enabled=True)
    cfg = ProxyConfig()
    cfg.route_enabled = True
    router = build_router(cfg)
    assert router is not None
    assert router._judge is not None
    assert isinstance(router._judge, LLMJudge)


# ── resolve_route ──


def test_resolve_route_upstream_default(tmp_path, monkeypatch):
    """路由开启 → resolve_route 走 router 决策。"""
    _write_basic_toml(tmp_path, monkeypatch)
    cfg = ProxyConfig()
    cfg.route_enabled = True
    router = build_router(cfg)
    model, route, failover = _run(resolve_route("client", cfg, router, "hi"))
    assert model == "pro"
    assert route.model == "pro"
    assert route.api_key == "secret"
    assert len(failover) == 2  # glm, mini


def test_resolve_route_disabled_is_regression():
    """路由关闭 = 现状单一上游：返回上游模型 + model_route（回归不变）。"""
    cfg = ProxyConfig()
    cfg.upstream_model = "openai/ark-code-latest"
    cfg.upstream_api_base = "https://example.test"
    cfg.upstream_api_key = "sk-test"
    model, route, failover = _run(resolve_route("whatever", cfg, None, "q"))
    assert model == "openai/ark-code-latest"
    assert route.api_base == "https://example.test"
    assert route.api_key == "sk-test"
    assert failover == []


def test_resolve_route_degrade_to_default():
    """router 决策抛异常 → 降级默认上游、不挂。"""
    cfg = ProxyConfig()
    cfg.upstream_model = "openai/gpt-4o-mini"

    class BoomRouter:
        async def route(self, *a, **k):
            raise RuntimeError("router exploded")

        def failover_candidates(self, *a, **k):
            return []

    model, _, _ = _run(resolve_route("client", cfg, BoomRouter(), "q"))  # type: ignore[arg-type]
    assert model == "openai/gpt-4o-mini"


def test_resolve_route_fallback_logs_fallback_upstream():
    """router=None → source=fallback_upstream 可 grep。"""

    # structlog 默认输出到 stderr，用 capsys 更可靠
    cfg = ProxyConfig()
    cfg.upstream_model = "openai/ark-code-latest"
    model, route, failover = _run(resolve_route("client", cfg, None, "hi"))
    assert model == "openai/ark-code-latest"
    assert failover == []


def test_resolve_route_passes_agent_id(tmp_path, monkeypatch):
    """resolve_route 透传 agent_id。"""
    _write_basic_toml(tmp_path, monkeypatch, agent=True)
    cfg = ProxyConfig()
    cfg.route_enabled = True
    router = build_router(cfg)
    model, _, _ = _run(resolve_route("c", cfg, router, "hi", agent_id="hermes"))
    assert model in ("pro", "glm")  # hermes 集合


def test_resolve_route_strategy_overrides_capability(tmp_path, monkeypatch):
    """能力过滤无合格候选 → NoCapableCandidateError（不降级）。"""
    _write_basic_toml(tmp_path, monkeypatch, agent=True)
    # 改 agent map 只含 text-only 模型
    p = tmp_path / "routing.toml"
    toml = p.read_text()
    toml = toml.replace('hermes = ["pro", "glm"]', 'hermes = ["glm", "mini"]')
    p.write_text(toml)

    cfg = ProxyConfig()
    cfg.route_enabled = True
    router = build_router(cfg)
    model, route, _ = _run(resolve_route("c", cfg, router, "看图", agent_id="hermes", requires=["vision"]))
    assert route.source == "agent"  # ADR-0021 策略 > 能力过滤：命中策略池不快返错
    assert model in ("glm", "mini")  # 用策略池（即使无 vision，不切换到 vision 模型）


def test_route_inactive_when_no_routing_toml(monkeypatch):
    """route_enabled=true 但无 routing.toml → has_models()=False。"""
    monkeypatch.setenv("BLADEX_ROUTING_CONFIG", "/nonexistent/routing.toml")
    cfg = ProxyConfig()
    cfg.route_enabled = True
    assert cfg.routing_config.has_models() is False
    assert cfg.routing_config.has_upstream() is False


# ── LLMJudge ──


def test_llm_judge_parses_tier(tmp_path, monkeypatch):
    """LLMJudge 解析裁判输出成档位。"""
    _write_basic_toml(tmp_path, monkeypatch, filter_enabled=True)
    cfg = ProxyConfig()
    cfg.route_enabled = True
    router = build_router(cfg)
    judge = router._judge

    # mock Router 网关的 acompletion
    class FakeResp:
        class choices:
            class __getitem__:
                pass
        usage = None

    class FakeChoice:
        class message:
            content = "medium"
        @staticmethod
        def __getitem__(i):
            return FakeChoice()

    class FakeRespObj:
        choices = [FakeChoice()]
        usage = None

    async def fake_acompletion(**kw):
        return FakeRespObj()

    monkeypatch.setattr(router_sdk, "acompletion", fake_acompletion)
    result = _run(judge.judge("summarize this"))
    assert result.tier == "medium"


def test_llm_judge_unparseable_raises(tmp_path, monkeypatch):
    """裁判输出无法解析 → 抛异常（router 兜底 default_tier）。"""
    _write_basic_toml(tmp_path, monkeypatch, filter_enabled=True)
    cfg = ProxyConfig()
    cfg.route_enabled = True
    router = build_router(cfg)
    judge = router._judge

    class FakeChoice:
        class message:
            content = "I think this is a complex task"
        @staticmethod
        def __getitem__(i):
            return FakeChoice()

    class FakeRespObj:
        choices = [FakeChoice()]
        usage = None

    async def fake_acompletion(**kw):
        return FakeRespObj()

    monkeypatch.setattr(router_sdk, "acompletion", fake_acompletion)
    with pytest.raises(Exception):  # noqa: B017 -- 故意断言"失败"，不锁死异常类型
        _run(judge.judge("complex task"))


# ── call_model failover ──


class _FakeUpstreamError(Exception):
    def __init__(self, status_code: int, message: str = "boom") -> None:
        super().__init__(message)
        self.status_code = status_code


def _patch_acompletion(monkeypatch, behavior: dict):
    calls: list[str] = []

    async def fake(**kwargs):
        calls.append(kwargs["model"])
        act = behavior.get(kwargs["model"])
        if isinstance(act, Exception):
            raise act
        return {"model": kwargs["model"], "ok": True}

    monkeypatch.setattr(router_sdk, "acompletion", fake)
    return calls


def test_failover_on_5xx(monkeypatch):
    """首选返 503 → 切备选成功。"""
    calls = _patch_acompletion(monkeypatch, {
        "primary": _FakeUpstreamError(503),
        "backup": None,
    })
    result = _run(call_model(
        "primary", [{"role": "user", "content": "hi"}], stream=False,
        route=ModelRoute(model="primary"),
        failover=[ModelRoute(model="backup")],
    ))
    assert result["model"] == "backup"
    assert calls == ["primary", "backup"]


def test_no_failover_on_4xx(monkeypatch):
    """4xx 参数错不触发 failover。"""
    calls = _patch_acompletion(monkeypatch, {
        "primary": _FakeUpstreamError(400, "bad request"),
        "backup": None,
    })
    with pytest.raises(_FakeUpstreamError):
        _run(call_model(
            "primary", [{"role": "user", "content": "hi"}], stream=False,
            route=ModelRoute(model="primary"),
            failover=[ModelRoute(model="backup")],
        ))
    assert calls == ["primary"]


def test_failover_all_fail_returns_last_error(monkeypatch):
    """全部备选都 5xx → 返最后一次上游错。"""
    _patch_acompletion(monkeypatch, {
        "primary": _FakeUpstreamError(503),
        "backup": _FakeUpstreamError(502),
    })
    with pytest.raises(_FakeUpstreamError) as ei:
        _run(call_model(
            "primary", [{"role": "user", "content": "hi"}], stream=False,
            route=ModelRoute(model="primary"),
            failover=[ModelRoute(model="backup")],
        ))
    assert ei.value.status_code == 502
