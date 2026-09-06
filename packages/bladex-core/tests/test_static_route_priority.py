"""刚性原则 10 验收：配置文件里的静态路由策略压过一切动态推断。

2026-07-28 回归事故：Claude Code 配了 GLM-5.2，实测 25 轮里
`source=auxiliary` 82 次、`source=requested` **1 次**，全部落到
`doubao-seed-2.0-lite`——因为 `auxiliary` 降档排在 `req_model` 与选池之前，
一轮被判 aux 就把用户配置整个旁路。

本文件把「静态压过动态」钉死在四个动态推断上：auxiliary / scale floor / sticky /
（LLM 裁判由静态选池命中即不进入，隐含覆盖）。
"""

from __future__ import annotations

import pytest

from bladex_core.routing import (
    AgentStrategyData,
    _DYNAMIC_SOURCES,
    _STATIC_SOURCES,
    MemoryAwareRouter,
    ModelCandidate,
    RouteSource,
)


def _models() -> dict[str, ModelCandidate]:
    return {
        "openai/glm-5.2": ModelCandidate(
            model="openai/glm-5.2", tier="medium", capabilities=["text", "code"]),
        "openai/doubao-lite": ModelCandidate(
            model="openai/doubao-lite", tier="weak", capabilities=["text", "code"]),
        "anthropic/doubao-lite": ModelCandidate(
            model="anthropic/doubao-lite", tier="weak", capabilities=["text", "code"]),
        "openai/strong-x": ModelCandidate(
            model="openai/strong-x", tier="strong", capabilities=["text", "code"]),
    }


def _router(**kw) -> MemoryAwareRouter:
    """构造 router：默认开 agent 策略把 claude-code 钉在 glm-5.2。"""
    from bladex_core.routing import (
        AgentStrategyData,
        FilterStrategyData,
        RequestStrategyData,
        ScaleStrategyData,
        StickyStrategyData,
    )

    defaults = dict(
        models=_models(),
        upstream=["openai/doubao-lite"],
        agent=AgentStrategyData(enabled=True, map={"claude-code": ["openai/glm-5.2"]}),
        request_strategy=RequestStrategyData(enabled=True),
        filter_strategy=FilterStrategyData(
            enabled=True,
            candidates={
                "weak": ["openai/doubao-lite", "anthropic/doubao-lite"],
                "medium": ["openai/glm-5.2"],
                "strong": ["openai/strong-x"],
            },
            default_tier="weak",
        ),
        scale=ScaleStrategyData(enabled=False),
        sticky=StickyStrategyData(enabled=False),
        aux_tier="weak",
    )
    defaults.update(kw)
    return MemoryAwareRouter(**defaults)


# ── 划分表本身 ──


def test_static_and_dynamic_sources_are_disjoint():
    assert not (_STATIC_SOURCES & _DYNAMIC_SOURCES)


def test_static_sources_are_the_config_driven_ones():
    assert _STATIC_SOURCES == {
        RouteSource.REQUESTED, RouteSource.TEAM,
        RouteSource.PRINCIPAL, RouteSource.AGENT,
    }


# ── 静态 vs auxiliary（事故本体）──


@pytest.mark.asyncio
async def test_agent_strategy_beats_auxiliary_cheap_tier():
    """🔴 事故形状：配了 agent 策略的请求被判 aux，不得掉到便宜档。"""
    r = _router()
    d = await r.route("任意问题", agent_id="claude-code", auxiliary=True)
    assert d.source is RouteSource.AGENT
    assert d.model == "openai/glm-5.2"


@pytest.mark.asyncio
async def test_req_model_beats_auxiliary_cheap_tier():
    """显式指定模型 + 被判 aux → 仍用指定的模型。"""
    r = _router(agent=AgentStrategyData(enabled=False))
    d = await r.route("任意问题", agent_id="whatever",
                      auxiliary=True, req_model="openai/strong-x")
    assert d.source is RouteSource.REQUESTED
    assert d.model == "openai/strong-x"


@pytest.mark.asyncio
async def test_auxiliary_still_works_without_static_config():
    """没有静态策略命中时，auxiliary 降档照常生效（T9 收益不丢）。"""
    r = _router(agent=AgentStrategyData(enabled=False))
    d = await r.route("任意问题", agent_id="unmapped-agent", auxiliary=True)
    assert d.source is RouteSource.AUXILIARY
    assert d.tier == "weak"


# ── 静态 vs 规模层 floor ──


@pytest.mark.asyncio
async def test_agent_strategy_beats_scale_floor():
    """超大上下文本会抬到 strong，但 agent 策略已钉死 glm-5.2 → 不得改写。"""
    from bladex_core.routing import AgentStrategyData, ScaleStrategyData

    r = _router(scale=ScaleStrategyData(
        enabled=True, thresholds={"strong": 1000}))
    d = await r.route("任意问题", agent_id="claude-code", context_chars=999_999)
    assert d.source is RouteSource.AGENT
    assert d.model == "openai/glm-5.2"


# ── 静态 vs 会话粘性 ──


@pytest.mark.asyncio
async def test_agent_strategy_beats_sticky():
    """上一轮粘住了别的模型，也不得改写 agent 策略的选择。"""
    from bladex_core.routing import AgentStrategyData, StickyStrategyData

    r = _router(sticky=StickyStrategyData(enabled=True))
    r.update_sticky("sess-1", "openai/strong-x")
    d = await r.route("任意问题", agent_id="claude-code", session_id="sess-1")
    assert d.source is RouteSource.AGENT
    assert d.model == "openai/glm-5.2"


@pytest.mark.asyncio
async def test_sticky_still_works_without_static_config():
    """无静态策略时粘性照常生效（T2 收益不丢）。"""
    from bladex_core.routing import AgentStrategyData, StickyStrategyData

    r = _router(agent=AgentStrategyData(enabled=False), sticky=StickyStrategyData(enabled=True))
    r.update_sticky("sess-1", "openai/strong-x")
    d = await r.route("任意问题", agent_id="unmapped", session_id="sess-1")
    assert d.model == "openai/strong-x"


# ── 能力不满足 = 配置问题，正常报错（2026-07-28 拍板）──


@pytest.mark.asyncio
async def test_static_strategy_wins_even_when_capability_mismatched(caplog):
    """静态策略命中时不做能力过滤——配置钉死了就用它，不匹配就让它正常报错。

    用户拍板（2026-07-28）：「配置文件允许 agent 自定义模型，对应模型不支持的
    模态类型，按正常报错即可，**不是程序问题，是配置问题**。」
    程序不替用户改选择——那才是把配置错误藏起来。

    但报错必须能**指向配置**：`route_static_capability_mismatch` 告警点名
    strategy / agent / model / 缺哪个能力，否则用户只看到上游那句
    "Model only support text input"，根本定位不到 routing.toml。
    """
    import logging

    models = _models()
    models["openai/vision-x"] = ModelCandidate(
        model="openai/vision-x", tier="medium", capabilities=["text", "vision"])
    r = _router(models=models)

    with caplog.at_level(logging.WARNING, logger="bladex_core.routing"):
        d = await r.route("看图", agent_id="claude-code", requires=["vision"])

    # 配置说了算：仍然是 glm-5.2，哪怕它不支持 vision
    assert d.source is RouteSource.AGENT
    assert d.model == "openai/glm-5.2"
    # 但配置问题必须可发现
    assert "route_static_capability_mismatch" in caplog.text
    assert "vision" in caplog.text
    assert "routing.toml" in caplog.text


@pytest.mark.asyncio
async def test_no_warning_when_static_capability_satisfied(caplog):
    """能力匹配时不该有噪音告警。"""
    import logging

    r = _router()
    with caplog.at_level(logging.WARNING, logger="bladex_core.routing"):
        d = await r.route("普通问题", agent_id="claude-code", requires=["code"])
    assert d.model == "openai/glm-5.2"
    assert "route_static_capability_mismatch" not in caplog.text


@pytest.mark.asyncio
async def test_req_model_capability_mismatch_raises_clear_error():
    """显式指定路径已有结构化报错（点名模型与缺失能力）——这就是"正常报错"。"""
    from bladex_core.routing import NoCapableCandidateError

    r = _router(agent=AgentStrategyData(enabled=False))
    with pytest.raises(NoCapableCandidateError, match="lacks capabilities"):
        await r.route("看图", agent_id="whatever",
                      req_model="openai/glm-5.2", requires=["vision"])
