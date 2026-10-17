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
    _DYNAMIC_SOURCES,
    _STATIC_SOURCES,
    AgentStrategyData,
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
    from bladex_core.routing import ScaleStrategyData

    r = _router(scale=ScaleStrategyData(
        enabled=True, thresholds={"strong": 1000}))
    d = await r.route("任意问题", agent_id="claude-code", context_chars=999_999)
    assert d.source is RouteSource.AGENT
    assert d.model == "openai/glm-5.2"


# ── 静态 vs 会话粘性 ──


@pytest.mark.asyncio
async def test_agent_strategy_beats_sticky():
    """上一轮粘住了别的模型，也不得改写 agent 策略的选择。"""
    from bladex_core.routing import StickyStrategyData

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


# ── 静态 vs filter 裁判（MQ-R13，2026-09-19 补） ──
#
# 🔴 本文件开头那句"（LLM 裁判由静态选池命中即不进入，隐含覆盖）"是这条红线上的缺口：
# filter 从未被**直接**钉过，且全文件的 router 都构造成 `RequestStrategyData(enabled=True)`，
# 而生产 routing.toml 是 `enabled=false` + 仅 claude-code 覆盖 ⇒ 真正出事的那条组合
# （req_model 在场、策略对该 agent 关、filter 接管）一个用例都没覆盖。
# 现场：R4 段 B 请求体逐轮写死 openai/deepseek-v4-flash，前两轮
# `route_decision ... model=openai/doubao-seed-2.0-lite source=filter`，全程零告警。


@pytest.mark.asyncio
async def test_req_model_beats_filter_judge():
    """判据①：写死一个配置里存在的模型 ⇒ source=requested，filter 裁判整段不进入。"""
    r = _router(agent=AgentStrategyData(enabled=False))
    judged: list[str] = []

    async def _spy(query, session_id=None):  # noqa: ANN001
        judged.append(query)
        return "weak"

    r._judge_tier = _spy
    d = await r.route("任意问题", agent_id="r4-ladder", req_model="openai/strong-x")
    assert d.source is RouteSource.REQUESTED
    assert d.model == "openai/strong-x"
    # 阴性对照的着力点：短路一旦拆掉，裁判就会被调用、source 变 filter。
    assert judged == []


@pytest.mark.asyncio
async def test_req_model_ignored_when_strategy_disabled_is_greppable(caplog):
    """判据③：策略对该 agent 关 ⇒ 照配置回落动态路由，但必须留具名告警。

    这就是 MQ-R13 的真实分支：模型名**在**选池里，静态层压根没进——
    因为 [strategies.request] enabled=false 且该 agent 无覆盖。
    """
    import logging

    from bladex_core.routing import RequestStrategyData

    r = _router(agent=AgentStrategyData(enabled=False),
                request_strategy=RequestStrategyData(enabled=False))
    with caplog.at_level(logging.WARNING, logger="bladex_core.routing"):
        d = await r.route("任意问题", agent_id="r4-ladder",
                          req_model="openai/strong-x")
    assert d.source is RouteSource.FILTER
    assert "route_request_model_ignored_disabled" in caplog.text
    assert "openai/strong-x" in caplog.text
    assert "routing.toml" in caplog.text


# ── 批 W W1（拍板 (12)，2026-09-22）：策略开 + 名字未知 ⇒ 报错，不替换 ──
#
# 批 V 当时留告警不抛错，理由是"CC 逐轮发 claude-* 名字，抛错打死全部 CC 流量"。
# 那条理由**只对"客户端没覆盖 model 名"的部署成立**，而那正是本卡要报错的那一格：
# 用户为这个 agent 明确把策略打开了 = 明说"我写哪个就用哪个"。
# live 校核（09-22 全量 logs/proxy-*.log）：`route_request_model_unknown` **0 次**；
# claude-code **99/99** 轮 `source=requested`；落「策略关」那一格的是
# hermes:default(80 次 ark-code-latest) 与 dsh(9 次 deepseek-v4-flash)——本改动不碰它们。


@pytest.mark.asyncio
async def test_req_model_unknown_name_raises_instead_of_silently_swapping(caplog):
    """判据①：策略开 + 名字不在 [[models]] ⇒ 抛 UnknownRequestedModelError，零替换。

    "零替换"在 core 这一层的判据 = **没有 RouteDecision 被造出来**（异常出口），
    且 filter 裁判一次都没被调用（下面 `judged == []`）——端到端的"零上游调用"
    由 proxy 侧 `test_route_unknown_req_model_wiring.py` 钉。
    """
    import logging

    from bladex_core.routing import UnknownRequestedModelError

    r = _router(agent=AgentStrategyData(enabled=False))
    judged: list[str] = []

    async def _spy(query, session_id=None):  # noqa: ANN001
        judged.append(query)
        return "weak"

    r._judge_tier = _spy
    with caplog.at_level(logging.WARNING, logger="bladex_core.routing"):
        with pytest.raises(UnknownRequestedModelError) as ei:
            await r.route("任意问题", agent_id="r4-ladder",
                          req_model="openai/no-such-model")
    assert ei.value.req_model == "openai/no-such-model"
    assert ei.value.agent_id == "r4-ladder"
    assert "openai/no-such-model" in str(ei.value)
    # 告警事件名不变（既有 grep 仍有效），措辞改为"拒绝"而不是"回落"
    assert "route_request_model_unknown" in caplog.text
    assert "routing.toml" in caplog.text
    assert "agent_overrides" in caplog.text
    assert judged == []


@pytest.mark.asyncio
async def test_req_model_unknown_negative_control_strategy_disabled(caplog):
    """判据②（阴性对照，**照 dsh 的 live 形态写**）：策略**关** ⇒ 不许抛错。

    live 形态（`logs/proxy-20260920-152146.log`，9/9 轮）：
      `route_request_model_ignored_disabled agent=dsh req_model=deepseek-v4-flash`
    —— 名字**在** [[models]] 里，但 `[strategies.request]` 对 dsh 是关的。
    拍板 (12) 明写这一格是配置语义、保持不动；改成报错会打死 dsh 全部流量。
    这里连"名字未知"也一并覆盖：策略关时**任意名**都只告警不抛错。
    """
    import logging

    from bladex_core.routing import RequestStrategyData, UnknownRequestedModelError

    r = _router(agent=AgentStrategyData(enabled=False),
                request_strategy=RequestStrategyData(enabled=False))
    with caplog.at_level(logging.WARNING, logger="bladex_core.routing"):
        # ① dsh 的真实形态：名字在池里
        d1 = await r.route("任意问题", agent_id="dsh", req_model="openai/strong-x")
        # ② 策略关 + 名字也不在池里
        d2 = await r.route("任意问题", agent_id="dsh", req_model="deepseek-v4-flash")
    assert d1.source is RouteSource.FILTER
    assert d2.source is RouteSource.FILTER
    assert caplog.text.count("route_request_model_ignored_disabled") == 2
    assert "route_request_model_unknown" not in caplog.text
    assert UnknownRequestedModelError.__name__ not in caplog.text


@pytest.mark.asyncio
async def test_req_model_known_name_still_routes_requested():
    """判据⑥（回归）：CC 的 live 形态——名字**在** [[models]] 内 ⇒ 仍 source=requested。"""
    r = _router(agent=AgentStrategyData(enabled=False))
    d = await r.route("任意问题", agent_id="claude-code", req_model="openai/strong-x")
    assert d.source is RouteSource.REQUESTED
    assert d.model == "openai/strong-x"


# ── 批 W W4（MQ-R14）：熔断挤掉静态首选 ⇒ warning，不是 info ──


class _Health:
    """HealthCheck Protocol 的最小实现：给定一组熔断中的模型名。"""

    def __init__(self, unhealthy: set[str]) -> None:
        self._unhealthy = unhealthy

    def is_healthy(self, model: str) -> bool:
        return model not in self._unhealthy


@pytest.mark.asyncio
async def test_static_pick_displaced_by_circuit_breaker_warns(caplog):
    """判据①：static_hit ∧ health_degraded ⇒ warning，点名 agent/来源/原模型/替代模型/理由。"""
    import logging

    r = _router(
        agent=AgentStrategyData(
            enabled=True,
            map={"claude-code": ["openai/glm-5.2", "openai/strong-x"]}),
        health=_Health({"openai/glm-5.2"}),
    )
    with caplog.at_level(logging.INFO, logger="bladex_core.routing"):
        d = await r.route("任意问题", agent_id="claude-code")
    assert d.model == "openai/strong-x"          # 熔断推翻了静态首选（硬约束，允许）
    assert d.source is RouteSource.AGENT
    assert "route_static_model_circuit_open" in caplog.text
    assert "agent=claude-code" in caplog.text
    assert "source=agent" in caplog.text
    assert "static_model=openai/glm-5.2" in caplog.text
    assert "used_model=openai/strong-x" in caplog.text
    assert "reason=health_circuit_open" in caplog.text
    # 且它必须是 WARNING——info 正是 MQ-R14 的本体
    levels = {rec.levelno for rec in caplog.records
              if "route_static_model_circuit_open" in rec.getMessage()}
    assert levels == {logging.WARNING}


@pytest.mark.asyncio
async def test_dynamic_pick_displaced_by_circuit_breaker_stays_info(caplog):
    """判据②（阴性对照）：非静态命中的熔断跳过 ⇒ 仍只有 info，不刷成告警。

    正常 failover 是设计内行为，把它升成 warning 等于把告警面淹掉。
    """
    import logging

    r = _router(agent=AgentStrategyData(enabled=False),
                health=_Health({"openai/doubao-lite"}))
    with caplog.at_level(logging.INFO, logger="bladex_core.routing"):
        d = await r.route("任意问题", agent_id="r4-ladder")
    assert d.model == "anthropic/doubao-lite"
    assert "route_skip_unhealthy" in caplog.text          # 原有 info 还在
    assert "route_static_model_circuit_open" not in caplog.text


@pytest.mark.asyncio
async def test_requested_source_never_reaches_health_filter(caplog):
    """判据③：REQUESTED **不在**本卡覆盖面内——它提前返回、不过 _filter_healthy。

    熔断中的 req_model 走的是 failover 池（"首选走不通再换"），不是"被换掉"，
    所以这里既不该有 route_static_model_circuit_open，模型也不该变。
    """
    import logging

    r = _router(agent=AgentStrategyData(enabled=False),
                health=_Health({"openai/strong-x"}))
    with caplog.at_level(logging.INFO, logger="bladex_core.routing"):
        d = await r.route("任意问题", agent_id="claude-code", req_model="openai/strong-x")
    assert d.source is RouteSource.REQUESTED
    assert d.model == "openai/strong-x"                   # 熔断中照样是它
    assert "route_static_model_circuit_open" not in caplog.text


# ── 批 Z · Z2.2（判据 ③）：静态命中 + 窗口不够 ⇒ 路由逐字不变、告警点名五项 ──


@pytest.mark.asyncio
async def test_static_strategy_wins_even_when_context_window_is_too_small(caplog):
    """刚性原则 9 的窗口版：配置钉死的模型放不下这轮 prompt 也**不换**（能力不满足 = 配置错误），
    但告警必须点名 strategy / agent / model / 窗口 / 估算 prompt 并指向 routing.toml。"""
    import logging

    models = _models()
    models["openai/glm-5.2"].context_window = 8000
    r = _router(models=models)
    with caplog.at_level(logging.WARNING, logger="bladex_core.routing"):
        d = await r.route("很长的请求", agent_id="claude-code",
                          prompt_tokens_est=10_000, output_floor_tokens=512)
    assert d.source is RouteSource.AGENT and d.model == "openai/glm-5.2"
    assert [c.model for c in d.pool] == ["openai/glm-5.2"], "静态池连 failover 顺序都不许动"
    for needle in ("route_static_context_window_exceeded", "source=agent", "agent=claude-code",
                   "model=openai/glm-5.2", "context_window=8000", "prompt_tokens_est=10000",
                   "output_floor=512", "routing.toml"):
        assert needle in caplog.text, needle


@pytest.mark.asyncio
async def test_static_pool_with_a_fitting_alternative_still_keeps_the_pinned_first(caplog):
    """反向变异的判别力：静态池里有第二个放得下的模型时，仍然首选钉死的那个、
    池顺序不变——把窗口过滤也套到静态池上（"自动换模型"）本条立刻红。"""
    import logging

    models = _models()
    models["openai/glm-5.2"].context_window = 8000
    r = _router(models=models,
                agent=AgentStrategyData(enabled=True,
                                        map={"claude-code": ["openai/glm-5.2", "openai/doubao-lite"]}))
    with caplog.at_level(logging.WARNING, logger="bladex_core.routing"):
        d = await r.route("很长的请求", agent_id="claude-code", prompt_tokens_est=10_000)
    assert d.model == "openai/glm-5.2"
    assert [c.model for c in d.pool] == ["openai/glm-5.2", "openai/doubao-lite"]
    assert "route_static_context_window_exceeded" in caplog.text


@pytest.mark.asyncio
async def test_no_context_window_warning_when_it_fits_or_is_unset(caplog):
    import logging

    r = _router()          # glm 没配窗口
    with caplog.at_level(logging.WARNING, logger="bladex_core.routing"):
        d = await r.route("q", agent_id="claude-code", prompt_tokens_est=10_000_000)
    assert d.model == "openai/glm-5.2"
    assert "route_static_context_window_exceeded" not in caplog.text
