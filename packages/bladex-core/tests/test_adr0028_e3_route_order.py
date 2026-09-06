"""ADR-0028 E3：路由编排重排——静态路由 > 身份 > 敏感度 > 智能路由。

优先级语义与计算顺序是两回事（ADR-0028 §1.1）：

  - **优先级语义**（谁说了算）：静态路由 > 身份推断 > 敏感度裁剪 > 智能路由；
  - **计算顺序**（先算什么）：身份必须最先算——`[strategies.agent/principal/team]`
    这些静态规则本身以身份为 key，没有身份就查不了静态表。身份是静态匹配的**输入**，
    不是压过静态的**决策层**。

本卡钉的三件事：
  ① 静态命中 → short-circuit：裁判**一次都不许被调用**（调用计数=0），
     aux 降档 / 规模 floor / 粘性同样整段跳过；
  ② 静态命中 + 敏感度越界 → 静态胜出 + `route_static_sensitivity_mismatch` 可 grep；
  ③ 静态未命中 → 敏感度裁剪照常 fail-closed（安全底线不因本卡放松）。
"""

from __future__ import annotations

import logging

import pytest
from bladex_core.routing import (
    AgentStrategyData,
    FilterStrategyData,
    JudgeResult,
    MemoryAwareRouter,
    ModelCandidate,
    NoCapableCandidateError,
    PrincipalStrategyData,
    RequestStrategyData,
    RouteSource,
    ScaleStrategyData,
    StickyStrategyData,
    TeamStrategyData,
)


class _CountingJudge:
    """记录调用次数——本卡的核心断言就是"静态命中时它是 0"。"""

    model_name = "local-weak-a"

    def __init__(self, tier: str = "strong") -> None:
        self._tier = tier
        self.call_count = 0

    async def judge(self, prompt: str) -> JudgeResult:
        self.call_count += 1
        return JudgeResult(tier=self._tier, reason="mock")


def _models() -> dict[str, ModelCandidate]:
    return {
        "local-weak-a": ModelCandidate(model="local-weak-a", tier="weak", exposure="local"),
        "local-medium": ModelCandidate(model="local-medium", tier="medium", exposure="local"),
        "pub-weak": ModelCandidate(model="pub-weak", tier="weak", exposure="public"),
        "pub-strong": ModelCandidate(model="pub-strong", tier="strong", exposure="public"),
    }


def _filter() -> FilterStrategyData:
    return FilterStrategyData(
        enabled=True,
        candidates={"weak": ["local-weak-a"], "medium": ["local-medium"],
                    "strong": ["pub-strong"]},
        default_tier="weak",
    )


# ── ① 静态命中 → 裁判调用计数 = 0 ────────────────────────────────────────


@pytest.mark.asyncio
async def test_static_agent_hit_never_calls_judge():
    judge = _CountingJudge("strong")
    router = MemoryAwareRouter(
        models=_models(), upstream=["local-weak-a"],
        agent=AgentStrategyData(enabled=True, map={"hermes": ["pub-weak"]}),
        filter_strategy=_filter(), judge=judge, judge_model="local-weak-a",
    )
    dec = await router.route("一个看起来很难的问题", agent_id="hermes")

    assert dec.source == RouteSource.AGENT
    assert dec.model == "pub-weak"
    assert judge.call_count == 0            # short-circuit：裁判整段跳过


@pytest.mark.asyncio
async def test_static_team_and_principal_also_short_circuit_judge():
    for kwargs, expect_src, expect_model in (
        ({"team": TeamStrategyData(enabled=True, map={"rd": ["local-medium"]})},
         RouteSource.TEAM, "local-medium"),
        ({"principal": PrincipalStrategyData(enabled=True, map={"alice": ["local-medium"]})},
         RouteSource.PRINCIPAL, "local-medium"),
    ):
        judge = _CountingJudge("strong")
        router = MemoryAwareRouter(
            models=_models(), upstream=["local-weak-a"],
            filter_strategy=_filter(), judge=judge, judge_model="local-weak-a",
            **kwargs,
        )
        dec = await router.route("q", principal_id="alice", team_ids=["rd"])
        assert dec.source == expect_src and dec.model == expect_model
        assert judge.call_count == 0


@pytest.mark.asyncio
async def test_req_model_short_circuits_judge():
    judge = _CountingJudge("strong")
    router = MemoryAwareRouter(
        models=_models(), upstream=["local-weak-a"],
        filter_strategy=_filter(), judge=judge, judge_model="local-weak-a",
        request_strategy=RequestStrategyData(enabled=True),
    )
    dec = await router.route("q", req_model="local-medium")
    assert dec.source == RouteSource.REQUESTED
    assert judge.call_count == 0


@pytest.mark.asyncio
async def test_no_static_still_calls_judge():
    """零回归：静态未命中时智能路由照常跑（否则本卡就是把裁判废了）。"""
    judge = _CountingJudge("strong")
    router = MemoryAwareRouter(
        models=_models(), upstream=["local-weak-a"],
        agent=AgentStrategyData(enabled=True, map={"other-agent": ["pub-weak"]}),
        filter_strategy=_filter(), judge=judge, judge_model="local-weak-a",
    )
    dec = await router.route("q", agent_id="hermes")
    assert judge.call_count == 1
    assert dec.source == RouteSource.FILTER
    assert dec.model == "pub-strong"


@pytest.mark.asyncio
async def test_static_hit_skips_scale_floor_and_sticky():
    """规模 floor 与会话粘性都是动态推断——静态命中时不得改写其选择。"""
    router = MemoryAwareRouter(
        models=_models(), upstream=["local-weak-a"],
        agent=AgentStrategyData(enabled=True, map={"hermes": ["local-weak-a"]}),
        filter_strategy=_filter(),
        scale=ScaleStrategyData(enabled=True, thresholds={"strong": 1000}),
        sticky=StickyStrategyData(enabled=True),
    )
    # 先粘住一个更强的模型
    router.update_sticky("s1", "pub-strong")
    dec = await router.route("q", agent_id="hermes", session_id="s1",
                             context_chars=500_000)
    assert dec.source == RouteSource.AGENT
    assert dec.model == "local-weak-a"      # 既没被 floor 抬档、也没被粘性改写


# ── ② 静态 × 敏感度：静态胜出 + 强告警 ─────────────────────────────────


@pytest.mark.asyncio
async def test_static_wins_over_sensitivity_with_warning(caplog):
    """ADR-0028 §1.2：静态命中 + 敏感度越界 → 用配置的模型 + 可 grep 告警。

    安全底线不破的理由：注入平面的 exposure 守卫独立于路由（ADR-0021 §3.3c）——
    敏感 Fact 无论路由到哪都不会被注入到越界目的地（见 test_adr0021_injection_guard）。
    """
    router = MemoryAwareRouter(
        models=_models(), upstream=["local-weak-a"],
        agent=AgentStrategyData(enabled=True, map={"hermes": ["pub-strong"]}),
        filter_strategy=_filter(),
    )
    with caplog.at_level(logging.WARNING, logger="bladex_core.routing"):
        dec = await router.route("q", agent_id="hermes", allowed_exposure="local")

    assert dec.source == RouteSource.AGENT
    assert dec.model == "pub-strong"                       # 静态胜出
    assert "route_static_sensitivity_mismatch" in caplog.text
    assert "pub-strong" in caplog.text and "local" in caplog.text


@pytest.mark.asyncio
async def test_req_model_wins_over_sensitivity_with_warning(caplog):
    router = MemoryAwareRouter(
        models=_models(), upstream=["local-weak-a"],
        filter_strategy=_filter(),
        request_strategy=RequestStrategyData(enabled=True),
    )
    with caplog.at_level(logging.WARNING, logger="bladex_core.routing"):
        dec = await router.route("q", req_model="pub-strong", allowed_exposure="local")
    assert dec.model == "pub-strong"
    assert "route_static_sensitivity_mismatch" in caplog.text


@pytest.mark.asyncio
async def test_no_warning_when_static_within_exposure(caplog):
    """不越界不告警——告警要稀有才有信噪比。"""
    router = MemoryAwareRouter(
        models=_models(), upstream=["local-weak-a"],
        agent=AgentStrategyData(enabled=True, map={"hermes": ["local-medium"]}),
        filter_strategy=_filter(),
    )
    with caplog.at_level(logging.WARNING, logger="bladex_core.routing"):
        dec = await router.route("q", agent_id="hermes", allowed_exposure="local")
    assert dec.model == "local-medium"
    assert "route_static_sensitivity_mismatch" not in caplog.text


@pytest.mark.asyncio
async def test_static_hit_survives_empty_allowed_pool(caplog):
    """极端情形：allowed 集合为空（一个 local 模型都没有）。

    智能路由此时必须 fail-closed 快返错；但静态命中时静态仍然胜出——
    否则"用户钉死的模型"会因为一条敏感度配置被静默换掉/打死。
    """
    models = {
        "pub-weak": ModelCandidate(model="pub-weak", tier="weak", exposure="public"),
    }
    router = MemoryAwareRouter(
        models=models, upstream=["pub-weak"],
        agent=AgentStrategyData(enabled=True, map={"hermes": ["pub-weak"]}),
        filter_strategy=FilterStrategyData(
            enabled=True, candidates={"weak": ["pub-weak"]}, default_tier="weak"),
    )
    with caplog.at_level(logging.WARNING, logger="bladex_core.routing"):
        dec = await router.route("q", agent_id="hermes", allowed_exposure="local")
    assert dec.model == "pub-weak"
    assert "route_static_sensitivity_mismatch" in caplog.text


# ── ③ 静态未命中 → 敏感度照常 fail-closed ──────────────────────────────


@pytest.mark.asyncio
async def test_dynamic_route_still_fail_closed_on_empty_allowed():
    models = {
        "pub-weak": ModelCandidate(model="pub-weak", tier="weak", exposure="public"),
    }
    router = MemoryAwareRouter(
        models=models, upstream=["pub-weak"],
        filter_strategy=FilterStrategyData(
            enabled=True, candidates={"weak": ["pub-weak"]}, default_tier="weak"),
    )
    with pytest.raises(NoCapableCandidateError, match="sensitivity"):
        await router.route("q", allowed_exposure="local")


@pytest.mark.asyncio
async def test_dynamic_route_still_cut_by_sensitivity():
    """静态未命中：敏感度裁剪照常生效，public 模型不进候选。"""
    router = MemoryAwareRouter(
        models=_models(), upstream=["pub-weak", "local-weak-a"],
        filter_strategy=FilterStrategyData(
            enabled=True,
            candidates={"weak": ["pub-weak", "local-weak-a"]},
            default_tier="weak",
        ),
    )
    dec = await router.route("q", allowed_exposure="local")
    assert dec.model == "local-weak-a"
    assert "pub-weak" not in [c.model for c in dec.pool]


# ── ④ 安全底线：注入守卫与路由解耦（本卡放松路由，绝不放松注入）────────


