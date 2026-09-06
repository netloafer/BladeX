"""ADR-0021 T4 验收：路由敏感硬边界 + 裁判钉本地（bladex_core routing）。

覆盖计划卡 T4 ①②③（⑤ 敏感层关闭=现状由全量套件零回归已证，④ strict 拒启动在 proxy 测）：
① failover/sticky/aux 不越界三剧本；
② 允许池空快返错（SENSITIVITY source）；
③ 无本地裁判跳裁判走允许池 default_tier。
"""

import pytest
from bladex_core.routing import (
    AgentStrategyData,
    FilterStrategyData,
    MemoryAwareRouter,
    ModelCandidate,
    NoCapableCandidateError,
    PrincipalStrategyData,
    RouteSource,
    StickyStrategyData,
    TeamStrategyData,
)


class _RecordingJudge:
    """记录调用次数的裁判 mock。"""

    def __init__(self, tier: str = "weak") -> None:
        self._tier = tier
        self.call_count = 0

    async def judge(self, prompt: str):  # type: ignore[no-untyped-def]
        self.call_count += 1
        from bladex_core.routing import JudgeResult
        return JudgeResult(tier=self._tier, reason="mock")


def _models() -> dict[str, ModelCandidate]:
    return {
        "local-weak-a": ModelCandidate(model="local-weak-a", tier="weak", exposure="local"),
        "local-weak-b": ModelCandidate(model="local-weak-b", tier="weak", exposure="local"),
        "pub-weak": ModelCandidate(model="pub-weak", tier="weak", exposure="public"),
        "local-medium": ModelCandidate(model="local-medium", tier="medium", exposure="local"),
        "pub-strong": ModelCandidate(model="pub-strong", tier="strong", exposure="public"),
    }


def _router(judge=None, judge_model: str = "", sticky: bool = True) -> MemoryAwareRouter:
    return MemoryAwareRouter(
        models=_models(),
        upstream=["pub-weak", "local-weak-a", "local-weak-b"],
        filter_strategy=FilterStrategyData(
            enabled=True,
            candidates={
                "weak": ["pub-weak", "local-weak-a", "local-weak-b"],
                "medium": ["local-medium"],
                "strong": ["pub-strong"],
            },
            default_tier="weak",
        ),
        judge=judge,
        judge_model=judge_model,
        sticky=StickyStrategyData(enabled=sticky),
    )


# ── ① failover 不越界 ──


@pytest.mark.asyncio
async def test_failover_does_not_cross_exposure_boundary():
    """allowed=local：primary 与 failover 全部 local，无 public 越界。"""
    judge = _RecordingJudge("weak")
    router = _router(judge=judge, judge_model="local-weak-a")
    # allowed=local -> 裁判 local（judge_ok=True），判 weak -> weak 池裁剪到 local
    _model, _route, failover = await _decide(router, allowed_exposure="local")
    assert _route.model == "local-weak-a"
    assert all(c.model.startswith("local-") for c in failover), \
        f"public model leaked into failover: {[c.model for c in failover]}"
    # 对比：allowed=public -> public 模型可选（primary=pub-weak）
    m2, _, failover2 = await _decide(router, allowed_exposure="public")
    assert m2 == "pub-weak"
    assert "pub-weak" not in [c.model for c in failover]  # local 路径 failover 无 public


# ── ① sticky 不越界 ──


@pytest.mark.asyncio
async def test_sticky_does_not_reuse_out_of_exposure_model():
    """会话粘住 pub-weak（public 请求）；后续敏感请求 allowed=local 不沿用 pub-weak。"""
    judge = _RecordingJudge("weak")
    router = _router(judge=judge, judge_model="local-weak-a")
    # 第一次：allowed=public -> primary=pub-weak（candidates 首位），粘住
    m1, _, _ = await _decide(router, allowed_exposure="public", session_id="s1")
    assert m1 == "pub-weak"
    # 第二次：同会话 allowed=local -> pub-weak 越界，sticky 不沿用，落允许池 local-weak-a
    m2, _, _ = await _decide(router, allowed_exposure="local", session_id="s1")
    assert m2 == "local-weak-a", f"sticky leaked public model: {m2}"


# ── ① aux 不越界 ──


@pytest.mark.asyncio
async def test_aux_does_not_cross_exposure_boundary():
    """auxiliary 调用 allowed=local：便宜档只选 local 模型。"""
    router = _router()
    m, route, failover = await _decide(router, allowed_exposure="local", auxiliary=True)
    assert m.startswith("local-")
    assert route.source == RouteSource.AUXILIARY
    assert "pub-weak" not in [m] + [c.model for c in failover]  # local 路径无 public
    # 对比 allowed=public -> pub-weak 在 aux 池里（primary 或 failover）
    m2, _, failover2 = await _decide(router, allowed_exposure="public", auxiliary=True)
    assert "pub-weak" in [m2] + [c.model for c in failover2]


# ── ② 允许池空快返错 ──


@pytest.mark.asyncio
async def test_empty_allowed_pool_fast_error():
    """allowed=local 但池里无任何 local 模型 -> NoCapableCandidateError。"""
    models = {
        "pub-weak": ModelCandidate(model="pub-weak", tier="weak", exposure="public"),
        "pub-strong": ModelCandidate(model="pub-strong", tier="strong", exposure="public"),
    }
    router = MemoryAwareRouter(
        models=models, upstream=["pub-weak"],
        filter_strategy=FilterStrategyData(enabled=True, candidates={"weak": ["pub-weak"]}, default_tier="weak"),
    )
    with pytest.raises(NoCapableCandidateError, match="sensitivity"):
        await router.route("q", allowed_exposure="local")


# ── ③ 无本地裁判跳裁判走 default_tier ──


@pytest.mark.asyncio
async def test_non_local_judge_skipped_in_sensitive_mode():
    """allowed=local + judge=pub-weak(非本地) -> 跳裁判，走允许池 default_tier(weak)。"""
    judge = _RecordingJudge("medium")  # 若被调用会判 medium
    router = _router(judge=judge, judge_model="pub-weak")
    m, route, _ = await _decide(router, allowed_exposure="local")
    assert judge.call_count == 0, "judge was called on sensitive query (leak!)"
    assert m == "local-weak-a"  # default_tier=weak 的 local 候选首位
    # 对比：allowed=public + judge local -> 裁判被调用
    judge2 = _RecordingJudge("weak")
    router2 = _router(judge=judge2, judge_model="local-weak-a")
    await _decide(router2, allowed_exposure="public")
    assert judge2.call_count == 1


# ── ⑤ 敏感层关闭（allowed=public）= 无过滤，所有模型可见 ──


@pytest.mark.asyncio
async def test_public_allowed_no_filtering():
    """allowed=public -> 不过滤，所有模型可被选（现状等价）。"""
    judge = _RecordingJudge("strong")
    router = _router(judge=judge, judge_model="local-weak-a")
    m, _, failover = await _decide(router, allowed_exposure="public")
    # 判 strong -> strong 池 = [pub-strong]，primary=pub-strong
    assert m == "pub-strong"


# ── ADR-0021 三层路由优先级 team > principal > agent ──


@pytest.mark.asyncio
async def test_team_route_highest_priority():
    """team > principal > agent：三者都配且命中 -> source=TEAM。"""
    router = MemoryAwareRouter(
        models=_models(), upstream=["pub-weak"],
        agent=AgentStrategyData(enabled=True, map={"hermes:accept": ["pub-weak"]}),
        principal=PrincipalStrategyData(enabled=True, map={"alice": ["local-medium"]}),
        team=TeamStrategyData(enabled=True, map={"rd": ["local-weak-a"]}),
    )
    dec = await router.route("q", agent_id="hermes:accept", principal_id="alice", team_ids=["rd"])
    assert dec.source == RouteSource.TEAM
    assert dec.model == "local-weak-a"


@pytest.mark.asyncio
async def test_principal_route_when_team_unset():
    """team 未命中 -> principal 命中 -> source=PRINCIPAL。"""
    router = MemoryAwareRouter(
        models=_models(), upstream=["pub-weak"],
        agent=AgentStrategyData(enabled=True, map={"hermes:accept": ["pub-weak"]}),
        principal=PrincipalStrategyData(enabled=True, map={"alice": ["local-medium"]}),
        team=TeamStrategyData(enabled=True, map={"ghost-team": ["local-weak-a"]}),
    )
    dec = await router.route("q", agent_id="hermes:accept", principal_id="alice", team_ids=[])
    assert dec.source == RouteSource.PRINCIPAL
    assert dec.model == "local-medium"


@pytest.mark.asyncio
async def test_agent_route_when_principal_team_unset():
    """team/principal 都未命中 -> agent 命中 -> source=AGENT。"""
    router = MemoryAwareRouter(
        models=_models(), upstream=["pub-weak"],
        agent=AgentStrategyData(enabled=True, map={"hermes:accept": ["pub-weak"]}),
        principal=PrincipalStrategyData(enabled=True, map={"ghost": ["local-medium"]}),
        team=TeamStrategyData(enabled=True, map={"ghost-team": ["local-weak-a"]}),
    )
    dec = await router.route("q", agent_id="hermes:accept", principal_id="alice", team_ids=[])
    assert dec.source == RouteSource.AGENT
    assert dec.model == "pub-weak"


@pytest.mark.asyncio
async def test_team_route_ancestor_chain_match():
    """team_ids 含祖先链，任一命中即用。"""
    router = MemoryAwareRouter(
        models=_models(), upstream=["pub-weak"],
        team=TeamStrategyData(enabled=True, map={"rd": ["local-weak-a"]}),
    )
    dec = await router.route("q", team_ids=["rd-backend", "rd"])  # rd-backend 未配，rd 配
    assert dec.source == RouteSource.TEAM
    assert dec.model == "local-weak-a"


async def _decide(router, *, allowed_exposure="public", session_id=None, auxiliary=False):
    """调 router.route -> (model, RouteDecision, failover_candidates)。纯 core 类型。"""
    dec = await router.route(
        "q", session_id=session_id, auxiliary=auxiliary, allowed_exposure=allowed_exposure,
    )
    failover = router.failover_candidates(dec)
    return dec.model, dec, failover


# ── ADR-0021 策略路由 > 能力过滤 ──


@pytest.mark.asyncio
async def test_strategy_route_skips_capability_filter():
    """策略路由(team/principal/agent)高于能力过滤：指定池模型无 vision 也用，不切换。"""
    router = MemoryAwareRouter(
        models=_models(), upstream=["pub-weak"],
        agent=AgentStrategyData(enabled=True, map={"hermes:accept": ["pub-weak"]}),
    )
    # pub-weak 无 vision，带图请求 requires=[vision]
    dec = await router.route("q", agent_id="hermes:accept", requires=["vision"])
    assert dec.source == RouteSource.AGENT
    assert dec.model == "pub-weak"  # 策略命中，跳过能力过滤，不切换到 vision 模型


@pytest.mark.asyncio
async def test_non_strategy_route_still_capability_filtered():
    """非策略路由(filter/upstream)仍走能力过滤：带图无 vision 模型 -> 快返错。"""
    router = MemoryAwareRouter(
        models=_models(), upstream=["pub-weak"],
        filter_strategy=FilterStrategyData(enabled=True, candidates={"weak": ["pub-weak"]}, default_tier="weak"),
    )
    with pytest.raises(NoCapableCandidateError):
        await router.route("q", requires=["vision"])  # 无策略 -> 能力过滤 -> 无 vision -> 错


# ── ADR-0021 E1 收尾红线：误粘 + 敏感 -> 不外泄（review adr-0021-review-20260723 §4.2）──


@pytest.mark.asyncio
async def test_sticky_mismatch_with_sensitivity_no_leak():
    """交叉场景误粘 + 敏感：principal sensitive -> allowed=local，即使 sticky 误粘到
    default agent_id（不在 agent map），敏感过滤仍约束 -> 走 local 模型，不外泄 public。

    review §4.2 红线：同 key 多 agent 误粘是已知遗留，但 T4 敏感过滤（选池最前）
    应天然兜底--principal sensitive 时 allowed_exposure=local，public 模型被裁剪，
    无论 agent_id 误粘到谁都不外泄。此剧本确认该兜底。
    """
    models = {
        "local-m": ModelCandidate(model="local-m", tier="medium", exposure="local"),
        "pub-m": ModelCandidate(model="pub-m", tier="medium", exposure="public"),
    }
    # agent 策略只配 hermes:accept（default 不在 map -> 误粘 default 走 filter）
    router = MemoryAwareRouter(
        models=models, upstream=["pub-m"],
        agent=AgentStrategyData(enabled=True, map={"hermes:accept": ["local-m"]}),
        filter_strategy=FilterStrategyData(
            enabled=True, candidates={"medium": ["pub-m", "local-m"]}, default_tier="medium",
        ),
    )
    # 误粘 hermes:default（不在 map）+ principal sensitive -> allowed=local
    dec = await router.route("q", agent_id="hermes:default", allowed_exposure="local")
    assert dec.model == "local-m"  # 走 local（不外泄 pub-m/public）
    assert dec.source != RouteSource.AGENT  # agent 策略没命中（default 不在 map）
