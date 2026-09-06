"""MemoryAwareRouter 路由决策测试（Phase 1：配置驱动确定性流水线）。

全策略默认关 → 回归 upstream 顺序 failover。
agent/multimodal/filter 三条策略可独立开启，流水线逐级收窄。
裁判全程 mock（不依赖网络）。
"""

from __future__ import annotations

import asyncio

import pytest
from bladex_core.fact import Fact, HardRule
from bladex_core.routing import (
    AgentStrategyData,
    FilterStrategyData,
    JudgeResult,
    MemoryAwareRouter,
    ModelCandidate,
    MultimodalStrategyData,
    NoCapableCandidateError,
    RouteSource,
)

# ── 工具 ──


def _models() -> dict[str, ModelCandidate]:
    return {
        "pro": ModelCandidate(model="pro", tier="strong", capabilities=["text", "vision"]),
        "glm": ModelCandidate(model="glm", tier="medium", capabilities=["text"]),
        "deepseek": ModelCandidate(model="deepseek", tier="weak", capabilities=["text"]),
        "cheap-vision": ModelCandidate(model="cheap-vision", tier="weak", capabilities=["text", "vision"]),
    }


def _upstream() -> list[str]:
    return ["pro", "glm", "deepseek"]


def _run(coro):
    return asyncio.run(coro)


class FakeJudge:
    """可控 LLM 裁判 mock。"""

    def __init__(self, tier: str = "medium", *, fail: bool = False) -> None:
        self._tier = tier
        self._fail = fail
        self.called = False

    async def judge(self, prompt: str) -> JudgeResult:
        self.called = True
        if self._fail:
            raise RuntimeError("judge boom")
        return JudgeResult(tier=self._tier, reason="fake")


# ── T2：流水线骨架 ──


def test_all_off_uses_upstream_default():
    """全关 → source=UPSTREAM_DEFAULT，primary=upstream[0]。"""
    router = MemoryAwareRouter(_models(), _upstream())
    d = _run(router.route("hi"))
    assert d.model == "pro"
    assert d.source == RouteSource.UPSTREAM_DEFAULT
    assert "upstream default" in d.reason


def test_all_off_failover_chain():
    """全关 → pool = upstream 全量，failover 按序。"""
    router = MemoryAwareRouter(_models(), _upstream())
    d = _run(router.route("hi"))
    fo = router.failover_candidates(d)
    assert [c.model for c in fo] == ["glm", "deepseek"]


def test_core_routing_does_not_import_proxy():
    """依赖方向铁律：core 模块源码不依赖 proxy/agent。"""
    import bladex_core.routing as r
    src = open(r.__file__, encoding="utf-8").read()
    assert "bladex_proxy" not in src


def test_router_does_not_query_index():
    """路由不持有 retriever（Phase 2 预留 facts/hard_rules 参数但不查 Memory Index）。"""
    router = MemoryAwareRouter(_models(), _upstream())
    assert not hasattr(router, "_retriever")


def test_hard_rule_source_reserved():
    """HARD_RULE 是 Phase 2 预留槽，Phase 1 不接逻辑。"""
    router = MemoryAwareRouter(_models(), _upstream())
    # 传 facts/hard_rules 也不触发（Phase 1 不实现）
    d = _run(router.route("hi", facts=[], hard_rules=[]))
    assert d.source == RouteSource.UPSTREAM_DEFAULT


# ── T3：agent 策略（开关 + profile 级 + 默认关）──


def _router_with_agent() -> MemoryAwareRouter:
    return MemoryAwareRouter(
        _models(), _upstream(),
        agent=AgentStrategyData(
            enabled=True,
            map={
                "hermes:accept": ["glm", "deepseek"],
                "hermes": ["pro"],
                "codex": ["deepseek"],
            },
        ),
    )


def test_agent_profile_specific_over_base():
    """hermes:accept 命中 → 走 accept 列表（glm），不走 base（pro）。"""
    router = _router_with_agent()
    d = _run(router.route("hi", agent_id="hermes:accept"))
    assert d.model == "glm"
    assert d.source == RouteSource.AGENT
    assert "hermes:accept" in d.reason


def test_agent_base_fallback():
    """hermes:default 未配 → 剥 :profile 查 base hermes → 走 pro。"""
    router = _router_with_agent()
    d = _run(router.route("hi", agent_id="hermes:default"))
    assert d.model == "pro"
    assert d.source == RouteSource.AGENT


def test_agent_absent_fallthrough():
    """未配的 agent → fall-through 到 upstream。"""
    router = _router_with_agent()
    d = _run(router.route("hi", agent_id="unknown-agent"))
    assert d.source == RouteSource.UPSTREAM_DEFAULT


def test_agent_disabled_fallthrough():
    """agent.enabled=False → 即使配了 map 也不生效。"""
    router = MemoryAwareRouter(
        _models(), _upstream(),
        agent=AgentStrategyData(enabled=False, map={"hermes": ["deepseek"]}),
    )
    d = _run(router.route("hi", agent_id="hermes"))
    assert d.source == RouteSource.UPSTREAM_DEFAULT
    assert d.model == "pro"  # upstream[0], not deepseek


def test_agent_match_with_capability_filter():
    """agent 命中 + 带图 → 集合内选 vision 模型（不越到全局）。"""
    router = MemoryAwareRouter(
        _models(), _upstream(),
        agent=AgentStrategyData(
            enabled=True,
            map={"hermes": ["pro", "glm"]},  # glm 无 vision
        ),
    )
    d = _run(router.route("看图", agent_id="hermes", requires=["vision"]))
    assert d.model == "pro"  # 集合内仅 pro 支持 vision
    assert d.source == RouteSource.AGENT


# ── T4：能力过滤（始终生效）+ 多模态映射 ──


def test_capability_filter_always_on():
    """带图 + 池有 vision 模型 → 选 vision 模型（即使排后）。"""
    router = MemoryAwareRouter(
        {
            "glm": ModelCandidate(model="glm", tier="medium", capabilities=["text"]),
            "pro": ModelCandidate(model="pro", tier="strong", capabilities=["text", "vision"]),
        },
        ["glm", "pro"],  # glm 在前
    )
    d = _run(router.route("看图", requires=["vision"]))
    assert d.model == "pro"  # 能力过滤后只剩 pro


def test_no_capable_fastfail():
    """带图 + 池无 vision → NoCapableCandidateError（不静默降级）。"""
    router = MemoryAwareRouter(
        {
            "glm": ModelCandidate(model="glm", tier="medium", capabilities=["text"]),
            "deepseek": ModelCandidate(model="deepseek", tier="weak", capabilities=["text"]),
        },
        ["glm", "deepseek"],
    )
    with pytest.raises(NoCapableCandidateError):
        _run(router.route("看图", requires=["vision"]))


def test_multimodal_map_optional_preference():
    """multimodal 开 + vision map 有配 → 偏好映射内模型（排前）。"""
    router = MemoryAwareRouter(
        {
            "pro": ModelCandidate(model="pro", tier="strong", capabilities=["text", "vision"]),
            "pro2": ModelCandidate(model="pro2", tier="strong", capabilities=["text", "vision"]),
        },
        ["pro2", "pro"],  # pro2 在前
        multimodal=MultimodalStrategyData(
            enabled=True, map={"vision": ["pro"]},  # 偏好 pro
        ),
    )
    d = _run(router.route("看图", requires=["vision"]))
    assert d.model == "pro"  # 偏好排序后 pro 排前


def test_multimodal_map_fallback_when_pool_empty():
    """池被能力过滤清空 + multimodal 开 + map 有配 → 用映射模型兜底。"""
    router = MemoryAwareRouter(
        {
            "glm": ModelCandidate(model="glm", tier="medium", capabilities=["text"]),
            "pro": ModelCandidate(model="pro", tier="strong", capabilities=["text", "vision"]),
        },
        ["glm"],  # upstream 只有 glm（无 vision）
        multimodal=MultimodalStrategyData(
            enabled=True, map={"vision": ["pro"]},
        ),
    )
    d = _run(router.route("看图", requires=["vision"]))
    assert d.model == "pro"
    assert d.source == RouteSource.CAPABILITY


def test_multimodal_disabled_no_fallback():
    """multimodal 关 + 池无 vision → 抛错（不做兜底）。"""
    router = MemoryAwareRouter(
        {
            "glm": ModelCandidate(model="glm", tier="medium", capabilities=["text"]),
            "pro": ModelCandidate(model="pro", tier="strong", capabilities=["text", "vision"]),
        },
        ["glm"],
        multimodal=MultimodalStrategyData(enabled=False, map={"vision": ["pro"]}),
    )
    with pytest.raises(NoCapableCandidateError):
        _run(router.route("看图", requires=["vision"]))


def test_text_request_unconstrained():
    """纯文本 requires=[] → 不约束，按 upstream 顺序。"""
    router = MemoryAwareRouter(_models(), _upstream())
    d = _run(router.route("hi", requires=[]))
    assert d.model == "pro"


def test_failover_preserves_capability():
    """带图请求 failover 只含 vision 模型（不退纯文本）。"""
    router = MemoryAwareRouter(
        {
            "pro": ModelCandidate(model="pro", tier="strong", capabilities=["text", "vision"]),
            "pro2": ModelCandidate(model="pro2", tier="strong", capabilities=["text", "vision"]),
            "glm": ModelCandidate(model="glm", tier="medium", capabilities=["text"]),
        },
        ["pro", "pro2", "glm"],
    )
    d = _run(router.route("看图", requires=["vision"]))
    assert d.model == "pro"
    fo = router.failover_candidates(d)
    assert all("vision" in c.capabilities for c in fo)
    assert "glm" not in [c.model for c in fo]


# ── T9：auxiliary 内部调用 ──


def test_aux_forces_cheap_tier():
    """auxiliary=True → 强制 weak 档（deepseek）。"""
    router = MemoryAwareRouter(_models(), _upstream(), aux_tier="weak")
    d = _run(router.route("complex task", auxiliary=True))
    assert d.model == "deepseek"
    assert d.tier == "weak"
    assert d.source == RouteSource.AUXILIARY


def test_aux_capability_still_applies():
    """auxiliary + 带图 + weak 档有 vision → 选便宜多模态。"""
    router = MemoryAwareRouter(_models(), _upstream(), aux_tier="weak")
    d = _run(router.route("看图压缩", auxiliary=True, requires=["vision"]))
    assert d.model == "cheap-vision"
    assert d.source == RouteSource.AUXILIARY


def test_aux_no_cheap_capable_falls_through():
    """auxiliary + 带图 + weak 档无 vision → 落到正常路由。"""
    router = MemoryAwareRouter(
        {
            "pro": ModelCandidate(model="pro", tier="strong", capabilities=["text", "vision"]),
            "deepseek": ModelCandidate(model="deepseek", tier="weak", capabilities=["text"]),
        },
        ["pro", "deepseek"],
        aux_tier="weak",
    )
    d = _run(router.route("看图压缩", auxiliary=True, requires=["vision"]))
    assert d.model == "pro"  # 正常路由选 pro
    assert d.source != RouteSource.AUXILIARY


# ── T5：筛选器策略（LLM 裁判）──


def _router_with_filter(judge: FakeJudge | None = None) -> MemoryAwareRouter:
    return MemoryAwareRouter(
        _models(), _upstream(),
        filter_strategy=FilterStrategyData(
            enabled=True,
            candidates={
                "weak": ["deepseek"],
                "medium": ["glm"],
                "strong": ["pro"],
            },
            default_tier="weak",
            max_prompt_chars=2000,
        ),
        judge=judge,
    )


def test_filter_judges_tier_small_prompt():
    """filter 开 + 小 prompt → 裁判判出的档对应模型在首位。"""
    judge = FakeJudge(tier="medium")
    router = _router_with_filter(judge)
    d = _run(router.route("summarize this", agent_id=None))
    assert d.model == "glm"  # medium 档
    assert d.source == RouteSource.FILTER
    assert judge.called is True


def test_filter_skips_large_prompt():
    """大 prompt → 跳过裁判走 default_tier（weak）。"""
    judge = FakeJudge(tier="medium")
    router = _router_with_filter(judge)
    big = "x" * 3000
    d = _run(router.route(big))
    assert d.model == "deepseek"  # default_tier=weak
    assert judge.called is False


def test_filter_timeout_fallback():
    """裁判超时/失败 → default_tier，请求不挂。"""
    judge = FakeJudge(fail=True)
    router = _router_with_filter(judge)
    d = _run(router.route("hi"))
    assert d.model == "deepseek"  # default_tier=weak
    assert d.source == RouteSource.FILTER


def test_filter_tier_empty_after_capability_falls_to_default():
    """判出档经能力过滤空 → 退 default_tier（再过能力），default 有能力则成功。"""
    judge = FakeJudge(tier="medium")
    router = MemoryAwareRouter(
        {
            "glm": ModelCandidate(model="glm", tier="medium", capabilities=["text"]),
            "pro": ModelCandidate(model="pro", tier="strong", capabilities=["text", "vision"]),
            "deepseek": ModelCandidate(model="deepseek", tier="weak", capabilities=["text"]),
        },
        ["pro", "glm", "deepseek"],
        filter_strategy=FilterStrategyData(
            enabled=True,
            candidates={
                "medium": ["glm"],    # glm 无 vision
                "strong": ["pro"],    # default_tier=strong, pro 有 vision
            },
            default_tier="strong",
        ),
        judge=judge,
    )
    # 判出 medium → glm → vision 过滤空 → 退 strong → pro 有 vision → pro
    d = _run(router.route("看图", requires=["vision"]))
    assert d.model == "pro"
    assert d.source == RouteSource.FILTER


def test_filter_tier_and_default_both_empty_raises():
    """判出档 + default_tier 都无能力 → NoCapableCandidateError。"""
    judge = FakeJudge(tier="medium")
    router = MemoryAwareRouter(
        {
            "glm": ModelCandidate(model="glm", tier="medium", capabilities=["text"]),
            "pro": ModelCandidate(model="pro", tier="strong", capabilities=["text", "vision"]),
            "deepseek": ModelCandidate(model="deepseek", tier="weak", capabilities=["text"]),
        },
        ["pro", "glm", "deepseek"],
        filter_strategy=FilterStrategyData(
            enabled=True,
            candidates={
                "medium": ["glm"],     # glm 无 vision
                "weak": ["deepseek"],  # deepseek 也无 vision
            },
            default_tier="weak",
        ),
        judge=judge,
    )
    with pytest.raises(NoCapableCandidateError):
        _run(router.route("看图", requires=["vision"]))



def test_filter_disabled_uses_upstream_head():
    """filter 关 → 走 upstream 默认池。"""
    judge = FakeJudge(tier="medium")
    router = MemoryAwareRouter(
        _models(), _upstream(),
        filter_strategy=FilterStrategyData(enabled=False, candidates={"medium": ["glm"]}),
        judge=judge,
    )
    d = _run(router.route("hi"))
    assert d.source == RouteSource.UPSTREAM_DEFAULT
    assert d.model == "pro"
    assert judge.called is False


# ── failover 列表 ──


def test_failover_candidates_same_tier_first():
    """failover 列表：同 tier 优先，排除 primary，保池顺序。"""
    router = MemoryAwareRouter(
        {
            "pro": ModelCandidate(model="pro", tier="strong"),
            "pro2": ModelCandidate(model="pro2", tier="strong"),
            "glm": ModelCandidate(model="glm", tier="medium"),
        },
        ["pro", "pro2", "glm"],
    )
    d = _run(router.route("hi"))
    assert d.model == "pro"
    fo = router.failover_candidates(d)
    assert fo[0].model == "pro2"  # 同 tier 优先
    assert fo[1].model == "glm"
    assert "pro" not in [c.model for c in fo]


# ── Phase 2 预留：facts/hard_rules 不影响 Phase 1 ──


def test_facts_hardrules_ignored_in_phase1():
    """Phase 1 不实现记忆驱动路由，facts/hard_rules 传入不影响决策。"""
    router = MemoryAwareRouter(_models(), _upstream())
    facts = [Fact(id="f1", content="必须用强模型", category="general")]
    hard_rules = [HardRule(content="MUST 用模型 deepseek", fact_id="hr1")]
    d = _run(router.route("hi", facts=facts, hard_rules=hard_rules))
    assert d.source == RouteSource.UPSTREAM_DEFAULT  # 不受影响
    assert d.model == "pro"  # upstream[0]


# ── multimodal 不再参与选池（RA9，routing-fix-plan-20260712 T5d）──


def test_multimodal_no_longer_bypasses_filter_for_vision():
    """RA9：multimodal + filter 都开 → 带图请求照常走裁判判档，不再直上 mapped 强模型。

    能力过滤保证不选到无 vision 模型；multimodal map 只做同档内偏好排序。
    """
    judge = FakeJudge(tier="medium")
    router = MemoryAwareRouter(
        {
            "pro": ModelCandidate(model="pro", tier="strong", capabilities=["text", "vision"]),
            "glm": ModelCandidate(model="glm", tier="medium", capabilities=["text", "vision"]),
            "mini": ModelCandidate(model="mini", tier="weak", capabilities=["text"]),
        },
        ["pro", "glm", "mini"],
        multimodal=MultimodalStrategyData(
            enabled=True, map={"vision": ["pro"]},
        ),
        filter_strategy=FilterStrategyData(
            enabled=True,
            candidates={"weak": ["mini"], "medium": ["glm"], "strong": ["pro"]},
            default_tier="weak",
        ),
        judge=judge,
    )
    # 带图 → 裁判照常跑（judged medium）→ medium 池 [glm]（glm 有 vision，能力过）
    d = _run(router.route("看这张图", requires=["vision"]))
    assert d.model == "glm"
    assert d.source == RouteSource.FILTER
    assert judge.called is True  # 裁判被调用（不再绕过）


def test_multimodal_fallback_still_works_when_tier_pool_lacks_capability():
    """RA9 之后：判出档全无所需能力 → 仍有 multimodal 映射兜底（CAPABILITY）。"""
    judge = FakeJudge(tier="weak")
    router = MemoryAwareRouter(
        {
            "pro": ModelCandidate(model="pro", tier="strong", capabilities=["text", "vision"]),
            "mini": ModelCandidate(model="mini", tier="weak", capabilities=["text"]),
        },
        ["pro", "mini"],
        multimodal=MultimodalStrategyData(
            enabled=True, map={"vision": ["pro"]},
        ),
        filter_strategy=FilterStrategyData(
            enabled=True,
            candidates={"weak": ["mini"], "strong": ["pro"]},
            default_tier="weak",
        ),
        judge=judge,
    )
    # 判 weak → weak 池 [mini] 无 vision → 能力过滤清空 → multimodal 兜底 pro
    d = _run(router.route("看这张图", requires=["vision"]))
    assert d.model == "pro"
    assert d.source == RouteSource.CAPABILITY


def test_filter_still_runs_for_text_when_multimodal_enabled():
    """multimodal + filter 都开 → 纯文本请求走 filter（无 multimodal 匹配）。"""
    judge = FakeJudge(tier="medium")
    router = MemoryAwareRouter(
        {
            "pro": ModelCandidate(model="pro", tier="strong", capabilities=["text", "vision"]),
            "glm": ModelCandidate(model="glm", tier="medium", capabilities=["text"]),
            "mini": ModelCandidate(model="mini", tier="weak", capabilities=["text"]),
        },
        ["pro", "glm", "mini"],
        multimodal=MultimodalStrategyData(
            enabled=True, map={"vision": ["pro"]},
        ),
        filter_strategy=FilterStrategyData(
            enabled=True,
            candidates={"weak": ["mini"], "medium": ["glm"], "strong": ["pro"]},
            default_tier="weak",
        ),
        judge=judge,
    )
    # 纯文本 → multimodal 无匹配 → filter 跑 → medium → glm
    d = _run(router.route("总结一下", requires=[]))
    assert d.model == "glm"
    assert d.source == RouteSource.FILTER
    assert judge.called is True


def test_agent_still_priorities_over_multimodal():
    """agent + multimodal + filter 都开 → agent 命中时优先（不跑 multimodal 也不跑 filter）。"""
    judge = FakeJudge(tier="strong")
    router = MemoryAwareRouter(
        {
            "pro": ModelCandidate(model="pro", tier="strong", capabilities=["text", "vision"]),
            "glm": ModelCandidate(model="glm", tier="medium", capabilities=["text", "vision"]),
            "mini": ModelCandidate(model="mini", tier="weak", capabilities=["text"]),
        },
        ["pro", "glm", "mini"],
        agent=AgentStrategyData(
            enabled=True, map={"hermes": ["glm", "mini"]},
        ),
        multimodal=MultimodalStrategyData(
            enabled=True, map={"vision": ["pro"]},
        ),
        filter_strategy=FilterStrategyData(
            enabled=True,
            candidates={"weak": ["mini"], "medium": ["glm"], "strong": ["pro"]},
            default_tier="weak",
        ),
        judge=judge,
    )
    # agent hermes 命中 → 走 agent 集合 [glm, mini]（即使带图也不走 multimodal 的 pro）
    d = _run(router.route("看图", agent_id="hermes", requires=["vision"]))
    # agent 集合里 glm 有 vision → 选 glm（不是 multimodal 的 pro）
    assert d.source == RouteSource.AGENT
    assert d.model == "glm"
    assert judge.called is False
