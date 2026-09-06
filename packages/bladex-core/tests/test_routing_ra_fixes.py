"""RA 修复卡验收剧本 — core 侧（路由信号层与容错修复 20260712 T7）。

覆盖：T1 规模层 / T2 会话粘性 / T3 裁判 LRU + 兜底 / T4 健康过滤 / T5b req_model。
裁判全程 mock（不依赖网络）。
"""

from __future__ import annotations

import asyncio

import pytest
from bladex_core.routing import (
    FilterStrategyData,
    JudgeResult,
    MemoryAwareRouter,
    ModelCandidate,
    NoCapableCandidateError,
    RequestStrategyData,
    RouteSource,
    ScaleStrategyData,
    StickyStrategyData,
)


def _run(coro):
    return asyncio.run(coro)


def _models() -> dict[str, ModelCandidate]:
    return {
        "pro": ModelCandidate(model="pro", tier="strong",
                              capabilities=["text", "vision"], provider="ark"),
        "glm": ModelCandidate(model="glm", tier="medium",
                              capabilities=["text", "vision"], provider="ark"),
        "mini": ModelCandidate(model="mini", tier="weak",
                               capabilities=["text"], provider="ark"),
    }


class CountingJudge:
    """可控裁判 mock：计调用次数，可设失败。"""

    def __init__(self, tier: str = "weak", *, fail: bool = False) -> None:
        self._tier = tier
        self._fail = fail
        self.calls = 0

    async def judge(self, prompt: str) -> JudgeResult:
        self.calls += 1
        if self._fail:
            raise RuntimeError("judge boom")
        return JudgeResult(tier=self._tier, reason="fake")


class FakeHealth:
    """可控健康检查 mock。"""

    def __init__(self, unhealthy: set[str] | None = None) -> None:
        self.unhealthy = unhealthy or set()

    def is_healthy(self, model: str) -> bool:
        return model not in self.unhealthy


def _filter_data() -> FilterStrategyData:
    return FilterStrategyData(
        enabled=True,
        candidates={"weak": ["mini"], "medium": ["glm"], "strong": ["pro"]},
        default_tier="weak",
    )


# ── T1 规模层（RA1）──


def test_scale_bypass_skips_judge_and_floors_tier():
    """验收剧本1：70K chars 上下文 + "继续" → 判定 ≥ medium、source=SCALE、裁判 0 次。"""
    judge = CountingJudge(tier="weak")
    router = MemoryAwareRouter(
        _models(), ["pro", "glm", "mini"],
        filter_strategy=_filter_data(), judge=judge,
        scale=ScaleStrategyData(enabled=True, floor_medium_chars=60000,
                                floor_strong_chars=400000, judge_bypass_chars=60000),
    )
    d = _run(router.route("继续", context_chars=70000))
    assert d.model == "glm"  # medium 档
    assert d.source == RouteSource.SCALE
    assert judge.calls == 0  # 裁判被跳过


def test_scale_floor_raises_judged_tier():
    """裁判判 weak 但 floor=medium → 抬到 medium（bypass 关，裁判照跑）。"""
    judge = CountingJudge(tier="weak")
    router = MemoryAwareRouter(
        _models(), ["pro", "glm", "mini"],
        filter_strategy=_filter_data(), judge=judge,
        scale=ScaleStrategyData(enabled=True, floor_medium_chars=60000,
                                judge_bypass_chars=10_000_000),
    )
    d = _run(router.route("继续", context_chars=70000))
    assert d.model == "glm"
    assert d.source == RouteSource.SCALE
    assert judge.calls == 1


def test_scale_strong_floor():
    """超 strong 阈值 → floor=strong。"""
    router = MemoryAwareRouter(
        _models(), ["pro", "glm", "mini"],
        filter_strategy=_filter_data(), judge=CountingJudge(tier="weak"),
        scale=ScaleStrategyData(enabled=True, floor_medium_chars=60000,
                                floor_strong_chars=400000, judge_bypass_chars=60000),
    )
    d = _run(router.route("继续", context_chars=500000))
    assert d.model == "pro"
    assert d.source == RouteSource.SCALE


def test_scale_small_context_unchanged():
    """验收剧本2：2K chars 短请求 → floor 不生效，原流程逐字不变。"""
    judge = CountingJudge(tier="weak")
    router = MemoryAwareRouter(
        _models(), ["pro", "glm", "mini"],
        filter_strategy=_filter_data(), judge=judge,
        scale=ScaleStrategyData(enabled=True, floor_medium_chars=60000,
                                judge_bypass_chars=60000),
    )
    d = _run(router.route("hi", context_chars=2000))
    assert d.model == "mini"
    assert d.source == RouteSource.FILTER
    assert judge.calls == 1


def test_scale_floor_empty_pool_keeps_pool():
    """验收剧本3：池里全 weak（无 medium+ 候选）→ 保留原池 + 不抛错。"""
    models = {"mini": ModelCandidate(model="mini", tier="weak", capabilities=["text"])}
    router = MemoryAwareRouter(
        models, ["mini"],
        scale=ScaleStrategyData(enabled=True, floor_medium_chars=60000,
                                judge_bypass_chars=0),
    )
    d = _run(router.route("hi", context_chars=70000))
    assert d.model == "mini"  # floor 是偏好性下限，不打死请求


def test_scale_disabled_no_effect():
    """验收剧本4：enabled=false → 现有路径无变化。"""
    judge = CountingJudge(tier="weak")
    router = MemoryAwareRouter(
        _models(), ["pro", "glm", "mini"],
        filter_strategy=_filter_data(), judge=judge,
    )
    d = _run(router.route("继续", context_chars=70000))
    assert d.model == "mini"
    assert d.source == RouteSource.FILTER


# ── T2 会话粘性（RA2）──


def _sticky_router(judge: CountingJudge) -> MemoryAwareRouter:
    return MemoryAwareRouter(
        _models(), ["pro", "glm", "mini"],
        filter_strategy=_filter_data(), judge=judge,
        sticky=StickyStrategyData(enabled=True),
    )


def test_sticky_holds_on_downgrade():
    """验收剧本1：轮1 判 strong 选 pro；轮2 判 weak → 仍 pro（source=STICKY）。"""
    judge = CountingJudge(tier="strong")
    router = _sticky_router(judge)
    d1 = _run(router.route("复杂架构设计", session_id="s1"))
    assert d1.model == "pro"

    judge._tier = "weak"
    d2 = _run(router.route("继续", session_id="s1"))
    assert d2.model == "pro"
    assert d2.source == RouteSource.STICKY


def test_sticky_escalates_and_updates():
    """验收剧本2：sticky=weak、判 strong → 升级切换并更新 sticky。"""
    judge = CountingJudge(tier="weak")
    router = _sticky_router(judge)
    d1 = _run(router.route("hi", session_id="s1"))
    assert d1.model == "mini"

    judge._tier = "strong"
    d2 = _run(router.route("设计一个分布式系统", session_id="s1"))
    assert d2.model == "pro"
    assert d2.source == RouteSource.FILTER

    judge._tier = "weak"
    d3 = _run(router.route("继续", session_id="s1"))
    assert d3.model == "pro"  # sticky 已更新为 pro
    assert d3.source == RouteSource.STICKY


def test_sticky_capability_mismatch_not_stuck():
    """验收剧本3：sticky 模型无 vision、带图请求 → 不粘，走能力合格池。"""
    judge = CountingJudge(tier="weak")
    router = _sticky_router(judge)
    d1 = _run(router.route("hi", session_id="s1"))
    assert d1.model == "mini"  # mini 无 vision

    judge._tier = "medium"
    d2 = _run(router.route("看图", session_id="s1", requires=["vision"]))
    assert d2.model == "glm"  # 不粘 mini


def test_sticky_sessions_isolated():
    """验收剧本5：不同 session 互不影响；session_id=None 行为同关闭。"""
    judge = CountingJudge(tier="strong")
    router = _sticky_router(judge)
    _run(router.route("q", session_id="s1"))

    judge._tier = "weak"
    d_other = _run(router.route("q2", session_id="s2"))
    assert d_other.model == "mini"  # s2 无粘性历史

    d_none = _run(router.route("q3"))
    assert d_none.model == "mini"  # 无 session → 正常流水线


def test_sticky_skips_unhealthy_model():
    """sticky 模型熔断中 → 不粘。"""
    judge = CountingJudge(tier="strong")
    health = FakeHealth()
    router = MemoryAwareRouter(
        _models(), ["pro", "glm", "mini"],
        filter_strategy=_filter_data(), judge=judge,
        sticky=StickyStrategyData(enabled=True), health=health,
    )
    _run(router.route("q", session_id="s1"))  # sticky = pro

    judge._tier = "weak"
    health.unhealthy = {"pro"}
    d = _run(router.route("继续", session_id="s1"))
    assert d.model == "mini"  # pro 熔断 → 不粘


def test_auxiliary_does_not_touch_sticky():
    """auxiliary 不参与粘性（不读不写）。"""
    judge = CountingJudge(tier="strong")
    router = _sticky_router(judge)
    _run(router.route("q", session_id="s1"))  # sticky = pro

    d_aux = _run(router.route("内部压缩", session_id="s1", auxiliary=True))
    assert d_aux.source == RouteSource.AUXILIARY
    assert d_aux.model == "mini"  # aux 强制便宜档，不受 sticky 影响

    judge._tier = "weak"
    d = _run(router.route("继续", session_id="s1"))
    assert d.model == "pro"  # sticky 仍是 pro（aux 没覆盖它）


# ── T3 裁判 LRU + 兜底（RA3）──


def test_judge_cache_hit_in_tool_loop():
    """验收剧本1：同 session 同 query 连续 5 次 → 裁判只调 1 次。"""
    judge = CountingJudge(tier="medium")
    router = MemoryAwareRouter(
        _models(), ["pro", "glm", "mini"],
        filter_strategy=_filter_data(), judge=judge,
    )
    for _ in range(5):
        d = _run(router.route("帮我改这个函数", session_id="s1"))
        assert d.model == "glm"
    assert judge.calls == 1


def test_judge_cache_isolated_by_query():
    """不同 query → 各判一次。"""
    judge = CountingJudge(tier="medium")
    router = MemoryAwareRouter(
        _models(), ["pro", "glm", "mini"],
        filter_strategy=_filter_data(), judge=judge,
    )
    _run(router.route("q1", session_id="s1"))
    _run(router.route("q2", session_id="s1"))
    assert judge.calls == 2


def test_judge_failure_falls_back_to_sticky_tier():
    """验收剧本2：裁判失败且 session 粘住 strong → 判定档 strong（非 weak）。"""
    judge = CountingJudge(tier="strong")
    router = MemoryAwareRouter(
        _models(), ["pro", "glm", "mini"],
        filter_strategy=_filter_data(), judge=judge,
        sticky=StickyStrategyData(enabled=True),
    )
    _run(router.route("复杂问题", session_id="s1"))  # sticky = pro/strong

    judge._fail = True
    d = _run(router.route("另一个问题", session_id="s1"))
    assert d.model == "pro"  # 兜底档 = sticky tier strong，而非 default weak


def test_judge_failure_default_without_history():
    """验收剧本3：无粘性历史 + 裁判失败 → default_tier（回归现状）。"""
    judge = CountingJudge(fail=True)
    router = MemoryAwareRouter(
        _models(), ["pro", "glm", "mini"],
        filter_strategy=_filter_data(), judge=judge,
    )
    d = _run(router.route("q", session_id="s1"))
    assert d.model == "mini"  # default_tier=weak


# ── T4 健康过滤 ──


def test_health_skips_unhealthy_in_pool():
    """熔断中的模型不出现在 primary；恢复后回来。"""
    health = FakeHealth(unhealthy={"pro"})
    router = MemoryAwareRouter(_models(), ["pro", "glm", "mini"], health=health)
    d = _run(router.route("hi"))
    assert d.model == "glm"

    health.unhealthy = set()
    d2 = _run(router.route("hi"))
    assert d2.model == "pro"


def test_health_all_unhealthy_keeps_pool():
    """验收剧本5：池内全 unhealthy → 保留原池，请求不挂。"""
    health = FakeHealth(unhealthy={"pro", "glm", "mini"})
    router = MemoryAwareRouter(_models(), ["pro", "glm", "mini"], health=health)
    d = _run(router.route("hi"))
    assert d.model == "pro"  # 宁试死模型不打死请求


# ── T5b req_model 显式指定（RA7）──


def test_requested_model_exact_match():
    """request 开 + 命中全名 → REQUESTED，跳裁判。"""
    judge = CountingJudge(tier="weak")
    router = MemoryAwareRouter(
        _models(), ["pro", "glm", "mini"],
        filter_strategy=_filter_data(), judge=judge,
        request_strategy=RequestStrategyData(enabled=True),
    )
    d = _run(router.route("q", req_model="glm"))
    assert d.model == "glm"
    assert d.source == RouteSource.REQUESTED
    assert judge.calls == 0


def test_requested_model_display_name_match():
    """展示名（去 provider 前缀）也命中。"""
    models = {
        "openai/glm-5.2": ModelCandidate(model="openai/glm-5.2", tier="medium"),
        "openai/mini": ModelCandidate(model="openai/mini", tier="weak"),
    }
    router = MemoryAwareRouter(
        models, ["openai/mini"],
        request_strategy=RequestStrategyData(enabled=True),
    )
    d = _run(router.route("q", req_model="glm-5.2"))
    assert d.model == "openai/glm-5.2"
    assert d.source == RouteSource.REQUESTED


def test_requested_model_disabled_ignored():
    """request 关（默认）→ req_model 命中也走流水线（回归现状）。"""
    router = MemoryAwareRouter(_models(), ["pro", "glm", "mini"])
    d = _run(router.route("q", req_model="glm"))
    assert d.model == "pro"  # upstream[0]
    assert d.source == RouteSource.UPSTREAM_DEFAULT


def test_requested_model_capability_fastfail():
    """显式指定模型无所需能力 → 快返 422 语义（NoCapableCandidateError）。"""
    router = MemoryAwareRouter(
        _models(), ["pro", "glm", "mini"],
        request_strategy=RequestStrategyData(enabled=True),
    )
    with pytest.raises(NoCapableCandidateError):
        _run(router.route("看图", req_model="mini", requires=["vision"]))


def test_requested_model_overrides_sticky_and_updates_it():
    """显式指定压过 sticky 并更新之。"""
    judge = CountingJudge(tier="strong")
    router = MemoryAwareRouter(
        _models(), ["pro", "glm", "mini"],
        filter_strategy=_filter_data(), judge=judge,
        sticky=StickyStrategyData(enabled=True),
        request_strategy=RequestStrategyData(enabled=True),
    )
    _run(router.route("q", session_id="s1"))  # sticky = pro

    d = _run(router.route("q2", session_id="s1", req_model="mini"))
    assert d.model == "mini"
    assert d.source == RouteSource.REQUESTED

    judge._tier = "weak"
    d3 = _run(router.route("q3", session_id="s1"))
    assert d3.model == "mini"  # sticky 已更新为 mini
    assert d3.source == RouteSource.STICKY


def test_requested_model_unknown_falls_through():
    """req_model 未命中名单（如 Hermes 展示名）→ 走流水线。"""
    router = MemoryAwareRouter(
        _models(), ["pro", "glm", "mini"],
        request_strategy=RequestStrategyData(enabled=True),
    )
    d = _run(router.route("q", req_model="bladex-auto-whatever"))
    assert d.source == RouteSource.UPSTREAM_DEFAULT


# ── T5b req_model per-agent 覆盖（ADR-0013 增补 2026-07-25）──


def test_requested_model_global_off_agent_override_on():
    """全局关 + override {claude-code: True} + agent=claude-code -> REQUESTED。"""
    router = MemoryAwareRouter(
        _models(), ["pro", "glm", "mini"],
        request_strategy=RequestStrategyData(
            enabled=False, agent_overrides={"claude-code": True},
        ),
    )
    d = _run(router.route("q", req_model="glm", agent_id="claude-code"))
    assert d.source == RouteSource.REQUESTED


def test_requested_model_global_off_override_not_matching_agent():
    """全局关 + override {claude-code: True} + agent=unknown -> 走流水线（不生效）。"""
    router = MemoryAwareRouter(
        _models(), ["pro", "glm", "mini"],
        request_strategy=RequestStrategyData(
            enabled=False, agent_overrides={"claude-code": True},
        ),
    )
    d = _run(router.route("q", req_model="glm", agent_id="unknown-agent"))
    assert d.source != RouteSource.REQUESTED


def test_requested_model_global_on_agent_override_off():
    """全局开 + override {hermes: False} + agent=hermes -> 走流水线（被覆盖关闭）。"""
    router = MemoryAwareRouter(
        _models(), ["pro", "glm", "mini"],
        request_strategy=RequestStrategyData(
            enabled=True, agent_overrides={"hermes": False},
        ),
    )
    d = _run(router.route("q", req_model="glm", agent_id="hermes"))
    assert d.source != RouteSource.REQUESTED


def test_requested_model_override_base_fallback():
    """override {hermes: False} + agent=hermes:accept -> base 命中，关闭。"""
    router = MemoryAwareRouter(
        _models(), ["pro", "glm", "mini"],
        request_strategy=RequestStrategyData(
            enabled=True, agent_overrides={"hermes": False},
        ),
    )
    d = _run(router.route("q", req_model="glm", agent_id="hermes:accept"))
    assert d.source != RouteSource.REQUESTED


def test_requested_model_override_exact_beats_base():
    """override {"hermes:accept": True, "hermes": False} + agent=hermes:accept -> 精确优先，放开。"""
    router = MemoryAwareRouter(
        _models(), ["pro", "glm", "mini"],
        request_strategy=RequestStrategyData(
            enabled=False,
            agent_overrides={"hermes:accept": True, "hermes": False},
        ),
    )
    d = _run(router.route("q", req_model="glm", agent_id="hermes:accept"))
    assert d.source == RouteSource.REQUESTED


def test_requested_model_no_override_falls_to_global_off():
    """空 overrides + 全局关 -> 纯全局开关（现状回归）。"""
    router = MemoryAwareRouter(
        _models(), ["pro", "glm", "mini"],
        request_strategy=RequestStrategyData(enabled=False, agent_overrides={}),
    )
    d = _run(router.route("q", req_model="glm", agent_id="claude-code"))
    assert d.source != RouteSource.REQUESTED


def test_requested_model_no_override_falls_to_global_on():
    """空 overrides + 全局开 -> 纯全局开关（现状回归）。"""
    router = MemoryAwareRouter(
        _models(), ["pro", "glm", "mini"],
        request_strategy=RequestStrategyData(enabled=True, agent_overrides={}),
    )
    d = _run(router.route("q", req_model="glm", agent_id="claude-code"))
    assert d.source == RouteSource.REQUESTED


def test_requested_model_override_none_agent_falls_to_global():
    """agent_id=None（unknown）-> 走全局 enabled，覆盖表不生效。"""
    router = MemoryAwareRouter(
        _models(), ["pro", "glm", "mini"],
        request_strategy=RequestStrategyData(
            enabled=False, agent_overrides={"claude-code": True},
        ),
    )
    d = _run(router.route("q", req_model="glm"))
    assert d.source != RouteSource.REQUESTED
