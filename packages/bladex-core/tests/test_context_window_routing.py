"""批 Z · Z2.2（MQ-A34 ①②）：窗口感知路由。

判据 ②：动态选池排除"估算 prompt + 输出下限 > context_window"的模型；**没配窗口的模型不参与
判断、照常入池**。全被排除 ⇒ 保留原池 + `route_context_window_no_fit`（宁试也不打死请求，
与规模 floor / 健康过滤同一条纪律）。粘性沿用同样要放得下；REQUESTED（静态）放不下只告警。
判据 ①（逐字零行为）在 core 侧的形态：`prompt_tokens_est=None` 或窗口=0 ⇒ 任何一处都不参与。
"""
from __future__ import annotations

import logging

import pytest
from bladex_core.routing import (
    MemoryAwareRouter,
    ModelCandidate,
    RequestStrategyData,
    RouteSource,
    StickyStrategyData,
    fits_context_window,
)

SMALL = "openai/small-32k"
BIG = "openai/big-1m"
NOWIN = "openai/no-window"


def _models() -> dict[str, ModelCandidate]:
    return {
        SMALL: ModelCandidate(model=SMALL, tier="medium", capabilities=["text", "code"],
                              context_window=32_000),
        BIG: ModelCandidate(model=BIG, tier="medium", capabilities=["text", "code"],
                            context_window=1_000_000),
        NOWIN: ModelCandidate(model=NOWIN, tier="medium", capabilities=["text", "code"]),
    }


def _router(**kw) -> MemoryAwareRouter:
    defaults = dict(models=_models(), upstream=[SMALL, BIG, NOWIN],
                    request_strategy=RequestStrategyData(enabled=True),
                    sticky=StickyStrategyData(enabled=False))
    defaults.update(kw)
    return MemoryAwareRouter(**defaults)


# ── 谓词本身 ──

def test_predicate_unset_window_or_no_estimate_always_fits():
    c = ModelCandidate(model="x", context_window=0)
    assert fits_context_window(c, 10**9)
    assert fits_context_window(ModelCandidate(model="y", context_window=100), None)


def test_predicate_counts_output_floor():
    c = ModelCandidate(model="x", context_window=1000)
    assert fits_context_window(c, 900, 100)
    assert not fits_context_window(c, 901, 100)


# ── 判据 ②：动态选池 ──

@pytest.mark.asyncio
async def test_dynamic_pool_skips_models_that_cannot_hold_the_prompt():
    r = _router()
    d = await r.route("q", prompt_tokens_est=40_000, output_floor_tokens=1000)
    assert d.model == BIG, d
    # 没配窗口的模型照常入池（在 failover 里），放不下的不在
    names = [c.model for c in d.pool]
    assert names == [BIG, NOWIN], names
    assert d.context_window == 1_000_000


@pytest.mark.asyncio
async def test_no_estimate_is_verbatim_old_behaviour():
    """判据 ①：调用方不给估算 ⇒ 池顺序逐字不变（小窗口模型仍是首选）。"""
    r = _router()
    d = await r.route("q")
    assert d.model == SMALL
    assert [c.model for c in d.pool] == [SMALL, BIG, NOWIN]


@pytest.mark.asyncio
async def test_all_excluded_keeps_pool_and_warns(caplog):
    models = {SMALL: _models()[SMALL]}
    r = _router(models=models, upstream=[SMALL])
    with caplog.at_level(logging.WARNING, logger="bladex_core.routing"):
        d = await r.route("q", prompt_tokens_est=40_000)
    assert d.model == SMALL, "剔空必须保留原池，不能打死请求"
    assert "route_context_window_no_fit" in caplog.text


@pytest.mark.asyncio
async def test_sticky_model_that_no_longer_fits_is_not_reused(caplog):
    r = _router(sticky=StickyStrategyData(enabled=True))
    d1 = await r.route("q", session_id="s1", prompt_tokens_est=1000)
    assert d1.model == SMALL
    with caplog.at_level(logging.INFO, logger="bladex_core.routing"):
        d2 = await r.route("q", session_id="s1", prompt_tokens_est=40_000)
    assert d2.model == BIG, d2
    assert d2.source is not RouteSource.STICKY
    assert "route_sticky_no_fit" in caplog.text


@pytest.mark.asyncio
async def test_requested_model_that_does_not_fit_is_kept_and_warned(caplog):
    """REQUESTED 是静态来源（原则 9）：放不下也不换，只留点名五项的告警。"""
    r = _router()
    with caplog.at_level(logging.WARNING, logger="bladex_core.routing"):
        d = await r.route("q", req_model=SMALL, agent_id="cc", prompt_tokens_est=40_000,
                          output_floor_tokens=512)
    assert d.model == SMALL and d.source is RouteSource.REQUESTED
    for needle in ("route_static_context_window_exceeded", "source=requested", "agent=cc",
                   f"model={SMALL}", "context_window=32000", "prompt_tokens_est=40000",
                   "routing.toml"):
        assert needle in caplog.text, needle


@pytest.mark.asyncio
async def test_aux_pool_also_skips_models_that_cannot_hold_the_prompt():
    models = _models()
    for m in models.values():
        m.tier = "weak"
    r = _router(models=models, aux_tier="weak")
    d = await r.route("q", auxiliary=True, prompt_tokens_est=40_000)
    assert d.source is RouteSource.AUXILIARY and d.model == BIG
