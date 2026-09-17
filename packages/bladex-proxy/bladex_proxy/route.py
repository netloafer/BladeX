"""路由 — 选模型 + 经 Router 网关转发（ADR-0008 §6.7 + Phase 1 路由规格卡）。

Phase 1 = 配置驱动的确定性路由 + 可选 LLM 裁判筛选器。
  - `LLMJudge`：调 judge_model 判一档（weak/medium/strong），no_think + temp=0 + 超时兜底。
  - `build_router`：按 routing.toml schema v2 组装 MemoryAwareRouter（决策逻辑在 bladex_core）。
  - `resolve_route`：路由开启且可用时走 MemoryAwareRouter，否则回落单一上游（= 回归安全）。
  - `call_model`：经 Router 网关转发 + failover（保档保能力）。

本模块只做**决策**；实际发出调用与上游 SDK 的一切细节（全局开关、日志接管、
异常措辞）收在 `router_sdk.py`，这里不直接碰供应商 SDK。

路由不改注入（叠加与模型无关，ADR-0008 §5.2）。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import structlog
from bladex_core.fact import Fact, HardRule
from bladex_core.routing import (
    JudgeResult,
    MemoryAwareRouter,
    ModelCandidate,
    NoCapableCandidateError,
    RouteDecision,
    RoutingError,
    StickyOrigin,
)

from bladex_proxy import router_sdk
from bladex_proxy.config import ModelRoute, ProxyConfig
from bladex_proxy.model_health import ModelHealth

logger = structlog.get_logger()


def resolve_model(requested_model: str, upstream_model: str) -> str:
    """选模型（路由关闭时的回落路径 = M-proxy-1 现状）。"""
    logger.info("route_resolved", model=upstream_model, requested=requested_model or "(none)")
    return upstream_model


# ── T5：LLM 裁判（proxy 侧，经 Router 网关）────────────────────────────────


_JUDGE_SYSTEM = """You are a routing classifier. Given the user's prompt, classify it into exactly one tier:
- weak: simple, routine, or mechanical tasks (greetings, simple lookups, formatting, short translations, single-step queries)
- medium: moderate tasks requiring some reasoning (summarization, explanation, multi-step but well-defined, code review)
- strong: complex tasks requiring deep reasoning (architecture design, system analysis, complex debugging, research, creative writing)

Respond with exactly one word: weak, medium, or strong."""


class LLMJudge:
    """LLM 裁判（Phase 1 T5）：调 judge_model 判一档（weak/medium/strong）。

    硬约束（来自真实流量复盘）：
    - no_think（ARK doubao: extra_body thinking disabled）——开思维链 6s/千 token；
    - temperature=0、max_tokens 极小、超时 5s（由 router 侧 asyncio.wait_for 控制）；
    - 系统提示给三档各一行描述（A/B 证明删描述→质量塌）；
    - 输出解析成档位（不是模型名），解析失败→抛异常（router 兜底 default_tier）。
    """

    def __init__(
        self,
        model: str,
        models: dict[str, ModelCandidate],
        no_think: bool = True,
    ) -> None:
        self._model = model
        cand = models.get(model)
        self._api_base = cand.api_base if cand else ""
        self._api_key = cand.api_key if cand else ""
        self._no_think = no_think

    async def judge(self, prompt: str) -> JudgeResult:
        """调 judge_model 判档；失败抛异常（router 侧兜底 default_tier）。"""
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": 10,
            "stream": False,
        }
        if self._api_base:
            kwargs["api_base"] = self._api_base
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._no_think:
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

        t0 = time.perf_counter()
        response = await router_sdk.acompletion(**kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        text = (response.choices[0].message.content or "").strip().lower()
        usage = getattr(response, "usage", None)
        total_tokens = getattr(usage, "total_tokens", 0) if usage else 0

        for tier in ("weak", "medium", "strong"):
            if tier in text:
                logger.info(
                    "judge_result",
                    tier=tier, elapsed_ms=round(elapsed_ms, 1),
                    total_tokens=total_tokens, raw=text[:60],
                )
                return JudgeResult(tier=tier, reason=text[:80])

        logger.warning(
            "judge_unparseable", raw=text[:80], elapsed_ms=round(elapsed_ms, 1),
        )
        raise RoutingError(f"judge output unparseable: {text[:80]}")


# ── 组装路由 ──────────────────────────────────────────────────────────────────


class _CacheWarmthAdapter:
    """把 `cache_state.SessionCacheRegistry` 适配成 core 的 `CacheWarmth` Protocol。

    core 不认识 proxy 侧的温度表，也不该认识每模型的 `cache_ttl_s`——
    所以查表 + 取该模型 TTL 这两件事都在这层做完（与 ModelHealth 同款分工）。

    `prefix_changed` 这里**不传**：注入侧已经在同一轮把它算过并喂给了
    `note_forward`/装配路径，而粘性释放只需要"距上次成功转发是否超过该模型 TTL"
    这一个判据。少一个入参 = 少一处可能与注入侧读数不一致的地方。
    """

    def __init__(self, models: dict[str, ModelCandidate]) -> None:
        self._models = models

    def is_warm(self, session_id: str, model: str) -> bool:
        from bladex_proxy.cache_state import registry
        cand = self._models.get(model)
        ttl = cand.cache_ttl_s if cand is not None else 0
        return registry().is_warm(session_id, model, ttl)


def build_router(
    config: ProxyConfig,
    health: ModelHealth | None = None,
) -> MemoryAwareRouter | None:
    """按 routing.toml schema v2 组装 MemoryAwareRouter；路由关闭或无可用模型/上游时返回 None。

    全关（策略全 false）= 只走 [upstream].models 顺序 failover（回归安全）。
    health: 熔断状态（T4），选池时跳过 unhealthy 候选。
    """
    if not config.route_enabled:
        return None

    rcfg = config.routing_config
    if not rcfg.has_models() or not rcfg.has_upstream():
        logger.warning(
            "route_disabled_missing_config",
            has_models=rcfg.has_models(), has_upstream=rcfg.has_upstream(),
        )
        return None

    models = rcfg.model_candidates()
    # upstream：过滤掉不可用/未知的模型名
    upstream = [n for n in rcfg.upstream.models if n in models]
    if not upstream:
        # [upstream].models 全未知 → 用所有可用模型兜底
        upstream = list(models.keys())
        logger.warning("route_upstream_all_unknown_fallback", hint="using all available models as upstream")

    agent = rcfg.agent_strategy()
    principal = rcfg.principal_strategy()
    team = rcfg.team_strategy()
    multimodal = rcfg.multimodal_strategy()
    filter_data = rcfg.filter_strategy()

    judge: LLMJudge | None = None
    if filter_data.enabled and rcfg.filter_judge_model:
        judge = LLMJudge(
            model=rcfg.filter_judge_model,
            models=models,
            no_think=rcfg.filter_no_think,
        )

    return MemoryAwareRouter(
        models=models,
        upstream=upstream,
        agent=agent,
        principal=principal,
        team=team,
        multimodal=multimodal,
        filter_strategy=filter_data,
        judge=judge,
        judge_model=rcfg.filter_judge_model,
        aux_tier=config.route_aux_tier,
        scale=rcfg.scale_strategy(),
        sticky=rcfg.sticky_strategy(),
        request_strategy=rcfg.request_strategy(),
        health=health,
        cache_warmth=_CacheWarmthAdapter(models),  # G14/MQ-RT1
    )


def init_router(config: ProxyConfig) -> tuple[MemoryAwareRouter | None, ModelHealth]:
    """lifespan 装配入口（V-A2 剥离）：熔断器的构造归 Router 模块自己。

    server 只拿返回值挂 app.state（`router` / `model_health`），不再 import
    ModelHealth——边界由 `tests/test_router_module_boundary.py` 钉住。
    返回 `(router, health)`；路由关闭时 router 为 None，health 仍构造
    （/ready 的 upstream informational 检查与 admin 面靠它，行为与拆前一致）。
    """
    health = ModelHealth(config.route_cb_cooldown_s)
    return build_router(config, health=health), health


def make_success_hook(
    router: MemoryAwareRouter | None,
    *,
    session_id: str,
    agent_id: str,
    primary_model: str = "",
) -> Callable[[str], None]:
    """给 `call_model` 的 on_success：failover 实际切换后回写会话粘性 + cache 温度。

    （V-A2 从 server._call_hooks 下沉；行为与事件名逐字保持。）

    `primary_model` = 路由决策选出的首选。实际用的模型与它不同 = 这轮发生了
    failover，粘性要标 `origin=FAILOVER`（G14/MQ-RT1）——被动降级和主动选择
    必须可区分，否则一次瞬时抖动会让会话永久漂移在降级模型上。

    cache 温度回写挂在 on_success 上而不是路由决策处：只有**真的转发成功**才算
    cache 被建立，failover 换过的模型才是真实的那个（ADR-0028 E4）。
    `note_forward` 与本模块 `_CacheWarmthAdapter.is_warm` 是同一契约的两半，
    收在同一模块里。
    """

    def _on_used(used_model: str) -> None:
        if router is not None:
            origin = (StickyOrigin.FAILOVER
                      if primary_model and used_model != primary_model
                      else StickyOrigin.ROUTED)
            if origin is StickyOrigin.FAILOVER:
                logger.warning(
                    "route_sticky_failover_acquired",
                    session_id=session_id[:16] if session_id else "",
                    primary=primary_model, used=used_model,
                    agent_id=agent_id,
                    hint="session is now pinned to a fallback model; it is released"
                         " once the upstream prompt cache goes cold or"
                         " strategies.sticky.failover_max_hold_s elapses",
                )
            router.update_sticky(session_id, used_model,
                                 origin=origin, agent_id=agent_id)
        from bladex_proxy.cache_state import registry as _cache_registry
        _cache_registry().note_forward(session_id, used_model)

    return _on_used


async def resolve_route(
    req_model: str,
    config: ProxyConfig,
    router: MemoryAwareRouter | None,
    query: str,
    *,
    requires: list[str] | None = None,
    agent_id: str | None = None,
    principal_id: str | None = None,
    team_ids: list[str] | None = None,
    auxiliary: bool = False,
    session_id: str | None = None,
    context_chars: int | None = None,
    allowed_exposure: str = "public",
    # Phase 2 预留（本卡不实现，签名保持兼容）
    facts: list[Fact] | None = None,
    hard_rules: list[HardRule] | None = None,
) -> tuple[str, ModelRoute, list[ModelRoute]]:
    """选模型 + 路由。路由开启且可用 → MemoryAwareRouter 决策；否则回落单一上游。

    requires: 请求需要的能力（如 ["vision"]），能力维硬约束。
    session_id: 会话标识（T2 粘性 / T3 裁判 LRU）。
    context_chars: 上下文规模估算（T1 规模层）。
    req_model: 客户端显式请求的模型（T5b，strategies.request 开启才生效）。
    返回 (model, route, failover)：failover 为同能力备选列表（空=无备选）。
    路由决策失败 → 降级默认上游，请求不挂。
    """
    if router is not None:
        try:
            decision: RouteDecision = await router.route(
                query, requires=requires, agent_id=agent_id, auxiliary=auxiliary,
                session_id=session_id, context_chars=context_chars,
                req_model=req_model or None, allowed_exposure=allowed_exposure,
                principal_id=principal_id, team_ids=team_ids,
                facts=facts, hard_rules=hard_rules,
            )
            # 🔴 `agent`/`session_id` 进日志（2026-09-16，MQ-V32）：两者**本来就在参数里**，
            # 只是从未落进这一行。后果是 `route_decision` **无法归属**——多 agent 并发时
            # 只能按时间窗切片，而时间窗不区分"谁的车"：T2 格 r-l-1 就因为窗内混进了
            # 另一个 agent 的流量被判作废（那 21 轮的 system prompt、session 都不同）。
            # 与刚性原则 13 层 3 同族：**读数存在 ≠ 读数可归属**。
            logger.info(
                "route_decision",
                source=decision.source.value, tier=decision.tier,
                model=decision.model, reason=decision.reason,
                agent=agent_id or "", session_id=session_id or "",
            )
            primary = ModelRoute(
                model=decision.model,
                api_base=decision.api_base,
                api_key=decision.api_key,
                provider=decision.provider,
                source=decision.source.value,
                tier=decision.tier,
                reason=decision.reason,
            )
            failover = [
                ModelRoute(model=c.model, api_base=c.api_base, api_key=c.api_key,
                           provider=c.provider, tier=c.tier)
                for c in router.failover_candidates(decision)
            ]
            return decision.model, primary, failover
        except NoCapableCandidateError:
            # T4：能力过滤无合格候选 → 不降级，透传给调用方返结构化错误。
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning("route_decision_failed", error=str(e), hint="degrade to default upstream")
    # 回落：路由关闭 / router 不可用 / 决策失败 → 单一上游，无 failover。
    logger.info(
        "route_decision",
        source="fallback_upstream", model=config.upstream_model,
        reason="router inactive or decision failed -> single upstream",
        agent=agent_id or "", session_id=session_id or "",
    )
    fallback = config.model_route
    fallback.source = "fallback_upstream"
    fallback.reason = "router inactive or decision failed -> single upstream"
    return resolve_model(req_model, config.upstream_model), fallback, []


# ── failover 转发 ──────────────────────────────────────────────────────────────


def _is_failover_trigger(exc: Exception) -> bool:
    """是否触发 failover（5xx / 超时 / 429 / 连接错误；不含 4xx）。"""
    status = getattr(exc, "status_code", None)
    if status is not None:
        try:
            code = int(status)
        except (TypeError, ValueError):
            code = None
        if code is not None:
            return code == 429 or code >= 500
    named = router_sdk.retryable_error_types()
    return bool(named) and isinstance(exc, named)


async def _do_call(
    model: str,
    messages: list[dict[str, Any]],
    stream: bool,
    route: ModelRoute | None,
    **kwargs: Any,
) -> Any:
    """单次经 Router 网关转发。"""
    logger.info("route_calling", model=model, stream=stream, msg_count=len(messages))
    call_kwargs: dict[str, Any] = {"model": model, "messages": messages, "stream": stream}
    call_kwargs.update(kwargs)
    # 🔴 MQ-P23（2026-09-09 实测立）：OpenAI 兼容的**流式**响应，只有在请求带
    # `stream_options.include_usage` 时才会补发那个带 usage 的收尾 chunk。
    # 全仓此前**没有任何地方设它** ⇒ `chunk.usage` 恒 falsy ⇒ `extract_usage` 从未
    # 被调用过 ⇒ `response_meta.usage` 在 claude-code / codex 上恒空。
    # （hermes 有 97% 是因为**客户端自己带**——`/v1/messages` 与 `/v1/responses`
    #  的上游请求是 BladeX 自己拼的，我们没加，所以没有。又一次"同一件事只在
    #  一条路径上做对了"。）
    # 落在 `_do_call` 是因为它是**唯一的转发点**，三条入站路径共用。
    # 调用方显式给了就尊重（不覆盖）；`stream_options` 在 supported_params 里，
    # 不支持的上游由 Router 的 drop_params 摘掉，不会 502。
    # 🔴 它是 BladeX vs 裸跑对照的**前置条件**：没有 usage 就没有 output_tokens/s，
    # 而那是仅有的两个协议无关可比项之一（另一个是 TTFB / MQ-P25）。
    if stream and "stream_options" not in call_kwargs:
        call_kwargs["stream_options"] = {"include_usage": True}
    if route:
        if route.api_base:
            call_kwargs["api_base"] = route.api_base
        if route.api_key:
            call_kwargs["api_key"] = route.api_key
    return await router_sdk.acompletion(**call_kwargs)


def _status_code(exc: Exception) -> int | None:
    """从异常提取 HTTP 状态码（无/不可解析 → None）。"""
    status = getattr(exc, "status_code", None)
    if status is None:
        return None
    try:
        return int(status)
    except (TypeError, ValueError):
        return None


async def call_model(
    model: str,
    messages: list[dict[str, Any]],
    stream: bool = True,
    route: ModelRoute | None = None,
    failover: list[ModelRoute] | None = None,
    health: ModelHealth | None = None,
    on_success: Callable[[str], None] | None = None,
    **kwargs: Any,
) -> Any:
    """经 Router 网关转发 + failover（保档保能力）。

    failover 为同能力备选列表（有序）：首选失败且属触发条件（5xx/超时/连接错）
    → 按序切备选；4xx 直接返回；全败返最后一次上游错。最多遍历一轮（≤ N-1 次重试）。

    T4（RA4）修订：
    - 429 只切 **provider 不同** 的备选——同配额池换模型名 = 重试风暴。
      无跨 provider 候选 → 直接抛原错（agent 侧退避），日志
      `route_429_no_cross_provider` 可 grep（= "该配第二 provider"的运营信号）。
    - 触发 failover 类错误 → health.mark_unhealthy（熔断冷却期内选池跳过）；
      成功 → mark_healthy。
    - on_success(实际使用的模型名)：供调用方回写会话粘性（T2，failover 切换后不乒乓回去）。
    """
    attempts: list[tuple[str, ModelRoute | None]] = [(model, route)]
    for fr in failover or []:
        if fr.model != model:
            attempts.append((fr.model, fr))

    last_exc: Exception | None = None
    while attempts:
        m, rt = attempts.pop(0)
        try:
            resp = await _do_call(m, messages, stream, rt, **kwargs)
            if health is not None:
                health.mark_healthy(m)
            if on_success is not None:
                on_success(m)
            return resp
        except Exception as e:
            last_exc = e
            if not _is_failover_trigger(e):
                raise
            if health is not None:
                health.mark_unhealthy(m)
            if not attempts:
                raise
            # T4: 429 → 只保留跨 provider 备选
            if _status_code(e) == 429:
                cur_provider = rt.provider if rt is not None else ""
                cross = [(m2, r2) for m2, r2 in attempts
                         if (r2.provider if r2 is not None else "") != cur_provider]
                if not cross:
                    logger.warning(
                        "route_429_no_cross_provider",
                        model=m, provider=cur_provider or "(ungrouped)",
                        hint="rate-limited and all failover candidates share the same "
                             "provider/quota; passing 429 through. Consider adding a "
                             "second provider to routing.toml",
                    )
                    raise
                attempts = cross
            logger.warning(
                "route_failover", from_model=m, to_model=attempts[0][0],
                error=router_sdk.error_text(e),
            )
    assert last_exc is not None
    raise last_exc
