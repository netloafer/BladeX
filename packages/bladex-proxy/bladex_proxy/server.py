"""FastAPI proxy 核心 — 串起全流程（T8，ADR-0008 §6.2）。

/v1/chat/completions: 验 key → 认身份 → 注入 → StreamingResponse(转发+捕获) → 异步入库

ISSUE-3: 入库在 try/finally 里执行，客户端断开也尽量存。
ISSUE-5: 上游出错时构造 Turn(status=FAILED) 入库 + 返回错误体。
ISSUE-6: 收紧类型注解。
ISSUE-8: 用 app.state 替代模块级全局变量。
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

import structlog
from bladex_core.fact import Fact
from bladex_core.matter import EdgeTargetType, Matter, MatterOrigin, MatterStatus
from bladex_core.task_unit import TaskUnit, build_task_units, derive_turn_metadata
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse, Response, StreamingResponse

from bladex_proxy import __version__, router_sdk
from bladex_proxy.admin_read import register_admin_read_routes
from bladex_proxy.anthropic import (
    anthropic_stream_generator,
    approx_count_tokens,
    format_anthropic_response,
    parse_anthropic_request,
)
from bladex_proxy.assembly import AssemblyConfig, ContextAssembler, estimate_context_chars
from bladex_proxy.capture import CaptureResult, capture_stream, output_truncated
from bladex_proxy.config import ModelRoute, ProxyConfig
from bladex_proxy.embedding import EmbedCallLog, build_embedder, validate_embed_sensitivity
from bladex_proxy.identity import (
    AUX_REASON_ISOLATED,
    AUX_REASON_NO_TOOLS,
    AUX_REASON_SOURCE,
    AUX_REASON_SUBAGENT,
    LEDGER_ONLY_AUX_REASONS,
    LOCAL_USER_ID,
    _SYSTEM_ROLES,
    is_cheap_tier_auxiliary,
    brings_no_tools,
    is_ledgerless_auxiliary,
    is_isolated_subcall,
    resolve_identity,
)
from bladex_proxy.inject import (
    InjectionSource,
    detect_required_capabilities,  # noqa: E502
    do_inject_async,
)
from bladex_proxy import metrics as metrics_mod
from bladex_proxy.agency import (
    build_agency,
    intercept_anthropic_stream,
    intercept_chat_stream,
    intercept_responses_stream,
)
from bladex_proxy.metrics import Metrics
from bladex_proxy.modules import module_enabled, validate_modules
from bladex_proxy.models import (
    AdminEventType,
    AgentClaimBody,
    AgentRuleBody,
    AgentSource,
    AssignMatterBody,
    ChatCompletionRequest,
    CreateMatterBody,
    DecisionMeta,
    DetachEdgeBody,
    Identity,
    MergeMattersBody,
    PromoteScopeBody,
    ReconstructionRecord,
    RememberFactBody,
    RenameMatterBody,
    RequestParams,
    ResponseMeta,
    SplitMatterBody,
    TombstoneSource,
    TombstoneTargetType,
    ToolEvent,
    Turn,
    TurnStatus,
)
from bladex_proxy.responses import (
    format_responses_response,
    parse_responses_request,
    responses_stream_generator,
    tool_events_from_output,
)
from bladex_proxy.route import (
    NoCapableCandidateError,
    call_model,
    init_router,
    make_success_hook,
    resolve_route,
)
from bladex_proxy.storage.pipeline_redis import DiskSpill, PipelineRedis
from bladex_proxy.storage.memory_index import MemoryIndex, build_agent_claim_map
from bladex_proxy.storage.memory_hub import MemoryHub
from bladex_proxy.storage.pipeline_worker import PipelineWorker

logger = structlog.get_logger()


# T8 安全基线：回环地址白名单（绑这些地址时无需 auth 警告）
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def warn_if_insecure_bind(cfg: ProxyConfig, *, stream: Any = None) -> bool:
    """T8 安全基线：绑非回环且未开 auth -> 强警告（横幅 + structlog warning）。

    /v1/* 在 auth_enabled=false 时无鉴权放行；/metrics 与 /health 不带鉴权、暴露运行数据。
    绑非回环地址意味着任何能到达该地址者都可无鉴权调用并读取运行数据。启动时高可见横幅
    写 stream（默认 sys.stderr）+ structlog warning（可 grep）。返回 True 表示触发了警告。
    """
    if cfg.host in _LOOPBACK_HOSTS or cfg.auth_enabled:
        return False
    banner = (
        "\n"
        "============================================================\n"
        "  BLADEX SECURITY WARNING -- INSECURE BIND\n"
        f"  Listening on {cfg.host}:{cfg.port} with BLADEX_AUTH_ENABLED=false.\n"
        "  /v1/* and /admin/* are OPEN (no key check -- admin write ops like\n"
        "  turn tombstone / Matter assign-merge / scope promote are unprotected);\n"
        "  /metrics and /health expose operational data to anyone reachable.\n"
        "  Fix one of:\n"
        "    - bind 127.0.0.1 (BLADEX_HOST=127.0.0.1, the default)\n"
        "    - set BLADEX_AUTH_ENABLED=true + BLADEX_CLIENT_KEYS\n"
        "    - front with a TLS reverse proxy that enforces auth\n"
        "============================================================\n"
    )
    out = stream if stream is not None else sys.stderr
    out.write(banner)
    try:
        out.flush()
    except Exception:
        pass
    logger.warning(
        "bind_non_loopback_no_auth",
        host=cfg.host, port=cfg.port,
        hint="set BLADEX_AUTH_ENABLED=true + BLADEX_CLIENT_KEYS, or bind 127.0.0.1, or use a TLS reverse proxy",
    )
    return True


def enforce_admin_key_policy(cfg: ProxyConfig) -> bool:
    """ADR-0027 §2.2：管理面 key 分级的启动校验。

    - 企业形态（存在 identity.toml）未配 BLADEX_ADMIN_KEYS -> ValueError 拒启动。
      理由：多 principal 下共用数据面 key = 一个员工的 agent key 拥有全库管理权
      （turn 墓碑 / Matter merge/split / scope promote），可见性拦得住读、拦不住写。
      报错直指配置（刚性原则 9）。
    - 个人模式未配 -> 回落共用 client key（逐字零回归）+ warning（可 grep）。

    返回 True 表示发出了回落 warning。
    """
    problem = cfg.validate_admin_key_policy()
    if problem is not None:
        logger.error("admin_key_policy_violation", reason=problem)
        raise ValueError(problem)
    if cfg.auth_enabled and not cfg.admin_keys_configured:
        logger.warning(
            "admin_key_shared_with_data_plane",
            hint="BLADEX_ADMIN_KEYS not set -- /admin/* accepts any BLADEX_CLIENT_KEYS entry; "
                 "any agent key can run destructive admin operations. "
                 "Set BLADEX_ADMIN_KEYS to separate the management plane.",
        )
        return True
    return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动/关闭：连 Redis、开 RocksDB、起 pipeline worker。"""
    cfg: ProxyConfig = app.state.config
    # V-A1（ADR-0032 §2.1）：模块注册表校验——misconfiguration fails loud，
    # 在打开任何资源之前拒启动（与 identity registry 校验同一形态）。
    # 违例：未知 BLADEX_MODULE_* / Router 配开关 / 依赖不满足。
    app.state.modules = validate_modules()
    logger.info("modules_resolved",
                **{f"module_{k}": v for k, v in app.state.modules.items()})
    # InjectionSource 在 Memory Index 打开后才初始化（需要 Memory Index 做真召回）

    # 2026-08-03：uvicorn access log 过滤 /v1/embeddings——共享模型档下
    # consolidator 每个嵌入批次一个 HTTP 请求（重建期实测 ~1000 行 access log），
    # 纯噪声；对话/管理端点的 access log 保留。只影响日志，不影响任何行为。
    import logging as _logging

    class _EmbedAccessFilter(_logging.Filter):
        def filter(self, record: _logging.LogRecord) -> bool:  # noqa: A003
            return "/v1/embeddings" not in record.getMessage()

    _al = _logging.getLogger("uvicorn.access")
    if not any(isinstance(f, _EmbedAccessFilter) for f in _al.filters):
        _al.addFilter(_EmbedAccessFilter())

    # 接管上游模型 SDK 的日志（署名统一成 Router、摘掉它自带的重复 handler）。
    # 放在入口点而不是模块级：import 必须无副作用（ADR-0027 §5.4 踩过的坑）。
    router_sdk.install_log_bridge()

    logger.info(
        "server_starting", host=cfg.host, port=cfg.port,
        upstream_model=cfg.upstream_model,
        upstream_api_base=cfg.upstream_api_base or "(router default endpoint)",
        auth_enabled=cfg.auth_enabled, client_keys=len(cfg.key_store.keys),
    )
    # T8 安全基线：非回环 + 无 auth -> 强警告（横幅 + structlog warning）
    warn_if_insecure_bind(cfg)
    # ADR-0027 §2.2 管理面鉴权分级：企业形态未配独立 admin key -> 拒启动；
    # 个人模式未配 -> 回落共用 client key + 可 grep 的 warning（零回归但不静默）。
    enforce_admin_key_policy(cfg)

    # Pipeline Redis（连不上不致命）
    pipeline = PipelineRedis(
        redis_url=cfg.redis_url, stream=cfg.redis_stream,
        group=cfg.redis_group, consumer=cfg.redis_consumer,
        queue_max_len=cfg.queue_max_len,
        overflow_dir=cfg.overflow_dir,
        warn_pct=cfg.pipeline_warn_pct,
        alert_pct=cfg.pipeline_alert_pct,
        critical_pct=cfg.pipeline_critical_pct,
        spill_timeout_ms=cfg.pipeline_spill_timeout_ms,
        heavy_payload_bytes=cfg.pipeline_heavy_payload_bytes,
    )
    # T1: app 级 DiskSpill--_enqueue_turn 的最终 fallback 无条件落盘用。
    # 与 PipelineRedis 内部 DiskSpill 同目录（replay 时统一回灌）。
    app.state.disk_spill = DiskSpill(cfg.overflow_dir)
    try:
        await pipeline.connect()
        app.state.pipeline = pipeline
    except Exception as e:
        logger.warning("pipeline_redis_unavailable", error=str(e),
                       hint="proxy will run without storage; "
                            "lazy recovery will attempt to reconnect on first enqueue")
        app.state.pipeline = None

    # ADR-0021 section 2: identity registry -- load early so validation violations
    # (dup key / dangling ref / team cycle) refuse startup before opening resources.
    identity_registry = cfg.identity_registry
    app.state.identity_registry = identity_registry
    if not identity_registry.empty:
        logger.info(
            "identity_registry_active",
            principals=len(identity_registry.principals),
            teams=len(identity_registry.teams),
        )

    # Memory Hub RocksDB
    hub = MemoryHub(cfg.rocksdb_path)
    app.state.hub = hub

    # ADR-0021 section 2.5: sync legacy_ids declarations to Memory Hub journal (idempotent).
    # Memory Index rebuild reads IDENTITY_MERGE events to map legacy user_id -> principal.
    if not identity_registry.empty:
        try:
            added = identity_registry.sync_legacy_to_journal(hub)
            if added:
                logger.info("identity_legacy_journal_synced", added=added)
        except Exception as e:
            logger.warning("identity_legacy_journal_sync_failed", error=str(e))

    # Pipeline worker
    if app.state.pipeline is not None:
        pipeline_worker = PipelineWorker(app.state.pipeline, hub)
        pipeline_worker.start()
        app.state.pipeline_worker = pipeline_worker
        # 回灌磁盘溢出文件（ADR-0009 §5b 三级降级恢复）
        replayed = await pipeline.replay_overflow()
        if replayed > 0:
            logger.info("pipeline_overflow_replayed_on_startup", count=replayed)
    else:
        logger.warning("pipeline_skipped_no_redis")

    # Memory Index 派生层 — proxy 只持有 MemoryIndex 供未来 M-proxy-3 prefetch 读用。
    # consolidation worker 不在 proxy 进程里跑（重操作：扫 RocksDB + embedding，
    # 会竞争 GIL 和磁盘 IO 影响热路径）。consolidation 由独立进程运行：
    #   .venv/bin/python scripts/run_index_consolidator.py
    try:
        # embedding 后端可选化（2026-07-26）：local（默认）| api | proxy（consolidator-only）
        validate_embed_sensitivity(cfg)  # 敏感层开 + api 档 -> strict 拒启动 / 告警+强制本地
        embedder = build_embedder(cfg, role="proxy")
        index = MemoryIndex(cfg.index_path, embedder=embedder, read_only=True,
                       embed_model_id=getattr(embedder, "model_identity", None),
                       novelty_threshold=cfg.novelty_threshold,
                       entity_aware_novelty=cfg.entity_aware_novelty,
                       entity_overlap_threshold=cfg.entity_overlap_threshold,
                       novelty_topk=cfg.novelty_topk,
                       entity_aware_rerank=cfg.entity_aware_rerank,
                       entity_rerank_alpha=cfg.entity_rerank_alpha,
                       entity_rerank_expand_k=cfg.entity_rerank_expand_k)
        index.open()
        app.state.index = index
        app.state.embedder = embedder  # T25: 共享给 /v1/embeddings 端点
        logger.info("index_initialized", hint="Memory Index open for reads; consolidation runs in separate process")
        # M-proxy-3 T3: InjectionSource 接 Memory Index 真召回
        # retrieval_top_k=20: wider retrieval for CAP summary reuse (ADR-0016 §3.3)
        # 注入条目默认 = core MAX_PREFETCH_K（2026-08-14 基准标定回调 3→15，
        # `BLADEX_INJECT_TOPK` 可调）。E6.4 曾按"重复注入 171 次"砍到 3，
        # LongMemEval 实测代价 −20 点；重复注入由工作集去重（U7）治理。
        # 宽检索仍保 20（供 L2 摘要复用，一次 embedding 不重复）。
        # 2026-08-14 T-C：注入字符预算接 config（`BLADEX_INJECT_MAX_CHARS`，默认
        # 1800 与 core 硬编码同值=零回归）。此前平面路径预算不可调：TOPK=15 ×
        # fact 均长 ~135 字必超 1800 → ③平面被整段静默砍掉 → 基准 0/10。
        app.state.inject_source = InjectionSource(
            hard_rules=cfg.effective_hard_rules, index=index,
            top_k=_inject_top_k(), retrieval_top_k=20,
            max_chars=cfg.inject_max_chars,
        )
        # Warm up embedding model: first embed() call loads the model (~800ms local)
        # or validates API config (api backend, fail-fast).
        # Without this, the first real request would exceed the hotpath budget
        # and fall back to hard-rules-only (no Memory Index facts in injection or CAP summary).
        try:
            embedder.embed(["warmup"])
            logger.info("embed_model_warmed_up",
                        identity=getattr(embedder, "model_identity", "?"))
        except Exception as we:
            logger.warning("embed_warmup_failed", error=str(we),
                           hint="first request may be slow / fall back to hard rules")
    except Exception as e:
        logger.warning("index_init_failed", error=str(e), hint="proxy will run without Memory Index derived layer")
        app.state.inject_source = InjectionSource(hard_rules=cfg.effective_hard_rules)
        app.state.index = None
        app.state.embedder = None  # T25: Memory Index 初始化失败时 embedder 也不可用

    # 上下文装配（ADR-0019，取代 ADR-0016 CAP）：
    # L1 证据降解常开 + L2 单元摘要触发式。env BLADEX_ASSEMBLY_*（兼容旧 BLADEX_CAP_*）。
    assembly_config = AssemblyConfig.from_env()
    app.state.assembler = ContextAssembler(assembly_config)
    # MQ-CA3：**每个影响降解行为的旋钮都要能在日志里读到**。
    # 2026-08-28 止血调了 keep_recent_tool_results，而它此前不在这行里
    # ——唯一控制 open 单元降解的参数读不到，只能从行为斜率反推它有没有生效。
    # 「可配置但不可观测 = 半个旋钮」，与刚性原则 12 同源。
    logger.info("assembly_initialized",
                enabled=assembly_config.enabled,
                budget_chars=assembly_config.budget_chars,
                msg_threshold=assembly_config.msg_threshold,
                evidence_min_chars=assembly_config.evidence_min_chars,
                evidence_excerpt_chars=assembly_config.evidence_excerpt_chars,
                keep_recent_closed=assembly_config.keep_recent_closed_units,
                keep_recent_tool_results=assembly_config.keep_recent_tool_results,
                preserve_units=assembly_config.preserve_units)

    # MQ-A10：agent 注册表缓存（origin_key→agent 绑定 + 待认领桶）跨重启恢复。
    # 🔴 在这里调、不在 import 时调——副作用归入口点（2026-08-04 consolidator
    # 那次的教训）；同时这一调用**武装自动落盘**，所以单测/脚本永远不写用户的 data/。
    # 不做的代价（live 实测）：重启后**第一个**请求若是 agent 的内部子调用，
    # 同源继承无记录可继承 ⇒ 必落 `unknown-<hash8>` 并被永久登记进待认领列表，
    # 哪怕它的 origin_key 与主 agent 完全相同。
    from bladex_proxy.agent_registry import load_cache as _load_agent_cache
    _load_agent_cache()

    # V-P5a（ADR-0032）：AgencyRuntime——工具面/拦截/账本的运行时装配。
    # 四个模块默认关（modules.py），关着时所有接线点第一行短路 = 零行为差异。
    app.state.agency = build_agency(app.state)

    # Phase 1 路由规格卡：组装路由（route_enabled=False 或无配置时为 None → 回落单一上游）
    # T4（RA4）：熔断状态注入（选池跳过 unhealthy 候选）
    app.state.router, app.state.model_health = init_router(cfg)
    _validate_sensitivity_judge(cfg)  # ADR-0021 section 3.2: 敏感层开 + 裁判非本地 -> strict 拒启动
    if app.state.router is not None:
        rcfg = cfg.routing_config
        logger.info(
            "router_enabled",
            models=rcfg.has_models(), upstream=rcfg.has_upstream(),
            agent=rcfg.strategies.agent.enabled,
            multimodal=rcfg.strategies.multimodal.enabled,
            filter=rcfg.strategies.filter.enabled,
        )
    # T8：路由开着但 routing.toml 无可用模型/上游（空转/未正确配置）→ 醒目告警。
    if cfg.route_enabled and (app.state.router is None or not cfg.routing_config.has_models() or not cfg.routing_config.has_upstream()):
        rcfg = cfg.routing_config
        reason = (
            "no available models in routing.toml (check [[models]] + api_key_env)"
            if not rcfg.has_models()
            else "no [upstream].models configured"
            if not rcfg.has_upstream()
            else "router build failed"
        )
        logger.error(
            "ROUTE_ENABLED_BUT_INACTIVE",
            reason=reason, hint="requests fall back to single upstream (no differentiation); "
            "set BLADEX_ROUTE_STRICT=true to refuse startup",
        )
        if cfg.route_strict:
            raise RuntimeError(
                f"BLADEX_ROUTE_STRICT=true but routing inactive: {reason}. "
                "Fix routing.toml / env or set BLADEX_ROUTE_ENABLED=false."
            )

    logger.info("server_ready", listen=f"{cfg.host}:{cfg.port}")
    yield

    if app.state.pipeline_worker:
        await app.state.pipeline_worker.stop()
    # A2: drain 后台入库任务（断开连接派发的 shield task），防进程退出时丢
    bg = getattr(app.state, "bg_tasks", None)
    if bg:
        pending = [t for t in bg if not t.done()]
        if pending:
            logger.info("server_draining_bg_enqueue", count=len(pending))
            await asyncio.gather(*pending, return_exceptions=True)
    if app.state.pipeline:
        await app.state.pipeline.close()
    if app.state.index:
        app.state.index.close()
    if app.state.hub:
        app.state.hub.close()
    logger.info("server_stopped")


def create_app(config: ProxyConfig | None = None) -> FastAPI:
    """创建 FastAPI app。"""
    if config is None:
        config = ProxyConfig()

    app = FastAPI(title="BladeX Proxy", version=__version__, lifespan=lifespan)
    app.state.config = config
    app.state.pipeline: PipelineRedis | None = None
    app.state.hub: MemoryHub | None = None
    app.state.pipeline_worker: PipelineWorker | None = None
    app.state.inject_source: InjectionSource | None = None
    app.state.index: MemoryIndex | None = None
    app.state.identity_registry = None  # ADR-0021: IdentityRegistry | None (lifespan 赋实例)
    app.state.metrics = Metrics()  # ADR-0021 T7: Prometheus 指标
    # （M4-2 读取侧漏斗的 InjectionSource 挂钩已随 2026-09-03 S1 删除：
    #  RECALL/INJECT 两段的唯一埋点 `build_planes` 不存在了。）
    app.state.assembler: ContextAssembler | None = None
    app.state.router = None  # MemoryAwareRouter | None (M-proxy-4 路由)
    app.state.model_health = None  # ModelHealth | None（T4 熔断，lifespan 里赋实例）
    # T1: app 级 DiskSpill 兜底（lifespan 里赋实际实例；此处占位防 AttributeError）
    app.state.disk_spill: DiskSpill = DiskSpill(config.overflow_dir)
    # A2: 后台入库任务引用集（防 shield 派发的 task 被 GC；lifespan 关闭时 drain）
    app.state.bg_tasks: set = set()
    # /v1/embeddings 调用归因（2026-08-09）：首次即报 + 周期汇总，见 EmbedCallLog
    app.state.embed_call_log = EmbedCallLog(
        window_s=float(os.environ.get("BLADEX_EMBED_LOG_WINDOW_S", "60")))

    @app.get("/health")
    async def health() -> dict[str, Any]:
        """T27: liveness 探针 - 免鉴权、轻量、不泄运行数据。"""
        return {"status": "ok"}

    @app.get("/ready")
    async def ready(request: Request) -> JSONResponse:
        """T27: readiness 探针 - 免鉴权，探子系统就绪。

        ready = redis ∧ hub（写路径必需）；index/upstream 为 informational 不 gate。
        只暴露布尔就绪位、不泄运行数据。
        """
        checks: dict[str, bool] = {"redis": False, "hub": False, "index": False, "upstream": False}
        pipeline: PipelineRedis | None = request.app.state.pipeline
        if pipeline is not None:
            try:
                checks["redis"] = await pipeline.ping() if hasattr(pipeline, "ping") else True
            except Exception:
                checks["redis"] = False
        # hub (RocksDB write path)
        hub: MemoryHub | None = request.app.state.hub
        checks["hub"] = hub is not None
        # index (informational)
        index = getattr(request.app.state, "index", None)
        checks["index"] = index is not None
        # upstream (informational - 反映是否有上游全部熔断)
        router = getattr(request.app.state, "router", None)
        if router is not None:
            mh = getattr(request.app.state, "model_health", None)
            if mh is not None and mh.any_unhealthy():
                checks["upstream"] = False
            else:
                checks["upstream"] = True
        else:
            checks["upstream"] = True
        ready = checks["redis"] and checks["hub"]
        return JSONResponse(
            status_code=200 if ready else 503,
            content={"ready": ready, "checks": checks},
        )

    @app.get("/metrics")
    async def metrics_endpoint(request: Request) -> Response:
        """ADR-0021 T7: Prometheus 文本格式指标。

        维度：请求数/延迟分位（by agent/sensitivity）、路由 source 分布、
        Pipeline 队列水位、注入过滤命中数。轻量手写 exposition。
        """
        m: Metrics = request.app.state.metrics
        # Pipeline 队列水位（gauge，实时读）
        pipeline: PipelineRedis | None = request.app.state.pipeline
        if pipeline is not None:
            try:
                depth = await pipeline.backlog_len() if hasattr(pipeline, "backlog_len") else 0
                m.set_gauge("bladex_pipeline_queue_depth", float(depth))
            except Exception:
                pass
        return Response(content=m.render(), media_type="text/plain; version=0.0.4; charset=utf-8")

    @app.get("/v1/models")
    async def list_models(request: Request) -> dict:
        """/v1/models：路由开启时返回模型池全列表（T6/RA 修复卡），否则返回单一上游。

        展示名去 Router provider 前缀（如 openai/glm-5.2 → glm-5.2）。
        与 T5b req_model 显式指定闭环：agent 可见名单 = 可显式指定的名单。
        """
        cfg_local: ProxyConfig = request.app.state.config
        names: list[str] = []
        if cfg_local.route_enabled and cfg_local.routing_config.has_models():
            seen: set[str] = set()
            for m in cfg_local.routing_config.models:
                if not m.available:
                    continue
                display = m.name.split("/", 1)[-1] if "/" in m.name else m.name
                if display not in seen:
                    seen.add(display)
                    names.append(display)
        if not names:
            model_name = cfg_local.upstream_model
            names = [model_name.split("/", 1)[-1] if "/" in model_name else model_name]
        is_anthropic = _is_anthropic_client(request)
        data = [_build_model_obj(n, is_anthropic) for n in names]
        if is_anthropic:
            return JSONResponse(content={
                "data": data,
                "has_more": False,
                "first_id": names[0] if names else None,
                "last_id": names[-1] if names else None,
            })
        return JSONResponse(content={"object": "list", "data": data})

    @app.get("/v1/models/{model_name:path}")
    async def retrieve_model(request: Request, model_name: str) -> JSONResponse:
        """T25: GET /v1/models/{model} retrieve。

        命中规则（T5b 语义）：展示名精确命中 / 全名命中（去 provider 前缀后等价）。
        无命中返 404 + OpenAI 错误信封。T26: 按 anthropic-version header 分流响应形态。
        """
        cfg_local: ProxyConfig = request.app.state.config
        # 收集所有可用展示名 -> 全名映射
        display_to_full: dict[str, str] = {}
        if cfg_local.route_enabled and cfg_local.routing_config.has_models():
            for m in cfg_local.routing_config.models:
                if not m.available:
                    continue
                full = m.name
                display = full.split("/", 1)[-1] if "/" in full else full
                display_to_full.setdefault(display, full)
        else:
            full = cfg_local.upstream_model
            display = full.split("/", 1)[-1] if "/" in full else full
            display_to_full[display] = full
        # 命中：model_name 是展示名，或 model_name 是全名（去前缀后等价）
        matched_display: str | None = None
        if model_name in display_to_full:
            matched_display = model_name
        else:
            req_display = model_name.split("/", 1)[-1] if "/" in model_name else model_name
            if req_display in display_to_full:
                matched_display = req_display
        if matched_display is None:
            is_anthropic = _is_anthropic_client(request)
            if is_anthropic:
                return JSONResponse(
                    status_code=404,
                    content={"type": "error", "error": {"type": "not_found_error",
                            "message": f"model not found: {model_name}"}},
                )
            return JSONResponse(
                status_code=404,
                content={"error": {"message": f"model not found: {model_name}", "type": "not_found"}},
            )
        is_anthropic = _is_anthropic_client(request)
        return JSONResponse(
            status_code=200,
            content=_build_model_obj(matched_display, is_anthropic),
        )

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(
        request: Request,
        authorization: str | None = Header(None),
        x_agent_id: str | None = Header(None, alias="X-Agent-ID"),
        x_session_id: str | None = Header(None, alias="X-Session-ID"),
        user_agent: str | None = Header(None, alias="User-Agent"),
    ) -> JSONResponse | StreamingResponse:
        cfg: ProxyConfig = request.app.state.config

        client_key = _extract_bearer(authorization)
        ok, reason = cfg.auth_check(client_key, endpoint="/v1/chat/completions")
        if not ok:
            logger.warning("auth_failed", reason=reason)
            return JSONResponse(
                status_code=401,
                content={"error": {"message": f"Invalid API key: {reason}", "type": "authentication_error"}},
            )

        t_start = time.perf_counter()
        body = await request.json()
        req = ChatCompletionRequest(**body)

        # G11.1：完整 header 集合进身份解析（厂商前缀 header 此前被丢在门外）。
        headers = _collect_headers(request, authorization, x_agent_id, x_session_id, user_agent)

        output_modalities = _extract_output_modalities(body)
        # V-A1 T2 编排函数化：身份→敏感度→入站准备→注入→路由 收在 _prepare_round，
        # 工具面/账本块/自我介绍收在 _apply_agency_surfaces（三端点单实现）。
        prep = await _prepare_round(request, req=req, headers=headers, body=body,
                                    output_modalities=output_modalities,
                                    log_route_debug=True)
        if prep.error_response is not None:
            return prep.error_response
        identity, agent_source = prep.identity, prep.agent_source
        req.tools, _toolface_injected = _apply_agency_surfaces(
            request, prep, tools_in=req.tools, stream=bool(req.stream))
        injected_messages = prep.injected_messages

        # T4: 构造请求参数（req.tools 已被增补 ⇒ params 记录实际转发形态）
        request_params = _build_request_params(request, req)
        decision_meta = prep.decision_meta
        model, route, failover = prep.model, prep.route, prep.failover
        injected_text, ms_identity, ms_inject = (
            prep.injected_text, prep.ms_identity, prep.ms_inject)
        allowed_exposure = prep.allowed_exposure

        if req.stream:
            return await _handle_stream(
                request, model, route, failover, injected_messages, req,
                identity, injected_text, ms_identity, ms_inject, t_start,
                agent_source, original_messages=req.messages,
                request_params=request_params, decision_meta=decision_meta,
                allowed_exposure=allowed_exposure,
            )
        else:
            return await _handle_non_stream(
                request, model, route, failover, injected_messages, req,
                identity, injected_text, ms_identity, ms_inject, t_start,
                agent_source, original_messages=req.messages,
                request_params=request_params, decision_meta=decision_meta,
                allowed_exposure=allowed_exposure,
            )

    @app.post("/v1/messages", response_model=None)
    async def anthropic_messages(
        request: Request,
        authorization: str | None = Header(None),
        x_api_key: str | None = Header(None, alias="x-api-key"),
        x_agent_id: str | None = Header(None, alias="X-Agent-ID"),
        x_session_id: str | None = Header(None, alias="X-Session-ID"),
        user_agent: str | None = Header(None, alias="User-Agent"),
    ) -> JSONResponse | StreamingResponse:
        """T4: Anthropic /v1/messages 端点（Claude Code 原生格式）。

        解析 Anthropic 请求 → 归一 → 认身份 / 注入 / 路由 / 捕获 / 存储（复用全流程）。
        """
        cfg: ProxyConfig = request.app.state.config

        # authorization(Bearer) 优先，x-api-key 兜底（Claude Code 走 x-api-key，Anthropic 风格）
        client_key = _extract_bearer(authorization) or x_api_key
        ok, reason = cfg.auth_check(client_key, endpoint="/v1/messages")
        if not ok:
            logger.warning("anthropic_auth_failed", reason=reason)
            return JSONResponse(
                status_code=401,
                content={"type": "error", "error": {"type": "authentication_error",
                        "message": f"Invalid API key: {reason}"}},
            )

        t_start = time.perf_counter()
        body = await request.json()

        # 解析 Anthropic 请求 → 归一成 OpenAI 格式
        messages, extra_kwargs = parse_anthropic_request(body)
        is_stream = body.get("stream", False)

        # G11.1：完整 header 集合进身份解析（厂商前缀 header 此前被丢在门外）。
        headers = _collect_headers(request, authorization, x_agent_id, x_session_id, user_agent)

        # 用归一后的 messages 构造一个临时 ChatCompletionRequest 供身份解析
        from bladex_proxy.models import ChatCompletionRequest
        req = ChatCompletionRequest(
            model=body.get("model", ""),
            messages=messages,
            stream=is_stream,
            **{k: v for k, v in extra_kwargs.items()
               if k in ("temperature", "max_tokens", "tools", "tool_choice", "top_p", "stop")},
        )

        output_modalities = _extract_output_modalities(body)
        # V-A1 T2 编排函数化（与 chat 端点同一实现，见彼处注释）。
        prep = await _prepare_round(request, req=req, headers=headers, body=body,
                                    output_modalities=output_modalities)
        if prep.error_response is not None:
            return prep.error_response
        identity, agent_source = prep.identity, prep.agent_source
        messages = prep.messages

        # T4: /v1/messages 原始请求体按 hash 入 __msg__ 池
        raw_request_ref = _store_raw_request(request, body)
        _tools_now, _tf_injected = _apply_agency_surfaces(
            request, prep, tools_in=extra_kwargs.get("tools"), stream=bool(is_stream))
        if _tf_injected:
            extra_kwargs["tools"] = _tools_now
        # 🔴 request_params 在 augment 之后建、读 extra_kwargs["tools"]——那才是
        # 转发上游的那份（V-R1 取证）。
        request_params = _build_request_params(
            request, req, forwarded_tools=extra_kwargs.get("tools"))
        injected_messages = prep.injected_messages
        decision_meta = prep.decision_meta
        model, route, failover = prep.model, prep.route, prep.failover
        injected_text, ms_identity, ms_inject = (
            prep.injected_text, prep.ms_identity, prep.ms_inject)
        allowed_exposure = prep.allowed_exposure

        if is_stream:
            return await _handle_anthropic_stream(
                request, model, route, failover, injected_messages, extra_kwargs,
                identity, injected_text, ms_identity, ms_inject, t_start,
                agent_source, original_messages=messages,
                request_params=request_params, decision_meta=decision_meta,
                raw_request_ref=raw_request_ref, allowed_exposure=allowed_exposure,
            )
        else:
            return await _handle_anthropic_non_stream(
                request, model, route, failover, injected_messages, extra_kwargs,
                identity, injected_text, ms_identity, ms_inject, t_start,
                agent_source, original_messages=messages,
                request_params=request_params, decision_meta=decision_meta,
                raw_request_ref=raw_request_ref,
                allowed_exposure=allowed_exposure,
            )

    # ── T26: POST /v1/messages/count_tokens（Claude Code 每轮请求前调用）──

    @app.post("/v1/messages/count_tokens", response_model=None)
    async def anthropic_count_tokens(
        request: Request,
        authorization: str | None = Header(None),
        x_api_key: str | None = Header(None, alias="X-API-Key"),
    ) -> JSONResponse:
        """T26: 本地近似 input_tokens 计数，不调上游。

        复用 parse_anthropic_request 归一化（system 折进 messages、tool_result 转换），
        再调 approx_count_tokens（含 CJK 加权、tools 计入）。
        """
        cfg_local: ProxyConfig = request.app.state.config
        client_key = _extract_bearer(authorization) or x_api_key
        ok, reason = cfg_local.auth_check(client_key, endpoint="/v1/messages/count_tokens")
        if not ok:
            logger.warning("anthropic_count_tokens_auth_failed", reason=reason)
            return JSONResponse(
                status_code=401,
                content={"type": "error", "error": {"type": "authentication_error",
                        "message": f"Invalid API key: {reason}"}},
            )
        try:
            body = await request.json()
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={"type": "error", "error": {"type": "invalid_request_error",
                        "message": f"Invalid JSON: {e}"}},
            )
        messages = body.get("messages")
        if not messages:
            return JSONResponse(
                status_code=400,
                content={"type": "error", "error": {"type": "invalid_request_error",
                        "message": "messages: field required"}},
            )
        try:
            norm_messages, extra = parse_anthropic_request(body)
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={"type": "error", "error": {"type": "invalid_request_error",
                        "message": f"Parse failed: {e}"}},
            )
        tools = extra.get("tools") if isinstance(extra, dict) else None
        tokens = approx_count_tokens(norm_messages, tools)
        return JSONResponse(content={"input_tokens": tokens})

    # ── ADR-0023: POST /v1/responses（Codex CLI 原生格式，第三种入站协议）──

    @app.post("/v1/responses", response_model=None)
    async def openai_responses(
        request: Request,
        authorization: str | None = Header(None),
        x_agent_id: str | None = Header(None, alias="X-Agent-ID"),
        x_session_id: str | None = Header(None, alias="X-Session-ID"),
        user_agent: str | None = Header(None, alias="User-Agent"),
    ) -> JSONResponse | StreamingResponse:
        """ADR-0023: OpenAI Responses API 端点（Codex 原生）。

        解析 Responses 请求 -> 归一成 OpenAI chat 格式 -> 认身份 / 注入 / 路由 / 捕获 / 存储
        （复用 /v1/messages 全流程，只换协议转换层）。
        """
        cfg: ProxyConfig = request.app.state.config

        client_key = _extract_bearer(authorization)
        ok, reason = cfg.auth_check(client_key, endpoint="/v1/responses")
        if not ok:
            logger.warning("responses_auth_failed", reason=reason)
            return JSONResponse(
                status_code=401,
                content={"error": {"message": f"Invalid API key: {reason}", "type": "authentication_error"}},
            )

        try:
            body = await request.json()
        except Exception as e:  # noqa: BLE001 - 请求体解析错误要明确返回
            logger.warning("responses_parse_error", error=str(e))
            return JSONResponse(
                status_code=400,
                content={"error": {"message": f"invalid JSON body: {e}", "type": "invalid_request_error"}},
            )

        # T6: 有状态请求防呆。BladeX 是无状态 proxy，每轮从完整 input 认身份 + 注入。
        # Codex 默认 store=false 每轮发完整 input；若客户端带了 previous_response_id，
        # 说明它依赖服务端会话状态，BladeX 无法满足 -> 400 明确告知。
        if body.get("previous_response_id"):
            logger.info(
                "responses_stateful_rejected",
                previous_response_id=body.get("previous_response_id"),
                hint="BladeX is stateless; set store=false and send full input each turn",
            )
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "type": "invalid_request_error",
                        "message": (
                            "BladeX is a stateless proxy and does not support previous_response_id "
                            "(server-side session state). Set store=false and send the full input "
                            "array each turn. See ADR-0023 §2.3."
                        ),
                        "param": "previous_response_id",
                    }
                },
            )

        # T6 观测：store=true 但无 previous_response_id 时放行（BladeX 不做服务端保留，
        # 本轮无状态处理不受影响）；记 info 便于统计 Codex 有状态用法出现频率。
        if body.get("store") is True:
            logger.info(
                "responses_store_requested",
                hint="BladeX is stateless; store=true is a no-op (no server-side retention)",
            )

        t_start = time.perf_counter()

        # 解析 Responses 请求 -> 归一成 OpenAI chat 格式
        messages, extra_kwargs = parse_responses_request(body)
        is_stream = body.get("stream", False)

        # G11.1：完整 header 集合进身份解析（厂商前缀 header 此前被丢在门外）。
        headers = _collect_headers(request, authorization, x_agent_id, x_session_id, user_agent)

        # 用归一后的 messages 构造临时 ChatCompletionRequest 供身份解析
        req = ChatCompletionRequest(
            model=body.get("model", ""),
            messages=messages,
            stream=is_stream,
            **{k: v for k, v in extra_kwargs.items()
               if k in ("temperature", "max_tokens", "tools", "tool_choice", "top_p", "stop")},
        )

        output_modalities = _extract_output_modalities_responses(body)
        # V-A1 T2 编排函数化（与 chat 端点同一实现，见彼处注释）。
        prep = await _prepare_round(request, req=req, headers=headers, body=body,
                                    output_modalities=output_modalities)
        if prep.error_response is not None:
            return prep.error_response
        identity, agent_source = prep.identity, prep.agent_source
        messages = prep.messages

        # 原始请求体按 hash 入 __msg__ 池
        raw_request_ref = _store_raw_request(request, body)
        _tools_now, _tf_injected = _apply_agency_surfaces(
            request, prep, tools_in=extra_kwargs.get("tools"), stream=bool(is_stream))
        if _tf_injected:
            extra_kwargs["tools"] = _tools_now
        # 🔴 request_params 在 augment 之后建、读 extra_kwargs["tools"]——那才是
        # 转发上游的那份（V-R1 取证）。
        request_params = _build_request_params(
            request, req, forwarded_tools=extra_kwargs.get("tools"))
        injected_messages = prep.injected_messages
        decision_meta = prep.decision_meta
        model, route, failover = prep.model, prep.route, prep.failover
        injected_text, ms_identity, ms_inject = (
            prep.injected_text, prep.ms_identity, prep.ms_inject)
        allowed_exposure = prep.allowed_exposure

        if is_stream:
            return await _handle_responses_stream(
                request, model, route, failover, injected_messages, extra_kwargs,
                identity, injected_text, ms_identity, ms_inject, t_start,
                agent_source, original_messages=messages,
                request_params=request_params, decision_meta=decision_meta,
                raw_request_ref=raw_request_ref, allowed_exposure=allowed_exposure,
            )
        else:
            return await _handle_responses_non_stream(
                request, model, route, failover, injected_messages, extra_kwargs,
                identity, injected_text, ms_identity, ms_inject, t_start,
                agent_source, original_messages=messages,
                request_params=request_params, decision_meta=decision_meta,
                raw_request_ref=raw_request_ref,
                allowed_exposure=allowed_exposure,
            )

    # ── T25: POST /v1/embeddings（与 Memory Index 同源 embedder）──

    @app.post("/v1/embeddings", response_model=None)
    async def openai_embeddings(
        request: Request,
        authorization: str | None = Header(None),
        x_api_key: str | None = Header(None, alias="X-API-Key"),
        x_bladex_caller: str | None = Header(None, alias="X-BladeX-Caller"),
        x_bladex_caller_pid: str | None = Header(None, alias="X-BladeX-Caller-PID"),
        x_bladex_caller_proc: str | None = Header(None, alias="X-BladeX-Caller-Proc"),
        x_bladex_purpose: str | None = Header(None, alias="X-BladeX-Embed-Purpose"),
        user_agent: str | None = Header(None, alias="User-Agent"),
    ) -> JSONResponse:
        """T25: OpenAI 兼容 embeddings 端点。

        复用 app.state 的共享 embedder（与 Memory Index 检索同源），支持单串/批量。

        日志（2026-08-09）：auth_ok 回到 debug，改为每次请求一条 `embed_request`——
        带调用方（`X-BladeX-Caller` / pid / 用途）+ 批量 + 字符数 + 耗时。原先靠
        auth_ok INFO 排查，但它只有 key label，共享模型档下 consolidator 与 agent
        对话请求用同一把 key，满屏同一行看不出谁在调 embedding。
        """
        cfg_local: ProxyConfig = request.app.state.config
        client_key = _extract_bearer(authorization) or x_api_key
        ok, reason = cfg_local.auth_check(client_key, quiet=True, endpoint="/v1/embeddings")
        if not ok:
            return JSONResponse(
                status_code=401,
                content={"error": {"message": f"Invalid API key: {reason}", "type": "authentication_error"}},
            )
        try:
            body = await request.json()
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": f"Invalid JSON: {e}", "type": "invalid_request_error"}},
            )
        inp = body.get("input")
        if inp is None or (isinstance(inp, (str, list)) and len(inp) == 0):
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "input: field required", "type": "invalid_request_error"}},
            )
        inputs = [inp] if isinstance(inp, str) else list(inp)
        embedder = request.app.state.embedder
        if embedder is None:
            return JSONResponse(
                status_code=503,
                content={"error": {"message": "embedder not initialized", "type": "service_unavailable"}},
            )
        # embedding 后端可选化：input_type 是 BladeX 增量扩展（方案 D 共享模型用）——
        # "query"/"passage" 分别走 embed_query/embed_passage（保证与本端检索同前缀）；
        # 缺省 passage = 历史行为（与 Memory Index 同源）。
        input_type = body.get("input_type", "passage")
        from bladex_core.consolidation_proxy import embed_passage_compat, embed_query_compat
        t_embed = time.perf_counter()
        if input_type == "query":
            vectors = embed_query_compat(embedder, inputs)
        else:
            vectors = embed_passage_compat(embedder, inputs)
        embed_ms = round((time.perf_counter() - t_embed) * 1000, 1)
        # 调用方归因：优先显式 header（ProxyEmbedAdapter 恒发），退 User-Agent，
        # 再退 unknown —— "unknown" 本身是信号：有人绕过 adapter 直连这个端点。
        request.app.state.embed_call_log.record(
            caller=x_bladex_caller or (user_agent or "unknown"),
            caller_pid=x_bladex_caller_pid or "-",
            caller_proc=x_bladex_caller_proc or "-",
            purpose=x_bladex_purpose or "-",
            key_label=reason,
            input_type=input_type,
            count=len(inputs),
            chars=sum(len(t) for t in inputs if isinstance(t, str)),
            embed_ms=embed_ms,
        )
        data = [
            {"object": "embedding", "index": i, "embedding": list(vectors[i])}
            for i in range(len(inputs))
        ]
        model_name = getattr(embedder, "model_name", "bladex-embed")
        return JSONResponse(content={
            "object": "list", "data": data, "model": model_name,
            # 方案 D：向量空间身份透传（ProxyEmbedAdapter 用它对齐 Memory Index model_id 不变量）
            "bladex_model_identity": getattr(embedder, "model_identity", f"local:{model_name}"),
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        })

    # ── ADR-0012 §3.6: 带外管理端点（删除 = 写墓碑）──

    @app.delete("/admin/turns/{turn_key:path}")
    async def delete_turn(
        request: Request,
        turn_key: str,
        cfg: ProxyConfig = _admin_key_dep,
    ) -> JSONResponse:
        """删除一个 turn：只写墓碑 + 触发 Memory Index 级联清除（ADR-0012 §3.6）。

        不同步物理擦除（compaction 周期留实测）。
        """
        hub: MemoryHub | None = request.app.state.hub
        if hub is None:
            return JSONResponse(
                status_code=503,
                content={"error": {"message": "Memory Hub storage unavailable",
                        "type": "storage_error"}},
            )

        turn = hub.get(turn_key)
        if turn is None:
            return JSONResponse(
                status_code=404,
                content={"error": {"message": "Turn not found",
                        "type": "not_found"}},
            )

        tombstone_key = hub.append_tombstone(
            target_type=TombstoneTargetType.TURN,
            target_key=turn_key,
            source=TombstoneSource.USER,
        )

        index: MemoryIndex | None = request.app.state.index
        if index is not None:
            try:
                index.cascade_delete_turn(turn_key)
            except Exception as e:
                logger.warning("index_cascade_delete_failed", error=str(e),
                               turn_key=turn_key, hint="tombstone written, Memory Index cleanup deferred")

        logger.info("admin_turn_deleted", turn_key=turn_key, tombstone_key=tombstone_key)
        return JSONResponse(content={
            "status": "deleted",
            "turn_key": turn_key,
            "tombstone_key": tombstone_key,
        })

    @app.post("/admin/facts")
    async def remember_fact(
        body: RememberFactBody,
        request: Request,
        cfg: ProxyConfig = _admin_key_dep,
    ) -> JSONResponse:
        """落一条 manual fact（Beta T17 MCP `remember`）。

        - trust=1.0 + tags=origin:manual（manual 压 auto 语义由 id 确定性保证）
        - 确定性 id（sha("manual"+content)）→ 同文本幂等
        - 以 FACT_IMPORT 管理事件入 Memory Hub journal → full rebuild 后不丢失
        - scope 越权（team/org 但无组织层）→ 400
        """
        content = (body.content or "").strip()
        if not content:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "content: field required",
                        "type": "validation_error"}},
            )
        scope = (body.scope or "").strip()
        if scope and not scope.startswith("personal"):
            if not (scope.startswith("team:") or scope.startswith("org:")):
                return JSONResponse(
                    status_code=400,
                    content={"error": {"message":
                            f"invalid scope {scope!r}: use personal / team:<id> / org:<id>",
                            "type": "validation_error"}},
                )
            registry = cfg.identity_registry
            if getattr(registry, "empty", True):
                return JSONResponse(
                    status_code=400,
                    content={"error": {"message":
                            f"scope {scope!r} requires an organization layer "
                            "(config/identity.toml); personal mode only supports personal scope",
                            "type": "scope_error"}},
                )
        hub: MemoryHub | None = request.app.state.hub
        if hub is None:
            return _hub_unavailable()

        from bladex_core.consolidation_proxy import (
            _deterministic_fact_id,
            embed_passage_compat,
        )
        fact_id = _deterministic_fact_id("manual", content)
        fact = Fact(
            id=fact_id, content=content, trust=1.0, tags="origin:manual",
            kind="general", source_user_id=(body.user_id or "manual"),
            scope=scope,
        )
        embedder = getattr(request.app.state, "embedder", None)
        if embedder is not None:
            try:
                fact.embedding = embed_passage_compat(embedder, [content])[0]
            except Exception as e:  # noqa: BLE001 —— 无向量仍入 meta
                logger.debug("remember_embed_skip", error=str(e))

        # 幂等：同文本重复 remember 不重复 journal
        p2r: MemoryIndex | None = request.app.state.index
        already = False
        if p2r is not None:
            try:
                already = p2r.get_fact(fact_id) is not None
            except Exception:  # noqa: BLE001
                already = False
        if not already:
            hub.append_admin_event(
                AdminEventType.FACT_IMPORT, "", fact_id,
                entity=fact.model_dump(mode="json", exclude={"embedding"}))

        applied, _ = _apply_to_index(cfg, lambda p2w: p2w.add_fact(fact) or True)
        logger.info("admin_fact_remembered", fact_id=fact_id, scope=scope,
                    index_applied=applied, already_existed=already)
        return _admin_response(
            200 if applied else 202,
            ("exists" if already else "remembered") if applied else "deferred",
            fact_id=fact_id,
        )

    @app.delete("/admin/facts/{fact_id}")
    async def delete_fact(
        request: Request,
        fact_id: str,
        cfg: ProxyConfig = _admin_key_dep,
    ) -> JSONResponse:
        """删除单条 Fact（Beta T11 `bladex memory forget`）：先写 Memory Hub FACT 墓碑，
        再 Memory Index 级联（meta + 向量 + 归属边）。rebuild 侧按墓碑过滤不复活。"""
        hub: MemoryHub | None = request.app.state.hub
        if hub is None:
            return _hub_unavailable()

        # 存在性检查走进程内 Memory Index 读句柄（可用时）；不可用不阻断——墓碑语义幂等
        p2r: MemoryIndex | None = request.app.state.index
        if p2r is not None:
            try:
                if p2r.get_fact(fact_id) is None:
                    return JSONResponse(
                        status_code=404,
                        content={"error": {"message": "Fact not found",
                                "type": "not_found"}},
                    )
            except Exception as e:  # noqa: BLE001
                logger.debug("admin_fact_exist_check_failed", error=str(e))

        tombstone_key = hub.append_tombstone(
            target_type=TombstoneTargetType.FACT,
            target_key=fact_id,
            source=TombstoneSource.USER,
        )
        applied, existed = _apply_to_index(cfg, lambda p2w: p2w.delete_fact(fact_id))
        logger.info("admin_fact_deleted", fact_id=fact_id,
                    tombstone_key=tombstone_key, index_applied=applied,
                    existed=bool(existed))
        return _admin_response(
            200 if applied else 202,
            "deleted" if applied else "deferred",
            fact_id=fact_id, tombstone_key=tombstone_key,
        )

    # ── ADR-0012 §3.5: Matter 手动映射操作 ──

    @app.post("/admin/matters")
    async def create_matter(
        body: CreateMatterBody,
        request: Request,
        cfg: ProxyConfig = _admin_key_dep,
    ) -> JSONResponse:
        """手动创建 Matter（先写管理事件到 Memory Hub，再应用到 Memory Index）。"""
        title = body.title
        summary = body.summary
        matter_id = body.matter_id or f"m-{uuid.uuid4().hex[:10]}"

        hub: MemoryHub | None = request.app.state.hub
        if hub is None:
            return _hub_unavailable()

        hub.append_admin_event(
            AdminEventType.MATTER_CREATE, matter_id,
            title=title, summary=summary,
        )

        matter = Matter(
            matter_id=matter_id, title=title, summary=summary,
            status=MatterStatus.ACTIVE, origin=MatterOrigin.MANUAL,
        )
        applied, _ = _apply_to_index(cfg, lambda p2w: p2w.add_matter(matter))

        logger.info("admin_matter_created", matter_id=matter_id, title=title, index_applied=applied)
        return _admin_response(
            200 if applied else 202,
            "created" if applied else "deferred",
            matter_id=matter_id, title=title,
        )

    @app.post("/admin/matters/{matter_id}/assign")
    async def assign_to_matter(
        body: AssignMatterBody,
        matter_id: str,
        request: Request,
        cfg: ProxyConfig = _admin_key_dep,
    ) -> JSONResponse:
        """手动指定归属（manual 压过 auto）。"""
        hub: MemoryHub | None = request.app.state.hub
        if hub is None:
            return _hub_unavailable()

        hub.append_admin_event(
            AdminEventType.MATTER_ASSIGN, matter_id, body.target_key,
            target_type=body.target_type,
        )

        et = EdgeTargetType.SESSION if body.target_type == "session" else EdgeTargetType.FACT
        applied, _ = _apply_to_index(cfg, lambda p2w: p2w.assign_manual(matter_id, et, body.target_key))

        logger.info("admin_matter_assigned", matter_id=matter_id,
                     target_key=body.target_key, index_applied=applied)
        return _admin_response(
            200 if applied else 202,
            "assigned" if applied else "deferred",
            matter_id=matter_id, target_key=body.target_key,
        )

    @app.post("/admin/matters/{matter_id}/merge")
    async def merge_matters(
        body: MergeMattersBody,
        matter_id: str,
        request: Request,
        cfg: ProxyConfig = _admin_key_dep,
    ) -> JSONResponse:
        """合并 Matter（source 合入 target=matter_id，保留 manual 边）。"""
        source_id = body.source_matter_id

        hub: MemoryHub | None = request.app.state.hub
        if hub is None:
            return _hub_unavailable()

        hub.append_admin_event(
            AdminEventType.MATTER_MERGE, matter_id, source_id,
            source_matter_id=source_id,
        )

        applied, moved = _apply_to_index(cfg, lambda p2w: p2w.merge_matters(source_id, matter_id))
        moved = moved or 0

        logger.info("admin_matter_merged", source=source_id, target=matter_id,
                     moved=moved, index_applied=applied)
        return _admin_response(
            200 if applied else 202,
            "merged" if applied else "deferred",
            source_matter_id=source_id, target_matter_id=matter_id, moved_edges=moved,
        )

    @app.get("/admin/agents")
    async def list_agents(
        request: Request,
        cfg: ProxyConfig = _admin_key_dep,
    ) -> JSONResponse:
        """列出见过的 agent（含未识别的 `unknown-<hash>` 桶）+ 认领历史。

        供 dashboard 的 agent 管理页使用（G11.11）。未识别桶带上分桶判据与脱敏
        header 快照——那是用户判断"这是什么 agent"的全部依据。

        **两个来源合并，缺一不可**：

        ==========  ================================================================
        Memory Hub   **持久**。`scan_agent_ids()` 只解 key（agent_id 本就是 key 第二段），
                    重启可存活。没有它，管理页只在"本次启动以来恰好来过"时才有东西
                    ——大多数时候是空的，等于没有管理接口；已经发生过的误署名
                    （08-18 那批）更是永远无从处理。
        进程内登记  **新鲜**。带本次启动以来的轮次计数；Memory Hub 落盘有异步延迟，
                    刚接进来的第一轮可能还没入库，靠它先亮出来。
        ==========  ================================================================

        两处都过一遍 `AGENT_CLAIM` 映射：已认领的桶不再出现在待认领列表里
        （认领是 journal 里的事实，不依赖进程内状态是否还记得）。
        """
        from bladex_proxy.agent_registry import _agent_registry
        from bladex_proxy.agent_rules import load_agent_rules

        # 🔴 `known_agents` 不能只读规则库。认领时不填关键词（关键词是**可选**的）
        # 就不会写规则，于是那个 agent 认领完既不在待认领列表（已认领被滤掉）、
        # 也不在已知列表（没进规则库）——**从界面上彻底消失**。
        # 实际发生：Jason 把 unknown-3b5f685d 认领成 `Pi`（rule_path=None），
        # 随即问"Known agents 怎么看不到"。
        #
        # 正确口径是三个来源的并集，且**标明各自来源**——因为它们的含义不同：
        #   rule    规则库里有 → 下次自动识别
        #   claimed 认领过 → 历史归属已改，但**没有规则就不会自动识别**
        #   seen    Memory Hub 里出现过 → 只是来过
        rule_agents = {r.agent_id for r in load_agent_rules()}
        # 声明了子代理 header 的 agent —— 它们的 `base:suffix` 变体是**内部角色**，
        # 不是用户配置文件，界面上不列（见下方 profiles 那段的立论）。
        _subagent_bases = {r.agent_id for r in load_agent_rules() if r.subagent_headers}

        def _meta(agent_id: str) -> dict[str, Any]:
            return known_meta.setdefault(agent_id, {
                "agent_id": agent_id, "has_rule": agent_id in rule_agents,
                "claimed": False, "turns": 0, "profiles": [],
            })

        known_meta: dict[str, dict[str, Any]] = {}
        for a in rule_agents:
            _meta(a)
        seen = _agent_registry.get_user_agents(LOCAL_USER_ID)
        # 🔴 未识别桶**不在** `get_user_agents()` 里：它读 `_records`，而 FALLBACK
        # 来源有意不 register（register 了就会成为同源继承的目标，把传输层同形的
        # 两个陌生 agent 合到一起）。待认领登记是另一处只读结构——列得出来、粘不上。
        # 第一版这里写的是 `[a for a in seen if a.startswith("unknown-")]`，恒空，
        # 整个认领 UX 不可达（"机制写完接线断"的又一例），而当时的测试只断言了
        # known_agents 与 claims，恰好绕开了坏掉的那一格。
        buckets: dict[str, dict[str, Any]] = {}
        # ① 进程内登记（新鲜）。不限身份：没带 API Key 的陌生客户端落
        #    ANONYMOUS_USER_ID，只列 LOCAL 会让它在界面上隐身，而那正是最该看见的。
        for s in _agent_registry.pending_claims():
            buckets[s.bucket_id] = {
                "bucket_id": s.bucket_id, "basis": s.basis,
                "headers": s.headers, "count": s.count,
                # 统一形状：两个来源返回同一组字段，前端不必分支判断
                # （字段可选 = 契约测不住 = 改名后页面静默渲染成空）
                "hub_turns": 0, "source": "live",
                # ③④（2026-08-26）：漂移证据 + 预填认领建议。空 dict = 没线索，
                # 就是个陌生客户端；有值时前端显示"疑似 <agent> 升级"+ 证据。
                "drift": dict(s.drift or {}),
            }

        claims: list[dict[str, Any]] = []
        claimed_map: dict[str, str] = {}
        hub = getattr(request.app.state, "hub", None)
        if hub is not None:
            try:
                # 🔴 **复用重建侧的同一份实现**，不另写一遍。
                # 曾经在这里手搓过一份"逐事件 src→to"的映射，**没有链式解析**：
                # 用户把 `Pi` 改名成 `Pi-old` 之后，重建侧算出
                # `unknown-3b5f685d → Pi-old`（对），列表页却仍算成 `→ Pi`，
                # 于是界面显示 Pi 7 轮 / Pi-old 0 轮，而真实归属是 3 / 4。
                # 同一个映射两份实现、一份带链一份不带 —— 正是 MS-17 那类漂移。
                claimed_map = build_agent_claim_map(hub)
                for _k, ev in hub.scan_admin_events():
                    if ev.event_type == AdminEventType.AGENT_CLAIM:
                        to_id = str(ev.payload.get("to_agent_id", "") or "")
                        if to_id:
                            _meta(to_id)["claimed"] = True
                        claims.append({
                            "from": ev.payload.get("from_agent_ids", []),
                            "to": to_id,
                            "merge": bool(ev.payload.get("merge", False)),
                            "ts": ev.ts.isoformat() if ev.ts else "",
                        })
            except Exception as e:  # noqa: BLE001 —— 列表页不因 journal 异常整页失败
                logger.warning("admin_agents_claims_scan_failed", error=str(e))

            # ② Memory Hub 派生（持久，重启可存活）。只解 key，不碰 value。
            try:
                for agent_id, (rep_key, count) in hub.scan_agent_ids().items():
                    if not agent_id.startswith("unknown"):
                        if agent_id not in seen:
                            seen.append(agent_id)   # 历史上出现过的具名 agent，供改名
                        # 🔴 profile 变体归并到 base agent。
                        # `hermes:default` / `hermes:accept` / `hermes:c7047465`
                        # 是**同一个 agent 的不同配置文件**（ADR-0010 §3.3），
                        # 规则也只挂在 base 上。摊平成六行只会让列表乱，
                        # 而"改 hermes 的规则"本来就该在 base 这一层做。
                        base = agent_id.split(":")[0] or agent_id
                        entry = _meta(base)
                        entry["turns"] += count
                        # 🔴 **子代理不进 profiles 列表**（Jason 2026-08-26）：
                        # `codex:guardian` 与 `hermes:accept` 都是 `base:suffix`，
                        # 但语义不同——后者是**用户选的配置文件**（要看见、要能改规则），
                        # 前者是 **agent 的内部角色**，名字随 agent 版本随时变，
                        # 列出来只是噪声。轮次仍计入 base（它确实是那个 agent 的流量）。
                        #
                        # 判据来自**配置**不是猜：声明了 `subagent_headers` 的 agent
                        # （codex）其 `base:*` 变体是子代理；`profile_aware` 的
                        # （hermes）是配置文件。实测两个标志零重叠，判得干净。
                        if agent_id != base and base not in _subagent_bases:
                            entry["profiles"].append(agent_id)
                        continue
                    # 裸 `unknown`（G11.9 之前的共用桶）也算未识别侧，
                    # 让它进待认领列表而不是混进已知 agent。
                    #
                    # 🔴 已认领的桶：轮次要算到**认领目标**头上。
                    # Memory Hub key 里仍写着 `unknown-xxxx`（认领是读侧映射，
                    # 不重写总账），所以认领目标的 key 数天然是 0。照实显示 0 会
                    # 误导——那 4 轮记忆下次重建就归 `Pi` 了，说它"0 轮"等于告诉
                    # 用户认领没生效。
                    if agent_id in claimed_map:
                        _meta(claimed_map[agent_id])["turns"] += count
                    entry = buckets.get(agent_id)
                    if entry is not None:
                        entry["hub_turns"] = count
                        continue
                    # 判据串与 header 快照只在这一条代表 turn 上取（列表页不付全量代价）
                    basis, headers = "", {}
                    try:
                        turn = hub.get(rep_key)
                        if turn is not None:
                            basis = getattr(turn.identity, "agent_bucket_basis", "") or ""
                            headers = dict(getattr(turn.identity, "request_headers", {}) or {})
                    except Exception as e:  # noqa: BLE001 —— 单条取不到不该毁掉整页
                        logger.debug("admin_agents_rep_turn_failed",
                                     key=rep_key, error=str(e))
                    buckets[agent_id] = {
                        "bucket_id": agent_id, "basis": basis, "headers": headers,
                        "count": 0, "hub_turns": count, "source": "hub",
                        # Memory Hub 这一路没有证据（证据是进程内检测的产物）；
                        # 形状仍要一致——前端不许对可选字段分支。
                        "drift": {},
                    }
            except Exception as e:  # noqa: BLE001
                logger.warning("admin_agents_ledger_scan_failed", error=str(e))

        # 已认领的不再挂在待认领里——认领是 journal 里的事实，
        # 不依赖进程内状态是否还记得（重启后尤其如此）。
        unknown = [b for bid, b in buckets.items() if bid not in claimed_map]
        unknown.sort(key=lambda b: (-(b.get("hub_turns") or b["count"]), b["bucket_id"]))

        return _admin_response(
            200, "ok",
            # 排序把"能自动识别的"顶在前面，历史残留（`a` / `test-agent` 这类）沉底，
            # 但**不隐藏**——数据确实在库里，藏起来比列出来更糟。
            # 被改名映射走的旧名字（`Pi → Pi-old` 里的 `Pi`）不再作为独立条目出现，
            # 否则改完名新旧两行并列，用户不知道哪个是现行的。
            known_agents=sorted(
                (dict(m, profiles=sorted(m["profiles"]))
                 for k, m in known_meta.items() if k not in claimed_map),
                key=lambda m: (not m["has_rule"], not m["claimed"], -m["turns"], m["agent_id"]),
            ),
            seen_agents=seen,
            unknown_buckets=unknown,
            claims=claims,
        )

    @app.post("/admin/agents/reattribute")
    async def reattribute_agents(
        request: Request,
        cfg: ProxyConfig = _admin_key_dep,
    ) -> JSONResponse:
        """⑤ 按 AGENT_CLAIM 把**存量**记忆的归属改过来（2026-08-26）。

        🔴 **必须经控制通道派给 consolidator**：Memory Index 的写权在它那儿
        （proxy 侧是只读 secondary）。这不是绕远路——两个进程同时写 LanceDB
        会锁冲突，而"谁拥有写权"是既有的架构约束（ADR-0009 §7）。

        为什么需要这个端点而不是让用户跑一次重建：`AGENT_CLAIM` 改的是"派生层的
        解释"，解释的更新**不需要重新蒸馏**。走重建等于为了改一个字段把整个派生层
        重算一遍（2.5 小时 + 全库 LLM 费用）；而走定向重放则**静默无效**
        （判重把纠正版当重复丢掉）。所以只剩这一条正确的路。
        """
        # 控制通道按需建（与 `bladex sync` 的 `_try_delegate` 同款）——proxy 侧
        # 常驻一个 Redis 连接只为这一个低频端点不值得。
        control = getattr(request.app.state, "sync_control", None)
        if control is None:
            try:
                import redis

                from bladex_proxy.sync_control import SyncControl
                _rc = redis.Redis.from_url(cfg.redis_url, decode_responses=True,
                                           socket_connect_timeout=2)
                _rc.ping()
                control = SyncControl(_rc)
            except Exception as e:  # noqa: BLE001
                logger.warning("admin_reattribute_no_control", error=str(e))
                return _admin_response(
                    503, "unavailable",
                    detail=f"control channel unavailable ({e}); run "
                           "`bladex-consolidator --reattribute` on the host instead")
        try:
            job_id = control.submit(action="reattribute")
        except Exception as e:  # noqa: BLE001
            logger.warning("admin_reattribute_submit_failed", error=str(e))
            return _admin_response(500, "error", detail=str(e))
        logger.info("admin_reattribute_submitted", job_id=job_id)
        return _admin_response(200, "submitted", job_id=job_id,
                               hint="poll /admin/sync/status for progress")

    @app.get("/admin/agents/{agent_id}/detail")
    async def agent_detail(
        agent_id: str,
        request: Request,
        cfg: ProxyConfig = _admin_key_dep,
    ) -> JSONResponse:
        """一个 agent（含未识别桶）的详情：它到底干了什么，够不够判断这是谁。

        列表页只给 `bucket_id` + 判据串，**多个未识别 agent 摆在一起分不出谁是谁**
        （Jason 2026-08-19 实测反馈）。认领要求用户先认出这是什么客户端，那就得把
        判断依据摆出来。最有辨识度的四样，按经验排序：

        1. **system prompt 开头** —— agent 通常在这里自我介绍，一眼定身份；
        2. **工具名集合** —— 生态特有，`cordis_*`/`skill_manage`/PascalCase 各不相同；
        3. **完整 header 快照** —— UA、厂商前缀、SDK 语言；
        4. **最近几轮的 user 消息摘录** —— 它在被拿来干什么。

        取样有界：key 列表只解 key，只对**最后 N 条**做 `get()`。
        """
        from bladex_proxy.agent_rules import (
            effective_rules_for,
            rule_as_dict,
            suggest_rule,
        )

        hub = getattr(request.app.state, "hub", None)
        if hub is None:
            return _hub_unavailable()

        try:
            keys = hub.keys_for_agent(agent_id)
        except Exception as e:  # noqa: BLE001
            logger.warning("admin_agent_detail_scan_failed", agent_id=agent_id, error=str(e))
            keys = []

        sample_keys = keys[-_AGENT_DETAIL_SAMPLE:] if keys else []
        sessions = {k.split("/")[2] for k in keys if len(k.split("/")) >= 4}

        tools: set[str] = set()
        models: set[str] = set()
        system_excerpt, headers, basis = "", {}, ""
        recent: list[dict[str, str]] = []
        first_ts, last_ts = "", ""

        for key in sample_keys:
            try:
                turn = hub.get(key)
            except Exception as e:  # noqa: BLE001 —— 单条坏了不该毁掉整页
                logger.debug("admin_agent_detail_turn_failed", key=key, error=str(e))
                continue
            if turn is None:
                continue
            ts = turn.ts.isoformat() if turn.ts else ""
            first_ts = first_ts or ts
            last_ts = ts or last_ts
            if turn.model:
                models.add(turn.model)
            headers = dict(getattr(turn.identity, "request_headers", {}) or {}) or headers
            basis = getattr(turn.identity, "agent_bucket_basis", "") or basis
            for ev in turn.tool_events or []:
                name = getattr(ev, "name", "") or ""
                if name:
                    tools.add(name)
            last_user = ""
            for msg in turn.request_messages or []:
                role = msg.get("role", "")
                text = _message_text(msg.get("content"))
                # 🔴 `developer` 与 `system` 同属系统指令角色（Pi 用的是前者）。
                # 只认 system 的话，辨识度最高的那段自我介绍一个字都取不到。
                if role in _SYSTEM_ROLES and not system_excerpt:
                    system_excerpt = text[:_AGENT_DETAIL_TEXT_CHARS]
                elif role == "user" and text:
                    last_user = text
                for tc in msg.get("tool_calls", []) or []:
                    fn = (tc or {}).get("function", {}).get("name", "")
                    if fn:
                        tools.add(fn)
            rp = turn.request_params
            if rp is not None and getattr(rp, "requested_model", ""):
                models.add(rp.requested_model)
            # tools schema 只存 hash，全文在 `__msg__` 内容池里 —— 不取回来，
            # 详情页就看不到工具名（辨识度仅次于 system prompt 的信号）。
            if rp is not None and getattr(rp, "tools_hash", ""):
                try:
                    for tdef in hub.get_tools(rp.tools_hash) or []:
                        name = ((tdef or {}).get("function", {}) or {}).get("name", "")
                        if name:
                            tools.add(name)
                except Exception as e:  # noqa: BLE001
                    logger.debug("admin_agent_detail_tools_failed", error=str(e))
            if last_user:
                recent.append({"ts": ts, "text": last_user[:_AGENT_DETAIL_TEXT_CHARS]})

        return _admin_response(
            200, "ok",
            agent_id=agent_id,
            recognized=not agent_id.startswith("unknown-"),
            basis=basis,
            headers=headers,
            turns=len(keys),
            sessions=len(sessions),
            first_seen=first_ts,
            last_seen=last_ts,
            models=sorted(models),
            tools=sorted(tools),
            system_excerpt=system_excerpt,
            recent_user_messages=list(reversed(recent)),
            # 认领对话直接预填这条：规则该由系统提议、用户改，不该让用户手写
            # （会话内容我们本来就抓到了）。
            # 已有规则：改名/编辑时预填它（而不是从流量重新推），
            # 否则用户一改就丢掉原规则里手工调过的部分。
            current_rules=[rule_as_dict(r) for r in effective_rules_for(agent_id)],
            # 未识别桶不拿 hash 后缀当 agent 名（`3b5f685d` 不是名字）——
            # 留空让用户填，其余维度照常建议。
            suggested_rule=suggest_rule(
                "" if agent_id.startswith("unknown") else agent_id,
                system_excerpt=system_excerpt,
                tools=sorted(tools),
                headers=headers,
            ),
        )

    @app.post("/admin/agents/rules")
    async def save_agent_rule(
        body: AgentRuleBody,
        request: Request,
        cfg: ProxyConfig = _admin_key_dep,
    ) -> JSONResponse:
        """新增 / 修改一条识别规则（写用户规则文件，立即生效）。

        存在理由（Jason 2026-08-19）：「如果想修改已知 agent 的规则，必须去改 toml
        配置文件吗？dashboard 应该可以一并操作呀。」——对。规则外置的意义是用户能参与，
        而"参与"不该等于"手编 TOML"。

        语义：按 `name` **upsert**。写盘走追加（不重写文件、不吃掉用户自己的注释），
        `load_agent_rules` 里"用户表同名取最后一条"保证追加即更新。
        预置规则改不了、也不需要改——写一条同名的用户规则即整体覆盖它。
        """
        from bladex_proxy.agent_rules import append_user_rule, load_agent_rules

        rule = dict(body.rule or {})
        if not rule.get("agent_id"):
            return _admin_response(400, "error", detail="rule.agent_id is required")
        rule.setdefault("name", rule["agent_id"])
        try:
            path = append_user_rule(rule)      # 写盘前会校验，非法直接抛
        except ValueError as e:
            return _admin_response(400, "error", detail=f"Invalid rule: {e}")
        except Exception as e:  # noqa: BLE001
            logger.warning("admin_agent_rule_save_failed", error=str(e))
            return _admin_response(500, "error", detail=f"Could not write rule: {e}")

        effective = [r.agent_id for r in load_agent_rules() if r.name == rule["name"]]
        logger.info("admin_agent_rule_saved", name=rule["name"],
                    agent_id=rule["agent_id"], path=str(path))
        return _admin_response(
            200, "saved", name=rule["name"], agent_id=rule["agent_id"],
            path=str(path), effective_agent_ids=effective,
        )

    @app.post("/admin/agents/claim")
    async def claim_agent(
        body: AgentClaimBody,
        request: Request,
        cfg: ProxyConfig = _admin_key_dep,
    ) -> JSONResponse:
        """认领 / 改名 / 合并 agent —— 一套语义三种用法（G11.11，台账接入域）。

        写 `AGENT_CLAIM` 管理事件（append-only journal），Memory Index 重建时读侧
        重放改归属。**不原地改 Memory Hub**：`agent_id` 是 hub key 的组成部分，
        而 hub 是追加式总账；改的是派生层的解释，不是事实本身。日志里已形成的
        agent_id 同理不追改。

        🔴 误合并红线：`len(from) > 1` 或 `to` 已存在 → 判定为合并，必须
        `confirm_merge=True`。把 dsh 认领成 hermes 就是手动制造跨 agent 记忆污染。
        """
        from bladex_proxy.agent_registry import _agent_registry
        from bladex_proxy.agent_rules import append_user_rule, load_agent_rules

        sources = [s.strip() for s in body.from_agent_ids if s and s.strip()]
        target = (body.to_agent_id or "").strip()
        if not sources or not target:
            return _admin_response(400, "error",
                                   detail="from_agent_ids and to_agent_id are both required")
        if target in sources:
            return _admin_response(400, "error", detail="to_agent_id must not appear in from_agent_ids")

        hub_for_check = getattr(request.app.state, "hub", None)
        existing = {r.agent_id for r in load_agent_rules()}
        existing.update(_agent_registry.get_user_agents(LOCAL_USER_ID))
        # 🔴 历史认领目标也算"已存在"。
        #
        # 少了这一条会出现**判据不一致**：把第二个桶认领到已有的 `Pi` 上，
        # 明明是"两组记忆并进同一个命名空间"，却因为 `Pi` 恰好没有规则、也没在
        # 流量里出现过（流量都解析成 `unknown-*`）而绕过合并确认。
        # 同一个动作，因为一件无关的事（目标有没有规则）走了不同的门。
        # 结果本身没错（Pi 只有一行、轮次累加），但门的判据不该依赖无关因素。
        if hub_for_check is not None:
            try:
                claim_map = build_agent_claim_map(hub_for_check)
                existing.update(claim_map.values())
                # 🔴 但**被改名走的名字要还回去**：`Pi → Pi-old` 之后，`Pi` 这个名字
                # 就空出来了，再把别的桶认领成 `Pi` 不是合并、不该要二次确认。
                # 判据是"它还是不是现行名字"——映射表的 key 都是已被取代的旧名字。
                existing.difference_update(claim_map.keys())
            except Exception as e:  # noqa: BLE001 —— journal 异常不该挡住认领
                logger.warning("admin_agent_claim_history_scan_failed", error=str(e))
        # 合并判据 = 多源 或 目标已存在（两者都意味着两组记忆要并进一个命名空间）
        is_merge = len(sources) > 1 or target in existing
        if is_merge and not body.confirm_merge:
            return _admin_response(
                400, "confirm_required",
                detail=("This is a merge (multiple sources, or the target agent already "
                        "exists): two memory namespaces will be joined. Resend with "
                        "confirm_merge=true if that is intended."),
                from_agent_ids=sources, to_agent_id=target, merge=True,
            )

        hub = hub_for_check
        if hub is None:
            return _hub_unavailable()

        # 🔴 matter_id 留空走 payload —— 该字段只在目标真是 Matter 时使用（MQ-A7）。
        hub.append_admin_event(
            AdminEventType.AGENT_CLAIM, "", "",
            from_agent_ids=sources, to_agent_id=target, merge=is_merge,
            rule=body.rule or None,
        )

        rule_path = ""
        if body.rule:
            try:
                rule_path = str(append_user_rule(body.rule))
            except Exception as e:  # noqa: BLE001 —— 规则写失败不该回滚已记的认领
                # 认领事件已入 journal（归属改得了），只是"下次自动识别"没配上。
                # 报出来而不是静默：否则用户会以为规则生效了。
                logger.warning("admin_agent_claim_rule_write_failed", error=str(e))
                return _admin_response(
                    202, "claimed_rule_failed",
                    detail=f"Claim recorded, but writing the recognition rule failed: {e}",
                    from_agent_ids=sources, to_agent_id=target, merge=is_merge,
                )

        for src in sources:
            _agent_registry.drop_pending(src)

        logger.info("admin_agent_claimed", from_agent_ids=sources,
                    to_agent_id=target, merge=is_merge, rule_path=rule_path or None)
        return _admin_response(
            200, "claimed",
            from_agent_ids=sources, to_agent_id=target, merge=is_merge,
            rule_path=rule_path or None,
            # 🔴 2026-08-26 订正：原文写的是"下一次 Memory Index 重建时改归属"，
            # **对增量重建不成立**——纠正版与库内旧 Fact 内容逐字相同，会被 novelty
            # 判重丢弃（verdict=vector_kill），归属改不成且不报错。只有全量重建
            # （clear_db 关掉判重）才追溯得了。所以这里必须明确指向 ⑤ 那条通路。
            hint=("New traffic is attributed immediately. EXISTING memories are NOT "
                  "re-attributed automatically — run POST /admin/agents/reattribute "
                  "(or `bladex-consolidator --reattribute`). Memory Hub keys and logs "
                  "are append-only and are never rewritten."),
            next_action="reattribute",
        )

    @app.post("/admin/matters/{matter_id}/detach")
    async def detach_from_matter(
        body: DetachEdgeBody,
        matter_id: str,
        request: Request,
        cfg: ProxyConfig = _admin_key_dep,
    ) -> JSONResponse:
        """手动摘除归属边（by edge_id 或 by target_key + target_type）。"""
        if not body.edge_id and not body.target_key:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "edge_id or target_key required",
                        "type": "validation_error"}},
            )

        hub: MemoryHub | None = request.app.state.hub
        if hub is None:
            return _hub_unavailable()

        hub.append_admin_event(
            AdminEventType.MATTER_DETACH, matter_id,
            edge_id=body.edge_id, target_key=body.target_key,
            target_type=body.target_type,
        )

        if body.edge_id:
            applied, removed = _apply_to_index(cfg, lambda p2w: p2w.detach_edge(body.edge_id))
        else:
            et = EdgeTargetType.SESSION if body.target_type == "session" else EdgeTargetType.FACT
            applied, removed = _apply_to_index(cfg, lambda p2w: p2w.detach_edges_for_target(body.target_key, et))
        removed = removed or 0

        logger.info("admin_matter_detached", matter_id=matter_id,
                     edge_id=body.edge_id, removed=removed, index_applied=applied)
        return _admin_response(
            200 if applied else 202,
            "detached" if applied else "deferred",
            matter_id=matter_id, removed=removed,
        )

    @app.post("/admin/matters/{matter_id}/split")
    async def split_matter_endpoint(
        body: SplitMatterBody,
        matter_id: str,
        request: Request,
        cfg: ProxyConfig = _admin_key_dep,
    ) -> JSONResponse:
        """拆分 Matter：创建新 Matter 并把指定归属边转移过去。"""
        hub: MemoryHub | None = request.app.state.hub
        if hub is None:
            return _hub_unavailable()

        new_matter_id = f"m-{uuid.uuid4().hex[:10]}"

        hub.append_admin_event(
            AdminEventType.MATTER_SPLIT, matter_id,
            new_matter_id=new_matter_id, new_title=body.new_title,
            edge_ids=body.edge_ids,
        )

        applied, result = _apply_to_index(cfg, lambda p2w: p2w.split_matter(
            matter_id, new_matter_id, body.new_title, body.edge_ids,
        ))
        moved = result[1] if result is not None else 0

        logger.info("admin_matter_split", source=matter_id,
                     new_matter_id=new_matter_id, moved=moved, index_applied=applied)
        return _admin_response(
            200 if applied else 202,
            "split" if applied else "deferred",
            source_matter_id=matter_id, new_matter_id=new_matter_id, moved_edges=moved,
        )

    @app.post("/admin/matters/{matter_id}/close")
    async def close_matter(
        matter_id: str,
        request: Request,
        cfg: ProxyConfig = _admin_key_dep,
    ) -> JSONResponse:
        """关闭 Matter。"""
        hub: MemoryHub | None = request.app.state.hub
        if hub is None:
            return _hub_unavailable()

        hub.append_admin_event(AdminEventType.MATTER_CLOSE, matter_id)

        applied, success = _apply_to_index(cfg, lambda p2w: p2w.close_matter(matter_id))
        success = success or False

        logger.info("admin_matter_closed", matter_id=matter_id, success=success, index_applied=applied)
        return _admin_response(
            200 if applied else 202,
            "closed" if applied else "deferred",
            matter_id=matter_id, success=success,
        )

    @app.post("/admin/matters/{matter_id}/rename")
    async def rename_matter(
        body: RenameMatterBody,
        matter_id: str,
        request: Request,
        cfg: ProxyConfig = _admin_key_dep,
    ) -> JSONResponse:
        """重命名 Matter。"""
        hub: MemoryHub | None = request.app.state.hub
        if hub is None:
            return _hub_unavailable()

        hub.append_admin_event(AdminEventType.MATTER_RENAME, matter_id, title=body.title)

        applied, success = _apply_to_index(cfg, lambda p2w: p2w.rename_matter(matter_id, body.title))
        success = success or False

        logger.info("admin_matter_renamed", matter_id=matter_id, title=body.title, index_applied=applied)
        return _admin_response(
            200 if applied else 202,
            "renamed" if applied else "deferred",
            matter_id=matter_id, title=body.title, success=success,
        )

    @app.post("/admin/ledgers/user-edit")
    async def admin_ledger_user_edit(
        request: Request,
        cfg: ProxyConfig = _admin_key_dep,
    ) -> JSONResponse:
        """V-F2b（2026-08-29 语义修订，Jason 拍板）：用户直编账本 = **采纳本地版本**。

        不再写 Hub 管理事件——采纳只进 agency 内存池（注入块/工具面立刻可见
        用户版本），Hub 由会话自然收敛：模型下一次 `bladex_ledger_update` 相对
        池中当前（=用户）版本提交，经正常工具路径落 `LEDGER_UPDATE` 事件。
        日志 `admin_ledger_user_edit_adopted` 是追溯锚（谁的版本、何时采纳）。
        proxy 重启会丢内存池采纳——daemon 侧对未收敛的直编限流重发（自愈，有上限）。

        🔴 MQ-L47（批 E E0.1，2026-09-06）：响应体带 **`ledger`（采纳后的池版，
        rev 已 +1）**——daemon 拿它当轮写回投影，"采纳 = 用户版进池 + 池版回投影"
        一步完成；此前 daemon 的收敛判据（渲染逐字 == 磁盘）在 rev 前进那一刻起恒假，
        每轮重发磁盘旧版把模型此后的 `ledger_update` 全盖回去（修前 09-02～05：
        proxy 侧 adopted 177 条 vs daemon 侧 2 条，`changed=['verified']` 11 条里 10 条
        是回滚）。`clobber_risk` 加 `same_as_last_adoption`：payload 与该本上次采纳版
        相等 ⇒ 这是重发不是新直编——**不拒绝**（兼容旧 daemon），只把两种形态分开可 grep。
        """
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return JSONResponse(status_code=400, content={
                "error": {"message": "invalid JSON body", "type": "validation_error"}})
        from bladex_core.ledger import Ledger as _Ledger
        try:
            led = _Ledger.model_validate((body or {}).get("ledger") or {})
        except Exception as e:  # noqa: BLE001 —— 坏 payload 响亮拒绝，不入 Hub
            return JSONResponse(status_code=400, content={
                "error": {"message": f"invalid ledger payload: {e}",
                          "type": "validation_error"}})
        agency = getattr(request.app.state, "agency", None)
        if agency is not None:
            prev = agency.pool.get(led.ledger_id)
            # 上次采纳的用户版（去掉 rev/last_writer——那两项是采纳时 proxy 改写的，
            # 不属于用户提交的内容）；按 ledger_id 记，进程内存即可（只为区分形态）。
            _last_adoptions: dict = getattr(agency, "_last_user_edit_dump", None) or {}
            if not hasattr(agency, "_last_user_edit_dump"):
                try:
                    agency._last_user_edit_dump = _last_adoptions
                except Exception:  # noqa: BLE001 —— 桩对象不可写属性时退化为不记
                    pass
            submitted = led.model_dump(mode="json", exclude={"rev", "last_writer"})
            same_as_last = _last_adoptions.get(led.ledger_id) == submitted
            if prev is not None:
                # 直编采纳=整本覆盖（用户主权），但 rev 必须前进、写者记 user；
                # 若并发 agent 在窗口内写过，盖掉的是它的新条目——响亮告警
                # （Hub 事件全留可恢复），不静默（刚性原则 13 同族）。
                led = led.model_copy(update={"rev": prev.rev + 1,
                                             "last_writer": "user"})
                if prev.last_writer and prev.last_writer != "user":
                    logger.warning("admin_ledger_user_edit_clobber_risk",
                                   ledger=led.ledger_id,
                                   concurrent_writer=prev.last_writer,
                                   prev_rev=prev.rev,
                                   prev_updated=prev.updated_at,
                                   same_as_last_adoption=same_as_last)
            _last_adoptions[led.ledger_id] = submitted
            agency.pool[led.ledger_id] = led
            if prev is None:
                changed = ["<new>"]
            else:
                changed = (["goal"] if prev.goal != led.goal else [])
                changed += sorted(
                    k for k in set(prev.sections) | set(led.sections)
                    if prev.sections.get(k) != led.sections.get(k))
            logger.info("admin_ledger_user_edit_adopted", ledger=led.ledger_id,
                        changed=changed, rev=led.rev,
                        same_as_last_adoption=same_as_last)
        else:
            # agency 不在（模块关/装配失败）：无处采纳，响亮拒绝——daemon 会重试。
            return JSONResponse(status_code=503, content={
                "error": {"message": "agency runtime unavailable; cannot adopt",
                          "type": "service_unavailable"}})
        # `ledger` = 采纳后的池版（MQ-L47：daemon 据此当轮写回投影，收敛不再靠猜）。
        return JSONResponse(content={"status": "ok", "ledger_id": led.ledger_id,
                                     "rev": led.rev,
                                     "same_as_last_adoption": same_as_last,
                                     "ledger": led.model_dump(mode="json")})

    @app.post("/admin/audience/set")
    async def admin_audience_set(
        request: Request,
        cfg: ProxyConfig = _admin_key_dep,
    ) -> JSONResponse:
        """MQ-L27 修法③：人设事实 audience 补标（管理事件，重建可重放）。

        只追加 AUDIENCE_SET 事件（matter_id 留空——MQ-A7 护栏；payload 带
        fact_id/audience）；应用交给 consolidator 的增量管理事件重放——
        Index 单写者纪律。注入面不等它：读侧派生（effective_audience）已兜底。
        """
        hub: MemoryHub | None = request.app.state.hub
        if hub is None:
            return _hub_unavailable()
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return JSONResponse(status_code=400, content={
                "error": {"message": "invalid JSON body", "type": "validation_error"}})
        fact_id = str((body or {}).get("fact_id", "")).strip()
        audience = str((body or {}).get("audience", "")).strip()
        import re as _re_aud
        if not fact_id or not _re_aud.fullmatch(r"all|agent:[\w.-]+(:[\w.-]+)?",
                                                audience):
            return JSONResponse(status_code=400, content={
                "error": {"message": "fact_id and audience "
                          "(all | agent:<base>) required",
                          "type": "validation_error"}})
        hub.append_admin_event(AdminEventType.AUDIENCE_SET, "",
                               fact_id=fact_id, audience=audience)
        logger.info("admin_audience_set", fact_id=fact_id, audience=audience)
        return JSONResponse(content={"status": "ok", "fact_id": fact_id,
                                     "audience": audience})

    @app.post("/admin/scope/promote")
    async def promote_scope(
        body: PromoteScopeBody,
        request: Request,
        cfg: ProxyConfig = _admin_key_dep,
    ) -> JSONResponse:
        """ADR-0021 section 2.4: scope 提升（personal->team/org）--显式管理操作。

        先写 SCOPE_PROMOTE 管理事件到 Memory Hub，再应用到 Memory Index（Fact/Matter.scope）。
        Memory Index 重建重放管理事件，提升不丢失。共享检索 UX 留 E2。
        """
        if not body.target_key or not body.new_scope:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "target_key and new_scope required",
                        "type": "validation_error"}},
            )

        hub: MemoryHub | None = request.app.state.hub
        if hub is None:
            return _hub_unavailable()

        hub.append_admin_event(
            AdminEventType.SCOPE_PROMOTE, body.target_key,
            target_type=body.target_type, new_scope=body.new_scope,
        )

        def _apply(p2w: MemoryIndex) -> bool:
            if body.target_type == "matter":
                m = p2w.get_matter(body.target_key)
                if m is None:
                    return False
                m.scope = body.new_scope
                p2w.update_matter(m)
                return True
            f = p2w.get_fact(body.target_key)
            if f is None:
                return False
            f.scope = body.new_scope
            p2w.add_fact(f)  # re-store metadata
            return True

        applied, success = _apply_to_index(cfg, _apply)
        success = success or False
        logger.info("admin_scope_promoted", target_type=body.target_type,
                     target_key=body.target_key, new_scope=body.new_scope, index_applied=applied)
        return _admin_response(
            200 if applied else 202,
            "promoted" if applied else "deferred",
            target_type=body.target_type, target_key=body.target_key,
            new_scope=body.new_scope, success=success,
        )

    @app.post("/admin/queue/flush")
    async def flush_queue(
        request: Request,
        apply: bool = False,
        max_batches: int = 200,
        sample: int = 0,
        cfg: ProxyConfig = _admin_key_dep,
    ) -> JSONResponse:
        """强制排空 Pipeline 队列（2026-08-06 事故驱动，供 `bladex queue flush`）。

        必须由 proxy 进程执行——Memory Hub 是 RocksDB 单写者，写锁在 proxy 手里，
        CLI 只能只读。所以 CLI 是控制面、这里是执行面（与 sync delegate 同一分工）。

        `apply=false`（默认）只巡检不改动：跑完整的分类与 Hub 核对，报告将要做什么。
        """
        worker = request.app.state.pipeline_worker
        if worker is None:
            return JSONResponse(
                status_code=503,
                content={"error": {"message":
                        "pipeline worker unavailable -- Redis is likely down;"
                        " it comes back on its own once Redis is reachable (ADR-0017)",
                        "type": "unavailable"}},
            )
        try:
            report = await worker.force_drain(apply=apply, max_batches=max_batches,
                                              sample=max(0, min(sample, 20)))
        except Exception as e:  # noqa: BLE001
            logger.warning("admin_queue_flush_failed", error=str(e))
            return JSONResponse(
                status_code=503,
                content={"error": {"message": f"queue flush failed: {e}",
                        "type": "unavailable"}},
            )
        return JSONResponse(content={"status": report["mode"], **report})

    @app.post("/admin/sticky/flush")
    async def flush_sticky(
        request: Request,
        agent_id: str = "",
        cfg: ProxyConfig = _admin_key_dep,
    ) -> JSONResponse:
        """强制释放会话粘性（G14/MQ-RT1，供 `bladex sticky flush`）。

        必须由 proxy 进程执行——粘性表是 `MemoryAwareRouter` 的进程内 LRU，
        CLI 够不到（与 queue flush 同一分工：CLI 是控制面，这里是执行面）。

        `agent_id` 为空 → 全局；否则两级匹配（给 "hermes" 连 "hermes:default"
        一起清，给 "hermes:default" 只清那一个），与 [strategies.agent.map] 同规则。

        自动释放（cache 冷 / 超 max_hold）覆盖不到的场景才需要它：比如刚修好上游、
        不想等 TTL；或 `failover_max_hold_s=0` 显式关了自动释放。
        """
        router = request.app.state.router
        if router is None:
            return JSONResponse(
                status_code=503,
                content={"error": {"message":
                        "routing is disabled (BLADEX_ROUTE_ENABLED=false) --"
                        " there is no sticky table to flush",
                        "type": "unavailable"}},
            )
        target = agent_id.strip() or None
        before = router.sticky_snapshot()
        cleared = router.flush_sticky(target)
        after_ids = {e["session_id"] for e in router.sticky_snapshot()}
        flushed = [e for e in before if e["session_id"] not in after_ids]
        logger.info("admin_sticky_flush", agent_id=target or "(all)", cleared=cleared)
        return JSONResponse(content={
            "status": "ok",
            "agent_id": target,
            "cleared": cleared,
            "remaining": len(after_ids),
            "flushed": flushed,
        })

    @app.get("/admin/sticky")
    async def get_sticky(
        request: Request,
        cfg: ProxyConfig = _admin_key_dep,
    ) -> JSONResponse:
        """当前粘性表（G14/MQ-RT3：让 failover 造成的降级漂移看得见）。

        每条带 `origin`（routed = 主动选出 / failover = 故障被动切换）与
        `held_s`（已持有秒数）——`origin=failover` 且 `held_s` 很大就是
        2026-08-21 那次事故的形状。
        """
        router = request.app.state.router
        if router is None:
            return JSONResponse(content={"enabled": False, "sessions": []})
        return JSONResponse(content={"enabled": True,
                                     "sessions": router.sticky_snapshot()})

    # ── V-L5 附卡：账本只读视图 ──
    @app.get("/admin/ledgers")
    async def list_ledgers(
        request: Request,
        cfg: ProxyConfig = _admin_key_dep,
    ) -> JSONResponse:
        """账本池 + 激活绑定 + Matter 锚（只读）。

        立卡背景：在此之前 **dashboard 完全看不到账本**，只能靠 probe 脚本。
        0.1.0 gate 的三个演示剧本里有一条就是「账本机制稳定运转」——
        看不见就演示不了，也没法在观察期判断机制到底有没有在动。

        口径三条，都是踩过的坑的对偶：
        - `active_in` 列的是**哪些 scope 把它设为激活**（scope = `agent␟project`，
          MQ-L11），不是 session ——同一本账本可以被多个 agent 激活（跨 agent
          交接正是北极星要接住的场景，live 上 `ldg-aaba…` 就同时被 hermes 与
          codex 绑过）。
        - `matter_id` 是**账本记的那个**（proxy 建账本时确定性派生并写死），
          它是权威边界；归属侧算出别的会告警但不换绑（ADR-0032 §4.5）。
        - 池来自 Hub 事件重放，重启存活；**不是进程内内存**。
        """
        from bladex_core.ledger_runtime import split_scope

        agency = getattr(request.app.state, "agency", None)
        if agency is None:
            return JSONResponse(content={"enabled": False, "ledgers": [],
                                         "total": 0})
        # scope 在这里就拆成 (agent, project) —— 分隔符是 core 的实现细节，
        # 让前端自己 split 就是把它复制成第二份真相。
        active_by_ledger: dict[str, list[dict[str, str]]] = {}
        for scope, lid in agency.activation.snapshot().items():
            agent, project = split_scope(scope)
            active_by_ledger.setdefault(lid, []).append(
                {"agent": agent, "project": project})
        rows = []
        for lid, led in agency.pool.items():
            rows.append({
                "ledger_id": lid,
                "title": led.title or "",
                "status": led.status or "active",
                "matter_id": led.matter_id or "",
                "parent_ledger_id": led.parent_ledger_id or "",
                "goal": led.goal or "",
                "goal_source": led.goal_source or "",
                "created_at": led.created_at or "",
                "updated_at": led.updated_at or "",
                "active_in": sorted(active_by_ledger.get(lid, []),
                                    key=lambda s: (s["agent"], s["project"])),
                # `entries` 是方法不是字段（goal 也在 section_order 里，但它是单块
                # 文本、不走 entries，故计数恒 0——界面上不展示 goal 的条目数）。
                "entry_counts": {s: len(led.entries(s)) for s in led.section_order
                                 if s != "goal"},
            })
        # 激活的排前面，组内按更新时间倒序——最想先看到的是"现在在跑哪本"。
        # 两趟稳定排序：先排时间，再按激活分组（Python sort 稳定，组内序保留）。
        from bladex_core.ledger import iso_ms   # 数值键，不比 ISO 串（MQ-L46）
        rows.sort(key=lambda r: iso_ms(r["updated_at"] or r["created_at"]), reverse=True)
        rows.sort(key=lambda r: not r["active_in"])
        return JSONResponse(content={"enabled": True, "ledgers": rows,
                                     "total": len(rows)})

    @app.get("/admin/ledgers/{ledger_id}")
    async def get_ledger(
        ledger_id: str,
        request: Request,
        cfg: ProxyConfig = _admin_key_dep,
    ) -> JSONResponse:
        """单本账本全文（五段正文 + 子账本 + 渲染后的 Markdown）。"""
        from bladex_core.ledger import children_of, render_ledger_md
        from bladex_core.ledger_runtime import split_scope

        agency = getattr(request.app.state, "agency", None)
        led = None if agency is None else agency.pool.get(ledger_id)
        if led is None:
            return JSONResponse(status_code=404, content={
                "status": "not_found", "detail": f"unknown ledger {ledger_id}"})
        active = [dict(zip(("agent", "project"), split_scope(s), strict=True))
                  for s, lid in agency.activation.snapshot().items() if lid == ledger_id]
        return JSONResponse(content={
            "ledger": led.model_dump(mode="json"),
            "markdown": render_ledger_md(led),
            "children": [{"ledger_id": c.ledger_id, "title": c.title}
                         for c in children_of(agency.pool, ledger_id)],
            "active_in": sorted(active, key=lambda s: (s["agent"], s["project"])),
        })

    @app.post("/admin/ledgers/bind-legacy-matters")
    async def admin_ledger_bind_legacy_matters(
        request: Request,
        cfg: ProxyConfig = _admin_key_dep,
    ) -> JSONResponse:
        """旧账本 `matter_id` 反向指针回填（`task-ledger-switch-candidates-20260904.md` §2）。

        V-L5（09-01）起建本即 `bind_matter`；之前建的 12 本 `matter_id` 空。Index 侧不受影响
        （重建按 `ledger_anchor_matter_id(lid)` 确定性派生，同一个 id），断的是账本→Matter
        的反向指针：`.md` / 本 API / `ledger show` 看不到 matter，且每次重建 `anchor_decision`
        对它们都走 `create_matter=True`。本端点对每本空的发一条 `LEDGER_UPDATE` 事件，
        payload = `bind_matter(led, ledger_anchor_matter_id(lid))`——Hub 真相、重放等价、
        id 与 Index 已有的一致；`bind_matter` 拒绝改绑已绑本 ⇒ 结构上不可能换绑。
        body `{"dry_run": true}`（默认）只列清单不写。
        """
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            body = {}
        dry_run = bool((body or {}).get("dry_run", True))
        agency = getattr(request.app.state, "agency", None)
        if agency is None:
            return JSONResponse(status_code=503, content={
                "error": {"message": "agency runtime unavailable", "type": "storage_error"}})
        from bladex_core.ledger_runtime import bind_matter, ledger_anchor_matter_id
        import datetime as _dt
        now_iso = _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")
        todo = [lid for lid, led in agency.pool.items() if not led.matter_id]
        bound: list[dict] = []
        for lid in sorted(todo):
            mid = ledger_anchor_matter_id(lid)
            bound.append({"ledger_id": lid, "matter_id": mid,
                          "title": agency.pool[lid].title})
            if dry_run:
                continue
            new = bind_matter(agency.pool[lid], mid, updated_at=now_iso)
            agency.pool[lid] = new
            agency._emit(AdminEventType.LEDGER_UPDATE, {"ledger": new.model_dump()})
        logger.info("admin_ledger_legacy_matters_bound", count=len(bound), dry_run=dry_run)
        return JSONResponse(status_code=200, content={
            "status": "dry_run" if dry_run else "bound",
            "checked": len(agency.pool), "unbound": len(todo), "ledgers": bound})

    @app.delete("/admin/ledgers/{ledger_id}")
    async def delete_ledger(
        ledger_id: str,
        request: Request,
        cfg: ProxyConfig = _admin_key_dep,
    ) -> JSONResponse:
        """给账本打墓碑（`task-ledger-tombstone-cleanup-20260901.md`）。

        墓碑 = 「重建不得复活」（ADR-0012 §3.6）。账本层打一次，
        三个消费方（agency 池 / 重建锚层 / flash 物化）全部跟随，
        且锚定 Matter **自然不会被创建**——`_la_ledger_at` 查不到它。

        🔴 **不可逆**：Hub 追加式，墓碑本身也是追加记录，没有"取消墓碑"。
        调用前请确认已备份 `data/bladex_hub` 与 `data/bladex_index`。

        🔴 **不做级联**：账本的锚定 Matter 不在这里删。下一次全量重建时它不会被
        重新创建，其成员 fact 落回 D1/五层兜底——那才是我们要的（让归属机制
        自己判，不替它做决定）。存量库里那张卡要等重建才消失。
        """
        hub: MemoryHub | None = request.app.state.hub
        if hub is None:
            return _hub_unavailable()

        agency = getattr(request.app.state, "agency", None)
        known = agency is not None and ledger_id in agency.pool
        if not known:
            # 池里没有 ≠ 不存在（proxy 可能刚重启、或它已被墓碑）——报 404 但说清楚。
            return JSONResponse(status_code=404, content={
                "status": "not_found",
                "detail": f"unknown ledger {ledger_id} (not in the live pool; "
                          f"it may already be tombstoned)"})

        tombstone_key = hub.append_tombstone(
            target_type=TombstoneTargetType.LEDGER,
            target_key=ledger_id,
            source=TombstoneSource.USER,
        )
        title = getattr(agency.pool.get(ledger_id), "title", "")
        # 立刻从活着的池与激活表里摘掉，不必等重启（投影即时收敛）。
        agency.pool.pop(ledger_id, None)
        freed = agency.activation.forget_ledger(ledger_id)
        logger.info("admin_ledger_tombstoned", ledger_id=ledger_id, title=title,
                    tombstone_key=tombstone_key, freed_scopes=freed,
                    hint="anchored Matter disappears on the next full rebuild; "
                         "its facts fall back to D1/DPL attribution")
        return _admin_response(200, "tombstoned", ledger_id=ledger_id,
                               tombstone_key=tombstone_key)

    # ── Beta T9: 只读 admin API（CLI / dashboard / MCP 共同地基）──
    register_admin_read_routes(app, _admin_key_dep)

    # ── Beta T18–T20: dashboard（单文件 SPA，包资源发布）──
    # 页面本身是静态资源无鉴权（不含任何秘密）；数据面全部走 /admin/*，
    # admin key 在页面内输入、存 sessionStorage——无 key 时数据请求 401。
    @app.get("/dashboard")
    async def dashboard() -> Any:
        from importlib import resources

        from fastapi.responses import HTMLResponse
        try:
            html = (resources.files("bladex_proxy") / "dashboard.html").read_text(
                encoding="utf-8")
        except OSError as e:
            return JSONResponse(status_code=500, content={
                "error": {"message": f"dashboard asset missing: {e}",
                          "type": "internal_error"}})
        return HTMLResponse(content=html)

    return app


_INDEX_WRITE_RETRIES = 3
_INDEX_WRITE_RETRY_DELAY_S = 0.05


def _open_writable_index(cfg: ProxyConfig) -> MemoryIndex | None:
    """打开一个临时可写 Memory Index 实例（admin 操作用，操作完立即 close）。

    P2 修复（review P2）：与后台 consolidation worker 可能竞争 RocksDB
    单写者锁。重试几次；仍失败则返回 None——调用方必须显式返回 202 deferred
    （Memory Hub 管理事件已写，下次重建时 _replay_admin_events 重放生效）。
    不再静默降级为 200 success。
    """
    for attempt in range(_INDEX_WRITE_RETRIES):
        try:
            index = MemoryIndex(cfg.index_path, embedder=None, read_only=False,
                           novelty_threshold=cfg.novelty_threshold,
                           entity_aware_novelty=cfg.entity_aware_novelty,
                           entity_overlap_threshold=cfg.entity_overlap_threshold,
                           novelty_topk=cfg.novelty_topk,
                           entity_aware_rerank=cfg.entity_aware_rerank,
                           entity_rerank_alpha=cfg.entity_rerank_alpha,
                           entity_rerank_expand_k=cfg.entity_rerank_expand_k)
            index.open()
            return index
        except Exception as e:
            if attempt < _INDEX_WRITE_RETRIES - 1:
                time.sleep(_INDEX_WRITE_RETRY_DELAY_S)
                continue
            logger.warning("admin_index_writable_open_failed", error=str(e),
                           retries=_INDEX_WRITE_RETRIES,
                           hint="Memory Hub event recorded; Memory Index deferred to next rebuild")
            return None


def _apply_to_index(
    cfg: ProxyConfig,
    apply_fn: Callable[[MemoryIndex], Any],
) -> tuple[bool, Any]:
    """打开可写 Memory Index 并应用操作（Memory Index: 重试 + 不静默降级）。

    返回 (applied, result)。applied=False 表示 Memory Index 写锁竞争失败，
    调用方应返回 202 deferred（Memory Hub 管理事件已写，下次重建重放生效）。
    """
    p2w = _open_writable_index(cfg)
    if p2w is None:
        return False, None
    try:
        return True, apply_fn(p2w)
    finally:
        p2w.close()


async def require_admin_key(request: Request) -> ProxyConfig:
    """管理端点鉴权依赖（P7: 统一 auth 逻辑，消除逐端点重复）。

    ADR-0027 §2.2：走 admin_auth_check（独立 BLADEX_ADMIN_KEYS；未配时个人模式
    回落 client key、企业形态拒绝）——数据面 key 不再天然拥有全库管理权。
    """
    cfg: ProxyConfig = request.app.state.config
    authorization = request.headers.get("authorization")
    client_key = _extract_bearer(authorization)
    ok, reason = cfg.admin_auth_check(client_key)
    if not ok:
        logger.warning("admin_auth_failed", reason=reason)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"message": f"Invalid API key: {reason}", "type": "authentication_error"},
        )
    return cfg


# 模块级单例（避免 B008: Depends() 不在参数默认值里直接调用）
_admin_key_dep = Depends(require_admin_key)


def _hub_unavailable() -> JSONResponse:
    """Memory Hub 不可用时的统一 503 响应。"""
    return JSONResponse(
        status_code=503,
        content={"error": {"message": "Memory Hub unavailable", "type": "storage_error"}},
    )


def _admin_response(status_code: int, status_text: str, **extra: Any) -> JSONResponse:
    """构造 admin 端点响应（统一 status + deferred 语义，Memory Index）。"""
    return JSONResponse(
        status_code=status_code,
        content={"status": status_text, **extra},
    )


async def _resolve_with_memory(
    request: Request,
    req_model: str,
    cfg: ProxyConfig,
    query: str,
    messages: list[dict],
    facts: list[Fact] | None = None,
    agent_id: str | None = None,
    auxiliary: bool = False,
    output_modalities: list[str] | None = None,
    session_id: str | None = None,
    context_chars: int | None = None,
    allowed_exposure: str = "public",
    principal_id: str | None = None,
    team_ids: list[str] | None = None,
) -> tuple[str, ModelRoute, list[ModelRoute]]:
    """路由决策（Phase 1：配置驱动确定性流水线 + RA 修复卡）。

    agent_id 来自身份解析（ADR-0010），命中 routing.toml [strategies.agent] 时收窄候选池。
    session_id / context_chars：T2 粘性 + T1 规模层（T0 管道）。
    facts 透传给 Phase 2 预留钩子；HardRule 构造延迟到 Phase 2 接入时再做（T6，
    省每请求开销——钩子当前恒返 None，构造了也没人消费）。
    """
    router = request.app.state.router
    if router is None:
        model, route, failover = await resolve_route(req_model, cfg, None, query, agent_id=agent_id, auxiliary=auxiliary)
    else:
        if facts is None:
            facts = []
        requires = _required_capabilities(messages, output_modalities=output_modalities, agent_id=agent_id, cfg=cfg)
        model, route, failover = await resolve_route(
            req_model, cfg, router, query,
            facts=facts, hard_rules=None, requires=requires, agent_id=agent_id,
            auxiliary=auxiliary, session_id=session_id, context_chars=context_chars,
            allowed_exposure=allowed_exposure,
            principal_id=principal_id, team_ids=team_ids,
        )
    # ADR-0021 T7: 路由 source 分布
    _m: Metrics | None = getattr(request.app.state, "metrics", None)
    if _m is not None and route.source:
        _m.inc("bladex_route_source_total", labels={"source": route.source})
    return model, route, failover


def _record_inject_metrics(request: Request, route_facts: list[Fact]) -> None:
    """ADR-0021 T7: 注入过滤命中数 + 已注入 fact 数（exposure 守卫活动信号）。"""
    m: Metrics | None = getattr(request.app.state, "metrics", None)
    if m is None:
        return
    src = getattr(request.app.state, "inject_source", None)
    filtered = getattr(src, "last_exposure_filtered", 0) if src is not None else 0
    if filtered:
        m.inc("bladex_inject_facts_filtered_total", float(filtered))
    m.set_gauge("bladex_inject_facts_injected", float(len(route_facts)))


def _validate_sensitivity_judge(cfg: ProxyConfig) -> None:
    """ADR-0021 section 3.2 启动校验：敏感层开 + 裁判非本地 -> 拒启动(strict)/告警(false)。

    敏感模式下裁判会看到 query，必须钉本地（exposure=local）。裁判非本地时：
    - BLADEX_ROUTE_STRICT=true -> 拒启动（fail-fast，配置错误必须显性）。
    - false -> 告警降级（运行时 route() 跳裁判走允许池 default_tier，敏感内容不外发）。
    """
    rcfg = cfg.routing_config
    sens = rcfg.sensitivity_config()
    if not sens.enabled:
        return
    if not rcfg.strategies.filter.enabled or not rcfg.filter_judge_model:
        return
    judge_name = rcfg.filter_judge_model
    judge_exposure = "public"
    for m in rcfg.models:
        if m.name == judge_name:
            judge_exposure = m.exposure
            break
    if judge_exposure == "local":
        return
    msg = (f"sensitivity enabled but judge '{judge_name}' exposure='{judge_exposure}' "
           f"(must be local); sensitive queries would leak via judge")
    if cfg.route_strict:
        raise RuntimeError(f"BLADEX_ROUTE_STRICT=true: {msg}")
    logger.error("SENSITIVITY_JUDGE_NOT_LOCAL", judge=judge_name,
                 exposure=judge_exposure, hint=msg + " -> runtime will skip judge")


def _resolve_sensitivity(cfg: ProxyConfig, identity: Identity) -> tuple[str, str]:
    """ADR-0021 section 3.3b：敏感度解析（身份的纯静态函数，查表 O(1)）。

    编排换序里在注入前完成（敏感度前移）。返回 (level, allowed_exposure)。
    敏感层关闭 -> (normal, public) = 现状零回归。解析异常 -> 最严（fail-closed）。

    V-A1 示范迁移：模块开关置于最前（`BLADEX_MODULE_SENSITIVITY`，默认开=零回归；
    细粒度仍归 [strategies.sensitivity]）。🔴 注意方向：模块关 = 行为等同敏感层
    未配置（normal/public），企业形态想关它应先想清楚——identity.toml 在而敏感
    模块关的组合由 sensitivity 配置侧的启动校验管，不在这里重写一遍。
    """
    if not module_enabled("sensitivity"):
        return "normal", "public"
    try:
        sens_cfg = cfg.routing_config.sensitivity_config()
        return sens_cfg.resolve(identity.agent_id, identity)
    except Exception as e:
        logger.warning("sensitivity_resolve_failed", error=str(e),
                       hint="fail-closed -> strictest (local)")
        return "sensitive", "local"



def _detect_prefix_changed(
    request: Request,
    identity: Identity,
    messages: list[dict],
) -> bool:
    """检测 agent 是否压缩了上下文（ADR-0016 §3.5 冲突检测）。

    用 Memory Hub 进程内缓存 O(1) 查上一轮的 prefix_hash，与当前 messages 前缀比较。
    Memory Hub 不可用或缓存未命中 -> False（保守不跳过 CAP）。
    """
    hub: MemoryHub | None = request.app.state.hub
    if hub is None:
        return False
    try:
        return hub.check_prefix_changed(identity.session_prefix(), messages)
    except Exception:
        return False


def _capability_error_response(agent_id: str | None, requires: list[str]) -> JSONResponse:
    """§3.4 快返：能力过滤无合格候选 → 结构化错误（不静默降级到可能答非所问的模型）。"""
    who = f"agent {agent_id} 集合" if agent_id else "模型池"
    msg = f"{who}无 {requires} 能力，请补充对应多模态/能力模型"
    logger.warning("route_no_capable_fastfail", agent=agent_id, requires=requires)
    return JSONResponse(
        status_code=422,
        content={"error": {"message": msg, "type": "no_capable_candidate"}},
    )

def _extract_output_modalities(body: dict) -> list[str]:
    """从请求体提取输出模态信号（标准 OpenAI 格式）。

    只检测 modalities 字段（OpenAI 规范中请求音频输出的显式信号）。
    不检测 audio 字段——它只是音频配置（voice/format），部分 agent（如 Hermes
    tts.provider=openai）对所有请求都带 audio 参数但不一定需要音频输出，
    用它做信号会导致所有请求误判为 TTS。

    tool call 式 TTS 信号不统一，不做自动检测（见 plan doc 遗留问题）。
    """
    modalities: set[str] = set()
    raw = body.get("modalities")
    if isinstance(raw, list):
        for m in raw:
            if m == "audio":
                modalities.add("audio")
    return sorted(modalities)


def _required_capabilities(
    messages: list[dict],
    output_modalities: list[str] | None = None,
    agent_id: str | None = None,
    cfg: ProxyConfig | None = None,
) -> list[str]:
    """从消息检测所需能力（多模态硬约束）：复用 inject.detect_required_capabilities。

    T5c（RA8）：agent base 命中 [capability].code_agents → requires += "code"
    （能力护栏：将来池里出现无 code 模型时不会被选中；live 配置全模型有 code = 无行为变化）。
    """
    caps = detect_required_capabilities(messages, output_modalities=output_modalities)
    if agent_id and cfg is not None:
        try:
            code_agents = cfg.routing_config.capability.code_agents
        except Exception:
            code_agents = []
        base = agent_id.split(":", 1)[0]
        if base in code_agents and "code" not in caps:
            caps = sorted([*caps, "code"])
    return caps


def _extract_output_modalities_responses(body: dict) -> list[str]:
    """从 Responses 请求体提取输出模态信号（ADR-0023）。

    Responses API 没有 modalities 字段；输入侧多模态（input_image/input_file）
    不强制要求输出同模态。当前返回空（Codex 纯文本/工具场景），保留接口对称。
    """
    return []


def _estimate_context_chars(messages: list[dict]) -> int:
    """T1（RA1）：请求上下文规模估算（字符数，不引 tokenizer——档位判断够用）。

    str content 计全长；list content 只计 text part，非文本 part（图/音/文件）
    记固定 1000 字符当量。
    """
    total = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text", "")
                    if isinstance(text, str) and text:
                        total += len(text)
                    elif part.get("type"):
                        total += 1000
        # tool_calls 参数也计入（工具循环的主要体积之一）
        for tc in msg.get("tool_calls", []) or []:
            if isinstance(tc, dict):
                func = tc.get("function", {})
                args = func.get("arguments", "") if isinstance(func, dict) else ""
                if isinstance(args, str):
                    total += len(args)
    return total



class RoundPrep:
    """一轮编排的中间产物（V-A1 T2 编排函数化，2026-08-28）。

    三端点（chat/anthropic/responses）此前各持一份编排胶水，MQ-P4/P7/P8 三类
    缺陷全长在胶水的分叉处（"同一件事只在一个端点做对了"）。收成单实现后，
    端点只剩：协议解析 → auth → `_prepare_round` → `_apply_agency_surfaces` →
    request_params → 分发 handler。分叉点显式参数化：`output_modalities`（responses
    的提取器不同）、`log_route_debug`（chat 独有的调试日志，保留原行为）。
    """

    __slots__ = ("identity", "agent_source", "allowed_exposure", "messages",
                 "injected_messages", "injected_text", "ms_identity", "ms_inject",
                 "route_facts", "cap_info", "model", "route", "failover",
                 "decision_meta", "error_response")

    def __init__(self) -> None:
        self.error_response: JSONResponse | None = None
        self.model = ""
        self.route = None
        self.failover: list = []
        self.decision_meta = None


async def _prepare_round(
    request: Request,
    *,
    req: ChatCompletionRequest,
    headers: dict,
    body: dict,
    output_modalities: list[str],
    log_route_debug: bool = False,
) -> RoundPrep:
    """认身份 → 敏感度 → 入站准备 → 注入 → 路由 → decision_meta（三端点共享）。

    行为与拆前逐字等价；逐步注释见各 helper 与历史卡（ADR-0021 §3.3b 换序 /
    V-P5a 入站准备 / MQ-L7 账本块后移 / ADR-0024 T1 task_units）。
    """
    cfg: ProxyConfig = request.app.state.config
    prep = RoundPrep()

    t0 = time.perf_counter()
    identity, agent_source = resolve_identity(headers, req, request.app.state.identity_registry)
    # 项目识别第一档（2026-08-29）：cwd 标记 → git 指纹键；确定性、按 cwd 缓存。
    from bladex_proxy.project_identity import resolve_from_request as _resolve_project
    _proj = _resolve_project(headers, req.messages)
    identity.project_id = _proj.project_id
    identity.project_source = _proj.source
    identity.project_name = _proj.name
    identity.project_root = _proj.root
    prep.ms_identity = (time.perf_counter() - t0) * 1000

    # ADR-0021 §3.3b: 敏感度解析前移到注入之前（身份的纯静态函数）。
    sens_level, allowed_exposure = _resolve_sensitivity(cfg, identity)
    identity.sensitivity = sens_level

    # V-P5a/MQ-P7：入站准备（剥回流+拼接恢复）在 query/prefix/注入之前——
    # 全下游看到同一份形态；三端点同一条路（MQ-P7 之前 anthropic/responses 漏接）。
    agency = request.app.state.agency
    req.messages = agency.prepare_inbound(req.messages, identity.session_prefix())
    messages = req.messages

    query = _extract_query(messages)
    prefix_changed = _detect_prefix_changed(request, identity, messages)
    injected_messages, injected_text, ms_inject, route_facts, cap_info = await do_inject_async(
        messages, query, cfg, request.app.state.inject_source,
        auxiliary=identity.auxiliary, user_id=identity.user_id,
        visibility=identity.visibility, allowed_exposure=allowed_exposure,
        assembler=request.app.state.assembler,
        prefix_changed=prefix_changed,
        session_id=identity.session_id, agent_id=identity.agent_id,
    )
    _record_inject_metrics(request, route_facts)
    # ADR-0024 T1：任务单元挂 request.state，_enqueue_turn 取用落 Memory Hub
    request.state.task_units = cap_info.get("units") or []
    # U9（ADR-0025）：重构决策落 Memory Hub 审计（Memory Index 蒸馏永不读，G3/I5）
    request.state.reconstruction = cap_info.get("reconstruction")

    # T1（RA1）：上下文规模估算（注入后的 final messages = 真实发上游的规模）
    context_chars = _estimate_context_chars(injected_messages)
    if log_route_debug:
        logger.info("route_debug", body_keys=sorted(body.keys()),
                     has_modalities="modalities" in body, has_audio="audio" in body,
                     output_modalities=output_modalities,
                     requires=_required_capabilities(messages, output_modalities=output_modalities, agent_id=identity.agent_id, cfg=cfg),
                     context_chars=context_chars,
                     msg_content_types=[p.get("type") for m in messages[-3:] if isinstance(m.get("content"), list) for p in m["content"] if isinstance(p, dict)])

    prep.identity = identity
    prep.agent_source = agent_source
    prep.allowed_exposure = allowed_exposure
    prep.messages = messages
    prep.injected_messages = injected_messages
    prep.injected_text = injected_text
    prep.ms_inject = ms_inject
    prep.route_facts = route_facts
    prep.cap_info = cap_info

    try:
        prep.model, prep.route, prep.failover = await _resolve_with_memory(request, req.model, cfg, query, messages, facts=route_facts, agent_id=identity.agent_id, auxiliary=is_cheap_tier_auxiliary(identity.aux_source), output_modalities=output_modalities, session_id=identity.session_id, context_chars=context_chars, allowed_exposure=allowed_exposure, principal_id=identity.user_id, team_ids=identity.team_ids)
    except NoCapableCandidateError:
        prep.error_response = _capability_error_response(identity.agent_id, _required_capabilities(messages, output_modalities=output_modalities, agent_id=identity.agent_id, cfg=cfg))
        return prep

    prep.decision_meta = _build_decision_meta(
        prep.route,
        cap_triggered=cap_info.get("triggered", False),
        cap_summary_hash=cap_info.get("summary_hash", ""),
        cap_kept_recent_n=cap_info.get("kept_recent_n", 0),
        inject_position="before_last_user",
        injected_fact_ids=cap_info.get("injected_fact_ids") or [],
        injected_matter_ids=cap_info.get("injected_matter_ids") or [],
        injected_fact_keys=cap_info.get("injected_fact_keys") or {},
    )
    return prep


def _apply_agency_surfaces(
    request: Request,
    prep: RoundPrep,
    *,
    tools_in: list | None,
    stream: bool,
) -> tuple[Any, bool]:
    """工具面增补 + 账本块 + 自我介绍（三端点共享；账本域 aux 判据单一副本）。

    此前 aux 判据表达式在三端点 × augment/ledger 两处 = **六份副本**（MQ-L20/L21
    的每次修订都要改六处）。判据语义：
      - `is_ledgerless_auxiliary`（第三套语义，MQ-L20）：续写/重试/恢复类信封的
        assistant 侧接着做的是同一件正事，该带账本——不读裸 `identity.auxiliary`；
      - `subagent` / `brings_no_tools`（MQ-L21：零工具=要一段文本，Codex 标题生成
        建噪声账本的病例）/ `is_isolated_subcall`（MQ-L8：无 system 单 user）。
    🔴 MQ-L34（2026-09-02）：这四条合成的 `ledgerless_aux` 只挡**账本族**；
    记忆族只被 `aux_source` / `subagent` 挡（`identity.LEDGER_ONLY_AUX_REASONS`
    的补集）。此前整个合成值传给 `auxiliary=` ⇒ 零工具请求连
    `bladex_memory_search` 都没有，与 08-30「记忆族无条件注入」相悖。
    顺序（MQ-L7 C 方案）：augment_tools（tier 已知）→ 账本块（与工具面同步决定
    首步指令，追加真末尾）→ 自我介绍（V-P6，前缀位，门控与工具面同步）。
    返回 (增补后的 tools, toolface_injected)；injected_messages 就地更新进 prep，
    并把「BladeX 加进正文的全部字符」停进 `request.state.bladex_added_chars`
    （MQ-A34 ③ 的分子，见函数尾部）。
    """
    agency = request.app.state.agency
    identity = prep.identity
    own_tools = list(tools_in or [])
    # MQ-L34（2026-09-02）：四个子条件按顺序取**第一个**命中的名字进日志；
    # `ledgerless_aux` 语义不变（任一命中 ⇒ 不给账本面），但只有
    # `LEDGER_ONLY_AUX_REASONS` 之外的子条件才连记忆族一起挡。
    aux_reason = (AUX_REASON_SOURCE if is_ledgerless_auxiliary(identity.aux_source)
                  else AUX_REASON_SUBAGENT if identity.subagent
                  else AUX_REASON_NO_TOOLS if brings_no_tools(own_tools)
                  else AUX_REASON_ISOLATED if is_isolated_subcall(prep.messages)
                  else "")
    ledgerless_aux = bool(aux_reason)
    memory_blocked = ledgerless_aux and aux_reason not in LEDGER_ONLY_AUX_REASONS
    tools_now, tf_injected = agency.augment_tools(
        tools_in, agent_id=identity.agent_id, stream=stream,
        auxiliary=memory_blocked, ledgerless=ledgerless_aux,
        aux_reason=aux_reason, tier=prep.route.tier,
        # 🔴 账本族的第二个放行口需要这两个（2026-08-30）：本会话绑着激活账本
        # 时即使档位在集合外也给账本工具。与下面 insert_ledger_block 传的是
        # 同一组值、同一个判据（`session_owns_active_ledger`）。
        session_id=identity.session_id, project_id=identity.project_id)
    prep.injected_messages = agency.insert_ledger_block(
        prep.injected_messages, identity.agent_id,
        project_id=identity.project_id,
        auxiliary=ledgerless_aux, with_instruction=tf_injected,
        # MQ-L28：块注入与工具面吃同一个档位判定 + 绑定会话判据
        tier=prep.route.tier, session_id=identity.session_id)
    prep.injected_messages = agency.insert_system_notes(
        prep.injected_messages, toolface_injected=tf_injected)
    # MQ-A34 ③ 读数口径（2026-09-04）：BladeX 加进正文的**全部**字符 = 注入后正文 −
    # agent 原始正文。`_warn_if_output_truncated` 拿它当 `injected_chars` 的分子。
    # 修前分子是 `prep.injected_text`（inject 阶段的硬规则正文），而 32K 病例里
    # 挤爆输出预算的恰是本函数加的这几块（about / 名册 / 账本块 / 候选段）——
    # 尺子量不到最大的那一段，与 MQ-A33 同族。
    # 🔴 停在 `request.state` 而不是给 `_enqueue_turn` 加第 N 个参数：它有 12 个调用点，
    # 漏传一个就是静默退化（取法同 `task_units` / `reconstruction`）。分子与分母必须
    # 用同一把尺子，故这里与那边都用 `assembly.estimate_context_chars`。
    # **不夹到 0**：`strip_previous_injection` 会剥掉 agent 回传的上一轮注入块，
    # 净值可以为负（这一轮 BladeX 反而让正文变短）——那是真读数，夹掉就是又一次
    # 静默丢信息。缺这个属性（没走过本函数）另有 `-1` 一格，与负净值不混。
    request.state.bladex_added_chars = (
        estimate_context_chars(prep.injected_messages)
        - estimate_context_chars(prep.messages))
    return tools_now, tf_injected


def _call_hooks(request: Request, identity: Identity, primary_model: str = ""):
    """T2/T4：给 call_model 的 health + on_success（failover 实际切换后回写粘性）。

    `primary_model` = 路由决策选出的首选。实际用的模型与它不同 = 这轮发生了
    failover，粘性要标 `origin=FAILOVER`（G14/MQ-RT1）——被动降级和主动选择
    必须可区分，否则一次瞬时抖动会让会话永久漂移在降级模型上。
    """
    health = getattr(request.app.state, "model_health", None)
    router = request.app.state.router
    # 粘性/温度回写逻辑在 route.make_success_hook（V-A2 下沉，行为逐字保持）。
    return health, make_success_hook(
        router, session_id=identity.session_id,
        agent_id=identity.agent_id, primary_model=primary_model)


def _extract_bearer(authorization: str | None) -> str | None:
    if not authorization:
        return None
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return authorization.strip()


def _message_text(content: Any) -> str:
    """把消息 content 归一成纯文本 —— **必须处理 list 形态**。

    OpenAI 多模态约定下 content 是 `[{"type":"text","text":...}, ...]`。
    Pi 的 user 消息实测就是这个形状；只处理 str 的话，详情页会显示"没有 user 消息"，
    而真实原因是格式没接住（ADR-0008 §2.5 早就为注入侧处理过同一形状）。
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            p.get("text", "") for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        ).strip()
    return ""


#: agent 详情页取样上限：只对最后这么多条 key 做 `get()`。
#: 详情页是"够不够判断这是谁"，不是审计——多取不会更好认，只会更慢。
_AGENT_DETAIL_SAMPLE = 8
#: 详情页里每段文本的截断长度（system prompt / user 消息）。
_AGENT_DETAIL_TEXT_CHARS = 400


def _collect_headers(
    request: Request,
    authorization: str | None,
    x_agent_id: str | None,
    x_session_id: str | None,
    user_agent: str | None,
) -> dict[str, str]:
    """G11.1：把**完整** header 集合交给身份解析，而不是四个写死的字段。

    改动前三个端点各自手搓一个四键字典（`authorization`/`x-agent-id`/
    `x-session-id`/`user-agent`），厂商前缀 header 根本进不了门——dsh 每轮发的
    `x-deepseek-harness-{user-id,session-id,compact}` 全被丢在门外，而那正是
    agent 主动告诉我们的身份、会话与 auxiliary 标记（台账 MQ-A1）。

    向后兼容：四个原有键仍显式覆盖，缺失时保持 `""` 而非键不存在——下游
    `_extract_api_key` 等一律走 `.get(name, "")`，语义逐字不变。

    :returns: 键全小写的 header 字典。
    """
    merged = {name.lower(): value for name, value in request.headers.items()}
    merged.update({
        "authorization": authorization or "",
        "x-agent-id": x_agent_id or "",
        "x-session-id": x_session_id or "",
        "user-agent": user_agent or "",
    })
    return merged


def _is_anthropic_client(request: Request) -> bool:
    """T26: 判断请求是否来自 Anthropic 客户端（Claude Code）。

    判据：anthropic-version header 存在，或 user-agent 含 claude。
    """
    if request.headers.get("anthropic-version"):
        return True
    ua = request.headers.get("user-agent", "").lower()
    return "claude" in ua


def _build_model_obj(display_name: str, is_anthropic: bool) -> dict[str, Any]:
    """T26: 构造单个 model 对象，按客户端类型分流响应形态。

    - OpenAI 形态: {"id", "object":"model", "created", "owned_by"}
    - Anthropic 形态: {"id", "type":"model", "display_name", "created_at"}
    """
    if is_anthropic:
        return {
            "id": display_name,
            "type": "model",
            "display_name": display_name,
            "created_at": 0,
        }
    return {
        "id": display_name,
        "object": "model",
        "created": 0,
        "owned_by": "bladex",
    }


async def _handle_stream(
    request: Request,
    model: str,
    route: ModelRoute,
    failover: list[ModelRoute],
    messages: list[dict],
    req: ChatCompletionRequest,
    identity: Identity,
    injected_text: str,
    ms_identity: float,
    ms_inject: float,
    t_start: float,
    agent_source: AgentSource = AgentSource.FALLBACK,
    original_messages: list[dict] | None = None,
    request_params: RequestParams | None = None,
    decision_meta: DecisionMeta | None = None,
    allowed_exposure: str = "public",
) -> StreamingResponse:
    """流式处理：转发 → 捕获 → 回传 SSE → 入库（try/finally 确保断开也存）。"""
    _ledger_messages = original_messages if original_messages is not None else messages
    import json as _json

    kwargs = _build_kwargs(req)
    result = CaptureResult()
    health, on_used = _call_hooks(request, identity, primary_model=model)

    # ISSUE-5: 上游调用可能抛异常
    try:
        stream = await call_model(model, messages, stream=True, route=route, failover=failover,
                                  health=health, on_success=on_used, **kwargs)
    except Exception as e:
        logger.error("upstream_call_failed", error=router_sdk.error_text(e))
        await _enqueue_turn(request, identity, model, _ledger_messages, injected_text,
                           "", [], TurnStatus.FAILED, str(e), ms_identity, ms_inject, t_start,
                           agent_source, request_params=request_params, decision_meta=decision_meta)
        return JSONResponse(
            status_code=502,
            content={"error": {"message": f"Upstream error: {router_sdk.error_text(e)}", "type": "upstream_error"}},
        )

    # V-P5a：流式拦截（interception 关 = 原路径逐字不变）。
    agency = request.app.state.agency
    if agency.interception_on:
        async def _loop_llm(msgs: list[dict]) -> dict:
            r2 = await call_model(model, msgs, stream=False, route=route,
                                  failover=failover, health=health,
                                  on_success=on_used, **kwargs)
            return _loop_reply(r2)

        sse_source = intercept_chat_stream(
            capture_stream(stream, result), agency=agency, capture_result=result,
            upstream_messages=messages,
            session_prefix=identity.session_prefix(),
            allowed_exposure=allowed_exposure, call_llm=_loop_llm,
            session_id=identity.session_id, agent_id=identity.agent_id,
            project_id=identity.project_id)
    else:
        sse_source = capture_stream(stream, result)

    async def generate():
        try:
            async for sse_chunk in sse_source:
                yield sse_chunk.encode()
        except Exception as e:
            logger.error("stream_error", error=str(e))
            err = {"error": {"message": str(e), "type": "proxy_error"}}
            yield f"data: {_json.dumps(err)}\n\n".encode()
        finally:
            # ISSUE-3: 无论正常结束还是客户端断开，都尽量入库
            # A2: shield 防断开时 cancel scope 取消入库 -> Turn 丢
            status = TurnStatus.OK if result.done and not result.error else TurnStatus.FAILED
            error = result.error
            await _enqueue_turn_shielded(
                request, identity, model, _ledger_messages, injected_text,
                result.full_text, result.tool_events, status, error,
                ms_identity, ms_inject, t_start, agent_source,
                response_meta=_build_response_meta_from_capture(result),
                request_params=request_params, decision_meta=decision_meta,
            )

    return StreamingResponse(
        generate(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


async def _handle_non_stream(
    request: Request,
    model: str,
    route: ModelRoute,
    failover: list[ModelRoute],
    messages: list[dict],
    req: ChatCompletionRequest,
    identity: Identity,
    injected_text: str,
    ms_identity: float,
    ms_inject: float,
    t_start: float,
    agent_source: AgentSource = AgentSource.FALLBACK,
    original_messages: list[dict] | None = None,
    request_params: RequestParams | None = None,
    decision_meta: DecisionMeta | None = None,
    allowed_exposure: str = "public",
) -> JSONResponse:
    """非流式处理。ISSUE-5: 上游出错时存 FAILED 轮次。"""
    _ledger_messages = original_messages if original_messages is not None else messages
    kwargs = _build_kwargs(req)
    health, on_used = _call_hooks(request, identity, primary_model=model)

    try:
        response = await call_model(model, messages, stream=False, route=route, failover=failover,
                                    health=health, on_success=on_used, **kwargs)
    except Exception as e:
        logger.error("upstream_call_failed", error=router_sdk.error_text(e))
        await _enqueue_turn(request, identity, model, _ledger_messages, injected_text,
                           "", [], TurnStatus.FAILED, str(e), ms_identity, ms_inject, t_start,
                           agent_source, request_params=request_params, decision_meta=decision_meta)
        return JSONResponse(
            status_code=502,
            content={"error": {"message": f"Upstream error: {router_sdk.error_text(e)}", "type": "upstream_error"}},
        )

    try:
        message = response.choices[0].message
        full_text = message.content or ""
        response_data = response.model_dump() if hasattr(response, "model_dump") else response
        status = TurnStatus.OK
        error = ""

        # T6: 非流式 tool_calls 提取（与流式路径统一，别恒空）
        # 🔴 在拦截**之前**提取：Hub 存完整真相（含 bladex 调用），agent 拿处置后的。
        tool_events = _extract_tool_events_from_response(message)

        # V-P5a：非流式拦截处置（interception 关 = 短路原样）。
        agency = request.app.state.agency
        if agency.interception_on:
            msg_dict = _message_as_dict(message)

            async def _loop_llm(msgs: list[dict]) -> dict:
                r2 = await call_model(model, msgs, stream=False, route=route,
                                      failover=failover, health=health,
                                      on_success=on_used, **kwargs)
                return _loop_reply(r2)

            processed, _transcript, _mode = await agency.process_message(
                msg_dict, upstream_messages=messages,
                session_prefix=identity.session_prefix(),
                allowed_exposure=allowed_exposure, call_llm=_loop_llm,
                session_id=identity.session_id, agent_id=identity.agent_id,
                project_id=identity.project_id)
            if _mode != "none" and isinstance(response_data, dict):
                choices = response_data.get("choices") or [{}]
                choices[0]["message"] = processed
                if _mode == "pure_bladex":
                    choices[0]["finish_reason"] = (
                        "tool_calls" if processed.get("tool_calls") else "stop")
    except (AttributeError, IndexError) as e:
        logger.error("non_stream_parse_failed", error=str(e))
        full_text = ""
        response_data = {"error": str(e)}
        status = TurnStatus.FAILED  # ISSUE-5: 解析失败也标 FAILED
        error = str(e)
        tool_events = []

    response_meta = _build_response_meta_from_response(response)
    await _enqueue_turn(request, identity, model, _ledger_messages, injected_text,
                       full_text, tool_events, status, error, ms_identity, ms_inject, t_start,
                       agent_source, response_meta=response_meta,
                       request_params=request_params, decision_meta=decision_meta)
    return JSONResponse(content=response_data)


_REDIS_RECOVER_COOLDOWN_S = 60.0
_last_recover_attempt: float = 0.0
_redis_skip_count: int = 0


async def _try_recover_redis(request: Request) -> PipelineRedis | None:
    """懒恢复：enqueue 时发现 pipeline is None，尝试连/启动 Redis（事件触发，不轮询）。

    流程：
      1. 尝试连接 Redis（可能已被手动启动）
      2. 连不上 -> 尝试启动 redis-server（subprocess）
      3. 再连一次
      4. 连上 -> 初始化 Pipeline + pipeline_worker + 回灌溢出文件

    冷却 60 秒防频繁重试（Redis 二进制不存在 / 端口被占等情况）。
    所有失败路径都有 warning/error 日志，便于排查。
    """
    global _last_recover_attempt, _redis_skip_count
    now = time.monotonic()
    if now - _last_recover_attempt < _REDIS_RECOVER_COOLDOWN_S:
        _redis_skip_count += 1
        if _redis_skip_count % 50 == 1:
            logger.warning(
                "pipeline_redis_still_unavailable",
                skip_count=_redis_skip_count,
                cooldown_remaining=round(_REDIS_RECOVER_COOLDOWN_S - (now - _last_recover_attempt), 1),
                hint="Redis unavailable, turns are NOT being stored; "
                     "will retry recovery after cooldown",
            )
        return None
    _last_recover_attempt = now

    cfg: ProxyConfig = request.app.state.config
    pipeline = PipelineRedis(
        redis_url=cfg.redis_url, stream=cfg.redis_stream,
        group=cfg.redis_group, consumer=cfg.redis_consumer,
        queue_max_len=cfg.queue_max_len, overflow_dir=cfg.overflow_dir,
    )

    # 1. 尝试直连
    try:
        await pipeline.connect()
    except Exception as connect_err:
        logger.info("pipeline_redis_connect_failed_trying_start", error=str(connect_err))
        # 2. 连不上 -> 尝试启动 Redis（仅本地实例）
        from urllib.parse import urlparse
        _parsed = urlparse(cfg.redis_url)
        _redis_host = _parsed.hostname or "127.0.0.1"
        _redis_port = _parsed.port or 6379
        if _redis_host not in ("127.0.0.1", "localhost", "::1"):
            logger.warning(
                "pipeline_redis_remote_skip_start",
                redis_host=_redis_host,
                hint="Redis URL points to a remote host; cannot auto-start. "
                     "Ensure the remote Redis is reachable.",
            )
            return None
        try:
            subprocess.Popen(
                ["redis-server", "--daemonize", "yes", "--port", str(_redis_port),
                 "--appendonly", "yes", "--appendfsync", "everysec",
                 "--tcp-keepalive", "0",
                 "--dir", "data/"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            await asyncio.sleep(1)
            await pipeline.connect()
        except FileNotFoundError:
            logger.error(
                "pipeline_redis_binary_not_found",
                hint="redis-server not in PATH; install Redis or set BLADEX_REDIS_URL "
                     "to an external Redis instance. Turns are NOT being stored.",
            )
            return None
        except Exception as start_err:
            logger.warning(
                "pipeline_redis_recover_failed",
                connect_error=str(connect_err),
                start_error=str(start_err),
                hint="Redis could not be started; turns are NOT being stored",
            )
            return None

    # 3. 连上了 -> 初始化 Pipeline + pipeline_worker
    request.app.state.pipeline = pipeline
    hub: MemoryHub = request.app.state.hub
    pipeline_worker = PipelineWorker(pipeline, hub)
    pipeline_worker.start()
    request.app.state.pipeline_worker = pipeline_worker

    # 回灌磁盘溢出文件
    replayed = await pipeline.replay_overflow()
    if replayed > 0:
        logger.info("pipeline_overflow_replayed_on_recover", count=replayed)

    _redis_skip_count = 0
    logger.info(
        "pipeline_redis_recovered",
        hint="lazy recovery succeeded on enqueue; storage pipeline resumed",
        replayed_overflow=replayed,
    )
    return pipeline


async def _enqueue_inner_loop_turns(
    request: Request, pipeline: "PipelineRedis", identity: Identity,
    loop_turns: list[Turn],
) -> None:
    """把主轮带出的内循环 aux 轮逐条入队（MQ-P9）。单条失败落盘，不影响其余。

    日志 `inner_loop_turn_enqueued` 是 MQ-W 对账的第二项：
    `route_calling ≈ turn_enqueued + Σ rounds`。
    """
    if not loop_turns:
        return
    ok = 0
    for lt in loop_turns:
        try:
            await pipeline.enqueue(lt)
            ok += 1
        except Exception as e:  # noqa: BLE001 —— 单条失败不阻断其余，落盘兜底
            logger.error("inner_loop_enqueue_failed", error=str(e),
                         roundtrip=lt.roundtrip)
            try:
                request.app.state.disk_spill.spill(lt)
            except Exception as se:  # noqa: BLE001
                logger.error("inner_loop_spill_failed", error=str(se))
    # 不按名读 Turn 的总耗时字段：H1 对账 gate 拿它当"写了没按名读"的样例字段
    # （`test_field_write_only_is_info_not_dead`），这里读一下会把样例翻成 ok。
    # 耗时读数走日志侧 `_loop_cost`（agency）与 Hub 侧 aux 轮本身。
    logger.info("inner_loop_turn_enqueued", key=identity.session_prefix(),
                rounds=ok, requested=len(loop_turns),
                tokens=sum(int((lt.response_meta.usage or {}).get("total", 0) or 0)
                           for lt in loop_turns))


async def _enqueue_turn(
    request: Request,
    identity: Identity,
    model: str,
    messages: list[dict],
    injected_text: str,
    response_text: str,
    tool_events: list,
    status: TurnStatus,
    error: str,
    ms_identity: float,
    ms_inject: float,
    t_start: float,
    agent_source: AgentSource = AgentSource.FALLBACK,
    # T4(ADR-0018 §4.4): 三组元数据（可选，调用方按可用性传）
    response_meta: ResponseMeta | None = None,
    request_params: RequestParams | None = None,
    decision_meta: DecisionMeta | None = None,
    raw_request_ref: str = "",
) -> None:
    """构造 Turn 并异步入库（不阻塞响应）。"""
    ms_total = (time.perf_counter() - t_start) * 1000

    # MQ-A34 ③：上游因输出预算耗尽而截断 ⇒ 一条可 grep 告警（放这里 = 三协议 × 流式/非流式
    # 唯一汇合点，且 agent / model / 注入正文 / 原始 prompt 都在手）。只记不改。
    _warn_if_output_truncated(request, identity, model, messages,
                              response_meta, request_params)

    # ADR-0021 T7: metrics（请求计数 + 延迟分位，by agent/sensitivity）
    _metrics: Metrics | None = getattr(request.app.state, "metrics", None)
    if _metrics is not None:
        _lbl = {"agent": identity.agent_id, "sensitivity": getattr(identity, "sensitivity", "normal")}
        _metrics.inc("bladex_requests_total", labels=_lbl)
        _metrics.observe("bladex_request_latency_ms", ms_total, labels=_lbl)

    # ADR-0024 T1/T2：任务单元一等实体化。热路径已算好的单元直接用，
    # logical_turn / roundtrip 从单元派生（旧 _infer_turn_metadata 已退役）。
    # 走 request.state 传递而非加 6 个 handler 形参：单元是**每请求**产物，
    # request.state 正是它的语义位置，且避免 3 个端点 × 流式/非流式的签名蔓延。
    # 双层 getattr：_enqueue_turn 在"绝不丢 Turn"的路径上，不得因属性缺失抛异常
    # （Starlette 真实 Request 恒有 .state；测试替身 / 未来的调用方不保证）。
    units = _resolve_task_units(
        messages, getattr(getattr(request, "state", None), "task_units", None)
    )
    logical_turn, roundtrip = derive_turn_metadata(units)

    # U9（ADR-0025 §5）：重构决策（A/B 层 degrade_plan + B 指纹 + C 澄清）落 Turn。
    # 双层 getattr 同 task_units——绝不因属性缺失丢 Turn。
    _recon_raw = getattr(getattr(request, "state", None), "reconstruction", None)
    _recon = None
    if isinstance(_recon_raw, dict):
        try:
            _recon = ReconstructionRecord(**_recon_raw)
        except Exception as _re:  # noqa: BLE001
            logger.warning("reconstruction_record_invalid", error=str(_re))

    # V-P4 硬点 1：本轮被剥离的 bladex 调用与结果随 Turn 入 Hub。
    # 从 agency 直接取而不是加参数：`_enqueue_turn` 有七八个调用点，
    # 加一个必传参数等于让每个调用点都记得传——漏一个就是静默不落库。
    _splice_recs: list = []
    _ag = getattr(request.app.state, "agency", None)
    if _ag is not None and getattr(_ag, "interception_on", False):
        try:
            _splice_recs = _ag.splice.new_in_turn(identity.session_prefix())
        except Exception as _se:  # noqa: BLE001 —— 落库辅助信息，不阻断入库
            logger.warning("splice_records_collect_failed", error=str(_se))

    # MQ-L26：这一轮给没给账本面（同上，从 agency 取而不是加第 13 个参数）。
    # 🔴 `None` 有实义 —— 见 `Turn.ledger_face` 的三态语义：缺数就报缺数，
    # 不能因为"取不到"就写 False，那会让全量重建时历史锚定一次性失效。
    _ledger_face = getattr(_ag, "last_ledger_face", None) if _ag is not None else None

    turn = Turn(
        identity=identity, model=model, request_messages=messages,
        task_units=units,
        splice_records=_splice_recs,
        ledger_face=_ledger_face,
        reconstruction=_recon,
        injected_memory=injected_text, response_text=response_text,
        tool_events=tool_events, status=status, error=error,
        ms_identity=ms_identity, ms_inject=ms_inject, ms_total=ms_total,
        agent_source=agent_source,
        logical_turn=logical_turn, roundtrip=roundtrip,
        auxiliary=identity.auxiliary,
        aux_source=identity.aux_source,
        session_id_source=identity.session_id_source,
        sensitivity=identity.sensitivity,
        response_meta=response_meta or ResponseMeta(),
        request_params=request_params or RequestParams(),
        decision_meta=decision_meta or DecisionMeta(),
        raw_request_ref=raw_request_ref,
    )

    # MQ-P9（ADR-0032 §3.2 硬点 6）：本会话待入账的内循环轮次，随主轮一起入库。
    # 取法同 `splice_records`——从 agency 取，不加第 N 个参数。主轮先入、aux 轮后入，
    # 每轮各自一条 Turn（entry_id 唯一，不撞键）。
    _loop_turns: list[Turn] = []
    if _ag is not None and getattr(_ag, "interception_on", False):
        try:
            from bladex_proxy.loopledger import build_inner_loop_turn
            for _pr in _ag.loop_ledger.drain(identity.session_prefix()):
                _loop_turns.append(build_inner_loop_turn(
                    identity, model, _pr, trigger_ts=turn.ts.isoformat()))
        except Exception as _le:  # noqa: BLE001 —— 入账附件，不阻断主轮入库
            logger.warning("inner_loop_turns_build_failed", error=str(_le))
            _loop_turns = []

    pipeline: PipelineRedis | None = request.app.state.pipeline
    if pipeline is None:
        pipeline = await _try_recover_redis(request)
    if pipeline is not None:
        try:
            await pipeline.enqueue(turn)
            logger.info("turn_enqueued", key=identity.session_prefix(),
                        status=status.value, ms_total=round(ms_total, 2))
            await _enqueue_inner_loop_turns(request, pipeline, identity, _loop_turns)
            return
        except Exception as e:
            logger.error("enqueue_failed", error=str(e))
            # 落到下面的 DiskSpill 兜底

    # T1: 最终 fallback 无条件落盘（消灭 ADR-0017 已知限制）
    # 三种情况 Turn 必须落文件：pipeline is None / 60s 冷却期未恢复 / enqueue 抛异常。
    # 恢复后由 replay_overflow() 回灌 Pipeline。
    disk_spill: DiskSpill = request.app.state.disk_spill
    disk_spill.spill(turn)
    for _lt in _loop_turns:
        disk_spill.spill(_lt)
    # ADR-0027 §3.3：Redis 缺席不再丢数据（T1 已修），但仍是降级——
    # 落盘的 turn 在回灌前进不了 Memory Index，用户侧表现为"这段时间的事没记住"。
    metrics_mod.record_degradation(metrics_mod.ENQUEUE_SKIPPED)
    logger.warning(
        "turn_spilled_to_disk",
        key=identity.session_prefix(),
        status=status.value,
        ms_total=round(ms_total, 2),
        hint="turn stored to disk overflow; will replay when Redis recovers",
    )


def _warn_if_output_truncated(
    request: Request,
    identity: Identity,
    model: str,
    messages: list[dict],
    response_meta: ResponseMeta | None,
    request_params: RequestParams | None,
) -> None:
    """MQ-A34 ③（2026-09-03 拍板：0.1.0 只做告警）：上游 `finish_reason` 落在
    `capture.OUTPUT_TRUNCATED_REASONS` ⇒ 打 `upstream_output_truncated`，带注入占比读数。

    病例：本机 32K 窗口模型上注入面把输出预算挤到 30–164 token，账本面 0/28——
    "模型不调 switch"与"模型调不出 switch"在没有这条告警前分不开
    （`task-hermes-accept-ledger-gap-20260903.md` §5/§6）。① 窗口字段 / ② 按预算裁剪
    归 0.2.0；本函数**零行为差异**：只读已捕获的元数据，不动请求也不动回复。

    `inject_ratio = injected_chars / (prompt_chars + injected_chars)`——`messages` 是 agent 原始
    消息（`_ledger_messages`，注入前），注入正文另计，分母才是模型真正看到的那份。

    🔴 **分子口径 2026-09-04 修正**（MQ-A34 补记）：修前分子 = `len(prep.injected_text)`，
    只算 inject 阶段的硬规则正文；about / 名册 / 账本块 / 候选段在
    `_apply_agency_surfaces` 才加进 `injected_messages`，全在分子之外——而 32K 病例里
    挤爆输出预算的**恰是这几块**（与 MQ-A33 同族：尺子量不到最大的那一段）。
    现分子 = `_apply_agency_surfaces` 停在 `request.state.bladex_added_chars` 的
    「BladeX 加进正文的全部字符」。**修前的读数按硬规则口径读，不与修后同列比较。**

    三个读数格互不合并：`injected_chars=-1` = 接线断（本轮没走过
    `_apply_agency_surfaces`）；`0` = 真的一个字没加；**负值** = 剥掉的回传注入
    多于本轮新注（`strip_previous_injection`），是真读数不是错。
    """
    fr = (response_meta.finish_reason if response_meta is not None else "") or ""
    if not output_truncated(fr):
        return
    added = getattr(getattr(request, "state", None), "bladex_added_chars", None)
    injected_chars = -1 if added is None else int(added)
    try:
        prompt_chars = estimate_context_chars(messages)
    except Exception:  # noqa: BLE001 —— 读数辅助，不因消息形态怪异阻断入库
        prompt_chars = -1
    seen = prompt_chars + injected_chars if added is not None and prompt_chars >= 0 else 0
    usage = response_meta.usage if response_meta is not None else {}
    logger.warning(
        "upstream_output_truncated",
        agent=identity.agent_id, model=model, finish_reason=fr,
        injected_chars=injected_chars, prompt_chars=prompt_chars,
        inject_ratio=(round(injected_chars / seen, 3) if seen > 0 else None),
        completion_tokens=usage.get("completion"), prompt_tokens=usage.get("prompt"),
        max_tokens=(request_params.max_tokens if request_params is not None else None),
        session=identity.session_prefix(),
    )


async def _enqueue_turn_shielded(
    request: Request,
    identity: Identity,
    model: str,
    messages: list[dict],
    injected_text: str,
    response_text: str,
    tool_events: list,
    status: TurnStatus,
    error: str,
    ms_identity: float,
    ms_inject: float,
    t_start: float,
    agent_source: AgentSource = AgentSource.FALLBACK,
    response_meta: ResponseMeta | None = None,
    request_params: RequestParams | None = None,
    decision_meta: DecisionMeta | None = None,
    raw_request_ref: str = "",
) -> None:
    """A2：shield 入库 await，防客户端断开时 cancel scope 取消入库 -> Turn 丢。

    Starlette/uvicorn 在客户端断开时取消响应任务，generator finally 里的
    `await _enqueue_turn(...)` 会立即再抛 CancelledError -> 断开那轮（往往长回复、
    信息量不低）最容易丢。本函数把入库派发为独立 Task（脱离请求 cancel scope），
    持引用防 GC（app.state.bg_tasks，done 回调清理），再 shield await：
      - 正常路径：等价于直接 await（测试无回归）；
      - 断开路径：shield 抛 CancelledError 交还响应任务，内部入库 Task 继续跑完
        （落 Redis 或 T1 的 DiskSpill 兜底），Turn 不丢。
    """
    task = asyncio.create_task(_enqueue_turn(
        request, identity, model, messages, injected_text,
        response_text, tool_events, status, error,
        ms_identity, ms_inject, t_start, agent_source,
        response_meta=response_meta, request_params=request_params,
        decision_meta=decision_meta, raw_request_ref=raw_request_ref,
    ))
    bg = request.app.state.bg_tasks
    bg.add(task)
    task.add_done_callback(bg.discard)
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        logger.info(
            "enqueue_shielded_on_disconnect",
            key=identity.session_prefix(), status=status.value,
            hint="client disconnected; enqueue continues in background",
        )
        raise


def _build_request_params(request: Request, req: ChatCompletionRequest, *,
                          forwarded_tools: list | None = None) -> RequestParams:
    """T4(ADR-0018 §4.4 C3): 从请求构造 RequestParams + tools schema 入 __msg__ 池。

    requested_model = agent 请求的原始 model（Turn.model 存路由后实际用的）。
    tools schema 全文经 Memory Hub __msg__ 池 hash 去重存储，Turn 只存 tools_hash。
    Memory Hub 不可用时 tools_hash 留空（不阻塞热路径）。

    🔴 `forwarded_tools` = **实际转发上游的那份**（工具面增补之后）。
    2026-08-26 取证发现：`/v1/chat/completions` 上 `augment_tools` 排在本函数
    **之前**（注释明写"Hub 的 tools_hash 记录的是实际转发形态"），而
    `/v1/messages` 与 `/v1/responses` 上排在**之后**，且增补写进的是
    `extra_kwargs["tools"]`——与本函数读的 `req.tools` 是**两个对象**，
    光调换顺序都救不回来。后果：那两个协议的 tools_hash 记的是**增补前**形态，
    "模型实际看到哪些工具"在 Claude Code / Codex 上**查不出来**
    （差点据此得出"CC 从没拿到过工具面"的错误结论）。
    顺带统一了记录格式：`extra_kwargs["tools"]` 已由 `parse_*` 归一为 OpenAI 形态。
    """
    rp = RequestParams(
        requested_model=req.model or "",
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        tool_choice=req.tool_choice,
    )
    tools = req.tools if forwarded_tools is None else forwarded_tools
    if tools:
        hub: MemoryHub | None = request.app.state.hub
        if hub is not None:
            try:
                rp.tools_hash = hub.store_tools_schema(tools)
            except Exception as e:
                logger.warning("tools_schema_store_failed", error=str(e))
    return rp


def _build_response_meta_from_capture(result: CaptureResult) -> ResponseMeta:
    """T4(C1/C2/C6): 从 CaptureResult 构造 ResponseMeta。"""
    return ResponseMeta(
        reasoning_text=result.reasoning_text,
        usage=dict(result.usage),
        finish_reason=result.finish_reason,
        ms_first_chunk=result.ms_first_chunk,
    )


def _message_as_dict(message: Any) -> dict:
    """上游 message 对象 → 拦截协议要的 plain dict（**唯一实现点**，三条非流式路径共用）。

    真上游（Router/LiteLLM）给的是 pydantic Message，走 `model_dump()`；
    Mapping 原样拷贝；其余对象（测试 mock 的 `SimpleNamespace`、SDK 的轻量对象）
    按 `__dict__` 递归转——此前三处各写一份 `model_dump() if hasattr else dict(m)`，
    2026-09-02 四模块翻默认后 `/v1/responses` 非流式测试用 `SimpleNamespace` 首次
    走到拦截路径 ⇒ `dict(SimpleNamespace)` TypeError。同一件事三份字面量，收成一处。
    """
    def _plain(o: Any) -> Any:
        if hasattr(o, "model_dump"):
            return o.model_dump()
        if isinstance(o, dict):
            return {k: _plain(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_plain(x) for x in o]
        if hasattr(o, "__dict__") and not isinstance(o, type):
            return {k: _plain(v) for k, v in vars(o).items() if not k.startswith("_")}
        return o
    out = _plain(message)
    return out if isinstance(out, dict) else {"content": out}


def _loop_reply(response: Any) -> dict:
    """内循环 `call_llm` 的返回形态：assistant message dict + 归一 `usage`。

    🔴 MQ-P9（2026-09-02）：此前五处 `_loop_llm` 都只返回 `message.model_dump()`，
    而 usage 挂在 response 上不在 message 上 ⇒ `LoopRound.usage` 恒空、
    `_loop_cost(...).tokens` 恒 0、入 Hub 的 aux 轮拿不到 token 支出。
    五处收成一个函数——同一件事五份字面量正是"改一处漏四处"的形状。
    `innerloop` 取走 usage 后会把它从外发消息里剥掉（wire 形态里没有这个字段）。
    """
    m2 = response.choices[0].message
    out = _message_as_dict(m2)
    try:
        usage = getattr(response, "usage", None)
        if usage is not None:
            out["usage"] = _extract_usage_dict(usage)
    except Exception:  # noqa: BLE001 —— usage 只是入账附件，取不到不阻断内循环
        pass
    return out


def _build_response_meta_from_response(response: Any) -> ResponseMeta:
    """T4(C1/C2/C6): 从非流式上游响应构造 ResponseMeta。

    提取 reasoning_content（C1）、usage（C2）、finish_reason（C6）。
    任一缺失留默认值（非流式无首字延迟，ms_first_chunk=0）。
    """
    meta = ResponseMeta()
    try:
        choice = response.choices[0]
        message = choice.message
        rc = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
        if rc:
            # ADR-0020 T4: 某些模型/Router 解析下 reasoning_content 可能是 list -> 强制 str
            meta.reasoning_text = "".join(str(x) for x in rc) if isinstance(rc, list) else str(rc)
        fr = getattr(choice, "finish_reason", None)
        if fr:
            meta.finish_reason = ",".join(str(x) for x in fr) if isinstance(fr, list) else str(fr)
    except (AttributeError, IndexError, TypeError) as e:
        logger.debug("response_meta_parse_failed", error=str(e))
    try:
        usage = getattr(response, "usage", None)
        if usage is not None:
            meta.usage = _extract_usage_dict(usage)
    except Exception:
        pass
    return meta


def _extract_usage_dict(usage: Any) -> dict:
    """从上游 usage 对象提取 token 计数（非流式复用）。"""
    try:
        if hasattr(usage, "model_dump"):
            data = usage.model_dump()
        elif isinstance(usage, dict):
            data = usage
        else:
            data = dict(usage)
        return {
            "prompt": int(data.get("prompt_tokens", 0)),
            "completion": int(data.get("completion_tokens", 0)),
            "total": int(data.get("total_tokens", 0)),
        }
    except Exception:
        return {}


def _inject_top_k() -> int:
    """注入条目数（默认 `flags.MAX_PREFETCH_K`=15，`BLADEX_INJECT_TOPK` 可调）。
    S1（2026-09-03）后主动检索路径已退役，本值只透传给 InjectionSource.top_k 与 config show。"""
    from bladex_core.flags import MAX_PREFETCH_K
    try:
        v = int(os.environ.get("BLADEX_INJECT_TOPK", "") or MAX_PREFETCH_K)
    except ValueError:
        return MAX_PREFETCH_K
    return max(1, v)


def _build_decision_meta(route: ModelRoute, *, cap_triggered: bool = False,
                         cap_summary_hash: str = "", cap_kept_recent_n: int = 0,
                         inject_position: str = "",
                         injected_fact_ids: list[str] | None = None,
                         injected_matter_ids: list[str] | None = None,
                         injected_fact_keys: dict[str, str] | None = None) -> DecisionMeta:
    """T4(C4): 从路由决策 + CAP 信息构造 DecisionMeta。

    ADR-0028 E2.3：追加 injected_fact_ids —— 本轮注入命中的 Memory Index 条目 id 随 turn 落 Memory Hub，
    consolidator 消费时聚合成 ref_count（proxy 侧 Memory Index 只读，写权不变）。
    """
    return DecisionMeta(
        route={"source": route.source, "tier": route.tier, "reason": route.reason},
        cap={
            "triggered": cap_triggered,
            "summary_hash": cap_summary_hash,
            "kept_recent_n": cap_kept_recent_n,
        },
        inject_position=inject_position,
        injected_fact_ids=list(injected_fact_ids or []),
        injected_fact_keys=dict(injected_fact_keys or {}),
        # M3-1：Matter 卡命中留痕（三时钟的相关钟此前对 Matter 完全没有信号源）
        injected_matter_ids=list(injected_matter_ids or []),
    )


def _store_raw_request(request: Request, body: dict) -> str:
    """T4(C5): /v1/messages 原始请求体按 hash 入 __msg__ 池，返回 raw_request_ref。

    parse_anthropic_request 归一化有损（cache_control 丢弃等），存原文供回溯定位。
    Memory Hub 不可用时返回空（不阻塞热路径）。
    """
    if not body:
        return ""
    hub: MemoryHub | None = request.app.state.hub
    if hub is None:
        return ""
    try:
        return hub.store_raw_request(body)
    except Exception as e:
        logger.warning("raw_request_store_failed", error=str(e))
        return ""


def _resolve_task_units(
    messages: list[dict],
    units: list[TaskUnit] | None,
) -> list[TaskUnit]:
    """拿到本轮任务单元：优先用热路径已算好的，缺失才现算（ADR-0024 T1）。

    ~~`_infer_turn_metadata` 已退役~~——它是 `build_task_units` 的劣化版重复实现
    （ADR-0024 D1「任务结构算三遍、留零遍」的第二遍）。现在只有一套切分算法。

    `units` 由 `do_inject*` 经 `cap_info["units"]` 传入（索引基 = 未剥离的 messages
    = Memory Hub 存的 original_messages）。为 None 只发生在不经注入路径的调用（少数错误
    分支、旧测试），此时就地补算，口径完全一致。
    """
    if units:
        return units
    return build_task_units(messages)


def _extract_tool_events_from_response(message: Any) -> list:
    """T6: 从非流式响应的 message 中提取 tool_calls → ToolEvent 列表。

    与流式路径（capture.py 聚合 tool_calls）统一，避免提炼管线两边都取造成重复。
    """
    from bladex_proxy.models import ToolEvent

    events: list[ToolEvent] = []
    tool_calls = getattr(message, "tool_calls", None)
    if not tool_calls:
        return events

    for tc in tool_calls:
        try:
            func = getattr(tc, "function", None)
            tool_name = getattr(func, "name", "") if func else ""
            args_str = getattr(func, "arguments", "") if func else ""
            try:
                import json as _json
                args = _json.loads(args_str) if args_str else None
            except (json.JSONDecodeError, ValueError):
                args = args_str
            events.append(ToolEvent(
                tool_name=tool_name, arguments=args, direction="call",
            ))
        except Exception as e:
            logger.warning("non_stream_tool_event_extract_failed", error=str(e))

    return events


def _extract_query(messages: list[dict]) -> str:
    """末条 user 消息文本（进理解层之前的原始 query）。

    实现只有一份，在 `bladex_core.query_understanding.last_user_text`——
    离线仪器要复现生产的检索 query，三份近似实现里任何一份漂移都会让读数
    悄悄换参照系。
    """
    from bladex_core.query_understanding import last_user_text

    return last_user_text(messages)


def _build_kwargs(req: ChatCompletionRequest) -> dict:
    return req.to_router_kwargs()



async def _handle_anthropic_stream(
    request: Request,
    model: str,
    route: ModelRoute,
    failover: list[ModelRoute],
    messages: list[dict],
    extra_kwargs: dict,
    identity: Identity,
    injected_text: str,
    ms_identity: float,
    ms_inject: float,
    t_start: float,
    agent_source: AgentSource = AgentSource.FALLBACK,
    original_messages: list[dict] | None = None,
    request_params: RequestParams | None = None,
    decision_meta: DecisionMeta | None = None,
    raw_request_ref: str = "",
    allowed_exposure: str = "public",
) -> StreamingResponse:
    """Anthropic 流式处理：转发 → 按 Anthropic SSE 回传 → 入库。"""
    _ledger_messages = original_messages if original_messages is not None else messages
    result = CaptureResult()
    health, on_used = _call_hooks(request, identity, primary_model=model)

    try:
        stream = await call_model(model, messages, stream=True, route=route, failover=failover,
                                  health=health, on_success=on_used, **extra_kwargs)
    except Exception as e:
        logger.error("anthropic_upstream_call_failed", error=router_sdk.error_text(e))
        await _enqueue_turn(request, identity, model, _ledger_messages, injected_text,
                           "", [], TurnStatus.FAILED, str(e), ms_identity, ms_inject, t_start,
                           agent_source, request_params=request_params, decision_meta=decision_meta,
                           raw_request_ref=raw_request_ref)
        return JSONResponse(
            status_code=502,
            content={"type": "error", "error": {"type": "upstream_error",
                    "message": f"Upstream error: {router_sdk.error_text(e)}"}},
        )

    # V-P5b：anthropic 流式拦截（interception 关 = 原路径逐字不变）。
    _agency = request.app.state.agency
    _src = anthropic_stream_generator(stream, result, model)
    if _agency.interception_on:
        # V-P5c：内循环的 call_llm——上游本就是 OpenAI 兼容，用同一路由再调一次；
        # 最终消息交回本协议的 generator 合成（协议转换只发生在出口）。
        async def _loop_llm(msgs: list[dict]) -> dict:
            r2 = await call_model(model, msgs, stream=False, route=route,
                                  failover=failover, health=health,
                                  on_success=on_used, **extra_kwargs)
            return _loop_reply(r2)

        _src = intercept_anthropic_stream(
            _src, agency=_agency, capture_result=result,
            session_prefix=identity.session_prefix(),
            allowed_exposure=allowed_exposure, session_id=identity.session_id,
            agent_id=identity.agent_id,
            project_id=identity.project_id, upstream_messages=messages,
            call_llm=_loop_llm)

    async def generate():
        try:
            async for sse_chunk in _src:
                yield sse_chunk
        except Exception as e:
            logger.error("anthropic_stream_error", error=str(e))
            err = {"type": "error", "error": {"type": "proxy_error", "message": str(e)}}
            yield f"event: error\ndata: {json.dumps(err)}\n\n".encode()
        finally:
            # A2: shield 防客户端断开时 cancel scope 取消入库 -> Turn 丢
            status = TurnStatus.OK if result.done and not result.error else TurnStatus.FAILED
            await _enqueue_turn_shielded(
                request, identity, model, _ledger_messages, injected_text,
                result.full_text, result.tool_events, status, result.error,
                ms_identity, ms_inject, t_start, agent_source,
                response_meta=_build_response_meta_from_capture(result),
                request_params=request_params, decision_meta=decision_meta,
                raw_request_ref=raw_request_ref,
            )

    return StreamingResponse(
        generate(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive",
                 "X-Accel-Buffering": "no"},
    )


def _shim_processed_response(response: Any, processed: dict,
                             finish_reason: str | None = None) -> Any:
    """把**处置后**的 OpenAI 形态消息套回 `format_*_response` 认识的形状。

    两个 formatter 只用 `getattr` 读 `choices[0].message.{content,tool_calls}`、
    `choices[0].finish_reason` 与 `response.usage`——所以给一个同形状的轻量替身
    就够，**不去就地改 litellm 的响应对象**（那是别人的可变状态，改了会连带
    影响 `_build_response_meta_from_response` 与入库口径）。

    `processed` 里的 tool_calls 是 dict，formatter 按属性读，故逐层转 namespace。
    """
    from types import SimpleNamespace

    def _tc(d: dict) -> Any:
        fn = d.get("function") or {}
        return SimpleNamespace(
            id=d.get("id", ""), type=d.get("type", "function"),
            function=SimpleNamespace(name=fn.get("name", ""),
                                     arguments=fn.get("arguments", "")))

    msg = SimpleNamespace(
        content=processed.get("content"),
        tool_calls=[_tc(t) for t in processed.get("tool_calls") or []] or None,
        role=processed.get("role", "assistant"))
    orig_fr = ""
    try:
        orig_fr = response.choices[0].finish_reason or ""
    except (AttributeError, IndexError):
        pass
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg,
                                 finish_reason=finish_reason or orig_fr or "stop")],
        usage=getattr(response, "usage", None))


async def _intercept_non_stream(request: Request, response: Any, *, model: str,
                                route: Any, failover: Any, messages: list[dict],
                                identity: Identity, allowed_exposure: str,
                                extra_kwargs: dict, health: Any, on_used: Any) -> Any:
    """非流式拦截处置，三协议共用。返回给 formatter 用的响应（或原对象）。

    🔴 V-P5（2026-08-27 勘察）：`/v1/messages` 与 `/v1/responses` 的非流式路径
    此前**完全没有拦截接线**——而 `augment_tools` 在流式/非流式分支**之前**执行，
    所以这两条路径**照发 `bladex_*` 工具面却不拦截**：模型一调，调用直接透传给
    agent，而 agent 没有这个工具的处理器。

    chat 协议早有这段（`_handle_non_stream` 里的 V-P5a）——**同一件事只在一个
    端点上做对了**，与 MQ-A18 / MQ-L23 同型，本仓今天第三次。
    """
    agency = request.app.state.agency
    if not agency.interception_on:
        return response
    try:
        message = response.choices[0].message
    except (AttributeError, IndexError):
        return response
    msg_dict = _message_as_dict(message)

    async def _loop_llm(msgs: list[dict]) -> dict:
        r2 = await call_model(model, msgs, stream=False, route=route,
                              failover=failover, health=health,
                              on_success=on_used, **extra_kwargs)
        return _loop_reply(r2)

    processed, _transcript, mode = await agency.process_message(
        msg_dict, upstream_messages=messages,
        session_prefix=identity.session_prefix(),
        allowed_exposure=allowed_exposure, call_llm=_loop_llm,
        session_id=identity.session_id, agent_id=identity.agent_id,
        project_id=identity.project_id)
    if mode == "none":
        return response                     # 零改动通道：逐字原样
    fr = None
    if mode == "pure_bladex":
        # 调用全被拦下 ⇒ 终止语义必须跟着改，否则 agent 收到
        # 「说要调工具却一个都没有」并重试（chat 侧 2026-08-25 实测过 4 轮）。
        fr = "tool_calls" if processed.get("tool_calls") else "stop"
    logger.info("agency_nonstream_intercepted", mode=mode,
                agent=identity.agent_id,
                calls_left=len(processed.get("tool_calls") or []))
    return _shim_processed_response(response, processed, fr)


async def _handle_anthropic_non_stream(
    request: Request,
    model: str,
    route: ModelRoute,
    failover: list[ModelRoute],
    messages: list[dict],
    extra_kwargs: dict,
    identity: Identity,
    injected_text: str,
    ms_identity: float,
    ms_inject: float,
    t_start: float,
    agent_source: AgentSource = AgentSource.FALLBACK,
    original_messages: list[dict] | None = None,
    request_params: RequestParams | None = None,
    decision_meta: DecisionMeta | None = None,
    raw_request_ref: str = "",
    allowed_exposure: str = "public",
) -> JSONResponse:
    """Anthropic 非流式处理。"""
    _ledger_messages = original_messages if original_messages is not None else messages
    health, on_used = _call_hooks(request, identity, primary_model=model)
    try:
        response = await call_model(model, messages, stream=False, route=route, failover=failover,
                                    health=health, on_success=on_used, **extra_kwargs)
    except Exception as e:
        logger.error("anthropic_upstream_call_failed", error=router_sdk.error_text(e))
        await _enqueue_turn(request, identity, model, _ledger_messages, injected_text,
                           "", [], TurnStatus.FAILED, str(e), ms_identity, ms_inject, t_start,
                           agent_source, request_params=request_params, decision_meta=decision_meta,
                           raw_request_ref=raw_request_ref)
        return JSONResponse(
            status_code=502,
            content={"type": "error", "error": {"type": "upstream_error",
                    "message": f"Upstream error: {router_sdk.error_text(e)}"}},
        )

    # 🔴 入库口径：Hub 存**完整真相**（含被拦下的 bladex 调用），agent 拿处置后的。
    # 故先用未处置的 response 抽 tool_events，再做拦截（chat 侧同序，V-P5a）。
    _truth = format_anthropic_response(response, model)
    full_text = ""
    tool_events = []
    for block in _truth.get("content", []):
        if block.get("type") == "text":
            full_text += block.get("text", "")
        elif block.get("type") == "tool_use":
            from bladex_proxy.models import ToolEvent
            tool_events.append(ToolEvent(tool_name=block.get("name", ""),
                                         arguments=block.get("input"),
                                         direction="call"))

    # V-P5：非流式拦截（interception 关 = 返回原对象，逐字不变）
    _processed = await _intercept_non_stream(
        request, response, model=model, route=route, failover=failover,
        messages=messages, identity=identity, allowed_exposure=allowed_exposure,
        extra_kwargs=extra_kwargs, health=health, on_used=on_used)
    anthropic_response = (_truth if _processed is response
                          else format_anthropic_response(_processed, model))

    status = TurnStatus.OK if anthropic_response.get("type") != "error" else TurnStatus.FAILED
    response_meta = _build_response_meta_from_response(response)
    await _enqueue_turn(request, identity, model, _ledger_messages, injected_text,
                       full_text, tool_events, status, "",
                       ms_identity, ms_inject, t_start, agent_source,
                       response_meta=response_meta, request_params=request_params,
                       decision_meta=decision_meta, raw_request_ref=raw_request_ref)

    return JSONResponse(content=anthropic_response)


async def _handle_responses_stream(
    request: Request,
    model: str,
    route: ModelRoute,
    failover: list[ModelRoute],
    messages: list[dict],
    extra_kwargs: dict,
    identity: Identity,
    injected_text: str,
    ms_identity: float,
    ms_inject: float,
    t_start: float,
    agent_source: AgentSource = AgentSource.FALLBACK,
    original_messages: list[dict] | None = None,
    request_params: RequestParams | None = None,
    decision_meta: DecisionMeta | None = None,
    raw_request_ref: str = "",
    allowed_exposure: str = "public",
) -> StreamingResponse:
    """Responses /v1/responses 流式处理：转发 -> 按 Responses SSE 回传 -> 入库。

    入库 request_messages 恒为 OpenAI chat 格式（与 /v1/messages 一致），
    保证 Memory Index 重建等价性跨协议不变。
    """
    _ledger_messages = original_messages if original_messages is not None else messages
    result = CaptureResult()
    health, on_used = _call_hooks(request, identity, primary_model=model)

    try:
        stream = await call_model(model, messages, stream=True, route=route, failover=failover,
                                  health=health, on_success=on_used, **extra_kwargs)
    except Exception as e:
        logger.error("responses_upstream_call_failed", error=router_sdk.error_text(e))
        await _enqueue_turn(request, identity, model, _ledger_messages, injected_text,
                           "", [], TurnStatus.FAILED, str(e), ms_identity, ms_inject, t_start,
                           agent_source, request_params=request_params, decision_meta=decision_meta,
                           raw_request_ref=raw_request_ref)
        return JSONResponse(
            status_code=502,
            content={"error": {"type": "upstream_error",
                    "message": f"Upstream error: {router_sdk.error_text(e)}"}},
        )

    # V-P5b：responses 流式拦截（interception 关 = 原路径逐字不变）。
    _agency = request.app.state.agency
    _src = responses_stream_generator(stream, result, model)
    if _agency.interception_on:
        # V-P5c：内循环的 call_llm——上游本就是 OpenAI 兼容，用同一路由再调一次；
        # 最终消息交回本协议的 generator 合成（协议转换只发生在出口）。
        async def _loop_llm(msgs: list[dict]) -> dict:
            r2 = await call_model(model, msgs, stream=False, route=route,
                                  failover=failover, health=health,
                                  on_success=on_used, **extra_kwargs)
            return _loop_reply(r2)

        _src = intercept_responses_stream(
            _src, agency=_agency, capture_result=result,
            session_prefix=identity.session_prefix(),
            allowed_exposure=allowed_exposure, session_id=identity.session_id,
            agent_id=identity.agent_id,
            project_id=identity.project_id, upstream_messages=messages,
            call_llm=_loop_llm)

    async def generate():
        try:
            async for sse_chunk in _src:
                yield sse_chunk
        except Exception as e:
            logger.error("responses_stream_error", error=str(e))
            err = {"type": "error", "error": {"type": "proxy_error", "message": str(e)}}
            yield f"event: error\ndata: {json.dumps(err)}\n\n".encode()
        finally:
            # A2: shield 防客户端断开时 cancel scope 取消入库 -> Turn 丢
            status = TurnStatus.OK if result.done and not result.error else TurnStatus.FAILED
            await _enqueue_turn_shielded(
                request, identity, model, _ledger_messages, injected_text,
                result.full_text, result.tool_events, status, result.error,
                ms_identity, ms_inject, t_start, agent_source,
                response_meta=_build_response_meta_from_capture(result),
                request_params=request_params, decision_meta=decision_meta,
                raw_request_ref=raw_request_ref,
            )

    return StreamingResponse(
        generate(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive",
                 "X-Accel-Buffering": "no"},
    )


async def _handle_responses_non_stream(
    request: Request,
    model: str,
    route: ModelRoute,
    failover: list[ModelRoute],
    messages: list[dict],
    extra_kwargs: dict,
    identity: Identity,
    injected_text: str,
    ms_identity: float,
    ms_inject: float,
    t_start: float,
    agent_source: AgentSource = AgentSource.FALLBACK,
    original_messages: list[dict] | None = None,
    request_params: RequestParams | None = None,
    decision_meta: DecisionMeta | None = None,
    raw_request_ref: str = "",
    allowed_exposure: str = "public",
) -> JSONResponse:
    """Responses /v1/responses 非流式处理。"""
    _ledger_messages = original_messages if original_messages is not None else messages
    health, on_used = _call_hooks(request, identity, primary_model=model)
    try:
        response = await call_model(model, messages, stream=False, route=route, failover=failover,
                                    health=health, on_success=on_used, **extra_kwargs)
    except Exception as e:
        logger.error("responses_upstream_call_failed", error=router_sdk.error_text(e))
        await _enqueue_turn(request, identity, model, _ledger_messages, injected_text,
                           "", [], TurnStatus.FAILED, str(e), ms_identity, ms_inject, t_start,
                           agent_source, request_params=request_params, decision_meta=decision_meta,
                           raw_request_ref=raw_request_ref)
        return JSONResponse(
            status_code=502,
            content={"error": {"type": "upstream_error",
                    "message": f"Upstream error: {router_sdk.error_text(e)}"}},
        )

    # 🔴 入库口径：Hub 存**完整真相**（含被拦下的 bladex 调用），agent 拿处置后的。
    # 故先用未处置的 response 抽 tool_events，再做拦截（chat 侧同序，V-P5a）。
    _truth = format_responses_response(response, model)
    # 复用 tool_events_from_output：带 tool_call_id + 解析后的 dict arguments，
    # 与流式路径一致，跨轮 result 匹配不断链。
    full_text = _truth.get("output_text", "")
    tool_events: list[ToolEvent] = tool_events_from_output(_truth.get("output", []))

    # V-P5：非流式拦截（interception 关 = 返回原对象，逐字不变）
    _processed = await _intercept_non_stream(
        request, response, model=model, route=route, failover=failover,
        messages=messages, identity=identity, allowed_exposure=allowed_exposure,
        extra_kwargs=extra_kwargs, health=health, on_used=on_used)
    responses_response = (_truth if _processed is response
                          else format_responses_response(_processed, model))

    status = TurnStatus.OK if responses_response.get("status") != "failed" else TurnStatus.FAILED
    response_meta = _build_response_meta_from_response(response)
    await _enqueue_turn(request, identity, model, _ledger_messages, injected_text,
                       full_text, tool_events, status, "",
                       ms_identity, ms_inject, t_start, agent_source,
                       response_meta=response_meta, request_params=request_params,
                       decision_meta=decision_meta, raw_request_ref=raw_request_ref)

    return JSONResponse(content=responses_response)


app = create_app()
