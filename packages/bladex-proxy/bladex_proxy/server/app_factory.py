"""`create_app` / `lifespan` / 元端点（`/health` `/ready` `/metrics` `/v1/models`）+ 启动安全基线。

09-06 F0.1 自 `server.py` 拆出，零行为（函数体逐字搬家）。门面 `bladex_proxy.server` re-export。
"""

from __future__ import annotations

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from bladex_proxy import __version__, router_sdk
from bladex_proxy.assembly import AssemblyConfig, ContextAssembler
from bladex_proxy.config import ProxyConfig
from bladex_proxy.embedding import EmbedCallLog, build_embedder, validate_embed_sensitivity
from bladex_proxy.inject import InjectionSource
from bladex_proxy.agency import build_agency
from bladex_proxy.metrics import Metrics
from bladex_proxy.modules import validate_modules
from bladex_proxy.route import init_router
from bladex_proxy.routing_config import RoutingConfigError
from bladex_proxy.storage.pipeline_redis import DiskSpill, PipelineRedis
from bladex_proxy.storage.memory_index import MemoryIndex
from bladex_proxy.storage.memory_hub import MemoryHub
from bladex_proxy.storage.pipeline_worker import PipelineWorker
from bladex_proxy.server.orchestration import (
    _build_model_obj, _inject_top_k, _is_anthropic_client, _validate_sensitivity_judge
)
from bladex_proxy.server.admin_api import register_admin_routes
from bladex_proxy.server.endpoints_anthropic import register_anthropic_routes
from bladex_proxy.server.endpoints_chat import register_chat_routes, register_embeddings_routes
from bladex_proxy.server.endpoints_responses import register_responses_routes

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
    except RoutingConfigError:
        # 🔴 MQ-L61（2026-09-08）：**配置错误不许降级**。
        #
        # 本块的 `except Exception` 是给**运行时/存储**故障准备的降级路径
        # （LanceDB 打不开、embedding 起不来 ⇒ 只注硬规则，proxy 照常服务）。
        # 但 `build_embedder` 会读 `cfg.routing_config`，于是一个 **routing.toml 的
        # 悬空引用**也从这里进来，被降级成一句
        # `index_init_failed hint='proxy will run without Memory Index derived layer'`
        # —— 把"配置写错了"报成"索引层不可用"。
        #
        # 实测（09-08 Jason 故意把 glm-5.3-flash 配成 glm-5.4-flash 验收 MQ-L58）：
        # 这条 warning 先打出来，随后 `init_router` 才真正把进程炸掉。
        # **要不是 `init_router` 也读同一份配置，proxy 会带着"没有 Memory Index"
        # 正常起来，而没人知道为什么** —— 与 MQ-L58 是同一个形状，只是换了个地方：
        # 告警有、严重性没有。
        #
        # 降级的前提是"这件事失败了但系统还能对"；配置写错时系统**不可能对**
        # （路由策略整个不生效）。故这一类原样上抛，由 lifespan 拒绝启动。
        raise
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

    # ── 三种入站协议 + embeddings + admin：登记顺序与拆前逐字相同（F0.1）──
    register_chat_routes(app)
    register_anthropic_routes(app)
    register_responses_routes(app)
    register_embeddings_routes(app)
    register_admin_routes(app)

    return app
