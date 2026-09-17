"""Beta T9 只读 admin API —— CLI / dashboard / MCP 的共同数据地基。

全部挂 /admin/ 下、与既有管理端点同一 client key 鉴权（依赖注入自 server 传入，
避免循环 import）。纯读、不进热路径：

  - Memory Index 读走 proxy 进程内既有**只读句柄**（app.state.index，read_only=True），
    与 consolidator 独占写不冲突（ADR-0020 T1 secondary 追新语义）。
  - Memory Hub 读走 app.state.hub 主句柄（读操作不竞争写锁）。
  - /admin/status 是 /metrics（Prometheus 文本）的结构化 JSON 版。

数据语义权威：ADR-0009（分层存储）/ 0012（MatterGraph + 删除）/ 0018（DPL）。
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .metrics import DEGRADATION
from .storage.memory_hub import MemoryHub
from .storage.memory_index import MemoryIndex
from .sync_control import HEARTBEAT_KEY, JOB_PREFIX, LAST_JOB_KEY

logger = structlog.get_logger()

# 序列化排除项：向量不出 API（体积大且无展示价值）
_FACT_EXCLUDE = {"embedding"}
_MATTER_EXCLUDE = {"centroid", "embedding"}

_MAX_LIMIT = 500


def _err(status_code: int, message: str, err_type: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": err_type}},
    )


def _index_unavailable() -> JSONResponse:
    return _err(503, "Memory Index derived layer unavailable", "storage_error")


def _hub_unavailable() -> JSONResponse:
    return _err(503, "Memory Hub unavailable", "storage_error")


def _fact_dict(fact: Any) -> dict:
    return fact.model_dump(mode="json", exclude=_FACT_EXCLUDE)


def _matter_dict(matter: Any) -> dict:
    return matter.model_dump(mode="json", exclude=_MATTER_EXCLUDE)


def _clamp_limit(limit: int) -> int:
    if limit < 1:
        return 1
    return min(limit, _MAX_LIMIT)


#: 心跳多久没更新就不再当作"活着"。蒸馏一条长轮次可能几十秒，节流又是 1s，
#: 所以门槛给到分钟级；超过就只报 last_seen_s，让读的人自己判断，不替它宣称在跑。
_HEARTBEAT_FRESH_S = 180.0


async def _read_consolidator_heartbeat(pipeline: Any) -> dict | None:
    """读 consolidator 后台循环心跳（2026-08-06 新增，见 sync_control 模块 docstring）。

    两件事分开报，不合并成一个 bool：
      `fresh`        —— 心跳是不是新鲜的（>_HEARTBEAT_FRESH_S 就不新鲜）
      `last_seen_s`  —— 多久没动了

    因为"不新鲜"有两种可能——进程死了，或者卡在一次超长上游调用里——读侧凭年龄和
    state 自己判断，端点不替它下结论。老版本 consolidator 不写心跳，读不到就是 None，
    status 退回原来的样子（无心跳 ≠ 没在跑）。
    """
    if pipeline is None:
        return None
    try:
        raw = await pipeline.redis.hgetall(HEARTBEAT_KEY)
    except Exception as e:  # noqa: BLE001
        logger.debug("admin_status_heartbeat_read_failed", error=str(e))
        return None
    if not raw:
        return None
    hb: dict[str, Any] = {}
    for k, v in raw.items():
        ks = k.decode() if isinstance(k, bytes) else str(k)
        hb[ks] = v.decode() if isinstance(v, bytes) else str(v)
    try:
        age = max(0.0, time.time() - float(hb.get("ts", 0)))
    except (TypeError, ValueError):
        return hb
    hb["last_seen_s"] = round(age, 1)
    hb["fresh"] = age <= _HEARTBEAT_FRESH_S
    for name in ("done", "total", "batch"):
        try:
            hb[name] = int(float(hb[name]))
        except (KeyError, TypeError, ValueError):
            continue
    return hb


async def _read_sync_job(pipeline: Any) -> dict | None:
    """读最近一次手工同步 job 状态（sync_control 协议，aioredis 句柄复用 Pipeline）。

    后台循环的运行迹象看 `_read_consolidator_heartbeat`（2026-08-06 补上，此前这里的
    docstring 写的"consolidator 无独立心跳机制"已不再成立）；本函数只管手工 job。
    读失败返回 None，不影响 status 主体。
    """
    if pipeline is None:
        return None
    try:
        r = pipeline.redis
        last_id = await r.get(LAST_JOB_KEY)
        if not last_id:
            return None
        if isinstance(last_id, bytes):
            last_id = last_id.decode()
        raw = await r.hgetall(JOB_PREFIX + last_id)
        if not raw:
            return None
        job: dict[str, str] = {}
        for k, v in raw.items():
            ks = k.decode() if isinstance(k, bytes) else str(k)
            vs = v.decode() if isinstance(v, bytes) else str(v)
            job[ks] = vs
        job["job_id"] = last_id
        return job
    except Exception as e:  # noqa: BLE001 —— status 端点不因单一子系统失败而 500
        logger.debug("admin_status_sync_job_read_failed", error=str(e))
        return None


def _dir_bytes(path: str | Path) -> int | None:
    """目录递归占用字节数。路径不存在 → None（"没这个目录"≠"0 字节"）。"""
    p = Path(path)
    if not p.exists():
        return None
    if p.is_file():
        try:
            return p.stat().st_size
        except OSError:
            return None
    total = 0
    for root, _dirs, files in os.walk(p, onerror=lambda _e: None):
        for name in files:
            try:
                total += os.stat(os.path.join(root, name)).st_size
            except OSError:
                continue  # 扫描期间被 compaction 删掉的文件，跳过即可
    return total


# ── 运行时记忆配置 + 与 config/.env 的漂移检测（2026-08-05 事故驱动）──
#
# 事故形态（同一类踩了两次）：`config/.env` 写 `BLADEX_HOTPATH_BUDGET_MS=200`，
# 而**运行中的进程**用的是 150（那个终端里残留了一个 export；`_load_env_file`
# 的语义是"不覆盖已设环境变量"）。后果是每轮注入超时 1–3ms 被整段丢弃——
# 记忆静默失效，而唯一的痕迹是 warning 级日志混在一堆 info 里。
#
# 所以这里把**进程实际在用的值**暴露出来，并与 .env 声明的值比对。
# 关键在"进程实际在用的"：读 .env 再打印一遍毫无意义，那正是骗人的绿灯。

#: 值得盯的运行时参数（名字 → 从 ProxyConfig 取值的方式）
_WATCHED_NUMERIC = ("BLADEX_HOTPATH_BUDGET_MS", "BLADEX_INJECT_MAX_CHARS")


def _declared_env(root: Path | None = None) -> dict[str, str]:
    """解析 config/.env 里声明的值（**不**注入 os.environ——只读来比对）。

    2026-08-06：根从 `Path(__file__).parents[3]` 改为 `deployment.find_root()`。
    前者是源码树相对路径——装成 wheel 后指向 site-packages，那里永远没有
    `config/.env`，于是**漂移检测在所有已安装形态下恒为空**：这个专门为
    "配置对了但进程里是旧值"造的仪器，在最需要它的部署形态里从不报警。
    """
    from bladex_proxy import deployment

    try:
        base = str(root) if root is not None else deployment.find_root()
        if base is None:
            return {}
        return deployment.parse_env_file(Path(base) / deployment.CONFIG_RELPATH)
    except Exception:  # noqa: BLE001 —— 读不到就当没声明
        return {}


def _memory_config(cfg: Any) -> dict[str, Any]:  # noqa: ANN401
    """进程**实际生效**的记忆配置 + 与 .env 声明的漂移清单。"""
    from bladex_core.flags import (
        MAX_PREFETCH_K,
        MEMORY_FLAG_DEFAULTS,
        MEMORY_NUMERIC_DEFAULTS,
        flag_enabled,
        flag_number,
    )

    effective: dict[str, Any] = {
        "BLADEX_HOTPATH_BUDGET_MS": int(getattr(cfg, "hotpath_budget_ms", 0)),
        "BLADEX_INJECT_MAX_CHARS": int(getattr(cfg, "inject_max_chars", 0)),
    }
    try:
        effective["BLADEX_INJECT_TOPK"] = int(
            os.environ.get("BLADEX_INJECT_TOPK", "") or MAX_PREFETCH_K)
    except ValueError:
        effective["BLADEX_INJECT_TOPK"] = MAX_PREFETCH_K
    for name in MEMORY_NUMERIC_DEFAULTS:
        effective[name] = flag_number(name)
    flags = {name: flag_enabled(name) for name in sorted(MEMORY_FLAG_DEFAULTS)}

    # 漂移：.env 声明了、但进程实际用的是别的值
    declared = _declared_env()
    drift: list[dict[str, Any]] = []
    for name, live in effective.items():
        raw = declared.get(name)
        if raw is None:
            continue
        try:
            if float(raw) != float(live):
                drift.append({"key": name, "declared": raw, "effective": live})
        except ValueError:
            continue
    for name, live in flags.items():
        raw = declared.get(name)
        if raw is None:
            continue
        if (raw.strip().lower() not in ("0", "false", "no", "off", "")) != live:
            drift.append({"key": name, "declared": raw, "effective": live})
    return {"effective": effective, "flags": flags, "drift": drift}


#: 磁盘占用缓存（dashboard 每 5s 轮询一次 /admin/status，而 RocksDB 目录
#: 上千个 SST 文件 walk 一遍不便宜；60s 粒度对"磁盘涨没涨"完全够）
_DISK_TTL_S = 60.0
_disk_cache: dict[str, Any] = {"at": 0.0, "value": None}


def _flash_dir() -> str:
    from bladex_proxy.config import resolve_flash_path
    return resolve_flash_path()


def _disk_usage(cfg: Any, *, ttl_s: float = _DISK_TTL_S,
                now: float | None = None) -> dict[str, int | None]:
    """三层存储 + 日志的磁盘占用（ADR-0027 §4.4 / O3）。"""
    t = time.monotonic() if now is None else now
    cached = _disk_cache.get("value")
    if cached is not None and (t - _disk_cache["at"]) < ttl_s:
        return cached
    value = {
        "hub_bytes": _dir_bytes(getattr(cfg, "rocksdb_path", "data/bladex_hub")),
        "index_bytes": _dir_bytes(getattr(cfg, "index_path", "data/bladex_index")),
        "overflow_bytes": _dir_bytes(getattr(cfg, "overflow_dir", "data/overflow")),
        # V-F2：Flash 池/树占用（路径解析与 daemon 同源）
        "flash_bytes": _dir_bytes(_flash_dir()),
        "logs_bytes": _dir_bytes("logs"),
    }
    _disk_cache["value"] = value
    _disk_cache["at"] = t
    return value


def register_admin_read_routes(app: FastAPI, admin_dep: Any) -> None:
    """注册 T9 只读端点。admin_dep = server._admin_key_dep（同一鉴权面）。"""

    # ── GET /admin/status ──────────────────────────────────────────

    @app.get("/admin/status")
    async def admin_status(  # noqa: ANN202
        request: Request,
        cfg: Any = admin_dep,
    ) -> JSONResponse:
        """全景状态：三层水位 / consolidator 追新滞后 / 路由 source 分布 /
        上游健康 / 请求延迟分位。/metrics 的结构化 JSON 版。"""
        state = request.app.state

        # Pipeline 接入缓冲
        pipeline = getattr(state, "pipeline", None)
        pipeline_info: dict[str, Any] = {"connected": pipeline is not None}
        if pipeline is not None:
            try:
                # backlog（XLEN）三段拆分：undelivered / pending / orphan。
                # 光一个 backlog 会把"排不动"说成"排队中"（2026-08-06）——三个数全是
                # O(1) 命令（XLEN/XPENDING/XINFO），进每次轮询也不心疼。
                detail = await pipeline.inspect()
                pipeline_info["backlog"] = detail["total"]
                pipeline_info["pending"] = detail["pending"]
                pipeline_info["undelivered"] = detail["undelivered"]
                pipeline_info["orphan"] = detail["orphan"]
            except Exception as e:  # noqa: BLE001
                pipeline_info["backlog"] = None
                pipeline_info["error"] = str(e)
            pipeline_info["overflow_count"] = getattr(pipeline, "overflow_count", 0)
            pipeline_info["spill_count"] = getattr(pipeline, "spill_count", 0)

        # Memory Hub 完整总账
        hub: MemoryHub | None = getattr(state, "hub", None)
        hub_turns: int | None = None
        if hub is not None:
            try:
                hub_turns = hub.count()
            except Exception as e:  # noqa: BLE001
                logger.warning("admin_status_hub_count_failed", error=str(e))
        hub_info = {"available": hub is not None, "turns": hub_turns}

        # Memory Index 派生层 + consolidator 追新滞后
        index: MemoryIndex | None = getattr(state, "index", None)
        index_info: dict[str, Any] = {"available": index is not None}
        if index is not None:
            try:
                index_info["facts"] = index.fact_count()
                index_info["matters"] = len(index.all_matters())
                consumed = index.consumed_count()
                index_info["consumed_turns"] = consumed
                if hub_turns is not None:
                    # MQ-R5：`hub.count()` 已扣墓碑（与 scan_meta 同一套跳过规则），
                    # 两端口径一致后这个差才是真积压；此前恒虚报 500 轮。
                    # MQ-R3：两个计数各自会先 catch-up，不再是冻结快照。
                    index_info["lag_turns"] = max(0, hub_turns - consumed)
                index_info["rebuild_backlog"] = index.last_rebuild_backlog
            except Exception as e:  # noqa: BLE001
                index_info["error"] = str(e)

        # consolidator：最近手工同步 job + 后台循环心跳（轮内进度）
        sync_job = await _read_sync_job(pipeline)
        heartbeat = await _read_consolidator_heartbeat(pipeline)

        # 路由 source 分布 + 请求指标（metrics 快照）
        metrics = getattr(state, "metrics", None)
        route_sources: dict[str, float] = {}
        requests_summary: dict[str, Any] = {}
        if metrics is not None:
            snap = metrics.snapshot()
            for labels, val in snap["counters"].get("bladex_route_source_total", []):
                route_sources[labels.get("source", "?")] = val
            totals = snap["counters"].get("bladex_requests_total", [])
            requests_summary["total"] = sum(v for _, v in totals)
            requests_summary["by_label"] = [
                {**labels, "count": v} for labels, v in totals
            ]
            lat = snap["histograms"].get("bladex_request_latency_ms", [])
            if lat:
                # 汇总所有 label 组合的分位（admin 面粗粒度足够）
                requests_summary["latency_ms"] = lat[0][1] if len(lat) == 1 else {
                    "series": [{**labels, **pct} for labels, pct in lat]
                }

        # 上游健康（熔断表）
        health = getattr(state, "model_health", None)
        upstream = {
            "unhealthy_models": health.unhealthy_models() if health is not None else [],
        }

        # 静默降级的滚动窗口计数（ADR-0027 §3.3）——CLI 横幅与 dashboard 状态页消费。
        # 注意与 counter 的区别：这里回答的是"最近 1h 还在不在降级"，不是累计值。
        degradation = DEGRADATION.snapshot()

        # 磁盘占用（O3）：60s 缓存，见 _disk_usage
        try:
            disk = _disk_usage(cfg)
        except Exception as e:  # noqa: BLE001 —— status 不因磁盘扫描失败而 500
            logger.debug("admin_status_disk_failed", error=str(e))
            disk = {}

        from . import __version__
        return JSONResponse(content={
            "version": __version__,
            "pipeline": pipeline_info,
            "hub": hub_info,
            "index": index_info,
            "sync_job": sync_job,
            "consolidator": heartbeat,
            "routing": {"source_counts": route_sources},
            "upstream": upstream,
            "requests": requests_summary,
            "degradation": degradation,
            "disk": disk,
            # 进程实际在用的记忆配置 + 与 .env 的漂移（配置对了但进程里是旧值 = 静默失效）
            "memory_config": _memory_config(cfg),
        })

    # ── GET /admin/queue ───────────────────────────────────────────

    @app.get("/admin/queue")
    async def admin_queue(  # noqa: ANN202
        request: Request,
        cfg: Any = admin_dep,
    ) -> JSONResponse:
        """Pipeline 队列的分段水位（供 `bladex queue status`）。

        /admin/status 里的 pipeline 段只是它的摘要；这里单列一个端点，是因为
        "队列卡住了"值得有个直接的问法，不必让人从全景 JSON 里挑三个字段。
        """
        pipeline = getattr(request.app.state, "pipeline", None)
        if pipeline is None:
            return JSONResponse(
                status_code=503,
                content={"error": {"message":
                        "pipeline unavailable -- Redis is likely down;"
                        " turns are spilling to disk until it is back (ADR-0017)",
                        "type": "unavailable"}},
            )
        try:
            detail = await pipeline.inspect()
        except Exception as e:  # noqa: BLE001
            return JSONResponse(
                status_code=503,
                content={"error": {"message": f"queue inspect failed: {e}",
                        "type": "unavailable"}},
            )
        detail["overflow_count"] = getattr(pipeline, "overflow_count", 0)
        detail["spill_count"] = getattr(pipeline, "spill_count", 0)
        return JSONResponse(content=detail)

    # ── GET /admin/hard_rules ──────────────────────────────────────

    @app.get("/admin/hard_rules")
    async def admin_hard_rules(  # noqa: ANN202
        request: Request,
        cfg: Any = admin_dep,
    ) -> JSONResponse:
        rules = list(getattr(cfg, "effective_hard_rules", None) or [])
        return JSONResponse(content={"hard_rules": rules, "count": len(rules)})

    # ── GET /admin/facts ───────────────────────────────────────────

    @app.get("/admin/facts")
    async def admin_facts(  # noqa: ANN202
        request: Request,
        q: str = "",
        limit: int = 50,
        offset: int = 0,
        scope: str = "",
        exposure: str = "",
        origin: str = "",
        kind: str = "",
        item_kind: str = "",
        user_id: str = "",
        trust_min: float = 0.0,
        current_only: bool = False,
        cfg: Any = admin_dep,
    ) -> JSONResponse:
        """Fact 浏览：分页列表（默认）或语义搜索（q 非空且 embedder 就绪）。

        过滤器：scope / exposure(=exposure_ceiling) / origin(tags 含 origin:<x>)
        / kind / item_kind / user_id / trust_min / current_only(t_invalid 为空)。
        """
        index: MemoryIndex | None = request.app.state.index
        if index is None:
            return _index_unavailable()
        limit = _clamp_limit(limit)
        offset = max(0, offset)

        def _match(f: Any) -> bool:
            if scope and getattr(f, "scope", "") != scope:
                return False
            if exposure and getattr(f, "exposure_ceiling", "") != exposure:
                return False
            if origin and f"origin:{origin}" not in (getattr(f, "tags", "") or ""):
                return False
            if kind and getattr(f, "kind", "") != kind:
                return False
            if item_kind:
                fk = getattr(f, "item_kind", "")
                if str(getattr(fk, "value", fk)) != item_kind:
                    return False
            if user_id and getattr(f, "source_user_id", "") != user_id:
                return False
            if trust_min and getattr(f, "trust", 1.0) < trust_min:
                return False
            if current_only and getattr(f, "t_invalid", None) is not None:
                return False
            return True

        mode = "list"
        if q:
            # 语义搜索（embedder 缺失时 index.search 返回空 → 退化为空结果而非 500）
            try:
                found = index.search(q, top_k=limit + offset,
                                  user_id=user_id or None)
            except Exception as e:  # noqa: BLE001
                logger.warning("admin_facts_search_failed", error=str(e))
                found = []
            facts = [f for f in found if _match(f)]
            total = len(facts)
            page = facts[offset:offset + limit]
            mode = "search"
        else:
            try:
                all_f = [f for f in index.all_facts() if _match(f)]
            except Exception as e:  # noqa: BLE001
                logger.warning("admin_facts_list_failed", error=str(e))
                return _err(500, f"facts listing failed: {e}", "storage_error")
            # 新的在前（created_at 降序）
            all_f.sort(key=lambda f: str(getattr(f, "created_at", "")), reverse=True)
            total = len(all_f)
            page = all_f[offset:offset + limit]

        return JSONResponse(content={
            "facts": [_fact_dict(f) for f in page],
            "total": total,
            "limit": limit,
            "offset": offset,
            "mode": mode,
        })

    @app.get("/admin/facts/{fact_id}")
    async def admin_fact_detail(  # noqa: ANN202
        request: Request,
        fact_id: str,
        cfg: Any = admin_dep,
    ) -> JSONResponse:
        """单条 Fact 详情（Beta T11 `bladex memory show`）。"""
        index: MemoryIndex | None = request.app.state.index
        if index is None:
            return _index_unavailable()
        fact = index.get_fact(fact_id)
        if fact is None:
            return _err(404, "Fact not found", "not_found")
        return JSONResponse(content=_fact_dict(fact))

    # ── GET /admin/matters + /admin/matters/{id} ───────────────────

    @app.get("/admin/matters")
    async def admin_matters(  # noqa: ANN202
        request: Request,
        status: str = "",
        limit: int = 100,
        offset: int = 0,
        cfg: Any = admin_dep,
    ) -> JSONResponse:
        index: MemoryIndex | None = request.app.state.index
        if index is None:
            return _index_unavailable()
        limit = _clamp_limit(limit)
        offset = max(0, offset)
        try:
            matters = index.all_matters()
        except Exception as e:  # noqa: BLE001
            return _err(500, f"matters listing failed: {e}", "storage_error")
        if status:
            matters = [m for m in matters if str(m.status.value) == status]
        matters.sort(key=lambda m: str(getattr(m, "updated_at", "")), reverse=True)
        total = len(matters)
        page = matters[offset:offset + limit]
        return JSONResponse(content={
            "matters": [_matter_dict(m) for m in page],
            "total": total,
            "limit": limit,
            "offset": offset,
        })

    # 🔴 路由顺序：必须注册在 /admin/matters/{matter_id} **之前**，
    # 否则 "merge_candidates" 会被当成 matter_id 吞掉。
    @app.get("/admin/matters/merge_candidates")
    async def admin_matter_merge_candidates(  # noqa: ANN202
        request: Request,
        limit: int = 50,
        cfg: Any = admin_dep,
    ) -> JSONResponse:
        """合并提案（matter-dedup-keywords 卡拍板 C）：主题键高重叠的 Matter 对。

        只读检测——合并本身永远走 POST /admin/matters/{id}/merge（manual 主权 +
        MATTER_MERGE 管理事件，重建可重放）。误合并率=0 红线不许自动合并。
        """
        index: MemoryIndex | None = request.app.state.index
        if index is None:
            return _index_unavailable()
        try:
            pairs = index.matter_merge_candidates(max_pairs=_clamp_limit(limit))
        except Exception as e:  # noqa: BLE001
            return JSONResponse(status_code=503, content={"error": {
                "message": f"merge candidate scan failed: {e}", "type": "unavailable"}})
        return JSONResponse(content={"candidates": pairs, "count": len(pairs)})

    @app.get("/admin/matters/{matter_id}")
    async def admin_matter_detail(  # noqa: ANN202
        request: Request,
        matter_id: str,
        facts_top_k: int = 50,
        cfg: Any = admin_dep,
    ) -> JSONResponse:
        """Matter 详情：卡片全字段 + 归属边（含 provenance/decision 审计）+ 成员 facts。"""
        index: MemoryIndex | None = request.app.state.index
        if index is None:
            return _index_unavailable()
        matter = index.get_matter(matter_id)
        if matter is None:
            return _err(404, "Matter not found", "not_found")
        edges = index.get_edges(matter_id)
        try:
            facts = index.get_facts_for_matter(matter_id, top_k=_clamp_limit(facts_top_k))
        except Exception as e:  # noqa: BLE001
            logger.warning("admin_matter_facts_failed", error=str(e))
            facts = []
        session_keys = [
            e.target_key for e in edges if str(e.target_type.value) == "session"
        ]
        return JSONResponse(content={
            "matter": _matter_dict(matter),
            "edges": [e.model_dump(mode="json") for e in edges],
            "facts": [_fact_dict(f) for f in facts],
            "session_keys": session_keys,
            "edge_count": len(edges),
            "manual_edge_count": sum(
                1 for e in edges if str(e.provenance.value) == "manual"
            ),
        })

    # ── GET /admin/sessions + /admin/turns/{key} ───────────────────

    @app.get("/admin/sessions")
    async def admin_sessions(  # noqa: ANN202
        request: Request,
        prefix: str = "",
        agent: str = "",
        limit: int = 50,
        offset: int = 0,
        cfg: Any = admin_dep,
    ) -> JSONResponse:
        """Memory Hub 会话浏览：scan_meta 惰性聚合（key = user/agent/session/entry），
        按 last_ts 降序。不做完整 Turn 反序列化（性能：仅 key+ts）。"""
        hub: MemoryHub | None = request.app.state.hub
        if hub is None:
            return _hub_unavailable()
        limit = _clamp_limit(limit)
        offset = max(0, offset)
        sessions: dict[str, dict[str, Any]] = {}
        try:
            for key, ts in hub.scan_meta(prefix=prefix):
                parts = key.split("/")
                if len(parts) < 4:
                    continue
                if agent and parts[1] != agent:
                    continue
                sess_prefix = "/".join(parts[:3]) + "/"
                s = sessions.get(sess_prefix)
                if s is None:
                    s = {
                        "session": sess_prefix,
                        "user_id": parts[0],
                        "agent_id": parts[1],
                        "session_id": parts[2],
                        "turns": 0,
                        "first_ts": ts,
                        "last_ts": ts,
                    }
                    sessions[sess_prefix] = s
                s["turns"] += 1
                if ts:
                    if not s["first_ts"] or ts < s["first_ts"]:
                        s["first_ts"] = ts
                    if ts > s["last_ts"]:
                        s["last_ts"] = ts
        except Exception as e:  # noqa: BLE001
            return _err(500, f"session scan failed: {e}", "storage_error")
        ordered = sorted(sessions.values(), key=lambda s: s["last_ts"], reverse=True)
        total = len(ordered)
        page = ordered[offset:offset + limit]
        return JSONResponse(content={
            "sessions": page, "total": total, "limit": limit, "offset": offset,
        })

    @app.get("/admin/turns")
    async def admin_turns_index(  # noqa: ANN202
        request: Request,
        prefix: str = "",
        limit: int = 100,
        offset: int = 0,
        cfg: Any = admin_dep,
    ) -> JSONResponse:
        """turn key 列表（Beta T20 回放页数据源）：scan_meta 惰性，按 ts 降序。

        prefix 通常是 /admin/sessions 返回的会话前缀（user/agent/session/）。
        """
        hub: MemoryHub | None = request.app.state.hub
        if hub is None:
            return _hub_unavailable()
        limit = _clamp_limit(limit)
        offset = max(0, offset)
        try:
            items = [{"key": k, "ts": ts} for k, ts in hub.scan_meta(prefix=prefix)]
        except Exception as e:  # noqa: BLE001
            return _err(500, f"turn scan failed: {e}", "storage_error")
        items.sort(key=lambda x: x["ts"], reverse=True)
        total = len(items)
        return JSONResponse(content={
            "turns": items[offset:offset + limit],
            "total": total, "limit": limit, "offset": offset,
        })

    @app.get("/admin/turns/{turn_key:path}")
    async def admin_turn_detail(  # noqa: ANN202
        request: Request,
        turn_key: str,
        include_messages: bool = False,
        cfg: Any = admin_dep,
    ) -> JSONResponse:
        """单 turn 回放：身份 / 注入内容 / 路由决策（decision_meta）/ 响应元数据。

        include_messages=true 时附带 request_messages + response_text
        （content_ref 已由 Memory Hub get() 还原）。默认不带（体积可达数百 KB）。
        """
        hub: MemoryHub | None = request.app.state.hub
        if hub is None:
            return _hub_unavailable()
        turn = hub.get(turn_key)
        if turn is None:
            return _err(404, "Turn not found", "not_found")
        exclude: set[str] = set()
        if not include_messages:
            exclude = {"request_messages", "response_text"}
        data = turn.model_dump(mode="json", exclude=exclude)
        data["key"] = turn_key
        return JSONResponse(content=data)

    # ── ADR-0028 其余四类记忆（E7.1 文件索引 / E7.2 项目 / E7.3 画像 / E7.4 工具习惯）──
    # 普通会话类走上面 facts/matters/sessions；这四组让五类在 admin 面（dashboard/CLI/MCP）
    # 都可见。全部只读——写入只有 consolidator（独占写语义不变）。

    def _file_row(r: dict) -> dict:
        r.pop("vector", None)
        kw = r.get("keywords")
        if isinstance(kw, str):
            r["keywords"] = [w for w in kw.split() if w]
        return r

    @app.get("/admin/files")
    async def admin_files(  # noqa: ANN202
        request: Request,
        q: str = "",
        mime_class: str = "",
        origin: str = "",
        agent: str = "",
        user_id: str = "",
        limit: int = 50,
        offset: int = 0,
        cfg: Any = admin_dep,
    ) -> JSONResponse:
        """文件索引浏览：列表（默认，按 path 排序）或语义搜索（q 非空且 embedder 就绪）。

        过滤器：mime_class(doc/code/image/av/other) / origin(read/produced)
        / agent(前缀匹配 agent_id) / user_id。
        """
        index: MemoryIndex | None = request.app.state.index
        if index is None:
            return _index_unavailable()
        limit = _clamp_limit(limit)
        offset = max(0, offset)

        def _match(r: dict) -> bool:
            if mime_class and r.get("mime_class") != mime_class:
                return False
            if origin and str(r.get("origin", "")) != origin:
                return False
            if agent and not str(r.get("agent_id", "")).startswith(agent):
                return False
            if user_id and r.get("user_id") != user_id:
                return False
            return True

        mode = "list"
        try:
            if q:
                vec = index.embed_query_vec(q)
                rows = (index.search_files(vec, k=limit + offset,
                                           user_id=user_id or None)
                        if vec is not None else [])
                mode = "search"
            else:
                rows = index.list_file_entries()
                rows.sort(key=lambda r: str(r.get("path", "")))
        except Exception as e:  # noqa: BLE001
            logger.warning("admin_files_list_failed", error=str(e))
            return _err(500, f"file listing failed: {e}", "storage_error")
        rows = [r for r in rows if _match(r)]
        total = len(rows)
        page = [_file_row(dict(r)) for r in rows[offset:offset + limit]]
        return JSONResponse(content={
            "files": page, "total": total, "limit": limit, "offset": offset,
            "mode": mode,
        })

    @app.get("/admin/files/{file_id}")
    async def admin_file_detail(  # noqa: ANN202
        request: Request,
        file_id: str,
        include_body: bool = False,
        cfg: Any = admin_dep,
    ) -> JSONResponse:
        """单条文件索引；include_body=true 回溯 PRODUCED 副本正文（blob store）。"""
        index: MemoryIndex | None = request.app.state.index
        if index is None:
            return _index_unavailable()
        row = index.get_file_entry(file_id)
        if row is None:
            return _err(404, "File entry not found", "not_found")
        row = _file_row(dict(row))
        if include_body:
            row["body"] = index.get_file_body(file_id)
        return JSONResponse(content=row)

    @app.get("/admin/projects")
    async def admin_projects(  # noqa: ANN202
        request: Request,
        limit: int = 100,
        offset: int = 0,
        cfg: Any = admin_dep,
    ) -> JSONResponse:
        """项目库（E7.2）：Matter 的容器，按 updated_at 降序。"""
        index: MemoryIndex | None = request.app.state.index
        if index is None:
            return _index_unavailable()
        limit = _clamp_limit(limit)
        offset = max(0, offset)
        try:
            projects = index.all_projects()
        except Exception as e:  # noqa: BLE001
            return _err(500, f"project listing failed: {e}", "storage_error")
        projects.sort(key=lambda p: str(getattr(p, "updated_at", "")), reverse=True)
        total = len(projects)
        page = projects[offset:offset + limit]
        return JSONResponse(content={
            "projects": [p.model_dump(mode="json") for p in page],
            "total": total, "limit": limit, "offset": offset,
        })

    @app.get("/admin/projects/{project_id}")
    async def admin_project_detail(  # noqa: ANN202
        request: Request,
        project_id: str,
        cfg: Any = admin_dep,
    ) -> JSONResponse:
        index: MemoryIndex | None = request.app.state.index
        if index is None:
            return _index_unavailable()
        p = index.get_project(project_id)
        if p is None:
            return _err(404, "Project not found", "not_found")
        return JSONResponse(content=p.model_dump(mode="json"))

    @app.get("/admin/profiles")
    async def admin_profiles(  # noqa: ANN202
        request: Request,
        cfg: Any = admin_dep,
    ) -> JSONResponse:
        """画像总览（E7.3/E7.4）：USER.md + 分 agent（习惯文件 / 规则文件副本 / 工具统计）。

        agent 之间相互隔离是产出侧语义（audience / per-base key）；这里是管理面全景。
        """
        index: MemoryIndex | None = request.app.state.index
        if index is None:
            return _index_unavailable()
        try:
            user_md, _ = index.get_profile_md(None)
            agents = []
            for base in index.list_profile_agents():
                _, agent_md = index.get_profile_md(None, base)
                agents.append({
                    "agent_base": base,
                    "agent_md": agent_md,
                    "rule_files": index.get_rule_files(base),
                    "tools": index.get_tool_stats(base, top=50),
                })
        except Exception as e:  # noqa: BLE001
            logger.warning("admin_profiles_failed", error=str(e))
            return _err(500, f"profile listing failed: {e}", "storage_error")
        return JSONResponse(content={
            "user_md": user_md, "agents": agents, "agent_count": len(agents),
        })

    @app.get("/admin/profiles/{agent_base}/rulefiles/{name}")
    async def admin_rule_file_detail(  # noqa: ANN202
        request: Request,
        agent_base: str,
        name: str,
        cfg: Any = admin_dep,
    ) -> JSONResponse:
        """单个规则文件副本正文（CLAUDE.md / AGENTS.md 等，分 agent 捕获）。"""
        index: MemoryIndex | None = request.app.state.index
        if index is None:
            return _index_unavailable()
        rec = index.get_rule_file(agent_base, name)
        if not rec:
            return _err(404, "Rule file copy not found", "not_found")
        rec = dict(rec)
        rec["agent_base"] = agent_base
        rec["name"] = name
        return JSONResponse(content=rec)

    @app.get("/admin/tools")
    async def admin_tools(  # noqa: ANN202
        request: Request,
        agent: str = "",
        top: int = 50,
        cfg: Any = admin_dep,
    ) -> JSONResponse:
        """工具习惯（E7.4）：分 agent 的 calls/ok/err/last_err_class，agent 间隔离。"""
        index: MemoryIndex | None = request.app.state.index
        if index is None:
            return _index_unavailable()
        top = _clamp_limit(top)
        try:
            if agent:
                agents = {agent: index.get_tool_stats(agent, top=top)}
            else:
                agents = {b: index.get_tool_stats(b, top=top)
                          for b in index.list_tool_agents()}
        except Exception as e:  # noqa: BLE001
            return _err(500, f"tool stats failed: {e}", "storage_error")
        return JSONResponse(content={"agents": agents})
