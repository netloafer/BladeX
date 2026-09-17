"""`/admin/*` 写端点 + `/dashboard` + admin 鉴权与 Memory Index 写句柄。

09-06 F0.1 自 `server.py` 拆出，零行为（函数体逐字搬家）。门面 `bladex_proxy.server` re-export。
端点原是 `create_app` 的闭包（只捕获 `app`），现挂在 `register_admin_routes(app)` 下，函数体不变。
只读 admin API 仍由 `bladex_proxy.admin_read.register_admin_read_routes` 提供，在本模块内按原顺序登记。
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from typing import Any

import structlog
from bladex_core.fact import Fact
from bladex_core.matter import EdgeTargetType, Matter, MatterOrigin, MatterStatus
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from bladex_proxy.admin_read import register_admin_read_routes
from bladex_proxy.config import ProxyConfig
from bladex_proxy.identity import _SYSTEM_ROLES, LOCAL_USER_ID
from bladex_proxy.models import (
    AdminEventType,
    AgentClaimBody,
    AgentRuleBody,
    AssignMatterBody,
    CreateMatterBody,
    DetachEdgeBody,
    MergeMattersBody,
    PromoteScopeBody,
    RememberFactBody,
    RenameMatterBody,
    SplitMatterBody,
    TombstoneSource,
    TombstoneTargetType,
)
from bladex_proxy.modules import module_enabled
from bladex_proxy.server.orchestration import _extract_bearer
from bladex_proxy.storage.memory_hub import MemoryHub
from bladex_proxy.storage.memory_index import MemoryIndex, build_agent_claim_map

logger = structlog.get_logger()


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


def register_admin_routes(app: FastAPI) -> None:
    """挂 `/admin/*` 写端点、只读 admin API 与 `/dashboard`（登记顺序与拆前逐字相同）。"""
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

        采纳 = 用户版进 agency 内存池（注入块/工具面立刻可见）**+ 落 Hub 一条
        `LEDGER_USER_EDIT` 事件**（MQ-L51，2026-09-06）。事件走 `AgencyRuntime._emit`——
        与 `ledger_update` 同一条路（Hub 追加 + Flash 门铃），`replay_ledger_events`
        早已把 `ledger_user_edit` 与 create/update 同等重放（`ledger.py`）。
        修前只进池不写 Hub：模型此后不写这本 ⇒ Hub 停在采纳前 ⇒ 每次 proxy 重启池回旧版
        ⇒ daemon 对账重采 ⇒ 一条假 `clobber_risk`（09-06 11:06 / 11:33 两次实录各 2 本）。
        日志 `admin_ledger_user_edit_adopted` 是追溯锚（谁的版本、何时采纳）。

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
            # MQ-L51：采纳落 Hub（重放等价 ⇒ 重启后池仍是采纳版）。桩 agency 无 `_emit` 时跳过。
            _emit = getattr(agency, "_emit", None)
            if callable(_emit):
                _emit(AdminEventType.LEDGER_USER_EDIT, {"ledger": led.model_dump()})
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
        from bladex_core.ledger import iso_ms  # 数值键，不比 ISO 串（MQ-L46）
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
        import datetime as _dt

        from bladex_core.ledger_runtime import bind_matter, ledger_anchor_matter_id
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

    # ── V-A3（F0.2）：只读 admin API + dashboard 归 `admin_read` 模块，默认开 = 现状；
    #    关 ⇒ 这两组路由**不登记**（404），上面的写端点照常。登记时机 = create_app，
    #    与其它模块的"每请求读 env"不同——路由表是进程级的，开关也只在启动时生效。
    if not module_enabled("admin_read"):
        logger.info("admin_read_module_disabled", hint="BLADEX_MODULE_ADMIN_READ=0")
        return

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
