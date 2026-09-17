"""Memory Index consolidation worker（ADR-0009 §7；ADR-0027 §4.2 起为包内入口）。

为什么独立进程：consolidation 是重操作（扫 RocksDB + embedding 推理），跑在 proxy
进程里会竞争 GIL 与磁盘 IO，影响热路径延迟；独立进程完全隔离，proxy 崩溃不影响
已积累的 Memory Index 数据。

为什么在包里（ADR-0027 §4.2）：此前唯一入口是仓库脚本 `scripts/run_index_consolidator.py`，
**不在 wheel 里**——`pip install` / `uv tool install` 的用户永远启动不了 Memory Index 提炼，
拿到的是一个"永远不产生任何记忆的记忆产品"。现在逻辑在包内，脚本转薄壳。

用法：
    bladex-consolidator                      # console script
    python -m bladex_proxy.consolidator      # 等价
    python scripts/run_index_consolidator.py    # 薄壳（仓库内保持兼容）

    # 定向重建：只重放某窗口/某 agent，不清库、不动其余 Memory Index 数据。
    # 两步——先反消费（删 consumed 标记），再跑增量重放：
    bladex-consolidator --unconsume --since 2026-07-13 --until 2026-07-24
    bladex-consolidator --once --since 2026-07-13 --until 2026-07-24

环境变量（同 proxy）：
    BLADEX_INDEX_PATH         Memory Index 存储路径（默认 data/bladex_index）
    BLADEX_ROCKSDB_PATH    Memory Hub 存储路径（默认 data/bladex_hub）
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import structlog

from bladex_proxy import deployment

#: `_load_env_file()` 跑在 `_configure_logging()` 之前（它决定日志读什么配置），
#: 那时 structlog 还没配好。冲突先攒在这里，日志就绪后补一条结构化记录。
_pending_env_conflicts: list[deployment.EnvConflict] = []


# ── 存储锁冲突：把 rocksdict 的原始 IO error 翻成一句能照做的话 ────────────────
#
# 为什么值得单独处理（2026-08-05 实测）：`bladex-consolidator --unconsume` 在常驻
# consolidator 还活着时，甩出来的是
#     Exception: IO error: While lock file: data/bladex_index/meta_rocksdb/LOCK:
#     Resource temporarily unavailable
# 加一整段 traceback。这不是 bug——Memory Index 单写者是设计（ADR-0009 §7）——但用户看到的
# 是一串没有动作指向的堆栈，得读源码才知道"先停另一个进程"。发布形态上，凡是
# **预期内**的失败都必须给出下一步该敲什么。
_LOCK_MARKERS = (
    "lock file",                        # rocksdb 自己的措辞
    "resource temporarily unavailable",  # macOS / Linux flock errno
    "no locks available",                # 某些文件系统
)


def _lock_conflict_hint(exc: BaseException, store: str) -> str | None:
    """是存储锁冲突就返回可照做的提示，否则 None（调用方原样抛，别吞真错误）。

    `store` 取 "Memory Index" / "Memory Hub"——两者的占用方与解法不同：Memory Index 被另一个 consolidator 占，
    Memory Hub read-write 被 proxy 占（增量模式走 secondary 读，根本不需要锁）。
    """
    msg = str(exc).lower()
    if not any(m in msg for m in _LOCK_MARKERS):
        return None
    if store == "Memory Hub":
        return ("--full opens Memory Hub read-write, but the running proxy holds that lock. "
                "Stop the proxy first (`bladex stop`), run the full rebuild, then start it "
                "again. Incremental mode (without --full) reads Memory Hub as a secondary and needs "
                "no lock.")
    return ("another consolidator already holds the Memory Index write lock (single writer by design). "
            "Stop it first -- `bash config/run_consolidator.sh stop`, or `bladex stop` for the "
            "whole stack -- then re-run this command. `bladex status` shows what is running.")


def _load_env_file() -> None:
    """加载 config/.env 并把配置冲突报进日志（实现在 deployment.py，全仓库唯一）。

    2026-08-06 前这里是第三份各自为政的实现，而且是最弱的一份：**只认 cwd，
    连 `BLADEX_HOME` 都不认**，全局安装后 consolidator 永远读不到 `.env`。
    加上"不覆盖已设环境变量"的旧语义，当天真实事故 = 用户改了上游 base URL 与
    API key、重启 consolidator，进程仍读终端里的残留 export → 大量
    `distill_failed`（认证失败），提炼整段停摆而配置文件看上去完全正确。

    找不到部署根就跳过——env 也可能由 compose/systemd 直接注入。
    """
    root = deployment.find_root()
    if root is None:
        return
    conflicts = deployment.load_env_file(root)
    if not conflicts:
        return
    env_path = os.path.join(root, deployment.CONFIG_RELPATH)
    # 日志在 _configure_logging() 之前还没配好，用 print 到 stderr 保证看得见；
    # 之后再补一条结构化的（`config_env_shadowed` 是可 grep 的运营信号）。
    for line in deployment.format_conflicts(conflicts, env_path):
        print(line, file=sys.stderr)
    _pending_env_conflicts.extend(conflicts)


logger = structlog.get_logger()


def _configure_logging() -> None:
    """桥接 stdlib logging -> structlog 输出。

    distillation.py / consolidation_proxy.py 用 stdlib logging，不配置则它们的
    warning/error 不会出现在日志里。

    顺带接管上游模型 SDK 的日志（2026-08-05）：那一层此前在本进程里**完全没人管**——
    全局开关只写在 `route.py` 模块级，而 consolidator 根本不 import 它。实测一次运行
    的日志里 767 行是供应商自己的自言自语（含 100 行重复 + 指向别人 issue 页的求助链接）。
    `install_log_bridge()` 摘掉它自带的 handler、把署名换成 Router，INFO 级保留
    （"这轮发给了哪个模型"有运维价值，只是不该署别人的名）。
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] %(name)-28s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    from bladex_proxy import router_sdk

    router_sdk.install_log_bridge()


def _chdir_to_deployment_root() -> None:
    """切到部署根，让 `data/bladex_index`、`data/bladex_hub` 这些相对路径有唯一落点。

    consolidator 之前假设"总是从部署目录启动"（`os.getcwd()` 硬编码）。这个假设在
    `bladex start` 拉起它时成立，在用户手工 `bladex-consolidator` 时不成立——
    那时它会在别处新建一个空的 `data/`，看上去在跑、实际什么也不提炼。
    """
    root = deployment.find_root()
    if root is not None and root != os.getcwd():
        os.chdir(root)


def _log_env_conflicts() -> None:
    """日志就绪后补记配置冲突（`config_env_shadowed` 是可 grep 的运营信号）。"""
    for conflict in _pending_env_conflicts:
        logger.warning(
            "config_env_overridden" if conflict.applied else "config_env_shadowed",
            key=conflict.key,
            effective=deployment.mask(conflict.key, conflict.winner),
            ignored=deployment.mask(conflict.key, conflict.loser),
            source="config/.env" if conflict.applied else "environment",
        )
    _pending_env_conflicts.clear()


def resolve_concurrency(explicit: int) -> int:
    """蒸馏并发路数：显式 `--concurrency` 优先，否则读 flags（默认 4）。

    抽成纯函数是为了能被测试直接调——2026-08-22 之前这个回落**根本不存在**
    （`0 → None → 1`），而它埋在 `main()` 里，任何单测都够不着。
    取值依据与两次事故记录见 `bladex_core.flags` 的 `BLADEX_DISTILL_CONCURRENCY`。
    """
    if explicit >= 1:
        return explicit
    from bladex_core.flags import flag_number
    return max(1, int(flag_number("BLADEX_DISTILL_CONCURRENCY")))


def main() -> int:
    # 🔴 env 加载与日志配置必须在 main() 里，**不能在模块导入时做**。
    # 这曾经是脚本，模块级 `_load_env_file()` 无害；变成包内模块后，任何
    # `import bladex_proxy.consolidator` 都会把真实 config/.env 灌进 os.environ——
    # 2026-08-04 实测：一个只是 import 它的测试，污染了同一进程后续全部 test_server
    # 用例（上游模型变成 live 值、auth 变成开启 → 一片 401）。
    # 导入必须无副作用；副作用归入口点。
    _load_env_file()
    _chdir_to_deployment_root()
    _configure_logging()
    _log_env_conflicts()
    parser = argparse.ArgumentParser(description="BladeX Memory Index consolidation worker")
    parser.add_argument("--interval", type=float, default=60.0,
                        help="consolidation 轮询间隔（秒，默认 60）")
    parser.add_argument("--once", action="store_true",
                        help="跑一次后退出（用于 cron / 手动触发）")
    parser.add_argument("--max-turns", type=int, default=30,
                        help="每批最多处理的非 aux 轮次（默认 30；离线重放可给 200+，"
                             "在线保持默认防上游限流）")
    parser.add_argument("--full", action="store_true",
                        help="全量重建：清空 Memory Index → 重放 Memory Hub（ADR-0014 L0，用于阈值变更后重归属）；"
                             "带过滤参数时不清库，退化为定向重放")
    # 定向重建过滤（2026-07-28）：只处理匹配的 turn，不匹配的不消费也不蒸馏。
    parser.add_argument("--since", default="",
                        help="只处理 ts >= 该值的 turn（ISO 前缀，如 2026-07-13）")
    parser.add_argument("--until", default="",
                        help="只处理 ts <= 该值的 turn（ISO 前缀含端，如 2026-07-24 = 含当天）")
    parser.add_argument("--agents", default="",
                        help="只处理这些 agent 的 turn（逗号分隔，如 claude-code,hermes:default）")
    parser.add_argument("--exclude-agents", default="",
                        help="排除这些 agent 的 turn（逗号分隔，如压测流量 agent-0,agent-1,test-agent）")
    parser.add_argument("--reattribute", action="store_true",
                        help="按 AGENT_CLAIM 映射**只改归属、不重蒸**，然后退出（⑤）。"
                             "认领/改名/合并之后用它把存量 Fact 与 Matter 卡的 agent_id "
                             "改过来——这是一次 UPDATE，不碰内容/向量、不调 LLM，秒级完成。"
                             "🔴 不要用 `--unconsume --agents <桶>` 干这件事：那条路会"
                             "重新蒸馏出内容逐字相同的候选、被 novelty 判重丢弃"
                             "（verdict=vector_kill），归属改不成**而且不报错**。")
    parser.add_argument("--reattribute-dry-run", action="store_true",
                        help="配合 --reattribute：只统计会改多少，不落盘")
    parser.add_argument("--unconsume", action="store_true",
                        help="删除匹配 turn 的 consumed 标记后退出（不蒸馏）。"
                             "配合过滤参数用于'让某窗口被下轮增量重新拾起'；必须带至少一个过滤条件")
    parser.add_argument("--concurrency", type=int, default=0,
                        help="蒸馏并发路数（1=串行；0/省略 → BLADEX_DISTILL_CONCURRENCY，默认 4）。"
                             "历史：2026-08-13 之前 help 写'默认取 env'、实现却是 0→1 串行，"
                             "bench 全程串行蒸馏、排空超时、整跑数据作废——当时只改了 help 文案，"
                             "把缺陷记录成了规格。2026-08-22 同一个默认值打在生产上（CLI 起守护进程"
                             "只传 --interval），Memory Index 积压 720 轮不降。现已按 help 对齐实现："
                             "本进程读 env，`bladex` CLI 也显式传值。")
    parser.add_argument("--no-control", action="store_true",
                        help="关闭手工同步控制通道（默认常驻模式开启：CLI `bladex sync` 经 "
                             "Redis 下发任务、本进程执行并回写进度，见 bladex_proxy/sync_control.py）")
    args = parser.parse_args()

    def _split(v: str) -> set[str] | None:
        items = {s.strip() for s in v.split(",") if s.strip()}
        return items or None

    agents = _split(args.agents)
    exclude_agents = _split(args.exclude_agents)

    args.concurrency = resolve_concurrency(args.concurrency)

    has_filter = bool(args.since or args.until or agents or exclude_agents)
    if args.unconsume and not has_filter:
        parser.error("--unconsume 必须至少带一个过滤条件"
                     "（--since/--until/--agents/--exclude-agents），"
                     "否则等于清空所有消费标记 = 意外全量重放")

    from bladex_proxy.config import ProxyConfig
    from bladex_proxy.distillation import LLMDistiller
    from bladex_proxy.embedding import (
        build_embedder,
        effective_thresholds,
        embed_purpose,
        resolve_embed_settings,
        validate_embed_sensitivity,
    )
    from bladex_proxy.storage.memory_index import MemoryIndex, IndexDistillJournal
    from bladex_proxy.storage.memory_hub import MemoryHub

    cfg = ProxyConfig()

    # embedding 后端可选化（2026-07-26）：local（默认）/ api / proxy（共享 proxy 模型）。
    # 敏感层开 + api 档 -> strict 拒启动 / 告警+强制本地（与 proxy 同一 gate）。
    validate_embed_sensitivity(cfg)
    embed_settings = resolve_embed_settings(cfg)  # env > routing.toml [embedding] > 默认
    thresholds = effective_thresholds(cfg)  # 显式 env > per-model profile > 默认+告警

    # ADR-0021 section 2.5: legacy_ids -> principal 读侧映射（identity.toml 声明）。
    # empty registry（无 identity.toml）-> 空映射 = 现状（user_id 不变）。
    legacy_map = cfg.identity_registry.legacy_map()
    if legacy_map:
        logger.info("index_consolidator_legacy_map", entries=len(legacy_map))

    # ADR-0021 §2.3 修订（2026-08-16）：个人模式（无 identity.toml）折叠一切 user_id
    # 到单一本地身份。存量 turn 里的 key hash8 是"把凭证当身份"时代的产物，不是不同的人；
    # 不折叠则新流量落 `local`、存量留 hash8 -> 检索按 visibility 过滤当场两分裂。
    from bladex_proxy.identity import LOCAL_USER_ID  # 局部导入：模块级导入须无副作用
    fold_user_id = "" if not cfg.identity_registry.empty else LOCAL_USER_ID
    if fold_user_id:
        logger.info("index_consolidator_fold_user_id", fold_user_id=fold_user_id)

    logger.info("index_consolidator_starting",
                index_path=cfg.index_path, ledger_path=cfg.rocksdb_path,
                interval=args.interval, once=args.once, full=args.full,
                max_turns=args.max_turns,
                # 2026-08-22：此前不打，于是"守护进程在串行蒸馏"这件事在日志里
                # 完全不可见——只能靠数 `Router completion()` 的间隔反推。
                distill_concurrency=args.concurrency,
                since=args.since or None, until=args.until or None,
                agents=sorted(agents) if agents else None,
                exclude_agents=sorted(exclude_agents) if exclude_agents else None,
                unconsume=args.unconsume,
                embed_backend=embed_settings.backend,
                novelty_threshold=thresholds["novelty"],
                semantic_threshold=thresholds["semantic"],
                digestion_threshold=thresholds["digestion"])

    # 打开 Memory Hub 和 Memory Index（读写）
    # ADR-0020 T1: 增量 consolidation 改 secondary 模式读 Memory Hub（try_catch_up_with_primary
    #   追新 proxy 写入，修 read_only 点时快照看不到新轮的停滞 bug）+ 派生台账迁 Memory Index meta
    #   （consolidator 独占 Memory Index 写锁，修 read_only Memory Hub 台账 100% 写失败）。
    # full rebuild 仍 read_write（维护操作，应在 proxy 停止时跑）。
    if args.full:
        ledger = MemoryHub(cfg.rocksdb_path, read_only=False)
        ledger_mode = "read_write"
    else:
        ledger = MemoryHub(cfg.rocksdb_path, secondary=True)
        ledger_mode = "secondary"
    try:
        ledger.open()
    except Exception as e:
        hint = _lock_conflict_hint(e, "Memory Hub")
        if hint is None:
            raise
        logger.error("hub_locked_by_another_process", path=cfg.rocksdb_path,
                     error=str(e), hint=hint)
        return 1
    logger.info("index_consolidator_ledger_mode", mode=ledger_mode, full=args.full)

    embedder = build_embedder(cfg, role="consolidator")

    # L2: 蒸馏器（ADR-0014 + ADR-0018 §3.1 结构化 + §3.6 自举 + §4.1 台账缓存）
    # 模型自举：env > routing [distill] > 模型池最弱档（写回 routing.toml）。
    # 台账缓存：read_only Memory Hub 读命中复用；写降级，T11 full rebuild 时写生效。
    from bladex_core.distillation import PassthroughDistiller
    from bladex_proxy.routing_config import bootstrap_distill_model
    distill_model = bootstrap_distill_model(
        cfg.distill_model, cfg.routing_config, cfg.routing_config_path,
    )
    distill_journal = IndexDistillJournal()  # ADR-0020 T1.3: 台账迁 Memory Index meta，延迟 bind(index)
    if distill_model:
        distiller = LLMDistiller(
            model=distill_model,
            api_base=cfg.upstream_api_base,
            api_key=cfg.upstream_api_key,
            journal=distill_journal,
        )
    else:
        distiller = PassthroughDistiller()
        logger.warning("index_distill_disabled_passthrough",
                       hint="no distill model (env/routing/pool all empty)")

    # ADR-0018 §3.2/§3.6: L4 链接裁决器（复用蒸馏同一模型，少一个行为面）
    link_judge = None
    if distill_model:
        from bladex_proxy.linking import LLMLinkJudge
        link_judge = LLMLinkJudge(
            model=distill_model,
            api_base=cfg.upstream_api_base,
            api_key=cfg.upstream_api_key,
        )
    # M2-2：写入时裁决器（复用蒸馏同一模型，拍板 4 —— 与 L4 同一条理由：
    # 少一个行为面、少一处配置、少一次"到底是哪个模型判的"的排查）。
    adjudicator = None
    if distill_model:
        from bladex_proxy.adjudicator import LLMAdjudicator
        adjudicator = LLMAdjudicator(
            model=distill_model,
            api_base=cfg.upstream_api_base,
            api_key=cfg.upstream_api_key,
        )
    logger.info("index_distiller_configured", model=distill_model or "passthrough",
                link_judge=distill_model or "none",
                adjudicator=distill_model or "none")

    # 检查 embedder 是否可用（触发首次加载 local 模型 / probe api / probe 远端 proxy）。
    # 必须在 MemoryIndex.open() 之前：model_identity（尤其 proxy 档远端身份）要先就绪，
    # 才能做 Memory Index 向量空间 model_id 不变量校验。
    try:
        # 共享模型档：run_proxy.sh 先拉 consolidator 再起 uvicorn，等 proxy 就绪
        # （proxy 侧还要加载/下载模型），避免启动竞态直接退出。
        if hasattr(embedder, "wait_ready"):
            embedder.wait_ready(timeout_s=float(os.environ.get(
                "BLADEX_EMBED_PROXY_WAIT_S", "180")))
        with embed_purpose("warmup"):
            embedder.embed(["warmup"])
    except Exception as e:
        logger.error("index_consolidator_embedder_unavailable", error=str(e),
                      backend=embed_settings.backend,
                      hint="local: install model / check disk; api: check key+model; "
                           "proxy: check BLADEX_EMBED_PROXY_URL is a running BladeX proxy")
        ledger.close()
        return 1

    index = MemoryIndex(
        cfg.index_path, embedder=embedder,
        novelty_threshold=thresholds["novelty"],
        semantic_threshold=thresholds["semantic"],
        digestion_threshold=thresholds["digestion"],
        distiller=distiller,
        link_judge=link_judge,
        adjudicator=adjudicator,
        sensitivity_config=cfg.routing_config.sensitivity_config(),
        entity_aware_novelty=cfg.entity_aware_novelty,
        entity_overlap_threshold=cfg.entity_overlap_threshold,
        novelty_topk=cfg.novelty_topk,
        embed_model_id=getattr(embedder, "model_identity", None),
        entity_aware_rerank=cfg.entity_aware_rerank,
        entity_rerank_alpha=cfg.entity_rerank_alpha,
        entity_rerank_expand_k=cfg.entity_rerank_expand_k,
    )
    try:
        # 向量空间不一致时 RuntimeError 拒启动（提示 scripts/reembed_index.py）——那条
        # 自带指引，原样抛；这里只翻译锁冲突。
        index.open()
    except Exception as e:
        hint = _lock_conflict_hint(e, "Memory Index")
        if hint is None:
            raise
        logger.error("index_locked_by_another_consolidator", path=cfg.index_path,
                     error=str(e), hint=hint)
        ledger.close()
        return 1
    distill_journal.bind(index)  # ADR-0020 T1.3: 台账迁 Memory Index，绑定 index（consolidator 独占写）

    logger.info("index_consolidator_ready",
                model=getattr(embedder, "model_name", "?"),
                backend=embed_settings.backend)

    # ⑤ 归属重映射（2026-08-26）：按 AGENT_CLAIM 只改归属、不重蒸，然后退出。
    # 认领/改名/合并之后走这条，秒级完成；它是 UPDATE 不是重建。
    if args.reattribute or args.reattribute_dry_run:
        try:
            ledger.catch_up()
        except Exception:  # noqa: BLE001 —— secondary 追新失败不阻断
            pass
        try:
            stats = index.reattribute_by_claims(
                ledger, dry_run=bool(args.reattribute_dry_run))
        finally:
            index.close()
            ledger.close()
        print(f"reattribute{' (dry-run)' if args.reattribute_dry_run else ''}: "
              f"facts={stats['facts']} matters={stats['matters']} "
              f"claim_pairs={stats['pairs']}")
        return 0

    # 定向反消费（2026-07-28）：删匹配 turn 的 consumed 标记后退出，
    # 下轮增量 rebuild 会重新拾起这批 turn（不清库、novelty 判重、台账命中不重调 LLM）。
    #
    # 🔴 **它不能用来改归属**（2026-08-26 验证）：重放会蒸出内容逐字相同的候选，
    # 被 novelty 判重丢弃（`verdict=vector_kill`，`consolidation_proxy` 里直接
    # `continue`）⇒ 归属改不成**而且不报错**。改归属走 `--reattribute`。
    if args.unconsume:
        _claimed = set()
        try:
            from bladex_proxy.storage.memory_index import build_agent_claim_map
            _claimed = set(build_agent_claim_map(ledger))
        except Exception:  # noqa: BLE001 —— 只为提示，失败不阻断
            pass
        if agents and (agents & _claimed):
            # 静默无效是本仓的高发病：宁可多一条刺眼的告警。
            logger.warning(
                "unconsume_will_not_change_attribution",
                agents=sorted(agents & _claimed),
                hint="these agents have AGENT_CLAIM mappings; replay re-distills "
                     "identical content and novelty drops it (verdict=vector_kill). "
                     "Use `--reattribute` to change attribution.")
        try:
            ledger.catch_up()
            removed = index.unconsume_matching(
                ledger, since=args.since, until=args.until,
                agents=agents, exclude_agents=exclude_agents,
            )
            logger.info("index_consolidator_unconsume_done", removed=removed,
                        hint="下轮增量 rebuild 会重放这些 turn")
        finally:
            index.close()
            ledger.close()
        return 0

    # ── 手工同步控制通道（2026-08-03，Beta T11 前置）──────────────────────────
    # 常驻模式下开启：`bladex sync run` 经 Redis 下发任务，本进程接单执行、
    # 逐步回写进度（CLI 轮询渲染 + 落日志）。Redis 不可达 → 通道关闭（只告警）。
    control = None
    if not args.once and not args.full and not args.no_control:
        try:
            import redis as _redis
            from bladex_proxy.sync_control import SyncControl
            _rc = _redis.Redis.from_url(cfg.redis_url, decode_responses=True,
                                        socket_connect_timeout=2)
            _rc.ping()
            control = SyncControl(_rc)
            logger.info("index_consolidator_sync_control_ready", redis=cfg.redis_url)
        except Exception as e:  # noqa: BLE001
            logger.warning("index_consolidator_sync_control_unavailable", error=str(e),
                           hint="`bladex sync` 将走 direct 模式或不可用")

    # ── 在线协同导出 worker（Beta T13/B3.2；无 config/export.toml = 关闭零回归）──
    export_worker = None
    from bladex_proxy.modules import module_enabled
    if not module_enabled("export"):
        # V-A3（F0.2）：模块门只挡 worker 构造；下面的 sync_control 心跳一行不动（B10 仪器消费）。
        logger.info("export_module_disabled", hint="BLADEX_MODULE_EXPORT=0")
    else:
        try:
            from bladex_proxy.export_sync import build_worker_from_config
            export_worker = build_worker_from_config(
                index, ledger, "config/export.toml",
                hard_rules=cfg.effective_hard_rules)
            if export_worker is not None:
                logger.info("export_sync_worker_ready")
        except Exception as e:  # noqa: BLE001 —— 导出配置问题不拖垮 consolidator
            logger.warning("export_sync_worker_init_failed", error=str(e))

    def _run_manual(cmd: dict) -> None:
        """执行一条手工同步任务（CLI 下发），进度回写控制通道。"""
        from bladex_proxy.sync_control import make_progress_writer
        job_id = cmd["job_id"]
        t0 = time.perf_counter()

        def _s(v: str) -> set[str] | None:
            items = {x for x in (v or []) if x}
            return items or None

        control.set_status(job_id, state="running", phase="start",
                           worker="consolidator")
        logger.info("index_sync_job_start", job_id=job_id, cmd=cmd)

        # ⑤ 归属重映射（2026-08-26）：按 AGENT_CLAIM 只改归属、不重蒸。
        # 🔴 走这条通道而不是让 dashboard 自己动手——Memory Index 的写权在
        # consolidator（proxy 侧是只读 secondary）。秒级完成，不进重建那条路。
        if str(cmd.get("action") or "sync") == "reattribute":
            try:
                ledger.catch_up()
                stats = index.reattribute_by_claims(ledger)
                control.set_status(
                    job_id, state="done", phase="reattribute",
                    facts=str(stats["facts"]), matters=str(stats["matters"]),
                    claim_pairs=str(stats["pairs"]),
                    elapsed_s=f"{time.perf_counter() - t0:.1f}")
                logger.info("index_reattribute_job_done", job_id=job_id, **stats)
            except Exception as e:  # noqa: BLE001 —— 单个 job 失败不该杀掉守护进程
                control.set_status(job_id, state="failed", phase="reattribute",
                                   error=str(e))
                logger.error("index_reattribute_job_failed", job_id=job_id, error=str(e))
            return

        try:
            ledger.catch_up()
            with embed_purpose(f"sync_job:{job_id}"):
                nf = index.rebuild_from_hub(
                    ledger, full=bool(cmd.get("full")), legacy_map=legacy_map,
                    fold_user_id=fold_user_id,
                    max_turns=int(cmd.get("max_turns") or 0),
                    since=cmd.get("since") or "", until=cmd.get("until") or "",
                    agents=_s(cmd.get("agents")),
                    exclude_agents=_s(cmd.get("exclude_agents")),
                    distill_concurrency=int(cmd.get("concurrency") or 0) or None,
                    progress_cb=make_progress_writer(control, job_id),
                )
            elapsed = round(time.perf_counter() - t0, 1)
            control.set_status(job_id, state="done", phase="done",
                               new_facts=nf, elapsed_s=elapsed,
                               backlog=str(index.last_rebuild_backlog))
            logger.info("index_sync_job_done", job_id=job_id, new_facts=nf,
                        elapsed_s=elapsed, distiller_stats=distiller.stats())
        except Exception as e:  # noqa: BLE001 —— 任务失败不拖垮常驻进程
            control.set_status(job_id, state="failed", error=str(e))
            logger.error("index_sync_job_failed", job_id=job_id, error=str(e))

    def _sleep_or_cmd(seconds: float) -> dict | None:
        """可中断的 sleep：1s 粒度轮询控制通道，有任务立即返回。"""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if control is not None:
                cmd = control.poll_cmd()
                if cmd:
                    return cmd
            time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
        return None

    # ── 后台循环心跳（2026-08-06）──────────────────────────────────────────
    # 手工 job 一直有进度回写，后台循环没有：队列恢复后 487 轮积压在消化，而
    # `bladex status` 的 lag 只在每轮提交时跳一次、轮内纹丝不动，和"卡死了"长得一样。
    # 心跳把轮内进度（第几批 / 蒸馏 71/305）送出来，让"慢"和"停"能被分开。
    heartbeat = None
    if control is not None:
        from bladex_proxy.sync_control import make_heartbeat_writer
        heartbeat = make_heartbeat_writer(control)
        control.beat(state="starting", phase="init", batch=0,
                     interval_s=args.interval, max_turns=args.max_turns)

    # M4-2：写入侧漏斗的**独立** metrics 端点（默认关，BLADEX_CONSOLIDATOR_METRICS_PORT）。
    # 两个进程各自暴露（2026-08-06 拍板）——哪半边没数一眼可见，
    # 不必经 P2 中转（中转层本身会成为一个可能坏掉且难以分辨的环节）。
    from bladex_proxy.consolidator_metrics import start_metrics_server
    from bladex_proxy.metrics import Metrics

    funnel_metrics = Metrics()
    index.attach_funnel(funnel_metrics)
    metrics_server = start_metrics_server(funnel_metrics)

    batch_no = 0
    try:
        while True:
            # 周期开始先看有没有手工任务（比常规增量优先）
            if control is not None:
                cmd = control.poll_cmd()
                if cmd:
                    _run_manual(cmd)
                    continue

            t0 = time.perf_counter()
            batch_no += 1
            if control is not None:
                control.beat(state="running", phase="scan", batch=batch_no,
                             done=0, total=0)
            try:
                ledger.catch_up()  # ADR-0020 T1: secondary 追新 proxy 写入（非 secondary 空操作）
                with embed_purpose(f"rebuild:batch{batch_no}"):
                    new_facts = index.rebuild_from_hub(
                        ledger, full=args.full, legacy_map=legacy_map,
                        fold_user_id=fold_user_id,
                        max_turns=args.max_turns,
                        since=args.since, until=args.until,
                        agents=agents, exclude_agents=exclude_agents,
                        distill_concurrency=args.concurrency or None,
                        progress_cb=heartbeat,
                    )
                backlog = index.last_rebuild_backlog
                elapsed = (time.perf_counter() - t0) * 1000
                if control is not None:
                    control.beat(state="idle" if not backlog else "running",
                                 phase="batch_done", batch=batch_no,
                                 last_new_facts=new_facts,
                                 last_elapsed_s=round(elapsed / 1000, 1),
                                 backlog=str(backlog))
                if new_facts > 0:
                    logger.info("index_consolidator_done",
                                new_facts=new_facts, elapsed_ms=round(elapsed, 1),
                                backlog=backlog,
                                distiller_stats=distiller.stats())
                else:
                    logger.debug("index_consolidator_idle", elapsed_ms=round(elapsed, 1),
                                 backlog=backlog)
            except Exception as e:
                logger.error("index_consolidator_error", error=str(e))
                backlog = False  # 失败要退避，不立即空转重试
                if control is not None:
                    # 出错也要打心跳：不打的话读侧只看到进度停住，会当成"还在慢慢跑"
                    control.beat(state="error", phase="batch_failed",
                                 batch=batch_no, error=str(e)[:200])

            # T13：主管线之后跑导出（内部逐目标退避 + 永不外抛，不阻塞 rebuild）
            if export_worker is not None:
                try:
                    with embed_purpose("export_sync"):
                        export_worker.run_once()
                except Exception as e:  # noqa: BLE001
                    logger.warning("export_sync_worker_error", error=str(e))

            if args.once or args.full:
                if args.full:
                    logger.info("index_consolidator_full_rebuild_done")
                break
            # ③（2026-07-28）：有积压立即续跑，只在追平后才 sleep。
            # 旧行为每批固定睡 interval，6793 轮积压 = 多睡 ~3.8 小时。
            if backlog:
                continue
            if control is not None:
                control.beat(state="idle", phase="sleeping", batch=batch_no)
            cmd = _sleep_or_cmd(args.interval)
            if cmd:
                _run_manual(cmd)
    except KeyboardInterrupt:
        logger.info("index_consolidator_stopped")
        if control is not None:
            control.beat(state="stopped", phase="stopped", batch=batch_no)
    finally:
        if metrics_server is not None:
            metrics_server.shutdown()
        index.close()
        ledger.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
