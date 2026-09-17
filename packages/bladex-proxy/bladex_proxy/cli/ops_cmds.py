"""运维命令：`sync` / `config` / `queue` / `sticky` / `connector` / `backup` / `restore` / `export` / `import`。

09-06 F0.1 自 `cli.py` 拆出，零行为（函数体逐字搬家）。共享 helper（`_admin_call` / `_http_json` / `_probe_ready` / pidfile 族…）留在门面 `bladex_proxy.cli`；本模块经 `from bladex_proxy.cli import …` 取用，
🔴 故 monkeypatch 要打在**本模块**上（`bladex_proxy.cli.<本模块>.<helper>`），打门面不生效。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import typer

from bladex_proxy import deployment
from bladex_proxy.cli import (  # noqa: E402
    _PROXY_PIDFILE,
    _admin_call,
    _consolidator_pid,
    _read_pidfile,
    _repo_root,
    _resolve_distill_concurrency,
    _run_repo_script,
    app,
)


class _SyncLog:
    """终端 + 日志双写：进度单行刷新（终端）+ 逐条追加（文件）。"""

    def __init__(self, path: Path) -> None:
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(path, "a", encoding="utf-8")  # noqa: SIM115 —— 生命周期随 CLI
        self._last_file_write = 0.0

    def line(self, msg: str) -> None:
        print(msg)
        self._f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
        self._f.flush()

    def progress(self, msg: str) -> None:
        print(f"\r  {msg:<76}", end="", flush=True)
        now = time.monotonic()
        if now - self._last_file_write >= 1.0:  # 文件里 1 行/秒，防日志膨胀
            self._last_file_write = now
            self._f.write(f"{time.strftime('%H:%M:%S')}   {msg}\n")
            self._f.flush()

    def progress_end(self) -> None:
        print()

    def close(self) -> None:
        self._f.close()


def _fmt_progress(st: dict) -> str:
    phase = st.get("phase", "?")
    done, total = st.get("done", "?"), st.get("total", "")
    extra = ""
    if st.get("new_facts"):
        extra = f" new_facts={st['new_facts']}"
    if total and total not in ("0", ""):
        try:
            pct = 100.0 * float(done) / float(total)
            return f"[{phase}] {done}/{total} ({pct:.0f}%){extra}"
        except (ValueError, ZeroDivisionError):
            pass
    return f"[{phase}] {done}{extra}"


# ── delegate 模式：经 Redis 下发给运行中的 consolidator ──────────────────────


def _try_delegate(args: argparse.Namespace, log: _SyncLog) -> int | None:
    """返回退出码；None = 控制面不可用/没人接单 → 调用方转 direct。"""
    from bladex_proxy.config import ProxyConfig
    from bladex_proxy.sync_control import SyncControl
    try:
        import redis
        rc = redis.Redis.from_url(ProxyConfig().redis_url, decode_responses=True,
                                  socket_connect_timeout=2)
        rc.ping()
    except Exception as e:  # noqa: BLE001
        log.line(f"· Redis unreachable ({e}) -> falling back to direct mode")
        return None

    control = SyncControl(rc)
    job_id = control.submit(
        full=args.full, concurrency=args.concurrency, max_turns=args.max_turns,
        since=args.since, until=args.until,
        agents=[s for s in args.agents.split(",") if s.strip()],
        exclude_agents=[s for s in args.exclude_agents.split(",") if s.strip()],
    )
    log.line(f"· Job submitted job_id={job_id}, waiting for a running consolidator to pick it up"
             f"({args.pickup_timeout}s timeout -> direct mode)...")

    deadline = time.monotonic() + args.pickup_timeout
    picked = False
    while time.monotonic() < deadline:
        st = control.get_status(job_id)
        if st.get("state") in ("running", "done", "failed"):
            picked = True
            break
        time.sleep(0.5)
    if not picked:
        log.line("· Nobody picked it up (consolidator not running, or too old to have the control channel) -> direct mode")
        return None

    log.line("· consolidator picked it up, running (Ctrl-C only detaches this view; the job keeps going; "
             "check later with `sync status`)")
    last_ts = ""
    while True:
        st = control.get_status(job_id)
        state = st.get("state", "?")
        if st.get("ts") != last_ts:
            last_ts = st.get("ts", "")
            log.progress(_fmt_progress(st))
        if state in ("done", "failed"):
            log.progress_end()
            if state == "done":
                log.line(f"✅ Sync complete: new_facts={st.get('new_facts', '?')} "
                         f"elapsed={st.get('elapsed_s', '?')}s backlog={st.get('backlog', '?')}")
                return 0
            log.line(f"❌ Sync failed: {st.get('error', 'unknown error')} (see the consolidator log)")
            return 1
        time.sleep(1.0)


# ── direct 模式：本进程执行（consolidator 已停；proxy 可继续跑）──────────────


def _run_direct(args: argparse.Namespace, log: _SyncLog) -> int:
    for p in ("packages/bladex-core", "packages/bladex-proxy"):
        full = os.path.join(_repo_root(), p)
        if os.path.isdir(full) and full not in sys.path:
            sys.path.insert(0, full)

    from bladex_core.distillation import PassthroughDistiller

    from bladex_proxy.config import ProxyConfig
    from bladex_proxy.distillation import LLMDistiller
    from bladex_proxy.embedding import (
        build_embedder,
        effective_thresholds,
        resolve_embed_settings,
        validate_embed_sensitivity,
    )
    from bladex_proxy.routing_config import bootstrap_distill_model
    from bladex_proxy.storage.memory_hub import MemoryHub
    from bladex_proxy.storage.memory_index import IndexDistillJournal, MemoryIndex

    cfg = ProxyConfig()
    validate_embed_sensitivity(cfg)
    settings = resolve_embed_settings(cfg)
    thresholds = effective_thresholds(cfg)
    log.line(f"· direct mode: Memory Hub opened read-only/secondary (the proxy can keep running)"
             f" embed={settings.backend}/{settings.model} concurrency={args.concurrency}")

    # Memory Hub 恒 secondary 只读 —— full 重建也只读 Memory Hub，proxy 主锁不受影响。
    hub = MemoryHub(cfg.rocksdb_path, secondary=True)
    hub.open()

    embedder = build_embedder(cfg, role="consolidator")
    try:
        if hasattr(embedder, "wait_ready"):
            embedder.wait_ready(timeout_s=30)
        embedder.embed(["warmup"])
    except Exception as e:  # noqa: BLE001
        if settings.backend != "local":
            log.line(f" · embedding backend {settings.backend} unavailable ({e}) -> falling back to local e5")
            os.environ["BLADEX_EMBED_BACKEND"] = "local"
            os.environ.setdefault("BLADEX_EMBED_MODEL", settings.model)
            cfg = ProxyConfig()
            embedder = build_embedder(cfg, role="consolidator")
            embedder.embed(["warmup"])
        else:
            log.line(f"❌ local embedding unavailable: {e}")
            hub.close()
            return 1

    distill_model = bootstrap_distill_model(
        cfg.distill_model, cfg.routing_config, cfg.routing_config_path)
    distill_journal = IndexDistillJournal()
    if distill_model:
        distiller = LLMDistiller(model=distill_model, api_base=cfg.upstream_api_base,
                                 api_key=cfg.upstream_api_key, journal=distill_journal)
    else:
        distiller = PassthroughDistiller()
        log.line("· no distillation model configured -> passthrough (short messages only, no LLM calls)")

    link_judge = None
    if distill_model:
        from bladex_proxy.linking import LLMLinkJudge
        link_judge = LLMLinkJudge(model=distill_model, api_base=cfg.upstream_api_base,
                                  api_key=cfg.upstream_api_key)

    try:
        index = MemoryIndex(
            cfg.index_path, embedder=embedder,
            novelty_threshold=thresholds["novelty"],
            semantic_threshold=thresholds["semantic"],
            digestion_threshold=thresholds["digestion"],
            distiller=distiller, link_judge=link_judge,
            sensitivity_config=cfg.routing_config.sensitivity_config(),
            entity_aware_novelty=cfg.entity_aware_novelty,
            entity_overlap_threshold=cfg.entity_overlap_threshold,
            novelty_topk=cfg.novelty_topk,
            embed_model_id=getattr(embedder, "model_identity", None),
            entity_aware_rerank=cfg.entity_aware_rerank,
            entity_rerank_alpha=cfg.entity_rerank_alpha,
            entity_rerank_expand_k=cfg.entity_rerank_expand_k,
        )
        index.open()
    except Exception as e:  # noqa: BLE001 —— 最常见 = 写锁被 consolidator 占
        log.line(f"❌ could not open Memory Index: {e}")
        log.line("   If you see lock/Resource temporarily unavailable: a consolidator is "
                 "running but did not pick up the job -- restart it after upgrading (only newer builds have the control channel), "
                 "or stop it (`bash config/run_consolidator.sh stop`) and retry in direct mode.")
        hub.close()
        return 1
    distill_journal.bind(index)

    legacy_map = cfg.identity_registry.legacy_map()
    # ADR-0021 §2.3 修订：个人模式折叠一切 user_id 到 local（理由见 consolidator 同名变量）
    from bladex_proxy.identity import LOCAL_USER_ID
    fold_user_id = "" if not cfg.identity_registry.empty else LOCAL_USER_ID
    t0 = time.perf_counter()

    def _cb(ev: dict) -> None:
        log.progress(_fmt_progress({k: str(v) for k, v in ev.items()}))

    def _s(v: str) -> set[str] | None:
        items = {x.strip() for x in v.split(",") if x.strip()}
        return items or None

    try:
        hub.catch_up()
        nf = index.rebuild_from_hub(
            hub, full=args.full, legacy_map=legacy_map,
            fold_user_id=fold_user_id, max_turns=args.max_turns,
            since=args.since, until=args.until,
            agents=_s(args.agents), exclude_agents=_s(args.exclude_agents),
            distill_concurrency=args.concurrency, progress_cb=_cb,
        )
        log.progress_end()
        elapsed = round(time.perf_counter() - t0, 1)
        stats = distiller.stats() if hasattr(distiller, "stats") else {}
        log.line(f"✅ Sync complete: new_facts={nf} elapsed={elapsed}s "
                 f"backlog={index.last_rebuild_backlog} distiller={json.dumps(stats, ensure_ascii=False)}")
        return 0
    except Exception as e:  # noqa: BLE001
        log.progress_end()
        log.line(f"❌ Sync failed: {e}")
        return 1
    finally:
        index.close()
        hub.close()


# ── 子命令 ───────────────────────────────────────────────────────────────────


def cmd_sync_run(args: argparse.Namespace) -> int:
    log = _SyncLog(Path(args.log or f"logs/sync-{time.strftime('%Y%m%d-%H%M%S')}.log"))
    log.line(f"═══ bladex sync run full={args.full} concurrency={args.concurrency} "
             f"max_turns={args.max_turns} ═══")
    log.line(f"· log: {log._path}")
    if args.full:
        log.line("⚠ --full clears Memory Index and replays from the Hub (Memory Hub is read-only; journal hits cost no LLM calls). "
                 "Retrieval may be briefly degraded during the rebuild.")
    try:
        if not args.direct:
            rc = _try_delegate(args, log)
            if rc is not None:
                return rc
        return _run_direct(args, log)
    finally:
        log.close()


def cmd_sync_status(args: argparse.Namespace) -> int:
    from bladex_proxy.config import ProxyConfig
    from bladex_proxy.sync_control import SyncControl
    try:
        import redis
        rc = redis.Redis.from_url(ProxyConfig().redis_url, decode_responses=True,
                                  socket_connect_timeout=2)
        rc.ping()
    except Exception as e:  # noqa: BLE001
        print(f"Redis unreachable: {e}")
        return 1
    control = SyncControl(rc)
    job_id = args.job_id or control.last_job_id()
    if not job_id:
        print("No recent sync jobs.")
        return 0
    st = control.get_status(job_id)
    if not st:
        print(f"Job {job_id} has no state (probably expired; TTL is 24h).")
        return 1
    print(f"job {job_id}")
    for k in ("state", "phase", "done", "total", "new_facts", "elapsed_s",
              "backlog", "error", "worker"):
        if st.get(k):
            print(f"  {k:10s} = {st[k]}")
    return 0 if st.get("state") != "failed" else 1


sync_app = typer.Typer(help="Manually sync Memory Hub -> Memory Index (delegate first, then direct)",
                       no_args_is_help=True)
config_app = typer.Typer(help="Configuration", no_args_is_help=True)


# ── config check ────────────────────────────────────────────────────────────


def _upstream_key_problems(cfg, routing_path: Path) -> list[tuple[str, str, str]]:
    """上游凭证检查（ADR-0027 §3.1，契约 A：绿灯不许骗人）。

    返回 [(label, err, fix)]；空 = 通过。两条路径都查：
    - `.env` 的 `BLADEX_UPSTREAM_API_KEY`（无 routing.toml 时的唯一凭证）；
    - `routing.toml` 里每个模型的 `api_key_env` 指向的环境变量是否真的有值
      （密钥只引 env 是既有纪律，但"引了个不存在的 env"此前无人检查）。
    """
    problems: list[tuple[str, str, str]] = []
    routing_models: list = []
    if routing_path.is_file():
        try:
            routing_models = list(getattr(cfg.routing_config, "models", []) or [])
        except Exception:  # noqa: BLE001 —— routing.toml 自身的错误上面已单独报过
            routing_models = []

    if not routing_models:
        if not (cfg.upstream_api_key or "").strip():
            problems.append((
                "upstream key",
                "BLADEX_UPSTREAM_API_KEY is empty -- every upstream request will fail",
                f"set BLADEX_UPSTREAM_API_KEY in {Path(_repo_root()) / 'config' / '.env'}",
            ))
        return problems

    for m in routing_models:
        env_name = getattr(m, "api_key_env", "") or ""
        name = getattr(m, "name", "?")
        if not env_name:
            # 模型可以不带 key（本地 ollama 等），只有显式声明了才检查。
            continue
        if not (os.environ.get(env_name) or "").strip():
            problems.append((
                f"upstream key [{name}]",
                f"routing.toml references api_key_env=\"{env_name}\" but that variable is empty/unset",
                f"set {env_name} in config/.env (routing.toml stores the variable name, never the value)",
            ))
    return problems


def _mask_secret(value: str) -> str:
    """密钥只报"有没有 / 多长"，绝不打印内容。

    这条输出的用途之一是贴进 doctor 报告发给别人看，所以它必须**天然不可泄密**。
    """
    v = (value or "").strip()
    if not v:
        return "(unset)"
    return f"{v[:3]}***({len(v)} chars)"


def render_effective_config(cfg, root: Path | None = None) -> str:
    """把"现在到底在按什么跑"渲染成一段确定性文本（G9.4 H3）。

    ## 为什么要它

    130 个 flag 的生效值散在 env / routing.toml / flags.py 三处，排查时只能逐个 grep。
    已经吃过两次亏：ADR-0027 §5.4「五个开关默认关且不在模板里」——那次"验收通过"
    是在临时 shell 里跑的，仓库默认形态从未变过；ADR-0028「配置写 200、进程跑 150」。
    两次的共同形状都是**没人能一眼看到生效值**。

    ## 输出为什么必须是确定性文本

    要可 diff（改配置前后对比）、可贴进报告。所以：键有序、不打时间戳、
    不打绝对路径、密钥只显示指纹。

    ## 🔴 有意不做：不打印"路由的线性装配顺序"

    那个顺序只存在于 `MemoryAwareRouter.route()` 的控制流里。在这里重写一份
    = 造第二份真相，它会在下一次改 route() 时静默过期，而**过期的配置视图比没有更糟**
    （CLAUDE.md 刚性原则 9 那次事故正是"以为按配置走、其实被动态层旁路"）。
    所以只打**分组**（静态 / 动态 / 硬约束）与各层开关，分组直接取自
    `bladex_core.routing._STATIC_SOURCES` / `_DYNAMIC_SOURCES`，与路由同源。
    """
    from bladex_core.flags import (
        MEMORY_FLAG_DEFAULTS,
        MEMORY_NUMERIC_DEFAULTS,
        flag_enabled,
        flag_number,
    )
    from bladex_core.routing import _DYNAMIC_SOURCES, _STATIC_SOURCES

    out: list[str] = ["BladeX effective configuration", "=" * 62, ""]

    out.append("[1] Memory flags            value    source (env > flags.py default)")
    for name in sorted(MEMORY_FLAG_DEFAULTS):
        src = "env" if name in os.environ else "default"
        out.append(f"  {name:<44} {'on' if flag_enabled(name) else 'off':<7} {src}")
    for name in sorted(MEMORY_NUMERIC_DEFAULTS):
        src = "env" if name in os.environ else "default"
        out.append(f"  {name:<44} {flag_number(name):<7g} {src}")

    out += ["", "[2] Embedding               value    source (env > routing.toml > default)"]
    try:
        from bladex_proxy.embedding import resolve_embed_settings
        s = resolve_embed_settings(cfg)
        shown = {
            "backend": s.backend, "model": s.model or "(built-in default)",
            "api_base": s.api_base or "(unset)", "api_key": _mask_secret(s.api_key),
            "proxy_url": s.proxy_url, "proxy_key": _mask_secret(s.proxy_key),
            "query_cache_size": str(s.query_cache_size),
        }
        for key in sorted(shown):
            out.append(f"  {key:<44} {shown[key]:<7} {s.sources.get(key, 'default')}")
    except Exception as e:  # noqa: BLE001 — 展示层不许因为一个字段炸掉整张表
        out.append(f"  (unavailable: {e})")

    out += ["", "[3] Routing strategy layers (grouping from routing._STATIC_SOURCES / _DYNAMIC_SOURCES)"]
    rc = getattr(cfg, "routing_config", None)
    strategies = getattr(rc, "strategies", None)
    static_names = sorted(s.value for s in _STATIC_SOURCES)
    dynamic_names = sorted(s.value for s in _DYNAMIC_SOURCES)

    def _state(layer: str) -> str:
        # routing.toml 的 [strategies.<name>] 段名与 RouteSource 值大多同名；
        # 对不上的（requested -> request）在这里显式映射，不猜。
        alias = {"requested": "request"}
        sub = getattr(strategies, alias.get(layer, layer), None)
        if sub is None:
            return "n/a"
        return "on" if getattr(sub, "enabled", False) else "off"

    out.append("  static  (config wins; dynamic layers must not override a static hit):")
    for n in static_names:
        out.append(f"    {n:<42} {_state(n)}")
    out.append("  dynamic (inference; only decides when no static layer matched):")
    for n in dynamic_names:
        out.append(f"    {n:<42} {_state(n)}")
    out.append("  hard    (objective constraints; may override static, always logged):")
    out.append(f"    {'sensitivity':<42} {_state('sensitivity')}")
    out.append(f"    {'capability filter':<42} always on")
    out.append(f"    {'failover / circuit breaker':<42} always on")

    out += ["", "[4] Injection & assembly"]
    budget = getattr(cfg, "hotpath_budget_ms", "?")
    out.append(f"  {'hotpath budget (ms)':<44} {budget}")
    for name in sorted(n for n in os.environ if n.startswith("BLADEX_ASSEMBLY_")):
        out.append(f"  {name:<44} {os.environ[name]:<7} env")
    out.append(f"  {'routing.toml':<44} "
               f"{'present' if Path(getattr(cfg, 'routing_config_path', '') or '').is_file() else 'absent'}")
    out.append(f"  {'identity.toml':<44} "
               f"{'present' if Path(getattr(cfg, 'identity_config_path', '') or '').is_file() else 'absent'}")
    out.append("")
    return "\n".join(out)


@config_app.command("show")
def config_show(
    effective: bool = typer.Option(True, "--effective/--raw",
                                   help="Show merged effective values (default) "
                                        "instead of raw file contents"),
) -> int:
    """Print the configuration this process would actually run with.

    Deterministic text: sorted keys, no timestamps, secrets fingerprinted only.
    Safe to diff across changes and to paste into a bug report.
    """
    if not effective:
        print("Only --effective is implemented; read config/.env and config/routing.toml "
              "directly for the raw contents.")
        return 1
    from bladex_proxy.config import ProxyConfig
    try:
        cfg = ProxyConfig()
    except Exception as e:  # noqa: BLE001
        print(f"Error: cannot build ProxyConfig: {e}")
        print("fix: run `bladex config check` first -- it reports the offending variable.")
        return 1
    print(render_effective_config(cfg))
    return 0


@config_app.command("check")
def config_check() -> int:
    """Validate configuration the same way startup does.

    报错尽量带定位（TOML 解析错误含行号）与修法。
    """
    problems: list[str] = []

    def _ok(label: str, detail: str = "") -> None:
        print(f"  ✓ {label}" + (f" ({detail})" if detail else ""))

    def _bad(label: str, err: str, fix: str) -> None:
        problems.append(label)
        print(f"  ✗ {label}: {err}\n    fix: {fix}")

    if not Path("config/.env").is_file():
        _bad("config/.env", "file not found",
             "bladex init (or cp config/.env.example config/.env)")
    else:
        _ok("config/.env")

    from bladex_proxy.config import ProxyConfig
    try:
        cfg = ProxyConfig()
        _ok("ProxyConfig", f"host={cfg.host} port={cfg.port}")
    except Exception as e:  # noqa: BLE001
        _bad("ProxyConfig", str(e), "check the value types of the corresponding variables in config/.env")
        print(f"Found {len(problems)} problem(s).")
        return 1

    # routing.toml（tomllib 错误自带行号；RoutingConfigError 是启动同款校验）
    import tomllib

    from bladex_proxy.routing_config import RoutingConfig, RoutingConfigError
    rpath = Path(cfg.routing_config_path)
    if not rpath.is_file():
        _ok("routing.toml", f"{rpath} not present -> routing disabled / env fallback (a valid state)")
    else:
        try:
            RoutingConfig.from_toml(rpath)
            _ok("routing.toml", str(rpath))
        except tomllib.TOMLDecodeError as e:
            _bad("routing.toml", f"TOML syntax error: {e}",
                 f"fix the reported line in {rpath} (tomllib errors carry line/column)")
        except RoutingConfigError as e:
            _bad("routing.toml", str(e),
                 "compare against config/routing.toml.example: model pool / strategies / capability vocabulary")
        except Exception as e:  # noqa: BLE001
            _bad("routing.toml", str(e), f"check {rpath}")

    # identity.toml（可选；存在才校验）
    ipath = Path(cfg.identity_config_path)
    if not ipath.is_file():
        _ok("identity.toml", "not present -> personal mode (a valid state)")
    else:
        try:
            _ = cfg.identity_registry
            _ok("identity.toml", str(ipath))
        except Exception as e:  # noqa: BLE001
            _bad("identity.toml", str(e),
                 "compare against config/identity.toml.example: keys/principals/teams")

    # auth 一致性
    if cfg.auth_enabled and not cfg.client_keys_raw.strip():
        _bad("auth", "BLADEX_AUTH_ENABLED=true but BLADEX_CLIENT_KEYS is empty (every request will 401)",
             "set BLADEX_CLIENT_KEYS=<key>||<label> in config/.env")
    elif not cfg.auth_enabled and cfg.host not in ("127.0.0.1", "localhost", "::1"):
        print("  ⚠ auth is off and the bind address is not loopback -- enable BLADEX_AUTH_ENABLED for anything exposed")

    if not cfg.upstream_model and not rpath.is_file():
        _bad("upstream", "BLADEX_UPSTREAM_MODEL is empty and there is no routing.toml (nowhere to forward)",
             "set BLADEX_UPSTREAM_MODEL=provider/model in config/.env")

    # ADR-0027 §3.1：上游 key 缺失是新用户第一个必踩的坑，必须是最响的那个错。
    # 此前 config check 全程不看它 —— 空 key 照样"配置检查通过"，第一条真实请求必然失败。
    for label, err, fix in _upstream_key_problems(cfg, rpath):
        _bad(label, err, fix)

    if problems:
        print(f"Found {len(problems)} problem(s).")
        return 1
    print("Configuration check passed.")
    return 0


# ── sync（收编：T11 前置切片，行为不变）────────────────────────────────────


@sync_app.command("run")
def sync_run(
    full: bool = typer.Option(False, "--full", help="Full rebuild (clear Memory Index, replay from the Hub)"),
    concurrency: int = typer.Option(
        -1, "--concurrency",
        help="Distillation concurrency (1=serial; defaults to BLADEX_DISTILL_CONCURRENCY, or 4)"),
    max_turns: int = typer.Option(0, "--max-turns", help="Max turns to process (0=unlimited)"),
    since: str = typer.Option("", "--since", help="Only turns with ts >= this (ISO prefix)"),
    until: str = typer.Option("", "--until", help="Only turns with ts <= this (ISO prefix, inclusive)"),
    agents: str = typer.Option("", "--agents", help="Only these agents (comma separated)"),
    exclude_agents: str = typer.Option("", "--exclude-agents", help="Exclude these agents (comma separated)"),
    direct: bool = typer.Option(False, "--direct", help="Skip the control channel and run in this process"),
    pickup_timeout: float = typer.Option(10.0, "--pickup-timeout", help="Seconds to wait for a consolidator to pick up the job"),
    log: str = typer.Option("", "--log", help="Log file path (default logs/sync-<ts>.log)"),
) -> int:
    """Run one sync pass (delegate first, falling back to direct)."""
    # main() 已 chdir 到部署根；命令行上给的相对路径要钉回用户敲命令的目录。
    log = deployment.resolve_user_path(log)
    if concurrency < 0:
        # 唯一真相源走 flags（与守护进程同一个默认值——两条路曾各有一个，
        # 结果是手工 sync 并发 4、守护进程串行 1，见 _start_consolidator）
        concurrency = _resolve_distill_concurrency()
    ns = argparse.Namespace(
        full=full, concurrency=concurrency, max_turns=max_turns, since=since,
        until=until, agents=agents, exclude_agents=exclude_agents, direct=direct,
        pickup_timeout=pickup_timeout, log=log)
    return cmd_sync_run(ns)


@sync_app.command("status")
def sync_status(job_id: str = typer.Argument("", help="Job ID (default: the most recent)")) -> int:
    """Show the status of a sync job."""
    return cmd_sync_status(argparse.Namespace(job_id=job_id))


queue_app = typer.Typer(help="Pipeline queue (turns waiting to reach the Memory Hub)",
                        no_args_is_help=True)
sticky_app = typer.Typer(help="Session model stickiness (which model each session is pinned to)",
                         no_args_is_help=True)


# ── queue（Pipeline -> Memory Hub）─────────────────────────────


def _queue_lines(d: dict) -> list[str]:
    """把 /admin/queue 的四个数翻成人话——每一段都要说清"谁该来搬它"。"""
    total = d.get("total")
    undelivered, pending, orphan = d.get("undelivered"), d.get("pending"), d.get("orphan")
    lines = [f"  Queued total:  {total} entries"]
    if undelivered is None:
        lines.append("  Undelivered:   ? (this Redis does not report group lag; "
                     "needs Redis 7.0+)")
    else:
        lines.append(f"  Undelivered:   {undelivered} -- not handed to the worker yet;"
                     " it picks them up on its own")
    lines.append(f"  Unconfirmed:   {pending} -- handed over, not yet written to the Hub;"
                 " retried automatically")
    if orphan is None:
        lines.append("  Stale:         ? (cannot be derived without group lag)")
    else:
        lines.append(f"  Stale:         {orphan} -- in neither queue, so nothing will ever"
                     " pick them up")
    spilled = (d.get("overflow_count") or 0) + (d.get("spill_count") or 0)
    if spilled:
        lines.append(f"  Spilled:       {spilled} to disk (replayed once Redis is back)")
    return lines


@queue_app.command("status")
def queue_status() -> int:
    """Break the Pipeline backlog into undelivered / unconfirmed / stale."""
    code, data = _admin_call("GET", "/admin/queue")
    if code != 200:
        print(f"✗ /admin/queue HTTP {code}: {json.dumps(data, ensure_ascii=False)}")
        return 1
    for line in _queue_lines(data):
        print(line)
    if data.get("orphan"):
        print("\n  Stale entries keep the backlog number from ever dropping."
              "\n  Preview the cleanup:  bladex queue flush"
              "\n  Then apply it:        bladex queue flush --apply")
    return 0


def _print_orphan_profile(data: dict) -> None:
    """孤儿画像。`same session in Hub` 是关键对照量——同会话别的轮次都在、
    独独缺这几条，才说明它们是真的没落过账，而不是 key 算法漂了导致全查不中。"""
    agents = data.get("orphan_agents") or {}
    first, last = data.get("orphan_first_ts"), data.get("orphan_last_ts")
    if agents:
        print("\n  Stale entries by agent: "
              + ", ".join(f"{k} {v}" for k, v in sorted(
                  agents.items(), key=lambda kv: -kv[1])))
    if first:
        print(f"  Arrived between:        {first}  ..  {last}")
    for s in data.get("samples") or []:
        neighbours = s.get("session_turns_in_hub")
        print(f"\n  - {s.get('entry_id')}  {s.get('ts')}  agent={s.get('agent')}"
              f" status={s.get('status')}"
              f"\n    key={s.get('key')}"
              f"\n    in Hub: {s.get('in_hub')}"
              f" / same session in Hub: "
              + ("unreadable" if neighbours == -1 else f"{neighbours} turns"))


@queue_app.command("flush")
def queue_flush(
    apply: bool = typer.Option(False, "--apply",
                               help="Actually do it (default is a preview that changes nothing)"),
    max_batches: int = typer.Option(200, "--max-batches",
                                    help="Batch cap per phase (100 entries each)"),
    sample: int = typer.Option(0, "--sample",
                               help="Also profile N stale entries: when they arrived,"
                                    " which agent, and how many turns of the same session"
                                    " are already in the Hub"),
) -> int:
    """Force the Pipeline queue to drain: deliver, reclaim, then clear stale entries.

    Runs inside the proxy (the Memory Hub is single-writer and the proxy holds the
    write lock). Stale entries are checked against the Hub one by one -- an entry
    that never made it is written first, and only then removed.
    """
    code, data = _admin_call("POST", "/admin/queue/flush",
                             params={"apply": str(apply).lower(),
                                     "max_batches": max_batches, "sample": sample},
                             body={}, timeout=180.0)
    if code != 200:
        print(f"✗ /admin/queue/flush HTTP {code}: {json.dumps(data, ensure_ascii=False)}")
        return 1

    before, after = data.get("before") or {}, data.get("after") or {}
    if apply:
        print(f"✓ delivered {data.get('delivered', 0)}"
              f" / reclaimed {data.get('reclaimed', 0)}"
              f" / stale cleared {data.get('orphan_deleted', 0)}"
              f" (of which {data.get('orphan_recovered', 0)} were missing from the Hub"
              " and were written first)")
        if data.get("failed"):
            print(f"  ⚠ {data['failed']} entries could not be written to the Hub --"
                  " they stay queued and are retried; check the proxy log for"
                  " pipeline_ledger_write_failed")
        print(f"  Queued: {before.get('total')} -> {after.get('total')}")
    else:
        print("Preview only -- nothing was changed. Add --apply to run it.\n")
        for line in _queue_lines(before):
            print(line)
        print(f"\n  Would deliver:     {before.get('undelivered')}"
              f"\n  Would reclaim:     {before.get('pending')}"
              f"\n  Would clear stale: {data.get('orphan_total', 0)}"
              f" ({data.get('orphan_in_hub', 0)} already in the Hub -> safe to drop,"
              f" {data.get('orphan_missing', 0)} missing -> written to the Hub first)")
        _print_orphan_profile(data)
    if data.get("truncated"):
        print("  ⚠ hit the batch cap -- run it again to continue"
              " (or raise --max-batches)")
    return 0


# ── sticky (G14 / MQ-RT1) ──────────────────────────────────────────


@sticky_app.command("status")
def sticky_status() -> int:
    """Show which model each session is pinned to, and why it got pinned.

    `origin=failover` means the session was pushed onto a fallback model by an
    upstream failure, not chosen. A large `held_s` on such a row is the shape of
    the 2026-08-21 incident: a 2-second upstream blip pinned a session to a local
    model for 53 minutes. Those rows release themselves once the upstream prompt
    cache goes cold; `sticky flush` forces it now.
    """
    code, data = _admin_call("GET", "/admin/sticky")
    if code != 200:
        print(f"✗ /admin/sticky HTTP {code}: {json.dumps(data, ensure_ascii=False)}")
        return 1
    if not data.get("enabled"):
        print("Routing is disabled (BLADEX_ROUTE_ENABLED=false) -- no sticky table.")
        return 0
    rows = data.get("sessions") or []
    if not rows:
        print("No sessions are pinned right now.")
        return 0
    print(f"{'SESSION':<18} {'AGENT':<20} {'MODEL':<32} {'ORIGIN':<9} HELD")
    for r in rows:
        held = r.get("held_s")
        held_s = "-" if held is None else f"{held:.0f}s"
        mark = "  ⚠" if r.get("origin") == "failover" else ""
        print(f"{str(r.get('session_id'))[:18]:<18} {str(r.get('agent_id') or '-')[:20]:<20}"
              f" {str(r.get('model'))[:32]:<32} {str(r.get('origin')):<9} {held_s}{mark}")
    degraded = [r for r in rows if r.get("origin") == "failover"]
    if degraded:
        print(f"\n⚠ {len(degraded)} session(s) are on a fallback model after an upstream"
              " failure.\n  They release automatically once the prompt cache goes cold"
              " or strategies.sticky.failover_max_hold_s elapses;"
              "\n  `bladex sticky flush` does it now.")
    return 0


@sticky_app.command("flush")
def sticky_flush(
    agent: str = typer.Option("", "--agent",
                              help="Only this agent (\"hermes\" also clears"
                                   " \"hermes:default\"); default is all sessions"),
) -> int:
    """Release session stickiness so the next turn re-picks a model from scratch.

    Use it after fixing an upstream: sessions that failed over to a fallback model
    otherwise keep using it until their prompt cache goes cold.

    Runs inside the proxy -- the sticky table is an in-process LRU on the router,
    so the CLI is the control plane and /admin/sticky/flush is the execution plane
    (same split as `queue flush`).
    """
    params = {"agent_id": agent} if agent else {}
    code, data = _admin_call("POST", "/admin/sticky/flush", params=params, body={})
    if code != 200:
        print(f"✗ /admin/sticky/flush HTTP {code}: {json.dumps(data, ensure_ascii=False)}")
        return 1
    scope = f"agent {agent}" if agent else "all agents"
    print(f"✓ released {data.get('cleared', 0)} session(s) for {scope}"
          f" ({data.get('remaining', 0)} still pinned)")
    for e in data.get("flushed") or []:
        held = e.get("held_s")
        print(f"    {str(e.get('session_id'))[:18]}  {e.get('model')}"
              f"  origin={e.get('origin')}  held={'-' if held is None else f'{held:.0f}s'}")
    if not data.get("cleared"):
        print("  (nothing was pinned -- next turn already re-picks a model)")
    return 0


@app.command()
def backup(dest: str = typer.Argument("", help="Backup target directory (default data/backup_<ts>)")) -> int:
    """Back up the Memory Hub checkpoint, Memory Index and the Redis AOF (online, no downtime)."""
    dest = deployment.resolve_user_path(dest)  # 相对路径钉回用户敲命令的目录
    return _run_repo_script("scripts/backup.sh", [dest] if dest else [])


@app.command()
def restore(src: str = typer.Argument(..., help="Backup directory")) -> int:
    """Restore from a backup (stop the proxy and consolidator first)."""
    if _read_pidfile(_PROXY_PIDFILE) is not None or _consolidator_pid() is not None:
        print("Error: the proxy/consolidator are still running -- run `bladex stop` before restoring (avoids write conflicts)")
        return 1
    return _run_repo_script("scripts/restore.sh", [deployment.resolve_user_path(src)])


# ── connector（T13 在线协同导出，B3.2）────────────────────────────

connector_app = typer.Typer(help="Continuous export to the targets configured in export.toml",
                            no_args_is_help=True)




def _build_export_worker():
    from bladex_proxy.config import ProxyConfig
    from bladex_proxy.export_sync import build_worker_from_config
    from bladex_proxy.storage.memory_hub import MemoryHub
    from bladex_proxy.storage.memory_index import MemoryIndex

    cfg = ProxyConfig()
    if not Path("config/export.toml").is_file():
        print("Not configured: config/export.toml does not exist"
              "（cp config/export.toml.example config/export.toml）")
        return None, None, None
    index = MemoryIndex(cfg.index_path, embedder=None, read_only=True)
    index.open()
    hub = None
    try:
        hub = MemoryHub(cfg.rocksdb_path, secondary=True)
        hub.open()
    except Exception as e:  # noqa: BLE001
        print(f"Warning: Memory Hub unreadable ({e}) -- tombstone sync skipped")
    worker = build_worker_from_config(index, hub, "config/export.toml",
                                      hard_rules=cfg.effective_hard_rules)
    return worker, index, hub


def _export_module_gate() -> bool:
    """V-A3（F0.2）：`connector` / `export` / `import` 五个入口的**唯一**模块门。关 ⇒ 打一行、退出 2。"""
    from bladex_proxy.modules import module_enabled
    if module_enabled("export"):
        return True
    print("export module disabled (BLADEX_MODULE_EXPORT=0); enable it to use connector/export/import",
          file=sys.stderr)
    return False


@connector_app.command("run")
def connector_run(
    force: bool = typer.Option(False, "--force", help="Ignore the backoff window and try now"),
) -> int:
    """Run one incremental export pass (a running consolidator also does this periodically)."""
    if not _export_module_gate():
        return 2
    worker, index, hub = _build_export_worker()
    if worker is None:
        if index is not None:
            index.close()
        return 1
    try:
        results = worker.run_once(force=force)
    finally:
        index.close()
        if hub is not None:
            hub.close()
    for name, r in results.items():
        if r.get("error"):
            print(f"  ✗ {name}: {r['error']} (backing off)")
        elif r.get("skipped"):
            print(f"  - {name}: skipped ({r['skipped']})")
        else:
            print(f"  ✓ {name}: synced={r['synced']} tombstones={r['tombstones']}")
    return 1 if any(r.get("error") for r in results.values()) else 0


@connector_app.command("status")
def connector_status() -> int:
    """Per-target cursor, backoff and last success."""
    if not _export_module_gate():
        return 2
    from bladex_proxy.export_sync import ExportStateStore, load_export_config
    configs = load_export_config("config/export.toml")
    if not configs:
        print("Not configured: config/export.toml is missing or has no [[exporters]]")
        return 0
    state = ExportStateStore("data/export_state.json")
    for c in configs:
        st = state.get(c.name)
        print(f"  {c.name} ({c.type}, exposure={c.exposure},"
              f" {'enabled' if c.enabled else 'disabled'})")
        print(f"    cursor={st.get('cursor') or '(initial)'} last_ok={st.get('last_ok') or '-'}"
              f" failures={st.get('failures', 0)}"
              f" tombstones_done={len(st.get('tombstones_done', []))}")
    return 0


@connector_app.command("reset")
def connector_reset(
    name: str = typer.Argument("", help="Target name (empty = all)"),
    yes: bool = typer.Option(False, "--yes"),
) -> int:
    """Clear cursor state (the next pass re-exports everything)."""
    if not _export_module_gate():
        return 2
    if not yes and not typer.confirm(f"Reset the sync cursor for {'all targets' if not name else name}?"):
        print("Cancelled.")
        return 130
    from bladex_proxy.export_sync import ExportStateStore
    ExportStateStore("data/export_state.json").reset(name or None)
    print("✓ Reset.")
    return 0


# ── export / import（T12 快照，B3.1）──────────────────────────────


@app.command("export")
def export_cmd(
    path: str = typer.Argument("", help="Output file (default data/export_<ts>.jsonl)"),
    full: bool = typer.Option(False, "--full", help="Include the full Memory Hub turn archive (off by default; large)"),
) -> int:
    """Export a memory snapshot to JSONL (facts, matters, edges, hard rules, admin events).

    墓碑数据不出现在导出中（删除语义，ADR-0012）。
    只读操作：Memory Hub secondary + Memory Index 只读句柄，proxy/consolidator 无需停。
    """
    if not _export_module_gate():
        return 2
    from bladex_proxy.config import ProxyConfig
    from bladex_proxy.snapshot import export_snapshot
    from bladex_proxy.storage.memory_hub import MemoryHub
    from bladex_proxy.storage.memory_index import MemoryIndex

    cfg = ProxyConfig()
    # 相对路径钉回用户敲命令的目录（main() 已 chdir 到部署根）；缺省值仍落部署根 data/。
    out = deployment.resolve_user_path(path) or f"data/export_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    index = MemoryIndex(cfg.index_path, embedder=None, read_only=True)
    try:
        index.open()
    except Exception as e:  # noqa: BLE001
        print(f"Error: could not open Memory Index ({e})")
        return 1
    hub = None
    try:
        hub = MemoryHub(cfg.rocksdb_path, secondary=True)
        hub.open()
    except Exception as e:  # noqa: BLE001
        print(f"Warning: Memory Hub unreadable ({e}) -- skipping the admin-event / turn sections")
        hub = None
    try:
        counts = export_snapshot(index, hub, cfg.effective_hard_rules, out,
                                 include_turns=full)
    finally:
        index.close()
        if hub is not None:
            hub.close()
    print(f"✓ Exported {out}: " + " ".join(f"{k}={v}" for k, v in counts.items() if v))
    return 0


@app.command("import")
def import_cmd(
    path: str = typer.Argument(..., help="Snapshot file (JSONL)"),
) -> int:
    """Import a memory snapshot (idempotent merge by id; manual overrides auto).

    冲突非阻断标注；
    导入实体以管理事件入 Memory Hub（full rebuild 后不丢失）。

    需 proxy + consolidator 已停（Memory Index/Memory Hub 写锁）。
    """
    if not _export_module_gate():
        return 2
    if _read_pidfile(_PROXY_PIDFILE) is not None or _consolidator_pid() is not None:
        print("Error: the proxy/consolidator are still running -- run `bladex stop` first (import needs the Memory Index/Memory Hub write locks)")
        return 1
    path = deployment.resolve_user_path(path)  # 相对路径钉回用户敲命令的目录
    if not Path(path).is_file():
        print(f"Error: not found: {path}")
        return 1
    from bladex_proxy.config import ProxyConfig
    from bladex_proxy.snapshot import import_snapshot
    from bladex_proxy.storage.memory_hub import MemoryHub
    from bladex_proxy.storage.memory_index import MemoryIndex

    cfg = ProxyConfig()
    embedder = None
    try:
        from bladex_proxy.embedding import build_embedder
        embedder = build_embedder(cfg, role="consolidator")
        embedder.embed(["warmup"])
    except Exception as e:  # noqa: BLE001
        print(f"Warning: embedding unavailable ({e}) -- imported facts land in meta only; "
              "run `bladex storage rebuild` afterwards to add vectors")
        embedder = None
    try:
        index = MemoryIndex(cfg.index_path, embedder=embedder, read_only=False,
                       embed_model_id=getattr(embedder, "model_identity", None))
        index.open()
    except Exception as e:  # noqa: BLE001
        print(f"Error: could not open Memory Index ({e}) -- write lock held? run `bladex stop` first")
        return 1
    hub = None
    try:
        hub = MemoryHub(cfg.rocksdb_path)
        hub.open()
    except Exception as e:  # noqa: BLE001
        print(f"Warning: Memory Hub not writable ({e}) -- imported entities do not reach the journal; a full rebuild would lose them!")
        hub = None
    try:
        stats = import_snapshot(index, hub, path)
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    finally:
        index.close()
        if hub is not None:
            hub.close()
    print("✓ Import complete: " + " ".join(f"{k}={v}" for k, v in stats.items() if v))
    if stats.get("hard_rules_seen"):
        print(f"  ⚠ the snapshot carries {stats['hard_rules_seen']} hard rule(s) (configuration plane) -- "
              "merge them into BLADEX_HARD_RULES in config/.env by hand to make them effective")
    if stats.get("conflicts"):
        print(f"  ⚠ {stats['conflicts']} fact(s) had conflicting content: the local version was kept (annotated, non-blocking; see the log)")
    return 0
