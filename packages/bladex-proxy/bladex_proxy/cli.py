"""BladeX CLI（Beta T10：typer 骨架 + 生命周期 + init/doctor；T11 前置 sync 已收编）。

命令族：

    bladex --version
    bladex start [--foreground] [--proxy-only]     # redis + consolidator + proxy
    bladex stop / status / restart                 # 生命周期（收编 run_*.sh 逻辑）
    bladex init [--yes] [--force]                  # 交互生成 config/.env + routing.toml
    bladex config check                            # 启动校验（报错带定位与修法）
    bladex doctor [--warmup]                       # redis/RocksDB/embed 模型/上游/端口体检
    bladex sync run/status                         # 手工同步 Memory Hub→Memory Index（ADR-0009 §12）
    bladex queue status/flush [--apply]            # Pipeline→Memory Hub 那段队列（2026-08-06）

设计要点：
  - 生命周期收编 config/run_proxy.sh + run_consolidator.sh（脚本留薄壳）；
    stop 不动本地 Redis（数据面便宜且被 Pipeline 懒恢复依赖，ADR-0017）。
  - sync delegate 模式：CLI=控制面，consolidator=执行面（sync_control.py Redis
    协议）；direct 模式本进程执行（Memory Hub 恒 secondary 只读，proxy 无需停）。
  - 兼容层：`main(argv) -> int` 维持旧签名（scripts/bladex.py 薄壳与
    test_sync_cli 依赖），内部走 typer（standalone_mode=False 拿返回码）。
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import time
from pathlib import Path

# 部署根发现与 .env 加载的**唯一实现**在 deployment.py（2026-08-06 收敛）。
# 此处只留薄封装：历史调用点与测试都按这些名字来，改名不值当。
from bladex_proxy import deployment  # noqa: E402


def _repo_root() -> str:
    """部署根目录（ADR-0027 §4.3；2026-08-06 起支持向上查找 + `~/.bladex`）。

    查找顺序：`BLADEX_HOME` → cwd 及其祖先 → `~/.bladex`；都没有则退回 cwd。
    **调用时求值**：import 时冻结会让 chdir 后仍读旧位置（2026-08-03 环境污染 bug）。
    """
    return deployment.deployment_root()


def _config_search_paths() -> list[str]:
    """config/.env 的候选位置（报错时原样列给用户，不静默用空目录）。"""
    return deployment.search_paths()


def _require_config_root() -> str | None:
    """返回含 config/.env 的部署根；找不到返回 None（调用方负责报错）。"""
    return deployment.find_root()


def _chdir_to_deployment_root() -> str | None:
    """切到部署根，让所有相对路径（data/、logs/、PID 文件）有唯一落点。

    此前只有 `start` 和 `consolidator` 会切，于是 `bladex status` / `doctor` /
    `memory ...` 在别的目录跑出来的是一个**空库**——正是 `_repo_root` 文档里说的
    "症状不是报错而是数据看起来消失了"。找不到部署根时什么都不做：报错留给
    真正需要配置的那个命令去报（`--version` / `--help` / `init` 不该被拦）。

    用户在命令行上给的相对路径由 `deployment.resolve_user_path` 钉回原 cwd。
    """
    root = deployment.find_root()
    if root is None or root == os.getcwd():
        return root
    os.chdir(root)
    print(f"note: using BladeX deployment at {root}", file=sys.stderr)
    return root


def _print_config_not_found() -> None:
    """不静默用空目录：把找过的位置原样列出来（ADR-0027 §4.3）。"""
    print("Error: config/.env not found. Looked in:")
    for path in _config_search_paths():
        print(f"  - {path}")
    print(f"  - and every parent directory of {os.getcwd()}")
    print("Run `bladex init` here, or set BLADEX_HOME to your deployment directory.")


def _load_env_file() -> None:
    """加载 config/.env 并把配置冲突报到 stderr。

    2026-08-06 起 `.env` 默认压过残留的环境变量（见 deployment.py 顶部两次事故）。
    冲突走 stderr 而不是 stdout：`bladex ... --json` 之类的输出不能被它污染。
    """
    root = deployment.find_root()
    if root is None:
        return
    env_path = os.path.join(root, deployment.CONFIG_RELPATH)
    conflicts = deployment.load_env_file(root)
    lines = deployment.format_conflicts(conflicts, env_path)
    for line in lines:
        print(line, file=sys.stderr)
    # 🔴 引号错配：Python 侧读得出本意、shell `source` 却从那行吞到文件末尾，
    # 于是同一个文件两个加载器读出两个不同的环境（2026-09-01 live 病例：
    # `BLADEX_MODULE_FLASH="1'` 让 `source` 丢掉三个 BLADEX_ASSEMBLY_* 键，
    # 而运行时毫无异常）。只报不改 —— 修值是用户的事，程序的责任是别让它静默。
    for lineno, key in deployment.unbalanced_quote_lines(env_path):
        print(f"  WARNING: {env_path}:{lineno} {key} has mismatched quotes -- "
              f"Python reads it fine, but shell `source` swallows the rest of "
              f"the file from this line on.", file=sys.stderr)


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
    from bladex_proxy.storage.memory_index import MemoryIndex, IndexDistillJournal
    from bladex_proxy.storage.memory_hub import MemoryHub

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


# ═══════════════════════════════════════════════════════════════════════════
# Beta T10：typer 骨架 + 生命周期 + init / config check / doctor
# ═══════════════════════════════════════════════════════════════════════════

import shutil  # noqa: E402
import signal  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import urllib.error  # noqa: E402
import urllib.request  # noqa: E402

import typer  # noqa: E402

app = typer.Typer(help="BladeX — memory-first LLM proxy middleware",
                  add_completion=False, no_args_is_help=True)
sync_app = typer.Typer(help="Manually sync Memory Hub -> Memory Index (delegate first, then direct)",
                       no_args_is_help=True)
config_app = typer.Typer(help="Configuration", no_args_is_help=True)
app.add_typer(sync_app, name="sync")
app.add_typer(config_app, name="config")

_PROXY_PIDFILE = "logs/proxy.pid"
_CONS_PIDFILE = "logs/consolidator.pid"
#: V-A6 embedding 模块（默认关；backend=ipc 时才需要它）
_EMBED_PIDFILE = "logs/embed.pid"
_FLASH_PIDFILE = "logs/flash.pid"   # V-F2：flash 维护进程


def _version_cb(value: bool) -> None:
    if value:
        from bladex_proxy import __version__
        print(f"bladex {__version__}")
        raise typer.Exit(0)


@app.callback()
def _root(
    version: bool = typer.Option(  # noqa: B008
        False, "--version", help="Print the version", callback=_version_cb, is_eager=True),
) -> None:
    """BladeX CLI。"""


# ── 生命周期 helpers（模块级，可单测）─────────────────────────────────────────


def _read_pidfile(path: str) -> int | None:
    try:
        pid = int(Path(path).read_text().strip())
    except (OSError, ValueError):
        return None
    return pid if _pid_alive(pid) else None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _redis_hostport(url: str) -> tuple[str, int]:
    """redis://host:port/db → (host, port)；解析失败回默认。"""
    from urllib.parse import urlparse
    try:
        u = urlparse(url)
        return (u.hostname or "127.0.0.1", u.port or 6379)
    except ValueError:
        return ("127.0.0.1", 6379)


def _redis_ping(url: str) -> bool:
    try:
        import redis
        rc = redis.Redis.from_url(url, socket_connect_timeout=2)
        return bool(rc.ping())
    except Exception:  # noqa: BLE001
        return False


def _is_local_host(host: str) -> bool:
    return host in ("127.0.0.1", "localhost", "::1")


def _ensure_redis(url: str) -> bool:
    """本地 Redis：未运行则拉起（AOF，dir=data/；与 run_proxy.sh/ADR-0017 同款）。

    远程实例只探测不启动。返回最终可达性。
    """
    host, port = _redis_hostport(url)
    if _redis_ping(url):
        print(f"  Redis:        connected ({host}:{port})")
        return True
    if not _is_local_host(host):
        print(f"  Redis:        remote instance unreachable ({host}:{port}); make sure it is up")
        return False
    if shutil.which("redis-server") is None:
        print("  Redis:        redis-server not installed (brew install redis / apt install redis-server)")
        return False
    Path("data").mkdir(exist_ok=True)
    subprocess.run(  # noqa: S603
        ["redis-server", "--port", str(port), "--appendonly", "yes",
         "--appendfsync", "everysec", "--tcp-keepalive", "0",
         "--dir", "data/", "--daemonize", "yes"],
        check=False, capture_output=True)
    time.sleep(1.0)
    ok = _redis_ping(url)
    print(f"  Redis:        {'started (AOF, port ' + str(port) + ')' if ok else 'failed to start -- the proxy will forward but NOT store anything'}")
    return ok


def _consolidator_pid() -> int | None:
    """PID 文件优先 + 进程名兜底（手动启动场景，与 run_proxy.sh 同款）。

    进程名要匹配两种形态（ADR-0027 §4.2）：包内入口
    `python -m bladex_proxy.consolidator` 与老脚本 `run_index_consolidator.py`。
    """
    pid = _read_pidfile(_CONS_PIDFILE)
    if pid is not None:
        return pid
    for pattern in ("bladex_proxy.consolidator", "bladex-consolidator",
                    "run_index_consolidator.py"):
        try:
            out = subprocess.run(  # noqa: S603
                ["pgrep", "-f", pattern],
                capture_output=True, text=True, check=False).stdout.strip()
            if out:
                return int(out.splitlines()[0])
        except (OSError, ValueError, IndexError):
            continue
    return None


def _consolidator_command() -> list[str] | None:
    """consolidator 启动命令。

    优先包内入口（`python -m bladex_proxy.consolidator`）——它随 wheel 发布，
    因此**在任意目录、任意安装形态下都成立**；仓库脚本只作为未装包时的兜底。
    此前只找 `scripts/run_index_consolidator.py`，装包用户永远走到"跳过"分支，
    结果是一个从不提炼事实的部署（ADR-0027 §4.2）。
    """
    try:
        import bladex_proxy.consolidator  # noqa: F401
        return [sys.executable, "-m", "bladex_proxy.consolidator"]
    except ImportError:
        pass
    script = Path(_repo_root()) / "scripts" / "run_index_consolidator.py"
    if script.is_file():
        return [sys.executable, str(script)]
    return None


def _flash_command() -> list[str] | None:
    """flash 维护进程启动命令（V-F2；包内入口优先，与 consolidator 同理由）。"""
    try:
        import bladex_proxy.flash_daemon  # noqa: F401
        return [sys.executable, "-m", "bladex_proxy.flash_daemon"]
    except ImportError:
        return None


def _flash_pid() -> int | None:
    return _read_pidfile(_FLASH_PIDFILE)


def _start_flash(interval: float = 60.0) -> bool:
    # 🔴 默认值与 flash_daemon argparse 的 --interval 必须相等且**显式传下去**
    # （刚性原则 12：同一参数两条路两个默认值 = 缺陷；consolidator --concurrency 事故）。
    # 2026-08-29 起 --interval 语义 = 降级兜底轮询间隔（正常运行为 Redis wake 驱动，
    # 零轮询），15 → 60。
    """拉起 flash 维护进程（仅当 module_enabled("flash")——调用方守门）。"""
    if (pid := _flash_pid()) is not None:
        print(f"  Flash daemon: already running (PID {pid})")
        return True
    cmd = _flash_command()
    if cmd is None:
        print("  Flash daemon: failed to start -- no bladex_proxy.flash_daemon entry "
              "point. Reinstall bladex-proxy.")
        return False
    Path("logs").mkdir(exist_ok=True)
    logfile = f"logs/flash-{time.strftime('%Y%m%d-%H%M%S')}.log"
    with open(logfile, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(  # noqa: S603
            [*cmd, "--interval", str(interval)],
            stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
    Path(_FLASH_PIDFILE).write_text(str(proc.pid))
    time.sleep(1.0)
    if _pid_alive(proc.pid):
        print(f"  Flash daemon: started (PID {proc.pid}, interval {interval}s, log {logfile})")
        return True
    Path(_FLASH_PIDFILE).unlink(missing_ok=True)
    print(f"  Flash daemon: failed to start, see {logfile}")
    return False


def _resolve_distill_concurrency() -> int:
    """守护进程的蒸馏并发路数（唯一真相源 = `flags.BLADEX_DISTILL_CONCURRENCY`）。

    迟解析：`main()` 已跑过 `_load_env_file`，此刻 config/.env 的值可见。
    """
    from bladex_core.flags import flag_number
    return max(1, int(flag_number("BLADEX_DISTILL_CONCURRENCY")))


def _embed_pid() -> int | None:
    """embedding 模块 PID（PID 文件优先 + 进程名兜底，与 consolidator 同款）。"""
    pid = _read_pidfile(_EMBED_PIDFILE)
    if pid is not None and _pid_alive(pid):
        return pid
    Path(_EMBED_PIDFILE).unlink(missing_ok=True)
    try:
        out = subprocess.run(  # noqa: S603
            ["pgrep", "-f", "bladex_proxy.embed_server"],  # noqa: S607
            capture_output=True, text=True, timeout=3)
        for line in out.stdout.split():
            if line.strip().isdigit():
                return int(line.strip())
    except Exception:  # noqa: BLE001
        pass
    return None


def _embed_wanted() -> bool:
    """是否该起 embedding 模块：`BLADEX_MODULE_EMBED` 开、或 backend 已配 ipc。

    🔴 判据取**两者之一**而不是只看模块开关：用户可能先配 `backend=ipc` 试水
    却忘了翻模块开关，那样 proxy 会连不上 socket 然后回落 local（还多占一份内存）
    ——起不该起的进程只是浪费，不起该起的进程是静默降级。
    """
    from bladex_proxy.modules import module_enabled
    if module_enabled("embed"):
        return True
    return (os.environ.get("BLADEX_EMBED_BACKEND", "").strip().lower() == "ipc")


def _embed_listening() -> bool:
    """embedding 服务是否**真的在监听**（判活的正确判据，不是"进程还在"）。"""
    from bladex_proxy import embed_transport
    d = embed_transport.resolve()
    if d.transport == embed_transport.TRANSPORT_UDS:
        return os.path.exists(d.address)
    if d.transport == embed_transport.TRANSPORT_TCP:
        import socket
        host, _, port = d.address.rpartition(":")
        try:
            with socket.create_connection((host or "127.0.0.1", int(port)), timeout=1):
                return True
        except OSError:
            return False
    return False


def _start_embed() -> bool:
    """拉起 embedding 模块（V-A6）。未启用时直接返回 True（不是失败）。"""
    if not _embed_wanted():
        return True
    if (pid := _embed_pid()) is not None:
        print(f"  Embedding:    already running (PID {pid})")
        return True
    try:
        import bladex_proxy.embed_server  # noqa: F401
        cmd = [sys.executable, "-m", "bladex_proxy.embed_server"]
    except ImportError:
        print("  Embedding:    failed to start -- no bladex_proxy.embed_server entry "
              "point. Reinstall bladex-proxy.")
        return False
    Path("logs").mkdir(exist_ok=True)
    logfile = f"logs/embed-{time.strftime('%Y%m%d-%H%M%S')}.log"
    with open(logfile, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(  # noqa: S603
            cmd, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
    Path(_EMBED_PIDFILE).write_text(str(proc.pid))
    # 首次启动要加载模型（本地档 ~1–3s，冷缓存更久）——比 consolidator 多等一会儿。
    # 🔴 判活不能只看进程活着（2026-08-25 live：bind 失败进程秒退，而 sleep 期间
    # 它"还活着" ⇒ CLI 打了 "started" 但 status 立刻说没起——**"started" 骗人**）。
    # 判据改为**服务真的在监听**：uds 看 socket 文件出现，tcp 试连一次。
    deadline = time.monotonic() + 20.0
    listening = False
    while time.monotonic() < deadline:
        if not _pid_alive(proc.pid):
            break
        if _embed_listening():
            listening = True
            break
        time.sleep(0.3)
    if listening:
        print(f"  Embedding:    started (PID {proc.pid}, log {logfile})")
        return True
    Path(_EMBED_PIDFILE).unlink(missing_ok=True)
    print(f"  Embedding:    failed to start, see {logfile}")
    return False


def _start_consolidator(interval: int, concurrency: int | None = None) -> bool:
    """拉起 consolidator 守护进程。

    🔴 `--concurrency` 必须显式传下去（2026-08-22 事故）：此前这里只传
    `--interval`，而 `consolidator.py` 的 argparse 默认是 `0 → 1` 串行，
    于是 `bladex start` / `bladex consolidator start` 起来的守护进程**并发恒为 1**，
    而 `bladex sync` 那条路默认 4——手工 sync 快、守护进程慢，同一个参数两条路
    两个默认值。实测后果：mean 23.2s/次 × 8–9 次调用/轮串行 = 5–26 轮/小时，
    Memory Index 积压 720 轮持续不降（新记忆写不进去 = 召不回）。
    参数含义与取值依据见 `flags.MEMORY_NUMERIC_DEFAULTS`。
    """
    if (pid := _consolidator_pid()) is not None:
        print(f"  Consolidator: already running (PID {pid})")
        return True
    cmd = _consolidator_command()
    if cmd is None:
        print("  Consolidator: failed to start -- no bladex_proxy.consolidator entry point, "
              "and no scripts/run_index_consolidator.py. **No facts will ever be distilled.** "
              "Reinstall bladex-proxy, or run from the repository root.")
        return False
    if concurrency is None:
        concurrency = _resolve_distill_concurrency()
    Path("logs").mkdir(exist_ok=True)
    logfile = f"logs/consolidator-{time.strftime('%Y%m%d-%H%M%S')}.log"
    with open(logfile, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(  # noqa: S603
            [*cmd, "--interval", str(interval), "--concurrency", str(concurrency)],
            stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
    Path(_CONS_PIDFILE).write_text(str(proc.pid))
    time.sleep(1.5)
    if _pid_alive(proc.pid):
        print(f"  Consolidator: started (PID {proc.pid}, interval {interval}s, "
              f"concurrency {concurrency}, log {logfile})")
        return True
    Path(_CONS_PIDFILE).unlink(missing_ok=True)
    print(f"  Consolidator: failed to start, see {logfile}")
    return False


def _start_proxy(host: str, port: int, foreground: bool) -> int:
    cmd = [sys.executable, "-m", "uvicorn", "bladex_proxy.server:app",
           "--host", host, "--port", str(port)]
    Path("logs").mkdir(exist_ok=True)
    if foreground:
        print(f"  Proxy:        foreground at http://{host}:{port} (Ctrl-C to stop)")
        return subprocess.run(cmd, check=False).returncode  # noqa: S603
    logfile = f"logs/proxy-{time.strftime('%Y%m%d-%H%M%S')}.log"
    with open(logfile, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(  # noqa: S603
            cmd, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
    Path(_PROXY_PIDFILE).write_text(str(proc.pid))
    time.sleep(2.0)
    if _pid_alive(proc.pid):
        print(f"  Proxy:        started (PID {proc.pid}, http://{host}:{port}, log {logfile})")
        return 0
    Path(_PROXY_PIDFILE).unlink(missing_ok=True)
    print(f"  Proxy:        failed to start, see {logfile}")
    return 1


def _stop_by_pidfile(path: str, name: str, grace_s: int = 5) -> None:
    pid = _read_pidfile(path)
    if pid is None:
        print(f"  {name}: not running")
        Path(path).unlink(missing_ok=True)
        return
    os.kill(pid, signal.SIGTERM)
    for _ in range(grace_s):
        if not _pid_alive(pid):
            break
        time.sleep(1.0)
    if _pid_alive(pid):
        os.kill(pid, signal.SIGKILL)
    Path(path).unlink(missing_ok=True)
    print(f"  {name}: stopped (PID {pid})")


def _http_json(url: str, key: str = "", timeout: float = 3.0) -> dict | None:
    req = urllib.request.Request(url)  # noqa: S310 —— 仅本机 admin 面
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return None


def _print_admin_unavailable() -> None:
    """proxy 活着但 /admin/status 读不到时，说清为什么（2026-08-06 用户报的现象）。

    此前这里是 `if admin: ...` 后面什么都没有——读不到就**整段消失**：
    Memory Hub / Memory Index / Queue / Disk / Memory 五行凭空不见，
    只剩 Redis + Proxy + Dashboard。用户在别的目录跑了一次，看到的是一份
    "短了一半"的输出，第一反应是"程序版本不对"——这正是 ADR-0027 §3.1 说的
    绿灯骗人：**缺席比报错更难查**，因为缺席不指向任何东西。

    最常见的成因就是没找到部署根：没有 `config/.env` → 没有 admin key →
    `/admin/*` 401 → 整段静默消失。所以先报这一条。
    """
    root = deployment.find_root()
    if root is None:
        print("  Memory:       unavailable -- no config/.env found, so there is no admin key "
              "to read /admin/status with.")
        print(f"                Set BLADEX_HOME to your deployment directory (looked in "
              f"{', '.join(_config_search_paths())} and the parents of {os.getcwd()}).")
        return
    if not _admin_key():
        print(f"  Memory:       unavailable -- no BLADEX_ADMIN_KEYS/BLADEX_CLIENT_KEYS in "
              f"{os.path.join(root, deployment.CONFIG_RELPATH)}.")
        return
    print("  Memory:       unavailable -- /admin/status rejected the key or did not answer. "
          "The proxy may be running with a different config than this shell sees "
          "(compare `bladex config check`).")


def _first_key_of(raw: str) -> str:
    """取 `key||label||key2||label2` 里的第一把 key。

    ADR-0027 §5.2：格式权威是 `auth.KeyStore.parse`——**纯 `||` 分隔，不是逗号**。
    此前这里先按 `,` split，label 里带逗号就会截断出错误的 key（三处口径不一之一）。
    """
    parts = [p.strip() for p in raw.split("||") if p.strip()]
    return parts[0] if parts else ""


def _first_client_key() -> str:
    """数据面 key（/v1/*）。"""
    return _first_key_of(os.environ.get("BLADEX_CLIENT_KEYS", ""))


def _admin_key() -> str:
    """管理面 key（/admin/*）——ADR-0027 §2.2：优先独立 admin key，未配回落 client key。"""
    return _first_key_of(os.environ.get("BLADEX_ADMIN_KEYS", "")) or _first_client_key()


# ── start / stop / status / restart ─────────────────────────────────────────


@app.command()
def start(
    foreground: bool = typer.Option(False, "--foreground", help="Run the proxy in the foreground (default: background + PID file)"),
    proxy_only: bool = typer.Option(False, "--proxy-only", help="Start only the proxy (skip redis/consolidator)"),
    host: str = typer.Option("", help="Override BLADEX_HOST"),
    port: int = typer.Option(0, help="Override BLADEX_PORT"),
) -> int:
    """Start BladeX (redis + consolidator + proxy).

    收编 run_proxy.sh 的启动逻辑。
    """
    root = _require_config_root()
    if root is None:
        _print_config_not_found()
        return 1
    if root != os.getcwd():
        os.chdir(root)  # 之后的相对路径（data/、logs/、PID 文件）都以部署根为准
    from bladex_proxy.config import ProxyConfig
    cfg = ProxyConfig()
    host = host or cfg.host
    port = port or cfg.port

    print("=== BladeX start ===")
    _prune_logs()
    redis_ok = True
    consolidator_ok = True
    if not proxy_only:
        redis_ok = _ensure_redis(cfg.redis_url)
        # V-A6：embedding 模块要在 proxy/consolidator **之前**起——它俩启动时
        # 就会解析传输并尝试连接；晚起会让首个请求走一次回落 local（多占内存）。
        _start_embed()
        consolidator_ok = _start_consolidator(
            int(os.environ.get("BLADEX_CONSOLIDATOR_INTERVAL", "60") or 60))
        # V-F2：flash 维护进程随 start 拉起——module 门在此守（默认关=不起）。
        from bladex_proxy.modules import module_enabled as _me
        if _me("flash"):
            _start_flash()
    if _read_pidfile(_PROXY_PIDFILE) is not None:
        print(f"  Proxy:        already running (PID {_read_pidfile(_PROXY_PIDFILE)}) -- run `bladex stop` first")
        return 1
    rc = _start_proxy(host, port, foreground)
    if foreground or rc != 0:
        return rc
    return _report_startup_state(host, port, redis_ok, consolidator_ok)


def _prune_logs(keep: int | None = None) -> int:
    """启动时清理旧日志（ADR-0027 §4.4）。

    每次启动新建一个时间戳日志文件、从不清理 —— 审查时实测已积到 157 个。
    Memory Hub 无限增长是设计（ADR-0009），日志堆积不是。默认保留最近 20 个，
    `BLADEX_LOG_KEEP=0` 关闭清理。返回删除数量。
    """
    if keep is None:
        try:
            keep = int(os.environ.get("BLADEX_LOG_KEEP", "20"))
        except ValueError:
            keep = 20
    if keep <= 0:
        return 0
    log_dir = Path(_repo_root()) / "logs"
    if not log_dir.is_dir():
        return 0
    removed = 0
    # 按前缀分组保留（proxy / consolidator / sync 各留 keep 个，互不挤占）
    groups: dict[str, list[Path]] = {}
    for path in log_dir.glob("*.log"):
        groups.setdefault(path.name.split("-")[0], []).append(path)
    for files in groups.values():
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for stale in files[keep:]:
            try:
                stale.unlink()
                removed += 1
            except OSError:
                pass
    if removed:
        print(f"  Logs:         pruned {removed} old log file(s) (keeping {keep} per kind; "
              f"set BLADEX_LOG_KEEP=0 to disable)")
    return removed


def _probe_ready(url: str, timeout: float = 2.0) -> dict | None:
    """读 /ready，**503 也要读出 body**。

    /ready 在未就绪时返回 503 + `{"ready": false, "checks": {...}}`——正是我们最需要
    看清楚的那种情况。通用的 `_http_json` 把 HTTPError 当失败吞掉会让"记忆管线死了"
    退化成"探不到"，恰好丢掉本要报给用户的信息。
    """
    req = urllib.request.Request(url)  # noqa: S310 —— 本机探针
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            return json.loads(e.read().decode("utf-8"))
        except (ValueError, OSError):
            return {"ready": False, "checks": {}}
    except (urllib.error.URLError, OSError, ValueError):
        return None


def _report_startup_state(host: str, port: int, redis_ok: bool, consolidator_ok: bool,
                          timeout_s: float = 30.0) -> int:
    """轮询 /ready，把真实状态回给 console（ADR-0027 §3.1 契约 A）。

    旧行为：打印"Proxy 已启动"就 rc=0 收工。冷启动实测下这意味着——Redis 缺席
    （每轮对话都不入库）、consolidator 缺席（永远不会有事实被提炼）时，用户视角
    仍是"启动成功"，代理照常转发、对话完全正常，**没有任何界面告诉他记忆管线是死的**。

    三态：ready / degraded（proxy 活着但记忆管线不成立，rc=0 但写明后果）/ failed。
    """
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "") else host  # noqa: S104
    base = f"http://{probe_host}:{port}"
    deadline = time.monotonic() + timeout_s
    ready: dict | None = None
    while time.monotonic() < deadline:
        ready = _probe_ready(f"{base}/ready")
        if ready is not None:
            break
        time.sleep(1.0)

    if ready is None:
        print("\n  state: failed -- the proxy process is up but /ready did not answer within "
              f"{int(timeout_s)}s. Check the newest proxy log under logs/.")
        return 1

    checks = ready.get("checks", ready) if isinstance(ready, dict) else {}
    degraded: list[str] = []
    if not checks.get("redis", redis_ok):
        degraded.append("Redis unreachable -> turns are NOT stored; no memory will be produced"
                        " (install redis-server, then `bladex restart`)")
    if not checks.get("hub", True):
        degraded.append("Memory Hub not writable -> Hub writes fail; memory has no source"
                        " (check BLADEX_ROCKSDB_PATH permissions and whether another process holds the lock)")
    if not consolidator_ok:
        degraded.append("consolidator not running -> no facts will be distilled; injection stays hard-rules-only"
                        " (`bladex start` launches it; see logs/consolidator-*.log for failures)")
    if not checks.get("index", True):
        degraded.append("Memory Index not readable -> retrieval unavailable; injection falls back to hard rules only")

    print(f"\n  Dashboard:    {base}/dashboard")
    if degraded:
        print("  state: degraded -- the proxy forwards fine, but the memory pipeline is incomplete:")
        for d in degraded:
            print(f"    ⚠ {d}")
        print("  Run `bladex doctor` for a per-item check, or `bladex doctor --e2e` to verify the memory loop.")
    else:
        print("  state: ready -- memory pipeline is up (`bladex doctor --e2e` verifies it end to end)")
    return 0


@app.command()
def stop() -> int:
    """Stop the proxy and consolidator (local Redis is left running).

    Redis 是 Pipeline 数据面且被懒恢复依赖（ADR-0017），停它无收益。
    """
    print("=== BladeX stop ===")
    _stop_by_pidfile(_PROXY_PIDFILE, "Proxy")
    _stop_by_pidfile(_CONS_PIDFILE, "Consolidator")
    # 四个进程行**无条件汇报**（与 Proxy/Consolidator 一致）：第二次 stop 时
    # "这行消失了" 与 "not running" 传达的信息量完全不同（Jason 2026-08-28 复核）。
    _stop_by_pidfile(_FLASH_PIDFILE, "Flash daemon")
    _stop_by_pidfile(_EMBED_PIDFILE, "Embedding")
    print("  Redis:        left running (stop it with: redis-cli shutdown)")
    return 0


@app.command()
def embed(
    action: str = typer.Argument("status", help="start | stop | status | restart"),
) -> int:
    """Manage the embedding module (V-A6; independent process, IPC + hot/bulk queue).

    单独一条命令的理由与 consolidator 同：它是独立进程，换嵌入模型或调让路粒度
    （`BLADEX_EMBED_BULK_SLICE`）后**只重启它**即可，不必打断正在跑的会话。
    """
    action = action.lower().strip()
    if action not in {"start", "stop", "status", "restart"}:
        print(f"Unknown action: {action}. Use start | stop | status | restart.")
        return 2

    root = _require_config_root()
    if root is not None and root != os.getcwd():
        os.chdir(root)

    if action == "status":
        from bladex_proxy import embed_transport
        d = embed_transport.resolve()
        pid = _embed_pid()
        if not _embed_wanted():
            print("  Embedding:    off (module disabled; set BLADEX_MODULE_EMBED=1 "
                  "or BLADEX_EMBED_BACKEND=ipc)")
            return 0
        if pid is not None and not _embed_listening():
            print(f"  Embedding:    ⚠ process alive (PID {pid}) but not listening on "
                  f"{d.address} -- see logs/embed-*.log")
            return 1
        if pid is None:
            print(f"  Embedding:    ✗ not running (transport={d.transport} "
                  f"address={d.address}) -- proxy/consolidator will fall back to "
                  f"loading their own model")
            return 1
        print(f"  Embedding:    ✓ running (PID {pid}, transport={d.transport} "
              f"address={d.address}, reason={d.reason})")
        return 0

    if action in {"stop", "restart"}:
        _stop_by_pidfile(_EMBED_PIDFILE, "Embedding")
        if action == "stop":
            return 0

    return 0 if _start_embed() else 1


@app.command()
def consolidator(
    action: str = typer.Argument("status", help="start | stop | status | restart"),
    concurrency: int = typer.Option(
        -1, "--concurrency",
        help="Distillation concurrency (1=serial; defaults to BLADEX_DISTILL_CONCURRENCY, or 4)"),
) -> int:
    """Manage the Memory Index consolidator on its own (independent process, ADR-0009 §7).

    单独一条命令的理由：consolidator 是独立进程，改了配置或代码后**只重启它**
    是高频动作（重启 proxy 会打断正在跑的会话）。此前只能连 proxy 一起重启，
    或者去记 run_consolidator.sh 的路径。
    """
    action = action.lower().strip()
    if action not in {"start", "stop", "status", "restart"}:
        print(f"Unknown action: {action}. Use start | stop | status | restart.")
        return 2

    root = _require_config_root()
    if root is not None and root != os.getcwd():
        os.chdir(root)

    if action == "status":
        pid = _consolidator_pid()
        if pid is None:
            print("  Consolidator: ✗ not running -- no facts are being distilled")
            return 1
        print(f"  Consolidator: ✓ running (PID {pid})")
        return 0

    if action in {"stop", "restart"}:
        _stop_by_pidfile(_CONS_PIDFILE, "Consolidator")
        if action == "stop":
            return 0

    interval = int(os.environ.get("BLADEX_CONSOLIDATOR_INTERVAL", "60") or 60)
    conc = concurrency if concurrency >= 1 else None  # <1 → 走 flags 默认
    return 0 if _start_consolidator(interval, conc) else 1


def _fmt_traffic_row(detail: dict) -> str:
    """一行 turn 摘要（ADR-0027 §3.3）。字段全在 Memory Hub，只是此前没有 CLI 出口。"""
    ts = str(detail.get("timestamp") or detail.get("ts") or "")[:19].replace("T", " ")
    ident = detail.get("identity") or {}
    agent = ident.get("agent_id") or "?"
    injected = detail.get("injected_memory") or ""
    lines = [ln.strip() for ln in injected.splitlines() if ln.strip().startswith("-")]
    hard = sum(1 for ln in lines if "[MUST/NEVER]" in ln)
    items = len(lines) - hard
    dm = detail.get("decision_meta") or {}
    source = dm.get("source") or dm.get("route_source") or "-"
    model = dm.get("model") or dm.get("selected_model") or ""
    status_txt = (detail.get("response_meta") or {}).get("status") or detail.get("status") or "ok"
    return (f"  {ts:19s}  {agent:22.22s}  inject(h={hard},items={items}):<0  "
            f"route={source:12.12s} {model:22.22s} {status_txt}").replace(":<0", "")


def _scrape_metrics(url: str) -> dict[str, float]:
    """抓一个 Prometheus 端点，返回 {"metric{labels}": value}。抓不到返回空。"""
    import urllib.request

    out: dict[str, float] = {}
    try:
        with urllib.request.urlopen(url, timeout=3) as r:
            body = r.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 —— 端点没开是常态，不是错误
        return out
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # 用 rsplit 而不是 rpartition：`test_no_module_reimplements_env_parsing`
        # 禁止本模块出现 `partition`（旧的三份 .env 解析实现就是那个形状）。
        # 这里虽然解析的是 Prometheus 文本不是 .env，但守卫是源码级的，
        # 绕开它比削弱它便宜——守卫宁可误报也不该被放宽。
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        try:
            out[parts[0].strip()] = float(parts[1])
        except ValueError:
            continue
    return out


def _print_funnel(base: str) -> None:
    """记忆漏斗（M4-2）：把两个进程的端点拼成一条链。

    两端各自暴露（2026-08-06 拍板），所以这里要拉两个地址；好处是
    **哪半边没数一眼可见**——某一段为 0 而它上一段非 0，那一段就是断的，
    而这正是复核里 D7/E8/F6/G7/H7/I5 六条"缺观测"共同指向的东西。
    """
    import os

    from bladex_core.funnel import FUNNEL_ORDER

    write_port = os.environ.get("BLADEX_CONSOLIDATOR_METRICS_PORT", "").strip()
    samples = _scrape_metrics(f"{base}/metrics")
    if write_port and write_port != "0":
        samples.update(_scrape_metrics(f"http://127.0.0.1:{write_port}/metrics"))

    print("\n  Memory funnel (write side = consolidator, read side = proxy):")
    if not samples:
        print("    no metrics available -- is the proxy running?")
        return
    missing_write = not write_port or write_port == "0"
    for metric in FUNNEL_ORDER:
        total = sum(v for k, v in samples.items() if k.split("{")[0] == metric)
        label = metric.replace("bladex_memory_", "").replace("_total", "")
        print(f"    {label:<12} {int(total)}")
    if missing_write:
        print("    (write-side stages read 0 because the consolidator metrics endpoint "
              "is off -- set BLADEX_CONSOLIDATOR_METRICS_PORT to see them)")


def _print_traffic(base: str, n: int) -> int:
    """最近 n 轮的一行摘要（ADR-0027 §3.3 契约 C）。

    这是用户回答"它在工作吗 / 为什么没记住"的日常工具——此前只能翻日志或读 Memory Hub。
    """
    turn_index = _http_json(f"{base}/admin/turns?limit={max(1, min(n, 200))}", key=_admin_key())
    if turn_index is None:
        print("  Cannot read /admin/turns -- the proxy is not running, or the admin key is wrong"
              "（BLADEX_ADMIN_KEYS / BLADEX_CLIENT_KEYS）。")
        return 1
    turns = turn_index.get("turns", [])
    if not turns:
        print("  No recent turns. If you are chatting and still see nothing, the storage step is broken -- "
              "run `bladex doctor` and check Redis/Memory Hub.")
        return 0
    print(f"  Last {len(turns)} turns (of {turn_index.get('total', '?')}):")
    from urllib.parse import quote
    for item in turns:
        detail = _http_json(f"{base}/admin/turns/{quote(item['key'])}", key=_admin_key())
        if detail is None:
            print(f"  {item.get('ts', '')}  {item.get('key', '')}  <read failed>")
            continue
        print(_fmt_traffic_row(detail))
    return 0


def _fmt_bytes(n: int | None) -> str:
    """人类可读的字节数。None = 路径不存在（与 0 字节是两回事，显示 '-'）。"""
    if n is None:
        return "-"
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def _print_memory_config(admin: dict | None) -> None:
    """打印**运行中进程实际在用的**记忆配置，并对与 config/.env 的漂移报黄灯。

    为什么必须是"进程实际在用的"而不是"读 .env 再打印一遍"：
    2026-08-05 的事故正是 `.env` 写 200、进程跑 150（终端里残留了一个 export，
    而 `_load_env_file` 不覆盖已设环境变量）。每轮注入超时 1–3ms 被整段丢弃，
    记忆静默失效，而唯一痕迹是混在 info 流里的一行 warning。
    读配置文件打印出来的绿灯，恰恰是那次骗过所有人的东西。
    """
    mc = (admin or {}).get("memory_config") or {}
    eff = mc.get("effective") or {}
    if not eff:
        return
    flags = mc.get("flags") or {}
    off = [k.replace("BLADEX_", "").lower() for k, v in flags.items() if not v]
    print(f"  Memory:       hotpath {eff.get('BLADEX_HOTPATH_BUDGET_MS', '?')}ms"
          f" / inject top-{eff.get('BLADEX_INJECT_TOPK', '?')}"
          f" / min-importance {eff.get('BLADEX_MIN_IMPORTANCE', '?')}"
          + (f"  (off: {', '.join(off)})" if off else ""))

    drift = mc.get("drift") or []
    if drift:
        # 用户可见字符串必须是英文（ADR-0027 §5.1，有 AST 守卫钉着）；
        # 中文只留在注释里。
        print()
        print("  ⚠ Config drift: the running process is NOT using what config/.env declares")
        for d in drift:
            print(f"      {d['key']}: .env says {d['declared']}, process uses {d['effective']}")
        print("    Usually a stale export in the shell that started the proxy"
              " (env overrides .env).")
        print("    Fix: unset it, then `bladex restart` in that same shell"
              " (the daemon inherits its launcher's environment).")


def _consolidator_activity(admin: dict | None) -> str:
    """consolidator 那行的后缀：它此刻在干什么（2026-08-06 事故驱动）。

    事故形状：队列恢复后 487 轮在消化，一轮 30 条要跑几十分钟，而 status 里的
    `-> Index N waiting` 只在每轮提交时跳一次——轮内那半小时纹丝不动，和进程卡死
    长得一模一样，只能 tail 日志才能确认它在干活。所以把轮内进度接出来。

    三条不能含糊的边界：
      · 没有心跳 → 一个字都不加。老版本 consolidator 不写心跳，"读不到"不等于"没在跑"。
      · 心跳不新鲜 → 只说多久没动了，不宣称在跑也不宣称死了（可能卡在一次长上游调用里）。
      · state=error → 直接说出错，别让它混在"正在蒸馏"里被当成进展。
    """
    hb = (admin or {}).get("consolidator") or {}
    if not hb:
        return ""
    state, phase = hb.get("state", ""), hb.get("phase", "")
    if not hb.get("fresh", False):
        age = hb.get("last_seen_s")
        return f" -- no heartbeat for {age}s (was: {state or 'unknown'})" if age else ""
    if state == "error":
        return f" -- last batch failed: {(hb.get('error') or '')[:80]}"
    done, total, batch = hb.get("done"), hb.get("total"), hb.get("batch")
    if phase == "distill" and isinstance(total, int) and total > 0:
        return f" -- distilling batch {batch}, {done}/{total} units"
    if phase == "scan":
        return f" -- batch {batch}, scanning the Hub"
    if state == "idle":
        return " -- idle, waiting for new turns"
    return f" -- {phase or state}" if (phase or state) else ""


def _format_queue(admin: dict | None, consolidator_running: bool) -> str:
    """两段管线各有多少轮在排队：Pipeline -> Memory Hub -> Memory Index。

    为什么单列一行（2026-08-06 事故驱动）：此前 lag 只是 Memory Index 那行末尾的一个
    尾巴，而 **Pipeline 侧的积压在 CLI 里根本没有出口**。08-05 `doctor --e2e` 卡在
    step 3（探针轮 180s 内没蒸出 fact），真实原因是 972 轮积压排在它前面——但 status
    看不出"排了多少、往哪一段排、有没有人在消化"，只能去翻 consolidator 日志。

    三个字段都来自 /admin/status（早就有，只是没打出来）：
      pipeline.backlog   Redis stream 里等着写进 Hub 的轮次
      index.lag_turns    已入 Hub、还没被蒸馏进 Index 的轮次
      overflow/spill     Redis 不可达时落盘的轮次（回灌前不产生记忆）

    孤儿单列（2026-08-06 第二次事故驱动）：backlog 里可能有一部分**谁都不会再碰**
    （已 XACK、未 XDEL 的残留，见 PipelineRedis.inspect）。它们和"正在排队"混在一个数里，
    症状是这个数永远不降——而原来的后缀还会安慰你一句"the pipeline worker is draining it"，
    并且只要 Index 侧有一条 lag，那句话连出现的机会都没有。所以孤儿必须自己有一句话，
    且不被前面任何分支吞掉。
    """
    pipeline = (admin or {}).get("pipeline") or {}
    index = (admin or {}).get("index") or {}

    def _int(value: object) -> int | None:
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    to_hub = _int(pipeline.get("backlog"))
    to_index = _int(index.get("lag_turns"))
    orphan = _int(pipeline.get("orphan")) or 0
    moving = None if to_hub is None else max(0, to_hub - orphan)
    line = (f"-> Hub {moving if moving is not None else '?'} waiting"
            f" / -> Index {to_index if to_index is not None else '?'} waiting")

    # 状态后缀按"谁该动而没动"排序：积压 + 没人消化，是最需要被看见的一种
    if to_index:
        line += (" -- catching up" if consolidator_running
                 else " -- consolidator is not running, so nothing is being distilled")
    elif moving:
        line += " -- the pipeline worker is draining it"
    elif moving == 0 and to_index == 0 and not orphan:
        line += " -- caught up"

    # 孤儿这句独立于上面所有分支：它不是"排队慢"，是"排不动"，两件事不能共用一句措辞
    if orphan:
        line += (f" / {orphan} stale entries nobody will ever pick up"
                 " -- run `bladex queue flush`")

    spilled = (_int(pipeline.get("overflow_count")) or 0) + (_int(pipeline.get("spill_count")) or 0)
    if spilled:
        line += f" / {spilled} spilled to disk (replayed once Redis is back)"
    return line


def _flash_status_line() -> str:
    """Memory Flash 那一行。从**本机文件系统**读，不经 /admin。

    理由：Flash 的写者是 bladex-flash 维护进程、载体是文件，proxy 起没起与它无关——
    经 admin 中转反而会让"proxy 挂了"伪装成"Flash 没渲染"。
    （2026-09-03 S5：老渲染器开关 BLADEX_FLASH_RENDER 已删，计数改按 ADR-0032 布局的账本池。）

    🔴 措辞上把两件事分开：显示的是"最近一次**内容变了**"，不是"最近一次渲染"。
    `write_if_changed` 内容没变就不写，所以旧 mtime 的正常含义是"这段时间没有新东西"，
    不是"渲染器停了"。据此报警 = 尺子按自己的假设判读被测系统（本仓库的高发病）。
    """
    import time

    from bladex_proxy.config import resolve_flash_path

    # V-F2：daemon 状态前置（module 开时 daemon 是 Flash 的唯一写者，
    # 它没起 = 账本池投影不更新，比"渲染了没有"更要紧）。
    from bladex_proxy.modules import module_enabled as _me
    daemon_prefix = ""
    if _me("flash"):
        pid = _flash_pid()
        # 风格与 Consolidator/Embedding 行统一：✓/✗ + (PID) + " -- " 细节
        daemon_prefix = (f"✓ running (PID {pid}) -- " if pid is not None
                         else "✗ not running (bladex start brings it up) -- ")

    root = resolve_flash_path()
    from bladex_core.flash import summarize_tree

    s = summarize_tree(root)
    if not s.principals:
        return daemon_prefix + f"nothing materialized yet ({root}) -- bladex-flash writes it"
    parts = [f"{s.ledgers} ledgers"]
    # 用户手写文件单列：它是"你改过什么"的入口，也是我们**永不覆盖**的那部分
    parts.append(f"{s.user_files} yours" if s.user_files else "no overrides yet")
    line = daemon_prefix + f"{' / '.join(parts)} ({root})"
    if s.newest_mtime:
        line += f", newest change {_ago(time.time() - s.newest_mtime)}"
    return line


def _ago(seconds: float) -> str:
    """人读的相对时间（英文，用户可见面）。"""
    seconds = max(0.0, seconds)
    if seconds < 90:
        return f"{int(seconds)}s ago"
    if seconds < 5400:
        return f"{int(seconds // 60)}m ago"
    if seconds < 172800:
        return f"{int(seconds // 3600)}h ago"
    return f"{int(seconds // 86400)}d ago"


def _print_degradation_banner(admin: dict | None) -> None:
    """静默降级的累计告警面（ADR-0027 §3.3）。

    inject_timeout / enqueue_skipped / consolidator lag 都是"记忆安静地失效"的形状，
    此前只散在日志里。非零就顶部黄字，让降级不再静默。
    """
    if not admin:
        return
    warnings: list[str] = []
    degr = admin.get("degradation") or {}
    inject_timeouts = degr.get("inject_timeout_1h", admin.get("inject_timeout_1h"))
    enqueue_skipped = degr.get("enqueue_skipped_1h", admin.get("enqueue_skipped_1h"))
    lag = (admin.get("index") or {}).get("lag_turns")
    if inject_timeouts:
        # 教训 40（2026-08-17）：这句提示此前只说"抬预算"，于是三次会话被自己的
        # 告警带着走，把预算从 100 一路抬到 250，而真实成因是 LanceDB 索引碎片
        # 让检索慢了 15 倍（6462ms -> 414ms）。抬预算还会掩盖读数：deadline 一砍，
        # `elapsed_ms` 恒等于 budget + ε，真实耗时永远测不到。**先查碎片。**
        warnings.append(f"in the last 1h: {inject_timeouts} turns degraded on inject timeout (hard rules only)"
                        " -- check index fragmentation first (`bladex-consolidator` compacts it;"
                        " scripts/compact_index.py reports it), then consider raising"
                        " BLADEX_HOTPATH_BUDGET_MS or narrowing retrieval")
    if enqueue_skipped:
        warnings.append(f"in the last 1h: {enqueue_skipped} turns spilled to disk (Redis unreachable)"
                        " -- they produce no memory until Redis is back and they are replayed")
    index_unavailable = degr.get("index_unavailable_1h")
    if index_unavailable:
        warnings.append(f"in the last 1h: {index_unavailable} lookups hit an unavailable Memory Index layer"
                        " -- only hard rules were injected; check the embedding backend")
    if isinstance(lag, int) and lag > 500:
        warnings.append(f"consolidator is behind by {lag} turns -- distillation cannot keep up; new memory stays unrecallable")
    for w in warnings:
        print(f"  ⚠ {w}")
    if warnings:
        print()


@app.command()
def status(
    traffic: bool = typer.Option(False, "--traffic",
                                 help="Show a one-line summary of the most recent turns"),
    n: int = typer.Option(10, "-n", help="How many turns to show with --traffic"),
) -> int:
    """Show runtime status: processes, /ready probe, and an /admin/status summary."""
    from bladex_proxy.config import ProxyConfig
    cfg = ProxyConfig()
    exit_code = 0

    redis_ok = _redis_ping(cfg.redis_url)
    print(f"  Redis:        {'✓ running' if redis_ok else '✗ unreachable'} ({cfg.redis_url})")

    base = f"http://{cfg.host}:{cfg.port}"
    # admin 提前取：consolidator 那行要用心跳，而它排在 proxy 行之前。
    # proxy 没起时返回 None，下面照常按"没有心跳"渲染。
    admin = _http_json(f"{base}/admin/status", key=_admin_key())

    cons_pid = _consolidator_pid()
    print(f"  Consolidator: {'✓ running (PID ' + str(cons_pid) + ')' if cons_pid else '✗ not running'}"
          + _consolidator_activity(admin))
    # V-A6：embedding 模块（未启用时明说"未启用"，与"挂了"区分开）
    if _embed_wanted():
        epid = _embed_pid()
        # 判据与 `_start_embed` 同款：**在监听**才算 running（进程活着不等于服务可用）
        if epid and _embed_listening():
            print(f"  Embedding:    ✓ running (PID {epid})")
        elif epid:
            print(f"  Embedding:    ⚠ process alive (PID {epid}) but not listening "
                  f"-- see logs/embed-*.log")
        else:
            print("  Embedding:    ✗ not running (backend=ipc needs it; `bladex start`)")
    else:
        print("  Embedding:    off (module disabled; set BLADEX_MODULE_EMBED=1)")
    # Memory Flash 紧跟 Consolidator：它是那个进程的产物，且与 proxy 死活无关
    # （所以放在下面 proxy 那条 `return 1` 之前——proxy 挂了也该看得见）。
    print(f"  {'Memory Flash:':<14}{_flash_status_line()}")

    health = _http_json(f"{base}/health")
    if health is None:
        print(f"  Proxy:        ✗ not running ({base})")
        return 1
    ready = _http_json(f"{base}/ready") or {}
    print(f"  Proxy:        ✓ running ({base}, ready={ready.get('ready')})")

    print(f"  Dashboard:    {base}/dashboard")

    if admin:
        index, hub = admin.get("index", {}), admin.get("hub", {})
        print(f"  {'Memory Hub:':<14}{hub.get('turns', '?')} turns")
        print(f"  {'Memory Index:':<14}{index.get('facts', '?')} facts"
              f" / {index.get('matters', '?')} matters")
        print(f"  {'Queue:':<14}{_format_queue(admin, cons_pid is not None)}")
        disk = admin.get("disk") or {}
        if any(v for v in disk.values()):
            print(f"  Disk:         Memory Hub {_fmt_bytes(disk.get('hub_bytes'))}"
                  f" / Memory Index {_fmt_bytes(disk.get('index_bytes'))}"
                  f" / Memory Flash {_fmt_bytes(disk.get('flash_bytes'))}"
                  f" / overflow {_fmt_bytes(disk.get('overflow_bytes'))}"
                  f" / logs {_fmt_bytes(disk.get('logs_bytes'))}")
        unhealthy = admin.get("upstream", {}).get("unhealthy_models", [])
        if unhealthy:
            print(f"  Upstream tripped: {', '.join(unhealthy)}")
        _print_memory_config(admin)
    else:
        _print_admin_unavailable()
    print()
    _print_degradation_banner(admin)

    if traffic:
        exit_code = _print_traffic(base, n) or exit_code
        _print_funnel(base)
    return exit_code


@app.command()
def restart(
    foreground: bool = typer.Option(False, "--foreground"),
    proxy_only: bool = typer.Option(False, "--proxy-only"),
) -> int:
    """Restart (stop + start)."""
    stop()
    # 直调需给全参数：typer 命令函数的缺省值是 OptionInfo 哨兵，不经 click 不解析
    return start(foreground=foreground, proxy_only=proxy_only, host="", port=0)


# ── init ────────────────────────────────────────────────────────────────────

_ENV_MINIMAL_TEMPLATE = """# BladeX configuration (generated by `bladex init`; full options in config/.env.example)
BLADEX_HOST=127.0.0.1
BLADEX_PORT={port}

# Upstream (Router model naming: provider/model)
BLADEX_UPSTREAM_MODEL={model}
BLADEX_UPSTREAM_API_BASE={api_base}
BLADEX_UPSTREAM_API_KEY={api_key}

# Client key auth. Format: key||label, MULTIPLE KEYS ALSO SEPARATED BY || :
#   BLADEX_CLIENT_KEYS=key1||laptop||key2||codex
# (there is no comma-separated form -- see auth.KeyStore.parse)
BLADEX_AUTH_ENABLED={auth_enabled}
BLADEX_CLIENT_KEYS={client_keys}
# Management plane key (/admin/*). Same format. When empty, the admin plane falls
# back to BLADEX_CLIENT_KEYS -- meaning any agent key can run destructive admin
# operations. Set this to separate the two planes (required with identity.toml).
BLADEX_ADMIN_KEYS={admin_keys}

# Storage
BLADEX_REDIS_URL=redis://127.0.0.1:6379/0
BLADEX_ROCKSDB_PATH=data/bladex_hub
BLADEX_INDEX_PATH=data/bladex_index

# Routing (config-driven strategies, see config/routing.toml)
BLADEX_ROUTE_ENABLED=false

# Memory mechanisms (ADR-0026; on by default -- set to 0 to roll back)
BLADEX_SOFT_SCORING=1
BLADEX_HOP_EXPAND=1
BLADEX_PROFILE_CARDS=1
BLADEX_HOTPATH_BUDGET_MS=200
"""

_ROUTING_MINIMAL_TEMPLATE = """# BladeX routing config (generated by `bladex init`; full example in config/routing.toml.example)
# Takes effect once BLADEX_ROUTE_ENABLED=true. Store only the env var NAME here, never the value.

[[models]]
name = "{model}"
tier = "medium"
capabilities = ["text", "code"]
api_base = "{api_base}"
api_key_env = "BLADEX_UPSTREAM_API_KEY"

[upstream]
default = "{model}"
"""


#: 🔴 模型面运行时资产（MQ-A41，2026-09-05）：`config/system/{AGENT,TOOLS,SKILLS}.md` 与
#: `config/ledger-template.md` **不是文档，是注入进模型上下文的运行时资产**。
#: 三份 system notes 全缺 ⇒ `load_system_notes()` 返回 ""，调用方据此**整块不注** ⇒
#: 模型不知道 BladeX 存在、不知道有工具面——而"对后端 LLM 可见"是 v5 的核心定位。
#:
#: G7 冷安装实跑撞到：wheel 里只有 .py（`pyproject.toml` 无 package-data），
#: `init` 也只写 `.env` + `routing.toml` ⇒ **pip 装出来的部署永远是"半个 BladeX"**。
#: 单一真相源放在**包内** `bladex_proxy/assets/`（随 wheel 走，任何安装形态都在），
#: 仓库 `config/` 下那份是同一内容的部署副本，由 `test_model_facing_assets.py` 逐字对账。
_ASSET_TARGETS: tuple[tuple[str, str], ...] = (
    ("assets/system/AGENT.md", "config/system/AGENT.md"),
    ("assets/system/TOOLS.md", "config/system/TOOLS.md"),
    ("assets/system/SKILLS.md", "config/system/SKILLS.md"),
    ("assets/ledger-template.md", "config/ledger-template.md"),
)


def assets_dir() -> Path:
    """包内模型面资产目录（`bladex_proxy/assets`）。"""
    return Path(__file__).resolve().parent / "assets"


def _install_model_facing_assets(root: Path, *, force: bool = False) -> list[str]:
    """把包内资产铺到部署根的 `config/`。已存在则**不覆盖**（用户可能改过 AGENT.md）。

    返回给用户看的行；缺源文件时**响亮报出**而不是静默跳过——
    源不在 = wheel 打包漏了，那正是本条要防的形态。
    """
    out: list[str] = []
    for rel_src, rel_dst in _ASSET_TARGETS:
        src = assets_dir().joinpath(*rel_src.split("/")[1:])
        dst = root / rel_dst
        if not src.is_file():
            out.append(f"  ! model-facing asset missing from the package: {rel_src} "
                       f"-- the wheel is incomplete, please report it")
            continue
        if dst.exists() and not force:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        out.append(f"✓ wrote {dst}")
    return out


@app.command()
def init(
    yes: bool = typer.Option(False, "--yes", help="Non-interactive (use defaults / passed options)"),
    force: bool = typer.Option(False, "--force", help="Overwrite existing config files"),
    upstream_model: str = typer.Option("", help="Upstream model (provider/model)"),
    api_base: str = typer.Option("", help="Upstream API base URL"),
    api_key: str = typer.Option("", help="Upstream API key (written to config/.env, never committed)"),
    port: int = typer.Option(38080, help="Proxy listen port"),
    no_auth: bool = typer.Option(False, "--no-auth",
                                 help="Do not generate client/admin keys (auth off; loopback only)"),
) -> int:
    """Generate config/.env, config/routing.toml and the model-facing assets.

    「模型面资产」= `config/system/{AGENT,TOOLS,SKILLS}.md` + `config/ledger-template.md`
    （MQ-A41，见 `_ASSET_TARGETS`）。首行必须英文——typer 拿它当命令 help
    （`test_user_facing_english.py` 守着，本次改动当场被它抓到一次）。

    ADR-0027 §2.4/§5.2：生成的配置必须是"能直接用且安全的"——
    自动生成 client key + admin key 并开启 auth（README 让 agent 带 key，
    init 不给 key 会让新用户第一步就卡住）；`.env` 落盘后 chmod 0600（含上游 key）；
    Memory Hub 路径与 `ProxyConfig` 默认一致（此前模板写 data/bladex_p3，老用户 `--force`
    重建后 proxy 指向空 Memory Hub = "记忆全没了"的工单形状）。
    """
    env_path = Path(_repo_root()) / "config" / ".env"
    routing_path = Path(_repo_root()) / "config" / "routing.toml"
    if env_path.exists() and not force:
        print(f"{env_path} already exists (use --force to overwrite). "
              f"routing.toml: {'present' if routing_path.exists() else 'missing'}")
        return 1

    if not yes:
        upstream_model = typer.prompt("Upstream model (Router format provider/model)",
                                      default=upstream_model or "openai/gpt-4o-mini")
        api_base = typer.prompt("Upstream API base (leave empty for OpenAI)", default=api_base or "")
        api_key = typer.prompt("Upstream API key", default=api_key or "", hide_input=True)
        port = int(typer.prompt("Proxy listen port", default=str(port)))
    else:
        upstream_model = upstream_model or "openai/gpt-4o-mini"

    client_key = "" if no_auth else f"bladex-{secrets.token_hex(12)}"
    admin_key = "" if no_auth else f"bladex-admin-{secrets.token_hex(12)}"

    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text(_ENV_MINIMAL_TEMPLATE.format(
        model=upstream_model, api_base=api_base, api_key=api_key, port=port,
        auth_enabled="false" if no_auth else "true",
        client_keys="" if no_auth else f"{client_key}||default",
        admin_keys="" if no_auth else f"{admin_key}||admin"),
        encoding="utf-8")
    try:
        env_path.chmod(0o600)  # 含上游 key 与 client key，不给同机其它用户读
    except OSError as e:
        print(f"  ! could not chmod 600 {env_path}: {e}")
    print(f"✓ wrote {env_path} (mode 0600)")

    if not routing_path.exists() or force:
        routing_path.write_text(_ROUTING_MINIMAL_TEMPLATE.format(
            model=upstream_model, api_base=api_base), encoding="utf-8")
        print(f"✓ wrote {routing_path} (takes effect once BLADEX_ROUTE_ENABLED=true)")

    for line in _install_model_facing_assets(Path(_repo_root()), force=force):
        print(line)

    if client_key:
        print("\nKeys generated (also stored in config/.env):")
        print(f"  client key (agents -> /v1/*):   {client_key}")
        print(f"  admin key  (CLI/dashboard/MCP): {admin_key}")
    if not api_key:
        print("\n⚠ BLADEX_UPSTREAM_API_KEY is empty -- requests will fail until you set it "
              f"in {env_path}")
    for stale in _stale_data_dirs():
        print(f"\n⚠ found existing data directory {stale} which is NOT referenced by this config. "
              "If that holds your memory, point BLADEX_ROCKSDB_PATH/BLADEX_INDEX_PATH at it "
              "instead of starting from an empty store.")
    print("\nNext: `bladex doctor` -> `bladex start`")
    return 0


def _stale_data_dirs() -> list[str]:
    """既有 data/ 子目录里，没有被当前配置引用的存储目录。

    防的是"配置路径漂移 → 指向空库 → 用户以为记忆全丢了"（ADR-0027 §5.2）。
    """
    data_dir = Path(_repo_root()) / "data"
    if not data_dir.is_dir():
        return []
    referenced = {
        os.path.basename(os.environ.get("BLADEX_ROCKSDB_PATH", "data/bladex_hub").rstrip("/")),
        os.path.basename(os.environ.get("BLADEX_INDEX_PATH", "data/bladex_index").rstrip("/")),
        "overflow", "backups",
    }
    out = []
    for entry in sorted(data_dir.iterdir()):
        if entry.is_dir() and entry.name not in referenced and any(entry.iterdir()):
            out.append(str(entry))
    return out


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


# ── doctor ──────────────────────────────────────────────────────────────────


def _probe_upstream(cfg, timeout_s: float = 5.0) -> tuple[str, str, str]:
    """带凭证探一次上游（ADR-0027 §3.1）。

    返回 `(state, detail, fix)`，state ∈:
      ok          — 拿到 2xx，凭证与网络都成立
      auth_failed — 拿到 401/403，网络通但 key 不对/为空
      unreachable — 网络层不通
      not_probed  — 没有可探的地址（走 Router 默认 endpoint），如实说未探测

    只打 `GET {api_base}/models`（OpenAI 兼容面通用、只读、无 token 消耗）。
    """
    api_base = (cfg.upstream_api_base or "").rstrip("/")
    api_key = (cfg.upstream_api_key or "").strip()
    if not api_base:
        return ("not_probed",
                "no BLADEX_UPSTREAM_API_BASE (Router default endpoint); cannot probe from here",
                "")
    if not api_key:
        return ("auth_failed", f"{api_base} -- BLADEX_UPSTREAM_API_KEY is empty",
                "set BLADEX_UPSTREAM_API_KEY in config/.env")

    url = f"{api_base}/models"
    req = urllib.request.Request(url)  # noqa: S310 —— 用户自配的上游地址
    req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310
            return ("ok", f"{api_base} (HTTP {resp.status})", "")
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            return ("auth_failed", f"{api_base} -- HTTP {e.code} (key rejected)",
                    "check BLADEX_UPSTREAM_API_KEY (and that the key matches this api_base)")
        # 404/405 等：说明服务在、只是没有 /models 这个路径 —— 凭证无法断言，但网络与
        # 端点都成立，不误报为失败。
        return ("ok", f"{api_base} (HTTP {e.code} on /models; endpoint reachable)", "")
    except (urllib.error.URLError, OSError, ValueError) as e:
        return ("unreachable", f"{api_base} -- {e}",
                "check BLADEX_UPSTREAM_API_BASE, network and proxy settings")


def _probe_port(host: str, port: int) -> tuple[str, str]:
    """端口状态 -> (state, detail)。

    state: free / ours / foreign / error

    为什么不是简单的"占用就警告"（2026-08-05 C7 实测）：`bladex start` 之后跑 doctor
    必然看到端口被占——那是**健康态**，旧实现却无条件打 ⚠。假红灯与假绿灯一样有害，
    它训练用户忽略警告，而 ADR-0027 §3 的整条契约就是"灯要说真话"。所以占用时先探
    `/ready`：应答的是我们自己的 proxy（正常），不应答才是真冲突（start 会失败）。
    """
    bind_host = host if host != "0.0.0.0" else "127.0.0.1"  # noqa: S104
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1.0)
        in_use = s.connect_ex((bind_host, port)) == 0
        s.close()
    except OSError as e:
        return "error", str(e)
    if not in_use:
        return "free", f"{port} free"
    if _probe_ready(f"http://{bind_host}:{port}/ready") is not None:
        pid = _read_pidfile(_PROXY_PIDFILE)
        return "ours", f"{port} serving this proxy" + (f" (PID {pid})" if pid else "")
    return ("foreign",
            f"port {port} is in use by something that is not a BladeX proxy "
            f"(`bladex start` will fail -- free the port or set BLADEX_PORT)")


@app.command()
def doctor(
    warmup: bool = typer.Option(False, "--warmup", help="Pre-download / load the embedding model (first run can be slow)"),
    e2e: bool = typer.Option(False, "--e2e",
                             help="End-to-end memory loop check: send a probe turn, wait for it "
                                  "to be distilled, then recall it semantically"),
    e2e_timeout: float = typer.Option(180.0, "--e2e-timeout",
                                      help="Seconds to wait for distillation (--e2e)"),
) -> int:
    """Health check: redis / RocksDB / embedding model / upstream / port."""
    from bladex_proxy.config import ProxyConfig
    cfg = ProxyConfig()
    failures = 0

    def _check(label: str, ok: bool, detail: str = "", fix: str = "") -> None:
        nonlocal failures
        mark = "✓" if ok else "✗"
        print(f"  {mark} {label}" + (f" ({detail})" if detail else ""))
        if not ok:
            failures += 1
            if fix:
                print(f"    fix: {fix}")

    _check("config/.env", Path("config/.env").is_file(),
           fix="bladex init")

    # 🔴 MQ-A41：模型面资产在位（system notes 三份 + 账本模板）。
    # 缺了不致命但是**降级**：`load_system_notes()` 返回 "" ⇒ 自我介绍整块不注 ⇒
    # 模型不知道 BladeX 存在、不知道有工具面。G7 冷安装实跑时这件事只在启动日志里
    # 有两条 warning，doctor 全绿 —— 判据比承诺窄，所以补这一条。
    _missing_assets = [dst for _src, dst in _ASSET_TARGETS if not Path(dst).is_file()]
    _check("Model-facing assets", not _missing_assets,
           "all present" if not _missing_assets
           else f"{len(_missing_assets)} missing: {', '.join(_missing_assets)}",
           fix=("bladex init copies them from the package. Without config/system/*.md the "
                "self-introduction is not injected at all -- the model will not know BladeX "
                "exists or that it has tools."))

    hidden_pth = _hidden_pth_files()
    _check("Python path files (.pth)", not hidden_pth,
           f"{len(hidden_pth)} hidden .pth in site-packages" if hidden_pth else "",
           fix=("macOS marked them hidden and Python 3.11+ SILENTLY skips hidden .pth files, "
                "so installed packages become unimportable. Run: chflags nohidden "
                "\"$(python -c 'import site;print(site.getsitepackages()[0])')\"/*.pth "
                "-- if the flag comes back, the folder is on iCloud/OneDrive: move the "
                "project (and its virtualenv) outside the synced folder."))

    _check("Redis", _redis_ping(cfg.redis_url), cfg.redis_url,
           fix="`bladex start` launches a local Redis; or run redis-server --appendonly yes yourself")

    # Memory Hub RocksDB：secondary 只读探测（不与运行中的 proxy 抢主锁）
    ledger_dir = Path(cfg.rocksdb_path)
    if not ledger_dir.exists():
        _check("Memory Hub RocksDB", True, "no data yet (created on first start)")
    else:
        try:
            from bladex_proxy.storage.memory_hub import MemoryHub
            hub = MemoryHub(cfg.rocksdb_path, secondary=True)
            hub.open()
            n = hub.count()
            hub.close()
            _check("Memory Hub RocksDB", True, f"{n} turns")
        except Exception as e:  # noqa: BLE001
            _check("Memory Hub RocksDB", False, str(e),
                   fix="directory corrupted or version mismatch; back up, then check permissions under data/")

    # embedding 模型
    try:
        from bladex_proxy.embedding import (
            _model_cached,
            resolve_embed_settings,
        )
        settings = resolve_embed_settings(cfg)
        if settings.backend == "local":
            model = settings.model or "(default)"
            cached = _model_cached(settings.model, cfg.fastembed_cache_path) if settings.model else False
            if warmup:
                from bladex_proxy.embedding import build_embedder
                print(f"  … loading embedding model {model} (downloaded automatically if missing; please wait)")
                emb = build_embedder(cfg, role="proxy")
                emb.embed(["warmup"])
                _check("Embedding", True, f"local/{model} ready")
            else:
                _check("Embedding", True,
                       f"local/{model} {'cached' if cached else 'not cached (downloaded on first start, or run doctor --warmup)'}")
        else:
            _check("Embedding", True, f"{settings.backend} backend (remote; no local cache check)")
    except Exception as e:  # noqa: BLE001
        _check("Embedding", False, str(e),
               fix="check BLADEX_EMBED_* and the [embedding] section of routing.toml; set HF_ENDPOINT for a mirror if the network is blocked")

    # 上游（ADR-0027 §3.1 契约 A）：带 key 做一次真实鉴权探测，不再只测网络层。
    # 旧行为是 urlopen(api_base) 且 api_base 为空直接打 ✓ —— 空 key 也是绿灯，
    # 而这正是新用户最常见的失败原因（冷启动实测）。
    state, detail, fix = _probe_upstream(cfg)
    if state == "ok":
        _check("Upstream", True, detail)
    elif state == "not_probed":
        # 探不了就如实说"未探测"，不打 ✓ —— 绿灯不许骗人。
        print(f"  · Upstream: not probed ({detail})")
    else:
        _check("Upstream", False, detail, fix=fix)

    # 端口
    state, detail = _probe_port(cfg.host, cfg.port)
    if state == "free":
        _check("Port", True, detail)
    elif state == "ours":
        _check("Port", True, detail)
    elif state == "foreign":
        print(f"  ⚠ {detail}")
    else:
        _check("Port", False, detail)

    if failures:
        print(f"Health check found {failures} problem(s).")
        if e2e:
            print("  (fix the problems above before running --e2e)")
        return 1
    print("Health check passed.")

    if e2e:
        return _run_e2e_check(cfg, timeout_s=e2e_timeout)
    return 0


def _hidden_pth_files() -> list[str]:
    """site-packages 里带 macOS `UF_HIDDEN` 的 `.pth` 文件。

    为什么值得单独查（2026-08-05 真实事故）：Python 3.11+ 的 `site.addpackage()`
    对带 hidden 标志的 `.pth` **静默 return**——不报错、不警告。后果是
    editable 安装的包全部导不进来，症状是 `ModuleNotFoundError`，而
    `pip list` / dist-info / `.pth` 内容**全都看起来正常**，指不到真因。

    触发场景：项目放在 iCloud Drive / OneDrive 等同步目录里（macOS 会给
    这些文件打 hidden）。非 macOS 上 `st_flags` 不存在，本检查恒为空。
    """
    import site
    import stat

    hidden: list[str] = []
    try:
        dirs = list(site.getsitepackages())
    except Exception:  # noqa: BLE001 —— 非常规解释器布局，跳过即可
        return hidden
    for d in dirs:
        try:
            entries = list(Path(d).glob("*.pth"))
        except OSError:
            continue
        for p in entries:
            try:
                st = p.lstat()
            except OSError:
                continue
            if getattr(st, "st_flags", 0) & getattr(stat, "UF_HIDDEN", 0):
                hidden.append(str(p))
    return hidden


# ── doctor --e2e：记忆闭环端到端自检（ADR-0027 §3.2）────────────────────────

# 探针事实：内容要足够特异，语义检索能唯一命中；带随机串避免与历史探针混淆。
#
# 🔴 三条硬约束，全是 2026-08-05 首次实跑（G4）用失败换来的。探针要穿过一个 LLM 改写
#    和一层判重，能活下来的形状比看上去窄得多：
#
# ① **token 必须是句子的语义载荷，不能是修饰语**。"my probe passphrase is {token}"
#    里 token 是被陈述的值，蒸馏成中文仍保留；改成 "the probe named {token} has the
#    favourite colour chartreuse" 后 token 降级为定语，模型直接把它省掉了
#    （实测：该轮 new_facts=1，但库里搜不到任何含 token 的 fact）。
# ② **整句每轮必须不同**，否则被 novelty 判重吃掉。原模板的颜色子句是常量，
#    第二次跑就 `candidates=2 new_facts=1 existing=1`，库里积着一堆无 token 的
#    "favourite probe colour is chartreuse"。token 在句中 → 每轮天然新颖。
# ③ **只写一句、只说一件事**。蒸馏器按"原子事实"拆分，两个子句会被拆成两条 fact，
#    token 只跟着其中一条走——而 step 4 问的是另一条，出的题和判分标准就对不上了。
#    颜色那半句纯属多余，这一连串麻烦全是它引出来的。
_E2E_TEMPLATE = (
    "Please remember this for later: my BladeX end-to-end probe passphrase is {token}."
)

# step 4 的提问：fact 的**语义改写**，且**不含答案**——问句里带 token 就退化成关键词
# 查找，测不到检索这条腿。历史探针事实的干扰由"通过后自动清理"消掉（见 step 5），
# 不靠问句去规避。
_E2E_RECALL_QUERY = "what is my BladeX end-to-end probe passphrase?"


def _run_e2e_check(cfg, timeout_s: float = 180.0) -> int:
    """端到端记忆闭环自检。

    为什么需要这一枪（ADR-0027 §3.2）：记忆从产生到可感知要穿过
    `入库（Redis）→ consolidator 活着 → 蒸馏 LLM 调用成功 → fact 落 Memory Index → 语义命中`
    这条长链，**每一环都可能安静断掉，而 agent 界面上看到的都是"正常回答"**。
    分项体检各自都能绿，链路仍可能是断的。这里真的走一遍，把静默失效变成显式失败。
    """
    token = f"bx{secrets.token_hex(4)}"
    base = f"http://{'127.0.0.1' if cfg.host == '0.0.0.0' else cfg.host}:{cfg.port}"  # noqa: S104
    client_key = _first_client_key()
    print("\n=== Memory loop check (--e2e) ===")
    print(f"  probe token: {token}")

    # ① 发一轮真实对话过完整管线
    t0 = time.monotonic()
    status_code, body = _http_request(
        "POST", f"{base}/v1/chat/completions", key=client_key,
        body={"model": cfg.upstream_model or "default",
              "messages": [{"role": "user", "content": _E2E_TEMPLATE.format(token=token)}],
              "stream": False},
        timeout=120.0)
    if status_code != 200:
        msg = (body.get("error") or {}).get("message") or body
        print(f"  ✗ step 1 forwarding failed (HTTP {status_code}):{msg}")
        print("    Stuck at: proxy -> upstream. Check the upstream key / quota / model name (`bladex config check`).")
        return 1
    print(f"  ✓ step 1 forwarded ({time.monotonic() - t0:.1f}s)")

    # ② turn 是否入库（Pipeline → Memory Hub）。入不了库，后面全都不可能发生。
    stored = False
    for _ in range(15):
        idx = _http_json(f"{base}/admin/turns?limit=20", key=_admin_key())
        if idx:
            for item in idx.get("turns", []):
                from urllib.parse import quote
                d = _http_json(f"{base}/admin/turns/{quote(item['key'])}?include_messages=true",
                               key=_admin_key())
                if d and token in json.dumps(d.get("request_messages", ""), ensure_ascii=False):
                    stored = True
                    break
        if stored:
            break
        time.sleep(2.0)
    if not stored:
        print("  ✗ step 2 not stored -- this turn will produce no memory")
        print("    Stuck at: Pipeline/Memory Hub write. Is Redis reachable (`bladex status`)? "
              "Grep the log for enqueue_skipped_no_redis / pipeline_skipped_no_redis.")
        return 1
    print(f"  ✓ step 2 stored in Memory Hub ({time.monotonic() - t0:.1f}s)")

    # ③ consolidator 是否把它蒸成 fact（这一步最慢，也最容易静默停滞）
    deadline = time.monotonic() + timeout_s
    hit = None
    while time.monotonic() < deadline:
        facts = _http_json(f"{base}/admin/facts?q={token}&limit=5", key=_admin_key())
        for f in (facts or {}).get("facts", []):
            if token in (f.get("content") or ""):
                hit = f
                break
        if hit:
            break
        time.sleep(5.0)

    if hit is None:
        elapsed = time.monotonic() - t0
        print(f"  ✗ step 3 no fact distilled within {int(timeout_s)}s (waited {elapsed:.0f}s)")
        cons_pid = _consolidator_pid()
        if cons_pid is None:
            print("    Stuck at: the consolidator is not running -- `bladex start` launches it; "
                  "see logs/consolidator-*.log for why it failed.")
        else:
            print(f"    Stuck at: the consolidator (PID {cons_pid}) is running but produced nothing. "
                  "Common causes: the distillation model is unavailable or out of quota (grep the log for distill), "
                  "the catch-up lag is too large (see lag_turns in `bladex status`), "
                  "or this turn was judged to carry no memory value.")
        return 1
    print(f"  ✓ step 3 distilled into a Fact (id={hit.get('id', '?')[:12]}, "
          f"{time.monotonic() - t0:.1f}s)")

    # ④ 语义召回：用自然语言问句（不是拿 fact 原文精确匹配），验证的是"检索这条腿"
    from urllib.parse import quote as _quote
    recall = _http_json(
        f"{base}/admin/facts?q={_quote(_E2E_RECALL_QUERY)}&limit=10",
        key=_admin_key())
    recalled = any(token in (f.get("content") or "")
                   for f in (recall or {}).get("facts", []))
    if not recalled:
        print("  ✗ step 4 semantic recall missed it (the fact exists but the query did not return it)")
        print("    Stuck at: retrieval. Check, in this order: "
              "(a) is the embedding backend the same model that wrote these vectors? "
              "(changing models requires a full re-embed via scripts/reembed_index.py); "
              "(b) is the fact's content actually about what was asked -- "
              f"`bladex memory show {hit.get('id', '?')}`; "
              "(c) is the top-10 crowded out by low-value facts -- "
              f"`bladex memory search \"{_E2E_RECALL_QUERY}\" --k 10`.")
        return 1

    total = time.monotonic() - t0
    print(f"\n  ✓ Memory loop OK: the probe fact was distilled and semantically recalled ({total:.1f}s)")

    # ⑤ 清理：探针是测试数据，不是用户的记忆。留着会一轮轮堆积、互相竞争排名，
    #    还会挤占真实记忆的召回位（C13 冷安装那一轮会有人反复跑 doctor）。
    #    只在**通过后**清理——失败时留着让人能 `bladex memory show` 现场取证。
    fact_id = hit.get("id", "")
    status_code, _ = _http_request("DELETE", f"{base}/admin/facts/{fact_id}",
                                   key=_admin_key(), timeout=10.0)
    if status_code in (200, 202):
        print(f"    Cleaned up the probe fact ({fact_id[:12]}, tombstoned -- never revived on rebuild)")
    else:
        print(f"    Note: could not clean up the probe fact ({fact_id}) -- "
              f"HTTP {status_code}; remove it with `bladex memory forget {fact_id}`")
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


# ═══════════════════════════════════════════════════════════════════════════
# Beta T11：存储运维 + 记忆管理（数据面一律打 admin HTTP API，不直捅存储）
# ═══════════════════════════════════════════════════════════════════════════

storage_app = typer.Typer(help="Storage operations", no_args_is_help=True)
memory_app = typer.Typer(help="Memory management (via the proxy admin API)", no_args_is_help=True)
matter_app = typer.Typer(help="Matter management (via the proxy admin API)", no_args_is_help=True)
inspect_app = typer.Typer(help="Storage inspection (hub/index)", no_args_is_help=True)
queue_app = typer.Typer(help="Pipeline queue (turns waiting to reach the Memory Hub)",
                        no_args_is_help=True)
sticky_app = typer.Typer(help="Session model stickiness (which model each session is pinned to)",
                         no_args_is_help=True)
# 注意与 `bladex inspect hub` 的区分（见 ledger_app 命令处注释）。
ledger_app = typer.Typer(help="Task ledgers (five-section working state; read-only except "
                              "`doctor --bind-legacy-matters --yes`)",
                         no_args_is_help=True)
app.add_typer(storage_app, name="storage")
app.add_typer(memory_app, name="memory")
app.add_typer(matter_app, name="matter")
app.add_typer(queue_app, name="queue")
app.add_typer(sticky_app, name="sticky")
app.add_typer(inspect_app, name="inspect")
app.add_typer(ledger_app, name="ledger")


def _admin_base() -> str:
    from bladex_proxy.config import ProxyConfig
    cfg = ProxyConfig()
    return f"http://{cfg.host}:{cfg.port}"


def _http_request(method: str, url: str, key: str = "",
                  body: dict | None = None, timeout: float = 15.0) -> tuple[int, dict]:
    """admin API 调用（urllib，仅本机管理面）。返回 (status, json)。"""
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)  # noqa: S310
    req.add_header("Content-Type", "application/json")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode("utf-8"))
        except ValueError:
            return e.code, {}
    except (urllib.error.URLError, OSError) as e:
        print(f"Error: proxy unreachable ({url}) -- run `bladex start` first. ({e})")
        raise typer.Exit(1) from e


def _admin_call(method: str, path: str, body: dict | None = None,
                params: dict | None = None, timeout: float = 15.0) -> tuple[int, dict]:
    from urllib.parse import urlencode
    url = _admin_base() + path
    if params:
        url += "?" + urlencode({k: v for k, v in params.items() if v not in ("", None)})
    return _http_request(method, url, key=_admin_key(), body=body, timeout=timeout)


def _print_op_result(status_code: int, data: dict) -> int:
    if status_code == 200:
        print(f"✓ {data.get('status', 'ok')}: "
              + json.dumps({k: v for k, v in data.items() if k != 'status'},
                           ensure_ascii=False))
        return 0
    if status_code == 202:
        print("⏳ deferred: the admin event is in Memory Hub, but the consolidator holds the Memory Index write lock -- "
              "it will take effect on the next rebuild/replay")
        return 0
    print(f"✗ HTTP {status_code}: {json.dumps(data, ensure_ascii=False)}")
    return 1


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


# ── storage ────────────────────────────────────────────────────────


@storage_app.command("status")
def storage_status() -> int:
    """Storage status across the three tiers (via /admin/status, or a local read-only probe)."""
    try:
        code, data = _admin_call("GET", "/admin/status")
        if code == 200:
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0
        print(f"✗ /admin/status HTTP {code}")
        return 1
    except typer.Exit:
        pass  # proxy 未运行 → 本地只读探测
    from bladex_proxy.config import ProxyConfig
    cfg = ProxyConfig()
    print("(proxy not running -- local read-only probe)")
    try:
        from bladex_proxy.storage.memory_hub import MemoryHub
        hub = MemoryHub(cfg.rocksdb_path, secondary=True)
        hub.open()
        print(f"  Memory Hub: {hub.count()} turns ({cfg.rocksdb_path})")
        hub.close()
    except Exception as e:  # noqa: BLE001
        print(f"  Memory Hub: unreadable ({e})")
    try:
        from bladex_proxy.storage.memory_index import MemoryIndex
        index = MemoryIndex(cfg.index_path, embedder=None, read_only=True)
        index.open()
        print(f"  Memory Index: {index.fact_count()} facts / {len(index.all_matters())} matters"
              f"（{cfg.index_path}）")
        index.close()
    except Exception as e:  # noqa: BLE001
        print(f"  Memory Index: unreadable ({e})")
    return 0


@storage_app.command("rebuild")
def storage_rebuild(
    concurrency: int = typer.Option(-1, "--concurrency"),
    direct: bool = typer.Option(False, "--direct"),
    since: str = typer.Option("", "--since"),
    until: str = typer.Option("", "--until"),
    agents: str = typer.Option("", "--agents"),
    exclude_agents: str = typer.Option("", "--exclude-agents"),
) -> int:
    """Full Memory Index rebuild (same as `bladex sync run --full`; zero-downtime via delegate)."""
    return sync_run(full=True, concurrency=concurrency, max_turns=0, since=since,
                    until=until, agents=agents, exclude_agents=exclude_agents,
                    direct=direct, pickup_timeout=10.0, log="")


def _run_repo_script(rel: str, args: list[str], runner: str = "bash") -> int:
    script = Path(rel)
    if not script.is_file():
        print(f"Error: not found: {rel} (this command must run from the repository root)")
        return 1
    cmd = ([runner, str(script)] if runner == "bash"
           else [sys.executable, str(script)]) + args
    return subprocess.run(cmd, check=False).returncode  # noqa: S603


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


@inspect_app.command("hub", context_settings={"allow_extra_args": True,
                                             "ignore_unknown_options": True})
def inspect_hub(ctx: typer.Context) -> int:
    """Inspect Memory Hub (passes every argument through to scripts/inspect_hub.py)."""
    return _run_repo_script("scripts/inspect_hub.py", list(ctx.args), runner="python")


@inspect_app.command("index", context_settings={"allow_extra_args": True,
                                             "ignore_unknown_options": True})
def inspect_index(ctx: typer.Context) -> int:
    """Inspect Memory Index (passes arguments through to scripts/inspect_index.py)."""
    return _run_repo_script("scripts/inspect_index.py", list(ctx.args), runner="python")


# ── connector（T13 在线协同导出，B3.2）────────────────────────────

connector_app = typer.Typer(help="Continuous export to the targets configured in export.toml",
                            no_args_is_help=True)
app.add_typer(connector_app, name="connector")


def _build_export_worker():
    from bladex_proxy.config import ProxyConfig
    from bladex_proxy.export_sync import build_worker_from_config
    from bladex_proxy.storage.memory_index import MemoryIndex
    from bladex_proxy.storage.memory_hub import MemoryHub

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


@connector_app.command("run")
def connector_run(
    force: bool = typer.Option(False, "--force", help="Ignore the backoff window and try now"),
) -> int:
    """Run one incremental export pass (a running consolidator also does this periodically)."""
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
    from bladex_proxy.config import ProxyConfig
    from bladex_proxy.snapshot import export_snapshot
    from bladex_proxy.storage.memory_index import MemoryIndex
    from bladex_proxy.storage.memory_hub import MemoryHub

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
    if _read_pidfile(_PROXY_PIDFILE) is not None or _consolidator_pid() is not None:
        print("Error: the proxy/consolidator are still running -- run `bladex stop` first (import needs the Memory Index/Memory Hub write locks)")
        return 1
    path = deployment.resolve_user_path(path)  # 相对路径钉回用户敲命令的目录
    if not Path(path).is_file():
        print(f"Error: not found: {path}")
        return 1
    from bladex_proxy.config import ProxyConfig
    from bladex_proxy.snapshot import import_snapshot
    from bladex_proxy.storage.memory_index import MemoryIndex
    from bladex_proxy.storage.memory_hub import MemoryHub

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


# ── memory ─────────────────────────────────────────────────────────


@memory_app.command("search")
def memory_search(
    query: str = typer.Argument(..., help="Search query (semantic + filters)"),
    k: int = typer.Option(10, "--k", help="How many results"),
    user: str = typer.Option("", "--user", help="Filter by user_id"),
) -> int:
    """Semantic search over facts (GET /admin/facts?q=)."""
    code, data = _admin_call("GET", "/admin/facts",
                             params={"q": query, "limit": k, "user_id": user})
    if code != 200:
        return _print_op_result(code, data)
    facts = data.get("facts", [])
    if not facts:
        print("(no matches)")
        return 0
    for f in facts:
        content = (f.get("content", "") or "").replace("\n", " ")
        if len(content) > 80:
            content = content[:77] + "..."
        print(f"  {f.get('id', '?'):20s} [{f.get('kind', '?'):10s}] {content}")
    print(f"Total {data.get('total', len(facts))}. Use `bladex memory show <id>` for details.")
    return 0


@memory_app.command("show")
def memory_show(fact_id: str = typer.Argument(...)) -> int:
    """Show one fact (GET /admin/facts/{id})."""
    code, data = _admin_call("GET", f"/admin/facts/{fact_id}")
    if code != 200:
        return _print_op_result(code, data)
    print(json.dumps(data, ensure_ascii=False, indent=2))
    return 0


@memory_app.command("forget")
def memory_forget(
    fact_id: str = typer.Argument(...),
    yes: bool = typer.Option(False, "--yes", help="Skip confirmation"),
) -> int:
    """Delete a fact (Hub tombstone + Memory Index cascade; never revived on rebuild)."""
    if not yes and not typer.confirm(f"Delete fact {fact_id}? (tombstone semantics; never revived on rebuild)"):
        print("Cancelled.")
        return 130
    code, data = _admin_call("DELETE", f"/admin/facts/{fact_id}")
    return _print_op_result(code, data)


# ── matter ─────────────────────────────────────────────────────────


@matter_app.command("list")
def matter_list(
    status: str = typer.Option("", "--status", help="Filter by status (active/closed/...)"),
    limit: int = typer.Option(50, "--limit"),
) -> int:
    code, data = _admin_call("GET", "/admin/matters",
                             params={"status": status, "limit": limit})
    if code != 200:
        return _print_op_result(code, data)
    for m in data.get("matters", []):
        print(f"  {m.get('matter_id', '?'):24s} [{m.get('status', '?'):8s}]"
              f" [{m.get('origin', '?'):6s}] {m.get('title', '')}")
    print(f"Total {data.get('total', 0)}.")
    return 0


@matter_app.command("show")
def matter_show(matter_id: str = typer.Argument(...)) -> int:
    code, data = _admin_call("GET", f"/admin/matters/{matter_id}")
    if code != 200:
        return _print_op_result(code, data)
    m = data.get("matter", {})
    print(f"{m.get('matter_id')}  [{m.get('status')}]  {m.get('title')}")
    if m.get("summary"):
        print(f"  summary: {m['summary']}")
    print(f"  edges: {data.get('edge_count', 0)} (manual {data.get('manual_edge_count', 0)})"
          f"  sessions: {len(data.get('session_keys', []))}")
    for f in data.get("facts", [])[:20]:
        content = (f.get("content", "") or "").replace("\n", " ")[:70]
        print(f"    - {f.get('id', '?')}: {content}")
    return 0


@matter_app.command("create")
def matter_create(title: str = typer.Argument(...),
                  summary: str = typer.Option("", "--summary")) -> int:
    code, data = _admin_call("POST", "/admin/matters",
                             body={"title": title, "summary": summary})
    return _print_op_result(code, data)


@matter_app.command("assign")
def matter_assign(
    matter_id: str = typer.Argument(...),
    target_key: str = typer.Argument(..., help="fact_id or session prefix"),
    target_type: str = typer.Option("fact", "--type", help="fact | session"),
) -> int:
    """Manually assign to a Matter (manual overrides auto)."""
    code, data = _admin_call("POST", f"/admin/matters/{matter_id}/assign",
                             body={"target_type": target_type, "target_key": target_key})
    return _print_op_result(code, data)


@matter_app.command("merge")
def matter_merge(
    target_id: str = typer.Argument(..., help="Merge into (kept)"),
    source_id: str = typer.Argument(..., help="Merged from (absorbed, then closed)"),
) -> int:
    code, data = _admin_call("POST", f"/admin/matters/{target_id}/merge",
                             body={"source_matter_id": source_id})
    return _print_op_result(code, data)


@matter_app.command("merge-candidates")
def matter_merge_candidates(limit: int = typer.Option(50, "--limit")) -> int:
    """List likely-duplicate Matter pairs (topic-key overlap). Detection only --
    merging always goes through `matter merge` (manual sovereignty, journaled)."""
    code, data = _admin_call("GET", f"/admin/matters/merge_candidates?limit={limit}")
    if code != 200 or not isinstance(data, dict):
        return _print_op_result(code, data)
    pairs = data.get("candidates", [])
    if not pairs:
        typer.echo("No merge candidates (no high-overlap matter pairs).")
        return 0
    for p in pairs:
        typer.echo(f"overlap={p['overlap']} jaccard={p['jaccard']}  "
                   f"{p['source_matter_id']} ({p['source_title'][:30]!r}, {p['source_edges']} edges)"
                   f" -> {p['target_matter_id']} ({p['target_title'][:30]!r}, {p['target_edges']} edges)")
        typer.echo(f"    shared: {', '.join(p['shared_keys'])}")
        typer.echo(f"    apply:  bladex matter merge {p['target_matter_id']} {p['source_matter_id']}")
    return 0


@matter_app.command("detach")
def matter_detach(
    matter_id: str = typer.Argument(...),
    edge_id: str = typer.Option("", "--edge-id"),
    target_key: str = typer.Option("", "--target-key"),
    target_type: str = typer.Option("fact", "--type"),
) -> int:
    body: dict = {"edge_id": edge_id}
    if target_key:
        body = {"target_key": target_key, "target_type": target_type}
    code, data = _admin_call("POST", f"/admin/matters/{matter_id}/detach", body=body)
    return _print_op_result(code, data)


@matter_app.command("split")
def matter_split(
    matter_id: str = typer.Argument(...),
    new_title: str = typer.Option(..., "--new-title"),
    edge_ids: list[str] = typer.Option([], "--edge-id", help="Edge to split out (repeatable)"),  # noqa: B006, B008
) -> int:
    code, data = _admin_call("POST", f"/admin/matters/{matter_id}/split",
                             body={"new_title": new_title, "edge_ids": list(edge_ids)})
    return _print_op_result(code, data)


@matter_app.command("close")
def matter_close(matter_id: str = typer.Argument(...)) -> int:
    code, data = _admin_call("POST", f"/admin/matters/{matter_id}/close", body={})
    return _print_op_result(code, data)


@matter_app.command("rename")
def matter_rename(matter_id: str = typer.Argument(...),
                  title: str = typer.Argument(...)) -> int:
    code, data = _admin_call("POST", f"/admin/matters/{matter_id}/rename",
                             body={"title": title})
    return _print_op_result(code, data)


# ── ledger（五段账本，只读）─────────────────────────────────────
# 命名口径（V-A4 符号层收口后，2026-08-28）："ledger" 一词只指**任务账本**--
# agent 侧模型维护的五段任务工作状态 Goal/Core/Verified/Open/Next（ADR-0032 §4），
# 作者在 agent 侧，CLI 只读。Memory Hub（RocksDB，data/bladex_hub 磁盘路径为
# 历史遗留不迁移）在符号/文案层一律叫 Hub：`bladex inspect hub`。
# 台账（distill/judgment journal）叫 journal。三者不再共用 "ledger" 一词。


def _ledger_entry_counts(row: dict) -> str:
    """五段条目数单元格（`g1/c2/v0/o1/n3`）。

    `entry_counts` 来自 `Ledger.section_order`（**默认值不是闭集**，自定义段会
    出现），goal 恒不在其中（端点明确排除）。渲染上仍按默认五段给主位，
    未知段追加在尾部，不丢信息也不假设闭集。
    """
    counts = row.get("entry_counts") or {}
    goal_flag = 1 if row.get("goal") else 0
    main = "/".join(
        f"{s[:1]}{counts.get(s, 0)}"
        for s in ("core", "verified", "open", "next"))
    extras = sorted(k for k in counts if k not in
                    ("core", "verified", "open", "next"))
    cell = f"g{goal_flag}/{main}"
    for k in extras:
        cell += f"/{k}={counts.get(k, 0)}"
    return cell


def _ledger_time_cell(ts: str | None) -> str:
    """UPDATED 列：端点给 UTC（真相层），这里转成 CLI 运行环境的本地时区（MQ-L46）。"""
    from bladex_core.ledger import local_iso
    return local_iso(ts or "") or "-"


def _ledger_goal_cell(row: dict) -> str:
    """Goal 单元格：一行截断 + 来源角标（`[u]`=user 原话 / `[m]`=model 提炼 /
    `[r]`=model_revised 经用户改过；空 = 未取到）。"""
    text = (row.get("goal", "") or "").replace("\n", " ").strip()
    if not text:
        return "-"
    src = row.get("goal_source", "") or ""
    tag = {"user": "u", "model": "m", "model_revised": "r"}.get(src, "?")
    return f"[{tag}] {text}"[:30]


def _ledger_active_cell(active_in: list) -> str:
    """激活 scope 单元格：`agent@project` 逗号连接；空 = 未被任何会话激活。"""
    cells = []
    for s in active_in or []:
        agent = (s or {}).get("agent", "?")
        project = (s or {}).get("project", "") or "-"
        cells.append(f"{agent}@{project}")
    return ", ".join(sorted(cells)) if cells else "-"


@ledger_app.command("list")
def ledger_list(
    json_output: bool = typer.Option(False, "--json", help="Print raw JSON from the admin API"),
) -> int:
    """List task ledgers (active ones first, order from the admin API)."""
    code, data = _admin_call("GET", "/admin/ledgers")
    if code != 200:
        return _print_op_result(code, data)
    if json_output:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0
    if not data.get("enabled", True):
        # 端点返回 {"enabled": false, "ledgers": [], "total": 0}：agency 未启用。
        print("Ledger feature is not enabled on this proxy "
              "(check the agency/ledger configuration and restart).")
        return 0
    rows = data.get("ledgers") or []
    if not rows:
        print("No task ledgers yet -- they are created by agents as they work "
              "(one ledger per task; this list will fill in once one does).")
        return 0
    total = data.get("total", len(rows))
    print(f"{total} task ledger(s):")
    # 表头与数据列一一对齐（缺陷 2 的教训：改列必须同步改两边，测试按列断言钉死）。
    header = (f"  {'ID':<16} {'TITLE':<24} {'GOAL':<30} {'ENTRIES':<24} "
              f"{'ACTIVE-IN':<24} {'MATTER':<13} {'UPDATED':<20} STATUS")
    print(header)
    for r in rows:
        print(f"  {r.get('ledger_id', '')[:16]:<16} "
              f"{(r.get('title') or r.get('ledger_id', ''))[:24]:<24} "
              f"{_ledger_goal_cell(r)[:30]:<30} "
              f"{_ledger_entry_counts(r)[:24]:<24} "
              f"{_ledger_active_cell(r.get('active_in'))[:24]:<24} "
              f"{(r.get('matter_id') or '-')[:13]:<13} "
              f"{_ledger_time_cell(r.get('updated_at'))[:19]:<20} "
              f"{r.get('status', '-')}")
    return 0


#: ── `ledger doctor` 的四个阈值 ──────────────────────────────────────────
#: 🔴 每个都写取值理由。本仓纪律：标定常数不许是随手取的圆整数。
#:
#: 标题归一化后相等 = 疑似重复。**不做模糊匹配**：doctor 是体检命令，
#: 误报一次就会让人不再看它；宁可漏报。
#: Goal 前缀比较长度：取 60 —— 2026-08-27 那批账本里，Codex 内部调用生成的
#: Goal 前 60 字符已足够区分（`Generate 0 to 3 hyperpersonalized suggestions…`
#: 与 `You are a helpful assistant. You will be presented…` 在 40 字符内就分开）；
#: 取太长会被后半段的差异救回去，等于关掉这条检查。
_DOCTOR_GOAL_PREFIX = 60
#: 长期无更新的天数。7 天 = ADR-0012 定义的 Matter「周级可完结」粒度——
#: 一件事跨过一周还没动静，值得看一眼。不是"坏"，是"该复核"。
_DOCTOR_STALE_DAYS = 7


def _doctor_norm_title(t: str) -> str:
    """标题归一化：小写 + 压空白。仅此而已。

    不去标点、不去词缀——那些会把「Codex ledger health 开发」与
    「Codex ledger health 复核」判成同一件事。宁可漏报。
    """
    return " ".join((t or "").lower().split())


def _doctor_age_days(iso: str) -> float | None:
    """ISO 时间串 → 距今天数。解析不了返回 None（不猜、不当成 0）。"""
    import datetime as _dt
    s = (iso or "").strip()
    if not s:
        return None
    try:
        ts = _dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=_dt.UTC)
    return (_dt.datetime.now(_dt.UTC) - ts).total_seconds() / 86400.0


def _doctor_scope_key(row: dict) -> str:
    """账本所属 scope。`active_in` 为空 = 当前没有 agent 激活它。

    重复检测**只在同一 scope 内比**：不同 agent 各有一本同名账本是
    合理形态（跨 agent 交接正是北极星要的），不该报成重复。
    """
    scopes = row.get("active_in") or []
    if not scopes:
        return ""
    s = scopes[0]
    return f"{s.get('agent','')}\x1f{s.get('project','')}"


def _doctor_findings(rows: list[dict], details: dict[str, dict]) -> list[dict]:
    """四类检查，全部确定性——不调 LLM、不用 embedding（体检命令要能随处跑）。"""
    from collections import defaultdict

    out: list[dict] = []

    # ① 疑似重复：同 scope 下标题归一化后相等，或 Goal 前缀相同
    by_title: dict[tuple[str, str], list[str]] = defaultdict(list)
    by_goal: dict[tuple[str, str], list[str]] = defaultdict(list)
    for r in rows:
        scope = _doctor_scope_key(r)
        t = _doctor_norm_title(r.get("title", ""))
        if t:
            by_title[(scope, t)].append(r["ledger_id"])
        g = (r.get("goal") or "").strip()[:_DOCTOR_GOAL_PREFIX]
        if g:
            by_goal[(scope, g)].append(r["ledger_id"])
    for (_scope, key), ids in sorted(by_title.items()):
        if len(ids) > 1:
            out.append({"kind": "duplicate_title", "ledger_ids": sorted(ids),
                        "detail": f"same title after normalisation: {key!r}",
                        "action": f"bladex ledger show {sorted(ids)[0]}"})
    seen = {tuple(sorted(f["ledger_ids"])) for f in out}
    for (_scope, key), ids in sorted(by_goal.items()):
        if len(ids) > 1 and tuple(sorted(ids)) not in seen:
            out.append({"kind": "duplicate_goal", "ledger_ids": sorted(ids),
                        "detail": f"same first {_DOCTOR_GOAL_PREFIX} chars of goal",
                        "action": f"bladex ledger show {sorted(ids)[0]}"})

    # ② 疑似机器文本：Goal 看着像客户端模板
    # 🔴 判据复用 `agency._machine_text_mark`（任务卡明确要求，两份判据早晚分叉）。
    from bladex_proxy.agency import _machine_text_mark
    for r in rows:
        d = details.get(r["ledger_id"]) or {}
        led = d.get("ledger") or {}
        # `goal_verbatim` 是**用户原话**，才是该检查的对象；`goal` 已被模型提炼过。
        # list 端点不返回它 —— 见 doctor 的 docstring「端点数据不够用」那段。
        probe = led.get("goal_verbatim") or r.get("goal") or ""
        mark = _machine_text_mark(probe)
        if mark:
            out.append({"kind": "machine_text_goal", "ledger_ids": [r["ledger_id"]],
                        "detail": f"goal looks like a client template ({mark})",
                        "action": f"bladex ledger show {r['ledger_id']}"})

    # ③ 长期无更新：段条目全 0 且 goal 从未修订过，且创建已久
    for r in rows:
        counts = r.get("entry_counts") or {}
        if any(int(v or 0) for v in counts.values()):
            continue
        d = details.get(r["ledger_id"]) or {}
        if (d.get("ledger") or {}).get("goal_revisions"):
            continue
        age = _doctor_age_days(r.get("created_at", ""))
        if age is not None and age >= _DOCTOR_STALE_DAYS:
            out.append({"kind": "never_used", "ledger_ids": [r["ledger_id"]],
                        "detail": f"no entries and no goal revision, {age:.0f} days old",
                        "action": f"bladex ledger show {r['ledger_id']}"})

    # ④ 没有 Matter 锚 —— **只报锚点机制上线之后建的**
    #
    # 🔴 初版报所有无锚账本：20 本里报了 12 本，信噪比 25%（2026-08-28 live 实测）。
    # 锚点是 08-26 才启用的，之前建的账本天然没锚，**它们不是问题**。
    # 一条必然大面积误报的检查，会让人不再看这个命令 —— 而这正是
    # `_doctor_norm_title` 注释里自己写过的话，却在这一项上犯了。
    #
    # 判据改为**数据自证**、不写死日期：取所有已锚账本里最早的 `created_at`
    # 当作机制上线时刻，只报晚于它、却仍无锚的。
    # 一本已锚账本都没有 ⇒ 机制没跑过 ⇒ 整条检查跳过（无从判断）。
    from bladex_core.ledger import iso_ms   # 比数值不比 ISO 串（MQ-L46）
    anchored_births = sorted(
        ((r.get("created_at") or "") for r in rows
         if (r.get("matter_id") or "").strip() and (r.get("created_at") or "")),
        key=iso_ms)
    if anchored_births:
        epoch = anchored_births[0]
        for r in rows:
            if (r.get("matter_id") or "").strip():
                continue
            born = r.get("created_at") or ""
            if born and iso_ms(born) > iso_ms(epoch):
                out.append({
                    "kind": "no_matter_anchor", "ledger_ids": [r["ledger_id"]],
                    "detail": f"created {born} (after anchoring went live at "
                              f"{epoch}) but still has no Matter",
                    "action": f"bladex ledger show {r['ledger_id']}"})
    return out


@ledger_app.command("doctor")
def ledger_doctor(
    json_output: bool = typer.Option(False, "--json", help="Print findings as JSON"),
    details: bool = typer.Option(
        True, "--details/--no-details",
        help="Fetch each ledger's detail (needed for goal_verbatim); one extra call per ledger"),
    bind_legacy_matters: bool = typer.Option(
        False, "--bind-legacy-matters",
        help="Backfill the ledger->Matter pointer on ledgers created before it was "
             "written at creation (lists them; add --yes to write)"),
    yes: bool = typer.Option(False, "--yes", help="With --bind-legacy-matters: actually write"),
) -> int:
    """Report suspicious ledgers. Read-only -- never edits or merges.

    唯一的写动作是显式的 `--bind-legacy-matters --yes`（2026-09-04，
    `task-ledger-switch-candidates-20260904.md` §2）：只给 `matter_id` 空的旧账本补上
    确定性派生的锚 id（与 Index 已有的同一个），不合并、不改绑、不碰 Goal。

    2026-08-27 事故驱动：用户只发了**一句话**，BladeX 建了**四本账本**
    （两本来自 Codex 客户端的内部功能调用），靠肉眼翻列表才发现，排查三小时。
    根因已修（MQ-L21），但**反方向「多本账本本该是一本」零检测**（V-L5 §1c）。
    本命令补的是那个检测 —— 不自动修，是让下一次三分钟内被看见。

    🔴 **只报告不自动修**：账本的作者是 agent 侧的模型，Goal 只有用户能改；
    自动合并会踩「误合并=0」第一红线。

    ⚠️ **端点数据不够用的地方（任务卡口径要求 1：说出来，别绕过去读库）**：
    `/admin/ledgers` 列表**不返回 `goal_verbatim`**（用户原话）与 `goal_revisions`，
    而检查 ② 和 ③ 需要它们。故默认对每本账本再取一次
    `/admin/ledgers/{id}`——账本池是几十本量级，N+1 可接受；
    嫌慢用 `--no-details`，那时 ② 退化为查已提炼的 `goal`（判据变弱，会漏报）。
    """
    if bind_legacy_matters:
        return _ledger_bind_legacy_matters(write=yes, json_output=json_output)
    code, data = _admin_call("GET", "/admin/ledgers")
    if code != 200:
        return _print_op_result(code, data)
    if not data.get("enabled", True):
        print("Ledger feature is not enabled on this proxy "
              "(check the agency/ledger configuration and restart).")
        return 0
    rows = data.get("ledgers") or []
    if not rows:
        print("No task ledgers yet -- nothing to check.")
        return 0

    detail_map: dict[str, dict] = {}
    if details:
        from urllib.parse import quote
        for r in rows:
            c, d = _admin_call("GET", f"/admin/ledgers/{quote(r['ledger_id'], safe='')}")
            if c == 200:
                detail_map[r["ledger_id"]] = d

    findings = _doctor_findings(rows, detail_map)

    if json_output:
        print(json.dumps({"checked": len(rows), "findings": findings},
                         ensure_ascii=False, indent=2))
        return 1 if findings else 0

    if not findings:
        print(f"Checked {len(rows)} ledger(s): no problems found.")
        return 0
    print(f"Checked {len(rows)} ledger(s), {len(findings)} finding(s):")
    print()
    for f in findings:
        ids = ", ".join(f["ledger_ids"])
        print(f"  [{f['kind']}] {ids}")
        print(f"      {f['detail']}")
        print(f"      -> {f['action']}")
    print()
    print("Nothing was changed. Review each one before acting.")
    return 1


def _ledger_bind_legacy_matters(*, write: bool, json_output: bool) -> int:
    """`ledger doctor --bind-legacy-matters [--yes]`：走 `/admin/ledgers/bind-legacy-matters`。"""
    code, data = _admin_call("POST", "/admin/ledgers/bind-legacy-matters",
                             body={"dry_run": not write})
    if code != 200:
        return _print_op_result(code, data)
    if json_output:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0
    rows = data.get("ledgers") or []
    print(f"Checked {data.get('checked', 0)} ledger(s): {data.get('unbound', 0)} without a Matter pointer.")
    for r in rows:
        print(f"  {r['ledger_id']} -> {r['matter_id']}  {r.get('title', '')}")
    if not rows:
        return 0
    if write:
        print(f"Bound {len(rows)} ledger(s) (LEDGER_UPDATE events written; replay-equivalent).")
    else:
        print("Nothing was changed (dry run). Re-run with --yes to write.")
    return 0


@ledger_app.command("show")
def ledger_show(
    ledger_id: str = typer.Argument(..., help="Ledger ID (ldg-...)"),
    json_output: bool = typer.Option(False, "--json", help="Print raw JSON from the admin API"),
) -> int:
    """Show one task ledger (goal, sections, children, Matter anchor)."""
    code, data = _admin_call("GET", f"/admin/ledgers/{ledger_id}")
    if code != 200:
        # 404 时端点给 {"status": "not_found", "detail": "unknown ledger ..."}
        # -- 照它的 detail 打印，不吞成 "HTTP 404"。
        detail = data.get("detail", "")
        if code == 404 and detail:
            print(f"✗ {detail}")
            return 1
        return _print_op_result(code, data)
    if json_output:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0
    print(data.get("markdown", ""))
    children = data.get("children") or []
    if children:
        print()
        print("Sub-ledgers:")
        for c in children:
            print(f"  {c.get('ledger_id', '')}  {c.get('title', '')}")
    return 0


# ── 入口（兼容层：scripts/bladex.py 与 test_sync_cli 依赖 main(argv)->int）──


def main(argv: list[str] | None = None) -> int:
    deployment.remember_invocation_cwd()
    _load_env_file()
    _chdir_to_deployment_root()
    import click
    try:
        rv = app(args=argv, standalone_mode=False, prog_name="bladex")
    except (typer.Exit, click.exceptions.Exit) as e:
        return int(getattr(e, "exit_code", 0) or 0)
    except SystemExit as e:
        return int(e.code or 0)
    except click.ClickException as e:
        e.show()
        return e.exit_code or 2
    except click.exceptions.Abort:
        print("Cancelled.")
        return 130
    return int(rv) if isinstance(rv, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
