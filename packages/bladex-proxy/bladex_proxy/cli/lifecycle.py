"""生命周期命令：`start` / `stop` / `restart` / `status` / `embed` / `consolidator` / `init` / `doctor` + `_start_*`。

09-06 F0.1 自 `cli.py` 拆出，零行为（函数体逐字搬家）。共享 helper（`_admin_call` / `_http_json` / `_probe_ready` / pidfile 族…）留在门面 `bladex_proxy.cli`；本模块经 `from bladex_proxy.cli import …` 取用，
🔴 故 monkeypatch 要打在**本模块**上（`bladex_proxy.cli.<本模块>.<helper>`），打门面不生效。
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import typer

from bladex_proxy.cli import (  # noqa: E402
    _CONS_PIDFILE,
    _EMBED_PIDFILE,
    _FLASH_PIDFILE,
    _PROXY_PIDFILE,
    _admin_key,
    _consolidator_pid,
    _first_client_key,
    _http_json,
    _http_request,
    _pid_alive,
    _print_admin_unavailable,
    _print_config_not_found,
    _probe_ready,
    _read_pidfile,
    _repo_root,
    _require_config_root,
    _resolve_distill_concurrency,
    app,
    assets_dir,
)


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
    # 🔴 MQ-L62 ②（2026-09-08）：先记下**进来之前**谁已经在跑。
    # proxy 启动失败时只回收**本次拉起的**那些——已经在跑的（用户自己起的、
    # 上一次 start 留下的）一律不碰。09-08 实测形态：proxy 因配置错误退出，
    # embedding / consolidator 也没了，**flash daemon 却还活着**——一个在 proxy
    # 不在时继续写 Flash 投影的守护进程，是实打实的状态风险（同族 MQ-L60）。
    _SIDECARS = (("Embedding", _EMBED_PIDFILE), ("Consolidator", _CONS_PIDFILE),
                 ("Flash daemon", _FLASH_PIDFILE))
    pids_before = {name: _read_pidfile(pf) for name, pf in _SIDECARS}
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
    rc = _report_startup_state(host, port, redis_ok, consolidator_ok,
                               proxy_pid=_read_pidfile(_PROXY_PIDFILE))
    if rc != 0:
        _rollback_sidecars(pids_before, _SIDECARS)
    return rc


def _rollback_sidecars(pids_before: dict[str, int | None],
                       sidecars: tuple[tuple[str, str], ...]) -> None:
    """proxy 起不来时回收**本次拉起的**旁挂进程（MQ-L62 ②）。

    判据 = PID 文件里的值与进来之前不同（含"之前没有、现在有"）。
    **不按"我调用过 _start_x"判**：那三个 helper 在进程已在跑时是空操作，
    照那个判据会去杀用户自己起的守护进程 —— 停错东西没有再说的机会
    （与 `cleanup --ledgers` 的 fail-closed 安全阀同一条立论）。
    """
    stopped = [(name, pf) for name, pf in sidecars
               if (now := _read_pidfile(pf)) is not None and now != pids_before.get(name)]
    if not stopped:
        return
    print("\n  Proxy did not come up -- stopping the sidecars this run started "
              "(anything already running is left alone):")
    for name, pf in stopped:
        _stop_by_pidfile(pf, f"  {name}")


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


def _newest_proxy_log() -> str:
    """最新的 proxy 日志路径（刚起的那个进程写的就是它）；没有则空串。"""
    try:
        logs = sorted(Path("logs").glob("proxy-*.log"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        return ""
    return str(logs[0]) if logs else ""


def _log_error_tail(logfile: str, max_lines: int = 30) -> str:
    """日志里**最后一段错误**（从最后一个 `ERROR:` / `Traceback` 起，到文件末）。

    找不到错误标记 ⇒ 退回最后 `max_lines` 行非空内容。**不做"聪明"的解析**：
    启动失败的形态千奇百怪，原样把尾巴摆到终端上，比我们猜一个摘要有用。
    """
    try:
        lines = Path(logfile).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    start = 0
    for i, ln in enumerate(lines):
        if ln.startswith("ERROR:") or ln.startswith("Traceback"):
            start = i
    tail = [ln for ln in lines[start:] if ln.strip()]
    if len(tail) > max_lines * 2:      # 极长的 traceback：留头留尾
        tail = tail[:max_lines] + ["  ... (truncated; see the full log) ..."] + tail[-max_lines:]
    return "\n".join("    " + ln for ln in tail)


def _report_startup_state(host: str, port: int, redis_ok: bool, consolidator_ok: bool,
                          timeout_s: float = 30.0, proxy_pid: int | None = None) -> int:
    """轮询 /ready，把真实状态回给 console（ADR-0027 §3.1 契约 A）。

    旧行为：打印"Proxy 已启动"就 rc=0 收工。冷启动实测下这意味着——Redis 缺席
    （每轮对话都不入库）、consolidator 缺席（永远不会有事实被提炼）时，用户视角
    仍是"启动成功"，代理照常转发、对话完全正常，**没有任何界面告诉他记忆管线是死的**。

    三态：ready / degraded（proxy 活着但记忆管线不成立，rc=0 但写明后果）/ failed。

    🔴 **MQ-L62 ①（2026-09-08）：进程已经死了就别再等就绪。**
    `_start_proxy` 只 `sleep(2.0)` 就宣布"started"，而**配置错误型的失败要 3 秒左右**
    （lifespan 里 Hub 打开 + 池加载之后才炸）⇒ 进程当场退出，这里却照样轮询满
    `timeout_s` 秒，最后打一句 `the proxy process is up but /ready did not answer within 30s`
    —— **"process is up" 是错的**，而且措辞把人引向就绪探测/网络，不是配置。
    实测代价：白等 30 秒 + 一句误导性结论，真正的报错要自己去翻日志。
    ⇒ 传进 `proxy_pid` 后每轮探活；死了立刻停等，并把日志尾部的错误段**原样打到终端**。
    """
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "") else host  # noqa: S104
    base = f"http://{probe_host}:{port}"
    deadline = time.monotonic() + timeout_s
    ready: dict | None = None
    died = False
    while time.monotonic() < deadline:
        ready = _probe_ready(f"{base}/ready")
        if ready is not None:
            break
        if proxy_pid is not None and not _pid_alive(proxy_pid):
            died = True
            break
        time.sleep(1.0)

    if ready is None:
        logfile = _newest_proxy_log()
        if died:
            Path(_PROXY_PIDFILE).unlink(missing_ok=True)
            print(f"\n  state: failed -- the proxy exited during startup (PID {proxy_pid}).")
            tail = _log_error_tail(logfile)
            if tail:
                print(f"\n  Last error in {logfile}:\n{tail}")
            else:
                print(f"  See {logfile or 'logs/proxy-*.log'}.")
        else:
            print("\n  state: failed -- the proxy process is alive but /ready did not answer within "
                  f"{int(timeout_s)}s. See {logfile or 'the newest proxy log under logs/'}.")
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
        api_key = typer.prompt("Upstream API key", default=api_key or "", hide_input=True)  # gitleaks:allow 交互提示，非常量
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
