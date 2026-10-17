"""生命周期命令：`start` / `stop` / `restart` / `status` / `embed` / `consolidator` / `init` / `doctor` + `_start_*`。

09-06 F0.1 自 `cli.py` 拆出，零行为（函数体逐字搬家）。共享 helper（`_admin_call` / `_http_json` / `_probe_ready` / pidfile 族…）留在门面 `bladex_proxy.cli`；本模块经 `from bladex_proxy.cli import …` 取用，
🔴 故 monkeypatch 要打在**本模块**上（`bladex_proxy.cli.<本模块>.<helper>`），打门面不生效。
"""

from __future__ import annotations

import contextlib
import json
import os
import re
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
from bladex_proxy.storage.redis_launch import (  # noqa: E402
    aof_manifest_problem,
    apply_manifest_repair,
    launch_refusal,
    plan_manifest_repair,
    redis_logfile_path,
    redis_server_command,
    redis_servers_on_data_dir,
    tail_lines,
    write_launch_record,
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
    # 🔴 MQ-A90：data/ 已有 redis-server（哪怕端口不同）⇒ 不拉第二个。
    # 两个实例共用一个目录，关机时各写一份 manifest，后退出的那份落盘。
    refusal = launch_refusal(port)
    if refusal:
        print("  Redis:        not started -- the proxy will forward but NOT store anything")
        print(f"                {refusal}")
        return False
    logfile = redis_logfile_path()
    write_launch_record(logfile, "cli-start", port)
    proc = subprocess.run(  # noqa: S603
        redis_server_command(port, logfile),
        check=False, capture_output=True, text=True, errors="replace")
    time.sleep(1.0)
    if _redis_ping(url):
        print(f"  Redis:        started (AOF, port {port}, log {_display_path(logfile)})")
        return True
    print("  Redis:        failed to start -- the proxy will forward but NOT store anything")
    for line in _redis_failure_detail(logfile, proc, port):
        print(f"                {line}")
    return False


#: manifest 坏了时给的下一步。🔴 不再建议"挪开 appendonlydir 回落 dump.rdb"：MQ-A90 那次
#: dump.rdb 是幽灵实例的快照，比 AOF 旧一周——照做就是静默回退。
_AOF_REPAIR_HINT = (
    "Preview a verified fix (changes nothing):  bladex repair-aof",
    "Then apply it:                             bladex repair-aof --apply",
    "Do not fall back to data/dump.rdb unless you have checked it is not older than the AOF.",
)


def _display_path(path: str) -> str:
    """能相对就相对（打印用）。"""
    try:
        return str(Path(path).relative_to(Path.cwd()))
    except ValueError:
        return path


def _redis_failure_detail(logfile: str, proc: subprocess.CompletedProcess | None,
                          port: int = 6379) -> list[str]:
    """redis 起不来时**把原因说出来**的那几行。

    优先级：① manifest 对账（最常见、且 redis 日志里只说"打不开文件"不说该有谁）；
    ② redis 自己的日志尾部；③ 进程 stderr/stdout（logfile 都没来得及建时）。
    """
    out: list[str] = []
    problem = aof_manifest_problem()
    if problem:
        out.append(f"AOF manifest is inconsistent: {problem}")
        out.extend(_AOF_REPAIR_HINT)
    tail = tail_lines(logfile)
    if tail:
        out.append(f"redis log tail ({_display_path(logfile)}):")
        out.extend(f"  {ln}" for ln in tail)
    elif proc is not None:
        captured = (proc.stderr or "") + (proc.stdout or "")
        lines = [ln.rstrip() for ln in captured.splitlines() if ln.strip()][-8:]
        if lines:
            out.append("redis-server said:")
            out.extend(f"  {ln}" for ln in lines)
    if not out:
        out.append(f"redis-server wrote nothing to {_display_path(logfile)}; "
                   f"check whether the port is taken: lsof -nP -iTCP:{port} -sTCP:LISTEN")
    return out


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


def _deployment_home() -> str:
    """本 CLI 正在操作的部署根（与 InstanceClaim 的身份口径一致）。"""
    from bladex_proxy.deployment import deployment_root
    return os.path.realpath(deployment_root())


def _home_args() -> list[str]:
    """子进程命令行里的部署根标记；分 argv 传递，路径带空格也不会被拆错。"""
    return ["--bladex-home", _deployment_home()]


def _home_from_command(command: str) -> str | None:
    """从命令行文本提取 `--bladex-home`（spawn 侧固定使用“选项 + 值”形态）。"""
    for marker in ("--bladex-home=", "--bladex-home "):
        start = command.find(marker)
        if start < 0:
            continue
        value = command[start + len(marker):]
        if marker.endswith(" "):
            # macOS 没有 /proc，ps 只给空格拼接的文本。home 路径本身可能带空格，
            # 因此以下一个“ --选项”为界，而不是按空白拆 argv。
            value = value.split(" --", 1)[0]
        else:
            value = value.split(" ", 1)[0]
        value = value.strip()
        if value:
            return os.path.realpath(os.path.expanduser(value))
    return None


def _process_home(pid: int) -> str | None:
    """读取 PID 的命令行并返回它声明的 `--bladex-home`（无法判定返回 None）。"""
    try:
        # Linux：/proc 是精确 argv，不受 ps 展示宽度影响。
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        argv = [part.decode("utf-8", errors="replace")
                for part in raw.split(b"\0") if part]
        for i, arg in enumerate(argv):
            if arg == "--bladex-home" and i + 1 < len(argv):
                return os.path.realpath(os.path.expanduser(argv[i + 1]))
            if arg.startswith("--bladex-home="):
                return os.path.realpath(os.path.expanduser(arg.partition("=")[2]))
        return None
    except OSError:
        pass
    try:
        proc = subprocess.run(  # noqa: S603
            ["ps", "-ww", "-p", str(pid), "-o", "command="],  # noqa: S607
            capture_output=True, text=True, timeout=1, check=False)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return _home_from_command(proc.stdout.strip())


def _discard_pidfile(path: str, label: str, reason: str) -> None:
    Path(path).unlink(missing_ok=True)
    print(f"  {label}: {reason}; removing pidfile and continuing")


def _owned_sidecar_pid(path: str, label: str, *, cleanup: bool = True) -> int | None:
    """pidfile 里的 PID 活着，且身份不是“另一个部署根”，才算本实例在跑。

    MQ-A78 复核修正：
    - pidfile 位于本部署 `logs/` 下本身就是所有权证据；老版本进程的命令行
      还没有 `--bladex-home`，因此 **None 不等于外来**，按本实例处理并提示
      重启后生效（否则 status 误删 pidfile，stop 从此停不掉、start 再起一套）。
    - 只有命令行**明确声明另一个** BLADEX_HOME 才判外来；start 可清理这类
      陈旧/异 home 文件，status 是读命令，只汇报、绝不写删。
    """
    try:
        pid = int(Path(path).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        if cleanup and Path(path).exists():
            _discard_pidfile(path, label, "pidfile has no valid PID")
        return None
    if not _pid_alive(pid):
        if cleanup:
            _discard_pidfile(path, label, f"stale pidfile (PID {pid} is not running)")
        return None
    home = _process_home(pid)
    if home is not None and home != _deployment_home():
        if cleanup:
            _discard_pidfile(
                path, label,
                f"PID {pid} belongs to BLADEX_HOME={home}, not this deployment "
                f"({_deployment_home()})")
        return None
    if home is None:
        print(f"  {label}: PID {pid} was started by an older BladeX version "
              "(no --bladex-home marker); treating it as this deployment "
              "-- ownership takes effect after restart")
    return pid


def _flash_pid() -> int | None:
    return _owned_sidecar_pid(_FLASH_PIDFILE, "Flash daemon", cleanup=False)


def _start_flash(interval: float = 60.0) -> bool:
    # 🔴 默认值与 flash_daemon argparse 的 --interval 必须相等且**显式传下去**
    # （刚性原则 12：同一参数两条路两个默认值 = 缺陷；consolidator --concurrency 事故）。
    # 2026-08-29 起 --interval 语义 = 降级兜底轮询间隔（正常运行为 Redis wake 驱动，
    # 零轮询），15 → 60。
    """拉起 flash 维护进程（仅当 module_enabled("flash")——调用方守门）。"""
    if (pid := _owned_sidecar_pid(
            _FLASH_PIDFILE, "Flash daemon", cleanup=True)) is not None:
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
            [*cmd, *_home_args(), "--interval", str(interval)],
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
    """embedding 模块 PID（只认本部署根的 pidfile，不做全机进程名兜底）。"""
    return _owned_sidecar_pid(_EMBED_PIDFILE, "Embedding", cleanup=False)


def _embed_wanted(cfg=None) -> bool:  # noqa: ANN001
    """是否该起 embedding 模块：`BLADEX_MODULE_EMBED` 开、或 backend 已配 ipc。

    🔴 判据取**两者之一**而不是只看模块开关：用户可能先配 `backend=ipc` 试水
    却忘了翻模块开关，那样 proxy 会连不上 socket 然后回落 local（还多占一份内存）
    ——起不该起的进程只是浪费，不起该起的进程是静默降级。

    🔴 `backend=api` 恒不起模块（G18.2，MQ-W9 断点②）：模块的存在意义是加载
    **本地**模型；api 档的向量来自线上供应商，本地起一个模型服务端没有任何消费方，
    只白占内存。冲突（api ∧ 模块开关开着）从静默改成 stderr 一行——不打断启动，
    但用户必须看得见"你开的那个开关没生效、为什么"。

    🔴 backend 读**生效配置**（`resolve_embed_settings`：env > routing.toml > 默认），
    不是裸 env（MQ-W11）：README 主推的 toml 配法配 `backend = "api"` 时，裸读 env
    的旧判据看不见它——模块照起、上面那行 skip 提示也不打，G18.2 在文档教的那条
    路上整个不成立。"这是不是 api 档"全 CLI 只许有这一种问法。
    """
    from bladex_proxy.embedding import resolve_embed_settings
    from bladex_proxy.modules import module_enabled
    if cfg is None:
        from bladex_proxy.config import ProxyConfig
        cfg = ProxyConfig()
    backend = resolve_embed_settings(cfg).backend
    if backend == "api":
        if module_enabled("embed"):
            print('embed_module_skipped backend=api hint="embedding module serves a '
                  'local model; api backend does not use it"', file=sys.stderr)
        return False
    if module_enabled("embed"):
        return True
    return backend == "ipc"


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


def _start_embed(cfg=None) -> bool:  # noqa: ANN001
    """拉起 embedding 模块（V-A6）。未启用时直接返回 True（不是失败）。"""
    if not _embed_wanted(cfg):
        return True
    if (pid := _owned_sidecar_pid(
            _EMBED_PIDFILE, "Embedding", cleanup=True)) is not None:
        print(f"  Embedding:    already running (PID {pid})")
        return True
    try:
        import bladex_proxy.embed_server  # noqa: F401
        cmd = [sys.executable, "-m", "bladex_proxy.embed_server", *_home_args()]
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
    # MQ-A42：模块首启同样要下模型 ⇒ 同一个下载感知（标记来自本模块进程自己）。
    cache_dir = getattr(cfg, "fastembed_cache_path", "") if cfg is not None else ""
    download = _DownloadWatch(cache_dir, proc.pid, "Embedding")
    deadline = time.monotonic() + 20.0
    listening = False
    next_dl_check = 0.0
    while time.monotonic() < deadline:
        if not _pid_alive(proc.pid):
            break
        if _embed_listening():
            listening = True
            break
        if time.monotonic() >= next_dl_check:
            next_dl_check = time.monotonic() + 1.0
            dl = download.poll()
            if dl in ("failed", "stalled"):
                break
            if dl == "downloading":
                deadline = max(deadline, time.monotonic() + 20.0)
        time.sleep(0.3)
    if listening:
        print(f"  Embedding:    started (PID {proc.pid}, log {logfile})")
        return True
    if download.poll() in ("failed", "stalled"):
        with contextlib.suppress(OSError):
            os.kill(proc.pid, signal.SIGTERM)
        Path(_EMBED_PIDFILE).unlink(missing_ok=True)
        print(f"  Embedding:    failed to start -- embedding model download failed: "
              f"{download.error} (log {logfile})")
        return False
    Path(_EMBED_PIDFILE).unlink(missing_ok=True)
    print(f"  Embedding:    failed to start, see {logfile}")
    return False


def _consolidator_pid() -> int | None:
    """consolidator PID（start/status 共用 `_owned_sidecar_pid` 这一个判据）。"""
    return _owned_sidecar_pid(_CONS_PIDFILE, "Consolidator", cleanup=False)


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
    if (pid := _owned_sidecar_pid(
            _CONS_PIDFILE, "Consolidator", cleanup=True)) is not None:
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
            [*cmd, *_home_args(), "--interval", str(interval), "--concurrency", str(concurrency)],
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
    # 批 Z1（MQ-A38 拒绝路径）：2 秒内就退出的形态（配置错误 / 实例冲突）同样把错误段
    # 原样打到终端——"see logfile" 让用户自己去翻，等于把失败原因藏起来（刚性原则 12 ③）。
    tail = _log_error_tail(logfile)
    if tail:
        print(f"\n  Last error in {logfile}:\n{tail}")
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
    force: bool = typer.Option(False, "--force",
                               help="Start even if the embedding preflight fails (engine "
                                    "mismatch with the stamped index, or api probe failure). "
                                    "Config errors that can never work (empty model, "
                                    "unresolved key env) still abort."),
) -> int:
    """Start BladeX (redis + consolidator + proxy).

    Exit codes: 0 = ready; 1 = failed (processes this start launched are stopped again);
    3 = degraded (the proxy is up and forwarding, but part of the memory pipeline is missing --
    run `bladex doctor`).

    \f
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
    # G18.5（MQ-W10）：embedding 配置预检在拉起**任何**进程之前——
    # 配置错误中止启动并指认原因，不起一个记忆已死的 proxy。
    if not _preflight_embedding(cfg, force=force):
        print("=== start aborted (embedding preflight failed) ===")
        return 1
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
        _start_embed(cfg)
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
                               proxy_pid=_read_pidfile(_PROXY_PIDFILE),
                               cache_dir=cfg.fastembed_cache_path)
    # 降级 ≠ 失败：proxy 在跑，旁挂进程是它的一部分，不回收（只有 rc=1 才回收）
    if rc not in (START_RC_READY, START_RC_DEGRADED):
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


_LOG_REF_RE = re.compile(r"[A-Za-z0-9_.-]+\.log\b")


def _referenced_logs(docs_dir: Path) -> set[str]:
    """`docs/` 下任何 Markdown 里出现过的日志文件名（basename）。

    MQ-V21：签收 / 结算稿引用的是具体日志文件，清掉它读数就不可复算。
    扫全部 `docs/`（planning / reviews / benchmarks 之外 ADR 也引日志）——宁多留不误删。
    读不了的文件跳过：少保护一个文件的代价是回到旧行为，不该让 `bladex start` 失败。
    """
    names: set[str] = set()
    if not docs_dir.is_dir():
        return names
    for md in docs_dir.rglob("*.md"):
        try:
            names.update(_LOG_REF_RE.findall(md.read_text(encoding="utf-8", errors="replace")))
        except OSError:
            continue
    return names


def _prune_logs(keep: int | None = None) -> int:
    """启动时清理旧日志（ADR-0027 §4.4）。

    每次启动新建一个时间戳日志文件、从不清理 —— 审查时实测已积到 157 个。
    Memory Hub 无限增长是设计（ADR-0009），日志堆积不是。默认保留最近 20 个，
    `BLADEX_LOG_KEEP=0` 关闭清理。返回删除数量。

    MQ-V21（批 Q3B）：被 `docs/` 引用过的日志**永不清理，也不占 keep 名额**——
    `BLADEX_LOG_KEEP=N` 的含义是"每类再留最近 N 个**未被引用的**"。
    取舍（对比"归档进 docs/benchmarks/"）：引用即保护零操作、不会忘；归档要求
    写稿的人记得跑一步，而这次出事正是因为没人记得。代价是 logs/ 会随被引用的
    文件慢慢长——那些文件本来就该留着。
    """
    if keep is None:
        # 默认值单一真相源在 flags.py（批 Q3C）；写坏回落默认值，与此前同。
        from bladex_core.flags import flag_number
        keep = int(flag_number("BLADEX_LOG_KEEP"))
    if keep <= 0:
        return 0
    root = Path(_repo_root())
    log_dir = root / "logs"
    if not log_dir.is_dir():
        return 0
    referenced = _referenced_logs(root / "docs")
    removed = 0
    # 按前缀分组保留（proxy / consolidator / sync 各留 keep 个，互不挤占）
    groups: dict[str, list[Path]] = {}
    for path in log_dir.glob("*.log"):
        if path.name in referenced:
            continue
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
        print(f"  Logs:         pruned {removed} old log file(s) (keeping {keep} per kind, "
              f"plus any log referenced under docs/; set BLADEX_LOG_KEEP=0 to disable)")
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
    # 优先从**最后一个 traceback** 起：uvicorn 在 lifespan 的 traceback 之后还会补一行
    # `ERROR:    Application startup failed. Exiting.`，只按"最后一个 ERROR:"取尾部就只剩
    # 这句无信息的收尾（批 Z1 实测：MQ-A38 的拒绝原因整段被它挡在终端外）。
    start = 0
    last_tb: int | None = None
    for i, ln in enumerate(lines):
        if ln.startswith("Traceback") or (ln.startswith("ERROR:") and "Traceback" in ln):
            last_tb = i
        if ln.startswith("ERROR:") or ln.startswith("Traceback"):
            start = i
    if last_tb is not None:
        start = last_tb
    tail = [ln for ln in lines[start:] if ln.strip()]
    if len(tail) > max_lines * 2:      # 极长的 traceback：留头留尾
        tail = tail[:max_lines] + ["  ... (truncated; see the full log) ..."] + tail[-max_lines:]
    return "\n".join("    " + ln for ln in tail)


#: MQ-A42：下载期间缓存目录**连续这么久没有长一个字节** ⇒ 判下载停滞、按失败报。
#: 不设总时长上限（2 GB 模型在慢网上下 20 分钟是正常的），只判"不动了"。
_DOWNLOAD_STALL_S = 120.0
#: 下载进度行的打印间隔（秒）。
_DOWNLOAD_PROGRESS_EVERY_S = 10.0


class _DownloadWatch:
    """MQ-A42（批 Z1）：等某个进程就绪时，识别"它正在首次下载 embedding 模型"。

    🔴 **信号来源（原则 13 层 4）**：只读 `embedding.read_download_state(cache_dir, pid)`
    ——即被等的那个进程自己在 `ensure_local_model` 里写的
    按 pid 分文件的下载标记（路径只由 embedding 模块拼）。不看日志文本、不看缓存目录"有没有东西"
    （那答不了"是不是**这个**进程在下"）。缓存字节数只用来画进度与判停滞。

    `poll()` 每秒调一次，返回：
      - `"downloading"`：在下且还在长 ⇒ 调用方把等待截止时间顺延一个完整窗口；
      - `"failed"` / `"stalled"`：下载失败（原文在 `self.error`）/ 停滞超过 `_DOWNLOAD_STALL_S`；
      - `None`：没在下（**真失败的阴性对照走这条**：窗口照旧，不因下载逻辑放大）。
    """

    def __init__(self, cache_dir: str, pid: int | None, label: str) -> None:
        self._cache_dir = cache_dir
        self._pid = pid
        self._label = label
        self.announced = False
        self.error = ""
        self.model = ""
        self._last_bytes = -1
        self._last_growth = 0.0
        self._last_print = 0.0
        self._started = 0.0

    def state(self) -> dict | None:
        if not self._cache_dir or not self._pid:
            return None
        from bladex_proxy.embedding import read_download_state
        return read_download_state(self._cache_dir, self._pid)

    def poll(self) -> str | None:
        st = self.state()
        if st is None:
            if self.announced and not self.error:
                self.announced = False  # 标记已删 = 下完了；只报一次
                print(f"  {self._label}: embedding model download finished "
                      f"({self._elapsed():.0f}s); waiting for startup to complete ...")
            return None
        self.model = str(st.get("model") or "")
        if st.get("state") == "failed":
            self.error = str(st.get("error") or "unknown error")
            return "failed"
        if st.get("state") != "downloading":
            return None
        from bladex_proxy.embedding import cache_bytes
        now = self._now()
        size = st.get("size_gb")
        if not self.announced:
            self.announced = True
            self._started = self._last_growth = self._last_print = now
            size_txt = f" (~{size} GB)" if size else ""
            print(f"  {self._label}: downloading embedding model {self.model}{size_txt} -- "
                  f"first start only, from {st.get('source') or 'huggingface'}; "
                  "this can take a few minutes ...")
        b = cache_bytes(self._cache_dir)
        if b != self._last_bytes:
            self._last_bytes, self._last_growth = b, now
        elif now - self._last_growth > _DOWNLOAD_STALL_S:
            self.error = (f"no download progress for {int(_DOWNLOAD_STALL_S)}s "
                          f"({b / 1e6:.0f} MB in {self._cache_dir})")
            return "stalled"
        if now - self._last_print >= _DOWNLOAD_PROGRESS_EVERY_S:
            self._last_print = now
            total = f" of ~{float(size) * 1e3:.0f}" if size else ""
            print(f"    ... {b / 1e6:.0f}{total} MB so far ({self._elapsed():.0f}s)")
        return "downloading"

    @staticmethod
    def _now() -> float:
        return time.monotonic()

    def _elapsed(self) -> float:
        return self._now() - self._started if self._started else 0.0


#: `bladex start` 的退出码（批 Q3B：此前降级也返回 0，脚本无从判断记忆管线死了没有）。
#: 1 = 失败（本次拉起的旁挂进程已回收）；3 = 降级（proxy 能转发、旁挂进程**保留**，
#: 但记忆管线不完整）。降级不与 1 共用：调用方要能区分"没起来"和"起来了但缺一块"——
#: 前者重试 start，后者该去看 doctor。
START_RC_READY = 0
START_RC_FAILED = 1
START_RC_DEGRADED = 3


def _report_startup_state(host: str, port: int, redis_ok: bool, consolidator_ok: bool,
                          timeout_s: float = 30.0, proxy_pid: int | None = None,
                          cache_dir: str = "") -> int:
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

    🔴 **MQ-A42（批 Z1）：窗口感知首次下载。** 本地档首启要在 proxy lifespan 里阻塞下
    embedding 模型，下载期间 /ready 必然不答 ⇒ 旧行为是每个新装用户首启都看到一次
    `state: failed`。现在 `_DownloadWatch` 读**被等的 proxy 进程自己写的下载标记**
    （信号来源见其 docstring）：在下且在长 ⇒ 打进度、把截止时间顺延一个完整窗口；
    下完后照常判 ready / degraded；下载失败或停滞 ⇒ `failed` 且点名是下载失败，
    并停掉这个 proxy（它会带着"没有 Memory Index"降级跑起来——记忆已死的 proxy
    不留着，与 G18.5 预检同一原则）。**没有下载标记时窗口一秒不多**（阴性对照）。
    """
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "") else host  # noqa: S104
    base = f"http://{probe_host}:{port}"
    deadline = time.monotonic() + timeout_s
    ready: dict | None = None
    died = False
    download = _DownloadWatch(cache_dir, proxy_pid, "Proxy")
    download_failed = False
    while time.monotonic() < deadline:
        ready = _probe_ready(f"{base}/ready")
        if ready is not None:
            # 下载失败时 proxy 会降级起来、/ready 照答——先看一眼标记再下结论
            download_failed = download.poll() == "failed"
            break
        if proxy_pid is not None and not _pid_alive(proxy_pid):
            died = True
            download_failed = download.poll() == "failed"
            break
        dl = download.poll()
        if dl in ("failed", "stalled"):
            download_failed = True
            break
        if dl == "downloading":
            deadline = max(deadline, time.monotonic() + timeout_s)
        time.sleep(1.0)

    if download_failed:
        print(f"\n  state: failed -- the embedding model download failed"
              f"{f' ({download.model})' if download.model else ''}: {download.error}")
        print("  fix: check network access to huggingface.co (or set HF_ENDPOINT to a "
              "reachable mirror), or pre-populate BLADEX_FASTEMBED_CACHE_PATH; then "
              "`bladex start` again.")
        if not died and _read_pidfile(_PROXY_PIDFILE) is not None:
            _stop_by_pidfile(_PROXY_PIDFILE, "Proxy")
        return 1

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
        print(f"  (exit code {START_RC_DEGRADED}: started, but degraded)")
        return START_RC_DEGRADED
    print("  state: ready -- memory pipeline is up (`bladex doctor --e2e` verifies it end to end)")
    return START_RC_READY


@app.command()
def stop() -> int:
    """Stop the proxy and consolidator (local Redis is left running).

    \f
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


@app.command("repair-aof")
def repair_aof(
    apply: bool = typer.Option(False, "--apply",
                               help="Actually rewrite the manifest (default is a preview that changes nothing)"),
) -> int:
    """Fix a Redis AOF manifest that points at files which no longer exist.

    Only moves forward: it rewrites the manifest to the newer files actually on disk,
    after checking that no redis-server is using data/ and that redis-check-aof accepts
    the result. Anything else is refused with the reason. The old manifest is kept
    next to data/appendonlydir.
    """
    plan = plan_manifest_repair()
    for c in plan.checks:
        print(f"  ✓ {c}")
    if not plan.ok:
        print(f"  ✗ not repairing: {plan.reason}")
        return 1
    print("  candidate manifest:")
    for ln in plan.new_manifest.splitlines():
        print(f"    {ln}")
    if not apply:
        print("\n  Preview only. Apply it with:  bladex repair-aof --apply")
        return 0
    kept = apply_manifest_repair(plan)
    print(f"  ✓ manifest rewritten; the old one is kept at {_display_path(str(kept))}")
    print("  Next: bladex start")
    return 0


@app.command()
def embed(
    action: str = typer.Argument("status", help="start | stop | status | restart"),
) -> int:
    """Manage the embedding module (V-A6; independent process, IPC + hot/bulk queue).

    \f
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
    from bladex_proxy.config import ProxyConfig
    cfg = ProxyConfig()

    if action == "status":
        from bladex_proxy import embed_transport
        d = embed_transport.resolve()
        pid = _embed_pid()
        if not _embed_wanted(cfg):
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

    return 0 if _start_embed(cfg) else 1


@app.command()
def consolidator(
    action: str = typer.Argument("status", help="start | stop | status | restart"),
    concurrency: int = typer.Option(
        -1, "--concurrency",
        help="Distillation concurrency (1=serial; defaults to BLADEX_DISTILL_CONCURRENCY, or 4)"),
) -> int:
    """Manage the Memory Index consolidator on its own (it runs as an independent process).

    \f
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


def _format_backlog_trend(samples: dict[str, float]) -> str:
    """Queue 行下面那行：两段积压在最近窗口里的**走向**（MQ-W4）。

    数据来自 proxy `/metrics` 的 `bladex_backlog_*`（proxy 后台每分钟采一次）。
    某一段没有序列 = 样本还不够（proxy 刚起不到三分钟）——明说 "n/a"，不当成持平。
    B10 的判读就看这一行：任一段出现 "rising, sustained" 即为持续上升（判据见 metrics.BacklogTrend）。
    """
    def _get(metric: str, stage: str) -> float | None:
        return samples.get(f'{metric}{{stage="{stage}"}}')

    parts: list[str] = []
    for stage, label in (("hub", "Hub"), ("index", "Index")):
        delta = _get("bladex_backlog_delta_turns", stage)
        trend = _get("bladex_backlog_trend", stage)
        if delta is None or trend is None:
            parts.append(f"-> {label} n/a (not enough samples yet)")
            continue
        span = _get("bladex_backlog_window_span_seconds", stage) or 0.0
        word = "rising" if trend > 0 else "falling" if trend < 0 else "flat"
        if trend > 0 and _get("bladex_backlog_monotonic_rise", stage) == 1.0:
            word = "rising, sustained"
        parts.append(f"-> {label} {int(delta):+d} over {_ago(span).removesuffix(' ago')} ({word})")
    return " / ".join(parts)


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
    # MQ-F4 ②：数的是**投影文件**，不是 Hub 里的账本数——措辞如实。
    parts = [f"{s.ledgers} ledgers projected"]
    # 用户手写文件单列：它是"你改过什么"的入口，也是我们**永不覆盖**的那部分
    parts.append(f"{s.user_files} yours" if s.user_files else "no overrides yet")
    line = daemon_prefix + f"{' / '.join(parts)} ({root})"
    if s.newest_mtime:
        line += f", newest change {_ago(time.time() - s.newest_mtime)}"
    line += ", " + _flash_lag_part(root)
    return line


def _flash_lag_part(root: str) -> str:
    """`flash_lag_s` 那一截（MQ-F4 ①，补遗 G7 口径；daemon 写 `.flash_state.json`）。

    三种形态分开说：没读数（daemon 没跑过画像轨道）/ 在界内 / 越界（ADR-0034 §3.4 上界）。
    "没读数"**不显示成 0**——原则 13 层 4。
    """
    from bladex_core.behavior import FLASH_LAG_BOUND_S
    from bladex_core.flash import read_behavior_status

    st = read_behavior_status(root)
    if not st or "flash_lag_s" not in st:
        return "lag n/a (no behavior pass yet)"
    lag = int(st.get("flash_lag_s", 0))
    bound = int(st.get("lag_bound_s", FLASH_LAG_BOUND_S))
    at = str(st.get("computed_at", ""))[:16].replace("T", " ")
    tail = f" ⚠ over {bound}s bound" if lag > bound else ""
    return f"lag {lag}s{tail} (behavior pass {at}Z)"


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
    # V-A6：embedding 模块（未启用时明说"未启用"，与"挂了"区分开）。
    # G18.3：api 档单列——它没有本地进程可谈，打模型名 + profile 的 dim；
    # 走老分支会打出误导的 "set BLADEX_MODULE_EMBED=1"（那个开关在 api 档下被
    # `_embed_wanted` 明确拒绝，见 G18.2）。
    from bladex_proxy.embedding import resolve_embed_settings as _embed_settings
    _es = _embed_settings(cfg)
    if _es.backend == "api":
        prof = _es.profiles.get(_es.model or "") or {}
        dim_part = f", dim={prof['dim']}" if prof.get("dim") else ""
        model_part = _es.model or "model unset -- set BLADEX_EMBED_MODEL"
        print(f"  Embedding:    api ({model_part}{dim_part}; remote, no local process)")
    elif _embed_wanted(cfg):
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
        print(f"  {'Queue trend:':<14}{_format_backlog_trend(_scrape_metrics(f'{base}/metrics'))}")
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
    return start(foreground=foreground, proxy_only=proxy_only, host="", port=0, force=False)


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

    \f
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

    # 批 Z · Z2.1（MQ-A34 ①）：本地模型没配 context_window 只**提示**，不替用户填——
    # 窗口是硬件常量（`ollama show <model>` 的 num_ctx），BladeX 不猜、不探测。
    _rc = getattr(cfg, "routing_config", None)
    _no_window = (_rc.local_models_without_context_window()
                  if _rc is not None and hasattr(_rc, "local_models_without_context_window") else [])
    if _no_window:
        print(f"  ! Local models without context_window: {', '.join(_no_window)}")
        print("    hint: add `context_window = <tokens>` (e.g. from `ollama show <model>` num_ctx) to "
              "those [[models]] in config/routing.toml so BladeX can route around them when a "
              "request does not fit and trim its own injected surface to the budget; "
              "without it nothing changes for that model.")

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

    # AOF manifest 对账（2026-09-19 事故）：manifest 列的文件缺一个 ⇒ redis 加载 AOF
    # 失败即退出，而 start 只说得出 "failed to start"。这条在 redis 起不来**之前**
    # 就点名是哪个文件不在——判据不能窄于被检查的对象。
    _aof_problem = aof_manifest_problem()
    _check("Redis AOF manifest", _aof_problem is None,
           _aof_problem or "consistent with data/appendonlydir/",
           fix=" / ".join(_AOF_REPAIR_HINT))

    # 🔴 MQ-A90：data/ 上只许有一个 redis-server，且就是配置里那个端口。
    # 第二个实例平时不出声，关机时才把自己那份过期 manifest 写回盘。
    _owners = redis_servers_on_data_dir()
    _cfg_port = _redis_hostport(cfg.redis_url)[1]
    _stray = [o for o in _owners if o.port != _cfg_port]
    _check("Redis servers on data/", len(_owners) <= 1 and not _stray,
           ", ".join(o.describe() for o in _owners) or "none running",
           fix=("more than one redis-server (or one on a port BladeX is not configured for) is "
                "using data/; they overwrite each other's AOF manifest on shutdown. Stop the "
                "stray one: " + "; ".join(
                    f"redis-cli -p {o.port} shutdown" if o.port is not None else f"kill {o.pid}"
                    for o in (_stray or _owners[1:]))))

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

    # embedding 模型（盖章对账 + 按档分流，见 _doctor_embedding）
    try:
        _doctor_embedding(cfg, warmup, _check)
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


def _embed_api_probe(settings) -> tuple[int, float]:  # noqa: ANN001
    """一次真实 api 探活（doctor 与 start 预检共用一份实现）。

    返回 (dim, rtt_ms)；失败原样抛，措辞与去留由调用方决定。
    """
    from bladex_proxy.embedding import RouterEmbedAdapter
    adapter = RouterEmbedAdapter(model=settings.model, api_base=settings.api_base,
                                 api_key=settings.api_key, query_cache_size=0)
    t0 = time.perf_counter()
    vecs = adapter.embed_query(["probe"])
    rtt_ms = (time.perf_counter() - t0) * 1000.0
    dim = len(vecs[0]) if vecs and vecs[0] else 0
    return dim, rtt_ms


def _preflight_embedding(cfg, force: bool = False) -> bool:  # noqa: ANN001
    """G18.5（MQ-W10，Jason 2026-09-18 拍板）：embedding 配置预检，错误中止启动。

    记忆是护城河——配置错误不许降级成"proxy 活着但记忆死了"（MQ-L61 同族）。
    预检只判**配置错误** + api 档一次真探活；运行时故障的降级路径
    （app_factory 的 except Exception）保持不动，那条线是 MQ-L61 划的。

    - ①模型名空 / ②key 声明未解析：**恒中止**（这两种配置下记忆必死，--force 无意义）；
    - ③引擎与盖章不符：中止 + 三条出路（回退配置 / reembed+重标定 / --force 强启）；
    - api 探活失败：中止，--force 可放行（网络临时不通时允许带病起）。
    ③只对 local/api 档判（proxy/ipc 档身份要握手后才知道，预检不做负面判定——
    与 memory_index 「空身份不作负面判定」同则）。
    """
    from bladex_proxy import router_sdk
    from bladex_proxy.embedding import (
        _sensitivity_enabled,
        embed_model_identity,
        resolve_embed_settings,
    )
    s = resolve_embed_settings(cfg)
    # 敏感层开着时 api 档在运行时被强制回本地（fail-closed 既有行为）——
    # 预检跟随**生效**形态：不按声明的 api 检查、不探活，只做③的本地身份对账。
    api_effective = s.backend == "api" and not _sensitivity_enabled(cfg)

    if api_effective:
        if not s.model:
            print("✗ Embedding preflight: backend=api but no embedding model configured.")
            print('  fix: set [embedding].model in config/routing.toml (e.g. '
                  '"openai/text-embedding-3-small") or BLADEX_EMBED_MODEL')
            return False
        key_env = getattr(getattr(getattr(cfg, "routing_config", None), "embedding", None),
                          "api_key_env", "") or ""
        if key_env and not s.api_key:
            print(f"✗ Embedding preflight: api key env {key_env!r} is declared in "
                  "routing.toml but resolves to empty.")
            print(f"  fix: export {key_env}=<key> in the environment that runs bladex "
                  "(keys never go into routing.toml itself)")
            return False

    if s.backend in ("local", "api"):
        expected = embed_model_identity(cfg, role="proxy")
        sidecar = Path(getattr(cfg, "index_path", "") or "") / "embed_model_id"
        stored = sidecar.read_text(encoding="utf-8").strip() if sidecar.is_file() else ""
        if stored and stored != expected:
            if not force:
                print(f"✗ Embedding preflight: engine changed -- Memory Index is stamped "
                      f"{stored!r} but current config resolves to {expected!r}.")
                print("  Starting like this would silently disable vector search (proxy) "
                      "and make the consolidator refuse to write. Pick one:")
                print("  1) keep your data: revert the [embedding] config (backend/model) "
                      "to the stamped engine;")
                print("  2) switch engines for real: rebuild the vector index with "
                      "`python scripts/reembed_index.py` (full re-embed; with backend=api "
                      "this sends ALL memory content to the provider) and re-calibrate "
                      "thresholds on real traffic (non-e5 profiles ship uncalibrated defaults);")
                print("  3) start anyway: `bladex start --force` (memory degraded as above).")
                return False
            print(f"⚠ Embedding preflight: engine mismatch (store={stored!r} "
                  f"current={expected!r}) -- continuing due to --force; vector search "
                  "disabled, consolidator will refuse to write.")

    if api_effective:
        try:
            dim, rtt_ms = _embed_api_probe(s)
        except Exception as exc:  # noqa: BLE001 — 探活失败=配置/网络问题，按 --force 决定去留
            print(f"✗ Embedding preflight: api probe failed: {router_sdk.error_text(exc)}")
            print("  fix: check [embedding] model / api_base and the api key; the model "
                  "name must be a Router embedding id like provider/model")
            if not force:
                print("  (start anyway with `bladex start --force`; memory writes will "
                      "fail until this is fixed)")
                return False
            print("  -- continuing due to --force")
            return True
        print(f"  Embedding:    embed_api_probe_ok dim={dim} rtt_ms={rtt_ms:.0f}")
    return True


def _doctor_embedding(cfg, warmup: bool, check) -> None:  # noqa: ANN001
    """doctor 的 embedding 段：**先**盖章对账，**再**按档分流（MQ-W14）。

    对账放在 api/local 分支**之外**、且在分流**之前**，与 `_preflight_embedding` 同序同判据：
    `expected = embed_model_identity(cfg, role="proxy")`（唯一来源——内部已处理敏感层
    fail-closed 强制回本地与 proxy 档的 role 折算），对 `<index>/embed_model_id` 侧车。
    两者都非空且不等 ⇒ 报，并**跳过分流**（与 preflight 一样，章不对时不去探活/查缓存）。

    为什么必须在分支之外（2026-09-18 复核真跑）：local 分支从来不对账（只 `_model_cached`），
    ⇒ 纯 local + 章不一致 = doctor ✓ / start ✗；W12 把"敏感层 + api"挪进 local 分支后，
    该形态也从"报错误的警"变成"不报警"。**doctor 说健康、start 说起不来** ——
    ADR-0027 §2「红灯不许骗人」的镜像面：绿灯也不许骗人。

    🔴 只对 local / api 两档对账；proxy / ipc 的身份要握手后才知道，不作负面判定
    （与 preflight「空身份不作负面判定」同则）。
    """
    from bladex_proxy.embedding import (
        _local_model_for,
        _model_cached,
        _sensitivity_enabled,
        embed_model_identity,
        resolve_embed_settings,
    )
    settings = resolve_embed_settings(cfg)
    if settings.backend == "api" and _sensitivity_enabled(cfg):
        # MQ-W12：敏感层 fail-closed 在运行时把 api 档强制回本地
        # （build_embedder / embed_model_identity 同判据）。doctor 按**生效**
        # 形态走 local 分支：不外呼探活（敏感层的语义就是不外发）。
        print("  · Embedding: backend=api is overridden by the sensitivity layer "
              "(fail-closed forces local) -- checking the effective local backend")
        settings.backend = "local"
        settings.model = _local_model_for(settings, forced=True)

    if settings.backend in ("local", "api"):
        expected = embed_model_identity(cfg, role="proxy")
        sidecar = Path(getattr(cfg, "index_path", "") or "") / "embed_model_id"
        stored = sidecar.read_text(encoding="utf-8").strip() if sidecar.is_file() else ""
        if stored and stored != expected:
            check("Embedding", False,
                  f"Memory Index embed model mismatch: store={stored!r} current={expected!r} "
                  "-- `bladex start` will refuse to start like this",
                  fix="keep your data: revert the [embedding] config (backend/model) to the "
                      "stamped engine; or switch engines for real: `python scripts/reembed_index.py` "
                      "(full re-embed; with backend=api this sends ALL memory content to the provider)")
            return

    if settings.backend == "local":
        model = settings.model or "(default)"
        cached = _model_cached(settings.model, cfg.fastembed_cache_path) if settings.model else False
        if warmup:
            from bladex_proxy.embedding import build_embedder
            print(f"  … loading embedding model {model} (downloaded automatically if missing; please wait)")
            emb = build_embedder(cfg, role="proxy")
            emb.embed(["warmup"])
            check("Embedding", True, f"local/{model} ready")
        else:
            check("Embedding", True,
                  f"local/{model} {'cached' if cached else 'not cached (downloaded on first start, or run doctor --warmup)'}")
    elif settings.backend == "api":
        _doctor_embed_api(cfg, settings, warmup, check)
    else:
        check("Embedding", True, f"{settings.backend} backend (remote; no local cache check)")


def _doctor_embed_api(cfg, settings, warmup: bool, check) -> None:  # noqa: ANN001
    """G18.4：`backend=api` 的 doctor 路径（MQ-W9 断点④）。

    三条失败各自点名配置（模型名空 / key 声明了但解析为空 / 与已盖章库不同源），
    成功路径报一次真实探活的维度与 RTT。`--warmup` 在本档下没有东西可下载，
    明说跳过而不是装作做了。
    """
    from bladex_proxy import router_sdk

    if warmup:
        print("  · Embedding warmup: nothing to download for backend=api "
              "(the probe below hits the remote endpoint instead)")
    if not settings.model:
        check("Embedding", False, "backend=api but no embedding model configured",
              fix='set [embedding].model in config/routing.toml (e.g. '
                  '"openai/text-embedding-3-small") or BLADEX_EMBED_MODEL')
        return
    key_env = getattr(getattr(getattr(cfg, "routing_config", None), "embedding", None),
                      "api_key_env", "") or ""
    if key_env and not settings.api_key:
        check("Embedding", False,
              f"api key env {key_env!r} is declared in routing.toml but resolves to empty",
              fix=f"export {key_env}=<key> in the environment that runs bladex "
                  "(keys never go into routing.toml itself)")
        return

    try:
        dim, rtt_ms = _embed_api_probe(settings)
    except Exception as exc:  # noqa: BLE001 — 探活失败要指配置，不能让整段 doctor 崩
        check("Embedding", False, router_sdk.error_text(exc),
              fix="check [embedding] model / api_base and the api key; the model name "
                  "must be a Router embedding id like provider/model")
        return

    # 盖章对账不在这里做——已提到 `_doctor_embedding` 的分流之前（MQ-W14），
    # 全 doctor 只许有那一处判定；能走到这里说明章已对上（或还没盖章）。
    check("Embedding", True, f"embed_api_probe_ok dim={dim} rtt_ms={rtt_ms:.0f}")


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
        # MQ-A91：/admin/turns 是全 Hub scan_meta 再排序，17.7k 轮 ≈ 4 s；默认 3 s 超时
        # 会把"已入库"读成 None ⇒ 本步假红。超时放宽只修判读，端点的 O(N) 另账。
        idx = _http_json(f"{base}/admin/turns?limit=20", key=_admin_key(), timeout=30.0)
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
