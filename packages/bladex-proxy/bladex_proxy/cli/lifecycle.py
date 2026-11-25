"""生命周期命令：`start` / `stop` / `restart` / `status` / `embed` / `consolidator` / `init` / `doctor` + `_start_*`。

09-06 F0.1 自 `cli.py` 拆出，零行为（函数体逐字搬家）。共享 helper（`_admin_call` / `_http_json` / `_probe_ready` / pidfile 族…）留在门面 `bladex_proxy.cli`；本模块经 `from bladex_proxy.cli import …` 取用，
🔴 故 monkeypatch 要打在**本模块**上（`bladex_proxy.cli.<本模块>.<helper>`），打门面不生效。
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import typer

from bladex_proxy.cli import (  # noqa: E402
    _CONS_PIDFILE,
    _EMBED_PIDFILE,
    _FLASH_PIDFILE,
    _PROXY_PIDFILE,
    _UNDETERMINED_TEXT,
    RC_UNDETERMINED,
    ShellRestricted,
    _admin_key,
    _admin_status_probe,
    _http_json,
    _is_denied,
    _pid_alive,
    _pid_state,
    _print_admin_unavailable,
    _print_config_not_found,
    _probe_ready,
    _read_pidfile,
    _repo_root,
    _require_config_root,
    _resolve_distill_concurrency,
    _tcp_state,
    app,
)
from bladex_proxy.cli.embed_preflight import _preflight_embedding  # noqa: E402
from bladex_proxy.cli.status_display import (  # noqa: E402
    _ago,
    _consolidator_activity,
    _fmt_bytes,
    _format_backlog_trend,
    _format_queue,
    _print_degradation_banner,
    _print_funnel,
    _print_memory_config,
    _print_traffic,
    _scrape_metrics,
)
from bladex_proxy.storage.redis_launch import (  # noqa: E402
    PID_UNREACHABLE,
    aof_manifest_problem,
    apply_manifest_repair,
    describe_lock_holder,
    instance_lock_holder,
    launch_refusal,
    listening_pids,
    plan_manifest_repair,
    read_instance_lock_record,
    redis_logfile_path,
    redis_server_command,
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
    """Redis 答不答。连不上 ⇒ False；**被系统拦 ⇒ ShellRestricted**（批 W33，MQ-W33）。

    11-05 沙箱里被拦的连接也返回 False，于是 status 打 ✗、start 去拉 Redis、再报「端口不同」。
    """
    try:
        import redis
        rc = redis.Redis.from_url(url, socket_connect_timeout=2)
        return bool(rc.ping())
    except Exception as e:  # noqa: BLE001
        if _is_denied(e):
            raise ShellRestricted(f"connecting to Redis ({url}) was denied: {e}") from e
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


def _process_command(pid: int) -> str | None:
    """进程的命令行；读不到（沙箱拦了 `ps` / 进程刚走）返回 None。"""
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        return b" ".join(p for p in raw.split(b"\0") if p).decode("utf-8", "replace") or None
    except OSError:
        pass
    try:
        proc = subprocess.run(  # noqa: S603
            ["ps", "-ww", "-p", str(pid), "-o", "command="],  # noqa: S607
            capture_output=True, text=True, timeout=1, check=False)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() or None


#: BladeX 进程命令行的标志（proxy 是 `uvicorn bladex_proxy.server:app`，没有 --bladex-home）。
_BLADEX_COMMAND_MARKERS = ("bladex_proxy", "bladex-", "run_index_consolidator")
_ID_OURS = "ours"            # 命令行写着本部署根
_ID_OURS_UNMARKED = "unmarked"  # BladeX 进程，但没有部署根标记（proxy / 老版本）
_ID_FOREIGN = "foreign"      # 命令行明确写着别的部署根
_ID_REUSED = "reused"        # 命令行读到了，不是 BladeX：PID 被复用（系统重启后）
_ID_UNKNOWN = "unknown"      # 命令行读不到


def _process_identity(pid: int) -> tuple[str, str]:
    """pidfile 里这个活着的 PID 是不是我们的进程 ->（身份, 细节）。

    读到了且不是 BladeX 才判复用；读不到就是不知道，绝不当成任何一边。
    """
    home = _process_home(pid)
    if home is not None:
        return (_ID_OURS, home) if home == _deployment_home() else (_ID_FOREIGN, home)
    command = _process_command(pid)
    if command is None:
        return _ID_UNKNOWN, ""
    if any(m in command for m in _BLADEX_COMMAND_MARKERS):
        return _ID_OURS_UNMARKED, command
    return _ID_REUSED, command


def _shell_restriction(*, need_bind: bool = False) -> str | None:
    """当前 shell 是不是被限制到判不了进程和端口（批 W33，MQ-W33）。是 ⇒ 返回原因。

    判据是系统给的 EPERM，不猜「是不是 Codex」：连本机端口被拦，或发信号给 pidfile 里的进程
    被拦。PID 被别的用户的程序复用时 kill 也是 EPERM（系统重启后可能撞上），能读到命令行且
    不是 BladeX 的不算受限，否则一个陈旧 pidfile 就会把 start / stop 永远挡住。
    """
    targets: list[tuple[str, int]] = []
    bind_host = "127.0.0.1"
    try:
        from bladex_proxy.config import ProxyConfig
        cfg = ProxyConfig()
    except Exception:  # noqa: BLE001 —— 配置写坏不该挡住 stop；这里只取要探测的端口
        cfg = None
    if cfg is not None:
        bind_host = "127.0.0.1" if cfg.host in ("0.0.0.0", "") else cfg.host  # noqa: S104
        targets.append((bind_host, cfg.port))
        r_host, r_port = _redis_hostport(cfg.redis_url)
        if _is_local_host(r_host):
            targets.append((r_host, r_port))
    for host, port in targets:
        if _tcp_state(host, port) == "denied":
            return f"connecting to {host}:{port} was denied"
    if need_bind and _is_local_host(bind_host):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind((bind_host, 0))
        except OSError as e:
            if _is_denied(e):
                return f"binding a port on {bind_host} was denied"
        finally:
            s.close()
    for pidfile in (_PROXY_PIDFILE, _CONS_PIDFILE, _FLASH_PIDFILE, _EMBED_PIDFILE):
        try:
            pid = int(Path(pidfile).read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
        if _pid_state(pid) == PID_UNREACHABLE and _process_identity(pid)[0] != _ID_REUSED:
            return f"signalling PID {pid} from {pidfile} was denied"
    return None


def _refuse_restricted(action: str, reason: str) -> int:
    """受限环境里 start / stop：什么都不做，说清楚，用单独的退出码。"""
    print(f"  bladex {action} refused: {_UNDETERMINED_TEXT}: {reason}.")
    print(f"  Nothing was started, stopped or deleted. Run `bladex {action}` from a normal terminal.")
    print(f"  (exit code {RC_UNDETERMINED})")
    return RC_UNDETERMINED


def _discard_pidfile(path: str, label: str, reason: str) -> None:
    Path(path).unlink(missing_ok=True)
    print(f"  {label}: {reason}; removing pidfile and continuing")


def _owned_sidecar_pid(path: str, label: str, *, cleanup: bool = True,
                       expects_home_marker: bool = True) -> int | None:
    """pidfile 里的 PID 活着，且身份不是“另一个部署根”，才算本实例在跑。

    MQ-A78 复核修正：
    - pidfile 位于本部署 `logs/` 下本身就是所有权证据；老版本进程的命令行
      还没有 `--bladex-home`，因此 **None 不等于外来**，按本实例处理并提示
      重启后生效（否则 status 误删 pidfile，stop 从此停不掉、start 再起一套）。
    - 只有命令行**明确声明另一个** BLADEX_HOME 才判外来；start 可清理这类
      陈旧/异 home 文件，status 是读命令，只汇报、绝不写删。

    批 W33（MQ-W33）：`_pid_alive` 只在确认进程不存在时返回 False，EPERM 不会走到删文件。
    命令行读到了且不是 BladeX ⇒ PID 被复用，文件记的那个进程已不存在，按陈旧处理。
    命令行读不到 ⇒ 不知道，按在跑算、不删（`restore` / `import` 前置检查宁可挡住）。
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
    identity, detail = _process_identity(pid)
    if identity == _ID_FOREIGN:
        if cleanup:
            _discard_pidfile(
                path, label,
                f"PID {pid} belongs to BLADEX_HOME={detail}, not this deployment "
                f"({_deployment_home()})")
        return None
    if identity == _ID_REUSED:
        if cleanup:
            _discard_pidfile(path, label, f"stale pidfile (PID {pid} is now another program)")
        return None
    if identity == _ID_OURS_UNMARKED and expects_home_marker:
        print(f"  {label}: PID {pid} was started by an older BladeX version "
              "(no --bladex-home marker); treating it as this deployment "
              "-- ownership takes effect after restart")
    return pid


def _instance_lock_free(name: str, label: str, pidfile: str) -> bool:
    """拉起前看单实例锁（MQ-W35）。有人持锁而 pidfile 里没有它 ⇒ 不拉，点名持有者。

    不看这一步也不会并存（新进程取不到锁就退出），但 CLI 会先打一句 started，
    再留下一个写着死 PID 的 pidfile。
    """
    holder = instance_lock_holder(name, _deployment_home())
    if holder is None:
        return True
    print(f"  {label + ':':<13} not started -- {describe_lock_holder(holder)} is already running "
          f"on this deployment but is not in {pidfile}. Stop that process first, then "
          f"`bladex start` again.")
    return False


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
    if not _instance_lock_free("flash", "Flash daemon", _FLASH_PIDFILE):
        return False
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

    🔴 backend 读**生效配置**（`resolve_embed_settings`：env > bladex.toml > routing.toml > 默认），
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
    """embedding 服务是否**真的在监听**（判活的正确判据，不是"进程还在"）。

    批 W33（MQ-W35）：uds 也要真连一次。文件还在不代表有人服务（进程被 kill -9 后留下的
    socket），连接被系统拦 ⇒ ShellRestricted，不当成「没在监听」。
    """
    from bladex_proxy import embed_transport
    d = embed_transport.resolve()
    if d.transport == embed_transport.TRANSPORT_UDS:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(1.0)
        try:
            s.connect(d.address)
            return True
        except OSError as e:
            if _is_denied(e):
                raise ShellRestricted(f"connecting to {d.address} was denied: {e}") from e
            return False
        finally:
            s.close()
    if d.transport == embed_transport.TRANSPORT_TCP:
        host, _, port = d.address.rpartition(":")
        host = host or "127.0.0.1"
        state = _tcp_state(host, int(port))
        if state == "denied":
            raise ShellRestricted(f"connecting to {host}:{port} was denied")
        return state == "open"
    return False


def _embed_lock_owner() -> int | None:
    return (read_instance_lock_record("embed", _deployment_home()) or {}).get("pid")


def _start_embed(cfg=None) -> bool:  # noqa: ANN001
    """拉起 embedding 模块（V-A6）。未启用时直接返回 True（不是失败）。"""
    if not _embed_wanted(cfg):
        return True
    if (pid := _owned_sidecar_pid(
            _EMBED_PIDFILE, "Embedding", cleanup=True)) is not None:
        print(f"  Embedding:    already running (PID {pid})")
        return True
    if not _instance_lock_free("embed", "Embedding", _EMBED_PIDFILE):
        return False
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
        # 🔴 批 W33：答话的得是**这个**进程（它持着单实例锁）。否则没持锁的旧 embed 还在答时，
        # 这里会打 started，而新进程随后因为 socket 在用而退出。
        if _embed_listening() and _embed_lock_owner() == proc.pid:
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


#: consolidator 等 `index_consolidator_ready` 的上限（秒，MQ-W36）。
#: 取值依据：live 最近 9 次成功启动，进程起来后 0–3 s 就打出这行；30 s 留 10 倍余量，也与
#: proxy 等 /ready 的窗口相同。起即退的路径不吃这个上限（进程一退就停等）；只有进程活着
#: 却一直不就绪才等满，而这时结论是「还在初始化」，不算失败：首装下模型可以比这长得多。
_CONSOLIDATOR_READY_WAIT_S = 30.0
_CONSOLIDATOR_READY_MARK = "index_consolidator_ready"


def _await_consolidator_ready(pid: int, logfile: str, timeout_s: float | None = None) -> str:
    """等到就绪日志或进程退出 -> `ready` / `exited` / `starting`（到上限仍活着且没就绪）。"""
    deadline = time.monotonic() + (_CONSOLIDATOR_READY_WAIT_S if timeout_s is None else timeout_s)
    while True:
        if not _pid_alive(pid):
            return "exited"
        try:
            if _CONSOLIDATOR_READY_MARK in Path(logfile).read_text(encoding="utf-8", errors="replace"):
                return "ready"
        except OSError:
            pass
        if time.monotonic() >= deadline:
            return "starting"
        time.sleep(0.5)


def _newest_consolidator_log() -> str:
    try:
        logs = sorted(Path("logs").glob("consolidator-*.log"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        return ""
    return str(logs[0]) if logs else ""


def _consolidator_pid() -> int | None:
    """consolidator PID（start/status 共用 `_owned_sidecar_pid` 这一个判据）。"""
    return _owned_sidecar_pid(_CONS_PIDFILE, "Consolidator", cleanup=False)


def _start_consolidator(interval: int, concurrency: int | None = None,
                        *, wait_ready: bool = False) -> bool:
    """拉起 consolidator 守护进程。

    🔴 `--concurrency` 必须显式传下去（2026-08-22 事故）：此前这里只传
    `--interval`，而 `consolidator.py` 的 argparse 默认是 `0 → 1` 串行，
    于是 `bladex start` / `bladex consolidator start` 起来的守护进程**并发恒为 1**，
    而 `bladex sync` 那条路默认 4——手工 sync 快、守护进程慢，同一个参数两条路
    两个默认值。实测后果：mean 23.2s/次 × 8–9 次调用/轮串行 = 5–26 轮/小时，
    Memory Index 积压 720 轮持续不降（新记忆写不进去 = 召不回）。
    参数含义与取值依据见 `flags.MEMORY_NUMERIC_DEFAULTS`。

    🔴 批 W33（MQ-W36）：这里的 1.5 s 只筛「当场就退」的。撞 Index 写锁要先开 Hub、建 embedder，
    晚于这 1.5 s（11-05 13:54 打了 started，随后退出）。真正的判定：
    - `bladex start`：`_report_startup_state` 在 proxy 起来**之后**等。不能排在 proxy 之前：
      空部署首启 consolidator 要等 proxy 建 Hub（MQ-A81），先等它就是干等到上限。
    - `bladex consolidator start`：`wait_ready=True`，当场等。
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
    state = _await_consolidator_ready(proc.pid, logfile) if wait_ready else None
    if state != "exited" and _pid_alive(proc.pid):
        print(f"  Consolidator: started (PID {proc.pid}, interval {interval}s, "
              f"concurrency {concurrency}, log {logfile})")
        if state == "starting":
            print(f"                still initializing after {_CONSOLIDATOR_READY_WAIT_S:.0f}s "
                  f"(no {_CONSOLIDATOR_READY_MARK} in the log yet)")
        return True
    Path(_CONS_PIDFILE).unlink(missing_ok=True)
    print(f"  Consolidator: failed to start, see {logfile}")
    if wait_ready:  # `bladex start` 走 `_report_startup_state`，尾部在那里打，不重复
        for line in tail_lines(logfile):
            print(f"    {line}")
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


def _stop_by_pidfile(path: str, name: str, grace_s: int = 5) -> bool:
    """停 pidfile 里的进程。返回 False = 没停成，**pidfile 留着**，调用方要给非 0 退出码。

    🔴 批 W33（MQ-W33）：只在确认进程不存在时删 pidfile。此前 EPERM 被判成「没在跑」⇒ 打印
    not running + 删文件，进程没收到任何信号、继续运行，此后 status / stop 都看不见它。
    身份也要核对：系统重启后 PID 可能被别的程序复用，只看活着就发信号会误杀。
    """
    try:
        raw = Path(path).read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        print(f"  {name}: not running")
        return True
    except OSError as e:
        print(f"  {name}: {_UNDETERMINED_TEXT}: cannot read {path}: {e}. Pidfile left in place.")
        return False
    pid = int(raw) if raw.isdigit() else 0
    if pid <= 0 or not _pid_alive(pid):
        print(f"  {name}: not running")
        Path(path).unlink(missing_ok=True)
        return True

    identity, detail = _process_identity(pid)
    if identity == _ID_FOREIGN:
        print(f"  {name}: not stopped -- PID {pid} belongs to BLADEX_HOME={detail}, not this "
              f"deployment; removing the pidfile")
        Path(path).unlink(missing_ok=True)
        return True
    if identity == _ID_REUSED:
        print(f"  {name}: not running -- PID {pid} is now another program ({detail[:80]}); "
              f"not signalling it, removing the stale pidfile")
        Path(path).unlink(missing_ok=True)
        return True
    if identity == _ID_UNKNOWN:
        print(f"  {name}: {_UNDETERMINED_TEXT}: PID {pid} is alive but its command line cannot "
              f"be read, so it was not signalled. Pidfile left in place.")
        return False

    def _signal(sig: signal.Signals) -> bool:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass
        except PermissionError:
            print(f"  {name}: {_UNDETERMINED_TEXT}: signalling PID {pid} was denied. "
                  f"It was not stopped; pidfile left in place.")
            return False
        return True

    if not _signal(signal.SIGTERM):
        return False
    for _ in range(grace_s):
        if not _pid_alive(pid):
            break
        time.sleep(1.0)
    if _pid_alive(pid):
        if not _signal(signal.SIGKILL):
            return False
        for _ in range(20):
            if not _pid_alive(pid):
                break
            time.sleep(0.1)
        if _pid_alive(pid):
            print(f"  {name}: PID {pid} is still alive after SIGKILL; pidfile left in place")
            return False
    Path(path).unlink(missing_ok=True)
    print(f"  {name}: stopped (PID {pid})")
    return True


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

    Exit codes: 0 = ready; 1 = failed (the output says what is still running: if the proxy exited,
    processes this start launched are stopped again; if it is alive but /ready did not answer in
    time, nothing is stopped); 3 = degraded (the proxy is up and forwarding, but part of the memory
    pipeline is missing -- run `bladex doctor`); 4 = refused, this shell is restricted by a sandbox
    or permissions and nothing was started.

    \f
    收编 run_proxy.sh 的启动逻辑。
    """
    root = _require_config_root()
    if root is None:
        _print_config_not_found()
        return 1
    if root != os.getcwd():
        os.chdir(root)  # 之后的相对路径（data/、logs/、PID 文件）都以部署根为准
    # 🔴 批 W33：受限环境直接拒绝。11-05 12:03 沙箱里那次 start 拉起的 embed 把 live socket 删了，
    # 留下的 consolidator.pid 又指向已退出的进程。
    if (restriction := _shell_restriction(need_bind=True)) is not None:
        return _refuse_restricted("start", restriction)
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
    # 🔴 批 W33（MQ-W34）：拉起任何进程之前查端口。陈旧 proxy.pid（含 PID 被复用）先清掉，
    # 否则后面会把它当成「already running」。
    _owned_sidecar_pid(_PROXY_PIDFILE, "Proxy", cleanup=True, expects_home_marker=False)
    if (occupied := _port_occupant_refusal(host, port)) is not None:
        print(f"  Proxy:        not started -- {occupied}")
        print("=== start aborted (port in use) ===")
        return START_RC_FAILED
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
    sidecar_problems: list[str] = []
    if not proxy_only:
        redis_ok = _ensure_redis(cfg.redis_url)
        # V-A6：embedding 模块要在 proxy/consolidator **之前**起——它俩启动时
        # 就会解析传输并尝试连接；晚起会让首个请求走一次回落 local（多占内存）。
        if _start_embed(cfg) is False:
            sidecar_problems.append("embedding module not started by this run"
                                    " (see the Embedding line above and logs/embed-*.log)")
        consolidator_ok = _start_consolidator(
            int(os.environ.get("BLADEX_CONSOLIDATOR_INTERVAL", "60") or 60))
        # V-F2：flash 维护进程随 start 拉起——module 门在此守（默认关=不起）。
        from bladex_proxy.modules import module_enabled as _me
        if _me("flash") and _start_flash() is False:
            sidecar_problems.append("Memory Flash daemon not started by this run"
                                    " (see the Flash daemon line above and logs/flash-*.log)")
    if _read_pidfile(_PROXY_PIDFILE) is not None:
        print(f"  Proxy:        already running (PID {_read_pidfile(_PROXY_PIDFILE)}) -- run `bladex stop` first")
        return 1
    rc = _start_proxy(host, port, foreground)
    if foreground or rc != 0:
        return rc
    # 只等这次拉起的；已经在跑的不等（它的启动日志可能早被清理了）
    cons_now = _read_pidfile(_CONS_PIDFILE)
    rc = _report_startup_state(host, port, redis_ok, consolidator_ok,
                               proxy_pid=_read_pidfile(_PROXY_PIDFILE),
                               cache_dir=cfg.fastembed_cache_path,
                               check_embed_fallback=not proxy_only,
                               check_consolidator=(not proxy_only and (
                                   not consolidator_ok
                                   or cons_now != pids_before["Consolidator"])),
                               extra_degraded=sidecar_problems)
    # 降级 ≠ 失败：proxy 在跑，旁挂进程是它的一部分，不回收（只有 rc=1 才回收）
    if rc not in (START_RC_READY, START_RC_DEGRADED):
        # 🔴 MQ-W30（批 W21b）：proxy 活着但窗口内没 ready ⇒ 什么都不停，如实报状态。
        # 11-03 事故：proxy 33 s 才 ready（窗口 30 s），CLI 停掉三个旁挂，打「Proxy did not come
        # up」，而 proxy 几秒后就 ready 了 ⇒ 转发正常、记忆管线全停，用户看到的却是「失败」。
        # 不选「连 proxy 一起停」：启动慢不等于起不来，停掉的话慢启动的部署（库越长越慢）每次都
        # 起不来，agent 全部 ECONNREFUSED；留着最坏是一个没 ready 的进程，`bladex stop` 就能收。
        # proxy 真退出 / 下载失败时 pidfile 已摘，仍走回收（MQ-L62 ②）。
        proxy_pid = _read_pidfile(_PROXY_PIDFILE)
        if proxy_pid is None:
            _rollback_sidecars(pids_before, _SIDECARS)
        else:
            _report_not_ready_processes(proxy_pid, _SIDECARS)
    return rc


def _describe_pids(pids: list[int]) -> str:
    """把 PID 连同命令行写出来：只给数字的话，用户还得自己查一遍那是什么。"""
    parts = []
    for pid in pids:
        command = _process_command(pid)
        parts.append(f"PID {pid} ({command[:100]})" if command else f"PID {pid}")
    return ", ".join(parts)


def _port_occupant_refusal(host: str, port: int) -> str | None:
    """端口被 pidfile 之外的进程占着 ⇒ 返回拒绝原因（批 W33，MQ-W34）。

    🔴 11-05 排查复现：旧 proxy 脱管后还占着端口，`/ready` 答话的是它，`start` 报 `state: ready`、
    退出码 0；新 proxy 随后 bind 失败退出，留下写着死 PID 的 pidfile。端口真被占时新 proxy 本来
    就起不来，这里拒绝不会误伤。
    """
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "") else host  # noqa: S104
    if _tcp_state(probe_host, port) != "open":
        return None
    own = _read_pidfile(_PROXY_PIDFILE)
    listeners = listening_pids(port)
    if own is not None and (listeners is None or own in listeners):
        return None  # 就是我们自己的 proxy，走后面的 already running
    who = (_describe_pids(listeners) if listeners
           else "a process lsof cannot name (check with: lsof -nP -iTCP:"
                f"{port} -sTCP:LISTEN)")
    answers = _probe_ready(f"http://{probe_host}:{port}/ready") is not None
    kind = ("It answers /ready, so it is probably a BladeX proxy this CLI does not track"
            if answers else "It does not answer /ready, so it is probably not a BladeX proxy")
    return (f"port {port} is held by {who}, which is not in {_PROXY_PIDFILE}. {kind}. "
            f"Check that process and stop it if it is stale, or set BLADEX_PORT; "
            f"a new proxy could not bind this port.")


def _report_not_ready_processes(proxy_pid: int, sidecars: tuple[tuple[str, str], ...]) -> None:
    """proxy 活着但未 ready 时，把进程实况打全（MQ-W30）。每行都现查，不复述启动时的结论。"""
    prefix = {"Embedding": "embed", "Consolidator": "consolidator", "Flash daemon": "flash"}
    print("\n  Nothing was stopped. Current processes:")
    print(f"    Proxy:        alive, not ready yet (PID {proxy_pid}, log "
          f"{_newest_proxy_log() or 'logs/proxy-*.log'})")
    for name, pf in sidecars:
        pid = _read_pidfile(pf)
        state = f"running (PID {pid})" if pid is not None else "not running"
        print(f"    {name + ':':<13} {state}, log logs/{prefix.get(name, name.lower())}-*.log")
    print("  It may still become ready: re-check with `bladex status`. "
          "To stop everything: `bladex stop`.")


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


_EMBED_FALLBACK_RE = re.compile(r"embed_ipc_fallback_local\b.*?\berror=(?:'([^']*)'|\"([^\"]*)\"|(\S*))")


def _embed_fallback_problems() -> list[str]:
    """本次启动的 proxy / consolidator 日志里有没有回落到进程内模型（批 W21b）。

    embed 服务活着、在监听，但回包全失败时，/ready 照样 200、转发和记忆都能跑 ⇒ 不看这个就会打
    「memory pipeline is up」，而事实是两个进程各多占一份模型内存，embed 服务形同虚设。
    只读每类最新一个日志（刚拉起的那个进程写的）。
    """
    who: dict[str, str] = {}
    for name, pattern in (("proxy", "proxy-*.log"), ("consolidator", "consolidator-*.log")):
        logs = sorted(Path("logs").glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if not logs:
            continue
        try:
            m = _EMBED_FALLBACK_RE.search(logs[0].read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
        if m:
            who[name] = next((g for g in m.groups() if g), "") or "unknown"
    if not who:
        return []
    reason = next(iter(who.values()))
    return [f"embedding module not serving -> {' and '.join(who)} fell back to an in-process model "
            f"(memory doubles until restart): {reason} (see logs/embed-*.log)"]


#: `bladex start` 报 ready 之后，再等 `/admin/status` 第一次回答的上限（批 LC，MQ-W40）。
#: 依据：scratch 实例套 live 规模的 Hub（23k 轮 / 27.6k 键）三次起，ready 之后 `/admin/status`
#: 首答都在 ≈9.6 s（Hub 进程内索引首次建成 ≈6 s + 会话状态重放，二者同在 lifespan 之后的后台里）；
#: 30 s = 3 倍余量，也与等 `/ready` 的窗口同值。超时 ⇒ degraded 并点名，不是 failed：proxy 能转发。
_ADMIN_STATUS_WAIT_S = 30.0
_ADMIN_STATUS_PROBE_TIMEOUT_S = 3.0


def _await_admin_status(base: str, proxy_pid: int | None, cap_s: float | None = None) -> str | None:
    """等 `/admin/status` 第一次回答。返回 None = 回答了（或不必等）；字符串 = 要进 degraded 的一句话。

    只在本次拉起了 proxy（`proxy_pid` 非空）时等；进程中途死了不判（上游已有「进程死了」的路径）。
    """
    if proxy_pid is None:
        return None
    cap = _ADMIN_STATUS_WAIT_S if cap_s is None else cap_s
    url = f"{base}/admin/status"
    t0 = time.monotonic()
    announced = False
    while True:
        outcome = _admin_status_probe(url, _admin_key(), _ADMIN_STATUS_PROBE_TIMEOUT_S)
        if outcome != "timeout":
            return None
        waited = time.monotonic() - t0
        if not _pid_alive(proxy_pid):
            return None
        if waited >= cap:
            return (f"/admin/status did not answer within {int(cap)}s after /ready -> `bladex status` "
                    f"will show no Memory rows until it does (the Hub index may still be building; "
                    f"see {_newest_proxy_log() or 'the newest proxy log under logs/'})")
        if not announced:
            print("  waiting for /admin/status (the proxy is still warming up its Memory Hub index) ...")
            announced = True


def _report_startup_state(host: str, port: int, redis_ok: bool, consolidator_ok: bool,
                          timeout_s: float = 30.0, proxy_pid: int | None = None,
                          cache_dir: str = "", check_embed_fallback: bool = False,
                          check_consolidator: bool = False,
                          extra_degraded: list[str] | None = None) -> int:
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

    🔴 **批 W33（MQ-W34）：`/ready` 答话不等于新 proxy 起来了。** 判 ready 要求新拉起的进程活着，
    且端口的监听者就是它。开跑前的端口检查挡住了绝大多数情况，这里挡检查之后才抢到端口的。
    lsof 不可用时只能退回「进程活着」，并明说没核实。

    🔴 **批 W33（MQ-W36）：出结论前重查 consolidator**，等到就绪日志或进程退出。
    """
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "") else host  # noqa: S104
    base = f"http://{probe_host}:{port}"
    deadline = time.monotonic() + timeout_s
    ready: dict | None = None
    died = False
    download = _DownloadWatch(cache_dir, proxy_pid, "Proxy")
    download_failed = False
    impostors: list[int] = []
    listener_unverified = False
    while time.monotonic() < deadline:
        ready = _probe_ready(f"{base}/ready")
        if ready is not None and proxy_pid is not None:
            if not _pid_alive(proxy_pid):
                ready, died = None, True
                download_failed = download.poll() == "failed"
                break
            listeners = listening_pids(port)
            listener_unverified = listeners is None
            if listeners is not None and proxy_pid not in listeners:
                ready, impostors = None, listeners
                time.sleep(1.0)
                continue
            impostors = []
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
        elif impostors:
            print(f"\n  state: failed -- not ready: /ready on port {port} is answered by "
                  f"{_describe_pids(impostors)}, not by the proxy this run started "
                  f"(PID {proxy_pid}).")
        else:
            print("\n  state: failed -- not ready: the proxy process is alive but /ready did not "
                  f"answer within {int(timeout_s)}s. See {logfile or 'the newest proxy log under logs/'}.")
        return 1

    consolidator_tail: list[str] = []
    consolidator_note = ""
    if check_consolidator:
        cons_log = _newest_consolidator_log()
        cons_pid = _read_pidfile(_CONS_PIDFILE) if consolidator_ok else None
        cons_state = _await_consolidator_ready(cons_pid, cons_log) if cons_pid else "exited"
        if cons_state == "exited":
            consolidator_ok = False
            Path(_CONS_PIDFILE).unlink(missing_ok=True)
            consolidator_tail = tail_lines(cons_log) if cons_log else []
        elif cons_state == "starting":
            consolidator_note = (f"consolidator (PID {cons_pid}) is still initializing "
                                 f"after {_CONSOLIDATOR_READY_WAIT_S:.0f}s; see {cons_log}")

    admin_note = _await_admin_status(base, proxy_pid)

    checks = ready.get("checks", ready) if isinstance(ready, dict) else {}
    degraded: list[str] = list(extra_degraded or [])
    if admin_note:
        degraded.append(admin_note)
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
    if check_embed_fallback:
        degraded.extend(_embed_fallback_problems())

    print(f"\n  Dashboard:    {base}/dashboard")
    if listener_unverified:
        print(f"  note: could not verify which process listens on port {port} (lsof unavailable); "
              f"checked only that PID {proxy_pid} is alive.")
    if consolidator_note:
        print(f"  note: {consolidator_note}")
    if consolidator_tail:
        print(f"  Consolidator log tail ({_newest_consolidator_log()}):")
        for line in consolidator_tail:
            print(f"    {line}")
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

    批 W33：受限环境直接拒绝（RC_UNDETERMINED）；单个进程没停成不中断其余几个，但退出码是 1。
    """
    print("=== BladeX stop ===")
    if (restriction := _shell_restriction()) is not None:
        return _refuse_restricted("stop", restriction)
    results = [
        _stop_by_pidfile(_PROXY_PIDFILE, "Proxy"),
        _stop_by_pidfile(_CONS_PIDFILE, "Consolidator"),
        # 四个进程行**无条件汇报**（与 Proxy/Consolidator 一致）：第二次 stop 时
        # "这行消失了" 与 "not running" 传达的信息量完全不同（Jason 2026-08-28 复核）。
        _stop_by_pidfile(_FLASH_PIDFILE, "Flash daemon"),
        _stop_by_pidfile(_EMBED_PIDFILE, "Embedding"),
    ]
    print("  Redis:        left running (stop it with: redis-cli shutdown)")
    return 0 if all(results) else 1


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
    if (restriction := _shell_restriction(need_bind=action in {"start", "restart"})) is not None:
        return _refuse_restricted(f"embed {action}", restriction)
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
        if not _stop_by_pidfile(_EMBED_PIDFILE, "Embedding"):
            return 1
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

    if (restriction := _shell_restriction()) is not None:
        return _refuse_restricted(f"consolidator {action}", restriction)

    if action == "status":
        pid = _consolidator_pid()
        if pid is None:
            print("  Consolidator: ✗ not running -- no facts are being distilled")
            return 1
        print(f"  Consolidator: ✓ running (PID {pid})")
        return 0

    if action in {"stop", "restart"}:
        if not _stop_by_pidfile(_CONS_PIDFILE, "Consolidator"):
            return 1
        if action == "stop":
            return 0

    interval = int(os.environ.get("BLADEX_CONSOLIDATOR_INTERVAL", "60") or 60)
    conc = concurrency if concurrency >= 1 else None  # <1 → 走 flags 默认
    return 0 if _start_consolidator(interval, conc, wait_ready=True) else 1


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


def _flash_refused_text() -> tuple[int | None, str]:
    """被拒直编的一行（批 DS2，MQ-F16）；`bladex status` / `doctor` 共用。

    (None, …) = 没有读数，**不说成 0**（原则 13 层 4）。读取点与 `/admin/status.flash` 是同一个函数。
    """
    from bladex_proxy.config import resolve_flash_path
    from bladex_proxy.flash_daemon import read_user_edit_refused

    rows = read_user_edit_refused(resolve_flash_path())
    if rows is None:
        return None, "n/a (the flash process has not reported yet)"
    if not rows:
        return 0, "none refused"
    shown = "; ".join(f"{r['ledger_id']}: {r['reason']}" for r in rows[:3])
    more = f"; and {len(rows) - 3} more" if len(rows) > 3 else ""
    return len(rows), f"{len(rows)} ledger file edit(s) not adopted -- {shown}{more}"


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


def _unmanaged_proxy_lines(port: int) -> list[str]:
    """端口上的监听者与 `logs/proxy.pid` 对不上时的警告行（批 LC，MQ-W38）。只告警，不杀。

    只用监听者（`lsof`）与 pidfile 比对，**不按 ps 命令行过滤**：uvicorn 的命令行没有 `--bladex-home`
    （按它过滤会漏报），而 `--foreground` / 手动起的进程又会被命令行过滤误报成脱管。
    `lsof` 不可用或读不到 ⇒ 不打任何东西（判不了不等于没问题，但也不该编一条警告）。
    """
    listeners = listening_pids(port)
    if not listeners:
        return []
    managed = _read_pidfile(_PROXY_PIDFILE)
    if managed is not None and managed in listeners:
        return []
    who = "; ".join(f"PID {p} ({_process_command(p) or 'command line unreadable'})" for p in listeners)
    pidfile_part = (f"`{_PROXY_PIDFILE}` says PID {managed}" if managed is not None
                    else f"`{_PROXY_PIDFILE}` is missing or stale")
    return [f"  ⚠ Proxy:        port {port} is served by {who}, but {pidfile_part} -- "
            "this proxy is not managed by `bladex stop` / `restart` (foreground or hand-started?). "
            "Nothing was killed; check it and stop it yourself if it is stray."]


def _contradiction_status_line(admin: dict | None) -> str | None:
    """批 BB · 2b：`bladex status` 的矛盾检测一行。只有开关打开时 `/admin/status` 才带 `contradiction`；
    关着（或 proxy 不可达）⇒ None，这一行不出现（关着时零行为差异）。"""
    cd = (admin or {}).get("contradiction")
    if not cd:
        return None
    used, cap = cd.get("daily_input_tokens_used"), cd.get("daily_input_tokens_cap")
    return (f"  {'Contradictions:':<14}on (experimental) · pending {cd.get('pending', '?')}"
            f" · today {'?' if used is None else f'{used:,}'} / {'?' if cap is None else f'{cap:,}'} input tokens")


@app.command()
def status(
    traffic: bool = typer.Option(False, "--traffic",
                                 help="Show a one-line summary of the most recent turns"),
    n: int = typer.Option(10, "-n", help="How many turns to show with --traffic"),
) -> int:
    """Show runtime status: processes, /ready probe, and an /admin/status summary.

    Exit code 4 = this shell is restricted by a sandbox or permissions, so nothing could be
    determined (that is different from 1, which means something is actually down).
    """
    from bladex_proxy.config import ProxyConfig
    cfg = ProxyConfig()
    exit_code = 0

    # 🔴 批 W33（MQ-W33）：11-05 沙箱里的 status 全 ✗，运维据此以为全停了，接着跑 stop / start。
    # 判不了的时候不许出 ✗，也不打任何看起来像读数的东西。
    if (restriction := _shell_restriction()) is not None:
        for label in ("Redis", "Consolidator", "Embedding", "Memory Flash", "Proxy"):
            print(f"  {label + ':':<14}? {_UNDETERMINED_TEXT}")
        print(f"  Reason: {restriction}. Run `bladex status` from a normal terminal.")
        print(f"  (exit code {RC_UNDETERMINED})")
        return RC_UNDETERMINED

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
            # MQ-W31：「在监听」不等于「在服务」。与 `start` 判 degraded 同一个判据
            # （`_embed_fallback_problems`，读日志里的 embed_ipc_fallback_local），不另写一份。
            if (problems := _embed_fallback_problems()):
                print(f"  Embedding:    ⚠ running (PID {epid}) but not serving -- {problems[0]}")
            else:
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
    print(f"  {'Direct edits:':<14}{_flash_refused_text()[1]}")

    health = _http_json(f"{base}/health")
    if health is None:
        print(f"  Proxy:        ✗ not running ({base})")
        return 1
    ready = _http_json(f"{base}/ready") or {}
    print(f"  Proxy:        ✓ running ({base}, ready={ready.get('ready')})")
    for line in _unmanaged_proxy_lines(cfg.port):
        print(line)

    print(f"  Dashboard:    {base}/dashboard")

    if admin:
        index, hub = admin.get("index", {}), admin.get("hub", {})
        print(f"  {'Memory Hub:':<14}{hub.get('turns', '?')} turns")
        print(f"  {'Memory Index:':<14}{index.get('facts', '?')} facts"
              f" / {index.get('matters', '?')} matters")
        print(f"  {'Queue:':<14}{_format_queue(admin, cons_pid is not None)}")
        print(f"  {'Queue trend:':<14}{_format_backlog_trend(_scrape_metrics(f'{base}/metrics'))}")
        if (cd_line := _contradiction_status_line(admin)):  # 批 BB · 2b：关着时 None ⇒ 这一行不出现
            print(cd_line)
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
    # 批 W33：没停成就不往下走。旧进程还在时再拉一套，正是脱管要防的那种并存。
    if (stop_rc := stop()):
        print("  Not starting: stop did not complete.")
        return stop_rc
    # 直调需给全参数：typer 命令函数的缺省值是 OptionInfo 哨兵，不经 click 不解析
    return start(foreground=foreground, proxy_only=proxy_only, host="", port=0, force=False)


# ── init ────────────────────────────────────────────────────────────────────


# ── doctor ──────────────────────────────────────────────────────────────────


# ── doctor --e2e：记忆闭环端到端自检（ADR-0027 §3.2）────────────────────────


