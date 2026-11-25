"""进程判活 / 端口探测 / 单实例锁 —— 全仓**唯一**的实现处（批 LC，MQ-W39）。

此前同一个「进程还在不在」有四份：`storage/redis_launch.py`（三态）、`cli/__init__.py`（三态 + waitpid）、
`embedding.py` 与 `storage/instance_claim.py` 各一份二值版（pid <= 0 会 `kill(0)` 发给自己的进程组；
`OverflowError` 不接）——刚性原则 12：同一件事多处实现，迟早有一处和别处不一样。

  - **判活三态**：`pid_state`（死 / 活 / 活着但碰不到）；只有 ESRCH 算死。`pid_maybe_alive` 是它的二值版。
  - **端口**：`listening_ports` / `listening_pids`（走 `lsof`，沙箱里 `ps` 被拦而 `lsof` 可用）、
    `tcp_state`（连得上 / 被拒 / 被系统拦）、`is_denied`（EPERM 识别）。
  - **单实例锁**：`acquire_instance_lock` 等（embed / flash 的 `logs/<name>.lock` flock）。

守卫 `tests/test_lc_process_probe_single_impl.py`：全仓只许本模块直接 `os.kill(pid, 0)`。
只用标准库，CLI 与 daemon 两侧都能 import。
"""

from __future__ import annotations

import errno
import json
import os
import re
import shutil
import socket
import subprocess
import time
from pathlib import Path

# ── 进程与端口探测（MQ-W33 / W34 / W37）────────────────────────────────────────


def _run_quiet(cmd: list[str], timeout: float = 3.0) -> str:
    """跑一个只读探测命令；拿不到输出返回空串（探测失败 ≠ 没有进程，调用方按"未知"处理）。"""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace",  # noqa: S603
                              timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    return proc.stdout or ""


PID_DEAD = "dead"
PID_ALIVE = "alive"
#: 活着但碰不到（EPERM）：沙箱，或者别的用户的进程。**不是**死了。
PID_UNREACHABLE = "unreachable"


def pid_state(pid: int, *, reap_child: bool = False) -> str:
    """进程三态。**只有 ESRCH 才算死**；其余一律不下「已死」的结论。全仓判活只有这一个实现。

    🔴 11-05 事故：把所有 OSError 判成死 ⇒ 沙箱里的 EPERM 变成「not running」+ 删 pidfile。
    pid <= 0 按死处理：`kill(0, sig)` 是发给**自己的进程组**，脏 pidfile 不能走到那一步。

    `reap_child=True`（CLI 用，批 Z1 实测 2026-09-23）：`bladex start` 自己 Popen 出来的进程若在启动期
    退出，CLI 作为父进程从不 `wait` ⇒ 它成了**僵尸**，而 `kill(pid, 0)` 对僵尸照样成功。
    是自己的子进程 ⇒ 先 `waitpid(WNOHANG)` 顺手回收并如实返回「死」；不是（ChildProcessError）⇒ 照常判。
    daemon 侧（embed / flash / instance_claim）不是任何人的父进程，不传。
    """
    if pid <= 0:
        return PID_DEAD
    if reap_child:
        try:
            done, _status = os.waitpid(pid, os.WNOHANG)
            if done == pid:
                return PID_DEAD
        except (OSError, OverflowError):
            pass  # 不是自己的子进程
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, OverflowError):
        return PID_DEAD
    except OSError:
        return PID_UNREACHABLE
    return PID_ALIVE


def pid_maybe_alive(pid: int) -> bool:
    """`pid_state` 的二值版：只有确认不存在才返回 False（EPERM 算在）。"""
    return pid_state(pid) != PID_DEAD


def is_denied(exc: BaseException | None) -> bool:
    """这个异常链里有没有权限拒绝（urllib 包在 `.reason`，redis-py 只留在消息文本里）。"""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, PermissionError):
            return True
        if isinstance(exc, OSError) and exc.errno in (errno.EPERM, errno.EACCES):
            return True
        if "Operation not permitted" in str(exc):
            return True
        reason = getattr(exc, "reason", None)
        if isinstance(reason, BaseException) and is_denied(reason):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def tcp_state(host: str, port: int, timeout: float = 1.0) -> str:
    """`open` / `closed` / `denied`。拒绝连接（ECONNREFUSED）与被系统拦（EPERM）是两回事。"""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return "open"
    except OSError as e:
        return "denied" if is_denied(e) else "closed"


def listening_ports(pid: int) -> list[int]:
    """进程在监听的 TCP 端口；读不到返回空表（调用方按「未知」处理，不当成「没有」）。"""
    if shutil.which("lsof") is None:
        return []
    out = _run_quiet(["lsof", "-nP", "-a", "-p", str(pid), "-iTCP", "-sTCP:LISTEN", "-Fn"])
    ports: list[int] = []
    for line in out.splitlines():
        m = re.search(r"^n.*:(\d+)$", line)
        if m and int(m.group(1)) not in ports:
            ports.append(int(m.group(1)))
    return ports


def listening_pids(port: int) -> list[int] | None:
    """监听某端口的进程。None = 没法查（没有 lsof / lsof 跑不了）；[] = 查了，没看到。"""
    if shutil.which("lsof") is None:
        return None
    try:
        proc = subprocess.run(  # noqa: S603
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-Fp"],  # noqa: S607
            capture_output=True, text=True, errors="replace", timeout=3.0, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return sorted({int(ln[1:]) for ln in proc.stdout.splitlines()
                   if ln.startswith("p") and ln[1:].isdigit()})


# ── embed / flash 单实例（MQ-W35）───────────────────────────────────────────────

#: 与 pidfile 同目录：`logs/` 被 git 忽略、`_prune_logs` 只清 `*.log`。
_INSTANCE_LOCK_DIR = "logs"


class InstanceLockHeld(RuntimeError):
    """另一个进程持有这把单实例锁。`holder` 是对方写下的记录（读不到时为空）。"""

    def __init__(self, path: Path, holder: dict) -> None:
        self.path = path
        self.holder = holder
        super().__init__(f"{describe_lock_holder(holder)} already holds {path}")


def instance_lock_path(name: str, root: str | Path) -> Path:
    return Path(root) / _INSTANCE_LOCK_DIR / f"{name}.lock"


def describe_lock_holder(holder: dict | None) -> str:
    if not holder or not holder.get("pid"):
        return "an unknown process"
    return f"PID {holder['pid']} (started {holder.get('started_at', '?')})"


def read_instance_lock_record(name: str, root: str | Path) -> dict | None:
    """锁文件里的持有者记录（只读，不碰锁）。没有 / 读不懂返回 None。"""
    try:
        data = json.loads(instance_lock_path(name, root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def acquire_instance_lock(name: str, root: str | Path):  # noqa: ANN201 —— 返回打开的文件对象
    """取部署根下的单实例锁并写下自己的身份。**调用方要把返回值留到进程结束。**

    锁随进程释放（含 kill -9），没有残留文件误拦这回事：文件还在但没人持锁 = 可以拿。
    `O_CREAT` 不带 `O_TRUNC`：抢不到的一方不能把持有者的记录清掉。

    :raises InstanceLockHeld: 已有持有者。
    :raises OSError: 取不了锁（文件系统不支持等）。**同样要退出**：防护失效不能静默放行。
    """
    import fcntl  # 只在 daemon 入口与探测时用；导入留在这里，import 本模块不依赖它

    path = instance_lock_path(name, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    fh = os.fdopen(fd, "r+", encoding="utf-8")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as e:
        holder = read_instance_lock_record(name, root) or {}
        fh.close()
        raise InstanceLockHeld(path, holder) from e
    except OSError:
        fh.close()
        raise
    fh.seek(0)
    fh.truncate()
    json.dump({"pid": os.getpid(), "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, fh)
    fh.flush()
    return fh


def instance_lock_holder(name: str, root: str | Path) -> dict | None:
    """现在谁持着这把锁。没人持锁 ⇒ None；有人持锁 ⇒ 对方的记录（读不到记录时为 {}）。

    试取共享锁判断，取到立刻放掉。锁文件不存在 ⇒ 没人持锁，也不去创建它。
    """
    import fcntl

    try:
        fd = os.open(instance_lock_path(name, root), os.O_RDONLY)
    except OSError:
        return None
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BlockingIOError:
        return read_instance_lock_record(name, root) or {}
    except OSError:
        return None
    finally:
        os.close(fd)
    return None
