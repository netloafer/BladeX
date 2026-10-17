"""本地 Redis 的拉起参数与数据目录诊断 —— **两条拉起路径的单一真相源**。

拉 redis 的地方有两处，长期各写各的：
  - `bladex_proxy.cli.lifecycle._ensure_redis()`（`bladex start`）
  - `bladex_proxy.server.orchestration._try_recover_redis()`（ADR-0017 懒恢复）
同一串参数两份拷贝 = 缺陷（刚性原则 12：同一个参数在两条调用路径上各有一个默认值，
与哪个值更合理无关）。本模块把命令行、日志路径、失败诊断收成一份。

🔴 两条都来自 2026-09-19 事故（Jason 主动重启系统后 redis 起不来）：

1. **daemonize 且不给 logfile ⇒ redis 侧零日志。** 两处原本还各自把子进程输出丢掉
   （`capture_output` / `DEVNULL`），于是 redis 自己那句"打不开 appendonly.aof.97.base.rdb"
   从未落盘，事后只剩一句 "failed to start"，真因从存量证据不可判定。
2. **logfile 必须是绝对路径。** redis 处理到 `--dir data/` 时当场 `chdir`，其后的相对
   `--logfile` 会落进 `data/` 里去。

🔴 第 3 条来自 2026-09-29 事故（MQ-A90，同一形态第二次）：**同一个 data/ 上跑了两个
redis-server。** 批 F3 隔离验证把 `redis://127.0.0.1:1/0` 当"连不上的假 Redis"，懒恢复
照常对 cwd 下的生产 `data/` 拉起了 `--port 1`（macOS 允许非 root 绑低端口），活了一周。
两个实例各自在内存里持有一份 manifest，redis 8.8 关机（SIGTERM）时会把它**重写一次**
（补 incr 的 endoffset）——谁最后退出谁的版本落盘。幽灵停在 seq 100，真实例已到 110，
幽灵晚 0.4 s 退出 ⇒ 重启后 manifest 指向不存在的文件。修法：
  - **拉起前查 data/ 的归属**（`redis_servers_on_data_dir`）：已有 redis-server 以它为工作
    目录 ⇒ 一律不拉第二个（`launch_refusal`），无论端口是否相同。
  - `--pidfile` 落在 data/ 里（此前 daemonize 用默认 `/var/run/redis_<port>.pid`，不可写，
    归属无迹可查）；`--dir` 用绝对路径。
  - 每次拉起在 redis 日志开头写一行来源（`write_launch_record`），下次可按日志判定谁拉的。
  - manifest 已坏时给**一条**可验证的恢复命令（`plan_manifest_repair`，默认只预演）。

本模块只用标准库，两侧（CLI 与 proxy 进程）都能 import。
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

#: AOF 目录名与 manifest 文件名（redis 7+ multi-part AOF 固定命名）
AOF_DIR_NAME = "appendonlydir"
AOF_MANIFEST_NAME = "appendonly.aof.manifest"
#: pidfile 落在数据目录里：它就是"这个目录归谁"的记录（MQ-A90）
REDIS_PIDFILE_NAME = "redis-server.pid"

_AOF_FILE_RE = re.compile(r"^appendonly\.aof\.(\d+)\.(base\.rdb|base\.aof|incr\.aof)$")


def redis_logfile_path(logs_dir: str | Path = "logs") -> str:
    """redis-server 的日志路径，**绝对**（理由见模块 docstring 第 2 条）。"""
    d = Path(logs_dir)
    d.mkdir(parents=True, exist_ok=True)
    return str((d / f"redis-{time.strftime('%Y%m%d-%H%M%S')}.log").resolve())


def redis_server_command(port: int, logfile: str, data_dir: str = "data") -> list[str]:
    """拉起本地 Redis 的完整命令行（AOF + everysec，与 ADR-0009 §4 一致）。

    `--dir` 绝对、`--pidfile` 落进 data/（MQ-A90）：归属判定靠它。
    """
    d = Path(data_dir).resolve()
    return [
        "redis-server",
        "--port", str(port),
        "--appendonly", "yes",
        "--appendfsync", "everysec",
        "--tcp-keepalive", "0",
        "--dir", f"{d}/",
        "--pidfile", str(d / REDIS_PIDFILE_NAME),
        "--daemonize", "yes",
        "--logfile", logfile,
    ]


# ── data/ 归属（MQ-A90）─────────────────────────────────────────────────────


@dataclass(frozen=True)
class RedisOwner:
    """一个以 data/ 为工作目录的 redis-server 进程。"""

    pid: int
    port: int | None
    source: str  # "pidfile" | "cwd"

    def describe(self) -> str:
        return f"PID {self.pid} (port {self.port if self.port is not None else '?'})"


def _run_quiet(cmd: list[str], timeout: float = 3.0) -> str:
    """跑一个只读探测命令；拿不到输出返回空串（探测失败 ≠ 没有进程，调用方按"未知"处理）。"""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace",  # noqa: S603
                              timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    return proc.stdout or ""


def _redis_port_of(pid: int) -> int | None:
    """redis-server 把进程标题改成 `redis-server *:6379`；从那里读端口。"""
    title = _run_quiet(["ps", "-p", str(pid), "-o", "command="]).strip()
    if "redis-server" not in title:
        return None
    m = re.search(r":(\d+)\s*$", title)
    return int(m.group(1)) if m else None


def _is_redis_pid(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass  # 活着但不是我们的：仍按命令名判
    return "redis-server" in _run_quiet(["ps", "-p", str(pid), "-o", "command="])


def _cwd_owners(data_dir: Path) -> list[int]:
    """所有工作目录 = data_dir 的 redis-server（redis 处理 `--dir` 时 chdir 过去）。

    `lsof -Fpn` 逐进程输出 `p<pid>` 后跟 `n<cwd>`。lsof 不在 / 被沙盒拦 ⇒ 返回空，
    由 pidfile 那一路兜。
    """
    if shutil.which("lsof") is None:
        return []
    out = _run_quiet(["lsof", "-a", "-c", "redis-server", "-d", "cwd", "-Fpn"])
    target = str(data_dir)
    pids: list[int] = []
    cur: int | None = None
    for line in out.splitlines():
        if line.startswith("p") and line[1:].isdigit():
            cur = int(line[1:])
        elif line.startswith("n") and cur is not None:
            try:
                same = str(Path(line[1:]).resolve()) == target
            except OSError:
                same = False
            if same:
                pids.append(cur)
    return pids


def redis_servers_on_data_dir(data_dir: str | Path = "data") -> list[RedisOwner]:
    """正在以 data_dir 为数据目录的 redis-server，两路取证合并。

    - pidfile（本模块拉起的实例都写在 data/ 里）；
    - 进程工作目录（覆盖旧版本拉起、没有 pidfile 的实例——09-22 的幽灵就是这种）。
    """
    d = Path(data_dir).resolve()
    owners: dict[int, RedisOwner] = {}
    pidfile = d / REDIS_PIDFILE_NAME
    try:
        pid = int(pidfile.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        pid = 0
    if pid > 0 and _is_redis_pid(pid):
        owners[pid] = RedisOwner(pid=pid, port=_redis_port_of(pid), source="pidfile")
    for pid in _cwd_owners(d):
        if pid not in owners:
            owners[pid] = RedisOwner(pid=pid, port=_redis_port_of(pid), source="cwd")
    return sorted(owners.values(), key=lambda o: o.pid)


def launch_refusal(port: int, data_dir: str | Path = "data") -> str | None:
    """要在 data_dir 上拉 redis 之前问一句：这个目录已经有主人了吗？

    有 ⇒ 返回拒绝原因（可直接打印/记日志，含下一步命令）；没有 ⇒ None。
    **无论端口是否相同都拒绝**：同端口说明它只是暂时答不上（忙 / 沙盒拦了连接），
    再拉一个只会绑端口失败；异端口就是 MQ-A90 的形态——两份 manifest 互相覆盖。
    """
    owners = redis_servers_on_data_dir(data_dir)
    if not owners:
        return None
    d = Path(data_dir).resolve()
    desc = ", ".join(o.describe() for o in owners)
    same = [o for o in owners if o.port == port]
    if same:
        return (f"redis-server {desc} already runs on {d} but did not answer on port {port}; "
                f"not starting a second one on the same directory. "
                f"Check it with: redis-cli -p {port} ping")
    other = owners[0]
    stop = (f"redis-cli -p {other.port} shutdown" if other.port is not None
            else f"kill {other.pid}")
    return (f"redis-server {desc} already uses {d} as its data directory, on a different port "
            f"than the configured {port}. Two servers on one directory overwrite each other's "
            f"AOF manifest (MQ-A90), so BladeX will not start another. If that server is the "
            f"right one, point BLADEX_REDIS_URL at its port; otherwise stop it first: {stop}")


def write_launch_record(logfile: str, source: str, port: int,
                        data_dir: str | Path = "data") -> None:
    """拉起前在 redis 日志开头记一行：谁拉的、为哪个端口、当时目录上已有谁。

    redis 以追加方式打开 logfile，这一行会留在它自己的启动日志之前。09-22 那个幽灵
    的来源是靠翻 docs 才对上的——下次直接看日志第一行。
    """
    owners = redis_servers_on_data_dir(data_dir)
    line = (f"# bladex-launch source={source} launcher_pid={os.getpid()} cwd={Path.cwd()} "
            f"port={port} dir={Path(data_dir).resolve()} "
            f"existing=[{', '.join(o.describe() for o in owners)}] "
            f"at={time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
    try:
        with open(logfile, "a", encoding="utf-8") as fh:
            fh.write(line)
    except OSError:
        pass  # 记录失败不挡拉起；redis 自己的日志仍会写


def aof_manifest_problem(data_dir: str | Path = "data") -> str | None:
    """AOF manifest 与实际文件对账：manifest 列的文件缺一个，redis 就起不来。

    2026-09-19 事故的形态：redis 8.8.0 在关机时落盘了一份**过期**的 manifest ——
    内容指向 seq 97（09-18 16:54 那次 rewrite 之前的状态），而磁盘上只有 seq 98。
    redis-server 加载 AOF 时打不开 97 的文件，启动即退出。

    这一条不依赖 redis 自己的日志，在拉起**之前**就点名是哪个文件不在
    （redis 的报错只说打不开某文件，不说本该有谁、现在有谁）。

    返回 None = 没问题（含"还没有 manifest"这种全新部署）；否则返回可直接打印的一行。
    """
    d = Path(data_dir) / AOF_DIR_NAME
    manifest = d / AOF_MANIFEST_NAME
    if not manifest.is_file():
        return None
    try:
        text = manifest.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return f"{manifest} is unreadable: {e}"
    missing: list[str] = []
    for line in text.splitlines():
        # 行格式：file <name> seq <n> type <b|i|h>；8.x 的 incr 行还带 startoffset/endoffset
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "file":
            fname = parts[1].strip('"')
            if not (d / fname).is_file():
                missing.append(fname)
    if not missing:
        return None
    present = sorted(p.name for p in d.glob("appendonly.aof.*") if p.name != AOF_MANIFEST_NAME)
    detail = (f"{manifest} lists {len(missing)} file(s) that do not exist: "
              f"{', '.join(missing)}")
    if present:
        detail += f" (present instead: {', '.join(present)})"
    return detail


@dataclass
class ManifestRepairPlan:
    """`bladex repair-aof` 的预演结果。ok=False 时 reason 说清为什么不修、该怎么办。"""

    ok: bool
    reason: str
    new_manifest: str = ""
    checks: list[str] = field(default_factory=list)


def _manifest_entries(text: str) -> list[tuple[str, int, str]]:
    """manifest 行 → (文件名, seq, type)。"""
    out: list[tuple[str, int, str]] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 6 and parts[0] == "file" and parts[2] == "seq" and parts[4] == "type":
            try:
                out.append((parts[1].strip('"'), int(parts[3]), parts[5]))
            except ValueError:
                continue
    return out


def plan_manifest_repair(data_dir: str | Path = "data") -> ManifestRepairPlan:
    """manifest 指向不存在的文件时，判断能否**只向前**改写到盘上实际那一组文件。

    刚性原则 15：自动改 manifest 会否决 redis 自己的记录，所以先举证，任何一条不成立
    就拒绝并说原因。举证项：
      1. 没有 redis-server 占着这个目录（占着时改盘上文件会被它关机时再覆盖一次）；
      2. manifest 列的文件**全部**不在（部分缺失是另一种损坏，不猜）；
      3. 盘上恰好一个 base + 至少一个 incr，没有 history / 未知文件；
      4. 盘上的 seq **全都大于** manifest 里的 seq —— 只向前、不回退；
      5. 同版本 `redis-check-aof` 对候选 manifest 判 valid（在临时目录里，原目录不动）。
    """
    d = Path(data_dir).resolve() / AOF_DIR_NAME
    manifest = d / AOF_MANIFEST_NAME
    checks: list[str] = []
    if aof_manifest_problem(data_dir) is None:
        return ManifestRepairPlan(False, "the AOF manifest is consistent (or absent); nothing to repair")

    owners = redis_servers_on_data_dir(data_dir)
    if owners:
        return ManifestRepairPlan(
            False, f"redis-server {', '.join(o.describe() for o in owners)} is using "
                   f"{d.parent}; stop it first -- it rewrites the manifest when it exits")
    checks.append("no redis-server is using the data directory")

    listed = _manifest_entries(manifest.read_text(encoding="utf-8", errors="replace"))
    if not listed:
        return ManifestRepairPlan(False, f"{manifest} has no readable file entries; not guessing")
    still_there = [name for name, _s, _t in listed if (d / name).is_file()]
    if still_there:
        return ManifestRepairPlan(
            False, f"only part of the manifest is missing ({', '.join(still_there)} still exist); "
                   "that is not the stale-manifest shape this command knows how to fix")
    checks.append(f"every file the manifest lists is gone ({len(listed)})")

    bases: list[tuple[int, str]] = []
    incrs: list[tuple[int, str]] = []
    unknown: list[str] = []
    for p in sorted(d.iterdir()):
        if p.name == AOF_MANIFEST_NAME:
            continue
        m = _AOF_FILE_RE.match(p.name)
        if not m:
            unknown.append(p.name)
        elif m.group(2).startswith("base"):
            bases.append((int(m.group(1)), p.name))
        else:
            incrs.append((int(m.group(1)), p.name))
    if unknown:
        return ManifestRepairPlan(False, f"unexpected files in {d}: {', '.join(unknown)}; not guessing")
    if len(bases) != 1 or not incrs:
        return ManifestRepairPlan(
            False, f"expected exactly one base file and at least one incr file on disk, found "
                   f"{len(bases)} base / {len(incrs)} incr; not guessing")
    checks.append(f"on disk: {bases[0][1]} + {', '.join(n for _s, n in sorted(incrs))}")

    listed_base = max((s for _n, s, t in listed if t == "b"), default=-1)
    listed_incr = max((s for _n, s, t in listed if t == "i"), default=-1)
    if bases[0][0] <= listed_base or min(s for s, _n in incrs) <= listed_incr:
        return ManifestRepairPlan(
            False, "the files on disk are not newer than what the manifest lists; "
                   "rewriting would move backwards -- refusing")
    checks.append(f"on-disk seq is newer than the manifest's (base {listed_base} -> {bases[0][0]}, "
                  f"incr {listed_incr} -> {min(s for s, _n in incrs)})")

    new_manifest = f"file {bases[0][1]} seq {bases[0][0]} type b\n" + "".join(
        f"file {n} seq {s} type i\n" for s, n in sorted(incrs))

    checker = shutil.which("redis-check-aof")
    if checker is None:
        return ManifestRepairPlan(False, "redis-check-aof is not installed; cannot verify the "
                                         "candidate manifest, so not writing it")
    with tempfile.TemporaryDirectory(prefix="bladex-aofcheck-") as tmp:
        t = Path(tmp)
        for _s, n in [bases[0], *incrs]:
            (t / n).symlink_to(d / n)
        (t / AOF_MANIFEST_NAME).write_text(new_manifest, encoding="utf-8")
        try:
            proc = subprocess.run([checker, str(t / AOF_MANIFEST_NAME)],  # noqa: S603
                                  capture_output=True, text=True, errors="replace",
                                  timeout=300, check=False)
            said = (proc.stdout or "") + (proc.stderr or "")
            valid = proc.returncode == 0 and "are valid" in said
        except (OSError, subprocess.SubprocessError) as e:
            said, valid = str(e), False
    if not valid:
        tail = " / ".join(ln for ln in said.splitlines() if ln.strip())[-400:]
        return ManifestRepairPlan(False, f"redis-check-aof rejected the candidate manifest: {tail}",
                                  new_manifest, checks)
    checks.append("redis-check-aof: all AOF files and the candidate manifest are valid")
    return ManifestRepairPlan(True, "the manifest can be rewritten forward to the files on disk",
                              new_manifest, checks)


def apply_manifest_repair(plan: ManifestRepairPlan, data_dir: str | Path = "data") -> Path:
    """写入 plan 的候选 manifest；旧 manifest 先复制到 data/ 下留档（不删）。返回留档路径。"""
    if not plan.ok:
        raise ValueError(f"refusing to apply a failed repair plan: {plan.reason}")
    root = Path(data_dir).resolve()
    d = root / AOF_DIR_NAME
    manifest = d / AOF_MANIFEST_NAME
    kept = root / f"{AOF_MANIFEST_NAME}.stale-{time.strftime('%Y%m%d-%H%M%S')}"
    tmp = d / f".{AOF_MANIFEST_NAME}.bladex-new"
    shutil.copy2(manifest, kept)
    tmp.write_text(plan.new_manifest, encoding="utf-8")
    os.replace(tmp, manifest)  # 原子替换：任何时刻目录里都有一份 manifest
    return kept


def tail_lines(path: str | Path, n: int = 8) -> list[str]:
    """文件尾部 n 条非空行；读不到返回空表（诊断辅助，不抛）。"""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return [ln.rstrip() for ln in text.splitlines() if ln.strip()][-n:]
