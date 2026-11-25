"""批 W21b · MQ-W30：`bladex start` 超时后不许「旁挂被停、proxy 还活着」，输出与事实一致。

走真实的 `start()` 分支，只替换拉起进程的 helper 和 `/ready` 探测；PID 文件是真文件。
"""

from __future__ import annotations

import os
import subprocess
import sys
import time as _real_time
from pathlib import Path

import pytest
from bladex_proxy import cli
from bladex_proxy.cli import lifecycle as lc


class _VirtualClock:
    """`lifecycle.time` 的替身（MQ-V44）：**只**缩短等待，不改轮询语义。

    这个文件的等待有两类，都是「轮询等某个状态」：等 `/ready` 到期（`timeout_s=2` 的截止线）、
    等被 SIGTERM 的真进程死透。它们依赖的是**状态**，不是「过了 1 秒」。
    `sleep(s)` 真睡一小截（让真进程有机会死透、`_pid_alive` 读到真实状态），其余部分记进偏移量，
    `monotonic()` = 真时钟 + 偏移量 ⇒ 截止线按虚拟时间到期，轮询次数与原先同量级。
    其余属性（`strftime` 等）透传给真 `time`。
    """

    REAL_NAP = 0.02

    def __init__(self) -> None:
        self._skipped = 0.0
        self.sleeps: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        nap = min(seconds, self.REAL_NAP)
        _real_time.sleep(nap)
        self._skipped += seconds - nap

    def monotonic(self) -> float:
        return _real_time.monotonic() + self._skipped

    def __getattr__(self, name: str):  # noqa: ANN204
        return getattr(_real_time, name)


@pytest.fixture
def home(tmp_path: Path, monkeypatch) -> Path:  # noqa: ANN001
    monkeypatch.setattr(lc, "time", _VirtualClock())
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / ".env").write_text("BLADEX_PORT=39997\n", encoding="utf-8")
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "proxy-1.log").write_text("x\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BLADEX_HOME", str(tmp_path))
    monkeypatch.setattr(lc, "_preflight_embedding", lambda *a, **k: True)
    monkeypatch.setattr(lc, "_prune_logs", lambda *a, **k: 0)
    monkeypatch.setattr(lc, "_ensure_redis", lambda *a, **k: True)
    monkeypatch.setattr(lc, "_probe_ready", lambda *a, **k: None)
    return tmp_path


@pytest.fixture
def procs():  # noqa: ANN201
    """真进程：判「停了没」看 PID 真死活，不看 mock 调用记录。"""
    started: list[subprocess.Popen] = []

    def spawn() -> int:
        # 命令行带 BladeX 标记：批 W33 起 stop 要核对身份，读到不是 BladeX 的命令行不发信号
        p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)",  # noqa: S603
                              "bladex_proxy.fake"])
        started.append(p)
        return p.pid

    yield spawn
    for p in started:
        p.kill()
        p.wait()


def _alive(pid: int) -> bool:
    p = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)],  # noqa: S603, S607
                       capture_output=True, text=True, check=False).stdout.strip()
    return bool(p) and not p.startswith("Z")


def _wire(monkeypatch, spawn, *, proxy_dies: bool) -> dict[str, int]:  # noqa: ANN001
    pids: dict[str, int] = {}

    def side(name: str, pidfile: str):  # noqa: ANN202
        def _start(*_a, **_k) -> bool:  # noqa: ANN002, ANN003
            pids[name] = spawn()
            Path(pidfile).write_text(str(pids[name]))
            return True
        return _start

    monkeypatch.setattr(lc, "_start_embed", side("embed", lc._EMBED_PIDFILE))
    monkeypatch.setattr(lc, "_start_consolidator", side("cons", lc._CONS_PIDFILE))
    monkeypatch.setattr(lc, "_start_flash", side("flash", lc._FLASH_PIDFILE))
    monkeypatch.setattr("bladex_proxy.modules.module_enabled", lambda *_a, **_k: True)

    def start_proxy(*_a, **_k) -> int:  # noqa: ANN002, ANN003
        pids["proxy"] = spawn()
        Path(lc._PROXY_PIDFILE).write_text(str(pids["proxy"]))
        return 0

    monkeypatch.setattr(lc, "_start_proxy", start_proxy)

    real = lc._report_startup_state

    def report(host, port, redis_ok, cons_ok, **kw):  # noqa: ANN001, ANN003, ANN202
        if proxy_dies:
            os.kill(pids["proxy"], 9)
        return real(host, port, redis_ok, cons_ok, timeout_s=2, **kw)

    monkeypatch.setattr(lc, "_report_startup_state", report)
    return pids


def test_alive_but_not_ready_stops_nothing_and_says_so(home, procs, monkeypatch, capsys) -> None:  # noqa: ANN001
    pids = _wire(monkeypatch, procs, proxy_dies=False)
    assert cli.main(["start"]) == lc.START_RC_FAILED
    out = capsys.readouterr().out
    for name, pid in pids.items():
        assert _alive(pid), f"{name} 被停了：半起状态"
        assert f"PID {pid}" in out, f"{name} 的 PID 没报"
    assert "did not answer within 2s" in out
    assert "Nothing was stopped" in out
    assert "did not come up" not in out and "stopping the sidecars" not in out
    assert "logs/proxy-1.log" in out and "logs/embed-" in out
    assert "bladex status" in out and "bladex stop" in out


def test_not_ready_report_reads_sidecar_state_now(home, procs, monkeypatch, capsys) -> None:  # noqa: ANN001
    """旁挂已死 ⇒ 报 not running，不复述启动时的 started。"""
    pids = _wire(monkeypatch, procs, proxy_dies=False)
    real = lc._report_startup_state

    def report(*a, **k):  # noqa: ANN002, ANN003, ANN202
        rc = real(*a, **k)
        os.kill(pids["embed"], 9)
        while lc._read_pidfile(lc._EMBED_PIDFILE) is not None:  # 进程死透（含回收僵尸）才算数
            pass
        return rc

    monkeypatch.setattr(lc, "_report_startup_state", report)
    assert cli.main(["start"]) == lc.START_RC_FAILED
    out = capsys.readouterr().out
    line = next(ln for ln in out.splitlines() if "Embedding:" in ln and "log logs/embed-" in ln)
    assert "not running" in line and str(pids["embed"]) not in line


def test_proxy_exited_still_rolls_back_sidecars(home, procs, monkeypatch, capsys) -> None:  # noqa: ANN001
    """MQ-L62 ② 不退步：proxy 真的死了 ⇒ 回收本次拉起的旁挂，输出也这么说。"""
    pids = _wire(monkeypatch, procs, proxy_dies=True)
    assert cli.main(["start"]) == lc.START_RC_FAILED
    out = capsys.readouterr().out
    assert "exited during startup" in out and "stopping the sidecars" in out
    assert "Nothing was stopped" not in out
    assert all(not _alive(pids[n]) for n in ("embed", "cons", "flash"))
    assert not Path(lc._PROXY_PIDFILE).exists()


@pytest.mark.parametrize(("name", "error"), [
    ("proxy", "embed module could not serialize its reply: vectors[0][0] is numpy.float32"),
    ("consolidator", "embed module closed the connection before replying"),
])
def test_ready_with_local_fallback_is_reported_degraded(home, monkeypatch, capsys, name, error) -> None:  # noqa: ANN001
    """/ready 200、进程都在，但 embed 回包全失败：不能打 memory pipeline is up。"""
    monkeypatch.setattr(lc, "_probe_ready", lambda *a, **k: {"checks": {}})
    (home / "logs" / f"{name}-2.log").write_text(
        f"2026-11-03 13:47:04 [warning  ] embed_ipc_fallback_local       address=/x chunk=1 "
        f"done=0 error='{error}' error_type=X\n", encoding="utf-8")
    rc = lc._report_startup_state("127.0.0.1", 39997, True, True, timeout_s=1,
                                  check_embed_fallback=True)
    out = capsys.readouterr().out
    assert rc == lc.START_RC_DEGRADED
    assert "memory pipeline is up" not in out
    assert "embedding module not serving" in out and name in out and error in out


def test_ready_without_fallback_stays_ready(home, monkeypatch, capsys) -> None:  # noqa: ANN001
    monkeypatch.setattr(lc, "_probe_ready", lambda *a, **k: {"checks": {}})
    rc = lc._report_startup_state("127.0.0.1", 39997, True, True, timeout_s=1,
                                  check_embed_fallback=True)
    assert rc == lc.START_RC_READY and "memory pipeline is up" in capsys.readouterr().out
