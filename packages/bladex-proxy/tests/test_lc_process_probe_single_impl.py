"""批 LC · 步骤 2（MQ-W39）：判活只有一个实现。

此前同一句「进程还在不在」有四份（redis_launch / cli / embedding / instance_claim），
其中两份是二值版，`pid <= 0` 会 `kill(0, 0)` 发给自己的进程组、`OverflowError` 不接。
守卫：全仓生产代码（`packages/*/bladex_*` 与 `scripts/`）只许 `process_probe.py` 直接 `os.kill(pid, 0)`。
"""

from __future__ import annotations

import ast
import errno
import os
from pathlib import Path

import pytest
from bladex_proxy import embedding, process_probe
from bladex_proxy.process_probe import PID_ALIVE, PID_DEAD, PID_UNREACHABLE
from bladex_proxy.storage import instance_claim

REPO = Path(__file__).resolve().parents[3]
ALLOWED = "packages/bladex-proxy/bladex_proxy/process_probe.py"


def _probe_sites(source: str) -> list[int]:
    """源码里所有「发信号 0 探活」的调用行号：`os.kill(x, 0)` / `kill(x, 0)`（含 `from os import kill`）。"""
    hits: list[int] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call) or len(node.args) < 2:
            continue
        fn = node.func
        is_kill = (isinstance(fn, ast.Attribute) and fn.attr == "kill") or (
            isinstance(fn, ast.Name) and fn.id == "kill")
        sig = node.args[1]
        if is_kill and isinstance(sig, ast.Constant) and sig.value == 0:
            hits.append(node.lineno)
    return hits


def _prod_files() -> list[Path]:
    out: list[Path] = []
    for root in [*sorted((REPO / "packages").glob("*/bladex_*")), REPO / "scripts"]:
        if not root.is_dir():
            continue
        for p in root.rglob("*.py"):
            if {"tests", "archive", "__pycache__"} & set(p.parts):
                continue
            out.append(p)
    assert len(out) >= 80, f"只扫到 {len(out)} 个文件，守卫自己漂了"
    return out


def test_only_process_probe_calls_kill_zero() -> None:
    offenders = {
        str(p.relative_to(REPO)): lines
        for p in _prod_files()
        if (lines := _probe_sites(p.read_text(encoding="utf-8"))) and str(p.relative_to(REPO)) != ALLOWED
    }
    assert not offenders, f"判活只许走 process_probe.pid_state / pid_maybe_alive：{offenders}"


def test_guard_is_not_vacuous() -> None:
    """守卫自己得读得到那一处，否则它永远绿。"""
    assert _probe_sites((REPO / ALLOWED).read_text(encoding="utf-8")), "process_probe 里没找到 os.kill(pid, 0)"


@pytest.mark.parametrize("snippet", [
    "import os\ndef f(p):\n    os.kill(p, 0)\n",
    "from os import kill\ndef f(p):\n    kill(p, 0)\n",
    "import os\ndef f(p):\n    try:\n        os.kill(p, 0)\n    except ProcessLookupError:\n        return False\n",
])
def test_guard_detects_a_second_implementation(snippet: str) -> None:
    """反向：在别处再写一份，探测函数必须抓到（守卫的变异检验）。"""
    assert _probe_sites(snippet)


def test_signal_sends_are_not_probes() -> None:
    """真发信号（SIGTERM / SIGKILL）不是探活，不在守卫范围内。"""
    assert not _probe_sites("import os, signal\nos.kill(1, signal.SIGTERM)\n")


# ── 三态语义：四处调用方行为一致 ──────────────────────────────────────────────


@pytest.mark.parametrize("pid", [0, -1, -4242])
def test_non_positive_pid_is_dead_and_never_signals(monkeypatch, pid: int) -> None:  # noqa: ANN001
    """`kill(0, 0)` 会发给自己的进程组；脏 pidfile 不能走到那一步。"""
    def _boom(*_a, **_k) -> None:  # noqa: ANN002, ANN003
        raise AssertionError("不该发信号")

    monkeypatch.setattr(os, "kill", _boom)
    assert process_probe.pid_state(pid) == PID_DEAD
    assert process_probe.pid_maybe_alive(pid) is False


def test_overflow_pid_is_dead(monkeypatch) -> None:  # noqa: ANN001
    def _overflow(*_a, **_k) -> None:  # noqa: ANN002, ANN003
        raise OverflowError("signed integer is greater than maximum")

    monkeypatch.setattr(os, "kill", _overflow)
    assert process_probe.pid_state(10**30) == PID_DEAD


def test_only_esrch_is_dead(monkeypatch) -> None:  # noqa: ANN001
    def _raise(exc: Exception):  # noqa: ANN202
        def _k(*_a, **_k2) -> None:  # noqa: ANN002, ANN003
            raise exc
        return _k

    monkeypatch.setattr(os, "kill", _raise(ProcessLookupError(errno.ESRCH, "no such process")))
    assert process_probe.pid_state(99999) == PID_DEAD
    monkeypatch.setattr(os, "kill", _raise(PermissionError(errno.EPERM, "denied")))
    assert process_probe.pid_state(99999) == PID_UNREACHABLE
    monkeypatch.setattr(os, "kill", _raise(OSError(errno.EINVAL, "weird")))
    assert process_probe.pid_state(99999) == PID_UNREACHABLE
    monkeypatch.setattr(os, "kill", lambda *_a: None)
    assert process_probe.pid_state(99999) == PID_ALIVE


def test_all_callers_route_through_the_single_implementation() -> None:
    """embedding / instance_claim 不再有自己的 `_pid_alive`，调用的就是 process_probe 那一份。"""
    assert not hasattr(embedding, "_pid_alive")
    assert not hasattr(instance_claim, "_pid_alive")
    assert embedding.pid_maybe_alive is process_probe.pid_maybe_alive
    assert instance_claim.pid_maybe_alive is process_probe.pid_maybe_alive
    from bladex_proxy import cli
    from bladex_proxy.storage import redis_launch
    assert redis_launch.pid_state is process_probe.pid_state
    assert redis_launch.listening_pids is process_probe.listening_pids
    assert cli._is_denied is process_probe.is_denied
    assert cli._tcp_state is process_probe.tcp_state


def test_eperm_still_counts_as_alive_for_embedding_download_state(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    """沙箱里碰不到的 PID 不能被清成「已死」：`downloading` 标记照样算数。"""
    cache = str(tmp_path)
    embedding._write_download_state(cache, state="downloading", model="m")
    monkeypatch.setattr(os, "kill", lambda *_a: (_ for _ in ()).throw(PermissionError(errno.EPERM, "x")))
    got = embedding.read_download_state(cache, os.getpid())
    assert got is not None and got["state"] == "downloading"
    monkeypatch.setattr(os, "kill", lambda *_a: (_ for _ in ()).throw(ProcessLookupError(errno.ESRCH, "x")))
    assert embedding.read_download_state(cache, os.getpid()) is None
