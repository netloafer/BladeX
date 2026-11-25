"""批 LC · 步骤 4（MQ-W38）：`bladex status` 看见脱管 proxy。

W33 之后 status 仍可能给脱管 proxy 打 ✓（pidfile 丢了 / 手动起的 / 旧进程占着端口）。
判据：端口监听者 PID 与 `logs/proxy.pid` 不一致，或 pidfile 不在而端口有人在听 ⇒ 告警并点名；
只告警不杀；只用监听者比对，不按 ps 命令行过滤。
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
from bladex_proxy.cli import _PROXY_PIDFILE
from bladex_proxy.cli import lifecycle as lc

OTHER = 424242


@pytest.fixture(autouse=True)
def _home(tmp_path: Path, monkeypatch) -> Path:  # noqa: ANN001
    for k in list(os.environ):
        if k.startswith("BLADEX_"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("BLADEX_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs").mkdir()
    return tmp_path


def _pidfile(pid: int) -> None:
    Path(_PROXY_PIDFILE).write_text(str(pid), encoding="utf-8")


def test_no_warning_when_listener_is_the_pidfile_proxy(monkeypatch) -> None:  # noqa: ANN001
    _pidfile(os.getpid())
    monkeypatch.setattr(lc, "listening_pids", lambda _p: [os.getpid()])
    assert lc._unmanaged_proxy_lines(38080) == []


def test_warns_when_listener_differs_from_pidfile(monkeypatch) -> None:  # noqa: ANN001
    _pidfile(os.getpid())
    monkeypatch.setattr(lc, "listening_pids", lambda _p: [OTHER])
    monkeypatch.setattr(lc, "_process_command", lambda _p: "uvicorn bladex_proxy.server:app --port 38080")
    (line,) = lc._unmanaged_proxy_lines(38080)
    assert str(OTHER) in line and "uvicorn bladex_proxy.server:app" in line
    assert f"says PID {os.getpid()}" in line and "Nothing was killed" in line


def test_warns_when_pidfile_missing_but_port_is_served(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(lc, "listening_pids", lambda _p: [OTHER])
    monkeypatch.setattr(lc, "_process_command", lambda _p: None)
    (line,) = lc._unmanaged_proxy_lines(38080)
    assert "missing or stale" in line and "command line unreadable" in line


def test_silent_when_lsof_cannot_tell(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(lc, "listening_pids", lambda _p: None)
    assert lc._unmanaged_proxy_lines(38080) == []
    monkeypatch.setattr(lc, "listening_pids", lambda _p: [])
    assert lc._unmanaged_proxy_lines(38080) == []


def test_command_line_without_bladex_home_is_still_reported(monkeypatch) -> None:  # noqa: ANN001
    """不按命令行过滤：没有 --bladex-home 的 uvicorn 命令行照样被点名（过滤法会漏报）。"""
    monkeypatch.setattr(lc, "listening_pids", lambda _p: [OTHER])
    monkeypatch.setattr(lc, "_process_command", lambda _p: "python -m uvicorn bladex_proxy.server:app")
    assert lc._unmanaged_proxy_lines(38080)


def test_real_orphan_listener_is_named_and_not_killed(capsys, monkeypatch) -> None:  # noqa: ANN001
    """真进程：不经 start 起一个监听者占端口，status 的 proxy 行之后出告警，进程仍活着。"""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = subprocess.Popen(  # noqa: S603
        [sys.executable, "-c",
         "import http.server,sys;http.server.HTTPServer(('127.0.0.1',int(sys.argv[1])),"
         "http.server.SimpleHTTPRequestHandler).serve_forever()", str(port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        import time
        for _ in range(50):
            if lc.listening_pids(port):
                break
            time.sleep(0.1)
        lines = lc._unmanaged_proxy_lines(port)
        assert lines and str(srv.pid) in lines[0], lines
        assert srv.poll() is None
    finally:
        srv.kill()
        srv.wait()


def test_status_prints_the_warning_under_the_proxy_line(capsys, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(lc, "_shell_restriction", lambda **_k: None)
    monkeypatch.setattr(lc, "_redis_ping", lambda _u: False)
    monkeypatch.setattr(lc, "_http_json", lambda url, *a, **k: {"ready": True} if url.endswith(("/health", "/ready")) else None)
    monkeypatch.setattr(lc, "_admin_key", lambda: "")
    monkeypatch.setattr(lc, "_embed_wanted", lambda _c=None: False)
    monkeypatch.setattr(lc, "_flash_status_line", lambda: "Flash test")
    monkeypatch.setattr(lc, "listening_pids", lambda _p: [OTHER])
    monkeypatch.setattr(lc, "_process_command", lambda _p: "uvicorn bladex_proxy.server:app")
    lc.status(traffic=False, n=10)
    out = capsys.readouterr().out.splitlines()
    i = next(n for n, ln in enumerate(out) if ln.startswith("  Proxy:        ✓ running"))
    assert "⚠ Proxy:" in out[i + 1] and str(OTHER) in out[i + 1]
