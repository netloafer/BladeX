"""批 LC · 步骤 3（MQ-W40）：`bladex start` 报 ready 时，`/admin/status` 已经能回答。

事故形态（11-03 / 11-05 / 11-06 三次同形）：`start` 报 ready 之后立刻 `bladex status`，Memory 行
`unavailable`，约 6 s 后恢复。`/ready` 免鉴权、不碰 Hub 索引；`/admin/status` 的 `hub.count()` 要等
Hub 进程内索引首次建成。scratch 实例套 live 规模 Hub 实测：ready 之后 ≈9.6 s 才首答。
修法 (a)：`start` 在 `/ready` 之后再等 `/admin/status` 首次回答，有上限，超时报 degraded 并点名。
"""

from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from bladex_proxy.cli import lifecycle as lc

READY_BODY = json.dumps({"ready": True, "checks": {"redis": True, "hub": True, "index": True}}).encode()


class _Srv:
    def __init__(self, warm_s: float, status_code: int = 200) -> None:
        self.t0 = time.monotonic()
        self.warm_until = self.t0 + warm_s
        self.status_code = status_code
        self.status_answered_at: float | None = None
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *_a) -> None:  # noqa: ANN002
                pass

            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/ready":
                    self._send(200, READY_BODY)
                elif self.path == "/admin/status":
                    if outer.status_code != 200:
                        self._send(outer.status_code, b"{}")
                        return
                    time.sleep(max(0.0, outer.warm_until - time.monotonic()))
                    try:
                        self._send(200, b"{}")
                    finally:
                        outer.status_answered_at = outer.status_answered_at or time.monotonic()
                else:
                    self._send(404, b"{}")

            def _send(self, code: int, body: bytes) -> None:
                try:
                    self.send_response(code)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except OSError:
                    pass  # 客户端已超时断开

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture()
def fast(monkeypatch):  # noqa: ANN001, ANN201
    monkeypatch.setattr(lc, "_ADMIN_STATUS_PROBE_TIMEOUT_S", 0.3)
    return monkeypatch


def _report(srv: _Srv, **kw) -> int:  # noqa: ANN003
    return lc._report_startup_state("127.0.0.1", srv.port, True, True, timeout_s=5,
                                    proxy_pid=kw.pop("proxy_pid", os.getpid()), **kw)


def test_start_waits_until_admin_status_answers(fast, capsys) -> None:  # noqa: ANN001
    srv = _Srv(warm_s=1.2)
    try:
        rc = _report(srv)
        returned_at = time.monotonic()
    finally:
        srv.close()
    out = capsys.readouterr().out
    assert rc == 0 and "state: ready" in out and "degraded" not in out, out
    assert "waiting for /admin/status" in out
    assert srv.status_answered_at is not None and srv.status_answered_at <= returned_at, \
        "start 报 ready 时 /admin/status 还没答——这正是 MQ-W40"


def test_start_reports_degraded_naming_admin_status_when_it_never_answers(fast, capsys) -> None:  # noqa: ANN001
    fast.setattr(lc, "_ADMIN_STATUS_WAIT_S", 0.8)
    srv = _Srv(warm_s=30.0)
    try:
        rc = _report(srv)
    finally:
        srv.close()
    out = capsys.readouterr().out
    assert rc == lc.START_RC_DEGRADED, out
    assert "state: degraded" in out and "/admin/status did not answer within" in out


@pytest.mark.parametrize("code", [401, 404, 503])
def test_http_error_codes_count_as_answered_no_wait(fast, capsys, code: int) -> None:  # noqa: ANN001
    """401 / 404 / 5xx 是配置或别的缺陷，不是「慢」：不该在这里干等，也不该借此报 degraded。"""
    srv = _Srv(warm_s=30.0, status_code=code)
    t0 = time.monotonic()
    try:
        rc = _report(srv)
    finally:
        srv.close()
    assert rc == 0 and time.monotonic() - t0 < 3.0
    assert "waiting for /admin/status" not in capsys.readouterr().out


def test_no_wait_when_this_run_did_not_start_the_proxy(fast, capsys) -> None:  # noqa: ANN001
    """没有 proxy_pid（不是 `start` 拉起的）⇒ 不等：保持原行为。"""
    srv = _Srv(warm_s=30.0)
    t0 = time.monotonic()
    try:
        rc = lc._report_startup_state("127.0.0.1", srv.port, True, True, timeout_s=5)
    finally:
        srv.close()
    assert rc == 0 and time.monotonic() - t0 < 3.0
    assert "waiting for /admin/status" not in capsys.readouterr().out


def test_connection_refused_is_not_treated_as_slow(fast) -> None:  # noqa: ANN001
    """/ready 刚答过，被拒只会是竞态或被替换；不是 Hub 慢，不继续等。"""
    srv = _Srv(warm_s=0)
    port = srv.port
    srv.close()
    t0 = time.monotonic()
    assert lc._await_admin_status(f"http://127.0.0.1:{port}", os.getpid(), cap_s=5.0) is None
    assert time.monotonic() - t0 < 2.0
