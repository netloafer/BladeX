"""批 W33：进程生命周期加固（MQ-W33 / W34 / W35 / W36 / W37 / V49）。

11-05 事故：Codex 沙箱里 `bladex stop` 把 EPERM 判成进程已死并删 pidfile，live 进程脱管。
排查结算稿 `docs/planning/report-task-w33-20261105.md`。

这里只注入 PermissionError，不依赖真沙箱；真沙箱读数在 `docs/benchmarks/w33/*-after.txt`。
进程、pidfile、flock、socket 都是真的，只有发给目标 PID 的 `kill` 被替换。
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from bladex_proxy import cli, embed_server
from bladex_proxy import process_probe as pp
from bladex_proxy.cli import doctor_cmd as dc
from bladex_proxy.cli import lifecycle as lc
from bladex_proxy.storage import redis_launch as rl
from bladex_proxy.storage.redis_launch import RedisOwner

REPO = Path(__file__).resolve().parents[3]
PIDFILES = (lc._PROXY_PIDFILE, lc._CONS_PIDFILE, lc._FLASH_PIDFILE, lc._EMBED_PIDFILE)
SLEEPER = "import time; time.sleep(120)"
#: 夹具收尾用：用例里的 `os.kill` 被替换成 EPERM，收尾时还没恢复
_REAL_KILL = os.kill


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_home: Path) -> Path:
    """一个真的部署根；端口全部隔离，不碰 live。"""
    root = tmp_path / "deploy"
    (root / "config").mkdir(parents=True)
    (root / "logs").mkdir()
    (root / "config" / ".env").write_text(
        f"BLADEX_PORT={_free_port()}\nBLADEX_REDIS_URL=redis://127.0.0.1:{_free_port()}/0\n",
        encoding="utf-8")
    monkeypatch.setenv("BLADEX_HOME", str(root))
    monkeypatch.chdir(root)
    return root.resolve()


@pytest.fixture
def daemons(home: Path):  # noqa: ANN201
    """真进程，命令行与 `bladex start` 拉起的一样带部署根标记。"""
    started: list[subprocess.Popen] = []

    def spawn(pidfile: str | None = None) -> int:
        p = subprocess.Popen([sys.executable, "-c", SLEEPER,  # noqa: S603
                              "--bladex-home", str(home)])
        started.append(p)
        if pidfile:
            Path(pidfile).write_text(str(p.pid), encoding="utf-8")
        return p.pid

    yield spawn
    for p in started:
        if p.poll() is None:
            _REAL_KILL(p.pid, signal.SIGKILL)
        p.wait()


def _exists(pid: int) -> bool:
    out = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)],  # noqa: S603, S607
                         capture_output=True, text=True, check=False).stdout.strip()
    return bool(out) and not out.startswith("Z")


def _deny_signals(monkeypatch: pytest.MonkeyPatch, pids: set[int]) -> list[tuple[int, int]]:
    """沙箱里的 `kill`：对这些 PID 一律 EPERM。记下每一次调用，好断言没有信号发出去。"""
    real = os.kill
    calls: list[tuple[int, int]] = []

    def _kill(pid: int, sig: int) -> None:
        calls.append((pid, int(sig)))
        if pid in pids:
            raise PermissionError(1, "Operation not permitted")
        real(pid, sig)

    monkeypatch.setattr(os, "kill", _kill)
    return calls


def _deny_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """沙箱里的网络：连接本机端口一律 EPERM。"""
    def _connect(self, _addr):  # noqa: ANN001, ANN202
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(socket.socket, "connect", _connect)
    monkeypatch.setattr(socket.socket, "connect_ex", lambda self, _addr: 1)


# ── #1 判活三态 ──────────────────────────────────────────────────────────────


def test_eperm_means_unreachable_not_dead(monkeypatch: pytest.MonkeyPatch, daemons) -> None:  # noqa: ANN001
    pid = daemons()
    _deny_signals(monkeypatch, {pid})
    assert rl.pid_state(pid) == rl.PID_UNREACHABLE
    assert cli._pid_alive(pid) is True
    assert cli._pid_state(pid) == rl.PID_UNREACHABLE


def test_only_esrch_means_dead() -> None:
    p = subprocess.Popen([sys.executable, "-c", "pass"])  # noqa: S603
    p.wait()
    assert rl.pid_state(p.pid) == rl.PID_DEAD
    assert cli._pid_alive(p.pid) is False
    assert rl.pid_state(os.getpid()) == rl.PID_ALIVE


@pytest.mark.parametrize("pid", [0, -1])
def test_nonpositive_pid_is_dead_and_never_signals_our_group(pid: int, monkeypatch) -> None:  # noqa: ANN001
    """`kill(0, sig)` 发给自己的进程组：脏 pidfile 不能走到那一步。"""
    calls = _deny_signals(monkeypatch, set())
    assert rl.pid_state(pid) == rl.PID_DEAD
    assert calls == []


@pytest.mark.parametrize("content", ["0", "-1", "garbage", ""])
def test_stop_on_junk_pidfile_signals_nothing(home: Path, content: str, monkeypatch) -> None:  # noqa: ANN001
    Path(lc._PROXY_PIDFILE).write_text(content, encoding="utf-8")
    calls = _deny_signals(monkeypatch, set())
    assert lc._stop_by_pidfile(lc._PROXY_PIDFILE, "Proxy") is True
    assert [c for c in calls if c[1] != 0] == []
    assert not Path(lc._PROXY_PIDFILE).exists()


def test_stale_pidfile_of_dead_process_is_still_removed(home: Path, capsys) -> None:  # noqa: ANN001
    """阴性对照：真死了照删，不是什么都留着。"""
    p = subprocess.Popen([sys.executable, "-c", "pass"])  # noqa: S603
    p.wait()
    Path(lc._PROXY_PIDFILE).write_text(str(p.pid), encoding="utf-8")
    assert lc._stop_by_pidfile(lc._PROXY_PIDFILE, "Proxy") is True
    assert "not running" in capsys.readouterr().out
    assert not Path(lc._PROXY_PIDFILE).exists()


def test_w33_repro_stop_in_sandbox_keeps_pidfiles(home: Path, daemons, monkeypatch, capsys) -> None:  # noqa: ANN001
    """🔴 排查 §3.1 步骤 3：修前四行 not running、退出码 0、pidfile 全没了、进程全活着。"""
    pids = {pf: daemons(pf) for pf in PIDFILES}
    calls = _deny_signals(monkeypatch, set(pids.values()))
    _deny_network(monkeypatch)
    rc = cli.main(["stop"])
    out = capsys.readouterr().out
    assert rc == cli.RC_UNDETERMINED
    assert "cannot determine" in out and "not running" not in out
    for pf, pid in pids.items():
        assert Path(pf).read_text(encoding="utf-8") == str(pid)
    assert [c for c in calls if c[1] != 0] == [], "受限环境不许发任何真信号"
    monkeypatch.undo()
    assert all(_exists(pid) for pid in pids.values())


def test_w33_repro_stop_signal_denied_only(home: Path, daemons, monkeypatch, capsys) -> None:  # noqa: ANN001
    """网络没被拦、只有信号被拦（别的用户的进程）：同样不删 pidfile。"""
    pid = daemons(lc._CONS_PIDFILE)
    _deny_signals(monkeypatch, {pid})
    rc = cli.main(["stop"])
    assert rc == cli.RC_UNDETERMINED
    assert Path(lc._CONS_PIDFILE).read_text(encoding="utf-8") == str(pid)
    assert "cannot determine" in capsys.readouterr().out


def test_stop_helper_keeps_pidfile_when_sigterm_is_denied(home: Path, daemons, monkeypatch, capsys) -> None:  # noqa: ANN001
    """跳过前置判定时的第二道：kill(0) 过了、SIGTERM 被拒 ⇒ 不崩、不删、返回 False。"""
    pid = daemons(lc._CONS_PIDFILE)
    real = os.kill

    def _kill(p: int, sig: int) -> None:
        if p == pid and sig != 0:
            raise PermissionError(1, "Operation not permitted")
        real(p, sig)

    monkeypatch.setattr(os, "kill", _kill)
    assert lc._stop_by_pidfile(lc._CONS_PIDFILE, "Consolidator") is False
    assert Path(lc._CONS_PIDFILE).exists()
    assert "cannot determine" in capsys.readouterr().out


def test_negative_control_stop_outside_sandbox_really_stops(home: Path, daemons, capsys) -> None:  # noqa: ANN001
    """排查第一跑的阴性对照：沙箱外照常停掉。"""
    pids = {pf: daemons(pf) for pf in PIDFILES}
    assert cli.main(["stop"]) == 0
    out = capsys.readouterr().out
    assert out.count("stopped (PID") == 4
    assert not any(Path(pf).exists() for pf in PIDFILES)
    assert not any(_exists(pid) for pid in pids.values())


def test_reused_pid_is_not_signalled(home: Path, monkeypatch, capsys) -> None:  # noqa: ANN001
    """系统重启后 PID 被别的程序复用：不杀，文件记的那个进程已不存在，pidfile 清掉。"""
    p = subprocess.Popen(["sleep", "120"])  # noqa: S603, S607
    try:
        Path(lc._PROXY_PIDFILE).write_text(str(p.pid), encoding="utf-8")
        assert lc._stop_by_pidfile(lc._PROXY_PIDFILE, "Proxy") is True
        assert p.poll() is None and _exists(p.pid), "误杀了不相干的进程"
        assert not Path(lc._PROXY_PIDFILE).exists()
        assert "another program" in capsys.readouterr().out
    finally:
        p.kill()
        p.wait()


def test_reused_pid_owned_by_someone_else_does_not_block_the_cli(home: Path, monkeypatch) -> None:  # noqa: ANN001
    """PID 被 root 的程序复用时 kill 也是 EPERM：能读到命令行且不是 BladeX ⇒ 不算受限。

    否则一个陈旧 pidfile 就把 start / stop 永远挡住。
    """
    Path(lc._CONS_PIDFILE).write_text("1", encoding="utf-8")  # launchd / init：真的 EPERM
    assert rl.pid_state(1) == rl.PID_UNREACHABLE
    assert lc._shell_restriction() is None


def test_unreadable_command_is_not_signalled(home: Path, daemons, monkeypatch, capsys) -> None:  # noqa: ANN001
    pid = daemons(lc._EMBED_PIDFILE)
    monkeypatch.setattr(lc, "_process_home", lambda _p: None)
    monkeypatch.setattr(lc, "_process_command", lambda _p: None)
    assert lc._stop_by_pidfile(lc._EMBED_PIDFILE, "Embedding") is False
    assert _exists(pid)
    assert Path(lc._EMBED_PIDFILE).exists()


def test_other_deployment_process_is_not_signalled(home: Path, tmp_path: Path) -> None:
    p = subprocess.Popen([sys.executable, "-c", SLEEPER,  # noqa: S603
                          "--bladex-home", str(tmp_path / "elsewhere")])
    try:
        Path(lc._CONS_PIDFILE).write_text(str(p.pid), encoding="utf-8")
        assert lc._stop_by_pidfile(lc._CONS_PIDFILE, "Consolidator") is True
        assert p.poll() is None
    finally:
        p.kill()
        p.wait()


def test_facade_consolidator_pid_keeps_pidfile_on_eperm(home: Path, daemons, monkeypatch) -> None:  # noqa: ANN001
    """`restore` / `import` 的前置检查：EPERM 时算在跑（挡住危险操作），也不删文件。"""
    pid = daemons(lc._CONS_PIDFILE)
    _deny_signals(monkeypatch, {pid})
    monkeypatch.setattr(lc, "_process_home", lambda _p: None)
    monkeypatch.setattr(lc, "_process_command", lambda _p: None)
    assert cli._consolidator_pid() == pid
    assert Path(lc._CONS_PIDFILE).read_text(encoding="utf-8") == str(pid)


def test_discard_path_keeps_pidfile_on_eperm(home: Path, daemons, monkeypatch) -> None:  # noqa: ANN001
    pid = daemons(lc._FLASH_PIDFILE)
    _deny_signals(monkeypatch, {pid})
    monkeypatch.setattr(lc, "_process_home", lambda _p: None)
    monkeypatch.setattr(lc, "_process_command", lambda _p: None)
    assert lc._owned_sidecar_pid(lc._FLASH_PIDFILE, "Flash daemon", cleanup=True) == pid
    assert Path(lc._FLASH_PIDFILE).exists()


def test_restart_does_not_start_when_stop_failed(home: Path, monkeypatch, capsys) -> None:  # noqa: ANN001
    started: list[str] = []
    monkeypatch.setattr(lc, "stop", lambda: 1)
    monkeypatch.setattr(lc, "start", lambda **_k: started.append("start") or 0)
    assert lc.restart(foreground=False, proxy_only=False) == 1
    assert started == []


# ── #3 沙箱感知 ──────────────────────────────────────────────────────────────


def test_w33_repro_status_in_sandbox_is_undetermined(home: Path, daemons, monkeypatch, capsys) -> None:  # noqa: ANN001
    """🔴 排查 §3.1 步骤 2：修前全 ✗。不许出 ✗，用单独退出码。"""
    pid = daemons(lc._CONS_PIDFILE)
    _deny_signals(monkeypatch, {pid})
    _deny_network(monkeypatch)
    rc = cli.main(["status"])
    out = capsys.readouterr().out
    assert rc == cli.RC_UNDETERMINED == 4
    assert "✗" not in out and "✓" not in out
    assert "cannot determine" in out
    assert "sandbox or permissions" in out


def test_start_in_sandbox_launches_nothing(home: Path, monkeypatch, capsys) -> None:  # noqa: ANN001
    launched: list[str] = []
    for fn in ("_ensure_redis", "_start_embed", "_start_consolidator", "_start_flash",
               "_start_proxy", "_prune_logs"):
        monkeypatch.setattr(lc, fn, lambda *a, _n=fn, **k: launched.append(_n) or True)
    _deny_network(monkeypatch)
    assert cli.main(["start"]) == cli.RC_UNDETERMINED
    assert launched == []
    assert "cannot determine" in capsys.readouterr().out


def test_bind_denied_alone_refuses_start(home: Path, monkeypatch) -> None:  # noqa: ANN001
    """11-05 沙箱也拦 bind。只拦 bind 时 connect 看不出来，start 要单独试一次。"""
    real = socket.socket.bind

    def _bind(self, addr):  # noqa: ANN001, ANN202
        if isinstance(addr, tuple):
            raise PermissionError(1, "Operation not permitted")
        return real(self, addr)

    monkeypatch.setattr(socket.socket, "bind", _bind)
    assert lc._shell_restriction() is None
    assert "binding" in (lc._shell_restriction(need_bind=True) or "")


def test_redis_ping_reports_denial_instead_of_false(monkeypatch) -> None:  # noqa: ANN001
    import redis

    class _Client:
        def ping(self) -> bool:
            raise redis.ConnectionError("Error 1 connecting to 127.0.0.1:6379. Operation not permitted.")

    monkeypatch.setattr(redis.Redis, "from_url", lambda *a, **k: _Client())
    with pytest.raises(cli.ShellRestricted):
        lc._redis_ping("redis://127.0.0.1:6379/0")


def test_redis_ping_refused_is_still_false() -> None:
    """阴性对照：真连不上不是受限。"""
    assert lc._redis_ping(f"redis://127.0.0.1:{_free_port()}/0") is False


def test_http_json_reports_denial_instead_of_none(monkeypatch) -> None:  # noqa: ANN001
    _deny_network(monkeypatch)
    with pytest.raises(cli.ShellRestricted):
        cli._http_json("http://127.0.0.1:9/health")


def test_http_json_refused_is_still_none() -> None:
    assert cli._http_json(f"http://127.0.0.1:{_free_port()}/health") is None


def test_main_turns_late_denial_into_undetermined(home: Path, monkeypatch, capsys) -> None:  # noqa: ANN001
    """前置判定之后才被拦（`--traffic`、e2e 等）：不打 traceback，也是单独退出码。"""
    def _raise(*_a, **_k) -> int:  # noqa: ANN002, ANN003
        raise cli.ShellRestricted("reaching x was denied")

    monkeypatch.setattr(lc, "_shell_restriction", _raise)
    assert cli.main(["status"]) == cli.RC_UNDETERMINED
    assert "cannot determine" in capsys.readouterr().out


def test_probe_port_does_not_call_denied_port_free(monkeypatch) -> None:  # noqa: ANN001
    """11-05 沙箱里打的是 `✓ Port (38080 free)`。"""
    _deny_network(monkeypatch)
    state, detail = dc._probe_port("127.0.0.1", 39123)
    assert state == "undetermined" and "free" not in detail


# ── #2 start 不被旧 proxy 骗 ──────────────────────────────────────────────────


class _ReadyHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        body = json.dumps({"ready": True, "checks": {"redis": True, "hub": True, "index": True}}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_a) -> None:  # noqa: ANN002
        pass


_OLD_PROXY = textwrap.dedent("""
    import json, sys
    from http.server import BaseHTTPRequestHandler, HTTPServer
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"ready": True, "checks": {}}).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body)
        def log_message(self, *a): pass
    HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
""")


@pytest.fixture
def old_proxy(home: Path):  # noqa: ANN201
    """另一个进程占着配置的端口、/ready 答 ready：顶替脱管的旧 proxy。"""
    port = _free_port()
    with (home / "config" / ".env").open("a", encoding="utf-8") as fh:
        fh.write(f"BLADEX_PORT={port}\n")
    p = subprocess.Popen([sys.executable, "-c", _OLD_PROXY, str(port)])  # noqa: S603
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                break
        time.sleep(0.05)
    yield p, port
    p.kill()
    p.wait()


def test_w33_repro_start_refuses_old_proxy_and_names_it(home: Path, old_proxy, monkeypatch, capsys) -> None:  # noqa: ANN001
    """🔴 排查 §3.3：修前 `state: ready`、rc 0，新 proxy 随后 bind 失败，留下写着死 PID 的 pidfile。"""
    p, port = old_proxy
    launched: list[str] = []
    for fn in ("_ensure_redis", "_start_embed", "_start_consolidator", "_start_flash", "_start_proxy"):
        monkeypatch.setattr(lc, fn, lambda *a, _n=fn, **k: launched.append(_n) or True)
    monkeypatch.setattr(lc, "_preflight_embedding", lambda *a, **k: True)
    rc = cli.main(["start", "--proxy-only"])
    out = capsys.readouterr().out
    assert rc == lc.START_RC_FAILED
    assert f"PID {p.pid}" in out and str(port) in out
    assert "answers /ready" in out
    assert "state: ready" not in out
    assert launched == []
    assert not Path(lc._PROXY_PIDFILE).exists()
    assert p.poll() is None


def test_start_refuses_when_listener_does_not_answer_ready(home: Path, monkeypatch, capsys) -> None:  # noqa: ANN001
    """端口被别的程序占：同样不起（排查里「报不清原因」那一行）。"""
    port = _free_port()
    code = ("import socket,time,sys; s=socket.socket(); "
            "s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1); "
            "s.bind(('127.0.0.1',int(sys.argv[1]))); s.listen(); time.sleep(120)")
    p = subprocess.Popen([sys.executable, "-c", code, str(port)])  # noqa: S603
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            with socket.socket() as s:
                if s.connect_ex(("127.0.0.1", port)) == 0:
                    break
            time.sleep(0.05)
        monkeypatch.setattr(lc, "_probe_ready", lambda *a, **k: None)
        refusal = lc._port_occupant_refusal("127.0.0.1", port)
        assert refusal is not None and f"PID {p.pid}" in refusal and "not answer /ready" in refusal
    finally:
        p.kill()
        p.wait()


def test_free_port_is_not_refused() -> None:
    assert lc._port_occupant_refusal("127.0.0.1", _free_port()) is None


def test_own_listener_is_not_a_refusal(home: Path, monkeypatch) -> None:  # noqa: ANN001
    """pidfile 里的 proxy 就是监听者：走后面的 already running，不是占用。"""
    port = _free_port()
    srv = socket.socket()
    srv.bind(("127.0.0.1", port))
    srv.listen()
    try:
        Path(lc._PROXY_PIDFILE).write_text(str(os.getpid()), encoding="utf-8")
        assert lc._port_occupant_refusal("127.0.0.1", port) is None
    finally:
        srv.close()


def test_ready_answered_by_someone_else_is_not_ready(home: Path, old_proxy, daemons, capsys) -> None:  # noqa: ANN001
    """开跑后才被抢占：/ready 答话，但监听者不是新 proxy ⇒ failed，点名占用者。"""
    p, port = old_proxy
    new_proxy = daemons(lc._PROXY_PIDFILE)
    rc = lc._report_startup_state("127.0.0.1", port, True, True, timeout_s=1, proxy_pid=new_proxy)
    out = capsys.readouterr().out
    assert rc == lc.START_RC_FAILED
    assert f"PID {p.pid}" in out and str(new_proxy) in out
    assert "state: ready" not in out


def test_ready_with_dead_new_proxy_is_not_ready(home: Path, old_proxy, capsys) -> None:  # noqa: ANN001
    _p, port = old_proxy
    dead = subprocess.Popen([sys.executable, "-c", "pass"])  # noqa: S603
    dead.wait()
    rc = lc._report_startup_state("127.0.0.1", port, True, True, timeout_s=1, proxy_pid=dead.pid)
    out = capsys.readouterr().out
    assert rc == lc.START_RC_FAILED
    assert "exited during startup" in out


def test_ready_when_new_proxy_is_the_listener(home: Path, monkeypatch, capsys) -> None:  # noqa: ANN001
    """阴性对照：监听者就是它 ⇒ ready。"""
    srv = HTTPServer(("127.0.0.1", 0), _ReadyHandler)
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
    port = srv.server_address[1]
    try:
        rc = lc._report_startup_state("127.0.0.1", port, True, True, timeout_s=3, proxy_pid=os.getpid())
    finally:
        srv.shutdown()
        srv.server_close()
    assert rc == lc.START_RC_READY
    assert "state: ready" in capsys.readouterr().out


def test_without_lsof_says_it_could_not_verify(home: Path, monkeypatch, capsys) -> None:  # noqa: ANN001
    monkeypatch.setattr(lc, "_probe_ready",
                        lambda *a, **k: {"ready": True, "checks": {}})
    monkeypatch.setattr(lc, "listening_pids", lambda _p: None)
    assert lc._report_startup_state("127.0.0.1", 39123, True, True, timeout_s=1,
                                    proxy_pid=os.getpid()) == lc.START_RC_READY
    assert "could not verify" in capsys.readouterr().out


# ── #4 embed / flash 单实例 ──────────────────────────────────────────────────


def _lock_holder_process(name: str, root: Path) -> subprocess.Popen:
    code = ("import sys,time; from bladex_proxy.storage.redis_launch import acquire_instance_lock as a; "
            "l=a(sys.argv[1],sys.argv[2]); print('held',flush=True); time.sleep(120)")
    p = subprocess.Popen([sys.executable, "-c", code, name, str(root)],  # noqa: S603
                         stdout=subprocess.PIPE, text=True, env=_child_env())
    assert p.stdout is not None and p.stdout.readline().strip() == "held"
    return p


def _child_env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("BLADEX_")}
    env["PYTHONPATH"] = f"{REPO}/packages/bladex-core:{REPO}/packages/bladex-proxy"
    env.update(extra)
    return env


def test_lock_second_taker_fails_and_names_holder(tmp_path: Path) -> None:
    first = rl.acquire_instance_lock("flash", tmp_path)
    try:
        with pytest.raises(rl.InstanceLockHeld) as e:
            rl.acquire_instance_lock("flash", tmp_path)
        assert f"PID {os.getpid()}" in str(e.value)
        # 抢输的一方不能清掉持有者的记录
        assert rl.read_instance_lock_record("flash", tmp_path)["pid"] == os.getpid()
    finally:
        first.close()


def test_lock_is_released_with_the_process_no_stale_lock(tmp_path: Path) -> None:
    p = _lock_holder_process("embed", tmp_path)
    assert rl.instance_lock_holder("embed", tmp_path)["pid"] == p.pid
    p.send_signal(signal.SIGKILL)  # 连 kill -9 也不留残留
    p.wait()
    assert rl.instance_lock_holder("embed", tmp_path) is None
    rl.acquire_instance_lock("embed", tmp_path).close()


def test_lock_lives_beside_the_pidfiles_and_survives_log_pruning(home: Path) -> None:
    """`logs/` 被 git 忽略，live 工作树不会多出未跟踪文件；`_prune_logs` 只清 `*.log`。"""
    fh = rl.acquire_instance_lock("flash", home)
    try:
        assert rl.instance_lock_path("flash", home) == home / "logs" / "flash.lock"
        lc._prune_logs(keep=1)
        assert (home / "logs" / "flash.lock").exists()
    finally:
        fh.close()


def test_lock_per_name_and_per_root(tmp_path: Path) -> None:
    a = rl.acquire_instance_lock("embed", tmp_path / "a")
    b = rl.acquire_instance_lock("flash", tmp_path / "a")
    c = rl.acquire_instance_lock("embed", tmp_path / "b")
    for fh in (a, b, c):
        fh.close()


def test_holder_probe_does_not_create_the_lock_file(tmp_path: Path) -> None:
    assert rl.instance_lock_holder("embed", tmp_path) is None
    assert not rl.instance_lock_path("embed", tmp_path).exists()


@pytest.mark.parametrize(("module", "name", "event", "extra"), [
    ("bladex_proxy.embed_server", "embed", "embed_server_already_running", {}),
    ("bladex_proxy.flash_daemon", "flash", "flash_daemon_already_running",
     {"BLADEX_MODULE_FLASH": "1"}),
])
def test_second_daemon_exits_and_names_holder(tmp_path: Path, module: str, name: str,
                                              event: str, extra: dict[str, str]) -> None:
    """🔴 真入口：同一部署根第二个进程退出并写明持有者，第一个不受影响。"""
    root = tmp_path / "deploy"
    (root / "config").mkdir(parents=True)
    (root / "config" / ".env").write_text("", encoding="utf-8")
    first = _lock_holder_process(name, root)
    try:
        proc = subprocess.run(  # noqa: S603
            [sys.executable, "-m", module, "--bladex-home", str(root)],
            capture_output=True, text=True, timeout=90, cwd=root,
            env=_child_env(HF_HUB_OFFLINE="1", **extra))
        said = proc.stdout + proc.stderr
        assert proc.returncode == 3, said
        assert event in said and f"PID {first.pid}" in said
        assert first.poll() is None
    finally:
        first.kill()
        first.wait()


@pytest.fixture
def short_sock():  # noqa: ANN201
    """AF_UNIX 路径长度有限，pytest 的 tmp_path 常超限。"""
    d = tempfile.mkdtemp(prefix="w33-", dir="/tmp")  # noqa: S108
    yield os.path.join(d, "e.sock")
    shutil.rmtree(d, ignore_errors=True)


def _run_server_once(address: str) -> None:
    class _Svc:
        _slice = 1

        async def serve_forever(self) -> None:
            await asyncio.sleep(3600)

        def stop(self) -> None:
            pass

        async def handle_client(self, _r, _w) -> None:  # noqa: ANN001
            pass

    async def _go() -> None:
        task = asyncio.ensure_future(embed_server.run_server(_Svc(), transport="uds", address=address))
        for _ in range(200):
            await asyncio.sleep(0.01)
            if task.done():
                task.result()
            if os.path.exists(address):
                with socket.socket(socket.AF_UNIX) as s:
                    try:
                        s.connect(address)
                    except OSError:
                        continue
                task.cancel()
                return

    asyncio.run(_go())


def test_embed_does_not_delete_a_socket_in_use(short_sock: str) -> None:
    """🔴 排查 §1 注：沙箱里那次 start 很可能删了 live 的 run/embed.sock。"""
    address = short_sock
    live = socket.socket(socket.AF_UNIX)
    live.bind(address)
    live.listen()
    ino = os.stat(address).st_ino
    try:
        with pytest.raises(embed_server.EmbedSocketInUse):
            _run_server_once(address)
        assert os.stat(address).st_ino == ino
        with socket.socket(socket.AF_UNIX) as c:
            c.connect(address)
    finally:
        live.close()


def test_embed_replaces_a_dead_socket(short_sock: str) -> None:
    """阴性对照：没人服务的残留 socket 照常替换。"""
    address = short_sock
    stale = socket.socket(socket.AF_UNIX)
    stale.bind(address)
    stale.close()
    assert os.path.exists(address)
    assert embed_server._uds_in_use(address) is False
    _run_server_once(address)


def test_embed_keeps_socket_when_connect_is_denied(short_sock: str, monkeypatch) -> None:  # noqa: ANN001
    """判不了就不删：沙箱里连 socket 也可能被拦。"""
    address = short_sock
    Path(address).touch()
    _deny_network(monkeypatch)
    assert embed_server._uds_in_use(address) is True


@pytest.mark.parametrize(("name", "label", "starter"), [
    ("embed", "Embedding", lambda: lc._start_embed(None)),
    ("flash", "Flash daemon", lambda: lc._start_flash()),
])
def test_cli_does_not_launch_when_lock_is_held(home: Path, name: str, label: str, starter,  # noqa: ANN001
                                               monkeypatch, capsys) -> None:
    """pidfile 丢了但进程还在：不拉，不说 started，不留 pidfile，点名持有者。"""
    monkeypatch.setattr(lc, "_embed_wanted", lambda _c=None: True)
    first = _lock_holder_process(name, home)

    def _no_spawn(*_a, **_k):  # noqa: ANN002, ANN003, ANN202
        raise AssertionError("must not spawn")

    monkeypatch.setattr(lc.subprocess, "Popen", _no_spawn)
    try:
        assert starter() is False
        out = capsys.readouterr().out
        assert f"PID {first.pid}" in out and "started (PID" not in out
        assert not Path(f"logs/{name}.pid").exists()
    finally:
        monkeypatch.undo()
        first.kill()
        first.wait()


def test_embed_listening_needs_a_real_listener(home: Path, short_sock: str, monkeypatch) -> None:  # noqa: ANN001
    """残留 socket 文件不算在监听（此前 os.path.exists 就算）。"""
    address = short_sock
    monkeypatch.setenv("BLADEX_EMBED_ADDRESS", address)
    monkeypatch.setenv("BLADEX_EMBED_TRANSPORT", "uds")
    stale = socket.socket(socket.AF_UNIX)
    stale.bind(address)
    stale.close()
    assert lc._embed_listening() is False
    os.unlink(address)
    live = socket.socket(socket.AF_UNIX)
    live.bind(address)
    live.listen()
    try:
        assert lc._embed_listening() is True
    finally:
        live.close()


def test_start_reports_degraded_when_a_sidecar_did_not_start(home: Path, monkeypatch, capsys) -> None:  # noqa: ANN001
    monkeypatch.setattr(lc, "_preflight_embedding", lambda *a, **k: True)
    monkeypatch.setattr(lc, "_ensure_redis", lambda *a, **k: True)
    monkeypatch.setattr(lc, "_start_embed", lambda *a, **k: False)
    monkeypatch.setattr(lc, "_start_consolidator", lambda *a, **k: True)
    monkeypatch.setattr(lc, "_start_proxy", lambda *a, **k: 0)
    monkeypatch.setattr(lc, "_probe_ready", lambda *a, **k: {"ready": True, "checks": {}})
    monkeypatch.setattr("bladex_proxy.modules.module_enabled", lambda *_a, **_k: False)
    assert cli.main(["start"]) == lc.START_RC_DEGRADED
    assert "embedding module not started" in capsys.readouterr().out


# ── #5 doctor 不再建议杀生产 Redis ───────────────────────────────────────────


def test_w33_refusal_with_unreadable_port_never_suggests_kill(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    """🔴 11-05 实际打出来的是 `kill 17932`，17932 正是生产 Redis。"""
    owner = RedisOwner(pid=17932, port=None, source="cwd")
    monkeypatch.setattr(rl, "redis_servers_on_data_dir", lambda _d="data": [owner])
    msg = rl.launch_refusal(6379, tmp_path)
    assert msg is not None and "17932" in msg
    assert "kill" not in msg.lower()
    assert "different port" not in msg
    assert "lsof" in msg


def test_redis_port_comes_from_lsof_first(monkeypatch) -> None:  # noqa: ANN001
    """沙箱里 `ps` 被拦、`lsof` 可用。"""
    def _run(cmd: list[str], timeout: float = 3.0) -> str:
        return "p17932\nf7\nn*:6379\nf8\nn[::1]:6379\n" if cmd[0] == "lsof" else ""

    monkeypatch.setattr(rl.shutil, "which", lambda n: "/usr/sbin/lsof")
    monkeypatch.setattr(pp, "_run_quiet", _run)   # 批 LC：lsof 探测住在 process_probe
    assert rl._redis_port_of(17932) == 6379


def test_listening_ports_real_lsof() -> None:
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen()
    try:
        port = srv.getsockname()[1]
        assert port in rl.listening_ports(os.getpid())
        assert os.getpid() in (rl.listening_pids(port) or [])
    finally:
        srv.close()


def test_listening_pids_none_when_lsof_missing(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(rl.shutil, "which", lambda _n: None)
    assert rl.listening_pids(6379) is None


def _doctor(monkeypatch, capsys, owners: list[RedisOwner]) -> tuple[int, str]:  # noqa: ANN001
    monkeypatch.setattr(dc, "redis_servers_on_data_dir", lambda *a, **k: owners)
    monkeypatch.setattr(dc, "_redis_ping", lambda _u: True)
    monkeypatch.setattr(dc, "_probe_upstream", lambda _c: ("ok", "stub", ""))
    monkeypatch.setattr(dc, "_doctor_embedding", lambda *a, **k: None)
    rc = cli.main(["doctor"])
    return rc, capsys.readouterr().out


def test_w33_doctor_unreadable_port_no_kill_no_cross(home: Path, monkeypatch, capsys) -> None:  # noqa: ANN001
    rc, out = _doctor(monkeypatch, capsys, [RedisOwner(17932, None, "cwd")])
    line = next(ln for ln in out.splitlines() if "Redis servers on data/" in ln)
    assert line.lstrip().startswith("?")
    assert "kill" not in out
    assert "17932" in line and "lsof" in line


def test_doctor_two_servers_one_unreadable_still_no_kill(home: Path, monkeypatch, capsys) -> None:  # noqa: ANN001
    rc, out = _doctor(monkeypatch, capsys, [RedisOwner(1, 6379, "cwd"), RedisOwner(2, None, "cwd")])
    assert "✗ Redis servers on data/" in out
    assert "kill" not in out


def test_doctor_known_stray_keeps_shutdown_hint(home: Path, monkeypatch, capsys) -> None:  # noqa: ANN001
    """阴性对照：真有端口不同的幽灵（MQ-A90）照报，给 redis-cli shutdown。"""
    rc, out = _doctor(monkeypatch, capsys, [RedisOwner(98573, 1, "cwd")])
    assert "✗ Redis servers on data/" in out
    assert "redis-cli -p 1 shutdown" in out


def test_doctor_in_sandbox_shows_no_false_cross_or_tick(home: Path, monkeypatch, capsys) -> None:  # noqa: ANN001
    _deny_network(monkeypatch)
    monkeypatch.setattr(dc, "redis_servers_on_data_dir", lambda *a, **k: [])
    monkeypatch.setattr(dc, "_doctor_embedding", lambda *a, **k: None)
    assert cli.main(["doctor"]) == cli.RC_UNDETERMINED
    out = capsys.readouterr().out
    assert "✗ Redis" not in out and "✓ Redis (" not in out
    assert "✗ Upstream" not in out
    assert "free" not in out


# ── #7 consolidator 启动判定 ─────────────────────────────────────────────────


_FAKE_CONSOLIDATOR = """
import sys, time
print("2026-11-05 13:54:34 [info     ] starting", flush=True)
time.sleep(float(sys.argv[1]))
print("2026-11-05 13:54:37 [error    ] index_locked_by_another_consolidator path=data/bladex_index", flush=True)
sys.exit(1)
"""


def _fake_consolidator_command(tmp_path: Path, script: str, *args: str) -> list[str]:
    f = tmp_path / "fake_consolidator.py"
    f.write_text(script, encoding="utf-8")
    # argparse 的参数（--bladex-home 等）跟在后面，脚本不解析；先放自己的参数
    return [sys.executable, str(f), *args]


def _wire_start(monkeypatch, tmp_path: Path, script: str, *args: str) -> None:  # noqa: ANN001
    monkeypatch.setattr(lc, "_preflight_embedding", lambda *a, **k: True)
    monkeypatch.setattr(lc, "_ensure_redis", lambda *a, **k: True)
    monkeypatch.setattr(lc, "_start_embed", lambda *a, **k: True)
    monkeypatch.setattr(lc, "_start_flash", lambda *a, **k: True)
    monkeypatch.setattr(lc, "_consolidator_command",
                        lambda: _fake_consolidator_command(tmp_path, script, *args))
    monkeypatch.setattr(lc, "_start_proxy", lambda *a, **k: 0)
    monkeypatch.setattr(lc, "_probe_ready", lambda *a, **k: {
        "ready": True, "checks": {"redis": True, "hub": True, "index": True}})


def test_w33_consolidator_exiting_after_the_old_window_is_degraded(home: Path, tmp_path: Path,
                                                                   monkeypatch, capsys) -> None:  # noqa: ANN001
    """🔴 排查 §1 13:54：撞 Index 写锁退出晚于 1.5 s。修前 started + state: ready + rc 0。"""
    _wire_start(monkeypatch, tmp_path, _FAKE_CONSOLIDATOR, "2.5")
    rc = cli.main(["start"])
    out = capsys.readouterr().out
    assert rc == lc.START_RC_DEGRADED, out
    assert "state: ready" not in out and "state: degraded" in out
    assert "index_locked_by_another_consolidator" in out, "要把日志尾部打出来"
    assert not Path(lc._CONS_PIDFILE).exists(), "不留写着死 PID 的 pidfile"


def test_consolidator_exiting_at_once_is_degraded_with_tail(home: Path, tmp_path: Path, monkeypatch, capsys) -> None:  # noqa: ANN001
    _wire_start(monkeypatch, tmp_path, _FAKE_CONSOLIDATOR, "0")
    rc = cli.main(["start"])
    out = capsys.readouterr().out
    assert rc == lc.START_RC_DEGRADED, out
    assert "index_locked_by_another_consolidator" in out


_READY_CONSOLIDATOR = """
import time
time.sleep(2.5)
print("2026-11-05 17:08:21 [info     ] index_consolidator_ready backend=ipc", flush=True)
time.sleep(120)
"""


def test_consolidator_ready_after_the_old_window_is_ready(home: Path, tmp_path: Path, monkeypatch, capsys) -> None:  # noqa: ANN001
    """阴性对照：慢一点但就绪了 ⇒ ready；等到日志就停，不吃满上限。"""
    _wire_start(monkeypatch, tmp_path, _READY_CONSOLIDATOR)
    monkeypatch.setattr(lc, "_CONSOLIDATOR_READY_WAIT_S", 20.0)
    t0 = time.monotonic()
    try:
        rc = cli.main(["start"])
        out = capsys.readouterr().out
        assert rc == lc.START_RC_READY, out
        assert "state: ready" in out and "still initializing" not in out
        assert time.monotonic() - t0 < 15
    finally:
        pid = lc._read_pidfile(lc._CONS_PIDFILE)
        if pid:
            os.kill(pid, signal.SIGKILL)


_SLOW_CONSOLIDATOR = "import time\ntime.sleep(120)\n"


def test_consolidator_alive_past_the_limit_is_not_called_failed(home: Path, tmp_path: Path,
                                                                monkeypatch, capsys) -> None:  # noqa: ANN001
    """首装下模型可以很久：到上限还活着 ⇒ 说还在初始化，不判失败。"""
    _wire_start(monkeypatch, tmp_path, _SLOW_CONSOLIDATOR)
    monkeypatch.setattr(lc, "_CONSOLIDATOR_READY_WAIT_S", 0.5)
    try:
        rc = cli.main(["start"])
        out = capsys.readouterr().out
        assert rc == lc.START_RC_READY, out
        assert "still initializing" in out
    finally:
        pid = lc._read_pidfile(lc._CONS_PIDFILE)
        if pid:
            os.kill(pid, signal.SIGKILL)


def test_consolidator_already_running_is_not_waited_on(home: Path, daemons, monkeypatch, capsys) -> None:  # noqa: ANN001
    """已经在跑的不等：它的启动日志可能早被清理了。"""
    daemons(lc._CONS_PIDFILE)
    monkeypatch.setattr(lc, "_preflight_embedding", lambda *a, **k: True)
    monkeypatch.setattr(lc, "_ensure_redis", lambda *a, **k: True)
    monkeypatch.setattr(lc, "_start_embed", lambda *a, **k: True)
    monkeypatch.setattr(lc, "_start_flash", lambda *a, **k: True)
    monkeypatch.setattr(lc, "_start_proxy", lambda *a, **k: 0)
    monkeypatch.setattr(lc, "_probe_ready", lambda *a, **k: {"ready": True, "checks": {}})
    monkeypatch.setattr(lc, "_await_consolidator_ready",
                        lambda *a, **k: pytest.fail("不该等已经在跑的 consolidator"))
    assert cli.main(["start"]) == lc.START_RC_READY


def test_bladex_consolidator_start_waits_too(home: Path, tmp_path: Path, monkeypatch, capsys) -> None:  # noqa: ANN001
    monkeypatch.setattr(lc, "_consolidator_command",
                        lambda: _fake_consolidator_command(tmp_path, _FAKE_CONSOLIDATOR, "2.5"))
    assert cli.main(["consolidator", "start"]) == 1
    out = capsys.readouterr().out
    assert "failed to start" in out and "index_locked_by_another_consolidator" in out
    assert not Path(lc._CONS_PIDFILE).exists()


# ── #8 测试防护 ──────────────────────────────────────────────────────────────


def _pytest_under(tmpdir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(  # noqa: S603
        [sys.executable, "-m", "pytest", "-q", "--co", "-p", "no:cacheprovider",
         "tests/unit/test_facade_exports.py"],
        capture_output=True, text=True, timeout=120, cwd=REPO,
        env=_child_env(TMPDIR=str(tmpdir)))


def test_w33_tmpdir_under_a_deployment_root_fails_the_run(tmp_path: Path) -> None:
    """🔴 排查 §3.2：TMPDIR 在部署根下，CLI 用例对它跑真实 `stop`，假进程全死。"""
    root = tmp_path / "fake-deploy"
    (root / "config").mkdir(parents=True)
    (root / "config" / ".env").write_text("", encoding="utf-8")
    inner = root / "tmp" / "deep"
    inner.mkdir(parents=True)
    proc = _pytest_under(inner)
    said = proc.stdout + proc.stderr
    assert proc.returncode != 0, said
    assert "部署根" in said and str(root.resolve()) in said and "MQ-V49" in said


def test_tmpdir_outside_any_deployment_root_is_fine(tmp_path: Path) -> None:
    clean = tmp_path / "clean"
    clean.mkdir()
    proc = _pytest_under(clean)
    assert proc.returncode == 0, proc.stdout + proc.stderr
