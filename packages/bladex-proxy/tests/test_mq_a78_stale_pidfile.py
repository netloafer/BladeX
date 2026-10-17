"""MQ-A78：pidfile 只有指向“本 BLADEX_HOME 的活进程”才算 running。

start/status 的旧判据只看 PID 活着，pidfile 缺失时还会全机 `pgrep -f`；
两种形态都会把陈旧文件或另一个部署根的进程误报成 already running。
本文件全部用 tmp_path + 假 Popen / monkeypatch，不启动真实 sidecar。
"""

from __future__ import annotations

import os
import signal
import sys
from pathlib import Path
from typing import Any

import pytest
from bladex_proxy.cli import lifecycle
from bladex_proxy.cli.lifecycle import (
    _CONS_PIDFILE,
    _EMBED_PIDFILE,
    _FLASH_PIDFILE,
)

DEAD_PID = 999_999_999
FAKE_PID = 246_80


class _StopMain(Exception):
    """让入口测试在 chdir / transport resolve 处提前收束，不进真实配置。"""


class _FakePopen:
    calls: list[list[str]] = []

    def __init__(self, argv: list[str], **_kwargs: Any) -> None:
        type(self).calls.append(list(argv))
        self.pid = FAKE_PID


@pytest.fixture(autouse=True)
def _isolated_deployment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """每个用例一个部署根；外部 BLADEX_* 不许影响判据。"""
    for key in list(os.environ):
        if key.startswith("BLADEX_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("BLADEX_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return tmp_path.resolve()


@pytest.fixture
def _fake_spawn(monkeypatch: pytest.MonkeyPatch) -> type[_FakePopen]:
    _FakePopen.calls = []
    monkeypatch.setattr(lifecycle.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(lifecycle, "_pid_alive", lambda pid: pid != DEAD_PID)
    monkeypatch.setattr(lifecycle.time, "sleep", lambda _seconds: None)
    return _FakePopen


def _write_pid(path: str, pid: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(str(pid), encoding="utf-8")


@pytest.mark.parametrize(
    ("pidfile", "label", "start"),
    [
        (_FLASH_PIDFILE, "Flash daemon", lambda: lifecycle._start_flash()),
        (_EMBED_PIDFILE, "Embedding", lambda: lifecycle._start_embed(None)),
        (_CONS_PIDFILE, "Consolidator", lambda: lifecycle._start_consolidator(60, 4)),
    ],
)
def test_stale_pidfile_is_cleared_and_sidecar_starts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    _fake_spawn: type[_FakePopen],
    pidfile: str,
    label: str,
    start: Any,
) -> None:
    """陈旧 pidfile 不许挡启动；新进程命令行必须携带部署根身份。"""
    monkeypatch.setattr(lifecycle, "_embed_wanted", lambda _cfg=None: True)
    monkeypatch.setattr(lifecycle, "_embed_listening", lambda: True)
    _write_pid(pidfile, DEAD_PID)

    assert start() is True
    out = capsys.readouterr().out
    assert "already running" not in out
    assert "stale pidfile" in out
    assert "started" in out
    assert Path(pidfile).read_text(encoding="utf-8") == str(FAKE_PID)

    assert len(_fake_spawn.calls) == 1
    argv = _fake_spawn.calls[0]
    assert "--bladex-home" in argv
    assert argv[argv.index("--bladex-home") + 1] == str(tmp_path.resolve())


@pytest.mark.parametrize(
    ("pidfile", "pid_reader"),
    [
        (_FLASH_PIDFILE, lifecycle._flash_pid),
        (_EMBED_PIDFILE, lifecycle._embed_pid),
        (_CONS_PIDFILE, lifecycle._consolidator_pid),
    ],
)
def test_live_pid_from_other_home_is_not_this_instance(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    pidfile: str,
    pid_reader: Any,
) -> None:
    """PID 活着还不够：命令行声明的部署根必须等于当前 BLADEX_HOME。"""
    other_home = tmp_path / "other-home"
    _write_pid(pidfile, FAKE_PID)
    monkeypatch.setattr(lifecycle, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(lifecycle, "_process_home", lambda _pid: str(other_home))

    assert pid_reader() is None
    assert Path(pidfile).read_text(encoding="utf-8") == str(FAKE_PID)


@pytest.mark.parametrize(
    ("pidfile", "pid_reader"),
    [
        (_FLASH_PIDFILE, lifecycle._flash_pid),
        (_EMBED_PIDFILE, lifecycle._embed_pid),
        (_CONS_PIDFILE, lifecycle._consolidator_pid),
    ],
)
def test_live_pid_with_matching_home_is_running(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    pidfile: str,
    pid_reader: Any,
) -> None:
    """归属正确的活进程照常被识别；这条是防“过度清理”的对照。"""
    _write_pid(pidfile, FAKE_PID)
    monkeypatch.setattr(lifecycle, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(lifecycle, "_process_home", lambda _pid: str(tmp_path.resolve()))

    assert pid_reader() == FAKE_PID
    assert Path(pidfile).read_text(encoding="utf-8") == str(FAKE_PID)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    ("pidfile", "pid_reader"),
    [
        (_FLASH_PIDFILE, lifecycle._flash_pid),
        (_EMBED_PIDFILE, lifecycle._embed_pid),
        (_CONS_PIDFILE, lifecycle._consolidator_pid),
    ],
)
def test_missing_pidfile_does_not_search_the_whole_machine(
    monkeypatch: pytest.MonkeyPatch, pidfile: str, pid_reader: Any
) -> None:
    """pidfile 缺失就是本实例没跑；不许再按进程名认领别人的进程。"""

    def _fail(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("must not run pgrep/ps when the pidfile is missing")

    monkeypatch.setattr(lifecycle.subprocess, "run", _fail)
    assert pid_reader() is None
    assert not Path(pidfile).exists()


@pytest.mark.parametrize(
    ("pidfile", "label", "start"),
    [
        (_FLASH_PIDFILE, "Flash daemon", lambda: lifecycle._start_flash()),
        (_EMBED_PIDFILE, "Embedding", lambda: lifecycle._start_embed(None)),
        (_CONS_PIDFILE, "Consolidator", lambda: lifecycle._start_consolidator(60, 4)),
    ],
)
def test_invalid_pidfile_is_cleared_by_start_not_readers(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    _fake_spawn: type[_FakePopen],
    pidfile: str,
    label: str,
    start: Any,
) -> None:
    """空/脏 pidfile 不是 running 证据；读命令不删，start 才清理并重建。"""
    monkeypatch.setattr(lifecycle, "_embed_wanted", lambda _cfg=None: True)
    monkeypatch.setattr(lifecycle, "_embed_listening", lambda: True)
    Path(pidfile).parent.mkdir(parents=True, exist_ok=True)
    Path(pidfile).write_text("not-a-pid", encoding="utf-8")
    monkeypatch.setattr(lifecycle, "_pid_alive", lambda pid: pid != DEAD_PID)

    if pidfile == _EMBED_PIDFILE:
        pid_reader = lifecycle._embed_pid
    elif pidfile == _FLASH_PIDFILE:
        pid_reader = lifecycle._flash_pid
    else:
        pid_reader = lifecycle._consolidator_pid

    assert pid_reader() is None
    assert Path(pidfile).read_text(encoding="utf-8") == "not-a-pid"
    capsys.readouterr()

    assert start() is True
    out = capsys.readouterr().out
    assert "no valid PID" in out
    assert Path(pidfile).read_text(encoding="utf-8") == str(FAKE_PID)


def test_status_and_start_agree_on_stale_pidfile(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    _fake_spawn: type[_FakePopen],
) -> None:
    """同一个陈旧 pidfile：status 先说 not running，start 随后能照常拉起。"""
    _write_pid(_CONS_PIDFILE, DEAD_PID)
    monkeypatch.setattr(lifecycle, "_redis_ping", lambda _url: False)
    monkeypatch.setattr(lifecycle, "_http_json", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(lifecycle, "_admin_key", lambda: "")
    monkeypatch.setattr(lifecycle, "_embed_wanted", lambda _cfg=None: False)
    monkeypatch.setattr(lifecycle, "_flash_status_line", lambda: "Flash test")

    assert lifecycle.status() == 1
    status_out = capsys.readouterr().out
    assert "Consolidator: ✗ not running" in status_out
    assert "stale pidfile" not in status_out
    assert Path(_CONS_PIDFILE).read_text(encoding="utf-8") == str(DEAD_PID)

    assert lifecycle._start_consolidator(60, 4) is True
    start_out = capsys.readouterr().out
    assert "already running" not in start_out
    assert "stale pidfile" in start_out
    assert "started" in start_out
    assert Path(_CONS_PIDFILE).read_text(encoding="utf-8") == str(FAKE_PID)


@pytest.mark.parametrize(
    ("pidfile", "label"),
    [
        (_FLASH_PIDFILE, "Flash daemon"),
        (_EMBED_PIDFILE, "Embedding"),
        (_CONS_PIDFILE, "Consolidator"),
    ],
)
def test_old_version_process_without_home_marker_is_this_instance(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    pidfile: str,
    label: str,
) -> None:
    """老版本进程没有 --bladex-home；本部署 pidfile 就是所有权证据。"""
    _write_pid(pidfile, FAKE_PID)
    monkeypatch.setattr(lifecycle, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(lifecycle, "_process_home", lambda _pid: None)

    if pidfile == _EMBED_PIDFILE:
        pid_reader = lifecycle._embed_pid
    elif pidfile == _FLASH_PIDFILE:
        pid_reader = lifecycle._flash_pid
    else:
        pid_reader = lifecycle._consolidator_pid

    assert pid_reader() == FAKE_PID
    out = capsys.readouterr().out
    assert "was started by an older BladeX version" in out
    assert "treating it as this deployment" in out
    assert Path(pidfile).read_text(encoding="utf-8") == str(FAKE_PID)


def test_old_version_process_status_stop_start_do_not_duplicate_or_lose_it(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    _fake_spawn: type[_FakePopen],
) -> None:
    """生产复核场景：旧 sidecar 仍在跑时，读/停/启都不能丢它或再起一套。"""
    for pidfile in (_CONS_PIDFILE, _FLASH_PIDFILE, _EMBED_PIDFILE):
        _write_pid(pidfile, FAKE_PID)
    monkeypatch.setattr(lifecycle, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(lifecycle, "_process_home", lambda _pid: None)
    monkeypatch.setattr(lifecycle, "_redis_ping", lambda _url: False)
    monkeypatch.setattr(lifecycle, "_http_json", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(lifecycle, "_admin_key", lambda: "")
    monkeypatch.setattr(lifecycle, "_embed_wanted", lambda _cfg=None: False)
    monkeypatch.setattr(lifecycle, "_flash_status_line", lambda: "Flash test")

    # status 是读命令：认得旧进程，而且三个 pidfile 都逐字不变。
    pidfile_bytes = {Path(path): Path(path).read_bytes()
                     for path in (_CONS_PIDFILE, _FLASH_PIDFILE, _EMBED_PIDFILE)}
    assert lifecycle.status() == 1
    status_out = capsys.readouterr().out
    assert "Consolidator: ✓ running (PID 24680)" in status_out
    for path, raw in pidfile_bytes.items():
        assert path.read_bytes() == raw

    # start 不允许再起第二套。
    assert lifecycle._start_consolidator(60, 4) is True
    assert "already running" in capsys.readouterr().out
    assert _fake_spawn.calls == []
    assert Path(_CONS_PIDFILE).read_bytes() == pidfile_bytes[Path(_CONS_PIDFILE)]

    # stop 仍从 pidfile 找得到旧进程；不杀真进程，只验证发给它的信号。
    kills: list[tuple[int, int]] = []
    monkeypatch.setattr(
        lifecycle.os, "kill",
        lambda pid, sig: (sig != signal.SIGTERM or kills.append((pid, int(sig)))))
    lifecycle._stop_by_pidfile(_CONS_PIDFILE, "Consolidator")
    stop_out = capsys.readouterr().out
    assert "Consolidator: stopped (PID 24680)" in stop_out
    assert kills == [(FAKE_PID, int(signal.SIGTERM))]
    assert not Path(_CONS_PIDFILE).exists()


def test_consolidator_main_pins_home_before_env_and_chdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--bladex-home` 先钉住，env 加载和 chdir 才能按新部署根自举。"""
    from bladex_proxy import consolidator

    home = tmp_path / "deployment"
    events: list[str] = []
    seen_env: dict[str, str | None] = {}

    def fail() -> None:
        seen_env["home"] = os.environ.get("BLADEX_HOME")
        events.append("chdir")
        raise _StopMain

    monkeypatch.setattr(consolidator, "_load_env_file", lambda: events.append("env"))
    monkeypatch.setattr(consolidator, "_chdir_to_deployment_root", fail)
    monkeypatch.setattr(
        consolidator.sys, "argv",
        ["consolidator", "--bladex-home", str(home), "--once"],
    )
    with pytest.raises(_StopMain):
        consolidator.main()

    assert events == ["env", "chdir"]
    assert seen_env["home"] == str(home.resolve())


def test_flash_daemon_main_pins_home_before_env_and_chdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """flash 入口同 consolidator：身份先落 env，再做部署根副作用。"""
    import bladex_proxy.consolidator as consolidator
    import bladex_proxy.flash_daemon as flash_daemon

    home = tmp_path / "deployment"
    events: list[str] = []
    seen_env: dict[str, str | None] = {}

    def fail() -> None:
        seen_env["home"] = os.environ.get("BLADEX_HOME")
        events.append("chdir")
        raise _StopMain

    def fail_load_env() -> None:
        events.append("env")

    monkeypatch.setattr(consolidator, "_load_env_file", fail_load_env)
    monkeypatch.setattr(consolidator, "_chdir_to_deployment_root", fail)
    monkeypatch.setattr(
        sys, "argv",
        ["flash-daemon", "--bladex-home", str(home)],
    )
    with pytest.raises(_StopMain):
        flash_daemon.main()

    assert events == ["env", "chdir"]
    assert seen_env["home"] == str(home.resolve())


def test_embed_server_main_pins_home_before_transport_and_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """embed 入口必须先钉 BLADEX_HOME，再解析 transport / 读配置。"""
    from bladex_proxy import embed_server, embed_transport

    home = tmp_path / "deployment"
    events: list[str] = []
    seen_env: dict[str, str | None] = {}

    def fail_resolve(**_kwargs: Any) -> None:
        seen_env["home"] = os.environ.get("BLADEX_HOME")
        events.append("resolve")
        raise _StopMain

    monkeypatch.setattr(embed_transport, "resolve", fail_resolve)
    monkeypatch.setattr(
        sys, "argv",
        ["embed-server", "--bladex-home", str(home)],
    )
    with pytest.raises(_StopMain):
        embed_server.main()

    assert events == ["resolve"]
    assert seen_env["home"] == str(home.resolve())


def test_home_is_extracted_even_when_path_contains_spaces() -> None:
    """macOS 只有命令文本；home 路径带空格时不能按空白拆 argv。"""
    home = "/tmp/BladeX deployment/root"
    command = f"/venv/python -m bladex_proxy.consolidator --bladex-home {home} --interval 60"
    assert lifecycle._home_from_command(command) == os.path.realpath(home)


def test_proc_cmdline_keeps_exact_home_argv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Linux /proc 路径：逐 NUL 读 argv，路径本身可以包含空格。"""
    home = tmp_path / "home with spaces"
    raw = b"/venv/python\0-m bladex_proxy.consolidator\0--bladex-home\0" + os.fsencode(home) + b"\0--interval\060\0"

    class _BytesPath(Path):
        def read_bytes(self) -> bytes:
            return raw

    monkeypatch.setattr(lifecycle.Path, "read_bytes", _BytesPath.read_bytes)
    assert lifecycle._process_home(FAKE_PID) == str(home)
