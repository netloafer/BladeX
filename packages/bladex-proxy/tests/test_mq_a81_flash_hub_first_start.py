"""MQ-A81：空部署第一次 `bladex start` 不许让 Flash 因 Hub 尚未创建而死。

Flash 以 secondary 模式打开 Memory Hub；secondary 不创建主库，而空部署的主库由
proxy lifespan 首次创建。这里不启动真实进程，用 tmp_path 里的真实 RocksDB /
假时钟模拟这一时序，另补 flash_daemon.main() 的接线断言。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest
import structlog
from bladex_proxy import flash_daemon
from bladex_proxy.storage.memory_hub import MemoryHub
from bladex_proxy.storage.startup import (
    hub_missing,
    open_hub_with_startup_window,
)


@pytest.fixture(autouse=True)
def _isolated_deployment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for key in list(os.environ):
        if key.startswith("BLADEX_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("BLADEX_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return tmp_path


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _secondary(home: Path) -> MemoryHub:
    path = home / "data" / "bladex_hub"
    return MemoryHub(path, secondary=True, secondary_dir=str(path) + "_secondary_flash")


def _create_primary(home: Path) -> None:
    hub = MemoryHub(home / "data" / "bladex_hub")
    hub.open()
    hub.close()


def test_empty_home_first_start_waits_until_proxy_creates_hub(tmp_path: Path) -> None:
    """首启时序：Flash 先打开失败，等 proxy 创建主库后成功，不要求第二次 start。"""
    hub = _secondary(tmp_path)
    clock = _Clock()

    def proxy_creates_hub(_seconds: float) -> None:
        _create_primary(tmp_path)
        clock.now = 0.5

    with structlog.testing.capture_logs() as logs:
        waited = open_hub_with_startup_window(
            hub, max_wait_s=20.0, sleep=proxy_creates_hub, now=clock)

    assert waited == 0.5
    assert hub._db is not None
    assert any(log["event"] == "hub_startup_waiting" for log in logs)
    assert any(log["event"] == "hub_startup_opened" for log in logs)
    hub.close()


def test_existing_hub_opens_without_waiting_or_retry_log(tmp_path: Path) -> None:
    """已有 Hub 的正常路径不变：一次成功，且没有任何 waiting 日志。"""
    _create_primary(tmp_path)
    hub = _secondary(tmp_path)
    clock = _Clock()
    sleeps: list[float] = []

    with structlog.testing.capture_logs() as logs:
        waited = open_hub_with_startup_window(
            hub, max_wait_s=20.0, sleep=sleeps.append, now=clock)

    assert waited == 0.0
    assert hub._db is not None
    assert sleeps == []
    assert all(log["event"] != "hub_startup_waiting" for log in logs)
    hub.close()


def test_missing_hub_timeout_raises_after_bounded_window(tmp_path: Path) -> None:
    """真实缺库不会无限等；窗口耗尽仍缺 CURRENT ⇒ 原样失败，不提示再跑一次。"""
    hub = _secondary(tmp_path)
    clock = _Clock()
    sleeps: list[float] = []

    with structlog.testing.capture_logs() as logs:
        with pytest.raises(Exception, match="/CURRENT"):
            open_hub_with_startup_window(
                hub, max_wait_s=20.0,
                sleep=lambda s: sleeps.append(s)
                or setattr(clock, "now", clock.now + s),
                now=clock)

    assert sum(sleeps) == 20.0
    assert any(log["event"] == "hub_startup_missing_after_window"
               for log in logs)
    assert hub._db is None


def test_other_rocksdb_errors_are_not_retried() -> None:
    """权限/锁/损坏等错误不是首启瞬态；必须 fail-fast，不能拖满 20 秒。"""
    class _PermissionHub:
        def open(self) -> None:
            raise PermissionError("IO error: permission denied while opening CURRENT")

    clock = _Clock()
    sleeps: list[float] = []
    with pytest.raises(PermissionError, match="permission denied"):
        open_hub_with_startup_window(
            _PermissionHub(), max_wait_s=20.0, sleep=sleeps.append, now=clock)
    assert sleeps == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("IO error: No such file or directory: /x/bladex_hub/CURRENT: No such file or directory", True),
        ("IO error: permission denied while opening CURRENT", False),
        ("IO error: No such file or directory: /x/other.db/LOCK", False),
    ],
)
def test_hub_missing_recognizes_only_missing_current(text: str, expected: bool) -> None:
    assert hub_missing(Exception(text)) is expected


def test_main_uses_startup_window_when_opening_hub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """接线测试：flash_daemon.main() 打开 Hub 必须经首启等待 helper。"""
    import bladex_proxy.storage.memory_hub as memory_hub

    calls: list[object] = []
    class _StopMain(Exception):
        pass

    class _FakeHub:
        def open(self) -> None:
            raise AssertionError("legacy direct hub.open() must not be used")

    def fail_open(hub: object, **_kwargs: Any) -> float:
        calls.append(hub)
        raise _StopMain

    (tmp_path / "config").mkdir()
    (tmp_path / "config" / ".env").write_text("", encoding="utf-8")
    monkeypatch.setattr(flash_daemon, "open_hub_with_startup_window", fail_open)
    monkeypatch.setattr(memory_hub, "MemoryHub", lambda *_args, **_kwargs: _FakeHub())
    monkeypatch.setattr(
        sys, "argv",
        ["flash-daemon", "--bladex-home", str(tmp_path), "--once"],
    )
    with pytest.raises(_StopMain):
        flash_daemon.main()

    assert len(calls) == 1
    assert isinstance(calls[0], _FakeHub)
