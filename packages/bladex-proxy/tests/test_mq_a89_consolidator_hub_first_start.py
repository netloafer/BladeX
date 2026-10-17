"""MQ-A89：空部署首启时 consolidator 不许因 Memory Hub 未创建而死。

复用 MQ-A81 抽出的 Hub 启动等待 helper；不启动真实进程、不触真实库。
时序用 tmp_path 里的真实 RocksDB + 假时钟模拟，main() 接线用假对象截停在打开 Hub。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import structlog
from bladex_proxy import consolidator
from bladex_proxy.storage.memory_hub import MemoryHub
from bladex_proxy.storage.startup import open_hub_with_startup_window


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
    return MemoryHub(path, secondary=True, secondary_dir=str(path) + "_secondary")


def _create_primary(home: Path) -> None:
    hub = MemoryHub(home / "data" / "bladex_hub")
    hub.open()
    hub.close()


def test_empty_home_first_start_waits_until_proxy_creates_hub(tmp_path: Path) -> None:
    """首启时序：consolidator 先打开失败，等 proxy 创建主库后一次成功。"""
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


def test_existing_hub_opens_without_waiting(tmp_path: Path) -> None:
    """已有 Hub 的正常路径不变：一次成功、零等待。"""
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


def test_non_missing_hub_error_is_not_retried() -> None:
    """权限/锁/损坏等其他错误不是首启瞬态；必须立即原样抛出。"""
    class _PermissionHub:
        def open(self) -> None:
            raise PermissionError("IO error: permission denied while opening CURRENT")

    sleeps: list[float] = []
    with pytest.raises(PermissionError, match="permission denied"):
        open_hub_with_startup_window(
            _PermissionHub(), max_wait_s=20.0, sleep=sleeps.append, now=_Clock())
    assert sleeps == []


class _StopMain(Exception):
    pass


class _FakeHub:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.opened_directly = False

    def open(self) -> None:
        self.opened_directly = True
        raise AssertionError("legacy direct hub.open() must not be used")

    def close(self) -> None:
        pass


def test_main_uses_startup_window_when_opening_hub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """接线测试：consolidator.main() 打开 Hub 必须经共享等待 helper。"""
    import bladex_proxy.config as config_module
    import bladex_proxy.embedding as embedding_module
    import bladex_proxy.storage.memory_hub as memory_hub_module

    calls: list[object] = []

    def fail_open(hub: object, **_kwargs: Any) -> float:
        calls.append(hub)
        raise _StopMain

    class _FakeIdentityRegistry:
        empty = True

        def legacy_map(self) -> dict[str, str]:
            return {}

    class _FakeConfig:
        index_path = str(tmp_path / "index")
        rocksdb_path = str(tmp_path / "data" / "bladex_hub")
        identity_registry = _FakeIdentityRegistry()

    monkeypatch.setattr(sys, "argv", ["bladex-consolidator", "--once"])
    monkeypatch.setattr(config_module, "ProxyConfig", lambda: _FakeConfig())
    monkeypatch.setattr(memory_hub_module, "MemoryHub", _FakeHub)
    monkeypatch.setattr(consolidator, "open_hub_with_startup_window", fail_open)
    monkeypatch.setattr(embedding_module, "validate_embed_sensitivity",
                        lambda *_args, **_kwargs: None)
    monkeypatch.setattr(embedding_module, "resolve_embed_settings",
                        lambda *_args, **_kwargs: SimpleNamespace(backend="local"))
    monkeypatch.setattr(embedding_module, "effective_thresholds",
                        lambda *_args, **_kwargs: {"novelty": 0.9,
                                                    "semantic": 0.85,
                                                    "digestion": 0.8})

    with pytest.raises(_StopMain):
        consolidator.main()

    assert len(calls) == 1
    assert isinstance(calls[0], _FakeHub)
