"""批 Q3C：`BLADEX_LOG_KEEP` 的默认值只在 `bladex_core.flags` 一处（刚性原则 12）。

既有用例都显式传 `keep=`，默认路径（`keep=None` 读环境 / 默认值）此前没人测。
与 `.env.example` 的对账在 `test_flag_defaults.py::test_env_example_declares_numeric_defaults`。
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from bladex_core import flags


def _logs(tmp_path: Path, n: int = 25) -> Path:
    logs = tmp_path / "logs"
    logs.mkdir()
    now = time.time()
    for i in range(n):
        f = logs / f"proxy-20260930-{i:04d}.log"
        f.write_text("x", encoding="utf-8")
        os.utime(f, (now + i, now + i))
    return logs


@pytest.fixture()
def home(monkeypatch, tmp_path):
    monkeypatch.setenv("BLADEX_HOME", str(tmp_path))
    monkeypatch.delenv("BLADEX_LOG_KEEP", raising=False)
    return tmp_path


def test_default_keeps_twenty(home):
    from bladex_proxy.cli import _prune_logs

    logs = _logs(home)
    assert _prune_logs() == 5
    assert len(list(logs.glob("*.log"))) == 20


def test_default_comes_from_the_flags_table(home, monkeypatch):
    from bladex_proxy.cli import _prune_logs

    monkeypatch.setitem(flags.MEMORY_NUMERIC_DEFAULTS, "BLADEX_LOG_KEEP", 3.0)
    logs = _logs(home)
    assert _prune_logs() == 22
    assert len(list(logs.glob("*.log"))) == 3


@pytest.mark.parametrize("raw", ["abc", ""])
def test_bad_value_falls_back_to_default(home, monkeypatch, raw):
    from bladex_proxy.cli import _prune_logs

    monkeypatch.setenv("BLADEX_LOG_KEEP", raw)
    _logs(home)
    assert _prune_logs() == 5


def test_zero_disables(home, monkeypatch):
    from bladex_proxy.cli import _prune_logs

    monkeypatch.setenv("BLADEX_LOG_KEEP", "0")
    _logs(home)
    assert _prune_logs() == 0
