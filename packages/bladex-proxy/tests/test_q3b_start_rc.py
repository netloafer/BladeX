"""批 Q3B：`bladex start` 降级时退出码仍是 0 ⇒ 脚本无法判断记忆管线死了没有。

取舍（结算稿 §2.4）：ready=0 / failed=1（回收本次拉起的旁挂进程）/ degraded=3（proxy 在跑，
旁挂进程**保留**）。钉住：
  ① `_report_startup_state` 三态三个码，降级那段写明退出码；
  ② `start` 全路径：降级 ⇒ rc=3 且**不回收**旁挂进程；失败 ⇒ rc=1 且回收（阴性对照）。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from bladex_proxy import cli
from bladex_proxy.cli import lifecycle as lc


@pytest.fixture(autouse=True)
def _tmp_cwd(tmp_path: Path, monkeypatch, isolated_home):
    monkeypatch.chdir(tmp_path)
    for k in list(os.environ):
        if k.startswith("BLADEX_"):
            monkeypatch.delenv(k, raising=False)
    yield tmp_path
    for k in list(os.environ):
        if k.startswith("BLADEX_"):
            del os.environ[k]


def _ready(checks: dict):
    return lambda *a, **k: {"ready": all(checks.values()), "checks": checks}


def test_ready_is_0(monkeypatch, capsys) -> None:
    monkeypatch.setattr(lc, "_probe_ready", _ready({"redis": True, "hub": True, "index": True}))
    assert lc._report_startup_state("127.0.0.1", 38080, True, True, timeout_s=1) == 0
    assert "state: ready" in capsys.readouterr().out


def test_degraded_is_3(monkeypatch, capsys) -> None:
    monkeypatch.setattr(lc, "_probe_ready", _ready({"redis": True, "hub": True, "index": True}))
    rc = lc._report_startup_state("127.0.0.1", 38080, True, consolidator_ok=False, timeout_s=1)
    out = capsys.readouterr().out
    assert rc == lc.START_RC_DEGRADED == 3
    assert "state: degraded" in out and "exit code 3" in out


def _start_harness(monkeypatch, tmp: Path, report_rc: int) -> list[str]:
    (tmp / "config").mkdir()
    (tmp / "config" / ".env").write_text("BLADEX_PORT=39998\n", encoding="utf-8")
    calls: list[str] = []
    monkeypatch.setattr(lc, "_preflight_embedding", lambda *a, **k: True)
    for fn in ("_ensure_redis", "_start_embed", "_start_consolidator", "_start_flash"):
        monkeypatch.setattr(lc, fn, lambda *a, **k: True)
    monkeypatch.setattr(lc, "_start_proxy", lambda *a, **k: 0)
    monkeypatch.setattr(lc, "_report_startup_state", lambda *a, **k: report_rc)
    monkeypatch.setattr(lc, "_rollback_sidecars", lambda *a, **k: calls.append("rollback"))
    return calls


def test_start_degraded_returns_3_and_keeps_sidecars(_tmp_cwd: Path, monkeypatch) -> None:
    calls = _start_harness(monkeypatch, _tmp_cwd, lc.START_RC_DEGRADED)
    assert cli.main(["start"]) == 3
    assert calls == []


def test_start_failed_returns_1_and_rolls_back(_tmp_cwd: Path, monkeypatch) -> None:
    calls = _start_harness(monkeypatch, _tmp_cwd, lc.START_RC_FAILED)
    assert cli.main(["start"]) == 1
    assert calls == ["rollback"]


def test_start_help_documents_exit_codes(capsys) -> None:
    cli.main(["start", "--help"])
    out = capsys.readouterr().out
    assert "3 = degraded" in out
