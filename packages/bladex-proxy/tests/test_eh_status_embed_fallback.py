"""批 EH · MQ-W31：`bladex status` 的 Embedding 行与 `start` 用同一个判据。

W21b 场景 A：embed 服务在监听但回包全失败 ⇒ `start` 报 degraded（rc 3），
`status` 只看「在监听」仍打 ✓。判据收在 `lifecycle._embed_fallback_problems()` 一处，两边共用。
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from bladex_proxy.cli import lifecycle as lc
from bladex_proxy.cli import main as cli_main

ERR = "embed module closed the connection before replying"


@pytest.fixture
def home(tmp_path: Path, monkeypatch) -> Path:  # noqa: ANN001
    for k in list(os.environ):
        if k.startswith("BLADEX_"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("BLADEX_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs").mkdir()
    # status 只关心 Embedding 行：把它前面的探测都钉成「ipc 档、进程在、在监听」
    monkeypatch.setattr(lc, "_shell_restriction", lambda: None)
    monkeypatch.setattr(lc, "_redis_ping", lambda *_a, **_k: True)
    monkeypatch.setattr(lc, "_http_json", lambda *_a, **_k: None)
    monkeypatch.setattr(lc, "_consolidator_pid", lambda: None)
    monkeypatch.setattr(lc, "_embed_wanted", lambda *_a, **_k: True)
    monkeypatch.setattr(lc, "_embed_pid", lambda: 4242)
    monkeypatch.setattr(lc, "_embed_listening", lambda: True)
    return tmp_path


def _embed_line(capsys) -> str:  # noqa: ANN001
    out = capsys.readouterr().out
    return next(ln for ln in out.splitlines() if "Embedding:" in ln)


def _fallback_log(home: Path, name: str = "proxy") -> None:
    (home / "logs" / f"{name}-2.log").write_text(
        "2026-11-03 13:47:04 [warning  ] embed_ipc_fallback_local       address=/x chunk=1 "
        f"done=0 error='{ERR}' error_type=X\n", encoding="utf-8")


def test_status_healthy_stays_check(home, capsys) -> None:  # noqa: ANN001
    cli_main(["status"])
    line = _embed_line(capsys)
    assert "✓ running (PID 4242)" in line and "⚠" not in line


@pytest.mark.parametrize("name", ["proxy", "consolidator"])
def test_status_fallback_is_warned_with_reason(home, capsys, name) -> None:  # noqa: ANN001
    _fallback_log(home, name)
    cli_main(["status"])
    line = _embed_line(capsys)
    assert "⚠" in line and "✓" not in line
    assert "not serving" in line and ERR in line


def test_status_and_start_agree_on_the_same_evidence(home, monkeypatch, capsys) -> None:  # noqa: ANN001
    """同一份日志：start 报 degraded、status 打 ⚠；原因文字同源（同一个函数产出）。"""
    _fallback_log(home)
    monkeypatch.setattr(lc, "_probe_ready", lambda *a, **k: {"checks": {}})
    rc = lc._report_startup_state("127.0.0.1", 39997, True, True, timeout_s=1,
                                  check_embed_fallback=True)
    start_out = capsys.readouterr().out
    cli_main(["status"])
    line = _embed_line(capsys)
    (problem,) = lc._embed_fallback_problems()
    assert rc == lc.START_RC_DEGRADED and problem in start_out
    assert problem in line


def test_status_embed_line_uses_shared_predicate_not_a_copy() -> None:
    """结构守卫：status 里不得另写一份日志读取——判据只在 `_embed_fallback_problems` 一处。"""
    src = Path(lc.__file__).read_text(encoding="utf-8")
    assert src.count("_EMBED_FALLBACK_RE.search(") == 1
    assert src.count("_embed_fallback_problems()") >= 2
