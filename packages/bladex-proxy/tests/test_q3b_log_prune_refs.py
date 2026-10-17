"""批 Q3B · MQ-V21：`bladex start` 的日志清理删掉了签收稿引用的日志 ⇒ 读数不可复算。

钉住：
  ① `docs/` 下任一 Markdown 引用过的日志永不清理，也不占 BLADEX_LOG_KEEP 名额；
  ② 未被引用的照旧按类各留最近 N 个（阴性对照：引用只保护被点名的那一个）；
  ③ BLADEX_LOG_KEEP=0 仍是全关。
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from bladex_proxy.cli import _prune_logs


def _mk_logs(home: Path, n: int = 25) -> Path:
    logs = home / "logs"
    logs.mkdir()
    now = time.time()
    for i in range(n):
        f = logs / f"proxy-20260901-{i:06d}.log"
        f.write_text("x", encoding="utf-8")
        os.utime(f, (now + i, now + i))
    return logs


def test_referenced_log_survives_and_does_not_take_a_slot(monkeypatch, tmp_path: Path) -> None:
    logs = _mk_logs(tmp_path)
    (tmp_path / "docs" / "planning").mkdir(parents=True)
    # 最旧的两份被签收稿引用（一份带 logs/ 前缀、一份裸名）
    (tmp_path / "docs" / "planning" / "report-x.md").write_text(
        "读数见 `logs/proxy-20260901-000000.log:55-70`，对照 proxy-20260901-000001.log。\n",
        encoding="utf-8")
    monkeypatch.setenv("BLADEX_HOME", str(tmp_path))
    removed = _prune_logs(keep=20)
    assert (logs / "proxy-20260901-000000.log").exists()
    assert (logs / "proxy-20260901-000001.log").exists()
    # 未引用 23 份留最新 20 ⇒ 删 3；总数 = 20 + 2 个被引用的
    assert removed == 3
    assert len(list(logs.glob("*.log"))) == 22
    assert not (logs / "proxy-20260901-000002.log").exists()


def test_unreferenced_logs_pruned_as_before(monkeypatch, tmp_path: Path) -> None:
    logs = _mk_logs(tmp_path)
    (tmp_path / "docs" / "reviews").mkdir(parents=True)
    (tmp_path / "docs" / "reviews" / "r.md").write_text("no log references here\n", encoding="utf-8")
    monkeypatch.setenv("BLADEX_HOME", str(tmp_path))
    assert _prune_logs(keep=20) == 5
    assert not (logs / "proxy-20260901-000000.log").exists()


def test_reference_in_any_docs_subdir_counts(monkeypatch, tmp_path: Path) -> None:
    logs = _mk_logs(tmp_path)
    (tmp_path / "docs" / "adr").mkdir(parents=True)
    (tmp_path / "docs" / "adr" / "0099.md").write_text("proxy-20260901-000003.log\n", encoding="utf-8")
    monkeypatch.setenv("BLADEX_HOME", str(tmp_path))
    _prune_logs(keep=20)
    assert (logs / "proxy-20260901-000003.log").exists()


def test_keep_zero_still_disables(monkeypatch, tmp_path: Path) -> None:
    logs = _mk_logs(tmp_path)
    monkeypatch.setenv("BLADEX_HOME", str(tmp_path))
    monkeypatch.setenv("BLADEX_LOG_KEEP", "0")
    assert _prune_logs() == 0
    assert len(list(logs.glob("*.log"))) == 25
