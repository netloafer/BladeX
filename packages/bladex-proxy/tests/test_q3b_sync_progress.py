"""批 Q3B · MQ-I25 ②③：生产 `sync run` 看不见嵌入 / novelty 进度；`--full` 与 `--max-turns` 语义不清。

T2a 冒烟事故：`--max-turns 5` 在全量重建下被静默忽略、嵌入阶段 48 分钟零输出。钉住：
  ② 嵌入 / novelty 两段的进度（带速率 / 剩余时间）进 progress_cb，并出现在 `sync run` 的日志里；
  ③ `--full --max-turns N` 当场报错（rc=2），不再静默跑全量。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from bladex_core.consolidation_proxy import ProxyConsolidator
from bladex_proxy import cli
from bladex_proxy.cli.ops_cmds import _fmt_progress

from .test_sync_cli import _env  # noqa: F401 —— 同一套隔离存储 + 打桩 embedder


class _Emb:
    def embed(self, texts):
        return [[float(len(t)), 1.0] for t in texts]


def test_embed_chunked_reports_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BLADEX_EMBED_BATCH", "2")
    events: list[dict] = []
    c = ProxyConsolidator(embedder=_Emb(), progress_cb=events.append)
    out = c._embed_chunked(["a", "bb", "ccc", "dddd", "e"], phase="candidates")
    assert len(out) == 5
    emb = [e for e in events if e["phase"] == "embed:candidates"]
    assert [(e["done"], e["total"]) for e in emb] == [(2, 5), (4, 5), (5, 5)]
    assert all("rate" in e and "eta_s" in e for e in emb)
    assert emb[-1]["eta_s"] == 0


def test_fmt_progress_shows_rate_and_eta() -> None:
    line = _fmt_progress({"phase": "embed:candidates", "done": "40", "total": "100",
                          "rate": "11.2", "eta_s": "5"})
    assert line == "[embed:candidates] 40/100 (40%) 11.2/s eta 5s"


def test_sync_run_log_shows_embed_and_novelty(_env: Path) -> None:  # noqa: F811
    log_path = _env / "sync.log"
    assert cli.main(["sync", "run", "--direct", "--log", str(log_path)]) == 0
    text = log_path.read_text(encoding="utf-8")
    assert "[embed:candidates]" in text
    assert "[novelty]" in text


def test_full_with_max_turns_is_rejected(_env: Path, capsys) -> None:  # noqa: F811
    log_path = _env / "sync.log"
    rc = cli.main(["sync", "run", "--direct", "--full", "--max-turns", "5",
                   "--log", str(log_path)])
    assert rc == 2
    out = capsys.readouterr().out
    assert "--full and --max-turns cannot be combined" in out and out.isascii()
    assert not log_path.exists()  # 什么都没跑


def test_max_turns_alone_still_runs(_env: Path) -> None:  # noqa: F811
    log_path = _env / "sync.log"
    assert cli.main(["sync", "run", "--direct", "--max-turns", "2", "--log", str(log_path)]) == 0
