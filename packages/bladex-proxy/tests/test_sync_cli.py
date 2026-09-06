"""bladex sync CLI（direct 模式端到端 + 参数面）。

真实路径：临时 Memory Hub 写 turn → cli.main(["sync","run","--direct",...]) →
Memory Index 产出 fact + 日志文件落盘。embedder/蒸馏自举打桩（不下载模型、不调 LLM）。
"""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest
from bladex_proxy import cli
from bladex_proxy.models import Identity, Turn, TurnStatus
from bladex_proxy.storage.memory_index import MemoryIndex
from bladex_proxy.storage.memory_hub import MemoryHub


class MockEmbedder:
    model_identity = "mock:test"
    model_name = "mock"

    def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * 32
            v[hash(t) % 32] = 1.0
            out.append(v)
        return out


def _seed_ledger(path: Path, n: int = 3) -> None:
    ledger = MemoryHub(path)
    ledger.open()
    for i in range(n):
        identity = Identity(user_id="u1", agent_id="claude-code",
                            session_id="s1", turn_index=i)
        turn = Turn(identity=identity, model="m",
                    request_messages=[{"role": "user",
                                       "content": f"第 {i} 条足够长的用户消息，主题编号 {i}，用于同步测试。"}],
                    response_text="ok", status=TurnStatus.OK,
                    ts=datetime(2026, 8, 3, 10, 0, i, tzinfo=UTC))
        ledger.put(identity.storage_key(f"1719990{i}-0"), turn)
    ledger.close()


@pytest.fixture()
def _env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """隔离存储路径 + 打桩 embedder/蒸馏自举（不碰 live 库、不下载、不调 LLM）。"""
    ledger_path = tmp_path / "ledger"
    index_path = tmp_path / "index"
    _seed_ledger(ledger_path)
    monkeypatch.setenv("BLADEX_ROCKSDB_PATH", str(ledger_path))
    monkeypatch.setenv("BLADEX_INDEX_PATH", str(index_path))
    import bladex_proxy.embedding as emb
    monkeypatch.setattr(emb, "build_embedder", lambda cfg, role="": MockEmbedder())
    import bladex_proxy.routing_config as rcfg
    monkeypatch.setattr(rcfg, "bootstrap_distill_model", lambda *a, **k: "")
    return tmp_path


def test_sync_run_direct_end_to_end(_env: Path):
    log_path = _env / "sync.log"
    rc = cli.main(["sync", "run", "--direct", "--concurrency", "2",
                   "--log", str(log_path)])
    assert rc == 0
    text = log_path.read_text(encoding="utf-8")
    assert "Sync complete" in text and "direct mode" in text
    # Memory Index 真有产出（passthrough 蒸馏：短消息透传成 fact）
    index = MemoryIndex(Path(str(_env / "index")), embedder=MockEmbedder(), read_only=False)
    index.open()
    assert index.fact_count() > 0
    index.close()


def test_sync_run_direct_full(_env: Path):
    log_path = _env / "sync_full.log"
    rc = cli.main(["sync", "run", "--direct", "--full", "--log", str(log_path)])
    assert rc == 0
    assert "full=True" in log_path.read_text(encoding="utf-8")


def test_sync_run_direct_agent_filter(_env: Path):
    """过滤参数贯通：排除唯一 agent → 零产出但正常完成。"""
    log_path = _env / "sync_filter.log"
    rc = cli.main(["sync", "run", "--direct", "--exclude-agents", "claude-code",
                   "--log", str(log_path)])
    assert rc == 0
    assert "new_facts=0" in log_path.read_text(encoding="utf-8")


def test_sync_run_index_lock_conflict_reports_clearly(_env: Path, monkeypatch):
    """Memory Index 写锁被占（模拟 consolidator 在跑未接单）→ 明确报错不排队。"""
    holder = MemoryIndex(Path(str(_env / "index")), embedder=MockEmbedder(), read_only=False)
    holder.open()  # 占住写锁
    log_path = _env / "sync_lock.log"
    rc = cli.main(["sync", "run", "--direct", "--log", str(log_path)])
    holder.close()
    assert rc == 1
    text = log_path.read_text(encoding="utf-8")
    assert "could not open Memory Index" in text and "consolidator" in text
