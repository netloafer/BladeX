"""三段式 T2（2026-08-10）：FTS schema v2（topic 列）迁移与检索。

钉五件事：
  - 旧 schema（无 topic 列）可写打开 → 自动 drop 重建为 v2，不炸；
  - 只读端遇旧 schema 不迁移、照常可用（degraded 不崩）；
  - backfill 幂等（两跑行数一致）；
  - topic 可被 FTS 检索命中（content 里没有的主题词靠 topic 列捞回）；
  - topic 经蒸馏台账重放两次逐字一致（G6 重建等价）。
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

from bladex_core.distillation import DistillFact
from bladex_core.fact import Fact
from bladex_proxy.storage.fts_index import _SCHEMA_VERSION, _TABLE, FtsIndex
from bladex_proxy.storage.memory_index import MemoryIndex


def _make_v1_db(tmpdir: str) -> Path:
    """手工造一个 v1 schema（无 topic 列）的 fts.sqlite。"""
    path = Path(tmpdir) / "fts.sqlite"
    db = sqlite3.connect(str(path))
    db.execute(
        f"CREATE VIRTUAL TABLE {_TABLE} USING fts5("
        "fact_id UNINDEXED, content, entities, subject, attribute, "
        "tokenize='trigram')"
    )
    db.execute(
        f"INSERT INTO {_TABLE} (fact_id, content, entities, subject, attribute) "
        "VALUES ('f-old', 'legacy content row', '', '', '')"
    )
    db.commit()
    db.close()
    return path


def _fact(fid: str, content: str, *, topic: str = "") -> Fact:
    return Fact(id=fid, content=content, topic=topic)


def test_old_schema_writable_open_migrates():
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_v1_db(tmpdir)
        fts = FtsIndex(tmpdir, read_only=False)
        assert fts.open() is True
        # 迁移后：新 schema 含 topic 列 + user_version 打上版本章
        sql = fts._db.execute(
            "SELECT sql FROM sqlite_master WHERE name = ?", (_TABLE,)).fetchone()[0]
        assert "topic" in sql
        ver = fts._db.execute("PRAGMA user_version").fetchone()[0]
        assert ver == _SCHEMA_VERSION
        # 旧行随 drop 归零（backfill_fts 的既有自动入口会灌回）
        assert fts.count() == 0
        # 新 schema 可正常写入含 topic 的行
        fts.upsert(_fact("f-new", "新内容", topic="三段式主题"))
        assert fts.count() == 1
        fts.close()


def test_old_schema_readonly_untouched():
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_v1_db(tmpdir)
        fts = FtsIndex(tmpdir, read_only=True)
        assert fts.open() is True          # degraded 但可用
        sql = fts._db.execute(
            "SELECT sql FROM sqlite_master WHERE name = ?", (_TABLE,)).fetchone()[0]
        assert "topic" not in sql          # 只读端不迁移（写权在 consolidator）
        assert fts.count() == 1            # 旧行还在，检索照常
        assert fts.search("legacy content")
        fts.close()


def test_backfill_idempotent_with_topic():
    with tempfile.TemporaryDirectory() as tmpdir:
        index = MemoryIndex(Path(tmpdir) / "index", embedder=None, read_only=False)
        index.open()
        for i in range(3):
            index.add_fact(_fact(f"f-{i}", f"事实内容第{i}条", topic="泰山啤酒破产重整"))
        b1, a1 = index.backfill_fts(force=True)
        b2, a2 = index.backfill_fts(force=True)
        assert a1 == a2 == 3               # 幂等：两跑行数一致
        index.close()


def test_topic_searchable_via_fts():
    """content 不含主题词的 fact，靠 topic 列被词法检索命中。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        fts = FtsIndex(tmpdir, read_only=False)
        assert fts.open()
        fts.upsert(_fact("f-t", "方案C核心资产估值约4.5亿元", topic="泰山啤酒破产重整"))
        fts.upsert(_fact("f-x", "完全无关的另一条内容", topic=""))
        rows = fts.search("泰山啤酒破产重整的进展")
        assert [fid for fid, _ in rows] and rows[0][0] == "f-t"
        fts.close()


def test_topic_replay_equivalence_via_ledger():
    """G6：topic 经 distill/ 台账重放两次逐字一致（DistillFact.topic 穿越台账）。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        index = MemoryIndex(Path(tmpdir) / "index", embedder=None, read_only=False)
        index.open()
        facts = [DistillFact(content="事实一", topic="主题T", item_kind="assertion",
                             subject="s", attribute="a")]
        index.append_distill("源文本", "m", "v5-three-seg-001", facts, [], "ctx")
        r1 = index.get_distill("源文本", "m", "v5-three-seg-001", "ctx")
        r2 = index.get_distill("源文本", "m", "v5-three-seg-001", "ctx")
        assert r1 is not None and r2 is not None
        assert r1.facts[0].topic == r2.facts[0].topic == "主题T"
        index.close()
