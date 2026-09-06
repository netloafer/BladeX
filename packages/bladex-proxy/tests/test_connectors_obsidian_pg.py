"""Beta T14 Obsidian + T15 PostgreSQL connector 验收。

Obsidian：真实文件系统 vault（tmp）——增量/幂等重写/改归属/墓碑/hard rules/
wikilink 结构。PostgreSQL：注入 fake 连接（capture SQL）——UPSERT 幂等形态/
墓碑 DELETE/建表脚本/sync_state 游标/断点续传（worker 层）；真 PG 剧本
gated 在 BLADEX_PG_TEST_DSN（用户机/CI 有 docker 时跑）。
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
from bladex_core.exporters import ExportBatch
from bladex_core.fact import Fact
from bladex_core.matter import (
    EdgeProvenance,
    EdgeTargetType,
    Matter,
    MatterEdge,
    MatterOrigin,
    MatterStatus,
)
from bladex_proxy.exporters.obsidian import ObsidianExporter
from bladex_proxy.exporters.postgres import PostgresExporter, _load_schema_sql


def _fact(fid: str, content: str, matter_id: str = "") -> Fact:
    return Fact(id=fid, content=content, source_user_id="u1",
                matter_id=matter_id, created_at=datetime.now(UTC))


def _matter(mid: str, title: str) -> Matter:
    return Matter(matter_id=mid, title=title, summary=f"{title} 摘要",
                  status=MatterStatus.ACTIVE, origin=MatterOrigin.AUTO)


# ═══ Obsidian ═════════════════════════════════════════════════════


@pytest.fixture()
def vault(tmp_path: Path) -> tuple[ObsidianExporter, Path]:
    root = tmp_path / "vault/BladeX"
    return ObsidianExporter("v", "private", {"vault_dir": str(root)}), root


def test_obsidian_matter_page_and_fact_wikilinks(vault):
    exp, root = vault
    exp.sync_incremental(ExportBatch(
        facts=[_fact("f1", "用户偏好中文", "m1"), _fact("f2", "无归属事实")],
        matters=[_matter("m1", "Beta 发布")],
        hard_rules=["MUST be polite"],
    ), "")
    page = (root / "Matters/m1.md").read_text()
    assert "bladex_id: m1" in page and "status: active" in page
    assert "# Beta 发布" in page
    assert "[[bx-fact-f1]]" in page  # wikilink 表达归属边
    assert "<!--bladex-fact:f1-->" in page
    inbox = (root / "Inbox.md").read_text()
    assert "[[bx-fact-f2]]" in inbox
    rules = (root / "Hard Rules.md").read_text()
    assert "MUST be polite" in rules


def test_obsidian_idempotent_rewrite(vault):
    """同一批同步两遍：不产生重复条目（按 id 定位重写）。"""
    exp, root = vault
    batch = ExportBatch(facts=[_fact("f1", "内容", "m1")],
                        matters=[_matter("m1", "事项")])
    exp.sync_incremental(batch, "")
    exp.sync_incremental(batch, "")
    page = (root / "Matters/m1.md").read_text()
    assert page.count("<!--bladex-fact:f1-->") == 1


def test_obsidian_fact_content_update_and_reassign(vault):
    exp, root = vault
    exp.sync_incremental(ExportBatch(
        facts=[_fact("f1", "旧内容", "m1")],
        matters=[_matter("m1", "甲"), _matter("m2", "乙")]), "")
    # 内容更新 + 改归属 m1 → m2
    exp.sync_incremental(ExportBatch(facts=[_fact("f1", "新内容", "m2")]), "")
    m1 = (root / "Matters/m1.md").read_text()
    m2 = (root / "Matters/m2.md").read_text()
    assert "<!--bladex-fact:f1-->" not in m1
    assert "新内容" in m2 and "旧内容" not in m2


def test_obsidian_edge_only_reassign_moves_line(vault):
    """只有边、无 fact 本体的批：已存在的行搬家。"""
    exp, root = vault
    exp.sync_incremental(ExportBatch(facts=[_fact("f1", "内容")]), "")
    assert "<!--bladex-fact:f1-->" in (root / "Inbox.md").read_text()
    exp.sync_incremental(ExportBatch(
        matters=[_matter("m1", "事项")],
        edges=[MatterEdge(edge_id="e1", matter_id="m1",
                          target_type=EdgeTargetType.FACT, target_key="f1",
                          provenance=EdgeProvenance.MANUAL)]), "")
    assert "<!--bladex-fact:f1-->" not in (root / "Inbox.md").read_text()
    assert "<!--bladex-fact:f1-->" in (root / "Matters/m1.md").read_text()


def test_obsidian_tombstones(vault):
    exp, root = vault
    exp.sync_incremental(ExportBatch(
        facts=[_fact("f1", "内容", "m1")], matters=[_matter("m1", "事项")]), "")
    exp.handle_tombstone("fact", "f1")
    assert "<!--bladex-fact:f1-->" not in (root / "Matters/m1.md").read_text()
    exp.handle_tombstone("matter", "m1")
    assert not (root / "Matters/m1.md").exists()


def test_obsidian_session_edges(vault):
    exp, root = vault
    exp.sync_incremental(ExportBatch(
        matters=[_matter("m1", "事项")],
        edges=[MatterEdge(edge_id="e1", matter_id="m1",
                          target_type=EdgeTargetType.SESSION,
                          target_key="u1/agentA/s1/")]), "")
    page = (root / "Matters/m1.md").read_text()
    assert "u1/agentA/s1/" in page
    # 幂等
    exp.sync_incremental(ExportBatch(
        edges=[MatterEdge(edge_id="e1", matter_id="m1",
                          target_type=EdgeTargetType.SESSION,
                          target_key="u1/agentA/s1/")]), "")
    assert (root / "Matters/m1.md").read_text().count(
        "<!--bladex-session:u1/agentA/s1/-->") == 1


# ═══ PostgreSQL（fake conn）═══════════════════════════════════════


class FakeCursor:
    def __init__(self, log: list):
        self._log = log

    def execute(self, sql: str, params: tuple | None = None) -> None:
        self._log.append((" ".join(sql.split()), params))


class FakeConn:
    def __init__(self, log: list):
        self._log = log
        self.closed = False

    def cursor(self) -> FakeCursor:
        return FakeCursor(self._log)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def close(self) -> None:
        self.closed = True


@pytest.fixture()
def pg() -> tuple[PostgresExporter, list]:
    log: list = []
    exp = PostgresExporter("pg", "private",
                           {"_connect_factory": lambda: FakeConn(log)})
    return exp, log


def test_pg_schema_sql_loads_and_is_idempotent():
    sql = _load_schema_sql()
    assert "CREATE TABLE IF NOT EXISTS facts" in sql
    assert "CREATE TABLE IF NOT EXISTS matters" in sql
    assert "CREATE TABLE IF NOT EXISTS matter_edges" in sql
    assert "CREATE TABLE IF NOT EXISTS hard_rules" in sql
    assert "CREATE TABLE IF NOT EXISTS sync_state" in sql
    assert "bladex_schema_version" in sql
    assert "DROP" not in sql.upper()


def test_pg_upserts_and_cursor(pg):
    exp, log = pg
    exp.sync_incremental(ExportBatch(
        facts=[_fact("f1", "内容")],
        matters=[_matter("m1", "事项")],
        edges=[MatterEdge(edge_id="e1", matter_id="m1",
                          target_type=EdgeTargetType.FACT, target_key="f1")],
        hard_rules=["MUST x"],
    ), "2026-08-03T00:00:00")
    sqls = [s for s, _ in log]
    assert any("INSERT INTO facts" in s and "ON CONFLICT (id) DO UPDATE" in s
               for s in sqls)
    assert any("INSERT INTO matters" in s and "ON CONFLICT (matter_id) DO UPDATE" in s
               for s in sqls)
    assert any("INSERT INTO matter_edges" in s and "ON CONFLICT (edge_id) DO UPDATE" in s
               for s in sqls)
    assert any("INSERT INTO hard_rules" in s and "DO NOTHING" in s for s in sqls)
    state = [(s, p) for s, p in log if "INSERT INTO sync_state" in s]
    assert state and state[0][1] == ("pg", "2026-08-03T00:00:00")


def test_pg_tombstone_deletes(pg):
    exp, log = pg
    exp.handle_tombstone("fact", "f1")
    exp.handle_tombstone("matter", "m1")
    exp.handle_tombstone("edge", "e1")
    sqls = [s for s, _ in log]
    assert any("DELETE FROM facts WHERE id" in s for s in sqls)
    assert any("DELETE FROM matter_edges WHERE target_key" in s for s in sqls)
    assert any("DELETE FROM matters WHERE matter_id" in s for s in sqls)
    assert any("DELETE FROM matter_edges WHERE edge_id" in s for s in sqls)


def test_pg_requires_dsn_env():
    with pytest.raises(ValueError, match="BLADEX_PG_DSN"):
        PostgresExporter("pg", "private", {})


@pytest.mark.skipif(not os.environ.get("BLADEX_PG_TEST_DSN"),
                    reason="真 PG 剧本需 BLADEX_PG_TEST_DSN（docker 环境）")
def test_pg_real_roundtrip():  # pragma: no cover —— 用户机/CI docker 跑
    import psycopg
    dsn = os.environ["BLADEX_PG_TEST_DSN"]
    exp = PostgresExporter("pg-real", "private",
                           {"_connect_factory": lambda: psycopg.connect(dsn)})
    exp.sync_incremental(ExportBatch(facts=[_fact("f-real", "真实往返")]), "c1")
    exp.sync_incremental(ExportBatch(facts=[_fact("f-real", "真实往返 v2")]), "c2")
    with psycopg.connect(dsn) as conn:
        row = conn.execute("SELECT content FROM facts WHERE id = 'f-real'").fetchone()
        assert row[0] == "真实往返 v2"  # UPSERT 幂等
    exp.handle_tombstone("fact", "f-real")
    with psycopg.connect(dsn) as conn:
        assert conn.execute(
            "SELECT count(*) FROM facts WHERE id = 'f-real'").fetchone()[0] == 0
