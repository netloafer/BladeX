"""Beta T15 PostgreSQL connector——四表 + sync_state 游标表，UPSERT 幂等。

- 连接串走 env（options.dsn_env，默认 BLADEX_PG_DSN；凭证不入 toml）。
- 建表脚本 postgres_schema.sql 随包发布（importlib.resources 读取），
  首次连接自动应用（全部 IF NOT EXISTS，幂等）。
- 墓碑 = DELETE；fact 墓碑同时清指向它的边。
- exposure 过滤在 worker 层已做（本类只管落库）。
- 依赖 psycopg3（可选依赖：`pip install 'bladex-proxy[postgres]'`）。
  测试可注入 options["_connect_factory"] 替身，不需要真 PG。
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from importlib import resources
from typing import Any

import structlog
from bladex_core.exporters import ExportBatch

logger = structlog.get_logger()

_SCHEMA_FILE = "postgres_schema.sql"


def _load_schema_sql() -> str:
    return (resources.files("bladex_proxy.exporters") / _SCHEMA_FILE).read_text(
        encoding="utf-8")


def _dt(v: Any) -> datetime | None:
    if isinstance(v, datetime):
        return v
    if isinstance(v, str) and v:
        try:
            return datetime.fromisoformat(v)
        except ValueError:
            return None
    return None


class PostgresExporter:
    def __init__(self, name: str, exposure: str, options: dict[str, Any]) -> None:
        self.name = name
        self.exposure = exposure
        self._connect = options.get("_connect_factory")
        if self._connect is None:
            dsn_env = str(options.get("dsn_env", "BLADEX_PG_DSN"))
            dsn = os.environ.get(dsn_env, "")
            if not dsn:
                raise ValueError(
                    f"postgres exporter: env {dsn_env} is empty "
                    "(set the connection string there; secrets stay out of toml)")
            try:
                import psycopg
            except ImportError as e:
                raise RuntimeError(
                    "postgres exporter needs psycopg3: "
                    "pip install 'psycopg[binary]>=3'") from e
            self._connect = lambda: psycopg.connect(dsn)
        self._schema_ready = False

    # ── Exporter 协议 ─────────────────────────────────────────────

    def sync_incremental(self, batch: ExportBatch, since_cursor: str) -> None:
        conn = self._connect()
        try:
            with conn:
                cur = conn.cursor()
                self._ensure_schema(cur)
                for fact in batch.facts:
                    self._upsert_fact(cur, fact)
                for matter in batch.matters:
                    self._upsert_matter(cur, matter)
                for edge in batch.edges:
                    self._upsert_edge(cur, edge)
                for rule in batch.hard_rules:
                    cur.execute(
                        "INSERT INTO hard_rules (content) VALUES (%s) "
                        "ON CONFLICT (content) DO NOTHING", (rule,))
                # 目标侧游标记录（观测/对账用；权威游标在 worker 状态文件）
                cur.execute(
                    "INSERT INTO sync_state (exporter_name, cursor_value, updated_at) "
                    "VALUES (%s, %s, now()) "
                    "ON CONFLICT (exporter_name) DO UPDATE SET "
                    "cursor_value = EXCLUDED.cursor_value, updated_at = now()",
                    (self.name, since_cursor))
        finally:
            conn.close()

    def handle_tombstone(self, target_type: str, target_key: str) -> None:
        conn = self._connect()
        try:
            with conn:
                cur = conn.cursor()
                self._ensure_schema(cur)
                if target_type == "fact":
                    cur.execute("DELETE FROM facts WHERE id = %s", (target_key,))
                    cur.execute("DELETE FROM matter_edges WHERE target_key = %s",
                                (target_key,))
                elif target_type == "matter":
                    cur.execute("DELETE FROM matters WHERE matter_id = %s",
                                (target_key,))
                    cur.execute("DELETE FROM matter_edges WHERE matter_id = %s",
                                (target_key,))
                elif target_type == "edge":
                    cur.execute("DELETE FROM matter_edges WHERE edge_id = %s",
                                (target_key,))
                # turn/distill 墓碑与 PG 侧无对应物
        finally:
            conn.close()

    # ── 内部 ─────────────────────────────────────────────────────

    def _ensure_schema(self, cur: Any) -> None:
        if self._schema_ready:
            return
        cur.execute(_load_schema_sql())
        self._schema_ready = True

    def _upsert_fact(self, cur: Any, fact: Any) -> None:
        payload = fact.model_dump(mode="json", exclude={"embedding"})
        cur.execute(
            """
            INSERT INTO facts (id, content, kind, item_kind, subject, attribute,
                               source_user_id, scope, exposure_ceiling, importance,
                               created_at, t_invalid, payload)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (id) DO UPDATE SET
                content = EXCLUDED.content, kind = EXCLUDED.kind,
                item_kind = EXCLUDED.item_kind, subject = EXCLUDED.subject,
                attribute = EXCLUDED.attribute,
                source_user_id = EXCLUDED.source_user_id,
                scope = EXCLUDED.scope,
                exposure_ceiling = EXCLUDED.exposure_ceiling,
                importance = EXCLUDED.importance,
                created_at = EXCLUDED.created_at,
                t_invalid = EXCLUDED.t_invalid, payload = EXCLUDED.payload
            """,
            (fact.id, fact.content, getattr(fact, "kind", "general"),
             str(getattr(getattr(fact, "item_kind", ""), "value",
                         getattr(fact, "item_kind", "assertion"))),
             getattr(fact, "subject", ""), getattr(fact, "attribute", ""),
             getattr(fact, "source_user_id", ""), getattr(fact, "scope", ""),
             getattr(fact, "exposure_ceiling", "public"),
             float(getattr(fact, "importance", 1.0)),
             _dt(getattr(fact, "created_at", None)),
             _dt(getattr(fact, "t_invalid", None)),
             json.dumps(payload, ensure_ascii=False)))

    def _upsert_matter(self, cur: Any, matter: Any) -> None:
        payload = matter.model_dump(mode="json", exclude={"centroid", "embedding"})
        cur.execute(
            """
            INSERT INTO matters (matter_id, title, summary, status, origin,
                                 scope, updated_at, payload)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (matter_id) DO UPDATE SET
                title = EXCLUDED.title, summary = EXCLUDED.summary,
                status = EXCLUDED.status, origin = EXCLUDED.origin,
                scope = EXCLUDED.scope, updated_at = EXCLUDED.updated_at,
                payload = EXCLUDED.payload
            """,
            (matter.matter_id, matter.title, getattr(matter, "summary", ""),
             str(getattr(matter.status, "value", matter.status)),
             str(getattr(matter.origin, "value", matter.origin)),
             getattr(matter, "scope", ""),
             _dt(getattr(matter, "updated_at", None)),
             json.dumps(payload, ensure_ascii=False)))

    def _upsert_edge(self, cur: Any, edge: Any) -> None:
        payload = edge.model_dump(mode="json")
        cur.execute(
            """
            INSERT INTO matter_edges (edge_id, matter_id, target_type, target_key,
                                      relation, provenance, weight, payload)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (edge_id) DO UPDATE SET
                matter_id = EXCLUDED.matter_id,
                target_type = EXCLUDED.target_type,
                target_key = EXCLUDED.target_key,
                relation = EXCLUDED.relation,
                provenance = EXCLUDED.provenance,
                weight = EXCLUDED.weight, payload = EXCLUDED.payload
            """,
            (edge.edge_id, edge.matter_id,
             str(getattr(edge.target_type, "value", edge.target_type)),
             edge.target_key,
             str(getattr(edge.relation, "value", edge.relation)),
             str(getattr(edge.provenance, "value", edge.provenance)),
             float(getattr(edge, "weight", 1.0)),
             json.dumps(payload, ensure_ascii=False)))
