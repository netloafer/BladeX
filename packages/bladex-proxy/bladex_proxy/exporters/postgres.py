"""Beta T15 PostgreSQL connector——四表 + sync_state 游标表，UPSERT 幂等。

- 连接串走 env（options.dsn_env，默认 BLADEX_PG_DSN；凭证不入 toml）。
- 建表脚本 postgres_schema.sql 随包发布（importlib.resources 读取），
  首次连接自动应用（全部 IF NOT EXISTS，幂等）。
- 墓碑 = DELETE；fact 墓碑同时清指向它的边。
- exposure 过滤在 worker 层已做（本类只管落库）。
- 依赖 psycopg3（可选依赖：`pip install 'bladex-proxy[postgres]'`）。
  测试可注入 options["_connect_factory"] 替身，不需要真 PG。

回流（Z4，MQ-P33）——**收件表，不监听导出表**：
- 导出表（facts / matters / matter_edges / hard_rules）保持只读镜像、行为一字不变：
  它们是 `ON CONFLICT DO UPDATE`，用户直改会被下一轮同步盖掉，监听它们等于邀请冲突。
- 新增只追加的 `bladex_edit_inbox`（`postgres_inbox_schema.sql`）：外部程序 INSERT
  修改请求，BladeX 轮询 ⇒ 经 proxy 落 Hub 管理事件 ⇒ 原地回写 accepted / rejected + 原因。
  **BladeX 从不删收件行**（审计留档）。
- 🔴 **默认关**：`options.inbox = true` 才生效（不开时连表都不建）。
- 🔴 **拒绝硬规则**：硬规则每轮注入每个请求——能写这张库的人就能往所有 agent 塞指令。
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
_INBOX_SCHEMA_FILE = "postgres_inbox_schema.sql"
#: 单次轮询取走的收件行上限（探针粒度，不是待标定旋钮）。
_INBOX_BATCH = 200
_INBOX_OPS = frozenset({"update", "assign", "delete"})


def _load_schema_sql(name: str = _SCHEMA_FILE) -> str:
    return (resources.files("bladex_proxy.exporters") / name).read_text(
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
        # Z4 收件表：默认关（拍板 (26)）；只有显式 true 才建表、才轮询。
        self.inbox_enabled = options.get("inbox", False) is True
        self._inbox_ready = False
        self._inbox_seen = 0  # Z4b：上次 collect 取走的最大收件行 id

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
                if batch.hard_rules:
                    # Z1.2 ⑦：硬规则变了会整套重导——镜像要与生效规则**一致**，不是新旧并存
                    # （此前只在首轮导一次、只插不删，热加载后表里会同时躺着新旧两套）。
                    cur.execute("DELETE FROM hard_rules WHERE NOT (content = ANY(%s))",
                                (list(batch.hard_rules),))
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

    # ── Z4 收件表 ────────────────────────────────────────────────

    def collect_edits(self, max_deletes: int) -> list:
        """取 pending 收件行 ⇒ `BackflowEdit`（通道侧已能判定的非法行直接成 `reject`）。"""
        from bladex_proxy.export_backflow import BackflowEdit
        if not self.inbox_enabled:
            return []
        conn = self._connect()
        try:
            with conn:
                cur = conn.cursor()
                if not self._inbox_ready:
                    cur.execute(_load_schema_sql(_INBOX_SCHEMA_FILE))
                    self._inbox_ready = True
                # Z4b 水位：先读全表 max(id)，再取 pending——id ≤ 水位的行要么本次取走、
                # 要么早已处理，唤醒探针只需盯"比它新"的行（见 inbox_has_new）。
                cur.execute("SELECT COALESCE(MAX(id), 0) FROM bladex_edit_inbox")
                _hw = cur.fetchone()
                high_water = int(_hw[0]) if _hw else 0
                cur.execute(
                    "SELECT id, target_type, target_id, op, new_value, submitter "
                    "FROM bladex_edit_inbox WHERE status = 'pending' "
                    "ORDER BY id LIMIT %s", (_INBOX_BATCH,))
                rows = list(cur.fetchall())
        finally:
            conn.close()
        if len(rows) >= _INBOX_BATCH:
            # 被 LIMIT 截断：水位只推进到取走的最后一行，剩下的 pending 仍会唤醒下一轮
            self._inbox_seen = max(self._inbox_seen, max(int(r[0]) for r in rows))
        else:
            self._inbox_seen = max(self._inbox_seen, high_water)
        edits: list = []
        deletes: list = []
        for row_id, ttype, tid, op, value, submitter in rows:
            who = f"postgres:{submitter or '?'}"

            def _rej(reason: str, _rid: int = row_id, _who: str = who) -> BackflowEdit:
                return BackflowEdit("reject", "postgres", _who, reason=reason, ref=_rid)

            if ttype != "fact":
                edits.append(_rej(
                    "hard rules cannot be changed through the Postgres inbox (they are "
                    "injected into every request; edit them in config/.env or the "
                    "Obsidian Hard Rules page)" if ttype in ("hard_rule", "hard_rules")
                    else f"unsupported target_type {ttype!r} (only 'fact')"))
            elif op not in _INBOX_OPS:
                edits.append(_rej(f"unsupported op {op!r} (update / assign / delete)"))
            elif not tid:
                edits.append(_rej("target_id is empty"))
            elif op == "update":
                content = (value or "").replace("\n", " ").strip()
                edits.append(BackflowEdit("correct", "postgres", who, fact_id=tid,
                                          content=content, ref=row_id)
                             if content else _rej("new_value is empty"))
            elif op == "assign":
                mid = (value or "").strip()
                edits.append(BackflowEdit("assign" if mid else "detach", "postgres", who,
                                          fact_id=tid, matter_id=mid, ref=row_id))
            else:
                deletes.append(BackflowEdit("delete", "postgres", who, fact_id=tid,
                                            ref=row_id))
        if len(deletes) > max_deletes:
            logger.warning("pg_inbox_mass_delete_refused", deletes=len(deletes),
                           threshold=max_deletes, exporter=self.name,
                           hint="more delete requests in one poll than "
                                "BLADEX_BACKFLOW_MAX_DELETES allows -- all of them are "
                                "rejected; resubmit in smaller batches or raise the knob")
            deletes = [BackflowEdit(
                "reject", "postgres", d.submitter, ref=d.ref,
                reason=f"mass-delete guard: {len(deletes)} delete requests in one poll "
                       f"> BLADEX_BACKFLOW_MAX_DELETES={max_deletes}; nothing deleted")
                for d in deletes]
        return edits + deletes

    def inbox_has_new(self) -> bool:
        """Z4b 唤醒探针：收件表里有没有比上次取走的更新的行（一条 `SELECT max(id)`）。

        以 `max(id)` 对比**上次 collect 取走的最大 id**，而不是对比某个时刻的快照：
        BladeX 处理收件行只 UPDATE 不 INSERT，max(id) 不动 ⇒ 不需要每轮重取基线，
        也不会被自己的回写唤醒；proxy 不在（retry）时那几行已被取走过，不会反复唤醒。
        `inbox` 关、或表还没建（首轮 collect 之前）⇒ **零 SQL**，直接 False。
        限频由调用方（`export_sync.ExternalChangeProbe`，`BLADEX_BACKFLOW_PG_PROBE_S`）负责。
        """
        if not self.inbox_enabled or not self._inbox_ready:
            return False
        conn = self._connect()
        try:
            with conn:
                cur = conn.cursor()
                cur.execute("SELECT COALESCE(MAX(id), 0) FROM bladex_edit_inbox")
                row = cur.fetchone()
        finally:
            conn.close()
        return bool(row) and int(row[0]) > self._inbox_seen

    def ack(self, edit: Any, result: Any) -> None:
        """回写处理结果（只 UPDATE 本行；从不 DELETE 收件行）。retry 不回写，留 pending。"""
        if result is None or result.status not in ("accepted", "rejected"):
            return
        conn = self._connect()
        try:
            with conn:
                conn.cursor().execute(
                    "UPDATE bladex_edit_inbox SET status = %s, result_reason = %s, "
                    "event_ref = %s, processed_at = now() "
                    "WHERE id = %s AND status = 'pending'",
                    (result.status, result.reason or "", result.event_ref or "", edit.ref))
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
