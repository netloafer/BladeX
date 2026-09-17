"""Beta T12 快照导出/导入（B3.1）——schema 版本化 JSONL。

格式（每行一个 JSON 对象，首行 meta）：

    {"type":"meta","schema":1,"exported_at":...,"embed_model_id":...,"counts":{...}}
    {"type":"hard_rule","schema":1,"content":"MUST ..."}
    {"type":"fact","schema":1,"id":...,...}          # 不含 embedding（导入侧重嵌）
    {"type":"matter","schema":1,"matter_id":...,...} # 不含 centroid/embedding
    {"type":"edge","schema":1,"edge_id":...,...}
    {"type":"admin_event","schema":1,"event_type":...,...}
    {"type":"turn","schema":1,"key":...,...}         # 仅 --full（Memory Hub 整档）

语义（ADR-0012 §3.5/3.6）：
  - 墓碑数据不出现：Memory Index 已级联清除；Memory Hub scan 跳过墓碑覆盖项。
  - 导入 = 按 id 幂等 merge + manual 压 auto + 冲突非阻断标注；
    导入实体以 FACT_IMPORT/MATTER_IMPORT/EDGE_IMPORT 管理事件入 Memory Hub journal
    —— full rebuild 重放 journal，导入的记忆不因重建丢失（重建等价）。
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

import structlog
from bladex_core.consolidation_proxy import embed_passage_compat
from bladex_core.fact import Fact
from bladex_core.matter import EdgeProvenance, Matter, MatterEdge, MatterOrigin

from bladex_proxy.models import AdminEvent, AdminEventType
from bladex_proxy.storage.memory_hub import MemoryHub
from bladex_proxy.storage.memory_index import MemoryIndex

logger = structlog.get_logger()

SNAPSHOT_SCHEMA = 1

_IMPORT_EVENT_TYPES = {"fact_import", "matter_import", "edge_import"}


def _line(f: TextIO, obj: dict) -> None:
    f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def export_snapshot(
    index: MemoryIndex,
    ledger: MemoryHub | None,
    hard_rules: list[str],
    out_path: str | Path,
    include_turns: bool = False,
) -> dict[str, int]:
    """导出快照。返回各类型条数。

    ledger 可为 None（无 Memory Hub 时跳过 admin_event/turn 段）。墓碑数据天然不出现
    （Memory Index 级联已清、Memory Hub scan 跳过覆盖项）。
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts = {"hard_rule": 0, "fact": 0, "matter": 0, "edge": 0,
              "admin_event": 0, "turn": 0}

    facts = index.all_facts()
    matters = index.all_matters()
    edges: list[MatterEdge] = []
    for m in matters:
        edges.extend(index.get_edges(m.matter_id))

    with out_path.open("w", encoding="utf-8") as f:
        _line(f, {
            "type": "meta", "schema": SNAPSHOT_SCHEMA,
            "exported_at": datetime.now(UTC).isoformat(),
            "embed_model_id": getattr(index, "_embed_model_id", None) or "",
            "include_turns": include_turns,
        })
        for rule in hard_rules:
            _line(f, {"type": "hard_rule", "schema": SNAPSHOT_SCHEMA, "content": rule})
            counts["hard_rule"] += 1
        for fact in facts:
            rec = fact.model_dump(mode="json", exclude={"embedding"})
            _line(f, {"type": "fact", "schema": SNAPSHOT_SCHEMA, **rec})
            counts["fact"] += 1
        for m in matters:
            rec = m.model_dump(mode="json", exclude={"centroid", "embedding"})
            _line(f, {"type": "matter", "schema": SNAPSHOT_SCHEMA, **rec})
            counts["matter"] += 1
        for e in edges:
            _line(f, {"type": "edge", "schema": SNAPSHOT_SCHEMA,
                      **e.model_dump(mode="json")})
            counts["edge"] += 1
        if ledger is not None:
            for _key, event in ledger.scan_admin_events():
                # 不导出 import 事件本体——导入侧会为快照实体重新 journal，
                # 二者叠加会让事件重复膨胀（实体本身已在 fact/matter/edge 段）。
                if str(getattr(event.event_type, "value", event.event_type)) in _IMPORT_EVENT_TYPES:
                    continue
                _line(f, {"type": "admin_event", "schema": SNAPSHOT_SCHEMA,
                          **event.model_dump(mode="json")})
                counts["admin_event"] += 1
            if include_turns:
                for key, turn in ledger.scan_prefix(""):
                    _line(f, {"type": "turn", "schema": SNAPSHOT_SCHEMA, "key": key,
                              **turn.model_dump(mode="json")})
                    counts["turn"] += 1

    logger.info("snapshot_exported", path=str(out_path), **counts)
    return counts


def import_snapshot(
    index: MemoryIndex,
    ledger: MemoryHub | None,
    in_path: str | Path,
) -> dict[str, int]:
    """导入快照：按 id 幂等 merge、manual 压 auto、冲突非阻断标注。

    每个真正落库的实体同时以 *_IMPORT 管理事件入 Memory Hub journal（ledger 非 None 时），
    保证 full rebuild 后导入的记忆仍在（重建等价）。
    返回统计：imported_* / skipped_existing / conflicts / journal_events。
    """
    in_path = Path(in_path)
    stats = {"facts": 0, "matters": 0, "edges": 0, "hard_rules_seen": 0,
             "admin_events": 0, "turns_skipped": 0,
             "skipped_existing": 0, "conflicts": 0, "journal_events": 0}
    embedder = getattr(index, "_embedder", None)

    # 幂等防线：已有管理事件指纹集合——重复 import 不重复 journal
    existing_events: set[tuple] = set()
    if ledger is not None:
        try:
            for _k, ev in ledger.scan_admin_events():
                existing_events.add((
                    str(getattr(ev.event_type, "value", ev.event_type)),
                    ev.matter_id, ev.target_key,
                    json.dumps(ev.payload, sort_keys=True, ensure_ascii=False),
                ))
        except Exception as e:  # noqa: BLE001
            logger.warning("snapshot_import_event_scan_failed", error=str(e))

    def _journal(event_type: AdminEventType, matter_id: str, target_key: str,
                 payload: dict) -> None:
        if ledger is None:
            return
        fp = (str(getattr(event_type, "value", event_type)), matter_id, target_key,
              json.dumps(payload, sort_keys=True, ensure_ascii=False))
        if fp in existing_events:
            return
        ledger.append_admin_event(event_type, matter_id, target_key, **payload)  # payload 键已保证不与位置参数撞名（entity 包裹 / 历史事件原样）
        existing_events.add(fp)
        stats["journal_events"] += 1

    with in_path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec: dict[str, Any] = json.loads(raw)
            except ValueError as e:
                logger.warning("snapshot_import_bad_line", lineno=lineno, error=str(e))
                continue
            rtype = rec.pop("type", "")
            schema = rec.pop("schema", None)
            if rtype == "meta":
                if schema is not None and int(schema) > SNAPSHOT_SCHEMA:
                    raise ValueError(
                        f"snapshot schema {schema} > supported {SNAPSHOT_SCHEMA}; "
                        "upgrade BladeX before importing")
                continue

            if rtype == "fact":
                fact = Fact.model_validate(rec)
                existing = index.get_fact(fact.id)
                if existing is not None:
                    if existing.content == fact.content:
                        stats["skipped_existing"] += 1
                    else:
                        # 冲突：非阻断标注——保留本地，记录不覆盖（一致性语义）
                        stats["conflicts"] += 1
                        logger.warning("snapshot_import_fact_conflict",
                                       fact_id=fact.id,
                                       hint="local kept; imported content differs")
                    continue
                fact.embedding = None
                if embedder is not None:
                    try:
                        fact.embedding = embed_passage_compat(embedder, [fact.content])[0]
                    except Exception as e:  # noqa: BLE001 —— 无向量仍入 meta
                        logger.debug("snapshot_import_embed_skip", error=str(e))
                index.add_fact(fact)
                # 实体 dump 包一层 "entity"——Fact/Matter 自带 matter_id 等字段，
                # 展开成 **kwargs 会与 append_admin_event 位置参数冲突
                _journal(AdminEventType.FACT_IMPORT, "", fact.id,
                         {"entity": fact.model_dump(mode="json",
                                                    exclude={"embedding"})})
                stats["facts"] += 1

            elif rtype == "matter":
                matter = Matter.model_validate(rec)
                matter.centroid = None
                matter.embedding = None
                existing_m = index.get_matter(matter.matter_id)
                if existing_m is not None:
                    # manual 压 auto：本地 manual 不被快照 auto 覆盖；
                    # 快照 manual 可覆盖本地 auto（用户劳动优先）。
                    if (existing_m.origin == MatterOrigin.MANUAL
                            and matter.origin != MatterOrigin.MANUAL):
                        stats["skipped_existing"] += 1
                        continue
                    if existing_m.title == matter.title and existing_m.origin == matter.origin:
                        stats["skipped_existing"] += 1
                        continue
                index.add_matter(matter)
                _journal(AdminEventType.MATTER_IMPORT, matter.matter_id, "",
                         {"entity": matter.model_dump(
                             mode="json", exclude={"centroid", "embedding"})})
                stats["matters"] += 1

            elif rtype == "edge":
                edge = MatterEdge.model_validate(rec)
                existing_edges = index.get_edges(edge.matter_id)
                same_id = any(e.edge_id == edge.edge_id for e in existing_edges)
                if same_id:
                    stats["skipped_existing"] += 1
                    continue
                # manual 压 auto：目标已有本地 manual 边时，快照 auto 边不进
                manual = index.get_manual_edge_for_target(edge.target_key, edge.target_type)
                if manual is not None and edge.provenance != EdgeProvenance.MANUAL:
                    stats["skipped_existing"] += 1
                    continue
                index.add_edge(edge)
                _journal(AdminEventType.EDGE_IMPORT, edge.matter_id, edge.target_key,
                         {"entity": edge.model_dump(mode="json")})
                stats["edges"] += 1

            elif rtype == "hard_rule":
                # HardRule 是配置面（config/.env），不写存储——提示用户手动合并
                stats["hard_rules_seen"] += 1

            elif rtype == "admin_event":
                # 历史管理事件：re-journal（重放语义幂等），指纹去重防重复导入膨胀
                if ledger is not None:
                    ev = AdminEvent.model_validate(rec)
                    before = stats["journal_events"]
                    _journal(ev.event_type, ev.matter_id, ev.target_key, ev.payload)
                    if stats["journal_events"] > before:
                        stats["admin_events"] += 1

            elif rtype == "turn":
                # Memory Hub 整档导入不在 v1 范围（--full 导出仅作离线归档）
                stats["turns_skipped"] += 1

            else:
                logger.warning("snapshot_import_unknown_type", lineno=lineno,
                               rtype=rtype)

    logger.info("snapshot_imported", path=str(in_path), **stats)
    return stats
