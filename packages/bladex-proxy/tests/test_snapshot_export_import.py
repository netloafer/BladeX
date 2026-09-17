"""Beta T12 快照导出/导入验收。

核心剧本（任务卡验收原文）：export → 清空 Memory Index → import → rebuild，记忆等价。
另覆盖：幂等重复导入 / manual 压 auto / 冲突非阻断 / 墓碑数据不出现 /
schema 版本拒绝 / journal 指纹去重。
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from bladex_core.fact import Fact
from bladex_core.matter import (
    EdgeProvenance,
    EdgeTargetType,
    Matter,
    MatterEdge,
    MatterOrigin,
)
from bladex_proxy.models import Identity, TombstoneTargetType, Turn, TurnStatus
from bladex_proxy.snapshot import SNAPSHOT_SCHEMA, export_snapshot, import_snapshot
from bladex_proxy.storage.memory_hub import MemoryHub
from bladex_proxy.storage.memory_index import MemoryIndex


class MockEmbedder:
    def embed(self, texts):
        out = []
        for t in texts:
            vec = [0.0] * 64
            vec[hash(t) % 64] = 1.0
            out.append(vec)
        return out

    @property
    def available(self):
        return True


def _open_index(base: Path, name: str, embedder=None) -> MemoryIndex:
    index = MemoryIndex(base / name, embedder=embedder, read_only=False)
    index.open()
    return index


def _seed(index: MemoryIndex) -> None:
    index.add_fact(Fact(id="f1", content="用户偏好中文回复", kind="preference",
                     source_user_id="u1"))
    index.add_fact(Fact(id="f2", content="T11 验收已通过", kind="event",
                     source_user_id="u1"))
    index.add_matter(Matter(matter_id="m1", title="Beta 发布",
                         origin=MatterOrigin.MANUAL))
    index.add_edge(MatterEdge(edge_id="e1", matter_id="m1",
                           target_type=EdgeTargetType.FACT, target_key="f1",
                           provenance=EdgeProvenance.MANUAL))


def test_export_then_import_roundtrip_and_rebuild_equivalence():
    """export → 清空 Memory Index → import → full rebuild，记忆等价（含 journal 存活）。"""
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        ledger = MemoryHub(base / "ledger")
        ledger.open()
        index = _open_index(base, "index", MockEmbedder())
        _seed(index)

        snap = base / "snap.jsonl"
        counts = export_snapshot(index, ledger, ["MUST be polite"], snap)
        assert counts["fact"] == 2 and counts["matter"] == 1 and counts["edge"] == 1
        assert counts["hard_rule"] == 1

        # 清空 Memory Index（模拟迁移到新机器 / 灾难恢复）
        index.clear()
        assert index.fact_count() == 0

        stats = import_snapshot(index, ledger, snap)
        assert stats["facts"] == 2 and stats["matters"] == 1 and stats["edges"] == 1
        assert stats["journal_events"] == 4  # 2 fact + 1 matter + 1 edge

        # 导入后即等价
        assert index.get_fact("f1") is not None
        assert index.get_matter("m1") is not None
        assert len(index.get_edges("m1")) == 1

        # full rebuild（Memory Hub 无会话数据）→ journal 重放，导入的记忆存活
        index.rebuild_from_hub(ledger, full=True)
        assert index.get_fact("f1") is not None, "fact_import journal 未被重放"
        assert index.get_fact("f2") is not None
        assert index.get_matter("m1") is not None
        edges = index.get_edges("m1")
        assert any(e.target_key == "f1" for e in edges)
        index.close()
        ledger.close()


def test_import_idempotent():
    """重复导入同一快照：全 skip、journal 不膨胀。"""
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        ledger = MemoryHub(base / "ledger")
        ledger.open()
        index = _open_index(base, "index", MockEmbedder())
        _seed(index)
        snap = base / "snap.jsonl"
        export_snapshot(index, ledger, [], snap)

        s1 = import_snapshot(index, ledger, snap)  # 实体已在 → 全 skip
        assert s1["facts"] == 0 and s1["matters"] == 0 and s1["edges"] == 0
        assert s1["skipped_existing"] >= 3  # manual matter 同 title 同 origin 也 skip
        assert s1["journal_events"] == 0

        events_before = len(list(ledger.scan_admin_events()))
        s2 = import_snapshot(index, ledger, snap)
        assert len(list(ledger.scan_admin_events())) == events_before
        assert s2["journal_events"] == 0
        index.close()
        ledger.close()


def test_manual_beats_auto_on_import():
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        index = _open_index(base, "index")
        # 本地 manual matter
        index.add_matter(Matter(matter_id="m1", title="本地手动版",
                             origin=MatterOrigin.MANUAL))
        snap = base / "snap.jsonl"
        with snap.open("w") as f:
            f.write(json.dumps({"type": "meta", "schema": SNAPSHOT_SCHEMA}) + "\n")
            # 快照里的 auto 版本不得覆盖本地 manual
            f.write(json.dumps({"type": "matter", "schema": SNAPSHOT_SCHEMA,
                                "matter_id": "m1", "title": "快照自动版",
                                "origin": "auto"}) + "\n")
        stats = import_snapshot(index, None, snap)
        assert stats["skipped_existing"] == 1
        assert index.get_matter("m1").title == "本地手动版"
        index.close()


def test_fact_conflict_nonblocking():
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        index = _open_index(base, "index")
        index.add_fact(Fact(id="f1", content="本地内容", source_user_id="u1"))
        snap = base / "snap.jsonl"
        with snap.open("w") as f:
            f.write(json.dumps({"type": "meta", "schema": SNAPSHOT_SCHEMA}) + "\n")
            f.write(json.dumps({"type": "fact", "schema": SNAPSHOT_SCHEMA,
                                "id": "f1", "content": "快照不同内容",
                                "source_user_id": "u1"}) + "\n")
        stats = import_snapshot(index, None, snap)
        assert stats["conflicts"] == 1
        assert index.get_fact("f1").content == "本地内容"  # 保留本地
        index.close()


def test_tombstoned_data_not_exported():
    """被 forget 的 fact（墓碑 + 级联）不出现在快照里。"""
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        ledger = MemoryHub(base / "ledger")
        ledger.open()
        index = _open_index(base, "index", MockEmbedder())
        _seed(index)
        # forget f2
        ledger.append_tombstone(target_type=TombstoneTargetType.FACT, target_key="f2")
        index.delete_fact("f2")

        snap = base / "snap.jsonl"
        counts = export_snapshot(index, ledger, [], snap)
        assert counts["fact"] == 1
        content = snap.read_text()
        assert "f2" not in content
        index.close()
        ledger.close()


def test_schema_version_rejected():
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        index = _open_index(base, "index")
        snap = base / "snap.jsonl"
        snap.write_text(json.dumps({"type": "meta", "schema": 99}) + "\n")
        with pytest.raises(ValueError, match="schema 99"):
            import_snapshot(index, None, snap)
        index.close()


def test_full_export_includes_turns_but_import_skips():
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        ledger = MemoryHub(base / "ledger")
        ledger.open()
        identity = Identity(user_id="u1", agent_id="a", session_id="s")
        ledger.put(identity.storage_key("0001-0"),
               Turn(identity=identity, model="m",
                    request_messages=[{"role": "user", "content": "hi"}],
                    response_text="hello", status=TurnStatus.OK))
        index = _open_index(base, "index")
        snap = base / "snap.jsonl"
        counts = export_snapshot(index, ledger, [], snap, include_turns=True)
        assert counts["turn"] == 1

        p2b = _open_index(base, "p2b")
        stats = import_snapshot(p2b, None, snap)
        assert stats["turns_skipped"] == 1
        index.close()
        p2b.close()
        ledger.close()
