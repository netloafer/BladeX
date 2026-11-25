"""批 UE：未知管理事件类型——全部读取点跳过并告警 + `must_understand` 拒启动（拍板 (107)）。

| 用例 | 判据 |
|---|---|
| `test_unknown_skipped_everywhere_and_known_replay_unchanged` | 无标志副本：三种打开方式都能开；scan / latest / meta 快照 / handoff 与没有该事件时逐字相同；每进程汇总告警每类恰一条 |
| `test_must_understand_refuses_open` | 有标志副本：各打开方式拒启动，报错点名类型与 key |
| `test_snapshot_roundtrip_preserves_unknown` | 快照导出再导入：未知事件原样还在（类型、载荷） |
| `test_snapshot_import_refuses_must_understand` | 快照里的 must_understand 未知事件：导入拒绝 |
"""

from __future__ import annotations

import msgpack
import pytest
import structlog
from bladex_proxy.models import KNOWN_ADMIN_EVENT_TYPES, AdminEventType
from bladex_proxy.snapshot import export_snapshot, import_snapshot
from bladex_proxy.storage.memory_hub import (
    HubRequiresNewerVersion,
    MemoryHub,
    _generate_entry_id,
)


def _seed(path: str, unknown: dict | None) -> None:
    hub = MemoryHub(path)
    hub.open()
    hub.append_admin_event(AdminEventType.HARD_RULES_SET, "", "", rules=["MUST a"])
    if unknown is not None:
        hub.db[f"admin_event/{_generate_entry_id()}".encode()] = msgpack.packb(
            {"event_type": "task_card_set", "matter_id": "", "target_key": "t",
             "ts": "2026-11-15T00:00:00+00:00", **unknown}, use_bin_type=True)
    hub.append_admin_event(AdminEventType.HARD_RULES_SET, "", "", rules=["MUST b"])
    hub.close()


def _view(hub: MemoryHub) -> list:
    return [(ev.event_type.value, ev.payload) for _k, ev in hub.scan_admin_events()]


def test_unknown_skipped_everywhere_and_known_replay_unchanged(tmp_path) -> None:
    _seed(str(tmp_path / "plain"), None)
    _seed(str(tmp_path / "unk"), {"payload": {"x": 1}})
    base = MemoryHub(str(tmp_path / "plain"), read_only=True)
    base.open()
    want = _view(base)
    base.close()
    for kw in ({}, {"read_only": True}):
        with structlog.testing.capture_logs() as logs:
            hub = MemoryHub(str(tmp_path / "unk"), **kw)
            hub.open()
            assert _view(hub) == want
            assert _view(hub) == want  # 再读一遍不重复告警
            latest = hub.latest_admin_event(AdminEventType.HARD_RULES_SET)
            assert latest is not None and latest[1].payload == {"rules": ["MUST b"]}
            snap = hub.scan_meta_snapshot()
            assert [e.event_type.value for _k, e in snap.admin_events] == ["hard_rules_set"] * 2
            assert [d["event_type"] for _k, d in hub.scan_admin_event_dicts()] == ["hard_rules_set"] * 2
            assert hub.unknown_admin_event_count == 1
            hub.close()
        warns = [x for x in logs if x["event"] == "hub_admin_event_unknown_type_skipped"]
        assert len(warns) == 1 and warns[0]["event_type"] == "task_card_set" and warns[0]["count"] == 1
    assert "task_card_set" not in KNOWN_ADMIN_EVENT_TYPES


@pytest.mark.parametrize("kw", [{}, {"read_only": True}])
def test_must_understand_refuses_open(tmp_path, kw) -> None:
    _seed(str(tmp_path / "must"), {"payload": {"must_understand": True, "target": "x"}})
    hub = MemoryHub(str(tmp_path / "must"), **kw)
    with pytest.raises(HubRequiresNewerVersion, match="task_card_set"):
        hub.open()


def test_must_understand_false_is_skipped(tmp_path) -> None:
    _seed(str(tmp_path / "f"), {"payload": {"must_understand": False}})
    hub = MemoryHub(str(tmp_path / "f"))
    hub.open()
    assert len(_view(hub)) == 2
    hub.close()


class _Index:
    """快照导出 / 导入用到的最小 Index 桩。"""

    _embed_model_id = ""
    _embedder = None

    def all_facts(self) -> list:
        return []

    def all_matters(self) -> list:
        return []

    def get_edges(self, _m: str) -> list:
        return []


def test_snapshot_roundtrip_preserves_unknown(tmp_path) -> None:
    _seed(str(tmp_path / "src"), {"payload": {"x": 1, "nested": {"k": "v"}}})
    src = MemoryHub(str(tmp_path / "src"))
    src.open()
    out = tmp_path / "snap.jsonl"
    export_snapshot(_Index(), src, [], out)
    src.close()
    dst = MemoryHub(str(tmp_path / "dst"))
    dst.open()
    import_snapshot(_Index(), dst, out)
    raws = [d for _k, d in dst.scan_admin_event_dicts(include_unknown=True)
            if d["event_type"] == "task_card_set"]
    assert len(raws) == 1 and raws[0]["payload"] == {"x": 1, "nested": {"k": "v"}}
    import_snapshot(_Index(), dst, out)  # 幂等：再导一次不重复
    assert len([d for _k, d in dst.scan_admin_event_dicts(include_unknown=True)
                if d["event_type"] == "task_card_set"]) == 1
    dst.close()


def test_snapshot_import_refuses_must_understand(tmp_path) -> None:
    snap = tmp_path / "s.jsonl"
    snap.write_text(
        '{"type": "meta", "schema": 1}\n'
        '{"type": "admin_event", "schema": 1, "event_type": "task_card_set", "matter_id": "", '
        '"target_key": "", "payload": {"must_understand": true}}\n', encoding="utf-8")
    dst = MemoryHub(str(tmp_path / "d"))
    dst.open()
    with pytest.raises(ValueError, match="task_card_set"):
        import_snapshot(_Index(), dst, snap)
    dst.close()
