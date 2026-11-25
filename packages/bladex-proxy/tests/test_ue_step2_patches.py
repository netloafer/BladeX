"""批 UE 步骤 2：两处补丁（拍板 (109)）。

| 用例 | 判据 |
|---|---|
| `test_running_hub_scan_meta_snapshot_raises_on_must_understand` | Hub 正常打开后再写入 must_understand 未知事件，`scan_meta_snapshot()` 抛 `HubRequiresNewerVersion`（不被记成脏值） |
| `test_swallow_sites_let_refusal_through` | 服务路径上包着 admin 事件读取的宽 `except Exception` 都先放行该异常（源码守卫） |
| `test_snapshot_import_checks_before_writing` | 快照里先有正常 fact / 已知事件、后有标志事件：导入抛错，Hub 事件数与 Index fact 数都不变 |
"""

from __future__ import annotations

import re
from pathlib import Path

import msgpack
import pytest
from bladex_core.fact import Fact
from bladex_proxy.models import AdminEventType
from bladex_proxy.snapshot import import_snapshot
from bladex_proxy.storage.memory_hub import (
    HubRequiresNewerVersion,
    MemoryHub,
    _generate_entry_id,
)


def _write_flag_event(hub: MemoryHub) -> None:
    hub.db[f"admin_event/{_generate_entry_id()}".encode()] = msgpack.packb(
        {"event_type": "task_card_set", "matter_id": "", "target_key": "t",
         "ts": "2026-11-15T00:00:00+00:00", "payload": {"must_understand": True}},
        use_bin_type=True)


def test_running_hub_scan_meta_snapshot_raises_on_must_understand(tmp_path) -> None:
    hub = MemoryHub(str(tmp_path / "h"))
    hub.open()
    hub.append_admin_event(AdminEventType.HARD_RULES_SET, "", "", rules=["MUST a"])
    hub.scan_meta_snapshot()  # 打开时干净：正常
    _write_flag_event(hub)
    with pytest.raises(HubRequiresNewerVersion, match="task_card_set"):
        hub.scan_meta_snapshot()
    hub.close()


def test_swallow_sites_let_refusal_through() -> None:
    root = Path(__file__).resolve().parents[1] / "bladex_proxy"
    sites = {
        "storage/memory_hub.py": 1, "identity_registry.py": 1, "server/admin_api.py": 1,
        "admin_read.py": 1, "agency/runtime.py": 1, "storage/memory_index.py": 3,
        "flash_daemon.py": 2, "snapshot.py": 1,
    }
    for rel, n in sites.items():
        src = (root / rel).read_text(encoding="utf-8")
        hits = len(re.findall(r"except HubRequiresNewerVersion:[^\n]*\n\s+raise", src))
        assert hits >= n, (rel, hits, n)


class _Index:
    _embed_model_id = ""
    _embedder = None

    def __init__(self) -> None:
        self.facts: dict[str, Fact] = {}

    def get_fact(self, fid: str) -> Fact | None:
        return self.facts.get(fid)

    def add_fact(self, fact: Fact) -> None:
        self.facts[fact.id] = fact


def test_snapshot_import_checks_before_writing(tmp_path) -> None:
    snap = tmp_path / "s.jsonl"
    fact = Fact(id="f1", content="hello").model_dump(mode="json", exclude={"embedding"})
    lines = [
        {"type": "meta", "schema": 1},
        {"type": "fact", "schema": 1, **fact},
        {"type": "admin_event", "schema": 1, "event_type": "hard_rules_set",
         "matter_id": "", "target_key": "", "payload": {"rules": ["MUST z"]}},
        {"type": "admin_event", "schema": 1, "event_type": "task_card_set",
         "matter_id": "", "target_key": "", "payload": {"must_understand": True}},
    ]
    import json
    snap.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
    hub = MemoryHub(str(tmp_path / "d"))
    hub.open()
    idx = _Index()
    before = len(list(hub.scan_admin_event_dicts(include_unknown=True)))
    with pytest.raises(ValueError, match=r"line 4.*task_card_set"):
        import_snapshot(idx, hub, snap)
    assert len(list(hub.scan_admin_event_dicts(include_unknown=True))) == before
    assert idx.facts == {}
    hub.close()
