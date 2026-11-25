"""批 BA2c（MQ-A101 第二处）：行为画像里同一项目的新旧键算一个项目。

翻译放在 `_behavior_metas`（全量与增量的汇合点），不放在 fold：增量状态 `.flash_state.json` 存着原值，
落盘的旧键行不用迁移就归到新键；回滚（删映射文件）也不留痕。
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from bladex_core.project import canonical_project_id, set_project_key_map
from bladex_proxy import ledger_events as LE
from bladex_proxy.flash_daemon import FlashDaemon, hub_behavior_source
from bladex_proxy.models import Identity, SessionIdSource, Turn, TurnStatus
from bladex_proxy.storage.memory_hub import MemoryHub

NOW = datetime(2026, 11, 2, 12, 0, tzinfo=UTC)
OLD, NEW = "p-c4eb07cb6d90", "p-c3e9d5dd67aea764"


def _fresh() -> None:
    set_project_key_map({})
    LE._key_map_state.update(path=None, mtime=None)


@pytest.fixture(autouse=True)
def _clean():
    _fresh()
    yield
    _fresh()


def _row(minutes: int, pid: str, seq: int) -> tuple[str, dict]:
    ts = NOW - timedelta(minutes=minutes)
    key = f"local/codex/fp:a/{int(ts.timestamp() * 1000)}-{seq:08d}"
    return key, {"ts": ts.strftime("%Y-%m-%dT%H:%M:%S.%fZ"), "project_id": pid, "project_name": "vllm",
                 "project_root": "/r/vllm", "auxiliary": False, "subagent": False}


class _Src:
    def __init__(self, rows) -> None:  # noqa: ANN001
        self.rows = list(rows)

    def __call__(self) -> dict:
        return {"rows": list(self.rows), "tombstone_hash": "t0", "ledger_events": [],
                "tombstoned_ledgers": {}, "tool_stats": {}, "habits": [], "hard_rules": [], "eligible": None}


def _daemon(tmp_path: Path, src) -> FlashDaemon:  # noqa: ANN001
    d = FlashDaemon(root=str(tmp_path), principal="local", ledger_source=lambda: ({}, {}),
                    emit_admin_event=lambda t, p: None, behavior_source=src,
                    behavior_every_s=0.0, tz_offset_s=8 * 3600)
    d.behavior_clock = lambda: NOW
    return d


def _projects(d: FlashDaemon) -> dict[str, dict]:
    data = json.load(open(d.behavior_path(), encoding="utf-8"))
    return {p["project_id"]: p for p in data["projects"]}


def test_ba2c_old_and_new_key_turns_are_one_project(tmp_path) -> None:
    set_project_key_map({OLD: NEW})
    d = _daemon(tmp_path, _Src([_row(30, OLD, 1), _row(20, NEW, 2)]))
    d.run_once()
    assert list(_projects(d)) == [NEW]


def test_ba2c_without_map_two_projects_as_before(tmp_path) -> None:
    d = _daemon(tmp_path, _Src([_row(30, OLD, 1), _row(20, NEW, 2)]))
    d.run_once()
    assert sorted(_projects(d)) == sorted([OLD, NEW])


def test_ba2c_persisted_old_key_state_merges_without_migration(tmp_path) -> None:
    """存量：迁移前落盘的增量状态只有旧键行。迁移 + 重启后新轮次进来，仍是一个项目。"""
    src = _Src([_row(30, OLD, 1)])
    d = _daemon(tmp_path, src)
    d.run_once()
    assert list(_projects(d)) == [OLD]

    set_project_key_map({OLD: NEW})
    src.rows = [_row(30, OLD, 1), _row(20, NEW, 2)]
    d2 = _daemon(tmp_path, src)                     # 新进程：状态从 .flash_state.json 读回
    assert d2._bh_state["projects"][0][0] == OLD    # 存量确实是旧键，增量不会重折它
    d2.push()
    d2.run_once()
    assert list(_projects(d2)) == [NEW]


def test_ba2c_state_on_disk_keeps_raw_keys_so_rollback_is_clean(tmp_path) -> None:
    set_project_key_map({OLD: NEW})
    src = _Src([_row(30, OLD, 1)])
    d = _daemon(tmp_path, src)
    d.run_once()
    assert d._bh_state["projects"][0][0] == OLD
    set_project_key_map({})
    d.push()
    d._materialize_behavior()
    assert list(_projects(d)) == [OLD]


def test_ba2c_behavior_source_loads_map_before_reading(tmp_path) -> None:
    """画像轨道与账本轨道各自节流，这一 tick 可能没走过真实 Hub 的 `load_ledger_events`。"""
    data = tmp_path / "data"
    data.mkdir()
    (data / "project_key_map.json").write_text(json.dumps({"map": {OLD: NEW}}), encoding="utf-8")
    hub = MemoryHub(data / "bladex_hub")
    hub.open()
    ident = Identity(user_id="local", agent_id="codex", session_id="s1", turn_index=1,
                     session_id_source=SessionIdSource.FINGERPRINT, project_id=OLD, project_name="vllm")
    hub.put(ident.storage_key("1800000000001-1"), Turn(
        identity=ident, model="m", status=TurnStatus.OK, response_text="好",
        request_messages=[{"role": "user", "content": "x"}], ts=NOW,
        session_id_source=SessionIdSource.FINGERPRINT))
    try:
        assert canonical_project_id(OLD) == OLD
        hub_behavior_source(hub)()
        assert canonical_project_id(OLD) == NEW
    finally:
        hub.close()
