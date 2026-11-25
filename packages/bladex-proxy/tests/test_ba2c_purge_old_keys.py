"""批 BA2c（MQ-A101，删除语义）：按项目预览 / 墓碑要覆盖旧键轮次。

缺陷：迁移后 vllm 新键 `p-c3e9…`，老轮次存旧键 `p-c4eb…`；`project_scope_preview` 直接拿存值比 ⇒
按新键清会**静默漏掉旧键轮次**。CLI `memory tombstone-project` 只走 `/admin`，所以真实 Hub 只有端点这一条路径。
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest
from bladex_core.project import set_project_key_map
from bladex_proxy import ledger_events as LE
from bladex_proxy.ledger_memory import tombstone_ledger_memory
from bladex_proxy.models import AdminEventType, Identity, SessionIdSource, Turn, TurnStatus
from bladex_proxy.project_purge import EmptyProjectRefused, project_scope_preview
from bladex_proxy.storage.memory_hub import MemoryHub

OLD, NEW, OTHER = "p-c4eb07cb6d90", "p-c3e9d5dd67aea764", "p-other0000001"
LEDGER = "ldg-ba2c00000001"


def _fresh() -> None:
    set_project_key_map({})
    LE._key_map_state.update(path=None, mtime=None)


@pytest.fixture(autouse=True)
def _clean():
    _fresh()
    yield
    _fresh()


def _put(hub: MemoryHub, seq: int, project: str, name: str = "vllm") -> str:
    ident = Identity(user_id="u1", agent_id="codex", session_id="s1", turn_index=seq,
                     session_id_source=SessionIdSource.FINGERPRINT,
                     project_id=project, project_name=name)
    key = ident.storage_key(f"{1800000000000 + seq}-{seq}")
    hub.put(key, Turn(identity=ident, model="m", status=TurnStatus.OK, response_text="好",
                      request_messages=[{"role": "user", "content": f"第 {seq} 轮"}],
                      ts=datetime(2026, 11, 2, 9, 0, seq, tzinfo=UTC),
                      session_id_source=SessionIdSource.FINGERPRINT))
    return key


@pytest.fixture
def hub(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    h = MemoryHub(data / "bladex_hub")
    h.open()
    h.keys_new = _put(h, 1, NEW)
    h.keys_old = _put(h, 2, OLD)
    h.keys_other = _put(h, 3, OTHER, name="other")
    h.append_admin_event(AdminEventType.LEDGER_SWITCH, "", agent_id="codex", project_id=OLD,
                         session_id="s1", turn_index=1, to_ledger_id=LEDGER)
    time.sleep(0.01)
    yield h
    h.close()


def _with_map(hub: MemoryHub) -> None:
    (Path(hub._path).parent / "project_key_map.json").write_text(
        json.dumps({"map": {OLD: NEW}}), encoding="utf-8")


def test_ba2c_preview_by_new_key_covers_old_key_turns(hub) -> None:
    """复现：映射文件在、进程表为空（新进程）⇒ 按新键预览两条都要在。"""
    _with_map(hub)
    row = project_scope_preview(hub, [NEW])[NEW]
    assert sorted(row["turn_keys"]) == sorted([hub.keys_new, hub.keys_old])
    assert row["turn_count"] == 2


def test_ba2c_preview_by_old_key_hits_the_same_group(hub) -> None:
    """用户手里可能还拿着旧键。"""
    _with_map(hub)
    row = project_scope_preview(hub, [OLD])[OLD]
    assert sorted(row["turn_keys"]) == sorted([hub.keys_new, hub.keys_old])


def test_ba2c_preview_ledger_count_counts_translated_scope(hub) -> None:
    """影响面里的账本数也要跟上：切换事件存的是旧键。"""
    _with_map(hub)
    assert project_scope_preview(hub, [NEW])[NEW]["ledgers"] == 1
    assert project_scope_preview(hub, [OLD])[OLD]["ledgers"] == 1


def test_ba2c_both_keys_given_turns_are_not_counted_twice(hub) -> None:
    """同时传新旧键：轮次只算一次（端点按行累加 total 并逐行打墓碑），第二个键不丢、给读数 0 + 指向。"""
    _with_map(hub)
    out = project_scope_preview(hub, [NEW, OLD])
    assert list(out) == [NEW, OLD]
    assert out[NEW]["turn_count"] == 2
    assert out[OLD]["turn_count"] == 0 and out[OLD]["turn_keys"] == [] and out[OLD]["same_as"] == NEW


def test_ba2c_without_map_preview_is_unchanged(hub) -> None:
    out = project_scope_preview(hub, [NEW, OLD])
    assert out[NEW]["turn_keys"] == [hub.keys_new] and out[OLD]["turn_keys"] == [hub.keys_old]
    assert out[NEW]["ledgers"] == 0 and out[OLD]["ledgers"] == 1
    assert set(out[NEW]) == {"turn_keys", "turn_count", "agents", "first_ts", "last_ts",
                             "project_names", "ledgers"}, "无映射时返回形状逐字不变"


def test_ba2c_other_project_untouched(hub) -> None:
    _with_map(hub)
    assert hub.keys_other not in project_scope_preview(hub, [NEW])[NEW]["turn_keys"]


def test_ba2c_tombstone_after_preview_hides_both_groups(hub) -> None:
    _with_map(hub)
    row = project_scope_preview(hub, [NEW])[NEW]
    tombstone_ledger_memory(hub, None, row["turn_keys"])
    visible = {k for k, _ts in hub.scan_meta()}      # 墓碑后不可见的口径 = 扫描面（`get` 按键直取不滤）
    assert hub.keys_new not in visible and hub.keys_old not in visible
    assert hub.keys_other in visible
    assert project_scope_preview(hub, [NEW])[NEW]["turn_count"] == 0


@pytest.mark.parametrize("ids", [[""], ["   "], [NEW, ""], []])
def test_ba2c_global_still_refused(hub, ids: list[str]) -> None:
    _with_map(hub)
    with pytest.raises(EmptyProjectRefused):
        project_scope_preview(hub, ids)


def test_ba2c_admin_endpoint_real_run_covers_both_groups(tmp_path, monkeypatch, isolated_home) -> None:
    """CLI `memory tombstone-project` 的唯一路径：`/admin`，真跑（后台 job）后两组都不可见。"""
    from bladex_proxy.config import ProxyConfig
    from bladex_proxy.server import create_app
    from fastapi.testclient import TestClient

    data = tmp_path / "d"
    data.mkdir()
    hub_path = data / "bladex_hub"
    h = MemoryHub(hub_path)
    h.open()
    new_key, old_key, other_key = _put(h, 1, NEW), _put(h, 2, OLD), _put(h, 3, OTHER, name="other")
    h.close()
    (data / "project_key_map.json").write_text(json.dumps({"map": {OLD: NEW}}), encoding="utf-8")
    _fresh()

    monkeypatch.setenv("BLADEX_AUTH_ENABLED", "false")
    monkeypatch.setenv("BLADEX_ROCKSDB_PATH", str(hub_path))
    monkeypatch.setenv("BLADEX_INDEX_PATH", str(tmp_path / "index"))
    with TestClient(create_app(ProxyConfig())) as c:
        dry = c.post("/admin/turns/tombstone-by-project", json={"project_ids": [NEW], "dry_run": True}).json()
        assert dry["total_turns"] == 2
        sub = c.post("/admin/turns/tombstone-by-project", json={"project_ids": [NEW], "dry_run": False}).json()
        assert sub["status"] == "submitted"
        for _ in range(200):
            job = c.get(f"/admin/jobs/{sub['job_id']}").json()["job"]
            if job["state"] != "running":
                break
            time.sleep(0.02)
        assert job["state"] == "done", job
    ro = MemoryHub(hub_path, read_only=True)
    ro.open()
    try:
        visible = {k for k, _ts in ro.scan_meta()}
        assert new_key not in visible and old_key not in visible
        assert other_key in visible
    finally:
        ro.close()
