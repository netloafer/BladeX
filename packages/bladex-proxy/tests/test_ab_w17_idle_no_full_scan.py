"""批 AB · MQ-W17 ①②③ + MQ-V41。用例随各 commit 增加。

| 用例 | 判据 |
|---|---|
| `test_unknown_event_skipped_and_counted` | 未知类型：其后事件仍可读，计数 = 1，重复扫描不重复计 |
| `test_bad_known_event_still_raises` | 已知类型缺字段：照旧抛，不静默 |
| `test_scan_admin_events_stops_at_prefix` | 只读 `admin_event/` 段（不碰段外的 value） |
| `test_snapshot_meta_and_latest_share_the_skip` | `scan_meta_snapshot` / `latest_admin_event` 同一语义 |
| `test_metrics_exposes_unknown_count` | `/metrics` gauge 读得到（不只日志） |
| `test_cursor_scan_matches_full_scan` | 多轮写入 / 消费 / 新 session：游标与全扫的待消费、`_prev_of` 逐字相同 |
| `test_cursor_scan_does_not_read_old_values` | 游标趟只解新 key 的 value；不调 `scan_meta` |
| `test_late_old_entry_found_by_reconcile` | 回灌（旧 entry_id 晚到）：游标看不见，对账打 `index_scan_cursor_missed` 并补上 |
| `test_reconcile_silent_when_nothing_missed` | 排在游标之后的新轮次不算漏（阴性对照） |
| `test_tombstone_forces_full_scan` | 墓碑让前驱链改变 ⇒ 与全扫相同 |
| `test_cursor_scan_filters_tombstoned_new_key` | 游标趟同样过滤墓碑 |
| `test_new_session_and_slash_in_session` | 新 session · session 名含斜杠 · 内部键夹在中间 |
| `test_full_and_filter_keep_full_scan` | 带过滤 / unconsume 后状态作废 |
| `test_profile_docs_short_circuit` | 输入不变不重写；fact / profile / rulefile 任一变化重写，下一轮回到短路 |
| `test_profile_docs_date_bucket` | 只有日期变 ⇒ 重写（7 天窗口滑动） |
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import msgpack
import pytest
from bladex_proxy.models import AdminEvent, AdminEventType
from bladex_proxy.storage.memory_hub import MemoryHub


def _hub(tmp: Path) -> MemoryHub:
    h = MemoryHub(tmp / "hub")
    h.open()
    return h


def _put_raw(h: MemoryHub, entry: str, data: dict[str, Any]) -> None:
    h.db[f"admin_event/{entry}".encode()] = msgpack.packb(data, use_bin_type=True)


def _ev(etype: str = "matter_rename", **kw: Any) -> dict[str, Any]:
    return {"event_type": etype, "matter_id": "m1", "target_key": "",
            "ts": "2026-11-02T00:00:00+00:00", "payload": {}, **kw}


def test_unknown_event_skipped_and_counted(tmp_path: Path) -> None:
    h = _hub(tmp_path)
    _put_raw(h, "0001", _ev())
    _put_raw(h, "0002", _ev("from_the_future"))
    _put_raw(h, "0003", _ev("matter_merge"))
    got = [k for k, _ in h.scan_admin_events()]
    assert got == ["admin_event/0001", "admin_event/0003"], "未知类型之后的事件必须还能读到"
    assert h.unknown_admin_event_count == 1
    list(h.scan_admin_events())
    assert h.unknown_admin_event_count == 1, "同一条多次扫描只算一次"
    h.close()


def test_bad_known_event_still_raises(tmp_path: Path) -> None:
    h = _hub(tmp_path)
    _put_raw(h, "0001", {"event_type": "matter_rename"})    # 缺 matter_id
    with pytest.raises(ValueError):
        list(h.scan_admin_events())
    h.close()


def test_scan_admin_events_stops_at_prefix(tmp_path: Path) -> None:
    h = _hub(tmp_path)
    _put_raw(h, "0001", _ev())
    # 段前段后各放一条解不开的值：全库遍历才会撞上（修前的形状靠 startswith 过滤，读取不受影响，
    # 但这里要钉的是**不读**——用计数迭代器量读了多少行）
    h.db[b"aaa_before/x/y/1"] = msgpack.packb({"ts": ""}, use_bin_type=True)
    h.db[b"zzz_after"] = b"\xc1"
    for i in range(50):
        h.db[f"local/a/s/{i:04d}".encode()] = msgpack.packb({"ts": ""}, use_bin_type=True)
    real = h.db

    class Counting:
        n = 0

        def items(self, **kw: Any) -> Any:
            for row in real.items(**kw):
                Counting.n += 1
                yield row

        def __getattr__(self, name: str) -> Any:
            return getattr(real, name)

    h._db = Counting()     # type: ignore[assignment]
    try:
        assert len(list(h.scan_admin_events())) == 1
        assert Counting.n <= 2, f"读了 {Counting.n} 行：扫描没有停在 admin_event/ 段尾"
    finally:
        h._db = real
    h.close()


def test_snapshot_meta_and_latest_share_the_skip(tmp_path: Path) -> None:
    h = _hub(tmp_path)
    _put_raw(h, "0001", _ev())
    _put_raw(h, "0002", _ev("from_the_future"))
    snap = h.scan_meta_snapshot()
    assert [k for k, _ in snap.scan_admin_events()] == ["admin_event/0001"]
    assert h.unknown_admin_event_count == 1
    found = h.latest_admin_event(AdminEventType("matter_rename"))
    assert found is not None and isinstance(found[1], AdminEvent)
    h.close()


def test_metrics_exposes_unknown_count() -> None:
    src = (Path(__file__).parents[1] / "bladex_proxy/server/app_factory.py").read_text("utf-8")
    assert 'set_gauge("bladex_hub_unknown_admin_events"' in src
    assert "unknown_admin_event_count" in src


# ── W17 ②：session 游标 ──────────────────────────────────────────────────────

def _turn(h: MemoryHub, key: str, ts: str = "2026-11-02T00:00:00") -> None:
    h.db[key.encode()] = msgpack.packb({"ts": ts}, use_bin_type=True)


def _index(tmp: Path) -> Any:
    from bladex_proxy.storage.memory_index import MemoryIndex

    idx = MemoryIndex(tmp / "idx", embedder=None)
    idx.open()
    return idx


def _assert_same_as_full(idx: Any, h: MemoryHub) -> list[tuple[str, str]]:
    batch, prev = idx._scan_with_cursor(h)
    full_batch, full_prev, _, _ = idx._scan_all_turn_keys(h)
    assert batch == full_batch
    assert prev == full_prev
    return batch


def test_cursor_scan_matches_full_scan(tmp_path: Path) -> None:
    h, idx = _hub(tmp_path), _index(tmp_path)
    for s in ("s1", "s2"):
        for e in range(3):
            _turn(h, f"local/a/{s}/{e:04d}", f"2026-11-02T00:0{e}:00")
    assert len(_assert_same_as_full(idx, h)) == 6       # 首趟全扫建状态
    idx._mark_consumed("local/a/s1/0000")
    idx._mark_consumed("local/a/s2/0001")               # 消费可以有洞（失败重试）
    assert len(_assert_same_as_full(idx, h)) == 4
    _turn(h, "local/a/s1/0003", "2026-11-02T00:09:00")
    _turn(h, "local/b/s3/0000", "2026-11-02T00:08:00")  # 新 agent + 新 session
    batch = _assert_same_as_full(idx, h)
    assert batch[-1][1] == "local/a/s1/0003" and idx._scan_state.rounds == 2
    for _t, k in batch:
        idx._mark_consumed(k)
    assert _assert_same_as_full(idx, h) == []
    assert _assert_same_as_full(idx, h) == []
    idx.close()
    h.close()


def test_cursor_scan_does_not_read_old_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    h, idx = _hub(tmp_path), _index(tmp_path)
    for e in range(5):
        _turn(h, f"local/a/s1/{e:04d}")
    idx._scan_with_cursor(h)

    def no_full_scan(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("游标趟不许全扫")

    monkeypatch.setattr(h, "scan_meta", no_full_scan)
    decoded: list[Any] = []
    real = msgpack.unpackb
    monkeypatch.setattr("bladex_proxy.storage.memory_hub.msgpack.unpackb",
                        lambda *a, **k: (decoded.append(1), real(*a, **k))[1])
    idx._scan_with_cursor(h)
    assert decoded == [], "没有新 key 时一条 value 都不解"
    _turn(h, "local/a/s1/0009")
    idx._scan_with_cursor(h)
    assert len(decoded) == 1
    idx.close()
    h.close()


def test_late_old_entry_found_by_reconcile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import structlog.testing

    h, idx = _hub(tmp_path), _index(tmp_path)
    for e in (0, 5):
        _turn(h, f"local/a/s1/{e:04d}")
    idx._scan_with_cursor(h)
    for k in ("local/a/s1/0000", "local/a/s1/0005"):
        idx._mark_consumed(k)
    _turn(h, "local/a/s1/0003")                         # overflow 回灌：旧 entry_id 晚到
    assert idx._scan_with_cursor(h)[0] == [], "游标看不见它（这正是要对账的原因）"
    monkeypatch.setenv("BLADEX_SCAN_RECONCILE_ROUNDS", "1")
    with structlog.testing.capture_logs() as logs:
        batch, prev = idx._scan_with_cursor(h)
    assert [k for _t, k in batch] == ["local/a/s1/0003"]
    assert prev["local/a/s1/0005"] == "local/a/s1/0003"
    ev = [e for e in logs if e["event"] == "index_scan_cursor_missed"]
    assert ev and ev[0]["count"] == 1
    idx.close()
    h.close()


def test_reconcile_silent_when_nothing_missed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import structlog.testing

    h, idx = _hub(tmp_path), _index(tmp_path)
    _turn(h, "local/a/s1/0000")
    idx._scan_with_cursor(h)
    _turn(h, "local/a/s1/0001")                         # 排在游标之后的新轮次不算漏
    monkeypatch.setenv("BLADEX_SCAN_RECONCILE_S", "0")
    with structlog.testing.capture_logs() as logs:
        idx._scan_with_cursor(h)
    assert not [e for e in logs if e["event"] == "index_scan_cursor_missed"]
    idx.close()
    h.close()


def test_tombstone_forces_full_scan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    h, idx = _hub(tmp_path), _index(tmp_path)
    for e in range(3):
        _turn(h, f"local/a/s1/{e:04d}")
    idx._scan_with_cursor(h)
    monkeypatch.setattr(h, "_tombstoned_keys", lambda: {"local/a/s1/0001"})
    _assert_same_as_full(idx, h)
    assert "local/a/s1/0001" not in dict((k, t) for t, k in idx._scan_with_cursor(h)[0])
    idx.close()
    h.close()


def test_cursor_scan_filters_tombstoned_new_key(tmp_path: Path,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    h, idx = _hub(tmp_path), _index(tmp_path)
    _turn(h, "local/a/s1/0000")
    monkeypatch.setattr(h, "_tombstoned_keys", lambda: {"local/a/s1/0001"})
    idx._scan_with_cursor(h)
    _turn(h, "local/a/s1/0001")                         # 墓碑条数没变，走游标趟
    _turn(h, "local/a/s1/0002")
    assert [k for _t, k in _assert_same_as_full(idx, h)] == ["local/a/s1/0000", "local/a/s1/0002"]
    assert idx._scan_state.rounds == 1
    idx.close()
    h.close()


def test_new_session_and_slash_in_session(tmp_path: Path) -> None:
    h, idx = _hub(tmp_path), _index(tmp_path)
    _turn(h, "local/a/s1/0000")
    _turn(h, "local/a/s1/x/0000")                       # session 名含斜杠：与 s1 交错
    _put_raw(h, "0001", _ev())
    h.db[b"__msg__/abc"] = msgpack.packb({"content": "x"}, use_bin_type=True)
    h.db[b"zz/a/s/0000"] = msgpack.packb({"ts": "t"}, use_bin_type=True)
    idx._scan_with_cursor(h)
    _turn(h, "local/a/s1/0001")
    _turn(h, "local/a/s1/x/0001")
    _turn(h, "local/a/s1/z")
    _turn(h, "aa/a/s/0000")                             # 排在所有内部键之前的新 session
    _assert_same_as_full(idx, h)
    idx.close()
    h.close()


def test_full_and_filter_keep_full_scan(tmp_path: Path) -> None:
    h, idx = _hub(tmp_path), _index(tmp_path)
    _turn(h, "local/a/s1/0000")
    idx._scan_with_cursor(h)
    idx._mark_consumed("local/a/s1/0000")
    assert idx._scan_with_cursor(h)[0] == []
    assert idx.unconsume_matching(h, agents={"a"}) == 1
    assert idx._scan_state is None, "撤消费标记后游标状态必须作废"
    assert [k for _t, k in idx._scan_with_cursor(h)[0]] == ["local/a/s1/0000"]
    idx.close()
    h.close()


# ── W17 ③：画像短路 ──────────────────────────────────────────────────────────

def _with_fact(tmp: Path) -> Any:
    from bladex_core.fact import Fact

    idx = _index(tmp)
    fact = Fact(id="f1", content="x", source_user_id="u", agent_id="codex")
    idx.meta_db[b"fact/f1"] = msgpack.packb(
        fact.model_dump(mode="json", exclude={"embedding"}), use_bin_type=True)
    return idx


def _renders(idx: Any) -> list[int]:
    calls: list[int] = []
    real = idx._render_profile_docs

    def counting() -> int:
        calls.append(1)
        return real()

    idx._render_profile_docs = counting
    return calls


def test_profile_docs_short_circuit(tmp_path: Path) -> None:
    idx = _with_fact(tmp_path)
    calls = _renders(idx)
    idx.refresh_profile_docs()
    idx.refresh_profile_docs()
    assert len(calls) == 1, "输入不变不重写"
    for key in (b"fact/f2", b"profile/tool/x", b"rulefile/a/b"):
        before = len(calls)
        value = idx.meta_db[b"fact/f1"] if key.startswith(b"fact/") else msgpack.packb({})
        idx.meta_db[key] = value
        idx.refresh_profile_docs()
        assert len(calls) == before + 1, f"{key!r} 变了必须重写"
        idx.refresh_profile_docs()
        assert len(calls) == before + 1, "写完回到短路"
    idx.close()


def test_profile_docs_date_bucket(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from datetime import UTC, datetime

    from bladex_proxy.storage import memory_index as mi

    idx = _with_fact(tmp_path)
    calls = _renders(idx)
    idx.refresh_profile_docs()

    class Tomorrow(datetime):
        @classmethod
        def now(cls, tz: Any = None) -> datetime:
            return datetime(2099, 1, 1, tzinfo=tz or UTC)

    monkeypatch.setattr(mi, "datetime", Tomorrow)
    idx.refresh_profile_docs()
    assert len(calls) == 2, "跨天必须重写（7 天窗口）"
    idx.close()
