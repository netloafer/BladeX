"""批 AC2 步骤 1：用量行缓存并进 `_RecentTurnIndex`（一个机制，不另起第二个进程内索引）。

判据：
- 建成后连续 `put()` N 轮，`MemoryHub.usage_rows()` 与「当时全扫重切」（`usage.iter_hub_turns` +
  `usage.usage_row`）**逐条相同**；
- 建索引期间的写入走 pending，建完不丢；
- 墓碑后不计入，物理擦除后仍一致；重开进程重建与在线维护出来的一致；
- 建成后不再全扫；
- `recent_turns` / `count` 的现有用例不动（`test_mq_a91_recent_turns.py` / `test_q3c_backlog_counts.py`）。
"""

from __future__ import annotations

import random
import tempfile
from datetime import UTC, datetime, timedelta

import pytest
from bladex_proxy.models import (
    INNER_LOOP_AUX_SOURCE,
    Identity,
    ResponseMeta,
    TombstoneTargetType,
    Turn,
    TurnStatus,
)
from bladex_proxy.storage.memory_hub import MemoryHub, _RecentTurnIndex
from bladex_proxy.usage import aggregate, iter_hub_turns, usage_row

_BASE = datetime(2026, 10, 1, tzinfo=UTC)

_RAW_SHAPES = [
    {"prompt_tokens": 100, "completion_tokens": 7, "total_tokens": 107,
     "prompt_tokens_details": {"cached_tokens": 64}},
    {"prompt_tokens": 50, "completion_tokens": 3, "total_tokens": 53,
     "cache_read_input_tokens": 40, "cache_creation_input_tokens": 2},
    {"prompt_tokens": 9, "completion_tokens": 1, "total_tokens": 10},
]


def _rescan(hub: MemoryHub) -> list:
    """「当时全扫重切」：与行缓存对照的参照物。"""
    return [r for k, d in iter_hub_turns(hub) if (r := usage_row(k, d)) is not None]


def _turn(rng: random.Random, i: int) -> tuple[str, Turn]:
    agent = rng.choice(["claude-code", "codex", "hermes"])
    sess = f"s{rng.randrange(5)}"
    kind = rng.randrange(6)
    meta = ResponseMeta()
    if kind in (0, 1, 2):
        meta = ResponseMeta(usage={"prompt": 1, "completion": 1, "total": 2},
                            usage_raw=_RAW_SHAPES[kind])
    elif kind == 3:  # 旧轮次：只有三键
        meta = ResponseMeta(usage={"prompt": 11 + i, "completion": 2, "total": 13 + i})
    # kind 4 / 5：上游没报
    turn = Turn(
        identity=Identity(user_id="local", agent_id=agent, session_id=sess),
        model=rng.choice(["openai/glm-5.3", "anthropic/glm-5.3-flash"]),
        ts=_BASE + timedelta(minutes=rng.randrange(60 * 24 * 3)),
        status=TurnStatus.FAILED if kind == 5 else TurnStatus.OK,
        auxiliary=rng.random() < 0.2,
        aux_source=INNER_LOOP_AUX_SOURCE if rng.random() < 0.1 else "",
        response_meta=meta,
    )
    return f"local/{agent}/{sess}/{rng.randrange(10**6):06d}-{i}", turn


def _seed(hub: MemoryHub, rng: random.Random, n: int, start: int = 0) -> list[str]:
    keys = []
    for i in range(start, start + n):
        k, t = _turn(rng, i)
        hub.put(k, t)
        keys.append(k)
    return keys


@pytest.fixture()
def hub() -> MemoryHub:
    h = MemoryHub(tempfile.mkdtemp(prefix="ac2_cache_"))
    h.open()
    yield h
    h.close()


def test_rows_equal_rescan_through_puts_tombstones_and_compaction(hub: MemoryHub) -> None:
    rng = random.Random(2)
    keys = _seed(hub, rng, 200)
    import msgpack
    hub.db[b"local/codex/s0/zz-empty-ts"] = msgpack.packb({"identity": {}}, use_bin_type=True)
    hub.db[b"local/codex/s0/zz-garbage"] = b"\xc1\xc1not-msgpack"
    assert hub.usage_rows() == _rescan(hub)  # 首次调用：一趟全扫建成

    # 建成后逐轮 put：每一轮都与当时全扫重切逐条相同
    for i in range(60):
        k, t = _turn(rng, 1000 + i)
        hub.put(k, t)
        keys.append(k)
        assert hub.usage_rows() == _rescan(hub), f"diverged after put #{i}"

    for k in rng.sample(keys, 30):
        hub.append_tombstone(TombstoneTargetType.TURN, k)
    rows = hub.usage_rows()
    assert rows == _rescan(hub)
    dead = {t.target_key for _, t in hub.scan_tombstones()}
    assert not ({r.key for r in rows} & dead)

    assert hub.compact_tombstoned() > 0
    assert hub.usage_rows() == _rescan(hub)

    online = hub.usage_rows()
    path = hub._path
    hub.close()
    reopened = MemoryHub(path)
    reopened.open()
    try:
        assert reopened.usage_rows() == online
    finally:
        reopened.close()


def test_tombstoned_turn_not_counted(hub: MemoryHub) -> None:
    rng = random.Random(3)
    keys = _seed(hub, rng, 40)
    before = aggregate((r.key, r) for r in hub.usage_rows())["totals"]["all"]["turns"]
    hub.append_tombstone(TombstoneTargetType.TURN, keys[0])
    after = hub.usage_rows()
    assert keys[0] not in {r.key for r in after}
    assert aggregate((r.key, r) for r in after)["totals"]["all"]["turns"] == before - 1


def test_write_during_build_goes_to_pending_and_is_kept(hub: MemoryHub, monkeypatch) -> None:
    rng = random.Random(4)
    _seed(hub, rng, 30)
    orig = hub._iter_turn_ts
    late: list[str] = []

    def _scan_with_a_concurrent_put():  # noqa: ANN202
        it = orig()  # RocksDB 迭代器 = 创建时刻快照，看不到下面这次写
        for n, item in enumerate(it):
            if n == 0:
                k, t = _turn(rng, 9999)
                hub.put(k, t)
                late.append(k)
            yield item

    monkeypatch.setattr(hub, "_iter_turn_ts", _scan_with_a_concurrent_put)
    rows = hub.usage_rows()
    assert late[0] in {r.key for r in rows}
    assert rows == _rescan(hub)


def test_no_rescan_after_built(hub: MemoryHub, monkeypatch) -> None:
    rng = random.Random(5)
    _seed(hub, rng, 20)
    hub.usage_rows()

    def _boom():  # noqa: ANN202
        raise AssertionError("full rescan after the cache was built")

    monkeypatch.setattr(hub, "_iter_turn_ts", _boom)
    _seed(hub, rng, 5, start=500)
    assert len(hub.usage_rows()) == 25
    hub.recent_turns(limit=5)
    hub.count()


def test_one_index_shared_by_recent_turns_and_usage(hub: MemoryHub, monkeypatch) -> None:
    """不另起第二个索引：`recent_turns` 建一次，`usage_rows` 直接复用（全扫只发生一次）。"""
    rng = random.Random(6)
    _seed(hub, rng, 20)
    builds: list[int] = []
    orig = hub._iter_turn_ts
    monkeypatch.setattr(hub, "_iter_turn_ts", lambda: (builds.append(1), orig())[1])
    hub.recent_turns(limit=3)
    hub.usage_rows()
    hub.count()
    assert builds == [1]
    assert [n for n, v in vars(hub).items() if isinstance(v, _RecentTurnIndex)] == ["_recent"]


def test_read_only_hub_falls_back_to_rescan_same_rows(hub: MemoryHub) -> None:
    rng = random.Random(7)
    _seed(hub, rng, 25)
    expected = _rescan(hub)
    path = hub._path
    hub.close()
    ro = MemoryHub(path, read_only=True)
    ro.open()
    try:
        assert ro.usage_rows() == expected
    finally:
        ro.close()
