"""批 Q3C：W4 积压采样与 `/admin/status` 的两个计数不再每次全库迭代。

- `MemoryHub.count()`：primary 模式改读 MQ-A91 的进程内 `_RecentTurnIndex`
  （可见数 = 索引项 − 被墓碑的索引项）；secondary / read_only 仍全扫。
  🔴 口径统一（MQ-R5「同一个『有多少轮』只许一个答案」）：**解码失败的脏值不计**——
  与 `scan_meta`（消费侧）/ `recent_turns` 的 total 同一套跳过规则。此前 `count()`
  只过 key、脏值照样计数 ⇒ 每有一条脏值，`lag_turns` 就多一个永远消不掉的常数
  （消费侧永远看不到它）——正是 MQ-R5 修过的那个形状。
- `MemoryIndex.consumed_count()`：从 consumed 前缀 seek、出前缀即停，结果逐条相同。
"""

from __future__ import annotations

import random
import tempfile
from datetime import UTC, datetime, timedelta

import msgpack
import pytest
from bladex_proxy.models import Identity, TombstoneTargetType, Turn
from bladex_proxy.storage.memory_hub import MemoryHub

_BASE = datetime(2026, 9, 1, tzinfo=UTC)


def _seed(hub: MemoryHub, rng: random.Random, n: int, start: int = 0) -> list[str]:
    keys = []
    for i in range(start, start + n):
        agent = rng.choice(["claude-code", "codex", "hermes"])
        sess = f"s{rng.randrange(4)}"
        key = f"local/{agent}/{sess}/{rng.randrange(10**6):06d}-{i}"
        hub.put(key, Turn(identity=Identity(user_id="local", agent_id=agent, session_id=sess),
                          ts=_BASE + timedelta(minutes=rng.randrange(40))))
        keys.append(key)
    return keys


def _scan_rule(hub: MemoryHub) -> int:
    """消费侧口径：`scan_meta` 产出的条数（内部 key / 墓碑 / 解码失败都跳过）。"""
    return sum(1 for _ in hub.scan_meta())


@pytest.fixture()
def hub() -> MemoryHub:
    h = MemoryHub(tempfile.mkdtemp())
    h.open()
    yield h
    h.close()


def test_count_matches_scan_rule_through_dirty_tombstones_and_erasure(hub: MemoryHub) -> None:
    rng = random.Random(3)
    keys = _seed(hub, rng, 200)
    hub.db[b"local/codex/s0/zz-empty-ts"] = msgpack.packb({"identity": {}}, use_bin_type=True)
    hub.db[b"local/codex/s0/zz-garbage"] = b"\xc1\xc1not-msgpack"
    for k in rng.sample(keys, 20):
        hub.append_tombstone(TombstoneTargetType.TURN, k)
    # 脏值不计（空 ts 的合法值照计）：201 − 20 墓碑
    assert hub.count() == _scan_rule(hub) == 181
    assert hub.count() == hub.recent_turns("", limit=1)[1]

    more = _seed(hub, rng, 50, start=1000)
    for k in rng.sample(more, 5):
        hub.append_tombstone(TombstoneTargetType.TURN, k)
    assert hub.count() == _scan_rule(hub) == 226
    assert hub.compact_tombstoned() == 25
    assert hub.count() == _scan_rule(hub) == 226


def test_secondary_and_readonly_give_the_same_answer(hub: MemoryHub) -> None:
    rng = random.Random(5)
    keys = _seed(hub, rng, 80)
    hub.db[b"local/codex/s0/zz-garbage"] = b"\xc1\xc1not-msgpack"
    for k in keys[:7]:
        hub.append_tombstone(TombstoneTargetType.TURN, k)
    primary = hub.count()
    path = hub._path
    hub.close()
    ro = MemoryHub(path, read_only=True)
    ro.open()
    try:
        assert ro.count() == primary == 73
    finally:
        ro.close()
    sec = MemoryHub(path, secondary=True, secondary_dir=tempfile.mkdtemp())
    sec.open()
    try:
        assert sec.count() == primary
    finally:
        sec.close()
    hub.open()


def test_count_does_not_iterate_the_hub_once_indexed(hub: MemoryHub, monkeypatch) -> None:
    rng = random.Random(11)
    _seed(hub, rng, 40)
    builds = []
    orig = hub._iter_turn_ts
    monkeypatch.setattr(hub, "_iter_turn_ts", lambda: (builds.append(1), orig())[1])
    assert hub.count() == 40
    hub._tombstoned_keys()  # 墓碑缓存已热（写墓碑后它按前缀 seek 重建，与本条无关）

    class _NoScan:
        def __init__(self, db) -> None:  # noqa: ANN001
            self._db = db

        def keys(self, *a, **k):  # noqa: ANN002, ANN003, ANN202
            raise AssertionError("full Hub key scan on the count() path")

        def items(self, *a, **k):  # noqa: ANN002, ANN003, ANN202
            if k.get("from_key") is None:
                raise AssertionError("full Hub item scan on the count() path")
            return self._db.items(*a, **k)

        def __getattr__(self, name):  # noqa: ANN001, ANN204
            return getattr(self._db, name)

        def __setitem__(self, k, v) -> None:  # noqa: ANN001
            self._db[k] = v

        def __getitem__(self, k):  # noqa: ANN001, ANN204
            return self._db[k]

    real = hub._db
    monkeypatch.setattr(hub, "_db", _NoScan(real))
    _seed(hub, rng, 10, start=500)
    hub.append_tombstone(TombstoneTargetType.TURN, hub.recent_turns("", limit=1)[0][0][0])
    for _ in range(3):
        assert hub.count() == 49
    assert builds == [1]
    monkeypatch.setattr(hub, "_db", real)


# ── consumed_count ───────────────────────────────────────────────────────────


class _Embedder:
    def embed(self, texts):  # noqa: ANN001, ANN201
        return [[1.0] + [0.0] * 31 for _ in texts]

    @property
    def available(self) -> bool:
        return True


def _old_consumed_count(ix) -> int:  # noqa: ANN001
    """修前原样：全 meta key 迭代 + 前缀判断。"""
    from bladex_proxy.storage.memory_index import _CONSUMED_PREFIX

    return sum(
        1 for k in ix.meta_db.keys()
        if (k.decode() if isinstance(k, bytes) else str(k)).startswith(_CONSUMED_PREFIX)
    )


def test_consumed_count_equals_full_scan_and_stops_at_prefix_end(tmp_path, monkeypatch) -> None:
    from bladex_proxy.storage.memory_index import MemoryIndex

    ix = MemoryIndex(tmp_path / "ix", embedder=_Embedder(), read_only=False)
    ix.open()
    try:
        assert ix.consumed_count() == 0
        for i in range(37):
            ix._mark_consumed(f"local/codex/s{i % 3}/k{i:04d}")
        # 字节序上紧挨着 consumed 前缀两侧的 key（前缀的前一位 / 后一位）都不许被算进去
        ix.meta_db[b"__meta__/p2_consumed."] = b"x"
        ix.meta_db[b"__meta__/p2_consumee/zzz"] = b"x"
        ix.meta_db[b"__meta__/p2_consumec/zzz"] = b"x"
        ix.meta_db[b"fact/zzz"] = b"x"
        assert ix.consumed_count() == _old_consumed_count(ix) == 37

        seen: list[bytes] = []
        real = ix._meta_db

        class _Spy:
            def keys(self, *a, **k):  # noqa: ANN002, ANN003, ANN202
                if k.get("from_key") is None:
                    raise AssertionError("full meta key scan on the consumed_count() path")
                for key in real.keys(*a, **k):
                    seen.append(key)
                    yield key

            def __getattr__(self, name):  # noqa: ANN001, ANN204
                return getattr(real, name)

        monkeypatch.setattr(ix, "_meta_db", _Spy())
        assert ix.consumed_count() == 37
        assert len(seen) == 38  # 37 条 + 出前缀的第一条即停
        monkeypatch.setattr(ix, "_meta_db", real)
    finally:
        ix.close()
