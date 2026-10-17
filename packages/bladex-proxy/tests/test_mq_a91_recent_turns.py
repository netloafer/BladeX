"""MQ-A91：`/admin/turns` 取最近 N 条不再随 Hub 总量线性增长。

修前：每次请求 `scan_meta` 全 Hub 再按 ts 排序（17.7k 轮 ≈ 4 s）。
修后：`MemoryHub.recent_turns` —— 进程内 (ts, key) 索引，首次调用建一次，`put()` 增量维护，
墓碑读侧过滤。本剧本钉两件事：

1. **结果逐条等于旧实现**（同 ts 并列、空 ts、墓碑、脏值、内部 key、带 / 不带 prefix、
   建索引后再写入 / 再墓碑 / 物理擦除、重开进程重建）；
2. **建成后不再全扫**：`scan_meta` 被换成抛错的桩，端点照常返回；全量建索引只发生一次。
"""

from __future__ import annotations

import random
import tempfile
from datetime import UTC, datetime, timedelta

import pytest
from bladex_proxy.models import Identity, TombstoneTargetType, Turn
from bladex_proxy.storage.memory_hub import MemoryHub

_BASE = datetime(2026, 9, 1, tzinfo=UTC)


def _old(hub: MemoryHub, prefix: str, limit: int, offset: int) -> tuple[list, int]:
    """修前 `/admin/turns` 的原样算法（key 序遍历 + 按 ts 稳定降序）。"""
    items = [{"key": k, "ts": ts} for k, ts in hub.scan_meta(prefix=prefix)]
    items.sort(key=lambda x: x["ts"], reverse=True)
    return [(x["key"], x["ts"]) for x in items[offset:offset + limit]], len(items)


def _turn(user: str, agent: str, sess: str, ts: datetime) -> Turn:
    return Turn(identity=Identity(user_id=user, agent_id=agent, session_id=sess), ts=ts)


def _seed(hub: MemoryHub, rng: random.Random, n: int, start: int = 0) -> list[str]:
    keys = []
    for i in range(start, start + n):
        user = rng.choice(["local", "u2"])
        agent = rng.choice(["claude-code", "codex", "hermes"])
        sess = f"s{rng.randrange(6)}"
        # 故意制造大量同 ts 并列（分钟粒度 + 小范围），检验并列时的 key 升序
        ts = _BASE + timedelta(minutes=rng.randrange(40))
        key = f"{user}/{agent}/{sess}/{rng.randrange(10**6):06d}-{i}"
        hub.put(key, _turn(user, agent, sess, ts))
        keys.append(key)
    return keys


_CASES = [("", 20, 0), ("", 100, 0), ("", 7, 13), ("", 500, 0), ("", 5, 10_000),
          ("local/", 20, 0), ("local/codex/", 10, 3), ("u2/hermes/s1/", 100, 0),
          ("nope/", 10, 0), ("tombstone/", 10, 0), ("local/claude-code/s2/", 1, 1)]


def _assert_same(hub: MemoryHub) -> None:
    for prefix, limit, offset in _CASES:
        assert hub.recent_turns(prefix, limit=limit, offset=offset) == \
            _old(hub, prefix, limit, offset), (prefix, limit, offset)


@pytest.fixture()
def hub() -> MemoryHub:
    h = MemoryHub(tempfile.mkdtemp())
    h.open()
    yield h
    h.close()


def test_recent_turns_equals_old_scan_through_writes_tombstones_and_compaction(hub: MemoryHub) -> None:
    rng = random.Random(91)
    keys = _seed(hub, rng, 300)
    # 空 ts（老 Turn 没这个字段）+ 解不开的脏值：旧实现一个排最后、一个跳过
    import msgpack
    hub.db[b"local/codex/s0/zz-empty-ts"] = msgpack.packb({"identity": {}}, use_bin_type=True)
    hub.db[b"local/codex/s0/zz-garbage"] = b"\xc1\xc1not-msgpack"
    for k in rng.sample(keys, 25):
        hub.append_tombstone(TombstoneTargetType.TURN, k)
    _assert_same(hub)  # 首次调用建索引

    # 建成之后：新写入、新墓碑、物理擦除都要跟上
    more = _seed(hub, rng, 120, start=1000)
    for k in rng.sample(more, 10) + rng.sample(keys, 10):
        hub.append_tombstone(TombstoneTargetType.TURN, k)
    _assert_same(hub)
    assert hub.compact_tombstoned() > 0
    _assert_same(hub)

    # 「重建」= 新进程首次调用时全扫；与在线维护出来的结果一致
    online = [hub.recent_turns(p, limit=lim, offset=o) for p, lim, o in _CASES]
    path = hub._path
    hub.close()
    reopened = MemoryHub(path)
    reopened.open()
    try:
        assert [reopened.recent_turns(p, limit=lim, offset=o) for p, lim, o in _CASES] == online
    finally:
        reopened.close()


def test_recent_turns_does_not_rescan_hub_after_built(hub: MemoryHub, monkeypatch) -> None:
    rng = random.Random(7)
    _seed(hub, rng, 50)
    builds = []
    orig = hub._iter_turn_ts
    monkeypatch.setattr(hub, "_iter_turn_ts", lambda: (builds.append(1), orig())[1])
    hub.recent_turns("", limit=5)

    def _boom(*_a, **_k):
        raise AssertionError("full Hub scan on the /admin/turns path")

    monkeypatch.setattr(hub, "scan_meta", _boom)
    _seed(hub, rng, 20, start=500)
    hub.append_tombstone(TombstoneTargetType.TURN, hub.recent_turns("", limit=1)[0][0][0])
    for _ in range(3):
        hub.recent_turns("", limit=20)
        hub.recent_turns("local/", limit=20)
    assert builds == [1]


def test_admin_turns_endpoint_uses_index_not_full_scan(monkeypatch) -> None:
    from bladex_proxy.config import ProxyConfig
    from bladex_proxy.server import create_app
    from fastapi.testclient import TestClient

    tmp = tempfile.mkdtemp()
    app = create_app(ProxyConfig(upstream_model="openai/test", upstream_api_key="sk-fake",
                                 rocksdb_path=f"{tmp}/rocksdb", index_path=f"{tmp}/index"))
    with TestClient(app) as client:
        hub: MemoryHub = app.state.hub
        for i in range(5):
            hub.put(f"u1/a/s1/{i:04d}", _turn("u1", "a", "s1", _BASE + timedelta(seconds=i)))
        hub.put("u1/b/s9/0000", _turn("u1", "b", "s9", _BASE + timedelta(seconds=99)))
        expected_all = _old(hub, "", 3, 0)
        expected_sess = _old(hub, "u1/a/s1/", 100, 1)

        def _boom(*_a, **_k):
            raise AssertionError("full Hub scan on the /admin/turns path")

        monkeypatch.setattr(hub, "scan_meta", _boom)
        d = client.get("/admin/turns", params={"limit": 3}).json()
        assert [(t["key"], t["ts"]) for t in d["turns"]] == expected_all[0]
        assert d["total"] == expected_all[1] == 6
        d = client.get("/admin/turns", params={"prefix": "u1/a/s1/", "offset": 1}).json()
        assert [(t["key"], t["ts"]) for t in d["turns"]] == expected_sess[0]
        assert d["total"] == 5
        assert d["turns"][0]["key"] == "u1/a/s1/0003"
