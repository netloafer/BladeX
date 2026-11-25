"""批 DS1a 步骤 1.5：Hub 墓碑缓存跨线程的失效丢失（结算稿 §1.2 ⑤）。

修前：`_tombstoned_keys()` 是"缓存为空就重扫、扫完赋值"，写墓碑把缓存置空。读线程重建到一半时
另一线程写墓碑 ⇒ 读线程把**不含新墓碑**的旧集合装回去 ⇒ 被删的轮次在本进程重新可见，直到下一次写墓碑。
修后：缓存带代次令牌，写墓碑换新令牌；令牌对不上就当没缓存。

两条交错都用确定性的暂停点复现（随机加压撞不上这个毫秒级窗口，读数为 0 不算证明）。
"""

from __future__ import annotations

import tempfile
import threading

from bladex_proxy.models import Identity, TombstoneSource, TombstoneTargetType, Turn
from bladex_proxy.storage.memory_hub import MemoryHub


def _hub() -> MemoryHub:
    hub = MemoryHub(f"{tempfile.mkdtemp(prefix='ds1a_tomb_')}/hub")
    hub.open()
    for n in range(3):
        hub.put(f"u1/a/s/{n:06d}",
                Turn(identity=Identity(user_id="u1", agent_id="a", session_id="s")))
    return hub


def _tomb(hub: MemoryHub, key: str) -> None:
    hub.append_tombstone(target_type=TombstoneTargetType.TURN, target_key=key,
                         source=TombstoneSource.USER)


def _paused(gen_fn):
    """包一层迭代器：吐出第一条后停住，等放行。"""
    paused, resume = threading.Event(), threading.Event()

    def wrapped(*a, **kw):
        for i, item in enumerate(gen_fn(*a, **kw)):
            if i == 0:
                paused.set()
                resume.wait(5)
            yield item

    return wrapped, paused, resume


def test_rebuild_racing_a_tombstone_write_does_not_hide_it() -> None:
    hub = _hub()
    _tomb(hub, "u1/a/s/000000")
    orig = hub.scan_tombstones
    slow, paused, resume = _paused(orig)
    hub.scan_tombstones = slow
    t = threading.Thread(target=hub._tombstoned_keys)
    t.start()
    assert paused.wait(5)
    hub.scan_tombstones = orig
    _tomb(hub, "u1/a/s/000001")  # 读线程重建到一半时写入
    resume.set()
    t.join(5)
    assert hub._tombstoned_keys() == {"u1/a/s/000000", "u1/a/s/000001"}
    assert hub.count() == 1
    assert [k for k, _ in hub.scan_meta()] == ["u1/a/s/000002"]
    hub.close()


def test_snapshot_racing_a_tombstone_write_does_not_hide_it() -> None:
    """`scan_meta_snapshot()` 扫完全库后赋缓存（live ≈4 s 的窗口），同一判据。"""
    hub = _hub()
    _tomb(hub, "u1/a/s/000000")
    db = hub.db
    paused, resume = threading.Event(), threading.Event()

    class _SlowDb:
        def __getattr__(self, name):  # noqa: ANN001, ANN204
            return getattr(db, name)

        def items(self, *a, **kw):  # noqa: ANN002, ANN003
            for i, item in enumerate(db.items(*a, **kw)):
                if i == 0:
                    paused.set()
                    resume.wait(5)
                yield item

    hub._db = _SlowDb()
    t = threading.Thread(target=hub.scan_meta_snapshot)
    t.start()
    assert paused.wait(5)
    hub._db = db
    _tomb(hub, "u1/a/s/000001")
    resume.set()
    t.join(5)
    assert hub._tombstoned_keys() == {"u1/a/s/000000", "u1/a/s/000001"}
    assert hub.count() == 1
    hub.close()


def test_cache_still_caches_without_writes() -> None:
    """没有新写入时缓存照常命中（不退化成每次都扫）。"""
    hub = _hub()
    _tomb(hub, "u1/a/s/000000")
    first = hub._tombstoned_keys()
    calls = []
    orig = hub.scan_tombstones
    hub.scan_tombstones = lambda: (calls.append(1), orig())[1]
    assert hub._tombstoned_keys() is first
    assert calls == []
    _tomb(hub, "u1/a/s/000001")
    assert hub._tombstoned_keys() == {"u1/a/s/000000", "u1/a/s/000001"}
    assert calls == [1]
    hub.close()


def test_read_during_a_tombstone_write_does_not_cache_a_stale_set() -> None:
    """写侧顺序（MQ-I40 尾巴）：`append_tombstone` 必须**先写入、再换令牌**。

    暂停点放在墓碑那次 `db[key] = …` 之前，读线程在暂停期间完整跑一次 `_tombstoned_keys()`：
    - 正确顺序：读线程看到的还是旧令牌 ⇒ 命中旧缓存（此刻墓碑本就没写进去）；写完后换令牌 ⇒
      下一次读重扫，新墓碑在。
    - 反过来（先换令牌再写）：读线程拿到**新**令牌、扫不到这条墓碑、把旧集合配新令牌装回缓存 ⇒
      写入之后再没有人换令牌，这条墓碑在本进程一直看不见。
    """
    hub = _hub()
    _tomb(hub, "u1/a/s/000000")
    assert hub._tombstoned_keys() == {"u1/a/s/000000"}  # 缓存已就位
    db = hub.db
    paused, resume = threading.Event(), threading.Event()

    class _SlowWriteDb:
        def __getattr__(self, name):  # noqa: ANN001, ANN204
            return getattr(db, name)

        def __getitem__(self, k):  # noqa: ANN001, ANN204
            return db[k]

        def __contains__(self, k):  # noqa: ANN001, ANN204
            return k in db

        def __delitem__(self, k):  # noqa: ANN001, ANN204
            del db[k]

        def __setitem__(self, k, v):  # noqa: ANN001, ANN204
            if bytes(k).startswith(b"tombstone/") and not paused.is_set():
                paused.set()
                resume.wait(5)
            db[k] = v

    hub._db = _SlowWriteDb()
    w = threading.Thread(target=_tomb, args=(hub, "u1/a/s/000001"))
    w.start()
    assert paused.wait(5)
    r = threading.Thread(target=hub._tombstoned_keys)  # 写入进行中完整读一次
    r.start()
    r.join(5)
    assert not r.is_alive()
    resume.set()
    w.join(5)
    hub._db = db
    assert hub._tombstoned_keys() == {"u1/a/s/000000", "u1/a/s/000001"}
    assert hub.count() == 1
    assert [k for k, _ in hub.scan_meta()] == ["u1/a/s/000002"]
    hub.close()
