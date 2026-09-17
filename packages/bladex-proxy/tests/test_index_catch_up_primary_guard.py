"""MQ-I16：primary 实例（consolidator 主写端）不得调 `try_catch_up_with_primary()`。

现场：consolidator 日志每小时 ~1,400 条
`index_catch_up_failed error='Not implemented: Supported only by secondary instance'`
（单份 11,185 条，最早可追到 2026-08-11），proxy 侧同期 0 条。机制：`_try_catch_up()`
守卫只判 `_meta_db is None`，不判本实例是不是 secondary；对 primary 调 catch-up
RocksDB 直接抛。危害不是正确性，是**挡视线**——proxy 端真出追新失败时落的是同一行
逐字相同的 debug。

阴性对照：去掉 `_try_catch_up` 里的 `not self._read_only` 短路 ⇒
`test_primary_never_calls_catch_up` 红（fake meta_db 被调到）+ 源码守卫红。
"""
from __future__ import annotations

from _source_probe import source_of
from bladex_proxy.storage.memory_index import MemoryIndex


class _FakeMetaDb:
    def __init__(self) -> None:
        self.calls = 0

    def try_catch_up_with_primary(self) -> None:
        self.calls += 1
        raise RuntimeError("Not implemented: Supported only by secondary instance")


def _bare_index(read_only: bool) -> MemoryIndex:
    """不 open()：直接摆 fake meta_db，只测 `_try_catch_up` 的分流。"""
    idx = MemoryIndex.__new__(MemoryIndex)
    idx._read_only = read_only
    idx._meta_db = _FakeMetaDb()
    idx._last_catchup_monotonic = 0.0
    idx._catchup_throttle_s = 0.0
    return idx


def test_primary_never_calls_catch_up() -> None:
    idx = _bare_index(read_only=False)
    idx._try_catch_up()
    idx._try_catch_up()
    assert idx._meta_db.calls == 0, "primary 实例调了 try_catch_up_with_primary —— MQ-I16 假失败回来了"


def test_secondary_still_calls_catch_up() -> None:
    """正向对照：secondary 路径不能被顺手关掉（proxy 侧追新是记忆闭环的一半）。"""
    idx = _bare_index(read_only=True)
    idx._try_catch_up()
    assert idx._meta_db.calls == 1


def test_try_catch_up_source_guards_read_only() -> None:
    src = source_of(MemoryIndex, "_try_catch_up")
    assert "not self._read_only" in src, "_try_catch_up 必须以 `not self._read_only` 短路 primary（MQ-I16）"
    assert src.index("not self._read_only") < src.index("self._meta_db.try_catch_up_with_primary()")
