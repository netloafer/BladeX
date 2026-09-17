"""MQ-I15 ②（2026-09-18）：LanceDB 三表版本自维护——磁盘不再单调上涨。

读数（09-13）：Memory Index 771MB 里 ~600MB 是死版本：files 表 7,670 版本 / 242MB、
matters 6,988 / 135MB，真身合起来不到 5MB。成因：LanceDB 每写一次新建一版、旧版永不自清；
仓里 `compact_fact_vectors` 只管 facts 且只在碎片超阈值时触发，`compact_matter_vectors`
drop+重建**不清版本**，files 表没有任何维护，`cleanup_old_versions` 全仓零调用。
一次性清到 305MB 之后 4 天，matters 又长回 827 版本——**机制没修，它就会再长回来**。

钉四件事：① 版本读数量的是盘上 `_versions/`（判据挂在被测对象上）；② 逐条写确实堆版本
（前提断言）；③ 超阈值 ⇒ vacuum 后版本数真的降、数据不丢；④ 阈值 0 关闭 + 节流不空转。

阴性对照：去掉 `rebuild_from_hub` 空闲分支里的 `maybe_vacuum_lance_versions()` 调用 ⇒
`test_idle_path_reaches_version_vacuum` 红；去掉 `maybe_vacuum_lance_versions` 里的
`optimize(...)` 调用 ⇒ `test_vacuum_reduces_versions_and_keeps_rows` 红。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from bladex_core.fact import Fact
from bladex_proxy.storage.memory_index import MemoryIndex


class _MockEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[((hash(t) >> i) & 1) * 1.0 for i in range(64)] for t in texts]

    @property
    def available(self) -> bool:
        return True


def _open_index(tmpdir: str) -> MemoryIndex:
    index = MemoryIndex(Path(tmpdir) / "index", embedder=_MockEmbedder())
    index.open()
    # 测试里立刻看到结果；生产 60s（proxy 只读进程握版本以毫秒计，见 _COMPACT_CLEANUP_S）
    index._COMPACT_CLEANUP_S = 0.0
    return index


def _add_facts(index: MemoryIndex, n: int) -> None:
    for i in range(n):
        text = f"passage: version vacuum fact {i}"
        index.add_fact(Fact(id=f"fact_{i:04d}", content=text, category="general",
                            source_user_id="u1", embedding=index._embedder.embed([text])[0]))


def _grow_files_table(index: MemoryIndex, n: int) -> None:
    """files 表没有公共逐条写入口可直接用；按生产同款（追加写）堆版本。"""
    tbl = index._lancedb.create_table("files", data=[{"path": "p0", "vector": [0.0] * 8}])
    for i in range(1, n):
        tbl.add([{"path": f"p{i}", "vector": [float(i)] * 8}])


def test_version_count_reads_disk_truth():
    with tempfile.TemporaryDirectory() as tmpdir:
        index = _open_index(tmpdir)
        assert index.lance_version_count("facts") == -1, "没有表 -> -1，不是 0"
        _add_facts(index, 3)
        assert index.lance_version_count("facts") >= 1
        index.close()


def test_append_writes_stack_versions():
    """前提断言：逐条写确实一写一版。挂了说明 LanceDB 写行为变了，阈值口径要重标。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        index = _open_index(tmpdir)
        _add_facts(index, 8)
        _grow_files_table(index, 8)
        assert index.lance_version_count("facts") >= 8
        assert index.lance_version_count("files") >= 8
        index.close()


def test_vacuum_reduces_versions_and_keeps_rows(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BLADEX_INDEX_MAX_VERSIONS", "5")
    with tempfile.TemporaryDirectory() as tmpdir:
        index = _open_index(tmpdir)
        _add_facts(index, 8)
        _grow_files_table(index, 8)
        before_facts = index.lance_version_count("facts")
        before_files = index.lance_version_count("files")
        assert before_facts > 5 and before_files > 5, "前提：版本确实堆过阈值"

        done = index.maybe_vacuum_lance_versions()

        assert set(done) == {"facts", "files"}, f"两张超阈值的表都该 vacuum：{done}"
        assert index.lance_version_count("facts") < before_facts
        assert index.lance_version_count("files") < before_files
        assert index.fact_count() == 8, "数据不能在 vacuum 里丢"
        assert index._lancedb.open_table("files").count_rows() == 8
        index.close()


def test_threshold_zero_disables_vacuum(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BLADEX_INDEX_MAX_VERSIONS", "0")
    with tempfile.TemporaryDirectory() as tmpdir:
        index = _open_index(tmpdir)
        _add_facts(index, 8)
        before = index.lance_version_count("facts")
        assert index.maybe_vacuum_lance_versions() == []
        assert index.lance_version_count("facts") == before
        index.close()


def test_vacuum_is_throttled_not_a_loop(monkeypatch: pytest.MonkeyPatch):
    """清理窗口内旧版本还在盘上、判据仍超阈值——没有节流就会每个空闲轮空压一次。"""
    monkeypatch.setenv("BLADEX_INDEX_MAX_VERSIONS", "2")
    with tempfile.TemporaryDirectory() as tmpdir:
        index = _open_index(tmpdir)
        _add_facts(index, 6)
        calls: list[str] = []
        real = index._lancedb.open_table

        def _spy(name: str):
            calls.append(name)
            return real(name)

        monkeypatch.setattr(index._lancedb, "open_table", _spy)
        first = index.maybe_vacuum_lance_versions()
        second = index.maybe_vacuum_lance_versions()
        assert first == ["facts"]
        assert second == [], "节流窗口内第二次必须空转"
        index.close()


def test_read_only_instance_never_vacuums():
    """proxy 侧（secondary）不许动版本——它只读，vacuum 是 consolidator 的活。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        w = _open_index(tmpdir)
        _add_facts(w, 8)
        w.close()
        r = MemoryIndex(Path(tmpdir) / "index", embedder=_MockEmbedder(), read_only=True)
        r.open()
        assert r.maybe_vacuum_lance_versions() == []
        r.close()


def test_idle_path_reaches_version_vacuum():
    """接线守卫：空闲轮卫生作业里必须调到它（与 maybe_compact_* 同址）。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        index = _open_index(tmpdir)
        called: list[bool] = []
        index.maybe_vacuum_lance_versions = lambda *a, **k: (called.append(True), [])[1]  # type: ignore[assignment]

        class _EmptyHub:
            def scan_meta(self, prefix: str = ""):
                return iter(())

            def is_tombstoned(self, *a, **k):
                return False

            def scan_admin_events(self, *a, **k):
                return iter(())

        index.rebuild_from_hub(_EmptyHub())  # type: ignore[arg-type]
        assert called, "空闲轮没调 maybe_vacuum_lance_versions —— MQ-I15 机制断线"
        index.close()
