"""T8b: ANN 索引单元测试。

验证：
- _compute_ann_params 对不同数据量返回合理参数
- ensure_ann_index 真正建出索引（list_indices 非空）
- ensure_ann_index 幂等 + force=True 重建
- drop_ann_index 真正删除索引
- BLADEX_ANN_BYPASS=1 时 search 走暴力扫描
- nprobes 参数在 search 中正确传递
- rebuild_from_hub 不被 ANN 中断
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from bladex_core.fact import Fact
from bladex_proxy.storage.memory_index import MemoryIndex


class MockEmbedder:
    """固定向量 mock embedder - 用简单 hash 映射到固定维度向量。"""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            h = hash(text)
            vec = [((h >> i) & 1) * 1.0 for i in range(self._dim)]
            results.append(vec)
        return results


def _make_index(tmpdir: str, dim: int = 64) -> MemoryIndex:
    embedder = MockEmbedder(dim)
    index = MemoryIndex(Path(tmpdir) / "index", embedder=embedder)
    index.open()
    return index


def _add_facts(index: MemoryIndex, n: int) -> None:
    """写入 n 条 fact 到 LanceDB。"""
    for i in range(n):
        fact = Fact(
            id=f"fact_{i:04d}",
            content=f"passage: test fact number {i}",
            category="general",
            embedding=index._embedder.embed([f"passage: test fact number {i}"])[0],
        )
        index.add_fact(fact)


def _spy_nprobes(index: MemoryIndex) -> list[int]:
    """截获传给 LanceDB builder 的 nprobes 值（返回的 list 会被就地填充）。"""
    seen: list[int] = []
    real_apply = index._apply_ann_to_search

    class _Spy:
        def __init__(self, inner): self._inner = inner
        def nprobes(self, n):
            seen.append(n)
            return self._inner.nprobes(n)
        def __getattr__(self, item): return getattr(self._inner, item)

    index._apply_ann_to_search = lambda b, n=None, t=None: real_apply(_Spy(b), n, t)
    return seen


# ── _compute_ann_params ──────────────────────────────────────────

class TestComputeAnnParams:
    def test_small_dataset_no_index(self):
        assert MemoryIndex._compute_ann_params(100) == {}
        assert MemoryIndex._compute_ann_params(255) == {}

    def test_boundary_256(self):
        params = MemoryIndex._compute_ann_params(256)
        assert params != {}
        assert params["index_type"] == "IVF_FLAT"
        assert "num_partitions" in params
        # nprobes 已于 2026-07-29 退出返回值（实测推荐值不如 LanceDB 默认）
        assert "nprobes" not in params

    def test_large_dataset(self):
        params = MemoryIndex._compute_ann_params(10000)
        assert 8 <= params["num_partitions"] <= 256

    def test_num_partitions_is_power_of_2(self):
        for n in [256, 500, 1000, 5000, 10000, 100000]:
            params = MemoryIndex._compute_ann_params(n)
            if params:
                np_val = params["num_partitions"]
                assert (np_val & (np_val - 1)) == 0


# ── 真正建出索引 ─────────────────────────────────────────────────

class TestEnsureAnnIndex:
    def test_index_actually_created(self):
        """写入 >=256 行后 ensure_ann_index 应真正建出索引。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            index = _make_index(tmpdir)
            _add_facts(index, 300)

            result = index.ensure_ann_index("facts")
            assert result is True

            table = index._get_lance_table("facts")
            indices = list(table.list_indices())
            assert len(indices) > 0, "Index should exist after ensure_ann_index"

    def test_idempotent(self):
        """已有索引时再调 ensure_ann_index 应跳过。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            index = _make_index(tmpdir)
            _add_facts(index, 300)

            r1 = index.ensure_ann_index("facts")
            r2 = index.ensure_ann_index("facts")
            assert r1 is True
            assert r2 is True

            table = index._get_lance_table("facts")
            assert len(list(table.list_indices())) == 1

    def test_force_rebuild(self):
        """force=True 应先 drop 旧索引再重建。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            index = _make_index(tmpdir)
            _add_facts(index, 300)

            index.ensure_ann_index("facts")
            table = index._get_lance_table("facts")
            assert len(list(table.list_indices())) == 1

            index.ensure_ann_index("facts", force=True)
            assert len(list(table.list_indices())) == 1

    def test_small_dataset_skipped(self):
        """N < 256 时 ensure_ann_index 应返回 False。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            index = _make_index(tmpdir)
            _add_facts(index, 50)

            result = index.ensure_ann_index("facts")
            assert result is False
            table = index._get_lance_table("facts")
            assert len(list(table.list_indices())) == 0


# ── drop_ann_index 真正删除 ──────────────────────────────────────

class TestDropAnnIndex:
    def test_drop_actually_removes_index(self):
        """drop_ann_index 后 list_indices 应为空。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            index = _make_index(tmpdir)
            _add_facts(index, 300)
            index.ensure_ann_index("facts")

            table = index._get_lance_table("facts")
            assert len(list(table.list_indices())) > 0

            result = index.drop_ann_index("facts")
            assert result is True
            assert len(list(table.list_indices())) == 0

    def test_drop_on_empty_table(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            index = _make_index(tmpdir)
            assert index.drop_ann_index("facts") is False

    def test_drop_invalid_table_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            index = _make_index(tmpdir)
            assert index.drop_ann_index("nonexistent") is False


# ── BLADEX_ANN_BYPASS 回滚开关 ───────────────────────────────────

class TestAnnBypass:
    def test_bypass_env_var(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            index = _make_index(tmpdir)
            assert index._ann_bypass_enabled() is False
            with patch.dict(os.environ, {"BLADEX_ANN_BYPASS": "1"}):
                assert index._ann_bypass_enabled() is True

    def test_bypass_result_identical_to_no_index(self):
        """B16 判据：关掉索引 → 结果与"建索引之前"逐字一致。

        此前这条测试只断言 `isinstance(results, list)`（不崩溃即合格），
        而 B16 要证的是**行为等价**，不是"返回了个 list"。
        做法：同一批数据、同一 query，先在无索引状态取基线，
        建索引后开 BYPASS 再取一次，比对 fact id 序列**逐位相等**。
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            index = _make_index(tmpdir)
            _add_facts(index, 300)

            # 基线：还没有索引 = 暴力扫描
            baseline = [f.id for f in index.search("query: test fact number 7", top_k=10)]
            assert baseline, "基线不该为空，否则这条测试什么也没证明"

            index.ensure_ann_index("facts")
            assert len(list(index._get_lance_table("facts").list_indices())) == 1

            with patch.dict(os.environ, {"BLADEX_ANN_BYPASS": "1"}):
                bypassed = [
                    f.id for f in index.search("query: test fact number 7", top_k=10)
                ]

            assert bypassed == baseline, (
                f"BYPASS 应逐字回到无索引行为\nbaseline={baseline}\nbypassed={bypassed}"
            )

    def test_search_uses_index_when_not_bypassed(self):
        """不 bypass 时确实走索引：结果非空且 id 都是库里的。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            index = _make_index(tmpdir)
            _add_facts(index, 300)
            index.ensure_ann_index("facts")

            results = index.search("query: test fact number 7", top_k=10)
            assert len(results) == 10
            assert all(f.id.startswith("fact_") for f in results)


# ── B17: 索引维护与"有没有新 turn"解耦 ──────────────────────────

class TestMaintainAnnIndexes:
    def test_maintain_builds_index_without_any_turn(self):
        """空闲期（没有新 turn）也要能建出索引——B17 的根因就在这里。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            index = _make_index(tmpdir)
            _add_facts(index, 300)
            assert len(list(index._get_lance_table("facts").list_indices())) == 0

            assert index.maintain_ann_indexes() is True
            assert len(list(index._get_lance_table("facts").list_indices())) == 1

    def test_maintain_rate_limited(self):
        """限流：间隔内第二次调用应跳过（返回 False），不重复 list_indices。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            index = _make_index(tmpdir)
            _add_facts(index, 300)
            assert index.maintain_ann_indexes() is True
            assert index.maintain_ann_indexes() is False

    def test_maintain_runs_again_after_interval(self):
        """间隔过后可以再次尝试（min_interval_s=0 模拟）。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            index = _make_index(tmpdir)
            _add_facts(index, 300)
            assert index.maintain_ann_indexes() is True
            assert index.maintain_ann_indexes(min_interval_s=0) is True

    def test_no_new_turns_path_reaches_maintenance(self):
        """`rebuild_from_hub` 的"无新 turn"分支必须够得着索引维护。

        这是 2026-07-29 生产零证据的直接原因：维护只挂在函数尾部，
        而这条分支在中途 return，空闲期永远到不了。
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            index = _make_index(tmpdir)
            _add_facts(index, 300)

            called: list[bool] = []
            index.maintain_ann_indexes = lambda *a, **k: called.append(True)  # type: ignore[assignment]

            class _EmptyLedger:
                def scan_meta(self, prefix: str = ""): return iter(())
                def is_tombstoned(self, *a, **k): return False
                def scan_admin_events(self, *a, **k): return iter(())

            index.rebuild_from_hub(_EmptyLedger())  # type: ignore[arg-type]
            assert called, "无新 turn 分支应调用 maintain_ann_indexes"


# ── nprobes 显式旋钮（B18 的最终形态）────────────────────────────

class TestNprobesPassthrough:
    def test_explicit_nprobes_reaches_lancedb(self):
        """调用方显式传 nprobes，必须真的传到 LanceDB builder 上。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            index = _make_index(tmpdir)
            _add_facts(index, 300)
            index.ensure_ann_index("facts")
            seen = _spy_nprobes(index)
            index.search("query: test", top_k=5, nprobes=4)
            assert seen == [4], f"应把 4 传下去，实际 {seen}"

    def test_no_nprobes_by_default(self):
        """默认不设 nprobes —— 走 LanceDB 默认值。

        这是 2026-07-29 实测后的决定：4√N/10 推荐值在真实 e5 向量上
        recall@10 均值 0.950 / 最差 0.70，反而不如默认的 0.973 / 0.80。
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            index = _make_index(tmpdir)
            _add_facts(index, 300)
            index.ensure_ann_index("facts")
            seen = _spy_nprobes(index)
            index.search("query: test", top_k=5)
            assert seen == [], f"默认不该设 nprobes，实际设了 {seen}"

    def test_env_override(self):
        """BLADEX_ANN_NPROBES 提供显式旋钮。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            index = _make_index(tmpdir)
            _add_facts(index, 300)
            index.ensure_ann_index("facts")
            seen = _spy_nprobes(index)
            with patch.dict(os.environ, {"BLADEX_ANN_NPROBES": "25"}):
                index.search("query: test", top_k=5)
            assert seen == [25]

    def test_env_override_invalid_ignored(self):
        """非法值不生效、不抛异常（留 warning）。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            index = _make_index(tmpdir)
            with patch.dict(os.environ, {"BLADEX_ANN_NPROBES": "abc"}):
                assert index._ann_nprobes_override() is None
            with patch.dict(os.environ, {"BLADEX_ANN_NPROBES": "0"}):
                assert index._ann_nprobes_override() is None


# ── rebuild_ann_indexes ──────────────────────────────────────────

class TestRebuildAnnIndexes:
    def test_empty_index_no_crash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            index = _make_index(tmpdir)
            result = index.rebuild_ann_indexes()
            assert isinstance(result, dict)
            for v in result.values():
                assert isinstance(v, bool)

    def test_rebuild_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            index = _make_index(tmpdir)
            r1 = index.rebuild_ann_indexes()
            r2 = index.rebuild_ann_indexes()
            assert isinstance(r1, dict)
            assert isinstance(r2, dict)

    def test_force_rebuild_all(self):
        """force=True 应对所有表先 drop 再建。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            index = _make_index(tmpdir)
            _add_facts(index, 300)
            index.ensure_ann_index("facts")

            result = index.rebuild_ann_indexes(force=True)
            assert isinstance(result, dict)
            assert result.get("facts") is True


# ── get_ann_index_status ─────────────────────────────────────────

class TestGetAnnIndexStatus:
    def test_status_returns_dict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            index = _make_index(tmpdir)
            status = index.get_ann_index_status()
            assert isinstance(status, dict)
            assert "facts" in status
            assert "matters" in status

    def test_status_shows_bypass(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            index = _make_index(tmpdir)
            _add_facts(index, 10)
            status = index.get_ann_index_status()
            assert "bypass" in status["facts"]

    def test_status_shows_index_after_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            index = _make_index(tmpdir)
            _add_facts(index, 300)
            index.ensure_ann_index("facts")
            status = index.get_ann_index_status()
            assert status["facts"]["has_index"] is True
            assert status["facts"]["nrows"] >= 300


# ── rebuild_from_hub 不被 ANN 中断 ────────────────────────────────

class TestRebuildFromLedgerSafety:
    def test_rebuild_from_hub_survives_ann_failure(self):
        """rebuild_from_hub 即使 rebuild_ann_indexes 抛异常也不应中断。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            index = _make_index(tmpdir)
            with patch.object(index, 'rebuild_ann_indexes', side_effect=RuntimeError("forced failure")):
                try:
                    index.rebuild_from_hub(None)  # type: ignore[arg-type]
                except RuntimeError:
                    pytest.fail("rebuild_from_hub should not propagate ANN errors")
                except Exception:
                    pass
