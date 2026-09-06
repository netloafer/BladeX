"""索引运行时：碎片自维护（MQ-R1）+ 热路径全表物化（MQ-R6），2026-08-17 立。

后半部分（文件末尾两条）是段内计时定位到的：`_fuse_rerank` 对每个候选调一次
`_get_fact_vector`，而它会把整张 facts 表物化一遍——40 候选 = 40 次，p50 207ms。

前半部分：


事故形态（**已兑现两次**，第二次让注入面空转了一个月）：LanceDB 追加写是
一行一碎片；1619 条 fact = 1619 个 datafile → dense 检索 6462ms → 每轮撞
250ms 热路径 deadline → 降级只注硬规则 → 2952 轮里只有 10 轮留下注入记录。
MS-14（08-09）记过同一形态，手工压了一次，「自维护待开发窗」没做；
08-16 一次全量重建把碎片一把堆回来。

碎片侧钉四件事：

1. 触发判据挂在**碎片数**这根轴上（旧的"行数 > fact 数 × 2"量的是僵尸行，
   一行一碎片时比值恒为 1，**永远够不着**——参照系脱钩的一个实例）；
2. 维护点在 **consolidation pass 收尾**，不是只在空闲轮（碎片是被"写"堆出来的，
   积压排空期一路写一路碎，空闲轮永远等不到——T8b B17 同一形态）；
3. **压缩必须让判据变好、且维护要能停下来**——这两条是 2026-08-17 gate 从首版
   实现里抓出来的：压缩不清旧版本 → 文件数压完 6 -> 7 → 每个 pass 重触发整表重写；
4. 阈值可配、置 0 即回退旧行为。
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
    return index


def _add_facts(index: MemoryIndex, n: int, offset: int = 0) -> None:
    for i in range(offset, offset + n):
        index.add_fact(Fact(
            id=f"fact_{i:04d}",
            content=f"passage: test fact number {i}",
            category="general",
            source_user_id="u1",
            embedding=index._embedder.embed([f"passage: test fact number {i}"])[0],
        ))


# ── 碎片读数本身 ────────────────────────────────────────────────────────────


def test_fragment_count_reads_disk_truth():
    """`fact_vector_fragments()` 数 `facts.lance/data/` 的文件，表不存在返回 -1。

    为什么不是"当前版本的 fragment 数"（语义更准的那个）：**本机没装 pylance**，
    2026-08-17 gate 实测 `to_lance()` 直接报 `The lance library is required`。
    live 那个 1619 也是按文件数量的，阈值标定与本读数同源。
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        index = _open_index(tmpdir)
        assert index.fact_vector_fragments() == -1, "还没有 facts 表 -> -1，不是 0"

        _add_facts(index, 5)
        assert index.fact_vector_fragments() >= 1
        index.close()


def test_append_writes_produce_fragments():
    """本组测试的**前提断言**：逐条追加写确实堆碎片。

    前提不成立时要当场说出来，而不是让下面的触发测试静默变成空转
    （"依赖没构造却照样输出结论"是 08-17 反复踩到的仪器骗人形态）。
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        index = _open_index(tmpdir)
        _add_facts(index, 6)
        assert index.fact_vector_fragments() >= 2, (
            "逐条 add_fact 应产生多个碎片；若这条挂了，说明 LanceDB 的写行为变了，"
            "MQ-R1 的阈值口径需要重标，而不是把这条测试改小"
        )
        index.close()


def test_compaction_actually_reduces_fragments():
    """压缩本体：合并 + 清理旧版本后，碎片数必须真的降下来。

    传 `cleanup_older_than_s=0` 是为了在一次调用里看到结果——生产用 60s
    （`_COMPACT_CLEANUP_S`：proxy 每次 search 前 `checkout_latest`，抽掉它正握着的
    版本会打断在线检索）。代价是生产要两次压缩才收敛，那由节流保证不空转。

    2026-08-17 gate 抓到的首版就是死在这里：默认保留窗口下旧文件全留着，
    压完 6 -> 7。**压缩没让判据变好 = 判据不可能被满足。**
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        index = _open_index(tmpdir)
        _add_facts(index, 6)
        before = index.fact_vector_fragments()
        assert before >= 2, "前提：碎片确实堆着"

        index.compact_fact_vectors(cleanup_older_than_s=0)
        after = index.fact_vector_fragments()

        assert after < before, f"压缩后碎片应减少：{before} -> {after}"
        assert index.fact_count() == 6, "数据不能在压缩里丢"
        index.close()


# ── 触发轴：碎片数 ──────────────────────────────────────────────────────────


def test_compaction_triggers_on_fragment_count(monkeypatch: pytest.MonkeyPatch):
    """碎片数超阈值即触发压缩——MQ-R1 的主轴。

    这里只验"触发了、且数据没丢"；"压完碎片真的降"由
    `test_compaction_actually_reduces_fragments` 单独验（生产的清理窗口是 60s，
    一次调用内看不到下降是**设计使然**，不该在这条里当失败读）。
    """
    monkeypatch.setenv("BLADEX_INDEX_MAX_FRAGMENTS", "2")
    with tempfile.TemporaryDirectory() as tmpdir:
        index = _open_index(tmpdir)
        _add_facts(index, 6)
        assert index.fact_vector_fragments() > 2, "前提：碎片数已超阈值"

        called: list[float | None] = []
        real = index.compact_fact_vectors
        index.compact_fact_vectors = lambda cleanup_older_than_s=None: (  # type: ignore[assignment]
            called.append(cleanup_older_than_s) or real(cleanup_older_than_s))

        assert index.maybe_compact_fact_vectors() is True
        assert called == [None], "维护路径必须走生产的清理窗口，不许偷偷传 0"
        assert index.fact_count() == 6, "数据不能在压缩里丢"
        index.close()


def test_compaction_is_not_a_loop(monkeypatch: pytest.MonkeyPatch):
    """🔴 压缩过一次之后，紧接着再问必须**不再触发**。

    2026-08-17 gate 抓到的实现缺陷的回归守卫：碎片数数的是盘上的文件，而刚压出来
    的旧版本文件要过 `_COMPACT_CLEANUP_S` 才被下一次压缩清掉——窗口内判据仍然超
    阈值。首版没有节流，于是每个 pass 都整表重写一次，**永远满足不了自己的退出
    条件**。维护作业必须能停下来；停不下来的"自维护"比不做更糟。
    """
    monkeypatch.setenv("BLADEX_INDEX_MAX_FRAGMENTS", "2")
    with tempfile.TemporaryDirectory() as tmpdir:
        index = _open_index(tmpdir)
        _add_facts(index, 6)
        assert index.maybe_compact_fact_vectors() is True

        assert index.maybe_compact_fact_vectors() is False, (
            f"压缩后立刻又触发 = 压缩循环；当前碎片数 {index.fact_vector_fragments()}"
        )
        # 阴性对照：节流窗口过去之后要能再压（不是被永久关掉）
        index._last_fact_compact_monotonic -= index._COMPACT_MIN_INTERVAL_S + 1
        assert index.maybe_compact_fact_vectors() is True, "节流不是一次性开关"
        index.close()


def test_no_compaction_below_threshold(monkeypatch: pytest.MonkeyPatch):
    """阴性对照：碎片数未超阈值时不压缩（否则每个 pass 都在重写整表）。"""
    monkeypatch.setenv("BLADEX_INDEX_MAX_FRAGMENTS", "100")
    with tempfile.TemporaryDirectory() as tmpdir:
        index = _open_index(tmpdir)
        _add_facts(index, 3)

        called: list[bool] = []
        index.compact_fact_vectors = lambda *a, **k: called.append(True) or (0, 0)  # type: ignore[assignment]
        assert index.maybe_compact_fact_vectors() is False
        assert not called
        index.close()


def test_threshold_zero_disables_fragment_axis(monkeypatch: pytest.MonkeyPatch):
    """置 0 = 关掉碎片轴，回退旧行为（回归通道）。

    顺带钉住 MQ-R1 的立论：**旧的行数轴在这个场景下永远触发不了**——
    一行一碎片时 `rows == fact_count`，比值恒为 1，够不着 ×2 门槛。
    这就是"判据量错了轴"的实例：碎片堆到 1619 个，旧判据一次都没响。
    """
    monkeypatch.setenv("BLADEX_INDEX_MAX_FRAGMENTS", "0")
    with tempfile.TemporaryDirectory() as tmpdir:
        index = _open_index(tmpdir)
        _add_facts(index, 6)
        assert index.fact_vector_fragments() > 2, "碎片确实堆着"
        assert index.maybe_compact_fact_vectors() is False, (
            "碎片轴关掉后，旧的行数轴对一行一碎片的形态无能为力——这正是 MS-14 "
            "复发时没有任何机制拦住它的原因"
        )
        index.close()


# ── 维护点：pass 收尾也要够得着 ─────────────────────────────────────────────


def test_busy_rebuild_path_reaches_maintenance():
    """有新 turn 的那条路径（pass 收尾）必须调用碎片维护。

    T8b B17 的同一形态：卫生作业只写在 `if not turns:` 分支里，而碎片恰恰是
    **有新 turn 时**堆出来的（08-17 实测：消费 27 轮 → 碎片 2 → 43，1:1 回涨）。
    只挂空闲轮 = 手工压缩撑不过一次积压排空。
    """
    from datetime import UTC, datetime

    from bladex_proxy.models import Identity, Turn, TurnStatus
    from bladex_proxy.storage.memory_hub import MemoryHub

    with tempfile.TemporaryDirectory() as tmpdir:
        ledger = MemoryHub(Path(tmpdir) / "rocksdb")
        ledger.open()
        identity = Identity(user_id="u1", agent_id="a1", session_id="s1", turn_index=0)
        ledger.put(identity.storage_key("1719907200-0"), Turn(
            identity=identity,
            model="test",
            request_messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user",
                 "content": "Message long enough for consolidation to produce a fact."},
            ],
            response_text="Response text",
            status=TurnStatus.OK,
            ts=datetime(2026, 8, 17, 12, 0, tzinfo=UTC),
        ))

        index = _open_index(tmpdir)
        called: list[bool] = []
        index.maybe_compact_fact_vectors = lambda *a, **k: bool(called.append(True))  # type: ignore[assignment]

        index.rebuild_from_hub(ledger)
        assert called, "有新 turn 的 pass 收尾必须调用 maybe_compact_fact_vectors（MQ-R1）"

        index.close()
        ledger.close()


# ── MQ-R6：热路径不许按候选数重复物化全表 ──────────────────────────────────


def test_fuse_rerank_materializes_table_once_per_search():
    """🔴 一轮检索只许 `to_arrow()` 一次，不许"每个候选一次"。

    2026-08-17 段内计时定位到的真实开销：`_fuse_rerank` p50 **207ms**，占 search 的
    53%，是那个"约 410ms 固定底"的一半。成因是它对每个候选调一次
    `_get_fact_vector`，而后者会把整张 facts 表（1799 行 × 1024 维）物化一遍——
    40 个候选 = 40 次全表物化。

    它能藏这么久是因为**既不走 `to_list()` 也不走 `get_fact()`**：
    段内计时里 LanceDB 查询本体只 3 次 / 9ms、`get_fact` 159 次 / 3.8ms，
    两个头号嫌疑都被读数排除掉之后，剩下的时间才落到这里。
    没有这条守卫，下次有人为了"简单"把批量拆回单条循环，就会静默退回 207ms。
    """
    class _CountingTable:
        """只数 `to_arrow()` 的转发壳（不给 LanceDB 对象打猴子补丁）。"""

        def __init__(self, inner, calls: list[int]) -> None:
            object.__setattr__(self, "_inner", inner)
            object.__setattr__(self, "_calls", calls)

        def __getattr__(self, name: str):
            attr = getattr(self._inner, name)
            if name != "to_arrow" or not callable(attr):
                return attr

            def counted(*a, **k):
                self._calls.append(1)
                return attr(*a, **k)

            return counted

    with tempfile.TemporaryDirectory() as tmpdir:
        index = _open_index(tmpdir)
        _add_facts(index, 12)

        # 预热 `_entity_freq`：它自己也会 to_arrow（pylance 缺失下的回落路径，
        # 台账里另账），TTL 缓存后不再干扰本条要量的东西。
        index._entity_freq()

        calls: list[int] = []
        index._table = _CountingTable(index._table, calls)
        index.search("test fact number 3", top_k=5)

        assert len(calls) <= 1, (
            f"一轮检索物化了 {len(calls)} 次全表——按候选数重复物化又回来了（MQ-R6）"
        )
        index.close()


def test_get_edges_skips_validation_for_other_matters():
    """`get_edges` 只对命中的边付 Pydantic 校验的钱（MQ-R6，29.5ms/次 的一部分）。

    语义不变是硬要求——本条既钉"结果正确"，也钉"没命中的边没被 validate"。
    没有后半句，下次有人"顺手简化"就会把这笔钱加回去，而结果仍然是对的、
    只是每轮慢 100ms，**没有任何测试会红**。
    """
    from bladex_core.matter import EdgeTargetType, MatterEdge

    with tempfile.TemporaryDirectory() as tmpdir:
        index = _open_index(tmpdir)
        for i in range(6):
            index.add_edge(MatterEdge(
                edge_id=f"e{i}", matter_id=f"m{i % 3}",
                target_type=EdgeTargetType.FACT,
                target_key=f"fact_{i}", confidence=0.9,
            ))

        validated: list[str] = []
        real_validate = MatterEdge.model_validate

        def counting_validate(data, *a, **k):
            validated.append(data.get("edge_id", "?") if isinstance(data, dict) else "?")
            return real_validate(data, *a, **k)

        MatterEdge.model_validate = counting_validate  # type: ignore[method-assign]
        try:
            edges = index.get_edges("m0")
        finally:
            MatterEdge.model_validate = real_validate  # type: ignore[method-assign]

        assert {e.edge_id for e in edges} == {"e0", "e3"}, "结果必须逐字不变"
        assert len(validated) == 2, (
            f"只该校验命中的 2 条，实际校验了 {len(validated)} 条：{validated}"
        )
        index.close()


def test_hop_expand_scans_meta_once_for_seed_matters():
    """🔴 一轮 `_hop_expand` 只许扫一遍 meta 取边，不许每张卡扫一遍。

    探针实测（`scripts/probe_meta_scan.py`，live 库 26608 个 key）：
    `get_edges` 的 25ms 里 **91% 是"走过全库 key"本身**——取 value 只加 2.2ms、
    解码 1.6ms、Pydantic 校验 2.5ms。所以减少解码/校验都没用（两次都试过，
    分别只省 1.5ms 与 0.1ms），唯一有效的是**少扫几遍**。
    改造前 `_hop_expand` 一轮扫 6 遍 = 168ms = 它的全部耗时。

    本条钉的是"扫几遍"，不是"结果对不对"（后者由既有 hop 剧本守着）——
    没有它，任何一次"顺手改回按 mid 逐张取"都会静默把 140ms 加回热路径。
    """
    from bladex_core.matter import EdgeTargetType, MatterEdge

    with tempfile.TemporaryDirectory() as tmpdir:
        index = _open_index(tmpdir)
        _add_facts(index, 6)
        for i in range(6):
            index.add_edge(MatterEdge(
                edge_id=f"e{i}", matter_id=f"m{i % 3}",
                target_type=EdgeTargetType.FACT,
                target_key=f"fact_{i:04d}", confidence=0.9,
            ))

        facts = index.all_facts()[:3]
        for i, f in enumerate(facts):
            f.matter_id = f"m{i}"          # 三个种子 Matter
            f._score = 0.5

        scans: list[str] = []
        real_get_edges = index.get_edges
        real_batch = index.get_edges_for_matters
        index.get_edges = lambda mid: scans.append(f"single:{mid}") or real_get_edges(mid)  # type: ignore[assignment]
        index.get_edges_for_matters = lambda mids: (  # type: ignore[assignment]
            scans.append("batch") or real_batch(mids))

        index._hop_expand(facts, top_k=6, pids=None, supersede_swaps=[], min_importance=0.0)

        assert scans.count("batch") == 1, "种子 Matter 的边必须一次批量取回"
        assert not [s for s in scans if s.startswith("single:")], (
            f"仍在逐张扫 meta：{scans}——每张卡一遍全库扫描 = 每轮多付 100ms+"
        )
        index.close()


def test_batch_and_single_edge_reads_agree():
    """批量与单张必须给出同一答案（少扫不等于少给）。"""
    from bladex_core.matter import EdgeTargetType, MatterEdge

    with tempfile.TemporaryDirectory() as tmpdir:
        index = _open_index(tmpdir)
        for i in range(9):
            index.add_edge(MatterEdge(
                edge_id=f"e{i}", matter_id=f"m{i % 3}",
                target_type=EdgeTargetType.FACT,
                target_key=f"fact_{i}", confidence=0.5 + i / 100,
            ))

        batch = index.get_edges_for_matters(["m0", "m1", "m_absent"])
        assert [e.edge_id for e in batch["m0"]] == [
            e.edge_id for e in index.get_edges("m0")]
        assert [e.edge_id for e in batch["m1"]] == [
            e.edge_id for e in index.get_edges("m1")]
        assert batch["m_absent"] == [], "不存在的卡给空列表，不是缺键"
        assert index.get_edges_for_matters([]) == {}, "空输入不扫库"
        index.close()


def test_batch_vector_lookup_returns_requested_ids():
    """批量取向量的正确性（守住上面那条守卫不是靠"少做事"换来的）。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        index = _open_index(tmpdir)
        _add_facts(index, 5)

        got = index._get_fact_vectors(["fact_0001", "fact_0003", "no_such_id"])
        assert set(got) == {"fact_0001", "fact_0003"}, "缺席 id 静默跳过，不报错"
        assert len(got["fact_0001"]) == 64

        # 单条版本必须与批量版本一致（同一件事只有一个实现）
        assert index._get_fact_vector("fact_0001") == got["fact_0001"]
        assert index._get_fact_vector("no_such_id") is None
        index.close()


def test_idle_rebuild_path_still_reaches_maintenance():
    """空闲轮那条路径原本就调（旧行为不许在本次改动里丢掉）。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        index = _open_index(tmpdir)
        called: list[bool] = []
        index.maybe_compact_fact_vectors = lambda *a, **k: bool(called.append(True))  # type: ignore[assignment]

        class _EmptyLedger:
            def scan_meta(self, prefix: str = ""):
                return iter(())

            def is_tombstoned(self, *a, **k):
                return False

            def scan_admin_events(self, *a, **k):
                return iter(())

        index.rebuild_from_hub(_EmptyLedger())  # type: ignore[arg-type]
        assert called, "无新 turn 分支也要调用 maybe_compact_fact_vectors"
        index.close()
