"""`/admin/status` 两个计数的正确性（MQ-R3 + MQ-R5，2026-08-17 立）。

两条缺陷单独看都像运维噪声，叠加起来把"消费停摆"这个假象做得非常可信
（2026-08-17 为此误判两次、查了三小时，还据此下过"要跑一夜"的错误排期）：

- **MQ-R3**：`fact_count()` / `consumed_count()` 不追新 → 数字**冻结**在上次
  catch-up 时刻（实测 status 报 facts 1775 / 同刻真实 1924）。症状是
  "数字看着正常，只是不动"——没有本文件的测试，下次重构会再把追新去掉而无人察觉。
- **MQ-R5**：`ledger.count()` 把墓碑覆盖的 turn 计进被减数，而消费侧按设计永不
  消费它们 → `lag_turns` 是一个**永不消失的常数**（08-16 迁移掉的 500 条残渣，
  让系统从那天起恒报"落后 500 轮"）。
"""

import inspect
import tempfile
from pathlib import Path

from bladex_core.fact import Fact
from bladex_proxy.models import (
    Identity,
    TombstoneSource,
    TombstoneTargetType,
    Turn,
    TurnStatus,
)
from bladex_proxy.storage.memory_hub import MemoryHub
from bladex_proxy.storage.memory_index import MemoryIndex


class _MockEmbedder:
    """固定向量 mock embedder（与 test_memory_index.py 同款）。"""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[((hash(t) >> i) & 1) * 1.0 for i in range(self._dim)] for t in texts]


def _fact(fid: str, content: str, embedder: _MockEmbedder) -> Fact:
    return Fact(
        id=fid,
        content=content,
        category="general",
        source_user_id="u1",
        embedding=embedder.embed([f"passage: {content}"])[0],
    )


def _turn(session_id: str, i: int) -> Turn:
    return Turn(
        identity=Identity(
            user_id="u1", agent_id="a1", session_id=session_id, turn_index=i,
        ),
        model="test",
        request_messages=[{"role": "user", "content": f"q{i}"}],
        response_text=f"a{i}",
        status=TurnStatus.OK,
    )


# ── MQ-R3：secondary 视图下的计数必须追新 ──────────────────────────────────


def test_fact_count_sees_primary_writes_without_reopen():
    """MQ-R3 核心：primary 写入后，**不重开** secondary 索引，`fact_count()` 就能读到新值。

    这正是 proxy 的读形态（read_only secondary，寿命以天计）。修复前该计数冻结在
    上次 catch-up 时刻，于是"排空看起来完全没动，实际一直在推进"。
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        embedder = _MockEmbedder()
        path = Path(tmpdir) / "index"

        index_write = MemoryIndex(path, embedder=embedder)
        index_write.open()
        index_write.add_fact(_fact("f1", "BladeX uses RocksDB", embedder))

        index_read = MemoryIndex(
            path, embedder=embedder, read_only=True, catchup_throttle_s=0.0,
        )
        index_read.open()
        assert index_read.fact_count() == 1

        # consolidator 再写两条，proxy 不重启
        index_write.add_fact(_fact("f2", "BladeX uses LanceDB", embedder))
        index_write.add_fact(_fact("f3", "BladeX uses Redis Streams", embedder))

        assert index_read.fact_count() == 3, (
            "secondary 视图的 fact_count 必须追新（MQ-R3）——"
            "冻结的数字比错误的数字更难发现"
        )

        index_read.close()
        index_write.close()


def test_consumed_count_sees_primary_writes_without_reopen():
    """MQ-R3：`consumed_count()` 同样必须追新——它是 `lag_turns` 的减数。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        embedder = _MockEmbedder()
        path = Path(tmpdir) / "index"

        index_write = MemoryIndex(path, embedder=embedder)
        index_write.open()
        index_write._mark_consumed("u1/a1/s1/e1")

        index_read = MemoryIndex(
            path, embedder=embedder, read_only=True, catchup_throttle_s=0.0,
        )
        index_read.open()
        assert index_read.consumed_count() == 1

        for eid in ("e2", "e3", "e4"):
            index_write._mark_consumed(f"u1/a1/s1/{eid}")

        assert index_read.consumed_count() == 4, (
            "secondary 视图的 consumed_count 必须追新（MQ-R3）"
        )

        index_read.close()
        index_write.close()


def test_status_counts_call_catch_up():
    """MQ-R3 守卫：两个计数方法的实现里必须出现追新两连。

    行为测试（上面两条）证明"现在是对的"；本条钉住"**怎么**做到的"，
    让下次重构删掉追新时在 diff 上立刻显形，而不是等到某次误判排期。
    """
    for method in (MemoryIndex.fact_count, MemoryIndex.consumed_count):
        src = inspect.getsource(method)
        assert "_ensure_meta_db()" in src, f"{method.__name__} 缺惰性重开"
        assert "_try_catch_up()" in src, f"{method.__name__} 缺 secondary 追新"


# ── MQ-R5：被减数扣掉墓碑 ──────────────────────────────────────────────────


def test_ledger_count_excludes_tombstoned_turns():
    """MQ-R5 核心：写 N 轮、墓碑 M 轮 → `count()` = N − M，于是 lag = N − M − consumed。

    墓碑的 turn 永远不会被消费（`scan_meta` 按设计跳过），把它留在被减数里
    就等于给积压加了一个永不消失的常数。
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger = MemoryHub(Path(tmpdir) / "rocksdb")
        ledger.open()

        n, m, consumed = 7, 3, 2
        keys = []
        for i in range(n):
            identity = Identity(
                user_id="u1", agent_id="a1", session_id="s1", turn_index=i,
            )
            key = identity.storage_key(f"171990720{i}-0")
            ledger.put(key, _turn("s1", i))
            keys.append(key)

        assert ledger.count() == n

        for key in keys[:m]:
            ledger.append_tombstone(
                TombstoneTargetType.TURN, key, source=TombstoneSource.USER,
            )

        assert ledger.count() == n - m, "被减数必须扣掉墓碑（MQ-R5）"
        assert ledger.count() - consumed == n - m - consumed

        ledger.close()


def test_ledger_count_agrees_with_scan_meta():
    """MQ-R5 不变式：**同一个「有多少轮」的问题只允许一个答案**。

    `count()`（被减数）与 `scan_meta()`（消费侧的候选来源）必须逐个墓碑一致——
    两者一旦用不同的跳过规则，差值就变成一个没人能解释的常数。
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger = MemoryHub(Path(tmpdir) / "rocksdb")
        ledger.open()

        keys = []
        for i in range(5):
            identity = Identity(
                user_id="u1", agent_id="a1", session_id="s1", turn_index=i,
            )
            key = identity.storage_key(f"171990730{i}-0")
            ledger.put(key, _turn("s1", i))
            keys.append(key)

        # 阴性对照：一个墓碑都没有时两者就该相等（否则下面的相等是巧合）
        assert ledger.count() == len(list(ledger.scan_meta())) == 5

        ledger.append_tombstone(TombstoneTargetType.TURN, keys[0])
        ledger.append_tombstone(TombstoneTargetType.TURN, keys[4])

        assert ledger.count() == len(list(ledger.scan_meta())) == 3

        ledger.close()
