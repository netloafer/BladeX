"""队列强制排空的验收剧本（2026-08-06 事故驱动）。

事故形状：`bladex status` 里 "-> Hub 487 waiting" 一动不动。487 条既不在待投递
队列、也不在未确认队列——是 ack 加 XDEL 之前那版代码留下的残留，后台 worker 的两条
路（XREADGROUP `>` / XAUTOCLAIM）都够不着它们，而 XLEN 把它们一直算作积压。

这里用一个最小 Redis Streams 模拟器（XLEN/XPENDING/XINFO/XRANGE/XREADGROUP/
XAUTOCLAIM/XACK/XDEL 的语义）跑真实的三段分类，而不是 mock 掉分类本身——
分类逻辑正是这次要验的东西。
"""

from __future__ import annotations

import msgpack
import pytest
from bladex_proxy.models import Identity, Turn, TurnStatus
from bladex_proxy.storage.pipeline_redis import PipelineRedis
from bladex_proxy.storage.pipeline_worker import PipelineWorker, _id_gt


def _make_turn(text: str = "hello") -> Turn:
    return Turn(
        identity=Identity(user_id="u1", agent_id="a1", session_id="s1"),
        model="test-model",
        request_messages=[{"role": "user", "content": "hi"}],
        response_text=text,
        status=TurnStatus.OK,
    )


def _pack(turn: Turn) -> bytes:
    return msgpack.packb(turn.model_dump(mode="json"), use_bin_type=True)


def _key(entry_id: str) -> str:
    return _make_turn().identity.storage_key(entry_id)


class FakeStream:
    """够用的 Redis Streams 语义模拟：条目表 + PEL + 投递游标。"""

    def __init__(self) -> None:
        self.entries: dict[str, bytes] = {}
        self.pel: set[str] = set()
        self.last_delivered = "0-0"
        self._seq = 0

    # -- 造数据的三个入口，对应三种处境 --------------------------------

    def add_undelivered(self, turn: Turn) -> str:
        eid = self._next_id()
        self.entries[eid] = _pack(turn)
        return eid

    def add_pending(self, turn: Turn) -> str:
        eid = self.add_undelivered(turn)
        self.pel.add(eid)
        self.last_delivered = eid
        return eid

    def add_orphan(self, turn: Turn) -> str:
        """已投递、已 XACK、但没被 XDEL——谁都不会再碰它。"""
        eid = self.add_undelivered(turn)
        self.last_delivered = eid
        return eid

    def _next_id(self) -> str:
        self._seq += 1
        return f"{self._seq}-0"

    @staticmethod
    def _parts(v: str) -> tuple[int, int]:
        ms, _, seq = v.partition("-")
        return (int(ms), int(seq or 0))

    def _sorted_ids(self) -> list[str]:
        return sorted(self.entries, key=self._parts)

    # -- Redis 命令 ---------------------------------------------------

    async def xlen(self, _stream: str) -> int:
        return len(self.entries)

    async def xpending(self, _stream: str, _group: str) -> dict:
        return {"pending": len(self.pel)}

    async def xpending_range(self, _stream: str, _group: str, min: str = "-",  # noqa: A002
                             max: str = "+", count: int = 10) -> list[dict]:  # noqa: A002
        return [{"message_id": eid.encode()}
                for eid in sorted(self.pel, key=self._parts)[:count]]

    async def xinfo_groups(self, _stream: str) -> list[dict]:
        lag = sum(1 for eid in self.entries
                  if self._parts(eid) > self._parts(self.last_delivered))
        return [{"name": b"g", "last-delivered-id": self.last_delivered.encode(),
                 "lag": lag, "consumers": 1}]

    async def xrange(self, _stream: str, min: str = "-",  # noqa: A002
                     max: str = "+", count: int = 50) -> list:  # noqa: A002
        ids = self._sorted_ids()
        if min.startswith("("):
            floor = self._parts(min[1:])
            ids = [i for i in ids if self._parts(i) > floor]
        return [(eid.encode(), {b"data": self.entries[eid]}) for eid in ids[:count]]

    async def xreadgroup(self, _group: str, _consumer: str, _streams: dict,
                         count: int = 1, block: int | None = None) -> list:
        assert block != 0, "BLOCK 0 是永远阻塞，force_drain 绝不能这么调"
        fresh = [eid for eid in self._sorted_ids()
                 if self._parts(eid) > self._parts(self.last_delivered)][:count]
        if not fresh:
            return []
        for eid in fresh:
            self.pel.add(eid)
            self.last_delivered = eid
        return [(b"s", [(eid.encode(), {b"data": self.entries[eid]}) for eid in fresh])]

    async def xautoclaim(self, _stream: str, _group: str, _consumer: str,
                         min_idle_time: int = 0, start_id: str = "0",
                         count: int = 10) -> list:
        claimed = sorted(self.pel, key=self._parts)[:count]
        return [b"0-0", [(eid.encode(), {b"data": self.entries[eid]}) for eid in claimed], []]

    async def xack(self, _stream: str, _group: str, entry_id: str) -> int:
        self.pel.discard(entry_id)
        return 1

    async def xdel(self, _stream: str, entry_id: str) -> int:
        return 1 if self.entries.pop(entry_id, None) is not None else 0


class FakeLedger:
    def __init__(self) -> None:
        self.store: dict[str, Turn] = {}
        self.fail = False

    def open(self) -> None:  # PipelineWorker.start 会调，这里用不到
        pass

    def scan_meta(self, prefix: str = ""):
        for key in self.store:
            if key.startswith(prefix):
                yield (key, "")

    def get(self, key: str) -> Turn | None:
        return self.store.get(key)

    def put(self, key: str, turn: Turn) -> None:
        if self.fail:
            raise RuntimeError("ledger write failed")
        self.store[key] = turn


def _wire(stream: FakeStream) -> tuple[PipelineRedis, FakeLedger, PipelineWorker]:
    pipeline = PipelineRedis(redis_url="redis://localhost:0", stream="s",
                             group="g", consumer="c", queue_max_len=100)
    pipeline._redis = stream  # type: ignore[assignment]
    ledger = FakeLedger()
    return pipeline, ledger, PipelineWorker(pipeline, ledger)


# ── 分类 ──────────────────────────────────────────────────────────


def test_id_gt_compares_numerically():
    """`10-0` 比 `9-0` 大——字符串比较会判反，孤儿就会被错当成待投递。"""
    assert _id_gt("10-0", "9-0")
    assert _id_gt("5-2", "5-1")
    assert not _id_gt("5-1", "5-1")


@pytest.mark.asyncio
async def test_inspect_splits_backlog_into_three():
    """XLEN 这一个数必须拆成三段，否则"排不动"会被读成"排队中"。"""
    stream = FakeStream()
    stream.add_orphan(_make_turn())
    stream.add_orphan(_make_turn())
    stream.add_pending(_make_turn())
    stream.add_undelivered(_make_turn())
    pipeline, _, _ = _wire(stream)

    d = await pipeline.inspect()
    assert d["total"] == 4
    assert d["undelivered"] == 1
    assert d["pending"] == 1
    assert d["orphan"] == 2


@pytest.mark.asyncio
async def test_inspect_reports_unknown_orphan_without_lag():
    """Redis 不报 lag 时孤儿数报 None，不拿 total 反推——反推会把正常排队误报成卡死。"""
    stream = FakeStream()
    stream.add_undelivered(_make_turn())

    async def _no_lag(_stream: str) -> list[dict]:
        return [{"name": b"g", "last-delivered-id": b"0-0", "lag": None}]

    stream.xinfo_groups = _no_lag  # type: ignore[assignment]
    pipeline, _, _ = _wire(stream)

    d = await pipeline.inspect()
    assert d["undelivered"] is None
    assert d["orphan"] is None


# ── dry-run ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_dry_run_changes_nothing():
    stream = FakeStream()
    e1 = stream.add_orphan(_make_turn())
    stream.add_orphan(_make_turn())
    stream.add_pending(_make_turn())
    stream.add_undelivered(_make_turn())
    _, ledger, worker = _wire(stream)
    ledger.store[_key(e1)] = _make_turn()

    report = await worker.force_drain(apply=False)

    assert report["mode"] == "dry-run"
    assert report["orphan_total"] == 2
    assert report["orphan_in_hub"] == 1
    assert report["orphan_missing"] == 1
    assert report["orphan_deleted"] == 0
    assert report["delivered"] == 0 and report["reclaimed"] == 0
    assert len(stream.entries) == 4, "预览绝不能改动队列"
    assert len(ledger.store) == 1, "预览绝不能写 Ledger"


# ── apply ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_apply_drains_all_three_kinds():
    stream = FakeStream()
    orphan_id = stream.add_orphan(_make_turn("orphan"))
    stream.add_pending(_make_turn("pending"))
    stream.add_undelivered(_make_turn("undelivered"))
    _, ledger, worker = _wire(stream)
    ledger.store[_key(orphan_id)] = _make_turn("orphan")

    report = await worker.force_drain(apply=True)

    assert report["delivered"] == 1
    assert report["reclaimed"] == 1
    assert report["orphan_in_hub"] == 1
    assert report["orphan_deleted"] == 1
    assert report["after"]["total"] == 0, "排空后 XLEN 必须归零"
    assert stream.pel == set()


@pytest.mark.asyncio
async def test_apply_writes_orphan_missing_from_ledger_before_deleting():
    """Ledger 里没有的孤儿必须先补写再删——删错一条就是永久丢一轮对话。"""
    stream = FakeStream()
    orphan_id = stream.add_orphan(_make_turn("never-stored"))
    _, ledger, worker = _wire(stream)

    report = await worker.force_drain(apply=True)

    assert report["orphan_missing"] == 1
    assert report["orphan_recovered"] == 1
    assert ledger.store[_key(orphan_id)].response_text == "never-stored"
    assert stream.entries == {}


@pytest.mark.asyncio
async def test_ledger_write_failure_keeps_entry_and_reports_it():
    """写 Ledger 失败就不许删条目——宁可继续卡着，也不能静默丢数据。"""
    stream = FakeStream()
    stream.add_orphan(_make_turn())
    _, ledger, worker = _wire(stream)
    ledger.fail = True

    report = await worker.force_drain(apply=True)

    assert report["orphan_missing"] == 1
    assert report["orphan_recovered"] == 0
    assert report["orphan_deleted"] == 0
    assert len(stream.entries) == 1, "没落账的条目必须留在队列里"


@pytest.mark.asyncio
async def test_stuck_pending_does_not_loop_forever():
    """写 Ledger 一直失败的 PEL 条目会被 XAUTOCLAIM 反复捞回——必须能停下来。

    这个命令本来就是给"卡住了"的人用的，它自己不能卡住。
    """
    stream = FakeStream()
    for _ in range(3):
        stream.add_pending(_make_turn())
    _, ledger, worker = _wire(stream)
    ledger.fail = True

    report = await worker.force_drain(apply=True, max_batches=50)

    assert report["reclaimed"] == 0
    assert report["failed"] == 3
    assert report["truncated"] is True
    assert len(stream.entries) == 3


@pytest.mark.asyncio
async def test_sample_contrasts_stale_entry_against_its_session_in_the_ledger():
    """"0 条在 Ledger 里"有两种读法——真没落账 vs key 算法漂了。

    对照量是同会话邻居数：邻居一堆而独独缺这条 = 真没落账；整个会话前缀都空 = 该怀疑
    是不是根本没查对地方。画像必须把这个对照量摆出来，而不是只报一个 in_ledger=False。
    """
    stream = FakeStream()
    stale_id = stream.add_orphan(_make_turn())
    _, ledger, worker = _wire(stream)
    ledger.store[_key("999-0")] = _make_turn()  # 同会话的邻居轮次

    report = await worker.force_drain(apply=False, sample=3)

    assert report["orphan_agents"] == {"a1": 1}
    assert report["orphan_first_ts"] and report["orphan_last_ts"]
    (s,) = report["samples"]
    assert s["entry_id"] == stale_id
    assert s["in_hub"] is False
    assert s["session_turns_in_hub"] == 1


@pytest.mark.asyncio
async def test_sample_zero_collects_nothing():
    """默认不取样：逐条扫 Ledger 前缀是有成本的，不能白付。"""
    stream = FakeStream()
    stream.add_orphan(_make_turn())
    _, _, worker = _wire(stream)

    report = await worker.force_drain(apply=False)
    assert report["samples"] == []
    assert report["orphan_total"] == 1


@pytest.mark.asyncio
async def test_scan_skips_undecodable_entry_without_failing_the_sweep():
    """单条解不开不能让整趟扫描断在那儿——否则一条毒条目挡住它后面所有孤儿。"""
    stream = FakeStream()
    stream.add_orphan(_make_turn())
    bad_id = stream.add_orphan(_make_turn())
    stream.entries[bad_id] = b"\xff\xfe not msgpack"
    pipeline, _, _ = _wire(stream)

    entries = await pipeline.scan_entries()
    assert [eid for eid, _ in entries] == [e for e in stream.entries if e != bad_id]
