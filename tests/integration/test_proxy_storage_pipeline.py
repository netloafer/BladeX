"""集成测试：Pipeline→Memory Hub 存储管线闭环（T9）。

需要本地 Redis（开 AOF）+ RocksDB。
验证：
  1. Turn 入队 Pipeline → 后台 worker 消费 → 写入 Memory Hub → Memory Hub 里能按 key 读出来
  2. 落库成功后 Pipeline 里已 ack（XACK 后待处理消息归零）
  3. 未 ack 的轮次留在 pending（崩溃重放不丢不重复）
  4. 多会话并发不串

运行条件：redis-cli ping 返回 PONG。
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path

import pytest
import redis
from bladex_proxy.models import Identity, Turn, TurnStatus
from bladex_proxy.storage.memory_hub import MemoryHub
from bladex_proxy.storage.pipeline_redis import PipelineRedis
from bladex_proxy.storage.pipeline_worker import PipelineWorker


def _redis_available() -> bool:
    try:
        r = redis.Redis(host="127.0.0.1", port=6379, socket_connect_timeout=2)
        r.ping()
        r.close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _redis_available(),
    reason="Redis not available at 127.0.0.1:6379",
)


def _make_turn(user: str, agent: str, session: str, seq: int, text: str = "reply") -> Turn:
    return Turn(
        identity=Identity(user_id=user, agent_id=agent, session_id=session, turn_index=seq),
        model="gpt-4o",
        request_messages=[{"role": "user", "content": f"msg-{seq}"}],
        injected_memory="<bladex-memory>test</bladex-memory>",
        response_text=text,
        status=TurnStatus.OK,
    )


def _unique_stream() -> str:
    return f"bladex:test:{int(time.time() * 1_000_000) % 10_000_000}"


@pytest.fixture
def rocksdb_path() -> str:
    return str(Path(tempfile.mkdtemp()) / "rocksdb")


class TestP1EnqueueAck:
    """Pipeline 入队 + ack 验证。"""

    @pytest.mark.asyncio
    async def test_enqueue_then_read(self):
        stream = _unique_stream()
        group = f"grp-{stream}"
        pipeline = PipelineRedis("redis://127.0.0.1:6379/0", stream, group, "c1")
        await pipeline.connect()
        try:
            turn = _make_turn("u1", "a1", "s1", 0, "hello")
            entry_id = await pipeline.enqueue(turn)
            assert entry_id

            turns = await pipeline.read(count=10, block_ms=1000)
            assert len(turns) == 1
            eid, loaded = turns[0]
            assert eid == entry_id
            assert loaded.response_text == "hello"
            assert loaded.identity.user_id == "u1"
        finally:
            await pipeline.close()
            r = redis.Redis.from_url("redis://127.0.0.1:6379/0")
            r.delete(stream)
            r.close()

    @pytest.mark.asyncio
    async def test_ack_removes_from_pending(self):
        stream = _unique_stream()
        group = f"grp-{stream}"
        pipeline = PipelineRedis("redis://127.0.0.1:6379/0", stream, group, "c1")
        await pipeline.connect()
        try:
            turn = _make_turn("u1", "a1", "s1", 0)
            await pipeline.enqueue(turn)

            turns = await pipeline.read(count=1, block_ms=1000)
            assert len(turns) == 1

            await pipeline.ack(turns[0][0])

            r = redis.Redis.from_url("redis://127.0.0.1:6379/0")
            pending = r.xpending(stream, group)
            assert pending["pending"] == 0
            r.close()
        finally:
            await pipeline.close()
            r = redis.Redis.from_url("redis://127.0.0.1:6379/0")
            r.delete(stream)
            r.close()

    @pytest.mark.asyncio
    async def test_unacked_stays_in_pending(self):
        """未 ack 的消息留在 pending（模拟崩溃后可重放）。"""
        stream = _unique_stream()
        group = f"grp-{stream}"
        pipeline = PipelineRedis("redis://127.0.0.1:6379/0", stream, group, "c1")
        await pipeline.connect()
        try:
            turn = _make_turn("u1", "a1", "s1", 0, "replay-me")
            await pipeline.enqueue(turn)

            # 读但不 ack
            turns = await pipeline.read(count=1, block_ms=1000)
            assert len(turns) == 1
            assert turns[0][1].response_text == "replay-me"

            # pending 仍有 1 条
            r = redis.Redis.from_url("redis://127.0.0.1:6379/0")
            pending = r.xpending(stream, group)
            assert pending["pending"] == 1
            r.close()
        finally:
            await pipeline.close()
            r = redis.Redis.from_url("redis://127.0.0.1:6379/0")
            r.delete(stream)
            r.close()


class TestPipelineToLedger:
    """Pipeline→Memory Hub 完整管线验证。"""

    @pytest.mark.asyncio
    async def test_pipeline_processes_turn_to_ledger(self, rocksdb_path: str):
        stream = _unique_stream()
        group = f"grp-{stream}"
        pipeline = PipelineRedis("redis://127.0.0.1:6379/0", stream, group, "c1")
        await pipeline.connect()
        ledger = MemoryHub(rocksdb_path)
        ledger.open()
        worker = PipelineWorker(pipeline, ledger)
        try:
            worker.start()

            # 入队 3 轮
            for i in range(3):
                await pipeline.enqueue(_make_turn("u1", "a1", "s1", i, f"reply-{i}"))

            # 等待 worker 处理
            deadline = time.time() + 5
            while time.time() < deadline:
                if ledger.count() >= 3:
                    break
                await asyncio.sleep(0.1)

            assert ledger.count() == 3

            # M-proxy-1.5 起 Memory Hub key 用 entry_id（追加式），按前缀扫描取回（不依赖 key 格式）
            turns = sorted(ledger.scan_prefix("u1/a1/s1/"),
                           key=lambda kv: kv[1].identity.turn_index)
            assert len(turns) == 3
            for i, (_, loaded) in enumerate(turns):
                assert loaded.response_text == f"reply-{i}"
                assert loaded.identity.turn_index == i

            # pending 归零
            r = redis.Redis.from_url("redis://127.0.0.1:6379/0")
            pending = r.xpending(stream, group)
            assert pending["pending"] == 0
            r.close()
        finally:
            await worker.stop()
            await pipeline.close()
            ledger.close()
            r = redis.Redis.from_url("redis://127.0.0.1:6379/0")
            r.delete(stream)
            r.close()

    @pytest.mark.asyncio
    async def test_pipeline_ledger_write_failure_no_ack(self, rocksdb_path: str):
        """Memory Hub 写入失败时不 ack，消息留在 pending 待重试。"""
        from unittest.mock import patch as _patch

        stream = _unique_stream()
        group = f"grp-{stream}"
        pipeline = PipelineRedis("redis://127.0.0.1:6379/0", stream, group, "c1")
        await pipeline.connect()
        ledger = MemoryHub(rocksdb_path)
        ledger.open()
        worker = PipelineWorker(pipeline, ledger)
        try:
            # mock Memory Hub.put 抛异常模拟写入失败
            with _patch.object(ledger, "put", side_effect=RuntimeError("disk full")):
                worker.start()

                await pipeline.enqueue(_make_turn("u1", "a1", "s1", 0, "should-stay-pending"))

                # 等 worker 尝试（会失败）
                await asyncio.sleep(2)

            # Memory Hub 里没有（put 一直失败）
            assert ledger.get("u1/a1/s1/000000") is None

            # pending 仍有消息
            r = redis.Redis.from_url("redis://127.0.0.1:6379/0")
            pending = r.xpending(stream, group)
            assert pending["pending"] >= 1
            r.close()
        finally:
            await worker.stop()
            await pipeline.close()
            ledger.close()
            r = redis.Redis.from_url("redis://127.0.0.1:6379/0")
            r.delete(stream)
            r.close()

    @pytest.mark.asyncio
    async def test_pipeline_multiple_sessions_no_mixing(self, rocksdb_path: str):
        """多用户×多agent×多轮并发入队，各自轮次不串。"""
        stream = _unique_stream()
        group = f"grp-{stream}"
        pipeline = PipelineRedis("redis://127.0.0.1:6379/0", stream, group, "c1")
        await pipeline.connect()
        ledger = MemoryHub(rocksdb_path)
        ledger.open()
        worker = PipelineWorker(pipeline, ledger)
        try:
            worker.start()

            for user in ["userA", "userB"]:
                for agent in ["codex", "claude"]:
                    for seq in range(2):
                        await pipeline.enqueue(
                            _make_turn(user, agent, "sess1", seq, f"{user}/{agent}/{seq}")
                        )

            deadline = time.time() + 5
            while time.time() < deadline:
                if ledger.count() >= 8:
                    break
                await asyncio.sleep(0.1)

            assert ledger.count() == 8

            for user in ["userA", "userB"]:
                for agent in ["codex", "claude"]:
                    turns = sorted(ledger.scan_prefix(f"{user}/{agent}/sess1/"),
                                   key=lambda kv: kv[1].response_text)
                    assert len(turns) == 2
                    assert [t.response_text for _, t in turns] == [
                        f"{user}/{agent}/0", f"{user}/{agent}/1"
                    ]
        finally:
            await worker.stop()
            await pipeline.close()
            ledger.close()
            r = redis.Redis.from_url("redis://127.0.0.1:6379/0")
            r.delete(stream)
            r.close()


class TestT1TrimAndSpill:
    """T1（ADR-0018）：ack 后 XDEL 使 XLEN≈XPENDING + DiskSpill 最终兜底。

    验收：① enqueue->消费->ack 后 XLEN≈XPENDING（趋近 0，消灭 R3/A3 假溢出）；
          ② pipeline=None 时 Turn 落溢出目录且重启后回灌（消灭 ADR-0017 已知限制）。
    """

    @pytest.mark.asyncio
    async def test_ack_xdels_stream_entry(self):
        """ack 后 stream 条目被 XDEL，XLEN≈XPENDING≈0（R3/A3 核心验收）。"""
        stream = _unique_stream()
        group = f"grp-{stream}"
        pipeline = PipelineRedis("redis://127.0.0.1:6379/0", stream, group, "c1")
        await pipeline.connect()
        try:
            for i in range(3):
                await pipeline.enqueue(_make_turn("u1", "a1", "s1", i, f"r{i}"))

            # 消费并 ack 全部
            turns = await pipeline.read(count=10, block_ms=1000)
            assert len(turns) == 3
            for eid, _ in turns:
                await pipeline.ack(eid)

            r = redis.Redis.from_url("redis://127.0.0.1:6379/0")
            pending = r.xpending(stream, group)["pending"]
            xlen = r.xlen(stream)
            r.close()

            # T1 核心断言：ack+xdel 后 XLEN 与 XPENDING 都趋近 0
            # （修复前：XLEN 持续增长、XPENDING=0，假溢出连锁的根源）
            assert pending == 0
            assert xlen == 0, f"XDEL should drain stream, got XLEN={xlen}"
        finally:
            await pipeline.close()
            r = redis.Redis.from_url("redis://127.0.0.1:6379/0")
            r.delete(stream)
            r.close()

    @pytest.mark.asyncio
    async def test_water_level_uses_pending_not_xlen(self):
        """T1: 水位/满队列判定基于 XPENDING，已 ack 历史不计入。

        修复前（A3）：XLEN 含已 ack 历史 -> 累积后恒满 -> 每次 enqueue 假溢出 + 阻塞 1s。
        修复后：ack+xdel 使 XLEN 回落，pending_count 也回落，不再假溢出。
        """
        stream = _unique_stream()
        group = f"grp-{stream}"
        pipeline = PipelineRedis(
            "redis://127.0.0.1:6379/0", stream, group, "c1",
            queue_max_len=5, spill_timeout_ms=100,
        )
        await pipeline.connect()
        try:
            # 入队 5 轮 + 消费 + ack（都已 xdel）-> stream 空、pending=0
            for i in range(5):
                await pipeline.enqueue(_make_turn("u", "a", "s", i))
            turns = await pipeline.read(count=10, block_ms=1000)
            for eid, _ in turns:
                await pipeline.ack(eid)

            # 此时 pending=0。再入队一轮不应触发 overflow（修复前 XLEN=5>=5 会假溢出）
            assert await pipeline.pending_count() == 0
            entry = await pipeline.enqueue(_make_turn("u", "a", "s", 99))
            assert not entry.startswith("overflow:"), "should not false-overflow when pending=0"
            assert pipeline.overflow_count == 0
        finally:
            await pipeline.close()
            r = redis.Redis.from_url("redis://127.0.0.1:6379/0")
            r.delete(stream)
            r.close()

    @pytest.mark.asyncio
    async def test_disk_spill_replays_after_recover(self, tmp_path):
        """T1/A4: DiskSpill 落盘的 Turn，Redis 恢复后能回灌。

        模拟 ADR-0017 场景：Turn 先落盘，之后 replay 回灌到 stream。
        """
        from bladex_proxy.storage.pipeline_redis import DiskSpill

        spill = DiskSpill(tmp_path)
        turn = _make_turn("u1", "a1", "s1", 0, "spilled-turn")
        spill.spill(turn)
        assert len(list(tmp_path.glob("overflow_*.msgpack"))) == 1

        stream = _unique_stream()
        group = f"grp-{stream}"
        pipeline = PipelineRedis("redis://127.0.0.1:6379/0", stream, group, "c1")
        await pipeline.connect()
        try:
            # 用 pipeline 的 redis 句柄回灌 DiskSpill 落盘的文件
            replayed = await spill.replay(pipeline.redis, stream)
            assert replayed == 1
            assert len(list(tmp_path.glob("overflow_*.msgpack"))) == 0

            # 回灌的条目可被消费
            turns = await pipeline.read(count=10, block_ms=1000)
            assert len(turns) == 1
            assert turns[0][1].response_text == "spilled-turn"
        finally:
            await pipeline.close()
            r = redis.Redis.from_url("redis://127.0.0.1:6379/0")
            r.delete(stream)
            r.close()
