"""Pipeline Redis fallback 路径单元测试（T0 - B026 修复验收 + T1 trim/spill 解耦验收）。"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import msgpack
import pytest
from bladex_proxy.models import Identity, Turn, TurnStatus
from bladex_proxy.storage.pipeline_redis import DiskSpill, PipelineRedis


def _make_turn() -> Turn:
    return Turn(
        identity=Identity(user_id="u1", agent_id="a1", session_id="s1"),
        model="test-model",
        request_messages=[{"role": "user", "content": "hi"}],
        response_text="hello",
        status=TurnStatus.OK,
    )


def _pack_turn(turn: Turn) -> bytes:
    return msgpack.packb(turn.model_dump(mode="json"), use_bin_type=True)


@pytest.mark.asyncio
async def test_claim_pending_fallback_uses_message_ids_keyword():
    """B026 修复后，fallback 用 message_ids= 关键字传 entry_ids。"""
    pipeline = PipelineRedis(
        redis_url="redis://localhost:0", stream="s", group="g",
        consumer="c", queue_max_len=100,
    )

    turn = _make_turn()
    packed = _pack_turn(turn)

    mock_redis = MagicMock()
    mock_redis.xpending_range = AsyncMock(return_value=[
        {"message_id": "1-0"},
        {"message_id": "1-1"},
    ])
    mock_redis.xclaim = AsyncMock(return_value=[
        (b"1-0", {b"data": packed}),
        (b"1-1", {b"data": packed}),
    ])
    pipeline._redis = mock_redis  # type: ignore[assignment]

    results = await pipeline._claim_pending_fallback(count=10, min_idle_ms=5000)

    mock_redis.xclaim.assert_awaited_once()
    call_kwargs = mock_redis.xclaim.call_args
    assert "message_ids" in call_kwargs.kwargs, \
        "xclaim must use message_ids= keyword (B026 fix)"
    assert call_kwargs.kwargs["message_ids"] == ["1-0", "1-1"]

    assert len(results) == 2
    entry_id, parsed_turn = results[0]
    assert entry_id == "1-0"
    assert parsed_turn.response_text == "hello"


@pytest.mark.asyncio
async def test_read_pending_handles_list_xautoclaim_return():
    """redis-py 8.x xautoclaim 返回 list（非 tuple）-> read_pending 必须正确解析。

    回归（PEL 毒消息根因之一）：之前 isinstance(result, tuple) 不匹配 list ->
    返回空 -> 回收的条目不处理不 ack -> 重新 pending -> delivery_count 无限涨。
    """
    pipeline = PipelineRedis(
        redis_url="redis://localhost:0", stream="s", group="g",
        consumer="c", queue_max_len=100,
    )
    turn = _make_turn()
    packed = _pack_turn(turn)
    mock_redis = MagicMock()
    # redis-py 8.x: [next_id, [(entry_id, fields)], [deleted_ids]]  -- list, not tuple
    mock_redis.xautoclaim = AsyncMock(return_value=[
        b"1234-1", [(b"1234-0", {b"data": packed})], [],
    ])
    pipeline._redis = mock_redis  # type: ignore[assignment]

    results = await pipeline.read_pending(count=10, min_idle_ms=5000)
    assert len(results) == 1, "list 返回必须正确解析（isinstance(tuple) 漏 list -> 空 = bug）"
    entry_id, parsed_turn = results[0]
    assert entry_id == "1234-0"  # _parse_raw_entries decode 成 str
    assert parsed_turn.response_text == "hello"


def test_response_meta_coerce_list_reasoning_text():
    """ADR-0020 T4 兼容：旧 Pipeline/Memory Hub 数据 reasoning_text/finish_reason 是 list -> coerce str。

    回归（PEL 毒消息根因之二）：旧数据 reasoning_text=[] 反序列化校验失败 ->
    pipeline_reclaim_error -> 不 ack -> 永远 pending。before validator coerce 修复。
    """
    from bladex_proxy.models import Turn
    turn = Turn(
        identity=Identity(user_id="u", agent_id="a", session_id="s"),
        model="m",
        response_meta={"reasoning_text": ["step1", "step2"], "finish_reason": ["stop", "tool"]},
    )
    assert turn.response_meta.reasoning_text == "step1step2"
    assert turn.response_meta.finish_reason == "stop,tool"

    # 空 list 也要 coerce 成空串（不抛）
    turn2 = Turn(
        identity=Identity(user_id="u", agent_id="a", session_id="s"),
        model="m",
        response_meta={"reasoning_text": [], "finish_reason": []},
    )
    assert turn2.response_meta.reasoning_text == ""
    assert turn2.response_meta.finish_reason == ""


@pytest.mark.asyncio
async def test_claim_pending_fallback_empty_pending():
    """No pending entries -> empty result, xclaim not called."""
    pipeline = PipelineRedis(
        redis_url="redis://localhost:0", stream="s", group="g",
        consumer="c", queue_max_len=100,
    )

    mock_redis = MagicMock()
    mock_redis.xpending_range = AsyncMock(return_value=[])
    mock_redis.xclaim = AsyncMock()
    pipeline._redis = mock_redis  # type: ignore[assignment]

    results = await pipeline._claim_pending_fallback(count=10, min_idle_ms=5000)
    assert results == []
    mock_redis.xclaim.assert_not_awaited()


@pytest.mark.asyncio
async def test_claim_pending_fallback_xpending_raises():
    """XPENDING raises -> graceful empty return."""
    pipeline = PipelineRedis(
        redis_url="redis://localhost:0", stream="s", group="g",
        consumer="c", queue_max_len=100,
    )

    mock_redis = MagicMock()
    mock_redis.xpending_range = AsyncMock(side_effect=Exception("conn lost"))
    pipeline._redis = mock_redis  # type: ignore[assignment]

    results = await pipeline._claim_pending_fallback(count=10, min_idle_ms=5000)
    assert results == []


# ── T1: 背压 + 磁盘溢出测试 ──

def _make_turn_for_t1(label: str = "t1") -> Turn:
    return Turn(
        identity=Identity(user_id="u1", agent_id="a1", session_id="s1"),
        model="test-model",
        request_messages=[{"role": "user", "content": label}],
        response_text="response",
        status=TurnStatus.OK,
    )


@pytest.mark.asyncio
async def test_pipeline_overflow_alert():
    """队列达阈值时产生分级结构化报警事件。

    N1 修复：水位判定改回 XLEN，mock xlen 返回满。
    """
    pipeline = PipelineRedis(
        redis_url="redis://localhost:0", stream="s", group="g",
        consumer="c", queue_max_len=100,
        overflow_dir="/tmp/test_bladex_overflow_alert",
        warn_pct=0.6, alert_pct=0.8, critical_pct=0.9,
        spill_timeout_ms=100,
    )

    turn = _make_turn_for_t1("overflow_alert")

    mock_redis = MagicMock()
    # N1 修复：backlog_len -> xlen 返回 100（满）-> overflow
    mock_redis.xlen = AsyncMock(return_value=100)
    # After wait, still 100 -> should spill
    mock_redis.xadd = AsyncMock(side_effect=Exception("redis full"))
    pipeline._redis = mock_redis  # type: ignore[assignment]

    result = await pipeline.enqueue(turn)

    # Should have spilled to disk
    assert result.startswith("overflow:")
    assert pipeline.overflow_count >= 1
    assert pipeline.spill_count >= 1

    # Cleanup
    overflow_path = Path("/tmp/test_bladex_overflow_alert")
    if overflow_path.exists():
        for f in overflow_path.glob("overflow_*.msgpack"):
            f.unlink()
        overflow_path.rmdir()


@pytest.mark.asyncio
async def test_pipeline_disk_overflow_spill():
    """模拟 Redis 不可写 -> Turn 落磁盘溢出文件、不丢。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        pipeline = PipelineRedis(
            redis_url="redis://localhost:0", stream="s", group="g",
            consumer="c", queue_max_len=100,
            overflow_dir=tmpdir,
            spill_timeout_ms=50,
        )

        turn = _make_turn_for_t1("disk_spill")

        mock_redis = MagicMock()
        # N1 修复：backlog_len 走 xlen；模拟 Redis 不可达 -> xlen 抛 -> spill
        mock_redis.xlen = AsyncMock(side_effect=Exception("connection refused"))
        pipeline._redis = mock_redis  # type: ignore[assignment]

        result = await pipeline.enqueue(turn)

        # Spilled to disk
        assert result.startswith("overflow:")
        spill_files = list(Path(tmpdir).glob("overflow_*.msgpack"))
        assert len(spill_files) == 1, f"Expected 1 spill file, got {len(spill_files)}"

        # Verify content is recoverable
        raw = spill_files[0].read_bytes()
        data = msgpack.unpackb(raw, raw=False)
        recovered = Turn.model_validate(data)
        assert recovered.response_text == "response"
        assert recovered.identity.user_id == "u1"


@pytest.mark.asyncio
async def test_pipeline_overflow_replay():
    """磁盘溢出文件恢复后能回灌 Pipeline。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        pipeline = PipelineRedis(
            redis_url="redis://localhost:0", stream="s", group="g",
            consumer="c", queue_max_len=100,
            overflow_dir=tmpdir,
            spill_timeout_ms=50,
        )

        turn = _make_turn_for_t1("replay_test")
        packed = msgpack.packb(turn.model_dump(mode="json"), use_bin_type=True)

        # Spill first (旧格式文件名也兼容 glob overflow_*.msgpack)
        spill_file = Path(tmpdir) / "overflow_1000_1.msgpack"
        spill_file.write_bytes(packed)

        # Mock Redis that accepts xadd
        mock_redis = MagicMock()
        mock_redis.xadd = AsyncMock(return_value=b"1719907200-0")
        pipeline._redis = mock_redis  # type: ignore[assignment]

        replayed = await pipeline.replay_overflow()

        assert replayed == 1
        mock_redis.xadd.assert_awaited_once()
        # File should be deleted after successful replay
        assert not spill_file.exists()


@pytest.mark.asyncio
async def test_pipeline_water_level_tiered_alerts():
    """不同水位触发不同报警级别（N1：参数为 XLEN stream 长度）。"""
    pipeline = PipelineRedis(
        redis_url="redis://localhost:0", stream="s", group="g",
        consumer="c", queue_max_len=1000,
        warn_pct=0.6, alert_pct=0.8, critical_pct=0.9,
    )

    pipeline._check_water_level(600)   # 60% -> warn
    pipeline._check_water_level(800)   # 80% -> alert
    pipeline._check_water_level(900)   # 90% -> critical
    pipeline._check_water_level(100)   # Below threshold -> no alert


@pytest.mark.asyncio
async def test_pipeline_backpressure_uses_xlen_not_xpending():
    """N1 核心：背压用 XLEN，消费者宕机（条目未投递、不进 PEL）时仍触发水位报警。

    XPENDING 对此失明（恒 0）；XLEN 计入未投递条目 -> 报警触发。enqueue 不应调用
    xpending 做背压判定。
    """
    pipeline = PipelineRedis(
        redis_url="redis://localhost:0", stream="s", group="g",
        consumer="c", queue_max_len=100,
        warn_pct=0.6, alert_pct=0.8, critical_pct=0.9,
    )
    turn = _make_turn_for_t1("consumer_down")
    mock_redis = MagicMock()
    # 消费者宕机：XLEN=90（critical），但 XPENDING=0（未投递不进 PEL）
    mock_redis.xlen = AsyncMock(return_value=90)
    mock_redis.xpending = AsyncMock(return_value={"pending": 0})
    mock_redis.xadd = AsyncMock(return_value=b"1719907200-0")
    pipeline._redis = mock_redis  # type: ignore[assignment]

    result = await pipeline.enqueue(turn)
    # XLEN=90 < 100 未满 -> enqueue 仍成功 xadd（但 _check_water_level 已触发 critical）
    assert result == "1719907200-0"
    mock_redis.xlen.assert_awaited()
    # 背压判定不应走 xpending（XPENDING 对消费者宕机失明，正是 N1 要修的）
    mock_redis.xpending.assert_not_awaited()


@pytest.mark.asyncio
async def test_pipeline_xlen_drops_after_ack_xdel_no_false_overflow():
    """N1 + T1 不回退：ack+XDEL 后 XLEN 回落 -> enqueue 不触发假溢出。"""
    pipeline = PipelineRedis(
        redis_url="redis://localhost:0", stream="s", group="g",
        consumer="c", queue_max_len=100,
        spill_timeout_ms=50,
    )
    turn = _make_turn_for_t1("after_drain")
    mock_redis = MagicMock()
    # XLEN=10（远低于 100）-> 不满、不报警、不 spill
    mock_redis.xlen = AsyncMock(return_value=10)
    mock_redis.xadd = AsyncMock(return_value=b"1719907200-0")
    pipeline._redis = mock_redis  # type: ignore[assignment]

    result = await pipeline.enqueue(turn)
    assert result == "1719907200-0"
    assert pipeline.overflow_count == 0
    assert pipeline.spill_count == 0


# ── T1: ack 后 XDEL + DiskSpill 独立类 ──

@pytest.mark.asyncio
async def test_pipeline_ack_xdels_entry():
    """T1/R3/A3: ack 成功后 XDEL 删除条目，使 XLEN≈XPENDING。

    XACK 只从 PEL 移除；XDEL 才真正删 stream 条目，消灭假溢出连锁。
    XDEL 失败不致命（只 warning），XACK 已成功。
    """
    pipeline = PipelineRedis(
        redis_url="redis://localhost:0", stream="s", group="g",
        consumer="c", queue_max_len=100,
    )
    mock_redis = MagicMock()
    mock_redis.xack = AsyncMock(return_value=1)
    mock_redis.xdel = AsyncMock(return_value=1)
    pipeline._redis = mock_redis  # type: ignore[assignment]

    await pipeline.ack("1719907200-0")

    mock_redis.xack.assert_awaited_once_with("s", "g", "1719907200-0")
    mock_redis.xdel.assert_awaited_once_with("s", "1719907200-0")


@pytest.mark.asyncio
async def test_pipeline_ack_xdel_failure_non_fatal():
    """XDEL 失败不致命：XACK 已成功，条目仍在 PEL 会被 XAUTOCLAIM 重投。"""
    pipeline = PipelineRedis(
        redis_url="redis://localhost:0", stream="s", group="g",
        consumer="c", queue_max_len=100,
    )
    mock_redis = MagicMock()
    mock_redis.xack = AsyncMock(return_value=1)
    mock_redis.xdel = AsyncMock(side_effect=Exception("xdel boom"))
    pipeline._redis = mock_redis  # type: ignore[assignment]

    # 不应抛异常
    await pipeline.ack("1719907200-0")
    mock_redis.xack.assert_awaited_once()


def test_disk_spill_independent_of_redis():
    """T1/A4: DiskSpill 不依赖 Redis 连接，纯文件写。

    server 侧最终 fallback 无条件落盘的基石：pipeline is None 时也能落盘。
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        spill = DiskSpill(tmpdir)
        turn = _make_turn_for_t1("no_redis")

        marker = spill.spill(turn)

        assert marker.startswith("overflow:")
        files = list(Path(tmpdir).glob("overflow_*.msgpack"))
        assert len(files) == 1
        assert spill.spill_count == 1

        # 内容可还原
        data = msgpack.unpackb(files[0].read_bytes(), raw=False)
        recovered = Turn.model_validate(data)
        assert recovered.response_text == "no_redis" or recovered.identity.user_id == "u1"


@pytest.mark.asyncio
async def test_disk_spill_replay():
    """T1: DiskSpill.replay 回灌到给定 redis stream，成功后删文件。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        spill = DiskSpill(tmpdir)
        turn = _make_turn_for_t1("replay")
        spill.spill(turn)
        spill.spill(turn)

        mock_redis = MagicMock()
        mock_redis.xadd = AsyncMock(return_value=b"1719907200-0")
        replayed = await spill.replay(mock_redis, "s")

        assert replayed == 2
        assert mock_redis.xadd.await_count == 2
        # 文件全删
        assert list(Path(tmpdir).glob("overflow_*.msgpack")) == []


# ── T1: L2 已实现（G7），L3 仍 stub ──

def test_pipeline_l2_truncate_implemented():
    """L2 降级已实现（G7）：超大 tool/assistant content 截断成 head+tail+标记。

    user/system 消息不截断（含事实/指令，截断损害 consolidation）。
    """
    pipeline = PipelineRedis(
        redis_url="redis://localhost:0", stream="s", group="g",
        consumer="c",
    )
    big = "x" * 10000
    turn = Turn(
        identity=Identity(user_id="u", agent_id="a", session_id="s"),
        model="m",
        request_messages=[
            {"role": "user", "content": "keep me"},
            {"role": "assistant", "content": big},
            {"role": "tool", "content": big},
        ],
        status=TurnStatus.OK,
    )

    truncated = pipeline._degrade_l2_truncate(turn)

    # user 不截断
    assert truncated.request_messages[0]["content"] == "keep me"
    # assistant / tool 截断（含标记）
    for idx in (1, 2):
        c = truncated.request_messages[idx]["content"]
        assert "truncated_under_load" in c
        assert len(c) < 10000


def test_pipeline_l3_still_stub():
    """三级降级完整版仍是 stub（M-proxy-2 不实现）。"""
    pipeline = PipelineRedis(
        redis_url="redis://localhost:0", stream="s", group="g",
        consumer="c",
    )
    with pytest.raises(NotImplementedError, match="L3 degradation"):
        pipeline._degrade_l3_spill_full()
