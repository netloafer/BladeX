"""Pipeline Redis Streams 缓冲层 - 接入快、异步往后送（T5，ADR-0009 §4-5）。

落库成功才 ack（不丢不重复）。
背压（ADR-0009 §5b）：
  - 队列水位过阈值时分级报警（warn/alert/critical）
  - 满队列阻塞超时后 spill 到磁盘溢出文件（绝不 MAXLEN trim 未 ack 数据）
  - 二级（削肥肉）、三级（完整版）降级为 stub，留后续

T1（ADR-0018）：ack 后 XDEL 删条目（消灭 R3/A3 假溢出连锁）；
N1 修复（ADR-0018 复核）：水位与满队列判定改回 XLEN--XDEL 后 XLEN = 未 ack + 未投递
= 真实积压；XPENDING 只见已投递未 ack，消费者宕机时新条目不进 PEL -> 报警失明、stream 无界增长。
磁盘溢出提为独立 DiskSpill 类（不依赖 Redis 连接）--server 侧最终 fallback
无条件落盘，消灭 ADR-0017"pipeline is None 期间 Turn 裸丢"已知限制。
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import msgpack
import redis.asyncio as aioredis
import structlog

from bladex_proxy.models import Turn

logger = structlog.get_logger()


class DiskSpill:
    """磁盘溢出 - Turn 落本地文件的最终兜底（ADR-0009 §5b，T1 提为独立类）。

    不依赖 Redis 连接：纯本地文件写，保证"任何情况下 Turn 至少落在本地磁盘"
    成为不变式（消灭 ADR-0017 已知限制：pipeline is None / 60s 冷却期 / enqueue 抛异常
    三种情况 Turn 不再裸丢）。

    文件名格式：overflow_{timestamp}_{counter}.msgpack
    恢复时由 replay() 回灌 Pipeline stream。
    """

    def __init__(self, overflow_dir: str | Path) -> None:
        self._overflow_dir = Path(overflow_dir)
        self._spill_count = 0

    @property
    def overflow_dir(self) -> Path:
        return self._overflow_dir

    @property
    def spill_count(self) -> int:
        """磁盘溢出 spill 次数（Redis 不可写时落盘的次数）。"""
        return self._spill_count

    def spill(self, turn: Turn, packed_value: bytes | None = None) -> str:
        """把 Turn 序列化落磁盘溢出文件（同步，纯文件 I/O）。

        返回溢出标记 "overflow:{filename}"，供调用方记录/测试断言。
        """
        self._spill_count += 1
        self._overflow_dir.mkdir(parents=True, exist_ok=True)

        # uuid 后缀防多实例（app 级 DiskSpill + PipelineRedis 内部 DiskSpill）同毫秒同计数撞名
        import uuid
        ts = int(time.time() * 1000)
        filename = f"overflow_{ts}_{self._spill_count}_{uuid.uuid4().hex[:6]}.msgpack"
        filepath = self._overflow_dir / filename

        if packed_value is None:
            data = turn.model_dump(mode="json")
            packed_value = msgpack.packb(data, use_bin_type=True)

        filepath.write_bytes(packed_value)
        logger.warning(
            "pipeline_disk_spill",
            filepath=str(filepath),
            spill_total=self._spill_count,
            size=len(packed_value),
        )
        return f"overflow:{filename}"

    async def replay(self, redis: aioredis.Redis, stream: str) -> int:
        """回灌磁盘溢出文件到 Redis Streams。

        在 pipeline 启动 / Redis 懒恢复成功后调用。成功回灌后删除溢出文件。
        返回回灌的条目数。
        """
        if not self._overflow_dir.exists():
            return 0

        replayed = 0
        for filepath in sorted(self._overflow_dir.glob("overflow_*.msgpack")):
            try:
                value = filepath.read_bytes()
                entry_id = await redis.xadd(stream, {"data": value})
                eid = entry_id.decode() if isinstance(entry_id, bytes) else entry_id
                filepath.unlink()
                replayed += 1
                logger.info("pipeline_overflow_replayed", filepath=str(filepath), entry_id=eid)
            except Exception as e:
                logger.warning("pipeline_overflow_replay_failed", filepath=str(filepath), error=str(e))
                # 留着文件等下次重试
                break

        if replayed > 0:
            logger.info("pipeline_overflow_replay_done", replayed=replayed)
        return replayed


class PipelineRedis:
    """Redis Streams 封装：enqueue + 消费组管理 + PEL 回收 + 背压。"""

    def __init__(
        self,
        redis_url: str,
        stream: str,
        group: str,
        consumer: str,
        queue_max_len: int = 10000,
        overflow_dir: str = "data/overflow",
        warn_pct: float = 0.6,
        alert_pct: float = 0.8,
        critical_pct: float = 0.9,
        spill_timeout_ms: int = 1000,
        heavy_payload_bytes: int = 262144,
    ) -> None:
        self._redis_url = redis_url
        self._stream = stream
        self._group = group
        self._consumer = consumer
        self._queue_max_len = queue_max_len
        self._disk_spill = DiskSpill(overflow_dir)
        self._warn_pct = warn_pct
        self._alert_pct = alert_pct
        self._critical_pct = critical_pct
        self._spill_timeout_s = spill_timeout_ms / 1000.0
        self._heavy_payload_bytes = heavy_payload_bytes
        self._redis: aioredis.Redis | None = None
        self._overflow_count = 0

    async def connect(self) -> None:
        """连接 Redis 并确保消费组存在。

        socket_timeout: 防止空闲连接卡死 event loop（Redis idle 后 TCP 连接半死，
        不设 timeout 会阻塞到系统级 TCP 超时 ~60s）。
        socket_keepalive: 主动探测死连接。
        health_check_interval: 定期 PING 检测连接健康。
        """
        self._redis = aioredis.from_url(
            self._redis_url,
            decode_responses=False,
            socket_timeout=5.0,
            socket_keepalive=True,
            health_check_interval=30,
        )
        try:
            await self._redis.xgroup_create(self._stream, self._group, id="0", mkstream=True)
            logger.info("pipeline_group_created", stream=self._stream, group=self._group)
        except aioredis.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.info("pipeline_group_exists", stream=self._stream)
            else:
                raise
        logger.info("pipeline_redis_connected", url=self._redis_url)

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None
            logger.info("pipeline_redis_closed")

    @property
    def redis(self) -> aioredis.Redis:
        if self._redis is None:
            raise RuntimeError("PipelineRedis not connected - call connect() first")
        return self._redis

    @property
    def disk_spill(self) -> DiskSpill:
        """暴露 DiskSpill 供 server 侧最终 fallback 直接落盘用（T1）。"""
        return self._disk_spill

    @property
    def overflow_count(self) -> int:
        """队列溢出次数（达到 max_len 的次数）。"""
        return self._overflow_count

    @property
    def spill_count(self) -> int:
        """磁盘溢出 spill 次数（Redis 不可写时落盘的次数）。"""
        return self._disk_spill.spill_count

    async def enqueue(self, turn: Turn) -> str:
        """把一轮 Turn 入队（XADD）。返回 stream entry id 或溢出标记。

        背压策略（ADR-0009 §5b）：
          1. 队列水位过阈值 -> 分级报警（warn/alert/critical）
          2. 队列满 -> 阻塞等待消费者追上（最多 spill_timeout_ms）
          3. 仍满或 Redis 不可写 -> spill 到磁盘溢出文件
          绝不 MAXLEN trim 未 ack 数据（§5b 红线）。

        N1 修复：水位与满队列判定用 XLEN（stream 长度）。XDEL 后 XLEN = 未 ack + 未投递
        = 真实积压；XPENDING 只数已投递未 ack，消费者宕机时新 XADD 不进 PEL -> 恒 ≈0、
        报警永不触发、stream 无界增长（正是背压要保护的场景）。pending_count()（XPENDING）
        保留作监控接口，不用于背压判定。
        """
        data = turn.model_dump(mode="json")
        value = msgpack.packb(data, use_bin_type=True)

        # 重车道判定（ADR-0009 §5b 两条道分流）
        is_heavy = len(value) > self._heavy_payload_bytes
        if is_heavy:
            logger.info("pipeline_heavy_payload", size=len(value),
                        threshold=self._heavy_payload_bytes)

        try:
            async with asyncio.timeout(2.0):
                pending = await self.backlog_len()
        except Exception as e:
            # Redis 不可用或超时 -> 直接 spill
            logger.warning("pipeline_redis_unreachable_on_enqueue", error=str(e))
            return self._disk_spill.spill(turn, value)

        # 分级水位报警（基于 XLEN 真实积压 = 未 ack + 未投递）
        self._check_water_level(pending)

        if pending >= self._queue_max_len:
            self._overflow_count += 1
            logger.warning(
                "pipeline_queue_overflow",
                pending=pending,
                max_len=self._queue_max_len,
                overflow_total=self._overflow_count,
            )
            # 阻塞等待消费者追上
            await self._wait_for_drain()

            # 重新检查
            try:
                pending = await self.backlog_len()
            except Exception as e:
                logger.warning("pipeline_redis_unreachable_after_wait", error=str(e))
                return self._disk_spill.spill(turn, value)

            if pending >= self._queue_max_len:
                # G7: heavy payload 先截断再重试，避免直接 spill
                if is_heavy:
                    truncated_turn = self._degrade_l2_truncate(turn)
                    truncated_data = truncated_turn.model_dump(mode="json")
                    truncated_value = msgpack.packb(truncated_data, use_bin_type=True)
                    try:
                        async with asyncio.timeout(2.0):
                            entry_id = await self.redis.xadd(
                                self._stream, {"data": truncated_value})
                        eid = entry_id.decode() if isinstance(entry_id, bytes) else entry_id
                        logger.warning("pipeline_l2_truncate_enqueued",
                                       original_size=len(value),
                                       truncated_size=len(truncated_value))
                        return eid
                    except Exception as e:
                        logger.warning("pipeline_l2_truncate_xadd_failed", error=str(e))
                # 仍满 -> spill 到磁盘
                return self._disk_spill.spill(turn, value)

        # 不用 MAXLEN - 那会静默删未 ack 条目（ADR-0009 §5b 红线）
        try:
            async with asyncio.timeout(2.0):
                entry_id = await self.redis.xadd(self._stream, {"data": value})
        except Exception as e:
            logger.warning("pipeline_xadd_failed", error=str(e))
            return self._disk_spill.spill(turn, value)

        eid = entry_id.decode() if isinstance(entry_id, bytes) else entry_id
        logger.debug("pipeline_enqueued", entry_id=eid, key=turn.identity.session_prefix())
        return eid

    def _check_water_level(self, pending: int) -> None:
        """根据 PEL 积压水位发分级报警（ADR-0009 §5b 一级降级观测）。

        T1：参数从 stream_len（XLEN）改为 pending（XPENDING），反映真实积压。
        N1 修复：改回 XLEN stream 长度（XDEL 后 = 未 ack + 未投递 = 真实积压）。
        """
        if self._queue_max_len <= 0:
            return
        ratio = pending / self._queue_max_len
        if ratio >= self._critical_pct:
            logger.error(
                "pipeline_water_level_critical",
                pending=pending,
                max_len=self._queue_max_len,
                ratio=round(ratio, 3),
            )
        elif ratio >= self._alert_pct:
            logger.warning(
                "pipeline_water_level_alert",
                pending=pending,
                max_len=self._queue_max_len,
                ratio=round(ratio, 3),
            )
        elif ratio >= self._warn_pct:
            logger.info(
                "pipeline_water_level_warn",
                pending=pending,
                max_len=self._queue_max_len,
                ratio=round(ratio, 3),
            )

    async def _wait_for_drain(self) -> None:
        """阻塞等待消费者追上（短轮询，不超过 spill_timeout）。

        T1：判空基准从 XLEN 改 XPENDING--消费者 ack+xdel 后 PEL 缩短即返回。
        N1 修复：改回 XLEN--消费者 ack+xdel 后 stream 缩短即返回（且对消费者宕机可见）。
        """
        deadline = time.monotonic() + self._spill_timeout_s
        while time.monotonic() < deadline:
            await asyncio.sleep(0.1)
            try:
                pending = await self.backlog_len()
                if pending < self._queue_max_len:
                    return
            except Exception:
                return  # Redis 不可用，上层会 spill

    async def replay_overflow(self) -> int:
        """回灌磁盘溢出文件到 Redis Streams（T1：委托给 DiskSpill）。"""
        return await self._disk_spill.replay(self.redis, self._stream)

    # ── 二、三级降级 stub（ADR-0009 §5b，M-proxy-2 不实现）──

    def _degrade_l2_truncate(self, turn: Turn) -> Turn:
        """二级降级：把超大内容截断成头尾摘录 + 标记（G7, ADR-0009 §5b）。

        遍历 request_messages，把超长 content 截断成 head + tail + 标记。
        只截断 tool / assistant 角色（巨型工具返回/模型输出），
        user / system 消息不截断（含事实/指令，截断会损害 consolidation）。
        返回截断后的新 Turn（不改原对象）。
        """
        import copy

        _TRUNCATE_ROLES = {"tool", "assistant"}
        _KEEP_HEAD = 2048
        _KEEP_TAIL = 2048
        _MIN_TRUNCATE_LEN = 8192

        truncated = copy.copy(turn)
        truncated.request_messages = []
        any_truncated = False

        for msg in turn.request_messages:
            msg_copy = dict(msg)
            role = msg_copy.get("role", "")
            content = msg_copy.get("content")

            if role in _TRUNCATE_ROLES and isinstance(content, str) and len(content) > _MIN_TRUNCATE_LEN:
                original_size = len(content)
                msg_copy["content"] = (
                    content[:_KEEP_HEAD]
                    + f"\n...[truncated_under_load, original_size={original_size}]...\n"
                    + content[-_KEEP_TAIL:]
                )
                any_truncated = True

            truncated.request_messages.append(msg_copy)

        if any_truncated:
            logger.info("pipeline_l2_truncated")
        return truncated

    def _degrade_l3_spill_full(self) -> None:
        """三级降级完整版：多消费者并行 + 磁盘溢出优先队列（stub）。

        TODO(M-proxy-3+): 实现完整的磁盘溢出优先队列管理--
        按会话/优先级排序回灌、限制磁盘溢出总量、过期清理。
        当前已有简版磁盘溢出（DiskSpill），三级增加优先级和清理。
        """
        raise NotImplementedError("L3 degradation not implemented in M-proxy-2")

    async def read(self, count: int = 1, block_ms: int | None = 5000) -> list[tuple[str, Turn]]:
        """读取新消息（XREADGROUP ... >）。返回 [(entry_id, Turn), ...]。

        block_ms=None 表示不带 BLOCK（读空即返回）。**不要传 0**——redis 的 BLOCK 0
        语义是"永远阻塞"，不是"不阻塞"。
        """
        results = await self.redis.xreadgroup(
            self._group, self._consumer, {self._stream: ">"},
            count=count, block=block_ms,
        )
        return self._parse_entries(results)

    async def read_pending(self, count: int = 10, min_idle_ms: int = 5000) -> list[tuple[str, Turn]]:
        """回收 PEL 里滞留的条目（XAUTOCLAIM）。

        包含本消费者上次崩溃遗留的、以及超时未 ack 的条目。
        min_idle_ms: 条目至少 idle 这么久才回收（避免抢活跃消费者的）。
        """
        try:
            result = await self.redis.xautoclaim(
                self._stream, self._group, self._consumer,
                min_idle_time=min_idle_ms, start_id="0", count=count,
            )
        except (aioredis.ResponseError, AttributeError):
            return await self._claim_pending_fallback(count, min_idle_ms)

        # redis-py 8.x xautoclaim 返回 list（旧版 tuple）；两者都接受，取 [1]=claimed entries。
        # 之前只检查 tuple -> list 命中 else 分支返回空 -> 回收的条目不处理不 ack ->
        # 重新 pending -> delivery_count 无限涨（PEL 毒消息根因）。
        if isinstance(result, (tuple, list)) and len(result) >= 2:
            entries = result[1]
        else:
            entries = []

        return self._parse_raw_entries(entries)

    async def _claim_pending_fallback(self, count: int, min_idle_ms: int) -> list[tuple[str, Turn]]:
        """Redis < 6.2 fallback：XPENDING 拿 ID 列表 -> XCLAIM 接管。"""
        try:
            pending = await self.redis.xpending_range(
                self._stream, self._group, min=min_idle_ms, max="+", count=count,
            )
        except Exception:
            return []

        if not pending:
            return []

        entry_ids = [p["message_id"] for p in pending if "message_id" in p]
        if not entry_ids:
            return []

        claimed = await self.redis.xclaim(
            self._stream, self._group, self._consumer,
            min_idle_time=min_idle_ms, message_ids=entry_ids,
        )
        return self._parse_raw_entries(claimed)

    def _parse_entries(self, results: list) -> list[tuple[str, Turn]]:
        """解析 XREADGROUP 返回格式。"""
        turns: list[tuple[str, Turn]] = []
        for _stream_name, entries in results:
            turns.extend(self._parse_raw_entries(entries))
        return turns

    def _parse_raw_entries(self, entries: list) -> list[tuple[str, Turn]]:
        """解析 [(entry_id, fields), ...] 格式。"""
        turns: list[tuple[str, Turn]] = []
        for entry_id, fields in entries:
            raw = fields.get(b"data") or fields.get("data")
            if raw is None:
                logger.warning("pipeline_empty_entry", entry_id=entry_id)
                continue
            data = msgpack.unpackb(raw, raw=False)
            turn = Turn.model_validate(data)
            eid = entry_id.decode() if isinstance(entry_id, bytes) else entry_id
            turns.append((eid, turn))
        return turns

    async def ack(self, entry_id: str) -> None:
        """确认处理完成（XACK）+ 删除已处理条目（XDEL，T1/R3/A3）。

        XACK 只从 PEL 移除，条目仍留在 stream 常驻 Redis 内存 + AOF 无界增长。
        XDEL 真正删除条目，使 XLEN ≈ XPENDING（趋近 0），消灭假溢出连锁：
        累积后 XLEN 恒满 -> 每次 enqueue 假报警 + 阻塞 1s + 绕道磁盘（A3）。
        单消费者组场景安全（无其它消费者需要重读已 ack 历史）。
        Memory Hub 才是总账，Pipeline 里已落库的副本无保留价值。

        XDEL 失败不致命（条目仍在 PEL，下次 XAUTOCLAIM 会重投），只 log。
        """
        await self.redis.xack(self._stream, self._group, entry_id)
        try:
            await self.redis.xdel(self._stream, entry_id)
        except Exception as e:
            logger.warning("pipeline_xdel_failed", entry_id=entry_id, error=str(e))
        logger.debug("pipeline_acked", entry_id=entry_id)

    async def backlog_len(self) -> int:
        """stream 长度（XLEN）--背压判定基准（N1 修复）。

        XDEL 后 XLEN = 未 ack + 未投递 = 真实积压；消费者宕机时新 XADD 仍计入 XLEN
        -> 水位报警能触发（XPENDING 对宕机失明：新条目未投递不进 PEL）。供
        enqueue/_wait_for_drain 用。
        """
        n = await self.redis.xlen(self._stream)
        return int(n)

    async def pending_count(self) -> int:
        """PEL 里已投递未 ack 的条目数（XPENDING）--仅监控接口，不用于背压判定。

        N1 修复后背压改用 backlog_len（XLEN）；本方法保留供监控/测试观察 PEL。
        """
        info = await self.redis.xpending(self._stream, self._group)
        if isinstance(info, dict):
            return int(info.get("pending", 0))
        return 0

    # ── 队列巡检（2026-08-06 事故驱动）─────────────────────────────

    async def group_info(self) -> dict:
        """本消费组那条 XINFO GROUPS 记录（str 键、bytes 值已解码）。

        取两个字段：`last-delivered-id`（投递游标）与 `lag`（还没投递给本组的条目数，
        Redis 7.0+；旧版或游标不可信时 Redis 自己返回 None，此处如实透传不猜）。
        """
        try:
            groups = await self.redis.xinfo_groups(self._stream)
        except aioredis.ResponseError:
            return {}  # stream 不存在
        for g in groups or []:
            name = g.get("name")
            if isinstance(name, bytes):
                name = name.decode()
            if name != self._group:
                continue
            last = g.get("last-delivered-id")
            if isinstance(last, bytes):
                last = last.decode()
            lag = g.get("lag")
            return {
                "last_delivered_id": last or "0-0",
                "lag": int(lag) if isinstance(lag, int) else None,
                "consumers": g.get("consumers"),
            }
        return {}

    async def inspect(self) -> dict:
        """把 XLEN 这一个数拆成四段——积压到底卡在哪一段（2026-08-06 事故驱动）。

        此前 status 只打 `backlog`（XLEN），而 XLEN 是三种完全不同处境的和：

          undelivered  还没投递给消费组   -> worker 一读就走，正常排队
          pending      已投递、还没 ack   -> worker 崩过/写 Hub 失败，XAUTOCLAIM 会重投
          orphan       既不待投递也不在 PEL -> **谁都不会再碰它**，XLEN 里永远躺着

        第三种是这次的形状：ack 加 XDEL 之前那版代码 XACK 完就撒手，条目留在 stream 里，
        XLEN 恒高 -> status 永远显示"排队中"、背压水位凭空少一截余量。它不是慢，是排不动，
        所以必须单独有个数、单独有个出口（`bladex queue flush`）。

        全是 O(1) 命令（XLEN/XPENDING/XINFO），可以放进 /admin/status 每次轮询。
        """
        total = await self.backlog_len()
        pending = await self.pending_count()
        info = await self.group_info()
        lag = info.get("lag")
        undelivered = lag if isinstance(lag, int) else None
        # lag 不可用（Redis < 7 或游标不可信）就报 None，不拿 total 反推——
        # 反推出来的 orphan 会把"正常排队"误报成"排不动"，比没有数更坏。
        orphan = None if undelivered is None else max(0, total - pending - undelivered)
        return {
            "total": total,
            "pending": pending,
            "undelivered": undelivered,
            "orphan": orphan,
            "last_delivered_id": info.get("last_delivered_id", "0-0"),
        }

    async def pending_ids(self, limit: int = 10000) -> set[str]:
        """PEL 里的 entry_id 集合（供孤儿判定排除已投递未 ack 的条目）。"""
        ids: set[str] = set()
        try:
            entries = await self.redis.xpending_range(
                self._stream, self._group, min="-", max="+", count=limit,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("pipeline_pending_range_failed", error=str(e))
            return ids
        for item in entries or []:
            mid = item.get("message_id")
            if isinstance(mid, bytes):
                mid = mid.decode()
            if mid:
                ids.add(mid)
        return ids

    async def scan_entries(self, start: str = "-", count: int = 50) -> list[tuple[str, Turn]]:
        """按 entry_id 升序扫一批 stream 条目（XRANGE），返回 [(entry_id, Turn), ...]。

        分批是刻意的：条目带完整 Turn payload（重车道阈值 256KB），一次性拉全量
        会把 proxy 的内存顶起来。调用方拿最后一个 id 递增后继续。

        单条 msgpack/schema 解不开不让整批失败——只 log 并跳过，让扫描能走到底。
        """
        entries = await self.redis.xrange(self._stream, min=start, max="+", count=count)
        out: list[tuple[str, Turn]] = []
        for entry_id, fields in entries or []:
            eid = entry_id.decode() if isinstance(entry_id, bytes) else entry_id
            raw = fields.get(b"data") or fields.get("data")
            if raw is None:
                logger.warning("pipeline_scan_empty_entry", entry_id=eid)
                continue
            try:
                turn = Turn.model_validate(msgpack.unpackb(raw, raw=False))
            except Exception as e:  # noqa: BLE001
                logger.warning("pipeline_scan_undecodable", entry_id=eid, error=str(e))
                continue
            out.append((eid, turn))
        return out

    async def delete_entry(self, entry_id: str) -> None:
        """XACK + XDEL 一条已确认落库的条目（孤儿清理用）。

        先 XACK 是为幂等：条目可能同时还挂在 PEL 上（清理与 worker 并发时），
        只 XDEL 会留下一条指向不存在条目的 PEL 记录，之后每次 XAUTOCLAIM 都空转。
        """
        await self.redis.xack(self._stream, self._group, entry_id)
        await self.redis.xdel(self._stream, entry_id)
