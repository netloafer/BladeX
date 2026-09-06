"""Pipeline→Memory Hub 后台 worker — 消费 Redis Streams、写 RocksDB、ack（T5，ADR-0009 §5）。

落库成功才 ack（不丢不重复）。
周期性回收 PEL 滞留条目（XAUTOCLAIM），确保崩溃/失败后重投。

T1（M-proxy-1.5）：Memory Hub key 改用 Redis stream entry_id 作唯一后缀，
重试/重新生成/工具循环不再覆盖。
"""

from __future__ import annotations

import asyncio
import re

import structlog

from bladex_proxy.models import Turn
from bladex_proxy.storage.pipeline_redis import PipelineRedis
from bladex_proxy.storage.memory_hub import MemoryHub

logger = structlog.get_logger()


# 从 tool result 内容中提取工具名的正则（Hermes 格式：<untrusted_tool_result source="...">）
_TOOL_SOURCE_RE = re.compile(r'source="([^"]+)"')


def _extract_tool_name(msg: dict, turn: Turn, request_call_names: dict[str, str] | None = None) -> str:
    """从 role=tool 消息提取工具名（四层 fallback）。

    1. OpenAI 标准 name 字段
    2. content 内的 source="..." 属性（Hermes untrusted_tool_result 格式）
    3. 当前轮 tool_events 里按 tool_call_id 查 call 的 tool_name
    4. request_messages 的 assistant tool_calls 里按 tool_call_id 查（跨轮补全）
    """
    # 1. OpenAI 标准 name 字段
    name = msg.get("name", "")
    if name:
        return name

    # 2. content 内的 source="..." 属性
    content = msg.get("content", "")
    if isinstance(content, str):
        m = _TOOL_SOURCE_RE.search(content)
        if m:
            return m.group(1)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                text = part.get("text", "")
                if isinstance(text, str):
                    m = _TOOL_SOURCE_RE.search(text)
                    if m:
                        return m.group(1)

    tc_id = msg.get("tool_call_id", "")

    # 3. 当前轮 tool_events 按 tool_call_id 查
    if tc_id:
        for te in turn.tool_events:
            if te.tool_call_id == tc_id and te.tool_name:
                return te.tool_name

    # 4. request_messages 的 assistant tool_calls 按 tool_call_id 查（跨轮补全）
    if tc_id and request_call_names and request_call_names.get(tc_id):
        return request_call_names[tc_id]

    return ""


def enrich_tool_results(turn: Turn) -> None:
    """从 request_messages 的 role=tool 消息回填 tool_events.result（ADR-0011 P1）。

    工具返回在 OpenAI 格式里是请求中的 role=tool 消息（带 tool_call_id），
    不是流式响应的一部分。本函数在入库前解析这些消息，按 tool_call_id
    匹配回对应的 tool_event（direction=call），填充 result 字段。

    未匹配到 call 的 result（属于上一轮的工具调用）也记录为
    direction=result 的 ToolEvent，确保工具返回不丢失。

    去重（review 20260707）：只处理最后一条 assistant 消息之后的 role=tool
    消息。之前的 role=tool 消息已在上一轮入库时记录过，重复记录会导致
    tool_events 跨轮 O(n) 堆积（实测 11x 冗余）。最后一条 assistant 之后
    的 tool results 是本轮新增的（模型刚收到、还未被记录的返回）。
    """
    if not turn.request_messages:
        return

    # T4(ADR-0018 §4.4 C7): Anthropic 格式的工具返回是 user 消息里的
    # type=tool_result block（无 role=tool 消息）。早退条件必须同时识别它，
    # 否则纯 Anthropic turn 在此 return，_enrich_anthropic_tool_results 永不执行。
    has_tool_results = any(
        m.get("role") == "tool" for m in turn.request_messages
    ) or any(
        m.get("role") == "user"
        and isinstance(m.get("content"), list)
        and any(
            isinstance(b, dict) and b.get("type") == "tool_result"
            for b in m["content"]
        )
        for m in turn.request_messages
    )
    if not has_tool_results:
        return

    # 去重：找到最后一条 assistant 消息的位置，只处理其后的 role=tool 消息
    last_assistant_idx = -1
    for i in range(len(turn.request_messages) - 1, -1, -1):
        if turn.request_messages[i].get("role") == "assistant":
            last_assistant_idx = i
            break

    # 无 assistant 消息时处理全部 role=tool（保守：可能含旧数据，但不丢新数据）
    scan_start = last_assistant_idx + 1 if last_assistant_idx >= 0 else 0

    # 构建 tool_call_id -> tool_event 映射（当前轮捕获的 calls）
    call_map: dict[str, int] = {}
    for i, te in enumerate(turn.tool_events):
        if te.direction == "call" and te.tool_call_id:
            call_map[te.tool_call_id] = i

    # 跨消息 call name 映射：从 request_messages 的 assistant tool_calls 提取
    # tool_call_id -> tool_name。用于补全上一轮工具返回（role=tool）的 name--
    # 这些 result 属于上一轮的 call，不在当前轮 call_map 里，但它们对应的
    # assistant tool_call 在本轮 request_messages（历史消息）里能找到 name。
    # 通用修复，惠及所有走工具循环的 agent（Codex/Claude Code/Hermes）。
    request_call_names: dict[str, str] = {}
    for m in turn.request_messages:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            tc_id = tc.get("id", "")
            func = tc.get("function") or {}
            name = func.get("name", "")
            if tc_id and name:
                request_call_names[tc_id] = name

    # 已记录的 result tool_call_id 集合（防同轮重复）
    existing_result_ids = {
        te.tool_call_id for te in turn.tool_events
        if te.direction == "result" and te.tool_call_id
    }

    matched = 0
    unmatched = 0
    skipped_old = 0
    for msg in turn.request_messages[scan_start:]:
        if msg.get("role") != "tool":
            # T4(ADR-0018 §4.4 C7): Anthropic tool_result block 在 user 消息里
            if msg.get("role") == "user":
                m, u, sk = _enrich_anthropic_tool_results(msg, turn, call_map, existing_result_ids)
                matched += m
                unmatched += u
                skipped_old += sk
            continue
        tc_id = msg.get("tool_call_id", "")
        content = msg.get("content", "")

        # content 可能是 str 或 list（多模态）
        if isinstance(content, list):
            text_parts = [
                p.get("text", "") for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            ]
            content_text = " ".join(text_parts)
        else:
            content_text = str(content)

        if tc_id and tc_id in call_map:
            idx = call_map[tc_id]
            turn.tool_events[idx].result = content_text
            matched += 1
        elif tc_id and tc_id in existing_result_ids:
            # 同轮已记录过此 result → 跳过
            skipped_old += 1
        else:
            # 属于上一轮的工具返回——记录为独立 result 事件
            name = _extract_tool_name(msg, turn, request_call_names)
            from bladex_proxy.models import ToolEvent
            turn.tool_events.append(ToolEvent(
                tool_name=name, result=content_text,
                direction="result", tool_call_id=tc_id,
            ))
            existing_result_ids.add(tc_id)
            unmatched += 1

    if matched or unmatched:
        logger.info(
            "tool_results_enriched",
            matched=matched, unmatched=unmatched,
            skipped_old=skipped_old,
            total_events=len(turn.tool_events),
        )

# 每 N 轮主循环回收一次 PEL
_RECLAIM_INTERVAL = 10
# PEL 条目至少 idle 这么久才回收（ms）
_RECLAIM_MIN_IDLE_MS = 5000
# Memory Hub 写失败后的有限重试次数
_LEDGER_MAX_RETRIES = 3
# Memory Hub 写失败后的退避（秒）
_LEDGER_RETRY_BACKOFF = 1.0
# force_drain 每批条数（条目带完整 Turn payload，别一次拉全量）
_FLUSH_BATCH = 100
# force_drain 每个阶段的批次上限（防毒条目把循环钉死）
_FLUSH_MAX_BATCHES = 200


def _id_gt(a: str, b: str) -> bool:
    """Redis stream entry_id（`ms-seq`）比大小。字符串比较会把 10-0 判成小于 9-0。"""
    def _parts(v: str) -> tuple[int, int]:
        ms, _, seq = v.partition("-")
        try:
            return (int(ms), int(seq or 0))
        except ValueError:
            return (0, 0)
    return _parts(a) > _parts(b)


class PipelineWorker:
    """后台 worker：Pipeline Redis → Memory Hub RocksDB。"""

    def __init__(self, pipeline: PipelineRedis, ledger: MemoryHub) -> None:
        self._pipeline = pipeline
        self._hub = ledger
        self._running = False
        self._task: asyncio.Task | None = None
        self._loop_count = 0
        self._flush_lock = asyncio.Lock()

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._hub.open()
        self._task = asyncio.create_task(self._run())
        logger.info("pipeline_started")

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("pipeline_stopped")

    async def _run(self) -> None:
        """主消费循环：XREADGROUP → 写 Memory Hub → XACK，周期性 XAUTOCLAIM 回收 PEL。

        Redis 连接断开后自动重连——空闲一段时间 Redis 会关闭连接，
        XREADGROUP 会抛 TimeoutError，重连后恢复正常。
        """
        logger.info("pipeline_worker_running")
        while self._running:
            try:
                self._loop_count += 1

                # 周期性回收 PEL 滞留条目（ISSUE-1 修复）
                if self._loop_count % _RECLAIM_INTERVAL == 0:
                    await self._reclaim_pending()

                # 读新消息
                turns = await self._pipeline.read(count=10, block_ms=2000)
                if not turns:
                    continue

                for entry_id, turn in turns:
                    await self._process_one(entry_id, turn)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("pipeline_worker_error", error=str(e))
                # Redis 连接可能断了，尝试重连
                try:
                    await self._pipeline.close()
                    await self._pipeline.connect()
                    logger.info("pipeline_redis_reconnected")
                except Exception as reconnect_err:
                    logger.warning("pipeline_reconnect_failed", error=str(reconnect_err))
                await asyncio.sleep(2)

    async def _reclaim_pending(self) -> None:
        """回收 PEL 滞留条目（XAUTOCLAIM）。"""
        try:
            turns = await self._pipeline.read_pending(count=10, min_idle_ms=_RECLAIM_MIN_IDLE_MS)
            if turns:
                logger.info("pipeline_reclaimed", count=len(turns))
                for entry_id, turn in turns:
                    await self._process_one(entry_id, turn)
        except Exception as e:
            logger.warning("pipeline_reclaim_error", error=str(e))

    async def _process_one(self, entry_id: str, turn: Turn) -> bool:
        """处理一条：写 Memory Hub（有限重试）→ ack。失败留 PEL 等下一轮回收。

        T1: Memory Hub key 用 entry_id 作唯一后缀，不再用 turn_index。
        Memory Hub 写入是同步操作，跑在线程池里避免阻塞事件循环。

        返回是否落库成功（force_drain 要按条计数；主循环不看返回值）。
        """
        key = turn.identity.storage_key(entry_id)
        last_err = None
        loop = asyncio.get_running_loop()

        # Pipeline（ADR-0011）：入库前从 request_messages 回填 tool_events.result
        enrich_tool_results(turn)

        for attempt in range(_LEDGER_MAX_RETRIES):
            try:
                # Memory Hub 写入（同步 RocksDB）放线程池，不阻塞事件循环
                await loop.run_in_executor(None, self._hub.put, key, turn)
                await self._pipeline.ack(entry_id)
                logger.info("pipeline_processed", key=key, status=turn.status.value,
                            reclaimed=attempt > 0)
                # Flash 门铃（2026-08-29 拍板：会话触发 push 取代定时轮询）——
                # Hub 写完才敲，fire-and-forget，故障不外溢到落库路径。
                from bladex_proxy.flash_wake import notify_wake
                await notify_wake(getattr(self._pipeline, "_redis", None), "turn")
                return True
            except Exception as e:
                last_err = e
                logger.warning("pipeline_ledger_write_retry", entry_id=entry_id,
                               attempt=attempt + 1, max=_LEDGER_MAX_RETRIES, error=str(e))
                if attempt < _LEDGER_MAX_RETRIES - 1:
                    await asyncio.sleep(_LEDGER_RETRY_BACKOFF)

        # 重试耗尽：不 ack，留 PEL 等下一轮 XAUTOCLAIM 回收
        logger.error("pipeline_ledger_write_failed", entry_id=entry_id, error=str(last_err))
        return False

    # ── 强制排空（2026-08-06 事故驱动）────────────────────────────

    async def force_drain(self, *, apply: bool = False,
                          max_batches: int = _FLUSH_MAX_BATCHES,
                          sample: int = 0) -> dict:
        """把 Pipeline 队列推到真正为空，并给"排不动"的那部分一个出口。

        为什么需要它：后台主循环只处理两种条目——待投递的（XREADGROUP `>`）和
        PEL 里滞留的（XAUTOCLAIM）。**第三种它永远看不见**：既已被 XACK、又没被 XDEL
        的残留（ack 加 XDEL 之前那版代码的产物）。它们不在 PEL、id 又落在投递游标之后，
        两条路都够不着，于是 XLEN 恒高——status 把它读成"排队中"，背压还凭空少一截余量。

        三步，顺序不能换（每一步都会改变下一步的判据）：
          1. 投递未读：XREADGROUP `>` 拉干净
          2. 认领滞留：XAUTOCLAIM min_idle=0 收 PEL（主循环要求 idle≥5s，这里手工触发不等）
          3. 清扫孤儿：1、2 之后 stream 里还剩的就是孤儿；**逐条拿 Turn 算出 Hub key
             去核对**——Hub 里有才 XDEL，没有的先补写再删。绝不因为"它看起来是残留"
             就直接删：删错了就是永久丢一轮对话，而 Hub 是唯一真相源。

        apply=False（默认）只巡检不改动：同样跑完整的分类与 Hub 核对，报告将要做什么。
        sample>0 时额外取样若干孤儿的画像（时间、agent、同会话在 Hub 里已有多少轮），
        用来回答"这些到底是什么、为什么没落账"——先看清楚再动手。
        """
        async with self._flush_lock:
            return await self._force_drain_locked(
                apply=apply, max_batches=max_batches, sample=sample)

    async def _force_drain_locked(self, *, apply: bool, max_batches: int,
                                  sample: int = 0) -> dict:
        before = await self._pipeline.inspect()
        report: dict = {
            "mode": "applied" if apply else "dry-run",
            "before": before,
            "delivered": 0,
            "reclaimed": 0,
            "failed": 0,
            "orphan_total": 0,
            "orphan_in_hub": 0,
            "orphan_missing": 0,
            "orphan_recovered": 0,
            "orphan_deleted": 0,
            "orphan_agents": {},
            "orphan_first_ts": "",
            "orphan_last_ts": "",
            "samples": [],
            "truncated": False,
        }

        if apply:
            report["delivered"] = await self._drain_undelivered(max_batches, report)
            report["reclaimed"] = await self._drain_pending(max_batches, report)

        await self._sweep_orphans(apply=apply, max_batches=max_batches,
                                  report=report, sample=sample)

        report["after"] = await self._pipeline.inspect()
        logger.info("pipeline_force_drain", **{
            k: v for k, v in report.items() if k not in ("before", "after")})
        return report

    async def _drain_undelivered(self, max_batches: int, report: dict) -> int:
        """步骤 1：XREADGROUP `>` 把未投递的拉干净。

        block_ms=None 是关键：redis-py 的 `block=0` 是"永远阻塞"，不是"不阻塞"。
        """
        delivered = 0
        for _ in range(max_batches):
            turns = await self._pipeline.read(count=_FLUSH_BATCH, block_ms=None)
            if not turns:
                return delivered
            for entry_id, turn in turns:
                if await self._process_one(entry_id, turn):
                    delivered += 1
                else:
                    report["failed"] += 1
        report["truncated"] = True
        return delivered

    async def _drain_pending(self, max_batches: int, report: dict) -> int:
        """步骤 2：XAUTOCLAIM 收 PEL 滞留条目。

        XAUTOCLAIM 每次都从 "0" 起扫，写 Hub 一直失败的条目会被反复捞回来。
        所以按 entry_id 记账：一批里没有任何**没见过的** id，就停——否则毒条目能把
        循环钉死在这儿，而这个命令本来是给"卡住了"的人用的。
        """
        reclaimed = 0
        seen: set[str] = set()
        for _ in range(max_batches):
            turns = await self._pipeline.read_pending(count=_FLUSH_BATCH, min_idle_ms=0)
            if not turns:
                return reclaimed
            fresh = [(eid, t) for eid, t in turns if eid not in seen]
            if not fresh:
                logger.warning("pipeline_force_drain_stuck_pending",
                               stuck=len(turns), sample=[eid for eid, _ in turns[:3]])
                report["truncated"] = True
                return reclaimed
            for entry_id, turn in fresh:
                seen.add(entry_id)
                if await self._process_one(entry_id, turn):
                    reclaimed += 1
                else:
                    report["failed"] += 1
        report["truncated"] = True
        return reclaimed

    async def _sweep_orphans(self, *, apply: bool, max_batches: int, report: dict,
                             sample: int = 0) -> None:
        """步骤 3：扫 stream，把既不在 PEL、又落在投递游标之前的条目按 Hub 核对处置。

        顺带做画像（时间跨度 / agent 分布 / 取样几条看同会话在 Hub 里有没有邻居）。
        "0 条在 Hub 里"有两种截然不同的读法——真的从没落过账，还是 key 算法漂了导致
        全查不中。画像就是用来把这两种分开的：同会话邻居多而独独缺这几条 = 真没落账。
        """
        info = await self._pipeline.group_info()
        last_delivered = info.get("last_delivered_id") or "0-0"
        pel_ids = await self._pipeline.pending_ids()
        loop = asyncio.get_running_loop()

        cursor = "-"
        for _ in range(max_batches):
            entries = await self._pipeline.scan_entries(start=cursor, count=_FLUSH_BATCH)
            if not entries:
                return
            last_id = entries[-1][0]
            for entry_id, turn in entries:
                if entry_id in pel_ids or _id_gt(entry_id, last_delivered):
                    continue  # 还在正常两条路上，不是孤儿
                report["orphan_total"] += 1
                self._profile_orphan(entry_id, turn, report)
                key = turn.identity.storage_key(entry_id)
                existing = await loop.run_in_executor(None, self._hub.get, key)
                if len(report["samples"]) < sample:
                    report["samples"].append(await self._sample_orphan(
                        entry_id, turn, key, existing is not None, loop))
                if existing is not None:
                    report["orphan_in_hub"] += 1
                    if apply:
                        await self._pipeline.delete_entry(entry_id)
                        report["orphan_deleted"] += 1
                    continue
                # Hub 里没有 -> 这一轮从没落过账，补写再删，别当垃圾扔掉
                report["orphan_missing"] += 1
                if apply and await self._process_one(entry_id, turn):
                    report["orphan_recovered"] += 1
                    report["orphan_deleted"] += 1
            cursor = f"({last_id}"  # XRANGE 独占起点，避免同一条重复计数
        report["truncated"] = True

    @staticmethod
    def _profile_orphan(entry_id: str, turn: Turn, report: dict) -> None:
        """孤儿的时间跨度与 agent 分布——一眼看出是"某段时间集体没落账"还是长期零星。"""
        ts = turn.ts.isoformat() if turn.ts else entry_id
        if not report["orphan_first_ts"] or ts < report["orphan_first_ts"]:
            report["orphan_first_ts"] = ts
        if ts > report["orphan_last_ts"]:
            report["orphan_last_ts"] = ts
        agent = turn.identity.agent_id or "unknown"
        report["orphan_agents"][agent] = report["orphan_agents"].get(agent, 0) + 1

    async def _sample_orphan(self, entry_id: str, turn: Turn, key: str,
                             in_hub: bool, loop: asyncio.AbstractEventLoop) -> dict:
        """一条孤儿的画像。`session_turns_in_hub` 是那个关键的对照量。"""
        prefix = turn.identity.session_prefix()

        def _count() -> int:
            try:
                return sum(1 for _ in self._hub.scan_meta(prefix))
            except Exception:  # noqa: BLE001 —— 画像失败不该拖垮巡检
                return -1

        return {
            "entry_id": entry_id,
            "ts": turn.ts.isoformat() if turn.ts else "",
            "agent": turn.identity.agent_id,
            "session": turn.identity.session_id,
            "status": turn.status.value,
            "key": key,
            "in_hub": in_hub,
            "session_turns_in_hub": await loop.run_in_executor(None, _count),
        }


def _enrich_anthropic_tool_results(
    msg: dict,
    turn: Turn,
    call_map: dict[str, int],
    existing_result_ids: set,
) -> tuple[int, int, int]:
    """T4(C7): Anthropic 格式 tool_result block 回填 tool_events.result。

    Anthropic 工具返回是 user 消息里 type=tool_result 的 content block，
    按 tool_use_id 匹配回 call（与 OpenAI 的 role=tool + tool_call_id 对称）。
    返回 (matched, unmatched, skipped_old)。
    """
    content = msg.get("content", "")
    if not isinstance(content, list):
        return (0, 0, 0)

    matched = 0
    unmatched = 0
    skipped_old = 0
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "tool_result":
            continue
        tc_id = block.get("tool_use_id", "")
        rc = block.get("content", "")
        if isinstance(rc, list):
            text_parts = [
                p.get("text", "") for p in rc
                if isinstance(p, dict) and p.get("type") == "text"
            ]
            content_text = " ".join(text_parts)
        else:
            content_text = str(rc)

        if tc_id and tc_id in call_map:
            idx = call_map[tc_id]
            turn.tool_events[idx].result = content_text
            matched += 1
        elif tc_id and tc_id in existing_result_ids:
            skipped_old += 1
        else:
            from bladex_proxy.models import ToolEvent
            turn.tool_events.append(ToolEvent(
                tool_name="", result=content_text,
                direction="result", tool_call_id=tc_id,
            ))
            existing_result_ids.add(tc_id)
            unmatched += 1
    return (matched, unmatched, skipped_old)
