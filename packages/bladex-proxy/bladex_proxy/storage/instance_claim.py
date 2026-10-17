"""MQ-A38（批 Z1，2026-09-23）：同一条 Redis 流只许一个 BladeX 实例 —— 启动即检测、响亮拒绝。

## 为什么要有它

同一台机器起第二个实例（演示 / 冷安装验证 / 排障），只换端口不换 Redis 命名空间时，两个
proxy 往**同一条 stream** 里写、在**同一个消费组**里读 ⇒ live 的轮次被第二个实例的
`PipelineWorker` 静默消费、写进**另一个** `BLADEX_HOME` 的 Hub。v0.1.0 只在
`.env.example` 里加了三个隔离旋钮（`BLADEX_REDIS_STREAM/GROUP/CONSUMER`），**没有任何检测**
——撞上了不报错，丢的轮次事后也无从追查。

## 判据：按 stream 登记，不按消费组

登记键 = `bladex:instance:<stream>`（落在 stream 所在的那个 db 里，换 db 天然隔离）。
为什么键上只有 stream：两个实例**只要共用 stream** 就互相污染——同组是偷（消息被分走），
异组是串（各自的组都收到对方的轮次，写进各自的 Hub）。没有一种"共用 stream"的配置是对的。
报错里 stream 与两边的消费组都点名。

值 = `{"home", "pid", "host", "port", "group", "consumer", "url"}`，带 TTL（`CLAIM_TTL_S`），
proxy 每 `CLAIM_TTL_S / 3` 秒续一次。

## 三种"已有登记"怎么判

| 情形 | 判定 | 理由 |
|---|---|---|
| 登记的 `home` 与我相同 | 接管（覆盖） | 同一实例重启（`bladex stop && bladex start`，或崩溃后再起）|
| `home` 不同、登记方与我同机且 pid 已死 | 接管 + WARNING `instance_claim_stale_takeover` | 上一个进程崩了没来得及删登记 —— **stale 不许挡住启动** |
| `home` 不同、pid 活着（或异机，无法判活） | **拒绝**：抛 `InstanceConflictError` | 这就是偷轮次的形态；异机的 stale 最多挡 `CLAIM_TTL_S` 秒（TTL 到期自动消失）|

还有第四种：**没有登记，但消费组里有活着的消费者** ⇒ 对方是 0.3.0 之前的版本（不会登记）——
升级窗口里最常见的形态恰恰是"live 还没升级、第二个实例是新装的"。**拒绝**，报错点名消费者名。

难点是别把"同一实例刚 stop 完就 start"误判成它：旧进程几秒前还在读，`XINFO CONSUMERS`
的 idle 很小。判活不看一次 idle 的大小，看**它会不会被重置**：`PipelineWorker` 每
`block_ms=2000` 重新发一次 XREADGROUP，活着的消费者 idle 恒 < ~2.5 s（Redis 8.8 实测
0.69–0.76 s 来回跳）；停掉的只会单调增长。故 idle ≤ `LIVENESS_WINDOW_S` 时等一个窗口再采一次，
仍 < 窗口 ⇒ 活着。代价：只有"没登记 + 组里有刚动过的消费者"时启动多等 `LIVENESS_WINDOW_S` 秒。
（2026-09-23 本批实测时的事故驱动：一个没加载 `.env` 的裸 uvicorn 连到了 live Redis，
当时这里还只是告警，于是它照常起了 PipelineWorker —— 那 2 分 14 秒里 live 恰好没有新轮次，
0 条被偷；有流量就不是 0。）

## 谁调

- `server/app_factory.lifespan`：连上 Redis 后 `acquire()`；冲突 ⇒ 异常原样上抛、lifespan
  失败、进程退出，`bladex start` 的 `_report_startup_state` 把日志尾部原样打到终端
  （不吞输出，刚性原则 12 第三条）。
- `server/orchestration._try_recover_redis`（懒恢复）：冲突 ⇒ 不接这条流（轮次落盘溢出），
  打 ERROR —— 运行中不能"拒绝启动"，但同样不许偷。
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
from typing import Any

import structlog

logger = structlog.get_logger()

#: 登记 TTL（秒）。心跳每 TTL/3 续一次；异机崩溃留下的 stale 登记最多挡这么久。
CLAIM_TTL_S = 30
#: 未登记消费者的判活窗口（秒）。须 > PipelineWorker 的阻塞读周期（2 s）+ 一批处理时长。
LIVENESS_WINDOW_S = 4.0
CLAIM_KEY_PREFIX = "bladex:instance:"


class InstanceConflictError(RuntimeError):
    """另一个 `BLADEX_HOME` 的实例正在用同一条 Redis stream。"""


def claim_key(stream: str) -> str:
    return f"{CLAIM_KEY_PREFIX}{stream}"


def instance_home() -> str:
    """本实例的身份 = 部署根的真实路径（`BLADEX_HOME` → 含 config/.env 的祖先 → cwd）。"""
    from bladex_proxy.deployment import deployment_root
    return os.path.realpath(deployment_root())


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _decode(raw: Any) -> dict | None:
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def conflict_message(stream: str, mine: dict, theirs: dict) -> str:
    """拒绝时的报错全文：冲突的流 / 组 / 对方的 BLADEX_HOME / 三旋钮怎么隔离。"""
    return (
        f"refusing to start: Redis stream {stream!r} on {mine.get('url')} is already used by "
        f"another BladeX instance -- BLADEX_HOME={theirs.get('home')} "
        f"(pid {theirs.get('pid')} on {theirs.get('host')}, port {theirs.get('port')}, "
        f"consumer group {theirs.get('group')!r}). This instance is "
        f"BLADEX_HOME={mine.get('home')} (consumer group {mine.get('group')!r}). "
        "Two instances on one stream silently consume each other's turns into the wrong "
        "Memory Hub (MQ-A38). To run a second instance, give it its own Redis namespace in "
        "ITS config/.env: BLADEX_REDIS_STREAM=\"bladex:turns:<name>\", "
        "BLADEX_REDIS_GROUP=\"bladex-workers-<name>\", BLADEX_REDIS_CONSUMER=\"worker-<name>\", "
        "and point BLADEX_REDIS_URL at another db (e.g. redis://127.0.0.1:6379/9). "
        f"If that instance is gone for good, its claim expires within {CLAIM_TTL_S}s."
    )


def unregistered_message(stream: str, mine: dict, consumers: list[str]) -> str:
    """没有登记、但组里有活着的消费者时的报错全文（对方是 0.3.0 之前的版本）。"""
    return (
        f"refusing to start: Redis stream {stream!r} on {mine.get('url')} is being consumed right "
        f"now by consumer(s) {consumers} in group {mine.get('group')!r}, and that process did not "
        "register its BLADEX_HOME -- most likely a BladeX older than 0.3.0 (your live instance?). "
        f"This instance is BLADEX_HOME={mine.get('home')}. Starting would silently split the "
        "turns between the two Memory Hubs (MQ-A38). To run a second instance, give it its own "
        "Redis namespace in ITS config/.env: BLADEX_REDIS_STREAM=\"bladex:turns:<name>\", "
        "BLADEX_REDIS_GROUP=\"bladex-workers-<name>\", BLADEX_REDIS_CONSUMER=\"worker-<name>\", "
        "and point BLADEX_REDIS_URL at another db (e.g. redis://127.0.0.1:6379/9). "
        "If this IS the same deployment being upgraded, stop the old process first."
    )


class InstanceClaim:
    """一条 stream 的实例登记（acquire / heartbeat / release）。`redis` 为 redis.asyncio 客户端。"""

    def __init__(self, redis: Any, *, stream: str, group: str, consumer: str,
                 url: str, port: int, home: str | None = None,
                 sleep: Any = asyncio.sleep) -> None:
        self._redis = redis
        self._sleep = sleep
        self._stream = stream
        self._key = claim_key(stream)
        self._task: asyncio.Task | None = None
        self.record = {
            "home": home or instance_home(), "pid": os.getpid(),
            "host": socket.gethostname(), "port": port,
            "group": group, "consumer": consumer, "url": url,
        }

    @property
    def key(self) -> str:
        return self._key

    def _payload(self) -> str:
        return json.dumps(self.record, ensure_ascii=False)

    async def acquire(self) -> None:
        """登记本实例；另一个活着的实例占着这条流 ⇒ 抛 `InstanceConflictError`。

        Redis 操作本身出错（不是冲突）⇒ 只告警、不登记、放行：检测设施故障不许拖垮启动。
        """
        try:
            theirs = _decode(await self._redis.get(self._key))
            if theirs is None:
                live = await self._live_unregistered_consumers()
                if live:
                    msg = unregistered_message(self._stream, self.record, live)
                    logger.error("instance_claim_unregistered_consumer", key=self._key,
                                 stream=self._stream, group=self.record["group"],
                                 consumers=live, my_home=self.record["home"])
                    raise InstanceConflictError(msg)
                if await self._redis.set(self._key, self._payload(), nx=True, ex=CLAIM_TTL_S):
                    logger.info("instance_claim_acquired", key=self._key,
                                home=self.record["home"])
                    return
                theirs = _decode(await self._redis.get(self._key))  # 刚被别人抢先登记
        except InstanceConflictError:
            raise
        except Exception as e:  # noqa: BLE001 —— 检测故障：告警放行
            logger.warning("instance_claim_unavailable", key=self._key, error=str(e),
                           hint="could not check for another instance on this stream")
            return
        if theirs is None or theirs.get("home") == self.record["home"]:
            reason = "same_home" if theirs is not None else "unreadable_claim"
        elif (theirs.get("host") == self.record["host"]
              and isinstance(theirs.get("pid"), int) and not _pid_alive(theirs["pid"])):
            reason = "stale"
            logger.warning("instance_claim_stale_takeover", key=self._key,
                           previous_home=theirs.get("home"), previous_pid=theirs.get("pid"),
                           hint="the previous owner of this stream is no longer running")
        else:
            msg = conflict_message(self._stream, self.record, theirs)
            logger.error("instance_claim_conflict", key=self._key, stream=self._stream,
                         their_home=theirs.get("home"), their_pid=theirs.get("pid"),
                         their_group=theirs.get("group"), my_home=self.record["home"],
                         my_group=self.record["group"])
            raise InstanceConflictError(msg)
        try:
            await self._redis.set(self._key, self._payload(), ex=CLAIM_TTL_S)
        except Exception as e:  # noqa: BLE001
            logger.warning("instance_claim_unavailable", key=self._key, error=str(e))
            return
        logger.info("instance_claim_acquired", key=self._key, home=self.record["home"],
                    takeover=reason)

    async def _consumer_idle(self) -> dict[str, int]:
        """本组各消费者的 idle（ms）；组 / 流不存在 ⇒ 空。"""
        try:
            consumers = await self._redis.xinfo_consumers(self._stream, self.record["group"])
        except Exception:  # noqa: BLE001 —— NOGROUP / 流还不存在 = 没有消费者
            return {}
        out: dict[str, int] = {}
        for c in consumers or []:
            c = {(k.decode() if isinstance(k, bytes) else k): v for k, v in dict(c).items()}
            name = c.get("name")
            name = name.decode() if isinstance(name, bytes) else str(name)
            out[name] = int(c.get("idle") or 0)
        return out

    async def _live_unregistered_consumers(self) -> list[str]:
        """没有登记时，组里**此刻还活着**的消费者名（两次采样看 idle 会不会被重置）。"""
        window_ms = int(LIVENESS_WINDOW_S * 1000)
        first = {n: i for n, i in (await self._consumer_idle()).items() if i <= window_ms}
        if not first:
            return []
        logger.info("instance_claim_liveness_probe", key=self._key, consumers=sorted(first),
                    wait_s=LIVENESS_WINDOW_S,
                    hint="unregistered consumer seen recently; checking whether it is still alive")
        await self._sleep(LIVENESS_WINDOW_S)
        second = await self._consumer_idle()
        return sorted(n for n in first if n in second and second[n] < window_ms)

    async def heartbeat_once(self) -> bool:
        """续一次 TTL；登记已被别的实例接管 ⇒ ERROR 并返回 False（不抢回来）。"""
        try:
            theirs = _decode(await self._redis.get(self._key))
            if theirs is not None and theirs.get("home") != self.record["home"]:
                logger.error("instance_claim_lost", key=self._key,
                             their_home=theirs.get("home"), their_pid=theirs.get("pid"),
                             hint="another instance took over this stream; turns may be split")
                return False
            await self._redis.set(self._key, self._payload(), ex=CLAIM_TTL_S)
            return True
        except Exception as e:  # noqa: BLE001
            logger.debug("instance_claim_heartbeat_failed", key=self._key, error=str(e))
            return True

    def start_heartbeat(self) -> None:
        async def _loop() -> None:
            while True:
                await asyncio.sleep(CLAIM_TTL_S / 3)
                await self.heartbeat_once()

        self._task = asyncio.create_task(_loop())

    async def release(self) -> None:
        """正常退出时删登记（仅当还是自己的）。"""
        if self._task is not None:
            self._task.cancel()
            self._task = None
        try:
            theirs = _decode(await self._redis.get(self._key))
            if theirs is not None and theirs.get("home") == self.record["home"] \
                    and theirs.get("pid") == self.record["pid"]:
                await self._redis.delete(self._key)
                logger.info("instance_claim_released", key=self._key)
        except Exception as e:  # noqa: BLE001
            logger.debug("instance_claim_release_failed", key=self._key, error=str(e))
