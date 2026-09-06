"""Flash 唤醒通道（Jason 2026-08-29 拍板：会话触发 push，不做定时轮询维护）。

生产者 = proxy 的 pipeline worker（Hub 落库成功后 XADD 一条轻量通知）；
消费者 = flash 维护进程（阻塞 XREAD——没有会话就挂着零消耗）。

Redis 是栈内现成组件（P0 暂存层同一实例），这里只借它的 stream 当门铃：
- 通知不携带业务数据（daemon 醒来后自己 catch_up Hub——数据永远以 Hub 为准，
  门铃丢了最多晚醒一轮，不产生正确性问题）；
- 生产侧 fire-and-forget（XADD 失败静默吞——不能让门铃故障影响落库路径）；
- 消费侧掉线退化为低频轮询 + 可 grep 告警，恢复靠每轮懒重连（ADR-0017 形态）。
"""

from __future__ import annotations

import time
from typing import Any

try:
    import structlog

    logger = structlog.get_logger(__name__)
except ImportError:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)  # type: ignore[assignment]

#: 门铃 stream（常量不做配置：协议常量，两端必须一致）。
WAKE_STREAM = "bladex:flash:wake"
#: 门铃只需要"最近响过"，历史无意义——XADD 带 maxlen 让 Redis 自己修剪。
WAKE_MAXLEN = 16


async def notify_wake(redis_client: Any, kind: str = "turn") -> None:
    """生产侧：Hub 写完后敲一下门铃（异步、尽力而为）。

    任何异常吞掉——门铃永远不许反向影响落库/事件路径。
    """
    if redis_client is None:
        return
    try:
        await redis_client.xadd(WAKE_STREAM, {"kind": kind},
                                maxlen=WAKE_MAXLEN, approximate=True)
    except Exception:  # noqa: BLE001,S110 —— 门铃故障不外溢
        pass


class WakeConsumer:
    """消费侧：阻塞等门铃；Redis 不在时退化为轮询节奏（返回值仍是 True，
    让调用方照常做一轮维护——降级模式 = 回到旧的定时形态，但有告警可见）。

    `wait(timeout_s)` 语义：
    - True  = 该干活（有新会话数据，或处于降级轮询）
    - False = 干等超时且确定没有新数据（真空闲，调用方只处理低频轨道）
    """

    #: 降级告警冷却（秒）——掉线期间每轮都喊会淹日志。
    _WARN_COOLDOWN_S = 300.0

    def __init__(self, redis_url: str) -> None:
        self._url = redis_url
        self._client: Any = None
        self._last_id = "$"
        self._last_warn = 0.0

    def _connect(self) -> Any:
        if self._client is not None:
            return self._client
        import redis as _redis

        client = _redis.Redis.from_url(self._url, socket_timeout=None,
                                       decode_responses=True)
        client.ping()
        self._client = client
        # 重连后从"现在"开始听：断线期间的门铃已由降级轮询覆盖，
        # 从旧 id 补读只会把同一轮维护做两遍。
        self._last_id = "$"
        return client

    def wait(self, timeout_s: float) -> bool:
        try:
            client = self._connect()
            resp = client.xread({WAKE_STREAM: self._last_id},
                                block=max(int(timeout_s * 1000), 1), count=64)
        except Exception as e:  # noqa: BLE001 —— 掉线/超时以外的一切异常走降级
            self._client = None
            now = time.time()
            if now - self._last_warn >= self._WARN_COOLDOWN_S:
                self._last_warn = now
                logger.warning("flash_wake_degraded_polling",
                               error=str(e), poll_s=timeout_s)
            time.sleep(timeout_s)
            return True          # 降级 = 按轮询节奏干活
        if not resp:
            return False         # 干净超时：确定没有新数据
        for _stream, entries in resp:
            if entries:
                self._last_id = entries[-1][0]
        return True
