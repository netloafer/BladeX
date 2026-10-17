"""Storage startup helpers shared by background workers."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

#: 空部署首启时，Memory Hub 由 proxy lifespan 首次创建；secondary/read-write 打不开
#: 缺库是预期瞬态。0.5s 足够让 uvicorn lifespan 完成 Rdict 创建，又不会把真实配置
#: 错误拖成一分钟。
HUB_POLL_INTERVAL_S = 0.5


def hub_missing(exc: BaseException) -> bool:
    """是否是“主库目录尚未创建”这一类缺库错误。

    RocksDict 把 IO error 包成内建 `Exception`，没有暴露 Status/RocksDBException，
    只能按消息判。只认 `.../CURRENT: No such file or directory`；权限错误、锁冲突、
    损坏等其他错误不重试，保持 fail-fast。
    """
    text = str(exc)
    return "No such file or directory" in text and "/CURRENT" in text


def open_hub_with_startup_window(
    hub: Any, *, max_wait_s: float | None = None,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.monotonic,
) -> float:
    """打开 Memory Hub；空部署首启允许在一个有界窗口内等 proxy 创建主库。

    MQ-A81 选方案②（等 Hub）而不是方案①（cli 等 proxy /ready 再拉 flash）：Flash 和
    consolidator 对 Hub 的依赖在这里自证，不要求调用方记住一个隐藏的进程顺序；也不会
    把 Hub 消费方的投影/consolidation 推迟到 proxy 就绪之后。非“缺 CURRENT”错误原样
    抛出；超时也原样抛出，不靠“让用户再跑一次 start”。
    """
    from bladex_core.flags import flag_number  # noqa: PLC0415 —— 入口读配置（红线 4）
    if max_wait_s is None:
        max_wait_s = flag_number("BLADEX_HUB_STARTUP_WAIT_S")
    t0 = now()
    while True:
        try:
            hub.open()
            waited = round(now() - t0, 1)
            if waited:
                logger.info("hub_startup_opened", waited_s=waited)
            return waited
        except Exception as exc:
            if not hub_missing(exc):
                raise
            elapsed = now() - t0
            if elapsed >= max_wait_s:
                logger.warning("hub_startup_missing_after_window",
                               waited_s=round(elapsed, 1), max_wait_s=max_wait_s,
                               error=str(exc))
                raise
            logger.warning("hub_startup_waiting", waited_s=round(elapsed, 1),
                           max_wait_s=max_wait_s, error=str(exc))
            sleep(HUB_POLL_INTERVAL_S)
