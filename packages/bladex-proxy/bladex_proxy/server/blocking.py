"""admin 路由里同步存储操作的**唯一**出口：`run_blocking`（批 DS1a，MQ-W24）。

# 为什么

`async def` 路由体里直接调 Hub / Index 的同步方法 = 在事件循环线程上跑它。live 实测
`GET /admin/matters/merge_candidates` 33 s、`/admin/agents` 7 s、`/admin/sessions` 4 s——
这段时间里事件循环什么都做不了，**所有 agent 的 `/v1` 请求一起排队**，Redis 读也会超时
（盘点稿 `docs/reviews/dashboard-gap-inventory-20261022.md` DS-07）。

# 为什么只有一个出口

各写各的 `asyncio.to_thread` 也能解卡，但那样就没有一处能回答两件事：
① 现在有多少同步读在线程里跑；② 关停时它们跑完了没有。第二件是真问题——
`metrics.backlog_sampler_loop` 的 docstring 记过：cancel 打断的只是 `await`，线程里的读还在跑，
随后 lifespan 关掉 Hub ⇒ 在已关闭的 RocksDB 上迭代。所以 lifespan 关句柄前调 `drain_blocking()`。

守卫 `tests/test_ds1a_admin_blocking_guard.py`：admin 路由体里直接调 Hub / Index 句柄（不经本出口）⇒ 红。

# 线程安全前提

批 DS1a 步骤 1 逐对象核过（结算稿 `docs/planning/report-batch-DS1a-20261025.md` §1.2）：
RocksDB / LanceDB / FTS sqlite 句柄可多线程同读；Hub 墓碑缓存已改为代次令牌（同批）。
`_agent_registry` **不是**线程安全的——它的读留在事件循环上，不许交给本出口。
"""

from __future__ import annotations

import asyncio
import weakref
from collections.abc import Callable
from typing import Any, TypeVar

import structlog

logger = structlog.get_logger()

T = TypeVar("T")

#: 每个事件循环一组在途 future（测试里每个 TestClient 有自己的循环，不能混在一起 gather）。
_INFLIGHT: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, set[asyncio.Future[Any]]] = (
    weakref.WeakKeyDictionary())


async def run_blocking(fn: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:  # noqa: ANN401
    """在工作线程里跑同步函数 `fn(*args, **kwargs)`，不占事件循环。

    请求被取消（客户端断开）时线程照样跑完——Python 线程打断不了；`shield` 让这个 future
    仍留在在途集合里，关停时 `drain_blocking()` 等得到它。
    """
    loop = asyncio.get_running_loop()
    fut: asyncio.Future[T] = asyncio.ensure_future(asyncio.to_thread(fn, *args, **kwargs))
    inflight = _INFLIGHT.setdefault(loop, set())
    inflight.add(fut)
    fut.add_done_callback(inflight.discard)
    return await asyncio.shield(fut)


def inflight_count() -> int:
    """当前事件循环上还在线程里跑的同步调用数（测试与关停日志用）。"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return 0
    return len(_INFLIGHT.get(loop, ()))


async def drain_blocking() -> int:
    """等当前事件循环上所有在途调用跑完（lifespan 关 Hub / Index 之前调）。返回等了几个。

    不设超时：句柄关在一次读的半截比停机慢几十秒更糟（见模块 docstring）。
    """
    loop = asyncio.get_running_loop()
    pending = [f for f in _INFLIGHT.get(loop, ()) if not f.done()]
    if pending:
        logger.info("admin_blocking_draining", count=len(pending))
        await asyncio.gather(*pending, return_exceptions=True)
    return len(pending)
