"""`behavior.json` 的**唯一读取点**（F3；写侧是 `flash_daemon._materialize_behavior`）。

消费方 = 运维面 `GET /admin/behavior`（`admin_read.py`，dashboard 的数据源）。
（F3 起还有模型面 `bladex_flash_read`；10-17 F5 判决退出注册，handler 已删。
再长出第二个读者时，照旧经本模块，不另开文件。）

🔴 **为什么必须是一处**：两个面各读一次文件，就会各自长出一套"文件不在怎么办 /
过期算不算数 / 时间怎么折"的处置，而这两套只有在同一天写的时候才一致
（本仓的 MQ-A18 / MQ-P7 形态：同一件事只在一个端点做对了）。守卫见
`tests/test_behavior_read_single_point.py`。

空态契约（判据④，与 `load_agents_roster` 缺文件返回 `""` 同族——**不抛异常、
不编一段**）：文件缺失 / 读不动 / 解析不了 / 过期太久，都返回 `behavior=None`
且 `reason` 具名，由调用方翻成自己那一面的文案。🔴 **`reason` 分四类而不是一个
布尔**：`missing`（daemon 还没跑过这条轨道）与 `stale`（跑过、停了）对运维是两件
完全不同的事，合成一个"没有数据"就是把断线藏起来（刚性原则 13 层 4）。
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

try:
    import structlog

    logger = structlog.get_logger(__name__)
except ImportError:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)  # type: ignore[assignment]

from bladex_core.behavior import Behavior
from bladex_core.flash import SCOPE_PERSONAL, behavior_path, read_behavior_status

from bladex_proxy.config import resolve_flash_path

#: 个人模式的 principal（与 `agency.notes.load_agents_roster` 同一处置：企业形态的
#: per-principal 读法归后续，稳定层是进程级冻结，与 per-request principal 天然不合）。
DEFAULT_PRINCIPAL = "local"

#: 超过这个年龄就当空态（`meta.generated_at` 起算）。取 24h 的理由：卡上最显眼的读数
#: 是 `last_seen` / `turns_30d`，它们隔一天就会把"停了的 agent"说成活跃的——那时候
#: **给一张旧卡比不给更坏**（它不会自己说"我旧了"，模型也没有别的通道知道）。
#: 24h 以内照给，并在返回值里带 `age_s`，让调用方自己决定要不要说。
#: 对照：`FLASH_LAG_BOUND_S`（360s）量的是**另一件事**——daemon 一趟没跟上 Hub 水位，
#: 那是告警阈值不是可用性阈值，两者不许混用同一个数。
BEHAVIOR_STALE_BOUND_S = 24 * 3600

REASON_OK = ""
REASON_MISSING = "missing"
REASON_UNREADABLE = "unreadable"
REASON_INVALID = "invalid"
REASON_STALE = "stale"


@dataclass(frozen=True)
class BehaviorCard:
    """一次读取的全部结果（两个面共用）。"""

    behavior: Behavior | None
    path: str
    reason: str = REASON_OK
    #: `meta.generated_at` 到现在的秒数；解析不出 ⇒ None（**不补 0**）
    age_s: int | None = None
    #: `.flash_state.json` 的 `behavior_status` 段（`flash_lag_s` 等）；没有 ⇒ 空 dict
    status: dict = field(default_factory=dict)
    #: 文件 mtime（epoch 秒）；读不到 ⇒ None
    mtime: float | None = None

    @property
    def ok(self) -> bool:
        return self.behavior is not None


def _age_s(behavior: Behavior, *, now: float) -> int | None:
    raw = (behavior.meta.generated_at or "").strip()
    if not raw:
        return None
    try:
        ts = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return max(0, int(now - ts.timestamp()))


#: MQ-F11（批 Z1b 续，2026-09-24）：墙上时钟的**唯一注入点**。生产恒为 `time.time`，默认行为逐字不变；
#: 判定逻辑与阈值（`BEHAVIOR_STALE_BOUND_S`）一概不动。存在理由：`/admin/behavior`（及 F5 前的工具面）都是无参调用
#: `read_behavior_card()`，用例够不到 `now=` 参数，只能拿写死的 NOW 去碰真实时钟——
#: 已删的 `test_flash_read_tool.py` 就这样在 09-23 12:00 UTC（NOW + 24 h）起对所有棒红。用例 monkeypatch 这里。
wall_clock: Callable[[], float] = time.time


def read_behavior_card(*, root: str = "", principal: str = DEFAULT_PRINCIPAL,
                       scope: str = SCOPE_PERSONAL, now: float | None = None,
                       stale_bound_s: int = BEHAVIOR_STALE_BOUND_S) -> BehaviorCard:
    """读一次 `behavior.json`（毫秒级：28KB 文件 + 一次 pydantic 解析，无锁、无进程通信）。

    `root` 留空 ⇒ `resolve_flash_path()`（与写侧 daemon 同一个解析口径）。
    `stale_bound_s <= 0` ⇒ 不做过期判定（给守卫与取证用）。
    """
    now = wall_clock() if now is None else now
    # 根目录解析**一次**（两处各解一次 = 同一件事两个来源，正是本模块要防的形状）
    flash_root = root or resolve_flash_path()
    path = behavior_path(flash_root, principal, scope=scope)
    status = read_behavior_status(flash_root)
    try:
        mtime: float | None = os.path.getmtime(path)
    except OSError:
        mtime = None
    try:
        with open(path, encoding="utf-8") as f:
            raw = f.read()
    except FileNotFoundError:
        logger.info("behavior_card_missing", path=path)
        return BehaviorCard(None, path, REASON_MISSING, status=status)
    except OSError as e:
        logger.warning("behavior_card_unreadable", path=path, error=str(e))
        return BehaviorCard(None, path, REASON_UNREADABLE, status=status)
    try:
        behavior = Behavior.model_validate_json(raw)
    except ValueError as e:
        logger.warning("behavior_card_invalid", path=path, error=str(e))
        return BehaviorCard(None, path, REASON_INVALID, status=status, mtime=mtime)
    age = _age_s(behavior, now=now)
    if stale_bound_s > 0 and age is not None and age > stale_bound_s:
        logger.warning("behavior_card_stale", path=path, age_s=age,
                       bound_s=stale_bound_s)
        return BehaviorCard(None, path, REASON_STALE, age_s=age, status=status,
                            mtime=mtime)
    return BehaviorCard(behavior, path, REASON_OK, age_s=age, status=status,
                        mtime=mtime)
