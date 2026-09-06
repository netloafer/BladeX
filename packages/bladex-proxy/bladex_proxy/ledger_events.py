"""账本事件的**唯一**装载入口（2026-09-01，`task-ledger-tombstone-cleanup-20260901.md`）。

# 为什么要有这个模块

账本事件有三个消费方，此前**各自扫一遍 Hub**：

- `agency._load_pool`（proxy 会话：LEDGERS 注入 / 工具面 / switch）
- `memory_index.rebuild_from_hub` 的锚层（全量重建：Matter 锚定）
- `flash_daemon` 的 pool source（`.md` 物化）

三处做同一件事，而判据已经开始分叉：前两处用 core 的 `LEDGER_EVENT_TYPES`
（两处注释都写着"两边各抄一份迟早分叉"），**重建路径却自己写了
`_et.startswith("ledger_")`**。今天两者恰好等价，加任何新 `ledger_*` 事件就不是了。

墓碑要生效必须**三处都过滤**，而"同一件事在三个地方各做一遍"正是本仓最高发的
缺陷形态（MQ-A18 / MQ-P4 / MQ-L23 都是它）。故收成一个函数：
**判据一处、墓碑一处、ts 一处。**

# 口径

- 事件类型判据取 core 的 `LEDGER_EVENT_TYPES`（单一真相源）。
- **`ts` 一律带上**：`switch_timeline` 的点时查找要它；此前只有重建路径传，
  另两处不传——它们今天不用时间轴，但少一个字段就是给下一个消费方埋雷。
- 墓碑过滤走 core 的 `apply_ledger_tombstones`（纯函数，语义与守卫都在那里）。
"""

from __future__ import annotations

from typing import Any

from bladex_core.ledger import LEDGER_EVENT_TYPES, apply_ledger_tombstones

#: 账本墓碑的 target_type 字面量。取字符串而非 import 枚举：本模块被 core 侧
#: 逻辑与 proxy 存储层共用，避免把 proxy 的 models 拖进更浅的依赖层。
LEDGER_TOMBSTONE_TYPE = "ledger"


def tombstoned_ledger_ids(hub: Any) -> set[str]:
    """Hub 里被墓碑的账本 id 集合；读不到返回空集（降级 = 不过滤，不是拒服务）。"""
    try:
        return {
            t.target_key
            for _k, t in hub.scan_tombstones()
            if str(getattr(t.target_type, "value", t.target_type)) == LEDGER_TOMBSTONE_TYPE
        }
    except Exception:  # noqa: BLE001 —— 墓碑读不到时宁可不过滤，也不让账本面整个瘫掉
        return set()


def load_ledger_events(hub: Any) -> list[dict]:
    """从 Hub 装载账本事件（已过滤类型、已应用墓碑）。三个消费方共用。"""
    tombstoned = tombstoned_ledger_ids(hub)
    events: list[dict] = []
    for _key, ev in hub.scan_admin_events():
        etype = getattr(ev.event_type, "value", str(ev.event_type))
        if etype in LEDGER_EVENT_TYPES:
            events.append({"type": etype, "payload": ev.payload,
                           "ts": getattr(ev, "ts", None)})
    return apply_ledger_tombstones(events, tombstoned)
