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
  🔴 2026-09-18（批 S · S2.1，MQ-I19）：**墓碑时刻一并传入**——墓碑从它发生的那一刻
  起生效，不倒填到切入那一刻（语义与理由见 core 侧 docstring）。三处消费方自动跟随；
  只有重建锚层会看到多出来的"墓碑前窗口"，那正是要修的对象。
- `keep_tombstoned_pool=True`（**只有重建锚层用**）：被墓碑账本的 create/update/user_edit
  也保留，让池里有它的标题与 `matter_id`——重建要为"墓碑前窗口"里的轮次建锚卡，
  没有标题就只能拿 id 当标题。agency / flash **不得**传它（它们的池就是注入面与物化面）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog
from bladex_core.ledger import LEDGER_EVENT_TYPES, apply_ledger_tombstones
from bladex_core.project import canonical_project_id, set_project_key_map

logger = structlog.get_logger()

#: 账本墓碑的 target_type 字面量。取字符串而非 import 枚举：本模块被 core 侧
#: 逻辑与 proxy 存储层共用，避免把 proxy 的 models 拖进更浅的依赖层。
LEDGER_TOMBSTONE_TYPE = "ledger"


def _ts_ms(raw: Any) -> float:
    if raw is None:
        return 0.0
    if hasattr(raw, "timestamp"):
        return float(raw.timestamp()) * 1000.0
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return v * 1000.0 if v < 1e11 else v


def tombstoned_ledgers(hub: Any) -> dict[str, float]:
    """Hub 里被墓碑的账本 `{ledger_id: 最早墓碑时刻 ms}`；读不到返回空表（降级 = 不过滤）。

    时刻取**最早**一条：同一本被打两次墓碑，生效点是第一次。取不到 ts 的记 0.0
    （`apply_ledger_tombstones` 对 0.0 回落旧行为，缺数不造强信号）。
    """
    try:
        out: dict[str, float] = {}
        for _k, t in hub.scan_tombstones():
            if str(getattr(t.target_type, "value", t.target_type)) != LEDGER_TOMBSTONE_TYPE:
                continue
            ms = _ts_ms(getattr(t, "ts", None))
            if t.target_key not in out or (ms and ms < out[t.target_key]):
                out[t.target_key] = ms
        return out
    except Exception:  # noqa: BLE001 —— 墓碑读不到时宁可不过滤，也不让账本面整个瘫掉
        return {}


def tombstoned_ledger_ids(hub: Any) -> set[str]:
    """Hub 里被墓碑的账本 id 集合（`tombstoned_ledgers` 的键集；老调用方沿用）。"""
    return set(tombstoned_ledgers(hub))


#: 批 BA2（拍板 (32)-b）：一次性迁移写在 Hub 旁边的派生文件；Hub 本身不改写。
PROJECT_KEY_MAP_NAME = "project_key_map.json"
_key_map_state: dict[str, Any] = {"path": None, "mtime": None}


def refresh_project_key_map(hub: Any) -> None:
    """按 Hub 目录找映射文件并装进 `canonical_project_id`。文件缺失 = 空表（回滚 = 删文件）。

    **唯一的装载实现**，按 mtime 去重、重复调用无开销。两个调用点：这里下面的
    `load_ledger_events`，以及 `MemoryIndex._rebuild_from_hub_impl` 入口（批 BA2b：逐轮写 Project
    早于锚层装事件，新进程第一趟要在第一条轮次之前就有表）。
    """
    hub_path = getattr(hub, "_path", None)
    if hub_path is None:
        return                       # 快照包装器等没有路径的 Hub：不动已装载的表
    path = Path(hub_path).parent / PROJECT_KEY_MAP_NAME
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        mtime = None
    if _key_map_state["path"] == path and _key_map_state["mtime"] == mtime:
        return
    mapping: dict[str, str] = {}
    if mtime is not None:
        try:
            mapping = dict(json.loads(path.read_text(encoding="utf-8")).get("map") or {})
        except (OSError, ValueError, AttributeError) as e:
            logger.warning("project_key_map_load_failed", path=str(path), error=str(e))
    set_project_key_map(mapping)
    _key_map_state.update(path=path, mtime=mtime)
    if mapping:
        logger.info("project_key_map_loaded", entries=len(mapping))


def _translate_project_keys(payload: Any) -> Any:
    """事件载荷里的旧项目键换现行键；**不改原对象**（payload 是 Hub 读出来的）。"""
    if not isinstance(payload, dict):
        return payload
    changed = {f: canonical_project_id(payload[f])
               for f in ("project_id", "from_project", "to_project")
               if payload.get(f) and canonical_project_id(payload[f]) != payload[f]}
    return {**payload, **changed} if changed else payload


def load_ledger_events(hub: Any, *, keep_tombstoned_pool: bool = False) -> list[dict]:
    """从 Hub 装载账本事件（已过滤类型、已应用墓碑）。三个消费方共用。"""
    refresh_project_key_map(hub)
    tombs = tombstoned_ledgers(hub)
    events: list[dict] = []
    for _key, ev in hub.scan_admin_events():
        etype = getattr(ev.event_type, "value", str(ev.event_type))
        if etype in LEDGER_EVENT_TYPES:
            events.append({"type": etype, "payload": _translate_project_keys(ev.payload),
                           "ts": getattr(ev, "ts", None)})
    filtered = apply_ledger_tombstones(events, set(tombs), tombstone_ts_ms=tombs)
    if not keep_tombstoned_pool or not tombs:
        return filtered
    # 重建锚层：把被墓碑账本的池事件（create/update/user_edit）按原序补回。
    # switch 侧仍是 `filtered` 的产物（墓碑时刻已合成"切到空"），池只是多了标题。
    from bladex_core.ledger import (
        EVENT_LEDGER_CREATE,
        EVENT_LEDGER_UPDATE,
        EVENT_LEDGER_USER_EDIT,
    )
    pool_events = [
        ev for ev in events
        if ev["type"] in (EVENT_LEDGER_CREATE, EVENT_LEDGER_UPDATE, EVENT_LEDGER_USER_EDIT)
        and ((ev.get("payload") or {}).get("ledger") or {}).get("ledger_id", "") in tombs
    ]
    # 🔴 **池事件不得进时间轴**（批 T0，2026-09-22 起这句话是承重的）：`switch_timeline`
    # 现在也认 `ledger_create` 的「建本即激活」起点（只要载荷带 `agent_id`）。被墓碑
    # 账本的 create 在 `apply_ledger_tombstones` 里已被丢弃——那是墓碑语义的一部分；
    # 这里为了池里的标题又把它补回来，若原样补回就会**把墓碑该关掉的窗口重新开出来**
    # （尤其墓碑时刻未知、切入事件被改写成"切到空"的那一档）。故补回时剥掉 scope 字段：
    # 池要的是 `ledger` 载荷，时间轴的判据仍只有 `apply_ledger_tombstones` 一处。
    pool_events = [
        {**ev, "payload": {k: v for k, v in (ev.get("payload") or {}).items()
                           if k not in ("agent_id", "project_id")}}
        for ev in pool_events
    ]
    return pool_events + filtered
