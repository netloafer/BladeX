"""账本的**派生记忆窗口**（拍板 0.3.0-(2)，MQ-I17；批 S · S3，2026-09-18）。

# 语义

**墓碑账本 ≠ 清派生记忆**。账本是任务状态，记忆是知道的事；任务收摊不等于知识作废。
默认路径只给账本打墓碑，并**如实提示**"派生记忆保留 N 条"；要连记忆一起清，走显式的
第二动作 `--with-memory`。

# 落点 = Hub 层，不是 Index 层

批 Q · Q3 真跑证明：只在 Index 删 fact / matter，Hub 的 turn 还在，下一次全量重建
原样蒸出、换一张内容卡长回来（MQ-I17 S3）。删除语义必须落在**唯一真相源**上：对该账本
激活窗口内的源轮次追加 **TURN 墓碑**——`scan_prefix` / `scan_meta` 天然跳过被墓碑的
turn ⇒ 全量重建不再蒸出；增量路径用既有 `MemoryIndex.cascade_delete_turn` 级联清 Index
（proxy 进程里的 Index 是只读的，级联会像 `DELETE /admin/turns` 一样报 deferred，
MQ-I14 已知形态，下次重建收敛）。**走墓碑，不物理删**：原文仍在 Hub，`compact_tombstoned`
才擦，重建可重放出被墓碑对象（ADR-0012 §3.6）。

# 窗口 = 与生产归属同一条时间轴

窗口来自 `switch_timeline` + `ledger_activation_windows`（core），事件流经
`load_ledger_events(keep_tombstoned_pool=True)`——它已把该账本的墓碑时刻合成成
"切到空"（MQ-I19），所以窗口在墓碑时刻自然闭合。**不另起第二套判据**（MQ-A18/P7 形态）。

🔴 **scope 必须带真 project_id**（MQ-I22，2026-09-22 真跑撞出，批 T0 修）：此前这里写死
`activation_scope(agent, "")`，而 `windows` 的键是真实 scope `agent␟project` ⇒ **凡有
project 身份的账本一律查不中** ⇒ `turn_keys=[]` ⇒ CLI 打印 "✓ tombstoned … 0 turn
tombstones"，**0 长得像成功**（原则 13 层 4 形态）。它一直被掩盖是因为 project 识别
此前恒空、全落 Global，窗口"碰巧"对得上；project 识别修好之后这条反而失效了。
真跑证据：`p-e479f97c661a` 下四本账本 `--with-memory` 各打 0 条墓碑。

取值来源写死在一处：轮次的 project 只认 `MemoryHub.scan_meta_snapshot()` 行里的
`identity.project_id`（**已存在**的字段，不是新生产点；20.9k 轮实测 ≈4s，与
`scan_meta` 同价）。老轮次没有该字段 ⇒ 空 ⇒ `activation_scope` 自然落 Global，
与历史窗口对得上——**这是正确行为，不是兜底**。

N（提示行的数）= 窗口内源轮次蒸出的、仍 current 的 fact 数（与批 Q 仪器 `window_facts`
同口径），不是锚卡上的 fact 数。
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import structlog
from bladex_core.ledger import (
    ledger_activation_windows,
    replay_ledger_events,
    switch_timeline,
)
from bladex_core.ledger_runtime import turn_active_ledger

from bladex_proxy.ledger_events import load_ledger_events
from bladex_proxy.models import TombstoneSource, TombstoneTargetType

logger = structlog.get_logger(__name__)


def ledger_window_turn_keys(hub: Any, ledger_id: str, *, now_ms: float | None = None) -> dict[str, Any]:
    """该账本激活窗口内的源轮次 key（只读；已被墓碑的 turn 不在内，快照已滤）。

    返回 `{"windows": …, "turn_keys": [...], "scopes": [...], "known": bool, "title": str}`。
    窗口末端 `None`（仍激活）按 `now_ms` 截止。

    一趟 `scan_meta_snapshot()` 同时供四件事：轮次元数据（含 `project_id`）、墓碑过滤、
    账本事件装载（快照会冒充 hub 喂给 `load_ledger_events`）、以及 `known`/`title`
    ——**不为同一件事扫两遍库**。

    🔴 `known` 的口径是**事件流里有没有这本账本**，与"在不在活池"无关（批 T0b，MQ-I23）。
    `DELETE /admin/ledgers/{id}` 的 404 判的是活池（那是对的：墓碑是对活账本的动作），
    但**补清记忆**要够得着已墓碑的账本——窗口本来就是从事件流算的，与池无关。
    两个口径分居两处，不互相顶替。
    """
    now_ms = now_ms if now_ms is not None else datetime.now(UTC).timestamp() * 1000.0
    snapshot = hub.scan_meta_snapshot()
    events = load_ledger_events(snapshot, keep_tombstoned_pool=True)
    pool, _bindings = replay_ledger_events(events)
    lg = pool.get(ledger_id)
    timeline = switch_timeline(events)
    windows = ledger_activation_windows(timeline, ledger_id, until_ms=now_ms)
    keys: list[str] = []
    if windows:
        for key, meta in snapshot.rows:
            # 🔴 project 只此一处取（MQ-I22）；"这一轮激活的是哪本"只在
            # `turn_active_ledger` 一处判（批 Y · Y1a，MQ-L88：重建锚层同一函数）。
            # 批 Z · Z3.2：完结停锚同一函数判；写者用户轮豁免的 turn_index 惰性取
            # （快照行不带它，只有命中完结区间的轮才 `hub.get` 一次）——与重建锚层同一答案。
            if turn_active_ledger(timeline, key, str(meta.get("project_id", "") or ""),
                                  turn_index=lambda k=key: _turn_index_of(hub, k)) == ledger_id:
                keys.append(key)
    return {"windows": windows, "turn_keys": keys, "scopes": sorted(windows),
            "known": lg is not None, "title": (lg.title if lg and lg.title else ledger_id)}


def _turn_index_of(hub: Any, key: str) -> int | None:
    """该轮 `identity.turn_index`；取不到 ⇒ None（完结停锚不豁免）。"""
    try:
        turn = hub.get(key)
    except Exception as e:  # noqa: BLE001 —— 取不到只影响这一轮的豁免
        logger.warning("ledger_window_turn_index_get_failed", key=key, error=str(e))
        return None
    ti = getattr(getattr(turn, "identity", None), "turn_index", None) if turn else None
    return int(ti) if isinstance(ti, int) else None


def count_window_facts(index: Any, turn_keys: list[str]) -> int | None:
    """窗口内源轮次蒸出的 current fact 数；Index 不可用 ⇒ None（NO-DATA ≠ 0）。"""
    if index is None or not turn_keys:
        return 0 if index is not None else None
    wanted = set(turn_keys)
    try:
        return sum(1 for f in index.all_facts()
                   if f.t_invalid is None and (f.source_ledger_key or "") in wanted)
    except Exception as e:  # noqa: BLE001 —— 数不出来要如实说，不当 0
        logger.warning("ledger_window_facts_count_failed", error=str(e))
        return None


def tombstone_ledger_memory(hub: Any, index: Any, turn_keys: list[str]) -> dict[str, Any]:
    """对窗口内每条源轮次追加 TURN 墓碑（Hub 层），并尽力级联清 Index（可失败=deferred）。

    与 `DELETE /admin/turns/{key}` 同一对调用（`append_tombstone(TURN)` + `cascade_delete_turn`），
    不另起路径。返回 `{"turn_tombstones": n, "index_cascaded": m, "index_deferred": k}`。
    """
    written = 0
    cascaded = 0
    deferred = 0
    for key in turn_keys:
        hub.append_tombstone(target_type=TombstoneTargetType.TURN, target_key=key,
                             source=TombstoneSource.USER)
        written += 1
        if index is None:
            deferred += 1
            continue
        try:
            index.cascade_delete_turn(key)
            cascaded += 1
        except Exception as e:  # noqa: BLE001 —— 墓碑已写，Index 下次重建收敛（MQ-I14 形态）
            deferred += 1
            logger.warning("index_cascade_delete_failed", error=str(e), turn_key=key,
                           hint="tombstone written, Memory Index cleanup deferred")
    logger.info("ledger_memory_tombstoned", turn_tombstones=written,
                index_cascaded=cascaded, index_deferred=deferred)
    return {"turn_tombstones": written, "index_cascaded": cascaded, "index_deferred": deferred}

