"""用量聚合（批 AC · AC1）—— Hub 轮次 → agent × 模型 × 会话 × 天 × lane 的汇总，**单一实现点**。

北极星基线（ADR-0029 N2/N3）、批 BD 的 cache 对照、DS2 用量面板、`GET /admin/usage` 与
`bladex usage` 都读这里，不各自再算一遍（一件事一个机制）。

## 口径

- **数据源只有 Hub**：每轮 `response_meta`。有原文（`usage_raw`，批 AC 起）走
  `capture.usage_norm`；没有原文的旧轮次只从三键 `usage.{prompt, completion}` 取 input / output——
  那两个数当年就是上游原值，不是推断；**cache_read 旧轮次没有，计入 NO-DATA，不计 0**
  （MQ-P34：它在批 AC 之前只进日志）。`usage` 也为空（上游没报 / 失败轮）⇒ 三个字段全 NO-DATA。
- **NO-DATA 单独计数**：每组每字段报 `<字段>_nodata` = 该字段没有数据的轮数；和只加有数据的轮。
  读数时和要与 nodata 一起看——"和小"可能只是"没数据的轮多"。
- **lane 三分**：`main`（agent 主请求）/ `aux`（agent 发来但判为辅助的请求，如 CC 子调用）/
  `bladex`（BladeX 自身发起、已入 Hub 的调用——目前只有内循环 `aux_source=inner_loop`）。
  BladeX 其余自身调用（蒸馏 / 裁决 / L4 链接 / 装配复核 / 文件摘要 / 路由裁判 / embedding）
  **不入 Hub**，这里看不到——清单与修法见批 AC 结算稿 §9。
- **天**按调用方给的时区切（`tz`，默认本机时区）；输出里写明所用偏移。Hub `ts` 是 UTC。
- **窗口** `[since, until)`，都是带时区的时刻；None = 不限。
- 墓碑覆盖的轮次不计（与 `scan_meta` 同一规则）。

纯函数 `turn_usage` / `usage_row` / `aggregate` 由 `tests/test_ac_usage_aggregate.py` 钉死。
"""

from __future__ import annotations

import sys
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, tzinfo
from typing import Any

import structlog

from bladex_proxy.capture import usage_norm
from bladex_proxy.models import INNER_LOOP_AUX_SOURCE

logger = structlog.get_logger()

LANE_MAIN = "main"
LANE_AUX = "aux"
LANE_BLADEX = "bladex"
LANES = (LANE_MAIN, LANE_AUX, LANE_BLADEX)

#: BladeX 自身发起、入 Hub 的 aux 来源。新增一类自身调用入 Hub 时加在这里。
BLADEX_SELF_AUX_SOURCES = frozenset({INNER_LOOP_AUX_SOURCE})

#: 汇总的三个用量字段（顺序即输出顺序）。
FIELDS = ("input", "output", "cache_read")


def turn_usage(response_meta: Mapping[str, Any] | None) -> dict[str, int | None]:
    """一轮的 input / output / cache_read（None = NO-DATA）。读方唯一实现点。"""
    meta = response_meta or {}
    raw = meta.get("usage_raw") or {}
    if raw:
        n = usage_norm(raw)
        return {"input": n["input"], "output": n["output"], "cache_read": n["cache_read"]}
    legacy = meta.get("usage") or {}
    if not legacy:
        return {"input": None, "output": None, "cache_read": None}

    def _int(v: Any) -> int | None:  # noqa: ANN401
        return v if isinstance(v, int) and not isinstance(v, bool) else None

    return {"input": _int(legacy.get("prompt")), "output": _int(legacy.get("completion")),
            "cache_read": None}


@dataclass(frozen=True)
class UsageRow:
    """一轮的聚合视图（从 Hub 值切出来，不含正文）。"""

    key: str
    ts: datetime
    agent: str
    model: str
    session: str
    lane: str
    failed: bool
    has_raw: bool
    input: int | None
    output: int | None
    cache_read: int | None


def _parse_ts(v: Any) -> datetime | None:  # noqa: ANN401
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=UTC)
    if not v:
        return None
    try:
        dt = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def lane_of(data: Mapping[str, Any]) -> str:
    """main / aux / bladex。"""
    if str(data.get("aux_source", "") or "") in BLADEX_SELF_AUX_SOURCES:
        return LANE_BLADEX
    return LANE_AUX if data.get("auxiliary") else LANE_MAIN


def usage_row(key: str, data: Mapping[str, Any]) -> UsageRow | None:
    """Hub 原始值（msgpack 解出的 dict，或 `Turn.model_dump(mode="json")`）→ `UsageRow`；没有 ts 返回 None。"""
    ts = _parse_ts(data.get("ts"))
    if ts is None:
        return None
    ident = data.get("identity") or {}
    meta = data.get("response_meta") or {}
    u = turn_usage(meta)
    # intern：Hub 行缓存（`_RecentTurnIndex`）常驻全部轮次，agent / model / session 高度重复
    return UsageRow(
        key=key, ts=ts,
        agent=sys.intern(str(ident.get("agent_id", "") or "")),
        model=sys.intern(str(data.get("model", "") or "")),
        session=sys.intern(str(ident.get("session_id", "") or "")),
        lane=lane_of(data),
        failed=str(data.get("status", "") or "") == "failed",
        has_raw=bool(meta.get("usage_raw")),
        input=u["input"], output=u["output"], cache_read=u["cache_read"],
    )


def _empty_bucket() -> dict[str, int]:
    b = {"turns": 0, "failed_turns": 0, "raw_turns": 0}
    for f in FIELDS:
        b[f] = 0
        b[f"{f}_nodata"] = 0
    return b


def _add(bucket: dict[str, int], row: UsageRow) -> None:
    bucket["turns"] += 1
    bucket["failed_turns"] += int(row.failed)
    bucket["raw_turns"] += int(row.has_raw)
    for f in FIELDS:
        v = getattr(row, f)
        if v is None:
            bucket[f"{f}_nodata"] += 1
        else:
            bucket[f] += v


def _offset_text(tz: tzinfo, at: datetime) -> str:
    off = at.astimezone(tz).utcoffset()
    if off is None:
        return "+00:00"
    total = int(off.total_seconds())
    sign = "+" if total >= 0 else "-"
    total = abs(total)
    return f"{sign}{total // 3600:02d}:{total % 3600 // 60:02d}"


def aggregate(
    turns: Iterable[tuple[str, Mapping[str, Any]]],
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    agent: str | None = None,
    model: str | None = None,
    session: str | None = None,
    tz: tzinfo | None = None,
) -> dict[str, Any]:
    """Hub 轮次 `(key, 原始值)` 的只读迭代 → 汇总（纯函数：同输入同输出，不碰存储）。

    返回 `{"window", "filters", "groups", "totals"}`：
    - `groups`：按 (agent, model, session, day, lane) 一行，按这五元组排序；
    - `totals`：`{"all": …, "main": …, "aux": …, "bladex": …}`。
    每个桶 = `turns / failed_turns / raw_turns / input / output / cache_read / <字段>_nodata`。
    """
    zone = tz or datetime.now().astimezone().tzinfo or UTC
    groups: dict[tuple[str, str, str, str, str], dict[str, int]] = {}
    totals = {name: _empty_bucket() for name in ("all", *LANES)}
    for key, data in turns:
        row = data if isinstance(data, UsageRow) else usage_row(key, data)
        if row is None:
            continue
        if since is not None and row.ts < since:
            continue
        if until is not None and row.ts >= until:
            continue
        if agent is not None and row.agent != agent:
            continue
        if model is not None and row.model != model:
            continue
        if session is not None and row.session != session:
            continue
        day = row.ts.astimezone(zone).date().isoformat()
        gk = (row.agent, row.model, row.session, day, row.lane)
        _add(groups.setdefault(gk, _empty_bucket()), row)
        _add(totals["all"], row)
        _add(totals[row.lane], row)
    ref = since or until or datetime.now(UTC)
    return {
        "window": {
            "since": since.isoformat() if since else None,
            "until": until.isoformat() if until else None,
            "tz": _offset_text(zone, ref),
        },
        "filters": {"agent": agent, "model": model, "session": session},
        "groups": [
            {"agent": a, "model": m, "session": s, "day": d, "lane": ln, **groups[(a, m, s, d, ln)]}
            for (a, m, s, d, ln) in sorted(groups)
        ],
        "totals": totals,
    }


def hub_usage_summary(hub: Any, **filters: Any) -> dict[str, Any]:  # noqa: ANN401  MemoryHub
    """`aggregate` 喂 Hub 行缓存（`MemoryHub.usage_rows()`，批 AC2）——`GET /admin/usage` 的数据出口。

    同步、可能触发首次建缓存的全扫：调用方（admin 路由）经 `run_blocking` 调。
    `filters` 原样转给 `aggregate`（since / until / agent / model / session / tz）。
    """
    return aggregate(((r.key, r) for r in hub.usage_rows()), **filters)


def iter_hub_turns(hub: Any) -> Iterator[tuple[str, dict[str, Any]]]:  # noqa: ANN401  MemoryHub
    """Hub 全部现行 Turn 的 `(key, 原始值)`——只 `msgpack.unpackb` 顶层，不校验、不解 content_ref。

    与 `MemoryHub.scan_meta` 同一套跳过规则（内部 key / 墓碑 / 解码失败）。只读：不写库。
    """
    import msgpack

    from bladex_proxy.storage.memory_hub import _is_internal_key

    tombstoned = hub._tombstoned_keys()
    for key_raw, value_raw in hub.db.items():
        key_str = key_raw.decode() if isinstance(key_raw, bytes) else str(key_raw)
        if _is_internal_key(key_str) or key_str in tombstoned:
            continue
        try:
            data = msgpack.unpackb(value_raw, raw=False)
        except Exception as e:  # noqa: BLE001 —— 脏值不该中断整轮扫描
            logger.warning("usage_scan_decode_failed", key=key_str, error=str(e))
            continue
        if isinstance(data, dict):
            yield key_str, data
