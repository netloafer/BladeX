"""交接包取数（批 CB）：从 Hub / Index 读出一本账本的 ②③，组装交给 `bladex_core.handoff`。

只读。**同步函数，admin 路由经 `run_blocking` 调**（整库 key 扫描约 3 s）。

为什么不用 `hub.scan_admin_events()`：它遍历全库连 Turn 正文一起读，而且对未知事件类型会抛错
（步骤 1 结算稿 §1.1）。这里按 `admin_event/` 前缀 seek，1,358 条 0.02 s，并且先看类型再解载荷，
不做枚举校验。

exposure 只走 `exposure_allows`，不另写一套（F1 边界）：
- ② Matter 没有 exposure 字段，取**成员 fact 里最严的 ceiling**（拍板 (41)-a-⑥）；读不出来 ⇒
  省略并计 `matters_unavailable`，不当 public。
- ③ 轮次取该 Turn 的 `sensitivity` 经 `SensitivityConfig` 映射；敏感度开着却找不到 Turn ⇒ 省略。
"""

from __future__ import annotations

import bisect
from datetime import datetime
from typing import Any

import structlog
from bladex_core.handoff import (
    HandoffLimits,
    HandoffOmitted,
    HandoffPackage,
    RawRound,
    build_package,
    limits_from_flags,
    rounds_from_events,
)
from bladex_core.ledger import Ledger
from bladex_core.ledger_runtime import ledger_anchor_matter_id, user_turn_index
from bladex_core.project import canonical_project_id
from bladex_core.sensitivity import (
    EXPOSURE_PUBLIC,
    SensitivityConfig,
    exposure_allows,
    exposure_rank,
)

from bladex_proxy.agency.notes import _last_user_text
from bladex_proxy.ledger_events import refresh_project_key_map

logger = structlog.get_logger(__name__)

_LEDGER_BODY_EVENTS = frozenset({"ledger_create", "ledger_update", "ledger_user_edit"})
_SCOPE_EVENTS = frozenset({"ledger_create", "ledger_switch", "ledger_scope_migrate"})
#: 没有项目信号的 scope 写法（`activation_scope` 的 Global 段与历史空串）。
_NO_PROJECT = frozenset({"", "Global"})
#: 同一会话里，从事件时刻往前 / 往后找几条 Turn。事件落在模型回复途中，Turn 写在回复之后。
_TURN_LOOK_BEHIND = 4
_TURN_LOOK_AHEAD = 8
#: 取一本 Matter 的成员 fact 时的上限：只为算最严 ceiling，宁可取全。
_MATTER_FACTS_MAX = 1_000_000


def _key_str(raw: Any) -> str:
    return raw.decode() if isinstance(raw, bytes) else str(raw)


def _parse_ts(ts: Any) -> datetime | None:
    if isinstance(ts, datetime):
        return ts
    if isinstance(ts, str) and ts:
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def read_ledger_events(hub: Any, ledger_id: str) -> tuple[list[dict[str, Any]], list[str]]:
    """这本账本的正文事件（按 Hub 顺序）与它出现过的项目键（现行键，先后序去重）。"""
    events: list[dict[str, Any]] = []
    projects: list[str] = []

    def _note(raw_project: Any) -> None:
        pid = canonical_project_id(str(raw_project or ""))
        if pid not in _NO_PROJECT and pid not in projects:
            projects.append(pid)

    # 批 UE：经 Hub 的单一判断点读原始载荷（未知类型汇总告警 / must_understand 拒绝都在那里）
    for _key, data in hub.scan_admin_event_dicts():
        etype = data.get("event_type", "")
        if etype not in _LEDGER_BODY_EVENTS and etype not in _SCOPE_EVENTS:
            continue
        payload = data.get("payload") or {}
        lid = (payload.get("ledger") or {}).get("ledger_id", "")
        if etype in _LEDGER_BODY_EVENTS and lid == ledger_id:
            events.append({"type": etype, "payload": payload, "ts": _parse_ts(data.get("ts"))})
        if etype == "ledger_create" and lid == ledger_id:
            _note(payload.get("project_id"))
        elif etype == "ledger_switch" and payload.get("to_ledger_id") == ledger_id:
            _note(payload.get("project_id"))
        elif etype == "ledger_scope_migrate" and payload.get("ledger_id") == ledger_id:
            _note(payload.get("to_project"))
    return events, projects


def _session_turn_keys(hub: Any, wanted: set[tuple[str, str]]) -> dict[tuple[str, str], list[tuple[str, str]]]:
    """`(agent, session)` → 升序 `[(entry_id, 完整 key)]`。只解 key，不碰 value。

    key 里的 principal 从事件载荷拿不到，所以要扫一遍 key。
    """
    out: dict[tuple[str, str], list[tuple[str, str]]] = {}
    if not wanted:
        return out
    for raw in hub.db.keys():
        key = _key_str(raw)
        parts = key.split("/")
        if len(parts) == 4 and (parts[1], parts[2]) in wanted:
            out.setdefault((parts[1], parts[2]), []).append((parts[3], key))
    for v in out.values():
        v.sort()
    return out


def _find_turn(hub: Any, keys: list[tuple[str, str]], turn_index: int, at: str) -> Any | None:
    """事件时刻附近、`user_turn_index` 相等的那条 Turn；找不到 ⇒ None（缺数不猜）。"""
    ts = _parse_ts(at)
    if ts is None:
        return None
    pos = bisect.bisect_left(keys, (f"{int(ts.timestamp() * 1000):013d}", ""))
    for _eid, key in keys[max(0, pos - _TURN_LOOK_BEHIND):pos + _TURN_LOOK_AHEAD]:
        turn = hub.get(key)
        if turn is not None and user_turn_index(turn.request_messages or []) == turn_index:
            return turn
    return None


def _turn_exposure(level: str, sensitivity: SensitivityConfig | None) -> str:
    """Turn 敏感等级 → 暴露上限；关闭态恒 public（与蒸馏血统同一语义）。"""
    if sensitivity is None or not sensitivity.enabled:
        return EXPOSURE_PUBLIC
    return sensitivity._level_exposure(level)


def collect_rounds(hub: Any, events: list[dict[str, Any]], ledger_id: str, *,
                   target_exposure: str, sensitivity: SensitivityConfig | None,
                   limits: HandoffLimits, omitted: HandoffOmitted,
                   project_roots: list[str]) -> list[RawRound]:
    """③：条目增删来自事件，用户话来自写者轮的 Turn。

    只为要进包的轮次取 Turn（因 exposure 省略的不占名额，再旧的留给 `build_package` 计数）。
    `_last_user_text` 是用户话的唯一取值点；2.6% 的首段不是用户话（MQ-L106），这里原样带出。
    """
    rounds = rounds_from_events(events, ledger_id)
    keys = _session_turn_keys(hub, {(r.agent, r.session_id) for r in rounds if r.session_id})
    sens_on = sensitivity is not None and sensitivity.enabled
    kept: list[RawRound] = []
    for pos, r in enumerate(rounds):
        if len(kept) >= limits.max_rounds:
            return kept + rounds[pos:]   # 再旧的只为计数：`build_package` 只取前 max_rounds 条
        turn = None
        if r.session_id and r.turn_index is not None:
            turn = _find_turn(hub, keys.get((r.agent, r.session_id), []), r.turn_index, r.at)
        if turn is None:
            if sens_on:              # 敏感度开着却无从判定 ⇒ 不出包
                omitted.rounds_exposure += 1
                continue
            kept.append(r)
            continue
        ceiling = _turn_exposure(getattr(turn, "sensitivity", "normal"), sensitivity)
        if not exposure_allows(ceiling, target_exposure):
            omitted.rounds_exposure += 1
            continue
        r.exposure = ceiling
        r.user_text = _last_user_text(turn.request_messages or [], for_goal=True)
        root = str(getattr(turn.identity, "project_root", "") or "")
        if root and root not in project_roots:
            project_roots.append(root)
        kept.append(r)
    return kept


def collect_matter(index: Any, ledger: Ledger, *, target_exposure: str,
                   omitted: HandoffOmitted) -> list[tuple[str, str]]:
    """②：锚定 Matter 的标题与摘要。不带成员 fact，只用它们算最严 ceiling。"""
    if index is None:
        omitted.matters_unavailable += 1
        return []
    matter_id = ledger.matter_id or ledger_anchor_matter_id(ledger.ledger_id)
    try:
        matter = index.get_matter(matter_id)
        if matter is None:           # 锚卡按需建：还没有 fact 落进来，不是读不出来
            return []
        facts = index.get_facts_for_matter(matter_id, top_k=_MATTER_FACTS_MAX)
    except Exception as e:  # noqa: BLE001 —— 判不出 exposure ⇒ 省略，绝不当 public
        logger.warning("ledger_handoff_matter_unavailable", ledger=ledger.ledger_id, error=str(e))
        omitted.matters_unavailable += 1
        return []
    ceilings = [getattr(f, "exposure_ceiling", "") or EXPOSURE_PUBLIC for f in facts]
    strictest = min(ceilings, key=exposure_rank, default=EXPOSURE_PUBLIC)
    if not exposure_allows(strictest, target_exposure):
        omitted.matters_exposure += 1
        return []
    return [(matter.title, matter.summary)]


def _project_roots(index: Any, project_ids: list[str]) -> list[str]:
    if index is None or not project_ids:
        return []
    try:
        return [p.root_path for p in index.all_projects()
                if p.project_id in project_ids and p.root_path]
    except Exception as e:  # noqa: BLE001 —— 拿不到项目根 ⇒ 退到家目录改写，路径照样不出包
        logger.warning("ledger_handoff_project_roots_failed", error=str(e))
        return []


def collect_package(hub: Any, index: Any, ledger: Ledger, *,
                    exported_at: str, target_exposure: str,
                    sensitivity: SensitivityConfig | None,
                    bladex_version: str = "",
                    limits: HandoffLimits | None = None) -> HandoffPackage:
    """组装一本账本的交接包（同步、只读）。"""
    lim = limits or limits_from_flags()
    refresh_project_key_map(hub)
    events, project_ids = read_ledger_events(hub, ledger.ledger_id)
    omitted = HandoffOmitted()
    roots = _project_roots(index, project_ids)
    rounds = collect_rounds(hub, events, ledger.ledger_id, target_exposure=target_exposure,
                            sensitivity=sensitivity, limits=lim, omitted=omitted,
                            project_roots=roots)
    matters = collect_matter(index, ledger, target_exposure=target_exposure, omitted=omitted)
    return build_package(
        ledger, exported_at=exported_at, bladex_version=bladex_version,
        source_project=project_ids[-1] if project_ids else "",
        target_exposure=target_exposure, project_roots=roots,
        matters=matters, rounds=rounds, omitted=omitted, limits=lim)
