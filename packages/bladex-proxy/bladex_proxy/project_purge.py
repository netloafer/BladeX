"""按 **project 作用域**选轮次打墓碑（批 T0b 路径②，2026-09-22）。

# 为什么要有它

拍板 (3) 的清仓有两条线：**按账本清**（`bladex ledger tombstone [--with-memory]`，
批 S/T0 已有）与**按 project 作用域整体清**（A 档 30 个 bench project / 3,375 轮、
B 档 `bladex-r4` 45 轮）。后者**全仓没有任何 CLI / 端点能做**——`memory forget` 是
fact 级、`ledger tombstone` 是账本窗口级，都够不着"这个 project 的全部轮次"。

# 数据来源写死一处

轮次的 project 只认 :meth:`MemoryHub.scan_meta_snapshot` 行里的 `identity.project_id`
（F2 的产物，本模块是它的第三个消费方）。**不改它的签名，也不新起第二个取值点**
——MQ-I22 的成因就是同一件事两处各做一次。快照已滤掉被墓碑的 turn ⇒ 本路径天然幂等：
补跑一次，第二次的 `turn_keys` 不含上一次已墓碑的轮次。

# 硬拒绝空 project

`project_id=""`（Global）是 **17,047 轮**——project 识别恒空那段时期的全部流量。
拍板 (3) 明写那一半**只能按账本清**。一次误操作就是灾难性的，故本模块在**计算层**
就拒绝空串，而不是指望调用方记得传对（端点、CLI 各自再拒一次是纵深，不是替代）。

# 只墓碑 turn，不碰账本

拍板 (3) 的两条线不合并：本路径一条 LEDGER 墓碑都不打。响应里给
`ledgers`（该 project 作用域上出现过的账本数）只是**影响面读数**，供调用方另行处置。
"""

from __future__ import annotations

from typing import Any

import structlog
from bladex_core.ledger import switch_timeline
from bladex_core.ledger_runtime import agent_of_turn_key, split_scope

from bladex_proxy.ledger_events import load_ledger_events

logger = structlog.get_logger(__name__)


class EmptyProjectRefused(ValueError):
    """空 project（Global）被拒 —— 见模块 docstring「硬拒绝空 project」。"""


def project_scope_preview(hub: Any, project_ids: list[str]) -> dict[str, dict[str, Any]]:
    """每个 project 一份影响面读数（只读；**一趟扫描服务全部 project**）。

    :param project_ids: 非空 `p-…` 列表。含空串 ⇒ :class:`EmptyProjectRefused`。
    :returns: `{project_id: {"turn_keys", "turn_count", "agents", "first_ts", "last_ts",
              "project_names", "ledgers"}}`。请求了但一条轮次都没有的 project 也在返回里
              （读数为 0），**不静默丢键**——"没有这个 project"与"我没查"必须能分开。
    """
    wanted = list(dict.fromkeys(project_ids))  # 去重保序
    if any(not (p or "").strip() for p in wanted):
        raise EmptyProjectRefused(
            "empty project_id (Global) is refused: it is the whole pre-project-identity "
            "backlog; clear it per ledger instead.")
    if not wanted:
        raise EmptyProjectRefused("no project_id given")

    snapshot = hub.scan_meta_snapshot()
    out: dict[str, dict[str, Any]] = {
        p: {"turn_keys": [], "turn_count": 0, "agents": {}, "first_ts": None,
            "last_ts": None, "project_names": [], "ledgers": 0}
        for p in wanted
    }
    names: dict[str, set[str]] = {p: set() for p in wanted}
    target = set(wanted)
    for key, meta in snapshot.rows:
        pid = str(meta.get("project_id", "") or "")
        if pid not in target:
            continue
        row = out[pid]
        row["turn_keys"].append(key)
        agent = agent_of_turn_key(key) or "(unknown)"
        row["agents"][agent] = row["agents"].get(agent, 0) + 1
        ts = str(meta.get("ts", "") or "")
        if ts:
            row["first_ts"] = ts if row["first_ts"] is None else min(row["first_ts"], ts)
            row["last_ts"] = ts if row["last_ts"] is None else max(row["last_ts"], ts)
        nm = str(meta.get("project_name", "") or "")
        if nm:
            names[pid].add(nm)

    # 关联账本数：该 project 作用域上出现过的 distinct ledger_id（含已墓碑的——
    # 它们同样是"影响面"的一部分，调用方要据此决定账本那条线怎么走）。
    timeline = switch_timeline(load_ledger_events(snapshot, keep_tombstoned_pool=True))
    per_project: dict[str, set[str]] = {p: set() for p in wanted}
    for scope, seq in timeline.items():
        _agent, project = split_scope(scope)
        if project in per_project:
            per_project[project].update(lid for _ts, lid in seq if lid)

    for pid, row in out.items():
        row["turn_count"] = len(row["turn_keys"])
        row["project_names"] = sorted(names[pid])
        row["ledgers"] = len(per_project[pid])
    logger.info("project_scope_preview", projects=len(wanted),
                turns=sum(r["turn_count"] for r in out.values()))
    return out
