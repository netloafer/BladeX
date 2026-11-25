"""批 BA2（MQ-A75 + MQ-I36，拍板 (32)-b/c）：项目键统一的一次性迁移。**可重跑、只动派生层，Hub 零改写。**

做三件事：

1. **旧键 → 现行键映射**（`data/project_key_map.json`）。Hub 里的老轮次 / 老事件存的是
   `p-` + sha256(`git-remote:{原始 url}`)[:12]，哈希不可逆，所以从轮次记下的 `project_root` 读
   `.git/config`，**正向算一遍旧键，对得上才算这个仓库的**（原则 13 层 4：值要能回溯到它声称的
   来源）。目录已不在 / remote 改过 ⇒ 对不上，如实计入 `unmapped`，旧键继续原样可查。
   激活位没有单独的迁移动作：`load_ledger_events` 读事件时按这张表翻译，重放自动落到新 scope。
2. **删假 Project**：root_path 不是绝对路径的（生产恰好一条 `project/a54ff182c7e8`，
   root_path=`AGENTS.md`）。新代码不会再产生它（`_maintain_projects_recompute`）。
3. **按轮次身份回填 Project 表**：键 = `canonical_project_id(identity.project_id)`。

回滚：删 `project_key_map.json` + 恢复 `data/bladex_index` 备份。
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import structlog
from bladex_core.project import (
    git_remote_project_id,
    legacy_git_remote_project_id,
    set_project_key_map,
)

from bladex_proxy.ledger_events import PROJECT_KEY_MAP_NAME
from bladex_proxy.project_identity import _git_remote_url

logger = structlog.get_logger()


@dataclass
class ProjectSighting:
    """全库扫出来的一个 (项目键, 根) 组合。"""

    project_id: str
    source: str = ""
    root: str = ""
    name: str = ""
    agent_ids: set[str] = field(default_factory=set)
    turns: int = 0


@dataclass
class MigrationReport:
    dry_run: bool = False
    projects_seen: int = 0
    git_remote_keys: int = 0
    mapped: dict[str, str] = field(default_factory=dict)
    unmapped: list[dict[str, str]] = field(default_factory=list)
    map_changed: bool = False
    fake_projects_deleted: list[str] = field(default_factory=list)
    projects_written: int = 0
    index_project_keys: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def collect_sightings(turns: Iterable[Any]) -> dict[tuple[str, str], ProjectSighting]:
    """轮次流 → {(项目键, 根): 组合}。`turns` 的元素要有 `.identity`。"""
    out: dict[tuple[str, str], ProjectSighting] = {}
    for turn in turns:
        idn = turn.identity
        pid = idn.project_id or ""
        if not pid:
            continue
        key = (pid, idn.project_root or "")
        row = out.setdefault(key, ProjectSighting(
            project_id=pid, source=idn.project_source or "", root=idn.project_root or "",
            name=idn.project_name or ""))
        row.turns += 1
        if idn.agent_id:
            row.agent_ids.add(idn.agent_id)
    return out


def build_key_map(sightings: Iterable[ProjectSighting]) -> tuple[dict[str, str], list[dict[str, str]], int]:
    """(旧键→新键, 对不上的, git-remote 键数)。一个旧键只在正向算得出来时才进表。"""
    rows = [s for s in sightings if s.source == "git-remote"]
    mapping: dict[str, str] = {}
    pending: dict[str, str] = {}
    for s in rows:
        url = _git_remote_url(s.root) if s.root and os.path.isdir(s.root) else ""
        if url and legacy_git_remote_project_id(url) == s.project_id:
            mapping[s.project_id] = git_remote_project_id(url)
        elif url and git_remote_project_id(url) == s.project_id:
            pending.pop(s.project_id, None)           # 已经是现行键（重跑 / 新轮次）
        elif s.project_id not in mapping:
            pending[s.project_id] = ("root_gone" if not url else "remote_changed") + f":{s.root}"
    unmapped = [{"project_id": k, "reason": v} for k, v in sorted(pending.items())
                if k not in mapping]
    return mapping, unmapped, len({s.project_id for s in rows})


def run_migration(turns: Iterable[Any], index: Any, data_dir: str | Path,
                  *, dry_run: bool = False) -> MigrationReport:
    """执行迁移。`index` 须可写（dry_run 时只读）。重跑结果相同。"""
    sightings = collect_sightings(turns)
    rep = MigrationReport(dry_run=dry_run, projects_seen=len({k[0] for k in sightings}))
    new_map, rep.unmapped, rep.git_remote_keys = build_key_map(sightings.values())

    # 表只增不减：上次迁移时目录还在、这次不在了，已确认的对应关系不该丢。
    path = Path(data_dir) / PROJECT_KEY_MAP_NAME
    existing: dict[str, str] = {}
    if path.exists():
        existing = dict(json.loads(path.read_text(encoding="utf-8")).get("map") or {})
    merged = {**existing, **new_map}
    rep.unmapped = [u for u in rep.unmapped if u["project_id"] not in merged]
    rep.mapped = merged
    rep.map_changed = merged != existing

    fakes = [p.project_id for p in index.all_projects()
             if not (p.root_path or "").startswith("/")
             and (p.root_path or "") != ""]
    rep.fake_projects_deleted = fakes

    if dry_run:
        rep.index_project_keys = len(index.all_projects())
        return rep

    if rep.map_changed:
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"map": merged}, ensure_ascii=False, indent=1, sort_keys=True),
                       encoding="utf-8")
        tmp.replace(path)
    set_project_key_map(merged)

    for pid in fakes:
        index.delete_project(pid)
    for s in sorted(sightings.values(), key=lambda r: (r.project_id, r.root)):
        for agent in sorted(s.agent_ids) or [""]:
            if index.note_project_identity(s.project_id, name=s.name, root=s.root, agent_id=agent):
                rep.projects_written += 1
    rep.index_project_keys = len(index.all_projects())
    logger.info("project_key_migration_done", mapped=len(merged), unmapped=len(rep.unmapped),
                fakes_deleted=len(fakes), projects_written=rep.projects_written,
                index_project_keys=rep.index_project_keys)
    return rep
