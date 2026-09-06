"""Flash 四级目录（User → Agent → Project → Session）—— ADR-0032 §5；批一卡 V-F1。

四级目录是**访问路径**，不是存储：账本真身在 User 级 `ledgers/` 池
（`ledger.ledger_md_path`），
本模块的树只放**清单文件（roster）与引用行**。清单文件同样是投影
（`flash.py` 红线 1/2 全部适用：纯函数渲染、无墙上时钟、单写者=Flash 维护进程）。

# 🔴 红线

1. **路径不做项目身份**（ADR-0032 拍板 #5）：项目识别键分层
   显式信号 → git 指纹 → 路径 → Global。反例就在本仓库：`~/dev/BladeX` 与旧软链
   是同一项目的两条路径；worktree 同项目多路径是常态。git 指纹优先于路径，
   路径只作最后一档且必须先 `realpath` 消掉软链。
2. **MATTERS.md 只有骨架**：数据开闸 gate 在账本↔Matter 绑定（V-L5/V-F4，
   ADR-0032 拍板 #7）——08-23 结论"现在的 matter 划分不可信、不能做参数"，
   在绑定落地前把 Index 查询结果推上清单 = 把已知脏数据推上注入面。
3. **无项目概念的 agent 一律 `Global`**：宁可归并到 Global 也不猜——
   猜错的项目归属会把 Session 清单拆散，与 Matter 碎片化同型。

清单文件外壳英文（ADR-0027 §5.1），内容（名称/简介）保持数据原文。
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass

from .flash import SCOPE_PERSONAL, safe_component, scope_dir

# ── 清单文件名 ──────────────────────────────────────────────────────────────

AGENTS_ROSTER = "AGENTS.md"      # User 级：用过的 agent 名册
PROJECTS_ROSTER = "PROJECTS.md"  # Agent 级：进行中的项目
SESSIONS_ROSTER = "SESSIONS.md"  # Project 级：session 列表
PROJECT_MD = "PROJECT.md"        # Project 级：项目简介（蒸馏产物落点，V-F2 填）
LEDGERS_LIST = "LEDGERS.md"      # Session 级：账本引用行（真身在池里）
MATTERS_LIST = "MATTERS.md"      # Session 级：相关 matter（gate 在 V-L5，红线 2）

#: 🔴 `tree/` 中间层已去除（Jason 2026-08-28 拍板）：ADR-0032 §5.1 字面 =
#: **User 级直含** AGENTS.md / 各 agent 子目录 / `ledgers/` 池。原隔离理由
#: （与习惯卡 `agents/` 混层 + 保留名撞名）改由 `_RESERVED_COMPONENTS` 守卫解：
#: agent_id 恰好撞上保留目录名时确定性加前缀；习惯卡体系（ADR-0026 载体）迁入
#: Flash 时按本布局定卡位，不再另起一层。
TREE_DIR = ""

#: scope 级的保留目录/文件名——agent 子目录不得与之同名（撞上 = 树目录吞掉
#: 池目录，静默数据混写）。
_RESERVED_COMPONENTS = frozenset({"ledgers", "agents", "matters", "manual",
                                  "AGENTS.md", "USER.md", "RULES.md"})

PROJECT_GLOBAL_ID = "global"
PROJECT_GLOBAL_NAME = "Global"


# ── 项目识别键（红线 1）─────────────────────────────────────────────────────

SOURCE_EXPLICIT = "explicit"
SOURCE_GIT = "git"
SOURCE_PATH = "path"
SOURCE_GLOBAL = "global"

_GIT_SSH_RE = re.compile(r"^(?:ssh://)?(?:[\w.-]+@)?([\w.-]+)[:/](.+)$")


def normalize_git_remote(url: str) -> str:
    """git remote URL → 规范键。`git@github.com:User/Repo.git`、
    `https://github.com/user/repo/`、`ssh://git@github.com/user/repo.git`
    归一到同一串 `github.com/user/repo`。空/不可解析返回 ""。"""
    s = (url or "").strip()
    if not s:
        return ""
    for scheme in ("https://", "http://", "git://"):
        if s.startswith(scheme):
            s = s[len(scheme):]
            break
    m = _GIT_SSH_RE.match(s)
    if m:
        host, path = m.group(1), m.group(2)
    elif "/" in s:
        host, path = s.split("/", 1)
    else:
        return ""
    path = path.strip("/")
    if path.endswith(".git"):
        path = path[:-4]
    if not host or not path:
        return ""
    return f"{host.lower()}/{path.lower()}"


def _pid(kind: str, key: str) -> str:
    return f"p-{hashlib.sha256(f'{kind}:{key}'.encode()).hexdigest()[:12]}"


@dataclass(frozen=True)
class ProjectIdentity:
    """项目身份。`project_id` 稳定（目录/清单用它关联）；`name` 只是显示。"""

    project_id: str
    name: str
    source: str  # SOURCE_* 之一


def resolve_project_identity(
    *,
    explicit_id: str = "",
    explicit_name: str = "",
    git_remote: str = "",
    git_root: str = "",
    path: str = "",
) -> ProjectIdentity:
    """分层识别：显式 → git 指纹 → 路径 → Global（ADR-0032 拍板 #5）。

    - **显式信号**：调用方从 header/配置拿到的 id，最高优先。
    - **git 指纹**：优先 remote 规范键（worktree/多路径克隆归到同一项目）；
      无 remote 时退 `realpath(git_root)`——本地裸仓的 worktree 无法免费归并，
      这是路径档的固有局限，不硬造（文档写明，靠 remote 或显式信号解）。
    - **路径**：`realpath` 后取键——软链两条路径归到同一项目（红线 1 反例）。
    - **Global**：什么都没有的 agent（Hermes 形态）。宁可归并不猜（红线 3）。

    per-agent 的信号**提取**（从哪段消息拿 cwd/remote）不在本函数——那是
    V-R1 dossier 的产出，接线归 V-F2；本函数只固定判定顺序与键规范。
    """
    if explicit_id:
        return ProjectIdentity(project_id=safe_component(explicit_id),
                               name=explicit_name or explicit_id,
                               source=SOURCE_EXPLICIT)
    remote_key = normalize_git_remote(git_remote)
    if remote_key:
        name = explicit_name or remote_key.rsplit("/", 1)[-1]
        return ProjectIdentity(project_id=_pid("git", remote_key), name=name,
                               source=SOURCE_GIT)
    if git_root:
        real = os.path.realpath(git_root)
        return ProjectIdentity(project_id=_pid("git-root", real),
                               name=explicit_name or os.path.basename(real) or real,
                               source=SOURCE_GIT)
    if path:
        real = os.path.realpath(path)
        return ProjectIdentity(project_id=_pid("path", real),
                               name=explicit_name or os.path.basename(real) or real,
                               source=SOURCE_PATH)
    return ProjectIdentity(project_id=PROJECT_GLOBAL_ID, name=PROJECT_GLOBAL_NAME,
                           source=SOURCE_GLOBAL)


# ── 树路径（全部从 scope 目录长出来，敏感度/多租户纪律免费继承）─────────────


def tree_dir(root: str, principal: str, *, scope: str = SCOPE_PERSONAL) -> str:
    """树根 = scope 级本身（`tree/` 层已去除，见 TREE_DIR 注记）。"""
    return scope_dir(root, principal, scope)


def agents_roster_path(root: str, principal: str, *, scope: str = SCOPE_PERSONAL) -> str:
    return f"{tree_dir(root, principal, scope=scope)}/{AGENTS_ROSTER}"


def agent_dir(root: str, principal: str, agent_id: str,
              *, scope: str = SCOPE_PERSONAL) -> str:
    comp = safe_component(agent_id)
    if comp in _RESERVED_COMPONENTS:
        comp = f"a-{comp}"      # 保留名守卫：确定性前缀，重建等价
    return f"{tree_dir(root, principal, scope=scope)}/{comp}"


def projects_roster_path(root: str, principal: str, agent_id: str,
                         *, scope: str = SCOPE_PERSONAL) -> str:
    return f"{agent_dir(root, principal, agent_id, scope=scope)}/{PROJECTS_ROSTER}"


def project_dir(root: str, principal: str, agent_id: str, project_id: str,
                *, scope: str = SCOPE_PERSONAL) -> str:
    """项目目录名用 `project_id`（不是显示名）：显示名会改、会撞、会含路径分隔符，
    id 才是身份（ADR §5.1"以项目名称为目录名"经 V-F1 修正为 id——名称进 PROJECTS.md）。"""
    return f"{agent_dir(root, principal, agent_id, scope=scope)}/{safe_component(project_id)}"


def sessions_roster_path(root: str, principal: str, agent_id: str, project_id: str,
                         *, scope: str = SCOPE_PERSONAL) -> str:
    return f"{project_dir(root, principal, agent_id, project_id, scope=scope)}/{SESSIONS_ROSTER}"


def project_md_path(root: str, principal: str, agent_id: str, project_id: str,
                    *, scope: str = SCOPE_PERSONAL) -> str:
    return f"{project_dir(root, principal, agent_id, project_id, scope=scope)}/{PROJECT_MD}"


def session_dir(root: str, principal: str, agent_id: str, project_id: str,
                session_id: str, *, scope: str = SCOPE_PERSONAL) -> str:
    return (f"{project_dir(root, principal, agent_id, project_id, scope=scope)}/"
            f"{safe_component(session_id)}")


def ledgers_list_path(root: str, principal: str, agent_id: str, project_id: str,
                      session_id: str, *, scope: str = SCOPE_PERSONAL) -> str:
    return (f"{session_dir(root, principal, agent_id, project_id, session_id, scope=scope)}/"
            f"{LEDGERS_LIST}")


def matters_list_path(root: str, principal: str, agent_id: str, project_id: str,
                      session_id: str, *, scope: str = SCOPE_PERSONAL) -> str:
    return (f"{session_dir(root, principal, agent_id, project_id, session_id, scope=scope)}/"
            f"{MATTERS_LIST}")


def ledger_pool_relpath(ledger_id: str) -> str:
    """Session 目录里的引用行 → 池内真身的相对路径。
    树深固定（<agent>/<project>/<session>/ = scope 下三层，tree/ 层已去除），
    池内固定两级分桶（`ledger.ledger_shard`），故相对路径确定：
    `../../../ledgers/<b1>/<b2>/<id>.md`。相对而非绝对：整棵 flash-root 可搬移/备份。"""
    from bladex_core.ledger import ledger_shard
    return f"../../../ledgers/{ledger_shard(ledger_id)}/{safe_component(ledger_id)}.md"


# ── 清单行 schema 与渲染（纯函数、无墙上时钟）───────────────────────────────

_MACHINE = "<!-- BladeX Memory Flash · machine-rendered roster, rewritten in full -->"


@dataclass(frozen=True)
class AgentRow:
    agent_id: str
    name: str = ""
    summary: str = ""        # 蒸馏 AGENT.md/CLAUDE.md 所得（V-F2 填）
    first_seen: str = ""
    last_seen: str = ""


@dataclass(frozen=True)
class ProjectRow:
    project_id: str
    name: str = ""
    summary: str = ""        # 蒸馏 PROJECT.md 所得（V-F2 填）
    source: str = ""         # ProjectIdentity.source——识别档位可审计
    created_at: str = ""
    updated_at: str = ""


@dataclass(frozen=True)
class SessionRow:
    session_id: str
    name: str = ""
    first_at: str = ""
    last_at: str = ""


@dataclass(frozen=True)
class LedgerRow:
    ledger_id: str
    title: str = ""
    status: str = "active"
    created_at: str = ""
    updated_at: str = ""


@dataclass(frozen=True)
class MatterRow:
    matter_id: str
    title: str = ""
    created_at: str = ""
    updated_at: str = ""


def _cell(s: str) -> str:
    return (s or "").replace("|", "\\|").replace("\n", " ").strip() or "—"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| # | " + " | ".join(header) + " |",
             "|---|" + "|".join("---" for _ in header) + "|"]
    lines += [f"| {i} | " + " | ".join(_cell(c) for c in row) + " |"
              for i, row in enumerate(rows, 1)]
    return lines


def render_agents_roster(rows: list[AgentRow]) -> str:
    body = _table(["agent_id", "name", "about", "first seen", "last seen"],
                  [[f"`{r.agent_id}`", r.name, r.summary, r.first_seen, r.last_seen]
                   for r in rows]) if rows else ["_(no agents yet)_"]
    return "\n".join(["# Agents", "", _MACHINE, "", *body]) + "\n"


def render_projects_roster(agent_id: str, rows: list[ProjectRow]) -> str:
    body = _table(["project_id", "name", "about", "created", "updated"],
                  [[f"`{r.project_id}`", r.name, r.summary, r.created_at, r.updated_at]
                   for r in rows]) if rows else ["_(no projects yet)_"]
    return "\n".join([f"# Projects · {agent_id}", "", _MACHINE, "", *body]) + "\n"


def render_sessions_roster(project_name: str, rows: list[SessionRow]) -> str:
    body = _table(["session_id", "name", "first turn", "last turn"],
                  [[f"`{r.session_id}`", r.name, r.first_at, r.last_at]
                   for r in rows]) if rows else ["_(no sessions yet)_"]
    return "\n".join([f"# Sessions · {project_name}", "", _MACHINE, "", *body]) + "\n"


def render_ledgers_list(rows: list[LedgerRow], *, active_ledger_id: str = "") -> str:
    """Session 级账本清单：**引用行**，真身在池里（ADR-0032 §4.2）。
    激活账本标 `(active)`——每会话绑一本（拍板集 §4.3）。"""
    def _title(r: LedgerRow) -> str:
        mark = " **(active)**" if r.ledger_id == active_ledger_id else ""
        return f"{r.title or r.ledger_id}{mark}"
    body = _table(["ledger", "title", "status", "created", "updated"],
                  [[f"[`{r.ledger_id}`]({ledger_pool_relpath(r.ledger_id)})",
                    _title(r), r.status, r.created_at, r.updated_at]
                   for r in rows]) if rows else ["_(no ledgers yet)_"]
    return "\n".join(["# Ledgers", "", _MACHINE, "", *body]) + "\n"


#: 红线 2 的可 grep 形态：绑定未落地时 MATTERS.md 只渲染这句，不渲染数据。
MATTERS_GATED_MARK = "_(matter listing is gated until ledger–matter binding lands)_"


def render_matters_list(rows: list[MatterRow], *, binding_live: bool = False) -> str:
    """Session 级相关 matter 清单。`binding_live=False`（默认）= 只出骨架（红线 2）；
    开闸动作属 V-F4，且调用方必须以账本↔Matter 绑定落地为前提。"""
    if not binding_live:
        body = [MATTERS_GATED_MARK]
    elif rows:
        body = _table(["matter", "title", "created", "updated"],
                      [[f"`{r.matter_id}`", r.title, r.created_at, r.updated_at]
                       for r in rows])
    else:
        body = ["_(no related matters)_"]
    return "\n".join(["# Related matters", "", _MACHINE, "", *body]) + "\n"
