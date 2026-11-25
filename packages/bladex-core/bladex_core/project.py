"""Project 实体（ADR-0028 E7.2）—— 主题轴的最上层：Project → Matter → TaskUnit。

ADR-0026 定的三轴坐标里，主题轴是 `Project → Matter → TaskUnit`。
TaskUnit（ADR-0024）和 Matter（ADR-0012）都已落地，**Project 一直是空的**——
于是"这是哪个项目的事"这个最粗粒度的定位，库里根本没有。

本模块给 Project 的模型与**确定性识别**（零 LLM，识别不出就不识别）：

    ① 规则文件副本路径所在目录 = root_path（最强信号：CLAUDE.md 在哪，项目就在哪）
    ② 同 session 内 tool 参数里高频出现的公共路径前缀（≥5 次且深度 ≥2）
    ③ 手动 `bladex project add`

字段填充也全是确定性的：languages = 文件扩展名统计 top3；
description/progress_note = 归属到该 Project 的 Matter 摘要**拼接**（不 LLM）。
"""

from __future__ import annotations

import hashlib
import os
from collections import Counter
from datetime import UTC, datetime

from pydantic import BaseModel, Field

# 识别②的门槛（任务卡 E7.2 写死）
MIN_PATH_HITS = 5
MIN_PATH_DEPTH = 2

# 扩展名 → 语言（判别性强的固定表；认不出就不算，宁漏勿误）
_EXT_LANG: dict[str, str] = {
    ".py": "Python", ".pyi": "Python",
    ".ts": "TypeScript", ".tsx": "TypeScript",
    ".js": "JavaScript", ".jsx": "JavaScript",
    ".go": "Go", ".rs": "Rust", ".java": "Java", ".kt": "Kotlin",
    ".rb": "Ruby", ".php": "PHP", ".swift": "Swift", ".c": "C",
    ".cc": "C++", ".cpp": "C++", ".h": "C", ".hpp": "C++",
    ".sh": "Shell", ".sql": "SQL", ".md": "Markdown", ".toml": "TOML",
    ".yaml": "YAML", ".yml": "YAML",
}

MAX_DESCRIPTION_CHARS = 200
MAX_PROGRESS_CHARS = 200


def normalize_git_remote(url: str) -> str:
    """git remote URL 归一（批 BA2，拍板 (32)-b，**唯一实现点**）：同一个仓库的不同写法归到一个串。

    规则 = scheme 归一 + host 小写 + `.git` 剥除 + 去尾 `/`。形状统一成 `host[:port]/path`：
    `https://GitHub.com/a/B.git/`、`ssh://git@github.com/a/B`、`git@github.com:a/B.git`
    ⇒ `github.com/a/B`。**凭证不进键**（`user:token@` 整段丢）。路径大小写保留——
    不是所有服务器都大小写不敏感，合错仓库比拆开更贵。
    本地路径 remote（`/srv/x.git`、`file:///srv/x.git`）没有 host，只去 `.git` 与尾 `/`。
    """
    raw = (url or "").strip()
    if not raw:
        return ""
    if "://" in raw:
        scheme, _, rest = raw.partition("://")
        if scheme.lower() == "file":
            host, path = "", rest
        else:
            authority, _, path = rest.partition("/")
            host = authority.rpartition("@")[2].lower()
    elif raw.startswith(("/", ".", "~")) or ":" not in raw.partition("/")[0]:
        host, path = "", raw                  # 本地路径
    else:
        authority, _, path = raw.partition(":")   # scp 形：[user@]host:path
        host = authority.rpartition("@")[2].lower()
    path = path.rstrip("/")
    if path.endswith(".git"):
        path = path[:-4].rstrip("/")
    if not host:
        return path
    return f"{host}/{path.lstrip('/')}"


def git_remote_project_id(url: str) -> str:
    """git-remote 档的项目键（拍板 (32)-b）：`p-` + sha256(`git-remote:{归一}`)[:16]。"""
    key = f"git-remote:{normalize_git_remote(url)}"
    return "p-" + hashlib.sha256(key.encode()).hexdigest()[:16]


def legacy_git_remote_project_id(url: str) -> str:
    """批 BA2 之前的 git-remote 键（原始 URL、12 位）。**只给迁移对账用**：Hub 里的老事件存的
    是这个哈希，哈希不可逆，只能从 remote 原文正向算一遍来确认「这个旧键确实来自这个仓库」。"""
    return "p-" + hashlib.sha256(f"git-remote:{url}".encode()).hexdigest()[:12]


def project_id_for(root_path: str) -> str:
    """磁盘上已认不出来的根目录的项目键：与 `resolve_project` 的 git-root 档**同一公式**。

    批 BA2（MQ-A75 甲）：此前是 sha256(root_path)[:12]，与账本 scope 的 `identity.project_id`
    从不相交。能走磁盘的一律先走 `resolve_project`（见 `MemoryIndex._project_key_for_root`），
    这里只兜目录已不在的情形——老 bench 目录都是 git-root 档，键因此仍与存量对得上。
    """
    return "p-" + hashlib.sha256(f"git-root:{root_path or ''}".encode()).hexdigest()[:12]


#: 旧项目键 → 现行键（批 BA2 一次性迁移的产物，派生层 `data/project_key_map.json`）。
#: 老事件 / 老轮次存的是旧键的哈希，Hub 不改写，读的时候在这里翻译。空表 = 翻译不生效。
_project_key_map: dict[str, str] = {}


def set_project_key_map(mapping: dict[str, str]) -> None:
    _project_key_map.clear()
    _project_key_map.update(mapping)


def canonical_project_id(project_id: str) -> str:
    """旧键 → 现行键；不在表里的原样返回。**读路径的唯一翻译点**。"""
    return _project_key_map.get(project_id, project_id)


class Project(BaseModel):
    """一个项目（ADR-0028 E7.2）。

    与 Matter 的关系：Matter 是"周级可完结的一件事"，Project 是它们的容器
    （`Matter --PART_OF--> Project`）。Project 不会"办完"，Matter 会。
    """

    project_id: str
    name: str = ""
    root_path: str = ""
    description: str = ""
    languages: list[str] = Field(default_factory=list)
    progress_note: str = ""
    agent_ids: list[str] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    version: int = 0
    origin: str = "auto"      # auto | manual


def infer_root_from_rulefile(path: str) -> str:
    """识别①：规则文件副本路径的目录 = root_path。"""
    if not path:
        return ""
    d = os.path.dirname(path.replace("\\", "/").rstrip("/"))
    return d or ""


def infer_root_from_paths(paths: list[str]) -> str:
    """识别②：tool 参数里高频出现的公共路径前缀。

    条件（任务卡写死）：同 session 内出现 **≥5 次** 且 **深度 ≥2**。
    取满足条件的**最长**前缀（最长 = 最具体，避免所有项目都归到 `/Users`）。
    """
    counter: Counter[str] = Counter()
    for raw in paths or []:
        p = (raw or "").replace("\\", "/")
        if not p.startswith("/"):
            continue
        parts = [x for x in p.split("/") if x]
        # 去掉文件名那一段（带扩展名的尾段）
        if parts and "." in parts[-1]:
            parts = parts[:-1]
        for depth in range(MIN_PATH_DEPTH, len(parts) + 1):
            counter["/" + "/".join(parts[:depth])] += 1
    best = ""
    for prefix, hits in counter.items():
        if hits < MIN_PATH_HITS:
            continue
        if len(prefix) > len(best):
            best = prefix
    return best


def languages_of(paths: list[str], top: int = 3) -> list[str]:
    """languages = 文件扩展名统计 top3（确定性）。"""
    c: Counter[str] = Counter()
    for p in paths or []:
        lang = _EXT_LANG.get(os.path.splitext(p or "")[1].lower())
        if lang:
            c[lang] += 1
    return [lang for lang, _ in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:top]]


def name_from_root(root_path: str) -> str:
    """name = 目录名（可手动改）。"""
    return os.path.basename((root_path or "").replace("\\", "/").rstrip("/")) or root_path


def compose_description(matter_titles: list[str]) -> str:
    """description = 归属 Matter 标题的确定性拼接（不 LLM）。"""
    seen: list[str] = []
    for t in matter_titles:
        t = (t or "").strip()
        if t and t not in seen:
            seen.append(t)
    return " / ".join(seen)[:MAX_DESCRIPTION_CHARS]


def compose_progress(matter_summaries: list[str]) -> str:
    """progress_note = 归属 Matter 摘要的确定性拼接（不 LLM）。"""
    parts = [s.strip() for s in matter_summaries if (s or "").strip()]
    return " | ".join(parts)[:MAX_PROGRESS_CHARS]


def matter_belongs_to_project(member_paths: list[str], root_path: str,
                              *, min_ratio: float = 0.5) -> bool:
    """判定 `Matter --PART_OF--> Project`：

    Matter 成员 fact 的 file_ref/路径落在 root_path 下的比例 ≥ 0.5（任务卡写死）。
    """
    if not member_paths or not root_path:
        return False
    root = root_path.replace("\\", "/").rstrip("/")
    hit = sum(1 for p in member_paths
              if (p or "").replace("\\", "/").startswith(root + "/"))
    return hit / len(member_paths) >= min_ratio


# （`render_project_card`——①平面 Project 卡——已于 2026-09-18 批 S 删除：注入面 09-03 起
#  不消费 Project 卡，零生产调用点；拍板 0.3.0-(5)，ADR-0034 §3.2。`Project` 对象保留。）
