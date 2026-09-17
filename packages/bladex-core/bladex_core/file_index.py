"""文件内容索引（ADR-0028 E7.1）—— 取代裸路径 file_ref 的**正确形态**。

## 为什么裸路径不行

E1.3 把 file_ref 从向量检索平面赶了出去，理由是实测：
LanceDB 270 行里 129 行是 `文件 /Users/alice/...` 这种裸路径，
30 条随机 query 的 top-10 平均 **20.7%** 被它们占据，相似度 mean 0.880 ——
"对任何问句都中等相似"的典型形状。裸路径没有语义，只有噪声。

但"文件"本身是要能检索的——用户会问「那个 md 里写的什么来着」。
正确形态不是把路径塞进向量空间，而是**索引文件的内容**：
名字 + 类型 + 内容摘要 + 关键词，向量建在这些**有语义**的东西上。

## 边界

- 来源：**读写类**工具（Read/Write/Edit 白名单，与 E1.5 的检索类黑名单互补）
  的 result 正文，≥200 字符才值得索引；
- 音视频/图片 v1 只录元数据（name/ext），不做转写；
- 独立通道 + 配额 1（E6.3 ④）：文件不挤占散点召回位；
- 同 path 新 hash → 覆盖更新（files 表非记忆总账，无 as-of 义务，历史在 Memory Hub）。
"""

from __future__ import annotations

import hashlib
import os
import re
from enum import Enum

from pydantic import BaseModel, Field

# 值得索引的最小正文长度（太短的读取结果没有摘要价值）
MIN_INDEX_CHARS = 200

# 读写类工具白名单（E1.5 检索类黑名单的反向）。
# 🔴 2026-08-05 真实 Memory Hub 复核修正：初版只按"通用命名"猜，实测**一次都不会命中**——
# 库里真实的读写工具是 `read_file`（Codex，324 次调用）与 `Read`（Claude Code，90 次），
# 两个都不在初版白名单里。这正是本 ADR 一路在修的那类缺陷（机制写完了但从不触发），
# 所以这里按**实测到的工具名**补全，而不是继续按想象补。
READ_WRITE_TOOLS: frozenset[str] = frozenset({
    # Claude Code
    "read", "write", "edit", "notebookedit",
    # Codex / OpenAI 工具族
    "read_file", "write_file", "edit_file", "apply_patch", "create_file",
    "str_replace_editor", "str_replace_based_edit_tool",
    # 连接器 / MCP 常见命名
    "read_file_content", "download_file_content",
})

# 文件索引的路径参数键。
# 与 `_extract_file_refs` 的 E1.5 白名单**故意不同**：这里额外收 `path`。
# E1.5 之所以删掉 `path`，是因为 `search_files{path:"/Users/alice"}` 把搜索根目录
# 当成了"被操作的文件"。而本通道已经先过了 READ_WRITE_TOOLS 白名单——
# 检索类工具根本进不来，`path` 在这里就是"读/写的那个文件"（实测 `read_file` 用的
# 正是 `path`，498 次）。两处规则分开，各自的理由都成立。
PATH_ARG_KEYS: tuple[str, ...] = (
    "file_path", "target_file", "filename", "notebook_path", "path",
)

_EXT_MIME: dict[str, str] = {
    ".md": "doc", ".txt": "doc", ".pdf": "doc", ".docx": "doc", ".rst": "doc",
    ".py": "code", ".ts": "code", ".tsx": "code", ".js": "code", ".go": "code",
    ".rs": "code", ".java": "code", ".sh": "code", ".sql": "code",
    ".toml": "code", ".yaml": "code", ".yml": "code", ".json": "code",
    ".png": "image", ".jpg": "image", ".jpeg": "image", ".gif": "image",
    ".svg": "image", ".webp": "image",
    ".mp3": "av", ".mp4": "av", ".wav": "av", ".mov": "av", ".m4a": "av",
}

MAX_SUMMARY_CHARS = 300
MAX_KEYWORDS = 8


class FileOrigin(str, Enum):
    """文件条目的来源——**两类，索引标记必须不同**（2026-08-05 拍板）。

    第一类「会话内容提到的文件」：
      MENTIONED  只在会话里出现了文件名/路径，我们从没拿到正文（≈ 旧的裸路径 file_ref）
      READ       工具（Read/read_file…）读到了正文 —— 建摘要索引，**不存副本**

    第二类「LLM 返回内容中包含的文件」：
      PRODUCED   模型在回复正文里**产出**的文件（带文件名标注的代码块 / 文档正文）
                 —— 建摘要索引 **并把正文按 hash 落盘存副本**，索引关联到该副本。

    为什么只有 PRODUCED 落盘：READ 的正文在磁盘上本来就有（原文件还在那儿），
    再存一份是冗余；而模型产出的那份**只存在于那一次回复里**，不落盘就永远找不回来了。
    """

    MENTIONED = "mentioned"
    READ = "read"
    PRODUCED = "produced"

# query 里的"文件意图"词表（命中才启用文件通道，E7.1 检索条件）
FILE_INTENT_WORDS: frozenset[str] = frozenset({
    "文件", "文档", "路径", "目录", "脚本", "配置",
    "file", "files", "doc", "docs", "path", "script", "config", "readme",
})


def mime_class_of(path: str) -> str:
    return _EXT_MIME.get(os.path.splitext(path or "")[1].lower(), "other")


def file_id_for(path: str) -> str:
    """file_id = sha256(path)[:16]（同一文件跨轮稳定，覆盖更新的键）。"""
    return hashlib.sha256((path or "").encode()).hexdigest()[:16]


class FileEntry(BaseModel):
    """一条文件索引（ADR-0028 E7.1 schema）。"""

    file_id: str
    path: str
    name: str = ""
    ext: str = ""
    mime_class: str = "other"      # doc | code | image | av | other
    summary: str = ""              # ≤300 字符，LLM 蒸馏
    keywords: list[str] = Field(default_factory=list)   # ≤8
    content_hash: str = ""
    mtime: str = ""
    agent_id: str = ""
    user_id: str = ""
    # 两类来源的索引标记（见 FileOrigin）
    origin: FileOrigin = FileOrigin.READ
    # PRODUCED 专有：本地内容寻址副本的 id（sha256 全长）。空 = 没有副本。
    blob_ref: str = ""
    vector: list[float] | None = None   # embed(name + " " + summary + " " + keywords)

    def embed_text(self) -> str:
        """向量文本 = 名字 + 摘要 + 关键词（**不含路径**——路径正是噪声源）。"""
        return " ".join(x for x in [self.name, self.summary, " ".join(self.keywords)] if x)


def make_entry(
    path: str, content: str, *, agent_id: str = "", user_id: str = "",
    summary: str = "", keywords: list[str] | None = None, mtime: str = "",
    origin: FileOrigin = FileOrigin.READ, blob_ref: str = "",
) -> FileEntry:
    """从一次文件读写构造索引条目（确定性部分；summary/keywords 由蒸馏填）。"""
    name = os.path.basename((path or "").replace("\\", "/"))
    return FileEntry(
        file_id=file_id_for(path),
        path=path,
        name=name,
        ext=os.path.splitext(name)[1].lower(),
        mime_class=mime_class_of(path),
        summary=(summary or "")[:MAX_SUMMARY_CHARS],
        keywords=list(keywords or [])[:MAX_KEYWORDS],
        content_hash=hashlib.sha256((content or "").encode()).hexdigest()[:16],
        mtime=mtime,
        agent_id=agent_id,
        user_id=user_id,
        origin=origin,
        blob_ref=blob_ref,
    )


def worth_indexing(tool_name: str, path: str, content: str) -> bool:
    """是否值得建内容索引（E7.1 准入）。

    - 工具必须是读写类（检索工具的路径参数是"在哪找"，不是"操作了什么"）；
    - 图片/音视频只录元数据，正文长度不作要求；
    - 其余类型正文 ≥200 字符才索引。
    """
    if (tool_name or "").strip().lower() not in READ_WRITE_TOOLS:
        return False
    if not path:
        return False
    if mime_class_of(path) in ("image", "av"):
        return True
    return len(content or "") >= MIN_INDEX_CHARS


def query_wants_files(query: str, identifiers: list[str] | None = None) -> bool:
    """query 是否带"文件意图"（E7.1：命中才并入 RRF 一路，配额 1）。

    判据：识别出带扩展名的标识符，或命中文件意图词表。
    """
    for ident in identifiers or []:
        if os.path.splitext(ident)[1] and len(os.path.splitext(ident)[1]) <= 9:
            return True
    low = (query or "").lower()
    return any(w in low for w in FILE_INTENT_WORDS)


# ── 第二类：从 LLM 回复正文里抽出"模型产出的文件"（确定性，零 LLM）──
#
# 真实形态是围栏代码块，文件名以三种方式之一给出：
#   ```python:packages/bladex_core/fusion.py     ← 语言:路径
#   ```packages/bladex_core/fusion.py            ← 直接路径当 info string
#   **packages/bladex_core/fusion.py**           ← 紧邻上一行的加粗/标题/"文件："标注
#   ```python
# 只认**带得出文件名**的块：认不出文件名的代码片段不是"一个文件"，不该进文件索引
# （宁漏勿误——这条通道的价值在于"那份产出还能找回来"，不在于收集所有代码片段）。

_FENCE_RE = re.compile(r"^[ \t]*```([^\n`]*)\n(.*?)^[ \t]*```[ \t]*$", re.S | re.M)
#: 紧邻代码块之前的一行里给出的文件名（`**x.py**` / `### x.py` / `文件：x.py` / `File: x.py`）
_NEARBY_NAME_RE = re.compile(
    r"(?:文件|檔案|路径|File|Path)\s*[:：]\s*([^\s`*]+\.[A-Za-z0-9]{1,8})"
    r"|\*\*([^\s*]+\.[A-Za-z0-9]{1,8})\*\*"
    r"|^#{1,6}\s+([^\s#]+\.[A-Za-z0-9]{1,8})\s*$"
    r"|`([^`\s]+\.[A-Za-z0-9]{1,8})`",
    re.M,
)
_INFO_PATH_RE = re.compile(r"([\w./\-]+\.[A-Za-z0-9]{1,8})$")

#: 单个产出文件的正文上限（超出截断——副本是给人回溯的，不是备份系统）
MAX_PRODUCED_CHARS = 200_000
#: 太短的块不当"文件"（一两行示例不值得建索引 + 存副本）
MIN_PRODUCED_CHARS = 80


def extract_produced_files(text: str) -> list[tuple[str, str]]:
    """从 assistant 回复正文里抽出 [(路径, 正文)]。

    纯函数、确定性、零 LLM —— 与 envelope/task_unit 同性质，重建等价性成立。
    同一路径在一段回复里出现多次时取**最后一次**（模型改过的那版才是最终产物）。
    """
    if not text:
        return []
    out: dict[str, str] = {}
    for m in _FENCE_RE.finditer(text):
        info = (m.group(1) or "").strip()
        body = m.group(2) or ""
        if len(body.strip()) < MIN_PRODUCED_CHARS:
            continue

        path = ""
        # ① info string 里带路径（`python:a/b.py` 或直接 `a/b.py`）
        cand = info.split(":", 1)[-1].strip() if ":" in info else info
        hit = _INFO_PATH_RE.search(cand)
        if hit:
            path = hit.group(1)
        # ② 紧邻上一行的标注
        if not path:
            head = text[:m.start()]
            prev = head.rsplit("\n", 3)[-3:]          # 往上看三行，够覆盖"标题+空行"
            nm = None
            for line in reversed(prev):
                nm = _NEARBY_NAME_RE.search(line)
                if nm:
                    break
            if nm:
                path = next(g for g in nm.groups() if g)
        if not path:
            continue        # 认不出文件名 → 不是"一个文件"，跳过
        out[path] = body.strip()[:MAX_PRODUCED_CHARS]
    return list(out.items())
