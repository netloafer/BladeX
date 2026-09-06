"""项目识别第一档（ADR-0032 §4：键分层 显式 → git 指纹 → 路径 → Global）。

2026-08-29 立（Jason 拍板），实证基础：
- codex 每会话首轮 `<environment_context>` 带 `<cwd>…</cwd>`（1280 轮实测）；
- claude-code env 块带 "working directory" 行（**写法因构建而异**：实测是
  ` - Primary working directory: …`，不是立卡时记的 `Working directory:`——
  2026-08-29 迁移批 live 复核修正，见 `_CWD_PATTERNS` 注释）；
- 🔴 路径不做项目身份的最硬实证：本仓库经软链有权威路径与 Desktop 兼容旧
  软链两个入口（刚性原则 11），live cwd 720/560 分裂——按路径当键一个项目
  裂成两个；git 指纹（realpath + 仓库锚）天然收敛。
- codex 项目可配多目录（451 轮 filesystem 块两 write root 同现）：
  项目键只取**包含 cwd 的仓库**，其余 roots 是附属目录不参与身份。

键的稳定序：显式声明（`x-codex-turn-metadata` 若证实带 project 字段，
MQ-A22 修复后重启可得全样本——当前占位）> git remote URL（跨搬家/改名/
多克隆稳定）> git 仓库根 realpath（跨软链稳定）> 路径档（realpath +
项目声明标志，无 git 时的落点）> Global（空 id）。

热路径纪律：cwd 提取 = 前几条消息（头部 + `<environment_context>` 标签块，
MQ-A25）的一次正则；解析结果按 cwd LRU 缓存，
文件系统访问（realpath / .git 发现 / config 读取）每 cwd 只发生一次。
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass

import structlog

logger = structlog.get_logger()

#: cwd 标记（判据取各 agent 的 wire 原文，2026-08-29 live 实证）：
#: codex CLI = `<cwd>…</cwd>`；claude-code = "working directory" 行；
#: codex desktop = `# AGENTS.md instructions for <路径>` 头（无 env_ctx）。
#:
#: 🔴 2026-08-29 修订（迁移批 live 复核）：claude-code 那条原写死
#: `Working directory:`（大写 W、无前缀、大小写敏感），而实测 wire 原文是
#:     ` - Primary working directory: /Users/alice/dev/BladeX`
#: （小写 w + `Primary ` 前缀 + 行首 `- ` 项目符号，旁边一行也是
#: `Is a git repository: true` 而非记录里的写法 ⇒ 另一个 CC 构建）。
#: 后果：claude-code 的 `project_id` **恒空**，全部落 Global，惰性作用域迁移
#: （MQ-L30）的第一道门 `if not project_id` 就返回，重启后不触发。
#: 立卡时那条"claude-code env 块带 `Working directory:` 行"的实证记录**记错了
#: 形态**——教训同「卡面上的状态会过期」：判据要落在能复算的 wire 原文上。
#: 修法只放宽**写法**（大小写 / 前缀词 / 行首项目符号），扫描面一个字节不动；
#: 同时补 `^` 锚（原来无锚，`... the Working directory: x` 这种散文中段也会中，
#: 收紧是顺带减假阳性）。
_CWD_PATTERNS = (
    re.compile(r"<cwd>([^<\n]+)</cwd>"),
    re.compile(r"^[ \t\-*>]*(?:primary\s+)?working\s+directory\s*:\s*(\S.*?)\s*$",
               re.IGNORECASE | re.MULTILINE),
    # codex desktop 形态（2026-08-29 live 实证）：无 <environment_context>，
    # 路径载体是 AGENTS.md 指令头（fp:6103d775f46f「在线吗」轮）。
    re.compile(r"^# AGENTS\.md instructions for (\S+)", re.MULTILINE),
)
#: 只看前几条消息的头部——env 块永远在会话开头（越界扫描是热路径浪费）。
_SCAN_MESSAGES = 3
_SCAN_CHARS = 8192

#: 🔴 标签定位（MQ-A25，2026-09-02 Jason 拍板）：codex 的 `<cwd>` 在 **user**
#: 消息 @11890（12 天 3754 轮读数 727/727 落在 `_SCAN_CHARS` 之外，
#: `docs/benchmarks/project-scan-window-20260902.txt`），此前识别全靠同一消息
#: offset 0 的 AGENTS.md 头"撞上"——换一个没有 AGENTS.md 的仓库就掉 Global。
#: 修法不是放大 `_SCAN_CHARS`（同一份读数里 tool 结果转引我们自己的文档时
#: 三个串同现，放大窗口就是往中毒源上扫），而是：前 `_SCAN_MESSAGES` 条消息里
#: 若存在 agent **自己的确定性标签块** `<environment_context>…</environment_context>`，
#: 对该块整段跑 `_CWD_PATTERNS`、不限 offset；块之外仍只扫头部 8192。
#: 块优先于头部：标签块是 agent 声明的环境，AGENTS.md 头只是指令载体。
#: 被否的"按角色分档（system/developer 全文扫）"：那两个角色里零非 env 命中
#: （中毒风险 0），但 codex 的 cwd 根本不在那两个角色里——够不着，不是中毒。
_ENV_BLOCK = re.compile(r"<environment_context>(.*?)</environment_context>", re.S)

#: 解析缓存（cwd -> ProjectIdentity）。进程生命期内 cwd 集合极小，不设逐出。
_cache: dict[str, "ProjectIdentity"] = {}


@dataclass(frozen=True)
class ProjectIdentity:
    """一次项目识别的结果。`project_id` 空串 = Global（无项目概念，ADR-0032）。

    :param project_id: `p-<sha12>`，键派生见 `resolve_project`；空 = Global。
    :param name: 人读名（仓库根目录 basename），展示用，不参与身份。
    :param source: 识别档位（可审计）：`git-remote` / `git-root` / `path` /
        `` (Global)。
    """

    project_id: str = ""
    name: str = ""
    source: str = ""
    root: str = ""       # 项目根 realpath（简介蒸馏读声明文件用；Global 为空）


def _text_of(content) -> str:
    """消息 content 的文本视图：str 原样；**分块列表**（Responses/Anthropic
    形态 `[{"type":"text","text":…}]`）取各块 text 拼接——识别跑在入站原始
    形态上，只认 str 会整条跳过 codex desktop / Pi 的消息（2026-08-29 live
    实证：pattern 全对、content 形态不对，project 恒空）。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(x.get("text", "")) for x in content
                          if isinstance(x, dict))
    return ""


def _first_match(text: str) -> str:
    """对一段文本按 `_CWD_PATTERNS` 顺序取第一个命中；无命中返回空串。"""
    for pat in _CWD_PATTERNS:
        mm = pat.search(text)
        if mm:
            return mm.group(1).strip()
    return ""


def extract_cwd(messages: list[dict]) -> str:
    """从请求消息里取 cwd 标记（确定性，零 LLM）。取不到返回空串。

    扫描面 = 前 `_SCAN_MESSAGES` 条非空消息 × （`<environment_context>` 标签块
    整段 ∪ 头部 `_SCAN_CHARS`）——见 `_ENV_BLOCK` 注释（MQ-A25）。
    `_SCAN_MESSAGES` 一个字节不动：tool 结果转引文档那类中毒源全在前 3 条之外。
    """
    scanned = 0
    for m in messages:
        if scanned >= _SCAN_MESSAGES:
            break
        content = _text_of(m.get("content"))
        if not content:
            continue
        scanned += 1
        block = _ENV_BLOCK.search(content)
        if block:
            found = _first_match(block.group(1))
            if found:
                return found
        found = _first_match(content[:_SCAN_CHARS])
        if found:
            return found
    return ""


def explicit_project_id(headers: dict[str, str]) -> str:
    """显式层占位（键分层第一档）。

    `x-codex-turn-metadata` 疑似还有未见字段（此前被 200 字符截断，MQ-A22
    已修）——全样本证实带 project/workspace 字段后在这里接入。
    在那之前恒返回空串：**占位不猜测**。
    """
    _ = headers
    return ""


def _git_repo_root(path: str) -> str:
    """从 path 向上找 `.git`（目录或 worktree 文件）。找不到返回空串。"""
    cur = path
    for _ in range(12):
        if os.path.exists(os.path.join(cur, ".git")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return ""
        cur = parent
    return ""


def _git_remote_url(repo_root: str) -> str:
    """读 `.git/config` 的第一个 remote url（纯文件读，不跑 git 二进制——
    热路径纪律 + 不依赖用户环境）。没有 remote 返回空串。"""
    cfg = os.path.join(repo_root, ".git", "config")
    if not os.path.isfile(cfg):
        return ""      # worktree 的 .git 是文件不是目录：降级走 root 键
    try:
        with open(cfg, encoding="utf-8", errors="replace") as f:
            in_remote = False
            for line in f:
                s = line.strip()
                if s.startswith("["):
                    in_remote = s.startswith("[remote ")
                elif in_remote and s.startswith("url"):
                    _, _, url = s.partition("=")
                    return url.strip()
    except OSError:
        return ""
    return ""


#: 路径档的项目声明标志（键分层第三档，2026-08-29 Jason 拍板落实——live 实证
#: ~/dev/BladeX 无 .git 时第二档落空，但目录带 AGENTS.md 明显是项目）。
#: 判据与"有独立配置文件=独立身份"同一哲学；纯家目录无这些标志 ⇒ 仍 Global。
_PROJECT_MARKERS = ("AGENTS.md", "CLAUDE.md", "pyproject.toml",
                    "package.json", "Cargo.toml", "go.mod")


def resolve_project(cwd: str) -> ProjectIdentity:
    """cwd → 项目身份。结果按 cwd 缓存（FS 访问每 cwd 一次）。

    准入（2026-08-29 护栏 C）：cwd 必须绝对且是真实目录，否则直接 Global。

    键分层（ADR-0032 §4 全四档已落实）：realpath(cwd)（软链收敛）→
    ① git 仓库根：remote URL 有则 `git-remote:<url>`（跨搬家稳定）、
      无则 `git-root:<realpath(root)>`；
    ② 无仓库但目录带项目声明标志（_PROJECT_MARKERS）→ `path:<realpath>`
      （source=path；搬家/改名会换身份——该档如实接受这个代价）；
    ③ 都没有 = Global（空 id）——纯家目录会话不成为项目。
    """
    if not cwd:
        return ProjectIdentity()
    # 🔴 护栏 C（2026-08-29 拍板，读码发现、本次读数未触发但通路是真的）：
    # 抽出来的串必须是**绝对路径且真实存在的目录**，否则一律不采纳。
    # 不加这道门的后果：pattern 抓到一段垃圾（文档里引用 `Working directory:`
    # 这个串本身就够了——codex 那轮的 tool 结果里出现过三次），
    # `os.path.realpath` 会把相对串接到 **proxy 自己的 cwd** 上，
    # `_git_repo_root` 再向上走 12 层，正好走回 proxy 所在仓库的 `.git`
    # ⇒ **任何垃圾匹配都会被"识别成 proxy 自己那个项目"**。
    # isabs 先判（纯字符串、零 syscall）且**不进缓存**——垃圾串的取值空间无界，
    # 缓存无逐出策略（见 `_cache` 注释），让它进去就是慢性泄漏。
    if not os.path.isabs(cwd):
        return ProjectIdentity()
    hit = _cache.get(cwd)
    if hit is not None:
        return hit
    try:
        if not os.path.isdir(cwd):
            ident = ProjectIdentity()
            _cache[cwd] = ident
            return ident
        real = os.path.realpath(cwd)
        root = _git_repo_root(real)
        if not root:
            if any(os.path.exists(os.path.join(real, mk))
                   for mk in _PROJECT_MARKERS):
                key = f"path:{real}"
                ident = ProjectIdentity(
                    project_id="p-" + hashlib.sha256(
                        key.encode()).hexdigest()[:12],
                    name=os.path.basename(real), source="path", root=real)
            else:
                ident = ProjectIdentity()
        else:
            url = _git_remote_url(root)
            key = f"git-remote:{url}" if url else f"git-root:{root}"
            ident = ProjectIdentity(
                project_id="p-" + hashlib.sha256(key.encode()).hexdigest()[:12],
                name=os.path.basename(root),
                source="git-remote" if url else "git-root", root=root)
    except OSError:
        ident = ProjectIdentity()
    _cache[cwd] = ident
    return ident


def resolve_from_request(headers: dict[str, str],
                         messages: list[dict]) -> ProjectIdentity:
    """键分层入口：显式（占位）→ cwd 派生 → Global。

    🔴 读数（2026-08-29 补）：此前项目识别**一条日志都没有**——`identity_resolved`
    不带 project 字段，整份 proxy 日志 grep `project` 零命中，于是"claude-code 的
    project 恒空"潜伏到迁移批 live 验证才靠离线探针发现（同族：MQ-A20 的
    `Turn.subagent` 673 轮全 0）。字段在 schema 里、生产点也在，但没有读数 =
    看不见。`cwd_found` 与 `project_id` 分开报，是因为两种空的修法完全不同：
    没抽到标记 ⇒ 改 pattern；抽到了不成项目 ⇒ 该 cwd 本就不是项目（家目录）。
    与紧邻的上一条 `identity_resolved` 同属一次请求（按顺序对应，本模块拿不到
    agent_id，为一个日志字段改签名不值）。
    """
    explicit = explicit_project_id(headers)
    if explicit:
        return ProjectIdentity(project_id=explicit, source="explicit")
    cwd = extract_cwd(messages)
    ident = resolve_project(cwd)
    logger.info("project_resolved", cwd_found=bool(cwd), cwd=cwd[:120],
                project_id=ident.project_id, source=ident.source or "global",
                name=ident.name)
    return ident
