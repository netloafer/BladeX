"""长内容分流（M1-2 / 复核附录 D.4）—— **分流，不是切分**。

## 这个模块在修什么

ADR-0024 T3 的"剥信封 + 超长分段"把候选覆盖救回 10 倍，方向没错；
但附录 D 的实测说明它同时把「长消息盲区」换成了「碎片投毒」：

    分段候选 672 个 / 整条候选 317 个 —— 分段路径已是 user 通道的主体
    seg0 命中的 50 条**全部空产出**（100%），seg1 86%
    因为被分段的长消息主体根本不是用户话语，而是规则文件 / 粘贴物，
    其开头是结构头部（`# ...` 之类），最没有蒸馏价值的那部分

根子在**拆分的对象就不该是它**。剥完信封仍超长的内容，几乎从来不是"很长的
用户话语"，而是文档。把文档切成六句假话语去走"用户陈述蒸馏"，每一段都在
错误的语义契约下被处理——准入规则、subject/attribute、preference 判定全部失准。
失效链 A（规则文件投毒）与失效链 B（半句幻觉）都从这里长出来。

正确形态是先问"这段内容是什么性质"，再决定怎么处理：

    剥信封后 ≤500 字符  → 话语路（契约成立：这确实是用户说的话）
    >500 字符          → 规则文件形态 → 画像原料路（E7.3），不产 assertion/preference
                       → 粘贴物/文档   → 文档路（E7.1 summarize_file），产 document 索引
                       → 真·超长话语   → 才分段（实测 <5%）

这与 mem0 的边界一致：它的抽取对象**只有对话消息对**，从不把粘贴文档当对话蒸——
文档记忆是另一个问题域。我们把两个问题域塞进了同一条流水线。

## 设计约束

- 纯函数、确定性、零 LLM（重建等价性 G6：同输入恒同输出）。
- **高精度优先**：判错方向的代价不对称——把话语误判成文档，那一轮的用户意图
  就只剩一句摘要；把文档误判成话语，就是投毒。所以文档侧的判据可以宽一点，
  话语侧要严（长且结构化 → 判文档）。
- 判据只看**结构**（长度、标记密度），不看任何领域词汇（X6 纪律）。
"""

from __future__ import annotations

import re
from enum import Enum

#: 话语契约成立的长度上限（与 consolidation 的 `_MAX_CONTENT_CHARS` 同源）
DISCOURSE_MAX_CHARS = 500

#: 超过这个长度，即便没有任何结构标记也判文档。
#: 依据：一条 4000+ 字符、连空行都没有的"用户消息"在真实语料里是粘贴物而非话语。
#: 人不会一口气打这么多字还不分段。
DOCUMENT_HARD_CHARS = 4000

#: 结构标记阈值——命中任一即判文档
_MIN_HEADINGS = 3        # markdown 标题
_MIN_LIST_ITEMS = 6      # 列表项
_MIN_TABLE_ROWS = 3      # 表格行

_HEADING_RE = re.compile(r"^#{1,6}\s+\S", re.M)
_FENCE_RE = re.compile(r"^\s*```", re.M)
_LIST_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)\S", re.M)
_TABLE_RE = re.compile(r"^\s*\|.*\|\s*$", re.M)

#: 规则文件正文的捕获标记（与 `rulefile._CONTENTS_RE` 同源形态）。
#: 单独放一份是因为分流只需要"**是不是**规则文件"，不需要把正文抽出来；
#: 依赖 rulefile 的完整抽取会让分流为了一个布尔值付整段解析的代价。
_RULEFILE_MARK_RE = re.compile(
    r"Contents of\s+\S+?\.(?:md|mdc|txt|toml|ya?ml|cursorrules)\b", re.I)

#: 规则文件的另一种形态：整条就是一份 AGENTS.md / CLAUDE.md 正文被回传，
#: 没有 `Contents of` 包头。判据是**文件名出现在头部** + markdown 结构。
_RULEFILE_NAME_RE = re.compile(
    r"\b(?:agents?|claude|cursor|copilot|codex)\s*\.(?:md|mdc)\b", re.I)


class InputRoute(str, Enum):
    """一条（剥完信封的）内容该走哪条路。"""

    DISCOURSE = "discourse"   # 用户话语 → 统一蒸馏调用（必要时分段）
    RULEFILE = "rulefile"     # 规则文件 → 画像原料（profile_obs），不产 assertion/preference
    DOCUMENT = "document"     # 粘贴物/文档 → 文件内容索引（summary + keywords）


def _looks_like_rulefile(text: str) -> bool:
    if _RULEFILE_MARK_RE.search(text):
        return True
    # 文件名出现在**头部**（前 400 字符）且带 markdown 结构 —— 正文回传形态。
    # 限定头部是为了不误伤"正文里顺口提到 CLAUDE.md"的真实话语。
    head = text[:400]
    return bool(_RULEFILE_NAME_RE.search(head)
                and len(_HEADING_RE.findall(text)) >= _MIN_HEADINGS)


def _looks_like_document(text: str) -> bool:
    """结构化程度是否达到"这是一份文档而不是一段话"。"""
    if len(text) >= DOCUMENT_HARD_CHARS:
        return True
    if _FENCE_RE.search(text):
        return True
    if len(_HEADING_RE.findall(text)) >= _MIN_HEADINGS:
        return True
    if len(_LIST_RE.findall(text)) >= _MIN_LIST_ITEMS:
        return True
    return len(_TABLE_RE.findall(text)) >= _MIN_TABLE_ROWS


def classify_input(
    text: str, *, discourse_max_chars: int = DISCOURSE_MAX_CHARS,
) -> InputRoute:
    """判定一条剥完信封的内容走哪条路。纯函数、确定性。

    `text` 必须**已经剥过信封**（`envelope.strip_envelopes`）——
    对未剥的文本判定没有意义：信封本身全是结构标记，什么都会被判成文档。
    """
    stripped = (text or "").strip()
    if not stripped:
        return InputRoute.DISCOURSE
    # 🔴 显式包头**先于**长度判定：`Contents of X.md` 是 agent 注入规则文件的
    # 无歧义结构证据，与长度无关。长度门槛问的是"这是不是用户说的话"，
    # 而这个标记本身就是"不是"的正面证据——一份 300 字符的规则文件同样不该
    # 走用户陈述蒸馏。（名称启发式不享受这个豁免：它要靠长度+结构才够可靠，
    # 否则"我改了下 CLAUDE.md"这种真实话语会被误伤。）
    if _RULEFILE_MARK_RE.search(stripped):
        return InputRoute.RULEFILE
    if len(stripped) <= discourse_max_chars:
        # 契约成立：这个长度的内容就是用户说的话，不必再问性质。
        return InputRoute.DISCOURSE
    if _looks_like_rulefile(stripped):
        return InputRoute.RULEFILE
    if _looks_like_document(stripped):
        return InputRoute.DOCUMENT
    # 长、但没有任何文档结构 —— 真·超长话语（实测 <5%），交给分段
    return InputRoute.DISCOURSE
