"""画像原语（ADR-0026 R5 / U10）——规则文件被动捕获，确定性、零 LLM。

规则文件（CLAUDE.md / AGENTS.md / .cursorrules 等）出现在 system/user 消息中时，
被动捕获为画像观察原料（profile_obs 语义）：文件名 + 内容 hash + 摘录。
consolidator 据此综合 User 画像卡；工具计数器在 Memory Index 侧从 Turn.tool_events 聚合。

纯函数、agent 中立；重放同一批消息 → 同一批观察（G6 重建等价性）。
"""

from __future__ import annotations

import hashlib

# 被动捕获的规则文件名（判别性强的固定词表；宁漏勿误）
_RULE_FILE_NAMES: tuple[str, ...] = (
    "CLAUDE.md", "AGENTS.md", "AGENT.md", ".cursorrules",
    "engineering-conventions.md",
)

# 摘录长度（画像卡是提示不是存档；全文在 Memory Hub 总账）
_EXCERPT_CHARS = 120


def detect_rule_files(text: str) -> list[tuple[str, str, str]]:
    """从一段消息文本中检出规则文件出现。

    返回 [(file_name, content_hash8, excerpt)]；每个文件名只取首次出现。
    excerpt = 提及处之后的首个非空行（通常是规则正文首行），截断 _EXCERPT_CHARS。
    确定性：同输入永远同输出。
    """
    if not text:
        return []
    out: list[tuple[str, str, str]] = []
    for name in _RULE_FILE_NAMES:
        pos = text.find(name)
        if pos < 0:
            continue
        tail = text[pos + len(name):]
        excerpt = ""
        for line in tail.splitlines():
            line = line.strip()
            if line and not line.startswith(("#", "-", "=", "*")) or len(line) > 8:
                excerpt = line[:_EXCERPT_CHARS]
                break
        h = hashlib.sha256(tail[:2000].encode()).hexdigest()[:8]
        out.append((name, h, excerpt))
    return out
