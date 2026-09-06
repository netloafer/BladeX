"""规则文件体系 + 画像渲染（ADR-0028 E7.3）—— 确定性、零 LLM。

## 这条卡在补什么

现状的"规则文件被动捕获"（U10 / `profile.py`）只记了 **文件名 + hash + 一行摘录**，
渲染出来是 trace 报告里那两行"路径串联乱码"，而且**每轮都在注**。
真正有价值的东西——规则文件正文——在信封剥离时被整段丢掉了。

本模块做三件事：

  1. **正文捕获**：写入侧剥掉的 `<user_claude_md>` / `<system-reminder>` 信封里，
     `Contents of <path>` 标记段就是规则文件正文。剥离时顺手交出来存副本
     （hash 变才更新；Session 首轮强制比对一次 = "新 Session 自动更新"）。
  2. **USER.md**：把身份/活跃度/偏好渲染成一份人读得懂的画像（确定性模板）。
  3. **per-agent 习惯文件**：工具 top5（含成功率，E7.4）+ 该 agent 的 audience
     专属条目 + 规则文件清单。

渲染全是纯函数：同输入同输出，重放等价（G6）。
"""

from __future__ import annotations

import hashlib
import re

# 规则文件正文的捕获标记（CLAUDE.md 的注入格式 / Claude Code 的 user_claude_md 信封）
_CONTENTS_RE = re.compile(
    r"Contents of ([^\s\n]+?\.(?:md|mdc|txt|toml|ya?ml|cursorrules))\s*(?:\([^)]*\))?\s*:?\s*\n",
    re.I,
)

# 单份规则文件副本的体积上限（32KB，超出截断——副本是给人看的，不是存档）
MAX_RULEFILE_CHARS = 32_000

# 规则文件名规范化：只取路径尾段（同一份 CLAUDE.md 在不同机器上路径不同）
_MAX_NAME_CHARS = 64


def norm_rulefile_name(path: str) -> str:
    """规范化规则文件名 = 路径尾段（小写）。"""
    tail = (path or "").replace("\\", "/").rstrip("/").split("/")[-1]
    return tail.strip().lower()[:_MAX_NAME_CHARS]


def extract_rule_files(text: str) -> list[tuple[str, str, str]]:
    """从一段（通常是信封内的）文本里抽出规则文件正文。

    返回 [(规范化文件名, 正文, content_hash)]，按出现顺序、同名只取首次。

    捕获依据是 `Contents of <path>` 这个标记段——它是 agent 注入规则文件时的
    固定形态（CLAUDE.md / AGENTS.md / .cursorrules 都走它）。
    正文 = 标记之后到下一个标记（或文本结束）之间的部分。
    """
    if not text:
        return []
    marks = list(_CONTENTS_RE.finditer(text))
    if not marks:
        return []
    out: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for i, m in enumerate(marks):
        name = norm_rulefile_name(m.group(1))
        if not name or name in seen:
            continue
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        body = text[m.end():end].strip()[:MAX_RULEFILE_CHARS]
        if not body:
            continue
        seen.add(name)
        out.append((name, body, hashlib.sha256(body.encode()).hexdigest()[:16]))
    return out


# ── USER.md / per-agent 习惯文件渲染（确定性模板）────────────────────────


def render_user_md(
    *,
    user_id: str,
    agents: list[dict],
    recent_activity: list[dict],
    preferences: list[str],
    coding_preferences: list[str],
    key_count: int = 0,
) -> str:
    """渲染 USER.md（ADR-0028 E7.3 模板）。

    agents:          [{"agent_id","turns","last_seen"}]（按 turns 降序）
    recent_activity: [{"agent_id","turns"}]（近 7 日）
    preferences:     item_kind=preference 且 t_invalid 空，按 importance 降序 top10 原句
    coding_preferences: 上面的子集（entities 命中语言/工具词表）
    """
    lines = ["# USER", "", f"- user: {user_id}", f"- keys: {key_count}"]
    if agents:
        lines.append("- agents:")
        for a in agents:
            last = a.get("last_seen", "") or "?"
            lines.append(f"  - {a['agent_id']}（{a.get('turns', 0)} turns，最近 {last}）")
    if recent_activity:
        lines += ["", "## 近 7 日活跃度", ""]
        lines += [f"- {a['agent_id']}: {a.get('turns', 0)} turns" for a in recent_activity]
    if preferences:
        lines += ["", "## 偏好", ""]
        lines += [f"- {p}" for p in preferences]
    if coding_preferences:
        lines += ["", "## 编程偏好", ""]
        lines += [f"- {p}" for p in coding_preferences]
    return "\n".join(lines) + "\n"


def render_agent_md(
    *,
    agent_base: str,
    tools: list[dict],
    own_facts: list[str],
    rule_files: list[str],
) -> str:
    """渲染 per-agent 习惯文件。

    tools: [{"name","calls","ok","err","last_err_class"}]（按 calls 降序 top5）
    """
    lines = [f"# AGENT {agent_base}", ""]
    if tools:
        lines.append("## 工具")
        lines.append("")
        for t in tools:
            calls = int(t.get("calls", 0))
            ok = int(t.get("ok", 0))
            rate = (ok / calls * 100) if calls else 0.0
            lines.append(f"- 工具: {t['name']} ({calls} 次, ok {rate:.0f}%)")
            # 常见失败：err 率 > 30% 且 calls ≥ 5 才提示（样本太小的比率没有意义）
            err = int(t.get("err", 0))
            if calls >= 5 and err / calls > 0.30 and t.get("last_err_class"):
                lines.append(f"  ⚠ 常见失败: {t['last_err_class']}")
    if own_facts:
        lines += ["", "## 该 agent 专属记忆", ""]
        lines += [f"- {c}" for c in own_facts]
    if rule_files:
        lines += ["", "## 规则文件", ""]
        lines += [f"- {n}" for n in rule_files]
    return "\n".join(lines) + "\n"


# ── ①平面画像卡 render v2（节选，≤5 行完整句子）────────────────────────

PROFILE_CARD_MAX_LINES = 5


def render_profile_card(user_md: str, agent_md: str = "", *,
                        max_lines: int = PROFILE_CARD_MAX_LINES) -> list[str]:
    """从 USER.md / agent 文件节选出 ①平面画像卡（ADR-0028 E7.3 render v2）。

    取代旧渲染（"[User 画像] 使用规则文件 CLAUDE.md：<一段路径乱码>"）：
    只选**完整句子**的条目行（`- ` 开头、去掉标题与统计行），≤5 行。
    """
    picked: list[str] = []
    for blob, tag in ((user_md, "User 画像"), (agent_md, "Agent 习惯")):
        if not blob:
            continue
        for raw in blob.splitlines():
            line = raw.strip()
            if not line.startswith("- ") or len(line) < 6:
                continue
            body = line[2:].strip()
            # 跳过纯统计/清单行（"agents:"、"key: 3"、"xxx: 12 turns"）
            if body.endswith(":") or re.fullmatch(r"[\w.:\-]+\s*[:：]\s*\d+.*", body):
                continue
            picked.append(f"[{tag}] {body}")
            if len(picked) >= max_lines:
                return picked
    return picked
