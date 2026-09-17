"""会话事件检测（ADR-0026 铁律3 / U4B-B2）——确定性、零 LLM。

铁律3：会话事件（「谁做的、卡在哪、被打回、验收通过」）**不产出为条目**，
折叠进对应 Matter 卡的 lifecycle（`Matter.record_lifecycle`）。蒸馏 prompt 的
准入规则（B1）挡住「用户问了什么」型条目；本模块补的是**正面通道**：
把有交接价值的会话事件从 user 消息里确定性识别出来，写进 Matter 卡。

设计约束：
  - 纯函数、确定性（重建等价性 G6：同输入永远同输出，重放两次逐字节一致）。
  - 高精度优先（宁漏勿误）：词表故意收窄，只认无歧义的强信号词；
    「完成/做好了」这类高频泛化词不收（误报会污染 lifecycle）。
  - detail 取匹配词所在句子的截断（上限 _MAX_DETAIL_CHARS），供交接阅读。

消费者：memory_index.rebuild_from_hub（BLADEX_MATTER_LIFECYCLE=1 gated）→
Matter.lifecycle → U7 注入②平面（Matter 卡交接）。
"""

from __future__ import annotations

import re

# detail 截断上限（lifecycle 是交接摘要，不是原文存档）
_MAX_DETAIL_CHARS = 120

# 事件词表（正则, 事件名）。顺序即优先级；同一事件类型在一段文本里只记一次。
# 词表收窄原则：只收「对交接有信息量 + 无歧义」的强信号。
_EVENT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # 打回 / 返工（多 agent 复核工作流的核心事件）
    (re.compile(r"打回|返工|重做一遍|rework(?:ed)?\b", re.IGNORECASE), "reworked"),
    # 卡点 / 阻塞
    (re.compile(r"卡在|卡住|被阻塞|阻塞在|blocked\s+(?:on|by)", re.IGNORECASE), "blocked"),
    # 验收通过 / 签收
    (re.compile(r"验收通过|复核通过|签收|验收过了|sign(?:ed)?[- ]off", re.IGNORECASE), "accepted"),
    # 验收不通过（与 accepted 分开，先匹配否定式）
    (re.compile(r"验收(?:不|未)通过|复核(?:不|未)通过|验收失败", re.IGNORECASE), "rejected"),
    # 指派 / 交接给某人
    (re.compile(r"交给|交由|指派给|移交给|hand(?:ed)?\s+over\s+to", re.IGNORECASE), "assigned"),
    # 暂停 / 搁置
    (re.compile(r"先暂停|搁置|挂起|parked\b", re.IGNORECASE), "paused"),
]

# 句子切分（中英文句读 + 换行）
_SENT_SPLIT = re.compile(r"[。！？!?\n]+")


def _sentence_of(text: str, pos: int) -> str:
    """取包含 pos 的句子（按句读切分），截断到 _MAX_DETAIL_CHARS。"""
    start = 0
    for m in _SENT_SPLIT.finditer(text):
        if m.start() >= pos:
            return text[start:m.start()].strip()[:_MAX_DETAIL_CHARS]
        start = m.end()
    return text[start:].strip()[:_MAX_DETAIL_CHARS]


def detect_session_events(text: str) -> list[tuple[str, str]]:
    """从一段 user 消息文本里确定性检出会话事件。

    返回 [(event, detail)]，同一事件类型只取首个匹配（去噪）。
    无匹配返回 []。纯函数：同输入永远同输出（G6 重建等价性）。
    """
    if not text or not text.strip():
        return []
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    # rejected 的模式含「验收…通过」子串，需先于 accepted 判定——
    # 逐条按词表顺序扫，但 rejected 命中时撤销同句 accepted（否定式优先）。
    hits: dict[str, tuple[int, str]] = {}
    for pattern, event in _EVENT_PATTERNS:
        m = pattern.search(text)
        if m is None:
            continue
        if event in hits:
            continue
        hits[event] = (m.start(), _sentence_of(text, m.start()))
    # 否定式优先：同一句同时命中 accepted 与 rejected → 只留 rejected
    if "rejected" in hits and "accepted" in hits:
        if hits["rejected"][1] == hits["accepted"][1]:
            hits.pop("accepted")
    for event, (_pos, detail) in sorted(hits.items(), key=lambda kv: kv[1][0]):
        if event not in seen:
            seen.add(event)
            out.append((event, detail))
    return out
