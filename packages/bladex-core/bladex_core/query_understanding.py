"""Query 理解层（ADR-0028 E6.1）—— 确定性，热路径零 LLM。

检索链路的第一层。真实数据里，进到检索的 query 有两种常见的坏形态：

  1. **信封当 query**：claude-code 的末条 user 消息 p50 = 70,261 字符，
     其中 98.21% 是 agent 内部协议信封（`<transcript>` / `<system-reminder>` …）。
     拿这坨东西去 embed，向量指向的是"信封的语义"，不是用户想问的事。
  2. **短指代**：「这个怎么办」「继续」「为什么」——8 个字符不到，
     向量几乎不携带信息，召回等同随机。

处置（两条都是确定性的）：

    q, _ = strip_envelopes(q_raw)                  # 复用 envelope.py，同一套规则
    if len(q) < 8 or q ∈ 指代词表:
        q = 当前 open 单元的 intent_text + " " + q  # 用当前任务意图扩写
    q = q[:512]                                    # e5 query 输入上限

**明确不做**：HyDE / LLM query 改写。热路径预算 200ms，容不下一次 LLM 往返；
HyDE 可用于 E5 离线扩充评估集，不进在线路径。
"""

from __future__ import annotations

from bladex_core.envelope import strip_envelopes
from bladex_core.identifiers import extract_identifiers

# e5 家族 query 侧输入上限（超出部分对向量几乎无贡献，白付算力）
MAX_QUERY_CHARS = 512

# 短查询判定门槛（低于此长度且无标识符 → 需要扩写）
MIN_SELF_SUFFICIENT_CHARS = 8

# 指代词表：本身不携带主题信息的 query（中英文）。
# 判定用"整条 query 规范化后落在表内"，不是子串匹配——
# 「这个方案为什么不работает」含"这个"但信息量充足，不该被当成指代。
_PRONOUN_QUERIES: frozenset[str] = frozenset({
    "这个", "那个", "它", "他", "她", "这", "那", "这些", "那些",
    "继续", "接着", "然后呢", "然后", "呢", "怎么办", "怎么样", "如何",
    "为什么", "为啥", "why", "and then", "then", "continue", "go on",
    "it", "this", "that", "these", "those", "ok", "okay", "嗯", "好的",
})

_TRIM_CHARS = " \t\n\r。，、？！?!.,:;：；~～"


def _normalize_for_pronoun_check(text: str) -> str:
    return text.strip(_TRIM_CHARS).strip().lower()


def is_pronoun_query(query: str) -> bool:
    """整条 query 就是个指代词/过场词（不携带主题信息）。"""
    return _normalize_for_pronoun_check(query) in _PRONOUN_QUERIES


def understand_query(
    q_raw: str,
    *,
    open_unit_intent: str = "",
    max_chars: int = MAX_QUERY_CHARS,
) -> tuple[str, list[str]]:
    """返回 (用于检索的 query, 标识符列表)。

    - query：剥信封 → 需要时用当前 open 单元意图扩写 → 截断到 max_chars。
    - 标识符：从**原始文本**抽（`q_raw`，不是剥完的）——`<transcript>` 里也可能
      带着用户提到的文件名/错误码，词法通道用得上，而剥离只是为了让**向量**别跑偏。

    纯函数、确定性：同输入同输出，重放等价。
    """
    ids = extract_identifiers(q_raw or "")

    q, _kinds = strip_envelopes(q_raw or "")
    q = q.strip()

    needs_expansion = (
        (len(q) < MIN_SELF_SUFFICIENT_CHARS and not ids) or is_pronoun_query(q)
    )
    if needs_expansion and open_unit_intent:
        intent = strip_envelopes(open_unit_intent)[0].strip()
        if intent:
            q = f"{intent} {q}".strip()

    return q[:max_chars], ids


def last_user_text(messages: list[dict]) -> str:
    """末条 user 消息的文本（含 list content 的多模态形态）。

    = 进入理解层之前的**原始 query**。此前同一段逻辑有三份实现
    （`server._extract_query` / `inject._understand` 里的 open 单元取文 /
    离线仪器各自的近似），而"筛子和被测系统各算各的"正是本仓 MQ-S4 的形态。
    这里是唯一实现，另两处委托过来。
    """
    for msg in reversed(messages or []):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                p.get("text", "") for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
    return ""


def open_unit_intent(messages: list[dict]) -> str:
    """当前 open 任务单元的意图正文（短指代扩写的语料）。

    取 `build_task_units(messages)[-1].intent_index` 指向的那条消息——单元的
    索引口径对的是**未剥离**的原始 messages（ADR-0024 T1 修正 D），故这里
    直接按下标回查正文。
    """
    from bladex_core.task_unit import build_task_units

    units = build_task_units(messages or [])
    if not units or not (0 <= units[-1].intent_index < len(messages)):
        return ""
    c = messages[units[-1].intent_index].get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return " ".join(
            part.get("text", "") for part in c
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def retrieval_query(
    messages: list[dict],
    q_raw: str | None = None,
    *,
    max_chars: int = MAX_QUERY_CHARS,
) -> tuple[str, list[str]]:
    """整轮 messages → **生产实际用于检索的 query**（纯函数、零 LLM、可离线复现）。

    热路径（`inject._understand`）与离线仪器（`scripts/eval_consumption.py`）
    共用这一条路。仪器若自己近似一份，就会拿"末条 user 原文"去判相关性，
    而检索用的是剥完信封、必要时被 open 单元意图扩写过的串——
    claude-code 那种 p50 七万字符的信封会让相关性虚高，「继续」这类短指代
    会让它虚低，且两个方向随 agent 而异。参照系脱钩的尺子给不出可用的判读。
    """
    raw = q_raw if q_raw is not None else last_user_text(messages)
    return understand_query(raw, open_unit_intent=open_unit_intent(messages),
                            max_chars=max_chars)
