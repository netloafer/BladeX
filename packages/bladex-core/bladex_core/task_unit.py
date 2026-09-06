"""TaskUnit — 任务单元，贯穿 Pipeline→Memory Hub→Memory Index 的一等实体（ADR-0024 §4.2）。

领域词汇表：TaskUnit 是"一次用户意图 + 其后全部证据往返"的逻辑单元，
禁止叫 Turn / Round / Segment（Turn 是一次 LLM 往返，粒度更细）。
agent 中立：不依赖 proxy 或任何 agent 的类型。

核心不变式（ADR-0024 §4.1）：
    任务单元在热路径被确定性地识别一次，随 Turn 落 Memory Hub，Memory Index 直接消费——
    归属从"推断"变成"记录"。

三条实测驱动的设计约束（基线见 ADR-0024 前的实测基线）：

  1. **跨轮身份用 unit_key，不用位置索引**（T0 修正 A）。
     prefix_changed 占真实流量 19–28%，agent 压缩历史后位置索引整体漂移。
     unit_key = sha256(intent_text) —— intent_text 是 agent 自己发的原文，
     压缩不改写它，跨轮匹配天然稳定。

  2. **只存引用不存正文**（T0 修正 D）。
     claude-code 的 user 消息 p50 = 70,261 字符、p99 = 510,002 字符。
     若把 intent_text/conclusion_text 存进 TaskUnit，Memory Hub 体积直接翻倍。
     故只存 index + chars + hash，正文从 request_messages[index] 取
     （Memory Hub.get() 已自动还原 content_ref）。

  3. **证据存聚合不存逐条**。claude-code 平均每 turn 245 条 tool 消息，
     逐条 EvidenceRef 是 schema 膨胀的主要来源。持久化只留计数/体积/
     去重后的 tool 名（ADR-0025 B 层关联匹配的命中面），
     逐条索引由 unit_indices() 在运行时从 msg_span 重算（确定性）。

零 LLM、纯结构派生 → 同一会话内逐字稳定（保上游 prompt cache）、
重建等价性成立（ADR-0012 §3.6）。
"""

from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# 单元内保留的去重 tool 名上限（防单元证据种类爆炸撑大 schema）
_MAX_TOOL_NAMES = 32


class TaskUnitStatus(str, Enum):
    """单元状态（ADR-0024 §4.2）。

    OPEN   = 最后一个单元，当前任务，证据仍在用。
    CLOSED = 其后已有新 user 消息 = 任务闭合，证据可降解。
    """

    OPEN = "open"
    CLOSED = "closed"


class TaskUnit(BaseModel):
    """一个任务单元：user 消息起始 + 其后全部非 user 消息，直到下一条 user。

    索引口径：全部 index 指向**构造时传入的 messages 列表**。proxy 侧调用方
    必须传 agent 原始 messages（Memory Hub 存的那份），否则 index 失去意义。
    """

    # ── 身份（ADR-0024 T0 修正 A）──
    # unit_key: sha256(intent_text)[:16]，跨轮匹配的唯一依据。
    # unit_index: 仅供本轮定位，prefix_changed 后会漂移，不做跨轮身份。
    unit_key: str = ""
    unit_index: int = 0
    status: TaskUnitStatus = TaskUnitStatus.OPEN

    # ── 内容引用（T0 修正 D：存引用不存正文）──
    intent_index: int = -1        # 起始 user 消息在 messages 中的下标
    intent_chars: int = 0
    conclusion_index: int = -1    # 单元内最后一条有正文、无 tool_calls 的 assistant
    conclusion_chars: int = 0

    # ── 跨度（逐条索引由 unit_indices() 重算，不持久化）──
    msg_start: int = -1           # = intent_index
    msg_end: int = -1             # 单元最后一条消息下标（含）

    # ── 证据聚合 ──
    evidence_count: int = 0
    evidence_chars: int = 0
    tool_names: list[str] = Field(default_factory=list)  # 去重、保序、封顶

    # ── 置信度 ──
    # 1.0 = 结构确定。<1.0 仅用于协议层面的边界歧义（Anthropic/Responses 下
    # tool_result 落在 user 角色内）——T0 实测全语料仅 2/4251 turn 命中。
    # **prefix_changed 不降级本字段**（T0 修正 A：切分只依赖当前 messages，结构自洽）。
    boundary_confidence: float = 1.0

    model_config = {"extra": "allow"}

    @property
    def closed(self) -> bool:
        return self.status is TaskUnitStatus.CLOSED


def text_of(content: Any) -> str:
    """从 message content 提取纯文本（兼容 str 与多模态 list）。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and isinstance(p.get("text"), str)
        )
    return ""


def make_unit_key(intent_text: str) -> str:
    """跨轮稳定的单元身份键（ADR-0024 T0 修正 A）。

    只依赖 agent 自己发的 user 原文——agent 压缩历史不改写它，
    故同一件事在压缩前后算出同一个 key。
    """
    return hashlib.sha256(intent_text.encode("utf-8", "ignore")).hexdigest()[:16]


def _has_tool_result_block(msg: dict) -> bool:
    """Anthropic 风格：tool_result 作为 content block 落在 user 消息里。

    命中 → 该 user 消息不是真实的用户意图，而是工具返回 → 边界歧义。
    T0 实测全语料仅 2/4251 turn 命中（ADR-0023 的协议转换已覆盖绝大多数）。
    """
    c = msg.get("content")
    return isinstance(c, list) and any(
        isinstance(b, dict) and b.get("type") == "tool_result" for b in c
    )


def build_task_units(messages: list[dict]) -> list[TaskUnit]:
    """把 messages 切成任务单元（确定性、零 LLM、纯函数）。

    切分规则（与 ADR-0019 `assembly._split_units` 语义一致，本函数是其唯一实现）：
      - system 消息永远不属于任何单元（原样保留，不进降解射程）
      - 每条 user 消息开启一个新单元
      - 首个 user 之前的消息不属于任何单元
      - 最后一个单元 = OPEN，其余 = CLOSED

    结论判定（与 ADR-0019 P2 `assistant_conclusion` 同口径）：
      单元内最后一条有正文且无 tool_calls 的 assistant = 该任务的 final answer。
    """
    units: list[TaskUnit] = []
    cur: TaskUnit | None = None
    cur_tools: list[str] = []
    ambiguous = False

    for i, msg in enumerate(messages):
        role = msg.get("role", "")

        if role == "system":
            continue  # system 不入单元（ADR-0019：原样保留）

        if role == "user":
            if _has_tool_result_block(msg):
                # 协议边界歧义：这条 user 其实是工具返回。仍按 user 起新单元
                # （保持与现有 _split_units 逐字一致），但标记置信度下降。
                ambiguous = True
            text = text_of(msg.get("content"))
            cur_tools = []
            cur = TaskUnit(
                unit_key=make_unit_key(text),
                unit_index=len(units),
                status=TaskUnitStatus.OPEN,
                intent_index=i,
                intent_chars=len(text),
                msg_start=i,
                msg_end=i,
                boundary_confidence=0.5 if ambiguous else 1.0,
            )
            units.append(cur)
            continue

        if cur is None:
            continue  # 首个 user 之前的消息，不属于任何单元

        cur.msg_end = i

        if role == "tool":
            cur.evidence_count += 1
            cur.evidence_chars += len(text_of(msg.get("content")))
            name = msg.get("name") or ""
            if name and name not in cur_tools and len(cur_tools) < _MAX_TOOL_NAMES:
                cur_tools.append(name)
                cur.tool_names = list(cur_tools)
        elif role == "assistant":
            # final answer 判定：有正文 + 无 tool_calls。后出现的覆盖先出现的。
            if not msg.get("tool_calls"):
                text = text_of(msg.get("content"))
                if text.strip():
                    cur.conclusion_index = i
                    cur.conclusion_chars = len(text)

    for u in units[:-1]:
        u.status = TaskUnitStatus.CLOSED

    return units


def unit_indices(unit: TaskUnit, messages: list[dict]) -> list[int]:
    """重算单元覆盖的逐条消息下标（确定性，故不持久化）。

    与 build_task_units 的切分口径一致：跳过 system 消息。
    """
    if unit.msg_start < 0:
        return []
    end = min(unit.msg_end, len(messages) - 1)
    return [
        i
        for i in range(unit.msg_start, end + 1)
        if messages[i].get("role") != "system"
    ]


def derive_turn_metadata(units: list[TaskUnit]) -> tuple[int, int]:
    """从单元派生 (logical_turn, roundtrip)——取代 `server._infer_turn_metadata`。

    logical_turn = 单元数（一个用户轮 = 一个单元）
    roundtrip    = open 单元内的证据数（工具循环往返序号）

    与旧实现的等价性：旧实现 logical_turn = user 消息数（= 单元数，逐字等价）；
    roundtrip = 最后一条 user 之后的 tool 消息数（= open 单元的 evidence_count，
    逐字等价）。差别只在：本函数不再独立遍历 messages，消灭第二次推断（ADR-0024 D1）。
    """
    if not units:
        return 0, 0
    return len(units), units[-1].evidence_count
