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
    CLOSED = 其后已有新轮次 = 任务闭合，证据可复核裁剪（ADR-0033）。
    """

    OPEN = "open"
    CLOSED = "closed"


class TaskUnit(BaseModel):
    """一个任务单元 = 一个**轮次**（ADR-0033 §2.1）：从用户一条消息开始，到 agent 停下为止，
    含其间的用户插话与全部工具往返（切分见 `turn_starts`）。

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
    intent_index: int = -1        # 轮次身份原文（unit_key 的原料）在 messages 中的下标
    intent_chars: int = 0
    conclusion_index: int = -1    # 单元内最后一条有正文、无 tool_calls 的 assistant
    conclusion_chars: int = 0

    # ── 跨度（逐条索引由 unit_indices() 重算，不持久化）──
    msg_start: int = -1           # 轮次首条 user（MQ-CA18 例外下 ≠ intent_index）
    msg_end: int = -1             # 单元最后一条消息下标（含）

    # ── 证据聚合 ──
    evidence_count: int = 0
    evidence_chars: int = 0
    tool_names: list[str] = Field(default_factory=list)  # 去重、保序、封顶

    # ── 置信度 ──
    # 1.0 = 结构确定。<1.0 仅用于协议层面的边界歧义（Anthropic/Responses 下
    # tool_result 落在 user 角色内，该单元含这种残留）——T0 实测全语料仅 2/4251 turn 命中。
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


def _field(msg: Any, name: str) -> Any:  # noqa: ANN401 —— dict 或带属性的消息对象
    """dict 与 pydantic 消息对象（`request.messages`）两种载体统一取字段。"""
    if isinstance(msg, dict):
        return msg.get(name)
    return getattr(msg, name, None)


def _has_tool_result_block(msg: Any) -> bool:  # noqa: ANN401
    """Anthropic 风格：tool_result 作为 content block 落在 user 消息里。

    命中 → 该 user 消息不是真实的用户意图，而是工具返回 → 不开轮次（ADR-0033 §2.2）。
    三个入站端点进 `do_inject` 前都已协议归一化（tool_result → role=tool），
    T0 实测全语料仅 2/4251 turn 命中这种残留形态。
    """
    c = _field(msg, "content")
    if not isinstance(c, list):
        return False
    for b in c:
        t = b.get("type") if isinstance(b, dict) else getattr(b, "type", None)
        if t == "tool_result":
            return True
    return False


def turn_starts(messages: list | None) -> list[int]:
    """ADR-0033 §2.2 轮次识别规则：开启新轮次的 user 消息下标（纯函数、O(n)、零 LLM）。

    > 一条非 `tool_result` 的 user 消息，若它前面最近的 assistant 消息**没有 `tool_calls`**
    > （agent 已停下），或它是首条 user，则开启新轮次；否则（agent 仍在工具循环中）并入
    > 当前轮次。连续多条 user 之间没有 assistant 的，合为同一轮次。

    🔴 **全仓唯一的轮次算法**（一套体系一个轮次定义，Jason 2026-09-14 拍板）：
    `build_task_units`（装配 / reconstruction / Index 归属的单元）与
    `ledger_runtime.user_turn_index`（账本防抖与陈旧判定的分母，MQ-L48）都调这里。
    守卫 `test_turn_rule_single_source.py`。

    输入形态：**协议归一化之后**的消息列表（tool_result 已转 role=tool）；
    接受 dict 或带 `.role` / `.tool_calls` / `.content` 属性的消息对象。
    """
    starts: list[int] = []
    last_assistant_calls = False
    user_since_assistant = False
    for i, m in enumerate(messages or ()):
        role = _field(m, "role")
        if role == "assistant":
            last_assistant_calls = bool(_field(m, "tool_calls"))
            user_since_assistant = False
            continue
        if role != "user" or _has_tool_result_block(m):
            continue
        if not starts:
            starts.append(i)                  # 首条 user
        elif user_since_assistant:
            pass                              # 连续 user（其间无 assistant）⇒ 合并
        elif not last_assistant_calls:
            starts.append(i)                  # agent 已停下 ⇒ 新轮次
        # else：agent 仍在工具循环中 ⇒ 用户插话并入当前轮次
        user_since_assistant = True
    return starts


def _is_machine_text(text: str) -> bool:
    """这条 user 文本是不是 agent 灌进 user 角色的机器文本（MQ-CA18）。

    判据 = `envelope.strip_envelopes` 命中任何一种信封（`# AGENTS.md instructions …<INSTRUCTIONS>` /
    `<environment_context>` / `<system-reminder>` 等）——**不另写一份指纹**，与蒸馏侧
    "什么是信封"同一张表（原则 12：同一件事一个判据）。
    """
    from bladex_core.envelope import strip_envelopes

    return bool(strip_envelopes(text)[1])


def _key_source_index(messages: list[dict], start: int) -> int:
    """轮次身份的原文下标（MQ-CA18 处置，批 U · U1.1）。

    默认 = 轮次首条 user（ADR-0033 §2.2）。例外：轮次**开头的连续 user 块**（其间无
    assistant / tool，同一个请求里一起到达）有 ≥2 条时，取其中**第一条不含任何信封**的那条。
    codex 每个会话首轮首条 user 是 `# AGENTS.md instructions` / `<environment_context>`
    （114/115 会话），同仓库跨会话逐字相同 ⇒ 按首条取 key 会让 32.4% 会话共用首轮 key，
    §2.2 合并后真人那条的 key 不再存在（111/111）。块里全是机器文本 ⇒ 回落首条（= §2.2 原文）。

    为什么只看开头块：块内消息同一个请求一起到达 ⇒ key 从轮次第一个请求起就稳定；
    若把工具循环里后到的插话也算进来，key 会在轮次中途翻转。
    """
    block: list[int] = []
    for j in range(start, len(messages)):
        role = messages[j].get("role")
        if role == "system":
            continue
        if role != "user" or _has_tool_result_block(messages[j]):
            break
        block.append(j)
    if len(block) < 2:
        return start
    for j in block:
        text = text_of(messages[j].get("content"))
        if text.strip() and not _is_machine_text(text):
            return j
    return start


def build_task_units(messages: list[dict]) -> list[TaskUnit]:
    """把 messages 切成任务单元 = **轮次**（确定性、零 LLM、纯函数；ADR-0033 §2.1/§2.2）。

    切分规则（`turn_starts`，全仓唯一的轮次算法）：
      - system 消息永远不属于任何单元（原样保留，不进降解射程）
      - 轮次 = agent 的一次运行段：从用户一条消息开始，到 agent 停下（给出无 tool_calls 的
        回答或反问）为止；**工具循环中的用户插话并入当前轮次**；连续 user 合为一轮
      - user content 里的 tool_result 残留不开轮次（记为该轮次的证据，置信度降为 0.5）
      - 首个 user 之前的消息不属于任何单元
      - 最后一个单元 = OPEN，其余 = CLOSED

    🔴 **口径切换（2026-09-23 批 U · U1.1）**：此前"每条 user 开一个单元"。unit_key 新旧
    **不互认**、存量**不迁移**、重建时按本规则**重放**（ADR-0033 §3）。

    身份：`unit_key = sha256(轮次身份原文)[:16]`，身份原文见 `_key_source_index`
    （默认轮次首条 user；MQ-CA18 例外）。`intent_index` 指向身份原文那条，
    `msg_start` 指向轮次首条 user（两者只在 MQ-CA18 例外下不同）。

    结论判定（与 ADR-0019 P2 `assistant_conclusion` 同口径）：
      单元内最后一条有正文且无 tool_calls 的 assistant = 该任务的 final answer。
    """
    starts = set(turn_starts(messages))
    units: list[TaskUnit] = []
    cur: TaskUnit | None = None
    cur_tools: list[str] = []

    for i, msg in enumerate(messages):
        role = msg.get("role", "")

        if role == "system":
            continue  # system 不入单元（ADR-0019：原样保留）

        if i in starts:
            k = _key_source_index(messages, i)
            key_text = text_of(messages[k].get("content"))
            cur_tools = []
            cur = TaskUnit(
                unit_key=make_unit_key(key_text),
                unit_index=len(units),
                status=TaskUnitStatus.OPEN,
                intent_index=k,
                intent_chars=len(key_text),
                msg_start=i,
                msg_end=i,
            )
            units.append(cur)
            continue

        if cur is None:
            continue  # 首个 user 之前的消息，不属于任何单元

        cur.msg_end = i

        if role == "user":
            if _has_tool_result_block(msg):
                # 协议残留：这条 user 其实是工具返回 ⇒ 记证据、标边界歧义
                cur.boundary_confidence = 0.5
                cur.evidence_count += 1
                cur.evidence_chars += len(text_of(msg.get("content")))
            continue  # 插话 / 连续 user：并入当前轮次

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

    logical_turn = 单元数（一个轮次 = 一个单元；2026-09-23 起按 ADR-0033 §2.2 口径，
                   工具循环中的用户插话不再计轮）
    roundtrip    = open 单元内的证据数（工具循环往返序号，含插话前的往返）

    本函数不独立遍历 messages，消灭第二次推断（ADR-0024 D1）。
    （与退役的 `server._infer_turn_metadata` "user 消息数"口径的逐字等价只在没有插话 /
    连续 user 时成立——U1.1 起口径有意改变。）
    """
    if not units:
        return 0, 0
    return len(units), units[-1].evidence_count


#: 无法按现行规则重算、沿用存下来的 key 时加的前缀（批 U · U1b，MQ-I29）。
#: 让旧口径 key 在库里**可见**、且不可能与新口径 key 相等（原则 13：不许静默混用）。
LEGACY_UNIT_KEY_PREFIX = "legacy:"

UNIT_KEY_RECOMPUTED = "recomputed"
UNIT_KEY_LEGACY_STORED = "legacy_stored"
UNIT_KEY_NONE = "none"


def index_unit_key(turn: Any) -> tuple[str, str]:  # noqa: ANN401 —— Hub Turn（proxy 类型，core 不依赖它）
    """Memory Index 侧"这一轮的 unit_key"——**全仓唯一取法**（批 U · U1b，MQ-I29）。

    = 本轮 open 单元的 `unit_key`，**由 `request_messages` 按现行规则（`build_task_units`）重算**，
    与热路径（`inject` → `cap_info["units"] = build_task_units(messages)`）同一个函数、同一份输入
    （Hub 存的就是那份 messages）。不读存下来的 `Turn.task_units`：U1.1（2026-09-23）起轮次口径变了，
    存量 task_units 是旧口径，读它 = 全量重建后新旧 unit_key 混在一个库里。

    返回 `(unit_key, source)`：
      - `recomputed`：原文完整 ⇒ 重算；
      - `legacy_stored`：原文缺失（`request_messages` 为空）或内容引用未还原（`content_ref` 残留，
        Hub 内容池丢了那段正文）⇒ 按现行规则算不出来，沿用存下的 key 并加 `legacy:` 前缀；
      - `none`：两样都没有 ⇒ `""`。
    """
    msgs = list(getattr(turn, "request_messages", None) or [])
    if msgs and not any(isinstance(m, dict) and "content_ref" in m for m in msgs):
        units = build_task_units(msgs)
        return (units[-1].unit_key if units else ""), UNIT_KEY_RECOMPUTED
    stored = getattr(turn, "task_units", None) or []
    key = getattr(stored[-1], "unit_key", "") if stored else ""
    if key:
        return LEGACY_UNIT_KEY_PREFIX + key, UNIT_KEY_LEGACY_STORED
    return "", UNIT_KEY_NONE
