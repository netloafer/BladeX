"""Fact — 蒸馏出的一条结构化事实（存 Memory Index / 派生）。

领域词汇表：Fact 是语义存储里的一条结构化事实，禁止叫 Entry/Item/Record。
agent 中立：不依赖 proxy 或任何 agent 的类型。
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum

from pydantic import AliasChoices, BaseModel, Field


class ItemKind(str, Enum):
    """条目六类封闭枚举（ADR-0026 §4.2）。

    不用开放 tag（现库 tags 字段无消费者即前车之鉴）。
    """

    ASSERTION = "assertion"      # 语义记忆：一个跨会话成立的结论（+ subject·attribute 取代键）
    PREFERENCE = "preference"    # 程序记忆：用户希望被如何对待
    PROCEDURE = "procedure"      # 程序记忆：怎么做有效/无效
    LESSON = "lesson"            # 情景→语义：为什么错 + 怎么纠正（根因通道）
    FILE_REF = "file_ref"        # 文件指针：路径 + content_hash + 关键判断（不存副本）
    PROFILE_OBS = "profile_obs"  # 规则文件观察原料（供画像卡综合，不直接注入）


class Provenance(str, Enum):
    """条目的**来源通道**（M1 数据结构变更总表）。

    为什么要单独一个字段而不是继续用 `tags="origin:progress"`：
    tags 是自由文本，没有封闭枚举就没有可靠的消费者——检索配额、importance、
    审计三处都想按"这条是用户自己说的，还是模型总结出来的"分权，而
    `tags.split(":")[-1]` 这种读法一改措辞就静默失效（现库 tags 字段无消费者
    正是前车之鉴，ADR-0026 §4.2）。

    空串 = 历史数据（未知来源），所有消费者必须能处理"未知"。
    """

    USER_DIRECT = "user_direct"    # 用户自己说的话（最高可信）
    CONCLUSION = "conclusion"      # 模型的最终回答（结论即蒸馏）
    PROGRESS = "progress"          # 模型调工具前写的过程说明/发现
    DOCUMENT = "document"          # 粘贴物 / 文件正文摘要（E7.1 文档路）
    SCAFFOLDING = "scaffolding"    # 规则文件 / 宿主脚手架（画像原料，不直接注入）


#: 合法 provenance 值集合（解析兜底用；未知一律折叠成空串 = 未知）
PROVENANCE_VALUES: frozenset[str] = frozenset(p.value for p in Provenance)


def fact_lane(tags: str, item_kind: "ItemKind | str" = "") -> str:
    """从 tags 读 lane 标记（M9 三条道：task / profile / tool；空 = 未标）。

    🔴 按逗号切开后**全串相等**，不做子串 `in` 判断——tags 是自由文本，
    子串判断一改措辞就静默失效（probe_memory 头注的同一条教训）。
    `item_kind == PROFILE_OBS` 视同 lane:profile（该 kind 按定义就是画像原料，
    历史数据可能只有 kind 没有 lane 标）。

    消费者（MQ-S27 拍板 2026-08-20 = 硬分流）：归属管线前置过滤
    （memory_index 归属循环，`BLADEX_LANE_HARD_SPLIT`）。
    """
    kind = item_kind.value if isinstance(item_kind, ItemKind) else str(item_kind or "")
    for t in (tags or "").split(","):
        t = t.strip()
        if t in ("lane:task", "lane:profile", "lane:tool"):
            return t[5:]
    if kind == ItemKind.PROFILE_OBS.value:
        return "profile"
    return ""


class ConversationTurn(BaseModel):
    """从 Memory Hub Turn 归一出的对话轮次（agent 中立，供 consolidation 消费）。

    proxy 侧把 Turn.request_messages + response_text 归一成此结构。
    """

    session_id: str = ""
    user_id: str = ""
    user_messages: list[str] = Field(default_factory=list)
    assistant_response: str = ""
    # ADR-0019 P2：任务结论（该轮 response 为最终回答——status=ok、有正文、无工具调用
    # ——时由 proxy 侧填入；空 = 中间往返/失败轮，consolidation 不蒸结论）。
    # 结论即蒸馏的闭环：装配器降解掉的 tool 证据，其关键信息经结论进入 Memory Index 可检索。
    assistant_conclusion: str = ""

    # 工作产出（2026-07-29 新增）：status=ok、有正文、**且本轮带工具调用**时的
    # assistant 正文——即模型在调工具前写的那段说明/分析/发现。
    #
    # 为什么必须单开一条通道：结论通道要求"无 tool call"，而编程 agent 全程泡在
    # 工具循环里（实测 codex 37 轮中 35 轮带 tool call、只有 2 轮符合结论条件），
    # 于是 Memory Index 里 18 条 fact 有 16 条是"用户询问 XXX"——记住了"问了什么"，
    # 几乎没记"做了什么、发现了什么"。而对编程 agent 最有价值的恰恰是后者。
    #
    # 与 assistant_conclusion 互斥：同一轮要么是结论（无 tool call）要么是产出（有）。
    assistant_progress: str = ""
    logical_turn: int | None = None
    roundtrip: int = 0
    ledger_key: str = ""
    timestamp: str = ""
    # ADR-0024 §4.5 / U4：本轮当前（open）TaskUnit 的 unit_key。
    # proxy 侧从 Memory Hub Turn.task_units[-1] 填入 → 蒸出的 facts 继承 → 归属 L0「同单元同属」。
    # 本轮新增 user 内容属当前单元；历史消息在后续轮走 novelty 去重不重复产出，故用
    # 当前单元 key 标注本轮候选是正确的（多单元误标风险由 dedup 兜住，见 consolidation）。
    unit_key: str = ""
    # agent_id：participants 聚合用（U4 Matter 卡 touch_participant）。
    agent_id: str = ""
    # ADR-0026 §4.3 / U4B-B3：本轮 tool 事件里的文件指针 [{path, content_hash}]。
    # proxy 侧从 Memory Hub Turn.tool_events 确定性提取；consolidation 产出 file_ref 条目（零 LLM）。
    tool_file_refs: list[dict[str, str]] = Field(default_factory=list)
    # ADR-0021 §3.3c：本轮敏感等级（身份解析时产出的静态等级，随 Turn 入 Memory Hub，重建可重放）。
    # consolidator 据此给蒸馏出的 Fact 设 exposure_ceiling（血统继承）。默认 normal = 现状。
    sensitivity: str = "normal"

    model_config = {"extra": "allow"}


class Fact(BaseModel):
    """蒸馏出的一条结构化事实（存 Memory Index / 派生，护城河底座）。

    一条 Fact 代表从对话中提炼出的、值得长期记住的一条信息。
    语义向量由 Memory Index 层的 embedder 生成、存入 LanceDB。
    """

    id: str = ""
    content: str
    category: str = "general"
    tags: str = ""
    source_session: str = ""
    source_user_id: str = ""
    # ADR-0026 §2.1 改名：字段本身叫 source_ledger_key，但**落盘过的旧行写的是
    # source_p3_key** —— Fact 经 model_dump 存进 Memory Index meta 库，直接改名会让
    # 存量事实读回来 provenance 全空（墓碑级联、评测集构建、重建溯源都依赖它）。
    # 故保留读侧别名：旧行照常读，新行写新名，被改写的行自然迁移。别删这个别名。
    source_ledger_key: str = Field(
        default="",
        validation_alias=AliasChoices("source_ledger_key", "source_p3_key"),
    )
    logical_turn: int | None = None
    trust: float = 1.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    # ADR-0027 §5.3：最后一次变更时间。Fact 不是只写一次的——非破坏取代（t_invalid /
    # superseded_by）、scope 提升、importance/strength/ref_count 更新都会改它，而导出
    # 增量游标此前只看 created_at，**这些变更永不重导**：Obsidian/PostgreSQL 里被取代
    # 的事实会持续显示为现行，与"记忆一致性是护城河"直接冲突。
    # 加法式演进：历史数据无此字段 → 默认等于 created_at，无需迁移。
    updated_at: datetime | None = None

    # L2 (ADR-0014): 蒸馏元数据
    # source_text: 蒸馏前的原始 user 消息（审计/回溯用，空 = 未蒸馏）
    # distill_model: 蒸馏使用的模型名（空 = 未蒸馏，"passthrough" = 透传）
    source_text: str = ""
    distill_model: str = ""
    # ADR-0018 §3.1: 蒸馏结构化分类 + 实体锚点（T7 五层链接归属用）
    kind: str = "general"            # preference | event | task | decision | general
    entities: list[str] = Field(default_factory=list)
    # ADR-0024 §4.5 / U3：归属 L0「同单元同属」的分组键。
    # 同一 TaskUnit 蒸出的 facts 共享此 unit_key（内容派生、跨轮稳定），归属管线据此
    # 确定性同归一个 Matter、跳过 LLM 裁决。空 = 历史/未带单元的 fact → 不分组（现状）。
    # 消费者：AttributionPipeline L0（U3）。生产者：consolidation 按单元蒸馏（U4 populate）。
    unit_key: str = ""

    # ── ADR-0026 §4.2 / U4：条目六类 + 三轴坐标 + 双时态 + 重要性 ──
    # 加法式演进（历史 Fact 走默认值）。消费者：检索 U6 / 注入 U7 / 生命周期 U5。
    # item_kind: 六值封闭枚举（assertion/preference/procedure/lesson/file_ref/profile_obs）。
    item_kind: ItemKind = ItemKind.ASSERTION
    # subject·attribute: 取代键的两半（同 (scope,subject,attribute) 新压旧 = O(1)，机制3/U5）。
    subject: str = ""
    attribute: str = ""
    # audience: 轴A 相关性维度（安全维度是 scope，二者不混用一个字段）。
    # all | agent:<base>（盖住该 agent 全部 profile）| agent:<base:profile>（只此 profile）。
    # 分层匹配见 prefetch 过滤点（2026-08-29 拍板：profile=独立 agent）。
    audience: str = "all"
    # matter_id: 归属结果直接落字段（L0/L1-L5 写入），检索边界硬过滤可用（轴B）。
    matter_id: str = ""
    # 轴C 双时态（参照 Graphiti/Memanto）：
    #   t_observed = 摄入时间（= created_at）；t_valid = 业务有效起点；
    #   t_invalid = 失效时间（supersede/确定性失效写入，空 = current 仍成立）；
    #   superseded_by = 取代它的新条目 id（非破坏取代，旧条保留供 as-of）。
    t_observed: datetime | None = None
    t_valid: datetime | None = None
    t_invalid: datetime | None = None
    superseded_by: str = ""
    # importance: 重要性评分（取代恒 1.0 的 trust；U5 公式 f(kind基线,strength,引用计数,半衰期)）。
    #   默认沿用 trust 语义 1.0，检索排序 + 生命周期驱逐消费（U5/U6）。
    importance: float = 1.0
    # strength: 合并计数（相近 preference/assertion 合为一条时 +1，U5 机制2）。
    strength: int = 1
    # ref_count: 注入命中回写计数（U7 注入侧生产 → U5.4 importance 引用增益消费）。
    # proxy 进程 Memory Index 只读 → 生产端在可写场景（评估回放/嵌入式）生效，只读时 no-op。
    #
    # 🔴 **口径（MQ-S38，2026-08-20）：它不是"全历史累计"**。命中信号按
    # `fact_id` 回写，而 fact_id 是内容派生的（`_deterministic_fact_id`）——
    # 蒸馏 prompt 换版会让全库 id 换代，历史命中随之对不上。
    # 带取代键的条目在回写时能按键接续（唯一匹配才接）；**无取代键的
    # （lesson / procedure——它们累积、不互相取代）在每次换版归零**。
    # 读它的人要按"**自最近一次蒸馏换版以来（无键条目）**"理解，
    # 别当成"这条一辈子被用过几次"——那会系统性低估老条目。
    # 换版丢了多少有读数：`inject_refcount_orphan_ids` / `index_rebuild_done`。
    ref_count: int = 0
    # ── M1 数据结构变更总表（2026-08-06 记忆核心链改造）────────────────────
    # provenance: 来源通道（user_direct|conclusion|progress|document|scaffolding）。
    #   生产者 = M1-2 蒸馏装配；消费者 = 检索配额 / importance / 审计。
    #   ""（默认）= 历史数据未知来源——消费者必须能处理未知，不许当成某一类。
    provenance: str = ""
    # importance_rating: 蒸馏器给的**内容内在分**（1–10）。
    #   0 = 未评分 → importance 公式退回 kind 基线（M1-5）。
    #   为什么要它：kind 基线只能区分"偏好比一般结论重要"，区分不了
    #   "这条 assertion 是核心决定" 与 "这条 assertion 是顺嘴一提"。
    importance_rating: int = 0
    # valid_until: 时效性记忆的到期点（蒸馏 ⑦槽位产出）。
    #   到期后由 M3-3 生命周期 job 置 t_invalid（退出 current-only 召回，数据不删）。
    #   None = 无到期（绝大多数条目）。
    valid_until: datetime | None = None
    # corrects: 这条**自称**修正了哪条旧记忆（自然语言描述，非 fact_id——
    #   蒸馏时模型看不到库）。M2 裁决器拿它当 corrects_hint 在邻居里找取代目标。
    #   复核 F4 把它叫作"最高精度信号"：一条自带"此前理解错了"的新事实，
    #   在裁决输入里是**明文证据**，比任何相似度都硬。
    #   保留在 Fact 上而不是只做瞬态：`superseded_by` 记的是取代的**结果**，
    #   这里记的是**主张**，两者都要有才说得清一条取代链当初为什么成立。
    corrects: str = ""
    # last_hit_at: **相关钟**——最后一次被注入且落地的时间（M3-1 回写）。
    #   与 updated_at（更新钟）正交：一条很久没变但天天被用到的记忆不该被判休眠。
    #   None = 从未命中（也可能是相关钟接线之前的历史数据，故 M3-2 有覆盖率护栏）。
    #   🔴 同 `ref_count` 的口径注记（MQ-S38）：无取代键的条目在蒸馏换版后
    #   会退回 None——"从未命中"与"命中记录随换版丢了"在这个字段上**同形**，
    #   判 dormant 前先看那一轮重建的 orphan 读数。
    last_hit_at: datetime | None = None

    # ADR-0018 §3.1/§3.2: 本消息的 Matter 提案标题（共享，T7 L3/L5 消费）。
    # 一条消息的多条 facts 共享同一组 proposals；L5 新开 Matter 标题取首个 proposal。
    proposal_titles: list[str] = Field(default_factory=list)

    # ── 主题键原料（2026-08-14 拍板 A）：轮级名词性 keywords（蒸馏 v5 产出，
    # 解析侧广播到该轮每条 fact，与 topic 同款继承式）。meta 层字段不进
    # LanceDB 列；消费者 = 归属 L3 主题键子层 + Matter.topic_keys 聚合
    # （topic_keys.derive_topic_keys 对空值做确定性派生兜底，存量 fact 不失效）。──
    keywords: list[str] = Field(default_factory=list)

    # ── 三段式蒸馏 T2（2026-08-10）：轮级主题（一轮一 topic，继承式）──
    # **meta 层字段**；🔴 LanceDB facts 表不加列（向量平面不需要，加列要走
    # schema 迁移收益为零）。词法侧进 FTS topic 列（fts_index schema v2）。
    # 拍板 P1：topic 只做三用途——检索轴 / 归属信号(T4b) / 注入分组标题(T4b)；
    # 不当 Matter 主键、不参与 fact id、不参与任何 identity 判定。
    # 历史 Fact 默认 ""（加法式演进）。
    topic: str = ""

    # ADR-0021 §2.4 / §3.3c：可见性 + 敏感血统（记忆平面数据流控制）。
    # scope: 可见范围 `personal:<pid>` | `team:<tid>` | `org:<oid>`，默认 personal
    #   （存量数据自动归此，无需迁移）。Memory Index 检索按 scope ∈ 请求者可见集合过滤。
    # exposure_ceiling: 该 Fact 允许到达的最高暴露等级（local/private/public），
    #   继承来源 Turn 的敏感等级映射（血统继承）。默认 public = 现状（个人流量发公网）。
    #   注入守卫：ceiling >= 请求 allowed_exposure 才注入（敏感 Fact 不注入公网请求）。
    scope: str = ""
    exposure_ceiling: str = "public"

    # 嵌入向量（不持久化到 LanceDB 的 metadata，仅传递用）
    embedding: list[float] | None = None

    # populate_by_name：source_ledger_key 有读侧别名，构造时仍用字段名传参。
    model_config = {"extra": "allow", "populate_by_name": True}


class HardRule(BaseModel):
    """硬规则（MUST/NEVER），每轮无条件注入、不参与相关性截断/衰减。

    领域词汇表：HardRule 禁止叫 Constraint/Policy。
    trust=1.0、category='hard_rule'，在 Memory Index 中钉在内存热区。
    """

    content: str
    fact_id: str = ""
    trust: float = 1.0

    model_config = {"extra": "allow"}


# ── 人设事实的 audience 派生（MQ-L27，2026-08-29 Jason 拍板）────────────────
#
# live 病例：codex 被注入 hermes 的「用户称呼助手为"小黑"」后自称小黑。
# 全库 8038 条实测：结构化 subject·attribute 闭集判据命中 24、真 22 误 2，
# 两条误报由负向词（占位）与收窄模式（去掉 agent/assistant 半边）消除。
# 行事记录类（"小黑于某日重建 Pi 配置"）**有意不拦**——工作事实跨 agent
# 共享正是价值，只有**定义性**人设事实（称呼/自称/人设配置）才该锁定来源。

import re as _re

#: 人设定义类判据（吃结构化字段不吃全文；判别力实测见 MQ-L27）。
_PERSONA_PAT = _re.compile(r"称呼|昵称|人设|自称|打招呼")
_PERSONA_NEG = _re.compile(r"占位")


def is_persona_fact(subject: str, attribute: str) -> bool:
    """该 fact 是否为**定义性**人设事实（助手称呼/自称/人设配置）。

    确定性零 LLM；只看蒸馏产出的 subject·attribute 结构化字段。
    """
    joined = f"{subject or ''}|{attribute or ''}"
    return bool(_PERSONA_PAT.search(joined)) and not _PERSONA_NEG.search(joined)


def source_agent(fact: "Fact") -> str:
    """从 fact 的来源 Hub key（`<principal>/<agent>/<sess>/<entry>`）解出
    来源 agent 的**完整 id（含 profile）**。解不出返回空串（不猜）。

    2026-08-29 Jason 拍板：有独立配置文件的 profile 就是独立 agent
    （hermes:default 与 hermes:accept 的人设各归各，kawaii 是 accept 的）。
    """
    key = fact.source_ledger_key or ""
    parts = key.split("/")
    if len(parts) < 4 or not parts[1]:
        return ""
    return parts[1]


def effective_audience(fact: "Fact") -> str:
    """注入面生效的 audience（MQ-L27 兜底派生）。

    已显式标注 → 原样；未标注（"all"）但是定义性人设事实 → 派生为
    `agent:<来源完整 id>`（人设跟随其主人——含 profile，profile=独立 agent；
    不随共享记忆传送）；其余照旧 "all"。
    蒸馏侧出生即标注（MQ-L27 修法①）落地后，本派生自然只剩兜底作用。
    """
    aud = getattr(fact, "audience", "all") or "all"
    if aud != "all":
        return aud
    if is_persona_fact(getattr(fact, "subject", ""), getattr(fact, "attribute", "")):
        src = source_agent(fact)
        if src:
            return f"agent:{src}"       # 完整 id 含 profile（profile=独立 agent）
    return aud
