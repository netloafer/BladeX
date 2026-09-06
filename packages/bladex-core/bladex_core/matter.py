"""Matter — 一件周级可完结的"事儿"，跨 session/project/agent 的逻辑单元（ADR-0012 §3.1）。

领域词汇表：Matter 禁止叫 Task/Topic/Thread/神经网络。
agent 中立：不依赖 proxy 或任何 agent 的类型。
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field


class MatterStatus(str, Enum):
    """Matter 生命周期状态（ADR-0012 §3.4 + ADR-0018 §3.3）。

    PROVISIONAL 是 ADR-0018 新增的胚芽态：新 Matter 从 proposal 诞生即 provisional，
    达临界质量（≥2 成员或 manual 触碰）才转 established(ACTIVE) 进注入。
    """

    PROVISIONAL = "provisional"  # 胚芽：未达临界质量，不进 prefetch 注入
    ACTIVE = "active"    # established：近期有归属写入、进注入
    DORMANT = "dormant"  # 超时无活动自动转入
    CLOSED = "closed"    # 用户手动关闭或明确完结信号


# Matter.entities 封顶（按频次淘汰的 v1 简化：超限 FIFO 丢弃，留实测换频次淘汰）
_MAX_MATTER_ENTITIES = 20


class MatterOrigin(str, Enum):
    """Matter 创建来源。"""

    AUTO = "auto"      # 归属管线自动创建
    MANUAL = "manual"  # 用户手动创建


class EdgeProvenance(str, Enum):
    """归属边的来源（ADR-0012 §3.5 manual 压过 auto）。"""

    AUTO = "auto"      # 管线判定（含置信度）
    MANUAL = "manual"  # 用户指定（重聚类不得移动）


class EdgeTargetType(str, Enum):
    """归属边的目标类型（ADR-0024 §4.6 扩展 MATTER/UNIT）。"""

    SESSION = "session"  # matter ↔ session
    FACT = "fact"        # matter ↔ fact
    MATTER = "matter"    # matter ↔ matter（Matter↔Matter 关系，如 FOLLOWS/PART_OF）
    UNIT = "unit"        # matter ↔ TaskUnit（L0 的落点）


class EdgeRelation(str, Enum):
    """归属边的关系类型（ADR-0024 §4.6）——把二部星型拓扑扩成类型化网络。

    存量边默认 BELONGS（= 现有语义），无需数据迁移。
    """

    BELONGS = "belongs"        # 现有语义（Matter→Fact/Session）
    SUPERSEDES = "supersedes"  # Fact→Fact：新事实取代旧（consistency + 取代键落点，U5.2）
    ELABORATES = "elaborates"  # Fact→Fact：同主题展开/细化
    FOLLOWS = "follows"        # Matter→Matter：时间/因果后继
    PART_OF = "part_of"        # Matter→Matter：子事归属大事（Project 脉络）


# ADR-0026 §4.1：Matter 卡 lifecycle 封顶（受控，超限 FIFO 丢最旧，保重建确定性）
_MAX_MATTER_LIFECYCLE = 50


class MatterParticipant(BaseModel):
    """Matter 卡 participants 成员（ADR-0026 §4.1）——「这卡之前谁开发的」。

    确定性从归属边/Turn.agent_id 聚合，可从 Memory Hub 零 LLM 重建。
    """

    agent_id: str
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    turns: int = 0


class MatterLifecycleEvent(BaseModel):
    """Matter 卡 lifecycle 事件（ADR-0026 §4.1）——「卡在哪、经历了什么」。

    折叠会话事件（铁律3：会话事件不产出为条目，归此）。确定性重算。
    """

    ts: datetime | None = None
    event: str = ""    # created | assigned | reworked | blocked | closed | ...
    detail: str = ""


class Matter(BaseModel):
    """Matter 卡 — 一件事儿的元信息（ADR-0012 §3.1 + ADR-0026 §4.1 扩展）。

    一个 Matter ≈ "给 BladeX 加 /v1/messages 支持"这种量级：
    有明确起止、周级时间尺度、可以"办完"。
    """

    matter_id: str
    title: str = ""
    summary: str = ""
    # ADR-0018 §3.4: 摘要来源标记。"concat"=保底拼接，"llm_judge"=L4 裁决顺带重写。
    summary_source: str = "concat"
    status: MatterStatus = MatterStatus.ACTIVE
    centroid: list[float] | None = None  # e5 语义质心（仅召回索引，不参与红线判定，ADR-0018 §3.4）
    centroid_weight: float = 0.0  # 质心累积权重（T2 加权更新用，manual 边权重 > auto）
    origin: MatterOrigin = MatterOrigin.AUTO
    # V-L6（ADR-0032 §4.5）：账本↔Matter 绑定——本卡由哪本账本锚定（外部边界锚点）。
    # 追加字段，历史 Matter 恒空，零迁移。消费者 = memory_index 账本锚层
    # （_la_matter_by_ledger 索引）+ dashboard 展示。
    ledger_id: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # ADR-0018 §3.4 + MS-16（2026-08-10）: L2/L3 匹配键 + L4 证据
    # aliases: **原生键**——创建时标题 + 首批成员实体/提案标题 + manual 追加。
    #   MS-16 起不再随成员 proposal_titles 自动累积（累积并集是归属雪球源，
    #   见 docs/reviews/matter-attribution-audit-20260810.md）。
    # entities: 成员 facts entities 的并集（封顶 _MAX_MATTER_ENTITIES）——
    #   仅展示/观察，已退出 L2 门控与 L3 匹配面。
    # 用 list[str] 而非 set 保证序列化/重建顺序确定（T12d 重建等价性）。
    aliases: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    # topic_keys（2026-08-14 拍板 A）：成员 fact 主题键计票（key → 出现次数）。
    # 与 aliases 的分工：aliases = 原生键（创建时 + manual），全串相等匹配；
    # topic_keys = 名词不变量，**多键重叠**才命中（≥3 或 ≥2 且 Jaccard≥0.6，
    # 见 topic_keys.is_strong_topic_match）——单键进表不构成 MS-16 那种吸尘器，
    # 且有界（top-30 计票裁剪）。消费者：归属 L3 子层 + L5 同胞拦截 + 合并提案检测。
    topic_keys: dict[str, int] = Field(default_factory=dict)

    # ADR-0021 §2.4：可见范围 `personal:<pid>` | `team:<tid>` | `org:<oid>`，默认 personal。
    # scope 提升（personal->team/org）是显式管理操作，记 ADR-0012 管理事件 journal
    # （SCOPE_PROMOTE），Memory Index 重建重放。默认 personal = 存量数据自动归此，无需迁移。
    scope: str = ""

    # 嵌入向量（仅传递用，存 LanceDB 时不持久化到 metadata）
    embedding: list[float] | None = None

    # ── ADR-0026 §4.1 / U4：Matter 卡扩展（全部确定性可从 Memory Hub 重算）──
    # participants: 「谁做过这卡」（交接原语，注入②平面 + dashboard 消费）。
    # lifecycle: 「卡经历了什么/卡在哪」（会话事件折叠于此，封顶 _MAX_MATTER_LIFECYCLE）。
    # open_issues: 当前卡点（T12 落点）。
    # version: 每次归属写入 +1（一致性三规则 + changed-since 交接，U7 消费）。
    participants: list[MatterParticipant] = Field(default_factory=list)
    lifecycle: list[MatterLifecycleEvent] = Field(default_factory=list)
    open_issues: list[str] = Field(default_factory=list)
    version: int = 0
    # M0-10（复核 H2②）：已折叠事件的内容指纹，**随卡持久化**。
    # 去重此前只在单次进程内生效，于是 rebuild 每重放一次同一段 Memory Hub，
    # 同一个事件就再折叠一遍——卡上 lifecycle 越滚越长、version 虚高，
    # 而 version 是 U7 ③平面 changed-since 的判据（虚高 = 每轮都判"有变更"）。
    # 加法式演进：历史 Matter 无此字段 → 默认空列表，首次 record 后自然填充。
    lifecycle_seen: list[str] = Field(default_factory=list)

    # ── G12.3 `task_goal`（ADR-0031 §4.1「取」不是「炼」）────────────────
    # 🔴 **只有用户能改**：唯一写入点 = 新开卡那一刻（memory_index._apply_decision），
    # 且仅当 task_goal 为空。任何 update / 蒸馏 / 归属 / 重建 / merge 路径**不得改写
    # 非空 task_goal**（判据 D1–D3）；用户覆盖发生在渲染层、不回写这里（D4）。
    # 模型能改目标 ⇒ 防漂移自我拆台——漂移的模型会把目标改成它正在做的事再声称没漂。
    # task_goal      : 首轮用户原文（剥信封后）**原样引用，不截断**——注入体积由渲染层
    #                  按 §6.7「goal 一行」预算处理（600 字符，G12.5 落点）。
    # task_goal_source: `bladex_core.task_goal.GOAL_SOURCES` 之一，空 = 未取到。
    # task_goal_reason: 未取到的分档原因 `GOAL_ABSENT_REASONS`——它同时是
    #                  "有多少卡是被机器文本开出来的"这把尺子的读数面。
    # task_goal_ledger_key: 首轮 ledger key，用户可回查原文、审计可复核、跨重建稳定。
    # 历史 Matter 无此四字段 → 默认空，加法式演进无需迁移。
    task_goal: str = ""
    task_goal_source: str = ""
    task_goal_reason: str = ""
    task_goal_ledger_key: str = ""

    # M3-1 相关钟：最后一次这张卡被注入（钉卡/增量）的时间。
    # 与 updated_at（更新钟）正交——M3-2 的 2×2 判定要两个钟同时停摆才判 DORMANT，
    # 否则"一件已定型但天天被翻出来参考的事"会被当成没人管的事关掉。
    # None = 从未命中；相关钟接线（M3-1）之前的历史数据同样是 None，
    # 故 M3-2 有"全库覆盖率 < 50% 只 log 不动状态"的观测期护栏。
    last_hit_at: datetime | None = None

    def lifecycle_fingerprint(self, event: str, detail: str = "") -> str:
        """事件内容指纹（M0-10）。**不含时间戳**——同一个事件在两次重放里
        ts 必然不同，把 ts 算进指纹等于没去重。"""
        import hashlib

        raw = f"{event}\x00{detail}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]

    def record_lifecycle(self, event: str, detail: str = "", ts: datetime | None = None) -> bool:
        """追加一条 lifecycle 事件（封顶 FIFO），并 version+1。确定性、零 LLM。

        M0-10：同内容事件只折叠一次（持久化去重）。重复事件**整段跳过**——
        既不追加也不 bump version，否则重放仍会把 changed-since 判据推高。

        返回 True = 真的折叠了一条；False = 判重跳过（调用方据此决定要不要落盘）。
        旧调用方忽略返回值，行为不受影响。
        """
        fp = self.lifecycle_fingerprint(event, detail)
        if fp in self.lifecycle_seen:
            return False
        self.lifecycle.append(MatterLifecycleEvent(
            ts=ts or datetime.now(UTC), event=event, detail=detail,
        ))
        self.lifecycle_seen.append(fp)
        if len(self.lifecycle) > _MAX_MATTER_LIFECYCLE:
            self.lifecycle = self.lifecycle[-_MAX_MATTER_LIFECYCLE:]
        # 指纹表与 lifecycle 同步封顶：被 FIFO 挤掉的事件重新变得"没见过"，
        # 这是刻意的——上限内保证不重复，上限外让位给"最近 N 条真实存在"。
        if len(self.lifecycle_seen) > _MAX_MATTER_LIFECYCLE:
            self.lifecycle_seen = self.lifecycle_seen[-_MAX_MATTER_LIFECYCLE:]
        self.version += 1
        return True

    def touch_participant(self, agent_id: str, ts: datetime | None = None) -> None:
        """归属写入时更新 participant 计数（确定性聚合，「谁开发的」）。"""
        ts = ts or datetime.now(UTC)
        for p in self.participants:
            if p.agent_id == agent_id:
                p.last_seen = ts
                p.turns += 1
                return
        self.participants.append(MatterParticipant(
            agent_id=agent_id, first_seen=ts, last_seen=ts, turns=1,
        ))

    model_config = {"extra": "allow"}


class MatterEdge(BaseModel):
    """归属边 — Matter 与 session/fact 的关联（ADR-0012 §3.1）。

    每条边带 provenance：auto（管线判定，含置信度）或 manual（用户指定）。
    manual 边重聚类不得移动，且作为质心强锚点（ADR-0012 §3.5）。
    """

    edge_id: str = ""
    matter_id: str
    target_type: EdgeTargetType
    target_key: str  # session key（user/agent/session/）或 fact_id
    # ADR-0024 §4.6：关系类型（存量边默认 BELONGS = 现有语义，无需迁移）。
    relation: EdgeRelation = EdgeRelation.BELONGS
    weight: float = 1.0
    provenance: EdgeProvenance = EdgeProvenance.AUTO
    confidence: float = 0.0  # auto 边的判定置信度
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    # ADR-0018 §3.5: auto 边的决策记录（可解释性对齐路由 RouteSource）。
    # {layer: L1..L5, candidates, verdict, judge_model, distill_model, ts}；manual 边为空。
    decision: dict = Field(default_factory=dict)

    model_config = {"extra": "allow"}


class MatterAggregateView(BaseModel):
    """Matter 的聚合视图 — 从归属边 + session 元数据即时聚合（ADR-0012 §3.1）。

    不落独立存储，每次查询时从归属边聚合。
    """

    matter_id: str
    title: str = ""
    summary: str = ""
    status: MatterStatus = MatterStatus.ACTIVE
    session_keys: list[str] = Field(default_factory=list)
    fact_ids: list[str] = Field(default_factory=list)
    edge_count: int = 0
    manual_edge_count: int = 0
