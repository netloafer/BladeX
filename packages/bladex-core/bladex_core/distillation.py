"""蒸馏接口 - 从原始 user 消息抽取原子事实 + Matter 提案（ADR-0014 L2 + ADR-0018 §3.1）。

consolidation 的原语错误：当前 Fact = 原始 user 消息，用"措辞像不像"
代替"是不是同一件事"。蒸馏后 Fact = 单句陈述（"用户询问世界杯赛程"），
dedup/聚类才有意义，系统提示类垃圾在构造上进不来。

ADR-0018 §3.1：蒸馏输出结构化 {facts[{content,kind,entities}], matter_proposals[≤3]}。
kind=junk 在构造上不入库；proposals 是归属映射的输入（T7 消费）。

agent 中立：本模块只定义协议 + domain 对象 + 透传实现，LLM 实现在 proxy 侧。
"""

from __future__ import annotations

import logging
from typing import Protocol

from pydantic import BaseModel, Field

from bladex_core.fact import Provenance

logger = logging.getLogger(__name__)

# ── M1-1：context_digest 的确定性装配（重建等价性 G6 依赖它）─────────────────
#
# 🔴 为什么 digest 只许取**这一轮自己带的**历史，不许取"最近 K 个 turn"：
# 台账 key 含 digest（M1-3），而 rebuild 的批次窗口每次都不一样
# （增量 30 条一批 / 全量一次过 / 定向重放某窗口）。若 digest 取自批次邻居，
# 同一轮在不同批次里会算出不同 key → 台账恒 miss → "rebuild 复跑零现蒸" 破功，
# 每次重建都要把 LLM 钱重烧一遍。而 `Turn.request_messages` 里本来就带着完整历史，
# 它是这条轮次的**内在属性**，跨批次恒定。
CONTEXT_DIGEST_MAX_CHARS = 500
_CONTEXT_DIGEST_TURNS = 3
_CONTEXT_DIGEST_PER_TURN = 160


def build_context_digest(
    prior_user_texts: list[str],
    matter_summary: str = "",
    *,
    max_chars: int = CONTEXT_DIGEST_MAX_CHARS,
) -> str:
    """装配 `context_digest`（≤max_chars）。纯函数、确定性、零 LLM。

    优先级（M1-2 卡）：当前 Matter 卡 summary > 最近 K 轮 user 文本截断。
    Matter summary 优先的理由：它已经是"这件事是什么"的浓缩，比几段原始对话
    更能给蒸馏器定位；拿不到（冷启动 / 未归属）才退回原文。

    `prior_user_texts` 必须是**已剥信封**的文本，且不含当前这条消息。
    """
    summary = (matter_summary or "").strip()
    if summary:
        return summary[:max_chars]
    picked = [t.strip() for t in prior_user_texts if t and t.strip()]
    if not picked:
        return ""
    tail = picked[-_CONTEXT_DIGEST_TURNS:]
    joined = " / ".join(t[:_CONTEXT_DIGEST_PER_TURN] for t in tail)
    return joined[:max_chars]


# ── ADR-0018 §3.1: 蒸馏结构化输出 domain 对象（agent 中立，bladex_proxy 引用）──


class DistillFact(BaseModel):
    """蒸馏出的一条结构化事实。

    kind=junk 在 LLMDistiller 构造上不入库（调用方拦截），台账/facts 里只存非 junk。
    kind=general 表示未分类（passthrough 透传用）。
    """
    content: str
    kind: str = "general"   # preference | event | task | decision | junk | general
    entities: list[str] = Field(default_factory=list)
    # ── ADR-0026 §4.2 / U4：条目六类 + 取代键槽位（蒸馏 prompt 产出，加法式）──
    # item_kind: assertion|preference|procedure|lesson|file_ref|profile_obs（空/general = 未分类，
    #   由调用方按旧 kind 兜底映射）。subject·attribute: 取代键两半（assertion/preference 产出）。
    item_kind: str = ""
    subject: str = ""
    attribute: str = ""

    # ── M1-1（prompt v4）：四个新槽位 + 来源通道 ────────────────────────────
    # importance: 内容内在重要性 1–10（0 = 模型没给分 → importance 公式退回 kind 基线）。
    importance: int = 0
    # valid_until: 时效性到期点（ISO8601 日期或日期时间字符串；"" = 无到期）。
    #   prompt 里给的是 turn_time 这个**已知量**，所以模型能把"下周三"算成具体日期
    #   ——时间锚是 v4 的硬要求（v3 产出里大量"某日"，检索与到期都无从判定）。
    valid_until: str = ""
    # corrects: 被本条修正的旧记忆的**自然语言描述**（"" = 没在修正谁）。
    #   它不是 fact_id——蒸馏时模型看不到库。M2 裁决器拿它当 corrects_hint，
    #   在邻居集合里找该被 UPDATE 的目标。
    corrects: str = ""
    # event: 任务事件分类 assigned|reworked|blocked|completed|none（"" 视同 none）。
    #   铁律3：会话事件不产出为条目，折叠进 Matter 卡 lifecycle（MS-4 消费）。
    event: str = ""
    # provenance: 这条来自哪个通道（user_direct|conclusion|progress|...）。
    #   v4 一轮一次统一调用（user+assistant 同包），所以**必须由产出侧标注来源**，
    #   否则通道信息在合流那一刻就丢了（v3 靠三条独立流水线区分，v4 靠这个字段）。
    provenance: str = ""
    # tags: 自由标记位（逗号分隔，形如 `origin:conclusion,lang:drift`）。
    #   2026-08-08：这个字段本来不存在，而 `distill_fidelity.mark_language_drift`
    #   直接写 `f.tags = ...` —— pydantic v2 对未声明字段的赋值抛 ValueError，
    #   整轮蒸馏 `return empty`。全量重建 784 次调用里 **163 次死在这**。
    #   没被任何测试碰过（`TAG_LANG_DRIFT` 在测试里零引用），所以 gate 一直全绿。
    #   读侧写的是 `getattr(f, "tags", "")`，把"字段不存在"掩盖成了空串 ——
    #   **防御性读 + 裸写 = 最难发现的组合**：读不报错，写才炸，而写在少数分支里。
    tags: str = ""
    # topic: 轮级主题（三段式 v5，一轮一 topic，解析侧广播到该轮每条 fact）。
    #   🔴 必须落在 DistillFact 上（与 event 同款理由，见 _parse_distill_json）：
    #   台账 DistillRecord 只持久化 facts + matter_proposals，挂在 DistillOutput
    #   上的字段在台账命中路径会丢 → 首蒸与缓存复用产出不一致 → G6 重建等价破。
    topic: str = ""
    # keywords: 轮级名词性关键词（v5 就在产出，此前定为瞬态被丢弃）。
    #   2026-08-14 拍板 A 转持久：Matter 去重与 Matter↔Fact 连接的名词不变量
    #   （标题动词漂移下唯一稳定的匹配信号，见 matter-dedup-keywords 卡）。
    #   广播与台账穿越理由同 topic。加法式：历史台账反序列化 = []（消费侧
    #   有确定性派生兜底 derive_topic_keys，存量数据不失效）。
    keywords: list[str] = Field(default_factory=list)


class DistillTurnInput(BaseModel):
    """M1-1：一轮的结构化蒸馏输入（v4 统一调用的载荷）。

    v3 有三条独立流水线（user / conclusion / progress），各自不知道对方存在：
    同一轮里"用户要求 X"和"我做完了 X，结论是 Y"被当成两件互不相干的事去蒸，
    于是产出大量"用户询问 XXX"型条目而丢掉真正的结论。v4 合成一包送进去。

    字段都是**已知量**，模型不需要猜：
      - context_digest: 这轮之前在聊什么（≤500 字，确定性装配，见 build_context_digest）
      - turn_time:      这轮发生的时间（ISO8601）——时间锚，禁止产出"某日"
      - user_text:      用户这轮真的说了什么（已剥信封）
      - assistant_text: 模型这轮的结论或工作产出（已剥信封）
      - assistant_role: assistant_text 属于哪个通道（conclusion | progress | ""）
    """

    context_digest: str = ""
    turn_time: str = ""
    user_text: str = ""
    assistant_text: str = ""
    assistant_role: str = ""

    def cache_text(self) -> str:
        """台账 key 用的**输入正文**（M1-3：与 context_digest 一起 hash）。

        只含"这轮说了什么"，不含 context——context 走 key 的另一半，
        这样同一句话在不同上下文下是不同的 key（v3 的 key 只看正文，
        于是同一句"继续"在任何语境里都命中同一条缓存）。
        """
        parts = [self.user_text.strip()]
        if self.assistant_text.strip():
            parts.append(f"[{self.assistant_role or 'assistant'}]\n{self.assistant_text.strip()}")
        return "\n\n".join(p for p in parts if p)


class MatterProposal(BaseModel):
    """蒸馏顺带提出的"这属于哪件事儿"提案（≤3 个/消息）。

    新 Matter 从提案诞生（标题=提案标题），是归属映射的输入（T7 消费）。
    """
    title: str
    entities: list[str] = Field(default_factory=list)


class DistillOutput(BaseModel):
    """一次蒸馏的完整输出（facts + proposals + 轮级 topic/keywords）。

    一条消息的多条 facts 共享该消息的 proposals。

    三段式 v5（T1b/T2）：
      - topic:    一句话核心主题（≤40 字，配置语言）。持久化路径 = 广播到每条
        DistillFact.topic（台账只存 facts，见 DistillFact.topic 注释）；本字段
        供解析契约与无 facts 时的观测，**不承诺穿越台账命中路径**。
      - keywords: ≤10 个名词性关键词（跨语言归一到配置语言，标识符原样）。
        瞬态：不入台账不入 Fact——fact 级判别信号仍是 entities（prompt 要求
        从轮级 keywords 中选子集），检索词法面由 FTS content/entities/topic 覆盖。
    """
    facts: list[DistillFact] = Field(default_factory=list)
    matter_proposals: list[MatterProposal] = Field(default_factory=list)
    topic: str = ""
    keywords: list[str] = Field(default_factory=list)
    # 蒸馏模型名（记录到 Fact.distill_model 供审计；空 = 透传/未蒸馏）
    model_name: str = ""
    # MS-1（复核 D4）：本次蒸馏是**失败**返回的空，而不是"真无事实"的空。
    #
    # 为什么必须在返回值上区分：`LLMDistiller.distill` 逮到异常/解析失败时
    # **返回空 DistillOutput 而不抛**（降级不停摆），于是调用方看到的
    # "这条没提炼出事实" 与 "上游断了一条都没成" 长得一模一样 —— 上游 DNS 断掉那次，
    # consolidator 就是这样把积压全部标记成已消费且零事实的。
    # 聚合计数（`stats()`）只能告诉你"这一批里有 N 次失败"，答不出"是哪几轮失败的"，
    # 而"哪几轮"正是决定哪些 turn 不该被标记消费的唯一依据。
    # 加法式：默认 False，历史台账反序列化后同样是 False（失败从不入台账）。
    failed: bool = False
    # 失败的**种类**——决定"这条 turn 该不该记一次重试"（MS-1 用户拍板：队列不许停摆）：
    #   "call"  = 调用本身抛了（连不上/超时/认证/配额）→ 上游的问题，与这条 turn 无关。
    #             这种失败不该记到 turn 头上，否则断线期间每条轮次都在攒重试次数，
    #             攒满就被丢弃 = 把积压烧光（2026-08-05 事故形态）。
    #   "parse" = 拿到响应了、但不是能用的 JSON → **上游活着**，问题出在这条输入
    #             （或模型对这条输入的输出）。它重放多少次都是同样的结果，
    #             所以必须计次并在超限后放行，否则永久坏轮会把整条队列堵死。
    #   ""      = 没失败。
    failure_kind: str = ""


class DistillJournalProtocol(Protocol):
    """蒸馏台账接口（ADR-0018 §4.1，可 mock）。

    蒸馏前查台账命中复用（零 LLM 成本），miss 才调 LLM、结果写台账。
    实现在 proxy 侧（HubDistillJournal 包装 MemoryHub），consolidator 注入。
    """

    def get_distill(
        self, source_text: str, distill_model: str, prompt_ver: str,
        context_digest: str = "",
    ) -> DistillOutput | None:
        """命中返回 DistillOutput（含 facts + proposals），miss / 被墓碑返回 None。

        M1-3：`context_digest` 并入 key（拍板 1）。实现可不支持该参数——
        调用方用 TypeError 探测后退回旧签名（v3 行为逐字不变）。
        """
        ...

    def put_distill(
        self, source_text: str, distill_model: str, prompt_ver: str,
        output: DistillOutput, context_digest: str = "",
    ) -> str:
        """写入台账（write-if-absent），返回台账 key。已存在则不覆盖。"""
        ...


class DistillerProtocol(Protocol):
    """蒸馏接口（可 mock，单测不跑真模型）。"""

    @property
    def model_name(self) -> str:
        """蒸馏使用的模型名（记录到 Fact.distill_model 供审计/对比）。"""
        ...

    @property
    def prompt_ver(self) -> str:
        """蒸馏 prompt 版本（台账 key 的一部分，prompt 改 -> 递增 -> 新台账）。"""
        ...

    def distill(self, user_message: str) -> DistillOutput:
        """从 user 消息抽取原子事实 + Matter 提案。

        返回 DistillOutput（facts 可能为空 = 无可提取事实/系统消息；junk 不在内）。
        实现负责台账缓存（命中复用）+ 失败降级（返回空 facts，不抛）。
        """
        ...

    # 可选能力（ADR-0019 P2）：distill_conclusion(text) -> DistillOutput
    # 从任务结论（final assistant 回答）抽取值得记住的事实/决定/结果。
    # 不列入 Protocol 硬约束——consolidator 用 getattr 探测，缺失则跳过结论蒸馏
    # （旧 mock/第三方实现零改造兼容）。
    #
    # 可选能力（M1-1 v4）：distill_turn(DistillTurnInput) -> DistillOutput
    # 一轮一次统一调用（user+assistant 同包）。同样用 getattr 探测：
    # 实现缺失时 consolidation 回落 v3 的三条独立流水线，行为逐字不变。
    # **不放进 Protocol 硬约束**——放进去等于逼所有既有 mock 立刻实现它，
    # 而它们绝大多数只想测别的东西。


class PassthroughDistiller:
    """透传蒸馏器 - 不蒸馏，直接返回原文（fallback / 测试用）。

    model_name = "passthrough"，标识这条 Fact 未经 LLM 蒸馏。
    """

    @property
    def model_name(self) -> str:
        return "passthrough"

    @property
    def prompt_ver(self) -> str:
        return "passthrough"

    def distill(self, user_message: str) -> DistillOutput:
        if not user_message or not user_message.strip():
            return DistillOutput()
        return DistillOutput(
            facts=[DistillFact(content=user_message, kind="general")],
            model_name="passthrough",
        )

    def distill_conclusion(self, conclusion: str) -> DistillOutput:
        """ADR-0019 P2：透传模式不蒸结论（长回答原文入库是污染，宁缺毋滥）。"""
        return DistillOutput(model_name="passthrough")

    def distill_turn(self, payload: DistillTurnInput) -> DistillOutput:
        """M1-1 v4 统一调用的透传实现：只把 user 正文原样透出。

        assistant 正文照旧不透传（理由同 distill_conclusion：长回答原文入库是污染），
        所以透传模式下 v4 与 v3 的产出逐字一致——这条是"关掉 LLM 也不回归"的锚点。
        """
        text = (payload.user_text or "").strip()
        if not text:
            return DistillOutput(model_name="passthrough")
        return DistillOutput(
            facts=[DistillFact(content=text, kind="general",
                               provenance=Provenance.USER_DIRECT.value)],
            model_name="passthrough",
        )
