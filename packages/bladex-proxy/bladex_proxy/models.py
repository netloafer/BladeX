"""Domain 对象 — Identity / Turn / InjectedRequest 等（T2）。

所有业务数据用 Pydantic v2 模型，禁止裸 dict 传递（CLAUDE.md 编码规范）。
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from bladex_core.distillation import DistillFact, MatterProposal
from bladex_core.task_unit import TaskUnit
from bladex_proxy.splice import SpliceRecord
from pydantic import BaseModel, Field, field_validator


class SessionIdSource(str, Enum):
    """session_id 的识别来源（T4/ADR-0018 §4.2 R7）。

    tw: 时间窗会话跳过 prefix_changed 判定与 CAP 冲突检测（R7）。
    """
    EXPLICIT = "explicit"              # X-Session-ID header
    TAIL_CONTINUATION = "tail_continuation"  # 尾部接续匹配
    FINGERPRINT = "fingerprint"        # 前缀指纹 fp:
    TIME_WINDOW = "time_window"        # 时间窗兜底 tw:


#: 内循环 aux 轮的 `aux_source`（MQ-P9）。放在 models 而不是 innerloop：存储层
#: （`memory_hub.put` 的前缀检测，MQ-P16）也要认它，storage 不该反向 import 拦截协议模块。
#: `innerloop.INNER_LOOP_AUX_SOURCE` 是同一对象的再导出。
INNER_LOOP_AUX_SOURCE = "inner_loop"


class Identity(BaseModel):
    """从请求解析出的身份四元组（ADR-0008 §6.4）。

    turn_index 保留为推断元数据（展示/聚合用），不再决定 Memory Hub 主键唯一性。
    Memory Hub 主键改用 Redis stream entry_id（全局有序且唯一，见 M-proxy-1.5 T1）。
    """

    user_id: str
    agent_id: str = "unknown"
    session_id: str = "default"
    turn_index: int = 0
    # ADR-0026 §2.3 / U2：Memory Hub key 结构 principal/agent/session/{entry_id}。
    # 个人模式 principal == user_id（identity.py:549 已把 user_id = resolved.principal_id），
    # 故此字段默认空 = 沿用 user_id，storage_key/session_prefix 字节一致（零迁移语义变化）。
    # 显式非空时（企业多 key → 一 principal）key 走 principal_id，与 IDENTITY_MERGE 对齐。
    principal_id: str = ""
    # T6: auxiliary 标记（agent 内部辅助调用：MoA reference、子 agent、压缩摘要等）
    # auxiliary 轮次只注硬规则、入库打标记、consolidation 跳过
    auxiliary: bool = False
    # T2(ADR-0018): auxiliary 命中的规则名（空=未命中），供审计 + rebuild 重判。
    # consolidation 不信任冻结的 turn.auxiliary，用 classify_auxiliary 重判。
    aux_source: str = ""
    # T4(ADR-0018 §4.2 R7): session_id 来源，tw: 会话跳过 prefix_changed/CAP 冲突检测
    session_id_source: SessionIdSource = SessionIdSource.TIME_WINDOW
    # ② 子代理（2026-08-26，MQ-A12）：agent **在协议里自报**自己是子代理时置位
    # （如 Codex 的 `x-openai-subagent: guardian` ⇒ agent_id=`codex:guardian`）。
    #
    # 🔴 **与 `auxiliary` 分开的理由**：`auxiliary` 已经被蒸馏与路由两个消费方共用过
    # 一次并出过事（2026-07-28，修法是拆 `DISTILL_ONLY_AUX_RULES`）。这里问的是
    # 第三件事——"**这一轮该不该写账本**"。子代理**有身份、有记忆价值**
    # （guardian 的安全审查结论要留），但**不是任务账本的作者**。
    # 三件事共用一个旋钮正是那次事故的形状。
    subagent: bool = False
    # 项目识别第一档（2026-08-29，ADR-0032 §4 键分层 显式→git 指纹→Global）：
    # 空串 = Global（无项目概念 agent，或 cwd 不在 git 仓库——家目录会话等）。
    # 生产者 = server._prepare_round（project_identity.resolve_from_request）；
    # 消费者 = Turn 元数据留存（非空率仪器可查）+ 目录树/账本激活作用域
    # （消费面切换 gate 在 flash 数据维护讨论——激活作用域换键要迁移方案）。
    project_id: str = ""
    project_source: str = ""   # explicit / git-remote / git-root / path / 空(Global)
    project_name: str = ""     # 人读名（仓库根 basename），树目录命名用，不参与身份
    project_root: str = ""     # 项目根 realpath——flash 蒸馏项目简介读声明文件用

    # ADR-0021 section 2.4/3.1: 可见集合 + 敏感度（身份解析产出，挂 Identity 供 Memory Index 检索 +
    # 敏感度解析消费）。空 = 个人模式现状回落（Memory Index search 走 user_id 兼容路径）。
    visibility: list[str] = Field(default_factory=list)
    principal_sensitivity: str = ""    # 空 = 未标注 -> default_level
    team_sensitivities: list[str] = Field(default_factory=list)
    # ADR-0021: principal/team 级路由策略用（team>principal>agent 选池）。
    team_ids: list[str] = Field(default_factory=list)
    # ADR-0021 section 3.3c: 本轮解析出的敏感等级（T3 写入，随 Turn 入 Memory Hub，重建可重放）。
    # 默认 normal = 现状。consolidator 据此给蒸馏 Fact 设 exposure_ceiling（血统继承）。
    sensitivity: str = "normal"

    # ── G11.1 / MQ-A5：接入形态留存（2026-08-18）──
    # 脱敏 header 快照（凭证值已替换为 <redacted>，见 agent_bucket.redact_headers）。
    # **署名消费者**：dashboard agent 认领界面（G11.11）、未识别登记（G11.6）、人工排查。
    # 存在理由：新 agent 第一次接入就有据可查（自愈），取代"跑多 agent 事前标定"
    # ——本机没有那么多 agent 可测，且各家 header 随版本漂移，预先枚举追不上。
    # 追加字段，历史 Turn 为空 dict，存量零迁移。
    request_headers: dict[str, str] = Field(default_factory=dict)
    # 分桶判据串（形如 "ua:deepseek-harness|vendor:deepseek-harness"），
    # 供 dashboard 告诉用户"这个 unknown-<hash> 是凭什么分出来的"。
    agent_bucket_basis: str = ""

    @property
    def principal_level(self) -> str | None:
        """IdentitySensitivity Protocol: principal 敏感等级（空 = None -> default_level）。"""
        return self.principal_sensitivity or None

    @property
    def team_levels(self) -> list[str]:
        """IdentitySensitivity Protocol: 各 team 敏感等级。"""
        return self.team_sensitivities

    @property
    def key_principal(self) -> str:
        """Memory Hub key 的 principal 段（ADR-0026 §2.3）。

        principal_id 非空则用之，否则回落 user_id（个人模式二者相等，字节一致）。
        """
        return self.principal_id or self.user_id

    def session_prefix(self) -> str:
        """Memory Hub key 的会话前缀：principal/agent/session/。

        用于 scan_prefix 取回整会话所有轮次，以及入库前尚无 entry_id 时的日志。
        个人模式 principal == user_id，前缀与旧格式字节一致。
        """
        return f"{self.key_principal}/{self.agent_id}/{self.session_id}/"

    def storage_key(self, entry_id: str) -> str:
        """Memory Hub 的 key 格式：principal/agent/session/{entry_id}（ADR-0026 §2.3）。

        entry_id 来自 Redis stream XADD，天然全局有序且唯一，
        重试/重新生成/工具循环都会产生不同 key，不再覆盖。
        个人模式 principal == user_id，key 与旧格式字节一致（零迁移语义变化）。
        """
        return f"{self.key_principal}/{self.agent_id}/{self.session_id}/{entry_id}"


class TurnStatus(str, Enum):
    OK = "ok"
    FAILED = "failed"


class AgentSource(str, Enum):
    """agent_id 的识别来源（记录在 Turn 里，供 Memory Hub 审计 + 未来策略管理）。"""

    EXPLICIT = "explicit"        # X-Agent-ID header（最可靠）
    FINGERPRINT = "fingerprint"  # 被动指纹：system prompt / tool 签名
    STICKY = "sticky"            # 会话粘性：同 user 最近识别过的 agent
    USER_AGENT = "user_agent"    # HTTP User-Agent header
    FALLBACK = "fallback"        # 全部落空 → unknown


# ── T4(ADR-0018 §4.4 C1-C7): Turn 三组元数据 ──


class ResponseMeta(BaseModel):
    """响应侧元数据（C1/C2/C6）。事后不可回补。"""
    reasoning_text: str = ""                              # C1: 思考流（delta.reasoning_content 累积）
    usage: dict[str, int] = Field(default_factory=dict)  # C2: {prompt, completion, total}
    finish_reason: str = ""                              # C6: length/tool_calls/stop
    ms_first_chunk: float = 0.0                          # C6: 首字延迟

    # ADR-0020 T4 兼容：旧 Pipeline/Memory Hub 数据 reasoning_text/finish_reason 可能是 list
    # （早期 capture 未强制 str）-> Pydantic 校验失败 -> PEL 毒消息（reclaim 反序列化失败）。
    # before validator 把 list coerce 成 str，与 ADR-0020 T4 写入侧强制 str 对齐。
    @field_validator("reasoning_text", mode="before")
    @classmethod
    def _coerce_reasoning_text(cls, v: Any) -> Any:
        if isinstance(v, list):
            return "".join(str(x) for x in v)
        return v

    @field_validator("finish_reason", mode="before")
    @classmethod
    def _coerce_finish_reason(cls, v: Any) -> Any:
        if isinstance(v, list):
            return ",".join(str(x) for x in v)
        return v


class RequestParams(BaseModel):
    """请求侧参数（C3）。tools schema 全文进 __msg__ 池按 hash 去重，Turn 只存 tools_hash。"""
    requested_model: str = ""        # agent 请求的原始 model（Turn.model 存路由后实际用的）
    temperature: float | None = None
    max_tokens: int | None = None
    tool_choice: Any | None = None
    tools_hash: str = ""             # tools schema 全文的 __msg__ 池 hash（空=无 tools）


class DecisionMeta(BaseModel):
    """决策侧元数据（C4）：路由 + CAP + 注入位置--"模型实际看到什么"可重建。"""
    route: dict[str, str] = Field(default_factory=dict)    # {source, tier, reason}
    cap: dict[str, Any] = Field(default_factory=dict)      # {triggered, summary_hash, kept_recent_n}
    inject_position: str = ""        # 注入位置标记
    # ADR-0028 E2.3：本轮**注入命中**的 fact id（cap 32）。
    # 为什么走 Memory Hub 而不是 proxy 直接写 Memory Index：写权只有 consolidator（ADR-0009 §7），
    # proxy 侧 Memory Index 是 read_only —— 此前 `record_injection_hits` 在 proxy 里恒 no-op，
    # 于是 importance 的引用增益项永远吃不到数据（"生产了没人消费"的反面：
    # 有消费者、没有生产者）。改为随 turn 落 Memory Hub、consolidator 消费时聚合回写，
    # 零新写路径、幂等性由 consumed 标记天然保证（每 turn 只被消费一次）。
    injected_fact_ids: list[str] = Field(default_factory=list)
    # M3-1（复核 5.9 三时钟）：本轮**钉住的 Matter 卡** id。
    #
    # 三个钟里这一个此前完全没有信号源：`injected_fact_ids` 只记散点条目，
    # Matter 卡注入**零留痕**——连计数都没有。于是 2×2 矩阵里"久无命中"那一列
    # 永远为真，dormant 判定要么不敢开、要么开了就误杀。
    # 注入器本来就知道自己钉了哪些卡，补这一个字段即可（5.9 修正 4）。
    #
    # 与 fact 侧同一支笔：随 turn 落 Memory Hub，consolidator 消费时回写
    # `Matter.last_hit_at`，幂等性由 consumed 标记天然保证。
    injected_matter_ids: list[str] = Field(default_factory=list)
    # MQ-S38（2026-08-20）：注入条目的**稳定身份**（fact_id → 取代键字符串）。
    #
    # 为什么 id 不够：`fact_id = sha256(source_ledger_key + ":" + content)[:12]`——
    # 内容派生。蒸馏 prompt 换版（v6→v7 那种全库重蒸）⇒ 正文变 ⇒ 全库 id 换代 ⇒
    # 历史 turn 里的 `injected_fact_ids` **整批指空**（live 实测 385/402 = 95.8%），
    # 而 `_apply_injection_hits` 当时是静默 continue ⇒ ref_count 引用增益、
    # 相关钟 `last_hit_at`、dormant 判定的历史信号在每次换版时被清零且无痕。
    #
    # 取代键 `(scope, subject, attribute)` 不随措辞变，是既有设计里唯一的跨版本身份。
    # 覆盖面诚实：有键的类型 = `_SUPERSEDABLE_KINDS`（assertion / preference /
    # file_ref）且 subject·attribute 非空；**lesson / procedure 无键**
    # ⇒ 不入本字段、换版后仍会丢（计入 orphan 读数）。
    #
    # 追加字段：历史 Turn 为空 dict → 回写侧退回"只按 id"（现状逐字不变）。
    injected_fact_keys: dict[str, str] = Field(default_factory=dict)


class ToolEvent(BaseModel):
    """一次工具调用或返回（捕获时尽量提取）。

    P1 修复（ADR-0011）：tool_call_id 用于匹配 call 和 result。
    """

    tool_name: str = ""
    arguments: Any | None = None
    result: Any | None = None
    direction: str = "call"
    tool_call_id: str = ""


class ReconstructionRecord(BaseModel):
    """重构决策（ADR-0025 §5，Memory Hub 追加字段，标签区分原始 vs 重构）。

    存"从 original 重放出 reconstructed 所需的最小信息"（diff），而非重构后 messages 全文：
    体积近零、重建等价性天然成立、diff 本身即可读审计记录。

    🔴 隔离不变式（I5 / U8 test_index_never_consumes_reconstruction）：Memory Index 蒸馏永不读此字段。
    """

    layer: str = "A"                              # A | AB | ABC | …P（P = ADR-0028 冷启裁剪）
    degrade_plan: dict[int, str] = Field(default_factory=dict)  # msg_index -> full|excerpt|minimal
    fingerprint: dict[str, list[str]] = Field(default_factory=dict)  # B 层关联指纹快照（可解释性）
    clarify_text: str = ""                        # C 层澄清块全文
    # ADR-0028 §3 约束4：冷启相关性裁剪决策（可回放可解释）。
    # {pruned: [unit_key], scores: {unit_key: float}, target, kept_chars, chars_before}
    prune: dict[str, Any] = Field(default_factory=dict)
    index_watermark: str = ""                        # 判定所依据的 Memory Index 时点（重放可定位）


class LedgerAnchor(BaseModel):
    """这一轮**模型实际看到的账本**：哪本、哪个 rev、各段各占多少字符（F-A1 / MQ-A35）。

    ## 为什么必须落 Turn 而不是只打日志

    日志答得了"上一轮注了什么"，答不了"**那一轮**注的是什么"——日志按时间滚、
    按天切、还会被 grep 不到；而 handoff bench §4.2 与 MQ-A34 ①② 的裁剪判断要的是
    "对这批 Turn 而言各段占比多少"，那是**按 Turn 聚合**的问题。字段落在 Turn 上，
    分母（这一轮）与分子（这一段）才在同一条记录里（分母纪律，
    feedback_instrument_reference_frame）。

    ## 三个可重放的锚

    - `(ledger_id, rev)`：拿这一对回 Hub 事件流重放，能得到逐字相同的那一版账本
      ——读数因此**可被第三方复算**，不必信 proxy 当时的自述。
    - `sections`：段级字符（键 = `agency.runtime.LEDGER_BLOCK_BUCKETS`，
      **按 key 读不按位置读**：账本可增段，新段自动多一个键）。
      `sum(sections) + about_chars + roster_chars` = 本函数加进正文的全部字符
      （即 `bladex_added_chars`，在 `_apply_agency_surfaces` 之前无其它注入时逐字相等）。
    - `added_hash`：注入后正文 − 原正文 的拼接文本 sha256 前 16 位。用途是
      **对账**：两条记录 hash 相同 ⇒ 加进去的正文逐字相同，不必存全文。

    🔴 `None` = 这一轮没有这个信息（历史 Turn / aux 轮 / 模块关）。缺数报缺数，
    不造 0 —— 与 `Turn.ledger_face` 三态同一条纪律。
    """

    ledger_id: str = ""
    rev: int = -1
    sections: dict[str, int] = Field(default_factory=dict)
    about_chars: int = 0
    roster_chars: int = 0
    added_hash: str = ""


class Turn(BaseModel):
    """完整一轮对话（请求 + 回复 + 工具事件），进 Pipeline→Memory Hub。"""

    identity: Identity
    model: str = ""
    request_messages: list[dict[str, Any]] = Field(default_factory=list)
    injected_memory: str = ""
    response_text: str = ""
    tool_events: list[ToolEvent] = Field(default_factory=list)
    status: TurnStatus = TurnStatus.OK
    error: str = ""
    ts: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # ── agent 识别（记录来源，供 Memory Hub 审计 + 未来策略管理）──
    agent_source: AgentSource = AgentSource.FALLBACK

    # ── 轮次元数据（T1）──
    # roundtrip: 同一逻辑用户轮内的 LLM 往返序号（工具循环会产生多条往返）
    roundtrip: int = 0
    # logical_turn: 逻辑用户轮序号（供后续 consolidation 按用户轮聚合，M-proxy-2 消费）
    logical_turn: int | None = None

    # ── Memory Hub 去重元数据（T4）──
    # prefix_hash: 本轮所有消息内容的级联 hash，用于校验/重放
    prefix_hash: str = ""
    # prefix_changed: agent 压缩致前缀变化时退化为全量存并打标记
    prefix_changed: bool = False

    # T6: auxiliary 标记（从 Identity 继承，供 consolidation 跳过）
    auxiliary: bool = False
    # T2(ADR-0018): auxiliary 命中规则名（空=未命中），Memory Hub 存初判、Memory Index 用 classify_auxiliary 重判
    aux_source: str = ""
    # T4(ADR-0018 §4.2): session_id 来源（从 Identity 继承，供 Memory Hub 审计）
    session_id_source: SessionIdSource = SessionIdSource.TIME_WINDOW

    # ADR-0021 §3.3c：本轮敏感等级（身份解析 + 敏感度解析产出，随 Turn 入 Memory Hub，重建可重放）。
    # consolidator 据此给蒸馏 Fact 设 exposure_ceiling（血统继承起点）。默认 normal = 现状。
    sensitivity: str = "normal"

    # ── MQ-L26（2026-08-28）：这一轮**给没给账本面** ──
    #
    # `None` = 记录里没有这个信息（schema 早于本次改动）⇒ Memory Index 按**旧行为**锚定。
    # `False` = 明确没给（aux / 子代理 / 零自带工具 / 孤立子调用）⇒ **不锚定**。
    # `True`  = 给了 ⇒ 正常锚定。
    #
    # ## 为什么需要它（T5 复验的净新增发现）
    #
    # 账本面 gating 与锚定归属此前用**两套完全不同的判据**，中间无对账：
    #   给不给账本面：`is_ledgerless_auxiliary(aux) or subagent or brings_no_tools
    #                  or is_isolated_subcall`（proxy 侧，请求时）
    #   锚不锚      ：只看 `ledger_active_at(scope, ts)`（Memory Index 侧，重建时）
    # ⇒ 一轮没拿到账本面（模型看不到账本、也没有 switch 工具），它的产出**照样**
    # 被锚到当时激活的旧账本上。live 病例：`[3] Codex ledger health 开发` 混入
    # 10 条 BladeX 全局状态，全是 08-28 13:32–13:33 的零工具「建议生成」轮
    # （`reason=aux`），而该卡其余成员是 08-27 20:xx。
    #
    # 锚定一条**模型从来没机会表态**的产出，等于替它做了归属判断——直接违反
    # ADR-0032 的作者边界。不锚 ⇒ 走 DPL 五层兜底（provisional 留池、≥2 成员才
    # 转正），比"无条件挂到激活账本"保守得多：**没有强信号时走保守路径，
    # 而不是用一个错的强信号。**
    #
    # ## 为什么可以存、且必须存（不违反 ADR-0018 R2）
    #
    # R2「重判不信任冻结字段」是为**补漏**立的：`turn.auxiliary` 是对 agent 的
    # **判断**，初判会漏 user 角色的委派，故重建必须用 `classify_auxiliary` 重算。
    # 本字段不同——它是「**我们自己做过什么**」的记录，不存在漏判；而且重建时
    # **根本无法重算**（`identity.subagent` 来自 HTTP header，消息正文里没有）。
    #
    # 🔴 三态而非 bool：历史轮缺这个信息，若默认 `False` 会让全量重建时**所有
    # 历史锚定一次性失效**（ADR-0012 §3.6）。缺数就报缺数，不假装成"明确没给"。
    ledger_face: bool | None = None

    # ── F-A1 / MQ-A35（2026-09-07）：这一轮模型看到的是哪本账本的哪个 rev、各段多长 ──
    # `None` = 没有这个信息（历史 Turn / aux 轮 / 账本模块关）。追加字段，
    # 旧 Turn 反序列化照读（缺省 None，不触发任何重建行为变化）。
    ledger_anchor: LedgerAnchor | None = None

    # ── Turn schema v3（ADR-0024 T2）：任务单元 ──
    # 热路径确定性识别一次，随 Turn 落 Memory Hub，Memory Index 直接消费——归属从"推断"变成"记录"。
    # 索引口径 = request_messages（即 agent 原始 messages）。
    # 追加字段：历史 Turn 为空列表 → Memory Index 侧 fallback 回现有推断路径，存量零迁移。
    task_units: list[TaskUnit] = Field(default_factory=list)

    # ── T4(ADR-0018 §4.4 C1-C7): 三组元数据（新写入生效，历史不回改）──
    response_meta: ResponseMeta = Field(default_factory=ResponseMeta)
    request_params: RequestParams = Field(default_factory=RequestParams)
    decision_meta: DecisionMeta = Field(default_factory=DecisionMeta)
    # C5: /v1/messages 原始请求体按 hash 入 __msg__ 池，Turn 记 raw_request_ref
    raw_request_ref: str = ""

    # ── V-P4 硬点 1（ADR-0032 §3.2）：拼接台账 = Hub 投影 ────────────────
    # 本轮被剥离的 bladex 调用与其结果。**进程内 `SpliceLedger` 是它的投影，
    # 不是唯一副本**——落在这里，proxy 重启后可由 `restore()` 重建。
    #
    # 🔴 2026-08-27 补：此前 `SpliceRecord` 只活在进程内存里，`restore()`
    # 零生产调用方，而 docstring 声称"随 turn 入 Hub、重启不致失忆"。
    # **文档声称已实现、实际没有**——比"把缺陷写进文档"更糟一档。
    #
    # 空列表 = 本轮没有剥离（绝大多数轮次），Pydantic 默认值不占存储。
    splice_records: list[SpliceRecord] = Field(default_factory=list)

    # （ADR-0026 §2.3 / U2 审计字段 `api_key_id` / `org_id` 已于 2026-09-06 F0.3 删：
    #   企业形态字段一个月零生产者——H1 销账。旧 Turn 里的空串键经 `extra="allow"` 忽略，schema 无 version bump。）

    # ── ADR-0025 §5 / U2：请求重构决策（标签：reconstructed）──
    # request_messages 恒为 agent 原始（标签：original，I1/I5 的 schema 层保险）；
    # 重构侧独立字段承载 —— 第 4 条隔离不变式在 schema 层默认安全。
    # 历史 Turn 为 None → 走 fallback（现状纯叠加/assembly）。
    reconstruction: ReconstructionRecord | None = None

    ms_identity: float = 0.0
    ms_inject: float = 0.0
    ms_total: float = 0.0


class ChatCompletionRequest(BaseModel):
    """OpenAI 兼容 /v1/chat/completions 请求体。

    显式列出常用字段，extra 字段（tools, tool_choice 等）通过 extra_fields 透传。
    """

    model: str = ""
    messages: list[dict[str, Any]] = Field(default_factory=list)
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    stop: Any | None = None
    user: str | None = None
    # 透传未显式列出的其它字段（response_format, seed, n 等）
    model_config = {"extra": "allow"}

    def to_router_kwargs(self) -> dict[str, Any]:
        """把所有非 None 的参数组装成 Router 转发（`router_sdk.acompletion`）的 kwargs。

        关键：tools / tool_choice 等必须透传，否则模型无法做工具调用。
        """
        kwargs: dict[str, Any] = {}

        # 显式字段
        for field_name in (
            "temperature", "max_tokens", "tools", "tool_choice",
            "top_p", "frequency_penalty", "presence_penalty", "stop", "user",
        ):
            val = getattr(self, field_name, None)
            if val is not None:
                kwargs[field_name] = val

        # extra 字段（response_format, seed, n, stream_options 等）
        if hasattr(self, "model_extra"):
            for key, val in self.model_extra.items():
                if val is not None:
                    kwargs[key] = val

        return kwargs


# ── ADR-0012: 墓碑 + 管理事件（Memory Hub 追加式记录）──


class TombstoneTargetType(str, Enum):
    """墓碑覆盖的目标类型（ADR-0012 §3.6 + ADR-0018 §4.1 台账级联）。"""

    TURN = "turn"
    FACT = "fact"
    MATTER = "matter"
    EDGE = "edge"
    # ADR-0018 §4.1: 派生台账墓碑（turn 被删时级联，Memory Index 重建不复活）
    DISTILL = "distill"
    JUDGMENT = "judgment"
    #: 2026-09-01：账本墓碑（`task-ledger-tombstone-cleanup-20260901.md`）。
    #: 🔴 选账本层而不是 matter 层：账本事件的三个消费方（agency 池装载 / 重建锚层 /
    #: flash 物化）同源于 `replay_ledger_events`，在装载处滤一次三处自动跟随，
    #: 且锚卡**自然不会被创建**（`_la_ledger_at` 查不到它）。matter 墓碑则要在锚层
    #: 单独拦截创建、与 fact 墓碑「纯减项」语义打架，且账本仍留在池里被注入被物化。
    LEDGER = "ledger"


class TombstoneSource(str, Enum):
    """删除操作来源。"""

    USER = "user"      # dashboard / API 用户删除
    SYSTEM = "system"  # 系统发起（如 compaction）


class Tombstone(BaseModel):
    """墓碑记录 — 标记 Memory Hub 中某条目已被删除（ADR-0012 §3.6）。

    墓碑本身是追加记录（不违反 Memory Hub 追加式不变式）。
    读路径（scan_prefix）和 Memory Index 重建跳过被墓碑覆盖的条目。
    原文物理擦除由后台 compaction 处理（周期留实测，本卡只留 stub）。
    """

    target_type: TombstoneTargetType
    target_key: str                                    # 被删条目的 Memory Hub key（turn/fact）或 matter_id / edge_id
    ts: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source: TombstoneSource = TombstoneSource.USER


class AdminEventType(str, Enum):
    """管理事件类型 — Matter 生命周期操作（ADR-0012 §3.5）。

    🔴 **新增事件类型前必读**（2026-08-18 立，台账 MQ-A7）：
    `AdminEvent.matter_id` 只在目标**真是 Matter** 时使用，其余一律留空、走 `payload`，
    并在本枚举的注释里写明载荷结构（`IDENTITY_MERGE`/`SCOPE_PROMOTE`/`*_IMPORT`
    已是此形态，照抄即可）。

    **不要再复用 `matter_id` 装别的 id**。它已被 `IDENTITY_MERGE` 复用为
    `principal_id` 一次，字段名对半数事件类型已经是错的；再叠一层就得靠
    `event_type` 反推该字段装了什么。同型事故见 MS-17（`consolidation:*` 命名空间
    被 L4 读成判决 → 伪 uncertain，靠"两个读数互相矛盾"才破案）——
    **一个命名空间承载两种语义，出问题时归因成本极高**。

    字段名泛化（`matter_id` → `target_id`）要动存量 Memory Hub 的管理事件 schema
    与重放侧，代价不匹配收益，故先记不改；本注释即护栏。
    """

    MATTER_CREATE = "matter_create"
    MATTER_ASSIGN = "matter_assign"
    MATTER_DETACH = "matter_detach"
    MATTER_MERGE = "matter_merge"
    MATTER_SPLIT = "matter_split"
    MATTER_CLOSE = "matter_close"
    MATTER_RENAME = "matter_rename"
    # ADR-0021 section 2.5: principal declares legacy_ids merge -> Memory Index rebuild maps legacy->principal.
    # matter_id field reused as principal_id; payload={principal_id, legacy_ids}.
    IDENTITY_MERGE = "identity_merge"
    # ADR-0021 section 2.4/3.3c: scope promotion (personal->team/org) - explicit admin op.
    # matter_id = target matter_id; payload={target_type, target_key, new_scope}.
    SCOPE_PROMOTE = "scope_promote"
    # Beta T12 快照导入（ADR-0012 §3.5 语义延伸）：导入的实体不来自本地 Memory Hub 会话，
    # 必须以管理事件入 journal 才能在 full rebuild 后存活（重建 = 重放会话 +
    # 重放管理事件 − 墓碑）。payload = 实体完整 dump（不含 embedding，重放时重嵌）。
    FACT_IMPORT = "fact_import"
    # MQ-L27（2026-08-29）：人设事实 audience 补标（存量修法③）。
    # 🔴 matter_id 留空（MQ-A7 护栏：新事件类型不复用该字段）；
    # payload = {fact_id, audience}，audience ∈ {"all", "agent:<base>"}。
    AUDIENCE_SET = "audience_set"
    MATTER_IMPORT = "matter_import"
    EDGE_IMPORT = "edge_import"
    # G11.11（台账接入域）：agent 认领 / 改名 / 合并 —— 一套语义三种用法。
    # matter_id **留空**（见本枚举 docstring 的护栏）；
    # payload = {from_agent_ids: [str], to_agent_id: str, rule: dict | None, merge: bool}。
    #
    # 为什么并列而非复用 IDENTITY_MERGE：后者的重放侧是"legacy user_id -> principal"，
    # 塞 agent 进去会让同一个消费者分支判断两种语义（MS-17 就是这个形态：
    # consolidation:* 命名空间被 L4 读成判决 -> 伪 uncertain，靠"两读数互相矛盾"才破案）。
    # 回滚也需隔离：撤销 agent 认领不该有任何机会碰到 user identity 映射。
    AGENT_CLAIM = "agent_claim"
    # ── V-L1（ADR-0032 §4）：账本域事件。matter_id **一律留空**（本枚举 docstring 护栏）。
    # 事件串与 bladex_core.ledger.EVENT_LEDGER_* 逐字一致（test_ledger_event_parity 对账）。
    # 状态式载荷：create/update/user_edit 的 payload = {"ledger": Ledger.model_dump()}
    # （FACT_IMPORT"完整 dump"先例——重放 = 覆盖，不做 diff 重放）。
    LEDGER_CREATE = "ledger_create"
    LEDGER_UPDATE = "ledger_update"
    # payload = {session_id, agent_id, project_id, from_ledger_id, to_ledger_id, ledger_id,
    #            turn_index}；turn_index = 落定时的**用户轮**（E0.2，2026-09-06 起；历史事件
    # 缺省 ⇒ 重放不恢复防抖基点，向后兼容）。
    # 切换=管理事件+防抖（ADR-0032 拍板 #7），防抖在写侧（V-L2），journal 只收落定的切换。
    LEDGER_SWITCH = "ledger_switch"
    # 惰性作用域迁移（08-29，MQ-A7 护栏：target 走 payload，matter_id 留空）
    LEDGER_SCOPE_MIGRATE = "ledger_scope_migrate"
    LEDGER_USER_EDIT = "ledger_user_edit"


class AdminEvent(BaseModel):
    """管理事件 — Matter 手动操作的 append-only journal（ADR-0012 §3.5）。

    每次手动操作先写 Memory Hub 管理事件，再应用到 Memory Index。
    Memory Index 重建时重放管理事件，手动映射不丢失。
    """

    event_type: AdminEventType
    matter_id: str                                     # 目标 Matter
    target_key: str = ""                               # 关联的 session key / fact id
    ts: datetime = Field(default_factory=lambda: datetime.now(UTC))
    payload: dict[str, Any] = Field(default_factory=dict)  # 扩展载荷（如 merge 源/目标、新标题）


# ── ADR-0012 §3.5: 管理端点请求模型（P6: Pydantic 校验，禁裸 dict）──


class RememberFactBody(BaseModel):
    """POST /admin/facts 请求体（Beta T17 MCP remember）。"""

    content: str = ""
    scope: str = ""        # "" = personal；team:<id> / org:<id> 需组织层
    user_id: str = ""      # 归属 user（留空 = "manual"）


class CreateMatterBody(BaseModel):
    """POST /admin/matters 请求体。"""

    title: str = ""
    summary: str = ""
    matter_id: str = ""  # 留空则自动生成


class AssignMatterBody(BaseModel):
    """POST /admin/matters/{id}/assign 请求体。"""

    target_type: str = "session"  # "session" | "fact"
    target_key: str = ""


class MergeMattersBody(BaseModel):
    """POST /admin/matters/{id}/merge 请求体。"""

    source_matter_id: str = ""


class RenameMatterBody(BaseModel):
    """POST /admin/matters/{id}/rename 请求体。"""

    title: str = ""


class DetachEdgeBody(BaseModel):
    """POST /admin/matters/{id}/detach 请求体。

    两种用法（二选一）：
      - edge_id：精确摘除某条归属边
      - target_key + target_type：摘除指向某 target 的边
    """

    edge_id: str = ""
    target_key: str = ""
    target_type: str = "session"


class SplitMatterBody(BaseModel):
    """POST /admin/matters/{id}/split 请求体。

    把指定归属边从 source Matter 拆到新 Matter。
    """

    new_title: str = ""
    edge_ids: list[str] = Field(default_factory=list)


class PromoteScopeBody(BaseModel):
    """POST /admin/scope/promote 请求体（ADR-0021 section 2.4 scope 提升）。

    把 Fact/Matter 的可见范围从 personal 提升到 team/org（显式管理操作，
    记 SCOPE_PROMOTE 管理事件，Memory Index 重建重放）。共享检索 UX 留 E2。
    """

    target_type: str = "fact"   # "fact" | "matter"
    target_key: str = ""        # fact_id 或 matter_id
    new_scope: str = ""         # "team:<tid>" | "org:<oid>"


class AgentRuleBody(BaseModel):
    """POST /admin/agents/rules 请求体（G11.11 增补，2026-08-19）。

    让 dashboard 能直接改识别规则，而不是逼用户手编 `config/agent_rules.toml`
    ——规则外置的意义是用户能参与，"参与"不该等于"手编 TOML"。

    按 `rule["name"]` upsert；字段与 `AgentFingerprintRule` 一致，
    校验在 `agent_rules._rule_from_dict`（未知字段响亮报错，不静默忽略）。
    """

    rule: dict[str, Any] = Field(default_factory=dict)


class AgentClaimBody(BaseModel):
    """POST /admin/agents/claim 请求体（G11.11，台账接入域）。

    **一套语义覆盖三种用法**（2026-08-18 Jason 拍板：不区分两套体系，避免复杂化）：

    ==================  ==========================  ================
    场景                from_agent_ids              to_agent_id
    ==================  ==========================  ================
    认领 unknown        ``["unknown-a1b2c3d4"]``    ``"dsh"``
    改名已确定的 agent  ``["hermes:default"]``      新名字
    合并                ``["a", "b"]``              ``"c"``
    ==================  ==========================  ================

    🔴 **误合并红线就落在这一条请求的校验里**，不需要第二套机制：
    ``len(from_agent_ids) > 1`` **或** ``to_agent_id`` 已存在 → 判定为合并，
    必须带 ``confirm_merge=True``；否则 400。纯改名直接过。

    合并是不可逆操作的反面——它可回滚（撤销只需再发一条反向 AGENT_CLAIM），
    但**默认不许悄悄发生**：把 dsh 认领成 hermes 就是手动制造跨 agent 记忆污染。
    """

    from_agent_ids: list[str] = Field(default_factory=list)
    to_agent_id: str = ""
    #: 命中合并判据时必须显式置 True（二次确认）
    confirm_merge: bool = False
    #: 可选：同时写回一条识别规则到 `config/agent_rules.toml`，
    #: 下次同形请求直接被规则库认出来，不再落 unknown 桶（这是"认领一次就永久认识"）。
    rule: dict[str, Any] | None = None


# ── ADR-0018 §4.1/§4.3: Memory Hub 派生台账（distill / judgment / annotation）──
# DistillFact / MatterProposal 在 bladex_core.distillation（agent 中立，consolidator 共用）


class DistillRecord(BaseModel):
    """一次蒸馏的完整输出（ADR-0018 §4.1，存 distill/{hash}/{model}/{ver}）。

    不可确定性重算（LLM 非确定性），故入 Memory Hub 台账。重建命中复用 = 零 LLM 成本。
    同 source_text + model + prompt_ver -> 同 key（write-if-absent，首次结果保留）。
    """
    source_text_hash: str       # sha256(source_text)[:16]，跨重建稳定（R6 闭环）
    distill_model: str
    prompt_ver: str
    facts: list[DistillFact] = Field(default_factory=list)
    matter_proposals: list[MatterProposal] = Field(default_factory=list)


class JudgmentRecord(BaseModel):
    """一次 L4 链接裁决（ADR-0018 §4.1/§3.5，存 judgment/{fact_id}/{seq}）。

    可重判（周期 pass），故用 seq 追加而非确定性 key。
    verdict: link:<matter_id> | none | uncertain。
    """
    fact_id: str
    candidates: list[dict[str, Any]] = Field(default_factory=list)  # 候选 Matter 卡摘要
    #: 🔴 裁决时**篮子里有谁**（fact_id 列表，2026-08-31 新增）。
    #:
    #: 与 `candidates` 严格分开：后者在 `consolidation:*` 命名空间下存的是
    #: **targets**（被取代的那些），ADD 的 targets 恒空 ⇒ 台账里看不出这条候选
    #: 当时到底看没看见同键兄弟。08-31 归因卡在这里：`pair_never_linked` 498 条
    #: （44.3%）分不出「兄弟当时不在篮子里」（召回/时序问题）与「在篮子里仍判
    #: ADD」（裁决判断问题），而两者的修法完全不同。
    #:
    #: **不许重载 `candidates` 来省这个字段**：它的现有语义被 `linking.py`、
    #: `scripts/why_no_supersede.py` 与既有测试共同依赖，重载 = 一次静默语义漂移。
    #: 空列表对存量记录是正确默认（"这条老记录没记篮子"），无需迁移。
    #: 设计见 `docs/planning/supersede-replay-and-basket-20260831.md` §2。
    considered: list[str] = Field(default_factory=list)
    verdict: str = ""            # link:<id> | none | uncertain
    judge_model: str = ""
    # ADR-0018 §3.4: L4 顺带重写的 Matter 摘要（裁决输出第二字段，重建可复现）
    summary_rewrite: str = ""
    reason: str = ""
    ts: datetime = Field(default_factory=lambda: datetime.now(UTC))
    # MS-13（2026-08-10）：这条判决属于**哪一轮重建/进程 run**（run 起始时间戳）。
    # 台账跨重建累积、clear() 有意不清（重建零 LLM 成本），但 08-09 复验两次被
    # 历史残渣弄脏读数（拿 12:44 失明期判决对 16:33 新库做落盘核对，报假 🔴）。
    # 读侧（why_no_supersede.py ④ 等）默认只用最新 rebuild_id 的判决核对，
    # 历史折叠。空 = MS-13 之前的存量记录（读侧当"历史"处理）。
    rebuild_id: str = ""


class AnnotationRecord(BaseModel):
    """Turn 元数据的手动修正（ADR-0018 §4.3，v1 仅 schema 占位，无写入口）。

    dashboard 手动修正 turn 属性（"这轮是 auxiliary"/"session 归错"）时，
    以 append-only annotation 记入 Memory Hub，重建时压过自动判定（与管理事件同构）。
    UI 随 dashboard 里程碑。
    """
    target_key: str              # 被标注的 turn key
    field: str = ""              # 修正的字段名（如 auxiliary / session_id）
    value: Any = None            # 修正后的值
    ts: datetime = Field(default_factory=lambda: datetime.now(UTC))


#: TaskStateJudgment 的封闭 verdict 集（决策表 v2 §4 R5 的三种结果）。
#: 🔴 与 L4 的 verdict 形态（link:<id>|none|uncertain）不共形——`linking._is_link_verdict`
#: 的白名单自动跳过这三个值；反向由 test_taskstate_judgment.py 钉住。
TASKSTATE_VERDICTS: tuple[str, ...] = ("new", "continue_to", "degraded_new")


class TaskStateJudgment(BaseModel):
    """一次新开/延续判定（G12.2 R5；ADR-0031 §14.1 判定可重放性的载体）。

    存 `judgment_taskstate/{turn_key}/{seq}` —— **独立 key 前缀，不与 L4/M2 共用
    `judgment/`**（MS-17 教训：命名空间共用、种类不共用 → 伪 uncertain 796 条；
    那次靠 verdict 白名单救回，这次从 key 前缀上就分开，两层保险）。

    为什么"否定判决"（continue_to）也要记（决策表 v2 §6）：新开判定的旋钮会标定，
    只记 new 的话，旋钮一变、重放会把这类轮改判成新开 → 首轮集合变 → matter_id 漂
    （§0.1b 换个变量重演）。重放**台账优先**：有记录的轮读记录不重判。
    R1–R4 纯延续轮有意不入账——旋钮变更最坏多一张碎片卡（探针可见、merge 可收），
    已有事的 id 零漂移守卫仍成立。

    ⚠️ 生产者 = G12.2 R5（同一 G12 窗口，紧随本 schema 落地）；消费者 = 重建重放
    路径 + 误延续归因审计。落地前本 schema 零生产是**有意的前置状态**（执行卡
    G12.2 前置⑤），不是又一个 open_issues——H1 纪律下这条注释就是署名。
    """

    turn_key: str                # 被判轮次的 Memory Hub key（判定主体是轮，不是 fact）
    unit_key: str = ""           # 该轮末单元的 unit_key（内容派生，跨重建稳定）
    verdict: str = ""            # new | continue_to | degraded_new（TASKSTATE_VERDICTS）
    matter_id: str = ""          # new/degraded_new = 新开卡 id；continue_to = 目标卡 id
    # 误延续归因的第一问："当时锚指向谁、活着没有"（决策表 v2 §6）
    anchor_matter_id: str = ""
    anchor_alive: bool = False
    # 没有它，事后无法审计"当时窗口里有没有正确答案"（G3.0 归因靠的就是这个问题）
    window_snapshot: list[str] = Field(default_factory=list)
    signals: dict[str, Any] = Field(default_factory=dict)   # S1–S6 证据摘要
    judge_model: str = ""        # 空 = 确定性降级路径（degraded_new 恒空）
    # 🔴 新开判定旋钮的版本串——标定即 bump；重放对账（旋钮值变更 = 重建产物变更，
    # 按执行卡 §4 约束 B 攒窗口）。与蒸馏 prompt_ver 同一条纪律。
    knob_ver: str = ""
    reason: str = ""
    ts: datetime = Field(default_factory=lambda: datetime.now(UTC))
    # MS-13 同款：这条判决属于哪一轮重建/进程 run，读侧默认只用最新 rebuild_id。
    rebuild_id: str = ""
