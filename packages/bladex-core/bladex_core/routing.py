"""路由决策（agent 中立）— 配置驱动的确定性流水线（Phase 1 + RA 修复卡 20260712）。

CLAUDE.md 定位：路由 = 入口，无护城河、不自研打分器。
Phase 1 = 纯配置驱动的确定性路由 + 一个可选的 LLM 裁判筛选器。
RA 修复卡（路由信号层与容错修复，20260712）：
  T1 规模层（上下文规模 → tier 下限 + 大上下文跳裁判）
  T2 会话粘性（只升不降，auxiliary 不参与）
  T3 裁判 LRU + 兜底沿用本会话上次档
  T4 健康检查注入（熔断状态由 proxy 侧提供，core 只消费 Protocol）
  T5b req_model 显式指定（opt-in）
  T5d 删选池阶段 multimodal 直选步骤（偏好排序/兜底仍在能力过滤内）

流水线（逐级收窄）：

  [Phase 2 记忆 MUST 覆盖（预留槽，本卡不实现）]
          ↓
  auxiliary 内部调用 → 强制便宜档（T9 保留；不参与粘性）
          ↓
  req_model 显式指定（strategies.request 开启且命中名单 → 跳选池与裁判，仍过能力过滤）
          ↓
  规模层：context_chars → tier 下限 floor（只升不降的质量下限）
          ↓
  ① 选池（首个"开且命中"者）：
       a. agent 策略开 且 命中 → 池 = agent.map[key]（有序）
       b. filter 策略开 → 裁判判档（超 bypass 阈值跳裁判取 floor）→ 池 = 该档模型
       c. 否则 → 池 = upstream.models（有序）
          ↓
  ② 能力过滤（始终生效，正确性不变式）+ multimodal 偏好排序 / 空池兜底
          ↓
  规模 floor 过滤（池内 tier < floor 剔除；剔空则保留原池，floor 是偏好非硬约束）
          ↓
  健康过滤（熔断中的模型跳过；跳空则保留原池，宁试死模型不打死请求）
          ↓
  会话粘性（sticky 模型能力合格 且 新档 <= 粘住档 → 沿用，不乒乓）
      └─ G14/MQ-RT1：failover 被动粘上的先判释放（cache 冷 / 超 max_hold → 放回选池）
          ↓
  ③ failover：池首=primary，其余按序（保能力）

路由不改注入（叠加与模型无关，ADR-0008 §5.2）。
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections import OrderedDict
from enum import Enum
from typing import Protocol

from pydantic import BaseModel, Field

from bladex_core.fact import Fact, HardRule

logger = logging.getLogger(__name__)

# ── 能力词表（固定枚举，T4）─────────────────────────────────────────────────
# models.capabilities 与 multimodal.map 的键都只能用这些值。
VALID_CAPABILITIES: frozenset[str] = frozenset({"text", "vision", "audio", "file", "code"})
# multimodal.map 的键只允许这三个（text 不用配；code 由 capability.code_agents 静态注入）。
VALID_MODALITIES: frozenset[str] = frozenset({"vision", "audio", "file"})

# tier 序（单一真相，T1/T2/T5a 共用）
_TIER_ORDER: dict[str, int] = {"weak": 0, "medium": 1, "strong": 2}

# 裁判 LRU 容量（T3）
_JUDGE_CACHE_MAX = 512


def tier_rank(tier: str) -> int:
    """tier → 序号（未知 tier 按 weak=0 处理）。"""
    return _TIER_ORDER.get(tier, 0)


class RoutingError(Exception):
    """路由决策失败（调用方应降级到默认上游）。"""


class NoCapableCandidateError(RoutingError):
    """能力过滤无合格候选且无兜底（T4）。

    调用方应快返结构化错误（"agent X 集合无 vision 能力，请补多模态模型"），不静默降级。
    """


class RouteSource(str, Enum):
    """路由决策来源（流水线可解释）。

    HARD_RULE 为 Phase 2 记忆覆盖预留槽（本卡不接逻辑）。
    """

    HARD_RULE = "hard_rule"            # reserved for phase2 memory override
    AUXILIARY = "auxiliary"            # internal call forced to cheap tier
    # 注：静态/动态的划分见模块底部 _STATIC_SOURCES（CLAUDE.md 刚性原则 10）。
    REQUESTED = "requested"            # explicit req_model matched (T5b)
    TEAM = "team"                      # ADR-0021: team-level route (priority: team>principal>agent)
    PRINCIPAL = "principal"            # ADR-0021: principal-level route
    AGENT = "agent"                    # agent strategy matched
    FILTER = "filter"                  # LLM judge selected the tier
    SCALE = "scale"                    # context-scale floor changed the outcome (T1)
    STICKY = "sticky"                  # session sticky reused previous model (T2)
    UPSTREAM_DEFAULT = "upstream_default"  # default upstream pool
    CAPABILITY = "capability"          # multimodal map fallback used
    SENSITIVITY = "sensitivity"        # ADR-0021: sensitivity hard boundary (allowed pool empty)
    FAILOVER = "failover"              # failover to next candidate


# ── 静态 / 动态的划分（CLAUDE.md 刚性原则 10）─────────────────────────────────
#
# **静态**：来源是配置文件里写死的策略。用户写了什么就是什么，动态推断不得改写。
# **动态**：运行时根据请求内容/历史/负载推断出来的。只在静态未命中时决定路由。
# **硬约束**：客观事实（能力不满足 / 熔断 / 敏感度越界）。可以推翻静态配置——
#            因为那不是"判断"而是"这条路走不通"——但必须留可 grep 的告警，
#            让配置错误可被发现，而不是静默旁路。
#
# 2026-07-28 回归教训：auxiliary（动态）原本排在 req_model / 选池（静态）之前，
# 一轮被判 aux 就旁路掉用户配置——实测 Claude Code 配了 GLM-5.2，
# 82 轮走 auxiliary 便宜档、只有 1 轮走 requested。
_STATIC_SOURCES: frozenset[RouteSource] = frozenset({
    RouteSource.REQUESTED,
    RouteSource.TEAM,
    RouteSource.PRINCIPAL,
    RouteSource.AGENT,
})

_DYNAMIC_SOURCES: frozenset[RouteSource] = frozenset({
    RouteSource.AUXILIARY,
    RouteSource.FILTER,
    RouteSource.SCALE,
    RouteSource.STICKY,
})


class ModelCandidate(BaseModel):
    """一个上游模型候选（能力 + tier + provider）。

    tier: 档位名（"strong"/"medium"/"weak"，供人读 + 日志 + auxiliary 选档）。
    capabilities: 该模型支持的能力（如 ["text"]、["text","vision"]）。
    provider: 故障域分组（T4）——429 只跨 provider failover；空 = 未分组。
    """

    model: str
    api_base: str = ""
    api_key: str = ""
    tier: str = "weak"
    capabilities: list[str] = Field(default_factory=lambda: ["text"])
    provider: str = ""
    # ADR-0021 section 3.3a: 暴露等级 local/private/public（缺省 public = 最坏暴露假设）。
    # 敏感路由硬边界：候选 exposure 不得超过请求 allowed_exposure（T4 选池最前裁剪）。
    exposure: str = "public"
    # ADR-0028 §2：该模型的 Prompt Cache TTL（秒）。0 = 无 cache，永远按冷处理。
    # 温热期禁止新增裁剪/降解（打碎前缀省的 token 不够 cache 损失）；
    # 冷启期才允许相关性裁剪（E4.3）。
    cache_ttl_s: int = 300


class RouteDecision(BaseModel):
    """路由决策结果（可解释：source + reason）。"""

    model: str
    api_base: str = ""
    api_key: str = ""
    tier: str = "weak"
    provider: str = ""
    source: RouteSource = RouteSource.UPSTREAM_DEFAULT
    reason: str = ""
    # primary 被选出时所用的合格池（供 failover：同池内切换，保能力）。
    pool: list[ModelCandidate] = Field(default_factory=list)


# ── 策略数据（agent 中立，由 proxy 侧 routing_config 产）────────────────────


class AgentStrategyData(BaseModel):
    """agent 策略：enabled + agent_id → 有序模型名列表。

    map 的键支持两种粒度（T3）：
      "hermes:accept" —— 特定 profile（精确优先）
      "hermes"        —— base 兜底
    """

    enabled: bool = False
    map: dict[str, list[str]] = Field(default_factory=dict)


class PrincipalStrategyData(BaseModel):
    """principal 级路由策略（ADR-0021：team > principal > agent 优先级中间）。

    map 键 = principal_id -> 有序模型名列表。identity.toml 声明的 principal。
    """

    enabled: bool = False
    map: dict[str, list[str]] = Field(default_factory=dict)


class TeamStrategyData(BaseModel):
    """team 级路由策略（ADR-0021：优先级最高，覆盖 principal/agent）。

    map 键 = team_id -> 有序模型名列表。identity.toml 声明的 team（含祖先链）。
    """

    enabled: bool = False
    map: dict[str, list[str]] = Field(default_factory=dict)


class MultimodalStrategyData(BaseModel):
    """多模态策略：enabled + 模态 → 有序模型名列表。

    map 的键只允许 vision/audio/file（T4 能力词表）。
    enabled 只控制"偏好排序 + 兜底映射"，能力过滤始终生效。
    RA9：不再参与选池（带图请求照常走 agent/filter/upstream 判档）。
    """

    enabled: bool = False
    map: dict[str, list[str]] = Field(default_factory=dict)


class FilterStrategyData(BaseModel):
    """筛选器策略：LLM 裁判在三档候选池里判一档。

    candidates: tier(weak/medium/strong) → 有序模型名列表。
    default_tier: 裁判超时/失败/prompt 超限且本会话无粘性历史时落此档。
    max_prompt_chars: 超过此值不判，直接走 default_tier。
    judge_timeout_s: 裁判超时（T6，可配）。
    """

    enabled: bool = False
    candidates: dict[str, list[str]] = Field(default_factory=dict)
    default_tier: str = "weak"
    max_prompt_chars: int = 2000
    judge_timeout_s: float = 5.0


class ScaleStrategyData(BaseModel):
    """上下文规模策略（T1，RA1）：规模 → tier 下限 + 跳裁判。

    只设下限不设上限——规模大 = 质量下限抬高，绝不因"看起来简单"送弱模型。
    阈值为占位默认值，待真实流量校准（对照 Memory Hub 真实请求 chars 分布重标）。
    设 0 = 关闭对应档。
    """

    enabled: bool = False
    floor_medium_chars: int = 60000
    floor_strong_chars: int = 400000
    judge_bypass_chars: int = 60000


class StickyStrategyData(BaseModel):
    """会话粘性策略（T2，RA2）：同会话粘住所选模型，只升不降。

    G14 / MQ-RT1（2026-08-21）新增 `failover_max_hold_s`：**failover 被动粘上的**
    模型有释放条件，裁判/选池主动选出的没有（那是设计意图，不该释放）。
    """

    enabled: bool = False
    max_sessions: int = 500
    # failover 粘性的墙钟上限（秒）。0 = 完全不主动释放（回到 G14 之前的行为）。
    # 注意这是**兜底**不是主触发器：主触发器是"上游 cache 已经冷了"（见 CacheWarmth），
    # 冷了的时候释放代价为零——粘性存在的唯一理由就是复用 cache。
    # 本参数只用于兜住"会话一直高频往来、cache 始终热"导致无限期粘在降级模型上。
    failover_max_hold_s: int = 300


class StickyOrigin(str, Enum):
    """粘性是**怎么**粘上的——决定它该不该被自动释放（MQ-RT1）。

    ROUTED: 选池/裁判正常判出来的。粘住是设计意图（防乒乓、保 prompt cache），
            不自动释放，只在需要升档时切换。
    FAILOVER: 首选模型故障后被动切过去的。它代表的是**降级状态**，不是选择——
            故障恢复后必须能回去，否则一次瞬时抖动就让整个会话永久漂移
            （2026-08-21 实测：ARK 2 秒连接抖动 → 会话粘在本地 ollama 53 分钟）。
    """

    ROUTED = "routed"
    FAILOVER = "failover"


class StickyEntry(BaseModel):
    """一条会话粘性记录。

    G14 之前这里是裸 `tuple[str, str]`（model, tier）——没有"当初为什么粘上的"，
    于是 failover 被动粘上的和裁判主动判出的完全同权，无法分别处置。
    """

    model: str
    tier: str
    origin: StickyOrigin = StickyOrigin.ROUTED
    # 供 `/admin/sticky/flush --agent` 按 agent 定向刷新（粘性表本身按 session 键）。
    agent_id: str = ""
    acquired_ts: float = 0.0


class RequestStrategyData(BaseModel):
    """req_model 显式指定策略（T5b，RA7）。

    默认关：agent 配置的静态模型名若恰好命中名单，会静默旁路整个路由——
    必须由用户显式打开，并理解其后果。
    """

    enabled: bool = False
    # per-agent 覆盖全局 enabled（ADR-0013 增补 2026-07-25）。agent_id -> bool，
    # 命中覆盖表以此为准，否则用全局 enabled。两级查找：精确 profile（"hermes:accept"）
    # > base（"hermes"）。全局关时对指定 agent 放开（如 claude-code 自带 model 名），
    # 全局开时对指定 agent 关闭（如 hermes 强制走选池防静态模型名旁路）。空 = 现状。
    agent_overrides: dict[str, bool] = Field(default_factory=dict)


class JudgeResult(BaseModel):
    """裁判结果：判出的档位 + 理由。"""

    tier: str  # weak/medium/strong
    reason: str = ""


class Judge(Protocol):
    """LLM 裁判接口（agent 中立）。proxy 侧提供 Router 实现。"""

    async def judge(self, prompt: str) -> JudgeResult:
        """判 prompt 属于哪一档（weak/medium/strong）。失败应抛异常（路由器兜底）。"""
        ...


class HealthCheck(Protocol):
    """模型健康检查接口（T4 熔断）。proxy 侧提供实现（ModelHealth）。

    core 只消费：选池时跳过 unhealthy 候选（跳空则保留原池）。
    """

    def is_healthy(self, model: str) -> bool:
        ...


class CacheWarmth(Protocol):
    """上游 Prompt Cache 温度查询（G14 / MQ-RT1）。proxy 侧提供实现。

    照 HealthCheck 同款：core 只消费 Protocol，实现留在 proxy
    （温度表 `cache_state.SessionCacheRegistry` 在 proxy 侧，core 不依赖它）。

    **为什么粘性释放要看 cache 温度**：粘性存在的唯一理由是复用上游 prompt cache
    （见 cache_state.py 模块 docstring：「粘性让模型别乱换，本模块决定在模型
    没换的前提下能不能改前缀」）。所以 cache 一旦冷掉，粘性就什么都没在保护了，
    此刻释放代价为零；反过来 cache 还热时释放，等于白付一次全量 prefill。
    这比"固定 N 分钟计时器"严格更优——计时器既可能在热的时候触发，
    也可能在早就冷透之后还不触发。

    实现为 None（未注入）时：只剩 `failover_max_hold_s` 墙钟兜底。
    """

    def is_warm(self, session_id: str, model: str) -> bool:
        ...


class MemoryAwareRouter:
    """配置驱动的确定性路由（Phase 1 + RA 修复）。

    流水线见模块 docstring。所有策略默认关闭 → 回归纯顺序 failover（upstream.models）。
    """

    def __init__(
        self,
        models: dict[str, ModelCandidate],
        upstream: list[str],
        agent: AgentStrategyData | None = None,
        principal: PrincipalStrategyData | None = None,
        team: TeamStrategyData | None = None,
        multimodal: MultimodalStrategyData | None = None,
        filter_strategy: FilterStrategyData | None = None,
        judge: Judge | None = None,
        judge_model: str = "",
        aux_tier: str = "weak",
        scale: ScaleStrategyData | None = None,
        sticky: StickyStrategyData | None = None,
        request_strategy: RequestStrategyData | None = None,
        health: HealthCheck | None = None,
        cache_warmth: CacheWarmth | None = None,
    ) -> None:
        if not upstream:
            raise RoutingError("router needs at least one upstream model")
        self._models = models
        self._upstream = upstream
        self._agent = agent or AgentStrategyData()
        self._principal = principal or PrincipalStrategyData()  # ADR-0021: team>principal>agent
        self._team = team or TeamStrategyData()
        self._multimodal = multimodal or MultimodalStrategyData()
        self._filter = filter_strategy or FilterStrategyData()
        self._judge = judge
        self._judge_model = judge_model  # ADR-0021: 裁判钉本地校验用
        self._aux_tier = aux_tier
        self._scale = scale or ScaleStrategyData()
        self._sticky_cfg = sticky or StickyStrategyData()
        self._request = request_strategy or RequestStrategyData()
        self._health = health
        self._cache_warmth = cache_warmth  # G14/MQ-RT1：failover 粘性的释放判据
        # T2: session_id → StickyEntry，LRU
        self._sticky: OrderedDict[str, StickyEntry] = OrderedDict()
        # T3: (session_id, query_hash) → tier，LRU
        self._judge_cache: OrderedDict[tuple[str, str], str] = OrderedDict()
        # ADR-0021 section 3.3b: current request's allowed candidate set (exposure <= allowed_exposure).
        # None = no filter (public/disabled = current behavior); non-None = _resolve_names skips models outside it.
        self._current_allowed: set[str] | None = None
        # T5b: 展示名（去 provider 前缀）→ 全名，供 req_model 命中
        self._display_lookup: dict[str, str] = {}
        for name in models:
            display = name.split("/", 1)[-1] if "/" in name else name
            self._display_lookup.setdefault(display, name)

    async def route(
        self,
        query: str,
        *,
        requires: list[str] | None = None,
        agent_id: str | None = None,
        principal_id: str | None = None,
        team_ids: list[str] | None = None,
        auxiliary: bool = False,
        session_id: str | None = None,
        context_chars: int | None = None,
        req_model: str | None = None,
        allowed_exposure: str = "public",
        # Phase 2 预留（本卡不实现，签名保持兼容）
        facts: list[Fact] | None = None,
        hard_rules: list[HardRule] | None = None,
    ) -> RouteDecision:
        """返回路由决策（固定流水线）。

        requires: 请求需要的能力（如 ["vision"]），能力过滤全程用。
        agent_id: 当前请求的 agent（base:profile 形式，ADR-0010）。
        session_id: 会话标识（T2 粘性 / T3 裁判 LRU 用；None = 关闭会话态）。
        context_chars: 请求上下文规模估算（T1 规模层用；None = 关闭规模层）。
        req_model: 客户端显式请求的模型（T5b；strategies.request 开启才生效）。
        facts / hard_rules: Phase 2 记忆覆盖用（本卡不接逻辑）。
        """
        # Phase 2 预留：HARD_RULE 记忆 MUST/NEVER 覆盖（本卡不实现）。
        _ = self._memory_override(facts, hard_rules, [])  # hook 保留，Phase 1 返回 None

        # ADR-0021 section 3.3b: 敏感度选池裁剪（候选集裁为 exposure <= allowed_exposure）。
        # ADR-0028 §1.2 修订：这条裁剪**只作用于智能路由**。静态命中时不裁剪、静态胜出，
        # 只留强告警——与"能力不满足=配置错误应正常报错、程序不替用户改选择"同一原则。
        # 安全底线不破的理由：注入平面的 exposure 守卫独立于路由（ADR-0021 §3.3c），
        # 敏感 Fact 无论路由到哪都不会被注入到越界目的地。
        allowed = self._allowed_names(allowed_exposure)
        judge_ok = self._judge_within(allowed_exposure)

        # ══ 静态配置优先（CLAUDE.md 刚性原则 9 / ADR-0028 §1.1）════════════════
        # 配置文件里写死的策略压过一切动态推断与敏感度裁剪。
        # 顺序：req_model → team → principal → agent（first-match-wins）。
        # 命中即 short-circuit：跳过 敏感度选池裁剪 / aux 降档 / 规模 floor / 裁判 / 粘性
        # 整段；只有硬约束（熔断、failover）还能推翻它，且必须留可 grep 的告警。
        #
        # 📌 已接受偏差（ADR-0028 复核 §3.3，**不要"纠偏"**）：任务卡伪代码把静态匹配
        # 画在注入之前，实现放在 route() 开头（即注入之后）。语义完全等价——注入不
        # 依赖路由结果、静态命中照样 short-circuit 全部动态层（裁判调用数=0 有测试钉死）
        # ——且少改一处编排。谁下次照着伪代码搬编排，先看这条和
        # `test_adr0028_e3_route_order.py`。

        # T5b: req_model 显式指定（opt-in；per-agent 覆盖表优先于全局 enabled）
        if req_model and self._request_enabled_for(agent_id):
            explicit = self._match_requested(req_model)
            if explicit is not None:
                self._warn_static_sensitivity(
                    RouteSource.REQUESTED, explicit.model, allowed, allowed_exposure)
                return self._route_requested(explicit, requires, session_id, None,
                                             agent_id=agent_id)

        # ① 静态选池（team → principal → agent）——**不带敏感度裁剪**（静态胜出）
        static_pool, static_source = self._select_static_pool(
            agent_id, principal_id=principal_id, team_ids=team_ids,
        )
        static_hit = static_pool is not None and static_source is not None
        if static_hit:
            self._warn_static_sensitivity(
                static_source, static_pool[0].model, allowed, allowed_exposure)

        # 静态未命中时才轮到敏感硬边界：允许池空 → 快返错（fail-closed 不变）
        if not static_hit and allowed is not None and not allowed:
            raise NoCapableCandidateError(
                f"sensitivity: no model with exposure <= {allowed_exposure}"
            )

        # T1: 规模层 floor（动态；只在静态选池未命中时参与，见下）
        floor_tier = self._scale_floor(context_chars)

        if static_hit:
            pool, source = static_pool, static_source
        else:
            # ② 智能路由选池（filter 裁判 → upstream 兜底），在敏感度裁剪后的集合内
            pool, source = await self._select_dynamic_pool(
                query, floor_tier=floor_tier,
                context_chars=context_chars, session_id=session_id,
                allowed=allowed, judge_ok=judge_ok,
            )

        # ══ 动态推断（仅在静态策略未命中时生效）══════════════════════════════════
        # auxiliary 内部调用 → 强制便宜档（ADR-0013 T9；不参与粘性）。
        # 🔴 2026-07-28 回归修复：本段原本排在 req_model 与选池**之前**，导致
        # 一轮被判 auxiliary 就旁路掉用户配置（实测 Claude Code `source=auxiliary` 82 次
        # vs `source=requested` 1 次，配置的 GLM-5.2 形同虚设）。
        # 现在它排在静态策略之后，且静态命中时整段跳过。
        if auxiliary and not static_hit:
            aux_pool = self._aux_pool(requires, allowed)
            if aux_pool:
                return self._decision(
                    aux_pool, RouteSource.AUXILIARY,
                    f"auxiliary call -> cheap tier {self._aux_tier} ({aux_pool[0].model})",
                )
            logger.warning(
                "route_aux_no_capable tier=%s requires=%s -> normal routing",
                self._aux_tier, requires,
            )
        elif auxiliary and static_hit:
            logger.info(
                "route_static_over_aux source=%s agent=%s "
                "(static config wins over auxiliary cheap-tier)",
                source.value, agent_id,
            )

        # ② 能力过滤
        # 刚性原则 9（2026-07-28 拍板）：静态策略命中时**不做能力过滤**——
        # 用户在配置里钉死了模型，模型不支持该模态就该正常报错，
        # 那是**配置问题不是程序问题**，程序不替用户改选择。
        # 但"报错"必须能指向配置：这里留一条可 grep 的告警，否则用户只会看到
        # 上游那句语焉不详的 "Model only support text input"（ADR-0020 T3 的形状）。
        if source in (RouteSource.TEAM, RouteSource.PRINCIPAL, RouteSource.AGENT):
            if requires and pool:
                missing = sorted(set(requires) - set(pool[0].capabilities))
                if missing:
                    logger.warning(
                        "route_static_capability_mismatch source=%s agent=%s model=%s "
                        "missing=%s -- config pins a model lacking required capability; "
                        "upstream will reject. Fix routing.toml, not the code.",
                        source.value, agent_id, pool[0].model, missing,
                    )
        else:
            pool, cap_override = self._apply_capability_filter(pool, requires, allowed)
            if cap_override is not None:
                source = cap_override

        # filter 兜底：判出档经能力过滤空 → 退 default_tier（再过能力；SCALE 亦来自 filter 路径）
            if not pool and source in (RouteSource.FILTER, RouteSource.SCALE):
                default_names = self._filter.candidates.get(self._filter.default_tier, [])
                default_pool = self._resolve_names(default_names, allowed)
                if default_pool:
                    pool, cap_override = self._apply_capability_filter(default_pool, requires, allowed)
                    if cap_override is not None:
                        source = cap_override

        # ③ empty pool → 结构化错误（不静默降级）
        if not pool:
            raise NoCapableCandidateError(
                f"no candidate with capabilities {sorted(set(requires or []))} "
                f"for source={source.value}"
            )

        # T1: floor 过滤（偏好性下限：剔空则保留原池 + warning，不打死请求）
        # 刚性原则 10：规模层是**动态推断**，不得改写静态策略选出的池。
        if floor_tier is not None and not static_hit:
            floored = [c for c in pool if tier_rank(c.tier) >= tier_rank(floor_tier)]
            if floored:
                if floored[0].model != pool[0].model:
                    source = RouteSource.SCALE
                pool = floored
            else:
                logger.warning(
                    "route_scale_floor_empty floor=%s context_chars=%s -> keeping pool",
                    floor_tier, context_chars,
                )

        # T4: 健康过滤（跳空则保留原池——宁试死模型不打死请求）
        # G14/MQ-RT1：记下"这次选择是不是被熔断改写过"——被熔断挤掉首选而选中的模型，
        # 与 failover 切过去的是同一件事（都是"首选走不通"的降级态，不是主动选择），
        # 所以粘性也必须标 FAILOVER，否则释放机制在这条路径上被绕过：
        # 释放 → 重选池 → 首选仍熔断 → 选中同一个降级模型 → 却标成 ROUTED → 从此永不释放。
        pre_health_pool = pool
        pool = self._filter_healthy(pool)
        health_degraded = bool(pool) and pool[0].model != pre_health_pool[0].model

        # T2: 会话粘性（只升不降）
        # 刚性原则 10：粘性是**动态推断**，静态策略命中时不得改写其选择——
        # 否则同一 agent 配了固定模型，却因为上一轮粘住别的模型而被改掉。
        if not static_hit:
            sticky_d = self._try_sticky(session_id, pool, requires, allowed)
            if sticky_d is not None:
                return sticky_d

        d = self._decision(
            pool, source,
            self._pool_reason(source, agent_id, requires, context_chars, floor_tier),
        )
        # 正常选池/裁判判出 → origin=ROUTED（设计意图的粘性，不自动释放）；
        # 首选被熔断挤掉才选中的 → FAILOVER（降级态，到点释放）。
        self.update_sticky(
            session_id, d.model, agent_id=agent_id,
            origin=StickyOrigin.FAILOVER if health_degraded else StickyOrigin.ROUTED,
        )
        return d

    def failover_candidates(self, decision: RouteDecision) -> list[ModelCandidate]:
        """failover 列表：primary 所在合格池内、排除 primary，保池顺序。

        保能力（池已按能力过滤过，T4）。同 tier 优先，其余按池顺序。
        """
        pool = decision.pool or []
        rest = [c for c in pool if c.model != decision.model]
        same_tier = [c for c in rest if c.tier == decision.tier]
        other = [c for c in rest if c.tier != decision.tier]
        return same_tier + other

    # ── T2: 会话粘性 ────────────────────────────────────────────────────────

    def update_sticky(
        self,
        session_id: str | None,
        model: str,
        *,
        origin: StickyOrigin = StickyOrigin.ROUTED,
        agent_id: str | None = None,
    ) -> None:
        """记录/更新会话粘住的模型（决策时与 failover 实际切换后都会调）。

        `origin` 决定这条粘性能否被自动释放（MQ-RT1）——调用方必须如实标：
        proxy 侧 `on_success` 拿到的 used_model != 决策 primary 时即 FAILOVER。

        已存在同模型的记录时**保留原 origin 与 acquired_ts**：同一次 failover
        持有期内每轮都会回写一次 on_success，若每次都刷新 acquired_ts，
        墙钟上限永远走不到头（held_s 每轮归零）。
        """
        if not self._sticky_cfg.enabled or not session_id:
            return
        cand = self._models.get(model)
        if cand is None:
            return
        prev = self._sticky.get(session_id)
        if prev is not None and prev.model == cand.model:
            # 同模型续期：只更 agent_id，origin/acquired_ts 保持——见 docstring。
            if agent_id:
                prev.agent_id = agent_id
            self._sticky.move_to_end(session_id)
            return
        self._sticky[session_id] = StickyEntry(
            model=cand.model, tier=cand.tier, origin=origin,
            agent_id=agent_id or (prev.agent_id if prev is not None else ""),
            acquired_ts=time.time(),
        )
        self._sticky.move_to_end(session_id)
        while len(self._sticky) > self._sticky_cfg.max_sessions:
            self._sticky.popitem(last=False)

    def flush_sticky(self, agent_id: str | None = None) -> int:
        """强制清空粘性表，返回清掉的条数（供 `/admin/sticky/flush`）。

        `agent_id=None` → 全局；否则只清该 agent 的会话。两级匹配与
        `[strategies.agent.map]` 同规则：给 "hermes" 连 "hermes:default"
        一起清（base 前缀），给 "hermes:default" 只清那一个（精确）。
        """
        if agent_id is None:
            n = len(self._sticky)
            self._sticky.clear()
            return n
        victims = [
            sid for sid, e in self._sticky.items()
            if e.agent_id == agent_id
            or (":" not in agent_id and e.agent_id.split(":", 1)[0] == agent_id)
        ]
        for sid in victims:
            self._sticky.pop(sid, None)
        return len(victims)

    def sticky_snapshot(self) -> list[dict[str, object]]:
        """粘性表当前内容（供 admin 只读展示；不含秘密）。"""
        now = time.time()
        return [
            {
                "session_id": sid,
                "model": e.model,
                "tier": e.tier,
                "origin": e.origin.value,
                "agent_id": e.agent_id,
                "held_s": round(now - e.acquired_ts, 1) if e.acquired_ts else None,
            }
            for sid, e in self._sticky.items()
        ]

    def _sticky_release_reason(self, session_id: str, entry: StickyEntry) -> str | None:
        """failover 粘性该不该释放？返回释放原因（可 grep），不释放返 None。

        只作用于 `origin=FAILOVER` 的记录——裁判/选池主动选出的粘性是设计意图。

        两个触发器，任一成立即释放：
          1. **cache 已冷**（主）：粘性此刻没在保护任何东西，释放代价为零。
             Hermes 自己压缩过历史（prefix_changed）也会让 cache 判冷，
             那类轮次占真实流量 19–28%，所以通常等不到墙钟就有释放窗口。
          2. **持有超过 failover_max_hold_s**（兜底）：防"会话一直高频往来、
             cache 始终热"导致无限期粘在降级模型上。0 = 关闭全部自动释放。

        释放后不需要"探测首选是否已恢复"——正常重选池自然会拿到 pool[0]，
        通了就回去了，没通就再 failover 再粘上（自校正）。
        """
        if entry.origin is not StickyOrigin.FAILOVER:
            return None
        max_hold = self._sticky_cfg.failover_max_hold_s
        if max_hold <= 0:
            return None  # 显式关闭自动释放（回到 G14 之前的行为）
        if self._cache_warmth is not None and not self._cache_warmth.is_warm(
            session_id, entry.model
        ):
            return "cache_cold"
        if entry.acquired_ts and (time.time() - entry.acquired_ts) >= max_hold:
            return "max_hold"
        return None

    def _try_sticky(
        self,
        session_id: str | None,
        pool: list[ModelCandidate],
        requires: list[str] | None,
        allowed: set[str] | None = None,
    ) -> RouteDecision | None:
        """粘性判定：新档 <= 粘住档 且 sticky 模型能力合格、健康 → 沿用。

        注意用能力直查而非池成员判断——filter 的三档池互斥，
        判 weak 时 sticky 的 medium 模型不在 weak 池内，但正是"不降档"要保的对象。
        """
        if not self._sticky_cfg.enabled or not session_id or not pool:
            return None
        entry = self._sticky.get(session_id)
        if entry is None:
            return None
        model, sticky_tier = entry.model, entry.tier
        cand = self._models.get(model)
        if cand is None:
            self._sticky.pop(session_id, None)
            return None
        # MQ-RT1：failover 被动粘上的 → 到点释放，回正常选池（下轮重新粘）。
        release = self._sticky_release_reason(session_id, entry)
        if release is not None:
            held = time.time() - entry.acquired_ts if entry.acquired_ts else -1.0
            logger.info(
                "route_sticky_released session=%s model=%s reason=%s held_s=%.1f agent=%s",
                session_id[:16], entry.model, release, held, entry.agent_id or "-",
            )
            self._sticky.pop(session_id, None)
            return None
        if allowed is not None and model not in allowed:
            return None  # ADR-0021: sticky 模型越界 exposure -> 不沿用
        new_tier = pool[0].tier
        if tier_rank(new_tier) > tier_rank(sticky_tier):
            return None  # 需要升级 → 切换（调用方随后 update_sticky）
        if requires and not set(requires).issubset(set(cand.capabilities)):
            return None  # 能力不合格（如带图但 sticky 无 vision）
        if self._health is not None and not self._health.is_healthy(cand.model):
            return None
        self._sticky.move_to_end(session_id)
        rest = [c for c in pool if c.model != cand.model]
        # MQ-RT3：reason 里带上 origin 与持有时长——G14 之前 failover 造成的
        # 长时间降级漂移在日志里读起来跟正常粘性完全一样，只能靠人撞见。
        held = time.time() - entry.acquired_ts if entry.acquired_ts else -1.0
        return self._decision(
            [cand] + rest, RouteSource.STICKY,
            f"sticky session model {cand.model} (tier {sticky_tier} >= new {new_tier}"
            f", origin={entry.origin.value}, held_s={held:.0f})",
        )

    # ── ① 选池 ──────────────────────────────────────────────────────────────

    def _warn_static_sensitivity(
        self,
        source: RouteSource,
        model: str,
        allowed: set[str] | None,
        allowed_exposure: str,
    ) -> None:
        """ADR-0028 §1.2：静态命中 + 敏感度越界 → 静态胜出 + 强告警（可 grep）。

        为什么不拒绝：与刚性原则 9「能力不满足 = 配置错误，程序不替用户改选择」同源
        ——用户在 routing.toml 里钉死了模型，敏感度层不该静默换掉他的选择。
        为什么安全底线不破：注入平面的 exposure 守卫独立于路由（ADR-0021 §3.3c），
        敏感 Fact 不会被注入到越界目的地。但这件事**必须可被发现**，故留告警。
        """
        if allowed is None or model in allowed:
            return
        logger.warning(
            "route_static_sensitivity_mismatch strategy=%s model=%s "
            "allowed_exposure=%s -- static config wins over sensitivity cut; "
            "the memory injection guard still blocks sensitive facts. "
            "Fix routing.toml if this is not intended.",
            source.value, model, allowed_exposure,
        )

    def _select_static_pool(
        self,
        agent_id: str | None,
        *,
        principal_id: str | None = None,
        team_ids: list[str] | None = None,
    ) -> tuple[list[ModelCandidate] | None, RouteSource | None]:
        """静态选池：team → principal → agent（first-match-wins）。

        ADR-0028 §1.1：**不带敏感度裁剪**——静态配置优先级高于敏感度选池裁剪，
        越界只告警（`_warn_static_sensitivity`）不改写。未命中返回 (None, None)。
        """
        # a. team 级（优先级最高，ADR-0021 team>principal>agent）
        if self._team.enabled and team_ids:
            for tid in team_ids:
                names = self._team.map.get(tid)
                if names:
                    pool = self._resolve_names(names, None)
                    if pool:
                        return pool, RouteSource.TEAM
        # b. principal 级
        if self._principal.enabled and principal_id:
            names = self._principal.map.get(principal_id)
            if names:
                pool = self._resolve_names(names, None)
                if pool:
                    return pool, RouteSource.PRINCIPAL
        # c. agent 策略开 且命中
        if self._agent.enabled and agent_id:
            names = self._match_agent(agent_id)
            if names:
                pool = self._resolve_names(names, None)
                if pool:
                    return pool, RouteSource.AGENT
        return None, None

    async def _select_dynamic_pool(
        self,
        query: str,
        *,
        floor_tier: str | None = None,
        context_chars: int | None = None,
        session_id: str | None = None,
        allowed: set[str] | None = None,
        judge_ok: bool = True,
    ) -> tuple[list[ModelCandidate], RouteSource]:
        """智能路由选池：filter 裁判 → upstream 兜底（在敏感度裁剪后的集合内）。

        只在静态策略未命中时调用（ADR-0028 §1.1）。
        RA9：multimodal 不再参与选池（偏好排序/兜底在能力过滤内，避免
        任何带图请求绕过档位判断直上强模型）。
        """
        # b. filter 策略开（T1: 超 bypass 阈值跳裁判）
        if self._filter.enabled:
            bypass = (
                self._scale.enabled
                and context_chars is not None
                and self._scale.judge_bypass_chars > 0
                and context_chars > self._scale.judge_bypass_chars
            )
            if bypass:
                tier = floor_tier or self._filter.default_tier
                if tier_rank(self._filter.default_tier) > tier_rank(tier):
                    tier = self._filter.default_tier
                src = RouteSource.SCALE
                logger.info(
                    "route_scale_judge_bypass context_chars=%s tier=%s",
                    context_chars, tier,
                )
            elif not judge_ok:
                # ADR-0021 section 3.2: sensitive mode + judge not local -> skip judge,
                # use allowed-pool default_tier (sensitive query never sent to judge).
                tier = floor_tier or self._filter.default_tier
                if tier_rank(self._filter.default_tier) > tier_rank(tier):
                    tier = self._filter.default_tier
                src = RouteSource.SCALE
                logger.info("route_sensitivity_skip_judge tier=%s (judge not local)", tier)
            else:
                tier = await self._judge_tier(query, session_id)
                src = RouteSource.FILTER
                if floor_tier is not None and tier_rank(floor_tier) > tier_rank(tier):
                    logger.info(
                        "route_scale_floor_raise judged=%s floor=%s context_chars=%s",
                        tier, floor_tier, context_chars,
                    )
                    tier = floor_tier
                    src = RouteSource.SCALE
            names = self._filter.candidates.get(tier, [])
            pool = self._resolve_names(names, allowed)
            if pool:
                return pool, src

        # c. upstream 默认池
        pool = self._resolve_names(self._upstream, allowed)
        return pool, RouteSource.UPSTREAM_DEFAULT

    def _match_agent(self, agent_id: str) -> list[str] | None:
        """agent 策略两级查找（T3）：精确 profile 优先、base 兜底。

        "hermes:accept" → 先查 "hermes:accept"，再查 "hermes"
        "hermes"        → 只查 "hermes"
        """
        # 精确 profile（如 "hermes:accept"）
        names = self._agent.map.get(agent_id)
        if names:
            return names
        # base 兜底（如 "hermes" from "hermes:accept"）
        if ":" in agent_id:
            base = agent_id.split(":", 1)[0]
            names = self._agent.map.get(base)
            if names:
                return names
        return None

    async def _judge_tier(self, query: str, session_id: str | None = None) -> str:
        """调 LLM 裁判判一档（T3: LRU 缓存 + 兜底沿用本会话上次档）。

        prompt 超限 → default_tier（不缓存）。
        失败/超时 → 本会话粘住档（有）> default_tier（无）——消除"越抖越弱"。
        """
        if len(query) > self._filter.max_prompt_chars:
            logger.info(
                "filter_skip_large_prompt len=%d max=%d -> default_tier %s",
                len(query), self._filter.max_prompt_chars, self._filter.default_tier,
            )
            return self._filter.default_tier
        if self._judge is None:
            return self._filter.default_tier

        # T3: LRU 命中（工具循环 N 次 roundtrip 不重判同一 query）
        key = (session_id or "", hashlib.sha256(query.encode()).hexdigest()[:16])
        cached = self._judge_cache.get(key)
        if cached is not None:
            self._judge_cache.move_to_end(key)
            logger.info("filter_judge_cache_hit tier=%s", cached)
            return cached

        try:
            result = await asyncio.wait_for(
                self._judge.judge(query), timeout=self._filter.judge_timeout_s,
            )
            if result.tier in ("weak", "medium", "strong"):
                logger.info("filter_judged tier=%s reason=%s", result.tier, result.reason[:80])
                self._judge_cache[key] = result.tier
                self._judge_cache.move_to_end(key)
                while len(self._judge_cache) > _JUDGE_CACHE_MAX:
                    self._judge_cache.popitem(last=False)
                return result.tier
            logger.warning("filter_invalid_tier %s -> fallback", result.tier)
            return self._judge_fallback_tier(session_id)
        except TimeoutError:
            logger.warning("filter_judge_timeout -> fallback")
            return self._judge_fallback_tier(session_id)
        except Exception as e:  # noqa: BLE001
            logger.warning("filter_judge_failed err=%s -> fallback", e)
            return self._judge_fallback_tier(session_id)

    def _judge_fallback_tier(self, session_id: str | None) -> str:
        """裁判失败兜底：本会话上次档（T3）> default_tier。"""
        if session_id:
            entry = self._sticky.get(session_id)
            if entry is not None:
                logger.info("filter_fallback_sticky_tier tier=%s", entry.tier)
                return entry.tier
        return self._filter.default_tier

    # ── T1: 规模层 ──────────────────────────────────────────────────────────

    def _scale_floor(self, context_chars: int | None) -> str | None:
        """上下文规模 → tier 下限（None = 不设限）。"""
        if not self._scale.enabled or context_chars is None:
            return None
        if self._scale.floor_strong_chars > 0 and context_chars > self._scale.floor_strong_chars:
            return "strong"
        if self._scale.floor_medium_chars > 0 and context_chars > self._scale.floor_medium_chars:
            return "medium"
        return None

    # ── T5b: req_model 显式指定 ─────────────────────────────────────────────

    def _request_enabled_for(self, agent_id: str | None) -> bool:
        """req_model 开关是否对该 agent 生效：per-agent 覆盖 > 全局 enabled。

        两级查找（复用 _match_agent 模式）：
        1. 精确 profile（含 scope，如 "hermes:accept"）
        2. base 兜底（"hermes"）

        覆盖表命中以此为准；未命中走全局 enabled；agent_id=None（unknown）走全局。
        """
        ov = self._request.agent_overrides
        if agent_id and agent_id in ov:
            return ov[agent_id]
        if agent_id and ":" in agent_id:
            base = agent_id.split(":", 1)[0]
            if base in ov:
                return ov[base]
        return self._request.enabled

    def _match_requested(self, req_model: str) -> ModelCandidate | None:
        """req_model 命中名单：全名精确 > 展示名（去 provider 前缀）。"""
        cand = self._models.get(req_model)
        if cand is not None:
            return cand
        full = self._display_lookup.get(req_model)
        if full is not None:
            return self._models.get(full)
        return None

    def _route_requested(
        self,
        explicit: ModelCandidate,
        requires: list[str] | None,
        session_id: str | None,
        allowed: set[str] | None = None,
        agent_id: str | None = None,
    ) -> RouteDecision:
        """显式指定：跳选池与裁判，仍过能力过滤（无能力 → 快返，同 §3.4 语义）。"""
        # ADR-0021: explicit request out of allowed exposure -> refuse (sensitivity overrides user pick)
        if allowed is not None and explicit.model not in allowed:
            raise NoCapableCandidateError(
                f"requested model {explicit.model} out of allowed exposure"
            )
        if requires and not set(requires).issubset(set(explicit.capabilities)):
            raise NoCapableCandidateError(
                f"requested model {explicit.model} lacks capabilities "
                f"{sorted(set(requires) - set(explicit.capabilities))}"
            )
        # failover 池：同 tier 能力合格候选（保档保能力）
        extra = [
            c for c in self._models.values()
            if c.model != explicit.model and c.tier == explicit.tier
        ]
        if allowed is not None:
            extra = [c for c in extra if c.model in allowed]
        if requires:
            req = set(requires)
            extra = [c for c in extra if req.issubset(set(c.capabilities))]
        d = self._decision(
            [explicit] + extra, RouteSource.REQUESTED,
            f"explicit model request {explicit.model}",
        )
        self.update_sticky(session_id, d.model, agent_id=agent_id)
        return d

    # ── ② 能力过滤（始终生效）────────────────────────────────────────────────

    def _apply_capability_filter(
        self, pool: list[ModelCandidate], requires: list[str] | None,
        allowed: set[str] | None = None,
    ) -> tuple[list[ModelCandidate], RouteSource | None]:
        """在池上过滤到 requires ⊆ capabilities 的模型（T4，正确性不变式）。

        返回 (过滤后池, source_override)。
        source_override = CAPABILITY（multimodal 兜底用了映射），否则 None。
        不抛错——空池由调用方处理（filter retry / 最终错误）。
        """
        if not requires:
            # 纯文本：不约束，但 multimodal 偏好排序仍可生效
            if self._multimodal.enabled:
                pool = self._apply_multimodal_preference(pool, requires)
            return pool, None

        req = set(requires)
        capable = [c for c in pool if req.issubset(set(c.capabilities))]

        if capable:
            # multimodal 偏好排序：mapped 模型排前
            if self._multimodal.enabled:
                capable = self._apply_multimodal_preference(capable, requires)
            return capable, None

        # 池被能力过滤清空 → multimodal 映射兜底
        if self._multimodal.enabled:
            for modality in requires:
                mapped_names = self._multimodal.map.get(modality, [])
                if mapped_names:
                    mapped = self._resolve_names(mapped_names, allowed)
                    mapped_capable = [c for c in mapped if req.issubset(set(c.capabilities))]
                    if mapped_capable:
                        logger.warning(
                            "route_capability_fallback requires=%s via multimodal.map[%s]",
                            sorted(req), modality,
                        )
                        return mapped_capable, RouteSource.CAPABILITY

        # 空池，无兜底
        return [], None

    def _apply_multimodal_preference(
        self, pool: list[ModelCandidate], requires: list[str] | None,
    ) -> list[ModelCandidate]:
        """multimodal 偏好：map 内模型排前（不剔除其余）。"""
        mapped_names: set[str] = set()
        for modality in requires or []:
            mapped_names.update(self._multimodal.map.get(modality, []))
        if not mapped_names:
            return pool
        preferred = [c for c in pool if c.model in mapped_names]
        rest = [c for c in pool if c.model not in mapped_names]
        return preferred + rest

    # ── T4: 健康过滤 ────────────────────────────────────────────────────────

    def _filter_healthy(self, pool: list[ModelCandidate]) -> list[ModelCandidate]:
        """跳过熔断中的候选；跳空则保留原池 + warning（不打死请求）。"""
        if self._health is None or not pool:
            return pool
        healthy = [c for c in pool if self._health.is_healthy(c.model)]
        if healthy:
            if len(healthy) < len(pool):
                skipped = [c.model for c in pool if c not in healthy]
                logger.info("route_skip_unhealthy models=%s", skipped)
            return healthy
        logger.warning(
            "route_all_unhealthy pool=%s -> keeping pool",
            [c.model for c in pool],
        )
        return pool

    # ── auxiliary（T9 保留）─────────────────────────────────────────────────

    def _aux_pool(
        self, requires: list[str] | None, allowed: set[str] | None = None,
    ) -> list[ModelCandidate]:
        """auxiliary 调用：选 aux_tier 档且满足 requires 的模型（保池顺序）。"""
        aux = [c for c in self._models.values() if c.tier == self._aux_tier]
        if allowed is not None:
            aux = [c for c in aux if c.model in allowed]
        if requires:
            req = set(requires)
            aux = [c for c in aux if req.issubset(set(c.capabilities))]
        return aux

    # ── 工具方法 ────────────────────────────────────────────────────────────

    def _resolve_names(
        self, names: list[str], allowed: set[str] | None = None,
    ) -> list[ModelCandidate]:
        """把模型名列表解析成 ModelCandidate（保持顺序；未知模型跳过 + warning）。

        allowed（ADR-0021 敏感硬边界）非 None 时，跳过不在允许集合内的模型
        （exposure 越界 = 视同不存在）。None = 不过滤（关闭态/现状）。
        """
        out: list[ModelCandidate] = []
        for name in names:
            if allowed is not None and name not in allowed:
                logger.info("route_skip_out_of_exposure model=%s", name)
                continue
            cand = self._models.get(name)
            if cand is not None:
                out.append(cand)
            else:
                logger.warning("route_unknown_model %s -> skipped", name)
        return out

    def _allowed_names(self, allowed_exposure: str) -> set[str] | None:
        """ADR-0021 section 3.3b：exposure <= allowed_exposure 的模型名集合。

        public（关闭态/现状）-> None（不过滤）。其余 -> 集合（可能空 = 允许池空）。
        线程安全：纯函数式，不存实例状态（router 是共享实例，并发请求不能共享）。
        """
        if allowed_exposure == "public":
            return None
        from bladex_core.sensitivity import exposure_within
        return {
            name for name, c in self._models.items()
            if exposure_within(c.exposure, allowed_exposure)
        }

    def _judge_within(self, allowed_exposure: str) -> bool:
        """ADR-0021 section 3.2：敏感模式下裁判必须本地（绝不外发敏感 query）。

        public（无敏感）-> 始终允许。敏感（> normal）-> judge_model 必须 exposure=local；
        judge 未配/非本地 -> False（跳裁判，走允许池 default_tier）。
        """
        if allowed_exposure == "public":
            return True
        if not self._judge_model:
            return False
        cand = self._models.get(self._judge_model)
        if cand is None:
            return False
        return cand.exposure == "local"

    def _decision(
        self, pool: list[ModelCandidate], source: RouteSource, reason: str,
    ) -> RouteDecision:
        """从池取 primary（pool[0]），pool 全量保留供 failover。"""
        if not pool:
            raise NoCapableCandidateError(f"empty pool for source={source.value}")
        c = pool[0]
        return RouteDecision(
            model=c.model, api_base=c.api_base, api_key=c.api_key,
            tier=c.tier, provider=c.provider, source=source, reason=reason, pool=pool,
        )

    def _pool_reason(
        self,
        source: RouteSource,
        agent_id: str | None,
        requires: list[str] | None,
        context_chars: int | None = None,
        floor_tier: str | None = None,
    ) -> str:
        """构造可解释的 reason。"""
        parts: list[str] = []
        if source == RouteSource.AGENT and agent_id:
            parts.append(f"agent {agent_id} matched")
        elif source == RouteSource.FILTER:
            parts.append("filter judge selected tier")
        elif source == RouteSource.SCALE:
            parts.append(f"scale floor {floor_tier} (context_chars={context_chars})")
        elif source == RouteSource.UPSTREAM_DEFAULT:
            parts.append("upstream default pool")
        elif source == RouteSource.CAPABILITY:
            parts.append("capability fallback via multimodal.map")
        if requires:
            parts.append(f"requires={requires}")
        return "; ".join(parts) if parts else source.value

    # ── Phase 2 预留钩子 ────────────────────────────────────────────────────

    def _memory_override(
        self,
        facts: list[Fact] | None,
        hard_rules: list[HardRule] | None,
        pool: list[ModelCandidate],
    ) -> RouteDecision | None:
        """Phase 2 记忆驱动路由钩子（MUST/NEVER 模型指令 + 偏好覆盖）。

        本卡（Phase 1）不接逻辑，始终返回 None。
        Phase 2 将在此实现：MUST 用模型 M → 锁定；NEVER 用模型 N → 剔除；
        route_pref / "用强模型" → 升级到 pool 最强档。
        """
        return None
