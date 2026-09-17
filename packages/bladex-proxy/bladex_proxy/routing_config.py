"""结构化路由配置 schema v2（Phase 1 路由规格卡 T1）— TOML 配置文件为路由唯一入口。

四条策略的配置入口：统一模型列表（[[models]]）+ 默认上游（[upstream]）+ 三条可选策略
（[strategies.agent] / [strategies.multimodal] / [strategies.filter]）。
密钥只引环境变量名（api_key_env），不落配置文件（对齐 ADR-0009 密钥不入 git）。

所有策略默认关闭 → 全关 = 只走 [upstream].models 顺序 failover（回归安全）。

依赖方向：proxy 包，可依赖 bladex_core（产 ModelCandidate / 策略数据），不反向。
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

import structlog
from bladex_core.routing import (
    VALID_CAPABILITIES,
    VALID_MODALITIES,
    AgentStrategyData,
    FilterStrategyData,
    ModelCandidate,
    MultimodalStrategyData,
    PrincipalStrategyData,
    RequestStrategyData,
    ScaleStrategyData,
    StickyStrategyData,
    TeamStrategyData,
)
from bladex_core.sensitivity import EXPOSURE_PUBLIC, SensitivityConfig
from pydantic import BaseModel, Field

logger = structlog.get_logger()


class RoutingConfigError(Exception):
    """routing.toml 配置非法 —— **拒绝启动**（不降级、不静默透传）。

    刚性原则 9 第三条：配置错误必须报错，且报错要指向配置。
    把非法值透传给下游（Router / httpx），会把配置问题伪装成网络问题——
    2026-07-29 的 `api_base` 事故正是这么发生的，代价是 5 小时无人察觉的主力模型池全挂。
    """


class ModelSpec(BaseModel):
    """模型池里一个模型的声明（TOML [[models]]）。

    api_key_env 只存环境变量名；解析时读环境变量取密钥，缺失则标不可用（不崩）。
    不再有 min_score（难度打分退役）和 fallback（改由 multimodal.map 兜底）。
    """

    name: str
    # api_base   = **字面 URL**（必须 http:// 或 https:// 开头）
    # api_base_env = **环境变量名**（与 api_key_env 同规则），运行时解析
    # 二选一；同时给出时 api_base_env 优先。两者都空 = 用 Router 默认端点。
    api_base: str = ""
    api_base_env: str = ""
    api_key_env: str = ""
    capabilities: list[str] = Field(default_factory=lambda: ["text"])
    tier: str = "weak"
    # T4（RA4）：故障域分组。空 = 解析时从 api_base host 推导。
    # 429 只跨 provider failover（同配额池换模型名没有意义）。
    provider: str = ""
    # ADR-0021 section 3.3a: 暴露等级 local/private/public（对应自建/裸卡/公网）。
    # 缺省 public = 最坏暴露假设（敏感流量自动避开）。敏感路由硬边界用。
    exposure: str = EXPOSURE_PUBLIC
    # ADR-0028 §2：该模型的 Prompt Cache TTL（秒）。默认 300；`0` = 该模型无 cache，
    # 视为永远冷（每轮都允许重新规划裁剪）。
    cache_ttl_s: int = 300

    # 解析后填充（不来自 TOML）
    api_key: str = ""
    available: bool = True

    model_config = {"extra": "ignore"}


class DistillConfig(BaseModel):
    """蒸馏配置（TOML [distill]，ADR-0018 §3.6）。model 空 = 自举/env。"""

    model: str = ""
    model_config = {"extra": "ignore"}


class EmbeddingProfile(BaseModel):
    """单个 embedding 模型的阈值 profile（TOML [embedding.profiles."<model>"]）。

    覆盖/新增内置 profile（embedding.MODEL_PROFILES）；未声明字段用内置或全局默认。
    """

    dim: int | None = None
    novelty: float | None = None
    semantic: float | None = None
    digestion: float | None = None
    calibrated: bool = False

    model_config = {"extra": "ignore"}


class EmbeddingConfig(BaseModel):
    """Embedding 后端配置（TOML [embedding]，2026-07-26 可选化）。

    优先级：显式 env（BLADEX_EMBED_*）> 本段 > 内置默认（local + e5-large 零回归）。
    密钥同 ModelSpec 纪律：api_key_env 只存环境变量名，密钥不入 toml。
    backend 空 = 未配置（回落 env/默认）。
    """

    backend: str = ""              # local | api | proxy；空 = 未配置
    model: str = ""                # local: fastembed 模型名；api: Router 模型名
    api_base: str = ""             # 字面 URL（http/https）
    api_base_env: str = ""         # 或引用环境变量名；二选一，env 优先
    api_key_env: str = ""          # 密钥只引 env（如 "ARK_EMBED_KEY"）
    proxy_url: str = ""            # proxy 档：BladeX proxy 地址
    proxy_key_env: str = ""        # proxy 档客户端 key 的 env 名
    query_cache_size: int | None = None
    profiles: dict[str, EmbeddingProfile] = Field(default_factory=dict)

    model_config = {"extra": "ignore"}


class UpstreamConfig(BaseModel):
    """默认上游（TOML [upstream]）：有序模型名列表 = failover 链。"""

    models: list[str] = Field(default_factory=list)

    model_config = {"extra": "ignore"}


class AgentStrategyConfig(BaseModel):
    """agent 策略（TOML [strategies.agent]）。默认关。"""

    enabled: bool = False
    map: dict[str, list[str]] = Field(default_factory=dict)

    model_config = {"extra": "ignore"}


class PrincipalStrategyConfig(BaseModel):
    """principal 级路由策略（TOML [strategies.principal]，ADR-0021）。默认关。

    map 键 = principal_id（identity.toml 声明）-> 有序模型名列表。
    优先级 team > principal > agent。
    """

    enabled: bool = False
    map: dict[str, list[str]] = Field(default_factory=dict)

    model_config = {"extra": "ignore"}


class TeamStrategyConfig(BaseModel):
    """team 级路由策略（TOML [strategies.team]，ADR-0021）。默认关。优先级最高。

    map 键 = team_id（identity.toml 声明，含祖先链）-> 有序模型名列表。
    """

    enabled: bool = False
    map: dict[str, list[str]] = Field(default_factory=dict)

    model_config = {"extra": "ignore"}


class MultimodalStrategyConfig(BaseModel):
    """多模态策略（TOML [strategies.multimodal]）。默认关。"""

    enabled: bool = False
    map: dict[str, list[str]] = Field(default_factory=dict)

    model_config = {"extra": "ignore"}


class FilterStrategyConfig(BaseModel):
    """筛选器策略（TOML [strategies.filter]）。默认关。

    judge_model / no_think 等 LLM 裁判调用参数由 proxy 侧 route.py 读取后构造 Judge。
    """

    enabled: bool = False
    judge_model: str = ""
    no_think: bool = True  # ARK 方言（extra_body thinking disabled），非 ARK 裁判模型无效
    max_prompt_chars: int = 2000
    default_tier: str = "weak"
    judge_timeout_s: float = 5.0  # T6：裁判超时可配
    # 可选：不配则由 [[models]].tier 自动生成（T5a 单一 tier 真相）
    candidates: dict[str, list[str]] = Field(default_factory=dict)

    model_config = {"extra": "ignore"}


class ScaleStrategyConfig(BaseModel):
    """上下文规模策略（TOML [strategies.scale]，T1/RA1）。默认关。

    阈值为占位默认值，待真实流量校准（对照 Memory Hub 真实请求 chars 分布重标）。
    """

    enabled: bool = False
    floor_medium_chars: int = 60000
    floor_strong_chars: int = 400000
    judge_bypass_chars: int = 60000

    model_config = {"extra": "ignore"}


class StickyStrategyConfig(BaseModel):
    """会话粘性策略（TOML [strategies.sticky]，T2/RA2）。默认关。"""

    enabled: bool = False
    max_sessions: int = 500
    # G14/MQ-RT1：failover 被动粘上的模型持有上限（秒）。0 = 不主动释放。
    # 🔴 默认非 0 是有意的：G14 之前的行为（永不释放）本身就是缺陷形态——
    # 一次瞬时上游抖动会让整个会话永久漂移到降级模型上，且日志读起来完全正常。
    # 参照 ADR-0027 §5.4 的教训：默认关 + 不进模板 ≈ 机制等于没做。
    failover_max_hold_s: int = 300

    model_config = {"extra": "ignore"}


class RequestStrategyConfig(BaseModel):
    """req_model 显式指定策略（TOML [strategies.request]，T5b/RA7）。默认关。

    开启后果：agent 请求体里的 model 名若命中 [[models]] 名单（全名或展示名），
    将跳过选池与裁判直接使用该模型（仍过能力过滤）。agent 配置的静态模型名
    若恰好命中会静默旁路整个路由——务必理解后再开。
    """

    enabled: bool = False
    # per-agent 覆盖全局 enabled（ADR-0013 增补 2026-07-25）。agent_id -> bool，
    # 两级查找：精确 profile（"hermes:accept"）> base（"hermes"）。
    # 全局关时对指定 agent 放开，全局开时对指定 agent 关闭。空 = 纯全局开关（现状）。
    # TOML: [strategies.request.agent_overrides] 下 "agent-id" = true/false
    agent_overrides: dict[str, bool] = Field(default_factory=dict)

    model_config = {"extra": "ignore"}


class SensitivityStrategyConfig(BaseModel):
    """敏感度策略（TOML [strategies.sensitivity]，ADR-0021 section 3）。默认关。

    关闭态 = 个人模式：resolve 恒返 (normal, public)，注入/路由逐字不变。
    - agents: agent base/id -> 敏感等级（如财务 agent 整体敏感）。
    - default_level: 未标注主体的默认等级（敏感部署建议配最严）。
    - levels: 敏感等级 -> 允许暴露上限（如 sensitive=local / internal=private / normal=public）。
    """

    enabled: bool = False
    agents: dict[str, str] = Field(default_factory=dict)
    default_level: str = "normal"
    levels: dict[str, str] = Field(default_factory=lambda: {"normal": EXPOSURE_PUBLIC})

    model_config = {"extra": "ignore"}


class CapabilityConfig(BaseModel):
    """能力检测补充配置（TOML [capability]，T5c/RA8）。

    code_agents: 这些 base agent 的请求自动 requires "code"
    （能力护栏：将来池里出现无 code 模型时不会被选中）。
    """

    code_agents: list[str] = Field(
        default_factory=lambda: ["codex", "claude-code", "cursor"],
    )

    model_config = {"extra": "ignore"}


class StrategiesConfig(BaseModel):
    """策略容器（TOML [strategies]）。"""

    agent: AgentStrategyConfig = Field(default_factory=AgentStrategyConfig)
    principal: PrincipalStrategyConfig = Field(default_factory=PrincipalStrategyConfig)
    team: TeamStrategyConfig = Field(default_factory=TeamStrategyConfig)
    multimodal: MultimodalStrategyConfig = Field(default_factory=MultimodalStrategyConfig)
    filter: FilterStrategyConfig = Field(default_factory=FilterStrategyConfig)
    scale: ScaleStrategyConfig = Field(default_factory=ScaleStrategyConfig)
    sticky: StickyStrategyConfig = Field(default_factory=StickyStrategyConfig)
    request: RequestStrategyConfig = Field(default_factory=RequestStrategyConfig)
    sensitivity: SensitivityStrategyConfig = Field(default_factory=SensitivityStrategyConfig)

    model_config = {"extra": "ignore"}


class RoutingConfig(BaseModel):
    """路由配置根（TOML 顶层）。

    加载 routing.toml → 结构化 RoutingConfig；密钥经 _resolve_api_keys 从环境变量注入。
    """

    models: list[ModelSpec] = Field(default_factory=list)
    upstream: UpstreamConfig = Field(default_factory=UpstreamConfig)
    strategies: StrategiesConfig = Field(default_factory=StrategiesConfig)
    # T5c（RA8）：能力检测补充（code_agents）
    capability: CapabilityConfig = Field(default_factory=CapabilityConfig)
    # ADR-0018 §3.6: 蒸馏模型配置（空=自举/env）
    distill: DistillConfig = Field(default_factory=DistillConfig)
    # embedding 后端可选化（2026-07-26）：结构化配置（env 为覆盖通道）
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)

    model_config = {"extra": "ignore"}

    @classmethod
    def from_toml(cls, path: str | Path | None) -> RoutingConfig:
        """从 TOML 文件加载；文件不存在 → 空配置（调用方回落旧 env 路径，回归安全）。"""
        if path is None:
            return cls()
        p = Path(path)
        if not p.is_file():
            logger.info("routing_config_not_found", path=str(path), hint="routing disabled or env fallback")
            return cls()
        with p.open("rb") as f:
            data = tomllib.load(f)
        cfg = cls.model_validate(data)
        cfg._resolve_api_bases()      # 先解析 api_base_env，再校验格式（拒启动）
        cfg._resolve_api_keys()
        cfg._derive_providers()
        cfg._validate_capabilities()
        cfg._validate_model_refs()
        cfg._validate_tier_consistency()
        return cfg

    def _derive_providers(self) -> None:
        """T4（RA4）：provider 未显式配置时从 api_base host 推导。"""
        from urllib.parse import urlparse

        for m in self.models:
            if m.provider:
                continue
            host = ""
            if m.api_base:
                try:
                    host = urlparse(m.api_base).hostname or ""
                except ValueError:
                    host = ""
            m.provider = host or "default"

    def _validate_tier_consistency(self) -> None:
        """T5a（RA6）：显式 filter.candidates 与 [[models]].tier 不一致 → warning 可 grep。

        不纠正（尊重显式配置），但单一 tier 真相以 [[models]] 为准，
        建议删除 candidates 段改用自动生成。
        """
        tier_by_name = {m.name: m.tier for m in self.models}
        for pool_tier, names in self.strategies.filter.candidates.items():
            for n in names:
                declared = tier_by_name.get(n)
                if declared is not None and declared != pool_tier:
                    logger.warning(
                        "routing_tier_mismatch",
                        model=n, declared_tier=declared, candidates_tier=pool_tier,
                        hint="single source of truth is [[models]].tier; "
                             "consider removing [strategies.filter.candidates]",
                    )

    def _resolve_api_bases(self) -> None:
        """解析 api_base_env → api_base，并校验 api_base 必须是合法 URL。

        **格式不对直接拒绝启动**（2026-07-29 事故驱动）。

        事故经过：用户为了"改一处即可切换上游"，把 `routing.toml` 里 8 个模型的
        `api_base` 从字面 URL 改成了 `"BLADEX_UPSTREAM_API_BASE"`——以为会像
        `api_key_env` 那样被解析。但当时 `api_base` 只做字面值透传，于是 Router
        拿到字符串 `"BLADEX_UPSTREAM_API_BASE"` 去建连，httpx 抛
        `Connection error.`——一句完全不指向配置的错误。

        结果：8 个 openai/* 模型**全部不可用**，路由每轮挨个 failover，最后总是
        退到唯一没被改到的 `anthropic/doubao-seed-2.0-lite`（它的 api_base 仍是
        硬编码 URL）。表面看"系统能用"，实际主力模型池全挂，持续 5 个多小时无人察觉。

        教训（对应 CLAUDE.md 刚性原则 9 第三条）：**配置错误必须报错，且报错要指向配置。**
        静默透传一个非 URL 的值，把配置问题伪装成网络问题，是最坏的处理方式。
        """
        from urllib.parse import urlparse

        for m in self.models:
            m.api_base = self._resolve_one_api_base(
                what=f'[[models]] name="{m.name}"',
                literal=m.api_base, env_name=m.api_base_env, urlparse=urlparse,
            )
        if self.embedding is not None:
            self.embedding.api_base = self._resolve_one_api_base(
                what="[embedding]",
                literal=self.embedding.api_base,
                env_name=self.embedding.api_base_env, urlparse=urlparse,
            )

    @staticmethod
    def _resolve_one_api_base(*, what: str, literal: str, env_name: str, urlparse) -> str:
        """返回最终 api_base；非法即抛 RoutingConfigError（拒启动）。"""
        if env_name:
            val = os.environ.get(env_name)
            if not val:
                raise RoutingConfigError(
                    f"{what}: api_base_env 指向的环境变量 {env_name!r} 未设置或为空。\n"
                    f"  请在 config/.env 里 export {env_name}=\"https://...\"，"
                    f"或改用字面值 api_base = \"https://...\"。"
                )
            literal = val
        if not literal:
            return ""                     # 两者都空 = 用 Router 默认端点，合法
        scheme = urlparse(literal).scheme
        if scheme in ("http", "https"):
            return literal

        # ── 非法：给出精确诊断，不做"你是不是想…"式的猜测 ──
        looks_like_env = literal.replace("_", "").isalnum() and literal.isupper()
        detail = (
            f"  实际值：api_base = {literal!r}\n"
            f"  问题：缺少协议头，URL 必须以 http:// 或 https:// 开头"
            f"（解析得到 scheme={scheme!r}）。\n"
        )
        if looks_like_env:
            detail += (
                f"  说明：该值形如环境变量名。`api_base` 字段**只接受字面 URL，不做环境变量解析**——\n"
                f"        这一点与 `api_key_env` 不同（后者存的是变量名）。\n"
            )
        raise RoutingConfigError(
            f"{what}: api_base 不是合法 URL，拒绝启动。\n"
            + detail
            + "  两种正确写法（二选一）：\n"
            f'    ① 字面 URL：    api_base = "https://api.example.com/v1"\n'
            f'    ② 引用环境变量：api_base_env = "BLADEX_UPSTREAM_API_BASE"\n'
            f"       （同时配置时 api_base_env 优先）"
        )

    def _resolve_api_keys(self) -> None:
        """把每个模型的 api_key_env 解析成实际密钥；引用的 env 缺失 → 标不可用 + warning。"""
        for m in self.models:
            if not m.api_key_env:
                m.api_key = ""
                m.available = True
                continue
            val = os.environ.get(m.api_key_env)
            if val:
                m.api_key = val
                m.available = True
            else:
                m.available = False
                logger.warning(
                    "routing_model_api_key_missing",
                    env=m.api_key_env, model=m.name, hint="model marked unavailable",
                )

    def _validate_capabilities(self) -> None:
        """能力词表校验（T4）：非法值 → warning 并忽略该项。"""
        for m in self.models:
            valid = [c for c in m.capabilities if c in VALID_CAPABILITIES]
            invalid = [c for c in m.capabilities if c not in VALID_CAPABILITIES]
            if invalid:
                logger.warning(
                    "routing_invalid_capability",
                    model=m.name, invalid=invalid, valid=valid,
                    hint=f"valid capabilities: {sorted(VALID_CAPABILITIES)}",
                )
            if not valid:
                valid = ["text"]
            m.capabilities = valid

        mm = self.strategies.multimodal
        invalid_keys = [k for k in mm.map if k not in VALID_MODALITIES]
        if invalid_keys:
            logger.warning(
                "routing_invalid_multimodal_key",
                invalid=invalid_keys, valid=sorted(VALID_MODALITIES),
                hint="invalid multimodal.map keys ignored",
            )
            mm.map = {k: v for k, v in mm.map.items() if k in VALID_MODALITIES}

    def _validate_model_refs(self) -> None:
        """策略里引用的模型名必须在 [[models]] 里存在；**缺失即拒绝启动**。

        🔴 **MQ-L58（2026-09-08，批 G 剧本③ 首次实跑当场撞到）：从 warning 改成拒启动。**

        事故经过：`[strategies.agent.map]` 写着 `"codex" = ["anthropic/glm-5.3-flash"]`，
        而 `[[models]]` 表里没有这个名字。启动时**这里确实打了**
        `routing_agent_unknown_model agent=codex missing=[...] hint='dropped from agent policy'`
        —— 结构化、带 agent、带 hint，一条不缺。**然后请求照常成功**：静态策略整个不命中，
        路由退回 sticky ⇒ `doubao-seed-2.0-lite`（weak）⇒ 账本族被 gate ⇒ Codex 全程
        拿不到账本块。剧本③ 的前提从头不成立，而现象是"跑起来了、只是效果不对"。

        **这条悬空引用在 10 个以上的历史日志里连着打了好几天，没人看。**

        ⇒ 缺的从来不是告警，是**严重性**。原则 12 的家族形态：**把缺陷写进日志 ≠ 处理了缺陷**；
        而那句 `hint='dropped from agent policy'` 更是把静默旁路描述成了正常行为
        （与"警告性 docstring 是缺陷的气味"同一条判据）。

        **为什么是拒启动而不是运行时报错**：这是**自包含**的配置错误，加载期就能判定
        （刚性原则：误配置失败要响亮，自包含者在加载期失败）。同仓已有先例——
        `IdentityRegistry.from_toml` 对 dangling ref 就是 `ValueError` 拒启动
        （见 `config.py::identity_registry` 的注释）。两个配置文件同类问题两种处置，
        本身就是"同一条判据只用在一半上"。

        **一次列全**：不是撞到第一条就抛——用户要能一遍改完。
        """
        known = {m.name for m in self.models}
        dangling: list[str] = []

        def _check(where: str, names) -> None:
            missing = [n for n in names if n not in known]
            if missing:
                dangling.append(f"  {where}: {', '.join(repr(n) for n in missing)}")

        _check("[upstream] models", self.upstream.models)
        # 🔴 三层静态策略一个都不许漏（team > principal > agent，ADR-0021）——
        # 此前只查了 agent 一层，而三层是同一类判据。同一条判据只用在一部分上，
        # 正是 MQ-L53 / MQ-A52 反复付过学费的形态。
        for where, mapping in (
            ("strategies.team.map", self.strategies.team.map),
            ("strategies.principal.map", self.strategies.principal.map),
            ("strategies.agent.map", self.strategies.agent.map),
        ):
            for key, names in mapping.items():
                _check(f'[{where}] "{key}"', names)
        for tier, names in self.strategies.filter.candidates.items():
            _check(f'[strategies.filter.candidates] "{tier}"', names)
        for modality, names in self.strategies.multimodal.map.items():
            _check(f'[strategies.multimodal.map] "{modality}"', names)

        if not dangling:
            return
        # 🔴 报错文本是**产品面**：英文、自足、不引用仓库内部条款编号
        # （2026-09-08 Jason 定的标准）。内部立论写在 docstring 里，不写进用户看到的字符串。
        raise RoutingConfigError(
            "routing.toml: strategy refers to model names that are not defined "
            "in any [[models]] block.\n\n  Unresolved references:\n"
            + "\n".join(dangling)
            + "\n\n  Defined model names:\n    "
            + "\n    ".join(sorted(known) or ["(none)"])
            + "\n\n  Startup is refused instead of skipping the unknown names: skipping "
            "makes the whole\n  static strategy miss, routing silently falls back to its "
            "dynamic layer, and requests\n  keep succeeding on a different model than the "
            "one configured.\n  Fix the name in routing.toml, or add the matching "
            "[[models]] block."
        )

    # ── 产 core 侧路由输入 ──────────────────────────────────────────────────

    def model_candidates(self) -> dict[str, ModelCandidate]:
        """产 core 的 ModelCandidate 字典（name → candidate），仅 available 模型。"""
        return {
            m.name: ModelCandidate(
                model=m.name,
                api_base=m.api_base,
                api_key=m.api_key,
                tier=m.tier,
                capabilities=list(m.capabilities),
                provider=m.provider,
                exposure=m.exposure,
                cache_ttl_s=m.cache_ttl_s,
            )
            for m in self.models
            if m.available
        }

    def agent_strategy(self) -> AgentStrategyData:
        """产 core 的 AgentStrategyData。"""
        return AgentStrategyData(
            enabled=self.strategies.agent.enabled,
            map=dict(self.strategies.agent.map),
        )

    def principal_strategy(self) -> PrincipalStrategyData:
        """产 core 的 PrincipalStrategyData（ADR-0021 team>principal>agent）。"""
        return PrincipalStrategyData(
            enabled=self.strategies.principal.enabled,
            map=dict(self.strategies.principal.map),
        )

    def team_strategy(self) -> TeamStrategyData:
        """产 core 的 TeamStrategyData（ADR-0021 优先级最高）。"""
        return TeamStrategyData(
            enabled=self.strategies.team.enabled,
            map=dict(self.strategies.team.map),
        )

    def multimodal_strategy(self) -> MultimodalStrategyData:
        """产 core 的 MultimodalStrategyData。"""
        return MultimodalStrategyData(
            enabled=self.strategies.multimodal.enabled,
            map=dict(self.strategies.multimodal.map),
        )

    def filter_strategy(self) -> FilterStrategyData:
        """产 core 的 FilterStrategyData（不含 judge，judge 由 route.py 构造）。

        T5a（RA6）：candidates 未显式配置时由 [[models]].tier 自动生成
        （同 tier 内保 [[models]] 声明顺序）——单一 tier 真相。
        """
        candidates = dict(self.strategies.filter.candidates)
        if not candidates:
            candidates = self._auto_candidates()
        return FilterStrategyData(
            enabled=self.strategies.filter.enabled,
            candidates=candidates,
            default_tier=self.strategies.filter.default_tier,
            max_prompt_chars=self.strategies.filter.max_prompt_chars,
            judge_timeout_s=self.strategies.filter.judge_timeout_s,
        )

    def _auto_candidates(self) -> dict[str, list[str]]:
        """由 [[models]].tier 自动生成三档候选池（仅 available，保声明顺序）。"""
        out: dict[str, list[str]] = {"weak": [], "medium": [], "strong": []}
        for m in self.models:
            if m.available and m.tier in out:
                out[m.tier].append(m.name)
        return {tier: names for tier, names in out.items() if names}

    def scale_strategy(self) -> ScaleStrategyData:
        """产 core 的 ScaleStrategyData（T1）。"""
        s = self.strategies.scale
        return ScaleStrategyData(
            enabled=s.enabled,
            floor_medium_chars=s.floor_medium_chars,
            floor_strong_chars=s.floor_strong_chars,
            judge_bypass_chars=s.judge_bypass_chars,
        )

    def sticky_strategy(self) -> StickyStrategyData:
        """产 core 的 StickyStrategyData（T2 + G14 释放参数）。"""
        s = self.strategies.sticky
        return StickyStrategyData(
            enabled=s.enabled,
            max_sessions=s.max_sessions,
            failover_max_hold_s=s.failover_max_hold_s,
        )

    def request_strategy(self) -> RequestStrategyData:
        """产 core 的 RequestStrategyData（T5b）。"""
        return RequestStrategyData(
            enabled=self.strategies.request.enabled,
            agent_overrides=dict(self.strategies.request.agent_overrides),
        )

    def sensitivity_config(self) -> SensitivityConfig:
        """产 core 的 SensitivityConfig（ADR-0021 section 3.1/3.3）。

        默认关 -> resolve 恒返 (normal, public) = 现状零回归。
        """
        s = self.strategies.sensitivity
        return SensitivityConfig(
            enabled=s.enabled,
            agents=dict(s.agents),
            default_level=s.default_level,
            levels=dict(s.levels),
        )

    @property
    def filter_judge_model(self) -> str:
        return self.strategies.filter.judge_model

    @property
    def filter_no_think(self) -> bool:
        return self.strategies.filter.no_think

    def has_models(self) -> bool:
        return any(m.available for m in self.models)

    def has_upstream(self) -> bool:
        """是否配了 [upstream].models（全关时的基础 failover 链）。"""
        return bool(self.upstream.models)


# ── ADR-0018 §3.6: 蒸馏模型自举 ──


def bootstrap_distill_model(
    cfg_distill_model: str,
    routing_config: RoutingConfig,
    config_path: str | Path | None,
) -> str:
    """蒸馏模型自举（ADR-0018 §3.6）。

    优先级：env distill_model（非空）> routing [distill].model（非空）> 从模型池按
    tier weak->medium->strong 取首个 available 写回 routing.toml [distill].model。
    池空 -> 返回空（蒸馏停摆，consolidation 走 PassthroughDistiller 降级）。

    首次启用时从用户已配的 proxy 模型池选最弱档（少一个行为面），写回配置文件
    （含"自动选择，可手动修改"语义），之后以配置文件为准。
    """
    if cfg_distill_model:
        return cfg_distill_model
    if routing_config.distill.model:
        return routing_config.distill.model

    candidates = routing_config.model_candidates()  # 仅 available
    if not candidates:
        logger.warning("distill_bootstrap_no_model", hint="pool empty, distill disabled")
        return ""

    chosen = ""
    for tier in ("weak", "medium", "strong"):
        for cand in candidates.values():
            if cand.tier == tier:
                chosen = cand.model
                break
        if chosen:
            break
    if not chosen:
        chosen = next(iter(candidates.values())).model

    if config_path is not None:
        _write_distill_to_toml(config_path, chosen)
    logger.info("distill_bootstrap_selected", model=chosen,
                hint="auto-selected from pool, manually editable in routing.toml [distill]")
    return chosen


def _write_distill_to_toml(path: str | Path, model: str) -> None:
    """把 [distill].model 写回 routing.toml（自举结果，可手动改）。

    简单文本操作（低频：仅首次自举）：有 [distill] 段则替换/补 model 行；无则追加。
    """
    import re
    p = Path(path)
    if not p.is_file():
        return
    text = p.read_text(encoding="utf-8")
    if "[distill]" in text:
        pattern = r'(\[distill\][^\[]*?model\s*=\s*)"[^"]*"'
        if re.search(pattern, text, re.DOTALL):
            text = re.sub(pattern, rf'\1"{model}"', text, count=1, flags=re.DOTALL)
        else:
            text = text.replace("[distill]", f'[distill]\nmodel = "{model}"', 1)
    else:
        sep = "\n\n" if text and not text.endswith("\n") else ("\n" if text else "")
        text += f'{sep}# ══ 蒸馏模型（ADR-0018 §3.6 自动选择，可手动修改）══════════\n[distill]\nmodel = "{model}"\n'
    p.write_text(text, encoding="utf-8")
