"""Proxy 配置 — Redis 地址、RocksDB 路径、硬规则、延迟预算、上游模型路由、客户端 key 验证。

所有配置可通过环境变量或 config/.env 文件覆盖，默认值面向本地开发。
启动脚本 run_proxy.sh 会自动 source config/.env。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bladex_core.flags import flag_number as _flag_number

from bladex_proxy.auth import KeyStore
from bladex_proxy.routing_config import RoutingConfig


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_list(key: str, default: str) -> list[str]:
    """环境变量按 || 分隔读成 list。"""
    raw = _env(key, default)
    return [item for item in raw.split("||") if item]


def _flag_number_default(name: str) -> float:
    """数值型开关的默认值（集中在 `bladex_core.flags`）。

    env 覆盖由 `flag_number` 自己处理，所以这里不再手写 `_env(name, "0.95")`
    ——写死一份默认值就等于让 flags 表与实际行为各说各话。
    """
    return _flag_number(name)


# ── 存储目录：**单一真相源**（2026-08-07）────────────────────────────────
#
# 改名前这两个默认值被 20 个脚本各自硬编码了一份（`os.path.join(_ROOT, "data",
# "bladex_index")`）。这种复制本身就是缺陷：
#
#   `MemoryIndex` 对**不存在的路径会开出一个空库** —— 于是任何一处没跟上改名的
#   脚本都不会报错，而是给出「0 条、一切干净」的报告。2026-08-07 一天内就栽过
#   两次同型（`audit_matter_generic` 空目录报告、只查 ≤500 长度带的分流探针）。
#
# 与 ADR-0027 §5.4 把 flags 默认值收进 `bladex_core.flags` 是同一条纪律：
# **默认值只许有一处**，别处一律 import。
#
# 命名：目录名跟层名走（Memory Hub / Memory Index），不写实现选型。
# `bladex_rocksdb` 那种名字在换存储引擎的那天就名不副实了。
# 2026-09-05 目录名 `bladex_ledger` → `bladex_hub`（与 P3 层名 Memory Hub 对齐，
# ADR-0032 §2.2；旧值与账本 LEDGER.md 撞名）。物理目录同日 mv，不留软链。
DEFAULT_HUB_DIR = "data/bladex_hub"
DEFAULT_INDEX_DIR = "data/bladex_index"
# P1 Memory Flash（ADR-0031 §5.2）：MD 文件形态的预装配注入包。
# 它是**投影不是存储**——删掉整个目录重渲染即可，不丢数据（这也是它可以直接
# 放在 data/ 下、不进备份关键路径的原因）。
DEFAULT_FLASH_DIR = "data/bladex_flash"


def resolve_hub_path(root: str = "") -> str:
    """Memory Hub 目录：`BLADEX_ROCKSDB_PATH` 优先，否则 `<root>/data/bladex_hub`。

    `root` 留空 = 相对 cwd（服务进程的形态）；脚本传 `_ROOT` 拿绝对路径，
    这样在任意 cwd 下跑 `scripts/*.py` 都指向同一个库。
    """
    return _resolve_storage_path("BLADEX_ROCKSDB_PATH", DEFAULT_HUB_DIR, root)


def resolve_index_path(root: str = "") -> str:
    """Memory Index 目录：`BLADEX_INDEX_PATH` 优先，否则 `<root>/data/bladex_index`。"""
    return _resolve_storage_path("BLADEX_INDEX_PATH", DEFAULT_INDEX_DIR, root)


def resolve_flash_path(root: str = "") -> str:
    """Memory Flash 目录：`BLADEX_FLASH_PATH` 优先，否则 `<root>/data/bladex_flash`。"""
    return _resolve_storage_path("BLADEX_FLASH_PATH", DEFAULT_FLASH_DIR, root)


def _resolve_storage_path(env_key: str, default_rel: str, root: str) -> str:
    """env 覆盖 > 默认；`root` 非空时把**相对**路径锚到 root（绝对路径原样保留）。"""
    raw = os.environ.get(env_key) or default_rel
    if root and not os.path.isabs(raw):
        return os.path.join(root, raw)
    return raw


@dataclass
class ModelRoute:
    """单个上游模型路由配置。

    Router 用 model 名的 provider 前缀判断走哪个适配器：
      openai/xxx  → OpenAI 兼容接口（含火山方舟、Deepseek、Together 等）
      anthropic/xxx  → Anthropic 原生
      不带前缀 → Router 自动推断
    """

    model: str           # Router 格式（provider/model），如 "openai/ark-code-latest"
    api_base: str = ""   # 自定义 endpoint
    api_key: str = ""    # 上游 API key
    provider: str = ""   # 故障域分组（T4/RA4：429 只跨 provider failover）
    # T4(ADR-0018 §4.4 C4): 路由决策元数据（resolve_route 填，供 Turn.decision_meta 透传）
    source: str = ""     # RouteSource.value（如 agent/filter/capability/upstream_default）
    tier: str = ""       # weak/medium/strong
    reason: str = ""     # 决策理由（可解释性）


@dataclass
class ProxyConfig:
    """M-proxy-1 运行配置。"""

    # ── 服务 ──
    host: str = field(default_factory=lambda: _env("BLADEX_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(_env("BLADEX_PORT", "38080")))

    # ── 客户端 key 验证 ──
    auth_enabled: bool = field(default_factory=lambda: _env("BLADEX_AUTH_ENABLED", "false").lower() in ("true", "1", "yes"))
    client_keys_raw: str = field(default_factory=lambda: _env("BLADEX_CLIENT_KEYS", ""))
    # ADR-0027 §2.2：管理面独立 key（与数据面分级）。格式同 BLADEX_CLIENT_KEYS（`key||label`）。
    # 空 = 回落共用 client key（个人模式零回归 + 启动 warning）；有 identity.toml 时空值拒启动。
    admin_keys_raw: str = field(default_factory=lambda: _env("BLADEX_ADMIN_KEYS", ""))

    # ── Pipeline Redis ──
    redis_url: str = field(default_factory=lambda: _env("BLADEX_REDIS_URL", "redis://127.0.0.1:6379/0"))
    redis_stream: str = field(default_factory=lambda: _env("BLADEX_REDIS_STREAM", "bladex:turns"))
    redis_group: str = field(default_factory=lambda: _env("BLADEX_REDIS_GROUP", "bladex-workers"))
    redis_consumer: str = field(default_factory=lambda: _env("BLADEX_REDIS_CONSUMER", "worker-1"))
    queue_max_len: int = field(default_factory=lambda: int(_env("BLADEX_QUEUE_MAX_LEN", "10000")))

    # ── Pipeline 背压（ADR-0009 §5b，阈值待真实流量校准）──
    overflow_dir: str = field(default_factory=lambda: _env("BLADEX_OVERFLOW_DIR", "data/overflow"))
    pipeline_warn_pct: float = field(default_factory=lambda: float(_env("BLADEX_PIPELINE_WARN_PCT", "0.6")))
    pipeline_alert_pct: float = field(default_factory=lambda: float(_env("BLADEX_PIPELINE_ALERT_PCT", "0.8")))
    pipeline_critical_pct: float = field(default_factory=lambda: float(_env("BLADEX_PIPELINE_CRITICAL_PCT", "0.9")))
    pipeline_spill_timeout_ms: int = field(default_factory=lambda: int(_env("BLADEX_PIPELINE_SPILL_TIMEOUT_MS", "1000")))
    pipeline_heavy_payload_bytes: int = field(default_factory=lambda: int(_env("BLADEX_PIPELINE_HEAVY_PAYLOAD_BYTES", "262144")))

    # ── Memory Hub / Memory Index 目录 ──
    # 🔴 默认值在模块顶部的 `DEFAULT_HUB_DIR` / `DEFAULT_INDEX_DIR`，**只此一处**。
    #
    # 2026-08-07 改名 `bladex_rocksdb` → `bladex_ledger`、`bladex_p2` → `bladex_index`；
    # 2026-09-05 再改 `bladex_ledger` → `bladex_hub`（跟层名 Memory Hub 走）。
    # 原注释写着"保留历史拼写：改了 = 老用户记忆库消失" —— 那条顾虑成立，但**前提是
    # 已经有装机用户**。beta（v0.1.0）尚未发布，趁 M5 全量重建（Index 本就要清空、
    # 三个进程全停）一次改掉；发布之后这个窗口就永久关闭了。
    rocksdb_path: str = field(default_factory=resolve_hub_path)
    index_path: str = field(default_factory=resolve_index_path)

    # ── fastembed 模型缓存路径 ──
    fastembed_cache_path: str = field(default_factory=lambda: _env("BLADEX_FASTEMBED_CACHE_PATH", "data/fastembed_cache"))

    # ── Embedding 后端可选化（2026-07-26，见 embedding 后端评估记录）──
    # local（默认，零回归）| api（经 Router 调线上 embedding，隐私警告 + 敏感层互斥）
    # | proxy（仅 consolidator：复用 proxy /v1/embeddings 共享模型，单机内存减半）。
    # 切换后端/模型 = 换向量空间，需先跑 scripts/reembed_index.py（Memory Index model_id 不变量拒混写）。
    embed_backend: str = field(default_factory=lambda: _env("BLADEX_EMBED_BACKEND", "local"))
    # local 档：fastembed 模型名，空 = e5-large；小模型档填 intfloat/multilingual-e5-base|small。
    # api 档：Router 格式模型名（如 openai/text-embedding-3-small、doubao-embedding），必填。
    embed_model: str = field(default_factory=lambda: _env("BLADEX_EMBED_MODEL", ""))
    embed_api_base: str = field(default_factory=lambda: _env("BLADEX_EMBED_API_BASE", ""))
    embed_api_key: str = field(default_factory=lambda: _env("BLADEX_EMBED_API_KEY", ""))
    # proxy 档：BladeX proxy 地址 + 客户端 key（proxy 开 auth 时需要）。
    embed_proxy_url: str = field(default_factory=lambda: _env("BLADEX_EMBED_PROXY_URL", "http://127.0.0.1:38080"))
    embed_proxy_key: str = field(default_factory=lambda: _env("BLADEX_EMBED_PROXY_KEY", ""))
    # query 嵌入 LRU 容量（api/proxy 档网络往返成本高，工具循环同 query 重复率高）。
    embed_query_cache_size: int = field(default_factory=lambda: int(_env("BLADEX_EMBED_QUERY_CACHE", "256")))

    # ── Consolidation 阈值（ADR-0014 L0）──
    # 不变式：novelty_threshold > semantic_threshold（去重线必须高于聚类线，
    # 否则 Matter 结构性锁死成单例）。随 embedding 模型/pooling 变动需重标。
    # M2-2：默认值收编 flags.MEMORY_NUMERIC_DEFAULTS（0.95 → 0.98，语义降为
    # "只拦完全重发"）。写死默认值会让 flags 表与实际行为脱节——
    # 那正是本仓库反复发作的那类腐化。
    novelty_threshold: float = field(
        default_factory=lambda: _flag_number_default("BLADEX_NOVELTY_THRESHOLD"))
    semantic_threshold: float = field(default_factory=lambda: float(_env("BLADEX_SEMANTIC_THRESHOLD", "0.85")))
    digestion_threshold: float = field(default_factory=lambda: float(_env("BLADEX_DIGESTION_THRESHOLD", "0.85")))

    # ── 实体感知 novelty（ADR-0014 L0 增强方案 B）──
    # 纯 cosine novelty 对"同义不同实体"误判（e5: cos('北京天气','上海天气')=0.9652>0.95
    # 导致不同实体的同类查询被判重复丢弃）。实体感知：cosine 召回 top-k 后比实体集合，
    # 实体有差异则仍判新颖。feature flag 关闭退回纯 cosine（回滚通道）。
    entity_aware_novelty: bool = field(default_factory=lambda: _env("BLADEX_ENTITY_AWARE_NOVELTY", "true").lower() in ("1", "true", "yes", "on"))
    entity_overlap_threshold: float = field(default_factory=lambda: float(_env("BLADEX_ENTITY_OVERLAP_THRESHOLD", "0.6")))
    novelty_topk: int = field(default_factory=lambda: int(_env("BLADEX_NOVELTY_TOPK", "5")))

    # ── 实体感知检索重排序（ADR-0024 T8a）──
    # cosine 召回扩大 k 后，按查询与候选 fact 实体重叠度重排序。
    # 重叠度越高，排名越靠前——帮助召回"同一实体但语义表述不同"的记忆。
    # 纯 cosine 往往把同义不同实体排在前面，实体信号修正这个偏差。
    # 不需要 LLM，确定性规则从 query 提取实体候选，不增加热路径延迟。
    entity_aware_rerank: bool = field(default_factory=lambda: _env("BLADEX_ENTITY_AWARE_RERANK", "true").lower() in ("1", "true", "yes", "on"))
    entity_rerank_alpha: float = field(default_factory=lambda: float(_env("BLADEX_ENTITY_RERANK_ALPHA", "0.7")))
    entity_rerank_expand_k: int = field(default_factory=lambda: int(_env("BLADEX_ENTITY_RERANK_EXPAND_K", "15")))

    # ── L2 蒸馏（ADR-0014 L2）──
    # consolidation 用便宜档 LLM 从 user 消息抽取原子事实。
    # 模型名记录到 Fact.distill_model 供审计/对比。空 = 交由 bootstrap_distill_model
    # 自举（优先级：env > routing [distill].model > 池最弱档写回 toml）；自举仍空
    # （池空）= 不蒸馏（透传原文）。N3 修复：默认空串让自举成为默认路径（原默认
    # "openai/deepseek-v4-flash" 非空 -> env 永远非空 -> 自举死代码，偏离拍板①）。
    distill_model: str = field(default_factory=lambda: _env("BLADEX_DISTILL_MODEL", ""))

    # ── 热路径延迟预算（ms）──
    # ADR-0027 §5.4：默认 100 → 200。软打分 + 有限跳默认开后热路径贴预算，08-04 D 步
    # 复核实测 inject_timeout 18% 且超时值全在 151–153ms——只超 1–3ms 就静默降级只注
    # 硬规则。50ms 尾延迟的代价远小于 18% 注入失效。超时仍有 inject_timeout 告警面。
    hotpath_budget_ms: int = field(default_factory=lambda: int(_env("BLADEX_HOTPATH_BUDGET_MS", "200")))

    # ── 硬规则（每轮无条件注入）──
    hard_rules: list[str] = field(default_factory=lambda: _env_list("BLADEX_HARD_RULES", ""))

    # ── 注入 ──
    # 2026-08-14 随 MAX_PREFETCH_K 3→15 连带 1800→4000（与 core 默认一致；基准标定见
    # naive-baseline-ablation 卡——只提 TOPK 不提预算会触发③平面整段砍）。
    inject_max_chars: int = field(default_factory=lambda: int(_env("BLADEX_INJECT_MAX_CHARS", "4000")))
    inject_merge_system: bool = field(default_factory=lambda: _env("BLADEX_INJECT_MERGE_SYSTEM", "false").lower() in ("true", "1", "yes"))

    # ── CAP 上下文感知代理（ADR-0016）──
    # 超阈值时主动压缩旧历史为 Memory Index 蒸馏摘要，保近期对话原文。
    # 关闭时回退纯叠加模式（向后兼容）。
    cap_enabled: bool = field(default_factory=lambda: _env("BLADEX_CAP_ENABLED", "true").lower() in ("true", "1", "yes"))
    cap_msg_threshold: int = field(default_factory=lambda: int(_env("BLADEX_CAP_MSG_THRESHOLD", "30")))
    cap_preserve_turns: int = field(default_factory=lambda: int(_env("BLADEX_CAP_PRESERVE_TURNS", "6")))
    cap_summary_max_chars: int = field(default_factory=lambda: int(_env("BLADEX_CAP_SUMMARY_MAX_CHARS", "1000")))

    # ── 上游模型路由 ──
    upstream_model: str = field(default_factory=lambda: _env("BLADEX_UPSTREAM_MODEL", "openai/gpt-4o-mini"))
    upstream_api_base: str = field(default_factory=lambda: _env("BLADEX_UPSTREAM_API_BASE", ""))
    upstream_api_key: str = field(default_factory=lambda: _env("BLADEX_UPSTREAM_API_KEY", ""))

    # ── 模型路由（Phase 1 路由规格卡：配置驱动确定性路由，默认关）──
    # 路由总开关；关闭 = 现状单一上游直转（回归）。
    route_enabled: bool = field(default_factory=lambda: _env("BLADEX_ROUTE_ENABLED", "false").lower() in ("true", "1", "yes"))
    # 结构化路由配置文件路径（routing.toml schema v2）。
    routing_config_path: str = field(default_factory=lambda: _env("BLADEX_ROUTING_CONFIG", "config/routing.toml"))
    # ADR-0021 section 2: 身份两体系配置文件（identity.toml，不入 git）。
    # 不存在 = 个人模式：单一本地身份 user_id="local"（ADR-0021 §2.3，2026-08-16 修订）。
    identity_config_path: str = field(default_factory=lambda: _env("BLADEX_IDENTITY_CONFIG", "config/identity.toml"))
    # T9: auxiliary 内部调用强制走的便宜档（默认 weak）。
    route_aux_tier: str = field(default_factory=lambda: _env("BLADEX_AUX_TIER", "weak"))
    # T8: 路由开着但不可用时是否拒绝启动（fail-fast），默认 false=告警+回落单一上游。
    route_strict: bool = field(default_factory=lambda: _env("BLADEX_ROUTE_STRICT", "false").lower() in ("true", "1", "yes"))
    # T4（RA4）：熔断冷却秒数——模型触发 failover 类错误后，冷却期内选池跳过。
    route_cb_cooldown_s: float = field(default_factory=lambda: float(_env("BLADEX_ROUTE_CB_COOLDOWN_S", "30")))
    # 懒加载的结构化路由配置缓存
    _routing_config_cache: RoutingConfig | None = None
    # ADR-0021: 懒加载的身份注册表缓存（None = 未加载；加载后可能为 empty registry）
    _identity_registry_cache: Any | None = None
    _identity_registry_loaded: bool = False

    @property
    def model_route(self) -> ModelRoute:
        return ModelRoute(
            model=self.upstream_model,
            api_base=self.upstream_api_base,
            api_key=self.upstream_api_key,
        )

    @property
    def routing_config(self) -> RoutingConfig:
        """懒加载结构化路由配置（routing.toml）；文件不存在 → 空配置."""
        if self._routing_config_cache is None:
            self._routing_config_cache = RoutingConfig.from_toml(self.routing_config_path)
        return self._routing_config_cache

    @property
    def identity_registry(self) -> Any:
        # ADR-0021 section 2: identity.toml; missing -> empty registry
        # -> personal mode single local identity (section 2.3, revised 2026-08-16).
        # Validation violation (dup key / dangling ref / team cycle) -> ValueError (refuse startup).
        # _loaded flag distinguishes "not loaded" from "loaded as empty" (cache empty too).
        if not self._identity_registry_loaded:
            from bladex_proxy.identity_registry import IdentityRegistry
            self._identity_registry_cache = IdentityRegistry.from_toml(self.identity_config_path)
            self._identity_registry_loaded = True
        return self._identity_registry_cache

    @property
    def effective_hard_rules(self) -> list[str]:
        if self.hard_rules:
            return self.hard_rules
        return _DEFAULT_HARD_RULES

    @property
    def key_store(self) -> KeyStore:
        """客户端 key 管理（解析 client_keys_raw 成 KeyStore）。"""
        return KeyStore.parse(self.client_keys_raw)

    def auth_check(
        self,
        api_key: str | None,
        quiet: bool = False,
        endpoint: str | None = None,
    ) -> tuple[bool, str]:
        """校验客户端 key（数据面 /v1/*）。

        quiet=True 时 auth_ok 降 debug（高频内部端点用）；
        endpoint 记进日志，让"同一把 key 的不同用途"可分辨（见 KeyStore.verify）。
        """
        if not self.auth_enabled:
            return True, "auth_disabled"
        return self.key_store.verify(api_key, quiet=quiet, endpoint=endpoint)

    # ── ADR-0027 §2.2：管理面鉴权（与数据面分级）──

    @property
    def admin_key_store(self) -> KeyStore:
        """管理面 key 集合（BLADEX_ADMIN_KEYS）。空 = 未配独立 admin key。"""
        return KeyStore.parse(self.admin_keys_raw)

    @property
    def admin_keys_configured(self) -> bool:
        return bool(self.admin_keys_raw.strip())

    @property
    def is_enterprise_identity(self) -> bool:
        """是否企业形态（存在 identity.toml）——多 principal 下共用 key = 越权面。"""
        return Path(self.identity_config_path).is_file()

    def admin_auth_check(self, api_key: str | None) -> tuple[bool, str]:
        """校验管理面 key（/admin/*）。

        三级语义（ADR-0027 §2.2）：
        - 配了 BLADEX_ADMIN_KEYS → 只认 admin key，数据面 key 一律拒绝。
        - 未配 + 个人模式（无 identity.toml）→ 回落 client key（与共用形态逐字一致）。
        - 未配 + 企业形态（有 identity.toml）→ 启动校验已拒启动，此处保守拒绝兜底。

        auth_enabled=false 时与数据面同语义（放行），由启动横幅承担告警。
        """
        if not self.auth_enabled:
            return True, "auth_disabled"
        if self.admin_keys_configured:
            return self.admin_key_store.verify(api_key)
        if self.is_enterprise_identity:
            return False, "admin_keys_required"
        return self.key_store.verify(api_key)

    def validate_admin_key_policy(self) -> str | None:
        """启动校验：企业形态必须配独立 admin key。

        返回 None = 通过；返回字符串 = 拒启动的原因（调用方抛 ValueError）。
        个人模式未配只回落 + warning（不阻塞），warning 由 server 启动路径发出。
        """
        if self.admin_keys_configured or not self.auth_enabled:
            return None
        if self.is_enterprise_identity:
            return (
                "identity.toml exists (enterprise deployment) but BLADEX_ADMIN_KEYS is not set. "
                "Sharing BLADEX_CLIENT_KEYS with the admin plane means any agent key can run "
                "destructive admin operations (turn tombstone, Matter merge/split, scope promote). "
                "Fix: set BLADEX_ADMIN_KEYS=\"bladex-admin-xxx||admin\" in config/.env "
                "(same `key||label` format as BLADEX_CLIENT_KEYS)."
            )
        return None


# 默认硬规则
_DEFAULT_HARD_RULES = [
    "NEVER use emojis in responses.",
    "MUST respond in the same language as the user's message.",
]
