"""Embedding 后端可选化（2026-07-26，见 embedding 后端评估记录）。

背景：强制本地 multilingual-e5-large（~2.2GB/进程 × proxy+consolidator 两份）在弱机器
（普通 ECS）上是部署障碍。本模块把 embedding 后端做成配置选项：

- **local**（默认，零回归）：fastembed 本地推理，模型可选 e5-large/base/small
  （方案 B：小模型档，内存/CPU 降 5–10 倍，中文检索有折损）。
- **api**（方案 C）：经 Router 网关调线上 embedding API（借用原则：多家适配是通用品，
  不自研）。隐私代价：全部记忆内容（fact + query）外发 —— 启动打强警告；
  敏感层（ADR-0021）启用时 fail-closed 强制本地。
- **proxy**（方案 D）：consolidator 复用 proxy 进程的 `/v1/embeddings` 端点，
  单机两进程共享一份模型，内存减半。仅 consolidator 侧可用。

硬约束（评估 §4）：
1. 向量空间全局唯一 —— query 与 fact 同模型同空间；换后端 = 换模型 = 全量重嵌
   （`scripts/reembed_index.py`），Memory Index 以 model_id 不变量拒绝混写（memory_index）。
2. 前缀责任在 adapter —— 调用方不再手拼 "query: "/"passage: "（e5 双前缀 quirk
   在 FastEmbedAdapter 内逐字保持，阈值标定兼容）。API 模型不加前缀。
3. 阈值 per-model profile —— novelty/semantic/digestion 是 e5-large 上标定的，
   非标定模型给保守默认 + 告警。
"""

from __future__ import annotations

import contextlib
import contextvars
import os
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger()

# 后端取值
BACKEND_LOCAL = "local"
BACKEND_API = "api"
BACKEND_PROXY = "proxy"
#: V-A6：独立 embedding 模块（IPC）。proxy 与 consolidator 都当客户端，
#: 优先级由 role 决定（proxy=hot / consolidator=bulk）——MQ-W8 的解。
BACKEND_IPC = "ipc"

# ── 调用用途归因（2026-08-09）───────────────────────────────────────────────
# proxy 档下 embedding 是跨进程 HTTP，远端日志里"谁在调"只能靠调用方自己说。
# key label 说不清（两个进程共用一把 key），端点路径也说不清（只有一个端点）。
# 故用 contextvar 在 consolidator 侧标注阶段，ProxyEmbedAdapter 随请求带上——
# 不改任何函数签名，不侵入 MemoryIndex 那一堆嵌入点。
_embed_purpose: contextvars.ContextVar[str] = contextvars.ContextVar(
    "bladex_embed_purpose", default="")


@contextlib.contextmanager
def embed_purpose(purpose: str) -> Iterator[None]:
    """标注这段代码里的 embedding 调用是干什么的（如 "rebuild" / "export"）。"""
    token = _embed_purpose.set(purpose)
    try:
        yield
    finally:
        _embed_purpose.reset(token)


def current_embed_purpose() -> str:
    """当前用途标注（未标注返回空串）。"""
    return _embed_purpose.get()


class EmbedCallLog:
    """`/v1/embeddings` 调用归因日志：**首次即报 + 周期汇总**。

    这条日志走过两次极端（都不好用）：
    - 每请求一条 INFO（`auth_ok label=...`）——刷屏，且只有 key label，
      共享模型档下 consolidator 与 agent 对话共用一把 key，看不出谁在调；
    - 整条降 debug ——安静了，但"谁在调 embedding"彻底不可见。

    折中：每个 (caller, purpose, input_type) 组合**第一次**出现立刻 INFO
    （新调用方 / 新阶段一眼可见），之后按窗口汇总一条（量、字符、耗时），
    逐条明细留 debug。
    """

    def __init__(self, window_s: float = 60.0) -> None:
        self._window_s = window_s
        self._seen: set[tuple[str, str, str]] = set()
        self._agg: dict[tuple[str, str, str], dict[str, float]] = {}
        self._t_last = 0.0

    def reset(self) -> None:
        self._seen.clear()
        self._agg.clear()
        self._t_last = 0.0

    def record(
        self,
        *,
        caller: str,
        caller_pid: str,
        purpose: str,
        input_type: str,
        count: int,
        chars: int,
        embed_ms: float,
        key_label: str = "",
        caller_proc: str = "-",
    ) -> None:
        import time

        key = (caller, purpose, input_type)
        logger.debug("embed_request", caller=caller, caller_pid=caller_pid,
                     caller_proc=caller_proc,
                     purpose=purpose, input_type=input_type, count=count,
                     chars=chars, embed_ms=embed_ms, key_label=key_label)
        now = time.monotonic()
        if key not in self._seen:
            self._seen.add(key)
            logger.info("embed_caller_seen", caller=caller, caller_pid=caller_pid,
                        caller_proc=caller_proc,
                        purpose=purpose, input_type=input_type, count=count,
                        chars=chars, embed_ms=embed_ms, key_label=key_label)
            self._t_last = self._t_last or now
        a = self._agg.setdefault(
            key, {"requests": 0, "texts": 0, "chars": 0, "ms": 0.0, "max_ms": 0.0})
        a["requests"] += 1
        a["texts"] += count
        a["chars"] += chars
        a["ms"] += embed_ms
        a["max_ms"] = max(a["max_ms"], embed_ms)
        if self._t_last == 0.0:
            self._t_last = now
        if now - self._t_last < self._window_s:
            return
        by = [
            {"caller": c, "purpose": p or "-", "input_type": it,
             "requests": int(v["requests"]), "texts": int(v["texts"]),
             "chars": int(v["chars"]), "total_ms": round(v["ms"], 1),
             "max_ms": round(v["max_ms"], 1)}
            for (c, p, it), v in sorted(self._agg.items(),
                                        key=lambda kv: -kv[1]["requests"])
        ]
        logger.info("embed_requests_summary",
                    window_s=round(now - self._t_last, 1), by=by)
        self._agg.clear()
        self._t_last = now

# 默认本地模型：面向全球开发者的开源默认 —— 英文优先、足够轻量（384d / 0.067GB），
# 新装用户开箱即用不必先下 2.24GB。中文/多语用户按 MODEL_PROFILES 换档（需 reembed）。
# 注：已有 Memory Index 库（__meta__/embed_model_id 已盖章）且未显式配置模型时，沿用库内模型
# —— 升级 BladeX 不会因默认值变更而废掉现有向量空间（见 resolve_embed_settings）。
DEFAULT_LOCAL_MODEL = "BAAI/bge-small-en-v1.5"

# 历史默认（2026-07-26 之前）；存量部署多为此模型
LEGACY_LOCAL_MODEL = "intfloat/multilingual-e5-large"

# per-model 阈值 profile（评估 §4.3）。
# calibrated=True 表示阈值在该模型上经过真实数据标定；False 表示沿用 e5-large
# 标定值作保守默认（启动告警，建议按真实流量重标）。
#
# 🔴 **novelty 不再是 per-model 标定量**（2026-08-07 修，M5-2 前夜发现）。
#
# M2（08-06）改的不只是数字，是**语义**：novelty 从"去重线"变成"只拦完全重发"，
# 0.95 → 0.98，默认值收进 `bladex_core.flags`。但这张表里还留着 07-26 按**旧语义**
# 标定的 0.95，而 `effective_thresholds` 的优先级是
#     显式 env > per-model profile > flags 默认
# —— profile **静默压过 flags**，于是 M2 的 0.98 在任何用 e5-large 的机器上
# （即生产机）**从未生效过**：0.95–0.98 那一带的候选在到达裁决层之前就被判重杀掉，
# 裁决漏斗从一开始就是饿的。
#
# 现在三档一律取 flags 的值，并由 `test_profile_novelty_matches_flags` 钉住。
# 想让某个模型用不同的 novelty，请**显式改那条断言**——让分歧是刻意的、看得见的，
# 而不是一个静默覆盖。semantic / digestion 语义未变，仍按模型标定。
# ⚠️ local 档模型必须在 fastembed 白名单内（TextEmbedding.list_supported_models()，
# 共 ~30 个）——多语 e5 系只有 e5-large，**没有 e5-small/base**。小模型档用下面
# 这些 fastembed 原生支持的多语模型（2026-07-26 实测核验）。
MODEL_PROFILES: dict[str, dict[str, Any]] = {
    # ── 英语优先（默认档）──
    "BAAI/bge-small-en-v1.5": {
        "dim": 384, "size_gb": 0.07, "languages": "en",
        "note": "default; lightest, English-only, best fit for weak machines/small VPS",
        "novelty": 0.98, "semantic": 0.85, "digestion": 0.85, "calibrated": False,
    },
    "BAAI/bge-base-en-v1.5": {
        "dim": 768, "size_gb": 0.21, "languages": "en",
        "note": "English, better recall than bge-small at 3x size",
        "novelty": 0.98, "semantic": 0.85, "digestion": 0.85, "calibrated": False,
    },
    "BAAI/bge-large-en-v1.5": {
        "dim": 1024, "size_gb": 1.20, "languages": "en",
        "note": "English, highest quality of the BGE line; needs ~2GB RAM per process",
        "novelty": 0.98, "semantic": 0.85, "digestion": 0.85, "calibrated": False,
    },
    # ── 多语 / 中文 ──
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": {
        "dim": 384, "size_gb": 0.22, "languages": "50+",
        "note": "lightest multilingual option; good CJK coverage for its size",
        "novelty": 0.98, "semantic": 0.85, "digestion": 0.85, "calibrated": False,
    },
    "jinaai/jina-embeddings-v2-base-zh": {
        "dim": 768, "size_gb": 0.64, "languages": "zh+en",
        "note": "tuned for Chinese-English bilingual corpora",
        "novelty": 0.98, "semantic": 0.85, "digestion": 0.85, "calibrated": False,
    },
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": {
        "dim": 768, "size_gb": 1.00, "languages": "50+",
        "note": "stronger multilingual than MiniLM-L12 at ~5x size",
        "novelty": 0.98, "semantic": 0.85, "digestion": 0.85, "calibrated": False,
    },
    "intfloat/multilingual-e5-large": {
        "dim": 1024, "size_gb": 2.24, "languages": "100+",
        "note": "legacy default; best multilingual quality, heaviest; "
                "the only model whose thresholds are calibrated on real traffic",
        "novelty": 0.98, "semantic": 0.85, "digestion": 0.85, "calibrated": True,
    },
}

# 阈值对应的显式 env（显式设置永远压过 profile）
_THRESHOLD_ENVS = {
    "novelty": "BLADEX_NOVELTY_THRESHOLD",
    "semantic": "BLADEX_SEMANTIC_THRESHOLD",
    "digestion": "BLADEX_DIGESTION_THRESHOLD",
}


@dataclass
class EmbedSettings:
    """解析后的 embedding 生效配置（三层合并结果）。

    优先级：显式 env（BLADEX_EMBED_*，仅当该 env 真的被设置）
          > routing.toml [embedding]（结构化管理，与 [distill] 同先例）
          > 内置默认（local + e5-large 零回归）。
    密钥纪律：toml 只存 api_key_env / proxy_key_env（环境变量名），密钥不入 toml。
    """

    backend: str = BACKEND_LOCAL
    model: str = ""
    api_base: str = ""
    api_key: str = ""
    proxy_url: str = "http://127.0.0.1:38080"
    proxy_key: str = ""
    query_cache_size: int = 256
    # 合并后的阈值 profile 表（内置 MODEL_PROFILES + toml [embedding.profiles] 覆盖/新增）
    profiles: dict[str, dict[str, Any]] = field(default_factory=dict)
    # 每个字段的**来源**：env | routing.toml | default（G9.4 H3 `config show --effective`）。
    # 🔴 由 `pick` 在合并的同一处记录，不另写一份判断——"生效值"与"它从哪来"
    # 必须出自同一次合并，否则展示层会和真实优先级各说各话（这正是 H3 要终结的排查成本）。
    sources: dict[str, str] = field(default_factory=dict)


def _stored_local_model(cfg: Any) -> str:
    """读 Memory Index 已盖章的向量空间模型名（仅 local: 前缀）；读不到返回空串。

    ⚠️ 绝不用自造 Options 打开 live Memory Index 库（2026-07-26 事故教训）：
    rocksdict 在库目录里维护 sidecar `rocksdict-config.json` 记录 raw_mode，
    用 `Options(raw_mode=True)` 打开会把 sidecar 改写成 raw_mode=true，之后
    MemoryIndex 的默认打开就会 comparator 不匹配（BytewiseComparator vs rocksdict）
    -> consolidator 拒启动。数据本身不受损，但库"打不开"。
    因此这里只读 sidecar 旁边的 stamp 侧车文件，**完全不碰 RocksDB**。
    stamp 侧车由 MemoryIndex 在盖章时同步写出（见 memory_index._check_embed_model_invariant）。
    """
    try:
        from pathlib import Path

        index_path = Path(getattr(cfg, "index_path", "") or "")
        if not index_path:
            return ""
        sidecar = index_path / "embed_model_id"
        if not sidecar.is_file():
            return ""
        stored = sidecar.read_text(encoding="utf-8").strip()
        if stored.startswith("local:"):
            return stored[len("local:"):]
    except Exception:  # noqa: BLE001 — 兼容性探测，失败即视为无历史库
        return ""
    return ""


def resolve_embed_settings(cfg: Any) -> EmbedSettings:
    """把 env / routing.toml [embedding] / 内置默认合并成生效配置。"""
    toml_cfg = getattr(getattr(cfg, "routing_config", None), "embedding", None)

    s = EmbedSettings()

    def pick(env_key: str, cfg_val: Any, toml_val: Any, default: Any,
             field_name: str = "") -> Any:
        # 显式优先级：env 真被设置（回滚通道，与 distill 同序）或 cfg 字段被程序化
        # 改离默认值（测试/嵌入式构造）> toml > 内置默认。
        # `field_name` 只用于把这次判断的**结果来源**记进 s.sources（H3 可见性），
        # 不参与任何取值逻辑；不传就不记，行为逐字不变。
        def note(src: str) -> None:
            if field_name:
                s.sources[field_name] = src

        if env_key in os.environ:
            note("env")
            return cfg_val
        if cfg_val not in (None, "", default):
            note("env")   # 程序化构造（测试/嵌入式）与 env 同级，对外都算"显式设置"
            return cfg_val
        if toml_val not in (None, ""):
            note("routing.toml")
            return toml_val
        note("default")
        return default

    s.backend = str(pick("BLADEX_EMBED_BACKEND", getattr(cfg, "embed_backend", ""),
                         getattr(toml_cfg, "backend", ""), BACKEND_LOCAL, "backend")).strip().lower()
    s.model = pick("BLADEX_EMBED_MODEL", getattr(cfg, "embed_model", ""),
                   getattr(toml_cfg, "model", ""), "", "model")
    s.api_base = pick("BLADEX_EMBED_API_BASE", getattr(cfg, "embed_api_base", ""),
                      getattr(toml_cfg, "api_base", ""), "", "api_base")
    # 密钥：显式 env BLADEX_EMBED_API_KEY > toml api_key_env 引用的环境变量
    key_env = getattr(toml_cfg, "api_key_env", "") or ""
    s.api_key = pick("BLADEX_EMBED_API_KEY", getattr(cfg, "embed_api_key", ""),
                     os.environ.get(key_env, "") if key_env else "", "", "api_key")
    s.proxy_url = pick("BLADEX_EMBED_PROXY_URL", getattr(cfg, "embed_proxy_url", ""),
                       getattr(toml_cfg, "proxy_url", ""), "http://127.0.0.1:38080", "proxy_url")
    pkey_env = getattr(toml_cfg, "proxy_key_env", "") or ""
    s.proxy_key = pick("BLADEX_EMBED_PROXY_KEY", getattr(cfg, "embed_proxy_key", ""),
                       os.environ.get(pkey_env, "") if pkey_env else "", "", "proxy_key")
    s.query_cache_size = int(pick("BLADEX_EMBED_QUERY_CACHE",
                                  getattr(cfg, "embed_query_cache_size", 256),
                                  getattr(toml_cfg, "query_cache_size", None), 256,
                                  "query_cache_size"))

    # 默认值变更保护（2026-07-26 默认 e5-large -> bge-small-en-v1.5）：
    # 用户没有显式指定模型、但 Memory Index 已盖章某个 local 模型时，沿用库内模型——
    # 升级 BladeX 不因默认值变更而废掉现有向量空间（否则 consolidator 会拒启动）。
    #
    # 🔴 2026-08-31：判据从 `== BACKEND_LOCAL` 扩到「**任何会加载本地模型的档**」。
    # 本条写于只有 local/api/proxy 的时候；后来加的 `ipc` **回落路径就是加载本地
    # 模型**（`build_ipc_adapter` 的 `_local_factory`），却不在保护范围里 ⇒
    # 配了 ipc 又没显式写 model 的部署，回落时会拿到新默认 `bge-small-en`，
    # 与库内已盖的 e5-large 章不符 ⇒ 索引转只读、检索整体停摆。
    # 枚举加了新成员、旧分支没跟上——本仓的高发形态，这次是提前发现的。
    if not s.model and s.backend in (BACKEND_LOCAL, BACKEND_IPC):
        stored = _stored_local_model(cfg)
        if stored and stored != DEFAULT_LOCAL_MODEL:
            logger.info("embed_model_from_store", model=stored,
                        hint="Memory Index already stamped with this model; keeping it "
                             "(set [embedding].model explicitly + run reembed_index.py to switch)")
            s.model = stored
            # 第四种来源：库里已盖的章。它压过"内置默认"，但不压任何显式设置——
            # 展示层必须能说出这一种，否则用户会看到一个模板里没有的模型名而不知道为什么。
            s.sources["model"] = "index stamp"

    # 阈值 profile 合并：内置为底，toml 覆盖/新增（只覆盖非 None 字段）
    s.profiles = {k: dict(v) for k, v in MODEL_PROFILES.items()}
    for name, prof in (getattr(toml_cfg, "profiles", None) or {}).items():
        base = dict(s.profiles.get(name, {}))
        for f_name in ("dim", "novelty", "semantic", "digestion"):
            val = getattr(prof, f_name, None)
            if val is not None:
                base[f_name] = val
        base["calibrated"] = bool(getattr(prof, "calibrated", False)) or base.get("calibrated", False)
        s.profiles[name] = base
    return s


# HuggingFace 镜像回退链（主站不可达时依次尝试，减少用户操作复杂度）。
# 用户显式设了 HF_ENDPOINT 则完全尊重，不做任何回退猜测。
_HF_MIRRORS = ("https://hf-mirror.com",)
_HF_PROBE_TIMEOUT_S = 5.0


def _hf_repos_for(model_name: str) -> list[str]:
    """模型在 HF 上的真实仓库名（可能与逻辑模型名不同）。

    ⚠️ fastembed 的下载源常是 qdrant 转换过的 ONNX 仓库，例如
      intfloat/multilingual-e5-large -> qdrant/multilingual-e5-large-onnx
      BAAI/bge-small-en-v1.5        -> qdrant/bge-small-en-v1.5-onnx-q
    按逻辑名拼缓存目录会永远匹配不到 -> 每次启动误判"未缓存"去探网，
    离线机器上更会误判成"缺失"而拒启动。故从 fastembed 元数据取 sources.hf。
    """
    repos: list[str] = []
    try:
        from fastembed import TextEmbedding

        for m in TextEmbedding.list_supported_models():
            if m["model"] == model_name:
                hf = (m.get("sources") or {}).get("hf")
                if hf:
                    repos.append(hf)
                break
    except Exception:  # noqa: BLE001 — 元数据不可用时退回逻辑名
        pass
    repos.append(model_name)  # 兜底：部分版本按逻辑名缓存
    return repos


def _model_cached(model_name: str, cache_dir: str) -> bool:
    """模型是否已在本地缓存（避免每次启动都探网/重下）。

    fastembed 走 huggingface_hub 缓存布局：<cache>/models--<org>--<name>/。
    判定宽松（目录存在且含权重/配置文件即认为已缓存）——完整性由加载时兜底。
    """
    from pathlib import Path

    base = Path(cache_dir)
    if not base.is_dir():
        return False
    for repo in _hf_repos_for(model_name):
        d = base / ("models--" + repo.replace("/", "--"))
        if not d.is_dir():
            continue
        if any(d.rglob("*.onnx")) or any(d.rglob("*.bin")) or any(d.rglob("*.json")):
            return True
    return False


def _endpoint_reachable(url: str) -> bool:
    try:
        import httpx

        r = httpx.head(url, timeout=_HF_PROBE_TIMEOUT_S, follow_redirects=True)
        return r.status_code < 500
    except Exception:
        return False


def ensure_local_model(model_name: str, cache_dir: str) -> bool:
    """启动前确保本地模型可用；缺失则自动下载（必要时切镜像）。

    流程（尽量少让用户操心）：
      1. 已缓存 -> 直接返回；
      2. 模型不在 fastembed 白名单 -> 报错并列出可选模型（不静默回落）；
      3. 未缓存 -> 探测 HF 主站；不可达则设 HF_ENDPOINT 到镜像再下；
      4. 阻塞下载（首次启动慢是预期的，日志说明体积与来源）。

    返回 True = 模型就绪。下载失败抛 RuntimeError（调用方决定降级/拒启动）。
    用户已显式设置 HF_ENDPOINT 时不改写它。
    """
    if _model_cached(model_name, cache_dir):
        logger.info("embed_model_cached", model=model_name, cache_dir=cache_dir)
        return True

    from fastembed import TextEmbedding

    supported = [m["model"] for m in TextEmbedding.list_supported_models()]
    if model_name not in supported:
        recommended = list(MODEL_PROFILES.keys())
        raise RuntimeError(
            f"embedding model {model_name!r} is not supported by fastembed. "
            f"Recommended models: {recommended}"
        )

    profile = MODEL_PROFILES.get(model_name, {})
    size = profile.get("size_gb")
    user_endpoint = os.environ.get("HF_ENDPOINT")
    source = user_endpoint or "https://huggingface.co"

    if not user_endpoint and not _endpoint_reachable("https://huggingface.co"):
        for mirror in _HF_MIRRORS:
            if _endpoint_reachable(mirror):
                os.environ["HF_ENDPOINT"] = mirror
                source = mirror
                logger.warning("embed_model_mirror_fallback", mirror=mirror,
                               hint="huggingface.co unreachable; using mirror")
                break
        else:
            raise RuntimeError(
                f"cannot download embedding model {model_name!r}: neither "
                f"huggingface.co nor mirrors {list(_HF_MIRRORS)} are reachable. "
                f"Set HF_ENDPOINT to a reachable mirror, or pre-populate {cache_dir}."
            )

    logger.info("embed_model_download_start", model=model_name,
                size_gb=size, source=source, cache_dir=cache_dir,
                hint="first start downloads the model; this blocks until finished")
    try:
        TextEmbedding(model_name=model_name, cache_dir=cache_dir)
    except Exception as e:
        raise RuntimeError(
            f"embedding model download failed for {model_name!r} from {source}: {e}"
        ) from e
    logger.info("embed_model_download_done", model=model_name, source=source)
    return True


class _QueryLRU:
    """query 嵌入 LRU 缓存（评估 §4.4）。

    工具循环内同 query 重复率高（与路由裁判 LRU 同理，RA3）；
    api/proxy 后端每次 embed 都是网络往返，缓存收益最大。
    """

    def __init__(self, capacity: int = 256) -> None:
        self._capacity = max(0, capacity)
        self._data: OrderedDict[str, list[float]] = OrderedDict()

    def get(self, key: str) -> list[float] | None:
        if key not in self._data:
            return None
        self._data.move_to_end(key)
        return self._data[key]

    def put(self, key: str, value: list[float]) -> None:
        if self._capacity <= 0:
            return
        self._data[key] = value
        self._data.move_to_end(key)
        while len(self._data) > self._capacity:
            self._data.popitem(last=False)


class RouterEmbedAdapter:
    """线上 embedding API adapter（方案 C，经 Router 网关）。

    - 无前缀约定：API 模型（doubao-embedding / text-embedding-3 / bge-m3 等）
      直接吃原文，query/passage 同一空间（对称模型）。
    - embed_query 走 LRU（网络往返成本高）。
    """

    def __init__(
        self,
        model: str,
        api_base: str = "",
        api_key: str = "",
        query_cache_size: int = 256,
        timeout_s: float = 30.0,
    ) -> None:
        if not model:
            raise ValueError("api embedding backend requires BLADEX_EMBED_MODEL")
        self._model = model
        self._api_base = api_base
        self._api_key = api_key
        self._timeout_s = timeout_s
        self._query_cache = _QueryLRU(query_cache_size)

    def _call(self, texts: list[str]) -> list[list[float]]:
        from bladex_proxy import router_sdk

        kwargs: dict[str, Any] = {
            "model": self._model,
            "input": texts,
            "timeout": self._timeout_s,
        }
        if self._api_base:
            kwargs["api_base"] = self._api_base
        if self._api_key:
            kwargs["api_key"] = self._api_key
        resp = router_sdk.embedding(**kwargs)
        data = resp["data"] if isinstance(resp, dict) else resp.data
        rows = sorted(data, key=lambda d: d["index"] if isinstance(d, dict) else d.index)
        out: list[list[float]] = []
        for row in rows:
            emb = row["embedding"] if isinstance(row, dict) else row.embedding
            out.append([float(x) for x in emb])
        return out

    def embed(self, texts: list[str]) -> list[list[float]]:
        """passage 语义（API 模型无前缀 = 原文）。"""
        return self._call(list(texts))

    def embed_passage(self, texts: list[str]) -> list[list[float]]:
        return self.embed(texts)

    def embed_query(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float] | None] = [self._query_cache.get(t) for t in texts]
        missing = [t for t, v in zip(texts, out, strict=True) if v is None]
        if missing:
            fresh = self._call(missing)
            it = iter(fresh)
            for i, v in enumerate(out):
                if v is None:
                    vec = next(it)
                    self._query_cache.put(texts[i], vec)
                    out[i] = vec
        return [v for v in out if v is not None]

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def model_identity(self) -> str:
        return f"api:{self._model}"


class ProxyEmbedAdapter:
    """共享模型 adapter（方案 D）：HTTP 调 BladeX proxy 的 `/v1/embeddings`。

    consolidator 用它复用 proxy 进程里已加载的模型 —— 单机免二次载模（省 ~2.2GB）。
    向量空间身份取远端的 `bladex_model_identity`（远端后端决定空间）。
    `input_type` 是 BladeX 对 OpenAI embeddings 协议的增量扩展（默认 passage，
    向后兼容），远端据此走 embed_query/embed_passage —— 保证与远端自身检索同前缀。
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        query_cache_size: int = 256,
        timeout_s: float = 60.0,
        caller: str = "",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout_s = timeout_s
        self._query_cache = _QueryLRU(query_cache_size)
        self._remote_model: str | None = None
        self._remote_identity: str | None = None
        # 调用方归因（2026-08-09）：共享模型档下 consolidator 用的就是数据面那把
        # key，远端只看 key label 分不出是"agent 在对话"还是"consolidator 在嵌入"。
        # 故恒发调用方 header —— 是 BladeX 自家两个进程之间的私有约定，
        # 对 OpenAI 协议是纯增量（远端不认也不影响）。
        self._caller = caller or os.environ.get("BLADEX_EMBED_CALLER") or "consolidator"
        self._caller_pid = str(os.getpid())
        # 进程名：role 只说"我是 proxy 档的客户端"，但 CLI（`bladex doctor`）、
        # scripts/reembed_index.py 也用 role="consolidator" —— 光看 caller 会把
        # 一次性工具误当常驻 consolidator。故把入口进程名一并带上。
        import sys

        self._caller_proc = os.path.basename(sys.argv[0] or "") or "?"

    def _call(self, texts: list[str], input_type: str) -> list[list[float]]:
        import httpx

        headers = {
            "X-BladeX-Caller": self._caller,
            "X-BladeX-Caller-PID": self._caller_pid,
            "X-BladeX-Caller-Proc": self._caller_proc,
        }
        purpose = current_embed_purpose()
        if purpose:
            headers["X-BladeX-Embed-Purpose"] = purpose
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        resp = httpx.post(
            f"{self._base_url}/v1/embeddings",
            json={"input": texts, "input_type": input_type},
            headers=headers,
            timeout=self._timeout_s,
        )
        resp.raise_for_status()
        body = resp.json()
        self._remote_model = body.get("model") or self._remote_model
        self._remote_identity = body.get("bladex_model_identity") or self._remote_identity
        rows = sorted(body["data"], key=lambda d: d["index"])
        return [[float(x) for x in row["embedding"]] for row in rows]

    def _probe(self) -> None:
        if self._remote_identity is None:
            self._call(["__probe__"], "passage")

    def wait_ready(self, timeout_s: float = 120.0, interval_s: float = 2.0) -> None:
        """等远端 proxy 起来（共享模式启动竞态）。

        run_proxy.sh 先拉 consolidator 再起 uvicorn，consolidator 首个 probe 往往
        早于 proxy 监听；proxy 侧还要加载模型（首次可能在下载）。故指数退避重试
        而非一次失败就退出。超时抛最后一次异常，由调用方决定拒启动。
        """
        import time

        deadline = time.monotonic() + timeout_s
        attempt = 0
        last: Exception | None = None
        while time.monotonic() < deadline:
            try:
                self._probe()
                if attempt:
                    logger.info("embed_proxy_ready", attempts=attempt + 1, url=self._base_url)
                return
            except Exception as e:  # noqa: BLE001 — 启动期重试，最后一次才上报
                last = e
                attempt += 1
                if attempt == 1:
                    logger.info("embed_proxy_waiting", url=self._base_url,
                                timeout_s=timeout_s,
                                hint="waiting for the BladeX proxy to serve /v1/embeddings")
                time.sleep(min(interval_s * attempt, 10.0))
        raise RuntimeError(
            f"BladeX proxy at {self._base_url} not ready after {timeout_s}s "
            f"(last error: {last}). Is the proxy running? "
            f"If it requires auth, set the client key via [embedding].proxy_key_env."
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        """passage 语义（远端 adapter 决定前缀）。"""
        return self._call(list(texts), "passage")

    def embed_passage(self, texts: list[str]) -> list[list[float]]:
        return self.embed(texts)

    def embed_query(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float] | None] = [self._query_cache.get(t) for t in texts]
        missing = [t for t, v in zip(texts, out, strict=True) if v is None]
        if missing:
            fresh = self._call(missing, "query")
            it = iter(fresh)
            for i, v in enumerate(out):
                if v is None:
                    vec = next(it)
                    self._query_cache.put(texts[i], vec)
                    out[i] = vec
        return [v for v in out if v is not None]

    @property
    def model_name(self) -> str:
        self._probe()
        return self._remote_model or "bladex-proxy-embed"

    @property
    def model_identity(self) -> str:
        """远端的向量空间身份（与远端 proxy/consolidator 盖章一致）。"""
        self._probe()
        return self._remote_identity or f"proxy:{self._base_url}"


def validate_embed_sensitivity(cfg: Any) -> None:
    """ADR-0021 邻接：敏感层开 + embedding 外发 -> 拒启动(strict)/告警+强制本地。

    embedding 后端看到**全部**记忆内容（每条 fact + 每轮 query），比裁判
    （只看 query）暴露面更大 —— 同一原则，更硬的 gate。
    """
    try:
        sens = cfg.routing_config.sensitivity_config()
    except Exception:
        return
    if not getattr(sens, "enabled", False):
        return
    backend = resolve_embed_settings(cfg).backend
    if backend == BACKEND_API:
        msg = ("sensitivity enabled but embedding backend='api': all memory content "
               "(facts + queries) would be sent to external provider")
        if cfg.route_strict:
            raise RuntimeError(f"BLADEX_ROUTE_STRICT=true: {msg}")
        logger.error("SENSITIVITY_EMBED_NOT_LOCAL",
                     hint=msg + " -> forcing local backend")
    elif backend == BACKEND_PROXY:
        logger.warning("sensitivity_embed_proxy_backend",
                       hint="ensure the remote BladeX proxy itself uses a local "
                            "embedding backend; data boundary = your deployment")


def _sensitivity_enabled(cfg: Any) -> bool:
    try:
        sens = cfg.routing_config.sensitivity_config()
        return bool(getattr(sens, "enabled", False))
    except Exception:
        return False


def _local_model_for(settings: EmbedSettings, forced: bool) -> str:
    """回落 local 档时的模型名。

    forced=True（敏感度强制 / 误配回落）时 model 可能是 api 模型名，
    不能拿去喂 fastembed —— 不在内置本地 profile 表内就退默认 e5-large。
    """
    m = settings.model or DEFAULT_LOCAL_MODEL
    if forced and m not in MODEL_PROFILES:
        return DEFAULT_LOCAL_MODEL
    return m


def build_embedder(cfg: Any, role: str = "proxy") -> Any:
    """按配置构建 embedder。

    role: "proxy" | "consolidator"。backend=proxy 只对 consolidator 有意义
    （proxy 自己调自己没有意义），proxy 侧误配时报错并回落 local。
    构建失败向上抛（调用方自己决定降级策略，与现状 FastEmbedAdapter 一致）。
    """
    from bladex_proxy.storage.memory_index import FastEmbedAdapter

    s = resolve_embed_settings(cfg)
    backend = s.backend or BACKEND_LOCAL
    forced = False

    # 敏感度 fail-closed：api 档强制回本地（strict 模式在 validate 阶段已拒启动）
    if backend == BACKEND_API and _sensitivity_enabled(cfg):
        logger.error("embed_backend_forced_local", reason="sensitivity_enabled")
        backend = BACKEND_LOCAL
        forced = True

    if backend == BACKEND_PROXY:
        if role == "proxy":
            # 单机共享模式的**服务端**：proxy 自己必须真加载本地模型（它就是那份
            # 被共享的模型，经 /v1/embeddings 供 consolidator 调用）。这是该模式的
            # 预期行为，不是误配 —— 故 info 而非 error。
            logger.info("embed_backend_proxy_serving_local",
                        hint="backend=proxy: this proxy loads the local model and "
                             "serves it via /v1/embeddings for the consolidator")
            backend = BACKEND_LOCAL
            forced = True
        else:
            logger.info("embedder_backend", backend="proxy", url=s.proxy_url)
            return ProxyEmbedAdapter(
                s.proxy_url,
                api_key=s.proxy_key,
                query_cache_size=s.query_cache_size,
                caller=role,  # 远端 embed_request 日志据此归因
            )

    if backend == BACKEND_IPC:
        # V-A6：走独立模块。构建**不**在这里加载模型——回落时才懒加载（省内存）。
        from bladex_proxy.embed_client import build_ipc_adapter

        def _local_factory():
            from bladex_proxy.storage.memory_index import FastEmbedAdapter
            return FastEmbedAdapter(model_name=_local_model_for(s, True),
                                    cache_dir=cfg.fastembed_cache_path)

        adapter = build_ipc_adapter(cfg, role, _local_factory)
        if adapter is not None:
            logger.info("embedder_backend", backend="ipc", role=role)
            return adapter
        # 决断为 local（显式配置）⇒ 落到下面的本地路径
        backend = BACKEND_LOCAL
        forced = True

    if backend == BACKEND_API:
        logger.warning(
            "EMBED_API_PRIVACY",
            model=s.model,
            hint="ALL memory content (every fact + every query) will be sent to the "
                 "external embedding provider. Use backend=local to keep memory on-device.",
        )
        return RouterEmbedAdapter(
            model=s.model,
            api_base=s.api_base,
            api_key=s.api_key,
            query_cache_size=s.query_cache_size,
        )

    if backend != BACKEND_LOCAL:
        logger.warning("embed_backend_unknown", backend=backend, hint="falling back to local")
        forced = True

    model = _local_model_for(s, forced)
    logger.info("embedder_backend", backend="local", model=model)
    # 启动前确保模型就绪（缺失自动下载，HF 不可达自动切镜像）。
    # 失败抛给调用方：proxy 捕获后降级只注硬规则，consolidator 拒启动。
    ensure_local_model(model, cfg.fastembed_cache_path)
    return FastEmbedAdapter(model_name=model, cache_dir=cfg.fastembed_cache_path)


def embed_model_identity(cfg: Any, role: str = "proxy") -> str:
    """当前配置对应的向量空间身份串（不加载模型，Memory Index 不变量校验用）。

    backend=proxy 的身份取决于远端，这里返回占位；ProxyEmbedAdapter.model_identity
    probe 后才是真身份 —— consolidator 侧应优先用 adapter 的 model_identity。
    """
    s = resolve_embed_settings(cfg)
    backend = s.backend or BACKEND_LOCAL
    forced = False
    if backend == BACKEND_API and _sensitivity_enabled(cfg):
        backend = BACKEND_LOCAL
        forced = True
    if backend == BACKEND_PROXY and role == "proxy":
        backend = BACKEND_LOCAL
        forced = True
    if backend == BACKEND_API:
        return f"api:{s.model}"
    if backend == BACKEND_PROXY:
        return f"proxy:{s.proxy_url}"
    return f"local:{_local_model_for(s, forced)}"


def effective_thresholds(cfg: Any) -> dict[str, float]:
    """解析生效阈值：显式 env > per-model profile > 全局默认（+告警）。

    profile 表 = 内置 MODEL_PROFILES + routing.toml [embedding.profiles] 覆盖/新增
    （api 模型也可声明 profile，声明 calibrated=true 后不再告警）。
    评估 §4.3：阈值是 e5-large 上标定的。换模型时若用户没显式设阈值，
    用 profile 的保守默认并告警（建议真实流量重标）。
    """
    s = resolve_embed_settings(cfg)
    backend = s.backend or BACKEND_LOCAL
    model = s.model or (DEFAULT_LOCAL_MODEL if backend == BACKEND_LOCAL else "")
    profile = s.profiles.get(model)

    out = {
        "novelty": cfg.novelty_threshold,
        "semantic": cfg.semantic_threshold,
        "digestion": cfg.digestion_threshold,
    }
    uncalibrated = profile is None or not profile.get("calibrated", False)
    for name, env_key in _THRESHOLD_ENVS.items():
        if env_key in os.environ:
            continue  # 显式设置压过一切
        if profile is not None and name in profile:
            out[name] = float(profile[name])
    if model != DEFAULT_LOCAL_MODEL and uncalibrated:
        logger.warning(
            "embed_thresholds_uncalibrated",
            backend=backend, model=model or "(unset)",
            thresholds=out,
            hint="novelty/semantic/digestion were calibrated on e5-large; "
                 "re-validate on real traffic, set BLADEX_*_THRESHOLD, or declare "
                 "[embedding.profiles.\"<model>\"] with calibrated=true in routing.toml",
        )
    return out
