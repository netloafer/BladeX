"""Embedding 后端可选化验收剧本（2026-07-26）。

覆盖：
  - compat 帮手：旧接口（只有 embed）降级路径行为逐字保持
  - FastEmbedAdapter：passage/query 前缀逐字兼容（含历史双前缀 quirk）+ model_identity
  - RouterEmbedAdapter：无前缀原文 + query LRU + model_identity
  - ProxyEmbedAdapter：input_type 透传 + 远端身份对齐 + query LRU
  - build_embedder：三档分发 + proxy 侧 backend=proxy 回落 + 敏感度强制本地
  - effective_thresholds：显式 env > profile > 默认
  - validate_embed_sensitivity：strict 拒启动 / 非 strict 告警
  - Memory Index model_id 不变量：写模式盖章/拒混写、只读禁用检索
  - /v1/embeddings：input_type 扩展 + bladex_model_identity 字段
"""

from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace
from typing import Any

import pytest
from bladex_core.consolidation_proxy import embed_passage_compat, embed_query_compat
from bladex_core.flags import MEMORY_NUMERIC_DEFAULTS as _FLAGS
from bladex_proxy.embedding import (
    DEFAULT_LOCAL_MODEL,
    MODEL_PROFILES,
    ProxyEmbedAdapter,
    RouterEmbedAdapter,
    build_embedder,
    effective_thresholds,
    embed_model_identity,
    validate_embed_sensitivity,
)
from bladex_proxy.storage.memory_index import FastEmbedAdapter, MemoryIndex

# e5 系（前缀约定 + 阈值标定基准，旧默认）
_E5_MODEL = "intfloat/multilingual-e5-large"

# fastembed 白名单内的多语小模型档（e5 系只有 e5-large，无 small/base）
_SMALL_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# 真实 ensure_local_model 引用：build_embedder 系列测试会把模块属性 stub 掉
# （见 _no_model_download fixture），下载行为测试须绕过 stub 调真实实现。
from bladex_proxy.embedding import ensure_local_model as _REAL_ENSURE  # noqa: E402


class _RecordingModel:
    """替身 fastembed TextEmbedding：记录送进模型的最终文本。"""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def embed(self, texts: list[str]):
        for t in texts:
            self.seen.append(t)
            yield [float(len(t))] * 4


class _LegacyEmbedder:
    """旧接口 mock：只有 embed()（历史第三方/单测形态）。"""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.seen.extend(texts)
        return [[1.0, 2.0] for _ in texts]


_FLAG_NOVELTY = _FLAGS["BLADEX_NOVELTY_THRESHOLD"]


def _fake_cfg(**over: Any) -> SimpleNamespace:
    """最小 cfg 替身（build_embedder / effective_thresholds 只用这些字段）。"""
    sens_enabled = over.pop("sens_enabled", False)
    base = dict(
        embed_backend="local", embed_model="", embed_api_base="", embed_api_key="",
        embed_proxy_url="http://127.0.0.1:38080", embed_proxy_key="",
        embed_query_cache_size=8, fastembed_cache_path="data/fastembed_cache",
        # 🔴 读 flags 而不是写字面量：这里曾是**第三份** novelty 默认值副本，
        # M2 把 0.95 改成 0.98 时它没跟上，于是替身 cfg 与生产 cfg 说两套话。
        novelty_threshold=_FLAG_NOVELTY,
        semantic_threshold=0.85, digestion_threshold=0.85,
        route_strict=False,
        routing_config=SimpleNamespace(
            sensitivity_config=lambda: SimpleNamespace(enabled=sens_enabled)
        ),
    )
    base.update(over)
    return SimpleNamespace(**base)


# ── compat 帮手 ──

def test_compat_helpers_legacy_fallback_preserves_bytes():
    """旧 mock 只有 embed()：query 走手拼 "query: " 前缀（与历史调用方逐字一致）。"""
    emb = _LegacyEmbedder()
    embed_query_compat(emb, ["天气"])
    assert emb.seen == ["query: 天气"]
    emb2 = _LegacyEmbedder()
    embed_passage_compat(emb2, ["天气"])
    assert emb2.seen == ["天气"]


def test_compat_helpers_prefer_new_methods():
    """实现了 embed_query/embed_passage 的 adapter 直调（前缀 adapter 说了算）。"""
    calls: list[str] = []

    class _New:
        def embed(self, texts):  # 不应被走到
            calls.append("embed")
            return [[0.0]]

        def embed_query(self, texts):
            calls.append("embed_query")
            return [[1.0]]

        def embed_passage(self, texts):
            calls.append("embed_passage")
            return [[2.0]]

    assert embed_query_compat(_New(), ["x"]) == [[1.0]]
    assert embed_passage_compat(_New(), ["x"]) == [[2.0]]
    assert calls == ["embed_query", "embed_passage"]


# ── FastEmbedAdapter 前缀逐字兼容 ──

def test_fastembed_passage_prefix_unchanged():
    a = FastEmbedAdapter(model_name=_E5_MODEL)
    a._model = _RecordingModel()
    a.embed(["hello"])
    a.embed_passage(["world"])
    assert a._model.seen == ["passage: hello", "passage: world"]


def test_fastembed_query_double_prefix_quirk_preserved():
    """历史 quirk：query 实际嵌 "passage: query: ..."——阈值标定兼容，逐字保持。"""
    a = FastEmbedAdapter(model_name=_E5_MODEL)
    a._model = _RecordingModel()
    a.embed_query(["天气"])
    assert a._model.seen == ["passage: query: 天气"]


def test_fastembed_model_identity():
    assert FastEmbedAdapter().model_identity == f"local:{DEFAULT_LOCAL_MODEL}"
    assert FastEmbedAdapter(model_name=_SMALL_MODEL).model_identity == f"local:{_SMALL_MODEL}"


def test_fastembed_non_e5_no_prefix():
    """非 e5 白名单模型（MiniLM/mpnet/jina/bge）无前缀约定：原文直嵌。"""
    a = FastEmbedAdapter(model_name=_SMALL_MODEL)
    a._model = _RecordingModel()
    a.embed_passage(["你好"])
    a.embed_query(["世界"])
    assert a._model.seen == ["你好", "世界"]


# ── RouterEmbedAdapter（api 档）──

def _patch_router_embedding(monkeypatch, calls: list[list[str]]):
    """mock 目标是 Router 网关，不是供应商 SDK（工程规范 §6：SDK 只在网关里出现）。"""
    from bladex_proxy import router_sdk

    def fake_embedding(**kwargs):
        texts = kwargs["input"]
        calls.append(list(texts))
        return {"data": [
            {"index": i, "embedding": [float(len(t)), 1.0]} for i, t in enumerate(texts)
        ]}

    monkeypatch.setattr(router_sdk, "embedding", fake_embedding)


def test_router_adapter_no_prefix_and_identity(monkeypatch):
    calls: list[list[str]] = []
    _patch_router_embedding(monkeypatch, calls)
    a = RouterEmbedAdapter(model="openai/text-embedding-3-small")
    a.embed_passage(["原文一"])
    a.embed_query(["原文二"])
    # API 模型无前缀：原文直送
    assert calls == [["原文一"], ["原文二"]]
    assert a.model_identity == "api:openai/text-embedding-3-small"


def test_router_adapter_query_lru(monkeypatch):
    calls: list[list[str]] = []
    _patch_router_embedding(monkeypatch, calls)
    a = RouterEmbedAdapter(model="m", query_cache_size=8)
    v1 = a.embed_query(["同一个query"])
    v2 = a.embed_query(["同一个query"])  # 命中缓存，不再发请求
    assert v1 == v2
    assert len(calls) == 1
    # passage 不走缓存
    a.embed_passage(["同一个query"])
    assert len(calls) == 2


def test_router_adapter_requires_model():
    with pytest.raises(ValueError):
        RouterEmbedAdapter(model="")


# ── ProxyEmbedAdapter（共享模型档）──

def _patch_httpx(monkeypatch, calls: list[dict]):
    import httpx

    class _Resp:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append({"url": url, "json": json, "headers": headers})
        texts = json["input"]
        return _Resp({
            "object": "list",
            "model": "intfloat/multilingual-e5-large",
            "bladex_model_identity": "local:intfloat/multilingual-e5-large",
            "data": [{"index": i, "embedding": [float(len(t))]} for i, t in enumerate(texts)],
        })

    monkeypatch.setattr(httpx, "post", fake_post)


def test_proxy_adapter_input_type_and_identity(monkeypatch):
    calls: list[dict] = []
    _patch_httpx(monkeypatch, calls)
    a = ProxyEmbedAdapter("http://127.0.0.1:38080/")
    a.embed_passage(["p"])
    a.embed_query(["q"])
    assert calls[0]["url"] == "http://127.0.0.1:38080/v1/embeddings"
    assert calls[0]["json"]["input_type"] == "passage"
    assert calls[1]["json"]["input_type"] == "query"
    # 身份 = 远端的向量空间身份（与远端盖章一致）
    assert a.model_identity == "local:intfloat/multilingual-e5-large"


def test_proxy_adapter_sends_caller_headers(monkeypatch):
    """调用方归因：caller + pid 恒发；purpose 只在标注作用域内发（2026-08-09）。"""
    import os

    from bladex_proxy.embedding import embed_purpose

    calls: list[dict] = []
    _patch_httpx(monkeypatch, calls)
    a = ProxyEmbedAdapter("http://x", caller="consolidator")

    a.embed_passage(["p"])
    assert calls[0]["headers"]["X-BladeX-Caller"] == "consolidator"
    assert calls[0]["headers"]["X-BladeX-Caller-PID"] == str(os.getpid())
    # 进程名区分常驻 consolidator 与一次性工具（CLI / reembed 脚本同 role）
    assert calls[0]["headers"]["X-BladeX-Caller-Proc"]
    # 未标注用途时不发这个 header（远端记 "-"）
    assert "X-BladeX-Embed-Purpose" not in calls[0]["headers"]

    with embed_purpose("rebuild:batch7"):
        a.embed_passage(["p2"])
    assert calls[1]["headers"]["X-BladeX-Embed-Purpose"] == "rebuild:batch7"

    # 作用域退出后复位（不泄漏到后续调用）
    a.embed_passage(["p3"])
    assert "X-BladeX-Embed-Purpose" not in calls[2]["headers"]


def test_proxy_adapter_caller_defaults_to_role(monkeypatch):
    """build_embedder 传 role -> caller；"unknown" 留给绕过 adapter 的直连调用。"""
    calls: list[dict] = []
    _patch_httpx(monkeypatch, calls)
    ProxyEmbedAdapter("http://x").embed_passage(["p"])
    assert calls[0]["headers"]["X-BladeX-Caller"] == "consolidator"


def test_proxy_adapter_query_lru_and_auth(monkeypatch):
    calls: list[dict] = []
    _patch_httpx(monkeypatch, calls)
    a = ProxyEmbedAdapter("http://x", api_key="bladex-k", query_cache_size=4)
    a.embed_query(["q1"])
    a.embed_query(["q1"])
    assert len(calls) == 1
    assert calls[0]["headers"]["Authorization"] == "Bearer bladex-k"


# ── build_embedder 分发 ──

@pytest.fixture(autouse=True)
def _no_model_download(monkeypatch):
    """单测不触发真实下载/探网：ensure_local_model 直接放行。

    自动下载本身的行为由上面 test_ensure_local_model_* 系列单独覆盖。
    """
    from bladex_proxy import embedding as em
    monkeypatch.setattr(em, "ensure_local_model", lambda *a, **k: True)

def test_build_embedder_local_default():
    e = build_embedder(_fake_cfg(), role="proxy")
    assert isinstance(e, FastEmbedAdapter)
    assert e.model_name == DEFAULT_LOCAL_MODEL


def test_build_embedder_local_small_model():
    e = build_embedder(_fake_cfg(embed_model=_SMALL_MODEL))
    assert isinstance(e, FastEmbedAdapter)
    assert e.model_name == _SMALL_MODEL


def test_build_embedder_api():
    e = build_embedder(_fake_cfg(embed_backend="api", embed_model="openai/text-embedding-3-small"))
    assert isinstance(e, RouterEmbedAdapter)


def test_build_embedder_proxy_for_consolidator():
    e = build_embedder(_fake_cfg(embed_backend="proxy"), role="consolidator")
    assert isinstance(e, ProxyEmbedAdapter)


def test_build_embedder_proxy_backend_serves_local_on_proxy_side():
    """共享模型档：backend=proxy 时 proxy 侧加载本地模型（它就是被共享的那份），
    consolidator 侧走 HTTP —— 同一份配置、两种角色。"""
    cfg = _fake_cfg(embed_backend="proxy", embed_model=_E5_MODEL)
    server_side = build_embedder(cfg, role="proxy")
    client_side = build_embedder(cfg, role="consolidator")
    assert isinstance(server_side, FastEmbedAdapter)
    assert server_side.model_name == _E5_MODEL        # 保留配置模型，不被强制回默认
    assert isinstance(client_side, ProxyEmbedAdapter)


def test_proxy_adapter_wait_ready_retries_then_succeeds(monkeypatch):
    """启动竞态：run_proxy.sh 先拉 consolidator 再起 uvicorn ——
    首个 probe 必然失败，须重试而非退出。"""
    import httpx
    attempts = {"n": 0}

    class _Resp:
        def raise_for_status(self): return None
        def json(self):
            return {"model": "m", "bladex_model_identity": "local:m",
                    "data": [{"index": 0, "embedding": [0.1]}]}

    def flaky_post(url, json=None, headers=None, timeout=None):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise httpx.ConnectError("connection refused")
        return _Resp()

    monkeypatch.setattr(httpx, "post", flaky_post)
    a = ProxyEmbedAdapter("http://x")
    a.wait_ready(timeout_s=10, interval_s=0.01)   # 不抛 = 重试成功
    assert attempts["n"] == 3
    assert a.model_identity == "local:m"


def test_proxy_adapter_wait_ready_timeout_message(monkeypatch):
    """一直连不上 -> 超时报错，提示 proxy 是否在跑 + auth key 怎么配。"""
    import httpx

    def dead_post(url, json=None, headers=None, timeout=None):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "post", dead_post)
    a = ProxyEmbedAdapter("http://x")
    with pytest.raises(RuntimeError, match="proxy_key_env"):
        a.wait_ready(timeout_s=0.05, interval_s=0.01)


def test_build_embedder_sensitivity_forces_local():
    """敏感层开 + api 档 -> 强制回 local（记忆内容不外发，fail-closed）。"""
    e = build_embedder(_fake_cfg(embed_backend="api", embed_model="m", sens_enabled=True))
    assert isinstance(e, FastEmbedAdapter)


def test_embed_model_identity_matches_forced_fallbacks():
    assert embed_model_identity(_fake_cfg()) == f"local:{DEFAULT_LOCAL_MODEL}"
    assert embed_model_identity(_fake_cfg(embed_backend="api", embed_model="m")) == "api:m"
    # 敏感强制回本地时身份也跟着回落（与 build_embedder 一致，防身份/实体错位）
    assert embed_model_identity(
        _fake_cfg(embed_backend="api", embed_model="m", sens_enabled=True)
    ) == f"local:{DEFAULT_LOCAL_MODEL}"
    assert embed_model_identity(_fake_cfg(embed_backend="proxy"), role="proxy") \
        == f"local:{DEFAULT_LOCAL_MODEL}"


# ── validate_embed_sensitivity ──

def test_validate_sensitivity_strict_refuses_api():
    cfg = _fake_cfg(embed_backend="api", embed_model="m", sens_enabled=True, route_strict=True)
    with pytest.raises(RuntimeError):
        validate_embed_sensitivity(cfg)


def test_validate_sensitivity_non_strict_warns_only():
    cfg = _fake_cfg(embed_backend="api", embed_model="m", sens_enabled=True, route_strict=False)
    validate_embed_sensitivity(cfg)  # 不抛（运行时 build_embedder 强制回本地）


def test_validate_sensitivity_disabled_noop():
    validate_embed_sensitivity(_fake_cfg(embed_backend="api", embed_model="m"))


# ── effective_thresholds ──

def test_thresholds_env_explicit_wins(monkeypatch):
    monkeypatch.setenv("BLADEX_NOVELTY_THRESHOLD", "0.90")
    cfg = _fake_cfg(embed_model=_SMALL_MODEL, novelty_threshold=0.90)
    out = effective_thresholds(cfg)
    assert out["novelty"] == 0.90  # 显式 env 压过 profile
    assert out["semantic"] == 0.85


def test_thresholds_profile_defaults_for_small_model(monkeypatch):
    """小模型档取 profile 的保守默认。

    novelty 2026-08-07 从 0.95 改到 0.98：M2 改的是**语义**（去重线 → 只拦完全
    重发），profile 表里那份 0.95 是 07-26 按旧语义标定的，却因为
    `effective_thresholds` 里「profile > flags」的优先级**静默压过** flags ——
    生产机上 M2 的 0.98 因此从未生效。详见
    `test_m2_adjudication.py::test_profile_novelty_matches_flags`。
    """
    monkeypatch.delenv("BLADEX_NOVELTY_THRESHOLD", raising=False)
    cfg = _fake_cfg(embed_model=_SMALL_MODEL)
    out = effective_thresholds(cfg)
    assert out == {"novelty": _FLAG_NOVELTY, "semantic": 0.85, "digestion": 0.85}


def test_thresholds_unknown_api_model_keeps_defaults(monkeypatch):
    monkeypatch.delenv("BLADEX_NOVELTY_THRESHOLD", raising=False)
    cfg = _fake_cfg(embed_backend="api", embed_model="some/unknown-model")
    out = effective_thresholds(cfg)
    assert out["novelty"] == _FLAG_NOVELTY  # 无 profile -> 回落 cfg（即 flags）


# ── Memory Index model_id 不变量 ──

def test_index_embed_model_stamp_and_refuse_mix():
    with tempfile.TemporaryDirectory() as tmp:
        index = MemoryIndex(f"{tmp}/index", embedder=None, embed_model_id="local:model-A")
        index.open()
        index.close()
        # 同模型重开 -> OK
        p2b = MemoryIndex(f"{tmp}/index", embedder=None, embed_model_id="local:model-A")
        p2b.open()
        p2b.close()
        # 换模型写模式 -> 拒启动
        p2c = MemoryIndex(f"{tmp}/index", embedder=None, embed_model_id="local:model-B")
        with pytest.raises(RuntimeError, match="reembed_index"):
            p2c.open()


def test_index_embed_model_mismatch_readonly_disables_search():
    with tempfile.TemporaryDirectory() as tmp:
        index = MemoryIndex(f"{tmp}/index", embedder=None, embed_model_id="local:model-A")
        index.open()
        index.close()
        legacy = _LegacyEmbedder()
        p2r = MemoryIndex(f"{tmp}/index", embedder=legacy, read_only=True,
                        embed_model_id="local:model-B")
        p2r.open()  # 不抛：proxy 本体不拒启
        assert p2r._embedder is None  # 向量检索禁用 -> search 降级空 -> 注入退硬规则
        assert p2r.search("q") == []
        p2r.close()


def test_index_embed_model_empty_does_not_disable_search():
    """🔴 2026-08-25 live 病例 6：**"还不知道"不等于"不一致"**。

    IPC adapter 在握手前 `model_identity` 是空串，`server.py` 恰好在那一刻把它
    当 `embed_model_id` 传进来 ⇒ 空串 vs 已盖章模型被判"不一致" ⇒ 只读端
    `_embedder=None` ⇒ **整个 proxy 进程终生禁用向量检索**（实测 6/6 轮
    `facts_count=0`，日志只有一行 warning，用户侧表现为"记忆全没了"）。

    这是"注入面空转"家族的第 N 次：**一条静默的负面判定，代价是整个记忆面**。
    对偶修复在 `embed_client._probe_identity`（让身份在被查询前就成立）——
    两道都要有：握手可能因为模块没起而失败，那时也不该误判成不一致。
    """
    with tempfile.TemporaryDirectory() as tmp:
        index = MemoryIndex(f"{tmp}/index", embedder=None, embed_model_id="local:model-A")
        index.open()
        index.close()
        legacy = _LegacyEmbedder()
        p2r = MemoryIndex(f"{tmp}/index", embedder=legacy, read_only=True,
                          embed_model_id="")          # ← 身份未知，不是"另一个模型"
        p2r.open()
        assert p2r._embedder is legacy, "空身份被当成不一致 ⇒ 记忆面整体空转"
        p2r.close()


def test_index_embed_model_none_skips_check():
    """embed_model_id=None（单测/legacy/迁移脚本）不校验 —— 零回归。"""
    with tempfile.TemporaryDirectory() as tmp:
        index = MemoryIndex(f"{tmp}/index", embedder=None, embed_model_id="local:model-A")
        index.open()
        index.close()
        p2b = MemoryIndex(f"{tmp}/index", embedder=None)
        p2b.open()
        p2b.close()


# ── routing.toml [embedding] 结构化配置（T9 配置化管理）──

def _cfg_with_toml(toml_text: str, **cfg_over: Any) -> Any:
    """把 toml 文本装进真 RoutingConfig，挂到 fake cfg 上。"""
    import tempfile as _tf

    from bladex_proxy.routing_config import RoutingConfig
    with _tf.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
        f.write(toml_text)
        path = f.name
    rc = RoutingConfig.from_toml(path)
    cfg = _fake_cfg(**cfg_over)
    # 保留敏感度 stub，附加真 embedding 段
    sens = cfg.routing_config.sensitivity_config
    cfg.routing_config = SimpleNamespace(
        sensitivity_config=sens, embedding=rc.embedding,
    )
    return cfg


def test_toml_embedding_section_parsed(monkeypatch):
    """[embedding] 段生效：backend/model/api_key_env 密钥引 env。"""
    from bladex_proxy.embedding import resolve_embed_settings
    monkeypatch.delenv("BLADEX_EMBED_BACKEND", raising=False)
    monkeypatch.setenv("TEST_EMBED_KEY", "sk-from-env")
    cfg = _cfg_with_toml(
        '[embedding]\nbackend = "api"\nmodel = "openai/text-embedding-3-small"\n'
        'api_base = "https://api.example.com/v1"\napi_key_env = "TEST_EMBED_KEY"\n'
    )
    s = resolve_embed_settings(cfg)
    assert s.backend == "api"
    assert s.model == "openai/text-embedding-3-small"
    assert s.api_key == "sk-from-env"  # 密钥只引 env，不入 toml


def test_env_overrides_toml(monkeypatch):
    """显式 env 压过 toml（回滚通道，与 distill 同优先级序）。"""
    from bladex_proxy.embedding import resolve_embed_settings
    monkeypatch.setenv("BLADEX_EMBED_BACKEND", "local")
    cfg = _cfg_with_toml('[embedding]\nbackend = "api"\nmodel = "m"\n',
                         embed_backend="local")
    assert resolve_embed_settings(cfg).backend == "local"


def test_toml_profile_overrides_thresholds(monkeypatch):
    """[embedding.profiles."model"] 为 api 模型声明阈值；calibrated=true 生效。"""
    for k in ("BLADEX_NOVELTY_THRESHOLD", "BLADEX_SEMANTIC_THRESHOLD",
              "BLADEX_DIGESTION_THRESHOLD", "BLADEX_EMBED_BACKEND",
              "BLADEX_EMBED_MODEL"):
        monkeypatch.delenv(k, raising=False)
    cfg = _cfg_with_toml(
        '[embedding]\nbackend = "api"\nmodel = "openai/doubao-embedding"\n'
        '[embedding.profiles."openai/doubao-embedding"]\n'
        'novelty = 0.93\nsemantic = 0.83\ndigestion = 0.83\ncalibrated = true\n',
        embed_backend="api", embed_model="openai/doubao-embedding",
    )
    out = effective_thresholds(cfg)
    assert out == {"novelty": 0.93, "semantic": 0.83, "digestion": 0.83}


def test_toml_absent_zero_regression():
    """无 [embedding] 段 = 现状（local + e5-large）。"""
    from bladex_proxy.embedding import resolve_embed_settings
    cfg = _cfg_with_toml("")  # 空 toml
    s = resolve_embed_settings(cfg)
    assert s.backend == "local"
    assert (s.model or DEFAULT_LOCAL_MODEL) == DEFAULT_LOCAL_MODEL


# ── 默认模型 + 七档 profile + 自动下载（开源默认：英语优先、开箱即用）──

def test_default_model_is_lightweight_english():
    """默认 = bge-small-en-v1.5（384d/0.07GB）：全球开发者开箱即用，不先下 2.24GB。"""
    assert DEFAULT_LOCAL_MODEL == "BAAI/bge-small-en-v1.5"
    prof = MODEL_PROFILES[DEFAULT_LOCAL_MODEL]
    assert prof["dim"] == 384
    assert prof["size_gb"] < 0.1


def test_recommended_profiles_present_and_described():
    """七档推荐模板齐备，且都带能力说明（dim/size/languages/note）。"""
    expected = {
        "BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5", "BAAI/bge-large-en-v1.5",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "jinaai/jina-embeddings-v2-base-zh",
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "intfloat/multilingual-e5-large",
    }
    assert expected <= set(MODEL_PROFILES)
    for name in expected:
        p = MODEL_PROFILES[name]
        assert p["dim"] > 0 and p["size_gb"] > 0
        assert p["languages"] and p["note"]
    # 只有 e5-large 经真实流量标定
    assert MODEL_PROFILES["intfloat/multilingual-e5-large"]["calibrated"] is True


def test_recommended_models_supported_by_fastembed():
    """推荐档必须都在 fastembed 白名单内（防再次写出不存在的 e5-small）。"""
    from fastembed import TextEmbedding
    supported = {m["model"] for m in TextEmbedding.list_supported_models()}
    assert set(MODEL_PROFILES) <= supported


def test_cache_detect_uses_real_hf_repo(tmp_path):
    """缓存检测必须按 fastembed 的真实下载仓库（qdrant/*-onnx），非逻辑模型名。

    回归锁：按逻辑名拼目录会永远判"未缓存" -> 每次启动探网，
    离线机器上误判"缺失"而拒启动（2026-07-26 实测到的真 bug）。
    """
    from bladex_proxy.embedding import _hf_repos_for, _model_cached
    repos = _hf_repos_for("intfloat/multilingual-e5-large")
    assert repos[0] == "qdrant/multilingual-e5-large-onnx"
    d = tmp_path / "models--qdrant--multilingual-e5-large-onnx"
    d.mkdir(parents=True)
    (d / "model.onnx").write_text("x")
    assert _model_cached("intfloat/multilingual-e5-large", str(tmp_path)) is True


def test_ensure_local_model_skips_when_cached(monkeypatch, tmp_path):
    """已缓存 -> 不探网、不下载（正常启动路径零网络开销）。"""
    from bladex_proxy import embedding as em
    d = tmp_path / "models--qdrant--bge-small-en-v1.5-onnx-q"
    d.mkdir(parents=True)
    (d / "model.onnx").write_text("x")
    monkeypatch.setattr(em, "_endpoint_reachable",
                        lambda url: pytest.fail("should not probe network"))
    assert _REAL_ENSURE("BAAI/bge-small-en-v1.5", str(tmp_path)) is True


def test_ensure_local_model_rejects_unsupported(tmp_path):
    """白名单外模型 -> 报错并列出推荐档（不静默回落）。"""
    with pytest.raises(RuntimeError, match="not supported by fastembed"):
        _REAL_ENSURE("intfloat/multilingual-e5-small", str(tmp_path))


def test_ensure_local_model_falls_back_to_mirror(monkeypatch, tmp_path):
    """HF 主站不可达 -> 自动设 HF_ENDPOINT 到镜像再下（减少用户操作）。"""
    from bladex_proxy import embedding as em
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.setattr(em, "_endpoint_reachable",
                        lambda url: "hf-mirror.com" in url)
    loaded: dict[str, Any] = {}

    class _FakeTE:
        def __init__(self, model_name=None, cache_dir=None):
            loaded["model"] = model_name
            loaded["endpoint"] = os.environ.get("HF_ENDPOINT")

        @staticmethod
        def list_supported_models():
            return [{"model": "BAAI/bge-small-en-v1.5"}]

    import fastembed
    monkeypatch.setattr(fastembed, "TextEmbedding", _FakeTE)
    assert _REAL_ENSURE("BAAI/bge-small-en-v1.5", str(tmp_path)) is True
    assert loaded["endpoint"] == "https://hf-mirror.com"


def test_ensure_local_model_respects_user_endpoint(monkeypatch, tmp_path):
    """用户显式设了 HF_ENDPOINT -> 完全尊重，不做回退猜测。"""
    from bladex_proxy import embedding as em
    monkeypatch.setenv("HF_ENDPOINT", "https://my-mirror.internal")
    monkeypatch.setattr(em, "_endpoint_reachable",
                        lambda url: pytest.fail("should not probe when user set HF_ENDPOINT"))
    seen: dict[str, Any] = {}

    class _FakeTE:
        def __init__(self, model_name=None, cache_dir=None):
            seen["endpoint"] = os.environ.get("HF_ENDPOINT")

        @staticmethod
        def list_supported_models():
            return [{"model": "BAAI/bge-small-en-v1.5"}]

    import fastembed
    monkeypatch.setattr(fastembed, "TextEmbedding", _FakeTE)
    _REAL_ENSURE("BAAI/bge-small-en-v1.5", str(tmp_path))
    assert seen["endpoint"] == "https://my-mirror.internal"


def test_ensure_local_model_all_endpoints_down(monkeypatch, tmp_path):
    """主站 + 镜像全不可达 -> 明确报错（含如何自救的提示）。"""
    from bladex_proxy import embedding as em
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.setattr(em, "_endpoint_reachable", lambda url: False)
    with pytest.raises(RuntimeError, match="HF_ENDPOINT"):
        _REAL_ENSURE("BAAI/bge-small-en-v1.5", str(tmp_path))


def test_stamp_probe_never_opens_rocksdb(tmp_path):
    """回归锁（2026-07-26 事故）：启动前探测**不得打开 live Memory Index RocksDB**。

    事故：用 Options(raw_mode=True) 开库 -> rocksdict 把库内 sidecar
    `rocksdict-config.json` 改写成 raw_mode=true -> MemoryIndex 默认打开时
    comparator 不匹配（BytewiseComparator vs rocksdict）-> consolidator 拒启动。
    数据未损但库打不开。故探测只读纯文本侧车，绝不碰 RocksDB。
    """
    import hashlib

    from bladex_proxy.storage.memory_index import MemoryIndex
    index_dir = tmp_path / "index"
    index = MemoryIndex(str(index_dir), embedder=None,
                   embed_model_id="local:intfloat/multilingual-e5-large")
    index.open()
    index.close()
    cfg_json = index_dir / "meta_rocksdb" / "rocksdict-config.json"
    before = hashlib.md5(cfg_json.read_bytes()).hexdigest()

    from bladex_proxy.embedding import _stored_local_model
    got = _stored_local_model(SimpleNamespace(index_path=str(index_dir)))

    assert got == "intfloat/multilingual-e5-large"          # 探测到章
    assert cfg_json.read_bytes()                              # 文件仍在
    assert hashlib.md5(cfg_json.read_bytes()).hexdigest() == before  # 且未被改写
    # 侧车存在 = 探测无需开库
    assert (index_dir / "embed_model_id").read_text() == "local:intfloat/multilingual-e5-large"
    # 库仍能被 MemoryIndex 正常打开（事故的直接症状）
    p2b = MemoryIndex(str(index_dir), embedder=None,
                    embed_model_id="local:intfloat/multilingual-e5-large")
    p2b.open()
    p2b.close()


def test_stamp_probe_missing_store_is_safe(tmp_path):
    """无 Memory Index 库 / 无侧车 -> 返回空串，不抛（全新安装走新默认）。"""
    from bladex_proxy.embedding import _stored_local_model
    assert _stored_local_model(SimpleNamespace(index_path=str(tmp_path / "nope"))) == ""
    assert _stored_local_model(SimpleNamespace(index_path="")) == ""


def test_existing_index_stamp_survives_default_change(monkeypatch):
    """默认值变更保护：已有 Memory Index 盖章 e5-large + 未显式配置 -> 沿用库内模型。

    否则升级 BladeX 会让 consolidator 因向量空间不一致拒启动。
    """
    from bladex_proxy import embedding as em
    from bladex_proxy.embedding import resolve_embed_settings
    monkeypatch.delenv("BLADEX_EMBED_MODEL", raising=False)
    monkeypatch.setattr(em, "_stored_local_model",
                        lambda cfg: "intfloat/multilingual-e5-large")
    s = resolve_embed_settings(_fake_cfg())
    assert s.model == "intfloat/multilingual-e5-large"


def test_fresh_install_uses_new_default(monkeypatch):
    """全新安装（Memory Index 无盖章）-> 用新默认 bge-small-en-v1.5。"""
    from bladex_proxy import embedding as em
    from bladex_proxy.embedding import resolve_embed_settings
    monkeypatch.delenv("BLADEX_EMBED_MODEL", raising=False)
    monkeypatch.setattr(em, "_stored_local_model", lambda cfg: "")
    s = resolve_embed_settings(_fake_cfg())
    assert (s.model or DEFAULT_LOCAL_MODEL) == "BAAI/bge-small-en-v1.5"


# ── /v1/embeddings input_type 扩展 ──

def test_embeddings_endpoint_input_type():
    from bladex_proxy.config import ProxyConfig
    from bladex_proxy.server import create_app
    from fastapi.testclient import TestClient

    tmpdir = tempfile.mkdtemp()
    cfg = ProxyConfig(upstream_model="openai/test", upstream_api_key="sk-fake",
                      rocksdb_path=f"{tmpdir}/rocksdb")
    app = create_app(cfg)
    client = TestClient(app)
    client.__enter__()
    try:
        emb = _LegacyEmbedder()
        app.state.embedder = emb
        r1 = client.post("/v1/embeddings", json={"input": "hello"})
        assert r1.status_code == 200
        assert "bladex_model_identity" in r1.json()
        r2 = client.post("/v1/embeddings", json={"input": "hello", "input_type": "query"})
        assert r2.status_code == 200
        # 旧接口 fake 走 compat：passage=原文、query="query: " 前缀（历史行为逐字）
        assert "hello" in emb.seen
        assert "query: hello" in emb.seen
    finally:
        client.__exit__(None, None, None)


# ── /v1/embeddings 调用方归因日志（2026-08-09）──

def _capture_embed_log(monkeypatch, window_s: float = 60.0):
    """把 EmbedCallLog 的日志接出来（返回 (log, events)）。"""
    from bladex_proxy import embedding as em

    events: list[tuple[str, dict]] = []

    class _Log:
        def info(self, event, **kw):
            events.append((event, kw))

        def __getattr__(self, _name):  # debug/warning/error 吞掉
            return lambda *a, **k: None

    monkeypatch.setattr(em, "logger", _Log())
    return em.EmbedCallLog(window_s=window_s), events


def test_embed_call_log_first_seen_then_rollup(monkeypatch):
    """首次即报（谁在调一眼可见）+ 周期汇总（不逐条刷屏）。

    起因：日志里只有 `auth_ok label=hermes-default` 重复刷屏 —— 共享模型档下
    consolidator 与 agent 对话用同一把 key，既看不出谁在调，也压不住量。
    """
    log, events = _capture_embed_log(monkeypatch, window_s=0.0)

    def _rec(**kw):
        base = {"caller": "consolidator", "caller_pid": "4242",
                "purpose": "rebuild:batch3", "input_type": "passage",
                "count": 2, "chars": 5, "embed_ms": 12.3}
        base.update(kw)
        log.record(**base)

    _rec()
    seen = [kw for ev, kw in events if ev == "embed_caller_seen"]
    assert len(seen) == 1
    assert seen[0]["caller"] == "consolidator"
    assert seen[0]["caller_pid"] == "4242"
    assert seen[0]["purpose"] == "rebuild:batch3"
    assert seen[0]["input_type"] == "passage"

    # 同组合再来不重复报（只进汇总）；window_s=0 → 每次都吐一条 summary
    events.clear()
    _rec(count=1, chars=3, embed_ms=4.0)
    assert not [ev for ev, _ in events if ev == "embed_caller_seen"]
    summary = [kw for ev, kw in events if ev == "embed_requests_summary"]
    assert len(summary) == 1
    row = summary[0]["by"][0]
    assert row["caller"] == "consolidator"
    assert row["purpose"] == "rebuild:batch3"
    assert row["requests"] == 1 and row["texts"] == 1 and row["chars"] == 3

    # 新用途 = 新组合 → 再报一次首次（阶段切换可见）
    events.clear()
    _rec(purpose="export_sync")
    assert [kw["purpose"] for ev, kw in events if ev == "embed_caller_seen"] == ["export_sync"]


def test_embeddings_endpoint_attributes_caller(monkeypatch):
    """端点把 header 归因喂给 EmbedCallLog；无 header 时退 User-Agent/unknown。"""
    from bladex_proxy.config import ProxyConfig
    from bladex_proxy.server import create_app
    from fastapi.testclient import TestClient

    log, events = _capture_embed_log(monkeypatch, window_s=60.0)

    tmpdir = tempfile.mkdtemp()
    cfg = ProxyConfig(upstream_model="openai/test", upstream_api_key="sk-fake",
                      rocksdb_path=f"{tmpdir}/rocksdb")
    app = create_app(cfg)
    client = TestClient(app)
    client.__enter__()
    try:
        app.state.embedder = _LegacyEmbedder()
        app.state.embed_call_log = log
        r = client.post(
            "/v1/embeddings",
            json={"input": ["abc", "de"], "input_type": "query"},
            headers={
                "X-BladeX-Caller": "consolidator",
                "X-BladeX-Caller-PID": "4242",
                "X-BladeX-Caller-Proc": "bladex-consolidator",
                "X-BladeX-Embed-Purpose": "rebuild:batch3",
            },
        )
        assert r.status_code == 200
        seen = [kw for ev, kw in events if ev == "embed_caller_seen"]
        assert len(seen) == 1, f"expected one embed_caller_seen, got {events}"
        got = seen[0]
        assert got["caller"] == "consolidator"
        assert got["caller_pid"] == "4242"
        assert got["caller_proc"] == "bladex-consolidator"
        assert got["purpose"] == "rebuild:batch3"
        assert got["input_type"] == "query"
        assert got["count"] == 2
        assert got["chars"] == 5
        assert isinstance(got["embed_ms"], float)

        # 绕过 adapter 的直连调用：caller 退 User-Agent，purpose 记 "-"
        events.clear()
        client.post("/v1/embeddings", json={"input": "x"})
        got2 = [kw for ev, kw in events if ev == "embed_caller_seen"][0]
        assert got2["purpose"] == "-"
        assert got2["caller"] != "consolidator"
    finally:
        client.__exit__(None, None, None)
