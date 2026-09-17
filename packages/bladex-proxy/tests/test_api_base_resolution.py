"""api_base 解析与校验（2026-07-29 事故驱动）。

事故：用户为"改一处即可切换上游"，把 8 个模型的 `api_base` 从字面 URL 改成
`"BLADEX_UPSTREAM_API_BASE"`，以为会像 `api_key_env` 那样被解析。当时不会——
Router 拿到这个字符串去建连，httpx 抛 `Connection error.`。

后果：8 个 openai/* 模型全部不可用，路由每轮挨个 failover，最后总退到唯一没被
改到的 `anthropic/doubao-seed-2.0-lite`（硬编码 URL）。表面"系统能用"，
实际主力池全挂 5 小时，且错误信息完全不指向配置。

修复：新增 `api_base_env`（与 `api_key_env` 对称）+ 启动时校验 api_base 必须是
http/https URL，非法即**拒绝启动**并给出精确诊断。
"""

from __future__ import annotations

import pathlib
import tempfile

import pytest
from bladex_proxy.routing_config import RoutingConfig, RoutingConfigError

_BASE = '[[models]]\nname = "openai/x"\ntier = "weak"\n'


def _load(toml: str) -> RoutingConfig:
    p = pathlib.Path(tempfile.mkdtemp()) / "routing.toml"
    p.write_text(toml)
    return RoutingConfig.from_toml(p)


# ── 事故本体 ──


def test_env_var_name_in_api_base_refuses_startup():
    """🔴 事故形状：把环境变量名写进 api_base → 必须拒绝启动，不得静默透传。"""
    with pytest.raises(RoutingConfigError) as ei:
        _load(_BASE + 'api_base = "BLADEX_UPSTREAM_API_BASE"\n')
    msg = str(ei.value)
    # 必须点名是哪个模型
    assert 'name="openai/x"' in msg
    # 必须给出实际值与具体问题
    assert "BLADEX_UPSTREAM_API_BASE" in msg
    assert "http://" in msg and "https://" in msg
    # 必须给出两种正确写法，而不是"你是不是想…"式的猜测
    assert "api_base_env" in msg
    assert "二选一" in msg


def test_error_explains_field_semantics_difference():
    """值形如环境变量名时，额外说明 api_base 与 api_key_env 的语义差别。"""
    with pytest.raises(RoutingConfigError) as ei:
        _load(_BASE + 'api_base = "SOME_UPSTREAM_BASE"\n')
    msg = str(ei.value)
    assert "api_key_env" in msg
    assert "不做环境变量解析" in msg


# ── 两种正确写法 ──


def test_literal_url_accepted():
    cfg = _load(_BASE + 'api_base = "https://ark.cn-beijing.volces.com/api/coding/v3"\n')
    assert cfg.models[0].api_base == "https://ark.cn-beijing.volces.com/api/coding/v3"


def test_api_base_env_resolved(monkeypatch):
    monkeypatch.setenv("MY_UPSTREAM_BASE", "https://ark.example/v3")
    cfg = _load(_BASE + 'api_base_env = "MY_UPSTREAM_BASE"\n')
    assert cfg.models[0].api_base == "https://ark.example/v3"


def test_api_base_env_wins_over_literal(monkeypatch):
    monkeypatch.setenv("MY_UPSTREAM_BASE", "https://from-env/v3")
    cfg = _load(_BASE
                + 'api_base = "https://from-literal/v1"\n'
                + 'api_base_env = "MY_UPSTREAM_BASE"\n')
    assert cfg.models[0].api_base == "https://from-env/v3"


def test_both_empty_is_valid():
    """都不配 = 用 Router 默认端点，合法，不该拦。"""
    cfg = _load(_BASE)
    assert cfg.models[0].api_base == ""


def test_http_scheme_accepted():
    """本地 ollama 之类走 http，不该被拦。"""
    cfg = _load(_BASE + 'api_base = "http://localhost:11434/v1"\n')
    assert cfg.models[0].api_base == "http://localhost:11434/v1"


# ── api_base_env 指向缺失变量 ──


def test_missing_env_var_refuses_startup(monkeypatch):
    monkeypatch.delenv("DEFINITELY_NOT_SET", raising=False)
    with pytest.raises(RoutingConfigError) as ei:
        _load(_BASE + 'api_base_env = "DEFINITELY_NOT_SET"\n')
    msg = str(ei.value)
    assert "DEFINITELY_NOT_SET" in msg
    assert "未设置或为空" in msg
    assert "config/.env" in msg


def test_empty_env_var_refuses_startup(monkeypatch):
    monkeypatch.setenv("EMPTY_BASE", "")
    with pytest.raises(RoutingConfigError):
        _load(_BASE + 'api_base_env = "EMPTY_BASE"\n')


# ── embedding 段同规则 ──


def test_embedding_api_base_validated():
    with pytest.raises(RoutingConfigError) as ei:
        _load(_BASE + 'api_base = "https://ok/v1"\n\n'
              '[embedding]\nbackend = "api"\napi_base = "BLADEX_EMBED_BASE"\n')
    assert "[embedding]" in str(ei.value)


def test_embedding_api_base_env_resolved(monkeypatch):
    monkeypatch.setenv("MY_EMBED_BASE", "https://embed.example/v1")
    cfg = _load(_BASE + 'api_base = "https://ok/v1"\n\n'
                '[embedding]\nbackend = "api"\napi_base_env = "MY_EMBED_BASE"\n')
    assert cfg.embedding is not None
    assert cfg.embedding.api_base == "https://embed.example/v1"


# ── 回归：真实配置文件仍可加载 ──


def test_shipped_example_config_loads(monkeypatch):
    """`routing.toml.example` 同时示范了两种写法，必须能加载。"""
    monkeypatch.setenv("BLADEX_UPSTREAM_API_BASE", "https://ark.example/api/coding/v3")
    example = pathlib.Path("config/routing.toml.example")
    if not example.is_file():
        pytest.skip("example config not present in this tree")
    cfg = RoutingConfig.from_toml(example)
    bases = {m.api_base for m in cfg.models if m.api_base}
    assert all(b.startswith(("http://", "https://")) for b in bases)
    assert "https://ark.example/api/coding/v3" in bases, "写法②未被解析"
