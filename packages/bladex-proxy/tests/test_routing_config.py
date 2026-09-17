"""结构化路由配置解析（Phase 1 schema v2）。

验：TOML → RoutingConfig（模型/上游/三策略）；能力词表校验；
api_key_env 缺失 → warning + 标不可用；策略引用未知模型 → warning 跳过。
"""

from __future__ import annotations

import pytest
from bladex_core.routing import (
    AgentStrategyData,
    FilterStrategyData,
    ModelCandidate,
    MultimodalStrategyData,
    RequestStrategyData,
)
from bladex_proxy.routing_config import RoutingConfig, RoutingConfigError

_SAMPLE_TOML = """
[[models]]
name = "doubao-seed-2.0-pro"
api_base = "https://ark/v3"
api_key_env = "ARK_API_KEY"
capabilities = ["text", "vision"]
tier = "strong"

[[models]]
name = "glm-5.2"
api_base = "https://glm/v1"
api_key_env = "GLM_API_KEY"
capabilities = ["text", "code"]
tier = "medium"

[[models]]
name = "deepseek-flash"
api_base = "https://deep/v1"
api_key_env = "DEEPSEEK_API_KEY"
capabilities = ["text"]
tier = "weak"

[upstream]
models = ["glm-5.2", "deepseek-flash"]

[strategies.agent]
enabled = true
[strategies.agent.map]
"hermes:accept" = ["doubao-seed-2.0-pro", "glm-5.2"]
"hermes" = ["glm-5.2"]

[strategies.multimodal]
enabled = false
[strategies.multimodal.map]
vision = ["doubao-seed-2.0-pro"]

[strategies.filter]
enabled = false
judge_model = "deepseek-flash"
no_think = true
max_prompt_chars = 2000
default_tier = "weak"
[strategies.filter.candidates]
weak = ["deepseek-flash"]
medium = ["glm-5.2"]
strong = ["doubao-seed-2.0-pro"]

[strategies.request]
enabled = false
[strategies.request.agent_overrides]
"claude-code" = true
"hermes" = false
"""


def _write_toml(tmp_path, text: str, name: str = "routing.toml"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_routing_config_parse(tmp_path, monkeypatch):
    """加载样例 routing.toml → 结构化 RoutingConfig，模型/上游/策略正确。"""
    monkeypatch.setenv("ARK_API_KEY", "ark-secret")
    monkeypatch.setenv("GLM_API_KEY", "glm-secret")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deep-secret")
    p = _write_toml(tmp_path, _SAMPLE_TOML)
    cfg = RoutingConfig.from_toml(p)

    assert len(cfg.models) == 3
    pro = next(m for m in cfg.models if m.name == "doubao-seed-2.0-pro")
    assert pro.capabilities == ["text", "vision"]
    assert pro.tier == "strong"
    assert pro.api_key == "ark-secret"
    assert pro.available is True

    # upstream
    assert cfg.upstream.models == ["glm-5.2", "deepseek-flash"]
    assert cfg.has_upstream() is True

    # agent 策略
    assert cfg.strategies.agent.enabled is True
    assert "hermes:accept" in cfg.strategies.agent.map
    assert cfg.strategies.agent.map["hermes:accept"] == ["doubao-seed-2.0-pro", "glm-5.2"]

    # multimodal 策略（默认关）
    assert cfg.strategies.multimodal.enabled is False

    # filter 策略（默认关）
    assert cfg.strategies.filter.enabled is False
    assert cfg.strategies.filter.judge_model == "deepseek-flash"
    assert cfg.strategies.filter.candidates["medium"] == ["glm-5.2"]

    # request 策略 per-agent 覆盖（ADR-0013 增补 2026-07-25）
    assert cfg.strategies.request.enabled is False
    assert cfg.strategies.request.agent_overrides == {"claude-code": True, "hermes": False}


def test_all_strategies_default_off(tmp_path):
    """只配 models + upstream，不配策略 → 三策略全关。"""
    toml = """
[[models]]
name = "m1"
tier = "weak"

[upstream]
models = ["m1"]
"""
    cfg = RoutingConfig.from_toml(_write_toml(tmp_path, toml))
    assert cfg.strategies.agent.enabled is False
    assert cfg.strategies.multimodal.enabled is False
    assert cfg.strategies.filter.enabled is False


def test_produces_core_data(tmp_path, monkeypatch):
    """RoutingConfig 产 core 侧数据（ModelCandidate dict + 策略 Data）。"""
    monkeypatch.setenv("K", "secret")
    toml = """
[[models]]
name = "pro"
api_key_env = "K"
capabilities = ["text", "vision"]
tier = "strong"

[[models]]
name = "mini"
api_key_env = "K"
tier = "weak"

[upstream]
models = ["pro", "mini"]

[strategies.agent]
enabled = true
[strategies.agent.map]
hermes = ["pro"]

[strategies.request]
enabled = true
[strategies.request.agent_overrides]
"claude-code" = true
"hermes" = false
"""
    cfg = RoutingConfig.from_toml(_write_toml(tmp_path, toml))

    models = cfg.model_candidates()
    assert all(isinstance(c, ModelCandidate) for c in models.values())
    assert "pro" in models and "mini" in models
    assert models["pro"].capabilities == ["text", "vision"]

    agent = cfg.agent_strategy()
    assert isinstance(agent, AgentStrategyData)
    assert agent.enabled is True
    assert agent.map["hermes"] == ["pro"]

    mm = cfg.multimodal_strategy()
    assert isinstance(mm, MultimodalStrategyData)
    assert mm.enabled is False

    flt = cfg.filter_strategy()
    assert isinstance(flt, FilterStrategyData)
    assert flt.enabled is False

    # request 策略 per-agent 覆盖透传到 core（ADR-0013 增补 2026-07-25）
    req = cfg.request_strategy()
    assert isinstance(req, RequestStrategyData)
    assert req.enabled is True
    assert req.agent_overrides == {"claude-code": True, "hermes": False}


def test_api_key_env_missing_warns(tmp_path, monkeypatch):
    """api_key_env 引用的 env 缺失 → 标不可用，不崩。"""
    monkeypatch.delenv("MISSING_KEY", raising=False)
    toml = """
[[models]]
name = "m1"
api_key_env = "MISSING_KEY"
tier = "weak"

[upstream]
models = ["m1"]
"""
    cfg = RoutingConfig.from_toml(_write_toml(tmp_path, toml))
    assert cfg.models[0].available is False
    assert cfg.has_models() is False


def test_api_key_env_empty_uses_default_auth(tmp_path):
    """api_key_env 为空 → 视为用 Router 默认鉴权，available=True。"""
    toml = """
[[models]]
name = "m1"
api_base = "https://x/v1"
tier = "weak"

[upstream]
models = ["m1"]
"""
    cfg = RoutingConfig.from_toml(_write_toml(tmp_path, toml))
    assert cfg.models[0].available is True
    assert cfg.models[0].api_key == ""


def test_agent_unknown_model_refuses_startup(tmp_path, monkeypatch):
    """agent 引用了模型池里没有的模型名 ⇒ **拒绝启动**（MQ-L58，2026-09-08 改）。

    🔴 **本条推翻了它自己的旧版**（原名 `test_agent_unknown_model_warns`，
    断言的是"warning 并跳过该名，map 保留原样"）。旧行为在 live 上的代价：
    `"codex" = ["anthropic/glm-5.3-flash"]` 写错一个字，启动打了一条结构化 warning
    之后**请求照常成功** —— 静态策略整个不命中、退回 sticky 到 weak 档、账本族被 gate，
    一次剧本③ 全程作废，而那条 warning 在 10 个以上历史日志里连打了好几天没人看。

    缺的不是告警是**严重性**：自包含的配置错误应当在**加载期**响亮失败
    （同仓先例：`IdentityRegistry.from_toml` 的 dangling ref 就是拒启动）。
    完整用例见 `tests/unit/test_routing_dangling_model_ref.py`。
    """
    monkeypatch.setenv("K", "v")
    toml = """
[[models]]
name = "m1"
api_key_env = "K"
tier = "weak"

[upstream]
models = ["m1"]

[strategies.agent]
enabled = true
[strategies.agent.map]
hermes = ["m1", "does-not-exist"]
"""
    with pytest.raises(RoutingConfigError) as e:
        RoutingConfig.from_toml(_write_toml(tmp_path, toml))
    assert "does-not-exist" in str(e.value) and "strategies.agent.map" in str(e.value)


def test_capability_validation(tmp_path, monkeypatch):
    """models.capabilities 出现词表外的值 → warning 并忽略该项。"""
    monkeypatch.setenv("K", "v")
    toml = """
[[models]]
name = "m1"
api_key_env = "K"
capabilities = ["text", "bogus", "vision"]
tier = "weak"

[upstream]
models = ["m1"]
"""
    cfg = RoutingConfig.from_toml(_write_toml(tmp_path, toml))
    assert cfg.models[0].capabilities == ["text", "vision"]  # bogus 被忽略


def test_multimodal_map_key_validation(tmp_path, monkeypatch):
    """multimodal.map 出现词表外的键 → warning 并忽略。"""
    monkeypatch.setenv("K", "v")
    toml = """
[[models]]
name = "m1"
api_key_env = "K"
tier = "weak"

[upstream]
models = ["m1"]

[strategies.multimodal]
enabled = true
[strategies.multimodal.map]
vision = ["m1"]
bogus = ["m1"]
"""
    cfg = RoutingConfig.from_toml(_write_toml(tmp_path, toml))
    mm = cfg.multimodal_strategy()
    assert "vision" in mm.map
    assert "bogus" not in mm.map


def test_missing_file_returns_empty(tmp_path):
    """配置文件不存在 → 空配置。"""
    cfg = RoutingConfig.from_toml(tmp_path / "nope.toml")
    assert cfg.models == []
    assert cfg.has_models() is False
    assert cfg.has_upstream() is False


def test_missing_upstream_returns_empty(tmp_path, monkeypatch):
    """有 models 但无 [upstream] → has_upstream=False。"""
    monkeypatch.setenv("K", "v")
    toml = """
[[models]]
name = "m1"
api_key_env = "K"
tier = "weak"
"""
    cfg = RoutingConfig.from_toml(_write_toml(tmp_path, toml))
    assert cfg.has_models() is True
    assert cfg.has_upstream() is False
