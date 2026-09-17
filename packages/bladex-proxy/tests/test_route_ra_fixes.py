"""RA 修复卡验收剧本 — proxy 侧（路由信号层与容错修复 20260712 T7）。

覆盖：T4 429 跨 provider / 熔断 ModelHealth / on_success 回写；
T5a candidates 自动生成 + tier 不一致告警；provider 推导；judge_timeout 透传。
Router 网关的 acompletion 用 monkeypatch 模拟。
"""

from __future__ import annotations

import asyncio

import pytest
from bladex_proxy import router_sdk
from bladex_proxy.config import ModelRoute
from bladex_proxy.model_health import ModelHealth
from bladex_proxy.route import call_model
from bladex_proxy.routing_config import RoutingConfig


def _run(coro):
    return asyncio.run(coro)


class _FakeUpstreamError(Exception):
    def __init__(self, status_code: int, message: str = "boom") -> None:
        super().__init__(message)
        self.status_code = status_code


def _patch_acompletion(monkeypatch, behavior: dict):
    calls: list[str] = []

    async def fake(**kwargs):
        calls.append(kwargs["model"])
        act = behavior.get(kwargs["model"])
        if isinstance(act, Exception):
            raise act
        return {"model": kwargs["model"], "ok": True}

    monkeypatch.setattr(router_sdk, "acompletion", fake)
    return calls


# ── T4: 429 只跨 provider（RA4）──


def test_429_same_provider_no_retry(monkeypatch):
    """验收剧本1：429 + 全同 provider → 不重试，原错误透传。"""
    calls = _patch_acompletion(monkeypatch, {
        "primary": _FakeUpstreamError(429, "rate limited"),
        "backup": None,
    })
    with pytest.raises(_FakeUpstreamError) as ei:
        _run(call_model(
            "primary", [{"role": "user", "content": "hi"}], stream=False,
            route=ModelRoute(model="primary", provider="ark"),
            failover=[ModelRoute(model="backup", provider="ark")],
        ))
    assert ei.value.status_code == 429
    assert calls == ["primary"]  # 没有 5 连锤


def test_429_cross_provider_switches(monkeypatch):
    """验收剧本2：429 + 存在跨 provider 候选 → 只切到该候选。"""
    calls = _patch_acompletion(monkeypatch, {
        "primary": _FakeUpstreamError(429),
        "same-quota": None,
        "other-provider": None,
    })
    result = _run(call_model(
        "primary", [{"role": "user", "content": "hi"}], stream=False,
        route=ModelRoute(model="primary", provider="ark"),
        failover=[
            ModelRoute(model="same-quota", provider="ark"),
            ModelRoute(model="other-provider", provider="glm"),
        ],
    ))
    assert result["model"] == "other-provider"
    assert calls == ["primary", "other-provider"]  # 跳过了同 provider 的 same-quota


def test_5xx_same_provider_still_switches(monkeypatch):
    """验收剧本3：5xx → 同 provider 仍按序切（现状不回归）。"""
    calls = _patch_acompletion(monkeypatch, {
        "primary": _FakeUpstreamError(503),
        "backup": None,
    })
    result = _run(call_model(
        "primary", [{"role": "user", "content": "hi"}], stream=False,
        route=ModelRoute(model="primary", provider="ark"),
        failover=[ModelRoute(model="backup", provider="ark")],
    ))
    assert result["model"] == "backup"
    assert calls == ["primary", "backup"]


def test_ungrouped_429_no_cross_provider(monkeypatch):
    """provider 全空（未分组）→ 429 视为同故障域，不切。"""
    calls = _patch_acompletion(monkeypatch, {
        "primary": _FakeUpstreamError(429),
        "backup": None,
    })
    with pytest.raises(_FakeUpstreamError):
        _run(call_model(
            "primary", [{"role": "user", "content": "hi"}], stream=False,
            route=ModelRoute(model="primary"),
            failover=[ModelRoute(model="backup")],
        ))
    assert calls == ["primary"]


# ── T4: 熔断 ModelHealth ──


def test_health_marked_on_failover_trigger(monkeypatch):
    """触发 failover 类错误 → mark_unhealthy；冷却期满恢复。"""
    _patch_acompletion(monkeypatch, {
        "primary": _FakeUpstreamError(503),
        "backup": None,
    })
    health = ModelHealth(cooldown_s=0.05)
    _run(call_model(
        "primary", [{"role": "user", "content": "hi"}], stream=False,
        route=ModelRoute(model="primary"),
        failover=[ModelRoute(model="backup")],
        health=health,
    ))
    assert health.is_healthy("primary") is False
    assert health.is_healthy("backup") is True  # 成功方保持健康

    import time
    time.sleep(0.06)
    assert health.is_healthy("primary") is True  # 冷却期满自动恢复


def test_health_not_marked_on_4xx(monkeypatch):
    """4xx 不是 failover 触发 → 不熔断。"""
    _patch_acompletion(monkeypatch, {"primary": _FakeUpstreamError(400)})
    health = ModelHealth(cooldown_s=30)
    with pytest.raises(_FakeUpstreamError):
        _run(call_model(
            "primary", [{"role": "user", "content": "hi"}], stream=False,
            route=ModelRoute(model="primary"), health=health,
        ))
    assert health.is_healthy("primary") is True


def test_success_recovers_health(monkeypatch):
    """成功调用 → mark_healthy 清除熔断。"""
    _patch_acompletion(monkeypatch, {"m": None})
    health = ModelHealth(cooldown_s=300)
    health.mark_unhealthy("m")
    assert health.is_healthy("m") is False
    _run(call_model("m", [{"role": "user", "content": "hi"}], stream=False,
                    route=ModelRoute(model="m"), health=health))
    assert health.is_healthy("m") is True


# ── T2: on_success 回写（failover 实际切换后粘住新模型）──


def test_on_success_reports_used_model(monkeypatch):
    """failover 切到 backup → on_success 收到 backup（供 update_sticky）。"""
    _patch_acompletion(monkeypatch, {
        "primary": _FakeUpstreamError(503),
        "backup": None,
    })
    used: list[str] = []
    _run(call_model(
        "primary", [{"role": "user", "content": "hi"}], stream=False,
        route=ModelRoute(model="primary"),
        failover=[ModelRoute(model="backup")],
        on_success=used.append,
    ))
    assert used == ["backup"]


# ── T5a: candidates 自动生成 + tier 一致性（RA6）──


def _toml_no_candidates(tmp_path, monkeypatch) -> RoutingConfig:
    monkeypatch.setenv("K", "secret")
    toml = '''
[[models]]
name = "pro"
api_key_env = "K"
api_base = "https://ark.example.com/v3"
tier = "strong"

[[models]]
name = "glm"
api_key_env = "K"
api_base = "https://glm.example.com/v4"
tier = "medium"

[[models]]
name = "mini"
api_key_env = "K"
api_base = "https://ark.example.com/v3"
tier = "weak"

[upstream]
models = ["glm", "mini"]

[strategies.filter]
enabled = true
judge_model = "mini"
default_tier = "weak"
judge_timeout_s = 2.5
'''
    p = tmp_path / "routing.toml"
    p.write_text(toml, encoding="utf-8")
    return RoutingConfig.from_toml(p)


def test_auto_candidates_from_models_tier(tmp_path, monkeypatch):
    """candidates 未配置 → 由 [[models]].tier 自动生成（单一 tier 真相）。"""
    cfg = _toml_no_candidates(tmp_path, monkeypatch)
    data = cfg.filter_strategy()
    assert data.candidates == {"weak": ["mini"], "medium": ["glm"], "strong": ["pro"]}


def test_judge_timeout_passthrough(tmp_path, monkeypatch):
    """judge_timeout_s 从 toml 透传到 core FilterStrategyData。"""
    cfg = _toml_no_candidates(tmp_path, monkeypatch)
    assert cfg.filter_strategy().judge_timeout_s == 2.5


def test_provider_derived_from_api_base(tmp_path, monkeypatch):
    """provider 未配置 → 从 api_base host 推导；同 host 同组。"""
    cfg = _toml_no_candidates(tmp_path, monkeypatch)
    by_name = {m.name: m.provider for m in cfg.models}
    assert by_name["pro"] == "ark.example.com"
    assert by_name["mini"] == "ark.example.com"
    assert by_name["glm"] == "glm.example.com"
    # model_candidates 带 provider
    cands = cfg.model_candidates()
    assert cands["pro"].provider == "ark.example.com"


def test_provider_explicit_wins(tmp_path, monkeypatch):
    """显式 provider 优先于推导。"""
    monkeypatch.setenv("K", "secret")
    toml = '''
[[models]]
name = "pro"
api_key_env = "K"
api_base = "https://ark.example.com/v3"
provider = "ark"
tier = "strong"

[upstream]
models = ["pro"]
'''
    p = tmp_path / "routing.toml"
    p.write_text(toml, encoding="utf-8")
    cfg = RoutingConfig.from_toml(p)
    assert cfg.models[0].provider == "ark"


def test_tier_mismatch_warning(tmp_path, monkeypatch):
    """显式 candidates 与 [[models]].tier 不一致 → routing_tier_mismatch 告警可 grep。"""
    from structlog.testing import capture_logs

    monkeypatch.setenv("K", "secret")
    toml = '''
[[models]]
name = "flash"
api_key_env = "K"
tier = "weak"

[upstream]
models = ["flash"]

[strategies.filter]
enabled = true
judge_model = "flash"
[strategies.filter.candidates]
medium = ["flash"]
'''
    p = tmp_path / "routing.toml"
    p.write_text(toml, encoding="utf-8")
    with capture_logs() as logs:
        RoutingConfig.from_toml(p)
    events = [entry["event"] for entry in logs]
    assert "routing_tier_mismatch" in events
