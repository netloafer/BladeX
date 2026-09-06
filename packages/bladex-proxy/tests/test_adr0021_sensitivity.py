"""ADR-0021 T3 验收：敏感度解析层 + 编排换序。

覆盖计划卡 T3 四项验收：
① enabled=false = 现状逐字不变（全量套件零回归已证，此处补显式断言）；
② 多维取最严（在 test_sensitivity.py，此处补编排层）；
③ 解析异常 -> 最严（fail-closed）；
④ Turn 落库带 sensitivity 且 rebuild 可读。
"""

import tempfile
import textwrap
from pathlib import Path

from bladex_proxy.config import ProxyConfig
from bladex_proxy.models import Identity, Turn, TurnStatus
from bladex_proxy.server import _resolve_sensitivity
from bladex_proxy.storage.memory_hub import MemoryHub


def _make_turn(sensitivity: str = "normal") -> Turn:
    return Turn(
        identity=Identity(user_id="u1", agent_id="a1", session_id="s1"),
        model="m",
        request_messages=[{"role": "user", "content": "sensitive note"}],
        response_text="ok",
        status=TurnStatus.OK,
        sensitivity=sensitivity,
    )


# ── ① 关闭态 = 现状逐字不变 ──


def test_sensitivity_disabled_returns_normal_public(tmp_path, monkeypatch):
    """无 [strategies.sensitivity] 段 -> 关闭态 -> (normal, public) = 现状。"""
    routing = tmp_path / "routing.toml"
    routing.write_text('[upstream]\nmodels = ["m1"]\n[[models]]\nname = "m1"\n', encoding="utf-8")
    monkeypatch.setenv("BLADEX_ROUTING_CONFIG", str(routing))
    cfg = ProxyConfig()
    ident = Identity(user_id="u1", agent_id="a1")
    level, exp = _resolve_sensitivity(cfg, ident)
    assert level == "normal"
    assert exp == "public"


# ── ② 编排层多维取最严（agent + team）──


def test_sensitivity_enabled_multi_dim_strictest(tmp_path, monkeypatch):
    """敏感层开 + agent normal + team sensitive -> sensitive -> local。"""
    routing = tmp_path / "routing.toml"
    routing.write_text(textwrap.dedent("""
        [upstream]
        models = ["m1"]
        [[models]]
        name = "m1"
        tier = "weak"
        [strategies.sensitivity]
        enabled = true
        default_level = "normal"
        [strategies.sensitivity.agents]
        "a1" = "normal"
        [strategies.sensitivity.levels]
        normal = "public"
        sensitive = "local"
    """), encoding="utf-8")
    monkeypatch.setenv("BLADEX_ROUTING_CONFIG", str(routing))
    cfg = ProxyConfig()
    ident = Identity(user_id="u1", agent_id="a1",
                     principal_sensitivity="", team_sensitivities=["sensitive"])
    level, exp = _resolve_sensitivity(cfg, ident)
    assert level == "sensitive"
    assert exp == "local"


# ── ③ 解析异常 -> 最严（fail-closed）──


def test_sensitivity_resolve_exception_fail_closed(monkeypatch):
    """routing_config 抛异常 -> _resolve_sensitivity fail-closed 返回最严。"""
    cfg = ProxyConfig()

    def _boom():
        raise RuntimeError("toml parse blew up")
    monkeypatch.setattr(type(cfg), "routing_config", property(lambda self: _boom()))
    ident = Identity(user_id="u1", agent_id="a1")
    level, exp = _resolve_sensitivity(cfg, ident)
    assert exp == "local"  # 最严
    assert level == "sensitive"


# ── ④ Turn 落库带 sensitivity 且 rebuild 可读 ──


def test_turn_sensitivity_roundtrip_ledger():
    """Turn.sensitivity 经 Memory Hub put/get 保持。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger = MemoryHub(Path(tmpdir) / "ledger")
        ledger.open()
        t = _make_turn(sensitivity="sensitive")
        ledger.put("u1/a1/s1/k1", t)
        got = ledger.get("u1/a1/s1/k1")
        assert got is not None
        assert got.sensitivity == "sensitive"
        ledger.close()


def test_turn_default_sensitivity_is_normal():
    """不显式设 sensitivity -> 默认 normal（现状等价）。"""
    t = Turn(identity=Identity(user_id="u"), model="m", status=TurnStatus.OK)
    assert t.sensitivity == "normal"


# ── T4 ④ strict 拒启动 / false 告警降级 ──

_SENS_TOML = textwrap.dedent("""
    [upstream]
    models = ["local-m", "pub-m"]
    [[models]]
    name = "local-m"
    tier = "weak"
    exposure = "local"
    [[models]]
    name = "pub-m"
    tier = "weak"
    exposure = "public"
    [strategies.filter]
    enabled = true
    judge_model = "pub-m"
    [strategies.sensitivity]
    enabled = true
    [strategies.sensitivity.levels]
    normal = "public"
    sensitive = "local"
""")


def test_strict_refuses_startup_when_judge_not_local(tmp_path, monkeypatch):
    """敏感层开 + 裁判 pub-m(非本地) + strict -> 拒启动。"""
    routing = tmp_path / "routing.toml"
    routing.write_text(_SENS_TOML, encoding="utf-8")
    monkeypatch.setenv("BLADEX_ROUTING_CONFIG", str(routing))
    monkeypatch.setenv("BLADEX_ROUTE_STRICT", "true")
    cfg = ProxyConfig()
    import pytest
    with pytest.raises(RuntimeError, match="sensitivity"):
        from bladex_proxy.server import _validate_sensitivity_judge
        _validate_sensitivity_judge(cfg)


def test_non_strict_warns_when_judge_not_local(tmp_path, monkeypatch):
    """敏感层开 + 裁判非本地 + strict=false -> 不拒启动（运行时跳裁判降级）。"""
    routing = tmp_path / "routing.toml"
    routing.write_text(_SENS_TOML, encoding="utf-8")
    monkeypatch.setenv("BLADEX_ROUTING_CONFIG", str(routing))
    monkeypatch.setenv("BLADEX_ROUTE_STRICT", "false")
    cfg = ProxyConfig()
    from bladex_proxy.server import _validate_sensitivity_judge
    _validate_sensitivity_judge(cfg)  # 不抛


def test_local_judge_passes_validation(tmp_path, monkeypatch):
    """敏感层开 + 裁判 local-m(本地) -> 校验通过。"""
    toml = _SENS_TOML.replace('judge_model = "pub-m"', 'judge_model = "local-m"')
    routing = tmp_path / "routing.toml"
    routing.write_text(toml, encoding="utf-8")
    monkeypatch.setenv("BLADEX_ROUTING_CONFIG", str(routing))
    monkeypatch.setenv("BLADEX_ROUTE_STRICT", "true")
    cfg = ProxyConfig()
    from bladex_proxy.server import _validate_sensitivity_judge
    _validate_sensitivity_judge(cfg)  # 不抛
