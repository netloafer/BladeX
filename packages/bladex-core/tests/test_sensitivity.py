"""ADR-0021 §3 敏感度/暴露等级原语单测（bladex_core，agent 中立）。"""

from bladex_core.sensitivity import (
    DEFAULT_SENSITIVITY_LEVEL,
    EXPOSURE_LOCAL,
    EXPOSURE_PRIVATE,
    EXPOSURE_PUBLIC,
    SensitivityConfig,
    exposure_allows,
    exposure_rank,
    exposure_within,
    strictest_exposure,
)

# ── 暴露等级比较原语 ──


def test_exposure_rank_order():
    """local < private < public（越小越严）。"""
    assert exposure_rank(EXPOSURE_LOCAL) < exposure_rank(EXPOSURE_PRIVATE)
    assert exposure_rank(EXPOSURE_PRIVATE) < exposure_rank(EXPOSURE_PUBLIC)


def test_exposure_rank_unknown_is_public():
    """未知 exposure 按 public（最坏暴露假设，§3.3a）。"""
    assert exposure_rank("mystery") == exposure_rank(EXPOSURE_PUBLIC)


def test_exposure_within_routing_boundary():
    """路由硬边界：模型 exposure 不得超过请求 allowed_exposure。"""
    # allowed=local：只有 local 模型合格
    assert exposure_within(EXPOSURE_LOCAL, EXPOSURE_LOCAL)
    assert not exposure_within(EXPOSURE_PUBLIC, EXPOSURE_LOCAL)
    assert not exposure_within(EXPOSURE_PRIVATE, EXPOSURE_LOCAL)
    # allowed=public：三类都合格
    assert exposure_within(EXPOSURE_LOCAL, EXPOSURE_PUBLIC)
    assert exposure_within(EXPOSURE_PRIVATE, EXPOSURE_PUBLIC)
    assert exposure_within(EXPOSURE_PUBLIC, EXPOSURE_PUBLIC)
    # allowed=private：local/private 合格，public 不合格
    assert exposure_within(EXPOSURE_LOCAL, EXPOSURE_PRIVATE)
    assert not exposure_within(EXPOSURE_PUBLIC, EXPOSURE_PRIVATE)


def test_exposure_allows_injection_guard():
    """注入守卫：Fact.ceiling >= 请求 allowed_exposure 才注入。"""
    # 本地 Fact 不注入公网请求
    assert not exposure_allows(EXPOSURE_LOCAL, EXPOSURE_PUBLIC)
    # 本地 Fact 可注入本地请求
    assert exposure_allows(EXPOSURE_LOCAL, EXPOSURE_LOCAL)
    # 公网 Fact 可注入任何请求
    assert exposure_allows(EXPOSURE_PUBLIC, EXPOSURE_PUBLIC)
    assert exposure_allows(EXPOSURE_PUBLIC, EXPOSURE_LOCAL)
    # private Fact：可注入 local/private，不可注入 public
    assert exposure_allows(EXPOSURE_PRIVATE, EXPOSURE_PRIVATE)
    assert exposure_allows(EXPOSURE_PRIVATE, EXPOSURE_LOCAL)
    assert not exposure_allows(EXPOSURE_PRIVATE, EXPOSURE_PUBLIC)


def test_strictest_exposure():
    """最严暴露 = 序最低者；空 -> local。"""
    assert strictest_exposure({"a": EXPOSURE_PUBLIC, "b": EXPOSURE_LOCAL}) == EXPOSURE_LOCAL
    assert strictest_exposure({"a": EXPOSURE_PRIVATE, "b": EXPOSURE_PUBLIC}) == EXPOSURE_PRIVATE
    assert strictest_exposure({}) == EXPOSURE_LOCAL


# ── SensitivityConfig.resolve 多维取最严 ──


class _FakeIdentity:
    """IdentitySensitivity Protocol 的测试桩。"""

    def __init__(self, principal_level=None, team_levels=None):
        self.principal_level = principal_level
        self.team_levels = team_levels or []


def test_resolve_disabled_is_normal_public():
    """关闭态恒返 (normal, public) = 现状零回归。"""
    cfg = SensitivityConfig(enabled=False)  # 默认 levels={normal:public}
    level, exp = cfg.resolve(agent_id="any", identity=None)
    assert level == DEFAULT_SENSITIVITY_LEVEL
    assert exp == EXPOSURE_PUBLIC


def test_resolve_multi_dimension_strictest():
    """多维取最严：agent normal + team sensitive -> sensitive -> local。"""
    cfg = SensitivityConfig(
        enabled=True,
        agents={"finance-agent": "normal"},
        default_level="normal",
        levels={"normal": EXPOSURE_PUBLIC, "internal": EXPOSURE_PRIVATE, "sensitive": EXPOSURE_LOCAL},
    )
    ident = _FakeIdentity(principal_level="normal", team_levels=["sensitive"])
    level, exp = cfg.resolve(agent_id="finance-agent", identity=ident)
    assert level == "sensitive"
    assert exp == EXPOSURE_LOCAL


def test_resolve_agent_base_fallback():
    """agent 维度 base:profile 精确优先、base 兜底。"""
    cfg = SensitivityConfig(
        enabled=True,
        agents={"hermes": "sensitive"},  # 只配 base
        default_level="normal",
        levels={"normal": EXPOSURE_PUBLIC, "sensitive": EXPOSURE_LOCAL},
    )
    level, exp = cfg.resolve(agent_id="hermes:accept", identity=None)
    assert level == "sensitive"
    assert exp == EXPOSURE_LOCAL


def test_resolve_untagged_uses_default():
    """未标注主体 -> default_level。"""
    cfg = SensitivityConfig(
        enabled=True,
        default_level="sensitive",  # 敏感部署默认最严
        levels={"normal": EXPOSURE_PUBLIC, "sensitive": EXPOSURE_LOCAL},
    )
    level, exp = cfg.resolve(agent_id="unknown-agent", identity=_FakeIdentity())
    assert level == "sensitive"
    assert exp == EXPOSURE_LOCAL


def test_resolve_unknown_level_fail_closed():
    """未知等级 -> fail-closed 最严。"""
    cfg = SensitivityConfig(
        enabled=True,
        agents={"x": "top-secret"},  # 不在 levels 里
        default_level="normal",
        levels={"normal": EXPOSURE_PUBLIC, "sensitive": EXPOSURE_LOCAL},
    )
    level, exp = cfg.resolve(agent_id="x", identity=None)
    # 未知 -> strictest_exposure(levels) = local
    assert exp == EXPOSURE_LOCAL


def test_team_overrides_principal_when_team_set():
    """team > principal 层级：team 标了（normal）用 team，不看 principal（sensitive）。

    ADR-0021 §3.1 修订：层级覆盖（非 max）。team=normal 覆盖 principal=sensitive。
    若要 principal 的 sensitive 生效，team 应不标（让层级下放到 principal）。
    """
    cfg = SensitivityConfig(
        enabled=True,
        default_level="normal",
        levels={"normal": EXPOSURE_PUBLIC, "sensitive": EXPOSURE_LOCAL},
    )
    ident = _FakeIdentity(principal_level="sensitive", team_levels=["normal"])
    level, exp = cfg.resolve(agent_id=None, identity=ident)
    assert level == "normal"
    assert exp == EXPOSURE_PUBLIC


def test_principal_used_when_team_unset():
    """team 未标注 -> 下放 principal。"""
    cfg = SensitivityConfig(
        enabled=True,
        default_level="normal",
        levels={"normal": EXPOSURE_PUBLIC, "sensitive": EXPOSURE_LOCAL},
    )
    ident = _FakeIdentity(principal_level="sensitive", team_levels=[])
    level, exp = cfg.resolve(agent_id=None, identity=ident)
    assert level == "sensitive"
    assert exp == EXPOSURE_LOCAL
