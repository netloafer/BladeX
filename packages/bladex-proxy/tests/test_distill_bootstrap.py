"""蒸馏模型自举单元测试（T6.3，ADR-0018 §3.6）。"""

from bladex_proxy.routing_config import (
    DistillConfig,
    ModelSpec,
    RoutingConfig,
    _write_distill_to_toml,
    bootstrap_distill_model,
)


def _cfg_with_pool(*specs: ModelSpec) -> RoutingConfig:
    cfg = RoutingConfig(models=list(specs))
    cfg._resolve_api_keys()  # 无 env -> available=True；api_key_env 缺 -> available=False
    return cfg


def test_bootstrap_prefers_env():
    """env distill_model 非空 -> 直接用，不自举。"""
    cfg = _cfg_with_pool()
    assert bootstrap_distill_model("openai/foo", cfg, None) == "openai/foo"


def test_bootstrap_prefers_routing_distill():
    """routing [distill].model 非空 -> 用，不自举。"""
    cfg = _cfg_with_pool()
    cfg.distill = DistillConfig(model="openai/from-toml")
    assert bootstrap_distill_model("", cfg, None) == "openai/from-toml"


def test_bootstrap_picks_weakest_tier(tmp_path):
    """env/[distill] 都空 -> 从模型池按 weak->medium->strong 取首个 available 写回。"""
    toml = tmp_path / "routing.toml"
    toml.write_text("[upstream]\nmodels = []\n", encoding="utf-8")
    cfg = _cfg_with_pool(
        ModelSpec(name="openai/strong1", tier="strong"),
        ModelSpec(name="openai/weak1", tier="weak"),
        ModelSpec(name="openai/medium1", tier="medium"),
    )
    chosen = bootstrap_distill_model("", cfg, toml)
    assert chosen == "openai/weak1"  # 最弱档优先
    text = toml.read_text(encoding="utf-8")
    assert "[distill]" in text
    assert 'model = "openai/weak1"' in text


def test_bootstrap_falls_to_medium_if_no_weak(tmp_path):
    """无 weak 档 -> 取 medium。"""
    toml = tmp_path / "routing.toml"
    toml.write_text("[upstream]\nmodels = []\n", encoding="utf-8")
    cfg = _cfg_with_pool(
        ModelSpec(name="openai/medium1", tier="medium"),
        ModelSpec(name="openai/strong1", tier="strong"),
    )
    assert bootstrap_distill_model("", cfg, toml) == "openai/medium1"


def test_bootstrap_empty_pool_returns_empty(tmp_path):
    """模型池空 -> 返回空（蒸馏停摆，走 PassthroughDistiller 降级）。"""
    toml = tmp_path / "routing.toml"
    toml.write_text("[upstream]\nmodels = []\n", encoding="utf-8")
    cfg = _cfg_with_pool()  # 无模型
    assert bootstrap_distill_model("", cfg, toml) == ""


def test_bootstrap_skips_unavailable(tmp_path):
    """不可用模型（api_key env 缺）跳过，取下一个 available 的最弱档。"""
    toml = tmp_path / "routing.toml"
    toml.write_text("[upstream]\nmodels = []\n", encoding="utf-8")
    cfg = _cfg_with_pool(
        ModelSpec(name="openai/weak1", tier="weak", api_key_env="MISSING_KEY"),
        ModelSpec(name="openai/weak2", tier="weak"),
    )
    chosen = bootstrap_distill_model("", cfg, toml)
    assert chosen == "openai/weak2"


def test_write_distill_to_toml_appends_when_absent(tmp_path):
    """routing.toml 无 [distill] 段 -> 追加。"""
    toml = tmp_path / "routing.toml"
    toml.write_text("[upstream]\nmodels = []\n", encoding="utf-8")
    _write_distill_to_toml(toml, "openai/x")
    text = toml.read_text(encoding="utf-8")
    assert "[distill]" in text
    assert 'model = "openai/x"' in text


def test_write_distill_to_toml_replaces_when_present(tmp_path):
    """routing.toml 已有 [distill] model -> 替换。"""
    toml = tmp_path / "routing.toml"
    toml.write_text('[distill]\nmodel = "old"\n', encoding="utf-8")
    _write_distill_to_toml(toml, "openai/new")
    text = toml.read_text(encoding="utf-8")
    assert 'model = "openai/new"' in text
    assert 'model = "old"' not in text


def test_config_distill_model_default_empty():
    """N3 修复：BLADEX_DISTILL_MODEL 默认空串 -> 自举成为默认路径（非死代码）。"""
    import os

    from bladex_proxy.config import ProxyConfig

    old = os.environ.pop("BLADEX_DISTILL_MODEL", None)
    try:
        assert ProxyConfig().distill_model == "", \
            "默认应空串，让 bootstrap_distill_model 走 env>toml>自举（原默认非空致自举死代码）"
    finally:
        if old is not None:
            os.environ["BLADEX_DISTILL_MODEL"] = old
