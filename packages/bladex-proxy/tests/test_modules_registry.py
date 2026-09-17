"""V-A1 验收剧本：模块注册表——开关生效 / Router 不可关 / 未知名拒启动 / 依赖校验 /
默认形态=现状 / .env.example 对账（刚性原则 12）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from bladex_proxy.modules import (
    ENV_PREFIX,
    MODULE_SPECS,
    ModuleConfigError,
    module_enabled,
    resolve_module_switches,
    validate_modules,
)

_REPO = Path(__file__).resolve().parents[3]


class TestDefaults:
    def test_default_shape_is_status_quo(self):
        """一个 BLADEX_MODULE_* 都不配 = 生产形态：v5 四模块 2026-09-02 翻默认开，
        只剩 embed 默认关（举证 = MQ-W8 复测）。"""
        switches = validate_modules(env={})
        assert switches == {
            "router": True, "inject": True, "sensitivity": True, "assembly": True,
            "embed": False,
            "ledger": True, "flash": True, "toolface": True, "interception": True,
            "export": True, "admin_read": True,   # V-A3（09-06 F0.2）：两个边缘模块，默认开 = 现状
        }

    def test_v5_modules_still_have_rollback_channel(self):
        """翻默认后 =0 仍可逐个关（回滚通道）；interception 依赖 toolface 的校验不变。"""
        off = validate_modules(env={"BLADEX_MODULE_LEDGER": "0", "BLADEX_MODULE_FLASH": "0",
                                    "BLADEX_MODULE_TOOLFACE": "0",
                                    "BLADEX_MODULE_INTERCEPTION": "0"})
        assert not any(off[n] for n in ("ledger", "flash", "toolface", "interception"))

    def test_router_is_required(self):
        assert MODULE_SPECS["router"].required is True


class TestSwitches:
    def test_switch_disables_module(self):
        assert module_enabled("sensitivity", env={"BLADEX_MODULE_SENSITIVITY": "0"}) is False
        assert module_enabled("sensitivity", env={}) is True

    def test_switch_enables_v5_module(self):
        assert module_enabled("ledger", env={"BLADEX_MODULE_LEDGER": "1"}) is True

    def test_bad_boolean_fails_loud(self):
        with pytest.raises(ModuleConfigError):
            validate_modules(env={"BLADEX_MODULE_INJECT": "maybe"})

    def test_undeclared_name_in_code_fails_loud(self):
        """调用点拼错模块名必须当场炸，不许静默返回 False。"""
        with pytest.raises(ModuleConfigError):
            module_enabled("sensitivty", env={})


class TestFailsLoud:
    def test_router_switch_refuses_startup_even_when_on(self):
        with pytest.raises(ModuleConfigError, match="mandatory"):
            validate_modules(env={"BLADEX_MODULE_ROUTER": "1"})
        with pytest.raises(ModuleConfigError, match="mandatory"):
            validate_modules(env={"BLADEX_MODULE_ROUTER": "0"})

    def test_unknown_switch_refuses_startup(self):
        with pytest.raises(ModuleConfigError, match="unknown module switch"):
            validate_modules(env={"BLADEX_MODULE_LEDGRE": "1"})  # 拼错

    def test_dependency_violation_refuses_startup(self):
        with pytest.raises(ModuleConfigError, match="depends on disabled"):
            validate_modules(env={"BLADEX_MODULE_INTERCEPTION": "1",
                                  "BLADEX_MODULE_TOOLFACE": "0"})
        # 翻默认后的新形态：只关 toolface、interception 沿默认开 ⇒ 同样拒启动
        with pytest.raises(ModuleConfigError, match="depends on disabled"):
            validate_modules(env={"BLADEX_MODULE_TOOLFACE": "0"})

    def test_dependency_satisfied_passes(self):
        switches = validate_modules(env={"BLADEX_MODULE_INTERCEPTION": "1",
                                         "BLADEX_MODULE_TOOLFACE": "1"})
        assert switches["interception"] and switches["toolface"]


class TestEnvExampleParity:
    def test_every_optional_module_is_documented(self):
        """默认值单一真相源在 modules.py，但用户发现面在 .env.example——两边由本测试
        对账（ADR-0027 §5.4 同款；改一边不改另一边 = 红）。"""
        text = (_REPO / "config" / ".env.example").read_text(encoding="utf-8")
        for name, spec in MODULE_SPECS.items():
            key = f"{ENV_PREFIX}{name.upper()}"
            if spec.required:
                assert f"{key}=" not in text or "拒启动" in text, \
                    f"required module {name} must not be offered as a switch"
            else:
                assert key in text, f"{key} missing from config/.env.example"

    def test_documented_defaults_match_specs(self):
        text = (_REPO / "config" / ".env.example").read_text(encoding="utf-8")
        for name, spec in MODULE_SPECS.items():
            if spec.required:
                continue
            expect = f"# BLADEX_MODULE_{name.upper()}={'1' if spec.default_enabled else '0'}"
            assert expect in text, f"{expect} not in .env.example (drifted default?)"

    def test_resolve_matches_specs_without_env(self):
        assert resolve_module_switches({}) == {
            n: s.default_enabled for n, s in MODULE_SPECS.items()}
