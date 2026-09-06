"""V-A3.2：assembly 模块开关（`BLADEX_MODULE_ASSEMBLY`）。

与既有细粒度开关 `BLADEX_ASSEMBLY_ENABLED` 是**与**关系：
模块门 = 注册表单一真相源的粗粒度开关；细粒度 env 保留为兼容通道。
两者默认都开 = 现状零回归；任一为关即关。
"""

from __future__ import annotations

from bladex_proxy.assembly import AssemblyConfig


def test_default_both_on(monkeypatch):
    monkeypatch.delenv("BLADEX_MODULE_ASSEMBLY", raising=False)
    monkeypatch.delenv("BLADEX_ASSEMBLY_ENABLED", raising=False)
    monkeypatch.delenv("BLADEX_CAP_ENABLED", raising=False)
    assert AssemblyConfig.from_env().enabled is True


def test_module_switch_off_wins(monkeypatch):
    monkeypatch.setenv("BLADEX_MODULE_ASSEMBLY", "0")
    monkeypatch.setenv("BLADEX_ASSEMBLY_ENABLED", "true")
    assert AssemblyConfig.from_env().enabled is False, \
        "模块门关时，细粒度开也不许把装配打开"


def test_legacy_switch_off_still_respected(monkeypatch):
    monkeypatch.setenv("BLADEX_MODULE_ASSEMBLY", "1")
    monkeypatch.setenv("BLADEX_ASSEMBLY_ENABLED", "false")
    assert AssemblyConfig.from_env().enabled is False, \
        "细粒度关是既有回归通道，模块门开不许压过它"
