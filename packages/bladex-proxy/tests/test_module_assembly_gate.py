"""V-A3.2 / 批 U · U1.4：装配只剩一个开关——模块开关 `BLADEX_MODULE_ASSEMBLY`。

细粒度 `BLADEX_ASSEMBLY_ENABLED` 与旧名 `BLADEX_CAP_ENABLED` 已于 2026-09-23 删除
（与模块开关重复，ADR-0033 §2.8）：设了也不读——下面最后一条钉住"不读"。
"""

from __future__ import annotations

from bladex_proxy.assembly import AssemblyConfig


def test_default_on(monkeypatch):
    monkeypatch.delenv("BLADEX_MODULE_ASSEMBLY", raising=False)
    assert AssemblyConfig.from_env().enabled is True


def test_module_switch_off(monkeypatch):
    monkeypatch.setenv("BLADEX_MODULE_ASSEMBLY", "0")
    assert AssemblyConfig.from_env().enabled is False


def test_retired_fine_grained_switches_are_not_read(monkeypatch):
    monkeypatch.setenv("BLADEX_MODULE_ASSEMBLY", "1")
    monkeypatch.setenv("BLADEX_ASSEMBLY_ENABLED", "false")
    monkeypatch.setenv("BLADEX_CAP_ENABLED", "false")
    assert AssemblyConfig.from_env().enabled is True, "已删的旋钮不许被偷偷读回来"
