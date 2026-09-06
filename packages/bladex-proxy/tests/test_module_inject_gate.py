"""V-A3.1：inject 模块开关（`BLADEX_MODULE_INJECT`，默认开 = 现状零回归）。

关的语义（ADR-0032 三红线之 3 的注入面版本）：注入平面整体停——
prefetch / 三平面 / 硬规则 / 理解层 / 澄清都不做，转发消息除 strip 卫生外
逐字不变；但 **task_units 写侧认知照常产出**——关掉注入不许连带废掉
记忆写入质量（ADR-0024 §4.3 装配解耦的同一条理由，这里对注入面重申）。
"""

from __future__ import annotations

import asyncio

from bladex_proxy import inject
from bladex_proxy.config import ProxyConfig


class _Source:
    """可观测的假检索源：记录 build_facts 是否被调过。"""

    last_facts: list = []
    last_facts_wide: list = []

    def __init__(self) -> None:
        self.calls = 0

    def build_facts(self, *a, **kw):  # noqa: ANN002, ANN003, ANN201, ARG002
        self.calls += 1
        return ["memory line"]


_MSGS = [{"role": "system", "content": "sys"},
         {"role": "user", "content": "第一问"},
         {"role": "user", "content": "hi"}]


def _run(cfg, source, **kw):
    return asyncio.run(inject.do_inject_async(
        [dict(m) for m in _MSGS], "hi", cfg, source=source, **kw))


def test_module_off_injects_nothing_but_units_survive(monkeypatch):
    monkeypatch.setenv("BLADEX_MODULE_INJECT", "0")
    src = _Source()
    new_messages, injected_text, _ms, route_facts, cap_info = _run(ProxyConfig(), src)

    assert src.calls == 0, "模块关时不许发起检索"
    assert injected_text == ""
    assert route_facts == []
    assert new_messages == _MSGS, "除 strip 外消息必须逐字不变（本例无旧注入可剥）"
    assert cap_info.get("injected_fact_ids") == []
    assert cap_info.get("injected_fact_keys") == {}
    # 🔴 写侧认知不受影响：units 仍按原始 messages 产出
    assert cap_info.get("units"), "task_units 不许随注入面一起关（ADR-0024 §4.3）"


def test_module_off_skips_hard_rules_even_for_auxiliary(monkeypatch):
    monkeypatch.setenv("BLADEX_MODULE_INJECT", "0")
    src = _Source()
    _, injected_text, _ms, _facts, _cap = _run(ProxyConfig(), src, auxiliary=True)
    assert injected_text == "", "关 = 零注入，aux 轮的硬规则也停"


def test_sync_entry_gated_identically(monkeypatch):
    """两个入口一个判据：sync `do_inject` 与 async 版同受门控。

    （首版门只落在了 sync 版上、async 版漏网——同一锚串两处出现，sed 只替换了
    第一处。本用例与上面的 async 用例互为对账，防再犯。）
    """
    monkeypatch.setenv("BLADEX_MODULE_INJECT", "0")
    src = _Source()
    new_messages, injected_text, _ms, route_facts, cap_info = inject.do_inject(
        [dict(m) for m in _MSGS], "hi", ProxyConfig(), source=src)
    assert src.calls == 0
    assert injected_text == ""
    assert route_facts == []
    assert new_messages == _MSGS
    assert cap_info.get("units")


def test_default_on_behavior_unchanged(monkeypatch):
    monkeypatch.delenv("BLADEX_MODULE_INJECT", raising=False)
    src = _Source()
    new_messages, injected_text, _ms, _facts, cap_info = _run(ProxyConfig(), src)

    assert src.calls == 1, "默认开 = 现状：照常检索"
    assert "memory line" in injected_text
    assert new_messages != _MSGS, "默认开时注入块应已插入"
    assert cap_info.get("units")
