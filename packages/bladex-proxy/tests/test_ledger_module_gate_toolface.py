"""MQ-L70（2026-09-12）：模块门 `BLADEX_MODULE_LEDGER=0` 必须同时挡**工具面**。

## 病例

`agency/runtime.py` `augment_tools` 里 `_ledger_ok` 只看档位与"本会话绑着账本"，
没并 `self.ledger_on`；而 `insert_ledger_block` 并了。⇒ 关 `ledger` 模块只挡
正文块、账本三工具照注。live：`logs/proxy-20260912-105234.log`
`module_ledger=False` 却 `ledger_face=True` **35/37 轮**。

## 为什么这是实验有效性前置而不是普通缺陷

那一窗当时是"账本关"对照臂。A 臂 = 正文块 + 工具，B 臂 = 只有工具 ⇒ 读出来的
差异只是"正文块的成本"，会被误读成"账本面不是原因"。后面每一次"账本关"臂
都可能在测一个开着的账本面。

## 三格

1. **阴性对照**（主判据）：`ledger=0, toolface=1`，档位在集合内、本会话还绑着
   激活账本（两个放行口都开着）⇒ 账本族一个都不许出现、`ledger_face=False`；
   记忆族照注（族拆分不受影响）。去掉那句 `and self.ledger_on` 这条必变红。
2. **阳性对照**：同样前提只把 `ledger=1` ⇒ 账本三工具都在。没有这一半，
   工具面整个坏掉时第 1 格也绿。
3. **结构守卫**：两个消费点（`augment_tools` / `_build_ledger_block`）都读
   `ledger_on` —— 与 `test_both_consumers_read_one_predicate` 同款做法，
   防"同一开关两个消费点只接了一个"（MQ-A18/P7 族）。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import structlog.testing
from bladex_core.ledger import Ledger
from bladex_core.ledger_runtime import activation_scope
from bladex_proxy.agency import AgencyRuntime

_LEDGER_TOOLS = {"bladex_ledger_read", "bladex_ledger_switch", "bladex_ledger_update"}


@pytest.fixture(autouse=True)
def _toolface_on(monkeypatch):
    monkeypatch.setenv("BLADEX_MODULE_TOOLFACE", "1")
    monkeypatch.setenv("BLADEX_LEDGER_TIERS", "auto")


def _rt_with_bound_ledger(session_id: str = "sess-A") -> AgencyRuntime:
    """本会话绑着激活账本 = 第二个放行口（MQ-L7 补完）也开着。"""
    rt = AgencyRuntime(index=None, hub=None)
    led = Ledger(ledger_id="ldg-l70-probe0001", title="L70 probe", goal="probe")
    rt.pool[led.ledger_id] = led
    scope = activation_scope("probe-agent", "")
    rt.activation.restore({scope: led.ledger_id}, sessions={scope: session_id})
    return rt


def _names(rt: AgencyRuntime) -> tuple[set[str], bool, dict]:
    with structlog.testing.capture_logs() as cap:
        tools, injected = rt.augment_tools(
            None, agent_id="probe-agent", auxiliary=False, tier="strong",
            session_id="sess-A")
    row = next(e for e in cap if e.get("event") == "agency_toolface_decision")
    names = {(t.get("function") or t).get("name", "") for t in (tools or [])}
    return names, injected, row


class TestModuleGateBlocksToolface:
    def test_ledger_off_blocks_ledger_tools_even_when_both_grants_open(self, monkeypatch):
        """阴性对照：档位在集合内 + 本会话绑着账本，`ledger=0` 仍一个账本工具不给。"""
        monkeypatch.setenv("BLADEX_MODULE_LEDGER", "0")
        rt = _rt_with_bound_ledger()
        names, injected, row = _names(rt)
        assert not (names & _LEDGER_TOOLS), (
            f"模块门只挡了正文块没挡工具面（MQ-L70 复发）：{sorted(names & _LEDGER_TOOLS)}")
        assert row["ledger_face"] is False, "日志读数与实际放行不一致"
        # 记忆族与账本模块无关，照注（MQ-L34 族拆分不受本修影响）
        assert injected and "bladex_memory_search" in names
        # `Turn.ledger_face` 的来源也必须是 False，否则 Hub 里那一窗还是假读数
        assert rt.last_ledger_face is False

    def test_ledger_on_keeps_ledger_tools(self, monkeypatch):
        """阳性对照：同前提只翻 `ledger=1` ⇒ 账本三工具都在。"""
        monkeypatch.setenv("BLADEX_MODULE_LEDGER", "1")
        rt = _rt_with_bound_ledger()
        names, injected, row = _names(rt)
        assert injected and names >= _LEDGER_TOOLS, sorted(names)
        assert row["ledger_face"] is True and rt.last_ledger_face is True

    def test_one_switch_closes_both_consumers(self, monkeypatch):
        """关账本 = **一个**开关：`ledger=0` 时正文块与工具面同时关。
        交接稿 `HANDOFF-20260912-H.md` §1.9 写的"要两个开关
        （MODULE_LEDGER=0 + LEDGER_TIERS=none）"自本修起作废。"""
        monkeypatch.setenv("BLADEX_MODULE_LEDGER", "0")
        rt = _rt_with_bound_ledger()
        msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]
        out = rt.insert_ledger_block([dict(m) for m in msgs], "probe-agent",
                                     tier="strong", session_id="sess-A")
        assert len(out) == len(msgs), "正文块没被模块门挡住"
        names, _, _ = _names(rt)
        assert not (names & _LEDGER_TOOLS)


def test_both_consumers_read_ledger_on():
    """结构守卫：`augment_tools` 与 `insert_ledger_block` 都读 `ledger_on`。"""
    src = Path(__file__).resolve().parents[1] / "bladex_proxy" / "agency" / "runtime.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    readers = {
        fn.name for fn in ast.walk(tree)
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(isinstance(n, ast.Attribute) and n.attr == "ledger_on"
                and isinstance(n.value, ast.Name) and n.value.id == "self"
                for n in ast.walk(fn))
    }
    # 正文块的门在 `_build_ledger_block`（`insert_ledger_block` 委托它），
    # 工具面的门在 `augment_tools`。
    assert readers >= {"augment_tools", "_build_ledger_block"}, (
        f"只有 {sorted(readers)} 读了模块门 —— 另一个消费点没接（MQ-L70 形态）")
