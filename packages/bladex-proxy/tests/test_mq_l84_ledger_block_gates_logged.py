"""MQ-L84（批 Q3A）：账本块「不注入」的每一道门都打同一个事件、带门名。

`insert_ledger_block` 三道门：① `auxiliary` ② 档位 ∉ LEDGER_TIERS ∧ 本会话没绑激活账本
③ 模块关（`BLADEX_MODULE_LEDGER` / `BLADEX_MODULE_TOOLFACE`）。MQ-L83 只给门 ② 补了
`agency_ledger_block_gated`；门 ① 与门 ③ 两跳都不留痕（门 ③：`_build_ledger_block`
开头 return None → 调用方 `if msg is None: return messages`）。

修后三道门走同一个生产点（函数内唯一一行 `logger.info("agency_ledger_block_gated"`），
`reason` = 门名；门 ③ 的同一跳里还有一格"无激活账本且不给指令 = 无器则无令"，一并具名
为 `no_ledger_no_instruction`（此前同样静默）。**零行为**：返回值逐字不变。
"""
from __future__ import annotations

import pytest
from bladex_core.ledger import Ledger
from bladex_core.ledger_runtime import activation_scope
from bladex_proxy.agency import AgencyRuntime
from structlog.testing import capture_logs

_MSGS = [{"role": "system", "content": "agent sys"},
         {"role": "user", "content": "在线吗？"}]


@pytest.fixture(autouse=True)
def _modules_on(monkeypatch):
    monkeypatch.setenv("BLADEX_MODULE_LEDGER", "1")
    monkeypatch.setenv("BLADEX_MODULE_TOOLFACE", "1")


def _rt(*, with_ledger: bool = True) -> AgencyRuntime:
    rt = AgencyRuntime(index=None, hub=None)
    if with_ledger:
        led = Ledger(ledger_id="ldg-l84000000001", title="t", goal="g")
        rt.pool[led.ledger_id] = led
        rt.activation.restore({activation_scope("codex", ""): led.ledger_id},
                              sessions={activation_scope("codex", ""): "sess-A"})
    return rt


def _gated(cap) -> list[dict]:
    return [r for r in cap if r.get("event") == "agency_ledger_block_gated"]


def _only_reason(cap) -> str:
    rows = _gated(cap)
    assert len(rows) == 1, f"期望恰好一条 gated 日志，实际 {len(rows)}"
    assert rows[0]["agent"] == "codex"
    return rows[0]["reason"]


def test_gate1_auxiliary_is_logged():
    with capture_logs() as cap:
        out = _rt().insert_ledger_block([dict(m) for m in _MSGS], "codex", auxiliary=True,
                                        tier="strong", session_id="sess-A")
    assert out == _MSGS
    assert _only_reason(cap) == "auxiliary"


def test_gate2_tier_gate_still_logged_with_its_own_reasons():
    with capture_logs() as cap:
        out = _rt(with_ledger=False).insert_ledger_block(
            [dict(m) for m in _MSGS], "codex", tier="weak", session_id="sess-B")
    assert out == _MSGS
    assert _only_reason(cap) == "tier_not_allowed_no_ledger"


@pytest.mark.parametrize("env", ["BLADEX_MODULE_LEDGER", "BLADEX_MODULE_TOOLFACE"])
def test_gate3_module_off_is_logged(monkeypatch, env):
    monkeypatch.setenv(env, "0")
    with capture_logs() as cap:
        out = _rt().insert_ledger_block([dict(m) for m in _MSGS], "codex",
                                        tier="strong", session_id="sess-A")
    assert out == _MSGS
    assert _only_reason(cap) == "module_off"


def test_no_ledger_and_no_instruction_is_named_too():
    """门 ③ 同一跳里的另一格：无激活账本 + 不给指令 ⇒ 整块不注，也要具名。"""
    with capture_logs() as cap:
        out = _rt(with_ledger=False).insert_ledger_block(
            [dict(m) for m in _MSGS], "codex", with_instruction=False,
            tier="strong", session_id="sess-A")
    assert out == _MSGS
    assert _only_reason(cap) == "no_ledger_no_instruction"


def test_injected_turn_has_no_gated_log():
    """阴性对照：真注了块的轮次不打 gated（否则读数被稀释）。"""
    with capture_logs() as cap:
        out = _rt().insert_ledger_block([dict(m) for m in _MSGS], "codex",
                                        tier="strong", session_id="sess-A")
    assert len(out) == len(_MSGS) + 1
    assert _gated(cap) == []
