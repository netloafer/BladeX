"""MQ-L80 / 执行卡 G19.2 —— 工具面的每一条拒绝都要在日志里可判定（零行为）。

live 病例（`docs/reviews/dsh-ledger-unreachable-20260920.md` §4）：00:27:57 模型
确实调了 `bladex_ledger_switch`（`agency_stream_mixed_stripped names=…`），但同一时刻
**没有** `hub_admin_event_appended event_type=ledger_switch` ⇒ 它失败了。走的是
`_h_ledger_switch` 的 unknown-ledger 分支 —— 而**同一个函数**的防抖分支走
`logger.warning`。后果：切换失败率与失败成因（猜 id / 已删 / 模块关）在存量日志里
全部不可判定，只能靠 `mixed_stripped` 反推、且反推不出目标 id。
与刚性原则 13「静默丢数据比丢数据更糟」同族 —— 这里丢的是**失败事件本身**。

🔴 修法不是「补撞见的那一条」，是**全扫**（刚性原则 12 的变体）。扫描结果见
`docs/planning/report-batch-G19-20260920.md` §G19.2。

判据：① 单测驱动 unknown id ⇒ 日志有 `agency_switch_unknown_ledger` 且带 `target`；
      ② 阴性对照 = 删掉日志行 ⇒ 下面的结构守卫必红（它按 AST 逐条查，不是数字符串）。
"""

from __future__ import annotations

import ast
import asyncio

import structlog.testing
from bladex_core.ledger import new_ledger
from bladex_proxy.agency import AgencyRuntime

_AGENT = "test-agent"


def _agency(monkeypatch) -> AgencyRuntime:
    for mod in ("LEDGER", "TOOLFACE", "INTERCEPTION"):
        monkeypatch.setenv(f"BLADEX_MODULE_{mod}", "1")
    a = AgencyRuntime()
    assert a.ledger_on and a.toolface_on
    a.pool["ldg-real0000001"] = new_ledger(
        ledger_id="ldg-real0000001", title="真账本", goal="做完这件事",
        goal_source="user", created_at="2026-09-01T10:00:00Z")
    return a


def _ctx() -> dict:
    return {"agent_id": _AGENT, "session_id": "s1", "project_id": "",
            "turn_index": 9, "user_query": "切到那本 Mac 的账本"}


class TestSwitchFailureIsVisible:
    """① 模型猜了一个不存在的 id ⇒ 日志说得出是谁、想切到哪、池里有多少本。"""

    def test_unknown_ledger_id_is_logged(self, monkeypatch):
        a = _agency(monkeypatch)
        with structlog.testing.capture_logs() as cap:
            out = asyncio.run(a._h_ledger_switch(
                {"ledger_id": "ldg-497d5e2d542b"},
                allowed_exposure="local", context=_ctx()))
        assert out.startswith("Error: unknown ledger")
        ev = next((e for e in cap
                   if e.get("event") == "agency_switch_unknown_ledger"), None)
        assert ev is not None, cap
        assert ev["target"] == "ldg-497d5e2d542b"
        assert ev["pool_size"] == 1 and ev["agent"] == _AGENT
        assert ev["scope"] and ev["log_level"] == "warning"

    def test_unknown_parent_is_logged(self, monkeypatch):
        a = _agency(monkeypatch)
        with structlog.testing.capture_logs() as cap:
            out = asyncio.run(a._h_ledger_switch(
                {"ledger_id": "", "title": "新任务", "parent_ledger_id": "ldg-nope"},
                allowed_exposure="local", context=_ctx()))
        assert out.startswith("Error: unknown parent_ledger_id")
        ev = next((e for e in cap
                   if e.get("event") == "agency_ledger_switch_refused"), None)
        assert ev is not None and ev.get("reason") == "unknown_parent", cap
        assert ev["target"] == "ldg-nope"

    def test_module_off_is_logged(self, monkeypatch):
        a = _agency(monkeypatch)
        monkeypatch.setattr(type(a), "ledger_on", property(lambda self: False))
        with structlog.testing.capture_logs() as cap:
            out = asyncio.run(a._h_ledger_switch(
                {"ledger_id": "x"}, allowed_exposure="local", context=_ctx()))
        assert out.startswith("Error: ledger module is disabled")
        assert any(e.get("event") == "agency_ledger_switch_refused"
                   and e.get("reason") == "module_off" for e in cap), cap

    def test_unknown_ledger_read_is_logged(self, monkeypatch):
        a = _agency(monkeypatch)
        with structlog.testing.capture_logs() as cap:
            out = asyncio.run(a._h_ledger_read(
                {"ledger_id": "ldg-nope"}, allowed_exposure="local", context=_ctx()))
        assert out.startswith("Error: unknown ledger")
        ev = next((e for e in cap
                   if e.get("event") == "agency_ledger_read_refused"), None)
        assert ev is not None and ev.get("reason") == "unknown_ledger", cap
        assert ev["target"] == "ldg-nope"

    def test_memory_search_without_selector_is_logged(self, monkeypatch):
        a = _agency(monkeypatch)
        with structlog.testing.capture_logs() as cap:
            out = asyncio.run(a._h_memory_search(
                {}, allowed_exposure="local", context=_ctx()))
        assert out.startswith("Error:")
        assert any(e.get("event") == "agency_memory_search_refused" for e in cap), cap


class TestEveryErrorBranchIsLogged:
    """② 结构守卫（**阴性对照的载体**）：按 AST 逐条查，不是数字符串。

    判据：每一个返回 `"Error: …"` 的 `return`，在**同一个语句列表**里紧挨着它的
    上一条语句必须是 `logger.warning(...)`。删掉任何一条日志，本用例立即指名道姓地红。

    🔴 为什么钉"紧挨着"而不是"这个函数里有日志"：`_h_ledger_switch` 修前就有
    `agency_switch_debounced`，按函数粒度查它是绿的 —— 而 unknown-ledger 那条
    恰恰没有。粒度错了的守卫比没有守卫更坏（它会报绿）。
    """

    @staticmethod
    def _error_returns() -> list[tuple[int, bool]]:
        import bladex_proxy.agency.handlers as h

        src = __import__("pathlib").Path(h.__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)
        out: list[tuple[int, bool]] = []

        def _is_error_return(node: ast.Return) -> bool:
            for sub in ast.walk(node):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str) \
                        and sub.value.startswith("Error:"):
                    return True
            return False

        def _is_log(stmt: ast.stmt) -> bool:
            return (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
                    and isinstance(stmt.value.func, ast.Attribute)
                    and stmt.value.func.attr in ("warning", "error")
                    and isinstance(stmt.value.func.value, ast.Name)
                    and stmt.value.func.value.id == "logger")

        for node in ast.walk(tree):
            for field in ("body", "orelse", "finalbody"):
                stmts = getattr(node, field, None)
                if not isinstance(stmts, list):
                    continue
                for i, stmt in enumerate(stmts):
                    if isinstance(stmt, ast.Return) and _is_error_return(stmt):
                        out.append((stmt.lineno, i > 0 and _is_log(stmts[i - 1])))
        return out

    def test_every_error_return_has_a_log_right_above_it(self):
        rows = self._error_returns()
        assert len(rows) >= 11, f"扫到 {len(rows)} 条，少于登记的 11 条"
        missing = [ln for ln, ok in rows if not ok]
        assert not missing, f"handlers.py 这些行的 Error 返回没有日志: {missing}"
