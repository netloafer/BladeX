"""MQ-L28：账本块注入的「档位 × 绑定来源会话」gate（2026-08-29 Jason 拍板）。

live 事故：Pi 新会话问「在线吗」，弱档（auto 集合外）无工具面但**照注只读
账本正文**，doubao-lite 读到昨天已完结的「北京天气查询」goal 后径直执行——
弱模型读得到任务、却没有工具去更新/关闭账本 ⇒ 每个新会话循环旧任务。

修法不是"集合外完全不注"（那会重新打开 C 方案堵的洞：同会话"继续"被路由
到弱档时账本失忆），而是绑定会话判据：
- 集合外档位：激活账本是**本会话**绑的 ⇒ 照注只读正文；继承/来源未知 ⇒ 不注；
- 集合内档位：不变（有工具 + 首步指令，错配由模型自行 switch）。
"""

from __future__ import annotations

import pytest

from bladex_core.ledger import Ledger, binding_sessions
from bladex_core.ledger_runtime import activation_scope
from bladex_proxy.agency import AgencyRuntime

_MSGS = [{"role": "system", "content": "agent sys"},
         {"role": "user", "content": "在线吗？"}]


@pytest.fixture(autouse=True)
def _modules_on(monkeypatch):
    """块注入前置：ledger/toolface 模块开（每请求 env 读，测试进程默认关）。"""
    monkeypatch.setenv("BLADEX_MODULE_LEDGER", "1")
    monkeypatch.setenv("BLADEX_MODULE_TOOLFACE", "1")


def _rt(bound_session: str | None = "sess-A") -> AgencyRuntime:
    rt = AgencyRuntime(index=None, hub=None)
    led = Ledger(ledger_id="ldg-weather00001", title="北京今日天气查询",
                 goal="获取北京今天（2026-08-27）的实时天气信息")
    rt.pool[led.ledger_id] = led
    scope = activation_scope("Pi", "")
    sessions = {scope: bound_session} if bound_session else None
    rt.activation.restore({scope: led.ledger_id}, sessions=sessions)
    return rt


def _injected(rt: AgencyRuntime, *, tier: str, session_id: str) -> bool:
    out = rt.insert_ledger_block([dict(m) for m in _MSGS], "Pi",
                                 tier=tier, session_id=session_id)
    return len(out) > len(_MSGS)


class TestTierSessionGate:
    def test_incident_shape_inherited_binding_weak_tier_not_injected(self):
        """案发形态：新会话（≠绑定来源会话）+ weak ⇒ 不注。"""
        assert not _injected(_rt("sess-A"), tier="weak", session_id="sess-B")

    def test_same_session_weak_keeps_c_scheme_continuity(self):
        """C 方案保留：本会话绑的账本 + weak ⇒ 照注（"继续"场景不失忆）。"""
        assert _injected(_rt("sess-A"), tier="weak", session_id="sess-A")

    def test_strong_tier_unchanged_model_adjudicates(self):
        """集合内档位不受判据影响——错配由模型经首步指令自行 switch。"""
        assert _injected(_rt("sess-A"), tier="strong", session_id="sess-B")
        assert _injected(_rt("sess-A"), tier="medium", session_id="sess-B")

    def test_unknown_binding_origin_treated_as_inherited(self):
        """来源未知（老事件无 session/未恢复）⇒ 按继承保守处理：weak 不注。
        方向拍板：误注旧任务的代价 > 弱档少注一轮上下文。"""
        assert not _injected(_rt(bound_session=None), tier="weak",
                             session_id="sess-B")

    def test_empty_tier_is_legacy_path_no_gate(self):
        """tier 未传（老调用面/测试）⇒ 不启 gate——渐进接线不破既有行为。"""
        assert _injected(_rt("sess-A"), tier="", session_id="sess-B")


class TestBindingSessionsRecovery:
    def test_pure_function_last_wins(self):
        evs = [
            {"type": "ledger_switch", "payload": {
                "agent_id": "Pi", "session_id": "s1",
                "from_ledger_id": "", "to_ledger_id": "ldg-a"}},
            {"type": "ledger_switch", "payload": {
                "agent_id": "Pi", "session_id": "s2",
                "from_ledger_id": "ldg-a", "to_ledger_id": "ldg-b"}},
            {"type": "ledger_update", "payload": {"ledger": {}}},   # 非 switch 不算
            {"type": "ledger_switch", "payload": {
                "agent_id": "codex", "to_ledger_id": "ldg-c"}},     # 无 session 跳过
        ]
        out = binding_sessions(evs)
        assert out == {activation_scope("Pi", ""): "s2"}

    def test_restore_roundtrip_feeds_last_session(self):
        rt = AgencyRuntime(index=None, hub=None)
        scope = activation_scope("Pi", "")
        rt.activation.restore({scope: "ldg-x"}, sessions={scope: "sess-A"})
        assert rt.activation.last_session(scope) == "sess-A", \
            "重启后绑定来源会话必须可恢复——否则 gate 全体退化成 unknown"


def test_server_passes_tier_and_session():
    """§3.2b 判据一致性：server 必须把同一个 tier 判定传给块注入
    （MQ-A18/P7 族防复发——同一开关两个消费点必须同源）。"""
    import os

    import bladex_proxy as _pkg
    from _source_probe import package_source
    src = package_source("server")   # F0.1 拆包：server.py → server/ 包，按包拼接读源码
    assert src.count("insert_ledger_block(") == 1, "单调用点"
    seg = src[src.index("insert_ledger_block("):]
    seg = seg[:seg.index(")") + 1] if ")" in seg else seg
    assert "tier=prep.route.tier" in seg and "session_id=identity.session_id" in seg


class TestSessionBoundGrantsLedgerTools:
    """🔴 MQ-L7 的补完（2026-08-30 Jason）：**只给正文不给工具不成立**。

    上面那组 gate 解决的是"该不该让它看见账本"。但 live 反馈：看见了、
    也发现任务不对，**却切不走**——集合外档位没有 `bladex_ledger_switch`，
    模型只能照着旧 Goal 继续做。那正是 MQ-L28 事故记录里写的第二半
    （"弱模型读得到任务、却没有工具去更新/关闭"），当时只修了第一半。

    修法：块注入 gate 的判据 `session_owns_active_ledger` **同时驱动账本
    工具族的放行**。三条边界各一格——补的是"本会话绑定"这一格的能力，
    另外两格（继承绑定 / 无账本）一个字不动。
    """

    def _tools(self, rt: AgencyRuntime, *, tier: str, session_id: str) -> list[str]:
        tools, _ = rt.augment_tools(None, agent_id="Pi", auxiliary=False,
                                    tier=tier, session_id=session_id)
        return [(t.get("function") or t).get("name", "") for t in (tools or [])]

    def test_same_session_weak_now_gets_ledger_tools(self):
        """案发形态的正解：本会话绑的账本 + weak ⇒ 正文**和**工具都给。"""
        rt = _rt("sess-A")
        names = self._tools(rt, tier="weak", session_id="sess-A")
        assert "bladex_ledger_switch" in names, "看得见账本却切不走 = 事故复现"
        assert "bladex_ledger_update" in names
        assert "bladex_memory_search" in names
        # 正文那一半仍然照注（C 方案不变）
        assert _injected(rt, tier="weak", session_id="sess-A")

    def test_inherited_binding_weak_still_gets_no_ledger_tools(self):
        """🔴 gate 一个字没动：继承来的绑定仍然既不注正文、也不给账本工具。

        这一格是 MQ-L28 事故的**病因本身**（跨会话继承旧任务），
        放宽它就是把那个事故请回来。记忆工具照给——它与账本无关。
        """
        rt = _rt("sess-A")
        names = self._tools(rt, tier="weak", session_id="sess-B")
        assert not [n for n in names if n.startswith("bladex_ledger_")]
        assert "bladex_memory_search" in names
        assert not _injected(rt, tier="weak", session_id="sess-B")

    def test_no_active_ledger_weak_gets_no_ledger_tools(self):
        """无账本 + weak ⇒ 不给账本工具（不诱导弱模型凭空建本）。"""
        rt = AgencyRuntime(index=None, hub=None)
        names = self._tools(rt, tier="weak", session_id="sess-A")
        assert not [n for n in names if n.startswith("bladex_ledger_")]

    def test_capability_not_instruction(self):
        """🔴 补的是**能力**不是催促：工具给了，首步指令仍按档位不给。

        一次只动一个变量——弱模型误调用的风险还没有读数
        （MQ-L6 那次 43% 调用率是在**有首步指令**的 all 配置下测的）。
        若观察下来 weak 档拿到工具也不主动切，再补指令，那时是另一个变量。
        """
        rt = _rt("sess-A")
        assert "bladex_ledger_switch" in self._tools(rt, tier="weak",
                                                     session_id="sess-A")
        out = rt.insert_ledger_block([dict(m) for m in _MSGS], "Pi",
                                     with_instruction=True, tier="weak",
                                     session_id="sess-A")
        assert "bladex_ledger_switch" not in out[-1]["content"], \
            "首步指令不该随工具一起给——那是第二个变量"

    def test_both_consumers_read_one_predicate(self):
        """判据只有一处定义：块注入 gate 与工具族放行读同一个方法。

        两边各拼一遍 `bound == session_id` 就是"同一件事两个消费点各判一次"，
        本仓刚性原则 9/12 的来源；这条把它钉成结构断言。
        """
        import ast
        from pathlib import Path
        # F0.1 拆包：两个消费点（augment_tools / insert_ledger_block）都在 agency/runtime.py
        src = Path(__file__).resolve().parents[1] / "bladex_proxy" / "agency" / "runtime.py"
        tree = ast.parse(src.read_text(encoding="utf-8"))
        callers = [
            fn.name for fn in ast.walk(tree)
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
            and any(isinstance(n, ast.Attribute)
                    and n.attr == "session_owns_active_ledger"
                    for n in ast.walk(fn))
        ]
        assert set(callers) >= {"augment_tools", "insert_ledger_block"}, (
            f"只有 {callers} 用了这个判据——另一个消费点要么没接、要么自己拼了一遍")
