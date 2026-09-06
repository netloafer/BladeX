"""账本并发批（2026-08-29 Jason 拍板：「两个 coding agent 同时激活同一账本」）。

三件事：① rev 乐观锁——模型带读到的 rev，落后即拒写并返回最新正文（合并者
是 agent 侧模型，BladeX 只保证不静默覆盖）；② 并发可见——另一写者在窗口内
动过 ⇒ 注入头提示一行；③ 作用域接线——project_id 从 server 贯通到注入面与
工具面（08-26 键已是 (agent, project)，此前信号未接全落 Global）。
"""

from __future__ import annotations

import asyncio
import datetime as dt
import os

import pytest

from bladex_core.ledger import Ledger, render_ledger_md
from bladex_core.ledger_runtime import activation_scope
from bladex_proxy.agency import AgencyRuntime


@pytest.fixture(autouse=True)
def _modules_on(monkeypatch):
    monkeypatch.setenv("BLADEX_MODULE_LEDGER", "1")
    monkeypatch.setenv("BLADEX_MODULE_TOOLFACE", "1")


def _now_iso(minutes_ago: float = 0.0) -> str:
    return (dt.datetime.now(dt.timezone.utc)
            - dt.timedelta(minutes=minutes_ago)).isoformat(timespec="seconds")


def _rt(led: Ledger, scope: str) -> AgencyRuntime:
    rt = AgencyRuntime(index=None, hub=None)
    rt.pool[led.ledger_id] = led
    rt.activation.restore({scope: led.ledger_id})
    return rt


def _led(**kw) -> Ledger:
    base = dict(ledger_id="ldg-rev00000001", title="并发测试",
                goal="测试 rev", updated_at=_now_iso())
    base.update(kw)
    return Ledger(**base)


def _update(rt, args, agent="codex", project=""):
    ctx = {"agent_id": agent, "project_id": project, "session_id": "s1"}
    return asyncio.run(
        rt._h_ledger_update(args, allowed_exposure="local", context=ctx))


class TestRevLock:
    def test_render_shows_rev(self):
        assert "- rev: 3" in render_ledger_md(_led(rev=3)), \
            "rev 必须渲染——模型读不到就回带不了"

    def test_stale_rev_rejected_with_latest_content(self):
        led = _led(rev=5)
        rt = _rt(led, activation_scope("codex", ""))
        out = _update(rt, {"section": "verified", "op": "add",
                           "text": "x", "rev": 3})
        assert out.startswith("Error: ledger changed"), out
        assert "current 5" in out and "- rev: 5" in out, \
            "拒写必须附最新正文（含当前 rev），模型据此合并重试"
        assert rt.pool[led.ledger_id].rev == 5, "拒写不动池"

    def test_current_rev_accepted_and_bumped(self):
        led = _led(rev=5)
        rt = _rt(led, activation_scope("codex", ""))
        out = _update(rt, {"section": "verified", "op": "add",
                           "text": "落库成功", "rev": 5})
        assert out.startswith("added"), out
        new = rt.pool[led.ledger_id]
        assert new.rev == 6 and new.last_writer == "codex"

    def test_missing_rev_is_legacy_path(self):
        led = _led(rev=5)
        rt = _rt(led, activation_scope("codex", ""))
        out = _update(rt, {"section": "open", "op": "add", "text": "老调用面"})
        assert out.startswith("added"), "不带 rev = 旧调用面，不拦（渐进接线）"
        assert rt.pool[led.ledger_id].rev == 6, "但写成功仍 bump"

    def test_interleave_remove_is_what_the_lock_is_for(self):
        """案发形态：A 读到 rev=1 后 B add 一条（rev→2），A 按旧 index remove
        必须被拦——这正是按旧视图错删的那一刀。"""
        led = _led(rev=1)
        rt = _rt(led, activation_scope("codex", ""))
        # 题面形态：两个 agent 各自的作用域绑同一本（账本池化的交接语义）
        rt.activation.restore(
            {activation_scope("claude-code", ""): led.ledger_id})
        assert _update(rt, {"section": "next", "op": "add", "text": "B 的新条目",
                            "rev": 1}, agent="claude-code").startswith("added")
        out = _update(rt, {"section": "next", "op": "remove", "index": 0,
                           "rev": 1}, agent="codex")
        assert out.startswith("Error: ledger changed")


class TestCoauthorNote:
    def _msg(self, rt, agent="codex"):
        return rt.ledger_injection_message(agent, with_instruction=False)

    def test_other_recent_writer_annotated(self):
        led = _led(last_writer="claude-code", updated_at=_now_iso(5))
        rt = _rt(led, activation_scope("codex", ""))
        msg = self._msg(rt)
        assert msg and "also being updated by claude-code" in msg["content"]

    def test_own_writes_not_annotated(self):
        led = _led(last_writer="codex", updated_at=_now_iso(5))
        rt = _rt(led, activation_scope("codex", ""))
        msg = self._msg(rt)
        assert msg and "also being updated" not in msg["content"]

    def test_old_writes_outside_window_not_annotated(self):
        led = _led(last_writer="claude-code", updated_at=_now_iso(60))
        rt = _rt(led, activation_scope("codex", ""))
        msg = self._msg(rt)
        assert msg and "also being updated" not in msg["content"]

    def test_zero_window_disables(self, monkeypatch):
        monkeypatch.setenv("BLADEX_LEDGER_COAUTHOR_WINDOW_S", "0")
        led = _led(last_writer="claude-code", updated_at=_now_iso(1))
        rt = _rt(led, activation_scope("codex", ""))
        msg = self._msg(rt)
        assert msg and "also being updated" not in msg["content"]


class TestProjectScopeWiring:
    def test_update_lands_on_project_scope(self):
        """同 agent 两个项目 = 两个作用域，各自的激活账本互不干扰。"""
        led_a = _led(ledger_id="ldg-projA0000001")
        led_b = _led(ledger_id="ldg-projB0000001")
        rt = AgencyRuntime(index=None, hub=None)
        rt.pool[led_a.ledger_id] = led_a
        rt.pool[led_b.ledger_id] = led_b
        rt.activation.restore({
            activation_scope("codex", "p-aaa"): led_a.ledger_id,
            activation_scope("codex", "p-bbb"): led_b.ledger_id})
        out = _update(rt, {"section": "verified", "op": "add", "text": "A 项目"},
                      project="p-aaa")
        assert out.startswith("added")
        assert rt.pool[led_a.ledger_id].rev == 1
        assert rt.pool[led_b.ledger_id].rev == 0, "B 项目的账本纹丝不动"

    def test_server_threads_project_id_all_entries(self):
        """§3.2b 判据一致性：**全部**账本消费入口必须吃 identity.project_id。

        🔴 live 教训（2026-08-29 首验）：首版守卫只数了 process_message 两处，
        漏了三条**流式**拦截入口（intercept_chat/anthropic/responses_stream）——
        codex 走流式，切换绑定落 codex@Global 而注入查 codex@p-10ad1f，
        账本在切换成功后对注入面隐形。守卫窄一寸，缺陷就从那一寸过。"""
        import re
        import bladex_proxy as _pkg
        with open(os.path.join(os.path.dirname(_pkg.__file__), "server.py"),
                  encoding="utf-8") as f:
            src = f.read()
        entries = ["insert_ledger_block(", "process_message(",
                   "intercept_chat_stream(", "intercept_anthropic_stream(",
                   "intercept_responses_stream("]
        for name in entries:
            hits = [m.start() for m in re.finditer(re.escape(name), src)]
            assert hits, f"{name} 调用点消失（结构变了要同步改守卫）"
            for i, idx in enumerate(hits):
                assert "project_id=identity.project_id" in src[idx:idx + 600], \
                    f"{name} 第 {i + 1} 处调用未传 project_id"

    def test_agency_every_ctx_site_carries_project(self):
        """agency 侧对账：每个工具执行 ctx 生产点都必须带 project_id——
        入口再加一个也逃不过（生产点判据，不数入口名单）。"""
        import bladex_proxy as _pkg
        with open(os.path.join(os.path.dirname(_pkg.__file__), "agency.py"),
                  encoding="utf-8") as f:
            src = f.read()
        # 2026-09-04 MQ-L40：三处字面量收成 `tool_context(...)` 单实现点（两条流式路径曾漏传
        # turn_index）。project_id 是它的**必填关键字参数**，漏传在调用处就是 TypeError；
        # 本守卫改为：生产点全部走 tool_context，且旧字面量形态不许再长出来。
        sites = src.count("ctx = tool_context(")
        assert sites >= 3, "ctx 生产点少于已知数（结构变了要同步改守卫）"
        assert src.count('ctx = {"session_id": session_id') == 0, \
            "又出现了手写 ctx 字面量——漏传 project_id / turn_index 就是这么来的"
        assert "def tool_context(*, session_id: str, agent_id: str, project_id: str," in src
        assert src.count("project_id=project_id") >= sites


class TestLazyScopeMigration:
    """惰性作用域迁移（08-29 收尾拍板）：项目键 miss ∧ Global hit ∧ 有项目
    信号 ⇒ 搬。实测依据：自发自愈 0/2、codex desktop 指纹跨天不换。"""

    def _rt_global_bound(self, agent="codex"):
        led = _led(ledger_id="ldg-legacy0000001")
        rt = AgencyRuntime(index=None, hub=None)
        rt.pool[led.ledger_id] = led
        rt.activation.restore({activation_scope(agent, ""): led.ledger_id})
        return rt, led

    def test_tool_path_migrates_and_finds_ledger(self):
        rt, led = self._rt_global_bound()
        out = _update(rt, {"section": "verified", "op": "add", "text": "迁移后可写"},
                      project="p-10ad1f61d0c2")
        assert out.startswith("added"), out
        assert rt.activation.active(
            activation_scope("codex", "p-10ad1f61d0c2")) == led.ledger_id
        assert rt.activation.active(activation_scope("codex", "")) == "", \
            "搬 = Global 侧清除，不是复制"

    def test_injection_path_migrates(self):
        rt, led = self._rt_global_bound()
        out = rt.insert_ledger_block(
            [dict(m) for m in _MSGS_STUB], "codex",
            project_id="p-10ad1f61d0c2")
        assert any(led.ledger_id in (m.get("content") or "") for m in out), \
            "迁移后注入面立即看见账本（正是 live 卡死的对偶）"

    def test_no_signal_no_migration(self):
        rt, led = self._rt_global_bound()
        rt.insert_ledger_block([dict(m) for m in _MSGS_STUB], "codex",
                               project_id="")
        assert rt.activation.active(
            activation_scope("codex", "")) == led.ledger_id, \
            "无项目信号（Pi/hermes 形态）零触碰"

    def test_project_already_bound_wins(self):
        rt, led = self._rt_global_bound()
        led2 = _led(ledger_id="ldg-projown00001")
        rt.pool[led2.ledger_id] = led2
        rt.activation.restore(
            {activation_scope("codex", "p-x"): led2.ledger_id})
        rt.insert_ledger_block([dict(m) for m in _MSGS_STUB], "codex",
                               project_id="p-x")
        assert rt.activation.active(
            activation_scope("codex", "p-x")) == led2.ledger_id
        assert rt.activation.active(
            activation_scope("codex", "")) == led.ledger_id, \
            "项目键已有绑定 ⇒ 不迁移不清 Global"

    def test_migration_emits_replayable_event(self):
        events = []
        rt, led = self._rt_global_bound()
        rt._emit = lambda t, p: events.append(
            {"type": t.value, "payload": p})
        rt.insert_ledger_block([dict(m) for m in _MSGS_STUB], "codex",
                               project_id="p-abc")
        assert events and events[0]["type"] == "ledger_scope_migrate"
        # 重放等价：switch@Global + migrate ⇒ 只剩项目键
        from bladex_core.ledger import binding_sessions, replay_ledger_events
        stream = [{"type": "ledger_switch", "payload": {
                       "agent_id": "codex", "session_id": "s0",
                       "to_ledger_id": led.ledger_id}},
                  events[0]]
        _, bindings = replay_ledger_events(stream)
        assert bindings == {
            activation_scope("codex", "p-abc"): led.ledger_id}
        # MQ-L28 保守：迁移后来源会话作废（dst 不设 ⇒ weak 档按继承不注）
        assert binding_sessions(stream) == {}


_MSGS_STUB = [{"role": "system", "content": "sys"},
              {"role": "user", "content": "干活"}]


class TestUserEditAdoption:
    def test_adoption_bumps_rev_and_marks_user(self):
        """直编采纳=整本覆盖（用户主权），但 rev 前进、写者记 user——
        并发 agent 的 rev 锁由此能拦住基于被盖版本的写。"""
        prev = _led(rev=4, last_writer="codex")
        adopted = prev.model_copy(update={"goal": "用户改过的 goal"})
        # 复刻 server 采纳逻辑的核心变换（行为等价断言在源码守卫里）
        adopted = adopted.model_copy(update={"rev": prev.rev + 1,
                                             "last_writer": "user"})
        assert adopted.rev == 5 and adopted.last_writer == "user"

    def test_server_adoption_source_guard(self):
        import bladex_proxy as _pkg
        with open(os.path.join(os.path.dirname(_pkg.__file__), "server.py"),
                  encoding="utf-8") as f:
            src = f.read()
        assert "admin_ledger_user_edit_clobber_risk" in src, \
            "并发覆盖必须响亮告警，不静默（刚性原则 13 同族）"
        i = src.index("admin_ledger_user_edit_clobber_risk")
        assert '"rev": prev.rev + 1' in src[i - 800:i], "采纳必须 bump rev"
