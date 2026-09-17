"""F-B3 的接线半边：写者从 `context["agent_id"]` 通到条目上（工具写入 + 建本初始条目）。

单测绿而接线断已是本仓多次踩过的形状（"单测过 ≠ 接线通"）。字段在 schema 里、
生产者断线 ⇒ 读数恒为默认值，静态守卫永远看不见它（原则 13 层 3 的形状）。
故本文件走 `AgencyRuntime` 的真实 handler，断言落在**池里那本账本**上。
"""

from __future__ import annotations

import asyncio

import pytest

from bladex_core.ledger import Ledger, render_ledger_md
from bladex_core.ledger_runtime import activation_scope
from bladex_proxy.agency import AgencyRuntime


@pytest.fixture(autouse=True)
def _modules_on(monkeypatch):
    monkeypatch.setenv("BLADEX_MODULE_LEDGER", "1")
    monkeypatch.setenv("BLADEX_MODULE_TOOLFACE", "1")


def _rt() -> AgencyRuntime:
    return AgencyRuntime(index=None, hub=None)


def _call(rt, fn, args, agent="claude-code"):
    return asyncio.run(fn(args, allowed_exposure="local", context={
        "agent_id": agent, "project_id": "", "session_id": "s1",
        "user_query": "做点事", "turn_index": 1}))


class TestUpdateCarriesWriter:
    def test_added_entry_carries_the_calling_agent(self):
        rt = _rt()
        led = Ledger(ledger_id="ldg-w1", title="t")
        rt.pool[led.ledger_id] = led
        rt.activation.restore({activation_scope("claude-code", ""): led.ledger_id})
        _call(rt, rt._h_ledger_update,
              {"section": "verified", "op": "add", "text": "gate 全绿",
               "ref": "gate#1"})
        e = rt.pool[led.ledger_id].entries("verified")[0]
        assert e.writer == "claude-code"
        assert "@claude-code" in render_ledger_md(rt.pool[led.ledger_id])

    def test_batch_entries_all_carry_it(self):
        rt = _rt()
        led = Ledger(ledger_id="ldg-w2", title="t")
        rt.pool[led.ledger_id] = led
        rt.activation.restore({activation_scope("hermes:default", ""):
                               led.ledger_id})
        _call(rt, rt._h_ledger_update, {"entries": [
            {"section": "verified", "op": "add", "text": "A"},
            {"section": "next", "op": "add", "text": "B"},
        ]}, agent="hermes:default")
        new = rt.pool[led.ledger_id]
        assert [e.writer for e in new.entries("verified")] == ["hermes:default"]
        assert [e.writer for e in new.entries("next")] == ["hermes:default"]

    def test_two_agents_on_one_ledger_stay_distinguishable(self):
        """轴 C 的题面：同一本账本上两个 agent 各写一条，条目级要分得开
        （账本级 `last_writer` 只剩最后那个）。"""
        rt = _rt()
        led = Ledger(ledger_id="ldg-w3", title="t")
        rt.pool[led.ledger_id] = led
        for agent in ("claude-code", "hermes:default"):
            rt.activation.restore({activation_scope(agent, ""): led.ledger_id})
            _call(rt, rt._h_ledger_update,
                  {"section": "verified", "op": "add", "text": f"{agent} 写的"},
                  agent=agent)
        new = rt.pool[led.ledger_id]
        assert [e.writer for e in new.entries("verified")] == \
            ["claude-code", "hermes:default"]
        assert new.last_writer == "hermes:default", \
            "账本级仍只记最后一个——这正是需要条目级的理由"


class TestSwitchCarriesWriter:
    def test_initial_entries_of_a_new_ledger_carry_the_creator(self):
        rt = _rt()
        out = _call(rt, rt._h_ledger_switch, {
            "ledger_id": "", "title": "新任务", "goal": "把事做完",
            "core": ["仓库在 ~/dev/BladeX"], "next": ["先跑 gate"]})
        lid = out.rsplit(" ", 1)[-1].rstrip(".")
        led = rt.pool[lid]
        assert [e.writer for e in led.entries("core")] == ["claude-code"]
        assert [e.writer for e in led.entries("next")] == ["claude-code"]

    def test_sub_task_entries_on_both_sides_carry_the_creator(self):
        rt = _rt()
        parent = _call(rt, rt._h_ledger_switch,
                       {"ledger_id": "", "title": "父任务"})
        pid = parent.rsplit(" ", 1)[-1].rstrip(".")
        child = _call(rt, rt._h_ledger_switch,
                      {"ledger_id": "", "title": "子任务",
                       "parent_ledger_id": pid}, agent="codex")
        cid = child.rsplit(" ", 1)[-1].rstrip(".")
        # 子侧的回写义务条目 + 父侧的等待条目，两边都要有写者
        assert rt.pool[cid].entries("next")[0].writer == "codex"
        assert rt.pool[pid].entries("next")[0].writer == "codex"
