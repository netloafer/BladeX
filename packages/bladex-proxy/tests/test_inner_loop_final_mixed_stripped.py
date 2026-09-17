"""MQ-L45 · 内循环的最终回复若是 MIXED，`bladex_*` 调用被原样转发给 agent（2026-09-05）。

## 病例（Hub 取证，不是推测）

Pi 说「我先切到最新那份泰山啤酒账本」→ 同一条 assistant 消息里
`tool_calls=['bladex_ledger_switch','bladex_memory_search','bash']` →
Hub 里 agent 侧的下一轮 `request_messages` 明明白白记着：

    [6] assistant tool_calls=['bladex_ledger_switch','bladex_memory_search','bash']
    [7] tool: Tool bladex_ledger_switch not found
    [8] tool: Tool bladex_memory_search not found

agent 由此判定「BladeX 工具在本会话不可用」，转去 `grep -rl "泰山啤酒" ~/.hermes ~/.dsh ~/.pi`
翻文件系统找账本存档。**账本没切成，跨 agent 交接失败。**
同型至少 3 条 assistant 消息 / 4 个调用（09-05 12:12 与 13:29 两个会话）。

## 根因：一条契约写了、三个接收方都没接

`innerloop.run_inner_loop` 在 LLM 回复为 MIXED 时原样返回，注释写着
「mixed 的剥离-拼接**归调用方**」；而三条调用方（非流式 `process_message` /
chat 流式 `intercept_chat_stream` / 协议流式 `_intercept_protocol_stream`）一条都没剥。
协议流式那条甚至把这件事写进了注释「而这条路径没做。先量再修」——**量到了，没修**
（刚性原则 12），而且**仪器装在了漏得最少的那条路上**：chat 流式的
`final_tool_calls` 只记数量不记名字，16 次 `>0` 里哪几次是泄漏读不出来。

## 为什么时好时坏（本组的判别力来源）

`bladex_*` 在模型**首条回复**里 ⇒ 走 `process_message` 的 MIXED 分支，剥离**正确**；
在**内循环某一轮之后的回复**里 ⇒ 走本缺陷。⇒ 模型"先 read 再 switch"这个
**本该被鼓励**的行为恰好触发缺陷——所以阳性用例必须造成"先 read、再同轮 switch + 自带工具"。
"""
from __future__ import annotations

import asyncio
import json

import pytest

from bladex_core.ledger import new_ledger
from bladex_proxy.agency import AgencyRuntime

_SYS = {"role": "system", "content": "You are a test agent."}
_MSGS = [_SYS, {"role": "user", "content": "泰山啤酒那件事进展到哪了"}]


def _on(monkeypatch):
    monkeypatch.setenv("BLADEX_MODULE_TOOLFACE", "1")
    monkeypatch.setenv("BLADEX_MODULE_LEDGER", "1")
    monkeypatch.setenv("BLADEX_MODULE_INTERCEPTION", "1")


def _call(name: str, args: dict, cid: str) -> dict:
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def _seed(ag: AgencyRuntime):
    for lid, t in (("ldg-a", "任务 A"), ("ldg-b", "泰山啤酒破产重整")):
        ag.pool[lid] = new_ledger(title=t, goal=t, ledger_id=lid,
                                  created_at="2026-09-01T00:00:00+00:00")


def _run(ag: AgencyRuntime, first: dict, replies: list[dict]):
    """first = 模型的首条回复；replies = 内循环里每次 call_llm 的返回（按序）。"""
    seq = list(replies)

    async def call_llm(_msgs):
        return seq.pop(0)

    return asyncio.run(ag.process_message(
        first, upstream_messages=_MSGS, session_prefix="p/",
        allowed_exposure="public", call_llm=call_llm,
        session_id="s1", agent_id="Pi", turn_index=2))


# ── 阳性：内循环最终回复是 MIXED（模型先 read、再同轮 switch + 自带工具）──────────

def test_bladex_calls_in_loop_final_are_never_forwarded(monkeypatch):
    """🔴 本组的核心：转发给 agent 的消息里不许出现任何 `bladex_*` 调用。"""
    _on(monkeypatch)
    ag = AgencyRuntime()
    _seed(ag)
    first = {"role": "assistant", "content": None,
             "tool_calls": [_call("bladex_ledger_read", {"ledger_id": "ldg-b"}, "c1")]}
    loop_final = {"role": "assistant", "content": "我先切到泰山那本",
                  "tool_calls": [_call("bladex_ledger_switch", {"ledger_id": "ldg-b"}, "c2"),
                                 _call("bash", {"cmd": "date"}, "c3")]}
    out, transcript, mode = _run(ag, first, [loop_final])

    names = [(t.get("function") or {}).get("name") for t in (out.get("tool_calls") or [])]
    assert not [n for n in names if str(n).startswith("bladex_")], (
        f"bladex 调用被转发给 agent 了 —— agent 会报 'Tool ... not found'：{names}")
    assert names == ["bash"], f"agent 自己的调用必须原样保留：{names}"
    assert "我先切到泰山那本" in str(out.get("content") or ""), "正文不该被剥掉"


def test_the_stripped_call_is_actually_executed(monkeypatch):
    """判别力对照之一：光"不转发"不够——**丢掉**同样能让上一条绿。
    被剥的 switch 必须真的执行（激活账本要变），否则模型以为切了、实际没切
    （那正是 C6c 修过的"把没切说成切了"的另一半）。"""
    _on(monkeypatch)
    ag = AgencyRuntime()
    _seed(ag)
    scope = ag.scope_of("Pi")
    first = {"role": "assistant", "content": None,
             "tool_calls": [_call("bladex_ledger_read", {"ledger_id": "ldg-b"}, "c1")]}
    loop_final = {"role": "assistant", "content": "切过去",
                  "tool_calls": [_call("bladex_ledger_switch", {"ledger_id": "ldg-b"}, "c2"),
                                 _call("bash", {"cmd": "date"}, "c3")]}
    _run(ag, first, [loop_final])
    assert ag.activation.active(scope) == "ldg-b", "被剥的 switch 没有被执行 = 静默丢弃"


def test_stripped_calls_and_results_are_spliced_for_replay(monkeypatch):
    """判别力对照之二：剥掉并执行之后，调用与结果必须进拼接台账——
    否则 agent 回传这条 assistant 消息时，模型的下一轮就看不到自己切过账本
    （V-P4 剥离-拼接的全部理由）。"""
    _on(monkeypatch)
    ag = AgencyRuntime()
    _seed(ag)
    first = {"role": "assistant", "content": None,
             "tool_calls": [_call("bladex_ledger_read", {"ledger_id": "ldg-b"}, "c1")]}
    loop_final = {"role": "assistant", "content": "切过去",
                  "tool_calls": [_call("bladex_ledger_switch", {"ledger_id": "ldg-b"}, "c2"),
                                 _call("bash", {"cmd": "date"}, "c3")]}
    out, transcript, _mode = _run(ag, first, [loop_final])
    assert ag.splice.new_in_turn("p/"), "没记拼接 ⇒ 下一轮模型看不到自己切过"
    tool_msgs = [m for m in transcript if m.get("role") == "tool"]
    assert tool_msgs, "transcript 里要有被剥调用的执行结果（入 Hub 的那一份）"


# ── 阴性对照：不该动的两种形态 ──────────────────────────────────────────────

def test_pure_agent_calls_in_loop_final_pass_through_untouched(monkeypatch):
    """内循环最终回复只带 agent 自己的调用 ⇒ 逐字转发，一个都不许少。"""
    _on(monkeypatch)
    ag = AgencyRuntime()
    _seed(ag)
    first = {"role": "assistant", "content": None,
             "tool_calls": [_call("bladex_ledger_read", {"ledger_id": "ldg-b"}, "c1")]}
    loop_final = {"role": "assistant", "content": "去查一下",
                  "tool_calls": [_call("bash", {"cmd": "date"}, "c2"),
                                 _call("read", {"path": "/tmp/x"}, "c3")]}
    out, _t, _m = _run(ag, first, [loop_final])
    names = [(t.get("function") or {}).get("name") for t in (out.get("tool_calls") or [])]
    assert names == ["bash", "read"], f"agent 调用被剥掉了：{names}"
    assert not ag.splice.new_in_turn("p/"), "没有 bladex 调用就不该记拼接"


def test_loop_final_without_tool_calls_is_unchanged(monkeypatch):
    """最常见的形态（内循环答完就收）：消息逐字不变。"""
    _on(monkeypatch)
    ag = AgencyRuntime()
    _seed(ag)
    first = {"role": "assistant", "content": None,
             "tool_calls": [_call("bladex_ledger_read", {"ledger_id": "ldg-b"}, "c1")]}
    loop_final = {"role": "assistant", "content": "结论是……"}
    out, _t, _m = _run(ag, first, [loop_final])
    assert out.get("content") == "结论是……" and not out.get("tool_calls")


# ── 结构守卫：剥离只许有一个实现点（这条缺陷的成因就是"归调用方"没有接收方）──

def test_strip_has_a_single_implementation_point():
    """🔴 `strip_bladex_calls` 只许在 `_strip_execute_splice` 里被调用一次。

    本缺陷的形态是「A 说归 B 做、三个 B 都没做」。收成一个 helper 之后，
    再长出第二处调用就意味着又有人自己实现了一遍——那正是分叉的起点。
    """
    from _source_probe import package_source
    src = package_source("agency")   # F0.1 拆包：def 在 runtime.py，两条流式调用点在 streams.py
    assert src.count("strip_bladex_calls(") == 1, \
        "剥离只许在 `_strip_execute_splice` 里发生一次；多出来的一处就是分叉的起点"
    assert src.count("_strip_execute_splice(") == 5, (
        "期望五处：def 一处 + 四个调用点（非流式首条 MIXED / 非流式内循环 final / "
        "chat 流式 final / 协议流式 final）")


@pytest.mark.parametrize("path_marker", [
    "final, leaked, extra = await self._strip_execute_splice(",       # 非流式
    "final, _leaked, _extra = await agency._strip_execute_splice(",   # 两条流式共用写法
])
def test_all_three_paths_strip_the_loop_final(path_marker):
    """接线守卫：三条路径都必须在转发前过一次剥离。少一条就是本缺陷复发。"""
    from _source_probe import package_source
    src = package_source("agency")   # F0.1 拆包
    assert path_marker in src, f"这条路径没接剥离：{path_marker}"
