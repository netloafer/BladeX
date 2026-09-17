""""读了"代理按**首动作**分桶（F1.1 / MQ-A48，2026-09-08 批 F1）。

## 这条卡修的是什么

`_ledger_next_referenced` 的分子里混进了**内容自指**：首动作若是
`bladex_ledger_update`，它的参数（`match` / `text`）就是 Next/Open 的原文——
词面交集必然满格。live 09-07 `usable=True` 17 行里有 1 行
`hits=66/66 hits_rare=62/62`（`ldg-c5103b17c024` rev 25→26），正是这一形态。

与 MQ-A43 的**时间自指**是两件事：那条修的是"单元集取自模型看到的那一版"
（尺子量自己的手）；这条修的是"首动作本身就是账本原文"（尺子量的是抄写）。
A43 修完之后这一形态仍在——因为模型看到的那一版里**确实**有那句话，
它只是把它抄进了 `match` 参数。

## 为什么单列而不是剔掉

`ledger_edit` 桶本身是有意义的读数（模型确实在记账，那是账本活着的证据），
只是它不能进"读了没有"的分子。剔掉 = 丢一类信息；单列 = 两个问题各有各的读数。

钉四件事：
① 首动作 `bladex_ledger_update` ⇒ `bucket=ledger_edit`；
② 首动作是干活工具（`terminal` / `write_file`）⇒ `work`；
③ 没有工具调用（纯文本）⇒ `work`（`first_action=text`）；
④ 判别力：把判据换成"看 `first_action` 是否为空" ⇒ ① 必红。
"""

from __future__ import annotations

import structlog.testing
from bladex_core.ledger import ACTOR_MODEL, LedgerEntry, add_entry, new_ledger
from bladex_proxy.models import LedgerAnchor, ToolEvent
from bladex_proxy.server.orchestration import _ledger_next_referenced

_LID = "ldg-f11"
_NEXT = "把 probe_ledger_trajectory 的分桶接上 self_conflict"


class _Agency:
    def __init__(self, pool, breakdown):
        self.pool = pool
        self.last_ledger_breakdown = breakdown


def _pool():
    led = new_ledger(ledger_id=_LID, title="批 F1 仪器棒", goal="立分桶")
    led = add_entry(led, "next", LedgerEntry(text=_NEXT), actor=ACTOR_MODEL)
    return {_LID: led}


def _bd(pool):
    """生产侧 `_build_ledger_block` 的那一句——单元集按注入那一刻算。

    **稀有轴留空**：本卡量的是分桶，与 `pending_rare` 的口径无关（那是 F1.2）。
    在这里跟着算一遍，只会让本文件跟着另一张卡的签名一起红。
    """
    from bladex_core.ledger_runtime import candidate_units
    from bladex_proxy.agency import LedgerBlockBreakdown
    led = pool[_LID]
    text = " ".join(e.text for sec in ("next", "open") for e in led.entries(sec))
    return LedgerBlockBreakdown(ledger_id=_LID, rev=led.rev,
                                pending_units=frozenset(candidate_units(text)))


def _run(*, response_text: str = "", tool_events=None):
    import types
    pool = _pool()
    bd = _bd(pool)
    # F1.3 / MQ-A49：单元集走 `request.state`（请求作用域），不走 agency 上的
    # 进程级"最近一次"——`app.state.agency` 只为 `rev_now` 那一轴留着。
    req = types.SimpleNamespace(
        state=types.SimpleNamespace(
            bladex_pending_units=frozenset(bd.pending_units),
            bladex_pending_rare=frozenset(bd.pending_rare)),
        app=types.SimpleNamespace(
            state=types.SimpleNamespace(agency=_Agency(pool, bd))))
    anchor = LedgerAnchor(ledger_id=_LID, rev=3, sections={"next": 40})
    with structlog.testing.capture_logs() as cap:
        _ledger_next_referenced(req, anchor, response_text, tool_events or [])
    recs = [e for e in cap if e["event"] == "agency_ledger_next_referenced"]
    assert len(recs) == 1
    return recs[0]


def _call(name: str, arguments: dict):
    return [ToolEvent(tool_name=name, direction="call", arguments=arguments)]


# ── ① 账本编辑桶 ───────────────────────────────────────────────────────────


def test_ledger_update_first_action_is_bucketed_as_ledger_edit():
    """live 病例回放：首动作把 Next 原文抄进 `match` ⇒ 满命中，但那不是"读了"。"""
    r = _run(tool_events=_call("bladex_ledger_update",
                               {"section": "next", "op": "remove", "match": _NEXT}))
    assert r["bucket"] == "ledger_edit"
    assert r["first_action"] == "bladex_ledger_update"
    # 阳性对照：这一档确实是满命中（否则本卡在解决一个不存在的问题）
    assert r["hits"] == r["next_units"] and r["next_units"] > 0, \
        "内容自指的形状就是 hits==next_units——读数上必须看得见"


def test_other_bladex_tools_are_ledger_edit_too():
    """判据是**工具族**（`bladex_*`），不是某一个工具名。

    `bladex_memory_search` 的参数不是账本原文，但它同样不是"按账本干活"——
    这一桶的语义是"首动作发生在 BladeX 自己的工具面上"，与 `work` 互斥。
    闭集从定义读（`toolface.is_bladex_tool`），不在这里重写一遍前缀判断。
    """
    assert _run(tool_events=_call("bladex_memory_search",
                                  {"query": "分桶"}))["bucket"] == "ledger_edit"


# ── ② 干活桶 ───────────────────────────────────────────────────────────────


def test_work_tools_are_bucketed_as_work():
    for name in ("terminal", "write_file", "Edit"):
        r = _run(tool_events=_call(name, {"file_path": "scripts/x.py"}))
        assert r["bucket"] == "work", f"{name} 是干活工具"
        assert r["first_action"] == name


# ── ③ 纯文本 ───────────────────────────────────────────────────────────────


def test_text_only_first_action_is_work():
    """没有工具调用 ⇒ `first_action=text` ⇒ `work`。

    "模型只说话没动手"仍然是**对着任务**的动作（它可能正在复述 Next 里那一条），
    与"编辑账本"不是一回事——归 `work` 而不是第三个桶。
    """
    r = _run(response_text="好，我先把 probe_ledger_trajectory 的分桶接上。")
    assert r["first_action"] == "text" and r["bucket"] == "work"
    assert r["hits"] >= 1, "阳性对照：这一档确实会命中，所以它属于分子"


# ── ④ 判别力 ───────────────────────────────────────────────────────────────


def test_discriminative_empty_first_action_criterion_flips_bucket(monkeypatch):
    """把判据换成"看 `first_action` 是否为空" ⇒ ① 必红。

    🔴 打**消费方**（`bladex_proxy.toolface`，函数体内 import 的那个模块），
    F0 拆包后打门面不生效。
    """
    import bladex_proxy.toolface as tf
    monkeypatch.setattr(tf, "is_bladex_tool", lambda name: not name)
    r = _run(tool_events=_call("bladex_ledger_update",
                               {"section": "next", "op": "remove", "match": _NEXT}))
    assert r["bucket"] == "work", \
        "换掉判据后 ① 必须翻——否则 bucket 不是由 is_bladex_tool 决定的"
