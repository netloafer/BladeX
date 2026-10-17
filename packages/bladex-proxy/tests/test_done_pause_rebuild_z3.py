"""批 Z · Z3.2（Y1b）完结停锚的**生产接线**：真 Hub + 真 `rebuild_from_hub`。

core 侧判定在 `bladex-core/tests/test_ledger_lifecycle.py`；本文件只管"重建锚层真的按它走"：

  ① 阳性（开关打开时；`BLADEX_LEDGER_DONE_PAUSE` 默认关）：账本自述完结（Open/Next 空、
     Verified 非空）之后的轮次**不产** ANCHOR 边；完结之前的轮次照产（锚层没有整本失效）；
  ② 关本（`ledger_close` 事件）之后的轮次不锚（判据 ③ 的重建半）；
  ③ 🔴 不换受害者（判据 ⑤）：被停锚的轮次**不许**经 DPL 挂进**别的**账本锚卡——
     准入闸② 用的是标签口径（`apply_done_pause=False`），停锚只摘 ANCHOR 不摘标签
     （接线守卫在此，契约在 core 侧）。
"""

from __future__ import annotations

from bladex_core.ledger import LedgerEntry, add_entry, new_ledger
from bladex_core.ledger_runtime import ledger_anchor_matter_id
from bladex_proxy.models import AdminEventType
from test_ledger_anchor_wiring import (  # noqa: F401 —— 同一套剧本工具，不另抄
    _anchor_edges,
    _open,
    _put_turn,
    _switch,
    _tick,
)

_L1 = "ldg-3333cccc3333"
_L2 = "ldg-4444dddd4444"


def _write(ledger, lid: str, *, verified: int, open_: int = 0, next_: int = 0) -> None:
    led = new_ledger(ledger_id=lid, title=f"账本 {lid[-4:]}", goal="做完可核对",
                     created_at="2026-09-01T00:00:00+00:00")
    for sec, n in (("verified", verified), ("open", open_), ("next", next_)):
        for i in range(n):
            led = add_entry(led, sec, LedgerEntry(text=f"{sec}-{i}", source="model"),
                            actor="model")
    ledger.append_admin_event(AdminEventType.LEDGER_UPDATE, "", ledger=led.model_dump())
    _tick()


def _edges_on(index, lid: str) -> list:
    mid = ledger_anchor_matter_id(lid)
    return [e for e in index.get_edges(mid)
            if getattr(e.target_type, "value", str(e.target_type)) == "fact"]


def _fact_keys(index, edges) -> set[str]:
    return {index.get_fact(e.target_key).source_ledger_key for e in edges
            if index.get_fact(e.target_key) is not None}


def test_turns_after_done_are_not_anchored(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("BLADEX_LEDGER_ANCHOR", "1")
    monkeypatch.setenv("BLADEX_LEDGER_DONE_PAUSE", "1")   # 默认关（读数见 flags 注释）
    ledger, index = _open(tmp_path)
    try:
        _switch(ledger, "claude-code", _L1, 0)
        _write(ledger, _L1, verified=1, next_=2)
        before = _put_turn(ledger, 1, "claude-code",
                           "第一件事：修掉路由层的回归问题，静态配置优先那条规则没有生效")
        _write(ledger, _L1, verified=3)                       # 自述完结
        after = _put_turn(ledger, 2, "claude-code",
                          "顺便问一下 ADR-0032 的三条红线分别是什么，Goal 段谁能改")
        index.rebuild_from_hub(ledger)

        anchored = _fact_keys(index, [e for e in _anchor_edges(index)
                                      if e.matter_id == ledger_anchor_matter_id(_L1)])
        assert before in anchored, "完结之前的轮次也不锚了——停锚把整本账本打失效了"
        assert after not in anchored, "完结之后的轮次仍被锚进来（Y0「做完不关」机判形态）"
    finally:
        index.close()
        ledger.close()


def test_turns_after_close_are_not_anchored(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("BLADEX_LEDGER_ANCHOR", "1")
    ledger, index = _open(tmp_path)
    try:
        _switch(ledger, "claude-code", _L1, 0)
        before = _put_turn(ledger, 1, "claude-code",
                           "第一件事：修掉路由层的回归问题，静态配置优先那条规则没有生效")
        ledger.append_admin_event(AdminEventType.LEDGER_CLOSE, "",
                                  ledger_id=_L1, origin="cli", reason="done")
        _tick()
        after = _put_turn(ledger, 2, "claude-code",
                          "另一件事：写 0.1.0 发布说明，把账本机制和记忆注入两条主线讲清楚")
        index.rebuild_from_hub(ledger)

        anchored = _fact_keys(index, [e for e in _anchor_edges(index)
                                      if e.matter_id == ledger_anchor_matter_id(_L1)])
        assert before in anchored
        assert after not in anchored
    finally:
        index.close()
        ledger.close()


def test_gate_reads_label_not_paused_anchor() -> None:
    """判据 ⑤ 的接线守卫：`_lg_reject` 喂给闸② 的是**标签口径**（`label=True` ⇒
    `apply_done_pause=False`）。停锚口径喂进去 ⇒ 被停锚的轮次标签为空、闸② 放行别人的锚卡。

    契约本身（标签口径挡别人的卡、放自己的卡；停锚口径两个都放）在 core
    `test_ledger_lifecycle.py::test_gate_label_semantics_keep_other_ledger_cards_closed`。
    为什么不在这里用真重建钉：单测夹具里 DPL 不会把 fact 路由到锚卡（实测三轮剧本都
    落 provisional），闸② 无物可挡，那条用例正反两种接线都绿——测不到东西的测试比没有更糟。
    """
    import inspect

    from bladex_proxy.storage import memory_index as mi
    src = inspect.getsource(mi)
    body = src[src.index("def _lg_reject("):src.index("# 第三个闸点")]
    assert "turn_ledger_id=_la_ledger_at(lk, label=True)" in body
