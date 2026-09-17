"""`remove` 的参数面适配：`match` 已唯一确定条目时 `section` 可选（F1.4 / MQ-L53）。

## 读数（09-08 四条 `agency_ledger_update_rejected`，两个 agent、两本账本）

| 错误 | 次数 | agent |
|---|---|---|
| `needs both section and op — got keys ['match','op']` | **2** | hermes:default 1 · Pi 1 |
| `remove requires match or index` | 2 | hermes:default 1 · Pi 1 |

**4/4 全出在 `remove` 的参数面。** 第一类是系统性的：两个互不相干的 agent
（不同厂商、不同 prompt）写出**逐字相同**的 `{op:"remove", match:"…"}`。

## 立论：模型是对的，schema 是错的

`match` 的契约是"条目全文或**唯一**前缀"——既然唯一，再指定段就是冗余，
模型推不出这个字段的必要性。刚性原则 10：**适配 agent 行为，别要求 agent 改**。

## 🔴 只放宽 + 只回显，不纠正

放宽 `required`、把收到的键回显出来；**不**把 `text` 静默当 `match` 用——
替模型改参数就是替它做决定（三红线：BladeX 不做作者），而且静默纠正会让同一个错
在下一个 agent 上再犯一次、永远不被发现。
"""

from __future__ import annotations

import pytest
from bladex_core.ledger import (
    ACTOR_MODEL,
    LedgerEntry,
    LedgerError,
    add_entry,
    new_ledger,
)
from bladex_core.ledger_runtime import apply_tool_update


def _ledger():
    led = new_ledger(ledger_id="ldg-f14", title="批 F1", goal="放宽 remove 参数面")
    led = add_entry(led, "next", LedgerEntry(text="核验对比图是否嵌入 HTML 第七章"),
                    actor=ACTOR_MODEL)
    led = add_entry(led, "next", LedgerEntry(text="补 CC BY 4.0 署名提示"),
                    actor=ACTOR_MODEL)
    led = add_entry(led, "open", LedgerEntry(text="标注质量抽样无记录"),
                    actor=ACTOR_MODEL)
    return led


def _texts(led, sec: str) -> list[str]:
    return [e.text for e in led.entries(sec)]


# ── ① 命中唯一 ⇒ 删对（单条 + 批量各一）──────────────────────────────────


def test_single_form_remove_without_section():
    """live 病例的逐字形态：`{op:"remove", match:"…"}`，没有 `section`。"""
    new, note = apply_tool_update(
        _ledger(), {"op": "remove", "match": "补 CC BY 4.0 署名提示"})
    assert _texts(new, "next") == ["核验对比图是否嵌入 HTML 第七章"]
    assert "next" in note, "确认文本要说清删的是哪一段（模型据此判断删对没有）"


def test_batch_form_remove_without_section():
    """批量同一条放宽——两条路各判各的就是两套契约（MQ-A18/P7 同族）。"""
    new, _ = apply_tool_update(_ledger(), {"entries": [
        {"section": "verified", "op": "add", "text": "对比图已嵌入"},
        {"op": "remove", "match": "核验对比图"},          # 唯一前缀，跨段唯一
    ]})
    assert _texts(new, "verified") == ["对比图已嵌入"]
    assert _texts(new, "next") == ["补 CC BY 4.0 署名提示"]


def test_match_finds_an_entry_in_a_section_the_model_did_not_name():
    """全账本搜索是真的全账本：条目在 `open` 里，模型一个字没提段名。"""
    new, note = apply_tool_update(
        _ledger(), {"op": "remove", "match": "标注质量抽样无记录"})
    assert _texts(new, "open") == []
    assert "open" in note


# ── ② 跨段同文本 ⇒ 歧义报错带段名与候选 ──────────────────────────────────


def test_cross_section_duplicate_reports_the_sections():
    led = add_entry(_ledger(), "open",
                    LedgerEntry(text="补 CC BY 4.0 署名提示"), actor=ACTOR_MODEL)
    with pytest.raises(LedgerError) as e:
        apply_tool_update(led, {"op": "remove", "match": "补 CC BY 4.0 署名提示"})
    msg = str(e.value)
    assert "next" in msg and "open" in msg, f"报错没点名段：{msg}"
    assert "署名提示" in msg, "报错没给候选文本，模型无从判断该补哪个 section"
    assert "section" in msg, "报错没说怎么办"


def test_ambiguity_inside_one_section_keeps_the_old_message():
    """段**内**的歧义仍走 `resolve_match` 的既有三条报错，一字不动。"""
    led = add_entry(_ledger(), "next",
                    LedgerEntry(text="核验对比图的另一件事"), actor=ACTOR_MODEL)
    with pytest.raises(LedgerError, match="make it longer"):
        apply_tool_update(led, {"op": "remove", "match": "核验对比图"})


# ── ③ 零命中 ⇒ 报错（不是静默无操作）────────────────────────────────────


def test_no_match_anywhere_is_an_error():
    """静默不删最坏：模型会以为删掉了，下一步就把它写进 Verified（MQ-L29 同族）。"""
    before = _ledger()
    with pytest.raises(LedgerError, match="no entry matching"):
        apply_tool_update(before, {"op": "remove", "match": "根本不存在的条目"})
    assert len(_texts(before, "next")) == 2, "抛错时调用方手上那份必须原封不动"


def test_goal_is_not_reachable_by_a_sectionless_remove():
    """🔴 三红线：Goal 仅用户可改。全账本搜索是新开的一条路，同一道门要再关一次。"""
    led = _ledger()
    with pytest.raises(LedgerError, match="no entry matching"):
        apply_tool_update(led, {"op": "remove", "match": led.goal})


# ── ④ `add` 缺 section 仍拒 ──────────────────────────────────────────────


def test_add_still_requires_section():
    """放宽只对 `remove`：`add` 没有段名就无处可加，那不是冗余是缺信息。"""
    for args in ({"op": "add", "text": "新结论"},
                 {"entries": [{"op": "add", "text": "新结论"}]}):
        with pytest.raises(LedgerError) as e:
            apply_tool_update(_ledger(), args)
        assert "section" in str(e.value)


def test_remove_by_index_still_requires_section():
    """`index` 是**段内**位置，没有段就没有意义 —— 这一条不放宽。"""
    with pytest.raises(LedgerError, match="section"):
        apply_tool_update(_ledger(), {"op": "remove", "index": 0})


# ── ⑤ 第二类报错也回显 keys ─────────────────────────────────────────────


def test_remove_without_match_or_index_echoes_the_keys():
    """🔴 同一条判据不能只用在一半上。

    第一类之所以能被判成"schema 错"，正是因为报错回显了键名；第二类此前没有回显
    ⇒ 模型当时写了什么键（很可能是 `text`）**在日志里不可判定**。
    """
    with pytest.raises(LedgerError) as e:
        apply_tool_update(_ledger(), {"entries": [
            {"section": "next", "op": "remove", "text": "补 CC BY 4.0 署名提示"}]})
    msg = str(e.value)
    assert "got keys" in msg and "'text'" in msg, f"没回显收到的键：{msg}"


def test_the_echo_does_not_silently_promote_text_to_match():
    """**只回显不纠正**：`text` 不会被当成 `match` 用，条目原封不动。"""
    before = _ledger()
    with pytest.raises(LedgerError):
        apply_tool_update(before, {"op": "remove",
                                   "text": "补 CC BY 4.0 署名提示"})
    assert len(_texts(before, "next")) == 2


# ── ⑥ schema 守卫 ───────────────────────────────────────────────────────


def test_schema_required_is_op_only_and_old_params_survive():
    from bladex_proxy.toolface import TOOL_SCHEMAS

    fn = next(t["function"] for t in TOOL_SCHEMAS
              if t["function"]["name"] == "bladex_ledger_update")
    items = fn["parameters"]["properties"]["entries"]["items"]
    assert items["required"] == ["op"]
    # 老参数集合 ⊆ 新：0.1.0 五个 agent 的 live 记录全走单条面，不许在这里被动过
    assert {"section", "op", "text", "ref", "match", "index"} <= set(items["properties"])
    assert "identifies the entry" in items["properties"]["match"]["description"], \
        "match 的说明没告诉模型 section 可省 —— 放宽了却不说 = 白放"


# ── ⑦ 判别力 ───────────────────────────────────────────────────────────


def test_discriminative_required_back_to_two_would_reject_case_one():
    """把 `required` 改回两项 ⇒ ① 必红。

    这里直接对着**判据本身**做对照：`_validate_spec` 里那条"remove + match ⇒
    section 可选"的短路一旦不成立，live 那个逐字形态就回到"needs both"。
    """
    import bladex_core.ledger_runtime as lr

    spec = {"op": "remove", "match": "补 CC BY 4.0 署名提示"}
    lr._validate_spec(0, spec, _ledger())          # 现在：过

    orig = lr._match_sections
    try:
        # 收紧成"必须给 section"的等价物：把可选判据抽掉
        lr._match_sections = lambda ledger, match: []
        with pytest.raises(LedgerError, match="no entry matching"):
            apply_tool_update(_ledger(), spec)
    finally:
        lr._match_sections = orig
    # 恢复后同一条又能过 —— 证明红/绿是由那条判据决定的
    assert apply_tool_update(_ledger(), spec)[0].entries("next")
