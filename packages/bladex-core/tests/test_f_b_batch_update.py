"""F-B1（MQ-L49 ①②）：`bladex_ledger_update` 批量形态 + `match` 删除。

修前形态（读数：09-02～05 全部 proxy 日志，46 次 `agency_stream_loop_done`）：
schema 一次一条 ⇒ "写 Verified + 删对应 Next" 要两次调用 ⇒ 纯内循环每轮把整段
上下文再发上游 ⇒ rounds 6（上限）×4 全是 `bladex_ledger_update ×5-6`，
单次 44–78 万 token。

本组钉三件事，每件都对着一种会安静失效的改法：

1. **原子性**：任一条失败 ⇒ 整批不落。判别力对照 `test_...would_be_red_if_applied_one_by_one`
   ——把实现改成"逐条应用"会让它必红（半本账本比不落更糟：模型看不出哪几条成功了，
   只能重发，于是把成功的那几条又写一遍）。
2. **老调用面一字不动**：单条 `section+op+text` 的确认回文逐字钉死。0.1.0 五个
   agent 的 live 记录全是它，批量是**新增**不是替换。
3. **schema 与实现同一个闭集**：op 枚举从 `UPDATE_OPS` 读，不在测试里默写
   （feedback_closed_set_from_definition：加了新成员、旧分支没跟上是本仓高发形态）。
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
from bladex_core.ledger_runtime import (
    apply_tool_update,
    resolve_match,
    update_edit_ops,
)


def _ledger(**kw):
    """两条 next + 一条 open 的账本（"写 verified 顺手删 next" 的题面）。"""
    led = new_ledger(ledger_id="ldg-fb1", title="批量更新", **kw)
    for text in ("跑 gate_check 全绿", "把批量形态接进 toolface"):
        led = add_entry(led, "next", LedgerEntry(text=text, source="model"),
                        actor=ACTOR_MODEL)
    return add_entry(led, "open", LedgerEntry(text="rev 冲突要不要分桶",
                                              source="model"),
                     actor=ACTOR_MODEL)


# ── 批量本体 ────────────────────────────────────────────────────────────────

class TestBatch:
    def test_two_adds_and_one_remove_land_in_one_call(self):
        led, msg = apply_tool_update(_ledger(), {"entries": [
            {"section": "verified", "op": "add", "text": "gate 全绿",
             "ref": "gate_check#20260907"},
            {"section": "core", "op": "add", "text": "批量是新增参数不是替换"},
            {"section": "next", "op": "remove", "index": 0},
        ]})
        assert [e.text for e in led.entries("verified")] == ["gate 全绿"]
        assert led.entries("verified")[0].ref == "gate_check#20260907"
        assert [e.text for e in led.entries("core")] == ["批量是新增参数不是替换"]
        assert [e.text for e in led.entries("next")] == ["把批量形态接进 toolface"]
        # 确认文本逐条拼接：模型要能对着它核对每一条都落了
        assert msg.splitlines() == [
            "added to verified: gate 全绿",
            "added to core: 批量是新增参数不是替换",
            "removed from next: 跑 gate_check 全绿",
        ]

    def test_batch_can_remove_by_match_what_it_just_added(self):
        """批内顺序生效：先 add 后按内容删同一条是合法形态。

        钉的是"状态相关的校验必须看**这一条执行时**的账本"——若把 match 唯一性
        预先对着批次开始时的账本校验，这条会被误拒。
        """
        led, _ = apply_tool_update(_ledger(), {"entries": [
            {"section": "open", "op": "add", "text": "临时问题"},
            {"section": "open", "op": "remove", "match": "临时问题"},
        ]})
        assert [e.text for e in led.entries("open")] == ["rev 冲突要不要分桶"]

    def test_bad_section_in_third_edit_lands_nothing(self):
        """🔴 原子性本体：第 3 条坏段名 ⇒ 前两条也不落。"""
        base = _ledger()
        with pytest.raises(LedgerError) as ei:
            apply_tool_update(base, {"entries": [
                {"section": "verified", "op": "add", "text": "A"},
                {"section": "core", "op": "add", "text": "B"},
                {"section": "verfied", "op": "add", "text": "C"},   # 拼错
            ]})
        assert "verfied" in str(ei.value)
        assert base.entries("verified") == [] and base.entries("core") == []

    def test_batch_would_be_red_if_applied_one_by_one(self):
        """判别力对照：把实现改成"逐条应用、坏的跳过/半途中断"，本条必红。

        题面是**状态相关**的失败（第 2 条 index 越界），它只可能在第 1 条已经
        应用之后才发现 —— 预校验抓不到它，只有"整批回滚"这个语义能救。
        """
        base = _ledger()
        with pytest.raises(LedgerError) as ei:
            apply_tool_update(base, {"entries": [
                {"section": "verified", "op": "add", "text": "先落一条"},
                {"section": "next", "op": "remove", "index": 99},
            ]})
        assert "edit 2 of 2" in str(ei.value) and "NOTHING" in str(ei.value)
        assert base.entries("verified") == [], "第 1 条不许留在账本上"

    def test_prevalidation_failure_says_nothing_was_written(self):
        """🔴 09-08 事故复现（MQ-L52）：**预校验**那一路此前直接抛原始消息，
        模型只看到 `entries[5] needs both section and op` —— 不知道整批都没落、
        也不知道该重发整批。live 后果：hermes 连撞 6 次、`rounds=6 degraded=True`、
        62 万 token、**零写入**（`agency_ledger_update_applied` 当天 0 条）。

        两条失败路径（预校验 / 应用）现在共用一个出口，本条钉预校验那一路。
        """
        with pytest.raises(LedgerError) as ei:
            apply_tool_update(_ledger(), {"entries": [
                {"section": "verified", "op": "add", "text": "A"},
                {"match": "漏了 section 和 op"},
            ]})
        msg = str(ei.value)
        assert "edit 2 of 2" in msg, "没说是第几条 ⇒ 模型不知道改哪一条"
        assert "NOTHING was written" in msg, \
            "没说整批没落 ⇒ 模型会以为第 1 条已经进去了，只补发第 2 条"
        assert "WHOLE batch" in msg, "没说重发整批"

    def test_missing_keys_are_echoed_back(self):
        """错误要**回显收到了什么**——只说"缺 section 和 op"时，模型看不出
        它写的与 BladeX 读到的差在哪，只能原样重发（live 实测 6 次）。
        回显**键名**足以暴露大小写/拼写；**不回显值**（值可能很长，且错误文本
        本身也是注入，ADR-0032 §3.2-7）。"""
        with pytest.raises(LedgerError) as ei:
            apply_tool_update(_ledger(), {"entries": [
                {"Section": "next", "Op": "remove", "match": "x"}]})
        msg = str(ei.value)
        assert "'Section'" in msg and "'Op'" in msg, f"没回显收到的键名：{msg}"
        assert "lowercase" in msg, "没指出大小写问题"
        assert "next" not in msg, "不该回显值"

    def test_alias_keys_are_named_not_auto_corrected(self):
        """键名近似 ⇒ 报出来，**不替它改**（三红线：BladeX 不做作者；
        静默纠正还会让同一个错在下一个 agent 上再犯、永远不被发现）。"""
        with pytest.raises(LedgerError) as ei:
            apply_tool_update(_ledger(), {"entries": [
                {"section": "next", "operation": "remove", "match": "x"}]})
        assert "'operation' is not 'op'" in str(ei.value), str(ei.value)

    def test_empty_entry_object_gets_its_own_hint(self):
        """模型用 `{}` 表示"就这些了"是常见写法，值得单独一句。"""
        with pytest.raises(LedgerError) as ei:
            apply_tool_update(_ledger(), {"entries": [
                {"section": "next", "op": "add", "text": "A"}, {}]})
        assert "empty" in str(ei.value) and "drop it" in str(ei.value)

    def test_unrecognised_shape_gets_no_invented_hint(self):
        """认不出就不猜——多说一句错的比不说更糟（它会照着错提示改第二遍）。"""
        with pytest.raises(LedgerError) as ei:
            apply_tool_update(_ledger(), {"entries": [{"foo": 1, "bar": 2}]})
        msg = str(ei.value)
        assert "'bar', 'foo'" in msg, "键名仍要回显"
        assert "must be" not in msg and "is not" not in msg, \
            f"不该编造提示：{msg}"

    def test_goal_and_entries_in_one_call_is_refused_loudly(self):
        """Goal 改动与条目改动互斥。**响亮失败**——静默吞掉一整批条目是原则 13
        的反面（丢数据要么不丢，要么留显式标记）。"""
        with pytest.raises(LedgerError, match="separate calls"):
            apply_tool_update(_ledger(), {
                "goal": "新目标", "goal_change_quote": "改成新目标",
                "entries": [{"section": "next", "op": "add", "text": "x"}],
            }, user_text="改成新目标")

    def test_entries_must_be_a_list(self):
        with pytest.raises(LedgerError, match="entries must be a list"):
            apply_tool_update(_ledger(), {"entries": {"section": "next"}})

    def test_empty_entries_falls_through_to_single_form(self):
        """空 `entries` 不是"零改动成功"，它落回单条形态的必填校验。"""
        with pytest.raises(LedgerError, match="section"):
            apply_tool_update(_ledger(), {"entries": []})


# ── match 删除 ──────────────────────────────────────────────────────────────

class TestMatch:
    def test_full_text(self):
        led, msg = apply_tool_update(_ledger(), {
            "section": "next", "op": "remove", "match": "跑 gate_check 全绿"})
        assert [e.text for e in led.entries("next")] == ["把批量形态接进 toolface"]
        assert msg == "removed from next: 跑 gate_check 全绿"

    def test_unique_prefix(self):
        led, _ = apply_tool_update(_ledger(), {
            "section": "next", "op": "remove", "match": "把批量"})
        assert [e.text for e in led.entries("next")] == ["跑 gate_check 全绿"]

    def test_ambiguous_prefix_names_the_candidates(self):
        led = new_ledger(ledger_id="ldg-amb")
        for t in ("修 A 的回归", "修 B 的回归", "别的"):
            led = add_entry(led, "next", LedgerEntry(text=t), actor=ACTOR_MODEL)
        with pytest.raises(LedgerError) as ei:
            apply_tool_update(led, {"section": "next", "op": "remove",
                                    "match": "修"})
        msg = str(ei.value)
        assert "matches 2 entries" in msg
        assert "修 A 的回归" in msg and "修 B 的回归" in msg, \
            "报歧义要给候选文本——模型据此把 match 写长，不然它只能改用 index"
        assert "别的" not in msg

    def test_identical_entries_send_the_model_to_index(self):
        """全文相等的重复条目：列文本没有信息量（一模一样），要点名 index。"""
        led = new_ledger(ledger_id="ldg-dup")
        for _ in range(2):
            led = add_entry(led, "next", LedgerEntry(text="同一条"),
                            actor=ACTOR_MODEL)
        with pytest.raises(LedgerError, match="not unique"):
            apply_tool_update(led, {"section": "next", "op": "remove",
                                    "match": "同一条"})

    def test_zero_hit_is_an_error_not_a_silent_no_op(self):
        """🔴 零命中必须报错。静默不删 ⇒ 模型以为删掉了 ⇒ 下一步把它写进
        Verified 而 Next 里还留着（MQ-L29 那个形态的另一条产生路径）。"""
        with pytest.raises(LedgerError, match="no entry matching"):
            apply_tool_update(_ledger(), {"section": "next", "op": "remove",
                                          "match": "根本没有这条"})

    def test_full_text_beats_prefix(self):
        """全文相等优先：`abc` 与 `abcdef` 同段时，match='abc' 删前者。"""
        led = new_ledger(ledger_id="ldg-pref")
        for t in ("abc", "abcdef"):
            led = add_entry(led, "next", LedgerEntry(text=t), actor=ACTOR_MODEL)
        out, _ = apply_tool_update(led, {"section": "next", "op": "remove",
                                         "match": "abc"})
        assert [e.text for e in out.entries("next")] == ["abcdef"]

    def test_resolve_match_is_a_pure_function(self):
        items = [LedgerEntry(text="第一条"), LedgerEntry(text="第二条")]
        assert resolve_match(items, "第二条") == 1
        assert resolve_match(items, "第一") == 0
        with pytest.raises(LedgerError):
            resolve_match(items, "")


# ── 老调用面（红线：一字不动）─────────────────────────────────────────────────

class TestLegacyShapeUnchanged:
    def test_single_add_confirmation_is_verbatim(self):
        led, msg = apply_tool_update(new_ledger(ledger_id="ldg-x"), {
            "section": "verified", "op": "add", "text": "pytest 全绿",
            "ref": "gate#1"})
        assert msg == "added to verified: pytest 全绿"
        e = led.entries("verified")[0]
        assert (e.source, e.ref) == ("tool", "gate#1")

    def test_single_remove_by_index_confirmation_is_verbatim(self):
        led, msg = apply_tool_update(_ledger(), {"section": "next",
                                                 "op": "remove", "index": 1})
        assert msg == "removed from next: 把批量形态接进 toolface"

    def test_unknown_op_message_is_verbatim(self):
        with pytest.raises(LedgerError, match=r"unknown op 'replace' \(add\|remove\)"):
            apply_tool_update(_ledger(), {"section": "next", "op": "replace"})

    def test_add_without_ref_is_model_source(self):
        led, _ = apply_tool_update(_ledger(), {"section": "next", "op": "add",
                                               "text": "跑基准"})
        assert led.entries("next")[-1].source == "model"


# ── 观测口径 ────────────────────────────────────────────────────────────────

class TestEditOps:
    def test_ops_of_each_shape(self):
        assert update_edit_ops({"goal": "g"}) == ["goal"]
        assert update_edit_ops({"section": "next", "op": "add"}) == ["add"]
        assert update_edit_ops({"entries": [
            {"section": "verified", "op": "add"},
            {"section": "next", "op": "remove"}]}) == ["add", "remove"]

    def test_unreadable_shape_is_marked_not_empty(self):
        """空列表会让"没读到"与"零改动"在读数里长成一样（分母纪律）。"""
        assert update_edit_ops({}) == ["?"]
        assert update_edit_ops({"entries": ["not a dict"]}) == ["?"]


# schema ⇄ 运行时的对账（op 闭集单一定义 / 老参数 ⊆ 新参数）在 proxy 侧：
# `packages/bladex-proxy/tests/test_f_b_batch_toolface.py`——core 不看 proxy 的源码
# （依赖方向 proxy → core，测试也不例外）。
