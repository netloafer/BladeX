"""F-B2（MQ-L49 ③）：rev 乐观锁只对 remove/goal 判 + 冲突按"自撞/跨 agent"分桶。

**为什么改**（live 读数，09-02～05 全部 proxy 日志）：`agency_ledger_rev_conflict`
8 次，**8/8 是同一个 agent 自撞** —— 并行 `tool_calls` 里几条 update 各带同一个
它读到的 rev，第一条落地把 rev 推到 n+1，后面几条全被判冲突。跨 agent 的真冲突
**0 次**。被拦的全是 add，而 add 是可交换的：两个写者各加一条，谁先谁后结果都是
两条都在，不存在"按旧视图错删"——那才是这把锁存在的理由。每一次误拒还要换模型
再发一轮全上下文，正是 L49 那条成本曲线。

三件事各有一条判别力对照：
1. add-only 带旧 rev ⇒ **放行**（对照：remove 带旧 rev 仍拒——把锁拆没了本条必红）；
2. 混合批（add+remove）算 remove ⇒ 拒（保守方向：认不出的形态一律照旧判）；
3. `self_conflict` 两态都要出现（恒 True 或恒 False 的分桶等于没有分桶）。
"""

from __future__ import annotations

import asyncio

import pytest
from bladex_core.ledger import ACTOR_MODEL, Ledger, LedgerEntry, add_entry
from bladex_core.ledger_runtime import activation_scope, update_is_add_only
from bladex_proxy.agency import AgencyRuntime


@pytest.fixture(autouse=True)
def _modules_on(monkeypatch):
    monkeypatch.setenv("BLADEX_MODULE_LEDGER", "1")
    monkeypatch.setenv("BLADEX_MODULE_TOOLFACE", "1")


def _rt(last_writer: str = "codex", rev: int = 5):
    led = Ledger(ledger_id="ldg-fb2", title="rev 分桶", rev=rev,
                 last_writer=last_writer)
    led = add_entry(led, "next", LedgerEntry(text="待办一", source="model"),
                    actor=ACTOR_MODEL)
    rt = AgencyRuntime(index=None, hub=None)
    rt.pool[led.ledger_id] = led
    rt.activation.restore({activation_scope("codex", ""): led.ledger_id})
    return rt, led


def _update(rt, args, agent="codex"):
    return asyncio.run(rt._h_ledger_update(
        args, allowed_exposure="local",
        context={"agent_id": agent, "project_id": "", "session_id": "s1"}))


class TestAddOnlyPassesStaleRev:
    def test_stale_rev_add_is_accepted(self):
        rt, led = _rt()
        out = _update(rt, {"section": "verified", "op": "add",
                           "text": "并行写的第二条", "rev": 3})
        assert out == "added to verified: 并行写的第二条", out
        assert rt.pool[led.ledger_id].rev == 6

    def test_stale_rev_batch_of_adds_is_accepted(self):
        rt, led = _rt()
        out = _update(rt, {"rev": 3, "entries": [
            {"section": "verified", "op": "add", "text": "A"},
            {"section": "core", "op": "add", "text": "B"},
        ]})
        assert out.startswith("added to verified"), out
        assert rt.pool[led.ledger_id].rev == 6

    def test_stale_rev_remove_is_still_rejected(self):
        """判别力对照：把锁整个拆掉（或把判据写成"一律放行"）⇒ 本条必红。"""
        rt, led = _rt()
        out = _update(rt, {"section": "next", "op": "remove", "index": 0,
                           "rev": 3})
        assert out.startswith("Error: ledger changed"), out
        assert rt.pool[led.ledger_id].rev == 5, "拒写不动池"

    def test_mixed_batch_counts_as_remove(self):
        """混合批（add + remove）按 remove 判——保守方向：一条 remove 就足以
        让"按旧视图错删"成立。"""
        rt, led = _rt()
        out = _update(rt, {"rev": 3, "entries": [
            {"section": "verified", "op": "add", "text": "A"},
            {"section": "next", "op": "remove", "match": "待办一"},
        ]})
        assert out.startswith("Error: ledger changed"), out
        assert rt.pool[led.ledger_id].rev == 5

    def test_goal_change_with_stale_rev_is_still_rejected(self):
        """Goal 是用户所有物，旧 rev 改它一律拒（红线 1 侧的保守）。"""
        rt, led = _rt()
        out = _update(rt, {"goal": "新目标", "goal_change_quote": "改成新目标",
                           "rev": 3})
        assert out.startswith("Error: ledger changed"), out
        assert rt.pool[led.ledger_id].rev == 5

    def test_current_rev_is_untouched_by_this_change(self):
        rt, led = _rt()
        assert _update(rt, {"section": "next", "op": "remove", "index": 0,
                            "rev": 5}).startswith("removed")
        assert rt.pool[led.ledger_id].rev == 6


class TestSelfConflictBucket:
    def _conflict(self, monkeypatch, *, last_writer: str, agent: str) -> dict:
        seen: dict = {}

        import bladex_proxy.agency.handlers as H

        class _Log:
            def warning(self, event, **kw):
                if event == "agency_ledger_rev_conflict":
                    seen.update(kw)

            def info(self, event, **kw):
                pass

        # 🔴 monkeypatch 打**消费方子模块**（F0 拆包后门面上打不生效）。
        monkeypatch.setattr(H, "logger", _Log())
        rt, _ = _rt(last_writer=last_writer)
        rt.activation.restore({activation_scope(agent, ""): "ldg-fb2"})
        _update(rt, {"section": "next", "op": "remove", "index": 0, "rev": 3},
                agent=agent)
        return seen

    def test_same_agent_is_self_conflict(self, monkeypatch):
        seen = self._conflict(monkeypatch, last_writer="codex", agent="codex")
        assert seen.get("self_conflict") is True, seen

    def test_other_agent_is_cross_conflict(self, monkeypatch):
        seen = self._conflict(monkeypatch, last_writer="hermes:default",
                              agent="codex")
        assert seen.get("self_conflict") is False, seen

    def test_unknown_writer_is_not_self_conflict(self, monkeypatch):
        """空写者不算自撞——"两个空字符串相等"会把未知伪装成已知。"""
        seen = self._conflict(monkeypatch, last_writer="", agent="")
        assert seen.get("self_conflict") is False, seen

    def test_conflict_carries_the_ops(self, monkeypatch):
        seen = self._conflict(monkeypatch, last_writer="codex", agent="codex")
        assert seen.get("ops") == "remove", seen


class TestAcceptedBucketIsReachable:
    """🔴 **桶可达**（feedback_instrument_reference_frame：读数为 0 先证明那个桶可达）。

    首版只断言了**行为**（旧 rev 的 add 被放行），没断言**埋点**——而 MQ-L49 的判据
    之一正是数 `agency_ledger_rev_stale_add_accepted` 这条日志。行为对而日志没打，
    读数就恒为 0，且与"这条判据从没被触发"长得一模一样（本仓已登记过这个形态：
    `Turn.subagent` 673 轮全 0，字段在 schema 里、生产者断线，静态守卫看不见）。

    09-07 首日 live 读数就是 0，本组把"0 = 没触发"与"0 = 埋点断了"分开——
    有了它，那个 0 才可以被读成"这条判据今天没派上用场"。
    """

    def _accepted(self, monkeypatch, args: dict) -> dict:
        seen: dict = {}

        import bladex_proxy.agency.handlers as H

        class _Log:
            def info(self, event, **kw):
                if event == "agency_ledger_rev_stale_add_accepted":
                    seen.update(kw)

            def warning(self, event, **kw):
                pass

        monkeypatch.setattr(H, "logger", _Log())   # 打消费方子模块，不打门面
        rt, _ = _rt(last_writer="hermes:default")
        _update(rt, args)
        return seen

    def test_the_accepted_log_is_actually_emitted(self, monkeypatch):
        seen = self._accepted(monkeypatch, {
            "section": "verified", "op": "add", "text": "并行的第二条", "rev": 3})
        assert seen, "放行了但没打日志 ⇒ MQ-L49 那条读数恒为 0 且无从分辨"
        assert seen["ledger"] == "ldg-fb2"
        assert (seen["want"], seen["current"]) == (3, 5), \
            "要能看出落后了多少——只记'放行了'读不出严重程度"
        assert seen["writer"] == "codex" and seen["last_writer"] == "hermes:default"
        assert seen["n_edits"] == 1

    def test_batch_of_adds_reports_its_size(self, monkeypatch):
        seen = self._accepted(monkeypatch, {"rev": 3, "entries": [
            {"section": "verified", "op": "add", "text": "A"},
            {"section": "core", "op": "add", "text": "B"},
        ]})
        assert seen.get("n_edits") == 2, seen

    def test_nothing_logged_when_rev_is_current(self, monkeypatch):
        """阴性对照：rev 没落后就不该出现在这个桶里（否则分子会被灌水）。"""
        seen = self._accepted(monkeypatch, {
            "section": "verified", "op": "add", "text": "x", "rev": 5})
        assert seen == {}, seen

    def test_nothing_logged_when_rev_is_absent(self, monkeypatch):
        """不带 rev = 旧调用面，本就不判锁，也不该计进这个桶。"""
        seen = self._accepted(monkeypatch, {
            "section": "verified", "op": "add", "text": "x"})
        assert seen == {}, seen


class TestAddOnlyPredicate:
    """纯函数半边（保守方向：认不出的形态返回 False = 照旧判 rev）。"""

    def test_shapes(self):
        assert update_is_add_only({"section": "next", "op": "add"}) is True
        assert update_is_add_only({"section": "next", "op": "remove"}) is False
        assert update_is_add_only({"goal": "g"}) is False
        assert update_is_add_only({"entries": [
            {"section": "next", "op": "add"},
            {"section": "core", "op": "add"}]}) is True
        assert update_is_add_only({"entries": [
            {"section": "next", "op": "add"},
            {"section": "core", "op": "remove"}]}) is False

    def test_unreadable_shapes_fall_back_to_locking(self):
        assert update_is_add_only({}) is False
        assert update_is_add_only({"entries": ["not a dict"]}) is False
        assert update_is_add_only({"entries": [{"section": "next"}]}) is False
