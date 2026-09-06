"""欠采修复的验收（2026-08-08，M5-2 后真实数据驱动）。

## 修的是什么

`_collect_candidates_v4` 原来一轮只蒸 `discourse[-1]`，其余降为 context_digest，
注释给的理由是「历史条目在它们**自己那轮**已经被蒸过」。

那个前提要求**每条话语都当过某一轮的最后一条**。全库实测
（`measure_discourse_undersampling.py`，2537 轮）：distinct 话语 347 条，
**111 条（32.0%）从未当过**，`hermes:c7047465` 更是 10/10 全丢。

丢的不是寒暄，是「AMD Helios 会不会威胁英伟达」「深度调查 unabyssapp 与 BladeX
对比」这类驱动过整条研究线的实质提问 —— 每条都是真实记忆损失，
**且没有任何报错**，所以潜伏至今。

四种情形让中间消息永远轮不到当最后一条：会话中断后 agent 一次带回 N 条历史 /
上下文压缩后重启 / agent 首次接入就带着既有对话 / 那一轮 failed 或被 aux 整轮过滤。

## 修法

意图不变（别重复蒸历史），取的量从「最后一条」改成「**从未蒸过的**」——
「最后一条」≠「新出现的」，这是原实现的错。判重靠纯内容维度的 memo，
不能用蒸馏台账（它的 key 故意含 context_digest，拿来当 memo 会次次 miss）。
"""

from __future__ import annotations

from bladex_core.consolidation_proxy import ProxyConsolidator
from bladex_core.distillation import DistillFact, DistillOutput
from bladex_core.fact import ConversationTurn

_Q1 = "小黑，帮我分析一下，AMD新发布的helios系统会对英伟达构成威胁吗？"
_Q2 = "Helios这个咱们再深度分析一下，目前出货量预估会是多少？是订单不足还是产能瓶颈？"
_Q3 = "unabyssapp也在提one memory的多agent方案，深度调查一下，跟BladeX做个对比。"


class _MemoDistiller:
    """带 memo 能力的蒸馏替身（对应生产里的 LLMDistiller + IndexDistillJournal）。"""

    model_name = "test-model"
    prompt_ver = "v4-turn-001"

    def __init__(self, *, fail_texts: set[str] | None = None) -> None:
        self.seen_payloads: list[str] = []
        self._memo: set[str] = set()
        self._fail = fail_texts or set()

    def distill_turn(self, payload) -> DistillOutput:  # noqa: ANN001
        text = payload.user_text or payload.assistant_text or ""
        self.seen_payloads.append(text)
        if text in self._fail:
            out = DistillOutput(facts=[], matter_proposals=[], model_name="test-model")
            out.failed = True     # 与生产同款：失败降级不抛
            return out
        return DistillOutput(
            facts=[DistillFact(content=f"f::{text[:30]}", kind="event", entities=["e"])],
            matter_proposals=[], model_name="test-model",
        )

    # core 用 getattr 探测这两个（单参：只问内容）
    def discourse_seen(self, text: str) -> bool:
        return text in self._memo

    def mark_discourse(self, text: str, ledger_key: str = "") -> None:
        self._memo.add(text)


class _LegacyDistiller:
    """**没有** memo 能力的替身 —— 旧部署 / 旧测试替身的形态。"""

    model_name = "test-model"

    def __init__(self) -> None:
        self.seen_payloads: list[str] = []

    def distill_turn(self, payload) -> DistillOutput:  # noqa: ANN001
        self.seen_payloads.append(payload.user_text or payload.assistant_text or "")
        return DistillOutput(
            facts=[DistillFact(content="f", kind="event", entities=["e"])],
            matter_proposals=[], model_name="test-model",
        )


def _turn(msgs: list[str], key: str = "k1", resp: str = "") -> ConversationTurn:
    return ConversationTurn(
        session_id="s", user_id="u", ledger_key=key,
        user_messages=list(msgs), assistant_response=resp,
        assistant_conclusion=resp, timestamp="2026-08-08T00:00:00+00:00",
    )


def _distilled_user_texts(d) -> list[str]:  # noqa: ANN001
    return [t for t in d.seen_payloads if t]


# ── ① 核心：历史话语不再被静默丢弃 ────────────────────────────────────


def test_history_messages_are_distilled_not_silently_dropped():
    """agent 一次带回三条历史 —— 三条都该进蒸馏，而不是只有最后一条。

    这正是「会话中断后重连」的形态：前两条从来没当过某轮的最后一条。
    """
    d = _MemoDistiller()
    ProxyConsolidator(distiller=d)._collect_candidates([_turn([_Q1, _Q2, _Q3])])
    got = _distilled_user_texts(d)
    assert _Q1 in got, "第一条历史被丢了 —— 正是 32% 欠采的形态"
    assert _Q2 in got, "第二条历史被丢了"
    assert _Q3 in got


def test_already_distilled_history_is_not_redistilled():
    """第二轮把同样的历史又回传一遍 —— 不该重复蒸（那是原设计要省的成本）。"""
    d = _MemoDistiller()
    c = ProxyConsolidator(distiller=d)
    c._collect_candidates([_turn([_Q1, _Q2], key="k1")])
    first_round = len(d.seen_payloads)
    c._collect_candidates([_turn([_Q1, _Q2, _Q3], key="k2")])
    added = [t for t in d.seen_payloads[first_round:] if t]
    assert _Q3 in added, "新消息该蒸"
    assert _Q1 not in added and _Q2 not in added, "旧消息不该重复蒸"


def test_failed_distillation_is_not_marked_seen():
    """🔴 蒸馏失败不许标记 —— 否则一次上游抖动让这条话语**永久**跳过。

    症状会是"库里少了点东西"且无任何报错，正是本次要根治的形态。
    """
    d = _MemoDistiller(fail_texts={_Q1})
    c = ProxyConsolidator(distiller=d)
    c._collect_candidates([_turn([_Q1], key="k1")])
    assert not d.discourse_seen(_Q1), "失败的不该进 memo"
    c._collect_candidates([_turn([_Q1, _Q2], key="k2")])
    assert d.seen_payloads.count(_Q1) == 2, "下一轮该重试它"


# ── ② 不回归：没有 memo 能力时行为逐字不变 ──────────────────────────


def test_legacy_distiller_falls_back_to_last_message_only():
    """替身没有 memo 方法 → 整体降级回「只蒸最后一条」，逐字不变。

    这条守的是**兼容面**：旧部署与既有测试替身不该因为本次修复而改变行为。
    """
    d = _LegacyDistiller()
    ProxyConsolidator(distiller=d)._collect_candidates([_turn([_Q1, _Q2, _Q3])])
    got = _distilled_user_texts(d)
    assert _Q3 in got
    assert _Q1 not in got and _Q2 not in got, "无 memo 能力时不该改变取样"


def test_memo_read_failure_degrades_to_old_behaviour():
    """memo 读失败不该阻断蒸馏 —— 只降级，不炸。"""
    class _Broken(_MemoDistiller):
        def discourse_seen(self, text: str) -> bool:
            raise RuntimeError("memo store down")

    d = _Broken()
    cands = ProxyConsolidator(distiller=d)._collect_candidates([_turn([_Q1, _Q2])])
    assert _Q2 in _distilled_user_texts(d), "最后一条仍要蒸"
    assert cands, "不该因为 memo 故障就一条候选都不产"


# ── ③ 重放等价性（G6）──────────────────────────────────────────────


def test_replay_is_deterministic_across_batching():
    """同样的数据分批喂 vs 一次喂，蒸馏输入集合必须一致。

    这是选「内容寻址 memo」而不是「与上一轮做差集」的**唯一理由**：
    后者在增量模式下每批 30 轮，批次边界会改变「新」的定义 ——
    同一批数据分两次重放结果不同，直接破 G6。
    """
    turns = [_turn([_Q1], "k1"), _turn([_Q1, _Q2], "k2"), _turn([_Q1, _Q2, _Q3], "k3")]

    one = _MemoDistiller()
    ProxyConsolidator(distiller=one)._collect_candidates(list(turns))

    split = _MemoDistiller()
    c = ProxyConsolidator(distiller=split)
    c._collect_candidates(turns[:1])
    c._collect_candidates(turns[1:])

    assert sorted(_distilled_user_texts(one)) == sorted(_distilled_user_texts(split))


def test_each_distinct_discourse_distilled_exactly_once():
    """每条 distinct 话语**恰好蒸一次** —— 覆盖面与成本同时成立。

    修前：每轮蒸 1 条 × N 轮，distinct 覆盖不全（实测 68%）。
    修后：distinct 各一次 —— 覆盖 100%，调用量反而更低。
    """
    d = _MemoDistiller()
    c = ProxyConsolidator(distiller=d)
    for i, msgs in enumerate([[_Q1], [_Q1, _Q2], [_Q1, _Q2, _Q3], [_Q1, _Q2, _Q3]]):
        c._collect_candidates([_turn(msgs, key=f"k{i}")])
    user_texts = _distilled_user_texts(d)
    for q in (_Q1, _Q2, _Q3):
        assert user_texts.count(q) == 1, f"{q[:20]!r} 蒸了 {user_texts.count(q)} 次"
