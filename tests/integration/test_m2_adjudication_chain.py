"""M2 归一层集成：Memory Hub → 蒸馏 → 判重 → **裁决** → 入库 + judgment 台账。

conventions §5.1：新增的"生产者→消费者"接线必须带一条从 Memory Hub Turn 出发的
端到端测试。裁决横跨 core（决策）与 proxy（落盘 + 台账），正是最容易"两端都对、
中间断了"的形状——M2 的 verdict 若没被应用，库里看起来只是"多了一条"，无任何告警。

同时覆盖 MS-5 ①（事件按主归属折叠一次）与 MS-7（NOOP → strength+1）。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from bladex_core.adjudication import AdjudicationOp, AdjudicationVerdict
from bladex_proxy.models import Identity, Turn, TurnStatus
from bladex_proxy.storage.memory_hub import MemoryHub
from bladex_proxy.storage.memory_index import MemoryIndex


class _Embedder:
    """确定性桩：把不同文本放进 **0.90 ~ 0.98 之间**的相似度带。

    这个带宽是本测试成立的前提，不是随手取的参数：

        cos ≥ 0.98  → novelty 直接判重，**候选到不了裁决**（首版桩就死在这里：
                      主方向权重给太大，任意两条 cos ≈ 0.9999，2 轮只入库 1 条）
        cos < 0.90  → 不进裁决包（`BLADEX_ADJUDICATE_MIN_SIM`），退化成无邻居 ADD

    真实数据的形态正在这个带里：复核 3.9 实测修正宣告与它推翻的旧条 cos=0.956。
    噪声系数 0.16 是实测出来的（这两条样本 → 0.9295），不是算出来的。
    """

    _DIM = 32
    _NOISE = 0.16

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()
            v = [1.0] + [(b / 255.0) * self._NOISE for b in h[:self._DIM - 1]]
            out.append(v)
        return out

    @property
    def available(self) -> bool:
        return True


class _Distiller:
    """每轮产一条 fact；第二轮带 corrects + event（供 MS-4/MS-5 断言）。"""

    model_name = "stub"

    def distill(self, text: str):
        return self.distill_turn(type("P", (), {
            "user_text": text, "assistant_text": "", "assistant_role": "",
            "context_digest": "", "turn_time": "",
            "cache_text": lambda self=None: text})())

    def distill_conclusion(self, text: str):
        return self.distill(text)

    def distill_turn(self, payload):
        from bladex_core.distillation import DistillFact, DistillOutput

        text = (payload.user_text or payload.assistant_text or "").strip()
        if not text:
            return DistillOutput(model_name=self.model_name)
        # 🔴 subject/attribute **刻意不同**：相同就会命中精确取代键的快路径，
        # 裁决器根本不会被调用（那是正确行为，但这组测试要验的是 LLM 那条路）。
        # E3 也说明真实蒸馏产出里 subject/attribute 极少精确对齐，
        # 所以"走裁决"才是主路径，"走快路径"是少数。
        tag = hashlib.sha256(text.encode()).hexdigest()[:6]
        f = DistillFact(content=f"事实::{text[:60]}", item_kind="assertion",
                        subject=f"主题{tag}", attribute=f"槽位{tag}",
                        provenance="user_direct")
        if "修正" in text:
            f.corrects = "先前的说法"
            f.event = "reworked"
        return DistillOutput(facts=[f], model_name=self.model_name)


class _Adjudicator:
    """记录收到的裁决包；对带 corrects 的候选判 UPDATE，其余 ADD。"""

    model_name = "adj-stub"
    prompt_ver = "test"

    def __init__(self) -> None:
        self.seen: list = []

    def adjudicate(self, items):
        self.seen.extend(items)
        out = []
        for it in items:
            if it.corrects_hint and it.neighbors:
                out.append(AdjudicationVerdict(
                    candidate_id=it.candidate_id, op=AdjudicationOp.UPDATE,
                    target_fact_ids=[it.neighbors[0].fact_id], reason="corrected"))
            else:
                out.append(AdjudicationVerdict(candidate_id=it.candidate_id,
                                               op=AdjudicationOp.ADD))
        return out


def _ledger(tmp: Path, messages: list[str]) -> MemoryHub:
    led = MemoryHub(tmp / "ledger")
    led.open()
    for i, msg in enumerate(messages):
        ident = Identity(user_id="u1", agent_id="a1", session_id="s1", turn_index=i)
        led.put(ident.storage_key(f"17199072{i:02d}-0"), Turn(
            identity=ident, model="m",
            request_messages=[{"role": "user", "content": msg}],
            response_text="ok", status=TurnStatus.OK, logical_turn=i,
        ))
    return led


_MSGS = [
    "这件事按原本的理解是那样处理的没有别的说法需要记住",
    "修正一下：这件事其实不是那样处理的，之前理解错了需要更新",
]


def test_adjudication_runs_and_is_persisted(tmp_path):
    """裁决必须真的跑起来、verdict 必须真的被应用到库里。

    "跑起来了"与"应用了"是两件事：verdict 算出来但没落盘时，
    库里看起来只是多了一条重复，**没有任何告警**——正是最难发现的那类断线。
    """
    led = _ledger(tmp_path, _MSGS)
    adj = _Adjudicator()
    index = MemoryIndex(tmp_path / "index", embedder=_Embedder(),
                        distiller=_Distiller(), adjudicator=adj)
    index.open()
    index.rebuild_from_hub(led, full=False, max_turns=50)

    assert adj.seen, "裁决器从未被调用 —— 接线断了"
    pkg = adj.seen[-1]
    assert pkg.neighbors, "裁决包里没有邻居，裁决无从判起"
    assert pkg.corrects_hint, "修正宣告没进裁决包（F4 说的最高精度信号）"

    facts = index.all_facts()
    invalidated = [f for f in facts if f.t_invalid is not None]
    assert invalidated, "UPDATE 判决没有被应用 —— 旧条仍是现行事实"
    assert all(f.superseded_by for f in invalidated), "取代链缺半边"
    index.close()
    led.close()


def test_judgment_journal_records_consolidation_verdicts(tmp_path):
    """裁决决策写 `judgment/` 台账（kind=consolidation）—— rebuild 重放零 LLM。"""
    led = _ledger(tmp_path, _MSGS)
    index = MemoryIndex(tmp_path / "index", embedder=_Embedder(),
                        distiller=_Distiller(), adjudicator=_Adjudicator())
    index.open()
    index.rebuild_from_hub(led, full=False, max_turns=50)

    rows = index.scan_judgments()
    kinds = [rec.verdict for _k, rec in rows]
    assert any(v.startswith("consolidation:") for v in kinds), (
        f"judgment 台账里没有 consolidation 决策：{kinds}")
    index.close()
    led.close()


def test_noop_bumps_strength_without_growing_the_library(tmp_path):
    """MS-7：同义重述 → 旧条 strength+1，库条数不变。"""
    class _NoopAdj(_Adjudicator):
        def adjudicate(self, items):
            self.seen.extend(items)
            return [AdjudicationVerdict(
                candidate_id=it.candidate_id, op=AdjudicationOp.NOOP,
                target_fact_ids=[it.neighbors[0].fact_id]) if it.neighbors
                else AdjudicationVerdict(candidate_id=it.candidate_id,
                                         op=AdjudicationOp.ADD)
                for it in items]

    led = _ledger(tmp_path, _MSGS)
    index = MemoryIndex(tmp_path / "index", embedder=_Embedder(),
                        distiller=_Distiller(), adjudicator=_NoopAdj())
    index.open()
    index.rebuild_from_hub(led, full=False, max_turns=50)

    facts = index.all_facts()
    assert len(facts) == 1, f"NOOP 不该增加库条数，实得 {len(facts)} 条"
    assert facts[0].strength == 2, "重复陈述 = 强化，strength 必须涨"
    assert facts[0].last_hit_at is None, "MS-7：NOOP 不动相关钟"
    index.close()
    led.close()


def test_adjudication_off_falls_back(tmp_path, monkeypatch):
    """回滚通道：关掉开关退回旧形态（supersede + 冲突检测），裁决器不被调用。"""
    monkeypatch.setenv("BLADEX_ADJUDICATE_ENABLED", "0")
    led = _ledger(tmp_path, _MSGS)
    adj = _Adjudicator()
    index = MemoryIndex(tmp_path / "index", embedder=_Embedder(),
                        distiller=_Distiller(), adjudicator=adj)
    index.open()
    index.rebuild_from_hub(led, full=False, max_turns=50)
    assert not adj.seen, "关掉后不该再调裁决器"
    index.close()
    led.close()


def test_session_event_folds_once_by_majority_matter(tmp_path):
    """MS-5 ①：一轮的事件只折进**主归属** Matter 一次。

    旧形态是每条 fact 各折一次——同一轮 facts 落到两张卡时，
    同一个事件被折进两张（H2「指派消息进了无关 Matter」的形态）。
    """
    led = _ledger(tmp_path, _MSGS)
    index = MemoryIndex(tmp_path / "index", embedder=_Embedder(),
                        distiller=_Distiller(), adjudicator=_Adjudicator())
    index.open()
    index.rebuild_from_hub(led, full=False, max_turns=50)

    folded = [(m.matter_id, ev.event)
              for m in index.all_matters()
              for ev in (m.lifecycle or [])]
    # 同一个事件不得出现在多张卡上
    by_event: dict[str, set[str]] = {}
    for mid, ev in folded:
        by_event.setdefault(ev, set()).add(mid)
    for ev, mids in by_event.items():
        assert len(mids) == 1, f"事件 {ev} 被折进了 {len(mids)} 张卡：{mids}"
    index.close()
    led.close()
