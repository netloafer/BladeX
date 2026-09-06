"""蒸馏并发（sync CLI，2026-08-03）：并发预蒸馏与串行产出逐字一致（G6）。

真实路径：ProxyConsolidator.consolidate_turns 全流程（mock 蒸馏器带真实延迟 +
并发峰值探针），不 mock _bulk_distill 本身。
"""

from __future__ import annotations

import threading
import time

from bladex_core.consolidation_proxy import ProxyConsolidator
from bladex_core.distillation import DistillFact, DistillOutput
from bladex_core.fact import ConversationTurn


class MockEmbedder:
    def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * 32
            v[hash(t) % 32] = 1.0
            out.append(v)
        return out


class SlowDistiller:
    """确定性输出 + 微延迟 + 并发峰值探针。"""

    model_name = "mock/slow"
    prompt_ver = "vtest"

    def __init__(self) -> None:
        self.calls = 0
        self.concl_calls = 0
        self._active = 0
        self.peak = 0
        self._lock = threading.Lock()

    def _enter(self):
        with self._lock:
            self._active += 1
            self.peak = max(self.peak, self._active)

    def _exit(self):
        with self._lock:
            self._active -= 1

    def distill(self, text: str) -> DistillOutput:
        self._enter()
        try:
            self.calls += 1
            time.sleep(0.02)
            return DistillOutput(facts=[DistillFact(
                content=f"fact::{text[:40]}", kind="general", entities=[])],
                model_name=self.model_name)
        finally:
            self._exit()

    def distill_conclusion(self, text: str) -> DistillOutput:
        self._enter()
        try:
            self.concl_calls += 1
            time.sleep(0.02)
            return DistillOutput(facts=[DistillFact(
                content=f"concl::{text[:40]}", kind="general", entities=[])],
                model_name=self.model_name)
        finally:
            self._exit()


def _turns(n: int = 6) -> list[ConversationTurn]:
    out = []
    for i in range(n):
        out.append(ConversationTurn(
            session_id="s1", user_id="u1", ledger_key=f"k{i}",
            user_messages=[f"第 {i} 条足够长的用户消息，讨论互不相同的主题编号 {i}。"],
            assistant_conclusion=(f"第 {i} 个任务的结论：产出了模块 {i} 并通过全部验证，"
                                  f"关键参数最终定为 {i}，相关阈值与实现细节均已确认落库。"),
        ))
    return out


def _contents(consolidator: ProxyConsolidator, turns) -> list[str]:
    facts = consolidator.consolidate_turns(turns, existing_facts=[])
    return [f.content for f in facts]


def test_concurrent_output_identical_to_serial():
    turns = _turns()
    serial = _contents(ProxyConsolidator(
        embedder=MockEmbedder(), distiller=SlowDistiller(),
        distill_concurrency=1), turns)
    d = SlowDistiller()
    concurrent = _contents(ProxyConsolidator(
        embedder=MockEmbedder(), distiller=d,
        distill_concurrency=4), turns)
    assert concurrent == serial and serial          # 顺序与内容逐字一致（G6）
    assert d.peak > 1                               # 真的并发跑了（不是伪并发）


def test_concurrent_dedupes_repeated_texts():
    """重复文本只蒸一次（唯一文本映射）。"""
    turns = _turns(3)
    dup = ConversationTurn(session_id="s1", user_id="u1", ledger_key="kdup",
                           user_messages=[turns[0].user_messages[0]])
    d = SlowDistiller()
    ProxyConsolidator(embedder=MockEmbedder(), distiller=d,
                      distill_concurrency=4).consolidate_turns(
        turns + [dup], existing_facts=[])
    # 3 条唯一 user 文本 → 3 次 distill；4 条唯一结论 → ... turns 有 3 条结论
    assert d.calls == 3
    assert d.concl_calls == 3


def test_progress_cb_receives_monotonic_events():
    events: list[dict] = []
    turns = _turns(4)
    ProxyConsolidator(embedder=MockEmbedder(), distiller=SlowDistiller(),
                      distill_concurrency=4,
                      progress_cb=events.append).consolidate_turns(
        turns, existing_facts=[])
    distill_ev = [e for e in events if e.get("phase") == "distill"]
    assert distill_ev, "并发路径应上报 distill 进度"
    dones = [e["done"] for e in distill_ev]
    assert dones == sorted(dones) and dones[-1] == distill_ev[-1]["total"]


def test_progress_cb_exception_swallowed():
    def bad_cb(ev):
        raise RuntimeError("boom")

    facts = ProxyConsolidator(
        embedder=MockEmbedder(), distiller=SlowDistiller(),
        distill_concurrency=2, progress_cb=bad_cb,
    ).consolidate_turns(_turns(2), existing_facts=[])
    assert facts  # 回调爆炸不影响产出


def test_default_serial_path_unchanged():
    """默认并发=1：不建线程池、逐条串行（回归保证）。"""
    d = SlowDistiller()
    c = ProxyConsolidator(embedder=MockEmbedder(), distiller=d)
    c.consolidate_turns(_turns(3), existing_facts=[])
    assert d.peak == 1
