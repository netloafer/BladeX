"""T-A（2026-08-13，HANDOFF-20260813 §2）：v4/v5 主路径蒸馏并发。

## 为什么单独立这个文件

既有 `test_distill_concurrency.py` 的 SlowDistiller **没有实现 `distill_turn`**，
于是它测的全是 v3 兜底路径（`_bulk_distill`）——而生产走 v4
（`BLADEX_DISTILL_V4` 默认开 + LLMDistiller 有 `distill_turn`）。
"并发有测试、测试全绿、生产串行 6.2h"正是这么发生的：
测试桩的能力面与生产实现不一致，测的就不是生产走的那条路。

本文件的桩**实现 `distill_turn`**，钉死 v4 两阶段（Pass1 构造 → 去重 →
并发蒸馏 map → Pass3 原顺序装配）与串行的产出等价（G6）。

验收（HANDOFF §2）：同一批语料"并发跑"与"串行跑"产出的 **fact id 集合逐字一致**。
"""

from __future__ import annotations

import hashlib
import threading
import time

from bladex_core.consolidation_proxy import ProxyConsolidator
from bladex_core.distillation import DistillFact, DistillOutput
from bladex_core.fact import ConversationTurn


class MockEmbedder:
    """确定性 embedder。

    用 sha256 而非内置 hash：字符串 hash 每进程随机化，桶碰撞会让不同内容
    被判重复、测试随机翻红（test_ledger_to_candidates 的同款教训）。
    """

    def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * 64
            h = int.from_bytes(hashlib.sha256(t.encode()).digest()[:4], "big") % 64
            v[h] = 0.9
            out.append(v)
        return out


class SlowTurnDistiller:
    """v4 桩：实现 `distill_turn`，确定性输出 + 微延迟 + 并发峰值探针。

    产出内容只由 payload 内容决定（不含时间/计数），保证同输入同输出——
    这是"并发 vs 串行逐字一致"断言成立的前提（生产侧由台账 write-if-absent
    与 temperature=0 承担同一角色）。
    """

    model_name = "mock/turnslow"
    prompt_ver = "vtest"

    def __init__(self, fail_user_texts: set[str] | None = None) -> None:
        self.turn_calls = 0
        self.seen_payloads: list = []
        self._fail_user_texts = fail_user_texts or set()
        self._active = 0
        self.peak = 0
        self._lock = threading.Lock()

    def distill(self, text: str) -> DistillOutput:  # v3 面（不应被走到）
        raise AssertionError("v4 桩不该被 v3 路径调用")

    def distill_turn(self, payload) -> DistillOutput:
        with self._lock:
            self._active += 1
            self.peak = max(self.peak, self._active)
            self.turn_calls += 1
            self.seen_payloads.append(payload)
        try:
            time.sleep(0.02)
            if payload.user_text in self._fail_user_texts:
                return DistillOutput(failed=True, failure_kind="call",
                                     model_name=self.model_name)
            facts = []
            if payload.user_text:
                facts.append(DistillFact(
                    content=f"U::{payload.user_text[:60]}",
                    kind="general", entities=[]))
            if payload.assistant_text:
                facts.append(DistillFact(
                    content=f"A::{payload.assistant_text[:60]}",
                    kind="general", entities=[]))
            return DistillOutput(facts=facts, model_name=self.model_name)
        finally:
            with self._lock:
                self._active -= 1


def _turns(n: int = 6) -> list[ConversationTurn]:
    out = []
    for i in range(n):
        out.append(ConversationTurn(
            session_id="s1", user_id="u1", ledger_key=f"k{i}", logical_turn=i,
            timestamp=f"2026-08-13T10:0{i}:00",
            user_messages=[f"第 {i} 条足够长的用户消息，讨论互不相同的主题编号 {i}。"],
            assistant_conclusion=(f"第 {i} 个任务的结论：产出了模块 {i} 并通过全部验证，"
                                  f"关键参数最终定为 {i}，相关阈值与实现细节均已确认落库。"),
        ))
    # 跨轮重复话语（工具循环 roundtrip 形态）：两轮同文、都无 assistant 正文
    # → 两个 payload 的 (context_digest, cache_text) 完全相同，Pass2 应收敛为一次。
    dup_text = "这句话在工具循环的多个往返里逐字重复出现，属于同一个用户提问。"
    for j in range(2):
        out.append(ConversationTurn(
            session_id="s1", user_id="u1", ledger_key=f"kdup{j}",
            logical_turn=n + j, timestamp=f"2026-08-13T10:0{8 + j}:00",
            user_messages=[dup_text],
        ))
    return out


def _run(concurrency: int, turns, distiller=None):
    d = distiller or SlowTurnDistiller()
    c = ProxyConsolidator(embedder=MockEmbedder(), distiller=d,
                          distill_concurrency=concurrency)
    facts = c.consolidate_turns(turns, existing_facts=[])
    return facts, d, c


# ── 验收主剧本：并发 vs 串行 fact id 集合逐字一致（HANDOFF §2）─────────────────


def test_v4_concurrent_fact_ids_identical_to_serial():
    turns = _turns()
    serial_facts, _, _ = _run(1, turns)
    concurrent_facts, d, _ = _run(4, turns)

    # id **列表**（含顺序）逐字一致，比"集合一致"的验收口径更严
    assert [f.id for f in concurrent_facts] == [f.id for f in serial_facts]
    assert [f.content for f in concurrent_facts] == [f.content for f in serial_facts]
    assert serial_facts, "空产出的'一致'没有意义"
    # 真的并发跑了（不是伪并发——`--concurrency` 接在死路上就是这么潜伏的）
    assert d.peak > 1, "concurrency=4 时蒸馏并发峰值应 >1，v4 主路径没接上并发"


def test_v4_serial_path_never_builds_thread_pool():
    """默认并发=1：v4 仍走原串行形态（构造与蒸馏逐轮交错，零回归）。"""
    _, d, _ = _run(1, _turns(3))
    assert d.peak == 1
    assert d.turn_calls > 0


def test_v4_concurrent_dedupes_repeated_payloads():
    """同 (context_digest, cache_text) 只蒸一次（Pass2 去重）。

    _turns() 里 kdup0/kdup1 同文、都无 assistant 正文与 prior 上下文：
    串行各蒸一次（memo 不可用时），并发收敛为一次——产出仍逐字一致
    （确定性桩下两次调用输出相同，装配只查表；重复候选由批内 novelty 判重收掉，
    且串行/并发收得一样，fact id 一致性由主剧本守）。
    """
    turns = _turns(3)   # 3 轮常规 + 2 轮同文重复
    _, d_serial, _ = _run(1, turns)
    _, d_conc, _ = _run(4, turns)
    assert d_serial.turn_calls == 5          # 3 常规 + 2 重复各蒸
    assert d_conc.turn_calls == 4            # 重复对收敛为一次


def test_v4_concurrent_failure_recorded_same_as_serial():
    """失败路径长得和成功不一样（纪律 5）：failed 产出 → failed_ledger_keys 记账，
    该轮不产候选；串行与并发行为一致。
    """
    turns = _turns(4)
    fail_text = turns[2].user_messages[0]

    sf, _, cs = _run(1, turns, SlowTurnDistiller(fail_user_texts={fail_text}))
    cf, _, cc = _run(4, turns, SlowTurnDistiller(fail_user_texts={fail_text}))

    assert "k2" in cs.failed_ledger_keys
    assert cs.failed_ledger_keys == cc.failed_ledger_keys
    assert [f.id for f in cf] == [f.id for f in sf]
    assert all("U::" + fail_text[:10] not in f.content for f in cf)


def test_v4_progress_cb_receives_monotonic_events():
    events: list[dict] = []
    c = ProxyConsolidator(embedder=MockEmbedder(), distiller=SlowTurnDistiller(),
                          distill_concurrency=4, progress_cb=events.append)
    c.consolidate_turns(_turns(4), existing_facts=[])
    distill_ev = [e for e in events if e.get("phase") == "distill"]
    assert distill_ev, "v4 并发路径应上报 distill 进度（consolidator 终端进度依赖它）"
    dones = [e["done"] for e in distill_ev]
    assert dones == sorted(dones) and dones[-1] == distill_ev[-1]["total"]


def test_v4_turn_events_and_route_counts_preserved_under_concurrency():
    """观测面不因并发丢失：route_counts / turn_events 与串行一致（MS-4 消费方在 rebuild）。"""

    class EventDistiller(SlowTurnDistiller):
        def distill_turn(self, payload) -> DistillOutput:
            out = super().distill_turn(payload)
            for f in out.facts:
                f.event = "progress"
            return out

    turns = _turns(3)
    _, _, cs = _run(1, turns, EventDistiller())
    _, _, cc = _run(4, turns, EventDistiller())
    assert dict(cs.route_counts) == dict(cc.route_counts)
    assert cs.turn_events == cc.turn_events
    assert cc.turn_events, "轮级事件在并发路径下不得丢失"
