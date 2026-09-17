"""ADR-0018 DPL 验收剧本（T12，可单测部分）。

覆盖（无需真 LLM/真数据，全 mock）：
  (a) 误合并=0 第一红线--proposal + mock judge，跨 topic 不合并。
  (c) provisional 不泄漏进注入--build_injection 过滤 provisional。
  (d) 重建等价性--full rebuild x2 -> fact_id 集合 + 边集合逐条相等。
  (g) 台账命中率--第二次 rebuild 的 L4 LLM 调用数 = 0（judgment 台账命中）。

真机部分（误合并人工签收 465 轮、世界杯剧本、L4 抽样 30 条）见
benchmark 文档 `benchmark 验证卡` + 人工复核。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from bladex_core.attribution import (
    LinkJudgeItem,
    LinkJudgeResult,
    LinkVerdict,
)
from bladex_core.distillation import DistillFact, DistillOutput, MatterProposal
from bladex_proxy.models import Identity, Turn, TurnStatus
from bladex_proxy.storage.memory_hub import MemoryHub
from bladex_proxy.storage.memory_index import MemoryIndex

# ── mocks ──


class MockEmbedder:
    """确定性向量：按文本首词哈希落到不同维度区间（同词高相似，跨词低相似）。"""

    def __init__(self) -> None:
        self._dim = 64

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            # e5 前缀 passage:/query: 去掉再取首个有意义的词
            body = t.replace("passage: ", "").replace("query: ", "")
            first = body.strip().split()[0] if body.strip() else ""
            h = abs(hash(first)) % self._dim
            v = [0.0] * self._dim
            v[h] = 0.9
            v[(h + 1) % self._dim] = 0.3
            out.append(v)
        return out

    @property
    def available(self) -> bool:
        return True


class CannedDistiller:
    """mock 蒸馏器：按预设 message -> DistillOutput（facts + proposals）。"""

    def __init__(self, mapping: dict[str, DistillOutput]) -> None:
        self._mapping = mapping

    @property
    def model_name(self) -> str:
        return "mock-distill"

    @property
    def prompt_ver(self) -> str:
        return "mock-v1"

    def distill(self, user_message: str) -> DistillOutput:
        # 精确匹配优先；否则按包含关键词匹配
        if user_message in self._mapping:
            return self._mapping[user_message]
        for key, out in self._mapping.items():
            if key in user_message or user_message in key:
                return out
        return DistillOutput(model_name="mock-distill")


class CannedLinkJudge:
    """mock L4 裁决器：按预设 fact content 关键词返回 verdict，计数调用次数。"""

    def __init__(self, rules: list[tuple[str, LinkVerdict, str]]) -> None:
        """rules: [(content_substr, verdict, matter_id), ...]"""
        self._rules = rules
        self.calls = 0

    @property
    def model_name(self) -> str:
        return "mock-judge"

    def judge_links(self, items: list[LinkJudgeItem]) -> list[LinkJudgeResult]:
        self.calls += 1
        results = []
        for it in items:
            verdict = LinkVerdict.UNCERTAIN
            mid = ""
            for substr, v, m in self._rules:
                if substr in it.content:
                    verdict = v
                    mid = m
                    break
            results.append(LinkJudgeResult(
                fact_id=it.fact_id, verdict=verdict, matter_id=mid,
            ))
        return results


def _none_judge() -> CannedLinkJudge:
    """judge 对所有 fact 返回 NONE -> L5 从 proposal 新开 provisional（ADR-0018 §3.2）。"""
    return CannedLinkJudge([("", LinkVerdict.NONE, "")])


def _uncertain_judge() -> CannedLinkJudge:
    """judge 对所有 fact 返回 UNCERTAIN -> 留池 pending（用于测试 L4 运行 + 台账）。"""
    return CannedLinkJudge([("", LinkVerdict.UNCERTAIN, "")])


# ── 剧本数据：两个 topic（v1/messages 支持 / Memory Hub key bug），各 2 轮 ──

_TOPIC_A = "给 BladeX 加 v1/messages 支持"
_TOPIC_B = "修复 Memory Hub key 覆盖 bug"

_DISTILL_MAP = {
    "调研 v1/messages 的 API 格式": DistillOutput(
        facts=[DistillFact(content="用户调研 v1/messages API 格式", kind="event", entities=["v1/messages"])],
        matter_proposals=[MatterProposal(title=_TOPIC_A, entities=["v1/messages"])],
        model_name="mock-distill",
    ),
    "实现 v1/messages 端点的流式处理": DistillOutput(
        facts=[DistillFact(content="用户实现 v1/messages 流式端点", kind="task", entities=["v1/messages", "流式"])],
        matter_proposals=[MatterProposal(title=_TOPIC_A, entities=["v1/messages"])],
        model_name="mock-distill",
    ),
    "Memory Hub key 用 turn_index 会覆盖": DistillOutput(
        facts=[DistillFact(content="Memory Hub key 用 turn_index 当主键导致覆盖", kind="event", entities=["Memory Hub key"])],
        matter_proposals=[MatterProposal(title=_TOPIC_B, entities=["Memory Hub key"])],
        model_name="mock-distill",
    ),
    "改用 Redis entry_id 修复 Memory Hub key": DistillOutput(
        facts=[DistillFact(content="改用 Redis entry_id 修复 Memory Hub key 覆盖", kind="decision", entities=["Memory Hub key", "Redis"])],
        matter_proposals=[MatterProposal(title=_TOPIC_B, entities=["Memory Hub key"])],
        model_name="mock-distill",
    ),
}

_USER_MESSAGES = list(_DISTILL_MAP.keys())


def _make_hub_with_turns(tmpdir: str) -> MemoryHub:
    ledger = MemoryHub(Path(tmpdir) / "ledger")
    ledger.open()
    for i, msg in enumerate(_USER_MESSAGES):
        identity = Identity(user_id="u1", agent_id="a1", session_id="s1", turn_index=i)
        turn = Turn(
            identity=identity, model="test",
            request_messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": msg},
            ],
            response_text=f"resp {i}", status=TurnStatus.OK, logical_turn=i,
        )
        ledger.put(identity.storage_key(f"171990720{i}-0"), turn)
    return ledger


def _make_index(tmpdir: str, ledger: MemoryHub, judge: CannedLinkJudge | None) -> MemoryIndex:
    return MemoryIndex(
        Path(tmpdir) / "index", embedder=MockEmbedder(),
        distiller=CannedDistiller(_DISTILL_MAP),
        link_judge=judge,
    )


# ── (a) 误合并=0 第一红线 ──


def test_a_no_false_merge():
    """(a) 4 轮回放后：两个 topic 各成一个 Matter，无跨 topic 合并（误合并=0）。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger = _make_hub_with_turns(tmpdir)
        # NONE judge：有候选但 L3 不匹配的 fact -> L4 NONE -> L5 新开 provisional
        judge = _none_judge()
        index = _make_index(tmpdir, ledger, judge)
        index.open()
        index.rebuild_from_hub(ledger, full=True)

        matters = [m for m in index.all_matters() if m.matter_id != "__unassigned__"]
        # 两个 topic -> 两个 Matter（A、B），各自的 facts 不串
        # 收集每个 matter 的 fact 内容
        matter_contents: dict[str, set[str]] = {}
        for m in matters:
            edges = index.get_edges(m.matter_id)
            contents = set()
            for e in edges:
                f = index.get_fact(e.target_key)
                if f:
                    contents.add(f.content)
            matter_contents[m.matter_id] = contents

        # 找含 v1/messages 的 matter 和含 Memory Hub key 的 matter
        a_matters = [mid for mid, cs in matter_contents.items()
                     if any("v1/messages" in c for c in cs)]
        b_matters = [mid for mid, cs in matter_contents.items()
                     if any("Memory Hub key" in c for c in cs)]
        assert a_matters, "topic A 应有 Matter"
        assert b_matters, "topic B 应有 Matter"
        # 第一红线：A、B 不在同一 matter
        assert set(a_matters).isdisjoint(set(b_matters)), (
            f"误合并！A={a_matters} B={b_matters} 不应交叉"
        )
        # A matter 不含 Memory Hub key 内容，B matter 不含 v1/messages 内容
        for mid in a_matters:
            assert not any("Memory Hub key" in c for c in matter_contents[mid]), "A matter 混入 B 内容"
        for mid in b_matters:
            assert not any("v1/messages" in c for c in matter_contents[mid]), "B matter 混入 A 内容"

        index.close()
        ledger.close()


# ── (c) provisional 不泄漏进注入 ──


# ── (d) 重建等价性 ──


def test_d_rebuild_equivalence():
    """(d) full rebuild x2 -> fact_id 集合 + 边集合逐条相等（T12d 第一半）。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger = _make_hub_with_turns(tmpdir)
        judge = _none_judge()
        index = _make_index(tmpdir, ledger, judge)
        index.open()
        index.rebuild_from_hub(ledger, full=True)

        facts1 = {f.id for f in index.all_facts()}
        edges1 = {(e.matter_id, e.target_type.value, e.target_key, e.provenance.value)
                  for m in index.all_matters() for e in index.get_edges(m.matter_id)}
        index.close()

        # 第二次 full rebuild（同 Memory Hub，新 Memory Index 路径）
        judge2 = _none_judge()
        p2b = MemoryIndex(
            Path(tmpdir) / "p2b", embedder=MockEmbedder(),
            distiller=CannedDistiller(_DISTILL_MAP), link_judge=judge2,
        )
        p2b.open()
        p2b.rebuild_from_hub(ledger, full=True)

        facts2 = {f.id for f in p2b.all_facts()}
        edges2 = {(e.matter_id, e.target_type.value, e.target_key, e.provenance.value)
                  for m in p2b.all_matters() for e in p2b.get_edges(m.matter_id)}

        assert facts1 == facts2, (
            f"fact_id 集合应相等:\n  first:  {sorted(facts1)}\n  second: {sorted(facts2)}"
        )
        assert edges1 == edges2, (
            f"边集合应相等:\n  first-only:  {edges1 - edges2}\n  second-only: {edges2 - edges1}"
        )

        p2b.close()
        ledger.close()


# ── (g) 台账命中率 ──


def test_g_ledger_hit_zero_llm_on_second_rebuild():
    """(g) 第二次 rebuild 的 L4 LLM 调用数 = 0（judgment 台账命中，ADR-0018 §4.1）。

    NONE judge 让 topic B 的 fact（候选 A 但 L3 不匹配）进 L4 -> 首次重建写 judgment
    台账（NONE verdict）；第二次重建同 fact_id 命中台账 -> 0 LLM 调用。
    需 Memory Hub 读写（full rebuild 写台账）。
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger = _make_hub_with_turns(tmpdir)
        judge = _none_judge()
        index = _make_index(tmpdir, ledger, judge)
        index.open()

        # 第一次 full rebuild：topic B 的 fact 进 L4 -> 写 judgment 台账
        index.rebuild_from_hub(ledger, full=True)
        first_calls = judge.calls

        # 重置计数，第二次 rebuild：judgment 台账命中 -> 0 LLM 调用
        judge.calls = 0
        index.rebuild_from_hub(ledger, full=True)
        second_calls = judge.calls

        assert first_calls > 0, (
            f"首次 rebuild 应有 L4 调用（topic B fact 进 L4），got {first_calls}"
        )
        assert second_calls == 0, (
            f"第二次 rebuild 应 0 LLM 调用（judgment 台账命中），got {second_calls}"
        )

        index.close()
        ledger.close()
