"""结论蒸馏验收剧本（ADR-0019 P2：结论蒸馏进 Memory Index —— 信息守恒第二条腿）。

覆盖：core consolidator 结论候选 / 能力探测降级 / 透传不污染 /
LLMDistiller.distill_conclusion 独立 prompt+台账版本 / rebuild 结论判定。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from bladex_core.consolidation_proxy import ProxyConsolidator
from bladex_core.distillation import (
    DistillFact,
    DistillOutput,
    MatterProposal,
    PassthroughDistiller,
)
from bladex_core.fact import ConversationTurn
from bladex_proxy.distillation import CONCLUSION_PROMPT_VER, PROMPT_VER, LLMDistiller
from bladex_proxy.models import Identity, ToolEvent, Turn, TurnStatus
from bladex_proxy.storage.memory_index import MemoryIndex
from bladex_proxy.storage.memory_hub import MemoryHub


class MockEmbedder:
    def embed(self, texts):
        results = []
        for t in texts:
            h = hash(t) % 100
            vec = [0.0] * 64
            vec[h % 64] = 1.0
            results.append(vec)
        return results

    @property
    def available(self):
        return True


class FakeDistiller:
    """带 distill_conclusion 能力的蒸馏器 mock。"""

    def __init__(self) -> None:
        self.conclusion_calls: list[str] = []

    @property
    def model_name(self) -> str:
        return "fake-llm"

    @property
    def prompt_ver(self) -> str:
        return "test-v1"

    def distill(self, user_message: str) -> DistillOutput:
        return DistillOutput(
            facts=[DistillFact(content=f"用户提到：{user_message[:30]}", kind="event")],
            model_name="fake-llm",
        )

    def distill_conclusion(self, conclusion: str) -> DistillOutput:
        self.conclusion_calls.append(conclusion)
        return DistillOutput(
            facts=[DistillFact(content="查询结果：今天有两场世界杯比赛", kind="event",
                               entities=["世界杯"])],
            matter_proposals=[MatterProposal(title="世界杯赛程查询", entities=["世界杯"])],
            model_name="fake-llm",
        )


class LegacyDistiller:
    """无 distill_conclusion 的旧蒸馏器（兼容性验证）。"""

    @property
    def model_name(self) -> str:
        return "legacy"

    @property
    def prompt_ver(self) -> str:
        return "legacy-v1"

    def distill(self, user_message: str) -> DistillOutput:
        return DistillOutput(
            facts=[DistillFact(content=user_message[:50], kind="general")],
            model_name="legacy",
        )


def _turn_with_conclusion(conclusion: str) -> ConversationTurn:
    return ConversationTurn(
        session_id="s1", user_id="u1",
        user_messages=["今天有世界杯的比赛吗？后面的赛程是怎么安排的，帮我查一下"],
        assistant_response=conclusion,
        assistant_conclusion=conclusion,
        ledger_key="u1/a1/s1/e1",
    )


# ── core：consolidator 结论候选 ──


def test_conclusion_facts_extracted_and_tagged():
    """结论 → distill_conclusion → Fact，tags=origin:conclusion 可审计。"""
    distiller = FakeDistiller()
    c = ProxyConsolidator(embedder=MockEmbedder(), distiller=distiller)
    turn = _turn_with_conclusion("经查询，今天世界杯有两场比赛：A vs B（20:00）、C vs D（23:00）。")
    facts = c.consolidate_turns([turn])

    conclusion_facts = [f for f in facts if f.tags == "origin:conclusion"]
    user_facts = [f for f in facts if f.tags == "origin:consolidation"]
    assert len(conclusion_facts) == 1
    assert conclusion_facts[0].content == "查询结果：今天有两场世界杯比赛"
    assert conclusion_facts[0].distill_model == "fake-llm"
    assert conclusion_facts[0].proposal_titles == ["世界杯赛程查询"]
    assert len(user_facts) == 1  # user 消息路径不受影响
    assert distiller.conclusion_calls  # 确实走了结论专用入口


def test_conclusion_skipped_when_distiller_lacks_capability():
    """旧蒸馏器（无 distill_conclusion）→ 结论跳过，user 路径照常（零改造兼容）。"""
    c = ProxyConsolidator(embedder=MockEmbedder(), distiller=LegacyDistiller())
    turn = _turn_with_conclusion("经查询，今天世界杯有两场比赛，分别在晚上八点和十一点。")
    facts = c.consolidate_turns([turn])
    assert all(f.tags != "origin:conclusion" for f in facts)
    assert len(facts) == 1  # 只有 user 消息的事实


def test_conclusion_skipped_without_distiller():
    """无蒸馏器 → 结论不透传入库（宁缺毋滥）。"""
    c = ProxyConsolidator(embedder=MockEmbedder(), distiller=None)
    turn = _turn_with_conclusion("经查询，今天世界杯有两场比赛，分别在晚上八点和十一点。")
    facts = c.consolidate_turns([turn])
    assert all(f.tags != "origin:conclusion" for f in facts)


def test_passthrough_conclusion_not_polluting():
    """透传蒸馏器的 distill_conclusion 返回空 → 长回答原文不入库。"""
    c = ProxyConsolidator(embedder=MockEmbedder(), distiller=PassthroughDistiller())
    turn = _turn_with_conclusion("这是一段很长的最终回答" * 20)
    facts = c.consolidate_turns([turn])
    assert all(f.tags != "origin:conclusion" for f in facts)


def test_short_conclusion_skipped():
    """太短的结论（"好的"类）没有信息量 → 跳过。"""
    distiller = FakeDistiller()
    c = ProxyConsolidator(embedder=MockEmbedder(), distiller=distiller)
    turn = _turn_with_conclusion("好的，已完成。")
    c.consolidate_turns([turn])
    assert distiller.conclusion_calls == []


# ── proxy：LLMDistiller.distill_conclusion（独立 prompt + 台账版本）──


def test_llm_distill_conclusion_uses_own_prompt_and_ledger_ver(monkeypatch):
    """distill_conclusion 走结论专用 system prompt + 独立台账版本。"""
    from bladex_proxy import router_sdk

    seen: dict = {}

    class FakeChoice:
        class message:
            content = '{"facts": [{"content": "已删除 HTML 第六部分", "kind": "task"}], "matter_proposals": []}'

    class FakeResp:
        choices = [FakeChoice()]

    def fake_completion(**kw):
        seen["system"] = kw["messages"][0]["content"]
        return FakeResp()

    monkeypatch.setattr(router_sdk, "completion", fake_completion)

    class MemLedger:
        def __init__(self):
            self.store: dict = {}

        def get_distill(self, source_text, model, prompt_ver):
            return self.store.get((source_text, model, prompt_ver))

        def put_distill(self, source_text, model, prompt_ver, output):
            self.store[(source_text, model, prompt_ver)] = output
            return "k"

    journal = MemLedger()
    d = LLMDistiller(model="m", journal=journal)
    out = d.distill_conclusion("已删除 HTML 第六部分并重排编号，第七部分改为第六部分。")

    assert out.facts[0].content == "已删除 HTML 第六部分"
    assert "FINAL ANSWER" in seen["system"]  # 结论专用 prompt
    # 台账 key 用 CONCLUSION_PROMPT_VER（与 user 蒸馏台账互不污染）
    assert any(k[2] == CONCLUSION_PROMPT_VER for k in journal.store)
    assert CONCLUSION_PROMPT_VER != PROMPT_VER

    # 二次调用命中台账（零 LLM）
    def boom(**kw):
        raise AssertionError("should hit journal")
    monkeypatch.setattr(router_sdk, "completion", boom)
    out2 = d.distill_conclusion("已删除 HTML 第六部分并重排编号，第七部分改为第六部分。")
    assert out2.facts[0].content == "已删除 HTML 第六部分"


# ── proxy：rebuild_from_hub 结论判定 ──


def test_rebuild_sets_conclusion_only_for_final_answer_turns():
    """status=ok + 有正文 + 无 tool call → 结论；中间往返/失败轮不蒸。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        journal = MemoryHub(Path(tmpdir) / "rocksdb")
        journal.open()
        identity = Identity(user_id="u1", agent_id="a1", session_id="s1")

        # 轮1：中间往返（response 带 tool call）
        journal.put(identity.storage_key("1000-0"), Turn(
            identity=identity, model="m",
            request_messages=[{"role": "user", "content": "今天有世界杯的比赛吗？后面的赛程是怎么安排的，帮我查一下"}],
            response_text="我来搜索一下",
            tool_events=[ToolEvent(tool_name="web_search", direction="call", tool_call_id="t1")],
            status=TurnStatus.OK,
        ))
        # 轮2：final answer（无 call，只有上一轮的 result 回填）
        journal.put(identity.storage_key("1001-0"), Turn(
            identity=identity, model="m",
            request_messages=[
                {"role": "user", "content": "今天有世界杯的比赛吗？后面的赛程是怎么安排的，帮我查一下"},
                {"role": "tool", "tool_call_id": "t1", "content": "search results..."},
            ],
            response_text="经查询，今天世界杯有两场比赛：A 队对 B 队（晚上八点）、C 队对 D 队（晚上十一点），均在多哈进行。",
            tool_events=[ToolEvent(tool_name="web_search", direction="result", tool_call_id="t1")],
            status=TurnStatus.OK,
        ))
        # 轮3：失败轮（有正文也不算结论）
        journal.put(identity.storage_key("1002-0"), Turn(
            identity=identity, model="m",
            request_messages=[{"role": "user", "content": "北京今天的天气到底怎么样？会不会下雨，要不要带伞出门"}],
            response_text="经查询，北京今天全天晴朗无雨，最高气温三十二度，空气质量良好，不需要带伞出门。",
            status=TurnStatus.FAILED,
        ))

        distiller = FakeDistiller()
        index = MemoryIndex(Path(tmpdir) / "index", embedder=MockEmbedder(),
                       read_only=False, distiller=distiller)
        index.open()
        index.rebuild_from_hub(journal, full=True)

        # 只有轮2 的 final answer 被送去结论蒸馏
        assert len(distiller.conclusion_calls) == 1
        assert "两场比赛" in distiller.conclusion_calls[0]

        conclusion_facts = [f for f in index.all_facts() if f.tags == "origin:conclusion"]
        assert len(conclusion_facts) == 1
        assert conclusion_facts[0].content == "查询结果：今天有两场世界杯比赛"

        index.close()
        journal.close()
