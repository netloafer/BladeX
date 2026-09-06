"""蒸馏 + consolidation 集成测试（ADR-0014 L2 + ADR-0018 §3.1 结构化输出）。"""

from bladex_core.consolidation_proxy import ProxyConsolidator
from bladex_core.distillation import DistillFact, DistillOutput, PassthroughDistiller
from bladex_core.fact import ConversationTurn


class MockEmbedder:
    """固定向量 mock - 用简单 hash 映射。"""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            h = hash(text)
            vec = [((h >> i) & 1) * 1.0 for i in range(self._dim)]
            results.append(vec)
        return results


class MockDistiller:
    """mock 蒸馏器 - 按规则模拟 LLM 抽取，返回 DistillOutput（ADR-0018 §3.1）。"""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def model_name(self) -> str:
        return "mock-distiller-v1"

    @property
    def prompt_ver(self) -> str:
        return "mock-v1"

    def distill(self, user_message: str) -> DistillOutput:
        self.calls += 1
        msg = user_message.strip()

        # 模拟系统/测试消息 -> 空 facts
        if msg.startswith("Reply with exactly") or msg.startswith("Fully describe"):
            return DistillOutput(model_name="mock-distiller-v1")
        if "test" in msg.lower() and len(msg) < 50:
            return DistillOutput(model_name="mock-distiller-v1")

        facts: list[DistillFact] = []
        if "世界杯" in msg and "赛程" in msg:
            facts.append(DistillFact(content="用户询问世界杯赛程安排", kind="event"))
        if "世界杯" in msg and ("预测" in msg or "四强" in msg):
            facts.append(DistillFact(content="用户询问世界杯四强预测", kind="event"))
        if "阿根廷" in msg and ("比赛" in msg or "赛" in msg):
            facts.append(DistillFact(content="用户询问阿根廷比赛结果", kind="event"))
        if "破产" in msg:
            facts.append(DistillFact(content="用户请求分析破产案件", kind="task"))

        if not facts:
            facts.append(DistillFact(content=msg, kind="general"))
        return DistillOutput(facts=facts, model_name="mock-distiller-v1")


def test_distiller_extracts_atomic_facts():
    """蒸馏器把原始消息转成原子事实（结构化 DistillOutput）。"""
    distiller = MockDistiller()
    consolidator = ProxyConsolidator(
        embedder=MockEmbedder(), distiller=distiller,
    )

    turns = [
        ConversationTurn(
            session_id="s1",
            user_messages=["今天世界杯赛程怎么样？帮我预测一下哪四支球队可以进入四强。"],
            ledger_key="k1",
        ),
    ]

    facts = consolidator.consolidate_turns(turns)
    assert len(facts) == 2
    assert any("赛程" in f.content for f in facts)
    assert any("预测" in f.content for f in facts)
    assert all(f.distill_model == "mock-distiller-v1" for f in facts)
    assert all(f.source_text == "今天世界杯赛程怎么样？帮我预测一下哪四支球队可以进入四强。" for f in facts)
    # ADR-0018 §3.1: kind 入 Fact
    assert all(f.kind == "event" for f in facts)


def test_distiller_filters_system_messages():
    """系统/测试消息蒸馏返回空 facts -> 不产出 Fact。"""
    distiller = MockDistiller()
    consolidator = ProxyConsolidator(
        embedder=MockEmbedder(), distiller=distiller,
    )

    turns = [
        ConversationTurn(
            session_id="s1",
            user_messages=['Reply with exactly: "Round 8 subagent test OK - delegate_task working normally."'],
            ledger_key="k1",
        ),
        ConversationTurn(
            session_id="s1",
            user_messages=["今天世界杯赛程怎么样？帮我分析一下各队的情况。"],
            ledger_key="k2",
        ),
    ]

    facts = consolidator.consolidate_turns(turns)
    assert len(facts) == 1
    assert "赛程" in facts[0].content
    assert facts[0].distill_model == "mock-distiller-v1"


def test_distiller_records_model_name():
    """Fact 记录蒸馏模型名（审计/对比用）。"""
    distiller = MockDistiller()
    consolidator = ProxyConsolidator(
        embedder=MockEmbedder(), distiller=distiller,
    )

    turns = [
        ConversationTurn(
            session_id="s1",
            user_messages=["帮我分析一下阿根廷的比赛结果，给我详细数据。"],
            ledger_key="k1",
        ),
    ]

    facts = consolidator.consolidate_turns(turns)
    assert len(facts) == 1
    assert facts[0].distill_model == "mock-distiller-v1"


def test_no_distiller_falls_back_to_passthrough():
    """无 distiller 时透传原文（向后兼容）。"""
    consolidator = ProxyConsolidator(embedder=MockEmbedder())

    turns = [
        ConversationTurn(
            session_id="s1",
            user_messages=["I am a software engineer working on AI memory systems."],
            ledger_key="k1",
        ),
    ]

    facts = consolidator.consolidate_turns(turns)
    assert len(facts) == 1
    assert facts[0].content == "I am a software engineer working on AI memory systems."
    assert facts[0].distill_model == ""
    assert facts[0].kind == "general"


def test_passthrough_distiller():
    """PassthroughDistiller 透传原文，model_name = passthrough。"""
    d = PassthroughDistiller()
    assert d.model_name == "passthrough"
    out = d.distill("hello")
    assert len(out.facts) == 1
    assert out.facts[0].content == "hello"
    assert out.model_name == "passthrough"
