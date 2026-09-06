"""三段式 T2（2026-08-10）：topic 的 schema 与落库路径（core 侧）。

钉四件事：
  - Fact.topic / DistillFact.topic / DistillOutput.topic+keywords 字段存在、
    默认空（历史数据反序列化零回归）；
  - 候选装配继承：轮级 topic 写进该轮**每条** candidate（一轮一 topic）；
  - candidate → Fact：topic 落到 Fact.topic；
  - topic 不参与 fact id（拍板 P1：不当主键、不进 identity 判定）。

FTS topic 列与迁移在 proxy 侧 test_fts_topic_migration.py。
"""

from __future__ import annotations

from bladex_core.consolidation_proxy import ProxyConsolidator, _deterministic_fact_id
from bladex_core.distillation import (
    DistillFact,
    DistillOutput,
    DistillTurnInput,
    PassthroughDistiller,
)
from bladex_core.fact import ConversationTurn, Fact


class _TurnDistiller:
    """v4/v5 形态的测试替身：distill_turn 返回带 topic 的三段式产出。"""

    model_name = "test-model"
    prompt_ver = "v5-three-seg-001"

    def distill_turn(self, payload: DistillTurnInput) -> DistillOutput:
        topic = "泰山啤酒生产线投资评估"
        facts = [
            DistillFact(content="新建10万吨生产线总投资约4.5亿元", kind="general",
                        item_kind="assertion", subject="生产线", attribute="投资额",
                        topic=topic, provenance="user_direct"),
            DistillFact(content="投资估算与佛山三水工厂2021年投资额吻合", kind="general",
                        item_kind="assertion", subject="投资估算", attribute="口径",
                        topic=topic, provenance="user_direct"),
        ]
        return DistillOutput(facts=facts, matter_proposals=[], model_name="test-model",
                             topic=topic, keywords=["泰山啤酒", "生产线"])

    def distill(self, text: str) -> DistillOutput:  # v3 兜底不应被走到
        raise AssertionError("v4 路存在时不该回落 v3")


def test_schema_defaults_backward_compatible():
    """历史数据（无 topic 键）反序列化 → 空默认，零回归。"""
    assert Fact(id="f", content="c").topic == ""
    assert Fact.model_validate({"id": "f", "content": "c"}).topic == ""
    assert DistillFact(content="c").topic == ""
    out = DistillOutput.model_validate({"facts": [], "matter_proposals": []})
    assert out.topic == "" and out.keywords == []


def test_candidates_inherit_turn_topic():
    """轮级 topic 继承式写进该轮每条 candidate（一轮一 topic）。"""
    turn = ConversationTurn(
        session_id="s", user_id="u", ledger_key="k1",
        user_messages=["帮我评估新建泰山啤酒生产线的总投资"],
    )
    cands = ProxyConsolidator(distiller=_TurnDistiller())._collect_candidates([turn])
    assert len(cands) == 2
    assert all(c["topic"] == "泰山啤酒生产线投资评估" for c in cands)


def test_candidate_topic_lands_on_fact():
    """candidate → Fact 构造：topic 落 Fact.topic（meta 层字段）。"""

    class _Embedder:
        def embed(self, texts):
            return [[1.0, 0.0] for _ in texts]

    turn = ConversationTurn(
        session_id="s", user_id="u", ledger_key="k1",
        user_messages=["帮我评估新建泰山啤酒生产线的总投资"],
    )
    consolidator = ProxyConsolidator(distiller=_TurnDistiller(), embedder=_Embedder())
    facts = consolidator.consolidate_turns([turn], existing_facts=[])
    assert facts and all(f.topic == "泰山啤酒生产线投资评估" for f in facts)


def test_topic_not_part_of_fact_id():
    """拍板 P1：topic 不参与 fact id（同 ledger_key+content 恒同 id，与 topic 无关）。"""
    fid = _deterministic_fact_id("k1", "同一条内容")
    f1 = Fact(id=fid, content="同一条内容", topic="主题A")
    f2 = Fact(id=fid, content="同一条内容", topic="主题B")
    assert f1.id == f2.id == _deterministic_fact_id("k1", "同一条内容")


def test_passthrough_turn_has_no_topic():
    """透传模式（关掉 LLM）不产 topic——v4/v5 与 v3 产出仍逐字一致的锚点。"""
    out = PassthroughDistiller().distill_turn(DistillTurnInput(user_text="原文"))
    assert out.topic == "" and out.keywords == []
    assert out.facts[0].topic == ""
