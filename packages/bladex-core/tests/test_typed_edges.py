"""U5.1：类型化边（ADR-0024 §4.6）——EdgeTargetType 扩 MATTER/UNIT + EdgeRelation。

存量边默认 BELONGS（现有语义），无需迁移。真跑 schema。
"""

from __future__ import annotations

from bladex_core.matter import EdgeRelation, EdgeTargetType, MatterEdge


def test_edge_target_type_extended() -> None:
    assert [t.value for t in EdgeTargetType] == ["session", "fact", "matter"]   # unit 已删（F0.3 H1）


def test_edge_relation_values() -> None:
    assert [r.value for r in EdgeRelation] == ["belongs", "part_of"]   # 三个孤儿取值已删（F0.3 H1）


def test_edge_default_relation_belongs() -> None:
    e = MatterEdge(matter_id="m1", target_type=EdgeTargetType.FACT, target_key="f1")
    assert e.relation == EdgeRelation.BELONGS


def test_old_edge_deserializes_to_belongs() -> None:
    """存量边（无 relation 字段）反序列化默认 BELONGS（无需迁移）。"""
    e = MatterEdge(matter_id="m1", target_type=EdgeTargetType.FACT, target_key="f1")
    d = e.model_dump(mode="json")
    d.pop("relation")
    e2 = MatterEdge.model_validate(d)
    assert e2.relation == EdgeRelation.BELONGS


def test_typed_relation_edge() -> None:
    # Matter↔Matter 关系边（Fact→Fact 的 SUPERSEDES 取值已删：取代经 `Fact.superseded_by` 落地）
    e2 = MatterEdge(
        matter_id="m1", target_type=EdgeTargetType.MATTER, target_key="m2",
        relation=EdgeRelation.PART_OF,
    )
    assert e2.target_type == EdgeTargetType.MATTER and e2.relation == EdgeRelation.PART_OF
