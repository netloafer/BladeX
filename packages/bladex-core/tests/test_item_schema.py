"""U4：条目六类 schema + 三轴/双时态字段（ADR-0026 §4.2）+ item_kind 解析。

加法式演进：新字段带默认、旧数据可反序列化（零迁移）。真跑 Fact/DistillFact/解析器。
"""

from __future__ import annotations

from bladex_core.consolidation_proxy import _resolve_item_kind
from bladex_core.distillation import DistillFact
from bladex_core.fact import Fact, ItemKind


def test_item_kind_six_closed_values() -> None:
    assert [k.value for k in ItemKind] == [
        "assertion", "preference", "procedure", "lesson", "file_ref", "profile_obs",
    ]


def test_fact_new_fields_defaults() -> None:
    f = Fact(content="泰山啤酒第二次招募针对重整投资人")
    assert f.item_kind == ItemKind.ASSERTION
    assert f.subject == "" and f.attribute == "" and f.audience == "all"
    assert f.matter_id == "" and f.unit_key == "" and f.superseded_by == ""
    assert f.t_invalid is None
    assert f.importance == 1.0 and f.strength == 1


def test_fact_backward_compatible_old_data_loads() -> None:
    """旧 Memory Index Fact（无 U4 新字段）经 model_validate 仍可读，新字段走默认。"""
    f = Fact(content="x")
    old = f.model_dump(mode="json")
    for k in ("item_kind", "subject", "attribute", "audience", "matter_id",
              "t_observed", "t_valid", "t_invalid", "superseded_by",
              "importance", "strength", "unit_key", "agent_id"):
        old.pop(k, None)
    f2 = Fact.model_validate(old)
    assert f2.item_kind == ItemKind.ASSERTION and f2.importance == 1.0 and f2.strength == 1


def test_distillfact_typed_slots() -> None:
    df = DistillFact(content="回复要简洁", item_kind="preference", subject="回复", attribute="风格")
    assert df.item_kind == "preference" and df.subject == "回复" and df.attribute == "风格"
    # 旧 DistillFact（无新槽位）仍可构造
    df2 = DistillFact(content="c", kind="event")
    assert df2.item_kind == "" and df2.subject == ""


def test_resolve_item_kind_explicit_then_legacy_then_fallback() -> None:
    # 蒸馏器显式 item_kind 优先
    assert _resolve_item_kind({"item_kind": "lesson"}) == ItemKind.LESSON
    assert _resolve_item_kind({"item_kind": "file_ref"}) == ItemKind.FILE_REF
    # 空 item_kind → 按旧 kind 兜底
    assert _resolve_item_kind({"item_kind": "", "kind": "preference"}) == ItemKind.PREFERENCE
    assert _resolve_item_kind({"kind": "decision"}) == ItemKind.ASSERTION
    # 非法值 / 全空 → assertion
    assert _resolve_item_kind({"item_kind": "garbage", "kind": "task"}) == ItemKind.ASSERTION
    assert _resolve_item_kind({}) == ItemKind.ASSERTION
