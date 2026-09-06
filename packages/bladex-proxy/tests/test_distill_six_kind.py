"""U4 Part B / B1：蒸馏六类 + subject/attribute + 准入（ADR-0026 §4.2 / 铁律3）。

真跑 _parse_distill_json（纯函数）：v3 六类解析、非法值兜底、旧 v2 台账兼容、junk 拦截。
"""

from __future__ import annotations

from bladex_proxy.distillation import (
    CONCLUSION_PROMPT_VER,
    PROMPT_VER,
    _VALID_ITEM_KINDS,
    _parse_distill_json,
)


def test_prompt_versions_bumped_to_items() -> None:
    # prompt 改了六类 → 版本号必须递增（否则复用旧台账、混淆 schema）
    assert PROMPT_VER == "v3-items-001"
    assert "conclusion" in CONCLUSION_PROMPT_VER and CONCLUSION_PROMPT_VER != "v1-conclusion-001"
    assert _VALID_ITEM_KINDS == {
        "assertion", "preference", "procedure", "lesson", "file_ref", "profile_obs",
    }


def test_parse_v3_six_kind_with_slots() -> None:
    raw = """{"facts":[
      {"content":"用户希望回复简洁","item_kind":"preference","subject":"回复","attribute":"风格","entities":[]},
      {"content":"ANN 测试需≥256行才触发建索引","item_kind":"lesson","subject":"ANN测试","attribute":"lesson","entities":["ANN"]},
      {"content":"routing.toml 钉了 e5-large","item_kind":"file_ref","subject":"routing.toml","attribute":"file","entities":["routing.toml"]}
    ],"matter_proposals":[{"title":"记忆检索重构","entities":["检索"]}]}"""
    out = _parse_distill_json(raw, "m")
    assert out is not None and len(out.facts) == 3
    pref, lesson, fref = out.facts
    assert pref.item_kind == "preference" and pref.subject == "回复" and pref.attribute == "风格"
    assert lesson.item_kind == "lesson" and lesson.subject == "ANN测试"
    assert fref.item_kind == "file_ref"
    assert out.matter_proposals[0].title == "记忆检索重构"


def test_invalid_item_kind_falls_to_empty() -> None:
    """非法 item_kind 留空 → 交给 consolidation._resolve_item_kind 兜底，不硬塞。"""
    out = _parse_distill_json('{"facts":[{"content":"c","item_kind":"garbage"}],"matter_proposals":[]}', "m")
    assert out is not None and out.facts[0].item_kind == ""


def test_v2_ledger_backward_compatible() -> None:
    """旧 v2 台账数据（kind=event/junk、无 item_kind）仍可解析；junk 构造上拦截。"""
    out = _parse_distill_json(
        '{"facts":[{"content":"keep","kind":"event"},{"content":"drop","kind":"junk"}],"matter_proposals":[]}',
        "m",
    )
    assert out is not None and len(out.facts) == 1
    assert out.facts[0].content == "keep" and out.facts[0].item_kind == ""


def test_missing_slots_default_empty() -> None:
    out = _parse_distill_json(
        '{"facts":[{"content":"用户是山东人","item_kind":"assertion"}],"matter_proposals":[]}', "m"
    )
    assert out is not None and out.facts[0].item_kind == "assertion"
    assert out.facts[0].subject == "" and out.facts[0].attribute == ""
