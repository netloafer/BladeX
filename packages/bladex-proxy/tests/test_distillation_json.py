"""LLMDistiller JSON 解析单元测试（T6.1 验收①，ADR-0018 §3.1）。

不调真 LLM，直接测 _parse_distill_json + junk 拦截 + 失败弃置。
"""

from bladex_proxy.distillation import PROMPT_VER, _parse_distill_json


def test_parse_valid_json():
    """合法 JSON -> DistillOutput(facts + proposals)。"""
    raw = ('{"facts": [{"content": "用户喜欢 e5", "kind": "preference", "entities": ["e5"]}], '
           '"matter_proposals": [{"title": "嵌入模型选型", "entities": ["e5"]}]}')
    out = _parse_distill_json(raw, "m1")
    assert len(out.facts) == 1
    assert out.facts[0].content == "用户喜欢 e5"
    assert out.facts[0].kind == "preference"
    assert out.facts[0].entities == ["e5"]
    assert len(out.matter_proposals) == 1
    assert out.matter_proposals[0].title == "嵌入模型选型"
    assert out.model_name == "m1"


def test_parse_junk_filtered():
    """kind=junk 不入库（构造上拦截，ADR-0018 §3.1 修 R4）。"""
    raw = ('{"facts": [{"content": "垃圾", "kind": "junk"}, '
           '{"content": "用户喜欢 e5", "kind": "preference"}], "matter_proposals": []}')
    out = _parse_distill_json(raw, "m1")
    assert len(out.facts) == 1
    assert out.facts[0].content == "用户喜欢 e5"


def test_parse_markdown_wrapped():
    """LLM 裹 ```json``` 也能解析。"""
    raw = '```json\n{"facts": [{"content": "x", "kind": "event"}], "matter_proposals": []}\n```'
    out = _parse_distill_json(raw, "m1")
    assert len(out.facts) == 1
    assert out.facts[0].content == "x"


def test_parse_invalid_json_returns_none():
    """解析失败（无合法 JSON）-> None（调用方不缓存，rebuild 可二次蒸馏）。"""
    out = _parse_distill_json("not json at all", "m1")
    assert out is None


def test_parse_empty_facts():
    """空 facts（系统消息/agent 模板）-> 空 DistillOutput。"""
    raw = '{"facts": [], "matter_proposals": []}'
    out = _parse_distill_json(raw, "m1")
    assert out.facts == []
    assert out.matter_proposals == []


def test_parse_invalid_kind_falls_to_general():
    """非法 kind -> general（不丢弃 fact）。"""
    raw = '{"facts": [{"content": "x", "kind": "unknown_kind"}], "matter_proposals": []}'
    out = _parse_distill_json(raw, "m1")
    assert len(out.facts) == 1
    assert out.facts[0].kind == "general"


def test_parse_proposals_capped_at_3():
    """matter_proposals 封顶 3 个（ADR-0018 §3.1）。"""
    raw = '{"facts": [], "matter_proposals": [{"title":"a"},{"title":"b"},{"title":"c"},{"title":"d"}]}'
    out = _parse_distill_json(raw, "m1")
    assert len(out.matter_proposals) == 3


def test_parse_skips_empty_content():
    """空 content 的 fact 跳过。"""
    raw = ('{"facts": [{"content": "", "kind": "event"}, '
           '{"content": "valid", "kind": "event"}], "matter_proposals": []}')
    out = _parse_distill_json(raw, "m1")
    assert len(out.facts) == 1
    assert out.facts[0].content == "valid"


def test_parse_extracts_json_from_prose():
    """LLM 在 JSON 前后带 prose 也能提取（找首 { 末 }）。"""
    raw = 'Here is the result:\n{"facts": [{"content": "x", "kind": "event"}], "matter_proposals": []}\nDone.'
    out = _parse_distill_json(raw, "m1")
    assert len(out.facts) == 1
    assert out.facts[0].content == "x"


def test_prompt_ver_constant():
    """PROMPT_VER 是非空常量（台账 key 用，prompt 改递增）。"""
    assert isinstance(PROMPT_VER, str)
    assert PROMPT_VER
