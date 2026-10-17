"""装配读数面（MQ-CA3）：`assembly_done` 的归属维度 + 上游 cache 读数 None ≠ 0。

原在 `test_assembly_open_unit_placeholder.py`（V-C2a）；该文件的 open 单元占位符文案（MQ-CA2）
随 L1 按位置降解于批 U · U1.4 删除（ADR-0033：open 单元永不裁，占位改为可见标记，见
`test_u14_assembly_trim.py`），这里只留仍然成立的两组读数用例。
"""

from __future__ import annotations

from bladex_proxy.anthropic import _cache_read_tokens
from bladex_proxy.assembly import AssemblyConfig, ContextAssembler
from bladex_proxy.inject import _apply_cap
from structlog.testing import capture_logs


def _msgs() -> list[dict]:
    return [{"role": "user", "content": "干活"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "tc1", "type": "function", "function": {"name": "Bash", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "tc1", "name": "Bash", "content": "X" * 1000},
            {"role": "assistant", "content": "结论"},
            {"role": "user", "content": "再干"}]


# ── assembly_done 的归属维度（MQ-CA3）──


def test_assembly_done_carries_session_and_agent(monkeypatch):
    """没有归属维度，攒下来的 chars_before 无法按会话切分。ADR-0033 起开了装配的 agent 每轮都打。"""
    monkeypatch.setenv("BLADEX_ASSEMBLY_AGENTS", "claude-code")
    with capture_logs() as logs:
        _apply_cap(_msgs(), ContextAssembler(AssemblyConfig()), [], False, "sess-abc",
                   agent_id="claude-code")
    done = [e for e in logs if e.get("event") == "assembly_done"]
    assert done and done[0]["session_id"] == "sess-abc" and done[0]["agent_id"] == "claude-code"
    for k in ("triggered", "reviewed", "trimmed", "turns", "cold_reason"):
        assert k in done[0], f"assembly_done 缺 {k}（ADR-0033 判据②④⑤ 的读数面）"


def test_assembly_done_agent_id_defaults_empty_not_missing():
    """调用方不传 agent_id 时字段仍在（缺字段 vs 空值，下游解析口径不同）。"""
    with capture_logs() as logs:
        _apply_cap(_msgs(), ContextAssembler(AssemblyConfig()), [], False)
    done = [e for e in logs if e.get("event") == "assembly_done"]
    assert done and "agent_id" in done[0] and done[0]["agent_id"] == ""


# ── 上游 cache 读数，None ≠ 0（MQ-CA3）──


class _Details:
    def __init__(self, cached: int) -> None:
        self.cached_tokens = cached


class _Usage:
    def __init__(self, **kw: object) -> None:
        for k, v in kw.items():
            setattr(self, k, v)


def test_cache_read_tokens_reads_anthropic_style():
    assert _cache_read_tokens(_Usage(cache_read_input_tokens=1234)) == 1234


def test_cache_read_tokens_reads_openai_style():
    assert _cache_read_tokens(_Usage(prompt_tokens_details=_Details(777))) == 777


def test_cache_read_tokens_reads_dict_usage():
    assert _cache_read_tokens({"cache_read_input_tokens": 42}) == 42
    assert _cache_read_tokens({"prompt_tokens_details": {"cached_tokens": 9}}) == 9


def test_cache_read_tokens_absent_is_none_not_zero():
    """🔴 None = 上游不报；0 = 报了且没命中。ADR-0033 触发门按"无读数"判冷，两者不可混。"""
    assert _cache_read_tokens(_Usage(prompt_tokens=100)) is None
    assert _cache_read_tokens({}) is None
    assert _cache_read_tokens(None) is None


def test_cache_read_tokens_zero_is_preserved_as_zero():
    assert _cache_read_tokens(_Usage(cache_read_input_tokens=0)) == 0
    assert _cache_read_tokens(_Usage(prompt_tokens_details=_Details(0))) == 0
