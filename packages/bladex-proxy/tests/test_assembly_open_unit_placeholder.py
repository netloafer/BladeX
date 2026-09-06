"""V-C2a 验收剧本：open 单元占位符文案（MQ-CA2）+ 三处埋点（MQ-CA3）。

背景（2026-08-28）：Jason 报「Claude Code 一直看到输出被归档截断，可上下文根本没超」。
排查排除 `max_tokens`（proxy 纯透传、日志零出现）与上游长度截断，
定位到常开的 L1 证据降解——而它的占位符对 **open 单元**说了假话：
"该任务已完成，结论见其后的 assistant 回复"，两个断言都不成立。

🔴 本文件的重心是**阴性对照**：closed 路径必须逐字不变。
黄金串硬编码在下面——不是为了测"字符串等于自己"，而是因为
**closed 占位符的字节就是上游 prompt cache 的稳定面**：
它一旦变，历史前缀整体失效。任何未来改动都该在这里先绊一跤，
然后由改动者明确回答"这次 cache 断裂值不值"。
"""

from __future__ import annotations

from structlog.testing import capture_logs

from bladex_proxy.anthropic import _cache_read_tokens
from bladex_proxy.assembly import AssemblyConfig, ContextAssembler, _placeholder

# ── 改动前（2026-08-28 之前）closed 路径的逐字输出，作为回归黄金值 ──
_GOLDEN_CLOSED_EXCERPT = (
    "[bladex-archived-evidence tool=Bash chars=1000]\n"
    + "X" * 200
    + "\n…(该任务已完成，结论见其后的 assistant 回复；证据全文已存 BladeX 记忆)"
)
_GOLDEN_CLOSED_MINIMAL = (
    "[bladex-archived-evidence tool=Bash chars=1000 unrelated]\n"
    "…(与当前任务无关；证据全文已存 BladeX 记忆)"
)


def _user(text: str) -> dict:
    return {"role": "user", "content": text}


def _assistant_tc(tc_id: str, name: str = "Bash") -> dict:
    return {"role": "assistant", "content": "",
            "tool_calls": [{"id": tc_id, "type": "function",
                            "function": {"name": name, "arguments": "{}"}}]}


def _tool(tc_id: str, chars: int = 10000, name: str = "Bash") -> dict:
    return {"role": "tool", "tool_call_id": tc_id, "name": name,
            "content": "X" * chars}


# ══════════════════════════════════════════════════════════════
# B1 阴性对照：closed 路径逐字不变
# ══════════════════════════════════════════════════════════════

def test_closed_placeholder_is_byte_identical_to_pre_change():
    """closed 摘录档：与改动前逐字一致。破了 = 上游 cache 全断。"""
    assert _placeholder("X" * 1000, "Bash", 200) == _GOLDEN_CLOSED_EXCERPT


def test_closed_minimal_placeholder_is_byte_identical_to_pre_change():
    """closed minimal 档（U9 无关档）：同样逐字一致。"""
    assert _placeholder("X" * 1000, "Bash", 0) == _GOLDEN_CLOSED_MINIMAL


def test_open_unit_default_is_false_so_old_callers_unchanged():
    """`open_unit` 有默认值 —— 任何没跟上的旧调用方仍走 closed 文案。

    这条守的是"新参数不得改变未显式传参者的行为"。
    """
    assert _placeholder("X" * 1000, "Bash", 200) == _placeholder(
        "X" * 1000, "Bash", 200, open_unit=False)


# ══════════════════════════════════════════════════════════════
# B2 阳性：open 单元不再声称任务已完成
# ══════════════════════════════════════════════════════════════

def test_open_unit_placeholder_does_not_claim_task_finished():
    out = _placeholder("X" * 1000, "Bash", 200, open_unit=True)
    assert "该任务已完成" not in out, "open 单元任务并未完成——这是 MQ-CA2 的整条要害"
    assert "结论见其后的 assistant 回复" not in out, \
        "open 单元其后没有 assistant 回复，说了就是假陈述"
    assert "重新调用工具" in out, "必须告诉模型证据可以重新取，否则它既不重取也不重做"


def test_open_unit_minimal_placeholder_also_honest():
    out = _placeholder("X" * 1000, "Bash", 0, open_unit=True)
    assert "该任务已完成" not in out
    assert "重新调用工具" in out


def test_open_and_closed_placeholders_differ():
    """两档必须真的不同 —— 防止有人把 open 分支写成 closed 的复制粘贴。"""
    assert (_placeholder("X" * 1000, "Bash", 200, open_unit=True)
            != _placeholder("X" * 1000, "Bash", 200, open_unit=False))


def test_placeholder_stays_a_pure_function():
    """同输入同输出 —— 工具循环内前缀逐字稳定的前提（ADR-0019）。"""
    for open_unit in (True, False):
        a = _placeholder("Y" * 5000, "Read", 200, open_unit=open_unit)
        b = _placeholder("Y" * 5000, "Read", 200, open_unit=open_unit)
        assert a == b


# ══════════════════════════════════════════════════════════════
# 端到端：closed 与 open 单元在同一次装配里各拿各的文案
# ══════════════════════════════════════════════════════════════

def _assembler() -> ContextAssembler:
    # keep_recent_tool_results=0：让 open 单元的 tool 也全部降解，
    # 否则保护窗口会把 open 那条留成全文，这条剧本就测不到东西了。
    return ContextAssembler(AssemblyConfig(
        enabled=True, evidence_min_chars=500, evidence_excerpt_chars=200,
        keep_recent_closed_units=0, keep_recent_tool_results=0,
        msg_threshold=10_000, budget_chars=0,   # 关掉 L2，只看 L1
    ))


def test_closed_and_open_units_get_different_placeholders_end_to_end():
    messages = [
        _user("第一件事"),
        _assistant_tc("tc1"), _tool("tc1"),
        {"role": "assistant", "content": "第一件事的结论"},
        _user("第二件事"),                      # ← 这条 user 让上一单元 closed
        _assistant_tc("tc2"), _tool("tc2"),
    ]
    out, info = _assembler().assemble(messages, [])

    assert info["evidence_degraded"] == 2, "两条 tool 结果都该被降解"
    closed_tool = out[2]["content"]
    open_tool = out[6]["content"]

    assert "该任务已完成" in closed_tool, "closed 单元照旧走原文案"
    assert "该任务已完成" not in open_tool, "open 单元不得声称已完成"
    assert "重新调用工具" in open_tool


def test_closed_unit_content_unchanged_end_to_end():
    """整条 closed 消息的 content 与改动前逐字一致（阴性对照的端到端版）。"""
    messages = [
        _user("第一件事"),
        _assistant_tc("tc1"), _tool("tc1", chars=1000),
        {"role": "assistant", "content": "结论"},
        _user("第二件事"),
    ]
    out, _ = _assembler().assemble(messages, [])
    assert out[2]["content"] == _GOLDEN_CLOSED_EXCERPT


# ══════════════════════════════════════════════════════════════
# B6 埋点：assembly_done 的归属维度（MQ-CA3）
# ══════════════════════════════════════════════════════════════

def test_assembly_done_carries_session_and_agent():
    """没有归属维度，攒下来的 chars_before 无法按会话切分（V-C2b 的硬前置）。"""
    messages = [
        _user("干活"), _assistant_tc("tc1"), _tool("tc1"),
        {"role": "assistant", "content": "结论"}, _user("再干"),
    ]
    with capture_logs() as logs:
        _assembler().assemble(
            messages, [], session_id="sess-abc", agent_id="claude-code")

    done = [e for e in logs if e.get("event") == "assembly_done"]
    assert done, "装配确实改了消息，assembly_done 必须打出来"
    assert done[0]["session_id"] == "sess-abc"
    assert done[0]["agent_id"] == "claude-code"


def test_assembly_done_agent_id_defaults_empty_not_missing():
    """旧调用方不传 agent_id 时，字段仍在（缺字段 vs 空值，下游解析口径不同）。"""
    messages = [
        _user("干活"), _assistant_tc("tc1"), _tool("tc1"),
        {"role": "assistant", "content": "结论"}, _user("再干"),
    ]
    with capture_logs() as logs:
        _assembler().assemble(messages, [])
    done = [e for e in logs if e.get("event") == "assembly_done"]
    assert done and "agent_id" in done[0] and done[0]["agent_id"] == ""


# ══════════════════════════════════════════════════════════════
# B6 埋点：上游 cache 读数，None ≠ 0（MQ-CA3）
# ══════════════════════════════════════════════════════════════

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
    """🔴 None = 上游不报；0 = 报了且没命中。

    混成 0 会让"这个上游不支持 cache"伪装成"cache 一直没命中"，
    而 V-C2b 正是要靠这个字段判断 L1 常开有没有在保 cache。
    """
    assert _cache_read_tokens(_Usage(prompt_tokens=100)) is None
    assert _cache_read_tokens({}) is None
    assert _cache_read_tokens(None) is None


def test_cache_read_tokens_zero_is_preserved_as_zero():
    assert _cache_read_tokens(_Usage(cache_read_input_tokens=0)) == 0
    assert _cache_read_tokens(_Usage(prompt_tokens_details=_Details(0))) == 0
