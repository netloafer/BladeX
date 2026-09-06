"""上下文装配验收剧本（ADR-0019 P1）。

覆盖：单元化 / L1 证据降解（结论即蒸馏）/ L2 单元摘要 / 确定性与工具循环前缀稳定 /
tool_call 配对合法性 / 2026-07-14 真实事故会话形状回放。
"""

from __future__ import annotations

from bladex_core.fact import Fact
from bladex_proxy.assembly import AssemblyConfig, ContextAssembler
from bladex_proxy.assembly import SUMMARY_OPEN, estimate_context_chars


def _user(text: str) -> dict:
    return {"role": "user", "content": text}


def _assistant(text: str) -> dict:
    return {"role": "assistant", "content": text}


def _assistant_tc(tc_id: str = "tc1", name: str = "web_search") -> dict:
    return {"role": "assistant", "content": "",
            "tool_calls": [{"id": tc_id, "type": "function",
                            "function": {"name": name, "arguments": "{}"}}]}


def _tool(tc_id: str = "tc1", chars: int = 10000, name: str = "web_search") -> dict:
    return {"role": "tool", "tool_call_id": tc_id, "name": name,
            "content": "X" * chars}


def _asm(**kw) -> ContextAssembler:
    defaults = dict(enabled=True, evidence_min_chars=500,
                    evidence_excerpt_chars=100, keep_recent_closed_units=0,
                    msg_threshold=30, budget_chars=60000, preserve_units=6)
    defaults.update(kw)
    return ContextAssembler(AssemblyConfig(**defaults))


def _closed_unit_session() -> list[dict]:
    """任务1（closed，带巨型证据）+ 任务2（open，带巨型证据）。"""
    return [
        {"role": "system", "content": "You are helpful."},
        _user("世界杯赛程？"),
        _assistant_tc("tc1"),
        _tool("tc1", 30000),
        _assistant("今天有两场比赛：A vs B、C vs D。"),  # 任务1结论
        _user("北京天气怎么样？"),
        _assistant_tc("tc2"),
        _tool("tc2", 10000),  # open 单元证据
    ]


# ── L1 证据降解 ──


def test_closed_unit_evidence_degraded_open_unit_intact():
    """closed 单元巨型 tool 结果降解；open 单元证据原样（模型答题要用）。"""
    msgs = _closed_unit_session()
    out, info = _asm().assemble(msgs, [])
    assert info["changed"] is True
    assert info["evidence_degraded"] == 1
    # closed 单元的 tool 被降解为占位摘录
    assert len(out[3]["content"]) < 1000
    assert "bladex-archived-evidence" in out[3]["content"]
    # open 单元的 tool 原样
    assert len(out[7]["content"]) == 10000


def test_conclusion_is_distillation():
    """结论即蒸馏：closed 单元的 user 与 final assistant 原样保留。"""
    msgs = _closed_unit_session()
    out, _ = _asm().assemble(msgs, [])
    assert out[1]["content"] == "世界杯赛程？"
    assert out[4]["content"] == "今天有两场比赛：A vs B、C vs D。"


def test_tool_pairing_intact():
    """降解只改 content：消息数/顺序/tool_call_id 不变（API 配对合法）。"""
    msgs = _closed_unit_session()
    out, _ = _asm().assemble(msgs, [])
    assert len(out) == len(msgs)
    assert [m.get("role") for m in out] == [m.get("role") for m in msgs]
    assert out[3]["tool_call_id"] == "tc1"


def test_small_evidence_untouched():
    """短 tool 结果（≤ evidence_min_chars）不降解。"""
    msgs = _closed_unit_session()
    msgs[3] = _tool("tc1", 300)
    out, info = _asm().assemble(msgs, [])
    assert info["evidence_degraded"] == 0
    assert out[3]["content"] == "X" * 300


def test_open_unit_recent_tools_kept():
    """ADR-0020 T2：open 单元保最近 K tool，更老降解；closed 单元 tool 全降解。"""
    msgs = [
        {"role": "system", "content": "sys"},
        _user("任务1"), _assistant_tc("t1"), _tool("t1", 20000), _assistant("结论1"),  # closed
        _user("任务2（open）"), _assistant_tc("t2"), _tool("t2", 20000),
        _assistant_tc("t3"), _tool("t3", 20000),
    ]
    out, info = _asm(keep_recent_tool_results=1).assemble(msgs, [])
    # closed t1 全降解；open 保最近 1（t3），t2 降解
    assert info["evidence_degraded"] == 2
    assert "bladex-archived-evidence" in out[3]["content"]  # t1（closed）降解
    assert "bladex-archived-evidence" in out[7]["content"]  # t2（open 更老）降解
    assert len(out[9]["content"]) == 20000                   # t3（open 最近）保全文


def test_assembly_disabled_noop():
    msgs = _closed_unit_session()
    out, info = _asm(enabled=False).assemble(msgs, [])
    assert out == msgs
    assert info["changed"] is False


def test_idempotent():
    """幂等：装配结果再装配，不再变化（占位符不二次降解）。"""
    asm = _asm()
    msgs = _closed_unit_session()
    once, info1 = asm.assemble(msgs, [])
    twice, info2 = asm.assemble(once, [])
    assert twice == once
    assert info2["evidence_degraded"] == 0


def test_tool_loop_prefix_stable():
    """工具循环内（同一 open 单元追加消息）→ 公共前缀逐字一致（保 prompt cache）。"""
    asm = _asm()
    base = _closed_unit_session()
    out1, _ = asm.assemble(base, [])
    # 同一 open 单元继续工具循环
    more = base + [_assistant_tc("tc3"), _tool("tc3", 8000)]
    out2, _ = asm.assemble(more, [])
    assert out2[: len(out1)] == out1


# ── L2 单元摘要（承接 CAP）──


def test_l2_count_trigger_drops_old_units():
    """消息数超阈值 → 最老 closed 单元换摘要，近期单元保留。"""
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(20):
        msgs.append(_user(f"question {i}"))
        msgs.append(_assistant(f"answer {i}"))
    out, info = _asm(msg_threshold=10, preserve_units=3).assemble(msgs, [])
    assert info["units_dropped"] > 0
    assert any(SUMMARY_OPEN in str(m.get("content", "")) for m in out)
    # 近期 3 单元保留
    for i in (17, 18, 19):
        assert any(f"question {i}" in str(m.get("content", "")) for m in out)
    # 最老单元被 drop
    assert not any("question 0" == str(m.get("content", "")) for m in out)


def test_l2_budget_trigger():
    """消息数不多但体积超预算 → L1 降解后若仍超 → L2 兜底。"""
    msgs = [{"role": "system", "content": "S" * 30000}]
    for i in range(8):
        msgs.append(_user(f"q{i}: " + "Y" * 5000))
        msgs.append(_assistant(f"a{i}: " + "Z" * 5000))
    out, info = _asm(budget_chars=50000, preserve_units=2, msg_threshold=30).assemble(msgs, [])
    assert info["units_dropped"] > 0
    assert estimate_context_chars(out) < estimate_context_chars(msgs)


def test_l2_respects_prefix_changed_guard():
    """§3.5 承接：Hermes 刚压缩过且规模不大（< 阈值×2）→ 不二次 L2；规模仍大 → 照压。"""
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(20):
        msgs.append(_user(f"q{i}"))
        msgs.append(_assistant(f"a{i}"))
    # 41 msgs > 10×2 → guard 不拦，照压
    out, info = _asm(msg_threshold=10, preserve_units=3).assemble(
        msgs, [], prefix_changed=True,
    )
    assert info["units_dropped"] > 0
    # 小会话（15 msgs < 6×2? 否，15>12）→ 用 msg_threshold=10：15 < 20 → guard 拦截
    small = msgs[:15]
    out2, info2 = _asm(msg_threshold=10, preserve_units=2).assemble(
        small, [], prefix_changed=True,
    )
    assert info2["units_dropped"] == 0


def test_l2_summary_contains_facts():
    """摘要含 Memory Index facts（非硬规则）+ dropped user 片段。"""
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(20):
        msgs.append(_user(f"unique question {i}"))
        msgs.append(_assistant(f"answer {i}"))
    facts = [Fact(id="f1", content="用户在做 BladeX 项目", category="general")]
    out, _ = _asm(msg_threshold=10, preserve_units=3).assemble(msgs, facts)
    summary = next(m for m in out if SUMMARY_OPEN in str(m.get("content", "")))
    assert "BladeX 项目" in summary["content"]
    assert "unique question 0" in summary["content"]


# ── 2026-07-14 真实事故会话回放（形状级） ──


def test_incident_replay_shape():
    """事故形状：26 条消息 / ~158K 字符 / 4 个 user 轮 / 证据散在近期轮次。

    v1 CAP：条数不触发（26<30）+ user 轮不够（4<6）→ 一个字不压 → 158K 直发。
    ADR-0019：L1 降解 closed 单元证据 → 体积塌到 ~1/4 以下，
    问天气时上下文不再携带世界杯搜索原文，但保留其结论。
    """
    msgs = [
        {"role": "system", "content": "H" * 23545},
        # 任务1：世界杯（closed）
        _user("今天有世界杯的比赛吗？后面是什么赛程"),
        _assistant_tc("t1"), _tool("t1", 2553),
        _assistant_tc("t2"), _tool("t2", 32663),
        _assistant_tc("t3"), _tool("t3", 23880),
        _assistant("世界杯赛程结论：今天两场，明天四场。"),
        # 任务2：HTML 编辑（closed）
        _user("html里面的第六部分不需要，删除掉吧"),
        _assistant_tc("t4"), _tool("t4", 11160),
        _assistant_tc("t5"), _tool("t5", 12533),
        _assistant_tc("t6"), _tool("t6", 10995),
        _assistant("已删除第六部分并重排编号。"),
        # 任务3：天气（open）
        _user("北京今天天气怎么样？"),
        _assistant_tc("t7"), _tool("t7", 10226),
    ]
    before = estimate_context_chars(msgs)
    assert before > 120000

    out, info = _asm(keep_recent_closed_units=0).assemble(msgs, [])
    after = estimate_context_chars(out)

    assert info["evidence_degraded"] == 6          # 两个 closed 单元共 6 条证据
    assert after < before * 0.30                    # 体积塌到 30% 以下
    # 世界杯搜索原文不再出现，但结论保留
    assert not any(
        isinstance(m.get("content"), str) and len(m["content"]) > 20000 and m.get("role") == "tool"
        and m is not out[-1]
        for m in out[:-2]
    )
    assert any("世界杯赛程结论" in str(m.get("content", "")) for m in out)
    # 当前任务（天气）的工具证据原样保留
    assert len(out[-1]["content"]) == 10226
    # 配对合法
    assert len(out) == len(msgs)
