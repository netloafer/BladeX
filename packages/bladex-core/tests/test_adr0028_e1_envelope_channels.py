"""ADR-0028 E1.4（F5 止血）：结论/进展通道在长度判定前剥信封。

真实事故（`docs/reviews/memory-pipeline-full-trace-20260805.md` §2 T1）：
LLM 的回复**以完整 `<bladex-memory>` 块开头**——模型把注入块回显了一遍。
该轮 status=ok、无 tool call → 整段回声成为 `assistant_conclusion` 进结论蒸馏，
**无剥离** → 自己注入的记忆被蒸馏回 Memory Index = ADR-0025 §6 担心的自反馈闭环，
而它已经在真实数据里发生。

修复：`_collect_conclusion_candidates` / `_collect_progress_candidates` 在
长度判定**之前**执行 `strip_envelopes()`。剥后 <40/60 字符照旧跳过。
"""

from __future__ import annotations

from bladex_core.consolidation_proxy import ProxyConsolidator
from bladex_core.distillation import DistillFact, DistillOutput
from bladex_core.fact import ConversationTurn


class _Distiller:
    model_name = "test-model"

    def __init__(self) -> None:
        self.seen: list[str] = []

    def distill(self, text: str) -> DistillOutput:
        return DistillOutput(facts=[], matter_proposals=[], model_name="test-model")

    def distill_conclusion(self, text: str) -> DistillOutput:
        self.seen.append(text)
        return DistillOutput(
            facts=[DistillFact(content=f"a::{text[:24]}", kind="decision", entities=["e"])],
            matter_proposals=[], model_name="test-model",
        )


# trace 报告 T1 的真实形状：注入块回声 + 末尾追加的一句
_INJECTION_ECHO = (
    "<bladex-memory>\n"
    "MUST: 回复用中文\n"
    "[Matter: BladeX ADR-0024/0025/0026 acceptance and beta release v0.1.0] "
    "The user's BladeX end-to-end probe code is bx3a984bcd.\n"
    "</bladex-memory>\n"
)
_REAL_CONCLUSION = (
    "已记住：端到端探针代码是 bxe3663a38，最喜欢的探针颜色是黄绿色 (chartreuse)，"
    "两条都会在后续会话里保持可召回。"
)


def _tags(cands, tag):
    return [c for c in cands if c.get("tags") == tag]


def test_conclusion_envelope_stripped_before_distill():
    """回声块不得进蒸馏输入——否则自己注入的记忆被蒸回 Memory Index。"""
    d = _Distiller()
    turn = ConversationTurn(
        session_id="s", user_id="u", ledger_key="k",
        assistant_conclusion=_INJECTION_ECHO + _REAL_CONCLUSION,
    )
    cands = ProxyConsolidator(distiller=d)._collect_candidates([turn])

    assert d.seen, "结论通道应仍然产出（剥后正文够长）"
    for text in d.seen:
        assert "<bladex-memory>" not in text
        assert "bx3a984bcd" not in text
    assert d.seen[0] == _REAL_CONCLUSION

    concl = _tags(cands, "origin:conclusion")
    assert len(concl) == 1
    assert "<bladex-memory>" not in concl[0]["source_text"]


def test_progress_envelope_stripped_before_distill():
    """进展通道同款。"""
    d = _Distiller()
    finding = (
        "我查了依赖版本，lancedb.index 并没有导出 IndexType，"
        "原来那行 create_index 必然抛 AttributeError，改用新的 config 形式重写。"
    )
    turn = ConversationTurn(
        session_id="s", user_id="u", ledger_key="k",
        assistant_progress=_INJECTION_ECHO + finding,
    )
    cands = ProxyConsolidator(distiller=d)._collect_candidates([turn])

    assert d.seen == [finding]
    prog = _tags(cands, "origin:progress")
    assert len(prog) == 1
    assert prog[0]["source_text"] == finding


def test_pure_envelope_conclusion_skipped():
    """整段都是回声 → 剥完为空 → 整条跳过（此前会因总长度达标而进蒸馏）。"""
    d = _Distiller()
    turn = ConversationTurn(session_id="s", user_id="u", ledger_key="k",
                            assistant_conclusion=_INJECTION_ECHO)
    cands = ProxyConsolidator(distiller=d)._collect_candidates([turn])

    assert d.seen == []
    assert _tags(cands, "origin:conclusion") == []


def test_length_gate_applies_after_stripping():
    """剥后不足门槛照旧跳过——门槛语义是"真实内容的长度"。"""
    d = _Distiller()
    turn = ConversationTurn(session_id="s", user_id="u", ledger_key="k",
                            assistant_conclusion=_INJECTION_ECHO + "好的。")
    ProxyConsolidator(distiller=d)._collect_candidates([turn])
    assert d.seen == []


def test_no_envelope_is_zero_regression():
    """无信封的聊天式 agent：逐字不变（剥离是纯函数，未命中原样返回）。"""
    d = _Distiller()
    plain = "结论：把降解阈值从 6 调到 3 之后，1-2 units 的超长会话全部触发改写。"
    turn = ConversationTurn(session_id="s", user_id="u", ledger_key="k",
                            assistant_conclusion=plain)
    cands = ProxyConsolidator(distiller=d)._collect_candidates([turn])
    assert d.seen == [plain]
    assert _tags(cands, "origin:conclusion")[0]["source_text"] == plain
