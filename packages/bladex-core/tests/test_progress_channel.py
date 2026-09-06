"""写入侧三处修复的验收（2026-07-29，探针驱动）。

探针 `scripts/probe_memory.py --inventory` 的结论：
    Memory Index 里 18 条 fact，16 条是「用户询问 XXX」/「用户要求 XXX」，
    claude-code 与 codex 各自跑完一整张任务卡，**各只沉淀 1 条**——
    还是那句"用户要求按照任务卡开发"。
    关于 visibility / except 纪律 / 150ms 预算 / e5 盖章，14 个关键词全部 0 命中。

即：**记住了"问了什么"，几乎没记"做了什么、发现了什么"。**

三处根因与修复：
    ① 结论通道要求"本轮无 tool call"——codex 37 轮里 35 轮带 tool call
       → 新增 `assistant_progress` 通道，收调工具**之前**写的说明
    ② aux 整轮跳过——claude-code 123 轮里 120 轮被跳
       → T3 的信封指纹不再整轮丢，只交给 envelope 剥离
    ③ 两条通道产出用 tags 区分，便于审计与调权
"""

from __future__ import annotations

from bladex_core.consolidation_proxy import ProxyConsolidator
from bladex_core.distillation import DistillFact, DistillOutput
from bladex_core.fact import ConversationTurn


class _Distiller:
    """记录每条进入蒸馏的文本，便于断言"哪些内容真的进了管线"。"""

    model_name = "test-model"

    def __init__(self) -> None:
        self.user_seen: list[str] = []
        self.assistant_seen: list[str] = []

    def distill(self, text: str) -> DistillOutput:
        self.user_seen.append(text)
        return DistillOutput(
            facts=[DistillFact(content=f"u::{text[:24]}", kind="event", entities=["e"])],
            matter_proposals=[], model_name="test-model",
        )

    def distill_conclusion(self, text: str) -> DistillOutput:
        self.assistant_seen.append(text)
        return DistillOutput(
            facts=[DistillFact(content=f"a::{text[:24]}", kind="decision", entities=["e"])],
            matter_proposals=[], model_name="test-model",
        )


_FINDING = (
    "我查了当前依赖版本，lancedb.index 里并没有 IndexType 这个导出，"
    "所以原来那行 create_index 必然抛 AttributeError。改用新的 config 形式重写。"
)


def _tags(cands, tag):
    return [c for c in cands if c.get("tags") == tag]


# ── ① 工作产出通道 ──


def test_progress_distilled_when_turn_has_tool_call():
    """带 tool call 的轮次，assistant 正文现在也会进蒸馏（此前整段丢弃）。"""
    d = _Distiller()
    turn = ConversationTurn(
        session_id="s", user_id="u", ledger_key="k1",
        user_messages=[],
        assistant_response=_FINDING,
        assistant_progress=_FINDING,     # 有 tool call → 走 progress 通道
    )
    cands = ProxyConsolidator(distiller=d)._collect_candidates([turn])

    assert _FINDING in d.assistant_seen
    prog = _tags(cands, "origin:progress")
    assert len(prog) == 1
    assert prog[0]["source_text"] == _FINDING


def test_conclusion_and_progress_are_tagged_apart():
    """两条通道的产出必须可区分——否则没法审计也没法按通道调权。"""
    d = _Distiller()
    turns = [
        ConversationTurn(session_id="s", user_id="u", ledger_key="k1",
                         assistant_conclusion="这是最终结论，" * 6),
        ConversationTurn(session_id="s", user_id="u", ledger_key="k2",
                         assistant_progress=_FINDING),
    ]
    cands = ProxyConsolidator(distiller=d)._collect_candidates(turns)

    assert len(_tags(cands, "origin:conclusion")) == 1
    assert len(_tags(cands, "origin:progress")) == 1


def test_progress_and_conclusion_mutually_exclusive_per_turn():
    """同一轮不该两条都走——proxy 侧按有无 tool call 二选一填。"""
    d = _Distiller()
    turn = ConversationTurn(
        session_id="s", user_id="u", ledger_key="k",
        assistant_conclusion="最终答案在这里，" * 6,
        assistant_progress="",
    )
    cands = ProxyConsolidator(distiller=d)._collect_candidates([turn])
    assert len(_tags(cands, "origin:conclusion")) == 1
    assert len(_tags(cands, "origin:progress")) == 0


def test_short_progress_filtered_as_noise():
    """"好的，我来看看" 这类过场话不该进库。"""
    d = _Distiller()
    turn = ConversationTurn(session_id="s", user_id="u", ledger_key="k",
                            assistant_progress="好的，我来看看。")
    cands = ProxyConsolidator(distiller=d)._collect_candidates([turn])
    assert _tags(cands, "origin:progress") == []
    assert d.assistant_seen == []


def test_progress_skipped_without_distiller():
    turn = ConversationTurn(session_id="s", user_id="u", ledger_key="k",
                            assistant_progress=_FINDING)
    cands = ProxyConsolidator()._collect_candidates([turn])
    assert _tags(cands, "origin:progress") == []


def test_progress_tolerates_distiller_without_capability():
    """旧蒸馏器没有 distill_conclusion → 跳过而不是崩（零改造兼容）。"""
    class _Legacy:
        model_name = "legacy"

        def distill(self, text: str) -> DistillOutput:
            return DistillOutput(facts=[], matter_proposals=[], model_name="legacy")

    turn = ConversationTurn(session_id="s", user_id="u", ledger_key="k",
                            assistant_progress=_FINDING)
    cands = ProxyConsolidator(distiller=_Legacy())._collect_candidates([turn])
    assert _tags(cands, "origin:progress") == []


# ── ② aux 收窄：信封指纹不再整轮丢弃 ──


def test_distill_only_aux_rules_are_declared_separately():
    """T3 的信封指纹必须与 Hermes 原生 aux 规则分开——前者不该整轮丢。"""
    from bladex_proxy.identity import DISTILL_ONLY_AUX_RULES, _AUX_USER_PATTERNS

    # 全集 = 模式表规则 + 结构化规则（U3 的 hermes_transcript_replay 用
    # "User:…\n\nAssistant:" 结构匹配、有意不进 _AUX_USER_PATTERNS——
    # "User:" 做子串前缀太泛会误伤真实用户消息，见 identity.classify_auxiliary）。
    all_rules = {name for name, _ in _AUX_USER_PATTERNS} | {"hermes_transcript_replay"}
    assert DISTILL_ONLY_AUX_RULES < all_rules, "信封指纹应是全部 aux 规则的真子集"
    # Hermes 原生的内部调用仍然整轮丢（T9 收益不丢）
    for native in ("summarization_checkpoint", "context_compaction_handoff",
                   "async_delegation_batch"):
        assert native in all_rules
        assert native not in DISTILL_ONLY_AUX_RULES


def test_envelope_turn_still_yields_assistant_output():
    """末条 user 是纯信封的轮次，assistant 的工作产出仍要收。

    这是 claude-code 120/123 轮被跳过的直接修复——信封说明的是"用户那侧没内容"，
    不代表"这一轮 assistant 没做实事"。
    """
    d = _Distiller()
    turn = ConversationTurn(
        session_id="s", user_id="u", ledger_key="k",
        user_messages=["<command-name>/model</command-name>"],   # 纯信封
        assistant_progress=_FINDING,
    )
    cands = ProxyConsolidator(distiller=d)._collect_candidates([turn])

    # 信封本身没进蒸馏
    assert d.user_seen == []
    # 但工作产出进了
    assert len(_tags(cands, "origin:progress")) == 1


# ── ③ 回归：原有通道不受影响 ──


def test_user_and_conclusion_channels_unchanged():
    d = _Distiller()
    turn = ConversationTurn(
        session_id="s", user_id="u", ledger_key="k",
        user_messages=["请帮我核实一下泰山啤酒破产案的共益债公告"],
        assistant_conclusion="已确认公示页面标题与链接，管理人为山东泰山啤酒。" * 2,
    )
    cands = ProxyConsolidator(distiller=d)._collect_candidates([turn])
    assert len(d.user_seen) == 1
    assert len(_tags(cands, "origin:conclusion")) == 1
