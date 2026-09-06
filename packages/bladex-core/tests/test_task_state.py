"""D1 决策引擎（task_state.py）——决策表 v2 的三剧本 + 分支边界。

文本样例直接复用 test_gm_continuation_wiring 里**量过的**那组（延续=1 个新主体、
换话题=19/23 个——分离得很开；改动文本必须重新量，不许编）。

钉住：
  ① 剧本 A（G3.0 四轮四卡）：首轮 R5、后续三轮全 R2 → 一张卡；
  ② 剧本 B（交错 A→B→A）：换话题 R5；回切无 unit_key 时**送裁决不猜**，
    同意图重发（unit_key 命中）才零 LLM 回切；
  ③ 剧本 C（并发双流）：锚 per-(agent, session) 互不污染——per-principal 锚
    会让 B 默认延续进 A（表 §1 反例检验，最容易被"简化"回去的地方）；
  ④ aux 四不：不判、不动锚、不改窗口；
  ⑤ 锚死（closed）= 无锚；unit 命中限窗口内且唯一。
"""

from __future__ import annotations

from bladex_core.task_state import (
    BRANCH_AUX,
    BRANCH_R1_UNIT_OVERRIDE,
    BRANCH_R2_ANCHOR,
    BRANCH_R4_UNIT,
    BRANCH_R5_JUDGE,
    ActiveWindow,
    AnchorTracker,
    TurnDecision,
    decide_turn,
    unique_unit_hit,
)

TAIL = "tail_continuation"
FP = "fingerprint"

# 量过的文本（出处 test_gm_continuation_wiring，实测新主体：延续 1 / 换话题 19）
_T1_USER = "复核一下泰安仁信入股泰山啤酒的增资扩股方式，看看有没有问题"
_T1_REF = _T1_USER + "\n复核完成，泰安仁信的增资扩股方式存在两处问题：出资时间与工商登记不一致，另外增资扩股的股权比例与公告披露的数字对不上，需要再核一遍。"
_FOLLOWUP = "增资扩股方式存在的问题，出资时间与工商登记不一致，需要再核一遍"
_SWITCH = "帮我在 macOS 上用 ollama 部署 qwen3.8 做本地推理，显存该怎么配置比较合适"


def _decide(anchor="", alive=True, ref="", window=(), src=TAIL, text=_FOLLOWUP,
            unit="", aux=False) -> TurnDecision:
    return decide_turn(
        anchor_matter_id=anchor, anchor_alive=alive, anchor_reference_text=ref,
        window=list(window), session_id_source=src, intent_text=text,
        unit_hit_matter=unit, auxiliary=aux)


# ── 剧本 A：G3.0 四轮四卡 → 一张卡 ───────────────────────────────────────


def test_scenario_a_four_turns_one_card() -> None:
    anchors = AnchorTracker()
    win = ActiveWindow()
    # 轮1：无锚、无 unit 命中 → R5（首轮，调用方裁决后开卡 m-A）
    d1 = _decide(src=FP, text=_T1_USER, window=win.snapshot())
    assert d1.branch == BRANCH_R5_JUDGE and d1.verdict == "needs_judgment"
    anchors.update("hermes", "s1", "m-A", _T1_REF)
    win.touch("m-A", 1000)
    # 轮2–4：追问 → 全部 R2 延续到锚，零 LLM
    for i in range(3):
        a, ref = anchors.get("hermes", "s1")
        d = _decide(anchor=a, ref=ref, window=win.snapshot(), text=_FOLLOWUP)
        assert d.branch == BRANCH_R2_ANCHOR and d.matter_id == "m-A", d
        anchors.update("hermes", "s1", d.matter_id, _T1_REF)
        win.touch(d.matter_id, 2000 + i)


# ── 剧本 B：交错 A→B→A ──────────────────────────────────────────────────


def test_scenario_b_interleave_switch_goes_to_judgment() -> None:
    # 锚在 A，换话题（新主体 19 ≥ 3）→ R5，窗口快照随判决走
    d = _decide(anchor="m-A", ref=_T1_REF, window=["m-A"], text=_SWITCH)
    assert d.branch == BRANCH_R5_JUDGE
    assert d.window_snapshot == ["m-A"]
    assert d.signals.get("from") == "R3"


def test_scenario_b_switch_back_without_unit_key_needs_judgment() -> None:
    """回切常态（换措辞）不许零 LLM 猜——S3/S5 只提名候选，裁决定夺。"""
    d = _decide(anchor="m-B", ref="ollama 部署相关参照文本", window=["m-B", "m-A"],
                text=_FOLLOWUP)
    assert d.branch == BRANCH_R5_JUDGE, "无 unit_key 的回切被零 LLM 直通了——违反键只提名纪律"


def test_scenario_b_switch_back_with_unit_key_is_zero_llm() -> None:
    """同意图重发（S2 unit_key 唯一命中窗口内 A）→ R1 零 LLM 回切。"""
    d = _decide(anchor="m-B", ref="ollama 部署相关参照文本", window=["m-B", "m-A"],
                text=_FOLLOWUP, unit="m-A")
    assert d.branch == BRANCH_R1_UNIT_OVERRIDE and d.matter_id == "m-A"


# ── 剧本 C：并发双流（表 §1 反例检验）────────────────────────────────────


def test_scenario_c_concurrent_streams_do_not_cross() -> None:
    anchors = AnchorTracker()
    anchors.update("claude-code", "sx", "m-X", "X 的参照")
    anchors.update("codex", "sy", "m-Y", "Y 的参照")
    # Codex 的轮只看得见自己的锚——锚 per-(agent, session)
    a_codex, _ = anchors.get("codex", "sy")
    a_cc, _ = anchors.get("claude-code", "sx")
    assert a_codex == "m-Y" and a_cc == "m-X"
    # 窗口 per-principal 共享（双方都看得见 X/Y），但 Y 流的轮 S1 不成立时
    # 走 R5 送裁决，不会默认延续进 X
    d = _decide(anchor="m-Y", ref="Y 的参照", window=["m-X", "m-Y"], text=_SWITCH)
    assert d.branch == BRANCH_R5_JUDGE
    assert d.matter_id == ""


# ── aux 四不（表 §2）─────────────────────────────────────────────────────


def test_aux_turn_is_transparent() -> None:
    d = _decide(anchor="m-A", ref=_T1_REF, window=["m-A"], text="内部子任务", aux=True)
    assert d.branch == BRANCH_AUX and d.verdict == "aux_skip" and not d.matter_id


def test_aux_does_not_move_anchor_or_window() -> None:
    anchors = AnchorTracker()
    win = ActiveWindow()
    anchors.update("a", "s", "m-A", "ref")
    anchors.update("a", "s", "m-B", "ref2", auxiliary=True)   # aux 不动锚
    assert anchors.get("a", "s")[0] == "m-A"
    win.touch("m-A", 100)
    win.touch("m-B", 200, auxiliary=True)                     # aux 不改窗口
    assert win.snapshot() == ["m-A"]


# ── 边界 ─────────────────────────────────────────────────────────────────


def test_dead_anchor_is_no_anchor() -> None:
    d = _decide(anchor="m-A", alive=False, ref=_T1_REF, window=["m-A"], text=_FOLLOWUP)
    assert d.branch == BRANCH_R5_JUDGE


def test_unit_hit_on_anchor_is_plain_r2() -> None:
    d = _decide(anchor="m-A", ref=_T1_REF, window=["m-A"], text=_FOLLOWUP, unit="m-A")
    assert d.branch == BRANCH_R2_ANCHOR and d.matter_id == "m-A"


def test_unit_hit_outside_window_is_ignored() -> None:
    """窗口是候选域的边界——窗口外的 unit 命中不算（闭合档案退出窗口后
    只能经 R5 显式召回，ADR-0031 §3.5）。"""
    d = _decide(anchor="", window=["m-B"], src=FP, text=_T1_USER, unit="m-Z")
    assert d.branch == BRANCH_R5_JUDGE


def test_no_anchor_unit_hit_is_r4() -> None:
    d = _decide(anchor="", window=["m-A"], src=FP, text=_T1_USER, unit="m-A")
    assert d.branch == BRANCH_R4_UNIT and d.matter_id == "m-A"


def test_unique_unit_hit_requires_uniqueness() -> None:
    idx = {"k1": {"m-A"}, "k2": {"m-A", "m-B"}}
    assert unique_unit_hit("k1", idx) == "m-A"
    assert unique_unit_hit("k2", idx) == ""      # 多卡命中 = 模糊 = 送裁决
    assert unique_unit_hit("", idx) == ""
    assert unique_unit_hit("k9", idx) == ""


def test_window_k_and_ordering_and_remove() -> None:
    win = ActiveWindow(k=3)
    for i, m in enumerate(["m-1", "m-2", "m-3", "m-4"]):
        win.touch(m, i)
    assert win.snapshot() == ["m-4", "m-3", "m-2"]   # 最旧的挤出
    win.remove("m-4")
    assert win.snapshot() == ["m-3", "m-2", "m-1"]
    win.touch("m-2", 100)
    assert win.snapshot()[0] == "m-2"
    # 时间不回拨
    win.touch("m-2", 50)
    assert win.snapshot()[0] == "m-2"


def test_default_is_continuation_not_new() -> None:
    """举证责任反转的最小断言：锚在 + 无新主体证据 → 延续，不是新开。
    （红线 7：不要把默认方向改回"默认无归属"——那是 80% 碎片率的直接来源。）"""
    d = _decide(anchor="m-A", ref=_T1_REF, window=["m-A"], text=_FOLLOWUP)
    assert d.verdict == "continue_to" and d.matter_id == "m-A"
