"""归属准确率剧本验收（ADR-0018 §3.2 DPL 五层链接）。

剧本设计（楝子期式）：
  Matter A "给 BladeX 加 /v1/messages 支持" - 跨 3 agent（Hermes 调研 / Claude Code 实现 / Codex review）
  Matter B "修复 P3 key 覆盖 bug" - 干扰事 1
  Matter C "写用户文档" - 干扰事 2

DPL 形态：每条 fact 带蒸馏 proposal（"这属于哪件事儿"）+ entities。
  - 同一 Matter 的 facts 共享 proposal 标题 -> L3 alias-link 命中同一 Matter。
  - 不同 Matter 的 proposal 标题不同 -> 不同确定性 matter_id -> 不合并。

量化三指标：
  - 归属准确率：fact 归到正确 Matter 的比例
  - 误合并率（第一红线，目标 0）：不同 Matter 的 fact 被合并到同一 Matter
  - 误分裂率：同一 Matter 的 fact 被拆到不同 Matter（可容忍，手动合并兜底）
"""

from __future__ import annotations

from bladex_core.attribution import (
    UNASSIGNED_MATTER_ID,
    AttributionPipeline,
    LinkJudgeItem,
    LinkJudgeResult,
    LinkVerdict,
)
from bladex_core.fact import Fact
from bladex_core.matter import Matter


class _NoneJudge:
    """mock L4 裁决器：对所有 fact 返回 NONE（-> L5 新开 provisional，ADR-0018 §3.2）。

    无 judge 时有候选的 fact 会留池（§3.7 降级）；本剧本用 NONE judge 让新 topic 能新开。
    """

    @property
    def model_name(self) -> str:
        return "scenario-judge"

    def judge_links(self, items: list[LinkJudgeItem]) -> list[LinkJudgeResult]:
        return [LinkJudgeResult(fact_id=it.fact_id, verdict=LinkVerdict.NONE, reason="none")
                for it in items]

# ── 剧本数据 ──

_DIM = 64  # 向量维度（测试用，非真实 e5 1024 维；DPL 下向量只用于 L4 召回，本剧本走 L3）


def _vec(topic: int, variant: int = 0) -> list[float]:
    """生成语义向量：同 topic 高相似，跨 topic 低相似（仅 L4 召回用，本剧本 L3 解决）。"""
    v = [0.0] * _DIM
    base = topic * 20
    v[base] = 0.9 + variant * 0.01
    v[base + 1] = 0.8 + variant * 0.01
    v[base + 2] = 0.7
    return v


# 每条 fact 的 proposal 标题（DPL 归属映射的输入）+ entities
PROPOSAL_A = "给 BladeX 加 /v1/messages 支持"
PROPOSAL_B = "修复 P3 key 覆盖 bug"
PROPOSAL_C = "写 BladeX 用户文档"

# Ground truth: fact_id -> 正确的 matter label
GROUND_TRUTH = {
    # Matter A: /v1/messages 支持（跨 3 agent）
    "f-a1-hermes": "A",
    "f-a2-cc": "A",
    "f-a3-codex": "A",
    # Matter B: 修复 P3 key 覆盖 bug
    "f-b1-hermes": "B",
    "f-b2-codex": "B",
    # Matter C: 写用户文档
    "f-c1-cc": "C",
}

# 剧本 facts（带 proposal + entities；向量仅 L4 召回用）
SCENARIO_FACTS = [
    Fact(id="f-a1-hermes", content="调研 Anthropic /v1/messages API 的请求和响应格式",
         embedding=_vec(0, 0), source_session="hermes-s1",
         proposal_titles=[PROPOSAL_A], entities=["v1/messages", "Anthropic"]),
    Fact(id="f-a2-cc", content="实现 Anthropic /v1/messages 端点，包括流式和非流式处理",
         embedding=_vec(0, 1), source_session="cc-s1",
         proposal_titles=[PROPOSAL_A], entities=["v1/messages", "流式"]),
    Fact(id="f-a3-codex", content="review /v1/messages 实现代码，检查错误处理和边界情况",
         embedding=_vec(0, 2), source_session="codex-s1",
         proposal_titles=[PROPOSAL_A], entities=["v1/messages", "review"]),
    Fact(id="f-b1-hermes", content="P3 RocksDB 的 key 用 turn_index 当主键会导致重试时覆盖",
         embedding=_vec(1, 0), source_session="hermes-s2",
         proposal_titles=[PROPOSAL_B], entities=["P3 key", "RocksDB"]),
    Fact(id="f-b2-codex", content="修复 P3 key 覆盖问题，改用 Redis stream entry_id 作唯一后缀",
         embedding=_vec(1, 1), source_session="codex-s2",
         proposal_titles=[PROPOSAL_B], entities=["P3 key", "Redis"]),
    Fact(id="f-c1-cc", content="编写 BladeX proxy 的用户文档，包括安装和配置说明",
         embedding=_vec(2, 0), source_session="cc-s2",
         proposal_titles=[PROPOSAL_C], entities=["用户文档", "BladeX"]),
]


def _run_scenario() -> tuple[dict[str, str], list[Matter]]:
    """跑一遍剧本，返回 (fact_id -> matter_id, candidate_matters)。

    用 NONE judge：有候选但 L3 不匹配的 fact -> L4 NONE -> L5 新开 provisional。
    同 proposal 的后续 fact 经 L3 alias-link 链入同一 Matter。
    """
    pipeline = AttributionPipeline(link_judge=_NoneJudge())
    candidate_matters: list[Matter] = []
    fact_to_matter: dict[str, str] = {}

    for fact in SCENARIO_FACTS:
        decision = pipeline.attribute(fact, candidate_matters)
        if decision.is_new_matter:
            matter = pipeline.create_matter_for_decision(decision, centroid=fact.embedding)
            candidate_matters.append(matter)
            decision.matter_id = matter.matter_id
        fact_to_matter[fact.id] = decision.matter_id

    return fact_to_matter, candidate_matters


def test_scenario_false_merge_rate_is_zero():
    """T7 验收：误合并率为 0（第一红线，ADR-0018 §3.2 宁分勿合）。

    不同 Matter 的 fact 绝不能被合并到同一个 Matter。
    DPL 下：不同 proposal 标题 -> 不同确定性 matter_id -> 不合并。
    """
    fact_to_matter, _ = _run_scenario()

    # 检查误合并：不同 ground truth label 的 fact 不应在同一个 matter
    matter_to_labels: dict[str, set[str]] = {}
    for fid, label in GROUND_TRUTH.items():
        mid = fact_to_matter.get(fid, UNASSIGNED_MATTER_ID)
        matter_to_labels.setdefault(mid, set()).add(label)

    false_merges = sum(1 for labels in matter_to_labels.values() if len(labels) > 1)

    assert false_merges == 0, (
        f"False merge detected! {false_merges} matters have cross-topic facts. "
        f"matter_to_labels={matter_to_labels}"
    )


def test_scenario_accuracy_and_split_rate():
    """T7 验收：归属准确率 + 误分裂率（如实记录）。

    误分裂率可容忍（靠手动合并兜底），但需量化。
    """
    fact_to_matter, candidate_matters = _run_scenario()

    # 准确率：同一 label 的 fact 应归到同一 Matter
    label_to_matter: dict[str, str] = {}
    correct = 0
    total = len(GROUND_TRUTH)
    for fid, label in GROUND_TRUTH.items():
        mid = fact_to_matter.get(fid, UNASSIGNED_MATTER_ID)
        if label not in label_to_matter:
            label_to_matter[label] = mid
            correct += 1
        elif label_to_matter[label] == mid:
            correct += 1

    accuracy = correct / total

    # 误分裂率：同一 label 的 fact 被拆到不同 Matter
    splits = 0
    for label in set(GROUND_TRUTH.values()):
        label_facts = [fid for fid, lbl in GROUND_TRUTH.items() if lbl == label]
        if len(label_facts) <= 1:
            continue
        mids = {fact_to_matter.get(f, "") for f in label_facts}
        if len(mids) > 1:
            splits += 1
    split_rate = splits / len(set(GROUND_TRUTH.values()))

    # 所有创建的 Matter 应是 provisional（首成员）-> 后续 L3 链入后转正（rebuild 中转正，
    # 这里只测 pipeline 决策，create_matter 产出 provisional）
    print("\n=== ADR-0018 DPL 归属准确率剧本验收 ===")
    print(f"准确率: {accuracy:.1%} ({correct}/{total})")
    print("误合并率: 0 (第一红线 PASS)")
    print(f"误分裂率: {split_rate:.1%} ({splits} matters split)")
    print(f"生成 Matter 数: {len(candidate_matters)}")

    assert false_merges_zero(fact_to_matter), "误合并必须为 0"
    assert accuracy == 1.0, f"DPL 下同 proposal 应 100% 归一，got {accuracy}"
    assert len(candidate_matters) == 3, (
        f"应生成 3 个 Matter（A/B/C），got {len(candidate_matters)}"
    )


def false_merges_zero(fact_to_matter: dict[str, str]) -> bool:
    matter_to_labels: dict[str, set[str]] = {}
    for fid, label in GROUND_TRUTH.items():
        mid = fact_to_matter.get(fid, UNASSIGNED_MATTER_ID)
        matter_to_labels.setdefault(mid, set()).add(label)
    return all(len(labels) == 1 for labels in matter_to_labels.values())
