"""M0 止血卡的回归剧本（core 侧）。

对应 `docs/planning/memory-core-dev-tasks-20260806.md` 的 M0-1 / M0-4 / M0-8 /
M0-10 / M0-11，权威缺陷编号见 `docs/reviews/memory-six-modules-review-20260806.md`。

这批测试的共同性质：**钉死"门有没有被焊开"**。E1/E2 两条漏洞的要害不是判错，
而是判重整段被跳过且日志上完全不可见——所以每条剧本都直接构造
"应当被拦下的输入"，断言它确实被拦下，而不是断言某个中间变量。
"""

from __future__ import annotations

import math

import pytest
from bladex_core.consolidation_proxy import ProxyConsolidator
from bladex_core.distill_fidelity import identifier_novelty_override
from bladex_core.fact import ConversationTurn, Fact

# ── 公共替身 ───────────────────────────────────────────────────────────────

class _StubEmbedder:
    """确定性嵌入替身：同文本 → 同向量，不同文本 → 高度相似但不同的向量。

    目的是让 cosine 判重成为**唯一**能拦下同文候选的机制，从而暴露
    "E6.5 前置放行把 cosine 整段跳过" 这个漏洞。
    """

    def __init__(self) -> None:
        self._cache: dict[str, list[float]] = {}

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            if t not in self._cache:
                # 基向量 + 由文本 hash 决定的极小扰动 → 同文完全相同，
                # 异文 cosine 仍 >0.99（模拟真实 e5 在近似文本上的表现）。
                h = abs(hash(t)) % 1000
                v = [1.0, 0.02 * (h % 7), 0.02 * (h % 11)]
                n = math.sqrt(sum(x * x for x in v))
                self._cache[t] = [x / n for x in v]
            out.append(list(self._cache[t]))
        return out


class _PassthroughDistiller:
    """无 LLM 的透传蒸馏器（候选 = 原文），让本组测试只考察判重段。"""

    model_name = "passthrough"

    def distill(self, text: str):  # noqa: ANN201
        from bladex_core.distillation import DistillFact, DistillOutput

        return DistillOutput(
            facts=[DistillFact(content=text, kind="general", entities=[])],
            matter_proposals=[],
            model_name="passthrough",
        )


def _turn(text: str, key: str) -> ConversationTurn:
    return ConversationTurn(
        session_id="s1",
        user_id="u1",
        user_messages=[text],
        ledger_key=key,
        logical_turn=1,
    )


# ── M0-1 【E1 🔴】批内判重 existing_texts 同步 ────────────────────────────


def test_m0_1_same_batch_identical_text_with_identifier_is_deduped():
    """同批两条完全同文、带标识符（路径）的候选 → 第二条必须被判重。

    复现实锤场景：`文件 /a/b/p2_derived.py` ×2 同批只入 1 条。

    机制：E6.5 标识符前置放行拿 `existing_texts` 做对照集。修复前该数组
    **不随新入选的 fact 增长**，于是第二条同文候选的标识符仍"不在已存集合里"
    → 直接判新颖 → cosine 判重被整段绕过。
    """
    # 关键前提：existing_facts 非空 → 三个平行数组都被初始化（走 brute-force 路径），
    # 前置放行才真正参与判定（existing_texts 为空时 override 根本不会被调用）。
    seed = Fact(
        id="seed",
        content="用户偏好中文回复",
        embedding=_StubEmbedder().embed(["用户偏好中文回复"])[0],
        entities=["中文"],
    )
    dup_text = "文件 /a/b/p2_derived.py（内容 hash 3f2a91bc4d55）"

    consolidator = ProxyConsolidator(
        embedder=_StubEmbedder(),
        distiller=_PassthroughDistiller(),
    )
    facts = consolidator.consolidate_turns(
        [_turn(dup_text, "k1"), _turn(dup_text, "k2")],
        existing_facts=[seed],
    )

    assert len(facts) == 1, (
        f"同批同文候选应只入 1 条，实际 {len(facts)} 条："
        f"{[f.content for f in facts]}"
    )


def test_m0_1_novel_identifier_still_passes():
    """收窄不能误伤：带**新**标识符的候选仍应入库（E6.5 的初衷是救误杀）。"""
    # 长度须过 _MIN_CONTENT_CHARS（20），否则候选在装配阶段就被滤掉，测不到判重段。
    seed_text = "本轮已把探针码 bxe3663a38 写入记忆库以便后续核对"
    seed = Fact(
        id="seed",
        content=seed_text,
        embedding=_StubEmbedder().embed([seed_text])[0],
        entities=[],
    )
    consolidator = ProxyConsolidator(
        embedder=_StubEmbedder(),
        distiller=_PassthroughDistiller(),
    )
    facts = consolidator.consolidate_turns(
        [_turn("本轮已把探针码 bx3a984bcd 写入记忆库以便后续核对", "k1")],
        existing_facts=[seed],
    )
    assert len(facts) == 1


# ── M0-4 【E2 🔴】E6.5 标识符放行收窄：日期/金额不是标识符 ────────────────


@pytest.mark.parametrize(
    "candidate,existing",
    [
        # 泰山啤酒 cluster：变体之间唯一的差异就是各自引用的日期
        ("泰山啤酒共益债公告发布于 1月30日", ["泰山啤酒共益债公告发布于 2月26日"]),
        ("泰山啤酒共益债公告发布于 3月5日", ["泰山啤酒共益债公告发布于 1月30日"]),
        # 纯数字编号
        ("招募公告第 3 批", ["招募公告第 2 批"]),
        # 金额
        ("债权总额约 12.5亿", ["债权总额约 9.8亿"]),
        # 年月日形态
        ("会议定在 2026年8月6日", ["会议定在 2026年8月5日"]),
    ],
)
def test_m0_4_date_and_amount_do_not_override_novelty(candidate, existing):
    """带日期/金额/短编号的近重复**不得**被前置放行（判重门必须关着）。"""
    assert identifier_novelty_override(candidate, existing) is False


@pytest.mark.parametrize(
    "candidate,existing",
    [
        # 探针 token（混合字母数字）——放行的初衷，必须保留
        ("探针 bx3a984bcd 已写入", ["探针 bxe3663a38 已写入"]),
        # 路径
        ("修改了 memory_index.py", ["修改了 prefetch.py"]),
        # 常量名
        ("阈值改到 BLADEX_MATTER_PIN_MIN", ["阈值改到 BLADEX_MIN_IMPORTANCE"]),
        # content hash
        ("内容 hash 3f2a91bc4d55", ["内容 hash aa11bb22cc33"]),
    ],
)
def test_m0_4_real_identifiers_still_override(candidate, existing):
    """探针码 / hash / 路径 / 常量名仍然放行（不回归 E6.5 的本意）。"""
    assert identifier_novelty_override(candidate, existing) is True


def test_m0_4_pure_numeric_identifier_subset_is_not_novel():
    """候选只带日期这类"非标识符"时，退回 cosine 判重（返回 False）。"""
    assert identifier_novelty_override("发布于 2026年8月6日", []) is False


# ── M0-6 【I2 🔴】Matter 钉卡相关性门槛 ────────────────────────────────────


class _MatterRetriever:
    """带分数通道的 Matter 检索替身。

    真实检索层（`memory_index.search_matters`）把 `1 - cosine_distance` 回填到
    `Matter.embedding`（单元素 list）作为分数通道，这里逐字模拟该约定。
    """

    def __init__(self, matters):
        self._matters = matters

    def search_matters_by_query(self, query, k=5, q_vec=None):
        return list(self._matters)

    def get_facts_for_matter(self, matter_id, top_k=5):
        from bladex_core.fact import Fact as _F

        return [_F(id=f"{matter_id}-f1", content="泰山啤酒共益债公告已核实", category="general")]


class _FactRetriever:
    def __init__(self, facts=None):
        self._facts = facts or []

    def embed_query(self, query):
        return [0.1, 0.2, 0.3]

    def search(self, query, top_k=10, user_id=None, visibility=None, q_vec=None):
        return list(self._facts)

    def list_hard_rules(self):
        from bladex_core.fact import HardRule

        return [HardRule(content="用中文回答", fact_id="hr0")]


def _scored_matter(score: float, title: str = "泰山啤酒破产案"):
    from bladex_core.matter import Matter, MatterStatus

    m = Matter(
        matter_id="m-taishan",
        title=title,
        summary="核实是否存在 3000 万共益债公告",
        status=MatterStatus.ACTIVE,
    )
    m.embedding = [score]  # 检索层的分数通道
    return m


# ── M0-10 【H2② 🔴】lifecycle 事件去重持久化 ───────────────────────────────


def test_m0_10_same_event_folded_once_across_rebuilds():
    """同一事件跨两次 rebuild 重放只折叠一次（指纹随卡持久化）。"""
    from bladex_core.matter import Matter

    m = Matter(matter_id="m1", title="任务卡 A")
    assert m.record_lifecycle("reworked", "任务卡被打回：测试跑在空表上") is True
    v_after_first = m.version

    # 模拟第二次 rebuild：Matter 从存储读回（序列化往返），再重放同一事件
    reloaded = Matter.model_validate(m.model_dump())
    assert reloaded.record_lifecycle("reworked", "任务卡被打回：测试跑在空表上") is False
    assert len(reloaded.lifecycle) == 1
    assert reloaded.version == v_after_first, "重复事件不得推高 version（changed-since 判据）"


def test_m0_10_different_events_still_recorded():
    """不同事件/不同 detail 仍各自折叠（去重不能变成吞事件）。"""
    from bladex_core.matter import Matter

    m = Matter(matter_id="m1")
    assert m.record_lifecycle("assigned", "指派给 codex") is True
    assert m.record_lifecycle("reworked", "指派给 codex") is True     # 事件不同
    assert m.record_lifecycle("assigned", "指派给 claude") is True    # detail 不同
    assert len(m.lifecycle) == 3
    assert m.version == 3


def test_m0_10_fingerprint_ignores_timestamp():
    """指纹不含 ts —— 否则同一事件在两次重放里 ts 不同，去重形同虚设。"""
    from datetime import UTC, datetime

    from bladex_core.matter import Matter

    m = Matter(matter_id="m1")
    assert m.record_lifecycle("blocked", "等上游修复", ts=datetime(2026, 8, 1, tzinfo=UTC))
    assert not m.record_lifecycle("blocked", "等上游修复", ts=datetime(2026, 8, 6, tzinfo=UTC))
    assert len(m.lifecycle) == 1


def test_m0_10_legacy_matter_without_field_still_works():
    """历史 Matter（无 lifecycle_seen 字段）读回后照常工作（加法式演进）。"""
    from bladex_core.matter import Matter

    legacy = Matter.model_validate({"matter_id": "m-old", "title": "旧卡", "version": 7})
    assert legacy.lifecycle_seen == []
    assert legacy.record_lifecycle("closed", "已完结") is True
    assert legacy.version == 8


