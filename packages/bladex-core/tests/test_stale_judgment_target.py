"""L4 台账缓存判决的目标存在性校验（`task-anchor-materialize-20260901.md` T3）。

病灶：`_apply_l4_verdict` 直接拿缓存判决里的 `result.matter_id` 落边，
`candidates` 参数收下不用、不校验目标是否还在。全量重建 `calls=0` 全靠台账
复用 ⇒ 任何在两次重建之间消失的 Matter 都留下**孤边**：卡不在库所以按 matter
聚合的读数看不见它，边还在所以成员计数照算。实测 259 条 / 50 个 id，全部 L4。

本文件把三件事钉死，每条带判别力对照：
① 目标还在 ⇒ 照常复用（不能因为加了校验就把正常复用打掉）；
② 目标没了 ⇒ 落 L5，**不调 LLM**（重建零 LLM 是硬约束）；
③ 没有谓词（`matter_exists=None`）⇒ 完全保持旧行为（其它调用方零回归）。
"""

from __future__ import annotations

from datetime import UTC, datetime

from bladex_core.attribution import (
    AttributionPipeline,
    LinkJudgeResult,
    LinkVerdict,
)
from bladex_core.fact import Fact
from bladex_core.matter import Matter, MatterStatus

NOW = datetime(2026, 9, 1, tzinfo=UTC)


class _Journal:
    """只回放一条固定判决的台账（命中即复用，miss 返回 None）。"""

    def __init__(self, result: LinkJudgeResult | None) -> None:
        self._result = result
        self.puts: list[str] = []

    def get_latest_judgment(self, fact_id: str) -> LinkJudgeResult | None:
        return self._result

    def put_judgment(self, fact_id: str, candidates: list[dict],
                     result: LinkJudgeResult) -> None:
        self.puts.append(fact_id)


class _ExplodingJudge:
    """🔴 判别力：一旦走到"调 LLM"就炸。

    重建路径的硬约束是零 LLM（`ledger_hits=3286 / calls=0`）。把缓存失效
    变成 LLM 调用，会让重建代价随存量线性上涨——那是另一个缺陷，不是修复。
    """

    model_name = "should-never-be-called"

    def judge_links(self, items: list) -> list:
        raise AssertionError("走到 LLM 裁决了 —— 缓存失效必须落 L5，不许回退调模型")


def _fact(fid: str = "fact_x") -> Fact:
    f = Fact(id=fid, content="v5 架构评估的一条结论", entities=["bladex"])
    f.proposal_titles = ["bladex v5 架构评估"]
    return f


def _matter(mid: str, title: str = "在库的卡") -> Matter:
    return Matter(matter_id=mid, title=title, status=MatterStatus.ACTIVE,
                  aliases=[title])


def _cached(target: str) -> LinkJudgeResult:
    return LinkJudgeResult(fact_id="fact_x", verdict=LinkVerdict.LINK,
                           matter_id=target, reason="cached")


def _pipeline(journal: _Journal, exists) -> AttributionPipeline:  # noqa: ANN001
    return AttributionPipeline(link_judge=_ExplodingJudge(),
                               judgment_journal=journal,
                               matter_exists=exists)


# ── ① 目标还在：照常复用 ──────────────────────────────────────────────────


def test_live_target_is_reused_from_journal() -> None:
    """卡还在 ⇒ 走 L4 复用，零 LLM。加了校验不能把正常路径打掉。"""
    p = _pipeline(_Journal(_cached("m-alive")), lambda mid: True)
    dec = p.attribute(_fact(), [_matter("m-alive")], now=NOW)
    assert dec.matter_id == "m-alive"
    assert dec.decision["layer"] == "L4"
    assert p.stale_judgment_targets == 0


def test_target_present_in_candidates_skips_the_probe() -> None:
    """目标就在本轮候选里 ⇒ 不问谓词（谓词返回 False 也不该改变结果）。

    判别力：候选是 top-k 召回，"在候选里"是比谓词更强的存在性证据；
    先看手上的再问外面，顺带省掉一次库查询。
    """
    probed: list[str] = []

    def _exists(mid: str) -> bool:
        probed.append(mid)
        return False

    p = _pipeline(_Journal(_cached("m-alive")), _exists)
    dec = p.attribute(_fact(), [_matter("m-alive")], now=NOW)
    assert dec.matter_id == "m-alive", "在候选里却被判失效"
    assert probed == [], "目标已在候选里，不该再问谓词"


# ── ② 目标没了：落 L5，不调 LLM ──────────────────────────────────────────


def test_stale_target_falls_to_l5_not_the_llm() -> None:
    p = _pipeline(_Journal(_cached("m-gone")), lambda mid: False)
    dec = p.attribute(_fact(), [_matter("m-other")], now=NOW)
    assert dec.matter_id != "m-gone", "🔴 边仍指向不存在的卡 = 孤边"
    assert dec.decision["layer"] == "L5"
    assert p.stale_judgment_targets == 1


def test_stale_target_counter_accumulates() -> None:
    """计数器是孤边的替代读数：修法生效后它应≈原先的孤边增量。"""
    p = _pipeline(_Journal(_cached("m-gone")), lambda mid: False)
    p.attribute(_fact("f1"), [_matter("m-other")], now=NOW)
    p.attribute(_fact("f2"), [_matter("m-other")], now=NOW)
    assert p.stale_judgment_targets == 2


def test_unassigned_target_is_not_treated_as_stale() -> None:
    """`__unassigned__` 是留池哨兵，不是丢失的 Matter。

    首版孤边审计把它算进了 277 条里（27 条 L5），分子混进不该算的东西——
    与 MQ-N8 同族：读数看着更糟，但指错方向。
    """
    p = _pipeline(_Journal(_cached("__unassigned__")), lambda mid: False)
    p.attribute(_fact(), [_matter("m-other")], now=NOW)
    assert p.stale_judgment_targets == 0


# ── ③ 没有谓词：旧行为一字不变 ──────────────────────────────────────────


def test_no_predicate_keeps_old_behaviour() -> None:
    """`matter_exists=None` ⇒ 不校验，照写缓存目标（其它调用方零回归）。"""
    p = AttributionPipeline(link_judge=_ExplodingJudge(),
                            judgment_journal=_Journal(_cached("m-gone")))
    dec = p.attribute(_fact(), [_matter("m-other")], now=NOW)
    assert dec.matter_id == "m-gone"
    assert dec.decision["layer"] == "L4"
    assert p.stale_judgment_targets == 0


def test_probe_failure_does_not_change_attribution() -> None:
    """谓词自己抛异常 ⇒ 按"没失效"处理，不因为仪器坏了改变归属结果。"""
    def _boom(mid: str) -> bool:
        raise RuntimeError("db closed")

    p = _pipeline(_Journal(_cached("m-gone")), _boom)
    dec = p.attribute(_fact(), [_matter("m-other")], now=NOW)
    assert dec.matter_id == "m-gone"
    assert p.stale_judgment_targets == 0
