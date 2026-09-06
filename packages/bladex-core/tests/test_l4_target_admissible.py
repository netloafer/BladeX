"""L4 裁决目标的准入谓词（ADR-0032 §4.5.1 的第三个闸点，2026-09-02）。

病灶：两条 L4 路径（台账缓存命中 / 现场裁决）都把 `result.matter_id` 直接落边，
**从不校验它还在不在 `linkable` 里**。调用方在候选池上做的过滤——例如账本
准入闸——因此被整段绕过：闸把锚卡从候选里摘掉，缓存判决又把它写了回来。

实测：第五次全量重建 4182 条归属漏 1 条（`gate-diff-split-20260902.md`
判据 3 那条：`from_ledger=True`，轮次早于账本创建 **77.2 小时**）。

与 `test_stale_judgment_target.py` 的分工——**两条都要有，互不覆盖**：

| 校验 | 问的问题 | 对本漏是否有效 |
|---|---|---|
| `_stale_target` | 这张卡**还在不在库** | ❌ 被闸摘掉的锚卡**仍然存在** |
| `target_admissible` | 这条 fact **能不能挂上去** | ✅ |

三组：① 两条路径都过闸；② 谓词未接 = 旧行为逐字不变；③ 谓词坏了不改归属。
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

NOW = datetime(2026, 9, 2, tzinfo=UTC)
ANCHOR = "m-4ac73ef7e0fa"          # live 病例里那张「评估 v5 架构」锚卡


class _Journal:
    def __init__(self, result: LinkJudgeResult | None) -> None:
        self._result = result

    def get_latest_judgment(self, fact_id: str) -> LinkJudgeResult | None:
        return self._result

    def put_judgment(self, fact_id: str, candidates: list[dict],
                     result: LinkJudgeResult) -> None:
        pass


class _ExplodingJudge:
    """🔴 判别力：走到"调 LLM"就炸 —— 重建零 LLM 是硬约束。

    被闸拦下必须落 L5，**不许回退调模型**：那会让重建代价随存量线性上涨，
    是另一个缺陷不是修复（与 `test_stale_judgment_target` 同一条）。
    """

    model_name = "should-never-be-called"

    def judge_links(self, items: list) -> list:
        raise AssertionError("走到 LLM 裁决了 —— 被闸拦下必须落 L5")


class _LinkJudge:
    """现场裁决桩：恒判 LINK 到 `target`（模拟"模型点名了一张不该挂的卡"）。"""

    model_name = "stub-judge"

    def __init__(self, target: str) -> None:
        self._target = target
        self.calls = 0

    def judge_links(self, items: list) -> list:
        self.calls += 1
        return [LinkJudgeResult(fact_id=getattr(it, "fact_id", "fact_x"),
                                verdict=LinkVerdict.LINK,
                                matter_id=self._target, reason="stub")
                for it in items]


def _fact(fid: str = "fact_x") -> Fact:
    """一条与锚卡**文本上毫无交集**的 fact —— 必须如此，否则走不到 L4。

    🔴 首版把 `proposal_titles` 写成与卡标题同一个字符串（"照着 live 长相搭"），
    结果 **L3 原生键当场命中**（`layer=L3, matched=title_alias_match`），
    L4 一次没跑、`judge.calls=0`，五条断言全红。
    与同日 `_TitleEchoDistiller` 那次同型、方向相反：那次匹配不上，这次匹配太早。
    ⇒ **搭夹具时先确认它落在哪一层**，长得像 live 不等于走同一条路。

    这么改剧本反而更忠实：live 那条漏里，把 fact 和锚卡连起来的**不是文本相似，
    是台账缓存判决本身**（ADR-0024 的开发过程 vs「评估 v5 架构」卡，主体不同）。
    """
    f = Fact(id=fid, content="ADR-0024 的 TaskUnit 切分实现于 07-29 落地",
             entities=["taskunit"])
    f.proposal_titles = ["ADR-0024 任务单元切分"]
    f.source_ledger_key = "u1/claude-code/s1/1756100000000-0"
    return f


def _matter(mid: str = ANCHOR, title: str = "评估 v5 架构开发进展") -> Matter:
    """锚卡。标题/aliases 与 `_fact()` 零 token 交集（见上）。"""
    return Matter(matter_id=mid, title=title, status=MatterStatus.ACTIVE,
                  aliases=[title])


# ── ① 两条 L4 路径都过闸 ─────────────────────────────────────────────────


def test_cached_verdict_is_gated() -> None:
    """台账缓存命中 ⇒ 过闸；被拒 ⇒ 落 L5，**不调 LLM**。

    这就是 live 那条漏的形状：`from_ledger=True` 的判决把早于账本创建
    77 小时的轮次挂回了锚卡。
    """
    p = AttributionPipeline(
        link_judge=_ExplodingJudge(),
        judgment_journal=_Journal(LinkJudgeResult(
            fact_id="fact_x", verdict=LinkVerdict.LINK,
            matter_id=ANCHOR, reason="cached")),
        matter_exists=lambda mid: True)      # 🔴 卡**确实还在** —— 存在性校验放行
    p.target_admissible = lambda _f, _mid: "before_created"

    dec = p.attribute(_fact(), [_matter()], now=NOW)
    assert dec.matter_id != ANCHOR, "🔴 缓存判决绕过了闸"
    assert dec.decision["layer"] == "L5"
    assert p.blocked_l4_targets["before_created"] == 1


def test_fresh_verdict_is_gated() -> None:
    """现场裁决路径也过闸 —— **纵深防御，不是主防线**。

    ⚠️ **一处更正**：我一度断言"现场裁决那半也在漏、两半生产里都活着"，
    **没读码就说了**。事实是 `_run_l4_batch` 已有候选成员校验
    （`link_id_not_in_candidates` ⇒ 降级 UNCERTAIN），而候选池已被调用方过滤，
    所以那一半**不会漏**。本条钉的是"这条路径上谓词确实被调用"，
    不是"它在生产里救过场"。

    本用例能走到谓词，是因为夹具让目标**在**候选内（不触发那道校验）——
    即主防线放行、本闸仍表态。
    """
    judge = _LinkJudge(ANCHOR)
    p = AttributionPipeline(link_judge=judge, judgment_journal=None)
    p.target_admissible = lambda _f, _mid: "other_ledger"

    dec = p.attribute(_fact(), [_matter()], now=NOW)
    assert judge.calls >= 1, "夹具前提破了：现场裁决没被调用"
    assert dec.matter_id != ANCHOR, "🔴 现场裁决绕过了闸"
    assert dec.decision["layer"] == "L5"
    assert p.blocked_l4_targets["other_ledger"] == 1


def test_admissible_target_still_links() -> None:
    """谓词放行 ⇒ L4 照常落边（加了闸不能把正常路径打掉）。"""
    p = AttributionPipeline(
        link_judge=_ExplodingJudge(),
        judgment_journal=_Journal(LinkJudgeResult(
            fact_id="fact_x", verdict=LinkVerdict.LINK,
            matter_id=ANCHOR, reason="cached")),
        matter_exists=lambda mid: True)
    p.target_admissible = lambda _f, _mid: ""     # 放行

    dec = p.attribute(_fact(), [_matter()], now=NOW)
    assert dec.matter_id == ANCHOR
    assert dec.decision["layer"] == "L4"
    assert sum(p.blocked_l4_targets.values()) == 0


# ── ② 谓词未接 = 旧行为逐字不变 ─────────────────────────────────────────


def test_no_predicate_keeps_old_behaviour() -> None:
    """`target_admissible is None` ⇒ 不校验（其它调用方零回归）。

    🔴 这条同时是**恒 0 读数的判据**：`blocked_l4_targets` 为空有两种成因
    ——谓词没接上 vs 谓词没开火。查的时候先看调用方有没有赋值。

    🔴 **也是本文件的夹具守卫**：`layer == "L4"` 这一句钉住"这条 fact 确实
    走到了 L4"。夹具一旦被改成能被 L1/L2/L3 提前命中，本条会先红——
    而不是让上面那些"过闸"的断言假绿（首版就是 L3 截胡，五条一起红）。
    """
    p = AttributionPipeline(
        link_judge=_ExplodingJudge(),
        judgment_journal=_Journal(LinkJudgeResult(
            fact_id="fact_x", verdict=LinkVerdict.LINK,
            matter_id=ANCHOR, reason="cached")),
        matter_exists=lambda mid: True)
    assert p.target_admissible is None, "默认必须是不校验"

    dec = p.attribute(_fact(), [_matter()], now=NOW)
    assert dec.matter_id == ANCHOR
    assert dec.decision["layer"] == "L4"


# ── ③ 谓词坏了不改归属 ──────────────────────────────────────────────────


def test_predicate_failure_does_not_change_attribution() -> None:
    """谓词自己抛异常 ⇒ 按放行处理 —— 仪器坏了不该改变归属结果。

    与 `_stale_target` 同一取舍：闸是**排除**机制，失效时的安全方向是"少排除"，
    不是"多排除"（多排除 = 静默丢边，比不排除更难发现）。
    """
    def _boom(_f, _mid):  # noqa: ANN001, ANN202
        raise RuntimeError("ledger pool closed")

    p = AttributionPipeline(
        link_judge=_ExplodingJudge(),
        judgment_journal=_Journal(LinkJudgeResult(
            fact_id="fact_x", verdict=LinkVerdict.LINK,
            matter_id=ANCHOR, reason="cached")),
        matter_exists=lambda mid: True)
    p.target_admissible = _boom

    dec = p.attribute(_fact(), [_matter()], now=NOW)
    assert dec.matter_id == ANCHOR
    assert dec.decision["layer"] == "L4"
    assert sum(p.blocked_l4_targets.values()) == 0


def test_unassigned_target_skips_the_predicate() -> None:
    """`__unassigned__` 是留池哨兵，不送谓词（与孤边审计那次同一教训）。"""
    probed: list[str] = []

    def _probe(_f, mid):  # noqa: ANN001, ANN202
        probed.append(mid)
        return "before_created"

    p = AttributionPipeline(
        link_judge=_ExplodingJudge(),
        judgment_journal=_Journal(LinkJudgeResult(
            fact_id="fact_x", verdict=LinkVerdict.LINK,
            matter_id="__unassigned__", reason="cached")),
        matter_exists=lambda mid: True)
    p.target_admissible = _probe

    p.attribute(_fact(), [_matter()], now=NOW)
    assert probed == [], "哨兵不该进谓词"
