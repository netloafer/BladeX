"""MS-16 归属层再门控回归（T4a，2026-08-10）。

证据：docs/reviews/matter-attribution-audit-20260810.md +
复现仪器 scripts/audit_l2l3_gate_replay.py——live 库 L2=466 边里
415 条走「proposal_title == 累积 alias」精确命中（雪球），43 条走
单个泛词（'bladex'）子串/实体命中。本文件钉死修后语义：

  - L2 门控：提案全串相等单票放行；弱命中需 >=2 个独立 fact 侧信号；
    单泛词（同一 token 多处出现）不足票。
  - L3：累积 matter.entities 退出匹配面（对照 = title + aliases 原生键）。
  - generic 放行 / 07-13 B+A 剧本语义不变（test_attribution.py 原样守护）。

aliases 停止自动累积（雪球源头）在 proxy 侧：
packages/bladex-proxy/tests/test_ms16_alias_accumulation.py。
"""

from __future__ import annotations

from bladex_core.attribution import (
    AttributionPipeline,
    AttributionSource,
    ExplicitSignalDetector,
)
from bladex_core.fact import Fact
from bladex_core.matter import Matter, MatterOrigin, MatterStatus


def _fact(content: str, fact_id: str, *, entities=None, proposals=None) -> Fact:
    return Fact(id=fact_id, content=content, embedding=[1.0, 0.0],
                entities=entities or [], proposal_titles=proposals or [])


def _matter(mid: str, title: str, aliases: list[str]) -> Matter:
    return Matter(matter_id=mid, title=title, aliases=aliases,
                  status=MatterStatus.ACTIVE, origin=MatterOrigin.AUTO)


# README 病例 matter 的原生形态（创建时标题 + 首批成员实体）
_README = _matter(
    "m-readme", "BladeX 双语 README 编写任务（beta-b9-1）",
    ["BladeX 双语 README 编写任务（beta-b9-1）", "BladeX", "README", "beta-b9"],
)
# 续接语 + 标题关键词 BladeX —— turn 级 L2 信号成立
_CONT_MSG = "继续 BladeX 这边的工作"


def test_gate_blocks_single_ubiquitous_token():
    """单个全项目高频 token（'BladeX'）出现在 entity 与 proposal 两处，
    仍只算 1 票 -> 阻断（README 吸尘器主路径回归）。"""
    pipeline = AttributionPipeline()
    fact = _fact(
        "SKILL.md 正文新增了 unabyssapp one memory 研究指针", "f-unabyss",
        entities=["unabyssapp", "BladeX"],
        proposals=["深入研究 unabyssapp 的 one memory 多 agent 架构并与 BladeX 深度对比"],
    )
    d = pipeline.attribute(fact, [_README], user_message=_CONT_MSG)
    assert d.source != AttributionSource.EXPLICIT
    assert d.matter_id != "m-readme"


def test_gate_allows_multi_signal_fact():
    """两个独立 fact 侧信号各自命中原生键 -> 放行（同题 fact 不受影响）。"""
    pipeline = AttributionPipeline()
    fact = _fact(
        "README.zh 与 README 的双语结构已按复核意见调整", "f-readme-ok",
        entities=["README", "beta-b9"],
        proposals=["按复核意见修改双语 README 规划文档"],
    )
    d = pipeline.attribute(fact, [_README], user_message=_CONT_MSG)
    assert d.source == AttributionSource.EXPLICIT
    assert d.matter_id == "m-readme"


def test_gate_exact_proposal_single_vote():
    """提案标题与原生键全串规范化相等 = 最强特异信号，单票放行。"""
    pipeline = AttributionPipeline()
    fact = _fact(
        "deep-research 技能缺少金融子技能", "f-exact",
        entities=["deep-research"],
        proposals=["bladex 双语 readme 编写任务（beta-b9-1）"],  # casefold 后 == title
    )
    d = pipeline.attribute(fact, [_README], user_message=_CONT_MSG)
    assert d.source == AttributionSource.EXPLICIT
    assert d.matter_id == "m-readme"


def test_gate_generic_fact_still_passes():
    """generic fact（无 entities/proposals）仍信任 turn 显式信号放行（语义不变）。"""
    pipeline = AttributionPipeline()
    fact = _fact("用户希望大模型直接回答", "f-generic")
    d = pipeline.attribute(fact, [_README], user_message=_CONT_MSG)
    assert d.source == AttributionSource.EXPLICIT
    assert d.matter_id == "m-readme"


def test_gate_accumulated_alias_snowball_shape_blocked():
    """雪球形态端到端：即使污染 alias 已在（存量库形态），后续外题 fact
    只有单泛词重叠时也不再链入——弱命中票数不足。

    注：全串相等的错提案仍会命中污染 alias（存量库在 T5 全量重建前
    保留污染），根治 = 停止累积（proxy 侧测试）+ T5 重建。本测试钉住
    的是"新 fact 与污染 alias 只有泛词重叠"这一多数形态。
    """
    pipeline = AttributionPipeline()
    polluted = _matter(
        "m-polluted", "BladeX 双语 README 编写任务（beta-b9-1）",
        ["BladeX 双语 README 编写任务（beta-b9-1）", "BladeX",
         "深入研究 unabyssapp 的 one memory 多 agent 架构并与 bladex 深度对比"],  # 历史污染
    )
    fact = _fact(
        "unabyssapp 的记忆架构分析出了第二稿", "f-unabyss-2",
        entities=["unabyssapp"],
        proposals=["unabyssapp 记忆架构第二稿"],
    )
    d = pipeline.attribute(fact, [polluted], user_message=_CONT_MSG)
    assert d.source != AttributionSource.EXPLICIT


def test_detector_gate_argentina_scenarios_unchanged():
    """07-13 B+A 语义保持：DNS fact 阻断 / 阿根廷双实体 fact 放行。"""
    det = ExplicitSignalDetector()
    arg = _matter("m-arg", "阿根廷 vs 佛得角比赛赔率查询与预测", ["阿根廷", "佛得角"])
    dns = _fact("用户设置了1.1.1.2的DNS服务器", "f-dns",
                entities=["1.1.1.2", "DNS"], proposals=["设置DNS服务器"])
    ok = _fact("用户询问阿根廷 vs 佛得角的市场赔率", "f-arg",
               entities=["阿根廷", "佛得角"], proposals=["查询阿根廷佛得角赔率"])
    assert det._fact_relevant(dns, arg) is False
    assert det._fact_relevant(ok, arg) is True


def test_l4_receives_blocked_facts():
    """T4a-3 方向：被 L2/L3 阻断的 fact 落到 L4（judge 不可用则留池），
    不是静默丢弃——'该归的没归'两端坏里的后一端由 L4/重判接手。"""
    pipeline = AttributionPipeline()  # 无 judge
    fact = _fact(
        "泰山啤酒方案C核心资产估值约4.5亿", "f-taishan",
        entities=["泰山啤酒", "方案C"], proposals=["泰山啤酒破产重整投资分析"],
    )
    yongan = _matter("m-yongan", "永安林业近三年一期经营情况分析",
                     ["永安林业近三年一期经营情况分析", "永安林业", "000663.SZ"])
    d = pipeline.attribute(fact, [yongan], user_message="继续 永安林业 的分析")
    # L2/L3 均不命中 -> 进 L4；judge 不可用 -> 降级留池 + pending
    # （ADR-0018 §3.7：新开暂停，恢复后重判补收；绝不静默丢）
    assert d.source == AttributionSource.UNASSIGNED
    assert d.pending_judgment is True
    assert (d.decision or {}).get("layer") == "L5"
