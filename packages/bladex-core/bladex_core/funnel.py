"""记忆漏斗指标的**共享词汇表**（M4-2 / MS-8）。

## 为什么要一个单独的模块

漏斗横跨两个进程：

    consolidator ── 候选 → 蒸馏 → 判重 → 裁决 → 入库 ──┐
                                                      ├─ 同一条漏斗
    proxy ─────────────────── 召回 → 注入 → 命中回写 ──┘

两端各自暴露 `/metrics`（2026-08-06 用户拍板：**两个独立端点**，便于检查和维护——
哪半边没数一眼可见，不会被中转层掩盖）。但**指标名必须是同一套**，
否则拼不出一条漏斗，而"两边名字对不上"这种事没人会主动发现。

所以名字定义在 `bladex-core`（两个包都依赖它），不定义在任何一端。
这与 `flags.py` 同一条纪律：**共享的常量只许有一处定义**。

## 漏斗读法

任何一段计数为 0 而它上一段非 0 = 那一段是断的。这正是复核里 D7/E8/F6/G7/H7/I5
六条"缺观测"共同指向的东西——机制有没有在跑，此前只能靠翻日志猜。

    candidates ──→ distill ──→ dedup ──→ adjudicate ──→ stored
                                                            │
    recall ←─────────────────────────────────────────────  ┘
      └─→ inject ──→ hit（回写）

## 命名约定

`bladex_memory_<段>_total{...}`，与既有 `bladex_requests_total` 同风格。
库存健康度（MS-8）是 gauge：`bladex_memory_library_<项>`。
"""

from __future__ import annotations

# ── 写入侧（consolidator 进程）────────────────────────────────────────────

#: 候选装配产出的候选数。label: channel=user|conclusion|progress|file_ref|document
FUNNEL_CANDIDATES = "bladex_memory_candidates_total"

#: 蒸馏调用。label: outcome=ok|empty|parse_fail|call_fail|ledger_hit
#: `empty` 与 `*_fail` 必须分开——"这轮真没事实"和"上游断了"在降级实现里长得一样，
#: 那正是 2026-08-05 事故里唯一缺的那条信息（MS-1 同源）。
FUNNEL_DISTILL = "bladex_memory_distill_total"

#: 判重四段。label: verdict=novel|cosine_kill|entity_rescue|identifier_pass|error_pass
#: 复核 E8：判重杀了多少、放行了多少此前**全部无指标**，E1/E2 那种"门被焊开"
#: 的状态在日志上完全不可见。
FUNNEL_DEDUP = "bladex_memory_dedup_total"

#: 写入时裁决（M2 之后）。label: verdict=add|update|noop|fallback_add
FUNNEL_ADJUDICATE = "bladex_memory_adjudicate_total"

#: 真正入库的条目。label: item_kind=assertion|preference|...
FUNNEL_STORED = "bladex_memory_stored_total"

# ── 读取侧（proxy 进程）──────────────────────────────────────────────────
#
# 2026-09-03 S1：`FUNNEL_RECALL` / `FUNNEL_INJECT` 随主动注入检索路径（唯一埋点
# `InjectionSource.build_planes`）一起删除。
# 2026-09-06 F0.3（H1 销账）：`FUNNEL_HIT`（bladex_memory_hit_total）也删——它自 M4-2 起零埋点，
# 命中记录本该随 S1 一起清（瘦身 F7 写了没清）。读取侧现在**零段**；0.3.0 重建检索时三段一起重立。

# ── MS-8：库存健康度（gauge，consolidator 空闲轮刷新）─────────────────────

#: 库内条目总数。label: item_kind=...
LIB_FACTS = "bladex_memory_library_facts"
#: 已失效（t_invalid 非空）条目数——取代链有没有在运转
LIB_INVALIDATED = "bladex_memory_library_invalidated"
#: Matter 状态分布。label: status=provisional|active|dormant|closed
LIB_MATTERS = "bladex_memory_library_matters"
#: provisional → established 的转正率（0–1）
LIB_PROMOTION_RATE = "bladex_memory_library_promotion_rate"
#: file_ref 在**向量平面**的残留数（应恒 0；E1.3 之后它不该进向量表）
LIB_FILE_REF_IN_VECTORS = "bladex_memory_library_file_ref_in_vectors"
#: 僵尸向量数（有向量行、meta 里没有对应 fact）——M0-7 的对账结果
LIB_ZOMBIE_VECTORS = "bladex_memory_library_zombie_vectors"
#: provenance 分布。label: provenance=user_direct|conclusion|progress|...
LIB_PROVENANCE = "bladex_memory_library_provenance"

#: 全部指标名（守卫测试用：两端不得各自造名字）
ALL_FUNNEL_METRICS: tuple[str, ...] = (
    FUNNEL_CANDIDATES, FUNNEL_DISTILL, FUNNEL_DEDUP,
    FUNNEL_ADJUDICATE, FUNNEL_STORED,
)

ALL_LIBRARY_METRICS: tuple[str, ...] = (
    LIB_FACTS, LIB_INVALIDATED, LIB_MATTERS, LIB_PROMOTION_RATE,
    LIB_FILE_REF_IN_VECTORS, LIB_ZOMBIE_VECTORS, LIB_PROVENANCE,
)

#: 漏斗的顺序（`status --traffic` 按此顺序渲染；也是"哪一段断了"的判读顺序）
FUNNEL_ORDER: tuple[str, ...] = (
    FUNNEL_CANDIDATES, FUNNEL_DISTILL, FUNNEL_DEDUP,
    FUNNEL_ADJUDICATE, FUNNEL_STORED,
)


class NullFunnel:
    """无操作漏斗（未接指标时用）。

    存在的理由：漏斗埋点会散布在热路径与提炼管线里，调用方**不该**为了
    "这里有没有 metrics" 到处写 `if self._metrics is not None`——那种散落的判空
    正是机制腐化的温床（某处漏写就静默少一段计数，而少一段恰恰看不出来）。
    """

    def inc(self, metric: str, value: float = 1.0,
            labels: dict[str, str] | None = None) -> None:
        return

    def set_gauge(self, metric: str, value: float,
                  labels: dict[str, str] | None = None) -> None:
        return


NULL_FUNNEL = NullFunnel()
