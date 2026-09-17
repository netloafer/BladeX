"""写入时裁决（M2-1 / 复核第 2 层 · 附录 C 机制 2）——模块 2/3 的统一解。

## 为什么把去重与冲突检测合成一件事

复核对这两个模块各查了一遍，结论指向同一处：

- **模块 2（去重合并）**：判据是 cosine + 实体 Jaccard，而 e5 在本库的噪声底就是
  0.816–0.93。阈值定高了漏网（带日期的近重复变体互相看不见），定低了误合并。
  `supersede` 精确键在真实蒸馏产出上区分度 ≈ 0（E3）——它只对模板化输入有效。
- **模块 3（冲突检测）**：检测器**输出无人消费**（F1），两条极性规则实为
  "任意两数不同"检测器（F2），反义词典就是验收剧本本身（F3）。

而 3.9 用 live 库 370 条向量做的可行性复盘给出了决定性的一条：

    冲突对手方在新事实写入时刻的近邻排名：3/4 在 **top-1**（cos 0.956/0.965/0.977），
    第 4 例（跨语言）在 top-8。
    → **e5 向量已经把冲突双方送到了同一张桌子上，只是桌边没坐判定者。**

所以正确的机制不是"更好的相似度阈值"，也不是"更全的反义词典"，
而是把 0.90 以上的邻居打成一个包，交给便宜档模型判一次
**ADD / UPDATE / NOOP**。成本实测 ≈ 9 次/天。

## 三个 verdict 的语义（**没有 DELETE**）

    ADD    这是新事实 → 入库
    UPDATE 它取代了某些旧条 → 旧条 `t_invalid` + `superseded_by`，新条入库（取代关系落在 Fact 字段上，不写边）
    NOOP   同义重述，库里已有 → 新条丢弃，旧条 `strength + 1`（MS-7）

UPDATE 是**非破坏取代**（Mem0g 同型）：旧条保留、可 as-of 查询、重建等价性不破。
删除会毁掉 G6，且"哪条被取代了"本身就是有价值的记忆。

## 本模块的边界

**只给决策，不做应用。** 写字段、写边、写台账都由 consolidation / rebuild 做——
与 `supersede.plan_supersede_merge` 同一个分工模式。
纯函数 + Protocol，agent 中立，不 import 任何 proxy/LLM 的东西。
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Protocol

from pydantic import BaseModel, Field

from bladex_core.fact import Fact
from bladex_core.supersede import should_merge, supersede_key


def _cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度（批内邻居用）。维度不一致 / 零向量 → 0.0。

    刻意不依赖 numpy：core 的硬依赖只有 pydantic，而这里是 O(维度) 的小运算，
    批内候选数是个位数量级。
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class AdjudicationOp(str, Enum):
    """裁决结果三选一。**刻意没有 DELETE**（见模块 docstring）。"""

    ADD = "add"        # 新事实，入库
    UPDATE = "update"  # 取代旧条（非破坏：旧条 t_invalid + superseded_by）
    NOOP = "noop"      # 同义重述，库中已有 → 丢弃新条、旧条 strength+1


class NeighborFact(BaseModel):
    """裁决输入包里的一条邻居（库中已有的近邻事实）。

    只带裁决需要的字段——把整个 Fact 塞进 prompt 是浪费 token，
    且会让模型被 embedding、scope 这类与判断无关的东西干扰。
    """

    fact_id: str
    content: str
    subject: str = ""
    attribute: str = ""
    item_kind: str = ""
    similarity: float = 0.0
    #: MS-11：这条事实的**业务时间**（ISO，`t_valid` → `t_observed` → `created_at`）。
    #: 没有它，裁决器无从判断谁先谁后 —— 见 `AdjudicationInput.observed_at`。
    observed_at: str = ""


class AdjudicationInput(BaseModel):
    """一条候选 + 它的邻居集合 = 一个裁决包。"""

    candidate_id: str
    content: str
    subject: str = ""
    attribute: str = ""
    item_kind: str = ""
    entities: list[str] = Field(default_factory=list)
    #: 蒸馏产出的 `corrects` 槽（M1-1 ⑥）——新事实**自带的修正自声明**。
    #: F4 说的"最高精度信号无人使用"，指的就是它：
    #: "此前误将 X 当作 Y" 这种句子在裁决输入里是明文证据，比任何相似度都硬。
    corrects_hint: str = ""
    #: 🔴 MS-11：候选被观测到的时间（ISO，取来源 turn 的 ts）。
    #:
    #: 为什么必须有：裁决的三个 verdict 里有两个（UPDATE / NOOP）本质上是**时序判断**
    #: ——"新的取代旧的"。此前 prompt 直接假设"候选是更新的那条"，
    #: 而这个假设只在**增量到达顺序**下隐式成立：live 流量里新 turn 天然后到。
    #: 全量重建/积压重放没有这个保证（RocksDB key 序 ≠ 时间序，跨 session 尤其），
    #: 于是旧断言会以"NEW item"身份进裁决，把修正宣告当成待取代的邻居 →
    #: **取代链方向整个反转**。
    #: 把时间摆到桌面上，判断就不必再依赖调用顺序这个隐含契约。
    observed_at: str = ""
    neighbors: list[NeighborFact] = Field(default_factory=list)


class AdjudicationVerdict(BaseModel):
    """一条候选的裁决结果。"""

    candidate_id: str
    op: AdjudicationOp = AdjudicationOp.ADD
    #: UPDATE 时 = 被取代的旧条；NOOP 时 = 被强化的那条（MS-7 取首个）。
    target_fact_ids: list[str] = Field(default_factory=list)
    reason: str = ""
    #: 是不是降级产生的（LLM 超时/失败 → 兜底 ADD）。观测用，不影响应用。
    fallback: bool = False


class AdjudicatorProtocol(Protocol):
    """裁决器接口（可 mock，单测不跑真模型）。实现在 proxy 侧。"""

    @property
    def model_name(self) -> str:
        ...

    @property
    def prompt_ver(self) -> str:
        """裁决 prompt 版本（进台账 key；prompt 改 → 递增 → 视为新裁决）。"""
        ...

    def adjudicate(self, items: list[AdjudicationInput]) -> list[AdjudicationVerdict]:
        """批量裁决。失败/超时/解析失败 → 全部降级 ADD（宁多勿丢），不抛。"""
        ...


# ── 快路径：精确取代键命中，不必调 LLM ────────────────────────────────────


def fast_path_verdict(
    candidate: Fact, neighbors: list[Fact], *, entity_overlap_min: float = 0.6,
) -> AdjudicationVerdict | None:
    """精确取代键命中时直接出裁决，**不调 LLM**。命中不了返回 None。

    这是 `plan_supersede_merge` 被"吸收退役"后的去处（复核第 2 层）：
    它没有被删掉，而是降为裁决前的确定性快路径——
    同 `(scope, subject, attribute)` 是**结构性证据**，不需要模型再判一次。

    E3 已经说明这条路在真实蒸馏产出上区分度 ≈ 0（subject/attribute 很少精确对齐），
    所以它命中率低；但命中时它 100% 正确且零成本，没有理由不先试。

    🔴 **2026-08-31：一次收敛整个键，不再命中第一条就 return**（MQ-S32 病 B 第 4 环）。

    旧实现在 `for old in neighbors` 里命中首个同键 old 就返回，于是 N 条同键并存
    要靠 **N−1 次配对正确的裁决**才收敛，而每次配对都要重新过一遍概率召回
    ⇒ **收敛概率随 N 增大而下降**，正反馈。08-31 全库读数把它从推断变成实证：
    `update_landed` 55 条 = **取代执行成功了、键仍并存**（`why_no_supersede --batch`，
    `docs/benchmarks/supersede-why-batch-20260831.md`）。

    新规则（`apply_verdict` 的 UPDATE 分支本就遍历全部 targets，不必改）：

    - 同键 live 邻居里**只要有一条不是重述** ⇒ `UPDATE`，targets = **全部同键 live
      邻居**（含那些重述的）。依据是取代键自己的语义（ADR-0026 §5 机制3
      「同键新压旧」）：同一 `(scope, subject, attribute)` 只该有一个 current 值，
      候选是这一键最新的陈述，其余一律让位。
    - 全部都是重述 ⇒ `NOOP`（候选丢弃、首条 strength+1），与旧行为一致。

    🔴 为什么不是"有一条重述就 NOOP"：那会让 `[B(旧值), A(重述)]` 这种邻居顺序
    判成 NOOP，**B 原地留 current**——比旧行为还差。顺序不该决定结果。
    """
    key = supersede_key(candidate)
    if key is None:
        return None
    same_key = [
        old for old in neighbors
        if old.id != candidate.id
        and old.t_invalid is None and not old.superseded_by
        and supersede_key(old) == key
    ]
    if not same_key:
        return None
    restated = [old for old in same_key
                if should_merge(candidate, old, entity_overlap_min=entity_overlap_min)]
    if len(restated) == len(same_key):
        return AdjudicationVerdict(
            candidate_id=candidate.id, op=AdjudicationOp.NOOP,
            target_fact_ids=[restated[0].id], reason="fast_path:same_key_restated",
        )
    return AdjudicationVerdict(
        candidate_id=candidate.id, op=AdjudicationOp.UPDATE,
        target_fact_ids=[old.id for old in same_key],
        reason=f"fast_path:same_key_new_value(targets={len(same_key)})",
    )


def _observed_iso(f: Fact) -> str:
    """一条 Fact 的**业务时间**（MS-11）：`t_valid` → `t_observed` → `created_at`。

    三者都取不到就返回空串，由 prompt 侧按"未知"处理，**不猜**——
    猜一个时间会让裁决基于虚构的先后关系做取代。
    """
    # 顺序有讲究：`t_valid`（这条事实**何时成立**）优先于 `t_observed`（何时被摄入）。
    # 全量重建时后者对全库几乎是同一个值（重建那一刻），拿它比先后等于没比。
    for attr in ("t_valid", "t_observed", "created_at"):
        v = getattr(f, attr, None)
        if v is not None:
            try:
                return v.isoformat()
            except AttributeError:
                return str(v)
    return ""


def build_input(
    candidate: Fact, neighbors: list[tuple[Fact, float]], *, top_k: int = 10,
    observed_at: str = "",
) -> AdjudicationInput:
    """把一条候选 + 它的近邻装成裁决包。

    `top_k=10` 不是随手取的：3.9 实测跨语言冲突对（EN 新事实 vs ZH 旧事实）
    排在 **top-8** —— k=5 会把它漏掉。语言钉死（M1-4）之后这类对的 cosine
    会回到 0.95+ 区间，但在存量数据上 k 必须够宽。
    """
    picked = sorted(neighbors, key=lambda p: -p[1])[:top_k]
    return AdjudicationInput(
        candidate_id=candidate.id,
        content=candidate.content,
        subject=candidate.subject,
        attribute=candidate.attribute,
        item_kind=candidate.item_kind.value if hasattr(candidate.item_kind, "value")
        else str(candidate.item_kind),
        entities=list(candidate.entities or []),
        corrects_hint=str(getattr(candidate, "corrects", "") or ""),
        # MS-11：候选时间优先取调用方给的 turn.ts（重放时那才是"这条何时发生"），
        # 没给则回落 fact 自身的观测时间。
        observed_at=observed_at or _observed_iso(candidate),
        neighbors=[
            NeighborFact(
                fact_id=f.id, content=f.content,
                subject=f.subject, attribute=f.attribute,
                item_kind=f.item_kind.value if hasattr(f.item_kind, "value")
                else str(f.item_kind),
                similarity=round(float(sim), 4),
                observed_at=_observed_iso(f),
            )
            for f, sim in picked
        ],
    )


def plan_supersede_replay(
    records: list[tuple[str, str, list[str], Any]], present: set[str],
) -> tuple[dict[str, tuple[str, Any]], dict[str, int]]:
    """按 judgment 台账重放取代关系（MQ-S32 病 B 第五环）。**纯函数**。

    入参 `records` = `(seq, candidate_id, target_ids, ts)` 列表，
    `present` = 本次重建实际写入的 fact_id 集合。
    返回 `(target_id -> (candidate_id, ts), 跳过分类计数)`。

    ## 为什么需要它

    `fact.id` 是确定性的（`sha256(ledger_key+content)`），全量重建会重建出同一批
    id；而取代关系（`t_invalid` / `superseded_by`）**只活在 Fact 对象上**，
    重建时从零重推。ADR-0018 §4.1 建台账的明写理由就是"让 rebuild 重放零 LLM"，
    但 `consolidation:*` 判决在生产里**零消费者**（`linking.py` 按白名单跳过它们
    ——那是对的，L4 不该读取代判决）。⇒ 每次全量重建都把已付费的结论扔掉。
    2026-08-31 实测：1124 条同键并存里 **237 条（21.1%）** 出自这里。

    ## 语义（逐条对齐生产，不另发明）

    - **按 seq 升序、先到先得**：`apply_verdict` 的 UPDATE 分支跳过
      `old.t_invalid is not None`（"取代链不许被后来者改写"，既有测试守着），
      所以重放也必须首判生效、后判跳过——**倒过来取"最后一条"会重放出一个
      生产从未产生过的状态**。
    - **`ts` 用记录自带的时间**，由调用方取自 `JudgmentRecord.ts`：
      用 `now()` 会让两次重建产出不同的 `t_invalid`，
      "重建等价性"就变成一句测不出来的话。

    ## 四条安全阀（每条一个计数，都要能 grep）

    1. `candidate_absent` —— 取代方不在本次产出里（内容变了 ⇒ id 变了，或被墓碑
       过滤）。**不能让一条不存在的 fact 去取代别人。**
    2. `target_absent` —— 被取代方不在本次产出里。墓碑天然走这条：墓碑过滤发生在
       重放之前 ⇒ 被 forget 的 fact 不会因重放复活（ADR-0012 §3.6 减项最后生效）。
    3. `already_superseded` —— 该 target 已被更早的判决取代。
    4. `self_target` —— 判决指向自己（脏数据护栏，不该出现但不许静默通过）。
    """
    plan: dict[str, tuple[str, Any]] = {}
    skipped: dict[str, int] = {}

    def _skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    for _seq, candidate_id, target_ids, ts in sorted(records, key=lambda r: r[0]):
        if candidate_id not in present:
            _skip("candidate_absent")
            continue
        for tid in target_ids:
            if tid == candidate_id:
                _skip("self_target")
                continue
            if tid not in present:
                _skip("target_absent")
                continue
            if tid in plan:
                _skip("already_superseded")
                continue
            plan[tid] = (candidate_id, ts)
    return plan, skipped


def adjudicate_outage(
    before: dict | None, after: dict | None, threshold: float,
) -> tuple[bool, int, int, float]:
    """本轮裁决是否判为**上游中断**。返回 `(要不要中止, 条数, 降级条数, 比例)`。

    与蒸馏中断守卫（`BLADEX_DISTILL_OUTAGE_ABORT_RATIO`）同型、同理由：
    **裁决失败是降级不抛的**——`_all_add` 把整块判成 ADD，而 ADD 是一个合法判决，
    于是"上游断了"和"这批确实都是新事实"在下游看来一模一样，消费标记照打。
    08-31 全库读数量到它的实害：`adjudicator_error` 46 条同键并存
    （`docs/benchmarks/supersede-why-batch-20260831.md`）。

    🔴 分母是**候选条数**不是调用次数：分块后一次调用判 25 条，
    按调用数算会让"一块全降级"被稀释成 1/N。

    信息不全（拿不到计数器 / 本轮零候选）⇒ `(False, …)` = 无从判断，按旧行为走，
    **不冒充"没问题"**——调用方据此不打日志，而不是打一条"守卫通过"。
    `threshold <= 0` = 守卫关闭（与蒸馏侧同一个约定）。

    core 决策、proxy 落盘：与 `supersede_plan` / `adjudication_records` 同一分工。
    """
    if not before or not after or threshold <= 0:
        return False, 0, 0, 0.0
    items = int(after.get("items", 0)) - int(before.get("items", 0))
    degraded = int(after.get("degraded_items", 0)) - int(before.get("degraded_items", 0))
    if items <= 0 or degraded <= 0:
        return False, max(items, 0), max(degraded, 0), 0.0
    ratio = degraded / items
    return ratio >= threshold, items, degraded, ratio


def apply_verdict(
    verdict: AdjudicationVerdict, candidate: Fact, by_id: dict[str, Fact],
    *, now,
) -> tuple[bool, list[Fact]]:
    """把裁决**应用**到对象上（in-place），返回 `(要不要入库新条, 被改动的旧条)`。

    刻意与决策分开：决策可能来自 LLM、快路径或台账重放，
    但应用必须只有一份实现——否则三条来源会各自漂移一点，
    最后没人说得清库里那条 `t_invalid` 是谁写的。

    - ADD    → (True, [])
    - UPDATE → 旧条 `t_invalid=now` + `superseded_by=新id`，新条入库 → (True, [旧条…])
    - NOOP   → 旧条 `strength+1`（MS-7），新条**不**入库 → (False, [旧条])

    MS-7 说明：NOOP 时**不动 `last_hit_at`** —— 重复陈述是"强化"，不是"命中"。
    相关钟只该被检索命中推动，掺进写入侧信号会让三时钟的判据失真。
    """
    changed: list[Fact] = []
    if verdict.op is AdjudicationOp.ADD:
        return True, changed

    targets = [by_id[fid] for fid in verdict.target_fact_ids if fid in by_id]
    if not targets:
        # 目标不在库里（可能已被别的裁决取代）→ 降级为 ADD，宁多勿丢
        return True, changed

    if verdict.op is AdjudicationOp.UPDATE:
        for old in targets:
            if old.id == candidate.id or old.t_invalid is not None:
                continue
            old.t_invalid = now
            old.superseded_by = candidate.id
            old.updated_at = now
            changed.append(old)
        return True, changed

    # NOOP：同义重述 → 旧条强化，新条丢弃
    old = targets[0]
    old.strength = int(old.strength or 1) + 1
    old.updated_at = now
    changed.append(old)
    return False, changed
