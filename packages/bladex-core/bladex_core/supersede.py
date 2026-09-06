"""取代键（ADR-0026 §5 机制3 / U5.2）——同 (scope, subject, attribute) 新压旧，非破坏。

「修正4：不是覆盖，是非破坏取代」——同取代键的新条目产生时：
  - 旧条写 `t_invalid=now` + `superseded_by=<新 id>`（退出 current-only 召回）；
  - 新→旧写一条 SUPERSEDES 边（可参与排序、可解释）；
  - **旧条保留**（as-of / 审计可见）。覆盖会毁重建等价性（G6）。

本模块只给**决策**（找到该被取代的旧条），应用（写字段 + 边）由 consolidation/rebuild 做。
纯函数、agent 中立、可 O(1) 判定（调用方维护 key→最新 id 索引）。

有取代键的类型见 `_SUPERSEDABLE_KINDS`（assertion / preference / **file_ref**，
subject·attribute 必填）；lesson / procedure 等累积型无键 → 不取代。
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from datetime import UTC, datetime

from bladex_core.fact import Fact, ItemKind

# 有取代键的条目类型（subject·attribute 构成键）。其它类型累积、不互相取代。
# FILE_REF（ADR-0026 §4.3 复盘补，2026-08-03）：键 = (scope, 路径, "file")——
# 「hash 变 = 旧 ref 置 t_invalid」的失效机制由取代键统一兑现；
# 同 hash（内容相同）走合并 strength+1，不同 hash 走取代。
_SUPERSEDABLE_KINDS = {ItemKind.ASSERTION, ItemKind.PREFERENCE, ItemKind.FILE_REF}


def _norm(s: str) -> str:
    """规范化（NFKC + casefold + collapse ws），与 attribution._normalize 同源。"""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    return " ".join(s.casefold().split())


def supersede_key(fact: Fact) -> tuple[str, str, str] | None:
    """取代键 = (scope, 规范化 subject, 规范化 attribute)。无键返回 None。

    仅 `_SUPERSEDABLE_KINDS`（assertion/preference/file_ref）且 subject·attribute
    均非空才有键。
    """
    if fact.item_kind not in _SUPERSEDABLE_KINDS:
        return None
    subj = _norm(fact.subject)
    attr = _norm(fact.attribute)
    if not subj or not attr:
        return None
    return (fact.scope or "", subj, attr)


def supersede_key_str(fact: Fact) -> str:
    """取代键的**单一字符串形态**（无键返回 ""）——跨进程、可落账的稳定身份。

    为什么需要它（MQ-S38，2026-08-20）：`fact_id = sha256(ledger_key + ":" + content)`
    是**内容派生**的，蒸馏 prompt 一换版（v6→v7 那种全库重蒸）就整批换代，
    历史 turn 里记的 `injected_fact_ids` 全部指空，注入命中信号（ref_count /
    相关钟）在换版时被静默清零。取代键 `(scope, subject, attribute)` 不随正文措辞变，
    是我们手上**唯一**已经定义好的跨版本身份。

    🔴 **只有一处定义**：注入侧（producer）与回写侧（consumer）必须用同一个函数——
    两边各拼一遍字符串，格式一漂就是"看起来在救、其实一条都对不上"，
    而那种失败完全静默（MS-17 命名空间那次的同型）。

    覆盖面诚实说明：有键的类型 = `_SUPERSEDABLE_KINDS`（**assertion / preference /
    file_ref**）且 subject·attribute 均非空；**lesson / procedure 天然无键**
    （它们累积、不互相取代），换版后仍会丢，计入 orphan 读数，不假装救回。
    🔴 2026-08-20 更正：本行初稿把 `file_ref` 误写进"无键"那一列，而它 08-03 起
    就在 `_SUPERSEDABLE_KINDS` 里（键 = `(scope, 路径, "file")`，见 :23）——
    判据要从枚举定义读，不从记忆默写。
    """
    key = supersede_key(fact)
    return "\x1f".join(key) if key else ""


def find_superseded(new_fact: Fact, existing: list[Fact]) -> Fact | None:
    """在 existing 里找应被 new_fact 取代的**当前有效**旧条（同取代键、t_invalid 为空）。

    返回该旧条（caller 置 t_invalid/superseded_by + 写 SUPERSEDES 边），无则 None。
    守 G1 误合并=0：只按精确取代键匹配，绝不按相似度合并跨主题。
    """
    key = supersede_key(new_fact)
    if key is None:
        return None
    for old in existing:
        if old.id == new_fact.id:
            continue
        if old.t_invalid is not None:
            continue  # 已失效的不重复取代
        if old.superseded_by:
            continue
        if supersede_key(old) == key:
            return old
    return None


def _entity_jaccard(a: list[str], b: list[str]) -> float:
    sa = {_norm(x) for x in a if x}
    sb = {_norm(x) for x in b if x}
    if not sa and not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def should_merge(
    new_fact: Fact, old_fact: Fact, *, entity_overlap_min: float = 0.6,
) -> bool:
    """判断 new 是否是 old 的**同义重述**（→ 合并：old.strength+1，不新增 new）。

    与取代区分：取代是同键**新值换旧值**；合并是同键**同义重述**——同取代键 +
    内容高度一致（实体 Jaccard ≥ 阈值 或 规范化 content 相等）。仅 assertion/preference。

    守 G1：只在同取代键 + 高重叠时合并，绝不跨主题合并。
    """
    k = supersede_key(new_fact)
    if k is None or supersede_key(old_fact) != k:
        return False
    if old_fact.t_invalid is not None or old_fact.superseded_by:
        return False
    # 规范化 content 相等 = 明显重述
    if _norm(new_fact.content) == _norm(old_fact.content):
        return True
    # FILE_REF：entities 恒 = [路径]，实体重叠恒 1.0 会把"hash 变了"误判成重述
    # → 只认 content 相等（含 hash）为重述；hash 变 = 取代（失效机制的入口）。
    if new_fact.item_kind == ItemKind.FILE_REF:
        return False
    # 实体高重叠 = 同义重述
    return _entity_jaccard(new_fact.entities, old_fact.entities) >= entity_overlap_min


@dataclass
class SupersedeMergePlan:
    """取代/合并计划（ADR-0026 §5 机制2/3 / U5.2+U5.3）。纯决策，caller 应用+持久化。"""

    add: list[Fact] = field(default_factory=list)               # 要新增的 fact（去掉被合并掉的）
    bump: list[tuple[Fact, int]] = field(default_factory=list)  # (旧 fact, 新 strength) —— 合并
    invalidate: list[tuple[Fact, str]] = field(default_factory=list)  # (旧 fact, 取代它的新 id) —— 取代
    supersede_edges: list[tuple[str, str]] = field(default_factory=list)  # (新 id, 旧 id) SUPERSEDES 边


def plan_supersede_merge(
    new_facts: list[Fact], existing_facts: list[Fact], *,
    entity_overlap_min: float = 0.6, now: datetime | None = None,
) -> SupersedeMergePlan:
    """对一批新 fact，按取代键决定合并 / 取代 / 新增（非破坏，G6 重建等价）。

    对每条新 fact（仅 assertion/preference 有键）：
      - 同键存在**同义重述**旧条 → 合并：旧 strength+1，新条不加入。
      - 同键存在**不同值**当前旧条 → 取代：旧置 t_invalid+superseded_by，新加入，写 SUPERSEDES 边。
      - 无键 / 无同键旧条 → 直接新增。

    existing_facts 视为「当前 Memory Index 已有 + 本批已决定加入」的并集（caller 传入时合并两者）；
    同键在本批内多条时，后者对前者也走取代/合并（用 running 视图）。
    """
    if now is None:
        now = datetime.now(UTC)
    plan = SupersedeMergePlan()
    # running 视图：已有 + 本批已加入的，供批内同键判定
    running: list[Fact] = list(existing_facts)

    for nf in new_facts:
        key = supersede_key(nf)
        if key is None:
            plan.add.append(nf)
            running.append(nf)
            continue
        # 找同键当前有效旧条
        target = next(
            (o for o in running
             if o.id != nf.id and o.t_invalid is None and not o.superseded_by
             and supersede_key(o) == key),
            None,
        )
        if target is None:
            plan.add.append(nf)
            running.append(nf)
        elif should_merge(nf, target, entity_overlap_min=entity_overlap_min):
            # 合并：旧 strength+1，新条丢弃
            new_strength = target.strength + 1
            plan.bump.append((target, new_strength))
            target.strength = new_strength  # 更新 running 视图
            target.updated_at = now  # ADR-0027 §5.3：变更要让导出增量看得见
        else:
            # 取代：旧失效，新加入
            target.t_invalid = now
            target.superseded_by = nf.id
            target.updated_at = now  # 同上——被取代是"变更"里最要紧的一种
            plan.invalidate.append((target, nf.id))
            plan.supersede_edges.append((nf.id, target.id))
            plan.add.append(nf)
            running.append(nf)
    return plan
