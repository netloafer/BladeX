"""冷启相关性裁剪（ADR-0028 §3 / E4.3）。

## 场景即动机

长 session 跑了很久，用户突然问一个简单问题——现状会把整段历史塞给 LLM。
真实事故：两条简单问题分别带着 147K / 221K 上下文直发上游。

## 为什么只在 cache 冷时做

ADR-0026「禁止向量裁剪上下文」在本卡改为**条件禁令**：

  - **cache 温热期：维持禁令**。裁剪 = 打碎前缀 = cache 全失效，
    省下的 token 远不够 cache 损失，且破坏 assembly 的确定性；
  - **cache 冷启期：允许裁剪**。cache 反正要重建，此时裁剪只赚不赔。

判定见 `cache_state.SessionCacheRegistry.is_warm`。

## 算法（确定性，同输入同裁剪 → 重放等价）

    units      = build_task_units(messages)          # ADR-0024 既有切分，天然保 tool 配对
    protected  = {system 消息} ∪ {open 单元} ∪ {最近 K=3 个单元}
    candidates = closed ∧ ∉protected ∧ unit_chars > MIN_PRUNE_CHARS
    score(u)   = cos(q_vec, embed_passage(u.intent_text))
    按 score 升序贪心移除，直到 total_chars ≤ TARGET 或 candidates 用尽
    score ≥ KEEP_FLOOR 的即使超预算也保留
    移除形态 = 整单元替换为一行占位（ADR-0025 重取机制可找回）

**不用 LLM 打分**（对齐 LLMLingua 的"占位 + 按需恢复"思路，但不采用其 LLM 打分）
——保重放等价 + 热路径零 LLM。
"""

from __future__ import annotations

import math
import time
from typing import Any

import structlog
from bladex_core.task_unit import TaskUnit, TaskUnitStatus, build_task_units, unit_indices

from bladex_proxy.cache_state import SessionCacheState

logger = structlog.get_logger()

# ── 参数（`[标定]`：默认值先行，E5 评估集 / 真实库分布出数后回写任务卡）──
PROTECT_RECENT_UNITS = 3      # [标定] 最近 K 个单元无条件保留
MIN_PRUNE_CHARS = 2000        # 小于此体积的单元不值得裁（占位符本身也占字符）
TARGET_CHARS = 60_000         # [标定] 裁剪目标体积
KEEP_FLOOR = 0.55             # [标定] 相关性高于此值的单元即使超预算也保留
COLD_BUDGET_MS_DEFAULT = 1000  # 冷启路径独立预算（cache 反正冷，不抢 200ms 热路径预算）

_PLACEHOLDER = "[BladeX: 早前任务「{intent}」上下文已归档，需要时请求恢复]"
_INTENT_PREVIEW_CHARS = 40


def _text_of(msg: dict) -> str:
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            p.get("text", "") for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        )
    return ""


def _unit_chars(unit: TaskUnit, messages: list[dict]) -> int:
    return sum(len(_text_of(messages[i])) for i in unit_indices(unit, messages))


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _intent_text(unit: TaskUnit, messages: list[dict]) -> str:
    if 0 <= unit.intent_index < len(messages):
        return _text_of(messages[unit.intent_index])
    return ""


def apply_frozen_plan(
    messages: list[dict],
    plan: dict[str, Any],
) -> tuple[list[dict], int]:
    """温热期：按冻结计划原样复用上一轮的 L2 摘要与裁剪（ADR-0028 §2）。

    为什么不能"重算一遍"：L2 摘要正文含 Memory Index 蒸馏事实，每轮召回结果都可能变，
    重算 = 摘要文本变 = 前缀变 = cache 全失效。裁剪同理（q_vec 每轮不同）。
    所以温热期只做一件事：**把上一轮的决定按 unit_key 原样再贴一遍**。

    计划形态：
        {"l2_summary_msg": dict|None, "l2_dropped_unit_keys": [...],
         "pruned_unit_keys": [...]}

    unit_key 是内容派生的（sha256(intent_text)），历史前缀没变时它跨轮稳定，
    故同一份计划贴到同一段历史上，产出逐字节一致。
    返回 (messages, 受影响单元数)。
    """
    summary_msg = plan.get("l2_summary_msg")
    l2_keys = set(plan.get("l2_dropped_unit_keys") or [])
    pruned_keys = set(plan.get("pruned_unit_keys") or [])
    if not l2_keys and not pruned_keys:
        return messages, 0

    units = build_task_units(messages)
    by_key = {u.unit_key: u for u in units}
    touched = 0

    drop: set[int] = set()
    placeholder_at: dict[int, str] = {}
    summary_at = -1

    for key in l2_keys:
        u = by_key.get(key)
        if u is None:
            continue
        idxs = unit_indices(u, messages)
        if not idxs:
            continue
        if summary_at < 0 or idxs[0] < summary_at:
            summary_at = idxs[0]
        drop.update(idxs)
        touched += 1

    for key in pruned_keys:
        u = by_key.get(key)
        if u is None:
            continue
        idxs = unit_indices(u, messages)
        if not idxs or idxs[0] in drop:
            continue
        intent = _intent_text(u, messages).strip().replace("\n", " ")
        placeholder_at[idxs[0]] = _PLACEHOLDER.format(
            intent=intent[:_INTENT_PREVIEW_CHARS])
        drop.update(idxs[1:])
        touched += 1

    if not touched:
        return messages, 0

    out: list[dict] = []
    inserted = False
    for i, msg in enumerate(messages):
        if i == summary_at and summary_msg is not None and not inserted:
            out.append(summary_msg)
            inserted = True
        if i in placeholder_at:
            out.append({"role": "user", "content": placeholder_at[i]})
        elif i not in drop:
            out.append(msg)
    return out, touched


def prune_cold_context(
    messages: list[dict],
    *,
    q_vec: list[float] | None,
    embed_passage: Any,          # noqa: ANN401  callable(list[str]) -> list[list[float]]
    state: SessionCacheState | None = None,
    target_chars: int = TARGET_CHARS,
    protect_recent: int = PROTECT_RECENT_UNITS,
    keep_floor: float = KEEP_FLOOR,
    min_prune_chars: int = MIN_PRUNE_CHARS,
    budget_ms: int = COLD_BUDGET_MS_DEFAULT,
) -> tuple[list[dict], dict[str, Any]]:
    """冷启相关性裁剪。返回 (messages, decision)。

    decision（落 Memory Hub reconstruction 审计，可回放可解释）：
        {pruned: [unit_key], scores: {unit_key: score}, target: int,
         kept_chars: int, chars_before: int, reason: str}

    任何前置条件不满足（无 q_vec / 无 embedder / 体积已达标 / 无候选）→
    原样返回 + `decision["pruned"] == []`（零改写）。
    """
    t0 = time.perf_counter()
    decision: dict[str, Any] = {
        "pruned": [], "scores": {}, "target": target_chars,
        "kept_chars": 0, "chars_before": 0, "reason": "",
    }
    if not messages:
        return messages, decision

    total = sum(len(_text_of(m)) for m in messages)
    decision["chars_before"] = total
    decision["kept_chars"] = total
    if total <= target_chars:
        decision["reason"] = "under_target"
        return messages, decision
    if not q_vec or embed_passage is None:
        decision["reason"] = "no_query_vector"
        return messages, decision

    units = build_task_units(messages)
    if len(units) <= protect_recent:
        decision["reason"] = "too_few_units"
        return messages, decision

    # protected：open 单元 + 最近 K 个单元（system 消息天然不属于任何单元）
    # 🔴 `units[-0:]` 是整个列表（Python 切片陷阱）—— K=0 必须显式取空集，
    #    否则"不保护最近单元"会被读成"保护全部单元"，裁剪静默失效。
    protected_keys = {u.unit_key for u in units[-protect_recent:]} if protect_recent > 0 else set()
    protected_keys |= {u.unit_key for u in units if u.status == TaskUnitStatus.OPEN}

    candidates = [
        u for u in units
        if u.status == TaskUnitStatus.CLOSED
        and u.unit_key not in protected_keys
        and _unit_chars(u, messages) > min_prune_chars
    ]
    if not candidates:
        decision["reason"] = "no_candidates"
        return messages, decision

    # 单元 intent 向量：按 unit_key 缓存于会话状态——unit_key 内容派生、跨轮稳定，
    # 每轮通常只新增 1 个单元需要 embed；冷启长会话首次全量 embed 上限 20 个。
    cache = state.unit_vec_cache if state is not None else {}
    todo = [u for u in candidates if u.unit_key not in cache][:20]
    if todo:
        try:
            vecs = embed_passage([_intent_text(u, messages) for u in todo])
            for u, v in zip(todo, vecs, strict=False):
                cache[u.unit_key] = list(v)
        except Exception as e:  # noqa: BLE001 —— 裁剪失败绝不打死请求
            logger.warning("prune_embed_failed", error=str(e))
            decision["reason"] = "embed_failed"
            return messages, decision

    if (time.perf_counter() - t0) * 1000 > budget_ms:
        logger.warning("prune_over_budget",
                       elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
                       budget_ms=budget_ms)
        decision["reason"] = "over_budget"
        return messages, decision

    scores = {u.unit_key: _cosine(q_vec, cache.get(u.unit_key, [])) for u in candidates}
    decision["scores"] = {k: round(v, 4) for k, v in scores.items()}

    # 按 score 升序贪心移除（同分按 unit_index 升序 = 先丢最老的，保确定性）
    ordered = sorted(candidates, key=lambda u: (scores[u.unit_key], u.unit_index))
    pruned_keys: list[str] = []
    kept = total
    for u in ordered:
        if kept <= target_chars:
            break
        if scores[u.unit_key] >= keep_floor:
            continue                              # 高相关：超预算也保留
        kept -= _unit_chars(u, messages)
        pruned_keys.append(u.unit_key)

    if not pruned_keys:
        decision["reason"] = "nothing_prunable"
        return messages, decision

    # 整单元替换为一行占位（保 tool 配对：整个单元一起走，不会切散 call/result）
    drop: set[int] = set()
    placeholder_at: dict[int, str] = {}
    for u in units:
        if u.unit_key not in pruned_keys:
            continue
        idxs = unit_indices(u, messages)
        if not idxs:
            continue
        intent = _intent_text(u, messages).strip().replace("\n", " ")
        placeholder_at[idxs[0]] = _PLACEHOLDER.format(
            intent=intent[:_INTENT_PREVIEW_CHARS])
        drop.update(idxs[1:])

    out: list[dict] = []
    for i, msg in enumerate(messages):
        if i in placeholder_at:
            out.append({"role": "user", "content": placeholder_at[i]})
        elif i not in drop:
            out.append(msg)

    decision["pruned"] = pruned_keys
    decision["kept_chars"] = sum(len(_text_of(m)) for m in out)
    decision["reason"] = "pruned"
    logger.info("prune_cold_done", units_pruned=len(pruned_keys),
                chars_before=total, chars_after=decision["kept_chars"],
                target=target_chars,
                elapsed_ms=round((time.perf_counter() - t0) * 1000, 1))
    return out, decision
