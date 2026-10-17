"""BladeX 自己的注入面按预算裁剪（批 Z · Z2.3，MQ-A34 ②，2026-09-24）。

## 只裁自己的，从不碰 agent 正文

预算 = `context_window − 估算 agent prompt − 输出下限`（全部换成字符，换算率见
`BLADEX_CHARS_PER_TOKEN`）。没配 `context_window` 的模型 ⇒ `surface_budget_for` 返回
None ⇒ 全链路逐字零行为。被裁的只可能是 BladeX 加进请求的五块：切换候选段 / 账本块 /
账本族（工具 + 说明书账本段）/ 自我介绍 / 记忆族工具；**硬规则永不裁**（`NEVER_TRIMMED`），
agent 的消息与工具一字不动（ADR-0033 的装配是另一件事，不在这里）。

## 稳定层与动态层分开裁（与 Z2.4 首轮定格不冲突）

稳定层（工具 schema + 自我介绍）的裁剪**只在会话首轮按窗口决定一次**，之后随定格冻结
（`StablePlan` 由 `AgencyRuntime` 按 session 冻结）；每轮的剩余预算只用来裁动态层
（账本块及其候选段）。否则稳定层会随对话变长而变，定格被裁剪自己打破。

## 裁剪顺序写死在一处

`TRIM_ORDER` 是唯一的顺序表，每步带理由；两条硬约束：
① 工具与它的说明书一起装卸（MQ-L7 / MQ-L77 的病：说明书还在教 `bladex_ledger_switch`、
   工具列表里却没有它；或反过来账本块还在、切换工具被裁了）——账本族按族整体装卸，
   沿用 `render_system_notes(ledger=…)` 与 `inject_tools(ledger_allowed=…)` 的现有机制；
② 每次裁剪打一行 `agency_surface_trimmed`（裁了哪块、多少字符、预算与估算）。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import structlog
from bladex_core.flags import flag_number

logger = structlog.get_logger()

LAYER_STABLE = "stable"
LAYER_DYNAMIC = "dynamic"


@dataclass(frozen=True)
class TrimStep:
    name: str
    layer: str
    why: str


#: 裁剪顺序（先裁前面的）。**改顺序 = 改产品行为**，每一步的理由写在这里，不写在调用点。
TRIM_ORDER: tuple[TrimStep, ...] = (
    TrimStep("candidates", LAYER_DYNAMIC,
             "账本块里的切换候选段（相关性候选 + recency top-5）：只在换题时有用，每轮重算，"
             "模型仍可经 bladex_ledger_read / switch 按 id 取；单位价值最低，最先裁。"),
    TrimStep("ledger_block", LAYER_DYNAMIC,
             "整个账本块（正文 + 首步指令）：正文是 Hub 的投影，模型可经 bladex_ledger_read 取回；"
             "裁正文不裁工具——「看得见却切不走」（MQ-L7）的反面是允许的（有器无令）。"),
    TrimStep("ledger_family", LAYER_STABLE,
             "账本族整体 = 账本工具 schema + 说明书里的账本段：必须成对装卸（MQ-L77），"
             "裁它等于这个会话在这个窗口上不用账本；记忆族保留，模型仍能取记忆。"),
    TrimStep("about", LAYER_STABLE,
             "自我介绍剩余部分（AGENT.md / TOOLS.md 记忆段 / SKILLS.md）：说明书没了工具还在，"
             "是「有器无令」，安全；只在窗口连记忆工具都快放不下时才裁。"),
    TrimStep("memory_family", LAYER_STABLE,
             "记忆族工具 schema：BladeX 取记忆的唯一通道（08-30 拆族拍板「无条件注入」），"
             "放在最后——裁到这一步说明 agent 自己的 prompt 已几乎占满窗口。"),
)

#: 永不裁的块。硬规则在 `<bladex-memory>` 块里、每轮随热加载变化（Z1.2），
#: 既不是稳定层、也不进任何裁剪清单——守卫 `test_surface_budget_z2.py` 钉死。
NEVER_TRIMMED: tuple[str, ...] = ("hard_rules",)

#: 第一步 "candidates" 对应账本块的哪两个桶（`LedgerBlockBreakdown.chars` 的键）。
LEDGER_BLOCK_TRIM_BUCKETS: tuple[str, ...] = ("candidates", "recent")


def chars_per_token_default() -> float:
    """全局 chars/token（`BLADEX_CHARS_PER_TOKEN`，默认 2.5 = Z2.0 读数④ p10 一侧）。"""
    v = float(flag_number("BLADEX_CHARS_PER_TOKEN"))
    return v if v > 0 else 0.0


def output_floor_tokens() -> int:
    """给输出预留的下限（token，`BLADEX_CONTEXT_OUTPUT_FLOOR_TOKENS`）。"""
    return max(0, int(flag_number("BLADEX_CONTEXT_OUTPUT_FLOOR_TOKENS")))


def estimate_tools_chars(tools: list[dict] | None) -> int:
    """工具 schema 的字符估算（`json.dumps(ensure_ascii=False)`，与 Z2.0 读数同口径）。"""
    if not tools:
        return 0
    try:
        return sum(len(json.dumps(t, ensure_ascii=False)) for t in tools)
    except (TypeError, ValueError):
        return sum(len(str(t)) for t in tools)


@dataclass(frozen=True)
class SurfaceBudget:
    """这一轮留给 BladeX 注入面的预算（字符）。只在模型配了窗口时存在。"""

    context_window: int
    chars_per_token: float
    output_floor_tokens: int
    agent_chars: int         # agent 正文 + agent 工具 schema（估算，字符）
    hard_rules_chars: int    # `<bladex-memory>` 块（永不裁，但占预算）

    @property
    def available_chars(self) -> int:
        usable = (self.context_window - self.output_floor_tokens) * self.chars_per_token
        return int(math.floor(usable)) - self.agent_chars - self.hard_rules_chars

    def as_log(self) -> dict[str, Any]:
        return {"context_window": self.context_window, "chars_per_token": self.chars_per_token,
                "output_floor_tokens": self.output_floor_tokens, "agent_chars": self.agent_chars,
                "hard_rules_chars": self.hard_rules_chars, "budget_chars": self.available_chars}


def surface_budget_for(*, context_window: int, agent_chars: int, hard_rules_chars: int,
                       chars_per_token: float | None = None,
                       floor_tokens: int | None = None) -> SurfaceBudget | None:
    """窗口未配（≤0）或换算率无效 ⇒ None（= 不裁）。"""
    if not context_window or int(context_window) <= 0:
        return None
    cpt = chars_per_token if chars_per_token is not None else chars_per_token_default()
    if not cpt or cpt <= 0:
        return None
    return SurfaceBudget(context_window=int(context_window), chars_per_token=float(cpt),
                         output_floor_tokens=(output_floor_tokens() if floor_tokens is None
                                              else max(0, int(floor_tokens))),
                         agent_chars=max(0, int(agent_chars)),
                         hard_rules_chars=max(0, int(hard_rules_chars)))


@dataclass(frozen=True)
class StableSizes:
    """稳定层各块的字符数（做计划用；由 `AgencyRuntime.stable_sizes` 产）。"""

    memory_tools: int
    ledger_tools: int
    about_full: int        # 自我介绍含账本段
    about_no_ledger: int   # 自我介绍剥掉账本段


@dataclass(frozen=True)
class StablePlan:
    """稳定层的装卸计划：会话首轮算一次，之后冻结（Z2.4）。全 True = 不裁。"""

    keep_memory_family: bool = True
    keep_ledger_family: bool = True
    keep_about: bool = True
    trimmed: tuple[str, ...] = ()
    stable_chars: int = 0
    budget_chars: int | None = None

    @property
    def is_full(self) -> bool:
        return self.keep_memory_family and self.keep_ledger_family and self.keep_about


FULL_PLAN = StablePlan()


def plan_stable(budget: SurfaceBudget | None, sizes: StableSizes) -> StablePlan:
    """按 `TRIM_ORDER` 里的稳定层步骤贪心裁，直到稳定层放得进预算。

    动态层的步骤（候选段 / 账本块）排在稳定层之前，所以到这里时它们已经算作可裁的——
    稳定层只在"连动态层全裁掉也放不下"时才动。
    """
    if budget is None:
        return FULL_PLAN
    keep = {"memory_family": True, "ledger_family": True, "about": True}

    def _total() -> int:
        t = 0
        if keep["memory_family"]:
            t += sizes.memory_tools
        if keep["ledger_family"]:
            t += sizes.ledger_tools
        if keep["about"]:
            t += sizes.about_full if keep["ledger_family"] else sizes.about_no_ledger
        return t

    trimmed: list[str] = []
    for step in TRIM_ORDER:
        if step.layer != LAYER_STABLE:
            continue
        if _total() <= budget.available_chars:
            break
        keep[step.name] = False
        trimmed.append(step.name)
    return StablePlan(keep_memory_family=keep["memory_family"],
                      keep_ledger_family=keep["ledger_family"],
                      keep_about=keep["about"],
                      trimmed=tuple(trimmed), stable_chars=_total(),
                      budget_chars=budget.available_chars)


def dynamic_trim_step(block_chars: int, bucket_chars: dict[str, int],
                      remaining_chars: int | None) -> str | None:
    """账本块（动态层）这一轮裁不裁：None = 原样；"candidates" = 去候选段；"ledger_block" = 整块不注。"""
    if remaining_chars is None:
        return None
    if block_chars <= remaining_chars:
        return None
    without = block_chars - sum(int(bucket_chars.get(b, 0) or 0) for b in LEDGER_BLOCK_TRIM_BUCKETS)
    if without < block_chars and without <= remaining_chars:
        return "candidates"
    return "ledger_block"


def log_trim(step: str, *, chars: int, budget: int | None, estimate: int,
             session: str = "", layer: str = "") -> None:
    """每次裁剪一行（Z2.3 硬约束②）。`estimate` = 裁之前该层估算字符。"""
    logger.info("agency_surface_trimmed", step=step, layer=layer or _layer_of(step),
                chars=int(chars), budget_chars=budget, estimate_chars=int(estimate),
                session=(session or "")[:24])


def _layer_of(step: str) -> str:
    for s in TRIM_ORDER:
        if s.name == step:
            return s.layer
    return ""
