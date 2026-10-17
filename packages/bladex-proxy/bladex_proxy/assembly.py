"""上下文装配（Context Assembly）—— ADR-0033 裁法（批 U · U1.4，2026-09-23）。

ADR-0033 取代了 ADR-0019 的 L1「每轮常开、按位置降解」与 L2「单元摘要」：

  - **何时动**：只在触发门打开的冷轮（`inject._assembly_trigger`：轮次数 > N ∧ 上一轮上游
    cache 命中为 0，§2.3）做一次复核（`assembly_review`，§2.4/§2.5），判定会话内复用（§2.7）；
    热轮原样重放冷轮冻结的判定，前缀逐字稳定。
  - **裁什么**：逐轮次单元独立判定（§2.6），与位置无关。被判 `trimmed` 的单元：user 原文与
    conclusion **保留**，tool 结果换成可见标记 `<bladex:evidence_trimmed tool=… chars=… unit=…>`；
    消息骨架与 tool_call_id 配对不动；open 单元永不裁。
  - **可复原**：`degrade_plan`（msg_index → `{tier: kept|trimmed, unit, source: embed|model|frozen}`）
    随 reconstruction 落 Memory Hub；`replay_trim(剥注入后的原文, degrade_plan)` 逐字复原
    "这一轮模型实际看到什么"（ADR-0025 U2 继承）。标记是原文的纯函数（名字 / 长度 / 单元键），
    不需要额外存正文。
  - **信息守恒**（ADR-0019 继承）：Memory Hub 总账存 agent 原文（入库走 original_messages，
    与装配无关）——裁掉的证据原文仍在 Hub。

删除（ADR-0033 §3 / §2.8，本批）：L1 `_degrade_evidence` / `_placeholder`（按位置降解）与它的三个旋钮
`BLADEX_ASSEMBLY_EVIDENCE_MIN_CHARS / _EVIDENCE_EXCERPT_CHARS / _KEEP_RECENT_TOOL_RESULTS`；
细粒度开关 `BLADEX_ASSEMBLY_ENABLED` 与旧名 `BLADEX_CAP_ENABLED`（与模块开关 `BLADEX_MODULE_ASSEMBLY`
重复）；E4.3 冷启相关性裁剪（`pruning.prune_cold_context`，整单元换占位——并入复核路径）。
L2 / B 层已于 2026-09-18 Q5 删除。
🔴 `SUMMARY_OPEN/CLOSE` 保留：`inject.py` 仍用它剥离历史请求里残留的摘要块（读侧防御，不是生产者）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog
from bladex_core.task_unit import TaskUnit, build_task_units, unit_indices

from bladex_proxy.modules import module_enabled

logger = structlog.get_logger()

# 摘要块标记 + 体积估算：唯一定义处（2026-09-03 S3：`context_manager.py`（CAP，ADR-0016）
# 删除，三符号迁入；`inject.py` 原有的那份重复定义改为从这里 import——刚性原则 12
# "同一个值两个家 = 缺陷"）。L2 生产者已删（2026-09-18 Q5），标记只剩读侧剥离一个消费方。
SUMMARY_OPEN = "<bladex-context-summary>"
SUMMARY_CLOSE = "</bladex-context-summary>"

#: 被裁证据的可见标记（ADR-0033 §2.6；原则 13 同族：丢了什么要说出来）。
EVIDENCE_TRIMMED_OPEN = "<bladex:evidence_trimmed"
TIER_KEPT = "kept"
TIER_TRIMMED = "trimmed"


def _msg_chars(msg: dict) -> int:
    """单条消息的字符数估算（str 全长；list 只计 text part）。"""
    content = msg.get("content")
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(
            len(p.get("text", ""))
            for p in content
            if isinstance(p, dict) and isinstance(p.get("text"), str)
        )
    return 0


def estimate_context_chars(messages: list[dict]) -> int:
    """消息列表总字符数估算（assembly_done 的 chars_* 读数 + orchestration 同一把尺子）。"""
    return sum(_msg_chars(m) for m in messages)


def assembly_enabled_for(agent_id: str) -> bool:
    """装配是否对该 agent 开（2026-09-03 拍板 e，MQ-CA6；默认值真身 `flags.py`）。

    `BLADEX_ASSEMBLY_AGENTS` 逗号清单：`*` 全开；命中完整 agent_id **或其 base**
    （`hermes:default` 命中 `hermes`）即开；空清单 = 全关（2026-09-23 起公开默认，MQ-CA19）。
    每轮读 env（不冻结）——它是止血开关，改了要立刻生效。
    `agent_id` 为空 = 调用方没带身份（直调 `do_inject` 的测试/脚本），无对象可 gate
    ⇒ 开（生产端点恒带 identity.agent_id，unknown-* 桶按 base 不命中即关）。
    """
    from bladex_core.flags import flag_csv
    aid = (agent_id or "").strip()
    if not aid:
        return True
    allowed = flag_csv("BLADEX_ASSEMBLY_AGENTS")
    if "*" in allowed:
        return True
    return aid in allowed or aid.partition(":")[0] in allowed


@dataclass
class AssemblyConfig:
    """装配配置。ADR-0033 起只剩一个开关——模块开关 `BLADEX_MODULE_ASSEMBLY`（注册表单一真相源）；
    触发门 / 复核的旋钮在 `inject._assembly_trigger` 与 `assembly_review.ReviewConfig`。"""

    enabled: bool = True

    @classmethod
    def from_env(cls) -> AssemblyConfig:
        return cls(enabled=module_enabled("assembly"))


def evidence_marker(msg: dict, unit_key: str) -> str:
    """被裁 tool 结果的替换正文（ADR-0033 §2.6）——**原文的纯函数**：同一条消息同一单元
    每次逐字相同（保上游 prompt cache；`replay_trim` 据此逐字复原）。"""
    name = msg.get("name", "") or "tool"
    return (f"{EVIDENCE_TRIMMED_OPEN} tool={name} chars={_msg_chars(msg)} unit={unit_key}>\n"
            "(earlier tool output trimmed by BladeX as unrelated to the current task; "
            "the full text is kept in BladeX memory — re-run the tool if you need it)")


def _is_trimmed(msg: dict) -> bool:
    c = msg.get("content")
    return isinstance(c, str) and c.startswith(EVIDENCE_TRIMMED_OPEN)


@dataclass
class _Unit:
    """一个轮次单元的索引视图（相对传入的 messages 列表）。"""

    indices: list[int] = field(default_factory=list)
    closed: bool = False
    key: str = ""  # TaskUnit.unit_key（reconstruction / 判定的锚点）


class ContextAssembler:
    """按判定裁证据（ADR-0033 §2.6）。判定来自 `assembly_review`（冷轮）或冻结计划（热轮）。"""

    def __init__(self, config: AssemblyConfig | None = None) -> None:
        self._config = config or AssemblyConfig.from_env()

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    # ── 单元化 ──

    def _split_units(self, messages: list[dict]) -> list[_Unit]:
        """委托 `bladex_core.task_unit.build_task_units`（全仓唯一的轮次切分，ADR-0033 §2.2）。

        索引口径：相对**传入的 messages 列表**（装配对 `cleaned`——剥离注入块后——调用）。
        """
        return [
            _Unit(indices=unit_indices(u, messages), closed=u.closed,
                  key=getattr(u, "unit_key", ""))
            for u in build_task_units(messages)
        ]

    # ── 裁法 ──

    def trim(
        self,
        messages: list[dict],
        verdicts: dict[str, tuple[str, str]],
        units: list[TaskUnit] | None = None,
    ) -> tuple[list[dict], dict[int, dict[str, str]], int]:
        """按判定裁：`verdicts = {unit_key: (tier, source)}`，tier ∈ kept | trimmed。

        只改**已闭合**单元里 `role == "tool"` 消息的 content；user / assistant（含 conclusion 与
        tool_calls）/ system 原样，消息条数与顺序不变 ⇒ tool_call_id 配对不动。open 单元永不裁
        （即使判定表里有它的键）。返回 `(messages, degrade_plan, 被裁条数)`；
        `degrade_plan` 记录**判定表覆盖到的每条 tool 消息**的档位（kept 也记——复原与审计都要）。
        """
        plan: dict[int, dict[str, str]] = {}
        if not verdicts or not messages:
            return messages, plan, 0
        units = units if units is not None else build_task_units(messages)
        trim_at: dict[int, str] = {}
        for u in units:
            if not u.closed:
                continue                       # open 单元永不裁（§2.6）
            v = verdicts.get(u.unit_key)
            if v is None:
                continue
            tier, source = v
            for i in unit_indices(u, messages):
                if messages[i].get("role") != "tool":
                    continue
                plan[i] = {"tier": tier, "unit": u.unit_key, "source": source}
                if tier == TIER_TRIMMED and not _is_trimmed(messages[i]):
                    trim_at[i] = u.unit_key
        if not trim_at:
            return messages, plan, 0
        out = [({**m, "content": evidence_marker(m, trim_at[i])} if i in trim_at else m)
               for i, m in enumerate(messages)]
        return out, plan, len(trim_at)


def replay_trim(messages: list[dict], degrade_plan: dict[Any, Any]) -> list[dict]:
    """从 Hub 逐字复原装配结果（ADR-0033 §2.7 末段 / ADR-0025 U2 继承）。

    `messages` = 该轮 Hub 存的原文经 `inject.strip_previous_injection`（与装配时同一个输入）；
    `degrade_plan` = 该轮 `ReconstructionRecord.degrade_plan`。只认 ADR-0033 形态的 dict 档位；
    历史 str 档位（`excerpt` / `minimal`，L1 / B 层时代）的原文摘录没有落 Hub，**不可复原**，原样跳过。
    """
    out = list(messages)
    for k, v in (degrade_plan or {}).items():
        i = int(k)
        if not isinstance(v, dict) or v.get("tier") != TIER_TRIMMED or not (0 <= i < len(out)):
            continue
        if out[i].get("role") == "tool" and not _is_trimmed(out[i]):
            out[i] = {**out[i], "content": evidence_marker(out[i], str(v.get("unit", "")))}
    return out
