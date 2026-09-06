"""三时钟生命周期判定（M3-2/M3-3 / 复核 5.9 用户拍板）。

## 拍板内容

生命周期以三个时间点为基础判据：

    创建钟 created_at   太久 → 可能无用
    更新钟 updated_at   太久不更新 → 该 Matter 不再活跃
    相关钟 last_hit_at  太久无命中 → 不再有相关性     ← M3-1 之前完全缺失

## 四点工程修正（5.9，全部落在本模块）

**1. 年龄钟按对象分层。** 对 Matter 语义强（"事儿"天然会过去，周级可完结）；
对 Fact 必须按 kind 区分 —— preference/lesson 不因老而无用
（Zep 的 "validity ≠ age"；G3 的教训正是均匀年龄衰减 66 天驱逐了用户偏好）。
所以本模块**只对 Matter 判状态**，Fact 侧只做确定性的 `valid_until` 到期（M3-3）。

**2. 更新钟 × 相关钟的 2×2 矩阵**给出 Matter 的完整出口逻辑（H1 的解）：

    |              | 近期有命中        | 久无命中                    |
    | 近期有更新    | active（正常）    | 在写从不被用 → 过度采集信号   |
    | 久无更新      | dormant（泰山类） | auto-close 候选（世界杯类）  |

**3. 时钟的输入卫生是前提。** 相关钟会被注入层健康度偏置——注入中断期间全库零命中，
此时若有"30 天无命中即退场"，就会发生**系统故障触发的大规模误杀**。
故设覆盖率护栏：`last_hit_at` 覆盖率不足时**只判不动**（返回决策但标 `observe_only`）。

**4. 只判不删。** 本模块是纯函数，产出**决策**，不碰存储。
dormant 的语义是"退出常驻注入、保留可召回"，不是删除。

kind 豁免表：preference / hard_rule / lesson 不参与任何时钟驱逐。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from bladex_core.flags import flag_number
from bladex_core.matter import Matter, MatterStatus

#: 🔴 kind 豁免表：这些条目**不因老而无用**，任何时钟都不驱逐它们。
#: G3 的教训：均匀年龄衰减 66 天驱逐用户偏好——一条"回复请简洁"不会因为
#: 说过很久就不再成立。Zep 的说法是 validity ≠ age。
CLOCK_EXEMPT_KINDS: frozenset[str] = frozenset({
    "preference", "hard_rule", "lesson",
})


@dataclass
class MatterClockDecision:
    """一张卡的三时钟判定结果（**决策，不是动作**）。"""

    matter_id: str
    quadrant: str            # active | write_only | dormant | close_candidate
    target_status: MatterStatus | None = None   # None = 不改状态
    reason: str = ""
    days_since_update: float = 0.0
    days_since_hit: float | None = None          # None = 从未命中
    observe_only: bool = False   # 护栏生效：只 log 不动


@dataclass
class ClockReport:
    """一次三时钟判定的整体结果。"""

    decisions: list[MatterClockDecision] = field(default_factory=list)
    coverage: float = 0.0        # last_hit_at 覆盖率
    observe_only: bool = False   # 全局护栏是否生效
    scanned: int = 0

    def to_apply(self) -> list[MatterClockDecision]:
        """真正要落库的那些（护栏生效时为空）。"""
        if self.observe_only:
            return []
        return [d for d in self.decisions if d.target_status is not None]


def _age_days(ts: datetime | None, now: datetime) -> float | None:
    if ts is None:
        return None
    t = ts if ts.tzinfo else ts.replace(tzinfo=UTC)
    return max(0.0, (now - t).total_seconds() / 86400.0)


def hit_coverage(matters: list[Matter]) -> float:
    """`last_hit_at` 的覆盖率——**这个钟有没有在被喂**的度量。

    相关钟接线（M3-1）之前的历史数据全是 None，M5 全量重建后同样归零。
    所以覆盖率低不代表"没人用"，而代表"我们还不知道谁被用过"——
    这两者在判定上必须区别对待，否则就是拿缺失当证据。
    """
    if not matters:
        return 0.0
    seen = sum(1 for m in matters if getattr(m, "last_hit_at", None) is not None)
    return seen / len(matters)


def decide_matter_clocks(
    matters: list[Matter], *, now: datetime | None = None,
    completion_signals: set[str] | None = None,
) -> ClockReport:
    """对一批 Matter 跑三时钟判定。纯函数、只判不动。

    `completion_signals`（MS-4②）：带 `completed` 语义信号的 matter_id 集合。
    有该信号时更新钟阈值**减半** —— 语义是加速证据，不是判据本身。
    5.9 明确：单信号不拍板，完结判定仍以三时钟为主、语义为辅，
    所以 completed 不直接改 status，只让它更快够到 dormant 门槛。
    """
    now = now or datetime.now(UTC)
    stale_update_d = flag_number("BLADEX_CLOCK_STALE_UPDATE_D")
    stale_hit_d = flag_number("BLADEX_CLOCK_STALE_HIT_D")
    provisional_max_d = flag_number("BLADEX_CLOCK_PROVISIONAL_MAX_D")
    min_coverage = flag_number("BLADEX_CLOCK_MIN_COVERAGE")
    signals = completion_signals or set()

    coverage = hit_coverage(matters)
    observe_only = coverage < min_coverage

    report = ClockReport(coverage=coverage, observe_only=observe_only,
                         scanned=len(matters))

    for m in matters:
        if m.status is MatterStatus.CLOSED:
            continue    # 已关闭的不再判（用户手动关的不该被自动改回来）

        upd_d = _age_days(m.updated_at, now) or 0.0
        hit_d = _age_days(getattr(m, "last_hit_at", None), now)

        # provisional 滞留出口（H3：60% 的 Matter 困在中间态且无出口）
        if m.status is MatterStatus.PROVISIONAL:
            created_d = _age_days(m.created_at, now) or 0.0
            if created_d > provisional_max_d:
                report.decisions.append(MatterClockDecision(
                    matter_id=m.matter_id, quadrant="dormant",
                    target_status=MatterStatus.DORMANT,
                    reason=f"provisional_stale:{created_d:.0f}d",
                    days_since_update=upd_d, days_since_hit=hit_d,
                    observe_only=observe_only))
            continue

        # MS-4②：有 completed 语义信号 → 更新钟阈值减半（加速，不是判据）
        eff_update_d = stale_update_d / 2 if m.matter_id in signals else stale_update_d

        stale_upd = upd_d > eff_update_d
        # 从未命中（hit_d is None）在护栏未解除前**不算"久无命中"** ——
        # 那是"还不知道"，不是"没人用"。拿缺失当证据正是 5.9 修正 3 要防的。
        stale_hit = hit_d is not None and hit_d > stale_hit_d

        if not stale_upd and not stale_hit:
            quadrant, target = "active", None
        elif not stale_upd and stale_hit:
            # 在写、从不被用 → **过度采集的审计信号**（写入侧问题的探测器）。
            # 刻意不改状态：这不是"该退场"，是"我们在采集没人要的东西"。
            quadrant, target = "write_only", None
        elif stale_upd and not stale_hit:
            quadrant, target = "dormant", MatterStatus.DORMANT
        else:
            # 双钟停摆 → auto-close **候选**。同样不自动关：
            # 关闭是用户的语义动作，自动关一张其实还在用的卡无法自愈。
            quadrant, target = "close_candidate", MatterStatus.DORMANT

        if quadrant == "active":
            continue
        report.decisions.append(MatterClockDecision(
            matter_id=m.matter_id, quadrant=quadrant, target_status=target,
            reason=f"upd={upd_d:.0f}d hit={'never' if hit_d is None else f'{hit_d:.0f}d'}"
                   + (" completion_signal" if m.matter_id in signals else ""),
            days_since_update=upd_d, days_since_hit=hit_d,
            observe_only=observe_only))
    return report


def is_expired(valid_until: datetime | None, now: datetime | None = None) -> bool:
    """M3-3：时效性记忆是否已到期。

    `valid_until` 由蒸馏 ⑦槽位产出（"等下周三"→ 具体日期）。
    到期 → 置 `t_invalid`（退出 current-only 召回），**数据不删**、as-of 可查。

    None（绝大多数条目）恒 False —— 没有到期点的记忆不因时间失效，
    这与 kind 豁免是同一条原则的两个面。
    """
    if valid_until is None:
        return False
    now = now or datetime.now(UTC)
    vu = valid_until if valid_until.tzinfo else valid_until.replace(tzinfo=UTC)
    return vu <= now


def exempt_from_clocks(item_kind: str) -> bool:
    """该条目类型是否豁免时钟驱逐。"""
    return (item_kind or "").strip().lower() in CLOCK_EXEMPT_KINDS


__all__ = [
    "CLOCK_EXEMPT_KINDS",
    "ClockReport",
    "MatterClockDecision",
    "decide_matter_clocks",
    "exempt_from_clocks",
    "hit_coverage",
    "is_expired",
    "timedelta",
]
