"""DPL 落账本锚卡的两道闸（`task-ledger-gate-20260902.md`，ADR-0032 §4.5）。

病灶：归属的准入顺序是反的——先算内容相似（L2/L3/L4 匹配面、CONT 延续判定），
账本信息只在 ANCHOR 层用过一次。于是账本创建**之前**的、属于**别的**账本的会话，
靠主题词相似被 DPL 补进锚卡：ADR-0024 的开发过程进「评估 v5 架构」卡（07-29 的轮次
落进 08-28 建的账本），qwen3.8:27b 与 ornith-1.5 的对比测试进「Qwen3.8 Flash Next」卡
（Jason 人工判读，2026-09-02）。

修法是把顺序正过来：**账本标定 → 时间 → 内容聚焦**。两条闸都是因果不变量，
零参数——这是它与"再调一个相似度阈值"的根本区别。

本文件钉三组，每组带判别力对照：
① 两条闸各自拦住该拦的；
② 三处"缺数不当违规"（非锚卡 / 无标签 / 时间取不出）一律放行 —— 零回归；
③ 优先级：标签命中时不再过时间闸。
"""

from __future__ import annotations

import inspect

from bladex_core.ledger_runtime import (
    GATE_BEFORE_CREATED,
    GATE_OTHER_LEDGER,
    GATE_PASS,
    iso_ms,
    ledger_gate,
)

# 「评估 v5 架构开发进展」实测：账本 08-28 12:54 创建，被误吸的轮次 07-29 02:06。
CREATED = iso_ms("2026-08-28T12:54:00+00:00")
EARLY = iso_ms("2026-07-29T02:06:00+00:00")
LATE = iso_ms("2026-08-29T09:00:00+00:00")
LDG = "ldg-468debdba521"
OTHER = "ldg-c9ab898f14ec"


def _gate(*, turn_ledger_id: str = "", turn_ms: float = LATE,
          target_ledger_id: str = LDG, target_created_ms: float = CREATED) -> str:
    return ledger_gate(turn_ledger_id=turn_ledger_id, turn_ms=turn_ms,
                       target_ledger_id=target_ledger_id,
                       target_created_ms=target_created_ms)


# ── ① 两条闸各自拦住该拦的 ────────────────────────────────────────────────


def test_gate1_blocks_turns_older_than_the_ledger() -> None:
    """闸①：一件事的记录不可能早于这件事被开立。

    这条单独就拦下实测 281 条边中的全部（v5 105 / 泰山 99 / Qwen 64）。
    """
    assert _gate(turn_ms=EARLY) == GATE_BEFORE_CREATED


def test_gate2_blocks_turns_labelled_to_another_ledger() -> None:
    """闸②：这一轮明确属于别的账本 ⇒ 拒。最硬的一条，零歧义。"""
    assert _gate(turn_ledger_id=OTHER) == GATE_OTHER_LEDGER


def test_gate2_fires_even_when_the_time_is_fine() -> None:
    """判别力：闸②不是闸①的推论。

    时间完全合规（晚于创建）但标签指向别的账本 —— 闸①放行、闸②必须拦。
    没有这条，"标签"会退化成"时间的近似"，而它们测的是两件事。
    """
    assert _gate(turn_ledger_id=OTHER, turn_ms=LATE) == GATE_OTHER_LEDGER


def test_own_ledger_passes() -> None:
    assert _gate(turn_ledger_id=LDG) == GATE_PASS


def test_unlabelled_but_late_enough_passes() -> None:
    """无标签 + 晚于创建 ⇒ 放行。闸①只挡"早于"，不挡"晚于"。

    这正是本修法比"锚卡只收 ANCHOR 层"松的地方：账本创建后、窗口外的
    同一件事的延续仍可落到锚卡上。
    """
    assert _gate(turn_ms=LATE) == GATE_PASS


# ── ② 三处缺数不当违规（零回归） ─────────────────────────────────────────


def test_non_anchor_card_is_untouched() -> None:
    """🔴 目标不是锚卡 ⇒ 本函数完全不表态。

    普通内容卡的归属不受账本机制影响。没有这条，改动会波及全部 DPL 边，
    而实测只有 11 张锚卡需要管。
    """
    assert _gate(target_ledger_id="", turn_ms=EARLY) == GATE_PASS
    assert _gate(target_ledger_id="", turn_ledger_id=OTHER) == GATE_PASS


def test_missing_created_at_does_not_block() -> None:
    """账本缺 `created_at` ⇒ 闸①不表态。

    缺数不等于"创建于 1970"。反过来实现（把 0.0 当时刻比大小）会让任何
    缺字段的账本判掉全部历史边 —— ADR-0012 §3.6"重建不得让历史归属
    一次性失效"的同族形状。
    """
    assert _gate(turn_ms=EARLY, target_created_ms=0.0) == GATE_PASS


def test_unparsable_turn_ts_does_not_block() -> None:
    """turn key 取不出 stream ms ⇒ 闸①不表态（同上）。"""
    assert _gate(turn_ms=0.0) == GATE_PASS


# ── ③ 优先级：账本标定优先 ───────────────────────────────────────────────


def test_matching_label_overrides_the_time_gate() -> None:
    """标签命中 ⇒ 不再过闸①。**这是优先级，不是漏判。**

    轮次不可能标到一个还不存在的账本上；真出现了也是标签更权威。
    Jason 2026-09-02："我们要优先保证有账本的内容归类正确。"
    """
    assert _gate(turn_ledger_id=LDG, turn_ms=EARLY) == GATE_PASS


# ── iso_ms 的 0.0 语义 ───────────────────────────────────────────────────


def test_iso_ms_returns_zero_for_missing_and_malformed() -> None:
    assert iso_ms("") == 0.0
    assert iso_ms("   ") == 0.0
    assert iso_ms("not-a-time") == 0.0


def test_iso_ms_handles_z_suffix_and_naive() -> None:
    """`Z` 后缀与无时区串都要能解析 —— 账本 `created_at` 两种都落过盘。"""
    assert iso_ms("2026-08-28T12:54:00Z") == iso_ms("2026-08-28T12:54:00+00:00")
    assert iso_ms("2026-08-28T12:54:00") == iso_ms("2026-08-28T12:54:00+00:00")


def test_iso_ms_is_ordered() -> None:
    """判别力：0.0 不参与排序（缺数 < 一切 会让闸①对缺数账本全拦）。"""
    assert iso_ms("2026-07-29T02:06:00Z") < iso_ms("2026-08-28T12:54:00Z")


# ── 结构性护栏 ───────────────────────────────────────────────────────────


def test_signature_is_keyword_only_and_fixed() -> None:
    """全 kwonly、四个参数、**没有阈值参数**。

    两条闸是因果不变量不是相似度判断。一旦出现 `threshold` / `window_ms` /
    `tolerance` 这类参数，就说明有人把它当成可调的内容判据在用了 ——
    那正是这次要改掉的东西（MQ-L31：按分档"调"出来的修法是净损失）。
    """
    sig = inspect.signature(ledger_gate)
    assert set(sig.parameters) == {
        "turn_ledger_id", "turn_ms", "target_ledger_id", "target_created_ms"}
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY
               for p in sig.parameters.values()), "位置参数会让调用点顺序写反"
