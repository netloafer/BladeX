"""批 Q3D · MQ-W20：「单调上升」被一两轮正常排队误触发。

旧判据 = delta > 0 ∧ 相邻样本从不下降 ⇒ 一串 0 之后单个样本变 1 即成立，
而锯齿上行（真正的消费跟不上）每两步回落一次，永远不报。
新判据看**底线**（后缀最小值）：delta > 0 ∧ 底线上升 ≥ BACKLOG_RISE_MIN_STEPS 步
∧ 底线爬升时长 ≥ BACKLOG_RISE_MIN_SPAN_S。

钉住：
  ① 五个 live 病例（`logs/proxy-20261001-145513.log` 10-07～10-09 的 `backlog_rising`）
     按日志字段（samples / span_s / current / delta）还原的序列，**逐个采样时刻**都不触发；
  ② 持续积压（匀速 / 阶梯 / 后半段起涨 / 慢滴 / 锯齿上行）必须触发；
  ③ 「涨到高位后不再降也不再涨」：窗口还盖着那段爬升时触发，滑过后清零转 flat；
  ④ 两条下限的恰好边界：恰好 3 步、恰好 900 s 触发，各差一点不触发。
"""

from __future__ import annotations

import pytest
from bladex_proxy.metrics import (
    BACKLOG_RISE_MIN_SPAN_S,
    BACKLOG_RISE_MIN_STEPS,
    TREND_FLAT,
    TREND_RISING,
    BacklogTrend,
)

T0 = 2_000_000.0


def _states(series: list[int], step: float = 60.0) -> list[dict | None]:
    """模拟采样循环：每进一个样本判一次（循环就是这么判的，不只看最后一刻）。"""
    t = BacklogTrend(window_s=1800.0)
    out = []
    for i, v in enumerate(series):
        now = T0 + i * step
        t.record({"index": v}, now=now)
        out.append(t.stage("index", now=now))
    return out


def _fired(series: list[int], step: float = 60.0) -> list[int]:
    return [i for i, st in enumerate(_states(series, step)) if st and st["monotonic_rise"]]


# ── ① live 病例：日志字段 ⇒ 序列 ─────────────────────────────────────────
#
# 旧规则触发 ⇒ 窗口内从不下降、first = current − delta = 0 ⇒ 序列 = 一串 0 + 末尾升到 current。
# delta = 2 的两例无法区分「一步到 2」还是「1 再到 2」，取步数更多的那种（对新判据更不利）。
# 10-09 13:39 那例 9 分钟后才清零：期间至少维持在 2（不降，否则旧规则当场就清了），补 8 个 2。
# 10-09 16:07 那例只有 13 个样本、span 753.7 s ⇒ 间隔 62.8 s（mac 睡眠把窗口切短）。

LIVE_CASES = {
    "10-07 11:26 hub  current=1 delta=1 samples=30": ([0] * 29 + [1], 60.0),
    "10-07 18:09 index current=1 delta=1 samples=30": ([0] * 29 + [1], 60.0),
    "10-09 10:58 index current=2 delta=2 samples=30": ([0] * 28 + [1, 2], 60.0),
    "10-09 13:39 index current=2 delta=2 samples=30 (+9 min)": ([0] * 28 + [1, 2] + [2] * 8, 60.0),
    "10-09 16:07 index current=1 delta=1 samples=13": ([0] * 12 + [1], 62.8),
}


@pytest.mark.parametrize("case", list(LIVE_CASES))
def test_live_normal_queueing_does_not_fire(case: str) -> None:
    series, step = LIVE_CASES[case]
    states = _states(series, step)
    # 序列与日志字段对得上（不是凭空造的）
    at_warn = states[-9] if "+9 min" in case else states[-1]
    assert at_warn["trend"] == TREND_RISING and at_warn["current"] in (1, 2)
    assert _fired(series, step) == []


# ── ② 持续积压必须触发 ────────────────────────────────────────────────────

def test_steady_climb_0_to_40_in_30min_fires() -> None:
    s = [round(i * 40 / 30) for i in range(31)]
    fired = _fired(s)
    assert fired and fired[0] <= 16  # 15 分钟左右就该报，不用等满窗口
    assert _states(s)[-1]["delta"] == 40


def test_staircase_with_plateaus_fires() -> None:
    s = [0] * 5 + [8] * 5 + [16] * 5 + [24] * 5 + [32] * 5 + [40] * 6
    st = _states(s)[-1]
    assert st["monotonic_rise"] is True and st["up_steps"] == 5


def test_late_onset_climb_fires() -> None:
    # 前 15 分钟为 0，后 15 分钟每分钟 +1：判据不要求上升覆盖整个窗口
    s = [0] * 15 + list(range(1, 16))
    fired = _fired(s)
    assert fired and s[fired[0]] >= BACKLOG_RISE_MIN_STEPS


def test_slow_trickle_never_drained_fires() -> None:
    # 每 3 分钟进一轮、消费方一条没吃：量小但持续 ⇒ 也是 B10 要抓的
    s = [i // 3 for i in range(31)]
    assert _states(s)[-1]["monotonic_rise"] is True


# ── ③ 高位平台 ───────────────────────────────────────────────────────────

def test_plateau_after_climb_fires_then_clears_to_flat() -> None:
    # 0 → 40（每分钟 +2，20 分钟）后一直停在 40，再看 45 分钟
    s = [2 * i for i in range(21)] + [40] * 45
    states = _states(s)
    assert states[20]["monotonic_rise"] is True        # 爬升结束那一刻
    assert states[35]["monotonic_rise"] is True        # 窗口 [5,35] 仍盖着 15 分钟爬升
    assert states[36]["monotonic_rise"] is False       # 窗口里只剩 14 分钟爬升 ⇒ 清零
    assert states[36]["trend"] == TREND_RISING         # 但净增仍在，B10 照 rising 写出 delta
    assert states[65]["trend"] == TREND_FLAT and states[65]["monotonic_rise"] is False


# ── 判据两条下限各自有效 ──────────────────────────────────────────────────

def test_fast_burst_is_not_sustained() -> None:
    # 3 分钟里涨了 3 步：步数够、时长不够（消费方一次 pass 就能追上的那种）
    s = [0] * 20 + [1, 2, 3]
    st = _states(s)[-1]
    assert st["up_steps"] == 3 and st["rise_span_s"] < BACKLOG_RISE_MIN_SPAN_S
    assert st["monotonic_rise"] is False


def test_two_steps_over_long_span_is_not_sustained() -> None:
    # 时长够、步数不够
    s = [0] * 5 + [1] * 16 + [2] * 5
    st = _states(s)[-1]
    assert st["up_steps"] == 2 and st["rise_span_s"] >= BACKLOG_RISE_MIN_SPAN_S
    assert st["monotonic_rise"] is False


def test_exact_boundary_three_steps_900s_fires() -> None:
    # 底线恰好 3 步：0→1（第 1 分钟）、1→2（第 8 分钟）、2→3（第 15 分钟）⇒ 爬升恰好 900 s
    s = [0, 1] + [1] * 6 + [2] + [2] * 6 + [3]
    st = _states(s)[-1]
    assert (st["up_steps"], st["rise_span_s"]) == (BACKLOG_RISE_MIN_STEPS, BACKLOG_RISE_MIN_SPAN_S)
    assert st["monotonic_rise"] is True
    # 各差一点：最后一步提前一分钟（840 s）/ 少一步
    assert _states(s[:-2] + [3, 3])[-1]["monotonic_rise"] is False
    assert _states([0] + [1] * 14 + [3])[-1]["monotonic_rise"] is False


# ── 锯齿：真正的消费跟不上 ────────────────────────────────────────────────

def _sawtooth(up: int, down: int, minutes: int = 30) -> list[int]:
    """奇数分钟 +up、偶数分钟 −down：60 s 采样看得见每一次回落。"""
    s = [0]
    for i in range(1, minutes + 1):
        s.append(s[-1] + (up if i % 2 else -down))
    return s


def test_sawtooth_rising_troughs_fires() -> None:
    # 每分钟进 5、每 2 分钟消化掉一批、样本上可见 −3：0,5,2,7,4,9,6… 谷底逐次抬升
    s = _sawtooth(5, 3)
    states = _states(s)
    assert all(st["trend"] == TREND_RISING for st in states[2:])
    fired = _fired(s)
    assert fired and fired[0] <= 16                    # 底线爬满 15 分钟就报
    assert states[-1]["monotonic_rise"] is True and states[-1]["delta"] == 30


def test_sawtooth_per_minute_net_never_dips() -> None:
    # 同一句话的另一种读法：+5/分钟、每 2 分钟 −3 落在同一个采样间隔里 ⇒ 样本是 0,5,7,12,14…
    # 60 s 采样上**看不到回落**，旧判据也会报；新判据照报
    s = [0]
    for i in range(1, 31):
        s.append(s[-1] + 5 - (3 if i % 2 == 0 else 0))
    assert _states(s)[-1]["monotonic_rise"] is True


def test_sawtooth_that_drains_to_baseline_is_not_sustained() -> None:
    # 涨 4 分钟、一次清零，循环：消费方每轮都追平 ⇒ 底线贴着 0
    s = ([0, 3, 6, 9, 12] * 6) + [0]
    assert _fired(s) == []


def test_climb_after_drain_in_a_falling_window_waits_for_trend() -> None:
    # 窗口开头还挂着旧高点 100，清到 0 后重新爬了 17 分钟：底线确实在涨，但窗口净值为负。
    # 要求 delta > 0，否则 status 会同时显示 "-83 (falling)" 和持续上升；
    # 窗口滑过旧高点后（delta 转正）照常触发。
    s = [100] + list(range(0, 18))
    st = _states(s)[-1]
    assert st["trend"] != TREND_RISING and st["up_steps"] >= BACKLOG_RISE_MIN_STEPS
    assert st["monotonic_rise"] is False
    assert _states(s + list(range(18, 32)))[-1]["monotonic_rise"] is True
