"""批 Q3B · MQ-W4：积压没有趋势、没有主动信号 ⇒ B10「backlog 不得单调上升」无读数。

钉住：
  ① 窗口内上升 / 持平 / 下降三种趋势各一，外加"上升但中途回落"≠ 单调上升；
  ② 样本不足不出趋势（没读数不许显示成持平），/metrics 也不出该序列；
  ③ `bladex status` 那一行把 /metrics 的序列译成人话，缺序列时说 n/a；
  ④ 采样循环：进入单调上升时打一次 `backlog_rising`，停机置位即退出。
"""

from __future__ import annotations

import asyncio
import types

from bladex_proxy.cli.lifecycle import _format_backlog_trend
from bladex_proxy.metrics import (
    TREND_FALLING,
    TREND_FLAT,
    TREND_RISING,
    BacklogTrend,
    Metrics,
    backlog_sampler_loop,
)
from structlog.testing import capture_logs

T0 = 1_000_000.0


def _trend(series: list[int], *, stage: str = "index", step: float = 60.0) -> BacklogTrend:
    t = BacklogTrend(window_s=1800.0)
    for i, v in enumerate(series):
        t.record({stage: v}, now=T0 + i * step)
    return t


def _now(series: list[int], step: float = 60.0) -> float:
    return T0 + (len(series) - 1) * step


# ── ① 三种趋势 ───────────────────────────────────────────────────────────

def test_rising_monotonic() -> None:
    # Q3D 改口：原序列 [100,120,120,150,180] 只覆盖 4 分钟，按 MQ-W20 新判据（底线爬升 ≥ 15 分钟）
    # 是 rising 而非 monotonic；拉长到 15 分钟、中间留平台，语义（"一路涨、从不回落"）不变。
    s = [100, 120, 120, 150, 150, 160, 160, 170, 175, 175, 180, 180, 185, 190, 190, 200]
    st = _trend(s).stage("index", now=_now(s))
    assert st is not None
    assert (st["trend"], st["delta"], st["monotonic_rise"]) == (TREND_RISING, 100, True)
    assert st["samples"] == 16 and st["span_s"] == 900.0


def test_flat() -> None:
    s = [40, 40, 40, 40]
    st = _trend(s).stage("index", now=_now(s))
    assert (st["trend"], st["delta"], st["monotonic_rise"]) == (TREND_FLAT, 0, False)


def test_falling() -> None:
    s = [720, 600, 610, 300]
    st = _trend(s).stage("index", now=_now(s))
    assert (st["trend"], st["delta"], st["monotonic_rise"]) == (TREND_FALLING, -420, False)


def test_rising_with_a_dip_is_not_monotonic() -> None:
    # Q3D 改口（二轮）：原 [100,130,110,160] 只有 3 分钟，新判据下本来就不够持续，钉不住"回落"。
    # "回落"的语义随 MQ-W20 收窄为**回落到抹掉爬升**（消费方追上了 ⇒ 底线归位）：
    # 7 分钟涨到 121 → 一次清回 100 → 再 7 分钟涨回 121。原始序列 14 步、15 分钟、净增 21，
    # 但底线（此后再没低于过的水位）在第 8 分钟归位，之后只爬了 7 分钟 ⇒ 不是持续上升。
    # 浅回落、谷底在抬升的锯齿反而**是**持续上升，见 test_q3d_backlog_sustained 的锯齿用例。
    s = [100 + 3 * i for i in range(8)] + [100] + [100 + 3 * i for i in range(1, 8)]
    st = _trend(s).stage("index", now=_now(s))
    assert st["trend"] == TREND_RISING and st["delta"] == 21 and st["span_s"] == 900.0
    assert st["monotonic_rise"] is False


def test_window_drops_old_samples() -> None:
    # 窗口 30 分钟：早于窗口的高点不参与判读（否则"曾经很高"会把现在的上升读成下降）
    t = BacklogTrend(window_s=1800.0)
    t.record({"index": 9999}, now=T0)
    # Q3D 改口：原 3 个样本只覆盖 2 分钟，新判据下不够"持续"；拉长到 15 分钟，断言不变
    for i, v in enumerate(range(10, 41, 2)):
        t.record({"index": v}, now=T0 + 3600 + i * 60)
    st = t.stage("index", now=T0 + 3600 + 900)
    assert st["delta"] == 30 and st["monotonic_rise"] is True


# ── ② 没读数就是没读数 ────────────────────────────────────────────────────

def test_too_few_samples_is_no_reading() -> None:
    s = [10, 20]
    assert _trend(s).stage("index", now=_now(s)) is None


def test_none_values_are_skipped_not_zero() -> None:
    t = BacklogTrend(window_s=1800.0)
    for i, v in enumerate([5, None, None, 7]):
        t.record({"hub": v}, now=T0 + i * 60)
    assert t.stage("hub", now=T0 + 180) is None  # 只有 2 个真样本


def test_export_gauges_only_for_stages_with_reading() -> None:
    t = BacklogTrend(window_s=1800.0)
    # Q3D 改口：[1,2,3] 两分钟不再算持续上升；拉长到 15 分钟以保留 monotonic=1.0 的断言
    for i, v in enumerate(range(1, 17)):
        t.record({"index": v, "hub": None}, now=T0 + i * 60)
    m = Metrics()
    t.export_gauges(m, now=T0 + 900)
    body = m.render()
    assert 'bladex_backlog_monotonic_rise{stage="index"} 1.0' in body
    assert 'bladex_backlog_delta_turns{stage="index"} 15.0' in body
    assert 'stage="hub"' not in body


# ── ③ status 那一行 ───────────────────────────────────────────────────────

def _scraped(stage: str, delta: float, trend: float, mono: float, span: float = 1800.0) -> dict:
    return {
        f'bladex_backlog_delta_turns{{stage="{stage}"}}': delta,
        f'bladex_backlog_trend{{stage="{stage}"}}': trend,
        f'bladex_backlog_monotonic_rise{{stage="{stage}"}}': mono,
        f'bladex_backlog_window_span_seconds{{stage="{stage}"}}': span,
    }


def test_status_line_three_trends() -> None:
    line = _format_backlog_trend({**_scraped("hub", 0, 0, 0), **_scraped("index", 340, 1, 1)})
    # Q3D 改口：锯齿上行也算持续上升 ⇒ "never dropped" 不再成立，标签改 "sustained"
    assert line == "-> Hub +0 over 30m (flat) / -> Index +340 over 30m (rising, sustained)"
    line = _format_backlog_trend({**_scraped("hub", 5, 1, 0), **_scraped("index", -120, -1, 0)})
    assert line == "-> Hub +5 over 30m (rising) / -> Index -120 over 30m (falling)"


def test_status_line_no_series_is_na() -> None:
    line = _format_backlog_trend({})
    assert line == ("-> Hub n/a (not enough samples yet) / "
                    "-> Index n/a (not enough samples yet)")
    assert line.isascii()


# ── ④ 采样循环 ────────────────────────────────────────────────────────────

class _Pipe:
    def __init__(self) -> None:
        self.total = 0

    async def inspect(self) -> dict:
        self.total += 10
        return {"total": self.total, "orphan": 0}


class _Hub:
    def count(self) -> int:
        return 50


class _Index:
    def consumed_count(self) -> int:
        return 50


def test_sampler_warns_once_on_monotonic_rise_and_stops() -> None:
    async def _run() -> tuple[list, BacklogTrend, Metrics]:
        # Q3D：循环 10 ms 一采，到不了 10 分钟爬升下限 ⇒ 本例把时长下限置 0，只验"告警一次、可停"
        trend = BacklogTrend(window_s=3600.0, min_rise_span_s=0.0)
        m = Metrics()
        state = types.SimpleNamespace(pipeline=_Pipe(), hub=_Hub(), index=_Index(), metrics=m)
        stop = asyncio.Event()
        with capture_logs() as logs:
            task = asyncio.create_task(backlog_sampler_loop(state, stop, trend, interval_s=0.01))
            for _ in range(200):
                await asyncio.sleep(0.01)
                if len(trend._samples) >= 5:
                    break
            stop.set()
            await asyncio.wait_for(task, timeout=2)
        return logs, trend, m

    logs, trend, m = asyncio.run(_run())
    rising = [e for e in logs if e["event"] == "backlog_rising"]
    assert [e["stage"] for e in rising] == ["hub"]  # hub 一路涨；index 恒 0 = 持平，不报
    assert trend.snapshot()["index"]["trend"] == TREND_FLAT
    assert 'bladex_backlog_monotonic_rise{stage="hub"} 1.0' in m.render()
