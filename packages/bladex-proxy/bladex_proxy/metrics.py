"""ADR-0021 T7 metrics - 轻量 Prometheus 文本采集器（不引重依赖，手写 exposition）。

线程安全（FastAPI async 并发）。指标维度覆盖：
  - bladex_requests_total{agent,sensitivity}        请求计数
  - bladex_request_latency_ms（histogram p50/p95/p99）延迟分位
  - bladex_route_source_total{source}               路由 source 分布
  - bladex_pipeline_queue_depth（gauge）                  Pipeline 队列水位
  - bladex_inject_facts_filtered_total              注入过滤命中数（exposure 守卫）

histogram 用有界 ring buffer（默认 2048 样本）+ render 时排序取分位--v1 轻量，
单机规模够用；大规模可换 prometheus_client。
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque


class Metrics:
    """进程级指标采集器（thread-safe）。"""

    def __init__(self, hist_size: int = 2048) -> None:
        self._lock = threading.Lock()
        self._hist_size = hist_size
        # counter: {(metric, label_str): value}
        self._counters: dict[tuple[str, str], float] = defaultdict(float)
        # histogram: {(metric, label_str): deque[float]}
        self._hists: dict[tuple[str, str], deque] = defaultdict(
            lambda: deque(maxlen=self._hist_size)
        )
        # gauge: {(metric, label_str): value}
        self._gauges: dict[tuple[str, str], float] = defaultdict(float)

    @staticmethod
    def _labels_str(labels: dict[str, str] | None) -> str:
        if not labels:
            return ""
        # Prometheus label 值转义（", \, \n）
        parts = []
        for k in sorted(labels):
            v = str(labels[k]).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
            parts.append(f'{k}="{v}"')
        return ",".join(parts)

    def inc(self, metric: str, value: float = 1.0, labels: dict[str, str] | None = None) -> None:
        key = (metric, self._labels_str(labels))
        with self._lock:
            self._counters[key] += value

    def observe(self, metric: str, value: float, labels: dict[str, str] | None = None) -> None:
        key = (metric, self._labels_str(labels))
        with self._lock:
            self._hists[key].append(float(value))

    def set_gauge(self, metric: str, value: float, labels: dict[str, str] | None = None) -> None:
        key = (metric, self._labels_str(labels))
        with self._lock:
            self._gauges[key] = float(value)

    @staticmethod
    def _percentile(sorted_vals: list[float], p: float) -> float:
        if not sorted_vals:
            return 0.0
        idx = max(0, min(len(sorted_vals) - 1, int(len(sorted_vals) * p) - 1))
        return sorted_vals[idx]

    def render(self) -> str:
        """渲染 Prometheus 文本格式 exposition。"""
        lines: list[str] = []
        with self._lock:
            # counters
            seen_metric: set[str] = set()
            for (metric, label_str), val in sorted(self._counters.items()):
                if metric not in seen_metric:
                    lines.append(f"# TYPE {metric} counter")
                    seen_metric.add(metric)
                lpart = "{" + label_str + "}" if label_str else ""
                lines.append(f"{metric}{lpart} {val}")
            # gauges
            seen_metric.clear()
            for (metric, label_str), val in sorted(self._gauges.items()):
                if metric not in seen_metric:
                    lines.append(f"# TYPE {metric} gauge")
                    seen_metric.add(metric)
                lpart = "{" + label_str + "}" if label_str else ""
                lines.append(f"{metric}{lpart} {val}")
            # histograms -> count + sum + p50/p95/p99
            seen_metric.clear()
            for (metric, label_str), vals in sorted(self._hists.items()):
                if metric not in seen_metric:
                    lines.append(f"# TYPE {metric} histogram")
                    seen_metric.add(metric)
                sorted_vals = sorted(vals)
                count = len(sorted_vals)
                total = sum(sorted_vals)
                # 用拼接避免 f-string 里字面量 } 需转义的混乱
                qbase = ("{" + label_str + ",") if label_str else "{"
                lbase = ("{" + label_str + "}") if label_str else ""
                for q, p in (("0.5", 0.5), ("0.95", 0.95), ("0.99", 0.99)):
                    lines.append(
                        metric + qbase + 'quantile="' + q + '"} '
                        + str(self._percentile(sorted_vals, p))
                    )
                lines.append(f"{metric}_count{lbase} {count}")
                lines.append(f"{metric}_sum{lbase} {round(total, 2)}")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _parse_labels(label_str: str) -> dict[str, str]:
        """把内部 label_str（k="v",...）解析回 dict（snapshot 用，值不含未转义引号）。"""
        labels: dict[str, str] = {}
        if not label_str:
            return labels
        for part in label_str.split('",'):
            if "=" not in part:
                continue
            k, _, v = part.partition("=")
            v = v.strip('"').replace('\\"', '"').replace("\\\\", "\\").replace("\\n", "\n")
            labels[k.strip()] = v
        return labels

    def snapshot(self) -> dict:
        """结构化快照（Beta T9 /admin/status 消费——/metrics 的 JSON 版）。

        返回：
          {"counters":   {metric: [(labels_dict, value), ...]},
           "gauges":     {metric: [(labels_dict, value), ...]},
           "histograms": {metric: [(labels_dict, {"count","p50","p95","p99"}), ...]}}
        """
        with self._lock:
            counters: dict[str, list] = {}
            for (metric, label_str), val in sorted(self._counters.items()):
                counters.setdefault(metric, []).append((self._parse_labels(label_str), val))
            gauges: dict[str, list] = {}
            for (metric, label_str), val in sorted(self._gauges.items()):
                gauges.setdefault(metric, []).append((self._parse_labels(label_str), val))
            hists: dict[str, list] = {}
            for (metric, label_str), vals in sorted(self._hists.items()):
                sorted_vals = sorted(vals)
                hists.setdefault(metric, []).append((self._parse_labels(label_str), {
                    "count": len(sorted_vals),
                    "p50": self._percentile(sorted_vals, 0.5),
                    "p95": self._percentile(sorted_vals, 0.95),
                    "p99": self._percentile(sorted_vals, 0.99),
                }))
        return {"counters": counters, "gauges": gauges, "histograms": hists}


# ── 静默降级的滚动窗口计数（ADR-0027 §3.3 契约 C）──────────────────────
#
# 为什么不用上面的 Metrics counter：counter 是自进程启动以来的累计值，
# 回答不了"**现在**还在不在降级"——一个跑了三周的 proxy 攒下 4000 次
# inject_timeout，和刚刚这一小时超时 40 次，在累计值上长得一样。而降级面
# 要回答的恰恰是后者。故单独存事件时间戳、按窗口数。
#
# 有界 deque（默认 4096/事件）：满了丢最旧的。后果是"1h 内超过 4096 次"会
# 少报——那种量级本身就已经是红色，少报不影响结论。

INJECT_TIMEOUT = "inject_timeout"
INJECT_FAILED = "inject_failed"
ENQUEUE_SKIPPED = "enqueue_skipped"
INDEX_UNAVAILABLE = "index_unavailable"

#: snapshot 恒定输出的键（即使从未发生也给 0——消费方拿到稳定 schema，
#: "没有这个字段"与"这个字段是 0"在 UI 上是两件事）
KNOWN_DEGRADATIONS = (INJECT_TIMEOUT, INJECT_FAILED, ENQUEUE_SKIPPED, INDEX_UNAVAILABLE)

_DEFAULT_WINDOW_S = 3600.0


class DegradationLog:
    """降级事件的滚动窗口记录器（进程级，thread-safe）。

    只记时间戳、不记内容——内容在结构化日志里，这里只回答"最近 N 秒发生了几次"。
    """

    def __init__(self, maxlen: int = 4096) -> None:
        self._lock = threading.Lock()
        self._maxlen = maxlen
        self._events: dict[str, deque] = defaultdict(lambda: deque(maxlen=maxlen))

    def record(self, name: str, *, now: float | None = None) -> None:
        ts = time.time() if now is None else now
        with self._lock:
            self._events[name].append(ts)

    def count(self, name: str, window_s: float = _DEFAULT_WINDOW_S,
              *, now: float | None = None) -> int:
        cutoff = (time.time() if now is None else now) - window_s
        with self._lock:
            events = self._events.get(name)
            if not events:
                return 0
            # 时间戳单调递增 -> 从右往左数到越界即可停
            n = 0
            for ts in reversed(events):
                if ts < cutoff:
                    break
                n += 1
            return n

    def snapshot(self, window_s: float = _DEFAULT_WINDOW_S,
                 *, now: float | None = None) -> dict[str, int | float]:
        """{"window_s": 3600, "<event>_1h": n, ...}

        键名后缀固定 `_1h` 是与 CLI/dashboard 的既有契约（`cli._print_degradation_banner`），
        window_s 非默认值时后缀仍是 `_1h`——窗口真值以 `window_s` 字段为准。
        """
        out: dict[str, int | float] = {"window_s": window_s}
        names = set(KNOWN_DEGRADATIONS)
        with self._lock:
            names.update(self._events.keys())
        for name in sorted(names):
            out[f"{name}_1h"] = self.count(name, window_s, now=now)
        return out

    def clear(self) -> None:
        with self._lock:
            self._events.clear()


#: 进程级单例。热路径打点方（inject / 入库）拿不到 app.state，用模块级句柄。
DEGRADATION = DegradationLog()


def record_degradation(name: str) -> None:
    """降级打点。

    打点本身绝不能成为新的失败点——调用点已经在降级路径上了，
    这里再抛异常就是"错误处理里的错误"。
    """
    try:
        DEGRADATION.record(name)
    except Exception:  # noqa: BLE001
        pass


# ── 积压趋势（MQ-W4，批 Q3B）──────────────────────────────────────────
#
# 为什么要趋势而不只是当前值：`bladex status` 的 `-> Index 720 waiting` 是一个点，
# 回答不了"它在追上还是在落后"。08-22 那次积压 20 小时才被人手敲 status 撞见，
# 而 Beta gate B10「三 agent 并行跑真实流量，backlog 不得单调上升」要的正是一条
# 线——**单调**只能在一个窗口的多个样本上判，拿两次手敲的 status 判不出来。
#
# 采样放在 proxy 进程里（它同时持有 pipeline / Hub / Index 三个句柄），
# 后台每 BACKLOG_SAMPLE_INTERVAL_S 一次；判读只看最近 BACKLOG_TREND_WINDOW_S。
# 样本不足 BACKLOG_MIN_SAMPLES 时**不出趋势**（"没读数"不许显示成"持平"——原则 13 层 4）。

BACKLOG_SAMPLE_INTERVAL_S = 60.0
BACKLOG_TREND_WINDOW_S = 1800.0
BACKLOG_MIN_SAMPLES = 3
# 「单调上升」要的是**持续积压**，不是一两轮正常排队（MQ-W20，批 Q3D）。
# 旧判据只要求 delta > 0 ∧ 从不下降 ⇒ 一串 0 之后单个样本变 1 就成立：
# `proxy-20261001-145513.log` 10-07～10-09 五次 `backlog_rising` 全是 current=1|2，
# 1–9 分钟内自行清零。而「从不下降」又放过了真正的消费跟不上——锯齿上行
# （进得多、批处理消化一点、又进更多）每两步就回落一次，旧判据永远不报。
# ⇒ 改看**底线**（后缀最小值 = 此后再没低于过的水位）：锯齿的谷底在抬升 = 底线在涨；
# 回落到把爬升抹掉（消费方追上了）= 底线归位。底线上的两条下限：
#   步数 ≥ 3：live 病例 delta ≤ 2 ⇒ 至多 2 步；持续积压在真实流量下每进一轮就多一步。
#   时长 ≥ 900 s：consolidator 单次 pass 耗时 p99 = 610 s（同期 consolidator 日志 287 次，
#                 p90 128 s；max 1056 s 为上游超时长尾；`scripts/probe_backlog_trend_replay.py` 复算）——一次 pass 期间积压照涨、pass 结束
#                 一次清掉，这段爬升是排队不是积压；底线连涨超过 p99 pass 才算消费跟不上。
#                 （live 病例最长 9 分钟清零，也在其内。）
BACKLOG_RISE_MIN_STEPS = 3
BACKLOG_RISE_MIN_SPAN_S = 900.0
BACKLOG_STAGES = ("hub", "index")

TREND_RISING = "rising"
TREND_FLAT = "flat"
TREND_FALLING = "falling"
_TREND_CODE = {TREND_RISING: 1.0, TREND_FLAT: 0.0, TREND_FALLING: -1.0}


class BacklogTrend:
    """两段积压（→ Hub / → Index）在滚动窗口内的趋势（thread-safe）。

    判读规则（B10 读数方法同此，全文见 `report-batch-Q3D-20261013.md` §7——二轮修订，取代同稿 §4；
    Q3B 的旧规则见 `report-batch-Q3B-20260929.md` §1，已被本规则取代）：
      delta      = 窗口内最后一个样本 − 第一个样本
      trend      = rising / flat / falling，按 delta 的符号
      floor[i]   = min(样本 i … 最后一个样本)：此后再没低于过的水位（底线）
      up_steps   = 底线上升的次数
      rise_span  = 底线最后一次上升的时刻 − 第一次上升之前那个样本的时刻
      monotonic  = delta > 0 ∧ up_steps ≥ BACKLOG_RISE_MIN_STEPS
                   ∧ rise_span ≥ BACKLOG_RISE_MIN_SPAN_S（B10 的红线：持续上升）
    从不下降的序列，底线 = 序列本身。一两轮的正常排队（0…0,1 / 0…0,1,2）是 rising 但不是 monotonic；
    锯齿上行（谷底逐次抬升）是 monotonic；回落到抹掉爬升（底线归位）不是。
    """

    def __init__(self, window_s: float = BACKLOG_TREND_WINDOW_S,
                 min_samples: int = BACKLOG_MIN_SAMPLES,
                 min_rise_steps: int = BACKLOG_RISE_MIN_STEPS,
                 min_rise_span_s: float = BACKLOG_RISE_MIN_SPAN_S) -> None:
        self._lock = threading.Lock()
        self.window_s = window_s
        self.min_samples = min_samples
        self.min_rise_steps = min_rise_steps
        self.min_rise_span_s = min_rise_span_s
        # 样本上限按"窗口 / 最小合理间隔"估，多留一倍余量；满了丢最旧的
        self._samples: deque[tuple[float, dict[str, int | None]]] = deque(maxlen=4096)

    def record(self, values: dict[str, int | None], *, now: float | None = None) -> None:
        ts = time.time() if now is None else now
        with self._lock:
            self._samples.append((ts, dict(values)))

    def stage(self, name: str, *, now: float | None = None) -> dict | None:
        """一段的趋势读数；样本不足返回 None（不是"持平"）。"""
        cutoff = (time.time() if now is None else now) - self.window_s
        with self._lock:
            points = [(ts, v[name]) for ts, v in self._samples
                      if ts >= cutoff and isinstance(v.get(name), int)]
        if len(points) < self.min_samples:
            return None
        first, last = points[0][1], points[-1][1]
        delta = last - first
        trend = TREND_RISING if delta > 0 else TREND_FALLING if delta < 0 else TREND_FLAT
        floor: list[tuple[float, int]] = []
        low = last
        for ts, v in reversed(points):
            low = min(low, v)
            floor.append((ts, low))
        floor.reverse()
        ups = [(ta, tb) for (ta, a), (tb, b) in zip(floor, floor[1:], strict=False) if b > a]
        rise_span = ups[-1][1] - ups[0][0] if ups else 0.0
        return {
            "current": last,
            "delta": delta,
            "trend": trend,
            "monotonic_rise": (delta > 0 and len(ups) >= self.min_rise_steps
                               and rise_span >= self.min_rise_span_s),
            "up_steps": len(ups),
            "rise_span_s": round(rise_span, 1),
            "samples": len(points),
            "span_s": round(points[-1][0] - points[0][0], 1),
        }

    def snapshot(self, *, now: float | None = None) -> dict[str, dict | None]:
        return {name: self.stage(name, now=now) for name in BACKLOG_STAGES}

    def export_gauges(self, m: Metrics, *, now: float | None = None) -> None:
        """写进 /metrics。没读数的段不写（抓取方看到的是"无此序列"，不是 0）。"""
        for name, st in self.snapshot(now=now).items():
            if st is None:
                continue
            labels = {"stage": name}
            m.set_gauge("bladex_backlog_turns", float(st["current"]), labels)
            m.set_gauge("bladex_backlog_delta_turns", float(st["delta"]), labels)
            m.set_gauge("bladex_backlog_trend", _TREND_CODE[st["trend"]], labels)
            m.set_gauge("bladex_backlog_monotonic_rise", 1.0 if st["monotonic_rise"] else 0.0, labels)
            m.set_gauge("bladex_backlog_window_span_seconds", float(st["span_s"]), labels)
            m.set_gauge("bladex_backlog_samples", float(st["samples"]), labels)

    def clear(self) -> None:
        with self._lock:
            self._samples.clear()


#: 进程级单例（proxy 的采样循环写、/metrics 读）
BACKLOG_TREND = BacklogTrend()


async def read_backlog(state) -> dict[str, int | None]:  # noqa: ANN001 —— app.state
    """读两段积压的当前值；与 `bladex status` 的 Queue 行同一口径。

    → Hub = Redis stream 里还会被消费的条目（XLEN − 孤儿）；
    → Index = Hub 可见轮次 − Index 已消费轮次。读不到的段给 None。
    """
    import asyncio

    out: dict[str, int | None] = {"hub": None, "index": None}
    pipeline = getattr(state, "pipeline", None)
    if pipeline is not None:
        try:
            detail = await pipeline.inspect()
            out["hub"] = max(0, int(detail["total"]) - int(detail.get("orphan") or 0))
        except Exception:  # noqa: BLE001 —— 读不到就是 None，下一次再采
            pass
    hub, index = getattr(state, "hub", None), getattr(state, "index", None)
    if hub is not None and index is not None:
        try:
            turns = await asyncio.to_thread(hub.count)
            consumed = await asyncio.to_thread(index.consumed_count)
            out["index"] = max(0, turns - consumed)
        except Exception:  # noqa: BLE001
            pass
    return out


async def backlog_sampler_loop(state, stop, trend: BacklogTrend = BACKLOG_TREND,  # noqa: ANN001
                               interval_s: float = BACKLOG_SAMPLE_INTERVAL_S) -> None:
    """proxy 后台采样循环：采一次 → 写 /metrics → 持续单调上升时打一条可 grep 的 warning。

    `stop` 是 asyncio.Event：停机时置位、再 await 本协程——**不用 cancel**。
    cancel 打断的是 `await to_thread(...)`，线程里的 `hub.count()` 仍在跑，
    随后 lifespan 关 Hub ⇒ 在已关闭的 RocksDB 上迭代。等它自己采完这一次再走。

    warning 只在**进入**单调上升时打一次（`backlog_rising`），离开时打 `backlog_rising_cleared`
    ——这是 MQ-W4 要的"主动信号"：不用有人去敲 status，日志里就有。
    """
    import asyncio

    import structlog

    log = structlog.get_logger("bladex_proxy.metrics")
    rising: dict[str, bool] = {}
    # 先等一个间隔再采：启动那一刻不碰 Hub/Index（lifespan 里别的初始化、以及只活
    # 几百毫秒的测试 app 会在这时关库——批 Q3B gate 实测撞上过）。
    while True:
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval_s)
            return
        except TimeoutError:
            pass
        try:
            trend.record(await read_backlog(state))
            m = getattr(state, "metrics", None)
            if m is not None:
                trend.export_gauges(m)
            for name, st in trend.snapshot().items():
                now_rising = bool(st and st["monotonic_rise"])
                if now_rising and not rising.get(name):
                    log.warning("backlog_rising", stage=name, current=st["current"],
                                delta=st["delta"], span_s=st["span_s"], samples=st["samples"],
                                up_steps=st["up_steps"], rise_span_s=st["rise_span_s"])
                elif rising.get(name) and not now_rising:
                    log.info("backlog_rising_cleared", stage=name,
                             current=(st or {}).get("current"))
                rising[name] = now_rising
        except Exception as e:  # noqa: BLE001 —— 采样失败不许拖垮 proxy
            log.warning("backlog_sample_failed", error=str(e))
