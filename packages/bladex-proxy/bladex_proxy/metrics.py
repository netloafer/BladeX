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
