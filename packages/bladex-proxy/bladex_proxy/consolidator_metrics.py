"""consolidator 侧的独立 metrics 端点（M4-2 / MS-8）。

## 为什么是**独立**端点而不是写进 P2 让 proxy 一起吐

2026-08-06 用户拍板：两个进程各自暴露，便于检查和维护。

理由站得住：漏斗写入侧（候选/蒸馏/判重/裁决/入库）在 consolidator，
读取侧（召回/注入/命中）在 proxy。**各归各**的话，"哪半边没数"一眼可见；
若让 consolidator 把计数中转进 P2 meta、由 proxy 统一渲染，
中转层本身就会成为一个可能坏掉且难以分辨的环节——
本仓库反复出现的"机制写完接线断"里，中转环节是重灾区。

代价是看全链要拉两个地址，由 `bladex status --traffic` 兜住。

## 形态

一个 stdlib `http.server` 线程，只服务 `GET /metrics`（Prometheus 文本）。
**默认关闭**（`BLADEX_CONSOLIDATOR_METRICS_PORT=0`）——consolidator 是后台进程，
默认不该多开一个监听口。绑定地址固定 `127.0.0.1`：这是本机观测面，
不做鉴权，所以绝不能听在 0.0.0.0（与 proxy 的 bind 安全同一条纪律）。
"""

from __future__ import annotations

import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import structlog

logger = structlog.get_logger()

_PORT_ENV = "BLADEX_CONSOLIDATOR_METRICS_PORT"
#: 本机观测面，不鉴权 → 只听回环。不提供改绑定地址的开关（那是脚枪）。
_BIND_HOST = "127.0.0.1"


def metrics_port() -> int:
    """读端口；0 / 未设 / 非法 → 0 = 不启动。"""
    raw = os.environ.get(_PORT_ENV)
    if raw is None or not raw.strip():
        return 0
    try:
        port = int(raw)
    except ValueError:
        logger.warning("consolidator_metrics_port_invalid", raw=raw)
        return 0
    return port if 0 < port < 65536 else 0


def _make_handler(metrics):  # noqa: ANN001, ANN202
    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path.rstrip("/") not in ("/metrics", ""):
                self.send_response(404)
                self.end_headers()
                return
            body = metrics.render().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args) -> None:  # noqa: ANN002
            # 默认实现往 stderr 打 Apache 风格日志，会把 consolidator 的
            # structlog 输出冲得七零八落。抓取是每 15s 一次的高频操作，静默即可。
            return

    return _Handler


def start_metrics_server(metrics, port: int | None = None):  # noqa: ANN001, ANN201
    """起一个后台 metrics 线程；返回 server（None = 未启动）。

    失败只告警不抛：观测面起不来绝不能拖垮提炼管线
    （端口被占用是最常见的情形——上一个 consolidator 没退干净）。
    """
    port = metrics_port() if port is None else port
    if port <= 0:
        logger.info("consolidator_metrics_disabled",
                    hint=f"set {_PORT_ENV}=<port> to expose the write-side funnel")
        return None
    try:
        server = ThreadingHTTPServer((_BIND_HOST, port), _make_handler(metrics))
    except Exception as e:  # noqa: BLE001
        logger.warning("consolidator_metrics_bind_failed", port=port, error=str(e))
        return None
    thread = threading.Thread(target=server.serve_forever, daemon=True,
                              name="consolidator-metrics")
    thread.start()
    logger.info("consolidator_metrics_serving", url=f"http://{_BIND_HOST}:{port}/metrics")
    return server
