"""模型健康状态 — 最小熔断（T4，RA4，routing-fix-plan-20260712）。

运行时状态（非决策规则），故放 proxy 侧不进 core；core 经 HealthCheck Protocol 消费。
触发 failover 类错误（5xx/超时/连接错/429）→ 标记 unhealthy 冷却 N 秒，
期间选池跳过该模型（跳空则保留原池，见 core `_filter_healthy`），
避免每个请求都对死模型付一次完整超时成本。成功调用即恢复。
"""

from __future__ import annotations

import time

import structlog

logger = structlog.get_logger()


class ModelHealth:
    """进程级模型健康表（实现 bladex_core.routing.HealthCheck Protocol）。"""

    def __init__(self, cooldown_s: float = 30.0) -> None:
        self._cooldown_s = cooldown_s
        # model → unhealthy 截止时刻（time.monotonic）
        self._unhealthy_until: dict[str, float] = {}

    def mark_unhealthy(self, model: str) -> None:
        """标记模型不健康（冷却期内选池跳过）。"""
        until = time.monotonic() + self._cooldown_s
        self._unhealthy_until[model] = until
        logger.warning(
            "model_health_unhealthy", model=model, cooldown_s=self._cooldown_s,
        )

    def mark_healthy(self, model: str) -> None:
        """标记模型恢复（成功调用后清除熔断）。"""
        if self._unhealthy_until.pop(model, None) is not None:
            logger.info("model_health_recovered", model=model)

    def is_healthy(self, model: str) -> bool:
        until = self._unhealthy_until.get(model)
        if until is None:
            return True
        if time.monotonic() >= until:
            # 冷却期满，惰性清除
            self._unhealthy_until.pop(model, None)
            return True
        return False

    def any_unhealthy(self) -> bool:
        """是否有任何模型当前处于熔断冷却中（T27 /ready 上游可达性缓存标记用）。

        惰性清过期项；仅报告"此刻是否有已知 outage"，供 readiness 探针做 informational
        信号（不 gate 就绪--单个模型熔断仍有 failover 候选可服务）。
        """
        now = time.monotonic()
        # 惰性清过期
        expired = [m for m, until in self._unhealthy_until.items() if until <= now]
        for m in expired:
            self._unhealthy_until.pop(m, None)
        return bool(self._unhealthy_until)

    def unhealthy_models(self) -> list[str]:
        """当前处于熔断冷却中的模型列表（Beta T9 /admin/status 消费）。

        惰性清过期项，与 any_unhealthy 同语义。
        """
        now = time.monotonic()
        expired = [m for m, until in self._unhealthy_until.items() if until <= now]
        for m in expired:
            self._unhealthy_until.pop(m, None)
        return sorted(self._unhealthy_until)
