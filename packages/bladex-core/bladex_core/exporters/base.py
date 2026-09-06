"""Exporter 协议（Beta T13 / B3.2 地基）。

设计（beta-release-plan B3.2）：
  - worker（bladex-proxy 侧）负责：水位游标持久化、失败重试退避、
    目标不可达只降级不阻塞主管线、scope/exposure 过滤。
  - Exporter 只负责：把一批 upsert 写到目标（幂等，按 id 定位）+
    处理墓碑（删除目标侧对应物）。
  - 敏感联动（ADR-0021）：exporter 声明 exposure（local/private/public），
    worker 用 `sensitivity.exposure_allows(fact.exposure_ceiling, exposure)`
    过滤——越级 Fact 不出境。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class ExportBatch:
    """一次增量同步的数据批（worker 按游标收集，exporter 幂等 upsert）。"""

    facts: list[Any] = field(default_factory=list)      # bladex_core.fact.Fact
    matters: list[Any] = field(default_factory=list)    # bladex_core.matter.Matter
    edges: list[Any] = field(default_factory=list)      # bladex_core.matter.MatterEdge
    hard_rules: list[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not (self.facts or self.matters or self.edges or self.hard_rules)


@runtime_checkable
class Exporter(Protocol):
    """同步目标协议。实现方须幂等（按 id 定位重写）且自包含错误信息
    （抛出的异常被 worker 捕获计入退避，不会拖垮主管线）。"""

    # 目标名（游标/状态按此隔离）与声明的暴露等级（worker 据此过滤 Fact）
    name: str
    exposure: str

    def sync_incremental(self, batch: ExportBatch, since_cursor: str) -> None:
        """把一批新增/变更实体写到目标。since_cursor 仅供目标侧记录参考。"""
        ...

    def handle_tombstone(self, target_type: str, target_key: str) -> None:
        """处理一条墓碑：删除目标侧对应物（fact/matter/turn/edge）。"""
        ...
