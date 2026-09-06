"""BladeX 记忆编排核心（agent 中立）。

v5 架构（ADR-0032；基座 ADR-0008/0009）：
  - fact: Fact + HardRule Pydantic 模型
  - consolidation_proxy: ProxyConsolidator（从 Memory Hub Turns 提炼 Fact）
  - attribution / matter / routing / ledger / flash_tree：归属、Matter、路由、账本、Flash
  （2026-09-03 S1/S2：v3 的 prefetch / store / semantic / consolidation / compress_bridge
   已删除——主动注入检索路径退役，记忆经 `bladex_memory_search` 工具面按需取。）

core 不依赖 proxy/agent（依赖方向铁律）。
"""

from bladex_core.attribution import (
    UNASSIGNED_MATTER_ID,
    AttributionDecision,
    AttributionPipeline,
    AttributionSource,
    ExplicitSignalDetector,
    ExplicitSignalMatch,
)
from bladex_core.fact import Fact, HardRule
from bladex_core.matter import (
    EdgeProvenance,
    EdgeTargetType,
    Matter,
    MatterAggregateView,
    MatterEdge,
    MatterOrigin,
    MatterStatus,
)
from bladex_core.routing import (
    AgentStrategyData,
    FilterStrategyData,
    Judge,
    JudgeResult,
    MemoryAwareRouter,
    ModelCandidate,
    MultimodalStrategyData,
    NoCapableCandidateError,
    RouteDecision,
    RouteSource,
    RoutingError,
)

__all__ = [
    "AttributionDecision",
    "AttributionPipeline",
    "AttributionSource",
    "ExplicitSignalDetector",
    "ExplicitSignalMatch",
    "EdgeProvenance",
    "EdgeTargetType",
    "Fact",
    "HardRule",
    "Matter",
    "MatterAggregateView",
    "MatterEdge",
    "MatterOrigin",
    "MatterStatus",
    "AgentStrategyData",
    "FilterStrategyData",
    "Judge",
    "JudgeResult",
    "MemoryAwareRouter",
    "ModelCandidate",
    "MultimodalStrategyData",
    "NoCapableCandidateError",
    "RouteDecision",
    "RouteSource",
    "RoutingError",
    "UNASSIGNED_MATTER_ID",
]
