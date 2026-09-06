"""BladeX proxy - 记忆优先的 LLM proxy 中间件（自建 FastAPI + Router 转发层）。

权威架构见 项目 ADR：0008（定位 + 架构）/ 0009（存储）/ 0018（DPL 判定层）/ 0019（上下文装配）。

核心能力：
  - OpenAI 兼容 /v1/chat/completions + Anthropic /v1/messages 端点
  - 记忆注入：prefetch 相关性 top-k + MUST/NEVER 硬规则 + CAP/Assembly 上下文管理
  - Router 转发（`router_sdk.py` 是唯一对接上游模型 SDK 的网关）+ 自迭代流式捕获（拼完整回复）
  - 三层存储管线：Pipeline（Redis Streams）-> Memory Hub（RocksDB）-> Memory Index（LanceDB 向量 + 事实 + MatterGraph）
  - 身份分层解析 + 被动指纹（含 MoA auxiliary 识别）
  - 配置驱动的确定性路由流水线（agent -> 多模态过滤 -> 能力 -> failover）+ 可选 LLM 裁判
"""

__version__ = "0.1.0"
