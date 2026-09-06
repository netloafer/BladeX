"""Beta T13（B3.2）：在线协同导出的 Exporter 接口层。

具体 connector（Obsidian / PostgreSQL）在 bladex-proxy 侧注册；
本包只定义协议与批类型（bladex-core 不依赖任何存储/网络组件）。
"""

from bladex_core.exporters.base import Exporter, ExportBatch

__all__ = ["ExportBatch", "Exporter"]
