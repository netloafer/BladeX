"""客户端 key 管理 — BladeX 发行 key 的校验（可开关）。

key 格式：bladex-<random>
配置方式：环境变量 BLADEX_CLIENT_KEYS 用 || 分隔，每条 "key||label"
  例：bladex-abc123||jason-macbook||bladex-def456||jason-codex
  奇数位是 key，偶数位是 label（给人看的备注）。

校验用恒定时间比较（防时序攻击）。
"""

from __future__ import annotations

import hmac
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger()


@dataclass
class ClientKey:
    """一条 BladeX 发行的客户端 key。"""

    key: str        # 实际 key 值，如 "bladex-abc123"
    label: str = "" # 备注，如 "jason-macbook"


@dataclass
class KeyStore:
    """客户端 key 集合。"""

    keys: list[ClientKey] = field(default_factory=list)

    @classmethod
    def parse(cls, raw: str) -> KeyStore:
        """解析配置字符串。

        格式：key1||label1||key2||label2||...
        奇数位=key，偶数位=label。label 可省略（只有 key 也行）。
        """
        if not raw.strip():
            return cls(keys=[])

        parts = [p.strip() for p in raw.split("||") if p.strip()]
        keys: list[ClientKey] = []
        i = 0
        while i < len(parts):
            key = parts[i]
            label = parts[i + 1] if i + 1 < len(parts) else ""
            keys.append(ClientKey(key=key, label=label))
            i += 2

        return cls(keys=keys)

    def verify(
        self,
        api_key: str | None,
        quiet: bool = False,
        endpoint: str | None = None,
    ) -> tuple[bool, str]:
        """校验 key 是否有效。

        返回 (通过, 原因)。
        用恒定时间比较防时序攻击。
        quiet=True：auth_ok 降 debug（高频内部端点如 /v1/embeddings——共享模型档下
        consolidator 每个嵌入批次一个请求，INFO 会刷屏；拒绝仍恒为 warning）。
        endpoint：调用的端点路径。**label 只说"哪把 key"，说不清"来干什么"**——
        个人模式下 consolidator 复用同一把 key 调 /v1/embeddings，日志里与 Hermes
        的对话请求长得一模一样（2026-08-09：满屏 `auth_ok label=hermes-default`
        无法分辨谁在调 embedding）。故把端点一并记上，让每条 auth 日志自解释。
        """
        if api_key is None:
            return False, "missing_key"

        for ck in self.keys:
            if hmac.compare_digest(api_key, ck.key):
                log = logger.debug if quiet else logger.info
                log("auth_ok", label=ck.label, endpoint=endpoint or "-")
                return True, ck.label

        logger.warning("auth_rejected",
                       key_prefix=api_key[:8] if len(api_key) > 8 else "short",
                       endpoint=endpoint or "-")
        return False, "invalid_key"

    @property
    def labels(self) -> list[str]:
        """所有 key 的 label（给人看的列表）。"""
        return [ck.label or ck.key[:12] for ck in self.keys]
