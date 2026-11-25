"""Embedding 模块客户端 adapter（`backend=ipc`；V-A6 T3）。

接口与 `RouterEmbedAdapter` / `ProxyEmbedAdapter` 一致
（`embed` / `embed_passage` / `embed_query` / `model_identity`），
所以对 `MemoryIndex`、prefetch、consolidator 全部透明。

# 🔴 两条纪律

1. **连不上 ⇒ 回落 local，不拒服务**（ADR-0017 Redis 懒恢复同一原则）：
   嵌入模块没起不该让 proxy 整个不可用。回落**必打 warning**（可 grep
   `embed_ipc_fallback_local`）——静默降级是本仓的高发病（注入面空转一个月）。
2. **优先级由角色决定**：proxy 侧 `hot`、consolidator 侧 `bulk`。这不是调用方
   随手传的参数，是**构造时钉死**的——让路机制的正确性依赖它不被写错。

同步接口（既有调用方全是同步的）里跑 asyncio：每次调用一个短命 event loop，
连接**不跨调用复用**——嵌入调用本身是几十毫秒到数秒，连接建立在 uds 上是
微秒级，复用带来的状态管理复杂度不值得（且能天然避开跨线程共享 socket 的坑）。
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from typing import Any

import structlog
from bladex_core.flags import flag_number

from bladex_proxy.embed_server import (
    ERROR_KIND_REPLY,
    PRIORITY_BULK,
    PRIORITY_HOT,
    STREAM_LIMIT,
)

logger = structlog.get_logger()

_HINT_LOCAL = "loading a local model in this process (memory doubles until restart)"


class EmbedServerReplyError(RuntimeError):
    """服务端在线，算了，但回包发不出去（序列化失败）。**不是**连接问题（MQ-W29）。"""


def _fallback_hint(e: Exception) -> str:
    """回落日志的 hint 按真实原因说：11-03 那次报「unreachable」，实际服务端活着、是回包失败。"""
    if isinstance(e, EmbedServerReplyError):
        return ("embed module is up but its reply failed; see embed_send_failed in "
                f"logs/embed-*.log; {_HINT_LOCAL}")
    return f"embed module unreachable; {_HINT_LOCAL}"


class IPCEmbedAdapter:
    """经 IPC 调 `bladex-embed` 模块；不可用时回落到本地 embedder。"""

    def __init__(self, transport: str, address: str, *, priority: str = PRIORITY_HOT,
                 timeout_s: float | None = None, chunk: int | None = None,
                 local_factory: Any = None) -> None:
        self._transport = transport
        self._address = address
        self._priority = PRIORITY_HOT if priority == PRIORITY_HOT else PRIORITY_BULK
        # 批 AA · MQ-A28：超时与单次请求块上限进 flags（此前写死 120 s、整批一次送）。
        # 显式传参仍优先（测试 / 特殊调用方）。
        self._timeout_s = (float(flag_number("BLADEX_EMBED_IPC_TIMEOUT_S"))
                           if timeout_s is None else timeout_s)
        self._chunk = max(1, int(flag_number("BLADEX_EMBED_IPC_CHUNK")
                                 if chunk is None else chunk))
        self._local_factory = local_factory      # () -> embedder（懒建，回落时才付钱）
        self._local: Any = None
        self._identity: str = ""
        self._fingerprint: str = ""
        self._degraded = False

    # ── 传输 ──
    async def _roundtrip(self, texts: list[str], input_type: str) -> list[list[float]]:
        if self._transport == "uds":
            reader, writer = await asyncio.open_unix_connection(
                self._address, limit=STREAM_LIMIT)
        else:
            host, _, port = self._address.rpartition(":")
            reader, writer = await asyncio.open_connection(
                host or "127.0.0.1", int(port), limit=STREAM_LIMIT)
        try:
            req = {"id": uuid.uuid4().hex[:12], "texts": texts,
                   "input_type": input_type, "priority": self._priority}
            writer.write((json.dumps(req, ensure_ascii=False) + "\n").encode())
            await writer.drain()
            line = await asyncio.wait_for(reader.readline(), timeout=self._timeout_s)
            if not line:
                raise ConnectionError(
                    "embed module closed the connection before replying (no reply frame; "
                    "the process likely exited or was stopped)")
            resp = json.loads(line.decode("utf-8", "replace"))
            if resp.get("error"):
                if resp.get("error_kind") == ERROR_KIND_REPLY:
                    raise EmbedServerReplyError(str(resp["error"]))
                raise RuntimeError(str(resp["error"]))
            self._identity = resp.get("model_identity") or self._identity
            self._fingerprint = resp.get("vector_fingerprint") or self._fingerprint
            return resp.get("vectors") or []
        finally:
            try:
                writer.close()
            except Exception:  # noqa: BLE001,S110
                pass

    @staticmethod
    def _run_sync(coro: Any) -> Any:
        """在同步接口里跑协程——**宿主可能已经有事件循环**。

        🔴 2026-08-25 live：首版直接 `asyncio.run()`，consolidator（普通脚本）没事，
        但 **proxy 是 uvicorn，本身跑在事件循环里** ⇒
        `asyncio.run() cannot be called from a running event loop` ⇒ 每次调用都
        异常 ⇒ 回落 local（日志 `embed_ipc_fallback_local`）——热路径根本没走 IPC，
        V-A6 对 proxy 侧等于没生效，而表面上一切"正常"（有回落兜底）。
        判据不是"我是谁"，是**当前线程有没有在跑的循环**：有就另起线程跑一个
        自己的循环（嵌入调用本就是阻塞等待，多一次线程切换可忽略）。
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)        # 无宿主循环：consolidator / 脚本 / 测试

        # 有宿主循环（proxy=uvicorn）：**另起线程跑自己的循环**。
        # 🔴 不能 `run_coroutine_threadsafe` 交回宿主循环再阻塞等结果——调用方
        # 往往就在循环线程里（async 处理器直接调同步 embedder），那样必死锁。
        # 另起循环在生产上没有问题：**服务端在独立进程**，跨循环连接本就正常。
        # （同进程内"服务端与客户端共用一个循环"只在测试里出现，见
        # `test_works_inside_running_event_loop` 的服务端独立线程写法。）
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(asyncio.run, coro).result()

    def _fallback(self) -> Any:
        if self._local is None:
            if self._local_factory is None:
                raise RuntimeError("embed module unreachable and no local fallback")
            self._local = self._local_factory()
        return self._local

    def _call(self, texts: list[str], input_type: str) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        # 批 AA · MQ-A28：按块往返。单次请求的耗时由块大小钉住（不再随批量线性逼近超时），
        # 块与块之间服务端可以先处理 hot 请求。某块失败 ⇒ 该块起回落（与原语义一致）。
        while not self._degraded and len(out) < len(texts):
            part = texts[len(out):len(out) + self._chunk]
            try:
                vecs = self._run_sync(self._roundtrip(part, input_type))
            except Exception as e:  # noqa: BLE001 —— 任何传输失败都回落
                self._degraded = True
                logger.warning("embed_ipc_fallback_local", error=str(e),
                               error_type=type(e).__name__,
                               transport=self._transport, address=self._address,
                               chunk=len(part), done=len(out), total=len(texts),
                               timeout_s=self._timeout_s,
                               hint=_fallback_hint(e))
                break
            out.extend(vecs)
            if len(vecs) != len(part):      # 服务端少回（旧语义：原样交给调用方）
                return out
        if not self._degraded:
            return out
        return out + self._local_embed(texts[len(out):], input_type)

    def _local_embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        emb = self._fallback()
        if input_type == "query" and hasattr(emb, "embed_query"):
            return emb.embed_query(texts)
        if hasattr(emb, "embed_passage"):
            return emb.embed_passage(texts)
        return emb.embed(texts)

    # ── 与既有 adapter 一致的接口 ──
    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._call(texts, "passage")

    def embed_passage(self, texts: list[str]) -> list[list[float]]:
        return self._call(texts, "passage")

    def embed_query(self, texts: list[str]) -> list[list[float]]:
        return self._call(texts, "query")

    @property
    def model_identity(self) -> str:
        """向量空间身份取**服务端**的（远端决定空间，与 ProxyEmbedAdapter 同语义）。
        回落后取本地 embedder 的——两者必须是同一个模型，否则向量空间会分裂，
        这由部署保证（同一份配置），`__meta__/embed_model_id` 盖章仍是最终防线。

        🔴 **读到就握手**（2026-08-25 live 病例 6，注入面整体空转）：
        `server.py` 建 `MemoryIndex` 时就拿这个值当 `embed_model_id` 做向量空间
        校验，而它此前要等第一次真实往返才有值 ⇒ `current=''` vs
        `stored='local:intfloat/multilingual-e5-large'` 被判**不一致** ⇒
        `EMBED_MODEL_MISMATCH_READONLY` ⇒ `_embedder=None` ⇒ **该 proxy 进程
        终生禁用向量检索**（实测 6/6 轮 `facts_count=0`，日志只有一行 warning，
        用户侧表现为"记忆全没了"）。紧随其后的 `embed(["warmup"])` 会填上身份，
        但**太晚**——校验已经在它之前跑完。

        握手是**零文本请求**（服务端 `_embed` 对空列表直接返回，不进模型），
        代价是一次 socket 往返。失败**不置 degraded**：模块可能只是还没起来，
        走 IPC 还是回落留给第一次真实调用（那条路径上有完整 warning）。

        🔴 不在 `__init__` 里做：构造函数不该有阻塞 I/O——第一版那么写，
        在"服务端与客户端共用一个事件循环"的测试形态下直接死锁。
        """
        if self._degraded and self._local is not None:
            return getattr(self._local, "model_identity", "") or self._identity
        if not self._identity and not self._degraded:
            try:
                self._run_sync(self._roundtrip([], "passage"))
            except Exception as e:  # noqa: BLE001 —— 握手失败不改变 adapter 状态
                logger.debug("embed_identity_probe_failed", error=str(e),
                             transport=self._transport, address=self._address)
        return self._identity

    @property
    def vector_fingerprint(self) -> str:
        """算向量的库版本与 dtype，取**服务端**报的；回落后取本地 embedder 的（MQ-I45）。
        空串 = 现在答不上来（模块没起 / 旧版服务端），台账这次旁路。握手同 `model_identity`。"""
        if self._degraded:
            return getattr(self._local, "vector_fingerprint", "") if self._local is not None else ""
        if not self._fingerprint:
            try:
                self._run_sync(self._roundtrip([], "passage"))
            except Exception as e:  # noqa: BLE001 —— 握手失败不改变 adapter 状态
                logger.debug("embed_identity_probe_failed", error=str(e),
                             transport=self._transport, address=self._address)
        return self._fingerprint

    @property
    def degraded(self) -> bool:
        return self._degraded


def build_ipc_adapter(cfg: Any, role: str, local_factory: Any) -> Any | None:
    """按启动期决断构建 IPC adapter；决断为 local 时返回 None（调用方走原路径）。"""
    from bladex_proxy import embed_transport

    decision = embed_transport.resolve()
    logger.info("embed_transport_resolved", transport=decision.transport,
                address=decision.address, reason=decision.reason, role=role)
    if decision.transport == embed_transport.TRANSPORT_LOCAL:
        return None
    if decision.transport == embed_transport.TRANSPORT_UDS and \
            not os.path.exists(decision.address):
        # 模块没起：不在这里硬失败——adapter 首次调用时会回落并打 warning。
        logger.warning("embed_socket_missing", address=decision.address,
                       hint="start `bladex-embed`, or set BLADEX_EMBED_TRANSPORT=local")
    return IPCEmbedAdapter(
        decision.transport, decision.address,
        priority=PRIORITY_HOT if role == "proxy" else PRIORITY_BULK,
        local_factory=local_factory)
