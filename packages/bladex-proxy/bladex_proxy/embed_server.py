"""Embedding 独立模块服务端 —— IPC + hot/bulk 优先级队列（V-A6；ADR-0032 §2.1）。

# 为什么要有这个进程（MQ-W8 两个 live 病例）

consolidator 与 proxy 共用一个嵌入服务时，一次 1063 条的重建批要 **66 秒**，
期间 proxy 热路径与 `/health` 一起排队 ⇒ ① CC 客户端超时重试（端到端 96s）；
② `bladex status` 谎报 `Proxy: ✗ not running`（红灯骗人，ADR-0027 §2 镜像面）。

🔴 **独立进程只解决"资源归属"，不解决"排队"**——那 66 秒绝大部分是模型推理本身。
所以本模块的核心不是"搬出去"，是**两级队列 + bulk 切片让路**：
`hot`（proxy 热路径）永远先出队；`bulk`（consolidator 重建）按 `slice` 切片，
**每片之间检查一次 hot 队列**——这一步才是"让路"的执行点，没有它优先级只是
排序不是抢占。

# 协议（BladeX 私有，故意做薄）

请求/响应都是一行 JSON + `\\n`（长度用换行分帧，不引入第二套编码）：

    → {"id": "...", "texts": [...], "input_type": "passage|query", "priority": "hot|bulk"}
    ← {"id": "...", "vectors": [[...]], "model_identity": "local:BAAI/bge-small-en-v1.5"}
    ← {"id": "...", "error": "..."}                     # 失败也带 id，客户端不悬挂

不用 HTTP：省一层解析、无端口鉴权面、uds 下延迟更低。传输选择在**启动时一次决断**
（`embed_transport.resolve`，Jason 2026-08-25 拍板），结果打进启动日志。
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger()

#: 🔴 流缓冲上限（协议契约的一部分，两端必须一致）。
#: asyncio `readline()` 默认只有 **64KB**——而 200 条 × 1024 维的响应就有 ~4MB，
#: consolidator 的 1063 条重建批约 20MB ⇒ 默认值下必炸
#: `Separator is not found, and chunk exceed the limit`（2026-08-25 live 探针抓到）。
#: 选"一行 JSON"分帧时我没算过响应体量，这是设计疏漏，不是调参。
#: 64MB 覆盖 ~3000 条 ×1024 维；再大应改分帧协议（长度前缀），不是继续调大。
STREAM_LIMIT = 64 * 1024 * 1024

PRIORITY_HOT = "hot"
PRIORITY_BULK = "bulk"

#: bulk 请求的切片大小：每片之间让 hot 插队。默认 64——1063 条 ≈ 17 片，
#: 单片推理 ~4s（实测 66s/1063×64），故 hot 最坏等待 ≈ 一片时间而非整批。
#: 🔴 这个值是"让路粒度"，不是吞吐旋钮：调大 = hot 等待变长，调小 = 批处理效率下降。
DEFAULT_BULK_SLICE = 64


@dataclass
class _Job:
    req_id: str
    texts: list[str]
    input_type: str
    priority: str
    writer: Any
    enqueued_ms: float = field(default_factory=lambda: time.monotonic() * 1000)


class EmbedQueue:
    """两级队列：hot 先出；bulk 只有在 hot 空时才被取出。纯数据结构，可单测。"""

    def __init__(self) -> None:
        self._hot: list[_Job] = []
        self._bulk: list[_Job] = []
        self._wake = asyncio.Event()

    def put(self, job: _Job) -> None:
        (self._hot if job.priority == PRIORITY_HOT else self._bulk).append(job)
        self._wake.set()

    def has_hot(self) -> bool:
        return bool(self._hot)

    def pop(self) -> _Job | None:
        if self._hot:
            return self._hot.pop(0)
        if self._bulk:
            return self._bulk.pop(0)
        return None

    async def wait(self) -> None:
        await self._wake.wait()
        if not self._hot and not self._bulk:
            self._wake.clear()

    def clear_if_empty(self) -> None:
        if not self._hot and not self._bulk:
            self._wake.clear()

    @property
    def depth(self) -> tuple[int, int]:
        return (len(self._hot), len(self._bulk))


class EmbedService:
    """单模型实例 + 单工作循环。模型加载与推理由注入的 `embedder` 承担
    （测试可注入假 embedder；生产传 `build_embedder(cfg, role='embed')`）。"""

    def __init__(self, embedder: Any, *, bulk_slice: int = DEFAULT_BULK_SLICE) -> None:
        self._embedder = embedder
        self._slice = max(1, bulk_slice)
        self.queue = EmbedQueue()
        self._stopping = False
        #: 可观测（MQ-W8 教训：没有读数就无法证明"让路"真的发生）
        self.counters: dict[str, float] = {
            "hot_jobs": 0, "bulk_jobs": 0, "slices_yielded": 0,
            "hot_wait_ms_max": 0.0, "bulk_wait_ms_max": 0.0,
        }

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        # 空请求 = **身份握手**（客户端构造时探测向量空间身份，见 embed_client
        # `_probe_identity`）。不进模型：有些 embedder 对空列表会抛。
        if not texts:
            return []
        if input_type == "query" and hasattr(self._embedder, "embed_query"):
            return self._embedder.embed_query(texts)
        if hasattr(self._embedder, "embed_passage"):
            return self._embedder.embed_passage(texts)
        return self._embedder.embed(texts)

    async def _run_job(self, job: _Job) -> None:
        wait_ms = time.monotonic() * 1000 - job.enqueued_ms
        key = "hot_wait_ms_max" if job.priority == PRIORITY_HOT else "bulk_wait_ms_max"
        self.counters[key] = max(self.counters[key], wait_ms)
        self.counters["hot_jobs" if job.priority == PRIORITY_HOT else "bulk_jobs"] += 1
        try:
            vectors: list[list[float]] = []
            if job.priority == PRIORITY_BULK and len(job.texts) > self._slice:
                # 🔴 让路执行点：切片跑，每片后把控制权交回事件循环；
                # 若期间来了 hot，工作循环会在下一轮先取它（见 `serve_forever`）。
                for i in range(0, len(job.texts), self._slice):
                    part = job.texts[i:i + self._slice]
                    vectors.extend(await asyncio.to_thread(
                        self._embed, part, job.input_type))
                    if self.queue.has_hot():
                        self.counters["slices_yielded"] += 1
                        await self._drain_hot()
            else:
                vectors = await asyncio.to_thread(self._embed, job.texts, job.input_type)
            payload = {"id": job.req_id, "vectors": vectors,
                       "model_identity": getattr(self._embedder, "model_identity", "")}
        except Exception as e:  # noqa: BLE001 —— 单请求失败不炸服务
            logger.warning("embed_job_failed", req=job.req_id, error=str(e))
            payload = {"id": job.req_id, "error": str(e)}
        await self._send(job.writer, payload)

    async def _drain_hot(self) -> None:
        """把当前排着的 hot 全部跑完再回到 bulk（hot 之间按 FIFO）。"""
        while self.queue.has_hot():
            hot = self.queue.pop()
            if hot is None:
                return
            await self._run_job(hot)

    async def _send(self, writer: Any, payload: dict) -> None:
        try:
            writer.write((json.dumps(payload, ensure_ascii=False) + "\n").encode())
            await writer.drain()
        except Exception as e:  # noqa: BLE001 —— 客户端断开不是服务端错误
            logger.debug("embed_send_failed", error=str(e))

    async def serve_forever(self) -> None:
        while not self._stopping:
            await self.queue.wait()
            job = self.queue.pop()
            if job is None:
                self.queue.clear_if_empty()
                continue
            await self._run_job(job)

    def stop(self) -> None:
        self._stopping = True
        self.queue._wake.set()

    async def handle_client(self, reader: Any, writer: Any) -> None:
        """一个连接可发多个请求（长连接；客户端复用）。"""
        try:
            while True:
                line = await reader.readline()
                if not line:
                    return
                try:
                    req = json.loads(line.decode("utf-8", "replace"))
                except ValueError:
                    await self._send(writer, {"id": "", "error": "bad json"})
                    continue
                texts = req.get("texts") or []
                if not isinstance(texts, list):
                    await self._send(writer, {"id": req.get("id", ""),
                                              "error": "texts must be a list"})
                    continue
                self.queue.put(_Job(
                    req_id=str(req.get("id", "")), texts=[str(t) for t in texts],
                    input_type=str(req.get("input_type", "passage")),
                    priority=(PRIORITY_HOT if req.get("priority") == PRIORITY_HOT
                              else PRIORITY_BULK),
                    writer=writer))
        finally:
            try:
                writer.close()
            except Exception:  # noqa: BLE001,S110
                pass


async def run_server(service: EmbedService, *, transport: str, address: str) -> None:
    """起监听 + 工作循环。`transport ∈ {uds, tcp}`（决断在调用方，见 embed_transport）。"""
    worker = asyncio.ensure_future(service.serve_forever())
    if transport == "uds":
        # 🔴 父目录必须先建（2026-08-25 live：`run/` 不存在 ⇒ `sock.bind` 抛
        # `FileNotFoundError` ⇒ 进程起来就死，而 `bladex start` 已经打了
        # "started (PID …)"——**父目录是 bind 的前置条件，不是可选项**）。
        # 🔴 路径长度先于 bind 检查：`sun_path` 超限时 `bind()` 抛的是
        # `OSError: AF_UNIX path too long`，进程当场死，而启动器只会打 "started"。
        # 决断层（embed_transport.resolve）已会降 tcp，这里是第二道门——防御
        # "调用方直接传了一个长路径"，且把错误说成人话（指向配置而非内核）。
        from bladex_proxy.embed_transport import _SUN_PATH_MAX, uds_path_too_long
        if uds_path_too_long(address):
            raise OSError(
                f"UDS path is {len(os.fsencode(address))} bytes, the kernel limit "
                f"is {_SUN_PATH_MAX} (sockaddr_un.sun_path): {address!r}. "
                f"Set BLADEX_EMBED_TRANSPORT=tcp, or BLADEX_EMBED_ADDRESS to a "
                f"shorter path.")
        parent = os.path.dirname(address)
        if parent:
            os.makedirs(parent, exist_ok=True)
        if os.path.exists(address):
            os.unlink(address)
        server = await asyncio.start_unix_server(service.handle_client, path=address,
                                                 limit=STREAM_LIMIT)
        os.chmod(address, 0o600)      # 只有本用户可连（无鉴权面的前提）
    else:
        host, _, port = address.rpartition(":")
        server = await asyncio.start_server(service.handle_client,
                                            host or "127.0.0.1", int(port),
                                            limit=STREAM_LIMIT)
    logger.info("embed_server_listening", transport=transport, address=address,
                bulk_slice=service._slice)
    async with server:
        try:
            await server.serve_forever()
        finally:
            service.stop()
            worker.cancel()


def main() -> int:  # pragma: no cover —— 入口点（env 读取只在这里，导入无副作用）
    import argparse

    from bladex_proxy import embed_transport
    from bladex_proxy.config import ProxyConfig
    from bladex_proxy.embedding import build_embedder

    ap = argparse.ArgumentParser(description="BladeX embedding module (V-A6)")
    ap.add_argument("--transport", default="", help="uds|tcp（空=启动期自动决断）")
    ap.add_argument("--address", default="", help="socket 路径或 host:port")
    ap.add_argument("--bulk-slice", type=int, default=0)
    args = ap.parse_args()

    decision = embed_transport.resolve(explicit=args.transport,
                                       explicit_address=args.address)
    logger.info("embed_transport_resolved", transport=decision.transport,
                address=decision.address, reason=decision.reason)
    if decision.transport == "local":
        logger.error("embed_server_refused",
                     hint="transport resolved to 'local'; the module has nothing to serve")
        return 2

    # 🔴 **服务端必须真加载本地模型**：它就是那份被共享的模型。
    # `.env` 里 `BLADEX_EMBED_BACKEND=ipc` 是给**客户端**（proxy/consolidator）看的，
    # 服务端读到它会去连自己（2026-08-25 live：日志出现 `embedder_backend
    # backend=ipc role=proxy` + `embed_socket_missing`）——与 `backend=proxy` 档下
    # "proxy 侧强制 local"同一形态、同一修法（embedding.py 已有先例）。
    os.environ["BLADEX_EMBED_BACKEND"] = "local"
    cfg = ProxyConfig()
    embedder = build_embedder(cfg, role="proxy")
    slice_n = args.bulk_slice or int(
        os.environ.get("BLADEX_EMBED_BULK_SLICE", DEFAULT_BULK_SLICE))
    service = EmbedService(embedder, bulk_slice=slice_n)
    asyncio.run(run_server(service, transport=decision.transport,
                           address=decision.address))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
