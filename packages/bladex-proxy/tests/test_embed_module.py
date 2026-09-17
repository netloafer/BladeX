"""V-A6 验收剧本：优先级让路 / 传输启动期决断 / 回落 local。

🔴 核心判据（卡 §6 T6）：**hot 请求能在进行中的 bulk 大批里插队**——
这是 MQ-W8 的解，也是"独立进程只解决资源归属、不解决排队"那句话的验收面。
"""

from __future__ import annotations

import asyncio
import time

import pytest
from bladex_proxy.embed_server import (
    PRIORITY_BULK,
    PRIORITY_HOT,
    EmbedQueue,
    EmbedService,
    _Job,
)
from bladex_proxy.embed_transport import (
    TRANSPORT_LOCAL,
    TRANSPORT_TCP,
    TRANSPORT_UDS,
    resolve,
)


class _FakeEmbedder:
    """每条文本 8ms 的假模型（可观察切片让路的时序）。"""

    model_identity = "test:fake-embedder"

    def __init__(self, per_text_s: float = 0.008):
        self.per_text_s = per_text_s
        self.calls: list[int] = []

    def embed_passage(self, texts):
        self.calls.append(len(texts))
        time.sleep(self.per_text_s * len(texts))
        return [[float(len(t))] for t in texts]

    def embed_query(self, texts):
        return self.embed_passage(texts)

    def embed(self, texts):
        return self.embed_passage(texts)


class _Writer:
    def __init__(self):
        self.payloads: list[bytes] = []

    def write(self, b: bytes):
        self.payloads.append(b)

    async def drain(self):
        return None

    def close(self):
        return None


# ── 队列语义 ────────────────────────────────────────────────────────────────

class TestQueue:
    def test_hot_before_bulk(self):
        q = EmbedQueue()
        w = _Writer()
        q.put(_Job("b1", ["x"], "passage", PRIORITY_BULK, w))
        q.put(_Job("h1", ["y"], "passage", PRIORITY_HOT, w))
        assert q.pop().req_id == "h1"      # 后进的 hot 先出
        assert q.pop().req_id == "b1"
        assert q.pop() is None

    def test_depth_and_has_hot(self):
        q = EmbedQueue()
        w = _Writer()
        assert not q.has_hot() and q.depth == (0, 0)
        q.put(_Job("h", ["a"], "passage", PRIORITY_HOT, w))
        assert q.has_hot() and q.depth == (1, 0)


# ── 🔴 让路（本卡的核心验收）──────────────────────────────────────────────

class TestYielding:
    def test_hot_preempts_running_bulk_batch(self):
        """bulk 大批进行中来了 hot ⇒ hot 必须在**整批跑完之前**被处理。"""
        emb = _FakeEmbedder(per_text_s=0.004)
        svc = EmbedService(emb, bulk_slice=8)
        wb, wh = _Writer(), _Writer()

        async def scenario():
            worker = asyncio.ensure_future(svc.serve_forever())
            svc.queue.put(_Job("bulk1", [f"t{i}" for i in range(64)],
                               "passage", PRIORITY_BULK, wb))
            await asyncio.sleep(0.05)          # 让 bulk 先跑起来
            t0 = time.monotonic()
            svc.queue.put(_Job("hot1", ["urgent"], "passage", PRIORITY_HOT, wh))
            while not wh.payloads:             # 等 hot 的响应
                await asyncio.sleep(0.005)
                if time.monotonic() - t0 > 5:
                    break
            hot_wait = time.monotonic() - t0
            while not wb.payloads:             # bulk 最终也要完成
                await asyncio.sleep(0.01)
                if time.monotonic() - t0 > 10:
                    break
            svc.stop()
            worker.cancel()
            return hot_wait

        hot_wait = asyncio.run(scenario())
        assert wh.payloads, "hot 请求没有被响应"
        # 整批 64×4ms ≈ 256ms；hot 若等整批结束才回，wait 会接近它。
        assert hot_wait < 0.15, f"hot 未插队（等待 {hot_wait:.3f}s）"
        assert svc.counters["slices_yielded"] >= 1, "让路计数为 0 ⇒ 切片没起作用"

    def test_bulk_completes_correctly_after_yield(self):
        emb = _FakeEmbedder(per_text_s=0.001)
        svc = EmbedService(emb, bulk_slice=4)
        wb, wh = _Writer(), _Writer()

        async def scenario():
            worker = asyncio.ensure_future(svc.serve_forever())
            svc.queue.put(_Job("b", [f"t{i}" for i in range(20)], "passage",
                               PRIORITY_BULK, wb))
            await asyncio.sleep(0.005)
            svc.queue.put(_Job("h", ["u"], "passage", PRIORITY_HOT, wh))
            for _ in range(400):
                if wb.payloads and wh.payloads:
                    break
                await asyncio.sleep(0.01)
            svc.stop()
            worker.cancel()

        asyncio.run(scenario())
        import json
        bulk_resp = json.loads(wb.payloads[-1].decode())
        assert len(bulk_resp["vectors"]) == 20, "让路后 bulk 结果不完整"
        assert bulk_resp["model_identity"] == "test:fake-embedder"

    def test_small_bulk_not_sliced(self):
        emb = _FakeEmbedder(per_text_s=0.0)
        svc = EmbedService(emb, bulk_slice=64)
        w = _Writer()

        async def scenario():
            worker = asyncio.ensure_future(svc.serve_forever())
            svc.queue.put(_Job("b", ["a", "b"], "passage", PRIORITY_BULK, w))
            for _ in range(200):
                if w.payloads:
                    break
                await asyncio.sleep(0.005)
            svc.stop()
            worker.cancel()

        asyncio.run(scenario())
        assert emb.calls == [2]          # 一次调用，没被切片


# ── 传输决断（启动期一次；Jason 拍板）────────────────────────────────────

class TestTransportResolution:
    def test_explicit_local(self):
        d = resolve(explicit="local", env={})
        assert d.transport == TRANSPORT_LOCAL and d.reason == "explicit:local"

    def test_explicit_tcp_with_address(self):
        d = resolve(explicit="tcp", explicit_address="127.0.0.1:9999", env={})
        assert (d.transport, d.address) == (TRANSPORT_TCP, "127.0.0.1:9999")

    def test_env_is_read(self):
        d = resolve(env={"BLADEX_EMBED_TRANSPORT": "tcp",
                         "BLADEX_EMBED_ADDRESS": "h:1"})
        assert (d.transport, d.address) == (TRANSPORT_TCP, "h:1")

    def test_auto_on_posix_host_prefers_uds(self, monkeypatch):
        monkeypatch.setattr("bladex_proxy.embed_transport._uds_supported",
                            lambda: True)
        monkeypatch.setattr("bladex_proxy.embed_transport._in_container",
                            lambda: False)
        d = resolve(env={})
        assert d.transport == TRANSPORT_UDS and d.reason == "auto:posix_host"

    def test_auto_in_container_uses_tcp(self, monkeypatch):
        """compose 里 proxy/consolidator 是两个容器 ⇒ 必须 tcp（uds 跨容器不通）。"""
        monkeypatch.setattr("bladex_proxy.embed_transport._uds_supported",
                            lambda: True)
        monkeypatch.setattr("bladex_proxy.embed_transport._in_container",
                            lambda: True)
        d = resolve(env={})
        assert d.transport == TRANSPORT_TCP and d.reason == "auto:container_detected"

    def test_auto_without_af_unix_uses_tcp(self, monkeypatch):
        """Windows：Python 无可用 AF_UNIX。"""
        monkeypatch.setattr("bladex_proxy.embed_transport._uds_supported",
                            lambda: False)
        d = resolve(env={})
        assert d.transport == TRANSPORT_TCP and "no_af_unix" in d.reason

    def test_explicit_uds_on_unsupported_platform_is_visible(self, monkeypatch):
        """显式要 uds 但平台不支持 ⇒ 降级到 tcp，但 reason 必须说得出为什么
        （用户明确指定过的东西被换掉不许静默）。"""
        monkeypatch.setattr("bladex_proxy.embed_transport._uds_supported",
                            lambda: False)
        d = resolve(explicit="uds", env={})
        assert d.transport == TRANSPORT_TCP
        assert "unsupported" in d.reason


# ── 回落 local（ADR-0017 同原则：模块没起不拒服务）──────────────────────

class TestFallback:
    def test_falls_back_and_warns(self, caplog):
        from bladex_proxy.embed_client import IPCEmbedAdapter
        local = _FakeEmbedder(per_text_s=0.0)
        ad = IPCEmbedAdapter("uds", "/nonexistent/embed.sock",
                             local_factory=lambda: local)
        with caplog.at_level("WARNING"):
            vecs = ad.embed_passage(["a", "b"])
        assert len(vecs) == 2
        assert ad.degraded
        assert ad.model_identity == "test:fake-embedder"

    def test_no_fallback_configured_raises(self):
        from bladex_proxy.embed_client import IPCEmbedAdapter
        ad = IPCEmbedAdapter("uds", "/nonexistent/embed.sock", local_factory=None)
        with pytest.raises(RuntimeError):
            ad.embed(["x"])

    def test_empty_texts_short_circuit(self):
        from bladex_proxy.embed_client import IPCEmbedAdapter
        ad = IPCEmbedAdapter("uds", "/nonexistent/embed.sock", local_factory=None)
        assert ad.embed([]) == []      # 不触发连接，也不回落
