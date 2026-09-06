"""V-A6 端到端：真起 UDS 服务 + 真跑客户端往返（2026-08-25 live 三连坑的回归）。

live 踩到的三件事，单测都没覆盖，因为它们全在"进程真的起起来"那一层：
1. `run/` 父目录不存在 ⇒ `sock.bind` 抛 FileNotFoundError，进程秒退；
2. 服务端读到 `.env` 的 `BLADEX_EMBED_BACKEND=ipc` ⇒ **去连自己**；
3. CLI 判活只看"进程还活着" ⇒ 打了 "started" 而服务其实已死（**started 骗人**）。
"""

from __future__ import annotations

import asyncio
import os

import pytest

from bladex_proxy.embed_client import IPCEmbedAdapter
from bladex_proxy.embed_server import EmbedService, run_server


@pytest.fixture
def sock_dir():
    """🔴 UDS 测试**不能用 pytest 的 tmp_path**：`sockaddr_un.sun_path` 上限是
    macOS 104 / Linux 108 字节，而 pytest 的
    `/private/var/folders/…/pytest-of-…/pytest-NN/<长测试名>0/` 轻松超过 ⇒
    `OSError: AF_UNIX path too long`。

    2026-08-26 用户机 gate 上 5 条全红、而沙盒（`/tmp/…` 短）全绿——
    **测试宿主的路径长度差异是一类盲区**（与"测试宿主无事件循环"、
    "测试规模 vs 生产规模"同族，这是第三次撞到同一族问题）。

    这条不只是测试问题：它暴露的产品缺陷已修在 `embed_transport.resolve`
    （路径超限 → 降 tcp 并在 reason 写明长度）与 `run_server`（先于 bind 拒绝、
    错误指向配置）。守卫见 `test_long_uds_path_falls_back_to_tcp`。
    """
    import shutil
    import tempfile
    d = tempfile.mkdtemp(prefix="bx", dir="/tmp")
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


class _FakeEmbedder:
    model_identity = "test:e2e"

    def embed_passage(self, texts):
        return [[float(len(t))] for t in texts]

    embed = embed_passage

    def embed_query(self, texts):
        return [[float(len(t)) + 0.5] for t in texts]


def test_uds_roundtrip_creates_parent_dir(sock_dir):
    """🔴 socket 落在**不存在的**子目录里，服务端必须自己建（live 病例 1）。"""
    sock = os.path.join(sock_dir, "run", "embed.sock")
    assert not os.path.exists(os.path.dirname(sock))

    async def scenario():
        svc = EmbedService(_FakeEmbedder(), bulk_slice=4)
        server = asyncio.ensure_future(
            run_server(svc, transport="uds", address=sock))
        for _ in range(100):                    # 等监听就绪
            if os.path.exists(sock):
                break
            await asyncio.sleep(0.05)
        assert os.path.exists(sock), "socket 未创建 ⇒ 父目录没建成"
        # 真跑一次往返（同步 adapter 在线程里跑，避免嵌套 event loop）
        ad = IPCEmbedAdapter("uds", sock, local_factory=None)
        vecs = await asyncio.to_thread(ad.embed_passage, ["abc", "de"])
        q = await asyncio.to_thread(ad.embed_query, ["abc"])
        server.cancel()
        return vecs, q, ad

    vecs, q, ad = asyncio.run(scenario())
    assert vecs == [[3.0], [2.0]]
    assert q == [[3.5]]                          # input_type 被正确转达
    assert ad.model_identity == "test:e2e"       # 身份取服务端的
    assert not ad.degraded                       # 没有回落


def test_works_inside_running_event_loop(sock_dir):
    """🔴 2026-08-25 live 病例 4（proxy 侧静默失效）：proxy 是 uvicorn，**宿主已有
    事件循环**，首版客户端直接 `asyncio.run()` ⇒ 每次调用抛
    `asyncio.run() cannot be called from a running event loop` ⇒ 回落 local ⇒
    热路径根本没走 IPC，而表面上一切正常（有兜底）。

    此前所有测试都在**无事件循环**的环境跑（consolidator 形态），所以全绿却漏掉了
    proxy 形态——测试环境与被测宿主的差异，本身就是一类盲区。
    """
    import threading

    sock = os.path.join(sock_dir, "run", "embed.sock")
    ready = threading.Event()

    def serve():
        # 🔴 服务端跑在**独立线程的独立循环**里——对应生产的"独立进程"。
        # 若与客户端共用一个循环，客户端的同步阻塞会把服务端一起卡死
        # （那是测试构造的假象，不是生产形态；第一版测试栽在这上面，121s 超时）。
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        svc = EmbedService(_FakeEmbedder(), bulk_slice=4)
        loop.create_task(run_server(svc, transport="uds", address=sock))
        loop.call_later(0.1, ready.set)
        loop.run_forever()

    t = threading.Thread(target=serve, daemon=True)
    t.start()
    ready.wait(timeout=10)
    for _ in range(100):
        if os.path.exists(sock):
            break
        import time as _t
        _t.sleep(0.05)

    async def scenario():
        ad = IPCEmbedAdapter("uds", sock, local_factory=_FakeEmbedder, timeout_s=10)
        # 🔴 关键：**直接同步调用**（不经 to_thread）——模拟 FastAPI 处理器里
        # 同步调 embedder 的真实形态。首版在这里必炸并回落。
        return ad.embed_passage(["abc"]), ad

    vecs, ad = asyncio.run(scenario())
    assert vecs == [[3.0]]
    assert not ad.degraded, "在宿主事件循环里回落了 ⇒ proxy 热路径没走 IPC"


def test_large_response_exceeds_default_stream_limit(sock_dir):
    """🔴 2026-08-25 live 病例 5：asyncio `readline()` 默认缓冲仅 **64KB**，
    而 200 条 ×1024 维响应约 4MB、consolidator 的 1063 条批约 20MB ⇒
    默认值下必炸 `Separator is not found, and chunk exceed the limit`。

    我选"一行 JSON"分帧时没算过响应体量——**这是设计疏漏不是调参**。
    此前的测试全用 1–20 条小批，永远碰不到这条线：**测试规模与生产规模的差距
    本身就是一类盲区**（与"测试宿主无事件循环"是同一族问题）。
    """
    sock = os.path.join(sock_dir, "run", "embed.sock")

    class _WideEmbedder:            # 1024 维，贴近真实 e5-large
        model_identity = "test:wide"

        def embed_passage(self, texts):
            return [[0.123456789] * 1024 for _ in texts]

        embed = embed_passage
        embed_query = embed_passage

    async def scenario():
        svc = EmbedService(_WideEmbedder(), bulk_slice=64)
        server = asyncio.ensure_future(run_server(svc, transport="uds", address=sock))
        for _ in range(100):
            if os.path.exists(sock):
                break
            await asyncio.sleep(0.05)
        ad = IPCEmbedAdapter("uds", sock, local_factory=None, timeout_s=60)
        # 200 条 ×1024 维 ≈ 4MB —— 远超默认 64KB
        vecs = await asyncio.to_thread(ad.embed_passage, [f"t{i}" for i in range(200)])
        server.cancel()
        return vecs, ad

    vecs, ad = asyncio.run(scenario())
    assert len(vecs) == 200 and len(vecs[0]) == 1024
    assert not ad.degraded, "大响应把连接打爆后回落了 ⇒ 生产上重建批必失败"


def test_identity_available_before_any_real_call(sock_dir):
    """🔴 2026-08-25 live 病例 6（**注入面整体空转**）：`server.py` 建 MemoryIndex
    时就读 `embedder.model_identity` 去做向量空间校验；IPC adapter 若要等第一次
    真实往返才有身份，那时读到的是空串 ⇒ 被判"与库内模型不一致" ⇒ 只读端
    `_embedder=None` ⇒ **该 proxy 进程终生没有记忆**（实测 6/6 轮 facts_count=0）。

    所以**读 `model_identity` 这个动作本身就要能拿到身份**——本例在一次真实
    embed 都没调过的情况下断言它非空。（不在 `__init__` 里握手：构造函数做阻塞
    I/O 在"服务端客户端共用一个循环"的测试形态下会死锁。）
    """
    sock = os.path.join(sock_dir, "run", "embed.sock")

    async def scenario():
        svc = EmbedService(_FakeEmbedder())
        server = asyncio.ensure_future(run_server(svc, transport="uds", address=sock))
        for _ in range(100):
            if os.path.exists(sock):
                break
            await asyncio.sleep(0.05)
        ad = IPCEmbedAdapter("uds", sock)
        ident = await asyncio.to_thread(lambda: ad.model_identity)
        server.cancel()
        return ad, svc, ident

    ad, svc, ident = asyncio.run(scenario())
    assert ident == "test:e2e", "读身份时没握手 ⇒ 向量空间校验会误判不一致"
    assert not ad.degraded
    # 握手是零文本请求：服务端记了一次 job，但**没进模型**
    assert svc.counters["hot_jobs"] == 1


def test_identity_probe_failure_is_not_degradation(tmp_path):
    """模块没起时的握手失败**不该**把 adapter 直接钉成 degraded——
    模块可能只是还没起来，回落与否留给第一次真实调用（那条路径有 warning）。"""
    ad = IPCEmbedAdapter("uds", str(tmp_path / "nope.sock"), local_factory=_FakeEmbedder)
    assert not ad.degraded
    assert ad.model_identity == ""


def test_socket_permissions_are_owner_only(sock_dir):
    """无鉴权面的前提：socket 只有本用户可连。"""
    sock = os.path.join(sock_dir, "run", "embed.sock")

    async def scenario():
        svc = EmbedService(_FakeEmbedder())
        server = asyncio.ensure_future(run_server(svc, transport="uds", address=sock))
        for _ in range(100):
            if os.path.exists(sock):
                break
            await asyncio.sleep(0.05)
        mode = os.stat(sock).st_mode & 0o777
        server.cancel()
        return mode

    assert asyncio.run(scenario()) == 0o600


def test_server_forces_local_backend(monkeypatch):
    """🔴 live 病例 2：服务端读到 `backend=ipc` 会去连自己。
    `main()` 必须在建 embedder 前把它强制成 local——这里验证那行代码存在且
    在 build_embedder 之前（读源码断言，避免真加载模型）。"""
    import inspect

    from bladex_proxy import embed_server
    src = inspect.getsource(embed_server.main)
    force = src.index('os.environ["BLADEX_EMBED_BACKEND"] = "local"')
    build = src.index("build_embedder(cfg")
    assert force < build, "强制 local 必须在 build_embedder 之前"


def test_client_reports_degraded_when_socket_dead(tmp_path):
    """服务不在时回落 local 并置 degraded（CLI 的 ⚠ 分支依赖这个语义）。"""
    ad = IPCEmbedAdapter("uds", str(tmp_path / "nope.sock"),
                         local_factory=_FakeEmbedder)
    assert ad.embed(["x"]) == [[1.0]]
    assert ad.degraded


@pytest.mark.parametrize("n", [1, 7])
def test_tcp_roundtrip(n, tmp_path):
    """tcp 传输（Windows / Docker 跨容器走这条）。"""
    async def scenario():
        svc = EmbedService(_FakeEmbedder(), bulk_slice=4)
        server = asyncio.ensure_future(
            run_server(svc, transport="tcp", address="127.0.0.1:38099"))
        await asyncio.sleep(0.3)
        ad = IPCEmbedAdapter("tcp", "127.0.0.1:38099", local_factory=None)
        out = await asyncio.to_thread(ad.embed_passage, ["x" * i for i in range(1, n + 1)])
        server.cancel()
        return out

    vecs = asyncio.run(scenario())
    assert len(vecs) == n


def test_long_uds_path_falls_back_to_tcp():
    """🔴 2026-08-26 用户机 gate 撞出的**产品缺陷**（不是测试环境问题）：

    `sockaddr_un.sun_path` 是内核 ABI 常量（macOS 104 / Linux 108 字节）。
    部署根较深的用户，`<root>/run/embed.sock` 会超限 ⇒ `bind()` 抛
    `OSError: AF_UNIX path too long` ⇒ **服务端进程当场死**，而 `bladex start`
    只会打 "started"——正是 live 病例 1/3 的形状（started 骗人）。

    决断层必须像"平台不支持 uds"那条一样**降 tcp 并在 reason 写明**，
    不能让它跑到 bind 才死。
    """
    from bladex_proxy import embed_transport as et

    long_path = "/" + "d" * 120 + "/embed.sock"
    d = et.resolve(explicit="uds", explicit_address=long_path, env={})
    assert d.transport == "tcp", "超长 uds 路径没有降级 ⇒ 服务端会在 bind 时死掉"
    assert "too_long" in d.reason and str(et._SUN_PATH_MAX) in d.reason, \
        f"降级了但 reason 说不清为什么：{d.reason!r}"

    # auto 档同样要降（用户没显式指定时更需要它自己躲开）
    d_auto = et.resolve(explicit_address=long_path, env={})
    assert d_auto.transport == "tcp" and "too_long" in d_auto.reason

    # 阳性对照：短路径必须**仍然**走 uds，否则上面两条只是"永远降 tcp"
    d_ok = et.resolve(explicit="uds", explicit_address="/tmp/bx/embed.sock", env={})
    assert d_ok.transport == "uds" and d_ok.address == "/tmp/bx/embed.sock"


def test_server_refuses_long_uds_path_with_actionable_error():
    """第二道门：调用方硬传长路径时，错误要指向**配置**，不是内核那句话。"""
    svc = EmbedService(_FakeEmbedder())
    long_path = "/" + "d" * 120 + "/embed.sock"
    with pytest.raises(OSError) as ei:
        asyncio.run(run_server(svc, transport="uds", address=long_path))
    msg = str(ei.value)
    assert "BLADEX_EMBED_TRANSPORT=tcp" in msg and "sun_path" in msg
