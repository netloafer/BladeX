"""批 W21b · MQ-W21 / MQ-W29：embed 服务回包的类型与失败上报。假模型，不加载真向量。

真实 embed_server + IPC 走 `docs/benchmarks/ab0/w21b_real_ipc.py`（两个 fastembed 环境各一次）；
本文件的用例在 gate 里跑，钉的是「`list(e)` 漏 numpy 标量」和「回包失败不再静默」这两件事。

| 用例 | 钉什么 |
|---|---|
| `test_adapter_output_is_plain_float32_values` | 两种原始 dtype 都输出纯 Python float，取值相同 |
| `test_adapter_output_survives_json` | 11-03 的直接死因：adapter 输出必须能 `json.dumps` |
| `test_type_assertion_*` | 回包前拦下 numpy 标量（含 `np.float64` 这个能混过 json 的） |
| `test_unserializable_reply_*` | 回错误帧、不让客户端干等；warning 带原因 |
| `test_write_failure_warns_*` | MQ-W29：写失败是 warning，带异常类型与原因 |
| `test_client_tells_*` | MQ-W29：客户端区分「回包序列化失败」和「连接真的断了」 |
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import threading

import numpy as np
import pytest
from bladex_proxy.embed_client import EmbedServerReplyError, IPCEmbedAdapter
from bladex_proxy.embed_server import (
    ERROR_KIND_REPLY,
    EmbedService,
    _Job,
    vector_type_problem,
)
from bladex_proxy.storage.memory_index import FastEmbedAdapter
from structlog.testing import capture_logs


class _RawModel:
    """按指定 dtype 吐 numpy 向量（fastembed 0.8 = float64，0.9 = float32）。"""

    def __init__(self, dtype: str) -> None:
        self.dtype = dtype

    def embed(self, texts: list[str]):  # noqa: ANN201
        for _ in texts:
            yield np.array([1.2291209995746613, -0.24222474917769432], dtype=self.dtype)


def _adapter(dtype: str) -> FastEmbedAdapter:
    a = FastEmbedAdapter(model_name="test/model")
    a._model = _RawModel(dtype)
    return a


@pytest.mark.parametrize("method", ["embed", "embed_query", "embed_passage"])
def test_adapter_output_is_plain_float32_values(method: str) -> None:
    v64 = getattr(_adapter("float64"), method)(["t"])[0]
    v32 = getattr(_adapter("float32"), method)(["t"])[0]
    assert all(type(x) is float for x in v64 + v32)
    assert v64 == v32, "两种原始 dtype 必须给出同一取值"
    assert v64 == np.array(v64, dtype=np.float32).tolist(), "取值必须能无损落回 float32"


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_adapter_output_survives_json(dtype: str) -> None:
    assert json.loads(json.dumps(_adapter(dtype).embed(["t"])))[0][0] == pytest.approx(1.22912097)


@pytest.mark.parametrize(("vectors", "needle"), [
    ([[np.float32(1.0)]], "numpy.float32"),
    ([[np.float64(1.0)]], "numpy.float64"),
    ([np.array([1.0])], "numpy.ndarray"),
    (np.zeros((1, 1)), "numpy.ndarray"),
])
def test_type_assertion_rejects_numpy(vectors, needle: str) -> None:  # noqa: ANN001
    problem = vector_type_problem(vectors)
    assert problem is not None and needle in problem


def test_type_assertion_accepts_plain_numbers() -> None:
    assert vector_type_problem([[1.0, 2], []]) is None
    assert vector_type_problem([]) is None


class _Writer:
    def __init__(self, *, fail: Exception | None = None) -> None:
        self.frames: list[bytes] = []
        self.fail = fail

    def write(self, b: bytes) -> None:
        if self.fail is not None:
            raise self.fail
        self.frames.append(b)

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        return None


class _LeakyEmbedder:
    model_identity = "test:leaky"

    def embed(self, texts: list[str]) -> list[list]:
        return [[np.float32(0.5)] for _ in texts]


def _run_one(svc: EmbedService, writer: _Writer) -> dict:
    job = _Job(req_id="r1", texts=["a"], input_type="passage", priority="hot", writer=writer)
    with capture_logs() as logs:
        asyncio.run(svc._run_job(job))
    return {"reply": json.loads(writer.frames[0]) if writer.frames else None, "logs": logs}


def test_unserializable_reply_gets_error_frame_and_warning() -> None:
    got = _run_one(EmbedService(_LeakyEmbedder()), _Writer())
    assert got["reply"]["id"] == "r1"
    assert got["reply"]["error_kind"] == ERROR_KIND_REPLY
    assert "numpy.float32" in got["reply"]["error"]
    hits = [x for x in got["logs"] if x["event"] == "embed_send_failed"]
    assert len(hits) == 1 and hits[0]["log_level"] == "warning"
    assert hits[0]["kind"] == "serialize" and "numpy.float32" in hits[0]["error"]


def test_unserializable_payload_in_send_is_still_answered() -> None:
    """类型断言之外的序列化失败（`_send` 那一道）同样回错误帧。"""
    w = _Writer()
    with capture_logs() as logs:
        asyncio.run(EmbedService(_LeakyEmbedder())._send(w, {"id": "r2", "x": object()}))
    reply = json.loads(w.frames[0])
    assert reply["id"] == "r2" and reply["error_kind"] == ERROR_KIND_REPLY
    hits = [x for x in logs if x["event"] == "embed_send_failed"]
    assert [(x["kind"], x["log_level"]) for x in hits] == [("serialize", "warning")]
    assert hits[0]["error_type"] == "TypeError" and "not JSON serializable" in hits[0]["error"]


def test_write_failure_warns_with_type_and_reason() -> None:
    w = _Writer(fail=BrokenPipeError("peer gone"))
    with capture_logs() as logs:
        asyncio.run(EmbedService(_LeakyEmbedder())._send(w, {"id": "r3", "vectors": []}))
    hits = [x for x in logs if x["event"] == "embed_send_failed"]
    assert len(hits) == 1 and hits[0]["log_level"] == "warning"
    assert hits[0]["kind"] == "connection"
    assert hits[0]["error_type"] == "BrokenPipeError" and "peer gone" in hits[0]["error"]


class _LocalEmbedder:
    model_identity = "test:local"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[9.0] for _ in texts]


@pytest.fixture
def sock():  # noqa: ANN201
    with tempfile.TemporaryDirectory(prefix="w21b", dir="/tmp") as d:  # noqa: S108 —— sun_path 上限
        yield os.path.join(d, "s.sock")


def _serve(handler, path: str) -> None:  # noqa: ANN001
    """线程里起一个只跑 handler 的 UDS 服务（客户端走真 socket）。"""
    ready = threading.Event()

    async def main() -> None:
        server = await asyncio.start_unix_server(handler, path=path)
        ready.set()
        async with server:
            await server.serve_forever()

    threading.Thread(target=lambda: asyncio.run(main()), daemon=True).start()
    assert ready.wait(5)


def _fallback_events(logs: list[dict]) -> list[dict]:
    return [x for x in logs if x["event"] == "embed_ipc_fallback_local"]


def test_client_tells_unserializable_reply_from_dropped_connection(sock: str) -> None:
    async def bad_reply(reader, writer) -> None:  # noqa: ANN001
        await reader.readline()
        await EmbedService(_LeakyEmbedder())._run_job(_Job(
            req_id="x", texts=["a"], input_type="passage", priority="hot", writer=writer))
        await writer.drain()

    _serve(bad_reply, sock)
    a = IPCEmbedAdapter("uds", sock, timeout_s=5, local_factory=_LocalEmbedder)
    with capture_logs() as logs:
        assert a.embed(["t"]) == [[9.0]]
    ev = _fallback_events(logs)
    assert len(ev) == 1
    assert ev[0]["error_type"] == EmbedServerReplyError.__name__
    assert "could not serialize" in ev[0]["error"] and "numpy.float32" in ev[0]["error"]
    assert "embed_send_failed" in ev[0]["hint"] and "unreachable" not in ev[0]["hint"]
    assert "closed the connection" not in ev[0]["error"]


def test_client_tells_dropped_connection_from_unserializable_reply(sock: str) -> None:
    async def hang_up(reader, writer) -> None:  # noqa: ANN001
        await reader.readline()
        writer.close()

    _serve(hang_up, sock)
    a = IPCEmbedAdapter("uds", sock, timeout_s=5, local_factory=_LocalEmbedder)
    with capture_logs() as logs:
        assert a.embed(["t"]) == [[9.0]]
    ev = _fallback_events(logs)
    assert len(ev) == 1 and ev[0]["error_type"] == "ConnectionError"
    assert "closed the connection" in ev[0]["error"] and "unreachable" in ev[0]["hint"]
    assert "serialize" not in ev[0]["error"]
