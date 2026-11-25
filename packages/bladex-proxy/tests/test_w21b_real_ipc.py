"""批 W21b · 真实 embed_server + 真实 IPC + 真实 e5-large（拍板 (49)-b：AB 的 gate 全是假 embedder，
真实回包路径从没跑过，这才让 11-03 的 float32 一路漏到生产）。

起一个真 `python -m bladex_proxy.embed_server` 子进程，客户端经 unix socket 取向量。
没有模型缓存（公开树 / 快照）时跳过；主仓 gate 用当前 `.venv` 的 fastembed 跑。
0.8.0 / 0.9.1 两个环境各跑一次的读数在 `report-batch-W21b-20261103.md` §2：
  W21B_FASTEMBED_CACHE=<缓存副本> W21B_IPC_OUT=<json> <venv>/bin/python -m pytest <本文件>
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
from bladex_proxy.embed_client import IPCEmbedAdapter
from structlog.testing import capture_logs

MODEL = "intfloat/multilingual-e5-large"
REPO = Path(__file__).resolve().parents[3]
CACHE = Path(os.environ.get("W21B_FASTEMBED_CACHE", REPO / "data" / "fastembed_cache"))
TEXTS = ["全量重建可以中断并续跑", "The consolidator scans the whole hub", "判重逐条决策与修前相同"]


@pytest.fixture(scope="module")
def server():  # noqa: ANN201
    if not (CACHE / "models--qdrant--multilingual-e5-large-onnx").is_dir():
        pytest.skip(f"没有 e5-large 模型缓存：{CACHE}")
    home = tempfile.mkdtemp(prefix="w21b", dir="/tmp")  # noqa: S108 —— sun_path 上限 104 字节
    address = os.path.join(home, "e.sock")
    log_path = os.path.join(home, "embed.log")
    env = {k: v for k, v in os.environ.items() if not k.startswith("BLADEX_")}
    env.update(BLADEX_HOME=home, BLADEX_EMBED_MODEL=MODEL, BLADEX_FASTEMBED_CACHE_PATH=str(CACHE),
               HF_HUB_OFFLINE="1", PYTHONPATH=os.pathsep.join(
                   [str(REPO / "packages/bladex-core"), str(REPO / "packages/bladex-proxy")]))
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(  # noqa: S603
            [sys.executable, "-m", "bladex_proxy.embed_server", "--bladex-home", home,
             "--transport", "uds", "--address", address],
            stdout=log, stderr=subprocess.STDOUT, cwd=home, env=env)
    try:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            assert proc.poll() is None, Path(log_path).read_text(encoding="utf-8")
            if os.path.exists(address):
                s = socket.socket(socket.AF_UNIX)
                try:
                    s.connect(address)
                    break
                except OSError:
                    pass
                finally:
                    s.close()
            time.sleep(0.2)
        else:
            pytest.fail("embed_server 120 s 内没有监听")
        yield address, log_path
    finally:
        proc.terminate()
        proc.wait(10)
        shutil.rmtree(home, ignore_errors=True)


def _no_local():  # noqa: ANN202
    raise AssertionError("发生了进程内回退")


def test_real_reply_round_trip_without_local_fallback(server) -> None:  # noqa: ANN001
    address, log_path = server
    a = IPCEmbedAdapter("uds", address, priority="hot", timeout_s=120, local_factory=_no_local)
    with capture_logs() as logs:
        identity = a.model_identity
        passage = a.embed(TEXTS)
        query = a.embed_query(TEXTS[:1])
    fingerprint = a.vector_fingerprint
    fallback = [x for x in logs if x["event"] == "embed_ipc_fallback_local"]
    server_log = Path(log_path).read_text(encoding="utf-8")

    reading = {
        "fastembed": importlib.metadata.version("fastembed"),
        "onnxruntime": importlib.metadata.version("onnxruntime"),
        "identity": identity, "fingerprint": fingerprint,
        "vectors": len(passage), "dim": len(passage[0]),
        "element_types": sorted({type(x).__name__ for v in passage + query for x in v}),
        "embed_ipc_fallback_local": len(fallback),
        "embed_send_failed": server_log.count("embed_send_failed"),
        "degraded": a.degraded,
    }
    if os.environ.get("W21B_IPC_OUT"):
        Path(os.environ["W21B_IPC_OUT"]).write_text(json.dumps(reading, indent=1), encoding="utf-8")

    assert not fallback and not a.degraded
    assert reading["embed_send_failed"] == 0, server_log
    assert identity == f"local:{MODEL}"
    assert len(passage) == 3 and len(query) == 1
    for v in passage + query:
        assert len(v) == 1024
        assert all(type(x) is float for x in v)
        assert np.array(v, dtype=np.float32).astype(np.float64).tolist() == v, "取值不是 float32"
    assert passage[0] != query[0], "query / passage 前缀要走到服务端"
    assert f"fastembed={reading['fastembed']};" in fingerprint and fingerprint.endswith("dtype=float32")
