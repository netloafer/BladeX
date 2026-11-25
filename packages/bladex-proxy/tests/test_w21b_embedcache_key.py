"""批 W21b · MQ-I45：`embedcache/` 键含 embedder 库版本与 dtype，换库不复用旧向量。"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from bladex_proxy.embed_client import IPCEmbedAdapter
from bladex_proxy.storage.memory_index import FastEmbedAdapter, MemoryIndex


class _Emb:
    model_identity = "local:m"
    vector_fingerprint = "fastembed=0.8.0;ort=1.27.0;dtype=float32"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.5] * 4 for _ in texts]


class _NoAttr:
    model_identity = "api:m"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.5] * 4 for _ in texts]


def _idx(path: Path, embedder: object) -> MemoryIndex:
    idx = MemoryIndex(path, embedder=embedder)
    idx.open()
    return idx


def test_changing_library_does_not_reuse_vectors(tmp_path: Path) -> None:
    emb = _Emb()
    idx = _idx(tmp_path / "i", emb)
    idx.put_embedding("local:m", "t", [1.0, 2.0])
    assert idx.get_embeddings("local:m", ["t"]) == {"t": [1.0, 2.0]}
    emb.vector_fingerprint = "fastembed=0.9.1;ort=1.27.0;dtype=float32"
    assert idx.get_embeddings("local:m", ["t"]) == {}, "换 fastembed 不得命中旧向量"
    idx.put_embedding("local:m", "t", [3.0, 4.0])
    assert idx.get_embeddings("local:m", ["t"]) == {"t": [3.0, 4.0]}
    emb.vector_fingerprint = "fastembed=0.8.0;ort=1.27.0;dtype=float32"
    assert idx.get_embeddings("local:m", ["t"]) == {"t": [1.0, 2.0]}, "换回去要读回原来的"
    idx.close()


@pytest.mark.parametrize("field", ["dtype=float32", "ort=1.27.0"])
def test_dtype_and_ort_are_part_of_the_key(tmp_path: Path, field: str) -> None:
    emb = _Emb()
    idx = _idx(tmp_path / "i", emb)
    idx.put_embedding("local:m", "t", [1.0])
    emb.vector_fingerprint = emb.vector_fingerprint.replace(field, field[:-1] + "9")
    assert idx.get_embeddings("local:m", ["t"]) == {}
    idx.close()


def test_key_written_before_the_change_is_kept_but_not_hit(tmp_path: Path) -> None:
    idx = _idx(tmp_path / "i", _Emb())
    h = hashlib.sha256(b"local:m\x00t").hexdigest()[:32]
    old_key = f"embedcache/{h}".encode()
    raw = np.array([7.0], dtype=np.float32).tobytes()
    idx.meta_db[old_key] = raw
    assert idx.get_embeddings("local:m", ["t"]) == {}
    idx.put_embedding("local:m", "t", [1.0])
    assert idx.meta_db.get(old_key) == raw, "旧键保留（回滚时还能用）"
    idx.close()


def test_unknown_fingerprint_neither_reads_nor_writes(tmp_path: Path) -> None:
    emb = _Emb()
    idx = _idx(tmp_path / "i", emb)
    emb.vector_fingerprint = ""
    assert idx.put_embedding("local:m", "t", [1.0]) == [1.0], "向量照常返回"
    assert not [k for k in idx.meta_db.keys() if bytes(k).startswith(b"embedcache/")]
    assert idx.get_embeddings("local:m", ["t"]) == {}
    idx.close()


def test_backend_without_fingerprint_still_caches(tmp_path: Path) -> None:
    idx = _idx(tmp_path / "i", _NoAttr())
    idx.put_embedding("api:m", "t", [1.0])
    assert idx.get_embeddings("api:m", ["t"]) == {"t": [1.0]}
    idx.close()


def test_adapter_fingerprint_names_library_and_dtype() -> None:
    fp = FastEmbedAdapter(model_name="test/m").vector_fingerprint
    assert fp.startswith("fastembed=") and ";ort=" in fp and fp.endswith(";dtype=float32")
    assert "unknown" not in fp


def test_ipc_fingerprint_comes_from_the_server_not_this_process() -> None:
    a = IPCEmbedAdapter("uds", "/nonexistent/x.sock", timeout_s=1, local_factory=_Emb)
    assert a.vector_fingerprint == "", "连不上时答不知道，不猜本进程的版本"
    a._fingerprint = "fastembed=server;ort=1;dtype=float32"
    assert a.vector_fingerprint == "fastembed=server;ort=1;dtype=float32"
    a._degraded, a._local = True, _Emb()
    assert a.vector_fingerprint == _Emb.vector_fingerprint, "回落后向量是本地算的"
