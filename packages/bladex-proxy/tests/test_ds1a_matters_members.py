"""批 DS1a（DS-26 前置）：`GET /admin/matters?include=members` 一次取回本页所有 Matter 的详情。

判据：`members[<id>]` 按 JSONResponse 同一渲染方式序列化后，与逐个
`GET /admin/matters/{id}?facts_top_k=<同值>` 的响应字节**逐字相同**。
"""

from __future__ import annotations

import tempfile

from bladex_core.fact import Fact
from bladex_core.matter import EdgeTargetType, Matter, MatterOrigin, MatterStatus
from bladex_proxy.config import ProxyConfig
from bladex_proxy.server import create_app
from bladex_proxy.storage.memory_index import MemoryIndex
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient


def _vec(i: int) -> list[float]:
    v = [0.0] * 64
    v[i % 64] = 1.0
    return v


def _seeded_index() -> MemoryIndex:
    idx = MemoryIndex(f"{tempfile.mkdtemp(prefix='ds1a_members_')}/p2w",
                      embedder=None, read_only=False)
    idx.open()
    for i in range(40):
        f = Fact(id=f"f{i:03d}", content=f"fact {i} about matter {i % 6}", kind="general",
                 source_user_id="u1")
        f.embedding = _vec(i)
        idx.add_fact(f)
    for m in range(6):
        idx.add_matter(Matter(matter_id=f"m{m:02d}", title=f"matter {m}",
                              status=MatterStatus.ACTIVE, origin=MatterOrigin.MANUAL))
        for i in range(m, 40, 6):
            idx.assign_manual(f"m{m:02d}", EdgeTargetType.FACT, f"f{i:03d}")
        idx.assign_manual(f"m{m:02d}", EdgeTargetType.SESSION, f"u1/a/s{m}/")
    return idx


def test_members_equal_per_matter_detail_byte_for_byte() -> None:
    cfg = ProxyConfig(upstream_model="openai/test", upstream_api_key="sk-fake",
                      rocksdb_path=f"{tempfile.mkdtemp()}/rocksdb",
                      index_path=f"{tempfile.mkdtemp()}/index")
    app = create_app(cfg)
    idx = _seeded_index()
    with TestClient(app) as c:
        started = app.state.index
        app.state.index = idx
        try:
            for top_k in (3, 30):
                r = c.get(f"/admin/matters?include=members&facts_top_k={top_k}&limit=500")
                assert r.status_code == 200, r.text
                body = r.json()
                ids = [m["matter_id"] for m in body["matters"]]
                assert len(ids) == 6 and set(body["members"]) == set(ids)
                for mid in ids:
                    d = c.get(f"/admin/matters/{mid}?facts_top_k={top_k}")
                    assert d.status_code == 200
                    assert JSONResponse(content=body["members"][mid]).body == d.content, mid
                    assert len(body["members"][mid]["facts"]) <= top_k
                # 对照不是两份空详情
                assert all(body["members"][mid]["facts"] and body["members"][mid]["edges"]
                           and body["members"][mid]["session_keys"] for mid in ids)
            # 不带 include：响应形状与修前相同（没有 members 键）
            plain = c.get("/admin/matters?limit=500").json()
            assert "members" not in plain and plain["matters"] == body["matters"]
            # include 值拼错：说出来，不静默返回"没有成员"
            assert c.get("/admin/matters?include=member").status_code == 400
        finally:
            app.state.index = started
            idx.close()
