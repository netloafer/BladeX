"""Beta T11 fact 删除（forget）验收：Memory Hub FACT 墓碑 + Memory Index 级联 + 重建不复活。

ADR-0012 §3.6 语义：删除 = 追加墓碑（Memory Hub 真相源不改写）→ Memory Index 级联清除 →
rebuild 按墓碑过滤（fact.id 确定性 hash，跨 rebuild 稳定）。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from bladex_core.fact import Fact
from bladex_core.matter import EdgeTargetType, Matter, MatterEdge
from bladex_proxy.config import ProxyConfig
from bladex_proxy.models import Identity, TombstoneTargetType, Turn, TurnStatus
from bladex_proxy.server import create_app
from bladex_proxy.storage.memory_hub import MemoryHub
from bladex_proxy.storage.memory_index import MemoryIndex
from fastapi.testclient import TestClient


class MockEmbedder:
    def embed(self, texts):
        out = []
        for t in texts:
            vec = [0.0] * 64
            vec[hash(t) % 64] = 1.0
            out.append(vec)
        return out

    @property
    def available(self):
        return True


def _make_config() -> ProxyConfig:
    tmpdir = tempfile.mkdtemp()
    return ProxyConfig(
        auth_enabled=False,  # 显式：不依赖进程 env（同批测试可能加载过 .env）
        hard_rules=["MUST be polite"],
        upstream_model="openai/test",
        upstream_api_key="sk-fake",
        rocksdb_path=f"{tmpdir}/rocksdb",
        index_path=f"{tmpdir}/index",
    )


def test_delete_fact_endpoint_tombstone_and_cascade():
    """DELETE /admin/facts/{id}：Memory Hub 墓碑写入 + cfg.index_path 里的 fact 被级联删除。"""
    cfg = _make_config()
    # 先把 fact 种进 cfg.index_path（_apply_to_index 打开的就是这个库）
    index = MemoryIndex(cfg.index_path, embedder=None, read_only=False)
    index.open()
    index.add_fact(Fact(id="f-del", content="要被删除的事实", source_user_id="u1"))
    index.close()

    app = create_app(cfg)
    with TestClient(app) as client:
        app.state.index = None  # 跳过读句柄存在性检查（沙盒无 embedder；语义幂等）
        resp = client.delete("/admin/facts/f-del")
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["status"] == "deleted"
        assert data["tombstone_key"]

        # Memory Hub 墓碑在场且类型正确
        ledger: MemoryHub = app.state.hub
        tombs = [(k, t) for k, t in ledger.scan_tombstones()]
        assert any(t.target_type == TombstoneTargetType.FACT
                   and t.target_key == "f-del" for _, t in tombs)

    # Memory Index 级联：fact 已不在
    index = MemoryIndex(cfg.index_path, embedder=None, read_only=False)
    index.open()
    assert index.get_fact("f-del") is None
    index.close()


def test_delete_fact_404_when_not_found():
    cfg = _make_config()
    app = create_app(cfg)
    with TestClient(app) as client:
        # 挂一个可查的 Memory Index 读句柄（空库）→ 未知 id 走 404，不写墓碑
        tmpdir = tempfile.mkdtemp()
        index = MemoryIndex(Path(tmpdir) / "index", embedder=None, read_only=False)
        index.open()
        app.state.index = index
        try:
            resp = client.delete("/admin/facts/nonexistent")
            assert resp.status_code == 404
            ledger: MemoryHub = app.state.hub
            assert not list(ledger.scan_tombstones())
        finally:
            index.close()


def test_delete_fact_removes_edges():
    """delete_fact 级联摘除指向该 fact 的归属边。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        index = MemoryIndex(Path(tmpdir) / "index", embedder=None, read_only=False)
        index.open()
        index.add_fact(Fact(id="f1", content="x", source_user_id="u1"))
        index.add_matter(Matter(matter_id="m1", title="t"))
        index.add_edge(MatterEdge(edge_id="e1", matter_id="m1",
                               target_type=EdgeTargetType.FACT, target_key="f1"))
        assert len(index.get_edges("m1")) == 1
        assert index.delete_fact("f1") is True
        assert index.get_fact("f1") is None
        assert index.get_edges("m1") == []
        # 幂等：再删返回 False 不抛
        assert index.delete_fact("f1") is False
        index.close()


def test_rebuild_does_not_resurrect_tombstoned_fact():
    """重建等价性 spot check：FACT 墓碑在 full rebuild 后仍然生效（不复活）。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger = MemoryHub(Path(tmpdir) / "rocksdb")
        ledger.open()
        identity = Identity(user_id="u1", agent_id="agentA", session_id="s1")
        turn = Turn(
            identity=identity,
            model="test",
            request_messages=[
                {"role": "user",
                 "content": "用户决定把 T11 的验收标准定为墓碑语义必须在重建后保持。"},
            ],
            response_text="ok",
            status=TurnStatus.OK,
        )
        ledger.put(identity.storage_key("0001-0"), turn)

        # 第一次重建：产出确定性 id 的 fact
        p2a = MemoryIndex(Path(tmpdir) / "p2a", embedder=MockEmbedder(), read_only=False)
        p2a.open()
        n = p2a.rebuild_from_hub(ledger, full=True)
        assert n >= 1
        facts = p2a.all_facts()
        assert facts
        fact_id = facts[0].id
        p2a.close()

        # 写 FACT 墓碑 → 新库 full rebuild → 该 id 不复活（其余不受影响）
        ledger.append_tombstone(target_type=TombstoneTargetType.FACT, target_key=fact_id)
        p2b = MemoryIndex(Path(tmpdir) / "p2b", embedder=MockEmbedder(), read_only=False)
        p2b.open()
        p2b.rebuild_from_hub(ledger, full=True)
        assert p2b.get_fact(fact_id) is None
        assert all(f.id != fact_id for f in p2b.all_facts())
        p2b.close()
        ledger.close()
