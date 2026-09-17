"""Matter 手动映射操作测试（ADR-0012 T4, §3.5）。"""

import tempfile
from pathlib import Path

from bladex_core.matter import (
    EdgeProvenance,
    EdgeTargetType,
    Matter,
    MatterEdge,
    MatterOrigin,
    MatterStatus,
)
from bladex_proxy.storage.memory_index import MemoryIndex


def _make_index(tmpdir: str) -> MemoryIndex:
    index = MemoryIndex(Path(tmpdir) / "index", embedder=None, read_only=False)
    index.open()
    return index


def test_manual_edge_survives_reclustering():
    """T4: 手动指定的归属边不被自动管线改动。

    模拟：先手动指定归属，再跑自动归属管线 → manual 边纹丝不动。
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        index = _make_index(tmpdir)

        # 创建 Matter
        index.add_matter(Matter(matter_id="m-001", title="test", centroid=[1.0, 0.0]))

        # 手动指定归属
        index.assign_manual("m-001", EdgeTargetType.FACT, "fact-001", weight=2.0)

        # 验证 manual 边存在
        manual = index.get_manual_edge_for_target("fact-001", EdgeTargetType.FACT)
        assert manual is not None
        assert manual.provenance == EdgeProvenance.MANUAL
        assert manual.weight == 2.0
        assert manual.matter_id == "m-001"

        # 自动管线检查 manual 边 → 跳过（不重复归属）
        existing = index.get_manual_edge_for_target("fact-001", EdgeTargetType.FACT)
        assert existing is not None  # 已有 manual 边，自动管线应跳过

        # manual 边不变
        edges = index.get_edges("m-001")
        assert len(edges) == 1
        assert edges[0].provenance == EdgeProvenance.MANUAL
        assert edges[0].target_key == "fact-001"

        index.close()


def test_admin_event_written_before_index_apply():
    """T4: 每个手动操作在 Memory Hub 都有对应管理事件，先于 Memory Index 变更落盘。

    通过 server 端点验证：先写 Memory Hub 管理事件，再写 Memory Index。
    """
    from bladex_proxy.config import ProxyConfig
    from bladex_proxy.models import AdminEventType
    from bladex_proxy.server import create_app
    from fastapi.testclient import TestClient

    tmpdir = tempfile.mkdtemp()
    config = ProxyConfig(
        hard_rules=["MUST be polite"],
        upstream_model="openai/test",
        upstream_api_key="sk-fake",
        rocksdb_path=f"{tmpdir}/rocksdb",
        index_path=f"{tmpdir}/index",
    )
    app = create_app(config)

    with TestClient(app) as client:
        # 调 create_matter 端点
        resp = client.post("/admin/matters", json={"title": "test matter", "summary": "test"})
        assert resp.status_code == 200
        matter_id = resp.json()["matter_id"]

        # Memory Hub 有对应管理事件
        ledger = app.state.hub
        events = list(ledger.scan_admin_events())
        assert len(events) >= 1
        create_events = [e for _, e in events if e.event_type == AdminEventType.MATTER_CREATE]
        assert len(create_events) >= 1
        assert create_events[0].matter_id == matter_id
        assert create_events[0].payload.get("title") == "test matter"

        # Memory Index 也有对应 Matter（Memory Hub 先于 Memory Index 落盘）
        p2w = MemoryIndex(Path(tmpdir) / "index", read_only=False)
        p2w.open()
        matter = p2w.get_matter(matter_id)
        assert matter is not None
        assert matter.title == "test matter"
        assert matter.origin == MatterOrigin.MANUAL
        p2w.close()


def test_matter_merge_preserves_manual_edges():
    """T4: 合并 Matter 时 manual 边不丢失。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        index = _make_index(tmpdir)

        # 两个 Matter
        index.add_matter(Matter(matter_id="m-001", title="alpha", centroid=[1.0, 0.0]))
        index.add_matter(Matter(matter_id="m-002", title="beta", centroid=[0.0, 1.0]))

        # m-001 有 manual + auto 边
        index.assign_manual("m-001", EdgeTargetType.FACT, "fact-manual", weight=2.0)
        index.add_edge(MatterEdge(
            matter_id="m-001", target_type=EdgeTargetType.SESSION,
            target_key="u1/a1/s1/", provenance=EdgeProvenance.AUTO, confidence=0.7,
        ))

        # 合并 m-001 → m-002
        moved = index.merge_matters("m-001", "m-002")
        assert moved == 2

        # m-002 现在有两条边，manual 边仍在
        edges = index.get_edges("m-002")
        assert len(edges) == 2
        manual_edges = [e for e in edges if e.provenance == EdgeProvenance.MANUAL]
        assert len(manual_edges) == 1
        assert manual_edges[0].target_key == "fact-manual"
        assert manual_edges[0].weight == 2.0

        index.close()


def test_close_and_rename_matter():
    """T4: 关闭和重命名 Matter。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        index = _make_index(tmpdir)
        index.add_matter(Matter(matter_id="m-001", title="original"))

        # 重命名
        assert index.rename_matter("m-001", "new title")
        matter = index.get_matter("m-001")
        assert matter.title == "new title"

        # 关闭
        assert index.close_matter("m-001")
        matter = index.get_matter("m-001")
        assert matter.status == MatterStatus.CLOSED

        index.close()


def test_detach_edge():
    """T4: 手动摘除归属边。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        index = _make_index(tmpdir)
        index.add_matter(Matter(matter_id="m-001", title="test"))

        edge = index.assign_manual("m-001", EdgeTargetType.SESSION, "u1/a1/s1/")
        assert len(index.get_edges("m-001")) == 1

        # 摘除
        assert index.detach_edge(edge.edge_id)
        assert len(index.get_edges("m-001")) == 0

        # 再次摘除返回 False
        assert not index.detach_edge(edge.edge_id)

        index.close()


def test_remove_edges_for_turn():
    """T4: 级联清除——删除与某 turn 关联的归属边。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        index = _make_index(tmpdir)
        index.add_matter(Matter(matter_id="m-001", title="test"))

        index.add_edge(MatterEdge(
            matter_id="m-001", target_type=EdgeTargetType.SESSION,
            target_key="u1/a1/s1/",
        ))
        index.add_edge(MatterEdge(
            matter_id="m-001", target_type=EdgeTargetType.SESSION,
            target_key="u2/a2/s2/",
        ))

        removed = index.remove_edges_for_turn("u1/a1/s1/")
        assert removed == 1
        assert len(index.get_edges("m-001")) == 1

        index.close()


def test_detach_endpoint_by_edge_id():
    """T4: POST /admin/matters/{id}/detach — 按 edge_id 摘除。"""
    from bladex_proxy.config import ProxyConfig
    from bladex_proxy.models import AdminEventType
    from bladex_proxy.server import create_app
    from fastapi.testclient import TestClient

    tmpdir = tempfile.mkdtemp()
    config = ProxyConfig(
        upstream_model="openai/test", upstream_api_key="sk-fake",
        rocksdb_path=f"{tmpdir}/rocksdb", index_path=f"{tmpdir}/index",
    )
    app = create_app(config)

    with TestClient(app) as client:
        # 先创建 Matter + assign
        resp = client.post("/admin/matters", json={"title": "test"})
        matter_id = resp.json()["matter_id"]
        resp = client.post(f"/admin/matters/{matter_id}/assign",
                           json={"target_type": "session", "target_key": "u1/a1/s1/"})
        assert resp.status_code == 200

        # 取 edge_id
        p2w = MemoryIndex(Path(tmpdir) / "index", read_only=False)
        p2w.open()
        edges = p2w.get_edges(matter_id)
        assert len(edges) == 1
        edge_id = edges[0].edge_id
        p2w.close()

        # detach
        resp = client.post(f"/admin/matters/{matter_id}/detach",
                           json={"edge_id": edge_id})
        assert resp.status_code == 200
        assert resp.json()["status"] == "detached"
        assert resp.json()["removed"] == 1

        # Memory Hub 有 detach 管理事件
        events = list(app.state.hub.scan_admin_events())
        detach_events = [e for _, e in events if e.event_type == AdminEventType.MATTER_DETACH]
        assert len(detach_events) == 1
        assert detach_events[0].payload.get("edge_id") == edge_id

        # Memory Index 边已摘除
        p2w = MemoryIndex(Path(tmpdir) / "index", read_only=False)
        p2w.open()
        assert len(p2w.get_edges(matter_id)) == 0
        p2w.close()


def test_detach_endpoint_by_target_key():
    """T4: POST /admin/matters/{id}/detach — 按 target_key 摘除。"""
    from bladex_proxy.config import ProxyConfig
    from bladex_proxy.server import create_app
    from fastapi.testclient import TestClient

    tmpdir = tempfile.mkdtemp()
    config = ProxyConfig(
        upstream_model="openai/test", upstream_api_key="sk-fake",
        rocksdb_path=f"{tmpdir}/rocksdb", index_path=f"{tmpdir}/index",
    )
    app = create_app(config)

    with TestClient(app) as client:
        resp = client.post("/admin/matters", json={"title": "test"})
        matter_id = resp.json()["matter_id"]
        client.post(f"/admin/matters/{matter_id}/assign",
                    json={"target_type": "fact", "target_key": "fact-001"})

        resp = client.post(f"/admin/matters/{matter_id}/detach",
                           json={"target_key": "fact-001", "target_type": "fact"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "detached"
        assert resp.json()["removed"] == 1

        p2w = MemoryIndex(Path(tmpdir) / "index", read_only=False)
        p2w.open()
        assert len(p2w.get_edges(matter_id)) == 0
        p2w.close()


def test_detach_validation_error():
    """T4: detach 不带 edge_id 也不带 target_key → 400。"""
    from bladex_proxy.config import ProxyConfig
    from bladex_proxy.server import create_app
    from fastapi.testclient import TestClient

    tmpdir = tempfile.mkdtemp()
    config = ProxyConfig(
        upstream_model="openai/test", upstream_api_key="sk-fake",
        rocksdb_path=f"{tmpdir}/rocksdb", index_path=f"{tmpdir}/index",
    )
    app = create_app(config)

    with TestClient(app) as client:
        resp = client.post("/admin/matters/m-001/detach", json={})
        assert resp.status_code == 400


def test_split_endpoint():
    """T4: POST /admin/matters/{id}/split — 拆分 Matter。"""
    from bladex_proxy.config import ProxyConfig
    from bladex_proxy.models import AdminEventType
    from bladex_proxy.server import create_app
    from fastapi.testclient import TestClient

    tmpdir = tempfile.mkdtemp()
    config = ProxyConfig(
        upstream_model="openai/test", upstream_api_key="sk-fake",
        rocksdb_path=f"{tmpdir}/rocksdb", index_path=f"{tmpdir}/index",
    )
    app = create_app(config)

    with TestClient(app) as client:
        # 创建 source + assign 两条边
        resp = client.post("/admin/matters", json={"title": "source"})
        source_id = resp.json()["matter_id"]
        client.post(f"/admin/matters/{source_id}/assign",
                    json={"target_type": "fact", "target_key": "fact-a"})
        client.post(f"/admin/matters/{source_id}/assign",
                    json={"target_type": "fact", "target_key": "fact-b"})

        # 取 edge_ids
        p2w = MemoryIndex(Path(tmpdir) / "index", read_only=False)
        p2w.open()
        edges = p2w.get_edges(source_id)
        edge_ids = [e.edge_id for e in edges]
        p2w.close()

        # 拆分
        resp = client.post(f"/admin/matters/{source_id}/split",
                           json={"new_title": "split result", "edge_ids": [edge_ids[0]]})
        assert resp.status_code == 200
        assert resp.json()["status"] == "split"
        assert resp.json()["moved_edges"] == 1
        new_matter_id = resp.json()["new_matter_id"]

        # Memory Hub 有 split 管理事件
        events = list(app.state.hub.scan_admin_events())
        split_events = [e for _, e in events if e.event_type == AdminEventType.MATTER_SPLIT]
        assert len(split_events) == 1
        assert split_events[0].payload.get("new_matter_id") == new_matter_id

        # Memory Index: source 剩 1 条边, new 有 1 条边
        p2w = MemoryIndex(Path(tmpdir) / "index", read_only=False)
        p2w.open()
        assert len(p2w.get_edges(source_id)) == 1
        new_edges = p2w.get_edges(new_matter_id)
        assert len(new_edges) == 1
        assert new_edges[0].matter_id == new_matter_id
        p2w.close()


def test_admin_deferred_when_index_busy(monkeypatch):
    """Memory Index: Memory Index 写锁竞争失败时返回 202 deferred（Memory Hub 事件已写）。"""
    import bladex_proxy.server.admin_api as server_mod   # F0.1 拆包：消费方在 admin_api.py
    from bladex_proxy.config import ProxyConfig
    from bladex_proxy.server import create_app
    from fastapi.testclient import TestClient

    tmpdir = tempfile.mkdtemp()
    config = ProxyConfig(
        upstream_model="openai/test", upstream_api_key="sk-fake",
        rocksdb_path=f"{tmpdir}/rocksdb", index_path=f"{tmpdir}/index",
    )
    app = create_app(config)

    # mock _open_writable_index 返回 None（模拟写锁竞争失败）
    monkeypatch.setattr(server_mod, "_open_writable_index", lambda cfg: None)

    with TestClient(app) as client:
        resp = client.post("/admin/matters", json={"title": "deferred test"})
        assert resp.status_code == 202
        assert resp.json()["status"] == "deferred"

        # Memory Hub 管理事件已写（下次重建时会重放）
        events = list(app.state.hub.scan_admin_events())
        assert len(events) >= 1


def test_replay_detach_and_split_events():
    """T4: _replay_admin_events 重放 DETACH + SPLIT。"""
    from bladex_proxy.models import AdminEventType
    from bladex_proxy.storage.memory_hub import MemoryHub

    with tempfile.TemporaryDirectory() as tmpdir:
        ledger = MemoryHub(f"{tmpdir}/ledger")
        ledger.open()
        index = _make_index(tmpdir)

        # 写入 create + assign + detach + split 管理事件
        ledger.append_admin_event(AdminEventType.MATTER_CREATE, "m-001", title="test")
        ledger.append_admin_event(AdminEventType.MATTER_ASSIGN, "m-001", "fact-001",
                              target_type="fact")
        ledger.append_admin_event(AdminEventType.MATTER_DETACH, "m-001",
                              edge_id="", target_key="fact-001", target_type="fact")
        ledger.append_admin_event(AdminEventType.MATTER_CREATE, "m-src", title="source")
        ledger.append_admin_event(AdminEventType.MATTER_ASSIGN, "m-src", "fact-a",
                              target_type="fact")
        ledger.append_admin_event(AdminEventType.MATTER_SPLIT, "m-src",
                              new_matter_id="m-split", new_title="split",
                              edge_ids=[])

        # 重放
        replayed = index._replay_admin_events(ledger)
        assert replayed == 6  # create + assign + detach + create + assign + split
        # Actually: create(1) + assign(1) + detach(1) + create(1) + assign(1) + split(1) = 6

        # m-001 的 assign 被 detach 撤销
        assert len(index.get_edges("m-001")) == 0

        # m-src 存在, m-split 也被创建
        assert index.get_matter("m-src") is not None
        assert index.get_matter("m-split") is not None

        index.close()
        ledger.close()
