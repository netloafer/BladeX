"""Beta T17 remember 端点验收：manual fact 落库 + journal 存活 + scope 越权拒绝。"""

from __future__ import annotations

import tempfile

from bladex_proxy.config import ProxyConfig
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


def test_remember_persists_and_survives_rebuild():
    cfg = _make_config()
    app = create_app(cfg)
    with TestClient(app) as client:
        resp = client.post("/admin/facts",
                           json={"content": "用户的生产环境是 arm64"})
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["status"] == "remembered"
        fact_id = data["fact_id"]

        # journal 事件在场
        ledger: MemoryHub = app.state.hub
        events = [e for _, e in ledger.scan_admin_events()]
        assert any(str(e.event_type.value) == "fact_import"
                   and e.target_key == fact_id for e in events)

        # 幂等：同文本二次 remember → exists、journal 不膨胀
        # （app.state.index 在沙盒可能为 None——existence 检查退化为再次 journal 也
        #   被指纹式 id 保证不重复入 Memory Index；这里以真实读句柄验证）
        p2r = MemoryIndex(cfg.index_path, embedder=None, read_only=False)
        p2r.open()
        assert p2r.get_fact(fact_id) is not None
        assert "origin:manual" in p2r.get_fact(fact_id).tags
        assert p2r.get_fact(fact_id).trust == 1.0
        p2r.close()

        app.state.index = None  # 明确走"读句柄不可用"路径
        resp2 = client.post("/admin/facts",
                            json={"content": "用户的生产环境是 arm64"})
        assert resp2.status_code == 200
        assert resp2.json()["fact_id"] == fact_id  # 确定性 id 幂等

    # full rebuild（无会话数据）→ journal 重放，remember 的 fact 不丢
    p3b = MemoryHub(cfg.rocksdb_path)
    p3b.open()
    p2b = MemoryIndex(f"{cfg.index_path}_rebuilt", embedder=MockEmbedder(),
                    read_only=False)
    p2b.open()
    p2b.rebuild_from_hub(p3b, full=True)
    assert p2b.get_fact(fact_id) is not None, "FACT_IMPORT journal 未被重放"
    p2b.close()
    p3b.close()


def test_remember_validation():
    cfg = _make_config()
    app = create_app(cfg)
    with TestClient(app) as client:
        # 空 content
        assert client.post("/admin/facts", json={"content": "  "}).status_code == 400
        # 非法 scope 形态
        r = client.post("/admin/facts",
                        json={"content": "x", "scope": "everyone"})
        assert r.status_code == 400
        # scope 越权：个人模式（无 identity.toml）写 org → 400 + 指向组织层
        r = client.post("/admin/facts",
                        json={"content": "x", "scope": "org:acme"})
        assert r.status_code == 400
        assert "identity.toml" in r.json()["error"]["message"]
        # personal scope 合法
        r = client.post("/admin/facts",
                        json={"content": "个人可见事实", "scope": "personal"})
        assert r.status_code == 200
