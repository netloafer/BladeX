"""测试进程不许连 live 的 embed 服务（批 GS2 · MQ-A87，2026-09-29）。

修前：主仓里每个跑 lifespan 的用例都经 `build_embedder` → IPC → 部署根 `run/embed.sock`
把 warmup 打到**生产 embed 进程**（一次全量 507 次连接、243 个用例）。根 conftest 现在
① 默认给 `build_embedder` 换确定性假 embedder；② IPC adapter 的地址是部署根 socket 即抛错并在
teardown 点名。本文件守这两件事，以及"去掉 ① 就会撞上 ②"这条阴性对照——
写成与环境无关的形式（把 socket 解析钉到 live 集合里），快照 / 公开树里同样成立。
"""

from __future__ import annotations

import os
import tempfile

import pytest
from bladex_proxy import embed_client, embed_transport
from bladex_proxy import embedding as _emb_mod
from bladex_proxy.config import ProxyConfig
from bladex_proxy.embed_client import IPCEmbedAdapter

# 模块顶层按名字取原函数：conftest 的 autouse 夹具只替换模块属性，这个引用不受影响。
from bladex_proxy.embedding import build_embedder as _REAL_BUILD_EMBEDDER
from bladex_proxy.server import app_factory, create_app
from fastapi.testclient import TestClient


def _live() -> list[str]:
    return sorted(embed_client._TEST_LIVE_SOCKETS)


def _take_violations(since: int) -> list[str]:
    """取走本例记下的违规（否则 conftest teardown 会把本例判红——那是给意外撞上的用例的）。"""
    got = embed_client._TEST_LIVE_VIOLATIONS[since:]
    del embed_client._TEST_LIVE_VIOLATIONS[since:]
    return got


def test_live_socket_set_covers_the_deployment_root():
    socks = _live()
    assert socks and all(s.endswith(os.path.join("run", "embed.sock")) for s in socks)
    from bladex_proxy import deployment

    root = deployment.find_root()
    if root:  # 主仓：cwd 祖先里有 config/.env —— live 的那个 socket 必须在集合里
        assert os.path.realpath(os.path.join(root, "run", "embed.sock")) in socks


def test_adapter_to_live_socket_is_refused_and_named():
    since = len(embed_client._TEST_LIVE_VIOLATIONS)
    with pytest.raises(RuntimeError, match="live embed socket"):
        IPCEmbedAdapter("uds", _live()[0], local_factory=None)
    got = _take_violations(since)
    assert len(got) == 1 and "test_adapter_to_live_socket_is_refused_and_named" in got[0]


def test_adapter_to_other_socket_is_untouched(tmp_path):
    """守卫只认部署根那个 socket：e2e 用例自建的临时 socket 照常构造。"""
    IPCEmbedAdapter("uds", str(tmp_path / "embed.sock"), local_factory=None)


def _app_cfg() -> ProxyConfig:
    return ProxyConfig(hard_rules=[], upstream_model="openai/test", upstream_api_key="sk-fake",
                       rocksdb_path=f"{tempfile.mkdtemp()}/rocksdb")


def test_lifespan_gets_the_deterministic_fake_by_default():
    with TestClient(create_app(_app_cfg())) as client:
        emb = client.app.state.embedder
        assert emb.model_identity == "test:fake-hash-64"
        assert emb.embed(["同一句话"]) == emb.embed(["同一句话"])


def test_without_the_default_fake_lifespan_hits_the_guard(monkeypatch):
    """阴性对照：换回真 `build_embedder` + socket 解析到部署根 ⇒ lifespan 撞守卫并留下点名。"""
    monkeypatch.setattr(app_factory, "build_embedder", _REAL_BUILD_EMBEDDER)
    monkeypatch.setattr(_emb_mod, "build_embedder", _REAL_BUILD_EMBEDDER)
    monkeypatch.setattr(embed_transport, "default_uds_path", lambda: _live()[0])
    monkeypatch.setenv("BLADEX_EMBED_BACKEND", "ipc")
    monkeypatch.delenv("BLADEX_EMBED_TRANSPORT", raising=False)
    monkeypatch.delenv("BLADEX_EMBED_ADDRESS", raising=False)
    since = len(embed_client._TEST_LIVE_VIOLATIONS)
    with TestClient(create_app(_app_cfg())):
        pass
    got = _take_violations(since)
    assert got, "没有默认假 embedder 时 lifespan 应当撞上守卫（否则守卫是摆设）"
    assert "test_without_the_default_fake_lifespan_hits_the_guard" in got[0]
