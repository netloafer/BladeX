"""五类记忆 admin 只读面验收：文件索引 / 项目 / 画像 / 工具习惯 + Matter topic_keys 暴露。

普通会话类（facts/matters/sessions）已由 test_admin_read_api 覆盖；本文件钉住其余四类
（ADR-0028 E7.1–E7.4）在 admin 面（dashboard/CLI/MCP 共同地基）可见，以及：

  - /admin/matters 序列化必须带 topic_keys（2026-08-14 主题键机制的管理面出口）；
  - 工具习惯 agent 间隔离（codex 的统计不出现在 hermes 的查询里）；
  - Memory Index 缺失 → 503（与其它 admin 读端点同语义）。

Memory Index 用测试自建可写句柄（embedder=None → 文件语义搜索退化为空列表），hermetic。
"""

from __future__ import annotations

import tempfile

from bladex_core.matter import Matter
from bladex_core.project import Project
from bladex_proxy.config import ProxyConfig
from bladex_proxy.server import create_app
from bladex_proxy.storage.memory_index import MemoryIndex
from fastapi.testclient import TestClient


def _make_config() -> ProxyConfig:
    tmpdir = tempfile.mkdtemp()
    return ProxyConfig(
        auth_enabled=False,
        hard_rules=["MUST be polite"],
        upstream_model="openai/test",
        upstream_api_key="sk-fake",
        rocksdb_path=f"{tmpdir}/rocksdb",
        index_path=f"{tmpdir}/index",
    )


def _make_index() -> MemoryIndex:
    tmpdir = tempfile.mkdtemp()
    index = MemoryIndex(f"{tmpdir}/p2w", embedder=None, read_only=False)
    index.open()
    index._ensure_meta_db()
    return index


# ── 文件索引（E7.1）──────────────────────────────────────────────


def test_admin_files_empty_and_search_degrade():
    """无 files 表 → 空列表 200；q 非空且 embedder 缺失 → search 模式空结果（不 500）。"""
    app = create_app(_make_config())
    with TestClient(app) as client:
        index = _make_index()
        app.state.index = index
        try:
            d = client.get("/admin/files").json()
            assert d["files"] == [] and d["total"] == 0 and d["mode"] == "list"
            d = client.get("/admin/files", params={"q": "报告"}).json()
            assert d["files"] == [] and d["mode"] == "search"
            assert client.get("/admin/files/nope").status_code == 404
        finally:
            index.close()


# ── 项目（E7.2）──────────────────────────────────────────────────


def test_admin_projects_list_and_detail():
    app = create_app(_make_config())
    with TestClient(app) as client:
        index = _make_index()
        app.state.index = index
        try:
            index.upsert_project(Project(
                project_id="p1", name="BladeX", root_path="/home/u/dev/BladeX",
                languages=["python"], progress_note="beta readiness",
                agent_ids=["claude-code"]))
            d = client.get("/admin/projects").json()
            assert d["total"] == 1
            assert d["projects"][0]["name"] == "BladeX"
            assert d["projects"][0]["languages"] == ["python"]

            p = client.get("/admin/projects/p1").json()
            assert p["root_path"] == "/home/u/dev/BladeX"
            assert client.get("/admin/projects/nope").status_code == 404
        finally:
            index.close()


# ── 画像（E7.3）──────────────────────────────────────────────────


def test_admin_profiles_overview_and_rulefile_detail():
    app = create_app(_make_config())
    with TestClient(app) as client:
        index = _make_index()
        app.state.index = index
        try:
            index._profile_put("profile/user_md", {"content": "# USER"})
            index._profile_put("profile/agent_md/codex", {"content": "# codex habits"})
            index._profile_put("rulefile/codex/AGENTS.md",
                               {"content": "# rules", "content_hash": "h1",
                                "updated_at": "2026-08-18T00:00:00+00:00"})
            d = client.get("/admin/profiles").json()
            assert d["user_md"] == "# USER"
            assert d["agent_count"] == 1
            a = d["agents"][0]
            assert a["agent_base"] == "codex"
            assert a["agent_md"] == "# codex habits"
            assert a["rule_files"] == ["AGENTS.md"]

            r = client.get("/admin/profiles/codex/rulefiles/AGENTS.md").json()
            assert r["content"] == "# rules" and r["name"] == "AGENTS.md"
            assert client.get(
                "/admin/profiles/codex/rulefiles/NOPE.md").status_code == 404
        finally:
            index.close()


def test_admin_profiles_empty_index_ok():
    """全新库：user_md 空串 + agents 空列表（空状态而非报错）。"""
    app = create_app(_make_config())
    with TestClient(app) as client:
        index = _make_index()
        app.state.index = index
        try:
            d = client.get("/admin/profiles").json()
            assert d["user_md"] == "" and d["agents"] == []
        finally:
            index.close()


# ── 工具习惯（E7.4）──────────────────────────────────────────────


def test_admin_tools_per_agent_isolated():
    app = create_app(_make_config())
    with TestClient(app) as client:
        index = _make_index()
        app.state.index = index
        try:
            index._profile_put("profile/tool/codex/bash",
                               {"calls": 10, "ok": 8, "err": 2,
                                "last_err_class": "not found"})
            index._profile_put("profile/tool/hermes/web_search",
                               {"calls": 3, "ok": 3})
            d = client.get("/admin/tools").json()
            assert set(d["agents"].keys()) == {"codex", "hermes"}
            assert d["agents"]["codex"][0]["name"] == "bash"
            assert d["agents"]["codex"][0]["calls"] == 10

            # 隔离：按 agent 查询只回该 agent（codex 的统计不混进 hermes）
            d = client.get("/admin/tools", params={"agent": "hermes"}).json()
            assert list(d["agents"].keys()) == ["hermes"]
            assert d["agents"]["hermes"][0]["name"] == "web_search"
        finally:
            index.close()


# ── Matter topic_keys 暴露 ───────────────────────────────────────


def test_admin_matters_expose_topic_keys():
    """主题键（2026-08-14 机制）必须出现在 /admin/matters 序列化里，dashboard 消费。"""
    app = create_app(_make_config())
    with TestClient(app) as client:
        index = _make_index()
        app.state.index = index
        try:
            index.add_matter(Matter(matter_id="m1", title="泰安项目",
                                    topic_keys={"泰安": 3, "部署": 1}))
            d = client.get("/admin/matters").json()
            row = next(m for m in d["matters"] if m["matter_id"] == "m1")
            assert row["topic_keys"] == {"泰安": 3, "部署": 1}
        finally:
            index.close()


# ── 降级语义 ─────────────────────────────────────────────────────


def test_admin_five_classes_503_without_index():
    app = create_app(_make_config())
    with TestClient(app) as client:
        app.state.index = None
        for path in ("/admin/files", "/admin/projects", "/admin/profiles",
                     "/admin/tools"):
            assert client.get(path).status_code == 503, path
