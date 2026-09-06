"""Beta T18–T20 dashboard 验收（服务面 + 数据端点；浏览器交互留用户机真机）。

钉住：/dashboard 从包资源正常出页（wheel 安装态可用）；页面不含内嵌密钥；
数据面走 /admin/*（auth 开启时无 key 401 = "无 key 访问被拒"）；
/admin/turns 列表端点（回放页数据源）prefix/分页/排序。
"""

from __future__ import annotations

import tempfile

from bladex_proxy.config import ProxyConfig
from bladex_proxy.models import Identity, Turn
from bladex_proxy.server import create_app
from fastapi.testclient import TestClient


def _make_config(auth: bool = False) -> ProxyConfig:
    tmpdir = tempfile.mkdtemp()
    return ProxyConfig(
        auth_enabled=auth,
        client_keys_raw="bladex-valid||test" if auth else "",
        hard_rules=["MUST be polite"],
        upstream_model="openai/test",
        upstream_api_key="sk-fake",
        rocksdb_path=f"{tmpdir}/rocksdb",
        index_path=f"{tmpdir}/index",
    )


def test_dashboard_serves_html_with_all_views():
    app = create_app(_make_config())
    with TestClient(app) as client:
        resp = client.get("/dashboard")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/html")
        html = resp.text
        # T18 状态总览 + T19 记忆/Matter 管理 + T20 图/回放 的关键部件都在
        for marker in ("/admin/status", "/admin/facts", "/admin/matters",
                       "/admin/sessions", "/admin/turns", "/admin/hard_rules",
                       "graph-canvas", "sessionStorage", "decision_meta",
                       "forgetFact", "matterOp"):
            assert marker in html, f"dashboard missing {marker}"


def test_dashboard_page_contains_no_secrets():
    """页面是无鉴权静态资源：不得内嵌任何 key/凭证。"""
    app = create_app(_make_config(auth=True))
    with TestClient(app) as client:
        html = client.get("/dashboard").text
        assert "bladex-valid" not in html
        # 数据面在 auth 开启时无 key 被拒（dashboard 的"登录"语义）
        assert client.get("/admin/status").status_code == 401


def test_admin_turns_index_prefix_and_order():
    app = create_app(_make_config())
    with TestClient(app) as client:
        ledger = app.state.hub
        for i, ts in enumerate(["2026-08-01T10:00:00+00:00",
                                "2026-08-03T10:00:00+00:00",
                                "2026-08-02T10:00:00+00:00"]):
            from datetime import datetime
            ledger.put(f"u1/agentA/s1/{i:04d}",
                   Turn(identity=Identity(user_id="u1", agent_id="agentA",
                                          session_id="s1"),
                        ts=datetime.fromisoformat(ts)))
        ledger.put("u1/agentB/s2/0000",
               Turn(identity=Identity(user_id="u1", agent_id="agentB",
                                      session_id="s2")))

        d = client.get("/admin/turns",
                       params={"prefix": "u1/agentA/s1/"}).json()
        assert d["total"] == 3
        # ts 降序
        keys = [t["key"] for t in d["turns"]]
        assert keys[0].endswith("0001")  # 08-03 最新
        # 分页
        d2 = client.get("/admin/turns",
                        params={"prefix": "u1/agentA/s1/", "limit": 2}).json()
        assert len(d2["turns"]) == 2 and d2["total"] == 3
        # 无 prefix 全量
        assert client.get("/admin/turns").json()["total"] == 4


class TestTabRegistration:
    """导航结构自洽：TABS / VIEWS / SUBS / IC 四张表必须对齐。

    加这条是因为差点漏了一步：`TABS` 里注册了 `ledgers`，`VIEWS` 里忘了写，
    页面**能显示导航项，一点就抛**——四张表各自看起来都对，只有对照才发现。
    纯字符串检查（页面是单文件 SPA，没有 JS 运行时可用）。
    """

    @staticmethod
    def _html() -> str:
        from importlib import resources
        return (resources.files("bladex_proxy") / "dashboard.html").read_text(
            encoding="utf-8")

    @staticmethod
    def _tab_ids(html: str) -> list[str]:
        import re
        block = re.search(r"const TABS=\[(.*?)\];", html, re.S)
        assert block, "TABS 声明找不到了——改了形态就得同步改这条守卫"
        ids = re.findall(r'\["([a-z]+)",', block.group(1))
        assert len(ids) >= 5, ids
        return ids

    def test_every_tab_has_a_view(self):
        html = self._html()
        import re
        views = re.search(r"const VIEWS=\{(.*?)\};", html, re.S)
        assert views
        registered = set(re.findall(r"(\w+):", views.group(1)))
        missing = [t for t in self._tab_ids(html) if t not in registered]
        assert not missing, f"TABS 里有、VIEWS 里没有（点了会抛）: {missing}"

    def test_every_tab_has_subtitle_and_icon(self):
        html = self._html()
        import re
        subs = re.search(r"const SUBS=\{(.*?)\};", html, re.S)
        ic = re.search(r"const IC=\{(.*?)\n\};", html, re.S)
        assert subs and ic
        has_sub = set(re.findall(r"(\w+):", subs.group(1)))
        has_ic = set(re.findall(r"(\w+):", ic.group(1)))
        for t in self._tab_ids(html):
            assert t in has_sub, f"{t} 缺副标题"
            assert t in has_ic, f"{t} 缺图标"

    def test_ledgers_tab_is_present(self):
        assert "ledgers" in self._tab_ids(self._html())
