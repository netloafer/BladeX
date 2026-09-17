"""V-A3（批 F0.2，2026-09-06）：`export` / `admin_read` 两个边缘模块进注册表，默认全开 = 现状。

矩阵（`none` 零差异的延伸）：
- 关 `export` ⇒ consolidator **不构造** 导出 worker，而 `sync_control` 心跳一行不动（B10 仪器消费）；
  CLI `connector run/status/reset` / `export` / `import` 入口打 `export module disabled` 退出 2。
- 关 `admin_read` ⇒ 只读面（`GET /admin/status` 404、`GET /admin/matters` 405——同路径写端点仍在），写面（`POST /admin/ledgers/user-edit`）仍 200。
- `mcp` **没有**加进注册表：proxy 进程与 bladex-mcp 零耦合点，没有调用点的开关就是假开关（本文件钉住这条）。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from bladex_proxy.modules import MODULE_SPECS, module_enabled, validate_modules

_PKG = Path(__file__).resolve().parents[1] / "bladex_proxy"


class TestRegistry:
    def test_two_edge_modules_default_on(self):
        assert MODULE_SPECS["export"].default_enabled is True
        assert MODULE_SPECS["admin_read"].default_enabled is True
        sw = validate_modules(env={})
        assert sw["export"] is True and sw["admin_read"] is True

    def test_mcp_is_not_a_module_because_proxy_has_no_coupling_point(self):
        assert "mcp" not in MODULE_SPECS
        hits = [p for p in _PKG.rglob("*.py")
                if "import bladex_mcp" in p.read_text(encoding="utf-8")
                or "from bladex_mcp" in p.read_text(encoding="utf-8")]
        assert not hits, f"proxy 里出现了 bladex_mcp 调用点 ⇒ 该给 mcp 加 ModuleSpec 了：{hits}"

    def test_switch_off_is_honoured(self, monkeypatch):
        monkeypatch.setenv("BLADEX_MODULE_EXPORT", "0")
        monkeypatch.setenv("BLADEX_MODULE_ADMIN_READ", "0")
        assert module_enabled("export") is False and module_enabled("admin_read") is False


class TestConsolidatorExportGate:
    """源码结构断言（consolidator.main 要 Redis/RocksDB，行为测试在 sandbox 起不来；
    这里钉的是"门只挡 worker 构造、不挡心跳"这条结构事实）。"""

    @staticmethod
    def _src() -> str:
        return (_PKG / "consolidator.py").read_text(encoding="utf-8")

    def test_gate_precedes_worker_construction(self):
        src = self._src()
        i_gate = src.index('module_enabled("export")')
        i_build = src.index("build_worker_from_config(")
        assert i_gate < i_build, "模块门必须在 build_worker_from_config 之前"
        assert 'logger.info("export_module_disabled"' in src

    def test_heartbeat_is_outside_the_gate(self):
        """`sync_control` 心跳与 export 门互不相干（B10 仪器消费；F0.2 硬约束「一行不动」）。"""
        src = self._src()
        i_gate = src.index('module_enabled("export")')
        i_hb = src.index("make_heartbeat_writer")
        between = src[i_gate:i_hb]
        # 门的 if/else 块在心跳之前就结束了：心跳行不在门的 else 分支缩进内
        assert "def _run_manual" in between, "心跳应在导出块之后的独立段落里"
        hb_line = next(ln for ln in src.splitlines() if "make_heartbeat_writer" in ln)
        assert not hb_line.startswith("            "), "心跳被缩进进了 export 门的分支"


class TestCliExportGate:
    @pytest.mark.parametrize("argv", [
        ["connector", "status"], ["connector", "reset", "--yes"], ["connector", "run"],
        ["export", "x.jsonl"], ["import", "x.jsonl"],
    ])
    def test_off_exits_2_with_one_line(self, argv, monkeypatch, tmp_path, capsys):
        from bladex_proxy import cli
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("BLADEX_MODULE_EXPORT", "0")
        rc = cli.main(argv)
        assert rc == 2
        err = capsys.readouterr().err
        assert "export module disabled" in err and "BLADEX_MODULE_EXPORT=0" in err

    def test_single_gate_implementation(self):
        src = (_PKG / "cli" / "ops_cmds.py").read_text(encoding="utf-8")
        assert src.count("def _export_module_gate(") == 1
        assert src.count("if not _export_module_gate():") == 5, "五个入口（connector×3 / export / import）都要过同一道门"
        assert src.count('module_enabled("export")') == 1, "门只许有一份实现"


class TestAdminReadGate:
    @staticmethod
    def _client(monkeypatch):
        from bladex_proxy.config import ProxyConfig
        from bladex_proxy.server import create_app
        from fastapi.testclient import TestClient
        monkeypatch.setenv("BLADEX_AUTH_ENABLED", "false")
        return TestClient(create_app(ProxyConfig()))

    def test_default_on_read_face_present(self, monkeypatch):
        with self._client(monkeypatch) as c:
            paths = {r.path for r in c.app.routes}
            assert "/admin/matters" in paths and "/dashboard" in paths and "/admin/status" in paths
            assert c.get("/admin/matters").status_code != 404

    def test_off_read_face_404_write_face_still_200(self, monkeypatch):
        from bladex_core.ledger import new_ledger
        monkeypatch.setenv("BLADEX_MODULE_ADMIN_READ", "0")
        with self._client(monkeypatch) as c:
            paths = {r.path for r in c.app.routes}
            assert "/dashboard" not in paths and "/admin/status" not in paths
            # `/admin/matters` 同路径还有写端点 POST ⇒ GET 是 405（路由表里已无 GET），
            # 纯只读路径 `/admin/status` 才是 404；两种都是"只读面没登记"。
            assert c.get("/admin/matters").status_code in (404, 405), "只读面必须没登记"
            assert c.get("/admin/status").status_code == 404
            assert c.get("/dashboard").status_code == 404

            class _Agency:
                pool: dict = {}
            c.app.state.agency = _Agency()
            led = new_ledger(ledger_id="ldg-gate000001", title="写面照常",
                             created_at="2026-09-06T00:00:00Z")
            r = c.post("/admin/ledgers/user-edit", json={"ledger": led.model_dump(mode="json")})
            assert r.status_code == 200, r.text
            # 写端点 POST /admin/matters 仍登记（同一路径只关 GET）
            assert c.post("/admin/matters", json={}).status_code != 404


def test_env_example_documents_both(monkeypatch):
    text = (Path(__file__).resolve().parents[3] / "config" / ".env.example").read_text(encoding="utf-8")
    assert "BLADEX_MODULE_EXPORT=1" in text and "BLADEX_MODULE_ADMIN_READ=1" in text
