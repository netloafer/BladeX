"""Beta T16/T17 bladex-mcp 工具验收（数据面 mock：不依赖运行中的 proxy）。

MCP inspector / Claude Code 真机剧本留用户机（发布验收 §4.5）；本文件钉住：
工具注册齐全、admin API 调用映射、proxy 不可达返回明确错误（不挂起/不抛）。
"""

from __future__ import annotations

import asyncio
import urllib.error

import pytest

pytest.importorskip("mcp", reason="mcp SDK 未安装（uv sync 或 uv pip install -e packages/bladex-mcp）")

import bladex_mcp.server as srv  # noqa: E402


def test_all_tools_registered():
    tools = asyncio.run(srv.server.list_tools())
    names = {t.name for t in tools}
    assert {"memory_search", "matter_list", "matter_get", "hard_rules",
            "session_recall", "ledger_list", "ledger_read",
            "remember"} <= names


def test_memory_search_maps_and_formats(monkeypatch):
    calls = []

    def fake(method, path, params=None, body=None):
        calls.append((method, path, params, body))
        return {"facts": [{"id": "f1", "kind": "event",
                           "content": "proxy 502 已修复",
                           "source_session": "u1/hermes/s1/"}], "total": 1}

    monkeypatch.setattr(srv, "_http_json", fake)
    out = srv.memory_search("proxy 502", k=5)
    assert calls[0][:2] == ("GET", "/admin/facts")
    assert calls[0][2] == {"q": "proxy 502", "limit": 5}
    assert "proxy 502 已修复" in out and "id=f1" in out


def test_matter_tools(monkeypatch):
    def fake(method, path, params=None, body=None):
        if path == "/admin/matters":
            return {"matters": [{"matter_id": "m1", "status": "active",
                                 "title": "Beta 发布", "summary": "s"}]}
        return {"matter": {"matter_id": "m1", "status": "active",
                           "title": "Beta 发布", "summary": "s",
                           "participants": [{"agent_id": "hermes"}]},
                "facts": [{"id": "f1", "kind": "event", "content": "x"}],
                "session_keys": ["u1/hermes/s1/"]}

    monkeypatch.setattr(srv, "_http_json", fake)
    assert "Beta 发布" in srv.matter_list()
    detail = srv.matter_get("m1")
    assert "hermes" in detail and "u1/hermes/s1/" in detail   # open_issues 已删（F0.3 H1 销账）


def test_hard_rules_and_session_recall(monkeypatch):
    def fake(method, path, params=None, body=None):
        if path == "/admin/hard_rules":
            return {"hard_rules": ["MUST 中文回复"], "count": 1}
        return {"facts": [
            {"content": "讨论过 502", "source_session": "u1/hermes/s1/"},
            {"content": "复盘了路由", "source_session": "u1/codex/s2/"},
        ]}

    monkeypatch.setattr(srv, "_http_json", fake)
    assert "MUST 中文回复" in srv.hard_rules()
    out = srv.session_recall("502")
    assert "u1/hermes/s1/" in out and "u1/codex/s2/" in out


def test_ledger_list_maps_and_formats(monkeypatch):
    calls = []

    def fake(method, path, params=None, body=None):
        calls.append((method, path, params, body))
        return {
            "enabled": True,
            "ledgers": [{
                "ledger_id": "ldg-123",
                "status": "active",
                "title": "Ship dashboard",
                "goal": "Ship V2.",
                "matter_id": "m-456",
                "updated_at": "2026-08-27T12:34:56+08:00",
                "entry_counts": {"verified": 2, "open": 1},
                "active_in": [{"agent": "codex", "project": "BladeX"}],
            }],
        }

    monkeypatch.setattr(srv, "_http_json", fake)
    out = srv.ledger_list()

    assert calls == [("GET", "/admin/ledgers", None, None)]
    assert srv.ledger_list(active_only=True) == out
    assert "ldg-123 [active] Ship dashboard" in out
    assert "  Goal: Ship V2." in out
    assert "  Matter: m-456" in out
    assert "  Entries: verified: 2, open: 1" in out
    assert "  Updated: 2026-08-27T12:34:56+08:00" in out
    assert "  Active in: codex@BladeX" in out


def test_ledger_list_active_only_filters(monkeypatch):
    def fake(method, path, params=None, body=None):
        return {
            "enabled": True,
            "ledgers": [
                {"ledger_id": "ldg-active", "active_in": [
                    {"agent": "codex", "project": "BladeX"}]},
                {"ledger_id": "ldg-idle", "active_in": []},
            ],
        }

    monkeypatch.setattr(srv, "_http_json", fake)

    out = srv.ledger_list(active_only=True)

    assert "ldg-active" in out
    assert "ldg-idle" not in out


def test_ledger_list_empty_and_disabled(monkeypatch):
    monkeypatch.setattr(
        srv, "_http_json",
        lambda method, path, params=None, body=None: {
            "enabled": True, "ledgers": []})
    assert srv.ledger_list() == "No task ledgers."

    monkeypatch.setattr(
        srv, "_http_json",
        lambda method, path, params=None, body=None: {"enabled": False})
    assert srv.ledger_list() == "Ledger agency is not enabled."


def test_ledger_read_maps_and_returns_markdown(monkeypatch):
    calls = []

    def fake(method, path, params=None, body=None):
        calls.append((method, path, params, body))
        return {"markdown": "# Task ledger"}

    monkeypatch.setattr(srv, "_http_json", fake)

    assert srv.ledger_read("ldg/123") == "# Task ledger"
    assert calls == [("GET", "/admin/ledgers/ldg%2F123", None, None)]


def test_ledger_read_empty_not_found_and_unreachable(monkeypatch):
    monkeypatch.setattr(
        srv, "_http_json",
        lambda method, path, params=None, body=None: {
            "markdown": "", "ledger": None})
    assert srv.ledger_read("ldg-missing") == (
        "No task ledger content available.")

    def raise_not_found(method, path, params=None, body=None):
        raise RuntimeError("HTTP 404: Not Found")

    monkeypatch.setattr(srv, "_http_json", raise_not_found)
    assert srv.ledger_read("ldg-missing") == "ERROR: HTTP 404: Not Found"


def test_http_json_uses_detail_from_4xx(monkeypatch):
    class Response:
        def read(self):
            return b'{"status": 404, "detail": "Ledger not found: ldg-x"}'

        def close(self):
            pass

    error = urllib.error.HTTPError(
        "http://proxy/admin/ledgers/ldg-x", 404, "Not Found", {}, Response())

    def urlopen(req, timeout=0):
        raise error

    monkeypatch.setattr(srv.urllib.request, "urlopen", urlopen)

    with pytest.raises(RuntimeError, match="Ledger not found: ldg-x"):
        srv._http_json("GET", "/admin/ledgers/ldg-x")


def test_remember_posts_to_admin_facts(monkeypatch):
    calls = []

    def fake(method, path, params=None, body=None):
        calls.append((method, path, body))
        return {"status": "remembered", "fact_id": "abc123"}

    monkeypatch.setattr(srv, "_http_json", fake)
    out = srv.remember("用户的生产环境是 arm64", scope="")
    assert calls[0][0] == "POST" and calls[0][1] == "/admin/facts"
    assert calls[0][2] == {"content": "用户的生产环境是 arm64", "scope": ""}
    assert "abc123" in out


def test_proxy_unreachable_returns_error_text(monkeypatch):
    monkeypatch.setenv("BLADEX_PROXY_URL", "http://127.0.0.1:1")  # 必不可达
    out = srv.memory_search("anything")
    assert out.startswith("ERROR:")
    assert "unreachable" in out
    # 全部只读工具同语义（不挂起、不抛）
    assert srv.matter_list().startswith("ERROR:")
    assert srv.hard_rules().startswith("ERROR:")
    assert srv.ledger_list().startswith("ERROR:")
    assert srv.ledger_read("ldg-123").startswith("ERROR:")
    assert srv.remember("x").startswith("ERROR:")
