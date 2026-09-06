"""`/admin/ledgers` 只读视图（V-L5 附卡）。

立卡背景：在此之前 **dashboard 完全看不到账本**——只有 probe 脚本能看，
而 0.1.0 gate 的演示剧本之一就是「账本机制稳定运转」。

钉住：
  ① 池来自 Hub 事件重放 ⇒ **重启存活**（不是进程内内存）；
  ② `active_in` 是 **(agent, project) 拆好的对象**，前端拿不到分隔符；
  ③ 一本账本可被**多个 agent** 激活（跨 agent 交接是北极星要接住的场景）；
  ④ Matter 锚出现在列里（V-L5 的可见性落点）；
  ⑤ 未知 id → 404 且 body 是 `{status, detail}`（dashboard `api()` 认这个形状，
     2026-08-26 那次 "HTTP 400 点了没反应" 就是错认成 `error.message` 造成的）；
  ⑥ agency 缺席 ⇒ `enabled: false`，不抛。
"""

from __future__ import annotations

import pytest
from bladex_core.ledger import new_ledger
from bladex_core.ledger_runtime import (
    ActivationTable,
    activation_scope,
    bind_matter,
    ledger_anchor_matter_id,
    split_scope,
)


class _StubAgency:
    def __init__(self, pool, bindings):
        self.pool = pool
        self.activation = ActivationTable(debounce_turns=0)
        self.activation.restore(bindings)


@pytest.fixture
def client_with_ledgers(monkeypatch):
    from fastapi.testclient import TestClient

    from bladex_proxy.config import ProxyConfig
    from bladex_proxy.server import create_app

    monkeypatch.setenv("BLADEX_AUTH_ENABLED", "false")
    app = create_app(ProxyConfig())

    parent = new_ledger(ledger_id="ldg-parent00001", title="主任务",
                        goal="帮我把路由回归修掉", goal_source="user",
                        created_at="2026-08-26T10:00:00Z")
    parent = bind_matter(parent, ledger_anchor_matter_id("ldg-parent00001"),
                         updated_at="2026-08-26T11:00:00Z")
    child = new_ledger(ledger_id="ldg-child000001", title="子任务",
                       parent_ledger_id="ldg-parent00001",
                       created_at="2026-08-26T10:30:00Z")
    idle = new_ledger(ledger_id="ldg-idle0000001", title="没人激活的",
                      created_at="2026-08-25T09:00:00Z")
    pool = {l.ledger_id: l for l in (parent, child, idle)}
    # 同一本被两个 agent 激活 —— 跨 agent 交接的真实形态
    bindings = {activation_scope("hermes:default", ""): "ldg-parent00001",
                activation_scope("codex", ""): "ldg-parent00001",
                activation_scope("claude-code", ""): "ldg-child000001"}

    with TestClient(app) as c:
        c.app.state.agency = _StubAgency(pool, bindings)
        yield c


def test_lists_pool_with_anchor_and_scopes(client_with_ledgers):
    d = client_with_ledgers.get("/admin/ledgers").json()
    assert d["enabled"] is True and d["total"] == 3
    by_id = {r["ledger_id"]: r for r in d["ledgers"]}

    p = by_id["ldg-parent00001"]
    assert p["matter_id"] == ledger_anchor_matter_id("ldg-parent00001")
    # ③ 两个 agent 同时激活同一本
    assert {s["agent"] for s in p["active_in"]} == {"hermes:default", "codex"}
    # ② 拆好的对象，不是带分隔符的裸串
    assert all(set(s) == {"agent", "project"} for s in p["active_in"])
    assert all(s["project"] == "Global" for s in p["active_in"])

    assert by_id["ldg-child000001"]["parent_ledger_id"] == "ldg-parent00001"
    assert by_id["ldg-idle0000001"]["active_in"] == []


def test_active_ledgers_sort_first(client_with_ledgers):
    """观察期最想先看到"现在在跑哪本"。"""
    rows = client_with_ledgers.get("/admin/ledgers").json()["ledgers"]
    assert rows[-1]["ledger_id"] == "ldg-idle0000001", [r["ledger_id"] for r in rows]


def test_scope_separator_never_reaches_the_client(client_with_ledgers):
    """🔴 分隔符是 core 的实现细节——响应体里一个字节都不该有。"""
    raw = client_with_ledgers.get("/admin/ledgers").content
    assert b"\x1f" not in raw


def test_detail_returns_sections_markdown_and_children(client_with_ledgers):
    d = client_with_ledgers.get("/admin/ledgers/ldg-parent00001").json()
    assert d["ledger"]["goal"] == "帮我把路由回归修掉"
    assert d["ledger"]["goal_source"] == "user"
    assert "ldg-parent00001" in d["markdown"]
    assert [c["ledger_id"] for c in d["children"]] == ["ldg-child000001"]
    assert {s["agent"] for s in d["active_in"]} == {"hermes:default", "codex"}


def test_unknown_ledger_is_404_in_dashboard_error_shape(client_with_ledgers):
    """⑤ dashboard 的 `api()` 读 `detail`/`status`——形状错了界面就"点了没反应"。"""
    r = client_with_ledgers.get("/admin/ledgers/ldg-nope")
    assert r.status_code == 404
    body = r.json()
    assert body["status"] == "not_found" and "ldg-nope" in body["detail"]


def test_no_agency_is_disabled_not_error(monkeypatch):
    from fastapi.testclient import TestClient

    from bladex_proxy.config import ProxyConfig
    from bladex_proxy.server import create_app

    monkeypatch.setenv("BLADEX_AUTH_ENABLED", "false")
    with TestClient(create_app(ProxyConfig())) as c:
        c.app.state.agency = None
        d = c.get("/admin/ledgers").json()
        assert d == {"enabled": False, "ledgers": [], "total": 0}


class TestSplitScope:
    """`split_scope` 是 `activation_scope` 的逆——两者必须严格互逆。"""

    @pytest.mark.parametrize("agent,project", [
        ("hermes:default", ""), ("codex", "BladeX"),
        ("unknown-9eb9e3a9", "Global"), ("Pi", "a/b:c"),
    ])
    def test_roundtrip(self, agent, project):
        got = split_scope(activation_scope(agent, project))
        assert got == (agent, project or "Global")

    def test_malformed_falls_back_to_global(self):
        assert split_scope("just-an-agent") == ("just-an-agent", "Global")
        assert split_scope("") == ("", "Global")
