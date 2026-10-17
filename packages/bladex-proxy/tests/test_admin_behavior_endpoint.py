"""批 F3 验收剧本 · `GET /admin/behavior`（判据 F3.2）。

- 端点 200 + `flash_lag_s` 在（**没有读数时显示"没有"，不补 0**）；
- 端点读的是**当前那份文件**：同一个 tmp 根下换掉内容，读数跟着变（10-17 F5 前此条
  是"模型面与运维面同时变"；模型面 `bladex_flash_read` 退出注册后只剩运维面）；
- 空态 200 + `available=false` + 具名 `reason`（不是 404——端点在，数据不在）。
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime, timedelta

from bladex_core.behavior import TurnMeta, aggregate
from bladex_core.flash import STATE_FILE, behavior_path
from bladex_proxy.config import ProxyConfig
from bladex_proxy.server import create_app
from fastapi.testclient import TestClient

NOW = datetime(2026, 9, 22, 12, 0, tzinfo=UTC)


def _config() -> ProxyConfig:
    tmpdir = tempfile.mkdtemp()
    return ProxyConfig(auth_enabled=False, upstream_model="openai/test",
                       upstream_api_key="sk-fake",
                       rocksdb_path=f"{tmpdir}/rocksdb", index_path=f"{tmpdir}/index")


def _behavior(agents: int = 2, *, now: datetime | None = None):
    now = now or datetime.now(UTC)
    metas = [TurnMeta(principal="local", agent_id=f"agent-{a}", session_id=f"fp:s{a}",
                      ts=now - timedelta(minutes=i + 1), project_id=f"p-{a}",
                      project_name=f"proj-{a}", key=f"local/agent-{a}/fp:s{a}/{i}-{a}")
             for a in range(agents) for i in range(5)]
    return aggregate(metas, now=now)


def _seed(tmp_path, monkeypatch, *, lag_s: int | None = 7, agents: int = 2) -> str:
    monkeypatch.setenv("BLADEX_FLASH_PATH", str(tmp_path))
    path = behavior_path(str(tmp_path), "local")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(_behavior(agents).to_json())
    if lag_s is not None:
        with open(f"{tmp_path}/{STATE_FILE}", "w", encoding="utf-8") as f:
            json.dump({"behavior_status": {"flash_lag_s": lag_s, "lag_bound_s": 360,
                                           "lag_exceeded": False, "agents": agents}}, f)
    return path


def test_behavior_endpoint_returns_dashboard_view_and_flash_lag(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, lag_s=7)
    with TestClient(create_app(_config())) as client:
        r = client.get("/admin/behavior")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True and body["reason"] == ""
    assert body["status"]["flash_lag_s"] == 7
    # dashboard 档独有的字段（模型卡上没有）——证明这一面拿的是全字段视图
    agent0 = body["behavior"]["agents"][0]
    assert "active_hours" in agent0 and "variants" in agent0
    assert len(agent0["active_hours"]) == 24


def test_missing_profile_is_200_with_named_reason_not_404(tmp_path, monkeypatch):
    monkeypatch.setenv("BLADEX_FLASH_PATH", str(tmp_path / "nowhere"))
    with TestClient(create_app(_config())) as client:
        r = client.get("/admin/behavior")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is False and body["reason"] == "missing"
    assert "behavior" not in body


def test_no_status_section_reports_absence_not_zero(tmp_path, monkeypatch):
    """🔴 原则 13 层 4：`flash_lag_s` 没有读数 ≠ lag=0。"""
    _seed(tmp_path, monkeypatch, lag_s=None)
    with TestClient(create_app(_config())) as client:
        body = client.get("/admin/behavior").json()
    assert body["status"] == {}
    assert "flash_lag_s" not in body["status"]


def test_admin_face_follows_the_file_on_disk(tmp_path, monkeypatch, fresh_redis_stream):
    """同一根目录下换掉文件内容 ⇒ 端点读数跟着变（接线判据，不是函数判据）。"""
    _seed(tmp_path, monkeypatch, agents=2)

    def _agents() -> int:
        with TestClient(create_app(_config())) as client:
            admin = client.get("/admin/behavior").json()
        return len(admin["behavior"]["agents"])

    assert _agents() == 2
    _seed(tmp_path, monkeypatch, agents=4)
    # 第二个 app 读换过的文件、不是重启语义：换一条 stream，免等 4 s 实例判活窗口（批 GS）
    monkeypatch.setenv("BLADEX_REDIS_STREAM", fresh_redis_stream())
    assert _agents() == 4
