"""MQ-I30（服务端一半，批 Q3A）：按 project 墓碑不再占住事件循环。

修前：`POST /admin/turns/tombstone-by-project` 在请求处理里同步逐轮写墓碑（live ≈3 轮/s），
期间 proxy 事件循环被占、`/ready` 无响应。修后：影响面预览在线程里算，写墓碑交后台线程，
端点立即回 `submitted` + `job_id`，结果经 `GET /admin/jobs/{job_id}` 取；CLI 轮询到完成再打印。

判据：① 500 轮规模执行期间 `/ready` p99 < 1 s；② 墓碑结果（条数、级联读数、逐 project 行）
与修前同一代码路径逐项相等；③ 三道闸（A dry-run 确认 / B 协议版本 / C 未知参数）照过；
④ 阴性对照：写墓碑改回在请求里同步跑 ⇒ ① 红（结算稿 report-batch-Q3A-20260929.md 记读数）。
"""

from __future__ import annotations

import os
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from bladex_proxy.admin_contract import ADMIN_CONTRACT_FIELD, ADMIN_CONTRACT_VERSION
from bladex_proxy.models import (
    Identity,
    SessionIdSource,
    TombstoneTargetType,
    Turn,
    TurnStatus,
)
from bladex_proxy.storage.memory_hub import MemoryHub

_PA = "p-i30projaaaa"
_PB = "p-i30projbbbb"
_N = 250  # 每个 project 250 轮，合计 500
_BASE = datetime(2026, 9, 29, tzinfo=UTC)


def _seed(hub: MemoryHub, project: str, n: int) -> list[str]:
    keys = []
    for i in range(n):
        ident = Identity(
            user_id="u1",
            agent_id="codex",
            session_id=f"s-{project}",
            turn_index=i,
            session_id_source=SessionIdSource.FINGERPRINT,
            project_id=project,
            project_name=project,
        )
        key = ident.storage_key(f"{1790000000000 + i}-{project[-4:]}")
        hub.put(
            key,
            Turn(
                identity=ident,
                model="test",
                status=TurnStatus.OK,
                request_messages=[{"role": "user", "content": f"t{i}"}],
                ts=_BASE + timedelta(seconds=i),
            ),
        )
        keys.append(key)
    return keys


@pytest.fixture
def app_client(tmp_path: Path, monkeypatch, isolated_home):
    from bladex_proxy.config import ProxyConfig
    from bladex_proxy.server import create_app
    from fastapi.testclient import TestClient

    hub_path = tmp_path / "hub"
    hub = MemoryHub(hub_path)
    hub.open()
    keys = {_PA: _seed(hub, _PA, _N), _PB: _seed(hub, _PB, _N)}
    hub.close()
    monkeypatch.setenv("BLADEX_AUTH_ENABLED", "false")
    monkeypatch.setenv("BLADEX_ROCKSDB_PATH", str(hub_path))
    monkeypatch.setenv("BLADEX_INDEX_PATH", str(tmp_path / "index"))
    app = create_app(ProxyConfig())
    with TestClient(app) as c:
        live_hub: MemoryHub = c.app.state.hub
        real = live_hub.append_tombstone

        def slow_append(*a, **k):  # live 实测 ≈3 轮/s；这里压到 5 ms/轮，500 轮 ≈ 2.5 s
            time.sleep(0.005)
            return real(*a, **k)

        monkeypatch.setattr(live_hub, "append_tombstone", slow_append)
        yield c, keys


def _run_and_probe_ready(c, body: dict) -> tuple[list[float], dict]:
    """在另一个线程发真跑请求，主线程持续打 /ready 计时，直到结果落定。"""
    box: dict = {}

    def fire() -> None:
        box["resp"] = c.post("/admin/turns/tombstone-by-project", json=body).json()

    t = threading.Thread(target=fire)
    t.start()
    lat: list[float] = []
    deadline = time.time() + 60
    result = None
    while time.time() < deadline:
        t0 = time.perf_counter()
        c.get("/ready")
        lat.append(time.perf_counter() - t0)
        resp = box.get("resp")
        if resp is not None:
            if resp.get("status") != "submitted":  # 同步形态：请求返回即完成
                result = resp
                break
            job = c.get(f"/admin/jobs/{resp['job_id']}").json()["job"]
            if job["state"] != "running":
                assert job["state"] == "done", job
                result = job["result"]
                break
        time.sleep(0.02)
    t.join(timeout=60)
    assert result is not None, "tombstone run never finished"
    return lat, result


def test_ready_stays_responsive_while_tombstoning_500_turns(app_client) -> None:
    c, keys = app_client
    lat, result = _run_and_probe_ready(c, {"project_ids": [_PA, _PB], "dry_run": False})
    lat.sort()
    p99 = lat[min(len(lat) - 1, int(len(lat) * 0.99))]
    assert len(lat) >= 10, f"only {len(lat)} /ready probes landed during the run"
    assert p99 < 1.0, f"/ready p99 {p99:.2f}s during the run (max {lat[-1]:.2f}s)"
    assert result["status"] == "tombstoned" and result["total_turns"] == 2 * _N


def test_result_equals_the_pre_fix_synchronous_path(app_client) -> None:
    """② 逐项相等：_PA 走新端点；_PB 按修前端点的原样代码路径直接调（同一 app、同一 index）。"""
    from bladex_proxy.ledger_memory import tombstone_ledger_memory
    from bladex_proxy.project_purge import project_scope_preview

    c, keys = app_client
    _lat, new = _run_and_probe_ready(c, {"project_ids": [_PA], "dry_run": False})

    hub = c.app.state.hub
    index = getattr(c.app.state, "index", None)
    prev = project_scope_preview(hub, [_PB])[_PB]
    old_row = {
        "project_id": _PB,
        "project_names": prev["project_names"],
        "turn_count": prev["turn_count"],
        "agents": prev["agents"],
        "first_ts": prev["first_ts"],
        "last_ts": prev["last_ts"],
        "ledgers": prev["ledgers"],
        "memory": tombstone_ledger_memory(hub, index, prev["turn_keys"]),
    }

    (new_row,) = new["projects"]
    assert set(new_row) == set(old_row)
    for k in ("turn_count", "agents", "ledgers", "memory"):
        assert new_row[k] == old_row[k], k
    assert new_row["memory"]["turn_tombstones"] == _N
    assert set(new) == {"status", "dry_run", "projects", "total_turns", "backup_age_days", "hint"}
    stones = {
        t.target_key for _k, t in hub.scan_tombstones() if t.target_type == TombstoneTargetType.TURN
    }
    assert stones == set(keys[_PA]) | set(keys[_PB])


def test_dry_run_unchanged_and_unknown_job_404(app_client) -> None:
    c, keys = app_client
    d = c.post(
        "/admin/turns/tombstone-by-project", json={"project_ids": [_PA], "dry_run": True}
    ).json()
    assert d["status"] == "dry_run" and d["dry_run"] is True and "job_id" not in d
    assert d["projects"][0]["turn_count"] == _N and "memory" not in d["projects"][0]
    # 闸 C：未知 query 参数照旧 400
    r = c.post(
        "/admin/turns/tombstone-by-project", params={"dry_run": "true"}, json={"project_ids": [_PA]}
    )
    assert r.status_code == 400 and "unknown query parameter" in r.text
    assert c.get("/admin/jobs/tbp-nosuchjob").status_code == 404
    hub = c.app.state.hub
    assert not any(t.target_type == TombstoneTargetType.TURN for _k, t in hub.scan_tombstones())


def test_contract_version_bumped_for_the_protocol_change() -> None:
    """闸 B：真跑响应形状变了（submitted + job）⇒ 协议版本必须 +1，旧 CLI 被拒而不是误读。"""
    assert ADMIN_CONTRACT_VERSION == 2


def test_failed_tombstone_reports_error_and_freezes_done(app_client, monkeypatch) -> None:
    """服务端抛错 ⇒ job 落 failed（不是 done）：error 带异常文本、done 停在出错前进度。

    反向变异（MQ-I28 同款）：把 _run() 的 except 分支改成 job["state"]="done" ⇒ 本用例红
    ——原剧本没有这条路径的用例，改了照样全绿。
    """
    import bladex_proxy.ledger_memory as ledger_memory

    c, keys = app_client
    # _PA 排前成功、_PB 排后抛错（preview 按请求顺序逐 project 跑）：
    # done 停在 _PA 的 250 才是"冻结在出错前进度"——全 0 分不出"没跑"和"跑了半截"。
    # 端点在请求处理时才 `from bladex_proxy.ledger_memory import tombstone_ledger_memory`，
    # 打源模块属性即可；_run() 经端点局部名引用，monkeypatch 在 POST 前生效即覆盖两笔调用。
    orig = ledger_memory.tombstone_ledger_memory

    def split(hub, index, turn_keys):
        if split.calls == 0:
            split.calls += 1
            return orig(hub, index, turn_keys)  # _PA：真写
        raise RuntimeError("disk full")  # _PB：抛错

    split.calls = 0
    monkeypatch.setattr(ledger_memory, "tombstone_ledger_memory", split)

    resp = c.post(
        "/admin/turns/tombstone-by-project", json={"project_ids": [_PA, _PB], "dry_run": False}
    ).json()
    assert resp["status"] == "submitted"
    job_id = resp["job_id"]
    deadline = time.time() + 30
    job = None
    while time.time() < deadline:
        job = c.get(f"/admin/jobs/{job_id}").json()["job"]
        if job["state"] != "running":
            break
        time.sleep(0.02)
    assert job is not None and job["state"] == "failed", job
    assert "RuntimeError: disk full" in job["error"]
    assert job["done"] == _N and job["total"] == 2 * _N, "done 停在 _PA 完成后、_PB 出错前"
    assert job["finished_at"] is not None and job["result"] is None
    assert "_task" not in job, "内部键不外泄"

    # _PA 真写了墓碑（进度真实发生），_PB 一条没写
    hub = c.app.state.hub
    stones = {
        t.target_key for _k, t in hub.scan_tombstones() if t.target_type == TombstoneTargetType.TURN
    }
    assert set(keys[_PA]) <= stones and not (set(keys[_PB]) & stones)


# ── CLI：轮询到完成再打印；失败如实报 ─────────────────────────────────────────


def _cli_env(monkeypatch, responses: list[tuple[int, dict]], seen: list) -> None:
    from bladex_proxy import cli
    from bladex_proxy.cli import memory_cmds

    def fake_call(method, path, body=None, params=None, timeout=15.0):
        seen.append((method, path, body))
        return responses.pop(0)

    for k in list(os.environ):
        if k.startswith("BLADEX_"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(cli, "_load_env_file", lambda: None)
    monkeypatch.setattr(cli, "_chdir_to_deployment_root", lambda: None)
    monkeypatch.setattr(
        cli,
        "_probe_ready",
        lambda *_a, **_k: {
            "ready": True,
            ADMIN_CONTRACT_FIELD: ADMIN_CONTRACT_VERSION,
            "build": "test",
        },
    )
    monkeypatch.setattr(memory_cmds, "_admin_call", fake_call)
    monkeypatch.setattr(memory_cmds, "_JOB_POLL_INTERVAL_S", 0.0)


def _done_result(n: int) -> dict:
    return {
        "status": "tombstoned",
        "dry_run": False,
        "total_turns": n,
        "backup_age_days": 1,
        "projects": [
            {
                "project_id": _PA,
                "project_names": ["a"],
                "turn_count": n,
                "agents": {"codex": n},
                "first_ts": "t0",
                "last_ts": "t1",
                "ledgers": 0,
                "memory": {"turn_tombstones": n, "index_cascaded": 0, "index_deferred": n},
            }
        ],
        "hint": "",
    }


def test_cli_polls_job_until_done_then_prints_result(monkeypatch, capsys) -> None:
    from bladex_proxy.cli import main as cli_main

    seen: list = []
    job = {"job_id": "tbp-x", "total": 500}
    _cli_env(
        monkeypatch,
        [
            (200, {"status": "submitted", "job_id": "tbp-x", "total_turns": 500}),
            (200, {"job": {**job, "state": "running", "done": 0}}),
            (503, {}),  # 一次读不到不放弃
            (200, {"job": {**job, "state": "running", "done": 250}}),
            (200, {"job": {**job, "state": "done", "done": 500, "result": _done_result(500)}}),
        ],
        seen,
    )
    rc = cli_main(["memory", "tombstone-project", _PA, "--yes"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "Submitted job tbp-x" in out
    assert "250/500" in out
    assert "✓ tombstoned 500 turns across 1 project(s)." in out
    assert "tombstones written: 500 (index cascaded 0, deferred 500)" in out
    assert [p for _m, p, _b in seen].count("/admin/jobs/tbp-x") == 4


def test_cli_reports_failed_job_with_check_command(monkeypatch, capsys) -> None:
    from bladex_proxy.cli import main as cli_main

    _cli_env(
        monkeypatch,
        [
            (200, {"status": "submitted", "job_id": "tbp-y", "total_turns": 10}),
            (
                200,
                {
                    "job": {
                        "job_id": "tbp-y",
                        "state": "failed",
                        "done": 3,
                        "total": 10,
                        "error": "RuntimeError: disk",
                    }
                },
            ),
        ],
        [],
    )
    rc = cli_main(["memory", "tombstone-project", _PA, "--yes"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "failed after 3/10 turns: RuntimeError: disk" in out
    assert f"bladex memory tombstone-project {_PA} --dry-run" in out
    assert "✓ tombstoned" not in out


def test_cli_refuses_a_non_job_response_on_real_run(monkeypatch, capsys) -> None:
    """旧形态（同步 tombstoned）或任何别的形状 ⇒ 不当成功渲染。"""
    from bladex_proxy.cli import main as cli_main

    _cli_env(monkeypatch, [(200, _done_result(5))], [])
    rc = cli_main(["memory", "tombstone-project", _PA, "--yes"])
    out = capsys.readouterr().out
    assert rc == 1 and "expected status=submitted" in out
