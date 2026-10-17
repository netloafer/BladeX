"""批 T0b · 缺的两条路径（2026-09-22）：①已墓碑账本补清记忆（MQ-I23）②按 project 打墓碑。

# 路径① —— 为什么是一条独立的动作

`DELETE /admin/ledgers/{id}` 的第一道判定是 `known = ledger_id in agency.pool`，
而墓碑那一刻账本就被 `agency.pool.pop()` 摘掉 ⇒ **第二次调用必 404**。MQ-I22 修好之后
这条从"不方便"升级成**挡路**：乙四本已在 09-22 被墓碑、且当时各打 **0** 条 TURN 墓碑，
它们窗口里那 41 条轮次**没有任何 CLI 路径够得着**。

选 **(b) 新增语义独立的动作**（调度倾向，我同意并加一条理由）：
404 应当继续只说一句话——"这本不在活池"。若让 `DELETE` 对已墓碑账本"放行一半"，
同一个状态码就要同时表示"不存在"和"只做了另一半"，**下一棒读不出它判的是什么**。
这与 MQ-A18/P7「同一件事只在一个地方做对」是同族要求的反面：两件不同的事不该共用一个判定。

# 路径② —— 全仓此前没有任何东西能按 project 选轮次

`memory forget` 是 fact 级、`ledger tombstone` 是账本窗口级。拍板 (3) 要的
"bench project **作用域整体**墓碑"落不了地。

🔴 **硬拒绝空 project**：Global 那 17,047 轮只能按账本清（拍板 (3) 原文）。
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

import pytest
from bladex_core.ledger import new_ledger
from bladex_core.ledger_runtime import ActivationTable, activation_scope
from bladex_proxy.admin_contract import ADMIN_CONTRACT_FIELD, ADMIN_CONTRACT_VERSION
from bladex_proxy.models import (
    AdminEventType,
    Identity,
    SessionIdSource,
    TombstoneSource,
    TombstoneTargetType,
    Turn,
    TurnStatus,
)
from bladex_proxy.storage.memory_hub import MemoryHub

_L = "ldg-t0bpath00001"          # 已墓碑的那本（路径①）
_LIVE = "ldg-t0blive00001"       # 活池里的那本（阴性对照）
_AGENT = "codex"
_PROJ = "p-t0bproj0001"          # 待清
_PROJ2 = "p-t0bproj0002"         # 待清（第二个）
_KEEP = "p-t0bkeep0001"          # 保留


def _put(hub: MemoryHub, seq: int, *, agent: str, project: str, name: str = "") -> str:
    ident = Identity(user_id="u1", agent_id=agent, session_id="s1", turn_index=seq,
                     session_id_source=SessionIdSource.FINGERPRINT,
                     project_id=project, project_name=name or project)
    turn = Turn(identity=ident, model="test",
                request_messages=[{"role": "user", "content": f"第 {seq} 轮"}],
                response_text="好的。", status=TurnStatus.OK, ts=datetime.now(UTC),
                session_id_source=SessionIdSource.FINGERPRINT)
    key = ident.storage_key(f"{int(datetime.now(UTC).timestamp() * 1000)}-{seq}")
    hub.put(key, turn)
    time.sleep(0.02)
    return key


def _create_and_switch(hub: MemoryHub, ledger_id: str, *, agent: str, project: str) -> None:
    led = new_ledger(ledger_id=ledger_id, title=f"{ledger_id} 的任务", goal="g")
    led.created_at = datetime.now(UTC).isoformat()
    hub.append_admin_event(AdminEventType.LEDGER_CREATE, "", ledger=led.model_dump(mode="json"))
    time.sleep(0.02)
    hub.append_admin_event(AdminEventType.LEDGER_SWITCH, "", agent_id=agent,
                           project_id=project, session_id="s1", turn_index=1,
                           to_ledger_id=ledger_id)
    time.sleep(0.02)


class _StubAgency:
    def __init__(self, pool: dict):
        self.pool = pool
        self.activation = ActivationTable()
        self.activation.restore({activation_scope(_AGENT, _PROJ): _LIVE})


@pytest.fixture
def live(tmp_path: Path, monkeypatch, isolated_home):
    """真 Hub：`_L` 已墓碑（窗口里 2 轮**未**墓碑，= 乙四本的形态）；`_LIVE` 仍在活池。"""
    import os

    from bladex_proxy.config import ProxyConfig
    from bladex_proxy.server import create_app
    from fastapi.testclient import TestClient

    hub_path = tmp_path / "hub"
    hub = MemoryHub(hub_path)
    hub.open()
    # ① 先建 `_L` 并切入，窗口里放两轮，然后切走 → 再给账本打墓碑（模拟 09-22 的现场：
    #    账本墓碑已在，窗口轮次一条没打）。
    _create_and_switch(hub, _L, agent=_AGENT, project=_PROJ)
    residue = [_put(hub, 1, agent=_AGENT, project=_PROJ),
               _put(hub, 2, agent=_AGENT, project=_PROJ)]
    _create_and_switch(hub, _LIVE, agent=_AGENT, project=_PROJ)
    live_turns = [_put(hub, 3, agent=_AGENT, project=_PROJ)]
    hub.append_tombstone(target_type=TombstoneTargetType.LEDGER, target_key=_L,
                         source=TombstoneSource.USER)
    # ② 路径② 的素材：两个待清 project + 一个保留 project，跨两个 agent。
    p2 = [_put(hub, 4, agent="claude-code", project=_PROJ2, name="bench-x"),
          _put(hub, 5, agent=_AGENT, project=_PROJ2, name="bench-x")]
    keep = [_put(hub, 6, agent=_AGENT, project=_KEEP, name="BladeX")]
    glob = [_put(hub, 7, agent=_AGENT, project="")]
    hub.close()

    monkeypatch.setenv("BLADEX_AUTH_ENABLED", "false")
    monkeypatch.setenv("BLADEX_ROCKSDB_PATH", str(hub_path))
    monkeypatch.setenv("BLADEX_INDEX_PATH", str(tmp_path / "index"))
    app = create_app(ProxyConfig())
    with TestClient(app) as c:
        c.app.state.agency = _StubAgency({_LIVE: new_ledger(ledger_id=_LIVE, title="t", goal="g")})

        def transport(method: str, path: str, body: dict | None = None,
                      params: dict | None = None):
            r = c.request(method, path, params={k: v for k, v in (params or {}).items()
                                                if v not in ("", None)}, json=body)
            try:
                return r.status_code, r.json()
            except ValueError:
                return r.status_code, {}

        monkeypatch.chdir(tmp_path)
        for k in list(os.environ):
            if k.startswith("BLADEX_"):
                monkeypatch.delenv(k, raising=False)
        from bladex_proxy import cli as _cli
        from bladex_proxy.cli import ledger_cmds as _ldg
        from bladex_proxy.cli import memory_cmds as _mem
        monkeypatch.setattr(_ldg, "_admin_call", transport)
        monkeypatch.setattr(_mem, "_admin_call", transport)
        monkeypatch.setattr(_cli, "_probe_ready", lambda *_a, **_k: {
            "ready": True, ADMIN_CONTRACT_FIELD: ADMIN_CONTRACT_VERSION, "build": "test"})
        yield c, hub_path, {"residue": residue, "live": live_turns, "p2": p2,
                            "keep": keep, "global": glob}


def _turn_tombstones(hub_path: Path) -> set[str]:
    hub = MemoryHub(hub_path, read_only=True)
    hub.open()
    try:
        return {t.target_key for _k, t in hub.scan_tombstones()
                if t.target_type == TombstoneTargetType.TURN}
    finally:
        hub.close()


# ══ 路径① ════════════════════════════════════════════════════════════════════

def test_1_delete_still_404s_on_a_tombstoned_ledger(live) -> None:
    """🔴 阴性对照：原来那条端点的 404 语义一字不改（本批没有稀释它）。"""
    c, _hp, _k = live
    r = c.request("DELETE", f"/admin/ledgers/{_L}", params={"dry_run": "true"})
    assert r.status_code == 404 and "not in the live pool" in r.text


def test_1_dry_run_reaches_a_tombstoned_ledger_window(live) -> None:
    """新端点对**已墓碑**账本给得出窗口读数，且一条墓碑都不写。"""
    c, hub_path, keys = live
    before = _turn_tombstones(hub_path)
    r = c.request("DELETE", f"/admin/ledgers/{_L}/memory", params={"dry_run": "true"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["status"] == "dry_run" and d["dry_run"] is True
    assert d["ledger_in_pool"] is False, "已墓碑 ⇒ 不在活池，但照样算得出窗口"
    assert d["window_turns"] == len(keys["residue"]) == 2
    assert d["title"] == f"{_L} 的任务", \
        "墓碑后仍取得到**标题**（不是回落成 id）——重建专用装载保留池"
    assert _turn_tombstones(hub_path) == before


def test_1_dry_run_is_repeatable(live) -> None:
    """判据③：同一本连跑两次 dry-run，读数逐字相同。"""
    c, _hp, _k = live
    a = c.request("DELETE", f"/admin/ledgers/{_L}/memory", params={"dry_run": "true"}).json()
    b = c.request("DELETE", f"/admin/ledgers/{_L}/memory", params={"dry_run": "true"}).json()
    assert a["window_turns"] == b["window_turns"] and a["window_scopes"] == b["window_scopes"]


def test_1_real_run_then_rerun_is_idempotent(live) -> None:
    """判据②：补打之后再跑一次 ⇒ 第二次 `turn_tombstones` = 0（不重复灌）。

    幂等不是靠"记得只跑一次"，是靠 `scan_meta_snapshot()` 已滤掉被墓碑的 turn。
    """
    c, hub_path, keys = live
    first = c.request("DELETE", f"/admin/ledgers/{_L}/memory").json()
    assert first["status"] == "memory_tombstoned"
    assert first["memory"]["turn_tombstones"] == 2
    assert _turn_tombstones(hub_path) == set(keys["residue"])
    second = c.request("DELETE", f"/admin/ledgers/{_L}/memory").json()
    assert second["memory"]["turn_tombstones"] == 0, "🔴 重复灌 tombstone/ 前缀"
    assert second["window_turns"] == 0
    assert _turn_tombstones(hub_path) == set(keys["residue"]), "第二次不该新增"


def test_1_does_not_touch_the_ledger_object(live) -> None:
    """判据⑤同族：本端点只墓碑 turn，不追加第二条 LEDGER 墓碑。"""
    c, hub_path, _k = live
    hub = MemoryHub(hub_path, read_only=True)
    hub.open()
    try:
        before = sum(1 for _k2, t in hub.scan_tombstones()
                     if t.target_type == TombstoneTargetType.LEDGER)
    finally:
        hub.close()
    c.request("DELETE", f"/admin/ledgers/{_L}/memory")
    hub = MemoryHub(hub_path, read_only=True)
    hub.open()
    try:
        after = sum(1 for _k2, t in hub.scan_tombstones()
                    if t.target_type == TombstoneTargetType.LEDGER)
    finally:
        hub.close()
    assert after == before == 1


def test_1_live_pool_ledger_behaviour_unchanged(live) -> None:
    """🔴 判据④ 阴性对照：活池里的账本，`DELETE /admin/ledgers/{id}` 行为一字不变。"""
    c, hub_path, keys = live
    d = c.request("DELETE", f"/admin/ledgers/{_LIVE}",
                  params={"with_memory": "true", "dry_run": "true"}).json()
    assert d["status"] == "dry_run" and d["window_turns"] == len(keys["live"]) == 1
    real = c.request("DELETE", f"/admin/ledgers/{_LIVE}", params={"with_memory": "true"}).json()
    assert real["status"] == "tombstoned" and real["tombstone_key"]
    assert real["memory"]["turn_tombstones"] == 1


def test_1_unknown_ledger_still_404(live) -> None:
    """阴性对照：真的不存在的 id 仍然 404（不是"什么都接住"）。"""
    c, _hp, _k = live
    r = c.request("DELETE", "/admin/ledgers/ldg-doesnotexist/memory", params={"dry_run": "true"})
    assert r.status_code == 404 and "no LEDGER_CREATE event" in r.text


def test_1_cli_dry_run_prints_the_window(live, capsys) -> None:
    from bladex_proxy import cli
    c, hub_path, keys = live
    before = _turn_tombstones(hub_path)
    assert cli.main(["ledger", "tombstone-memory", _L, "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "DRY RUN -- nothing written." in out
    assert "turns in window: 2" in out
    assert "ledger object: NOT touched" in out
    assert _turn_tombstones(hub_path) == before


# ══ 路径② ════════════════════════════════════════════════════════════════════

def test_2_dry_run_per_project_readout(live) -> None:
    """判据①③：每个 project **单独**一行读数（turn 数 / agent 分布 / 时间范围 / 关联账本数）。"""
    c, hub_path, keys = live
    before = _turn_tombstones(hub_path)
    r = c.post("/admin/turns/tombstone-by-project",
               json={"project_ids": [_PROJ, _PROJ2], "dry_run": True})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["status"] == "dry_run" and d["dry_run"] is True
    rows = {row["project_id"]: row for row in d["projects"]}
    assert set(rows) == {_PROJ, _PROJ2}, "🔴 判据③：不许只给总数"
    assert rows[_PROJ]["turn_count"] == 3        # residue 2 + live 1
    assert rows[_PROJ]["agents"] == {_AGENT: 3}
    assert rows[_PROJ]["ledgers"] == 2           # _L + _LIVE 都在这个 scope 上
    assert rows[_PROJ2]["turn_count"] == 2
    assert rows[_PROJ2]["agents"] == {"claude-code": 1, _AGENT: 1}, "agent 分布要拆开"
    assert rows[_PROJ2]["project_names"] == ["bench-x"]
    assert rows[_PROJ2]["ledgers"] == 0
    for row in d["projects"]:
        assert row["first_ts"] and row["last_ts"] and row["first_ts"] <= row["last_ts"]
    assert d["total_turns"] == 5
    assert _turn_tombstones(hub_path) == before


def test_2_empty_project_is_refused_hard(live) -> None:
    """🔴 判据②：`project_id=""`（Global）一律拒绝 —— live 上那是 17,047 轮。"""
    c, hub_path, _k = live
    before = _turn_tombstones(hub_path)
    for body in ({"project_ids": [""], "dry_run": True},
                 {"project_ids": [_PROJ, ""], "dry_run": False},
                 {"project_ids": ["   "], "dry_run": False},
                 {"project_ids": [], "dry_run": True}):
        r = c.post("/admin/turns/tombstone-by-project", json=body)
        assert r.status_code == 400, f"{body} 没被拒：{r.text}"
    assert _turn_tombstones(hub_path) == before
    # 计算层自己也拒一次（纵深，不是替代）。
    from bladex_proxy.project_purge import EmptyProjectRefused, project_scope_preview
    hub = MemoryHub(hub_path, read_only=True)
    hub.open()
    try:
        with pytest.raises(EmptyProjectRefused):
            project_scope_preview(hub, [""])
    finally:
        hub.close()


def test_2_dry_run_is_the_default(live) -> None:
    """body 不写 `dry_run` ⇒ **只读**（一次几千轮的口径，默认必须是安全的那一边）。"""
    c, hub_path, _k = live
    before = _turn_tombstones(hub_path)
    d = c.post("/admin/turns/tombstone-by-project", json={"project_ids": [_PROJ]}).json()
    assert d["status"] == "dry_run" and d["dry_run"] is True
    assert _turn_tombstones(hub_path) == before


def _real_run(c, body: dict) -> dict:
    """真跑自 MQ-I30（批 Q3A）起是后台 job：先回 submitted，结果从 /admin/jobs 取。"""
    sub = c.post("/admin/turns/tombstone-by-project", json=body).json()
    assert sub["status"] == "submitted" and sub["job_id"], sub
    deadline = time.time() + 30
    while time.time() < deadline:
        job = c.get(f"/admin/jobs/{sub['job_id']}").json()["job"]
        if job["state"] != "running":
            assert job["state"] == "done", job
            return job["result"]
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_2_real_run_tombstones_only_the_named_projects(live) -> None:
    """判据⑤⑥：只墓碑 turn、只墓碑点名的 project、连跑两次幂等。"""
    c, hub_path, keys = live
    d = _real_run(c, {"project_ids": [_PROJ2], "dry_run": False})
    assert d["status"] == "tombstoned"
    assert d["projects"][0]["memory"]["turn_tombstones"] == 2
    got = _turn_tombstones(hub_path)
    assert got == set(keys["p2"])
    assert not (got & set(keys["keep"])), "保留的 project 被误伤"
    assert not (got & set(keys["global"])), "Global 轮次被误伤"
    assert not (got & set(keys["residue"])), "别的 project 被误伤"
    # 账本维度不连带（拍板 (3) 的两条线不合并）
    hub = MemoryHub(hub_path, read_only=True)
    hub.open()
    try:
        assert not any(t.target_type == TombstoneTargetType.LEDGER
                       for _k2, t in hub.scan_tombstones()
                       if t.target_key not in (_L,)), "本路径不得打新的 LEDGER 墓碑"
    finally:
        hub.close()
    # 幂等
    again = _real_run(c, {"project_ids": [_PROJ2], "dry_run": False})
    assert again["projects"][0]["turn_count"] == 0
    assert _turn_tombstones(hub_path) == got


def test_2_requested_but_empty_project_is_reported_not_dropped(live) -> None:
    """请求了却一条轮次都没有的 project **要出现在读数里**（0 ≠ 我没查）。"""
    c, _hp, _k = live
    d = c.post("/admin/turns/tombstone-by-project",
               json={"project_ids": [_PROJ, "p-nosuchproject"], "dry_run": True}).json()
    rows = {row["project_id"]: row for row in d["projects"]}
    assert rows["p-nosuchproject"]["turn_count"] == 0
    assert rows["p-nosuchproject"]["first_ts"] is None, "没有数据 ⇒ None，不许编一个时间"


def test_2_cli_prints_one_line_per_project(live, capsys) -> None:
    from bladex_proxy import cli
    c, hub_path, _k = live
    before = _turn_tombstones(hub_path)
    assert cli.main(["memory", "tombstone-project", _PROJ, _PROJ2, "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "DRY RUN -- nothing written." in out
    assert _PROJ in out and _PROJ2 in out
    assert "turns=     3" in out and "turns=     2" in out
    assert "ledger objects are NOT touched" in out
    assert _turn_tombstones(hub_path) == before


def test_2_cli_refuses_empty_project(live, capsys) -> None:
    from bladex_proxy import cli
    c, hub_path, _k = live
    before = _turn_tombstones(hub_path)
    assert cli.main(["memory", "tombstone-project", "", "--dry-run"]) == 1
    assert _turn_tombstones(hub_path) == before
