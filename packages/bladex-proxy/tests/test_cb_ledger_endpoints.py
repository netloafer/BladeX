"""批 CB：交接包 / 建本 / 改本的 admin 端点。真 Hub（tmp RocksDB），每条写路径都按重放核一遍。

钉：导出（exposure、路径、③ 用户话）· 导入（哈希 / 版本 / 同 id 不覆盖 / 墓碑 id / Goal 身份 /
②③ 只存不注入）· create（不激活、绑锚）· edit（expected_rev 必填、409 不写、Goal 只经 update_goal）·
user-edit 的 Goal 身份（MQ-L107）· 全部只落 `ledger_user_edit`。
"""

from __future__ import annotations

import copy
import json
import re
import tempfile
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
import structlog.testing
from bladex_core.handoff import content_hash
from bladex_core.ledger import (
    LedgerEntry,
    add_entry,
    new_ledger,
    replay_ledger_events,
)
from bladex_core.ledger_runtime import ledger_anchor_matter_id
from bladex_core.sensitivity import SensitivityConfig
from bladex_proxy.agency import AgencyRuntime
from bladex_proxy.config import ProxyConfig
from bladex_proxy.ledger_events import load_ledger_events
from bladex_proxy.models import (
    AdminEventType,
    Identity,
    SessionIdSource,
    TombstoneTargetType,
    Turn,
    TurnStatus,
)
from bladex_proxy.server import create_app
from bladex_proxy.storage.memory_hub import MemoryHub
from fastapi.testclient import TestClient

LID = "ldg-cb0000000001"
ROOT = "/Users/alice/dev/proj"
SECRET = "abcdef-ghijk-lmnopqrstuv"


class _Index:
    """Index 只读面的最小桩。`facts` 是 Matter 成员的 ceiling 列表；`boom` 模拟读不出来。"""

    def __init__(self, *, title="锚定事儿", summary=f"摘要提到 {ROOT}/a.py", ceilings=("public",), boom=False):
        self.title, self.summary, self.ceilings, self.boom = title, summary, ceilings, boom
        self.asked: list[str] = []

    def get_matter(self, matter_id):
        self.asked.append(matter_id)
        if self.boom:
            raise RuntimeError("index down")
        return SimpleNamespace(title=self.title, summary=self.summary)

    def get_facts_for_matter(self, matter_id, top_k=0):
        return [SimpleNamespace(exposure_ceiling=c) for c in self.ceilings]

    def all_projects(self):
        return [SimpleNamespace(project_id="p-1", root_path=ROOT)]


def _seed() -> object:
    led = new_ledger(ledger_id=LID, title="任务", goal=f"按 {ROOT}/plan.md 做完", goal_source="model",
                     goal_verbatim="用户原话", created_at="2026-11-02T00:00:00+00:00")
    led = add_entry(led, "core", LedgerEntry(text=f"key 是 api_key={SECRET}", source="model", writer="codex"),
                    actor="model")
    led = add_entry(led, "verified", LedgerEntry(text="/admin/ledgers 可用", source="tool", ref="/tmp/x.log",
                                                 writer="codex"), actor="model")
    return led.model_copy(update={"rev": 3, "matter_id": ledger_anchor_matter_id(LID)})


@pytest.fixture()
def env(monkeypatch):
    monkeypatch.setenv("BLADEX_AUTH_ENABLED", "false")
    tmp = tempfile.mkdtemp(prefix="cb_ep_")
    cfg = ProxyConfig(upstream_model="openai/test", upstream_api_key="sk-fake",
                      rocksdb_path=f"{tmp}/app_hub", index_path=f"{tmp}/index")
    app = create_app(cfg)
    with TestClient(app) as c:
        hub = MemoryHub(f"{tmp}/hub")
        hub.open()
        led = _seed()
        hub.append_admin_event(AdminEventType.LEDGER_CREATE, "", ledger=led.model_dump(),
                               agent_id="codex", project_id="p-1", session_id="s1", turn_index=1)
        ag = AgencyRuntime(hub=hub)
        orig_hub, orig_index = c.app.state.hub, c.app.state.index
        c.app.state.agency = ag
        c.app.state.hub = hub
        c.app.state.index = _Index()
        yield SimpleNamespace(c=c, ag=ag, hub=hub, app=c.app, cfg=cfg)
        c.app.state.hub, c.app.state.index = orig_hub, orig_index   # 交还给 lifespan 关
        hub.close()


def _events(hub) -> list[tuple[str, dict]]:
    return [(e.event_type.value, e.payload) for _k, e in hub.scan_admin_events()]


def _replayed(hub) -> dict:
    return replay_ledger_events(load_ledger_events(hub))[0]


def _tombstone(hub, ledger_id: str) -> None:
    hub.append_tombstone(TombstoneTargetType.LEDGER, ledger_id)


def _export(env, **params) -> dict:
    r = env.c.get(f"/admin/ledgers/{LID}/handoff", params=params)
    assert r.status_code == 200, r.text
    return r.json()


# ── 导出 ────────────────────────────────────────────────────────────────────


def test_export_is_read_only_and_writes_no_event(env) -> None:
    before = _events(env.hub)
    data = _export(env)
    assert _events(env.hub) == before
    pkg = data["package"]
    assert pkg["target_exposure"] == "public", "默认 public：目的地未知按最坏算"
    assert pkg["source_ledger_id"] == LID
    assert pkg["source_project"] == "p-1"
    assert pkg["ledger"]["goal"] == "按 <project>/plan.md 做完"
    assert pkg["content_hash"] == content_hash(pkg)
    assert data["markdown"].startswith("# Handoff: 任务")


def test_export_and_log_never_contain_original_secret_or_paths(env) -> None:
    with structlog.testing.capture_logs() as logs:
        data = _export(env)
    text = json.dumps(data, ensure_ascii=False)
    assert SECRET not in text
    assert "/Users/" not in text and "/tmp/" not in text
    assert data["package"]["rewrites"]["credential"] == 1
    ev = [x for x in logs if x["event"] == "ledger_handoff_exported"]
    assert len(ev) == 1
    assert ev[0]["rewrites"]["credential"] == 1
    assert SECRET not in json.dumps(logs, ensure_ascii=False, default=str)


def test_export_private_matter_to_public_is_omitted_and_counted(env) -> None:
    env.app.state.index = _Index(title="私有标题", summary="私有摘要", ceilings=("public", "private"))
    data = _export(env)
    assert data["package"]["matters"] == []
    assert data["package"]["omitted"]["matters_exposure"] == 1
    text = json.dumps(data, ensure_ascii=False)
    assert "私有标题" not in text and "私有摘要" not in text
    assert "1 omitted" in data["markdown"]
    # 同一个 Matter 导出到 private / local 目标：strictest=private，private 放行，local 目标也放行
    assert _export(env, exposure="private")["package"]["matters"][0]["title"] == "私有标题"
    assert _export(env, exposure="local")["package"]["matters"][0]["title"] == "私有标题"


def test_export_local_matter_is_stricter_than_private_target(env) -> None:
    env.app.state.index = _Index(ceilings=("local",))
    assert _export(env, exposure="private")["package"]["omitted"]["matters_exposure"] == 1


def test_export_matter_unreadable_is_omitted_not_treated_as_public(env) -> None:
    env.app.state.index = _Index(boom=True)
    pkg = _export(env)["package"]
    assert pkg["matters"] == [] and pkg["omitted"]["matters_unavailable"] == 1
    env.app.state.index = None
    assert _export(env)["package"]["omitted"]["matters_unavailable"] == 1


def _put_turn(hub, *, turn_index: int, text: str, entry_ms: int, sensitivity: str = "normal",
              project_root: str = "") -> None:
    ident = Identity(user_id="u1", agent_id="codex", session_id="s1", turn_index=turn_index,
                     session_id_source=SessionIdSource.FINGERPRINT, project_root=project_root)
    msgs = [m for i in range(turn_index - 1) for m in ({"role": "user", "content": f"第{i}句"},
                                                   {"role": "assistant", "content": "好"})]
    msgs.append({"role": "user", "content": text})
    hub.put(ident.storage_key(f"{entry_ms}-0"), Turn(
        identity=ident, model="m", status=TurnStatus.OK, response_text="assistant 正文不许进包",
        request_messages=msgs, sensitivity=sensitivity, session_id_source=SessionIdSource.FINGERPRINT,
        ts=datetime.fromtimestamp(entry_ms / 1000, UTC)))


def _update(env, text: str, *, turn_index: int) -> None:
    """模型写一次账本：真事件形状（整本 dump + 写者四格）。"""
    led = env.ag.pool[LID]
    led = add_entry(led, "verified", LedgerEntry(text=text, source="model", writer="codex"), actor="model")
    env.ag.pool[LID] = led
    env.hub.append_admin_event(AdminEventType.LEDGER_UPDATE, "", ledger=led.model_dump(),
                               agent_id="codex", project_id="p-1", session_id="s1", turn_index=turn_index)


def test_export_rounds_carry_entry_diff_and_user_head_only(env) -> None:
    _update(env, "第一条结论", turn_index=4)
    now = int(datetime.now(UTC).timestamp() * 1000)
    _put_turn(env.hub, turn_index=4, text=f"用户问 {ROOT}/q.py\n\n第二段不要", entry_ms=now + 500)
    pkg = _export(env)["package"]
    assert len(pkg["recent_rounds"]) == 1
    r = pkg["recent_rounds"][0]
    assert r["user_text_head"] == "用户问 <project>/q.py"
    assert r["added"] == [{"section": "verified", "text": "第一条结论"}]
    text = json.dumps(pkg, ensure_ascii=False)
    assert "assistant 正文" not in text and "第二段不要" not in text


def test_export_round_without_turn_is_kept_and_counted(env) -> None:
    _update(env, "没有对应 Turn", turn_index=9)
    pkg = _export(env)["package"]
    assert pkg["recent_rounds"][0]["user_text_head"] == ""
    assert pkg["omitted"]["rounds_without_user_text"] == 1


def _sens(env) -> None:
    cfg = SensitivityConfig(enabled=True, levels={"normal": "public", "sensitive": "local"})
    rc = env.cfg.routing_config
    object.__setattr__(rc, "sensitivity_config", lambda *a, **k: cfg)


def test_export_sensitive_round_is_omitted_for_public_target(env) -> None:
    _update(env, "敏感结论", turn_index=4)
    now = int(datetime.now(UTC).timestamp() * 1000)
    _put_turn(env.hub, turn_index=4, text="敏感原话", entry_ms=now + 500, sensitivity="sensitive")
    _sens(env)
    pkg = _export(env)["package"]
    assert pkg["recent_rounds"] == [] and pkg["omitted"]["rounds_exposure"] == 1
    assert "敏感原话" not in json.dumps(pkg, ensure_ascii=False)
    assert _export(env, exposure="local")["package"]["recent_rounds"][0]["user_text_head"] == "敏感原话"


def test_export_round_omitted_when_sensitivity_on_and_turn_missing(env) -> None:
    _update(env, "判定不了", turn_index=9)
    _sens(env)
    pkg = _export(env)["package"]
    assert pkg["recent_rounds"] == [] and pkg["omitted"]["rounds_exposure"] == 1


def test_export_omitted_rounds_do_not_let_older_unchecked_rounds_in(env, monkeypatch) -> None:
    """被 exposure 省略的轮不占名额，再旧的那轮要**过同一道判定**，不是直接带进包。"""
    monkeypatch.setenv("BLADEX_HANDOFF_MAX_ROUNDS", "1")
    now = int(datetime.now(UTC).timestamp() * 1000)
    _update(env, "旧结论", turn_index=4)
    _put_turn(env.hub, turn_index=4, text="旧原话", entry_ms=now + 500, sensitivity="sensitive")
    _update(env, "新结论", turn_index=5)
    _put_turn(env.hub, turn_index=5, text="新原话", entry_ms=now + 600, sensitivity="sensitive")
    _sens(env)
    pkg = _export(env)["package"]
    assert pkg["recent_rounds"] == []
    assert pkg["omitted"]["rounds_exposure"] == 2


def test_export_project_root_from_turn_when_index_missing(env) -> None:
    env.app.state.index = None
    _update(env, "x", turn_index=4)
    now = int(datetime.now(UTC).timestamp() * 1000)
    _put_turn(env.hub, turn_index=4, text="问", entry_ms=now + 500, project_root=ROOT)
    assert _export(env)["package"]["ledger"]["goal"] == "按 <project>/plan.md 做完"


def test_export_bad_inputs(env) -> None:
    r = env.c.get(f"/admin/ledgers/{LID}/handoff", params={"exposure": "secret"})
    assert r.status_code == 400 and "secret" in r.json()["error"]["message"]
    r = env.c.get(f"/admin/ledgers/{LID}/handoff", params={"exposur": "local"})
    assert r.status_code == 400 and "exposur" in r.json()["detail"]
    assert env.c.get("/admin/ledgers/ldg-nope/handoff").status_code == 404


# ── 导入 ────────────────────────────────────────────────────────────────────


def _import(env, package: dict, **body):
    return env.c.post("/admin/ledgers/import", json={"package": package, **body})


def test_import_same_id_never_overwrites_and_gets_new_id(env) -> None:
    pkg = _export(env)["package"]
    before = env.ag.pool[LID].model_dump(mode="json")
    r = _import(env, pkg)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["id_conflict"] is True and d["ledger_id"] != LID
    assert env.ag.pool[LID].model_dump(mode="json") == before
    again = _import(env, pkg).json()
    assert again["ledger_id"] not in (LID, d["ledger_id"])
    assert _replayed(env.hub)[LID].model_dump(mode="json") == before


def test_import_free_id_is_kept_and_identity_is_user(env) -> None:
    pkg = _export(env)["package"]
    del env.ag.pool[LID]
    d = _import(env, pkg).json()
    assert d["ledger_id"] == LID and d["id_conflict"] is False
    led = env.ag.pool[LID]
    assert led.goal == pkg["ledger"]["goal"]
    assert led.goal_source == "user" and led.last_writer == "user"
    assert led.goal_origin == f"handoff:{LID}@{pkg['content_hash'][:12]}"


def test_import_tombstoned_id_gets_new_id_and_survives_replay(env) -> None:
    pkg = _export(env)["package"]
    from bladex_proxy.ledger_events import tombstoned_ledger_ids
    _tombstone(env.hub, LID)
    assert LID in tombstoned_ledger_ids(env.hub)
    del env.ag.pool[LID]
    d = _import(env, pkg).json()
    assert d["ledger_id"] != LID and d["id_conflict"] is True
    assert d["ledger_id"] in _replayed(env.hub)


def test_import_event_is_ledger_user_edit_and_keeps_whole_package(env) -> None:
    pkg = _export(env)["package"]
    n = len(_events(env.hub))
    with structlog.testing.capture_logs() as logs:
        d = _import(env, pkg).json()
    new = _events(env.hub)[n:]
    assert [t for t, _p in new] == ["ledger_user_edit"]
    payload = new[0][1]
    assert payload["origin"] == "handoff_import"
    assert payload["handoff"]["package"] == pkg
    assert [x for x in logs if x["event"] == "ledger_handoff_imported"][0]["ledger"] == d["ledger_id"]


def test_import_rounds_and_matter_stay_out_of_sections_and_injection(env) -> None:
    _update(env, "轮次里的结论", turn_index=4)
    now = int(datetime.now(UTC).timestamp() * 1000)
    _put_turn(env.hub, turn_index=4, text="轮次用户话不注入", entry_ms=now + 500)
    env.app.state.index = _Index(title="摘要标题不注入", summary="摘要正文不注入")
    pkg = _export(env)["package"]
    assert pkg["matters"] and pkg["recent_rounds"][0]["user_text_head"] == "轮次用户话不注入"
    d = _import(env, pkg).json()
    led = env.ag.pool[d["ledger_id"]]
    dump = led.model_dump_json()
    for marker in ("轮次用户话不注入", "摘要标题不注入", "摘要正文不注入"):
        assert marker not in dump
    assert d["ledger"]["sections"]["core"]


def test_import_does_not_bind_anything(env) -> None:
    pkg = _export(env)["package"]
    before = env.ag.activation.snapshot()
    n = [t for t, _ in _events(env.hub) if t == "ledger_switch"]
    _import(env, pkg)
    assert env.ag.activation.snapshot() == before
    assert [t for t, _ in _events(env.hub) if t == "ledger_switch"] == n


@pytest.mark.parametrize(("mutate", "code"), [
    (lambda p: p["ledger"].__setitem__("goal", p["ledger"]["goal"] + "x"), "hash"),
    (lambda p: p.__setitem__("format_version", 2), "version"),
    (lambda p: p.__setitem__("nope", 1), "shape"),
])
def test_import_rejects_and_writes_nothing(env, mutate, code) -> None:
    pkg = copy.deepcopy(_export(env)["package"])
    mutate(pkg)
    n = len(_events(env.hub))
    pool = set(env.ag.pool)
    r = _import(env, pkg)
    assert r.status_code == 400, r.text
    assert r.json()["error"]["code"] == code
    assert len(_events(env.hub)) == n and set(env.ag.pool) == pool


def test_import_unknown_version_message_names_it(env) -> None:
    pkg = _export(env)["package"]
    pkg["format_version"] = 7
    assert "7" in _import(env, pkg).json()["error"]["message"]


def test_import_size_cap(env, monkeypatch) -> None:
    monkeypatch.setenv("BLADEX_HANDOFF_IMPORT_MAX_BYTES", "10")
    r = _import(env, _export(env)["package"])
    assert r.status_code == 400 and r.json()["error"]["code"] == "size"


def test_import_dry_run_writes_nothing_and_acks(env) -> None:
    pkg = _export(env)["package"]
    n, pool = len(_events(env.hub)), set(env.ag.pool)
    d = _import(env, pkg, dry_run=True).json()
    assert d["status"] == "dry_run" and d["dry_run"] is True
    assert d["goal"] == pkg["ledger"]["goal"]
    assert d["id_conflict"] is True
    assert d["entry_counts"]["core"] == 1
    assert len(_events(env.hub)) == n and set(env.ag.pool) == pool


def test_import_body_strict(env) -> None:
    r = env.c.post("/admin/ledgers/import", json={"package": {}, "dryRun": True})
    assert r.status_code == 400 and "dryRun" in r.json()["detail"]


# ── 建本 ────────────────────────────────────────────────────────────────────


def test_create_user_identity_anchor_and_no_activation(env) -> None:
    before = env.ag.activation.snapshot()
    n = len(_events(env.hub))
    with structlog.testing.capture_logs() as logs:
        r = env.c.post("/admin/ledgers/create", json={
            "goal": "先定目标", "core": ["约束一"], "next": ["第一步"], "origin": "cli"})
    assert r.status_code == 200, r.text
    d = r.json()
    lid = d["ledger_id"]
    led = env.ag.pool[lid]
    assert (led.goal, led.goal_source, led.goal_verbatim) == ("先定目标", "user", "先定目标")
    assert led.title == "先定目标" and led.rev == 1 and led.last_writer == "user"
    assert led.matter_id == ledger_anchor_matter_id(lid)
    assert [(e.text, e.source) for e in led.sections["core"]] == [("约束一", "user")]
    assert env.ag.activation.snapshot() == before
    new = _events(env.hub)[n:]
    assert [(t, p["origin"]) for t, p in new] == [("ledger_user_edit", "cli_create")]
    assert _replayed(env.hub)[lid] == led
    assert [x for x in logs if x["event"] == "ledger_user_created"]


def test_create_shows_up_in_next_turn_candidates(env) -> None:
    lid = env.c.post("/admin/ledgers/create", json={"goal": "迁移交接流程", "title": "交接流程迁移"}).json()["ledger_id"]
    msgs = env.ag.insert_ledger_block([{"role": "user", "content": "帮我做交接流程迁移"}], "cc",
                                      session_id="sx")
    assert lid in json.dumps(msgs, ensure_ascii=False)


@pytest.mark.parametrize("body", [{"goal": "   "}, {}, {"goal": "g", "origin": "model"}, {"goal": "g", "orgin": "cli"}])
def test_create_rejects(env, body) -> None:
    n = len(_events(env.hub))
    r = env.c.post("/admin/ledgers/create", json=body)
    assert r.status_code == 400, r.text
    assert len(_events(env.hub)) == n


# ── 改本 ────────────────────────────────────────────────────────────────────


def _edit(env, **body):
    return env.c.post(f"/admin/ledgers/{LID}/edit", json=body)


def test_edit_goal_goes_through_update_goal(env) -> None:
    with structlog.testing.capture_logs() as logs:
        r = _edit(env, expected_rev=3, goal="新目标")
    assert r.status_code == 200, r.text
    led = env.ag.pool[LID]
    assert (led.goal, led.goal_source, led.goal_revisions, led.rev, led.last_writer) == (
        "新目标", "user", 1, 4, "user")
    assert _events(env.hub)[-1][1]["origin"] == "cli_edit"
    assert _events(env.hub)[-1][0] == "ledger_user_edit"
    assert _replayed(env.hub)[LID] == led
    assert [x for x in logs if x["event"] == "ledger_user_edited"][0]["goal_changed"] is True


def test_edit_same_goal_is_noop_no_event_no_revision(env) -> None:
    n = len(_events(env.hub))
    goal = env.ag.pool[LID].goal
    d = _edit(env, expected_rev=3, goal=goal).json()
    assert d["status"] == "noop"
    assert len(_events(env.hub)) == n
    assert env.ag.pool[LID].goal_revisions == 0 and env.ag.pool[LID].rev == 3


def test_edit_sections_only_named_ones_and_goal_untouched(env) -> None:
    before = env.ag.pool[LID]
    r = _edit(env, expected_rev=3, sections={"next": [{"text": "用户加的", "source": "user"}]})
    assert r.status_code == 200, r.text
    led = env.ag.pool[LID]
    assert [e.text for e in led.sections["next"]] == ["用户加的"]
    assert led.sections["core"] == before.sections["core"]
    assert (led.goal, led.goal_source, led.goal_revisions) == (before.goal, "model", 0)


@pytest.mark.parametrize("rev", [2, 4])
def test_edit_stale_rev_409_writes_nothing(env, rev) -> None:
    n = len(_events(env.hub))
    before = env.ag.pool[LID].model_dump(mode="json")
    r = _edit(env, expected_rev=rev, goal="不该写入")
    assert r.status_code == 409
    d = r.json()
    assert d["current_rev"] == 3 and d["expected_rev"] == rev and "nothing was written" in d["detail"]
    assert d["ledger"]["goal"] == before["goal"]
    assert env.ag.pool[LID].model_dump(mode="json") == before
    assert len(_events(env.hub)) == n


def test_edit_expected_rev_required(env) -> None:
    n = len(_events(env.hub))
    r = _edit(env, goal="x")
    assert r.status_code == 400 and "expected_rev" in r.json()["error"]["message"]
    assert len(_events(env.hub)) == n


@pytest.mark.parametrize("body", [
    {"goal": "  "},
    {"origin": "model", "goal": "冒充"},
    {"sections": {"nxet": [{"text": "拼错段名"}]}},
    {"sections": {"goal": [{"text": "x"}]}},
    {"sections": {"next": [{"text": "x", "source": "bogus"}]}},
    {"new_sections": ["Bad Name"]},
])
def test_edit_rejects_and_writes_nothing(env, body) -> None:
    n = len(_events(env.hub))
    before = env.ag.pool[LID].model_dump(mode="json")
    r = _edit(env, expected_rev=3, **body)
    assert r.status_code == 400, r.text
    assert env.ag.pool[LID].model_dump(mode="json") == before
    assert len(_events(env.hub)) == n


def test_edit_new_section_must_be_declared(env) -> None:
    r = _edit(env, expected_rev=3, new_sections=["lessons"], sections={"lessons": [{"text": "教训"}]})
    assert r.status_code == 200, r.text
    assert "lessons" in env.ag.pool[LID].section_order
    assert _replayed(env.hub)[LID].sections["lessons"][0].text == "教训"


def test_edit_closed_ledger_stays_closed_and_unknown_404(env) -> None:
    from bladex_core.ledger import close_ledger
    env.ag.pool[LID] = close_ledger(env.ag.pool[LID])
    assert _edit(env, expected_rev=3, goal="改已关的本").status_code == 200
    assert env.ag.pool[LID].status == "closed"
    assert env.c.post("/admin/ledgers/ldg-nope/edit", json={"expected_rev": 0}).status_code == 404


def test_model_rev_lock_sees_cli_edit(env) -> None:
    """模型拿旧 rev 写会被现有乐观锁拒：CLI 改本让 rev 前进，这条才接得上。"""
    assert _edit(env, expected_rev=3, title="改名").json()["rev"] == 4
    assert env.ag.pool[LID].rev == 4 and env.ag.pool[LID].last_writer == "user"


# ── MQ-L107 ─────────────────────────────────────────────────────────────────


def test_user_edit_goal_change_goes_through_update_goal(env) -> None:
    dump = env.ag.pool[LID].model_dump(mode="json")
    dump["goal"] = "直编改的目标"          # goal_source 仍是 md 带回来的 model
    r = env.c.post("/admin/ledgers/user-edit", json={"ledger": dump})
    assert r.status_code == 200, r.text
    led = env.ag.pool[LID]
    assert (led.goal, led.goal_source, led.goal_revisions) == ("直编改的目标", "user", 1)
    assert _replayed(env.hub)[LID] == led


def test_user_edit_without_goal_change_is_as_before(env) -> None:
    dump = env.ag.pool[LID].model_dump(mode="json")
    dump["sections"]["next"] = [{"text": "只改条目", "source": "user", "ref": "", "writer": ""}]
    assert env.c.post("/admin/ledgers/user-edit", json={"ledger": dump}).status_code == 200
    led = env.ag.pool[LID]
    assert (led.goal_source, led.goal_revisions, led.rev) == ("model", 0, 4)
    assert "origin" not in _events(env.hub)[-1][1], "Flash 与旧调用方的载荷形状不变"


def test_every_write_path_only_emits_ledger_user_edit(env) -> None:
    n = len(_events(env.hub))
    pkg = _export(env)["package"]
    _import(env, pkg)
    env.c.post("/admin/ledgers/create", json={"goal": "g"})
    _edit(env, expected_rev=3, goal="g2")
    assert {t for t, _ in _events(env.hub)[n:]} == {"ledger_user_edit"}


def test_behavior_counts_user_created_ledger_created_time() -> None:
    """用户建的本只有 `ledger_user_edit`：`created` 不能是 None，否则寿命统计把它跳过。"""
    from bladex_core.behavior import _ledger_facets
    now = datetime(2026, 11, 2, tzinfo=UTC)
    dump = {"ledger_id": LID, "created_at": "2026-11-01T00:00:00+00:00", "last_writer": "user",
            "status": "active", "sections": {}}
    facets, *_ = _ledger_facets([{"type": "ledger_user_edit", "ts": now, "payload": {"ledger": dump}}],
                                {}, now=now)
    assert facets[LID]["created"] == datetime(2026, 11, 1, tzinfo=UTC)
    # 已有 ledger_create 的本：后续 user_edit 不改 created
    later = {**dump, "created_at": "2026-11-01T12:00:00+00:00"}
    events = [{"type": "ledger_create", "ts": now, "payload": {"ledger": dump}},
              {"type": "ledger_user_edit", "ts": now, "payload": {"ledger": later}}]
    assert _ledger_facets(events, {}, now=now)[0][LID]["created"] == datetime(2026, 11, 1, tzinfo=UTC)


# ── 端到端：CLI → admin → Hub ────────────────────────────────────────────────


@pytest.fixture()
def cli_env(env, monkeypatch):
    from bladex_proxy.cli import ledger_cmds as ldg

    def transport(method, path, body=None, params=None, timeout=15.0):
        r = env.c.request(method, path, json=body, params=params)
        return r.status_code, r.json()

    monkeypatch.setattr(ldg, "_admin_call", transport)
    return env


def test_cli_create_edit_export_import_end_to_end(cli_env, tmp_path, capsys, isolated_home) -> None:
    from bladex_proxy import cli
    env = cli_env
    assert cli.main(["ledger", "create", "--goal", f"目标在 {ROOT}/plan.md", "--core", "约束",
                     "--next", "第一步"]) == 0
    lid = re.search(r"created (ldg-\w+)", capsys.readouterr().out).group(1)
    assert env.ag.pool[lid].rev == 1
    assert cli.main(["ledger", "edit", lid, "--goal", f"新目标在 {ROOT}/plan.md", "--add", "verified=查过了"]) == 0
    capsys.readouterr()
    led = env.ag.pool[lid]
    assert (led.rev, led.goal_source, led.goal_revisions) == (2, "user", 1)

    # 编辑一份 show 的输出：只动一条
    assert cli.main(["ledger", "show", lid]) == 0
    shown = capsys.readouterr().out
    edited = tmp_path / "e.md"
    edited.write_text(shown.replace("- 第一步", "- 第一步改过了"), encoding="utf-8")
    assert cli.main(["ledger", "edit", lid, "--from-file", str(edited)]) == 0
    capsys.readouterr()
    assert [e.text for e in env.ag.pool[lid].sections["next"]] == ["第一步改过了"]
    assert env.ag.pool[lid].sections["core"] == led.sections["core"]

    pkg_file = tmp_path / "p.json"
    assert cli.main(["ledger", "handoff", lid, "--format", "json", "--out", str(pkg_file)]) == 0
    capsys.readouterr()
    pkg = json.loads(pkg_file.read_text(encoding="utf-8"))
    api = env.c.get(f"/admin/ledgers/{lid}/handoff").json()["package"]
    for k in ("exported_at", "content_hash"):
        pkg.pop(k), api.pop(k)
    assert pkg == api, "CLI 与 API 同一本、同一 exposure 的包逐字相等（导出时刻除外）"

    before = env.ag.pool[lid].model_dump(mode="json")
    assert cli.main(["ledger", "handoff", "--import", str(pkg_file), "--dry-run"]) == 0
    assert "DRY RUN" in capsys.readouterr().out
    assert cli.main(["ledger", "handoff", "--import", str(pkg_file)]) == 0
    assert "already taken" in capsys.readouterr().out
    assert env.ag.pool[lid].model_dump(mode="json") == before
    imported = next(v for k, v in env.ag.pool.items() if k not in (lid, LID))
    assert imported.goal == "新目标在 ~/dev/proj/plan.md", "没有项目事件时项目根未知，退到家目录改写"
    assert imported.goal_source == "user"
    assert _replayed(env.hub)[imported.ledger_id] == imported
