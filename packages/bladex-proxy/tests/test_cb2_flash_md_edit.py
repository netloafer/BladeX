"""批 CB2 · MQ-F15：Flash 捕获用户直编不改用户没碰的条目（拍板 (43)-a-①）。

渲染再解析不是恒等的（`report-batch-CB-20261102.md` §3.3）：池里 24 条是两种真实形态，修前用户在这本里
改任何一条，这些条目就被 `parse_ledger_md` 一起改写。内容是构造的，形态是真实的。

走真实 `/admin/ledgers/user-edit`（真 AgencyRuntime + 真端点），不桩 emit 的语义。
"""

from __future__ import annotations

import os

import pytest
import structlog.testing
from bladex_core.ledger import (
    LedgerEntry,
    ledger_md_path,
    new_ledger,
    parse_ledger_md,
    render_ledger_md,
    replay_ledger_events,
)
from bladex_proxy.agency import AgencyRuntime
from bladex_proxy.config import ProxyConfig
from bladex_proxy.flash_daemon import FlashDaemon
from bladex_proxy.models import AdminEventType
from bladex_proxy.server import create_app
from fastapi.testclient import TestClient

LID = "ldg-cb2000000001"


def _shape_a(i: int) -> LedgerEntry:
    return LedgerEntry(text=f"结论{i}。  <sub>[tool · install.md + models.md {i}]</sub>", source="model")


def _shape_b(i: int) -> LedgerEntry:
    return LedgerEntry(text=f"写入{i}成功", source="tool", ref=f"write_file {i} · @hermes:default",
                       writer="hermes:default")


PLAIN = LedgerEntry(text="普通条目", source="model", ref="r1", writer="codex")


def _seed():
    led = new_ledger(ledger_id=LID, title="任务", goal="第一段\n第二段", goal_source="model",
                     goal_verbatim="用户原话", created_at="2026-11-02T00:00:00+00:00")
    return led.model_copy(update={
        "rev": 5, "last_writer": "codex",
        "closed_by_agent": "codex",        # MD 不携带的字段：不许被直编清掉
        "sections": {**led.sections,
                     "verified": [*[_shape_a(i) for i in range(3)], *[_shape_b(i) for i in range(21)]],
                     "next": [PLAIN, LedgerEntry(text="另一条", source="model")]}})


class _Ev:
    def __init__(self, etype: str, payload: dict):
        self.event_type, self.payload, self.ts = etype, payload, None


class _Hub:
    def __init__(self) -> None:
        self.events: list[_Ev] = []

    def append_admin_event(self, etype, matter_id, **payload):
        self.events.append(_Ev(getattr(etype, "value", str(etype)), payload))

    def scan_admin_events(self):
        yield from (("k", e) for e in self.events)

    def scan_tombstones(self):
        return iter(())

    def catch_up(self):
        return None


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("BLADEX_AUTH_ENABLED", "false")
    hub = _Hub()
    hub.append_admin_event(AdminEventType.LEDGER_CREATE, "", ledger=_seed().model_dump())
    app = create_app(ProxyConfig())
    with TestClient(app) as c:
        ag = AgencyRuntime(hub=hub)
        c.app.state.agency = ag

        def source():
            return replay_ledger_events([{"type": e.event_type, "payload": e.payload} for e in hub.events])

        def emit(_etype, payload):
            r = c.post("/admin/ledgers/user-edit", json=payload)
            assert r.status_code == 200, r.text
            return r.json()

        d = FlashDaemon(root=str(tmp_path), principal="u1", ledger_source=source, emit_admin_event=emit)
        d.run_once()
        yield d, ag, ledger_md_path(str(tmp_path), "u1", LID)


def _edit(path: str, old: str, new: str) -> None:
    text = open(path, encoding="utf-8").read()
    assert old in text
    open(path, "w", encoding="utf-8").write(text.replace(old, new))


def test_precondition_these_shapes_really_change_on_roundtrip() -> None:
    led = _seed()
    parsed = parse_ledger_md(render_ledger_md(led), base=led)
    assert sum(a != b for a, b in zip(led.sections["verified"], parsed.sections["verified"], strict=True)) == 24


def test_editing_one_entry_leaves_all_24_untouched(env) -> None:
    d, ag, path = env
    _edit(path, "- 另一条  <sub>[model]</sub>", "- 另一条改过了")
    assert d.run_once()["user_edits"] == 1
    led = ag.pool[LID]
    assert led.sections["verified"] == _seed().sections["verified"], "24 条逐字不变，含 source/ref/writer"
    assert led.sections["next"][0] == PLAIN
    assert (led.sections["next"][1].text, led.sections["next"][1].source) == ("另一条改过了", "user")
    assert led.rev == 6


def test_fields_md_does_not_carry_survive(env) -> None:
    d, ag, path = env
    _edit(path, "- 另一条  <sub>[model]</sub>", "- 另一条改过了")
    d.run_once()
    led = ag.pool[LID]
    assert led.goal_verbatim == "用户原话" and led.closed_by_agent == "codex"
    assert led.goal == "第一段\n第二段" and led.goal_source == "model" and led.goal_revisions == 0


def test_goal_edit_goes_through_update_goal(env) -> None:
    d, ag, path = env
    _edit(path, "第二段", "第二段改了")
    d.run_once()
    led = ag.pool[LID]
    assert (led.goal, led.goal_source, led.goal_revisions) == ("第一段\n第二段改了", "user", 1)
    assert led.sections["verified"] == _seed().sections["verified"]


def test_delete_one_and_add_one_take_effect(env) -> None:
    d, ag, path = env
    _edit(path, "- 另一条  <sub>[model]</sub>", "- 用户加的")
    _edit(path, "- 普通条目  <sub>[model · r1 · @codex]</sub>\n", "")
    d.run_once()
    assert [(e.text, e.source) for e in ag.pool[LID].sections["next"]] == [("用户加的", "user")]
    assert ag.pool[LID].sections["verified"] == _seed().sections["verified"]


def test_status_edit_in_file_still_works_and_title_too(env) -> None:
    """元数据行是直编的既有能力，改了才生效；没改的不动。"""
    d, ag, path = env
    _edit(path, "# 任务", "# 新标题")
    _edit(path, "- status: active", "- status: closed")
    d.run_once()
    led = ag.pool[LID]
    assert led.title == "新标题" and led.status == "closed"
    assert led.closed_by_agent == "codex"


def test_unreadable_line_is_reported_once_kept_on_disk_and_not_emitted(env) -> None:
    d, ag, path = env
    before = ag.pool[LID].model_dump(mode="json")
    _edit(path, "- 另一条  <sub>[model]</sub>", "用户随手写的一行\n- 另一条改过了")
    disk = open(path, encoding="utf-8").read()
    with structlog.testing.capture_logs() as logs:
        assert d.run_once()["user_edits"] == 0
        d.push()
        d.run_once()
    refused = [x for x in logs if x["event"] == "flash_daemon_user_edit_refused"]
    assert len(refused) == 1, "同一份内容只报一次"
    assert "用户随手写的一行" in refused[0]["error"]
    assert ag.pool[LID].model_dump(mode="json") == before
    assert open(path, encoding="utf-8").read() == disk, "不采纳，也不覆盖用户的文件"
    # 用户修好之后照常采纳
    _edit(path, "用户随手写的一行\n", "- 用户随手写的一行\n")
    assert d.run_once()["user_edits"] == 1
    assert "用户随手写的一行" in [e.text for e in ag.pool[LID].sections["next"]]
    assert ag.pool[LID].sections["verified"] == _seed().sections["verified"]


def test_id_mismatch_still_guarded(env) -> None:
    d, ag, path = env
    before = ag.pool[LID].model_dump(mode="json")
    _edit(path, f"- ledger: `{LID}`", "- ledger: `ldg-other0000001`")
    assert d.run_once()["user_edits"] == 0
    assert ag.pool[LID].model_dump(mode="json") == before


def test_reemit_and_offline_paths_use_the_same_merge(env, tmp_path) -> None:
    """三个调用点走同一个函数：停机期间的直编（启动对账）也不改未碰条目。"""
    d, ag, path = env
    _edit(path, "- 另一条  <sub>[model]</sub>", "- 停机时改的")
    assert os.path.isfile(path)
    d2 = FlashDaemon(root=str(tmp_path), principal="u1", ledger_source=d._source, emit_admin_event=d._emit)
    d2.startup_check()
    led = ag.pool[LID]
    assert led.sections["next"][1].text == "停机时改的"
    assert led.sections["verified"] == _seed().sections["verified"]
    src = open(os.path.join(os.path.dirname(__file__), "..", "bladex_proxy", "flash_daemon.py"),
               encoding="utf-8").read()
    body = src[src.index("def _user_version"):src.index("# ── 收件箱")]
    assert src.count("parse_ledger_md(") == body.count("parse_ledger_md("), "解析只在 _user_version 里"
    assert src.count("self._user_version(") == 3, "采纳 / 捕获 / 重发三处都走它"
