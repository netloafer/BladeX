"""批 DS2 第三步 · 步骤 E：Flash 拒收用户直编的持久读数（G7，MQ-F16）。

🔴 同时钉住一个修前的缺陷：`id_mismatch` / `unparseable` 把快照更新成坏文件，
下一次有流量物化时用户的文件被投影静默覆盖（注释和旧用例都写着「不覆盖」，旧用例只跑到不物化的那一轮）。
"""

from __future__ import annotations

import json
import os
import re

import pytest
from bladex_core.flash import STATE_FILE
from bladex_core.ledger import ACTOR_MODEL, LedgerEntry, add_entry, ledger_md_path, new_ledger
from bladex_proxy.config import ProxyConfig
from bladex_proxy.flash_daemon import (
    REFUSED_ID_MISMATCH,
    REFUSED_LINE,
    REFUSED_UNPARSEABLE,
    FlashDaemon,
    read_user_edit_refused,
)
from bladex_proxy.server import create_app
from fastapi.testclient import TestClient

LID = "ldg-a1"


def _pool() -> dict:
    led = new_ledger(ledger_id=LID, title="任务甲", goal="把批二做完", goal_source="user",
                     created_at="2026-08-25T10:00:00Z")
    return {LID: add_entry(led, "next", LedgerEntry(text="写测试", source="model"), actor=ACTOR_MODEL)}


class _Env:
    def __init__(self, tmp_path) -> None:
        self.root = str(tmp_path)
        self.pool = _pool()
        self.events: list = []
        self.source_ok = True
        self.daemon = self.new_daemon()
        self.daemon.startup_check()
        self.path = ledger_md_path(self.root, "u1", LID)

    def _source(self):
        if not self.source_ok:
            raise RuntimeError("hub down")
        return self.pool, {}

    def new_daemon(self) -> FlashDaemon:
        return FlashDaemon(root=self.root, principal="u1", ledger_source=self._source,
                           emit_admin_event=lambda t, p: self.events.append((t, p)))

    def text(self) -> str:
        with open(self.path, encoding="utf-8") as f:
            return f.read()

    def write(self, text: str) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(text)

    def rows(self):
        return read_user_edit_refused(self.root)


@pytest.fixture()
def env(tmp_path) -> _Env:
    return _Env(tmp_path)


def _bad_line(env: _Env) -> str:
    return env.text().replace("- 写测试", "用户随手写的一行\n- 写测试")


def _bad_id(env: _Env) -> str:
    return re.sub(r"- ledger: `[^`]*`", "- ledger: `ldg-other`", env.text())


# ── 写入与移除 ───────────────────────────────────────────────────────────────

def test_clean_daemon_reports_an_empty_list_not_no_reading(env) -> None:
    assert env.rows() == []


def test_no_state_file_is_no_reading(tmp_path) -> None:
    assert read_user_edit_refused(str(tmp_path)) is None


def test_old_state_without_the_key_is_no_reading(tmp_path) -> None:
    (tmp_path / STATE_FILE).write_text(json.dumps({"baseline": {}}), encoding="utf-8")
    assert read_user_edit_refused(str(tmp_path)) is None


def test_corrupt_state_is_no_reading(tmp_path) -> None:
    (tmp_path / STATE_FILE).write_text("{not json", encoding="utf-8")
    assert read_user_edit_refused(str(tmp_path)) is None


def test_unreadable_line_is_recorded_with_all_four_fields(env) -> None:
    env.write(_bad_line(env))
    env.daemon.run_once()
    (row,) = env.rows()
    assert set(row) == {"ledger_id", "reason", "at", "kind"}   # 内部的 sha 不外露
    assert row["ledger_id"] == LID and row["kind"] == REFUSED_LINE
    assert "用户随手写的一行" in row["reason"] and row["at"]
    assert env.events == []


def test_id_mismatch_is_recorded(env) -> None:
    env.write(_bad_id(env))
    env.daemon.run_once()
    (row,) = env.rows()
    assert row["kind"] == REFUSED_ID_MISMATCH and LID in row["reason"]


def test_unexpected_parse_error_is_recorded(env, monkeypatch) -> None:
    env.write(env.text() + "\n- 新的一条\n")

    def boom(_text, _base):  # noqa: ANN001, ANN202
        raise ValueError("x" * 1000)

    monkeypatch.setattr(env.daemon, "_user_version", boom)
    env.daemon.run_once()
    (row,) = env.rows()
    assert row["kind"] == REFUSED_UNPARSEABLE and len(row["reason"]) == 300


@pytest.mark.parametrize("make_bad", [_bad_line, _bad_id, lambda e: "完全不是账本格式的东西\n"])
def test_refused_file_is_not_overwritten_by_the_next_traffic(env, make_bad) -> None:
    """修前的缺陷：后两种在下一次物化时被投影盖掉。"""
    bad = make_bad(env)
    env.write(bad)
    env.daemon.run_once()
    for _ in range(2):
        env.daemon.push()
        env.daemon.run_once()
    assert env.text() == bad, "用户的文件被覆盖了"
    assert len(env.rows()) == 1


def test_fixing_the_file_adopts_it_and_removes_the_record(env) -> None:
    env.write(_bad_line(env))
    env.daemon.run_once()
    env.write(env.text().replace("用户随手写的一行\n", "- 用户随手写的一行\n"))
    assert env.daemon.run_once()["user_edits"] == 1
    assert env.rows() == []
    assert len(env.events) == 1


def test_undoing_the_edit_removes_the_record(env) -> None:
    original = env.text()
    env.write(_bad_id(env))
    env.daemon.run_once()
    assert len(env.rows()) == 1
    env.write(original)
    env.daemon.run_once()
    assert env.rows() == []
    assert env.events == []


def test_deleting_the_file_removes_the_record(env) -> None:
    env.write(_bad_line(env))
    env.daemon.run_once()
    os.remove(env.path)
    env.daemon.run_once()
    assert env.rows() == []


def test_a_new_bad_version_replaces_the_record(env) -> None:
    env.write(_bad_line(env))
    env.daemon.run_once()
    env.write(env.text().replace("用户随手写的一行", "又一行读不懂的"))
    env.daemon.run_once()
    (row,) = env.rows()
    assert "又一行读不懂的" in row["reason"]


def test_refusals_pending_tracks_the_records(env) -> None:
    assert env.daemon.refusals_pending() is False
    env.write(_bad_line(env))
    env.daemon.run_once()
    assert env.daemon.refusals_pending() is True


# ── daemon 重启 ──────────────────────────────────────────────────────────────

def _age_the_record(env: _Env, at: str) -> None:
    """把盘上记录的时刻改旧。`at` 只精确到秒：不改的话，读回与重新判定的结果分不出来。"""
    path = os.path.join(env.root, STATE_FILE)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    data["user_edit_refused"][0]["at"] = at
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def test_restart_keeps_the_record_and_its_time(env) -> None:
    env.write(_bad_line(env))
    env.daemon.run_once()
    _age_the_record(env, "2020-01-01T00:00:00+00:00")
    bad = env.text()
    d2 = env.new_daemon()
    assert d2.refusals_pending() is True          # 读回了，不等重新判定
    d2.startup_check()
    (row,) = env.rows()
    assert row["at"] == "2020-01-01T00:00:00+00:00", "同一份内容不该算新的一次拒收"
    d2.push()
    d2.run_once()
    assert env.text() == bad


def test_restart_with_the_hub_down_does_not_lose_the_record(env) -> None:
    env.write(_bad_line(env))
    env.daemon.run_once()
    before = env.rows()
    env.source_ok = False
    d2 = env.new_daemon()
    with pytest.raises(RuntimeError):      # startup_check 读不到 Hub 时本来就抛（修前如此）
        d2.startup_check()
    assert env.rows() == before


def test_file_fixed_while_the_daemon_was_down_clears_on_restart(env) -> None:
    env.write(_bad_line(env))
    env.daemon.run_once()
    env.write(env.text().replace("用户随手写的一行\n", "- 用户随手写的一行\n"))
    env.new_daemon().startup_check()
    assert env.rows() == []
    assert len(env.events) == 1


def test_edit_undone_while_the_daemon_was_down_clears_on_restart(env) -> None:
    original = env.text()
    env.write(_bad_line(env))
    env.daemon.run_once()
    env.write(original)
    env.new_daemon().startup_check()
    assert env.rows() == []


def test_new_refusal_found_on_restart_is_recorded(env) -> None:
    env.write(_bad_line(env))
    env.new_daemon().startup_check()
    assert [r["kind"] for r in env.rows()] == [REFUSED_LINE]


# ── /admin/status 与 CLI 对账 ────────────────────────────────────────────────

def _status_flash(root: str, monkeypatch) -> dict:
    import tempfile

    monkeypatch.setenv("BLADEX_FLASH_PATH", root)
    d = tempfile.mkdtemp(prefix="ds2_flash_")
    app = create_app(ProxyConfig(upstream_model="openai/test", upstream_api_key="sk-fake",
                                 rocksdb_path=f"{d}/r", index_path=f"{d}/i"))
    with TestClient(app) as c:
        return c.get("/admin/status").json()["flash"]


def test_admin_status_carries_the_records(env, monkeypatch) -> None:
    env.write(_bad_line(env))
    env.daemon.run_once()
    assert _status_flash(env.root, monkeypatch)["user_edit_refused"] == env.rows()


def test_admin_status_no_reading_is_null(tmp_path, monkeypatch) -> None:
    assert _status_flash(str(tmp_path), monkeypatch)["user_edit_refused"] is None


def test_status_line_matches_admin_status(env, monkeypatch) -> None:
    from bladex_proxy.cli.lifecycle import _flash_refused_text

    env.write(_bad_line(env))
    env.daemon.run_once()
    rows = _status_flash(env.root, monkeypatch)["user_edit_refused"]
    n, text = _flash_refused_text()
    assert n == len(rows) == 1
    assert rows[0]["ledger_id"] in text and rows[0]["reason"] in text


def test_status_line_says_none_and_no_reading_differently(tmp_path, monkeypatch) -> None:
    from bladex_proxy.cli.lifecycle import _flash_refused_text

    monkeypatch.setenv("BLADEX_FLASH_PATH", str(tmp_path))
    n, text = _flash_refused_text()
    assert n is None and text.startswith("n/a")
    (tmp_path / STATE_FILE).write_text(json.dumps({"user_edit_refused": []}), encoding="utf-8")
    assert _flash_refused_text() == (0, "none refused")


def test_status_line_caps_how_many_it_lists(tmp_path, monkeypatch) -> None:
    from bladex_proxy.cli.lifecycle import _flash_refused_text

    monkeypatch.setenv("BLADEX_FLASH_PATH", str(tmp_path))
    rows = [{"ledger_id": f"l{i}", "reason": "r", "at": "t", "kind": "refused"} for i in range(5)]
    (tmp_path / STATE_FILE).write_text(json.dumps({"user_edit_refused": rows}), encoding="utf-8")
    n, text = _flash_refused_text()
    assert n == 5 and "and 2 more" in text and "l3" not in text
