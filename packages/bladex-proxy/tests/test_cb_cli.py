"""批 CB：`bladex ledger handoff | create | edit` 的 CLI 层。`_admin_call` 换录制桩，只验 CLI 自己的事。"""

from __future__ import annotations

import io
import json
import os
import stat
import sys
from pathlib import Path

import pytest
from bladex_core.ledger import LedgerEntry, new_ledger, render_ledger_md
from bladex_proxy import cli
from bladex_proxy.cli import ledger_cmds as ldg

LID = "ldg-cb0000000001"


class _Rec:
    def __init__(self, responses: dict | None = None):
        self.calls: list[tuple] = []
        self.responses = responses or {}

    def __call__(self, method, path, body=None, params=None, timeout=15.0):
        self.calls.append((method, path, body, params))
        return self.responses.get((method, path), (200, {"status": "ok", "ledger_id": LID, "rev": 8}))


@pytest.fixture(autouse=True)
def _isolate(tmp_path: Path, monkeypatch, isolated_home):
    monkeypatch.chdir(tmp_path)


def _use(monkeypatch, responses: dict | None = None) -> _Rec:
    rec = _Rec(responses)
    monkeypatch.setattr(ldg, "_admin_call", rec)
    return rec


def _ledger():
    led = new_ledger(ledger_id=LID, title="任务", goal="目标", goal_source="model",
                     created_at="2026-11-02T00:00:00+00:00")
    return led.model_copy(update={"rev": 7, "sections": {
        **led.sections,
        "verified": [LedgerEntry(text="带尾巴。  <sub>[tool · a + b]</sub>", source="model"),
                     LedgerEntry(text="普通", source="model", ref="r", writer="codex")],
        "next": [LedgerEntry(text="下一步", source="model")]}})


def _read_ok(led=None) -> tuple:
    led = led or _ledger()
    return (200, {"ledger": led.model_dump(mode="json"), "markdown": render_ledger_md(led)})


_EXPORT = ("GET", f"/admin/ledgers/{LID}/handoff")
_PKG = {"format_version": 1, "ledger": {"goal": "g"}, "content_hash": "abc"}


# ── handoff 导出 ─────────────────────────────────────────────────────────────


def test_export_default_is_markdown_to_stdout_with_public_exposure(monkeypatch, capsys) -> None:
    rec = _use(monkeypatch, {_EXPORT: (200, {"package": _PKG, "markdown": "# Handoff 页\n"})})
    assert cli.main(["ledger", "handoff", LID]) == 0
    assert rec.calls[0][3] == {"exposure": "public"}
    cap = capsys.readouterr()
    assert cap.out == "# Handoff 页\n", "stdout 只放包，能直接重定向"
    assert "personal information" in cap.err


def test_export_json_to_file_is_0600_even_if_it_existed(monkeypatch, tmp_path, capsys) -> None:
    _use(monkeypatch, {_EXPORT: (200, {"package": _PKG, "markdown": "m"})})
    out = tmp_path / "pkg.json"
    out.write_text("old")
    os.chmod(out, 0o644)
    assert cli.main(["ledger", "handoff", LID, "--out", str(out), "--format", "json",
                     "--exposure", "local"]) == 0
    assert stat.S_IMODE(out.stat().st_mode) == 0o600
    assert json.loads(out.read_text()) == _PKG
    assert capsys.readouterr().out == ""


def test_export_errors_are_named(monkeypatch, capsys) -> None:
    _use(monkeypatch, {_EXPORT: (404, {"status": "not_found", "detail": f"unknown ledger {LID}"})})
    assert cli.main(["ledger", "handoff", LID]) == 1
    assert "unknown ledger" in capsys.readouterr().err
    _use(monkeypatch, {_EXPORT: (400, {"error": {"message": "unknown exposure 'x'"}})})
    assert cli.main(["ledger", "handoff", LID, "--exposure", "x"]) == 1
    assert "unknown exposure" in capsys.readouterr().err
    assert cli.main(["ledger", "handoff", LID, "--format", "pdf"]) == 2


@pytest.mark.parametrize("argv", [
    ["ledger", "handoff"],
    ["ledger", "handoff", LID, "--import", "x.json"],
    ["ledger", "handoff", LID, "--dry-run"],
])
def test_handoff_argument_shapes_exit_2(monkeypatch, argv) -> None:
    rec = _use(monkeypatch)
    assert cli.main(argv) == 2
    assert rec.calls == []


def test_help_says_review_before_sending_and_trust_boundary(monkeypatch, capsys) -> None:
    with pytest.raises(SystemExit):
        cli.main(["ledger", "handoff", "--help"]) and None
        raise SystemExit
    text = " ".join(capsys.readouterr().out.split())
    assert "NOT filtered for personal information" in text
    assert "only import packages you trust" in text
    assert "catches damage, not forgery" in text


# ── handoff 导入 ─────────────────────────────────────────────────────────────


_IMPORT = ("POST", "/admin/ledgers/import")


def test_import_posts_package_and_reports_new_id(monkeypatch, tmp_path, capsys) -> None:
    rec = _use(monkeypatch, {_IMPORT: (200, {"status": "ok", "ledger_id": "ldg-new", "id_conflict": True})})
    f = tmp_path / "p.json"
    f.write_text(json.dumps(_PKG))
    assert cli.main(["ledger", "handoff", "--import", str(f)]) == 0
    assert rec.calls[0][2] == {"package": _PKG, "dry_run": False}
    out = capsys.readouterr().out
    assert "ldg-new" in out and "already taken" in out and "not active" in out


def test_import_dry_run_requires_server_ack(monkeypatch, tmp_path, capsys) -> None:
    f = tmp_path / "p.json"
    f.write_text(json.dumps(_PKG))
    _use(monkeypatch, {_IMPORT: (200, {"status": "dry_run", "dry_run": True, "ledger_id": "ldg-x",
                                       "goal": "第一行\n第二行", "title": "t", "entry_counts": {"core": 2}})})
    assert cli.main(["ledger", "handoff", "--import", str(f), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "第一行" in out and "第二行" in out
    _use(monkeypatch, {_IMPORT: (200, {"status": "ok", "ledger_id": "ldg-x"})})
    assert cli.main(["ledger", "handoff", "--import", str(f), "--dry-run"]) == 2
    assert "REFUSING" in capsys.readouterr().out


def test_import_markdown_file_points_to_json(monkeypatch, tmp_path, capsys) -> None:
    rec = _use(monkeypatch)
    f = tmp_path / "p.md"
    f.write_text("# Handoff: 页\n")
    assert cli.main(["ledger", "handoff", "--import", str(f)]) == 1
    assert "--format json" in capsys.readouterr().err
    assert rec.calls == []


def test_import_rejections_print_server_message(monkeypatch, tmp_path, capsys) -> None:
    f = tmp_path / "p.json"
    f.write_text(json.dumps(_PKG))
    _use(monkeypatch, {_IMPORT: (400, {"error": {"message": "unsupported format_version 2", "code": "version"}})})
    assert cli.main(["ledger", "handoff", "--import", str(f)]) == 1
    assert "format_version 2" in capsys.readouterr().err


def test_import_oversize_and_missing_file_do_not_call_server(monkeypatch, tmp_path, capsys) -> None:
    rec = _use(monkeypatch)
    monkeypatch.setenv("BLADEX_HANDOFF_IMPORT_MAX_BYTES", "5")
    f = tmp_path / "p.json"
    f.write_text(json.dumps(_PKG))
    assert cli.main(["ledger", "handoff", "--import", str(f)]) == 1
    assert cli.main(["ledger", "handoff", "--import", str(tmp_path / "nope.json")]) == 1
    assert rec.calls == []


# ── create ──────────────────────────────────────────────────────────────────


def test_create_sends_user_origin_and_says_not_active(monkeypatch, capsys) -> None:
    rec = _use(monkeypatch)
    assert cli.main(["ledger", "create", "--goal", "先定目标", "--title", "T", "--core", "c1",
                     "--core", "c2", "--next", "n1"]) == 0
    assert rec.calls[0][:2] == ("POST", "/admin/ledgers/create")
    assert rec.calls[0][2] == {"goal": "先定目标", "title": "T", "core": ["c1", "c2"], "verified": [],
                               "open": [], "next": ["n1"], "origin": "cli"}
    assert "not active" in capsys.readouterr().out


def test_create_goal_file_and_argument_errors(monkeypatch, tmp_path) -> None:
    rec = _use(monkeypatch)
    f = tmp_path / "g.txt"
    f.write_text("文件里的目标\n")
    assert cli.main(["ledger", "create", "--goal-file", str(f)]) == 0
    assert rec.calls[-1][2]["goal"] == "文件里的目标"
    n = len(rec.calls)
    assert cli.main(["ledger", "create"]) == 2
    assert cli.main(["ledger", "create", "--goal", "a", "--goal-file", str(f)]) == 2
    assert len(rec.calls) == n


# ── edit ────────────────────────────────────────────────────────────────────


_READ = ("GET", f"/admin/ledgers/{LID}")
_EDIT = ("POST", f"/admin/ledgers/{LID}/edit")


def test_edit_goal_sends_read_rev_as_expected_rev(monkeypatch, capsys) -> None:
    rec = _use(monkeypatch, {_READ: _read_ok()})
    assert cli.main(["ledger", "edit", LID, "--goal", "新目标"]) == 0
    assert rec.calls[-1][2] == {"goal": "新目标", "origin": "cli", "expected_rev": 7}
    assert "rev 8" in capsys.readouterr().out


def test_edit_explicit_expected_rev_wins(monkeypatch) -> None:
    rec = _use(monkeypatch, {_READ: _read_ok()})
    assert cli.main(["ledger", "edit", LID, "--title", "T", "--expected-rev", "3"]) == 0
    assert rec.calls[-1][2]["expected_rev"] == 3


def test_edit_409_says_nothing_written(monkeypatch, capsys) -> None:
    _use(monkeypatch, {_READ: _read_ok(), _EDIT: (409, {"status": "conflict", "current_rev": 9, "expected_rev": 7})})
    assert cli.main(["ledger", "edit", LID, "--goal", "x"]) == 1
    err = capsys.readouterr().err
    assert "nothing was written" in err and "rev 9" in err


def test_edit_add_remove_keep_other_entries_verbatim(monkeypatch) -> None:
    rec = _use(monkeypatch, {_READ: _read_ok()})
    assert cli.main(["ledger", "edit", LID, "--add", "next=再加一条", "--remove", "next:0",
                     "--remove", "verified:1"]) == 0
    sections = rec.calls[-1][2]["sections"]
    assert [e["text"] for e in sections["next"]] == ["再加一条"]
    assert sections["next"][0]["source"] == "user"
    assert [e["text"] for e in sections["verified"]] == ["带尾巴。  <sub>[tool · a + b]</sub>"]
    assert sections["verified"][0]["source"] == "model"


@pytest.mark.parametrize("args", [["--remove", "next:9"], ["--remove", "next"], ["--add", "next"],
                                  ["--add", "next=  "]])
def test_edit_bad_flag_values_do_not_post(monkeypatch, args, capsys) -> None:
    rec = _use(monkeypatch, {_READ: _read_ok()})
    assert cli.main(["ledger", "edit", LID, *args]) == 1
    assert [c for c in rec.calls if c[0] == "POST"] == []


def test_edit_from_file_only_sends_what_changed(monkeypatch, tmp_path) -> None:
    led = _ledger()
    rec = _use(monkeypatch, {_READ: _read_ok(led)})
    f = tmp_path / "e.md"
    f.write_text(render_ledger_md(led).replace("- 下一步", "- 下一步改过了"))
    assert cli.main(["ledger", "edit", LID, "--from-file", str(f)]) == 0
    body = rec.calls[-1][2]
    assert set(body) == {"sections", "origin", "expected_rev"}
    assert set(body["sections"]) == {"next"}, "verified 没动，不进请求"


def test_edit_from_file_unchanged_makes_no_post(monkeypatch, tmp_path, capsys) -> None:
    led = _ledger()
    rec = _use(monkeypatch, {_READ: _read_ok(led)})
    f = tmp_path / "e.md"
    f.write_text(render_ledger_md(led))
    assert cli.main(["ledger", "edit", LID, "--from-file", str(f)]) == 0
    assert [c for c in rec.calls if c[0] == "POST"] == []
    assert "Nothing changed" in capsys.readouterr().out


def test_edit_from_file_with_other_flags_exit_2(monkeypatch, tmp_path) -> None:
    rec = _use(monkeypatch, {_READ: _read_ok()})
    assert cli.main(["ledger", "edit", LID, "--goal", "g", "--from-file", "x.md"]) == 2
    assert rec.calls == []


def test_edit_no_flags_no_tty_never_blocks_and_never_opens_editor(monkeypatch, capsys) -> None:
    rec = _use(monkeypatch, {_READ: _read_ok()})
    monkeypatch.setenv("EDITOR", "/bin/false-should-not-run")
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    monkeypatch.setattr(ldg.subprocess, "run", lambda *a, **k: pytest.fail("editor must not start"))
    assert cli.main(["ledger", "edit", LID]) == 2
    assert "needs a terminal" in capsys.readouterr().err
    assert rec.calls == []


def _fake_tty(monkeypatch) -> None:
    class _T(io.StringIO):
        def isatty(self) -> bool:
            return True
    monkeypatch.setattr(sys, "stdin", _T(""))
    monkeypatch.setattr(sys, "stdout", _T())


def test_edit_opens_editor_with_tty_and_applies_changes(monkeypatch) -> None:
    led = _ledger()
    rec = _use(monkeypatch, {_READ: _read_ok(led)})
    _fake_tty(monkeypatch)
    monkeypatch.setenv("EDITOR", "myeditor --wait")

    def _editor(argv, check=False):
        assert argv[:2] == ["myeditor", "--wait"]
        assert stat.S_IMODE(os.stat(argv[2]).st_mode) == 0o600
        text = Path(argv[2]).read_text()
        Path(argv[2]).write_text(text.replace("- 下一步", "- 编辑器改的"))
        return ldg.subprocess.CompletedProcess(argv, 0)
    monkeypatch.setattr(ldg.subprocess, "run", _editor)
    assert cli.main(["ledger", "edit", LID]) == 0
    assert [e["text"] for e in rec.calls[-1][2]["sections"]["next"]] == ["编辑器改的"]
    assert set(rec.calls[-1][2]["sections"]) == {"next"}


def test_edit_editor_failure_writes_nothing(monkeypatch) -> None:
    rec = _use(monkeypatch, {_READ: _read_ok()})
    _fake_tty(monkeypatch)
    monkeypatch.setenv("EDITOR", "myeditor")
    monkeypatch.setattr(ldg.subprocess, "run",
                        lambda argv, check=False: ldg.subprocess.CompletedProcess(argv, 1))
    assert cli.main(["ledger", "edit", LID]) == 1
    assert [c for c in rec.calls if c[0] == "POST"] == []
