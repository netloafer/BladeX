"""MQ-L94：doctor 看得见同 agent 跨 scope 的同名账本。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from bladex_proxy import cli
from bladex_proxy.cli.ledger_cmds import _doctor_findings


def _row(lid: str, *, agent: str = "codex", project: str = "Global",
         status: str = "active") -> dict[str, Any]:
    return {
        "ledger_id": lid,
        "title": "同一件事",
        "status": status,
        "matter_id": "m-x",
        "parent_ledger_id": "",
        "goal": "",
        "goal_source": "model",
        "created_at": "2026-09-29T10:00:00Z",
        "updated_at": "2026-09-29T11:00:00Z",
        "active_in": [{"agent": agent, "project": project}],
        "entry_counts": {"core": 1, "verified": 1, "open": 0, "next": 0},
    }


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path: Path, monkeypatch, isolated_home) -> None:
    monkeypatch.setenv("BLADEX_HOME", str(tmp_path))


class _Recorder:
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows

    def __call__(self, method: str, path: str, *args: Any, **kwargs: Any) -> tuple[int, dict[str, Any]]:
        if path == "/admin/ledgers":
            return 200, {"enabled": True, "ledgers": self.rows, "total": len(self.rows)}
        return 404, {"status": "not_found"}


def test_same_agent_global_and_project_is_cross_scope_duplicate() -> None:
    findings = _doctor_findings([
        _row("ldg-global", project="Global"),
        _row("ldg-project", project="p-123"),
    ], {})

    assert findings == [{
        "kind": "duplicate_title_cross_scope",
        "ledger_ids": ["ldg-global", "ldg-project"],
        "detail": "same title after normalisation for the same agent across scopes: '同一件事'",
        "action": "bladex ledger show ldg-global",
    }]


def test_different_agents_with_same_title_are_not_flagged() -> None:
    findings = _doctor_findings([
        _row("ldg-codex", agent="codex"),
        _row("ldg-claude", agent="claude-code"),
    ], {})

    assert findings == []


def test_closed_ledgers_do_not_join_duplicate_comparison() -> None:
    findings = _doctor_findings([
        _row("ldg-open", project="Global"),
        _row("ldg-closed", project="p-123", status="closed"),
    ], {})

    assert findings == []


def test_same_scope_duplicate_title_keeps_original_kind() -> None:
    findings = _doctor_findings([
        _row("ldg-a", project="Global"),
        _row("ldg-b", project="Global"),
    ], {})

    assert findings == [{
        "kind": "duplicate_title",
        "ledger_ids": ["ldg-a", "ldg-b"],
        "detail": "same title after normalisation: '同一件事'",
        "action": "bladex ledger show ldg-a",
    }]


def test_cli_reports_cross_scope_duplicate(monkeypatch, capsys) -> None:
    from bladex_proxy.cli import ledger_cmds

    rows = [_row("ldg-global", project="Global"),
            _row("ldg-project", project="p-123")]
    monkeypatch.setattr(ledger_cmds, "_admin_call", _Recorder(rows))

    rc = cli.main(["ledger", "doctor", "--json", "--no-details"])

    payload = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert [f["kind"] for f in payload["findings"]] == ["duplicate_title_cross_scope"]
    assert payload["findings"][0]["ledger_ids"] == ["ldg-global", "ldg-project"]
