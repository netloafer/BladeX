"""MQ-I30：长耗时 admin 写命令的客户端等待不许误报 unreachable。"""

from __future__ import annotations

import re
import urllib.error
from pathlib import Path

import pytest
from bladex_core.flags import MEMORY_NUMERIC_DEFAULTS, flag_number
from bladex_proxy.cli import _admin_long_timeout_s, memory_cmds
from bladex_proxy.cli import main as cli_main

_ENV_EXAMPLE = Path(__file__).resolve().parents[3] / "config" / ".env.example"


def test_admin_long_timeout_default_matches_env_example(monkeypatch) -> None:
    """默认值只有一个真相源：flags.py 与 config/.env.example 必须一致。"""
    name = "BLADEX_ADMIN_LONG_TIMEOUT_S"
    monkeypatch.delenv(name, raising=False)

    assert MEMORY_NUMERIC_DEFAULTS[name] == 900
    assert flag_number(name) == 900
    assert _admin_long_timeout_s() == 900

    match = re.search(rf'^export {name}="([^"]+)"$', _ENV_EXAMPLE.read_text(encoding="utf-8"),
                      re.MULTILINE)
    assert match is not None
    assert float(match.group(1)) == 900


@pytest.mark.parametrize("error", [
    TimeoutError("timed out"),
    urllib.error.URLError(TimeoutError("timed out")),
])
def test_tombstone_write_timeout_uses_knob_and_reports_check_command(
    monkeypatch, capsys, error
) -> None:
    """非 dry-run 超时：请求已发出、等待超时，并给出安全核对命令。"""
    from bladex_proxy import cli

    seen: dict[str, object] = {}

    def fake_urlopen(_req, timeout):
        seen["timeout"] = timeout
        raise error

    monkeypatch.setenv("BLADEX_ADMIN_LONG_TIMEOUT_S", "1234")
    monkeypatch.setattr(cli, "_load_env_file", lambda: None)
    monkeypatch.setattr(cli, "_chdir_to_deployment_root", lambda: None)
    monkeypatch.setattr(cli, "_admin_base", lambda: "http://proxy.invalid")
    monkeypatch.setattr(cli, "_admin_key", lambda: "test-key")
    monkeypatch.setattr(cli.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(memory_cmds, "_require_server_contract", lambda _what: None)

    rc = cli_main(["memory", "tombstone-project", "p-demo", "p-other", "--yes"])

    out = capsys.readouterr().out
    assert rc == 1
    assert seen["timeout"] == 1234
    assert "unreachable" not in out
    assert "request was issued" in out
    assert "timed out after 1234s" in out
    assert "bladex memory tombstone-project p-demo p-other --dry-run" in out


def test_ledger_tombstone_with_memory_timeout_uses_knob_and_reports_check_commands(
    monkeypatch, capsys
) -> None:
    """with-memory 墓碑是非 dry-run 长耗时写：用旋钮并给出两条核对命令。"""
    from bladex_proxy import cli
    from bladex_proxy.cli import ledger_cmds

    seen: dict[str, object] = {}

    def fake_urlopen(_req, timeout):
        seen["timeout"] = timeout
        raise TimeoutError("timed out")

    monkeypatch.setenv("BLADEX_ADMIN_LONG_TIMEOUT_S", "1234")
    monkeypatch.setattr(cli, "_load_env_file", lambda: None)
    monkeypatch.setattr(cli, "_chdir_to_deployment_root", lambda: None)
    monkeypatch.setattr(cli, "_admin_base", lambda: "http://proxy.invalid")
    monkeypatch.setattr(cli, "_admin_key", lambda: "test-key")
    monkeypatch.setattr(cli.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(ledger_cmds, "_require_server_contract", lambda _what: None)

    rc = cli_main(["ledger", "tombstone", "l-demo", "--with-memory", "--yes"])

    out = capsys.readouterr().out
    assert rc == 1
    assert seen["timeout"] == 1234
    assert "unreachable" not in out
    assert "request was issued" in out
    assert "timed out after 1234s" in out
    assert "bladex ledger tombstone l-demo --with-memory --dry-run" in out
    assert "bladex ledger show l-demo" in out
    assert "Do not run the write again until you have checked it." in out
