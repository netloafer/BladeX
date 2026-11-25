"""批 DS2 第二步 · 步骤 C：CLI 对账缺口。

- `bladex usage --since / --until / --tz`：绝对窗口，与 `--days` 互斥（MQ-V45）；
- `bladex project list [--json]`：只调 `/admin/projects`（G1）；
- `config show --effective` 表头、`config check` 校验 `bladex.toml`（MQ-V46）。

页面与 CLI 的逐字对账在 `test_dashboard.py`（要 node）；这里钉 CLI 自身的行为，没有 node 也能跑。
"""

from __future__ import annotations

import ast
import inspect
import json
import os
import textwrap
from pathlib import Path

import pytest
from bladex_proxy import cli
from bladex_proxy.cli import ops_cmds
from typer.testing import CliRunner


@pytest.fixture()
def admin_calls(monkeypatch) -> list[dict]:  # noqa: ANN001
    calls: list[dict] = []

    def fake(method, path, body=None, params=None, timeout=15.0):  # noqa: ANN001
        calls.append({"method": method, "path": path, "params": params or {}})
        if path == "/admin/projects":
            return 200, {"projects": [
                {"project_id": "p-1", "name": "BladeX", "root_path": "/r/BladeX", "agent_ids": ["codex"]},
                {"project_id": "p-2", "name": "", "root_path": "", "agent_ids": []}],
                "total": 12, "limit": 100, "offset": 0}
        return 200, {"window": {"since": None, "until": None, "tz": "+00:00"}, "filters": {},
                     "groups": [], "totals": {}}

    monkeypatch.setattr(ops_cmds, "_admin_call", fake)
    return calls


def _run(*argv: str):
    return CliRunner().invoke(cli.app, list(argv))


# ── bladex usage ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("extra", [("--since", "2026-10-01"), ("--until", "2026-10-02"),
                                   ("--since", "2026-10-01", "--until", "2026-10-02")])
def test_usage_days_and_absolute_window_are_mutually_exclusive(admin_calls, extra) -> None:
    code = cli.main(["usage", "--days", "7", *extra])
    assert code == 2
    assert admin_calls == []        # 不发请求：服务端也不会替它选一个


def test_usage_mutual_exclusion_message_names_both(admin_calls, capsys) -> None:
    assert cli.main(["usage", "--days", "7", "--since", "2026-10-01"]) == 2
    err = capsys.readouterr().err
    assert "--days" in err and "--since" in err


def test_usage_default_window_is_still_seven_days(admin_calls) -> None:
    assert _run("usage", "--json").exit_code == 0
    assert admin_calls[0]["params"]["days"] == 7.0
    assert not admin_calls[0]["params"]["since"] and not admin_calls[0]["params"]["until"]


def test_usage_absolute_window_is_passed_and_does_not_add_days(admin_calls) -> None:
    assert _run("usage", "--since", "2026-10-01T00:00:00", "--until", "2026-10-02T00:00:00",
                "--tz", "Asia/Shanghai").exit_code == 0
    p = admin_calls[0]["params"]
    assert p["since"] == "2026-10-01T00:00:00" and p["until"] == "2026-10-02T00:00:00"
    assert p["tz"] == "Asia/Shanghai" and p["days"] is None


def test_usage_until_alone_has_no_implicit_days(admin_calls) -> None:
    assert _run("usage", "--until", "2026-10-02").exit_code == 0
    assert admin_calls[0]["params"]["days"] is None


def test_usage_tz_alone_keeps_the_default_window(admin_calls) -> None:
    assert _run("usage", "--tz", "+08:00").exit_code == 0
    assert admin_calls[0]["params"]["days"] == 7.0 and admin_calls[0]["params"]["tz"] == "+08:00"


def test_usage_new_params_are_endpoint_params() -> None:
    from bladex_proxy.admin_read import _USAGE_PARAMS

    src = inspect.getsource(ops_cmds.usage_cmd)
    for p in ("since", "until", "tz"):
        assert p in _USAGE_PARAMS and f'"{p}": {p}' in src


def test_usage_all_no_data_is_not_printed_as_zero() -> None:
    lines = ops_cmds._usage_lines({"window": {}, "filters": {}, "groups": [], "totals": {
        "all": {"turns": 2, "input": 0, "input_nodata": 2, "output": 0, "cache_read": 5,
                "cache_read_nodata": 1}}})
    row = next(ln for ln in lines if ln.startswith("  all "))
    assert "n/a (+2 n/d)" in row and "5 (+1 n/d)" in row


# ── bladex project list ──────────────────────────────────────────────────────

def test_project_list_json_prints_the_server_response(admin_calls) -> None:
    res = _run("project", "list", "--json", "--limit", "5", "--offset", "10")
    assert res.exit_code == 0, res.output
    assert admin_calls == [{"method": "GET", "path": "/admin/projects",
                            "params": {"limit": 5, "offset": 10}}]
    assert json.loads(res.stdout)["total"] == 12


def test_project_list_table_shows_no_reading_and_total(admin_calls) -> None:
    res = _run("project", "list")
    assert res.exit_code == 0
    assert "p-1" in res.stdout and "/r/BladeX" in res.stdout and "codex" in res.stdout
    assert "(unnamed)" in res.stdout and "n/a" in res.stdout
    assert "2 shown of 12" in res.stdout


def test_project_list_error_is_nonzero(monkeypatch, capsys) -> None:
    monkeypatch.setattr(ops_cmds, "_admin_call", lambda *a, **k: (503, {"error": {"message": "x"}}))
    assert cli.main(["project", "list"]) == 1
    assert "503" in capsys.readouterr().err


def test_project_list_only_calls_the_endpoint() -> None:
    tree = ast.parse(textwrap.dedent(inspect.getsource(ops_cmds.project_list)))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert "_admin_call" in names and not (names & {"MemoryIndex", "MemoryHub"})
    assert not [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]


# ── config show / config check（MQ-V46）──────────────────────────────────────

@pytest.fixture()
def cfg_dir(tmp_path: Path, monkeypatch, isolated_home) -> Path:  # noqa: ANN001
    monkeypatch.chdir(tmp_path)
    for k in list(os.environ):
        if k.startswith("BLADEX_"):
            monkeypatch.delenv(k)
    assert cli.main(["init", "--yes", "--upstream-model", "openai/test-model", "--api-key", "sk-test"]) == 0
    monkeypatch.setenv("BLADEX_UPSTREAM_API_KEY", "sk-test")
    (tmp_path / "config" / "bladex.toml").unlink(missing_ok=True)
    return tmp_path / "config"


def test_config_show_header_names_all_four_levels(cfg_dir: Path, capsys) -> None:
    assert cli.main(["config", "show"]) == 0
    assert "source (env > bladex.toml > routing.toml > default)" in capsys.readouterr().out


def test_config_check_without_bladex_toml_is_fine(cfg_dir: Path, capsys) -> None:
    assert cli.main(["config", "check"]) == 0
    assert "✓ bladex.toml" in capsys.readouterr().out


def test_config_check_accepts_valid_bladex_toml(cfg_dir: Path, capsys) -> None:
    (cfg_dir / "bladex.toml").write_text('[embedding]\nbackend = "local"\n', encoding="utf-8")
    assert cli.main(["config", "check"]) == 0
    out = capsys.readouterr().out
    assert "✓ bladex.toml" in out and "bladex.toml (" in out


def test_config_check_reports_syntax_error_with_location(cfg_dir: Path, capsys) -> None:
    (cfg_dir / "bladex.toml").write_text("[embedding\nbackend = 1\n", encoding="utf-8")
    assert cli.main(["config", "check"]) == 1
    out = capsys.readouterr().out
    assert "✗ bladex.toml" in out and "syntax error" in out and "line" in out


@pytest.mark.parametrize(("text", "named"), [
    ('[embedding]\nbakend = "local"\n', "[embedding] bakend"),
    ('[distill]\nmodel = "x"\n', "[distill]"),
])
def test_config_check_fails_on_keys_that_would_not_be_read(cfg_dir: Path, capsys, text, named) -> None:
    (cfg_dir / "bladex.toml").write_text(text, encoding="utf-8")
    assert cli.main(["config", "check"]) == 1
    out = capsys.readouterr().out
    assert "✗ bladex.toml" in out and named in out


def test_config_check_reports_invalid_value(cfg_dir: Path, capsys) -> None:
    (cfg_dir / "bladex.toml").write_text('embedding = "not a table"\n', encoding="utf-8")
    assert cli.main(["config", "check"]) == 1
    assert "must be a table" in capsys.readouterr().out


def test_config_check_hints_on_coexisting_legacy_section_without_failing(cfg_dir: Path, capsys) -> None:
    """旧段保留才能回滚 0.3.0：并存合法，只提示。"""
    routing = cfg_dir / "routing.toml"
    routing.write_text(routing.read_text(encoding="utf-8") + '\n[embedding]\nmodel = "m-old"\nbackend = "local"\n',
                       encoding="utf-8")
    (cfg_dir / "bladex.toml").write_text('[embedding]\nbackend = "local"\n', encoding="utf-8")
    assert cli.main(["config", "check"]) == 0, capsys.readouterr().out
    out = capsys.readouterr().out
    assert "⚠ routing.toml [embedding] is deprecated" in out
    assert "backend" in out and "model" in out


def test_config_check_reads_through_the_single_reader(cfg_dir: Path, monkeypatch, capsys) -> None:
    """`config check` 不另写解析：把单一读取点换掉，它的判断跟着变。"""
    from bladex_proxy import global_config
    from bladex_proxy.global_config import GlobalConfigError

    def boom(cfg=None):  # noqa: ANN001, ANN202
        raise GlobalConfigError("sentinel from load_global_embedding")

    monkeypatch.setattr(global_config, "load_global_embedding", boom)
    assert cli.main(["config", "check"]) == 1
    assert "sentinel from load_global_embedding" in capsys.readouterr().out
