"""Beta T11 CLI 记忆/Matter/存储运维命令验收：命令 → admin HTTP 调用映射。

数据面语义（墓碑/级联/重建）由 test_admin_fact_delete.py 钉住；本文件验证
CLI 层：路径/方法/请求体正确、deferred(202) 语义呈现、脚本收编防呆。
_admin_call monkeypatch 成录制桩——CLI 不直捅存储的纪律由此结构性保证。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from bladex_proxy import cli


@pytest.fixture(autouse=True)
def _isolate_env(tmp_path: Path, monkeypatch, isolated_home):
    """隔离：cli.main 会把 cwd 下 config/.env 灌进 os.environ（且不回收）——
    在仓库根跑会吸入真实配置（BLADEX_AUTH_ENABLED 等）污染同进程后续测试。
    统一切到无 config/.env 的 tmp cwd + 清 BLADEX_* env。
    `isolated_home` 顺带堵上 2026-08-06 新增的 `~/.bladex` 兜底（见 conftest）。"""
    monkeypatch.chdir(tmp_path)
    for k in list(os.environ):
        if k.startswith("BLADEX_"):
            monkeypatch.delenv(k, raising=False)
    yield
    # _load_env_file 在测试中新设的 BLADEX_*（monkeypatch 未跟踪）也要清
    for k in list(os.environ):
        if k.startswith("BLADEX_"):
            del os.environ[k]


class _Recorder:
    def __init__(self, responses: dict | None = None):
        self.calls: list[tuple[str, str, dict | None, dict | None]] = []
        self._responses = responses or {}

    def __call__(self, method: str, path: str, body: dict | None = None,
                 params: dict | None = None):
        self.calls.append((method, path, body, params))
        return self._responses.get((method, path), (200, {"status": "ok"}))


@pytest.fixture()
def rec(monkeypatch):
    r = _Recorder()
    from bladex_proxy.cli import (
        memory_cmds as _mem,  # F0.1 拆包：消费方在 cli/memory_cmds.py（memory/matter/storage）
    )
    monkeypatch.setattr(_mem, "_admin_call", r)
    return r


# ── memory ────────────────────────────────────────────────────────


def test_memory_search_maps_to_facts_query(rec, monkeypatch):
    rec._responses[("GET", "/admin/facts")] = (200, {
        "facts": [{"id": "f1", "kind": "event", "content": "proxy 502 已修复"}],
        "total": 1,
    })
    rc = cli.main(["memory", "search", "proxy 502", "--k", "5", "--user", "u1"])
    assert rc == 0
    method, path, body, params = rec.calls[0]
    assert (method, path) == ("GET", "/admin/facts")
    assert params == {"q": "proxy 502", "limit": 5, "user_id": "u1"}


def test_memory_show_and_forget(rec):
    rec._responses[("GET", "/admin/facts/f1")] = (200, {"id": "f1", "content": "x"})
    assert cli.main(["memory", "show", "f1"]) == 0
    assert rec.calls[0][:2] == ("GET", "/admin/facts/f1")

    rec._responses[("DELETE", "/admin/facts/f1")] = (
        200, {"status": "deleted", "fact_id": "f1", "tombstone_key": "tomb/1"})
    assert cli.main(["memory", "forget", "f1", "--yes"]) == 0
    assert rec.calls[1][:2] == ("DELETE", "/admin/facts/f1")


def test_memory_forget_deferred_202(rec, capsys):
    rec._responses[("DELETE", "/admin/facts/f2")] = (202, {"status": "deferred"})
    assert cli.main(["memory", "forget", "f2", "--yes"]) == 0
    assert "deferred" in capsys.readouterr().out


# ── matter ────────────────────────────────────────────────────────


def test_matter_commands_map_to_endpoints(rec):
    rec._responses[("GET", "/admin/matters")] = (200, {"matters": [], "total": 0})
    rec._responses[("GET", "/admin/matters/m1")] = (200, {
        "matter": {"matter_id": "m1", "title": "t", "status": "active"},
        "edges": [], "facts": [], "session_keys": [],
        "edge_count": 0, "manual_edge_count": 0,
    })
    assert cli.main(["matter", "list", "--status", "active"]) == 0
    assert cli.main(["matter", "show", "m1"]) == 0
    assert cli.main(["matter", "create", "新事项", "--summary", "s"]) == 0
    assert cli.main(["matter", "assign", "m1", "f1", "--type", "fact"]) == 0
    assert cli.main(["matter", "merge", "m1", "m2"]) == 0
    assert cli.main(["matter", "detach", "m1", "--target-key", "f1"]) == 0
    assert cli.main(["matter", "split", "m1", "--new-title", "拆出",
                     "--edge-id", "e1", "--edge-id", "e2"]) == 0
    assert cli.main(["matter", "close", "m1"]) == 0
    assert cli.main(["matter", "rename", "m1", "新名"]) == 0

    by_path = {(m, p): (b, q) for m, p, b, q in rec.calls}
    assert ("GET", "/admin/matters") in by_path
    assert by_path[("POST", "/admin/matters")][0] == {"title": "新事项", "summary": "s"}
    assert by_path[("POST", "/admin/matters/m1/assign")][0] == {
        "target_type": "fact", "target_key": "f1"}
    assert by_path[("POST", "/admin/matters/m1/merge")][0] == {"source_matter_id": "m2"}
    assert by_path[("POST", "/admin/matters/m1/detach")][0] == {
        "target_key": "f1", "target_type": "fact"}
    assert by_path[("POST", "/admin/matters/m1/split")][0] == {
        "new_title": "拆出", "edge_ids": ["e1", "e2"]}
    assert by_path[("POST", "/admin/matters/m1/close")][0] == {}
    assert by_path[("POST", "/admin/matters/m1/rename")][0] == {"title": "新名"}


# ── storage / backup / restore ────────────────────────────────────


def test_storage_rebuild_delegates_to_sync_full(monkeypatch):
    captured = {}

    def _fake_sync(ns):
        captured.update(vars(ns))
        return 0

    from bladex_proxy.cli import ops_cmds as _ops  # F0.1 拆包：消费方在 cli/ops_cmds.py（sync_run）
    monkeypatch.setattr(_ops, "cmd_sync_run", _fake_sync)
    rc = cli.main(["storage", "rebuild", "--agents", "hermes:default",
                   "--concurrency", "2"])
    assert rc == 0
    assert captured["full"] is True
    assert captured["agents"] == "hermes:default"
    assert captured["concurrency"] == 2


def test_backup_requires_repo_script(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)  # 无 scripts/ 的目录
    rc = cli.main(["backup"])
    assert rc == 1
    assert "scripts/backup.sh" in capsys.readouterr().out


def test_restore_refuses_while_running(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    Path("logs").mkdir()
    Path(cli._PROXY_PIDFILE).write_text(str(os.getpid()))  # 活 pid
    rc = cli.main(["restore", "some_backup_dir"])
    assert rc == 1
    assert "bladex stop" in capsys.readouterr().out
