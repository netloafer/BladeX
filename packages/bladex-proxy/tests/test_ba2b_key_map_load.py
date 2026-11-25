"""批 BA2b（拍板 (36)-a，MQ-A75）：映射表在处理第一条轮次之前就装好。

缺陷：`data/project_key_map.json` 只在 `load_ledger_events` 里装进进程，而 `_rebuild_from_hub_impl`
的逐轮循环（`note_project_identity`）在锚层装载之前 ⇒ 新起的 consolidator 第一趟拿空表，
把旧键轮次按旧键写进 Project 表。迁移命令在自己的进程里 `set_project_key_map` 过，所以看不到。
"""

from __future__ import annotations

import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from bladex_core.project import (
    canonical_project_id,
    git_remote_project_id,
    legacy_git_remote_project_id,
    set_project_key_map,
)
from bladex_proxy import ledger_events as LE

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "_aa_reliability", _HERE / "test_aa_rebuild_reliability.py")
_aa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_aa)          # 复用批 AA 的桩（Stub / BigramEmbedder / build_index）

URL = "git@github.com:jy/thing.git"
OLD = legacy_git_remote_project_id(URL)
NEW = git_remote_project_id(URL)


def _fresh_process_state() -> None:
    set_project_key_map({})
    LE._key_map_state.update(path=None, mtime=None)


@pytest.fixture(autouse=True)
def _clean():
    _fresh_process_state()
    yield
    _fresh_process_state()


def _make_hub(data: Path, *, with_map: bool = True) -> Path:
    from bladex_proxy.models import Identity, Turn, TurnStatus
    from bladex_proxy.storage.memory_hub import MemoryHub

    data.mkdir(parents=True, exist_ok=True)
    path = data / "bladex_hub"
    hub = MemoryHub(path)
    hub.open()
    ident = Identity(user_id="u1", agent_id="codex", session_id="s1", turn_index=0,
                     project_id=OLD, project_name="thing", project_source="git-remote",
                     project_root="/gone/thing")
    hub.put(ident.storage_key("1799910000-0"), Turn(
        identity=ident, model="m", status=TurnStatus.OK, response_text="收到",
        request_messages=[{"role": "user", "content": "thing 项目的验收清单已经核对完毕，编号 R41 已归档"}],
        ts=datetime(2026, 11, 2, 9, 0, 0, tzinfo=UTC)))
    hub.close()
    if with_map:
        (data / "project_key_map.json").write_text(json.dumps({"map": {OLD: NEW}}), encoding="utf-8")
    return path


def _rebuild(tmp: Path, hub_path: Path, monkeypatch: pytest.MonkeyPatch, *, full: bool) -> Any:
    from bladex_proxy import adjudicator as adj
    from bladex_proxy import router_sdk
    from bladex_proxy.storage.memory_hub import MemoryHub

    stubs = _aa._stubs()
    stub = stubs["Stub"]()
    monkeypatch.setattr(router_sdk, "completion", stub.completion)
    clock = [0.0]
    monkeypatch.setattr(adj, "_monotonic", lambda: clock[0])
    monkeypatch.setattr(adj, "_sleep", lambda s: clock.__setitem__(0, clock[0] + s))
    idx = stubs["build_index"](tmp / "idx", stub)
    router_sdk.completion = stub.completion
    hub = MemoryHub(hub_path)
    hub.open()
    try:
        idx.rebuild_from_hub(hub, full=full, max_turns=30, distill_concurrency=1)
    finally:
        hub.close()
    return idx


@pytest.mark.parametrize("full", [False, True], ids=["incremental", "full"])
def test_ba2b_new_process_first_pass_writes_new_key_only(tmp_path, monkeypatch, full: bool) -> None:
    hub_path = _make_hub(tmp_path / "data")
    assert canonical_project_id(OLD) == OLD      # 前提：这个进程还没装过表
    idx = _rebuild(tmp_path, hub_path, monkeypatch, full=full)
    keys = {p.project_id for p in idx.all_projects()}
    assert OLD not in keys, "旧键写进了 Project 表：映射表装载晚于逐轮循环"
    assert NEW in keys


def test_ba2b_without_map_file_old_key_stays(tmp_path, monkeypatch) -> None:
    """没迁移（或已回滚）⇒ 不翻译：删映射文件就是回滚，不许有隐藏状态。"""
    hub_path = _make_hub(tmp_path / "data", with_map=False)
    idx = _rebuild(tmp_path, hub_path, monkeypatch, full=False)
    assert {p.project_id for p in idx.all_projects()} == {OLD}


def test_ba2b_single_loader_implementation() -> None:
    """装载只有一份实现：只有写映射的迁移与读映射的 `ledger_events` 碰映射文件。"""
    from bladex_proxy import project_migration as PM
    from bladex_proxy.storage import memory_index as MI

    root = Path(LE.__file__).parent
    touching = {p for p in root.rglob("*.py")
                if LE.PROJECT_KEY_MAP_NAME in p.read_text(encoding="utf-8")
                or "PROJECT_KEY_MAP_NAME" in p.read_text(encoding="utf-8")}
    assert touching == {Path(LE.__file__), Path(PM.__file__)}
    assert Path(MI.__file__).read_text(encoding="utf-8").count("refresh_project_key_map(") == 1
