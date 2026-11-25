"""批 BA2（MQ-A75 甲 + MQ-I36，拍板 (32)-b/c）：Index Project 键 = 账本 scope 键；假 Project 不再产生；
一次性迁移幂等、只动派生层。"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from bladex_core.fact import Fact, ItemKind
from bladex_core.ledger import EVENT_LEDGER_SWITCH, replay_ledger_events, switch_timeline
from bladex_core.ledger_runtime import activation_scope, turn_activation_scope
from bladex_core.project import (
    canonical_project_id,
    git_remote_project_id,
    legacy_git_remote_project_id,
    set_project_key_map,
)
from bladex_proxy import ledger_events as LE
from bladex_proxy import project_identity as pi
from bladex_proxy.project_identity import resolve_project
from bladex_proxy.project_migration import run_migration
from bladex_proxy.storage.memory_index import MemoryIndex

URL = "git@GitHub.com:jy/Thing.git"


class _Embedder:
    available = True

    def embed(self, texts):  # noqa: ANN001, ANN201
        return [[1.0] + [0.0] * 31 for _ in texts]


@pytest.fixture(autouse=True)
def _clean():
    pi._cache.clear()
    set_project_key_map({})
    LE._key_map_state.update(path=None, mtime=None)
    yield
    pi._cache.clear()
    set_project_key_map({})
    LE._key_map_state.update(path=None, mtime=None)


@pytest.fixture
def index(tmp_path):
    ix = MemoryIndex(tmp_path / "idx", embedder=_Embedder(), read_only=False)
    ix.open()
    yield ix
    ix.close()


def _git_repo(tmp_path, name="thing", url=URL):
    root = tmp_path / name
    (root / ".git").mkdir(parents=True)
    if url:
        (root / ".git" / "config").write_text(f'[remote "origin"]\n\turl = {url}\n')
    return root


def _rulefile(fid: str, path: str) -> Fact:
    return Fact(id=fid, content=f"文件 {path}", item_kind=ItemKind.FILE_REF, subject=path,
                attribute="file", source_user_id="u1")


def _turn(pid: str, root: str, *, source: str = "git-remote", agent: str = "codex", name: str = "thing"):
    return SimpleNamespace(identity=SimpleNamespace(
        project_id=pid, project_root=root, project_source=source, project_name=name, agent_id=agent))


# ── I36：取根 ────────────────────────────────────────────────────────────

def test_ba2_i36_bare_filename_makes_no_project(index) -> None:
    index.add_fact(_rulefile("f1", "AGENTS.md"))
    index._maintain_projects_recompute()
    assert index.all_projects() == []


def test_ba2_i36_relative_path_makes_no_project(index) -> None:
    index.add_fact(_rulefile("f1", "pkg/CLAUDE.md"))
    index._maintain_projects_recompute()
    assert index.all_projects() == []


def test_ba2_i36_absolute_rulefile_still_recognised(index, tmp_path) -> None:
    root = _git_repo(tmp_path)
    index.add_fact(_rulefile("f1", f"{root}/CLAUDE.md"))
    assert index._maintain_projects_recompute() == 1
    assert [p.root_path for p in index.all_projects()] == [str(root.resolve())]


# ── 甲：键一致 ───────────────────────────────────────────────────────────

def test_ba2_index_project_key_equals_ledger_scope_key(index, tmp_path) -> None:
    """同一个仓库：规则文件识别出的 Project 键 == 请求侧 identity.project_id。"""
    root = _git_repo(tmp_path)
    request_side = resolve_project(str(root)).project_id
    index.add_fact(_rulefile("f1", f"{root}/CLAUDE.md"))
    index._maintain_projects_recompute()
    assert [p.project_id for p in index.all_projects()] == [request_side]
    assert request_side == git_remote_project_id(URL)


def test_ba2_git_root_and_gone_root_share_the_formula(index, tmp_path) -> None:
    """目录已不在时键与 git-root 档同公式——老 bench 目录的存量键仍对得上。"""
    root = _git_repo(tmp_path, url="")
    live_key = resolve_project(str(root)).project_id
    gone = str(root.resolve())
    pi._cache.clear()
    import shutil
    shutil.rmtree(root)
    assert index._project_key_for_root(gone)[0] == live_key


def test_ba2_note_project_identity_uses_turn_key_verbatim(index) -> None:
    assert index.note_project_identity("p-abc", name="n", root="/r", agent_id="dsh") is True
    assert [p.project_id for p in index.all_projects()] == ["p-abc"]
    v = index.get_project("p-abc").version
    assert index.note_project_identity("p-abc", name="n", root="/r", agent_id="dsh") is False
    assert index.get_project("p-abc").version == v


def test_ba2_note_project_identity_fills_root_seen_later(index) -> None:
    """生产 p-10ad 有 4 轮身份里没记根；先见到它不得让 Project 永远 root_path 为空。"""
    index.note_project_identity("p-x", name="", root="", agent_id="dsh")
    assert index.get_project("p-x").root_path == ""
    assert index.note_project_identity("p-x", name="x", root="/r/x", agent_id="dsh") is True
    p = index.get_project("p-x")
    assert p.root_path == "/r/x" and p.name == "x"


# ── 迁移 ─────────────────────────────────────────────────────────────────

def _project_rows(ix) -> list:  # noqa: ANN001
    return sorted((p.project_id, p.root_path, tuple(p.agent_ids)) for p in ix.all_projects())


def test_ba2_migration_maps_confirmed_key_deletes_fake_and_is_idempotent(index, tmp_path) -> None:
    from bladex_core.project import Project

    root = _git_repo(tmp_path)
    old = legacy_git_remote_project_id(URL)
    new = git_remote_project_id(URL)
    index.upsert_project(Project(project_id="a54ff182c7e8", root_path="AGENTS.md", name="AGENTS.md"))
    turns = [_turn(old, str(root)), _turn("p-groot", "/gone/x", source="git-root", agent="dsh")]
    data = tmp_path / "data"
    data.mkdir()

    rep = run_migration(turns, index, data)
    assert rep.mapped == {old: new}
    assert rep.fake_projects_deleted == ["a54ff182c7e8"]
    assert {p.project_id for p in index.all_projects()} == {new, "p-groot"}
    assert json.loads((data / "project_key_map.json").read_text())["map"] == {old: new}
    assert canonical_project_id(old) == new

    snapshot = _project_rows(index)
    versions = {p.project_id: p.version for p in index.all_projects()}
    rep2 = run_migration(turns, index, data)
    assert rep2.map_changed is False and rep2.fake_projects_deleted == []
    assert _project_rows(index) == snapshot
    assert {p.project_id: p.version for p in index.all_projects()} == versions


def test_ba2_migration_unconfirmed_key_is_reported_not_guessed(index, tmp_path) -> None:
    """目录不在 / remote 改过 ⇒ 正向算不出旧键 ⇒ 不进表（原则 13 层 4）。"""
    changed = _git_repo(tmp_path, "changed", url="https://h.io/now/other")
    turns = [_turn("p-gone00000000", "/no/such/dir"), _turn("p-changed00000", str(changed))]
    data = tmp_path / "data"
    data.mkdir()
    rep = run_migration(turns, index, data)
    assert rep.mapped == {}
    assert {u["project_id"]: u["reason"].split(":")[0] for u in rep.unmapped} == {
        "p-gone00000000": "root_gone", "p-changed00000": "remote_changed"}


def test_ba2_migration_keeps_mapping_after_root_disappears(index, tmp_path) -> None:
    root = _git_repo(tmp_path)
    old = legacy_git_remote_project_id(URL)
    data = tmp_path / "data"
    data.mkdir()
    run_migration([_turn(old, str(root))], index, data)
    import shutil
    shutil.rmtree(root)
    rep = run_migration([_turn(old, str(root))], index, data)
    assert rep.mapped == {old: git_remote_project_id(URL)} and rep.unmapped == []


def test_ba2_migration_dry_run_writes_nothing(index, tmp_path) -> None:
    from bladex_core.project import Project

    root = _git_repo(tmp_path)
    index.upsert_project(Project(project_id="fake", root_path="AGENTS.md"))
    data = tmp_path / "data"
    data.mkdir()
    rep = run_migration([_turn(legacy_git_remote_project_id(URL), str(root))], index, data, dry_run=True)
    assert rep.fake_projects_deleted == ["fake"] and rep.mapped
    assert not (data / "project_key_map.json").exists()
    assert [p.project_id for p in index.all_projects()] == ["fake"]


# ── 激活位：读事件时翻译，Hub 原值不动 ─────────────────────────────────

class _Hub:
    def __init__(self, path, events) -> None:  # noqa: ANN001
        self._path = path
        self._events = events

    def scan_admin_events(self):  # noqa: ANN201
        return [(str(i), e) for i, e in enumerate(self._events)]


def _switch_event(project: str, to: str, ts: float = 1_800_000_000.0):
    return SimpleNamespace(event_type=EVENT_LEDGER_SWITCH, ts=ts, payload={
        "agent_id": "codex", "project_id": project, "session_id": "s1",
        "from_ledger_id": "", "to_ledger_id": to, "turn_index": 1})


def test_ba2_activation_follows_map_and_hub_payload_untouched(tmp_path) -> None:
    old, new = legacy_git_remote_project_id(URL), git_remote_project_id(URL)
    (tmp_path / "project_key_map.json").write_text(json.dumps({"map": {old: new}}))
    ev = _switch_event(old, "ldg-1")
    hub = _Hub(tmp_path / "bladex_hub", [ev])
    events = LE.load_ledger_events(hub)
    _pool, bindings = replay_ledger_events(events)
    assert bindings == {activation_scope("codex", new): "ldg-1"}
    assert ev.payload["project_id"] == old            # Hub 读出来的对象原样
    tl = switch_timeline(events)
    # 老轮次存的是旧键，查询时翻译 ⇒ 落到同一条时间轴
    assert turn_activation_scope(tl, "local/codex/s1/1800000001000-0", old) == activation_scope("codex", new)


def test_ba2_rollback_is_deleting_the_map(tmp_path) -> None:
    old, new = legacy_git_remote_project_id(URL), git_remote_project_id(URL)
    f = tmp_path / "project_key_map.json"
    f.write_text(json.dumps({"map": {old: new}}))
    hub = _Hub(tmp_path / "bladex_hub", [_switch_event(old, "ldg-1")])
    LE.load_ledger_events(hub)
    assert canonical_project_id(old) == new
    f.unlink()
    _pool, bindings = replay_ledger_events(LE.load_ledger_events(hub))
    assert canonical_project_id(old) == old
    assert bindings == {activation_scope("codex", old): "ldg-1"}
