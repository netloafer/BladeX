"""V-F1 验收剧本：识别键分层（含软链/worktree 反例）/ 树路径 / 清单渲染 / MATTERS gate。"""

from __future__ import annotations

import os

from bladex_core.flash_tree import (
    MATTERS_GATED_MARK,
    PROJECT_GLOBAL_ID,
    SOURCE_EXPLICIT,
    SOURCE_GIT,
    SOURCE_GLOBAL,
    SOURCE_PATH,
    AgentRow,
    LedgerRow,
    MatterRow,
    ProjectRow,
    SessionRow,
    agents_roster_path,
    ledger_pool_relpath,
    ledgers_list_path,
    matters_list_path,
    normalize_git_remote,
    project_dir,
    render_agents_roster,
    render_ledgers_list,
    render_matters_list,
    render_projects_roster,
    render_sessions_roster,
    resolve_project_identity,
    sessions_roster_path,
)

# ── 项目识别键分层（红线 1）─────────────────────────────────────────────────

class TestProjectIdentity:
    def test_precedence_explicit_over_git(self):
        pid = resolve_project_identity(explicit_id="bladex",
                                       git_remote="git@github.com:x/y.git")
        assert pid.source == SOURCE_EXPLICIT and pid.project_id == "bladex"

    def test_git_remote_unifies_worktrees(self):
        # worktree：同 remote、不同路径 → 必须同一 project_id
        a = resolve_project_identity(git_remote="git@github.com:User/BladeX.git",
                                     git_root="/home/u/dev/BladeX")
        b = resolve_project_identity(git_remote="https://github.com/user/bladex",
                                     git_root="/home/u/dev/BladeX-wt2")
        assert a.project_id == b.project_id
        assert a.source == SOURCE_GIT

    def test_symlink_paths_unify(self, tmp_path):
        # 软链反例：~/dev/BladeX 与旧软链是同一项目的两条路径
        real = tmp_path / "dev" / "BladeX"
        real.mkdir(parents=True)
        link = tmp_path / "Projects-BladeX"
        os.symlink(real, link)
        a = resolve_project_identity(path=str(real))
        b = resolve_project_identity(path=str(link))
        assert a.project_id == b.project_id
        assert a.source == SOURCE_PATH

    def test_no_signal_is_global(self):
        pid = resolve_project_identity()
        assert pid.project_id == PROJECT_GLOBAL_ID
        assert pid.source == SOURCE_GLOBAL

    def test_different_paths_are_different_projects(self, tmp_path):
        a = tmp_path / "proj-a"
        b = tmp_path / "proj-b"
        a.mkdir()
        b.mkdir()
        assert (resolve_project_identity(path=str(a)).project_id
                != resolve_project_identity(path=str(b)).project_id)


class TestGitRemoteNormalization:
    def test_ssh_https_scp_forms_unify(self):
        forms = [
            "git@github.com:User/Repo.git",
            "https://github.com/user/repo",
            "https://github.com/User/Repo/",
            "ssh://git@github.com/User/Repo.git",
            "git://github.com/user/repo.git",
        ]
        keys = {normalize_git_remote(f) for f in forms}
        assert keys == {"github.com/user/repo"}

    def test_garbage_returns_empty(self):
        assert normalize_git_remote("") == ""
        assert normalize_git_remote("not a url") == ""


# ── 树路径 ──────────────────────────────────────────────────────────────────

class TestTreePaths:
    def test_four_levels(self):
        p = ledgers_list_path("/r", "u1", "claude-code", "p-abc", "s-1")
        assert p == "/r/u1/personal/claude-code/p-abc/s-1/LEDGERS.md"
        assert matters_list_path("/r", "u1", "claude-code", "p-abc", "s-1").endswith(
            "/s-1/MATTERS.md")
        assert agents_roster_path("/r", "u1") == "/r/u1/personal/AGENTS.md"
        assert sessions_roster_path("/r", "u1", "a", "p").endswith("/a/p/SESSIONS.md")

    def test_unsafe_components_sanitized(self):
        p = project_dir("/r", "u1", "../evil", "also/../evil")
        assert "/../" not in p

    def test_ledger_pool_relpath_resolves_to_pool(self):
        # session 目录 + 相对引用 = 池内真身
        sess = "/r/u1/personal/a/p/s"
        resolved = os.path.normpath(os.path.join(sess, ledger_pool_relpath("ldg-x")))
        assert resolved == "/r/u1/personal/ledgers/x0/00/ldg-x.md"


# ── 清单渲染 ────────────────────────────────────────────────────────────────

class TestRosters:
    def test_agents_roster_renders_rows(self):
        md = render_agents_roster([AgentRow(agent_id="claude-code", name="Claude Code",
                                            summary="coding agent",
                                            first_seen="2026-07-01", last_seen="2026-08-25")])
        assert "`claude-code`" in md and "coding agent" in md
        assert md == render_agents_roster([AgentRow(
            agent_id="claude-code", name="Claude Code", summary="coding agent",
            first_seen="2026-07-01", last_seen="2026-08-25")])  # 确定性

    def test_projects_roster_shows_identity_source(self):
        md = render_projects_roster("codex", [ProjectRow(
            project_id="p-abc", name="BladeX", source="git")])
        assert "`p-abc`" in md and "BladeX" in md

    def test_sessions_roster_empty(self):
        assert "_(no sessions yet)_" in render_sessions_roster("BladeX", [])
        assert "s-1" in render_sessions_roster("BladeX", [SessionRow(session_id="s-1")])

    def test_pipe_in_cells_is_escaped(self):
        md = render_agents_roster([AgentRow(agent_id="a", name="x|y")])
        assert "x\\|y" in md

    def test_ledgers_list_marks_active_and_links_pool(self):
        md = render_ledgers_list(
            [LedgerRow(ledger_id="ldg-a", title="修路由"),
             LedgerRow(ledger_id="ldg-b", title="写文档")],
            active_ledger_id="ldg-a")
        assert "**(active)**" in md
        assert "(../../../ledgers/a0/00/ldg-a.md)" in md
        assert md.index("ldg-a") < md.index("ldg-b")


# ── MATTERS gate（红线 2）───────────────────────────────────────────────────

class TestMattersGate:
    def test_default_is_gated_even_with_rows(self):
        md = render_matters_list([MatterRow(matter_id="m-1", title="不该出现")])
        assert MATTERS_GATED_MARK in md
        assert "m-1" not in md

    def test_live_renders_rows(self):
        md = render_matters_list([MatterRow(matter_id="m-1", title="泰安项目")],
                                 binding_live=True)
        assert "`m-1`" in md and "泰安项目" in md
