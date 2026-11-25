"""批 A95（MQ-A95）：cwd 候选规范化 + 按序尝试 + 不可解析告警。

病例：dsh 0.2.1 的 system 里两行同现——
    Current working directory: "/…/dsh".
    … may modify files under the session workspace: "/…/dsh".
通用 `working directory:` 规则先抓到带引号与句点的整段 ⇒ 路径不存在 ⇒ 静默落 Global。
目录一律用 `tmp_path` 造，不依赖 live 路径。
"""

from __future__ import annotations

import pytest
from bladex_proxy import project_identity as pi
from bladex_proxy.project_identity import (
    extract_cwd,
    extract_cwd_after_developer,
    resolve_from_request,
    resolve_project,
)
from structlog.testing import capture_logs


@pytest.fixture(autouse=True)
def _fresh_state():
    pi._unresolved_seen.clear()
    pi._cache.clear()
    yield
    pi._unresolved_seen.clear()
    pi._cache.clear()


def _repo(tmp_path, name: str):
    root = tmp_path / name
    (root / ".git").mkdir(parents=True)
    return root


def _dsh_021_system(path: str) -> str:
    return (
        "You are dsh.\n\n"
        f'Current working directory: "{path}".\n'
        "Current runtime context. The agent may modify files under the session "
        f'workspace: "{path}".\n'
    )


def _unresolved(logs):
    return [e for e in logs if e["event"] == "project_cwd_unresolved"]


class TestA95Normalization:
    def test_a95_dsh_021_both_lines_resolve_to_real_dir(self, tmp_path):
        root = _repo(tmp_path, "dsh")
        msgs = [{"role": "system", "content": _dsh_021_system(str(root))}]
        assert extract_cwd(msgs) == str(root)
        ident = resolve_from_request({}, msgs)
        assert ident.project_id and ident.root == str(root.resolve())

    def test_a95_quoted_period_line_alone_normalizes(self, tmp_path):
        root = _repo(tmp_path, "only")
        msgs = [{"role": "system",
                 "content": f'Current working directory: "{root}".\nPlatform: darwin'}]
        assert extract_cwd(msgs) == str(root)
        assert resolve_from_request({}, msgs).project_id

    @pytest.mark.parametrize("wrap", ['"{}"', "'{}'", "`{}`", '"{}",', "{};", "'{}':"])
    def test_a95_each_quote_and_trailing_punct(self, tmp_path, wrap):
        root = _repo(tmp_path, "w")
        line = "Working directory: " + wrap.format(root)
        assert extract_cwd([{"role": "system", "content": line}]) == str(root)

    def test_a95_path_with_spaces_kept_whole(self, tmp_path):
        root = _repo(tmp_path, "My Projects/Blade X")
        msgs = [{"role": "system",
                 "content": f'Current working directory: "{root}".'}]
        assert extract_cwd(msgs) == str(root)
        assert resolve_project(extract_cwd(msgs)).project_id


class TestA95OrderedFallback:
    def test_a95_first_pattern_missing_second_resolves(self, tmp_path):
        """第 2 条（working directory）抓到不存在的路径，第 4 条（session workspace）可用 ⇒ 用第 4 条。"""
        real = _repo(tmp_path, "real")
        msgs = [{"role": "system", "content": (
            f"Current working directory: {tmp_path / 'gone'}\n"
            f'under the session workspace: "{real}".\n')}]
        assert extract_cwd(msgs) == str(real)

    def test_a95_fallback_across_messages(self, tmp_path):
        real = _repo(tmp_path, "real")
        msgs = [{"role": "system", "content": f"Working directory: {tmp_path / 'gone'}"},
                {"role": "developer", "content": f"Working directory: {real}"}]
        assert extract_cwd(msgs) == str(real)

    def test_a95_after_developer_uses_same_ordering(self, tmp_path):
        real = _repo(tmp_path, "real")
        msgs = [{"role": "developer", "content": "rules"},
                {"role": "user", "content": (
                    f"<environment_context><cwd>{tmp_path / 'gone'}</cwd></environment_context>\n"
                    f"# AGENTS.md instructions for {real}\n")}]
        assert extract_cwd_after_developer(msgs) == str(real)


class TestA95Unresolved:
    def test_a95_nothing_resolvable_global_and_warn_once(self, tmp_path):
        gone = tmp_path / "gone"
        msgs = [{"role": "system", "content": _dsh_021_system(str(gone))}]
        with capture_logs() as logs:
            ident = resolve_from_request({}, msgs)
            resolve_from_request({}, msgs)          # 同一原串第二次不再告警
        assert ident.project_id == ""
        warns = _unresolved(logs)
        assert len(warns) == 1
        assert warns[0]["raw"] == f'"{gone}".'[:120]
        assert warns[0]["pattern"] == 1             # 第 2 条（通用 working directory）先命中
        resolved = [e for e in logs if e["event"] == "project_resolved"]
        assert resolved[0]["cwd_found"] is True and resolved[0]["source"] == "global"

    def test_a95_no_candidate_no_warning(self):
        with capture_logs() as logs:
            assert extract_cwd([{"role": "system", "content": "hello"}]) == ""
        assert _unresolved(logs) == []

    def test_a95_hermes_home_dir_still_global(self, tmp_path):
        """hermes `Current working directory: /Users/you`（家目录、非仓库）：目录存在 ⇒ 被采用、判 Global，不告警。"""
        home = tmp_path / "you"
        home.mkdir()
        msgs = [{"role": "system",
                 "content": f"User home directory: {home}\nCurrent working directory: {home}\n"}]
        with capture_logs() as logs:
            assert extract_cwd(msgs) == str(home)
            assert resolve_from_request({}, msgs).project_id == ""
        assert _unresolved(logs) == []


class TestA95bDeclarationOnlyFallback:
    """批 A95b（MQ-A96）：agent 自己的环境声明不可用时，只能由另一条声明接手，
    user 自由文本（非 system/developer、块外、头部）不能接手——否则落到用户随口贴的别的项目上。"""

    def _assert_not_taken_over(self, msgs, other):
        with capture_logs() as logs:
            assert extract_cwd(msgs) != str(other)
            assert resolve_from_request({}, msgs).project_id == ""
        assert len(_unresolved(logs)) == 1

    def test_a95b_user_text_working_directory_cannot_take_over(self, tmp_path):
        other = _repo(tmp_path, "other")
        msgs = [{"role": "system", "content": f"Current working directory: {tmp_path / 'agentcwd'}"},
                {"role": "user", "content": f"看下这个\nworking directory: {other}\n"}]
        self._assert_not_taken_over(msgs, other)

    def test_a95b_user_text_cwd_tag_outside_env_block_cannot_take_over(self, tmp_path):
        other = _repo(tmp_path, "other")
        msgs = [{"role": "system", "content": f"Current working directory: {tmp_path / 'agentcwd'}"},
                {"role": "user", "content": f"贴一段日志 <cwd>{other}</cwd> 你看看"}]
        self._assert_not_taken_over(msgs, other)

    def test_a95b_env_block_in_user_role_is_a_declaration(self, tmp_path):
        real = _repo(tmp_path, "real")
        msgs = [{"role": "system", "content": f"Current working directory: {tmp_path / 'agentcwd'}"},
                {"role": "user", "content": (
                    f"<environment_context>\n  <cwd>{real}</cwd>\n</environment_context>")}]
        assert extract_cwd(msgs) == str(real)

    def test_a95b_developer_declaration_can_take_over(self, tmp_path):
        real = _repo(tmp_path, "real")
        msgs = [{"role": "system", "content": f"Current working directory: {tmp_path / 'agentcwd'}"},
                {"role": "developer", "content": f"Working directory: {real}"}]
        assert extract_cwd(msgs) == str(real)

    def test_a95b_user_text_without_prior_candidate_unchanged(self, tmp_path):
        real = _repo(tmp_path, "real")
        msgs = [{"role": "system", "content": "You are an agent."},
                {"role": "user", "content": f"working directory: {real}"}]
        assert extract_cwd(msgs) == str(real)

    def test_a95b_unresolved_dedup_set_is_bounded(self, tmp_path):
        for k in range(300):
            extract_cwd([{"role": "system",
                          "content": f"Working directory: {tmp_path / f'gone-{k}'}"}])
        assert len(pi._unresolved_seen) <= pi._UNRESOLVED_SEEN_MAX
