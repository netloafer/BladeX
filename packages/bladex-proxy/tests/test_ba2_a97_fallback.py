"""批 BA2（MQ-A97）：全部 cwd 候选不可用时，不再挡 codex header / developer 后一条两路声明。

修前：`extract_cwd` 返回不可用的第一个候选 ⇒ cwd 非空 ⇒ 后两路不走 ⇒ 判 Global。
A95b 的规则不变：user 正文里的非声明候选仍然接手不了（本文件最后一例 + `test_a95_cwd_candidates`）。
"""

from __future__ import annotations

import json

import pytest
from bladex_proxy import project_identity as pi
from bladex_proxy.project_identity import resolve_from_request, resolve_project
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


def _ws_header(*roots) -> dict[str, str]:
    return {"x-codex-turn-metadata": json.dumps({"workspaces": {str(r): {} for r in roots}})}


def test_ba2_a97_unusable_candidate_does_not_block_workspace_header(tmp_path) -> None:
    real = _repo(tmp_path, "real")
    msgs = [{"role": "system", "content": f"Current working directory: {tmp_path / 'gone'}"}]
    ident = resolve_from_request(_ws_header(real), msgs)
    assert ident.project_id == resolve_project(str(real)).project_id
    assert ident.project_id


def test_ba2_a97_unusable_candidate_does_not_block_after_developer(tmp_path) -> None:
    real = _repo(tmp_path, "real")
    # 候选在窗口外（前 4 条被 user 占满），只有 developer 紧后那一条带可用 cwd。
    msgs = [{"role": "system", "content": f"Current working directory: {tmp_path / 'gone'}"}]
    msgs += [{"role": "user", "content": f"hello {i}"} for i in range(4)]
    msgs += [{"role": "developer", "content": "dev"},
             {"role": "user", "content": f"<environment_context><cwd>{real}</cwd></environment_context>"}]
    ident = resolve_from_request({}, msgs)
    assert ident.project_id == resolve_project(str(real)).project_id
    assert ident.project_id


def test_ba2_a97_all_unusable_stays_global_and_keeps_first_reading(tmp_path) -> None:
    gone = tmp_path / "gone"
    msgs = [{"role": "system", "content": f"Current working directory: {gone}"}]
    with capture_logs() as logs:
        ident = resolve_from_request(_ws_header(tmp_path / "also-gone"), msgs)
    assert ident.project_id == ""
    row = next(r for r in logs if r["event"] == "project_resolved")
    assert row["cwd_source"] == "messages"
    assert row["cwd"] == str(gone)[:120]


def test_ba2_a97_usable_message_cwd_still_wins_over_header(tmp_path) -> None:
    """改前能认出项目的路径逐字不变（批 E 调度改序）。"""
    near, far = _repo(tmp_path, "near"), _repo(tmp_path, "far")
    msgs = [{"role": "system", "content": f"Current working directory: {near}"}]
    ident = resolve_from_request(_ws_header(far), msgs)
    assert ident.project_id == resolve_project(str(near)).project_id


def test_ba2_a97_user_text_still_cannot_take_over(tmp_path) -> None:
    """A95b：放开的只有两条声明路；user 正文的 `working directory:` 仍接手不了。"""
    other = _repo(tmp_path, "other")
    msgs = [{"role": "system", "content": f"Current working directory: {tmp_path / 'gone'}"},
            {"role": "user", "content": f"看下这个\nworking directory: {other}\n"}]
    assert resolve_from_request({}, msgs).project_id == ""


def test_ba2_resolve_project_two_clones_same_key(tmp_path) -> None:
    """甲方案：同仓两个克隆（https / ssh 写法）解析成同一个项目键。"""
    keys = []
    for name, url in (("a", "https://github.com/jy/BladeX.git"), ("b", "git@GitHub.com:jy/BladeX")):
        root = _repo(tmp_path, name)
        (root / ".git" / "config").write_text(f'[remote "origin"]\n\turl = {url}\n')
        ident = resolve_project(str(root))
        assert ident.source == "git-remote"
        keys.append(ident.project_id)
    assert keys[0] == keys[1]
