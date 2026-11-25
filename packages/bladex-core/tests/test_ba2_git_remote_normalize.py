"""批 BA2（MQ-A75 甲，拍板 (32)-b）：git remote 归一——一处实现。"""

from __future__ import annotations

import hashlib

import pytest
from bladex_core.project import (
    git_remote_project_id,
    legacy_git_remote_project_id,
    normalize_git_remote,
)


@pytest.mark.parametrize("variant", [
    "https://github.com/jy/BladeX.git",
    "https://github.com/jy/BladeX",
    "https://github.com/jy/BladeX/",
    "https://github.com/jy/BladeX.git/",
    "https://GitHub.COM/jy/BladeX.git",
    "http://github.com/jy/BladeX",
    "ssh://git@github.com/jy/BladeX.git",
    "git://github.com/jy/BladeX.git",
    "git@github.com:jy/BladeX.git",
    "git@GITHUB.com:jy/BladeX",
    "https://user:s3cret@github.com/jy/BladeX.git",
    "  git@github.com:jy/BladeX.git\n",
])
def test_ba2_normalize_same_repo_all_spellings(variant: str) -> None:
    assert normalize_git_remote(variant) == "github.com/jy/BladeX"


def test_ba2_multi_clone_same_repo_same_key() -> None:
    assert (git_remote_project_id("git@github.com:jy/BladeX.git")
            == git_remote_project_id("https://github.com/jy/BladeX/"))


def test_ba2_key_shape_is_the_ruled_formula() -> None:
    expect = "p-" + hashlib.sha256(b"git-remote:github.com/jy/BladeX").hexdigest()[:16]
    assert git_remote_project_id("https://github.com/jy/BladeX.git") == expect


def test_ba2_path_case_and_port_are_kept() -> None:
    """路径大小写不归一（合错仓库比拆开贵）；端口是身份的一部分。"""
    assert normalize_git_remote("https://h.io/a/B") != normalize_git_remote("https://h.io/a/b")
    assert normalize_git_remote("ssh://git@h.io:2222/a/b.git") == "h.io:2222/a/b"
    assert normalize_git_remote("https://h.io/a/b") != normalize_git_remote("ssh://git@h.io:2222/a/b")


def test_ba2_credentials_do_not_enter_key() -> None:
    assert (git_remote_project_id("https://tok1@h.io/a/b")
            == git_remote_project_id("https://tok2@h.io/a/b"))


@pytest.mark.parametrize(("raw", "want"), [
    ("/srv/git/x.git", "/srv/git/x"),
    ("/srv/git/x/", "/srv/git/x"),
    ("file:///srv/git/x.git", "/srv/git/x"),
    ("../x.git", "../x"),
    ("", ""),
])
def test_ba2_local_path_remotes(raw: str, want: str) -> None:
    assert normalize_git_remote(raw) == want


def test_ba2_legacy_key_is_raw_url_twelve_hex() -> None:
    url = "git@github.com:jy/BladeX.git"
    assert legacy_git_remote_project_id(url) == (
        "p-" + hashlib.sha256(f"git-remote:{url}".encode()).hexdigest()[:12])
