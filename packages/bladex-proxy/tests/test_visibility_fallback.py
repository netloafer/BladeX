"""可见集合三种"空"的语义边界（2026-07-28 真实流量事故修复）。

事故：13/13 轮 `index_search_empty_visibility`，`index_search_done` 一次未出现——
个人模式下 Fact 散点召回 **100% 被 fail-closed 阻断**，注入全靠 Matter 两级召回兜着。
自 ADR-0021 T1（2026-07-22）起潜伏 6 天。

根因：`_visibility_pids` 写的是 `if visibility is not None`，而个人模式下
`Identity.visibility` 默认是**空列表**（`Field(default_factory=list)`，非 None），
于是被当成"显式声明可见集合为空"→ fail-closed。ADR-0021 §2.4 的原意是
"空 = 个人模式现状回落，走 user_id 兼容路径"。

本文件把三种"空"钉死，防止再把"未标注"与"显式空"混为一谈。
"""

from __future__ import annotations

import pytest

from bladex_proxy.storage.memory_index import MemoryIndex


@pytest.fixture
def index(tmp_path):
    """只用 _visibility_pids（纯函数式判定），不需要真的开库。"""
    return MemoryIndex(str(tmp_path / "index"))


# ── 三行语义表 ──


def test_empty_list_falls_back_to_user_id(index):
    """🔴 事故本体：个人模式 visibility=[] 必须回落 user_id，不得 fail-closed。"""
    assert index._visibility_pids([], "24345848") == ["24345848"]


def test_none_falls_back_to_user_id(index):
    """visibility=None（未传）同样回落。"""
    assert index._visibility_pids(None, "24345848") == ["24345848"]


def test_explicit_non_personal_scope_stays_fail_closed(index):
    """显式只可见 team/org 且无 personal → 空 list → fail-closed（ADR-0021 语义保持）。"""
    assert index._visibility_pids(["team:legal"], "24345848") == []
    assert index._visibility_pids(["org:acme", "team:legal"], "24345848") == []


def test_personal_scope_extracted(index):
    assert index._visibility_pids(["personal:abc"], None) == ["abc"]


def test_mixed_scope_keeps_only_personal(index):
    assert index._visibility_pids(
        ["team:legal", "personal:abc", "org:acme", "personal:def"], None
    ) == ["abc", "def"]


def test_no_visibility_no_user_id_means_no_filter(index):
    """两者皆空 → None = 不过滤（legacy 全量，向后兼容）。"""
    assert index._visibility_pids(None, None) is None
    assert index._visibility_pids([], None) is None


# ── 回归防线：区分"未标注"与"显式空" ──


def test_unannotated_and_explicit_empty_are_not_conflated(index):
    """同一个 user_id 下，未标注应召回、显式 team-only 应阻断——两者不得同解。"""
    unannotated = index._visibility_pids([], "u1")
    explicit_team_only = index._visibility_pids(["team:x"], "u1")
    assert unannotated == ["u1"], "未标注被误判成显式空 = 事故复现"
    assert explicit_team_only == [], "显式 team-only 必须 fail-closed"
    assert unannotated != explicit_team_only


def test_identity_default_visibility_is_empty_list_not_none(index):
    """守住前提：Identity.visibility 默认是空列表——这正是踩坑的原因。

    若哪天默认值改成 None，本测试提醒同步复核 _visibility_pids 的分支。
    """
    from bladex_proxy.models import Identity

    ident = Identity(user_id="u1")
    assert ident.visibility == []
    assert ident.visibility is not None
    # 个人模式的实际调用形状必须能召回
    assert index._visibility_pids(ident.visibility, ident.user_id) == ["u1"]
