"""G12.12 ① 纯函数半边：Flash 的作用域目录（②布局迁移 / ③暴露度过滤随 S5 删除）。

本文件对着 D1/D2 的三件事，每件都有一次具体的失效形态在背后：

1. **作用域目录**（D2）。§6.9 选"按 principal/scope 切目录"的**唯一理由**是
   「文件级的边界比行级的过滤更难写错」（对照 2026-07-28 `_visibility_pids` 写成
   `is not None`、潜伏 6 天那次）。所以这里钉两件事：映射表本身，以及
   **目录名单射**——一个会撞名的目录名把那条理由整个抵消掉，且失效是静默的。
2. **布局迁移**（D2 的连带义务）。布局一变，旧位置的 `*.local.md` 既不会被删
   （prune 按设计不碰它）也不再被读，**静默失效**：用户改过的 Goal 从此不进注入，
   而他看不出任何异常。既有守卫 `test_user_local_file_is_never_touched` 覆盖的是
   "同布局下不被删"，覆盖不到"换布局后读不到"——不同的失效面要有不同的守卫。
3. **暴露度过滤**（D1 主体）。血统自 G12.5 起就继承在 `SlotItem.exposure` 上，
   但**从来没有人按它过滤过**——继承而不消费，等于登记了一个不生效的边界。

纯函数 + tmp_path 文件操作，不碰 Memory Index、不碰 live 存储。
"""

from __future__ import annotations

import pytest

from bladex_core import flash


# ── 1. 作用域映射 ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("scope,expect", [
    ("", flash.SCOPE_PERSONAL),                 # 未标注：存量 100% 是它
    ("   ", flash.SCOPE_PERSONAL),
    ("personal:u1", flash.SCOPE_PERSONAL),      # 标了、且就是本人
    ("team:legal", "team-legal"),
    ("org:acme", "org-acme"),
    ("team:", "team"),                          # 退化形态，仍是确定的目录
    ("org:", "org"),
])
def test_scope_dir_mapping(scope, expect):
    assert flash.scope_dir_of(scope, principal="u1") == expect


@pytest.mark.parametrize("scope", ["personal:someone-else", "weird:x", "personal", "x"])
def test_unplaceable_scope_is_refused_not_guessed(scope):
    """🔴 说不准就**拒绝落盘**，不猜一个目录。

    猜一个目录 = 用一次猜测定下一条可见性边界。`personal:<别人>` 出现在这个
    principal 的树下本身就是异常数据，把它写进 `personal/` 就是把别人的东西
    标成自己的；未知前缀同理。fail-closed 的含义在文件层就是"不写"。
    """
    assert flash.scope_dir_of(scope, principal="u1") == flash.SCOPE_DIR_REFUSED


def test_empty_scope_is_stricter_at_file_layer_than_at_row_layer():
    """具名例外：空 scope 在行层 fail-open、在文件层归 `personal`。

    行层那条 fail-open 是**迁移兼容**（`_matter_visible`：存量 120/120 都是空，
    过滤掉等于把②平面清零）。文件层不需要那个兼容——它本来就按 principal 分根，
    未标注的数据只可能是这个 principal 自己的。**不一致的方向是"文件层更严"**，
    即安全侧；写成用例是为了让它是**具名的**，而不是悄悄存在的。
    """
    assert flash.scope_dir_of("", principal="u1") == flash.SCOPE_PERSONAL
    # 而且它落在"自己"的目录里，别的 principal 的可见集合选不到这里
    assert flash.SCOPE_PERSONAL not in flash.readable_scope_dirs(
        ["other"], ["personal:other"], principal="u1")


# ── 2. 读侧选目录（与行层三态语义对账）────────────────────────────────────


def test_readable_dirs_personal_mode():
    """个人模式：`visibility=[]` → pids 回落 `[user_id]` → 只读 `personal/`。"""
    assert flash.readable_scope_dirs(["u1"], [], principal="u1") == [flash.SCOPE_PERSONAL]


def test_readable_dirs_team_only_excludes_personal():
    """显式只可见 team（无 personal）⇒ 读不到 `personal/`，与行层 fail-closed 同向。"""
    dirs = flash.readable_scope_dirs([], ["team:legal"], principal="u1")
    assert dirs == ["team-legal"]
    assert flash.SCOPE_PERSONAL not in dirs


def test_readable_dirs_enterprise_mixed():
    dirs = flash.readable_scope_dirs(
        ["u1"], ["personal:u1", "team:legal", "org:acme"], principal="u1")
    assert dirs == [flash.SCOPE_PERSONAL, "team-legal", "org-acme"]


def test_readable_dirs_legacy_none_is_not_a_wildcard():
    """`pids is None` = legacy 不过滤。文件层给 `personal` 一个目录，**不当通配符**。

    那条路径连 principal 都不确定，给不出更多目录也不该猜——同样是更严的一侧。
    """
    assert flash.readable_scope_dirs(None, None, principal="u1") == [flash.SCOPE_PERSONAL]
    assert flash.readable_scope_dirs(None, None, principal="") == []


# （第 3 节「布局迁移」与第 4 节「暴露度过滤」随 S5 2026-09-03 删除：
#  `migrate_legacy_layout` / `prune_empty_dirs` / `filter_matter_by_exposure` /
#  `MatterFlash` 属 ADR-0031 老渲染器，live 树 `*.local.md` = 0、无保护对象。）
