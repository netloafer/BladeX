"""G12.12 ①：文件层选目录 与 行层 scope 过滤 的**逐格对账**。

为什么要有这一份，而不是各自测各自：ADR-0021 的可见性语义现在有了**第二个实现**
——行层是 `MemoryIndex._matter_visible`（配 `_visibility_pids` 的三态判定），
文件层是 `flash.scope_dir_of` + `flash.readable_scope_dirs`。
`_matter_visible` 的 docstring 逐字写着这套语义「只允许有一个实现」，
理由是 2026-07-28 那次：同一套语义写了两遍，其中一遍把"未标注"当成"声明为空",
个人模式 Fact 散点召回 fail-closed 锁死 13/13 轮、潜伏 6 天。

现在我们**不得不**有第二个实现（文件路径不能用 SQL 过滤表达），所以退而求其次：
**让两个实现互相对账**。任一边改动导致判定分叉，本文件当场红。

两处**具名例外**（都在"文件层更严"这一侧，即安全侧）：
  E1 `matter.scope == ""` —— 行层 fail-open（迁移兼容：存量 120/120 都是空，
     过滤掉等于把②平面清零）；文件层归 `personal/`（树本来就按 principal 分根）。
  E2 `pids is None` —— 行层不过滤；文件层只给 `personal/`（那条路径连 principal
     都不确定，给不出更多目录也不该猜）。
  E3 **跨 principal 的 `personal:` 可见性**（可见集合里有别人的 personal scope）
     —— 行层能表达（`pids` 是个列表），文件层表达不了：树按 principal 分根，
     读侧只走请求者自己那一棵。已登记为企业形态的表达力缺口，不在本卡修。
例外是**枚举出来的**，不是用 skip 抹掉的 —— 差异存在可以，悄悄存在不行。
"""

from __future__ import annotations

import itertools

import pytest
from bladex_core import flash
from bladex_core.matter import Matter
from bladex_proxy.storage.memory_index import MemoryIndex

_PRINCIPAL = "u1"

_SCOPES = ["", "personal:u1", "personal:other", "team:legal", "org:acme", "weird:x"]

#: 请求者恒为 `_PRINCIPAL`、读的也是它自己那棵树（写侧只产出这种形态）。
#: 「可见集合里有别人的 personal」单列在 E3，不进这张矩阵——那是表达力缺口，
#: 混进来只会把一条已登记的缺口伪装成一次对账失败。
_VIS_CASES: list[list[str]] = [
    [],                                            # 个人模式（未标注）
    ["personal:u1"],
    ["team:legal"],                                # 显式只可见 team ⇒ 行层 fail-closed
    ["personal:u1", "team:legal", "org:acme"],
    ["org:acme"],
]


@pytest.fixture
def index(tmp_path):
    """只用 `_visibility_pids` / `_matter_visible` 两个纯判定，不需要真的开库。"""
    return MemoryIndex(str(tmp_path / "index"))


@pytest.mark.parametrize("scope,vis", list(itertools.product(_SCOPES, _VIS_CASES)))
def test_file_layer_matches_row_layer(index, scope, vis):
    pids = index._visibility_pids(vis, _PRINCIPAL)
    row_visible = index._matter_visible(Matter(matter_id="m-1", scope=scope), pids, vis)

    card_dir = flash.scope_dir_of(scope, principal=_PRINCIPAL)
    readable = flash.readable_scope_dirs(pids, vis, principal=_PRINCIPAL)
    file_visible = bool(card_dir) and card_dir in readable

    if scope == "":                       # 具名例外 E1
        assert row_visible is True
        assert file_visible is (flash.SCOPE_PERSONAL in readable)
        return
    assert file_visible == row_visible, (
        f"scope={scope!r} vis={vis!r} pids={pids!r} "
        f"row={row_visible} file={file_visible} dir={card_dir!r} readable={readable}"
    )


def test_legacy_none_is_the_second_named_exception(index):
    """具名例外 E2：行层 `pids is None` 全放行，文件层只给 `personal/`。"""
    pids = index._visibility_pids(None, None)
    assert pids is None
    assert index._matter_visible(Matter(matter_id="m-1", scope="team:legal"), pids, None) is True
    assert flash.readable_scope_dirs(pids, None, principal=_PRINCIPAL) == [flash.SCOPE_PERSONAL]


def test_unplaceable_scope_is_invisible_at_both_layers(index):
    """`personal:<别人>`：行层判不可见，文件层拒绝落盘 —— 两边同向，不是例外。"""
    vis = ["personal:u1"]
    pids = index._visibility_pids(vis, _PRINCIPAL)
    m = Matter(matter_id="m-1", scope="personal:other")
    assert index._matter_visible(m, pids, vis) is False
    assert flash.scope_dir_of("personal:other", principal=_PRINCIPAL) == flash.SCOPE_DIR_REFUSED


def test_cross_principal_personal_visibility_is_the_third_named_exception(index):
    """具名例外 E3：可见集合里有**别人的 personal** —— 行层能表达，文件层不能。

    行层 `pids` 是列表，天然容得下别人的 pid；文件层的树按 principal 分根，
    读侧只走请求者自己那一棵，**没有地方表达"去读另一个人的 personal 目录"**。
    方向仍是文件层更严（读不到 ≠ 泄露），故不阻塞 D1/D2；
    真要支持得改布局，属企业形态，已登记不在本卡修。

    在**卡自己那棵树里**判定时两层是一致的——这条顺带证明缺口只在跨树读，
    不是映射本身错了。
    """
    vis = ["personal:other"]
    pids = index._visibility_pids(vis, _PRINCIPAL)
    assert pids == ["other"]
    m = Matter(matter_id="m-1", scope="personal:other")
    assert index._matter_visible(m, pids, vis) is True          # 行层：看得见
    assert flash.readable_scope_dirs(pids, vis, principal=_PRINCIPAL) == []  # 文件层：无门可进
    # 同一张卡在 other 自己那棵树里，两层一致
    assert flash.scope_dir_of("personal:other", principal="other") == flash.SCOPE_PERSONAL
    assert flash.readable_scope_dirs(pids, vis, principal="other") == [flash.SCOPE_PERSONAL]
