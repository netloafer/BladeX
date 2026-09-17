"""`bladex_core.flash` 路径 / 落盘助手（原 G12.4/G12.5 渲染纯函数半边）。

2026-09-03 S5：ADR-0031 老渲染器（六槽卡 / 注入装配 / 反枢纽 / 布局迁移 / 孤儿清理）
连同其用例一起删除；本文件只剩三件仍在生产的事：

1. **路径成分单射清洗**（`safe_component`）——principal / agent / scope 是外部输入，
   未清洗直接拼路径 = 目录穿越；只清洗不补指纹 = 两个 team 撞成一个目录（静默）。
2. **落盘**（`write_if_changed`）：内容不变不写、原子替换不留 `.tmp`。
3. **树体检**（`summarize_tree`）：按 ADR-0032 布局数 `ledgers/`，`newest_mtime`
   是"内容最后一次变"不是"最后一次渲染"。

纯函数 + tmp_path，不碰 Memory Index。
"""

from __future__ import annotations

import pytest
from bladex_core import flash

# ── 1. 路径成分清洗 ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("raw,prefix", [
    ("../../etc/passwd", "_.._.._etc_passwd-"),
    ("..", "_..-"),
    (".", "_.-"),
    (".hidden", "_.hidden-"),
    ("", "_-"),
    ("   ", "_-"),
    ("a/b", "a_b-"),
    ("hermes:accept", "hermes_accept-"),
])
def test_unsafe_components_are_sanitized_with_fingerprint(raw, prefix):
    """principal / agent / scope 是外部输入，未清洗直接拼路径 = 目录穿越。

    清洗后**再补一段原串指纹**（G12.12 ①）：只清洗的话是多对一的，见下一条。
    空输入必须返回非空——空串会让 `root//USER.md` 把文件写到上一层。
    """
    got = flash.safe_component(raw)
    assert got.startswith(prefix), got
    assert len(got) == len(prefix) + 8
    assert "/" not in got and not got.startswith(".")


@pytest.mark.parametrize("raw", ["m-1a2b3c", "24345848", "claude-code", "personal",
                                 "org", "team-legal", "USER"])
def test_safe_components_pass_through_unchanged(raw):
    """真实 id 清洗是**恒等式** ⇒ 加了指纹机制也一个目录都不改名（存量零churn）。"""
    assert flash.safe_component(raw) == raw


def test_sanitization_is_injective_for_colliding_inputs():
    """🔴 两个不同的 scope 串**绝不许**折进同一个目录。

    只做字符替换的话 `team:a-b` 与 `team:a:b` 同折成 `team_a_b`，两个 team 的
    Flash 目录会合成一个——而 §6.9 选"文件级边界"的全部理由就是它比行级过滤更难
    写错。**一个会撞名的目录名把这条理由整个抵消，且失效是静默的**（两边都能读写）。
    """
    colliding = ["a-b", "a:b", "a/b", "a b", "a.b"]
    dirs = [flash.safe_component(x) for x in colliding]
    assert len(set(dirs)) == len(colliding), dirs
    # scope 层同理：两个 team 名不同 ⇒ 目录不同
    d1 = flash.scope_dir_of("team:a-b", principal="u1")
    d2 = flash.scope_dir_of("team:a:b", principal="u1")
    assert d1 != d2


def test_sanitized_component_never_escapes_root():
    p = flash.scope_dir("/flash", "../../root", "../../../etc/x")
    assert p.startswith("/flash/")
    assert "/../" not in p


# ── 2. 落盘：投影 = 源 ──────────────────────────────────────────────────────


def test_write_only_when_content_changed(tmp_path):
    """内容一字不变就不写——否则整棵树的 mtime 每两分钟跳一次，
    用户看目录时分不出"哪张卡真的动了"。"""
    p = str(tmp_path / "a" / "USER.md")
    assert flash.write_if_changed(p, "x\n") == 1
    assert flash.write_if_changed(p, "x\n") == 0
    assert flash.write_if_changed(p, "y\n") == 1
    assert (tmp_path / "a" / "USER.md").read_text(encoding="utf-8") == "y\n"


def test_write_leaves_no_tmp_behind(tmp_path):
    """原子替换：写完不许留 `.tmp`（读侧撞见半份注入包比没有更糟）。"""
    flash.write_if_changed(str(tmp_path / "m.md"), "content")
    assert [p.name for p in tmp_path.iterdir()] == ["m.md"]


# ── 3. 树体检 ───────────────────────────────────────────────────────────────


def test_summarize_counts_ledgers_and_user_files(tmp_path):
    """按 ADR-0032 布局：`ledgers/<桶>/<id>.md` 计入账本数；`*.local.md` 单列。"""
    root = str(tmp_path)
    base = f"{flash.scope_dir(root, 'u1')}"
    flash.write_if_changed(f"{base}/ledgers/0f/ldg-1.md", "a")
    flash.write_if_changed(f"{base}/ledgers/24/ldg-2.md", "b")
    flash.write_if_changed(f"{base}/ledgers/24/ldg-2.local.md", "我的目标")
    flash.write_if_changed(f"{base}/claude-code/AGENT.md", "# Agent")   # 机器文件，不是账本
    s = flash.summarize_tree(root)
    assert (s.principals, s.ledgers, s.user_files) == (1, 2, 1)
    assert s.newest_mtime > 0


def test_summarize_missing_root_is_all_zero(tmp_path):
    """还没渲染过 ⇒ 全 0，不炸也不猜。"""
    s = flash.summarize_tree(str(tmp_path / "nope"))
    assert (s.principals, s.ledgers, s.newest_mtime) == (0, 0, 0.0)


def test_summary_field_is_named_change_not_render():
    """🔴 参照系：`newest_mtime` 是"内容最后一次变"，**不是"最后一次渲染"**。

    `write_if_changed` 内容没变就不写，所以旧 mtime 的正常含义是"这段时间没有新东西"。
    据此报"渲染停了"就是尺子按自己的假设判读被测系统——本仓库的高发病。
    这条守的是**命名与文档**：字段名不许出现 render，docstring 必须写明两者不同。
    """
    assert "render" not in "newest_mtime"
    doc = flash.FlashTreeSummary.__doc__ or ""
    assert "不是" in doc and "渲染" in doc


