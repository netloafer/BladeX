"""M1-2：长内容分流（附录 D.4）+ 通道合一的验收剧本。

复核附录 D 的判定：**拆分修的是提取率，付的是投毒率——单元定义错了**。
剥完信封仍超长的内容几乎从来不是"很长的用户话语"，而是粘贴物/规则文件；
把文档切成六句假话语去走"用户陈述蒸馏"，每一段都在错误的语义契约下被处理。

X6 纪律：本文件的样本词汇不得出现在实现代码里。分流判据只看**结构**
（长度、标记密度），不看任何领域词。
"""

from __future__ import annotations

import pytest

from bladex_core.distill_routing import (
    DISCOURSE_MAX_CHARS,
    DOCUMENT_HARD_CHARS,
    InputRoute,
    classify_input,
)


def _long(text: str, filler: str = "补充说明的文字内容。") -> str:
    """把样本补到超过话语上限。

    刻意用程序补而不是手写重复次数：手工数 CJK 字数很容易数错，
    一错样本就落在 500 以下走早返回，测到的是"短文本恒话语"而不是分流本身
    （本文件初版三条测试就是这么假过的）。
    """
    while len(text) <= DISCOURSE_MAX_CHARS:
        text += filler
    return text


# ── 话语路：契约成立的那一段 ──────────────────────────────────────────────


@pytest.mark.parametrize("text", [
    "帮我看一下这个报错是怎么回事",
    "回复请简洁一些，不要长篇大论",
    "",
    "   ",
])
def test_short_text_is_always_discourse(text):
    """≤500 字符不必再问性质——这个长度的内容就是用户说的话。"""
    assert classify_input(text) is InputRoute.DISCOURSE


def test_long_prose_without_structure_is_still_discourse():
    """真·超长话语（实测 <5%）仍走话语路，交给分段。

    判据是**结构**不是长度：一段长但没有任何文档标记的连续叙述，
    是人在认真描述一件事，不是粘贴物。
    """
    prose = _long("我想说的是这件事有点复杂，")  # 无任何结构标记
    assert len(prose) > DISCOURSE_MAX_CHARS
    assert classify_input(prose) is InputRoute.DISCOURSE


# ── 文档路：粘贴物（失效链 A/B 的入口）────────────────────────────────────


def test_fenced_code_block_is_document():
    """带代码围栏的长内容是粘贴物。硬切它会产出半句（失效链 B）。"""
    text = "看看这个：\n\n```python\n" + "x = 1\n" * 200 + "```\n"
    assert classify_input(text) is InputRoute.DOCUMENT


def test_many_headings_is_document():
    text = _long("\n\n".join(f"## 第{i}节\n这一节的正文。" for i in range(6)))
    assert classify_input(text) is InputRoute.DOCUMENT


def test_long_list_is_document():
    text = _long("清单如下：\n" + "\n".join(f"- 第 {i} 项说明" for i in range(12)))
    assert classify_input(text) is InputRoute.DOCUMENT


def test_table_is_document():
    rows = "\n".join(f"| 甲{i} | 乙{i} | 丙{i} |" for i in range(6))
    text = _long("对比表：\n" + rows + "\n")
    assert classify_input(text) is InputRoute.DOCUMENT


def test_very_long_unstructured_paste_is_document():
    """超过硬上限即便零结构也判文档。

    人不会一口气打 4000+ 字还不分段——那是粘贴物。
    """
    text = "无结构的连续文字" * 600
    assert len(text) >= DOCUMENT_HARD_CHARS
    assert classify_input(text) is InputRoute.DOCUMENT


# ── 规则文件路：失效链 A 的根修 ───────────────────────────────────────────


def test_contents_of_marker_is_rulefile():
    """`Contents of X.md` 是 agent 注入规则文件的固定形态。"""
    text = ("Contents of /somewhere/AGENTS.md (project instructions):\n\n"
            + "这里是规则正文。" * 60)
    assert classify_input(text) is InputRoute.RULEFILE


def test_rulefile_body_without_wrapper_is_rulefile():
    """整条就是一份规则文件正文回传（无 `Contents of` 包头）。"""
    text = _long("# AGENTS.md\n\n## 第一节\n正文。\n\n## 第二节\n正文。\n\n## 第三节\n正文。")
    assert classify_input(text) is InputRoute.RULEFILE


def test_mentioning_a_rulefile_in_prose_is_not_rulefile():
    """零误伤：正文里顺口提到规则文件名 ≠ 这条内容是规则文件。

    文件名判据限定在**头部**，且要求同时有 markdown 结构。
    """
    text = _long("我刚才把 CLAUDE.md 里那段说明改了一下，你看看还有没有问题。")
    assert classify_input(text) is not InputRoute.RULEFILE


def test_rulefile_wins_over_document():
    """规则文件同时满足文档特征——必须先判规则文件，否则走错通道。

    两条路的产出不同：规则文件 → profile_obs（画像原料，不进注入面）；
    文档 → 文件内容索引。判反了就是把规则文件的内容做成可召回的记忆。
    """
    text = ("Contents of AGENTS.md:\n\n## 甲\n" + "正文" * 100
            + "\n\n## 乙\n" + "正文" * 100 + "\n\n## 丙\n" + "正文" * 100)
    assert classify_input(text) is InputRoute.RULEFILE


# ── 确定性（重建等价性 G6 的前提）────────────────────────────────────────


def test_classification_is_deterministic():
    samples = ["短句", "长文" * 400, "## 甲\n## 乙\n## 丙\n" + "文" * 300]
    for s in samples:
        assert classify_input(s) is classify_input(s)


def test_short_rulefile_marker_still_routes_to_rulefile():
    """显式包头先于长度判定：短的规则文件同样不是用户话语。

    长度门槛问的是"这是不是用户说的话"，而 `Contents of X.md` 本身就是
    "不是"的正面证据。本条是跑 compare_distill_calls.py 时发现的缺口
    （一条 327 字符的规则文件被长度早返回当成了话语）。
    """
    text = "Contents of /w/AGENTS.md:\n\n" + "规则正文。" * 20
    assert len(text) < DISCOURSE_MAX_CHARS
    assert classify_input(text) is InputRoute.RULEFILE


def test_short_prose_mentioning_rulefile_is_not_hijacked():
    """反向红线：名称启发式**不**享受长度豁免，短话语不许被误判。"""
    assert classify_input("我改了下 CLAUDE.md 你看看") is InputRoute.DISCOURSE
    assert classify_input("AGENTS.md 那段要不要同步改？") is InputRoute.DISCOURSE
