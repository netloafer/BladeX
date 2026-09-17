"""MQ-L38 · 切换候选面的相关性位（`ledger_runtime.relevant_ledgers`，2026-09-04）。

病例：候选列表只有 `updated_at` top-5，08-25 的「泰山啤酒」旧本排第 9，用户原话逐字重提时
模型面前没有"切回"这个选项，只能新建。本文件用**真实池的 27 个标题/Goal 回放**（09-04 快照，
写死在这里——池会变，尺子不能跟着变）钉住：该召的召、不该召的不召、判别力来自哪一项。
"""
from __future__ import annotations

import pytest
from bladex_core.ledger import new_ledger
from bladex_core.ledger_runtime import (
    _CANDIDATE_STOP,
    candidate_units,
    relevant_ledgers,
)

# 09-04 真实池快照（id 缩写 → title / Goal 摘要）。Goal 只取首句，够判别。
_POOL_SNAPSHOT = [
    ("c9ab", "泰山啤酒破产重整最新进展梳理", "小黑，帮我再梳理一下泰山啤酒破产重整的最新进展，看看有没有新的变化。"),
    ("d50e", "泰山啤酒破产重整进展梳理", "梳理泰山啤酒破产重整的最新进展：重整进程节点、法院/管理人公告、债权人申报"),
    ("3f90", "GitHub freetoken 项目调研", "调研 GitHub 上的 freetoken 项目：定位、原理、支持平台，只研究不执行"),
    ("5cc6", "设置默认报告输出目录", "把报告默认输出目录设置为 ~/Reports"),
    ("c4e1", "三星PIM存内计算深度分析", "深度分析三星 PIM 存内计算技术路线与产业化前景"),
    ("c4cb", "微信团队开源 embedding 模型调研", "调研微信团队开源的 embedding 模型，评估能否替换 BladeX 当前嵌入后端"),
    ("468d", "评估 v5 架构开发进展", "评估 BladeX v5 架构（ADR-0032）的开发进展"),
    ("fd18", "北京今日天气查询", "查询北京今日天气"),
    ("e5fe", "Codex ledger health 开发", "开发 BladeX Codex ledger health 任务卡"),
    ("497d", "苹果新 Mac mini / Mac Studio 亮点与性价比分析", "分析苹果新发布的 Mac mini / Mac Studio 的亮点与性价比"),
    ("bf8c", "jydesignhk.com CSS/JS 架构合理性评估", "评估 jydesignhk.com 的 CSS/JS 架构是否合理"),
    ("aaba", "BladeX 项目进展分析与简报", "分析 BladeX 项目进展并写一份简报"),
    ("2cfd", "V5 架构（ADR-0032）详细评价", "对 BladeX V5 架构 ADR-0032 做详细评价"),
    ("66d0", "[stale] 新调整的V5架构（CC 子调用误建）", "新调整的 V5 架构"),
    ("73cd", "Codex MCP Ledger Tools 开发", "在 Codex MCP 里开发 ledger tools"),
    ("24e0", "Codex MCP ledger tools 开发", "Codex MCP ledger tools 开发"),
    ("3cf9", "~/dev/Minecraft 网页版开发问题评估", "评估 ~/dev/Minecraft 网页版的开发问题"),
    ("8de8", "pi update 检查与更新", "检查 pi 有没有更新并更新"),
    ("8a54", "听山堂艺术家店铺详情页专业设计评价", "对听山堂艺术家店铺详情页做专业设计评价"),
    ("d7b9", "移动端首页375px专业设计评价", "移动端首页 375px 专业设计评价"),
]


def _pool():
    return {f"ldg-{k}": new_ledger(title=t, goal=g, goal_source="model",
                                  created_at="2026-08-25T00:00:00+00:00")
            for k, t, g in _POOL_SNAPSHOT}


def _ids(matches):
    return [m.ledger_id for m in matches]


# ── 病例回放 ─────────────────────────────────────────────────────────────

def test_taishan_verbatim_recalls_the_old_ledger_first():
    """09-03 23:58 那句原话 ⇒ 08-25 旧本必须在候选里，且排第一。"""
    m = relevant_ledgers(_pool(), "小黑，梳理泰山啤酒破产重整最新进展")
    assert _ids(m)[:2] == ["ldg-c9ab", "ldg-d50e"], m
    assert m[0].score >= 0.9


def test_active_ledger_is_excluded():
    m = relevant_ledgers(_pool(), "小黑，梳理泰山啤酒破产重整最新进展", exclude_id="ldg-c9ab")
    assert "ldg-c9ab" not in _ids(m) and _ids(m)[0] == "ldg-d50e"


@pytest.mark.parametrize("query,expect", [
    ("小黑，帮我梳理一下三星的PIM技术", "ldg-c4e1"),
    ("北京今天天气怎么样", "ldg-fd18"),
    ("把报告输出目录改一下", "ldg-5cc6"),
    ("微信开源的那个 embedding 模型后来怎么样了", "ldg-c4cb"),
    ("帮我看看 freetoken 那个项目还有什么新动向", "ldg-3f90"),   # 拉丁 token 计 2
])
def test_paraphrases_recall_the_right_ledger(query, expect):
    m = relevant_ledgers(_pool(), query)
    assert m and m[0].ledger_id == expect, (query, m)


@pytest.mark.parametrize("query", [
    "继续",
    "写一个 python 脚本统计文件数",
    "评估一下 BladeX v5 架构",     # 交集只有池里人人都有的词 ⇒ 无稀有锚
    "",
    "小黑，帮我看看",              # 全是停表词
])
def test_generic_requests_recall_nothing(query):
    assert relevant_ledgers(_pool(), query) == [], query


def test_top_n_and_ordering():
    m = relevant_ledgers(_pool(), "Codex MCP ledger tools 开发进展", top_n=2)
    assert len(m) <= 2
    assert m == sorted(m, key=lambda x: (-x.score, -x.shared, x.ledger_id))


# ── 判别力对照：拿掉某一项，测试必须红 ────────────────────────────────────

def test_stoplist_is_what_keeps_taishan_goal_from_matching_everything():
    """泰山旧本 Goal 是逐字原话「小黑，帮我…一下」——不停这些词，任何以「小黑」开头的
    请求都会命中它（沙盒回放：「三星 PIM」原本召回泰山 0.37）。"""
    units_stop = candidate_units("小黑，帮我梳理一下三星的PIM技术")
    assert not ({"小黑", "帮我", "一下", "梳理"} & units_stop)
    raw = {"小黑", "帮我", "一下", "梳理"}
    assert raw <= _CANDIDATE_STOP


def test_rare_anchor_is_what_keeps_bladex_generic_query_silent():
    """去掉稀有锚 ⇒「评估 BladeX v5 架构」召回 V5 架构评价等卡（归一后 0.47）——
    沙盒回放实测。这条钉的是"判别力来自稀有锚"，不是某个阈值碰巧。"""
    q = "评估一下 BladeX v5 架构"
    assert relevant_ledgers(_pool(), q) == []
    assert relevant_ledgers(_pool(), q, require_rare_anchor=False) != [], \
        "拿掉稀有锚后本该误召——如果这里绿了，说明判别力已不来自稀有锚，测试与实现同时失效"


def test_latin_token_counts_double():
    """`freetoken` 一个 token 就该定一件事；纯 CJK 单个 2-gram 不够。"""
    pool = {"ldg-x": new_ledger(title="GitHub freetoken 项目调研", goal="",
                                created_at="2026-08-25T00:00:00+00:00"),
            "ldg-y": new_ledger(title="别的", goal="", created_at="2026-08-25T00:00:00+00:00")}
    assert _ids(relevant_ledgers(pool, "freetoken 怎么样了")) == ["ldg-x"]
    assert relevant_ledgers(pool, "别的事") == []


def test_closed_ledgers_are_still_candidates():
    pool = _pool()
    pool["ldg-c9ab"] = pool["ldg-c9ab"].model_copy(update={"status": "closed"})
    assert "ldg-c9ab" in _ids(relevant_ledgers(pool, "小黑，梳理泰山啤酒破产重整最新进展"))


def test_units_shape():
    u = candidate_units("Mac Studio 值不值 freetoken")
    assert "mac" in u and "studio" in u and "freetoken" in u
    assert "值不" in u and "不值" in u
