"""G12.1：`matter_id` 与标题解耦（ADR-0031 §3.4）。

## 这一条改的是什么

旧规则 `matter_id = "m-" + sha256(_normalize(title))[:12]` 让 **Matter 的身份
就是它的标题字符串**。于是"改标题"与"换一件事"在系统里是同一个动作：

    一轮只看得见一个动作 → 蒸馏产出动作句 → 动作句当身份哈希
    → 换个动词 = 一张新卡 → 碎片率 ≈80%（MQ-S17：14 张人工签收成 3 张）

新规则取**这件事首轮那条 Memory Hub key**。

## 代价已经兑现过一次（所以这组测试不是防御性演习）

2026-08-19 的全量重建（`logs/rebuild-v7-20260819.log`）就是"标题改了、身份还绑在
标题上"的后果：

    index_admin_events_replayed  count=46
    index_matters_merge_noop     reason=source_missing   ← 44 条，成功合并 0

历史上所有手动合并（含当天刚人工签收的那批 14→3）被静默丢弃。

## 四条断言，逐条对应验收项

1. 同一首轮重放两次 → 同一 id（重建等价性；ADR-0009 没被破）
2. **改标题不改 id**（范式转换的技术前提：身份首轮定、之后不重推）
3. 不同的事 → 不同 id（否则解耦就成了"全都同一张卡"这个最容易达标的坏解）
4. **存量 id 零变动**：旧规则的函数必须原样还在、且逐字给出旧值
"""

from __future__ import annotations

import hashlib

from bladex_core.attribution import (
    _deterministic_matter_id,
    _normalize,
    new_matter_id,
)

# ── 1. 重建等价性 ──────────────────────────────────────────────────────────


def test_same_first_turn_replays_to_the_same_id() -> None:
    """同一条 ledger key 重放两次 → 同一 id。

    这是 ADR-0009 重建等价性在本改动上的落点：首轮那条 Turn 是确定的、
    append-only 的、跨 rebuild 稳定的 ⇒ 重放同样的 Memory Hub 得同样的 id。
    对照旧规则：它的"稳定"依赖蒸馏每次产出同一个标题，而那是个 LLM 输出。
    """
    key = "local/hermes:default/sess-abc/1723459200-0"
    assert new_matter_id(ledger_key=key) == new_matter_id(ledger_key=key)


def test_id_is_a_pure_function_of_the_ledger_key() -> None:
    key = "local/claude-code/sess-1/1700000000-0"
    assert new_matter_id(ledger_key=key) == "m-" + hashlib.sha256(
        key.encode()).hexdigest()[:12]


# ── 2. 改标题不改 id（本改动的全部目的）────────────────────────────────────


def test_renaming_does_not_change_identity() -> None:
    """同一首轮、四个不同标题 → 同一个 id。

    四个标题取自 MQ-S17 的真实病例：它们是同一件事（泰山啤酒破产重整）
    的四个侧面，旧规则下是四张卡。
    """
    key = "local/hermes:default/sess-taishan/1723000000-0"
    titles = [
        "泰山啤酒股东结构及股权变动核查",
        "泰山啤酒知识产权分析",
        "泰山啤酒资产与品牌价值评估",
        "泰山啤酒破产重整",
    ]
    ids = {new_matter_id(ledger_key=key, title=t) for t in titles}
    assert len(ids) == 1, "改标题改动了身份 —— 解耦没生效"


def test_old_rule_would_have_produced_four_ids() -> None:
    """对照组：同一批标题走旧规则确实是四个 id。

    没有这条，上一条断言可能只是因为"这四个标题恰好规范化后相同"。
    对照组把"新规则做对了"与"输入本来就一样"分开。
    """
    titles = [
        "泰山啤酒股东结构及股权变动核查",
        "泰山啤酒知识产权分析",
        "泰山啤酒资产与品牌价值评估",
        "泰山啤酒破产重整",
    ]
    assert len({_deterministic_matter_id(t) for t in titles}) == 4


# ── 3. 不同的事仍然是不同的卡（防"全合成一张"这个坏解）────────────────────


def test_different_first_turns_get_different_ids() -> None:
    a = new_matter_id(ledger_key="local/a/sess-1/1700000000-0", title="同一个标题")
    b = new_matter_id(ledger_key="local/a/sess-2/1700009999-0", title="同一个标题")
    assert a != b


# ── 4. 存量 id 零变动 ─────────────────────────────────────────────────────


def test_legacy_rule_is_unchanged_bit_for_bit() -> None:
    """🔴 存量卡的身份由旧函数生成，它必须逐字不动。

    这条断言把旧规则的输出**钉成常量**：任何"顺手统一一下哈希写法"的改动
    都会让全库 Matter 失联（管理事件重放全部 source_missing），
    而那正是 08-19 已经发生过一次的事故。
    """
    title = "查询泰安仁信文化投资合伙企业（有限合伙）入股泰山啤酒的方式"
    expected = "m-" + hashlib.sha256(_normalize(title).encode()).hexdigest()[:12]
    assert _deterministic_matter_id(title) == expected
    # 规范化仍是 NFKC + casefold + 折叠空白（换了它等于换了全库 id）
    assert _normalize("  Ａ  B  ") == "a b"


def test_missing_ledger_key_falls_back_to_the_legacy_rule() -> None:
    """没有 ledger key 的调用方（合成 fact / 存量回放 / 手工构造）要有确定性回落。

    回落到旧规则而不是抛错：id 生成在归属管线深处，抛错会让一条 fact 整个丢掉，
    而"退回旧身份"是无损的——它本来就是那条 fact 在旧世界里的身份。
    """
    for empty in ("", "   ", None):
        assert new_matter_id(ledger_key=empty or "", title="某个标题") == \
            _deterministic_matter_id("某个标题")


def test_new_ids_do_not_collide_with_legacy_ids_in_format() -> None:
    """两代 id 会在库里共存（存量不改），格式必须一致，下游按 `m-` + 12 hex 解析。"""
    new = new_matter_id(ledger_key="local/a/s/1-0")
    old = _deterministic_matter_id("t")
    for mid in (new, old):
        assert mid.startswith("m-") and len(mid) == 14
        int(mid[2:], 16)  # 必须是合法 hex，否则截断 id 那类事故会重演
