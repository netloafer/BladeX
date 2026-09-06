"""账本锚卡的匹配面（`task-anchor-materialize-20260901.md` T2，2026-09-02 回退后）。

锚卡此前建出来时 `aliases=[]`，而 L2/L3 匹配的就是「标题 + aliases」⇒ 匹配面为空、
结构上不可能赢（v5 那件事 18 条边里 17 条在锚卡已在场 11 分钟时仍被内容卡截胡）。

🔴 **首版给多了**：`[title] + derive_topic_keys(topic=goal)` 把 Goal 整句按标点
切碎成 `adr` / `qwen3` / `模型` / `codex` 这类库内高频残片，一轮全量重建把
ADR-0024 的开发过程吸进「v5 架构」卡、把 qwen3.8:27b 与 ornith-1.5 的对比测试
吸进「Qwen3.8 Flash Next」卡（Jason 人工判读，2026-09-02）。

本文件钉死回退后的口径：**只有标题**，以及「多给会出事」这条判别力对照。
"""

from __future__ import annotations

import inspect

from bladex_core.ledger_runtime import ledger_anchor_aliases


def test_only_the_title() -> None:
    assert ledger_anchor_aliases("评估 v5 架构开发进展") == ["评估 v5 架构开发进展"]


def test_whitespace_is_trimmed() -> None:
    assert ledger_anchor_aliases("  北京今日天气查询  ") == ["北京今日天气查询"]


def test_empty_title_yields_empty() -> None:
    """空标题 ⇒ 空匹配面，不能返回 `['']`——空串会匹配一切。"""
    assert ledger_anchor_aliases("") == []
    assert ledger_anchor_aliases("   ") == []


def test_deterministic() -> None:
    """锚卡 id 是确定性的，它的匹配面也必须是（重建等价）。"""
    assert ledger_anchor_aliases("三星PIM存内计算深度分析") == \
        ledger_anchor_aliases("三星PIM存内计算深度分析")


# ── 🔴 结构性护栏：签名窄到不可能重犯 ──────────────────────────────────────


def test_signature_takes_only_the_title() -> None:
    """签名**只有 title**——Goal 与成员都进不来，两类事故结构上不可能复发。

    - 进 Goal ⇒ 2026-09-02 那次：整句按标点切碎成库内高频残片，撑开匹配面。
      泰山啤酒卡当时几乎没误吸**纯属标点偶然**（Goal 用全角逗号 U+FF0C，而
      `_SPLIT` 只含半角 `,`，整句没被切开 ⇒ 实际等于只有标题）。
      **一个正确结果由标点决定，说明那个机制本身不成立。**
    - 进成员 ⇒ MS-16 的累积并集雪球（永安←泰山 25 条错边）。

    要加参数，先在 PR 里回答这两条各自怎么不复发。
    """
    params = set(inspect.signature(ledger_anchor_aliases).parameters)
    assert params == {"title"}, (
        f"签名变了：{sorted(params)}。加 goal ⇒ 句子切碎撑开匹配面；"
        "加成员 ⇒ MS-16 滚雪球。两条都有实测代价，不是理论担忧。")


def test_no_sentence_shredding() -> None:
    """判别力对照：整句 Goal 式的输入进来也只当一个标题，**不许切碎**。

    首版会把这句切成 ['基于此前对', '架构的分析,评估当前', '对照', 'adr', ...]，
    其中 `adr` / `评估` 在本库遍地都是。
    """
    sentence = "基于此前对 v5 架构的分析,评估当前 v5 开发进展,对照 ADR-0032 任务框架,给出简报。"
    out = ledger_anchor_aliases(sentence)
    assert out == [sentence], f"标题被切碎了：{out}"
    assert "adr" not in [a.lower() for a in out]
    assert len(out) == 1
