"""检索 query 的推导只有一份实现（`query_understanding.retrieval_query`）。

## 为什么需要这组测试

离线仪器要判"注进去的条目跟这一轮相不相关"，就必须拿**生产真正用于检索的
query** 去比。首版 `eval_consumption` 自己近似了一份（末条 user 消息原文），
参照系与被测系统脱钩，且偏差方向随 agent 相反：

    claude-code  末条 user p50 七万字符、98% 是内部协议信封
                 → 子串命中太容易 → 相关性**虚高**
    hermes       「继续」「这个怎么办」这类短指代
                 → 命中太难       → 相关性**虚低**

而这根轴要用来决定"先修检索（G3）还是先修蒸馏侧"。尺子偏了，选的就是尺子的偏差。

所以三处（`server._extract_query` / `inject._understand` / 离线仪器）统一委托到
`bladex_core.query_understanding`。本组钉两件事：**委托关系还在**，
以及**理解层的行为逐字未变**（这次是纯重构，不是行为变更）。
"""

from __future__ import annotations

from bladex_core.envelope import strip_envelopes
from bladex_core.query_understanding import (
    last_user_text,
    open_unit_intent,
    retrieval_query,
    understand_query,
)
from bladex_core.task_unit import build_task_units


def _legacy_understand(messages: list[dict], query: str) -> tuple[str, list[str]]:
    """重构前 `inject._understand` 里的那段（2026-08-18 之前的逐字副本）。

    留在测试里当对照物：生产实现改了行为，这里会红。
    """
    units = build_task_units(messages)
    open_intent = ""
    if units and 0 <= units[-1].intent_index < len(messages):
        c = messages[units[-1].intent_index].get("content")
        if isinstance(c, str):
            open_intent = c
        elif isinstance(c, list):
            open_intent = " ".join(
                part.get("text", "") for part in c
                if isinstance(part, dict) and part.get("type") == "text"
            )
    return understand_query(query, open_unit_intent=open_intent)


# ── 末条 user 取文 ──────────────────────────────────────────────────────


def test_last_user_text_takes_last_user_message():
    msgs = [
        {"role": "user", "content": "先看泰山啤酒"},
        {"role": "assistant", "content": "好"},
        {"role": "user", "content": "再看泰安仁信"},
    ]
    assert last_user_text(msgs) == "再看泰安仁信"


def test_last_user_text_handles_multimodal_list_content():
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "这张图"},
        {"type": "image_url", "image_url": {"url": "x"}},
        {"type": "text", "text": "里的表格"},
    ]}]
    assert last_user_text(msgs) == "这张图 里的表格"


def test_last_user_text_without_user_message_is_empty():
    """阴性对照：没有 user 消息时返回空串，不得抛、也不得回落到 assistant 正文。"""
    assert last_user_text([{"role": "system", "content": "x"}]) == ""
    assert last_user_text([]) == ""


# ── 委托关系（重构不得改行为）────────────────────────────────────────────


def test_retrieval_query_matches_legacy_inject_implementation():
    """逐字对照重构前的实现——三种形态都要一致。"""
    cases = [
        # ① 自足的正常 query：理解层不该动它
        [{"role": "user", "content": "泰安仁信入股泰山啤酒用的是增资扩股还是受让老股"}],
        # ② 短指代：应被 open 单元意图扩写
        [{"role": "user", "content": "查一下 LanceDB 碎片对检索延迟的影响"},
         {"role": "assistant", "content": "查到了"},
         {"role": "user", "content": "继续"}],
        # ③ 信封形态：应被剥离
        [{"role": "user", "content": "<transcript>一大段内部协议</transcript>\n真正的问题在这"}],
    ]
    for msgs in cases:
        raw = last_user_text(msgs)
        assert retrieval_query(msgs, raw) == _legacy_understand(msgs, raw)


def test_retrieval_query_defaults_raw_to_last_user_text():
    """不传 q_raw 时自己取末条 user——离线仪器走的就是这条路。"""
    msgs = [{"role": "user", "content": "BLADEX_HOTPATH_BUDGET_MS 现在是多少"}]
    assert retrieval_query(msgs) == retrieval_query(msgs, last_user_text(msgs))


def test_open_unit_intent_is_always_the_last_user_message():
    """🔴 **现状刻画，不是期望行为**（2026-08-18 写这组测试时撞出来的）。

    `open_unit_intent` 取 `build_task_units(messages)[-1].intent_index` 指向的
    消息，而 `build_task_units` 对**每条** user 消息都起一个新单元
    （`task_unit.py:155-172`）——所以"当前 open 单元的意图"永远就是末条 user
    消息本身，而它同时就是 `q_raw`。两个输入是同一条消息。

    后果（E3，代码恒等式，规模待读数）：E6.1 那两条扩写路径在生产里都是恒等的

        「继续」          → 用「继续」扩写自己 → "继续 继续"
        整条是信封的 query → 剥完为空，拿来扩写的那条剥完也为空 → 仍为空

    `test_adr0028_e6_retrieval.py` 里那两条扩写测试**手工传** `open_unit_intent`，
    从未走过 messages→units 这条生产路径，所以这件事一直绿着。

    本测试钉住现状，不当它是对的：**扩写真接上时这里会红**，届时把它改成
    断言扩写生效，并在 docstring 写明是有意的行为变更（纪律 6）。
    规模读数挂在 `eval_consumption` 的「短指代/信封轮」计数上，出数前不动它。
    """
    msgs = [
        {"role": "user", "content": "分析 LanceDB 索引碎片对热路径的影响"},
        {"role": "assistant", "content": "在查"},
        {"role": "user", "content": "继续"},
    ]
    assert open_unit_intent(msgs) == last_user_text(msgs) == "继续"
    q, _ids = retrieval_query(msgs)
    assert "LanceDB" not in q          # 上文没有被带进来
    assert q == "继续 继续"             # 用自己扩写自己


def test_envelope_is_stripped_before_retrieval():
    """信封剥离确实发生（同上，防"两边都没生效"的假一致）。"""
    body = "真正的问题：inject_timeout 为什么是 151ms"
    msgs = [{"role": "user", "content": f"<transcript>{'噪声 ' * 200}</transcript>\n{body}"}]
    q, _ids = retrieval_query(msgs)
    assert "transcript" not in q
    assert body in q
    assert q == strip_envelopes(msgs[0]["content"])[0].strip()[:512]


def test_open_unit_intent_is_empty_without_units():
    assert open_unit_intent([]) == ""
    assert open_unit_intent([{"role": "system", "content": "x"}]) == ""


def test_retrieval_query_is_deterministic():
    """离线重算与当时那次必须逐字相同——否则"复现生产 query"这句话不成立。"""
    msgs = [{"role": "user", "content": "泰山啤酒破产重整的关键节点"}]
    assert retrieval_query(msgs) == retrieval_query(msgs)
