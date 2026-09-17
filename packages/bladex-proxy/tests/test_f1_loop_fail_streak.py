"""内循环失败连击埋点（F1.5 / MQ-L54，2026-09-08）——**只埋点不熔断**。

## 病例

MQ-L52 那次 62 万 token 的放大器：同一条写错的 `bladex_ledger_update` 连撞 6 次，
仍照常烧到 `BLADEX_INNER_LOOP_MAX_ROUNDS=6`、零写入。而 `loop_done` 只报
`rounds` / `tools` / `degraded` —— **看不出这几轮是不是同一个错**。
`rounds=6` 有两种成因（六件不同的事 / 同一条错撞六次），在日志里长得一模一样。

## 🔴 为什么不顺手加个熔断（拍板 ⑬）

L52 修完之后模型**已经能自愈**：09-08 实测 3 个批次、4 次拒绝，重试 1–2 次后全部
成功，`rounds` 最大 3。此时拍一个阈值就有可能打断已经验证过的自愈
（同族先例：MQ-L44 的防抖阈值 —— 先量再改）。一周分布出来再定：正常自愈（1–2）
与病态（6）之间要有肉眼可见的间隔；**没间隔就说明熔断这条路不成立**，
改从错误消息侧解。这也是本卡要回答的问题之一。

## 判据三条并联

同名工具 ∧ 参数逐字相同 ∧ 结果以 `Error:` 开头。少了"参数相同"这条，
模型换着参数试探（正常自愈的形状）会被读成病态连击——那正是要区分开的两件事。
"""

from __future__ import annotations

import asyncio

from bladex_proxy.innerloop import run_inner_loop


def _call(cid: str, name: str, args: str) -> dict:
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": args}}


_BAD = '{"section": "next", "op": "remove"}'
_OK = '{"section": "next", "op": "remove", "match": "x"}'


def _run(script: list[tuple[str, str, str]], *, rounds_cap: int = 6,
         never_stops: bool = False):
    """`script` = 每轮一条 (工具名, 参数, 结果)。最后一轮之后回一条纯文本收尾。

    `call_llm` 按脚本喂下一轮的调用；脚本走完就回纯文本 ⇒ 循环正常结束
    （`classify_message` 判 none/mixed 即返回）。
    `never_stops=True` ⇒ 脚本走完仍重发最后一条 ⇒ 撞 `rounds_cap`（live 病例的形状）。
    """
    seq = list(script)

    async def dispatch(name: str, args: str) -> str:
        for n, a, result in seq:
            if n == name and a == args:
                return result
        return "ok"

    state = {"i": 1}

    async def call_llm(_messages):
        i = state["i"]
        state["i"] += 1
        if i >= len(seq):
            if not never_stops:
                return {"role": "assistant", "content": "done"}
            i = len(seq) - 1          # 原样重发最后一条
        n, a, _ = seq[i]
        return {"role": "assistant", "content": None,
                "tool_calls": [_call(f"c{i}", n, a)]}

    first = seq[0]
    return asyncio.run(run_inner_loop(
        messages=[{"role": "user", "content": "go"}],
        initial_calls=[_call("c0", first[0], first[1])],
        call_llm=call_llm, dispatch=dispatch,
        budget_s=9999, max_rounds=rounds_cap))


# ── ① 同工具同参数连续全失败 ────────────────────────────────────────────


def test_six_identical_failures_report_a_streak_of_six():
    """live 病例的形状：同一条写错原样重发，烧到 `MAX_ROUNDS`。"""
    res = _run([("bladex_ledger_update", _BAD,
                 "Error: entries[0] needs both section and op")],
               never_stops=True)
    assert res.fail_streak == 6
    assert res.degraded is True and res.degrade_reason == "max_rounds"
    assert sum(r.errors for r in res.rounds) == 6


# ── ② 失败-成功-失败 ⇒ 连击断在成功那一下 ───────────────────────────────


def test_success_between_failures_breaks_the_streak():
    """正常自愈的形状：撞一次、改对、再撞一次 ⇒ `fail_streak=1`，不是 2。"""
    res = _run([
        ("bladex_ledger_update", _BAD, "Error: needs both section and op"),
        ("bladex_ledger_update", _OK, "removed from next: x"),
        ("bladex_ledger_update", _BAD, "Error: needs both section and op"),
    ])
    assert res.fail_streak == 1
    assert sum(r.errors for r in res.rounds) == 2, "总量与连击是两轴，都要在"


# ── ③ 同工具、参数不同 ⇒ 不计连击 ──────────────────────────────────────


def test_different_arguments_do_not_count_as_a_streak():
    """模型换着参数试探 = 它在自愈，不是撞墙。三次失败但连击只算 1。"""
    res = _run([
        ("bladex_ledger_update", '{"op": "remove", "match": "a"}',
         "Error: no entry matching 'a'"),
        ("bladex_ledger_update", '{"op": "remove", "match": "ab"}',
         "Error: no entry matching 'ab'"),
        ("bladex_ledger_update", '{"op": "remove", "match": "abc"}',
         "Error: no entry matching 'abc'"),
    ])
    assert res.fail_streak == 1, "参数每次都不同 ⇒ 不是同一个错撞三遍"
    assert sum(r.errors for r in res.rounds) == 3


def test_different_tools_do_not_count_as_a_streak():
    res = _run([
        ("bladex_memory_search", '{"query": "q"}', "Error: memory index unavailable."),
        ("bladex_ledger_update", _BAD, "Error: needs both section and op"),
    ])
    assert res.fail_streak == 1


# ── ④ 全成功 ⇒ 0 ────────────────────────────────────────────────────────


def test_all_success_reports_zero():
    res = _run([("bladex_memory_search", '{"query": "q"}', "3 facts found")] * 3)
    assert res.fail_streak == 0
    assert sum(r.errors for r in res.rounds) == 0
    assert res.degraded is False


def test_loop_cost_carries_both_numbers():
    """三条 `*_loop_done` 日志都 `**_loop_cost(result)` —— 加在那里三处同时带。"""
    from bladex_proxy.agency import _loop_cost

    res = _run([("bladex_ledger_update", _BAD, "Error: boom")] * 3)
    cost = _loop_cost(res)
    assert cost["fail_streak"] == 3 and cost["errors"] == 3
    assert {"tool_ms", "llm_ms", "tokens", "tools"} <= set(cost), "老字段不许丢"


def test_instrument_does_not_break_the_loop():
    """🔴 只埋点不熔断：连撞 6 次照样跑满 6 轮（打断已验证的自愈是本卡明令禁止的）。"""
    res = _run([("bladex_ledger_update", _BAD, "Error: boom")], never_stops=True)
    assert len(res.rounds) == 6, "加了阈值就是打断 L52 修后已验证的自愈"


# ── ⑤ 判别力：去掉"参数相同" ⇒ ③ 必红 ─────────────────────────────────


def test_discriminative_ignoring_arguments_would_call_case_three_a_streak():
    """把判据缩成"同名工具 + 失败"（去掉参数相同）⇒ ③ 会读出 3。

    直接在测试里按那个错判据重算一遍同一份脚本——不必改生产码就能证明
    这一条**有区分度**（没有它，③ 的 `fail_streak==1` 可能只是因为脚本没触发连击）。
    """
    script = [("bladex_ledger_update", f'{{"match": "{s}"}}', "Error: no entry")
              for s in ("a", "ab", "abc")]
    res = _run(script)
    assert res.fail_streak == 1

    wrong, streak, key = 0, 0, None
    for name, _args, result in script:                  # 只看工具名
        if result.startswith("Error:"):
            streak = streak + 1 if key == name else 1
            key = name
            wrong = max(wrong, streak)
        else:
            key, streak = None, 0
    assert wrong == 3, "错判据会把三次不同的试探读成连撞三次"
