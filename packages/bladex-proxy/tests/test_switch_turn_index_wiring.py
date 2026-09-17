"""MQ-L40 · 流式拦截路径漏传 `turn_index` ⇒ 账本切换防抖恒拒（2026-09-04）。

live（`proxy-20260904-003029.log`）：09-04 五次 `agency_switch_debounced`，目标是相关性候选段里的
正当旧本（freetoken / 泰山 ×2），相隔 4–13 轮——防抖窗口只有 3 轮，不该拦。根因：两条流式路径
（`intercept_chat_stream` / `_intercept_protocol_stream`）构造工具上下文时没有 `turn_index`，
非流式路径 server 也没传 ⇒ `context.get("turn_index", 0)` 恒 0 ⇒ `turn_index - last = 0 < 3`
⇒ 一个 scope 上**只有进程内第一次切换能落定**（`last is None`），之后"切到已有账本"全被拒；
唯一能过的是 `ledger_id=''` 新建（`newly_created` 豁免）——09-03 泰山重复建本的第二个成因。
历史证据：每个 proxy 进程恰好 1 条 `ledger_switch` 事件（08-30 起 7 个进程日志）。

钉三件事：① `tool_context` 是唯一实现点（源码里 `"user_query": _last_user_text(` 只许出现一次）；
② 缺省时 turn_index 按 identity 同一公式从 `upstream_messages` 派生；③ 端到端：先切一本、
再过 ≥3 轮切另一本 ⇒ 落定；用旧行为（turn_index 恒 0）同一剧本必被防抖——判别力对照。

**E0.2（2026-09-06，MQ-L44 根治 + MQ-L48）**：轮的单位改为**用户轮**（`role=user` 消息数，
`ledger_runtime.user_turn_index`），防抖只在同 session 内判。本文件既有用例按新口径改判据、
不删；新增：同一用户轮内的工具循环不算轮（判别力对照：口径改回 `len//2` ⇒ 该用例必红）、
跨 session 任意差放行。
"""
from __future__ import annotations

import asyncio
import json
import pathlib

import structlog.testing

from bladex_core.ledger import new_ledger
from bladex_proxy.agency import AgencyRuntime, tool_context

_SYS = {"role": "system", "content": "You are a test agent."}


def _on(monkeypatch):
    monkeypatch.setenv("BLADEX_MODULE_TOOLFACE", "1")
    monkeypatch.setenv("BLADEX_MODULE_LEDGER", "1")
    monkeypatch.setenv("BLADEX_MODULE_INTERCEPTION", "1")


def _msgs(n_turns: int) -> list[dict]:
    out = [_SYS]
    for i in range(n_turns):
        out += [{"role": "user", "content": f"u{i}"}, {"role": "assistant", "content": f"a{i}"}]
    out.append({"role": "user", "content": "切一下账本"})
    return out


def _msgs_tool_loop(n_turns: int, n_tools: int) -> list[dict]:
    """`_msgs(n_turns)` 之后追加 n_tools 个 assistant/tool 对——同一用户轮内的工具循环。"""
    out = _msgs(n_turns)
    for k in range(n_tools):
        out += [{"role": "assistant", "content": None,
                 "tool_calls": [{"id": f"t{k}", "type": "function",
                                 "function": {"name": "bash", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": f"t{k}", "content": "ok"}]
    return out


def _switch_msg(lid: str) -> dict:
    return {"role": "assistant", "content": None,
            "tool_calls": [{"id": "c1", "type": "function",
                            "function": {"name": "bladex_ledger_switch",
                                         "arguments": json.dumps({"ledger_id": lid})}}]}


async def _final(*_a, **_k):
    return {"role": "assistant", "content": "ok"}


def _process(ag: AgencyRuntime, lid: str, n_turns: int, *, turn_index: int = 0,
             session_id: str = "s1", messages: list[dict] | None = None):
    return asyncio.run(ag.process_message(
        _switch_msg(lid), upstream_messages=messages or _msgs(n_turns), session_prefix="p/",
        allowed_exposure="public", call_llm=_final, session_id=session_id,
        agent_id="hermes:default", turn_index=turn_index))


# ── ① 单实现点 ────────────────────────────────────────────────────────────

def test_tool_context_is_the_only_ctx_builder():
    from _source_probe import package_source
    src = package_source("agency")   # F0.1 拆包：tool_context 在 runtime.py，三个调用点分布 runtime/streams
    assert src.count('"user_query": _last_user_text(') == 1, \
        "工具上下文又长出第二份字面量了——漏传 turn_index 就是这么来的（三处各写一份，两处漏）"
    assert src.count("ctx = tool_context(") == 3, "三条拦截路径都要走 tool_context"


# ── ② 缺省派生 ────────────────────────────────────────────────────────────

def test_turn_index_defaults_to_identity_formula():
    """E0.2：公式 = `user_turn_index`（用户轮）。`_msgs(7)` 有 8 条 user；追加 5 个工具往返不变。"""
    from bladex_core.ledger_runtime import user_turn_index
    ctx = tool_context(session_id="s", agent_id="a", project_id="", upstream_messages=_msgs(7))
    assert ctx["turn_index"] == user_turn_index(_msgs(7)) == 8
    assert tool_context(session_id="s", agent_id="a", project_id="",
                        upstream_messages=_msgs_tool_loop(7, 5))["turn_index"] == 8, \
        "工具循环不是用户轮（修前 len//2 会算成 13）"
    assert tool_context(session_id="s", agent_id="a", project_id="",
                        upstream_messages=_msgs(7), turn_index=42)["turn_index"] == 42
    assert tool_context(session_id="s", agent_id="a", project_id="",
                        upstream_messages=[])["turn_index"] == 0


def test_identity_and_tool_context_share_the_single_formula():
    """单一实现点守卫：identity / agency 源码里不许再出现 `len(...) // 2` 的轮次公式。"""
    import re
    root = pathlib.Path(__file__).resolve().parents[1] / "bladex_proxy"
    # 只抓**代码形态**（赋值/返回里的公式），docstring 里的"修前 len(messages)//2"是历史记录。
    pat = re.compile(r"(=|return)\s*(max\(0,\s*)?len\((request\.)?(upstream_)?messages( or \[\])?\)\s*//\s*2")
    hits = [f.name for f in (root / "identity.py", *sorted((root / "agency").glob("*.py")))   # F0.1 拆包
            if pat.search(f.read_text(encoding="utf-8"))]
    assert hits == [], f"轮次公式又长出第二份：{hits}（应调 ledger_runtime.user_turn_index）"


# ── ③ 端到端 + 判别力对照 ─────────────────────────────────────────────────

def _seed(ag: AgencyRuntime):
    for lid, t in (("ldg-a", "任务 A"), ("ldg-b", "任务 B")):
        ag.pool[lid] = new_ledger(title=t, goal=t, ledger_id=lid,
                                  created_at="2026-09-01T00:00:00+00:00")


def test_switch_after_debounce_window_lands(monkeypatch):
    _on(monkeypatch)
    ag = AgencyRuntime()
    _seed(ag)
    scope = ag.scope_of("hermes:default")
    with structlog.testing.capture_logs() as cap:
        _process(ag, "ldg-a", n_turns=2)          # 用户轮 3：首次绑定，落定
        assert ag.activation.active(scope) == "ldg-a"
        _process(ag, "ldg-b", n_turns=3)          # 用户轮 4：相隔 1 < 3，防抖（这是防抖该拦的）
        assert ag.activation.active(scope) == "ldg-a"
        _process(ag, "ldg-b", n_turns=8)          # 用户轮 9：相隔 6 ≥ 3，必须落定
        assert ag.activation.active(scope) == "ldg-b", "隔 6 轮的正当切换被防抖拦下 = MQ-L40 复发"
    rows = [e for e in cap if e.get("event") == "agency_switch_debounced"]
    assert len(rows) == 1 and rows[0]["turn_index"] == 4 and rows[0]["last_switch_turn"] == 3, \
        "防抖告警必须带 turn_index / last_switch_turn 读数——没有它们 09-04 那五条只能猜"
    assert rows[0]["last_switch_session"] == "s1", "E0.2：读数要带参照系（同 session 才可比）"


def test_tool_loop_within_one_user_turn_is_still_debounced(monkeypatch):
    """E0.2 / MQ-L48：首次切换后同一用户轮内跑 6 个工具调用（消息 +12，修前算 6 轮）再切
    ⇒ 仍在防抖窗内（用户轮没动）；下一个用户轮再隔 3 轮才放行。事件 payload 带 turn_index。"""
    _on(monkeypatch)
    ag = AgencyRuntime()
    _seed(ag)
    scope = ag.scope_of("hermes:default")
    _process(ag, "ldg-a", n_turns=2)                                   # 用户轮 3 落定
    _process(ag, "ldg-b", n_turns=2, messages=_msgs_tool_loop(2, 6))   # 仍是用户轮 3
    assert ag.activation.active(scope) == "ldg-a", "工具循环被当成轮次 ⇒ 防抖失效（MQ-L48）"
    _process(ag, "ldg-b", n_turns=5, messages=_msgs_tool_loop(5, 6))   # 用户轮 6：Δ=3 放行
    assert ag.activation.active(scope) == "ldg-b"


def test_old_request_count_formula_lets_tool_loop_through(monkeypatch):
    """判别力对照：口径改回 `len//2`（请求数）⇒ 上一条的工具循环用例**必红**
    （6 个工具往返 = +6 "轮"，防抖窗被工具调用刷穿）。"""
    _on(monkeypatch)
    import bladex_core.ledger_runtime as lr
    monkeypatch.setattr(lr, "user_turn_index", lambda msgs: max(0, len(msgs or []) // 2))
    ag = AgencyRuntime()
    _seed(ag)
    scope = ag.scope_of("hermes:default")
    _process(ag, "ldg-a", n_turns=2)
    _process(ag, "ldg-b", n_turns=2, messages=_msgs_tool_loop(2, 6))
    assert ag.activation.active(scope) == "ldg-b", \
        "请求数口径下工具循环本该刷穿防抖——没刷穿说明防抖不再依赖 user_turn_index"


def test_old_behaviour_turn_index_zero_debounces_forever(monkeypatch):
    """判别力对照：把 turn_index 钉成 0（修前形态）⇒ 第二次切换永远被拒。"""
    _on(monkeypatch)
    ag = AgencyRuntime()
    _seed(ag)
    scope = ag.scope_of("hermes:default")
    import bladex_proxy.agency.runtime as agency_mod   # F0.1 拆包：消费方 process_message 在 runtime.py
    real = agency_mod.tool_context
    monkeypatch.setattr(agency_mod, "tool_context",
                        lambda **kw: {**real(**kw), "turn_index": 0})
    _process(ag, "ldg-a", n_turns=2)
    _process(ag, "ldg-b", n_turns=8)
    assert ag.activation.active(scope) == "ldg-a", \
        "turn_index 恒 0 时本该拒——这里绿了说明防抖不再看 turn_index，测试与实现同时失效"


# ── ④ C6c：防抖回文必须是明确否定态（09-04 09:10 病例：旧回文被读成"已切过"）──

def test_debounce_reply_is_an_explicit_negative(monkeypatch):
    _on(monkeypatch)
    ag = AgencyRuntime()
    _seed(ag)
    _process(ag, "ldg-a", n_turns=2)                       # turn 3 落定
    msg, transcript, _mode = _process(ag, "ldg-b", n_turns=3)   # turn 4：拒
    tool_texts = " ".join(str(m.get("content", "")) for m in transcript if m.get("role") == "tool")
    assert "NOT switched" in tool_texts and "ldg-a" in tool_texts and "任务 A" in tool_texts, tool_texts
    assert "after 2 more turn" in tool_texts, "要告诉模型还差几轮（3 − (4−3) = 2）"
    assert "Do NOT record" in tool_texts and "do not report the switch as done" in tool_texts
    assert "switched too recently" not in tool_texts, "旧措辞不许再出现——它被读成'已经切过'"


# ── ⑤ MQ-L44：跨会话（turn_index < last_switch_turn）不判防抖 + 模型面文档对账 ──

def test_new_session_lower_turn_index_can_still_switch(monkeypatch):
    """🔴 MQ-L44：新会话的 `turn_index` 从头算，`last_switch_turn` 跨会话持久 ⇒
    差值为负。修前 `-16 < 3` 判成抖动，新会话直接切不动（live 13:06:09）。
    E0.2 根治后判据 = 同 session 才相减：这里**换 session**，任意差都放行（含 Δ 为正的小差）。"""
    _on(monkeypatch)
    ag = AgencyRuntime()
    _seed(ag)
    scope = ag.scope_of("hermes:default")
    _process(ag, "ldg-a", n_turns=17, session_id="tw:local:Pi:1")        # 用户轮 18 落定
    assert ag.activation.active(scope) == "ldg-a"
    _process(ag, "ldg-b", n_turns=1, session_id="tw:local:Pi:2")         # 新会话：用户轮 2 < 18
    assert ag.activation.active(scope) == "ldg-b", \
        "换了会话不是抖动 —— MQ-L44 复发"
    _process(ag, "ldg-a", n_turns=2, session_id="tw:local:Pi:3")         # 又一个新会话：Δ=+1
    assert ag.activation.active(scope) == "ldg-a", "跨 session 的正小差也不是抖动（不同尺子）"


def test_tools_md_debounce_row_matches_the_real_message():
    """🔴 模型面文档必须与运行时回文对账（MQ-L42 同族：给模型看的假话最贵）。

    C6c 改了运行时回文，`config/system/TOOLS.md` 的错误对照表**没跟**——那一行还写着
    旧串 "switched too recently" 与旧建议 "continue, don't retry"，而模型是**先读这张表
    再看回文**的：表告诉它"这多半说明上次那次切换是对的，别重试"，回文说的却是"没切成"。
    """
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[3]
    txt = (root / "config" / "system" / "TOOLS.md").read_text(encoding="utf-8")
    assert "NOT switched (debounce)" in txt, "对照表没有当前回文的那一行"
    for stale in ("switched too recently", "You switched moments ago",
                  "continue, don't retry"):
        assert stale not in txt, f"TOOLS.md 仍带修前措辞：{stale!r}"
