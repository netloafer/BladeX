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
`ledger_runtime.user_turn_index`），防抖只在同 session 内判。

**批 Y · Y1c（2026-09-24，MQ-L91）：轮数防抖已删**（改为举证，见 `test_switch_evidence.py`）。
① ② 两件（单实现点 / 缺省派生）原样保留——`turn_index` 仍进 `LEDGER_SWITCH` 载荷与
`agency_switch_evidence` 观测；③ 起的防抖剧本改判为"全部落定"，判别力对照改钉在载荷的
`turn_index` 上（口径改回 `len//2` ⇒ 工具循环用例的载荷值必错）。
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


class _Rec:
    def __init__(self):
        self.switches: list[dict] = []

    def __call__(self, etype, payload):
        if getattr(etype, "value", str(etype)) == "ledger_switch":
            self.switches.append(payload)


def _runtime():
    ag = AgencyRuntime()
    rec = _Rec()
    ag._emit = rec   # noqa: SLF001
    _seed(ag)
    return ag, rec


def test_rapid_switches_all_land_mq_l40_shape_gone(monkeypatch):
    """Y1c：隔 1 轮（原防抖拒）与隔 6 轮都落定；MQ-L40 那种"只有进程内第一次能落定"
    的形态在结构上不再可能——没有任何按轮数拒绝的分支。"""
    _on(monkeypatch)
    ag, rec = _runtime()
    scope = ag.scope_of("hermes:default")
    with structlog.testing.capture_logs() as cap:
        _process(ag, "ldg-a", n_turns=2)          # 用户轮 3
        _process(ag, "ldg-b", n_turns=3)          # 用户轮 4：原防抖在这里拒
        assert ag.activation.active(scope) == "ldg-b"
        _process(ag, "ldg-a", n_turns=8)          # 用户轮 9
        assert ag.activation.active(scope) == "ldg-a"
    assert [p["turn_index"] for p in rec.switches] == [3, 4, 9]
    assert not [e for e in cap if "debounce" in str(e.get("event", ""))]


def test_tool_loop_payload_turn_index_is_user_turn(monkeypatch):
    """E0.2 / MQ-L48 口径保留在载荷上：同一用户轮内跑 6 个工具调用再切，
    载荷 `turn_index` 仍是用户轮（3），`agency_switch_evidence` 标本轮第 2 次切换。"""
    _on(monkeypatch)
    ag, rec = _runtime()
    _process(ag, "ldg-a", n_turns=2)                                   # 用户轮 3
    with structlog.testing.capture_logs() as cap:
        _process(ag, "ldg-b", n_turns=2, messages=_msgs_tool_loop(2, 6))   # 仍是用户轮 3
    assert rec.switches[-1]["turn_index"] == 3
    row = [e for e in cap if e.get("event") == "agency_switch_evidence"][-1]
    assert row["switch_in_turn"] == 2 and row["delta_turns"] == 0


def test_old_request_count_formula_changes_payload(monkeypatch):
    """判别力对照：口径改回 `len//2`（请求数）⇒ 上一条的载荷值必错（工具往返被算成轮）。"""
    _on(monkeypatch)
    import bladex_core.ledger_runtime as lr
    monkeypatch.setattr(lr, "user_turn_index", lambda msgs: max(0, len(msgs or []) // 2))
    ag, rec = _runtime()
    _process(ag, "ldg-a", n_turns=2)
    _process(ag, "ldg-b", n_turns=2, messages=_msgs_tool_loop(2, 6))
    assert rec.switches[-1]["turn_index"] != 3, \
        "请求数口径下工具循环本该改变 turn_index——没变说明载荷不再依赖 user_turn_index"


def test_switch_reply_is_plain_success(monkeypatch):
    """C6c 的否定态回文随防抖一起删除：落定的切换只有一种回文。"""
    _on(monkeypatch)
    ag, _rec = _runtime()
    _process(ag, "ldg-a", n_turns=2)
    _msg, transcript, _mode = _process(ag, "ldg-b", n_turns=3)
    tool_texts = " ".join(str(m.get("content", "")) for m in transcript if m.get("role") == "tool")
    assert "Active ledger is now ldg-b." in tool_texts
    assert "NOT switched" not in tool_texts


# ── MQ-L44：跨会话照常落定 ────────────────────────────────────────────────

def test_new_session_lower_turn_index_can_still_switch(monkeypatch):
    """🔴 MQ-L44：新会话的 `turn_index` 从头算。防抖删除后这条是恒真，留作回归。"""
    _on(monkeypatch)
    ag, _rec = _runtime()
    scope = ag.scope_of("hermes:default")
    _process(ag, "ldg-a", n_turns=17, session_id="tw:local:Pi:1")        # 用户轮 18 落定
    assert ag.activation.active(scope) == "ldg-a"
    _process(ag, "ldg-b", n_turns=1, session_id="tw:local:Pi:2")         # 新会话：用户轮 2 < 18
    assert ag.activation.active(scope) == "ldg-b"
    _process(ag, "ldg-a", n_turns=2, session_id="tw:local:Pi:3")
    assert ag.activation.active(scope) == "ldg-a"


def test_tools_md_has_no_debounce_row():
    """🔴 模型面文档与运行时对账（MQ-L42 同族）：回文删了，对照表那一行也必须删——
    给模型看一条永远不会出现的回文说明，是在教它防一个不存在的拒绝。"""
    root = pathlib.Path(__file__).resolve().parents[3]
    txt = (root / "config" / "system" / "TOOLS.md").read_text(encoding="utf-8")
    for stale in ("NOT switched", "debounce", "switched too recently",
                  "continue, don't retry"):
        assert stale not in txt, f"TOOLS.md 仍带已删机制的措辞：{stale!r}"
