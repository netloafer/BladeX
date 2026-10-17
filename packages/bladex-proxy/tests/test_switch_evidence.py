"""批 Y · Y1c（MQ-L91，2026-09-24）· 账本切换：轮数防抖改为举证（观测阶段，只记不拦）。

卡 `docs/planning/task-ledger-switch-debounce-rethink-20260924.md` §5 步骤 1 / §6 判据 1–7，
提示词 `prompt-batch-Y1c-20260924.md` §3 判据 9–14。本文件钉：

- 判据 1  §1 病例剧本回放（夹具 = 病例**形状**，用户原话与账本标题已占位化，结构逐字同病例）：隔 1 轮当轮落定、带 evidence。
- 判据 2  子账本回写：建子本后 ≤3 轮切回父本 → 落定，来源 `child_writeback`。
- 判据 3  A→B→A（同一轮内三次）：全部落定，`agency_switch_evidence` 标出回切与本轮第几次。
- 判据 4  编造的 evidence ⇒ `evidence_ok=False` 且可 grep；真实摘录 ⇒ True（阴性对照）。
- 判据 5  取值与 `user_query` 同源：evidence 出现在模型参数里、不在用户原话里 ⇒ False；源码守卫。
- 判据 6  反向变异：把任一轮数拒绝加回 `request_switch` ⇒ 剧本 1–3 红（本文件用补丁实测）。
- 判据 9  重启后不从 Hub 恢复任何轮位基点。
- 判据 12 `LEDGER_SWITCH` 载荷带 evidence 三字段，经真 Hub 落盘重读仍在；老事件缺省不报错。
- 判据 13 不带 evidence 照常落定（`evidence_ok=None`）；schema 里 evidence 不是必填。
- 判据 14 三协议流式 + 非流式内循环四条路：`user_query` 非空，evidence 校验为 True。
"""
from __future__ import annotations

import ast
import asyncio
import json
import pathlib

import pytest
import structlog.testing
from bladex_core.ledger import new_ledger
from bladex_core.ledger_runtime import ActivationTable, SwitchVerdict
from bladex_proxy.agency import (
    AgencyRuntime,
    intercept_anthropic_stream,
    intercept_chat_stream,
    intercept_responses_stream,
    tool_context,
)

_AGENT = "hermes:default"
_FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "y1c_feishu_case_turn.json"
_LEDGER_A = "ldg-0a0a0a0a0a01"       # §1：当时激活的那本（占位标题「示例项目 A 调研」）
_LEDGER_B = "ldg-0b0b0b0b0b02"       # §1：目标那本（占位标题「网关断连排查（系统升级后）」）
#: 模型该摘的那段（取自夹具里占位化后的用户话；fixture 载入时断言它确在原话里）。
_CASE_EVIDENCE = "排查一下最近网关服务的连接情况"


@pytest.fixture(autouse=True)
def _modules_on(monkeypatch):
    for m in ("TOOLFACE", "LEDGER", "INTERCEPTION"):
        monkeypatch.setenv(f"BLADEX_MODULE_{m}", "1")
    monkeypatch.setenv("BLADEX_STREAM_KEEPALIVE_S", "0")
    monkeypatch.setenv("BLADEX_NARRATE_INTERCEPT", "0")


# ── 公共 ─────────────────────────────────────────────────────────────────


def _case_user_messages() -> list[dict]:
    """§1 病例 16:27:37 那一轮在 Hub 里的原始 user 消息（逐字，未剥信封）。"""
    data = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    msgs = [m for m in data["messages"] if m.get("role") == "user"]
    assert msgs, "fixture 里没有 user 消息"
    return msgs


class _Rec:
    """收 `_emit` 的事件（不接真 Hub 的用例用）。"""

    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    def __call__(self, etype, payload):
        self.events.append((getattr(etype, "value", str(etype)), payload))

    def switches(self) -> list[dict]:
        return [p for t, p in self.events if t == "ledger_switch"]


def _runtime() -> tuple[AgencyRuntime, _Rec]:
    ag = AgencyRuntime()
    rec = _Rec()
    ag._emit = rec          # noqa: SLF001 —— 只替换事件出口，判定路径原样
    for lid, title in ((_LEDGER_A, "示例项目 A 调研"),
                       (_LEDGER_B, "网关断连排查（系统升级后）"),
                       ("ldg-aaaa00000001", "任务 A"), ("ldg-bbbb00000002", "任务 B")):
        ag.pool[lid] = new_ledger(title=title, goal=title, ledger_id=lid,
                                  created_at="2026-09-20T00:00:00+00:00")
    return ag, rec


def _call(args: dict, cid: str = "c1") -> dict:
    return {"id": cid, "type": "function",
            "function": {"name": "bladex_ledger_switch", "arguments": json.dumps(args)}}


async def _final(*_a, **_k):
    return {"role": "assistant", "content": "ok"}


def _nonstream(ag: AgencyRuntime, calls: list[dict], upstream: list[dict], *,
               turn_index: int, session_id: str = "fp:1a2b2c60dc79"):
    msg = {"role": "assistant", "content": None, "tool_calls": calls}
    return asyncio.run(ag.process_message(
        msg, upstream_messages=upstream, session_prefix="p/", allowed_exposure="public",
        call_llm=_final, session_id=session_id, agent_id=_AGENT, turn_index=turn_index))


def _plain(user_text: str) -> list[dict]:
    return [{"role": "system", "content": "sys"}, {"role": "user", "content": user_text}]


def _evidence_rows(cap) -> list[dict]:
    return [e for e in cap if e.get("event") == "agency_switch_evidence"]


# ── 判据 1：§1 病例剧本回放 ───────────────────────────────────────────────


def test_case_fixture_is_real_and_contains_the_evidence():
    """fixture 取自 Hub（不是手写的句子）；剥信封后的 user_query 含病例那句。"""
    data = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    assert "Hub" in data.get("source", "") and data.get("turn_key")
    ctx = tool_context(session_id="s", agent_id=_AGENT, project_id="",
                       upstream_messages=[{"role": "system", "content": "sys"},
                                          *_case_user_messages()])
    assert _CASE_EVIDENCE in ctx["user_query"], ctx["user_query"][:200]


def test_case_replay_switch_lands_in_the_same_turn():
    ag, rec = _runtime()
    scope = ag.scope_of(_AGENT)
    _nonstream(ag, [_call({"ledger_id": _LEDGER_A})], _plain("确定账本更新了？"), turn_index=7)
    assert ag.activation.active(scope) == _LEDGER_A
    upstream = [{"role": "system", "content": "sys"}, *_case_user_messages()]
    with structlog.testing.capture_logs() as cap:
        _nonstream(ag, [_call({"ledger_id": _LEDGER_B, "evidence": _CASE_EVIDENCE})],
                   upstream, turn_index=8)
    assert ag.activation.active(scope) == _LEDGER_B, "Δ=1 的正当换题当轮没落定 = 防抖复活"
    row = _evidence_rows(cap)[-1]
    assert row["evidence_ok"] is True and row["source"] == "user_text"
    assert row["delta_turns"] == 1 and row["to_ledger"] == _LEDGER_B
    sw = rec.switches()[-1]
    assert (sw["evidence"], sw["evidence_ok"], sw["evidence_source"]) == \
        (_CASE_EVIDENCE, True, "user_text")
    assert not [e for e in cap if "debounce" in str(e.get("event", ""))]


# ── 判据 2：子账本回写 ─────────────────────────────────────────────────────


def test_child_writeback_to_parent_lands_and_is_sourced():
    ag, rec = _runtime()
    scope = ag.scope_of(_AGENT)
    _nonstream(ag, [_call({"ledger_id": "ldg-aaaa00000001"})], _plain("做 A"), turn_index=3)
    _nonstream(ag, [_call({"ledger_id": "", "title": "A 的子任务", "goal": "拆出来的子项",
                           "parent_ledger_id": "ldg-aaaa00000001"})],
               _plain("把子项单独拆一本"), turn_index=4)
    child = ag.activation.active(scope)
    assert child not in ("", "ldg-aaaa00000001")
    with structlog.testing.capture_logs() as cap:
        _nonstream(ag, [_call({"ledger_id": "ldg-aaaa00000001"})], _plain("好了继续"),
                   turn_index=5)
    assert ag.activation.active(scope) == "ldg-aaaa00000001", "子本回写父本被拦"
    row = _evidence_rows(cap)[-1]
    assert row["source"] == "child_writeback" and row["child_writeback"] is True
    assert row["evidence_ok"] is True
    sw = rec.switches()[-1]
    assert sw["evidence_source"] == "child_writeback" and "ldg-aaaa00000001" in sw["evidence"]
    # 建子本那一次：来源 new_ledger，证据 = 模型提炼的 Goal
    created = [p for p in rec.switches() if p["to_ledger_id"] == child][0]
    assert created["evidence_source"] == "new_ledger" and created["evidence"] == "拆出来的子项"


# ── 判据 3：A→B→A 同一轮 ──────────────────────────────────────────────────


def test_a_b_a_in_one_turn_all_land_and_are_visible():
    ag, _rec = _runtime()
    scope = ag.scope_of(_AGENT)
    _nonstream(ag, [_call({"ledger_id": "ldg-aaaa00000001"})], _plain("先做 A"), turn_index=2)
    with structlog.testing.capture_logs() as cap:
        _nonstream(ag, [_call({"ledger_id": "ldg-bbbb00000002"}, "c1"),
                        _call({"ledger_id": "ldg-aaaa00000001"}, "c2")],
                   _plain("B 顺手看一下，然后回 A"), turn_index=2)
    assert ag.activation.active(scope) == "ldg-aaaa00000001"
    rows = _evidence_rows(cap)
    assert [r["switch_in_turn"] for r in rows] == [2, 3]
    assert [r["back_to_prev"] for r in rows] == [False, True]
    assert all(r["delta_turns"] == 0 for r in rows)


def test_in_turn_switches_affect_anchor_from_switch_time_on():
    """锚层按轮次时刻查激活账本（`turn_active_ledger`）⇒ 一轮内的多次切换只影响
    **切换之后**的轮次：该轮开头（轮次 key 时刻）查到的仍是切换前那本。"""
    from bladex_core.ledger import switch_timeline
    from bladex_core.ledger_runtime import activation_scope, turn_active_ledger

    sc = activation_scope(_AGENT, "")
    t0 = 1_758_600_000_000.0
    events = [
        {"type": "ledger_switch", "ts": t0, "payload": {
            "agent_id": _AGENT, "project_id": "Global", "to_ledger_id": "ldg-a"}},
        # 同一轮（key 时刻 t0+60s）里 A→B→A：三次切换都在轮次 key 之后
        {"type": "ledger_switch", "ts": t0 + 61_000, "payload": {
            "agent_id": _AGENT, "project_id": "Global", "to_ledger_id": "ldg-b"}},
        {"type": "ledger_switch", "ts": t0 + 62_000, "payload": {
            "agent_id": _AGENT, "project_id": "Global", "to_ledger_id": "ldg-a"}},
    ]
    tl = switch_timeline(events)
    assert sc in tl
    this_turn = f"u/{_AGENT}/s/{int(t0 + 60_000)}"
    next_turn = f"u/{_AGENT}/s/{int(t0 + 120_000)}"
    got_this = turn_active_ledger(tl, this_turn, project_id="")
    got_next = turn_active_ledger(tl, next_turn, project_id="")
    assert got_this == "ldg-a" and got_next == "ldg-a"
    mid_turn = f"u/{_AGENT}/s/{int(t0 + 61_500)}"
    assert turn_active_ledger(tl, mid_turn, project_id="") == "ldg-b", \
        "点时查表：只有落在 A→B 与 B→A 之间的轮次才看得到 B"


# ── 判据 4：编造 vs 真实 ──────────────────────────────────────────────────


def test_fabricated_evidence_is_false_and_greppable_real_is_true():
    ag, rec = _runtime()
    _nonstream(ag, [_call({"ledger_id": "ldg-aaaa00000001"})], _plain("做 A"), turn_index=1)
    with structlog.testing.capture_logs() as cap:
        _nonstream(ag, [_call({"ledger_id": "ldg-bbbb00000002",
                               "evidence": "用户说切到任务 B"})],
                   _plain("嗯，那个问题再看看"), turn_index=2)
    assert ag.activation.active(ag.scope_of(_AGENT)) == "ldg-bbbb00000002", "观测阶段不拦"
    assert _evidence_rows(cap)[-1]["evidence_ok"] is False
    assert [e for e in cap if e.get("event") == "agency_switch_evidence_mismatch"]
    assert rec.switches()[-1]["evidence_ok"] is False
    assert rec.switches()[-1]["evidence"] == "用户说切到任务 B", "evidence 原文无损入 Hub"
    with structlog.testing.capture_logs() as cap:
        _nonstream(ag, [_call({"ledger_id": "ldg-aaaa00000001", "evidence": "回到 A"})],
                   _plain("好，回到 A 吧"), turn_index=3)
    assert _evidence_rows(cap)[-1]["evidence_ok"] is True
    assert not [e for e in cap if e.get("event") == "agency_switch_evidence_mismatch"]


# ── 判据 5：取值与 user_query 同源 ────────────────────────────────────────


def test_evidence_checked_against_user_query_not_model_args():
    """evidence 出现在模型自己的参数里（title/goal）、却不在本轮用户原话里 ⇒ False。
    把校验改成读模型参数（或读 context 以外任何东西）⇒ 本条红。"""
    ag, _rec = _runtime()
    _nonstream(ag, [_call({"ledger_id": "ldg-aaaa00000001"})], _plain("做 A"), turn_index=1)
    with structlog.testing.capture_logs() as cap:
        _nonstream(ag, [_call({"ledger_id": "ldg-bbbb00000002", "evidence": "切到任务 B",
                               "title": "切到任务 B", "goal": "切到任务 B"})],
                   # 上一轮 user 里说过，本轮没说 ⇒ 不是本轮证据
                   [{"role": "user", "content": "切到任务 B"},
                    {"role": "assistant", "content": "好"},
                    {"role": "user", "content": "继续"}], turn_index=2)
    assert _evidence_rows(cap)[-1]["evidence_ok"] is False


def test_handler_reads_user_text_from_the_single_point():
    import bladex_proxy.agency.handlers as h

    src = pathlib.Path(h.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.AsyncFunctionDef) and n.name == "_h_ledger_switch")
    body = ast.unparse(fn)
    # 两个判定都只吃 `_turn_user_text(context)` 的产物
    assert "_user_text = _turn_user_text(context)" in body
    assert "verify_switch_evidence(_evidence_raw, _user_text)" in body
    assert "_user_named_ledger(_user_text," in body
    assert 'context.get("user_query"' not in body, "切换 handler 里另起了一个取值点"
    helper = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "_turn_user_text")
    assert "context.get('user_query'" in ast.unparse(helper)


# ── 判据 6：反向变异（轮数拒绝加回 ⇒ 剧本 1–3 红）────────────────────────


def _mutant_debounce(monkeypatch):
    real = ActivationTable.request_switch
    last: dict = {}

    def mutant(self, scope, to, *, agent_id, turn_index, session_id="", project_id=""):
        prev = last.get(scope)
        if (self.active(scope) and to != self.active(scope) and prev is not None
                and prev[0] == session_id and 0 <= turn_index - prev[1] < 3):
            return SwitchVerdict(allowed=False, reason="noop")
        v = real(self, scope, to, agent_id=agent_id, turn_index=turn_index,
                 session_id=session_id, project_id=project_id)
        if v.allowed:
            last[scope] = (session_id, turn_index)
        return v

    monkeypatch.setattr(ActivationTable, "request_switch", mutant)


def test_mutation_reinstating_turn_debounce_breaks_scenarios(monkeypatch):
    _mutant_debounce(monkeypatch)
    # 剧本 1
    ag, _ = _runtime()
    _nonstream(ag, [_call({"ledger_id": _LEDGER_A})], _plain("x"), turn_index=7)
    _nonstream(ag, [_call({"ledger_id": _LEDGER_B, "evidence": _CASE_EVIDENCE})],
               [{"role": "system", "content": "sys"}, *_case_user_messages()], turn_index=8)
    assert ag.activation.active(ag.scope_of(_AGENT)) != _LEDGER_B
    # 剧本 3
    ag, _ = _runtime()
    _nonstream(ag, [_call({"ledger_id": "ldg-aaaa00000001"})], _plain("A"), turn_index=2)
    _nonstream(ag, [_call({"ledger_id": "ldg-bbbb00000002"})], _plain("B"), turn_index=2)
    assert ag.activation.active(ag.scope_of(_AGENT)) != "ldg-bbbb00000002"


# ── 判据 9：重启不恢复轮位基点 ────────────────────────────────────────────


class _Ev:
    def __init__(self, etype, payload, ts=None):
        self.event_type, self.payload, self.ts = etype, payload, ts


class _MiniHub:
    def __init__(self, events):
        self.events = list(events)

    def append_admin_event(self, etype, matter_id, **payload):
        self.events.append(_Ev(getattr(etype, "value", str(etype)), payload))

    def scan_admin_events(self):
        yield from (("k", e) for e in self.events)

    def scan_tombstones(self):
        return iter(())


def test_restart_does_not_restore_turn_baseline():
    led = new_ledger(title="老本", goal="g", ledger_id="ldg-old000000001",
                     created_at="2026-09-20T00:00:00+00:00")
    led2 = new_ledger(title="新本", goal="g", ledger_id="ldg-new000000002",
                      created_at="2026-09-20T00:00:00+00:00")
    hub = _MiniHub([
        _Ev("ledger_create", {"ledger": led.model_dump()}),
        _Ev("ledger_create", {"ledger": led2.model_dump()}),
        _Ev("ledger_switch", {"session_id": "sess", "agent_id": _AGENT, "project_id": "Global",
                              "from_ledger_id": "", "to_ledger_id": "ldg-old000000001",
                              "turn_index": 10}),
    ])
    ag = AgencyRuntime(hub=hub)
    scope = ag.scope_of(_AGENT)
    assert ag.activation.active(scope) == "ldg-old000000001"
    with structlog.testing.capture_logs() as cap:
        _nonstream(ag, [_call({"ledger_id": "ldg-new000000002"})], _plain("换新本"),
                   turn_index=11, session_id="sess")
    assert ag.activation.active(scope) == "ldg-new000000002"
    assert _evidence_rows(cap)[-1]["delta_turns"] is None, \
        "重启后拿 Hub turn_index 当基点相减 = 调度复核第 3 条的跨口径差值"


def test_load_pool_source_has_no_turn_restore():
    from _source_probe import package_source
    src = package_source("agency")
    assert "switch_turns" not in src.replace("原 `switch_turns=`", "")
    assert "binding_switch_turns" not in src


# ── 判据 12：Hub 载荷三字段，落盘重读仍在；老事件缺省不报错 ──────────────


def test_evidence_fields_persist_through_a_real_hub(tmp_path):
    from bladex_core.ledger import binding_sessions, replay_ledger_events, switch_timeline
    from bladex_proxy.ledger_events import load_ledger_events
    from bladex_proxy.storage.memory_hub import MemoryHub

    hub = MemoryHub(tmp_path / "hub")
    hub.open()
    try:
        # 老事件：没有 evidence 三字段（09-24 之前的全部存量）
        from bladex_proxy.models import AdminEventType
        hub.append_admin_event(AdminEventType.LEDGER_SWITCH, "", session_id="old", agent_id=_AGENT,
                               project_id="Global", from_ledger_id="",
                               to_ledger_id="ldg-aaaa00000001")
        ag = AgencyRuntime(hub=hub)
        for lid, title in (("ldg-aaaa00000001", "任务 A"), ("ldg-bbbb00000002", "任务 B")):
            ag.pool[lid] = new_ledger(title=title, goal=title, ledger_id=lid,
                                      created_at="2026-09-20T00:00:00+00:00")
        _nonstream(ag, [_call({"ledger_id": "ldg-bbbb00000002", "evidence": "任务 B"})],
                   _plain("现在去做任务 B"), turn_index=4, session_id="new")
    finally:
        hub.close()
    hub2 = MemoryHub(tmp_path / "hub")
    hub2.open()
    try:
        events = load_ledger_events(hub2)
        sw = [e["payload"] for e in events if e["type"] == "ledger_switch"]
        assert len(sw) == 2
        assert "evidence" not in sw[0] and "evidence_ok" not in sw[0]
        assert (sw[1]["evidence"], sw[1]["evidence_ok"], sw[1]["evidence_source"]) == \
            ("任务 B", True, "user_text")
        assert sw[1]["turn_index"] == 4, "turn_index 仍写（Hub 原始数据，原则 13）"
        # 老事件缺省：投影函数照常工作
        _pool, bindings = replay_ledger_events(events)
        assert binding_sessions(events) and switch_timeline(events) and bindings
    finally:
        hub2.close()


# ── 判据 13：不带 evidence 照常落定；schema 可选 ──────────────────────────


def test_switch_without_evidence_lands_with_none():
    ag, rec = _runtime()
    _nonstream(ag, [_call({"ledger_id": "ldg-aaaa00000001"})], _plain("A"), turn_index=1)
    with structlog.testing.capture_logs() as cap:
        _nonstream(ag, [_call({"ledger_id": "ldg-bbbb00000002"})], _plain("B"), turn_index=2)
    assert ag.activation.active(ag.scope_of(_AGENT)) == "ldg-bbbb00000002"
    assert _evidence_rows(cap)[-1]["evidence_ok"] is None
    assert _evidence_rows(cap)[-1]["source"] == "none"
    assert rec.switches()[-1]["evidence_ok"] is None


def test_schema_evidence_is_optional():
    from bladex_proxy.toolface import TOOL_SCHEMAS
    fn = next(t["function"] for t in TOOL_SCHEMAS
              if t["function"]["name"] == "bladex_ledger_switch")
    assert "evidence" in fn["parameters"]["properties"]
    assert "evidence" not in fn["parameters"].get("required", []), \
        "必填会让弱模型的工具调用直接失败 —— 一个新的误拦"


# ── 判据 14：三协议 + 内循环四条路 user_query 非空 ────────────────────────


class _Cap:
    full_text = ""
    reasoning_text = ""
    usage: dict = {}
    finish_reason = None

    def __init__(self):
        self.tool_events: list = []


def _sse(name, obj) -> bytes:
    head = f"event: {name}\n" if name else ""
    return (head + "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode()


_ARGS = json.dumps({"ledger_id": _LEDGER_B, "evidence": _CASE_EVIDENCE}, ensure_ascii=False)


async def _chat_src():
    base = {"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": "m"}
    yield "data: " + json.dumps({**base, "choices": [{"index": 0, "delta": {"tool_calls": [
        {"index": 0, "id": "cx", "type": "function",
         "function": {"name": "bladex_ledger_switch", "arguments": _ARGS}}]},
        "finish_reason": None}]}, ensure_ascii=False) + "\n\n"
    yield "data: " + json.dumps({**base, "choices": [{"index": 0, "delta": {},
                                                      "finish_reason": "tool_calls"}]}) + "\n\n"
    yield "data: [DONE]\n\n"


async def _anthropic_src():
    yield _sse("message_start", {"type": "message_start", "message": {
        "id": "m1", "type": "message", "role": "assistant", "model": "m", "content": [],
        "stop_reason": None, "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1}}})
    yield _sse("content_block_start", {"type": "content_block_start", "index": 0,
                                       "content_block": {"type": "tool_use", "id": "tx",
                                                         "name": "bladex_ledger_switch",
                                                         "input": {}}})
    yield _sse("content_block_delta", {"type": "content_block_delta", "index": 0,
                                       "delta": {"type": "input_json_delta",
                                                 "partial_json": _ARGS}})
    yield _sse("content_block_stop", {"type": "content_block_stop", "index": 0})
    yield _sse("message_delta", {"type": "message_delta", "delta": {"stop_reason": "tool_use"},
                                 "usage": {"output_tokens": 2}})
    yield _sse("message_stop", {"type": "message_stop"})


async def _responses_src():
    def resp(out, status):
        return {"id": "r1", "object": "response", "created_at": 0, "model": "m",
                "status": status, "output": out, "parallel_tool_calls": False,
                "tool_choice": "auto", "tools": []}
    item = {"type": "function_call", "id": "fc0", "call_id": "rx",
            "name": "bladex_ledger_switch", "arguments": _ARGS, "status": "completed"}
    yield _sse("response.created", {"type": "response.created",
                                    "response": resp([], "in_progress")})
    yield _sse("response.output_item.added", {"type": "response.output_item.added",
                                              "output_index": 0,
                                              "item": {**item, "arguments": "",
                                                       "status": "in_progress"}})
    yield _sse("response.function_call_arguments.delta", {
        "type": "response.function_call_arguments.delta", "item_id": "fc0",
        "output_index": 0, "delta": _ARGS})
    yield _sse("response.output_item.done", {"type": "response.output_item.done",
                                             "output_index": 0, "item": item})
    yield _sse("response.completed", {"type": "response.completed",
                                      "response": resp([item], "completed")})


def _case_text_raw() -> str:
    m = _case_user_messages()[-1]
    c = m["content"]
    return c if isinstance(c, str) else " ".join(b.get("text", "") for b in c)


def _upstream_for(proto: str, ag: AgencyRuntime) -> list[dict]:
    """按各协议入站的**真实归一路径**得到 upstream messages，再过注入面（账本块 + 自我介绍）。"""
    user = _case_text_raw()
    if proto == "anthropic":
        from bladex_proxy.anthropic import parse_anthropic_request
        msgs, _ = parse_anthropic_request({"model": "m", "max_tokens": 10, "system": "sys",
                                           "messages": [{"role": "user", "content": [
                                               {"type": "text", "text": user}]}]})
    elif proto == "responses":
        from bladex_proxy.responses import parse_responses_request
        msgs, _ = parse_responses_request({"model": "m", "instructions": "sys", "input": [
            {"type": "message", "role": "user",
             "content": [{"type": "input_text", "text": user}]}]})
    else:
        msgs = [{"role": "system", "content": "sys"}, *_case_user_messages()]
    msgs = ag.insert_ledger_block(msgs, _AGENT)
    return ag.insert_system_notes(msgs, toolface_injected=True)


def _drain(agen):
    async def run():
        async for _c in agen:
            pass
    asyncio.run(run())


@pytest.mark.parametrize("proto,fn,src", [
    ("chat", intercept_chat_stream, _chat_src),
    ("anthropic", intercept_anthropic_stream, _anthropic_src),
    ("responses", intercept_responses_stream, _responses_src),
])
def test_stream_paths_carry_user_query(proto, fn, src):
    ag, rec = _runtime()
    _nonstream(ag, [_call({"ledger_id": _LEDGER_A})], _plain("x"), turn_index=1,
               session_id="s")
    upstream = _upstream_for(proto, ag)
    assert tool_context(session_id="s", agent_id=_AGENT, project_id="",
                        upstream_messages=upstream)["user_query"], proto
    _drain(fn(src(), agency=ag, capture_result=_Cap(), session_prefix="p/",
              allowed_exposure="public", session_id="s", agent_id=_AGENT,
              call_llm=_final, upstream_messages=upstream))
    assert ag.activation.active(ag.scope_of(_AGENT)) == _LEDGER_B, proto
    assert rec.switches()[-1]["evidence_ok"] is True, \
        f"{proto}: user_query 断了（或被注入面污染）⇒ 真实摘录被判成编造"


def test_nonstream_inner_loop_carries_user_query():
    ag, rec = _runtime()
    _nonstream(ag, [_call({"ledger_id": _LEDGER_A})], _plain("x"), turn_index=1, session_id="s")
    upstream = _upstream_for("chat", ag)
    _msg, _transcript, mode = _nonstream(
        ag, [_call({"ledger_id": _LEDGER_B, "evidence": _CASE_EVIDENCE})], upstream,
        turn_index=2, session_id="s")
    assert mode == "pure_bladex", "纯 bladex 调用应走内循环"
    assert rec.switches()[-1]["evidence_ok"] is True


# ── 模型面文档两副本 ─────────────────────────────────────────────────────


def test_tools_md_both_copies_teach_evidence_and_drop_debounce():
    root = pathlib.Path(__file__).resolve().parents[3]
    for rel in ("config/system/TOOLS.md",
                "packages/bladex-proxy/bladex_proxy/assets/system/TOOLS.md"):
        txt = (root / rel).read_text(encoding="utf-8")
        assert "NOT switched" not in txt and "debounce" not in txt, rel
        assert "`evidence`" in txt, rel
