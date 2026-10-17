"""批 E · E4（MQ-L98，拍板 (38)(40)）：关本后返工能重开 + 关本举证。

🔴 **E4.1 自批 HO（MQ-L99，拍板 (43)，2026-10-15）起并入相关性段**：E4 原先单列的
「Ledgers you closed in this session recently」小节只对关它的那个会话列出，交接（换会话、
换 agent）必然看不见；L98 原病例（09-28 任务 2 返工原话）在相关性段上按内容命中原本
（0.48 / shared 17）⇒ 并成一条规则、删掉同会话分支。下面 E4.1 的判据按新规则改写：
① 返工病例形状——同会话关本 → 打回 → 相关性段里出现原本（标 closed）→ 切过去 = 自动 reopen、
rev 接续、不新建；② 别的会话 / CLI 关的本**按内容同样列出**（MQ-L99 推广；CLI 关的不带 "by"）；
③ 超过 24 h 不出现；④ 无已关命中时候选块**逐字不变**、与会话无关（三协议消息形状各一条）；
⑤ 注入面字符变化量（读数打印进断言信息）。交接病例与渲染判据见 `test_ho_closed_ledger_handoff.py`。
E4.2 判据：带 / 不带 / 编造 evidence 三种；Hub 载荷三字段；schema 不许变必填；取值走
`_turn_user_text` 单点（AST 守卫，与 `test_switch_evidence.py` 同款）；重放同源。
"""

from __future__ import annotations

import ast
import asyncio
import datetime as dt
import inspect
import json

import pytest
from bladex_core.ledger import (
    STATUS_ACTIVE,
    STATUS_CLOSED,
    close_ledger,
    new_ledger,
    reopen_ledger,
    replay_ledger_events,
)
from bladex_core.ledger_runtime import activation_scope
from bladex_proxy.agency import _MATCHING_LEDGERS_HEADER, AgencyRuntime
from bladex_proxy.toolface import TOOL_SCHEMAS

AGENT = "codex"
PROJECT = "p-10ad1f61d0c2"
SCOPE = activation_scope(AGENT, PROJECT)
TASK2 = "ldg-939ac7c8b024"
OTHER = "ldg-8f759e60ba72"
S_MAIN = "01a0e6eb-6bb1-72c3-800d-841fcb67bdf7"
S_OTHER = "01a0e6cf-ce78-7fd2-bc95-bd86a97bee64"
#: 09-28 21:31:44 的用户原话（Hub `local/codex/01a0e6eb…`，截短到前两条）。
REWORK_TEXT = ("任务 2 复核退回，已提交的 21bb68a 上补一个修正提交：\n"
               "1. 升级兼容：命令行没有 --bladex-home（旧版本拉起的进程）⇒ _process_home 返回 None，"
               "现在被判成外来并删 pidfile，导致 bladex stop 再也停不掉它、下次 start 再起一套。\n"
               "2. 读命令只读：bladex status 不得删除任何 pidfile；清理只在 start 里做。")
#: 任务 2 账本的真实 Goal（Hub `ldg-939ac7c8b024`）。
TASK2_GOAL = ("统一 `bladex start` 与 `status` 的运行判断：pidfile 里的 PID 必须活着且命令行属于当前 "
              "`BLADEX_HOME` 才算在跑；陈旧 pidfile 清除后照常启动；避免 `pgrep -f` 把其他 "
              "BLADEX_HOME 实例误判为本实例，并补齐对应测试。")
DONE_TEXT = "任务1已经完成，继续下一个任务。"


@pytest.fixture(autouse=True)
def _modules_on(monkeypatch):
    for m in ("TOOLFACE", "LEDGER", "INTERCEPTION"):
        monkeypatch.setenv(f"BLADEX_MODULE_{m}", "1")
    monkeypatch.setenv("BLADEX_LEDGER_TIERS", "strong,medium")


class _Rec:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    def __call__(self, etype, payload):
        self.events.append((getattr(etype, "value", str(etype)), payload))

    def of(self, etype: str) -> list[dict]:
        return [p for t, p in self.events if t == etype]


def _ctx(session: str = S_MAIN, user: str = DONE_TEXT, turn_index: int = 2) -> dict:
    return {"session_id": session, "agent_id": AGENT, "project_id": PROJECT,
            "turn_index": turn_index, "user_query": user}


def _runtime() -> tuple[AgencyRuntime, _Rec]:
    ag = AgencyRuntime()
    rec = _Rec()
    ag._emit = rec  # noqa: SLF001
    for lid, title, goal in (
            (TASK2, "MQ-A78：修复陈旧 pidfile 导致 start 误报 running", TASK2_GOAL),
            (OTHER, "MQ-I33 修复全量重建中止误报完成",
             "修复 full 模式全量重建在守卫中止时误报完成的问题")):
        ag.pool[lid] = new_ledger(ledger_id=lid, title=title, goal=goal,
                                  goal_source="model", created_at="2026-09-28T11:44:29+00:00")
    return ag, rec


def _run(coro):
    return asyncio.run(coro)


def _close(ag, args: dict, **ctx):
    return _run(ag._h_ledger_close(args, allowed_exposure="public", context=_ctx(**ctx)))


def _switch(ag, args: dict, **ctx):
    return _run(ag._h_ledger_switch(args, allowed_exposure="public", context=_ctx(**ctx)))


def _block(ag, messages: list[dict], *, session: str) -> str:
    out = ag.insert_ledger_block(messages, AGENT, project_id=PROJECT, tier="strong",
                                 with_instruction=True, session_id=session)
    return out[-1]["content"]


_USER_TURN = [{"role": "system", "content": "You are Codex, a coding agent."},
              {"role": "user", "content": REWORK_TEXT}]


# ── E4.1 ① 病例形状：关本 → 打回 → 看得见 → 切过去即重开 ─────────────────────
def _closed_row(content: str, lid: str) -> str:
    return next(ln for ln in content.splitlines() if ln.startswith(f"- {lid}:"))


def test_rework_case_sees_closed_ledger_and_switching_reopens_without_new_ledger():
    ag, rec = _runtime()
    _switch(ag, {"ledger_id": TASK2})
    ag.pool[TASK2] = ag.pool[TASK2].model_copy(update={"rev": 14})
    assert f"Ledger {TASK2} closed." in _close(ag, {"evidence": "任务1已经完成"})
    assert ag.pool[TASK2].status == STATUS_CLOSED
    assert ag.pool[TASK2].closed_by_session == S_MAIN and ag.pool[TASK2].closed_at
    assert ag.pool[TASK2].closed_by_agent == AGENT

    content = _block(ag, _USER_TURN, session=S_MAIN)
    assert _MATCHING_LEDGERS_HEADER in content
    assert "[closed by codex " in _closed_row(content, TASK2)
    assert "Ledgers you closed in this session recently" not in content, "E4 小节已并入相关性段"
    n_before = len(ag.pool)
    out = _switch(ag, {"ledger_id": TASK2, "evidence": "复核退回"}, user=REWORK_TEXT, turn_index=3)
    assert "switching reopened it" in out
    assert ag.pool[TASK2].status == STATUS_ACTIVE and ag.pool[TASK2].rev == 14
    assert ag.pool[TASK2].closed_by_session == "" and ag.pool[TASK2].closed_at == ""
    assert ag.pool[TASK2].closed_by_agent == ""
    assert ag.activation.active(SCOPE) == TASK2
    assert len(ag.pool) == n_before, "返工不该新建账本"
    assert len(rec.of("ledger_reopen")) == 1 and len(rec.of("ledger_create")) == 0


# ── E4.1 ② 别的会话 / CLI 关的本：MQ-L99 起按内容同样列出 ─────────────────────
def test_ledger_closed_by_another_session_or_cli_is_listed_by_content():
    ag, _rec = _runtime()
    _switch(ag, {"ledger_id": TASK2}, session=S_OTHER)
    _close(ag, {}, session=S_OTHER)                         # 别的会话关的
    content = _block(ag, _USER_TURN, session=S_MAIN)
    assert "[closed by codex " in _closed_row(content, TASK2)
    _switch(ag, {"ledger_id": TASK2}, session=S_OTHER, user=REWORK_TEXT, turn_index=3)  # 重开
    ag.lifecycle_close(TASK2, origin="cli", context=None)   # CLI 关的（无会话、无 agent）
    assert ag.pool[TASK2].closed_by_session == "" and ag.pool[TASK2].closed_by_agent == ""
    row = _closed_row(_block(ag, _USER_TURN, session=S_MAIN), TASK2)
    assert "[closed " in row and " by " not in row, row
    # 内容对不上的已关本不出现（OTHER 是 I33，与返工原话无关）
    ag.lifecycle_close(OTHER, origin="cli", context=None)
    assert OTHER not in _block(ag, _USER_TURN, session=S_MAIN)


# ── E4.1 ③ 超过 24 h 不出现 ───────────────────────────────────────────────
def test_closed_more_than_24h_ago_is_not_listed():
    ag, _rec = _runtime()
    _switch(ag, {"ledger_id": TASK2})
    _close(ag, {})
    old = (dt.datetime.now(dt.UTC) - dt.timedelta(hours=25)).isoformat(timespec="seconds")
    ag.pool[TASK2] = ag.pool[TASK2].model_copy(update={"closed_at": old})
    assert TASK2 not in _block(ag, _USER_TURN, session=S_MAIN)
    fresh = (dt.datetime.now(dt.UTC) - dt.timedelta(hours=23)).isoformat(timespec="seconds")
    ag.pool[TASK2] = ag.pool[TASK2].model_copy(update={"closed_at": fresh})
    assert TASK2 in _block(ag, _USER_TURN, session=S_MAIN)


# ── E4.1 ④ 无已关命中时逐字不变、与会话无关（三协议消息形状各一条）─────────────
_SHAPES = {
    "chat": [{"role": "system", "content": "You are Hermes Agent."},
             {"role": "user", "content": "继续做这个任务"}],
    "anthropic": [{"role": "system", "content": "You are Claude Code, Anthropic's official CLI."},
                  {"role": "user", "content": [{"type": "text", "text": "继续做这个任务"}]}],
    "responses": [{"role": "system", "content": "You are Codex, a coding agent."},
                  {"role": "developer", "content": "<permissions instructions>…"},
                  {"role": "user", "content": "继续做这个任务"}],
}


@pytest.mark.parametrize("shape", sorted(_SHAPES))
def test_block_is_byte_identical_when_no_closed_ledger_matches(shape):
    ag, _rec = _runtime()
    _switch(ag, {"ledger_id": OTHER})
    _close(ag, {})                                            # 刚关、但内容对不上
    msgs = _SHAPES[shape]
    with_session = _block(ag, msgs, session=S_MAIN)
    without_session = ag.insert_ledger_block(msgs, AGENT, project_id=PROJECT, tier="strong",
                                             with_instruction=True)[-1]["content"]
    assert with_session == without_session
    assert "[closed" not in with_session and OTHER not in with_session


# ── E4.1 ⑤ 读数：已关命中让候选段多出的字符 ───────────────────────────────────
def test_closed_match_chars_reading():
    ag, _rec = _runtime()
    _switch(ag, {"ledger_id": TASK2})
    _close(ag, {})
    _block(ag, _USER_TURN, session=S_MAIN)
    bd = ag.last_ledger_breakdown
    assert bd is not None and bd.chars.get("candidates", 0) > 0, bd.chars


# ── E4.2 关本举证：带 / 不带 / 编造 ───────────────────────────────────────
@pytest.mark.parametrize("evidence, expect_ok, expect_source", [
    ("任务1已经完成", True, "user_text"),
    ("", None, ""),
    ("用户明确说这件事作废了", False, "user_text"),
])
def test_close_evidence_is_recorded_not_enforced(evidence, expect_ok, expect_source):
    ag, rec = _runtime()
    _switch(ag, {"ledger_id": TASK2})
    out = _close(ag, {"evidence": evidence} if evidence else {})
    assert f"Ledger {TASK2} closed." in out, "只记不拦：三种都照常关"
    (ev,) = rec.of("ledger_close")
    assert ev["evidence"] == evidence and ev["evidence_ok"] is expect_ok
    assert ev["evidence_source"] == expect_source
    assert ev["session_id"] == S_MAIN


def test_close_schema_keeps_evidence_optional():
    close = next(t["function"] for t in TOOL_SCHEMAS if t["function"]["name"] == "bladex_ledger_close")
    assert "evidence" in close["parameters"]["properties"]
    assert "evidence" not in close["parameters"].get("required", []), "设必填 = 拦截"


def test_close_handler_reads_user_text_from_the_single_point():
    from bladex_proxy.agency import handlers
    src = inspect.getsource(handlers)
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.AsyncFunctionDef) and n.name == "_h_ledger_close")
    calls = {c.func.id for c in ast.walk(fn)
             if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
    assert {"_turn_user_text", "verify_switch_evidence"} <= calls


# ── 重放与运行时同源 ─────────────────────────────────────────────────────
def test_replay_restores_closed_by_session_and_closed_at_and_reopen_clears():
    led = new_ledger(ledger_id=TASK2, title="t", goal="g", goal_source="model",
                     created_at="2026-09-28T11:44:29+00:00")
    ts = dt.datetime(2026, 9, 28, 13, 21, 44, tzinfo=dt.UTC)
    events = [
        {"type": "ledger_create", "payload": {"ledger": json.loads(led.model_dump_json()),
                                              "agent_id": AGENT, "project_id": PROJECT}},
        {"type": "ledger_switch", "payload": {"session_id": S_MAIN, "agent_id": AGENT,
                                              "project_id": PROJECT, "from_ledger_id": "",
                                              "to_ledger_id": TASK2, "turn_index": 3}},
        {"type": "ledger_close", "payload": {"ledger_id": TASK2, "origin": "model",
                                             "session_id": S_MAIN, "agent_id": AGENT},
         "ts": ts},
    ]
    pool, bindings = replay_ledger_events(events)
    assert pool[TASK2].status == STATUS_CLOSED
    assert pool[TASK2].closed_by_session == S_MAIN
    assert pool[TASK2].closed_at == "2026-09-28T13:21:44+00:00"
    pool2, _ = replay_ledger_events(events + [{"type": "ledger_reopen",
                                               "payload": {"ledger_id": TASK2, "origin": "model"}}])
    assert pool2[TASK2].closed_by_session == "" and pool2[TASK2].closed_at == ""
    # CLI 关本（无 session_id）⇒ 空
    pool3, _ = replay_ledger_events(events[:2] + [{"type": "ledger_close",
                                                   "payload": {"ledger_id": TASK2, "origin": "cli"},
                                                   "ts": ts}])
    assert pool3[TASK2].closed_by_session == "" and pool3[TASK2].closed_at
    # 原语本身
    assert reopen_ledger(close_ledger(led, closed_by_session="s", closed_at="x")).closed_at == ""
