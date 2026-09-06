"""MQ-L41 · `request_switch(user_requested=True)` 防抖豁免无生产者（批 E E0.3，2026-09-06）。

`ledger_runtime.request_switch` 自 08-25 起就有 `user_requested` 豁免（"防抖是防模型抖动的，
用户明说切到那本时拦下来等于跟用户对着干"），但 toolface / agency 零引用——字段在、生产者断线
（MQ-A20 形态）。修：`agency._h_ledger_switch` 从**本轮用户原话**（`context["user_query"]`，已剥
信封）派生：点名目标账本的 id、或 ≥6 字符的 title ⇒ 豁免。**不接受模型自报**（schema 无该参数）。

钉：① 点名 id ⇒ 隔 1 轮仍放行 ② 点名 title（≥6 字符）⇒ 放行 ③ 未点名 ⇒ 拒
    ④ 模型 args 塞 `user_requested=True` 无效 ⑤ title < 6 字符不算点名 ⑥ 命中打 info 日志带 by=
    ⑦ TOOLS.md 两份副本都写了这条豁免
"""
from __future__ import annotations

import asyncio
import json
import pathlib

import structlog.testing

from bladex_core.ledger import new_ledger
from bladex_proxy.agency import USER_NAMED_TITLE_MIN_CHARS, AgencyRuntime, _user_named_ledger

_SYS = {"role": "system", "content": "You are a test agent."}
_AGENT = "hermes:default"


def _on(monkeypatch):
    for m in ("TOOLFACE", "LEDGER", "INTERCEPTION"):
        monkeypatch.setenv(f"BLADEX_MODULE_{m}", "1")


def _msgs(n_turns: int, last_user: str) -> list[dict]:
    out = [_SYS]
    for i in range(n_turns):
        out += [{"role": "user", "content": f"u{i}"}, {"role": "assistant", "content": f"a{i}"}]
    out.append({"role": "user", "content": last_user})
    return out


def _switch_msg(lid: str, **extra) -> dict:
    return {"role": "assistant", "content": None,
            "tool_calls": [{"id": "c1", "type": "function",
                            "function": {"name": "bladex_ledger_switch",
                                         "arguments": json.dumps({"ledger_id": lid, **extra})}}]}


async def _final(*_a, **_k):
    return {"role": "assistant", "content": "ok"}


def _process(ag: AgencyRuntime, lid: str, n_turns: int, last_user: str, **extra):
    return asyncio.run(ag.process_message(
        _switch_msg(lid, **extra), upstream_messages=_msgs(n_turns, last_user),
        session_prefix="p/", allowed_exposure="public", call_llm=_final,
        session_id="s1", agent_id=_AGENT))


def _seed(ag: AgencyRuntime):
    ag.pool["ldg-aaaa11112222"] = new_ledger(title="路由回归修复", goal="修路由回归",
                                             ledger_id="ldg-aaaa11112222",
                                             created_at="2026-09-01T00:00:00+00:00")
    ag.pool["ldg-bbbb33334444"] = new_ledger(title="freetoken 项目调研", goal="调研 freetoken",
                                             ledger_id="ldg-bbbb33334444",
                                             created_at="2026-09-01T00:00:00+00:00")
    ag.pool["ldg-cccc55556666"] = new_ledger(title="调研", goal="短标题的那本",
                                             ledger_id="ldg-cccc55556666",
                                             created_at="2026-09-01T00:00:00+00:00")


def _bind_a_then_try(monkeypatch, target: str, last_user: str, **extra):
    """先切到 A（用户轮 3 落定），下一用户轮（Δ=1 < 3）尝试切 target；返回 (runtime, scope, logs)。"""
    _on(monkeypatch)
    ag = AgencyRuntime()
    _seed(ag)
    scope = ag.scope_of(_AGENT)
    _process(ag, "ldg-aaaa11112222", 2, "先干路由")
    assert ag.activation.active(scope) == "ldg-aaaa11112222"
    with structlog.testing.capture_logs() as cap:
        _process(ag, target, 3, last_user, **extra)
    return ag, scope, cap


# ── ① 点名 id ──────────────────────────────────────────────────────────────

def test_naming_the_id_bypasses_debounce(monkeypatch):
    ag, scope, cap = _bind_a_then_try(monkeypatch, "ldg-bbbb33334444",
                                      "切到 ldg-bbbb33334444，先把调研结论给我")
    assert ag.activation.active(scope) == "ldg-bbbb33334444", "用户点名 id 仍被防抖 = MQ-L41 未修"
    rows = [e for e in cap if e.get("event") == "agency_switch_user_requested"]
    assert len(rows) == 1 and rows[0]["by"] == "id" and rows[0]["ledger"] == "ldg-bbbb33334444"
    assert not [e for e in cap if e.get("event") == "agency_switch_debounced"]


# ── ② 点名 title ───────────────────────────────────────────────────────────

def test_naming_the_title_bypasses_debounce(monkeypatch):
    ag, scope, cap = _bind_a_then_try(monkeypatch, "ldg-bbbb33334444",
                                      "回到 freetoken 项目调研，现在结论是什么？")
    assert ag.activation.active(scope) == "ldg-bbbb33334444"
    rows = [e for e in cap if e.get("event") == "agency_switch_user_requested"]
    assert len(rows) == 1 and rows[0]["by"] == "title"


# ── ③ 未点名 ⇒ 拒（判别力对照：豁免不是恒真）─────────────────────────────

def test_not_naming_it_is_still_debounced(monkeypatch):
    ag, scope, cap = _bind_a_then_try(monkeypatch, "ldg-bbbb33334444",
                                      "那个东西的结论是什么？")
    assert ag.activation.active(scope) == "ldg-aaaa11112222", "没点名也放行 = 豁免恒真，防抖形同虚设"
    assert [e for e in cap if e.get("event") == "agency_switch_debounced"]
    assert not [e for e in cap if e.get("event") == "agency_switch_user_requested"]


# ── ④ 模型自报无效 ────────────────────────────────────────────────────────

def test_model_cannot_claim_user_requested(monkeypatch):
    ag, scope, cap = _bind_a_then_try(monkeypatch, "ldg-bbbb33334444",
                                      "那个东西的结论是什么？", user_requested=True)
    assert ag.activation.active(scope) == "ldg-aaaa11112222", \
        "模型在 args 里塞 user_requested=True 就能绕防抖 —— 判据来源必须是用户原话"
    assert not [e for e in cap if e.get("event") == "agency_switch_user_requested"]


# ── ⑤ 短 title 不算点名 ───────────────────────────────────────────────────

def test_short_title_does_not_count_as_naming(monkeypatch):
    ag, scope, cap = _bind_a_then_try(monkeypatch, "ldg-cccc55556666",
                                      "顺手帮我调研一下别的")
    assert ag.activation.active(scope) == "ldg-aaaa11112222", \
        "'调研' 两个字出现在用户话里就算点名 ⇒ 日常词全成豁免"
    assert not [e for e in cap if e.get("event") == "agency_switch_user_requested"]


def test_user_named_ledger_pure_function():
    led = new_ledger(title="freetoken 项目调研", ledger_id="ldg-bbbb33334444",
                     created_at="2026-09-01T00:00:00+00:00")
    assert _user_named_ledger("切到 ldg-bbbb33334444", led) == (True, "id")
    assert _user_named_ledger("freetoken 项目调研怎么样了", led) == (True, "title")
    assert _user_named_ledger("freetoken", led) == (False, "")      # 部分标题不算
    assert _user_named_ledger("", led) == (False, "")
    assert _user_named_ledger("随便", None) == (False, "")
    short = new_ledger(title="调研", ledger_id="ldg-cccc55556666", created_at="2026-09-01T00:00:00+00:00")
    assert len("调研") < USER_NAMED_TITLE_MIN_CHARS
    assert _user_named_ledger("帮我调研", short) == (False, "")
    assert _user_named_ledger("帮我看 ldg-cccc55556666", short) == (True, "id")   # 短 title 仍可按 id 点名


# ── ⑦ 模型面文档两份副本 ─────────────────────────────────────────────────

def test_tools_md_states_the_exemption_in_both_copies():
    root = pathlib.Path(__file__).resolve().parents[3]
    for rel in ("config/system/TOOLS.md",
                "packages/bladex-proxy/bladex_proxy/assets/system/TOOLS.md"):
        txt = (root / rel).read_text(encoding="utf-8")
        assert "If the user named the ledger (by id or title) in this turn, the switch is not debounced." in txt, rel
