"""批 Z · Z2.3（MQ-A34 ②）：BladeX 自己的注入面按预算裁剪。

守的判据：
④ 只裁 BladeX 注入块；硬规则永不裁（`NEVER_TRIMMED`，且不在 `TRIM_ORDER`）；每次裁剪有日志。
⑤ 工具与说明书成对装卸：任何计划 × 任何账本门下，说明书提到的 `bladex_*` ⊆ 注入的工具名。
⑦ 稳定层计划按会话冻结；动态层（账本块）按剩余预算逐轮裁。
反向变异（手工，结算稿记录）：把 "hard_rules" 放进 TRIM_ORDER ⇒ 本文件红；只裁说明书不裁工具 ⇒ ⑤ 红。
"""
from __future__ import annotations

import json
import re

import pytest
import structlog.testing
from bladex_proxy.surface_budget import (
    LAYER_DYNAMIC,
    LAYER_STABLE,
    LEDGER_BLOCK_TRIM_BUCKETS,
    NEVER_TRIMMED,
    TRIM_ORDER,
    StablePlan,
    StableSizes,
    dynamic_trim_step,
    estimate_tools_chars,
    plan_stable,
    surface_budget_for,
)

_TOOL_RE = re.compile(r"bladex_[a-z_]+")


# ── 顺序表本身 ──

def test_trim_order_is_unique_dynamic_first_and_never_touches_hard_rules():
    names = [s.name for s in TRIM_ORDER]
    assert len(names) == len(set(names))
    assert "hard_rules" in NEVER_TRIMMED
    assert not (set(names) & set(NEVER_TRIMMED)), "硬规则不许进可裁清单"
    layers = [s.layer for s in TRIM_ORDER]
    first_stable = layers.index(LAYER_STABLE)
    assert all(layer == LAYER_DYNAMIC for layer in layers[:first_stable])
    assert all(layer == LAYER_STABLE for layer in layers[first_stable:]), "动态层必须全部排在稳定层之前"
    assert all(s.why.strip() for s in TRIM_ORDER), "每一步都要写理由"
    assert names == ["candidates", "ledger_block", "ledger_family", "about", "memory_family"]


def test_ledger_family_step_covers_tools_and_manual_together():
    """账本族 = 工具 + 说明书账本段，是一步而不是两步（MQ-L7 / MQ-L77）。"""
    assert "ledger_tools" not in [s.name for s in TRIM_ORDER]
    assert "ledger_manual" not in [s.name for s in TRIM_ORDER]


# ── 预算与计划 ──

def test_budget_none_without_window_or_bad_ratio():
    assert surface_budget_for(context_window=0, agent_chars=1, hard_rules_chars=1) is None
    assert surface_budget_for(context_window=1000, agent_chars=1, hard_rules_chars=1,
                              chars_per_token=0) is None


def test_budget_arithmetic_uses_floor_and_ratio():
    b = surface_budget_for(context_window=10_000, agent_chars=5_000, hard_rules_chars=100,
                           chars_per_token=2.0, floor_tokens=2_000)
    assert b is not None and b.available_chars == (10_000 - 2_000) * 2 - 5_000 - 100


@pytest.mark.parametrize(("avail", "want_trimmed"), [
    (100_000, ()),
    (12_000, ("ledger_family",)),                       # 2.5k + 5.5k + 8.7k=16.7k > 12k；去账本族后 2.5k+3k
    (5_000, ("ledger_family", "about")),
    (1_000, ("ledger_family", "about", "memory_family")),
])
def test_plan_stable_is_greedy_in_trim_order(avail, want_trimmed):
    sizes = StableSizes(memory_tools=2_500, ledger_tools=5_500, about_full=8_700, about_no_ledger=3_000)
    b = surface_budget_for(context_window=1, agent_chars=0, hard_rules_chars=0, chars_per_token=1.0,
                           floor_tokens=0)
    # 直接构造想要的 available：窗口 1 token × 1 chars/token 太小，用 dataclass 替换
    from dataclasses import replace
    b = replace(b, context_window=avail)
    plan = plan_stable(b, sizes)
    assert plan.trimmed == want_trimmed
    assert plan.stable_chars <= avail or plan.trimmed == ("ledger_family", "about", "memory_family")


def test_plan_stable_without_budget_is_full():
    p = plan_stable(None, StableSizes(1, 1, 1, 1))
    assert p.is_full and p.trimmed == ()


def test_dynamic_trim_step_prefers_dropping_candidates():
    buckets = {"candidates": 300, "recent": 300, "goal": 100}
    assert dynamic_trim_step(9_000, buckets, None) is None
    assert dynamic_trim_step(9_000, buckets, 9_000) is None
    assert dynamic_trim_step(9_000, buckets, 8_500) == "candidates"
    assert dynamic_trim_step(9_000, buckets, 8_000) == "ledger_block"
    assert dynamic_trim_step(9_000, {}, 8_500) == "ledger_block", "没有候选段可去 ⇒ 整块"
    assert LEDGER_BLOCK_TRIM_BUCKETS == ("candidates", "recent")


def test_estimate_tools_chars_matches_probe_calibre():
    t = [{"type": "function", "function": {"name": "x", "parameters": {"a": "中文"}}}]
    assert estimate_tools_chars(t) == len(json.dumps(t[0], ensure_ascii=False))
    assert estimate_tools_chars(None) == 0


# ── 判据 ⑤：与真实 AgencyRuntime 对账 ──

def _runtime(monkeypatch):
    for n in ("TOOLFACE", "LEDGER"):
        monkeypatch.setenv(f"BLADEX_MODULE_{n}", "1")
    monkeypatch.delenv("BLADEX_LEDGER_TIERS", raising=False)
    from bladex_proxy.agency import AgencyRuntime
    ag = AgencyRuntime()
    assert ag.about, "config/system/*.md 没加载到 —— 本条会假绿"
    return ag


_PLANS = [
    StablePlan(),
    StablePlan(keep_ledger_family=False, trimmed=("ledger_family",)),
    StablePlan(keep_ledger_family=False, keep_about=False, trimmed=("ledger_family", "about")),
    StablePlan(keep_ledger_family=False, keep_about=False, keep_memory_family=False,
               trimmed=("ledger_family", "about", "memory_family")),
]


@pytest.mark.parametrize("plan", _PLANS, ids=lambda p: "+".join(p.trimmed) or "full")
@pytest.mark.parametrize("tier", ["medium", "weak"])
def test_manual_mentions_subset_of_injected_tools(monkeypatch, plan, tier):
    """⑤ 任何裁剪结果里「说明书提到的工具」⊆「工具列表」——MQ-L7 / MQ-L77 的不变式。"""
    ag = _runtime(monkeypatch)
    tools, injected = ag.augment_tools([], agent_id="probe", auxiliary=False, tier=tier,
                                       session_id="s", stable_plan=plan)
    names = {t["function"]["name"] for t in (tools or [])}
    out = ag.insert_system_notes([{"role": "system", "content": "agent"},
                                  {"role": "user", "content": "hi"}],
                                 toolface_injected=injected,
                                 ledger_face=bool(getattr(ag, "last_ledger_face", False)),
                                 session_id="s", stable_plan=plan)
    manual = [m["content"] for m in out if "<bladex-about>" in str(m.get("content", ""))]
    mentioned = set(_TOOL_RE.findall(manual[0])) if manual else set()
    assert mentioned <= names, f"说明书提到 {sorted(mentioned - names)} 但工具列表里没有（plan={plan.trimmed}, tier={tier}）"
    if not plan.keep_about:
        assert not manual, "计划裁掉自我介绍后不许再注"
    if not plan.keep_memory_family:
        assert not names, "记忆族被裁 ⇒ 整个工具面为空"
    if not plan.keep_ledger_family:
        assert not {n for n in names if n.startswith("bladex_ledger_")}


def test_plan_alone_keeps_manual_consistent_even_if_caller_passes_ledger_face_true(monkeypatch):
    """⑤ 的判别力：调用方把账本门传成 True（例如没同步计划的旧调用点），计划里账本族已裁
    ⇒ 说明书仍不许提账本工具。只在 `augment_tools` 里看计划、说明书不看 ⇒ 本条红。"""
    ag = _runtime(monkeypatch)
    plan = StablePlan(keep_ledger_family=False, trimmed=("ledger_family",))
    tools, injected = ag.augment_tools([], agent_id="probe", auxiliary=False, tier="medium",
                                       session_id="s", stable_plan=plan)
    names = {t["function"]["name"] for t in (tools or [])}
    assert injected and not any(n.startswith("bladex_ledger_") for n in names)
    out = ag.insert_system_notes([{"role": "user", "content": "hi"}], toolface_injected=True,
                                 ledger_face=True, session_id="s", stable_plan=plan)
    mentioned = set(_TOOL_RE.findall(out[0]["content"]))
    assert mentioned <= names, sorted(mentioned - names)


def test_ledger_face_off_never_leaks_ledger_mentions_even_with_full_plan(monkeypatch):
    """反向对照：账本门关（weak 档、无绑定）时说明书也不提账本工具（MQ-L77 原有行为保持）。"""
    ag = _runtime(monkeypatch)
    tools, injected = ag.augment_tools([], agent_id="probe", auxiliary=False, tier="weak",
                                       session_id="s")
    names = {t["function"]["name"] for t in (tools or [])}
    assert injected and not any(n.startswith("bladex_ledger_") for n in names)
    out = ag.insert_system_notes([{"role": "user", "content": "hi"}], toolface_injected=True,
                                 ledger_face=False, session_id="s")
    assert "bladex_ledger_" not in out[0]["content"]


# ── ⑦ 稳定层计划按会话冻结 ──

def test_stable_plan_is_frozen_per_session_and_logged(monkeypatch):
    ag = _runtime(monkeypatch)
    tiny = surface_budget_for(context_window=4_000, agent_chars=0, hard_rules_chars=0,
                              chars_per_token=1.0, floor_tokens=0)
    with structlog.testing.capture_logs() as cap:
        p1 = ag.stable_plan("sess-A", tiny)
    assert p1.trimmed, "4000 字符放不下 16K 的稳定层，必须裁"
    rows = [e for e in cap if e.get("event") == "agency_surface_trimmed"]
    assert [r["step"] for r in rows] == list(p1.trimmed)
    assert all(r["layer"] == "stable" and r["budget_chars"] == tiny.available_chars for r in rows)
    # 同会话再来：预算变大（甚至没窗口）也不改——定格
    assert ag.stable_plan("sess-A", None) is p1
    assert ag.stable_plan("sess-A", surface_budget_for(context_window=10**7, agent_chars=0,
                                                       hard_rules_chars=0)) is p1
    # 另一个会话独立
    assert ag.stable_plan("sess-B", None).is_full
    # 没有 session_id ⇒ 不冻结，按当轮预算
    assert ag.stable_plan("", None).is_full
    assert ag.stable_plan("", tiny).trimmed


def test_hard_rules_are_outside_the_plan_entirely():
    """④：计划里根本没有"硬规则"这个可裁对象（不是"默认保留"，是没有这一格）。"""
    assert not any("hard" in f for f in StablePlan.__dataclass_fields__)


# ── 动态层：账本块按剩余预算裁 ──

def _mk_ledger(ag, agent_id: str, *, goal: str, title: str = "t"):
    import asyncio
    asyncio.run(ag.toolface.dispatch(
        "bladex_ledger_switch", json.dumps({"ledger_id": "", "title": title}),
        allowed_exposure="public",
        context={"agent_id": agent_id, "user_query": goal, "turn_index": 0}))
    return ag.pool[ag.activation.active(ag.scope_of(agent_id))]


def test_ledger_block_trims_candidates_then_whole_block(monkeypatch):
    ag = _runtime(monkeypatch)
    _mk_ledger(ag, "probe", goal="修好路由", title="修好路由")
    _mk_ledger(ag, "probe", goal="梳理泰山啤酒破产重整", title="泰山啤酒")   # 第二本成为激活，第一本进 recency
    msgs = [{"role": "user", "content": "继续修路由"}]
    full = ag.insert_ledger_block(list(msgs), "probe", tier="medium", session_id="s")
    block = full[-1]["content"]
    assert "<bladex-ledger>" in block
    bd = ag.last_ledger_breakdown
    cand = sum(bd.chars.get(b, 0) for b in LEDGER_BLOCK_TRIM_BUCKETS)
    assert cand > 0, "夹具要先证明候选段存在（否则'去候选'那一步测不到）"
    # 预算刚好容不下候选段 ⇒ 去候选段
    with structlog.testing.capture_logs() as cap:
        out = ag.insert_ledger_block(list(msgs), "probe", tier="medium", session_id="s",
                                     budget_chars=len(block) - 1)
    assert "<bladex-ledger>" in out[-1]["content"] and len(out[-1]["content"]) < len(block)
    assert "修好路由" not in out[-1]["content"] or "Other recent ledgers" not in out[-1]["content"]
    rows = [e for e in cap if e.get("event") == "agency_surface_trimmed"]
    assert [r["step"] for r in rows] == ["candidates"] and rows[0]["chars"] == cand
    assert rows[0]["layer"] == "dynamic" and rows[0]["budget_chars"] == len(block) - 1
    assert ag.last_ledger_breakdown is not None
    # 预算连正文都容不下 ⇒ 整块不注，agent 消息逐字不变
    with structlog.testing.capture_logs() as cap:
        out2 = ag.insert_ledger_block(list(msgs), "probe", tier="medium", session_id="s",
                                      budget_chars=10)
    assert out2 == msgs
    rows = [e for e in cap if e.get("event") == "agency_surface_trimmed"]
    assert [r["step"] for r in rows] == ["ledger_block"]
    assert ag.last_ledger_breakdown is None, "没注块 ⇒ 读数是 None，不是上一轮的"
    # 没预算 ⇒ 原样
    same = ag.insert_ledger_block(list(msgs), "probe", tier="medium", session_id="s", budget_chars=None)
    assert same[-1]["content"] == block
