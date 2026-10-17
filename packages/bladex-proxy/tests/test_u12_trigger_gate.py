"""批 U · U1.2 —— ADR-0033 §2.3 装配触发门（MQ-CA17 修订版）。

钉四件事：
① 冷 = **上一轮** `cache_read_tokens == 0`；首轮 / 无读数 / agent 自压缩按冷；上一轮 > 0 ⇒ 热。
   上一轮读数是 None（上游没报）⇒ 按冷，**不沿用更早的旧值**。
② **不看 TTL**：距上次转发多久与判定无关（反过来，TTL 判热也不能让门判热）。
③ 门 = 冷 ∧ 轮次数 > `BLADEX_ASSEMBLY_MIN_UNITS`（口径 = §2.2 轮次）。
   反向变异 2：把门改成"永远冷" ⇒ `test_warm_round_never_triggers` 红。
④ 读数唯一登记点 `capture.note_cache_reading`：三条流式 capture 路径各调一次（源码守卫）。

⑤（判据② 探针 `scripts/probe_cold_gate.py` 的纯函数守卫）已拆至 `test_probe_cold_gate.py`
—— 2026-10-15 混装分家：探针用例独立成文件，本文件的生产用例随公开树发。
"""

from __future__ import annotations

import inspect

import pytest
from bladex_proxy import capture, inject
from bladex_proxy.cache_state import SessionCacheRegistry, registry


@pytest.fixture(autouse=True)
def _clean_registry():
    registry().clear()
    yield
    registry().clear()


# ── ① ② cold_reason ────────────────────────────────────────────────────────


def test_cold_reason_follows_previous_round_reading():
    r = SessionCacheRegistry()
    assert r.cold_reason("s") == "no_reading", "首轮（没有会话状态）按冷"
    r.note_cache_read("s", 0)
    assert r.cold_reason("s") == "cache_read_zero"
    r.note_cache_read("s", 12_345)
    assert r.cold_reason("s") == ""
    assert r.cold_reason("s", prefix_changed=True) == "prefix_changed"
    r.note_cache_read("s", None)
    assert r.cold_reason("s") == "no_reading", "上游这轮没报 ⇒ 按冷，不沿用上一轮的 12345"


def test_cold_reason_ignores_ttl():
    r = SessionCacheRegistry()
    r.note_forward("s", "m", now=1.0)            # 距今很久：TTL 口径早已冷
    r.note_cache_read("s", 900)
    assert r.is_warm("s", "m", 300) is False
    assert r.cold_reason("s") == "", "TTL 冷不许让门判冷——门只看上游回报的读数"
    r2 = SessionCacheRegistry()
    r2.note_forward("s", "m")                    # 刚转发：TTL 口径热
    r2.note_cache_read("s", 0)
    assert r2.is_warm("s", "m", 300) is True
    assert r2.cold_reason("s") == "cache_read_zero", "TTL 热也不许让门判热"


# ── ③ 门 = 冷 ∧ 轮次数 > N ─────────────────────────────────────────────────


def test_trigger_needs_both_cold_and_enough_turns(monkeypatch):
    monkeypatch.setenv("BLADEX_ASSEMBLY_MIN_UNITS", "3")
    registry().note_cache_read("s", 0)
    t = inject._assembly_trigger("s", False, 4)
    assert t == {"triggered": True, "cold_reason": "cache_read_zero", "turns": 4, "min_units": 3}
    assert inject._assembly_trigger("s", False, 3)["triggered"] is False, "轮次数 = N 不触发（> N 才触发）"


def test_warm_round_never_triggers(monkeypatch):
    """反向变异 2 的判别用例：门改成"永远冷" ⇒ 这里红。"""
    monkeypatch.setenv("BLADEX_ASSEMBLY_MIN_UNITS", "3")
    registry().note_cache_read("s", 50_000)
    t = inject._assembly_trigger("s", False, 40)
    assert t["triggered"] is False and t["cold_reason"] == ""


def test_min_units_default_is_in_flags():
    from bladex_core.flags import MEMORY_NUMERIC_DEFAULTS

    assert MEMORY_NUMERIC_DEFAULTS["BLADEX_ASSEMBLY_MIN_UNITS"] == 10.0


def test_apply_cap_records_trigger_for_enabled_agent(monkeypatch):
    from bladex_proxy.assembly import AssemblyConfig, ContextAssembler

    monkeypatch.setenv("BLADEX_ASSEMBLY_AGENTS", "claude-code")
    monkeypatch.setenv("BLADEX_ASSEMBLY_MIN_UNITS", "1")
    msgs = [{"role": "system", "content": "s"},
            {"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"}]
    _out, info = inject._apply_cap(msgs, ContextAssembler(AssemblyConfig()), [], False,
                                   "sess", agent_id="claude-code")
    assert info["trigger"] == {"triggered": True, "cold_reason": "no_reading",
                               "turns": 2, "min_units": 1}


# ── ④ 唯一登记点 ────────────────────────────────────────────────────────────


def test_note_cache_reading_registers_by_bound_session():
    res = capture.CaptureResult()
    capture.bind_round(res, agent_id="a", model="m", session_id="sess-1", path="/v1/x", messages=[])
    capture.note_cache_reading(res, 0)
    assert res.cache_read_tokens == 0
    assert registry().cold_reason("sess-1") == "cache_read_zero"
    capture.note_cache_reading(res, 777)
    assert registry().cold_reason("sess-1") == ""


def test_three_capture_paths_call_the_single_registration_point():
    from bladex_proxy import anthropic, responses

    for mod, fn in ((capture, capture.capture_stream), (anthropic, anthropic.anthropic_stream_generator),
                    (responses, responses.responses_stream_generator)):
        src = inspect.getsource(fn)
        assert src.count("note_cache_reading(result, ") == 1, f"{mod.__name__}: 没有登记 cache 读数"
