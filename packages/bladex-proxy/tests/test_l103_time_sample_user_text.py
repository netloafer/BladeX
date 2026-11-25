"""MQ-L103 —— dsh 0.2.1 的三行时间采样 user 消息不许再成为「用户原话」。

dsh 在用户停顿后追加：

    Time sampled while preparing turn N, step M: …
    Browser time zone for this request: …
    Elapsed since the preceding model-visible message: …

MQ-L78 已解决「最后一条 user 剥空后继续往前找」；本批补的是清洗口径漏认这个
新机器消息形态。修法仍落在一处：`_MACHINE_ONLY_PREFIXES` + `_clean_user_text`。
"""

from __future__ import annotations

import asyncio

import pytest
import structlog
from bladex_proxy.agency import (
    AgencyRuntime,
    _clean_user_text,
    _last_user_text,
    tool_context,
)

USER_SENTENCE = "请派两个子代理并行干活，分别检查账本和 Hub 读数。"

TIME_SAMPLE = """Time sampled while preparing turn 4, step 1: 2026-10-27T20:52:35+08:00[Asia/Shanghai]
Browser time zone for this request: Asia/Shanghai. Interpret otherwise-unqualified dates and times in this zone.
Elapsed since the preceding model-visible message: 4m 30s."""

RUNTIME_CONTEXT = """Current runtime context. This snapshot supersedes earlier runtime-context snapshots.
Sandbox: workspace-write
Approval mode: default"""


def _dsh_messages() -> list[dict]:
    return [
        {"role": "user", "content": USER_SENTENCE},
        {"role": "user", "content": RUNTIME_CONTEXT},
        {"role": "user", "content": TIME_SAMPLE},
    ]


def test_l103_user_sentence_then_time_sample_returns_user_sentence() -> None:
    messages = [
        {"role": "user", "content": USER_SENTENCE},
        {"role": "user", "content": TIME_SAMPLE},
    ]
    assert _last_user_text(messages, for_goal=True) == USER_SENTENCE


def test_l103_user_sentence_runtime_context_and_time_sample_skip_both() -> None:
    assert _last_user_text(_dsh_messages(), for_goal=True) == USER_SENTENCE


def test_l103_time_sample_only_uses_all_envelopes_warning() -> None:
    messages = [
        {"role": "user", "content": "<system-reminder>old context</system-reminder>"},
        {"role": "user", "content": TIME_SAMPLE},
    ]
    with structlog.testing.capture_logs() as cap:
        got = _last_user_text(messages, for_goal=True)
    assert got == TIME_SAMPLE
    event = next(e for e in cap if e.get("event") == "agency_user_text_all_envelopes")
    assert event["users"] == 2
    assert "dsh_time_sample" in event["skipped"]


@pytest.mark.parametrize("sample", [TIME_SAMPLE, TIME_SAMPLE.lower()],
                         ids=["canonical_case", "lowercase_variant"])
def test_l103_machine_prefix_is_recorded_and_mid_sentence_text_survives(
        sample: str) -> None:
    cleaned, marks = _clean_user_text(sample)
    assert cleaned == ""
    assert marks[-1] == "dsh_time_sample"

    raw = ("dsh 为什么每轮都发 Time sampled while preparing turn 4 这种消息？"
           "这条不是用户原话。")
    cleaned, marks = _clean_user_text(raw)
    assert cleaned == raw
    assert "dsh_time_sample" not in marks


@pytest.fixture
def _ledger_modules_on(monkeypatch: pytest.MonkeyPatch) -> None:
    for module in ("TOOLFACE", "LEDGER", "INTERCEPTION"):
        monkeypatch.setenv(f"BLADEX_MODULE_{module}", "1")


def test_l103_ledger_switch_stores_real_user_sentence_as_goal_verbatim(
        monkeypatch: pytest.MonkeyPatch, _ledger_modules_on: None) -> None:
    runtime = AgencyRuntime(hub=None)
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(runtime, "_emit",
                        lambda event_type, payload: events.append(
                            (getattr(event_type, "value", str(event_type)), payload)))
    context = tool_context(session_id="sess-l103", agent_id="dsh",
                           project_id="p-l103test00001",
                           upstream_messages=_dsh_messages(), turn_index=0)
    result = asyncio.run(runtime._h_ledger_switch(
        {"title": "L103 时间采样修复", "goal": "修复 dsh 时间采样误判"},
        allowed_exposure="public", context=context))
    assert not result.startswith("Error"), result

    created = next(payload for kind, payload in events if kind == "ledger_create")
    verbatim = created["ledger"]["goal_verbatim"]
    assert verbatim == USER_SENTENCE
    assert "Time sampled" not in verbatim


def test_l103_recall_user_query_uses_real_user_sentence() -> None:
    context = tool_context(session_id="sess-l103", agent_id="dsh",
                           project_id="p-l103test00001",
                           upstream_messages=_dsh_messages(), turn_index=4)
    assert context["user_query"] == USER_SENTENCE
