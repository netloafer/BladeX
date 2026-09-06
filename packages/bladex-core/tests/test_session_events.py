"""U4B-B2：会话事件确定性检测（ADR-0026 铁律3）。

detect_session_events 是纯函数：高精度词表 + 同输入永远同输出（G6）。
"""

from __future__ import annotations

from bladex_core.session_events import detect_session_events


def test_detects_rework_event_with_detail():
    events = detect_session_events("上一轮任务卡被打回了，原因是测试没跑真实路径。")
    assert ("reworked" in dict(events))
    detail = dict(events)["reworked"]
    assert "打回" in detail and len(detail) <= 120


def test_detects_blocked_and_assigned():
    text = "现在卡在 LanceDB 版本不支持 FTS。这块先交给 codex 处理。"
    got = dict(detect_session_events(text))
    assert "blocked" in got and "assigned" in got
    assert "卡在" in got["blocked"]


def test_rejected_wins_over_accepted_in_same_sentence():
    got = dict(detect_session_events("T8b 验收不通过，打回重做。"))
    assert "rejected" in got
    assert "accepted" not in got


def test_accepted_detected():
    got = dict(detect_session_events("M-proxy-5 验收通过，下一步深化归属。"))
    assert "accepted" in got


def test_no_events_for_plain_questions():
    assert detect_session_events("今天世界杯赛程怎么样？") == []
    assert detect_session_events("帮我看下 proxy 502 是什么原因") == []
    assert detect_session_events("") == []


def test_generic_completion_words_do_not_fire():
    # 「完成/做好了」故意不收——高频泛化词会污染 lifecycle
    assert detect_session_events("这个功能已经完成了，做好了。") == []


def test_deterministic_same_input_same_output():
    text = "任务卡被打回；同时卡在 embedding 超时上。"
    assert detect_session_events(text) == detect_session_events(text)


def test_same_event_only_once():
    got = detect_session_events("打回了两次：第一次打回是测试问题，第二次打回是文档。")
    assert len([e for e, _ in got if e == "reworked"]) == 1


def test_english_patterns():
    got = dict(detect_session_events("The task card was reworked because CI was blocked on redis."))
    assert "reworked" in got and "blocked" in got
