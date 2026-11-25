"""批 L110 · MQ-L111：Hermes 写进用户消息的 `[System note: …]` 不算用户话（原则 13 层 4）。

注记的字面量取自 Hermes `gateway/run.py` 的模板，位置两种：重置注记挂在用户话之后，
中断 / 自动续轮注记在用户话之前，后者可能整条都是注记。
"""

from __future__ import annotations

import pytest
from bladex_core.ledger_runtime import candidate_units
from bladex_proxy.agency import _clean_user_text, _last_user_text

_WORDS = "全面分析一下这个市场的切入机会"
_INACTIVE = ("[System note: The user's previous session expired due to inactivity. "
             "This is a fresh conversation with no prior context.]")
_DAILY = ("[System note: The user's session was automatically reset by the daily schedule. "
          "This is a fresh conversation with no prior context.]")
_SUSPENDED = ("[System note: The user's previous session was stopped and suspended. "
              "This is a fresh conversation with no prior context.]")
_NEW_MESSAGE = ("[System note: A new message has arrived. The conversation history contains "
                "pending tool outputs from an interrupted turn. IGNORE those pending results. "
                "Address the user's NEW message below FIRST. "
                "Do NOT re-execute old tool calls from the history.]")
_CONTINUE = "[System note: Your previous turn was interrupted before it finished.]"


def _user(text: str) -> dict:
    return {"role": "user", "content": text}


@pytest.mark.parametrize("note", [_INACTIVE, _DAILY, _SUSPENDED])
def test_reset_note_after_the_words_is_stripped(note: str) -> None:
    got = _last_user_text([_user(f"{_WORDS}\n\n{note}")], for_goal=True)
    assert got == _WORDS
    assert "[System note:" not in got


def test_note_before_the_words_is_stripped() -> None:
    got = _last_user_text([_user(f"{_NEW_MESSAGE}\n\n{_WORDS}")], for_goal=True)
    assert got == _WORDS


def test_notes_on_both_ends_are_stripped() -> None:
    got = _last_user_text([_user(f"{_NEW_MESSAGE}\n\n{_WORDS}\n\n{_DAILY}")], for_goal=True)
    assert got == _WORDS


def test_note_only_message_walks_back_to_the_real_words() -> None:
    """整条都是注记 ⇒ 这条里没有用户说的话，与 MQ-L78 的往前找同一条路。"""
    msgs = [_user(_WORDS), {"role": "assistant", "content": "好"}, _user(_CONTINUE)]
    assert _clean_user_text(_CONTINUE) == ("", ["system_note"])
    assert _last_user_text(msgs, for_goal=True) == _WORDS


def test_query_units_are_the_same_as_without_the_note() -> None:
    """L109 读数：带注记时 `query_units` 多出 15–16 个，修后应与不带注记相等。"""
    raw = f"{_WORDS}\n\n{_INACTIVE}"
    assert len(candidate_units(raw)) > len(candidate_units(_WORDS))
    assert (candidate_units(_last_user_text([_user(raw)], for_goal=True))
            == candidate_units(_WORDS))


@pytest.mark.parametrize("text", [
    "这段注记 [System note: The user's previous session expired] 是谁写的？",
    "看一下这个字段：[System note:] 为什么没被剥",
    "[System note: 这是开头的引用] 后面紧跟着我的问题，没有空行",
])
def test_quoted_note_inside_the_words_is_kept(text: str) -> None:
    """🔴 只剥两端独立成段的注记：用户讨论这句话时原文不动（MQ-A19 自指陷阱）。"""
    assert _last_user_text([_user(text)], for_goal=True) == text


def test_text_without_note_is_untouched() -> None:
    assert _clean_user_text(_WORDS) == (_WORDS, [])


def test_non_goal_path_keeps_the_raw_text() -> None:
    """`for_goal=False` 是原文通道，不经清洗。"""
    raw = f"{_WORDS}\n\n{_DAILY}"
    assert _last_user_text([_user(raw)]) == raw


def test_stripping_has_a_single_implementation_point() -> None:
    """🔴 注记只在 `_clean_user_text` 这一条路上剥，agency 包里不许另写一份。"""
    from tests._source_probe import package_source

    src = package_source("agency")
    assert src.count("def _strip_system_notes(") == 1
    assert src.count("_strip_system_notes(") == 2, "一个定义 + `_clean_user_text` 一个调用点"
    assert src.count(r"\[System note:") == 2, "正则形态只在两条模式里"
    assert src.count('"[System note') + src.count("'[System note") == 0, "不许有字符串形态的第二份判据"
