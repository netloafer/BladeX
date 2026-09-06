"""`fact_lane`（MQ-S27 硬分流的判据函数）。

钉住：① 逗号切开全串相等，**不是子串 in**（tags 是自由文本，子串判断一改措辞
静默失效——probe_memory 头注教训）；② PROFILE_OBS kind 视同 lane:profile
（历史数据可能只有 kind 没有 lane 标）；③ 未标 = 空串，消费者必须能处理未知。
"""

from __future__ import annotations

from bladex_core.fact import ItemKind, fact_lane


def test_exact_match_per_comma_segment() -> None:
    assert fact_lane("origin:user_direct,lane:profile") == "profile"
    assert fact_lane("lane:tool,lang:drift") == "tool"
    assert fact_lane("origin:conclusion,lane:task") == "task"


def test_substring_lookalikes_do_not_match() -> None:
    """子串陷阱：`xlane:profile` / `lane:profiles` 都不是 lane 标。"""
    assert fact_lane("xlane:profile") == ""
    assert fact_lane("lane:profiles") == ""
    assert fact_lane("plane:tool") == ""


def test_profile_obs_kind_counts_as_profile_lane() -> None:
    assert fact_lane("", ItemKind.PROFILE_OBS) == "profile"
    assert fact_lane("", "profile_obs") == "profile"
    # 有显式 lane 标时以标为准（tool 标 + PROFILE_OBS kind：标先查到）
    assert fact_lane("lane:tool", ItemKind.PROFILE_OBS) == "tool"


def test_unmarked_is_empty_not_task() -> None:
    """未标 ≠ task——把未知折叠成任何一条道都会让分流吞掉历史数据。"""
    assert fact_lane("") == ""
    assert fact_lane("origin:user_direct", ItemKind.ASSERTION) == ""
    assert fact_lane(None if False else "", "") == ""


def test_whitespace_tolerant() -> None:
    assert fact_lane(" origin:user_direct , lane:profile ") == "profile"
