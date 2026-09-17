"""G11.3 厂商前缀 header 回收 session 信号 + `x-{vendor}-user-id` 的否决守卫。

卡面（执行卡 G11.3）两半，落地时**只做了一半**，另一半被否并在这里钉死：

  ✅ `x-{vendor}-session-id` → `session_id_source=explicit`（通配，不认识的 vendor 同样生效）
  ❌ `x-{vendor}-user-id`  → **不接进 user_id**（未认证 header 不得跨越租户边界）

否决理由写在 `identity.resolve_identity` 的 user_id 段注释里，本文件负责让它**被违反
时会红**——注释会过期，测试不会。
"""

import pytest
from bladex_proxy.agent_bucket import (
    INFRA_VENDORS,
    RESERVED_VENDOR_SEGMENTS,
    VENDOR_SESSION_SUFFIXES,
    VENDOR_SUFFIXES,
    extract_vendor_ids,
    extract_vendor_segments,
)
from bladex_proxy.identity import (
    _MAX_EXTERNAL_SESSION_ID_CHARS,
    _screen_external_session_id,
    resolve_identity,
)
from bladex_proxy.models import ChatCompletionRequest, SessionIdSource

_BASE_HEADERS = {
    "authorization": "Bearer sk-test-g113",
    "user-agent": "deepseek-harness/0.4.0",
}


@pytest.fixture(autouse=True)
def _clean_identity_state():
    """本文件所有剧本用同一串 messages —— 不清缓存，尾部接续会把上一条剧本
    remember 过的 session 匹配回来，"退回推断"那几条就测不到自己想测的分支。"""
    from bladex_proxy.agent_registry import _agent_registry
    from bladex_proxy.identity import _session_cache

    _session_cache.clear()
    _agent_registry.clear()
    yield
    _session_cache.clear()
    _agent_registry.clear()


def _req() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        messages=[
            {"role": "user", "content": "G11.3 剧本用的一句话"},
            {"role": "assistant", "content": "好"},
        ]
    )


def _resolve(extra: dict[str, str]):
    identity, _src = resolve_identity({**_BASE_HEADERS, **extra}, _req())
    return identity


# ── ① 卡面判据：dsh 真实 header + 虚构 vendor ────────────────────────────────


def test_dsh_vendor_session_header_adopted():
    """卡面第一判据：dsh 的 `x-deepseek-harness-session-id` 被采纳。

    改动前实测为 `session_id_source=time_window`（整条 header 被忽略）。
    """
    identity = _resolve({"x-deepseek-harness-session-id": "dsh-sess-0001"})
    assert identity.session_id == "dsh-sess-0001"
    assert identity.session_id_source is SessionIdSource.EXPLICIT


def test_unknown_vendor_session_header_adopted():
    """卡面第二判据：虚构 vendor 同样生效 —— 证明这是**通配**不是名单。

    刚性原则 10：接新 agent 不该回来改我们的代码。
    """
    identity = _resolve({"x-foo-bar-session-id": "foo-sess-42"})
    assert identity.session_id == "foo-sess-42"
    assert identity.session_id_source is SessionIdSource.EXPLICIT


def test_header_name_case_insensitive():
    """真实客户端发的是 `X-Deepseek-Harness-Session-Id` 这种大小写混排。"""
    identity = _resolve({"X-Deepseek-Harness-Session-Id": "MixedCase-1"})
    assert identity.session_id == "MixedCase-1"
    assert identity.session_id_source is SessionIdSource.EXPLICIT


# ── ② 准入边界：基础设施 vendor / 我方保留段不得冒充会话信号 ────────────────


def test_infra_vendor_session_header_ignored():
    """Cloudflare 的 `x-cf-session-id` 不是 agent 的会话 id。

    准入判据与分桶共用一处（`_iter_vendor_headers`）——这条同时守着那次合并。
    """
    identity = _resolve({"x-cf-session-id": "edge-abc"})
    assert identity.session_id != "edge-abc"
    assert identity.session_id_source is not SessionIdSource.EXPLICIT


def test_reserved_segment_not_treated_as_vendor():
    """`x-client-session-id` 的 `client` 是保留段，不是厂商名。"""
    assert extract_vendor_ids(
        {"x-client-session-id": "v"}, VENDOR_SESSION_SUFFIXES
    ) == []


def test_our_own_header_wins_the_tie():
    """两个都在时挑我方约定那个 —— 只为定序，不代表更可信。"""
    identity = _resolve({
        "x-session-id": "ours",
        "x-deepseek-harness-session-id": "theirs",
    })
    assert identity.session_id == "ours"


def test_multi_vendor_choice_is_order_independent():
    """多家同时发时选谁**不得**取决于 header 插入顺序。

    否则同一个客户端会因为 header 顺序变化而换 session —— 那是每轮换身份的
    另一种形态（G11.9 的头号反面用例同源）。
    """
    a = _resolve({"x-zzz-session-id": "z1", "x-aaa-session-id": "a1"})
    b = _resolve({"x-aaa-session-id": "a1", "x-zzz-session-id": "z1"})
    assert a.session_id == b.session_id == "a1"


# ── ③ 值校验：session_id 会原样进 Memory Hub key ─────────────────────────


def test_slash_in_session_id_rejected_and_falls_back():
    """带 `/` 的值会多切一段 Ledger key，读侧按段解析全部错位。

    拒绝后退回推断，不让这一轮失败。
    """
    identity = _resolve({"x-foo-bar-session-id": "a/b"})
    assert identity.session_id != "a/b"
    assert identity.session_id_source is not SessionIdSource.EXPLICIT


def test_screen_rejection_reasons_are_a_closed_set():
    """四种拒因逐一走到 —— 日志按它聚合，混一档就读不出是哪种坏。"""
    assert _screen_external_session_id("   ") == ("", "empty")
    assert _screen_external_session_id("a/b") == ("", "contains_slash")
    assert _screen_external_session_id("a\nb") == ("", "control_chars")
    assert _screen_external_session_id("x" * (_MAX_EXTERNAL_SESSION_ID_CHARS + 1)) == (
        "",
        "too_long",
    )
    assert _screen_external_session_id("  ok-1  ") == ("ok-1", "")


def test_screening_also_covers_our_own_header():
    """同一把尺子管两条路 —— 不给新开的厂商通配路径单独立一套标准。"""
    identity = _resolve({"x-session-id": "a/b"})
    assert identity.session_id != "a/b"
    assert identity.session_id_source is not SessionIdSource.EXPLICIT


# ── ④ 否决守卫：user-id 不得跨越租户边界 ────────────────────────────────────


def test_vendor_user_id_never_touches_user_id():
    """🔴 `x-{vendor}-user-id` 不得改变 user_id。

    user_id 是 P2 可见性过滤的判据，而 header 未经认证 —— 采纳它等于"改个 header
    就能读别人的记忆"。同时它与 2026-08-16 修掉的 `user_id = key hash8` 同形
    （外部可变的值当身份）。这条红了 = 有人把那半边接上了，先读注释再说。
    """
    baseline = _resolve({}).user_id
    spoofed = _resolve({"x-foo-bar-user-id": "somebody-else"})
    assert spoofed.user_id == baseline
    assert "somebody-else" not in spoofed.user_id


def test_vendor_user_id_still_retained_in_header_snapshot():
    """否决的是"当身份用"，不是"丢掉" —— 值仍随 Turn 入库供 dashboard 认领。"""
    identity = _resolve({"x-foo-bar-user-id": "somebody-else"})
    assert identity.request_headers.get("x-foo-bar-user-id") == "somebody-else"


def test_vendor_user_id_header_still_feeds_bucketing():
    """它照旧是分桶信号（只取名字里的 vendor 段，不取值）。"""
    assert extract_vendor_segments({"x-foo-bar-user-id": "whatever"}) == ["foo-bar"]


# ── ⑤ 闭集守卫（硬约束 8：新加一档先让守卫红一次）──────────────────────────


def test_session_suffixes_are_a_subset_of_the_single_definition():
    """`VENDOR_SESSION_SUFFIXES` 必须来自 `VENDOR_SUFFIXES`，不许另抄字面量。"""
    assert VENDOR_SESSION_SUFFIXES <= frozenset(VENDOR_SUFFIXES)


def test_session_suffix_set_is_deliberately_singleton():
    """粒度对不上的 id 当 session_id = 会话被并成一段或切碎。

    要加 `conversation-id` / `thread-id`，先拿真实流量标定粒度关系，
    连带把这条守卫一起改 —— 别让它无声地长大。
    """
    assert VENDOR_SESSION_SUFFIXES == frozenset({"session-id"})


def test_exclusion_sets_have_no_overlap():
    """保留段与基础设施名单各管一件事，重叠说明有一处该收窄。"""
    assert not (RESERVED_VENDOR_SEGMENTS & INFRA_VENDORS)
