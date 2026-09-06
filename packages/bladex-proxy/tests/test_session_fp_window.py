"""MQ-S9 / G12.2-pre：fp: 会话桶时间窗切分（窗口 4h，2026-08-20 Jason 拍板）。

钉住的语义（每条对应 `_FpEpochWindow` 类注的一条边界）：
  ① 首纪元不带后缀——存量零漂移，重启退回原形（最坏 = 旧撞桶行为，不更坏）；
  ② 超窗才切，后缀 = 新纪元起点小时（无计数器，两纪元不可能同小时开始）；
  ③ 只碰 fp: 桶——tw:/显式 session 原样通过；
  ④ 尾部接续轮 touch() 续命——长会话不被误判超窗；
  ⑤ 0 = 关闭，逐字回旧行为（回归通道）。
"""

from __future__ import annotations

import pytest

from bladex_proxy.identity import _FpEpochWindow
from bladex_proxy.models import ChatCompletionRequest

T0 = 1_755_600_000.0  # 固定起点，避免真实时钟参与断言
H = 3600.0


@pytest.fixture()
def win(monkeypatch) -> _FpEpochWindow:
    monkeypatch.setenv("BLADEX_SESSION_FP_WINDOW_S", "14400")  # 4h，与拍板值一致
    return _FpEpochWindow()


# ── ①② 纪元语义 ──────────────────────────────────────────────────────────


def test_first_epoch_keeps_legacy_form(win) -> None:
    assert win.apply("u", "a", "fp:abc123def456", now=T0) == "fp:abc123def456"


def test_within_window_stays_in_epoch(win) -> None:
    win.apply("u", "a", "fp:abc123def456", now=T0)
    assert win.apply("u", "a", "fp:abc123def456", now=T0 + 3.9 * H) == "fp:abc123def456"


def test_gap_beyond_window_starts_suffixed_epoch(win) -> None:
    import time as _t
    win.apply("u", "a", "fp:abc123def456", now=T0)
    got = win.apply("u", "a", "fp:abc123def456", now=T0 + 5 * H)
    expect = "fp:abc123def456.e" + _t.strftime("%Y%m%d%H", _t.gmtime(T0 + 5 * H))
    assert got == expect


def test_second_split_gets_a_different_suffix(win) -> None:
    win.apply("u", "a", "fp:abc123def456", now=T0)
    first = win.apply("u", "a", "fp:abc123def456", now=T0 + 5 * H)
    second = win.apply("u", "a", "fp:abc123def456", now=T0 + 12 * H)
    assert first != second and second.startswith("fp:abc123def456.e")


def test_restart_degrades_to_legacy_not_worse(monkeypatch) -> None:
    """重启（状态清空）→ 退回无后缀原形 = 旧撞桶行为，严格不劣于现状（边界 ①）。"""
    monkeypatch.setenv("BLADEX_SESSION_FP_WINDOW_S", "14400")
    a = _FpEpochWindow()
    a.apply("u", "x", "fp:abc123def456", now=T0)
    assert a.apply("u", "x", "fp:abc123def456", now=T0 + 9 * H).startswith("fp:abc123def456.e")
    b = _FpEpochWindow()  # 模拟重启
    assert b.apply("u", "x", "fp:abc123def456", now=T0 + 10 * H) == "fp:abc123def456"


def test_buckets_are_isolated_per_user_agent(win) -> None:
    """锚是 per-(agent, session) 的地基：不同 agent 的同名 fp 互不影响计时。"""
    win.apply("u", "codex", "fp:abc123def456", now=T0)
    got = win.apply("u", "claude-code", "fp:abc123def456", now=T0 + 5 * H)
    assert got == "fp:abc123def456"  # claude-code 首见 → 首纪元，不吃 codex 的 gap


# ── ③⑤ 范围与开关 ───────────────────────────────────────────────────────


def test_non_fp_sessions_pass_through(win) -> None:
    assert win.apply("u", "a", "tw:u:a:12345", now=T0) == "tw:u:a:12345"


def test_zero_window_disables(monkeypatch) -> None:
    monkeypatch.setenv("BLADEX_SESSION_FP_WINDOW_S", "0")
    w = _FpEpochWindow()
    w.apply("u", "a", "fp:abc123def456", now=T0)
    assert w.apply("u", "a", "fp:abc123def456", now=T0 + 100 * H) == "fp:abc123def456"


# ── ④ 尾部接续续命 ──────────────────────────────────────────────────────


def test_tail_continuation_touch_keeps_epoch_alive(win) -> None:
    """6 小时连续工作（全程尾部接续）后再来一问，不得被误判超窗切开。"""
    win.apply("u", "a", "fp:abc123def456", now=T0)
    win.touch("u", "a", "fp:abc123def456", now=T0 + 3 * H)
    win.touch("u", "a", "fp:abc123def456", now=T0 + 6 * H)
    assert win.apply("u", "a", "fp:abc123def456", now=T0 + 7 * H) == "fp:abc123def456"


def test_touch_on_suffixed_id_seeds_suffix(win) -> None:
    """重启后先来的是接续轮（缓存里带后缀的 id）：touch 要把后缀种回状态，
    随后的指纹路径轮必须回到同一纪元，不得退回原形另开。"""
    win.touch("u", "a", "fp:abc123def456.e2026082010", now=T0)
    assert win.apply("u", "a", "fp:abc123def456", now=T0 + 1 * H) \
        == "fp:abc123def456.e2026082010"


# ── 接线：resolve_identity 的指纹路径真的过这道窗 ─────────────────────────


def test_resolve_path_splits_across_cache_miss(monkeypatch) -> None:
    """病的真实形态：会话缓存失效/重启后，同开场白隔天重来 → 旧行为撞回同一桶。
    模拟：两次 resolve 之间清掉会话缓存（缓存命中走 touch 不走 apply，是另一条路）。"""
    from bladex_proxy import identity as I

    monkeypatch.setenv("BLADEX_SESSION_FP_WINDOW_S", "14400")
    monkeypatch.setattr(I, "_fp_epochs", I._FpEpochWindow())
    monkeypatch.setattr(I, "_session_cache", type(I._session_cache)())

    msgs = [{"role": "user", "content": "mqs9 独特开场白 alpha"},
            {"role": "assistant", "content": "ok"}]
    req = ChatCompletionRequest(model="m", messages=msgs)

    monkeypatch.setattr(I.time, "time", lambda: T0)
    ident1, _ = I.resolve_identity({}, req)
    assert ident1.session_id.startswith("fp:") and ".e" not in ident1.session_id

    monkeypatch.setattr(I, "_session_cache", type(I._session_cache)())  # 缓存失效
    monkeypatch.setattr(I.time, "time", lambda: T0 + 20 * H)
    ident2, _ = I.resolve_identity({}, req)
    assert ident2.session_id.startswith(ident1.session_id + ".e"), (
        f"隔天同开场白仍落 {ident2.session_id}，撞桶未被切开")

    monkeypatch.setattr(I, "_session_cache", type(I._session_cache)())
    monkeypatch.setattr(I.time, "time", lambda: T0 + 20.5 * H)
    ident3, _ = I.resolve_identity({}, req)
    assert ident3.session_id == ident2.session_id, "窗内重来不得再切"
