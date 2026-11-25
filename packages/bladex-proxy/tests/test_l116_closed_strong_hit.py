"""MQ-L115（批 L116 · 2b）· 久关本（关闭 > 24 h）按「强命中」列出：W_B + 长度闸门 80。

L115 形态（文本全部自编）：周期性重复的同一件事隔 16 天再做一次，旧本按设计不可见。
强命中 = 共享稀有拉丁标识（df ≤ rare_df）∧ 未取整分 ≥ 0.40 ∧ 用户话 ≤ 80 字。
"""
from __future__ import annotations

import structlog.testing
from bladex_core.ledger import STATUS_CLOSED, new_ledger
from bladex_core.ledger_runtime import (
    CLOSED_MATCH_WINDOW_S,
    CLOSED_STRONG_MAX_USER_CHARS,
    CLOSED_STRONG_MIN_SCORE,
    iso_ms,
    relevant_ledgers,
)
from bladex_proxy.agency import AgencyRuntime

NOW = "2026-11-06T12:00:00+00:00"
NOW_MS = iso_ms(NOW)
OLD = "ldg-old"


def _led(lid: str, title: str, goal: str, *, closed_at: str | None = None, superseded_by: str = "",
         agent: str = "hermes:default"):
    led = new_ledger(title=title, goal=goal, goal_source="model",
                     created_at="2026-10-01T00:00:00+00:00", ledger_id=lid)
    if closed_at:
        led.status = STATUS_CLOSED
        led.closed_at = closed_at
        led.closed_by_agent = agent
    led.superseded_by = superseded_by
    return led


def _pool(old_title: str, old_goal: str, **kw) -> dict:
    """40 本：1 久关本 + 39 本无关填充（n=40 ⇒ rare_df=4；填充文本互不共享 query 里的任何词）。"""
    pool = {OLD: _led(OLD, old_title, old_goal, closed_at=kw.pop("closed_at", "2026-10-21T10:40:00+00:00"), **kw)}
    for i in range(39):
        pool[f"ldg-fill{i}"] = _led(f"ldg-fill{i}", f"杂项{chr(0x5000 + i)}{chr(0x5100 + i)}号",
                                    f"{chr(0x5200 + i)}{chr(0x5300 + i)}整理")
    return pool


_SHORT = "检查一下 zorbit 有没有新版本，值不值得升级"            # < 80 字，含拉丁标识 zorbit
_LATIN_OLD = ("zorbit 版本检查与升级", "检查 zorbit 有没有新版本并评估是否升级")


def _ids(ms) -> list[str]:
    return [m.ledger_id for m in ms]


def test_l115_shape_short_text_with_latin_id_lists_the_16_day_old_ledger_first() -> None:
    pool = _pool(*_LATIN_OLD)
    diag: dict[str, int] = {}
    ms = relevant_ledgers(pool, _SHORT, now_ms=NOW_MS, diag=diag)
    assert _ids(ms) == [OLD] and ms[0].closed
    assert (NOW_MS - iso_ms(pool[OLD].closed_at)) / 1000 > CLOSED_MATCH_WINDOW_S
    assert diag == {"closed_old_listed": 1, "closed_old_gated_by_len": 0, "score_gate_waived": 0}


def test_user_text_over_80_chars_does_not_list_the_old_closed_ledger() -> None:
    pool = _pool(*_LATIN_OLD)
    long_text = _SHORT + "，" + "顺带说明一下背景" * 8
    assert len(long_text) > CLOSED_STRONG_MAX_USER_CHARS
    diag: dict[str, int] = {}
    assert relevant_ledgers(pool, long_text, now_ms=NOW_MS, diag=diag) == []
    assert diag == {"closed_old_listed": 0, "closed_old_gated_by_len": 1, "score_gate_waived": 0}


def test_old_closed_ledger_without_shared_latin_identifier_is_not_listed_known_boundary() -> None:
    """W_B 的已知边界：纯中文主题的久关本，即使词面很像也不列。"""
    pool = _pool("星云望远镜固件评估", "评估星云望远镜固件新版本的升级价值")
    diag: dict[str, int] = {}
    assert relevant_ledgers(pool, "检查星云望远镜固件有没有新版本", now_ms=NOW_MS, diag=diag) == []
    assert diag == {"closed_old_listed": 0, "closed_old_gated_by_len": 0, "score_gate_waived": 0}


def test_latin_identifier_but_score_below_040_is_not_listed() -> None:
    # 久关本只共享拉丁标识；query 里还有大量池内常见词（填充本里的词）压低分数
    pool = _pool(*_LATIN_OLD)
    filler_words = "".join(f"{chr(0x5200 + i)}{chr(0x5300 + i)}" for i in range(2))
    text = f"zorbit {filler_words}"
    ms_all = relevant_ledgers(pool, text, now_ms=NOW_MS, closed_strong=False, closed_window_s=None)
    assert ms_all and ms_all[0].ledger_id == OLD and ms_all[0].score_raw < CLOSED_STRONG_MIN_SCORE
    assert relevant_ledgers(pool, text, now_ms=NOW_MS) == []


def test_closed_within_24h_superseded_and_open_behave_as_before() -> None:
    # ≤ 24 h：不要求拉丁标识、不受长度闸门影响
    recent = _pool("星云望远镜固件评估", "评估星云望远镜固件新版本的升级价值", closed_at="2026-11-06T02:00:00+00:00")
    long_text = "检查星云望远镜固件有没有新版本" + "，顺带说明一下背景" * 12
    assert len(long_text) > CLOSED_STRONG_MAX_USER_CHARS
    assert _ids(relevant_ledgers(recent, long_text, now_ms=NOW_MS)) == [OLD]
    # 被取代：久关本即使强命中也不列
    sup = _pool(*_LATIN_OLD, superseded_by="ldg-fill0")
    assert relevant_ledgers(sup, _SHORT, now_ms=NOW_MS) == []
    # 开着的：不受强命中条件约束（纯中文、长文本照列，与 2a 一致）
    opened = _pool("星云望远镜固件评估", "评估星云望远镜固件新版本的升级价值", closed_at="")
    opened[OLD].status = "active"
    opened[OLD].closed_at = None
    assert _ids(relevant_ledgers(opened, long_text, now_ms=NOW_MS)) == [OLD]
    # 与 2a 口径（closed_strong=False）对比：这三种情形结果逐项相同
    for pool, text in ((recent, long_text), (sup, _SHORT), (opened, long_text)):
        assert relevant_ledgers(pool, text, now_ms=NOW_MS) == relevant_ledgers(
            pool, text, now_ms=NOW_MS, closed_strong=False)


def test_2a_arm_never_lists_old_closed() -> None:
    assert relevant_ledgers(_pool(*_LATIN_OLD), _SHORT, now_ms=NOW_MS, closed_strong=False) == []


# ── 渲染与日志（AgencyRuntime）────────────────────────────────────────────

def _rt(monkeypatch, **kw) -> AgencyRuntime:
    monkeypatch.setenv("BLADEX_MODULE_TOOLFACE", "1")
    monkeypatch.setenv("BLADEX_MODULE_LEDGER", "1")
    ag = AgencyRuntime()
    ag.pool.update(_pool(*_LATIN_OLD, **kw))
    return ag


def _block(ag: AgencyRuntime, text: str):
    msgs = [{"role": "system", "content": "You are a test agent."}, {"role": "user", "content": text}]
    with structlog.testing.capture_logs() as cap:
        out = ag.insert_ledger_block(msgs, "hermes:default", tier="medium", with_instruction=True)
    row = next(e for e in cap if e.get("event") == "agency_ledger_candidates")
    return out[-1]["content"], row


def test_render_old_closed_line_carries_how_long_ago_and_log_counts(monkeypatch) -> None:
    # 久关本的「关闭 N 天前」取现在时刻：关闭时刻放在 16 天前
    import datetime as dt
    closed = (dt.datetime.now(dt.UTC) - dt.timedelta(days=16, hours=3)).isoformat(timespec="seconds")
    ag = _rt(monkeypatch, closed_at=closed)
    block, row = _block(ag, _SHORT)
    line = next(r for r in block.splitlines() if r.startswith(f"- {OLD}:"))
    assert "(16 days ago); switching to it reopens it]" in line and "[closed by hermes:default " in line
    assert row["relevant"] == [OLD] and row["closed_old_listed"] == 1 and row["closed_gt24h_gated_by_len"] == 0


def test_log_counts_when_gated_by_length_and_when_not_strong(monkeypatch) -> None:
    import datetime as dt
    closed = (dt.datetime.now(dt.UTC) - dt.timedelta(days=16)).isoformat(timespec="seconds")
    ag = _rt(monkeypatch, closed_at=closed)
    long_text = _SHORT + "，" + "顺带说明一下背景" * 8
    block, row = _block(ag, long_text)
    assert OLD not in block.split("Other recent ledgers")[0] and row["relevant"] == []
    assert row["closed_old_listed"] == 0 and row["closed_gt24h_gated_by_len"] == 1
    _, row2 = _block(ag, "今天北京天气怎么样")                       # 不满足强命中
    assert row2["closed_old_listed"] == 0 and row2["closed_gt24h_gated_by_len"] == 0
