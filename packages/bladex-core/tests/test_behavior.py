"""批 F1a 验收剧本：行为画像两个口径（S4 last_seen 聚合 / S5 会话去重过渡口径）。

判据编号对应 `docs/planning/prompt-batch-F1a-20260922.md` §F1a-1 / §F1a-2。
"""

from __future__ import annotations

import inspect
from datetime import UTC, datetime, timedelta

import bladex_core.behavior as _b
import pytest
from bladex_core.behavior import (
    HELD_AMBIGUOUS_FP,
    HELD_AMBIGUOUS_HEAD,
    HELD_HEAD_TOO_LONG,
    TW_WINDOW_S,
    TurnMeta,
    agent_base,
    dedupe_sessions,
    last_seen_by_agent,
    session_source,
)

T0 = datetime(2026, 9, 19, 9, 0, tzinfo=UTC)
NOW = datetime(2026, 9, 22, 12, 0, tzinfo=UTC)


def _m(agent: str, sess: str, minutes: float, principal: str = "local") -> TurnMeta:
    return TurnMeta(principal=principal, agent_id=agent, session_id=sess,
                    ts=T0 + timedelta(minutes=minutes))


# ───────────────────────────── TurnMeta.from_scan ─────────────────────────────


class TestTurnMeta:
    def test_from_scan_key_shape(self):
        m = TurnMeta.from_scan("local/hermes:default/fp:cad110ffe432/1789810000000-0",
                               "2026-09-19T09:25:22.314383Z")
        assert m is not None
        assert (m.principal, m.agent_id, m.session_id) == ("local", "hermes:default", "fp:cad110ffe432")
        assert m.ts == datetime(2026, 9, 19, 9, 25, 22, 314383, tzinfo=UTC)

    def test_from_scan_falls_back_to_entry_ms_then_none(self):
        m = TurnMeta.from_scan("local/codex/fp:abc/1789810000000-3", "")
        assert m is not None and m.ts == datetime.fromtimestamp(1789810000, tz=UTC)
        assert TurnMeta.from_scan("local/codex/fp:abc/garbage", "") is None
        assert TurnMeta.from_scan("__internal__", "2026-09-19T09:00:00Z") is None
        assert TurnMeta.from_scan("local//fp:abc/1-0", "2026-09-19T09:00:00Z") is None


# ───────────────────────────── S4 · last_seen ─────────────────────────────


class TestLastSeenS4:
    def test_1_base_merges_hermes_variants(self):
        metas = [_m("hermes", "tw:local:hermes:1", 0), _m("hermes:default", "fp:a", 5),
                 _m("hermes:accept", "fp:b", 3), _m("codex", "fp:c", 1)]
        out = last_seen_by_agent(metas, now=NOW)
        assert set(out) == {"hermes", "codex"}
        assert out["hermes"].last_seen == T0 + timedelta(minutes=5)
        assert out["hermes"].first_seen == T0
        assert out["hermes"].turns == 3

    def test_2_pure_now_is_input_and_deterministic(self):
        metas = [_m("codex", "fp:c", 1), _m("dsh", "fp:d", 2)]
        a = last_seen_by_agent(metas, now=NOW)
        b = last_seen_by_agent(list(reversed(metas)), now=NOW)
        assert a == b and list(a) == list(b)
        assert a["dsh"].age_s == int((NOW - (T0 + timedelta(minutes=2))).total_seconds())
        # 另一个 now ⇒ 只有 age 变，last_seen 不变
        c = last_seen_by_agent(metas, now=NOW + timedelta(hours=1))
        assert c["dsh"].last_seen == a["dsh"].last_seen and c["dsh"].age_s == a["dsh"].age_s + 3600
        with pytest.raises(ValueError):
            last_seen_by_agent(metas, now=datetime(2026, 9, 22, 12, 0))  # naive
        src = inspect.getsource(_b)
        assert "datetime.now(" not in src and "time.time(" not in src and "utcnow(" not in src

    def test_3_zero_traffic_declared_agent_not_in_output(self):
        # 阴性对照（MQ-F1 防线 / MQ-F6 残留）：规则库声明但零流量的 agent 不得出现。
        # 本函数没有"声明清单"入参——把 cursor/openclaw 塞进 eligible 也进不来。
        declared = {"cursor", "openclaw", "codex", "dsh"}
        metas = [_m("codex", "fp:c", 1), _m("dsh", "fp:d", 2)]
        out = last_seen_by_agent(metas, now=NOW, eligible=lambda a: a in declared)
        assert "cursor" not in out and "openclaw" not in out
        assert set(out) <= {agent_base(m.agent_id) for m in metas}
        assert set(out) == {"codex", "dsh"}

    def test_3b_eligible_predicate_sees_full_agent_id(self):
        metas = [_m("codex", "fp:c", 1), _m("codex:guardian", "fp:g", 9)]
        out = last_seen_by_agent(metas, now=NOW, eligible=lambda a: a == "codex")
        assert out["codex"].last_seen == T0 + timedelta(minutes=1)
        assert out["codex"].turns == 1

    def test_4_unknown_buckets_excluded(self):
        metas = [_m("unknown", "tw:local:unknown:1", 0), _m("unknown-0eaa8faf", "tw:x", 1),
                 _m("codex", "fp:c", 1)]
        out = last_seen_by_agent(metas, now=NOW)
        assert set(out) == {"codex"}

    def test_5_negative_without_base_normalisation_three_rows(self, monkeypatch):
        # 阴性对照：去掉 base 归一 ⇒ hermes 三行不合并（本用例证明判据①不是碰巧）。
        metas = [_m("hermes", "tw:local:hermes:1", 0), _m("hermes:default", "fp:a", 5),
                 _m("hermes:accept", "fp:b", 3)]
        assert len(last_seen_by_agent(metas, now=NOW)) == 1
        monkeypatch.setattr(_b, "agent_base", lambda a: a)
        assert len(last_seen_by_agent(metas, now=NOW)) == 3


# ───────────────────────────── S5 · 会话去重 ─────────────────────────────


def _live_case() -> list[TurnMeta]:
    """批 S2 SESSIONS.md 双行病例（hermes:default，2026-09-19）：tw 头 09:24:58 一轮，
    fp 身 09:25:22 起 74 轮；同一时刻还有裸 `hermes` 的 tw 桶（checkpoint 子调用）。"""
    a = "hermes:default"
    metas = [TurnMeta(principal="local", agent_id="hermes", session_id="tw:local:hermes:5966032",
                      ts=datetime(2026, 9, 19, 9, 24, 31, tzinfo=UTC)),
             TurnMeta(principal="local", agent_id=a, session_id="tw:local:hermes:default:5966032",
                      ts=datetime(2026, 9, 19, 9, 24, 58, tzinfo=UTC))]
    start = datetime(2026, 9, 19, 9, 25, 22, tzinfo=UTC)
    for i in range(74):
        metas.append(TurnMeta(principal="local", agent_id=a, session_id="fp:cad110ffe432",
                              ts=start + timedelta(minutes=6 * i)))
    return metas


class TestSessionDedupS5:
    def test_1_docstring_names_transitional_and_owner(self):
        doc = dedupe_sessions.__doc__ or ""
        assert "过渡口径" in doc
        assert "identity" in doc and "真修归 identity 域" in doc
        assert "MQ-A73" in doc and "MQ-L13" in doc

    def test_2_live_case_two_rows_become_one(self):
        r = dedupe_sessions(_live_case())
        ids = [s.session_id for s in r.sessions]
        assert ids == ["tw:local:hermes:5966032", "fp:cad110ffe432"]
        body = r.sessions[1]
        assert body.merged_heads == ("tw:local:hermes:default:5966032",)
        assert body.first_ts == datetime(2026, 9, 19, 9, 24, 58, tzinfo=UTC)
        assert body.turns == 75
        assert r.merged == (("tw:local:hermes:default:5966032", "fp:cad110ffe432"),)
        assert r.held == ()
        # 裸 hermes 的 tw 桶：agent_id 不同（同 base 也不算），原样保留
        assert r.sessions[0].merged_heads == ()

    def test_2b_deterministic_under_input_order(self):
        metas = _live_case()
        a = dedupe_sessions(metas)
        b = dedupe_sessions(list(reversed(metas)))
        assert a == b
        assert a.model_dump_json() == b.model_dump_json()

    def test_3_ambiguity_holds_both_rows(self):
        # 一个 tw 头、两个紧随的 fp 身 ⇒ 不知道并给谁 ⇒ 都不并
        metas = [_m("dsh", "tw:local:dsh:1", 0), _m("dsh", "fp:a", 1), _m("dsh", "fp:b", 2)]
        r = dedupe_sessions(metas)
        assert r.merged == () and len(r.sessions) == 3
        assert r.held == (("tw:local:dsh:1", HELD_AMBIGUOUS_FP),)
        # 两个 tw 头争同一个 fp 身 ⇒ 都不并
        metas = [_m("dsh", "tw:local:dsh:1", 0), _m("dsh", "tw:local:dsh:2", 1), _m("dsh", "fp:a", 2)]
        r = dedupe_sessions(metas)
        assert r.merged == () and len(r.sessions) == 3
        assert set(r.held) == {("tw:local:dsh:1", HELD_AMBIGUOUS_HEAD),
                               ("tw:local:dsh:2", HELD_AMBIGUOUS_HEAD)}

    def test_4_negative_no_false_merge(self):
        # 🔴 红线：真正不同的两个会话不许并。
        # (a) 同 agent、ts 不相接（tw 头在 fp 身之前 > 一个窗口）
        far = [_m("dsh", "tw:local:dsh:1", 0), _m("dsh", "fp:a", TW_WINDOW_S / 60 + 1)]
        r = dedupe_sessions(far)
        assert r.merged == () and r.held == () and len(r.sessions) == 2
        # (b) 恰好在窗口边界内才并
        edge = [_m("dsh", "tw:local:dsh:1", 0), _m("dsh", "fp:a", TW_WINDOW_S / 60)]
        assert dedupe_sessions(edge).merged == (("tw:local:dsh:1", "fp:a"),)
        # (c) tw 落在 fp 中间（长会话里的单消息子调用）⇒ 不并
        mid = [_m("dsh", "fp:a", 0), _m("dsh", "tw:local:dsh:1", 30), _m("dsh", "fp:a", 60)]
        r = dedupe_sessions(mid)
        assert r.merged == () and len(r.sessions) == 2
        # (d) tw 在 fp 之后 ⇒ 不并
        after = [_m("dsh", "fp:a", 0), _m("dsh", "tw:local:dsh:1", 1)]
        assert dedupe_sessions(after).merged == ()
        # (e) 不同 agent（同 base 也不算）/ 不同 principal ⇒ 不并
        cross_agent = [_m("hermes", "tw:local:hermes:1", 0), _m("hermes:default", "fp:a", 1)]
        assert dedupe_sessions(cross_agent).merged == ()
        cross_principal = [_m("dsh", "tw:u:dsh:1", 0, principal="u1"), _m("dsh", "fp:a", 1)]
        assert dedupe_sessions(cross_principal).merged == ()
        # (f) tw 桶里落了两轮（可能是两次会话的首条消息）⇒ 不并，记 head_too_long
        two_heads = [_m("dsh", "tw:local:dsh:1", 0), _m("dsh", "tw:local:dsh:1", 0.5),
                     _m("dsh", "fp:a", 1)]
        r = dedupe_sessions(two_heads)
        assert r.merged == () and r.held == (("tw:local:dsh:1", HELD_HEAD_TOO_LONG),)
        # (g) explicit 会话 id 不参与任何合并
        explicit = [_m("claude-code", "29fd2e19-e89f", 0), _m("claude-code", "fp:a", 1),
                    _m("claude-code", "tw:x:claude-code:1", 0)]
        r = dedupe_sessions(explicit)
        assert {s.session_id for s in r.sessions} >= {"29fd2e19-e89f"}
        assert all("29fd2e19" not in pair for pair in r.merged)

    def test_4b_unknown_buckets_excluded(self):
        metas = [_m("unknown-0eaa8faf", "tw:x", 0), _m("dsh", "fp:a", 1)]
        assert [s.agent_id for s in dedupe_sessions(metas).sessions] == ["dsh"]

    def test_session_source(self):
        assert session_source("fp:abc") == "fp"
        assert session_source("tw:local:dsh:1") == "tw"
        assert session_source("29fd2e19-e89f") == "explicit"


def test_tw_window_matches_identity():
    """core 不 import proxy；两处的窗口宽度由本测试对账（原则 12：默认值单一真相）。"""
    identity = pytest.importorskip("bladex_proxy.identity")
    assert TW_WINDOW_S == identity._TIME_WINDOW_SECONDS
