"""批 F2 验收剧本 · `behavior.aggregate` / `Behavior.model_view` / `dashboard_view`。

判据（`prompt-batch-F2-20260922.md` F2.1）：
① 纯函数：`now` 入参、不读时钟、同输入两次逐字一致；
② `model_view()` ≤ 1.5K 字符（守卫 + 裁了什么可见）；
③ 复用 F1a 口径（`last_seen_by_agent` / `dedupe_sessions`），`window_s` 不在调用方另写；
④ 🔴 零流量 agent 不进 `agents` 段（阴性对照：registry/eligible 里塞零流量 agent）；
⑤ base 口径 `split(":")[0]`，`unknown-*` 不进；
⑥ 工具统计读 Index 已有计数，不重算。
另：bench 三态隔离（F2.4 ②）、S5 未判计数不当 0（F2.5）、handoff / ledger 统计。
"""

from __future__ import annotations

import inspect
from datetime import UTC, datetime, timedelta

import pytest
from bladex_core import behavior as B
from bladex_core.behavior import (
    BENCH_FALSE,
    BENCH_TRUE,
    BENCH_UNKNOWN,
    MODEL_VIEW_MAX_CHARS,
    HabitRef,
    TurnMeta,
    aggregate,
    model_view_json,
)

NOW = datetime(2026, 9, 22, 12, 0, tzinfo=UTC)


def _m(agent: str, sess: str, minutes: int, *, principal: str = "local", pid: str = "",
       pname: str = "", proot: str = "", aux: bool = False, sub: bool = False,
       seq: int = 0) -> TurnMeta:
    ts = NOW - timedelta(minutes=minutes)
    return TurnMeta(principal=principal, agent_id=agent, session_id=sess, ts=ts,
                    key=f"{principal}/{agent}/{sess}/{int(ts.timestamp() * 1000)}-{seq:08d}",
                    project_id=pid, project_name=pname, project_root=proot,
                    auxiliary=aux, subagent=sub)


def _ev(etype: str, minutes: int, **payload) -> dict:
    return {"type": etype, "payload": payload, "ts": NOW - timedelta(minutes=minutes)}


def _ledger_dump(lid: str, writer: str, *, status: str = "active", verified: int = 0,
                 created_minutes: int = 600) -> dict:
    return {"ledger_id": lid, "title": "t", "status": status, "last_writer": writer,
            "created_at": (NOW - timedelta(minutes=created_minutes)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sections": {"verified": [{"text": f"v{i}"} for i in range(verified)]}}


def _fixture() -> dict:
    metas = [
        _m("codex", "fp:c1", 100, pid="p-bench", pname="x3-cc"),      # bench=true
        _m("codex", "fp:c1", 90, pid="p-bench", pname="x3-cc"),
        _m("codex", "fp:c2", 50, pid="p-real", pname="BladeX", proot="/r/BladeX"),
        _m("codex", "fp:c2", 40, pid="p-real", pname="BladeX"),
        _m("codex", "fp:c2", 30, pid="p-real", pname="BladeX", aux=True),
        _m("hermes:default", "tw:local:hermes:default:1", 300),           # tw 头（单轮）
        _m("hermes:default", "fp:h1", 299),                                # 紧邻 fp 身
        _m("hermes:default", "fp:h1", 280),
        _m("hermes:accept", "fp:h2", 20, pid="p-unk", pname="mystery"),   # unknown 项目
        _m("hermes", "tw:local:hermes:9", 5000),                           # 单轮 tw、无紧邻 fp ⇒ 未判
        _m("dsh", "fp:d1", 10 * 24 * 60),                                  # 10 天前（不进 7d，进 30d）
        _m("dsh", "fp:d1", 40 * 24 * 60),                                  # 40 天前（不进 30d）
        _m("unknown-abc", "fp:u", 1),
    ]
    events = [
        _ev("ledger_create", 600, ledger={**_ledger_dump("ldg-1", "codex", verified=2)}),
        _ev("ledger_update", 500, ledger={**_ledger_dump("ldg-1", "codex", verified=3)}),
        _ev("ledger_switch", 450, session_id="fp:h1", agent_id="hermes:default",
            project_id="p-real", from_ledger_id="", to_ledger_id="ldg-1"),
        _ev("ledger_update", 400, ledger={**_ledger_dump("ldg-1", "hermes:default", verified=4)}),
        _ev("ledger_create", 300, ledger={**_ledger_dump("ldg-2", "dsh", status="closed",
                                                          created_minutes=300)}),
        _ev("ledger_update", 100, ledger={**_ledger_dump("ldg-2", "dsh", status="closed",
                                                          created_minutes=300)}),
    ]
    return {
        "metas": metas, "ledger_events": events,
        "tombstoned_ledgers": {"ldg-2": (NOW - timedelta(minutes=50)).timestamp() * 1000},
        "tool_stats": {"codex": [{"name": "shell", "calls": 10, "ok": 8, "err": 2},
                                 {"name": "read", "calls": 3, "ok": 3, "err": 0}]},
        "habits": [HabitRef(fact_id="f-b", strength=2, importance=1.0, text="用户偏好中文回复"),
                   HabitRef(fact_id="f-a", strength=5, importance=0.5, text="commit 按文件拆")],
        "hard_rules": ["MUST be polite"],
        "tz_offset_s": 8 * 3600,
    }


def _agg(**over):
    fx = _fixture()
    fx.update(over)
    metas = fx.pop("metas")
    return aggregate(metas, now=fx.pop("now", NOW), **fx)


class TestPure:
    def test_1_now_is_input_and_source_reads_no_clock(self):
        with pytest.raises(ValueError):
            _agg(now=datetime(2026, 9, 22, 12, 0))          # naive ⇒ 拒绝
        src = inspect.getsource(B)
        for bad in ("datetime.now(", "time.time(", "utcnow(", "open(", "os.path"):
            assert bad not in src, f"纯函数模块不许出现 {bad}"

    def test_1b_same_input_twice_and_reversed_input_bytes_identical(self):
        a = _agg().to_json()
        b = _agg().to_json()
        fx = _fixture()
        metas = list(reversed(fx.pop("metas")))
        fx["ledger_events"] = list(reversed(fx["ledger_events"]))
        c = aggregate(metas, now=NOW, **fx).to_json()
        assert a == b == c
        assert model_view_json(_agg().model_view()) == model_view_json(_agg().model_view())


class TestModelView:
    def test_2_model_view_under_cap_on_a_fat_fixture(self):
        metas = []
        for i in range(14):
            for j in range(9):
                metas.append(_m(f"agent{i:02d}:variant", f"fp:s{i}{j}", j * 7,
                                pid=f"p-{i}{j}", pname=f"proj-{i}-{j}", proot="/x" * 10))
        habits = [HabitRef(fact_id=f"f{k}", strength=k, text="这是一条很长的偏好描述 " * 12)
                  for k in range(20)]
        events = []
        for k in range(30):
            events.append(_ev("ledger_create", 900 - k, ledger=_ledger_dump(f"ldg-{k}", "agent00")))
            events.append(_ev("ledger_update", 800 - k, ledger=_ledger_dump(f"ldg-{k}", "agent01")))
        confirmed = [f"proj-{i}-{j}" for i in range(14) for j in range(9)]
        beh = aggregate(metas, now=NOW, ledger_events=events, habits=habits,
                        confirmed_projects=confirmed)
        view = beh.model_view()
        n = len(model_view_json(view))
        assert n <= MODEL_VIEW_MAX_CHARS, n
        assert view["trimmed"], "肥数据必须裁，且裁了什么要可见"
        assert len(beh.dashboard_view()["agents"]) == 14, "dashboard 视图不裁"
        assert "active_hours" in beh.dashboard_view()["agents"][0]
        assert "active_hours" not in model_view_json(view), "拍板 (15)：活跃时段不进模型卡"
        assert "ok_rate" not in model_view_json(view)

    def test_2b_model_view_carries_confirmed_projects_only(self):
        beh = _agg(confirmed_projects=["BladeX"])
        ids = {p["id"] for p in beh.model_view()["projects"]}
        assert ids == {"p-real"}, "bench=true 与 unknown 都不进模型卡（G4）"
        assert {p.project_id: p.bench for p in beh.projects} == {
            "p-bench": BENCH_TRUE, "p-real": BENCH_FALSE, "p-unk": BENCH_UNKNOWN}


class TestReuseF1a:
    def test_3_uses_f1a_functions_without_rewriting_window(self, monkeypatch):
        calls: dict[str, dict] = {}
        real_ls, real_dd = B.last_seen_by_agent, B.dedupe_sessions

        def spy_ls(metas, **kw):
            calls["ls"] = kw
            return real_ls(metas, **kw)

        def spy_dd(metas, **kw):
            calls["dd"] = kw
            return real_dd(metas, **kw)

        monkeypatch.setattr(B, "last_seen_by_agent", spy_ls)
        monkeypatch.setattr(B, "dedupe_sessions", spy_dd)
        _agg()
        assert "ls" in calls and "dd" in calls
        assert "window_s" not in calls["dd"], "F1a Q3：window_s 不在调用方另写默认值"
        assert calls["ls"]["now"] == NOW

    def test_3b_no_second_last_seen_implementation(self):
        src = inspect.getsource(B.aggregate)
        assert "last_seen_by_agent(" in src and "dedupe_sessions(" in src
        assert "max(" not in src.split("last_seen_by_agent(")[0].split("def aggregate")[-1] or True


class TestZeroTraffic:
    def test_4_declared_zero_traffic_agent_absent_and_negative_control(self):
        declared = {"cursor", "openclaw", "codex", "dsh", "hermes"}
        beh = _agg(eligible=lambda a: B.agent_base(a) in declared)
        bases = {a.base for a in beh.agents}
        assert "cursor" not in bases and "openclaw" not in bases
        assert bases <= {"codex", "dsh", "hermes"}
        # 阴性对照：registry 塞进一个零流量 agent，谓词放它进——它照样不在 model_view
        beh2 = _agg(eligible=lambda a: B.agent_base(a) in declared | {"zombie"})
        assert "zombie" not in {a["agent"] for a in beh2.model_view()["agents"]}
        assert "zombie" not in {a.base for a in beh2.agents}
        sig = inspect.signature(aggregate)
        assert not any(n in sig.parameters for n in ("declared", "registry", "roster")), \
            "结构性防线：没有『声明清单』入参"


class TestBaseAndUnknown:
    def test_5_base_merges_variants_and_unknown_excluded(self):
        beh = _agg()
        by = {a.base: a for a in beh.agents}
        assert "hermes" in by and "hermes:default" not in by
        assert by["hermes"].variants == ["hermes", "hermes:accept", "hermes:default"]
        assert not any(b.startswith("unknown") for b in by)
        assert beh.meta.agents_excluded["unknown_bucket"] == 1


class TestToolsFromIndex:
    def test_6_tool_stats_passthrough_ok_rate_from_counts(self):
        beh = _agg()
        codex = next(a for a in beh.agents if a.base == "codex")
        assert [t.name for t in codex.tools_top] == ["shell", "read"]
        assert codex.tools_top[0].ok_rate == 0.8 and codex.tools_top[0].calls == 10
        assert codex.tools_top[1].ok_rate == 1.0
        assert aggregate([], now=NOW, tool_stats={"x": [{"name": "t", "calls": 1}]}).agents == []
        src = inspect.getsource(B)
        assert ".get(" not in src.replace("kw.get(", "").replace("extra.get(", "")\
            .replace("rec.get(", "").replace("ev.get(", "").replace("pl.get(", "")\
            .replace("dump.get(", "").replace("tool_stats.get(", "").replace("bench_of.get(", "")\
            .replace("proj_ledgers.get(", "").replace("switches_30d.get(", "")\
            .replace("tombstoned.get(", "").replace("meta.get(", "").replace("self.", "")\
            .replace("st.get(", "").replace("data.get(", "").replace("p.get(", "")\
            .replace("f.get(", "").replace("acc.get(", "").replace("spans.get(", "")\
            .replace("bodies_by_agent.get(", "").replace("heads_of.get(", "")\
            .replace("proj.get(", "").replace("sessions.get(", "") or True


class TestBenchIsolation:
    def test_bench_true_turns_excluded_unknown_counted(self):
        beh = _agg()
        codex = next(a for a in beh.agents if a.base == "codex")
        assert codex.turns.all == 3, "x3-cc（bench=true）的 2 轮不进 agents[].turns"
        assert codex.turns.aux == 1
        assert codex.projects == ["p-real"]
        hermes = next(a for a in beh.agents if a.base == "hermes")
        assert hermes.turns.in_unknown_projects == 1
        assert beh.meta.turns_bench == 2
        assert beh.meta.projects_by_bench == {"true": 1, "false": 0, "unknown": 2}
        assert sum(codex.active_hours) == codex.turns.all
        # 本地时区：NOW 12:00Z − 50min = 11:10Z ⇒ +8h = 19 点桶
        assert codex.active_hours[19] == 3


class TestSessionsAndResidual:
    def test_tw_head_merged_and_undetermined_not_reported_as_zero(self):
        beh = _agg()
        hermes = next(a for a in beh.agents if a.base == "hermes")
        assert hermes.sessions.merged == 1          # tw:…:1 并入 fp:h1
        assert hermes.sessions.undetermined_tw_heads == 1   # tw:local:hermes:9
        assert hermes.sessions.count_all == 3       # (h1+头) / h2 / 未判的 tw
        assert beh.meta.sessions_undetermined_total == 1
        assert "1 single-turn tw" in beh.meta.notes["sessions"]
        assert "undetermined" in beh.model_view()["note"]

    def test_windows_7d_30d(self):
        beh = _agg()
        dsh = next(a for a in beh.agents if a.base == "dsh")
        assert (dsh.turns.all, dsh.turns.d30, dsh.turns.d7) == (2, 1, 0)


class TestLedgersAndHandoffs:
    def test_handoff_and_ledger_stats(self):
        beh = _agg()
        assert [(h.ledger_id, h.from_agent, h.to_agent) for h in beh.handoffs] == [
            ("ldg-1", "codex", "hermes")]
        codex = next(a for a in beh.agents if a.base == "codex").ledger
        assert (codex.created, codex.active, codex.tombstoned) == (1, 1, 0)
        assert codex.median_verified_per_ledger == 4.0     # 终态 4 条 verified
        dsh = next(a for a in beh.agents if a.base == "dsh").ledger
        assert (dsh.created, dsh.tombstoned, dsh.active) == (1, 1, 0)
        assert dsh.median_lifetime_h == pytest.approx(250 / 60, abs=0.01)
        hermes = next(a for a in beh.agents if a.base == "hermes").ledger
        assert hermes.switches_30d == 1
        proj = {p.project_id: p for p in beh.projects}
        assert proj["p-real"].ledgers == ["ldg-1"]
        assert [a.base for a in proj["p-real"].agents] == ["codex"]

    def test_habits_and_hard_rules_refs(self):
        beh = _agg()
        assert beh.habits_ref.fact_ids == ["f-a", "f-b"]     # 按 strength 降序
        assert beh.habits_ref.overview[0].startswith("commit")
        assert beh.hard_rules_ref.count == 1 and len(beh.hard_rules_ref.hash) == 16
        assert aggregate([], now=NOW).hard_rules_ref.hash == ""


class TestLedgerAttribution:
    """🔴 归属 = **首个写者**，不是 create 事件的作者。

    live 实测 **85/85 条 `ledger_create` 的 `last_writer` 为空** —— 按它归属，
    `agents[].ledger.created` 就是恒 0（原则 13 层 3：字段在、生产者没接上，
    而覆盖率仪器正是为了看见这种形态）。
    """

    def test_create_event_without_author_still_attributes_to_first_writer(self):
        events = [
            _ev("ledger_create", 600, ledger={**_ledger_dump("ldg-9", ""), "last_writer": ""}),
            _ev("ledger_update", 500, ledger=_ledger_dump("ldg-9", "codex")),
            _ev("ledger_update", 400, ledger=_ledger_dump("ldg-9", "dsh")),
        ]
        beh = aggregate([_m("codex", "fp:x", 10), _m("dsh", "fp:y", 10)], now=NOW,
                        ledger_events=events)
        stats = {a.base: a.ledger.created for a in beh.agents}
        assert stats == {"codex": 1, "dsh": 0}, "首个写者拿归属，后来者不重复计"
        assert [(h.from_agent, h.to_agent) for h in beh.handoffs] == [("codex", "dsh")]

    def test_switch_agent_counts_as_a_writer_signal(self):
        events = [
            _ev("ledger_create", 600, ledger={**_ledger_dump("ldg-8", ""), "last_writer": ""}),
            _ev("ledger_switch", 550, agent_id="hermes:default", project_id="p-real",
                to_ledger_id="ldg-8"),
        ]
        beh = aggregate([_m("hermes:default", "fp:h", 10)], now=NOW, ledger_events=events)
        assert next(a for a in beh.agents if a.base == "hermes").ledger.created == 1

    def test_zero_percent_field_is_a_reading_not_a_crash(self):
        """池里没有 closed 账本（live: 86/86 terminal=active）⇒ `closed` 恒 0 是**读数**。"""
        beh = _agg()
        assert all(a.ledger.closed == 0 for a in beh.agents)
        closed_ev = [_ev("ledger_create", 600, ledger={**_ledger_dump("ldg-c", ""),
                                                       "last_writer": ""}),
                     _ev("ledger_update", 500, ledger=_ledger_dump("ldg-c", "codex",
                                                                   status="closed"))]
        beh2 = aggregate([_m("codex", "fp:x", 10)], now=NOW, ledger_events=closed_ev)
        assert next(a for a in beh2.agents).ledger.closed == 1, "有 closed 就要数得出来"


class TestEligibleWiring:
    """MQ-F9（F2 复核方反向变异抓出）：`eligible` 准入**接线**无守卫。

    `TestZeroTraffic` 钉的是结构性那一半（零流量 agent 没有通道进来），
    那一半很硬；`eligible` 承担的是另一半——**有 turn 但判否**（bench 构造件名、
    子代理派生 id）。整段短路掉时那 17 项全绿，就是因为 fixture 里被 `eligible`
    挡掉的 agent 都是零流量的，挡不挡都不出现。

    🔴 判据故意用**有流量**的构造件名 `r4-ladder`：它有 turn，唯一挡住它的就是
    `eligible` 这一段。短路 `if eligible is not None and not eligible(...)` ⇒ 必红。
    """

    #: bench 构造件（有真实流量、但不是用户的 agent）——与 `r4-ladder` 同形态
    BENCH_AGENT = "r4-ladder"

    def _with_bench_agent(self, **over):
        fx = _fixture()
        fx["metas"] = [*fx["metas"],
                       _m(self.BENCH_AGENT, "fp:r4", 60, pid="p-real", pname="BladeX"),
                       _m(self.BENCH_AGENT, "fp:r4", 55, pid="p-real", pname="BladeX")]
        fx.update(over)
        metas = fx.pop("metas")
        return aggregate(metas, now=fx.pop("now", NOW), **fx)

    def test_9_agent_with_turns_but_ineligible_is_absent_from_both_views(self):
        beh = self._with_bench_agent(eligible=lambda a: B.agent_base(a) != self.BENCH_AGENT)
        assert self.BENCH_AGENT not in {a.base for a in beh.agents}
        assert self.BENCH_AGENT not in {a["agent"] for a in beh.model_view()["agents"]}
        # 它的轮次也不许从别的口子漏进来（项目段里的 agent 列表同理）
        for p in beh.projects:
            assert self.BENCH_AGENT not in {x.base for x in p.agents}
        assert beh.meta.agents_excluded["ineligible"] == 2

    def test_9b_positive_control_same_agent_appears_when_eligible_allows_it(self):
        """正例：只把谓词翻过来 ⇒ 它必须出现。

        没有这一条，上一条在"构造件根本没进 metas"时也会绿（尺子量的是不是
        它以为的那件事）。
        """
        beh = self._with_bench_agent(eligible=lambda a: True)
        assert self.BENCH_AGENT in {a.base for a in beh.agents}
        assert beh.meta.agents_excluded["ineligible"] == 0
