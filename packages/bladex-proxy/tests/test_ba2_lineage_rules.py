"""批 BA2（MQ-A94）：`subtask_lineage` 的规则用例（不依赖 Hub 夹具）。回放集逐行见 `test_ba2_lineage.py`。"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from bladex_proxy import identity as ID
from bladex_proxy.project_identity import ProjectIdentity
from bladex_proxy.subtask_lineage import (
    LedgerOp,
    SessionLineage,
    SubtaskLineage,
    active_at_spawn,
    spawn_texts,
)
from structlog.testing import capture_logs


def _sub(session: str, *, agent: str = "dsh", project: str = "") -> SimpleNamespace:
    return SimpleNamespace(user_id="local", agent_id=agent, session_id=session, project_id=project,
                           project_name="", project_source="", project_root="",
                           parent_ledger_id="", project_inherited="")


PROMPT = "please count the python source lines under bladex_core and report back"


def _call(prompt: str = PROMPT, name: str = "subagent") -> dict:
    return {"tool_name": name, "direction": "call", "tool_call_id": "c1", "arguments": {"prompt": prompt}}


def _msgs(prompt: str = PROMPT) -> list[dict]:
    return [{"role": "user", "content": f"Task:\n{prompt}\n"}]


def _mother(session: str, project: str, *, agent: str = "dsh") -> SimpleNamespace:
    return SimpleNamespace(user_id="local", agent_id=agent, session_id=session, project_id=project,
                           project_name="n", project_source="git-root", project_root="/r")


def test_ba2_child_cwd_beats_unique_mother_project() -> None:
    """自报 cwd 优先于一切继承：唯一母候选的项目不同也不覆盖。"""
    lin = SubtaskLineage()
    lin.note_spawns(_mother("m", "p-mother"), [_call()], ledger_ops=[], active_now="ldg-m", now=100.0)
    idn = _sub("c", project="p-own")
    lin.apply(idn, ProjectIdentity("p-own", cwd_declared=True), _msgs(), is_subtask=True, now=101.0)
    assert idn.project_id == "p-own" and idn.project_inherited == ""
    assert lin.parent_for("dsh", "c") == "ldg-m"


def test_ba2_r1_unique_mother_project_when_child_has_no_cwd() -> None:
    lin = SubtaskLineage()
    lin.note_spawns(_mother("m", "p-mother"), [_call()], ledger_ops=[], active_now="ldg-m", now=100.0)
    idn = _sub("c")
    with capture_logs() as logs:
        lin.apply(idn, ProjectIdentity(), _msgs(), is_subtask=True, now=101.0)
    assert idn.project_id == "p-mother" and idn.project_inherited == "mother_unique"
    assert any(r["event"] == "subtask_project_inherited" and r["rule"] == "mother_unique" for r in logs)


def test_ba2_r1_declared_unusable_cwd_does_not_inherit() -> None:
    """请求带了 cwd 声明（哪怕不可用）⇒ 不继承：用户说了自己在哪，不替他改。"""
    lin = SubtaskLineage()
    lin.note_spawns(_mother("m", "p-mother"), [_call()], ledger_ops=[], active_now="ldg-m", now=100.0)
    idn = _sub("c")
    lin.apply(idn, ProjectIdentity(cwd_declared=True), _msgs(), is_subtask=True, now=101.0)
    assert idn.project_id == ""


def test_ba2_r1_ambiguous_mothers_without_cwd_stays_global() -> None:
    lin = SubtaskLineage()
    lin.note_spawns(_mother("m1", "p-1"), [_call()], ledger_ops=[], active_now="ldg-1", now=100.0)
    lin.note_spawns(_mother("m2", "p-2"), [_call()], ledger_ops=[], active_now="ldg-2", now=100.0)
    idn = _sub("c")
    lin.apply(idn, ProjectIdentity(), _msgs(), is_subtask=True, now=101.0)
    assert idn.project_id == "" and idn.parent_ledger_id == ""


def test_ba2_main_session_never_looks_for_a_mother() -> None:
    lin = SubtaskLineage()
    lin.note_spawns(_mother("m", "p-mother"), [_call()], ledger_ops=[], active_now="ldg-m", now=100.0)
    idn = _sub("main")
    lin.apply(idn, ProjectIdentity(), _msgs(), is_subtask=False, now=101.0)
    assert idn.project_id == "" and idn.parent_ledger_id == ""


@pytest.mark.parametrize(("kw", "why"), [
    ({"now": 100.0 + 25 * 3600}, "过期"),
    ({"now": 50.0}, "早于派出"),
])
def test_ba2_spawn_outside_window_is_not_a_candidate(kw: dict, why: str) -> None:
    lin = SubtaskLineage()
    lin.note_spawns(_mother("m", "p"), [_call()], ledger_ops=[], active_now="l", now=100.0)
    assert lin.find_mothers(_sub("c"), _msgs(), **kw) == [], why


def test_ba2_other_user_and_other_agent_are_not_candidates() -> None:
    lin = SubtaskLineage()
    lin.note_spawns(_mother("m", "p"), [_call()], ledger_ops=[], active_now="l", now=100.0)
    other_user = _sub("c")
    other_user.user_id = "someone-else"
    assert lin.find_mothers(other_user, _msgs(), now=101.0) == []
    assert lin.find_mothers(_sub("c", agent="hermes:default"), _msgs(), now=101.0) == []
    # 同一 agent 的命名侧面算（codex → codex:collab_spawn）
    lin2 = SubtaskLineage()
    lin2.note_spawns(_mother("m", "p", agent="codex"), [_call(name="ns-multi_agent_v1-spawn_agent")],
                     ledger_ops=[], active_now="l", now=100.0)
    assert len(lin2.find_mothers(_sub("m", agent="codex:collab_spawn"), _msgs(), now=101.0)) == 1


def test_ba2_short_prompt_and_non_spawn_tool_are_not_evidence() -> None:
    lin = SubtaskLineage()
    assert lin.note_spawns(_mother("m", "p"), [_call(name="Bash")], ledger_ops=[], active_now="l") == []
    assert spawn_texts({"prompt": "go on"}) == []
    assert spawn_texts({"tasks": [{"goal": PROMPT}, {"goal": "x"}]}) == [PROMPT]


def _rt():
    from bladex_proxy.agency import AgencyRuntime
    return AgencyRuntime()


def _dispatch(rt, name: str, args: dict, *, session: str = "c", agent: str = "dsh", project: str = "p-1") -> str:  # noqa: ANN001
    import asyncio

    ctx = {"session_id": session, "agent_id": agent, "project_id": project, "turn_index": 1,
           "user_query": "继续"}
    return asyncio.run(rt.toolface.dispatch(name, args, allowed_exposure="internal", context=ctx))


def _with_lineage(monkeypatch, lin: SubtaskLineage) -> None:  # noqa: ANN001
    import bladex_proxy.subtask_lineage as SL
    monkeypatch.setattr(SL, "lineage", lin)


def _new_ledger(rt, title: str, **kw) -> str:  # noqa: ANN001, ANN003
    out = _dispatch(rt, "bladex_ledger_switch", {"ledger_id": "", "title": title, "goal": title}, **kw)
    assert "Error" not in out, out
    return max(rt.pool.values(), key=lambda x: x.created_at + x.ledger_id
               if x.title != title else "~" + x.ledger_id).ledger_id


def test_ba2_case_child_ledger_gets_parent_automatically(monkeypatch) -> None:  # noqa: ANN001
    """病例：`ldg-5c6347e100fb` 这种子本建出来就挂母本，不靠模型自填。"""
    lin = SubtaskLineage()
    _with_lineage(monkeypatch, lin)
    rt = _rt()
    parent = _new_ledger(rt, "母本", session="m")
    lin._sessions[("dsh", "c")] = SessionLineage(parent_ledger_id=parent, mother_checked=True)
    with capture_logs() as logs:
        child = _new_ledger(rt, "子本", session="c", project="")
    assert rt.pool[child].parent_ledger_id == parent
    assert any(r["event"] == "ledger_parent_attached" and r["parent"] == parent for r in logs)


def test_ba2_s6_child_scope_active_ledger_is_not_the_parent(monkeypatch) -> None:  # noqa: ANN001
    """S6：母会话派出时无激活本 ⇒ 不挂；子任务当时 scope 上有激活本也不算（「现行形态」的错）。"""
    lin = SubtaskLineage()
    _with_lineage(monkeypatch, lin)
    rt = _rt()
    _new_ledger(rt, "别人的本", session="other")          # 同 (agent, project) scope 上有激活本
    lin._sessions[("dsh", "c")] = SessionLineage(parent_ledger_id="", mother_checked=True)
    child = _new_ledger(rt, "子本", session="c")
    assert rt.pool[child].parent_ledger_id == ""


def test_ba2_main_session_ledger_has_no_auto_parent(monkeypatch) -> None:  # noqa: ANN001
    """MQ-L1 不回退：主会话建本时「当时激活的那本」不是父。"""
    _with_lineage(monkeypatch, SubtaskLineage())
    rt = _rt()
    _new_ledger(rt, "第一本", session="main")
    second = _new_ledger(rt, "第二本", session="main")
    assert rt.pool[second].parent_ledger_id == ""


def test_ba2_explicit_parent_is_not_overridden(monkeypatch) -> None:  # noqa: ANN001
    lin = SubtaskLineage()
    _with_lineage(monkeypatch, lin)
    rt = _rt()
    a = _new_ledger(rt, "A", session="x")
    b = _new_ledger(rt, "B", session="x")
    lin._sessions[("dsh", "c")] = SessionLineage(parent_ledger_id=a, mother_checked=True)
    out = _dispatch(rt, "bladex_ledger_switch",
                    {"ledger_id": "", "title": "C", "goal": "C", "parent_ledger_id": b}, session="c")
    assert "Error" not in out
    c = next(x for x in rt.pool.values() if x.title == "C")
    assert c.parent_ledger_id == b


def test_ba2_auto_parent_gone_from_pool_is_skipped_with_a_log(monkeypatch) -> None:  # noqa: ANN001
    lin = SubtaskLineage()
    _with_lineage(monkeypatch, lin)
    rt = _rt()
    lin._sessions[("dsh", "c")] = SessionLineage(parent_ledger_id="ldg-gone", mother_checked=True)
    with capture_logs() as logs:
        child = _new_ledger(rt, "子本", session="c")
    assert rt.pool[child].parent_ledger_id == ""
    assert any(r["event"] == "ledger_parent_skipped" for r in logs)


# ── R3 位置计算：运行时真的按调用序记 ─────────────────────────────────

def test_ba2_active_at_spawn_positions() -> None:
    ops = [LedgerOp("A", ""), LedgerOp("", "B")]
    assert active_at_spawn(ops, 2, "B") == "A"      # 全在 spawn 之后
    assert active_at_spawn(ops, 1, "B") == ""       # close 在前、switch 在后
    assert active_at_spawn(ops, 0, "B") == "B"      # 全在 spawn 之前
    assert active_at_spawn([], 0, "Z") == "Z"       # 本轮没动过


def test_ba2_runtime_records_ops_in_call_order_including_failures() -> None:
    rt = _rt()
    a = _new_ledger(rt, "A", session="m")
    rt.drain_ledger_ops("dsh", "m", "p-1")
    # 失败的 switch 也占一个位置：tool_events 里它同样是一次调用，少记就错位。
    assert "Error" in _dispatch(rt, "bladex_ledger_switch", {"ledger_id": "ldg-nope"}, session="m")
    _dispatch(rt, "bladex_ledger_close", {"ledger_id": a}, session="m")
    ops, active = rt.drain_ledger_ops("dsh", "m", "p-1")
    assert [(o.before, o.after) for o in ops] == [(a, a), (a, "")]
    assert active == ""
    assert rt.drain_ledger_ops("dsh", "m", "p-1")[0] == []


def test_ba2_s9_shape_end_to_end_through_runtime() -> None:
    rt = _rt()
    lin = SubtaskLineage()
    a = _new_ledger(rt, "A", session="m")
    rt.drain_ledger_ops("dsh", "m", "p-1")
    _dispatch(rt, "bladex_ledger_close", {"ledger_id": a}, session="m")
    _new_ledger(rt, "B", session="m")
    events = [_call(name="Agent"), {"tool_name": "bladex_ledger_close", "direction": "call"},
              {"tool_name": "bladex_ledger_switch", "direction": "call"}]
    ops, active = rt.drain_ledger_ops("dsh", "m", "p-1")
    assert active != a
    out = lin.note_spawns(_mother("m", "p-1"), events, ledger_ops=ops, active_now=active, now=1.0)
    assert out[0]["ledger_id"] == a


# ── R2 ───────────────────────────────────────────────────────────────────

def test_ba2_r2_sticky_only_when_no_cwd_declaration() -> None:
    lin = SubtaskLineage()
    first = _sub("s", project="p-1")
    lin.apply(first, ProjectIdentity("p-1", cwd_declared=True), [], is_subtask=False)
    silent = _sub("s")
    lin.apply(silent, ProjectIdentity(), [], is_subtask=False)
    assert silent.project_id == "p-1" and silent.project_inherited == "session_sticky"
    moved = _sub("s")                       # 用户 cd 到家目录：有声明、不是项目 ⇒ 不粘
    lin.apply(moved, ProjectIdentity(cwd_declared=True), [], is_subtask=False)
    assert moved.project_id == ""


def test_ba2_r2_sticky_follows_latest_declaration() -> None:
    lin = SubtaskLineage()
    for pid in ("p-1", "p-2"):
        lin.apply(_sub("s", project=pid), ProjectIdentity(pid, cwd_declared=True), [], is_subtask=False)
    silent = _sub("s")
    lin.apply(silent, ProjectIdentity(), [], is_subtask=False)
    assert silent.project_id == "p-2"


def test_ba2_r2_time_window_bucket_is_not_a_conversation() -> None:
    """`tw:` 是 5 分钟桶（回放集 N4 两个会话同桶）⇒ 不粘。"""
    lin = SubtaskLineage()
    lin.apply(_sub("tw:a", project="p-1"), ProjectIdentity("p-1", cwd_declared=True), [], is_subtask=False)
    silent = _sub("tw:a")
    lin.apply(silent, ProjectIdentity(), [], is_subtask=False)
    assert silent.project_id == ""


def test_ba2_r2_other_session_does_not_inherit() -> None:
    lin = SubtaskLineage()
    lin.apply(_sub("s1", project="p-1"), ProjectIdentity("p-1", cwd_declared=True), [], is_subtask=False)
    other = _sub("s2")
    lin.apply(other, ProjectIdentity(), [], is_subtask=False)
    assert other.project_id == ""


# ── 重启 ─────────────────────────────────────────────────────────────────

def _record(**kw) -> ID.SessionReplayRecord:  # noqa: ANN003
    base = {"ts": 100.0, "user_id": "local", "agent_id": "dsh", "session_id": "s", "messages": [],
            "request_headers": {}, "response_text": "", "tool_events": [], "aux_source": ""}
    return ID.SessionReplayRecord(**{**base, **kw})


def test_ba2_rebuild_restores_session_project_parent_and_mothers(monkeypatch) -> None:  # noqa: ANN001
    now = 200.0
    lin = SubtaskLineage()
    n = lin.rebuild_lineage([
        _record(session_id="child", project_id="p-1", project_name="n", project_source="git-root",
                project_root="/r", parent_ledger_id="ldg-m"),
        _record(session_id="m", project_id="p-1", tool_events=[_call()],
                spawn_dispatches=[{"tool_call_id": "c1", "tool_name": "subagent", "ledger_id": "ldg-x"}]),
    ])
    assert n == 1
    assert lin.parent_for("dsh", "child") == "ldg-m"
    silent = _sub("child")
    lin.apply(silent, ProjectIdentity(), [], is_subtask=False, now=now)
    assert silent.project_id == "p-1" and silent.parent_ledger_id == "ldg-m"
    fresh_child = _sub("c2")
    lin.apply(fresh_child, ProjectIdentity(), _msgs(), is_subtask=True, now=now)
    assert fresh_child.parent_ledger_id == "ldg-x" and fresh_child.project_id == "p-1"


def test_ba2_rebuild_does_not_overwrite_live_state() -> None:
    lin = SubtaskLineage()
    lin.apply(_sub("s", project="p-live"), ProjectIdentity("p-live", cwd_declared=True), [], is_subtask=False)
    lin.rebuild_lineage([_record(session_id="s", project_id="p-old")])
    silent = _sub("s")
    lin.apply(silent, ProjectIdentity(), [], is_subtask=False)
    assert silent.project_id == "p-live"


def test_ba2_tool_call_string_arguments_parse() -> None:
    lin = SubtaskLineage()
    te = {"tool_name": "Agent", "direction": "call", "tool_call_id": "c",
          "arguments": json.dumps({"prompt": PROMPT})}
    lin.note_spawns(_mother("m", "p"), [te], ledger_ops=[], active_now="l", now=1.0)
    assert len(lin.find_mothers(_sub("c"), _msgs(), now=2.0)) == 1
