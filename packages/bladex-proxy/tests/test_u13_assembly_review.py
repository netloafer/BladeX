"""批 U · U1.3 —— ADR-0033 §2.4 / §2.5 / §2.7 两级关联复核 + 判定持久化。

钉的事：
① 候选 = 已闭合 ∧ 带工具结果；open 单元永不进；被比对表示 = user 原文（剥信封）+ conclusion，不含证据。
② 两级开关四种组合（§2.5 表）：embed+model / 只 embed（灰区保留）/ 只 model（全量）/ 都关 = 只观测。
③ 判据④：同会话第二个冷轮起，模型只看**新进灰区**的单元（goal 锚点未变时判过的不再送）。
④ 缓存键含 unit_key（反向变异 4：去掉 ⇒ 两个单元串判定 ⇒ 红）。
⑤ 超时：embed / 模型各自超时 ⇒ 灰区保留 + `assembly_review_timeout`。
⑥ 写穿 Hub：判定随 reconstruction 落 Turn；`MemoryHub.latest_reconstruction` 读回；`hydrate` 恢复。
"""

from __future__ import annotations

import asyncio

import pytest
import structlog
from bladex_core.task_unit import build_task_units
from bladex_proxy import assembly_review as ar
from bladex_proxy.models import ReconstructionRecord

_VEC = {"alpha": [1.0, 0.0, 0.0], "beta": [0.0, 1.0, 0.0], "gamma": [0.77, 0.638, 0.0],
        "delta": [0.76, 0.65, 0.0], "goalword": [0.0, 0.0, 1.0]}


class _Embedder:
    """按关键词给向量：alpha 与 request 余弦 1（留）/ beta 0（裁）/ gamma≈0.77、delta≈0.76（灰区）。"""

    def __init__(self, delay: float = 0.0) -> None:
        self.passages: list[str] = []
        self.queries: list[str] = []
        self.delay = delay

    def _vec(self, text: str) -> list[float]:
        for k, v in _VEC.items():
            if k in text:
                return list(v)
        return [0.1, 0.1, 0.1]

    def embed_passage(self, texts: list[str]) -> list[list[float]]:
        if self.delay:
            import time
            time.sleep(self.delay)
        self.passages.extend(texts)
        return [self._vec(t) for t in texts]

    def embed_query_vec(self, text: str) -> list[float]:
        self.queries.append(text)
        return self._vec(text)


def _turn(word: str, i: int, tools: bool = True) -> list[dict]:
    out = [{"role": "user", "content": f"<system-reminder>noise</system-reminder>{word} task"}]
    if tools:
        out += [{"role": "assistant", "content": "", "tool_calls": [
                    {"id": f"c{i}", "type": "function", "function": {"name": "read", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": f"c{i}", "name": "read", "content": f"EVIDENCE-{word} " * 50}]
    out.append({"role": "assistant", "content": f"{word} done"})
    return out


def _msgs(*words: str, open_word: str = "alpha", untooled: tuple[str, ...] = ()) -> list[dict]:
    m: list[dict] = [{"role": "system", "content": "sys"}]
    for i, w in enumerate(words):
        m += _turn(w, i, tools=w not in untooled)
    m.append({"role": "user", "content": f"{open_word} again"})
    return m


def _cfg(**kw) -> ar.ReviewConfig:
    base = {"embed": True, "model": "", "hi": 0.82, "lo": 0.72}
    base.update(kw)
    return ar.ReviewConfig(**base)


def _run(msgs, *, config, embedder=None, judge=None, goal=None, session="s", rc=None):
    return asyncio.run(ar.review_units(msgs, build_task_units(msgs), session_id=session, config=config,
                                       goal=goal, embedder=embedder, judge=judge,
                                       review_cache=rc or ar.ReviewCache()))


class _Judge:
    def __init__(self, relevant: dict[str, bool] | None = None, delay: float = 0.0) -> None:
        self.calls: list[list[str]] = []
        self.relevant = relevant or {}
        self.delay = delay
        self.gray_flags: list[bool] = []

    async def __call__(self, request, items, goal_text, gray):
        self.calls.append([it["id"] for it in items])
        self.gray_flags.append(gray)
        if self.delay:
            await asyncio.sleep(self.delay)
        return {it["id"]: self.relevant.get(it["id"], False) for it in items}


# ── ① 候选与表示 ──────────────────────────────────────────────────────────


def test_candidates_are_closed_units_with_tool_results():
    msgs = _msgs("alpha", "beta", "gamma", untooled=("gamma",))
    units = build_task_units(msgs)
    cands = ar.review_candidates(units)
    assert [u.unit_index for u in cands] == [0, 1], "无工具结果的 gamma 与 open 单元都不进复核"


def test_representation_is_user_plus_conclusion_without_evidence_or_envelope():
    msgs = _msgs("alpha")
    u = build_task_units(msgs)[0]
    rep = ar.unit_representation(u, msgs)
    assert "alpha task" in rep and "alpha done" in rep
    assert "EVIDENCE" not in rep, "不 embed 证据（§2.4）"
    assert "system-reminder" not in rep and "noise" not in rep, "user 原文先剥信封"


# ── ② 四种组合 ────────────────────────────────────────────────────────────


def test_embed_and_model_gray_goes_to_model():
    msgs = _msgs("alpha", "beta", "gamma")
    j = _Judge(relevant={})
    res = _run(msgs, config=_cfg(model="lite"), embedder=_Embedder(), judge=j)
    by = {build_task_units(msgs)[i].unit_key: w for i, w in enumerate(("alpha", "beta", "gamma"))}
    got = {by[k]: (v.verdict, v.source) for k, v in res.verdicts.items()}
    assert got == {"alpha": ("kept", "embed"), "beta": ("trimmed", "embed"), "gamma": ("trimmed", "model")}
    assert res.mode == "embed+model" and res.model_calls == 1 and res.model_units == 1
    assert j.gray_flags == [True], "embed 开 ⇒ 用灰区 prompt"


def test_embed_only_keeps_gray():
    msgs = _msgs("alpha", "beta", "gamma")
    res = _run(msgs, config=_cfg(model=""), embedder=_Embedder())
    reasons = sorted(v.reason for v in res.verdicts.values())
    assert reasons == ["gray_kept", "hi", "lo"] and res.model_calls == 0
    assert res.mode == "embed"


def test_model_only_reviews_every_candidate_with_full_prompt():
    msgs = _msgs("alpha", "beta", "gamma")
    j = _Judge(relevant={})
    res = _run(msgs, config=_cfg(embed=False, model="lite"), embedder=_Embedder(), judge=j)
    assert res.mode == "model" and res.model_units == 3 and len(j.calls[0]) == 3
    assert j.gray_flags == [False], "embed 关 ⇒ 全量 prompt"
    assert {v.source for v in res.verdicts.values()} == {"model"}


def test_both_off_is_observe_only():
    """判据⑤ 的判定层：都关 ⇒ reviewed=0、不产任何判定（正文逐字不变在 U1.4 的装配层钉）。"""
    msgs = _msgs("alpha", "beta", "gamma")
    emb = _Embedder()
    res = _run(msgs, config=_cfg(embed=False, model=""), embedder=emb)
    assert res.mode == "observe" and res.reviewed == 0 and res.verdicts == {}
    assert emb.passages == [] and emb.queries == [], "只观测：连 embed 都不做"


def test_prompts_differ_for_gray_and_full():
    a = ar.build_judge_prompt("r", "g", [{"id": "x", "text": "t"}], gray=True)[0]["content"]
    b = ar.build_judge_prompt("r", "g", [{"id": "x", "text": "t"}], gray=False)[0]["content"]
    assert a != b and "borderline" in a and "ALL earlier" in b


def test_parse_judge_reply_is_conservative():
    assert ar.parse_judge_reply('```json\n{"units":[{"id":"a","relevant":false}]}\n```') == {"a": False}
    assert ar.parse_judge_reply("not json") == {}
    assert ar.parse_judge_reply('{"units":[{"id":"a","relevant":"no"}]}') == {}, "非布尔不采信（缺 ⇒ 保留）"


def test_model_missing_unit_is_kept():
    msgs = _msgs("alpha", "gamma")

    async def judge(request, items, goal_text, gray):
        return {}

    res = _run(msgs, config=_cfg(model="lite"), embedder=_Embedder(), judge=judge)
    gamma = build_task_units(msgs)[1].unit_key
    assert res.verdicts[gamma].verdict == "kept" and res.verdicts[gamma].reason == "model_missing"


# ── ③ 判据④：第二个冷轮只送新灰区 ──────────────────────────────────────────


def test_second_cold_round_sends_only_new_gray_units_with_goal():
    rc = ar.ReviewCache()
    goal = ar.GoalAnchor(ledger_id="ldg-1", rev=3, text="goalword")
    j = _Judge(relevant={})
    m1 = _msgs("alpha", "beta", "gamma", open_word="alpha")
    r1 = _run(m1, config=_cfg(model="lite"), embedder=_Embedder(), judge=j, goal=goal, rc=rc)
    assert r1.model_units == 1
    # 下一个冷轮：多了一个新的灰区单元 delta，request 也换了
    m2 = _msgs("alpha", "beta", "gamma", "delta", open_word="alpha now")
    r2 = _run(m2, config=_cfg(model="lite"), embedder=_Embedder(), judge=j, goal=goal, rc=rc)
    delta = build_task_units(m2)[3].unit_key
    gamma = build_task_units(m2)[2].unit_key
    assert j.calls[-1] == [delta], "第二个冷轮模型只看新进灰区的 delta"
    assert r2.model_calls == 1 and r2.model_units == 1
    assert r2.verdicts[gamma].source == "frozen" and r2.verdicts[gamma].decided_anchor_key.endswith(goal.key)


def test_goal_rev_change_invalidates_model_reuse():
    rc = ar.ReviewCache()
    j = _Judge(relevant={})
    m = _msgs("alpha", "gamma")
    _run(m, config=_cfg(model="lite"), embedder=_Embedder(), judge=j,
         goal=ar.GoalAnchor("ldg-1", 1, "goalword"), rc=rc)
    _run(m, config=_cfg(model="lite"), embedder=_Embedder(), judge=j,
         goal=ar.GoalAnchor("ldg-1", 2, "goalword"), rc=rc)
    assert len(j.calls) == 2, "goal rev 变 ⇒ 原判定失效、重新送模型"


def test_without_goal_still_gray_unit_is_not_resent():
    rc = ar.ReviewCache()
    j = _Judge(relevant={})
    _run(_msgs("alpha", "gamma", open_word="alpha"), config=_cfg(model="lite"),
         embedder=_Embedder(), judge=j, rc=rc)
    _run(_msgs("alpha", "gamma", open_word="alpha two"), config=_cfg(model="lite"),
         embedder=_Embedder(), judge=j, rc=rc)
    assert len(j.calls) == 1, "没有 goal：仍在灰区且模型判过 ⇒ 复用，不是新进灰区"


def test_unit_vectors_are_cached_per_unit_key():
    rc = ar.ReviewCache()
    emb = _Embedder()
    _run(_msgs("alpha", "beta"), config=_cfg(), embedder=emb, rc=rc)
    n = len(emb.passages)
    _run(_msgs("alpha", "beta", "beta2"), config=_cfg(), embedder=emb, rc=rc)
    assert len(emb.passages) == n + 1, "只 embed 新单元（单元向量按 (session, unit_key) 永久缓存）"


# ── ④ 反向变异 4 的判别用例 ─────────────────────────────────────────────────


def test_reused_verdicts_stay_per_unit():
    """同锚点第二次复核全走复用；复用出来的判定必须逐单元对得上（键去掉 unit_key 就会串）。"""
    rc = ar.ReviewCache()
    goal = ar.GoalAnchor("ldg-9", 1, "goalword")
    m = _msgs("alpha", "beta", "gamma")
    j = _Judge(relevant={})
    r1 = _run(m, config=_cfg(model="lite"), embedder=_Embedder(), judge=j, goal=goal, rc=rc)
    r2 = _run(m, config=_cfg(model="lite"), embedder=_Embedder(), judge=j, goal=goal, rc=rc)
    assert {k: v.verdict for k, v in r2.verdicts.items()} == {k: v.verdict for k, v in r1.verdicts.items()}
    assert {v.verdict for v in r2.verdicts.values()} == {"kept", "trimmed"}
    assert all(v.source == "frozen" for v in r2.verdicts.values()) and len(j.calls) == 1


# ── ⑤ 超时 ────────────────────────────────────────────────────────────────


def test_model_timeout_keeps_gray_and_logs():
    events: list[dict] = []

    def _sink(logger, method, ed):
        events.append(dict(ed))
        raise structlog.DropEvent

    structlog.configure(processors=[_sink])
    try:
        res = _run(_msgs("alpha", "gamma"), config=_cfg(model="lite", model_budget_ms=50, cold_budget_ms=5000),
                   embedder=_Embedder(), judge=_Judge(delay=0.5))
    finally:
        structlog.reset_defaults()
    assert res.timeout == "model"
    assert [v.reason for v in res.verdicts.values() if v.source == "model"] == ["timeout_kept"]
    assert any(e.get("event") == "assembly_review_timeout" and e.get("stage") == "model" for e in events)


def test_embed_timeout_treats_all_as_gray():
    res = _run(_msgs("alpha", "beta"), config=_cfg(model="", embed_budget_ms=20),
               embedder=_Embedder(delay=0.3))
    assert res.timeout == "embed"
    assert {v.verdict for v in res.verdicts.values()} == {"kept"}, "embed 超时 ⇒ 全灰区 ⇒ 无模型时全保留"


# ── ⑥ 写穿 Hub + 重启恢复 ────────────────────────────────────────────────


def test_records_validate_and_hydrate_back():
    rc = ar.ReviewCache()
    j = _Judge(relevant={})
    goal = ar.GoalAnchor("ldg-1", 1, "goalword")
    m = _msgs("alpha", "beta", "gamma")
    r1 = _run(m, config=_cfg(model="lite"), embedder=_Embedder(), judge=j, goal=goal, rc=rc)
    recs = [v.as_record() for v in r1.verdicts.values()]
    rr = ReconstructionRecord.model_validate({"verdicts": recs, "review": {"mode": r1.mode}})
    assert {v.source for v in rr.verdicts} == {"embed", "model"}
    assert all(v.anchor_key for v in rr.verdicts), "原则 13：每条判定都说得出对着哪个锚点判的"
    # 进程重启：新缓存从记录读回 ⇒ 同锚点再判全部复用，模型不再被调用
    rc2 = ar.ReviewCache()
    assert rc2.hydrate("s", [r.model_dump() for r in rr.verdicts]) == 3
    r2 = _run(m, config=_cfg(model="lite"), embedder=_Embedder(), judge=j, goal=goal, rc=rc2)
    assert len(j.calls) == 1 and all(v.source == "frozen" for v in r2.verdicts.values())


def test_hub_latest_reconstruction_reads_back(tmp_path):
    from bladex_proxy.models import Identity, Turn
    from bladex_proxy.storage.memory_hub import MemoryHub

    hub = MemoryHub(tmp_path / "hub")
    hub.open()
    try:
        ident = Identity(user_id="u", agent_id="a", session_id="s1")
        rec = {"layer": "A", "verdicts": [{"unit_key": "k1", "verdict": "trimmed", "source": "embed",
                                          "anchor_key": "req:x"}], "review": {"mode": "embed"}}
        hub.put("u/a/s1/0000000000001-0", Turn(identity=ident, request_messages=[{"role": "user", "content": "q"}],
                                               reconstruction=ReconstructionRecord.model_validate(rec)))
        hub.put("u/a/s1/0000000000002-0", Turn(identity=ident, request_messages=[{"role": "user", "content": "q2"}]))
        hub.put("u/a/s2/0000000000003-0", Turn(identity=ident, request_messages=[{"role": "user", "content": "x"}]))
        got = hub.latest_reconstruction("u/a/s1/")
        assert got is not None and got["verdicts"][0]["unit_key"] == "k1"
        assert hub.latest_reconstruction("u/a/s2/") is None
        assert hub.latest_reconstruction("u/a/nope/") is None
    finally:
        hub.close()


@pytest.fixture(autouse=True)
def _clean_global_cache():
    ar.cache().clear()
    yield
    ar.cache().clear()


# ── ⑦ 编排接线：goal 锚点只取**本会话**绑着的账本；判定读回走 Hub ─────────────


def test_orchestration_goal_anchor_uses_session_owned_ledger():
    from types import SimpleNamespace

    from bladex_core.ledger import LedgerEntry, new_ledger
    from bladex_proxy.server.orchestration import _assembly_goal_anchor, _assembly_verdict_loader

    led = new_ledger(ledger_id="ldg-7", title="t", goal="修好并发写")
    led.rev = 4
    led.sections["core"] = [LedgerEntry(text="rev 乐观锁")]
    seen: list[tuple] = []

    class _Agency:
        def session_active_ledger(self, agent_id, project_id, session_id):
            seen.append((agent_id, project_id, session_id))
            return led if session_id == "mine" else None

    ident = SimpleNamespace(agent_id="claude-code", project_id="p1", session_id="mine",
                            session_prefix=lambda: "local/claude-code/mine/")
    req = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(agency=_Agency(), hub=None)))
    g = _assembly_goal_anchor(req, ident)
    assert g == ar.GoalAnchor("ldg-7", 4, "修好并发写\nrev 乐观锁") and g.key == "goal:ldg-7@4"
    ident.session_id = "inherited"
    assert _assembly_goal_anchor(req, ident) is None, "继承来的绑定 ⇒ 只用 request 锚点"
    assert _assembly_verdict_loader(req, ident) is None, "没有 Hub ⇒ 不读回"

    class _Hub:
        def latest_reconstruction(self, prefix):
            return {"verdicts": [], "prefix": prefix}

    req.app.state.hub = _Hub()
    assert _assembly_verdict_loader(req, ident)()["prefix"] == "local/claude-code/mine/"
