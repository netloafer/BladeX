"""批 BB · 2b · consolidator 接线：关着零动作、不回扫存量、D8 整链（真 `rebuild_from_hub`）、端到端冒烟（假上游）。

端到端冒烟口径：**假上游**（`FakeJudge`，调用数可数），其余全是真的——真 MemoryIndex（bigram 假 embedder，
真 LanceDB / RocksDB）、真 proxy 应用（`TestClient`，写 Hub 的唯一出口）、真 `ProxySink` / `IndexContradictionStore` /
`run_pass` / CLI 命令（`_admin_call` 接到 TestClient）。全部在 tmp 目录，不碰 live。
"""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from _bb_support import FakeJudge
from bladex_core import consistency as c
from bladex_core.fact import Fact
from bladex_proxy import contradiction as contra
from bladex_proxy.cli import memory_cmds
from bladex_proxy.config import ProxyConfig
from bladex_proxy.models import AdminEventType
from bladex_proxy.server import create_app
from bladex_proxy.storage.contradiction_store import IndexContradictionStore
from bladex_proxy.storage.memory_hub import MemoryHub
from bladex_proxy.storage.memory_index import MemoryIndex
from fastapi.testclient import TestClient
from test_aa_rebuild_reliability import _make_hub, _rebuild_with, _stubs
from typer.testing import CliRunner

_NS = _stubs()
_EMB = _NS["BigramEmbedder"]()


def _fact(fid: str, text: str, *, t: datetime, created: datetime) -> Fact:
    return Fact(id=fid, content=text, subject="Redis 端口", attribute="端口", entities=["Redis", "端口"],
                t_valid=t, created_at=created, audience="all", embedding=_EMB.embed([text])[0])


def _index(path: Path) -> MemoryIndex:
    ix = MemoryIndex(str(path), embedder=_EMB, embed_model_id=_EMB.model_identity)
    ix.open()
    return ix


class _Sink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    def append_flag(self, pair_key: str, payload: dict[str, Any]) -> None:
        self.events.append((pair_key, payload))


def _store_keys(ix: MemoryIndex) -> list[str]:
    return sorted(k for p in ("contradiction/", "contradiction_judged/", "contradiction_state/")
                  for k, _ in ix.contradiction_kv_scan(p))


# ── 关着零动作 / 不回扫存量 ───────────────────────────────────────────────


def test_pass_is_a_noop_when_switch_is_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(c.ENABLED_FLAG, raising=False)
    ix = _index(tmp_path / "ix")
    now = datetime.now(UTC)
    ix.add_fact(_fact("old", "Redis 端口是 6379", t=now - timedelta(days=2), created=now - timedelta(days=2)))
    ix.add_fact(_fact("new", "Redis 端口是 6380", t=now, created=now + timedelta(seconds=1)))
    judge, sink = FakeJudge(), _Sink()
    det = contra.build_detector(ix, judge=judge, sink=sink)
    s = contra.run_pass(ix, det, pass_started=now)
    assert s.skipped == "disabled" and (judge.calls, sink.events, _store_keys(ix)) == (0, [], [])
    ix.close()


def test_switch_off_mutant_in_run_pass_is_red(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """去掉 run_pass 第一行的开关判断，同一场景下游标会被写进 Index——守卫必须能看见。"""
    monkeypatch.delenv(c.ENABLED_FLAG, raising=False)
    src = Path(contra.__file__).read_text(encoding="utf-8")
    anchor = "    if not flag_enabled(cs.ENABLED_FLAG):\n        out.skipped = \"disabled\"\n        return out\n"
    assert src.count(anchor) == 1
    import sys
    import types

    mod = types.ModuleType("bladex_proxy._contradiction_mutant")
    mod.__file__ = contra.__file__
    mod.__package__ = "bladex_proxy"
    sys.modules[mod.__name__] = mod
    exec(compile(src.replace(anchor, ""), contra.__file__, "exec"), mod.__dict__)  # noqa: S102 —— 反向变异
    ix = _index(tmp_path / "ix")
    now = datetime.now(UTC)
    mod.run_pass(ix, mod.build_detector(ix, judge=FakeJudge(), sink=_Sink(), enabled=lambda: False), pass_started=now)
    with pytest.raises(AssertionError):
        assert _store_keys(ix) == [], "去掉开关判断，关着也写了游标，必须红"
    ix.close()


def test_first_pass_does_not_backfill_existing_facts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(c.ENABLED_FLAG, "1")
    ix = _index(tmp_path / "ix")
    now = datetime.now(UTC)
    for i in range(5):  # 存量：全部在 pass 开始之前创建
        ix.add_fact(_fact(f"e{i}", f"Redis 端口是 63{70 + i}", t=now - timedelta(days=9 - i),
                          created=now - timedelta(days=9 - i)))
    judge = FakeJudge("不矛盾")
    s = contra.run_pass(ix, contra.build_detector(ix, judge=judge, sink=_Sink()), pass_started=now)
    assert s.new_facts == 0 and judge.calls == 0
    assert IndexContradictionStore(ix).cursor() == now.isoformat()
    ix.close()


def test_rebuilding_pass_judges_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(c.ENABLED_FLAG, "1")
    ix = _index(tmp_path / "ix")
    now = datetime.now(UTC)
    ix.add_fact(_fact("a", "Redis 端口是 6379", t=now - timedelta(days=1), created=now - timedelta(days=1)))
    ix.add_fact(_fact("b", "Redis 端口是 6380", t=now, created=now + timedelta(seconds=1)))
    judge = FakeJudge()
    s = contra.run_pass(ix, contra.build_detector(ix, judge=judge, sink=_Sink()), pass_started=now, rebuilding=True)
    assert s.skipped == "rebuild" and judge.calls == 0 and _store_keys(ix) == []
    ix.close()


# ── 端到端冒烟（假上游）──────────────────────────────────────────────────


@pytest.fixture()
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(c.ENABLED_FLAG, "1")
    d = tempfile.mkdtemp(prefix="bb2_smoke_")
    ix = _index(Path(d) / "index")                      # consolidator 侧：Index 主写者
    now = datetime.now(UTC)
    ix.add_fact(_fact("old1", "Redis 端口是 6379", t=now - timedelta(days=3), created=now - timedelta(days=3)))
    ix.add_fact(_fact("old2", "Jason 喜欢喝茶", t=now - timedelta(days=2), created=now - timedelta(days=2)))
    cfg = ProxyConfig(upstream_model="openai/test", upstream_api_key="sk-fake",
                      rocksdb_path=f"{d}/rocksdb", index_path=f"{d}/index")
    app = create_app(cfg)
    with TestClient(app) as cl:
        yield {"ix": ix, "client": cl, "now": now, "rocksdb": f"{d}/rocksdb"}
    ix.close()


def _requester(cl: TestClient):
    def _req(method: str, path: str, body: dict | None):
        r = cl.request(method, path, json=body)
        return r.status_code, r.json()

    return _req


def _replay(ix: MemoryIndex, rocksdb: str) -> int:
    hub = MemoryHub(rocksdb, secondary=True)
    hub.open()
    hub.catch_up()
    try:
        return ix._replay_new_admin_events(hub)
    finally:
        hub.close()


def test_smoke_flag_list_settle_and_no_rejudge(world: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    ix, cl, now = world["ix"], world["client"], world["now"]
    judge = FakeJudge("矛盾")
    det = contra.build_detector(ix, judge=judge, sink=contra.ProxySink(_requester(cl)))

    # 1) pass 开始 → 新 fact 落库（明显矛盾 + 一条无关）→ pass 尾部
    started = now - timedelta(seconds=1)
    ix.add_fact(_fact("new1", "Redis 端口是 6380", t=now, created=now))
    s = contra.run_pass(ix, det, pass_started=started)
    assert s.new_facts == 1 and 1 <= judge.calls <= 20 and s.flagged >= 1
    flagged_pairs = [it for _k, it in ix.contradiction_kv_scan("contradiction_judged/")]
    assert flagged_pairs, "判定台账应有留痕（含理由全文）"

    # 2) Hub 里恰好是 kind=flag 的 audience_set；consolidator 下一次增量重放物化进 Index
    evs = [e for _k, e in cl.app.state.hub.scan_admin_events()]
    assert evs and all(e.event_type == AdminEventType.AUDIENCE_SET and e.payload["kind"] == c.KIND_FLAG for e in evs)
    assert _replay(ix, world["rocksdb"]) == len(evs)

    # 3) memory conflicts 看得到（CLI → 端点）
    monkeypatch.setattr(memory_cmds, "_admin_call", lambda m, p, body=None, params=None: _via(cl, m, p, body, params))
    out = CliRunner().invoke(memory_cmds.memory_app, ["conflicts"])
    assert out.exit_code == 0 and "new1|old1" in out.output.replace("old1|new1", "new1|old1") and "unsettled" in out.output
    pair = next(it["pair_id"] for it in cl.get("/admin/contradictions").json()["items"] if "old1" in it["pair_id"])

    # 4) memory settle → 只追加 verdict 事件；重放后状态变更
    calls_before = judge.calls
    out = CliRunner().invoke(memory_cmds.memory_app, ["settle", pair, "--verdict", "not_same_thing", "--note", "不是一回事"])
    assert out.exit_code == 0, out.output
    _replay(ix, world["rocksdb"])
    cl.app.state.index._catchup_throttle_s = 0  # proxy 侧是 secondary，追新本有秒级节流；测试里取消
    assert cl.get("/admin/contradictions", params={"status": "not_same_thing"}).json()["total"] == 1
    assert not list(cl.app.state.hub.scan_tombstones())

    # 5) 再次 pass / 再次处理同一条 fact：不重判（已裁定 + 台账留痕）
    new1 = ix.get_fact("new1")
    r = det.process(new1, ix.contradiction_neighbors(new1))
    assert r.judged == 0 and judge.calls == calls_before
    s2 = contra.run_pass(ix, det, pass_started=started)
    assert s2.new_facts == 0 and judge.calls == calls_before


def _via(cl: TestClient, method: str, path: str, body: dict | None, params: dict | None):
    r = cl.request(method, path, json=body, params={k: v for k, v in (params or {}).items() if v not in ("", None)})
    return r.status_code, r.json()


def test_smoke_proxy_down_keeps_pending_and_a_later_pass_resumes(
        world: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    ix, now = world["ix"], world["now"]
    down = {"on": True}

    def req(method: str, path: str, body: dict | None):
        if down["on"]:
            raise ConnectionRefusedError("proxy down")
        return _requester(world["client"])(method, path, body)

    judge = FakeJudge("矛盾")
    det = contra.build_detector(ix, judge=judge, sink=contra.ProxySink(req))
    ix.add_fact(_fact("new1", "Redis 端口是 6380", t=now, created=now))
    s = contra.run_pass(ix, det, pass_started=now - timedelta(seconds=1))
    assert s.flagged == 0 and s.pending == 1 and IndexContradictionStore(ix).ids() == ["new1"]
    down["on"] = False
    s2 = contra.run_pass(ix, det, pass_started=now)
    assert s2.flagged >= 1 and s2.pending == 0
    assert ix.get_fact("new1") is not None


# ── D8 整链：真 rebuild_from_hub ──────────────────────────────────────────


def test_d8_full_chain_real_rebuild_judges_nothing_and_flag_set_is_identical(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(c.ENABLED_FLAG, "1")
    hub_dir = _make_hub(tmp_path / "hub", n=6)
    stub = _NS["Stub"]()
    judge_prompts: list[str] = []

    def completion(**kw: Any) -> Any:
        msgs = "\n".join(str(m.get("content", "")) for m in kw["messages"])
        if "记忆一致性审查员" in msgs:
            judge_prompts.append(msgs)
        return stub.completion(**kw)

    idx = _rebuild_with(tmp_path, hub_dir, "index", stub, completion, monkeypatch)
    facts = sorted(idx.all_facts(), key=lambda f: f.id)
    assert len(facts) >= 2, "夹具应蒸出 fact"
    a, b = facts[0], facts[1]

    # 往 Hub 写一条标记 + 一条裁定，并在 Index 里留一条判定台账（带全文理由）
    h = MemoryHub(hub_dir)
    h.open()
    lo, hi = sorted((a, b), key=lambda f: f.id)
    h.append_admin_event(AdminEventType.AUDIENCE_SET, "", c.pair_key(a.id, b.id), **c.build_flag_payload(
        hi, lo, judge_model="fake/judge", reason="理由", rank=1, cos=0.9))
    h.append_admin_event(AdminEventType.AUDIENCE_SET, "", c.pair_key(a.id, b.id), **c.build_verdict_payload(
        (lo.id, hi.id), (c.content_sha(lo), c.content_sha(hi)), verdict="confirmed", origin="cli"))
    h.close()
    store = IndexContradictionStore(idx)
    store.record(c.pair_key(a.id, b.id), (c.content_sha(lo), c.content_sha(hi)), {"verdict": "矛盾", "reason": "全文理由"})

    def _snapshot() -> dict[str, Any]:
        return {"flags": sorted((k, json.dumps(v, sort_keys=True)) for k, v in idx.contradiction_kv_scan("contradiction/")),
                "judged": sorted((k, json.dumps(v, sort_keys=True))
                                 for k, v in idx.contradiction_kv_scan("contradiction_judged/"))}

    assert idx.contradiction_kv_scan("contradiction/") == [], "写事件之前物化视图为空：基线只能来自重放"
    # 第一次全量：把 Hub 事件重放进物化视图，作为「重建前」基线
    h = MemoryHub(hub_dir)
    h.open()
    try:
        idx.rebuild_from_hub(h, full=True, distill_concurrency=1)
    finally:
        h.close()
    before = _snapshot()
    assert before["flags"] and json.loads(before["flags"][0][1]).get("verdict"), "基线应含标记与裁定"

    # 第二次全量：真走 staged 重建；重建期 + 重建后的 pass 判定调用 = 0
    judge = FakeJudge("矛盾")
    det = contra.build_detector(idx, judge=judge, sink=_Sink())
    h = MemoryHub(hub_dir)
    h.open()
    try:
        idx.rebuild_from_hub(h, full=True, distill_concurrency=1)
    finally:
        h.close()
    s = contra.run_pass(idx, det, pass_started=datetime.now(UTC), rebuilding=True)
    assert s.skipped == "rebuild" and judge.calls == 0
    assert judge_prompts == [], "重建期不得有任何判定 prompt 到达上游"
    assert _snapshot() == before, "重建前后标记集合逐字相等；判定台账跨重建保留"
    idx.close()


def test_d8_chain_mutant_without_replay_dispatch_is_red(tmp_path: Path) -> None:
    """去掉 consistency 的重放分派 ⇒ 真重放后物化视图是空的，与重建前不等。"""
    from _bb_support import load_mutant

    mut = load_mutant(("        state.flags[rec.key] = rec\n", "        pass\n"))
    hub = MemoryHub(str(tmp_path / "hub"))
    a, b = _fact("a", "甲", t=datetime.now(UTC), created=datetime.now(UTC)), _fact("b", "乙", t=datetime.now(UTC), created=datetime.now(UTC))
    hub.append_admin_event(AdminEventType.AUDIENCE_SET, "", "a|b", **c.build_flag_payload(
        b, a, judge_model="m", reason="r", rank=1, cos=0.9))
    real = c.replay(e for _k, e in hub.scan_admin_events()).snapshot()
    broken = mut.replay(e for _k, e in hub.scan_admin_events()).snapshot()
    assert real["flags"]
    with pytest.raises(AssertionError):
        assert broken == real, "重放分派被去掉，标记集合必须与重建前不等"
