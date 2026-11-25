"""批 BB · 2b · Index 侧：`kind` 分派顺序、四个存储重启不清零、`obsolete` 读时过滤、0.3.0 回滚读法不受影响。"""

from __future__ import annotations

import json
import subprocess
import sys
import types
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from _bb_support import mk_fact
from bladex_core import consistency as c
from bladex_proxy.models import AdminEvent, AdminEventType
from bladex_proxy.storage import memory_index as mi
from bladex_proxy.storage.contradiction_store import IndexContradictionStore
from bladex_proxy.storage.memory_index import MemoryIndex

_REPO = Path(__file__).resolve().parents[3]


def _index(path: Path) -> MemoryIndex:
    ix = MemoryIndex(str(path))
    ix.open()
    return ix


def _flag_event(a, b, *, rank: int = 1) -> AdminEvent:
    new, old = (b, a) if b.id > a.id else (a, b)
    return AdminEvent(event_type=AdminEventType.AUDIENCE_SET, matter_id="", target_key=c.pair_key(a.id, b.id),
                      payload=c.build_flag_payload(new, old, judge_model="m", reason="理由", rank=rank, cos=0.9))


def _verdict_event(a, b, verdict: str = "not_same_thing") -> AdminEvent:
    lo, hi = sorted((a, b), key=lambda f: f.id)
    return AdminEvent(event_type=AdminEventType.AUDIENCE_SET, matter_id="", target_key=c.pair_key(a.id, b.id),
                      payload=c.build_verdict_payload((lo.id, hi.id), (c.content_sha(lo), c.content_sha(hi)),
                                                      verdict=verdict, origin="cli"))


@pytest.fixture()
def ix(tmp_path: Path):
    index = _index(tmp_path / "ix")
    fa, fb = mk_fact("fa", "端口是 6379", day=2), mk_fact("fb", "端口是 6380", day=3)
    for f in (fa, fb):
        index.add_fact(f)
    index._fa, index._fb = fa, fb  # type: ignore[attr-defined]
    yield index
    index.close()


# ── kind 分派顺序 ─────────────────────────────────────────────────────────


def _load_index_mutant(*, move_kind_after_legacy: bool) -> types.ModuleType:
    """读 memory_index.py 源码，把 `kind` 分派块挪到老 `audience_set` 分支**之后**，编译成独立模块。"""
    path = Path(mi.__file__)
    src = path.read_text(encoding="utf-8")
    start = src.index('                if event.event_type == "audience_set" and event.payload.get("kind") in _contradiction_kinds():')
    end = src.index('                if event.event_type == "audience_set":\n                    # MQ-L27')
    block = src[start:end]
    assert move_kind_after_legacy and block.count("return self._apply_contradiction_event(event)") == 1
    src = src[:start] + src[end:]
    anchor = '                if event.event_type == "matter_create":'
    assert src.count(anchor) == 1
    src = src.replace(anchor, block + anchor)
    mod = types.ModuleType("bladex_proxy.storage._memory_index_mutant")
    mod.__file__ = str(path)
    mod.__package__ = "bladex_proxy.storage"
    sys.modules[mod.__name__] = mod
    exec(compile(src, str(path), "exec"), mod.__dict__)  # noqa: S102 —— 仅测试里的反向变异
    return mod


def test_kind_is_dispatched_before_the_legacy_audience_branch(ix: MemoryIndex) -> None:
    fa, fb = ix._fa, ix._fb  # type: ignore[attr-defined]
    before = json.dumps({f.id: f.model_dump(mode="json") for f in ix.all_facts()}, sort_keys=True)
    assert ix._apply_admin_event(_flag_event(fa, fb)) is True
    assert ix.contradiction_kv_get("contradiction/fa|fb")["flag"]["pair"] == ["fa", "fb"]
    assert json.dumps({f.id: f.model_dump(mode="json") for f in ix.all_facts()}, sort_keys=True) == before


def test_dispatch_order_mutant_is_red(tmp_path: Path, ix: MemoryIndex) -> None:
    mod = _load_index_mutant(move_kind_after_legacy=True)
    mix = mod.MemoryIndex(str(tmp_path / "ix_mut"))
    mix.open()
    try:
        for f in (ix._fa, ix._fb):  # type: ignore[attr-defined]
            mix.add_fact(f)
        mix._apply_admin_event(_flag_event(ix._fa, ix._fb))  # type: ignore[attr-defined]
        with pytest.raises(AssertionError):
            assert mix.contradiction_kv_get("contradiction/fa|fb") is not None, "kind 判断挪到老分支之后：标记丢了，必须红"
    finally:
        mix.close()


def test_legacy_audience_event_is_unchanged(ix: MemoryIndex) -> None:
    ev = AdminEvent(event_type=AdminEventType.AUDIENCE_SET, matter_id="", target_key="x",
                    payload={"fact_id": "fa", "audience": "agent:x"})
    assert ix._apply_admin_event(ev) is True and ix.get_fact("fa").audience == "agent:x"


def test_invalid_kind_payload_is_swallowed_not_applied_to_facts(ix: MemoryIndex) -> None:
    bad = AdminEvent(event_type=AdminEventType.AUDIENCE_SET, matter_id="", target_key="x",
                     payload={"kind": c.KIND_FLAG, "schema": 1, "pair": ["fb", "fa"], "content_sha": ["1", "2"],
                              "fact_id": "fa", "audience": "agent:x"})
    assert ix._apply_admin_event(bad) is False and ix.get_fact("fa").audience == "all"


# ── 读面：状态 / obsolete / 内容绑定 ──────────────────────────────────────


def test_read_face_status_and_obsolete_are_derived_at_read_time(ix: MemoryIndex) -> None:
    fa, fb = ix._fa, ix._fb  # type: ignore[attr-defined]
    ix._apply_admin_event(_flag_event(fa, fb))
    (it,) = ix.contradictions()["items"]
    assert it["status"] == "unsettled" and it["pair_id"] == "fa|fb" and it["facts"][0]["content"] == fa.content
    assert [x["other_fact_id"] for x in ix.contradictions_for("fa")] == ["fb"]

    ix._apply_admin_event(_verdict_event(fa, fb, "confirmed"))
    assert ix.contradictions(status="confirmed")["total"] == 1 and ix.contradictions(status="unsettled")["total"] == 0

    # 文字被改 ⇒ 旧裁定不再覆盖
    edited = fa.model_copy(update={"content": "端口改成了 7000"})
    ix.add_fact(edited)
    assert ix.contradictions()["items"][0]["status"] == "unsettled"
    ix.add_fact(fa)

    # 任一侧被取代 ⇒ obsolete；读时过滤、不写任何东西
    keys_before = sorted(k for k, _ in ix.contradiction_kv_scan("contradiction/"))
    gone = fb.model_copy(update={"t_invalid": datetime.now(UTC)})
    ix.add_fact(gone)
    assert ix.contradictions()["items"][0]["status"] == "obsolete"
    assert ix.contradictions(status="obsolete")["total"] == 1
    assert sorted(k for k, _ in ix.contradiction_kv_scan("contradiction/")) == keys_before
    ix.add_fact(fb)


def test_obsolete_when_a_side_is_gone(ix: MemoryIndex) -> None:
    ix._apply_admin_event(_flag_event(ix._fa, mk_fact("ghost", "不在库里")))  # type: ignore[attr-defined]
    (it,) = ix.contradictions()["items"]
    assert it["status"] == "obsolete"


def test_pair_with_only_a_verdict_is_not_listed(ix: MemoryIndex) -> None:
    ix._apply_admin_event(_verdict_event(ix._fa, ix._fb))  # type: ignore[attr-defined]
    assert ix.contradictions()["total"] == 0


# ── 四个存储：重启不清零；重建里的命运 ────────────────────────────────────


def _fill_store(s: IndexContradictionStore) -> None:
    s.record("fa|fb", ("1", "2"), {"verdict": "矛盾", "reason": "全文理由"})
    s.record_parse_failure("fa|fc", ("3", "4"))
    s.add("fb")
    s.add_day_input("2026-11-16", 1234)
    s.add_fact_call("fb")
    s.add_fact_call("fb")
    s.set_cursor("2026-11-16T00:00:00+00:00")


def _read_store(s: IndexContradictionStore) -> dict[str, Any]:
    return {"judged": s.is_judged("fa|fb", ("1", "2")), "fails": s.parse_failures("fa|fc", ("3", "4")),
            "pending": s.ids(), "day": s.day_input("2026-11-16"), "fact": s.fact_calls("fb"), "cursor": s.cursor()}


_EXPECT = {"judged": True, "fails": 1, "pending": ["fb"], "day": 1234, "fact": 2,
           "cursor": "2026-11-16T00:00:00+00:00"}


def test_four_stores_survive_restart(tmp_path: Path) -> None:
    ix = _index(tmp_path / "ix")
    _fill_store(IndexContradictionStore(ix))
    ix.close()
    ix2 = _index(tmp_path / "ix")
    try:
        assert _read_store(IndexContradictionStore(ix2)) == _EXPECT
    finally:
        ix2.close()


def test_clear_drops_only_the_derived_view_not_ledger_or_state(ix: MemoryIndex) -> None:
    _fill_store(IndexContradictionStore(ix))
    ix._apply_admin_event(_flag_event(ix._fa, ix._fb))  # type: ignore[attr-defined]
    ix.clear()  # 全量重建写入前清库
    assert ix.contradiction_kv_scan("contradiction/") == [], "物化视图是 Hub 事件的派生物，重建清掉、由重放写回"
    assert _read_store(IndexContradictionStore(ix)) == _EXPECT, "台账 / 状态跨重建保留（重建不重判）"


def test_prefixes_are_registered_with_the_intended_fate() -> None:
    assert mi._CONTRADICTION_PREFIX in mi._META_CLEARED
    assert mi._META_KEPT[mi._CONTRADICTION_JUDGED_PREFIX] == "丙"
    assert mi._META_KEPT[mi._CONTRADICTION_STATE_PREFIX] == "丙"
    assert not any(p.startswith(q) for p in (mi._CONTRADICTION_PREFIX, mi._CONTRADICTION_JUDGED_PREFIX,
                                              mi._CONTRADICTION_STATE_PREFIX)
                   for q in ("fact/", "edge/", "matter/", "profile/"))


# ── 0.3.0 回滚：这些前缀被 all_facts 等读法跳过 ───────────────────────────


def test_rollback_readers_ignore_contradiction_prefixes(ix: MemoryIndex) -> None:
    n_before = (len(ix.all_facts()), ix.fact_count(), len(ix.all_matters()))
    _fill_store(IndexContradictionStore(ix))
    ix._apply_admin_event(_flag_event(ix._fa, ix._fb))  # type: ignore[attr-defined]
    assert (len(ix.all_facts()), ix.fact_count(), len(ix.all_matters())) == n_before


_V030_PROBE = r"""
import json, sys
root, path = sys.argv[1], sys.argv[2]
sys.path[:0] = [f"{root}/packages/bladex-core", f"{root}/packages/bladex-proxy"]
from bladex_proxy.storage import memory_index as mi
assert mi.__file__.startswith(root), mi.__file__
ix = mi.MemoryIndex(path)
ix.open()
print("RESULT " + json.dumps({"facts": sorted(f.id for f in ix.all_facts()), "count": ix.fact_count(),
                  "matters": len(ix.all_matters())}))
ix.close()
"""


def test_v030_tree_reads_the_same_facts_with_contradiction_keys_present(tmp_path: Path) -> None:
    """真 v0.3.0 导出树打开写过矛盾前缀的 Index：fact 集合与计数不变（回滚无害）。"""
    if subprocess.run(["git", "-C", str(_REPO), "rev-parse", "-q", "--verify", "v0.3.0"],
                      capture_output=True).returncode != 0:
        pytest.skip("本仓没有 v0.3.0 tag")
    tree = tmp_path / "v030"
    tree.mkdir()
    arc = subprocess.run(["git", "-C", str(_REPO), "archive", "v0.3.0", "packages"], capture_output=True, check=True)
    subprocess.run(["tar", "-x", "-C", str(tree)], input=arc.stdout, check=True)
    ix = _index(tmp_path / "ix_rb")
    for f in (mk_fact("fa", "甲", day=2), mk_fact("fb", "乙", day=3)):
        ix.add_fact(f)
    _fill_store(IndexContradictionStore(ix))
    ix._apply_admin_event(_flag_event(mk_fact("fa", "甲", day=2), mk_fact("fb", "乙", day=3)))
    assert ix.contradiction_kv_scan("contradiction/") and ix.contradiction_kv_scan("contradiction_judged/")
    ix.close()
    out = subprocess.run([sys.executable, "-I", "-c", _V030_PROBE, str(tree), str(tmp_path / "ix_rb")],
                         capture_output=True, text=True, check=True)
    got = json.loads(next(ln for ln in out.stdout.splitlines() if ln.startswith("RESULT "))[7:])
    assert got == {"facts": ["fa", "fb"], "count": 2, "matters": 0}
