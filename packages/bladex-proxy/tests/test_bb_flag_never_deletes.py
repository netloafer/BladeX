"""批 BB · 2a · D5「只标不删」+ 载荷纪律（设计 §6.2 四条）。

1. 静态：AST 扫矛盾模块（2b 起含端点 handler / CLI），禁删除类调用与 `t_invalid` / `superseded_by` 赋值。
2. 运行时：标记 → 裁定 → 重放 → 重建，fact dump 逐字相等、`tombstone_hash` 不变、Hub 新增事件全是 `kind` 闭集内的 `audience_set`。
3. 反向变异：路径里加一处写墓碑 / 改 fact，守卫必须红（用例名见结算稿 §2）。
4. 载荷：flag / verdict 不含 `fact_id` / `audience`。
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest
from _bb_support import (
    CONSISTENCY_SRC,
    FakeJudge,
    FakeLedger,
    FakePending,
    load_mutant,
    make_detector,
    mk_fact,
)
from bladex_core import consistency as c
from bladex_proxy.models import AdminEventType, TombstoneTargetType
from bladex_proxy.storage.memory_hub import MemoryHub
from bladex_proxy.storage.memory_index import MemoryIndex

_REPO = Path(__file__).resolve().parents[3]
#: 2b 接入后把端点 handler / CLI 命令所在文件补进来（不存在的跳过，存在的必扫）。
_PKG = _REPO / "packages/bladex-proxy/bladex_proxy"
#: 批 BB · 2b：补上接线文件（同一 AST 规则）。`memory_index.py` 太大且合法含大量 fact 写，不扫整文件，
#: 只扫它的矛盾方法（见 `test_static_index_contradiction_methods_are_clean`）。
_SCANNED = [CONSISTENCY_SRC, _PKG / "contradiction.py", _PKG / "storage/contradiction_store.py"]

FORBIDDEN_CALLS = frozenset({
    "append_tombstone", "compact_tombstoned", "delete_fact", "correct_fact", "add_fact", "update_fact",
    "remove_fact", "forget",
})
FORBIDDEN_ASSIGN = frozenset({"t_invalid", "superseded_by"})


def static_violations(src: str, funcs: frozenset[str] | None = None) -> list[str]:
    """AST 扫描。`funcs` 给定时只扫这些名字的函数 / 方法（含嵌套），用于不能整文件扫的大文件。"""
    tree = ast.parse(src)
    roots: list[ast.AST] = [tree]
    if funcs is not None:
        roots = [n for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in funcs]
    out: list[str] = []
    for root in roots:
        for node in ast.walk(root):
            if isinstance(node, ast.Call):
                fn = node.func
                name = fn.attr if isinstance(fn, ast.Attribute) else fn.id if isinstance(fn, ast.Name) else ""
                if name in FORBIDDEN_CALLS or name.startswith("supersede"):
                    out.append(f"call:{name}@{node.lineno}")
            targets: list[ast.expr] = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                targets = [node.target]
            for t in targets:
                if isinstance(t, ast.Attribute) and t.attr in FORBIDDEN_ASSIGN:
                    out.append(f"assign:{t.attr}@{node.lineno}")
    return out


#: 批 BB · 2b：大文件里只扫矛盾相关函数（文件名 → 函数名）。名字不存在 ⇒ 用例红（防改名后扫描空转）。
_SCOPED: dict[Path, frozenset[str]] = {
    _PKG / "server/admin_api.py": frozenset({"admin_contradiction_flag", "admin_contradiction_settle"}),
    _PKG / "admin_read.py": frozenset({"admin_contradictions", "_contradiction_row", "_contradiction_status"}),
    _PKG / "cli/memory_cmds.py": frozenset({"memory_conflicts", "memory_settle", "_conflict_lines"}),
    _PKG / "storage/memory_index.py": frozenset({
        "_apply_contradiction_event", "contradiction_kv_get", "contradiction_kv_put", "contradiction_kv_delete",
        "contradiction_kv_scan", "contradiction_settled", "contradiction_usage", "contradiction_neighbors",
        "contradictions", "contradictions_for", "_contradiction_item", "_contradiction_kinds"}),
}


def payload_violations(payload: dict[str, Any]) -> list[str]:
    return sorted(c.FORBIDDEN_PAYLOAD_KEYS & set(payload)) + ([] if payload.get("kind") in c.KINDS else ["kind"])


# ── 1. 静态 ──────────────────────────────────────────────────────────────


def test_static_no_delete_paths_in_contradiction_code() -> None:
    for p in _SCANNED:
        assert p.is_file(), p
        assert static_violations(p.read_text(encoding="utf-8")) == [], p


def test_static_scoped_handlers_cli_and_index_methods_are_clean() -> None:
    for path, names in _SCOPED.items():
        src = path.read_text(encoding="utf-8")
        found = {n.name for n in ast.walk(ast.parse(src))
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        assert names <= found, f"{path.name}: 扫描目标不存在 {sorted(names - found)}（改名了？扫描会空转）"
        assert static_violations(src, names) == [], path.name


def test_static_scoped_mutant_tombstone_in_handler_is_caught() -> None:
    path = _PKG / "server/admin_api.py"
    src = path.read_text(encoding="utf-8")
    anchor = "            logger.info(\"admin_contradiction_settled\""
    assert src.count(anchor) == 1
    bad = src.replace(anchor, "            hub.append_tombstone(None, lo)\n" + anchor)
    assert static_violations(bad, _SCOPED[path]), "端点里加写墓碑，必须红"


def test_static_scoped_mutant_supersede_in_index_method_is_caught() -> None:
    path = _PKG / "storage/memory_index.py"
    src = path.read_text(encoding="utf-8")
    anchor = "        self.contradiction_kv_put(key, cur)\n        return True"
    assert src.count(anchor) == 1
    bad = src.replace(anchor, "        f = self.get_fact(rec.pair[0])\n        f.t_invalid = 1\n" + anchor)
    assert static_violations(bad, _SCOPED[path]), "Index 矛盾方法里改 t_invalid，必须红"


def test_static_mutant_tombstone_call_is_caught() -> None:
    src = CONSISTENCY_SRC.read_text(encoding="utf-8")
    bad = src.replace("self._pending.remove(fact.id)\n        return res",
                      "self._pending.remove(fact.id)\n        self._sink.append_tombstone(fact.id)\n        return res")
    assert bad != src
    assert static_violations(bad), "加一处写墓碑，静态守卫必须红"


def test_static_mutant_assign_t_invalid_is_caught() -> None:
    src = CONSISTENCY_SRC.read_text(encoding="utf-8")
    bad = src.replace("res = ProcessResult(fact.id)\n", "res = ProcessResult(fact.id)\n        old_fact.t_invalid = 1\n", 1)
    assert bad != src
    assert static_violations(bad), "给 t_invalid 赋值，静态守卫必须红"


def test_static_allowlist_is_only_the_sink() -> None:
    """白名单只有 `append_flag`（2b 落成 append_admin_event(AUDIENCE_SET, …)）：模块里没有任何 hub / index 写调用。"""
    calls = {n.func.attr for n in ast.walk(ast.parse(CONSISTENCY_SRC.read_text(encoding="utf-8")))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert not {x for x in calls if x.startswith(("append_", "put", "set_", "write"))} - {"append_flag"}


# ── 4. 载荷 ──────────────────────────────────────────────────────────────


def test_payloads_never_carry_fact_id_or_audience() -> None:
    a, b = mk_fact("fa", "甲"), mk_fact("fb", "乙")
    flag = c.build_flag_payload(a, b, judge_model="m", reason="r", rank=1, cos=0.9)
    verdict = c.build_verdict_payload(("fa", "fb"), (c.content_sha(a), c.content_sha(b)), verdict="confirmed", origin="cli")
    for p in (flag, verdict):
        assert payload_violations(p) == []
        assert "reason" not in p and p.get("reason_sha", "x")  # Hub 只存 hash


def test_payload_guard_mutant_with_fact_id_is_red() -> None:
    a, b = mk_fact("fa", "甲"), mk_fact("fb", "乙")
    p = c.build_flag_payload(a, b, judge_model="m", reason="r", rank=1, cos=0.9)
    p["fact_id"] = "fa"
    assert payload_violations(p) == ["fact_id"]
    assert c.parse_payload(p) is None, "带 fact_id 的载荷不得被当成矛盾记录解析"
    with pytest.raises(ValueError):
        c._check_clean(p)


def test_builder_mutant_that_adds_audience_is_rejected_by_check() -> None:
    m = load_mutant(("def _check_clean(payload: Mapping[str, Any]) -> None:\n    bad =",
                     "def _check_clean(payload: Mapping[str, Any]) -> None:\n    return\n    bad ="))
    a, b = mk_fact("fa", "甲"), mk_fact("fb", "乙")
    p = m.build_flag_payload(a, b, judge_model="m", reason="r", rank=1, cos=0.9)
    p["audience"] = "all"
    m._check_clean(p)  # 变异后 builder 守卫失效……
    assert payload_violations(p) == ["audience"], "……但载荷守卫仍然红"


def test_verdict_origin_model_rejected() -> None:
    with pytest.raises(ValueError):
        c.build_verdict_payload(("a", "b"), ("1", "2"), verdict="confirmed", origin="model")
    bad = {"kind": c.KIND_VERDICT, "schema": 1, "pair": ["a", "b"], "content_sha": ["1", "2"],
           "verdict": "confirmed", "origin": "model", "note": ""}
    assert c.parse_payload(bad) is None


# ── 2. 运行时：标记 → 裁定 → 重放 → 重建 ──────────────────────────────────


class HubSink:
    """2b 出口的形状：`append_admin_event(AUDIENCE_SET, "", pair_key, **payload)`。"""

    def __init__(self, hub: MemoryHub) -> None:
        self.hub = hub

    def append_flag(self, pair_key: str, payload: dict[str, Any]) -> None:
        self.hub.append_admin_event(AdminEventType.AUDIENCE_SET, "", pair_key, **payload)


def _tomb(hub: MemoryHub) -> str:
    """墓碑集合指纹（`HubMetaSnapshot.tombstone_hash`，flash_daemon 同口径）。"""
    return hub.scan_meta_snapshot().tombstone_hash()


def _dump(ix: MemoryIndex) -> str:
    return json.dumps({f.id: f.model_dump(mode="json") for f in ix.all_facts()}, sort_keys=True)


def _world(tmp_path: Path, tag: str = "a") -> tuple[MemoryHub, MemoryIndex, list]:
    hub = MemoryHub(str(tmp_path / f"hub{tag}"))
    ix = MemoryIndex(str(tmp_path / f"ix{tag}"))
    ix.open()
    facts = [mk_fact(f"f{i}", f"内容{i}", day=2 + i) for i in range(4)]
    for f in facts:
        ix.add_fact(f)
    return hub, ix, facts


def run_scenario(hub: MemoryHub, ix: MemoryIndex, facts: list, *, sink: Any = None, mod: Any = c) -> dict[str, Any]:
    new, olds = facts[-1], facts[:-1]
    det, parts = make_detector(mod, sink=sink or HubSink(hub), judge=FakeJudge("矛盾"))
    res = det.process(new, [(o, 0.95 - i * 0.01) for i, o in enumerate(olds)])
    flagged = res.flagged[0]
    lo, hi = flagged.split("|")
    sha = {f.id: c.content_sha(f) for f in facts}
    hub.append_admin_event(AdminEventType.AUDIENCE_SET, "", flagged, **c.build_verdict_payload(
        (lo, hi), (sha[lo], sha[hi]), verdict="not_same_thing", origin="cli"))
    return {"res": res, "parts": parts}


def assert_no_destruction(hub: MemoryHub, ix: MemoryIndex, before_dump: str, before_tomb: str, before_events: int) -> None:
    assert _dump(ix) == before_dump, "fact dump 变了"
    assert _tomb(hub) == before_tomb, "墓碑集合变了"
    new_events = [e for _, e in hub.scan_admin_events()][before_events:]
    assert new_events, "场景没产生事件"
    for e in new_events:
        assert e.event_type == AdminEventType.AUDIENCE_SET and e.matter_id == ""
        assert e.payload.get("kind") in c.KINDS
        assert payload_violations(e.payload) == []


def test_runtime_flag_settle_replay_rebuild_is_non_destructive(tmp_path: Path) -> None:
    hub, ix, facts = _world(tmp_path)
    before, tomb, n0 = _dump(ix), _tomb(hub), len(list(hub.scan_admin_events()))
    run_scenario(hub, ix, facts)
    ix._replay_admin_events(hub)       # 重放：audience_set 带 kind 载荷 = 空操作
    ix._replay_new_admin_events(hub)
    assert_no_destruction(hub, ix, before, tomb, n0)
    # 重建：全新 Index 灌同一批 fact，再重放同一份 Hub 事件，dump 与原 Index 逐字相等
    ix2 = MemoryIndex(str(tmp_path / "ix_rebuilt"))
    ix2.open()
    for f in facts:
        ix2.add_fact(f)
    ix2._replay_admin_events(hub)
    assert _dump(ix2) == before
    assert _tomb(hub) == tomb


def test_runtime_mutant_tombstone_in_sink_is_red(tmp_path: Path) -> None:
    hub, ix, facts = _world(tmp_path)
    before, tomb, n0 = _dump(ix), _tomb(hub), len(list(hub.scan_admin_events()))

    class EvilSink(HubSink):
        def append_flag(self, pair_key: str, payload: dict[str, Any]) -> None:
            super().append_flag(pair_key, payload)
            self.hub.append_tombstone(TombstoneTargetType.FACT, pair_key.split("|")[0])

    run_scenario(hub, ix, facts, sink=EvilSink(hub))
    with pytest.raises(AssertionError, match="墓碑集合变了"):
        assert_no_destruction(hub, ix, before, tomb, n0)


def test_runtime_mutant_fact_mutation_is_red(tmp_path: Path) -> None:
    hub, ix, facts = _world(tmp_path)
    before, tomb, n0 = _dump(ix), _tomb(hub), len(list(hub.scan_admin_events()))

    class EvilSink(HubSink):
        def append_flag(self, pair_key: str, payload: dict[str, Any]) -> None:
            super().append_flag(pair_key, payload)
            f = ix.get_fact(pair_key.split("|")[0])
            f.content += "!"
            ix.add_fact(f)

    run_scenario(hub, ix, facts, sink=EvilSink(hub))
    with pytest.raises(AssertionError, match="fact dump 变了"):
        assert_no_destruction(hub, ix, before, tomb, n0)


def test_runtime_mutant_event_type_outside_closed_set_is_red(tmp_path: Path) -> None:
    hub, ix, facts = _world(tmp_path)
    before, tomb, n0 = _dump(ix), _tomb(hub), len(list(hub.scan_admin_events()))
    run_scenario(hub, ix, facts)
    hub.append_admin_event(AdminEventType.AUDIENCE_SET, "", "x|y", fact_id="f0", audience="all")  # 旧语义载荷混入
    with pytest.raises(AssertionError):
        assert_no_destruction(hub, ix, before, tomb, n0)


def test_detector_sees_only_ledger_and_pending_writes() -> None:
    det, parts = make_detector(judge=FakeJudge("不矛盾"))
    new = mk_fact("fn", "新", day=5)
    res = det.process(new, [(mk_fact("fo", "旧"), 0.9)])
    assert res.judged == 1 and parts["sink"].events == [] and parts["ledger"].writes == 1
    _ = FakeLedger, FakePending


# ── D6 回滚无害：重放这两种载荷不改任何 fact（2a 时 HEAD 是空操作；2b 起 HEAD 把它们物化到
#    `contradiction/` 前缀，仍不碰 fact；v0.3.0 上是空操作见 probe）；把 fact_id 加回去就会真改 fact（阴性对照）──


def test_replay_of_contradiction_payloads_is_noop_and_fact_id_variant_is_not(tmp_path: Path) -> None:
    from bladex_proxy.models import AdminEvent

    hub, ix, facts = _world(tmp_path)
    a, b = facts[0], facts[1]
    flag = c.build_flag_payload(b, a, judge_model="m", reason="r", rank=1, cos=0.9)
    verdict = c.build_verdict_payload((a.id, b.id), (c.content_sha(a), c.content_sha(b)), verdict="confirmed", origin="cli")
    before = _dump(ix)
    for p in (flag, verdict):
        ev = AdminEvent(event_type=AdminEventType.AUDIENCE_SET, matter_id="", target_key=f"{a.id}|{b.id}", payload=p)
        assert ix._apply_admin_event(ev) is True  # 2b：被 consistency 认领、物化到 contradiction/
    assert _dump(ix) == before, "物化不得改任何 fact"
    assert ix.contradiction_kv_get(f"contradiction/{a.id}|{b.id}") is not None
    # 带 kind 又带 fact_id / audience 的混合载荷：2b 起整条被 consistency 拒收，**不会**落到老分支改 fact
    mixed = AdminEvent(event_type=AdminEventType.AUDIENCE_SET, matter_id="", target_key="x",
                       payload={**flag, "fact_id": a.id, "audience": "agent:x"})
    assert ix._apply_admin_event(mixed) is False and _dump(ix) == before
    # 阴性对照：老事件（无 kind）行为逐字不变——带 fact_id / audience 的载荷会真改 fact
    legacy = AdminEvent(event_type=AdminEventType.AUDIENCE_SET, matter_id="", target_key="x",
                        payload={"fact_id": a.id, "audience": "agent:x"})
    assert ix._apply_admin_event(legacy) is True and _dump(ix) != before
