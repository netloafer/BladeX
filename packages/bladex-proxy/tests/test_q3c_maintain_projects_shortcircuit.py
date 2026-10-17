"""批 Q3C · MQ-W19：`maintain_projects` 没变化就不重算。

判据（提示词 §2.1）：
① 连跑两轮，第二轮命中短路；
② 两轮之间制造每一种会影响结果的变化——短路都**不命中**，且重算结果与
   「不带短路的旧实现」（`_maintain_projects_recompute`，旧函数体原样）逐字段相同；
③ 命中与否可 grep：`index_projects_maintained skipped=true|false reason=...`。

"逐字段相同"排除三个非语义字段：`Project.version`（每次 upsert +1 的计数器，
无读方）、`Project.updated_at`、`MatterEdge.created_at`（重写即刷新的时间戳）。
"""

from __future__ import annotations

from collections.abc import Callable

import msgpack
import pytest
import structlog
from bladex_core.fact import Fact, ItemKind
from bladex_core.matter import (
    EdgeProvenance,
    EdgeTargetType,
    Matter,
    MatterEdge,
    MatterStatus,
)
from bladex_core.project import project_id_for
from bladex_proxy.storage.memory_index import MemoryIndex


class _Embedder:
    def embed(self, texts):  # noqa: ANN001, ANN201
        out = []
        for t in texts:
            v = [0.0] * 32
            v[sum(t.encode()) % 32] = 1.0
            out.append(v)
        return out

    @property
    def available(self) -> bool:
        return True


ROOT = "/dev/BladeX"


def _file_fact(fid: str, path: str) -> Fact:
    return Fact(id=fid, content=f"文件 {path}", item_kind=ItemKind.FILE_REF,
                subject=path, attribute="file", source_user_id="u1")


def _fact_edge(mid: str, fid: str) -> MatterEdge:
    return MatterEdge(matter_id=mid, target_type=EdgeTargetType.FACT,
                      target_key=fid, provenance=EdgeProvenance.AUTO)


def _put_fact_raw(ix: MemoryIndex, fact: Fact) -> None:
    """直接覆写 meta 里的 fact（模拟任何写路径对已有 fact 字段的修改）。"""
    ix.meta_db[f"fact/{fact.id}".encode()] = msgpack.packb(
        fact.model_dump(mode="json", exclude={"embedding"}), use_bin_type=True)


def _seed(ix: MemoryIndex) -> None:
    ix.add_fact(_file_fact("rule", f"{ROOT}/CLAUDE.md"))
    ix.add_fact(_file_fact("fa", f"{ROOT}/a.py"))
    ix.add_fact(_file_fact("fb", f"{ROOT}/b.rs"))
    ix.add_fact(_file_fact("fx", f"{ROOT}/x.ts"))  # 先入库、不挂边（「新增边」用）
    ix.add_matter(Matter(matter_id="m1", title="检索重构", summary="E6 融合已落地",
                         status=MatterStatus.ACTIVE))
    ix.add_edge(_fact_edge("m1", "fa"))
    ix.add_edge(_fact_edge("m1", "fb"))


def _state(ix: MemoryIndex) -> tuple:
    projects = sorted(
        tuple(sorted(p.model_dump(mode="json", exclude={"version", "updated_at"}).items(),
                     key=lambda kv: kv[0]))
        for p in ix.all_projects()
    )
    edges = sorted(
        (e.edge_id, e.matter_id, e.target_type.value, e.target_key, e.relation.value,
         e.provenance.value, e.weight, e.confidence)
        for m in ix.all_matters() for e in ix.get_edges(m.matter_id)
    )
    return repr(projects), edges


def _maintain(ix: MemoryIndex) -> dict:
    with structlog.testing.capture_logs() as cap:
        ix.maintain_projects()
    rows = [r for r in cap if r.get("event") == "index_projects_maintained"]
    assert len(rows) == 1, cap
    return rows[0]


def _open(path) -> MemoryIndex:  # noqa: ANN001
    ix = MemoryIndex(path, embedder=_Embedder(), read_only=False)
    ix.open()
    return ix


@pytest.fixture()
def pair(tmp_path):
    """A = 带短路的新实现；B = 旧实现（每次都跑旧函数体）。两边状态同源。"""
    a, b = _open(tmp_path / "a"), _open(tmp_path / "b")
    _seed(a)
    _seed(b)
    yield a, b
    a.close()
    b.close()


def test_second_run_without_changes_is_skipped(pair):
    a, _b = pair
    first = _maintain(a)
    assert first["skipped"] is False and first["reason"] == "first_run"
    v1 = a.get_project(project_id_for(ROOT)).version
    second = _maintain(a)
    assert second["skipped"] is True and second["reason"] == "inputs_unchanged"
    # 短路真的没写：版本计数器不动（旧实现每轮 +2）
    assert a.get_project(project_id_for(ROOT)).version == v1
    assert a.maintain_projects() == 0


def test_writes_outside_the_inputs_do_not_defeat_the_skip(pair):
    a, _b = pair
    _maintain(a)
    a.meta_db[b"profile/user_md"] = msgpack.packb({"x": 1}, use_bin_type=True)
    a.meta_db[b"__meta__/p2_consumed/k1"] = b"1"
    assert _maintain(a)["skipped"] is True


def _new_rulefile(ix: MemoryIndex) -> None:
    ix.add_fact(_file_fact("rule2", "/dev/Other/AGENTS.md"))


def _new_matter(ix: MemoryIndex) -> None:
    ix.add_fact(_file_fact("fc", f"{ROOT}/c.rs"))
    ix.add_matter(Matter(matter_id="m2", title="账本切换", summary="L91 已修",
                         status=MatterStatus.ACTIVE))
    ix.add_edge(_fact_edge("m2", "fc"))


def _new_edge(ix: MemoryIndex) -> None:
    ix.add_edge(_fact_edge("m1", "fx"))


def _change_edge(ix: MemoryIndex) -> None:
    """同 edge_id 覆写：m1→fb 的边改指 fx（归属改判）。"""
    old = next(e for e in ix.get_edges("m1") if e.target_key == "fb")
    moved = old.model_copy(update={"target_key": "fx"})
    ix.add_edge(moved)


def _drop_edge(ix: MemoryIndex) -> None:
    """无按边删除的公开 API（只有按 turn 删），直接删 meta。"""
    old = next(e for e in ix.get_edges("m1") if e.target_key == "fb")
    del ix.meta_db[f"edge/{old.edge_id}".encode()]


def _matter_title(ix: MemoryIndex) -> None:
    m = next(m for m in ix.all_matters() if m.matter_id == "m1")
    ix.update_matter(m.model_copy(update={"title": "检索重构（二期）"}))


def _matter_summary(ix: MemoryIndex) -> None:
    m = next(m for m in ix.all_matters() if m.matter_id == "m1")
    ix.update_matter(m.model_copy(update={"summary": "E6 融合已回滚"}))


def _fact_subject(ix: MemoryIndex) -> None:
    _put_fact_raw(ix, _file_fact("fa", f"{ROOT}/a.kt"))


def _fact_kind(ix: MemoryIndex) -> None:
    f = ix.get_fact("fb")
    _put_fact_raw(ix, f.model_copy(update={"item_kind": ItemKind.ASSERTION}))


def _rulefile_removed(ix: MemoryIndex) -> None:
    f = ix.get_fact("rule")
    _put_fact_raw(ix, f.model_copy(update={"subject": f"{ROOT}/README.md"}))


def _project_record_edited(ix: MemoryIndex) -> None:
    p = ix.get_project(project_id_for(ROOT))
    p.description = "被别处改过"
    p.languages = []
    ix.meta_db[f"project/{p.project_id}".encode()] = msgpack.packb(
        p.model_dump(mode="json"), use_bin_type=True)


def _project_record_deleted(ix: MemoryIndex) -> None:
    del ix.meta_db[f"project/{project_id_for(ROOT)}".encode()]


MUTATIONS: dict[str, Callable[[MemoryIndex], None]] = {
    "new_rulefile_fact": _new_rulefile,
    "new_matter": _new_matter,
    "new_edge": _new_edge,
    "change_edge": _change_edge,
    "drop_edge": _drop_edge,
    "matter_title": _matter_title,
    "matter_summary": _matter_summary,
    "fact_subject": _fact_subject,
    "fact_item_kind": _fact_kind,
    "rulefile_no_longer_rulefile": _rulefile_removed,
    "project_record_edited": _project_record_edited,
    "project_record_deleted": _project_record_deleted,
}


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_every_input_change_defeats_the_skip_and_matches_old(pair, name):
    a, b = pair
    _maintain(a)
    b._maintain_projects_recompute()
    assert _state(a) == _state(b)
    assert _maintain(a)["skipped"] is True

    MUTATIONS[name](a)
    MUTATIONS[name](b)

    row = _maintain(a)
    assert row["skipped"] is False and row["reason"] == "inputs_changed", name
    b._maintain_projects_recompute()
    assert _state(a) == _state(b), name
    # 重算之后又回到可短路
    assert _maintain(a)["skipped"] is True


def test_restart_recomputes_once(tmp_path):
    ix = _open(tmp_path / "ix")
    _seed(ix)
    _maintain(ix)
    ix.close()
    ix = _open(tmp_path / "ix")
    try:
        assert _maintain(ix)["reason"] == "first_run"
        assert _maintain(ix)["skipped"] is True
    finally:
        ix.close()


def test_readonly_is_a_noop(tmp_path):
    ix = _open(tmp_path / "ix")
    _seed(ix)
    ix.close()
    ro = MemoryIndex(tmp_path / "ix", embedder=_Embedder(), read_only=True)
    ro.open()
    try:
        assert ro.maintain_projects() == 0
    finally:
        ro.close()
