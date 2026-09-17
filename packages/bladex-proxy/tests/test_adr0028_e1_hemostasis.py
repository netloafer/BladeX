"""ADR-0028 E1 止血验收（F1 / F2 / F5 + file_ref 白名单）。

四条止血都是"越晚改历史数据越脏"的形态，且 E5/E6 的所有检索侧验收都必须跑在
止血之后的库上，故本卡是整个 ADR-0028 的前置。

覆盖：
    E1.1  Matter summary 守卫 —— established Matter 的 summary 不被 L4 rewrite 改写
    E1.3  file_ref 退出向量检索平面 —— add_fact 不写 LanceDB，meta 照常
    E1.5  _extract_file_refs 白名单 —— 检索类工具 / 目录路径 / 泛化键全部出局
（E1.4 结论/进展通道信封剥离在 core 侧，见 test_adr0028_e1_envelope_channels.py）
"""

from __future__ import annotations

import pytest
from bladex_core.attribution import AttributionSource
from bladex_core.fact import Fact, ItemKind
from bladex_core.matter import Matter, MatterStatus
from bladex_proxy.models import ToolEvent
from bladex_proxy.storage.memory_index import MemoryIndex, _extract_file_refs

# ── E1.1 Matter summary 守卫 ───────────────────────────────────────


class _Decision:
    """最小 decision 替身（只带 _apply_decision 用到的字段）。"""

    def __init__(self, matter_id: str, summary_rewrite: str) -> None:
        self.matter_id = matter_id
        self.summary_rewrite = summary_rewrite
        self.source = AttributionSource.LLM_LINK
        self.is_new_matter = False
        self.confidence = 0.9
        self.reason = "test"


class _Pipeline:
    """最小 pipeline 替身。"""

    def create_edge_for_decision(self, decision, fact):  # noqa: ANN001, ARG002
        from bladex_core.matter import EdgeProvenance, EdgeTargetType, MatterEdge
        return MatterEdge(
            matter_id=decision.matter_id,
            target_type=EdgeTargetType.FACT,
            target_key=fact.id,
            provenance=EdgeProvenance.AUTO,
        )

    def reactivate_on_hit(self, matter):  # noqa: ANN001, ARG002
        return None

    def create_matter_for_decision(self, decision, centroid=None):  # noqa: ANN001, ARG002
        return Matter(matter_id=decision.matter_id)


@pytest.fixture()
def index(tmp_path):
    store = MemoryIndex(tmp_path / "index")
    store.open()
    yield store


def _fact(fid: str, content: str, kind: ItemKind = ItemKind.ASSERTION) -> Fact:
    return Fact(id=fid, content=content, item_kind=kind,
                source_user_id="u", source_session="s")


@pytest.mark.parametrize("status", [
    MatterStatus.ACTIVE, MatterStatus.DORMANT, MatterStatus.CLOSED,
])
def test_established_matter_summary_not_rewritten(index, status, capsys):
    """F1：established Matter 的 summary 不许被单条新 fact 的裁决整段替换。

    真实事故：一句 e2e 探针 fact 的 L4 裁决把 beta 发布旗舰 Matter 的摘要整段
    换成 `The user's BladeX end-to-end probe code is bx3a984bcd.`，
    该摘要作为 Matter 卡首行**每轮注入**给所有命中 BladeX 话题的请求。
    """
    m = Matter(matter_id="m-established", title="beta release",
               summary="真实工作摘要", status=status)
    index.add_matter(m)
    member = _fact("f-member", "真实工作摘要")
    index.add_fact(member)
    from bladex_core.matter import EdgeProvenance, EdgeTargetType, MatterEdge
    index.add_edge(MatterEdge(matter_id="m-established", target_type=EdgeTargetType.FACT,
                           target_key="f-member", provenance=EdgeProvenance.AUTO))

    probe = _fact("f-probe", "探针轮的一句话")
    index.add_fact(probe)
    rewrite = "The user's BladeX end-to-end probe code is bx3a984bcd."

    index._apply_decision(  # noqa: SLF001
        probe, _Decision("m-established", rewrite), _Pipeline(),
        __import__("datetime").datetime.now(__import__("datetime").UTC),
    )
    logged = capsys.readouterr().out

    after = index.get_matter("m-established")
    # 摘要没有被 rewrite 整段替换，而是回落到确定性聚合
    assert rewrite not in after.summary
    assert "bx3a984bcd" not in after.summary
    assert "真实工作摘要" in after.summary       # 原成员内容仍在
    assert after.summary_source == "concat"
    assert "matter_summary_rewrite_rejected" in logged   # 可 grep 的告警


def test_provisional_matter_summary_rewrite_still_applies(index):
    """PROVISIONAL（胚芽态）Matter 保持现行为：rewrite 生效。"""
    index.add_matter(Matter(matter_id="m-prov", title="new",
                         summary="旧", status=MatterStatus.PROVISIONAL))
    f = _fact("f1", "内容")
    index.add_fact(f)

    index._apply_decision(  # noqa: SLF001
        f, _Decision("m-prov", "裁决重写的摘要"), _Pipeline(),
        __import__("datetime").datetime.now(__import__("datetime").UTC),
    )

    after = index.get_matter("m-prov")
    assert after.summary == "裁决重写的摘要"
    assert after.summary_source == "llm_judge"


# ── E1.3 file_ref 退出向量检索平面 ──────────────────────────────────


def test_file_ref_fact_not_written_to_lancedb(index):
    """file_ref 只写 meta，不进 LanceDB —— 不再侵占散点向量召回位。"""
    dim = 8
    ref = _fact("f-ref", "文件 /Users/alice/x.md", ItemKind.FILE_REF)
    ref.embedding = [0.1] * dim
    index.add_fact(ref)

    # meta 侧照常可读（Matter 归属 / supersede / hash 失效全部不受影响）
    got = index.get_fact("f-ref")
    assert got is not None
    assert got.item_kind == ItemKind.FILE_REF

    # 向量表要么根本没建，要么不含该 id
    if index._table is not None:  # noqa: SLF001
        ids = set(index._table.to_arrow()["id"].to_pylist())  # noqa: SLF001
        assert "f-ref" not in ids


def test_non_file_ref_fact_still_written_to_lancedb(index):
    """零回归：非 file_ref 条目照常入向量表。"""
    dim = 8
    f = _fact("f-assert", "BladeX 的热路径预算是 200ms")
    f.embedding = [0.2] * dim
    index.add_fact(f)

    assert index._table is not None  # noqa: SLF001
    ids = set(index._table.to_arrow()["id"].to_pylist())  # noqa: SLF001
    assert "f-assert" in ids


# ── E1.5 _extract_file_refs 白名单 ──────────────────────────────────


def _te(tool_name: str, args: dict, result: str = "x") -> ToolEvent:
    return ToolEvent(tool_name=tool_name, arguments=args, result=result)


def test_search_tool_path_arg_is_not_a_file_ref():
    """事故来源：`search_files{path:"/Users/alice"}` → `文件 /Users/alice`。

    双重出局：① 检索类工具黑名单；② "path" 已不在白名单键里。
    """
    assert _extract_file_refs([_te("search_files", {"path": "/Users/alice"})]) == []


def test_directory_path_rejected_no_extension():
    """目录没有扩展名 → _PATH_RE 不匹配 → 天然出局。"""
    assert _extract_file_refs([
        _te("Read", {"file_path": "/Users/alice/dev/BladeX"}),
    ]) == []


def test_generic_path_and_file_keys_dropped():
    """泛化键 "path" / "file" 已从白名单删除——即便值形似文件也不抓。"""
    assert _extract_file_refs([_te("SomeTool", {"path": "/a/b/c.md"})]) == []
    assert _extract_file_refs([_te("SomeTool", {"file": "/a/b/c.md"})]) == []


def test_value_shaped_like_path_fallback_removed():
    """"值形似路径就抓"的兜底分支已删除。"""
    assert _extract_file_refs([
        _te("SomeTool", {"random_arg": "/a/b/c.md"}),
    ]) == []


def test_whitelisted_keys_still_extract():
    """正例：白名单键 + 带扩展名 + 非检索工具 → 照常产出。"""
    refs = _extract_file_refs([
        _te("Read", {"file_path": "/Users/alice/dev/BladeX/README.md"}, "hello"),
        _te("Edit", {"target_file": "packages/bladex_core/fact.py"}, "world"),
        _te("NotebookEdit", {"notebook_path": "nb/analysis.ipynb"}, ""),
    ])
    assert [r["path"] for r in refs] == [
        "/Users/alice/dev/BladeX/README.md",
        "packages/bladex_core/fact.py",
        "nb/analysis.ipynb",
    ]
    assert refs[0]["content_hash"]          # result 非空 → 有 hash
    assert refs[2]["content_hash"] == ""    # result 空 → 无 hash


def test_blacklisted_tools_are_skipped_entirely():
    """检索类工具即便用了白名单键也整体跳过（路径语义是"在哪找"）。"""
    for name in ("search_files", "glob", "Grep", "list_dir", "LS"):
        assert _extract_file_refs([
            _te(name, {"file_path": "/a/b/c.md"}),
        ]) == [], name


def test_dedup_by_path():
    """同一路径多次出现只产一条（现行为不变）。"""
    refs = _extract_file_refs([
        _te("Read", {"file_path": "/a/b.md"}),
        _te("Edit", {"file_path": "/a/b.md"}),
    ])
    assert len(refs) == 1
