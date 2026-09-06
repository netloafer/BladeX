"""Memory Index Matter 存储 + 归属边单元测试（ADR-0012 T2, §3.1/3.2）。"""

import tempfile
from pathlib import Path

from bladex_core.attribution import UNASSIGNED_MATTER_ID, AttributionPipeline, AttributionSource
from bladex_core.fact import Fact
from bladex_core.matter import (
    EdgeProvenance,
    EdgeTargetType,
    Matter,
    MatterEdge,
    MatterOrigin,
    MatterStatus,
)
from bladex_proxy.storage.memory_index import MemoryIndex


def _make_index(tmpdir: str) -> MemoryIndex:
    """创建一个可写的 Memory Index 实例（不带 embedder，向量操作手动注入）。"""
    index = MemoryIndex(Path(tmpdir) / "index", embedder=None, read_only=False)
    index.open()
    return index


def test_matter_model_roundtrip():
    """T2: Matter 卡写入再读出，字段完整。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        index = _make_index(tmpdir)

        matter = Matter(
            matter_id="m-001",
            title="给 BladeX 加 /v1/messages 支持",
            summary="实现 Anthropic /v1/messages 端点",
            status=MatterStatus.ACTIVE,
            origin=MatterOrigin.MANUAL,
            centroid=[0.1] * 1024,
        )
        index.add_matter(matter)

        loaded = index.get_matter("m-001")
        assert loaded is not None
        assert loaded.matter_id == "m-001"
        assert loaded.title == "给 BladeX 加 /v1/messages 支持"
        assert loaded.status == MatterStatus.ACTIVE
        assert loaded.origin == MatterOrigin.MANUAL
        assert loaded.centroid == [0.1] * 1024

        index.close()


def test_matter_get_nonexistent():
    """T2: 不存在的 Matter 返回 None。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        index = _make_index(tmpdir)
        assert index.get_matter("nonexistent") is None
        index.close()


def test_edge_provenance_manual_vs_auto():
    """T2: manual 和 auto 边共存，get_edges 返回全部。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        index = _make_index(tmpdir)

        matter = Matter(matter_id="m-001", title="test matter")
        index.add_matter(matter)

        # auto 边
        auto_edge = MatterEdge(
            matter_id="m-001",
            target_type=EdgeTargetType.SESSION,
            target_key="u1/a1/s1/",
            provenance=EdgeProvenance.AUTO,
            confidence=0.72,
        )
        index.add_edge(auto_edge)

        # manual 边
        manual_edge = MatterEdge(
            matter_id="m-001",
            target_type=EdgeTargetType.FACT,
            target_key="fact-abc",
            provenance=EdgeProvenance.MANUAL,
            weight=2.0,
        )
        index.add_edge(manual_edge)

        edges = index.get_edges("m-001")
        assert len(edges) == 2

        provenances = {e.provenance for e in edges}
        assert EdgeProvenance.AUTO in provenances
        assert EdgeProvenance.MANUAL in provenances

        # 验证 manual 边权重
        manual = [e for e in edges if e.provenance == EdgeProvenance.MANUAL][0]
        assert manual.weight == 2.0
        assert manual.target_key == "fact-abc"

        # 验证 auto 边置信度
        auto = [e for e in edges if e.provenance == EdgeProvenance.AUTO][0]
        assert auto.confidence == 0.72

        index.close()


def test_matter_aggregate_view_from_edges():
    """T2: 聚合视图从归属边即时聚合，不落独立存储。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        index = _make_index(tmpdir)

        matter = Matter(
            matter_id="m-001",
            title="test matter",
            summary="a summary",
            status=MatterStatus.ACTIVE,
        )
        index.add_matter(matter)

        # 添加多条边
        index.add_edge(MatterEdge(
            matter_id="m-001", target_type=EdgeTargetType.SESSION,
            target_key="u1/a1/s1/", provenance=EdgeProvenance.AUTO,
        ))
        index.add_edge(MatterEdge(
            matter_id="m-001", target_type=EdgeTargetType.SESSION,
            target_key="u1/codex/s2/", provenance=EdgeProvenance.MANUAL,
        ))
        index.add_edge(MatterEdge(
            matter_id="m-001", target_type=EdgeTargetType.FACT,
            target_key="fact-001", provenance=EdgeProvenance.AUTO,
        ))

        view = index.get_matter_aggregate_view("m-001")
        assert view is not None
        assert view.matter_id == "m-001"
        assert view.title == "test matter"
        assert view.status == MatterStatus.ACTIVE
        assert len(view.session_keys) == 2
        assert "u1/a1/s1/" in view.session_keys
        assert "u1/codex/s2/" in view.session_keys
        assert len(view.fact_ids) == 1
        assert "fact-001" in view.fact_ids
        assert view.edge_count == 3
        assert view.manual_edge_count == 1

        index.close()


def test_search_matters_by_centroid():
    """T2: 按质心向量检索 Matter（LanceDB top-k）。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        index = _make_index(tmpdir)

        # 创建 3 个 Matter，各有不同质心
        dim = 128
        m1 = Matter(matter_id="m-001", title="alpha", centroid=[1.0] + [0.0] * (dim - 1))
        m2 = Matter(matter_id="m-002", title="beta", centroid=[0.0, 1.0] + [0.0] * (dim - 2))
        m3 = Matter(matter_id="m-003", title="gamma", centroid=[0.0, 0.0, 1.0] + [0.0] * (dim - 3))
        index.add_matter(m1)
        index.add_matter(m2)
        index.add_matter(m3)

        # 查询向量接近 m-001
        query = [0.9] + [0.1] * (dim - 1)
        results = index.search_matters(query, k=2)
        assert len(results) <= 2
        assert len(results) >= 1
        # 最相似的应该是 m-001
        assert results[0].matter_id == "m-001"

        index.close()


def test_update_matter_centroid_rolling():
    """T2: 质心滚动更新（简单均值策略）。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        index = _make_index(tmpdir)

        matter = Matter(matter_id="m-001", title="test", centroid=[1.0, 0.0, 0.0])
        index.add_matter(matter)

        # 更新质心
        index.update_matter_centroid("m-001", [0.0, 1.0, 0.0])

        loaded = index.get_matter("m-001")
        assert loaded is not None
        # 简单均值：(1+0)/2=0.5, (0+1)/2=0.5, (0+0)/2=0
        assert loaded.centroid == [0.5, 0.5, 0.0]

        index.close()


def test_clear_removes_matters_and_edges():
    """T2: clear() 清除 matter + edge 数据。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        index = _make_index(tmpdir)

        index.add_matter(Matter(matter_id="m-001", title="test"))
        index.add_edge(MatterEdge(
            matter_id="m-001", target_type=EdgeTargetType.SESSION,
            target_key="u1/a1/s1/",
        ))

        index.clear()

        assert index.get_matter("m-001") is None
        assert len(index.get_edges("m-001")) == 0

        index.close()


def test_split_matter_moves_edges():
    """T4: 拆分 Matter——指定边转移到新 Matter，source 不删除。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        index = _make_index(tmpdir)

        # source Matter 有 3 条边
        index.add_matter(Matter(matter_id="m-src", title="source"))
        e1 = index.assign_manual("m-src", EdgeTargetType.FACT, "fact-1")
        e2 = index.assign_manual("m-src", EdgeTargetType.FACT, "fact-2")
        index.add_edge(MatterEdge(
            matter_id="m-src", target_type=EdgeTargetType.SESSION,
            target_key="u1/a1/s1/", provenance=EdgeProvenance.AUTO,
        ))

        # 拆分：把 e1, e2 转移到新 Matter
        new_matter, moved = index.split_matter("m-src", "m-new", "split result", [e1.edge_id, e2.edge_id])
        assert moved == 2
        assert new_matter.matter_id == "m-new"
        assert new_matter.title == "split result"
        assert new_matter.origin == MatterOrigin.MANUAL

        # source 只剩 1 条边
        src_edges = index.get_edges("m-src")
        assert len(src_edges) == 1
        assert src_edges[0].target_key == "u1/a1/s1/"

        # new 有 2 条边
        new_edges = index.get_edges("m-new")
        assert len(new_edges) == 2
        new_keys = {e.target_key for e in new_edges}
        assert new_keys == {"fact-1", "fact-2"}

        # source Matter 仍在
        assert index.get_matter("m-src") is not None
        index.close()


def test_split_matter_nonexistent_edge():
    """T4: 拆分时指定的 edge_id 不存在 → 跳过，不报错。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        index = _make_index(tmpdir)
        index.add_matter(Matter(matter_id="m-src", title="source"))
        index.assign_manual("m-src", EdgeTargetType.FACT, "fact-1")

        _, moved = index.split_matter("m-src", "m-new", "split", ["nonexistent-id"])
        assert moved == 0
        # 新 Matter 仍创建
        assert index.get_matter("m-new") is not None
        index.close()


def test_detach_edges_for_target():
    """T4: 按 target_key + target_type 摘除归属边。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        index = _make_index(tmpdir)
        index.add_matter(Matter(matter_id="m-001", title="test"))

        index.assign_manual("m-001", EdgeTargetType.SESSION, "u1/a1/s1/")
        index.add_edge(MatterEdge(
            matter_id="m-001", target_type=EdgeTargetType.SESSION,
            target_key="u2/a2/s2/", provenance=EdgeProvenance.AUTO,
        ))

        # 摘除 u1/a1/s1/ 的边
        removed = index.detach_edges_for_target("u1/a1/s1/", EdgeTargetType.SESSION)
        assert removed == 1
        assert len(index.get_edges("m-001")) == 1

        # 不匹配 target_type 不摘除
        removed = index.detach_edges_for_target("u2/a2/s2/", EdgeTargetType.FACT)
        assert removed == 0
        assert len(index.get_edges("m-001")) == 1

        index.close()


# ── T2: 质心加权 + manual 强锚点 + summary 滚动 ──



def test_manual_edge_anchors_centroid():
    """T2: manual 边权重 > auto 边，auto 内容漂移不得把质心拉离用户意志。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        index = _make_index(tmpdir)

        matter = Matter(
            matter_id="m-001", title="test",
            centroid=[1.0, 0.0], centroid_weight=1.0,
        )
        index.add_matter(matter)

        # 手动指定 → 质心朝 [0, 1] 方向移动（manual_weight=3.0）
        index.update_matter_centroid("m-001", [0.0, 1.0], provenance=EdgeProvenance.MANUAL)
        loaded = index.get_matter("m-001")
        # weighted: (1.0*1.0 + 0.0*3.0)/4.0 = 0.25, (0.0*1.0 + 1.0*3.0)/4.0 = 0.75
        assert loaded.centroid == [0.25, 0.75]
        assert loaded.centroid_weight == 4.0

        # auto 内容 [1, 0] 尝试拉回 → manual 锚点生效，不完全拉回
        index.update_matter_centroid("m-001", [1.0, 0.0], provenance=EdgeProvenance.AUTO)
        loaded = index.get_matter("m-001")
        # weighted: (0.25*4.0 + 1.0*1.0)/5.0 = 0.4, (0.75*4.0 + 0.0*1.0)/5.0 = 0.6
        assert loaded.centroid == [0.4, 0.6]
        # 质心仍偏向 manual 方向 [0, 1]
        assert loaded.centroid[1] > loaded.centroid[0]

        index.close()


def test_manual_correction_improves_later_auto_attribution():
    """ADR-0018 DPL: 手动累积 Matter aliases 后，同 proposal 的 fact 经 L3 链入。

    取代旧 cosine 反哺测试（质心不再驱动归属，ADR-0018 §3.4）。DPL 下"手动纠偏改善
    后续归属"= 手动 enrich matter.aliases -> L3 alias-link 命中。
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        index = _make_index(tmpdir)

        matter_a = Matter(
            matter_id="m-a", title="topic A",
            centroid=[0.0, 1.0, 0.0], centroid_weight=1.0,
            aliases=["topic A"],
        )
        index.add_matter(matter_a)

        pipeline = AttributionPipeline()

        # fact X 的 proposal 与 A.aliases 不匹配 -> 不归到 A
        fact_x = Fact(id="f-x", content="some content about topic X",
                      embedding=[0.9, 0.1, 0.0], proposal_titles=["topic X"])
        decision = pipeline.attribute(fact_x, [matter_a])
        assert decision.source != AttributionSource.ALIAS
        assert decision.matter_id != "m-a"

        # 手动把 A 的 aliases 扩到包含 X 的 proposal -> L3 能命中
        matter_a.aliases.append("topic X")
        index.update_matter(matter_a)

        # 现在 fact Y（与 X 同 proposal）应该经 L3 归到 A
        fact_y = Fact(id="f-y", content="similar content about topic X",
                      embedding=[0.85, 0.15, 0.0], proposal_titles=["topic X"])
        decision_y = pipeline.attribute(fact_y, [matter_a])
        assert decision_y.source == AttributionSource.ALIAS
        assert decision_y.matter_id == "m-a"

        index.close()


def test_matter_summary_rolls_with_facts():
    """T2: Matter summary 随归入内容更新、非空。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        index = _make_index(tmpdir)

        matter = Matter(matter_id="m-001", title="test matter")
        index.add_matter(matter)

        for i in range(3):
            fact = Fact(id=f"f-{i}", content=f"fact content number {i} about something")
            index.add_fact(fact)
            index.add_edge(MatterEdge(
                matter_id="m-001",
                target_type=EdgeTargetType.FACT,
                target_key=f"f-{i}",
                provenance=EdgeProvenance.AUTO,
            ))

        index.update_matter_summary("m-001")
        loaded = index.get_matter("m-001")
        assert loaded.summary != ""
        assert "fact content" in loaded.summary

        index.close()


# ── T3: 候选集 top-k + 未归属池消化 ──


def test_attribution_uses_topk_candidates_not_full_scan():
    """T3: 归属判定用 top-k 候选检索，不全量扫。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        index = _make_index(tmpdir)

        dim = 128
        for i in range(20):
            vec = [0.0] * dim
            vec[i] = 1.0
            index.add_matter(Matter(matter_id=f"m-{i:03d}", title=f"matter {i}", centroid=vec))

        query = [0.0] * dim
        query[5] = 0.95
        query[6] = 0.05

        results = index.search_matters(query, k=5)
        assert len(results) <= 5
        assert len(results) >= 1
        assert results[0].matter_id == "m-005"

        index.close()


def test_unassigned_pool_reclusters_into_matter():
    """ADR-0018 §3.3: 同 proposal 的低置信 fact 进池 -> 周期消化按 proposal 标题成簇。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        index = _make_index(tmpdir)

        dim = 128
        for i, fid in enumerate(["f-1", "f-2", "f-3"]):
            vec = [0.0] * dim
            vec[0] = 0.9 + i * 0.01
            vec[1] = 0.8
            # 同 proposal 标题 -> 消化时归一组；不同 logical_turn 才成簇
            fact = Fact(id=fid, content=f"content about topic X variant {i}",
                        embedding=vec, logical_turn=i,
                        proposal_titles=["topic X 事务"])
            index.add_fact(fact)
            index.add_edge(MatterEdge(
                matter_id=UNASSIGNED_MATTER_ID,
                target_type=EdgeTargetType.FACT,
                target_key=fid,
                provenance=EdgeProvenance.AUTO,
            ))

        new_matters = index.digest_unassigned_pool()
        assert new_matters == 1

        all_matters = index.all_matters()
        assert len(all_matters) == 1
        assert all_matters[0].origin == MatterOrigin.AUTO
        # >=2 成员 -> 转正 established
        assert all_matters[0].status == MatterStatus.ACTIVE

        edges = index.get_edges(all_matters[0].matter_id)
        assert len(edges) == 3

        unassigned_edges = index.get_edges(UNASSIGNED_MATTER_ID)
        assert len(unassigned_edges) == 0

        index.close()


def test_scattered_unassigned_stays_pooled():
    """T3: 散乱 fact 仍留池（宁分勿合）。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        index = _make_index(tmpdir)

        dim = 128
        for i, fid in enumerate(["f-1", "f-2", "f-3"]):
            vec = [0.0] * dim
            vec[i * 40] = 0.9
            fact = Fact(id=fid, content=f"content about completely different topic {i}", embedding=vec)
            index.add_fact(fact)
            index.add_edge(MatterEdge(
                matter_id=UNASSIGNED_MATTER_ID,
                target_type=EdgeTargetType.FACT,
                target_key=fid,
                provenance=EdgeProvenance.AUTO,
            ))

        new_matters = index.digest_unassigned_pool(digestion_threshold=0.7)
        assert new_matters == 0

        unassigned_edges = index.get_edges(UNASSIGNED_MATTER_ID)
        assert len(unassigned_edges) == 3

        index.close()
