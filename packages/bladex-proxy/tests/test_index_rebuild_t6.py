"""Memory Index 重建等价性测试（ADR-0012 T6, §3.6/§4）。"""

import tempfile
from pathlib import Path

from bladex_core.matter import (
    EdgeProvenance,
    EdgeTargetType,
    Matter,
    MatterOrigin,
)
from bladex_proxy.models import (
    AdminEventType,
    Identity,
    TombstoneTargetType,
    Turn,
    TurnStatus,
)
from bladex_proxy.storage.memory_index import MemoryIndex
from bladex_proxy.storage.memory_hub import MemoryHub


class MockEmbedder:
    """Mock embedder for rebuild tests."""

    def embed(self, texts):
        # Return deterministic vectors based on text hash
        results = []
        for t in texts:
            h = hash(t) % 100
            vec = [0.0] * 64
            vec[h % 64] = 1.0
            results.append(vec)
        return results

    @property
    def available(self):
        return True


def _make_hub_with_turns(tmpdir: str) -> tuple[MemoryHub, list[str]]:
    """创建 Memory Hub 并写入几个 turn，返回 (ledger, turn_keys)。"""
    ledger = MemoryHub(Path(tmpdir) / "rocksdb")
    ledger.open()

    keys = []
    for i in range(3):
        identity = Identity(
            user_id="u1", agent_id="a1", session_id="s1", turn_index=i,
        )
        turn = Turn(
            identity=identity,
            model="test",
            request_messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": f"This is a test message about topic {i}. It has enough length for consolidation."},
            ],
            response_text=f"Response {i}",
            status=TurnStatus.OK,
        )
        key = identity.storage_key(f"171990720{i}-0")
        ledger.put(key, turn)
        keys.append(key)

    return ledger, keys


def test_rebuild_excludes_tombstoned():
    """T6: 重建时跳过被墓碑覆盖的 turn。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger, keys = _make_hub_with_turns(tmpdir)

        # 给第二个 turn 写墓碑
        ledger.append_tombstone(TombstoneTargetType.TURN, keys[1])

        # 重建 Memory Index
        index = MemoryIndex(Path(tmpdir) / "index", embedder=MockEmbedder(), read_only=False)
        index.open()
        index.rebuild_from_hub(ledger, full=True)

        # 墓碑覆盖的 turn 不应产生 fact
        facts = index.all_facts()
        for f in facts:
            assert f.source_ledger_key != keys[1], f"Tombstoned turn should not produce facts, got {f.source_ledger_key}"

        index.close()
        ledger.close()


def test_rebuild_replays_admin_events():
    """T6: 全量重建时重放管理事件，手动映射不丢失。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger, keys = _make_hub_with_turns(tmpdir)

        # 写管理事件：创建 Matter + 指定归属
        ledger.append_admin_event(
            AdminEventType.MATTER_CREATE, "m-manual-001",
            title="手动创建的 Matter", summary="用户手动创建",
        )
        ledger.append_admin_event(
            AdminEventType.MATTER_ASSIGN, "m-manual-001",
            target_key="u1/a1/s1/",
            target_type="session",
        )

        # 全量重建
        index = MemoryIndex(Path(tmpdir) / "index", embedder=MockEmbedder(), read_only=False)
        index.open()
        index.rebuild_from_hub(ledger, full=True)

        # 手动创建的 Matter 在
        matter = index.get_matter("m-manual-001")
        assert matter is not None
        assert matter.title == "手动创建的 Matter"
        assert matter.origin == MatterOrigin.MANUAL

        # 手动归属边在
        edges = index.get_edges("m-manual-001")
        assert len(edges) >= 1
        manual_edges = [e for e in edges if e.provenance == EdgeProvenance.MANUAL]
        assert len(manual_edges) >= 1

        index.close()
        ledger.close()


def test_cascade_delete_clears_vectors_and_edges():
    """T6: 墓碑级联清除——删除 Fact + 向量 + 归属边。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger, keys = _make_hub_with_turns(tmpdir)

        index = MemoryIndex(Path(tmpdir) / "index", embedder=MockEmbedder(), read_only=False)
        index.open()
        index.rebuild_from_hub(ledger, full=True)

        # 找到第一个 turn 的 fact
        facts_before = index.all_facts()
        assert len(facts_before) > 0

        target_key = keys[0]
        facts_for_turn = [f for f in facts_before if f.source_ledger_key == target_key]

        if facts_for_turn:
            # 手动添加归属边
            fact_id = facts_for_turn[0].id
            index.add_edge(__import__("bladex_core.matter", fromlist=["MatterEdge"]).MatterEdge(
                matter_id="m-001", target_type=EdgeTargetType.FACT,
                target_key=fact_id,
            ))

            # 级联删除
            removed = index.cascade_delete_turn(target_key)
            assert removed > 0

            # Fact 已删
            facts_after = index.all_facts()
            for f in facts_after:
                assert f.source_ledger_key != target_key

            # 归属边已删
            edges = index.get_edges("m-001")
            for e in edges:
                assert e.target_key != fact_id

        index.close()
        ledger.close()


def test_full_rebuild_preserves_fact_level_manual_assignment():
    """G2: full rebuild 后 fact 级手动映射不丢失（fact_id 确定性生成）。

    修复前：consolidate_turns 用 uuid.uuid4() 生成 fact_id，每次重建都变，
    _replay_admin_events 重放的 matter_assign 里 target_key 是旧 fact_id，
    新 fact 永远匹配不到 -> fact 级手动映射丢失。

    修复后：fact_id = hash(source_ledger_key + content)，确定性生成，
    同一 Memory Hub turn 同一内容 -> 同一 fact_id -> 手动映射存活。
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger, keys = _make_hub_with_turns(tmpdir)

        # 第一次重建
        index = MemoryIndex(Path(tmpdir) / "index", embedder=MockEmbedder(), read_only=False)
        index.open()
        index.rebuild_from_hub(ledger, full=True)

        # 拿到第一条 fact 的 id
        facts = index.all_facts()
        assert len(facts) > 0
        fact_id = facts[0].id
        fact_content = facts[0].content

        # 手动创建 Matter + 指定 fact 级归属
        ledger.append_admin_event(
            AdminEventType.MATTER_CREATE, "m-fact-manual",
            title="手动关联到具体 fact",
        )
        ledger.append_admin_event(
            AdminEventType.MATTER_ASSIGN, "m-fact-manual",
            target_key=fact_id, target_type="fact",
        )

        # 全量重建
        index.rebuild_from_hub(ledger, full=True)

        # 手动 Matter 在
        matter = index.get_matter("m-fact-manual")
        assert matter is not None
        assert matter.origin == MatterOrigin.MANUAL

        # fact 级手动归属边在，且 target_key 等于原 fact_id
        edges = index.get_edges("m-fact-manual")
        manual_edges = [e for e in edges if e.provenance == EdgeProvenance.MANUAL]
        assert len(manual_edges) >= 1
        assert any(e.target_key == fact_id for e in manual_edges), (
            f"Manual edge should point to original fact_id={fact_id}, "
            f"got target_keys={[e.target_key for e in manual_edges]}"
        )

        # 同一 ledger_key + content 的 fact_id 在重建后不变
        rebuilt_facts = index.all_facts()
        matching = [f for f in rebuilt_facts if f.content == fact_content]
        assert len(matching) >= 1
        assert matching[0].id == fact_id, (
            f"fact_id should be deterministic: expected {fact_id}, got {matching[0].id}"
        )

        index.close()
        ledger.close()


def test_fact_id_is_deterministic_across_rebuilds():
    """G2: 同一 Memory Hub 数据两次重建，fact_id 完全一致。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger, keys = _make_hub_with_turns(tmpdir)

        # 第一次重建
        index = MemoryIndex(Path(tmpdir) / "index", embedder=MockEmbedder(), read_only=False)
        index.open()
        index.rebuild_from_hub(ledger, full=True)
        first_ids = sorted(f.id for f in index.all_facts())
        index.close()

        # 第二次全量重建
        p2b = MemoryIndex(Path(tmpdir) / "p2b", embedder=MockEmbedder(), read_only=False)
        p2b.open()
        p2b.rebuild_from_hub(ledger, full=True)
        second_ids = sorted(f.id for f in p2b.all_facts())
        p2b.close()
        ledger.close()

        assert first_ids == second_ids, (
            f"fact_ids should be deterministic across rebuilds:\n"
            f"first:  {first_ids}\nsecond: {second_ids}"
        )


def test_full_rebuild_preserves_manual_edges():
    """T6: 全量重建后 manual 边全在、已删数据零复活、聚合视图一致。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger, keys = _make_hub_with_turns(tmpdir)

        # 第一次重建
        index = MemoryIndex(Path(tmpdir) / "index", embedder=MockEmbedder(), read_only=False)
        index.open()
        index.rebuild_from_hub(ledger, full=True)

        # 手动创建 Matter + 指定归属
        index.add_matter(Matter(
            matter_id="m-manual", title="manual matter",
            origin=MatterOrigin.MANUAL,
        ))
        index.assign_manual("m-manual", EdgeTargetType.SESSION, "u1/a1/s1/")

        # 写管理事件到 Memory Hub（模拟通过 API 操作）
        ledger.append_admin_event(
            AdminEventType.MATTER_CREATE, "m-manual",
            title="manual matter",
        )
        ledger.append_admin_event(
            AdminEventType.MATTER_ASSIGN, "m-manual",
            target_key="u1/a1/s1/", target_type="session",
        )

        # 删除一个 turn
        ledger.append_tombstone(TombstoneTargetType.TURN, keys[0])

        # 全量重建
        index.rebuild_from_hub(ledger, full=True)

        # manual Matter 在
        matter = index.get_matter("m-manual")
        assert matter is not None
        assert matter.origin == MatterOrigin.MANUAL

        # manual 边在
        edges = index.get_edges("m-manual")
        manual_edges = [e for e in edges if e.provenance == EdgeProvenance.MANUAL]
        assert len(manual_edges) >= 1

        # 已删 turn 的 fact 零复活
        facts = index.all_facts()
        for f in facts:
            assert f.source_ledger_key != keys[0], "Tombstoned turn fact should not revive"

        # 聚合视图一致
        view = index.get_matter_aggregate_view("m-manual")
        assert view is not None
        assert "u1/a1/s1/" in view.session_keys

        index.close()
        ledger.close()


def test_rebuild_attributes_facts_to_matters():
    """T6: 重建时自动归属判定——新 fact 创建 Matter 或归到已有 Matter。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger, keys = _make_hub_with_turns(tmpdir)

        index = MemoryIndex(Path(tmpdir) / "index", embedder=MockEmbedder(), read_only=False)
        index.open()
        index.rebuild_from_hub(ledger, full=True)

        # 重建后应该有 Matter（auto 创建）
        matters = index.all_matters()
        # 至少有一些 Matter 被创建（取决于 fact 内容长度）
        # 如果有 matter，应该有归属边
        if matters:
            total_edges = sum(len(index.get_edges(m.matter_id)) for m in matters)
            assert total_edges > 0, "Matters should have attribution edges"

        index.close()
        ledger.close()


# ── review 20260707: 安全重建（先 consolidate 后 clear）──


class FailingEmbedder:
    """embed() 总是抛异常的 mock embedder（模拟 consolidation 失败）。"""

    def embed(self, texts):
        raise RuntimeError("simulated embedder failure")

    @property
    def available(self):
        return True


def test_full_rebuild_preserves_data_on_consolidation_failure():
    """安全重建：full=True consolidation 失败时，旧 Memory Index 数据完整保留。

    review 20260707：原来 clear() 在开头，consolidation 失败 = 数据全丢。
    改为先 consolidate 成功后才 clear，失败时旧数据不动。
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger, keys = _make_hub_with_turns(tmpdir)

        # 第一次重建：成功，Memory Index 有 facts
        index = MemoryIndex(Path(tmpdir) / "index", embedder=MockEmbedder(), read_only=False)
        index.open()
        index.rebuild_from_hub(ledger, full=True)
        facts_before = index.all_facts()
        assert len(facts_before) > 0, "first rebuild should produce facts"

        # 换成会失败的 embedder，再跑 full rebuild
        index._embedder = FailingEmbedder()
        result = index.rebuild_from_hub(ledger, full=True)

        # consolidation 失败 → 返回 0
        assert result == 0

        # 旧数据完整保留（clear 没被执行）
        facts_after = index.all_facts()
        assert len(facts_after) == len(facts_before), (
            "old facts should be preserved when consolidation fails"
        )
        # 确认是同样的 facts（按 id 比对）
        ids_before = {f.id for f in facts_before}
        ids_after = {f.id for f in facts_after}
        assert ids_before == ids_after

        index.close()
        ledger.close()


def test_full_rebuild_clears_after_successful_consolidation():
    """安全重建：full=True consolidation 成功时，旧 Memory Index 被清空后存入新 facts。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger, keys = _make_hub_with_turns(tmpdir)

        index = MemoryIndex(Path(tmpdir) / "index", embedder=MockEmbedder(), read_only=False)
        index.open()

        # 第一次重建
        index.rebuild_from_hub(ledger, full=True)
        facts_first = index.all_facts()
        assert len(facts_first) > 0

        # 第二次 full rebuild（相同数据，应产生相同 facts）
        result = index.rebuild_from_hub(ledger, full=True)

        # 成功 → 有新 facts
        assert result > 0

        # facts 存在（旧的被 clear，新的被存入）
        facts_second = index.all_facts()
        assert len(facts_second) > 0

        # full rebuild 用 existing=[]，所以不会因为 novelty 检测跳过
        # facts 数量应与第一次一致（相同输入）
        assert len(facts_second) == len(facts_first), (
            "full rebuild with same Memory Hub data should produce same fact count"
        )

        index.close()
        ledger.close()
