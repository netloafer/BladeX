"""ADR-0021 T1 验收：scope + 敏感度字段 + Memory Index 可见性过滤 + 重建等价。

嵌入 mock，不跑真模型。覆盖计划卡 T1 四项验收：
① 个人模式（可见集合=单 personal）检索结果与现状逐条一致；
② 旧 Memory Hub 数据 rebuild 后 Fact 全部落默认 scope/ceiling；
③（in-list 延迟基准见 scripts/benchmark_index_visibility.py）；
④ 重建等价性回归。
"""

import tempfile
from pathlib import Path

from bladex_proxy.models import Identity, Turn, TurnStatus
from bladex_proxy.storage.memory_index import MemoryIndex
from bladex_proxy.storage.memory_hub import MemoryHub


class MockEmbedder:
    """固定向量 mock embedder。"""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            h = hash(text)
            out.append([((h >> i) & 1) * 1.0 for i in range(self._dim)])
        return out


def _make_turn(user_id: str, session_id: str, user_msg: str, ledger_key: str = "") -> Turn:
    return Turn(
        identity=Identity(user_id=user_id, agent_id="a1", session_id=session_id),
        model="test",
        request_messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": user_msg},
        ],
        response_text="OK",
        status=TurnStatus.OK,
    )


# ── ① 个人模式可见性过滤 = 现状等价 ──


def test_personal_visibility_equivalent_to_user_id():
    """visibility=["personal:u1"] 与 user_id="u1" 检索结果逐条一致。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        embedder = MockEmbedder()
        ledger = MemoryHub(Path(tmpdir) / "ledger")
        ledger.open()
        index = MemoryIndex(Path(tmpdir) / "index", embedder=embedder)
        index.open()

        ledger.put("u1/a1/s1/k1", _make_turn("u1", "s1", "I love BladeX memory layer.", "k1"))
        ledger.put("u2/a1/s1/k1", _make_turn("u2", "s1", "Another user loves BladeX too.", "k1"))
        index.rebuild_from_hub(ledger)

        via_uid = index.search("BladeX", top_k=10, user_id="u1")
        via_vis = index.search("BladeX", top_k=10, visibility=["personal:u1"])

        assert [f.id for f in via_uid] == [f.id for f in via_vis]
        # 只命中 u1 的 fact，不串号
        assert all(f.source_user_id == "u1" for f in via_vis)
        index.close()
        ledger.close()


def test_visibility_isolates_users():
    """两用户 fact 互不可见（visibility 过滤生效）。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        embedder = MockEmbedder()
        ledger = MemoryHub(Path(tmpdir) / "ledger")
        ledger.open()
        index = MemoryIndex(Path(tmpdir) / "index", embedder=embedder)
        index.open()

        ledger.put("u1/a1/s1/k1", _make_turn("u1", "s1", "Alpha project BladeX memory.", "k1"))
        ledger.put("u2/a1/s1/k1", _make_turn("u2", "s1", "Beta project BladeX memory.", "k1"))
        index.rebuild_from_hub(ledger)

        u1_facts = index.search("BladeX", top_k=10, visibility=["personal:u1"])
        u2_facts = index.search("BladeX", top_k=10, visibility=["personal:u2"])
        assert all(f.source_user_id == "u1" for f in u1_facts)
        assert all(f.source_user_id == "u2" for f in u2_facts)
        assert {f.id for f in u1_facts}.isdisjoint({f.id for f in u2_facts})
        index.close()
        ledger.close()


def test_multi_pid_visibility_sees_merged_identities():
    """多 key 归一同一 principal（多设备）：visibility 含多个 legacy pid 都可见。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        embedder = MockEmbedder()
        ledger = MemoryHub(Path(tmpdir) / "ledger")
        ledger.open()
        index = MemoryIndex(Path(tmpdir) / "index", embedder=embedder)
        index.open()

        # 两个 legacy pid 的 fact（多设备场景）
        ledger.put("pid-a/a1/s1/k1", _make_turn("pid-a", "s1", "BladeX design note alpha.", "k1"))
        ledger.put("pid-b/a1/s1/k1", _make_turn("pid-b", "s1", "BladeX design note beta.", "k1"))
        index.rebuild_from_hub(ledger)

        # 归一后可见集合含两个 personal pid
        seen = index.search("BladeX", top_k=10, visibility=["personal:pid-a", "personal:pid-b"])
        seen_ids = {f.source_user_id for f in seen}
        assert seen_ids == {"pid-a", "pid-b"}
        index.close()
        ledger.close()


def test_empty_visibility_fail_closed():
    """显式空可见集合 -> fail-closed 返回空（不跨身份泄漏）。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        embedder = MockEmbedder()
        ledger = MemoryHub(Path(tmpdir) / "ledger")
        ledger.open()
        index = MemoryIndex(Path(tmpdir) / "index", embedder=embedder)
        index.open()
        ledger.put("u1/a1/s1/k1", _make_turn("u1", "s1", "BladeX secret note.", "k1"))
        index.rebuild_from_hub(ledger)

        # 仅 team/org scope（本卡无对应 fact）-> 空
        assert index.search("BladeX", top_k=10, visibility=["team:rd"]) == []
        index.close()
        ledger.close()


# ── ② 旧 Memory Hub 数据 rebuild 后 Fact 落默认 scope/ceiling ──


def test_old_data_rebuild_falls_to_default_scope_ceiling():
    """旧 Turn（无 sensitivity/scope 概念）rebuild 后 Fact 落默认值。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        embedder = MockEmbedder()
        ledger = MemoryHub(Path(tmpdir) / "ledger")
        ledger.open()
        index = MemoryIndex(Path(tmpdir) / "index", embedder=embedder)
        index.open()

        # 旧 Turn：不设 sensitivity（用默认 normal）
        turn = _make_turn("u1", "s1", "Legacy user note about BladeX storage.", "k1")
        assert turn.sensitivity == "normal"  # 默认值
        ledger.put("u1/a1/s1/k1", turn)
        index.rebuild_from_hub(ledger)

        facts = index.all_facts()
        assert facts
        for f in facts:
            # 默认 exposure_ceiling=public（现状等价：个人流量发公网）
            assert f.exposure_ceiling == "public"
            # scope 默认空（personal 由 source_user_id 体现，T5 consolidator 显式设）
        index.close()
        ledger.close()


# ── ④ 重建等价性：full rebuild x2 -> Fact 集合逐字段一致（scope/ceiling 含）──


def test_rebuild_equivalence_scope_ceiling_stable():
    """full rebuild 两次 -> Fact 的 scope/exposure_ceiling 逐条一致。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        embedder = MockEmbedder()
        ledger = MemoryHub(Path(tmpdir) / "ledger")
        ledger.open()
        index = MemoryIndex(Path(tmpdir) / "index", embedder=embedder)
        index.open()

        ledger.put("u1/a1/s1/k1", _make_turn("u1", "s1", "Equivalence test BladeX note one.", "k1"))
        ledger.put("u1/a1/s1/k2", _make_turn("u1", "s1", "Equivalence test BladeX note two.", "k2"))

        index.rebuild_from_hub(ledger, full=True)
        first = {f.id: (f.scope, f.exposure_ceiling) for f in index.all_facts()}
        index.rebuild_from_hub(ledger, full=True)
        second = {f.id: (f.scope, f.exposure_ceiling) for f in index.all_facts()}

        assert first == second
        index.close()
        ledger.close()
