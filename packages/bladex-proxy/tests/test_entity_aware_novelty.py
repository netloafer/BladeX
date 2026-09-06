"""实体感知 novelty check 回归测试（方案 B）。

背景：纯 cosine novelty（阈值 0.95）对"同义不同实体"的查询误判重复--multilingual-e5-large
让 cos('今天北京天气怎么样','今天上海天气怎么样')=0.9652>0.95，导致不同实体的同类查询
被判重复丢弃，Claude Code 522 turn 只产出 20 个 fact（提取率 3%）。

方案 B：cosine 召回 top-k 后比实体集合，Jaccard >= entity_overlap_threshold 才判重；
实体有差异仍判新颖。entities 缺失退化为纯 cosine（向后兼容）。
"""

from __future__ import annotations

import tempfile

from bladex_proxy.storage.memory_index import (
    FastEmbedAdapter,
    MemoryIndex,
    _LanceDBNoveltyChecker,
)
from bladex_core.consolidation_proxy import ProxyConsolidator


def _make_index(tmpdir: str, **kw) -> MemoryIndex:
    """临时 Memory Index 实例（可写，用于真实 LanceDB 路径测试）。"""
    embedder = FastEmbedAdapter()
    defaults = dict(
        novelty_threshold=0.95,
        entity_aware_novelty=True,
        entity_overlap_threshold=0.6,
        novelty_topk=5,
    )
    defaults.update(kw)
    index = MemoryIndex(f"{tmpdir}/index", embedder=embedder, read_only=False, **defaults)
    index.open()
    return index


def _add_fact(index: MemoryIndex, content: str, entities: list[str], embedding: list[float]) -> None:
    """直接构造 Fact 写入 Memory Index（绕过 distiller，用给定 embedding）。"""
    from bladex_core.fact import Fact
    import uuid
    f = Fact(
        id=f"fact-{uuid.uuid4().hex[:8]}",
        content=content,
        category="event",
        entities=entities,
        scope="personal:test",
        embedding=embedding,
    )
    index.add_fact(f)
    # lancedb 0.33 无 reload；add 后重新 open_table 让 search 读到新数据
    index._table = index._lancedb.open_table("facts")


# ── _LanceDBNoveltyChecker 实体感知 ──


def test_lancedb_entity_aware_different_entities_novel(tmp_path):
    """cosine 高但实体不同 -> 仍新颖（核心修复：北京 vs 上海场景）。"""
    with tempfile.TemporaryDirectory() as d:
        index = _make_index(d)
        emb = index._embedder.embed(["今天北京天气怎么样"])[0]
        _add_fact(index, "今天上海天气怎么样", ["上海", "天气"], emb.copy())
        checker = _LanceDBNoveltyChecker(
            index._table, 0.95, entity_aware=True, entity_overlap_threshold=0.6, topk=5,
        )
        # 北京 vs 上海：cosine 高但实体 {北京} vs {上海} Jaccard=0 < 0.6 -> 新颖
        assert checker.is_novel(emb, entities=["北京", "天气"]) is True


def test_lancedb_entity_aware_same_entities_duplicate(tmp_path):
    """cosine 高且实体高度重叠 -> 判重。"""
    with tempfile.TemporaryDirectory() as d:
        index = _make_index(d)
        emb = index._embedder.embed(["泰山啤酒破产案3000万共益债"])[0]
        _add_fact(index, "泰山啤酒破产案的3000万共益债", ["泰山啤酒", "3000万共益债", "政府"], emb.copy())
        checker = _LanceDBNoveltyChecker(
            index._table, 0.95, entity_aware=True, entity_overlap_threshold=0.6, topk=5,
        )
        # 同实体 -> Jaccard=1.0 >= 0.6 -> 判重
        assert checker.is_novel(emb, entities=["泰山啤酒", "3000万共益债"]) is False


def test_lancedb_entity_aware_disabled_falls_back_to_cosine(tmp_path):
    """entity_aware=False -> 退化为纯 cosine（同实体不同也会因 cosine 高判重）。"""
    with tempfile.TemporaryDirectory() as d:
        index = _make_index(d)
        emb = index._embedder.embed(["今天北京天气怎么样"])[0]
        _add_fact(index, "今天上海天气怎么样", ["上海", "天气"], emb.copy())
        checker = _LanceDBNoveltyChecker(
            index._table, 0.95, entity_aware=False, entity_overlap_threshold=0.6, topk=5,
        )
        # 关闭实体感知 + cosine 高 -> 判重（旧行为）
        assert checker.is_novel(emb, entities=["北京", "天气"]) is False


def test_lancedb_empty_entities_degrades_to_cosine(tmp_path):
    """候选无 entities -> 退化为纯 cosine。"""
    with tempfile.TemporaryDirectory() as d:
        index = _make_index(d)
        emb = index._embedder.embed(["今天北京天气怎么样"])[0]
        _add_fact(index, "今天上海天气怎么样", ["上海", "天气"], emb.copy())
        checker = _LanceDBNoveltyChecker(
            index._table, 0.95, entity_aware=True, entity_overlap_threshold=0.6, topk=5,
        )
        # 候选 entities=[] -> 退化纯 cosine -> 高相似判重
        assert checker.is_novel(emb, entities=[]) is False
        assert checker.is_novel(emb, entities=None) is False


def test_lancedb_low_cosine_always_novel(tmp_path):
    """cosine 低于阈值 -> 直接新颖（不看实体）。"""
    with tempfile.TemporaryDirectory() as d:
        index = _make_index(d)
        emb_a = index._embedder.embed(["今天北京天气怎么样"])[0]
        emb_b = index._embedder.embed(["如何用 Python 实现快速排序算法"])[0]
        _add_fact(index, "今天上海天气怎么样", ["上海", "天气"], emb_a.copy())
        checker = _LanceDBNoveltyChecker(
            index._table, 0.95, entity_aware=True, entity_overlap_threshold=0.6, topk=5,
        )
        # 完全不同主题，cosine 低 -> 新颖
        assert checker.is_novel(emb_b, entities=["Python", "排序"]) is True


def test_entity_jaccard_static():
    """_entity_jaccard 边界：空集返回 -1（退化信号）。"""
    assert _LanceDBNoveltyChecker._entity_jaccard([], ["a"]) == -1.0
    assert _LanceDBNoveltyChecker._entity_jaccard(["a"], []) == -1.0
    assert _LanceDBNoveltyChecker._entity_jaccard(None, ["a"]) == -1.0
    # 大小写归一
    assert _LanceDBNoveltyChecker._entity_jaccard(["北京"], ["北京"]) == 1.0
    j = _LanceDBNoveltyChecker._entity_jaccard(["a", "b"], ["b", "c"])
    assert abs(j - 1/3) < 1e-6  # {b} / {a,b,c}


# ── ProxyConsolidator._is_novel（within-batch brute-force）──


def _make_consolidator(entity_aware=True, overlap=0.6, threshold=0.95):
    return ProxyConsolidator(
        embedder=None,
        novelty_threshold=threshold,
        entity_aware=entity_aware,
        entity_overlap_threshold=overlap,
    )


def test_batch_is_novel_different_entities_novel():
    """within-batch：cosine 高但实体不同 -> 仍新颖。"""
    c = _make_consolidator()
    emb = [1.0, 0.0]
    # 已有 fact 同向量（cosine=1.0），但实体不同
    assert c._is_novel(emb, [emb], entities=["北京"], existing_entities=[["上海"]]) is True


def test_batch_is_novel_same_entities_duplicate():
    """within-batch：cosine 高且实体重叠 -> 判重。"""
    c = _make_consolidator()
    emb = [1.0, 0.0]
    assert c._is_novel(emb, [emb], entities=["北京", "天气"], existing_entities=[["北京", "天气"]]) is False


def test_batch_is_novel_no_entities_degrades_cosine():
    """within-batch：无 entities -> 退化纯 cosine。"""
    c = _make_consolidator()
    emb = [1.0, 0.0]
    # cosine 高 + 无实体感知 -> 判重
    assert c._is_novel(emb, [emb], entities=[], existing_entities=[]) is False
    assert c._is_novel(emb, [emb]) is False  # 完全不传 entities


def test_batch_is_novel_low_cosine_novel():
    """within-batch：cosine 低 -> 新颖。"""
    c = _make_consolidator()
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert c._is_novel(b, [a], entities=["x"], existing_entities=[["y"]]) is True


def test_batch_is_novel_entity_aware_disabled():
    """entity_aware=False -> 纯 cosine（实体不同也判重）。"""
    c = _make_consolidator(entity_aware=False)
    emb = [1.0, 0.0]
    assert c._is_novel(emb, [emb], entities=["北京"], existing_entities=[["上海"]]) is False
