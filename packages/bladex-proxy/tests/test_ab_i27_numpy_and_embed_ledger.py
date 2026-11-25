"""批 AB · MQ-I27：矩阵化不改判定 + 嵌入台账。

| 用例 | 判据 |
|---|---|
| `test_novelty_matrix_matches_legacy_formula` | 随机向量流上，增量矩阵与修前算式（每条重建）逐位相同 |
| `test_novelty_matrix_grows_with_the_same_list_only` | 换 list 对象 / 变短 ⇒ 重来，不串 |
| `test_novelty_parallel_arrays_stay_in_step` | M0-1：同文第二条必须被判重（三个平行数组同步增长） |
| `test_neighbors_match_scalar_cosine` | `within` 与逐条 `_cosine` 的集合、顺序相同，相似度差 < 1e-9 |
| `test_neighbors_edge_cases_like_cosine` | 零向量 / 维度不一致 / discard / 无向量条目 |
| `test_indexed_running_matches_plain_list` | `_neighbors_for` 传 IndexedFacts 与普通 list 结果相同 |
| `test_embed_ledger_second_run_computes_nothing` | 第二跑 0 次计算；同文只算一次；批大小恒为 1 |
| `test_embed_ledger_key_has_model_identity` | 换模型身份 ⇒ 不读旧向量 |
| `test_embed_ledger_cold_and_hit_return_same_value` | 冷算与命中逐位相同 |
| `test_embed_cache_is_not_a_fact` | `embedcache/` 不进 fact_count / all_facts |
| `test_embed_cache_is_harvested` | 中断暂存的台账会被收割 |
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
from bladex_core.adjudication import _cosine
from bladex_core.consolidation_proxy import InRunNeighborSource, ProxyConsolidator
from bladex_core.fact import Fact
from bladex_core.vector_matrix import FactVectors, IndexedFacts, UnitMatrix


def _vecs(n: int, dim: int = 16, seed: int = 7) -> list[list[float]]:
    r = random.Random(seed)
    return [[r.gauss(0, 1) for _ in range(dim)] for _ in range(n)]


def _legacy_sims(q: list[float], rows: list[list[float]]) -> Any:
    qv = np.asarray(q, dtype=np.float32)
    qn = qv / (float(np.linalg.norm(qv)) or 1.0)
    mat = np.stack([np.asarray(e, dtype=np.float32) for e in rows])
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (mat / norms) @ qn, qn


def test_novelty_matrix_matches_legacy_formula() -> None:
    rows: list[list[float]] = []
    m = UnitMatrix()
    for v in _vecs(3000):               # 越过预分配 1024 的扩容点
        if rows:
            want, qn = _legacy_sims(v, rows)
            m.sync(rows)
            got = m.sims(qn)
            assert got.shape == want.shape
            assert np.abs(got - want).max() < 1e-6
        rows.append(v)


def test_novelty_matrix_grows_with_the_same_list_only() -> None:
    a, b = _vecs(5, seed=1), _vecs(3, seed=2)
    m = UnitMatrix()
    m.sync(a)
    assert m.sims(np.ones(16, dtype=np.float32)).shape == (5,)
    m.sync(b)
    assert m.sims(np.ones(16, dtype=np.float32)).shape == (3,)
    del b[2:]                           # 同一对象变短
    m.sync(b)
    assert m.sims(np.ones(16, dtype=np.float32)).shape == (2,)


class _Emb:
    model_identity = "test:emb"
    model_name = "emb"
    available = True

    def __init__(self) -> None:
        self.batches: list[int] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(len(texts))
        out = []
        for t in texts:
            v = [0.0] * 8
            for ch in t:
                v[ord(ch) % 8] += 1.0
            out.append(v)
        return out


def test_novelty_parallel_arrays_stay_in_step() -> None:
    c = ProxyConsolidator(embedder=_Emb(), novelty_threshold=0.98)
    embs: list[list[float]] = []
    ents: list[list[str]] = []
    texts: list[str] = []
    v = [1.0, 2.0, 3.0]
    assert c._is_novel(v, embs, entities=["a"], existing_entities=ents,
                       content="路径 /a/b.py", existing_texts=texts)
    embs.append(v)
    ents.append(["a"])
    texts.append("路径 /a/b.py")
    assert not c._is_novel(v, embs, entities=["a"], existing_entities=ents,
                           content="路径 /a/b.py", existing_texts=texts)
    # 实体不同 ⇒ 判新颖；高相似的第二行也要看到（flatnonzero 与 enumerate 同序）
    embs.append([1.0, 2.0, 3.0001])
    ents.append(["c"])
    texts.append("别的")
    assert c._is_novel(v, embs, entities=["x"], existing_entities=ents,
                       content="路径", existing_texts=texts)


def _fact(i: int, emb: list[float] | None) -> Fact:
    return Fact(id=f"f{i}", content=f"c{i}", embedding=emb or [])


def test_neighbors_match_scalar_cosine() -> None:
    base = np.asarray(_vecs(1, dim=32)[0])
    r = np.random.default_rng(3)
    rows = [(base + r.normal(0, s, 32)).tolist() for s in r.uniform(0.01, 1.2, 1500)]
    facts = [_fact(i, v) for i, v in enumerate(rows)]
    fv = FactVectors(facts)
    for q in [base.tolist(), rows[0], rows[700]]:
        want = [(f, _cosine(q, f.embedding)) for f in facts]
        want = [(f, s) for f, s in want if s >= 0.9]
        got = fv.within(q, 0.9)
        assert [f.id for f, _ in got] == [f.id for f, _ in want]
        assert max((abs(a[1] - b[1]) for a, b in zip(got, want, strict=True)), default=0) < 1e-9


def test_neighbors_edge_cases_like_cosine() -> None:
    zero, ok, none = _fact(0, [0.0, 0.0]), _fact(1, [1.0, 0.0]), _fact(2, None)
    src = InRunNeighborSource()
    for f in (zero, ok, none):
        src.add(f)
    assert [f.id for f, _ in src.neighbors([1.0, 0.0], min_similarity=0.5)] == ["f1"]
    assert src.neighbors([0.0, 0.0], min_similarity=0.5) == []
    assert src.neighbors([1.0, 0.0, 0.0], min_similarity=0.5) == []     # 维度不一致 ⇒ 0
    src.discard("f1")
    assert src.neighbors([1.0, 0.0], min_similarity=0.5) == []
    mixed = FactVectors([_fact(3, [1.0, 0.0]), _fact(4, [1.0, 0.0, 0.0])])  # 混维回落
    assert [f.id for f, _ in mixed.within([1.0, 0.0], 0.5)] == ["f3"]
    mixed.discard("f3")
    assert mixed.within([1.0, 0.0], 0.5) == []


def test_indexed_running_matches_plain_list() -> None:
    c = ProxyConsolidator(embedder=_Emb())
    rows = _vecs(200, dim=8)
    facts = [_fact(i, v) for i, v in enumerate(rows)]
    indexed = IndexedFacts(facts[:100])
    for f in facts[100:]:
        indexed.append(f)
    for probe in (facts[3], facts[150]):
        a = c._neighbors_for(probe, list(facts))
        b = c._neighbors_for(probe, indexed)
        assert [f.id for f, _ in a] == [f.id for f, _ in b]


# ── 嵌入台账 ─────────────────────────────────────────────────────────────────

def _index(tmp: Path, emb: Any = None) -> Any:
    from bladex_proxy.storage.memory_index import MemoryIndex

    idx = MemoryIndex(tmp / "idx", embedder=emb or _Emb())
    idx.open()
    return idx


def test_embed_ledger_second_run_computes_nothing(tmp_path: Path) -> None:
    emb = _Emb()
    idx = _index(tmp_path, emb)
    c = ProxyConsolidator(embedder=emb, embed_ledger=idx)
    emb.batches.clear()
    first = c._embed_chunked(["甲", "乙", "甲", "丙"], phase="candidates")
    assert emb.batches == [1, 1, 1], "同文只算一次，批大小恒为 1"
    emb.batches.clear()
    second = c._embed_chunked(["丙", "甲", "乙", "甲"], phase="candidates")
    assert emb.batches == []
    assert second[1] == first[0] and second[0] == first[3]
    idx.close()


def test_embed_ledger_key_has_model_identity(tmp_path: Path) -> None:
    idx = _index(tmp_path)
    idx.put_embedding("model-a", "t", [1.0, 2.0])
    assert idx.get_embeddings("model-a", ["t"]) == {"t": [1.0, 2.0]}
    assert idx.get_embeddings("model-b", ["t"]) == {}
    idx.close()


def test_embed_ledger_cold_and_hit_return_same_value(tmp_path: Path) -> None:
    idx = _index(tmp_path)
    v = [0.1, 0.2, 1 / 3]               # 不是 float32 能精确表示的数
    cold = idx.put_embedding("m", "t", v)
    assert cold == idx.get_embeddings("m", ["t"])["t"]
    idx.close()


def test_embed_cache_is_not_a_fact(tmp_path: Path) -> None:
    idx = _index(tmp_path)
    before = idx.fact_count()
    idx.put_embedding("m", "t", [1.0])
    assert idx.fact_count() == before and idx.all_facts() == []
    idx.close()


def test_embed_cache_is_harvested(tmp_path: Path) -> None:
    from bladex_proxy.storage import index_staging

    assert b"embedcache/" in index_staging.HARVEST_PREFIXES
