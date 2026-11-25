"""批 AB · MQ-I27：可增长的单位向量矩阵（两处 cosine 扫描的公共底座）。

全量重建的成本 96% 在 CPU（T2a：novelty 51% + 嵌入 26% + 裁决 20%）。`_is_novel` 每条候选都
重建整个归一化矩阵，`_neighbors_for` / `InRunNeighborSource.neighbors` 每条候选逐个跑纯 Python
`_cosine`——都是「同一算式、常数项里藏了一次全量重建」。这里只把矩阵留住、新行入池时归一化一次；
判定算式不变（守卫：`docs/benchmarks/ab0/candidate_arm.py` 修前 / 修后逐候选比对）。

numpy 是 core 的可选依赖（`semantic` extra）⇒ 没有 numpy 时 `FactVectors` 回落逐条 `_cosine`。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any


def _numpy() -> Any:  # noqa: ANN401
    try:
        import numpy as np
    except ImportError:
        return None
    return np


class UnitMatrix:
    """只增的单位向量矩阵（float32，`_is_novel` 用）。

    `sync(rows)` 把 `rows` 里还没入池的尾部归一化入池。🔴 **一个矩阵只配一个只增的列表**：换列表
    由调用方换矩阵（`consolidate_turns` 每次开头新建）；这里只认长度，变短才重来。批 AB2 · MQ-I44：
    修前按 `id(rows)` 判同一列表，而旧列表释放后地址可能被新列表复用 ⇒ 沿用上一批的行、判重判错。
    归一化沿用 `np.linalg.norm(axis=1)`（与修前同一种算法，逐位相同），零向量按范数 1。
    维度不一致时 `np.stack` / 赋值抛错，由调用方按修前语义处理。
    """

    def __init__(self) -> None:
        self._mat: Any = None
        self._n = 0

    def sync(self, rows: Sequence[Sequence[float]]) -> None:
        import numpy as np

        if self._n > len(rows):
            self._mat, self._n = None, 0
        if self._n == len(rows):
            return
        new = np.stack([np.asarray(e, dtype=np.float32) for e in rows[self._n:]])
        norms = np.linalg.norm(new, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        new = new / norms
        need = self._n + len(new)
        if self._mat is None:
            self._mat = np.empty((max(need, 1024), new.shape[1]), dtype=np.float32)
        elif need > self._mat.shape[0]:
            grown = np.empty((max(need, self._mat.shape[0] * 2), new.shape[1]), dtype=np.float32)
            grown[: self._n] = self._mat[: self._n]
            self._mat = grown
        self._mat[self._n: need] = new
        self._n = need

    def sims(self, q_norm: Any) -> Any:  # noqa: ANN401
        return self._mat[: self._n] @ q_norm


class FactVectors:
    """一组 Fact 的向量视图：按插入顺序返回 `sim >= min_sim` 的条目，语义同逐条 `_cosine`。

    float64（`_cosine` 是 Python 浮点运算）。`_cosine` 的边界：维度不一致 / 零向量 ⇒ 0.0，不入选。
    混入不同维度（只有测试替身会这样）⇒ 整个视图回落逐条 `_cosine`，语义不变。
    """

    def __init__(self, facts: Iterable[Any] = ()) -> None:
        self._np = _numpy()
        self._facts: list[Any] = []
        self._alive: list[bool] = []
        self._mat: Any = None
        self._n = 0
        self._dim = 0
        self._scalar = self._np is None
        for f in facts:
            self.add(f)

    def __len__(self) -> int:
        return sum(self._alive)

    def add(self, fact: Any) -> None:  # noqa: ANN401
        """没有向量的条目查不到，不收（与修前 `if not embedding: continue` 同义）。"""
        emb = getattr(fact, "embedding", None)
        if not emb:
            return
        self._facts.append(fact)
        self._alive.append(True)
        if self._scalar:
            return
        np = self._np
        if self._dim == 0:
            self._dim = len(emb)
        elif len(emb) != self._dim:
            self._scalar = True     # 回落，已入池的矩阵不再用
            self._mat = None
            return
        v = np.asarray(emb, dtype=np.float64)
        norm = float(np.sqrt(v @ v))
        if norm:
            v = v / norm
        if self._mat is None:
            self._mat = np.empty((1024, self._dim), dtype=np.float64)
        elif self._n == self._mat.shape[0]:
            grown = np.empty((self._n * 2, self._dim), dtype=np.float64)
            grown[: self._n] = self._mat
            self._mat = grown
        self._mat[self._n] = v
        self._n += 1

    def discard(self, fact_id: str) -> None:
        for i, f in enumerate(self._facts):
            if self._alive[i] and f.id == fact_id:
                self._alive[i] = False

    def within(self, embedding: Sequence[float], min_sim: float) -> list[tuple[Any, float]]:
        """`[(fact, sim)]`，按插入顺序。"""
        if self._scalar:
            from bladex_core.adjudication import _cosine

            out = []
            for i, f in enumerate(self._facts):
                if self._alive[i]:
                    sim = _cosine(embedding, f.embedding)
                    if sim >= min_sim:
                        out.append((f, sim))
            return out
        if self._n == 0 or len(embedding) != self._dim:
            return []
        np = self._np
        q = np.asarray(embedding, dtype=np.float64)
        norm = float(np.sqrt(q @ q))
        if not norm:
            return []
        sims = self._mat[: self._n] @ (q / norm)
        return [(self._facts[int(i)], float(sims[i])) for i in np.flatnonzero(sims >= min_sim)
                if self._alive[int(i)]]


class IndexedFacts(list):
    """`_adjudicate` 的 `running`：普通 list + 同步维护的 `FactVectors`。只支持 append 增长。"""

    def __init__(self, facts: Iterable[Any] = ()) -> None:
        super().__init__(facts)
        self.vectors = FactVectors(self)

    def append(self, fact: Any) -> None:  # noqa: ANN401
        super().append(fact)
        self.vectors.add(fact)
