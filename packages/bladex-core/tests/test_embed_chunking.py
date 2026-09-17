"""候选嵌入分块（MQ-A28/A29，2026-08-31 两次全量重建失败驱动）。

**行为等价是第一红线**：分块只改"一次送多少"，不改文本、不改顺序、不改向量。
一旦顺序错位，fact 与 embedding 就会**静默错配**——每条 fact 挂着别人的向量，
判重/裁决/检索全部失真，而且没有任何报错。所以这里先钉等价，再钉分块。
"""

from __future__ import annotations

import pytest
from bladex_core.consolidation_proxy import ProxyConsolidator


class _Embedder:
    """记录每次调用收到的批大小；向量与文本一一对应且可反查。"""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(len(texts))
        # 向量内容由文本决定 ⇒ 顺序错位一定能被断言抓到
        return [[float(len(t)), float(hash(t) % 1000)] for t in texts]


def _consolidator(embedder) -> ProxyConsolidator:
    """只装配本测试用到的那一个协作者，其余留空（不碰存储、不碰 LLM）。"""
    c = ProxyConsolidator.__new__(ProxyConsolidator)
    c._embedder = embedder                                   # noqa: SLF001
    return c


@pytest.fixture
def texts() -> list[str]:
    return [f"text-{i}" for i in range(1050)]


def test_result_is_identical_to_one_shot(texts, monkeypatch) -> None:
    """🔴 分块前后**逐字相同**——这是本改动唯一不能破的东西。"""
    monkeypatch.setenv("BLADEX_EMBED_BATCH", "512")
    chunked = _consolidator(_Embedder())._embed_chunked(texts, phase="t")
    one_shot = _Embedder().embed(texts)
    assert chunked == one_shot
    assert len(chunked) == len(texts)


def test_it_actually_chunks(texts, monkeypatch) -> None:
    """判别力：不分块的话下面这条会读到 [1050]。"""
    monkeypatch.setenv("BLADEX_EMBED_BATCH", "512")
    emb = _Embedder()
    _consolidator(emb)._embed_chunked(texts, phase="t")
    assert emb.calls == [512, 512, 26]


def test_chunk_size_is_configurable(texts, monkeypatch) -> None:
    """旋钮必须真的通到调用上——"有 flag 但没接线"是本仓的高发形态。"""
    monkeypatch.setenv("BLADEX_EMBED_BATCH", "100")
    emb = _Embedder()
    _consolidator(emb)._embed_chunked(texts, phase="t")
    assert emb.calls == [100] * 10 + [50]


@pytest.mark.parametrize("raw", ["0", "-5", "abc"])
def test_bad_chunk_size_never_yields_a_zero_step(texts, monkeypatch, raw) -> None:
    """0/负数/垃圾值都不得让 `range(0, n, 0)` 抛或死循环——至少退到 1。"""
    monkeypatch.setenv("BLADEX_EMBED_BATCH", raw)
    emb = _Embedder()
    out = _consolidator(emb)._embed_chunked(texts[:5], phase="t")
    assert len(out) == 5
    assert all(n >= 1 for n in emb.calls)


def test_empty_input_makes_no_call(monkeypatch) -> None:
    """空输入不该产生一次空调用（IPC 侧那是一次真实往返）。"""
    emb = _Embedder()
    assert _consolidator(emb)._embed_chunked([], phase="t") == []
    assert emb.calls == []


def test_single_chunk_when_input_fits(monkeypatch) -> None:
    monkeypatch.setenv("BLADEX_EMBED_BATCH", "512")
    emb = _Embedder()
    _consolidator(emb)._embed_chunked([f"t{i}" for i in range(10)], phase="t")
    assert emb.calls == [10]
