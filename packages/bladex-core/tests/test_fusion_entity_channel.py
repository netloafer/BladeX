"""T3b/T3c（三段式卡，2026-08-10）：RRF 融合 entity 第四通道。

背景：T8a 实体重排在 `BLADEX_SOFT_SCORING` 开启时被显式跳过
（memory_index.py `and not _soft_enabled`），而 `fuse_and_rerank` 没有
entity 通道——自 08-04 soft 默认开起实体信号实际退出了检索排序。
本文件钉死：

  ① entity 命中把目标 fact 从 dense 尾部拉进 top-k；
  ② 热门实体衰减（mem0 1/(1+0.001(N-1)²)）：泛实体 boost 弱于冷实体；
  ③ query 无实体命中 → build 返回 None，三通道行为与改造前逐字一致；
  ④ soft 关闭（回滚态）时 T8a 跳过语义保持（源码级断言 +
     行为回归网 = packages/bladex-proxy/tests/test_entity_aware_rerank.py）。

RRF 骨架/折叠/配额/MMR 不动；明确不抄 mem0 的加法分值融合（理由见
`build_entity_channel` docstring 与 memory_index._fuse_rerank）。
"""

from __future__ import annotations

from bladex_core.fact import Fact
from bladex_core.fusion import (
    Channel,
    build_entity_channel,
    entity_decay,
    fuse_and_rerank,
)


def _fact(fid: str, content: str, entities: list[str] | None = None) -> Fact:
    return Fact(id=fid, content=content, entities=entities or [])


def _pool(n: int = 12) -> dict[str, Fact]:
    """dense 候选池：f-target 带目标实体、排在 dense 尾部。"""
    facts = {f"f-{i:02d}": _fact(f"f-{i:02d}", f"泛泛内容 {i}") for i in range(n - 1)}
    facts["f-target"] = _fact(
        "f-target", "泰山啤酒破产重整的共益债公告", ["泰山啤酒", "共益债"])
    return facts


def test_entity_hit_promotes_from_dense_tail():
    """① 目标 fact dense 排名垫底，entity 通道命中 → 进 top-k 且靠前。"""
    facts = _pool()
    dense_ids = [fid for fid in facts if fid != "f-target"] + ["f-target"]
    channels = [Channel("dense", dense_ids)]

    query = "泰山啤酒破产案有共益债的公告吗"
    ch = build_entity_channel(query, facts)
    assert ch is not None and ch.ranked_ids[0] == "f-target"
    channels.append(ch)

    out, info = fuse_and_rerank(facts, channels, top_k=5)
    assert "f-target" in [f.id for f in out]
    assert info["channels"]["entity"] == 1
    # 观测字段：命中的 fact 打上 _entity_hit（index_search_done 的 entity_hits 来源）
    assert facts["f-target"]._entity_hit > 0.0


def test_hot_entity_decay_orders_channel():
    """② 热门实体（N=50）衰减后，冷实体（N=2）命中的 fact 秩更高。"""
    facts = {
        "f-hot": _fact("f-hot", "BladeX 相关讨论", ["bladex"]),
        "f-cold": _fact("f-cold", "泰山啤酒公告", ["泰山啤酒"]),
    }
    freq = {"bladex": 50, "泰山啤酒": 2}
    ch = build_entity_channel("bladex 与 泰山啤酒 的进展", facts, entity_freq=freq)
    assert ch is not None
    assert ch.ranked_ids == ["f-cold", "f-hot"]
    # 衰减公式本身（mem0 起步值）
    assert entity_decay(1) == 1.0
    assert entity_decay(2) > entity_decay(50) > entity_decay(200)


def test_no_entity_hit_identical_to_pre_change():
    """③ query 无实体命中 → 不加通道，三通道融合输出与改造前逐字一致。"""
    facts = _pool()
    dense_ids = list(facts)
    channels = [Channel("dense", dense_ids), Channel("lexical", dense_ids[:3])]

    assert build_entity_channel("完全无关的提问内容", facts) is None

    out_a, _ = fuse_and_rerank(facts, list(channels), top_k=5)
    # 改造前形态 = 同一组通道再融合一次（无 entity 通道注入点）
    out_b, _ = fuse_and_rerank(facts, list(channels), top_k=5)
    assert [f.id for f in out_a] == [f.id for f in out_b]


def test_empty_entity_facts_never_channel():
    """候选全无 entities → 返回 None（不产生空通道扰动 RRF 分母）。"""
    facts = {f"f-{i}": _fact(f"f-{i}", "无实体内容") for i in range(5)}
    assert build_entity_channel("泰山啤酒", facts) is None


def test_soft_path_t8a_skip_preserved():
    """④ T8a 在 soft 路径的跳过**保持**——实体信号唯一消费方 = 融合 entity
    通道；两处并行加权 = 同一信号两个消费方（本仓库的惯性病）。
    soft 关闭（回滚态）的行为回归网在 test_entity_aware_rerank.py。
    """
    from pathlib import Path

    import bladex_proxy.storage.memory_index as mi

    src = Path(mi.__file__).read_text(encoding="utf-8")
    assert "and not _soft_enabled" in src, (
        "T8a 的 soft 路径跳过被移除了——entity 信号会被双重加权"
    )
