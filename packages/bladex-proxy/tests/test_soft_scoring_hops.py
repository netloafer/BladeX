"""U6（ADR-0026 §6.3/§6.4）+ ADR-0028 E6.3：融合重排 + 有限跳。

真实路径：MemoryIndex（LanceDB + meta + FTS5）真写真查，不 mock search 内部。

**ADR-0028 E6.3 起，融合从"五信号分值加权"换成 RRF（秩融合）**，
原因是分值不可比：库内 cosine 噪声底 0.88–0.93、top-15 极差仅 0.024–0.060，
而 importance 的差分 ≈0.007 —— 没有一路信号能对抗噪声。
故本文件里三条原"软打分分值"剧本改写为融合剧本，
`_soft_components` 这个分量字段随实现一并退役。

覆盖：词法通道救 dense 死角 / RRF 排序 / SUPERSEDES 反向替换 / 同 Matter 兄弟 /
PART_OF 父脉络 / 可见性边界不破 / 开关关闭零回归。
"""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest
from bladex_core.fact import Fact
from bladex_core.matter import (
    EdgeRelation,
    EdgeTargetType,
    Matter,
    MatterEdge,
    MatterStatus,
)
from bladex_proxy.storage.memory_index import MemoryIndex


class MockEmbedder:
    """确定性 hash embedder（同 test_index_rebuild_filters）。"""

    def embed(self, texts):
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


def _fact(fid: str, content: str, *, entities: list[str] | None = None,
          importance: float = 1.0, user: str = "u1",
          matter_id: str = "", t_invalid=None, superseded_by: str = "",
          embedding: list[float] | None = None) -> Fact:
    f = Fact(id=fid, content=content, source_user_id=user,
             entities=entities or [], importance=importance,
             matter_id=matter_id, superseded_by=superseded_by,
             created_at=datetime(2026, 8, 1, tzinfo=UTC))
    f.t_invalid = t_invalid
    f.embedding = embedding
    return f


def _vec(slot: int) -> list[float]:
    v = [0.0] * 64
    v[slot] = 1.0
    return v


def _open_index(tmpdir: str) -> MemoryIndex:
    index = MemoryIndex(Path(tmpdir) / "index", embedder=MockEmbedder(), read_only=False)
    index.open()
    return index


# ── 软打分 ──


def test_lexical_channel_rescues_dense_blind_spot():
    """E6.2：词法通道救 dense 死角——专有名词/ID/报错串。

    这条 fact 与 query 的向量正交（MockEmbedder 按内容 hash 打槽位，
    "BladeX 502" 与它落在不同槽），纯 dense 永远召不回；
    FTS5 倒排按 `BladeX` / `502` 命中，RRF 把它融进来。
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        index = _open_index(tmpdir)
        index.add_fact(_fact("f_hit", "BladeX proxy 502 的根因是上游配额",
                          entities=["BladeX", "502"], embedding=_vec(1)))
        index.add_fact(_fact("f_miss", "用户下个月计划去日本",
                          entities=["日本"], embedding=_vec(2)))
        got = index.search("帮我看 BladeX 502 问题", top_k=5, user_id="u1")
        assert [f.id for f in got][0] == "f_hit"
        index.close()


def test_identifier_phrase_match_not_confused_by_near_duplicate():
    """E6.2 的存在理由（实测）：`bxe3663a38` 与 `bx3a984bcd` 的 cosine = 0.9890。

    dense 分不开这两个 token（排名基本随机），词法按 phrase 精确匹配能分开。
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        index = _open_index(tmpdir)
        index.add_fact(_fact("f_a", "端到端探针代码是 bxe3663a38",
                          entities=["bxe3663a38"], embedding=_vec(1)))
        index.add_fact(_fact("f_b", "端到端探针代码是 bx3a984bcd",
                          entities=["bx3a984bcd"], embedding=_vec(1)))
        got = index.search("bxe3663a38 是什么", top_k=5, user_id="u1",
                        identifiers=["bxe3663a38"])
        assert [f.id for f in got][0] == "f_a"
        index.close()


def test_fusion_uses_rank_not_absolute_score():
    """RRF 的分数是**秩**的函数，不是 cosine 的量纲。

    钉这条是为了防止有人把 `min_relevance=0.3` 这类绝对阈值再加回来——
    融合后 `_score` 已归一到 [0, ~1.25]，与 cosine 不可比。
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        index = _open_index(tmpdir)
        index.add_fact(_fact("f1", "第一条内容甲", embedding=_vec(1)))
        index.add_fact(_fact("f2", "第二条内容乙", embedding=_vec(2)))
        got = index.search("查询内容", top_k=5, user_id="u1")
        assert got
        for f in got:
            assert 0.0 <= f._score <= 1.5
        # 软打分五信号的分量字段已随实现退役
        assert not hasattr(got[0], "_soft_components")
        index.close()


def test_fusion_explicitly_off_falls_back_to_pure_dense(monkeypatch: pytest.MonkeyPatch):
    """回滚通道：`BLADEX_SOFT_SCORING=0` → 纯 dense（不进融合），G10 回退保证。"""
    monkeypatch.setenv("BLADEX_SOFT_SCORING", "0")
    with tempfile.TemporaryDirectory() as tmpdir:
        index = _open_index(tmpdir)
        index.add_fact(_fact("f1", "普通条目内容", embedding=_vec(1)))
        got = index.search("查询", top_k=5, user_id="u1")
        assert got and not hasattr(got[0], "_rrf")
        index.close()


# ── 有限跳 ──


def test_hop_supersede_reverse_swap(monkeypatch: pytest.MonkeyPatch):
    """dense 命中被取代旧条 → 补进取代它的新条（新条无向量，只有跳能带回）。"""
    monkeypatch.setenv("BLADEX_HOP_EXPAND", "1")
    with tempfile.TemporaryDirectory() as tmpdir:
        index = _open_index(tmpdir)
        index.add_fact(_fact("f_old", "项目截止日期是 8 月 1 日",
                          t_invalid=datetime(2026, 8, 1, tzinfo=UTC),
                          superseded_by="f_new", embedding=_vec(1)))
        index.add_fact(_fact("f_new", "项目截止日期改到 8 月 15 日"))  # 无向量
        got = index.search("项目截止日期", top_k=5, user_id="u1",
                        identifiers=["8月15日"])
        ids = [f.id for f in got]
        assert "f_old" not in ids  # current-only
        assert "f_new" in ids      # 反向替换带回
        # ADR-0028 E6.2 之后，f_new 也可能由**词法通道**先召回（它是有效条目、
        # 只是没有向量——正是"dense 死角"）。此时 hop 无需重复带回，
        # 故这里只断言"新条到场"，不再要求它必须带 hop 标记。
        swapped = next(f for f in got if f.id == "f_new")
        assert getattr(swapped, "_hop", "supersede") in ("supersede", "sibling")
        index.close()


def test_hop_sibling_same_matter(monkeypatch: pytest.MonkeyPatch):
    """种子 fact 的 Matter 域内兄弟被带回（BELONGS 边 + matter_id 字段）。"""
    monkeypatch.setenv("BLADEX_HOP_EXPAND", "1")
    with tempfile.TemporaryDirectory() as tmpdir:
        index = _open_index(tmpdir)
        index.add_matter(Matter(matter_id="m1", title="T8b", status=MatterStatus.ACTIVE))
        index.add_fact(_fact("f_seed", "T8b ANN 索引建好了", matter_id="m1", embedding=_vec(1)))
        index.add_fact(_fact("f_sib", "T8b recall 阈值定为均值 0.95"))  # 无向量，只有边
        index.add_edge(MatterEdge(matter_id="m1", target_type=EdgeTargetType.FACT,
                               target_key="f_seed"))
        index.add_edge(MatterEdge(matter_id="m1", target_type=EdgeTargetType.FACT,
                               target_key="f_sib"))
        got = index.search("ANN 索引", top_k=6, user_id="u1")
        ids = [f.id for f in got]
        assert "f_sib" in ids
        assert next(f for f in got if f.id == "f_sib")._hop == "sibling"
        index.close()


def test_hop_part_of_parent_matter(monkeypatch: pytest.MonkeyPatch):
    """PART_OF 父 Matter 域内 fact 被带回（脉络补全）。"""
    monkeypatch.setenv("BLADEX_HOP_EXPAND", "1")
    with tempfile.TemporaryDirectory() as tmpdir:
        index = _open_index(tmpdir)
        index.add_matter(Matter(matter_id="m_child", title="T8b", status=MatterStatus.ACTIVE))
        index.add_matter(Matter(matter_id="m_parent", title="ADR-0024", status=MatterStatus.ACTIVE))
        index.add_fact(_fact("f_seed", "T8b ANN 索引建好了", matter_id="m_child",
                          embedding=_vec(1)))
        index.add_fact(_fact("f_parent", "ADR-0024 的目标是任务单元一等实体化"))
        index.add_edge(MatterEdge(matter_id="m_child", target_type=EdgeTargetType.MATTER,
                               target_key="m_parent", relation=EdgeRelation.PART_OF))
        index.add_edge(MatterEdge(matter_id="m_parent", target_type=EdgeTargetType.FACT,
                               target_key="f_parent"))
        got = index.search("ANN 索引", top_k=6, user_id="u1")
        assert "f_parent" in [f.id for f in got]
        assert next(f for f in got if f.id == "f_parent")._hop == "part_of"
        index.close()


def test_hop_respects_visibility_boundary(monkeypatch: pytest.MonkeyPatch):
    """跳入的 fact 必须过可见性边界（别人的 fact 不因兄弟关系泄漏）。"""
    monkeypatch.setenv("BLADEX_HOP_EXPAND", "1")
    with tempfile.TemporaryDirectory() as tmpdir:
        index = _open_index(tmpdir)
        index.add_matter(Matter(matter_id="m1", title="共享 Matter", status=MatterStatus.ACTIVE))
        index.add_fact(_fact("f_seed", "我的种子条目", matter_id="m1", embedding=_vec(1)))
        index.add_fact(_fact("f_other", "别人的条目", user="u2"))
        index.add_edge(MatterEdge(matter_id="m1", target_type=EdgeTargetType.FACT,
                               target_key="f_other"))
        got = index.search("种子", top_k=6, user_id="u1")
        assert "f_other" not in [f.id for f in got]
        index.close()


def test_hop_explicitly_off(monkeypatch: pytest.MonkeyPatch):
    """显式关闭：被取代旧条只是消失，新条不自动带回（旧行为逐字一致）。

    ADR-0027 §5.4 起该机制**默认开**（此前默认关，导致验收过了但发出去的仍是旧行为），
    所以这条回退保证要显式设开关为 0 —— 回滚通道本身没变，变的是默认值。
    """
    monkeypatch.setenv("BLADEX_HOP_EXPAND", "0")
    with tempfile.TemporaryDirectory() as tmpdir:
        index = _open_index(tmpdir)
        index.add_fact(_fact("f_old", "项目截止日期是 8 月 1 日",
                          t_invalid=datetime(2026, 8, 1, tzinfo=UTC),
                          superseded_by="f_new", embedding=_vec(1)))
        index.add_fact(_fact("f_new", "项目截止日期改到 8 月 15 日"))
        got = index.search("项目截止日期", top_k=5, user_id="u1")
        # 关掉 hop 之后没有任何条目带 hop 标记（"被取代旧条自动带回新条"这条路径关闭）。
        # 注意 f_new 仍可能经**词法通道**被召回——那是 E6.2 的正常能力，与 hop 无关。
        assert all(not getattr(f, "_hop", "") for f in got)
        index.close()
