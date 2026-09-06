"""ADR-0028 E6.2/E6.3/E6.4：FTS5 词法路 + 融合接线 + 分层配额（proxy 侧）。

真实路径：MemoryIndex（LanceDB + meta + FTS5）真写真查，不 mock search 内部。

E6.2 的存在理由是一条实测：`bxe3663a38` 与 `bx3a984bcd` 的 cosine = **0.9890**
——dense 通道对标识符是瞎的，排名基本随机。trigram 倒排里它们是两个不同的 key。
"""

from __future__ import annotations

import pytest
from bladex_core.fact import Fact, ItemKind
from bladex_proxy.storage.fts_index import FtsIndex, _expand_cjk, _is_cjk
from bladex_proxy.storage.memory_index import MemoryIndex


class _Embedder:
    """确定性 hash embedder（同 test_soft_scoring_hops）。"""

    def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * 64
            v[hash(t) % 64] = 1.0
            out.append(v)
        return out

    @property
    def available(self):
        return True


def _vec(slot: int) -> list[float]:
    v = [0.0] * 64
    v[slot] = 1.0
    return v


def _fact(fid: str, content: str, *, entities: list[str] | None = None,
          kind: ItemKind = ItemKind.ASSERTION, user: str = "u1",
          session: str = "s1", embedding: list[float] | None = None,
          importance: float = 0.7) -> Fact:
    f = Fact(id=fid, content=content, source_user_id=user, source_session=session,
             entities=entities or [], item_kind=kind, importance=importance)
    f.embedding = embedding
    return f


@pytest.fixture()
def index(tmp_path):
    store = MemoryIndex(tmp_path / "index", embedder=_Embedder(), read_only=False)
    store.open()
    yield store
    store.close()


# ── E6.2 FTS5 词法索引 ─────────────────────────────────────────────────


def test_fts_index_roundtrip(tmp_path):
    idx = FtsIndex(tmp_path)
    assert idx.open()
    idx.upsert(_fact("f1", "Memory Index 的 ref_count 恒为 0，根因是 proxy 只读",
                     entities=["ref_count"]))
    assert idx.count() == 1
    hits = idx.search("ref_count")
    assert [h[0] for h in hits] == ["f1"]
    idx.delete("f1")
    assert idx.search("ref_count") == []
    idx.close()


def test_fts_index_excludes_file_ref(tmp_path):
    """file_ref 不入词法索引——与 E1.3 向量平面同边界（裸路径不参与检索）。"""
    idx = FtsIndex(tmp_path)
    idx.open()
    idx.upsert(_fact("f_ref", "文件 /Users/alice/x.md", kind=ItemKind.FILE_REF))
    assert idx.count() == 0
    idx.close()


def test_fts_index_drops_invalidated_fact(tmp_path):
    """被取代/失效的条目退出词法召回（与 dense 的 current-only 同口径）。"""
    from datetime import UTC, datetime

    idx = FtsIndex(tmp_path)
    idx.open()
    f = _fact("f1", "项目截止日期是 8 月 1 日")
    idx.upsert(f)
    assert idx.count() == 1
    f.t_invalid = datetime.now(UTC)
    idx.upsert(f)
    assert idx.count() == 0
    idx.close()


def test_fts_read_only_on_empty_dir_is_disabled(tmp_path):
    """读端遇到空库不建文件（写权只有 consolidator），静默禁用。"""
    idx = FtsIndex(tmp_path / "nothing-here", read_only=True)
    assert idx.open() is False
    assert idx.available is False
    assert idx.search("anything") == []


def test_cjk_expansion_for_long_terms():
    """中文没有空格：整句 phrase 查永远匹配不上——按三元组查才行。"""
    assert _is_cjk("泰山啤酒破产案")
    grams = _expand_cjk("泰山啤酒破产案")
    assert "泰山啤" in grams and "破产案" in grams
    # 短词保持整体
    assert _expand_cjk("泰山啤酒") == ["泰山啤酒"]


def test_lexical_finds_cjk_substring_match(tmp_path):
    idx = FtsIndex(tmp_path)
    idx.open()
    idx.upsert(_fact("a", "关于泰山啤酒破产案的公告", entities=["泰山啤酒"]))
    idx.upsert(_fact("b", "关于贵州茅台的记录", entities=["贵州茅台"]))
    hits = [h[0] for h in idx.search("泰山啤酒破产案有共益债吗")]
    assert hits and hits[0] == "a"
    idx.close()


# ── E6.3 融合接线（Memory Index.search）───────────────────────────────────────────


def test_identifier_phrase_beats_near_duplicate_token(index):
    """实测 cosine 0.9890 分不开的两个 token，词法能分开。"""
    index.add_fact(_fact("f_new", "端到端探针代码是 bxe3663a38",
                      entities=["bxe3663a38"], embedding=_vec(1)))
    index.add_fact(_fact("f_old", "端到端探针代码是 bx3a984bcd",
                      entities=["bx3a984bcd"], embedding=_vec(1)))
    got = index.search("探针代码 bxe3663a38 是多少", top_k=5, user_id="u1",
                    identifiers=["bxe3663a38"])
    assert [f.id for f in got][0] == "f_new"


def test_lexical_channel_reported_in_search(index, capsys):
    index.add_fact(_fact("f1", "BladeX proxy 502 的根因是上游配额",
                      entities=["BladeX"], embedding=_vec(3)))
    index.search("BladeX 502", top_k=5, user_id="u1")
    out = capsys.readouterr().out
    assert "index_search_fused" in out
    assert "lexical" in out


def test_file_ref_never_enters_retrieval_plane(index):
    """E1.3 + E6.2 合力：file_ref 既不进向量表，也不进词法索引。"""
    index.add_fact(_fact("f_ref", "文件 /Users/alice/dev/BladeX/README.md",
                      kind=ItemKind.FILE_REF, embedding=_vec(5)))
    index.add_fact(_fact("f_ok", "README 里写了安装步骤", embedding=_vec(6)))
    got = index.search("README 安装", top_k=10, user_id="u1")
    assert "f_ref" not in [f.id for f in got]


def test_fusion_scores_are_normalized_not_cosine(index):
    """融合后 `_score` 是 RRF 的量纲，不再是 cosine —— 绝对阈值就此退役。"""
    for i in range(3):
        index.add_fact(_fact(f"f{i}", f"条目内容第 {i} 条", embedding=_vec(i + 1)))
    got = index.search("条目内容", top_k=3, user_id="u1")
    assert got
    assert all(hasattr(f, "_rrf") for f in got)
    assert all(0.0 <= f._score <= 1.5 for f in got)


# ── E6.4 注入收敛与分层配额 ────────────────────────────────────────────


def test_inject_top_k_env_override(monkeypatch):
    from bladex_proxy.server import _inject_top_k

    monkeypatch.delenv("BLADEX_INJECT_TOPK", raising=False)
    assert _inject_top_k() == 15
    monkeypatch.setenv("BLADEX_INJECT_TOPK", "5")
    assert _inject_top_k() == 5
    monkeypatch.setenv("BLADEX_INJECT_TOPK", "bogus")
    assert _inject_top_k() == 15


