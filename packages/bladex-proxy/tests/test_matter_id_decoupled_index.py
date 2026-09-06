"""G12.1 存储侧：消化路的 id 解耦 + **同名汇入必须原样保留**（ADR-0031 §3.4）。

纯逻辑那半边在 `packages/bladex-core/tests/test_matter_id_decoupled.py`。
这里测的是改动的另一半，也是更容易出事的一半：

    旧写法  matter_id = _deterministic_matter_id(title)

这一行同时干了两件事——①生成 id，②顺带实现"同标题汇入同一张卡"
（同标题算出同 id ⇒ `get_matter(...) is not None` ⇒ 接上去）。
**解耦 ① 的时候极容易把 ② 一起删掉**，而 ② 的方向正是碎片化的反方向，
删掉它的代价比解耦本身还大。所以这里把 ② 单独钉住。

需要 rocksdict/lancedb，只能在用户机跑。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from bladex_core.attribution import (
    UNASSIGNED_MATTER_ID,
    _deterministic_matter_id,
    new_matter_id,
)
from bladex_core.fact import Fact
from bladex_core.matter import EdgeProvenance, EdgeTargetType, MatterEdge

from bladex_proxy.storage.memory_index import MemoryIndex

_TITLE = "泰山啤酒破产重整"


def _make_index(tmpdir: str) -> MemoryIndex:
    index = MemoryIndex(Path(tmpdir) / "index", embedder=None, read_only=False)
    index.open()
    return index


def _pool(index: MemoryIndex, ids: list[str], *, title: str = _TITLE,
          ledger_prefix: str = "local/hermes:default/s1/") -> None:
    """往未归属池塞几条同 proposal 标题、不同 logical_turn 的 fact。"""
    for i, fid in enumerate(ids):
        vec = [0.0] * 128
        vec[0] = 0.9 + i * 0.01
        index.add_fact(Fact(
            id=fid, content=f"内容 {fid}", embedding=vec, logical_turn=i,
            source_ledger_key=f"{ledger_prefix}{1723000000 + i}-0",
            proposal_titles=[title],
        ))
        index.add_edge(MatterEdge(
            matter_id=UNASSIGNED_MATTER_ID, target_type=EdgeTargetType.FACT,
            target_key=fid, provenance=EdgeProvenance.AUTO,
        ))


def test_digestion_id_comes_from_the_seed_ledger_key() -> None:
    """新卡的 id = 组内**最早那条** fact 的 ledger key 派生，不再是标题哈希。"""
    with tempfile.TemporaryDirectory() as tmp:
        index = _make_index(tmp)
        _pool(index, ["f-1", "f-2", "f-3"])

        assert index.digest_unassigned_pool() == 1
        (matter,) = index.all_matters()

        expected = new_matter_id(ledger_key="local/hermes:default/s1/1723000000-0")
        assert matter.matter_id == expected
        assert matter.matter_id != _deterministic_matter_id(_TITLE), "还在用标题哈希"
        index.close()


def test_digestion_is_replay_stable() -> None:
    """同一批输入重放两次 → 同一个 id（重建等价性）。

    种子取"组内 logical_turn 最小者"而不是"遍历时第一个"，就是为了这条：
    dict/edge 的遍历顺序不该决定一件事的身份。
    """
    ids = []
    for _ in range(2):
        with tempfile.TemporaryDirectory() as tmp:
            index = _make_index(tmp)
            # 有意打乱插入顺序：身份不能跟着插入顺序走
            _pool(index, ["f-3", "f-1", "f-2"])
            index.digest_unassigned_pool()
            ids.append(index.all_matters()[0].matter_id)
            index.close()
    assert ids[0] == ids[1]


def test_same_title_still_folds_into_the_existing_matter() -> None:
    """🔴 本文件的重点：解耦之后，同标题**仍然**汇入同一张卡。

    第二批 fact 的 ledger key 完全不同（另一个 session、另一天），
    按新 id 规则算出来是另一个 id；但它们的 proposal 标题与已有卡逐字相同，
    所以必须挂到那张卡上，而不是开第二张。

    这一条挂了 = 解耦顺手把"同名汇入"删了 = 碎片化变多。
    """
    with tempfile.TemporaryDirectory() as tmp:
        index = _make_index(tmp)
        _pool(index, ["f-1", "f-2"])
        assert index.digest_unassigned_pool() == 1
        first = index.all_matters()[0].matter_id

        _pool(index, ["g-1", "g-2"], ledger_prefix="local/claude-code/s2/")
        index.digest_unassigned_pool()

        assert len(index.all_matters()) == 1, "同标题开出了第二张卡"
        assert index.all_matters()[0].matter_id == first, "同标题落到了新 id 上"
        assert len(index.get_edges(first)) == 4
        index.close()


def test_different_titles_still_get_different_matters() -> None:
    """阴性对照：不同标题不许被折到一起。

    没有这条，"永远返回第一张卡"这种坏实现也能让上一条测试通过——
    而那是最容易达标的坏解（把所有东西合成一张卡）。
    """
    with tempfile.TemporaryDirectory() as tmp:
        index = _make_index(tmp)
        _pool(index, ["f-1", "f-2"], title="泰山啤酒破产重整")
        _pool(index, ["g-1", "g-2"], title="中国啤酒行业产能结构与利用率分析报告",
              ledger_prefix="local/hermes:default/s9/")
        index.digest_unassigned_pool()
        assert len({m.matter_id for m in index.all_matters()}) == 2
        index.close()


def test_legacy_matters_keep_their_ids() -> None:
    """存量卡零变动：库里已有的旧 id 不因为本改动被重算。

    改了 = 全库失联（管理事件重放全部 `source_missing`）——
    2026-08-19 那次重建已经把这个代价演示过一遍了。
    """
    with tempfile.TemporaryDirectory() as tmp:
        index = _make_index(tmp)
        from bladex_core.matter import Matter, MatterStatus

        legacy_id = _deterministic_matter_id(_TITLE)
        index.add_matter(Matter(matter_id=legacy_id, title=_TITLE,
                                status=MatterStatus.ACTIVE, aliases=[_TITLE]))

        _pool(index, ["f-1", "f-2"])
        index.digest_unassigned_pool()

        assert index.get_matter(legacy_id) is not None, "存量卡的 id 被动过"
        assert {m.matter_id for m in index.all_matters()} == {legacy_id}, \
            "存量同名卡在，却仍然开了新卡"
        index.close()
