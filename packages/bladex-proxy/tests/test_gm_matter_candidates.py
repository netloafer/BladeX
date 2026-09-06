"""GM-2 按主题键反查候选 Matter + 判别力门槛（MQ-S11）。

归属候选此前只有 `search_matters(fact.embedding)` 一条纯向量通路。它在话题稳定、
语料充足时够用（实测同一张卡跨 6 session / 3 agent 持续吸附成员），本项补的是
**向量散开的那一段**——同一对象不同动作（进度评估／进展评估／问题报告／风险复核）
各轮内容差异大时向量捡不回来。

🔴 本文件的重心是**两道门**，不是"能不能召回"：
  ① 判别力门——阴性对照实测泛词 `bladex` 一个键命中 **57 张卡**，不设门
     = 把 MS-16「单泛词吸尘器」从合并侧原样搬到候选侧；
  ② 单键永不放行——与 `topic_keys` 模块既定纪律同源，也是 T4a 那次误合并的修法。

覆盖率 ≠ 精度（GM 纪律 3）：本文件只钉机制边界，"合对没合对"由
`probe_attribution_quality.py` + 人工签收判。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from bladex_core.attribution import UNASSIGNED_MATTER_ID
from bladex_core.matter import Matter, MatterStatus
from bladex_proxy.storage.memory_index import MemoryIndex


def _matter(mid: str, title: str, keys: dict[str, int],
            status: MatterStatus = MatterStatus.ACTIVE) -> Matter:
    return Matter(matter_id=mid, title=title, status=status,
                  aliases=[title], topic_keys=keys)


@pytest.fixture
def index():
    with tempfile.TemporaryDirectory() as td:
        idx = MemoryIndex(Path(td) / "index", read_only=False)
        idx.open()
        try:
            yield idx
        finally:
            idx.close()


def _seed_vacuum_corpus(idx: MemoryIndex, n_generic: int = 40) -> None:
    """泛词 `bladex` 遍布全库，另有两张真正共享判别性键的姐妹卡。

    形态照抄实测：`bladex` 42 卡 / `skill.md` 19 / `ollama` 14，
    而 903 个键里 732 个只出现在 1 张卡上。
    """
    for i in range(n_generic):
        idx.add_matter(_matter(f"m-generic-{i:02d}", f"泛词卡 {i}",
                               {"bladex": 3, f"独有键{i:02d}": 2}))
    idx.add_matter(_matter("m-sister-a", "任务卡进度评估",
                           {"bladex": 3, "memory-quality-tasks-20260815": 4,
                            "matter碎片化": 2}))
    idx.add_matter(_matter("m-sister-b", "任务卡进展评估与问题报告",
                           {"bladex": 2, "memory-quality-tasks-20260815": 3,
                            "matter碎片化": 3}))


# ── ① 判别力门 ───────────────────────────────────────────────────────────────


def test_generic_key_alone_recalls_nothing(index: MemoryIndex) -> None:
    """🔴 阴性对照：只带泛词的 fact 一张卡都召不回。

    这是本通路存在与否的分水岭——不设门的话这一条会拉回 42 个候选。
    """
    _seed_vacuum_corpus(index)
    inverted, total = index.build_matter_key_index()
    hits = index.key_based_matter_candidates(["bladex"], inverted, total)
    assert hits == []


def test_generic_key_dropped_even_when_paired(index: MemoryIndex) -> None:
    """泛词不计入"共享键数"：`bladex` + 一个判别键 = 1 个有效键 → 仍不放行。"""
    _seed_vacuum_corpus(index)
    inverted, total = index.build_matter_key_index()
    hits = index.key_based_matter_candidates(
        ["bladex", "memory-quality-tasks-20260815"], inverted, total)
    assert hits == [], "泛词凑数不得把单键命中抬成合法候选"


def test_discriminative_pair_recalls_sister_card(index: MemoryIndex) -> None:
    """两个判别性键共享 → 姐妹卡进候选（本通路要抓的那个形态）。"""
    _seed_vacuum_corpus(index)
    inverted, total = index.build_matter_key_index()
    hits = index.key_based_matter_candidates(
        ["memory-quality-tasks-20260815", "matter碎片化"], inverted, total)
    ids = {mid for mid, _n, _k in hits}
    assert ids == {"m-sister-a", "m-sister-b"}
    assert all(n >= 2 for _m, n, _k in hits)


def test_threshold_scales_with_library_size(index: MemoryIndex) -> None:
    """🔴 门槛的分母是库规模，不是绝对数。

    同一个"出现在 5 张卡上"在小库里是泛词、在大库里是强标识符——绝对阈值当门槛
    正是 MQ-S1 清查的那个设计模式。这里用同一个 `inverted`、只换 `total_matters`
    来演示门确实随分母动。
    """
    _seed_vacuum_corpus(index)
    inverted, _real_total = index.build_matter_key_index()
    # 假装库有 2000 张卡 → 门槛 = 100，`bladex`(42 卡) 变成合法判别键
    hits_big = index.key_based_matter_candidates(
        ["bladex", "memory-quality-tasks-20260815"], inverted, 2000)
    assert {mid for mid, _n, _k in hits_big} == {"m-sister-a", "m-sister-b"}


def test_tiny_library_keeps_floor_of_two(index: MemoryIndex) -> None:
    """小库下限 2：`ratio × 3` 会算出 0，把"恰好两张卡共享"这个唯一目标形态也挡掉。"""
    index.add_matter(_matter("m-a", "甲", {"判别键一": 2, "判别键二": 2}))
    index.add_matter(_matter("m-b", "乙", {"判别键一": 1, "判别键二": 1}))
    index.add_matter(_matter("m-c", "丙", {"无关键": 1}))
    inverted, total = index.build_matter_key_index()
    assert total == 3
    hits = index.key_based_matter_candidates(["判别键一", "判别键二"], inverted, total)
    assert {mid for mid, _n, _k in hits} == {"m-a", "m-b"}


# ── ② 单键永不放行 ───────────────────────────────────────────────────────────


def test_single_shared_key_never_passes(index: MemoryIndex) -> None:
    """即便是全库独一无二的键，单键命中也不放行（T4a 修法，纪律同 topic_keys）。"""
    index.add_matter(_matter("m-only", "唯一卡", {"极其独特的标识符abc": 5}))
    for i in range(9):
        index.add_matter(_matter(f"m-f{i}", f"填充 {i}", {f"填充键{i}": 1}))
    inverted, total = index.build_matter_key_index()
    assert index.key_based_matter_candidates(
        ["极其独特的标识符abc"], inverted, total) == []


# ── 候选池卫生 ───────────────────────────────────────────────────────────────


def test_closed_and_unassigned_are_excluded(index: MemoryIndex) -> None:
    """已关闭的卡与未归属池不进倒排表（与 `matter_merge_candidates` 同一口径）。"""
    index.add_matter(_matter("m-closed", "已关闭", {"判别键一": 2, "判别键二": 2},
                             status=MatterStatus.CLOSED))
    index.add_matter(_matter(UNASSIGNED_MATTER_ID, "未归属池",
                             {"判别键一": 2, "判别键二": 2}))
    index.add_matter(_matter("m-open", "在办", {"判别键一": 2, "判别键二": 2}))
    inverted, total = index.build_matter_key_index()
    assert total == 1
    assert {mid for mid, _n, _k in index.key_based_matter_candidates(
        ["判别键一", "判别键二"], inverted, total)} == {"m-open"}


def test_legacy_matter_without_topic_keys_still_indexed(index: MemoryIndex) -> None:
    """存量卡 `topic_keys` 空 → 走 `effective_topic_keys` 派生兜底。

    两处口径必须一致，否则离线 merge-candidates 检出的重复卡在线检不出来
    （2026-08-14 上线即踩过：泰安四开对自身检不出来）。
    """
    index.add_matter(Matter(matter_id="m-legacy", title="泰安仁信 增资扩股",
                            status=MatterStatus.ACTIVE,
                            aliases=["泰安仁信 增资扩股"],
                            entities=["泰安仁信", "增资扩股"]))
    inverted, _total = index.build_matter_key_index()
    assert any("m-legacy" in owners for owners in inverted.values())


def test_max_candidates_is_respected(index: MemoryIndex) -> None:
    """候选数有上限——确定性通路不得把候选池撑爆（top_k 预算是共享的）。

    `total_matters=200` 是显式传的：8 张卡共享同一个键，在 58 张卡的库里
    (门槛 int(58×0.05)=2) 那个键本身就会被判别力门滤掉，测不到 `max_candidates`。
    传大分母 = 把判别力门让开，只留上限这一个变量。
    """
    for i in range(8):
        index.add_matter(_matter(f"m-s{i}", f"姐妹 {i}",
                                 {"判别键一": 2, "判别键二": 2, f"独有{i}": 1}))
    inverted, _total = index.build_matter_key_index()
    hits = index.key_based_matter_candidates(
        ["判别键一", "判别键二"], inverted, 200, max_candidates=3)
    assert len(hits) == 3


def test_empty_fact_keys_recall_nothing(index: MemoryIndex) -> None:
    _seed_vacuum_corpus(index)
    inverted, total = index.build_matter_key_index()
    assert index.key_based_matter_candidates([], inverted, total) == []
    assert index.key_based_matter_candidates(["", "  "], inverted, total) == []
