"""judgment 台账命名空间隔离（2026-08-11 回归修复，T5 复验抓到）。

事故：`judgment/{fact_id}/{seq}` 自 M2-2（08-08 写入时裁决）起被两个写入方
共用——L4 写 `link:<id>|none|uncertain`，M2 写 `consolidation:add|update|noop`。
`get_latest_judgment` 只取最新 seq、不区分种类 → 几乎每条 fact 在 L4 阶段命中
**自己的 M2 记录** → `_record_to_result` 认不出 verdict → 降级 UNCERTAIN →
留池 + pending、**不送 judge**。

08-10 全量重建实测：907 条落 L4 里 796 条走这条伪 uncertain，未归属池
43%→77%，L4 真实参与仅 111 条。bug 自 08-08 存在，被 L2/L3 吸尘器掩盖
（当时走到 L4 只有 26 条），MS-16 收紧 L2/L3 后全面暴露。

钉死：读侧只认 L4 三种 verdict 形态（白名单——黑名单会漏下一个写入方）。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from bladex_core.attribution import LinkJudgeResult, LinkVerdict
from bladex_proxy.linking import IndexJudgmentJournal, _is_link_verdict
from bladex_proxy.storage.memory_index import MemoryIndex


@pytest.fixture()
def index():
    with tempfile.TemporaryDirectory() as tmpdir:
        idx = MemoryIndex(Path(tmpdir) / "index", embedder=None, read_only=False)
        idx.open()
        yield idx
        idx.close()


@pytest.mark.parametrize("verdict,is_l4", [
    ("link:m-abc123", True),
    ("none", True),
    ("uncertain", True),
    ("LINK:M-ABC", True),                 # 大小写不敏感
    ("consolidation:add", False),         # M2 写入时裁决
    ("consolidation:update", False),
    ("consolidation:noop", False),
    ("", False),
    ("something:else", False),            # 未来的第三个写入方也不许污染
])
def test_link_verdict_whitelist(verdict: str, is_l4: bool):
    assert _is_link_verdict(verdict) is is_l4


def test_m2_record_does_not_masquerade_as_l4_judgment(index):
    """核心回归：只有 M2 记录时，L4 读台账必须 miss（→ 正常送 judge）。"""
    index.append_judgment("f-1", [{"fact_id": "f-0"}], "consolidation:add", "m",
                          reason="Adds new detail not covered by neighbours")
    j = IndexJudgmentJournal(index, judge_model="m")
    assert j.get_latest_judgment("f-1") is None, (
        "M2 的 consolidation 记录被当成了一条 L4 判决 —— 那会让 fact 伪 uncertain 留池"
    )


def test_l4_record_still_hits(index):
    """L4 自己的记录照常命中（台账复用 = 重建零 LLM 成本，不能一起关掉）。"""
    index.append_judgment("f-2", [], "link:m-target", "m")
    r = IndexJudgmentJournal(index, judge_model="m").get_latest_judgment("f-2")
    assert r is not None and r.verdict == LinkVerdict.LINK and r.matter_id == "m-target"


def test_latest_l4_wins_even_if_m2_written_later(index):
    """M2 记录**写在后面**（seq 更大）也不许盖住 L4 判决——
    这正是生产时序：归属在写入裁决之后。"""
    index.append_judgment("f-3", [], "link:m-first", "m")
    index.append_judgment("f-3", [], "consolidation:add", "m")
    r = IndexJudgmentJournal(index, judge_model="m").get_latest_judgment("f-3")
    assert r is not None and r.verdict == LinkVerdict.LINK
    assert r.matter_id == "m-first"


def test_multiple_l4_records_takes_latest(index):
    index.append_judgment("f-4", [], "link:m-old", "m")
    index.append_judgment("f-4", [], "consolidation:noop", "m")
    index.append_judgment("f-4", [], "none", "m")
    r = IndexJudgmentJournal(index, judge_model="m").get_latest_judgment("f-4")
    assert r is not None and r.verdict == LinkVerdict.NONE


def test_uncertain_l4_record_still_readable(index):
    """L4 自己写的 uncertain 仍是合法命中（与"没有判决"要分得开）。"""
    j = IndexJudgmentJournal(index, judge_model="m")
    j.put_judgment("f-5", [], LinkJudgeResult(
        fact_id="f-5", verdict=LinkVerdict.UNCERTAIN, reason="genuinely ambiguous"))
    r = j.get_latest_judgment("f-5")
    assert r is not None and r.verdict == LinkVerdict.UNCERTAIN
