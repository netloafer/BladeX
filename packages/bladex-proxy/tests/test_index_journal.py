"""Memory Index 派生台账单元测试（ADR-0020 T1.3，台账从 Memory Hub 迁 Memory Index meta）。

验收：MemoryIndex 台账读写 + write-if-absent + 命中语义 + scan 隔离；
      IndexDistillJournal / IndexJudgmentJournal 实现 protocol 适配；
      IndexDistillJournal 延迟 bind 安全降级（consolidator 启动顺序：journal 先于 index 创建）。

背景：ADR-0020 T1 把派生台账从 Memory Hub 迁 Memory Index meta--consolidator 独占 Memory Index 写锁，
修 Memory Hub 旧实现因 read_only Memory Hub 而写失败 100% 的根因。
"""

import pytest
from bladex_core.attribution import LinkJudgeResult, LinkVerdict
from bladex_core.distillation import DistillFact, DistillOutput, MatterProposal
from bladex_proxy.linking import IndexJudgmentJournal
from bladex_proxy.storage.memory_index import MemoryIndex, IndexDistillJournal


@pytest.fixture
def index(tmp_path):
    db = MemoryIndex(tmp_path / "index", embedder=None, read_only=False)
    db.open()
    yield db
    db.close()


def _facts() -> list[DistillFact]:
    return [DistillFact(content="用户喜欢 e5", kind="preference", entities=["e5"])]


def _proposals() -> list[MatterProposal]:
    return [MatterProposal(title="嵌入模型选型", entities=["e5"])]


# ── MemoryIndex 台账读写 ──


def test_index_distill_write_read(index):
    """写蒸馏台账 -> 读取命中，内容正确还原。"""
    key = index.append_distill("some user text", "deepseek-v4", "v1", _facts(), _proposals())
    assert key.startswith("distill/")

    rec = index.get_distill("some user text", "deepseek-v4", "v1")
    assert rec is not None
    assert rec.distill_model == "deepseek-v4"
    assert rec.prompt_ver == "v1"
    assert len(rec.facts) == 1
    assert rec.facts[0].content == "用户喜欢 e5"
    assert len(rec.matter_proposals) == 1
    assert rec.matter_proposals[0].title == "嵌入模型选型"


def test_index_distill_idempotent_write_if_absent(index):
    """同 key 二次 append 不覆盖--首次蒸馏结果保留（命中复用=零 LLM 成本）。"""
    k1 = index.append_distill("text A", "m1", "v1", _facts(), [])
    k2 = index.append_distill("text A", "m1", "v1", _facts(), [])
    assert k1 == k2


def test_index_distill_hit_semantics(index):
    """命中语义：同 source_text + model + ver 才命中，任一变则 miss。"""
    index.append_distill("text A", "m1", "v1", _facts(), [])
    assert index.get_distill("text A", "m1", "v1") is not None   # 命中
    assert index.get_distill("text B", "m1", "v1") is None       # source_text 变 -> miss
    assert index.get_distill("text A", "m2", "v1") is None       # model 变 -> miss
    assert index.get_distill("text A", "m1", "v2") is None       # prompt_ver 变 -> miss


def test_index_judgment_scan_and_latest(index):
    """裁决台账追加 + scan 隔离（按 fact_id 前缀）。"""
    k1 = index.append_judgment("fact_1", [{"id": "m_a"}], "link:m_a", "glm-flash")
    k2 = index.append_judgment("fact_1", [{"id": "m_b"}], "none", "glm-flash")
    assert k1 != k2
    assert len(index.scan_judgments()) == 2
    assert len(index.scan_judgments(fact_id="fact_1")) == 2
    assert len(index.scan_judgments(fact_id="fact_2")) == 0


# ── IndexDistillJournal protocol 适配 ──


def test_index_distill_ledger_protocol(index):
    """IndexDistillJournal 实现 DistillJournalProtocol：miss 写、命中读。"""
    journal = IndexDistillJournal(index)
    assert journal.get_distill("text", "m1", "v1") is None  # 空
    out = DistillOutput(facts=_facts(), matter_proposals=_proposals(), model_name="m1")
    journal.put_distill("text", "m1", "v1", out)
    got = journal.get_distill("text", "m1", "v1")
    assert got is not None
    assert len(got.facts) == 1
    assert got.model_name == "m1"


def test_index_distill_ledger_lazy_bind_safe(index):
    """未 bind 时 get/put 安全降级不抛（consolidator 启动：journal 先于 index 创建）。"""
    journal = IndexDistillJournal()  # index=None
    assert journal.get_distill("text", "m1", "v1") is None
    out = DistillOutput(facts=[], matter_proposals=[], model_name="m1")
    assert journal.put_distill("text", "m1", "v1", out) == ""
    # bind 后可用
    journal.bind(index)
    journal.put_distill("text", "m1", "v1", out)
    assert journal.get_distill("text", "m1", "v1") is not None


# ── IndexJudgmentJournal protocol 适配 ──


def test_index_judgment_ledger_protocol(index):
    """IndexJudgmentJournal 实现 JudgmentJournalProtocol：put + get_latest。"""
    journal = IndexJudgmentJournal(index, judge_model="glm-flash")
    assert journal.get_latest_judgment("fact_1") is None  # 空
    result = LinkJudgeResult(
        fact_id="fact_1", verdict=LinkVerdict.LINK, matter_id="m_a",
        summary_rewrite="摘要", reason="test",
    )
    journal.put_judgment("fact_1", [{"id": "m_a"}], result)
    latest = journal.get_latest_judgment("fact_1")
    assert latest is not None
    assert latest.verdict == LinkVerdict.LINK
    assert latest.matter_id == "m_a"


def test_index_judgment_ledger_latest_wins(index):
    """同 fact_id 多条裁决，get_latest 返回最新（最大 seq）。"""
    journal = IndexJudgmentJournal(index, judge_model="glm-flash")
    journal.put_judgment("fact_1", [], LinkJudgeResult(
        fact_id="fact_1", verdict=LinkVerdict.LINK, matter_id="m_a", reason="r1"))
    journal.put_judgment("fact_1", [], LinkJudgeResult(
        fact_id="fact_1", verdict=LinkVerdict.NONE, reason="r2"))  # 后写=最新
    latest = journal.get_latest_judgment("fact_1")
    assert latest is not None
    assert latest.verdict == LinkVerdict.NONE  # 最新覆盖
