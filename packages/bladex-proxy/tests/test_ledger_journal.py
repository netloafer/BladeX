"""Memory Hub 派生台账单元测试（T5，ADR-0018 §4.1/§4.3）。

验收①: 写读台账 + scan 隔离 + 台账墓碑不复活。
验收②: get_distill 命中语义（同 hash+model+ver 命中，任一变则 miss）。

注：turn->judgment 自动级联（删 turn 连带墓碑其专属裁决台账）留 T7 实现
（需 fact_id = hash(ledger_key+content)，依赖 T6 蒸馏产出 content）。distill 台账
是内容寻址的共享资源，不级联（见 test_turn_tombstone_does_not_cascade_to_shared_distill）。
"""

import pytest
from bladex_core.distillation import DistillFact, MatterProposal
from bladex_proxy.models import (
    Identity,
    TombstoneTargetType,
    Turn,
    TurnStatus,
)
from bladex_proxy.storage.memory_hub import (
    _DISTILL_PREFIX,
    MemoryHub,
    _source_text_hash,
)


@pytest.fixture
def db(tmp_path):
    ledger = MemoryHub(tmp_path / "ledger")
    ledger.open()
    yield ledger
    ledger.close()


def _facts() -> list[DistillFact]:
    return [DistillFact(content="用户喜欢 e5", kind="preference", entities=["e5"])]


def _proposals() -> list[MatterProposal]:
    return [MatterProposal(title="嵌入模型选型", entities=["e5"])]


def _make_turn(user_text: str, turn_key: str = "u1/a1/s1/100-0") -> tuple[Turn, str]:
    turn = Turn(
        identity=Identity(user_id="u1", agent_id="a1", session_id="s1"),
        model="test",
        request_messages=[{"role": "user", "content": user_text}],
        response_text="ok",
        status=TurnStatus.OK,
    )
    return turn, turn_key


# ── 验收①: 写读台账 ──


def test_distill_write_read(db):
    """写蒸馏台账 -> 读取命中，内容正确还原。"""
    key = db.append_distill("some user text", "deepseek-v4", "v1",
                            _facts(), _proposals())
    assert key.startswith("distill/")

    rec = db.get_distill("some user text", "deepseek-v4", "v1")
    assert rec is not None
    assert rec.distill_model == "deepseek-v4"
    assert rec.prompt_ver == "v1"
    assert len(rec.facts) == 1
    assert rec.facts[0].content == "用户喜欢 e5"
    assert rec.facts[0].kind == "preference"
    assert len(rec.matter_proposals) == 1
    assert rec.matter_proposals[0].title == "嵌入模型选型"


def test_distill_idempotent_write_if_absent(db):
    """同 key 二次 append 不覆盖--首次蒸馏结果保留（命中复用=零 LLM 成本）。"""
    db.append_distill("text A", "m1", "v1",
                      [DistillFact(content="首次结果", kind="preference")], [])
    db.append_distill("text A", "m1", "v1",
                      [DistillFact(content="二次结果不应覆盖", kind="event")], [])

    rec = db.get_distill("text A", "m1", "v1")
    assert rec.facts[0].content == "首次结果"


def test_distill_returns_key_stable(db):
    """同输入两次 append 返回同 key（确定性 key）。"""
    k1 = db.append_distill("text A", "m1", "v1", _facts(), [])
    k2 = db.append_distill("text A", "m1", "v1", _facts(), [])
    assert k1 == k2


# ── 验收②: get_distill 命中语义 ──


def test_distill_hit_semantics(db):
    """同 source_text+model+ver 命中，任一变则 miss。"""
    db.append_distill("text A", "m1", "v1", _facts(), _proposals())

    assert db.get_distill("text A", "m1", "v1") is not None   # 命中
    assert db.get_distill("text B", "m1", "v1") is None       # source_text 变 -> miss
    assert db.get_distill("text A", "m2", "v1") is None       # model 变 -> miss
    assert db.get_distill("text A", "m1", "v2") is None       # prompt_ver 变 -> miss


# ── 验收①: scan 隔离 ──


def test_scan_prefix_isolates_ledger(db):
    """写 distill/judgment 后 scan_prefix/count 不含它们（前缀隔离）。"""
    db.append_distill("text A", "m1", "v1", _facts(), [])
    db.append_judgment("fact_1", [{"id": "m_a"}], "none", "m")

    assert list(db.scan_prefix("")) == []
    assert db.count() == 0


def test_scan_prefix_returns_only_turns_not_ledger(db):
    """有 turn + 台账时，scan_prefix 只返回 turn。"""
    turn, turn_key = _make_turn("hello")
    db.put(turn_key, turn)
    db.append_distill("hello", "m1", "v1", _facts(), [])
    db.append_judgment("fact_1", [], "none", "m")

    turns = list(db.scan_prefix("u1/a1/s1/"))
    assert len(turns) == 1
    assert turns[0][0] == turn_key


# ── 台账墓碑不复活 ──


def test_distill_tombstoned_not_revived(db):
    """蒸馏台账被墓碑 -> get 返回 None（Memory Index 重建不复活）。"""
    db.append_distill("text A", "m1", "v1", _facts(), [])
    h = _source_text_hash("text A")
    distill_key = f"{_DISTILL_PREFIX}{h}/m1/v1"

    db.append_tombstone(TombstoneTargetType.DISTILL, distill_key)

    assert db.is_tombstoned(distill_key)
    assert db.get_distill("text A", "m1", "v1") is None


def test_judgment_append_and_scan(db):
    """写裁决 -> scan 命中；多次写可重判（追加式，不同 key）。"""
    k1 = db.append_judgment("fact_1", [{"id": "m_a"}], "link:m_a", "glm-flash")
    k2 = db.append_judgment("fact_1", [{"id": "m_b"}], "none", "glm-flash")
    assert k1 != k2  # 不同 seq -> 不同 key

    all_j = list(db.scan_judgments())
    assert len(all_j) == 2
    assert {j.verdict for _, j in all_j} == {"link:m_a", "none"}
    assert all(j.fact_id == "fact_1" for _, j in all_j)


def test_judgment_scan_by_fact_id(db):
    """按 fact_id 前缀扫描。"""
    db.append_judgment("fact_1", [], "none", "m")
    db.append_judgment("fact_2", [], "none", "m")

    assert len(list(db.scan_judgments(fact_id="fact_1"))) == 1
    assert len(list(db.scan_judgments(fact_id="fact_2"))) == 1
    assert len(list(db.scan_judgments(fact_id="fact_3"))) == 0


def test_judgment_tombstoned_skipped(db):
    """墓碑的 judgment 被 scan 跳过（不复活）。"""
    k1 = db.append_judgment("fact_1", [], "none", "m")
    db.append_tombstone(TombstoneTargetType.JUDGMENT, k1)
    db.append_judgment("fact_1", [], "link:m_x", "m")

    all_j = list(db.scan_judgments())
    assert len(all_j) == 1
    assert all_j[0][1].verdict == "link:m_x"


# ── distill 共享保护（turn 墓碑不级联删共享 distill）──


def test_turn_tombstone_does_not_cascade_to_shared_distill(db):
    """distill 是内容寻址的共享资源：删一个 turn 不删其共享的 distill。

    同 source_text 跨 turn 复用是 feature（同内容同蒸馏结果）。删 turn A 级联删
    distill 会误伤同样引用它的 turn B，且 write-if-absent + 不复活三者冲突--
    故 distill 不级联。turn->judgment 级联（专属，fact_id 含 turn key）留 T7。
    """
    user_text = "共享文本"
    turn, turn_key = _make_turn(user_text)
    db.put(turn_key, turn)
    db.append_distill(user_text, "m1", "v1", _facts(), [])

    db.append_tombstone(TombstoneTargetType.TURN, turn_key)

    # distill 仍可读（turn 墓碑不级联删共享 distill）
    assert db.get_distill(user_text, "m1", "v1") is not None
