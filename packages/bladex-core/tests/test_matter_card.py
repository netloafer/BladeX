"""U4：Matter 卡扩展（ADR-0026 §4.1）——participants / lifecycle / open_issues / version。

全部确定性可重算（touch_participant / record_lifecycle 零 LLM）。真跑聚合逻辑。
"""

from __future__ import annotations

from bladex_core.matter import (
    _MAX_MATTER_LIFECYCLE,
    Matter,
    MatterOrigin,
    MatterStatus,
)


def _matter() -> Matter:
    return Matter(matter_id="m1", title="给 BladeX 加 /v1/messages",
                  status=MatterStatus.ACTIVE, origin=MatterOrigin.AUTO)


def test_card_fields_default_empty() -> None:
    m = _matter()
    assert m.participants == [] and m.lifecycle == []   # open_issues 已删（F0.3 H1 销账）
    assert m.version == 0


def test_touch_participant_aggregates_by_agent() -> None:
    m = _matter()
    m.touch_participant("hermes")
    m.touch_participant("hermes")
    m.touch_participant("codex")
    assert len(m.participants) == 2
    hermes = next(p for p in m.participants if p.agent_id == "hermes")
    assert hermes.turns == 2 and hermes.first_seen is not None and hermes.last_seen is not None
    codex = next(p for p in m.participants if p.agent_id == "codex")
    assert codex.turns == 1


def test_record_lifecycle_bumps_version() -> None:
    m = _matter()
    m.record_lifecycle("created", "从提案诞生")
    m.record_lifecycle("reworked", "任务卡被打回")
    assert m.version == 2
    assert [e.event for e in m.lifecycle] == ["created", "reworked"]
    assert m.lifecycle[1].detail == "任务卡被打回"


def test_lifecycle_capped_fifo() -> None:
    m = _matter()
    for i in range(_MAX_MATTER_LIFECYCLE + 10):
        m.record_lifecycle("tick", f"e{i}")
    # 封顶 FIFO：只保留最近 _MAX_MATTER_LIFECYCLE 条，最旧被丢
    assert len(m.lifecycle) == _MAX_MATTER_LIFECYCLE
    assert m.lifecycle[-1].detail == f"e{_MAX_MATTER_LIFECYCLE + 9}"
    # version 仍按总次数递增（确定性）
    assert m.version == _MAX_MATTER_LIFECYCLE + 10


def test_card_backward_compatible() -> None:
    """旧 Matter（无卡扩展字段）可反序列化，新字段走默认。"""
    m = _matter()
    old = m.model_dump(mode="json")
    for k in ("participants", "lifecycle", "version"):
        old.pop(k, None)
    m2 = Matter.model_validate(old)
    assert m2.version == 0 and m2.participants == [] and m2.lifecycle == []
