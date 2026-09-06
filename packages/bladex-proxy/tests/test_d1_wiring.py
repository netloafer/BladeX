"""G12.2 D1 **生产接线**（决策表 v2 → memory_index 归属循环）。

引擎单测在 core（test_task_state.py，14 组）；本文件只管"生产循环真的按表走"
（第七/八例教训：单测过 ≠ 接线通、跨批次路径必须真跨批）。全部走真实路径：
Memory Hub 写 turn → 两次 `rebuild_from_hub`（consolidator 60s 一批，
相邻轮天生跨批）。

钉住：
  ① 首轮 R5 → 管线 L5 新开（id = 首轮 ledger key，G12.1）+ taskstate 台账
    verdict=new（新开判决入账，ADR-0031 §14.1）；
  ② 次轮 R2 直挂：边 decision.layer=CONT / branch=R2、**零 L4 调用**、
    激活态日志 index_d1_decisions 可见；
  ③ 台账优先：预写 continue_to 记录 → 重放照办（branch=LEDGER），不重判；
  ④ 开关关 = 回归通道：无 CONT 边、无台账写入、行为回到改动前。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from bladex_core.attribution import LinkJudgeItem, LinkJudgeResult, LinkVerdict, new_matter_id
from bladex_core.distillation import DistillFact, DistillOutput, MatterProposal
from bladex_core.matter import Matter, MatterStatus
from bladex_proxy.models import Identity, SessionIdSource, TaskStateJudgment, Turn, TurnStatus
from bladex_proxy.storage.memory_hub import MemoryHub
from bladex_proxy.storage.memory_index import MemoryIndex

_TS = datetime(2026, 8, 20, 14, 0, 0, tzinfo=UTC)

# 量过的文本（出处 test_gm_continuation_wiring：延续=1 个新主体，分离得很开）
_T1_USER = "复核一下泰安仁信入股泰山啤酒的增资扩股方式，看看有没有问题"
_T1_REPLY = (
    "复核完成，泰安仁信的增资扩股方式存在两处问题：出资时间与工商登记不一致，"
    "另外增资扩股的股权比例与公告披露的数字对不上，需要再核一遍。"
)
_T2_FOLLOWUP = "增资扩股方式存在的问题，出资时间与工商登记不一致，需要再核一遍"


class StubDistiller:
    @property
    def model_name(self) -> str:
        return "d1-stub"

    @property
    def prompt_ver(self) -> str:
        return "d1-stub-001"

    def distill(self, user_message: str) -> DistillOutput:
        return DistillOutput(
            facts=[DistillFact(content=f"结论：{user_message[:40]}", kind="event",
                               entities=["泰安仁信", "增资扩股"])],
            matter_proposals=[MatterProposal(title="泰安仁信增资扩股复核",
                                             entities=["泰安仁信"])],
            model_name="d1-stub",
        )


class RecordingNoneJudge:
    def __init__(self) -> None:
        self.calls = 0

    @property
    def model_name(self) -> str:
        return "none-stub"

    def judge_links(self, items: list[LinkJudgeItem]) -> list[LinkJudgeResult]:
        self.calls += len(items)
        return [LinkJudgeResult(fact_id=it.fact_id, verdict=LinkVerdict.NONE, reason="stub")
                for it in items]


class MockEmbedder:
    def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * 64
            for i, ch in enumerate(t[:32]):
                v[(ord(ch) + i) % 64] += 0.1
            out.append(v)
        return out

    @property
    def available(self):
        return True


def _key(idx: int) -> str:
    return f"u1/claude-code/s1/{1755700000000 + idx * 60000}-0"


def _put_turn(ledger: MemoryHub, idx: int, user: str, reply: str,
              source: SessionIdSource) -> str:
    identity = Identity(user_id="u1", agent_id="claude-code", session_id="s1",
                        turn_index=idx, session_id_source=source)
    turn = Turn(identity=identity, model="test",
                request_messages=[{"role": "user", "content": user}],
                response_text=reply, status=TurnStatus.OK,
                ts=_TS + timedelta(minutes=idx), session_id_source=source)
    key = identity.storage_key(f"{1755700000000 + idx * 60000}-0")
    ledger.put(key, turn)
    return key


def _open(tmp_path: Path):
    ledger = MemoryHub(tmp_path / "rocksdb")
    ledger.open()
    judge = RecordingNoneJudge()
    index = MemoryIndex(tmp_path / "index", embedder=MockEmbedder(),
                        read_only=False, link_judge=judge,
                        distiller=StubDistiller())
    index.open()
    return ledger, index, judge


def _cont_edges(index: MemoryIndex) -> list:
    return [e for m in index.all_matters() for e in index.get_edges(m.matter_id)
            if (e.decision or {}).get("layer") == "CONT"]


def test_first_turn_r5_then_second_turn_r2_direct(tmp_path, monkeypatch) -> None:
    import structlog.testing

    monkeypatch.setenv("BLADEX_TASK_STATE_D1", "1")
    ledger, index, judge = _open(tmp_path)
    try:
        k1 = _put_turn(ledger, 0, _T1_USER, _T1_REPLY, SessionIdSource.FINGERPRINT)
        index.rebuild_from_hub(ledger)

        # ① 首轮：R5 → L5 新开（id = 首轮 ledger key）+ 台账 verdict=new
        rec1 = index.get_taskstate_judgment(k1)
        assert rec1 is not None and rec1.verdict == "new", rec1
        assert rec1.matter_id == new_matter_id(ledger_key=k1)
        assert index.get_matter(rec1.matter_id) is not None

        # ② 次轮（跨批次）：R2 直挂，零 L4
        k2 = _put_turn(ledger, 1, _T2_FOLLOWUP, "好的。", SessionIdSource.TAIL_CONTINUATION)
        judge.calls = 0
        with structlog.testing.capture_logs() as cap:
            index.rebuild_from_hub(ledger)
        cont = _cont_edges(index)
        assert cont, "次轮没有产出 CONT 边——直挂路径没接通"
        assert all((e.decision or {}).get("branch") == "R2" for e in cont), cont
        assert {e.matter_id for e in cont} == {rec1.matter_id}, "R2 没延续到锚"
        assert judge.calls == 0, "R2 轮仍调了 L4——零 LLM 直挂没生效"
        ev = next((e for e in cap if e.get("event") == "index_d1_decisions"), None)
        assert ev is not None and ev.get("R2", 0) >= 1 and ev.get("direct_facts", 0) >= 1, cap
        # R2 是延续不是新开判决——不入台账（决策表 v2 §6：R1–R4 有意不记）
        assert index.get_taskstate_judgment(k2) is None
    finally:
        index.close()
        ledger.close()


def test_ledger_priority_overrides_branching(tmp_path, monkeypatch) -> None:
    """③ 台账优先：预写 continue_to 记录 → 重放照办，不重判（防旋钮漂移的机制半边）。"""
    monkeypatch.setenv("BLADEX_TASK_STATE_D1", "1")
    ledger, index, judge = _open(tmp_path)
    try:
        k1 = _put_turn(ledger, 0, _T1_USER, _T1_REPLY, SessionIdSource.FINGERPRINT)
        index.add_matter(Matter(matter_id="m-pinnedcard99", title="台账钉住的卡",
                                status=MatterStatus.ACTIVE))
        index.append_taskstate_judgment(TaskStateJudgment(
            turn_key=k1, verdict="continue_to", matter_id="m-pinnedcard99",
            knob_ver="d1-knob-000"))
        index.rebuild_from_hub(ledger)
        edges = index.get_edges("m-pinnedcard99")
        assert edges, "台账 continue_to 记录没被照办"
        assert all((e.decision or {}).get("branch") == "LEDGER" for e in edges
                   if (e.decision or {}).get("layer") == "CONT")
    finally:
        index.close()
        ledger.close()


def test_flag_off_is_regression_channel(tmp_path, monkeypatch) -> None:
    """④ 开关关：无 CONT 边、无台账写入——归属行为回到改动前。"""
    monkeypatch.setenv("BLADEX_TASK_STATE_D1", "0")
    ledger, index, judge = _open(tmp_path)
    try:
        k1 = _put_turn(ledger, 0, _T1_USER, _T1_REPLY, SessionIdSource.FINGERPRINT)
        index.rebuild_from_hub(ledger)
        k2 = _put_turn(ledger, 1, _T2_FOLLOWUP, "好的。", SessionIdSource.TAIL_CONTINUATION)
        index.rebuild_from_hub(ledger)
        assert not _cont_edges(index)
        assert index.get_taskstate_judgment(k1) is None
        assert index.get_taskstate_judgment(k2) is None
    finally:
        index.close()
        ledger.close()
