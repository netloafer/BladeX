"""MQ-S27 lane 硬分流：**生产接线**（2026-08-20 Jason 拍板；决策表 v2 §7 前置边界）。

走真实路径（Memory Hub 写 turn → `rebuild_from_hub`，蒸馏用带 lane 标的
stub——lane 标本来就是蒸馏产出，stub 只是把 LLM 换成确定性）。单测 `fact_lane`
在 core（test_fact_lane.py），本文件只管"生产循环真的跳了"这半边
（第七例教训：机制单测通过 ≠ 生产接通）。

钉住：
  ① lane:profile 的 fact **入库但不挂卡**（fact 还在、可检索；无边、matter_id 空）；
  ② 同一轮的 lane:task fact 照常归属（分流不是整轮跳过）；
  ③ 激活态日志 `index_lane_hard_split lane_skipped>=1`（恒为 0 与没在跑必须可区分）；
  ④ 开关置 0 = 回归通道（profile fact 照旧挂卡，行为回到改动前）；
  ⑤ manual 手动映射压过分流（红线 6：用户显式指定的归属永远算数）。
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from bladex_core.attribution import UNASSIGNED_MATTER_ID, LinkJudgeItem, LinkJudgeResult, LinkVerdict
from bladex_core.consolidation_proxy import _deterministic_fact_id
from bladex_core.distillation import DistillFact, DistillOutput, MatterProposal
from bladex_core.matter import EdgeTargetType, Matter, MatterStatus
from bladex_proxy.models import Identity, SessionIdSource, Turn, TurnStatus
from bladex_proxy.storage.memory_hub import MemoryHub
from bladex_proxy.storage.memory_index import MemoryIndex

_TS = datetime(2026, 8, 20, 12, 0, 0, tzinfo=UTC)
_USER = "帮我复核泰安仁信入股泰山啤酒的持股比例，另外记一下我的终端提示符换了"
_PROFILE_CONTENT = "用户的终端提示符是 'ook-Pro BladeX %'，主机名 ook-Pro"
_TASK_CONTENT = "泰安仁信入股泰山啤酒的持股比例为 2.0607%，以入股时间口径为准"


class LaneStubDistiller:
    """确定性蒸馏 stub：一轮吐两条 fact——lane:profile 一条、lane:task 一条。"""

    @property
    def model_name(self) -> str:
        return "lane-stub"

    @property
    def prompt_ver(self) -> str:
        return "lane-stub-001"

    def distill(self, user_message: str) -> DistillOutput:
        # 🔴 matter_proposals 不能省：L5 新开用第一个非空提案标题开卡（MQ-D8），
        # 无提案 → 全轮 fact 留池、谁都没有边 —— 第一版夹具就是这么把
        # "task fact 照常归属"的断言饿死的（2026-08-20 用户机 gate 两红）。
        return DistillOutput(
            facts=[
                DistillFact(content=_PROFILE_CONTENT, kind="general",
                            item_kind="profile_obs", tags="lane:profile",
                            entities=["ook-Pro"]),
                DistillFact(content=_TASK_CONTENT, kind="event",
                            item_kind="assertion", tags="lane:task",
                            entities=["泰安仁信", "泰山啤酒"]),
            ],
            matter_proposals=[MatterProposal(title="泰安仁信入股比例复核",
                                             entities=["泰安仁信", "泰山啤酒"])],
            model_name="lane-stub",
        )


class NoneJudge:
    """一律判 none 的 L4 桩——让 task fact 走 L5 新开（无 judge 会留池，
    task fact 就拿不到 matter_id，②的断言无从谈起）。"""

    @property
    def model_name(self) -> str:
        return "none-stub"

    def judge_links(self, items: list[LinkJudgeItem]) -> list[LinkJudgeResult]:
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


def _put_turn(ledger: MemoryHub, idx: int) -> str:
    identity = Identity(user_id="u1", agent_id="claude-code", session_id="s1",
                        turn_index=idx, session_id_source=SessionIdSource.FINGERPRINT)
    turn = Turn(identity=identity, model="test",
                request_messages=[{"role": "user", "content": _USER}],
                response_text="好的。", status=TurnStatus.OK, ts=_TS,
                session_id_source=SessionIdSource.FINGERPRINT)
    key = identity.storage_key(f"{1755690000000 + idx * 1000}-0")
    ledger.put(key, turn)
    return key


def _rebuild(tmp_path: Path, *, pre=None) -> tuple[MemoryIndex, list]:
    import structlog.testing

    ledger = MemoryHub(tmp_path / "rocksdb")
    ledger.open()
    index = MemoryIndex(tmp_path / "index", embedder=MockEmbedder(),
                        read_only=False, link_judge=NoneJudge(),
                        distiller=LaneStubDistiller())
    index.open()
    key = _put_turn(ledger, 0)
    if pre is not None:
        pre(index, key)
    with structlog.testing.capture_logs() as cap:
        index.rebuild_from_hub(ledger)
    ledger.close()
    return index, cap


def _fact_by_content(index: MemoryIndex, marker: str):
    hits = [f for f in index.all_facts() if marker in (f.content or "")]
    assert hits, f"stub 蒸出的 fact 没入库（找 {marker!r}）——测试前提破了"
    return hits[0]


def _edge_targets(index: MemoryIndex) -> set[str]:
    return {e.target_key for m in index.all_matters()
            for e in index.get_edges(m.matter_id)}


def test_profile_lane_is_stored_but_not_attributed(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("BLADEX_LANE_HARD_SPLIT", "1")
    index, cap = _rebuild(tmp_path)
    try:
        prof = _fact_by_content(index, "终端提示符")
        task = _fact_by_content(index, "持股比例")
        # 前提断言：lane 标真的进了库（若 _merge_tags 把它丢了，本测试会靠
        # PROFILE_OBS kind 兜底蒙混过关——标丢失必须显式红，不许静默）
        assert "lane:profile" in (prof.tags or ""), prof.tags
        assert "lane:task" in (task.tags or ""), task.tags
        # ① 入库但不挂卡
        assert not prof.matter_id or prof.matter_id == UNASSIGNED_MATTER_ID
        assert prof.id not in _edge_targets(index)
        # ② 同轮 task fact 照常归属（L5 新开 provisional）
        assert task.matter_id and task.matter_id != UNASSIGNED_MATTER_ID
        assert task.id in _edge_targets(index)
        # ③ 激活态日志
        ev = next((e for e in cap if e.get("event") == "index_lane_hard_split"), None)
        assert ev is not None and ev.get("lane_skipped", 0) >= 1, cap
    finally:
        index.close()


def test_flag_off_restores_old_behaviour(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("BLADEX_LANE_HARD_SPLIT", "0")
    index, cap = _rebuild(tmp_path)
    try:
        prof = _fact_by_content(index, "终端提示符")
        task = _fact_by_content(index, "持股比例")
        # 前提断言：归属真的发生了（task 有边）——否则"prof 没边"两种成因分不开
        # （v1 夹具无提案时全轮留池，这条断言就是那次两红的教训）
        assert task.id in _edge_targets(index), "task fact 都没归属——夹具前提破了"
        assert prof.id in _edge_targets(index), "开关关了还在分流——回归通道坏了"
        assert not any(e.get("event") == "index_lane_hard_split" for e in cap)
    finally:
        index.close()


def test_manual_mapping_beats_lane_split(tmp_path, monkeypatch) -> None:
    """红线 6：用户手动把画像 fact 钉到某张卡上，分流不得推翻。"""
    monkeypatch.setenv("BLADEX_LANE_HARD_SPLIT", "1")

    def pre(index: MemoryIndex, key: str) -> None:
        index.add_matter(Matter(matter_id="m-manualcard01", title="手动钉的卡",
                                status=MatterStatus.ACTIVE))
        index.assign_manual("m-manualcard01", EdgeTargetType.FACT,
                            _deterministic_fact_id(key, _PROFILE_CONTENT))

    index, _cap = _rebuild(tmp_path, pre=pre)
    try:
        prof = _fact_by_content(index, "终端提示符")
        edges = {e.target_key for e in index.get_edges("m-manualcard01")}
        assert prof.id in edges, "manual 映射被 lane 分流吞了（红线 6）"
    finally:
        index.close()
