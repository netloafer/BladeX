"""U4B-B2：会话事件折叠进 Matter 卡 lifecycle（ADR-0026 铁律3）。

真实路径：Memory Hub 写 turn → rebuild_from_hub（manual 边钉归属）→ Matter.lifecycle。
覆盖：折叠生效 / 同轮去重 / ts 用 turn 时间（G6 重建等价）/ 开关关闭零回归。
"""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest
from bladex_core.consolidation_proxy import _deterministic_fact_id
from bladex_core.matter import EdgeTargetType, Matter, MatterStatus
from bladex_proxy.models import Identity, Turn, TurnStatus
from bladex_proxy.storage.memory_hub import MemoryHub
from bladex_proxy.storage.memory_index import MemoryIndex


class MockEmbedder:
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


_TS = datetime(2026, 7, 20, 12, 0, 0, tzinfo=UTC)
_MSG = "上一轮任务卡被打回了，原因是测试没跑真实路径，请修复后重新提交。"


def _put_turn(ledger: MemoryHub, content: str = _MSG, idx: int = 0) -> str:
    identity = Identity(user_id="u1", agent_id="claude-code", session_id="s1", turn_index=idx)
    turn = Turn(
        identity=identity,
        model="test",
        request_messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": content},
        ],
        response_text="收到，开始修复。",
        status=TurnStatus.OK,
        ts=_TS,
    )
    key = identity.storage_key(f"1719900{idx}-0")
    ledger.put(key, turn)
    return key


def _setup(tmpdir: str) -> tuple[MemoryHub, MemoryIndex, str]:
    """Memory Hub 一条打回消息的 turn + Memory Index 里预建 Matter + manual 边钉住归属。"""
    ledger = MemoryHub(Path(tmpdir) / "rocksdb")
    ledger.open()
    key = _put_turn(ledger)
    index = MemoryIndex(Path(tmpdir) / "index", embedder=MockEmbedder(), read_only=False)
    index.open()
    matter = Matter(matter_id="m_card", title="T8b ANN 索引", status=MatterStatus.ACTIVE)
    index.add_matter(matter)
    # manual 边钉住归属（L1 manual 压过一切自动层）——无蒸馏器时 fact 内容=原文
    fact_id = _deterministic_fact_id(key, _MSG)
    index.assign_manual("m_card", EdgeTargetType.FACT, fact_id)
    return ledger, index, key


def test_lifecycle_folding_records_event(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BLADEX_MATTER_LIFECYCLE", "1")
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger, index, _key = _setup(tmpdir)
        index.rebuild_from_hub(ledger)

        m = index.get_matter("m_card")
        events = [e.event for e in m.lifecycle]
        assert "reworked" in events
        ev = next(e for e in m.lifecycle if e.event == "reworked")
        assert "打回" in ev.detail
        # ts 用 turn 时间戳，不是 rebuild 时刻（G6 重建等价性）
        assert ev.ts == _TS
        # record_lifecycle bump version
        assert m.version >= 1
        index.close()
        ledger.close()


def test_lifecycle_folding_dedup_same_turn(monkeypatch: pytest.MonkeyPatch):
    """同 (matter, ledger_key, event) 只记一次——同轮多 fact 归同 Matter 不重复。"""
    monkeypatch.setenv("BLADEX_MATTER_LIFECYCLE", "1")
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger = MemoryHub(Path(tmpdir) / "rocksdb")
        ledger.open()
        # 同一 turn 两条 user 消息 → 两条 fact，同一 ledger_key
        identity = Identity(user_id="u1", agent_id="claude-code", session_id="s1", turn_index=0)
        turn = Turn(
            identity=identity,
            model="test",
            request_messages=[
                {"role": "user", "content": "任务卡被打回了，原因是测试没跑真实路径，请注意。"},
                {"role": "user", "content": "另外打回的还有文档部分，同样需要修改后重新提交。"},
            ],
            response_text="ok",
            status=TurnStatus.OK,
            ts=_TS,
        )
        key = identity.storage_key("17199001-0")
        ledger.put(key, turn)

        index = MemoryIndex(Path(tmpdir) / "index", embedder=MockEmbedder(), read_only=False)
        index.open()
        matter = Matter(matter_id="m_card", title="任务卡", status=MatterStatus.ACTIVE)
        index.add_matter(matter)
        for content in ("任务卡被打回了，原因是测试没跑真实路径，请注意。",
                        "另外打回的还有文档部分，同样需要修改后重新提交。"):
            index.assign_manual("m_card", EdgeTargetType.FACT,
                             _deterministic_fact_id(key, content))
        index.rebuild_from_hub(ledger)

        m = index.get_matter("m_card")
        assert [e.event for e in m.lifecycle].count("reworked") == 1
        index.close()
        ledger.close()


def test_lifecycle_folding_rollback_channel(monkeypatch: pytest.MonkeyPatch):
    """回滚通道：`BLADEX_MATTER_LIFECYCLE=0` → lifecycle 不写、逐字回到旧行为（G10）。

    ADR-0028 §4 / E2.1 起该机制**默认开启**（本文件其余用例即默认形态），
    本条守的是关闭通道仍然有效——旧的"默认关"断言随拍板一并改写。
    """
    monkeypatch.setenv("BLADEX_MATTER_LIFECYCLE", "0")
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger, index, _key = _setup(tmpdir)
        index.rebuild_from_hub(ledger)
        m = index.get_matter("m_card")
        assert m.lifecycle == []
        index.close()
        ledger.close()


def test_lifecycle_folding_on_by_default():
    """ADR-0028 §4：默认开——不设任何环境变量也应折叠出 lifecycle 事件。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger, index, _key = _setup(tmpdir)
        index.rebuild_from_hub(ledger)
        m = index.get_matter("m_card")
        assert [e.event for e in m.lifecycle].count("reworked") == 1
        index.close()
        ledger.close()


def test_lifecycle_folding_deterministic_across_replays(monkeypatch: pytest.MonkeyPatch):
    """两个独立 Memory Index 重放同一 Memory Hub → lifecycle 逐字节一致（G6）。"""
    monkeypatch.setenv("BLADEX_MATTER_LIFECYCLE", "1")
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger = MemoryHub(Path(tmpdir) / "rocksdb")
        ledger.open()
        key = _put_turn(ledger)

        dumps = []
        for name in ("p2a", "p2b"):
            index = MemoryIndex(Path(tmpdir) / name, embedder=MockEmbedder(), read_only=False)
            index.open()
            matter = Matter(matter_id="m_card", title="T8b ANN 索引",
                            status=MatterStatus.ACTIVE)
            index.add_matter(matter)
            index.assign_manual("m_card", EdgeTargetType.FACT,
                             _deterministic_fact_id(key, _MSG))
            index.rebuild_from_hub(ledger)
            m = index.get_matter("m_card")
            dumps.append([e.model_dump(mode="json") for e in m.lifecycle])
            index.close()
        assert dumps[0] == dumps[1] and dumps[0]  # 一致且非空
        ledger.close()


def test_no_event_message_no_lifecycle(monkeypatch: pytest.MonkeyPatch):
    """无事件信号的消息 → 不写 lifecycle（准入：宁漏勿误）。"""
    monkeypatch.setenv("BLADEX_MATTER_LIFECYCLE", "1")
    msg = "帮我查一下 LanceDB 的向量索引参数应该怎么配置比较合适。"
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger = MemoryHub(Path(tmpdir) / "rocksdb")
        ledger.open()
        key = _put_turn(ledger, content=msg)
        index = MemoryIndex(Path(tmpdir) / "index", embedder=MockEmbedder(), read_only=False)
        index.open()
        index.add_matter(Matter(matter_id="m_card", title="配置", status=MatterStatus.ACTIVE))
        index.assign_manual("m_card", EdgeTargetType.FACT, _deterministic_fact_id(key, msg))
        index.rebuild_from_hub(ledger)
        assert index.get_matter("m_card").lifecycle == []
        index.close()
        ledger.close()
