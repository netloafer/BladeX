"""U11：全开关下的重建等价性（G6）——重放两次逐字节一致。

新机制全开（supersede / participants / lifecycle / profile / file_ref）时，
两个独立 Memory Index 从同一 Memory Hub 全量重建 → facts / matters / edges / profiles 逐字节一致。
这是活库全量重建（scripts/full_rebuild_u11.sh，需用户机跑）的机械前提。
"""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest
from bladex_proxy.models import Identity, ToolEvent, Turn, TurnStatus
from bladex_proxy.storage.memory_index import MemoryIndex
from bladex_proxy.storage.memory_hub import MemoryHub


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


def _put(ledger: MemoryHub, idx: int, agent: str, content: str,
         tools: list[str] | None = None) -> str:
    identity = Identity(user_id="u1", agent_id=agent, session_id=f"s-{agent}",
                        turn_index=idx)
    turn = Turn(
        identity=identity, model="m",
        request_messages=[{"role": "user", "content": content}],
        response_text="ok",
        tool_events=[ToolEvent(tool_name=t, direction="call") for t in (tools or [])],
        status=TurnStatus.OK,
        ts=datetime(2026, 8, 1, 12, 0, idx, tzinfo=UTC),
    )
    key = identity.storage_key(f"1719900{idx}-0")
    ledger.put(key, turn)
    return key


@pytest.fixture()
def _all_on(monkeypatch: pytest.MonkeyPatch):
    for k in ("BLADEX_SUPERSEDE_ENABLED", "BLADEX_MATTER_PARTICIPANTS",
              "BLADEX_MATTER_LIFECYCLE", "BLADEX_PROFILE_CARDS"):
        monkeypatch.setenv(k, "1")


# 等价性口径：排除"重放时刻"产生的易变时间戳（created_at/t_observed/t_valid/
# updated_at/first_seen/last_seen）——它们是摄入时钟，不是业务内容；
# 业务时间（lifecycle ts = turn 时间戳）保留在对比内。
_VOLATILE_FACT = {"embedding", "created_at", "t_observed", "t_valid"}
_VOLATILE_MATTER = {"embedding", "centroid", "created_at", "updated_at"}


def _dump(index: MemoryIndex) -> dict:
    facts = sorted((f.model_dump(mode="json", exclude=_VOLATILE_FACT)
                    for f in index.all_facts()), key=lambda d: d["id"])
    matters = []
    for m in index.all_matters():
        d = m.model_dump(mode="json", exclude=_VOLATILE_MATTER)
        d["participants"] = sorted(
            (p["agent_id"], p["turns"]) for p in d.get("participants", [])
        )
        matters.append(d)
    matters.sort(key=lambda d: d["matter_id"])
    edges = []
    for m in index.all_matters():
        edges.extend(e.model_dump(mode="json", exclude={"created_at"})
                     for e in index.get_edges(m.matter_id))
    edges.sort(key=lambda d: d["edge_id"])
    profiles = index.get_profile_cards("u1", "claude-code")
    return {"facts": facts, "matters": matters, "edges": edges, "profiles": profiles}


def test_full_rebuild_twice_byte_identical(_all_on):
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger = MemoryHub(Path(tmpdir) / "rocksdb")
        ledger.open()
        _put(ledger, 0, "claude-code", "任务卡被打回了，原因是测试没跑真实路径，请修复。",
             tools=["bash", "read_file"])
        _put(ledger, 1, "claude-code", "现在卡在 LanceDB 版本不支持 FTS 上，需要换方案。",
             tools=["bash"])
        _put(ledger, 2, "codex", "T8b 验收通过，recall 阈值定为均值 0.95、最差 0.70。")
        _put(ledger, 3, "hermes:default", "帮我查一下下周去东京的机票价格区间。")

        dumps = []
        for name in ("p2a", "p2b"):
            index = MemoryIndex(Path(tmpdir) / name, embedder=MockEmbedder(),
                           read_only=False)
            index.open()
            index.rebuild_from_hub(ledger, full=True)
            dumps.append(_dump(index))
            index.close()

        for section in ("facts", "matters", "edges", "profiles"):
            assert dumps[0][section] == dumps[1][section], f"{section} 不一致（违 G6）"
        assert dumps[0]["facts"], "应有 fact 产出"
        ledger.close()


def test_incremental_then_full_converges(_all_on):
    """增量消费后再 full 重建 → 与一次性 full 重建等价（clear+重放语义）。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger = MemoryHub(Path(tmpdir) / "rocksdb")
        ledger.open()
        _put(ledger, 0, "claude-code", "任务卡被打回了，原因是测试没跑真实路径，请修复。")
        _put(ledger, 1, "codex", "T8b 验收通过，recall 阈值定为均值 0.95。")

        # 路径A：增量两次（第一次消费全部，第二次无新）→ full
        p2a = MemoryIndex(Path(tmpdir) / "p2a", embedder=MockEmbedder(), read_only=False)
        p2a.open()
        p2a.rebuild_from_hub(ledger)
        p2a.rebuild_from_hub(ledger)
        p2a.rebuild_from_hub(ledger, full=True)
        da = _dump(p2a)
        p2a.close()

        # 路径B：直接 full
        p2b = MemoryIndex(Path(tmpdir) / "p2b", embedder=MockEmbedder(), read_only=False)
        p2b.open()
        p2b.rebuild_from_hub(ledger, full=True)
        db = _dump(p2b)
        p2b.close()

        for section in ("facts", "matters", "edges", "profiles"):
            assert da[section] == db[section], f"{section} 增量→full 与直接 full 不一致"
        ledger.close()
