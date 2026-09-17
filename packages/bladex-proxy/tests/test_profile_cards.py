"""U10（ADR-0026 R5）：画像卡与工具习惯。

覆盖：规则文件被动捕获（确定性）/ 工具计数器聚合 / 渲染 / audience 隔离
（codex 工具习惯不出现在 hermes 注入）/ 重建等价（重放两次一致）/
U7 ①平面接线（get_profile_cards 进常驻卡）/ 默认关零回归。
"""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest
from bladex_core.profile import detect_rule_files
from bladex_proxy.models import Identity, ToolEvent, Turn, TurnStatus
from bladex_proxy.storage.memory_hub import MemoryHub
from bladex_proxy.storage.memory_index import MemoryIndex

# （原 `_pin_index_recall_on` 夹具随 2026-09-03 S1 删除：该开关与主动检索路径已不存在。）


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


_RULE_TEXT = ("Contents of CLAUDE.md (project instructions):\n"
              "# BladeX — AI 协作开发指南\n跑测试只用 .venv/bin/python -m pytest。")


def test_detect_rule_files_deterministic():
    got = detect_rule_files(_RULE_TEXT)
    assert got and got[0][0] == "CLAUDE.md"
    assert got == detect_rule_files(_RULE_TEXT)  # 确定性
    assert detect_rule_files("普通消息没有规则文件") == []


def _put_turn(ledger: MemoryHub, *, agent: str, idx: int, system: str = "",
              tools: list[str] | None = None) -> str:
    identity = Identity(user_id="u1", agent_id=agent, session_id=f"s-{agent}",
                        turn_index=idx)
    events = [ToolEvent(tool_name=t, direction="call") for t in (tools or [])]
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user",
                 "content": f"这是第 {idx} 条足够长的普通用户消息，用于消费。"})
    turn = Turn(identity=identity, model="m", request_messages=msgs,
                response_text="ok", tool_events=events, status=TurnStatus.OK,
                ts=datetime(2026, 8, 1, 12, 0, idx, tzinfo=UTC))
    key = identity.storage_key(f"1719900{idx}-0")
    ledger.put(key, turn)
    return key


def _make_hub(tmpdir: str) -> MemoryHub:
    ledger = MemoryHub(Path(tmpdir) / "rocksdb")
    ledger.open()
    _put_turn(ledger, agent="codex", idx=0, system=_RULE_TEXT,
              tools=["bash", "bash", "read_file"])
    _put_turn(ledger, agent="codex", idx=1, tools=["bash"])
    _put_turn(ledger, agent="hermes:default", idx=2, tools=["web_search"])
    return ledger


def test_profile_capture_and_render(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BLADEX_PROFILE_CARDS", "1")
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger = _make_hub(tmpdir)
        index = MemoryIndex(Path(tmpdir) / "index", embedder=MockEmbedder(), read_only=False)
        index.open()
        index.rebuild_from_hub(ledger)

        # User 卡：规则文件捕获
        cards = index.get_profile_cards("u1", "codex")
        flat = [ln for card in cards for ln in card]
        assert any("CLAUDE.md" in ln for ln in flat)
        # Agent 卡：工具计数（bash 3 次 > read_file 1 次）
        agent_line = next(ln for ln in flat if ln.startswith("[Agent 习惯: codex]"))
        assert "bash(3)" in agent_line and "read_file(1)" in agent_line
        index.close()
        ledger.close()


def test_profile_audience_isolation(monkeypatch: pytest.MonkeyPatch):
    """codex 的工具习惯不出现在 hermes 的卡里（audience 隔离）。"""
    monkeypatch.setenv("BLADEX_PROFILE_CARDS", "1")
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger = _make_hub(tmpdir)
        index = MemoryIndex(Path(tmpdir) / "index", embedder=MockEmbedder(), read_only=False)
        index.open()
        index.rebuild_from_hub(ledger)
        flat = [ln for card in index.get_profile_cards("u1", "hermes:default") for ln in card]
        agent_lines = [ln for ln in flat if ln.startswith("[Agent 习惯")]
        assert agent_lines and "hermes" in agent_lines[0]
        assert "bash" not in agent_lines[0]
        assert "web_search(1)" in agent_lines[0]
        index.close()
        ledger.close()


def test_profile_rebuild_equivalence(monkeypatch: pytest.MonkeyPatch):
    """两个独立 Memory Index 重放同一 Memory Hub → 画像卡逐字一致（G6）。"""
    monkeypatch.setenv("BLADEX_PROFILE_CARDS", "1")
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger = _make_hub(tmpdir)
        rendered = []
        for name in ("p2a", "p2b"):
            index = MemoryIndex(Path(tmpdir) / name, embedder=MockEmbedder(), read_only=False)
            index.open()
            index.rebuild_from_hub(ledger)
            rendered.append(index.get_profile_cards("u1", "codex"))
            index.close()
        assert rendered[0] == rendered[1] and rendered[0]
        ledger.close()


def test_profile_explicitly_off(monkeypatch: pytest.MonkeyPatch):
    """显式关闭：不产出画像卡（回退保证）。

    ADR-0027 §5.4 起该机制**默认开**（此前默认关，导致验收过了但发出去的仍是旧行为），
    所以这条回退保证要显式设开关为 0 —— 回滚通道本身没变，变的是默认值。
    """
    monkeypatch.setenv("BLADEX_PROFILE_CARDS", "0")
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger = _make_hub(tmpdir)
        index = MemoryIndex(Path(tmpdir) / "index", embedder=MockEmbedder(), read_only=False)
        index.open()
        index.rebuild_from_hub(ledger)
        assert index.get_profile_cards("u1", "codex") == []
        index.close()
        ledger.close()


