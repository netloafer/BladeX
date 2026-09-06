"""U8 / G3（ADR-0025 I5）：Memory Index 蒸馏只读原始请求，永不读 reconstruction。

I5 从「天然满足」变「主动维护」——本测试是 schema 层保险的执行侧守护：
构造带 reconstruction（clarify_text + fingerprint + degrade_plan）的 Turn，
断言 rebuild 产出的蒸馏候选/Fact 不含任何重构痕迹。
"""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path

from bladex_proxy.models import Identity, ReconstructionRecord, Turn, TurnStatus
from bladex_proxy.storage.memory_index import MemoryIndex
from bladex_proxy.storage.memory_hub import MemoryHub

_CLARIFY = "[BladeX 澄清] 这里的『它』指代 T8b 的 ANN 索引任务（候选说明，可忽略）"
_FP_TOKEN = "fingerprint-matter-m1-token"


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


def test_index_never_consumes_reconstruction():
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger = MemoryHub(Path(tmpdir) / "rocksdb")
        ledger.open()
        identity = Identity(user_id="u1", agent_id="claude-code",
                            session_id="s1", turn_index=0)
        turn = Turn(
            identity=identity,
            model="test",
            request_messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user",
                 "content": "原始用户消息：把 recall 阈值定为均值 0.95、最差 0.70。"},
            ],
            response_text="好的，已记录阈值决定。",
            status=TurnStatus.OK,
            ts=datetime(2026, 8, 1, tzinfo=UTC),
            reconstruction=ReconstructionRecord(
                layer="ABC",
                degrade_plan={1: "excerpt"},
                fingerprint={"matters": [_FP_TOKEN]},
                clarify_text=_CLARIFY,
                index_watermark="wm-1",
            ),
        )
        key = identity.storage_key("17199001-0")
        ledger.put(key, turn)

        # Memory Hub 往返后 reconstruction 字段本身要在（审计），蒸馏不吃它
        loaded = ledger.get(key)
        assert loaded.reconstruction is not None
        assert loaded.reconstruction.clarify_text == _CLARIFY

        index = MemoryIndex(Path(tmpdir) / "index", embedder=MockEmbedder(), read_only=False)
        index.open()
        index.rebuild_from_hub(ledger)

        facts = index.all_facts()
        assert facts, "原始 user 消息应正常蒸出候选（passthrough）"
        for f in facts:
            blob = f"{f.content} {f.source_text} {' '.join(f.entities or [])}"
            assert _CLARIFY not in blob, "澄清块泄漏进 Memory Index（违反 I5/G3）"
            assert _FP_TOKEN not in blob, "重构指纹泄漏进 Memory Index（违反 I5/G3）"
            assert "[BladeX 澄清]" not in blob
        # 蒸馏源头恒为原始 request_messages
        assert any("原始用户消息" in f.content for f in facts)
        index.close()
        ledger.close()


def test_reconstruction_none_fallback():
    """历史 Turn（无 reconstruction 字段）反序列化走 None 默认——U2 迁移兼容。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger = MemoryHub(Path(tmpdir) / "rocksdb")
        ledger.open()
        identity = Identity(user_id="u1", agent_id="a", session_id="s", turn_index=0)
        turn = Turn(identity=identity, model="m",
                    request_messages=[{"role": "user", "content": "普通消息，足够长的普通消息内容。"}],
                    response_text="ok", status=TurnStatus.OK,
                    ts=datetime(2026, 8, 1, tzinfo=UTC))
        key = identity.storage_key("17199002-0")
        ledger.put(key, turn)
        loaded = ledger.get(key)
        assert loaded.reconstruction is None
        ledger.close()
