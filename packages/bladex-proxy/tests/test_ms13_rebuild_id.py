"""MS-13（2026-08-10）：judgment 台账记 rebuild_id。

台账跨重建累积、clear() 有意不清（重建零 LLM 成本），但不带"哪一轮重建"
维度——08-09 复验两次被历史残渣弄脏读数（12:44 失明期判决被拿去对 16:33
新库做落盘核对，报假 🔴）。钉三件事：

  - append_judgment 逐条盖 rebuild_id（run 起始时间戳，非空）；
  - full rebuild 刷新 rebuild_id（一次全量 = 一个新 run）；
  - 存量记录（无 rebuild_id 键）反序列化 -> ""（读侧当"历史"处理，零回归）。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from bladex_proxy.models import JudgmentRecord
from bladex_proxy.storage.memory_hub import MemoryHub
from bladex_proxy.storage.memory_index import MemoryIndex


class _Embedder:
    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


def test_append_judgment_stamps_rebuild_id():
    with tempfile.TemporaryDirectory() as tmpdir:
        index = MemoryIndex(Path(tmpdir) / "index", embedder=None, read_only=False)
        index.open()
        index.append_judgment("f-1", [{"fact_id": "f-0"}], "consolidation:update", "m")
        recs = index.scan_judgments("f-1")
        assert recs and recs[0][1].rebuild_id != ""
        assert recs[0][1].rebuild_id == index._rebuild_id
        index.close()


def test_full_rebuild_refreshes_rebuild_id(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        index = MemoryIndex(Path(tmpdir) / "index", embedder=_Embedder(),
                            read_only=False)
        index.open()
        old_id = index._rebuild_id
        index._rebuild_id = "19700101T000000"   # 人为造旧 id，验证 full 会刷新
        ledger = MemoryHub(Path(tmpdir) / "ledger")
        ledger.open()
        index.rebuild_from_hub(ledger, full=True)
        assert index._rebuild_id != "19700101T000000"
        assert index._rebuild_id >= old_id      # 时间戳单调
        ledger.close()
        index.close()


def test_legacy_record_without_rebuild_id_deserializes_empty():
    rec = JudgmentRecord.model_validate({
        "fact_id": "f-legacy", "candidates": [], "verdict": "none",
    })
    assert rec.rebuild_id == ""
