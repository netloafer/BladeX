"""批 AB2：MQ-I44（`UnitMatrix` 不靠 `id()`）+ MQ-V41（`/metrics` 真实读值）。

| 用例 | 判据 |
|---|---|
| `test_second_call_compares_against_its_own_vectors` | 同一 consolidator 跨两次调用、第二次的列表撞上第一次的地址：novelty 按第二次的向量算 |
| `test_second_call_with_longer_list` | 同上，第二次的列表更长（尾部入池、头部不许沿用旧行） |
| `test_no_id_identity_in_matrix_code` | 矩阵相关的两个模块里没有 `id()` 调用 |
| `test_unit_matrix_contract` | 一个矩阵配一个只增列表；变短才重来 |
| `test_metrics_reads_real_unknown_event_count` | Hub 里写一条未知类型事件、扫描后，`/metrics` 真实输出的 gauge = 1 |
"""

from __future__ import annotations

import ast
import re
import tempfile
from pathlib import Path
from typing import Any

import msgpack
import numpy as np
import pytest
from bladex_core import consolidation_proxy, vector_matrix
from bladex_core.consolidation_proxy import ProxyConsolidator
from bladex_core.fact import Fact
from bladex_core.vector_matrix import UnitMatrix
from bladex_proxy.config import ProxyConfig
from bladex_proxy.server import create_app
from fastapi.testclient import TestClient

X, Y, Z, W = ([1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0])
CANDIDATE_TEXT = "候选正文"


class _Embedder:
    model_identity = "test:i44"
    model_name = "i44"
    available = True

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [Z for _ in texts]


def _fact(i: int, emb: list[float]) -> Fact:
    return Fact(id=f"f{i}", content=f"已有 {i}", embedding=list(emb))


def _consolidator(monkeypatch: pytest.MonkeyPatch) -> ProxyConsolidator:
    """候选固定一条、向量恒为 Z；`id()` 恒定 = 新列表复用旧地址的最坏情形。"""
    monkeypatch.setattr(vector_matrix, "id", lambda _o: 1, raising=False)
    c = ProxyConsolidator(embedder=_Embedder(), novelty_threshold=0.98)
    monkeypatch.setattr(c, "_collect_candidates", lambda _turns: [{
        "content": CANDIDATE_TEXT, "ledger_key": "local/a/s/1", "session_id": "s", "user_id": "u",
        "logical_turn": 1,
    }])
    return c


def test_second_call_compares_against_its_own_vectors(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _consolidator(monkeypatch)
    first = c.consolidate_turns([], existing_facts=[_fact(1, X), _fact(2, Y)])
    assert [f.content for f in first] == [CANDIDATE_TEXT], "第一次：Z 与 X、Y 正交，新颖"
    second = c.consolidate_turns([], existing_facts=[_fact(3, Z), _fact(4, W)])
    assert second == [], "第二次：库里已有 Z，必须判重；沿用第一次的行就会判成新颖"


def test_second_call_with_longer_list(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _consolidator(monkeypatch)
    assert c.consolidate_turns([], existing_facts=[_fact(1, X), _fact(2, Y)])
    # 头部两行换成 Z：沿用旧行时只有第三行（W）入池，看不到 Z
    assert c.consolidate_turns([], existing_facts=[_fact(3, Z), _fact(4, Z), _fact(5, W)]) == []


def test_no_id_identity_in_matrix_code() -> None:
    for mod in (vector_matrix, consolidation_proxy):
        tree = ast.parse(Path(mod.__file__).read_text("utf-8"))
        calls = [n.lineno for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "id"]
        assert calls == [], f"{mod.__name__} 里有 id() 调用：{calls}"


def test_unit_matrix_contract() -> None:
    q = np.array([1, 0, 0, 0], dtype=np.float32)
    rows = [X, Y]
    m = UnitMatrix()
    m.sync(rows)
    rows.append(X)
    m.sync(rows)
    assert m.sims(q).tolist() == [1.0, 0.0, 1.0]
    del rows[1:]                    # 变短 ⇒ 重来
    m.sync(rows)
    assert m.sims(q).tolist() == [1.0]


def _put_unknown_event(hub: Any) -> None:  # noqa: ANN401
    hub.db[b"admin_event/0001"] = msgpack.packb(
        {"event_type": "from_the_future", "matter_id": "m1", "target_key": "",
         "ts": "2026-11-03T00:00:00+00:00", "payload": {}}, use_bin_type=True)


def _gauge(text: str) -> float:
    found = re.findall(r"^bladex_hub_unknown_admin_events (\S+)$", text, flags=re.M)
    assert len(found) == 1, text
    return float(found[0])


def test_metrics_reads_real_unknown_event_count() -> None:
    cfg = ProxyConfig(hard_rules=["MUST be polite"], upstream_model="openai/test",
                      upstream_api_key="sk-fake", rocksdb_path=f"{tempfile.mkdtemp()}/rocksdb")
    with TestClient(create_app(cfg)) as client:
        hub = client.app.state.hub
        assert _gauge(client.get("/metrics").text) == 0
        _put_unknown_event(hub)
        assert list(hub.scan_admin_events()) == []
        assert _gauge(client.get("/metrics").text) == 1
        assert list(hub.scan_admin_events()) == []
        assert _gauge(client.get("/metrics").text) == 1, "同一条多次扫描只算一次"
