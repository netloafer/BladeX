"""ADR-0021 T5 验收：Fact 血统继承 + scope 提升 -- consolidator + rebuild。

覆盖 T5 ③④：
③ rebuild 后血统保持（Turn.sensitivity 重放 -> Fact.exposure_ceiling 一致）；
④ scope 提升（SCOPE_PROMOTE 管理事件重放，rebuild 不丢失）。
"""

import tempfile
from pathlib import Path

from bladex_core.sensitivity import EXPOSURE_LOCAL, EXPOSURE_PUBLIC, SensitivityConfig
from bladex_proxy.models import Identity, Turn, TurnStatus
from bladex_proxy.storage.memory_index import MemoryIndex
from bladex_proxy.storage.memory_hub import MemoryHub


class _MockEmbedder:
    def __init__(self, dim=64):
        self._dim = dim

    def embed(self, texts):
        out = []
        for t in texts:
            h = hash(t)
            out.append([((h >> i) & 1) * 1.0 for i in range(self._dim)])
        return out


def _sensitive_turn(user_id, sensitivity, msg):
    return Turn(
        identity=Identity(user_id=user_id, agent_id="a", session_id="s"),
        model="m",
        request_messages=[{"role": "user", "content": msg}],
        response_text="ok", status=TurnStatus.OK,
        sensitivity=sensitivity,
    )


_SENS_CFG = SensitivityConfig(
    enabled=True,
    default_level="normal",
    levels={"normal": EXPOSURE_PUBLIC, "sensitive": EXPOSURE_LOCAL},
)


# ── ③ Fact 血统继承 + rebuild 等价 ──


def test_fact_exposure_ceiling_inherits_turn_sensitivity():
    """sensitive Turn 蒸馏的 Fact.exposure_ceiling == local；normal Turn -> public。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger = MemoryHub(Path(tmpdir) / "ledger")
        ledger.open()
        ledger.put("u1/a/s/k1", _sensitive_turn("u1", "sensitive", "secret project BladeX architecture details."))
        ledger.put("u2/a/s/k1", _sensitive_turn("u2", "normal", "public BladeX project status update."))

        index = MemoryIndex(Path(tmpdir) / "index", embedder=_MockEmbedder(),
                       sensitivity_config=_SENS_CFG)
        index.open()
        index.rebuild_from_hub(ledger)

        facts = {f.source_user_id: f for f in index.all_facts()}
        assert facts["u1"].exposure_ceiling == EXPOSURE_LOCAL
        assert facts["u2"].exposure_ceiling == EXPOSURE_PUBLIC
        # scope 默认 personal
        assert all(f.scope == f"personal:{f.source_user_id}" for f in facts.values())
        index.close()
        ledger.close()


def test_bloodline_rebuild_equivalence():
    """full rebuild x2 -> Fact.exposure_ceiling 逐条一致（血统可重放）。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger = MemoryHub(Path(tmpdir) / "ledger")
        ledger.open()
        ledger.put("u1/a/s/k1", _sensitive_turn("u1", "sensitive", "secret BladeX alpha."))
        ledger.put("u1/a/s/k2", _sensitive_turn("u1", "normal", "public BladeX beta."))

        index = MemoryIndex(Path(tmpdir) / "index", embedder=_MockEmbedder(),
                       sensitivity_config=_SENS_CFG)
        index.open()
        index.rebuild_from_hub(ledger, full=True)
        first = {f.id: (f.exposure_ceiling, f.scope) for f in index.all_facts()}
        index.rebuild_from_hub(ledger, full=True)
        second = {f.id: (f.exposure_ceiling, f.scope) for f in index.all_facts()}
        assert first == second
        index.close()
        ledger.close()


def test_disabled_sensitivity_all_facts_public():
    """sensitivity_config 关闭 -> 全部 Fact.exposure_ceiling=public（现状零回归）。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger = MemoryHub(Path(tmpdir) / "ledger")
        ledger.open()
        ledger.put("u1/a/s/k1", _sensitive_turn("u1", "sensitive", "secret BladeX."))
        index = MemoryIndex(Path(tmpdir) / "index", embedder=_MockEmbedder())  # no sens cfg
        index.open()
        index.rebuild_from_hub(ledger)
        for f in index.all_facts():
            assert f.exposure_ceiling == EXPOSURE_PUBLIC
        index.close()
        ledger.close()


# ── ④ scope 提升（SCOPE_PROMOTE 重放）──


def test_scope_promotion_replays_on_rebuild():
    """scope 提升记 SCOPE_PROMOTE 事件 -> rebuild 后 Fact.scope 保持提升值。"""
    from bladex_proxy.models import AdminEventType

    with tempfile.TemporaryDirectory() as tmpdir:
        ledger = MemoryHub(Path(tmpdir) / "ledger")
        ledger.open()
        ledger.put("u1/a/s/k1", _sensitive_turn("u1", "normal", "team BladeX knowledge."))

        index = MemoryIndex(Path(tmpdir) / "index", embedder=_MockEmbedder())
        index.open()
        index.rebuild_from_hub(ledger)
        facts = index.all_facts()
        assert facts
        fact_id = facts[0].id
        assert facts[0].scope == "personal:u1"  # 默认

        # 写 scope 提升事件：该 fact 提升到 team:rd
        ledger.append_admin_event(AdminEventType.SCOPE_PROMOTE, fact_id,
                              target_type="fact", new_scope="team:rd")

        # rebuild -> scope 提升重放
        index.rebuild_from_hub(ledger, full=True)
        promoted = index.get_fact(fact_id)
        assert promoted is not None
        assert promoted.scope == "team:rd"  # 提升不丢失
        index.close()
        ledger.close()


def test_scope_promote_endpoint_writes_event_and_applies_index():
    """`/admin/scope/promote` 端点：先写 SCOPE_PROMOTE 事件到 Memory Hub，再应用到 Memory Index。"""
    from bladex_proxy.config import ProxyConfig
    from bladex_proxy.models import AdminEventType
    from bladex_proxy.server import create_app
    from fastapi.testclient import TestClient

    tmpdir = tempfile.mkdtemp()
    config = ProxyConfig(
        hard_rules=["MUST be polite"], upstream_model="openai/test",
        upstream_api_key="sk-fake", rocksdb_path=f"{tmpdir}/rocksdb", index_path=f"{tmpdir}/index",
    )
    app = create_app(config)
    with TestClient(app) as client:
        # 先建一个 matter
        r = client.post("/admin/matters", json={"title": "team knowledge"})
        matter_id = r.json()["matter_id"]

        # 提升 scope 到 team:rd
        resp = client.post("/admin/scope/promote", json={
            "target_type": "matter", "target_key": matter_id, "new_scope": "team:rd",
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "promoted"

        # Memory Hub 有 SCOPE_PROMOTE 事件
        events = [e for _k, e in app.state.hub.scan_admin_events()
                  if e.event_type == AdminEventType.SCOPE_PROMOTE]
        assert len(events) == 1
        assert events[0].matter_id == matter_id
        assert events[0].payload["new_scope"] == "team:rd"

        # Memory Index matter.scope 已更新
        p2w = MemoryIndex(Path(tmpdir) / "index", read_only=False)
        p2w.open()
        m = p2w.get_matter(matter_id)
        assert m is not None
        assert m.scope == "team:rd"
        p2w.close()
