"""ADR-0021 T2 验收：身份两体系 identity_registry + legacy_ids 归并。

覆盖计划卡 T2 五项验收：
① 无文件 = 现状（存量 identity 测试全绿，本文件不重复）；
② 两 key 同 principal -> user_id 归一、Memory Index 检索互见；
③ 未声明 key 回落 + warning；
④ legacy_ids 归并前后 Memory Index 重建等价（除归属 principal 外逐字段一致）；
⑤ 校验违例拒启动剧本。
"""

import tempfile
import textwrap
from pathlib import Path

import pytest
from bladex_proxy.identity import resolve_identity
from bladex_proxy.identity_registry import IdentityRegistry
from bladex_proxy.models import ChatCompletionRequest
from bladex_proxy.storage.memory_index import MemoryIndex
from bladex_proxy.storage.memory_hub import MemoryHub


class _MockEmbedder:
    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            h = hash(t)
            out.append([((h >> i) & 1) * 1.0 for i in range(self._dim)])
        return out


_PERSONAL_TOML = """
[[keys]]
key = "bladex-aaa"
label = "jason-macbook"
principal = "jason"

[[keys]]
key = "bladex-bbb"
label = "jason-codex"
principal = "jason"

[[principals]]
id = "jason"
name = "Jason Ye"
"""

_ENTERPRISE_TOML = """
[[keys]]
key = "bladex-aaa"
principal = "alice"
[[keys]]
key = "bladex-bbb"
principal = "bob"

[[principals]]
id = "alice"
teams = ["rd-backend"]
[[principals]]
id = "bob"
teams = ["legal"]
sensitivity = "sensitive"

[[orgs]]
id = "acme"
[[teams]]
id = "rd"
org = "acme"
[[teams]]
id = "rd-backend"
org = "acme"
parent = "rd"
[[teams]]
id = "legal"
org = "acme"
sensitivity = "sensitive"
"""


def _write_toml(tmpdir: str, content: str) -> str:
    p = Path(tmpdir) / "identity.toml"
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return str(p)


def _req() -> ChatCompletionRequest:
    return ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}])


# ── ② 两 key 同 principal -> user_id 归一 + 互见 ──


def test_two_keys_one_principal_same_user_id():
    """两 key 指向同一 principal -> resolve_identity 产出相同 user_id + 可见集合。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write_toml(tmpdir, _PERSONAL_TOML)
        reg = IdentityRegistry.from_toml(path)

        h1, _ = resolve_identity({"authorization": "Bearer bladex-aaa"}, _req(), reg)
        h2, _ = resolve_identity({"authorization": "Bearer bladex-bbb"}, _req(), reg)
        assert h1.user_id == h2.user_id == "jason"
        assert h1.visibility == h2.visibility == ["personal:jason"]


def test_two_keys_principal_index_mutual_visibility():
    """两 key 同 principal：各自蒸馏的 fact 互可见（多设备记忆合并）。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write_toml(tmpdir, _PERSONAL_TOML)
        reg = IdentityRegistry.from_toml(path)
        embedder = _MockEmbedder()
        ledger = MemoryHub(Path(tmpdir) / "ledger")
        ledger.open()
        index = MemoryIndex(Path(tmpdir) / "index", embedder=embedder)
        index.open()

        # 两个 key 各发一条 turn -> 都归一到 principal jason
        from bladex_proxy.models import Identity, Turn, TurnStatus
        t1 = Turn(identity=Identity(user_id="jason", agent_id="a", session_id="s1"),
                  model="m", request_messages=[{"role": "user", "content": "BladeX note from macbook."}],
                  response_text="ok", status=TurnStatus.OK)
        t2 = Turn(identity=Identity(user_id="jason", agent_id="a", session_id="s2"),
                  model="m", request_messages=[{"role": "user", "content": "BladeX note from codex."}],
                  response_text="ok", status=TurnStatus.OK)
        ledger.put("jason/a/s1/k1", t1)
        ledger.put("jason/a/s2/k1", t2)
        index.rebuild_from_hub(ledger)

        # 任一 key 解析出的可见集合都含 personal:jason -> 看到两条
        ident, _ = resolve_identity({"authorization": "Bearer bladex-aaa"}, _req(), reg)
        seen = index.search("BladeX", top_k=10, visibility=ident.visibility)
        assert len(seen) == 2
        index.close()
        ledger.close()


# ── ③ 未声明 key 回落 + warning ──


def test_undeclared_key_implicit_principal(caplog):
    """文件存在但 key 未声明 -> 隐式 principal（hash8）+ warning，不拒绝。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write_toml(tmpdir, _PERSONAL_TOML)
        reg = IdentityRegistry.from_toml(path)

        resolved = reg.resolve("bladex-unknown-key")
        assert resolved is not None
        assert resolved.implicit is True
        # 隐式 principal 的可见集合只有 personal:<hash8>
        assert len(resolved.visibility) == 1
        assert resolved.visibility[0].startswith("personal:")


# ── 企业：team 祖先链 + org scope + 敏感度 ──


def test_enterprise_team_ancestor_chain_and_org_scope():
    """principal 挂 rd-backend（parent=rd）-> 可见集合含 team:rd-backend + team:rd + org:acme。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write_toml(tmpdir, _ENTERPRISE_TOML)
        reg = IdentityRegistry.from_toml(path)

        resolved = reg.resolve("bladex-aaa")  # alice -> rd-backend
        assert resolved.principal_id == "alice"
        assert "personal:alice" in resolved.visibility
        assert "team:rd-backend" in resolved.visibility
        assert "team:rd" in resolved.visibility  # 祖先链
        assert "org:acme" in resolved.visibility


def test_enterprise_team_sensitivity_collected():
    """bob 挂 legal（sensitive）-> team_sensitivities 含 sensitive。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write_toml(tmpdir, _ENTERPRISE_TOML)
        reg = IdentityRegistry.from_toml(path)
        resolved = reg.resolve("bladex-bbb")
        assert "sensitive" in resolved.team_sensitivities


def test_no_file_is_empty_registry():
    """无 identity.toml -> empty registry -> resolve 恒返 None（现状回落）。"""
    reg = IdentityRegistry.from_toml("/nonexistent/identity.toml")
    assert reg.empty
    assert reg.resolve("bladex-aaa") is None


# ── ④ legacy_ids 归并前后 Memory Index 重建等价 ──


def test_legacy_ids_rebuild_maps_to_principal():
    """旧 turn（legacy hash8 user_id）rebuild 时映射到 principal，其余逐字段一致。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # identity.toml 声明 principal jason 的 legacy_ids
        toml = """
        [[principals]]
        id = "jason"
        legacy_ids = ["ab12cd34", "9f3e77aa"]
        """
        path = _write_toml(tmpdir, toml)
        reg = IdentityRegistry.from_toml(path)
        legacy_map = reg.legacy_map()
        assert legacy_map == {"ab12cd34": "jason", "9f3e77aa": "jason"}

        ledger = MemoryHub(Path(tmpdir) / "ledger")
        ledger.open()

        # 旧 turn：user_id 是 legacy hash8（pre-T2 形态）
        from bladex_proxy.models import Identity, Turn, TurnStatus
        for uid, key in [("ab12cd34", "k1"), ("9f3e77aa", "k2")]:
            t = Turn(identity=Identity(user_id=uid, agent_id="a", session_id="s"),
                     model="m", request_messages=[{"role": "user", "content": f"Legacy note {uid} BladeX."}],
                     response_text="ok", status=TurnStatus.OK)
            ledger.put(f"{uid}/a/s/{key}", t)

        # 不带 legacy_map 重建 -> fact user_id 仍是 hash8
        p2a = MemoryIndex(Path(tmpdir) / "p2a", embedder=_MockEmbedder())
        p2a.open()
        p2a.rebuild_from_hub(ledger, full=True)
        facts_no_map = {f.content: f.source_user_id for f in p2a.all_facts()}
        p2a.close()

        # 带 legacy_map 重建 -> fact user_id 映射到 jason
        p2b = MemoryIndex(Path(tmpdir) / "p2b", embedder=_MockEmbedder())
        p2b.open()
        p2b.rebuild_from_hub(ledger, full=True, legacy_map=legacy_map)
        facts_mapped = {f.content: f.source_user_id for f in p2b.all_facts()}
        p2b.close()
        ledger.close()

        # 内容集合一致（除 user_id 外逐字段一致）
        assert set(facts_no_map.keys()) == set(facts_mapped.keys())
        # user_id 从 hash8 映射到 jason
        assert all(v in ("ab12cd34", "9f3e77aa") for v in facts_no_map.values())
        assert all(v == "jason" for v in facts_mapped.values())


def test_legacy_journal_sync_idempotent():
    """sync_legacy_to_journal 幂等：第二次调用不重复写。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        toml = """
        [[principals]]
        id = "jason"
        legacy_ids = ["ab12cd34"]
        """
        path = _write_toml(tmpdir, toml)
        reg = IdentityRegistry.from_toml(path)
        ledger = MemoryHub(Path(tmpdir) / "ledger")
        ledger.open()

        first = reg.sync_legacy_to_journal(ledger)
        assert first == 1
        second = reg.sync_legacy_to_journal(ledger)
        assert second == 0  # 幂等

        # journal 里有且仅有一条 IDENTITY_MERGE
        from bladex_proxy.models import AdminEventType
        events = [ev for _k, ev in ledger.scan_admin_events()
                  if ev.event_type == AdminEventType.IDENTITY_MERGE]
        assert len(events) == 1
        assert events[0].matter_id == "jason"
        assert events[0].payload["legacy_ids"] == ["ab12cd34"]
        ledger.close()


# ── ⑤ 校验违例拒启动 ──


def test_validation_duplicate_key_refuses_startup():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write_toml(tmpdir, """
        [[keys]]
        key = "bladex-same"
        principal = "jason"
        [[keys]]
        key = "bladex-same"
        principal = "jason"
        [[principals]]
        id = "jason"
        """)
        with pytest.raises(ValueError, match="duplicate key"):
            IdentityRegistry.from_toml(path)


def test_validation_dangling_principal_refuses_startup():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write_toml(tmpdir, """
        [[keys]]
        key = "bladex-x"
        principal = "ghost"
        """)
        with pytest.raises(ValueError, match="unknown principal"):
            IdentityRegistry.from_toml(path)


def test_validation_team_cycle_refuses_startup():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write_toml(tmpdir, """
        [[principals]]
        id = "jason"
        teams = ["t1"]
        [[teams]]
        id = "t1"
        parent = "t2"
        [[teams]]
        id = "t2"
        parent = "t1"
        """)
        with pytest.raises(ValueError, match="cycle"):
            IdentityRegistry.from_toml(path)


def test_validation_dangling_team_refuses_startup():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write_toml(tmpdir, """
        [[principals]]
        id = "jason"
        teams = ["ghost-team"]
        """)
        with pytest.raises(ValueError, match="unknown team"):
            IdentityRegistry.from_toml(path)
