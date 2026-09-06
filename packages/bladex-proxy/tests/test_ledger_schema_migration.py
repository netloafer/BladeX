"""U2 Memory Hub schema 统一迁移测试（ADR-0026 §2.3 + ADR-0025 §5）。

两组：
  A. schema 字节一致性（纯 Pydantic，无 rocksdict）：storage_key/session_prefix 在
     principal_id 空时与旧格式逐字一致；非空时走 principal；Turn 新字段默认 + 旧数据兼容。
  B. 迁移集成（真实 rocksdict）：建小 Memory Hub → migrate() → 计数逐条一致 + 内部 key 全保留 +
     request_messages 逐字不变（I1/I5 红线）+ 新库能按新 schema 反序列化。

工程纪律：B 组真跑 migrate/verify 的真实路径，不占位。
"""

from __future__ import annotations

import importlib.util
import os
import sys

import msgpack
import rocksdict

from bladex_proxy.models import Identity, ReconstructionRecord, Turn, TurnStatus

# 迁移脚本按文件路径加载（scripts/ 非包）
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_MIG = os.path.join(_ROOT, "scripts", "migrate_ledger_schema.py")
_spec = importlib.util.spec_from_file_location("migrate_ledger_schema", _MIG)
assert _spec and _spec.loader
mig = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = mig
_spec.loader.exec_module(mig)


# ── A. schema 字节一致性 ──────────────────────────────────────────────────


def test_storage_key_byte_identical_when_principal_empty() -> None:
    idn = Identity(user_id="alice", agent_id="hermes", session_id="s1")
    assert idn.storage_key("e9") == "alice/hermes/s1/e9"
    assert idn.session_prefix() == "alice/hermes/s1/"
    assert idn.key_principal == "alice"


def test_storage_key_uses_principal_when_set() -> None:
    idn = Identity(user_id="legacy_a", agent_id="hermes", session_id="s1", principal_id="alice")
    assert idn.storage_key("e9") == "alice/hermes/s1/e9"
    assert idn.session_prefix() == "alice/hermes/s1/"
    assert idn.key_principal == "alice"


def test_turn_new_fields_default_and_old_data_loads() -> None:
    idn = Identity(user_id="alice")
    t = Turn(identity=idn)
    assert t.api_key_id == "" and t.org_id == "" and t.reconstruction is None
    # 模拟旧 Memory Hub 数据（无新字段）经 model_validate 仍可读，新字段走默认
    old = t.model_dump(mode="json")
    for f in ("api_key_id", "org_id", "reconstruction"):
        old.pop(f, None)
    t2 = Turn.model_validate(old)
    assert t2.api_key_id == "" and t2.org_id == "" and t2.reconstruction is None


def test_reconstruction_record_roundtrips_and_isolated() -> None:
    idn = Identity(user_id="alice")
    rr = ReconstructionRecord(layer="AB", degrade_plan={3: "minimal"}, clarify_text="推测：那个方案")
    t = Turn(identity=idn, reconstruction=rr, request_messages=[{"role": "user", "content": "hi"}])
    d = t.model_dump(mode="json")
    t2 = Turn.model_validate(d)
    assert t2.reconstruction is not None
    assert t2.reconstruction.layer == "AB"
    assert t2.reconstruction.clarify_text == "推测：那个方案"
    # request_messages 与 reconstruction 分离（I5 schema 层保险）
    assert t2.request_messages == [{"role": "user", "content": "hi"}]


# ── B. 迁移集成（真实 rocksdict）─────────────────────────────────────────


def _make_src_ledger(path: str) -> tuple[int, int]:
    """建一个含 Turn key + 内部 key 的小 Memory Hub。返回 (turn 数, 内部 key 数)。"""
    db = rocksdict.Rdict(path)
    turns = 0
    # 三条 Turn（去重后形态无所谓，migrate 是 raw 拷贝）
    for i in range(3):
        idn = Identity(user_id="alice", agent_id="hermes", session_id="s1")
        t = Turn(
            identity=idn,
            request_messages=[{"role": "user", "content": f"msg {i}"}],
            response_text=f"resp {i}",
            status=TurnStatus.OK,
        )
        db[idn.storage_key(f"e{i}").encode()] = msgpack.packb(
            t.model_dump(mode="json"), use_bin_type=True
        )
        turns += 1
    internal = 0
    for k in ("__msg__/abc", "tombstone/x", "admin_event/y", "distill/z", "judgment/w"):
        db[k.encode()] = msgpack.packb({"k": k}, use_bin_type=True)
        internal += 1
    db.close()
    return turns, internal


def test_migration_personal_mode_byte_identical(tmp_path) -> None:
    src = str(tmp_path / "src")
    dst = str(tmp_path / "dst")
    src_turns, src_internal = _make_src_ledger(src)

    counts = mig.migrate(src, dst, {})  # 个人模式：空映射
    assert counts["turn"] == src_turns
    assert counts["internal"] == src_internal
    assert counts["turn_key_rewritten"] == 0  # key 未改写

    # 校验（migrate + verify 的真实路径，assert 通过即达标）
    mig.verify(src, dst, {}, sample=10)

    # key 与值均字节一致
    sdb = rocksdict.Rdict(src, rocksdict.Options(), None, rocksdict.AccessType.read_only())
    ddb = rocksdict.Rdict(dst, rocksdict.Options(), None, rocksdict.AccessType.read_only())
    try:
        skeys = {mig._decode(k) for k in sdb.keys()}
        dkeys = {mig._decode(k) for k in ddb.keys()}
        assert skeys == dkeys
        # Turn 值逐字节一致
        k = "alice/hermes/s1/e0"
        assert bytes(sdb.get(k.encode())) == bytes(ddb.get(k.encode()))
        # 新库能按新 schema 读出 + 新字段 materialize
        t = Turn.model_validate(msgpack.unpackb(bytes(ddb.get(k.encode())), raw=False))
        assert t.api_key_id == "" and t.reconstruction is None
        assert t.request_messages == [{"role": "user", "content": "msg 0"}]
    finally:
        sdb.close()
        ddb.close()


def test_migration_enterprise_rewrites_principal_segment(tmp_path) -> None:
    src = str(tmp_path / "src")
    dst = str(tmp_path / "dst")
    # 造两个 legacy 首段。entry_id 全局唯一（真实来自 Redis stream），故两条 key 不碰撞——
    # 迁移后同映射到 alice 但 session/entry 不同，仍是两条独立 Turn。
    db = rocksdict.Rdict(src)
    for i, legacy in enumerate(("legacy_a", "legacy_b")):
        idn = Identity(user_id=legacy, agent_id="hermes", session_id=f"s{i}")
        t = Turn(identity=idn, request_messages=[{"role": "user", "content": "x"}])
        db[idn.storage_key(f"e{i}").encode()] = msgpack.packb(t.model_dump(mode="json"), use_bin_type=True)
    db[b"__msg__/keep"] = msgpack.packb({"a": 1}, use_bin_type=True)
    db.close()

    pmap = {"legacy_a": "alice", "legacy_b": "alice"}
    counts = mig.migrate(src, dst, pmap)
    assert counts["turn"] == 2
    assert counts["turn_key_rewritten"] == 2
    mig.verify(src, dst, pmap, sample=10)

    ddb = rocksdict.Rdict(dst, rocksdict.Options(), None, rocksdict.AccessType.read_only())
    try:
        dkeys = {mig._decode(k) for k in ddb.keys()}
        # 两条不同 legacy → 同 principal alice（不同 session/entry，不碰撞）
        assert "alice/hermes/s0/e0" in dkeys
        assert "alice/hermes/s1/e1" in dkeys
        assert "__msg__/keep" in dkeys        # 内部 key 保留
        assert not any(k.startswith("legacy_") for k in dkeys)
    finally:
        ddb.close()


def test_migration_refuses_nonempty_dst(tmp_path) -> None:
    src = str(tmp_path / "src")
    dst = str(tmp_path / "dst")
    _make_src_ledger(src)
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, "sentinel"), "w") as f:
        f.write("x")
    try:
        mig.migrate(src, dst, {})
        raise AssertionError("应拒绝非空目标目录")
    except SystemExit:
        pass
