"""批 S · S3（拍板 0.3.0-(2)，MQ-I17）：`bladex ledger tombstone [--with-memory]` 两条路径。

真 Hub（tmp）+ 真 Index（tmp，StubDistiller/MockEmbedder）+ TestClient 端点 + CLI 录制桩。钉住：
  ① 默认路径：只写 LEDGER 墓碑；窗口 turn 不墓碑；响应 `derived_facts_retained` = 窗口 fact 数（非 0）+ hint；
  ② `--with-memory`：窗口内每条 turn 一条 TURN 墓碑；窗口外 turn 不动；
  ③ 🔴 与批 Q 探针同一判据：`--with-memory` 后**全量重建** fact 不得长回来；默认路径会（对照）；
  ④ 可重建：墓碑不物理删，`hub.db.get(key)` 原文仍在；
  ⑤ 阴性对照：去掉 TURN 墓碑写入 ⇒ ③ 必须红（用 monkeypatch 把 `tombstone_ledger_memory` 换成空操作）；
  ⑥ CLI：两条命令各打对应 admin 调用；无 `--yes` 时取消。
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

import pytest
from bladex_core.distillation import DistillFact, DistillOutput, MatterProposal
from bladex_core.ledger import new_ledger
from bladex_core.ledger_runtime import ActivationTable, activation_scope
from bladex_proxy import ledger_memory as _lm
from bladex_proxy.models import (
    AdminEventType,
    Identity,
    SessionIdSource,
    TombstoneTargetType,
    Turn,
    TurnStatus,
)
from bladex_proxy.storage.memory_hub import MemoryHub
from bladex_proxy.storage.memory_index import MemoryIndex

_L = "ldg-s3memory0001"


class StubDistiller:
    @property
    def model_name(self) -> str:
        return "s3-stub"

    @property
    def prompt_ver(self) -> str:
        return "s3-stub-001"

    def distill(self, user_message: str) -> DistillOutput:
        return DistillOutput(
            facts=[DistillFact(content=f"结论：{user_message[:40]}", kind="event", entities=["ledger-history"])],
            matter_proposals=[MatterProposal(title="ledger history 端点开发", entities=["ledger-history"])],
            model_name="s3-stub")


class MockEmbedder:
    def embed(self, texts):  # noqa: ANN001, ANN201
        out = []
        for t in texts:
            v = [0.0] * 64
            for i, ch in enumerate(t[:32]):
                v[(ord(ch) + i) % 64] += 0.1
            out.append(v)
        return out

    @property
    def available(self) -> bool:
        return True


def _put(hub: MemoryHub, seq: int, text: str, agent: str = "claude-code") -> str:
    ident = Identity(user_id="u1", agent_id=agent, session_id="s1", turn_index=seq,
                     session_id_source=SessionIdSource.FINGERPRINT)
    turn = Turn(identity=ident, model="test", request_messages=[{"role": "user", "content": text}],
                response_text="好的。", status=TurnStatus.OK, ts=datetime.now(UTC),
                session_id_source=SessionIdSource.FINGERPRINT)
    key = ident.storage_key(f"{int(datetime.now(UTC).timestamp() * 1000)}-0")
    hub.put(key, turn)
    time.sleep(0.02)
    return key


def _seed(hub: MemoryHub) -> tuple[list[str], str]:
    led = new_ledger(ledger_id=_L, title="S3 账本", goal="g")
    led.created_at = datetime.now(UTC).isoformat()
    hub.append_admin_event(AdminEventType.LEDGER_CREATE, "", ledger=led.model_dump(mode="json"))
    time.sleep(0.02)
    hub.append_admin_event(AdminEventType.LEDGER_SWITCH, "", agent_id="claude-code",
                           project_id="", session_id="s1", to_ledger_id=_L)
    time.sleep(0.02)
    inside = [_put(hub, 1, "实现 GET /admin/ledgers/{id}/history 端点"),
              _put(hub, 2, "加 CLI 子命令 bladex ledger history")]
    outside = _put(hub, 3, "另一个 agent 的无账本轮次，窗口外", agent="codex")
    return inside, outside


class _StubAgency:
    def __init__(self, pool):
        self.pool = pool
        self.activation = ActivationTable()
        self.activation.restore({activation_scope("claude-code", ""): _L})


@pytest.fixture
def env(tmp_path: Path, monkeypatch):
    """真 Hub + 真 Index（先重建出 fact），再起带真实 Hub 的 TestClient。"""
    from bladex_proxy.config import ProxyConfig
    from bladex_proxy.server import create_app
    from fastapi.testclient import TestClient

    hub_path, idx_path = tmp_path / "hub", tmp_path / "index"
    hub = MemoryHub(hub_path)
    hub.open()
    index = MemoryIndex(idx_path, embedder=MockEmbedder(), read_only=False, distiller=StubDistiller())
    index.open()
    inside, outside = _seed(hub)
    index.rebuild_from_hub(hub, full=True)
    n_facts = len(index.all_facts())
    index.close()
    hub.close()
    assert n_facts >= 3, "夹具没蒸出 fact，测试失去判别力"

    monkeypatch.setenv("BLADEX_AUTH_ENABLED", "false")
    monkeypatch.setenv("BLADEX_ROCKSDB_PATH", str(hub_path))
    monkeypatch.setenv("BLADEX_INDEX_PATH", str(idx_path))
    app = create_app(ProxyConfig())
    with TestClient(app) as c:
        led = new_ledger(ledger_id=_L, title="S3 账本", goal="g")
        c.app.state.agency = _StubAgency({_L: led})
        yield c, hub_path, idx_path, inside, outside


def _reopen(hub_path: Path, idx_path: Path):
    hub = MemoryHub(hub_path)
    hub.open()
    index = MemoryIndex(idx_path, embedder=MockEmbedder(), read_only=False, distiller=StubDistiller())
    index.open()
    return hub, index


def _turn_tombstones(hub: MemoryHub) -> set[str]:
    return {t.target_key for _k, t in hub.scan_tombstones() if t.target_type == TombstoneTargetType.TURN}


def test_default_path_keeps_memory_and_reports_count(env) -> None:
    c, hub_path, idx_path, inside, outside = env
    r = c.delete(f"/admin/ledgers/{_L}")
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["with_memory"] is False and d["window_turns"] == 2
    assert d["derived_facts_retained"] >= 2 and "--with-memory" in d["hint"]
    c.app.state.hub.close()
    hub, index = _reopen(hub_path, idx_path)
    try:
        assert _turn_tombstones(hub) == set(), "① 默认路径不许动 turn"
        n_before = len(index.all_facts())
        index.rebuild_from_hub(hub, full=True)
        assert len(index.all_facts()) == n_before, "③ 对照：默认路径全量重建 fact 仍在（记忆保留）"
    finally:
        index.close()
        hub.close()


def test_with_memory_tombstones_window_turns_and_rebuild_does_not_regrow(env) -> None:
    c, hub_path, idx_path, inside, outside = env
    r = c.delete(f"/admin/ledgers/{_L}", params={"with_memory": "true"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["with_memory"] is True and d["memory"]["turn_tombstones"] == 2
    c.app.state.hub.close()
    hub, index = _reopen(hub_path, idx_path)
    try:
        assert _turn_tombstones(hub) == set(inside), "② 窗口内两轮各一条 TURN 墓碑，窗口外不动"
        assert hub.get(outside) is not None
        visible = {k for k, _t in hub.scan_prefix("u1/claude-code/s1/")}
        for k in inside:
            assert hub.db.get(k.encode()) is not None, "④ 走墓碑不物理删：原文仍在 Hub（可重放）"
            assert k not in visible, "读路径（scan_prefix / 重建）跳过被墓碑 turn"
        index.rebuild_from_hub(hub, full=True)
        keys = {f.source_ledger_key for f in index.all_facts()}
        assert not (keys & set(inside)), "③ 全量重建后窗口 fact 不得长回来（MQ-I17 的反面）"
        assert outside in keys, "窗口外的 fact 照常重建"
    finally:
        index.close()
        hub.close()


def test_negative_control_no_turn_tombstones_means_regrowth(env, monkeypatch) -> None:
    """⑤ 去掉 TURN 墓碑写入 ⇒ 全量重建长回来（③ 必须红）。"""
    c, hub_path, idx_path, inside, outside = env
    monkeypatch.setattr(_lm, "tombstone_ledger_memory",
                        lambda hub, index, keys: {"turn_tombstones": 0, "index_cascaded": 0, "index_deferred": 0})
    r = c.delete(f"/admin/ledgers/{_L}", params={"with_memory": "true"})
    assert r.status_code == 200 and r.json()["memory"]["turn_tombstones"] == 0
    c.app.state.hub.close()
    hub, index = _reopen(hub_path, idx_path)
    try:
        assert _turn_tombstones(hub) == set()
        index.rebuild_from_hub(hub, full=True)
        keys = {f.source_ledger_key for f in index.all_facts()}
        assert keys & set(inside), "没有 TURN 墓碑 ⇒ 长回来（缺陷本体，MQ-I17）"
    finally:
        index.close()
        hub.close()


# ── ⑥ CLI ────────────────────────────────────────────────────────────────

class _Recorder:
    def __init__(self, resp):
        self.calls = []
        self.resp = resp

    def __call__(self, method, path, body=None, params=None, **kwargs):
        self.calls.append((method, path, body, params, kwargs))
        return self.resp


@pytest.fixture
def cli_env(tmp_path, monkeypatch, isolated_home):
    import os
    monkeypatch.chdir(tmp_path)
    for k in list(os.environ):
        if k.startswith("BLADEX_"):
            monkeypatch.delenv(k, raising=False)
    # 批 T0b 闸 B（MQ-I24）：不可逆操作前 CLI 会读 /ready 校验服务端协议版本。
    # 本文件的录制桩只替 `_admin_call`，没有服务端 ⇒ 给一个**匹配的**探针读数，
    # 让闸照常跑（它真的比对一次），只是比中。不匹配的那一支由
    # test_t0b_dryrun_gates.py 专门钉。
    from bladex_proxy import cli as _cli
    from bladex_proxy.admin_contract import ADMIN_CONTRACT_FIELD, ADMIN_CONTRACT_VERSION
    monkeypatch.setattr(_cli, "_probe_ready", lambda *_a, **_k: {
        "ready": True, "checks": {}, ADMIN_CONTRACT_FIELD: ADMIN_CONTRACT_VERSION,
        "build": "test"})


def test_cli_default_path_prints_retained_hint(cli_env, monkeypatch, capsys) -> None:
    from bladex_proxy import cli
    from bladex_proxy.cli import ledger_cmds as _ldg
    rec = _Recorder((200, {"status": "tombstoned", "ledger_id": _L, "with_memory": False,
                           "window_turns": 2, "derived_facts_retained": 5, "hint": "x"}))
    monkeypatch.setattr(_ldg, "_admin_call", rec)
    assert cli.main(["ledger", "tombstone", _L, "--yes"]) == 0
    assert rec.calls == [("DELETE", f"/admin/ledgers/{_L}", None, None, {"timeout": 15.0})]
    out = capsys.readouterr().out
    assert "derived memory kept: 5 facts" in out and "--with-memory" in out


def test_cli_with_memory_passes_flag_and_reports(cli_env, monkeypatch, capsys) -> None:
    from bladex_proxy import cli
    from bladex_proxy.cli import ledger_cmds as _ldg
    rec = _Recorder((200, {"status": "tombstoned", "ledger_id": _L, "with_memory": True, "window_turns": 2,
                           "memory": {"turn_tombstones": 2, "index_cascaded": 0, "index_deferred": 2}}))
    monkeypatch.setattr(_ldg, "_admin_call", rec)
    assert cli.main(["ledger", "tombstone", _L, "--with-memory", "--yes"]) == 0
    assert rec.calls == [("DELETE", f"/admin/ledgers/{_L}", None, {"with_memory": "true"}, {"timeout": 900.0})]
    assert "2 turn tombstones" in capsys.readouterr().out


def test_cli_dry_run_calls_readonly_and_never_confirms(cli_env, monkeypatch, capsys) -> None:
    """批 T0 / MQ-I21：`--dry-run` 传 `dry_run=true`、**不问确认**、不打 ✓ tombstoned。"""
    import typer
    from bladex_proxy import cli
    from bladex_proxy.cli import ledger_cmds as _ldg
    rec = _Recorder((200, {"status": "dry_run", "ledger_id": _L, "with_memory": True,
                           "dry_run": True, "window_turns": 45,
                           "window_scopes": ["codex\x1fp-e479f97c661a"],
                           "derived_facts_retained": 12, "backup_age_days": 0.3}))
    monkeypatch.setattr(_ldg, "_admin_call", rec)
    monkeypatch.setattr(typer, "confirm", lambda *_a, **_k: pytest.fail("dry-run 不许问确认"))
    assert cli.main(["ledger", "tombstone", _L, "--with-memory", "--dry-run"]) == 0
    assert rec.calls == [("DELETE", f"/admin/ledgers/{_L}", None,
                          {"with_memory": "true", "dry_run": "true"}, {})]
    out = capsys.readouterr().out
    assert "DRY RUN -- nothing written" in out and "turns in window: 45" in out
    assert "codex @ p-e479f97c661a" in out, "scope 分隔符是不可见的 \\x1f，必须拆开显示"
    assert "derived facts in window: 12" in out
    assert "✓ tombstoned" not in out, "dry-run 不许打成功态"


def test_cli_backup_notice_none_is_no_data_not_zero(cli_env, monkeypatch, capsys) -> None:
    """软闸：`None` = 没有备份 / 读不到，**不许**显示成"0 天前刚备份过"。"""
    from bladex_proxy import cli
    from bladex_proxy.cli import ledger_cmds as _ldg
    rec = _Recorder((200, {"status": "dry_run", "ledger_id": _L, "with_memory": False,
                           "dry_run": True, "window_turns": 0, "window_scopes": [],
                           "derived_facts_retained": 0, "backup_age_days": None}))
    monkeypatch.setattr(_ldg, "_admin_call", rec)
    assert cli.main(["ledger", "tombstone", _L, "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "backup: NO-DATA" in out and "0 days old" not in out


def test_cli_backup_notice_warns_when_stale(cli_env, monkeypatch, capsys) -> None:
    from bladex_proxy import cli
    from bladex_proxy.cli import ledger_cmds as _ldg
    rec = _Recorder((200, {"status": "tombstoned", "ledger_id": _L, "with_memory": False,
                           "window_turns": 2, "derived_facts_retained": 5, "hint": "x",
                           "backup_age_days": 42.0}))
    monkeypatch.setattr(_ldg, "_admin_call", rec)
    assert cli.main(["ledger", "tombstone", _L, "--yes"]) == 0
    out = capsys.readouterr().out
    assert "⚠ backup" in out and "42.0 days old" in out
    assert "✓ tombstoned" in out, "软闸只告警不阻断"


def test_cli_without_yes_cancels(cli_env, monkeypatch, capsys) -> None:
    import typer
    from bladex_proxy import cli
    from bladex_proxy.cli import ledger_cmds as _ldg
    rec = _Recorder((200, {}))
    monkeypatch.setattr(_ldg, "_admin_call", rec)
    monkeypatch.setattr(typer, "confirm", lambda *_a, **_k: False)
    assert cli.main(["ledger", "tombstone", _L]) == 130
    assert rec.calls == []
