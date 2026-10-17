"""批 F2 验收剧本 · daemon 的 `behavior.json` 轨道 + `flash_lag_s` + Hub 快照 + MQ-A74。

F2.2：① 删状态重跑逐字一致（增量 == 全量）② 墓碑集合变 ⇒ 丢状态全量重算 ④ 零 LLM；
F2.3：lag 按补遗 G7（无新轮 = 0）、越界 warning、`/admin/status` 与 `bladex status` 可见；
F2.6：MQ-A74 同 agent 换 project 重报一次，同 project 不刷屏。
F2.4（仪器自身守卫）已拆至 `test_probe_behavior_coverage.py`——2026-10-15 混装分家：
探针用例独立成文件，本文件的生产用例随公开树发。
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta

import structlog.testing
from bladex_core.flash import read_behavior_status
from bladex_proxy.flash_daemon import FlashDaemon

NOW = datetime(2026, 9, 22, 12, 0, tzinfo=UTC)


def _row(agent: str, sess: str, minutes: int, *, pid: str = "", pname: str = "",
         aux: bool = False, seq: int = 1) -> tuple[str, dict]:
    ts = NOW - timedelta(minutes=minutes)
    key = f"local/{agent}/{sess}/{int(ts.timestamp() * 1000)}-{seq:08d}"
    return key, {"ts": ts.strftime("%Y-%m-%dT%H:%M:%S.%fZ"), "project_id": pid,
                 "project_name": pname, "project_root": "", "auxiliary": aux, "subagent": False}


ROWS_A = [_row("codex", "fp:a", 100, pid="p1", pname="BladeX"), _row("codex", "fp:a", 90),
          _row("dsh", "fp:d", 80, seq=2)]
ROWS_B = [_row("codex", "fp:a", 30, seq=3), _row("hermes:default", "fp:h", 20, seq=4)]


class _Src:
    def __init__(self, rows, th="t0"):
        self.rows, self.th, self.calls = list(rows), th, 0

    def __call__(self) -> dict:
        self.calls += 1
        return {"rows": list(self.rows), "tombstone_hash": self.th, "ledger_events": [],
                "tombstoned_ledgers": {}, "tool_stats": {}, "habits": [], "hard_rules": ["MUST x"],
                "eligible": None}


def _daemon(tmp_path, src, *, root=None) -> FlashDaemon:
    d = FlashDaemon(root=str(root or tmp_path), principal="local",
                    ledger_source=lambda: ({}, {}), emit_admin_event=lambda t, p: None,
                    behavior_source=src, behavior_every_s=0.0, tz_offset_s=8 * 3600)
    d.behavior_clock = lambda: NOW
    return d


def _read(d: FlashDaemon) -> str:
    return open(d.behavior_path(), encoding="utf-8").read()


class TestIncrementalEqualsFull:
    def test_1_incremental_then_state_delete_rerun_identical(self, tmp_path):
        src = _Src(ROWS_A)
        d = _daemon(tmp_path, src)
        assert d.run_once()["written"] == 1
        src.rows = ROWS_A + ROWS_B
        d.push()
        d.run_once()
        incremental = _read(d)
        assert d._bh_status["turns_folded"] == 2 and d._bh_status["turns_total"] == 5
        # 删状态重跑：新进程、无 `.flash_state.json`
        os.unlink(os.path.join(str(tmp_path), ".flash_state.json"))
        d2 = _daemon(tmp_path, _Src(ROWS_A + ROWS_B))
        d2.push()
        d2.run_once()
        assert _read(d2) == incremental, "增量 == 全量（逐字）"
        assert d2._bh_status["turns_folded"] == 5

    def test_1b_state_survives_restart(self, tmp_path):
        d = _daemon(tmp_path, _Src(ROWS_A))
        d.run_once()
        d3 = _daemon(tmp_path, _Src(ROWS_A))
        assert d3._bh_keys == d._bh_keys and len(d3._bh_keys) == 3
        d3.push()
        d3.run_once()
        assert d3._bh_status["turns_folded"] == 0

    def test_2_tombstone_hash_change_drops_state_equals_full(self, tmp_path):
        src = _Src(ROWS_A + ROWS_B, th="t0")
        d = _daemon(tmp_path, src)
        d.run_once()
        # 墓碑掉一轮：快照少一行、指纹变
        src.rows = ROWS_A[:2] + ROWS_B
        src.th = "t1"
        d.push()
        with structlog.testing.capture_logs() as cap:
            d.run_once()
        assert any(e["event"] == "flash_behavior_state_reset" for e in cap)
        after = _read(d)
        assert d._bh_status["turns_total"] == 4
        d2 = _daemon(tmp_path / "fresh", _Src(ROWS_A[:2] + ROWS_B, th="t1"))
        d2.push()
        d2.run_once()
        assert _read(d2) == after, "墓碑后读数逐字等于删状态重跑"
        data = json.loads(after)
        assert {a["base"] for a in data["agents"]} == {"codex", "hermes"}, "被墓碑的 dsh 不再出现"

    def test_4_zero_llm_no_router_symbol_in_track(self):
        import inspect

        import bladex_proxy.flash_daemon as fd
        src = inspect.getsource(fd.hub_behavior_source) + inspect.getsource(
            fd.FlashDaemon._materialize_behavior)
        code = "\n".join(ln.split("#")[0] for ln in src.splitlines())
        for sym in ("router_sdk", "acompletion", "litellm",
                    "self._summarize", "summarize_project"):
            assert sym not in code, f"画像轨道不许出现 {sym}（零 LLM）"


class TestLag:
    def test_lag_zero_when_no_new_rows_and_grows_with_hub(self, tmp_path):
        src = _Src(ROWS_A)
        d = _daemon(tmp_path, src)
        d.run_once()
        assert d._bh_status["flash_lag_s"] == 0          # 首轮无上次 watermark
        d.push()
        d.run_once()
        assert d._bh_status["flash_lag_s"] == 0 and d._bh_status["turns_folded"] == 0, "无新轮 = 0"
        src.rows = ROWS_A + [_row("codex", "fp:a", 70, seq=9)]   # Hub 前进 10 分钟
        d.push()
        d.run_once()
        assert d._bh_status["flash_lag_s"] == 600
        assert d._bh_status["lag_exceeded"] is True
        st = read_behavior_status(str(tmp_path))
        assert st["flash_lag_s"] == 600 and st["lag_bound_s"] == 360

    def test_lag_over_bound_logs_warning_and_status_line_shows_it(self, tmp_path, monkeypatch):
        src = _Src(ROWS_A)
        d = _daemon(tmp_path, src)
        d.run_once()
        src.rows = ROWS_A + [_row("codex", "fp:a", 60, seq=9)]
        d.push()
        with structlog.testing.capture_logs() as cap:
            d.run_once()
        warn = [e for e in cap if e["event"] == "flash_lag_exceeded"]
        assert warn and warn[0]["log_level"] == "warning" and warn[0]["flash_lag_s"] == 1200
        from bladex_proxy.cli.lifecycle import _flash_lag_part
        part = _flash_lag_part(str(tmp_path))
        assert part.startswith("lag 1200s") and "over 360s bound" in part

    def test_status_line_says_projected_and_na_without_pass(self, tmp_path, monkeypatch):
        from bladex_core import flash
        from bladex_proxy.cli import _flash_status_line
        monkeypatch.setenv("BLADEX_FLASH_PATH", str(tmp_path))
        base = flash.scope_dir(str(tmp_path), "local")
        flash.write_if_changed(f"{base}/ledgers/0f/ldg-1.md", "a")
        line = _flash_status_line()
        assert "1 ledgers projected" in line
        assert "lag n/a" in line, "没读数不显示成 0（原则 13 层 4）"

    def test_admin_status_exposes_flash_section(self, tmp_path, monkeypatch):
        import tempfile

        from bladex_proxy.config import ProxyConfig
        from bladex_proxy.server import create_app
        from fastapi.testclient import TestClient
        monkeypatch.setenv("BLADEX_FLASH_PATH", str(tmp_path))
        d = _daemon(tmp_path, _Src(ROWS_A))
        d.run_once()
        tmpdir = tempfile.mkdtemp()
        app = create_app(ProxyConfig(upstream_model="openai/test", upstream_api_key="sk-fake",
                                     rocksdb_path=f"{tmpdir}/rocksdb", index_path=f"{tmpdir}/index"))
        with TestClient(app) as client:
            body = client.get("/admin/status").json()
        assert body["flash"]["available"] is True
        assert body["flash"]["flash_lag_s"] == 0 and body["flash"]["lag_bound_s"] == 360
        assert "ledgers_projected" in body["flash"]


class TestHubSnapshot:
    def test_snapshot_rows_tombstones_events_one_pass(self, tmp_path):
        from bladex_proxy.ledger_events import load_ledger_events, tombstoned_ledgers
        from bladex_proxy.models import AdminEventType, Identity, TombstoneTargetType, Turn
        from bladex_proxy.storage.memory_hub import MemoryHub
        hub = MemoryHub(str(tmp_path / "hub"))
        hub.open()
        try:
            for i, (agent, pid) in enumerate((("codex", "p1"), ("codex", "p1"), ("dsh", ""))):
                ident = Identity(user_id="u", agent_id=agent, session_id="fp:s", api_key_id="k",
                                 project_id=pid, project_name="BladeX" if pid else "")
                hub.put(f"local/{agent}/fp:s/{1000 + i}-1", Turn(identity=ident, auxiliary=(i == 1)))
            hub.append_admin_event(AdminEventType.LEDGER_CREATE, "",
                                   ledger={"ledger_id": "ldg-x", "last_writer": "codex"})
            snap = hub.scan_meta_snapshot()
            assert len(snap.rows) == 3 and snap.tombstoned == frozenset()
            row = dict(snap.rows)["local/codex/fp:s/1001-1"]
            assert row["project_id"] == "p1" and row["auxiliary"] is True and row["ts"]
            h0 = snap.tombstone_hash()
            hub.append_tombstone(TombstoneTargetType.TURN, "local/dsh/fp:s/1002-1")
            snap2 = hub.scan_meta_snapshot()
            assert len(snap2.rows) == 2 and snap2.tombstone_hash() != h0
            assert "local/dsh/fp:s/1002-1" in snap2.tombstoned
            # 快照冒充 hub 喂账本装载：与 hub 直读同一结果
            assert [e["type"] for e in load_ledger_events(snap2)] == ["ledger_create"]
            assert load_ledger_events(snap2) == load_ledger_events(hub)
            assert tombstoned_ledgers(snap2) == {}
        finally:
            hub.close()


class TestOnboardingA74:
    def test_same_agent_new_project_reports_again_same_project_once(self):
        from bladex_proxy.onboarding import note_onboarding, pending, reset_seen
        reset_seen()
        kw = dict(agent_source="fingerprint", auxiliary=False, cwd_found=True,
                  toolface_injected=True)
        with structlog.testing.capture_logs() as cap:
            assert note_onboarding(agent_id="dsh", project_id="p-old", project_source="git-remote", **kw)
            assert not note_onboarding(agent_id="dsh", project_id="p-old", project_source="git-remote", **kw)
            assert not pending(agent_id="dsh", auxiliary=False, project_id="p-old")
            assert pending(agent_id="dsh", auxiliary=False, project_id="p-new")
            assert note_onboarding(agent_id="dsh", project_id="p-new", project_source="git-remote", **kw)
            assert not note_onboarding(agent_id="dsh", project_id="p-new", project_source="git-remote", **kw)
        rows = [e for e in cap if e["event"] == "agent_onboarding_check"]
        assert [(r["project_id"], r["project_source"]) for r in rows] == [
            ("p-old", "git-remote"), ("p-new", "git-remote")], "换 project 的来源档在日志里可读"
        reset_seen()
