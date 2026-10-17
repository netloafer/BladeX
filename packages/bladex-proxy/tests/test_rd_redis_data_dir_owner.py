"""批 RD（MQ-A90）：同一个 data/ 上跑了两个 redis-server ⇒ 关机时过期 manifest 落盘。

事故（2026-09-29，同形态第二次）：批 F3 隔离实例配 `redis://127.0.0.1:1/0` 当"假 Redis"，
懒恢复对生产 data/ 拉起了 `--port 1`，与 6379 上的真实例共用目录一周；redis 8.8 在
SIGTERM 时重写 manifest，幽灵晚 0.4 s 退出，把 seq 100 写回盘（盘上只有 110）。

这组钉住：
  ① 两条拉起路径在 data/ 已有主人时都不拉第二个（异端口 = 事故形态，同端口也不拉）；
  ② 无主时照常拉起（阴性对照），且日志首行记下来源；
  ③ `bladex repair-aof` 只向前改、先举证，任何一条不成立就拒绝，原 manifest 不动；
  ④ doctor 点名目录上多出来的实例；
  ⑤ e2e 第 2 步不再因 /admin/turns 慢于 3 s 假红（MQ-A91）。
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import types
from pathlib import Path

import pytest
from bladex_proxy.storage import redis_launch as rl
from bladex_proxy.storage.redis_launch import RedisOwner

_GHOST = RedisOwner(pid=98573, port=1, source="cwd")
_LIVE = RedisOwner(pid=6836, port=6379, source="pidfile")


@pytest.fixture()
def _cwd(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    return tmp_path


# ── 命令行 ────────────────────────────────────────────────────────────────


def test_command_has_absolute_dir_and_pidfile_inside_it(_cwd: Path):
    cmd = rl.redis_server_command(6379, "/abs/redis.log")
    d = cmd[cmd.index("--dir") + 1]
    assert os.path.isabs(d) and Path(d) == (_cwd / "data").resolve()
    pidfile = cmd[cmd.index("--pidfile") + 1]
    assert Path(pidfile) == (_cwd / "data").resolve() / rl.REDIS_PIDFILE_NAME


# ── 归属探测 ──────────────────────────────────────────────────────────────


def test_cwd_owner_parsing_matches_only_this_dir(_cwd: Path, monkeypatch):
    target = (_cwd / "data").resolve()
    lsof_out = f"p123\nfcwd\nn{target}\np456\nfcwd\nn/somewhere/else\n"
    monkeypatch.setattr(rl.shutil, "which", lambda n: "/usr/sbin/lsof" if n == "lsof" else None)
    monkeypatch.setattr(rl, "_run_quiet", lambda cmd, timeout=3.0: lsof_out if cmd[0] == "lsof" else "")
    assert rl._cwd_owners(target) == [123]


def test_pidfile_owner_detected_and_stale_pidfile_ignored(_cwd: Path, monkeypatch):
    monkeypatch.setattr(rl, "_cwd_owners", lambda _d: [])
    (_cwd / "data" / rl.REDIS_PIDFILE_NAME).write_text("4242\n")
    monkeypatch.setattr(rl, "_redis_port_of", lambda _p: 6379)

    monkeypatch.setattr(rl, "_is_redis_pid", lambda p: p == 4242)
    assert rl.redis_servers_on_data_dir() == [RedisOwner(4242, 6379, "pidfile")]

    # 阴性对照：pid 已不是 redis（重启后被复用 / 进程已退）⇒ 不算主人
    monkeypatch.setattr(rl, "_is_redis_pid", lambda _p: False)
    assert rl.redis_servers_on_data_dir() == []


def test_redis_port_read_from_process_title(monkeypatch):
    monkeypatch.setattr(rl, "_run_quiet", lambda cmd, timeout=3.0: "redis-server *:1\n")
    assert rl._redis_port_of(98573) == 1
    monkeypatch.setattr(rl, "_run_quiet", lambda cmd, timeout=3.0: "python3 x.py\n")
    assert rl._redis_port_of(98573) is None


# ── 拒绝判定 ──────────────────────────────────────────────────────────────


def test_refusal_on_other_port_names_the_stray_and_how_to_stop_it(_cwd: Path, monkeypatch):
    """事故形态：6379 要起，目录上已有 port=1 的幽灵。"""
    monkeypatch.setattr(rl, "redis_servers_on_data_dir", lambda _d="data": [_GHOST])
    msg = rl.launch_refusal(6379)
    assert msg is not None
    assert "98573" in msg and "port 1" in msg
    assert "redis-cli -p 1 shutdown" in msg
    assert "MQ-A90" in msg


def test_refusal_on_same_port_too(_cwd: Path, monkeypatch):
    """同端口答不上（忙 / 沙盒拦了连接）：再拉一个只会绑端口失败，照样不拉。"""
    monkeypatch.setattr(rl, "redis_servers_on_data_dir", lambda _d="data": [_LIVE])
    msg = rl.launch_refusal(6379)
    assert msg is not None and "redis-cli -p 6379 ping" in msg


def test_no_refusal_when_dir_is_unowned(_cwd: Path, monkeypatch):
    monkeypatch.setattr(rl, "redis_servers_on_data_dir", lambda _d="data": [])
    assert rl.launch_refusal(6379) is None


def test_launch_record_is_first_line_of_the_log(_cwd: Path, monkeypatch):
    monkeypatch.setattr(rl, "redis_servers_on_data_dir", lambda _d="data": [])
    log = _cwd / "logs" / "redis-x.log"
    log.parent.mkdir()
    rl.write_launch_record(str(log), "lazy-recovery", 1)
    first = log.read_text().splitlines()[0]
    assert first.startswith("# bladex-launch source=lazy-recovery")
    assert "port=1" in first and f"launcher_pid={os.getpid()}" in first


# ── CLI 拉起路径（bladex start）────────────────────────────────────────────


def _cli_env(monkeypatch):
    from bladex_proxy.cli import lifecycle
    monkeypatch.setattr(lifecycle.shutil, "which", lambda _n: "/usr/bin/redis-server")
    monkeypatch.setattr(lifecycle, "_redis_ping", lambda _u: False)
    monkeypatch.setattr(lifecycle.time, "sleep", lambda _s: None)
    calls: list[list[str]] = []

    def _fake_run(cmd, **_kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")

    monkeypatch.setattr(lifecycle.subprocess, "run", _fake_run)
    return lifecycle, calls


def test_cli_start_refuses_when_data_dir_is_owned(_cwd: Path, monkeypatch, capsys):
    lifecycle, calls = _cli_env(monkeypatch)
    monkeypatch.setattr(lifecycle, "launch_refusal",
                        lambda port: f"redis-server PID 98573 (port 1) already uses data (asked {port})")
    assert lifecycle._ensure_redis("redis://127.0.0.1:6379/0") is False
    assert calls == [], "目录已有主人时不许再拉 redis-server"
    out = capsys.readouterr().out
    assert "not started" in out and "98573" in out


def test_cli_start_launches_and_records_source_when_unowned(_cwd: Path, monkeypatch, capsys):
    """阴性对照：无主 ⇒ 照常拉起，日志首行记 source=cli-start。"""
    lifecycle, calls = _cli_env(monkeypatch)
    monkeypatch.setattr(lifecycle, "launch_refusal", lambda _port: None)
    monkeypatch.setattr(rl, "redis_servers_on_data_dir", lambda _d="data": [])
    lifecycle._ensure_redis("redis://127.0.0.1:6379/0")
    assert len(calls) == 1 and calls[0][0] == "redis-server"
    logfile = calls[0][calls[0].index("--logfile") + 1]
    assert Path(logfile).read_text().startswith("# bladex-launch source=cli-start")


def test_cli_failure_hint_points_at_repair_not_blind_dump_fallback(_cwd: Path):
    from bladex_proxy.cli import lifecycle
    joined = " ".join(lifecycle._AOF_REPAIR_HINT)
    assert "bladex repair-aof" in joined and "--apply" in joined
    assert "Do not fall back to data/dump.rdb" in joined


# ── 懒恢复路径（ADR-0017）──────────────────────────────────────────────────


def _recover(monkeypatch, refusal: str | None) -> list:
    from bladex_proxy.config import ProxyConfig
    from bladex_proxy.server import orchestration as orch
    from bladex_proxy.storage.pipeline_redis import PipelineRedis

    async def _fail_connect(self):
        raise ConnectionError("refused")

    monkeypatch.setattr(PipelineRedis, "connect", _fail_connect)
    monkeypatch.setattr(orch, "_last_recover_attempt", 0.0)
    monkeypatch.setattr(rl, "launch_refusal", lambda _port, _d="data": refusal)
    monkeypatch.setattr(rl, "redis_servers_on_data_dir", lambda _d="data": [])
    popened: list = []

    def _popen(cmd, **_kw):
        popened.append(cmd)
        raise FileNotFoundError("redis-server")  # 让恢复在拉起之后立即收尾

    monkeypatch.setattr(orch.subprocess, "Popen", _popen)
    monkeypatch.setenv("BLADEX_REDIS_URL", "redis://127.0.0.1:1/0")  # 批 F3 的"假 Redis"
    cfg = ProxyConfig()
    request = types.SimpleNamespace(app=types.SimpleNamespace(state=types.SimpleNamespace(config=cfg)))
    assert asyncio.run(orch._try_recover_redis(request)) is None
    return popened


def test_lazy_recovery_refuses_when_data_dir_is_owned(_cwd: Path, monkeypatch):
    """事故路径本身：配了 port 1 的隔离实例，目录上已有 6379 的真实例 ⇒ 不拉。"""
    popened = _recover(monkeypatch, "redis-server PID 6836 (port 6379) already uses data")
    assert popened == []


def test_lazy_recovery_still_starts_when_unowned(_cwd: Path, monkeypatch):
    """阴性对照：ADR-0017 的本职（Redis 掉线后自动拉起）不能被这道闸误伤。"""
    popened = _recover(monkeypatch, None)
    assert len(popened) == 1 and popened[0][:3] == ["redis-server", "--port", "1"]
    logfile = popened[0][popened[0].index("--logfile") + 1]
    assert Path(logfile).read_text().startswith("# bladex-launch source=lazy-recovery")


# ── bladex repair-aof ─────────────────────────────────────────────────────

_PING = b"*1\r\n$4\r\nPING\r\n"


def _aofdir(root: Path, manifest: str, files: dict[str, bytes]) -> Path:
    d = root / "data" / "appendonlydir"
    d.mkdir(parents=True, exist_ok=True)
    (d / "appendonly.aof.manifest").write_text(manifest)
    for name, body in files.items():
        (d / name).write_bytes(body)
    return d


_STALE = ("file appendonly.aof.100.base.rdb seq 100 type b\n"
          "file appendonly.aof.100.incr.aof seq 100 type i startoffset 51353254 endoffset 51354231\n")
_NEW_FILES = {"appendonly.aof.110.base.aof": _PING, "appendonly.aof.110.incr.aof": _PING}


@pytest.fixture()
def _unowned(monkeypatch):
    monkeypatch.setattr(rl, "redis_servers_on_data_dir", lambda _d="data": [])


@pytest.mark.skipif(shutil.which("redis-check-aof") is None, reason="needs redis-check-aof")
def test_repair_incident_shape_forward_and_keeps_old_manifest(_cwd: Path, _unowned):
    """09-29 原形：manifest 指 seq 100，盘上只有 110 ⇒ 预演通过、应用后可被 redis-check-aof 接受。"""
    d = _aofdir(_cwd, _STALE, _NEW_FILES)
    plan = rl.plan_manifest_repair()
    assert plan.ok, plan.reason
    assert plan.new_manifest == ("file appendonly.aof.110.base.aof seq 110 type b\n"
                                 "file appendonly.aof.110.incr.aof seq 110 type i\n")
    assert (d / "appendonly.aof.manifest").read_text() == _STALE, "预演不许动盘"

    kept = rl.apply_manifest_repair(plan)
    assert kept.read_text() == _STALE and kept.parent == (_cwd / "data").resolve()
    assert (d / "appendonly.aof.manifest").read_text() == plan.new_manifest
    assert rl.aof_manifest_problem() is None
    proc = subprocess.run(["redis-check-aof", str(d / "appendonly.aof.manifest")],
                          capture_output=True, text=True, check=False)
    assert proc.returncode == 0


@pytest.mark.skipif(shutil.which("redis-check-aof") is None, reason="needs redis-check-aof")
def test_repair_refuses_when_checker_rejects(_cwd: Path, _unowned):
    d = _aofdir(_cwd, _STALE, {"appendonly.aof.110.base.aof": _PING,
                               "appendonly.aof.110.incr.aof": b"*1\r\n$4\r\nPI"})
    plan = rl.plan_manifest_repair()
    assert not plan.ok and "redis-check-aof" in plan.reason
    assert (d / "appendonly.aof.manifest").read_text() == _STALE


def test_repair_refuses_when_a_server_owns_the_dir(_cwd: Path, monkeypatch):
    """占着目录的实例会在退出时再写一次 manifest —— 先停它。"""
    _aofdir(_cwd, _STALE, _NEW_FILES)
    monkeypatch.setattr(rl, "redis_servers_on_data_dir", lambda _d="data": [_GHOST])
    plan = rl.plan_manifest_repair()
    assert not plan.ok and "98573" in plan.reason


def test_repair_refuses_to_move_backwards(_cwd: Path, _unowned):
    _aofdir(_cwd, _STALE, {"appendonly.aof.90.base.aof": _PING, "appendonly.aof.90.incr.aof": _PING})
    plan = rl.plan_manifest_repair()
    assert not plan.ok and "backwards" in plan.reason


def test_repair_refuses_partial_missing(_cwd: Path, _unowned):
    _aofdir(_cwd, _STALE, {"appendonly.aof.100.base.rdb": b"x", "appendonly.aof.110.incr.aof": _PING})
    plan = rl.plan_manifest_repair()
    assert not plan.ok and "only part" in plan.reason


def test_repair_refuses_ambiguous_files(_cwd: Path, _unowned):
    _aofdir(_cwd, _STALE, {**_NEW_FILES, "appendonly.aof.111.base.aof": _PING})
    plan = rl.plan_manifest_repair()
    assert not plan.ok and "exactly one base" in plan.reason


def test_repair_nothing_to_do_when_consistent(_cwd: Path, _unowned):
    _aofdir(_cwd, "file appendonly.aof.110.base.aof seq 110 type b\n", _NEW_FILES)
    plan = rl.plan_manifest_repair()
    assert not plan.ok and "nothing to repair" in plan.reason
    with pytest.raises(ValueError):
        rl.apply_manifest_repair(plan)


def test_repair_cli_preview_changes_nothing_and_names_apply(_cwd: Path, monkeypatch, capsys):
    from bladex_proxy.cli import lifecycle
    d = _aofdir(_cwd, _STALE, _NEW_FILES)
    monkeypatch.setattr(lifecycle, "plan_manifest_repair",
                        lambda: rl.ManifestRepairPlan(True, "ok", "file a seq 110 type b\n", ["c1"]))
    assert lifecycle.repair_aof(apply=False) == 0
    out = capsys.readouterr().out
    assert "bladex repair-aof --apply" in out
    assert (d / "appendonly.aof.manifest").read_text() == _STALE


# ── doctor ────────────────────────────────────────────────────────────────


def _doctor(monkeypatch, capsys, owners: list[RedisOwner]) -> str:
    from bladex_proxy import cli
    from bladex_proxy.cli import lifecycle
    for k in list(os.environ):  # 与 test_cli_lifecycle._tmp_cwd 同款：外部 BLADEX_* 不许泄进来
        if k.startswith("BLADEX_"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("BLADEX_REDIS_URL", "redis://127.0.0.1:6379/0")
    monkeypatch.setattr(lifecycle, "_redis_ping", lambda _u: False)
    monkeypatch.setattr(lifecycle, "redis_servers_on_data_dir", lambda _d="data": owners)
    cli.main(["doctor"])
    return capsys.readouterr().out


def test_doctor_flags_a_stray_server_on_data_dir(_cwd: Path, monkeypatch, capsys, isolated_home):
    out = _doctor(monkeypatch, capsys, [_LIVE, _GHOST])
    assert "✗ Redis servers on data/" in out
    assert "redis-cli -p 1 shutdown" in out


def test_doctor_passes_with_the_one_configured_server(_cwd: Path, monkeypatch, capsys, isolated_home):
    out = _doctor(monkeypatch, capsys, [_LIVE])
    assert "✓ Redis servers on data/" in out


# ── e2e 第 2 步（MQ-A91）──────────────────────────────────────────────────


def test_e2e_step2_tolerates_a_slow_turn_listing(monkeypatch, capsys):
    """/admin/turns 在 17.7k 轮上要 ≈4 s：3 s 超时会读成 None ⇒ 假报"not stored"。"""
    from bladex_proxy.cli import lifecycle
    monkeypatch.setattr(lifecycle.secrets, "token_hex", lambda _n: "deadbeef")
    monkeypatch.setattr(lifecycle.time, "sleep", lambda _s: None)
    monkeypatch.setattr(lifecycle, "_http_request", lambda *a, **k: (200, {}))
    token = "bxdeadbeef"

    def _http_json(url, key="", timeout=3.0):
        if "/admin/turns?limit" in url:
            return {"turns": [{"key": "k1"}]} if timeout >= 5.0 else None  # 慢端点
        if "/admin/turns/" in url:
            return {"request_messages": [{"content": token}]}
        if "/admin/facts" in url:
            return {"facts": [{"id": "fact_x", "content": token}]}
        return None

    monkeypatch.setattr(lifecycle, "_http_json", _http_json)
    cfg = types.SimpleNamespace(host="127.0.0.1", port=1, upstream_model="m")
    assert lifecycle._run_e2e_check(cfg, timeout_s=1.0) == 0
    assert "step 2 stored" in capsys.readouterr().out
