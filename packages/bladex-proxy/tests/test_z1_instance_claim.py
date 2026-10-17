"""批 Z1 · Z1.3 —— MQ-A38：同机第二实例偷 live 轮次 ⇒ 启动即检测、响亮拒绝。

判据（提示词 §1 Z1.3）：
  ① 不同 `BLADEX_HOME` 的实例连到同一条流 ⇒ 拒绝，报错点名流 / 组 / 对方 home / 三旋钮；
  ② 🔴 阴性对照：同一实例重启不误拒；三旋钮隔离的两个实例正常共存；
  ③ stale 登记（上一个进程崩了）不许永久挡住启动；
  ④ 拒绝路径不吞输出：lifespan 原样上抛（`bladex start` 打日志尾部）。
真 Redis 上的两实例实测见结算稿 `report-batch-Z1-20260923.md` §3。
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
from pathlib import Path

import pytest
from bladex_proxy.storage import instance_claim as ic

_PKG = Path(__file__).resolve().parents[1] / "bladex_proxy"


class _FakeRedis:
    """redis.asyncio 的最小替身：set(nx/ex) / get / delete / xinfo_consumers。"""

    def __init__(self) -> None:
        self.kv: dict[str, bytes] = {}
        self.consumers: dict[tuple[str, str], list[dict]] = {}

    async def set(self, key, value, nx=False, ex=None):
        if nx and key in self.kv:
            return None
        self.kv[key] = value.encode() if isinstance(value, str) else value
        return True

    async def get(self, key):
        return self.kv.get(key)

    async def delete(self, key):
        self.kv.pop(key, None)

    async def xinfo_consumers(self, stream, group):
        if (stream, group) not in self.consumers:
            raise RuntimeError("NOGROUP")
        return self.consumers[(stream, group)]


def _claim(r, home, *, stream="bladex:turns", group="bladex-workers", port=38080):
    return ic.InstanceClaim(r, stream=stream, group=group, consumer="worker-1",
                            url="redis://127.0.0.1:6379/0", port=port, home=home)


def _run(coro):
    return asyncio.run(coro)


def test_second_home_on_same_stream_is_refused_and_named():
    """判据①：另一个 home、pid 活着 ⇒ 拒绝；报错点名流、两边的组、对方 home、三旋钮。"""
    r = _FakeRedis()
    _run(_claim(r, "/live/home").acquire())
    with pytest.raises(ic.InstanceConflictError) as ei:
        _run(_claim(r, "/tmp/second", group="bladex-workers", port=18989).acquire())
    msg = str(ei.value)
    assert "'bladex:turns'" in msg
    assert "BLADEX_HOME=/live/home" in msg and "BLADEX_HOME=/tmp/second" in msg
    assert "consumer group 'bladex-workers'" in msg
    for knob in ("BLADEX_REDIS_STREAM", "BLADEX_REDIS_GROUP", "BLADEX_REDIS_CONSUMER",
                 "BLADEX_REDIS_URL"):
        assert knob in msg
    # live 的登记没被覆盖
    assert json.loads(r.kv["bladex:instance:bladex:turns"])["home"] == "/live/home"


def test_same_stream_different_group_is_also_refused():
    """异组共用 stream 也是串流（两组各收对方的轮次）⇒ 同样拒绝。"""
    r = _FakeRedis()
    _run(_claim(r, "/live/home").acquire())
    with pytest.raises(ic.InstanceConflictError):
        _run(_claim(r, "/tmp/second", group="other-group").acquire())


def test_negative_control_same_home_restart_is_not_refused():
    """判据② 🔴：同一 home 重启（旧登记还在 TTL 内）⇒ 接管，不拒。"""
    r = _FakeRedis()
    _run(_claim(r, "/live/home").acquire())
    _run(_claim(r, "/live/home").acquire())  # 不抛


def test_negative_control_isolated_namespace_coexists():
    """判据② 🔴：第二实例用三旋钮换了 stream ⇒ 两个登记并存。"""
    r = _FakeRedis()
    _run(_claim(r, "/live/home").acquire())
    _run(_claim(r, "/tmp/second", stream="bladex:turns:demo", group="bladex-workers-demo").acquire())
    assert set(r.kv) == {"bladex:instance:bladex:turns", "bladex:instance:bladex:turns:demo"}


def test_stale_claim_of_dead_process_on_same_host_is_taken_over():
    """判据③：登记方同机且 pid 已死 ⇒ 接管，不等 TTL。"""
    r = _FakeRedis()
    r.kv["bladex:instance:bladex:turns"] = json.dumps({
        "home": "/crashed/home", "pid": 2 ** 22 + 4321, "host": socket.gethostname(),
        "port": 38080, "group": "bladex-workers"}).encode()
    _run(_claim(r, "/tmp/second").acquire())
    assert json.loads(r.kv["bladex:instance:bladex:turns"])["home"] == "/tmp/second"


def test_foreign_host_claim_is_refused_until_ttl():
    """异机登记无法判活 ⇒ 拒绝（登记带 TTL，对方真没了最多挡 CLAIM_TTL_S 秒）。"""
    r = _FakeRedis()
    r.kv["bladex:instance:bladex:turns"] = json.dumps({
        "home": "/other/box", "pid": 1, "host": "not-this-host", "port": 38080,
        "group": "bladex-workers"}).encode()
    with pytest.raises(ic.InstanceConflictError, match=f"expires within {ic.CLAIM_TTL_S}s"):
        _run(_claim(r, "/tmp/second").acquire())


def test_release_only_deletes_own_claim():
    r = _FakeRedis()
    mine = _claim(r, "/live/home")
    _run(mine.acquire())
    _run(mine.release())
    assert r.kv == {}
    other = _claim(r, "/other")
    _run(other.acquire())
    _run(mine.release())  # 不是自己的：不删
    assert "bladex:instance:bladex:turns" in r.kv


class _ClockRedis(_FakeRedis):
    """消费者 idle 随假时钟走：`alive` 的每 2 秒被重置一次（PipelineWorker 的阻塞读周期）。"""

    def __init__(self, consumers: dict[str, tuple[bool, int]]) -> None:
        super().__init__()
        self.t_ms = 0
        self._c = consumers  # name -> (alive, idle_at_t0_ms)

    async def xinfo_consumers(self, stream, group):
        out = []
        for name, (alive, idle0) in self._c.items():
            idle = (idle0 + self.t_ms) % 2000 if alive else idle0 + self.t_ms
            out.append({b"name": name.encode(), b"idle": idle})
        return out

    async def sleep(self, s):
        self.t_ms += int(s * 1000)


def _clock_claim(r, home):
    return ic.InstanceClaim(r, stream="bladex:turns", group="bladex-workers",
                            consumer="worker-1", url="redis://127.0.0.1:6379/0",
                            port=18989, home=home, sleep=r.sleep)


def test_live_unregistered_consumer_is_refused(capsys):
    """升级窗口：live 是 0.3.0 之前的版本（不登记）、组里有活着的消费者 ⇒ 拒绝并点名消费者。

    （本批实测事故的形态：裸 uvicorn 没加载 .env 连到 live Redis，旧版这里只告警 ⇒ 照常开读。）
    """
    r = _ClockRedis({"worker-1": (True, 700)})
    with pytest.raises(ic.InstanceConflictError) as ei:
        _run(_clock_claim(r, "/tmp/second").acquire())
    msg = str(ei.value)
    assert "['worker-1']" in msg and "'bladex-workers'" in msg and "older than 0.3.0" in msg
    assert "BLADEX_REDIS_STREAM" in msg
    assert r.kv == {}, "拒绝时不许留下登记"
    assert "instance_claim_unregistered_consumer" in capsys.readouterr().out


def test_negative_control_just_stopped_consumer_is_not_refused():
    """🔴 阴性对照：同一实例刚 stop（消费者 idle 1.5s、之后单调增长）⇒ 等一个窗口后放行。"""
    r = _ClockRedis({"worker-1": (False, 1500)})
    _run(_clock_claim(r, "/live/home").acquire())
    assert r.t_ms == int(ic.LIVENESS_WINDOW_S * 1000), "应当只多等一个判活窗口"
    assert json.loads(r.kv["bladex:instance:bladex:turns"])["home"] == "/live/home"


def test_long_idle_consumer_needs_no_probe():
    """消费者早就不动了（idle 远大于窗口）⇒ 不等、直接登记。"""
    r = _ClockRedis({"worker-1": (False, 600_000)})
    _run(_clock_claim(r, "/live/home").acquire())
    assert r.t_ms == 0


def test_redis_error_during_check_does_not_block_startup():
    class _Broken(_FakeRedis):
        async def set(self, *a, **k):
            raise ConnectionError("boom")

    _run(_claim(_Broken(), "/tmp/x").acquire())  # 不抛


def test_heartbeat_reports_lost_claim(capsys):
    r = _FakeRedis()
    mine = _claim(r, "/live/home")
    _run(mine.acquire())
    r.kv["bladex:instance:bladex:turns"] = json.dumps({"home": "/intruder", "pid": 1}).encode()
    assert _run(mine.heartbeat_once()) is False
    assert "instance_claim_lost" in capsys.readouterr().out


# ── 接线守卫（判据④：冲突不许被 Redis 降级路径吞掉）───────────────────────────


def test_lifespan_reraises_conflict_outside_the_redis_degrade_try():
    src = (_PKG / "server" / "app_factory.py").read_text(encoding="utf-8")
    body = src.split("async def lifespan(", 1)[1]
    connect_try = body.index("await pipeline.connect()")
    degrade = body.index('logger.warning("pipeline_redis_unavailable"')
    acquire = body.index("await claim.acquire()")
    worker = body.index("PipelineWorker(app.state.pipeline, hub)")
    assert connect_try < degrade < acquire < worker, "登记必须在降级 try 之后、worker 开读之前"
    seg = body[acquire:worker]
    assert "except InstanceConflictError:" in seg and "raise" in seg


def test_lazy_recovery_claims_before_starting_worker():
    src = (_PKG / "server" / "orchestration.py").read_text(encoding="utf-8")
    body = src.split("async def _try_recover_redis(", 1)[1].split("\nasync def ", 1)[0]
    assert body.index("await claim.acquire()") < body.index("PipelineWorker(pipeline, hub)")
    assert "pipeline_redis_recover_refused" in body


def test_lifespan_refuses_start_end_to_end(monkeypatch, tmp_path):
    """真跑 lifespan：Redis 替身里已有另一个 home 的活登记 ⇒ lifespan 抛冲突、不进 yield。"""
    from bladex_proxy.server import app_factory as af

    r = _FakeRedis()
    closed: list = []

    async def _connect(self):
        self._redis = r

    async def _close(self):
        closed.append(True)

    monkeypatch.setattr(af.PipelineRedis, "connect", _connect)
    monkeypatch.setattr(af.PipelineRedis, "close", _close)
    monkeypatch.setenv("BLADEX_HOME", str(tmp_path))
    monkeypatch.setenv("BLADEX_ROCKSDB_PATH", str(tmp_path / "hub"))
    monkeypatch.setenv("BLADEX_OVERFLOW_DIR", str(tmp_path / "overflow"))
    app = af.create_app()
    stream = app.state.config.redis_stream  # conftest 给每次测试隔离的 stream 名
    r.kv[ic.claim_key(stream)] = json.dumps({
        "home": "/live/home", "pid": os.getpid(), "host": socket.gethostname(),
        "port": 38080, "group": "bladex-workers"}).encode()

    async def _go():
        async with af.lifespan(app):
            pytest.fail("lifespan 不该走到 yield")

    with pytest.raises(ic.InstanceConflictError, match="BLADEX_HOME=/live/home"):
        asyncio.run(_go())
    assert closed, "拒绝前要关掉 Redis 连接"


# ── 拒绝路径的输出能上终端（判据④）：两处既有缺陷 ───────────────────────────────


def test_pid_alive_reaps_exited_child_instead_of_reporting_zombie_alive():
    """CLI 自己 Popen 出来、已退出却未 wait 的子进程是僵尸，`kill(pid, 0)` 对它照样成功。

    修前：一个 1 秒内就因实例冲突退出的 proxy，CLI 等满 30 秒报"活着但 /ready 不答"。
    """
    import subprocess
    import sys
    import time

    from bladex_proxy import cli

    proc = subprocess.Popen([sys.executable, "-c", "pass"])  # noqa: S603
    deadline = time.monotonic() + 10
    while cli._pid_alive(proc.pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not cli._pid_alive(proc.pid), "已退出的子进程被判成活着（僵尸）"
    proc.wait(timeout=5)  # 被抢先回收后 Popen 仍能正常收尾（ECHILD 由 subprocess 自己兜）
    assert cli._pid_alive(os.getpid()), "活着的非子进程（自己）仍须判活"


def test_log_error_tail_starts_at_traceback_not_uvicorn_summary(tmp_path):
    """uvicorn 在 lifespan traceback 之后补一行 `Application startup failed. Exiting.`；
    尾部必须从 traceback 起，否则冲突原因整段被挡在终端外。"""
    from bladex_proxy.cli import lifecycle as lc
    log = tmp_path / "proxy.log"
    log.write_text("\n".join([
        "INFO:     Waiting for application startup.",
        "ERROR:    Traceback (most recent call last):",
        '  File "x.py", line 1, in lifespan',
        "bladex_proxy.storage.instance_claim.InstanceConflictError: refusing to start: ...",
        "",
        "ERROR:    Application startup failed. Exiting.",
    ]), encoding="utf-8")
    tail = lc._log_error_tail(str(log))
    assert "InstanceConflictError: refusing to start" in tail
    assert tail.lstrip().startswith("ERROR:    Traceback")


def test_start_proxy_fast_failure_prints_error_tail():
    src = (_PKG / "cli" / "lifecycle.py").read_text(encoding="utf-8")
    body = src.split("def _start_proxy(", 1)[1].split("\ndef ", 1)[0]
    assert "_log_error_tail(logfile)" in body
