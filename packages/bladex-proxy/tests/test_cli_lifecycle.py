"""Beta T10 CLI 骨架验收剧本：--version / init / config check / doctor / status /
生命周期 helpers。start/stop 的真实进程拉起不在单测里跑（用 fake Popen/pid 检查），
sync 命令族回归由 test_sync_cli.py 钉住（typer 化后 main(argv) 签名不变）。
"""

from __future__ import annotations

import os
import socket
from pathlib import Path

import pytest
from bladex_proxy import cli


@pytest.fixture(autouse=True)
def _tmp_cwd(tmp_path: Path, monkeypatch, isolated_home):
    """干净工作目录（init/config check 都按相对 cwd 找 config/）。

    autouse：cli.main → _load_env_file 会把 cwd 下 config/.env 灌进 os.environ
    且不回收——任何一个用例在仓库根跑都会吸入真实配置污染同进程后续测试
    （2026-08-03 踩坑：test_version_flag 未隔离 → admin 套件集体 401）。

    `isolated_home`（2026-08-06）：根发现补了 `~/.bladex` 兜底，开发机上真有那个
    目录时 `cli.main()` 会跑去加载它——同一个污染换了个入口。见 conftest。
    """
    monkeypatch.chdir(tmp_path)
    # 环境卫生：不让外部 BLADEX_* 泄进剧本
    for k in list(os.environ):
        if k.startswith("BLADEX_"):
            monkeypatch.delenv(k, raising=False)
    yield tmp_path
    # _load_env_file 在测试中新设的 BLADEX_*（monkeypatch 未跟踪）也要清
    for k in list(os.environ):
        if k.startswith("BLADEX_"):
            del os.environ[k]


# ── --version ─────────────────────────────────────────────────────


def test_version_flag(capsys):
    rc = cli.main(["--version"])
    assert rc == 0
    out = capsys.readouterr().out
    from bladex_proxy import __version__
    assert f"bladex {__version__}" in out


# ── init ──────────────────────────────────────────────────────────


def test_init_yes_generates_env_and_routing(_tmp_cwd: Path):
    rc = cli.main(["init", "--yes", "--upstream-model", "openai/test-model",
                   "--api-base", "https://api.example.com/v1",
                   "--api-key", "sk-test", "--port", "39999"])
    assert rc == 0
    env = (_tmp_cwd / "config/.env").read_text()
    assert "BLADEX_UPSTREAM_MODEL=openai/test-model" in env
    assert "BLADEX_UPSTREAM_API_BASE=https://api.example.com/v1" in env
    assert "BLADEX_UPSTREAM_API_KEY=sk-test" in env
    assert "BLADEX_PORT=39999" in env
    routing = (_tmp_cwd / "config/routing.toml").read_text()
    assert 'name = "openai/test-model"' in routing
    assert "api_key_env" in routing  # 密钥纪律：toml 只引 env 名


def test_init_refuses_overwrite_without_force(_tmp_cwd: Path):
    assert cli.main(["init", "--yes"]) == 0
    assert cli.main(["init", "--yes"]) == 1  # 已存在 → 拒绝
    assert cli.main(["init", "--yes", "--force"]) == 0


# ── config check ──────────────────────────────────────────────────


def test_config_check_ok_on_generated_config(_tmp_cwd: Path, capsys, monkeypatch):
    cli.main(["init", "--yes", "--upstream-model", "openai/test-model",
              "--api-key", "sk-test"])
    # ADR-0027 §3.1：上游 key 是 config check 的必检项，所以"通过"这条剧本必须真的有 key。
    monkeypatch.setenv("BLADEX_UPSTREAM_API_KEY", "sk-test")
    rc = cli.main(["config", "check"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "Configuration check passed" in out


def test_config_check_fails_on_empty_upstream_key(_tmp_cwd: Path, capsys, monkeypatch):
    """空 key 曾是"配置检查通过"（rc=0）—— 新用户第一条真实请求必然失败却毫无提示。"""
    cli.main(["init", "--yes", "--upstream-model", "openai/test-model"])
    monkeypatch.setenv("BLADEX_UPSTREAM_API_KEY", "")
    rc = cli.main(["config", "check"])
    out = capsys.readouterr().out
    assert rc == 1, out
    assert "BLADEX_UPSTREAM_API_KEY" in out


def test_config_check_reports_toml_error_with_location(_tmp_cwd: Path, capsys):
    cli.main(["init", "--yes"])
    # 制造 TOML 语法错误
    (_tmp_cwd / "config/routing.toml").write_text("[[models]\nname = broken\n")
    rc = cli.main(["config", "check"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "routing.toml" in out
    assert "line" in out  # tomllib 报错自带行号
    assert "fix:" in out


def test_config_check_auth_enabled_without_keys(_tmp_cwd: Path, capsys, monkeypatch):
    cli.main(["init", "--yes"])
    monkeypatch.setenv("BLADEX_AUTH_ENABLED", "true")
    monkeypatch.setenv("BLADEX_CLIENT_KEYS", "")
    rc = cli.main(["config", "check"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "BLADEX_CLIENT_KEYS" in out


def test_config_check_missing_env_file(_tmp_cwd: Path, capsys):
    rc = cli.main(["config", "check"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "config/.env" in out
    assert "bladex init" in out


# ── doctor ────────────────────────────────────────────────────────


def test_doctor_reports_failures_without_crashing(_tmp_cwd: Path, capsys, monkeypatch):
    """无 redis / 无 env 的干净环境：doctor 给 ✗ + 修法、退出码 1、不崩。"""
    monkeypatch.setenv("BLADEX_REDIS_URL", "redis://127.0.0.1:1/0")  # 必不可达
    rc = cli.main(["doctor"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "✗" in out
    assert "Redis" in out
    assert "fix:" in out


# ── 生命周期 helpers ──────────────────────────────────────────────


def test_redis_hostport_parse():
    assert cli._redis_hostport("redis://127.0.0.1:6379/0") == ("127.0.0.1", 6379)
    assert cli._redis_hostport("redis://myhost:7000/2") == ("myhost", 7000)
    assert cli._redis_hostport("garbage") == ("127.0.0.1", 6379)


# ── redis 启动诊断（2026-09-19 事故）─────────────────────────────
#
# 事故：主动重启系统后 redis 8.8.0 落盘了一份过期 AOF manifest（指向 seq 97，
# 盘上只有 seq 98）⇒ redis 加载 AOF 失败即退出。而启动器 daemonize 不给 logfile
# 又吞掉 stderr ⇒ 全部输出只有一句 "failed to start"，真因不可判定。
# 下面四条钉的是"失败说得出原因"这件事本身。


def _write_aof(data_dir: Path, manifest: str, files: tuple[str, ...]) -> None:
    d = data_dir / "appendonlydir"
    d.mkdir(parents=True, exist_ok=True)
    (d / "appendonly.aof.manifest").write_text(manifest)
    for f in files:
        (d / f).write_bytes(b"x")


def test_aof_manifest_problem_is_silent_when_consistent(_tmp_cwd: Path):
    from bladex_proxy.storage.redis_launch import aof_manifest_problem

    assert aof_manifest_problem("data") is None  # 全新部署：还没有 manifest

    _write_aof(
        _tmp_cwd / "data",
        "file appendonly.aof.98.base.rdb seq 98 type b\n"
        "file appendonly.aof.98.incr.aof seq 98 type i startoffset 0 endoffset 12\n",
        ("appendonly.aof.98.base.rdb", "appendonly.aof.98.incr.aof"),
    )
    assert aof_manifest_problem("data") is None


def test_aof_manifest_problem_names_the_missing_files(_tmp_cwd: Path):
    """事故现场的形状：manifest 说 97，盘上只有 98。"""
    from bladex_proxy.storage.redis_launch import aof_manifest_problem

    _write_aof(
        _tmp_cwd / "data",
        "file appendonly.aof.97.base.rdb seq 97 type b\n"
        "file appendonly.aof.97.incr.aof seq 97 type i startoffset 557354 endoffset 50388438\n",
        ("appendonly.aof.98.base.rdb", "appendonly.aof.98.incr.aof"),
    )
    problem = aof_manifest_problem("data")
    assert problem is not None
    # 缺的两个都要点名——只说"manifest 有问题"等于没说
    assert "appendonly.aof.97.base.rdb" in problem
    assert "appendonly.aof.97.incr.aof" in problem
    # 还要说现在盘上有谁，否则用户不知道该往哪改
    assert "appendonly.aof.98.base.rdb" in problem


def test_redis_logfile_is_absolute(_tmp_cwd: Path):
    """🔴 相对 logfile 会落进 data/：redis 处理 `--dir data/` 时当场 chdir。"""
    from bladex_proxy.storage.redis_launch import redis_logfile_path, redis_server_command

    logfile = redis_logfile_path("logs")
    assert os.path.isabs(logfile)
    assert Path(logfile).parent == (_tmp_cwd / "logs").resolve()

    cmd = redis_server_command(6379, logfile)
    assert "--logfile" in cmd
    assert cmd[cmd.index("--logfile") + 1] == logfile
    assert os.path.isabs(cmd[cmd.index("--logfile") + 1])


def test_ensure_redis_failure_prints_why(_tmp_cwd: Path, capsys, monkeypatch):
    """起不来时必须说出原因：manifest 对账 + redis 日志尾部，缺一不可。"""
    from bladex_proxy.cli import lifecycle

    _write_aof(
        _tmp_cwd / "data",
        "file appendonly.aof.97.base.rdb seq 97 type b\n",
        ("appendonly.aof.98.base.rdb",),
    )

    def _fake_run(cmd, **_kw):
        import subprocess as _sp
        if cmd[0] != "redis-server":  # 批 RD：拉起前的归属探测（ps / lsof）⇒ 目录无主
            return _sp.CompletedProcess(cmd, 0, stdout="", stderr="")
        logfile = cmd[cmd.index("--logfile") + 1]
        Path(logfile).write_text(
            "# Redis version=8.8.0, bits=64, commit=00000000\n"
            "# Can't open the append-only file appendonly.aof.97.base.rdb: "
            "No such file or directory\n"
        )
        import subprocess as _sp
        return _sp.CompletedProcess(cmd, 1, stdout="", stderr="")

    monkeypatch.setattr(lifecycle.shutil, "which", lambda _n: "/usr/bin/redis-server")
    monkeypatch.setattr(lifecycle, "_redis_ping", lambda _u: False)
    monkeypatch.setattr(lifecycle.subprocess, "run", _fake_run)
    monkeypatch.setattr(lifecycle.time, "sleep", lambda _s: None)

    assert lifecycle._ensure_redis("redis://127.0.0.1:6379/0") is False
    out = capsys.readouterr().out
    assert "failed to start" in out
    assert "appendonly.aof.97.base.rdb" in out          # manifest 对账点了名
    assert "No such file or directory" in out            # redis 自己那句话也在
    assert "dump.rdb" in out                             # 给了兜底走法


def test_pidfile_roundtrip(tmp_path: Path):
    pidfile = tmp_path / "x.pid"
    pidfile.write_text(str(os.getpid()))
    assert cli._read_pidfile(str(pidfile)) == os.getpid()
    pidfile.write_text("999999999")  # 不存在的 pid
    assert cli._read_pidfile(str(pidfile)) is None
    pidfile.write_text("not-a-pid")
    assert cli._read_pidfile(str(pidfile)) is None
    assert cli._read_pidfile(str(tmp_path / "missing.pid")) is None


def test_first_client_key(monkeypatch):
    monkeypatch.setenv("BLADEX_CLIENT_KEYS", "sk-aaa||hermes, sk-bbb||codex")
    assert cli._first_client_key() == "sk-aaa"
    monkeypatch.setenv("BLADEX_CLIENT_KEYS", "")
    assert cli._first_client_key() == ""


def test_start_requires_env_file(_tmp_cwd: Path, capsys):
    rc = cli.main(["start"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "config/.env" in out


def test_status_when_nothing_running(_tmp_cwd: Path, capsys, monkeypatch):
    """全下线状态：Redis ✗ + proxy ✗，退出码 1（探不到 /health）。"""
    monkeypatch.setenv("BLADEX_REDIS_URL", "redis://127.0.0.1:1/0")
    # 找一个必然空闲的端口给 proxy 探测
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    free_port = s.getsockname()[1]
    s.close()
    monkeypatch.setenv("BLADEX_PORT", str(free_port))
    rc = cli.main(["status"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "not running" in out


def test_stop_when_nothing_running(_tmp_cwd: Path, capsys):
    rc = cli.main(["stop"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "not running" in out
    assert "Redis" in out  # 明示不动 Redis


# ── Queue 排队面（2026-08-06，doctor --e2e 卡 step 3 事故驱动）──────────────
#
# 事故形状：探针轮 180s 内没蒸出 fact，真因是 972 轮积压排在它前面，而 status
# 只在 Memory Index 那行末尾挂了个 lag，Pipeline 侧的积压根本没有出口。这组钉住
# 四件事：两段队列都要出现、有积压但没人消化时必须点名 consolidator、清空时说
# "caught up"、落盘轮次不许被吞掉。

def test_queue_line_reports_both_hops():
    line = cli._format_queue(
        {"pipeline": {"backlog": 7}, "index": {"lag_turns": 972}}, True)
    assert "-> Hub 7 waiting" in line
    assert "-> Index 972 waiting" in line


def test_queue_names_the_consolidator_when_backlog_has_nobody_working_it():
    """有积压 + consolidator 没跑 = 记忆静默失效，这是最该被看见的一种。"""
    line = cli._format_queue({"pipeline": {"backlog": 0}, "index": {"lag_turns": 972}}, False)
    assert "consolidator is not running" in line
    line_running = cli._format_queue({"pipeline": {"backlog": 0}, "index": {"lag_turns": 972}}, True)
    assert "catching up" in line_running
    assert "not running" not in line_running


def test_queue_says_caught_up_when_both_empty():
    line = cli._format_queue({"pipeline": {"backlog": 0}, "index": {"lag_turns": 0}}, True)
    assert "caught up" in line


def test_queue_surfaces_spilled_turns():
    """落盘轮次在回灌前不产生记忆，不能只躺在 /admin/status 里。"""
    line = cli._format_queue(
        {"pipeline": {"backlog": 0, "overflow_count": 3, "spill_count": 2},
         "index": {"lag_turns": 0}}, True)
    assert "5 spilled to disk" in line


def test_queue_degrades_when_admin_payload_is_missing():
    """proxy 在跑但 /admin/status 读不到时，status 不许崩，也不许谎报 0。"""
    line = cli._format_queue(None, True)
    assert "?" in line
    assert "caught up" not in line


# ── 孤儿单列（2026-08-06 第二次事故：487 一动不动）─────────────────────────
#
# backlog 里混着"谁都不会再碰"的残留时，原来的措辞会安慰人说 worker 正在排空它，
# 而且只要 Index 侧有一条 lag，连这句话都被 catching up 顶掉——积压永远不降却
# 没有任何一句话提到它。这组钉住：孤儿不计入排队数、自己有一句话、任何分支都吞不掉它。

def test_queue_excludes_stale_entries_from_the_waiting_count():
    line = cli._format_queue(
        {"pipeline": {"backlog": 487, "orphan": 485}, "index": {"lag_turns": 0}}, True)
    assert "-> Hub 2 waiting" in line
    assert "485 stale" in line
    assert "queue flush" in line


def test_stale_hint_survives_the_index_lag_branch():
    """Index 侧有 lag 时也必须看得见孤儿——两件事不共用一句措辞。"""
    line = cli._format_queue(
        {"pipeline": {"backlog": 487, "orphan": 485}, "index": {"lag_turns": 2}}, True)
    assert "catching up" in line
    assert "485 stale" in line


def test_all_stale_is_not_reported_as_caught_up():
    line = cli._format_queue(
        {"pipeline": {"backlog": 485, "orphan": 485}, "index": {"lag_turns": 0}}, True)
    assert "caught up" not in line
    assert "485 stale" in line


# ── consolidator 轮内进度（2026-08-06：一轮 30 条跑半小时，lag 纹丝不动）──────
#
# `-> Index N waiting` 只在每轮提交时跳一次，轮内看着和进程卡死一模一样。这组钉住：
# 有心跳就说清在干什么、没心跳就一个字都不加、不新鲜时不许宣称在跑、出错不许伪装成进展。

def test_activity_shows_progress_inside_a_batch():
    suffix = cli._consolidator_activity({"consolidator": {
        "fresh": True, "state": "running", "phase": "distill",
        "batch": 3, "done": 71, "total": 305}})
    assert "batch 3" in suffix and "71/305" in suffix


def test_activity_is_silent_without_a_heartbeat():
    """老版本 consolidator 不写心跳——"读不到"不等于"没在跑"，不许瞎猜。"""
    assert cli._consolidator_activity({"consolidator": None}) == ""
    assert cli._consolidator_activity({}) == ""
    assert cli._consolidator_activity(None) == ""


def test_activity_reports_age_instead_of_claiming_it_runs():
    """心跳不新鲜可能是进程死了，也可能卡在一次长上游调用里——只报事实。"""
    suffix = cli._consolidator_activity({"consolidator": {
        "fresh": False, "last_seen_s": 640.0, "state": "running",
        "phase": "distill", "done": 71, "total": 305}})
    assert "640" in suffix
    assert "distilling" not in suffix


def test_activity_surfaces_a_failed_batch():
    suffix = cli._consolidator_activity({"consolidator": {
        "fresh": True, "state": "error", "phase": "batch_failed",
        "error": "upstream 429"}})
    assert "failed" in suffix and "429" in suffix


# ── 守护进程蒸馏并发透传（2026-08-22 事故）───────────────────────────────────
#
# 防的是什么：`_start_consolidator()` 起子进程时**只传 `--interval`**，而
# `consolidator.py` 的 argparse 默认是 `0 → 1` 串行，于是 `bladex start` /
# `bladex consolidator start` 拉起的守护进程并发恒为 1；而 `bladex sync` 那条路
# 默认 4。同一个参数两条路两个默认值，谁也说不出生产在跑哪个。
# 实测后果：mean 23.2s/次 × 8–9 次调用/轮串行 = 5–26 轮/小时，白天流入高于消费，
# Memory Index 积压 720 轮持续不降（新记忆写不进去 = 召不回）。
#
# 同型缺陷 2026-08-13 已经兑现过一次（bench 全程串行、排空超时、整跑作废），
# 当时只改了 help 文案没改实现 —— 所以这里钉的是**命令行本身**，不是文档。


class _FakePopen:
    """记录子进程 argv 的最小替身（不真的起进程）。"""
    calls: list[list[str]] = []

    def __init__(self, argv, **kw):
        type(self).calls.append(list(argv))
        self.pid = 424242


@pytest.fixture
def _fake_spawn(monkeypatch):
    _FakePopen.calls = []
    monkeypatch.setattr(cli.subprocess, "Popen", _FakePopen)
    from bladex_proxy.cli import lifecycle as _lc  # F0.1 拆包：消费方在 cli/lifecycle.py
    monkeypatch.setattr(_lc, "_consolidator_pid", lambda: None)  # 假装没在跑
    monkeypatch.setattr(_lc, "_pid_alive", lambda pid: True)
    return _FakePopen


def _spawn_argv() -> list[str]:
    assert len(_FakePopen.calls) == 1, f"期望正好起一次子进程，实际 {len(_FakePopen.calls)}"
    return _FakePopen.calls[0]


def test_start_consolidator_passes_concurrency(_tmp_cwd: Path, _fake_spawn):
    """🔴 核心断言：命令行里必须有 --concurrency，且不是串行的 1。"""
    assert cli._start_consolidator(60) is True
    argv = _spawn_argv()
    assert "--concurrency" in argv, (
        "守护进程 argv 里没有 --concurrency —— consolidator.py 的 argparse "
        "会回落串行，这正是 2026-08-22 积压 720 轮的那个缺陷"
    )
    assert argv[argv.index("--concurrency") + 1] == "4"


def test_start_consolidator_reads_env_override(_tmp_cwd: Path, _fake_spawn, monkeypatch):
    """.env / 环境里配了就听它的（上游限流时调小的通道）。"""
    monkeypatch.setenv("BLADEX_DISTILL_CONCURRENCY", "2")
    assert cli._start_consolidator(60) is True
    argv = _spawn_argv()
    assert argv[argv.index("--concurrency") + 1] == "2"


def test_start_consolidator_honours_explicit_serial(_tmp_cwd: Path, _fake_spawn):
    """阴性对照：显式要串行就得真串行，回滚通道不能被默认值吃掉。"""
    assert cli._start_consolidator(60, 1) is True
    argv = _spawn_argv()
    assert argv[argv.index("--concurrency") + 1] == "1"


def test_consolidator_command_flag_reaches_the_spawn(_tmp_cwd: Path, _fake_spawn, monkeypatch):
    """`bladex consolidator start --concurrency 8` 端到端到达 argv。"""
    from bladex_proxy.cli import lifecycle as _lc  # F0.1 拆包：消费方在 cli/lifecycle.py
    monkeypatch.setattr(_lc, "_require_config_root", lambda: None)
    rc = cli.main(["consolidator", "start", "--concurrency", "8"])
    assert rc == 0
    argv = _spawn_argv()
    assert argv[argv.index("--concurrency") + 1] == "8"


def test_sync_and_daemon_share_one_default(_tmp_cwd: Path, monkeypatch):
    """两条路同源：`bladex sync` 与守护进程读的必须是同一个默认值。

    缺陷的本质不是"某条路慢"，是**同一个参数有两个默认值**。
    """
    from bladex_core.flags import MEMORY_NUMERIC_DEFAULTS
    monkeypatch.delenv("BLADEX_DISTILL_CONCURRENCY", raising=False)
    assert cli._resolve_distill_concurrency() == int(
        MEMORY_NUMERIC_DEFAULTS["BLADEX_DISTILL_CONCURRENCY"])


def test_consolidator_module_falls_back_to_env(monkeypatch):
    """`python -m bladex_proxy.consolidator` 直接跑（不经 CLI）也不许静默串行。

    这是第二道保险：CLI 现在显式传值了，但守护进程也可能被手工/脚本拉起。
    """
    from bladex_proxy.consolidator import resolve_concurrency
    monkeypatch.delenv("BLADEX_DISTILL_CONCURRENCY", raising=False)
    assert resolve_concurrency(0) == 4      # 省略 → flags 默认
    assert resolve_concurrency(-1) == 4     # 负数同理
    assert resolve_concurrency(1) == 1      # 阴性对照：显式串行不被吃掉
    assert resolve_concurrency(8) == 8
    monkeypatch.setenv("BLADEX_DISTILL_CONCURRENCY", "3")
    assert resolve_concurrency(0) == 3
    assert resolve_concurrency(6) == 6      # 显式仍压过 env


def test_no_stray_concurrency_defaults_in_source():
    """守卫：`BLADEX_DISTILL_CONCURRENCY` 不许再有第二个默认值。

    缺陷本体是"同一个参数三条路三个默认值"（sync 4 / 守护进程 1 /
    rebuild_from_hub 1）。修完这三处还不够——下一个人再写一个
    `os.environ.get("BLADEX_DISTILL_CONCURRENCY", "…")` 就复发了。
    所以这里禁的是**就地带默认值地读这个 env**，读默认值只能走 flags。
    """
    import re
    from pathlib import Path
    root = Path(cli.__file__).resolve().parents[3]
    offenders = []
    for pkg in ("bladex-core/bladex_core", "bladex-proxy/bladex_proxy"):
        for py in (root / "packages" / pkg).rglob("*.py"):
            for i, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
                # 就地读 env 且带回落默认值 → 又一个真相源
                if re.search(r'environ\.get\(\s*["\']BLADEX_DISTILL_CONCURRENCY["\']\s*,', line):
                    offenders.append(f"{py.relative_to(root)}:{i}")
    assert not offenders, (
        "这些地方就地给 BLADEX_DISTILL_CONCURRENCY 定了默认值，"
        "应改读 bladex_core.flags.flag_number：\n  " + "\n  ".join(offenders)
    )


# ── G18.2：backend=api 与 embedding 模块冲突从静默改报错（MQ-W9 断点②）──────


def test_embed_module_skipped_for_api_backend(monkeypatch, capsys):
    """api ∧ MODULE_EMBED=1：不起模块，且 stderr 一行说清为什么（不是静默）。"""
    from bladex_proxy.cli import lifecycle
    monkeypatch.setenv("BLADEX_EMBED_BACKEND", "api")
    monkeypatch.setenv("BLADEX_MODULE_EMBED", "1")
    assert lifecycle._embed_wanted() is False
    err = capsys.readouterr().err
    assert "embed_module_skipped" in err and "backend=api" in err


def test_embed_api_without_module_flag_stays_quiet(monkeypatch, capsys):
    """api 档但模块本来就关：没有冲突，不打提示（提示只指真实冲突）。"""
    from bladex_proxy.cli import lifecycle
    monkeypatch.setenv("BLADEX_EMBED_BACKEND", "api")
    monkeypatch.delenv("BLADEX_MODULE_EMBED", raising=False)
    assert lifecycle._embed_wanted() is False
    assert "embed_module_skipped" not in capsys.readouterr().err


def test_embed_wanted_unchanged_for_ipc_and_module_flag(monkeypatch):
    """回归：ipc 档与显式模块开关的既有行为一字不变。"""
    from bladex_proxy.cli import lifecycle
    monkeypatch.delenv("BLADEX_MODULE_EMBED", raising=False)
    monkeypatch.setenv("BLADEX_EMBED_BACKEND", "ipc")
    assert lifecycle._embed_wanted() is True
    monkeypatch.setenv("BLADEX_EMBED_BACKEND", "local")
    assert lifecycle._embed_wanted() is False
    monkeypatch.setenv("BLADEX_MODULE_EMBED", "1")
    assert lifecycle._embed_wanted() is True


def test_embed_wanted_reads_routing_toml_backend(_tmp_cwd: Path, capsys, monkeypatch):
    """MQ-W11：README（G18.1）主推的 toml 配法也必须被认——判据=生效配置。

    旧代码裸读 env：toml 配 backend="api" + 模块开关开 ⇒ `_embed_wanted()` 为 True，
    模块照起、G18.2 的 skip 提示也不打——承诺在文档教的那条路上整个不成立。
    阴性对照：判据改回裸读 env ⇒ 本条必红（上面三条 env 剧本不许跟着红）。
    """
    from bladex_proxy.cli import lifecycle
    (_tmp_cwd / "config").mkdir()
    (_tmp_cwd / "config" / "routing.toml").write_text(
        '[embedding]\nbackend = "api"\nmodel = "openai/text-embedding-3-small"\n',
        encoding="utf-8")
    monkeypatch.setenv("BLADEX_MODULE_EMBED", "1")
    assert lifecycle._embed_wanted() is False
    err = capsys.readouterr().err
    assert "embed_module_skipped" in err and "backend=api" in err
    # 同一判据的另一面：toml 配 ipc（模块开关关）也该起模块——不起该起的进程是静默降级
    monkeypatch.delenv("BLADEX_MODULE_EMBED", raising=False)
    (_tmp_cwd / "config" / "routing.toml").write_text(
        '[embedding]\nbackend = "ipc"\n', encoding="utf-8")
    assert lifecycle._embed_wanted() is True


def test_status_api_backend_prints_model_and_dim(_tmp_cwd: Path, capsys, monkeypatch):
    """G18.3 判据：api 档下 status 打模型名 + profile 的 dim，不再劝人开模块开关。"""
    monkeypatch.setenv("BLADEX_REDIS_URL", "redis://127.0.0.1:1/0")
    monkeypatch.setenv("BLADEX_EMBED_BACKEND", "api")
    monkeypatch.setenv("BLADEX_EMBED_MODEL", "openai/text-embedding-3-small")
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    monkeypatch.setenv("BLADEX_PORT", str(s.getsockname()[1]))
    s.close()
    cli.main(["status"])
    out = capsys.readouterr().out
    assert "api (openai/text-embedding-3-small, dim=1536" in out
    assert "set BLADEX_MODULE_EMBED=1" not in out


# ── G18.4：doctor 认 api 档（MQ-W9 断点④）────────────────────────────────


def _api_doctor_run(tmp_path, monkeypatch, *, model="openai/text-embedding-3-small",
                    api_key="sk-x", key_env="", sidecar="", warmup=False, embed_fn=None,
                    sens_enabled=False):
    """驱动 doctor 的 embedding 段 `_doctor_embedding`，用记录版 check 收结果（不探真网络）。

    cfg 带全 embed_* 字段：MQ-W12 后期望身份复用 `embed_model_identity(cfg)`
    （真解析生效配置），残缺的 SimpleNamespace 会让它解析回 local 默认档。
    MQ-W14 后盖章对账提到了分流之前，所以驱动的是整段而不是 `_doctor_embed_api` 本体
    （五条 G18.4 用例的断言逐字未动）。
    """
    from types import SimpleNamespace

    from bladex_proxy import router_sdk
    from bladex_proxy.cli import lifecycle

    if embed_fn is None:
        def embed_fn(**kwargs):
            return {"data": [{"index": 0, "embedding": [0.0] * 1536}]}
    monkeypatch.setattr(router_sdk, "embedding", embed_fn)
    index_dir = tmp_path / "index"
    index_dir.mkdir(exist_ok=True)
    if sidecar:
        (index_dir / "embed_model_id").write_text(sidecar, encoding="utf-8")
    cfg = SimpleNamespace(
        embed_backend="api", embed_model=model, embed_api_base="", embed_api_key=api_key,
        embed_proxy_url="http://127.0.0.1:38080", embed_proxy_key="",
        embed_query_cache_size=8, fastembed_cache_path=str(tmp_path / "fe"),
        routing_config=SimpleNamespace(
            embedding=SimpleNamespace(api_key_env=key_env),
            sensitivity_config=lambda: SimpleNamespace(enabled=sens_enabled)),
        index_path=str(index_dir))
    calls: list[tuple] = []
    lifecycle._doctor_embedding(cfg, warmup,
                                lambda label, ok, detail="", fix="": calls.append((ok, detail, fix)))
    assert len(calls) == 1
    return calls[0]


def test_doctor_api_fails_when_model_unset(tmp_path, monkeypatch):
    ok, detail, fix = _api_doctor_run(tmp_path, monkeypatch, model="")
    assert ok is False and "no embedding model configured" in detail
    assert "[embedding].model" in fix and "BLADEX_EMBED_MODEL" in fix


def test_doctor_api_fails_when_key_env_declared_but_empty(tmp_path, monkeypatch):
    ok, detail, fix = _api_doctor_run(tmp_path, monkeypatch, api_key="", key_env="MY_EMBED_KEY")
    assert ok is False and "MY_EMBED_KEY" in detail and "resolves to empty" in detail
    assert "MY_EMBED_KEY" in fix


def test_doctor_api_fails_on_stamped_index_mismatch(tmp_path, monkeypatch):
    """探活成功但与已盖章库不同源：复用 model_id 不变量的措辞（reembed_index）。"""
    ok, detail, fix = _api_doctor_run(tmp_path, monkeypatch,
                                      sidecar="local:intfloat/multilingual-e5-large")
    assert ok is False and "embed model mismatch" in detail
    assert "api:openai/text-embedding-3-small" in detail
    assert "reembed_index" in fix


def test_doctor_api_success_prints_dim_and_rtt(tmp_path, monkeypatch, capsys):
    ok, detail, _fix = _api_doctor_run(tmp_path, monkeypatch, warmup=True)
    assert ok is True
    assert "embed_api_probe_ok" in detail and "dim=1536" in detail and "rtt_ms=" in detail
    out = capsys.readouterr().out
    assert "warmup" in out and "nothing to download" in out


def test_doctor_api_probe_failure_points_at_config(tmp_path, monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("connection refused")
    ok, _detail, fix = _api_doctor_run(tmp_path, monkeypatch, embed_fn=boom)
    assert ok is False and "[embedding]" in fix


def test_doctor_api_under_sensitivity_accepts_the_runtime_local_stamp(tmp_path, monkeypatch):
    """MQ-W12：敏感层开时运行时被强制回本地，库盖 local 章**本来就对**。

    旧代码手搭 expected=f"api:{model}" ⇒ 对着正确的章报 mismatch，还给出
    `reembed_index.py` 这种破坏性指令（api 档下=全部记忆内容外发+按量计费）。
    阴性对照：把 expected 改回手搭串 ⇒ 本条必红。
    """
    from bladex_proxy.embedding import DEFAULT_LOCAL_MODEL
    ok, detail, _fix = _api_doctor_run(tmp_path, monkeypatch, sens_enabled=True,
                                       sidecar=f"local:{DEFAULT_LOCAL_MODEL}")
    assert ok is True
    assert "mismatch" not in detail


def test_doctor_dispatch_sensitivity_forced_local_skips_probe_and_mismatch(
        _tmp_cwd: Path, capsys, monkeypatch):
    """MQ-W12 判据①（经 doctor() 本体）：敏感层开 + toml 配 api ⇒
    ① _embed_api_probe 零外呼（敏感层的语义就是不外发，用 mock 计数断言）；
    ② 不报 mismatch（章=运行时真实 local 身份）；③ 一行说清被谁改写。"""
    from bladex_proxy import router_sdk
    from bladex_proxy.config import ProxyConfig
    from bladex_proxy.embedding import embed_model_identity
    monkeypatch.setenv("BLADEX_REDIS_URL", "redis://127.0.0.1:1/0")
    (_tmp_cwd / "config").mkdir()
    (_tmp_cwd / "config" / "routing.toml").write_text(
        '[embedding]\nbackend = "api"\nmodel = "openai/text-embedding-3-small"\n\n'
        '[strategies.sensitivity]\nenabled = true\n', encoding="utf-8")
    probes: list[int] = []
    monkeypatch.setattr(
        router_sdk, "embedding",
        lambda **kw: probes.append(1) or {"data": [{"index": 0, "embedding": [0.0] * 1536}]})
    stamp_dir = Path(ProxyConfig().index_path)
    stamp_dir.mkdir(parents=True, exist_ok=True)
    (stamp_dir / "embed_model_id").write_text(
        embed_model_identity(ProxyConfig(), role="proxy"), encoding="utf-8")
    cli.main(["doctor"])
    out = capsys.readouterr().out
    assert probes == [], "敏感层开着时 doctor 不许外呼 embedding 供应商"
    assert "mismatch" not in out
    assert "sensitivity layer" in out


def test_doctor_wires_api_backend_to_the_api_check(_tmp_cwd: Path, capsys, monkeypatch):
    """MQ-W13：backend=api 必须**经 doctor() 本体**走到 `_doctor_embed_api`。

    五条 G18.4 用例都直接驱动那个函数——把 doctor() 里的 `elif backend == "api"`
    接线整段删掉，51 条照样全绿（复核方实测）。判据=反向变异：删那个 elif ⇒
    本条必红（通用分支打的是 "api backend (remote; no local cache check)"，
    不会有 embed_api_probe_ok）。函数有测试 ≠ 接线有守卫（MQ-A53 同族）。"""
    from bladex_proxy import router_sdk
    monkeypatch.setenv("BLADEX_REDIS_URL", "redis://127.0.0.1:1/0")
    monkeypatch.setenv("BLADEX_EMBED_BACKEND", "api")
    monkeypatch.setenv("BLADEX_EMBED_MODEL", "openai/text-embedding-3-small")
    monkeypatch.setattr(
        router_sdk, "embedding",
        lambda **kw: {"data": [{"index": 0, "embedding": [0.0] * 1536}]})
    cli.main(["doctor"])
    out = capsys.readouterr().out
    assert "embed_api_probe_ok" in out


# ── MQ-W14（批 Q0b）：盖章对账提到分支之外，doctor 与 preflight 同向 ──────────


def _doctor_cfg(tmp_path, **over):
    """与 `_preflight_cfg` 同一份 cfg 再补 doctor 要的 `fastembed_cache_path`——
    判据④要求两个入口吃**同一份**配置。"""
    cfg = _preflight_cfg(tmp_path, **over)
    cfg.fastembed_cache_path = str(tmp_path / "fe")
    return cfg


def _doctor_verdict(cfg):
    """doctor 的 embedding 段对这份配置的结论：(全部 ok?, 各条 detail)。"""
    from bladex_proxy.cli import lifecycle
    calls: list[tuple] = []
    lifecycle._doctor_embedding(cfg, False,
                                lambda label, ok, detail="", fix="": calls.append((ok, detail)))
    return all(ok for ok, _ in calls), [d for _, d in calls]


def test_doctor_sensitivity_api_stale_stamp_reports_mismatch_without_probe(tmp_path, monkeypatch):
    """判据①：敏感层开 + api + 章不一致 ⇒ 报 mismatch（拿**生效**身份 local:... 对），且零探活。

    W12 之前这格是"报一个错的警（expected=api:...）"，W12 之后是"不报警"——两个都错。
    阴性对照：对账挪回分支内 ⇒ 红；expected 换回手搭 f"api:{model}" ⇒ 红（detail 里的 current 不再是 local:）。
    """
    from bladex_proxy import router_sdk
    from bladex_proxy.embedding import DEFAULT_LOCAL_MODEL
    probes: list[int] = []
    monkeypatch.setattr(router_sdk, "embedding",
                        lambda **kw: probes.append(1) or {"data": [{"index": 0, "embedding": [0.0] * 1536}]})
    _write_stamp(tmp_path, "local:intfloat/multilingual-e5-large")
    cfg = _doctor_cfg(tmp_path, embed_backend="api", sens_enabled=True,
                      embed_model="openai/text-embedding-3-small", embed_api_key="sk-x")
    ok, details = _doctor_verdict(cfg)
    assert ok is False
    assert any("embed model mismatch" in d and f"current='local:{DEFAULT_LOCAL_MODEL}'" in d
               for d in details), details
    assert probes == []


def test_doctor_local_stale_stamp_reports_mismatch(tmp_path, monkeypatch):
    """判据②：纯 local + 显式模型 + 章不一致 ⇒ 报（既有缺口：local 分支从来不对账，今天是错的绿）。"""
    _write_stamp(tmp_path, "local:intfloat/multilingual-e5-large")
    cfg = _doctor_cfg(tmp_path, embed_model="BAAI/bge-small-en-v1.5")
    ok, details = _doctor_verdict(cfg)
    assert ok is False
    assert any("embed model mismatch" in d for d in details), details


def test_doctor_matching_or_missing_stamp_stays_green(tmp_path, monkeypatch):
    """判据③（正向）：章一致 / 侧车不存在 ⇒ 不报——新库、首次启动是常态，别修成天天报警。"""
    cfg = _doctor_cfg(tmp_path, embed_model="BAAI/bge-small-en-v1.5")
    ok, details = _doctor_verdict(cfg)                      # 无侧车
    assert ok is True, details
    _write_stamp(tmp_path, "local:BAAI/bge-small-en-v1.5")
    ok, details = _doctor_verdict(cfg)                      # 章一致
    assert ok is True, details


@pytest.mark.parametrize("label, over, stamp", [
    ("local-consistent",   dict(embed_model="BAAI/bge-small-en-v1.5"), "local:BAAI/bge-small-en-v1.5"),
    ("local-stale",        dict(embed_model="BAAI/bge-small-en-v1.5"), "local:intfloat/multilingual-e5-large"),
    ("sens-api-stale",     dict(embed_backend="api", sens_enabled=True,
                                embed_model="openai/text-embedding-3-small", embed_api_key="sk-x"),
                           "local:intfloat/multilingual-e5-large"),
    ("api-stale",          dict(embed_backend="api", embed_model="openai/text-embedding-3-small",
                                embed_api_key="sk-x"), "local:intfloat/multilingual-e5-large"),
    ("proxy-any-stamp",    dict(embed_backend="proxy"), "local:intfloat/multilingual-e5-large"),
])
def test_doctor_and_preflight_agree_on_the_same_config(tmp_path, monkeypatch, label, over, stamp):
    """判据④（本批最值钱的一条）：doctor 与 preflight 在**同一配置**上结论必须同向。

    钉的是"两个入口不许各说各话"，不是某一行代码：MQ-W14 的形态 = doctor 说健康、
    start 说起不来，而 doctor 正是用户拿来诊断 start 的工具（绿灯也不许骗人）。
    proxy 档两边都刻意不作负面判定（身份握手后才知道），也要同向。
    """
    from bladex_proxy.cli import lifecycle
    _mock_probe_ok(monkeypatch)
    _write_stamp(tmp_path, stamp)
    cfg = _doctor_cfg(tmp_path, **over)
    preflight_ok = lifecycle._preflight_embedding(cfg, force=False)
    doctor_ok, details = _doctor_verdict(cfg)
    assert doctor_ok == preflight_ok, (
        f"[{label}] doctor={doctor_ok} preflight={preflight_ok} -- 两个入口各说各话: {details}")


# ── G18.5：start 预检——embedding 配置错误中止启动（MQ-W10）──────────────


def _preflight_cfg(tmp_path, **over):
    from types import SimpleNamespace
    sens = over.pop("sens_enabled", False)
    key_env = over.pop("key_env", "")
    base = dict(
        embed_backend="local", embed_model="", embed_api_base="", embed_api_key="",
        embed_proxy_url="http://127.0.0.1:38080", embed_proxy_key="",
        embed_query_cache_size=8,
        index_path=str(tmp_path / "index"),
        routing_config=SimpleNamespace(
            embedding=SimpleNamespace(api_key_env=key_env),
            sensitivity_config=lambda: SimpleNamespace(enabled=sens)),
    )
    base.update(over)
    return SimpleNamespace(**base)


def _mock_probe_ok(monkeypatch):
    from bladex_proxy import router_sdk
    monkeypatch.setattr(router_sdk, "embedding",
                        lambda **kw: {"data": [{"index": 0, "embedding": [0.0] * 1536}]})


def _write_stamp(tmp_path, identity):
    (tmp_path / "index").mkdir(exist_ok=True)
    (tmp_path / "index" / "embed_model_id").write_text(identity, encoding="utf-8")


def test_preflight_api_model_unset_aborts_even_with_force(tmp_path, capsys):
    """①模型名空：--force 也不放行（这种配置下记忆必死，强启无意义）。"""
    from bladex_proxy.cli import lifecycle
    cfg = _preflight_cfg(tmp_path, embed_backend="api")
    assert lifecycle._preflight_embedding(cfg, force=False) is False
    assert lifecycle._preflight_embedding(cfg, force=True) is False
    out = capsys.readouterr().out
    assert "[embedding].model" in out and "BLADEX_EMBED_MODEL" in out


def test_preflight_api_key_env_empty_aborts_even_with_force(tmp_path, capsys):
    from bladex_proxy.cli import lifecycle
    cfg = _preflight_cfg(tmp_path, embed_backend="api",
                         embed_model="openai/text-embedding-3-small", key_env="MY_EMBED_KEY")
    assert lifecycle._preflight_embedding(cfg, force=False) is False
    assert lifecycle._preflight_embedding(cfg, force=True) is False
    out = capsys.readouterr().out
    assert "MY_EMBED_KEY" in out and "resolves to empty" in out


def test_preflight_engine_mismatch_aborts_with_three_exits(tmp_path, capsys, monkeypatch):
    """③引擎变更：中止 + 三条出路（回退 / reembed+重标定 / --force）逐条在场。"""
    from bladex_proxy.cli import lifecycle
    _mock_probe_ok(monkeypatch)
    _write_stamp(tmp_path, "local:intfloat/multilingual-e5-large")
    cfg = _preflight_cfg(tmp_path, embed_backend="api",
                         embed_model="openai/text-embedding-3-small", embed_api_key="sk-x")
    assert lifecycle._preflight_embedding(cfg, force=False) is False
    out = capsys.readouterr().out
    assert "local:intfloat/multilingual-e5-large" in out
    assert "api:openai/text-embedding-3-small" in out
    assert "revert" in out                      # 出路 1：更正配置
    assert "reembed_index" in out and "re-calibrate" in out   # 出路 2：重建 + 重标定
    assert "--force" in out                     # 出路 3：强启


def test_preflight_engine_mismatch_force_starts_with_warning(tmp_path, capsys, monkeypatch):
    from bladex_proxy.cli import lifecycle
    _mock_probe_ok(monkeypatch)
    _write_stamp(tmp_path, "local:intfloat/multilingual-e5-large")
    cfg = _preflight_cfg(tmp_path, embed_backend="api",
                         embed_model="openai/text-embedding-3-small", embed_api_key="sk-x")
    assert lifecycle._preflight_embedding(cfg, force=True) is True
    out = capsys.readouterr().out
    assert "continuing due to --force" in out and "vector search disabled" in out


def test_preflight_probe_failure_aborts_unless_forced(tmp_path, capsys, monkeypatch):
    from bladex_proxy import router_sdk
    from bladex_proxy.cli import lifecycle

    def boom(**kw):
        raise RuntimeError("connection refused")
    monkeypatch.setattr(router_sdk, "embedding", boom)
    cfg = _preflight_cfg(tmp_path, embed_backend="api",
                         embed_model="openai/text-embedding-3-small", embed_api_key="sk-x")
    assert lifecycle._preflight_embedding(cfg, force=False) is False
    assert lifecycle._preflight_embedding(cfg, force=True) is True
    out = capsys.readouterr().out
    assert "api probe failed" in out and "--force" in out


def test_preflight_api_success_prints_probe_ok(tmp_path, capsys, monkeypatch):
    from bladex_proxy.cli import lifecycle
    _mock_probe_ok(monkeypatch)
    cfg = _preflight_cfg(tmp_path, embed_backend="api",
                         embed_model="openai/text-embedding-3-small", embed_api_key="sk-x")
    assert lifecycle._preflight_embedding(cfg, force=False) is True
    assert "embed_api_probe_ok dim=1536" in capsys.readouterr().out


def test_preflight_local_matching_stamp_passes_without_probe(tmp_path, monkeypatch):
    """local 档 + 盖章一致（未显式配 model 时走库内沿用）：通过且绝不探网。"""
    from bladex_proxy import router_sdk
    from bladex_proxy.cli import lifecycle

    def never(**kw):
        raise AssertionError("local preflight must not hit the embedding API")
    monkeypatch.setattr(router_sdk, "embedding", never)
    _write_stamp(tmp_path, "local:intfloat/multilingual-e5-large")
    cfg = _preflight_cfg(tmp_path)  # backend=local, model 未显式配
    assert lifecycle._preflight_embedding(cfg, force=False) is True


def test_start_aborts_on_preflight_and_launches_nothing(_tmp_cwd: Path, capsys, monkeypatch):
    """全路径：预检失败 ⇒ rc=1 且 redis/embed/consolidator/proxy 一个都没拉起。"""
    from bladex_proxy.cli import lifecycle
    (_tmp_cwd / "config").mkdir()
    (_tmp_cwd / "config" / ".env").write_text("BLADEX_PORT=39999\n", encoding="utf-8")
    monkeypatch.setenv("BLADEX_EMBED_BACKEND", "api")  # model 未配 ⇒ 预检①红
    launched: list[str] = []
    for fn in ("_ensure_redis", "_start_embed", "_start_consolidator",
               "_start_flash", "_start_proxy"):
        monkeypatch.setattr(lifecycle, fn,
                            lambda *a, _n=fn, **k: launched.append(_n) or True)
    rc = cli.main(["start"])
    out = capsys.readouterr().out
    assert rc == 1
    assert launched == []
    assert "[embedding].model" in out and "start aborted" in out
