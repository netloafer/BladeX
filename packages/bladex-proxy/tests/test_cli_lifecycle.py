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
    from bladex_proxy.cli import lifecycle as _lc   # F0.1 拆包：消费方在 cli/lifecycle.py
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
    from bladex_proxy.cli import lifecycle as _lc   # F0.1 拆包：消费方在 cli/lifecycle.py
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
