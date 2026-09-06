"""存储隔离守卫 —— 保证测试流量永远进不了生产 Pipeline/Memory Hub/Memory Index。

背景见根目录 `conftest.py`：2026-07-28 发现 pytest 的 13 条 turn 写进了刚清干净
的生产 Memory Hub。根因是测试覆盖了 `rocksdb_path` 却没覆盖 `redis_stream`，测试 Turn
经生产 Pipeline stream 被运行中的 proxy 消费落库。

隔离本身必须被测——否则哪天 conftest 被改坏、或某个测试自己 setenv 回生产值，
会**静默**失效，而症状（生产库多出几十条垃圾）要等下次巡检才发现。
"""

from __future__ import annotations

import os

import pytest

from conftest import PRODUCTION_DEFAULTS


@pytest.mark.parametrize("env_key,production_value", sorted(PRODUCTION_DEFAULTS.items()))
def test_env_not_pointing_at_production(env_key: str, production_value: str) -> None:
    """五个存储 env 都不得等于生产默认值。"""
    actual = os.environ.get(env_key, "")
    assert actual, f"{env_key} 未设置——conftest.py 的隔离没生效"
    assert actual != production_value, (
        f"{env_key}={actual!r} 指向生产位置。测试流量会污染真实记忆库；"
        f"检查根目录 conftest.py 是否被绕过"
    )


def test_default_config_uses_isolated_storage() -> None:
    """不传参构造的 ProxyConfig 必须落在隔离位置——这是事故的直接形状。

    出事的那批测试正是这样：显式传了 rocksdb_path，却没管 redis_stream。
    """
    from bladex_proxy.config import ProxyConfig

    cfg = ProxyConfig()
    assert cfg.redis_stream.startswith("bladex:test:"), (
        f"redis_stream={cfg.redis_stream!r} —— 测试 Turn 会进生产 Pipeline 流，"
        f"被运行中的 proxy 消费后写进生产 Memory Hub（2026-07-28 事故形状）"
    )
    assert cfg.redis_group != PRODUCTION_DEFAULTS["BLADEX_REDIS_GROUP"]
    assert not cfg.rocksdb_path.startswith("data/")
    assert not cfg.index_path.startswith("data/")
    assert not cfg.overflow_dir.startswith("data/")


def test_isolated_paths_are_writable_tempdirs() -> None:
    """隔离路径必须是可写临时目录（否则测试会以别的方式失败，掩盖真因）。"""
    from bladex_proxy.config import ProxyConfig

    cfg = ProxyConfig()
    for path in (cfg.rocksdb_path, cfg.index_path, cfg.overflow_dir):
        parent = os.path.dirname(path.rstrip("/")) or "."
        assert os.path.isdir(parent), f"{path} 的父目录不存在"
        assert os.access(parent, os.W_OK), f"{path} 的父目录不可写"


def test_stream_and_group_unique_per_run() -> None:
    """stream / group 带 run id —— 并发跑两个 pytest 不会互相串消费。"""
    from bladex_proxy.config import ProxyConfig

    cfg = ProxyConfig()
    run_id = cfg.redis_stream.rsplit(":", 1)[-1]
    assert len(run_id) >= 6
    assert run_id in cfg.redis_group
    assert run_id in cfg.redis_consumer


# ══════════════════════════════════════════════════════════════════════════
# 环境卫生（2026-08-05，C6 实测驱动）
#
# 公开树上 `pytest` 一次 19 条假失败：跑 BladeX 的终端 source 过 config/.env，
# AUTH_ENABLED / CLIENT_KEYS / UPSTREAM_* 压过测试默认值 -> 端点全 401。
# 清洗原先只在 scripts/gate_check.sh 里（包装脚本级），公开仓没有它。
# 现在搬进 conftest；这几条守住"搬过去了、且没被改坏"。
# ══════════════════════════════════════════════════════════════════════════

# 会把测试带偏的运行时开关（不是存储路径，是**行为**开关）
_BEHAVIOUR_ENV = [
    "BLADEX_AUTH_ENABLED",
    "BLADEX_CLIENT_KEYS",
    "BLADEX_ADMIN_KEYS",
    "BLADEX_UPSTREAM_API_BASE",
    "BLADEX_UPSTREAM_API_KEY",
    "BLADEX_UPSTREAM_MODEL",
    "BLADEX_ROUTE_ENABLED",
    "BLADEX_HARD_RULES",
]


@pytest.mark.parametrize("env_key", _BEHAVIOUR_ENV)
def test_runtime_env_not_inherited_at_startup(env_key: str) -> None:
    """继承来的运行时开关必须在 conftest 导入时就被清掉。

    留一个就够坏事：`BLADEX_AUTH_ENABLED=true` 让所有 TestClient 端点 401，
    而报错信息（401）完全指不到真因（你的 shell）。

    断言的是 conftest 记下的**启动快照**，不是 `os.environ` 的当下状态——
    后者会被任何合法 monkeypatch.setenv 的测试打破，是个按执行顺序抽风的断言。
    """
    import conftest as ct

    assert env_key not in ct.POST_SCRUB_BLADEX_ENV, (
        f"{env_key} 在测试启动时仍在环境里 —— conftest 的环境卫生没生效。"
        f"症状会是成片的 401 / 行为翻转，且报错指不到真因。"
    )


def test_only_isolation_vars_survive_the_scrub() -> None:
    """启动后剩下的 BLADEX_* 只能是 conftest 自己写的隔离值。"""
    import conftest as ct

    assert set(ct.POST_SCRUB_BLADEX_ENV) == set(ct._ISOLATED)


def test_auth_defaults_off_when_env_absent(monkeypatch) -> None:
    """env 里没有鉴权开关时，ProxyConfig 必须是"关"。

    这是 C6 那 19 条假失败里最直白的一条（test_auth_disabled_by_default）。
    """
    from bladex_proxy.config import ProxyConfig

    monkeypatch.delenv("BLADEX_AUTH_ENABLED", raising=False)
    monkeypatch.delenv("BLADEX_CLIENT_KEYS", raising=False)
    assert ProxyConfig().auth_enabled is False


def test_keep_env_escape_hatch_is_declared() -> None:
    """逃生门要存在且被记下来——否则调试特殊场景的人只能去读源码。"""
    import conftest as ct

    assert "BLADEX_TEST_KEEP_ENV" in ct.KEEP_ENV_FLAGS
    assert "BLADEX_GATE_KEEP_ENV" in ct.KEEP_ENV_FLAGS


def test_scrubbed_env_is_recorded() -> None:
    """被清掉的值留个记录（诊断用：'我明明设了 X，怎么没生效'）。"""
    import conftest as ct

    assert isinstance(ct.SCRUBBED_ENV, dict)


def test_test_env_wins_over_the_repo_env_file() -> None:
    """隔离值必须被显式钉住，不能依赖默认优先级碰巧站在我们这边。

    2026-08-06 那天 `config/.env` 的默认优先级被翻过去又翻回来（现默认 env 赢，
    见 `bladex_proxy/deployment.py`）。**这一钉与默认值无关**：测试与编排器同类，
    环境里的值是本进程刻意注入的隔离、不是残留，所以要显式声明而不是靠默认。
    少了它，任何从仓库根跑 `cli.main()` 的用例都可能被真实 `.env` 里的
    `data/bladex_rocksdb` 盖掉隔离路径——**测试流量重新写进生产库**。
    """
    import conftest as ct

    assert ct._ISOLATED.get("BLADEX_ENV_PRECEDENCE") == "env"


def test_cli_invocation_does_not_leak_production_env() -> None:
    """🔴 第三次防同一个洞：调过 CLI 的用例不得把真实 `config/.env` 留给后面的用例。

    形态（2026-08-06 gate 实测）：`deployment.load_env_file()` 把部署根的 `.env`
    整份灌进 `os.environ` 且不回收（monkeypatch 没记录这些键，管不着）。
    `test_sync_cli.py` 跑完后进程里多出 28 个 BLADEX_*，含 `BLADEX_ADMIN_KEYS`；
    紧接着 `test_admin_read_api::test_admin_read_requires_key_when_auth_enabled`
    拿合法数据面 key 收到 401——因为 ADMIN_KEYS 一存在，管理面就要求管理 key。
    全量 pytest 里同一条是绿的（collection 顺序不同），**只有 gate 的
    NEW_TESTS 顺序会踩到**：一条按顺序抽风、且指错方向的假失败。

    2026-08-03 修过一次，修在了 `test_cli_lifecycle.py` 单个文件里；
    后来多了第二个调 CLI 的测试文件，洞就回来了。现在护栏在 conftest 层
    （`_restore_bladex_env_baseline`），本测试守它别被删掉。

    本用例**故意**跑一次真实 CLI 入口（`--version` 走 `cli.main` → `_load_env_file`），
    然后断言进程环境仍是基线——不 mock，因为要测的正是"真跑之后干不干净"。
    """
    from bladex_proxy import cli

    from conftest import _BASELINE_BLADEX_ENV

    cli.main(["--version"])
    # 本用例自身的 teardown 还没跑，所以这里看到的是"CLI 刚污染完"的现场；
    # 由 conftest 的 autouse fixture 负责收拾——下一个用例看到的必须是基线。
    # 断言基线快照本身没被改坏（它是恢复的依据）。
    assert "BLADEX_REDIS_STREAM" in _BASELINE_BLADEX_ENV
    assert _BASELINE_BLADEX_ENV["BLADEX_REDIS_STREAM"].startswith("bladex:test:")


def test_previous_test_did_not_leak_admin_keys() -> None:
    """紧接上一条：污染必须已经被收拾干净（顺序敏感，刻意排在它后面）。

    直接断言最要命的那个键不在——它就是 401 假失败的元凶。
    """
    assert "BLADEX_ADMIN_KEYS" not in os.environ, (
        "上一个用例调用 CLI 后把生产 config/.env 留在了进程里 —— "
        "conftest 的 _restore_bladex_env_baseline 没生效"
    )
    assert "BLADEX_UPSTREAM_API_KEY" not in os.environ


def test_env_file_precedence_default_is_env(tmp_path, monkeypatch) -> None:
    """反过来也要钉：产品默认是"环境变量赢"（2026-08-06 复盘拍板）。

    冲突的可见性由 `format_conflicts` 保证（见
    `test_deployment_root_and_env.py::test_stale_shell_export_still_wins_but_is_reported`）
    ——那才是两次事故真正缺的东西。
    """
    from bladex_proxy import deployment

    monkeypatch.delenv("BLADEX_ENV_PRECEDENCE", raising=False)
    monkeypatch.delenv("BLADEX_ENV_KEEP", raising=False)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / ".env").write_text("BLADEX_PORT=1\n", encoding="utf-8")
    monkeypatch.setenv("BLADEX_PORT", "2")

    conflicts = deployment.load_env_file(str(tmp_path))

    assert os.environ["BLADEX_PORT"] == "2"
    assert conflicts and not conflicts[0].applied
