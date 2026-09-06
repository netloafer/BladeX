"""ADR-0027 验收剧本（T28–T33 / T39）。

每组对应评审里的一条实测缺陷，测的是"改前必然失败"的那个判据，不是"不崩溃即合格"。
"""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]


# ══════════════════════════════════════════════════════════════════════════
# T28 · 镜像不得携带运行时密钥（R1 / §2.1）
# ══════════════════════════════════════════════════════════════════════════


def test_dockerignore_exists_and_covers_credentials():
    """没有 .dockerignore + `COPY config ./config` = 真实 .env 烧进镜像层。"""
    di = _REPO / ".dockerignore"
    assert di.is_file(), "仓库根缺 .dockerignore —— 构建上下文会带上 config/.env"
    text = di.read_text(encoding="utf-8")
    for pattern in ("config/.env", "config/routing.toml", "config/identity.toml",
                    "data/", "logs/", "appendonlydir/"):
        assert pattern in text, f".dockerignore 未覆盖 {pattern}"


def test_dockerfile_does_not_copy_whole_config_dir():
    df = (_REPO / "deploy" / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY config ./config" not in df, (
        "Dockerfile 仍在整目录 COPY config —— 运行时凭证会进镜像层（compose 已 volume 挂载，"
        "build 期这个 COPY 是多余的）"
    )
    assert "COPY config/*.example" in df, "应只带模板文件"


# ══════════════════════════════════════════════════════════════════════════
# T29 · 管理面与数据面鉴权分级（R2 / §2.2）
# ══════════════════════════════════════════════════════════════════════════


def _cfg(monkeypatch, tmp_path, *, client="", admin="", identity=False, auth=True):
    from bladex_proxy.config import ProxyConfig

    monkeypatch.setenv("BLADEX_AUTH_ENABLED", "true" if auth else "false")
    monkeypatch.setenv("BLADEX_CLIENT_KEYS", client)
    monkeypatch.setenv("BLADEX_ADMIN_KEYS", admin)
    ipath = tmp_path / "identity.toml"
    if identity:
        ipath.write_text("[[principals]]\nid = \"pipeline\"\n", encoding="utf-8")
    monkeypatch.setenv("BLADEX_IDENTITY_CONFIG", str(ipath))
    return ProxyConfig()


def test_data_plane_key_rejected_on_admin_when_admin_keys_configured(monkeypatch, tmp_path):
    """配了 admin key 后，agent 手里的数据面 key 不再拥有全库管理权。"""
    cfg = _cfg(monkeypatch, tmp_path, client="bladex-agent||a", admin="bladex-adm||adm")
    assert cfg.admin_auth_check("bladex-adm")[0] is True
    ok, reason = cfg.admin_auth_check("bladex-agent")
    assert ok is False, "数据面 key 不应通过管理面鉴权"
    assert reason == "invalid_key"
    # 数据面本身不受影响
    assert cfg.auth_check("bladex-agent")[0] is True


def test_personal_mode_falls_back_to_client_key(monkeypatch, tmp_path):
    """个人模式未配 admin key —— 与今天逐字一致（零回归）。"""
    cfg = _cfg(monkeypatch, tmp_path, client="bladex-agent||a", admin="")
    assert cfg.admin_auth_check("bladex-agent")[0] is True
    assert cfg.admin_auth_check("wrong-key")[0] is False


def test_personal_mode_fallback_emits_warning(monkeypatch, tmp_path):
    from bladex_proxy.server import enforce_admin_key_policy

    cfg = _cfg(monkeypatch, tmp_path, client="bladex-agent||a", admin="")
    assert enforce_admin_key_policy(cfg) is True, "回落必须留下可 grep 的 warning"


def test_enterprise_without_admin_keys_refuses_startup(monkeypatch, tmp_path):
    """企业形态（有 identity.toml）共用 key = 一个员工的 agent key 拥有全库管理权。"""
    from bladex_proxy.server import enforce_admin_key_policy

    cfg = _cfg(monkeypatch, tmp_path, client="bladex-agent||a", admin="", identity=True)
    with pytest.raises(ValueError) as ei:
        enforce_admin_key_policy(cfg)
    # 报错必须指向配置（刚性原则 9）
    assert "BLADEX_ADMIN_KEYS" in str(ei.value)


def test_enterprise_with_admin_keys_starts(monkeypatch, tmp_path):
    from bladex_proxy.server import enforce_admin_key_policy

    cfg = _cfg(monkeypatch, tmp_path, client="bladex-agent||a",
               admin="bladex-adm||adm", identity=True)
    assert enforce_admin_key_policy(cfg) is False


def test_auth_disabled_keeps_open_admin(monkeypatch, tmp_path):
    """auth 关闭时与数据面同语义（由启动横幅承担告警），不引入新的拒绝面。"""
    cfg = _cfg(monkeypatch, tmp_path, client="", admin="", auth=False)
    assert cfg.admin_auth_check(None)[0] is True


# ══════════════════════════════════════════════════════════════════════════
# T30 · 公开树交付物对账 —— 剧本已移到 tests/unit/test_build_public_tree.py
#
# 理由（C6 实测）：这些用例按路径加载 scripts/build_public_tree.py，而那是**私有
# 发布工具、不进公开仓**。留在本文件里，公开树上它们会带着 FileNotFoundError 失败
# ——一个只在公开仓里红的测试，比没有测试更糟。移到已被 TESTS_EXCLUDE 排除的那个
# 文件里，私仓照跑、公开仓不带。
# ══════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════
# T31 · 配置口径统一（O1 / §5.2）
# ══════════════════════════════════════════════════════════════════════════


def test_init_template_ledger_path_matches_config_default(monkeypatch):
    """init 模板写 data/bladex_p3、config.py 默认 data/bladex_hub
    = 老用户 `init --force` 后指向空 Memory Hub —— "记忆全没了"的工单形状。"""
    from bladex_proxy.cli import _ENV_MINIMAL_TEMPLATE
    from bladex_proxy.config import ProxyConfig

    monkeypatch.delenv("BLADEX_ROCKSDB_PATH", raising=False)
    default = ProxyConfig().rocksdb_path
    m = re.search(r"^BLADEX_ROCKSDB_PATH=(.+)$", _ENV_MINIMAL_TEMPLATE, re.MULTILINE)
    assert m and m.group(1).strip() == default


@pytest.mark.parametrize("raw,expected", [
    ("k1||label1||k2||label2", "k1"),
    ("k1", "k1"),
    ("k1||a label, with comma||k2||l2", "k1"),  # 逗号 split 会在这里出错
    ("", ""),
])
def test_key_parsing_is_pipe_only(raw, expected):
    """格式权威是 KeyStore.parse（纯 ||）。CLI/MCP 曾先按逗号 split，label 带逗号即出错。"""
    from bladex_mcp.server import _first_key_of as mcp_first
    from bladex_proxy.auth import KeyStore
    from bladex_proxy.cli import _first_key_of as cli_first

    assert cli_first(raw) == expected
    assert mcp_first(raw) == expected
    store = KeyStore.parse(raw)
    assert (store.keys[0].key if store.keys else "") == expected


def test_key_parsing_agrees_with_keystore_on_multiple_keys():
    from bladex_proxy.auth import KeyStore
    from bladex_proxy.cli import _first_key_of

    raw = "k1||laptop||k2||codex"
    store = KeyStore.parse(raw)
    assert [k.key for k in store.keys] == ["k1", "k2"]
    assert _first_key_of(raw) == store.keys[0].key


def test_admin_key_prefers_admin_env(monkeypatch):
    from bladex_proxy.cli import _admin_key

    monkeypatch.setenv("BLADEX_CLIENT_KEYS", "cli-key||a")
    monkeypatch.setenv("BLADEX_ADMIN_KEYS", "adm-key||adm")
    assert _admin_key() == "adm-key"
    monkeypatch.delenv("BLADEX_ADMIN_KEYS")
    assert _admin_key() == "cli-key"


# ══════════════════════════════════════════════════════════════════════════
# T32 · 绿灯不许骗人（U1 / §3.1）
# ══════════════════════════════════════════════════════════════════════════


class _FakeCfg:
    def __init__(self, api_base="", api_key="", upstream_model="m"):
        self.upstream_api_base = api_base
        self.upstream_api_key = api_key
        self.upstream_model = upstream_model
        self.routing_config = type("RC", (), {"models": []})()


def test_probe_upstream_empty_key_is_failure():
    """空 key 曾是绿灯（doctor 只测网络层）——冷启动实测里最误导人的一条。"""
    from bladex_proxy.cli import _probe_upstream

    state, detail, fix = _probe_upstream(_FakeCfg(api_base="https://x/v1", api_key=""))
    assert state == "auth_failed"
    assert "BLADEX_UPSTREAM_API_KEY" in detail + fix


def test_probe_upstream_no_api_base_is_not_probed():
    """探不了就说"未探测"，不打 ✓。"""
    from bladex_proxy.cli import _probe_upstream

    state, _, _ = _probe_upstream(_FakeCfg(api_base="", api_key="k"))
    assert state == "not_probed"


def test_probe_upstream_401_is_auth_failed(monkeypatch):
    import urllib.error

    from bladex_proxy import cli

    def _raise(*a, **k):
        raise urllib.error.HTTPError("u", 401, "Unauthorized", {}, None)

    monkeypatch.setattr(cli.urllib.request, "urlopen", _raise)
    state, detail, _ = cli._probe_upstream(_FakeCfg(api_base="https://x/v1", api_key="bad"))
    assert state == "auth_failed" and "401" in detail


def test_probe_upstream_network_error_is_unreachable(monkeypatch):
    import urllib.error

    from bladex_proxy import cli

    def _raise(*a, **k):
        raise urllib.error.URLError("no route")

    monkeypatch.setattr(cli.urllib.request, "urlopen", _raise)
    state, _, _ = cli._probe_upstream(_FakeCfg(api_base="https://x/v1", api_key="k"))
    assert state == "unreachable"


def test_config_check_flags_empty_upstream_key(tmp_path):
    from bladex_proxy.cli import _upstream_key_problems

    problems = _upstream_key_problems(_FakeCfg(api_key=""), tmp_path / "nope.toml")
    assert problems and "BLADEX_UPSTREAM_API_KEY" in problems[0][1]


def test_config_check_flags_dangling_api_key_env(monkeypatch, tmp_path):
    """routing.toml 只引 env 名是既有纪律，但"引了个不存在的 env"此前无人检查。"""
    from bladex_proxy.cli import _upstream_key_problems

    monkeypatch.delenv("SOME_MISSING_KEY_ENV", raising=False)
    cfg = _FakeCfg(api_key="")
    cfg.routing_config = type("RC", (), {
        "models": [type("M", (), {"name": "m1", "api_key_env": "SOME_MISSING_KEY_ENV"})()]
    })()
    rpath = tmp_path / "routing.toml"
    rpath.write_text("", encoding="utf-8")
    problems = _upstream_key_problems(cfg, rpath)
    assert problems and "SOME_MISSING_KEY_ENV" in problems[0][1]


def test_config_check_passes_when_api_key_env_present(monkeypatch, tmp_path):
    from bladex_proxy.cli import _upstream_key_problems

    monkeypatch.setenv("PRESENT_KEY_ENV", "sk-abc")
    cfg = _FakeCfg(api_key="")
    cfg.routing_config = type("RC", (), {
        "models": [type("M", (), {"name": "m1", "api_key_env": "PRESENT_KEY_ENV"})()]
    })()
    rpath = tmp_path / "routing.toml"
    rpath.write_text("", encoding="utf-8")
    assert _upstream_key_problems(cfg, rpath) == []


def test_probe_ready_reads_body_on_503(monkeypatch):
    """/ready 未就绪返回 503 + body —— 恰恰是我们最需要读的那种情况。"""
    import io
    import urllib.error

    from bladex_proxy import cli

    payload = json.dumps({"ready": False, "checks": {"redis": False, "hub": True}}).encode()

    def _raise(*a, **k):
        raise urllib.error.HTTPError("u", 503, "unavailable", {}, io.BytesIO(payload))

    monkeypatch.setattr(cli.urllib.request, "urlopen", _raise)
    got = cli._probe_ready("http://x/ready")
    assert got is not None and got["checks"]["redis"] is False


def test_startup_state_reports_degraded_with_consequences(monkeypatch, capsys):
    """记忆管线的死亡必须出现在 console，且写明后果（不是一句 'degraded'）。"""
    from bladex_proxy import cli

    monkeypatch.setattr(cli, "_probe_ready",
                        lambda *a, **k: {"ready": False,
                                         "checks": {"redis": False, "hub": True, "index": True}})
    rc = cli._report_startup_state("127.0.0.1", 38080, redis_ok=False,
                                   consolidator_ok=False, timeout_s=1)
    out = capsys.readouterr().out
    assert rc == 0, "代理确实能转发 —— degraded 不是启动失败"
    assert "degraded" in out
    # 后果句必须写出来 —— 一句 "degraded" 用户看不出记忆没了
    assert "no memory will be produced" in out       # Redis 缺席的后果
    assert "no facts will be distilled" in out       # consolidator 缺席的后果
    assert "/dashboard" in out                       # U13：dashboard 可发现


def test_startup_state_reports_ready(monkeypatch, capsys):
    from bladex_proxy import cli

    monkeypatch.setattr(cli, "_probe_ready",
                        lambda *a, **k: {"ready": True,
                                         "checks": {"redis": True, "hub": True, "index": True}})
    rc = cli._report_startup_state("127.0.0.1", 38080, True, True, timeout_s=1)
    out = capsys.readouterr().out
    assert rc == 0 and "ready" in out and "degraded" not in out


def test_startup_state_reports_failed_when_no_response(monkeypatch, capsys):
    from bladex_proxy import cli

    monkeypatch.setattr(cli, "_probe_ready", lambda *a, **k: None)
    rc = cli._report_startup_state("127.0.0.1", 38080, True, True, timeout_s=0.1)
    assert rc == 1 and "failed" in capsys.readouterr().out


# ══════════════════════════════════════════════════════════════════════════
# T33 · consolidator 进包（U2 / §4.2）
# ══════════════════════════════════════════════════════════════════════════


def test_consolidator_is_importable_from_package():
    """入口在 wheel 里 —— 否则 pip 安装的用户永远启动不了 Memory Index 提炼。"""
    import bladex_proxy.consolidator as c

    assert callable(c.main)


def test_consolidator_console_script_declared():
    text = (_REPO / "packages" / "bladex-proxy" / "pyproject.toml").read_text(encoding="utf-8")
    assert "bladex-consolidator = \"bladex_proxy.consolidator:main\"" in text


def test_consolidator_command_prefers_package_entry():
    from bladex_proxy.cli import _consolidator_command

    cmd = _consolidator_command()
    assert cmd is not None
    assert cmd[1:] == ["-m", "bladex_proxy.consolidator"]


def test_repo_script_is_thin_shell():
    src = (_REPO / "scripts" / "run_index_consolidator.py").read_text(encoding="utf-8")
    assert "from bladex_proxy.consolidator import main" in src
    assert len(src.splitlines()) < 60, "脚本应是薄壳，实现留在包里"


def test_consolidator_pid_matches_module_form(monkeypatch):
    """包内入口的进程名与老脚本名不同，pgrep 兜底必须两种都认。"""
    import subprocess

    from bladex_proxy import cli

    monkeypatch.setattr(cli, "_read_pidfile", lambda *_: None)
    seen: list[str] = []

    def _fake_run(argv, **kw):
        seen.append(argv[-1])
        out = "4242\n" if argv[-1] == "bladex_proxy.consolidator" else ""
        return subprocess.CompletedProcess(argv, 0, stdout=out, stderr="")

    monkeypatch.setattr(cli.subprocess, "run", _fake_run)
    assert cli._consolidator_pid() == 4242
    assert "bladex_proxy.consolidator" in seen


# ══════════════════════════════════════════════════════════════════════════
# T35 · BLADEX_HOME 路径发现（Y2 / U19 / §4.3）
# ══════════════════════════════════════════════════════════════════════════


def test_repo_root_defaults_to_cwd(monkeypatch, tmp_path):
    """cwd 仍是默认 —— 仓库内行为逐字不变。

    2026-08-06：查找顺序补了"cwd 的祖先目录"与 `~/.bladex` 兜底，所以要把 HOME
    也重定向到 tmp——否则这条断言会随"跑测试的机器上有没有 ~/.bladex"漂移。
    """
    from bladex_proxy.cli import _repo_root

    monkeypatch.delenv("BLADEX_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "fake_home"))
    monkeypatch.chdir(tmp_path)
    assert _repo_root() == str(tmp_path)


def test_repo_root_honours_bladex_home(monkeypatch, tmp_path):
    from bladex_proxy.cli import _repo_root

    home = tmp_path / "deploy"
    home.mkdir()
    monkeypatch.setenv("BLADEX_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    assert _repo_root() == str(home)


def test_config_root_found_via_bladex_home(monkeypatch, tmp_path):
    """全局安装后在任意目录跑命令 —— 此前会静默看到一个空库。"""
    from bladex_proxy.cli import _require_config_root

    home = tmp_path / "deploy"
    (home / "config").mkdir(parents=True)
    (home / "config" / ".env").write_text("BLADEX_PORT=1\n", encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.setenv("BLADEX_HOME", str(home))
    assert _require_config_root() == str(home)


def test_config_root_missing_lists_searched_paths(monkeypatch, tmp_path, capsys):
    """找不到配置时报错要列出找过的位置，而不是静默用空目录。

    2026-08-06 起是三个锚点（BLADEX_HOME / cwd / ~/.bladex）；cwd 的祖先目录由
    报错文案单独说一行，不逐条列（深路径下会刷屏）。
    """
    from bladex_proxy import cli

    monkeypatch.setenv("BLADEX_HOME", str(tmp_path / "nowhere"))
    monkeypatch.chdir(tmp_path)
    assert cli._require_config_root() is None
    paths = cli._config_search_paths()
    assert len(paths) == 3 and all(p.endswith("config/.env") for p in paths)


# ══════════════════════════════════════════════════════════════════════════
# T38 · 日志保留策略（O3 / §4.4）
# ══════════════════════════════════════════════════════════════════════════


def test_prune_logs_keeps_newest_per_kind(monkeypatch, tmp_path):
    """审查时实测 logs/ 已 157 个文件、永不清理。按类分组各留 N 个。"""
    import time as _t

    from bladex_proxy.cli import _prune_logs

    logs = tmp_path / "logs"
    logs.mkdir()
    for kind in ("proxy", "consolidator"):
        for i in range(25):
            f = logs / f"{kind}-2026080{i // 10}-{i:04d}.log"
            f.write_text("x", encoding="utf-8")
            os.utime(f, (_t.time() + i, _t.time() + i))
    monkeypatch.setenv("BLADEX_HOME", str(tmp_path))
    removed = _prune_logs(keep=20)
    assert removed == 10  # 两类各删 5
    for kind in ("proxy", "consolidator"):
        assert len(list(logs.glob(f"{kind}-*.log"))) == 20
    # 保留的是最新的
    assert (logs / "proxy-20260802-0024.log").exists()


def test_prune_logs_disabled_by_zero(monkeypatch, tmp_path):
    from bladex_proxy.cli import _prune_logs

    logs = tmp_path / "logs"
    logs.mkdir()
    for i in range(5):
        (logs / f"proxy-{i}.log").write_text("x", encoding="utf-8")
    monkeypatch.setenv("BLADEX_HOME", str(tmp_path))
    monkeypatch.setenv("BLADEX_LOG_KEEP", "0")
    assert _prune_logs() == 0
    assert len(list(logs.glob("*.log"))) == 5


# ══════════════════════════════════════════════════════════════════════════
# T39 · dashboard 转义 / 导出游标（O5 / O4 / §2.3 / §5.3）
# ══════════════════════════════════════════════════════════════════════════


def test_dashboard_escapes_single_quote():
    """onclick="fn('${esc(id)}')" 用单引号包裹，漏 ' 就是一条注入路径。"""
    html = (_REPO / "packages" / "bladex-proxy" / "bladex_proxy"
            / "dashboard.html").read_text(encoding="utf-8")
    m = re.search(r"function esc\(s\)\{[^\n]*\}", html)
    assert m, "找不到 esc()"
    assert "'" in m.group(0).split("replace(")[1].split(",")[0], "esc 未转义单引号"
    assert "&#39;" in m.group(0)


def test_dashboard_lang_is_english():
    html = (_REPO / "packages" / "bladex-proxy" / "bladex_proxy"
            / "dashboard.html").read_text(encoding="utf-8")
    assert '<html lang="en"' in html


def test_fact_has_updated_at_defaulting_to_none():
    from bladex_core.fact import Fact

    f = Fact(content="x")
    assert f.updated_at is None, "历史数据零迁移：默认 None，取时退化成 created_at"


def test_export_cursor_uses_max_of_created_and_updated():
    """核心判据：一条老 fact 被更新后，必须重新出现在增量批里。"""
    from bladex_core.fact import Fact
    from bladex_proxy.export_sync import _fact_change_ts

    old = datetime(2026, 1, 1, tzinfo=UTC)
    f = Fact(content="x", created_at=old)
    assert _fact_change_ts(f) == old.isoformat()

    f.updated_at = old + timedelta(days=200)
    assert _fact_change_ts(f) == (old + timedelta(days=200)).isoformat()
    assert _fact_change_ts(f) > old.isoformat(), "改前这里等于 created_at → 变更永不重导"


def test_supersede_marks_updated_at():
    """被取代是"变更"里最要紧的一种：目标端不能继续把它当现行。"""
    from bladex_core.fact import Fact
    from bladex_core.supersede import plan_supersede_merge

    now = datetime(2026, 8, 4, tzinfo=UTC)
    old = Fact(id="f1", content="用户偏好深色主题", subject="theme", attribute="preference",
               created_at=datetime(2026, 1, 1, tzinfo=UTC))
    new = Fact(id="f2", content="用户偏好浅色主题", subject="theme", attribute="preference",
               created_at=now)
    plan = plan_supersede_merge([new], [old], now=now)
    if plan.invalidate:  # 走了取代分支
        assert old.updated_at == now
        assert old.t_invalid == now


# ══════════════════════════════════════════════════════════════════════════
# 入口点可用性（2026-08-05，C7 实测驱动）
#
# 事故形状：`bladex --version` 在作者本机 `ModuleNotFoundError: bladex_proxy`，
# 而三个 dist-info 都在、editable 的 .pth 内容也正确。**所有既定跑法都显式带
# PYTHONPATH**（gate_check / 各 scripts 自举 sys.path），于是"装了但导不进来"
# 被掩盖了很久——直到有人第一次真的敲 `bladex`。
#
# 所以这里不 import 模块（那会被 PYTHONPATH 蒙混），而是**在剥掉 PYTHONPATH 的
# 子进程里**跑真实入口点：判据是"如果它自称已安装，就必须真的能跑"。
# ══════════════════════════════════════════════════════════════════════════


def _installed_version(dist: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version(dist)
    except PackageNotFoundError:
        return None


@pytest.mark.parametrize(("dist", "module"), [
    ("bladex-core", "bladex_core"),
    ("bladex-proxy", "bladex_proxy"),
])
def test_installed_package_is_importable_without_pythonpath(dist: str, module: str):
    """自称已安装的包，必须在没有 PYTHONPATH 兜底时也能 import。"""
    import subprocess
    import sys

    if _installed_version(dist) is None:
        pytest.skip(f"{dist} 未安装到当前解释器（源码树直跑），本条不适用")

    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run(  # noqa: S603
        [sys.executable, "-c", f"import {module}"],
        capture_output=True, text=True, check=False, env=env,
    )
    assert r.returncode == 0, (
        f"{dist} 记为已安装，但剥掉 PYTHONPATH 后 `import {module}` 失败：\n"
        f"{r.stderr.strip()}\n"
        f"典型原因：editable 安装坏了 / venv 半陈旧。修法：删掉 .venv 重建 "
        f"（rm -rf .venv && uv sync），别靠 PYTHONPATH 长期掩盖。"
    )


def test_cli_entry_point_runs_without_pythonpath():
    """`bladex --version` 这条真实入口点必须能跑。

    只 import 模块不够——console script 是另一条路径（shebang + entry point），
    用户敲的就是它。
    """
    import subprocess
    import sys

    if _installed_version("bladex-proxy") is None:
        pytest.skip("bladex-proxy 未安装到当前解释器，本条不适用")

    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "bladex_proxy.cli", "--version"],
        capture_output=True, text=True, check=False, env=env,
    )
    assert r.returncode == 0, f"入口点跑不起来：\n{r.stderr.strip()}"
    assert "bladex" in r.stdout.lower()


# ── doctor 端口检查：占用 ≠ 问题（2026-08-05 C7）─────────────────────────────


class _FakeSocket:
    """connect_ex 的返回值就是"端口通不通"，其余方法是噪音。"""

    def __init__(self, connect_rc: int) -> None:
        self._rc = connect_rc

    def settimeout(self, _t):
        pass

    def connect_ex(self, _addr):
        return self._rc

    def close(self):
        pass


def _fake_connect(monkeypatch, connect_rc: int) -> None:
    from bladex_proxy import cli

    monkeypatch.setattr(cli.socket, "socket", lambda *a, **k: _FakeSocket(connect_rc))


# ── doctor --e2e 探针的三条结构约束（2026-08-05 首次实跑 G4 用失败换来的）─────
#
# 探针要穿过一个 LLM 改写和一层 novelty 判重，能活下来的形状比看上去窄：
# ① token 是句子的语义载荷（当定语会被模型改写时省掉，实测 new_facts=1 却搜不到 token）；
# ② 整句每轮不同（常量子句第二次跑就被判重：candidates=2 new_facts=1 existing=1）；
# ③ 只说一件事（两个子句会被按原子事实拆开，token 只跟其中一条走）。
# 这三条都是纯字符串性质，不必跑真机就能守住——而这个检查此前从未真跑过。


def test_probe_is_a_single_sentence_about_one_thing():
    """两个子句会被蒸馏器拆成两条 fact，token 只跟着其中一条走。"""
    from bladex_proxy.cli import _E2E_TEMPLATE

    text = _E2E_TEMPLATE.format(token="bxdeadbeef")
    assert text.count(".") == 1 and text.endswith("."), f"探针必须是单句：{text!r}"
    assert ", and " not in text, f"并列子句会被拆开：{text!r}"


def test_probe_token_is_the_payload_not_a_modifier():
    """token 要处在"被陈述的值"的位置——句末的 `is {token}.`。

    实测反例：`the probe named {token} has the favourite colour chartreuse`
    里 token 降级成定语，模型改写成中文时直接省掉了它，该轮库里搜不到 token。
    """
    from bladex_proxy.cli import _E2E_TEMPLATE

    assert _E2E_TEMPLATE.rstrip().endswith("is {token}."), (
        f"token 不在句末的值位上：{_E2E_TEMPLATE!r}")


def test_recall_query_is_a_paraphrase_without_the_answer():
    """问句要是 fact 的语义改写，且**不含答案**——带 token 就退化成关键词查找。"""
    from bladex_proxy.cli import _E2E_RECALL_QUERY, _E2E_TEMPLATE

    assert "{token}" not in _E2E_RECALL_QUERY
    assert _E2E_RECALL_QUERY.endswith("?")
    probe = _E2E_TEMPLATE.format(token="bxdeadbeef")
    assert _E2E_RECALL_QUERY not in probe, "问句不能是探针原文的子串"
    # 与探针共享足以定位这条 fact 的主题词，否则问的根本不是同一件事
    for word in ("BladeX", "probe", "passphrase"):
        assert word in _E2E_RECALL_QUERY and word in probe


def test_probe_port_free(monkeypatch):
    from bladex_proxy import cli

    _fake_connect(monkeypatch, 1)
    state, detail = cli._probe_port("127.0.0.1", 38080)
    assert state == "free" and "free" in detail


def test_probe_port_in_use_by_our_proxy_is_not_a_warning(monkeypatch):
    """`bladex start` 之后端口必然被占 —— 那是健康态。

    旧实现无条件打 ⚠，于是正常启动后跑 doctor 必见一条警告。假红灯和假绿灯一样有害：
    它训练用户忽略警告，而 ADR-0027 §3 整条契约就是"灯要说真话"。
    """
    from bladex_proxy import cli

    _fake_connect(monkeypatch, 0)
    monkeypatch.setattr(cli, "_probe_ready", lambda *a, **k: {"ready": True, "checks": {}})
    state, detail = cli._probe_port("127.0.0.1", 38080)
    assert state == "ours"
    assert "serving this proxy" in detail


def test_probe_port_in_use_by_stranger_still_warns(monkeypatch):
    """真冲突（占端口的不应答 /ready）仍要报 —— start 会失败，用户得知道。"""
    from bladex_proxy import cli

    _fake_connect(monkeypatch, 0)
    monkeypatch.setattr(cli, "_probe_ready", lambda *a, **k: None)
    state, detail = cli._probe_port("127.0.0.1", 38080)
    assert state == "foreign"
    assert "not a BladeX proxy" in detail and "BLADEX_PORT" in detail


# ══════════════════════════════════════════════════════════════════════════
# 存储目录名：单一真相源（2026-08-07 改名 bladex_p2/bladex_rocksdb）
# ══════════════════════════════════════════════════════════════════════════
#
# 为什么要守：`MemoryIndex` 对**不存在的路径会开出一个空库**。所以一处漏改
# 不会报错，只会安静地给出「0 条、一切干净」的报告 —— 而那种假绿灯在
# 2026-08-07 一天内就出现过两次（空目录审计报告、只查一个长度带的探针）。
# 20 个脚本各自硬编码过一份默认值，这条守卫钉住"以后只许有一处"。


def test_storage_dir_defaults_have_single_source():
    """默认目录名只许写在 `config.DEFAULT_*_DIR`，别处一律 import。"""
    import re as _re
    from pathlib import Path

    from bladex_proxy.config import DEFAULT_INDEX_DIR, DEFAULT_HUB_DIR

    assert DEFAULT_INDEX_DIR == "data/bladex_index"
    assert DEFAULT_HUB_DIR == "data/bladex_hub"

    # conftest 的 `PRODUCTION_DEFAULTS` 是**第二份**同样的值（测试隔离守卫靠它
    # 判断"有没有指向生产库"）。两份一旦漂开，隔离守卫就会去比一个不存在的路径
    # —— 永远通过，永远没在守。所以这里对账。
    import conftest as _ct
    assert _ct.PRODUCTION_DEFAULTS["BLADEX_INDEX_PATH"] == DEFAULT_INDEX_DIR
    assert _ct.PRODUCTION_DEFAULTS["BLADEX_ROCKSDB_PATH"] == DEFAULT_HUB_DIR

    root = Path(__file__).resolve().parents[3]
    # 归档树 / 冻结树 / 生成树不参与；归档目录里那些名字是真实存在的历史目录
    skip_dir = ("bench_frozen", "public-tree", "data", ".venv", "__pycache__",
                "archive", ".git")
    archive_mark = _re.compile(r"pre0024_archive|pre_v4_archive|archive_2026")
    offenders: list[str] = []
    for pat in ("scripts/*.py", "scripts/*.sh", "packages/**/*.py",
                "deploy/*.yml", "conftest.py"):
        for f in root.glob(pat):
            if any(s in f.parts for s in skip_dir) or not f.is_file():
                continue
            if f.name in ("config.py", "restore.sh", "test_release_readiness.py"):
                continue  # 真相源本身 / 备份兼容层 / 本测试
            for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                if archive_mark.search(line):
                    continue
                if "bladex_p2" in line or "bladex_rocksdb" in line:
                    offenders.append(f"{f.relative_to(root)}:{i}: {line.strip()[:70]}")
    assert not offenders, (
        "这些地方还写着旧目录名（应改用 config.resolve_index_path / "
        "resolve_hub_path，或 DEFAULT_*_DIR）：\n  " + "\n  ".join(offenders))


def test_init_template_index_path_matches_config_default(monkeypatch):
    """init 模板的 INDEX_PATH 也要跟 config 默认一致。

    此前只守了 `BLADEX_ROCKSDB_PATH` —— 而 Index 走岔同样是"记忆全没了"的形状，
    只是症状更隐蔽（Ledger 指错会立刻空，Index 指错会先被当成"重建还没跑完"）。
    """
    from bladex_proxy.cli import _ENV_MINIMAL_TEMPLATE
    from bladex_proxy.config import ProxyConfig

    monkeypatch.delenv("BLADEX_INDEX_PATH", raising=False)
    default = ProxyConfig().index_path
    m = re.search(r"^BLADEX_INDEX_PATH=(.+)$", _ENV_MINIMAL_TEMPLATE, re.MULTILINE)
    assert m and m.group(1).strip() == default


def test_resolve_paths_honour_env_and_root(monkeypatch, tmp_path):
    """env 覆盖 > 默认；`root` 只锚相对路径，绝对路径原样保留。"""
    from bladex_proxy.config import resolve_index_path, resolve_hub_path

    monkeypatch.delenv("BLADEX_INDEX_PATH", raising=False)
    monkeypatch.delenv("BLADEX_ROCKSDB_PATH", raising=False)
    assert resolve_index_path() == "data/bladex_index"
    assert resolve_index_path("/repo") == "/repo/data/bladex_index"
    assert resolve_hub_path("/repo") == "/repo/data/bladex_hub"

    monkeypatch.setenv("BLADEX_INDEX_PATH", "custom/idx")
    assert resolve_index_path("/repo") == "/repo/custom/idx"
    monkeypatch.setenv("BLADEX_INDEX_PATH", "/abs/idx")
    assert resolve_index_path("/repo") == "/abs/idx", "绝对路径不该被 root 前缀污染"
