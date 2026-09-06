"""部署根发现 + config/.env 优先级（2026-08-06 两个用户报的问题）。

## 问题一：必须在部署目录里启动

`_repo_root` / `_require_config_root` 的候选只有 `BLADEX_HOME` 和 cwd 本身——
不向上找、没有默认位置。于是 `cd packages && bladex status` 就会看到一个**空库**，
而且不报错。更糟的是同一件事有三份实现且各不相同：consolidator 连 `BLADEX_HOME`
都不认（只有 cwd），admin_read 用 `Path(__file__).parents[3]`（装成 wheel 后指向
site-packages，漂移检测因此恒空）。

## 问题二：改了 .env 不生效

`if key not in os.environ` —— 环境里已有的值永远赢，配置文件静默失效：

  - 08-05：`.env` 写 `BLADEX_HOTPATH_BUDGET_MS=200`、进程跑 150（残留 export）；
  - 08-06：改了上游 base URL 与 API key、重启 consolidator 仍读旧值 →
    大量 `distill_failed`，提炼整段停摆。

**修复方向（2026-08-06 复盘后拍板）：保持"环境变量优先"，把冲突变成可见的。**
当天早些时候曾把默认翻成"文件优先"，复盘认为翻错了地方——两次事故的病根是
**冲突当时完全不可见**（症状是"文件写对了、进程用的是别的值"，而唯一痕迹要么没有、
要么混在 info 流里），不是优先级方向本身。既然 `format_conflicts` 现在无条件把每个
被遮蔽的键连值一起 WARNING 出来，翻转优先级的收益就没了，只剩"文件悄悄改写编排器
注入的值"这个新风险。想让文件赢用 `BLADEX_ENV_PRECEDENCE=file` 显式声明。

这组测试守的是：发现顺序、覆盖方向（默认 env 赢）、密钥不外泄、
以及**无论谁赢都必须有声音**——最后这条才是两次事故真正缺的东西。
"""

from __future__ import annotations

import os

import pytest
from bladex_proxy import deployment


def _make_root(path, body: str = "BLADEX_PORT=1\n"):
    (path / "config").mkdir(parents=True, exist_ok=True)
    (path / "config" / ".env").write_text(body, encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    """把三个"决定加载方式"的键清干净——否则跑测试的那台机器上的真实
    BLADEX_HOME 会让断言随环境漂移。~/.bladex 兜底也重定向到 tmp。"""
    for key in ("BLADEX_HOME", "BLADEX_ENV_PRECEDENCE", "BLADEX_ENV_KEEP"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "fake_home"))
    monkeypatch.setattr(deployment, "_invocation_cwd", None)


# ── 问题一：根发现 ──────────────────────────────────────────────────────────


def test_finds_root_from_a_subdirectory(monkeypatch, tmp_path):
    """🔴 用户报的第一个症状：换个路径就不认了。

    `cd packages && bladex status` 此前看到的是一个空库（而不是报错）。
    """
    root = _make_root(tmp_path / "deploy")
    nested = root / "packages" / "bladex-proxy" / "bladex_proxy"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    assert deployment.find_root() == str(root)


def test_bladex_home_still_wins(monkeypatch, tmp_path):
    """显式声明压过自动发现（老语义不变）。"""
    home = _make_root(tmp_path / "home_root")
    cwd = _make_root(tmp_path / "cwd_root")
    monkeypatch.chdir(cwd)
    monkeypatch.setenv("BLADEX_HOME", str(home))
    assert deployment.find_root() == str(home)


def test_bladex_home_is_final_even_when_it_has_no_config(monkeypatch, tmp_path):
    """🔴 显式声明是终局答案，不 fall through 回当前仓库。

    起草时本想"过期的 BLADEX_HOME 就回退到 cwd 找"，写完才发现 `bladex init`
    写的正是 `deployment_root()` —— fall through 会让"我指定了部署目录、
    顺手在某个仓库里跑 init"悄悄把配置写进那个仓库。
    """
    cwd = _make_root(tmp_path / "cwd_root")
    empty = tmp_path / "brand_new"
    empty.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setenv("BLADEX_HOME", str(empty))

    assert deployment.find_root() is None            # 读：那儿还没有配置
    assert deployment.deployment_root() == str(empty)  # 写：init 正该建在那儿


def test_default_home_is_last_resort(monkeypatch, tmp_path):
    """全局安装（pip / uv tool）的人没有"仓库"可站——~/.bladex 兜底。"""
    fake_home = tmp_path / "fake_home"
    _make_root(fake_home / ".bladex")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert deployment.find_root() == str((fake_home / ".bladex").resolve())


def test_no_root_returns_none_and_root_falls_back_to_cwd(monkeypatch, tmp_path):
    """找不到 ≠ 崩溃：find_root 返回 None，deployment_root 退回 cwd。"""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert deployment.find_root() is None
    assert deployment.deployment_root() == str(elsewhere.resolve())


def test_search_paths_lists_the_three_anchors(monkeypatch, tmp_path):
    """报错时把找过的位置原样列出来，不静默用空目录。"""
    monkeypatch.setenv("BLADEX_HOME", str(tmp_path / "nowhere"))
    monkeypatch.chdir(tmp_path)
    paths = deployment.search_paths()
    assert all(p.endswith(os.path.join("config", ".env")) for p in paths)
    assert len(paths) == 3  # BLADEX_HOME / cwd / ~/.bladex


def test_user_paths_stay_relative_to_where_the_command_was_typed(monkeypatch, tmp_path):
    """CLI 会 chdir 到部署根，但 `bladex export out.jsonl` 必须写在用户那边。"""
    typed_from = tmp_path / "somewhere"
    typed_from.mkdir()
    monkeypatch.chdir(typed_from)
    deployment.remember_invocation_cwd()
    monkeypatch.chdir(_make_root(tmp_path / "deploy"))

    assert deployment.resolve_user_path("out.jsonl") == str(typed_from / "out.jsonl")
    assert deployment.resolve_user_path("/tmp/abs.jsonl") == "/tmp/abs.jsonl"
    assert deployment.resolve_user_path("") == ""  # 空 = 用默认值，别替它编一个


# ── 问题二：.env 优先级 ─────────────────────────────────────────────────────


def test_stale_shell_export_still_wins_but_is_reported(monkeypatch, tmp_path):
    """🔴 用户报的第二个症状，逐字复刻——但**修的是可见性，不是优先级**。

    改了上游 base URL 与 API key、重启 consolidator，进程仍读终端里的残留 export
    → 大量 distill_failed。默认语义仍是环境赢（12 因子 / 编排器必须能覆盖），
    但现在这件事**必须留下痕迹**：两个键都进冲突清单、都是"被遮蔽"、
    并由 `format_conflicts` 报成 WARNING。事故当时缺的正是这条。
    """
    root = _make_root(tmp_path / "deploy", (
        'export BLADEX_UPSTREAM_API_BASE="https://new.example.com/v1"\n'
        'export BLADEX_UPSTREAM_API_KEY="sk-new-key-value-1234"\n'
    ))
    monkeypatch.setenv("BLADEX_UPSTREAM_API_BASE", "https://old.example.com/v1")
    monkeypatch.setenv("BLADEX_UPSTREAM_API_KEY", "sk-old-key-value-9876")

    conflicts = deployment.load_env_file(str(root))

    assert os.environ["BLADEX_UPSTREAM_API_BASE"] == "https://old.example.com/v1"
    assert os.environ["BLADEX_UPSTREAM_API_KEY"] == "sk-old-key-value-9876"
    assert {c.key for c in conflicts} == {"BLADEX_UPSTREAM_API_BASE", "BLADEX_UPSTREAM_API_KEY"}
    assert not any(c.applied for c in conflicts)

    lines = deployment.format_conflicts(conflicts, "config/.env")
    blob = "\n".join(lines)
    assert "WARNING" in blob and "NOT in effect" in blob
    assert "sk-new-key-value-1234" not in blob and "sk-old-key-value-9876" not in blob


def test_the_2026_08_05_budget_incident_is_now_audible(monkeypatch, tmp_path):
    """.env 写 200、终端残留 150 —— 150 仍然赢，但不再是**静默**失效。"""
    root = _make_root(tmp_path / "deploy", "BLADEX_HOTPATH_BUDGET_MS=200\n")
    monkeypatch.setenv("BLADEX_HOTPATH_BUDGET_MS", "150")
    conflicts = deployment.load_env_file(str(root))
    assert os.environ["BLADEX_HOTPATH_BUDGET_MS"] == "150"
    assert len(conflicts) == 1 and not conflicts[0].applied
    assert "WARNING" in "\n".join(deployment.format_conflicts(conflicts, "config/.env"))


def test_precedence_file_lets_the_config_win(monkeypatch, tmp_path):
    """显式 `BLADEX_ENV_PRECEDENCE=file` 时文件压过环境（逃生口的另一侧）。"""
    root = _make_root(tmp_path / "deploy", "BLADEX_HOTPATH_BUDGET_MS=200\n")
    monkeypatch.setenv("BLADEX_HOTPATH_BUDGET_MS", "150")
    monkeypatch.setenv("BLADEX_ENV_PRECEDENCE", "file")
    conflicts = deployment.load_env_file(str(root))
    assert os.environ["BLADEX_HOTPATH_BUDGET_MS"] == "200"
    assert len(conflicts) == 1 and conflicts[0].applied


def test_unset_keys_are_filled_in(monkeypatch, tmp_path):
    root = _make_root(tmp_path / "deploy", "BLADEX_PORT=39999\n")
    monkeypatch.delenv("BLADEX_PORT", raising=False)
    assert deployment.load_env_file(str(root)) == []
    assert os.environ["BLADEX_PORT"] == "39999"


def test_identical_values_are_not_reported_as_conflicts(monkeypatch, tmp_path):
    """告警要稀有才有信噪比。"""
    root = _make_root(tmp_path / "deploy", "BLADEX_PORT=39999\n")
    monkeypatch.setenv("BLADEX_PORT", "39999")
    assert deployment.load_env_file(str(root)) == []


def test_orchestrator_env_wins_by_default(monkeypatch, tmp_path):
    """容器场景：compose 用 `environment:` 覆盖容器内路径，同时挂了宿主的 `.env`。

    默认就该是环境赢——否则宿主那份相对路径会把容器里的绝对路径清掉
    （这正是 compose 里那两行 `BLADEX_ENV_PRECEDENCE: "env"` 原本要防的事，
    默认翻回来之后它们成了冗余的显式声明，留着无害）。
    """
    root = _make_root(tmp_path / "deploy", "BLADEX_ROCKSDB_PATH=data/bladex_hub\n")
    monkeypatch.setenv("BLADEX_ROCKSDB_PATH", "/app/data/bladex_hub")

    conflicts = deployment.load_env_file(str(root))

    assert os.environ["BLADEX_ROCKSDB_PATH"] == "/app/data/bladex_hub"
    assert len(conflicts) == 1 and not conflicts[0].applied


def test_env_keep_pins_individual_keys(monkeypatch, tmp_path):
    """`BLADEX_ENV_KEEP` 在 precedence=file 下保住个别键（它只在文件赢时有意义）。"""
    root = _make_root(tmp_path / "deploy", "BLADEX_PORT=1\nBLADEX_HOST=1.2.3.4\n")
    monkeypatch.setenv("BLADEX_PORT", "2")
    monkeypatch.setenv("BLADEX_HOST", "9.9.9.9")
    monkeypatch.setenv("BLADEX_ENV_PRECEDENCE", "file")
    monkeypatch.setenv("BLADEX_ENV_KEEP", "BLADEX_HOST")

    deployment.load_env_file(str(root))

    assert os.environ["BLADEX_PORT"] == "1"        # 文件赢（显式 precedence=file）
    assert os.environ["BLADEX_HOST"] == "9.9.9.9"  # 被显式保住


def test_the_file_cannot_rewrite_how_it_is_loaded(monkeypatch, tmp_path):
    """`.env` 声明 BLADEX_HOME / PRECEDENCE / KEEP 不得覆盖环境里的值——
    否则"配置怎么加载"变成自指，出问题时无从下手。"""
    root = _make_root(tmp_path / "deploy", (
        "BLADEX_HOME=/somewhere/else\n"
        "BLADEX_ENV_PRECEDENCE=file\n"
    ))
    monkeypatch.setenv("BLADEX_HOME", "/real/home")
    monkeypatch.setenv("BLADEX_ENV_PRECEDENCE", "env")

    deployment.load_env_file(str(root))

    assert os.environ["BLADEX_HOME"] == "/real/home"
    assert os.environ["BLADEX_ENV_PRECEDENCE"] == "env"


def test_precedence_can_be_declared_in_the_file_when_env_is_silent(monkeypatch, tmp_path):
    """环境里没声明时，文件里的 precedence 生效（部署目录自带语义）。"""
    root = _make_root(tmp_path / "deploy", (
        "BLADEX_ENV_PRECEDENCE=file\nBLADEX_PORT=1\n"
    ))
    monkeypatch.setenv("BLADEX_PORT", "2")
    conflicts = deployment.load_env_file(str(root))
    assert os.environ["BLADEX_PORT"] == "1"
    assert conflicts[0].applied


def test_default_precedence_is_env(monkeypatch, tmp_path):
    """默认方向单独钉一条——它被翻过一次，值得有一条只讲这件事的测试。"""
    root = _make_root(tmp_path / "deploy", "BLADEX_PORT=1\n")
    monkeypatch.setenv("BLADEX_PORT", "2")
    conflicts = deployment.load_env_file(str(root))
    assert os.environ["BLADEX_PORT"] == "2"
    assert len(conflicts) == 1 and not conflicts[0].applied


# ── 报告：必须有声音，且不能泄密 ────────────────────────────────────────────


def test_secrets_are_masked_in_reports():
    """冲突报告会进日志和终端 —— API key 不能原样吐出来。"""
    assert deployment.mask("BLADEX_UPSTREAM_API_KEY", "sk-abcdef123456") == "sk-a***(15 chars)"
    assert deployment.mask("BLADEX_CLIENT_KEYS", "short") == "***(5 chars)"
    assert deployment.mask("BLADEX_ADMIN_TOKEN", "tok-1234567890") == "tok-***(14 chars)"
    # 非密钥原样显示，否则报告没法用来核对
    assert deployment.mask("BLADEX_PORT", "38080") == "38080"
    assert deployment.mask("BLADEX_UPSTREAM_API_BASE", "https://x/v1") == "https://x/v1"


def test_report_names_the_file_and_the_shadowed_keys():
    """被环境遮蔽 = 配置文件没生效，措辞必须说清后果 + 怎么办。"""
    shadowed = [deployment.EnvConflict("BLADEX_PORT", "2", "1", applied=False)]
    lines = "\n".join(deployment.format_conflicts(shadowed, "/d/config/.env"))
    assert "NOT in effect" in lines and "BLADEX_PORT" in lines and "/d/config/.env" in lines
    assert "Unset" in lines  # 给出下一步该敲什么

    applied = [deployment.EnvConflict("BLADEX_PORT", "2", "1", applied=True)]
    lines = "\n".join(deployment.format_conflicts(applied, "/d/config/.env"))
    assert "overrode a stale value" in lines

    assert deployment.format_conflicts([], "/d/config/.env") == []


def test_report_never_leaks_a_secret_value():
    """🔴 端到端：格式化后的整段文本里不许出现明文密钥。"""
    conflicts = [deployment.EnvConflict(
        "BLADEX_UPSTREAM_API_KEY", "sk-old-secret-aaa", "sk-new-secret-bbb", applied=True)]
    lines = "\n".join(deployment.format_conflicts(conflicts, "/d/config/.env"))
    assert "sk-old-secret-aaa" not in lines
    assert "sk-new-secret-bbb" not in lines


# ── 三份实现收敛：谁都不许再自己写一遍 ──────────────────────────────────────


def test_no_module_reimplements_env_parsing():
    """cli / consolidator / admin_read 必须走 deployment.py，否则又会各自漂移。

    源码级守卫而非 import 后反射：这三个模块拖着 fastapi/rocksdict 一堆重依赖，
    而这条规则跟运行时状态无关。同款做法见 test_user_facing_english 的 AST 守卫。
    """
    import io
    import tokenize
    from pathlib import Path

    def code_only(path: Path) -> str:
        """丢掉注释与字符串 —— 这几处的**注释里**正记着旧实现长什么样。"""
        out: list[str] = []
        with open(path, "rb") as f:
            for tok in tokenize.tokenize(io.BytesIO(f.read()).readline):
                if tok.type not in (tokenize.COMMENT, tokenize.STRING):
                    out.append(tok.string)
        return " ".join(out)

    pkg = Path(__file__).resolve().parents[1] / "bladex_proxy"
    for name in ("cli.py", "consolidator.py", "admin_read.py"):
        src = (pkg / name).read_text(encoding="utf-8")
        code = code_only(pkg / name)
        assert "from bladex_proxy import deployment" in src, f"{name} 没走 deployment.py"
        # 自己按行 partition 解析 .env 的老实现（三份漂移的形状）。
        # 🔴 判据收紧到 `.partition(`（2026-08-25）：裸 "partition" 会误伤任何
        # 无关的字符串切分——V-A6 在 cli.py 加了 `address.rpartition(":")`
        # 拆 host:port，与 .env 解析毫无关系却把守卫打红。
        # 守卫要认的是"按 `=` 切 env 行"这件事，不是 partition 这个词。
        assert '.partition("=")' not in code and ".partition('=')" not in code, \
            f"{name} 又自己解析了一遍 .env"
        # 源码树相对路径找部署根：装成 wheel 后指向 site-packages
        assert "parents [ 3 ]" not in code, f"{name} 又用源码树相对路径找部署根"


def test_consolidator_honours_bladex_home(monkeypatch, tmp_path):
    """此前 consolidator 只认 cwd —— 全局安装后它永远读不到 .env。"""
    root = _make_root(tmp_path / "deploy", "BLADEX_INDEX_PATH=/from/file\n")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.setenv("BLADEX_HOME", str(root))
    monkeypatch.delenv("BLADEX_INDEX_PATH", raising=False)

    from bladex_proxy import consolidator

    consolidator._load_env_file()
    assert os.environ["BLADEX_INDEX_PATH"] == "/from/file"


# ── status 不许静默变短 ─────────────────────────────────────────────────────


def test_status_explains_a_missing_admin_section(monkeypatch, tmp_path, capsys):
    """🔴 2026-08-06 用户现象：在别的目录跑 `bladex status`，输出短了一半。

    `if admin:` 读不到就整段跳过 —— Memory Hub / Index / Queue / Disk / Memory
    五行凭空消失，没有任何解释，用户的第一反应是"程序版本不对"。
    缺席比报错更难查：它不指向任何东西。
    """
    from bladex_proxy import cli

    monkeypatch.chdir(tmp_path)          # 没有 config/.env 的目录
    monkeypatch.setenv("HOME", str(tmp_path / "fake_home"))
    monkeypatch.delenv("BLADEX_ADMIN_KEYS", raising=False)
    monkeypatch.delenv("BLADEX_CLIENT_KEYS", raising=False)

    cli._print_admin_unavailable()

    out = capsys.readouterr().out
    assert "unavailable" in out
    assert "config/.env" in out and "BLADEX_HOME" in out  # 说清成因与下一步


def test_status_distinguishes_missing_config_from_missing_key(monkeypatch, tmp_path, capsys):
    """找到了部署根、但里面没配 key —— 成因不同，话术也得不同。"""
    from bladex_proxy import cli

    root = _make_root(tmp_path / "deploy", "BLADEX_PORT=1\n")
    monkeypatch.setenv("BLADEX_HOME", str(root))
    monkeypatch.delenv("BLADEX_ADMIN_KEYS", raising=False)
    monkeypatch.delenv("BLADEX_CLIENT_KEYS", raising=False)

    cli._print_admin_unavailable()

    out = capsys.readouterr().out
    assert "BLADEX_ADMIN_KEYS" in out and str(root) in out
