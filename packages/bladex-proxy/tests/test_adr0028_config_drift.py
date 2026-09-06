"""运行时记忆配置的可见性 + 配置漂移告警（2026-08-05 事故驱动）。

## 事故（同一类踩了两次）

- 08-04：`flags.py` 默认值改了、`config/.env.example` 没跟上 → 发出去的是旧行为；
- 08-05：`config/.env` 写 `BLADEX_HOTPATH_BUDGET_MS=200`，而**运行中的进程**用 150
  （那个终端里残留了一个 export，`_load_env_file` 的语义是"不覆盖已设环境变量"）。
  每轮注入超时 1–3ms 被整段丢弃 —— 跨 profile 记忆全都算出来了却一条没送出去，
  而唯一的痕迹是 warning 级日志混在一堆 info 里。

两次的共同点：**记忆静默失效，没有任何地方喊出来**。

## 这组测试钉的东西

`/admin/status` 必须暴露**进程实际在用的**值（不是"把 .env 读出来再打印一遍"——
那正是上次骗过所有人的绿灯），并在与 `.env` 声明不符时给出可读的漂移清单。
"""

from __future__ import annotations

import pytest
from bladex_proxy.admin_read import _memory_config
from bladex_proxy.cli import _print_memory_config
from bladex_proxy.config import ProxyConfig


def _stub_declared_env(monkeypatch, mapping: dict[str, str]) -> None:
    """让漂移检测用这份"声明值"，而非 live config/.env 的真实内容。

    这组测试钉的是**比对逻辑**（进程值 vs 声明值），不是文件解析；后者由
    `test_reading_env_file_does_not_pollute_process_env` 间接覆盖。直接 stub
    避免 08-05 事故快照值（200）被当成永恒常量绑定到 live 磁盘文件上——
    live `.env` 是用户配置（不入 git、会变），绑死它正是这组测试本要揭露的
    "绿灯骗人"形态。
    """
    monkeypatch.setattr(
        "bladex_proxy.admin_read._declared_env",
        lambda root=None: dict(mapping),
    )


def test_effective_values_come_from_the_process_not_the_file(monkeypatch):
    """🔴 核心断言：报的是进程实际生效的值。

    环境变量压过 .env（这正是事故成因），所以这里设了环境变量之后，
    `effective` 必须跟着变——如果它还等于 .env 里的数，说明又在读文件糊弄人。
    """
    monkeypatch.setenv("BLADEX_HOTPATH_BUDGET_MS", "150")
    mc = _memory_config(ProxyConfig())
    assert mc["effective"]["BLADEX_HOTPATH_BUDGET_MS"] == 150


def test_drift_detected_when_process_differs_from_env_file(monkeypatch):
    """复刻今天的事故：.env 写 200、进程 150 → 必须出现在 drift 里。"""
    _stub_declared_env(monkeypatch, {"BLADEX_HOTPATH_BUDGET_MS": "200"})
    monkeypatch.setenv("BLADEX_HOTPATH_BUDGET_MS", "150")
    mc = _memory_config(ProxyConfig())
    keys = [d["key"] for d in mc["drift"]]
    assert "BLADEX_HOTPATH_BUDGET_MS" in keys
    entry = next(d for d in mc["drift"] if d["key"] == "BLADEX_HOTPATH_BUDGET_MS")
    assert entry["effective"] == 150
    assert str(entry["declared"]) == "200"


def test_no_drift_when_process_matches_env_file(monkeypatch):
    """一致时不报——告警要稀有才有信噪比。"""
    _stub_declared_env(monkeypatch, {"BLADEX_HOTPATH_BUDGET_MS": "200"})
    monkeypatch.setenv("BLADEX_HOTPATH_BUDGET_MS", "200")
    mc = _memory_config(ProxyConfig())
    assert [d["key"] for d in mc["drift"]] == []


def test_flag_drift_detected(monkeypatch):
    """开关同样比对：.env 写 1、进程被关掉 → 漂移。

    🔴 2026-09-05（G3 公开树实跑）：本条**原来没 stub 声明值**，隐式依赖 live
    `config/.env` 里恰好写着 `BLADEX_SOFT_SCORING=1` —— 私仓有 `.env` 所以绿，
    公开树只有 `.env.example` ⇒ declared 为空 ⇒ drift 为空 ⇒ 红。
    而 `_stub_declared_env` 的 docstring 就写着为什么不该绑 live 磁盘文件
    （"live .env 是用户配置、不入 git、会变，绑死它正是这组测试本要揭露的
    '绿灯骗人'形态"）——**本条违反了它自己文件里写下的纪律**。改为显式 stub。
    """
    _stub_declared_env(monkeypatch, {"BLADEX_SOFT_SCORING": "1"})
    monkeypatch.setenv("BLADEX_SOFT_SCORING", "0")
    mc = _memory_config(ProxyConfig())
    assert "BLADEX_SOFT_SCORING" in [d["key"] for d in mc["drift"]]


def test_effective_includes_the_things_that_silently_kill_memory(monkeypatch):
    """这几个值定错了都表现为"记忆没生效"，所以必须可见。"""
    for k in ("BLADEX_HOTPATH_BUDGET_MS", "BLADEX_INJECT_TOPK",
              "BLADEX_MIN_IMPORTANCE", "BLADEX_INJECT_MAX_CHARS"):
        monkeypatch.delenv(k, raising=False)
    eff = _memory_config(ProxyConfig())["effective"]
    assert eff["BLADEX_INJECT_TOPK"] == 15         # 2026-08-14 基准标定回调（48%→68%）
    # M0-8（复核 G3）：三时钟（M3）落地前不做重要性驱逐 —— 门槛回 0。
    assert eff["BLADEX_MIN_IMPORTANCE"] == 0.0


def test_flags_snapshot_covers_whole_table():
    from bladex_core.flags import MEMORY_FLAG_DEFAULTS

    assert set(_memory_config(ProxyConfig())["flags"]) == set(MEMORY_FLAG_DEFAULTS)


def test_reading_env_file_does_not_pollute_process_env(monkeypatch):
    """读 .env 来比对时**不许**把它灌进 os.environ。

    否则这条观测本身就变成了污染源（ADR-0027 §5.4 那个反复踩的坑）。
    """
    import os

    monkeypatch.delenv("BLADEX_AUTH_ENABLED", raising=False)
    before = dict(os.environ)
    _memory_config(ProxyConfig())
    assert dict(os.environ) == before


# ── CLI 呈现 ────────────────────────────────────────────────────────────


def test_cli_prints_drift_with_actionable_fix(monkeypatch, capsys):
    _stub_declared_env(monkeypatch, {"BLADEX_HOTPATH_BUDGET_MS": "200"})
    monkeypatch.setenv("BLADEX_HOTPATH_BUDGET_MS", "150")
    _print_memory_config({"memory_config": _memory_config(ProxyConfig())})
    out = capsys.readouterr().out

    assert "hotpath 150ms" in out
    # 用户可见字符串统一英文（ADR-0027 §5.1，AST 守卫钉着）——这条也顺带守住它
    assert "Config drift" in out
    assert ".env says 200, process uses 150" in out
    assert "unset" in out and "restart" in out      # 告诉人怎么修，不只是报错


def test_cli_quiet_when_no_drift(monkeypatch, capsys):
    _stub_declared_env(monkeypatch, {"BLADEX_HOTPATH_BUDGET_MS": "200"})
    monkeypatch.setenv("BLADEX_HOTPATH_BUDGET_MS", "200")
    _print_memory_config({"memory_config": _memory_config(ProxyConfig())})
    out = capsys.readouterr().out
    assert "Memory:" in out
    assert "Config drift" not in out


def test_cli_lists_disabled_flags(monkeypatch, capsys):
    monkeypatch.setenv("BLADEX_HOP_EXPAND", "0")
    _print_memory_config({"memory_config": _memory_config(ProxyConfig())})
    assert "hop_expand" in capsys.readouterr().out


@pytest.mark.parametrize("admin", [None, {}, {"memory_config": {}}])
def test_cli_tolerates_old_proxy_without_the_field(admin, capsys):
    """滚动升级：老 proxy 的 /admin/status 没有这个字段 → 安静跳过，不崩。"""
    _print_memory_config(admin)
    assert capsys.readouterr().out == ""
