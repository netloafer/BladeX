"""C1/C2 验收剧本（ADR-0027 §3.3 契约 C + §4.4 O3）：

  - 降级事件的**滚动窗口**计数（不是自启动以来的累计值——这正是它存在的理由）
  - /admin/status 的 degradation / disk 两个新字段
  - CLI 降级横幅与磁盘行
  - run_*.sh 薄壳化（Y4）：不再各自实现一遍启动逻辑

对应发布就绪批的降级可见性与磁盘治理两项。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from bladex_proxy import admin_read, cli, metrics


# ── DegradationLog：滚动窗口 ────────────────────────────────────────


def test_count_only_within_window():
    log = metrics.DegradationLog()
    now = 10_000.0
    log.record("inject_timeout", now=now - 7200)  # 2h 前
    log.record("inject_timeout", now=now - 1800)  # 30min 前
    log.record("inject_timeout", now=now - 10)
    assert log.count("inject_timeout", 3600.0, now=now) == 2
    assert log.count("inject_timeout", 60.0, now=now) == 1
    # 这是与 Metrics counter 的分界：累计值会报 3，而"最近一小时"是 2
    assert log.count("inject_timeout", 86400.0, now=now) == 3


def test_count_unknown_event_is_zero():
    assert metrics.DegradationLog().count("never_happened", now=1.0) == 0


def test_snapshot_always_carries_known_keys():
    """没发生过也要给 0——消费方拿到稳定 schema。

    "字段不存在"与"字段是 0"在 UI 上是两件事：前者会让横幅逻辑
    走 falsy 分支而看起来"正常"，掩盖的恰恰是数据面没接上。
    """
    snap = metrics.DegradationLog().snapshot(now=1.0)
    for name in metrics.KNOWN_DEGRADATIONS:
        assert snap[f"{name}_1h"] == 0
    assert snap["window_s"] == 3600.0


def test_snapshot_counts_and_custom_events():
    log = metrics.DegradationLog()
    now = 500.0
    for _ in range(3):
        log.record(metrics.INJECT_TIMEOUT, now=now - 5)
    log.record("some_new_event", now=now - 5)
    snap = log.snapshot(now=now)
    assert snap["inject_timeout_1h"] == 3
    assert snap["enqueue_skipped_1h"] == 0
    assert snap["some_new_event_1h"] == 1


def test_bounded_deque_does_not_grow_unbounded():
    log = metrics.DegradationLog(maxlen=10)
    for i in range(50):
        log.record("x", now=1000.0 + i)
    # 满了丢最旧的：少报，但那个量级本身已经是红色，不影响结论
    assert log.count("x", 3600.0, now=1100.0) == 10


def test_record_degradation_never_raises(monkeypatch):
    """打点在降级路径上被调用——它自己绝不能成为新的失败点。"""
    class Boom:
        def record(self, name, **kw):  # noqa: ANN001, ANN201, ARG002
            raise RuntimeError("disk on fire")

    monkeypatch.setattr(metrics, "DEGRADATION", Boom())
    metrics.record_degradation("inject_timeout")  # 不抛就算过


def test_clear_resets():
    log = metrics.DegradationLog()
    log.record("a", now=1.0)
    log.clear()
    assert log.count("a", now=1.0) == 0


# ── 打点接线：真实降级路径确实会记 ───────────────────────────────


def test_inject_timeout_is_recorded(monkeypatch):
    """inject.py 的 TimeoutError 分支必须打点。

    没有这条断言，"降级面"很容易变成一个永远显示 0 的漂亮组件。
    """
    import asyncio

    from bladex_proxy import inject
    from bladex_proxy.config import ProxyConfig

    metrics.DEGRADATION.clear()

    class SlowSource:
        last_facts: list = []
        last_facts_wide: list = []

        def build_facts(self, *a, **kw):  # noqa: ANN002, ANN003, ANN201, ARG002
            import time as _t
            _t.sleep(0.3)
            return ["never gets here"]

    cfg = ProxyConfig()
    cfg.hotpath_budget_ms = 1  # 1ms 预算 -> 必超时
    messages = [{"role": "user", "content": "hi"}]
    # session_id 留空 = 不走三平面（走 build_facts 那条），超时判据不受开关影响
    asyncio.run(inject.do_inject_async(messages, "hi", cfg, source=SlowSource()))

    assert metrics.DEGRADATION.count(metrics.INJECT_TIMEOUT) >= 1


def test_inject_failure_is_recorded_separately(monkeypatch):
    """检索抛异常与检索超时是两种降级，横幅要能分开（修法不同）。"""
    import asyncio

    from bladex_proxy import inject
    from bladex_proxy.config import ProxyConfig

    metrics.DEGRADATION.clear()

    class BrokenSource:
        last_facts: list = []
        last_facts_wide: list = []

        def build_facts(self, *a, **kw):  # noqa: ANN002, ANN003, ANN201, ARG002
            raise RuntimeError("lancedb gone")

    cfg = ProxyConfig()
    messages = [{"role": "user", "content": "hi"}]
    asyncio.run(inject.do_inject_async(messages, "hi", cfg, source=BrokenSource()))

    assert metrics.DEGRADATION.count(metrics.INJECT_FAILED) >= 1
    assert metrics.DEGRADATION.count(metrics.INJECT_TIMEOUT) == 0


# ── 磁盘占用 ───────────────────────────────────────────────────────


def test_dir_bytes_missing_path_is_none(tmp_path: Path):
    """路径不存在 ≠ 0 字节——前者要显示 '-'，否则用户会以为 Memory Hub 是空的。"""
    assert admin_read._dir_bytes(tmp_path / "nope") is None


def test_dir_bytes_sums_recursively(tmp_path: Path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.bin").write_bytes(b"x" * 100)
    (tmp_path / "sub" / "b.bin").write_bytes(b"y" * 50)
    assert admin_read._dir_bytes(tmp_path) == 150


def test_disk_usage_is_cached(tmp_path: Path, monkeypatch):
    """dashboard 每 5s 轮询，RocksDB 目录 walk 一遍不便宜 -> 60s 缓存。"""
    class Cfg:
        rocksdb_path = str(tmp_path / "ledger")
        index_path = str(tmp_path / "index")
        overflow_dir = str(tmp_path / "ov")

    (tmp_path / "ledger").mkdir()
    (tmp_path / "ledger" / "f").write_bytes(b"z" * 10)
    monkeypatch.setattr(admin_read, "_disk_cache", {"at": 0.0, "value": None})
    first = admin_read._disk_usage(Cfg(), now=1000.0)
    assert first["hub_bytes"] == 10
    (tmp_path / "ledger" / "g").write_bytes(b"z" * 90)
    # TTL 内：拿到的还是旧值
    assert admin_read._disk_usage(Cfg(), now=1030.0)["hub_bytes"] == 10
    # TTL 外：重扫
    assert admin_read._disk_usage(Cfg(), now=1100.0)["hub_bytes"] == 100


# ── CLI 呈现 ───────────────────────────────────────────────────────


@pytest.mark.parametrize(("n", "expect"), [
    (None, "-"),
    (0, "0B"),
    (512, "512B"),
    (1536, "1.5KB"),
    (5 * 1024 * 1024, "5.0MB"),
])
def test_fmt_bytes(n, expect):
    assert cli._fmt_bytes(n) == expect


def test_degradation_banner_prints_consequences(capsys):
    cli._print_degradation_banner({
        "degradation": {"inject_timeout_1h": 23, "enqueue_skipped_1h": 4,
                        "index_unavailable_1h": 2},
        "index": {"lag_turns": 900},
    })
    out = capsys.readouterr().out
    # 数字 + 后果 + 修法，三样缺一不可（U8 的判据）
    assert "23 turns degraded on inject timeout" in out
    assert "BLADEX_HOTPATH_BUDGET_MS" in out
    assert "4 turns spilled to disk" in out
    assert "2 lookups hit an unavailable Memory Index layer" in out
    assert "900 turns" in out


def test_degradation_banner_silent_when_clean(capsys):
    cli._print_degradation_banner({
        "degradation": {f"{n}_1h": 0 for n in metrics.KNOWN_DEGRADATIONS},
        "index": {"lag_turns": 3},
    })
    assert capsys.readouterr().out == ""


def test_degradation_banner_tolerates_missing_field(capsys):
    """老版本 proxy（无 degradation 字段）不能让 status 崩。"""
    cli._print_degradation_banner({"index": {}})
    assert capsys.readouterr().out == ""


# ── dashboard ──────────────────────────────────────────────────────


def _dashboard_text() -> str:
    import bladex_proxy as _pkg
    return (Path(_pkg.__file__).parent / "dashboard.html").read_text(encoding="utf-8")   # F0.1 拆包：cli 成了子包，按包根定位


def test_dashboard_renders_degradation_and_disk():
    html = _dashboard_text()
    assert "degradationBanner" in html
    assert "inject_timeout_1h" in html
    assert "Disk usage" in html
    assert "fmtBytes" in html


def test_dashboard_banner_is_escaped():
    """横幅文本进 innerHTML —— 必须过 esc()（§2.3 那条注入路径的同款教训）。"""
    html = _dashboard_text()
    banner = html.split("function degradationBanner")[1].split("\n}")[0]
    assert "esc(x)" in banner


# ── Y4 薄壳化 ──────────────────────────────────────────────────────


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(("script", "delegate"), [
    ("config/run_proxy.sh", "start --foreground"),
    ("config/run_consolidator.sh", "consolidator"),
])
def test_run_scripts_delegate_to_cli(script, delegate):
    text = (_repo_root() / script).read_text(encoding="utf-8")
    assert delegate in text


def _code_lines(path: Path) -> str:
    """去掉注释行——注释里提到旧入口是**说明**，不是重复实现。"""
    return "\n".join(ln for ln in path.read_text(encoding="utf-8").splitlines()
                     if not ln.lstrip().startswith("#"))


def test_run_scripts_no_longer_duplicate_startup_logic():
    """两处实现会漂移，而漂移的那一侧在用户机上悄悄失效（Y4 的理由）。"""
    proxy = _code_lines(_repo_root() / "config/run_proxy.sh")
    cons = _code_lines(_repo_root() / "config/run_consolidator.sh")
    assert "redis-server --port" not in proxy      # Redis 自启只留 CLI 一处
    assert "uvicorn" not in proxy                  # 起服务只留 CLI 一处
    assert "nohup" not in cons                     # 后台化只留 CLI 一处
    assert "run_index_consolidator.py" not in cons    # 入口点已是包内模块


def test_consolidator_command_rejects_unknown_action(capsys):
    rc = cli.consolidator("frobnicate")
    assert rc == 2
    assert "Unknown action" in capsys.readouterr().out


def test_consolidator_status_reports_not_running(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    for k in list(os.environ):
        if k.startswith("BLADEX_"):
            monkeypatch.delenv(k, raising=False)
    from bladex_proxy.cli import lifecycle as _lc   # F0.1 拆包：消费方在 cli/lifecycle.py
    monkeypatch.setattr(_lc, "_consolidator_pid", lambda: None)
    rc = cli.consolidator("status")
    assert rc == 1
    out = capsys.readouterr().out
    assert "not running" in out
    assert "no facts are being distilled" in out  # 后果，不只是状态


# ══════════════════════════════════════════════════════════════════════════
# hidden .pth 检查（2026-08-05 真实事故）
#
# Python 3.11+ 的 site.addpackage() 对带 macOS UF_HIDDEN 的 .pth **静默 return**。
# 后果：editable 装的包全部导不进来，而 dist-info / .pth 内容全都正常，
# `ModuleNotFoundError` 完全指不到真因。触发场景 = 项目放在 iCloud/OneDrive 里。
# ══════════════════════════════════════════════════════════════════════════


def test_hidden_pth_detector_returns_list(monkeypatch, tmp_path: Path):
    """没有 hidden 标志时返回空——不能反过来天天误报。"""
    (tmp_path / "x.pth").write_text("/somewhere\n", encoding="utf-8")
    monkeypatch.setattr("site.getsitepackages", lambda: [str(tmp_path)])
    assert cli._hidden_pth_files() == []


def test_hidden_pth_detector_flags_hidden_file(monkeypatch, tmp_path: Path):
    import stat as _stat
    if not hasattr(_stat, "UF_HIDDEN") or not hasattr(os, "chflags"):
        pytest.skip("平台无 UF_HIDDEN（非 macOS/BSD），本检查恒为空")

    p = tmp_path / "_editable_impl_pkg.pth"
    p.write_text("/somewhere\n", encoding="utf-8")
    os.chflags(p, _stat.UF_HIDDEN)  # type: ignore[attr-defined]
    monkeypatch.setattr("site.getsitepackages", lambda: [str(tmp_path)])
    try:
        assert str(p) in cli._hidden_pth_files()
    finally:
        os.chflags(p, 0)  # type: ignore[attr-defined]


def test_hidden_pth_detector_survives_bad_sitepackages(monkeypatch):
    """解释器布局异常时不能把 doctor 整个带崩。"""
    monkeypatch.setattr("site.getsitepackages", lambda: ["/nonexistent/xyz"])
    assert cli._hidden_pth_files() == []
