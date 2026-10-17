"""批 Z1 · Z1.2 —— MQ-W18：硬规则热加载（三个进程、两个来源、后写者赢）。

判据（提示词 §1 Z1.2 + §4.2）：
  ① 改 `.env` 的 BLADEX_HARD_RULES 不重启 ⇒ 下一周期生效，打 `hard_rules_reloaded`；
  ② 解析只有一个实现点（`hard_rules.parse_hard_rules_env`），三个进程共用；
  ③ 坏输入 ⇒ 保留旧规则 + 响亮告警，不许清空；
  ④ 阴性对照：关掉重载（BLADEX_HARD_RULES_RELOAD_S=0）⇒ ① 必须红；
  ⑥ `.env` ⊕ Hub `hard_rules_set` 后写者赢，生效打 `source=env|hub`；
  ⑧ Hub 坏载荷 ⇒ 保留旧规则 + 告警。
三进程的真进程实测见结算稿 `report-batch-Z1-20260923.md` §2。
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

import pytest
from bladex_proxy import hard_rules as hr
from bladex_proxy.config import ProxyConfig

_PKG = Path(__file__).resolve().parents[1] / "bladex_proxy"


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("BLADEX_HARD_RULES", raising=False)
    hr.disarm()
    yield
    hr.disarm()


def _root(tmp_path: Path, value: str | None, extra: str = "") -> Path:
    (tmp_path / "config").mkdir(exist_ok=True)
    lines = ['export BLADEX_PORT="18989"']
    if value is not None:
        lines.append(f'export BLADEX_HARD_RULES="{value}"')
    (tmp_path / "config" / ".env").write_text("\n".join(lines) + "\n" + extra, encoding="utf-8")
    return tmp_path


def _edit_env(root: Path, value: str, bump: float = 2.0) -> None:
    p = root / "config" / ".env"
    text = re.sub(r'export BLADEX_HARD_RULES="[^"\n]*"?', f'export BLADEX_HARD_RULES="{value}"',
                  p.read_text(encoding="utf-8"))
    p.write_text(text, encoding="utf-8")
    st = p.stat()
    os.utime(p, (st.st_atime, st.st_mtime + bump))  # 保证 mtime 前进（粗粒度文件系统）


class _Hub:
    def __init__(self) -> None:
        self.hit: hr.HubRules | None = None

    def set(self, rules, at: float, key: str = "") -> None:
        self.hit = hr.HubRules(key=key or f"admin_event/{at}", at=at,
                               payload={"rules": rules, "origin": "obsidian"})

    def __call__(self):
        return self.hit


def _reloader(root: Path, hub=None, initial=None) -> hr.HardRulesReloader:
    return hr.HardRulesReloader(
        process="test", env_path=str(root / "config" / ".env"),
        seen_path=str(root / "data" / hr.SEEN_FILENAME), hub=hub,
        initial=initial if initial is not None else ["A"])


def test_parse_is_the_single_implementation():
    """判据②：`||` 解析全仓只此一处；ProxyConfig 的启动读也走它。"""
    assert hr.parse_hard_rules_env(" MUST a || NEVER b ||  ") == ["MUST a", "NEVER b"]
    offenders = []
    for f in _PKG.rglob("*.py"):
        src = f.read_text(encoding="utf-8")
        if re.search(r"""split\(\s*["']\|\|["']\s*\)""", src) and "BLADEX_HARD_RULES" in src \
                and f.name != "hard_rules.py":
            offenders.append(f.relative_to(_PKG).as_posix())
    assert offenders == [], offenders
    cfg_src = (_PKG / "config.py").read_text(encoding="utf-8")
    assert '_parse_hard_rules_env(_env("BLADEX_HARD_RULES"' in cfg_src


def test_env_edit_takes_effect_without_restart(tmp_path, capsys):
    """判据①：同一个 reloader 实例（= 不重启）下一次对账就换成新规则，打 hard_rules_reloaded。"""
    root = _root(tmp_path, "MUST a||NEVER b")
    r = _reloader(root, initial=["MUST a", "NEVER b"])
    r.refresh()
    assert r.rules == ["MUST a", "NEVER b"]
    _edit_env(root, "MUST a||NEVER b||MUST c")
    assert r.refresh() is True
    assert r.rules == ["MUST a", "NEVER b", "MUST c"]
    out = capsys.readouterr().out
    assert "hard_rules_reloaded" in out and "source=env" in out
    assert "old=2" in out and "new=3" in out and "process=test" in out


def test_other_key_edit_does_not_count_as_rules_change(tmp_path):
    """改 .env 里别的键（mtime 变、规则值没变）不许刷新 env 的"变化时刻"——否则会盖掉 Obsidian。"""
    root = _root(tmp_path, "MUST a")
    hub = _Hub()
    r = _reloader(root, hub=hub, initial=["MUST a"])
    r.refresh()
    hub.set(["MUST from obsidian"], at=time.time() + 10)
    r.refresh()
    assert r.rules == ["MUST from obsidian"] and r.source == "hub"
    p = root / "config" / ".env"
    p.write_text(p.read_text(encoding="utf-8") + 'export BLADEX_PORT="18990"\n', encoding="utf-8")
    st = p.stat()
    os.utime(p, (st.st_atime, st.st_mtime + 100))
    r.refresh()
    assert r.rules == ["MUST from obsidian"], "改别的键把 Obsidian 的规则盖回去了"


def test_later_writer_wins_both_directions(tmp_path, capsys):
    """判据⑥：Hub 事件晚于 .env 值变化 ⇒ hub；之后再改 .env ⇒ env 赢回来。"""
    root = _root(tmp_path, "MUST env1")
    hub = _Hub()
    r = _reloader(root, hub=hub, initial=["MUST env1"])
    r.refresh()
    env_at = (root / "config" / ".env").stat().st_mtime
    hub.set(["MUST hub1"], at=env_at + 5)
    r.refresh()
    assert (r.rules, r.source) == (["MUST hub1"], "hub")
    _edit_env(root, "MUST env2", bump=30)
    r.refresh()
    assert (r.rules, r.source) == (["MUST env2"], "env")
    out = capsys.readouterr().out
    assert "source=hub" in out and "source=env" in out


def test_older_hub_event_does_not_override_newer_env(tmp_path):
    root = _root(tmp_path, "MUST env")
    hub = _Hub()
    env_at = (root / "config" / ".env").stat().st_mtime
    hub.set(["MUST stale hub"], at=env_at - 3600)
    r = _reloader(root, hub=hub, initial=["MUST env"])
    r.refresh()
    assert (r.rules, r.source) == (["MUST env"], "env")


def test_change_time_survives_restart_via_fingerprint_ledger(tmp_path):
    """重启后（新实例）env 的变化时刻取账本里记的，而不是当前 mtime——Hub 较新时仍是 hub 赢。"""
    root = _root(tmp_path, "MUST env")
    hub = _Hub()
    r1 = _reloader(root, hub=hub, initial=["MUST env"])
    r1.refresh()
    env_at = (root / "config" / ".env").stat().st_mtime
    hub.set(["MUST hub"], at=env_at + 5)
    p = root / "config" / ".env"
    p.write_text(p.read_text(encoding="utf-8") + "# unrelated edit\n", encoding="utf-8")
    st = p.stat()
    os.utime(p, (st.st_atime, env_at + 100))  # 别的改动把 mtime 推到 Hub 事件之后
    r2 = _reloader(root, hub=hub, initial=["MUST env"])  # = 进程重启
    r2.refresh()
    assert (r2.rules, r2.source) == (["MUST hub"], "hub")


@pytest.mark.parametrize("bad", ["", "||"])
def test_empty_env_value_keeps_old_rules(tmp_path, capsys, bad):
    """判据③：把那一行清空 ⇒ 保留旧规则 + WARNING，不降级到零条也不回默认。"""
    root = _root(tmp_path, "MUST a||NEVER b")
    r = _reloader(root, initial=["MUST a", "NEVER b"])
    r.refresh()
    _edit_env(root, bad)
    assert r.refresh() is False
    assert r.rules == ["MUST a", "NEVER b"]
    out = capsys.readouterr().out
    assert "hard_rules_reload_rejected" in out and "reason=empty" in out


def test_unbalanced_quotes_keep_old_rules(tmp_path, capsys):
    root = _root(tmp_path, "MUST a")
    r = _reloader(root, initial=["MUST a"])
    r.refresh()
    p = root / "config" / ".env"
    p.write_text('export BLADEX_HARD_RULES="MUST broken\n', encoding="utf-8")
    st = p.stat()
    os.utime(p, (st.st_atime, st.st_mtime + 2))
    r.refresh()
    assert r.rules == ["MUST a"]
    assert "reason=unbalanced_quotes" in capsys.readouterr().out


def test_deleted_line_keeps_old_rules(tmp_path, capsys):
    root = _root(tmp_path, "MUST a")
    r = _reloader(root, initial=["MUST a"])
    r.refresh()
    p = root / "config" / ".env"
    p.write_text('export BLADEX_PORT="18989"\n', encoding="utf-8")
    st = p.stat()
    os.utime(p, (st.st_atime, st.st_mtime + 2))
    r.refresh()
    assert r.rules == ["MUST a"]
    assert "reason=empty" in capsys.readouterr().out


@pytest.mark.parametrize("payload", [{"rules": []}, {"rules": "MUST a"}, {"rules": [1, 2]},
                                     {}, {"rules": ["  "]}])
def test_bad_hub_payload_keeps_old_rules(tmp_path, capsys, payload):
    """判据⑧：Hub 事件载荷坏 ⇒ 保留旧规则 + 告警。"""
    root = _root(tmp_path, "MUST a")
    hub = _Hub()
    r = _reloader(root, hub=hub, initial=["MUST a"])
    r.refresh()
    hub.hit = hr.HubRules(key="admin_event/x", at=time.time() + 60, payload=payload)
    r.refresh()
    assert r.rules == ["MUST a"]
    out = capsys.readouterr().out
    assert "hard_rules_reload_rejected" in out and "reason=bad_payload" in out


def test_env_shadowed_by_process_env_warns_and_ignores_file(tmp_path, monkeypatch, capsys):
    root = _root(tmp_path, "MUST file")
    monkeypatch.setenv("BLADEX_HARD_RULES", "MUST shell")
    monkeypatch.setenv("BLADEX_ENV_PRECEDENCE", "env")
    r = _reloader(root, initial=["MUST shell"])
    r.refresh()
    _edit_env(root, "MUST file2")
    r.refresh()
    assert r.rules == ["MUST shell"]
    assert "hard_rules_env_shadowed" in capsys.readouterr().out


# ── 生效面：ProxyConfig.effective_hard_rules 是唯一咽喉 ───────────────────────


def test_armed_process_reads_hot_value_through_proxyconfig(tmp_path, capsys):
    """arm + 真守护线程：改 .env 后 `ProxyConfig().effective_hard_rules` 在一个周期内变。"""
    root = _root(tmp_path, "MUST a")
    cfg = ProxyConfig(hard_rules=["MUST a"])
    hr.arm("proxy", root=str(root), interval_s=0.05, initial=cfg.effective_hard_rules)
    assert cfg.effective_hard_rules == ["MUST a"]
    _edit_env(root, "MUST a||NEVER z")
    assert hr.wait_until(lambda: cfg.effective_hard_rules == ["MUST a", "NEVER z"], 3.0)
    assert "hard_rules_reloaded" in capsys.readouterr().out


def test_negative_control_reload_disabled_means_no_hot_value(tmp_path, monkeypatch):
    """判据④ 🔴：BLADEX_HARD_RULES_RELOAD_S=0 ⇒ arm 不启用，改 .env 不生效（①必须红）。"""
    root = _root(tmp_path, "MUST a")
    monkeypatch.setenv("BLADEX_HARD_RULES_RELOAD_S", "0")
    cfg = ProxyConfig(hard_rules=["MUST a"])
    assert hr.arm("proxy", root=str(root), initial=cfg.effective_hard_rules) is None
    _edit_env(root, "MUST a||NEVER z")
    time.sleep(0.3)
    assert cfg.effective_hard_rules == ["MUST a"]


def test_startup_applies_existing_hub_event(tmp_path, capsys):
    """Z4 已落 Hub 的 Obsidian 改动：arm 时同步生效（此前"没有任何进程读它"）。"""
    root = _root(tmp_path, "MUST env")
    hub = _Hub()
    hub.set(["MUST obsidian"], at=time.time() + 60)
    hr.arm("consolidator", root=str(root), hub=hub, interval_s=60, initial=["MUST env"])
    assert ProxyConfig().effective_hard_rules == ["MUST obsidian"]
    out = capsys.readouterr().out
    assert "hard_rules_reloaded" in out and "source=hub" in out and "process=consolidator" in out


def test_on_change_callback_fires(tmp_path):
    root = _root(tmp_path, "MUST a")
    seen: list = []
    hr.arm("flash_daemon", root=str(root), interval_s=0.05, initial=["MUST a"],
           on_change=seen.append)
    _edit_env(root, "MUST b")
    assert hr.wait_until(lambda: seen == [["MUST b"]], 3.0)


def test_hub_reader_on_real_memory_hub(tmp_path):
    """`latest_admin_event` 反向 seek 取最近一条 hard_rules_set（夹着别的事件类型）。"""
    from bladex_proxy.models import AdminEventType
    from bladex_proxy.storage.memory_hub import MemoryHub
    hub = MemoryHub(str(tmp_path / "hub"))
    hub.open()
    try:
        assert hr.hub_reader(hub)() is None
        hub.append_admin_event(AdminEventType.HARD_RULES_SET, "", "", rules=["MUST 1"],
                               previous=[], origin="obsidian", submitter="x")
        hub.append_admin_event(AdminEventType.HARD_RULES_SET, "", "", rules=["MUST 2"],
                               previous=["MUST 1"], origin="obsidian", submitter="x")
        hub.append_admin_event(AdminEventType.FACT_CORRECT, "", "f1", content="c")
        hit = hr.hub_reader(hub)()
        assert hit is not None and hit.payload["rules"] == ["MUST 2"]
        assert abs(hit.at - time.time()) < 60
    finally:
        hub.close()


# ── 三个进程的接线守卫（漏一个进程 = 未兑现）──────────────────────────────────


@pytest.mark.parametrize("path,process,hub_expr", [
    ("server/app_factory.py", "proxy", "hub_reader(hub)"),
    ("flash_daemon.py", "flash_daemon", "hub_reader(hub, catch_up=True)"),
    ("consolidator.py", "consolidator", "hub_reader(ledger, catch_up=True)"),
])
def test_each_process_arms_reload_with_its_hub(path, process, hub_expr):
    src = (_PKG / path).read_text(encoding="utf-8")
    assert f'arm("{process}"' in src, f"{path} 没有 arm 硬规则热加载"
    seg = src.split(f'arm("{process}"', 1)[1][:300]
    assert hub_expr in seg, f"{path} 的 arm 没接 Hub（Obsidian 改动读不到）"


def test_hard_rule_readers_go_through_effective_hard_rules():
    """三个进程读硬规则都经 `effective_hard_rules`（热值咽喉）；不许直读 `cfg.hard_rules`。"""
    offenders = []
    for f in _PKG.rglob("*.py"):
        if f.name in ("config.py", "hard_rules.py"):
            continue
        src = f.read_text(encoding="utf-8")
        if re.search(r"\bcfg\.hard_rules\b|\bconfig\.hard_rules\b", src):
            offenders.append(f.relative_to(_PKG).as_posix())
    assert offenders == [], offenders


# ── ⑦ 生效后回写导出目标（consolidator 进程的消费面）────────────────────────────


class _MockExporter:
    def __init__(self) -> None:
        self.name, self.exposure = "mock", "private"
        self.batches: list = []

    def sync_incremental(self, batch, since_cursor) -> None:
        self.batches.append(batch)

    def handle_tombstone(self, target_type, target_key) -> None:
        pass


@pytest.fixture()
def _stores(tmp_path):
    from bladex_proxy.storage.memory_hub import MemoryHub
    from bladex_proxy.storage.memory_index import MemoryIndex
    index = MemoryIndex(tmp_path / "index", embedder=None, read_only=False)
    index.open()
    hub = MemoryHub(tmp_path / "hub")
    hub.open()
    yield index, hub
    index.close()
    hub.close()


def test_worker_reexports_hard_rules_when_effective_set_changes(_stores, tmp_path, capsys):
    """⑦：规则经可调用对象每轮现取；变了 ⇒ 下一轮带上整套规则（游标非空也带），没变 ⇒ 不带。"""
    from bladex_proxy.export_sync import ExporterConfig, ExportSyncWorker
    index, hub = _stores
    current = {"rules": ["MUST a"]}
    exp = _MockExporter()
    worker = ExportSyncWorker(index, hub, [(ExporterConfig(type="mock", name="mock",
                                                           exposure="private"), exp)],
                              tmp_path / "state.json", hard_rules=lambda: current["rules"])
    worker.run_once()
    assert exp.batches[-1].hard_rules == ["MUST a"]
    n = len(exp.batches)
    worker.run_once()
    assert len(exp.batches) == n, "规则没变就不该再推"
    current["rules"] = ["MUST a", "NEVER b"]
    worker.run_once()
    assert exp.batches[-1].hard_rules == ["MUST a", "NEVER b"]
    assert "export_hard_rules_resync" in capsys.readouterr().out
    # 重启（新 worker 读同一状态文件）：指纹已记账 ⇒ 不重复推
    worker2 = ExportSyncWorker(index, hub, [(ExporterConfig(type="mock", name="mock",
                                                            exposure="private"), exp)],
                               tmp_path / "state.json", hard_rules=lambda: current["rules"])
    n = len(exp.batches)
    worker2.run_once()
    assert len(exp.batches) == n


def test_negative_control_static_list_never_reexports(_stores, tmp_path):
    """阴性对照：传静态列表（= 没接热加载）⇒ 首轮之后永不重推——⑦ 的承重点是可调用对象。"""
    from bladex_proxy.export_sync import ExporterConfig, ExportSyncWorker
    index, hub = _stores
    rules = ["MUST a"]
    exp = _MockExporter()
    worker = ExportSyncWorker(index, hub, [(ExporterConfig(type="mock", name="mock",
                                                           exposure="private"), exp)],
                              tmp_path / "state.json", hard_rules=list(rules))
    worker.run_once()
    rules.append("NEVER b")  # 外面改了，worker 手里是副本
    n = len(exp.batches)
    worker.run_once()
    assert len(exp.batches) == n


def test_obsidian_page_rewritten_without_echo(tmp_path):
    """⑦：vault 页按新规则重写一次；重写后回流扫描**零**编辑（自写快照，不产生回声事件）。"""
    from bladex_core.exporters import ExportBatch
    from bladex_proxy.exporters.obsidian import ObsidianExporter, parse_hard_rules_page
    root = tmp_path / "vault" / "BladeX"
    exp = ObsidianExporter("v", "private", {"vault_dir": str(root)})
    exp.sync_incremental(ExportBatch(hard_rules=["MUST a"]), "")
    assert exp.collect_edits(5) == []
    exp.sync_incremental(ExportBatch(hard_rules=["MUST a", "NEVER b"]), "c1")
    page = root / "Hard Rules.md"
    assert parse_hard_rules_page(page.read_text(encoding="utf-8")) == ["MUST a", "NEVER b"]
    assert exp.collect_edits(5) == [], "导出器自己重写的页被当成了用户编辑（回声）"


def test_obsidian_page_already_matching_is_not_rewritten(tmp_path):
    """用户在 Obsidian 改的规则回流生效后，worker 重导同一套规则 ⇒ 不重写（保住用户排版）。"""
    from bladex_core.exporters import ExportBatch
    from bladex_proxy.exporters.obsidian import ObsidianExporter
    root = tmp_path / "vault" / "BladeX"
    exp = ObsidianExporter("v", "private", {"vault_dir": str(root)})
    exp.sync_incremental(ExportBatch(hard_rules=["MUST a"]), "")
    page = root / "Hard Rules.md"
    custom = "---\nbladex_id: hard-rules\n---\n\n# My rules\n\nintro text\n\n* MUST a\n* NEVER b\n"
    page.write_text(custom, encoding="utf-8")
    edits = exp.collect_edits(5)
    assert [e.rules for e in edits] == [["MUST a", "NEVER b"]]
    from bladex_proxy.export_backflow import BackflowResult
    exp.ack(edits[0], BackflowResult("accepted"))
    exp.sync_incremental(ExportBatch(hard_rules=["MUST a", "NEVER b"]), "c1")
    assert page.read_text(encoding="utf-8") == custom
    assert exp.collect_edits(5) == []


def test_postgres_mirror_replaces_rule_set():
    """⑦ Postgres：重导时先删掉不在新集合里的行再插（镜像 = 生效规则，不是新旧并存）。"""
    from bladex_core.exporters import ExportBatch
    from bladex_proxy.exporters.postgres import PostgresExporter

    log: list = []

    class _Cur:
        def execute(self, sql, params=None):
            log.append((" ".join(sql.split()), params))

    class _Conn:
        def cursor(self):
            return _Cur()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def close(self):
            pass

    exp = PostgresExporter("pg", "private", {"_connect_factory": lambda: _Conn()})
    exp.sync_incremental(ExportBatch(hard_rules=["MUST a", "NEVER b"]), "c1")
    sqls = [s for s, _ in log]
    d = next(i for i, s in enumerate(sqls) if s.startswith("DELETE FROM hard_rules"))
    ins = [i for i, s in enumerate(sqls) if s.startswith("INSERT INTO hard_rules")]
    assert ins and d < min(ins)
    assert log[d][1] == (["MUST a", "NEVER b"],)
    log.clear()
    exp.sync_incremental(ExportBatch(facts=[]), "c2")
    assert not any(s.startswith("DELETE FROM hard_rules") for s, _ in log), "没带规则就不许动表"


def test_reload_wake_reports_once_then_delegates():
    class _Inner:
        def __init__(self):
            self.marked = 0

        def has_change(self):
            return "obsidian"

        def mark(self):
            self.marked += 1

    inner = _Inner()
    w = hr.ReloadWake()
    assert w.has_change() is None
    w.inner = inner
    w.notify(["MUST x"])
    assert w.has_change() == "hard_rules"
    assert w.has_change() == "obsidian"
    w.mark()
    assert inner.marked == 1


def test_consolidator_wires_callable_rules_and_wake():
    src = (_PKG / "consolidator.py").read_text(encoding="utf-8")
    assert "hard_rules=lambda: cfg.effective_hard_rules" in src
    assert "on_change=hr_wake.notify" in src
    assert "external = hr_wake" in src
