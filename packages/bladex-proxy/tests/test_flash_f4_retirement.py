"""批 F4（2026-09-23）退役守卫：目录树 / 名册 / 简介蒸馏删掉之后不许悄悄长回来。

三条判据各钉一格（提示词 §1，dispatch F4 行）：

① **零 Router**：`flash_daemon.py` 全文（含注释）逐字不出现 Router 网关符号。
   反向变异：在 daemon 里加回一次 `router_sdk.completion` ⇒ 本文件 `test_daemon_zero_router` 红。
② **名册退出注入**：flash 根下即使躺着一份旧 `AGENTS.md`，自我介绍块里也没有它；
   `agency_about_injected` 日志不再带 `roster_chars`。
   反向变异：把 `load_agents_roster` 接回 `insert_system_notes` ⇒ `test_stable_block_has_no_roster` 红。
③ **树不再物化**：daemon 一轮跑完，scope 目录下只有账本池与 `behavior.json`，
   也不再生成两份简介缓存。

`behavior.json` 逐字一致（删树前后）是一次性的 live 快照对照，读数在结算稿
`docs/planning/report-batch-F4-20260923.md`，不在单测里（单测里的"前"已经不存在）。
"""

from __future__ import annotations

import importlib.util
import inspect
import os
from pathlib import Path

import structlog.testing
from bladex_core.ledger import Ledger

_DAEMON_SRC = Path(__file__).resolve().parents[1] / "bladex_proxy" / "flash_daemon.py"


# ── ① 零 Router ──────────────────────────────────────────────────────────────


def test_daemon_zero_router():
    """判据原文：`flash_daemon.py` grep `router_` = 0。注释也算——grep 不分注释。"""
    text = _DAEMON_SRC.read_text(encoding="utf-8")
    for sym in ("router_", "litellm", "acompletion"):
        assert sym not in text, f"flash_daemon.py 出现 {sym!r}：daemon 必须全程零 LLM（批 F4）"


def test_summarizers_and_tree_sources_gone():
    import bladex_proxy.flash_daemon as fd
    for name in ("router_summarizer", "router_project_summarizer", "hub_tree_source",
                 "registry_agents_source", "TREE_AUX_SAMPLE_ROUNDS"):
        assert not hasattr(fd, name), f"flash_daemon.{name} 应已随树/简介蒸馏删除"
    for name in ("_materialize_tree", "_materialize_roster", "_materialize_profiles",
                 "_reclaim_dirs", "_refresh_summaries", "_project_summary"):
        assert not hasattr(fd.FlashDaemon, name), f"FlashDaemon.{name} 应已删除"
    params = set(inspect.signature(fd.FlashDaemon.__init__).parameters)
    for gone in ("tree_source", "agents_source", "summarize", "summarize_project",
                 "profiles_source", "tree_every_s"):
        assert gone not in params, f"FlashDaemon 构造参数 {gone} 应已删除"


def test_flash_tree_module_gone():
    assert importlib.util.find_spec("bladex_core.flash_tree") is None, \
        "bladex_core/flash_tree.py 应已删除（项目识别的生产实现在 bladex_proxy.project_identity）"


# ── ② 名册退出注入 ────────────────────────────────────────────────────────────


def _plant_old_roster(root: Path) -> None:
    """模拟生产机：F4 前 daemon 写下的 AGENTS.md 还躺在盘上（旧树目录不删，留给用户）。"""
    d = root / "local" / "personal"
    d.mkdir(parents=True)
    (d / "AGENTS.md").write_text(
        "# Agents\n\n<!-- BladeX Memory Flash · machine-rendered roster -->\n\n"
        "| # | agent_id | name |\n|---|---|---|\n| 1 | `pi` | Pi |\n", encoding="utf-8")


def test_stable_block_has_no_roster(tmp_path, monkeypatch):
    monkeypatch.setenv("BLADEX_FLASH_PATH", str(tmp_path))
    _plant_old_roster(tmp_path)
    from bladex_proxy.agency import AgencyRuntime
    rt = AgencyRuntime(index=None, hub=None)
    rt.about = "BladeX is a memory proxy."
    assert not hasattr(rt, "agents_roster") and not hasattr(rt, "last_roster_chars")
    msgs = [{"role": "system", "content": "agent sys"}, {"role": "user", "content": "hi"}]
    with structlog.testing.capture_logs() as cap:
        out = rt.insert_system_notes(msgs, toolface_injected=True)
    [block] = [m for m in out if "bladex-about" in str(m.get("content"))]
    assert "`pi`" not in block["content"] and "# Agents" not in block["content"], \
        "名册不得再进稳定层（拍板 (16)）"
    assert rt.last_about_chars == len(block["content"]), "about 读数 = 整块（无名册段可扣）"
    [ev] = [e for e in cap if e.get("event") == "agency_about_injected"]
    assert "roster_chars" not in ev, "roster_chars 日志字段已退役（批 F4）"
    assert ev["chars"] == len(block["content"])


# ── ③ 树不再物化 ──────────────────────────────────────────────────────────────


def test_daemon_writes_only_pool_and_behavior(tmp_path):
    from bladex_proxy.flash_daemon import FlashDaemon
    led = Ledger(ledger_id="ldg-f4f4f4f4f4f4", title="退役守卫")
    d = FlashDaemon(root=str(tmp_path), principal="local",
                    ledger_source=lambda: ({led.ledger_id: led}, {}),
                    emit_admin_event=lambda t, p: None,
                    behavior_source=lambda: {"rows": [], "tombstone_hash": ""},
                    behavior_every_s=0.0)
    d.startup_check()
    scope = tmp_path / "local" / "personal"
    assert sorted(os.listdir(scope)) == ["behavior.json", "ledgers"], \
        "scope 目录下只剩账本池与行为画像（无 AGENTS.md / USER.md / RULES.md / agent 子目录）"
    top = set(os.listdir(tmp_path))
    assert not top & {".agent_summaries.json", ".project_summaries.json"}, "简介缓存不再生成"
