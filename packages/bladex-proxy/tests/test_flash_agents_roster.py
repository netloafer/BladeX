"""V-F3 第一批：AGENTS.md 名册——registry 缓存 → daemon 物化 → 稳定层注入。

三段接线各自钉住 + 两条边界：
① unknown-* 指纹桶不进名册（那是待认领噪声，进名册 = 把垃圾注进每轮前缀）；
② 名册是**机器文件**，不进直编捕获（进了就会被 parse_ledger_md 逐轮试啃告警）。
"""

from __future__ import annotations

import json
import os

from bladex_core.ledger import Ledger
from bladex_proxy.flash_daemon import FlashDaemon, registry_agents_source


def _cache(tmp_path, bindings):
    f = tmp_path / "agent_registry.json"
    f.write_text(json.dumps({"version": 1, "bindings": bindings, "pending": [
        {"user_id": "local", "bucket_id": "unknown-9d9bdd3e", "basis": "ua",
         "headers": {}, "first_seen": 1.0, "count": 3, "drift": 0},
    ]}), encoding="utf-8")
    return str(f)


class TestRegistrySource:
    def test_bindings_deduped_and_sorted(self, tmp_path):
        src = registry_agents_source(_cache(tmp_path, [
            {"user_id": "local", "origin_key": "a1", "agent_id": "hermes:default",
             "trigger": "fp", "identified_at": 200.0},
            {"user_id": "local", "origin_key": "a2", "agent_id": "hermes:default",
             "trigger": "fp", "identified_at": 100.0},   # 同 agent 更早 → first_seen
            {"user_id": "local", "origin_key": "b1", "agent_id": "claude-code",
             "trigger": "header", "identified_at": 150.0},
        ]))
        rows = src()
        # 2026-08-29 拍板 v2：profile 变体是独立 agent，名册保留完整 id
        assert [r.agent_id for r in rows] == ["hermes:default", "claude-code"]
        assert rows[0].first_seen == "1970-01-01"
        assert rows[0].last_seen == "" and rows[0].summary == "", \
            "缓存里没有的字段留空——名册诚实于数据源（last_seen 是运行期数据）"

    def test_unknown_buckets_excluded(self, tmp_path):
        src = registry_agents_source(_cache(tmp_path, [
            {"user_id": "local", "origin_key": "x", "agent_id": "unknown-9d9bdd3e",
             "trigger": "", "identified_at": 1.0},
        ]))
        assert src() == []

    def test_missing_cache_is_empty_not_error(self, tmp_path):
        assert registry_agents_source(str(tmp_path / "nope.json"))() == []

    def test_eligible_predicate_filters_subagent_ids(self, tmp_path):
        src = registry_agents_source(_cache(tmp_path, [
            {"user_id": "local", "origin_key": "c1", "agent_id": "codex",
             "trigger": "fp", "identified_at": 1.0},
            {"user_id": "local", "origin_key": "c2", "agent_id": "codex:guardian",
             "trigger": "header:x-openai-subagent", "identified_at": 2.0},
        ]), eligible=lambda a: a == "codex")
        assert [r.agent_id for r in src()] == ["codex"], \
            "判据①：子代理派生 id 不进名册（谓词由 browsing_admittance 提供）"


class TestRosterPopulationMQF1:
    """MQ-F1（2026-09-03）：行来源 population 曾 = bindings，简介来源 =
    `.agent_summaries.json`——`claude-code` / `codex`（X-Agent-ID 直认）与 `dsh`
    （规则库指纹）都不进 bindings ⇒ live 名册三行全是构造件/Pi。
    判别力对照：把 `summaries_file` / `declared` 拿掉（= 旧行来源）本类必红。"""

    def _summaries(self, tmp_path, agents):
        f = tmp_path / ".agent_summaries.json"
        f.write_text(json.dumps({a: {"hash": "x", "summary": f"{a} about", "name": a}
                                 for a in agents}), encoding="utf-8")
        return str(f)

    def test_rows_union_bindings_summaries_declared(self, tmp_path):
        src = registry_agents_source(
            _cache(tmp_path, [
                {"user_id": "local", "origin_key": "p1", "agent_id": "Pi",
                 "trigger": "system_prompt:pi", "identified_at": 100.0}]),
            summaries_file=self._summaries(tmp_path, ["claude-code", "codex", "Pi"]),
            declared=["dsh", "codex", "claude-code", "Pi"])
        ids = [r.agent_id for r in src()]
        assert ids[0] == "Pi", "bindings 行带 identified_at，排最前"
        assert set(ids) == {"Pi", "claude-code", "codex", "dsh"}, \
            "行来源 ⊇ 简介来源 ∪ 规则库声明——直认/指纹 agent 不再因不进 bindings 而缺行"
        by = {r.agent_id: r for r in src()}
        assert by["claude-code"].first_seen == "", "非 bindings 来源没有 identified_at，留空不编"

    def test_eligible_applies_to_every_source(self, tmp_path):
        src = registry_agents_source(
            _cache(tmp_path, [
                {"user_id": "local", "origin_key": "t1", "agent_id": "test-agent",
                 "trigger": "header:X-Agent-ID", "identified_at": 1.0}]),
            summaries_file=self._summaries(tmp_path, ["test-agent", "codex:guardian",
                                                      "unknown-abc", "codex"]),
            declared=["codex"],
            eligible=lambda a: a == "codex")
        assert [r.agent_id for r in src()] == ["codex"], \
            "三个来源过同一把准入：构造件绑定 / 子代理 / unknown 桶一个都不进"

    def test_missing_summaries_file_is_not_error(self, tmp_path):
        src = registry_agents_source(_cache(tmp_path, []),
                                     summaries_file=str(tmp_path / "nope.json"),
                                     declared=["dsh"])
        assert [r.agent_id for r in src()] == ["dsh"]


class TestRosterWorkedGate:
    """判据②「真干过活」由树数据源回答：`agent_last_seen` 没有的 agent
    （规则库声明了但从没跑过流量 / 抽样全 aux 的内部桶）不进名册；
    树没扫过之前不写名册（不注一版没过判据的表）。"""

    def _daemon(self, tmp_path, rows, tree):
        from bladex_core.ledger import Ledger as _L
        led = _L(ledger_id="ldg-roster000003", title="t")
        return FlashDaemon(root=str(tmp_path), principal="local",
                           ledger_source=lambda: ({led.ledger_id: led}, {}),
                           emit_admin_event=lambda t, p: None,
                           agents_source=lambda: rows,
                           tree_source=lambda: tree, tree_every_s=0.0)

    def test_declared_but_never_active_is_dropped(self, tmp_path):
        from bladex_core.flash_tree import AgentRow, agents_roster_path
        rows = [AgentRow(agent_id="claude-code"), AgentRow(agent_id="cursor"),
                AgentRow(agent_id="hermes")]
        tree = {"sessions": {}, "session_ledgers": {},
                "agent_last_seen": {"claude-code": "2026-09-01"},
                "agent_prompts": {}, "session_project": {},
                "excluded": ["hermes"]}
        d = self._daemon(tmp_path, rows, tree)
        d.run_once()
        text = open(agents_roster_path(str(tmp_path), "local"), encoding="utf-8").read()
        assert "`claude-code`" in text
        assert "`cursor`" not in text, "规则库声明但 Hub 无非 aux 轮 ⇒ 没干过活，不进"
        assert "`hermes`" not in text, "抽样全 aux 的内部功能桶不进"

    def test_no_roster_before_first_tree_pass(self, tmp_path):
        from bladex_core.flash_tree import AgentRow, agents_roster_path
        d = self._daemon(tmp_path, [AgentRow(agent_id="claude-code")], {
            "sessions": {}, "session_ledgers": {}, "agent_last_seen": {},
            "agent_prompts": {}, "session_project": {}, "excluded": []})
        import time as _t
        d._tree_every_s, d._tree_last = 10 ** 9, _t.time()   # 树窗口未到 ⇒ 本轮无树数据
        d.run_once()
        assert not os.path.exists(agents_roster_path(str(tmp_path), "local")), \
            "树没扫过 ⇒ 判据②无法回答 ⇒ 不写名册"


class TestDaemonMaterializesRoster:
    def _daemon(self, tmp_path, rows):
        led = Ledger(ledger_id="ldg-roster000001", title="t")
        return FlashDaemon(root=str(tmp_path), principal="local",
                           ledger_source=lambda: ({led.ledger_id: led}, {}),
                           emit_admin_event=lambda t, p: None,
                           agents_source=lambda: rows)

    def test_roster_written_and_not_edit_captured(self, tmp_path):
        from bladex_core.flash_tree import AgentRow, agents_roster_path
        d = self._daemon(tmp_path, [AgentRow(agent_id="pi", first_seen="2026-08-28")])
        d.run_once()
        path = agents_roster_path(str(tmp_path), "local")
        assert os.path.isfile(path)
        with open(path, encoding="utf-8") as f:
            assert "`pi`" in f.read()
        assert path not in d._snapshots, \
            "名册是机器文件，不得进直编捕获快照（否则逐轮 parse_ledger_md 告警）"
        # 用户改名册 → 不产生直编捕获（机器文件没有直编语义）
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n| hacked |\n")
        stats = d.run_once()
        assert stats["user_edits"] == 0

    def test_source_failure_does_not_block_ledger_pool(self, tmp_path):
        def boom():
            raise RuntimeError("cache corrupt")
        led = Ledger(ledger_id="ldg-roster000002", title="t2")
        d = FlashDaemon(root=str(tmp_path), principal="local",
                        ledger_source=lambda: ({led.ledger_id: led}, {}),
                        emit_admin_event=lambda t, p: None, agents_source=boom)
        stats = d.run_once()
        assert stats["written"] >= 1, "名册源坏了不许连累账本池物化"


class TestStableLayerInjection:
    def _rt(self, roster: str):
        from bladex_proxy.agency import AgencyRuntime
        rt = AgencyRuntime(index=None, hub=None)
        rt.about = "BladeX is a memory proxy."
        rt.agents_roster = roster
        return rt

    _MSGS = [{"role": "system", "content": "agent sys"},
             {"role": "user", "content": "hi"}]

    def test_roster_joins_the_about_block(self):
        rt = self._rt("# Agents\n\n| agent_id |\n| --- |\n| `pi` |")
        out = rt.insert_system_notes([dict(m) for m in self._MSGS],
                                     toolface_injected=True)
        [block] = [m for m in out if "bladex-about" in str(m.get("content"))]
        assert "`pi`" in block["content"], "名册必须并入同一稳定块"

    def test_empty_roster_keeps_old_shape(self):
        rt = self._rt("")
        out = rt.insert_system_notes([dict(m) for m in self._MSGS],
                                     toolface_injected=True)
        [block] = [m for m in out if "bladex-about" in str(m.get("content"))]
        assert "Agents" not in block["content"]


def test_load_agents_roster_caps_and_strips_machine_marker(tmp_path, monkeypatch):
    from bladex_core.flash_tree import AgentRow, agents_roster_path, render_agents_roster
    monkeypatch.setenv("BLADEX_FLASH_PATH", str(tmp_path))
    rows = [AgentRow(agent_id=f"a{i:02d}", first_seen=f"2026-08-{i+1:02d}")
            for i in range(20)]
    path = agents_roster_path(str(tmp_path), "local")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(render_agents_roster(rows))
    from bladex_proxy.agency import load_agents_roster
    text = load_agents_roster(max_rows=15)
    assert "<!--" not in text, "机器标记不进注入面"
    assert "`a19`" in text and "`a00`" not in text, "封顶保尾（最新的留下）"
    assert text.count("| `a") == 15


def test_reserved_component_guard():
    """agent 恰好叫 ledgers/agents 时不得吞掉池目录（tree/ 层去除后的守卫）。"""
    from bladex_core.flash_tree import agent_dir
    assert agent_dir("/r", "u", "ledgers").endswith("/a-ledgers")
    assert agent_dir("/r", "u", "pi").endswith("/pi")
