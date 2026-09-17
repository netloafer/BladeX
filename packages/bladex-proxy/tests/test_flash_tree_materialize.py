"""V-F3 第二批：目录树物化（Jason 拍板"基础目录机制先立，账本可以为空"）
+ V-F2b 蒸馏简介（预算 = 每 agent 每源变更 1 次，sha 缓存落盘）。
"""

from __future__ import annotations

import os

import pytest
from bladex_core.ledger import Ledger
from bladex_proxy.flash_daemon import FlashDaemon, hub_tree_source


@pytest.fixture(autouse=True)
def _retention_off(monkeypatch):
    """本文件的假轮次都是 1970 时间戳——保鲜窗口全关（=0 禁用），
    让既有断言只测各自的机制；保鲜自身在 test_flash_retention.py 单测。"""
    for k in ("BLADEX_FLASH_SESSIONS_PER_PROJECT",
              "BLADEX_FLASH_PROJECTS_PER_AGENT",
              "BLADEX_FLASH_PROJECT_ACTIVE_DAYS",
              "BLADEX_FLASH_AGENT_RETIRE_DAYS"):
        monkeypatch.setenv(k, "0")


class _Ev:
    def __init__(self, etype, payload):
        self.event_type = etype
        self.payload = payload


class _Ident:
    def __init__(self, project_id="", project_name=""):
        self.project_id = project_id
        self.project_name = project_name


class _Turn:
    def __init__(self, sys_text, aux=False, project=("", "")):
        self.auxiliary = aux
        self.identity = _Ident(*project)
        self.request_messages = [{"role": "system", "content": sys_text},
                                 {"role": "user", "content": "hi"}]


class _FakeHub:
    def __init__(self, keys, events=(), turns=None):
        self.keys = keys
        self.events = list(events)
        self.turns = turns or {}

    def catch_up(self):
        pass

    def scan_meta(self):
        yield from ((k, "ts") for k in self.keys)

    def scan_admin_events(self):
        yield from (("k", e) for e in self.events)

    def get(self, key):
        return self.turns.get(key)


def _keys(agent, sess, *ms_list):
    return [f"local/{agent}/{sess}/{m}-0" for m in ms_list]


class TestHubTreeSource:
    def test_sessions_capped_and_unknown_excluded(self):
        keys = []
        for i in range(25):
            keys += _keys("hermes", f"s{i:02d}", 1000 + i)
        keys += _keys("unknown-9d9bdd3e", "sx", 9999)
        src = hub_tree_source(_FakeHub(keys), max_sessions_per_agent=20)
        tree = src()
        assert len(tree["sessions"]["hermes"]) == 20, "每 agent 近 N 个封顶"
        assert tree["sessions"]["hermes"][0]["session_id"] == "s24", "按最近排序"
        assert "unknown-9d9bdd3e" not in tree["sessions"], "未认领桶不建目录"

    def test_session_ledgers_from_switch_events(self):
        src = hub_tree_source(_FakeHub(
            _keys("pi", "s1", 1000),
            events=[_Ev("ledger_switch", {"agent_id": "pi", "session_id": "s1",
                                          "from_ledger_id": "",
                                          "to_ledger_id": "ldg-tree00000001"})]))
        tree = src()
        assert tree["session_ledgers"][("pi", "s1")] == ["ldg-tree00000001"]

    def test_agent_prompt_excerpt_from_latest_turn(self):
        keys = _keys("pi", "s1", 1000, 2000)
        hub = _FakeHub(keys, turns={keys[1]: _Turn("You are Pi, a coding agent.")})
        tree = hub_tree_source(hub)()
        assert tree["agent_prompts"]["pi"].startswith("You are Pi")
        assert tree["agent_last_seen"]["pi"] == "1970-01-01"


class TestTreeMaterialize:
    def _daemon(self, tmp_path, hub, summarize=None):
        led = Ledger(ledger_id="ldg-tree00000001", title="树测试")
        return FlashDaemon(
            root=str(tmp_path), principal="local",
            ledger_source=lambda: ({led.ledger_id: led}, {}),
            emit_admin_event=lambda t, p: None,
            agents_source=lambda: [],
            tree_source=hub_tree_source(hub), summarize=summarize,
            tree_every_s=0.0)

    def test_full_tree_with_empty_ledgers_allowed(self, tmp_path):
        keys = _keys("pi", "s1", 1000) + _keys("pi", "s2", 2000)
        hub = _FakeHub(keys, events=[
            _Ev("ledger_switch", {"agent_id": "pi", "session_id": "s1",
                                  "from_ledger_id": "",
                                  "to_ledger_id": "ldg-tree00000001"})])
        d = self._daemon(tmp_path, hub)
        d.run_once()
        base = tmp_path / "local" / "personal" / "pi"
        assert (base / "PROJECTS.md").is_file()
        assert (base / "global" / "SESSIONS.md").is_file()
        led_list = base / "global" / "s1" / "LEDGERS.md"
        assert led_list.is_file()
        assert "ldg-tree00000001" in led_list.read_text(encoding="utf-8")
        # 账本可以为空：无账本 session 也建目录与清单（Jason 拍板）
        empty = (base / "global" / "s2" / "LEDGERS.md").read_text(encoding="utf-8")
        assert "no ledgers yet" in empty
        # 树文件是机器文件：不进直编捕获快照
        assert not any("PROJECTS.md" in p or "SESSIONS.md" in p or "LEDGERS.md" in p
                       for p in d._snapshots)


class TestSummaries:
    def test_budget_one_call_per_source_change(self, tmp_path):
        keys = _keys("pi", "s1", 1000)
        hub = _FakeHub(keys, turns={keys[0]: _Turn("You are Pi.")})
        calls = []

        def summarize(src):
            calls.append(src)
            return "Pi is a lightweight coding agent."

        d = self._mk(tmp_path, hub, summarize)
        d.run_once()
        d._tree_last = 0.0
        d.run_once()                      # 源未变 ⇒ 内存/盘缓存命中，零新调用
        assert len(calls) == 1, "预算：每 agent 每源变更 1 次"
        # 缓存落盘：新进程（新 daemon 实例）同源仍零调用
        d2 = self._mk(tmp_path, hub, summarize)
        d2.run_once()
        assert len(calls) == 1
        assert d2._summaries["pi"] == "Pi is a lightweight coding agent."

    def test_failure_leaves_blank_not_fabricated(self, tmp_path):
        keys = _keys("pi", "s1", 1000)
        hub = _FakeHub(keys, turns={keys[0]: _Turn("You are Pi.")})

        def boom(src):
            raise RuntimeError("upstream down")

        d = self._mk(tmp_path, hub, boom)
        d.run_once()
        assert "pi" not in d._summaries, "失败留空——空着比编一句强"

    @staticmethod
    def _mk(tmp_path, hub, summarize):
        return FlashDaemon(root=str(tmp_path), principal="local",
                           ledger_source=lambda: ({}, {}),
                           emit_admin_event=lambda t, p: None,
                           agents_source=lambda: [],
                           tree_source=hub_tree_source(hub),
                           summarize=summarize, tree_every_s=0.0)


class TestSummarySourceHygiene:
    """live 病例（2026-08-29 首跑名册）：hermes 简介蒸成 {"title": ...}——
    源取到了标题生成子调用的 system prompt，模型服从了源文本内嵌指令。"""

    def test_aux_turn_skipped_as_prompt_source(self, tmp_path):
        keys = _keys("hermes", "s1", 1000, 2000)
        hub = _FakeHub(keys, turns={
            keys[1]: _Turn("Generate a JSON title for this session.", aux=True),
            keys[0]: _Turn("You are Hermes, a helpful agent."),
        })
        tree = hub_tree_source(hub)()
        assert tree["agent_prompts"]["hermes"].startswith("You are Hermes"),             "最新轮是 aux ⇒ 回退到最近的非 aux 轮"

    def test_json_shaped_summary_discarded_and_not_retried(self, tmp_path):
        keys = _keys("pi", "s1", 1000)
        hub = _FakeHub(keys, turns={keys[0]: _Turn("You are Pi.")})
        calls = []

        def hijacked(src):
            calls.append(src)
            return '{"title": "Friendly greeting"}'

        d = FlashDaemon(root=str(tmp_path), principal="local",
                        ledger_source=lambda: ({}, {}),
                        emit_admin_event=lambda t, p: None,
                        agents_source=lambda: [],
                        tree_source=hub_tree_source(hub),
                        summarize=hijacked, tree_every_s=0.0)
        d.run_once()
        assert d._summaries.get("pi", "") == "", "JSON 形态 = 劫持，不进名册"
        d._tree_last = 0.0
        d.run_once()
        assert len(calls) == 1, "同源不重试（重试只会再被劫持）"


class TestProjectLayer:
    """树 project 层接线（2026-08-29，识别闭环后的第一个消费面）：
    会话按最新轮的 identity.project_id 分组；无归属（历史轮）落 global；
    目录名 = 人读名-短hash（身份永远是 project_id，目录名只是投影）。"""

    def test_sessions_grouped_by_project(self, tmp_path):
        keys = _keys("codex", "s-proj", 1000) + _keys("codex", "s-old", 2000)
        hub = _FakeHub(keys, turns={
            keys[0]: _Turn("You are Codex.",
                           project=("p-10ad1f61d0c2", "BladeX")),
            keys[1]: _Turn("You are Codex."),          # 无归属 → global
        })
        led = Ledger(ledger_id="ldg-proj0000001", title="项目层")
        d = FlashDaemon(root=str(tmp_path), principal="local",
                        ledger_source=lambda: ({led.ledger_id: led}, {}),
                        emit_admin_event=lambda t, p: None,
                        agents_source=lambda: [],
                        tree_source=hub_tree_source(hub), tree_every_s=0.0)
        d.run_once()
        base = tmp_path / "local" / "personal" / "codex"
        assert (base / "BladeX-10ad1f" / "SESSIONS.md").is_file(),             "有归属的会话进项目目录（人读名-短hash）"
        assert (base / "global" / "SESSIONS.md").is_file(),             "历史会话（无 project_id）仍在 global"
        projects = (base / "PROJECTS.md").read_text(encoding="utf-8")
        assert "BladeX" in projects and "p-10ad1f61d0c2" in projects
        assert (base / "BladeX-10ad1f" / "s-proj" / "LEDGERS.md").is_file()

    def test_project_name_aggregated_across_sessions(self, tmp_path):
        """live 病例：project_name 晚于 project_id 上线，老轮次有 id 没名 ⇒
        渲染成 `proj-10ad1f`。修法：同 pid 任何一轮带名，所有会话共用。"""
        keys = _keys("codex", "s-old", 1000) + _keys("codex", "s-new", 2000)
        hub = _FakeHub(keys, turns={
            keys[0]: _Turn("sys", project=("p-10ad1f61d0c2", "")),      # 老轮无名
            keys[1]: _Turn("sys", project=("p-10ad1f61d0c2", "BladeX")),
        })
        tree = hub_tree_source(hub)()
        assert tree["session_project"][("codex", "s-old")] ==             ("p-10ad1f61d0c2", "BladeX", ""), "老会话借到同 pid 的名字"

    def test_turns_without_identity_stay_global(self, tmp_path):
        """旧 fake/历史轮无 identity 属性 ⇒ 零回归全落 global。"""
        keys = _keys("pi", "s1", 1000)
        hub = _FakeHub(keys, turns={keys[0]: _Turn("You are Pi.")})
        tree = hub_tree_source(hub)()
        assert tree["session_project"] == {}


class TestBrowsingAdmittance:
    """2026-08-29 拍板（v2，撤销当日早间的 base 聚合 v1）：
    **有独立 agent 声明才算 agent** ——
    profile 变体（hermes:default/accept 各有 profile 与配置）= 独立 agent，分开展示；
    子代理派生 id（codex:guardian）内容不进 Index，也不进 flash；
    纯 aux 桶（裸 hermes：checkpoint/命名内部功能流量）不建目录。
    记忆命名空间仍按完整 agent_id 隔离不动。
    """

    @staticmethod
    def _pred(bindings=()):
        from bladex_proxy.agent_rules import AgentFingerprintRule as R
        from bladex_proxy.flash_daemon import browsing_admittance
        rules = [
            R(agent_id="hermes", auxiliary=True,
              system_prompt_keywords=["checkpoint"]),   # aux 规则也挂 hermes
            R(agent_id="hermes", profile_aware=True),
            R(agent_id="codex", subagent_headers=["x-openai-subagent"]),
            R(agent_id="pi"),
            R(agent_id="titler", auxiliary=True),       # 只有 aux 声明的"agent"
        ]
        return browsing_admittance(rules, list(bindings))

    def test_predicate_matrix(self):
        ok = self._pred()
        assert ok("hermes")                    # 非 aux base 声明存在
        assert ok("hermes:default") and ok("hermes:accept"), \
            "profile 变体 = 独立 agent（各有 profile 与配置文件）"
        assert ok("codex")
        assert not ok("codex:guardian"), "子代理派生 id：内容不进 Index，不进 flash"
        assert not ok("titler"), "只有 auxiliary 声明 ≠ 独立 agent 声明"
        assert not ok("unknown-9d9bdd3e") and not ok("")

    def test_explicit_header_declaration_admits(self):
        ok = self._pred(bindings=[{"agent_id": "my-bot",
                                   "trigger": "header:X-Agent-ID"}])
        assert ok("my-bot"), "X-Agent-ID 显式声明 = 用户亲手起的名"
        assert not self._pred()("my-bot")

    def test_tree_profiles_distinct_and_subagent_excluded(self):
        keys = (_keys("hermes:default", "s1", 1000)
                + _keys("hermes:accept", "s2", 2000)
                + _keys("codex", "s3", 3000)
                + _keys("codex:guardian", "s4", 4000))
        tree = hub_tree_source(_FakeHub(keys), eligible=self._pred())()
        assert set(tree["sessions"]) == {"hermes:default", "hermes:accept",
                                         "codex"}

    def test_pure_aux_bucket_excluded(self):
        keys = _keys("hermes", "s1", 1000, 2000)
        hub = _FakeHub(keys, turns={
            keys[0]: _Turn("name this chat session", aux=True),
            keys[1]: _Turn("summarization checkpoint", aux=True),
        })
        tree = hub_tree_source(hub, eligible=self._pred())()
        assert "hermes" not in tree["sessions"], \
            "判据②：声明存在但桶内全是内部功能轮次（裸 hermes）⇒ 不建目录"
        assert "hermes" in tree["excluded"], "排除名单要传给名册过滤"

    def test_active_ledger_via_direct_binding(self, tmp_path):
        from bladex_core.flash_tree import agent_dir
        from bladex_core.ledger_runtime import activation_scope
        keys = _keys("hermes:default", "s1", 1000)
        hub = _FakeHub(keys, events=[
            _Ev("ledger_switch", {"agent_id": "hermes:default", "session_id": "s1",
                                  "from_ledger_id": "",
                                  "to_ledger_id": "ldg-base00000001"})])
        led = Ledger(ledger_id="ldg-base00000001", title="独立profile")
        bindings = {activation_scope("hermes:default", ""): "ldg-base00000001"}
        d = FlashDaemon(root=str(tmp_path), principal="local",
                        ledger_source=lambda: ({led.ledger_id: led}, bindings),
                        emit_admin_event=lambda t, p: None,
                        agents_source=lambda: [],
                        tree_source=hub_tree_source(hub), tree_every_s=0.0)
        d.run_once()
        base = agent_dir(str(tmp_path), "local", "hermes:default")
        with open(os.path.join(base, "global", "s1", "LEDGERS.md"),
                  encoding="utf-8") as f:
            text = f.read()
        assert "ldg-base00000001" in text and "(active)" in text


class TestAuxSampleWindow:
    """判据②抽样窗口（2026-09-03 B0 首验）：hermes:default 最近 5 轮全是 checkpoint/标题子调用
    （aux），真实流量在第 6 轮以后——窗口 5 会把干过 176 轮活的 agent 判成内部桶。"""

    def test_agent_with_recent_aux_burst_is_not_excluded(self):
        from bladex_proxy.flash_daemon import TREE_AUX_SAMPLE_ROUNDS
        assert TREE_AUX_SAMPLE_ROUNDS >= 20
        keys = _keys("hermes:default", "s1", *range(1000, 1000 + 12 * 100, 100))  # 12 轮
        turns = {k: _Turn("title json", aux=True) for k in keys}
        turns[keys[0]] = _Turn("You are Hermes.")          # 最老那轮才是真活
        hub = _FakeHub(keys, turns=turns)
        tree = hub_tree_source(hub)()
        assert "hermes:default" not in tree["excluded"]
        assert tree["agent_prompts"]["hermes:default"].startswith("You are Hermes")
        # 判别力对照：窗口收回 5 就落榜
        tree5 = hub_tree_source(hub, aux_sample_rounds=5)()
        assert "hermes:default" in tree5["excluded"]
