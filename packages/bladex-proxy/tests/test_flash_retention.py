"""Flash 内容规则统一批（2026-08-29 Jason 拍板，方案 v2）：
保鲜（贵在精：目录数量也是维护对象，窗口滑出即回收；参数配置化 flags.py）+
甲类内容五项（项目简介/PROJECT.md、PROJECTS 时间列、AGENTS 名称搭车、
SESSIONS 机械摘录名、USER/RULES/AGENT.md Index 读侧投影）。
"""

from __future__ import annotations

import time

import pytest
from bladex_core.ledger import Ledger
from bladex_proxy.flash_daemon import FlashDaemon, hub_tree_source


class _Ident:
    def __init__(self, project=("", "", "")):
        self.project_id, self.project_name, self.project_root = project


class _Turn:
    def __init__(self, sys_text="sys", aux=False, project=("", "", ""),
                 user_text="修一下登录页的 bug"):
        self.auxiliary = aux
        self.identity = _Ident(project)
        self.request_messages = [{"role": "system", "content": sys_text},
                                 {"role": "user", "content": user_text}]


class _FakeHub:
    def __init__(self, keys, turns=None):
        self.keys = keys
        self.turns = turns or {}

    def catch_up(self):
        pass

    def scan_meta(self):
        yield from ((k, "ts") for k in self.keys)

    def scan_admin_events(self):
        yield from ()

    def get(self, key):
        return self.turns.get(key, _Turn())


def _now_ms(days_ago: float = 0.0) -> int:
    return int((time.time() - days_ago * 86400) * 1000)


def _keys(agent, sess, ms):
    return [f"local/{agent}/{sess}/{ms}-0"]


def _daemon(tmp_path, hub, **kw):
    led = Ledger(ledger_id="ldg-ret00000001", title="保鲜测试")
    return FlashDaemon(root=str(tmp_path), principal="local",
                       ledger_source=lambda: ({led.ledger_id: led}, {}),
                       emit_admin_event=lambda t, p: None,
                       agents_source=lambda: [],
                       tree_source=hub_tree_source(hub),
                       tree_every_s=0.0, **kw)


@pytest.fixture(autouse=True)
def _windows(monkeypatch):
    monkeypatch.setenv("BLADEX_FLASH_SESSIONS_PER_PROJECT", "2")
    monkeypatch.setenv("BLADEX_FLASH_PROJECTS_PER_AGENT", "2")
    monkeypatch.setenv("BLADEX_FLASH_PROJECT_ACTIVE_DAYS", "30")
    monkeypatch.setenv("BLADEX_FLASH_AGENT_RETIRE_DAYS", "90")


class TestRetention:
    def test_session_cap_per_project(self, tmp_path):
        keys = []
        for i in range(4):
            keys += _keys("pi", f"s{i}", _now_ms(i * 0.1))
        d = _daemon(tmp_path, _FakeHub(keys))
        d.run_once()
        gdir = tmp_path / "local" / "personal" / "pi" / "global"
        dirs = {p.name for p in gdir.iterdir() if p.is_dir()}
        assert dirs == {"s0", "s1"}, "每 project 只留最近 N 个会话目录"
        sess_md = (gdir / "SESSIONS.md").read_text(encoding="utf-8")
        assert "s3" not in sess_md, "清单与目录同窗"

    def test_stale_dirs_reclaimed_but_pool_and_rosters_kept(self, tmp_path):
        keys = _keys("pi", "s-new", _now_ms(0))
        d = _daemon(tmp_path, _FakeHub(keys))
        base = tmp_path / "local" / "personal"
        # 预置窗口外残留：退休 agent 目录 + pi 下的过期会话目录
        (base / "olddog" / "global" / "sx").mkdir(parents=True)
        (base / "pi" / "global" / "s-stale").mkdir(parents=True)
        d.run_once()
        assert not (base / "olddog").exists(), "窗口外 agent 目录整删"
        assert not (base / "pi" / "global" / "s-stale").exists()
        assert (base / "pi" / "global" / "s-new").exists()
        assert (base / "ledgers").exists(), "账本池永不回收"
        assert (base / "AGENTS.md").exists(), "名册保留"

    def test_user_content_blocks_reclaim(self, tmp_path):
        keys = _keys("pi", "s-new", _now_ms(0))
        d = _daemon(tmp_path, _FakeHub(keys))
        stale = tmp_path / "local" / "personal" / "pi" / "global" / "s-stale"
        stale.mkdir(parents=True)
        (stale / "notes.local.md").write_text("用户手记", encoding="utf-8")
        d.run_once()
        assert stale.exists(), "含 *.local.md 的目录宁留勿删"

    def test_agent_retired_by_age(self, tmp_path):
        keys = _keys("olddog", "s1", _now_ms(120))   # 120 天前
        d = _daemon(tmp_path, _FakeHub(keys))
        d.run_once()
        assert not (tmp_path / "local" / "personal" / "olddog").exists(), \
            "超过退休窗口的 agent 不再物化目录"

    def test_windows_disabled_by_zero(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BLADEX_FLASH_SESSIONS_PER_PROJECT", "0")
        monkeypatch.setenv("BLADEX_FLASH_AGENT_RETIRE_DAYS", "0")
        keys = []
        for i in range(4):
            keys += _keys("pi", f"s{i}", _now_ms(i * 0.1))
        keys += _keys("olddog", "sx", _now_ms(120))
        d = _daemon(tmp_path, _FakeHub(keys))
        d.run_once()
        gdir = tmp_path / "local" / "personal" / "pi" / "global"
        assert len([p for p in gdir.iterdir() if p.is_dir()]) == 4
        assert (tmp_path / "local" / "personal" / "olddog").exists(), \
            "0 = 禁用该窗口（用户可配，回到不回收行为）"


class TestProjectBrief:
    def test_brief_and_columns(self, tmp_path):
        root = tmp_path / "proj-root"
        root.mkdir()
        (root / "CLAUDE.md").write_text("# BladeX\n记忆中间件", encoding="utf-8")
        keys = _keys("codex", "s1", _now_ms(0))
        hub = _FakeHub(keys, turns={keys[0]: _Turn(
            project=("p-10ad1f61d0c2", "BladeX", str(root)))})

        def summarize_project(src):
            return "记忆为核心的 proxy 中间件。通过账本与记忆层服务多 agent。"

        def agent_flavored(src):
            raise AssertionError("项目简介不得走 agent 口味蒸馏器"
                                 "（NAME|ABOUT 会把项目蒸成 agent 人设）")

        d = _daemon(tmp_path, hub, summarize=agent_flavored,
                    summarize_project=summarize_project)
        d.run_once()
        agent = tmp_path / "local" / "personal" / "codex"
        projects = (agent / "PROJECTS.md").read_text(encoding="utf-8")
        assert "记忆为核心的 proxy 中间件" in projects, "about 列 = 段落首句"
        assert "通过账本" not in projects, "列里只放一句话"
        brief = (agent / "BladeX-10ad1f" / "PROJECT.md").read_text(encoding="utf-8")
        assert "通过账本与记忆层" in brief, "PROJECT.md 放蒸馏段落全文"
        assert projects.count("20") >= 2, "created/updated 列有值（会话 span 聚合）"

    def test_missing_decl_file_leaves_blank(self, tmp_path):
        root = tmp_path / "no-decl"
        root.mkdir()
        keys = _keys("codex", "s1", _now_ms(0))
        hub = _FakeHub(keys, turns={keys[0]: _Turn(
            project=("p-xx", "X", str(root)))})
        d = _daemon(tmp_path, hub, summarize_project=lambda s: "不该出现的简介")
        d.run_once()
        agent = tmp_path / "local" / "personal" / "codex"
        # 判据是产物：无声明文件 ⇒ PROJECT.md 不生成、about 列空
        # （_project_summary 未发起调用）。
        assert not list(agent.glob("*/PROJECT.md")), "无声明文件 ⇒ 无 PROJECT.md"
        projects = (agent / "PROJECTS.md").read_text(encoding="utf-8")
        assert "不该出现的简介" not in projects


class TestSessionName:
    def test_mechanical_excerpt_skips_machine_messages(self, tmp_path):
        keys = _keys("codex", "s1", _now_ms(0))
        t = _Turn(project=("", "", ""))
        t.request_messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "# AGENTS.md instructions for /x\n…"},
            {"role": "user", "content": [{"type": "input_text",
                                          "text": "在线吗？"}]},
        ]
        tree = hub_tree_source(_FakeHub(keys, turns={keys[0]: t}))()
        assert tree["sessions"]["codex"][0]["name"] == "在线吗？", \
            "跳过 AGENTS.md 头等机器消息，取首条真实 user（零 LLM 机械摘录）"


class TestAgentNameRideAlong:
    def test_name_about_parsed(self, tmp_path):
        keys = _keys("pi", "s1", _now_ms(0))
        hub = _FakeHub(keys, turns={keys[0]: _Turn("You are Pi.")})
        d = _daemon(tmp_path, hub,
                    summarize=lambda s: "Pi | Expert coding agent in a harness.")
        d.run_once()
        assert d._agent_names["pi"] == "Pi"
        assert d._summaries["pi"] == "Expert coding agent in a harness."

    def test_no_separator_degrades_to_about_only(self, tmp_path):
        keys = _keys("pi", "s1", _now_ms(0))
        hub = _FakeHub(keys, turns={keys[0]: _Turn("You are Pi.")})
        d = _daemon(tmp_path, hub, summarize=lambda s: "Just an agent.")
        d.run_once()
        assert d._agent_names["pi"] == "" and d._summaries["pi"] == "Just an agent."

    def test_legacy_cache_without_name_key_redistills_once(self, tmp_path):
        """缓存 schema v2：旧格式条目（缺 name 键）视为失效重蒸一次——
        否则 sha 命中永不重蒸，name 列无限期留空（2026-08-29 live 读数）。"""
        import json
        keys = _keys("pi", "s1", _now_ms(0))
        hub = _FakeHub(keys, turns={keys[0]: _Turn("You are Pi.")})
        d1 = _daemon(tmp_path, hub, summarize=lambda s: "Pi | New agent.")
        d1.run_once()
        cache_file = tmp_path / ".agent_summaries.json"
        cache = json.loads(cache_file.read_text(encoding="utf-8"))
        cache["pi"].pop("name")             # 模拟旧版缓存
        cache["pi"]["summary"] = "old"
        cache_file.write_text(json.dumps(cache), encoding="utf-8")
        calls = []

        def summ(src):
            calls.append(src)
            return "Pi | Fresh again."

        d2 = _daemon(tmp_path, hub, summarize=summ)
        d2.run_once()
        assert calls, "缺 name 键必须重蒸"
        assert d2._agent_names["pi"] == "Pi"
        d3 = _daemon(tmp_path, hub, summarize=lambda s: (_ for _ in ()).throw(
            AssertionError("v2 命中后不得再蒸")))
        d3.run_once()                        # sha 未变 + name 在 ⇒ 纯命中


class TestProfilesProjection:
    def test_index_side_projection(self, tmp_path):
        keys = _keys("pi", "s1", _now_ms(0))
        d = _daemon(tmp_path, _FakeHub(keys), profiles_source=lambda: (
            "# User\n偏好中文回复", "# Rules\n- [hard_rule] NEVER use emojis",
            {"pi": "# Pi habits\n常用 rg"}))
        d.run_once()
        base = tmp_path / "local" / "personal"
        assert "偏好中文" in (base / "USER.md").read_text(encoding="utf-8")
        assert "NEVER" in (base / "RULES.md").read_text(encoding="utf-8")
        assert "常用 rg" in (base / "pi" / "AGENT.md").read_text(encoding="utf-8")

    def test_source_failure_keeps_existing(self, tmp_path):
        keys = _keys("pi", "s1", _now_ms(0))

        def boom():
            raise RuntimeError("index closed")

        d = _daemon(tmp_path, _FakeHub(keys), profiles_source=boom)
        d.run_once()          # 不炸、不写、不删
        assert not (tmp_path / "local" / "personal" / "USER.md").exists()
