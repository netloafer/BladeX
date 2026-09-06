"""G11.11 验收剧本 —— AGENT_CLAIM（认领 / 改名 / 合并）+ 误合并红线。

一套语义三种用法（Jason 2026-08-18 拍板：不区分两套体系）：
`from_agent_ids[] → to_agent_id`。红线检查就落在这一条请求的校验里，
不需要第二套机制：`len(from) > 1` 或 `to` 已存在 → 判定为合并，需二次确认。
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from bladex_proxy.agent_rules import (
    append_user_rule,
    load_agent_rules,
    parse_rules,
    render_rule_toml,
    reset_rules_cache,
)
from bladex_proxy.models import AdminEvent, AdminEventType


@pytest.fixture(autouse=True)
def _clean():
    reset_rules_cache()
    yield
    reset_rules_cache()


# ── 规则写回（认领一次就永久认识）────────────────────────────────────────────


def test_append_rule_creates_file_and_takes_effect(tmp_path: Path) -> None:
    target = tmp_path / "agent_rules.toml"
    append_user_rule(
        {
            "name": "my-agent",
            "agent_id": "my-agent",
            "system_prompt_keywords": ["you are my custom agent"],
        },
        path=target,
    )
    assert target.exists()
    rules = load_agent_rules(user_path=target)
    assert rules[0].agent_id == "my-agent"
    assert rules[0].source == "user"


def test_append_preserves_existing_content(tmp_path: Path) -> None:
    """🔴 追加而不是重写 —— 那个文件是用户自己在维护的，重写会丢掉他的注释。"""
    target = tmp_path / "agent_rules.toml"
    target.write_text(
        textwrap.dedent("""
            # 我自己写的注释，不许被吃掉
            [[rules]]
            name = "mine"
            agent_id = "mine"
            system_prompt_keywords = ["mine"]
        """),
        encoding="utf-8",
    )
    append_user_rule({"name": "added", "agent_id": "added",
                      "system_prompt_keywords": ["added"]}, path=target)
    text = target.read_text(encoding="utf-8")
    assert "我自己写的注释，不许被吃掉" in text
    ids = {r.agent_id for r in parse_rules(text, source="user")}
    assert ids == {"mine", "added"}


def test_invalid_rule_rejected_before_touching_disk(tmp_path: Path) -> None:
    """非法规则在写盘**之前**失败，不留半个坏文件。"""
    target = tmp_path / "agent_rules.toml"
    with pytest.raises(ValueError):
        append_user_rule({"agent_id": "x", "bogus_field": 1}, path=target)
    assert not target.exists()


def test_rendered_rule_round_trips() -> None:
    rule = {
        "name": "vendor",
        "agent_id": "vendor",
        "system_prompt_keywords": ["you are vendor"],
        "tool_signatures": ["vendor_do"],
        "min_tool_match": 1,
        "header_patterns": [{"name": "user-agent", "pattern": "(?i)^vendor/"}],
    }
    (parsed,) = parse_rules(render_rule_toml(rule), source="user")
    assert parsed.agent_id == "vendor"
    assert parsed.header_patterns[0].pattern == "(?i)^vendor/"
    assert parsed.tool_signatures == ["vendor_do"]


# ── 读侧映射（历史数据改归属）────────────────────────────────────────────────


class _FakeHub:
    """只实现 scan_admin_events 的最小替身（构造映射不需要真 ledger）。"""

    def __init__(self, events: list[AdminEvent]) -> None:
        self._events = events

    def scan_admin_events(self):
        for i, ev in enumerate(self._events):
            yield f"admin/{i:04d}", ev


def _claim(from_ids: list[str], to_id: str) -> AdminEvent:
    return AdminEvent(
        event_type=AdminEventType.AGENT_CLAIM,
        matter_id="",          # 🔴 MQ-A7：该字段只在目标真是 Matter 时使用
        payload={"from_agent_ids": from_ids, "to_agent_id": to_id},
    )


def _map(events: list[AdminEvent]) -> dict[str, str]:
    from bladex_proxy.storage.memory_index import MemoryIndex

    return MemoryIndex.build_agent_claim_map(object.__new__(MemoryIndex), _FakeHub(events))


def test_claim_maps_unknown_bucket_to_name() -> None:
    assert _map([_claim(["unknown-a1b2c3d4"], "dsh")]) == {"unknown-a1b2c3d4": "dsh"}


def test_rename_is_the_same_mechanism() -> None:
    """改名与认领同一套语义 —— 不区分两套体系（Jason 拍板）。"""
    assert _map([_claim(["hermes:default"], "hermes:work")]) == {
        "hermes:default": "hermes:work"
    }


def test_merge_maps_multiple_sources() -> None:
    m = _map([_claim(["a", "b"], "c")])
    assert m == {"a": "c", "b": "c"}


def test_chained_claims_converge() -> None:
    """a→b 之后 b→c，则 a 最终指向 c（撤销/连续改名要能收敛）。"""
    m = _map([_claim(["a"], "b"), _claim(["b"], "c")])
    assert m["a"] == "c"
    assert m["b"] == "c"


def test_reused_name_drops_the_stale_rename_mapping() -> None:
    """🔴 名字被**重新启用**时，以它为 key 的旧映射必须失效。

    场景是 Jason 2026-08-19 的两步操作：先 `Pi → Pi-old`（老记忆改名让位），
    再 `unknown-b1d8a6f4 → Pi`（新桶用回这个名字）。

    留着 `Pi → Pi-old` 有两处后果，第二处是数据错误：
      ① dashboard 把 `Pi` 当成"已改名走"的条目，整行不显示；
      ② 等 `Pi` 有了规则、真流量以 `local/Pi/...` 落库，重建时会把**新数据错划给
         `Pi-old`** —— 一条早已完成的改名，反过来污染后来的数据。
    """
    m = _map([
        _claim(["unknown-old"], "Pi"),      # 认领
        _claim(["Pi"], "Pi-old"),           # 改名让位
        _claim(["unknown-new"], "Pi"),      # 新桶用回这个名字
    ])
    assert m["unknown-old"] == "Pi-old"     # 老记忆跟着改名走
    assert m["unknown-new"] == "Pi"         # 新桶归新 Pi
    assert "Pi" not in m, "`Pi → Pi-old` 残留会把后来的数据错划给 Pi-old"


def test_renamed_name_becomes_free_again(client) -> None:
    """改名让出来的名字不再算"已存在"，认领到它不该要合并确认。"""
    client.post("/admin/agents/claim",
                json={"from_agent_ids": ["unknown-r1"], "to_agent_id": "Pi"})
    client.post("/admin/agents/claim",
                json={"from_agent_ids": ["Pi"], "to_agent_id": "Pi-old"})
    # `Pi` 已让位 → 这一步是纯认领，不是合并
    r = client.post("/admin/agents/claim",
                    json={"from_agent_ids": ["unknown-r2"], "to_agent_id": "Pi"})
    assert r.status_code == 200, r.text
    assert r.json()["merge"] is False


def test_self_claim_ignored() -> None:
    assert _map([_claim(["a"], "a")]) == {}


def test_other_admin_events_ignored() -> None:
    other = AdminEvent(event_type=AdminEventType.MATTER_RENAME,
                       matter_id="m1", payload={"title": "t"})
    assert _map([other, _claim(["x"], "y")]) == {"x": "y"}


def test_matter_id_left_empty_on_claim_events() -> None:
    """🔴 MQ-A7 护栏：新增事件类型不得复用 `matter_id` 装别的 id。"""
    ev = _claim(["a"], "b")
    assert ev.matter_id == ""
    assert ev.payload["from_agent_ids"] == ["a"]


# ── 🔴 误合并红线（端点级）──────────────────────────────────────────────────


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from bladex_proxy.agent_registry import _agent_registry
    from bladex_proxy.server import create_app

    _agent_registry.clear()
    app = create_app()
    with TestClient(app) as c:
        # 🔴 换 ledger 必须满足两个条件，缺一个就出事（两个都真踩过）：
        #
        # ① **在 lifespan 启动之后换**——startup 会自己打开一个真 RocksDB 并覆盖
        #    app.state.hub。写在 `with` 之前，断言就打在真库上、随环境漂移。
        # ② **退出前换回去**——lifespan shutdown 执行 `app.state.hub.close()`。
        #    留着替身，真 RocksDB 永不 close，进程内 LOCK 不释放，
        #    下一个用例开同路径就 `IO error: lock hold by current process`。
        #    修①的时候没想到②，全量回归里 7 个用例 ERROR 就是这么来的
        #    （单跑本文件不复现：只有一个 app，锁没人跟它抢）。
        real_ledger = app.state.hub
        app.state.hub = _RecordingLedger()
        # Pipeline 同样换掉：本文件的用例会真发 /v1/chat/completions（那是"测用户
        # 走得通"的必要代价），但**不需要**把这些探针轮次真写进 Redis + Memory Hub。
        # 不换的后果实测可见——gate 输出里混进一串
        # `pipeline_processed key=anonymous/unknown-xxxx/... status=failed`，
        # 既是噪声，也是无谓的写入（上游没凭证，注定 failed）。
        # 用 mock 而非 None：`_enqueue_turn` 见 pipeline is None 会触发 ADR-0017 懒恢复
        # （尝试连 Redis、连不上就 subprocess 拉起 redis-server），比不换还糟。
        # 写法照抄 `test_server.py` 既有约定，不自创。
        real_pipeline = app.state.pipeline
        mock_pipeline = AsyncMock()
        mock_pipeline.close = AsyncMock()
        app.state.pipeline = mock_pipeline
        try:
            yield c
        finally:
            # 🔴 两个都要换回去：shutdown 走 `app.state.<x>.close()`，留着替身
            # 就是真对象永不关闭（ledger 那次是 RocksDB 锁泄漏，pipeline 这次
            # 会是 Redis 连接泄漏）。同一个坑，两个字段。
            app.state.hub = real_ledger
            app.state.pipeline = real_pipeline
    # 自检：确认 shutdown 真的关掉了那个 RocksDB。
    #
    # 锁泄漏的完整链条（用户机 gate 上炸出来的，7 个 ERROR）：
    #   Redis 可用 → lifespan 起 PipelineWorker → `ledger.open()` 取 RocksDB LOCK
    #   → `PipelineWorker.stop()` **不关 ledger**（只 cancel task）
    #   → 全仓唯一关它的地方是 `server.py` shutdown 的 `app.state.hub.close()`
    #   → 我把那个字段换成了替身 → 真库永不 close → 进程内 LOCK 不放
    #   → 下一个用例开同路径：`IO error: lock hold by current process`
    #
    # 🔴 **本断言在没有 Redis 的环境里是空转的**：没有 Redis 就不起 worker、
    # ledger 压根没被打开，`_db` 本来就是 None。开发沙盒正是这种环境，所以这个
    # 自检在那里"通过"不构成证据（验证过：故意退回旧写法，沙盒照样绿）。
    # 真正的保证来自上面 `finally` 里把字段换回去——那是构造性的，不依赖环境。
    assert getattr(real_ledger, "_db", None) is None, (
        "真 ledger 未被关闭 —— RocksDB 进程内 LOCK 会泄漏到下一个用例"
    )
    _agent_registry.clear()


class _RecordingLedger:
    def __init__(self) -> None:
        self.events: list[AdminEvent] = []

    def append_admin_event(self, event_type, matter_id, target_key="", **payload):
        self.events.append(AdminEvent(event_type=event_type, matter_id=matter_id,
                                      target_key=target_key, payload=payload))

    def scan_admin_events(self):
        for i, ev in enumerate(self.events):
            yield f"admin/{i:04d}", ev

    def close(self) -> None:
        """lifespan shutdown 会调它 —— 替身也得实现，否则 teardown 报 AttributeError。"""

    def scan_agent_ids(self) -> dict[str, tuple[str, int]]:
        """替身没有历史数据，返回空。

        🔴 **显式实现而不是靠端点的 try/except 兜着**：缺方法时端点会静默降级，
        整条 Memory Hub 派生路径一行不跑而测试全绿——这个坑本文件已经踩过一次
        （见下面 `_seeded_ledger` 的说明）。替身该有的面要齐，缺什么要看得见。
        """
        return {}

    def get(self, key: str):
        return None


def test_plain_rename_needs_no_confirmation(client) -> None:
    r = client.post("/admin/agents/claim",
                    json={"from_agent_ids": ["unknown-abc12345"], "to_agent_id": "brand-new"})
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "claimed"
    assert r.json()["merge"] is False


def test_multi_source_requires_confirmation(client) -> None:
    """🔴 多来源 = 合并，默认拦下。"""
    r = client.post("/admin/agents/claim",
                    json={"from_agent_ids": ["unknown-a", "unknown-b"], "to_agent_id": "x"})
    assert r.status_code == 400
    assert r.json()["status"] == "confirm_required"


def test_claiming_into_existing_agent_requires_confirmation(client) -> None:
    """🔴 把 dsh 认领成 hermes 就是手动制造跨 agent 记忆污染 —— 必须拦。"""
    r = client.post("/admin/agents/claim",
                    json={"from_agent_ids": ["unknown-dsh"], "to_agent_id": "hermes"})
    assert r.status_code == 400
    assert r.json()["status"] == "confirm_required"
    assert r.json()["merge"] is True


def test_second_bucket_into_an_existing_claim_target_is_a_merge(client) -> None:
    """🔴 认领到"之前认领过的名字"也算合并，判据不许因目标有没有规则而不同。

    Jason 2026-08-19 问"再点另一个桶 claim 到 Pi 会怎样"时暴露：`existing` 只看
    规则库 + 进程内注册表，**不看 journal 里的历史认领目标**。于是把第二个桶并到
    已有的 `Pi` 上——明明是"两组记忆并进一个命名空间"——因为 `Pi` 恰好没规则、
    也没在流量里出现过而**绕过了合并确认**。

    结果本身没错（Pi 只有一行、轮次累加 4+3=7），但**同一个动作因为一件无关的事
    走了不同的门**，这种不一致比放行本身更危险。
    """
    first = client.post("/admin/agents/claim",
                        json={"from_agent_ids": ["unknown-p1"], "to_agent_id": "Pi"})
    assert first.status_code == 200

    second = client.post("/admin/agents/claim",
                         json={"from_agent_ids": ["unknown-p2"], "to_agent_id": "Pi"})
    assert second.status_code == 400
    assert second.json()["status"] == "confirm_required"
    assert second.json()["merge"] is True

    confirmed = client.post("/admin/agents/claim",
                            json={"from_agent_ids": ["unknown-p2"], "to_agent_id": "Pi",
                                  "confirm_merge": True})
    assert confirmed.status_code == 200


def test_confirmed_merge_goes_through(client) -> None:
    r = client.post("/admin/agents/claim",
                    json={"from_agent_ids": ["unknown-dsh"], "to_agent_id": "hermes",
                          "confirm_merge": True})
    assert r.status_code == 200
    assert r.json()["merge"] is True


def test_merge_is_reversible_by_a_reverse_claim(client) -> None:
    """合并可回滚 —— 撤销只需再发一条反向 AGENT_CLAIM（这才是敢放行的前提）。"""
    client.post("/admin/agents/claim",
                json={"from_agent_ids": ["unknown-dsh"], "to_agent_id": "hermes",
                      "confirm_merge": True})
    client.post("/admin/agents/claim",
                json={"from_agent_ids": ["hermes"], "to_agent_id": "unknown-dsh",
                      "confirm_merge": True})
    events = [e for e in client.app.state.hub.events
              if e.event_type == AdminEventType.AGENT_CLAIM]
    assert len(events) == 2
    final = _map(events)
    assert final.get("unknown-dsh") != "hermes"


def test_empty_payload_rejected(client) -> None:
    assert client.post("/admin/agents/claim", json={"to_agent_id": "x"}).status_code == 400
    assert client.post("/admin/agents/claim",
                       json={"from_agent_ids": ["a"]}).status_code == 400


def test_target_cannot_appear_in_sources(client) -> None:
    r = client.post("/admin/agents/claim",
                    json={"from_agent_ids": ["a", "b"], "to_agent_id": "a",
                          "confirm_merge": True})
    assert r.status_code == 400


def test_claim_event_written_with_empty_matter_id(client) -> None:
    client.post("/admin/agents/claim",
                json={"from_agent_ids": ["unknown-zz"], "to_agent_id": "fresh-name"})
    (ev,) = client.app.state.hub.events
    assert ev.event_type == AdminEventType.AGENT_CLAIM
    assert ev.matter_id == ""                       # MQ-A7 护栏
    assert ev.payload["to_agent_id"] == "fresh-name"


def test_agents_listing_exposes_buckets_and_claims(client) -> None:
    client.post("/admin/agents/claim",
                json={"from_agent_ids": ["unknown-zz"], "to_agent_id": "fresh-name"})
    body = client.get("/admin/agents").json()
    by_id = {a["agent_id"]: a for a in body["known_agents"]}
    assert by_id["hermes"]["has_rule"] is True
    assert body["claims"][0]["to"] == "fresh-name"


def test_claimed_agent_without_rule_still_shows_up(client) -> None:
    """🔴 认领完不许从界面上消失。

    实际发生（Jason 2026-08-19）：把 `unknown-3b5f685d` 认领成 `Pi`、关键词留空
    （它是选填的）→ 没写规则 → 待认领列表滤掉了它（已认领），已知列表又只读规则库
    → **Pi 哪儿都不在**，用户当场问"Known agents 怎么看不到"。

    修法是 `known_agents` 取三源并集并标明来源；`claimed, no rule` 这个状态必须
    可见，因为它的含义是"历史归属已改，但下次来还是认不出"。
    """
    client.post("/admin/agents/claim",
                json={"from_agent_ids": ["unknown-vanish"], "to_agent_id": "Pi"})
    by_id = {a["agent_id"]: a for a in client.get("/admin/agents").json()["known_agents"]}

    assert "Pi" in by_id, "认领后的 agent 从界面上消失了"
    assert by_id["Pi"]["claimed"] is True
    assert by_id["Pi"]["has_rule"] is False        # 没填关键词 → 不会自动识别
    # 该状态在 UI 上必须有对应文案，否则用户不知道"还得补规则"
    assert "claimed, no rule" in _dashboard_html()


# ── 🔴 接线检查：认领 UX 的入口必须真的有东西可认 ──────────────────────────


def test_unrecognized_client_actually_shows_up_for_claiming(client) -> None:
    """🔴 一个没见过的客户端发一轮请求后，必须出现在待认领列表里。

    这条补的是一个**真实存在过的接线断裂**：`/admin/agents` 最初从
    `get_user_agents()` 里筛 `unknown-` 前缀，而未识别桶**有意不进那个结构**
    （进了会成为同源继承目标）→ 列表恒空 → 整个认领 UX 不可达。

    当时的测试只断言了 `known_agents` 与 `claims`，恰好绕开坏掉的那一格 ——
    "机制写完接线断"的又一例。所以这条不测函数，测**用户走得通**。
    """
    r = client.post(
        "/v1/chat/completions",
        headers={"user-agent": "totally-new-agent/0.1"},
        json={"model": "gpt-4o-mini", "stream": False,
              "messages": [{"role": "user", "content": "hi"}]},
    )
    # 上游没配也没关系 —— 身份解析在转发之前就跑完了
    assert r.status_code in (200, 400, 401, 402, 500, 502, 503)

    buckets = client.get("/admin/agents").json()["unknown_buckets"]
    assert buckets, "未识别客户端没有出现在待认领列表里 —— 认领 UX 不可达"
    entry = buckets[0]
    assert entry["bucket_id"].startswith("unknown-")
    assert entry["basis"] == "ua:totally-new-agent"     # 判据可读，用户据此认人
    assert "user-agent" in entry["headers"]              # header 快照带上了


def test_claimed_bucket_leaves_the_pending_list(client) -> None:
    """认领之后它不该还挂在待认领里（否则用户认一次它还在，以为没生效）。"""
    client.post(
        "/v1/chat/completions",
        headers={"user-agent": "another-new-agent/2.0"},
        json={"model": "gpt-4o-mini", "stream": False,
              "messages": [{"role": "user", "content": "hi"}]},
    )
    bucket_id = client.get("/admin/agents").json()["unknown_buckets"][0]["bucket_id"]

    client.post("/admin/agents/claim",
                json={"from_agent_ids": [bucket_id], "to_agent_id": "named-agent"})

    remaining = [b["bucket_id"] for b in
                 client.get("/admin/agents").json()["unknown_buckets"]]
    assert bucket_id not in remaining


# ── 🔴 前后端字段契约（页面空白 vs 链路没跑，必须能分开）─────────────────────


def _dashboard_html() -> str:
    from importlib import resources

    return (resources.files("bladex_proxy") / "dashboard.html").read_text(encoding="utf-8")


def _dashboard_const_block(name: str) -> str:
    """取出 `const <NAME>={...}` / `=[...]` 的整段文本（到下一个顶层 const 为止）。"""
    import re

    m = re.search(rf"const {name}\s*=\s*[{{\[](.*?)[}}\]];", _dashboard_html(), re.S)
    assert m, f"dashboard 里找不到 const {name}"
    return m.group(1)


def _dashboard_tab_ids() -> list[str]:
    import re

    return re.findall(r'\["([a-z]+)","[^"]+"\]', _dashboard_const_block("TABS"))


def test_every_tab_is_wired_into_all_four_maps() -> None:
    """🔴 每个 tab 必须同时进 TABS / SUBS / VIEWS / **IC** 四处。

    治的是这一类，不是我这一个：加 tab 的人容易只接前三处，第四处（图标表 `IC`）
    因为 `IC[id]||""` 有兜底，**漏了不报错、只是图标不见了**——静默降级。
    实际发生过：Agents 页上线后 Jason 一眼看出"跟别的标签不一样"，才发现漏接。

    这条测四张表的**交叉完整性**，所以下一个加 tab 的人漏哪张都会红。
    """
    tabs = _dashboard_tab_ids()
    assert len(tabs) >= 9, f"TABS 解析异常，只拿到 {tabs}"

    ic, subs, views = (_dashboard_const_block(n) for n in ("IC", "SUBS", "VIEWS"))
    missing: list[str] = []
    for tab in tabs:
        for label, block in (("IC 图标", ic), ("SUBS 副标题", subs), ("VIEWS 路由", views)):
            if f"{tab}:" not in block:
                missing.append(f"{tab} 缺 {label}")
    assert not missing, "导航接线不完整：\n  " + "\n  ".join(missing)


def test_agents_view_functions_exist() -> None:
    html = _dashboard_html()
    assert "agents" in _dashboard_tab_ids()
    assert "async function viewAgents()" in html   # 视图函数
    assert "async function agentClaim(" in html    # 认领交互


@pytest.mark.parametrize(
    "field",
    ["bucket_id", "basis", "headers", "count", "hub_turns"],
)
def test_bucket_field_names_match_between_api_and_dashboard(field: str, client) -> None:
    """🔴 端点返回的字段名与 dashboard 实际读的必须一致。

    对不上的后果不是报错，是**页面渲染成空**——"没有未识别客户端"和"UI 读错字段"
    在界面上长得一模一样。Pi 验收时这会让人分不清是链路没跑还是前端坏了，
    而现存的 dashboard 测试只检查标记字符串在不在 HTML 里，抓不到这一类。
    """
    client.post(
        "/v1/chat/completions",
        headers={"user-agent": "contract-probe-agent/1.0"},
        json={"model": "gpt-4o-mini", "stream": False,
              "messages": [{"role": "user", "content": "hi"}]},
    )
    entry = client.get("/admin/agents").json()["unknown_buckets"][0]
    assert field in entry, f"端点没返回 {field}"
    assert f"b.{field}" in _dashboard_html(), f"dashboard 没读 {field}（字段改名了？）"


@pytest.mark.parametrize("field", ["from", "to", "merge", "ts"])
def test_claim_field_names_match_between_api_and_dashboard(field: str, client) -> None:
    client.post("/admin/agents/claim",
                json={"from_agent_ids": ["unknown-contract"], "to_agent_id": "x"})
    entry = client.get("/admin/agents").json()["claims"][0]
    assert field in entry
    assert f"c.{field}" in _dashboard_html()


def test_dashboard_uses_only_existing_css_classes() -> None:
    """🔴 视图里引用的 class 必须真的定义过，否则样式静默失效。

    实际踩到：详情页第一版写了 `class="pre"` 与 `class="grid2"`，两个都是我编的——
    `pre` 在这份 dashboard 里是**元素选择器**不是类，`grid2` 根本不存在。
    浏览器不会报错，只是排版不对；而"排版不对"在截图之外没人看得见。
    """
    import re

    html = _dashboard_html()
    # 从 <style> 里取**选择器任意位置**出现的类名。只认紧跟 `{`/`,` 的写法会漏掉
    # 后代选择器（`.fields .f .fk{}` 里的 `.f`）——第一版就这么误报了一次。
    style = "\n".join(re.findall(r"<style[^>]*>(.*?)</style>", html, re.S))
    defined = set(re.findall(r"\.([a-zA-Z][\w-]*)", style))
    used: set[str] = set()
    for m in re.findall(r'class="([^"$]+)"', html):    # 跳过含模板插值的
        used |= set(m.split())
    missing = sorted(used - defined)
    assert not missing, f"dashboard 用了未定义的 class：{missing}"


def test_agent_lists_are_tables_not_flex_rows() -> None:
    """🔴 多列列表用表格，不用 `.kv`。

    实际炸过（Jason 2026-08-19 截图）：Agents 页三个列表都用 `.kv`，而
    `.kv{display:flex;justify-content:space-between}` 会把子元素**两端撑开**。
    每行子元素个数不同（有的有 profile、有的没有 turns），徽标和数字就飘在
    各自不同的横向位置上——数据全对，但看着是乱的。

    多列对齐是表格的活；这份 dashboard 本来就有 `.tbl-wrap` + `<table>` 的既有写法。
    """
    import re

    html = _dashboard_html()
    view = html.split("async function viewAgents()")[1].split("async function ")[0]
    assert view.count("<table>") == 3, "Agents 页的三个列表应各是一张表"
    assert 'class="kv"' not in view, "多列列表不该用 .kv（两端对齐会让列飘）"
    # 表头列数与每行单元格数必须一致，否则列会错位
    for thead, tbody in zip(re.findall(r"<thead>(.*?)</thead>", view, re.S),
                            re.findall(r"<tbody>(.*?)</tbody>", view, re.S), strict=True):
        assert thead.count("<th") == tbody.count("<td"), "表头列数与行内单元格数不一致"


def test_dashboard_follows_existing_markup_conventions() -> None:
    """🔴 新视图必须沿用既有写法，不许自创一套。

    实际踩到（Jason 2026-08-19：「链接加得太丑，风格要用原有 css 样式」）：
    详情页第一版写了 `<a href="javascript:void(0)" class="mono">`，而全文件既有约定是
    `<a class="link" onclick=...>`（无 href，靠 `a.link` 上色）；分节标题用了 `<h4>`，
    而 `<h4>` 只在 `.banner` 里有样式，抽屉里应该用 `.sect`。

    两处都不会报错，只是**长得跟别处不一样**——这类问题只有人眼看得见，
    所以把已经踩过的形状钉成规则。
    """
    html = _dashboard_html()
    assert 'href="javascript:void(0)"' not in html, (
        "可点击文本用 `<a class=\"link\" onclick=...>`，不要 href=javascript:void(0)"
    )
    # `<h4>` 仅限 .banner 内部：全文件出现次数应等于 banner 里的用量
    import re

    h4_total = len(re.findall(r"<h4>", html))
    h4_in_banner = len(re.findall(r'class="banner[^"]*"><h4>', html))
    assert h4_total == h4_in_banner, (
        f"抽屉/卡片里的分节标题请用 `<div class=\"sect\">`；"
        f"<h4> 共 {h4_total} 处，其中只有 {h4_in_banner} 处在 banner 内"
    )


def test_agent_detail_endpoint_reports_what_the_agent_did(tmp_path: Path) -> None:
    """详情页要给出足以认人的四样：system prompt / 工具 / header / 干了什么。

    用真 Memory Hub：详情靠 `keys_for_agent()` + 抽样 `get()`，替身测不到。
    """
    from fastapi.testclient import TestClient

    from bladex_proxy.models import Identity, Turn, TurnStatus
    from bladex_proxy.server import create_app
    from bladex_proxy.storage.memory_hub import MemoryHub

    ledger = MemoryHub(tmp_path / "ledger")
    ledger.open()
    identity = Identity(
        user_id="local", agent_id="unknown-feed0001", session_id="s1",
        agent_bucket_basis="sdk:openai/js|tools:abcd1234",
        request_headers={"user-agent": "OpenAI/JS 6.40.0", "x-stainless-lang": "js"},
    )
    ledger.put(
        "local/unknown-feed0001/s1/1700000000-0",
        Turn(identity=identity, model="openai/glm-5.3",
             request_messages=[
                 {"role": "system", "content": "You are a helpful coding agent named Ghost."},
                 {"role": "assistant", "content": "",
                  "tool_calls": [{"function": {"name": "bash"}}]},
                 {"role": "user", "content": "这个目录里有多少文件？"},
             ],
             response_text="ok", status=TurnStatus.OK),
    )

    app = create_app()
    with TestClient(app) as c:
        started = app.state.hub
        app.state.hub = ledger
        try:
            d = c.get("/admin/agents/unknown-feed0001/detail").json()
        finally:
            app.state.hub = started
    ledger.close()

    assert d["turns"] == 1 and d["sessions"] == 1
    assert d["recognized"] is False
    assert "Ghost" in d["system_excerpt"]              # ① 自我介绍最能定身份
    assert "bash" in d["tools"]                        # ② 工具集
    assert d["headers"]["x-stainless-lang"] == "js"    # ③ header 快照
    assert "多少文件" in d["recent_user_messages"][0]["text"]   # ④ 干了什么
    assert d["basis"] == "sdk:openai/js|tools:abcd1234"


@pytest.mark.parametrize(
    "field",
    ["turns", "sessions", "basis", "headers", "models", "tools",
     "system_excerpt", "recent_user_messages", "recognized"],
)
def test_detail_field_names_match_dashboard(field: str) -> None:
    """详情页字段名同样要与 dashboard 实读的对上（对不上就是空白面板）。"""
    assert f"d.{field}" in _dashboard_html(), f"dashboard 没读 {field}"


def test_rule_can_be_saved_from_the_dashboard(client, tmp_path, monkeypatch) -> None:
    """🔴 改规则不该逼用户去编 TOML（Jason 2026-08-19）。"""
    import bladex_proxy.agent_rules as ar

    target = tmp_path / "agent_rules.toml"
    monkeypatch.setattr(ar, "_discover_user_rules_path", lambda: target)

    r = client.post("/admin/agents/rules", json={"rule": {
        "name": "pi", "agent_id": "Pi",
        "system_prompt_keywords": ["You are Pi, a coding agent"],
        "tool_signatures": ["pi_edit"], "min_tool_match": 1,
    }})
    assert r.status_code == 200, r.text
    assert r.json()["effective_agent_ids"] == ["Pi"]
    assert target.exists()
    assert any(x.agent_id == "Pi" for x in ar.load_agent_rules(user_path=target))


def test_saving_an_invalid_rule_is_rejected(client) -> None:
    """非法规则响亮报错，不写半个坏文件。"""
    assert client.post("/admin/agents/rules",
                       json={"rule": {"agent_id": "x", "typo_field": 1}}).status_code == 400
    assert client.post("/admin/agents/rules", json={"rule": {}}).status_code == 400


def test_detail_exposes_current_rule_for_editing(client) -> None:
    """改名/编辑要预填**现有规则**，不是从流量重推——否则一改就丢手工调过的部分。"""
    d = client.get("/admin/agents/hermes/detail").json()
    names = {r["name"] for r in d["current_rules"]}
    assert "hermes" in names
    assert "d.current_rules" in _dashboard_html()


def test_top_level_response_keys_match(client) -> None:
    html = _dashboard_html()
    body = client.get("/admin/agents").json()
    for key in ("known_agents", "unknown_buckets", "claims"):
        assert key in body
        assert f"d.{key}" in html, f"dashboard 没读 {key}"


# ── 🔴 重启可存活：待认领列表必须从 Memory Hub 派生 ─────────────────────


def _seeded_ledger(tmp_path: Path):
    """造一个真 Memory Hub，里面躺着一个未识别桶的历史轮次。

    这组用真库不用替身：本节验的正是"进程内状态没了以后还剩什么"，
    拿替身测等于把被验的对象换掉了。（第一版这里确实用了替身，
    端点里的 try/except 把 `scan_agent_ids` 不存在这件事静默吞掉，
    26 个用例全绿而那条路径一行都没跑到。）
    """
    from bladex_proxy.models import Identity, Turn, TurnStatus
    from bladex_proxy.storage.memory_hub import MemoryHub

    ledger = MemoryHub(tmp_path / "ledger")
    ledger.open()
    identity = Identity(
        user_id="local", agent_id="unknown-cafe1234", session_id="s1",
        agent_bucket_basis="ua:ghost-agent",
        request_headers={"user-agent": "ghost-agent/1.0",
                         "authorization": "<redacted>"},
    )
    ledger.put(
        "local/unknown-cafe1234/s1/1700000000-0",
        Turn(identity=identity, model="m",
             request_messages=[{"role": "user", "content": "hi"}],
             response_text="ok", status=TurnStatus.OK),
    )
    return ledger


def test_scan_agent_ids_reads_keys_only(tmp_path: Path) -> None:
    """`scan_agent_ids` 从 key 第二段取 agent_id，不反序列化 value。"""
    ledger = _seeded_ledger(tmp_path)
    try:
        found = ledger.scan_agent_ids()
        assert "unknown-cafe1234" in found
        rep_key, count = found["unknown-cafe1234"]
        assert rep_key.startswith("local/unknown-cafe1234/")
        assert count == 1
    finally:
        ledger.close()


def test_bucket_survives_restart_via_ledger(tmp_path: Path) -> None:
    """🔴 进程内登记为空（模拟刚重启）时，待认领列表仍须列出历史未识别桶。

    没有这条，管理页只在"本次启动以来恰好来过"时才有东西——大多数时候是空的，
    等于没有管理接口；而已经发生过的误署名更是永远无从处理。
    """
    from fastapi.testclient import TestClient

    from bladex_proxy.agent_registry import _agent_registry
    from bladex_proxy.server import create_app

    ledger = _seeded_ledger(tmp_path)
    app = create_app()
    with TestClient(app) as c:
        started_ledger = app.state.hub
        app.state.hub = ledger
        _agent_registry.clear()          # 进程内登记空 = 刚重启的形态
        try:
            body = c.get("/admin/agents").json()
        finally:
            app.state.hub = started_ledger
    ledger.close()

    entry = next(b for b in body["unknown_buckets"]
                 if b["bucket_id"] == "unknown-cafe1234")
    assert entry["source"] == "hub"
    assert entry["hub_turns"] == 1
    assert entry["basis"] == "ua:ghost-agent"        # 判据串从代表 turn 取回来了
    assert "user-agent" in entry["headers"]


def test_known_list_folds_profiles_and_sorts_legacy_last(tmp_path: Path) -> None:
    """🔴 已知列表要能看：profile 变体归并、历史残留沉底、裸 unknown 归未识别侧。

    实际炸过（Jason 2026-08-19：「已知 agent 列表全乱了」）：live 总账里有 14 个
    agent_id，我把它们原样倒进 Known agents ——
      · `hermes:default` / `hermes:accept` / `hermes:c7047465` 等**同一个 agent 的
        5 个 profile** 摊平成 5 行（规则只挂在 base 上，摊平既乱又误导）；
      · `a` / `test-agent` / `tool-test` 这些测试残留和真 agent 混排；
      · 裸 `unknown`（G11.9 之前的共用桶）被当成"已知 agent"。
    """
    from fastapi.testclient import TestClient

    from bladex_proxy.agent_registry import _agent_registry
    from bladex_proxy.models import Identity, Turn, TurnStatus
    from bladex_proxy.server import create_app
    from bladex_proxy.storage.memory_hub import MemoryHub

    ledger = MemoryHub(tmp_path / "ledger")
    ledger.open()
    for agent, n in [("hermes:default", 3), ("hermes:accept", 2),
                     ("test-agent", 1), ("unknown", 1)]:
        for i in range(n):
            ledger.put(
                f"local/{agent}/s1/17000000{i:02d}-0",
                Turn(identity=Identity(user_id="local", agent_id=agent, session_id="s1"),
                     model="m", request_messages=[{"role": "user", "content": "hi"}],
                     response_text="ok", status=TurnStatus.OK),
            )

    app = create_app()
    with TestClient(app) as c:
        started = app.state.hub
        app.state.hub = ledger
        _agent_registry.clear()
        try:
            body = c.get("/admin/agents").json()
        finally:
            app.state.hub = started
    ledger.close()

    by_id = {a["agent_id"]: a for a in body["known_agents"]}
    # ① profile 归并到 base，轮次累加
    assert "hermes:default" not in by_id and "hermes:accept" not in by_id
    assert by_id["hermes"]["turns"] == 5
    assert by_id["hermes"]["profiles"] == ["hermes:accept", "hermes:default"]
    # ② 裸 unknown 归未识别侧，不进已知列表
    assert "unknown" not in by_id
    assert "unknown" in [b["bucket_id"] for b in body["unknown_buckets"]]
    # ③ 无规则无认领的历史残留标 seen only，且排在有规则的后面
    ids = [a["agent_id"] for a in body["known_agents"]]
    assert by_id["test-agent"]["has_rule"] is False
    assert by_id["test-agent"]["claimed"] is False
    assert ids.index("test-agent") > ids.index("hermes")


def test_claim_target_inherits_the_bucket_turn_count(tmp_path: Path) -> None:
    """🔴 认领目标的轮次要算上它认领的桶。

    Memory Hub key 里仍写着 `unknown-xxxx`（认领是**读侧映射**，不重写总账），
    所以认领目标自己的 key 数天然是 0。照实显示 `Pi 0 turn(s)` 会误导——
    那几轮记忆下次重建就归 Pi 了，说它 0 轮等于告诉用户认领没生效
    （Jason 2026-08-19 追问"为什么已知里有 Pi"时暴露）。
    """
    from fastapi.testclient import TestClient

    from bladex_proxy.agent_registry import _agent_registry
    from bladex_proxy.models import Identity, Turn, TurnStatus
    from bladex_proxy.server import create_app
    from bladex_proxy.storage.memory_hub import MemoryHub

    ledger = MemoryHub(tmp_path / "ledger")
    ledger.open()
    for i in range(4):
        ledger.put(
            f"local/unknown-cafe0002/s1/17000000{i:02d}-0",
            Turn(identity=Identity(user_id="local", agent_id="unknown-cafe0002",
                                   session_id="s1"),
                 model="m", request_messages=[{"role": "user", "content": "hi"}],
                 response_text="ok", status=TurnStatus.OK),
        )

    app = create_app()
    with TestClient(app) as c:
        started = app.state.hub
        app.state.hub = ledger
        _agent_registry.clear()
        try:
            r = c.post("/admin/agents/claim",
                       json={"from_agent_ids": ["unknown-cafe0002"], "to_agent_id": "Zed"})
            assert r.status_code == 200, r.text
            body = c.get("/admin/agents").json()
        finally:
            app.state.hub = started
    ledger.close()

    by_id = {a["agent_id"]: a for a in body["known_agents"]}
    assert by_id["Zed"]["turns"] == 4, "认领目标没继承被认领桶的轮次"
    assert "unknown-cafe0002" not in [b["bucket_id"] for b in body["unknown_buckets"]]


def test_claimed_bucket_stays_gone_after_restart(tmp_path: Path) -> None:
    """认领过的桶不得因为重启又冒出来——认领是 journal 里的事实，不是进程内记忆。"""
    from fastapi.testclient import TestClient

    from bladex_proxy.agent_registry import _agent_registry
    from bladex_proxy.server import create_app

    ledger = _seeded_ledger(tmp_path)
    app = create_app()
    with TestClient(app) as c:
        started_ledger = app.state.hub
        app.state.hub = ledger
        _agent_registry.clear()
        try:
            # 🔴 前置断言：认领之前它必须在列表里。没有这条，"认领后不在"会在
            # 整条 ledger 路径失效时**自动成立**——阳性对照实测：把
            # `scan_agent_ids` 改名后另两条红了，本条照绿（断言为真但理由是错的）。
            before = [b["bucket_id"] for b in c.get("/admin/agents").json()["unknown_buckets"]]
            assert "unknown-cafe1234" in before, "前置不成立，后面的断言是空转的"

            r = c.post("/admin/agents/claim",
                       json={"from_agent_ids": ["unknown-cafe1234"],
                             "to_agent_id": "ghost"})
            assert r.status_code == 200, r.text
            _agent_registry.clear()      # 再次模拟重启：进程内什么都不记得了
            body = c.get("/admin/agents").json()
        finally:
            app.state.hub = started_ledger
    ledger.close()

    ids = [b["bucket_id"] for b in body["unknown_buckets"]]
    assert "unknown-cafe1234" not in ids
    assert body["claims"][-1]["to"] == "ghost"


def test_credentials_absent_from_the_claim_listing(client) -> None:
    """🔴 负例：待认领列表会把 header 快照显示给用户，凭证不得混在里面。"""
    client.post(
        "/v1/chat/completions",
        headers={"user-agent": "secretive-agent/1.0",
                 "authorization": "Bearer sk-must-not-be-listed"},
        json={"model": "gpt-4o-mini", "stream": False,
              "messages": [{"role": "user", "content": "hi"}]},
    )
    body = client.get("/admin/agents").text
    assert "sk-must-not-be-listed" not in body
