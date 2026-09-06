"""V-L1 验收剧本（批一卡 §V-L1）：往返等价 / Goal 写保护 / 事件重放等价 / 增段扩展性。"""

from __future__ import annotations

import pytest

from bladex_core.ledger import (
    ACTOR_DASHBOARD,
    ACTOR_MAINTENANCE,
    ACTOR_MODEL,
    ACTOR_USER,
    DEFAULT_SECTION_ORDER,
    ENTRY_SOURCE_MODEL,
    ENTRY_SOURCE_TOOL,
    EVENT_LEDGER_CREATE,
    EVENT_LEDGER_SWITCH,
    EVENT_LEDGER_UPDATE,
    EVENT_LEDGER_USER_EDIT,
    GoalWriteViolation,
    Ledger,
    LedgerEntry,
    LedgerError,
    add_entry,
    add_section,
    ledger_local_path,
    ledger_md_path,
    ledgers_dir,
    new_ledger,
    new_ledger_id,
    parse_ledger_md,
    remove_entry,
    render_ledger_md,
    render_template,
    replay_ledger_events,
    update_goal,
)


def _sample() -> Ledger:
    led = new_ledger(ledger_id="ldg-abc123def456", title="修复路由回归",
                     goal="修复 auxiliary 旁路 req_model 的回归，pytest 全绿",
                     goal_source="user", goal_origin="hermes/default/s1/0001",
                     created_at="2026-08-25T10:00:00Z")
    led = add_entry(led, "core", LedgerEntry(
        text="routing.toml 的静态策略优先级最高", source=ENTRY_SOURCE_MODEL),
        actor=ACTOR_MODEL)
    led = add_entry(led, "verified", LedgerEntry(
        text="pytest 783 全绿", source=ENTRY_SOURCE_TOOL, ref="gate_check#20260728"),
        actor=ACTOR_MODEL)
    led = add_entry(led, "open", LedgerEntry(text="hermes aux 指纹未补齐"),
                    actor=ACTOR_MODEL)
    led = add_entry(led, "next", LedgerEntry(
        text="补 DISTILL_ONLY_AUX_RULES 回归测试", source=ENTRY_SOURCE_MODEL),
        actor=ACTOR_MODEL)
    return led


# ── 往返等价 ────────────────────────────────────────────────────────────────

class TestRoundTrip:
    def test_render_parse_render_is_identity(self):
        led = _sample()
        md = render_ledger_md(led)
        assert render_ledger_md(parse_ledger_md(md)) == md

    def test_parse_recovers_fields(self):
        led2 = parse_ledger_md(render_ledger_md(_sample()))
        assert led2.ledger_id == "ldg-abc123def456"
        assert led2.title == "修复路由回归"
        assert led2.goal.startswith("修复 auxiliary")
        assert led2.goal_source == "user"
        assert led2.goal_origin == "hermes/default/s1/0001"
        assert led2.entries("verified")[0].ref == "gate_check#20260728"
        assert led2.entries("open")[0].source == "user"  # 裸行默认 user
        assert led2.section_order == list(DEFAULT_SECTION_ORDER)

    def test_render_is_deterministic_no_wall_clock(self):
        led = _sample()
        assert render_ledger_md(led) == render_ledger_md(led)

    def test_hand_edited_bare_entry_defaults_to_user(self):
        md = render_ledger_md(_sample()).replace(
            "_(empty)_", "", 1)  # no-op guard; append bare line below
        md += ""  # keep explicit
        md_lines = md.splitlines()
        # 用户在 Next 段手加一行（无来源标记）
        idx = md_lines.index("## Next") + 1
        md_lines.insert(idx, "- 顺手把 README 改了")
        led2 = parse_ledger_md("\n".join(md_lines) + "\n")
        texts = {e.text: e.source for e in led2.entries("next")}
        assert texts["顺手把 README 改了"] == "user"

    def test_empty_template_round_trips(self):
        md = render_template()
        led = parse_ledger_md(md)
        assert led.section_order == list(DEFAULT_SECTION_ORDER)
        assert render_ledger_md(led) != ""


# ── Goal 写保护（红线 1）────────────────────────────────────────────────────

class TestGoalProtection:
    def test_model_cannot_write_goal(self, caplog):
        led = _sample()
        with caplog.at_level("WARNING"):
            with pytest.raises(GoalWriteViolation):
                update_goal(led, "偷偷改目标", actor=ACTOR_MODEL)
        assert any("ledger_goal_write_denied" in r.message for r in caplog.records)

    def test_maintenance_cannot_write_goal(self):
        with pytest.raises(GoalWriteViolation):
            update_goal(_sample(), "自检顺手修正", actor=ACTOR_MAINTENANCE)

    @pytest.mark.parametrize("actor", [ACTOR_USER, ACTOR_DASHBOARD])
    def test_user_and_dashboard_can_write_goal(self, actor):
        led = update_goal(_sample(), "新目标", actor=actor,
                          updated_at="2026-08-25T11:00:00Z")
        assert led.goal == "新目标"
        assert led.goal_source == "user"
        assert led.updated_at == "2026-08-25T11:00:00Z"

    def test_goal_is_not_an_entry_section(self):
        with pytest.raises(LedgerError):
            add_entry(_sample(), "goal", LedgerEntry(text="x"), actor=ACTOR_USER)


# ── 段扩展性（红线 4 / 拍板 #8）────────────────────────────────────────────

class TestSectionExtensibility:
    def test_add_lessons_section_round_trips(self):
        led = add_section(_sample(), "lessons")
        led = add_entry(led, "lessons", LedgerEntry(
            text="调函数前先 grep 签名", source=ENTRY_SOURCE_MODEL), actor=ACTOR_MODEL)
        md = render_ledger_md(led)
        led2 = parse_ledger_md(md)
        assert "lessons" in led2.section_order
        assert led2.entries("lessons")[0].text == "调函数前先 grep 签名"
        assert render_ledger_md(led2) == md

    def test_unknown_section_in_file_is_preserved(self):
        md = render_ledger_md(_sample())
        md += "\n## Scratch\n\n- 用户自己加的段落里的一条\n"
        led = parse_ledger_md(md)
        assert "scratch" in led.section_order
        assert led.entries("scratch")[0].text == "用户自己加的段落里的一条"

    def test_write_to_unregistered_section_refused(self):
        with pytest.raises(LedgerError):
            add_entry(_sample(), "lesons", LedgerEntry(text="typo 段名"),
                      actor=ACTOR_MODEL)

    def test_add_section_validates_key(self):
        with pytest.raises(LedgerError):
            add_section(_sample(), "Bad Key!")


# ── 条目原语 ────────────────────────────────────────────────────────────────

class TestEntryOps:
    def test_remove_entry_returns_removed(self):
        led = _sample()
        led2, removed = remove_entry(led, "next", 0)
        assert removed.text.startswith("补 DISTILL_ONLY")
        assert led2.entries("next") == []
        assert led.entries("next")  # 原对象不变（immutability）

    def test_remove_out_of_range(self):
        with pytest.raises(LedgerError):
            remove_entry(_sample(), "next", 5)


# ── 池化路径 ────────────────────────────────────────────────────────────────

class TestPoolPaths:
    def test_pool_is_user_scope_level_with_shard(self):
        """两级哈希分桶（Jason 2026-08-28 拍板，一套方案无旋钮）：
        桶 = id 去前缀后 hex 第 1–2 / 3–4 位，从 id 自身确定性派生。"""
        p = ledger_md_path("/root", "hash8", "ldg-66d05bda21d0")
        assert p == "/root/hash8/personal/ledgers/66/d0/ldg-66d05bda21d0.md"

    def test_shard_is_deterministic_and_bounded(self):
        from bladex_core.ledger import ledger_shard, new_ledger_id
        assert ledger_shard("ldg-66d05bda21d0") == "66/d0"
        assert ledger_shard("weird") == "we/ir"
        assert ledger_shard("x") == "x0/00" and ledger_shard("") == "00/00"
        for _ in range(64):
            a, b = ledger_shard(new_ledger_id()).split("/")
            assert len(a) == 2 and len(b) == 2

    def test_local_override_path(self):
        assert ledger_local_path("/root", "hash8", "ldg-abc").endswith(
            "/ledgers/ab/c0/ldg-abc.local.md")

    def test_path_has_no_agent_dimension(self):
        import inspect
        from bladex_core import ledger as mod
        for fn in (mod.ledger_md_path, mod.ledger_local_path, mod.ledgers_dir):
            assert "agent" not in inspect.signature(fn).parameters

    def test_unsafe_ledger_id_sanitized(self):
        p = ledger_md_path("/root", "hash8", "../evil")
        assert "/../" not in p

    def test_ledger_id_content_independent(self):
        a, b = new_ledger_id(), new_ledger_id()
        assert a != b and a.startswith("ldg-") and len(a) == 16


# ── 事件重放等价（红线 2）────────────────────────────────────────────────────

class TestParentChild:
    """V-L6e（Jason 2026-08-25）：子任务可以建账本，但必须与父账本互相可见。"""

    def test_child_records_parent_and_renders_it(self):
        from bladex_core.ledger import children_of
        parent = new_ledger(ledger_id="ldg-p", title="父任务")
        child = new_ledger(ledger_id="ldg-c", title="子任务",
                           parent_ledger_id="ldg-p")
        assert child.parent_ledger_id == "ldg-p"
        assert "parent ledger: `ldg-p`" in render_ledger_md(child)
        # 父侧由池派生（单一方向存储，不双写）
        pool = {"ldg-p": parent, "ldg-c": child}
        assert [c.ledger_id for c in children_of(pool, "ldg-p")] == ["ldg-c"]
        assert children_of(pool, "ldg-c") == []

    def test_children_sorted_by_recency(self):
        from bladex_core.ledger import children_of
        pool = {
            "a": new_ledger(ledger_id="a", parent_ledger_id="p",
                            created_at="2026-08-25T01:00:00Z"),
            "b": new_ledger(ledger_id="b", parent_ledger_id="p",
                            created_at="2026-08-25T02:00:00Z"),
        }
        assert [c.ledger_id for c in children_of(pool, "p")] == ["b", "a"]

    def test_parent_link_survives_round_trip(self):
        child = new_ledger(ledger_id="ldg-c", title="子", parent_ledger_id="ldg-p")
        assert parse_ledger_md(render_ledger_md(child)).parent_ledger_id == "ldg-p"

    def test_no_parent_renders_nothing(self):
        assert "parent ledger" not in render_ledger_md(new_ledger(ledger_id="x"))


class TestReplay:
    def test_pool_rebuilds_from_events(self):
        led = _sample()
        led_v2 = update_goal(led, "目标改了", actor=ACTOR_USER,
                             updated_at="2026-08-25T12:00:00Z")
        events = [
            {"type": EVENT_LEDGER_CREATE, "payload": {"ledger": led.model_dump()}},
            {"type": EVENT_LEDGER_UPDATE, "payload": {"ledger": led_v2.model_dump()}},
            {"type": EVENT_LEDGER_SWITCH, "payload": {
                "session_id": "s1", "agent_id": "hermes",
                "from_ledger_id": "", "to_ledger_id": led.ledger_id}},
        ]
        pool, bindings = replay_ledger_events(events)
        assert render_ledger_md(pool[led.ledger_id]) == render_ledger_md(led_v2)
        # 🔴 2026-08-26 拍板（MQ-L11）：绑定键从 session 改 (agent, project)。
        # 历史事件没有 project_id ⇒ Global。零迁移：agent_id 从第一天就在 payload 里。
        from bladex_core.ledger_runtime import activation_scope
        assert bindings == {activation_scope("hermes"): led.ledger_id}

    def test_binding_key_is_agent_not_session(self):
        """同一 agent 跨 session（context window 交接）**保持**同一激活账本。

        病例（MQ-L11，jydesignhk）：Hermes 压缩后首条 user 变成
        `[CONTEXT COMPACTION — REFERENCE ONLY] … handoff from a previous context
        window`，指纹随之变成新 `fp:` ⇒ 按 session 绑定时旧账本**看不见了**，
        模型拿到的是冷启动而不是 Goal 对照。跨会话交接正是北极星要接住的场景。
        """
        led = _sample()
        events = [
            {"type": EVENT_LEDGER_CREATE, "payload": {"ledger": led.model_dump()}},
            {"type": EVENT_LEDGER_SWITCH, "payload": {
                "session_id": "fp:955be74ada9f", "agent_id": "hermes:default",
                "from_ledger_id": "", "to_ledger_id": led.ledger_id}},
        ]
        _pool, bindings = replay_ledger_events(events)
        from bladex_core.ledger_runtime import activation_scope
        # 压缩之后的新 session 用同一 agent 来取 —— 必须还能取到
        assert bindings.get(activation_scope("hermes:default")) == led.ledger_id
        assert "fp:955be74ada9f" not in bindings

    def test_same_agent_different_project_do_not_collide(self):
        """两个 Claude Code 窗口开在不同仓库 ⇒ 不许抢同一本（per-agent 的回归风险）。"""
        a, b = _sample(), _sample()
        object.__setattr__  # noqa: B018 —— 仅表明下面用 model_copy 而非原地改
        b = b.model_copy(update={"ledger_id": "ldg-bbbbbbbbbbbb"})
        _pool, bindings = replay_ledger_events([
            {"type": EVENT_LEDGER_CREATE, "payload": {"ledger": a.model_dump()}},
            {"type": EVENT_LEDGER_CREATE, "payload": {"ledger": b.model_dump()}},
            {"type": EVENT_LEDGER_SWITCH, "payload": {
                "session_id": "s1", "agent_id": "claude-code",
                "project_id": "git-repo-a", "to_ledger_id": a.ledger_id}},
            {"type": EVENT_LEDGER_SWITCH, "payload": {
                "session_id": "s2", "agent_id": "claude-code",
                "project_id": "git-repo-b", "to_ledger_id": b.ledger_id}},
        ])
        from bladex_core.ledger_runtime import activation_scope
        assert bindings[activation_scope("claude-code", "git-repo-a")] == a.ledger_id
        assert bindings[activation_scope("claude-code", "git-repo-b")] == b.ledger_id

    def test_user_edit_event_wins_last(self):
        led = _sample()
        edited = led.model_copy(update={"goal": "用户直编后的目标"})
        pool, _ = replay_ledger_events([
            {"type": EVENT_LEDGER_CREATE, "payload": {"ledger": led.model_dump()}},
            {"type": EVENT_LEDGER_USER_EDIT, "payload": {"ledger": edited.model_dump()}},
        ])
        assert pool[led.ledger_id].goal == "用户直编后的目标"

    def test_bad_event_does_not_break_replay(self):
        led = _sample()
        pool, _ = replay_ledger_events([
            {"type": EVENT_LEDGER_CREATE, "payload": {"ledger": {"nonsense": 1}}},
            {"type": EVENT_LEDGER_CREATE, "payload": {"ledger": led.model_dump()}},
        ])
        assert led.ledger_id in pool and len(pool) == 1


# ── 与 proxy 枚举对账（事件串一致性）────────────────────────────────────────

class TestEventParity:
    def test_event_strings_match_proxy_enum(self):
        try:
            from bladex_proxy.models import AdminEventType
        except ImportError:
            pytest.skip("bladex_proxy not importable in this environment")
        assert AdminEventType.LEDGER_CREATE.value == EVENT_LEDGER_CREATE
        assert AdminEventType.LEDGER_UPDATE.value == EVENT_LEDGER_UPDATE
        assert AdminEventType.LEDGER_SWITCH.value == EVENT_LEDGER_SWITCH
        assert AdminEventType.LEDGER_USER_EDIT.value == EVENT_LEDGER_USER_EDIT


# ── 时区纪律（MQ-L46，2026-09-05）：存 UTC / 比数值 / 显本地 ─────────────────
#
# 修前样本：`ldg-d50ebb77820d` 的 `updated: 2026-09-05T04:58:56+00:00`，Pi 在 CST
# 12:58:56 写的——值没错，三个面（MD / 注入块 / CLI）原样打 UTC，用户读到早 8 小时。
# 不能只改渲染：两处按 ISO 串排序 + parse 把文件串原样读回池 ⇒ 文件里出现 `+08:00`
# 而池里还有 `+00:00` 时字符串序错，候选面 top-5 排错。

import time as _time


@pytest.fixture(params=["Asia/Shanghai", "America/Los_Angeles"])
def pinned_tz(request, monkeypatch):
    """钉两个时区各跑一遍（一东一西，且 LA 有夏令时）。"""
    monkeypatch.setenv("TZ", request.param)
    _time.tzset()
    yield request.param
    monkeypatch.undo()
    _time.tzset()


class TestTimeZone:
    UTC_S = "2026-09-05T04:58:56+00:00"

    def test_normalize_any_offset_to_utc(self):
        from bladex_core.ledger import normalize_iso_utc
        assert normalize_iso_utc("2026-09-05T12:58:56+08:00") == self.UTC_S
        assert normalize_iso_utc("2026-09-04T21:58:56-07:00") == self.UTC_S
        assert normalize_iso_utc("2026-09-05T04:58:56Z") == self.UTC_S
        # naive 串按 UTC 解（与 iso_ms 同判）
        assert normalize_iso_utc("2026-09-05T04:58:56") == self.UTC_S
        # 空 / 坏串原样返回——不伪造时间
        assert normalize_iso_utc("") == ""
        assert normalize_iso_utc("not-a-time") == "not-a-time"

    def test_local_iso_uses_process_timezone(self, pinned_tz):
        from bladex_core.ledger import local_iso
        out = local_iso(self.UTC_S)
        expected = {"Asia/Shanghai": "2026-09-05T12:58:56+08:00",
                    "America/Los_Angeles": "2026-09-04T21:58:56-07:00"}[pinned_tz]
        assert out == expected
        assert local_iso("") == "" and local_iso("garbage") == "garbage"

    def test_render_shows_local_but_pool_stays_utc(self, pinned_tz):
        from bladex_core.ledger import local_iso
        led = new_ledger(ledger_id="ldg-tz", title="t", created_at=self.UTC_S)
        md = render_ledger_md(led)
        assert f"- created: {local_iso(self.UTC_S)}" in md
        assert self.UTC_S not in md                      # 面上不再出现 UTC
        back = parse_ledger_md(md)
        assert back.created_at == self.UTC_S             # 读回池：归一回 UTC
        assert back.updated_at == self.UTC_S

    def test_render_parse_render_round_trip_under_tz(self, pinned_tz):
        led = new_ledger(ledger_id="ldg-rt", title="t", created_at=self.UTC_S)
        md = render_ledger_md(led)
        assert render_ledger_md(parse_ledger_md(md)) == md

    def test_user_hand_written_offset_is_normalized(self):
        """用户直编文件里手写 `+08:00`（或裸串）⇒ 池里仍是 UTC，不混偏移。"""
        md = render_ledger_md(new_ledger(ledger_id="ldg-u", title="t", created_at=self.UTC_S))
        md = md.replace(f"- updated: ", "- updated: 2026-09-05T13:00:00+08:00\n- x_updated: ", 1)
        led = parse_ledger_md(md)
        assert led.updated_at == "2026-09-05T05:00:00+00:00"

    def test_children_sorted_by_real_instant_not_string(self):
        """混偏移排序不错序：`+08:00` 的 01:00 其实早于 `+00:00` 的 00:30。"""
        from bladex_core.ledger import children_of
        pool = {
            "late": new_ledger(ledger_id="late", parent_ledger_id="p",
                               created_at="2026-09-05T00:30:00+00:00"),
            "early": new_ledger(ledger_id="early", parent_ledger_id="p",
                                created_at="2026-09-05T01:00:00+08:00"),  # = 前一天 17:00Z
        }
        assert [c.ledger_id for c in children_of(pool, "p")] == ["late", "early"]
        # 判别力对照：字符串序会把 "01:00+08" 排在 "00:30+00" 前面——即修前的错
        assert sorted(pool, key=lambda k: pool[k].created_at, reverse=True) == ["early", "late"]

    def test_iso_ms_offsets_are_real_instants(self):
        from bladex_core.ledger import iso_ms
        assert iso_ms("2026-09-05T12:58:56+08:00") == iso_ms(self.UTC_S)
        # ledger_runtime 的导出名仍可用（调用方路径不变）
        from bladex_core.ledger_runtime import iso_ms as rt_iso_ms
        assert rt_iso_ms is iso_ms
