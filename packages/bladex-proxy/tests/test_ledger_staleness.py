"""V-L4 陈旧兜底 **生产接线**（ADR-0032 §4.4）。

🔴 **为什么这个文件必须存在**：MQ-L17 —— `staleness_marker` 有单测、有 flag
默认值、有 `.env.example` 条目，**生产侧零调用方**。从外面看跟"已实现"分不出来。
live 后果：`ldg-c9ab` 169 轮只更新 5 次（每 34 轮一次），系统里没有任何一处会说话。
同型：MQ-L14（锚层报告自己已就绪却一条都没命中）。

钉住：
  ① 阳性：账本上过了 ≥N 轮没更新 ⇒ 注入块里出现陈旧标记；
  ② 阴性：刚更新过 ⇒ 不出现（**误报比漏报有害**——对刚更新的账本喊陈旧
    会让模型去怀疑正确的状态）；
  ③ 更新即归零；新建即归零；
  ④ 计数只随**真实请求**走，不随渲染次数漂移；
  ⑤ 无工具面（weak 档 / C 方案）⇒ 不出现（有令无器，MQ-L7 同一条理由）；
  ⑥ 阈值 0 = 关。
"""

from __future__ import annotations

from bladex_core.ledger import new_ledger
from bladex_core.ledger_runtime import activation_scope
from bladex_proxy.agency import AgencyRuntime

_AGENT = "test-agent"
_SCOPE_LID = "ldg-stale0000001"


def _agency(monkeypatch, *, threshold: int = 5) -> AgencyRuntime:
    for mod in ("LEDGER", "TOOLFACE", "INTERCEPTION"):
        monkeypatch.setenv(f"BLADEX_MODULE_{mod}", "1")
    monkeypatch.setenv("BLADEX_LEDGER_STALE_TURNS", str(threshold))
    a = AgencyRuntime()
    # 🔴 前置断言：机制真的开着。缺了它，下面那些"不该出现标记"的阴性用例
    # 会在**整块根本没注**的情况下全绿——第一版就是这样，5 个用例假绿。
    assert a.ledger_on and a.toolface_on
    led = new_ledger(ledger_id=_SCOPE_LID, title="长任务",
                     goal="把这件事做完", goal_source="user",
                     created_at="2026-08-26T10:00:00Z")
    a.pool[led.ledger_id] = led
    a.activation.restore({activation_scope(_AGENT, ""): led.ledger_id})
    return a


def _history(a: AgencyRuntime, *, tool_calls: int = 0) -> list[dict]:
    """构造一条对话历史：每调一次 = 新的一个**用户轮**（E0.2 口径：role=user 消息数）；
    `tool_calls` > 0 时在末条 user 之后追加 assistant/tool 对（同一用户轮内的工具往返）。"""
    n = a.__dict__.setdefault("_test_user_turns", 0) + 1
    a.__dict__["_test_user_turns"] = n
    msgs: list[dict] = []
    for i in range(n):
        msgs.append({"role": "user", "content": f"继续 {i}"})
        if i < n - 1:
            msgs.append({"role": "assistant", "content": f"好 {i}"})
    for k in range(tool_calls):
        msgs.append({"role": "assistant", "content": None,
                     "tool_calls": [{"id": f"c{k}", "type": "function",
                                     "function": {"name": "bash", "arguments": "{}"}}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{k}", "content": "ok"})
    return msgs


def _turn(a: AgencyRuntime, *, with_instruction: bool = True) -> str:
    """跑一轮真实注入路径（= 一个新的用户轮），返回账本块正文。

    🔴 块必须真的注进去——返回空串会让所有"不该出现标记"的断言假绿。
    """
    out = a.insert_ledger_block(_history(a), agent_id=_AGENT,
                                with_instruction=with_instruction, session_id="s1")
    assert len(out) >= 2, "账本块没注进来——阴性断言会失去判别力"
    return str(out[-1].get("content") or "")


def test_marker_appears_after_threshold_turns(monkeypatch):
    """① 阳性对照。"""
    a = _agency(monkeypatch, threshold=5)
    seen = [_turn(a) for _ in range(5)]
    assert not any("turns on this ledger" in s for s in seen[:4]), \
        "阈值前就报了——误报比漏报有害"
    assert "5 turns on this ledger" in seen[4]
    assert "bladex_ledger_update" in seen[4]


def test_fresh_ledger_is_never_marked(monkeypatch):
    """② 阴性对照：刚建的账本不该被说成陈旧。"""
    a = _agency(monkeypatch, threshold=5)
    assert "turns on this ledger" not in _turn(a)


def test_update_resets_the_count(monkeypatch):
    """③ 更新即归零。"""
    import asyncio

    a = _agency(monkeypatch, threshold=5)
    for _ in range(5):
        _turn(a)
    assert "turns on this ledger" in _turn(a)

    asyncio.run(a._h_ledger_update(
        {"section": "verified", "op": "add", "text": "确认了一件事"},
        allowed_exposure="local",
        context={"agent_id": _AGENT, "session_id": "s1", "project_id": ""}))
    assert a._turns_since_update[_SCOPE_LID] == 0
    assert "turns on this ledger" not in _turn(a)


def test_count_follows_requests_not_renders(monkeypatch):
    """④ 计数只随真实请求走。

    渲染函数在测试与内部路径里会被反复调用；副作用若放在渲染里，计数会随
    调用次数漂移，阈值就没有稳定含义。
    """
    a = _agency(monkeypatch, threshold=5)
    _turn(a)
    before = a._turns_since_update[_SCOPE_LID]
    for _ in range(3):
        a.ledger_injection_message(_AGENT)          # 纯渲染
    assert a._turns_since_update[_SCOPE_LID] == before == 1


def test_no_toolface_no_marker(monkeypatch):
    """⑤ 没有工具面就不提工具——有令无器是纯注意力税（C 方案，MQ-L7）。"""
    a = _agency(monkeypatch, threshold=5)
    for _ in range(9):
        _turn(a, with_instruction=False)
    body = _turn(a, with_instruction=False)
    assert "turns on this ledger" not in body
    # 但只读正文照常给（看得见 ≠ 改得了）
    assert "把这件事做完" in body


def test_aux_turns_do_not_count(monkeypatch):
    """aux 轮不注账本块，也就不该计进"在这本账本上过了多少轮"。"""
    a = _agency(monkeypatch, threshold=5)
    for _ in range(10):
        a.insert_ledger_block(_history(a), agent_id=_AGENT, auxiliary=True,
                              session_id="s1")
    assert a._turns_since_update[_SCOPE_LID] == 0
    assert "turns on this ledger" not in _turn(a)


def test_threshold_zero_disables(monkeypatch):
    """⑥ 阈值 0 = 关（回归通道）。"""
    a = _agency(monkeypatch, threshold=0)
    for _ in range(30):
        _turn(a)
    assert "turns on this ledger" not in _turn(a)


# ── E0.2（MQ-L48）：计数单位 = 用户轮，工具循环内的多次请求不重复计 ─────────

def test_tool_loop_requests_within_one_user_turn_count_once(monkeypatch):
    """20 次请求不跨用户轮（同一条 user 之后 20 个工具调用往返）⇒ 计数只动 1、陈旧不响。
    修前每请求 +1 ⇒ 一次 grep+读+改+测就把 STALE_TURNS=20 打满（MQ-L48）。"""
    a = _agency(monkeypatch, threshold=5)
    base = _history(a)                       # 用户轮 1
    for k in range(20):
        msgs = list(base)
        for j in range(k + 1):
            msgs.append({"role": "assistant", "content": None,
                         "tool_calls": [{"id": f"c{j}", "type": "function",
                                         "function": {"name": "bash", "arguments": "{}"}}]})
            msgs.append({"role": "tool", "tool_call_id": f"c{j}", "content": "ok"})
        out = a.insert_ledger_block(msgs, agent_id=_AGENT, session_id="s1")
        assert len(out) == len(msgs) + 1
        assert "turns on this ledger" not in str(out[-1].get("content") or "")
    assert a._turns_since_update[_SCOPE_LID] == 1, "同一用户轮内的 20 次工具请求只算 1 轮"
    # 跨用户轮照常累加：再过 4 个用户轮 ⇒ 5 ⇒ 响
    for _ in range(3):
        assert "turns on this ledger" not in _turn(a)
    assert "5 turns on this ledger" in _turn(a)


def test_old_request_count_semantics_would_fire_in_tool_loop(monkeypatch):
    """判别力对照：把口径改回 `len(messages)//2`（请求数）⇒ 同一用户轮内的工具循环
    在第 5 个请求就把陈旧标记打出来。这条红了说明上一条的"不响"不是靠运气。"""
    import bladex_core.ledger_runtime as lr

    a = _agency(monkeypatch, threshold=5)
    monkeypatch.setattr(lr, "user_turn_index", lambda msgs: max(0, len(msgs or []) // 2))
    base = _history(a)
    fired = False
    for k in range(8):
        msgs = list(base)
        for j in range(k + 1):
            msgs.append({"role": "assistant", "content": None,
                         "tool_calls": [{"id": f"c{j}", "type": "function",
                                         "function": {"name": "bash", "arguments": "{}"}}]})
            msgs.append({"role": "tool", "tool_call_id": f"c{j}", "content": "ok"})
        out = a.insert_ledger_block(msgs, agent_id=_AGENT, session_id="s1")
        fired = fired or ("turns on this ledger" in str(out[-1].get("content") or ""))
    assert fired, "请求数口径下工具循环必然触发陈旧——没触发说明计数不再依赖 user_turn_index"


def test_compaction_in_same_session_still_advances(monkeypatch):
    """同 session 压缩：user_turn 回落（10 → 2）也算新的一轮（用 `!=` 不用 `>`）。"""
    a = _agency(monkeypatch, threshold=5)
    for _ in range(3):
        _turn(a)                              # 用户轮 1..3
    assert a._turns_since_update[_SCOPE_LID] == 3
    short = [{"role": "user", "content": "压缩后的新窗口"}]
    a.insert_ledger_block(short, agent_id=_AGENT, session_id="s1")
    assert a._turns_since_update[_SCOPE_LID] == 4


def test_counts_are_per_ledger(monkeypatch):
    """两本账本各算各的——切过去不该继承别人的陈旧度。"""
    a = _agency(monkeypatch, threshold=5)
    other = new_ledger(ledger_id="ldg-other0000001", title="另一件事",
                       created_at="2026-08-26T10:00:00Z")
    a.pool[other.ledger_id] = other
    for _ in range(6):
        _turn(a)
    a.activation.restore({activation_scope(_AGENT, ""): other.ledger_id})
    assert "turns on this ledger" not in _turn(a)
    assert a._turns_since_update["ldg-other0000001"] == 1


# ── Goal 提炼与改动的**生产接线**（2026-08-26 拍板）────────────────────────

def test_goal_prefers_model_refinement(monkeypatch):
    """建账本时模型给了 goal ⇒ 用它；原话另存不注。"""
    import asyncio

    a = _agency(monkeypatch)
    a.pool.clear()
    a.activation.restore({activation_scope("newbie", ""): ""})
    user = "小黑，帮我看看那个网站的CSS架构合不合理，随便说说就行"
    asyncio.run(a._h_ledger_switch(
        {"ledger_id": "", "title": "CSS 架构评估",
         "goal": "Assess whether the site's CSS architecture is sound; "
                 "output a findings list with concrete issues."},
        allowed_exposure="local",
        context={"agent_id": "newbie", "session_id": "s9", "project_id": "",
                 "user_query": user, "turn_index": 0}))
    led = next(iter(a.pool.values()))
    assert led.goal.startswith("Assess whether")
    assert led.goal_source == "model"
    assert led.goal_verbatim == user
    # 只存不注
    assert user not in (a.ledger_injection_message("newbie") or {}).get("content", "")


def test_goal_falls_back_to_verbatim_when_model_gives_none(monkeypatch):
    """模型没提炼 ⇒ 回落原话（向后兼容；"没有 Goal"比"原话 Goal"更糟）。"""
    import asyncio

    a = _agency(monkeypatch)
    a.pool.clear()
    user = "帮我修一下路由"
    asyncio.run(a._h_ledger_switch(
        {"ledger_id": "", "title": "路由"},
        allowed_exposure="local",
        context={"agent_id": "newbie", "session_id": "s9", "project_id": "",
                 "user_query": user, "turn_index": 0}))
    led = next(iter(a.pool.values()))
    assert led.goal == user and led.goal_source == "user"


def test_model_cannot_revise_goal_without_a_real_quote(monkeypatch):
    """🔴 阴性对照：编造依据 ⇒ 拒绝，且 Goal 一个字没动。"""
    import asyncio

    a = _agency(monkeypatch)
    before = a.pool[_SCOPE_LID].goal
    out = asyncio.run(a._h_ledger_update(
        {"goal": "换个容易的目标", "goal_change_quote": "用户说算了不用做了"},
        allowed_exposure="local",
        context={"agent_id": _AGENT, "session_id": "s1", "project_id": "",
                 "user_query": "继续按原计划做"}))
    assert out.startswith("Error:")
    assert a.pool[_SCOPE_LID].goal == before
    assert a.pool[_SCOPE_LID].goal_revisions == 0


def test_model_revises_goal_when_user_asked(monkeypatch):
    """阳性对照：依据确实在本轮 user 消息里 ⇒ 放行 + 计数 + 留痕。"""
    import asyncio

    import structlog.testing

    a = _agency(monkeypatch)
    with structlog.testing.capture_logs() as cap:
        out = asyncio.run(a._h_ledger_update(
            {"goal": "范围收窄到只做移动端适配",
             "goal_change_quote": "只做移动端就行"},
            allowed_exposure="local",
            context={"agent_id": _AGENT, "session_id": "s1", "project_id": "",
                     "user_query": "改一下，只做移动端就行，桌面端先不管"}))
    assert not out.startswith("Error:"), out
    led = a.pool[_SCOPE_LID]
    assert led.goal_revisions == 1 and led.goal_source == "model_revised"
    ev = next((e for e in cap if e.get("event") == "agency_ledger_goal_revised"), None)
    assert ev is not None and ev.get("revision") == 1, cap


def test_bare_prefix_envelope_is_stripped_from_verbatim(monkeypatch):
    """live 病例：agent 的裸前缀包装不该进 Goal（jydesignhk 那 5 本）。"""
    from bladex_proxy.agency import _last_user_text

    raw = ("Fully describe and explain everything about this image, then answer "
           "the following question:\n\n这是家具电商网站的产品列表页。请从专业电商设计角度评价。")
    got = _last_user_text([{"role": "user", "content": raw}], for_goal=True)
    assert got.startswith("这是家具电商网站")
    assert "Fully describe" not in got


# ── 段语义必须写进工具 schema（MQ-L19，2026-08-27）────────────────────────

class TestSectionSemanticsAreStated:
    """🔴 只写段名 = 让模型猜，而猜是不稳定的。

    实证：同一个模型（deepseek-v4-flash / medium 档）在两本账本上**一对一错**——
    Hermes 那本 core 填的是上下文常量（对），Pi 那本填成了待办清单（错）。
    不是模型能力差异，是我们没说清楚。一个依赖模型猜的机制对弱模型必然失效
    （Jason 2026-08-27：不能强模型跑得通、弱模型就不管）。
    """

    @staticmethod
    def _params(tool_name: str) -> dict:
        from bladex_proxy.toolface import TOOL_SCHEMAS
        for t in TOOL_SCHEMAS:
            fn = t.get("function") or {}
            if fn.get("name") == tool_name:
                return (fn.get("parameters") or {}).get("properties") or {}
        raise AssertionError(f"{tool_name} 不在工具面里了")

    def test_switch_sections_say_what_they_are_for(self):
        props = self._params("bladex_ledger_switch")
        for sec in ("core", "open", "next"):
            d = (props[sec].get("description") or "")
            assert len(d) > 60, f"{sec} 的描述还是空洞的: {d!r}"

    def test_core_is_explicitly_not_a_todo_list(self):
        """Pi 那本的病灶：core 被当成开局 TODO 填。"""
        d = self._params("bladex_ledger_switch")["core"]["description"].lower()
        assert "not a to-do list" in d
        assert "next" in d, "要指明步骤该去哪一段，否则只是禁止、没有出路"

    def test_core_at_creation_is_scoped_to_what_the_user_said(self):
        """模板说 core 只放已验证的，而建账本时什么都还没验证。

        这个张力此前无人收口（模板要求验证过、参数却邀请它填），模型只好拿
        待办去凑。参数侧必须给出建本时的范围 + 没有内容时的出路。
        """
        d = self._params("bladex_ledger_switch")["core"]["description"].lower()
        assert "user" in d, "没说建本时 core 装的是用户给的约束"
        assert "leave empty" in d, "没给『没有约束时怎么办』的出路"
        assert "bladex_ledger_update" in d, "没说后续验证出来的常量该怎么补进去"

    def test_update_section_enum_carries_semantics(self):
        d = (self._params("bladex_ledger_update")["section"].get("description") or "")
        for sec in ("core", "verified", "open", "next"):
            assert sec in d, f"update 的 section 描述没提 {sec}"

    def test_goal_change_asks_to_recheck_core_and_next(self):
        """MQ-L19 加重版：Pi 那本改了 Goal，core 还写着"只分析不做任何动作"，

        与新 Goal 直接矛盾，每轮都往模型眼前送。
        """
        d = (self._params("bladex_ledger_update")["goal"].get("description") or "").lower()
        assert "core" in d and "next" in d
        assert "contradict" in d or "re-read" in d
