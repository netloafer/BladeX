"""G16.0 · 用户面/工具面判别器的**观测面**（MQ-A56，父条目 MQ-L48；2026-09-09）。

本批**只观测不判决**：`with_instruction` 的取值一字不动，只把判别器结果打进
`agency_ledger_block_injected` 的 `face` / `face_last_role` / `uti` 三个字段，
让 live 流量在四个 agent 上对账，再由 G16.2 接门控。

钉五件事：
① 同一用户轮内的工具往返判 `face=tool`，新用户消息判 `face=user`；
② 键是 **(激活作用域, session_id)** 而不是 `ledger_id` —— 无账本分支也要判得出来；
③ 🔴 **反面证据**：`face_last_role` 证明"末条 role == 'user'"这个更简的无状态判据
   **不可用** —— claude-code 在尾部追加 `role=system` 的 system-reminder，
   真实用户轮的末条也是 `system`（E 档 88 个注块轮里该判据 0 次触发）；
④ 两个 face 值都可达（读数为 0 先证明桶可达）；
⑤ **零行为改变**：块正文逐字不受新代码影响。
"""
from __future__ import annotations

import structlog.testing
from bladex_core.ledger import new_ledger
from bladex_proxy.agency import AgencyRuntime

_SYS = {"role": "system", "content": "You are a test agent."}
_USER = {"role": "user", "content": "给账本加 rev 时间线端点"}
_TOOL = {"role": "tool", "tool_call_id": "t1", "content": "ok"}
#: 🔴 live 形态：claude-code 每轮在尾部追加的 system-reminder（E 档 88 轮里 20 次末条是它）
_SYSREM = {"role": "system", "content": "<system-reminder>…</system-reminder>"}


def _on(monkeypatch):
    monkeypatch.setenv("BLADEX_MODULE_TOOLFACE", "1")
    monkeypatch.setenv("BLADEX_MODULE_LEDGER", "1")


def _inject(ag, msgs, *, agent="claude-code", session="s-1"):
    with structlog.testing.capture_logs() as cap:
        out = ag.insert_ledger_block(msgs, agent, tier="medium",
                                     with_instruction=True, session_id=session)
    row = next(e for e in cap if e.get("event") == "agency_ledger_block_injected")
    return out, row


def _seed(ag: AgencyRuntime) -> None:
    led = new_ledger(title="t", goal="g", ledger_id="ldg-face000001",
                     created_at="2026-09-09T10:00:00+00:00")
    ag.pool[led.ledger_id] = led
    ag.activation.restore({ag.scope_of("claude-code"): led.ledger_id,
                           ag.scope_of("codex"): led.ledger_id})


# ── ① 同一用户轮内的工具往返 = 工具面 ──────────────────────────────────────

def test_first_request_is_user_face(monkeypatch):
    _on(monkeypatch)
    ag = AgencyRuntime()
    _seed(ag)
    _, row = _inject(ag, [_SYS, _USER])
    assert row["face"] == "user"
    assert row["uti"] == 1


def test_tool_roundtrips_in_same_user_turn_are_tool_face(monkeypatch):
    _on(monkeypatch)
    ag = AgencyRuntime()
    _seed(ag)
    _inject(ag, [_SYS, _USER])
    faces = []
    msgs = [_SYS, _USER]
    for _ in range(4):
        msgs = [*msgs, {"role": "assistant", "content": ""}, _TOOL]
        faces.append(_inject(ag, msgs)[1]["face"])
    assert faces == ["tool"] * 4, "同一用户轮内的工具往返必须全判工具面"


def test_new_user_message_flips_back_to_user_face(monkeypatch):
    _on(monkeypatch)
    ag = AgencyRuntime()
    _seed(ag)
    msgs = [_SYS, _USER]
    _inject(ag, msgs)
    msgs = [*msgs, {"role": "assistant", "content": ""}, _TOOL]
    assert _inject(ag, msgs)[1]["face"] == "tool"
    msgs = [*msgs, {"role": "user", "content": "再加一个 CLI 子命令"}]
    row = _inject(ag, msgs)[1]
    assert row["face"] == "user" and row["uti"] == 2


# ── ② 键是作用域+会话，不是 ledger_id（无账本分支也要判得出来）──────────────

def test_face_is_decided_without_any_active_ledger(monkeypatch):
    """无账本 ⇒ 走 `_FIRST_STEP_NO_LEDGER` 分支，仍必须给出 face。

    这正是不能照抄 V-L4 那处 `_last_seen_user_turn[ledger_id]` 的原因：那时没有 id。
    """
    _on(monkeypatch)
    ag = AgencyRuntime()          # 不 seed：池空、无激活账本
    _, row = _inject(ag, [_SYS, _USER])
    assert row["ledger"] == "" and row["face"] == "user"
    msgs = [_SYS, _USER, {"role": "assistant", "content": ""}, _TOOL]
    assert _inject(ag, msgs)[1]["face"] == "tool"


def test_different_sessions_do_not_share_face_state(monkeypatch):
    """两个会话各自的第一轮都该是 user 面——键里带 session_id 才做得到。"""
    _on(monkeypatch)
    ag = AgencyRuntime()
    _seed(ag)
    assert _inject(ag, [_SYS, _USER], session="s-a")[1]["face"] == "user"
    assert _inject(ag, [_SYS, _USER], session="s-b")[1]["face"] == "user"
    # s-a 的工具往返（消息变长）不受 s-b 影响
    assert _inject(ag, [_SYS, _USER, _TOOL], session="s-a")[1]["face"] == "tool"


def test_byte_identical_retry_keeps_the_previous_face(monkeypatch):
    """🔴 重试沿用上次判定，不翻面。

    工具往返 = uti 不变**且消息变长**；重试 = 形状**完全相同**。只按 uti 判会把
    重试也当工具往返 ⇒ 真实用户轮被 185 秒超时打断后重发（MQ-A57 实测单会话 11 次），
    重试就丢掉 (a) 比对/切换半句。本用例是这条规则的阴性对照：
    同样是"uti 没变"，变长 ⇒ tool，不变 ⇒ 沿用 user。
    """
    _on(monkeypatch)
    ag = AgencyRuntime()
    _seed(ag)
    msgs = [_SYS, _USER]
    assert _inject(ag, msgs, session="s-r")[1]["face"] == "user"
    assert _inject(ag, msgs, session="s-r")[1]["face"] == "user", "逐字重试 ⇒ 仍是用户面"
    assert _inject(ag, [*msgs, _TOOL], session="s-r")[1]["face"] == "tool", "变长 ⇒ 工具面"
    # 工具往返之后再重试那一条，同样沿用
    assert _inject(ag, [*msgs, _TOOL], session="s-r")[1]["face"] == "tool"


# ── ③ 🔴 反面证据：为什么"末条 role == 'user'"不能用 ────────────────────────

def test_last_message_role_is_not_a_usable_discriminator(monkeypatch):
    """live 形态回放：真实用户轮的**末条也是 `role=system`**（CC 的 system-reminder）。

    E 档 88 个注块轮里"末条 role == 'user'"**0 次**为真 —— 若按它接门控，
    首步指令将**一次都不注**。本用例把这个事实钉住：
    `face` 必须判 `user`，而 `face_last_role` 必须如实记下 `system`。
    """
    _on(monkeypatch)
    ag = AgencyRuntime()
    _seed(ag)
    _, row = _inject(ag, [_SYS, _USER, _SYSREM])
    assert row["face"] == "user", "判别器不得依赖末条角色"
    assert row["face_last_role"] == "system", "反面证据要如实记录，不许粉饰"


def test_face_last_role_records_the_real_tail(monkeypatch):
    _on(monkeypatch)
    ag = AgencyRuntime()
    _seed(ag)
    seen = {_inject(ag, m)[1]["face_last_role"]
            for m in ([_SYS, _USER], [_SYS, _USER, _TOOL], [_SYS, _USER, _SYSREM])}
    assert seen == {"user", "tool", "system"}, "三种真实尾部形态都要能记下来"


# ── ④ 两个 face 值都可达（分母纪律）────────────────────────────────────────

def test_both_face_values_are_reachable(monkeypatch):
    _on(monkeypatch)
    ag = AgencyRuntime()
    _seed(ag)
    faces = {_inject(ag, [_SYS, _USER])[1]["face"],
             _inject(ag, [_SYS, _USER, _TOOL])[1]["face"]}
    assert faces == {"user", "tool"}, "任一桶不可达 ⇒ 后面的 live 判据没有分母"


# ── ⑤ 零行为改变 ──────────────────────────────────────────────────────────

def test_flag_off_makes_both_faces_identical(monkeypatch):
    """🔴 回滚通道：`BLADEX_LEDGER_FACE_SPLIT=0` ⇒ 两面块正文**逐字相同**。

    G16.0 时这条断言的是"本批零行为改变"；G16.2 落地后行为**故意**改了，
    于是它改钉**开关关掉后是否真的回到修前**——codex / Pi 的 face 分布仍是
    NO-DATA，回滚通道必须是活的。
    """
    _on(monkeypatch)
    monkeypatch.setenv("BLADEX_LEDGER_FACE_SPLIT", "0")
    ag = AgencyRuntime()
    _seed(ag)
    out_u, row_u = _inject(ag, [_SYS, _USER])
    out_t, row_t = _inject(ag, [_SYS, _USER, _TOOL])
    assert row_u["face"] == "user" and row_t["face"] == "tool", "观测面不受开关影响"
    assert out_u[-1]["content"] == out_t[-1]["content"]
    assert row_t["instruction_kind"] == "switch", "关掉之后工具轮也注完整原文"


# ── ⑦ G16.1/G16.2 · 拆指令 + 接门控 ───────────────────────────────────────

def _block(ag, msgs, **kw):
    return _inject(ag, msgs, **kw)[0][-1]["content"]


def test_user_face_block_is_byte_identical_to_before(monkeypatch):
    """🔴 用户面**一字不改**：开关开/关渲染出的用户面块必须逐字相同。

    这是本卡的零回归红线 —— 我们只新写了工具面的文本，用户面继续用
    `_FIRST_STEP_WITH_LEDGER` 原常量。这条红了说明有人顺手改了用户面措辞，
    那会让 G16.3 的判据归因不了（"两边同时改了"）。
    """
    _on(monkeypatch)
    ag1 = AgencyRuntime()
    _seed(ag1)
    monkeypatch.setenv("BLADEX_LEDGER_FACE_SPLIT", "1")
    on = _block(ag1, [_SYS, _USER], session="s-on")
    ag2 = AgencyRuntime()
    _seed(ag2)
    monkeypatch.setenv("BLADEX_LEDGER_FACE_SPLIT", "0")
    off = _block(ag2, [_SYS, _USER], session="s-off")
    assert on == off


def test_tool_face_drops_switch_half_and_keeps_record_half(monkeypatch):
    _on(monkeypatch)
    monkeypatch.setenv("BLADEX_LEDGER_FACE_SPLIT", "1")
    ag = AgencyRuntime()
    _seed(ag)
    _inject(ag, [_SYS, _USER])                       # 先走掉用户轮
    tool = _block(ag, [_SYS, _USER, _TOOL])
    assert "bladex_ledger_switch" not in tool, "(a) 切换半句必须没了"
    assert "FIRST STEP" not in tool
    assert "bladex_ledger_update" in tool, "(b) 记账半句必须还在"
    assert "do NOT switch or create ledgers here" in tool


def test_tool_face_keeps_the_ledger_body(monkeypatch):
    """工具面拿掉的是催促，不是账本正文 —— 五段仍要看得见。"""
    _on(monkeypatch)
    monkeypatch.setenv("BLADEX_LEDGER_FACE_SPLIT", "1")
    ag = AgencyRuntime()
    _seed(ag)
    _inject(ag, [_SYS, _USER])
    _, row = _inject(ag, [_SYS, _USER, _TOOL])
    body = {k: row["sections"] for k in ()} or row["sections"]
    import json as _json
    sec = _json.loads(body) if isinstance(body, str) else body
    assert sec.get("goal", 0) > 0 and sec.get("header", 0) > 0
    assert sec.get("instruction", 0) > 0, "(b) 仍算 instruction 段"


def test_tool_face_drops_candidate_and_recent_lists(monkeypatch):
    """候选/LEDGERS 列表随 (a) 走：工具面既不该切、也没有新用户话可匹配。"""
    _on(monkeypatch)
    monkeypatch.setenv("BLADEX_LEDGER_FACE_SPLIT", "1")
    ag = AgencyRuntime()
    _seed(ag)
    ag.pool["ldg-other01"] = new_ledger(title="别的任务", goal="g2",
                                        ledger_id="ldg-other01",
                                        created_at="2026-09-08T10:00:00+00:00")
    with structlog.testing.capture_logs() as cap_u:
        ag.insert_ledger_block([_SYS, _USER], "claude-code", tier="medium",
                               with_instruction=True, session_id="s-c")
    with structlog.testing.capture_logs() as cap_t:
        ag.insert_ledger_block([_SYS, _USER, _TOOL], "claude-code", tier="medium",
                               with_instruction=True, session_id="s-c")
    assert any(e.get("event") == "agency_ledger_candidates" for e in cap_u)
    assert not any(e.get("event") == "agency_ledger_candidates" for e in cap_t), \
        "工具面不该再算候选（算了就等于还在邀请切换）"


def test_instruction_kind_three_values_are_reachable(monkeypatch):
    """判据 4「工具轮 switch = 0」的分母：三个取值都要可达。"""
    _on(monkeypatch)
    monkeypatch.setenv("BLADEX_LEDGER_FACE_SPLIT", "1")
    ag = AgencyRuntime()
    _seed(ag)
    kinds = {_inject(ag, [_SYS, _USER])[1]["instruction_kind"],
             _inject(ag, [_SYS, _USER, _TOOL])[1]["instruction_kind"]}
    with structlog.testing.capture_logs() as cap:
        ag.insert_ledger_block([_SYS, _USER], "claude-code", tier="medium",
                               with_instruction=False, session_id="s-ro")
    row = next(e for e in cap if e.get("event") == "agency_ledger_block_injected")
    kinds.add(row["instruction_kind"])
    assert kinds == {"switch", "record", "none"}


# ── ⑥ candidates 事件带 agent（跨 agent 复核被它卡住过）────────────────────

def test_candidates_event_carries_agent(monkeypatch):
    _on(monkeypatch)
    ag = AgencyRuntime()
    _seed(ag)
    with structlog.testing.capture_logs() as cap:
        ag.insert_ledger_block([_SYS, _USER], "codex", tier="medium",
                               with_instruction=True, session_id="s-9")
    rows = [e for e in cap if e.get("event") == "agency_ledger_candidates"]
    assert rows, "候选段应打点"
    assert rows[0]["agent"] == "codex"
