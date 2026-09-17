"""F-A1 / MQ-A35：账本块段级构成尺 + `Turn.ledger_anchor`（2026-09-07，批 F）。

**这是仪器卡，第一条不变式是"仪器不许改行为"**（批 F 高危 1）：块正文逐字等同
修前、`BLADEX_MODULE_LEDGER=0` 时锚为 None 且注入面零差异。

分母纪律（feedback_instrument_reference_frame）：
- `breakdown.chars` 的分母是**块自身**——`sum == len(block_content)` 是构造出来的
  恒等式（`_build_ledger_block` 按行累加），本文件 ① 直接钉它；
- `Turn.ledger_anchor` 的覆盖率分母是 `ledger_face=True` 的轮，那把尺在
  `probe_turn_field_coverage.py`，不在这里（本文件只证明字段可达且往返不丢）。

钉六件事：
① 有账本轮：`sections` 之和 + about + roster == `bladex_added_chars`（逐字，不带余项）；
② 无账本轮：`ledger_id=""` ∧ `template>0` ∧ 五个账本段全 0；
③ `BLADEX_MODULE_LEDGER=0` ⇒ 锚为 None **且注入正文逐字同修前**；
   判别力：把锚组装挪到 module 门之前 ⇒ 本条必红；
④ Hub 往返：Turn dump → load，锚不丢；
⑤ 旧 Turn dump（无该字段）照读（追加字段不破存量）；
⑥ `_split_ledger_md` 无损：切分再拼回逐字等于原文，且新增段自成一桶
   （闭集从定义读——`section_order` 才是真结构）。
"""

from __future__ import annotations

import json

import structlog.testing

from bladex_core.ledger import (
    ACTOR_MODEL, LedgerEntry, add_entry, add_section, new_ledger, render_ledger_md
)
from bladex_proxy.agency import LEDGER_BLOCK_BUCKETS, AgencyRuntime, _split_ledger_md
from bladex_proxy.models import Identity, LedgerAnchor, Turn
from bladex_proxy.server.orchestration import _added_text, _build_ledger_anchor

_SYS = {"role": "system", "content": "You are a test agent."}
_USER = {"role": "user", "content": "继续做批 F 的仪器棒"}
_LEDGER_SECTIONS = ("goal", "core", "verified", "open", "next")


def _on(monkeypatch) -> None:
    monkeypatch.setenv("BLADEX_MODULE_TOOLFACE", "1")
    monkeypatch.setenv("BLADEX_MODULE_LEDGER", "1")


def _seeded(ag: AgencyRuntime, agent: str = "hermes:default") -> AgencyRuntime:
    led = new_ledger(ledger_id="ldg-fa1", title="批 F 仪器棒",
                     goal="把段级构成尺立起来", goal_source="user",
                     created_at="2026-09-07T00:00:00+00:00")
    led = add_entry(led, "next", LedgerEntry(text="落 Turn 锚字段", source="model"),
                    actor=ACTOR_MODEL)
    led = add_entry(led, "verified", LedgerEntry(text="daemon 等 proxy 已修",
                                                 source="model", ref="test_flash_daemon"),
                    actor=ACTOR_MODEL)
    ag.pool[led.ledger_id] = led
    ag.activation.restore({ag.scope_of(agent, ""): led.ledger_id})
    return ag


# ── ① 恒等式：桶和 == 块正文；桶和 + about + roster == bladex_added_chars ──


def test_breakdown_sums_to_block_and_added_chars(monkeypatch):
    _on(monkeypatch)
    ag = _seeded(AgencyRuntime())
    ag.about = "BladeX keeps your task ledger."
    ag.agents_roster = "- hermes: the agent"
    msgs = [_SYS, _USER]

    out = ag.insert_ledger_block(msgs, "hermes:default", tier="medium",
                                 with_instruction=True)
    out = ag.insert_system_notes(out, toolface_injected=True)
    bd = ag.last_ledger_breakdown
    assert bd is not None and bd.ledger_id == "ldg-fa1"
    assert bd.rev == ag.pool["ldg-fa1"].rev

    block = next(m["content"] for m in out
                 if isinstance(m.get("content"), str) and "<bladex-ledger>" in m["content"])
    assert bd.total() == len(block), "段级构成必须逐字覆盖块正文（无余项、无重复计）"

    # `bladex_added_chars` 的同一把差集（口径同 assembly.estimate_context_chars）
    from bladex_proxy.assembly import estimate_context_chars
    added = estimate_context_chars(out) - estimate_context_chars(msgs)
    assert bd.total() + ag.last_about_chars + ag.last_roster_chars == added, \
        "sections + about + roster 必须逐字等于 BladeX 加进正文的全部字符"

    # 五个账本段都有内容（这本账本 goal/verified/next 非空；core/open 是 _(empty)_ 行）
    for sec in _LEDGER_SECTIONS:
        assert bd.chars[sec] > 0, f"{sec} 段字符数不该为 0（渲染恒有标题行）"
    assert bd.chars["instruction"] > 0, "有账本 + 有工具面 ⇒ 首步指令必在"
    assert bd.chars["template"] == 0, "有账本轮不注模板"


def test_block_body_is_byte_identical_to_pre_change(monkeypatch):
    """仪器不许改行为：块正文 = 旧调用面（`ledger_injection_message`）的返回值。"""
    _on(monkeypatch)
    ag = _seeded(AgencyRuntime())
    old = ag.ledger_injection_message("hermes:default", with_instruction=True,
                                      user_text="继续做批 F 的仪器棒")
    new, bd = ag._build_ledger_block("hermes:default", with_instruction=True,
                                     user_text="继续做批 F 的仪器棒")
    assert old == new
    assert bd.total() == len(new["content"])


def test_sections_logged_as_json(monkeypatch):
    """live 判据的载体：`agency_ledger_block_injected` 带 `rev=` 与 `sections=`。"""
    _on(monkeypatch)
    ag = _seeded(AgencyRuntime())
    with structlog.testing.capture_logs() as cap:
        ag.insert_ledger_block([_SYS, _USER], "hermes:default", tier="medium",
                               with_instruction=True)
    rec = next(e for e in cap if e["event"] == "agency_ledger_block_injected")
    assert rec["rev"] == ag.pool["ldg-fa1"].rev
    sections = json.loads(rec["sections"])
    assert set(LEDGER_BLOCK_BUCKETS) <= set(sections)
    assert sum(sections.values()) == rec["chars"]


# ── ② 无账本轮：template 非零，五个账本段全 0 ──────────────────────────────


def test_no_ledger_turn_has_template_only(monkeypatch):
    _on(monkeypatch)
    ag = AgencyRuntime()          # 空池、无激活账本
    ag.template = "## Goal\n\n(write it)\n"
    ag.insert_ledger_block([_SYS, _USER], "hermes:default", tier="medium",
                           with_instruction=True)
    bd = ag.last_ledger_breakdown
    assert bd is not None
    assert bd.ledger_id == "" and bd.rev == -1, "无账本 ⇒ 缺数报缺数，不造一个 rev"
    assert bd.chars["template"] > 0
    for sec in _LEDGER_SECTIONS:
        assert bd.chars[sec] == 0, f"无账本轮 {sec} 段必须是 0"
    assert bd.chars["stale"] == 0 and bd.chars["children"] == 0


def test_breakdown_is_cleared_on_gated_turn(monkeypatch):
    """aux 轮不注块 ⇒ 读数**清空**而不是沿用上一轮（参照系脱钩是最贵的一类）。"""
    _on(monkeypatch)
    ag = _seeded(AgencyRuntime())
    ag.insert_ledger_block([_SYS, _USER], "hermes:default", tier="medium",
                           with_instruction=True)
    assert ag.last_ledger_breakdown is not None
    ag.insert_ledger_block([_SYS, _USER], "hermes:default", tier="medium",
                           with_instruction=True, auxiliary=True)
    assert ag.last_ledger_breakdown is None, "aux 轮必须清空，否则读数串到错的轮上"


# ── ③ 模块门：关掉 ⇒ 锚 None 且注入面零差异 ────────────────────────────────


class _Prep:
    """`_build_ledger_anchor` 只读 prep 的两份正文——够用的最小替身。"""

    def __init__(self, messages, injected):
        self.messages = messages
        self.injected_messages = injected


def test_module_off_means_no_anchor_and_no_injection(monkeypatch):
    monkeypatch.setenv("BLADEX_MODULE_TOOLFACE", "1")
    monkeypatch.setenv("BLADEX_MODULE_LEDGER", "0")
    ag = _seeded(AgencyRuntime())
    msgs = [_SYS, _USER]
    out = ag.insert_ledger_block(msgs, "hermes:default", tier="medium",
                                 with_instruction=True)
    assert out == msgs, "模块关 ⇒ 注入正文逐字同修前"
    assert ag.last_ledger_breakdown is None
    # 🔴 判别力：锚组装若挪到 module 门之前（例如直接读 pool 而不读 breakdown），
    # 下面这条会拿到一个非 None 的锚 ⇒ 本条必红。
    assert _build_ledger_anchor(ag, _Prep(msgs, out)) is None


def test_anchor_carries_sections_and_added_hash(monkeypatch):
    _on(monkeypatch)
    ag = _seeded(AgencyRuntime())
    msgs = [_SYS, _USER]
    out = ag.insert_ledger_block(msgs, "hermes:default", tier="medium",
                                 with_instruction=True)
    anchor = _build_ledger_anchor(ag, _Prep(msgs, out))
    assert isinstance(anchor, LedgerAnchor)
    assert anchor.ledger_id == "ldg-fa1"
    assert sum(anchor.sections.values()) == len(out[-1]["content"])
    assert len(anchor.added_hash) == 16
    # 同样的差集 ⇒ 同样的 hash（对账用途成立）；正文变一个字 ⇒ hash 变
    again = _build_ledger_anchor(ag, _Prep(msgs, out))
    assert again.added_hash == anchor.added_hash
    tweaked = [*out[:-1], {**out[-1], "content": out[-1]["content"] + "x"}]
    assert _build_ledger_anchor(ag, _Prep(msgs, tweaked)).added_hash != anchor.added_hash


def test_added_text_is_a_multiset_difference():
    """重复正文只抵消一次——集合差集会把第二份算成"没加"（静默少算）。"""
    orig = [{"role": "user", "content": "同一段话"}]
    inj = [{"role": "user", "content": "同一段话"},
           {"role": "system", "content": "同一段话"}]
    assert _added_text(orig, inj) == "同一段话"
    assert _added_text(orig, orig) == ""


# ── ④⑤ Hub 往返 + 旧 Turn 兼容 ─────────────────────────────────────────────


def _turn(**kw) -> Turn:
    return Turn(identity=Identity(user_id="u1", agent_id="hermes:default",
                                  session_id="s1"), **kw)


def test_anchor_survives_dump_and_load():
    anchor = LedgerAnchor(ledger_id="ldg-fa1", rev=7,
                          sections={"header": 10, "goal": 20},
                          about_chars=30, roster_chars=5, added_hash="abcdef0123456789")
    back = Turn.model_validate(_turn(ledger_anchor=anchor).model_dump())
    assert back.ledger_anchor == anchor


def test_old_turn_without_field_still_loads():
    """追加字段不破存量：旧 dump 里没有 `ledger_anchor` ⇒ None，不是报错也不是 0 值。"""
    d = _turn().model_dump()
    d.pop("ledger_anchor")
    assert Turn.model_validate(d).ledger_anchor is None


# ── ⑥ `_split_ledger_md` 无损 + 新增段自成一桶 ───────────────────────────


def test_split_ledger_md_is_lossless_and_open_ended():
    led = new_ledger(ledger_id="ldg-x", title="t", goal="g")
    led = add_section(led, "lessons")          # 0.2.0 会加的那一段
    led = add_entry(led, "lessons", LedgerEntry(text="教训一条"), actor=ACTOR_MODEL)
    md = render_ledger_md(led).strip()
    parts = _split_ledger_md(md, led.section_order)
    assert "\n".join(c for _, c in parts) == md, "切分必须逐字可拼回（原则 13 无损）"
    keys = [k for k, _ in parts]
    assert keys[0] == "header"
    assert "lessons" in keys, "未登记的段要自成一桶，不许并进 header 静默消失"
