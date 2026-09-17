"""并发下账本读数不许串台（F1.3 / MQ-A49，2026-09-08）。

## 形态

`agency.last_ledger_breakdown` / `last_ledger_face` 是**进程级"最近一次"**。
`_apply_agency_surfaces` 在同步链里读它们（无 await 插入，安全）；但
`_enqueue_turn` 在**上游调用之后**才跑——期间另一请求的 `augment_tools` /
`insert_ledger_block` 已经把它们覆写了。

    A 注块（units=A）→ B 注块（units=B）→ A 上游返回 → A 入库读到 B 的 units

单用户串行流量**不触发**，所以 09-07 的 live 读数看不出来；批 G 剧本③正是两 agent
并行 —— 尺子会在最需要它的那个场景失真，而且两条日志各自都自洽、肉眼看不出来。

⇒ 两个字段都改由 `_apply_agency_surfaces` 停进 `request.state`（请求作用域）。
`ledger_face` 比 `next_referenced` 更贵：它**是入库的**，错的值会让全量重建的
历史锚定选错轮。

## 为什么 live 判据要等批 G

今天没有并行流量 ⇒ 串了也读不出来。本卡靠单测钉住，live 判据（同一轮的
`next_referenced.ledger` 与 `block_injected.ledger` 逐条一致）在批 G 剧本③兑现。
"""

from __future__ import annotations

import types

import structlog.testing
from bladex_core.ledger import ACTOR_MODEL, LedgerEntry, add_entry, new_ledger
from bladex_core.ledger_runtime import candidate_units
from bladex_proxy.models import LedgerAnchor
from bladex_proxy.server.orchestration import _ledger_next_referenced

_A = "ldg-aaa"
_B = "ldg-bbb"
_A_TEXT = "把 probe_ledger_trajectory 的分桶接上 self_conflict"
_B_TEXT = "核验 pooltool 仿真的落点与 map50 基准"


def _led(lid: str, text: str):
    led = new_ledger(ledger_id=lid, title=lid, goal="g")
    return add_entry(led, "next", LedgerEntry(text=text), actor=ACTOR_MODEL)


class _Agency:
    """进程级 agency：`last_ledger_breakdown` 每次 `insert_block` 被覆写一次。"""

    def __init__(self):
        self.pool = {_A: _led(_A, _A_TEXT), _B: _led(_B, _B_TEXT)}
        self.last_ledger_breakdown = None
        self.last_ledger_face = None

    def insert_block(self, request, lid: str, *, face: bool):
        """生产链的两步压缩版：`insert_ledger_block` 覆写进程级字段，
        随后 `_apply_agency_surfaces` 把它们停进本请求的 state。"""
        from bladex_proxy.agency import LedgerBlockBreakdown
        text = " ".join(e.text for e in self.pool[lid].entries("next"))
        units = frozenset(candidate_units(text))
        self.last_ledger_breakdown = LedgerBlockBreakdown(
            ledger_id=lid, rev=1, pending_units=units, pending_rare=units)
        self.last_ledger_face = face
        _stash(request, self)


def _stash(request, agency) -> None:
    """`_apply_agency_surfaces` 尾部那三行的逐字等价物。

    🔴 判别力对照就打在这里：把它换成"入库时再读 agency"即为修前的实现。
    """
    bd = agency.last_ledger_breakdown
    request.state.bladex_pending_units = (
        None if bd is None else frozenset(bd.pending_units))
    request.state.bladex_pending_rare = (
        frozenset() if bd is None else frozenset(bd.pending_rare))
    request.state.bladex_ledger_face = agency.last_ledger_face


def _request(agency):
    return types.SimpleNamespace(
        state=types.SimpleNamespace(),
        app=types.SimpleNamespace(state=types.SimpleNamespace(agency=agency)))


def _enqueue_read(request, lid: str, first_action_text: str):
    """`_enqueue_turn` 里那两行的等价物：读 `request.state` 并跑仪器。"""
    anchor = LedgerAnchor(ledger_id=lid, rev=1, sections={"next": 40})
    with structlog.testing.capture_logs() as cap:
        _ledger_next_referenced(request, anchor, first_action_text, [])
    recs = [e for e in cap if e["event"] == "agency_ledger_next_referenced"]
    assert len(recs) == 1
    face = getattr(request.state, "bladex_ledger_face", None)
    return recs[0], face


# ── ① 交错：A 注块 → B 注块 → A 入库 ───────────────────────────────────────


def test_interleaved_requests_keep_their_own_units():
    """A 入库时用的必须是 **A** 的单元集，哪怕 B 已经覆写了进程级字段。"""
    ag = _Agency()
    req_a, req_b = _request(ag), _request(ag)
    ag.insert_block(req_a, _A, face=True)
    ag.insert_block(req_b, _B, face=False)      # B 覆写 agency.last_*

    rec_a, face_a = _enqueue_read(req_a, _A, "我先把 probe_ledger_trajectory 接上")
    assert rec_a["ledger"] == _A
    assert rec_a["hits"] >= 1, "A 的首动作引用了 A 的 Next —— 必须命中"
    assert "probe_ledger_trajectory" in rec_a["sample"]
    assert "pooltool" not in rec_a["sample"] and "map50" not in rec_a["sample"], \
        "A 的读数里出现 B 的词 = 串台"
    assert face_a is True, "A 的 ledger_face 被 B 的覆写了"

    rec_b, face_b = _enqueue_read(req_b, _B, "核验 pooltool 的落点")
    assert rec_b["hits"] >= 1 and "pooltool" in rec_b["sample"]
    assert face_b is False


def test_b_is_not_polluted_by_a_either():
    """反向：先 A 后 B，B 入库同样只看自己的（对称性不是自动的，要钉）。"""
    ag = _Agency()
    req_a, req_b = _request(ag), _request(ag)
    ag.insert_block(req_a, _A, face=False)
    ag.insert_block(req_b, _B, face=True)
    rec_b, face_b = _enqueue_read(req_b, _B, "我先把 probe_ledger_trajectory 接上")
    assert rec_b["hits"] == 0, "B 的单元集里没有 A 的词 —— 必须零命中"
    assert face_b is True


def test_no_surfaces_step_logs_nothing():
    """没走过 `_apply_agency_surfaces`（state 上没有这个属性）⇒ 不造读数。

    缺数报缺数：`getattr` 兜底是 `None` 而不是空集合，空集合会让"没量"
    长得像"量到 0"。
    """
    ag = _Agency()
    req = _request(ag)
    anchor = LedgerAnchor(ledger_id=_A, rev=1, sections={"next": 40})
    with structlog.testing.capture_logs() as cap:
        _ledger_next_referenced(req, anchor, "probe_ledger_trajectory", [])
    assert [e for e in cap if e["event"] == "agency_ledger_next_referenced"] == []


# ── ② 判别力：改回读 agency ⇒ ① 必红 ──────────────────────────────────────


def test_discriminative_reading_agency_at_enqueue_time_crosses_talk():
    """修前的实现（入库时读 `agency.last_*`）在同一个交错序列上必然串台。

    这一条不是"测试旧代码"——它证明**交错序列本身有区分度**：
    没有它，① 全绿也可能只是因为这个用例根本触发不到串台
    （feedback_instrument_reference_frame：先证明那个桶可达）。
    """
    ag = _Agency()
    req_a, req_b = _request(ag), _request(ag)
    ag.insert_block(req_a, _A, face=True)
    ag.insert_block(req_b, _B, face=False)

    # 修前：入库时才从 agency 取 —— 拿到的是 B 的
    stale_units = set(ag.last_ledger_breakdown.pending_units)
    stale_face = ag.last_ledger_face
    seen = candidate_units("我先把 probe_ledger_trajectory 接上")
    assert stale_units & seen == set(), \
        "旧实现下 A 的命中会掉到 0（拿 B 的单元集去比 A 的首动作）"
    assert stale_face is False, "旧实现下 A 的 ledger_face 会被写成 B 的 False"
    # 新实现下同一序列是对的（① 的断言，这里并排放着好读）
    assert req_a.state.bladex_ledger_face is True
    assert set(req_a.state.bladex_pending_units) & seen


# ── ③ 三态守卫的**行为版**（MQ-A53 / 批 G G0.1，2026-09-09）─────────────────
#
# 🔴 为什么补这一节：复核用变异测试把 `_apply_agency_surfaces` 里
# `frozenset(...) if _bd is not None else **None**` 的 `None` 改成 `frozenset()`，
# 上面 ①② 与另外三个文件**全绿** —— 因为它们全都走 `_stash`（本文件的等价物），
# 碰不到生产那一行。判别力对照证明的是"我写的那条路会红"，
# 证明不了"我没写的那条路不会静默"。
#
# ⇒ 本节直接跑**生产的** `_apply_agency_surfaces`，断言的是它停进 `request.state`
# 的值本身与下游后果，不是源码字面量（源码断言换个等价写法就误红，
# 而行为层面照样没人守）。


def _round_prep(messages: list[dict]):
    """`_apply_agency_surfaces` 只用到 RoundPrep 的这五个槽（其余不碰）。"""
    from bladex_proxy.server.orchestration import RoundPrep
    prep = RoundPrep()
    prep.identity = types.SimpleNamespace(
        agent_id="claude-code", session_id="s-1", project_id="p-1",
        aux_source="", subagent=False)
    prep.messages = list(messages)
    prep.injected_messages = list(messages)
    prep.route = types.SimpleNamespace(tier="strong")
    return prep


class _NoLedgerAgency:
    """走过了注入面，但这一轮**没有账本块**（`last_ledger_breakdown is None`）。

    🔴 刻意**不定义** `last_ledger_face`：生产里 `augment_tools` 没走到那一步时
    就是这个形态，而 `getattr(agency, "last_ledger_face", None)` 的兜底值
    正是本节要钉的三态语义（缺数 ≠ False）。
    """

    pool: dict = {}
    last_ledger_breakdown = None

    def augment_tools(self, tools_in, **_kw):
        return list(tools_in or []), False

    def insert_ledger_block(self, messages, _agent_id, **_kw):
        return messages

    def insert_system_notes(self, messages, **_kw):
        return messages


def _run_surfaces(agency):
    """跑一遍**生产的** `_apply_agency_surfaces`，返回它写过的那个 request。"""
    from bladex_proxy.server.orchestration import _apply_agency_surfaces
    request = _request(agency)
    prep = _round_prep([{"role": "user", "content": "接着做，别复述"}])
    _apply_agency_surfaces(
        request, prep,
        tools_in=[{"type": "function", "function": {"name": "read_file"}}],
        stream=False)
    return request


def test_no_breakdown_stashes_none_not_an_empty_set():
    """没有 breakdown ⇒ `bladex_pending_units` 是 **`None`**（缺数），不是空集合。

    判别力：把生产那一行的 `else None` 改成 `else frozenset()` ⇒ 本条必红。
    """
    state = _run_surfaces(_NoLedgerAgency()).state
    assert state.bladex_pending_units is None, \
        "缺数被夹成了空集合 —— '没量'从此长得像'量到 0'"
    # `pending_rare` 反过来：它没有三态语义（只喂 `&`），空集合是对的。两者不同族，
    # 一起钉住免得下次"顺手统一"成同一个兜底。
    assert state.bladex_pending_rare == frozenset()


def test_no_breakdown_logs_no_next_referenced_row_at_all():
    """A53 的**后果**面：没走出 breakdown 的轮次，仪器一条日志都不许打。

    `_ledger_next_referenced` 靠 `units is None ⇒ return` 来"不给没有分母的轮次
    造读数"。改成空集合后它会继续往下打一条 `hits=0 next_units=0 usable=False`
    ——**给一个不该有分母的轮次造出一行读数**（A43 修掉的零分母形态的另一条产生
    路径）。`usable=False` 目前挡住它进基线，所以危害有限；守卫缺口是真的。

    判别力：同上一条（`else frozenset()` ⇒ 这里冒出一行）。
    """
    req = _run_surfaces(_NoLedgerAgency())
    anchor = LedgerAnchor(ledger_id=_A, rev=1, sections={"next": 40})
    with structlog.testing.capture_logs() as cap:
        _ledger_next_referenced(req, anchor, "probe_ledger_trajectory", [])
    assert [e for e in cap if e["event"] == "agency_ledger_next_referenced"] == []


def test_missing_ledger_face_stays_none_and_false_stays_false():
    """`ledger_face` 三态的**行为版**（此前只有源码字面量断言守着）。

    缺数 ⇒ `None`（`Turn.ledger_face` 因此是 `None` 而不是 `False`：这个字段
    **是入库的**，夹成 False 会让全量重建的历史锚定一次性失效）。
    第二半是正向对照——`False` 必须原样传下去，否则"永远 None"也能骗过上半条。

    判别力：把生产那一行改成 `bool(getattr(agency, "last_ledger_face", None))`
    ⇒ 上半条必红。
    """
    from bladex_proxy.models import Turn

    missing = _run_surfaces(_NoLedgerAgency()).state
    assert missing.bladex_ledger_face is None, "缺数被夹成了 False"
    # `_enqueue_turn` 逐字读的就是这个属性 ⇒ 缺数进 Turn 仍是 None
    assert Turn.model_fields["ledger_face"].default is None

    class _FaceFalse(_NoLedgerAgency):
        last_ledger_face = False

    assert _run_surfaces(_FaceFalse()).state.bladex_ledger_face is False, \
        "给了 False 却读成 None —— 三态塌成两态，反向同样是缺陷"
