""""读了"代理读数（F-A2 / MQ-A35，2026-09-07 批 F）。

量的是：模型收到账本块后，**这一轮的第一个动作**有没有引用 Next/Open 里的内容。
零 LLM、纯词面（`ledger_runtime.candidate_units`，与账本相关性位同一把分词、同一张
停表）。**只打日志不入 Turn** —— 口径还要改，日志改口径不欠债、schema 欠债。

分母纪律：本文件只证明"这把尺子能读出命中与不命中"。live 的分母（有账本且非 aux
的轮数）在 HANDOFF 的 live 判据里，不在单测里——单测里造一个分母只会自证。

钉四件事：
① 命中 ≥1：首个动作的文本里出现 Next 条目的词面单元；
② 零命中：首个动作与账本完全无关 ⇒ hits=0（且这一条**照样打日志**——
   0 是读数，不打才是缺数）；
③ 无账本锚（aux / 模块关）⇒ **一条都不打**（否则分母被稀释）；
④ 判别力对照：把 Next 文本换成随机串，同一个首动作 ⇒ hits 归 0。
   —— 没有这一条，①的命中可能只是"两段中文总会撞几个 2-gram"。
"""

from __future__ import annotations

import structlog.testing

from bladex_core.ledger import ACTOR_MODEL, LedgerEntry, add_entry, new_ledger
from bladex_proxy.models import LedgerAnchor, ToolEvent
from bladex_proxy.server.orchestration import _ledger_next_referenced

_LID = "ldg-fa2"


class _Agency:
    """`pool` 只为 `rev_now` 那一轴——够用的最小替身。

    🔴 F1.3 / MQ-A49 起，单元集不再从这里读（进程级"最近一次"，并发会串台），
    改由 `request.state` 传（见 `_req`）。
    """

    def __init__(self, pool, breakdown=None):
        self.pool = pool
        self.last_ledger_breakdown = breakdown


def _req(agency, breakdown):
    """请求替身：`_apply_agency_surfaces` 停下的那三个字段 + `app.state.agency`。

    停放语义与生产逐字相同：没有 breakdown ⇒ `pending_units = None`
    （缺数，不是空集合）。
    """
    import types
    state = types.SimpleNamespace(
        bladex_pending_units=(None if breakdown is None
                              else frozenset(breakdown.pending_units)),
        bladex_pending_rare=(frozenset() if breakdown is None
                             else frozenset(breakdown.pending_rare)))
    return types.SimpleNamespace(
        state=state, app=types.SimpleNamespace(
            state=types.SimpleNamespace(agency=agency)))


def _pool(next_text: str = "把 probe_ledger_trajectory 的分桶接上 self_conflict",
          open_text: str = "确认 flash daemon 的重发通道还需不需要"):
    led = new_ledger(ledger_id=_LID, title="批 F 仪器棒", goal="立三轴的尺子")
    led = add_entry(led, "next", LedgerEntry(text=next_text), actor=ACTOR_MODEL)
    led = add_entry(led, "open", LedgerEntry(text=open_text), actor=ACTOR_MODEL)
    return {_LID: led}


def _bd(pool, lid: str = _LID, *, df_pool=None):
    """按注入那一刻的账本算 `pending_units` / `pending_rare`
    （生产侧 `_build_ledger_block` 的同两句）。"""
    from bladex_core.ledger_runtime import candidate_units
    from bladex_proxy.agency import LedgerBlockBreakdown, rare_pending_units
    led = pool.get(lid)
    text = "" if led is None else " ".join(
        e.text for sec in ("next", "open") for e in led.entries(sec))
    units = frozenset(candidate_units(text)) if led is not None else frozenset()
    return LedgerBlockBreakdown(
        ledger_id=lid, rev=getattr(led, "rev", -1), pending_units=units,
        # F1.2/MQ-A51：`pending_text` 必传（路径类判据要看原文，从 token 反推不出）。
        pending_rare=rare_pending_units(df_pool if df_pool is not None else pool,
                                        units, pending_text=text))


def _anchor() -> LedgerAnchor:
    return LedgerAnchor(ledger_id=_LID, rev=3, sections={"next": 40})


def _run(pool, anchor, response_text="", tool_events=None, breakdown=-1,
         pool_now=None, df_pool=None):
    """`breakdown` 缺省 = 按 `pool` 现算（模型看到的那一版）；
    `pool_now` 可传"这一轮结束后的池"，用来演 MQ-A43 那个自指形态；
    `df_pool` 可传更大的池，用来演 MQ-A47 的稀有锚。"""
    bd = _bd(pool, df_pool=df_pool) if breakdown == -1 else breakdown
    ag = _Agency(pool_now if pool_now is not None else pool, bd)
    with structlog.testing.capture_logs() as cap:
        _ledger_next_referenced(_req(ag, bd), anchor, response_text,
                                tool_events or [])
    return [e for e in cap if e["event"] == "agency_ledger_next_referenced"]


# ── ① 命中 ─────────────────────────────────────────────────────────────────


def test_hits_when_first_action_quotes_next():
    recs = _run(_pool(), _anchor(),
                response_text="好，我先把 probe_ledger_trajectory 的分桶接上。")
    assert len(recs) == 1
    r = recs[0]
    assert r["hits"] >= 1 and r["next_units"] > 0
    assert r["first_action"] == "text", "没有工具调用 ⇒ 首动作是正文"
    assert r["ledger"] == _LID and r["rev"] == 3
    assert "probe_ledger_trajectory" in r["sample"]


def test_first_action_is_the_tool_call_when_there_is_one():
    """有工具调用 ⇒ 首动作记工具名，且 `arguments` 参与词面比对。

    🔴 分词是**整 token**（`[a-z0-9][a-z0-9_\\-.]{2,}`，扩展名算在 token 里）：
    `probe_ledger_trajectory` 与 `probe_ledger_trajectory.py` 是**两个不同单元**。
    条目里写全名才命中——这不是缺陷而是这把尺子的口径，写在这里免得下一棒
    看到 live 上"改了对的文件却零命中"去改机制（feedback_verify_local_facts_first）。
    """
    ev = [ToolEvent(tool_name="Edit", direction="call",
                    arguments={"file_path": "scripts/probe_ledger_trajectory.py"}),
          ToolEvent(tool_name="Edit", direction="result", result="ok")]
    recs = _run(_pool(next_text="把 scripts/probe_ledger_trajectory.py 的分桶接上"),
                _anchor(), response_text="", tool_events=ev)
    assert len(recs) == 1
    assert recs[0]["first_action"] == "Edit"
    assert recs[0]["hits"] >= 1, "工具参数里的路径应当命中 Next 条目"
    assert "probe_ledger_trajectory.py" in recs[0]["sample"]


# ── ② 零命中照样是读数 ─────────────────────────────────────────────────────


def test_zero_hits_is_still_logged():
    recs = _run(_pool(), _anchor(),
                response_text="The weather in Beijing is sunny today.")
    assert len(recs) == 1, "0 是读数，不打才是缺数"
    assert recs[0]["hits"] == 0
    assert recs[0]["next_units"] > 0, "分母必须非零，否则这条读数无意义"


# ── ③ 无锚不打 ─────────────────────────────────────────────────────────────


def test_no_anchor_logs_nothing():
    assert _run(_pool(), None, response_text="probe_ledger_trajectory") == []
    assert _run(_pool(), LedgerAnchor(), response_text="probe_ledger_trajectory") == []


def test_no_breakdown_logs_nothing():
    """没有 breakdown = 这一轮没注块 ⇒ 无对象可比，不造读数。"""
    assert _run(_pool(), _anchor(), response_text="probe_ledger_trajectory",
                breakdown=None) == []


# ── 🔴 MQ-A43：单元集必须来自**模型看到的那一版**，不是这一轮结束后的池 ──


def test_units_come_from_the_version_the_model_saw_not_the_pool_now():
    """live 首验的自指形态：模型本轮 `bladex_ledger_update` 写了 Next，
    池已是新版；若从池现算，U ⊆ V 恒成立（实录 hits=20 next_units=20）。

    判别力：把 `pending_units` 换成按 `pool_now` 算的 ⇒ hits 跳到满命中，本条必红。
    """
    seen = _pool(next_text="", open_text="")            # 模型看到时 Next/Open 全空
    seen[_LID] = seen[_LID].model_copy(update={"sections": {"next": [], "open": []}})
    after = _pool(next_text="把 F-A3 的分桶接上 self_conflict",
                  open_text="确认重发通道")              # 它自己刚写进去的
    ev = [ToolEvent(tool_name="bladex_ledger_update", direction="call",
                    arguments={"section": "next", "op": "add",
                               "text": "把 F-A3 的分桶接上 self_conflict"})]
    recs = _run(seen, _anchor(), tool_events=ev, pool_now=after)
    assert len(recs) == 1
    r = recs[0]
    assert r["next_units"] == 0 and r["hits"] == 0, \
        "模型看到的那一版是空的 —— 命中必须为 0，不许拿它刚写的东西自证"
    assert r["usable"] is False, "零分母 ⇒ 这条不进命中率的分母"


def test_usable_flag_splits_zero_denominator_rows():
    """`usable` 是**生产侧**判据（同 F-B2 的 self_conflict）——消费侧照读不重算。"""
    assert _run(_pool(), _anchor(), response_text="x")[0]["usable"] is True
    empty = _pool()
    empty[_LID] = empty[_LID].model_copy(update={"sections": {"next": [], "open": []}})
    assert _run(empty, _anchor(), response_text="x")[0]["usable"] is False


# ── ④ 判别力：Next 换随机串 ⇒ 命中归 0 ─────────────────────────────────────


def test_discriminative_random_next_drops_hits_to_zero():
    action = "好，我先把 probe_ledger_trajectory 的分桶接上。"
    hit = _run(_pool(), _anchor(), response_text=action)[0]["hits"]
    miss = _run(_pool(next_text="qzxvk9 wprmt4 zzz", open_text="lmnop7 qqqq2"),
                _anchor(), response_text=action)[0]["hits"]
    assert hit >= 1 and miss == 0, \
        "同一首动作换掉 Next 内容后必须归零——否则命中只是中文 2-gram 的底噪"


# ── 🔴 MQ-A47：稀有锚剔掉路径底噪（live 病例回放）──────────────────────────


def _pool_with_paths(n: int = 6):
    """n 本账本，待办里都写 `~/Documents/Hermes/…` 路径 —— 复现 live 的语料形状。

    live 读数：124 条可用里 **63 条（51%）** 命中的就是 `documents,hermes` 两个 token，
    因为几乎每本账本的 Next/Open 都提到同一个目录树。
    """
    pool = {}
    for i in range(n):
        led = new_ledger(ledger_id=f"ldg-p{i}", title=f"任务{i}", goal="g")
        led = add_entry(led, "next", LedgerEntry(
            text=f"更新 ~/Documents/Hermes/proj{i}/notes.md"), actor=ACTOR_MODEL)
        pool[led.ledger_id] = led
    # 被观测的那一本：路径 + 一个只有它才有的标识符
    led = new_ledger(ledger_id=_LID, title="台球视觉", goal="g")
    led = add_entry(led, "next", LedgerEntry(
        text="在 ~/Documents/Hermes/pool/ 下把 map50 跑到 0.8"), actor=ACTOR_MODEL)
    pool[_LID] = led
    return pool


def test_rare_anchor_strips_the_path_noise_floor():
    """🔴 live 病例回放：首动作只是在同一目录下动文件 ⇒ `hits` 被 `documents,hermes`
    撑起来，而 `hits_rare` 必须是 0（那不是"引用了待办"）。

    判别力：不做稀有锚（`hits_rare` 直接等于 `hits`）⇒ 本条必红。
    """
    pool = _pool_with_paths()
    ev = [ToolEvent(tool_name="Edit", direction="call",
                    arguments={"file_path": "/Users/j/Documents/Hermes/other/x.md"})]
    r = _run(pool, _anchor(), tool_events=ev)[0]
    assert r["hits"] >= 2, "阳性对照：底噪确实存在（否则这条测试没在测东西）"
    assert "documents" in r["sample"] or "hermes" in r["sample"]
    assert r["hits_rare"] == 0, "通用路径 token 不算引用待办"


def test_rare_anchor_keeps_the_real_signal():
    """只有这件事才有的词（`map50`）必须留下来 —— 剔底噪不能把信号一起剔掉。"""
    pool = _pool_with_paths()
    r = _run(pool, _anchor(),
             response_text="我把 map50 提到 0.82 了")[0]
    assert r["hits_rare"] >= 1 and "map50" in r["rare_sample"]
    assert r["rare_units"] >= 1, "分母也要报——为 0 的行同样不该进命中率"


def test_rare_anchor_does_not_pretend_to_filter_a_tiny_pool():
    """df 语料不够 ⇒ df 那一条原样返回（不猜，也不假装筛过）。

    🔴 F1.2/MQ-A51 起，小池保护的分母是"**Next/Open 非空**的本"而不是
    `len(pool)`；路径类剔除与 df 无关，小池上照做（对照见
    `test_f1_rare_anchor_pool.py::test_small_corpus_still_strips_paths_but_does_not_fake_df`）。
    """
    from bladex_proxy.agency import rare_pending_units
    units = frozenset({"documents", "hermes", "map50"})
    assert rare_pending_units({}, units, pending_text="") == units
    assert rare_pending_units({"a": None}, units, pending_text="") == units


def test_instrument_never_raises():
    """高危 1：仪器绝不阻断入库。request.state 上塞个坏对象也只是一条 warning。"""
    import types

    class _Boom:
        def __iter__(self):
            raise RuntimeError("boom")

    req = types.SimpleNamespace(
        state=types.SimpleNamespace(bladex_pending_units=_Boom(),
                                    bladex_pending_rare=frozenset()),
        app=types.SimpleNamespace(state=types.SimpleNamespace(agency=None)))
    with structlog.testing.capture_logs() as cap:
        _ledger_next_referenced(req, _anchor(), "x", [])
    assert any(e["event"] == "agency_ledger_next_referenced_failed" for e in cap)
