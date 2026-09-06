"""MS-11：乱序重放下裁决方向必须正确（复核 20260807 发现一）。

## 这条测试守的是什么

写入时裁决的三个 verdict 里有两个（UPDATE / NOOP）本质是**时序判断**——
"新的取代旧的"。而这个"新"此前完全依赖**到达顺序**：

    增量：新 turn 天然后到 → 假设隐式成立 → 一直没出事
    重放：`scan_meta` 走 RocksDB key 序，key = `principal/agent/session/entry_id`
          entry_id 有序，但 **session 段是指纹哈希** → 跨 session 时间序被打乱

实证（复核给出、本文件 `test_key_order_really_scrambles_time` 复现）：
一条 08-05 的修正宣告（session `fp:2...`）的 key 字典序**排在**
一条 08-03 的旧断言（session `fp:7...`）**之前**。

于是旧断言以 "NEW item" 身份进裁决、修正宣告成了它的邻居 →
**旧值把新值 t_invalid**，取代链方向整个反转。而且确定性没破
（每次重建结果一致），只是**语义确定地错** —— 最难发现的那一类。

X6 纪律：本文件的词汇不得进实现代码。修法只认时间戳，不认任何领域词。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from bladex_core.adjudication import AdjudicationOp, AdjudicationVerdict, build_input
from bladex_core.fact import Fact
from bladex_proxy.models import Identity, Turn, TurnStatus
from bladex_proxy.storage.memory_hub import MemoryHub
from bladex_proxy.storage.memory_index import MemoryIndex


# ── 先证明问题存在（不然修的是想象中的 bug）────────────────────────────


def test_key_order_really_scrambles_time():
    """RocksDB key 序 ≠ 时间序：session 段是哈希，跨 session 会乱。"""
    early = Identity(user_id="u", agent_id="a", session_id="fp:7bbb", turn_index=0)
    late = Identity(user_id="u", agent_id="a", session_id="fp:2aaa", turn_index=0)
    k_early = early.storage_key("1785000000000-0")   # 时间靠前
    k_late = late.storage_key("1786000000000-0")     # 时间靠后
    assert k_late < k_early, (
        "本测试的前提是「晚的 key 反而排前面」——若这条不成立，"
        "说明 key 结构变了，下面几条要重新设计")


# ── 修法一：重放批按 (ts, key) 排序 ──────────────────────────────────────


class _Embedder:
    _DIM = 32
    _NOISE = 0.16   # 落在 0.90–0.98 相似度带（见 test_m2_adjudication_chain）

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()
            out.append([1.0] + [(b / 255.0) * self._NOISE for b in h[:self._DIM - 1]])
        return out

    @property
    def available(self) -> bool:
        return True


class _Distiller:
    """每轮一条 fact；内容取 user 正文，subject/attribute 各轮不同（逼走 LLM 路）。"""

    model_name = "stub"

    def distill(self, text: str):
        return self.distill_turn(type("P", (), {
            "user_text": text, "assistant_text": "", "assistant_role": "",
            "context_digest": "", "turn_time": "",
            "cache_text": lambda self=None: text})())

    def distill_conclusion(self, text: str):
        return self.distill(text)

    def distill_turn(self, payload):
        from bladex_core.distillation import DistillFact, DistillOutput

        text = (payload.user_text or "").strip()
        if not text:
            return DistillOutput(model_name=self.model_name)
        tag = hashlib.sha256(text.encode()).hexdigest()[:6]
        return DistillOutput(
            facts=[DistillFact(content=f"事实::{text[:60]}", item_kind="assertion",
                               subject=f"主题{tag}", attribute=f"槽位{tag}",
                               provenance="user_direct")],
            model_name=self.model_name)


class _OrderRecordingAdjudicator:
    """记录每个裁决包的 (candidate_id, observed_at, 邻居 observed_at)。"""

    model_name = "adj-stub"
    prompt_ver = "test"

    def __init__(self) -> None:
        self.packages: list = []

    def adjudicate(self, items):
        self.packages.extend(items)
        return [AdjudicationVerdict(candidate_id=it.candidate_id,
                                    op=AdjudicationOp.ADD) for it in items]


_EARLY = "较早那一轮说的是原本的处理方式没有别的补充需要记住"
_LATE = "较晚那一轮把上面的说法更正了，实际情况与之前理解的不同"


#: 🔴 `Turn.ts` **必须显式给**。它默认取构造时刻，于是同一份 fixture 建两次
#: 会得到不同的时间戳——"同输入同输出"的测试连输入都不同，测的就不是代码。
#: （首版就栽在这里：确定性测试因为造数不确定而红。）
_TS_EARLY = "2026-08-03T10:00:00+00:00"
_TS_LATE = "2026-08-05T10:00:00+00:00"


def _scrambled_hub(tmp: Path) -> MemoryHub:
    """构造「key 序与时间序相反」的 Ledger —— 复核举的那个形态。"""
    from datetime import datetime

    led = MemoryHub(tmp / "ledger")
    led.open()
    # 时间早、但 session 指纹排后
    early = Identity(user_id="u", agent_id="a", session_id="fp:7bbb", turn_index=0)
    t_early = Turn(
        identity=early, model="m",
        request_messages=[{"role": "user", "content": _EARLY}],
        response_text="ok", status=TurnStatus.OK, logical_turn=0)
    t_early.ts = datetime.fromisoformat(_TS_EARLY)
    led.put(early.storage_key("1785000000000-0"), t_early)
    # 时间晚、但 session 指纹排前
    late = Identity(user_id="u", agent_id="a", session_id="fp:2aaa", turn_index=0)
    t_late = Turn(
        identity=late, model="m",
        request_messages=[{"role": "user", "content": _LATE}],
        response_text="ok", status=TurnStatus.OK, logical_turn=0)
    t_late.ts = datetime.fromisoformat(_TS_LATE)
    led.put(late.storage_key("1786000000000-0"), t_late)
    return led


def _run(tmp: Path, adj):
    idx = MemoryIndex(tmp / "index", embedder=_Embedder(),
                      distiller=_Distiller(), adjudicator=adj)
    idx.open()
    idx.rebuild_from_hub(_scrambled_hub(tmp), full=False, max_turns=50)
    return idx


def test_replay_consumes_in_time_order(tmp_path):
    """🔴 核心断言：先处理时间早的那一轮，哪怕它的 key 排在后面。"""
    adj = _OrderRecordingAdjudicator()
    idx = _run(tmp_path, adj)
    facts = sorted(idx.all_facts(), key=lambda f: f.t_valid or f.created_at)
    assert len(facts) == 2, f"两轮该产出两条，实得 {len(facts)}"
    assert _EARLY[:12] in facts[0].content, "时间早的那条应先入库"
    assert _LATE[:12] in facts[1].content
    idx.close()


def test_candidate_sees_earlier_neighbor_not_the_other_way(tmp_path):
    """排序之后：晚到的那条才是「候选」，早的那条是它的邻居。

    反过来（旧断言当候选）就是取代链反转的入口。
    """
    adj = _OrderRecordingAdjudicator()
    idx = _run(tmp_path, adj)
    assert adj.packages, "裁决器没被调用 —— 相似度带没落对，测不到时序"
    pkg = adj.packages[-1]
    assert pkg.observed_at, "候选没有 observed_at —— 裁决器无从判断先后"
    assert pkg.neighbors and pkg.neighbors[0].observed_at
    assert pkg.observed_at > pkg.neighbors[0].observed_at, (
        f"候选 {pkg.observed_at} 不比邻居 {pkg.neighbors[0].observed_at} 晚 —— "
        "时序假设被打破，裁决方向可能反转")
    idx.close()


def test_rebuild_is_still_deterministic(tmp_path):
    """排序不能破坏重放等价性：两次重建逐位一致（G6）。"""
    snap = []
    for i in range(2):
        idx = MemoryIndex(tmp_path / f"index{i}", embedder=_Embedder(),
                          distiller=_Distiller(),
                          adjudicator=_OrderRecordingAdjudicator())
        idx.open()
        idx.rebuild_from_hub(_scrambled_hub(tmp_path / f"l{i}"),
                                full=False, max_turns=50)
        snap.append(sorted(
            (f.id, f.content, (f.t_valid or f.created_at).isoformat())
            for f in idx.all_facts()))
        idx.close()
    assert snap[0] == snap[1]


# ── 修法二：t_valid 取轮次时间（否则时序规则退化成没比）────────────────


def test_t_valid_is_the_turn_time_not_the_rebuild_time(tmp_path):
    """🔴 `t_valid` 必须是「这条事实何时成立」，不是「何时被重建出来」。

    复核处方原本写的是「邻居取 `t_observed or created_at`」，但那两个字段
    在全量重建时对**全库**都≈重建那一刻、彼此几乎相同 ——
    拿它们比先后等于没比，时序规则会退化成"候选永远更旧 → 永不 UPDATE"。
    """
    idx = _run(tmp_path, _OrderRecordingAdjudicator())
    facts = idx.all_facts()
    assert len(facts) == 2
    tvs = sorted((f.t_valid.isoformat() for f in facts if f.t_valid))
    assert tvs == [_TS_EARLY, _TS_LATE], (
        f"t_valid 应逐字等于两轮的发生时间，实得 {tvs} —— "
        "取成重建时刻的话两条会几乎相同，时序规则退化成没比")
    # 重建时刻在两条轮次时间之后 → t_observed 落在 t_valid 之后
    for f in facts:
        assert f.t_valid < f.t_observed, "t_valid（何时成立）应早于 t_observed（何时摄入）"
    idx.close()


# ── 裁决包本身（不碰存储的快测）────────────────────────────────────────


def _fact(fid: str, content: str, iso: str) -> Fact:
    from datetime import datetime

    f = Fact(id=fid, content=content, scope="personal:u")
    f.t_valid = datetime.fromisoformat(iso)
    return f


def test_build_input_carries_both_sides_time():
    cand = _fact("c", "候选", "2026-08-05T10:00:00+00:00")
    nb = _fact("n", "邻居", "2026-08-03T10:00:00+00:00")
    pkg = build_input(cand, [(nb, 0.95)])
    assert pkg.observed_at.startswith("2026-08-05")
    assert pkg.neighbors[0].observed_at.startswith("2026-08-03")


def test_explicit_observed_at_wins():
    """重放时调用方给的 turn.ts 优先 —— 那才是「这条何时发生」。"""
    cand = _fact("c", "候选", "2026-08-05T10:00:00+00:00")
    pkg = build_input(cand, [], observed_at="2026-08-01T00:00:00+00:00")
    assert pkg.observed_at.startswith("2026-08-01")


def test_missing_time_is_unknown_not_guessed():
    """取不到就留空，由 prompt 按「未知」处理 —— **不猜**。

    猜一个时间会让裁决基于虚构的先后关系做取代，而那是静默的。
    """
    from bladex_proxy.adjudicator import _build_prompt

    f = Fact(id="c", content="候选", scope="personal:u")
    f.t_valid = None
    f.t_observed = None
    f.created_at = None
    pkg = build_input(f, [])
    assert pkg.observed_at == ""
    assert "observed_at: unknown" in _build_prompt([pkg])


def test_prompt_states_the_time_rule():
    """时序规则必须真的进 prompt —— 字段加了不渲染就是又一次接线断。"""
    from bladex_proxy.adjudicator import _ADJ_SYSTEM

    low = _ADJ_SYSTEM.lower()
    assert "observed_at" in low
    assert "later" in low, "必须说明「只能被更晚的取代」"
    assert "not necessarily" in low, "必须告诉模型输入未必按时间顺序"


def test_adj_ver_bumped():
    """prompt 语义改了必须 bump —— 否则旧台账会被当成新规则下的结论复用。"""
    from bladex_proxy.adjudicator import ADJ_VER

    assert ADJ_VER == "v2-adj-001"
