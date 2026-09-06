"""归一层集成测试：Memory Hub Turn → ConversationTurn → 蒸馏候选（M0-9 / D1）。

## 这块砖为什么必须有

D1 的形态是：`ConversationTurn.assistant_progress` 与它的消费端
`_collect_progress_candidates` 2026-07-29 就都写好了、各自有单测、各自绿——
但**归一处从来没给这个字段赋过值**。两端都对，中间断了，于是那条通道自建成起
一条 fact 都没产出过，而任何一侧的单元测试都发现不了。

同型事故这已经是第三次（`refresh_profile_docs` 没有调用点、五个机制开关默认关
且不在模板、consolidator 入口只在仓库脚本里）。共同点都是**跨层的接线**没有测试。
所以这里立一条纪律：凡是"Memory Hub 里有 → 候选里应当有"的判断，都在这条链上断言，
不在任何一端的单测里断言。M1/M2 的验收都要走这条链。

链路：`MemoryHub.put(Turn)` → `MemoryIndex.rebuild_from_hub` 归一 →
`ProxyConsolidator._collect_candidates` → 候选 dict。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from bladex_core.consolidation_proxy import ProxyConsolidator
from bladex_core.fact import ConversationTurn
from bladex_proxy.models import Identity, ToolEvent, Turn, TurnStatus
from bladex_proxy.storage.memory_hub import MemoryHub
from bladex_proxy.storage.memory_index import MemoryIndex


class _Embedder:
    """确定性 mock embedder（同 test_distill_journal_rebuild 套路）。"""

    def __init__(self) -> None:
        self._dim = 64

    def embed(self, texts: list[str]) -> list[list[float]]:
        import hashlib

        out: list[list[float]] = []
        for t in texts:
            body = t.replace("passage: ", "").replace("query: ", "")
            first = body.strip().split()[0] if body.strip() else ""
            # 用 sha256 而非内置 hash：字符串 hash 每进程随机化，
            # 桶碰撞会让不同内容被判重复、测试随机翻红
            # （M4-2 的同款桩 embedder 实测两跑挂一次，2026-08-06 已修）。
            h = int.from_bytes(hashlib.sha256(first.encode()).digest()[:4], "big") % self._dim
            v = [0.0] * self._dim
            v[h] = 0.9
            out.append(v)
        return out

    @property
    def available(self) -> bool:
        return True


class _EchoDistiller:
    """把输入原样当成一条 fact 返回，并实现 distill_conclusion（结论/产出通道共用）。

    这样候选里就能逐字看到"到底是哪一段文本进了蒸馏"，比断言"有几条"更能定位断点。
    """

    model_name = "echo"

    def _out(self, text: str):
        from bladex_core.distillation import DistillFact, DistillOutput

        return DistillOutput(
            facts=[DistillFact(content=text[:200], kind="general", entities=[])],
            matter_proposals=[],
            model_name="echo",
        )

    def distill(self, text: str):
        return self._out(text)

    def distill_conclusion(self, text: str):
        return self._out(text)


# 有 tool call 的一轮：assistant 正文 = 「调工具前写的说明/分析/发现」= 工作产出
_PROGRESS_TEXT = (
    "我先看了一下 memory_index.py 的归一处，发现 assistant_progress 字段"
    "从头到尾没有被赋过值，所以工作产出通道其实一条数据都没收到过。"
    "接下来我去 rebuild_from_hub 里补上这一行。"
)
# 无 tool call 的一轮：assistant 正文 = 最终回答 = 结论
_CONCLUSION_TEXT = (
    "结论是：工作产出通道的断点在归一层，不在蒸馏侧，已在 rebuild 归一处补齐赋值。"
)
_USER_TEXT = "帮我查一下工作产出通道为什么一条事实都没有产出过"


def _make_hub(tmpdir: Path) -> MemoryHub:
    ledger = MemoryHub(tmpdir / "ledger")
    ledger.open()

    # ① 带 tool call 的一轮 → 应走 progress 通道
    ident0 = Identity(user_id="u1", agent_id="claude-code", session_id="s1", turn_index=0)
    ledger.put(ident0.storage_key("1719907200-0"), Turn(
        identity=ident0, model="test",
        request_messages=[{"role": "user", "content": _USER_TEXT}],
        response_text=_PROGRESS_TEXT,
        tool_events=[ToolEvent(direction="call", name="read_file")],
        status=TurnStatus.OK, logical_turn=0,
    ))

    # ② 无 tool call 的一轮 → 应走 conclusion 通道（互斥性对照）
    ident1 = Identity(user_id="u1", agent_id="claude-code", session_id="s1", turn_index=1)
    ledger.put(ident1.storage_key("1719907201-0"), Turn(
        identity=ident1, model="test",
        request_messages=[{"role": "user", "content": _USER_TEXT}],
        response_text=_CONCLUSION_TEXT,
        tool_events=[],
        status=TurnStatus.OK, logical_turn=1,
    ))
    return ledger


def _normalized_turns(tmp_path: Path, monkeypatch) -> list[ConversationTurn]:
    """跑真实归一路径，把 MemoryIndex 交给 consolidator 的 ConversationTurn 截下来。

    刻意不去调 `_normalize`（它不存在）或复制一份归一逻辑——**必须是生产路径本身**，
    否则这条链上的断点照样测不出来（那正是 D1 逃过所有测试的方式）。
    """
    ledger = _make_hub(tmp_path)
    index = MemoryIndex(tmp_path / "index", embedder=_Embedder(), distiller=None)
    index.open()

    captured: list[list[ConversationTurn]] = []
    real = ProxyConsolidator.consolidate_turns

    def spy(self, turns, existing_facts=None):
        captured.append(list(turns))
        return real(self, turns, existing_facts)

    monkeypatch.setattr(ProxyConsolidator, "consolidate_turns", spy)
    index.rebuild_from_hub(ledger, full=True)
    index.close()
    ledger.close()

    assert captured, "rebuild 没有把任何 ConversationTurn 交给 consolidator"
    return [t for batch in captured for t in batch]


# ── M0-9：progress 通道从 Memory Hub 一路通到候选 ─────────────────────────


def test_progress_field_is_populated_by_normalization(tmp_path, monkeypatch):
    """带 tool call 的轮次，归一后 `assistant_progress` 必须非空（D1 的直接断言）。"""
    turns = _normalized_turns(tmp_path, monkeypatch)
    with_call = [t for t in turns if t.assistant_progress]
    assert with_call, (
        "归一层没有填 assistant_progress —— 工作产出通道断在这里，"
        "两端的单测都发现不了（D1）"
    )
    assert _PROGRESS_TEXT[:20] in with_call[0].assistant_progress


def test_conclusion_and_progress_are_mutually_exclusive(tmp_path, monkeypatch):
    """同一轮要么是结论（无 tool call）要么是产出（有），不得两者都填。"""
    turns = _normalized_turns(tmp_path, monkeypatch)
    for t in turns:
        assert not (t.assistant_conclusion and t.assistant_progress), (
            f"轮次 {t.ledger_key} 同时填了结论与产出，语义互斥被破坏"
        )
    assert any(t.assistant_conclusion for t in turns), "结论通道不得被本次改动误伤"
    assert any(t.assistant_progress for t in turns)


def test_progress_reaches_distill_candidates(tmp_path, monkeypatch):
    """端到端：归一出来的 turn 交给真 consolidator，progress 通道产出非空候选。"""
    turns = _normalized_turns(tmp_path, monkeypatch)
    consolidator = ProxyConsolidator(embedder=_Embedder(), distiller=_EchoDistiller())
    candidates = consolidator._collect_candidates(turns)

    by_tag = {}
    for c in candidates:
        by_tag.setdefault(c.get("tags", ""), []).append(c)

    assert "origin:progress" in by_tag, (
        f"progress 通道零候选；实际通道：{sorted(by_tag)}"
    )
    assert "origin:conclusion" in by_tag, "结论通道回归（它扛着 66% 的 Fact 产出）"
    # 内容确实来自 assistant 正文，而不是把 user 消息重收了一遍
    assert any(_PROGRESS_TEXT[:20] in c["content"] for c in by_tag["origin:progress"])


def test_failed_turn_produces_no_progress(tmp_path, monkeypatch):
    """失败轮不进产出通道（status != ok），否则会把上游报错当成"发现"记下来。"""
    ledger = MemoryHub(tmp_path / "ledger_err")
    ledger.open()
    ident = Identity(user_id="u1", agent_id="a1", session_id="s9", turn_index=0)
    ledger.put(ident.storage_key("1719907300-0"), Turn(
        identity=ident, model="test",
        request_messages=[{"role": "user", "content": _USER_TEXT}],
        response_text=_PROGRESS_TEXT,
        tool_events=[ToolEvent(direction="call", name="read_file")],
        status=TurnStatus.FAILED, error="upstream 502", logical_turn=0,
    ))

    index = MemoryIndex(tmp_path / "index_err", embedder=_Embedder(), distiller=None)
    index.open()
    captured: list[ConversationTurn] = []
    real = ProxyConsolidator.consolidate_turns

    def spy(self, turns, existing_facts=None):
        captured.extend(turns)
        return real(self, turns, existing_facts)

    monkeypatch.setattr(ProxyConsolidator, "consolidate_turns", spy)
    index.rebuild_from_hub(ledger, full=True)
    index.close()
    ledger.close()

    assert all(not t.assistant_progress for t in captured)


def test_envelope_turn_is_not_dropped_wholesale(tmp_path, monkeypatch):
    """🔴 MS-2：末条 user 是**信封**的轮次不得整轮丢——assistant 那侧照常收。

    信封指纹说的是"用户那侧这条消息是信封"，不代表这一轮 assistant 没做实事。
    整轮丢掉的实测代价：claude-code 123 轮里 120 轮被跳过。

    这条同时是 `DISTILL_ONLY_AUX_RULES` 第二个消费方的接线测试——在此之前
    那个集合只有路由侧一个读者，rebuild 照旧整轮丢，于是 2026-07-29 记录的
    "信封指纹改为交给 envelope 剥离"从未真正生效（第六次"机制写完接线断"）。
    """
    ledger = MemoryHub(tmp_path / "ledger_env")
    ledger.open()
    ident = Identity(user_id="u1", agent_id="claude-code", session_id="s1", turn_index=0)
    ledger.put(ident.storage_key("1719907500-0"), Turn(
        identity=ident, model="test",
        request_messages=[{"role": "user", "content":
                           "<transcript>历史会话转录……</transcript>"}],
        response_text=_PROGRESS_TEXT,
        tool_events=[ToolEvent(direction="call", name="read_file")],
        status=TurnStatus.OK, logical_turn=0,
    ))

    index = MemoryIndex(tmp_path / "index_env", embedder=_Embedder(), distiller=None)
    index.open()
    captured: list[ConversationTurn] = []
    real = ProxyConsolidator.consolidate_turns
    monkeypatch.setattr(
        ProxyConsolidator, "consolidate_turns",
        lambda self, turns, existing_facts=None: (
            captured.extend(turns) or real(self, turns, existing_facts)
        ),
    )
    index.rebuild_from_hub(ledger, full=True)
    index.close()
    ledger.close()

    assert captured, "信封轮被整轮丢弃了 —— assistant 的工作产出一起没了"
    assert captured[0].assistant_progress, "该轮的工作产出必须仍被收集"


def test_native_aux_turn_is_still_dropped(tmp_path, monkeypatch):
    """反向红线：Hermes 原生 aux（委派回执类）仍然整轮丢，T9 收益不得回退。"""
    ledger = MemoryHub(tmp_path / "ledger_aux")
    ledger.open()
    ident = Identity(user_id="u1", agent_id="hermes", session_id="s1", turn_index=0)
    ledger.put(ident.storage_key("1719907600-0"), Turn(
        identity=ident, model="test",
        request_messages=[{"role": "user", "content":
                           "[ASYNC DELEGATION BATCH COMPLETE] 3 tasks finished"}],
        response_text=_PROGRESS_TEXT,
        tool_events=[ToolEvent(direction="call", name="t")],
        status=TurnStatus.OK, logical_turn=0,
    ))

    index = MemoryIndex(tmp_path / "index_aux", embedder=_Embedder(), distiller=None)
    index.open()
    captured: list[ConversationTurn] = []
    real = ProxyConsolidator.consolidate_turns
    monkeypatch.setattr(
        ProxyConsolidator, "consolidate_turns",
        lambda self, turns, existing_facts=None: (
            captured.extend(turns) or real(self, turns, existing_facts)
        ),
    )
    index.rebuild_from_hub(ledger, full=True)
    index.close()
    ledger.close()

    assert captured == [], "Hermes 原生内部调用仍应整轮跳过"


@pytest.mark.parametrize("status,has_call,want_channel", [
    (TurnStatus.OK, True, "progress"),
    (TurnStatus.OK, False, "conclusion"),
    (TurnStatus.FAILED, True, None),
    (TurnStatus.FAILED, False, None),
])
def test_channel_matrix(tmp_path, monkeypatch, status, has_call, want_channel):
    """四象限矩阵钉死通道判定（status × 有无 tool call）。"""
    ledger = MemoryHub(tmp_path / f"l_{status.value}_{has_call}")
    ledger.open()
    ident = Identity(user_id="u1", agent_id="a1", session_id="sx", turn_index=0)
    ledger.put(ident.storage_key("1719907400-0"), Turn(
        identity=ident, model="test",
        request_messages=[{"role": "user", "content": _USER_TEXT}],
        response_text=_PROGRESS_TEXT,
        tool_events=[ToolEvent(direction="call", name="t")] if has_call else [],
        status=status, logical_turn=0,
    ))
    index = MemoryIndex(tmp_path / f"i_{status.value}_{has_call}",
                        embedder=_Embedder(), distiller=None)
    index.open()
    captured: list[ConversationTurn] = []
    real = ProxyConsolidator.consolidate_turns
    monkeypatch.setattr(
        ProxyConsolidator, "consolidate_turns",
        lambda self, turns, existing_facts=None: (
            captured.extend(turns) or real(self, turns, existing_facts)
        ),
    )
    index.rebuild_from_hub(ledger, full=True)
    index.close()
    ledger.close()

    got = None
    for t in captured:
        if t.assistant_progress:
            got = "progress"
        elif t.assistant_conclusion:
            got = "conclusion"
    assert got == want_channel


# ══════════════════════════════════════════════════════════════════════════
# M1-2：v4 两级路由 —— 分流 + 通道合一，仍然走这条链（不在任何一端单测里断言）
# ══════════════════════════════════════════════════════════════════════════


class _TurnDistiller:
    """实现 v4 `distill_turn` 的桩：把收到的 payload 记下来，产一条 fact。

    记 payload 是为了断言"模型到底看见了什么"——通道合一之后，
    "user 与 assistant 有没有同包"只能从这里看出来。
    """

    model_name = "turnstub"

    def __init__(self) -> None:
        self.payloads: list = []

    def distill(self, text: str):
        from bladex_core.distillation import DistillFact, DistillOutput

        return DistillOutput(facts=[DistillFact(content=text[:200])],
                             model_name=self.model_name)

    def distill_conclusion(self, text: str):
        return self.distill(text)

    def distill_turn(self, payload):
        from bladex_core.distillation import DistillFact, DistillOutput
        from bladex_core.fact import Provenance

        self.payloads.append(payload)
        facts = []
        if payload.user_text:
            facts.append(DistillFact(
                content=f"U::{payload.user_text[:80]}",
                provenance=Provenance.USER_DIRECT.value, importance=7))
        if payload.assistant_text:
            facts.append(DistillFact(
                content=f"A::{payload.assistant_text[:80]}",
                provenance=payload.assistant_role or Provenance.CONCLUSION.value,
                importance=6))
        return DistillOutput(facts=facts, model_name=self.model_name)

    def summarize_file(self, path: str, content: str):
        return f"摘要:{len(content)}字符", ["关键词"]


def _candidates_via_chain(tmp_path, monkeypatch, ledger_builder, distiller=None):
    """从 Memory Hub 走生产归一路径，截下 consolidator 实际收到的候选。"""
    ledger = ledger_builder(tmp_path)
    distiller = distiller or _TurnDistiller()
    index = MemoryIndex(tmp_path / "index", embedder=_Embedder(), distiller=distiller)
    index.open()

    captured: list[list[dict]] = []
    real = ProxyConsolidator._collect_candidates

    def spy(self, turns):
        out = real(self, turns)
        captured.append(list(out))
        spy.consolidator = self
        return out

    monkeypatch.setattr(ProxyConsolidator, "_collect_candidates", spy)
    index.rebuild_from_hub(ledger, full=True)
    index.close()
    ledger.close()
    flat = [c for batch in captured for c in batch]
    return flat, distiller, getattr(spy, "consolidator", None)


def test_v4_packs_user_and_assistant_in_one_call(tmp_path, monkeypatch):
    """通道合一：一轮**一次**调用，user 与 assistant 同包。

    v3 是三条独立流水线（user / conclusion / progress 各蒸各的），
    彼此不知道对方存在——同一轮里"要求做什么"和"做完的结论"被当成两件事，
    产出因此大量是「用户询问 XXX」而丢掉真正的结论。
    """
    cands, distiller, _ = _candidates_via_chain(tmp_path, monkeypatch, _make_hub)

    # 两轮 → 两次调用（不是六次）
    assert len(distiller.payloads) == 2, (
        f"一轮该只调一次统一蒸馏，实际 {len(distiller.payloads)} 次")
    both = [p for p in distiller.payloads if p.user_text and p.assistant_text]
    assert both, "至少有一轮应该 user 与 assistant 同包"


def test_v4_carries_time_anchor(tmp_path, monkeypatch):
    """④ 时间锚：turn 时间是已知量，必须进 payload（禁止产出"某日"的前提）。"""
    _, distiller, _ = _candidates_via_chain(tmp_path, monkeypatch, _make_hub)
    assert all(p.turn_time for p in distiller.payloads), "每次调用都该带 turn 时间"


def test_v4_provenance_and_tags_are_dual_written(tmp_path, monkeypatch):
    """🔴 Q4：`provenance` 是新判据，`tags=origin:*` 是待验证项 A3 的**现有**判据。

    通道合一后来源只剩 provenance 一处，若不双写，就等于在 A3 被验证之前
    把它的判据拆掉了——那条验证再也做不成。
    """
    cands, _, _ = _candidates_via_chain(tmp_path, monkeypatch, _make_hub)
    assert cands, "候选为空 —— 链断了"
    for c in cands:
        prov = c.get("provenance", "")
        assert prov, f"候选缺 provenance: {c['content'][:40]}"
        assert c.get("tags") == f"origin:{prov}", (
            f"tags 与 provenance 必须双写且一致：{c.get('tags')} vs {prov}")

    provs = {c["provenance"] for c in cands}
    assert "progress" in provs, "带 tool call 的轮次应落 progress 通道"
    assert "conclusion" in provs, "无 tool call 的轮次应落 conclusion 通道"


def test_v4_passes_rating_through_to_fact_importance(tmp_path, monkeypatch):
    """⑤ importance：蒸馏给的内容内在分要一路落到 Fact 上（M1-5 的接线）。"""
    cands, _, _ = _candidates_via_chain(tmp_path, monkeypatch, _make_hub)
    assert any(c.get("importance_rating", 0) > 0 for c in cands), (
        "rating 没有从蒸馏产出传到候选 —— 接线断了")


def _make_rulefile_ledger(tmpdir: Path) -> MemoryHub:
    """一轮：user 消息是 agent 回传的规则文件（失效链 A 的入口形态）。"""
    ledger = MemoryHub(tmpdir / "ledger")
    ledger.open()
    body = ("Contents of /w/AGENTS.md (project instructions):\n\n"
            + "这里是一大段规则正文，描述项目怎么做事。" * 40)
    ident = Identity(user_id="u1", agent_id="codex", session_id="s9", turn_index=0)
    ledger.put(ident.storage_key("1719907300-0"), Turn(
        identity=ident, model="test",
        request_messages=[{"role": "user", "content": body}],
        response_text="好的", tool_events=[], status=TurnStatus.OK, logical_turn=0,
    ))
    return ledger


def test_rulefile_turn_produces_profile_obs_not_preference(tmp_path, monkeypatch):
    """🔴 失效链 A 的根修：规则文件不许被蒸成 assertion/preference。

    库里那两条与当前架构直接矛盾的"定位"、以及 6 条 `## Goal` 形态的伪 preference，
    都是规则文件走了"用户陈述蒸馏"进来的——admission rule 看见陈述句就产出条目，
    它无法知道这是过时文档而非用户当下的陈述。
    """
    cands, distiller, consolidator = _candidates_via_chain(
        tmp_path, monkeypatch, _make_rulefile_ledger)

    # ① 规则文件正文**不进统一蒸馏调用** —— 这才是失效链 A 的根修
    assert not any(p.user_text for p in distiller.payloads), (
        "规则文件正文进了话语路，admission rule 会把它当成用户当下的陈述")
    # ② 分流确实认出了它（不是"恰好没产出"）
    routes = dict(getattr(consolidator, "route_counts", {}) or {})
    assert routes.get("rulefile", 0) >= 1, f"分流没认出规则文件：{routes}"
    # ③ 也**不产条目**：正文捕获归 E7.3（update_profiles_from_turn 读未剥离原文
    #    调 _capture_rule_file_bodies），这里再造一条摘要式 profile_obs 是重复劳动
    #    + 给库添噪声（X3：先问生产端能不能不产生）
    kinds = {c.get("item_kind") for c in cands}
    assert not (kinds & {"assertion", "preference"}), (
        f"规则文件不该产出 assertion/preference，实得 {kinds}")


def _make_paste_ledger(tmpdir: Path) -> MemoryHub:
    """一轮：user 粘了一大段结构化文档（不是话语）。"""
    ledger = MemoryHub(tmpdir / "ledger")
    ledger.open()
    body = "看看这个：\n\n```python\n" + "value = compute(x)\n" * 200 + "```\n"
    ident = Identity(user_id="u1", agent_id="codex", session_id="s8", turn_index=0)
    ledger.put(ident.storage_key("1719907400-0"), Turn(
        identity=ident, model="test",
        request_messages=[{"role": "user", "content": body}],
        response_text="好的", tool_events=[], status=TurnStatus.OK, logical_turn=0,
    ))
    return ledger


def test_pasted_document_becomes_an_index_entry_not_assertions(tmp_path, monkeypatch):
    """粘贴物走文档路：产文件内容索引，**不产 assertion/preference**。

    附录 D.4：把一份文档切成六句假话语去蒸，每一段都在错误的语义契约下被处理
    （准入规则、subject/attribute、preference 判定全部失准）。
    """
    cands, distiller, consolidator = _candidates_via_chain(
        tmp_path, monkeypatch, _make_paste_ledger)

    assert consolidator is not None
    docs = getattr(consolidator, "document_entries", [])
    assert docs, "粘贴物没有排队进文档路"
    assert docs[0]["path"].startswith("paste://"), "粘贴物没有磁盘路径，用内容 hash 标识"

    kinds = {c.get("item_kind") for c in cands}
    assert not (kinds & {"assertion", "preference"}), (
        f"文档内容不是用户断言，不该产出 {kinds}")


def test_v4_off_falls_back_to_v3(tmp_path, monkeypatch):
    """回滚通道：关掉开关就退回 v3 三条流水线，行为逐字不变。

    这条是"默认开"能被接受的前提——出问题时有一个确定的退路。
    """
    monkeypatch.setenv("BLADEX_DISTILL_V4", "0")
    cands, distiller, _ = _candidates_via_chain(tmp_path, monkeypatch, _make_hub)
    assert not distiller.payloads, "关掉 v4 后不该再调 distill_turn"
    assert cands, "v3 路径仍应产出候选"
    # v3 没有 provenance 概念
    assert all(not c.get("provenance") for c in cands)
