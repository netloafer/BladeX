"""GM-1 延续信号进候选：**生产接线**（MQ-S13）。

🔴 为什么必须有这一份，而不能只有 `test_continuation.py`：
本仓刚吃过第七例「机制单测通过 ≠ 机制在生产接通」——E6.1 短指代扩写的两条测试
一直绿着，因为它们**手工传参**绕开了 messages→units 那段接线，而生产里那个扩写
是个恒等式。所以本文件只走真实路径：Memory Hub 写 turn → `rebuild_from_hub`。

## 🔴 判据换过一次：不看"L4 有没有看见姐妹卡"，看延续通路自己的计数器

初版断言「姐妹卡出现在 L4 的候选里」，gate 上三条阴性对照连红才发现：
**姐妹卡本来就在**——`search_matters` 是纯向量召回，库里统共一张卡，top-k 必然返回它。
那条断言对开关关、非延续、换话题**全都成立**，于是阳性用例也是**假通过**。
教训 51 第⑤种形态的又一例：会证实任何输入的断言，比没有断言更糟。

判据改成 `index_attrib_continuation` 的三个计数器（从真实日志里捞，不是测试自己算的）：

    cont_pairs      判为「延续 ∧ 无新主体」的轮数        ← 判据成立与否
    cont_supplied   延续通路**提出**的卡数（去重前）      ← 通路有没有接上
    cont_candidates 去重后真进候选池的次数               ← 向量是否已经召回了

承重的是 `cont_supplied`：>0 即证明"扫描判出延续 → 解析到上一轮的 Matter →
走到候选注入点"整条链通，且**不受向量路径干扰**。
`cont_candidates` 可以是 0（向量本来就召回了 = 好事），不能拿它当接线判据。

## 同一趟还抓出一个真 bug

首版把"上一轮"记在单次 pass 的字典里 → **跨批次的延续对全部丢失**（consolidator
每 60s 一批，人说话的间隔通常大于 60s，绝大多数相邻轮天生落在不同批）。
是日志里的 `cont_pairs=0` 把它抖出来的。本文件的两轮**故意分两次
`rebuild_from_hub`**，就是为了钉住这条跨批次路径。

阴性对照三组：非延续 / 引入新主体 / 开关关闭。
"""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from bladex_core.attribution import LinkJudgeItem, LinkJudgeResult, LinkVerdict
from bladex_core.matter import Matter, MatterStatus
from bladex_proxy.models import Identity, SessionIdSource, Turn, TurnStatus
from bladex_proxy.storage.memory_hub import MemoryHub
from bladex_proxy.storage.memory_index import MemoryIndex

_TS = datetime(2026, 8, 18, 12, 0, 0, tzinfo=UTC)
_SISTER_ID = "m-sister"


@pytest.fixture(autouse=True)
def _isolate_d1():
    """🔴 本文件的剧本测的是 **GM-1 延续通路**，必须把 D1 钉关。

    2026-08-21 翻 `BLADEX_TASK_STATE_D1` 默认时这三条一起红，读下来是同一件事：
    **D1 的 R2 直挂在延续通路需要提名候选之前就把 fact 挂好了**，于是
    `cont_supplied` 恒为 0、L4 一次都不被咨询。那不是通路坏了，是通路没轮到跑。

    这三条此前是绿的，只因为**仓库默认（关）与 live 配置（开）分叉了 20 小时**
    —— 也就是说它们一直在测一个 live 上并不存在的形态。钉死开关是为了让每条剧本
    只量它自己那一个机制；D1 那一侧的契约由
    :func:`test_d1_r2_direct_attach_is_the_intended_contract` 单独守
    （红线不许因为默认翻转而变暗）。
    """
    import os
    prev = os.environ.get("BLADEX_TASK_STATE_D1")
    os.environ["BLADEX_TASK_STATE_D1"] = "0"
    yield
    if prev is None:
        os.environ.pop("BLADEX_TASK_STATE_D1", None)
    else:
        os.environ["BLADEX_TASK_STATE_D1"] = prev

#: 上一轮：建立主题。下一轮：措辞不同的延续追问。
#:
#: 🔴 这几段文本是**量出来的，不是编出来的**：第一版随手写的追问
#: 「增资扩股那两处问题你再复核一下」实测新主体数 = 3，恰好卡在默认门槛外
#: （口径局限 C：`再复`/`核一` 这类跨词边界 bigram），阳性用例会不成立。
#: 现值实测：延续 1、换话题 19 / 23——分离得很开。改动文本必须重新量。
_TURN1_USER = "复核一下泰安仁信入股泰山啤酒的增资扩股方式，看看有没有问题"
_TURN1_REPLY = (
    "复核完成，泰安仁信的增资扩股方式存在两处问题：出资时间与工商登记不一致，"
    "另外增资扩股的股权比例与公告披露的数字对不上，需要再核一遍。"
)
_TURN2_FOLLOWUP = "增资扩股方式存在的问题，出资时间与工商登记不一致，需要再核一遍"
#: 阴性对照：完全换话题
_TURN2_NEW_TOPIC = "帮我在 macOS 上用 ollama 部署 qwen3.8 做本地推理，显存该怎么配置比较合适"


class MockEmbedder:
    """确定性 embedder：句子越像向量越像，但**故意不让姐妹轮撞上**。

    姐妹卡的 centroid 与两轮 fact 都不接近 —— 这正是四开事故的形态
    （同一对象不同动作，各轮内容差异大时向量会散开）。向量能捡回来的话，
    本项就没有存在的必要，测试也就测不到东西。
    """

    def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * 64
            for i, ch in enumerate(t[:32]):
                v[(ord(ch) + i) % 64] += 0.1
            out.append(v)
        return out

    @property
    def available(self):
        return True


class RecordingJudge:
    """一律判 none 的 L4 桩，顺带记下每次看到的候选卡 id。

    判 none = 走 L5，与"没有这道通路"时的归属结果一致 —— 本文件因此**不改变
    任何归属结论**，只观测机制有没有跑到。

    🔴 `all_seen` **不能当接线判据**：`search_matters` 是纯向量召回，测试库里
    统共一张卡，top-k 必然把它返回，于是"姐妹卡在候选里"对任何输入都成立。
    初版就是这么写的，gate 上三条阴性对照连红才发现阳性用例也是假通过。
    它现在只用来断言"L4 确实被调用过"（证明测试没有空转）。
    """

    def __init__(self) -> None:
        self.seen: list[set[str]] = []

    @property
    def model_name(self) -> str:
        return "recording-stub"

    def judge_links(self, items: list[LinkJudgeItem]) -> list[LinkJudgeResult]:
        for it in items:
            self.seen.append({c.matter_id for c in it.candidates})
        return [
            LinkJudgeResult(fact_id=it.fact_id, verdict=LinkVerdict.NONE, reason="stub")
            for it in items
        ]

    @property
    def all_seen(self) -> set[str]:
        return {mid for s in self.seen for mid in s}


def _put_turn(
    ledger: MemoryHub, idx: int, user: str, reply: str,
    *, source: SessionIdSource,
) -> str:
    identity = Identity(
        user_id="u1", agent_id="claude-code", session_id="s1",
        turn_index=idx, session_id_source=source,
    )
    turn = Turn(
        identity=identity, model="test",
        request_messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": user},
        ],
        response_text=reply, status=TurnStatus.OK,
        ts=_TS + timedelta(minutes=idx),
        # 生产在 Turn 与 Identity 两处都带（server.py:2187 从 identity 复制），
        # 测试照做——只设一处会让测试从生产不走的路径上通过。
        session_id_source=source,
    )
    key = identity.storage_key(f"{1755500000000 + idx * 1000}-0")
    ledger.put(key, turn)
    return key


def _run(
    tmpdir: str, second_user: str, *, source: SessionIdSource,
) -> tuple[RecordingJudge, dict[str, int]]:
    """两轮真实流量过 rebuild，返回 (judge, 第二轮的延续计数器)。

    🔴 两轮**故意分两次 `rebuild_from_hub`**：这正是首版丢掉的那条路径
    （consolidator 每 60s 一批，绝大多数相邻轮天生落在不同批）。写成一次调用
    会把跨批次的接线整段绕过去，测了个寂寞。
    """
    import structlog.testing

    ledger = MemoryHub(Path(tmpdir) / "rocksdb")
    ledger.open()
    judge = RecordingJudge()
    index = MemoryIndex(Path(tmpdir) / "index", embedder=MockEmbedder(),
                        read_only=False, link_judge=judge)
    index.open()
    try:
        # 第一轮先入库并归属 → 它的 fact 落到姐妹卡上（用 manual 边钉住，
        # 免得测试依赖 L4/L5 的随机行为；GM-1 读的是"上一轮 fact 所属的 Matter"，
        # 怎么归到那张卡的与本项无关）。
        from bladex_core.consolidation_proxy import _deterministic_fact_id
        from bladex_core.matter import EdgeTargetType

        index.add_matter(Matter(matter_id=_SISTER_ID, title="泰安仁信增资扩股复核",
                                status=MatterStatus.ACTIVE))
        k1 = _put_turn(ledger, 0, _TURN1_USER, _TURN1_REPLY,
                       source=SessionIdSource.FINGERPRINT)
        index.assign_manual(_SISTER_ID, EdgeTargetType.FACT,
                            _deterministic_fact_id(k1, _TURN1_USER))
        index.rebuild_from_hub(ledger)
        judge.seen.clear()          # 只看第二轮

        _put_turn(ledger, 1, second_user, "好的。", source=source)
        # 🔴 用 structlog.testing 而不是 pytest caplog：本仓日志走 structlog，
        # caplog 抓不到（`test_personal_identity_semantics` 已记过这条）。
        with structlog.testing.capture_logs() as cap:
            index.rebuild_from_hub(ledger)
        counters = next(
            (e for e in cap if e.get("event") == "index_attrib_continuation"), {})
        return judge, {k: v for k, v in counters.items() if isinstance(v, int)}
    finally:
        index.close()
        ledger.close()


# ── 🔴 前置断言：先证明夹具本身满足判据 ──────────────────────────────────────


def test_fixture_texts_actually_satisfy_the_predicate() -> None:
    """夹具自检——不通过的话，下面的阳性用例就是个空转的 no-op。

    没有这条的话，改一个字把追问推过门槛，阳性用例会**静默变成什么都没测**
    （阴性对照照样绿，因为它们本来就期望"看不见"）。仪器要先证明自己在量东西。
    """
    from bladex_core.continuation import is_task_continuation

    ref = f"{_TURN1_USER}\n{_TURN1_REPLY}"
    cont, n_follow = is_task_continuation(
        session_id_source="tail_continuation",
        current_text=_TURN2_FOLLOWUP, reference_text=ref)
    switch, n_switch = is_task_continuation(
        session_id_source="tail_continuation",
        current_text=_TURN2_NEW_TOPIC, reference_text=ref)
    assert cont is True, f"延续夹具失效（新主体数 {n_follow}）——重新量文本，别改门槛"
    assert switch is False, f"换话题夹具失效（新主体数 {n_switch}）"
    assert n_switch > n_follow * 5, "两档必须分离得开，否则阴性对照没有说服力"


# ── 🔴 主判据：延续通路把上一轮的卡提出来了（跨批次） ────────────────────────


def test_continuation_supplies_previous_matter_across_rebuild_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """「延续 ∧ 无新主体」→ 延续通路提出上一轮的 Matter，**且跨批次成立**。

    三个计数器一起断言：`cont_pairs` 说判据成立、`cont_supplied` 说通路接上了。
    不断言 `cont_candidates>0`——向量本来就召回了姐妹卡时它合法地为 0，
    拿它当接线判据会得出相反结论。
    """
    monkeypatch.setenv("BLADEX_ATTRIB_CONTINUATION", "1")
    with tempfile.TemporaryDirectory() as td:
        judge, c = _run(td, _TURN2_FOLLOWUP, source=SessionIdSource.TAIL_CONTINUATION)
        assert c, "没抓到 index_attrib_continuation —— 开关或日志名变了，先修再读结论"
        assert c["cont_pairs"] >= 1, "跨批次的延续对丢了（首版就死在这里）"
        assert c["cont_supplied"] >= 1, "判出延续但没解析到上一轮的 Matter"
        assert judge.seen, "L4 一次都没被调用——本测试测不到东西"


# ── 阴性对照（三组，缺一不可）────────────────────────────────────────────────


def test_non_continuation_supplies_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """判据①不成立（指纹会话，不是尾部接续）→ 通路一张卡都不提。

    这条守的是"别把整个 session 合成一张卡"：`session_id` 实际是跨天的指纹桶
    （MQ-S9，前十名里 5 个跨天、最宽 47 张卡）。
    """
    monkeypatch.setenv("BLADEX_ATTRIB_CONTINUATION", "1")
    with tempfile.TemporaryDirectory() as td:
        _judge, c = _run(td, _TURN2_FOLLOWUP, source=SessionIdSource.FINGERPRINT)
        assert c["cont_pairs"] == 0
        assert c["cont_supplied"] == 0


def test_new_subject_supplies_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """判据②不成立（换了话题）→ 通路一张卡都不提。

    ①几乎每轮都命中（agent 每轮重发整段对话），承重的是②。
    """
    monkeypatch.setenv("BLADEX_ATTRIB_CONTINUATION", "1")
    with tempfile.TemporaryDirectory() as td:
        _judge, c = _run(td, _TURN2_NEW_TOPIC, source=SessionIdSource.TAIL_CONTINUATION)
        assert c["cont_pairs"] == 0
        assert c["cont_supplied"] == 0


def test_flag_off_is_exact_rollback(monkeypatch: pytest.MonkeyPatch) -> None:
    """开关关 = 整条通路不存在（回滚通道必须真的能回滚）。

    判据是"连日志行都不该出现"：计数器归零可能是没触发，而**整行消失**说明
    这段代码根本没进（`_cont_enabled` 短路），才是真回滚。
    """
    monkeypatch.setenv("BLADEX_ATTRIB_CONTINUATION", "0")
    monkeypatch.setenv("BLADEX_ATTRIB_KEY_RECALL", "0")
    with tempfile.TemporaryDirectory() as td:
        _judge, c = _run(td, _TURN2_FOLLOWUP, source=SessionIdSource.TAIL_CONTINUATION)
        assert c == {}, "开关关掉后仍在跑延续判定"


# ── 🔴 只进候选、不做判据 ────────────────────────────────────────────────────


def test_candidate_only_judge_verdict_still_decides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """L4 判 none 时，fact **不得**因为"是延续"就挂到姐妹卡上。

    这是 GM 纪律 1 的可执行形态：确定性信号只进候选、判定一律交 L4。
    2026-08-18 一天内这条被撞到五次（共轮精度 50% ／ 键强匹配须送 L4 ／
    同 session 最大牵 47 张卡 ／ 按键反查泛词一键 57 卡 ／ 延续信号）。
    """
    monkeypatch.setenv("BLADEX_ATTRIB_CONTINUATION", "1")
    with tempfile.TemporaryDirectory() as td:
        ledger = MemoryHub(Path(td) / "rocksdb")
        ledger.open()
        judge = RecordingJudge()
        index = MemoryIndex(Path(td) / "index", embedder=MockEmbedder(),
                            read_only=False, link_judge=judge)
        index.open()
        try:
            from bladex_core.consolidation_proxy import _deterministic_fact_id
            from bladex_core.matter import EdgeTargetType

            index.add_matter(Matter(matter_id=_SISTER_ID, title="泰安仁信增资扩股复核",
                                    status=MatterStatus.ACTIVE))
            k1 = _put_turn(ledger, 0, _TURN1_USER, _TURN1_REPLY,
                           source=SessionIdSource.FINGERPRINT)
            index.assign_manual(_SISTER_ID, EdgeTargetType.FACT,
                                _deterministic_fact_id(k1, _TURN1_USER))
            index.rebuild_from_hub(ledger)

            k2 = _put_turn(ledger, 1, _TURN2_FOLLOWUP, "好的。",
                           source=SessionIdSource.TAIL_CONTINUATION)
            index.rebuild_from_hub(ledger)

            f2 = _deterministic_fact_id(k2, _TURN2_FOLLOWUP)
            fact = index.get_fact(f2)
            if fact is None:          # 蒸馏形态变化导致 fact_id 口径不同 = 缺数
                pytest.skip("第二轮未产出可定位的 fact，判不了（不要读成通过）")
            assert fact.matter_id != _SISTER_ID, (
                "L4 判了 none，却仍挂到姐妹卡上 —— 确定性信号变成了判据"
            )
        finally:
            index.close()
            ledger.close()


# ── 🔴 D1 侧的伴生守卫（2026-08-21，翻默认时立）─────────────────────────────


def _run_two_turns_with_d1(tmpdir: str, second_user: str,
                           source: SessionIdSource) -> tuple[str | None, dict]:
    """两轮两批，**D1 开**，返回 (第二轮 fact 的 matter_id, d1 计数器)。

    与 :func:`_run` 同形（两次 `rebuild_from_hub` = 跨批次真实路径），
    差别只有开关与读的那条日志。
    """
    import os

    import structlog.testing
    from bladex_core.consolidation_proxy import _deterministic_fact_id
    from bladex_core.matter import EdgeTargetType

    os.environ["BLADEX_TASK_STATE_D1"] = "1"
    os.environ["BLADEX_ATTRIB_CONTINUATION"] = "1"
    ledger = MemoryHub(Path(tmpdir) / "rocksdb")
    ledger.open()
    index = MemoryIndex(Path(tmpdir) / "index", embedder=MockEmbedder(),
                        read_only=False, link_judge=RecordingJudge())
    index.open()
    try:
        index.add_matter(Matter(matter_id=_SISTER_ID, title="泰安仁信增资扩股复核",
                                status=MatterStatus.ACTIVE))
        k1 = _put_turn(ledger, 0, _TURN1_USER, _TURN1_REPLY,
                       source=SessionIdSource.FINGERPRINT)
        index.assign_manual(_SISTER_ID, EdgeTargetType.FACT,
                            _deterministic_fact_id(k1, _TURN1_USER))
        index.rebuild_from_hub(ledger)

        k2 = _put_turn(ledger, 1, second_user, "好的。", source=source)
        with structlog.testing.capture_logs() as cap:
            index.rebuild_from_hub(ledger)
        counters = next(
            (e for e in cap if e.get("event") == "index_d1_decisions"), {})
        fact = index.get_fact(_deterministic_fact_id(k2, second_user))
        return (fact.matter_id if fact is not None else None,
                {k: v for k, v in counters.items() if isinstance(v, int)})
    finally:
        index.close()
        ledger.close()


def test_d1_r2_direct_attach_is_the_intended_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D1 开时，延续轮**允许**零 L4 直挂到锚卡 —— 这是 ADR-0031 的契约，不是回归。

    🔴 存在理由（2026-08-21 翻默认时立）：GM 纪律 1「确定性信号只进候选、判定
    一律交 L4」是 2026-08-18 撞五次之后立的，而 ADR-0031 D1 的 **R2 就是直挂零 L4**
    （举证责任反转：默认延续，要证明是新的事才切换）。两条已 Accepted 的决定在
    R2 这条路上正面冲突，后者胜出。

    把上面那条守卫钉成 D1=0 之后，**live 形态就没人守了**——本条补上，
    且不是同义反复：它同时钉住"允许什么"与"不许什么"，
    阴性对照见 :func:`test_d1_does_not_blanket_attach_on_topic_switch`。
    """
    monkeypatch.setenv("BLADEX_ATTRIB_CONTINUATION", "1")
    with tempfile.TemporaryDirectory() as td:
        matter_id, d1 = _run_two_turns_with_d1(
            td, _TURN2_FOLLOWUP, SessionIdSource.TAIL_CONTINUATION)
        assert d1, "没抓到 index_d1_decisions —— D1 没在跑，本条测不到东西"
        assert d1.get("R2", 0) >= 1, f"延续轮没走 R2：{d1}"
        if matter_id is None:
            pytest.skip("第二轮未产出可定位的 fact，判不了（不要读成通过）")
        assert matter_id == _SISTER_ID, (
            f"R2 判了延续却没挂到锚卡（落在 {matter_id}）—— 直挂通路没接上"
        )


def test_d1_does_not_blanket_attach_on_topic_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """🔴 阴性对照：换话题轮**不得**走 R2 直挂，且 L4 的 none 在 R5 路径上仍说了算。

    没有这一条，上面那条就退化成"D1 把什么都挂到锚卡上"也能过 —— 那正是
    举证责任反转最容易滑坡的方向。判据取 `decide_turn` 自己的条件：
    `is_task_continuation` 不成立 → R3 → **R5 需要裁决** → 落回五层管线，
    而管线里的 L4（`RecordingJudge` 恒判 none）必须能拦住它。

    夹具的分离度由 `test_fixture_texts_actually_satisfy_the_predicate` 保证
    （延续新主体 1 / 换话题 19–23），改文本要重新量。
    """
    monkeypatch.setenv("BLADEX_ATTRIB_CONTINUATION", "1")
    with tempfile.TemporaryDirectory() as td:
        matter_id, d1 = _run_two_turns_with_d1(
            td, _TURN2_NEW_TOPIC, SessionIdSource.TAIL_CONTINUATION)
        assert d1, "没抓到 index_d1_decisions —— D1 没在跑，本条测不到东西"
        assert d1.get("R2", 0) == 0, (
            f"换话题轮也走了 R2 直挂：{d1} —— 默认延续变成了无条件延续"
        )
        assert d1.get("R5", 0) >= 1, f"换话题轮没落到 R5 兜底：{d1}"
        if matter_id is None:
            pytest.skip("第二轮未产出可定位的 fact，判不了（不要读成通过）")
        assert matter_id != _SISTER_ID, (
            "R5 路径上 L4 判了 none，却仍挂到姐妹卡上 —— GM 纪律 1 在兜底路径上也失守了"
        )
