"""V-L5 账本锚层**生产接线**（ADR-0032 §4.5）。

core 侧纯函数在 `test_ledger_anchor.py`；本文件只管"生产循环真的按锚走"。

🔴 **为什么这个文件必须存在**：MQ-L14 —— 锚层的查找键（session）与绑定键
（`activation_scope`）不同构，对 live 5390 轮**命中 0**，而 `ledger_anchor_ready`
照常打，日志上一切正常。潜伏原因就是**开启态没有一条测试**（机制默认关，
测试只覆盖了纯函数）。同型教训：08-22「默认与 live 分叉期间，测试在测一个
不存在的形态」。默认翻开之前，这里必须一直有阳性 + 阴性对照。

钉住：
  ① 阳性：有账本的轮次 → 边 `decision.layer=ANCHOR`，Matter id = 账本派生；
  ② 阴性：无账本的轮次 → 不产 ANCHOR 边，走 D1/五层兜底（锚层不越界）；
  ③ 点时（MQ-L15）：一个 agent 先后切两本账本 ⇒ **两张卡**，不是一张；
  ④ 建卡之前的轮次不被塞进锚定 Matter（反向误合并）；
  ⑤ 开关关 = 回归通道：零 ANCHOR 边。
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from bladex_core.distillation import DistillFact, DistillOutput, MatterProposal
from bladex_core.ledger import EVENT_LEDGER_SWITCH
from bladex_core.ledger_runtime import ledger_anchor_matter_id
from bladex_proxy.models import (
    AdminEventType,
    Identity,
    SessionIdSource,
    Turn,
    TurnStatus,
)
from bladex_proxy.storage.memory_hub import MemoryHub
from bladex_proxy.storage.memory_index import MemoryIndex

_L1 = "ldg-1111aaaa1111"
_L2 = "ldg-2222bbbb2222"


def _now_ms() -> int:
    """🔴 turn key 的 stream ms 必须与 `AdminEvent.ts` 同一墙钟。

    `append_admin_event` 用 `datetime.now(UTC)` 打时间戳（生产就是这样），
    所以测试里的 turn 时间**不能写死常量**——常量与 now 的先后关系会随
    "今天是哪天"翻转，测试时灵时不灵。用 now±offset 表达先后。
    """
    return int(datetime.now(UTC).timestamp() * 1000)


class StubDistiller:
    @property
    def model_name(self) -> str:
        return "anchor-stub"

    @property
    def prompt_ver(self) -> str:
        return "anchor-stub-001"

    def distill(self, user_message: str) -> DistillOutput:
        return DistillOutput(
            facts=[DistillFact(content=f"结论：{user_message[:40]}", kind="event",
                               entities=["锚点"])],
            matter_proposals=[MatterProposal(title="提案标题", entities=["锚点"])],
            model_name="anchor-stub",
        )


class MockEmbedder:
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


def _put_turn(ledger: MemoryHub, seq: int, agent: str, user: str,
              *, ms_offset: int = 0) -> str:
    """写一轮 turn。key 尾段 ms = **现在** + `ms_offset`。

    🔴 时间轴只有一条：`AdminEvent.ts` 是真 now，turn 的 stream ms 也必须是真 now。
    早期版本把 turn 排在 `now + seq*60s`，结果**所有 turn 都排在所有 switch 之后**，
    "先后两本账本"的剧本就不成立了（判别力靠"注入 MQ-L15 看它红不红"才发现）。
    `ms_offset` 只用来表达"建账本很久之前"这种刻意的远距离。
    """
    identity = Identity(user_id="u1", agent_id=agent, session_id=f"s{seq}",
                        turn_index=seq,
                        session_id_source=SessionIdSource.FINGERPRINT)
    turn = Turn(identity=identity, model="test",
                request_messages=[{"role": "user", "content": user}],
                response_text="好的。", status=TurnStatus.OK,
                ts=datetime.now(UTC) + timedelta(milliseconds=ms_offset),
                session_id_source=SessionIdSource.FINGERPRINT)
    key = identity.storage_key(f"{_now_ms() + ms_offset}-0")
    ledger.put(key, turn)
    _tick()
    return key


def _switch(ledger: MemoryHub, agent: str, lid: str, seq: int) -> None:
    """写一条 LEDGER_SWITCH 管理事件（proxy 侧 `_emit` 的等价形态）。"""
    ledger.append_admin_event(
        AdminEventType.LEDGER_SWITCH, "",
        agent_id=agent, project_id="", session_id=f"s{seq}", to_ledger_id=lid)
    _tick()


def _tick() -> None:
    """让墙钟真的走一步——事件与轮次靠 ms 分先后，同毫秒会让 bisect 取错边。"""
    time.sleep(0.02)


def _open(tmp_path: Path):
    ledger = MemoryHub(tmp_path / "rocksdb")
    ledger.open()
    index = MemoryIndex(tmp_path / "index", embedder=MockEmbedder(),
                        read_only=False, distiller=StubDistiller())
    index.open()
    return ledger, index


def _anchor_edges(index: MemoryIndex) -> list:
    return [e for m in index.all_matters() for e in index.get_edges(m.matter_id)
            if (e.decision or {}).get("layer") == "ANCHOR"]


def test_anchored_turn_lands_on_ledger_matter(tmp_path, monkeypatch) -> None:
    """① 阳性对照：有账本的轮次归到账本派生的 Matter。"""
    monkeypatch.setenv("BLADEX_LEDGER_ANCHOR", "1")
    ledger, index = _open(tmp_path)
    try:
        _switch(ledger, "claude-code", _L1, 0)
        _put_turn(ledger, 1, "claude-code", "帮我把路由层的回归问题修掉，静态配置优先那条没生效，先复现再动手")
        index.rebuild_from_hub(ledger)

        edges = _anchor_edges(index)
        assert edges, "锚层没产出 ANCHOR 边——生产接线没通（MQ-L14 的形状）"
        assert {e.matter_id for e in edges} == {ledger_anchor_matter_id(_L1)}
    finally:
        index.close()
        ledger.close()


def test_turn_without_ledger_falls_through(tmp_path, monkeypatch) -> None:
    """② 阴性对照：无账本流量不进锚层，照走 D1/五层兜底。"""
    monkeypatch.setenv("BLADEX_LEDGER_ANCHOR", "1")
    ledger, index = _open(tmp_path)
    try:
        _switch(ledger, "claude-code", _L1, 0)
        _put_turn(ledger, 1, "codex", "另一个 agent 发来的请求，它从来没有建过任何账本，应该走兜底归属路径")
        index.rebuild_from_hub(ledger)

        assert not _anchor_edges(index), "锚层越界：给没有账本的 agent 也挂了锚"
        assert index.all_matters(), "兜底路径也没归属——那是另一个 bug"
    finally:
        index.close()
        ledger.close()


def test_two_ledgers_in_sequence_give_two_matters(tmp_path, monkeypatch) -> None:
    """③ MQ-L15 点时查找：先后两本账本 ⇒ 两张卡。

    终态绑定会把两段都归给第二本 —— live 实测那是 3622 轮拖进一张卡的形状。

    🔴 **两轮必须在同一次 rebuild 里**：分两次 rebuild 时，第一次跑的时候
    终态绑定恰好还是第一本，错法也能蒙对 —— 判别力会消失（本测试初版就是
    这个形状，靠"注入 MQ-L15 看它红不红"才发现）。
    """
    monkeypatch.setenv("BLADEX_LEDGER_ANCHOR", "1")
    ledger, index = _open(tmp_path)
    try:
        _switch(ledger, "claude-code", _L1, 0)
        _put_turn(ledger, 1, "claude-code", "第一件事：修掉路由层的回归问题，静态配置优先那条规则没有生效，先复现")
        _switch(ledger, "claude-code", _L2, 2)
        _put_turn(ledger, 3, "claude-code", "第二件事：写 0.1.0 的发布说明，把账本机制和记忆注入两条主线讲清楚")
        index.rebuild_from_hub(ledger)

        got = {e.matter_id for e in _anchor_edges(index)}
        assert got == {ledger_anchor_matter_id(_L1), ledger_anchor_matter_id(_L2)}, got
    finally:
        index.close()
        ledger.close()


def test_turns_before_first_switch_are_not_anchored(tmp_path, monkeypatch) -> None:
    """④ 建账本之前的历史不得被塞进锚定 Matter（反向误合并）。"""
    monkeypatch.setenv("BLADEX_LEDGER_ANCHOR", "1")
    ledger, index = _open(tmp_path)
    try:
        _put_turn(ledger, 0, "claude-code",
                  "建账本之前问的是完全不相干的事：泰山啤酒破产重整的最新进展怎么样了",
                  ms_offset=-3_600_000)   # 一小时前
        _switch(ledger, "claude-code", _L1, 1)
        _put_turn(ledger, 2, "claude-code", "建账本之后的一轮，开始处理路由层的回归问题，先把复现步骤整理出来")
        index.rebuild_from_hub(ledger)

        anchor_mid = ledger_anchor_matter_id(_L1)
        assert {e.matter_id for e in _anchor_edges(index)} == {anchor_mid}
        # 旧轮的 fact 不得出现在锚定卡里 —— 它应该走兜底另开
        anchored_facts = {e.target_key for e in index.get_edges(anchor_mid)}
        old = [f for f in index.all_facts() if "泰山啤酒" in f.content]
        assert old, "旧轮没被蒸出来，这条测试就失去了判别力"
        assert not (anchored_facts & {f.id for f in old})
    finally:
        index.close()
        ledger.close()


def test_flag_off_is_regression_channel(tmp_path, monkeypatch) -> None:
    """⑤ 开关关 = 零 ANCHOR 边（默认形态逐字不变）。"""
    monkeypatch.setenv("BLADEX_LEDGER_ANCHOR", "0")
    ledger, index = _open(tmp_path)
    try:
        _switch(ledger, "claude-code", _L1, 0)
        _put_turn(ledger, 1, "claude-code", "开关关闭的时候不该产生任何锚定边，这一轮走的应该是原来的兜底通道")
        index.rebuild_from_hub(ledger)
        assert not _anchor_edges(index)
    finally:
        index.close()
        ledger.close()


# ── 账本准入闸（ADR-0032 §4.5，2026-09-02）────────────────────────────────
#
# ④ 只钉住 ANCHOR 层不锚建账本前的轮次，而**病灶在 DPL 层**：那些轮次照样被
# CONT/L2/L3/L4 按内容相似补进锚卡。④ 之所以没抓到，是它的旧轮用了「泰山啤酒」
# 这种与账本正题无关的内容 —— DPL 本来就不会链，**测试通过靠的是题材不同，
# 不是机制正确**。下面三条把题材换成"同题"，让内容相似真的发力。
#
# live 形状：07-29 的 ADR-0024 开发过程被吸进 08-28 建的「评估 v5 架构」账本卡
# （96 条新增边里 CONT 66 / L2 29），主体名同为 bladex/adr（Jason 判读 2026-09-02）。


def _all_facts_on(index: MemoryIndex, mid: str) -> set[str]:
    """落在这张卡上的 fact id —— **不分层**。

    🔴 用 `_anchor_edges` 那种按 layer 过滤的读法看不见本闸要管的东西：
    被拦的边就是 CONT/L2/L3/L4，恰好都不是 ANCHOR。
    """
    return {e.target_key for e in index.get_edges(mid)}


_GATE_TITLE = "评估 v5 架构开发进展"


class _TitleEchoDistiller:
    """提案标题恒等于账本标题——复现 live 的 L2/L3 命中形状。

    🔴 **为什么不能用 `StubDistiller`**（首版的假绿，2026-09-02 当场撞）：
    它的提案标题是常量「提案标题」，与账本标题永不相等 ⇒ L2/L3 原生键匹配
    一次也不会发生。于是"闸①拦住旧轮"的断言**空过成绿**——不是闸生效，
    是这个夹具里本来就没有边会落上去。
    判别力对照（`test_gate_off_reproduces_the_defect`）就是为抓这个写的。

    live 形状：被误吸的 fact 提案标题正是「评估 v5 架构开发进展」，
    与账本同名 ⇒ 命中锚卡的原生键（`ledger_anchor_aliases` = 只有标题）。
    """

    model_name = "gate-stub"
    prompt_ver = "gate-stub-001"

    def distill(self, user_message: str) -> DistillOutput:
        return DistillOutput(
            facts=[DistillFact(content=user_message[:60], kind="event",
                               entities=["v5 架构"])],
            matter_proposals=[MatterProposal(title=_GATE_TITLE,
                                             entities=["v5 架构"])],
            model_name="gate-stub",
        )


def _create_ledger(hub: MemoryHub, lid: str, title: str, *,
                   ms_offset: int = 0) -> None:
    """写一条 LEDGER_CREATE 事件，让账本**真的进池**（带 `created_at`）。

    🔴 光有 `_switch` 不够：`_la_pool` 由 `replay_ledger_events` 从
    CREATE/UPDATE 事件重建，只发 SWITCH 时 `ledgers=0` ⇒ 账本闸拿不到
    `created_at` ⇒ 闸①恒不表态（缺数放行）。首版就是这个形态，
    日志里 `ledger_anchor_ready ... ledgers=0` 是它的可 grep 痕迹。
    """
    from bladex_core.ledger import new_ledger
    at = (datetime.now(UTC) + timedelta(milliseconds=ms_offset)).isoformat()
    led = new_ledger(ledger_id=lid, title=title, created_at=at)
    hub.append_admin_event(AdminEventType.LEDGER_CREATE, "",
                           ledger=led.model_dump())
    _tick()


def _open_gate(tmp_path: Path):
    """与 `_open` 同构，只换蒸馏桩（见 `_TitleEchoDistiller`）。"""
    hub = MemoryHub(tmp_path / "rocksdb")
    hub.open()
    index = MemoryIndex(tmp_path / "index", embedder=MockEmbedder(),
                        read_only=False, distiller=_TitleEchoDistiller())
    index.open()
    return hub, index


def _same_topic_before_creation(tmp_path, monkeypatch, *, gate: str):
    """剧本：同题旧轮（建账本一小时前）+ 建账本 + 同题新轮。返回旧 fact 是否落锚卡。"""
    monkeypatch.setenv("BLADEX_LEDGER_ANCHOR", "1")
    monkeypatch.setenv("BLADEX_LEDGER_GATE", gate)
    ledger, index = _open_gate(tmp_path)
    try:
        # 两轮内容必须**足够不同**，否则 novelty 判重把它们折成一条 fact，
        # "旧轮那条"就不存在了（首版两句只差五个字，实测 new_facts=1）。
        _put_turn(ledger, 0, "claude-code",
                  "账本机制这条线现在做到哪一步了，切换防抖和陈旧兜底都验过没有",
                  ms_offset=-3_600_000)
        _create_ledger(ledger, _L1, _GATE_TITLE)
        _switch(ledger, "claude-code", _L1, 1)
        _put_turn(ledger, 2, "claude-code",
                  "记忆注入面砍到只剩硬规则之后，工具调用率的基线读数是多少")
        index.rebuild_from_hub(ledger)

        old = [f for f in index.all_facts() if "账本机制" in f.content]
        assert old, "旧轮没被蒸出来，本剧本失去判别力"
        return bool({f.id for f in old} & _all_facts_on(
            index, ledger_anchor_matter_id(_L1)))
    finally:
        index.close()
        ledger.close()


def test_gate_blocks_same_topic_turns_older_than_the_ledger(
        tmp_path, monkeypatch) -> None:
    """闸①：同题但早于账本创建的轮次，**任何层**都不得落到锚卡上。

    这是 ④ 抓不到的那半 —— ④ 的旧轮题材无关，DPL 本来就不链。
    """
    assert not _same_topic_before_creation(tmp_path, monkeypatch, gate="1"), (
        "🔴 建账本之前的同题轮次落进了锚卡 —— 闸①没生效。"
        "live 形状：07-29 的 ADR-0024 开发过程进 08-28 的「评估 v5 架构」卡。")


def test_gate_off_reproduces_the_defect(tmp_path, monkeypatch) -> None:
    """判别力对照：关掉闸 ⇒ 同一剧本里旧轮**会**落进锚卡（09-02 之前的行为）。

    没有这条，上一条测的可能只是"这个剧本本来就不会链"，与闸是否生效无关
    —— 那正是 ④ 的问题。

    ⚠️ 若本条转红（关掉闸也不落），说明剧本已不能复现病灶（embedder 换了、
    候选阈值变了…）：那时该修的是**剧本**，不是把上一条一起删掉。
    """
    assert _same_topic_before_creation(tmp_path, monkeypatch, gate="0"), (
        "关掉闸也没复现病灶 —— 剧本失去判别力，见 docstring")


def test_gate_blocks_turns_labelled_to_another_ledger(tmp_path, monkeypatch) -> None:
    """闸②：时间完全合规，但这一轮明确属于**别的**账本 ⇒ 不得落 A 的锚卡。

    判别力：与闸①正交 —— 两轮都在各自账本创建之后，闸①对二者都放行。
    两本账本**同名**（同题材），所以内容判定会想把两边的 fact 互相链过去；
    拦住它的只能是标签。
    """
    monkeypatch.setenv("BLADEX_LEDGER_ANCHOR", "1")
    monkeypatch.setenv("BLADEX_LEDGER_GATE", "1")
    ledger, index = _open_gate(tmp_path)
    try:
        _create_ledger(ledger, _L1, _GATE_TITLE)
        _switch(ledger, "claude-code", _L1, 0)
        _put_turn(ledger, 1, "claude-code",
                  "账本机制这条线现在做到哪一步了，切换防抖和陈旧兜底都验过没有")
        _create_ledger(ledger, _L2, _GATE_TITLE)
        _switch(ledger, "claude-code", _L2, 2)
        _put_turn(ledger, 3, "claude-code",
                  "记忆注入面砍到只剩硬规则之后，工具调用率的基线读数是多少")
        index.rebuild_from_hub(ledger)

        a, b = ledger_anchor_matter_id(_L1), ledger_anchor_matter_id(_L2)
        assert not (_all_facts_on(index, a) & _all_facts_on(index, b)), (
            "🔴 同一条 fact 同时落在两本账本的锚卡上 —— 闸②没生效")
    finally:
        index.close()
        ledger.close()


def test_gate_does_not_touch_non_anchor_matters(tmp_path, monkeypatch) -> None:
    """零回归：没有任何账本的流量，开闸与关闸结果**逐字相同**。

    本闸只对 `Matter.ledger_id` 非空的卡表态；普通内容卡的归属一步不动。
    """
    def _run(gate: str) -> set[frozenset[str]]:
        """返回**按 Matter 分组的 fact 内容**——不是 id。

        🔴 首版比 `(fact_id, matter_id)` 恒红：两者都从 turn key 的墙钟 ms 派生
        （`fact_id = f(ledger_key, content)`、`matter_id = f(首轮 ledger_key)`），
        两次 `_run` 相隔约 100ms ⇒ id 必然不同，而结构完全一样
        （实测两边都是 3 条边挂 1 张卡，`gate_*` 读数全 0＝闸一次没开火）。
        零回归要钉的是"同样的 fact 分到同样的组"，id 只是它的时变表示。
        """
        monkeypatch.setenv("BLADEX_LEDGER_ANCHOR", "1")
        monkeypatch.setenv("BLADEX_LEDGER_GATE", gate)
        d = tmp_path / f"nogate-{gate}"
        ledger, index = _open_gate(d)
        try:
            for i, msg in enumerate([
                "泰山啤酒破产重整的最新进展如何，债权人会议开过了吗",
                "泰山啤酒破产重整里那家投资人的接盘方案有没有公开细节",
                "苹果新款 Mac mini 和 Mac Studio 的亮点与性价比对比一下",
            ]):
                _put_turn(ledger, i, "codex", msg)
            index.rebuild_from_hub(ledger)
            groups: set[frozenset[str]] = set()
            for m in index.all_matters():
                facts = [index.get_fact(e.target_key)
                         for e in index.get_edges(m.matter_id)]
                got = frozenset(f.content for f in facts if f is not None)
                if got:
                    groups.add(got)
            return groups
        finally:
            index.close()
            ledger.close()

    on, off = _run("1"), _run("0")
    assert on, "两边都没归属 —— 夹具前提破了，本条零回归测不到东西"
    assert on == off, f"无账本流量的归属被闸改变了：开{on} 关{off}"


def test_switch_event_type_is_the_one_replay_reads(tmp_path) -> None:
    """接线前提：`AdminEventType.LEDGER_SWITCH` 的字面值 = core 侧重放认的那个。

    两边靠字符串对齐，改名会静默断开（MQ-L14 同型：两处各自看起来都对）。
    """
    assert AdminEventType.LEDGER_SWITCH.value == EVENT_LEDGER_SWITCH
