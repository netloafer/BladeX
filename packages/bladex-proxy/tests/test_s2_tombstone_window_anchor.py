"""批 S · S2.1（MQ-I19）生产接线：账本墓碑**不倒填**——墓碑前窗口的轮次仍锚到该账本锚卡。

与 `test_ledger_anchor_wiring.py` 同款夹具（真 Hub + 真 Index + StubDistiller + MockEmbedder）。
钉住：
  ① 墓碑前的两轮 ⇒ 全量重建后仍落在 L 的锚卡上（`ledger_id=L`，标题来自池而非 id）；
  ② 墓碑后的一轮 ⇒ 不锚（scope 无账本，走兜底）；
  ③ 没有窗口 fact 的墓碑本 ⇒ **不**预物化锚卡（拍板 0.3.0-(2)：任务收摊不等于给它留一张空卡）；
  ④ 默认装载（agency / flash 用的）池里仍没有被墓碑账本；
  ⑤ 阴性对照：把墓碑时刻打成未知（0.0）⇒ 回到旧语义，① 的两轮**不再**锚——证明是新语义在起作用。
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

from bladex_core.distillation import DistillFact, DistillOutput, MatterProposal
from bladex_core.ledger import new_ledger, replay_ledger_events
from bladex_core.ledger_runtime import ledger_anchor_matter_id
from bladex_proxy import ledger_events as _le
from bladex_proxy.models import (
    AdminEventType,
    Identity,
    SessionIdSource,
    TombstoneSource,
    TombstoneTargetType,
    Turn,
    TurnStatus,
)
from bladex_proxy.storage.memory_hub import MemoryHub
from bladex_proxy.storage.memory_index import MemoryIndex

_L = "ldg-s2window0001"
_IDLE = "ldg-s2idle000002"
_TITLE = "账本历史端点与 CLI（S2 窗口）"


class StubDistiller:
    @property
    def model_name(self) -> str:
        return "s2-stub"

    @property
    def prompt_ver(self) -> str:
        return "s2-stub-001"

    def distill(self, user_message: str) -> DistillOutput:
        return DistillOutput(
            facts=[DistillFact(content=f"结论：{user_message[:40]}", kind="event",
                               entities=["ledger-history"])],
            matter_proposals=[MatterProposal(title="ledger history 端点开发", entities=["ledger-history"])],
            model_name="s2-stub")


class MockEmbedder:
    def embed(self, texts):  # noqa: ANN001, ANN201
        out = []
        for t in texts:
            v = [0.0] * 64
            for i, ch in enumerate(t[:32]):
                v[(ord(ch) + i) % 64] += 0.1
            out.append(v)
        return out

    @property
    def available(self) -> bool:
        return True


def _now_ms() -> int:
    return int(datetime.now(UTC).timestamp() * 1000)


def _tick() -> None:
    time.sleep(0.02)


def _put_turn(hub: MemoryHub, seq: int, text: str) -> str:
    ident = Identity(user_id="u1", agent_id="claude-code", session_id="s1", turn_index=seq,
                     session_id_source=SessionIdSource.FINGERPRINT)
    turn = Turn(identity=ident, model="test", request_messages=[{"role": "user", "content": text}],
                response_text="好的。", status=TurnStatus.OK, ts=datetime.now(UTC),
                session_id_source=SessionIdSource.FINGERPRINT)
    key = ident.storage_key(f"{_now_ms()}-0")
    hub.put(key, turn)
    _tick()
    return key


def _open(tmp_path: Path):
    hub = MemoryHub(tmp_path / "hub")
    hub.open()
    index = MemoryIndex(tmp_path / "index", embedder=MockEmbedder(), read_only=False,
                        distiller=StubDistiller())
    index.open()
    return hub, index


def _seed(hub: MemoryHub) -> tuple[list[str], str]:
    """建 L + 一本从未有轮次的 IDLE，切到 L，两轮，墓碑 L，再一轮。返回 (墓碑前两轮 key, 墓碑后 key)。"""
    for lid, title in ((_L, _TITLE), (_IDLE, "从未有窗口的账本")):
        led = new_ledger(ledger_id=lid, title=title, goal="g")
        led.created_at = datetime.now(UTC).isoformat()
        hub.append_admin_event(AdminEventType.LEDGER_CREATE, "", ledger=led.model_dump(mode="json"))
        _tick()
    hub.append_admin_event(AdminEventType.LEDGER_SWITCH, "", agent_id="claude-code",
                           project_id="", session_id="s1", to_ledger_id=_L)
    _tick()
    before = [_put_turn(hub, 1, "实现 GET /admin/ledgers/{id}/history 端点，返回 rev 时间线"),
              _put_turn(hub, 2, "加 CLI 子命令 bladex ledger history，把同一份数据打成表")]
    hub.append_tombstone(target_type=TombstoneTargetType.LEDGER, target_key=_L,
                         source=TombstoneSource.USER)
    hub.append_tombstone(target_type=TombstoneTargetType.LEDGER, target_key=_IDLE,
                         source=TombstoneSource.USER)
    _tick()
    after = _put_turn(hub, 3, "墓碑之后的一轮：端点与 CLI 各补一条测试，账本不存在返回 404")
    return before, after


def _facts_by_turn(index: MemoryIndex) -> dict[str, list]:
    out: dict[str, list] = {}
    for f in index.all_facts():
        out.setdefault(f.source_ledger_key, []).append(f)
    return out


def test_window_before_tombstone_anchors_after_tombstone_does_not(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("BLADEX_LEDGER_ANCHOR", "1")
    hub, index = _open(tmp_path)
    try:
        before, after = _seed(hub)
        index.rebuild_from_hub(hub, full=True)
        matters = {m.matter_id: m for m in index.all_matters()}
        anchor = matters.get(ledger_anchor_matter_id(_L))
        assert anchor is not None and anchor.ledger_id == _L, "① 墓碑前窗口的锚卡必须存在"
        assert anchor.title == _TITLE, "标题来自池（keep_tombstoned_pool），不是拿 id 当标题"
        by_turn = _facts_by_turn(index)
        for k in before:
            assert by_turn.get(k), "墓碑前轮次没蒸出 fact，测试失去判别力"
            assert all(f.matter_id == anchor.matter_id for f in by_turn[k]), "① 墓碑前 fact 必须锚到 L"
        assert by_turn.get(after), "墓碑后轮次没蒸出 fact"
        assert all(f.matter_id != anchor.matter_id for f in by_turn[after]), "② 墓碑后 fact 不得锚到 L"
        assert all(not matters[f.matter_id].ledger_id for f in by_turn[after] if f.matter_id in matters)
        assert ledger_anchor_matter_id(_IDLE) not in matters, "③ 无窗口 fact 的墓碑本不预物化"
    finally:
        index.close()
        hub.close()


def test_default_loader_pool_still_excludes_tombstoned(tmp_path) -> None:
    hub, _index = _open(tmp_path)
    try:
        _seed(hub)
        pool, bindings = replay_ledger_events(_le.load_ledger_events(hub))
        assert _L not in pool and _IDLE not in pool, "④ agency/flash 的池不含墓碑本"
        assert not bindings, "墓碑那一刻仍激活 ⇒ 绑定被合成的空切换 pop 掉"
        pool2, _ = replay_ledger_events(_le.load_ledger_events(hub, keep_tombstoned_pool=True))
        assert _L in pool2 and pool2[_L].title == _TITLE, "重建专用装载保留标题"
    finally:
        _index.close()
        hub.close()


def test_negative_control_unknown_tombstone_time_restores_old_semantics(tmp_path, monkeypatch) -> None:
    """⑤ 把墓碑时刻打成未知 ⇒ 旧语义（倒填）⇒ 墓碑前两轮也不锚。红了才说明 ① 靠的是新语义。"""
    monkeypatch.setenv("BLADEX_LEDGER_ANCHOR", "1")
    real = _le.tombstoned_ledgers
    monkeypatch.setattr(_le, "tombstoned_ledgers", lambda hub: dict.fromkeys(real(hub), 0.0))
    hub, index = _open(tmp_path)
    try:
        before, _after = _seed(hub)
        index.rebuild_from_hub(hub, full=True)
        matters = {m.matter_id: m for m in index.all_matters()}
        by_turn = _facts_by_turn(index)
        for k in before:
            assert all(not matters[f.matter_id].ledger_id for f in by_turn[k] if f.matter_id in matters), \
                "旧语义下墓碑前 fact 应落内容卡——若这里锚了，说明判别力丢了"
        assert ledger_anchor_matter_id(_L) not in matters
    finally:
        index.close()
        hub.close()
