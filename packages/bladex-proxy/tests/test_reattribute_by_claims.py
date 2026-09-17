"""⑤ 归位：按 AGENT_CLAIM 只改归属、不重蒸（2026-08-26）。

🔴 **本卡存在的理由是一次验证结论**：`AGENT_CLAIM` 的读侧映射只在
`rebuild_from_hub` 构造 `ConversationTurn` 时生效 —— 也就是**只有重新蒸馏才写新归属**。
而定向重放（`--unconsume --agents …`）不清库、保留 novelty，纠正版与库内旧 Fact
内容逐字相同 ⇒ 判重命中 ⇒ `consolidation_proxy` 里直接 `continue`
（`verdict="vector_kill"`）⇒ **归属没改成，而且不报错**。

于是"事后改归属"此前只有全量重建一条路（2.5 小时 + 全库重蒸的 LLM 费用），
**为了改一个字段把整个派生层重算一遍**。

实证背景：库里 `Pi`(13) 与 `Pi-old`(6) 并存，来自 Jason 2026-08-19 的三步认领
（`3b5f→Pi` / `Pi→Pi-old` / `b1d8→Pi`）。那批 fact 归属是对的，但 `created_at`
全是 claim 之后一次**全量重建**的同一分钟——证明的是"全量重建能追溯"，
不是"定向重放能"。
"""

from __future__ import annotations

from datetime import UTC, datetime

from bladex_core.fact import Fact
from bladex_core.matter import Matter, MatterParticipant


class _FakeHub:
    """只需要 `scan_admin_events`——`build_agent_claim_map` 只用它。"""

    def __init__(self, events):
        self._events = events

    def scan_admin_events(self):
        for i, e in enumerate(self._events):
            yield (f"admin_event/{i}", e)


class _Ev:
    def __init__(self, from_ids, to_id):
        self.event_type = "agent_claim"
        self.matter_id = ""
        self.payload = {"from_agent_ids": from_ids, "to_agent_id": to_id}


class _FakeIndex:
    """只实现 `reattribute_by_claims` 用到的四个方法，避免起真 LanceDB。"""

    def __init__(self, facts, matters=()):
        self._facts = list(facts)
        self._matters = list(matters)
        self.added: list[Fact] = []
        self.updated_matters: list[Matter] = []

    def all_facts(self): return self._facts
    def all_matters(self): return self._matters
    def add_fact(self, f): self.added.append(f)
    def update_matter(self, m): self.updated_matters.append(m)

    reattribute_by_claims = None            # 下面用真实实现绑上


def _bind():
    from bladex_proxy.storage.memory_index import MemoryIndex
    _FakeIndex.reattribute_by_claims = MemoryIndex.reattribute_by_claims


def _fact(fid: str, agent: str) -> Fact:
    return Fact(id=fid, content=f"content-{fid}", agent_id=agent)


def _matter(mid: str, agents: list[str]) -> Matter:
    now = datetime.now(UTC)
    return Matter(matter_id=mid, title=mid, participants=[
        MatterParticipant(agent_id=a, first_seen=now, last_seen=now, turns=3)
        for a in agents])


class TestFactReattribution:
    def test_maps_old_agent_to_new(self):
        _bind()
        idx = _FakeIndex([_fact("f1", "unknown-9d9bdd3e"), _fact("f2", "codex")])
        out = idx.reattribute_by_claims(
            _FakeHub([_Ev(["unknown-9d9bdd3e"], "codex")]))
        assert out["facts"] == 1 and out["pairs"] == 1
        assert idx.added[0].id == "f1" and idx.added[0].agent_id == "codex"

    def test_updated_at_bumped_so_exports_see_it(self):
        """归属变更是**变更**——导出增量必须看得见（ADR-0027 §5.3 同款理由）。"""
        _bind()
        f = _fact("f1", "old")
        before = f.updated_at
        idx = _FakeIndex([f])
        idx.reattribute_by_claims(_FakeHub([_Ev(["old"], "new")]))
        assert idx.added[0].updated_at != before

    def test_chain_resolution_follows(self):
        """`a→b` 之后 `b→c`，`a` 要收敛到 `c`（与 build_agent_claim_map 一致）。"""
        _bind()
        idx = _FakeIndex([_fact("f1", "a")])
        idx.reattribute_by_claims(_FakeHub([_Ev(["a"], "b"), _Ev(["b"], "c")]))
        assert idx.added[0].agent_id == "c"

    def test_name_reuse_does_not_move_current_traffic(self):
        """🔴 实测过的刁钻场景（Jason 08-19 三步操作）：`Pi→Pi-old` 让位之后
        `b1d8→Pi` 复用这个名字 ⇒ `Pi` 不再是旧名字，**以 Pi 落库的数据不许被划走**。"""
        _bind()
        idx = _FakeIndex([_fact("f_pi", "Pi"), _fact("f_b1d8", "unknown-b1d8a6f4"),
                          _fact("f_3b5f", "unknown-3b5f685d")])
        idx.reattribute_by_claims(_FakeHub([
            _Ev(["unknown-3b5f685d"], "Pi"),
            _Ev(["Pi"], "Pi-old"),
            _Ev(["unknown-b1d8a6f4"], "Pi"),
        ]))
        got = {f.id: f.agent_id for f in idx.added}
        assert "f_pi" not in got, "现行名字 Pi 的数据被错划成 Pi-old"
        assert got["f_3b5f"] == "Pi-old"        # 链式：3b5f→Pi→Pi-old
        assert got["f_b1d8"] == "Pi"

    def test_dry_run_counts_without_writing(self):
        _bind()
        idx = _FakeIndex([_fact("f1", "old")])
        out = idx.reattribute_by_claims(_FakeHub([_Ev(["old"], "new")]), dry_run=True)
        assert out["facts"] == 1 and idx.added == []

    def test_idempotent(self):
        """跑两遍不该改第二次（第一遍之后已经没有命中映射的 fact 了）。"""
        _bind()
        f = _fact("f1", "old")
        led = _FakeHub([_Ev(["old"], "new")])
        idx = _FakeIndex([f])
        idx.reattribute_by_claims(led)
        idx2 = _FakeIndex([idx.added[0]])
        assert idx2.reattribute_by_claims(led)["facts"] == 0

    def test_no_claims_is_a_noop(self):
        _bind()
        idx = _FakeIndex([_fact("f1", "codex")])
        assert idx.reattribute_by_claims(_FakeHub([]))["facts"] == 0
        assert idx.added == []


class TestMatterParticipants:
    """🔴 Matter 卡的 participants 也按 agent_id 聚合，而 Matter 卡是**整卡注入**的
    ——只改 Fact 会让旧名字每轮出现在模型眼前。"""

    def test_participant_renamed(self):
        _bind()
        idx = _FakeIndex([], [_matter("m1", ["unknown-9d9bdd3e"])])
        out = idx.reattribute_by_claims(_FakeHub([_Ev(["unknown-9d9bdd3e"], "codex")]))
        assert out["matters"] == 1
        assert [p.agent_id for p in idx.updated_matters[0].participants] == ["codex"]

    def test_old_and_new_participants_are_merged(self):
        """同一 Matter 上"旧名 + 新名"两条要合并，否则轮次计数被拆成两份。"""
        _bind()
        idx = _FakeIndex([], [_matter("m1", ["unknown-9d9bdd3e", "codex"])])
        idx.reattribute_by_claims(_FakeHub([_Ev(["unknown-9d9bdd3e"], "codex")]))
        parts = idx.updated_matters[0].participants
        assert len(parts) == 1 and parts[0].agent_id == "codex"
        assert parts[0].turns == 6, "两条 participant 的轮次没有累加"

    def test_untouched_matter_not_rewritten(self):
        _bind()
        idx = _FakeIndex([], [_matter("m1", ["hermes:default"])])
        idx.reattribute_by_claims(_FakeHub([_Ev(["x"], "y")]))
        assert idx.updated_matters == []


def test_cli_declares_reattribute_and_warns_on_unconsume():
    """陷阱要堵在 CLI 上：`--unconsume --agents <已认领桶>` 必须留可 grep 的告警，
    因为它**不报错也不生效**——静默无效比报错更糟。"""
    import inspect

    from bladex_proxy import consolidator
    src = inspect.getsource(consolidator.main)
    assert '"--reattribute"' in src
    assert "unconsume_will_not_change_attribution" in src
    assert "vector_kill" in src           # 告警要说清为什么无效
