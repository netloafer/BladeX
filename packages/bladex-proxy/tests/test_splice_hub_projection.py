"""V-P4 硬点 1：拼接台账 = Hub 投影（ADR-0032 §3.2）。

2026-08-27 勘察发现的缺口：`SpliceRecord` 只活在进程内存里 —— `Turn` schema
没有槽位、`server.py` 零出现、`SpliceLedger.restore()` **生产零调用方**，
而它的 docstring 写着「随 turn 入 Hub」「重启不致失忆」。

**文档声称已实现、实际没有**，比刚性原则 12 说的"把缺陷写进文档"更糟一档：
那至少还留了一句警告，这里留的是一句错误的承诺。

本文件钉三件事：
1. 本轮新增的条目**落进 Turn**（且只落新增，不落全量快照——见 `new_in_turn`）
2. 重启后能从 Hub **懒恢复**，且 age 按"它之后还有几轮"重算
3. 恢复失败/无 Hub = **优雅降级**，不阻断这一轮（ADR §3.2：拼接状态丢失时
   会话仍合法，agent 侧历史本就自洽）
"""
from __future__ import annotations

import pytest
from bladex_proxy.agency import AgencyRuntime
from bladex_proxy.models import Turn
from bladex_proxy.splice import SpliceLedger, SpliceRecord


@pytest.fixture(autouse=True)
def _interception_on(monkeypatch):
    """`interception_on` 是 env 派生的只读属性 —— 只能从开关那头开。"""
    monkeypatch.setenv("BLADEX_MODULE_INTERCEPTION", "1")


def _rec(anchor: str, *, session: str = "u/a/s/", age: int = 0) -> SpliceRecord:
    return SpliceRecord(
        session_prefix=session, anchor=anchor, age_turns=age,
        calls=[{"id": f"c-{anchor}", "type": "function",
                "function": {"name": "bladex_memory_search", "arguments": "{}"}}],
        results=[{"role": "tool", "tool_call_id": f"c-{anchor}", "content": "hit"}])


class _FakeTurn:
    def __init__(self, records):
        self.splice_records = records


class _FakeHub:
    """只实现 `scan_prefix`——`_restore_splice` 只用这一个方法。"""

    def __init__(self, turns_by_prefix: dict[str, list]):
        self._t = turns_by_prefix
        self.calls = 0

    def scan_prefix(self, prefix: str):
        self.calls += 1
        return [(f"{prefix}{i}", t) for i, t in enumerate(self._t.get(prefix, []))]

    def scan_admin_events(self):
        return iter(())          # 账本池装载走这个，与本卡无关


class _BoomHub:
    def scan_prefix(self, prefix: str):
        raise RuntimeError("hub unavailable")

    def scan_admin_events(self):
        return iter(())


class TestNewInTurnOnly:
    """只落本轮新增 —— 落全量 = O(n²)，ADR-0008 §2.2 那个坑。"""

    def test_only_age_zero_returned(self):
        led = SpliceLedger()
        led.add(_rec("a1"))
        assert len(led.new_in_turn("u/a/s/")) == 1
        led.tick_and_prune("u/a/s/")          # age 0 -> 1
        assert led.new_in_turn("u/a/s/") == []
        led.add(_rec("a2"))
        got = led.new_in_turn("u/a/s/")
        assert [r.anchor for r in got] == ["a2"], "老条目不该被重复落库"

    def test_turn_schema_has_slot_and_defaults_empty(self):
        from bladex_proxy.models import Identity
        t = Turn(identity=Identity(user_id="u", agent_id="a", session_id="s"))
        assert t.splice_records == []
        # 阴性对照：没有拼接的轮次不写内容（绝大多数轮次是这种）
        assert Turn.model_fields["splice_records"].default_factory() == []

    def test_roundtrips_through_pydantic(self):
        """落库=序列化，条目必须能原样还原（calls/results 是 dict 列表）。"""
        from bladex_proxy.models import Identity
        t = Turn(identity=Identity(user_id="u", agent_id="a", session_id="s"),
                 splice_records=[_rec("a1")])
        back = Turn.model_validate(t.model_dump())
        assert back.splice_records[0].anchor == "a1"
        assert back.splice_records[0].calls[0]["function"]["name"] == "bladex_memory_search"
        assert back.splice_records[0].results[0]["content"] == "hit"


class TestLazyRestore:
    def _runtime(self, hub):
        return AgencyRuntime(hub=hub)

    def test_restores_records_from_hub(self):
        hub = _FakeHub({"u/a/s/": [_FakeTurn([_rec("a1")]), _FakeTurn([])]})
        ag = self._runtime(hub)
        got = ag._restore_splice("u/a/s/")
        assert [r.anchor for r in got] == ["a1"]
        assert ag.splice.records("u/a/s/"), "恢复的条目必须进内存表"

    def test_age_recomputed_by_position(self):
        """🔴 age 必须按"它之后还有几轮"重算。

        沿用写入时的 0 = 老化窗口从恢复那一刻重新计时 = 永不老化，
        等于把一个可配的机制悄悄关掉。
        """
        hub = _FakeHub({"u/a/s/": [
            _FakeTurn([_rec("old")]),      # 之后还有 2 轮
            _FakeTurn([]),
            _FakeTurn([_rec("new")]),      # 之后 0 轮
        ]})
        ag = self._runtime(hub)
        got = {r.anchor: r.age_turns for r in ag._restore_splice("u/a/s/")}
        assert got == {"old": 2, "new": 0}

    def test_same_anchor_last_write_wins(self):
        hub = _FakeHub({"u/a/s/": [_FakeTurn([_rec("dup")]), _FakeTurn([_rec("dup")])]})
        ag = self._runtime(hub)
        got = ag._restore_splice("u/a/s/")
        assert len(got) == 1, "同锚拼两次 = 重复注入"
        assert got[0].age_turns == 0, "保留的应是较新那条"

    def test_scans_hub_once_per_session(self):
        """🔴 热路径护栏：没有拼接条目的会话是绝大多数。

        不记"已尝试"就会每一轮都扫一次 Hub —— 200ms 预算下这是实打实的开销。
        """
        hub = _FakeHub({"u/a/s/": []})
        ag = self._runtime(hub)
        for _ in range(5):
            ag._restore_splice("u/a/s/")
        assert hub.calls == 1, f"扫了 {hub.calls} 次，应只扫 1 次"

    def test_prepare_inbound_triggers_restore(self):
        """接线断言：懒恢复必须真的挂在生产路径上。

        `restore()` 此前正是"实现了但没人调"——只测函数不测接线，
        这个缺陷会原封不动地再来一次。
        """
        hub = _FakeHub({"u/a/s/": [_FakeTurn([_rec("a1")])]})
        ag = self._runtime(hub)
        ag.prepare_inbound([{"role": "user", "content": "hi"}], "u/a/s/")
        assert hub.calls == 1, "prepare_inbound 没有触发懒恢复"


class TestGracefulDegradation:
    """恢复失败不得阻断本轮 —— ADR §3.2：拼接状态丢失时会话仍合法。"""

    def test_hub_error_is_swallowed(self):
        ag = AgencyRuntime(hub=_BoomHub())
        assert ag._restore_splice("u/a/s/") == []

    def test_no_hub_returns_empty(self):
        ag = AgencyRuntime(hub=None)
        assert ag._restore_splice("u/a/s/") == []

    def test_prepare_inbound_survives_hub_error(self):
        ag = AgencyRuntime(hub=_BoomHub())
        msgs = [{"role": "user", "content": "hi"}]
        assert ag.prepare_inbound(list(msgs), "u/a/s/") == msgs


class TestAgingKnob:
    """V-P4 老化参数：ADR §10-2 拍板"不预定默认值、真实流量标定"。

    所以判据不是"值合不合理"，而是**旋钮存不存在、默认行为变没变**。
    """

    def test_flag_registered_and_defaults_to_disabled(self):
        from bladex_core.flags import MEMORY_NUMERIC_DEFAULTS, flag_number
        assert "BLADEX_SPLICE_MAX_AGE_TURNS" in MEMORY_NUMERIC_DEFAULTS
        assert int(flag_number("BLADEX_SPLICE_MAX_AGE_TURNS")) == 0, "默认必须不老化"

    def test_runtime_reads_the_flag(self, monkeypatch):
        monkeypatch.setenv("BLADEX_SPLICE_MAX_AGE_TURNS", "3")
        ag = AgencyRuntime()
        assert ag.splice._max_age == 3, "构造时没读 flag = 旋钮转了没用"

    def test_env_example_documents_it(self):
        """与 `.env.example` 对账（ADR-0027 §5.4 同款纪律）。"""
        from pathlib import Path
        root = Path(__file__).resolve().parents[3]
        text = (root / "config" / ".env.example").read_text(encoding="utf-8")
        assert "BLADEX_SPLICE_MAX_AGE_TURNS" in text, (
            "默认值不在模板里 = 用户看不见这个旋钮（ADR-0027 §5.4 的教训）")


@pytest.mark.parametrize("interception", [True, False])
def test_disabled_interception_is_untouched(interception, monkeypatch):
    """回归通道：拦截关掉时 `prepare_inbound` 逐字不变。"""
    monkeypatch.setenv("BLADEX_MODULE_INTERCEPTION", "1" if interception else "0")
    ag = AgencyRuntime(hub=_FakeHub({"u/a/s/": [_FakeTurn([_rec("a1")])]}))
    msgs = [{"role": "user", "content": "hi"}]
    out = ag.prepare_inbound(list(msgs), "u/a/s/")
    if not interception:
        assert out == msgs
