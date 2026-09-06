"""蒸馏中断守卫（2026-08-05 事故驱动）。

事故形态（C7 测试时撞上的真实数据）：上游 `ark.cn-beijing.volces.com` 在本机
DNS 解析不了，consolidator 连续多轮打出

    index_rebuild_done  consumed=30  new_facts=0  backlog=True

以 ~30 turn/分钟的速度把 440 轮积压全部标记成"已消费"且零事实，日志里没有
任何一行说这些轮次是白烧的，只能靠人工 `--unconsume --since ...` 捞回来。

根因是两件事叠加：
1. `LLMDistiller.distill` 失败时**降级返回空 DistillOutput 而不抛**（不停摆），
   于是"上游断了"与"这批本来就没事实"在 rebuild 看来完全一样；
2. `rebuild_from_hub` 在 pass 末尾**无条件** `_mark_consumed(k)`。

守卫钉的就是这两者的交叉点：本轮蒸馏失败率超阈值 → 不消费、不写 Memory Index、退避重试
（失败不进台账，重试零成本）。未达阈值也要打 warning——那些轮次的事实是真丢了。
"""

from __future__ import annotations

import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest
from bladex_core.consolidation_proxy import ProxyConsolidator
from bladex_core.distillation import DistillOutput
from bladex_proxy.models import Identity, Turn, TurnStatus
from bladex_proxy.storage.memory_index import (
    _DISTILL_OUTAGE_ABORT_RATIO_DEFAULT,
    _DISTILL_OUTAGE_ABORT_RATIO_ENV,
    MemoryIndex,
    _distill_delta,
    _distill_outage_abort_ratio,
)
from bladex_proxy.storage.memory_hub import MemoryHub


class MockEmbedder:
    def embed(self, texts):
        out = []
        for t in texts:
            vec = [0.0] * 64
            vec[hash(t) % 64] = 1.0
            out.append(vec)
        return out

    @property
    def available(self):
        return True


class _StubDistiller:
    """按 fail_ratio 决定每次调用成功还是"失败降级返回空"。

    刻意复刻真实蒸馏器的两个关键行为：失败**不抛**、失败**不写台账**，
    并维护与 `LLMDistiller.stats()` 同形状的计数器。
    """

    model_name = "stub/distiller"
    prompt_ver = "v1-test"

    def __init__(self, *, fail_every: int = 1) -> None:
        # fail_every=1 全失败；=2 隔一次失败；=0 全成功
        self._fail_every = fail_every
        self.calls = 0
        self.fails = 0

    def _run(self, text: str) -> DistillOutput:
        self.calls += 1
        if self._fail_every and self.calls % self._fail_every == 0:
            self.fails += 1
            return DistillOutput(model_name=self.model_name)  # 降级：空且不抛
        return DistillOutput(
            model_name=self.model_name,
            facts=[{"kind": "event", "item_kind": "assertion",
                    "content": f"stub fact from {text[:20]}"}],
        )

    def distill(self, user_message: str) -> DistillOutput:
        return self._run(user_message)

    def stats(self) -> dict[str, int]:
        return {"calls": self.calls, "fails": self.fails,
                "parse_fails": 0, "ledger_hits": 0}


def _put_turn(ledger: MemoryHub, idx: int) -> str:
    identity = Identity(user_id="u1", agent_id="hermes:default", session_id="s1", turn_index=idx)
    turn = Turn(
        identity=identity,
        model="test",
        request_messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user",
             "content": f"Please remember detail number {idx} about the outage guard project."},
        ],
        response_text=f"Acknowledged {idx}",
        status=TurnStatus.OK,
        ts=datetime(2026, 8, 5, 13, idx % 60, tzinfo=UTC),
    )
    key = identity.storage_key(f"17546000{idx:04d}-0")
    ledger.put(key, turn)
    return key


# ── 1. 阈值解析 ──────────────────────────────────────────────────────────────


def test_default_ratio_is_half(monkeypatch):
    monkeypatch.delenv(_DISTILL_OUTAGE_ABORT_RATIO_ENV, raising=False)
    assert _distill_outage_abort_ratio() == _DISTILL_OUTAGE_ABORT_RATIO_DEFAULT == 0.5


def test_env_override(monkeypatch):
    monkeypatch.setenv(_DISTILL_OUTAGE_ABORT_RATIO_ENV, "0.25")
    assert _distill_outage_abort_ratio() == 0.25


def test_zero_disables_guard(monkeypatch):
    """0 = 关闭守卫的回滚通道（与其它开关一致，旧行为必须可退回）。"""
    monkeypatch.setenv(_DISTILL_OUTAGE_ABORT_RATIO_ENV, "0")
    assert _distill_outage_abort_ratio() == 0.0


@pytest.mark.parametrize("raw", ["", "  ", "abc", "-0.1", "1.5"])
def test_invalid_values_fall_back_to_default(monkeypatch, raw):
    """配置笔误不得静默关掉守卫——非法值一律退回默认，而不是当 0 处理。"""
    monkeypatch.setenv(_DISTILL_OUTAGE_ABORT_RATIO_ENV, raw)
    assert _distill_outage_abort_ratio() == _DISTILL_OUTAGE_ABORT_RATIO_DEFAULT


# ── 2. 计数器增量语义 ────────────────────────────────────────────────────────


def test_delta_counts_only_this_pass():
    before = {"calls": 100, "fails": 3}
    after = {"calls": 130, "fails": 33}
    assert _distill_delta(before, after) == (30, 30)


def test_delta_counts_parse_failures_too():
    """坏 JSON 与调用失败后果相同：消费了、零事实、且都不写台账 —— 都得算进失败率。

    2026-08-05 实测：模型被 max_tokens 掐断 → `distill_parse_failed` 不计入 `fails`，
    守卫因此拦不住"批量坏 JSON"这一形态。
    """
    before = {"calls": 10, "fails": 0, "parse_fails": 0}
    after = {"calls": 40, "fails": 0, "parse_fails": 30}
    assert _distill_delta(before, after) == (30, 30)


def test_delta_sums_both_failure_kinds():
    before = {"calls": 0, "fails": 0, "parse_fails": 0}
    after = {"calls": 20, "fails": 8, "parse_fails": 4}
    assert _distill_delta(before, after) == (20, 12)


def test_delta_none_when_no_calls_moved():
    """计数器没动（例如全部命中蒸馏台账）→ 无失败信号，不该触发守卫。"""
    assert _distill_delta({"calls": 10, "fails": 0}, {"calls": 10, "fails": 0}) is None


def test_delta_none_when_stats_unavailable():
    assert _distill_delta(None, {"calls": 1, "fails": 1}) is None
    assert _distill_delta({"calls": 1, "fails": 0}, None) is None


def test_delta_none_on_counter_reset():
    """计数器倒退（重建了蒸馏器）→ 视为无信息，按旧行为走，不误判为中断。"""
    assert _distill_delta({"calls": 50, "fails": 10}, {"calls": 5, "fails": 1}) is None


# ── 3. distiller_stats() 直通 ────────────────────────────────────────────────


def test_consolidator_exposes_distiller_stats():
    stub = _StubDistiller(fail_every=1)
    consolidator = ProxyConsolidator(embedder=MockEmbedder(), distiller=stub)
    stub.distill("x")
    assert consolidator.distiller_stats() == {
        "calls": 1, "fails": 1, "parse_fails": 0, "ledger_hits": 0,
    }


def test_distiller_stats_none_without_distiller():
    assert ProxyConsolidator(embedder=MockEmbedder()).distiller_stats() is None


def test_distiller_stats_none_for_stub_without_stats():
    class _Bare:
        model_name = "bare"
        prompt_ver = "v0"

        def distill(self, user_message):
            return DistillOutput(model_name="bare")

    assert ProxyConsolidator(embedder=MockEmbedder(), distiller=_Bare()).distiller_stats() is None


# ── 4. rebuild 集成：中断时不消费 ───────────────────────────────────────────


def _rebuild_with(distiller, tmpdir: str, *, turns: int = 4):
    ledger = MemoryHub(Path(tmpdir) / "rocksdb")
    ledger.open()
    keys = [_put_turn(ledger, i) for i in range(turns)]
    index = MemoryIndex(Path(tmpdir) / "index", embedder=MockEmbedder(),
                   read_only=False, distiller=distiller)
    index.open()
    n = index.rebuild_from_hub(ledger)
    return index, ledger, keys, n


def test_total_distill_outage_marks_nothing_consumed():
    """🔴 第一红线：蒸馏全失败时，一条 turn 都不许被标记消费。

    否则这批轮次的事实永久沉默——台账不缓存失败，本来重试是零成本的。
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        index, ledger, keys, n = _rebuild_with(_StubDistiller(fail_every=1), tmpdir)
        try:
            assert n == 0
            assert index.consumed_count() == 0
            for k in keys:
                assert not index.is_consumed(k)
            # 也不该留下半截 Memory Index 数据
            assert index.all_facts() == []
            # 退避：不置 backlog，避免立即空转重试（空转正是烧积压的加速器）
            assert index.last_rebuild_backlog is False
        finally:
            index.close()
            ledger.close()


def test_outage_turns_are_picked_up_on_next_pass():
    """上游恢复后，同一批 turn 下一轮照常提炼——守卫只推迟、不丢弃。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger = MemoryHub(Path(tmpdir) / "rocksdb")
        ledger.open()
        keys = [_put_turn(ledger, i) for i in range(4)]
        broken = _StubDistiller(fail_every=1)
        index = MemoryIndex(Path(tmpdir) / "index", embedder=MockEmbedder(),
                       read_only=False, distiller=broken)
        index.open()
        assert index.rebuild_from_hub(ledger) == 0
        assert index.consumed_count() == 0

        # 上游恢复
        index._distiller = _StubDistiller(fail_every=0)
        index.rebuild_from_hub(ledger)
        assert index.consumed_count() == len(keys)
        assert index.all_facts()

        index.close()
        ledger.close()


def test_guard_disabled_restores_old_behavior(monkeypatch):
    """回滚通道：设 0 时行为逐字回到旧形态（全失败也照常消费）。"""
    monkeypatch.setenv(_DISTILL_OUTAGE_ABORT_RATIO_ENV, "0")
    with tempfile.TemporaryDirectory() as tmpdir:
        index, ledger, keys, _ = _rebuild_with(_StubDistiller(fail_every=1), tmpdir)
        try:
            assert index.consumed_count() == len(keys)
        finally:
            index.close()
            ledger.close()


def test_healthy_pass_consumes_as_before():
    """零失败 = 零行为变化（守卫不得动正常路径）。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        index, ledger, keys, _ = _rebuild_with(_StubDistiller(fail_every=0), tmpdir)
        try:
            assert index.consumed_count() == len(keys)
            assert index.all_facts()
        finally:
            index.close()
            ledger.close()


def test_partial_failure_below_threshold_still_consumes(monkeypatch):
    """低于阈值：照常消费（不因个别失败卡住管线），但事实丢失要有日志可查。"""
    monkeypatch.setenv(_DISTILL_OUTAGE_ABORT_RATIO_ENV, "0.9")
    with tempfile.TemporaryDirectory() as tmpdir:
        index, ledger, keys, _ = _rebuild_with(_StubDistiller(fail_every=2), tmpdir)
        try:
            assert index.consumed_count() == len(keys)
        finally:
            index.close()
            ledger.close()


# ── 5. 与配置模板对账（ADR-0027 §5.4 的纪律：默认值不许只活在代码里）────────


def test_env_example_declares_guard_with_matching_default():
    env_example = Path(__file__).resolve().parents[3] / "config" / ".env.example"
    text = env_example.read_text(encoding="utf-8")
    m = re.search(rf'^\s*export\s+{_DISTILL_OUTAGE_ABORT_RATIO_ENV}="?([\d.]+)"?\s*$',
                  text, re.MULTILINE)
    assert m, f"config/.env.example 未声明 {_DISTILL_OUTAGE_ABORT_RATIO_ENV}"
    assert float(m.group(1)) == _DISTILL_OUTAGE_ABORT_RATIO_DEFAULT
