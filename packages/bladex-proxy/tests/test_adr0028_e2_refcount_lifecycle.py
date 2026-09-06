"""ADR-0028 E2.3 / E2.4：ref_count 注入命中回写 + 生命周期重算 job。

E2.3 修的是一个"有消费者、没有生产者"的闭环缺口：
    `compute_importance` 的引用增益项 `1 + 0.25·ln(1+ref_count)` 一直在算，
    但 ref_count 全库恒 0 —— 因为唯一的写入点 `record_injection_hits` 由 **proxy**
    调用，而 proxy 的 Memory Index 是 read_only（写权只有 consolidator，ADR-0009 §7）→ 恒 no-op。
    新路径：命中 id 随 `Turn.decision_meta.injected_fact_ids` 落 Memory Hub，
    consolidator 消费该 turn 时聚合回写。零新写路径。

E2.4 修的是时间半衰减从未真正生效：
    `0.5^(age/half_life)` 只在写入那一刻算过一次，一年前的 fact 的 importance
    与它刚写入时逐字相同。空闲轮全量重算让 age 项随时间真的走起来。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from bladex_core.fact import Fact, ItemKind
from bladex_proxy.storage.memory_index import MemoryIndex


@pytest.fixture()
def index(tmp_path):
    store = MemoryIndex(tmp_path / "index")
    store.open()
    yield store


def _fact(fid: str, *, kind: ItemKind = ItemKind.ASSERTION,
          age_days: float = 0.0, strength: int = 1) -> Fact:
    t = datetime.now(UTC) - timedelta(days=age_days)
    f = Fact(id=fid, content=f"内容 {fid}", item_kind=kind,
             source_user_id="u", source_session="s", strength=strength)
    f.created_at = t
    f.t_observed = t
    from bladex_core.importance import compute_importance
    # 模拟"写入那一刻算一次"（age=0）——正是 E2.4 要修的静态快照
    f.importance = compute_importance(kind.value, strength=strength, ref_count=0,
                                      age_seconds=0.0)
    return f


# ── E2.3 ref_count 聚合回写 ────────────────────────────────────────────


def test_injection_hits_bump_refcount_and_importance(index):
    f = _fact("f1")
    index.add_fact(f)
    before = index.get_fact("f1").importance

    updated = index._apply_injection_hits({"f1": 3})  # noqa: SLF001

    assert updated == 1
    after = index.get_fact("f1")
    assert after.ref_count == 3
    assert after.importance > before          # 引用增益真的进了分数
    assert after.updated_at is not None       # 导出增量游标看得见（ADR-0027 §5.3）


def test_injection_hits_accumulate_across_passes(index):
    index.add_fact(_fact("f1"))
    index._apply_injection_hits({"f1": 2})  # noqa: SLF001
    index._apply_injection_hits({"f1": 1})  # noqa: SLF001
    assert index.get_fact("f1").ref_count == 3


def test_injection_hits_ignore_hard_rules_and_missing(index):
    """硬规则不是 Memory Index 条目；不存在的 id 静默跳过（不崩）。"""
    index.add_fact(_fact("f1"))
    updated = index._apply_injection_hits(  # noqa: SLF001
        {"hard_0": 5, "nonexistent": 2, "f1": 1})
    assert updated == 1
    assert index.get_fact("f1").ref_count == 1


def test_injection_hits_noop_on_readonly(tmp_path):
    """读端（proxy）永不写 Memory Index —— 写权只有 consolidator（架构不变）。"""
    w = MemoryIndex(tmp_path / "index")
    w.open()
    w.add_fact(_fact("f1"))
    w.close()

    r = MemoryIndex(tmp_path / "index", read_only=True)
    r.open()
    assert r._apply_injection_hits({"f1": 9}) == 0  # noqa: SLF001


def test_decision_meta_carries_injected_fact_ids():
    """Memory Hub schema 侧：字段存在、默认空、可序列化往返。"""
    from bladex_proxy.models import DecisionMeta

    dm = DecisionMeta()
    assert dm.injected_fact_ids == []
    dm2 = DecisionMeta.model_validate(
        {**dm.model_dump(), "injected_fact_ids": ["fact_a", "fact_b"]})
    assert dm2.injected_fact_ids == ["fact_a", "fact_b"]


# （`_capture_injected_ids` 两条随 2026-09-03 S1 删除：生产者已删，注入命中反馈环
#  永久断线——`injected_fact_ids` 键保留恒空，0.3.0 重建检索时连环一起重建。）


# ── E2.4 生命周期重算 job ──────────────────────────────────────────────


def test_lifecycle_recompute_applies_time_decay(index):
    """90 天前的 fact：importance 应按 0.5^(90/30) 降到约原值的 1/8。"""
    old = _fact("f-old", age_days=90)
    frozen = old.importance            # 写入那一刻算的（age=0）
    index.add_fact(old)

    updated = index.recompute_lifecycle(force=True)

    assert updated == 1
    after = index.get_fact("f-old").importance
    assert after < frozen
    assert after == pytest.approx(frozen * 0.125, rel=0.02)


def test_lifecycle_recompute_skips_unchanged(index):
    """刚写入的 fact 重算后几乎无变化 → |Δ|≤0.01 不写回（省写放大）。"""
    index.add_fact(_fact("f-new", age_days=0))
    assert index.recompute_lifecycle(force=True) == 0


def test_lifecycle_recompute_throttled_by_interval(index, monkeypatch):
    """节流：间隔内重复调用返回 -1（未执行）。"""
    index.add_fact(_fact("f-old", age_days=90))
    monkeypatch.setenv("BLADEX_LIFECYCLE_INTERVAL_S", "3600")

    assert index.recompute_lifecycle(force=True) == 1   # 首次（force 忽略节流并落时间戳）
    assert index.recompute_lifecycle() == -1            # 6h 内不再跑


def test_lifecycle_recompute_never_deletes(index):
    """遗忘 = 退出召回，不删数据（G6 重建等价性）。"""
    index.add_fact(_fact("f-ancient", age_days=3650))
    index.recompute_lifecycle(force=True)

    survivor = index.get_fact("f-ancient")
    assert survivor is not None                       # 条目还在
    assert survivor.importance < 0.01                 # 但已远低于召回门槛
    assert survivor.t_invalid is None                 # 也没被标记失效


def test_lifecycle_recompute_noop_on_readonly(tmp_path):
    w = MemoryIndex(tmp_path / "index")
    w.open()
    w.add_fact(_fact("f1", age_days=90))
    w.close()

    r = MemoryIndex(tmp_path / "index", read_only=True)
    r.open()
    assert r.recompute_lifecycle(force=True) == 0


def test_min_importance_gate_is_read_side_only(index, monkeypatch):
    """`BLADEX_MIN_IMPORTANCE` 是**读侧**门槛：低分条目退出召回，meta 仍可点查。

    M0-8（复核 G3）：默认值已回 0（三时钟 M3 落地前不做重要性驱逐——importance 的
    三个动态输入全部断粮时开门槛 = 按年龄无差别遗忘）。所以这里显式设一个非零门槛
    来考察**机制本身**，而不是把默认值写死在断言里——两件事必须分开测，
    否则默认值一变就误以为机制坏了。
    """
    from bladex_core.flags import flag_number

    monkeypatch.delenv("BLADEX_MIN_IMPORTANCE", raising=False)
    assert flag_number("BLADEX_MIN_IMPORTANCE") == 0.0   # M0-8：默认不驱逐

    monkeypatch.setenv("BLADEX_MIN_IMPORTANCE", "0.2")
    faded = _fact("f-faded", age_days=3650)
    index.add_fact(faded)
    index.recompute_lifecycle(force=True)

    got = index.get_fact("f-faded")
    assert got is not None and got.importance < flag_number("BLADEX_MIN_IMPORTANCE")
