"""MQ-S38：注入命中回写里的**悬空 id 必须可见**（不改行为，只让丢失有痕迹）。

## 病灶

`fact_id = sha256(source_ledger_key + ":" + content)[:12]`（内容派生，
`consolidation_proxy._deterministic_fact_id`）。它对**同一 turn 同一正文**是稳定的
——普通重建不换 id，这正是当初为了保住手动归属映射而设计的。

但它对**正文变化**不稳定：蒸馏 prompt 一换版（v6→v7 那种九款合一），全库重蒸
⇒ 全库 id 换代 ⇒ 历史 turn 里记的 `decision_meta.injected_fact_ids` **整批指空**。
2026-08-20 live 实测：**385/402 = 95.8% 悬空**，按天看台阶落在换版当天。

## 为什么必须有这组测试

`_apply_injection_hits` 原本是 `if fact is None: continue` —— **静默丢弃**。
后果不是"少写几条"，而是三个下游同时失去历史信号且**分不清两种情况**：

    ref_count 的引用增益 / 相关钟 last_hit_at / dormant 判定

"这条从没被命中过" 与 "命中记录在换版时丢了" 在库里长得一模一样。
本仓已为这个形态付过五次（FUNNEL_HIT 零埋点 / open_issues 零生产者 /
ref_count 曾恒 0 / 五开关默认关 / L0 没接线）。

**本次只做可见**（数出来 + warning + 进重建汇总行）；id 迁移/重映射属修法，
涉及重建等价性，按范围冻结走 0.2 候选。
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from bladex_core.fact import Fact, ItemKind
from bladex_proxy.storage.memory_index import MemoryIndex


@pytest.fixture()
def index(tmp_path):
    store = MemoryIndex(tmp_path / "index")
    store.open()
    yield store


def _fact(fid: str, *, subject: str = "", attribute: str = "") -> Fact:
    f = Fact(id=fid, content=f"内容 {fid}", item_kind=ItemKind.ASSERTION,
             source_user_id="u", source_session="s",
             subject=subject, attribute=attribute)
    f.created_at = f.t_observed = datetime.now(UTC)
    return f


# ── 悬空 id 被数出来 ──────────────────────────────────────────────────────


def test_orphan_ids_are_counted_not_silently_skipped(index):
    index.add_fact(_fact("alive"))

    updated = index._apply_injection_hits(  # noqa: SLF001
        {"alive": 2, "fact_deadbeef01": 3, "fact_deadbeef02": 1})

    assert updated == 1                                   # 只有一条真落库
    orphan_ids, orphan_hits = index.last_injection_orphans
    assert orphan_ids == 2
    assert orphan_hits == 4        # 丢掉的是 **命中次数**，不只是 id 个数


def test_clean_run_reports_zero_orphans(index):
    """全命中时必须显式是 0——"没有这一行"不能当成"没有丢"。"""
    index.add_fact(_fact("a"))
    index.add_fact(_fact("b"))
    index._apply_injection_hits({"a": 1, "b": 1})  # noqa: SLF001
    assert index.last_injection_orphans == (0, 0)


def test_hard_rule_ids_are_not_counted_as_orphans(index):
    """硬规则不是 Memory Index 条目（`inject.py:389` 已滤），不该算进丢失。

    不区分的话，每一轮都会报几条"悬空"，真正的换版丢失就淹在常态噪声里。
    """
    index.add_fact(_fact("a"))
    index._apply_injection_hits({"a": 1, "hard_rule_x": 5})  # noqa: SLF001
    assert index.last_injection_orphans == (0, 0)


def test_orphans_do_not_change_the_write_path(index):
    """可见性改动**不改行为**：能写的照写、写入值与没有悬空时逐字相同。"""
    index.add_fact(_fact("a"))
    index._apply_injection_hits({"a": 2})  # noqa: SLF001
    only = index.get_fact("a").ref_count

    index2_fact = _fact("a2")
    index.add_fact(index2_fact)
    index._apply_injection_hits({"a2": 2, "fact_missing0001": 9})  # noqa: SLF001
    assert index.get_fact("a2").ref_count == only


# ── 修法：按取代键接续（Jason 2026-08-20 破例进 0.1.0）────────────────────


def _key_of(f: Fact) -> str:
    from bladex_core.supersede import supersede_key_str
    return supersede_key_str(f)


def test_hit_is_rescued_by_supersede_key_after_id_churn(index):
    """换版剧本：旧 id 已不存在，同一取代键的**新世代条目**接住这次命中。

    这正是 v6→v7 那次的形状——正文重蒸、id 换代，而 (scope, subject, attribute)
    没变。不接续的话，`ref_count` 与相关钟在每次换版归零且无痕。
    """
    new_gen = _fact("fact_newgen01", subject="用户开发机", attribute="型号")
    index.add_fact(new_gen)
    old_id = "fact_oldgen99"          # 旧世代 id，库里已经没有了

    updated = index._apply_injection_hits(  # noqa: SLF001
        {old_id: 2}, None, {old_id: _key_of(new_gen)})

    assert updated == 1
    assert index.get_fact("fact_newgen01").ref_count == 2
    assert index.last_injection_rescued == 1
    assert index.last_injection_orphans == (0, 0)


def test_ambiguous_key_is_not_guessed(index):
    """同键多条 live（MQ-S33 实测 92 键 / 217 条）时**不猜**。

    接给谁都是猜，而猜错的方向是把命中记到错误条目上——比丢了更坏
    （它会污染排序信号，且再也看不出来）。故记为 ambiguous + orphan。
    """
    a = _fact("fact_dup_a", subject="用户开发机", attribute="型号")
    b = _fact("fact_dup_b", subject="用户开发机", attribute="型号")
    index.add_fact(a)
    index.add_fact(b)

    updated = index._apply_injection_hits(  # noqa: SLF001
        {"fact_gone": 3}, None, {"fact_gone": _key_of(a)})

    assert updated == 0
    assert index.last_injection_ambiguous == 1
    assert index.last_injection_orphans == (1, 3)
    assert index.get_fact("fact_dup_a").ref_count == 0     # 谁都没被记上


def test_invalidated_facts_do_not_absorb_rescued_hits(index):
    """已被取代的条目不参与接续——它不进召回，接了也只是把信号埋进死条。"""
    dead = _fact("fact_dead", subject="用户开发机", attribute="型号")
    dead.t_invalid = datetime.now(UTC)
    live = _fact("fact_live", subject="用户开发机", attribute="型号")
    index.add_fact(dead)
    index.add_fact(live)

    index._apply_injection_hits(  # noqa: SLF001
        {"fact_gone": 1}, None, {"fact_gone": _key_of(live)})

    assert index.get_fact("fact_live").ref_count == 1
    assert index.last_injection_ambiguous == 0    # dead 不算进候选 ⇒ 唯一匹配成立


def test_keyless_kinds_still_lost_and_counted(index):
    """lesson / procedure 没有取代键（累积、不互相取代）⇒ 换版后仍会丢。

    **不假装救回**：这条测试钉的是"读数诚实"，不是"功能完整"。
    """
    updated = index._apply_injection_hits({"fact_gone": 4}, None, {})  # noqa: SLF001
    assert updated == 0
    assert index.last_injection_orphans == (1, 4)
    assert index.last_injection_rescued == 0


def test_historical_turns_without_keys_behave_exactly_as_before(index):
    """历史 turn 没有 `injected_fact_keys` ⇒ 走"只按 id"，与改造前逐字一致。"""
    index.add_fact(_fact("a"))
    before = index._apply_injection_hits({"a": 1})  # noqa: SLF001（不传 keys）
    assert before == 1
    assert index.get_fact("a").ref_count == 1


# ── Matter 侧同款 ─────────────────────────────────────────────────────────


def test_matter_hit_orphans_are_counted(index):
    """`matter_id` 在重建/合并后同样会换代（MQ-S30），卡侧也要数。"""
    ts = datetime.now(UTC).isoformat()
    index._apply_matter_injection_hits({"m-doesnotexist": ts})  # noqa: SLF001
    assert index.last_matter_hit_orphans == 1


def test_matter_hit_orphans_reset_between_runs(index):
    """计数是**本次**的读数，不是累计——累计会让"这次没丢"永远显示成丢过。"""
    ts = datetime.now(UTC).isoformat()
    index._apply_matter_injection_hits({"m-x": ts})  # noqa: SLF001
    assert index.last_matter_hit_orphans == 1
    from bladex_core.matter import Matter
    m = Matter(matter_id="m-real", title="真卡")
    index.add_matter(m)
    index._apply_matter_injection_hits({"m-real": ts})  # noqa: SLF001
    assert index.last_matter_hit_orphans == 0
