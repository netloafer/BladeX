"""U5.4 importance 公式 + U5.2 取代键决策（ADR-0026 §5）。纯函数，真跑。"""

from __future__ import annotations

from datetime import UTC, datetime

from bladex_core.fact import Fact, ItemKind
from bladex_core.importance import compute_importance
from bladex_core.supersede import (
    find_superseded,
    plan_supersede_merge,
    should_merge,
    supersede_key,
)

# ── U5.4 importance ──


def test_kind_baseline_new_item_equals_baseline() -> None:
    # 新条 strength=1/ref=0/age=0 → importance = kind 基线
    assert abs(compute_importance("preference") - 0.90) < 1e-9
    assert abs(compute_importance("assertion") - 0.70) < 1e-9
    assert abs(compute_importance("profile_obs") - 0.30) < 1e-9


def test_baseline_ordering() -> None:
    imp = compute_importance
    assert imp("hard_rule") > imp("preference") > imp("assertion") > imp("profile_obs")
    assert imp("lesson") > imp("file_ref")


def test_strength_and_refcount_increase_importance() -> None:
    assert compute_importance("assertion", strength=5) > compute_importance("assertion", strength=1)
    assert compute_importance("assertion", ref_count=10) > compute_importance("assertion", ref_count=0)


def test_time_half_life_decays() -> None:
    base = compute_importance("assertion")
    hl = 30 * 24 * 3600.0
    half = compute_importance("assertion", age_seconds=hl, half_life_seconds=hl)
    assert abs(half - base * 0.5) < 1e-6
    quarter = compute_importance("assertion", age_seconds=2 * hl, half_life_seconds=hl)
    assert abs(quarter - base * 0.25) < 1e-6


def test_unknown_kind_falls_to_default() -> None:
    assert abs(compute_importance("garbage") - 0.60) < 1e-9


# ── U5.2 取代键 ──


def _f(fid: str, kind: ItemKind, subj: str = "", attr: str = "", scope: str = "personal:u",
       t_invalid=None, superseded_by: str = "") -> Fact:
    f = Fact(id=fid, content=f"c-{fid}", item_kind=kind, subject=subj, attribute=attr, scope=scope)
    f.t_invalid = t_invalid
    f.superseded_by = superseded_by
    return f


def test_supersede_key_only_assertion_preference_with_slots() -> None:
    assert supersede_key(_f("a", ItemKind.ASSERTION, "用户住址", "城市")) == ("personal:u", "用户住址", "城市")
    assert supersede_key(_f("p", ItemKind.PREFERENCE, "回复", "风格")) is not None
    # 无 subject/attribute → 无键
    assert supersede_key(_f("a", ItemKind.ASSERTION, "", "")) is None
    # procedure/lesson/file_ref 无取代键（累积不互斥）
    assert supersede_key(_f("x", ItemKind.PROCEDURE, "a", "b")) is None
    assert supersede_key(_f("x", ItemKind.LESSON, "a", "b")) is None


def test_find_superseded_same_key_newest_wins() -> None:
    new = _f("f2", ItemKind.ASSERTION, "用户住址", "城市")
    old = _f("f1", ItemKind.ASSERTION, "用户住址", "城市")
    other = _f("f3", ItemKind.ASSERTION, "用户住址", "邮编")  # 异 attribute
    assert find_superseded(new, [old, other]) is old


def test_find_superseded_normalizes() -> None:
    new = _f("f2", ItemKind.ASSERTION, " 用户住址 ", "城市")
    old = _f("f1", ItemKind.ASSERTION, "用户住址", "城市")
    assert find_superseded(new, [old]) is old  # 规范化后同键


def test_find_superseded_skips_already_invalid() -> None:
    new = _f("f2", ItemKind.ASSERTION, "用户住址", "城市")
    dead = _f("f1", ItemKind.ASSERTION, "用户住址", "城市", t_invalid=datetime.now(UTC))
    assert find_superseded(new, [dead]) is None


def test_find_superseded_different_scope_no_match() -> None:
    new = _f("f2", ItemKind.ASSERTION, "用户住址", "城市", scope="personal:a")
    old = _f("f1", ItemKind.ASSERTION, "用户住址", "城市", scope="personal:b")
    assert find_superseded(new, [old]) is None  # 不同 scope 不取代


# ── U5.3 合并（同键同义重述 → strength+1，区别于取代）──


def _pref(fid: str, content: str, ents=None) -> Fact:
    f = _f(fid, ItemKind.PREFERENCE, "回复", "风格")
    f.content = content
    f.entities = ents or []
    return f


def test_should_merge_on_restatement() -> None:
    # content 规范化相等 = 重述 → 合并
    assert should_merge(_pref("n", "回复要简洁"), _pref("o", "回复要简洁 "))
    # 实体高重叠 → 合并
    assert should_merge(_pref("n", "简洁点", ["简洁", "回复"]), _pref("o", "简明扼要", ["简洁", "回复"]))


def test_should_not_merge_different_value() -> None:
    # 同键不同值（应走取代，不是合并）
    new = _f("n", ItemKind.ASSERTION, "住址", "城市")
    new.content, new.entities = "住上海", ["上海"]
    old = _f("o", ItemKind.ASSERTION, "住址", "城市")
    old.content, old.entities = "住北京", ["北京"]
    assert not should_merge(new, old)


def test_should_not_merge_no_key() -> None:
    a = _f("n", ItemKind.PROCEDURE, "a", "b")
    b = _f("o", ItemKind.PROCEDURE, "a", "b")
    assert not should_merge(a, b)


# ── U5.2/5.3 planner（取代/合并/新增 一体决策，非破坏）──


def _av(fid: str, content: str, subj: str, attr: str, ents=None) -> Fact:
    f = _f(fid, ItemKind.ASSERTION, subj, attr)
    f.content = content
    f.entities = ents or []
    return f


def test_plan_supersede_invalidates_old_and_adds_new() -> None:
    old = _av("o", "住北京", "住址", "城市", ["北京"])
    new = _av("n", "住上海", "住址", "城市", ["上海"])
    plan = plan_supersede_merge([new], [old])
    assert plan.add == [new]
    assert len(plan.invalidate) == 1 and plan.invalidate[0][0] is old and plan.invalidate[0][1] == "n"
    assert old.t_invalid is not None and old.superseded_by == "n"
    assert plan.supersede_edges == [("n", "o")]
    assert not plan.bump


def test_plan_merge_bumps_strength_drops_new() -> None:
    old = _f("o", ItemKind.PREFERENCE, "回复", "风格")
    old.content, old.entities, old.strength = "回复简洁", ["简洁"], 1
    new = _f("n", ItemKind.PREFERENCE, "回复", "风格")
    new.content, new.entities = "回复要简洁", ["简洁"]
    plan = plan_supersede_merge([new], [old])
    assert plan.add == [] and plan.bump == [(old, 2)] and old.strength == 2


def test_plan_no_key_just_adds() -> None:
    f = _f("p", ItemKind.PROCEDURE, "", "")
    plan = plan_supersede_merge([f], [])
    assert plan.add == [f] and not plan.bump and not plan.invalidate


def test_plan_within_batch_same_key_second_supersedes_first() -> None:
    a = _av("a", "住广州", "住址", "城市", ["广州"])
    b = _av("b", "住深圳", "住址", "城市", ["深圳"])
    plan = plan_supersede_merge([a, b], [])
    assert a in plan.add and b in plan.add
    assert any(o is a and nid == "b" for o, nid in plan.invalidate)


# ── 复盘补（2026-08-03，ADR-0026 §4.3）：file_ref 失效机制走取代键 ──


def test_file_ref_hash_change_supersedes():
    """同路径 hash 变 → 旧 ref 置 t_invalid（取代）；hash 同 → 合并不重复。"""
    from bladex_core.fact import Fact, ItemKind
    from bladex_core.supersede import plan_supersede_merge

    old = Fact(id="fr1", content="文件 config/routing.toml（内容 hash aaaa1111）",
               item_kind=ItemKind.FILE_REF, subject="config/routing.toml",
               attribute="file", entities=["config/routing.toml"])
    new = Fact(id="fr2", content="文件 config/routing.toml（内容 hash bbbb2222）",
               item_kind=ItemKind.FILE_REF, subject="config/routing.toml",
               attribute="file", entities=["config/routing.toml"])
    plan = plan_supersede_merge([new], [old])
    # hash 变 = 内容不同 → 取代（不是合并——entities 恒同路径，不能拿实体重叠判重述）
    assert plan.invalidate and plan.invalidate[0][0].id == "fr1"
    assert old.t_invalid is not None and old.superseded_by == "fr2"
    assert plan.add == [new] and not plan.bump

    # 对照：内容完全相同 → 合并 strength+1
    old2 = Fact(id="fr3", content="文件 a.py（内容 hash cccc3333）",
                item_kind=ItemKind.FILE_REF, subject="a.py", attribute="file",
                entities=["a.py"])
    dup = Fact(id="fr4", content="文件 a.py（内容 hash cccc3333）",
               item_kind=ItemKind.FILE_REF, subject="a.py", attribute="file",
               entities=["a.py"])
    plan2 = plan_supersede_merge([dup], [old2])
    assert plan2.bump and plan2.bump[0][1] == 2 and not plan2.add


def test_file_ref_different_paths_do_not_interfere():
    from bladex_core.fact import Fact, ItemKind
    from bladex_core.supersede import plan_supersede_merge

    a = Fact(id="fa", content="文件 x.py（内容 hash 1111）", item_kind=ItemKind.FILE_REF,
             subject="x.py", attribute="file", entities=["x.py"])
    b = Fact(id="fb", content="文件 y.py（内容 hash 2222）", item_kind=ItemKind.FILE_REF,
             subject="y.py", attribute="file", entities=["y.py"])
    plan = plan_supersede_merge([b], [a])
    assert plan.add == [b] and not plan.invalidate and not plan.bump
    assert a.t_invalid is None
