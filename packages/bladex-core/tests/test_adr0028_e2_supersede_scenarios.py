"""ADR-0028 E2.2：supersede / 冲突覆盖的验收剧本（机制不改，写死语义防漂移）。

这批机制的代码 2026-08-03 就写完了，只是开关从未打开（`BLADEX_SUPERSEDE_ENABLED`）。
E2.1 通电之后，真正的风险不再是"没通"，而是**语义漂移**——尤其是那个最诱人的
错误方向：「三条偏好没归到一个键上？把键匹配放宽成 cosine 不就好了」。

**本卡明令禁止**。取代键 = `(scope, NFKC+casefold(subject), norm(attribute))`
**精确匹配**，守 G1 误合并=0。蒸馏给出的 subject 不归一是 E6.5（蒸馏保真 / 语言钉死）
要解的问题，不是放宽键匹配的理由。
"""

from __future__ import annotations

from datetime import UTC, datetime

from bladex_core.fact import Fact, ItemKind
from bladex_core.supersede import plan_supersede_merge, supersede_key


def _f(fid: str, content: str, subject: str, attribute: str,
       kind: ItemKind = ItemKind.PREFERENCE, entities: list[str] | None = None) -> Fact:
    return Fact(
        id=fid, content=content, item_kind=kind,
        subject=subject, attribute=attribute,
        entities=entities if entities is not None else [],
        source_user_id="u", source_session="s",
    )


# ── 剧本① 偏好归一：三轮同义表达 → 1 条 preference，strength=3 ──────────


def test_scenario_1_preference_normalization_strength_3():
    """「回复请简洁」「别说废话」「长话短说」三轮 → 1 条，strength=3。

    前提：蒸馏把三次表达归一到同一个 subject/attribute（prompt 已要求 normalized
    subject）。这里钉的是**机制**：同键 + 同义（实体重叠）→ 合并而非新增。
    """
    existing: list[Fact] = []
    plan1 = plan_supersede_merge(
        [_f("f1", "用户希望回复简洁", "用户", "回复风格", entities=["简洁"])], existing)
    assert len(plan1.add) == 1
    existing = list(plan1.add)

    plan2 = plan_supersede_merge(
        [_f("f2", "用户不喜欢废话", "用户", "回复风格", entities=["简洁"])], existing)
    assert plan2.add == []                      # 不新增行
    assert len(plan2.bump) == 1
    old, strength = plan2.bump[0]
    assert old.id == "f1" and strength == 2

    plan3 = plan_supersede_merge(
        [_f("f3", "长话短说", "用户", "回复风格", entities=["简洁"])], existing)
    assert plan3.add == []
    assert plan3.bump[0][1] == 3

    assert existing[0].strength == 3
    assert existing[0].t_invalid is None        # 合并不失效任何东西


def test_scenario_1_note_unnormalized_subject_is_an_e65_problem():
    """🔴 反例钉死：蒸馏给出的 subject 不归一 → 三条键不同 → 三条独立记录。

    这是**预期行为**，不是 bug。处置方向是 E6.5（蒸馏保真 + 语言钉死），
    **不是**把键匹配放宽成相似度——那会直接击穿 G1 误合并=0 红线。
    """
    facts = [
        _f("f1", "用户希望回复简洁", "用户", "回复风格"),
        _f("f2", "用户不喜欢废话", "回复", "冗长度"),
        _f("f3", "长话短说", "对话", "长度偏好"),
    ]
    keys = {supersede_key(f) for f in facts}
    assert len(keys) == 3
    plan = plan_supersede_merge(facts, [])
    assert len(plan.add) == 3
    assert plan.bump == [] and plan.invalidate == []


# ── 剧本② React→Vue：同键异值 → 非破坏取代 ─────────────────────────────


def test_scenario_2_react_to_vue_supersedes_non_destructively():
    """同 subject=项目技术栈 / attribute=前端框架，值从 React 换成 Vue。

    非破坏取代：旧条 t_invalid + superseded_by，**旧条保留**（as-of 可见），
    并写一条 SUPERSEDES 边供 hop 反向替换召回。
    """
    old = _f("f-react", "项目前端框架用 React", "项目技术栈", "前端框架",
             kind=ItemKind.ASSERTION, entities=["React"])
    new = _f("f-vue", "项目前端框架改用 Vue", "项目技术栈", "前端框架",
             kind=ItemKind.ASSERTION, entities=["Vue"])

    now = datetime.now(UTC)
    plan = plan_supersede_merge([new], [old], now=now)

    assert plan.add == [new]                     # 新条入库
    assert plan.bump == []                       # 不是合并
    assert len(plan.invalidate) == 1
    invalidated, by_id = plan.invalidate[0]
    assert invalidated.id == "f-react" and by_id == "f-vue"
    # 非破坏：旧条对象仍在，只是被标记失效
    assert old.t_invalid == now
    assert old.superseded_by == "f-vue"
    assert old.content == "项目前端框架用 React"
    # SUPERSEDES 边（新 → 旧），供 hop 反向替换
    assert plan.supersede_edges == [("f-vue", "f-react")]


def test_scenario_2_already_invalidated_is_not_superseded_twice():
    """已失效的旧条不再被重复取代（幂等）。"""
    old = _f("f-react", "React", "项目技术栈", "前端框架", kind=ItemKind.ASSERTION)
    old.t_invalid = datetime.now(UTC)
    old.superseded_by = "f-vue"
    newer = _f("f-svelte", "Svelte", "项目技术栈", "前端框架", kind=ItemKind.ASSERTION)

    plan = plan_supersede_merge([newer], [old])
    assert plan.invalidate == []
    assert plan.add == [newer]


# ── 键匹配语义的护栏（禁止放宽成 cosine）────────────────────────────────


def test_supersede_key_is_exact_normalized_triple():
    """键 = (scope, NFKC+casefold(subject), norm(attribute))，精确匹配。"""
    a = _f("a", "x", "  Project  Tech-Stack ", "前端框架")
    b = _f("b", "y", "project tech-stack", " 前端框架 ")
    assert supersede_key(a) == supersede_key(b)   # 规范化后同键
    c = _f("c", "z", "project techstack", "前端框架")
    assert supersede_key(a) != supersede_key(c)   # 少个连字符就是另一个键——精确匹配


def test_different_scope_never_supersedes():
    """scope 是键的一部分：跨 scope 不互相取代（ADR-0021 可见性边界）。"""
    old = _f("f1", "React", "项目技术栈", "前端框架", kind=ItemKind.ASSERTION)
    old.scope = "personal"
    new = _f("f2", "Vue", "项目技术栈", "前端框架", kind=ItemKind.ASSERTION)
    new.scope = "team"
    plan = plan_supersede_merge([new], [old])
    assert plan.invalidate == []
    assert len(plan.add) == 1


def test_kinds_without_key_never_supersede():
    """只有 assertion/preference/file_ref 有取代键；lesson/procedure 累积不互相取代。"""
    for kind in (ItemKind.LESSON, ItemKind.PROCEDURE, ItemKind.PROFILE_OBS):
        a = _f("a", "内容 A", "同一个主语", "同一个属性", kind=kind)
        b = _f("b", "内容 B", "同一个主语", "同一个属性", kind=kind)
        assert supersede_key(a) is None, kind
        plan = plan_supersede_merge([b], [a])
        assert plan.invalidate == [] and len(plan.add) == 1, kind


def test_file_ref_hash_change_is_supersede_not_merge():
    """file_ref：同路径新 hash → 取代（失效机制入口），不是"同义重述"合并。"""
    old = _f("f-old", "文件 /a/b.md（内容 hash aaa）", "/a/b.md", "file",
             kind=ItemKind.FILE_REF, entities=["/a/b.md"])
    new = _f("f-new", "文件 /a/b.md（内容 hash bbb）", "/a/b.md", "file",
             kind=ItemKind.FILE_REF, entities=["/a/b.md"])
    plan = plan_supersede_merge([new], [old])
    assert plan.bump == []                       # 实体重叠 1.0 也不算重述
    assert len(plan.invalidate) == 1
    assert old.superseded_by == "f-new"
