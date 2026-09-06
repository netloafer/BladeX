"""`Matter.scope` 写侧 + 读侧过滤（ADR-0021 §2.4 补平，2026-08-16）。

背景：ADR-0021 T1 声称做了「Fact.scope / **Matter.scope** / P2 检索按可见集合过滤」，
实测 Matter 侧**读写两端都没做**——四个构造点全不传 scope（live 120/120 恒空），
三个检索 API 的**签名里根本没有可见性参数**。字段有定义、读侧按"已迁移"设计，
写侧从未落地：ADR-0026 铁律三的镜像形态（无生产者的字段被当作可用）。

🔴 **验收诚实性声明**：读侧过滤在个人模式下是**恒真的 no-op**（单一身份，
所有 scope 都匹配），live 验不了。本文件用构造数据覆盖多 principal 场景 ——
**不得拿"live 没报错"当读侧的通过依据**（那是"第七例"的温床）。
"""

from __future__ import annotations

import pytest

from bladex_core.attribution import (
    AttributionDecision,
    AttributionPipeline,
    AttributionSource,
)
from bladex_core.matter import Matter


def _decision(title: str = "泰山啤酒破产重整尽调") -> AttributionDecision:
    return AttributionDecision(
        fact_id="f-1", matter_id="", confidence=0.9,
        source=AttributionSource.PROVISIONAL,
        is_new_matter=True, new_matter_title=title, new_matter_aliases=[])


# ── 写侧：四个构造点都要落 scope ────────────────────────────────────────────


def test_create_matter_for_decision_inherits_founder_scope() -> None:
    """L5 新开卡：scope 取创始 fact。此前该参数不存在，落库恒空。"""
    m = AttributionPipeline().create_matter_for_decision(
        _decision(), centroid=None, scope="personal:local")
    assert m.scope == "personal:local"


def test_create_matter_for_decision_defaults_to_empty_not_a_guess() -> None:
    """不传就留空，**不许瞎猜一个 scope**。

    空 scope 在读侧走回落（可见），猜错则要么泄漏要么把卡藏起来——
    两种错都比"留空 + 明确回落"难查。
    """
    assert AttributionPipeline().create_matter_for_decision(_decision("t")).scope == ""


# ── 读侧：可见性判定（复用 _visibility_pids 的三态）─────────────────────────


class _Idx:
    """只借 MemoryIndex 的判定方法，不起库。"""

    from bladex_proxy.storage.memory_index import MemoryIndex

    _visibility_pids = MemoryIndex._visibility_pids
    _matter_visible = MemoryIndex._matter_visible


def _visible(scope: str, visibility=None, user_id=None) -> bool:
    ix = _Idx()
    pids = ix._visibility_pids(visibility, user_id)
    return ix._matter_visible(Matter(matter_id="m-x", title="t", scope=scope),
                              pids, visibility)


def test_empty_scope_stays_visible_migration_compat() -> None:
    """🔴 存量 120/120 都是空 scope。过滤掉 = 把整个 ②平面清零。

    这是**有意的 fail-open**，代价写在 `_matter_visible` docstring 里：
    企业部署要靠 Matter 侧隔离，必须先跑一次全量重建把 scope 回填。
    """
    assert _visible("", user_id="local") is True
    assert _visible("   ", user_id="local") is True


def test_personal_scope_matches_only_its_own_principal() -> None:
    assert _visible("personal:alice", visibility=["personal:alice"]) is True
    assert _visible("personal:bob", visibility=["personal:alice"]) is False


def test_personal_mode_empty_visibility_falls_back_to_user_id() -> None:
    """🔴 2026-07-28 事故的镜像：空 visibility 是"未标注"，不是"显式空集合"。

    当时 `if visibility is not None` 把个人模式的空列表当成后者，
    Fact 散点召回 fail-closed 锁死 13/13 轮、潜伏 6 天。Matter 侧**复用**
    同一个 `_visibility_pids`，所以这条语义只有一个实现——本测试守住它。
    """
    assert _visible("personal:local", visibility=[], user_id="local") is True
    assert _visible("personal:other", visibility=[], user_id="local") is False
    assert _visible("personal:anything", visibility=None, user_id=None) is True  # 不过滤


def test_team_scope_compares_whole_string() -> None:
    """team:/org: 没有 personal pid，整串比对。本轮无此形态，留通路。"""
    assert _visible("team:rd", visibility=["personal:a", "team:rd"]) is True
    assert _visible("team:legal", visibility=["personal:a", "team:rd"]) is False


def test_explicit_team_only_visibility_is_fail_closed_for_personal_matters() -> None:
    """显式声明只可见 team（无 personal）→ personal 卡不可见。

    与 `_visibility_pids` 表格第二行一致：`["team:x"]` → pids=[] → fail-closed。
    """
    assert _visible("personal:alice", visibility=["team:rd"]) is False


# ── 接线：参数存在 ≠ 生产会传 ───────────────────────────────────────────────


class _RecordingRetriever:
    """记录收到的可见性参数。新签名（接受 user_id/visibility）。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, object, object]] = []
        from bladex_core.matter import MatterStatus
        self._m = Matter(matter_id="m1", title="卡", summary="摘要",
                         status=MatterStatus.ACTIVE)

    def search(self, query, top_k=10, user_id=None, visibility=None, q_vec=None):
        return []

    def list_hard_rules(self):
        return []

    def search_matters_by_query(self, query, k=5, q_vec=None,
                                user_id=None, visibility=None):
        self.calls.append(("search_matters", user_id, visibility))
        return [self._m]

    def get_facts_for_matter(self, matter_id, top_k=5, user_id=None, visibility=None):
        self.calls.append(("get_facts", user_id, visibility))
        return []

    def get_matter(self, matter_id):
        return self._m if matter_id == "m1" else None


class _LegacyRetriever(_RecordingRetriever):
    """旧签名：**不接受** user_id/visibility（历史实现 / 第三方 retriever）。"""

    def search_matters_by_query(self, query, k=5):          # type: ignore[override]
        self.calls.append(("search_matters", "LEGACY", "LEGACY"))
        return [self._m]

    def get_facts_for_matter(self, matter_id, top_k=5):     # type: ignore[override]
        self.calls.append(("get_facts", "LEGACY", "LEGACY"))
        return []


def test_index_apis_accept_visibility_kwargs() -> None:
    """三个 API 的签名必须真有这两个参数（适配器的 TypeError 兜底会掩盖缺失）。"""
    import inspect

    from bladex_proxy.storage.memory_index import MemoryIndex

    for name in ("search_matters", "search_matters_by_query", "get_facts_for_matter"):
        params = inspect.signature(getattr(MemoryIndex, name)).parameters
        assert "user_id" in params, f"{name} 缺 user_id"
        assert "visibility" in params, f"{name} 缺 visibility"


def test_get_facts_for_matter_filters_before_truncation() -> None:
    """🔴 先滤后截。反过来的话，不可见的 fact 会占掉 top_k 配额，
    表现为"卡明明有内容却召回不到"——而且只在混合 user 的卡上出现，极难复现。"""
    import inspect

    from bladex_proxy.storage.memory_index import MemoryIndex

    src = inspect.getsource(MemoryIndex.get_facts_for_matter)
    assert "for e in fact_edges:" in src, "不许再对 fact_edges[:top_k] 切片后再过滤"
    assert "fact_edges[:top_k]" not in src


def test_scope_filter_does_not_reuse_admissible() -> None:
    """`_admissible` 还滤 t_invalid 与 min_importance —— 那是另一个问题。

    混进本步会打破"个人模式零回归"（可见性在个人模式恒真、那两条不是），
    也会让 G7 重建的 delta 归因不清。
    """
    import ast
    import inspect
    import textwrap

    from bladex_proxy.storage.memory_index import MemoryIndex

    src = textwrap.dedent(inspect.getsource(MemoryIndex.get_facts_for_matter))
    tree = ast.parse(src)
    # 🔴 只看**代码**，不看 docstring/注释 —— 本函数的 docstring 里就解释了
    # "为什么不复用 `_admissible`"，字符串包含检查会被自己的说明命中（首跑假红）。
    fn = tree.body[0]
    body = fn.body[1:] if (fn.body and isinstance(fn.body[0], ast.Expr)
                           and isinstance(fn.body[0].value, ast.Constant)) else fn.body
    code = "\n".join(ast.unparse(n) for n in body)
    assert "_admissible" not in code, "可见性过滤不得复用 _admissible（它还滤 t_invalid/importance）"


# ── 个人模式零回归 ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("scope", ["", "personal:local"])
def test_personal_mode_is_a_no_op(scope: str) -> None:
    """个人模式下（单一身份）过滤恒真 —— 这正是 live 验不了读侧的原因。"""
    assert _visible(scope, visibility=[], user_id="local") is True
