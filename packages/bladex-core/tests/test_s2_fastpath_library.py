"""批 S · S2.1（MQ-I20）：库内同键邻居 —— 坐标补齐 + 保守快路径。

缺陷：LanceDB facts 表没有 subject/attribute 列，`InRunNeighborSource` 从库内视图拼出的
Fact 坐标恒空 ⇒ `fast_path_verdict` 对库内邻居从不命中同键 ⇒ 跨 pass 的同键条目一律并存
（批 S 实测 374 组 / 358 组进过同一篮子）。修法两半：① `hydrate` 用 meta 里的完整 Fact 补齐；
② 库内同键**重述 ⇒ NOOP**、**新值 ⇒ 不表态**（交裁决器），不走同批那条"强制 UPDATE"。
每条带阴性对照：拿掉 hydrate / 拿掉 library_ids 就回到旧行为。
"""

from __future__ import annotations

from bladex_core.adjudication import AdjudicationOp, fast_path_verdict
from bladex_core.consolidation_proxy import InRunNeighborSource
from bladex_core.fact import Fact, ItemKind


def _f(fid: str, content: str, *, subject: str = "BladeX", attribute: str = "状态",
       entities: list[str] | None = None) -> Fact:
    return Fact(id=fid, content=content, subject=subject, attribute=attribute,
                item_kind=ItemKind.ASSERTION, entities=entities or ["BladeX"],
                embedding=[1.0, 0.0, 0.0])


# ── ② fast_path 的库内规则 ─────────────────────────────────────────────────

def test_library_same_key_restated_is_noop() -> None:
    old = _f("f-old", "BladeX 状态：0.2.0 已发布。")
    cand = _f("f-new", "BladeX 状态：0.2.0 已发布！")
    v = fast_path_verdict(cand, [old], library_ids={"f-old"})
    assert v is not None and v.op == AdjudicationOp.NOOP and v.target_fact_ids == ["f-old"]


def test_library_same_key_new_value_defers_to_adjudicator() -> None:
    """库内同键**新值**：不强制取代（267/374 组是一键多值用法，强制取代 = 知识丢失）。"""
    old = _f("f-old", "BladeX 交付约束：必须用 .venv 跑 pytest。", attribute="交付约束",
             entities=["pytest"])
    cand = _f("f-new", "BladeX 交付约束：不许 print 调试。", attribute="交付约束",
              entities=["print"])
    assert fast_path_verdict(cand, [old], library_ids={"f-old"}) is None


def test_batch_same_key_new_value_still_forces_update() -> None:
    """阴性对照：同一批内（不在 library_ids）的同键新值 ⇒ 旧行为 UPDATE 不变。"""
    old = _f("f-old", "BladeX 交付约束：必须用 .venv 跑 pytest。", attribute="交付约束",
             entities=["pytest"])
    cand = _f("f-new", "BladeX 交付约束：不许 print 调试。", attribute="交付约束",
              entities=["print"])
    v = fast_path_verdict(cand, [old])
    assert v is not None and v.op == AdjudicationOp.UPDATE and v.target_fact_ids == ["f-old"]
    # library_ids 为空集合 = 旧签名，逐字同行为
    v2 = fast_path_verdict(cand, [old], library_ids=frozenset())
    assert v2 is not None and v2.op == AdjudicationOp.UPDATE


def test_mixed_batch_and_library_new_value_defers() -> None:
    """篮子里同键既有同批新值又有库内新值 ⇒ 只要库内那条不是重述就不表态。"""
    lib = _f("f-lib", "BladeX 交付约束：必须用 .venv 跑 pytest。", attribute="交付约束",
             entities=["pytest"])
    batch = _f("f-batch", "BladeX 交付约束：提交按文件显式 add。", attribute="交付约束",
               entities=["git"])
    cand = _f("f-new", "BladeX 交付约束：不许 print 调试。", attribute="交付约束",
              entities=["print"])
    assert fast_path_verdict(cand, [lib, batch], library_ids={"f-lib"}) is None


# ── ① InRunNeighborSource 的 hydrate ────────────────────────────────────────

class _RowChecker:
    """模拟 `_LanceDBNoveltyChecker.neighbors`：行里只有向量表那几列，没有坐标。"""

    def __init__(self, rows: list[dict]):
        self.rows = rows

    def neighbors(self, embedding, *, k: int = 10, min_similarity: float = 0.90):  # noqa: ANN001, ANN202
        return [(r, 0.95) for r in self.rows]


def test_hydrate_fills_coordinates_and_marks_library_ids() -> None:
    full = _f("f-lib", "BladeX 状态：0.2.0 已发布。")
    src = InRunNeighborSource(checker=_RowChecker([{"id": "f-lib", "content": full.content}]),
                              hydrate=lambda fid: full if fid == "f-lib" else None)
    out = src.neighbors([1.0, 0.0, 0.0])
    assert [f.id for f, _ in out] == ["f-lib"]
    assert out[0][0].subject == "BladeX" and out[0][0].attribute == "状态"
    assert src.library_ids() == frozenset({"f-lib"})
    assert src.hydrated() == {"f-lib": full}, "补齐过的对象要能进 by_id（判决改写落在同一对象上）"


def test_negative_control_without_hydrate_coordinates_stay_empty() -> None:
    """阴性对照 = 缺陷本体：没有 hydrate 时库内邻居坐标恒空 ⇒ 同键永不命中。"""
    src = InRunNeighborSource(checker=_RowChecker([{"id": "f-lib", "content": "BladeX 状态：0.2.0 已发布。"}]))
    out = src.neighbors([1.0, 0.0, 0.0])
    lib_fact = out[0][0]
    assert lib_fact.subject == "" and lib_fact.attribute == ""
    cand = _f("f-new", "BladeX 状态：0.2.0 已发布！")
    assert fast_path_verdict(cand, [lib_fact], library_ids=src.library_ids()) is None


def test_hydrate_miss_falls_back_to_row_shape() -> None:
    src = InRunNeighborSource(checker=_RowChecker([{"id": "f-gone", "content": "x"}]),
                              hydrate=lambda _fid: None)
    out = src.neighbors([1.0, 0.0, 0.0])
    assert out[0][0].id == "f-gone" and out[0][0].content == "x"
    assert "f-gone" in src.library_ids()
    assert src.hydrated() == {}, "壳对象不得进 by_id（UPDATE 会把壳写回库里抹掉原条）"
