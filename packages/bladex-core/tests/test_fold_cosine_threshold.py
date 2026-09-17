"""同义折叠阈值参数化（2026-08-20，读数卡 `docs/benchmarks/channel-votes-20260820.md`）。

这组测试钉三件事：

1. **缺省逐字不变**：`fold_synonyms` / `fuse_and_rerank` 不传阈值时仍用
   `FOLD_COSINE`（0.92）—— 纯函数的行为不因这次改造而变，回滚只需不接线。
2. **阈值真的接进去了**：传 0.97 时，cos 落在 [0.92, 0.97) 的那对**不再被折**。
   这条是本次修法的全部内容，没有它整改就是空的。
3. **生效值可观测**：`info["fold_cosine"]` 回传实际用的阈值 ——
   「改了没生效」此前只能靠翻配置猜（本仓多次踩过），日志里必须看得见。

🔴 阴性对照（第 4 组）：真重复（cos 0.98）在 0.97 阈值下**仍然被折**。
只测「提高阈值 → 少折」没有判别力——把阈值设成 1.1 也能过，那是把折叠关掉，
不是把它调准。两条一起才说明阈值是在**分辨**而不是在**关闭**。
"""

from __future__ import annotations

import math

import pytest
from bladex_core.fusion import (
    FOLD_COSINE,
    Channel,
    fold_synonyms,
    fuse_and_rerank,
)


class _F:
    """最小 fact 替身（fusion 只用 id / entities / item_kind / importance）。"""

    def __init__(self, fid: str, ents: list[str]) -> None:
        self.id = fid
        self.entities = ents
        self.item_kind = "assertion"
        self.importance = 0.5
        self.subject = ""
        self.attribute = ""
        self.scope = "personal:local"
        self._score = 0.0

    def __repr__(self) -> str:  # 失败信息可读
        return f"<{self.id}>"


def _unit(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _pair_with_cos(target: float) -> tuple[list[float], list[float]]:
    """构造两个夹角余弦恰为 target 的单位向量（二维即可）。"""
    a = [1.0, 0.0]
    b = _unit([target, math.sqrt(max(0.0, 1.0 - target * target))])
    return a, b


@pytest.mark.parametrize("cos", [0.93, 0.95, 0.965])
def test_gray_zone_pair_folds_at_default_but_not_at_097(cos: float) -> None:
    """灰区（0.92–0.97）的一对：默认阈值折掉，0.97 阈值保留。

    这三个取值正是 2026-08-20 人工签收里**误折**那批的实测区间（0.926–0.964）。
    """
    f1, f2 = _F("a", ["虎彩集团", "泰山啤酒"]), _F("b", ["虎彩集团", "泰山啤酒"])
    va, vb = _pair_with_cos(cos)
    vecs = {"a": va, "b": vb}

    kept_default, folded_default = fold_synonyms([f1, f2], vectors=vecs)
    assert folded_default == 1, f"cos={cos} 应被默认阈值 {FOLD_COSINE} 折掉"
    assert [f.id for f in kept_default] == ["a"]

    kept_097, folded_097 = fold_synonyms([f1, f2], vectors=vecs, cos_threshold=0.97)
    assert folded_097 == 0, f"cos={cos} 在 0.97 阈值下不该被折"
    assert [f.id for f in kept_097] == ["a", "b"]


def test_true_duplicate_still_folds_at_097() -> None:
    """🔴 阴性对照：真重复（cos 0.98，签收实测 0.972/0.979 那档）在 0.97 下仍被折。

    没有这一条，「提高阈值」与「关掉折叠」在测试上不可区分。
    """
    f1, f2 = _F("a", ["qwen3.8:27b"]), _F("b", ["qwen3.8:27b"])
    va, vb = _pair_with_cos(0.98)
    kept, folded = fold_synonyms([f1, f2], vectors={"a": va, "b": vb}, cos_threshold=0.97)
    assert folded == 1
    assert [f.id for f in kept] == ["a"]


def test_entity_jaccard_still_gates() -> None:
    """实体不重合时，即使 cos 很高也不折——第二条件的语义没被本次改动碰过。

    （它在真实库里几乎恒真、无判别力，见 `fold_synonyms` docstring；
    但"无判别力"是**数据**性质，不是把它删了的理由，这里守住它仍在生效。）
    """
    f1, f2 = _F("a", ["虎彩集团"]), _F("b", ["完全无关的实体"])
    va, vb = _pair_with_cos(0.99)
    _kept, folded = fold_synonyms([f1, f2], vectors={"a": va, "b": vb}, cos_threshold=0.97)
    assert folded == 0


def test_fuse_and_rerank_default_unchanged_and_threshold_wired() -> None:
    """`fuse_and_rerank`：不传 = 旧行为；传 0.97 = 灰区那对都活下来；info 回传生效值。"""
    f1, f2 = _F("a", ["虎彩集团", "泰山啤酒"]), _F("b", ["虎彩集团", "泰山啤酒"])
    by_id = {"a": f1, "b": f2}
    va, vb = _pair_with_cos(0.95)
    vecs = {"a": va, "b": vb}
    ch = [Channel("dense", ["a", "b"])]

    out_def, info_def = fuse_and_rerank(by_id, ch, 5, vectors=vecs)
    assert info_def["folded"] == 1
    assert info_def["fold_cosine"] == FOLD_COSINE
    assert [f.id for f in out_def] == ["a"]

    out_097, info_097 = fuse_and_rerank(by_id, ch, 5, vectors=vecs, fold_cosine=0.97)
    assert info_097["folded"] == 0
    assert info_097["fold_cosine"] == 0.97
    assert {f.id for f in out_097} == {"a", "b"}


def test_flag_default_is_092_after_ab_rollback(monkeypatch: pytest.MonkeyPatch) -> None:
    """flag 默认值 = **0.92**（整表对账在 `test_flag_defaults.py`，这里只钉本项）。

    曾短暂为 0.97（08-20 读数驱动），08-21 共享库 A/B 终验 0.92→72.0% /
    0.97→68.0% 后回退——−4pp 不显著（McNemar p=0.727）但也无证据支持改动，
    按「无证据不改默认」处置。**机制不回滚**：上面几组测试仍钉 `cos_threshold`
    参数化本身，设 `BLADEX_FOLD_COSINE=0.97` 即复现实验档。

    🔴 必须先 `delenv`：本机 `config/.env` 里也有这一项，不隔离的话这条测的是
    「用户今天配了什么」而不是「默认值是什么」（本仓已因此红过一次）。
    """
    from bladex_core.flags import flag_number

    monkeypatch.delenv("BLADEX_FOLD_COSINE", raising=False)
    assert flag_number("BLADEX_FOLD_COSINE") == pytest.approx(0.92)
