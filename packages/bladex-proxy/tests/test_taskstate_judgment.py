"""G12.2 R5 新开/延续判定台账（TaskStateJudgment，ADR-0031 §14.1 / 决策表 v2 §6）。

钉住的语义：
  ① 三种 verdict（new / continue_to / degraded_new）都能写读——**否定判决与降级
    判决必须入账**（只记 new 的话旋钮一标定重放就漂 id）；封闭枚举，写别的抛；
  ② key 前缀 `judgment_taskstate/` 与 L4/M2 的 `judgment/` **互不可见**（MS-17
    命名空间共用事故的根治半边），且 verdict 形态不被 L4 白名单吞（第二道保险）；
  ③ 重放语义 = 台账优先：get 取最大 seq；无记录 → None（调用方才现场判）；
  ④ clear() 不清本台账（清了 = 首轮集合随旋钮漂 = §0.1b 重演）；
  ⑤ rebuild_id 未给时由 index 当前 run 盖章（MS-13 纪律）。
"""

from __future__ import annotations

import pytest
from bladex_proxy.models import TASKSTATE_VERDICTS, TaskStateJudgment
from bladex_proxy.storage.memory_index import MemoryIndex

TURN = "local/hermes:default/fp:abc/1755650000123-0"


@pytest.fixture()
def idx(tmp_path):
    ix = MemoryIndex(tmp_path / "index", embedder=None, read_only=False)
    ix.open()
    yield ix
    ix.close()


def _rec(verdict: str = "new", **kw) -> TaskStateJudgment:
    base = dict(
        turn_key=TURN, unit_key="u" * 16, verdict=verdict, matter_id="m-aaa111bbb222",
        anchor_matter_id="m-anchor000001", anchor_alive=True,
        window_snapshot=["m-anchor000001", "m-aaa111bbb222"],
        signals={"s1_new_subjects": 7}, knob_ver="d1-knob-001",
    )
    base.update(kw)
    return TaskStateJudgment(**base)


# ── ① 三种 verdict 全入账；封闭枚举 ──────────────────────────────────────


@pytest.mark.parametrize("verdict", TASKSTATE_VERDICTS)
def test_all_three_verdicts_roundtrip(idx, verdict) -> None:
    idx.append_taskstate_judgment(_rec(verdict, judge_model="" if verdict == "degraded_new" else "m"))
    got = idx.get_taskstate_judgment(TURN)
    assert got is not None and got.verdict == verdict
    assert got.window_snapshot == ["m-anchor000001", "m-aaa111bbb222"]
    assert got.anchor_matter_id == "m-anchor000001"


def test_l4_verdict_shapes_are_rejected(idx) -> None:
    """把 L4 的 verdict 写进本台账 = 重演 MS-17 的第一步，构造上拒绝。"""
    for bad in ("link:m-x", "none", "uncertain", ""):
        with pytest.raises(ValueError):
            idx.append_taskstate_judgment(_rec(bad))


# ── ② 与 judgment/ 命名空间互不可见 ──────────────────────────────────────


def test_namespaces_are_mutually_invisible(idx) -> None:
    idx.append_taskstate_judgment(_rec())
    idx.append_judgment("fact_x", [], "link:m-b", "judge-m")
    # L4 读侧扫不到 taskstate；taskstate 读侧扫不到 L4
    assert all(not k.startswith("judgment_taskstate/")
               for k, _ in idx.scan_judgments())
    assert all(k.startswith("judgment_taskstate/")
               for k, _ in idx.scan_taskstate_judgments())
    assert len(idx.scan_taskstate_judgments()) == 1


def test_taskstate_verdicts_dodge_l4_whitelist() -> None:
    """第二道保险：即便有人把记录混进 `judgment/`，L4 白名单也认不出这三个值。"""
    from bladex_proxy.linking import _is_link_verdict
    for v in TASKSTATE_VERDICTS:
        assert not _is_link_verdict(v), f"{v} 被 L4 白名单吞了——两个台账要串味"


# ── ③ 台账优先：最新 seq；无记录 = None ──────────────────────────────────


def test_latest_seq_wins_and_missing_is_none(idx) -> None:
    assert idx.get_taskstate_judgment(TURN) is None  # 无记录 → 调用方现场判
    idx.append_taskstate_judgment(_rec("new"))
    idx.append_taskstate_judgment(_rec("continue_to", matter_id="m-later9999999"))
    got = idx.get_taskstate_judgment(TURN)
    assert got.verdict == "continue_to" and got.matter_id == "m-later9999999"
    # 其它轮不串
    assert idx.get_taskstate_judgment(TURN + "-other") is None


# ── ④ clear() 不清本台账 ─────────────────────────────────────────────────


def test_clear_preserves_taskstate_ledger(idx) -> None:
    idx.append_taskstate_judgment(_rec())
    idx.clear()
    got = idx.get_taskstate_judgment(TURN)
    assert got is not None, (
        "clear() 把新开判定清掉了——重放首轮集合会随旋钮漂，matter_id 跟着漂"
        "（§0.1b 重演）。ADR-0018 §4.1 收录原则：不可确定性重算的记录跨重建保留。")


# ── ⑤ rebuild_id 盖章（MS-13）──────────────────────────────────────────


def test_rebuild_id_is_stamped_when_absent(idx) -> None:
    idx.append_taskstate_judgment(_rec())
    got = idx.get_taskstate_judgment(TURN)
    assert got.rebuild_id, "未盖 rebuild_id 的判决会重演 MS-13（历史残渣弄脏读数）"


def test_explicit_rebuild_id_is_kept(idx) -> None:
    idx.append_taskstate_judgment(_rec(rebuild_id="run-fixed"))
    assert idx.get_taskstate_judgment(TURN).rebuild_id == "run-fixed"
