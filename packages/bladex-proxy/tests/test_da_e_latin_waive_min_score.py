"""MQ-L125（批 DA · E）· 短话 + 共享稀有拉丁标识 ⇒ 放行 `min_score`。

病例形态（文本全部自编）：用户话里几个横跨词边界的 CJK 2-gram 在池里各有少量出现，抬高分母，
唯一拉丁标识 zorbit（池内 df=1）自己的分只有 ~0.26 < 0.30，同主题旧本不进候选。
"""
from __future__ import annotations

import bladex_core.ledger_runtime as lr
from bladex_core.ledger import new_ledger
from bladex_core.ledger_runtime import LATIN_WAIVE_MAX_DF, relevant_ledgers

TARGET = "ldg-target"
_SHORT = "之前调研过的 zorbit 有没有新的进展"
_CJK_ONLY = "之前调研过的项目有没有新的进展"


def _led(lid: str, title: str, goal: str):
    return new_ledger(title=title, goal=goal, goal_source="model",
                      created_at="2026-10-01T00:00:00+00:00", ledger_id=lid)


def _pool(*, extra_latin: int = 0, target_cjk: str = "") -> dict:
    """40+ 本：目标本含 zorbit；三本填充各含 query 里的 CJK 单元（抬分母）；其余无关。"""
    pool = {TARGET: _led(TARGET, "zorbit 项目调研", f"调研 zorbit {target_cjk}")}
    for i in range(3):
        pool[f"ldg-cjk{i}"] = _led(f"ldg-cjk{i}", f"杂项{chr(0x5000 + i)}号", "之前 有新 的进 研过")
    for i in range(extra_latin):
        pool[f"ldg-lat{i}"] = _led(f"ldg-lat{i}", f"另一件 zorbit 事{chr(0x5400 + i)}", f"{chr(0x5500 + i)}{chr(0x5600 + i)}")
    for i in range(36):
        pool[f"ldg-fill{i}"] = _led(f"ldg-fill{i}", f"杂项{chr(0x6000 + i)}{chr(0x6100 + i)}号",
                                    f"{chr(0x6200 + i)}{chr(0x6300 + i)}整理")
    return pool


def _ids(ms) -> list[str]:
    return [m.ledger_id for m in ms]


def test_precondition_score_below_gate_without_waive() -> None:
    """夹具自检：不放行时（min_score=0.30 且关掉放行）该本确实被分数门挡掉，分数 < 0.30。"""
    pool = _pool()
    ms = relevant_ledgers(pool, _SHORT, min_score=0.0, top_n=10)
    hit = [m for m in ms if m.ledger_id == TARGET]
    assert hit and hit[0].score_raw < 0.30


def test_case1_short_text_shared_rare_latin_low_score_is_listed() -> None:
    diag: dict[str, int] = {}
    ms = relevant_ledgers(_pool(), _SHORT, diag=diag, top_n=10)
    assert TARGET in _ids(ms)
    m = next(x for x in ms if x.ledger_id == TARGET)
    assert m.score_raw < 0.30                    # 分数照常计算、不被改写
    assert diag["score_gate_waived"] == 1


def test_case2_same_but_over_length_gate_is_not_listed() -> None:
    long_text = _SHORT + "，" + "顺便说一句" * 20
    assert len(long_text) > lr.CLOSED_STRONG_MAX_USER_CHARS
    diag: dict[str, int] = {}
    ms = relevant_ledgers(_pool(), long_text, diag=diag, top_n=10)
    assert TARGET not in _ids(ms)
    assert diag["score_gate_waived"] == 0


def test_case3_shared_latin_df_3_is_not_listed() -> None:
    pool = _pool(extra_latin=2)                   # zorbit 现在 df=3 > LATIN_WAIVE_MAX_DF
    assert LATIN_WAIVE_MAX_DF == 2
    diag: dict[str, int] = {}
    ms = relevant_ledgers(pool, _SHORT, diag=diag, top_n=10)
    assert TARGET not in _ids(ms)
    assert diag["score_gate_waived"] == 0


def test_case4_cjk_only_shared_low_score_is_not_listed() -> None:
    pool = _pool()
    pool[TARGET] = _led(TARGET, "项目调研", "调研 项目 进展")      # 不含任何拉丁标识
    diag: dict[str, int] = {}
    ms = relevant_ledgers(pool, _CJK_ONLY, diag=diag, top_n=10)
    assert TARGET not in _ids(ms)
    assert diag["score_gate_waived"] == 0


def test_case5_high_score_hit_is_not_counted_as_waived() -> None:
    pool = {TARGET: _led(TARGET, "zorbit 项目调研", "调研 zorbit")}
    for i in range(30):
        pool[f"ldg-fill{i}"] = _led(f"ldg-fill{i}", f"杂项{chr(0x6000 + i)}{chr(0x6100 + i)}号",
                                    f"{chr(0x6200 + i)}{chr(0x6300 + i)}整理")
    diag: dict[str, int] = {}
    ms = relevant_ledgers(pool, "zorbit 怎么样了", diag=diag)
    assert _ids(ms) == [TARGET] and ms[0].score_raw >= 0.30
    assert diag["score_gate_waived"] == 0


def test_closed_and_superseded_rules_unchanged_by_waive() -> None:
    """放行只松 min_score：已关且被取代的本照旧不列；久关本仍须过 0.40 强命中门。"""
    from bladex_core.ledger import STATUS_CLOSED
    pool = _pool()
    pool[TARGET].status = STATUS_CLOSED
    pool[TARGET].closed_at = "2026-11-16T10:00:00+00:00"
    pool[TARGET].superseded_by = "ldg-cjk0"
    now = lr.iso_ms("2026-11-16T10:30:00+00:00")
    assert TARGET not in _ids(relevant_ledgers(pool, _SHORT, top_n=10, now_ms=now))
    pool[TARGET].superseded_by = ""
    assert TARGET in _ids(relevant_ledgers(pool, _SHORT, top_n=10, now_ms=now))          # ≤24h 的已关本照列
    far = lr.iso_ms("2026-12-30T10:30:00+00:00")
    assert TARGET not in _ids(relevant_ledgers(pool, _SHORT, top_n=10, now_ms=far))     # 久关：分 <0.40 仍不列


def test_case6_shared_rare_latin_but_score_below_floor_is_not_listed() -> None:
    """gate 暴露的形态：只撞了一个名字（`codex` 类，df=2）、分数极低 ⇒ 不放行（判别力）。"""
    pool = _pool()
    for i in range(3):                               # 同一条 query 里更多池内词（各 df=3）⇒ 分母更大，zorbit 占比 < 地板
        pool[f"ldg-cjk{i}"] = _led(f"ldg-cjk{i}", f"杂项{chr(0x5000 + i)}号",
                                   "之前 有新 的进 研过 过的 前调 下之 查一")
    diag: dict[str, int] = {}
    ms = relevant_ledgers(pool, _SHORT, diag=diag, top_n=50)
    m = relevant_ledgers(pool, _SHORT, min_score=0.0, top_n=50)
    low = next(x for x in m if x.ledger_id == TARGET)
    assert low.score_raw < lr.LATIN_WAIVE_MIN_SCORE
    assert TARGET not in _ids(ms) and diag["score_gate_waived"] == 0
