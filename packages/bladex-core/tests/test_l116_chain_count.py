"""MQ-L116（批 L116 · 2a）· 相关性段按「连续命中的链」计 shared，而不是逐个 2-gram。

病例形态（文本全部自编）：用户话 = 一句主题话 + 一句常见约束套话。套话「不做任何动作」按重叠
2-gram 拆成 5 个单元，在池里 df 恰好 ≤ `rare_df`，于是只共享这一句套话的账本同时过稀有锚和
shared≥2 两道门，把真正同主题、但只共享 3 个单元的账本挤出 `top_n=3`。
"""
from __future__ import annotations

from bladex_core.ledger import new_ledger
from bladex_core.ledger_runtime import relevant_ledgers

_TARGET = "ldg-target"
_PHRASE = "不做任何动作"
_QUERY = f"检查新版本升级，{_PHRASE}"


def _pool(extra: list[tuple[str, str, str]] | None = None) -> dict:
    """40 本：1 同主题 + 4 本只共享套话 + 35 本无关填充（n=40 ⇒ rare_df=4，套话 df=4 正好落在锚内）。"""
    rows: list[tuple[str, str, str]] = [
        (_TARGET, "星云望远镜固件评估", "评估新版本的升级价值"),
    ]
    for tag in "甲乙丙丁":
        rows.append((f"ldg-junk{tag}", f"{tag}类调研星云观测", f"研究星云观测资料，{_PHRASE}"))
    for i in range(35):
        rows.append((f"ldg-fill{i}", f"杂项{chr(0x5000 + i)}{chr(0x5100 + i)}号", f"{chr(0x5200 + i)}{chr(0x5300 + i)}整理"))
    rows += extra or []
    return {lid: new_ledger(title=t, goal=g, goal_source="model", created_at="2026-11-06T00:00:00+00:00")
            for lid, t, g in rows}


def _ids(ms) -> list[str]:
    return [m.ledger_id for m in ms]


def test_case_shape_boilerplate_ledgers_no_longer_crowd_out_the_topic_match() -> None:
    pool = _pool()
    legacy = relevant_ledgers(pool, _QUERY, chain_rare_df=None)
    assert _TARGET not in _ids(legacy)                      # 旧口径：3 本套话账本占满 top_n
    assert all(i.startswith("ldg-junk") for i in _ids(legacy))
    now = relevant_ledgers(pool, _QUERY)
    assert _ids(now) == [_TARGET]                          # 新口径：同主题第 1，套话账本不进


def test_long_verbatim_sentence_still_listed() -> None:
    """HO 旗舰形态（自编）：整句逐字命中，是一条很长的链，且整句只有这一本含有。"""
    sentence = "把雾凇星图渲染管线的缓存失效逻辑改成按瓦片版本号判定"
    pool = _pool([("ldg-long", "雾凇星图缓存", sentence)])
    ms = relevant_ledgers(pool, f"接着收尾：{sentence}")
    assert _ids(ms) == ["ldg-long"]
    assert ms[0].shared >= 2 and ms[0].score >= 0.3
    # 对照：没有「极稀有整句」例外（纯链合并）时这条会掉成 shared=1 而被门槛挡下
    assert "ldg-long" not in _ids(relevant_ledgers(pool, f"接着收尾：{sentence}", chain_rare_df=0))


def test_latin_identifier_scoring_unchanged() -> None:
    pool = _pool([("ldg-lat", "frobnicate 调优", "调优 frobnicate 的参数")])
    q = "看看 frobnicate 怎么样了"
    new, old = relevant_ledgers(pool, q), relevant_ledgers(pool, q, chain_rare_df=None)
    assert _ids(new) == _ids(old) == ["ldg-lat"]
    assert new[0].shared == old[0].shared == 2 and new[0].score == old[0].score


def test_isolated_single_bigrams_count_like_the_baseline() -> None:
    """链长 1（孤立 2-gram，彼此被未命中的字隔开）：新旧口径 shared 完全一致。"""
    pool = _pool([("ldg-iso", "固件与升级", "固件 与 升级 备忘")])
    q = "固件如何如何升级"          # 固件 / 升级 各自成链（中间的「如何」是停表词）
    new, old = relevant_ledgers(pool, q), relevant_ledgers(pool, q, chain_rare_df=None)
    assert _ids(new) == _ids(old)
    assert [m.shared for m in new] == [m.shared for m in old]
    assert next(m.shared for m in new if m.ledger_id == "ldg-iso") == 2
