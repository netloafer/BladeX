"""M3-2/M3-3：三时钟判定 + 时效到期（复核 5.9 用户拍板 + 四点工程修正）。

判定是**纯函数、只判不动**——所以这一层可以完整测，不必碰存储。
落库那一步（`recompute_matter_clocks`）在集成测试里验。

X6 纪律：本文件的场景词汇不得出现在实现代码里。
判定只看三个时间戳与 kind，不看任何领域词。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from bladex_core.clocks import (
    CLOCK_EXEMPT_KINDS,
    decide_matter_clocks,
    exempt_from_clocks,
    hit_coverage,
    is_expired,
)
from bladex_core.matter import Matter, MatterStatus

_NOW = datetime(2026, 8, 6, tzinfo=UTC)


def _matter(mid: str, *, upd_days: float, hit_days: float | None,
            status: MatterStatus = MatterStatus.ACTIVE,
            created_days: float = 400.0) -> Matter:
    m = Matter(matter_id=mid, title=mid, status=status)
    m.created_at = _NOW - timedelta(days=created_days)
    m.updated_at = _NOW - timedelta(days=upd_days)
    m.last_hit_at = None if hit_days is None else _NOW - timedelta(days=hit_days)
    return m


def _decide(matters, **kw):
    return decide_matter_clocks(matters, now=_NOW, **kw)


def _q(report, mid: str) -> str:
    for d in report.decisions:
        if d.matter_id == mid:
            return d.quadrant
    return "active"     # 不在决策列表里 = active（正常，不产出决策）


# ── 2×2 矩阵四象限（H1 的解）────────────────────────────────────────────


def _covered(*matters):
    """凑够覆盖率，让护栏不生效（护栏本身另有专测）。"""
    return list(matters) + [
        _matter(f"filler{i}", upd_days=1, hit_days=1) for i in range(10)
    ]


def test_quadrant_active():
    """近期有更新 + 近期有命中 → 正常，不产出任何决策。"""
    r = _decide(_covered(_matter("m", upd_days=1, hit_days=1)))
    assert _q(r, "m") == "active"
    assert all(d.matter_id != "m" for d in r.decisions)


def test_quadrant_write_only_does_not_change_status():
    """🔴 在写、从不被用 → **过度采集的审计信号**，不是"该退场"。

    这一格刻意不改状态：它说明的是"我们在采集没人要的东西"（写入侧问题），
    把它 dormant 掉等于把写入侧的问题伪装成读取侧的清理。
    """
    r = _decide(_covered(_matter("m", upd_days=1, hit_days=999)))
    assert _q(r, "m") == "write_only"
    d = next(x for x in r.decisions if x.matter_id == "m")
    assert d.target_status is None, "审计信号不许改状态"


def test_quadrant_dormant():
    """久无更新 + 近期仍被引用 → dormant（退出常驻注入、保留可召回）。"""
    r = _decide(_covered(_matter("m", upd_days=999, hit_days=1)))
    assert _q(r, "m") == "dormant"
    d = next(x for x in r.decisions if x.matter_id == "m")
    assert d.target_status is MatterStatus.DORMANT


def test_quadrant_close_candidate_is_not_auto_closed():
    """🔴 双钟停摆 → auto-close **候选**，但落到 DORMANT 而不是 CLOSED。

    关闭是用户的语义动作。自动关一张其实还在用的卡**无法自愈**——
    dormant 至少还能被召回、被重新激活。宁可留一步余地。
    """
    r = _decide(_covered(_matter("m", upd_days=999, hit_days=999)))
    assert _q(r, "m") == "close_candidate"
    d = next(x for x in r.decisions if x.matter_id == "m")
    assert d.target_status is MatterStatus.DORMANT
    assert d.target_status is not MatterStatus.CLOSED


# ── 🔴 修正 3：观测期护栏（相关钟被注入层健康度偏置）──────────────────────


def test_guard_blocks_application_when_coverage_is_low():
    """覆盖率不足 → 只判不动。

    实测过的事故形态：注入中断期间全库零命中。若当时已有"30 天无命中即退场"，
    会发生**系统故障触发的大规模误杀**。钟只有在喂它的事件干净时才可信。
    """
    matters = [_matter(f"m{i}", upd_days=999, hit_days=None) for i in range(10)]
    r = _decide(matters)
    assert r.coverage == 0.0
    assert r.observe_only is True
    assert r.decisions, "仍要产出决策（供人工看清单）"
    assert r.to_apply() == [], "但一条都不许落库"


def test_guard_lifts_when_coverage_is_enough():
    matters = ([_matter(f"hit{i}", upd_days=999, hit_days=1) for i in range(6)]
               + [_matter(f"cold{i}", upd_days=999, hit_days=None) for i in range(4)])
    r = _decide(matters)
    assert r.coverage == 0.6 and r.observe_only is False
    assert r.to_apply(), "护栏解除后决策要能落库"


def test_never_hit_is_not_treated_as_stale_hit():
    """🔴 `last_hit_at is None` 是"还不知道"，不是"没人用"。

    拿缺失当证据正是修正 3 要防的：M3-1 刚上线时全库都是 None，
    M5 全量重建后又会清零。把它读成"久无命中"会在这两个时点各误杀一次。
    """
    r = _decide(_covered(_matter("m", upd_days=999, hit_days=None)))
    d = next(x for x in r.decisions if x.matter_id == "m")
    assert d.quadrant == "dormant", "只因久无更新判 dormant"
    assert d.quadrant != "close_candidate", "不许因为「从未命中」就升级成双钟停摆"
    assert d.days_since_hit is None


def test_hit_coverage_math():
    assert hit_coverage([]) == 0.0
    assert hit_coverage([_matter("a", upd_days=1, hit_days=1)]) == 1.0
    assert hit_coverage([_matter("a", upd_days=1, hit_days=1),
                         _matter("b", upd_days=1, hit_days=None)]) == 0.5


# ── provisional 滞留出口（H3：60% 困在中间态且无出口）────────────────────


def test_stale_provisional_gets_an_exit():
    r = _decide(_covered(_matter("m", upd_days=1, hit_days=1,
                                 status=MatterStatus.PROVISIONAL,
                                 created_days=90)))
    d = next(x for x in r.decisions if x.matter_id == "m")
    assert d.target_status is MatterStatus.DORMANT
    assert "provisional_stale" in d.reason


def test_fresh_provisional_is_left_alone():
    r = _decide(_covered(_matter("m", upd_days=1, hit_days=1,
                                 status=MatterStatus.PROVISIONAL,
                                 created_days=3)))
    assert all(x.matter_id != "m" for x in r.decisions)


def test_closed_matters_are_never_reopened():
    """用户手动关掉的卡不该被自动改回来。"""
    r = _decide(_covered(_matter("m", upd_days=999, hit_days=999,
                                 status=MatterStatus.CLOSED)))
    assert all(x.matter_id != "m" for x in r.decisions)


# ── MS-4②：completion_signal 是**加速证据**，不是判据 ────────────────────


def test_completion_signal_halves_the_update_clock():
    """有完结信号 → 更新钟阈值减半，更快够到 dormant。"""
    m = _matter("m", upd_days=10, hit_days=1)      # 10 天 < 默认 14 天
    assert _q(_decide(_covered(m)), "m") == "active"

    m2 = _matter("m", upd_days=10, hit_days=1)     # 同样 10 天
    r = _decide(_covered(m2), completion_signals={"m"})
    assert _q(r, "m") == "dormant", "10 天 > 减半后的 7 天"


def test_completion_signal_alone_does_not_close():
    """🔴 单信号不拍板：completed 不直接改 status，完结仍以三时钟为主。"""
    m = _matter("m", upd_days=1, hit_days=1)
    r = _decide(_covered(m), completion_signals={"m"})
    assert _q(r, "m") == "active", "刚更新过的卡不因一条完结信号就退场"


# ── 🔴 修正 1：kind 豁免（G3 的教训）────────────────────────────────────


@pytest.mark.parametrize("kind", sorted(CLOCK_EXEMPT_KINDS))
def test_exempt_kinds(kind):
    """preference / hard_rule / lesson **不因老而无用**。

    G3 实测：均匀年龄衰减 66 天驱逐了用户偏好。
    一条"回复请简洁"不会因为说过很久就不再成立（Zep：validity ≠ age）。
    """
    assert exempt_from_clocks(kind) is True


@pytest.mark.parametrize("kind", ["assertion", "procedure", "file_ref", "profile_obs", ""])
def test_non_exempt_kinds(kind):
    assert exempt_from_clocks(kind) is False


# ── M3-3：时效到期 ──────────────────────────────────────────────────────


def test_expired_when_past():
    assert is_expired(_NOW - timedelta(days=1), _NOW) is True


def test_not_expired_when_future():
    assert is_expired(_NOW + timedelta(days=1), _NOW) is False


def test_none_never_expires():
    """绝大多数条目没有到期点 —— 没有到期点的记忆不因时间失效。

    这与 kind 豁免是同一条原则的两个面。
    """
    assert is_expired(None, _NOW) is False


def test_naive_datetime_is_treated_as_utc():
    """蒸馏产出的日期常常不带时区 —— 不该因此判错。"""
    assert is_expired(datetime(2026, 8, 5), _NOW) is True
    assert is_expired(datetime(2026, 8, 7), _NOW) is False


# ── 决策不动数据（只判不删）────────────────────────────────────────────


def test_decisions_do_not_mutate_matters():
    """判定是纯函数：跑完之后对象本身一个字段都不该变。"""
    m = _matter("m", upd_days=999, hit_days=999)
    before = m.model_dump(mode="json")
    _decide(_covered(m))
    assert m.model_dump(mode="json") == before


# ══════════════════════════════════════════════════════════════════════════
# MS-6：错卡解钉（钉卡自愈）+ dormant 不钉卡但仍可召回
# ══════════════════════════════════════════════════════════════════════════


class _Retriever:
    """最小检索桩：只提供 ②平面需要的接口。"""

    def __init__(self, matters):
        self._matters = matters

    def search(self, query, top_k=10, user_id=None, visibility=None, q_vec=None):
        return []

    def list_hard_rules(self):
        return []

    def search_matters_by_query(self, query, k=5, q_vec=None):
        return list(self._matters)

    def get_facts_for_matter(self, matter_id, top_k=5):
        return []

    def get_matter(self, matter_id):
        return next((m for m in self._matters if m.matter_id == matter_id), None)


