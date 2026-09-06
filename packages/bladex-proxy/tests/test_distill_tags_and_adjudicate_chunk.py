"""2026-08-08 全量重建暴露的两个缺陷的回归剧本。

两个缺陷都**不是逻辑写错**，是"验证形态不对"，所以值得一起放在这里：

  ① `DistillFact` 没有 `tags` 字段，而 `mark_language_drift` 直接 `f.tags = ...`
     → pydantic v2 抛 ValueError → 整轮蒸馏 return empty（784 次调用里死了 163 次）。
     **既有测试 `test_language_drift_tagged` 一直是绿的** —— 它喂的是本地替身
     `_DFact`（普通 class，随便赋属性都行），而生产走的是 pydantic 模型。
     测试断言的正是崩掉的那行行为，却因为替身有那个字段而通过。
     → 本文件的规矩：**保真层的测试一律喂真 `DistillFact`**。

  ② 裁决把全部 pending 打包成**一次** LLM 调用。805 条 → 返回 JSON 被输出上限
     截断 → `_parse` 全量降级 ADD。而降级结果 `ops={'add': 805}` 与"裁决正常跑完、
     判定就是全 ADD"**在日志上长得一模一样**，所以它瞒过了整整一轮 5 小时重建。
     → 本文件钉死：分块生效，且**单块失败不得污染其它块**。
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from bladex_core.adjudication import AdjudicationInput, AdjudicationOp
from bladex_core.distillation import DistillFact
from bladex_core.distill_fidelity import TAG_LANG_DRIFT, mark_language_drift

from bladex_proxy import adjudicator as adj_mod
from bladex_proxy.adjudicator import LLMAdjudicator


# ══════════════════════════════════════════════════════════════════════════
# ① 保真层 × 真 DistillFact
# ══════════════════════════════════════════════════════════════════════════

def test_mark_language_drift_on_real_distillfact():
    """🔴 用真模型，不用替身 —— 这正是 163 次失败躲过测试的地方。"""
    facts = [DistillFact(content="x"), DistillFact(content="y")]
    mark_language_drift(facts)                      # 修复前：ValueError
    assert all(TAG_LANG_DRIFT in f.tags for f in facts)


def test_mark_language_drift_is_idempotent():
    f = DistillFact(content="x")
    mark_language_drift([f])
    mark_language_drift([f])
    assert f.tags.count(TAG_LANG_DRIFT) == 1


def test_distillfact_has_tags_field_declared():
    """字段必须是**声明**出来的，不能靠 model_config extra 兜。

    `getattr(f, "tags", "")` 那种防御性读会把"字段不存在"掩盖成空串，
    读不报错、写才炸，而写只在少数分支 —— 是最难发现的组合。
    """
    assert "tags" in DistillFact.model_fields


# ══════════════════════════════════════════════════════════════════════════
# ② tags 合并 / 通道解析
# ══════════════════════════════════════════════════════════════════════════

def test_merge_tags_keeps_origin_first_and_preserves_drift():
    from bladex_core.consolidation_proxy import _merge_tags
    out = _merge_tags("conclusion", TAG_LANG_DRIFT)
    assert out.split(",")[0] == "origin:conclusion"
    assert TAG_LANG_DRIFT in out


def test_channel_of_survives_second_token():
    """旧写法 `tags.split(":")[-1]` 在这里会返回 `drift`。

    不报错、不崩，只是漏斗标签从此没意义 —— 与今天查了半天的
    "progress 恒 0"是同一类：**测量口径悄悄歪掉**。
    """
    from bladex_core.consolidation_proxy import _channel_of
    assert _channel_of(f"origin:conclusion,{TAG_LANG_DRIFT}") == "conclusion"
    assert _channel_of("origin:progress") == "progress"
    assert _channel_of("") == "consolidation"


# ══════════════════════════════════════════════════════════════════════════
# ③ 裁决分块 + 爆炸半径
# ══════════════════════════════════════════════════════════════════════════

#: 🔴 裁决器返回契约的键是 **`id`**，不是 `candidate_id`（`_parse` 读 `row["id"]`）。
#:
#: 本文件初版在这里写了 `candidate_id`，于是每一行都匹配不上、全部走
#: `missing_in_output` 降级 —— 测试红了，但红的原因是**假响应不符合真契约**，
#: 与被测代码无关。这与本文件开头记的 `_DFact` 事故是同一类错误（替身与真实不符），
#: 且是在写完那段注释的半小时内再犯的一次。
#: **造假响应之前先读一遍解析端要什么**，别照着"看起来合理"的形状编。
def _verdict_row(cid: str, op: str = "add") -> dict[str, Any]:
    return {"id": cid, "op": op, "reason": "ok"}


def _items(n: int) -> list[AdjudicationInput]:
    return [AdjudicationInput(candidate_id=f"c{i}", content=f"fact {i}")
            for i in range(n)]


class _Resp:
    def __init__(self, text: str) -> None:
        self.choices = [type("C", (), {"message": type("M", (), {"content": text})()})()]


@pytest.fixture
def _chunk_of_5(monkeypatch):
    monkeypatch.setenv("BLADEX_ADJUDICATE_CHUNK", "5")


def test_adjudicate_splits_into_chunks(monkeypatch, _chunk_of_5):
    calls: list[int] = []

    def fake(**kw: Any) -> Any:
        body = kw["messages"][1]["content"]
        ids = [ln.split()[-1] for ln in body.splitlines() if ln.startswith("### NEW ")]
        calls.append(len(ids))
        return _Resp(json.dumps([_verdict_row(i) for i in ids]))

    monkeypatch.setattr(adj_mod.router_sdk, "completion", fake)
    out = LLMAdjudicator(model="m").adjudicate(_items(12))

    assert calls == [5, 5, 2], f"没分块或分块大小不对: {calls}"
    assert len(out) == 12
    # 带上 reason 一起断言：降级和正常判决都可能是 op=add，
    # 只看 op 分不出"裁决判它该 ADD"和"裁决没跑成所以 ADD"——
    # 这正是缺陷 2 瞒过 5 小时重建的方式。
    assert not any(v.fallback for v in out), \
        f"降级原因: {sorted({v.reason for v in out if v.fallback})}"


def test_one_bad_chunk_does_not_poison_the_rest(monkeypatch, _chunk_of_5):
    """🔴 核心判据：第二块返回截断 JSON，只有第二块降级。

    修复前整批共用一次调用，这一个截断会让 12 条全部 fallback。
    """
    seen: list[int] = []

    def fake(**kw: Any) -> Any:
        body = kw["messages"][1]["content"]
        ids = [ln.split()[-1] for ln in body.splitlines() if ln.startswith("### NEW ")]
        seen.append(len(seen))
        if len(seen) == 2:
            return _Resp('[{"candidate_id": "c5", "op": "add", "reas')   # 截断
        return _Resp(json.dumps([_verdict_row(i) for i in ids]))

    monkeypatch.setattr(adj_mod.router_sdk, "completion", fake)
    out = LLMAdjudicator(model="m").adjudicate(_items(12))

    by_id = {v.candidate_id: v for v in out}
    assert len(out) == 12
    bad = [f"c{i}" for i in range(5, 10)]
    assert all(by_id[c].fallback for c in bad), "坏块应当降级"
    good = [f"c{i}" for i in range(5)] + [f"c{i}" for i in (10, 11)]
    assert not any(by_id[c].fallback for c in good), \
        f"好块被坏块污染了；降级原因: {sorted({by_id[c].reason for c in good})}"


def test_all_verdicts_are_add_when_every_chunk_fails(monkeypatch, _chunk_of_5):
    """宁多勿丢：全失败也必须返回等长的 ADD，不能丢条目。"""
    monkeypatch.setattr(adj_mod.router_sdk, "completion",
                        lambda **kw: _Resp("not json at all"))
    out = LLMAdjudicator(model="m").adjudicate(_items(7))
    assert len(out) == 7
    assert all(v.op is AdjudicationOp.ADD and v.fallback for v in out)


# ══════════════════════════════════════════════════════════════════════════
# ④ 截断打捞 + 输出预算（2026-08-09 第二轮全量暴露）
#
# 第一轮修了"一次失败拖垮全部"（分块），第二轮发现**每块都失败**：
# `max_tokens=1024` 装不下 25 条判决，34 块 34 次截断。
# 分块把爆炸半径限住了，却没让任何一块活下来 —— 两个问题必须分开钉。
# ══════════════════════════════════════════════════════════════════════════

def test_salvage_recovers_complete_objects_from_truncated_array():
    from bladex_proxy.adjudicator import _salvage_objects
    text = ('[{"id": "c0", "op": "add", "reason": "ok"},\n'
            ' {"id": "c1", "op": "noop", "reason": "dup"},\n'
            ' {"id": "c2", "op": "ad')          # 第三条被剪断
    got = _salvage_objects(text)
    assert [o["id"] for o in got] == ["c0", "c1"]


def test_truncated_chunk_keeps_the_complete_verdicts(monkeypatch, _chunk_of_5):
    """🔴 截断只该损失最后半条，不是整块连坐。

    修复前：数组少个右括号 -> 整块 5 条全降级 ADD。
    """
    def fake(**kw: Any) -> Any:
        body = kw["messages"][1]["content"]
        ids = [ln.split()[-1] for ln in body.splitlines() if ln.startswith("### NEW ")]
        rows = ",".join(json.dumps(_verdict_row(i)) for i in ids[:-1])
        return _Resp(f'[{rows},{{"id": "{ids[-1]}", "op": "ad')   # 末条剪断

    monkeypatch.setattr(adj_mod.router_sdk, "completion", fake)
    out = LLMAdjudicator(model="m").adjudicate(_items(5))
    by_id = {v.candidate_id: v for v in out}

    assert not any(by_id[f"c{i}"].fallback for i in range(4)), "完整的四条被连坐了"
    assert by_id["c4"].fallback, "被剪断的那条应当降级"


def test_truncation_salvaged_on_both_failure_paths(monkeypatch, _chunk_of_5):
    """🔴 截断有两条路径，都得能走到打捞。

    线上 34 次全是 `json_err`（判决里 `"targets": []` 的右括号让 rfind 找得到），
    而上一条剧本造的是 `no_array`（不带 targets）。初版打捞只挂在 json_err 分支，
    **对线上有效、对剧本无效** —— 两种形状必须各钉一次，
    否则修好一条就以为全好了。
    """
    def make(with_targets: bool):
        def fake(**kw: Any) -> Any:
            body = kw["messages"][1]["content"]
            ids = [ln.split()[-1] for ln in body.splitlines()
                   if ln.startswith("### NEW ")]
            rows = []
            for i in ids[:-1]:
                row = _verdict_row(i)
                if with_targets:
                    row["targets"] = []          # 这个 `]` 决定走哪条失败路径
                rows.append(json.dumps(row))
            return _Resp("[" + ",".join(rows) + f',{{"id": "{ids[-1]}", "op": "ad')
        return fake

    for with_targets in (True, False):
        monkeypatch.setattr(adj_mod.router_sdk, "completion", make(with_targets))
        out = LLMAdjudicator(model="m").adjudicate(_items(5))
        by_id = {v.candidate_id: v for v in out}
        kept = [i for i in range(4) if not by_id[f"c{i}"].fallback]
        assert len(kept) == 4, \
            f"with_targets={with_targets}: 只救回 {len(kept)}/4 条"
        assert by_id["c4"].fallback


def test_output_budget_scales_with_chunk_size():
    """固定 max_tokens 是第二轮的真根因 —— 预算必须随条数走。"""
    seen: dict[str, Any] = {}

    def fake(**kw: Any) -> Any:
        seen.update(kw)
        return _Resp("[]")

    import pytest as _pt
    with _pt.MonkeyPatch.context() as mp:
        mp.setattr(adj_mod.router_sdk, "completion", fake)
        mp.setenv("BLADEX_ADJUDICATE_CHUNK", "50")
        LLMAdjudicator(model="m", max_tokens=1024).adjudicate(_items(50))

    assert seen["max_tokens"] >= 50 * 160, \
        f"50 条只给了 {seen['max_tokens']} token —— 又会被截断"


def test_configured_max_tokens_is_a_floor_not_a_cap():
    """用户调高有效；调低不至于自伤。"""
    seen: dict[str, Any] = {}

    def fake(**kw: Any) -> Any:
        seen.update(kw)
        return _Resp("[]")

    import pytest as _pt
    with _pt.MonkeyPatch.context() as mp:
        mp.setattr(adj_mod.router_sdk, "completion", fake)
        LLMAdjudicator(model="m", max_tokens=99999).adjudicate(_items(3))

    assert seen["max_tokens"] == 99999
