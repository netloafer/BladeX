"""批 AA2 · full 重建只取决于 Hub（MQ-I37）+ 增量中止可重试（MQ-I32）。

| 用例 | 判据 |
|---|---|
| `test_full_rebuild_is_start_point_independent` | 同一 Hub：① 空库 ② 跑过一次 full ③ 有增量历史 —— 三种起点做 full，fact 内容集合逐字相同 |
| `test_second_full_on_user_only_turns_is_not_empty` | AA 夹具形态（只有已见 user、assistant 空）：第二次 full 不再是 0 条 |
| `test_incremental_abort_retry_loses_nothing_and_does_not_stall` | 确定性解析失败：不永久停摆，终态 fact 集合 = 无注入对照 |
| `test_every_meta_prefix_is_classified` / `…_detects_unclassified` | 前缀闭集守卫 |

夹具里两类轮次正是被 `seen_discourse/` memo 吃掉的那两种：
- assistant 正文 < 40 字（结论通道为空）、只有一条 user 话语 —— 第二次 full 时"蒸过且 assistant 空就跳"；
- user 消息里带着历史话语（agent 首次接入 / 压缩后重启的真实形态）—— 历史话语靠"补采"蒸，memo 说蒸过就不补。
存储一律 tmp_path，桩上游（不调模型）。
"""

from __future__ import annotations

import importlib.util
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "_aa_reliability", _HERE / "test_aa_rebuild_reliability.py")
_aa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_aa)          # 复用批 AA 的桩（Stub / BigramEmbedder / build_index）


def _stubs() -> dict[str, Any]:
    return _aa._stubs()


def _make_hub(path: Path) -> Path:
    from bladex_proxy.models import Identity, Turn, TurnStatus
    from bladex_proxy.storage.memory_hub import MemoryHub

    hub = MemoryHub(path)
    hub.open()
    n = 0

    def put(session: str, msgs: list[dict], response: str) -> None:
        nonlocal n
        ident = Identity(user_id="u1", agent_id="codex", session_id=session, turn_index=n)
        hub.put(ident.storage_key(f"17999100{n:02d}-0"), Turn(
            identity=ident, model="m", status=TurnStatus.OK, response_text=response,
            request_messages=msgs, ts=datetime(2026, 10, 22, 9, 0, n, tzinfo=UTC)))
        n += 1

    # ① 只有 user 话语、assistant 短（< 40 字 ⇒ 结论通道空）
    for i in range(4):
        put("s-short", [{"role": "user", "content":
                         f"BladeX 全量重建的当前进度是第 {i} 阶段，负责人是 Jason。"
                         f"编号 R{i * 41} 的样本记录已经归档"}], "收到")
    # ② 带历史的轮次：首轮就带着三条从未当过"最后一条"的历史话语（需补采）
    hist = [f"历史话语第 {j} 条：项目 H{j} 的验收清单已经核对完毕" for j in range(3)]
    for i in range(3):
        msgs: list[dict] = []
        for h in hist + [f"补采会话第 {k} 轮：新的问题 P{k} 已经登记" for k in range(i)]:
            msgs += [{"role": "user", "content": h}, {"role": "assistant", "content": "好"}]
        msgs.append({"role": "user", "content": f"补采会话第 {i} 轮：新的问题 P{i} 已经登记"})
        put("s-hist", msgs, "收到")
    hub.close()
    return path


@pytest.fixture()
def hub(tmp_path: Path) -> Path:
    return _make_hub(tmp_path / "hub")


def _rebuild(tmp: Path, hub: Path, name: str, monkeypatch: pytest.MonkeyPatch, *,
             full: bool = True, stub: Any = None, passes: int = 1,
             max_turns: int = 30) -> tuple[Any, list[dict]]:
    """在 tmp/name 上跑 `passes` 次重建（full 或增量）；返回 (index, 每 pass 读数)。"""
    from bladex_proxy import adjudicator as adj
    from bladex_proxy import router_sdk
    from bladex_proxy.storage.memory_hub import MemoryHub

    stub = stub or _stubs()["Stub"]()
    monkeypatch.setattr(router_sdk, "completion", stub.completion)
    clock = [0.0]
    monkeypatch.setattr(adj, "_monotonic", lambda: clock[0])
    monkeypatch.setattr(adj, "_sleep", lambda s: clock.__setitem__(0, clock[0] + s))
    monkeypatch.setenv("BLADEX_ADJUDICATE_CHUNK", "1")
    idx = _stubs()["build_index"](tmp / name, stub)
    router_sdk.completion = stub.completion
    h = MemoryHub(hub)
    h.open()
    reads = []
    try:
        for _ in range(passes):
            n = idx.rebuild_from_hub(h, full=full, max_turns=max_turns, distill_concurrency=4)
            reads.append({"new": n, "abort": idx.last_rebuild_abort_reason,
                          "consumed": idx.consumed_count(), "facts": idx.fact_count()})
    finally:
        h.close()
    return idx, reads


def _contents(idx: Any) -> list[str]:
    return sorted(f.content for f in idx.all_facts())


# ── MQ-I37：起点无关 ─────────────────────────────────────────────────────────

def test_full_rebuild_is_start_point_independent(
        tmp_path: Path, hub: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # ① 空库起跑
    a, _ = _rebuild(tmp_path, hub, "empty", monkeypatch)
    ref = _contents(a)
    a.close()
    assert len(ref) >= 10, "夹具应产出足够多的 fact"
    # ② 在跑过一次 full 的库上再 full
    b, _ = _rebuild(tmp_path, hub, "twice", monkeypatch, passes=2)
    got_b = _contents(b)
    b.close()
    # ③ 有增量历史的库（先增量消化完，再 full）
    c, _ = _rebuild(tmp_path, hub, "incr", monkeypatch, full=False, passes=2)
    assert c.consumed_count() > 0
    c.close()
    c, _ = _rebuild(tmp_path, hub, "incr", monkeypatch)
    got_c = _contents(c)
    c.close()
    assert got_b == ref, f"在跑过 full 的库上：缺 {sorted(set(ref) - set(got_b))[:3]}"
    assert got_c == ref, f"在有增量历史的库上：缺 {sorted(set(ref) - set(got_c))[:3]}"


def test_second_full_on_user_only_turns_is_not_empty(
        tmp_path: Path, hub: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """AA 夹具形态：第二次 full 曾经整轮跳过（16 → 0）。"""
    idx, reads = _rebuild(tmp_path, hub, "again", monkeypatch, passes=2)
    short = [c for c in _contents(idx) if "全量重建的当前进度" in c]
    idx.close()
    assert reads[0]["facts"] > 0 and reads[1]["facts"] == reads[0]["facts"]
    assert len(short) == 4, "只有 user 话语的四轮，第二次 full 应同样产出"


# ── MQ-I32：增量中止可重试 ───────────────────────────────────────────────────

def test_incremental_abort_retry_loses_nothing_and_does_not_stall(
        tmp_path: Path, hub: Path, monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture) -> None:
    import structlog.testing

    ns = _stubs()
    ctl, _ = _rebuild(tmp_path, hub, "control", monkeypatch, full=False, passes=2)
    control = _contents(ctl)
    total = ctl.consumed_count()
    ctl.close()

    bad = ns["Stub"]()
    bad.garbage_marker = "第 2 阶段"            # 这一条候选每次都回非 JSON（确定性）
    with structlog.testing.capture_logs() as logs:
        idx, reads = _rebuild(tmp_path, hub, "inject", monkeypatch, full=False,
                              stub=bad, passes=8)
    got = _contents(idx)
    idx.close()
    aborts = [r for r in reads if r["abort"] == "adjudicate_outage"]
    events = [e["event"] for e in logs]
    # 先判丢数据（病因），再判停摆与放行（机制）
    # 中止 pass 不留副作用 ⇒ 已见话语 / 补采话语没有因为"已标记蒸过"而在重试时被跳过
    missing = sorted(set(control) - set(got))
    assert not missing, f"中止后重试丢了 fact：{missing[:3]}"
    assert set(got) - set(control) <= {c for c in got if "第 2 阶段" in c}
    assert reads[-1]["consumed"] == total, f"不许永久停摆：{reads}"
    assert 1 <= len(aborts) <= 3, f"应先中止、计满次数后放行：{reads}"
    assert "index_rebuild_adjudicate_retry_exhausted" in events


def test_adjudicate_retry_count_clears_after_successful_pass(
        tmp_path: Path, hub: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """调度变异 Z3（AA2 复核 0 红，拍板 (9)）：成功的 pass 必须清掉候选的裁决重试计数。

    否则计数跨 pass 累积：偶发失败过一次的候选，再失败时会被提前放行成降级 ADD。
    上限设 2 让差别落到行为上：清零 ⇒ 失败两次仍中止；不清 ⇒ 第二次失败就放行。
    """
    from bladex_proxy import router_sdk
    from bladex_proxy.storage.memory_hub import MemoryHub
    from bladex_proxy.storage.memory_index import _ADJ_RETRY_PREFIX, _CONSUMED_PREFIX

    monkeypatch.setenv("BLADEX_ADJUDICATE_RETRY_MAX", "2")
    bad = _stubs()["Stub"]()
    bad.garbage_marker = "第 2 阶段"
    idx, reads = _rebuild(tmp_path, hub, "z3", monkeypatch, full=False, stub=bad)
    assert reads[-1]["abort"] == "adjudicate_outage"

    def counted() -> dict[str, int]:
        prefix = _ADJ_RETRY_PREFIX.encode()
        return {k.decode(): int(v) for k, v in idx.meta_db.items()
                if isinstance(k, bytes) and k.startswith(prefix)}

    first = counted()
    assert first and set(first.values()) == {1}, f"中止的 pass 应给降级候选计 1：{first}"

    good = _stubs()["Stub"]()
    monkeypatch.setattr(router_sdk, "completion", good.completion)
    h = MemoryHub(hub)
    h.open()
    try:
        idx.rebuild_from_hub(h, full=False, max_turns=30, distill_concurrency=4)
        assert idx.last_rebuild_abort_reason is None
        assert counted() == {}, "成功的 pass 必须把计数清零"

        # 让同一批候选再走一遍裁决：只撤消费标记与话语 memo，不碰计数前缀（`clear()` 会连它一起清，
        # 那样就测不到「成功后清零」了）；判重阈值拉满，免得已入库的同文候选被判重挡在裁决之外。
        consumed = _CONSUMED_PREFIX.encode()
        for k in [k for k in idx.meta_db.keys() if isinstance(k, bytes)
                  and (k.startswith(consumed) or k.startswith(b"seen_discourse/"))]:
            del idx.meta_db[k]
        idx._novelty_threshold = 2.0
        monkeypatch.setattr(router_sdk, "completion", bad.completion)

        idx.rebuild_from_hub(h, full=False, max_turns=30, distill_concurrency=4)
        assert idx.last_rebuild_abort_reason == "adjudicate_outage"
        assert set(counted().values()) == {1}, f"成功之后再失败应从 1 数起：{counted()}"
        idx.rebuild_from_hub(h, full=False, max_turns=30, distill_concurrency=4)
        assert idx.last_rebuild_abort_reason == "adjudicate_outage", "没到上限不许放行"
        assert set(counted().values()) == {2}
    finally:
        h.close()
        idx.close()


# ── 前缀闭集守卫 ─────────────────────────────────────────────────────────────

_CONST_RE = re.compile(r'^\s*(_[A-Z0-9_]+_(?:PREFIX|KEY))\s*=\s*b?"([^"]+)"', re.MULTILINE)


def _unclassified(source: str) -> list[str]:
    from bladex_proxy.storage import memory_index as mi

    classified = [*mi._META_CLEARED, *mi._META_RESET_AT_FULL_START, *mi._META_KEPT]
    out = []
    for name, value in _CONST_RE.findall(source):
        hits = classified.count(value)
        if hits != 1:
            out.append(f"{name}={value!r} (登记 {hits} 次)")
    return out


def test_every_meta_prefix_is_classified() -> None:
    from bladex_proxy.storage import memory_index as mi

    src = Path(mi.__file__).read_text(encoding="utf-8")
    assert len(_CONST_RE.findall(src)) >= 19, "前缀常量没读全（正则失效？）"
    assert _unclassified(src) == [], "Index meta 前缀必须在且只在一张表里（清除 / 重置 / 保留）"


def test_prefix_guard_detects_unclassified() -> None:
    from bladex_proxy.storage import memory_index as mi

    src = Path(mi.__file__).read_text(encoding="utf-8") + '\n_NEW_THING_PREFIX = "newthing/"\n'
    assert _unclassified(src) == ["_NEW_THING_PREFIX='newthing/' (登记 0 次)"]


def test_harvest_only_takes_class_jia() -> None:
    from bladex_proxy.storage import index_staging
    from bladex_proxy.storage import memory_index as mi

    assert all(mi._META_KEPT.get(p) == "甲" for p in mi._META_HARVESTABLE)
    assert index_staging.HARVEST_PREFIXES == tuple(p.encode() for p in mi._META_HARVESTABLE)


def test_outage_ratio_default_is_001_and_reconciled() -> None:
    from bladex_core.flags import MEMORY_NUMERIC_DEFAULTS

    assert MEMORY_NUMERIC_DEFAULTS["BLADEX_ADJUDICATE_OUTAGE_ABORT_RATIO"] == 0.01
    assert MEMORY_NUMERIC_DEFAULTS["BLADEX_ADJUDICATE_RETRY_MAX"] == 3.0
    text = (_HERE.parents[2] / "config" / ".env.example").read_text(encoding="utf-8")
    for name, val in (("BLADEX_ADJUDICATE_OUTAGE_ABORT_RATIO", 0.01),
                      ("BLADEX_ADJUDICATE_RETRY_MAX", 3.0)):
        m = re.search(rf'^\s*export\s+{name}="?([^"\n]*)"?\s*$', text, re.MULTILINE)
        assert m and float(m.group(1)) == val
