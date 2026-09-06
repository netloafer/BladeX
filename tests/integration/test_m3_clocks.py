"""M3-1：相关钟从注入面一路通到 `last_hit_at`（复核 5.9 三时钟）。

三个钟里这一个**完全缺失**：`injected_fact_ids` 只记散点条目，
Matter 卡注入零留痕、连计数都没有。于是 2×2 矩阵的"久无命中"那一列恒为真，
dormant 判定要么不敢开、要么开了就误杀。

X1 说的"全系统没有一个学习回路是通的"，断点之一就在这里 ——
所以这条链必须端到端测，不能只测两端。
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from bladex_proxy.models import DecisionMeta, Identity, Turn, TurnStatus
from bladex_proxy.storage.memory_hub import MemoryHub
from bladex_proxy.storage.memory_index import MemoryIndex


class _Embedder:
    _DIM = 32

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()
            out.append([1.0] + [(b / 255.0) * 0.5 for b in h[:self._DIM - 1]])
        return out

    @property
    def available(self) -> bool:
        return True


def _turn(idx: int, *, session: str, fact_ids: list[str] = (),
          matter_ids: list[str] = (), ts: str = "") -> tuple[str, Turn]:
    ident = Identity(user_id="u1", agent_id="a1", session_id=session, turn_index=idx)
    t = Turn(
        identity=ident, model="m",
        request_messages=[{"role": "user", "content": f"第 {idx} 轮的用户提问内容若干字"}],
        response_text="ok", status=TurnStatus.OK, logical_turn=idx,
        decision_meta=DecisionMeta(injected_fact_ids=list(fact_ids),
                                   injected_matter_ids=list(matter_ids)),
    )
    if ts:
        t.ts = datetime.fromisoformat(ts)
    return ident.storage_key(f"17199072{idx:02d}-0"), t


def _index(tmp: Path) -> MemoryIndex:
    idx = MemoryIndex(tmp / "index", embedder=_Embedder(), distiller=None)
    idx.open()
    return idx


def _seed_fact(idx: MemoryIndex, fid: str) -> None:
    from bladex_core.fact import Fact

    f = Fact(id=fid, content=f"被注入过的条目 {fid}", scope="personal:u1")
    f.embedding = _Embedder().embed([f.content])[0]
    idx.add_fact(f)


def _seed_matter(idx: MemoryIndex, mid: str) -> None:
    from bladex_core.matter import Matter

    idx.add_matter(Matter(matter_id=mid, title=f"卡 {mid}"))


def test_fact_last_hit_at_is_written_from_turn_time(tmp_path):
    """🔴 相关钟取 **turn 时间**，不取 `now` —— 重放两次必须逐字节一致（G6）。"""
    led = MemoryHub(tmp_path / "ledger")
    led.open()
    k, t = _turn(0, session="s1", fact_ids=["f_hit"],
                 ts="2026-07-01T10:00:00+00:00")
    led.put(k, t)

    idx = _index(tmp_path)
    _seed_fact(idx, "f_hit")
    idx.rebuild_from_hub(led, full=False, max_turns=50)

    f = idx.get_fact("f_hit")
    assert f is not None and f.last_hit_at is not None, "相关钟没有被写入 —— 链断了"
    assert f.last_hit_at.isoformat() == "2026-07-01T10:00:00+00:00", (
        "取了 now 而不是 turn 时间 —— 重放两次会得到不同的库")
    assert f.ref_count >= 1
    idx.close()
    led.close()


def test_matter_last_hit_at_is_written(tmp_path):
    """Matter 卡的相关钟 —— 三个钟里此前唯一没有信号源的那个。"""
    led = MemoryHub(tmp_path / "ledger")
    led.open()
    k, t = _turn(0, session="s1", matter_ids=["m_pinned"],
                 ts="2026-07-02T09:00:00+00:00")
    led.put(k, t)

    idx = _index(tmp_path)
    _seed_matter(idx, "m_pinned")
    idx.rebuild_from_hub(led, full=False, max_turns=50)

    m = idx.get_matter("m_pinned")
    assert m is not None and m.last_hit_at is not None, (
        "Matter 相关钟没写入 —— injected_matter_ids 这条链断了")
    assert m.last_hit_at.isoformat() == "2026-07-02T09:00:00+00:00"
    idx.close()
    led.close()


def test_matter_hit_does_not_bump_version(tmp_path):
    """🔴 命中回写**不许 bump `version`**。

    version 是 U7 ③平面 changed-since 的判据。被"有人看了一眼"推高，
    每轮都会判"有变更"，增量注入退化成全量重注 —— 正是 H6 里
    version churn 消耗注入 cache 的那个形态。
    """
    led = MemoryHub(tmp_path / "ledger")
    led.open()
    for i in range(3):
        led.put(*_turn(i, session=f"s{i}", matter_ids=["m_x"],
                       ts=f"2026-07-0{i + 1}T09:00:00+00:00"))

    idx = _index(tmp_path)
    _seed_matter(idx, "m_x")
    before = idx.get_matter("m_x").version
    idx.rebuild_from_hub(led, full=False, max_turns=50)
    after = idx.get_matter("m_x").version

    assert after == before, f"命中把 version 从 {before} 推到了 {after}"
    idx.close()
    led.close()


def test_hits_are_deduped_within_a_session(tmp_path):
    """🔴 会话内去重（复核 X1/G5 的"机会归一"护栏）。

    不去重的话，一个工具循环里连注 30 轮的条目拿到 30 次命中，
    而在别的会话被真正用上一次的只有 1 次 ——
    相关钟会把"碰巧落在长会话里"读成"更有相关性"（富者愈富）。
    """
    led = MemoryHub(tmp_path / "ledger")
    led.open()
    # 同一会话注 5 轮 + 另一会话注 1 轮 → 期望计 2 次，不是 6 次
    for i in range(5):
        led.put(*_turn(i, session="s_long", fact_ids=["f_a"],
                       ts=f"2026-07-0{i + 1}T09:00:00+00:00"))
    led.put(*_turn(9, session="s_other", fact_ids=["f_a"],
                   ts="2026-07-09T09:00:00+00:00"))

    idx = _index(tmp_path)
    _seed_fact(idx, "f_a")
    idx.rebuild_from_hub(led, full=False, max_turns=50)

    f = idx.get_fact("f_a")
    assert f.ref_count == 2, f"会话内没去重，ref_count={f.ref_count}（应为 2）"
    idx.close()
    led.close()


def test_last_hit_at_only_moves_forward(tmp_path):
    """乱序重放也稳定：只前进不后退。"""
    led = MemoryHub(tmp_path / "ledger")
    led.open()
    led.put(*_turn(0, session="s1", fact_ids=["f_b"], matter_ids=["m_b"],
                   ts="2026-07-20T09:00:00+00:00"))
    led.put(*_turn(1, session="s2", fact_ids=["f_b"], matter_ids=["m_b"],
                   ts="2026-07-05T09:00:00+00:00"))   # 更早

    idx = _index(tmp_path)
    _seed_fact(idx, "f_b")
    _seed_matter(idx, "m_b")
    idx.rebuild_from_hub(led, full=False, max_turns=50)

    assert idx.get_fact("f_b").last_hit_at.isoformat().startswith("2026-07-20")
    assert idx.get_matter("m_b").last_hit_at.isoformat().startswith("2026-07-20")
    idx.close()
    led.close()


def test_no_injection_means_no_clock_movement(tmp_path):
    """没被注入过的条目，相关钟保持 None —— 它是 M3-2 观测期护栏的输入。"""
    led = MemoryHub(tmp_path / "ledger")
    led.open()
    led.put(*_turn(0, session="s1", ts="2026-07-01T10:00:00+00:00"))

    idx = _index(tmp_path)
    _seed_fact(idx, "f_cold")
    _seed_matter(idx, "m_cold")
    idx.rebuild_from_hub(led, full=False, max_turns=50)

    assert idx.get_fact("f_cold").last_hit_at is None
    assert idx.get_matter("m_cold").last_hit_at is None
    idx.close()
    led.close()


# ── 注入面：四条分支都要产出 injected_matter_ids ─────────────────────────


def test_every_inject_branch_defines_matter_ids():
    """🔴 回归红线：`injected_matter_ids` 在**每一条分支**都要有定义。

    开发时同步路径补齐了、异步路径的 aux 分支漏了 —— 测试没覆盖到，
    但生产上 aux 轮走异步路径会直接 `UnboundLocalError` 打断注入。
    四条分支：aux / 正常 / 超预算降级 / 异常降级。

    用 AST 静态查而不是跑四遍：跑不到的那条恰恰是会出事的那条。
    """
    import ast
    from pathlib import Path as _P

    src = _P(__file__).resolve().parents[2] / "packages/bladex-proxy/bladex_proxy/inject.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    checked = 0
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        names = [n for n in ast.walk(fn)
                 if isinstance(n, ast.Name) and n.id == "injected_matter_ids"]
        if not names:
            continue
        stores = sum(1 for n in names if isinstance(n.ctx, ast.Store))
        assert stores >= 4, (
            f"{fn.name} 只有 {stores} 处赋值，少于四条分支"
            "（aux / 正常 / 超预算 / 异常）—— 漏掉的那条会 UnboundLocalError")
        checked += 1
    assert checked == 2, f"应覆盖 do_inject 与 do_inject_async，实得 {checked}"
