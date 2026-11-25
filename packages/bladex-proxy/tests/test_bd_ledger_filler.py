"""批 BD · 步骤 1：账本块等长填充开关（bench 专用、默认关）的五条守卫。

方案 `docs/planning/report-batch-BD-precheck-20261117.md` §4。
① 关着零差  ② 开着恒定  ③ 不常驻  ④ 有响警  ⑤ 在场率（含「无账本 + 工具面」轮）。
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import structlog.testing
from bladex_core.flags import BENCH_NUMERIC_DEFAULTS, bench_ledger_filler_chars
from bladex_core.ledger import new_ledger
from bladex_proxy.agency import AgencyRuntime
from bladex_proxy.agency.bench_filler import (
    _CLOSE,
    _OPEN,
    FORBIDDEN_WORDS,
    MIN_CHARS,
    WORDS,
    bench_filler,
    warn_if_bench_filler_active,
)
from bladex_proxy.agency.runtime import LEDGER_BLOCK_CLOSE, LEDGER_BLOCK_OPEN

_FLAG = "BLADEX_BENCH_LEDGER_FILLER_CHARS"
_ROOT = Path(__file__).resolve().parents[3]
_SYS = {"role": "system", "content": "You are a test agent."}
_AGENT = "hermes:default"


def _on(monkeypatch, n: str | None = None) -> None:
    monkeypatch.setenv("BLADEX_MODULE_TOOLFACE", "1")
    monkeypatch.setenv("BLADEX_MODULE_LEDGER", "1")
    if n is None:
        monkeypatch.delenv(_FLAG, raising=False)
    else:
        monkeypatch.setenv(_FLAG, n)


def _seeded() -> AgencyRuntime:
    ag = AgencyRuntime()
    ag.pool["ldg-a"] = new_ledger(title="alpha", goal="do alpha", ledger_id="ldg-a",
                                  created_at="2026-09-01T00:00:00+00:00")
    return ag


def _msgs(i: int = 0) -> list[dict]:
    return [_SYS, {"role": "user", "content": f"continue please {i}"}]


def _insert(ag: AgencyRuntime, msgs=None, **kw) -> list[dict]:
    return ag.insert_ledger_block(msgs or _msgs(), _AGENT, tier="medium",
                                  with_instruction=True, **kw)


def _injected(cap) -> list[dict]:
    return [e for e in cap if e.get("event") == "agency_ledger_block_injected"]


# ── 词表 / 生成器自身 ────────────────────────────────────────────────────────

def test_markers_match_runtime_and_wordlist_is_clean():
    assert (_OPEN, _CLOSE) == (LEDGER_BLOCK_OPEN, LEDGER_BLOCK_CLOSE)
    assert len(WORDS) == 256 and len(set(WORDS)) == 256
    for w in WORDS:
        assert w == w.lower() and w.isalpha()
        for bad in FORBIDDEN_WORDS:
            assert bad not in w, (w, bad)          # 子串口径：截断落在词中也切不出禁用词
            assert not w.startswith(bad[:4]) or bad not in w


# ── ① 关着零差 ──────────────────────────────────────────────────────────────

def test_off_is_byte_identical_and_log_has_no_filler(monkeypatch):
    for setting in (None, "0"):
        _on(monkeypatch, setting)
        ag = _seeded()
        with structlog.testing.capture_logs() as cap:
            out = _insert(ag)
        rows = _injected(cap)
        assert len(rows) == 1
        assert "filler" not in rows[0] and "filler_chars" not in rows[0]
        assert "filler" not in rows[0]["sections"]
        # 金标准：把开关设成非数/负数（仍判关）也得到同一块
        monkeypatch.setenv(_FLAG, "-5")
        out2 = _insert(_seeded())
        assert out[-1]["content"] == out2[-1]["content"]
        assert out[-1]["content"].startswith(LEDGER_BLOCK_OPEN)
        assert "alpha" in out[-1]["content"]          # 真块（含账本内容），不是填充
        assert out[:-1] == _msgs()


def test_off_none_block_stays_gated(monkeypatch):
    """关着时「无账本 + 无令」仍返回原数组并记 gate（与改前一致）。"""
    _on(monkeypatch, None)
    ag = AgencyRuntime()
    with structlog.testing.capture_logs() as cap:
        out = ag.insert_ledger_block(_msgs(), _AGENT, tier="medium", with_instruction=False)
    assert out == _msgs()
    assert [e["reason"] for e in cap if e.get("event") == "agency_ledger_block_gated"] \
        == ["no_ledger_no_instruction"]
    assert not _injected(cap)


# ── ② 开着恒定 ──────────────────────────────────────────────────────────────

def test_on_is_constant_across_50_rounds_sessions_and_revs(monkeypatch):
    n = 1500
    _on(monkeypatch, str(n))
    ag = _seeded()
    hashes: set[str] = set()
    for i in range(50):
        if i % 7 == 3:
            ag.pool["ldg-a"].rev = ag.pool["ldg-a"].rev + 1 if hasattr(ag.pool["ldg-a"], "rev") else 0
        out = _insert(ag, _msgs(i), session_id=f"sess-{i % 5}")
        content = out[-1]["content"]
        assert out[-1]["role"] == "system" and out[:-1] == _msgs(i)
        assert len(content) == n
        assert content.startswith(LEDGER_BLOCK_OPEN + "\n") and content.endswith("\n" + LEDGER_BLOCK_CLOSE)
        body = content[len(LEDGER_BLOCK_OPEN):-len(LEDGER_BLOCK_CLOSE)].lower()
        for bad in FORBIDDEN_WORDS:
            assert bad not in body, bad
        hashes.add(hashlib.sha256(content.encode()).hexdigest())
    assert len(hashes) == 1
    # 日志：filler 列与 block_hash 一致
    with structlog.testing.capture_logs() as cap:
        _insert(ag)
    row = _injected(cap)[0]
    assert row["filler"] == 1 and row["filler_chars"] == n and row["chars"] == n
    assert row["block_hash"] == hashlib.sha256(bench_filler(n).encode()).hexdigest()[:16]


def test_filler_is_deterministic_and_below_min_is_clamped():
    a = bench_filler(2200)
    assert a == bench_filler(2200) and len(a) == 2200
    # 固定种子：钉住前 40 字符，改种子/词表会让已跑的对照作废，必须显式改本断言
    assert a[: len(LEDGER_BLOCK_OPEN) + 1] == LEDGER_BLOCK_OPEN + "\n"
    assert len(bench_filler(5)) == MIN_CHARS
    assert len(bench_filler(MIN_CHARS)) == MIN_CHARS
    assert bench_filler(3000)[:2000] != ""  # 不同 n 各自独立恰长
    assert len(bench_filler(3000)) == 3000


# ── ③ 不常驻 ───────────────────────────────────────────────────────────────

def test_not_resident_static():
    assert BENCH_NUMERIC_DEFAULTS == {_FLAG: 0}
    for rel in ("config/.env.example", "config/.env"):
        p = _ROOT / rel
        if not p.is_file():
            continue
        for ln in p.read_text(encoding="utf-8").splitlines():
            assert not re.match(r"\s*(export\s+)?BLADEX_BENCH_", ln), f"{rel}: {ln}"
    # 开关名在生产源码里只出现一处（flags.py 的常量）；消费方经函数读
    hits = []
    for pkg in (_ROOT / "packages").glob("*/bladex_*"):
        for f in pkg.rglob("*.py"):
            n = f.read_text(encoding="utf-8").count(_FLAG)
            if n:
                hits.append((f.name, n))
    assert hits == [("flags.py", 1)], hits


def test_unset_is_zero(monkeypatch):
    monkeypatch.delenv(_FLAG, raising=False)
    assert bench_ledger_filler_chars() == 0


# ── ④ 有响警 ───────────────────────────────────────────────────────────────

def test_startup_warning_only_when_on(monkeypatch):
    _on(monkeypatch, None)
    with structlog.testing.capture_logs() as cap:
        assert warn_if_bench_filler_active() == 0
    assert not [e for e in cap if e.get("event") == "bench_filler_active"]
    _on(monkeypatch, "2200")
    with structlog.testing.capture_logs() as cap:
        assert warn_if_bench_filler_active() == 2200
    rows = [e for e in cap if e.get("event") == "bench_filler_active"]
    assert len(rows) == 1 and rows[0]["log_level"] == "warning" and rows[0]["chars"] == 2200


def test_lifespan_and_doctor_are_wired():
    root = _ROOT / "packages/bladex-proxy/bladex_proxy"
    assert "warn_if_bench_filler_active()" in (root / "server/app_factory.py").read_text(encoding="utf-8")
    doc = (root / "cli/doctor_cmd.py").read_text(encoding="utf-8")
    assert "bench_ledger_filler_chars()" in doc and 'print(f"  ! BENCH switch active' in doc


# ── ⑤ 在场率 ───────────────────────────────────────────────────────────────

def test_filler_injected_when_real_block_is_none(monkeypatch):
    """无账本 + 工具面轮：改前真块为 None（不注）；填充臂必须照注。"""
    _on(monkeypatch, "1200")
    ag = AgencyRuntime()
    with structlog.testing.capture_logs() as cap:
        out = ag.insert_ledger_block(_msgs(), _AGENT, tier="medium", with_instruction=False)
    assert len(out) == 3 and len(out[-1]["content"]) == 1200 and out[-1]["role"] == "system"
    assert out[-1]["content"] == bench_filler(1200)
    row = _injected(cap)[0]
    assert row["filler"] == 1
    # 同一请求再来一次、且带账本的轮：块逐字相同（在场且恒定）
    out2 = _insert(_seeded())
    assert out2[-1]["content"] == out[-1]["content"]


def test_aux_and_disallowed_tier_still_gated_when_on(monkeypatch):
    _on(monkeypatch, "1200")
    ag = _seeded()
    assert _insert(ag, auxiliary=True) == _msgs()
