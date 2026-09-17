"""F0.5（2026-09-06）：生产源文件行数守卫——巨石不许再长回来。

批 F0.1 把 `agency.py`（2,365）/ `server.py`（4,141）/ `cli.py`（3,465）拆成包；这条守卫钉住
"拆完之后不许再堆"：`packages/*/bladex_*/**/*.py`（排除 tests / archive）任一文件 > 2,500 行即红。
白名单只收**已排期拆分**的文件，且每项带到期说明；F0.1 做完后 server / agency / cli 三个都不许在白名单里。
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LIMIT = 2500

#: 白名单：相对仓库根路径 → 到期说明（谁、何时拆）。只减不增。
_BASELINE: dict[str, str] = {
    "packages/bladex-proxy/bladex_proxy/storage/memory_index.py":
        "F0.6 单独一棒（E2 后）拆分；7,880 行，是唯一的存量巨石",
}


def _prod_files() -> list[Path]:
    out: list[Path] = []
    for pkg in sorted((REPO / "packages").glob("*/bladex_*")):
        if not pkg.is_dir():
            continue
        for p in pkg.rglob("*.py"):
            parts = set(p.parts)
            if "tests" in parts or "archive" in parts or "__pycache__" in parts:
                continue
            out.append(p)
    assert len(out) >= 80, f"只扫到 {len(out)} 个生产文件，守卫自己漂了"
    return out


def _lines(p: Path) -> int:
    return p.read_text(encoding="utf-8").count("\n")


def test_no_production_file_exceeds_limit() -> None:
    offenders = {
        str(p.relative_to(REPO)): _lines(p)
        for p in _prod_files()
        if _lines(p) > LIMIT and str(p.relative_to(REPO)) not in _BASELINE
    }
    assert not offenders, (
        f"超过 {LIMIT} 行的生产文件（拆分或先登记白名单并带到期说明）：{offenders}")


def test_baseline_entries_are_still_oversized_and_scheduled() -> None:
    """白名单只减不增：已经拆到线下的项必须从白名单删掉；每项都要有到期说明。"""
    for rel, why in _BASELINE.items():
        p = REPO / rel
        assert p.is_file(), f"白名单项不存在：{rel}"
        assert _lines(p) > LIMIT, f"{rel} 已在 {LIMIT} 行以下，从白名单删掉"
        assert len(why) >= 8 and ("F0.6" in why or "拆" in why), f"{rel} 的到期说明太短或没写谁来拆"


def test_the_three_split_monoliths_are_not_whitelisted() -> None:
    for rel in ("packages/bladex-proxy/bladex_proxy/server.py",
                "packages/bladex-proxy/bladex_proxy/agency.py",
                "packages/bladex-proxy/bladex_proxy/cli.py"):
        assert rel not in _BASELINE
        assert not (REPO / rel).exists(), f"{rel} 又回来了——F0.1 已把它拆成包"
    for pkg in ("server", "agency", "cli"):
        assert (REPO / "packages/bladex-proxy/bladex_proxy" / pkg / "__init__.py").is_file()
