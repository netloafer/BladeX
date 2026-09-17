"""Memory Index（LanceDB）版本/碎片清理 —— 一次性维护仪器（MQ-I15，2026-09-13）。

## 为什么要有它

`bladex status` 报 Memory Index 771.7MB，而真身很小（9,376 facts × 1024 维 ≈ 40MB 向量）。盘上：
`files.lance` 7,668 个版本 / 3,834 个数据文件 / 293MB，`matters.lance` 6,986 / 3,492 / 173MB，`facts.lance` 141 个旧索引目录。
LanceDB 每次写都新建一个版本、旧版本永不自动清；仓库里只有 facts 表按行冗余/碎片数压缩（`compact_fact_vectors`），
matters 只按行冗余压（碎片不算），files 表**没人管**，`cleanup_old_versions` 全仓零调用。⇒ ~600MB 是死重。

## 用法（用户机）—— 🔴 先停 consolidator（它是 Index 的写者；proxy 只读 Index 可以不停）

  bash config/run_consolidator.sh stop            # 或按 bladex 的方式停 consolidator
  .venv/bin/python scripts/index_vacuum.py        # 干跑：只打读数
  .venv/bin/python scripts/index_vacuum.py --apply
  再起 consolidator，`bladex status` 看 Memory Index 体积。

对每张表依次尝试：`optimize(cleanup_older_than=0)` → `optimize()` → `compact_files()`（版本差异，能跑哪个算哪个，与 `compact_fact_vectors` 同款退让）。
09-14 首跑读数：三表都是第一种走通，570→119 MB（files 242→2.9、matters 135→1.1、facts 193→115；facts 旧 `_indices/` 145 个未回收）。
只合并/清理，不重建表、不动行——facts 的向量真身与 meta 的对账另有 `reconcile_vector_meta`，这里不碰。
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packages" / "bladex-core"))
sys.path.insert(0, str(ROOT / "packages" / "bladex-proxy"))


def _du(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def table_stats(tbl_dir: Path) -> dict:
    return {"versions": len(list((tbl_dir / "_versions").glob("*"))) if (tbl_dir / "_versions").exists() else 0,
            "data_files": len(list((tbl_dir / "data").glob("*"))) if (tbl_dir / "data").exists() else 0,
            "index_dirs": len(list((tbl_dir / "_indices").glob("*"))) if (tbl_dir / "_indices").exists() else 0,
            "bytes": _du(tbl_dir)}


def vacuum_table(tbl, keep_s: float = 0.0) -> str:
    """返回实际走通的方法名（读数用）。"""
    attempts = [
        ("optimize", {"cleanup_older_than": timedelta(seconds=keep_s)}),
        ("optimize", {}),
        ("compact_files", {}),
    ]
    used = "none"
    for name, kwargs in attempts:
        fn = getattr(tbl, name, None)
        if fn is None:
            continue
        try:
            fn(**kwargs)
            used = f"{name}({','.join(kwargs)})"
            break
        except Exception as e:  # noqa: BLE001 —— 版本差异
            print(f"      {name} 失败：{e}")
    # 09-14 实跑：`optimize(cleanup_older_than)` 一次走通（570→119 MB）；`cleanup_old_versions` 自 0.21 起 deprecated
    # 且要 pylance，三张表都报错，去掉。未回收的是 facts 的旧 `_indices/` 目录（145 个），optimize 不管它，见 main 的读数。
    return used


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="LanceDB 版本/碎片清理")
    ap.add_argument("--index", default=os.environ.get("BLADEX_INDEX_PATH", "data/bladex_index"))
    ap.add_argument("--apply", action="store_true", help="不带此参只打读数")
    ap.add_argument("--keep-seconds", type=float, default=0.0, help="保留多新的旧版本（默认全清）")
    a = ap.parse_args(argv)
    root = (ROOT / a.index) if not os.path.isabs(a.index) else Path(a.index)
    ldir = root / "lancedb"
    if not ldir.exists():
        print(f"没有 {ldir}")
        return 2
    tables = sorted(p for p in ldir.glob("*.lance") if p.is_dir())
    print(f"Index {root} · lancedb 总 {_du(ldir)/1e6:.0f} MB")
    before = {t.name: table_stats(t) for t in tables}
    for n, s in before.items():
        print(f"  {n:16s} versions={s['versions']:5d} data_files={s['data_files']:5d} index_dirs={s['index_dirs']:4d} {s['bytes']/1e6:7.1f} MB")
    if not a.apply:
        print("干跑结束（加 --apply 执行；先停 consolidator）")
        return 0
    import lancedb  # noqa: WPS433 —— 与 memory_index 同一依赖
    db = lancedb.connect(str(ldir))
    for t in tables:
        name = t.name[:-len(".lance")]
        try:
            tbl = db.open_table(name)
        except Exception as e:  # noqa: BLE001
            print(f"  {name}: 打不开（{e}），跳过")
            continue
        rows = tbl.count_rows()
        used = vacuum_table(tbl, a.keep_seconds)
        after = table_stats(t)
        print(f"  {name:16s} rows={rows} 走通={used} → versions {before[t.name]['versions']}→{after['versions']} "
              f"files {before[t.name]['data_files']}→{after['data_files']} idx {before[t.name]['index_dirs']}→{after['index_dirs']} "
              f"{before[t.name]['bytes']/1e6:.1f}→{after['bytes']/1e6:.1f} MB")
    print(f"lancedb 总 {_du(ldir)/1e6:.0f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
