"""一次性压缩 Memory Index 的 LanceDB 表（碎片 + 历史版本清理）。

背景（2026-08-09 实测）：facts.lance 72MB / 1086 个碎片文件对应 ~1015 行
（真实向量内容 ~4MB）；files.lance 36MB / 1860 个版本；matters.lance 6.6MB /
560 个版本。追加写一行一个碎片、历史版本从不清理——110.7MB 里约 100MB 是卫生债。
`maybe_compact_fact_vectors` 的触发条件量错了轴（rows > facts×2 抓的是僵尸行，
不抓碎片），files 表则根本没有压缩作业。

**2026-08-17 更新（MQ-R1）**：facts 表的代码侧根治已落地——碎片数超
`BLADEX_INDEX_MAX_FRAGMENTS`（默认 100）即在 consolidation pass 收尾自动压。
"自维护待开发窗"那一年欠账的代价是：08-16 一次全量重建把 1619 条 fact 写成
1619 个碎片，检索慢 15 倍，注入面空转一个月。
本脚本仍有用：**报数**（dry-run 看各表碎片/版本/体积）、**手工急救**、
以及 facts 之外的表（matters / files 仍无自动作业）。

沙盒对整库副本实测（lancedb 0.33）：115MB → 15MB，ANN 索引保留、检索正常。

用法（🔴 必须先停 proxy 与 consolidator——cleanup 会删掉读进程还握着的旧版本）：

    bladex stop
    .venv/bin/python scripts/compact_index.py
    .venv/bin/python scripts/compact_index.py --apply
    bladex start

默认 dry-run 只打印各表行数/版本数/体积，不动任何文件。
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import timedelta

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _du(path: str) -> int:
    total = 0
    for dirpath, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return total


def _mb(n: int) -> str:
    return f"{n / 1024 / 1024:.1f}MB"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default=os.environ.get(
        "BLADEX_INDEX_PATH", "data/bladex_index"))
    ap.add_argument("--apply", action="store_true",
                    help="执行 optimize + 版本清理（默认 dry-run 只报数）")
    args = ap.parse_args()

    idx = args.index if os.path.isabs(args.index) else os.path.join(_ROOT, args.index)
    lance_dir = os.path.join(idx, "lancedb")
    if not os.path.isdir(lance_dir):
        print(f"⛔ 没找到 {lance_dir}")
        return 1

    import lancedb

    db = lancedb.connect(lance_dir)
    # 0.33 的 list_tables() 返回的不是 list[str]（实测踩到）；
    # 直接按目录名取表——它就是磁盘上的事实，不猜 API。
    names = sorted(d[:-len(".lance")] for d in os.listdir(lance_dir)
                   if d.endswith(".lance")
                   and os.path.isdir(os.path.join(lance_dir, d)))
    total_before = _du(lance_dir)
    print(f"库目录: {lance_dir}  总体积: {_mb(total_before)}\n")
    for name in names:
        tdir = os.path.join(lance_dir, f"{name}.lance")
        versions = len(os.listdir(os.path.join(tdir, "_versions"))) \
            if os.path.isdir(os.path.join(tdir, "_versions")) else -1
        datafiles = len(os.listdir(os.path.join(tdir, "data"))) \
            if os.path.isdir(os.path.join(tdir, "data")) else -1
        tb = db.open_table(name)
        rows = tb.count_rows()
        print(f"== {name}: rows={rows} versions={versions} "
              f"datafiles={datafiles} size={_mb(_du(tdir))}")
        if not args.apply:
            continue
        # lancedb 0.33：optimize(cleanup_older_than=0) = 合并碎片 + 清历史版本，
        # ANN 索引保留（沙盒副本实测验证）。
        tb.optimize(cleanup_older_than=timedelta(0))
        print(f"   -> compacted: versions="
              f"{len(os.listdir(os.path.join(tdir, '_versions')))} "
              f"datafiles={len(os.listdir(os.path.join(tdir, 'data')))} "
              f"size={_mb(_du(tdir))}")
    if args.apply:
        print(f"\n总体积: {_mb(total_before)} -> {_mb(_du(lance_dir))}")
    else:
        print("\n（dry-run。加 --apply 执行；执行前先 `bladex stop`。）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
