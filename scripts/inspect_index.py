"""Memory Index 巡检 + 水位游标诊断 —— 看 consolidator 到底派生了什么，以及为什么没派生。

对称于 scripts/inspect_hub.py（Memory Hub 看会话；Memory Index 看蒸馏出的 Fact / Matter / Edge）。

本机跑（uv 环境；只读打开 Memory Index/Memory Hub，可与 proxy / consolidator 进程共存）：

  # 1) 概览：Matter / Edge / Fact 计数 + watermark 值 + Fact 样例
  uv run python scripts/inspect_index.py

  # 2) 列 Matter（含成员 edge 数 / 状态 / 摘要）
  uv run python scripts/inspect_index.py --matters

  # 3) 列 Fact（可加 --limit N / --grep 子串）
  uv run python scripts/inspect_index.py --facts --limit 50
  uv run python scripts/inspect_index.py --grep 世界杯

  # 4) 水位游标诊断：watermark 是否把真实数据挡在门外
  #    （对照 Memory Hub：统计 key<=watermark 被跳过 vs key>watermark 被处理，
  #     并确认 watermark 是否 == Memory Hub 字典序最大 key = 中毒特征）
  uv run python scripts/inspect_index.py --watermark

Memory Index 路径默认读环境变量 BLADEX_INDEX_PATH，否则 data/bladex_index。
Memory Hub 路径默认读环境变量 BLADEX_ROCKSDB_PATH，否则 data/bladex_hub（仅 --watermark 用）。

背景：2026-07-07 定位到 watermark 中毒（脏 key 顶到字典序最大值 → 所有真实轮次每轮被判"已消费"跳过）。
详见 Memory Index 水位游标中毒问题复盘 20260707。
"""

from __future__ import annotations

import argparse
import os
from collections import Counter

import msgpack
from rocksdict import AccessType, Options, Rdict

_FACT_PREFIX = "fact/"
_MATTER_PREFIX = "matter/"
_EDGE_PREFIX = "edge/"
_WATERMARK_KEY = "__meta__/p2_watermark"


def _index_meta_path() -> str:
    base = os.environ.get("BLADEX_INDEX_PATH", "data/bladex_index")
    return os.path.join(base, "meta_rocksdb")


def _ledger_path() -> str:
    return os.environ.get("BLADEX_ROCKSDB_PATH", "data/bladex_hub")


def _open_ro(path: str) -> Rdict:
    """只读打开一个 RocksDB（不取 LOCK，可与写进程共存）。"""
    return Rdict(path, Options(), None, AccessType.read_only())


def _decode(v: object) -> dict:
    try:
        return msgpack.unpackb(v, raw=False)
    except Exception:
        return {}


def _get_watermark(db: Rdict) -> str | None:
    for k in db.keys():
        ks = k.decode(errors="replace") if isinstance(k, (bytes, bytearray)) else str(k)
        if ks == _WATERMARK_KEY:
            v = db[k]
            if isinstance(v, (bytes, bytearray)):
                try:
                    return msgpack.unpackb(v, raw=False)
                except Exception:
                    return v.decode(errors="replace")
            return str(v)
    return None


def cmd_overview(db: Rdict) -> None:
    matters = edges = facts = 0
    fact_samples: list[dict] = []
    for k in db.keys():
        ks = k.decode(errors="replace")
        if ks.startswith(_MATTER_PREFIX):
            matters += 1
        elif ks.startswith(_EDGE_PREFIX):
            edges += 1
        elif ks.startswith(_FACT_PREFIX):
            facts += 1
            if len(fact_samples) < 8:
                fact_samples.append(_decode(db[k]))
    print("=== Memory Index 概览 ===")
    print(f"  Matter: {matters}")
    print(f"  Edge:   {edges}")
    print(f"  Fact:   {facts}")
    print(f"  watermark: {_get_watermark(db)}")
    if facts == 0:
        print("\n⚠ 0 条 Fact —— consolidator 从未成功派生，或被 watermark 挡住（跑 --watermark 诊断）。")
    if matters == 0 and facts > 0:
        print("\n⚠ 有 Fact 但 0 个 Matter —— 归属管线未建 Matter（检查 attribution / ADR-0012 接线）。")
    if fact_samples:
        print("\n--- Fact 样例 ---")
        for d in fact_samples:
            c = (d.get("content") or "").replace("\n", " ")[:70]
            print(f"  {d.get('id', '?')}  [{d.get('created_at', '?')}]  {c}")


def cmd_matters(db: Rdict) -> None:
    edge_by_matter: Counter[str] = Counter()
    for k in db.keys():
        ks = k.decode(errors="replace")
        if ks.startswith(_EDGE_PREFIX):
            edge_by_matter[_decode(db[k]).get("matter_id", "?")] += 1
    rows = []
    for k in db.keys():
        ks = k.decode(errors="replace")
        if ks.startswith(_MATTER_PREFIX):
            d = _decode(db[k])
            rows.append(d)
    if not rows:
        print("（无 Matter）")
        return
    rows.sort(key=lambda d: -edge_by_matter.get(d.get("matter_id", ""), 0))
    print(f"=== Matter ×{len(rows)} ===")
    for d in rows:
        mid = d.get("matter_id", "?")
        title = d.get("title") or d.get("summary") or ""
        print(f"  {mid[:18]}  edges={edge_by_matter.get(mid, 0):>3}  "
              f"status={d.get('status')}  origin={d.get('origin')}  '{str(title)[:60]}'")


def cmd_facts(db: Rdict, limit: int, grep: str | None) -> None:
    n = 0
    for k in db.keys():
        ks = k.decode(errors="replace")
        if not ks.startswith(_FACT_PREFIX):
            continue
        d = _decode(db[k])
        content = d.get("content") or ""
        if grep and grep not in content:
            continue
        tags = d.get("tags", "")
        print(f"  {d.get('id', '?')}  [{d.get('created_at', '?')}]  {tags}")
        print(f"    {content.replace(chr(10), ' ')[:120]}")
        n += 1
        if n >= limit:
            break
    if n == 0:
        print("（无匹配 Fact）")


def cmd_watermark(index_db: Rdict) -> None:
    """诊断 watermark 是否把真实数据挡在门外（对照 Memory Hub）。"""
    wm = _get_watermark(index_db)
    print(f"watermark = {wm}")
    if not wm:
        print("watermark 为空 —— 下次 rebuild 会全量处理（无中毒）。")
        return
    ledger = _open_ro(_ledger_path())
    below = above = 0
    max_key = ""
    real_below = real_above = 0
    for k in ledger.keys():
        ks = k.decode(errors="replace")
        if ks.startswith("__"):
            continue
        if ks > max_key:
            max_key = ks
        is_real = ks.startswith("53136499/")  # 真实用户；按需改
        if ks <= wm:
            below += 1
            real_below += is_real
        else:
            above += 1
            real_above += is_real
    ledger.close()
    print(f"  Memory Hub key <= watermark（本轮跳过）: {below}  (真实用户 53136499: {real_below})")
    print(f"  Memory Hub key >  watermark（本轮处理）: {above}  (真实用户 53136499: {real_above})")
    print(f"  Memory Hub 字典序最大 key: {max_key}")
    if wm == max_key:
        print("\n🔴 中毒特征命中：watermark == Memory Hub 最大 key → 每轮增量都判'无新增'空转，"
              "所有更小字典序的真实数据被永久跳过。")
        print("   Memory Hub key 形如 user/agent/…（非时间序），字典序 ≠ 插入序；"
              "脏 key（如测试/压测的高位 user 段）顶起水位即中毒。")
        print("   修：治标 rebuild_from_hub(full=True) 或复位 watermark；"
              "治本改时间序/分维游标 + 清脏 key。见 Memory Index 水位游标中毒问题复盘 20260707")


def main() -> int:
    ap = argparse.ArgumentParser(description="BladeX Memory Index 巡检 + 水位游标诊断")
    ap.add_argument("--matters", action="store_true", help="列 Matter")
    ap.add_argument("--facts", action="store_true", help="列 Fact")
    ap.add_argument("--grep", type=str, default=None, help="Fact content 子串过滤（隐含 --facts）")
    ap.add_argument("--limit", type=int, default=30, help="--facts 上限")
    ap.add_argument("--watermark", action="store_true", help="水位游标诊断（对照 Memory Hub）")
    args = ap.parse_args()

    meta_path = _index_meta_path()
    if not os.path.exists(meta_path):
        print(f"Memory Index meta 库不存在：{meta_path}（consolidator 从未跑过？）")
        return 1
    db = _open_ro(meta_path)
    try:
        if args.watermark:
            cmd_watermark(db)
        elif args.matters:
            cmd_matters(db)
        elif args.facts or args.grep:
            cmd_facts(db, args.limit, args.grep)
        else:
            cmd_overview(db)
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
