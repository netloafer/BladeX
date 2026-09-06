"""规模层阈值校准 —— 从 Memory Hub 统计真实流量的 context_chars 分布（RA 修复卡 T7 校准步骤）。

背景（routing-fix-plan-20260712 + 2026-07-14 首批真实流量观察）：
Hermes 带工具会话的上下文轻松超过 floor_medium_chars=60000 占位阈值
（实测 sys ~23K + tool 结果 95–149K → 主流量 100% 触发 medium 下限、裁判被 bypass）。
本脚本给"先采样再定"提供数据：跑几天流量后执行，看分布再重标阈值。

本机跑（uv 环境；只读打开 Memory Hub，可与 proxy 进程共存）：

  # 概览：主流量/aux 的 context_chars 分位数 + 各候选阈值下的档位分布
  uv run python scripts/route_scale_calibration.py

  # 只看最近 N 天
  uv run python scripts/route_scale_calibration.py --days 3

  # 自定义候选阈值（逗号分隔）
  uv run python scripts/route_scale_calibration.py --thresholds 60000,100000,200000,300000

Memory Hub 路径默认读环境变量 BLADEX_ROCKSDB_PATH，否则 data/bladex_hub。
统计口径与热路径一致（server._estimate_context_chars：str 全长、list 只计 text、
非文本块 1000 当量、tool_calls 参数计入）。注意 Memory Hub 存的是 agent 原文 messages
（注入/CAP 前），比热路径统计（注入后）略小——差值 = 注入块 + CAP 影响，量级不影响定档。
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from datetime import UTC, datetime, timedelta

from bladex_proxy.server import _estimate_context_chars
from bladex_proxy.storage.memory_hub import MemoryHub


def _percentiles(values: list[int], pts: tuple[int, ...] = (10, 25, 50, 75, 90, 95, 99)) -> dict[int, int]:
    if not values:
        return {}
    s = sorted(values)
    return {p: s[min(len(s) - 1, int(len(s) * p / 100))] for p in pts}


def main() -> None:
    parser = argparse.ArgumentParser(description="规模层阈值校准（Memory Hub context_chars 分布）")
    parser.add_argument("--days", type=int, default=0, help="只统计最近 N 天（0=全部）")
    parser.add_argument(
        "--thresholds", type=str, default="60000,100000,150000,200000,300000",
        help="候选 floor 阈值，逗号分隔",
    )
    args = parser.parse_args()

    thresholds = [int(x) for x in args.thresholds.split(",") if x.strip()]
    cutoff = None
    if args.days > 0:
        cutoff = datetime.now(UTC) - timedelta(days=args.days)

    ledger_path = os.environ.get("BLADEX_ROCKSDB_PATH", "data/bladex_hub")
    ledger = MemoryHub(ledger_path, read_only=True)
    ledger.open()

    main_chars: list[int] = []
    aux_chars: list[int] = []
    model_by_bucket: dict[str, Counter] = {}
    total = 0
    for _key, turn in ledger.scan_prefix(""):
        if cutoff is not None and turn.ts < cutoff:
            continue
        total += 1
        chars = _estimate_context_chars(turn.request_messages)
        if turn.auxiliary:
            aux_chars.append(chars)
            continue
        main_chars.append(chars)
        # 该轮实际用的模型，按最高命中阈值分桶（观察现状分布）
        bucket = "<" + str(thresholds[0])
        for th in thresholds:
            if chars > th:
                bucket = ">" + str(th)
        model_by_bucket.setdefault(bucket, Counter())[turn.model] += 1
    ledger.close()

    print(f"\n== 规模层校准（{total} 轮；主流量 {len(main_chars)} / aux {len(aux_chars)}"
          f"{f'；最近 {args.days} 天' if cutoff else ''}）==\n")

    for label, vals in (("主流量", main_chars), ("auxiliary", aux_chars)):
        if not vals:
            continue
        pct = _percentiles(vals)
        line = "  ".join(f"p{p}={v:,}" for p, v in pct.items())
        print(f"[{label}] context_chars 分位数：\n  {line}\n")

    if main_chars:
        print("[主流量] 各候选 floor 阈值的触发占比（= 会被抬到 medium+ 的流量比例）：")
        n = len(main_chars)
        for th in thresholds:
            hit = sum(1 for c in main_chars if c > th)
            print(f"  floor > {th:>7,}: {hit:>5} / {n}  ({hit / n:6.1%})")
        print()

    if model_by_bucket:
        print("[主流量] 现状：规模桶 × 实际模型：")
        for bucket in sorted(model_by_bucket, key=lambda b: (b[0] == ">", b)):
            counts = model_by_bucket[bucket]
            detail = ", ".join(f"{m.split('/')[-1]}×{c}" for m, c in counts.most_common())
            print(f"  {bucket:>9}: {detail}")
        print()

    print("参考：阈值应满足「floor 触发占比 ≈ 你愿意为质量下限付 medium 价的流量比例」。")
    print("改 config/routing.toml [strategies.scale] 后重启 proxy 生效。")


if __name__ == "__main__":
    main()
