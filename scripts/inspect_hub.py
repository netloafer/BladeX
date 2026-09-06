"""Memory Hub 巡检 + 路由解释 —— 从 Memory Hub RocksDB 看真实会话，或用 e5 解释某条 query 为何这样路由。

本机跑（uv 环境；只读打开 Memory Hub，可与 proxy 进程共存）：

  # 1) 概览：模型分布 / 注入率 / agent 识别来源 / 失败数
  uv run python scripts/inspect_hub.py

  # 2) 逐轮明细（可加 --limit N）
  uv run python scripts/inspect_hub.py --list --limit 50

  # 3) 找某条 query（子串匹配最后一条 user 消息），看它落到哪个模型
  uv run python scripts/inspect_hub.py --grep 破产案件

  # 5) T10：按 error 分桶统计失败原因
  uv run python scripts/inspect_hub.py --failed

  # 6) T10：列 agent 识别落空(unknown/fallback)的轮次 + 指纹线索
  uv run python scripts/inspect_hub.py --unknown

  # 7) 同会话同指纹同回复的重复轮次（`bladex queue flush --apply` 之后的对账）
  uv run python scripts/inspect_hub.py --dupes

Memory Hub 路径默认读环境变量 BLADEX_ROCKSDB_PATH，否则 data/bladex_hub。
注意：Turn 只存"最终用的 model"，不存路由 source/tier/reason（那些只在 proxy 日志里，
`route_decision` 行）；--grep/--list 给"是什么"。
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

# 自举：把 workspace 成员包加进 sys.path，避免 `uv run` 重同步裁掉 member 后 import 失败。
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in ("packages/bladex-core", "packages/bladex-proxy"):
    _full = os.path.join(_ROOT, _p)
    if _full not in sys.path:
        sys.path.insert(0, _full)

# Turn key 之外的特殊前缀（内容/墓碑/管理事件/元数据），巡检时跳过。
_SKIP_PREFIXES = ("__msg__/", "__meta__/", "tombstone/", "admin_event/")

# e5 类别 → 落档（对齐 routing.toml；code 另走能力过滤）。


def _last_user_text(turn: object) -> str:
    msgs = getattr(turn, "request_messages", []) or []
    for m in reversed(msgs):
        if m.get("role") != "user":
            continue
        c = m.get("content")
        if isinstance(c, str):
            return c
        if isinstance(c, list):  # 多模态：拼文本块
            parts = [b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text"]
            return " ".join(p for p in parts if p)
    return ""


def _oneline(s: str, n: int = 70) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _classify_error(error: str, auxiliary: bool) -> str:
    """把 Turn.error 粗分桶（供 T10 失败率排查）。"""
    if not error and auxiliary:
        return "auxiliary"
    low = (error or "").lower()
    if any(k in low for k in ("nocapable", "routing", "route_decision", "no candidate")):
        return "routing"
    # "litellm" 保留：Memory Hub 里 2026-08-05 之前入库的错误文本用的是旧措辞，
    # 分类器要能认出**历史数据**。新写入的走 `router_sdk.error_text()`，命中 "router."。
    if any(k in low for k in ("upstream", "502", "503", "504", "timeout", "timed out",
                              "connection", "connect", "router.", "litellm", "apierror",
                              "rate limit", "429")):
        return "upstream_error"
    if auxiliary:
        return "auxiliary"
    if error:
        return "other"
    return "no_error_recorded"


def _fingerprint_clues(turn: object) -> tuple[str, str]:
    """提取首条 system prompt 片段 + tool 名集合（诊断为何 agent 没被识别）。"""
    msgs = getattr(turn, "request_messages", []) or []
    sys_snip = ""
    for m in msgs:
        if m.get("role") != "system":
            continue
        c = m.get("content")
        text = c if isinstance(c, str) else (
            " ".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")
            if isinstance(c, list) else ""
        )
        if text:
            sys_snip = _oneline(text, 90)
            break
    tool_names: set[str] = set()
    for m in msgs:
        c = m.get("content")
        if isinstance(c, list):
            for b in c:
                if isinstance(b, dict) and b.get("type") == "function":
                    tool_names.add(b.get("function", {}).get("name", ""))
    tools = getattr(turn, "tools", None) or []
    for t in tools:
        if isinstance(t, dict):
            tool_names.add(t.get("function", {}).get("name", "") if "function" in t else t.get("name", ""))
    tool_names.discard("")
    return sys_snip, ",".join(sorted(tool_names))[:60]


def inspect_dupes(path: str, limit: int = 10) -> int:
    """同一会话里内容完全相同、只是 key 不同的轮次（2026-08-06 队列恢复后的对账）。

    为什么需要：`bladex queue flush --apply` 会把队列里没落账的孤儿补写进 Memory Hub。
    "没落账"这个判断依赖 `identity.storage_key(entry_id)` 查不到——万一是 key 算法漂了
    （而不是真的没写过），补写就会造出同一轮的第二份，下游再蒸馏成重复事实。

    判据用 (会话前缀, prefix_hash, 回复正文) 三元组：prefix_hash 是入库时算的请求前缀
    指纹，回复正文相同基本排除"同前缀不同结果"的正常重试。只报组数与样例，不动数据——
    要不要删是另一个决定（删要走墓碑，见 ADR-0012 §3.5）。
    """
    from bladex_proxy.storage.memory_hub import MemoryHub

    ledger = MemoryHub(path, read_only=True)
    ledger.open()
    groups: dict[tuple[str, str, str], list[str]] = {}
    total = 0
    for key_raw, _ in ledger.db.items():
        key = key_raw.decode() if isinstance(key_raw, bytes) else str(key_raw)
        if key.startswith(_SKIP_PREFIXES):
            continue
        turn = ledger.get(key)
        if turn is None:
            continue
        total += 1
        prefix_hash = getattr(turn, "prefix_hash", "") or ""
        if not prefix_hash:
            continue  # 无指纹不参与判重，宁可漏报也不误报
        sig = (turn.identity.session_prefix(), prefix_hash,
               (getattr(turn, "response_text", "") or "")[:2000])
        groups.setdefault(sig, []).append(key)
    ledger.close()

    dupes = {sig: keys for sig, keys in groups.items() if len(keys) > 1}
    extra = sum(len(keys) - 1 for keys in dupes.values())
    print(f"Memory Hub: {path}")
    print(f"  轮次总数:     {total}")
    print(f"  重复组:       {len(dupes)}（多出来的轮次 {extra}）")
    if not dupes:
        print("\n  没有同会话同指纹同回复的重复轮次——补写没有造出第二份。")
        return 0
    print("\n  样例（同一组内 key 不同、内容相同）：")
    for sig, keys in list(sorted(dupes.items(), key=lambda kv: -len(kv[1])))[:limit]:
        print(f"\n  session={sig[0]}  prefix_hash={sig[1][:12]}  ×{len(keys)}")
        for k in sorted(keys)[:5]:
            print(f"    {k}")
    print("\n  注：删除要走墓碑（ADR-0012 §3.5），别直接删 key——Memory Index 重建会复活它。")
    return 0


def inspect_failed(path: str) -> int:
    """T10：按 Turn.error 分桶统计失败原因（先出数据，不预设改法）。"""
    from bladex_proxy.storage.memory_hub import MemoryHub

    ledger = MemoryHub(path, read_only=True)
    ledger.open()
    buckets: Counter[str] = Counter()
    samples: dict[str, str] = {}
    total = 0
    failed = 0
    for key_raw, _ in ledger.db.items():
        key = key_raw.decode() if isinstance(key_raw, bytes) else str(key_raw)
        if key.startswith(_SKIP_PREFIXES):
            continue
        turn = ledger.get(key)
        if turn is None:
            continue
        total += 1
        status = getattr(getattr(turn, "status", None), "value", "")
        if status != "failed":
            continue
        failed += 1
        aux = bool(getattr(turn.identity, "auxiliary", False)) if getattr(turn, "identity", None) else False
        err = getattr(turn, "error", "") or ""
        bucket = _classify_error(err, aux)
        buckets[bucket] += 1
        if bucket not in samples:
            samples[bucket] = _oneline(err, 100) or "(empty error)"
    ledger.close()

    print(f"Memory Hub: {path}")
    print(f"Turn 总数: {total}  失败: {failed} ({failed * 100 // total if total else 0}%)")
    print("\n失败原因分桶:")
    for b, c in buckets.most_common():
        print(f"  {c:>4}  {b}")
        print(f"        样例: {samples.get(b, '')}")
    return 0


def inspect_unknown(path: str, limit: int) -> int:
    """T10：列 agent 识别落空（fallback/unknown）的轮次 + 指纹线索（为何没识别出 agent）。"""
    from bladex_proxy.storage.memory_hub import MemoryHub

    ledger = MemoryHub(path, read_only=True)
    ledger.open()
    rows: list[tuple[str, str, str, str, str, str]] = []
    total = 0
    unknown = 0
    for key_raw, _ in ledger.db.items():
        key = key_raw.decode() if isinstance(key_raw, bytes) else str(key_raw)
        if key.startswith(_SKIP_PREFIXES):
            continue
        turn = ledger.get(key)
        if turn is None:
            continue
        total += 1
        ident = getattr(turn, "identity", None)
        agent_id = getattr(ident, "agent_id", "") if ident else ""
        src = getattr(getattr(turn, "agent_source", None), "value", "")
        if agent_id not in ("unknown", "") and src != "fallback":
            continue
        unknown += 1
        sys_snip, tools = _fingerprint_clues(turn)
        rows.append((str(getattr(turn, "ts", ""))[:19], agent_id or "unknown", src,
                     getattr(turn, "model", "") or "(none)", sys_snip, tools))
    ledger.close()

    print(f"Memory Hub: {path}")
    print(f"Turn 总数: {total}  识别落空(unknown/fallback): {unknown} "
          f"({unknown * 100 // total if total else 0}%)")
    print(f"\n最近 {min(limit, len(rows))} 条落空轮次的指纹线索（ts | agent | src | model | system片段 | tools）:")
    for ts, agent_id, src, model, sys_snip, tools in rows[-limit:]:
        print(f"  {ts} | {agent_id} | {src} | {model}")
        print(f"      system: {sys_snip or '(无 system 消息)'}")
        print(f"      tools:  {tools or '(无)'}")
    return 0


# （`--explain`（e5 难度打分解释）已于 2026-09-03 C2 删除：它 import 的
#  `bladex_core.scorers` 随 ADR-0013 Phase 1 难度打分层退役早已不存在，子命令自 07-06 起就是坏的。）


def inspect(path: str, list_all: bool, grep: str | None, limit: int) -> int:
    from bladex_proxy.storage.memory_hub import MemoryHub

    ledger = MemoryHub(path, read_only=True)
    ledger.open()

    turn_keys: list[str] = []
    for key_raw, _ in ledger.db.items():
        key = key_raw.decode() if isinstance(key_raw, bytes) else str(key_raw)
        if key.startswith(_SKIP_PREFIXES):
            continue
        turn_keys.append(key)

    models: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    status_c: Counter[str] = Counter()
    injected_n = 0
    rows: list[tuple[str, str, str, str, int, str, str]] = []

    for key in turn_keys:
        turn = ledger.get(key)
        if turn is None:
            continue
        model = getattr(turn, "model", "") or "(none)"
        src = getattr(getattr(turn, "agent_source", None), "value", str(getattr(turn, "agent_source", "")))
        status = getattr(getattr(turn, "status", None), "value", str(getattr(turn, "status", "")))
        inj = getattr(turn, "injected_memory", "") or ""
        agent_id = getattr(turn.identity, "agent_id", "") if getattr(turn, "identity", None) else ""
        utext = _last_user_text(turn)

        models[model] += 1
        sources[src] += 1
        status_c[status] += 1
        if inj.strip():
            injected_n += 1

        if grep and grep not in utext:
            continue
        rows.append((str(getattr(turn, "ts", "")), agent_id, src, model, len(inj), status, utext))

    total = len(turn_keys)
    print(f"Memory Hub: {path}")
    print(f"Turn 总数: {total}")
    print("\n模型分布（Turn.model = 实际转发的模型）:")
    for m, c in models.most_common():
        print(f"  {c:>4}  {m}")
    print(f"\n注入率: {injected_n}/{total} 轮 injected_memory 非空"
          f"（注：同硬规则块内容去重，非空≠每轮都重存正文）")
    print("agent 识别来源: " + ", ".join(f"{s}={c}" for s, c in sources.most_common()))
    print("状态: " + ", ".join(f"{s}={c}" for s, c in status_c.most_common()))

    if list_all or grep:
        shown = rows if grep else rows[-limit:]
        title = f"\n匹配 “{grep}” 的轮次" if grep else f"\n最近 {min(limit, len(rows))} 轮明细"
        print(title + "（ts | agent | src | model | inj字数 | status | user）:")
        for ts, agent_id, src, model, inj_len, status, utext in shown:
            print(f"  {ts[:19]} | {agent_id or '-'} | {src} | {model} | inj={inj_len} | {status}")
            print(f"      user: {_oneline(utext)}")

    ledger.close()
    return 0


def inspect_key(path: str, key: str) -> int:
    """按 ledger key 摊开**一轮**：每条消息的 role + 长度 + 开头。

    为什么需要它（2026-08-07）：追查"某条 fact 的 source_text 到底来自哪条消息"时，
    只有聚合视图（`--list` 只打最后一条 user 消息）根本回答不了。
    实例：21 条 `## Goal` 来源的 preference，`compare_distill_calls` 在**全库未截断**
    的扫描里读到 checkpoint 形态 0 条 —— 因为那个诊断只看 `t.user_messages`，
    而那段文本可能根本不在 user 消息里。

    猜了三轮都没猜对（窗口截断 / 全是旧数据 / aux 过滤），最后只能摊开原始数据看。
    这个入口就是那次的产物：**先看数据，再提假设。**
    """
    from bladex_proxy.storage.memory_hub import MemoryHub

    ledger = MemoryHub(path, read_only=True)
    ledger.open()
    turn = ledger.get(key)
    if turn is None:
        print(f"❌ Hub 里没有这个 key：{key}\n"
              "   （可能是重置前的轮次 —— P2 里的 fact 比 Hub 活得久时会这样）")
        ledger.close()
        return 2

    ident = getattr(turn, "identity", None)
    print(f"key      : {key}")
    print(f"ts       : {getattr(turn, 'ts', '')}")
    print(f"agent    : {getattr(ident, 'agent_id', '')}   model: {getattr(turn, 'model', '')}")
    print(f"status   : {getattr(turn, 'status', '')}")
    aux = getattr(turn, "aux_source", "") or ""
    print(f"aux_source: {aux or '(非 aux)'}")

    msgs = getattr(turn, "request_messages", None) or []
    print(f"\nrequest_messages（{len(msgs)} 条）：")
    for i, m in enumerate(msgs):
        role = m.get("role", "?")
        c = m.get("content", "")
        if isinstance(c, list):
            c = " ".join(p.get("text", "") for p in c
                         if isinstance(p, dict) and p.get("type") == "text")
        c = str(c)
        head = c[:180].replace("\n", "⏎")
        mark = "  ← 含 '## Goal'" if "## Goal" in c else (
            "  ← 含 'Preference order for skills'"
            if "Preference order for skills" in c else "")
        print(f"  [{i}] {role:<9} {len(c):>7} 字{mark}\n      {head!r}")

    resp = getattr(turn, "response_text", "") or ""
    print(f"\nresponse_text: {len(resp)} 字\n  {resp[:200]!r}")
    ledger.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Memory Hub 巡检")
    ap.add_argument("--key", help="按 ledger key 摊开某一轮的全部消息（追 source_text 来源）")
    ap.add_argument("--path", default=os.environ.get("BLADEX_ROCKSDB_PATH", "data/bladex_hub"))
    ap.add_argument("--list", action="store_true", help="打印逐轮明细")
    ap.add_argument("--grep", help="只看最后一条 user 消息含此子串的轮次")
    ap.add_argument("--limit", type=int, default=30, help="--list 时显示最近 N 轮")
    ap.add_argument("--failed", action="store_true", help="T10：按 error 分桶统计失败原因")
    ap.add_argument("--unknown", action="store_true", help="T10：列 agent 识别落空的轮次 + 指纹线索")
    ap.add_argument("--dupes", action="store_true",
                    help="同会话同指纹同回复的重复轮次（队列补写后的对账）")
    args = ap.parse_args()

    if args.key:
        return inspect_key(args.path, args.key)
    if args.dupes:
        return inspect_dupes(args.path, args.limit)
    if args.failed:
        return inspect_failed(args.path)
    if args.unknown:
        return inspect_unknown(args.path, args.limit)
    return inspect(args.path, args.list, args.grep, args.limit)


if __name__ == "__main__":
    raise SystemExit(main())
