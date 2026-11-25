"""`status` 的展示 helper（流量 / 漏斗 / 队列 / 降级横幅 / Memory 配置行）。批 LC 自 `cli/__init__.py` 零行为搬来（W33 为避 2,500 行守卫临时放在门面）。"""

from __future__ import annotations

from bladex_proxy.cli import (  # noqa: E402
    _admin_key,
    _http_json,
)


def _fmt_traffic_row(detail: dict) -> str:
    """一行 turn 摘要（ADR-0027 §3.3）。字段全在 Memory Hub，只是此前没有 CLI 出口。"""
    ts = str(detail.get("timestamp") or detail.get("ts") or "")[:19].replace("T", " ")
    ident = detail.get("identity") or {}
    agent = ident.get("agent_id") or "?"
    injected = detail.get("injected_memory") or ""
    lines = [ln.strip() for ln in injected.splitlines() if ln.strip().startswith("-")]
    hard = sum(1 for ln in lines if "[MUST/NEVER]" in ln)
    items = len(lines) - hard
    dm = detail.get("decision_meta") or {}
    source = dm.get("source") or dm.get("route_source") or "-"
    model = dm.get("model") or dm.get("selected_model") or ""
    status_txt = (detail.get("response_meta") or {}).get("status") or detail.get("status") or "ok"
    return (f"  {ts:19s}  {agent:22.22s}  inject(h={hard},items={items}):<0  "
            f"route={source:12.12s} {model:22.22s} {status_txt}").replace(":<0", "")


def _scrape_metrics(url: str) -> dict[str, float]:
    """抓一个 Prometheus 端点，返回 {"metric{labels}": value}。抓不到返回空。"""
    import urllib.request

    out: dict[str, float] = {}
    try:
        with urllib.request.urlopen(url, timeout=3) as r:
            body = r.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 —— 端点没开是常态，不是错误
        return out
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # 用 rsplit 而不是 rpartition：`test_no_module_reimplements_env_parsing`
        # 禁止本模块出现 `partition`（旧的三份 .env 解析实现就是那个形状）。
        # 这里虽然解析的是 Prometheus 文本不是 .env，但守卫是源码级的，
        # 绕开它比削弱它便宜——守卫宁可误报也不该被放宽。
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        try:
            out[parts[0].strip()] = float(parts[1])
        except ValueError:
            continue
    return out


def _print_funnel(base: str) -> None:
    """记忆漏斗（M4-2）：把两个进程的端点拼成一条链。

    两端各自暴露（2026-08-06 拍板），所以这里要拉两个地址；好处是
    **哪半边没数一眼可见**——某一段为 0 而它上一段非 0，那一段就是断的，
    而这正是复核里 D7/E8/F6/G7/H7/I5 六条"缺观测"共同指向的东西。
    """
    import os

    from bladex_core.funnel import FUNNEL_ORDER

    write_port = os.environ.get("BLADEX_CONSOLIDATOR_METRICS_PORT", "").strip()
    samples = _scrape_metrics(f"{base}/metrics")
    if write_port and write_port != "0":
        samples.update(_scrape_metrics(f"http://127.0.0.1:{write_port}/metrics"))

    print("\n  Memory funnel (write side = consolidator, read side = proxy):")
    if not samples:
        print("    no metrics available -- is the proxy running?")
        return
    missing_write = not write_port or write_port == "0"
    for metric in FUNNEL_ORDER:
        total = sum(v for k, v in samples.items() if k.split("{")[0] == metric)
        label = metric.replace("bladex_memory_", "").replace("_total", "")
        print(f"    {label:<12} {int(total)}")
    if missing_write:
        print("    (write-side stages read 0 because the consolidator metrics endpoint "
              "is off -- set BLADEX_CONSOLIDATOR_METRICS_PORT to see them)")


def _print_traffic(base: str, n: int) -> int:
    """最近 n 轮的一行摘要（ADR-0027 §3.3 契约 C）。

    这是用户回答"它在工作吗 / 为什么没记住"的日常工具——此前只能翻日志或读 Memory Hub。
    """
    turn_index = _http_json(f"{base}/admin/turns?limit={max(1, min(n, 200))}", key=_admin_key())
    if turn_index is None:
        print("  Cannot read /admin/turns -- the proxy is not running, or the admin key is wrong"
              "（BLADEX_ADMIN_KEYS / BLADEX_CLIENT_KEYS）。")
        return 1
    turns = turn_index.get("turns", [])
    if not turns:
        print("  No recent turns. If you are chatting and still see nothing, the storage step is broken -- "
              "run `bladex doctor` and check Redis/Memory Hub.")
        return 0
    print(f"  Last {len(turns)} turns (of {turn_index.get('total', '?')}):")
    from urllib.parse import quote
    for item in turns:
        detail = _http_json(f"{base}/admin/turns/{quote(item['key'])}", key=_admin_key())
        if detail is None:
            print(f"  {item.get('ts', '')}  {item.get('key', '')}  <read failed>")
            continue
        print(_fmt_traffic_row(detail))
    return 0


def _fmt_bytes(n: int | None) -> str:
    """人类可读的字节数。None = 路径不存在（与 0 字节是两回事，显示 '-'）。"""
    if n is None:
        return "-"
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def _print_memory_config(admin: dict | None) -> None:
    """打印**运行中进程实际在用的**记忆配置，并对与 config/.env 的漂移报黄灯。

    为什么必须是"进程实际在用的"而不是"读 .env 再打印一遍"：
    2026-08-05 的事故正是 `.env` 写 200、进程跑 150（终端里残留了一个 export，
    而 `_load_env_file` 不覆盖已设环境变量）。每轮注入超时 1–3ms 被整段丢弃，
    记忆静默失效，而唯一痕迹是混在 info 流里的一行 warning。
    读配置文件打印出来的绿灯，恰恰是那次骗过所有人的东西。
    """
    mc = (admin or {}).get("memory_config") or {}
    eff = mc.get("effective") or {}
    if not eff:
        return
    flags = mc.get("flags") or {}
    off = [k.replace("BLADEX_", "").lower() for k, v in flags.items() if not v]
    print(f"  Memory:       hotpath {eff.get('BLADEX_HOTPATH_BUDGET_MS', '?')}ms"
          f" / inject top-{eff.get('BLADEX_INJECT_TOPK', '?')}"
          f" / min-importance {eff.get('BLADEX_MIN_IMPORTANCE', '?')}"
          + (f"  (off: {', '.join(off)})" if off else ""))

    drift = mc.get("drift") or []
    if drift:
        # 用户可见字符串必须是英文（ADR-0027 §5.1，有 AST 守卫钉着）；
        # 中文只留在注释里。
        print()
        print("  ⚠ Config drift: the running process is NOT using what config/.env declares")
        for d in drift:
            print(f"      {d['key']}: .env says {d['declared']}, process uses {d['effective']}")
        print("    Usually a stale export in the shell that started the proxy"
              " (env overrides .env).")
        print("    Fix: unset it, then `bladex restart` in that same shell"
              " (the daemon inherits its launcher's environment).")


def _consolidator_activity(admin: dict | None) -> str:
    """consolidator 那行的后缀：它此刻在干什么（2026-08-06 事故驱动）。

    事故形状：队列恢复后 487 轮在消化，一轮 30 条要跑几十分钟，而 status 里的
    `-> Index N waiting` 只在每轮提交时跳一次——轮内那半小时纹丝不动，和进程卡死
    长得一模一样，只能 tail 日志才能确认它在干活。所以把轮内进度接出来。

    三条不能含糊的边界：
      · 没有心跳 → 一个字都不加。老版本 consolidator 不写心跳，"读不到"不等于"没在跑"。
      · 心跳不新鲜 → 只说多久没动了，不宣称在跑也不宣称死了（可能卡在一次长上游调用里）。
      · state=error → 直接说出错，别让它混在"正在蒸馏"里被当成进展。
    """
    hb = (admin or {}).get("consolidator") or {}
    if not hb:
        return ""
    state, phase = hb.get("state", ""), hb.get("phase", "")
    if not hb.get("fresh", False):
        age = hb.get("last_seen_s")
        return f" -- no heartbeat for {age}s (was: {state or 'unknown'})" if age else ""
    if state == "error":
        return f" -- last batch failed: {(hb.get('error') or '')[:80]}"
    done, total, batch = hb.get("done"), hb.get("total"), hb.get("batch")
    if phase == "distill" and isinstance(total, int) and total > 0:
        return f" -- distilling batch {batch}, {done}/{total} units"
    if phase == "scan":
        return f" -- batch {batch}, scanning the Hub"
    if state == "idle":
        return " -- idle, waiting for new turns"
    return f" -- {phase or state}" if (phase or state) else ""


def _format_queue(admin: dict | None, consolidator_running: bool) -> str:
    """两段管线各有多少轮在排队：Pipeline -> Memory Hub -> Memory Index。

    为什么单列一行（2026-08-06 事故驱动）：此前 lag 只是 Memory Index 那行末尾的一个
    尾巴，而 **Pipeline 侧的积压在 CLI 里根本没有出口**。08-05 `doctor --e2e` 卡在
    step 3（探针轮 180s 内没蒸出 fact），真实原因是 972 轮积压排在它前面——但 status
    看不出"排了多少、往哪一段排、有没有人在消化"，只能去翻 consolidator 日志。

    三个字段都来自 /admin/status（早就有，只是没打出来）：
      pipeline.backlog   Redis stream 里等着写进 Hub 的轮次
      index.lag_turns    已入 Hub、还没被蒸馏进 Index 的轮次
      overflow/spill     Redis 不可达时落盘的轮次（回灌前不产生记忆）

    孤儿单列（2026-08-06 第二次事故驱动）：backlog 里可能有一部分**谁都不会再碰**
    （已 XACK、未 XDEL 的残留，见 PipelineRedis.inspect）。它们和"正在排队"混在一个数里，
    症状是这个数永远不降——而原来的后缀还会安慰你一句"the pipeline worker is draining it"，
    并且只要 Index 侧有一条 lag，那句话连出现的机会都没有。所以孤儿必须自己有一句话，
    且不被前面任何分支吞掉。
    """
    pipeline = (admin or {}).get("pipeline") or {}
    index = (admin or {}).get("index") or {}

    def _int(value: object) -> int | None:
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    to_hub = _int(pipeline.get("backlog"))
    to_index = _int(index.get("lag_turns"))
    orphan = _int(pipeline.get("orphan")) or 0
    moving = None if to_hub is None else max(0, to_hub - orphan)
    line = (f"-> Hub {moving if moving is not None else '?'} waiting"
            f" / -> Index {to_index if to_index is not None else '?'} waiting")

    # 状态后缀按"谁该动而没动"排序：积压 + 没人消化，是最需要被看见的一种
    if to_index:
        line += (" -- catching up" if consolidator_running
                 else " -- consolidator is not running, so nothing is being distilled")
    elif moving:
        line += " -- the pipeline worker is draining it"
    elif moving == 0 and to_index == 0 and not orphan:
        line += " -- caught up"

    # 孤儿这句独立于上面所有分支：它不是"排队慢"，是"排不动"，两件事不能共用一句措辞
    if orphan:
        line += (f" / {orphan} stale entries nobody will ever pick up"
                 " -- run `bladex queue flush`")

    spilled = (_int(pipeline.get("overflow_count")) or 0) + (_int(pipeline.get("spill_count")) or 0)
    if spilled:
        line += f" / {spilled} spilled to disk (replayed once Redis is back)"
    return line


def _format_backlog_trend(samples: dict[str, float]) -> str:
    """Queue 行下面那行：两段积压在最近窗口里的**走向**（MQ-W4）。

    数据来自 proxy `/metrics` 的 `bladex_backlog_*`（proxy 后台每分钟采一次）。
    某一段没有序列 = 样本还不够（proxy 刚起不到三分钟）——明说 "n/a"，不当成持平。
    B10 的判读就看这一行：任一段出现 "rising, sustained" 即为持续上升（判据见 metrics.BacklogTrend）。
    """
    def _get(metric: str, stage: str) -> float | None:
        return samples.get(f'{metric}{{stage="{stage}"}}')

    parts: list[str] = []
    for stage, label in (("hub", "Hub"), ("index", "Index")):
        delta = _get("bladex_backlog_delta_turns", stage)
        trend = _get("bladex_backlog_trend", stage)
        if delta is None or trend is None:
            parts.append(f"-> {label} n/a (not enough samples yet)")
            continue
        span = _get("bladex_backlog_window_span_seconds", stage) or 0.0
        word = "rising" if trend > 0 else "falling" if trend < 0 else "flat"
        if trend > 0 and _get("bladex_backlog_monotonic_rise", stage) == 1.0:
            word = "rising, sustained"
        parts.append(f"-> {label} {int(delta):+d} over {_ago(span).removesuffix(' ago')} ({word})")
    return " / ".join(parts)


def _ago(seconds: float) -> str:
    """人读的相对时间（英文，用户可见面）。"""
    seconds = max(0.0, seconds)
    if seconds < 90:
        return f"{int(seconds)}s ago"
    if seconds < 5400:
        return f"{int(seconds // 60)}m ago"
    if seconds < 172800:
        return f"{int(seconds // 3600)}h ago"
    return f"{int(seconds // 86400)}d ago"


def _print_degradation_banner(admin: dict | None) -> None:
    """静默降级的累计告警面（ADR-0027 §3.3）。

    inject_timeout / enqueue_skipped / consolidator lag 都是"记忆安静地失效"的形状，
    此前只散在日志里。非零就顶部黄字，让降级不再静默。
    """
    if not admin:
        return
    warnings: list[str] = []
    degr = admin.get("degradation") or {}
    inject_timeouts = degr.get("inject_timeout_1h", admin.get("inject_timeout_1h"))
    enqueue_skipped = degr.get("enqueue_skipped_1h", admin.get("enqueue_skipped_1h"))
    lag = (admin.get("index") or {}).get("lag_turns")
    if inject_timeouts:
        # 教训 40（2026-08-17）：这句提示此前只说"抬预算"，于是三次会话被自己的
        # 告警带着走，把预算从 100 一路抬到 250，而真实成因是 LanceDB 索引碎片
        # 让检索慢了 15 倍（6462ms -> 414ms）。抬预算还会掩盖读数：deadline 一砍，
        # `elapsed_ms` 恒等于 budget + ε，真实耗时永远测不到。**先查碎片。**
        warnings.append(f"in the last 1h: {inject_timeouts} turns degraded on inject timeout (hard rules only)"
                        " -- check index fragmentation first (`bladex-consolidator` compacts it;"
                        " scripts/compact_index.py reports it), then consider raising"
                        " BLADEX_HOTPATH_BUDGET_MS or narrowing retrieval")
    if enqueue_skipped:
        warnings.append(f"in the last 1h: {enqueue_skipped} turns spilled to disk (Redis unreachable)"
                        " -- they produce no memory until Redis is back and they are replayed")
    index_unavailable = degr.get("index_unavailable_1h")
    if index_unavailable:
        warnings.append(f"in the last 1h: {index_unavailable} lookups hit an unavailable Memory Index layer"
                        " -- only hard rules were injected; check the embedding backend")
    if isinstance(lag, int) and lag > 500:
        warnings.append(f"consolidator is behind by {lag} turns -- distillation cannot keep up; new memory stays unrecallable")
    for w in warnings:
        print(f"  ⚠ {w}")
    if warnings:
        print()
