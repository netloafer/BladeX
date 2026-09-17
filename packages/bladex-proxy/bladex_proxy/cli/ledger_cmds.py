"""账本命令：`ledger list / show / doctor`（五段账本只读面）。

09-06 F0.1 自 `cli.py` 拆出，零行为（函数体逐字搬家）。共享 helper（`_admin_call` / `_http_json` / `_probe_ready` / pidfile 族…）留在门面 `bladex_proxy.cli`；本模块经 `from bladex_proxy.cli import …` 取用，
🔴 故 monkeypatch 要打在**本模块**上（`bladex_proxy.cli.<本模块>.<helper>`），打门面不生效。
"""

from __future__ import annotations

import json

import typer

from bladex_proxy.cli import _admin_call, _print_op_result  # noqa: E402

ledger_app = typer.Typer(help="Task ledgers (five-section working state; read-only except "
                              "`doctor --bind-legacy-matters --yes`)",
                         no_args_is_help=True)


# ── ledger（五段账本，只读）─────────────────────────────────────
# 命名口径（V-A4 符号层收口后，2026-08-28）："ledger" 一词只指**任务账本**--
# agent 侧模型维护的五段任务工作状态 Goal/Core/Verified/Open/Next（ADR-0032 §4），
# 作者在 agent 侧，CLI 只读。Memory Hub（RocksDB，data/bladex_hub 磁盘路径为
# 历史遗留不迁移）在符号/文案层一律叫 Hub：`bladex inspect hub`。
# 台账（distill/judgment journal）叫 journal。三者不再共用 "ledger" 一词。


def _ledger_entry_counts(row: dict) -> str:
    """五段条目数单元格（`g1/c2/v0/o1/n3`）。

    `entry_counts` 来自 `Ledger.section_order`（**默认值不是闭集**，自定义段会
    出现），goal 恒不在其中（端点明确排除）。渲染上仍按默认五段给主位，
    未知段追加在尾部，不丢信息也不假设闭集。
    """
    counts = row.get("entry_counts") or {}
    goal_flag = 1 if row.get("goal") else 0
    main = "/".join(
        f"{s[:1]}{counts.get(s, 0)}"
        for s in ("core", "verified", "open", "next"))
    extras = sorted(k for k in counts if k not in
                    ("core", "verified", "open", "next"))
    cell = f"g{goal_flag}/{main}"
    for k in extras:
        cell += f"/{k}={counts.get(k, 0)}"
    return cell


def _ledger_time_cell(ts: str | None) -> str:
    """UPDATED 列：端点给 UTC（真相层），这里转成 CLI 运行环境的本地时区（MQ-L46）。"""
    from bladex_core.ledger import local_iso
    return local_iso(ts or "") or "-"


def _ledger_goal_cell(row: dict) -> str:
    """Goal 单元格：一行截断 + 来源角标（`[u]`=user 原话 / `[m]`=model 提炼 /
    `[r]`=model_revised 经用户改过；空 = 未取到）。"""
    text = (row.get("goal", "") or "").replace("\n", " ").strip()
    if not text:
        return "-"
    src = row.get("goal_source", "") or ""
    tag = {"user": "u", "model": "m", "model_revised": "r"}.get(src, "?")
    return f"[{tag}] {text}"[:30]


def _ledger_active_cell(active_in: list) -> str:
    """激活 scope 单元格：`agent@project` 逗号连接；空 = 未被任何会话激活。"""
    cells = []
    for s in active_in or []:
        agent = (s or {}).get("agent", "?")
        project = (s or {}).get("project", "") or "-"
        cells.append(f"{agent}@{project}")
    return ", ".join(sorted(cells)) if cells else "-"


@ledger_app.command("list")
def ledger_list(
    json_output: bool = typer.Option(False, "--json", help="Print raw JSON from the admin API"),
) -> int:
    """List task ledgers (active ones first, order from the admin API)."""
    code, data = _admin_call("GET", "/admin/ledgers")
    if code != 200:
        return _print_op_result(code, data)
    if json_output:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0
    if not data.get("enabled", True):
        # 端点返回 {"enabled": false, "ledgers": [], "total": 0}：agency 未启用。
        print("Ledger feature is not enabled on this proxy "
              "(check the agency/ledger configuration and restart).")
        return 0
    rows = data.get("ledgers") or []
    if not rows:
        print("No task ledgers yet -- they are created by agents as they work "
              "(one ledger per task; this list will fill in once one does).")
        return 0
    total = data.get("total", len(rows))
    print(f"{total} task ledger(s):")
    # 表头与数据列一一对齐（缺陷 2 的教训：改列必须同步改两边，测试按列断言钉死）。
    header = (f"  {'ID':<16} {'TITLE':<24} {'GOAL':<30} {'ENTRIES':<24} "
              f"{'ACTIVE-IN':<24} {'MATTER':<13} {'UPDATED':<20} STATUS")
    print(header)
    for r in rows:
        print(f"  {r.get('ledger_id', '')[:16]:<16} "
              f"{(r.get('title') or r.get('ledger_id', ''))[:24]:<24} "
              f"{_ledger_goal_cell(r)[:30]:<30} "
              f"{_ledger_entry_counts(r)[:24]:<24} "
              f"{_ledger_active_cell(r.get('active_in'))[:24]:<24} "
              f"{(r.get('matter_id') or '-')[:13]:<13} "
              f"{_ledger_time_cell(r.get('updated_at'))[:19]:<20} "
              f"{r.get('status', '-')}")
    return 0


#: ── `ledger doctor` 的四个阈值 ──────────────────────────────────────────
#: 🔴 每个都写取值理由。本仓纪律：标定常数不许是随手取的圆整数。
#:
#: 标题归一化后相等 = 疑似重复。**不做模糊匹配**：doctor 是体检命令，
#: 误报一次就会让人不再看它；宁可漏报。
#: Goal 前缀比较长度：取 60 —— 2026-08-27 那批账本里，Codex 内部调用生成的
#: Goal 前 60 字符已足够区分（`Generate 0 to 3 hyperpersonalized suggestions…`
#: 与 `You are a helpful assistant. You will be presented…` 在 40 字符内就分开）；
#: 取太长会被后半段的差异救回去，等于关掉这条检查。
_DOCTOR_GOAL_PREFIX = 60
#: 长期无更新的天数。7 天 = ADR-0012 定义的 Matter「周级可完结」粒度——
#: 一件事跨过一周还没动静，值得看一眼。不是"坏"，是"该复核"。
_DOCTOR_STALE_DAYS = 7


def _doctor_norm_title(t: str) -> str:
    """标题归一化：小写 + 压空白。仅此而已。

    不去标点、不去词缀——那些会把「Codex ledger health 开发」与
    「Codex ledger health 复核」判成同一件事。宁可漏报。
    """
    return " ".join((t or "").lower().split())


def _doctor_age_days(iso: str) -> float | None:
    """ISO 时间串 → 距今天数。解析不了返回 None（不猜、不当成 0）。"""
    import datetime as _dt
    s = (iso or "").strip()
    if not s:
        return None
    try:
        ts = _dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=_dt.UTC)
    return (_dt.datetime.now(_dt.UTC) - ts).total_seconds() / 86400.0


def _doctor_scope_key(row: dict) -> str:
    """账本所属 scope。`active_in` 为空 = 当前没有 agent 激活它。

    重复检测**只在同一 scope 内比**：不同 agent 各有一本同名账本是
    合理形态（跨 agent 交接正是北极星要的），不该报成重复。
    """
    scopes = row.get("active_in") or []
    if not scopes:
        return ""
    s = scopes[0]
    return f"{s.get('agent','')}\x1f{s.get('project','')}"


def _doctor_findings(rows: list[dict], details: dict[str, dict]) -> list[dict]:
    """四类检查，全部确定性——不调 LLM、不用 embedding（体检命令要能随处跑）。"""
    from collections import defaultdict

    out: list[dict] = []

    # ① 疑似重复：同 scope 下标题归一化后相等，或 Goal 前缀相同
    by_title: dict[tuple[str, str], list[str]] = defaultdict(list)
    by_goal: dict[tuple[str, str], list[str]] = defaultdict(list)
    for r in rows:
        scope = _doctor_scope_key(r)
        t = _doctor_norm_title(r.get("title", ""))
        if t:
            by_title[(scope, t)].append(r["ledger_id"])
        g = (r.get("goal") or "").strip()[:_DOCTOR_GOAL_PREFIX]
        if g:
            by_goal[(scope, g)].append(r["ledger_id"])
    for (_scope, key), ids in sorted(by_title.items()):
        if len(ids) > 1:
            out.append({"kind": "duplicate_title", "ledger_ids": sorted(ids),
                        "detail": f"same title after normalisation: {key!r}",
                        "action": f"bladex ledger show {sorted(ids)[0]}"})
    seen = {tuple(sorted(f["ledger_ids"])) for f in out}
    for (_scope, _key), ids in sorted(by_goal.items()):
        if len(ids) > 1 and tuple(sorted(ids)) not in seen:
            out.append({"kind": "duplicate_goal", "ledger_ids": sorted(ids),
                        "detail": f"same first {_DOCTOR_GOAL_PREFIX} chars of goal",
                        "action": f"bladex ledger show {sorted(ids)[0]}"})

    # ② 疑似机器文本：Goal 看着像客户端模板
    # 🔴 判据复用 `agency._machine_text_mark`（任务卡明确要求，两份判据早晚分叉）。
    from bladex_proxy.agency import _machine_text_mark
    for r in rows:
        d = details.get(r["ledger_id"]) or {}
        led = d.get("ledger") or {}
        # `goal_verbatim` 是**用户原话**，才是该检查的对象；`goal` 已被模型提炼过。
        # list 端点不返回它 —— 见 doctor 的 docstring「端点数据不够用」那段。
        probe = led.get("goal_verbatim") or r.get("goal") or ""
        mark = _machine_text_mark(probe)
        if mark:
            out.append({"kind": "machine_text_goal", "ledger_ids": [r["ledger_id"]],
                        "detail": f"goal looks like a client template ({mark})",
                        "action": f"bladex ledger show {r['ledger_id']}"})

    # ③ 长期无更新：段条目全 0 且 goal 从未修订过，且创建已久
    for r in rows:
        counts = r.get("entry_counts") or {}
        if any(int(v or 0) for v in counts.values()):
            continue
        d = details.get(r["ledger_id"]) or {}
        if (d.get("ledger") or {}).get("goal_revisions"):
            continue
        age = _doctor_age_days(r.get("created_at", ""))
        if age is not None and age >= _DOCTOR_STALE_DAYS:
            out.append({"kind": "never_used", "ledger_ids": [r["ledger_id"]],
                        "detail": f"no entries and no goal revision, {age:.0f} days old",
                        "action": f"bladex ledger show {r['ledger_id']}"})

    # ④ 没有 Matter 锚 —— **只报锚点机制上线之后建的**
    #
    # 🔴 初版报所有无锚账本：20 本里报了 12 本，信噪比 25%（2026-08-28 live 实测）。
    # 锚点是 08-26 才启用的，之前建的账本天然没锚，**它们不是问题**。
    # 一条必然大面积误报的检查，会让人不再看这个命令 —— 而这正是
    # `_doctor_norm_title` 注释里自己写过的话，却在这一项上犯了。
    #
    # 判据改为**数据自证**、不写死日期：取所有已锚账本里最早的 `created_at`
    # 当作机制上线时刻，只报晚于它、却仍无锚的。
    # 一本已锚账本都没有 ⇒ 机制没跑过 ⇒ 整条检查跳过（无从判断）。
    from bladex_core.ledger import iso_ms  # 比数值不比 ISO 串（MQ-L46）
    anchored_births = sorted(
        ((r.get("created_at") or "") for r in rows
         if (r.get("matter_id") or "").strip() and (r.get("created_at") or "")),
        key=iso_ms)
    if anchored_births:
        epoch = anchored_births[0]
        for r in rows:
            if (r.get("matter_id") or "").strip():
                continue
            born = r.get("created_at") or ""
            if born and iso_ms(born) > iso_ms(epoch):
                out.append({
                    "kind": "no_matter_anchor", "ledger_ids": [r["ledger_id"]],
                    "detail": f"created {born} (after anchoring went live at "
                              f"{epoch}) but still has no Matter",
                    "action": f"bladex ledger show {r['ledger_id']}"})
    return out


@ledger_app.command("doctor")
def ledger_doctor(
    json_output: bool = typer.Option(False, "--json", help="Print findings as JSON"),
    details: bool = typer.Option(
        True, "--details/--no-details",
        help="Fetch each ledger's detail (needed for goal_verbatim); one extra call per ledger"),
    bind_legacy_matters: bool = typer.Option(
        False, "--bind-legacy-matters",
        help="Backfill the ledger->Matter pointer on ledgers created before it was "
             "written at creation (lists them; add --yes to write)"),
    yes: bool = typer.Option(False, "--yes", help="With --bind-legacy-matters: actually write"),
) -> int:
    """Report suspicious ledgers. Read-only -- never edits or merges.

    唯一的写动作是显式的 `--bind-legacy-matters --yes`（2026-09-04，
    `task-ledger-switch-candidates-20260904.md` §2）：只给 `matter_id` 空的旧账本补上
    确定性派生的锚 id（与 Index 已有的同一个），不合并、不改绑、不碰 Goal。

    2026-08-27 事故驱动：用户只发了**一句话**，BladeX 建了**四本账本**
    （两本来自 Codex 客户端的内部功能调用），靠肉眼翻列表才发现，排查三小时。
    根因已修（MQ-L21），但**反方向「多本账本本该是一本」零检测**（V-L5 §1c）。
    本命令补的是那个检测 —— 不自动修，是让下一次三分钟内被看见。

    🔴 **只报告不自动修**：账本的作者是 agent 侧的模型，Goal 只有用户能改；
    自动合并会踩「误合并=0」第一红线。

    ⚠️ **端点数据不够用的地方（任务卡口径要求 1：说出来，别绕过去读库）**：
    `/admin/ledgers` 列表**不返回 `goal_verbatim`**（用户原话）与 `goal_revisions`，
    而检查 ② 和 ③ 需要它们。故默认对每本账本再取一次
    `/admin/ledgers/{id}`——账本池是几十本量级，N+1 可接受；
    嫌慢用 `--no-details`，那时 ② 退化为查已提炼的 `goal`（判据变弱，会漏报）。
    """
    if bind_legacy_matters:
        return _ledger_bind_legacy_matters(write=yes, json_output=json_output)
    code, data = _admin_call("GET", "/admin/ledgers")
    if code != 200:
        return _print_op_result(code, data)
    if not data.get("enabled", True):
        print("Ledger feature is not enabled on this proxy "
              "(check the agency/ledger configuration and restart).")
        return 0
    rows = data.get("ledgers") or []
    if not rows:
        print("No task ledgers yet -- nothing to check.")
        return 0

    detail_map: dict[str, dict] = {}
    if details:
        from urllib.parse import quote
        for r in rows:
            c, d = _admin_call("GET", f"/admin/ledgers/{quote(r['ledger_id'], safe='')}")
            if c == 200:
                detail_map[r["ledger_id"]] = d

    findings = _doctor_findings(rows, detail_map)

    if json_output:
        print(json.dumps({"checked": len(rows), "findings": findings},
                         ensure_ascii=False, indent=2))
        return 1 if findings else 0

    if not findings:
        print(f"Checked {len(rows)} ledger(s): no problems found.")
        return 0
    print(f"Checked {len(rows)} ledger(s), {len(findings)} finding(s):")
    print()
    for f in findings:
        ids = ", ".join(f["ledger_ids"])
        print(f"  [{f['kind']}] {ids}")
        print(f"      {f['detail']}")
        print(f"      -> {f['action']}")
    print()
    print("Nothing was changed. Review each one before acting.")
    return 1


def _ledger_bind_legacy_matters(*, write: bool, json_output: bool) -> int:
    """`ledger doctor --bind-legacy-matters [--yes]`：走 `/admin/ledgers/bind-legacy-matters`。"""
    code, data = _admin_call("POST", "/admin/ledgers/bind-legacy-matters",
                             body={"dry_run": not write})
    if code != 200:
        return _print_op_result(code, data)
    if json_output:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0
    rows = data.get("ledgers") or []
    print(f"Checked {data.get('checked', 0)} ledger(s): {data.get('unbound', 0)} without a Matter pointer.")
    for r in rows:
        print(f"  {r['ledger_id']} -> {r['matter_id']}  {r.get('title', '')}")
    if not rows:
        return 0
    if write:
        print(f"Bound {len(rows)} ledger(s) (LEDGER_UPDATE events written; replay-equivalent).")
    else:
        print("Nothing was changed (dry run). Re-run with --yes to write.")
    return 0


@ledger_app.command("show")
def ledger_show(
    ledger_id: str = typer.Argument(..., help="Ledger ID (ldg-...)"),
    json_output: bool = typer.Option(False, "--json", help="Print raw JSON from the admin API"),
) -> int:
    """Show one task ledger (goal, sections, children, Matter anchor)."""
    code, data = _admin_call("GET", f"/admin/ledgers/{ledger_id}")
    if code != 200:
        # 404 时端点给 {"status": "not_found", "detail": "unknown ledger ..."}
        # -- 照它的 detail 打印，不吞成 "HTTP 404"。
        detail = data.get("detail", "")
        if code == 404 and detail:
            print(f"✗ {detail}")
            return 1
        return _print_op_result(code, data)
    if json_output:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0
    print(data.get("markdown", ""))
    children = data.get("children") or []
    if children:
        print()
        print("Sub-ledgers:")
        for c in children:
            print(f"  {c.get('ledger_id', '')}  {c.get('title', '')}")
    return 0
