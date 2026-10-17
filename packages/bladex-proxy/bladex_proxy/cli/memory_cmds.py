"""记忆与存储命令：`memory` / `matter` / `inspect` / `storage`（数据面一律打 admin HTTP API）。

09-06 F0.1 自 `cli.py` 拆出，零行为（函数体逐字搬家）。共享 helper（`_admin_call` / `_http_json` / `_probe_ready` / pidfile 族…）留在门面 `bladex_proxy.cli`；本模块经 `from bladex_proxy.cli import …` 取用，
🔴 故 monkeypatch 要打在**本模块**上（`bladex_proxy.cli.<本模块>.<helper>`），打门面不生效。
"""

from __future__ import annotations

import json
import time

import typer

from bladex_proxy.cli import (  # noqa: E402
    _admin_call,
    _admin_long_timeout_s,
    _print_op_result,
    _require_dry_run_ack,
    _require_server_contract,
    _run_repo_script,
)
from bladex_proxy.cli.ops_cmds import sync_run  # noqa: E402

storage_app = typer.Typer(help="Storage operations", no_args_is_help=True)
memory_app = typer.Typer(help="Memory management (via the proxy admin API)", no_args_is_help=True)
matter_app = typer.Typer(help="Matter management (via the proxy admin API)", no_args_is_help=True)
inspect_app = typer.Typer(help="Storage inspection (hub/index)", no_args_is_help=True)


# ── storage ────────────────────────────────────────────────────────


@storage_app.command("status")
def storage_status() -> int:
    """Storage status across the three tiers (via /admin/status, or a local read-only probe)."""
    try:
        code, data = _admin_call("GET", "/admin/status")
        if code == 200:
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0
        print(f"✗ /admin/status HTTP {code}")
        return 1
    except typer.Exit:
        pass  # proxy 未运行 → 本地只读探测
    from bladex_proxy.config import ProxyConfig
    cfg = ProxyConfig()
    print("(proxy not running -- local read-only probe)")
    try:
        from bladex_proxy.storage.memory_hub import MemoryHub
        hub = MemoryHub(cfg.rocksdb_path, secondary=True)
        hub.open()
        print(f"  Memory Hub: {hub.count()} turns ({cfg.rocksdb_path})")
        hub.close()
    except Exception as e:  # noqa: BLE001
        print(f"  Memory Hub: unreadable ({e})")
    try:
        from bladex_proxy.storage.memory_index import MemoryIndex
        index = MemoryIndex(cfg.index_path, embedder=None, read_only=True)
        index.open()
        print(f"  Memory Index: {index.fact_count()} facts / {len(index.all_matters())} matters"
              f"（{cfg.index_path}）")
        index.close()
    except Exception as e:  # noqa: BLE001
        print(f"  Memory Index: unreadable ({e})")
    return 0


@storage_app.command("rebuild")
def storage_rebuild(
    concurrency: int = typer.Option(-1, "--concurrency"),
    direct: bool = typer.Option(False, "--direct"),
    since: str = typer.Option("", "--since"),
    until: str = typer.Option("", "--until"),
    agents: str = typer.Option("", "--agents"),
    exclude_agents: str = typer.Option("", "--exclude-agents"),
) -> int:
    """Full Memory Index rebuild (same as `bladex sync run --full`; zero-downtime via delegate)."""
    return sync_run(full=True, concurrency=concurrency, max_turns=0, since=since,
                    until=until, agents=agents, exclude_agents=exclude_agents,
                    direct=direct, pickup_timeout=10.0, log="")


@inspect_app.command("hub", context_settings={"allow_extra_args": True,
                                             "ignore_unknown_options": True})
def inspect_hub(ctx: typer.Context) -> int:
    """Inspect Memory Hub (passes every argument through to scripts/inspect_hub.py)."""
    return _run_repo_script("scripts/inspect_hub.py", list(ctx.args), runner="python")


@inspect_app.command("index", context_settings={"allow_extra_args": True,
                                             "ignore_unknown_options": True})
def inspect_index(ctx: typer.Context) -> int:
    """Inspect Memory Index (passes arguments through to scripts/inspect_index.py)."""
    return _run_repo_script("scripts/inspect_index.py", list(ctx.args), runner="python")


# ── memory ─────────────────────────────────────────────────────────


@memory_app.command("search")
def memory_search(
    query: str = typer.Argument(..., help="Search query (semantic + filters)"),
    k: int = typer.Option(10, "--k", help="How many results"),
    user: str = typer.Option("", "--user", help="Filter by user_id"),
) -> int:
    """Semantic search over facts (GET /admin/facts?q=)."""
    code, data = _admin_call("GET", "/admin/facts",
                             params={"q": query, "limit": k, "user_id": user})
    if code != 200:
        return _print_op_result(code, data)
    facts = data.get("facts", [])
    if not facts:
        print("(no matches)")
        return 0
    for f in facts:
        content = (f.get("content", "") or "").replace("\n", " ")
        if len(content) > 80:
            content = content[:77] + "..."
        print(f"  {f.get('id', '?'):20s} [{f.get('kind', '?'):10s}] {content}")
    print(f"Total {data.get('total', len(facts))}. Use `bladex memory show <id>` for details.")
    return 0


@memory_app.command("show")
def memory_show(fact_id: str = typer.Argument(...)) -> int:
    """Show one fact (GET /admin/facts/{id})."""
    code, data = _admin_call("GET", f"/admin/facts/{fact_id}")
    if code != 200:
        return _print_op_result(code, data)
    print(json.dumps(data, ensure_ascii=False, indent=2))
    return 0


@memory_app.command("forget")
def memory_forget(
    fact_id: str = typer.Argument(...),
    yes: bool = typer.Option(False, "--yes", help="Skip confirmation"),
) -> int:
    """Delete a fact (Hub tombstone + Memory Index cascade; never revived on rebuild)."""
    if not yes and not typer.confirm(f"Delete fact {fact_id}? (tombstone semantics; never revived on rebuild)"):
        print("Cancelled.")
        return 130
    code, data = _admin_call("DELETE", f"/admin/facts/{fact_id}")
    return _print_op_result(code, data)


#: MQ-I30：轮询后台 job 的间隔，与连续多少次读不到状态就放弃等待（服务端仍可能在跑）。
_JOB_POLL_INTERVAL_S = 1.0
_JOB_POLL_MAX_MISSES = 30


def _wait_admin_job(job_id: str, what: str, check_cmd: str) -> dict | None:
    """轮询 `GET /admin/jobs/{job_id}` 到 done / failed（MQ-I30）。

    返回 done 时的 `result`；失败 / 丢失联系返回 None（已打印原因与核对命令）。
    **不设总时长上限**：只要服务端还在答、job 还在 running 就一直等——长任务的
    "客户端说失败、服务端已成"正是本条要消灭的形状。连续 `_JOB_POLL_MAX_MISSES`
    次读不到状态才放弃，并如实说"服务端可能仍在执行"。
    """
    misses = 0
    last_done = -1
    while True:
        code, data = _admin_call("GET", f"/admin/jobs/{job_id}")
        job = (data or {}).get("job") if code == 200 else None
        if not isinstance(job, dict):
            misses += 1
            if misses >= _JOB_POLL_MAX_MISSES:
                print(f"Lost contact with the proxy while waiting for job {job_id} ({what}); "
                      f"the server may still be executing. Check with: {check_cmd}")
                return None
            time.sleep(_JOB_POLL_INTERVAL_S)
            continue
        misses = 0
        state = job.get("state")
        if state == "done":
            return job.get("result") or {}
        if state == "failed":
            print(f"✗ job {job_id} failed after {job.get('done', '?')}/{job.get('total', '?')} "
                  f"turns: {job.get('error') or '(no error text)'}")
            print(f"  Tombstones are append-only and this path is idempotent -- "
                  f"check what is left with: {check_cmd}")
            return None
        if job.get("done") != last_done:
            last_done = job.get("done")
            print(f"  ... {what}: {last_done}/{job.get('total', '?')} turns")
        time.sleep(_JOB_POLL_INTERVAL_S)


#: 可变注解（`list[str]`）的默认值不能就地写 typer.Argument(...)——ruff B008
#: 只豁免 str/bool 这类不可变注解。按它给的第二条路：模块级单例。
_PROJECT_IDS_ARG = typer.Argument(..., help="One or more p-<sha12> project ids")


@memory_app.command("tombstone-project")
def memory_tombstone_project(
    project_ids: list[str] = _PROJECT_IDS_ARG,
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Preview only: per-project impact, write nothing"),
    yes: bool = typer.Option(False, "--yes", help="Skip confirmation"),
) -> int:
    """Tombstone every turn whose identity.project_id is one of these projects.

    \f
    批 T0b 路径②（拍板 (3) 的"按 project 作用域整体清"那条线）。此前**全仓没有任何
    CLI / 端点能做这件事**：`memory forget` 是 fact 级、`ledger tombstone` 是账本窗口级。

    🔴 **空 project（Global）一律拒绝** —— 那是 project 识别恒空那段时期的全部流量
    （live ≈17,047 轮），拍板 (3) 明写那一半只能按账本清。服务端与计算层各拒一次。

    🔴 **每个 project 单独出读数**，不只给总数：批量操作最容易出的错是"总数看着对、
    其中一个 project 选错了"。

    **只墓碑 turn**，一条 LEDGER 墓碑都不打；`ledgers` 列只是影响面读数。
    **幂等**：窗口来自 `scan_meta_snapshot()`（已滤墓碑）⇒ 连跑两次读数不重复计。
    """
    _require_server_contract(f"tombstone-project ({len(project_ids)} project(s))")
    body = {"project_ids": list(project_ids), "dry_run": bool(dry_run)}
    if not dry_run and not yes and not typer.confirm(
            f"Tombstone every turn in {len(project_ids)} project(s)? "
            f"(append-only; no undo) -- run with --dry-run first if you have not"):
        print("Cancelled.")
        return 130
    if dry_run:
        code, data = _admin_call("POST", "/admin/turns/tombstone-by-project", body=body)
    else:
        code, data = _admin_call(
            "POST", "/admin/turns/tombstone-by-project", body=body,
            timeout=_admin_long_timeout_s())
    if code != 200:
        return _print_op_result(code, data)
    if not dry_run:
        # MQ-I30：真跑是后台 job——服务端先回 submitted，结果轮询取回后照旧渲染。
        job_id = data.get("job_id")
        if data.get("status") != "submitted" or not job_id:
            print(f"Unexpected response from the proxy (expected status=submitted with a "
                  f"job_id): {json.dumps(data, ensure_ascii=False)[:300]}")
            return 1
        import shlex
        check = "bladex memory tombstone-project " + shlex.join(project_ids) + " --dry-run"
        print(f"Submitted job {job_id}: tombstoning {data.get('total_turns', '?')} turns "
              f"in the background; waiting for it to finish ...")
        result = _wait_admin_job(job_id, "tombstone-project", check)
        if result is None:
            return 1
        data = result
    if dry_run:
        _require_dry_run_ack(data, f"tombstone turns in {len(project_ids)} project(s)")
        print(f"DRY RUN -- nothing written. Would tombstone "
              f"{data.get('total_turns', '?')} turns across "
              f"{len(data.get('projects') or [])} project(s).")
    else:
        print(f"✓ tombstoned {data.get('total_turns', '?')} turns across "
              f"{len(data.get('projects') or [])} project(s).")
    for row in data.get("projects") or []:
        agents = ", ".join(f"{a}:{n}" for a, n in sorted((row.get("agents") or {}).items()))
        name = "/".join(row.get("project_names") or []) or "(no name)"
        print(f"  {row.get('project_id', '?'):16s} {name:24s} turns={row.get('turn_count', '?'):>6} "
              f"ledgers={row.get('ledgers', '?'):>3}  {row.get('first_ts') or '?'} .. {row.get('last_ts') or '?'}")
        print(f"      agents: {agents or '(none)'}")
        mem = row.get("memory")
        if mem:
            print(f"      tombstones written: {mem.get('turn_tombstones', '?')} "
                  f"(index cascaded {mem.get('index_cascaded', '?')}, "
                  f"deferred {mem.get('index_deferred', '?')})")
    age = data.get("backup_age_days")
    if age is None:
        print("  ⚠ backup: NO-DATA -- no data/backup_* found. This operation is irreversible.")
    else:
        print(f"  backup: newest data/backup_* is {age} days old.")
    print("  📌 ledger objects are NOT touched by this path -- clear those separately.")
    return 0


# ── matter ─────────────────────────────────────────────────────────


@matter_app.command("list")
def matter_list(
    status: str = typer.Option("", "--status", help="Filter by status (active/closed/...)"),
    limit: int = typer.Option(50, "--limit"),
) -> int:
    code, data = _admin_call("GET", "/admin/matters",
                             params={"status": status, "limit": limit})
    if code != 200:
        return _print_op_result(code, data)
    for m in data.get("matters", []):
        print(f"  {m.get('matter_id', '?'):24s} [{m.get('status', '?'):8s}]"
              f" [{m.get('origin', '?'):6s}] {m.get('title', '')}")
    print(f"Total {data.get('total', 0)}.")
    return 0


@matter_app.command("show")
def matter_show(matter_id: str = typer.Argument(...)) -> int:
    code, data = _admin_call("GET", f"/admin/matters/{matter_id}")
    if code != 200:
        return _print_op_result(code, data)
    m = data.get("matter", {})
    print(f"{m.get('matter_id')}  [{m.get('status')}]  {m.get('title')}")
    if m.get("summary"):
        print(f"  summary: {m['summary']}")
    print(f"  edges: {data.get('edge_count', 0)} (manual {data.get('manual_edge_count', 0)})"
          f"  sessions: {len(data.get('session_keys', []))}")
    for f in data.get("facts", [])[:20]:
        content = (f.get("content", "") or "").replace("\n", " ")[:70]
        print(f"    - {f.get('id', '?')}: {content}")
    return 0


@matter_app.command("create")
def matter_create(title: str = typer.Argument(...),
                  summary: str = typer.Option("", "--summary")) -> int:
    code, data = _admin_call("POST", "/admin/matters",
                             body={"title": title, "summary": summary})
    return _print_op_result(code, data)


@matter_app.command("assign")
def matter_assign(
    matter_id: str = typer.Argument(...),
    target_key: str = typer.Argument(..., help="fact_id or session prefix"),
    target_type: str = typer.Option("fact", "--type", help="fact | session"),
) -> int:
    """Manually assign to a Matter (manual overrides auto)."""
    code, data = _admin_call("POST", f"/admin/matters/{matter_id}/assign",
                             body={"target_type": target_type, "target_key": target_key})
    return _print_op_result(code, data)


@matter_app.command("merge")
def matter_merge(
    target_id: str = typer.Argument(..., help="Merge into (kept)"),
    source_id: str = typer.Argument(..., help="Merged from (absorbed, then closed)"),
) -> int:
    code, data = _admin_call("POST", f"/admin/matters/{target_id}/merge",
                             body={"source_matter_id": source_id})
    return _print_op_result(code, data)


@matter_app.command("merge-candidates")
def matter_merge_candidates(limit: int = typer.Option(50, "--limit")) -> int:
    """List likely-duplicate Matter pairs (topic-key overlap). Detection only --
    merging always goes through `matter merge` (manual sovereignty, journaled)."""
    code, data = _admin_call("GET", f"/admin/matters/merge_candidates?limit={limit}")
    if code != 200 or not isinstance(data, dict):
        return _print_op_result(code, data)
    pairs = data.get("candidates", [])
    if not pairs:
        typer.echo("No merge candidates (no high-overlap matter pairs).")
        return 0
    for p in pairs:
        typer.echo(f"overlap={p['overlap']} jaccard={p['jaccard']}  "
                   f"{p['source_matter_id']} ({p['source_title'][:30]!r}, {p['source_edges']} edges)"
                   f" -> {p['target_matter_id']} ({p['target_title'][:30]!r}, {p['target_edges']} edges)")
        typer.echo(f"    shared: {', '.join(p['shared_keys'])}")
        typer.echo(f"    apply:  bladex matter merge {p['target_matter_id']} {p['source_matter_id']}")
    return 0


@matter_app.command("detach")
def matter_detach(
    matter_id: str = typer.Argument(...),
    edge_id: str = typer.Option("", "--edge-id"),
    target_key: str = typer.Option("", "--target-key"),
    target_type: str = typer.Option("fact", "--type"),
) -> int:
    body: dict = {"edge_id": edge_id}
    if target_key:
        body = {"target_key": target_key, "target_type": target_type}
    code, data = _admin_call("POST", f"/admin/matters/{matter_id}/detach", body=body)
    return _print_op_result(code, data)


@matter_app.command("split")
def matter_split(
    matter_id: str = typer.Argument(...),
    new_title: str = typer.Option(..., "--new-title"),
    edge_ids: list[str] = typer.Option([], "--edge-id", help="Edge to split out (repeatable)"),  # noqa: B006, B008
) -> int:
    code, data = _admin_call("POST", f"/admin/matters/{matter_id}/split",
                             body={"new_title": new_title, "edge_ids": list(edge_ids)})
    return _print_op_result(code, data)


@matter_app.command("close")
def matter_close(matter_id: str = typer.Argument(...)) -> int:
    code, data = _admin_call("POST", f"/admin/matters/{matter_id}/close", body={})
    return _print_op_result(code, data)


@matter_app.command("rename")
def matter_rename(matter_id: str = typer.Argument(...),
                  title: str = typer.Argument(...)) -> int:
    code, data = _admin_call("POST", f"/admin/matters/{matter_id}/rename",
                             body={"title": title})
    return _print_op_result(code, data)
