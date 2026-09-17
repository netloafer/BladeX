"""BladeX CLI（Beta T10：typer 骨架 + 生命周期 + init/doctor；T11 前置 sync 已收编）。

命令族：

    bladex --version
    bladex start [--foreground] [--proxy-only]     # redis + consolidator + proxy
    bladex stop / status / restart                 # 生命周期（收编 run_*.sh 逻辑）
    bladex init [--yes] [--force]                  # 交互生成 config/.env + routing.toml
    bladex config check                            # 启动校验（报错带定位与修法）
    bladex doctor [--warmup]                       # redis/RocksDB/embed 模型/上游/端口体检
    bladex sync run/status                         # 手工同步 Memory Hub→Memory Index（ADR-0009 §12）
    bladex queue status/flush [--apply]            # Pipeline→Memory Hub 那段队列（2026-08-06）

设计要点：
  - 生命周期收编 config/run_proxy.sh + run_consolidator.sh（脚本留薄壳）；
    stop 不动本地 Redis（数据面便宜且被 Pipeline 懒恢复依赖，ADR-0017）。
  - sync delegate 模式：CLI=控制面，consolidator=执行面（sync_control.py Redis
    协议）；direct 模式本进程执行（Memory Hub 恒 secondary 只读，proxy 无需停）。
  - 兼容层：`main(argv) -> int` 维持旧签名（scripts/bladex.py 薄壳与
    test_sync_cli 依赖），内部走 typer（standalone_mode=False 拿返回码）。

# 09-06 F0.1：本文件原为 3,465 行的 `cli.py`，零行为拆成包——
#   __init__.py     `app` 根 Typer + `main` + add_typer 登记 + 共享 helper（被 monkeypatch 的那些）
#   lifecycle.py    init / start / stop / status / doctor / _start_*
#   ledger_cmds.py  ledger_app
#   memory_cmds.py  memory_app / matter_app / inspect_app / storage_app
#   ops_cmds.py     queue_app / sticky_app / sync_app / connector_app / export / import / config_app
# `[project.scripts] bladex = "bladex_proxy.cli:main"` 不变。
# 🔴 monkeypatch 目标要打在**消费方模块**上（如 `bladex_proxy.cli.lifecycle._probe_ready`），
#    打在门面上只换门面的引用、消费方看不到。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import typer

# 部署根发现与 .env 加载的**唯一实现**在 deployment.py（2026-08-06 收敛）。
# 此处只留薄封装：历史调用点与测试都按这些名字来，改名不值当。
from bladex_proxy import deployment  # noqa: E402


def _repo_root() -> str:
    """部署根目录（ADR-0027 §4.3；2026-08-06 起支持向上查找 + `~/.bladex`）。

    查找顺序：`BLADEX_HOME` → cwd 及其祖先 → `~/.bladex`；都没有则退回 cwd。
    **调用时求值**：import 时冻结会让 chdir 后仍读旧位置（2026-08-03 环境污染 bug）。
    """
    return deployment.deployment_root()


def _config_search_paths() -> list[str]:
    """config/.env 的候选位置（报错时原样列给用户，不静默用空目录）。"""
    return deployment.search_paths()


def _require_config_root() -> str | None:
    """返回含 config/.env 的部署根；找不到返回 None（调用方负责报错）。"""
    return deployment.find_root()


def _chdir_to_deployment_root() -> str | None:
    """切到部署根，让所有相对路径（data/、logs/、PID 文件）有唯一落点。

    此前只有 `start` 和 `consolidator` 会切，于是 `bladex status` / `doctor` /
    `memory ...` 在别的目录跑出来的是一个**空库**——正是 `_repo_root` 文档里说的
    "症状不是报错而是数据看起来消失了"。找不到部署根时什么都不做：报错留给
    真正需要配置的那个命令去报（`--version` / `--help` / `init` 不该被拦）。

    用户在命令行上给的相对路径由 `deployment.resolve_user_path` 钉回原 cwd。
    """
    root = deployment.find_root()
    if root is None or root == os.getcwd():
        return root
    os.chdir(root)
    print(f"note: using BladeX deployment at {root}", file=sys.stderr)
    return root


def _print_config_not_found() -> None:
    """不静默用空目录：把找过的位置原样列出来（ADR-0027 §4.3）。"""
    print("Error: config/.env not found. Looked in:")
    for path in _config_search_paths():
        print(f"  - {path}")
    print(f"  - and every parent directory of {os.getcwd()}")
    print("Run `bladex init` here, or set BLADEX_HOME to your deployment directory.")


def _load_env_file() -> None:
    """加载 config/.env 并把配置冲突报到 stderr。

    2026-08-06 起 `.env` 默认压过残留的环境变量（见 deployment.py 顶部两次事故）。
    冲突走 stderr 而不是 stdout：`bladex ... --json` 之类的输出不能被它污染。
    """
    root = deployment.find_root()
    if root is None:
        return
    env_path = os.path.join(root, deployment.CONFIG_RELPATH)
    conflicts = deployment.load_env_file(root)
    lines = deployment.format_conflicts(conflicts, env_path)
    for line in lines:
        print(line, file=sys.stderr)
    # 🔴 引号错配：Python 侧读得出本意、shell `source` 却从那行吞到文件末尾，
    # 于是同一个文件两个加载器读出两个不同的环境（2026-09-01 live 病例：
    # `BLADEX_MODULE_FLASH="1'` 让 `source` 丢掉三个 BLADEX_ASSEMBLY_* 键，
    # 而运行时毫无异常）。只报不改 —— 修值是用户的事，程序的责任是别让它静默。
    for lineno, key in deployment.unbalanced_quote_lines(env_path):
        print(f"  WARNING: {env_path}:{lineno} {key} has mismatched quotes -- "
              f"Python reads it fine, but shell `source` swallows the rest of "
              f"the file from this line on.", file=sys.stderr)


app = typer.Typer(help="BladeX — memory-first LLM proxy middleware",
                  add_completion=False, no_args_is_help=True)

_PROXY_PIDFILE = "logs/proxy.pid"
_CONS_PIDFILE = "logs/consolidator.pid"
#: V-A6 embedding 模块（默认关；backend=ipc 时才需要它）
_EMBED_PIDFILE = "logs/embed.pid"
_FLASH_PIDFILE = "logs/flash.pid"   # V-F2：flash 维护进程


def _version_cb(value: bool) -> None:
    if value:
        from bladex_proxy import __version__
        print(f"bladex {__version__}")
        raise typer.Exit(0)


@app.callback()
def _root(
    version: bool = typer.Option(  # noqa: B008
        False, "--version", help="Print the version", callback=_version_cb, is_eager=True),
) -> None:
    """BladeX CLI。"""


# ── 生命周期 helpers（模块级，可单测）─────────────────────────────────────────


def _read_pidfile(path: str) -> int | None:
    try:
        pid = int(Path(path).read_text().strip())
    except (OSError, ValueError):
        return None
    return pid if _pid_alive(pid) else None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _consolidator_pid() -> int | None:
    """PID 文件优先 + 进程名兜底（手动启动场景，与 run_proxy.sh 同款）。

    进程名要匹配两种形态（ADR-0027 §4.2）：包内入口
    `python -m bladex_proxy.consolidator` 与老脚本 `run_index_consolidator.py`。
    """
    pid = _read_pidfile(_CONS_PIDFILE)
    if pid is not None:
        return pid
    for pattern in ("bladex_proxy.consolidator", "bladex-consolidator",
                    "run_index_consolidator.py"):
        try:
            out = subprocess.run(  # noqa: S603
                ["pgrep", "-f", pattern],
                capture_output=True, text=True, check=False).stdout.strip()
            if out:
                return int(out.splitlines()[0])
        except (OSError, ValueError, IndexError):
            continue
    return None


def _resolve_distill_concurrency() -> int:
    """守护进程的蒸馏并发路数（唯一真相源 = `flags.BLADEX_DISTILL_CONCURRENCY`）。

    迟解析：`main()` 已跑过 `_load_env_file`，此刻 config/.env 的值可见。
    """
    from bladex_core.flags import flag_number
    return max(1, int(flag_number("BLADEX_DISTILL_CONCURRENCY")))


def _http_json(url: str, key: str = "", timeout: float = 3.0) -> dict | None:
    req = urllib.request.Request(url)  # noqa: S310 —— 仅本机 admin 面
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return None


def _print_admin_unavailable() -> None:
    """proxy 活着但 /admin/status 读不到时，说清为什么（2026-08-06 用户报的现象）。

    此前这里是 `if admin: ...` 后面什么都没有——读不到就**整段消失**：
    Memory Hub / Memory Index / Queue / Disk / Memory 五行凭空不见，
    只剩 Redis + Proxy + Dashboard。用户在别的目录跑了一次，看到的是一份
    "短了一半"的输出，第一反应是"程序版本不对"——这正是 ADR-0027 §3.1 说的
    绿灯骗人：**缺席比报错更难查**，因为缺席不指向任何东西。

    最常见的成因就是没找到部署根：没有 `config/.env` → 没有 admin key →
    `/admin/*` 401 → 整段静默消失。所以先报这一条。
    """
    root = deployment.find_root()
    if root is None:
        print("  Memory:       unavailable -- no config/.env found, so there is no admin key "
              "to read /admin/status with.")
        print(f"                Set BLADEX_HOME to your deployment directory (looked in "
              f"{', '.join(_config_search_paths())} and the parents of {os.getcwd()}).")
        return
    if not _admin_key():
        print(f"  Memory:       unavailable -- no BLADEX_ADMIN_KEYS/BLADEX_CLIENT_KEYS in "
              f"{os.path.join(root, deployment.CONFIG_RELPATH)}.")
        return
    print("  Memory:       unavailable -- /admin/status rejected the key or did not answer. "
          "The proxy may be running with a different config than this shell sees "
          "(compare `bladex config check`).")


def _first_key_of(raw: str) -> str:
    """取 `key||label||key2||label2` 里的第一把 key。

    ADR-0027 §5.2：格式权威是 `auth.KeyStore.parse`——**纯 `||` 分隔，不是逗号**。
    此前这里先按 `,` split，label 里带逗号就会截断出错误的 key（三处口径不一之一）。
    """
    parts = [p.strip() for p in raw.split("||") if p.strip()]
    return parts[0] if parts else ""


def _first_client_key() -> str:
    """数据面 key（/v1/*）。"""
    return _first_key_of(os.environ.get("BLADEX_CLIENT_KEYS", ""))


def _admin_key() -> str:
    """管理面 key（/admin/*）——ADR-0027 §2.2：优先独立 admin key，未配回落 client key。"""
    return _first_key_of(os.environ.get("BLADEX_ADMIN_KEYS", "")) or _first_client_key()


def _probe_ready(url: str, timeout: float = 2.0) -> dict | None:
    """读 /ready，**503 也要读出 body**。

    /ready 在未就绪时返回 503 + `{"ready": false, "checks": {...}}`——正是我们最需要
    看清楚的那种情况。通用的 `_http_json` 把 HTTPError 当失败吞掉会让"记忆管线死了"
    退化成"探不到"，恰好丢掉本要报给用户的信息。
    """
    req = urllib.request.Request(url)  # noqa: S310 —— 本机探针
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            return json.loads(e.read().decode("utf-8"))
        except (ValueError, OSError):
            return {"ready": False, "checks": {}}
    except (urllib.error.URLError, OSError, ValueError):
        return None


def assets_dir() -> Path:
    """包内模型面资产目录（`bladex_proxy/assets`）。"""
    # F0.1 拆包：本文件从 `bladex_proxy/cli.py` 移到 `bladex_proxy/cli/__init__.py`，多了一层目录。
    return Path(__file__).resolve().parent.parent / "assets"


def _admin_base() -> str:
    from bladex_proxy.config import ProxyConfig
    cfg = ProxyConfig()
    return f"http://{cfg.host}:{cfg.port}"


def _http_request(method: str, url: str, key: str = "",
                  body: dict | None = None, timeout: float = 15.0) -> tuple[int, dict]:
    """admin API 调用（urllib，仅本机管理面）。返回 (status, json)。"""
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)  # noqa: S310
    req.add_header("Content-Type", "application/json")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode("utf-8"))
        except ValueError:
            return e.code, {}
    except (urllib.error.URLError, OSError) as e:
        print(f"Error: proxy unreachable ({url}) -- run `bladex start` first. ({e})")
        raise typer.Exit(1) from e


def _admin_call(method: str, path: str, body: dict | None = None,
                params: dict | None = None, timeout: float = 15.0) -> tuple[int, dict]:
    from urllib.parse import urlencode
    url = _admin_base() + path
    if params:
        url += "?" + urlencode({k: v for k, v in params.items() if v not in ("", None)})
    return _http_request(method, url, key=_admin_key(), body=body, timeout=timeout)


def _print_op_result(status_code: int, data: dict) -> int:
    if status_code == 200:
        print(f"✓ {data.get('status', 'ok')}: "
              + json.dumps({k: v for k, v in data.items() if k != 'status'},
                           ensure_ascii=False))
        return 0
    if status_code == 202:
        print("⏳ deferred: the admin event is in Memory Hub, but the consolidator holds the Memory Index write lock -- "
              "it will take effect on the next rebuild/replay")
        return 0
    print(f"✗ HTTP {status_code}: {json.dumps(data, ensure_ascii=False)}")
    return 1


def _run_repo_script(rel: str, args: list[str], runner: str = "bash") -> int:
    script = Path(rel)
    if not script.is_file():
        print(f"Error: not found: {rel} (this command must run from the repository root)")
        return 1
    cmd = ([runner, str(script)] if runner == "bash"
           else [sys.executable, str(script)]) + args
    return subprocess.run(cmd, check=False).returncode  # noqa: S603



# ── 子命令族登记（顺序与拆前逐字相同：sync/config → storage/memory/matter/queue/sticky/inspect/ledger → connector）──
# 🔴 必须放在上面全部 helper 之后：子模块在 import 时 `from bladex_proxy.cli import app, _admin_call …`，
#    此时本包处于半初始化状态，只有已执行过的名字可见。
from bladex_proxy.cli import ledger_cmds, lifecycle, memory_cmds, ops_cmds  # noqa: E402,F401

app.add_typer(ops_cmds.sync_app, name="sync")
app.add_typer(ops_cmds.config_app, name="config")
app.add_typer(memory_cmds.storage_app, name="storage")
app.add_typer(memory_cmds.memory_app, name="memory")
app.add_typer(memory_cmds.matter_app, name="matter")
app.add_typer(ops_cmds.queue_app, name="queue")
app.add_typer(ops_cmds.sticky_app, name="sticky")
app.add_typer(memory_cmds.inspect_app, name="inspect")
app.add_typer(ledger_cmds.ledger_app, name="ledger")
app.add_typer(ops_cmds.connector_app, name="connector")

# ── 门面 re-export：拆前 `bladex_proxy.cli` 上的全部顶层名（守卫 `test_facade_exports.py`）──
from bladex_proxy.cli.ledger_cmds import (  # noqa: E402,F401
    _DOCTOR_GOAL_PREFIX,
    _DOCTOR_STALE_DAYS,
    _doctor_age_days,
    _doctor_findings,
    _doctor_norm_title,
    _doctor_scope_key,
    _ledger_active_cell,
    _ledger_bind_legacy_matters,
    _ledger_entry_counts,
    _ledger_goal_cell,
    _ledger_time_cell,
    ledger_app,
    ledger_doctor,
    ledger_list,
    ledger_show,
)
from bladex_proxy.cli.lifecycle import (  # noqa: E402,F401
    _ASSET_TARGETS,
    _E2E_RECALL_QUERY,
    _E2E_TEMPLATE,
    _ENV_MINIMAL_TEMPLATE,
    _ROUTING_MINIMAL_TEMPLATE,
    _ago,
    _consolidator_activity,
    _consolidator_command,
    _embed_listening,
    _embed_pid,
    _embed_wanted,
    _ensure_redis,
    _flash_command,
    _flash_pid,
    _flash_status_line,
    _fmt_bytes,
    _fmt_traffic_row,
    _format_queue,
    _hidden_pth_files,
    _install_model_facing_assets,
    _is_local_host,
    _print_degradation_banner,
    _print_funnel,
    _print_memory_config,
    _print_traffic,
    _probe_port,
    _probe_upstream,
    _prune_logs,
    _redis_hostport,
    _redis_ping,
    _report_startup_state,
    _run_e2e_check,
    _scrape_metrics,
    _stale_data_dirs,
    _start_consolidator,
    _start_embed,
    _start_flash,
    _start_proxy,
    _stop_by_pidfile,
    consolidator,
    doctor,
    embed,
    init,
    restart,
    start,
    status,
    stop,
)
from bladex_proxy.cli.memory_cmds import (  # noqa: E402,F401
    inspect_app,
    inspect_hub,
    inspect_index,
    matter_app,
    matter_assign,
    matter_close,
    matter_create,
    matter_detach,
    matter_list,
    matter_merge,
    matter_merge_candidates,
    matter_rename,
    matter_show,
    matter_split,
    memory_app,
    memory_forget,
    memory_search,
    memory_show,
    storage_app,
    storage_rebuild,
    storage_status,
)
from bladex_proxy.cli.ops_cmds import (  # noqa: E402,F401
    _build_export_worker,
    _fmt_progress,
    _mask_secret,
    _print_orphan_profile,
    _queue_lines,
    _run_direct,
    _SyncLog,
    _try_delegate,
    _upstream_key_problems,
    backup,
    cmd_sync_run,
    cmd_sync_status,
    config_app,
    config_check,
    config_show,
    connector_app,
    connector_reset,
    connector_run,
    connector_status,
    export_cmd,
    import_cmd,
    queue_app,
    queue_flush,
    queue_status,
    render_effective_config,
    restore,
    sticky_app,
    sticky_flush,
    sticky_status,
    sync_app,
    sync_run,
    sync_status,
)

# ── 入口（兼容层：scripts/bladex.py 与 test_sync_cli 依赖 main(argv)->int）──


def main(argv: list[str] | None = None) -> int:
    deployment.remember_invocation_cwd()
    _load_env_file()
    _chdir_to_deployment_root()
    import click
    try:
        rv = app(args=argv, standalone_mode=False, prog_name="bladex")
    except (typer.Exit, click.exceptions.Exit) as e:
        return int(getattr(e, "exit_code", 0) or 0)
    except SystemExit as e:
        return int(e.code or 0)
    except click.ClickException as e:
        e.show()
        return e.exit_code or 2
    except click.exceptions.Abort:
        print("Cancelled.")
        return 130
    return int(rv) if isinstance(rv, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
