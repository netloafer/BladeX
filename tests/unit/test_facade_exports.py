"""F0.1 拆包门面守卫（2026-09-06）：三个巨石拆成包后，门面模块必须逐名保住拆前的顶层名。

`agency.py`（2,365 行）/ `server.py`（4,141 行）/ `cli.py`（3,465 行）零行为拆成
`agency/` `server/` `cli/` 三个包。生产代码与 ~60 个测试文件按 `from bladex_proxy.agency import X`
这类名字取符号，拆包**不许**让任何一个名字消失。名单 = 拆前三个文件 `ast` 顶层定义的
全部名字（函数 / 类 / 模块级赋值），**写死**在本文件里——不从当前源码推导，
否则拆包时漏掉一个名字、名单也跟着漏，守卫假绿。

另外钉两条结构事实（拆包时各栽过一次）：
- `bladex_proxy.server.app` 必须是 FastAPI 实例（`uvicorn bladex_proxy.server:app`），
  所以 create_app 所在子模块**不能叫 `app.py`**——同名子模块会被门面上的 `app` 属性遮住；
- `bladex_proxy.cli.assets_dir()` 指向 `bladex_proxy/assets`（cli 成子包后多了一层目录，
  `Path(__file__).parent` 会指到 `cli/assets`，G7 冷安装那条线当场断）。
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

_AGENCY = [
    "logger", "LEDGER_BLOCK_OPEN", "LEDGER_BLOCK_CLOSE", "ABOUT_BLOCK_OPEN", "ABOUT_BLOCK_CLOSE",
    "INJECTION_MARKERS", "_FIRST_STEP_WITH_LEDGER", "SUBTASK_TAG", "_MACHINE_TEXT_MARKS",
    "_machine_text_mark", "_FIRST_STEP_NO_LEDGER", "_MATCHING_LEDGERS_HEADER", "_DESPITE_MATCH_SCORE",
    "load_ledger_template", "_SYSTEM_NOTE_FILES", "load_system_notes", "load_agents_roster",
    "_GOAL_MAX_CHARS", "_WRAPPER_TAG_RE", "_BARE_PREFIX_RES", "_last_user_text", "_call_names",
    "_loop_cost", "USER_NAMED_TITLE_MIN_CHARS", "_user_named_ledger", "tool_context", "AgencyRuntime",
    "intercept_chat_stream", "_SSE_KEEPALIVE", "_Narrator", "_ev", "_ResponsesNarrator",
    "_AnthropicNarrator", "_ChatNarrator", "_idle_too_long", "_narrate_on", "_keepalive_interval_s",
    "_narrate_until_done", "_intercept_protocol_stream", "_rewrite_stop_anthropic",
    "_rewrite_stop_responses", "_synth_responses_tool_call_events", "_synth_anthropic_tool_call_events",
    "_synth_anthropic_text_events", "_SYNTH_ITEM_ID", "_synth_responses_text_events",
    "intercept_anthropic_stream", "intercept_responses_stream", "build_agency",
]

_SERVER = [
    "logger", "_LOOPBACK_HOSTS", "warn_if_insecure_bind", "enforce_admin_key_policy", "lifespan",
    "create_app", "_INDEX_WRITE_RETRIES", "_INDEX_WRITE_RETRY_DELAY_S", "_open_writable_index",
    "_apply_to_index", "require_admin_key", "_admin_key_dep", "_hub_unavailable", "_admin_response",
    "_resolve_with_memory", "_record_inject_metrics", "_validate_sensitivity_judge",
    "_resolve_sensitivity", "_detect_prefix_changed", "_capability_error_response",
    "_extract_output_modalities", "_required_capabilities", "_extract_output_modalities_responses",
    "_estimate_context_chars", "RoundPrep", "_prepare_round", "_apply_agency_surfaces", "_call_hooks",
    "_extract_bearer", "_message_text", "_AGENT_DETAIL_SAMPLE", "_AGENT_DETAIL_TEXT_CHARS",
    "_collect_headers", "_is_anthropic_client", "_build_model_obj", "_handle_stream",
    "_handle_non_stream", "_REDIS_RECOVER_COOLDOWN_S", "_last_recover_attempt", "_redis_skip_count",
    "_try_recover_redis", "_enqueue_inner_loop_turns", "_enqueue_turn", "_warn_if_output_truncated",
    "_enqueue_turn_shielded", "_build_request_params", "_build_response_meta_from_capture",
    "_message_as_dict", "_loop_reply", "_build_response_meta_from_response", "_extract_usage_dict",
    "_inject_top_k", "_build_decision_meta", "_store_raw_request", "_resolve_task_units",
    "_extract_tool_events_from_response", "_extract_query", "_build_kwargs",
    "_handle_anthropic_stream", "_shim_processed_response", "_intercept_non_stream",
    "_handle_anthropic_non_stream", "_handle_responses_stream", "_handle_responses_non_stream", "app",
]

_CLI = [
    "_repo_root", "_config_search_paths", "_require_config_root", "_chdir_to_deployment_root",
    "_print_config_not_found", "_load_env_file", "_SyncLog", "_fmt_progress", "_try_delegate",
    "_run_direct", "cmd_sync_run", "cmd_sync_status", "app", "sync_app", "config_app",
    "_PROXY_PIDFILE", "_CONS_PIDFILE", "_EMBED_PIDFILE", "_FLASH_PIDFILE", "_version_cb", "_root",
    "_read_pidfile", "_pid_alive", "_redis_hostport", "_redis_ping", "_is_local_host", "_ensure_redis",
    "_consolidator_pid", "_consolidator_command", "_flash_command", "_flash_pid", "_start_flash",
    "_resolve_distill_concurrency", "_embed_pid", "_embed_wanted", "_embed_listening", "_start_embed",
    "_start_consolidator", "_start_proxy", "_stop_by_pidfile", "_http_json", "_print_admin_unavailable",
    "_first_key_of", "_first_client_key", "_admin_key", "start", "_prune_logs", "_probe_ready",
    "_report_startup_state", "stop", "embed", "consolidator", "_fmt_traffic_row", "_scrape_metrics",
    "_print_funnel", "_print_traffic", "_fmt_bytes", "_print_memory_config", "_consolidator_activity",
    "_format_queue", "_flash_status_line", "_ago", "_print_degradation_banner", "status", "restart",
    "_ENV_MINIMAL_TEMPLATE", "_ROUTING_MINIMAL_TEMPLATE", "_ASSET_TARGETS", "assets_dir",
    "_install_model_facing_assets", "init", "_stale_data_dirs", "_upstream_key_problems",
    "_mask_secret", "render_effective_config", "config_show", "config_check", "_probe_upstream",
    "_probe_port", "doctor", "_hidden_pth_files", "_E2E_TEMPLATE", "_E2E_RECALL_QUERY",
    "_run_e2e_check", "sync_run", "sync_status", "storage_app", "memory_app", "matter_app",
    "inspect_app", "queue_app", "sticky_app", "ledger_app", "_admin_base", "_http_request",
    "_admin_call", "_print_op_result", "_queue_lines", "queue_status", "_print_orphan_profile",
    "queue_flush", "sticky_status", "sticky_flush", "storage_status", "storage_rebuild",
    "_run_repo_script", "backup", "restore", "inspect_hub", "inspect_index", "connector_app",
    "_build_export_worker", "connector_run", "connector_status", "connector_reset", "export_cmd",
    "import_cmd", "memory_search", "memory_show", "memory_forget", "matter_list", "matter_show",
    "matter_create", "matter_assign", "matter_merge", "matter_merge_candidates", "matter_detach",
    "matter_split", "matter_close", "matter_rename", "_ledger_entry_counts", "_ledger_time_cell",
    "_ledger_goal_cell", "_ledger_active_cell", "ledger_list", "_DOCTOR_GOAL_PREFIX",
    "_DOCTOR_STALE_DAYS", "_doctor_norm_title", "_doctor_age_days", "_doctor_scope_key",
    "_doctor_findings", "ledger_doctor", "_ledger_bind_legacy_matters", "ledger_show", "main",
]


@pytest.mark.parametrize("modname, names, expected", [
    ("bladex_proxy.agency", _AGENCY, 49),
    ("bladex_proxy.server", _SERVER, 65),
    ("bladex_proxy.cli", _CLI, 145),
])
def test_facade_keeps_every_pre_split_name(modname: str, names: list[str], expected: int) -> None:
    assert len(names) == expected, "名单被动过——它是拆前 ast 顶层名的快照，不该增删"
    mod = importlib.import_module(modname)
    missing = [n for n in names if not hasattr(mod, n)]
    assert not missing, f"{modname} 门面丢了拆前的名字：{missing}"


def test_server_app_is_the_fastapi_instance_and_factory_is_not_named_app() -> None:
    from bladex_proxy import server
    from fastapi import FastAPI

    assert isinstance(server.app, FastAPI), "uvicorn 入口 `bladex_proxy.server:app` 必须是 FastAPI 实例"
    pkg_dir = Path(server.__file__).parent
    assert not (pkg_dir / "app.py").exists(), \
        "server/app.py 会被门面上的 `app`（FastAPI 实例）遮住——create_app 所在子模块不能叫 app.py"
    assert server.create_app.__module__ == "bladex_proxy.server.app_factory"


def test_agency_runtime_keeps_handlers_via_mixin() -> None:
    """handlers 以 mixin 挂回：`AgencyRuntime._h_*` 经 MRO 仍可取（既有 `inspect.getsource` 断言依赖）。"""
    from bladex_proxy.agency import AgencyRuntime

    for name in ("_h_memory_search", "_h_ledger_read", "_h_ledger_update", "_h_ledger_switch",
                 "_maybe_migrate_binding", "_coauthor_note", "_mark_subtask_on_parent",
                 "session_owns_active_ledger"):
        assert callable(getattr(AgencyRuntime, name)), f"AgencyRuntime.{name} 丢了"


def test_cli_assets_dir_points_at_package_assets() -> None:
    from bladex_proxy import cli

    d = cli.assets_dir()
    assert d.name == "assets" and d.parent.name == "bladex_proxy", d
    assert (d / "ledger-template.md").is_file(), "cli 成子包后 assets_dir 指错了一层"


def test_cli_subcommand_groups_registered_in_pre_split_order() -> None:
    from bladex_proxy import cli

    assert [g.name for g in cli.app.registered_groups] == [
        "sync", "config", "storage", "memory", "matter", "queue", "sticky", "inspect", "ledger", "connector"]
