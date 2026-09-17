"""FastAPI proxy 核心 — 串起全流程（T8，ADR-0008 §6.2）。

/v1/chat/completions: 验 key → 认身份 → 注入 → StreamingResponse(转发+捕获) → 异步入库

ISSUE-3: 入库在 try/finally 里执行，客户端断开也尽量存。
ISSUE-5: 上游出错时构造 Turn(status=FAILED) 入库 + 返回错误体。
ISSUE-6: 收紧类型注解。
ISSUE-8: 用 app.state 替代模块级全局变量。

# 09-06 F0.1：本文件原为 4,141 行的 `server.py`，零行为拆成包——
#   app_factory.py         create_app / lifespan / 元端点 / 启动安全基线
#                          （不叫 app.py：`bladex_proxy.server:app` 是 uvicorn 入口的 FastAPI 实例，
#                           同名子模块会被门面上的 `app` 属性遮住，monkeypatch 会打到实例上）
#   endpoints_chat.py      /v1/chat/completions (+ /v1/embeddings) 与 _handle_stream / _handle_non_stream
#   endpoints_anthropic.py /v1/messages (+ count_tokens) 与 _handle_anthropic_*
#   endpoints_responses.py /v1/responses 与 _handle_responses_*
#   admin_api.py           /admin/* 写端点 + /dashboard + require_admin_key / _open_writable_index
#   orchestration.py       _prepare_round / _apply_agency_surfaces / _enqueue_turn* / _build_* / _extract_*
# 本模块只做门面：re-export 全部原顶层名（守卫 `test_facade_exports.py`），并保留 `app = create_app()`
# 供 `uvicorn bladex_proxy.server:app`。
# 🔴 monkeypatch 目标要打在**消费方模块**上（如 `bladex_proxy.server.admin_api._open_writable_index`），
#    打在门面上只换门面的引用、消费方看不到。
"""

from __future__ import annotations

import structlog

from bladex_proxy.server.orchestration import (  # noqa: F401
    _resolve_with_memory, _record_inject_metrics, _validate_sensitivity_judge,
    _resolve_sensitivity, _detect_prefix_changed, _capability_error_response,
    _extract_output_modalities, _required_capabilities, _extract_output_modalities_responses,
    _estimate_context_chars, RoundPrep, _prepare_round, _apply_agency_surfaces, _call_hooks,
    _extract_bearer, _collect_headers, _is_anthropic_client, _build_model_obj,
    _REDIS_RECOVER_COOLDOWN_S, _last_recover_attempt, _redis_skip_count, _try_recover_redis,
    _enqueue_inner_loop_turns, _enqueue_turn, _warn_if_output_truncated, _enqueue_turn_shielded,
    _build_request_params, _build_response_meta_from_capture, _message_as_dict, _loop_reply,
    _build_response_meta_from_response, _extract_usage_dict, _inject_top_k,
    _build_decision_meta, _store_raw_request, _resolve_task_units,
    _extract_tool_events_from_response, _extract_query, _build_kwargs, _shim_processed_response,
    _intercept_non_stream, _added_text, _content_text, _build_ledger_anchor,
    _ledger_next_referenced
)
from bladex_proxy.server.admin_api import (  # noqa: F401
    _INDEX_WRITE_RETRIES, _INDEX_WRITE_RETRY_DELAY_S, _open_writable_index, _apply_to_index,
    require_admin_key, _admin_key_dep, _hub_unavailable, _admin_response, _message_text,
    _AGENT_DETAIL_SAMPLE, _AGENT_DETAIL_TEXT_CHARS, register_admin_routes
)
from bladex_proxy.server.endpoints_chat import (  # noqa: F401
    register_chat_routes, register_embeddings_routes, _handle_stream, _handle_non_stream
)
from bladex_proxy.server.endpoints_anthropic import (  # noqa: F401
    register_anthropic_routes, _handle_anthropic_stream, _handle_anthropic_non_stream
)
from bladex_proxy.server.endpoints_responses import (  # noqa: F401
    register_responses_routes, _handle_responses_stream, _handle_responses_non_stream
)
from bladex_proxy.server.app_factory import (  # noqa: F401
    _LOOPBACK_HOSTS, warn_if_insecure_bind, enforce_admin_key_policy, lifespan, create_app
)

logger = structlog.get_logger()

app = create_app()
