"""Anthropic messages 协议入站：`/v1/messages` + `/v1/messages/count_tokens` 与其流式/非流式处理。

09-06 F0.1 自 `server.py` 拆出，零行为（函数体逐字搬家）。门面 `bladex_proxy.server` re-export。
"""

from __future__ import annotations

import json
import time

import structlog
from fastapi import FastAPI, Header, Request, status
from fastapi.responses import JSONResponse, StreamingResponse

from bladex_proxy import router_sdk
from bladex_proxy.anthropic import (
    anthropic_stream_generator, approx_count_tokens, format_anthropic_response,
    parse_anthropic_request
)
from bladex_proxy.capture import CaptureResult
from bladex_proxy.config import ModelRoute, ProxyConfig
from bladex_proxy.agency import intercept_anthropic_stream
from bladex_proxy.models import (
    AgentSource, ChatCompletionRequest, DecisionMeta, Identity, RequestParams, ToolEvent,
    TurnStatus
)
from bladex_proxy.route import call_model
from bladex_proxy.server.orchestration import (
    _apply_agency_surfaces, _build_request_params, _build_response_meta_from_capture,
    _build_response_meta_from_response, _call_hooks, _collect_headers, _enqueue_turn,
    _enqueue_turn_shielded, _extract_bearer, _extract_output_modalities, _intercept_non_stream,
    _loop_reply, _prepare_round, _store_raw_request
)

logger = structlog.get_logger()


def register_anthropic_routes(app: FastAPI) -> None:
    """挂 `/v1/messages` 与 `/v1/messages/count_tokens`（闭包只捕获 `app`，函数体与拆前逐字相同）。"""
    @app.post("/v1/messages", response_model=None)
    async def anthropic_messages(
        request: Request,
        authorization: str | None = Header(None),
        x_api_key: str | None = Header(None, alias="x-api-key"),
        x_agent_id: str | None = Header(None, alias="X-Agent-ID"),
        x_session_id: str | None = Header(None, alias="X-Session-ID"),
        user_agent: str | None = Header(None, alias="User-Agent"),
    ) -> JSONResponse | StreamingResponse:
        """T4: Anthropic /v1/messages 端点（Claude Code 原生格式）。

        解析 Anthropic 请求 → 归一 → 认身份 / 注入 / 路由 / 捕获 / 存储（复用全流程）。
        """
        cfg: ProxyConfig = request.app.state.config

        # authorization(Bearer) 优先，x-api-key 兜底（Claude Code 走 x-api-key，Anthropic 风格）
        client_key = _extract_bearer(authorization) or x_api_key
        ok, reason = cfg.auth_check(client_key, endpoint="/v1/messages")
        if not ok:
            logger.warning("anthropic_auth_failed", reason=reason)
            return JSONResponse(
                status_code=401,
                content={"type": "error", "error": {"type": "authentication_error",
                        "message": f"Invalid API key: {reason}"}},
            )

        t_start = time.perf_counter()
        body = await request.json()

        # 解析 Anthropic 请求 → 归一成 OpenAI 格式
        messages, extra_kwargs = parse_anthropic_request(body)
        is_stream = body.get("stream", False)

        # G11.1：完整 header 集合进身份解析（厂商前缀 header 此前被丢在门外）。
        headers = _collect_headers(request, authorization, x_agent_id, x_session_id, user_agent)

        # 用归一后的 messages 构造一个临时 ChatCompletionRequest 供身份解析
        from bladex_proxy.models import ChatCompletionRequest
        req = ChatCompletionRequest(
            model=body.get("model", ""),
            messages=messages,
            stream=is_stream,
            **{k: v for k, v in extra_kwargs.items()
               if k in ("temperature", "max_tokens", "tools", "tool_choice", "top_p", "stop")},
        )

        output_modalities = _extract_output_modalities(body)
        # V-A1 T2 编排函数化（与 chat 端点同一实现，见彼处注释）。
        prep = await _prepare_round(request, req=req, headers=headers, body=body,
                                    output_modalities=output_modalities)
        if prep.error_response is not None:
            return prep.error_response
        identity, agent_source = prep.identity, prep.agent_source
        messages = prep.messages

        # T4: /v1/messages 原始请求体按 hash 入 __msg__ 池
        raw_request_ref = _store_raw_request(request, body)
        _tools_now, _tf_injected = _apply_agency_surfaces(
            request, prep, tools_in=extra_kwargs.get("tools"), stream=bool(is_stream))
        if _tf_injected:
            extra_kwargs["tools"] = _tools_now
        # 🔴 request_params 在 augment 之后建、读 extra_kwargs["tools"]——那才是
        # 转发上游的那份（V-R1 取证）。
        request_params = _build_request_params(
            request, req, forwarded_tools=extra_kwargs.get("tools"))
        injected_messages = prep.injected_messages
        decision_meta = prep.decision_meta
        model, route, failover = prep.model, prep.route, prep.failover
        injected_text, ms_identity, ms_inject = (
            prep.injected_text, prep.ms_identity, prep.ms_inject)
        allowed_exposure = prep.allowed_exposure

        if is_stream:
            return await _handle_anthropic_stream(
                request, model, route, failover, injected_messages, extra_kwargs,
                identity, injected_text, ms_identity, ms_inject, t_start,
                agent_source, original_messages=messages,
                request_params=request_params, decision_meta=decision_meta,
                raw_request_ref=raw_request_ref, allowed_exposure=allowed_exposure,
            )
        else:
            return await _handle_anthropic_non_stream(
                request, model, route, failover, injected_messages, extra_kwargs,
                identity, injected_text, ms_identity, ms_inject, t_start,
                agent_source, original_messages=messages,
                request_params=request_params, decision_meta=decision_meta,
                raw_request_ref=raw_request_ref,
                allowed_exposure=allowed_exposure,
            )

    # ── T26: POST /v1/messages/count_tokens（Claude Code 每轮请求前调用）──

    @app.post("/v1/messages/count_tokens", response_model=None)
    async def anthropic_count_tokens(
        request: Request,
        authorization: str | None = Header(None),
        x_api_key: str | None = Header(None, alias="X-API-Key"),
    ) -> JSONResponse:
        """T26: 本地近似 input_tokens 计数，不调上游。

        复用 parse_anthropic_request 归一化（system 折进 messages、tool_result 转换），
        再调 approx_count_tokens（含 CJK 加权、tools 计入）。
        """
        cfg_local: ProxyConfig = request.app.state.config
        client_key = _extract_bearer(authorization) or x_api_key
        ok, reason = cfg_local.auth_check(client_key, endpoint="/v1/messages/count_tokens")
        if not ok:
            logger.warning("anthropic_count_tokens_auth_failed", reason=reason)
            return JSONResponse(
                status_code=401,
                content={"type": "error", "error": {"type": "authentication_error",
                        "message": f"Invalid API key: {reason}"}},
            )
        try:
            body = await request.json()
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={"type": "error", "error": {"type": "invalid_request_error",
                        "message": f"Invalid JSON: {e}"}},
            )
        messages = body.get("messages")
        if not messages:
            return JSONResponse(
                status_code=400,
                content={"type": "error", "error": {"type": "invalid_request_error",
                        "message": "messages: field required"}},
            )
        try:
            norm_messages, extra = parse_anthropic_request(body)
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={"type": "error", "error": {"type": "invalid_request_error",
                        "message": f"Parse failed: {e}"}},
            )
        tools = extra.get("tools") if isinstance(extra, dict) else None
        tokens = approx_count_tokens(norm_messages, tools)
        return JSONResponse(content={"input_tokens": tokens})


async def _handle_anthropic_stream(
    request: Request,
    model: str,
    route: ModelRoute,
    failover: list[ModelRoute],
    messages: list[dict],
    extra_kwargs: dict,
    identity: Identity,
    injected_text: str,
    ms_identity: float,
    ms_inject: float,
    t_start: float,
    agent_source: AgentSource = AgentSource.FALLBACK,
    original_messages: list[dict] | None = None,
    request_params: RequestParams | None = None,
    decision_meta: DecisionMeta | None = None,
    raw_request_ref: str = "",
    allowed_exposure: str = "public",
) -> StreamingResponse:
    """Anthropic 流式处理：转发 → 按 Anthropic SSE 回传 → 入库。"""
    _ledger_messages = original_messages if original_messages is not None else messages
    result = CaptureResult()
    health, on_used = _call_hooks(request, identity, primary_model=model)

    try:
        # 🔴 MQ-P25：TTFB 的时钟必须在**发起上游调用之前**起。
        # 生成器的 `t0` 在函数体第一行，而 async generator 创建时不执行函数体
        # ⇒ 等它跑起来，`await call_model` 早已返回，整段 TTFB 在窗口之外
        # （量出 0.32ms，而 tap 录的真值是 1,958ms）。
        result.t_upstream = time.perf_counter()
        stream = await call_model(model, messages, stream=True, route=route, failover=failover,
                                  health=health, on_success=on_used, **extra_kwargs)
    except Exception as e:
        logger.error("anthropic_upstream_call_failed", error=router_sdk.error_text(e))
        await _enqueue_turn(request, identity, model, _ledger_messages, injected_text,
                           "", [], TurnStatus.FAILED, str(e), ms_identity, ms_inject, t_start,
                           agent_source, request_params=request_params, decision_meta=decision_meta,
                           raw_request_ref=raw_request_ref)
        return JSONResponse(
            status_code=502,
            content={"type": "error", "error": {"type": "upstream_error",
                    "message": f"Upstream error: {router_sdk.error_text(e)}"}},
        )

    # V-P5b：anthropic 流式拦截（interception 关 = 原路径逐字不变）。
    _agency = request.app.state.agency
    # MQ-P20：把"这一轮输入有多大"算出来交给生成器 —— `message_start.usage.input_tokens`
    # 此前硬编码 0，claude-code 因此永不压缩上下文。
    # 🔴 数的是**真正发给上游的那份 `messages`**（含本轮注入），不是客户端原始请求：
    # 客户端要据此判断"我的上下文还剩多少"，报少了就是把注入的开销藏起来
    # （原则 13 同族：静默丢数比丢数更糟）。
    # `approx_count_tokens` 本身**一行不改**——它已有生产消费者
    # （`POST /v1/messages/count_tokens`，T26 为它而建），这里只是**加第二个消费者**，
    # 那个端点的返回值逐字不变（`test_p20_count_tokens_endpoint_unchanged` 钉住）。
    # tools 只在计数时被 `json.dumps` —— 万一某条路径塞进来非 JSON 对象，
    # 热路径不能因为"一个近似计数"而 500：退到只数 messages（**仍非 0**，
    # 退成 0 就是把缺陷原样复活），并留一条可 grep 的告警。
    try:
        _input_estimate = approx_count_tokens(messages, extra_kwargs.get("tools"))
    except (TypeError, ValueError) as e:
        logger.warning("input_tokens_estimate_tools_skipped", error=str(e))
        _input_estimate = approx_count_tokens(messages)
    _src = anthropic_stream_generator(
        stream, result, model, input_tokens_estimate=_input_estimate,
    )
    if _agency.interception_on:
        # V-P5c：内循环的 call_llm——上游本就是 OpenAI 兼容，用同一路由再调一次；
        # 最终消息交回本协议的 generator 合成（协议转换只发生在出口）。
        async def _loop_llm(msgs: list[dict]) -> dict:
            r2 = await call_model(model, msgs, stream=False, route=route,
                                  failover=failover, health=health,
                                  on_success=on_used, **extra_kwargs)
            return _loop_reply(r2)

        _src = intercept_anthropic_stream(
            _src, agency=_agency, capture_result=result,
            session_prefix=identity.session_prefix(),
            allowed_exposure=allowed_exposure, session_id=identity.session_id,
            agent_id=identity.agent_id,
            project_id=identity.project_id, upstream_messages=messages,
            call_llm=_loop_llm)

    async def generate():
        try:
            async for sse_chunk in _src:
                yield sse_chunk
        except Exception as e:
            logger.error("anthropic_stream_error", error=str(e))
            err = {"type": "error", "error": {"type": "proxy_error", "message": str(e)}}
            yield f"event: error\ndata: {json.dumps(err)}\n\n".encode()
        finally:
            # A2: shield 防客户端断开时 cancel scope 取消入库 -> Turn 丢
            status = TurnStatus.OK if result.done and not result.error else TurnStatus.FAILED
            await _enqueue_turn_shielded(
                request, identity, model, _ledger_messages, injected_text,
                result.full_text, result.tool_events, status, result.error,
                ms_identity, ms_inject, t_start, agent_source,
                response_meta=_build_response_meta_from_capture(result),
                request_params=request_params, decision_meta=decision_meta,
                raw_request_ref=raw_request_ref,
            )

    return StreamingResponse(
        generate(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive",
                 "X-Accel-Buffering": "no"},
    )


async def _handle_anthropic_non_stream(
    request: Request,
    model: str,
    route: ModelRoute,
    failover: list[ModelRoute],
    messages: list[dict],
    extra_kwargs: dict,
    identity: Identity,
    injected_text: str,
    ms_identity: float,
    ms_inject: float,
    t_start: float,
    agent_source: AgentSource = AgentSource.FALLBACK,
    original_messages: list[dict] | None = None,
    request_params: RequestParams | None = None,
    decision_meta: DecisionMeta | None = None,
    raw_request_ref: str = "",
    allowed_exposure: str = "public",
) -> JSONResponse:
    """Anthropic 非流式处理。"""
    _ledger_messages = original_messages if original_messages is not None else messages
    health, on_used = _call_hooks(request, identity, primary_model=model)
    try:
        response = await call_model(model, messages, stream=False, route=route, failover=failover,
                                    health=health, on_success=on_used, **extra_kwargs)
    except Exception as e:
        logger.error("anthropic_upstream_call_failed", error=router_sdk.error_text(e))
        await _enqueue_turn(request, identity, model, _ledger_messages, injected_text,
                           "", [], TurnStatus.FAILED, str(e), ms_identity, ms_inject, t_start,
                           agent_source, request_params=request_params, decision_meta=decision_meta,
                           raw_request_ref=raw_request_ref)
        return JSONResponse(
            status_code=502,
            content={"type": "error", "error": {"type": "upstream_error",
                    "message": f"Upstream error: {router_sdk.error_text(e)}"}},
        )

    # 🔴 入库口径：Hub 存**完整真相**（含被拦下的 bladex 调用），agent 拿处置后的。
    # 故先用未处置的 response 抽 tool_events，再做拦截（chat 侧同序，V-P5a）。
    _truth = format_anthropic_response(response, model)
    full_text = ""
    tool_events = []
    for block in _truth.get("content", []):
        if block.get("type") == "text":
            full_text += block.get("text", "")
        elif block.get("type") == "tool_use":
            from bladex_proxy.models import ToolEvent
            tool_events.append(ToolEvent(tool_name=block.get("name", ""),
                                         arguments=block.get("input"),
                                         direction="call"))

    # V-P5：非流式拦截（interception 关 = 返回原对象，逐字不变）
    _processed = await _intercept_non_stream(
        request, response, model=model, route=route, failover=failover,
        messages=messages, identity=identity, allowed_exposure=allowed_exposure,
        extra_kwargs=extra_kwargs, health=health, on_used=on_used)
    anthropic_response = (_truth if _processed is response
                          else format_anthropic_response(_processed, model))

    status = TurnStatus.OK if anthropic_response.get("type") != "error" else TurnStatus.FAILED
    response_meta = _build_response_meta_from_response(response)
    await _enqueue_turn(request, identity, model, _ledger_messages, injected_text,
                       full_text, tool_events, status, "",
                       ms_identity, ms_inject, t_start, agent_source,
                       response_meta=response_meta, request_params=request_params,
                       decision_meta=decision_meta, raw_request_ref=raw_request_ref)

    return JSONResponse(content=anthropic_response)
