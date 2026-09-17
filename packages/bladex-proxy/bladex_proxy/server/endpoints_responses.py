"""OpenAI Responses 协议入站（ADR-0023，Codex）：`/v1/responses` 与其流式/非流式处理。

09-06 F0.1 自 `server.py` 拆出，零行为（函数体逐字搬家）。门面 `bladex_proxy.server` re-export。
"""

from __future__ import annotations

import json
import time

import structlog
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse

from bladex_proxy import router_sdk
from bladex_proxy.agency import intercept_responses_stream
from bladex_proxy.capture import CaptureResult
from bladex_proxy.config import ModelRoute, ProxyConfig
from bladex_proxy.models import (
    AgentSource,
    ChatCompletionRequest,
    DecisionMeta,
    Identity,
    RequestParams,
    ToolEvent,
    TurnStatus,
)
from bladex_proxy.responses import (
    format_responses_response,
    parse_responses_request,
    responses_stream_generator,
    tool_events_from_output,
)
from bladex_proxy.route import call_model
from bladex_proxy.server.orchestration import (
    _apply_agency_surfaces,
    _build_request_params,
    _build_response_meta_from_capture,
    _build_response_meta_from_response,
    _call_hooks,
    _collect_headers,
    _enqueue_turn,
    _enqueue_turn_shielded,
    _extract_bearer,
    _extract_output_modalities_responses,
    _intercept_non_stream,
    _loop_reply,
    _prepare_round,
    _store_raw_request,
)

logger = structlog.get_logger()


def register_responses_routes(app: FastAPI) -> None:
    """挂 `/v1/responses`（闭包只捕获 `app`，函数体与拆前逐字相同）。"""
    # ── ADR-0023: POST /v1/responses（Codex CLI 原生格式，第三种入站协议）──

    @app.post("/v1/responses", response_model=None)
    async def openai_responses(
        request: Request,
        authorization: str | None = Header(None),
        x_agent_id: str | None = Header(None, alias="X-Agent-ID"),
        x_session_id: str | None = Header(None, alias="X-Session-ID"),
        user_agent: str | None = Header(None, alias="User-Agent"),
    ) -> JSONResponse | StreamingResponse:
        """ADR-0023: OpenAI Responses API 端点（Codex 原生）。

        解析 Responses 请求 -> 归一成 OpenAI chat 格式 -> 认身份 / 注入 / 路由 / 捕获 / 存储
        （复用 /v1/messages 全流程，只换协议转换层）。
        """
        cfg: ProxyConfig = request.app.state.config

        client_key = _extract_bearer(authorization)
        ok, reason = cfg.auth_check(client_key, endpoint="/v1/responses")
        if not ok:
            logger.warning("responses_auth_failed", reason=reason)
            return JSONResponse(
                status_code=401,
                content={"error": {"message": f"Invalid API key: {reason}", "type": "authentication_error"}},
            )

        try:
            body = await request.json()
        except Exception as e:  # noqa: BLE001 - 请求体解析错误要明确返回
            logger.warning("responses_parse_error", error=str(e))
            return JSONResponse(
                status_code=400,
                content={"error": {"message": f"invalid JSON body: {e}", "type": "invalid_request_error"}},
            )

        # T6: 有状态请求防呆。BladeX 是无状态 proxy，每轮从完整 input 认身份 + 注入。
        # Codex 默认 store=false 每轮发完整 input；若客户端带了 previous_response_id，
        # 说明它依赖服务端会话状态，BladeX 无法满足 -> 400 明确告知。
        if body.get("previous_response_id"):
            logger.info(
                "responses_stateful_rejected",
                previous_response_id=body.get("previous_response_id"),
                hint="BladeX is stateless; set store=false and send full input each turn",
            )
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "type": "invalid_request_error",
                        "message": (
                            "BladeX is a stateless proxy and does not support previous_response_id "
                            "(server-side session state). Set store=false and send the full input "
                            "array each turn. See ADR-0023 §2.3."
                        ),
                        "param": "previous_response_id",
                    }
                },
            )

        # T6 观测：store=true 但无 previous_response_id 时放行（BladeX 不做服务端保留，
        # 本轮无状态处理不受影响）；记 info 便于统计 Codex 有状态用法出现频率。
        if body.get("store") is True:
            logger.info(
                "responses_store_requested",
                hint="BladeX is stateless; store=true is a no-op (no server-side retention)",
            )

        t_start = time.perf_counter()

        # 解析 Responses 请求 -> 归一成 OpenAI chat 格式
        messages, extra_kwargs = parse_responses_request(body)
        is_stream = body.get("stream", False)

        # G11.1：完整 header 集合进身份解析（厂商前缀 header 此前被丢在门外）。
        headers = _collect_headers(request, authorization, x_agent_id, x_session_id, user_agent)

        # 用归一后的 messages 构造临时 ChatCompletionRequest 供身份解析
        req = ChatCompletionRequest(
            model=body.get("model", ""),
            messages=messages,
            stream=is_stream,
            **{k: v for k, v in extra_kwargs.items()
               if k in ("temperature", "max_tokens", "tools", "tool_choice", "top_p", "stop")},
        )

        output_modalities = _extract_output_modalities_responses(body)
        # V-A1 T2 编排函数化（与 chat 端点同一实现，见彼处注释）。
        prep = await _prepare_round(request, req=req, headers=headers, body=body,
                                    output_modalities=output_modalities)
        if prep.error_response is not None:
            return prep.error_response
        identity, agent_source = prep.identity, prep.agent_source
        messages = prep.messages

        # 原始请求体按 hash 入 __msg__ 池
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
            return await _handle_responses_stream(
                request, model, route, failover, injected_messages, extra_kwargs,
                identity, injected_text, ms_identity, ms_inject, t_start,
                agent_source, original_messages=messages,
                request_params=request_params, decision_meta=decision_meta,
                raw_request_ref=raw_request_ref, allowed_exposure=allowed_exposure,
            )
        else:
            return await _handle_responses_non_stream(
                request, model, route, failover, injected_messages, extra_kwargs,
                identity, injected_text, ms_identity, ms_inject, t_start,
                agent_source, original_messages=messages,
                request_params=request_params, decision_meta=decision_meta,
                raw_request_ref=raw_request_ref,
                allowed_exposure=allowed_exposure,
            )


async def _handle_responses_stream(
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
    """Responses /v1/responses 流式处理：转发 -> 按 Responses SSE 回传 -> 入库。

    入库 request_messages 恒为 OpenAI chat 格式（与 /v1/messages 一致），
    保证 Memory Index 重建等价性跨协议不变。
    """
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
        logger.error("responses_upstream_call_failed", error=router_sdk.error_text(e))
        await _enqueue_turn(request, identity, model, _ledger_messages, injected_text,
                           "", [], TurnStatus.FAILED, str(e), ms_identity, ms_inject, t_start,
                           agent_source, request_params=request_params, decision_meta=decision_meta,
                           raw_request_ref=raw_request_ref)
        return JSONResponse(
            status_code=502,
            content={"error": {"type": "upstream_error",
                    "message": f"Upstream error: {router_sdk.error_text(e)}"}},
        )

    # V-P5b：responses 流式拦截（interception 关 = 原路径逐字不变）。
    _agency = request.app.state.agency
    _src = responses_stream_generator(stream, result, model)
    if _agency.interception_on:
        # V-P5c：内循环的 call_llm——上游本就是 OpenAI 兼容，用同一路由再调一次；
        # 最终消息交回本协议的 generator 合成（协议转换只发生在出口）。
        async def _loop_llm(msgs: list[dict]) -> dict:
            r2 = await call_model(model, msgs, stream=False, route=route,
                                  failover=failover, health=health,
                                  on_success=on_used, **extra_kwargs)
            return _loop_reply(r2)

        _src = intercept_responses_stream(
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
            logger.error("responses_stream_error", error=str(e))
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


async def _handle_responses_non_stream(
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
    """Responses /v1/responses 非流式处理。"""
    _ledger_messages = original_messages if original_messages is not None else messages
    health, on_used = _call_hooks(request, identity, primary_model=model)
    try:
        response = await call_model(model, messages, stream=False, route=route, failover=failover,
                                    health=health, on_success=on_used, **extra_kwargs)
    except Exception as e:
        logger.error("responses_upstream_call_failed", error=router_sdk.error_text(e))
        await _enqueue_turn(request, identity, model, _ledger_messages, injected_text,
                           "", [], TurnStatus.FAILED, str(e), ms_identity, ms_inject, t_start,
                           agent_source, request_params=request_params, decision_meta=decision_meta,
                           raw_request_ref=raw_request_ref)
        return JSONResponse(
            status_code=502,
            content={"error": {"type": "upstream_error",
                    "message": f"Upstream error: {router_sdk.error_text(e)}"}},
        )

    # 🔴 入库口径：Hub 存**完整真相**（含被拦下的 bladex 调用），agent 拿处置后的。
    # 故先用未处置的 response 抽 tool_events，再做拦截（chat 侧同序，V-P5a）。
    _truth = format_responses_response(response, model)
    # 复用 tool_events_from_output：带 tool_call_id + 解析后的 dict arguments，
    # 与流式路径一致，跨轮 result 匹配不断链。
    full_text = _truth.get("output_text", "")
    tool_events: list[ToolEvent] = tool_events_from_output(_truth.get("output", []))

    # V-P5：非流式拦截（interception 关 = 返回原对象，逐字不变）
    _processed = await _intercept_non_stream(
        request, response, model=model, route=route, failover=failover,
        messages=messages, identity=identity, allowed_exposure=allowed_exposure,
        extra_kwargs=extra_kwargs, health=health, on_used=on_used)
    responses_response = (_truth if _processed is response
                          else format_responses_response(_processed, model))

    status = TurnStatus.OK if responses_response.get("status") != "failed" else TurnStatus.FAILED
    response_meta = _build_response_meta_from_response(response)
    await _enqueue_turn(request, identity, model, _ledger_messages, injected_text,
                       full_text, tool_events, status, "",
                       ms_identity, ms_inject, t_start, agent_source,
                       response_meta=response_meta, request_params=request_params,
                       decision_meta=decision_meta, raw_request_ref=raw_request_ref)

    return JSONResponse(content=responses_response)
