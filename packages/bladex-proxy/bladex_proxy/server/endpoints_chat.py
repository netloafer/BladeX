"""OpenAI chat 协议入站：`/v1/chat/completions`（+ `/v1/embeddings`）与其流式/非流式处理。

09-06 F0.1 自 `server.py` 拆出，零行为（函数体逐字搬家）。门面 `bladex_proxy.server` re-export。
"""

from __future__ import annotations

import time

import structlog
from fastapi import FastAPI, Header, Request, status
from fastapi.responses import JSONResponse, StreamingResponse

from bladex_proxy import router_sdk
from bladex_proxy.capture import CaptureResult, capture_stream
from bladex_proxy.config import ModelRoute, ProxyConfig
from bladex_proxy.agency import intercept_chat_stream
from bladex_proxy.models import (
    AgentSource, ChatCompletionRequest, DecisionMeta, Identity, RequestParams, TurnStatus
)
from bladex_proxy.route import call_model
from bladex_proxy.server.orchestration import (
    _apply_agency_surfaces, _build_kwargs, _build_request_params,
    _build_response_meta_from_capture, _build_response_meta_from_response, _call_hooks,
    _collect_headers, _enqueue_turn, _enqueue_turn_shielded, _extract_bearer,
    _extract_output_modalities, _extract_tool_events_from_response, _loop_reply,
    _message_as_dict, _prepare_round
)

logger = structlog.get_logger()


def register_chat_routes(app: FastAPI) -> None:
    """挂 `/v1/chat/completions`（闭包只捕获 `app`，函数体与拆前逐字相同）。"""
    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(
        request: Request,
        authorization: str | None = Header(None),
        x_agent_id: str | None = Header(None, alias="X-Agent-ID"),
        x_session_id: str | None = Header(None, alias="X-Session-ID"),
        user_agent: str | None = Header(None, alias="User-Agent"),
    ) -> JSONResponse | StreamingResponse:
        cfg: ProxyConfig = request.app.state.config

        client_key = _extract_bearer(authorization)
        ok, reason = cfg.auth_check(client_key, endpoint="/v1/chat/completions")
        if not ok:
            logger.warning("auth_failed", reason=reason)
            return JSONResponse(
                status_code=401,
                content={"error": {"message": f"Invalid API key: {reason}", "type": "authentication_error"}},
            )

        t_start = time.perf_counter()
        body = await request.json()
        req = ChatCompletionRequest(**body)

        # G11.1：完整 header 集合进身份解析（厂商前缀 header 此前被丢在门外）。
        headers = _collect_headers(request, authorization, x_agent_id, x_session_id, user_agent)

        output_modalities = _extract_output_modalities(body)
        # V-A1 T2 编排函数化：身份→敏感度→入站准备→注入→路由 收在 _prepare_round，
        # 工具面/账本块/自我介绍收在 _apply_agency_surfaces（三端点单实现）。
        prep = await _prepare_round(request, req=req, headers=headers, body=body,
                                    output_modalities=output_modalities,
                                    log_route_debug=True)
        if prep.error_response is not None:
            return prep.error_response
        identity, agent_source = prep.identity, prep.agent_source
        req.tools, _toolface_injected = _apply_agency_surfaces(
            request, prep, tools_in=req.tools, stream=bool(req.stream))
        injected_messages = prep.injected_messages

        # T4: 构造请求参数（req.tools 已被增补 ⇒ params 记录实际转发形态）
        request_params = _build_request_params(request, req)
        decision_meta = prep.decision_meta
        model, route, failover = prep.model, prep.route, prep.failover
        injected_text, ms_identity, ms_inject = (
            prep.injected_text, prep.ms_identity, prep.ms_inject)
        allowed_exposure = prep.allowed_exposure

        if req.stream:
            return await _handle_stream(
                request, model, route, failover, injected_messages, req,
                identity, injected_text, ms_identity, ms_inject, t_start,
                agent_source, original_messages=req.messages,
                request_params=request_params, decision_meta=decision_meta,
                allowed_exposure=allowed_exposure,
            )
        else:
            return await _handle_non_stream(
                request, model, route, failover, injected_messages, req,
                identity, injected_text, ms_identity, ms_inject, t_start,
                agent_source, original_messages=req.messages,
                request_params=request_params, decision_meta=decision_meta,
                allowed_exposure=allowed_exposure,
            )


def register_embeddings_routes(app: FastAPI) -> None:
    """挂 `/v1/embeddings`（单独一个登记函数只为保住拆前的路由登记顺序）。"""
    # ── T25: POST /v1/embeddings（与 Memory Index 同源 embedder）──

    @app.post("/v1/embeddings", response_model=None)
    async def openai_embeddings(
        request: Request,
        authorization: str | None = Header(None),
        x_api_key: str | None = Header(None, alias="X-API-Key"),
        x_bladex_caller: str | None = Header(None, alias="X-BladeX-Caller"),
        x_bladex_caller_pid: str | None = Header(None, alias="X-BladeX-Caller-PID"),
        x_bladex_caller_proc: str | None = Header(None, alias="X-BladeX-Caller-Proc"),
        x_bladex_purpose: str | None = Header(None, alias="X-BladeX-Embed-Purpose"),
        user_agent: str | None = Header(None, alias="User-Agent"),
    ) -> JSONResponse:
        """T25: OpenAI 兼容 embeddings 端点。

        复用 app.state 的共享 embedder（与 Memory Index 检索同源），支持单串/批量。

        日志（2026-08-09）：auth_ok 回到 debug，改为每次请求一条 `embed_request`——
        带调用方（`X-BladeX-Caller` / pid / 用途）+ 批量 + 字符数 + 耗时。原先靠
        auth_ok INFO 排查，但它只有 key label，共享模型档下 consolidator 与 agent
        对话请求用同一把 key，满屏同一行看不出谁在调 embedding。
        """
        cfg_local: ProxyConfig = request.app.state.config
        client_key = _extract_bearer(authorization) or x_api_key
        ok, reason = cfg_local.auth_check(client_key, quiet=True, endpoint="/v1/embeddings")
        if not ok:
            return JSONResponse(
                status_code=401,
                content={"error": {"message": f"Invalid API key: {reason}", "type": "authentication_error"}},
            )
        try:
            body = await request.json()
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": f"Invalid JSON: {e}", "type": "invalid_request_error"}},
            )
        inp = body.get("input")
        if inp is None or (isinstance(inp, (str, list)) and len(inp) == 0):
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "input: field required", "type": "invalid_request_error"}},
            )
        inputs = [inp] if isinstance(inp, str) else list(inp)
        embedder = request.app.state.embedder
        if embedder is None:
            return JSONResponse(
                status_code=503,
                content={"error": {"message": "embedder not initialized", "type": "service_unavailable"}},
            )
        # embedding 后端可选化：input_type 是 BladeX 增量扩展（方案 D 共享模型用）——
        # "query"/"passage" 分别走 embed_query/embed_passage（保证与本端检索同前缀）；
        # 缺省 passage = 历史行为（与 Memory Index 同源）。
        input_type = body.get("input_type", "passage")
        from bladex_core.consolidation_proxy import embed_passage_compat, embed_query_compat
        t_embed = time.perf_counter()
        if input_type == "query":
            vectors = embed_query_compat(embedder, inputs)
        else:
            vectors = embed_passage_compat(embedder, inputs)
        embed_ms = round((time.perf_counter() - t_embed) * 1000, 1)
        # 调用方归因：优先显式 header（ProxyEmbedAdapter 恒发），退 User-Agent，
        # 再退 unknown —— "unknown" 本身是信号：有人绕过 adapter 直连这个端点。
        request.app.state.embed_call_log.record(
            caller=x_bladex_caller or (user_agent or "unknown"),
            caller_pid=x_bladex_caller_pid or "-",
            caller_proc=x_bladex_caller_proc or "-",
            purpose=x_bladex_purpose or "-",
            key_label=reason,
            input_type=input_type,
            count=len(inputs),
            chars=sum(len(t) for t in inputs if isinstance(t, str)),
            embed_ms=embed_ms,
        )
        data = [
            {"object": "embedding", "index": i, "embedding": list(vectors[i])}
            for i in range(len(inputs))
        ]
        model_name = getattr(embedder, "model_name", "bladex-embed")
        return JSONResponse(content={
            "object": "list", "data": data, "model": model_name,
            # 方案 D：向量空间身份透传（ProxyEmbedAdapter 用它对齐 Memory Index model_id 不变量）
            "bladex_model_identity": getattr(embedder, "model_identity", f"local:{model_name}"),
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        })


async def _handle_stream(
    request: Request,
    model: str,
    route: ModelRoute,
    failover: list[ModelRoute],
    messages: list[dict],
    req: ChatCompletionRequest,
    identity: Identity,
    injected_text: str,
    ms_identity: float,
    ms_inject: float,
    t_start: float,
    agent_source: AgentSource = AgentSource.FALLBACK,
    original_messages: list[dict] | None = None,
    request_params: RequestParams | None = None,
    decision_meta: DecisionMeta | None = None,
    allowed_exposure: str = "public",
) -> StreamingResponse:
    """流式处理：转发 → 捕获 → 回传 SSE → 入库（try/finally 确保断开也存）。"""
    _ledger_messages = original_messages if original_messages is not None else messages
    import json as _json

    kwargs = _build_kwargs(req)
    result = CaptureResult()
    health, on_used = _call_hooks(request, identity, primary_model=model)

    # ISSUE-5: 上游调用可能抛异常
    try:
        # 🔴 MQ-P25：TTFB 的时钟必须在**发起上游调用之前**起。
        # 生成器的 `t0` 在函数体第一行，而 async generator 创建时不执行函数体
        # ⇒ 等它跑起来，`await call_model` 早已返回，整段 TTFB 在窗口之外
        # （量出 0.32ms，而 tap 录的真值是 1,958ms）。
        result.t_upstream = time.perf_counter()
        stream = await call_model(model, messages, stream=True, route=route, failover=failover,
                                  health=health, on_success=on_used, **kwargs)
    except Exception as e:
        logger.error("upstream_call_failed", error=router_sdk.error_text(e))
        await _enqueue_turn(request, identity, model, _ledger_messages, injected_text,
                           "", [], TurnStatus.FAILED, str(e), ms_identity, ms_inject, t_start,
                           agent_source, request_params=request_params, decision_meta=decision_meta)
        return JSONResponse(
            status_code=502,
            content={"error": {"message": f"Upstream error: {router_sdk.error_text(e)}", "type": "upstream_error"}},
        )

    # V-P5a：流式拦截（interception 关 = 原路径逐字不变）。
    agency = request.app.state.agency
    if agency.interception_on:
        async def _loop_llm(msgs: list[dict]) -> dict:
            r2 = await call_model(model, msgs, stream=False, route=route,
                                  failover=failover, health=health,
                                  on_success=on_used, **kwargs)
            return _loop_reply(r2)

        sse_source = intercept_chat_stream(
            capture_stream(stream, result), agency=agency, capture_result=result,
            upstream_messages=messages,
            session_prefix=identity.session_prefix(),
            allowed_exposure=allowed_exposure, call_llm=_loop_llm,
            session_id=identity.session_id, agent_id=identity.agent_id,
            project_id=identity.project_id)
    else:
        sse_source = capture_stream(stream, result)

    async def generate():
        try:
            async for sse_chunk in sse_source:
                yield sse_chunk.encode()
        except Exception as e:
            logger.error("stream_error", error=str(e))
            err = {"error": {"message": str(e), "type": "proxy_error"}}
            yield f"data: {_json.dumps(err)}\n\n".encode()
        finally:
            # ISSUE-3: 无论正常结束还是客户端断开，都尽量入库
            # A2: shield 防断开时 cancel scope 取消入库 -> Turn 丢
            status = TurnStatus.OK if result.done and not result.error else TurnStatus.FAILED
            error = result.error
            await _enqueue_turn_shielded(
                request, identity, model, _ledger_messages, injected_text,
                result.full_text, result.tool_events, status, error,
                ms_identity, ms_inject, t_start, agent_source,
                response_meta=_build_response_meta_from_capture(result),
                request_params=request_params, decision_meta=decision_meta,
            )

    return StreamingResponse(
        generate(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


async def _handle_non_stream(
    request: Request,
    model: str,
    route: ModelRoute,
    failover: list[ModelRoute],
    messages: list[dict],
    req: ChatCompletionRequest,
    identity: Identity,
    injected_text: str,
    ms_identity: float,
    ms_inject: float,
    t_start: float,
    agent_source: AgentSource = AgentSource.FALLBACK,
    original_messages: list[dict] | None = None,
    request_params: RequestParams | None = None,
    decision_meta: DecisionMeta | None = None,
    allowed_exposure: str = "public",
) -> JSONResponse:
    """非流式处理。ISSUE-5: 上游出错时存 FAILED 轮次。"""
    _ledger_messages = original_messages if original_messages is not None else messages
    kwargs = _build_kwargs(req)
    health, on_used = _call_hooks(request, identity, primary_model=model)

    try:
        response = await call_model(model, messages, stream=False, route=route, failover=failover,
                                    health=health, on_success=on_used, **kwargs)
    except Exception as e:
        logger.error("upstream_call_failed", error=router_sdk.error_text(e))
        await _enqueue_turn(request, identity, model, _ledger_messages, injected_text,
                           "", [], TurnStatus.FAILED, str(e), ms_identity, ms_inject, t_start,
                           agent_source, request_params=request_params, decision_meta=decision_meta)
        return JSONResponse(
            status_code=502,
            content={"error": {"message": f"Upstream error: {router_sdk.error_text(e)}", "type": "upstream_error"}},
        )

    try:
        message = response.choices[0].message
        full_text = message.content or ""
        response_data = response.model_dump() if hasattr(response, "model_dump") else response
        status = TurnStatus.OK
        error = ""

        # T6: 非流式 tool_calls 提取（与流式路径统一，别恒空）
        # 🔴 在拦截**之前**提取：Hub 存完整真相（含 bladex 调用），agent 拿处置后的。
        tool_events = _extract_tool_events_from_response(message)

        # V-P5a：非流式拦截处置（interception 关 = 短路原样）。
        agency = request.app.state.agency
        if agency.interception_on:
            msg_dict = _message_as_dict(message)

            async def _loop_llm(msgs: list[dict]) -> dict:
                r2 = await call_model(model, msgs, stream=False, route=route,
                                      failover=failover, health=health,
                                      on_success=on_used, **kwargs)
                return _loop_reply(r2)

            processed, _transcript, _mode = await agency.process_message(
                msg_dict, upstream_messages=messages,
                session_prefix=identity.session_prefix(),
                allowed_exposure=allowed_exposure, call_llm=_loop_llm,
                session_id=identity.session_id, agent_id=identity.agent_id,
                project_id=identity.project_id)
            if _mode != "none" and isinstance(response_data, dict):
                choices = response_data.get("choices") or [{}]
                choices[0]["message"] = processed
                if _mode == "pure_bladex":
                    choices[0]["finish_reason"] = (
                        "tool_calls" if processed.get("tool_calls") else "stop")
    except (AttributeError, IndexError) as e:
        logger.error("non_stream_parse_failed", error=str(e))
        full_text = ""
        response_data = {"error": str(e)}
        status = TurnStatus.FAILED  # ISSUE-5: 解析失败也标 FAILED
        error = str(e)
        tool_events = []

    response_meta = _build_response_meta_from_response(response)
    await _enqueue_turn(request, identity, model, _ledger_messages, injected_text,
                       full_text, tool_events, status, error, ms_identity, ms_inject, t_start,
                       agent_source, response_meta=response_meta,
                       request_params=request_params, decision_meta=decision_meta)
    return JSONResponse(content=response_data)
