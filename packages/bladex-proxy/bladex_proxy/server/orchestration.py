"""三协议共享的编排层：`_prepare_round` / `_apply_agency_surfaces` / `_enqueue_turn*` / `_build_*` / `_extract_*`。

09-06 F0.1 自 `server.py` 拆出，零行为（函数体逐字搬家）。门面 `bladex_proxy.server` re-export。
🔴 `_apply_agency_surfaces` 是三端点单实现（MQ-A18/P7 族）；`_enqueue_turn` 的调用点分布在三个 endpoints_* 模块。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import time
from typing import Any

import structlog
from fastapi import Request, status
from fastapi.responses import JSONResponse
from bladex_core.fact import Fact
from bladex_core.task_unit import TaskUnit, build_task_units, derive_turn_metadata

from bladex_proxy.assembly import estimate_context_chars
from bladex_proxy.capture import CaptureResult, output_truncated
from bladex_proxy.config import ModelRoute, ProxyConfig
from bladex_proxy.identity import (
    AUX_REASON_ISOLATED, AUX_REASON_NO_TOOLS, AUX_REASON_SOURCE, AUX_REASON_SUBAGENT,
    LEDGER_ONLY_AUX_REASONS, is_cheap_tier_auxiliary, brings_no_tools, is_ledgerless_auxiliary,
    is_isolated_subcall, resolve_identity
)
from bladex_proxy import onboarding as _onb
from bladex_proxy.inject import detect_required_capabilities, do_inject_async
from bladex_proxy.project_identity import extract_cwd
from bladex_proxy import metrics as metrics_mod
from bladex_proxy.metrics import Metrics
from bladex_proxy.modules import module_enabled
from bladex_proxy.models import (
    AgentSource, ChatCompletionRequest, DecisionMeta, Identity, LedgerAnchor,
    ReconstructionRecord, RequestParams, ResponseMeta, ToolEvent, Turn, TurnStatus
)
from bladex_proxy.route import NoCapableCandidateError, call_model, make_success_hook, resolve_route
from bladex_proxy.storage.pipeline_redis import DiskSpill, PipelineRedis
from bladex_proxy.storage.memory_hub import MemoryHub
from bladex_proxy.storage.pipeline_worker import PipelineWorker

logger = structlog.get_logger()


async def _resolve_with_memory(
    request: Request,
    req_model: str,
    cfg: ProxyConfig,
    query: str,
    messages: list[dict],
    facts: list[Fact] | None = None,
    agent_id: str | None = None,
    auxiliary: bool = False,
    output_modalities: list[str] | None = None,
    session_id: str | None = None,
    context_chars: int | None = None,
    allowed_exposure: str = "public",
    principal_id: str | None = None,
    team_ids: list[str] | None = None,
) -> tuple[str, ModelRoute, list[ModelRoute]]:
    """路由决策（Phase 1：配置驱动确定性流水线 + RA 修复卡）。

    agent_id 来自身份解析（ADR-0010），命中 routing.toml [strategies.agent] 时收窄候选池。
    session_id / context_chars：T2 粘性 + T1 规模层（T0 管道）。
    facts 透传给 Phase 2 预留钩子；HardRule 构造延迟到 Phase 2 接入时再做（T6，
    省每请求开销——钩子当前恒返 None，构造了也没人消费）。
    """
    router = request.app.state.router
    if router is None:
        model, route, failover = await resolve_route(req_model, cfg, None, query, agent_id=agent_id, auxiliary=auxiliary)
    else:
        if facts is None:
            facts = []
        requires = _required_capabilities(messages, output_modalities=output_modalities, agent_id=agent_id, cfg=cfg)
        model, route, failover = await resolve_route(
            req_model, cfg, router, query,
            facts=facts, hard_rules=None, requires=requires, agent_id=agent_id,
            auxiliary=auxiliary, session_id=session_id, context_chars=context_chars,
            allowed_exposure=allowed_exposure,
            principal_id=principal_id, team_ids=team_ids,
        )
    # ADR-0021 T7: 路由 source 分布
    _m: Metrics | None = getattr(request.app.state, "metrics", None)
    if _m is not None and route.source:
        _m.inc("bladex_route_source_total", labels={"source": route.source})
    return model, route, failover


def _record_inject_metrics(request: Request, route_facts: list[Fact]) -> None:
    """ADR-0021 T7: 注入过滤命中数 + 已注入 fact 数（exposure 守卫活动信号）。"""
    m: Metrics | None = getattr(request.app.state, "metrics", None)
    if m is None:
        return
    src = getattr(request.app.state, "inject_source", None)
    filtered = getattr(src, "last_exposure_filtered", 0) if src is not None else 0
    if filtered:
        m.inc("bladex_inject_facts_filtered_total", float(filtered))
    m.set_gauge("bladex_inject_facts_injected", float(len(route_facts)))


def _validate_sensitivity_judge(cfg: ProxyConfig) -> None:
    """ADR-0021 section 3.2 启动校验：敏感层开 + 裁判非本地 -> 拒启动(strict)/告警(false)。

    敏感模式下裁判会看到 query，必须钉本地（exposure=local）。裁判非本地时：
    - BLADEX_ROUTE_STRICT=true -> 拒启动（fail-fast，配置错误必须显性）。
    - false -> 告警降级（运行时 route() 跳裁判走允许池 default_tier，敏感内容不外发）。
    """
    rcfg = cfg.routing_config
    sens = rcfg.sensitivity_config()
    if not sens.enabled:
        return
    if not rcfg.strategies.filter.enabled or not rcfg.filter_judge_model:
        return
    judge_name = rcfg.filter_judge_model
    judge_exposure = "public"
    for m in rcfg.models:
        if m.name == judge_name:
            judge_exposure = m.exposure
            break
    if judge_exposure == "local":
        return
    msg = (f"sensitivity enabled but judge '{judge_name}' exposure='{judge_exposure}' "
           f"(must be local); sensitive queries would leak via judge")
    if cfg.route_strict:
        raise RuntimeError(f"BLADEX_ROUTE_STRICT=true: {msg}")
    logger.error("SENSITIVITY_JUDGE_NOT_LOCAL", judge=judge_name,
                 exposure=judge_exposure, hint=msg + " -> runtime will skip judge")


def _resolve_sensitivity(cfg: ProxyConfig, identity: Identity) -> tuple[str, str]:
    """ADR-0021 section 3.3b：敏感度解析（身份的纯静态函数，查表 O(1)）。

    编排换序里在注入前完成（敏感度前移）。返回 (level, allowed_exposure)。
    敏感层关闭 -> (normal, public) = 现状零回归。解析异常 -> 最严（fail-closed）。

    V-A1 示范迁移：模块开关置于最前（`BLADEX_MODULE_SENSITIVITY`，默认开=零回归；
    细粒度仍归 [strategies.sensitivity]）。🔴 注意方向：模块关 = 行为等同敏感层
    未配置（normal/public），企业形态想关它应先想清楚——identity.toml 在而敏感
    模块关的组合由 sensitivity 配置侧的启动校验管，不在这里重写一遍。
    """
    if not module_enabled("sensitivity"):
        return "normal", "public"
    try:
        sens_cfg = cfg.routing_config.sensitivity_config()
        return sens_cfg.resolve(identity.agent_id, identity)
    except Exception as e:
        logger.warning("sensitivity_resolve_failed", error=str(e),
                       hint="fail-closed -> strictest (local)")
        return "sensitive", "local"



def _detect_prefix_changed(
    request: Request,
    identity: Identity,
    messages: list[dict],
) -> bool:
    """检测 agent 是否压缩了上下文（ADR-0016 §3.5 冲突检测）。

    用 Memory Hub 进程内缓存 O(1) 查上一轮的 prefix_hash，与当前 messages 前缀比较。
    Memory Hub 不可用或缓存未命中 -> False（保守不跳过 CAP）。
    """
    hub: MemoryHub | None = request.app.state.hub
    if hub is None:
        return False
    try:
        return hub.check_prefix_changed(identity.session_prefix(), messages)
    except Exception:
        return False


def _capability_error_response(agent_id: str | None, requires: list[str]) -> JSONResponse:
    """§3.4 快返：能力过滤无合格候选 → 结构化错误（不静默降级到可能答非所问的模型）。"""
    who = f"agent {agent_id} 集合" if agent_id else "模型池"
    msg = f"{who}无 {requires} 能力，请补充对应多模态/能力模型"
    logger.warning("route_no_capable_fastfail", agent=agent_id, requires=requires)
    return JSONResponse(
        status_code=422,
        content={"error": {"message": msg, "type": "no_capable_candidate"}},
    )

def _extract_output_modalities(body: dict) -> list[str]:
    """从请求体提取输出模态信号（标准 OpenAI 格式）。

    只检测 modalities 字段（OpenAI 规范中请求音频输出的显式信号）。
    不检测 audio 字段——它只是音频配置（voice/format），部分 agent（如 Hermes
    tts.provider=openai）对所有请求都带 audio 参数但不一定需要音频输出，
    用它做信号会导致所有请求误判为 TTS。

    tool call 式 TTS 信号不统一，不做自动检测（见 plan doc 遗留问题）。
    """
    modalities: set[str] = set()
    raw = body.get("modalities")
    if isinstance(raw, list):
        for m in raw:
            if m == "audio":
                modalities.add("audio")
    return sorted(modalities)


def _required_capabilities(
    messages: list[dict],
    output_modalities: list[str] | None = None,
    agent_id: str | None = None,
    cfg: ProxyConfig | None = None,
) -> list[str]:
    """从消息检测所需能力（多模态硬约束）：复用 inject.detect_required_capabilities。

    T5c（RA8）：agent base 命中 [capability].code_agents → requires += "code"
    （能力护栏：将来池里出现无 code 模型时不会被选中；live 配置全模型有 code = 无行为变化）。
    """
    caps = detect_required_capabilities(messages, output_modalities=output_modalities)
    if agent_id and cfg is not None:
        try:
            code_agents = cfg.routing_config.capability.code_agents
        except Exception:
            code_agents = []
        base = agent_id.split(":", 1)[0]
        if base in code_agents and "code" not in caps:
            caps = sorted([*caps, "code"])
    return caps


def _extract_output_modalities_responses(body: dict) -> list[str]:
    """从 Responses 请求体提取输出模态信号（ADR-0023）。

    Responses API 没有 modalities 字段；输入侧多模态（input_image/input_file）
    不强制要求输出同模态。当前返回空（Codex 纯文本/工具场景），保留接口对称。
    """
    return []


def _estimate_context_chars(messages: list[dict]) -> int:
    """T1（RA1）：请求上下文规模估算（字符数，不引 tokenizer——档位判断够用）。

    str content 计全长；list content 只计 text part，非文本 part（图/音/文件）
    记固定 1000 字符当量。
    """
    total = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text", "")
                    if isinstance(text, str) and text:
                        total += len(text)
                    elif part.get("type"):
                        total += 1000
        # tool_calls 参数也计入（工具循环的主要体积之一）
        for tc in msg.get("tool_calls", []) or []:
            if isinstance(tc, dict):
                func = tc.get("function", {})
                args = func.get("arguments", "") if isinstance(func, dict) else ""
                if isinstance(args, str):
                    total += len(args)
    return total



class RoundPrep:
    """一轮编排的中间产物（V-A1 T2 编排函数化，2026-08-28）。

    三端点（chat/anthropic/responses）此前各持一份编排胶水，MQ-P4/P7/P8 三类
    缺陷全长在胶水的分叉处（"同一件事只在一个端点做对了"）。收成单实现后，
    端点只剩：协议解析 → auth → `_prepare_round` → `_apply_agency_surfaces` →
    request_params → 分发 handler。分叉点显式参数化：`output_modalities`（responses
    的提取器不同）、`log_route_debug`（chat 独有的调试日志，保留原行为）。
    """

    __slots__ = ("identity", "agent_source", "allowed_exposure", "messages",
                 "injected_messages", "injected_text", "ms_identity", "ms_inject",
                 "route_facts", "cap_info", "model", "route", "failover",
                 "decision_meta", "error_response")

    def __init__(self) -> None:
        self.error_response: JSONResponse | None = None
        self.model = ""
        self.route = None
        self.failover: list = []
        self.decision_meta = None


async def _prepare_round(
    request: Request,
    *,
    req: ChatCompletionRequest,
    headers: dict,
    body: dict,
    output_modalities: list[str],
    log_route_debug: bool = False,
) -> RoundPrep:
    """认身份 → 敏感度 → 入站准备 → 注入 → 路由 → decision_meta（三端点共享）。

    行为与拆前逐字等价；逐步注释见各 helper 与历史卡（ADR-0021 §3.3b 换序 /
    V-P5a 入站准备 / MQ-L7 账本块后移 / ADR-0024 T1 task_units）。
    """
    cfg: ProxyConfig = request.app.state.config
    prep = RoundPrep()

    t0 = time.perf_counter()
    identity, agent_source = resolve_identity(headers, req, request.app.state.identity_registry)
    # 项目识别第一档（2026-08-29）：cwd 标记 → git 指纹键；确定性、按 cwd 缓存。
    from bladex_proxy.project_identity import resolve_from_request as _resolve_project
    _proj = _resolve_project(headers, req.messages)
    identity.project_id = _proj.project_id
    identity.project_source = _proj.source
    identity.project_name = _proj.name
    identity.project_root = _proj.root
    prep.ms_identity = (time.perf_counter() - t0) * 1000

    # ADR-0021 §3.3b: 敏感度解析前移到注入之前（身份的纯静态函数）。
    sens_level, allowed_exposure = _resolve_sensitivity(cfg, identity)
    identity.sensitivity = sens_level

    # V-P5a/MQ-P7：入站准备（剥回流+拼接恢复）在 query/prefix/注入之前——
    # 全下游看到同一份形态；三端点同一条路（MQ-P7 之前 anthropic/responses 漏接）。
    agency = request.app.state.agency
    req.messages = agency.prepare_inbound(req.messages, identity.session_prefix())
    messages = req.messages

    query = _extract_query(messages)
    prefix_changed = _detect_prefix_changed(request, identity, messages)
    injected_messages, injected_text, ms_inject, route_facts, cap_info = await do_inject_async(
        messages, query, cfg, request.app.state.inject_source,
        auxiliary=identity.auxiliary, user_id=identity.user_id,
        visibility=identity.visibility, allowed_exposure=allowed_exposure,
        assembler=request.app.state.assembler,
        prefix_changed=prefix_changed,
        session_id=identity.session_id, agent_id=identity.agent_id,
    )
    _record_inject_metrics(request, route_facts)
    # ADR-0024 T1：任务单元挂 request.state，_enqueue_turn 取用落 Memory Hub
    request.state.task_units = cap_info.get("units") or []
    # U9（ADR-0025）：重构决策落 Memory Hub 审计（Memory Index 蒸馏永不读，G3/I5）
    request.state.reconstruction = cap_info.get("reconstruction")

    # T1（RA1）：上下文规模估算（注入后的 final messages = 真实发上游的规模）
    context_chars = _estimate_context_chars(injected_messages)
    if log_route_debug:
        logger.info("route_debug", body_keys=sorted(body.keys()),
                     has_modalities="modalities" in body, has_audio="audio" in body,
                     output_modalities=output_modalities,
                     requires=_required_capabilities(messages, output_modalities=output_modalities, agent_id=identity.agent_id, cfg=cfg),
                     context_chars=context_chars,
                     msg_content_types=[p.get("type") for m in messages[-3:] if isinstance(m.get("content"), list) for p in m["content"] if isinstance(p, dict)])

    prep.identity = identity
    prep.agent_source = agent_source
    prep.allowed_exposure = allowed_exposure
    prep.messages = messages
    prep.injected_messages = injected_messages
    prep.injected_text = injected_text
    prep.ms_inject = ms_inject
    prep.route_facts = route_facts
    prep.cap_info = cap_info

    try:
        prep.model, prep.route, prep.failover = await _resolve_with_memory(request, req.model, cfg, query, messages, facts=route_facts, agent_id=identity.agent_id, auxiliary=is_cheap_tier_auxiliary(identity.aux_source), output_modalities=output_modalities, session_id=identity.session_id, context_chars=context_chars, allowed_exposure=allowed_exposure, principal_id=identity.user_id, team_ids=identity.team_ids)
    except NoCapableCandidateError:
        prep.error_response = _capability_error_response(identity.agent_id, _required_capabilities(messages, output_modalities=output_modalities, agent_id=identity.agent_id, cfg=cfg))
        return prep

    prep.decision_meta = _build_decision_meta(
        prep.route,
        cap_triggered=cap_info.get("triggered", False),
        cap_summary_hash=cap_info.get("summary_hash", ""),
        cap_kept_recent_n=cap_info.get("kept_recent_n", 0),
        inject_position="before_last_user",
        injected_fact_ids=cap_info.get("injected_fact_ids") or [],
        injected_matter_ids=cap_info.get("injected_matter_ids") or [],
        injected_fact_keys=cap_info.get("injected_fact_keys") or {},
    )
    return prep


def _apply_agency_surfaces(
    request: Request,
    prep: RoundPrep,
    *,
    tools_in: list | None,
    stream: bool,
) -> tuple[Any, bool]:
    """工具面增补 + 账本块 + 自我介绍（三端点共享；账本域 aux 判据单一副本）。

    此前 aux 判据表达式在三端点 × augment/ledger 两处 = **六份副本**（MQ-L20/L21
    的每次修订都要改六处）。判据语义：
      - `is_ledgerless_auxiliary`（第三套语义，MQ-L20）：续写/重试/恢复类信封的
        assistant 侧接着做的是同一件正事，该带账本——不读裸 `identity.auxiliary`；
      - `subagent` / `brings_no_tools`（MQ-L21：零工具=要一段文本，Codex 标题生成
        建噪声账本的病例）/ `is_isolated_subcall`（MQ-L8：无 system 单 user）。
    🔴 MQ-L34（2026-09-02）：这四条合成的 `ledgerless_aux` 只挡**账本族**；
    记忆族只被 `aux_source` / `subagent` 挡（`identity.LEDGER_ONLY_AUX_REASONS`
    的补集）。此前整个合成值传给 `auxiliary=` ⇒ 零工具请求连
    `bladex_memory_search` 都没有，与 08-30「记忆族无条件注入」相悖。
    顺序（MQ-L7 C 方案）：augment_tools（tier 已知）→ 账本块（与工具面同步决定
    首步指令，追加真末尾）→ 自我介绍（V-P6，前缀位，门控与工具面同步）。
    返回 (增补后的 tools, toolface_injected)；injected_messages 就地更新进 prep，
    并把「BladeX 加进正文的全部字符」停进 `request.state.bladex_added_chars`
    （MQ-A34 ③ 的分子，见函数尾部）。
    """
    agency = request.app.state.agency
    identity = prep.identity
    own_tools = list(tools_in or [])
    # MQ-L34（2026-09-02）：四个子条件按顺序取**第一个**命中的名字进日志；
    # `ledgerless_aux` 语义不变（任一命中 ⇒ 不给账本面），但只有
    # `LEDGER_ONLY_AUX_REASONS` 之外的子条件才连记忆族一起挡。
    aux_reason = (AUX_REASON_SOURCE if is_ledgerless_auxiliary(identity.aux_source)
                  else AUX_REASON_SUBAGENT if identity.subagent
                  else AUX_REASON_NO_TOOLS if brings_no_tools(own_tools)
                  else AUX_REASON_ISOLATED if is_isolated_subcall(prep.messages)
                  else "")
    ledgerless_aux = bool(aux_reason)
    memory_blocked = ledgerless_aux and aux_reason not in LEDGER_ONLY_AUX_REASONS
    tools_now, tf_injected = agency.augment_tools(
        tools_in, agent_id=identity.agent_id, stream=stream,
        auxiliary=memory_blocked, ledgerless=ledgerless_aux,
        aux_reason=aux_reason, tier=prep.route.tier,
        # 🔴 账本族的第二个放行口需要这两个（2026-08-30）：本会话绑着激活账本
        # 时即使档位在集合外也给账本工具。与下面 insert_ledger_block 传的是
        # 同一组值、同一个判据（`session_owns_active_ledger`）。
        session_id=identity.session_id, project_id=identity.project_id)
    prep.injected_messages = agency.insert_ledger_block(
        prep.injected_messages, identity.agent_id,
        project_id=identity.project_id,
        auxiliary=ledgerless_aux, with_instruction=tf_injected,
        # MQ-L28：块注入与工具面吃同一个档位判定 + 绑定会话判据
        tier=prep.route.tier, session_id=identity.session_id)
    # MQ-L77：说明书的账本段随账本面一起装卸。`last_ledger_face` 由上面那次
    # `augment_tools` 同步写入、中间无 await ⇒ 这里读的就是本请求的值（MQ-A49 的
    # 跨 await 陷阱在此不成立）。`getattr` 兜底与下面 `Turn.ledger_face` 同款：
    # 没走到 augment 那一步的形态里属性不存在，那一轮本就没有账本面。
    prep.injected_messages = agency.insert_system_notes(
        prep.injected_messages, toolface_injected=tf_injected,
        ledger_face=bool(getattr(agency, "last_ledger_face", False)))
    # MQ-A34 ③ 读数口径（2026-09-04）：BladeX 加进正文的**全部**字符 = 注入后正文 −
    # agent 原始正文。`_warn_if_output_truncated` 拿它当 `injected_chars` 的分子。
    # 修前分子是 `prep.injected_text`（inject 阶段的硬规则正文），而 32K 病例里
    # 挤爆输出预算的恰是本函数加的这几块（about / 名册 / 账本块 / 候选段）——
    # 尺子量不到最大的那一段，与 MQ-A33 同族。
    # 🔴 停在 `request.state` 而不是给 `_enqueue_turn` 加第 N 个参数：它有 12 个调用点，
    # 漏传一个就是静默退化（取法同 `task_units` / `reconstruction`）。分子与分母必须
    # 用同一把尺子，故这里与那边都用 `assembly.estimate_context_chars`。
    # **不夹到 0**：`strip_previous_injection` 会剥掉 agent 回传的上一轮注入块，
    # 净值可以为负（这一轮 BladeX 反而让正文变短）——那是真读数，夹掉就是又一次
    # 静默丢信息。缺这个属性（没走过本函数）另有 `-1` 一格，与负净值不混。
    request.state.bladex_added_chars = (
        estimate_context_chars(prep.injected_messages)
        - estimate_context_chars(prep.messages))
    # F-A1 / MQ-A35：把这一轮的账本锚停进 request.state（取法同 `task_units` /
    # `bladex_added_chars`——`_enqueue_turn` 有 12 个调用点，加参数 = 漏传即静默退化）。
    request.state.bladex_ledger_anchor = _build_ledger_anchor(agency, prep)
    # 🔴 F1.3 / MQ-A49：`agency.last_*` 是**进程级"最近一次"**。本函数在同步链里
    # （无 await 插入）读它是安全的；但 `_enqueue_turn` 在**上游调用之后**才跑，
    # 期间另一请求的 `augment_tools` / `insert_ledger_block` 已经把它覆写了。
    # 单用户串行流量不触发，**批 G 剧本③正是两 agent 并行** —— 读数会在最需要它的
    # 场景失真，而且失真得看不出来（两条日志各自都自洽）。
    # ⇒ 这一轮的三个每请求读数在这里就停进 `request.state`（请求作用域），
    # `_enqueue_turn` 只读它。
    #
    # **不塞进 `LedgerAnchor` 落库**：`pending_*` 是每轮几十个词的集合，
    # 入 Hub 就是体积按轮涨，而且入 schema 要承诺重建语义——这把尺子的口径还在改
    # （与 `pending_units` 不进 anchor 是同一条理由）。
    # 🔴 没有 breakdown ⇒ `None` 而不是空集合：**缺数与零是两件事**（这一轮没注块
    # vs 注了块但待办为空）。旧实现在 `_ledger_next_referenced` 里用 `bd is None`
    # 短路，语义逐字搬过来——本卡零行为。
    _bd = getattr(agency, "last_ledger_breakdown", None)
    request.state.bladex_pending_units = (
        frozenset(_bd.pending_units) if _bd is not None else None)
    request.state.bladex_pending_rare = (
        frozenset(_bd.pending_rare) if _bd is not None else frozenset())
    # 同族第二处（MQ-A49 复核确认）：`Turn.ledger_face` 也是 await 之后读进程级字段。
    # 🔴 `None` 有实义（三态：给了 / 没给 / 取不到），所以这里**不夹成 False**——
    # `augment_tools` 每请求恰好覆写一次，取不到只可能是没走过工具面那一步。
    # 也不能用手边的 `tf_injected` 顶替：`last_ledger_face = injected and _ledger_ok`，
    # 08-30 族拆分后 weak 档只拿记忆工具，两者不再等价（字段名宽于语义的陷阱）。
    request.state.bladex_ledger_face = getattr(agency, "last_ledger_face", None)

    # 🔴 接入体检（G17.22 / MQ-A69）：新 agent 第一次真正跑起来时报一行。
    # 挂在这里的三个理由：
    #   ① 这是**三端点单实现**，三条入站路径都会经过（MQ-A18/P7 族的老教训——
    #      挂在端点上就会"只在一条路径上做对"，而本模块要防的恰恰是这类静默）；
    #   ② 身份 / 项目 / 工具面三个信号在这一点**第一次同时可得**
    #      （工具面决策就在上面几行）；
    #   ③ 每 agent 每进程只报一次，`extract_cwd` 这一次重算的代价可忽略
    #      —— 换来的是不必为一个日志字段去加宽 `Identity` 模型。
    # 🔴 要不要体检**只问 `_onb.pending`**（批 H 复核）：去重键归 onboarding 模块
    # 唯一定义，这里不复述 `agent_id not in _seen`——G17.22 改键成 (agent_id, project_id)
    # 时外面这句会静默挡住新行为。
    # 🔴 整段 try/except（批 H 复核）：体检是纯观测，`extract_cwd` 要在几十万字符的
    # 消息上跑正则；观测代码不该有能力让请求失败（热路径预算 200ms 之内的一段
    # 副作用，出错只许记一行）。
    _ident = prep.identity
    _agent_id = str(getattr(_ident, "agent_id", "") or "")
    _aux = bool(getattr(_ident, "auxiliary", False))
    if _onb.pending(agent_id=_agent_id, auxiliary=_aux):
        try:
            _onb.note_onboarding(
                agent_id=_agent_id,
                # 传 enum 本体，短名归一在 onboarding 里做（`str(AgentSource.X)` 是 "AgentSource.X"）
                agent_source=getattr(prep, "agent_source", "") or "",
                auxiliary=_aux,
                # 🔴 判据是 `cwd_found` 不是 `project_id`：抽不到标记 ⇒ 改 pattern；
                # 抽到了不成项目 ⇒ 该 cwd 本就不是项目（家目录）。两种空的修法完全不同，
                # `project_identity.resolve_from_request` 的 docstring 2026-08-29 就写了这条，
                # 而 09-11 我照样把四个 agent 的 0% 全当成了缺陷。
                cwd_found=bool(extract_cwd(prep.messages)),
                project_id=getattr(_ident, "project_id", "") or "",
                project_source=getattr(_ident, "project_source", "") or "",
                toolface_injected=bool(tf_injected),
            )
        except Exception as exc:  # noqa: BLE001 —— 观测失败只记一行，不进热路径
            logger.warning("agent_onboarding_check_failed", agent=_agent_id,
                           error=str(exc), error_type=type(exc).__name__)
    return tools_now, tf_injected


def _added_text(original: list[dict], injected: list[dict]) -> str:
    """注入后正文 − 原正文（多重集差集），按注入顺序拼接。

    与 `bladex_added_chars` **同一把差集**：那边算长度差，这边取内容本身。
    多重集而非集合——同一段文本出现两次时只抵消一次，否则重复内容会被算作
    "没加"（静默少算，仪器骗人的第一类）。
    """
    from collections import Counter
    seen = Counter(_content_text(m) for m in original)
    out: list[str] = []
    for m in injected:
        t = _content_text(m)
        if seen.get(t, 0) > 0:
            seen[t] -= 1
        else:
            out.append(t)
    return "\n".join(out)


def _content_text(msg: dict) -> str:
    """一条消息的文本正文（口径同 `assembly._msg_chars`：str 全长 / list 只取 text part）。"""
    c = msg.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(p.get("text", "") for p in c
                       if isinstance(p, dict) and isinstance(p.get("text"), str))
    return ""


def _build_ledger_anchor(agency: Any, prep: RoundPrep) -> LedgerAnchor | None:
    """组 `Turn.ledger_anchor`（F-A1）。这一轮没注账本块 ⇒ `None`（缺数报缺数）。

    🔴 **仪器不许改行为**（批 F 高危 1）：整个函数只读 —— 读 agency 上每请求
    覆写一次的三个读数（`last_ledger_breakdown` / `last_about_chars` /
    `last_roster_chars`）与 prep 的两份正文，不碰任何注入结果。
    """
    bd = getattr(agency, "last_ledger_breakdown", None)
    if bd is None:
        return None
    added = _added_text(prep.messages, prep.injected_messages)
    return LedgerAnchor(
        ledger_id=bd.ledger_id, rev=bd.rev, sections=dict(bd.chars),
        about_chars=int(getattr(agency, "last_about_chars", 0) or 0),
        roster_chars=int(getattr(agency, "last_roster_chars", 0) or 0),
        added_hash=hashlib.sha256(added.encode("utf-8")).hexdigest()[:16])


#: F-A2：第一个动作是工具调用时，`arguments` 取多少字符参与词面比对。
#: 长参数（整份文件正文、大段 patch）会把交集撑成噪声——量的是"引用了 Next 里
#: 那句话"，不是"这一轮打了多少字"。
_FIRST_ACTION_ARG_CHARS = 2000


def _ledger_next_referenced(request: Any, anchor: LedgerAnchor | None,
                            response_text: str, tool_events: list) -> None:
    """"读了"代理读数（F-A2 / MQ-A35）：**只打日志，不入 Turn**。

    ## 量的是什么

    模型这一轮收到了账本块（`anchor` 非 None），那么它**这一轮的第一个动作**
    ——首条 assistant 正文 + 第一个 agent 工具调用的参数——有没有出现 Next/Open
    条目里的词面单元。零 LLM、纯词面（`candidate_units`，与账本相关性位同一把
    分词与同一张停表，读数因此可与 `agency_ledger_candidates` 并排看）。

    ## 为什么是"代理"而不是"读了"

    词面命中 ≠ 模型真的按账本行事；零命中也可能是它读了却换了说法。它能承受的
    结论只有一个：**这个比值随注入面改动而变化**。所以判据写成"占比作为基线记进
    MQ-A35"，不定目标值（ADR-0029 §1 的口径纪律：先建尺子再谈数值）。

    ## 不入 schema

    读数成熟前不进 `Turn`——字段一旦入库就要承诺重建语义与向后兼容，而这把尺子
    的口径（停表、取多少字符、算不算工具名）几乎肯定还要改。日志改口径不欠债，
    schema 改口径欠债（原则 12 的反面用法）。

    ## 🔴 为什么收 `request` 而不是 `agency`（F1.3 / MQ-A49）

    本函数在**上游调用之后**跑。`agency.last_ledger_breakdown` 是进程级
    "最近一次"，期间另一请求的 `insert_ledger_block` 已经把它覆写了 ⇒ 并发下
    这一轮的命中会拿**别的请求**的单元集去算，而且两条日志各自都自洽、看不出来。
    单元集改由 `_apply_agency_surfaces`（同步链，无 await 插入）停进
    `request.state`，本函数只读请求作用域。**签名里没有 `agency` 是刻意的**：
    拿不到它，就没人能在这里再读一次进程级字段。
    """
    if anchor is None or not anchor.ledger_id:
        return
    # 🔴 MQ-A43：单元集是**注入那一刻**算好的（= 模型看到的那一版），不是从池现算。
    # 池此刻可能已被本轮的 `bladex_ledger_update` 改过，拿它求交就是"尺子量自己的手"。
    # 🔴 MQ-A49：而且取自 `request.state`（本请求的），不是 agency 上的"最近一次"。
    state = getattr(request, "state", None)
    units = getattr(state, "bladex_pending_units", None)
    if units is None:
        return          # 这一轮没走过 `_apply_agency_surfaces` ⇒ 无对象可比，不造读数
    try:
        from bladex_core.ledger_runtime import candidate_units, path_like_units
        from bladex_proxy.toolface import is_bladex_tool
        units = set(units)
        # `rev_now` 是**此刻**这本账本的 rev（"这一轮被改过没有"），所以它就该读活池。
        # 这不是 A49 那个形态：池按 `anchor.ledger_id` 取，不是进程级"最近一次"。
        _pool = getattr(getattr(getattr(request, "app", None), "state", None),
                        "agency", None)
        led = (getattr(_pool, "pool", None) or {}).get(anchor.ledger_id)
        first_call = next((te for te in (tool_events or [])
                           if getattr(te, "direction", "") == "call"), None)
        action = "text"
        parts = [response_text or ""]
        if first_call is not None:
            action = getattr(first_call, "tool_name", "") or "?"
            args = getattr(first_call, "arguments", None)
            # lossless-ok：截断只喂词面比对，**不入库**（本函数只打日志）——
            # 原则 13 管的是"采集到的原始信息入 Memory Hub 必须无损"。
            parts.append((args if isinstance(args, str) else json.dumps(
                args, ensure_ascii=False, default=str))[:_FIRST_ACTION_ARG_CHARS])
        first_action_text = " ".join(parts)
        seen = candidate_units(first_action_text)
        hits = units & seen
        # 🔴 MQ-A48：**内容自指**分桶。首动作若是 `bladex_ledger_update`，它的参数
        # （`match` / `text`）就是 Next/Open 的原文——词面交集必然满格，量的是
        # "模型把账本原文抄回来了"，不是"模型按账本行事"。live 09-07：`usable=True`
        # 17 行里 1 行 `hits=66/66`（`ldg-c5103b17c024` rev 25→26），正是这一形态。
        # 与 MQ-A43 的**时间自指**（单元集取自模型看到的那一版）是两件事。
        # 不删这一档、只单列：`ledger_edit` 桶本身是有意义的读数（模型确实在记账），
        # 只是它不能进"读了没有"的分子。判据取 `toolface.is_bladex_tool`——
        # 同一件事一个实现点，不在这里另写一遍前缀判断。
        bucket = "ledger_edit" if is_bladex_tool(action) else "work"
        # MQ-A47：稀有锚口径并列跑，**旧读数不作废**（口径变更要能对照，
        # 否则修前修后的基线不可比——ADR-0029 §4 的口径纪律）。
        # MQ-A49：与 `units` 同一条——请求作用域，不读 agency。
        rare = set(getattr(state, "bladex_pending_rare", frozenset()) or frozenset())
        # 🔴 MQ-A52 ①（G0.2，2026-09-09）：路径类剔除**双侧对称**。
        # `pending_rare` 已经在 `rare_pending_units` 里按 `path_like_units` 剔过账本侧，
        # 而首动作侧原样 ⇒ "账本里是散文的 `hermes`" 撞上 "首动作里是路径 token 的
        # `hermes`（`~/Documents/Hermes/x.md`）"照样计命中。live 09-08：`hits_rare>0`
        # 17 行里 **11 行的 `rare_sample` 就是 `hermes` 一个词**。
        # 同一条判据只用在一半上 —— 与 MQ-L53 第二类报错犯的是同一个错。
        # **只作用在 rare 轴**：`hits` / `sample` 是 A47 之前的旧口径，并列跑不作废
        # （口径变更要能对照，ADR-0029 §4）；路径类剔除本就是 rare 口径的一半（A51）。
        seen_rare = seen - path_like_units(first_action_text)
        hits_rare = rare & seen_rare
        logger.info("agency_ledger_next_referenced",
                    ledger=anchor.ledger_id, rev=anchor.rev,
                    # 🔴 `rev_now != rev` = 这一轮账本被改过（模型自己 update 了）。
                    # 不合并成一个 bool：分桶要的是"改没改"与"命中没命中"两轴独立。
                    rev_now=getattr(led, "rev", -1),
                    hits=len(hits), next_units=len(units),
                    # 🔴 MQ-A47：`hits_rare` 才是能当基线的那一列（`hits` 51% 是
                    # `documents,hermes` 这类路径底噪）。`rare_units` 是它的分母——
                    # 为 0 说明这本账本的待办里没有任何"只有这件事才有"的词，
                    # 那一行同样不该进命中率（与 `usable` 同一条零分母纪律）。
                    hits_rare=len(hits_rare), rare_units=len(rare),
                    rare_sample=",".join(sorted(hits_rare)[:5]),
                    # 🔴 MQ-A43 第二半：`next_units=0` 时 `hits=0` **不是"没引用"，
                    # 是"没东西可引用"** —— 零分母。判据放在**生产侧**（同 F-B2 的
                    # `self_conflict` 做法），消费侧按它过滤即可，不必各自重算口径。
                    # live 首验 16/17 是这一档：假分母比缺分母更骗人。
                    usable=len(units) > 0,
                    first_action=action, bucket=bucket,
                    sample=",".join(sorted(hits)[:5]))
    except Exception as e:  # noqa: BLE001 —— 仪器绝不阻断入库（高危 1）
        logger.warning("agency_ledger_next_referenced_failed", error=str(e))


def _call_hooks(request: Request, identity: Identity, primary_model: str = ""):
    """T2/T4：给 call_model 的 health + on_success（failover 实际切换后回写粘性）。

    `primary_model` = 路由决策选出的首选。实际用的模型与它不同 = 这轮发生了
    failover，粘性要标 `origin=FAILOVER`（G14/MQ-RT1）——被动降级和主动选择
    必须可区分，否则一次瞬时抖动会让会话永久漂移在降级模型上。
    """
    health = getattr(request.app.state, "model_health", None)
    router = request.app.state.router
    # 粘性/温度回写逻辑在 route.make_success_hook（V-A2 下沉，行为逐字保持）。
    return health, make_success_hook(
        router, session_id=identity.session_id,
        agent_id=identity.agent_id, primary_model=primary_model)


def _extract_bearer(authorization: str | None) -> str | None:
    if not authorization:
        return None
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return authorization.strip()


def _collect_headers(
    request: Request,
    authorization: str | None,
    x_agent_id: str | None,
    x_session_id: str | None,
    user_agent: str | None,
) -> dict[str, str]:
    """G11.1：把**完整** header 集合交给身份解析，而不是四个写死的字段。

    改动前三个端点各自手搓一个四键字典（`authorization`/`x-agent-id`/
    `x-session-id`/`user-agent`），厂商前缀 header 根本进不了门——dsh 每轮发的
    `x-deepseek-harness-{user-id,session-id,compact}` 全被丢在门外，而那正是
    agent 主动告诉我们的身份、会话与 auxiliary 标记（台账 MQ-A1）。

    向后兼容：四个原有键仍显式覆盖，缺失时保持 `""` 而非键不存在——下游
    `_extract_api_key` 等一律走 `.get(name, "")`，语义逐字不变。

    :returns: 键全小写的 header 字典。
    """
    merged = {name.lower(): value for name, value in request.headers.items()}
    merged.update({
        "authorization": authorization or "",
        "x-agent-id": x_agent_id or "",
        "x-session-id": x_session_id or "",
        "user-agent": user_agent or "",
    })
    return merged


def _is_anthropic_client(request: Request) -> bool:
    """T26: 判断请求是否来自 Anthropic 客户端（Claude Code）。

    判据：anthropic-version header 存在，或 user-agent 含 claude。
    """
    if request.headers.get("anthropic-version"):
        return True
    ua = request.headers.get("user-agent", "").lower()
    return "claude" in ua


def _build_model_obj(display_name: str, is_anthropic: bool) -> dict[str, Any]:
    """T26: 构造单个 model 对象，按客户端类型分流响应形态。

    - OpenAI 形态: {"id", "object":"model", "created", "owned_by"}
    - Anthropic 形态: {"id", "type":"model", "display_name", "created_at"}
    """
    if is_anthropic:
        return {
            "id": display_name,
            "type": "model",
            "display_name": display_name,
            "created_at": 0,
        }
    return {
        "id": display_name,
        "object": "model",
        "created": 0,
        "owned_by": "bladex",
    }


_REDIS_RECOVER_COOLDOWN_S = 60.0
_last_recover_attempt: float = 0.0
_redis_skip_count: int = 0


async def _try_recover_redis(request: Request) -> PipelineRedis | None:
    """懒恢复：enqueue 时发现 pipeline is None，尝试连/启动 Redis（事件触发，不轮询）。

    流程：
      1. 尝试连接 Redis（可能已被手动启动）
      2. 连不上 -> 尝试启动 redis-server（subprocess）
      3. 再连一次
      4. 连上 -> 初始化 Pipeline + pipeline_worker + 回灌溢出文件

    冷却 60 秒防频繁重试（Redis 二进制不存在 / 端口被占等情况）。
    所有失败路径都有 warning/error 日志，便于排查。
    """
    global _last_recover_attempt, _redis_skip_count
    now = time.monotonic()
    if now - _last_recover_attempt < _REDIS_RECOVER_COOLDOWN_S:
        _redis_skip_count += 1
        if _redis_skip_count % 50 == 1:
            logger.warning(
                "pipeline_redis_still_unavailable",
                skip_count=_redis_skip_count,
                cooldown_remaining=round(_REDIS_RECOVER_COOLDOWN_S - (now - _last_recover_attempt), 1),
                hint="Redis unavailable, turns are NOT being stored; "
                     "will retry recovery after cooldown",
            )
        return None
    _last_recover_attempt = now

    cfg: ProxyConfig = request.app.state.config
    pipeline = PipelineRedis(
        redis_url=cfg.redis_url, stream=cfg.redis_stream,
        group=cfg.redis_group, consumer=cfg.redis_consumer,
        queue_max_len=cfg.queue_max_len, overflow_dir=cfg.overflow_dir,
    )

    # 1. 尝试直连
    try:
        await pipeline.connect()
    except Exception as connect_err:
        logger.info("pipeline_redis_connect_failed_trying_start", error=str(connect_err))
        # 2. 连不上 -> 尝试启动 Redis（仅本地实例）
        from urllib.parse import urlparse
        _parsed = urlparse(cfg.redis_url)
        _redis_host = _parsed.hostname or "127.0.0.1"
        _redis_port = _parsed.port or 6379
        if _redis_host not in ("127.0.0.1", "localhost", "::1"):
            logger.warning(
                "pipeline_redis_remote_skip_start",
                redis_host=_redis_host,
                hint="Redis URL points to a remote host; cannot auto-start. "
                     "Ensure the remote Redis is reachable.",
            )
            return None
        try:
            subprocess.Popen(
                ["redis-server", "--daemonize", "yes", "--port", str(_redis_port),
                 "--appendonly", "yes", "--appendfsync", "everysec",
                 "--tcp-keepalive", "0",
                 "--dir", "data/"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            await asyncio.sleep(1)
            await pipeline.connect()
        except FileNotFoundError:
            logger.error(
                "pipeline_redis_binary_not_found",
                hint="redis-server not in PATH; install Redis or set BLADEX_REDIS_URL "
                     "to an external Redis instance. Turns are NOT being stored.",
            )
            return None
        except Exception as start_err:
            logger.warning(
                "pipeline_redis_recover_failed",
                connect_error=str(connect_err),
                start_error=str(start_err),
                hint="Redis could not be started; turns are NOT being stored",
            )
            return None

    # 3. 连上了 -> 初始化 Pipeline + pipeline_worker
    request.app.state.pipeline = pipeline
    hub: MemoryHub = request.app.state.hub
    pipeline_worker = PipelineWorker(pipeline, hub)
    pipeline_worker.start()
    request.app.state.pipeline_worker = pipeline_worker

    # 回灌磁盘溢出文件
    replayed = await pipeline.replay_overflow()
    if replayed > 0:
        logger.info("pipeline_overflow_replayed_on_recover", count=replayed)

    _redis_skip_count = 0
    logger.info(
        "pipeline_redis_recovered",
        hint="lazy recovery succeeded on enqueue; storage pipeline resumed",
        replayed_overflow=replayed,
    )
    return pipeline


async def _enqueue_inner_loop_turns(
    request: Request, pipeline: "PipelineRedis", identity: Identity,
    loop_turns: list[Turn],
) -> None:
    """把主轮带出的内循环 aux 轮逐条入队（MQ-P9）。单条失败落盘，不影响其余。

    日志 `inner_loop_turn_enqueued` 是 MQ-W 对账的第二项：
    `route_calling ≈ turn_enqueued + Σ rounds`。
    """
    if not loop_turns:
        return
    ok = 0
    for lt in loop_turns:
        try:
            await pipeline.enqueue(lt)
            ok += 1
        except Exception as e:  # noqa: BLE001 —— 单条失败不阻断其余，落盘兜底
            logger.error("inner_loop_enqueue_failed", error=str(e),
                         roundtrip=lt.roundtrip)
            try:
                request.app.state.disk_spill.spill(lt)
            except Exception as se:  # noqa: BLE001
                logger.error("inner_loop_spill_failed", error=str(se))
    # 不按名读 Turn 的总耗时字段：H1 对账 gate 拿它当"写了没按名读"的样例字段
    # （`test_field_write_only_is_info_not_dead`），这里读一下会把样例翻成 ok。
    # 耗时读数走日志侧 `_loop_cost`（agency）与 Hub 侧 aux 轮本身。
    logger.info("inner_loop_turn_enqueued", key=identity.session_prefix(),
                rounds=ok, requested=len(loop_turns),
                tokens=sum(int((lt.response_meta.usage or {}).get("total", 0) or 0)
                           for lt in loop_turns))


async def _enqueue_turn(
    request: Request,
    identity: Identity,
    model: str,
    messages: list[dict],
    injected_text: str,
    response_text: str,
    tool_events: list,
    status: TurnStatus,
    error: str,
    ms_identity: float,
    ms_inject: float,
    t_start: float,
    agent_source: AgentSource = AgentSource.FALLBACK,
    # T4(ADR-0018 §4.4): 三组元数据（可选，调用方按可用性传）
    response_meta: ResponseMeta | None = None,
    request_params: RequestParams | None = None,
    decision_meta: DecisionMeta | None = None,
    raw_request_ref: str = "",
) -> None:
    """构造 Turn 并异步入库（不阻塞响应）。"""
    ms_total = (time.perf_counter() - t_start) * 1000

    # MQ-A34 ③：上游因输出预算耗尽而截断 ⇒ 一条可 grep 告警（放这里 = 三协议 × 流式/非流式
    # 唯一汇合点，且 agent / model / 注入正文 / 原始 prompt 都在手）。只记不改。
    _warn_if_output_truncated(request, identity, model, messages,
                              response_meta, request_params)

    # ADR-0021 T7: metrics（请求计数 + 延迟分位，by agent/sensitivity）
    _metrics: Metrics | None = getattr(request.app.state, "metrics", None)
    if _metrics is not None:
        _lbl = {"agent": identity.agent_id, "sensitivity": getattr(identity, "sensitivity", "normal")}
        _metrics.inc("bladex_requests_total", labels=_lbl)
        _metrics.observe("bladex_request_latency_ms", ms_total, labels=_lbl)

    # ADR-0024 T1/T2：任务单元一等实体化。热路径已算好的单元直接用，
    # logical_turn / roundtrip 从单元派生（旧 _infer_turn_metadata 已退役）。
    # 走 request.state 传递而非加 6 个 handler 形参：单元是**每请求**产物，
    # request.state 正是它的语义位置，且避免 3 个端点 × 流式/非流式的签名蔓延。
    # 双层 getattr：_enqueue_turn 在"绝不丢 Turn"的路径上，不得因属性缺失抛异常
    # （Starlette 真实 Request 恒有 .state；测试替身 / 未来的调用方不保证）。
    units = _resolve_task_units(
        messages, getattr(getattr(request, "state", None), "task_units", None)
    )
    logical_turn, roundtrip = derive_turn_metadata(units)

    # U9（ADR-0025 §5）：重构决策（A/B 层 degrade_plan + B 指纹 + C 澄清）落 Turn。
    # 双层 getattr 同 task_units——绝不因属性缺失丢 Turn。
    _recon_raw = getattr(getattr(request, "state", None), "reconstruction", None)
    _recon = None
    if isinstance(_recon_raw, dict):
        try:
            _recon = ReconstructionRecord(**_recon_raw)
        except Exception as _re:  # noqa: BLE001
            logger.warning("reconstruction_record_invalid", error=str(_re))

    # V-P4 硬点 1：本轮被剥离的 bladex 调用与结果随 Turn 入 Hub。
    # 从 agency 直接取而不是加参数：`_enqueue_turn` 有七八个调用点，
    # 加一个必传参数等于让每个调用点都记得传——漏一个就是静默不落库。
    _splice_recs: list = []
    _ag = getattr(request.app.state, "agency", None)
    if _ag is not None and getattr(_ag, "interception_on", False):
        try:
            _splice_recs = _ag.splice.new_in_turn(identity.session_prefix())
        except Exception as _se:  # noqa: BLE001 —— 落库辅助信息，不阻断入库
            logger.warning("splice_records_collect_failed", error=str(_se))

    # MQ-L26：这一轮给没给账本面（走 request.state，不加第 13 个参数）。
    # 🔴 `None` 有实义 —— 见 `Turn.ledger_face` 的三态语义：缺数就报缺数，
    # 不能因为"取不到"就写 False，那会让全量重建时历史锚定一次性失效。
    # 🔴 F1.3 / MQ-A49 **同族第二处**：此前读 `_ag.last_ledger_face`，与
    # `_ledger_next_referenced` 一样是"await 之后读进程级最近一次"——并发下
    # 会把另一请求的账本面决定写进本轮 Turn，而这个字段**是入库的**
    # （比只打日志的那处更贵：错的 `ledger_face` 会让全量重建的历史锚定选错轮）。
    # 改由 `_apply_agency_surfaces` 在同步链里停进 `request.state`。
    _ledger_face = getattr(getattr(request, "state", None),
                           "bladex_ledger_face", None)

    # F-A1 / MQ-A35：账本锚（取法同 `task_units` —— 走 request.state，不加参数）。
    # 双层 getattr 同上：`_enqueue_turn` 在"绝不丢 Turn"的路径上。
    _anchor = getattr(getattr(request, "state", None), "bladex_ledger_anchor", None)
    if not isinstance(_anchor, LedgerAnchor):
        _anchor = None

    # F-A2：注意力半边的第一把尺——模型收到账本块后，这一轮的第一个动作有没有
    # 引用 Next/Open 里的内容。零 LLM、纯词面、只打日志（见函数 docstring）。
    # 🔴 F1.3 / MQ-A49：收 `request`（请求作用域）不收 `agency`（进程级最近一次）。
    _ledger_next_referenced(request, _anchor, response_text, tool_events)

    turn = Turn(
        identity=identity, model=model, request_messages=messages,
        task_units=units,
        splice_records=_splice_recs,
        ledger_face=_ledger_face,
        ledger_anchor=_anchor,
        reconstruction=_recon,
        injected_memory=injected_text, response_text=response_text,
        tool_events=tool_events, status=status, error=error,
        ms_identity=ms_identity, ms_inject=ms_inject, ms_total=ms_total,
        agent_source=agent_source,
        logical_turn=logical_turn, roundtrip=roundtrip,
        auxiliary=identity.auxiliary,
        aux_source=identity.aux_source,
        session_id_source=identity.session_id_source,
        sensitivity=identity.sensitivity,
        response_meta=response_meta or ResponseMeta(),
        request_params=request_params or RequestParams(),
        decision_meta=decision_meta or DecisionMeta(),
        raw_request_ref=raw_request_ref,
    )

    # MQ-P9（ADR-0032 §3.2 硬点 6）：本会话待入账的内循环轮次，随主轮一起入库。
    # 取法同 `splice_records`——从 agency 取，不加第 N 个参数。主轮先入、aux 轮后入，
    # 每轮各自一条 Turn（entry_id 唯一，不撞键）。
    _loop_turns: list[Turn] = []
    if _ag is not None and getattr(_ag, "interception_on", False):
        try:
            from bladex_proxy.loopledger import build_inner_loop_turn
            for _pr in _ag.loop_ledger.drain(identity.session_prefix()):
                _loop_turns.append(build_inner_loop_turn(
                    identity, model, _pr, trigger_ts=turn.ts.isoformat()))
        except Exception as _le:  # noqa: BLE001 —— 入账附件，不阻断主轮入库
            logger.warning("inner_loop_turns_build_failed", error=str(_le))
            _loop_turns = []

    pipeline: PipelineRedis | None = request.app.state.pipeline
    if pipeline is None:
        pipeline = await _try_recover_redis(request)
    if pipeline is not None:
        try:
            await pipeline.enqueue(turn)
            logger.info("turn_enqueued", key=identity.session_prefix(),
                        status=status.value, ms_total=round(ms_total, 2))
            await _enqueue_inner_loop_turns(request, pipeline, identity, _loop_turns)
            return
        except Exception as e:
            logger.error("enqueue_failed", error=str(e))
            # 落到下面的 DiskSpill 兜底

    # T1: 最终 fallback 无条件落盘（消灭 ADR-0017 已知限制）
    # 三种情况 Turn 必须落文件：pipeline is None / 60s 冷却期未恢复 / enqueue 抛异常。
    # 恢复后由 replay_overflow() 回灌 Pipeline。
    disk_spill: DiskSpill = request.app.state.disk_spill
    disk_spill.spill(turn)
    for _lt in _loop_turns:
        disk_spill.spill(_lt)
    # ADR-0027 §3.3：Redis 缺席不再丢数据（T1 已修），但仍是降级——
    # 落盘的 turn 在回灌前进不了 Memory Index，用户侧表现为"这段时间的事没记住"。
    metrics_mod.record_degradation(metrics_mod.ENQUEUE_SKIPPED)
    logger.warning(
        "turn_spilled_to_disk",
        key=identity.session_prefix(),
        status=status.value,
        ms_total=round(ms_total, 2),
        hint="turn stored to disk overflow; will replay when Redis recovers",
    )


def _warn_if_output_truncated(
    request: Request,
    identity: Identity,
    model: str,
    messages: list[dict],
    response_meta: ResponseMeta | None,
    request_params: RequestParams | None,
) -> None:
    """MQ-A34 ③（2026-09-03 拍板：0.1.0 只做告警）：上游 `finish_reason` 落在
    `capture.OUTPUT_TRUNCATED_REASONS` ⇒ 打 `upstream_output_truncated`，带注入占比读数。

    病例：本机 32K 窗口模型上注入面把输出预算挤到 30–164 token，账本面 0/28——
    "模型不调 switch"与"模型调不出 switch"在没有这条告警前分不开
    （`task-hermes-accept-ledger-gap-20260903.md` §5/§6）。① 窗口字段 / ② 按预算裁剪
    归 0.2.0；本函数**零行为差异**：只读已捕获的元数据，不动请求也不动回复。

    `inject_ratio = injected_chars / (prompt_chars + injected_chars)`——`messages` 是 agent 原始
    消息（`_ledger_messages`，注入前），注入正文另计，分母才是模型真正看到的那份。

    🔴 **分子口径 2026-09-04 修正**（MQ-A34 补记）：修前分子 = `len(prep.injected_text)`，
    只算 inject 阶段的硬规则正文；about / 名册 / 账本块 / 候选段在
    `_apply_agency_surfaces` 才加进 `injected_messages`，全在分子之外——而 32K 病例里
    挤爆输出预算的**恰是这几块**（与 MQ-A33 同族：尺子量不到最大的那一段）。
    现分子 = `_apply_agency_surfaces` 停在 `request.state.bladex_added_chars` 的
    「BladeX 加进正文的全部字符」。**修前的读数按硬规则口径读，不与修后同列比较。**

    三个读数格互不合并：`injected_chars=-1` = 接线断（本轮没走过
    `_apply_agency_surfaces`）；`0` = 真的一个字没加；**负值** = 剥掉的回传注入
    多于本轮新注（`strip_previous_injection`），是真读数不是错。
    """
    fr = (response_meta.finish_reason if response_meta is not None else "") or ""
    if not output_truncated(fr):
        return
    added = getattr(getattr(request, "state", None), "bladex_added_chars", None)
    injected_chars = -1 if added is None else int(added)
    try:
        prompt_chars = estimate_context_chars(messages)
    except Exception:  # noqa: BLE001 —— 读数辅助，不因消息形态怪异阻断入库
        prompt_chars = -1
    seen = prompt_chars + injected_chars if added is not None and prompt_chars >= 0 else 0
    usage = response_meta.usage if response_meta is not None else {}
    logger.warning(
        "upstream_output_truncated",
        agent=identity.agent_id, model=model, finish_reason=fr,
        injected_chars=injected_chars, prompt_chars=prompt_chars,
        inject_ratio=(round(injected_chars / seen, 3) if seen > 0 else None),
        completion_tokens=usage.get("completion"), prompt_tokens=usage.get("prompt"),
        max_tokens=(request_params.max_tokens if request_params is not None else None),
        session=identity.session_prefix(),
    )


async def _enqueue_turn_shielded(
    request: Request,
    identity: Identity,
    model: str,
    messages: list[dict],
    injected_text: str,
    response_text: str,
    tool_events: list,
    status: TurnStatus,
    error: str,
    ms_identity: float,
    ms_inject: float,
    t_start: float,
    agent_source: AgentSource = AgentSource.FALLBACK,
    response_meta: ResponseMeta | None = None,
    request_params: RequestParams | None = None,
    decision_meta: DecisionMeta | None = None,
    raw_request_ref: str = "",
) -> None:
    """A2：shield 入库 await，防客户端断开时 cancel scope 取消入库 -> Turn 丢。

    Starlette/uvicorn 在客户端断开时取消响应任务，generator finally 里的
    `await _enqueue_turn(...)` 会立即再抛 CancelledError -> 断开那轮（往往长回复、
    信息量不低）最容易丢。本函数把入库派发为独立 Task（脱离请求 cancel scope），
    持引用防 GC（app.state.bg_tasks，done 回调清理），再 shield await：
      - 正常路径：等价于直接 await（测试无回归）；
      - 断开路径：shield 抛 CancelledError 交还响应任务，内部入库 Task 继续跑完
        （落 Redis 或 T1 的 DiskSpill 兜底），Turn 不丢。
    """
    task = asyncio.create_task(_enqueue_turn(
        request, identity, model, messages, injected_text,
        response_text, tool_events, status, error,
        ms_identity, ms_inject, t_start, agent_source,
        response_meta=response_meta, request_params=request_params,
        decision_meta=decision_meta, raw_request_ref=raw_request_ref,
    ))
    bg = request.app.state.bg_tasks
    bg.add(task)
    task.add_done_callback(bg.discard)
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        logger.info(
            "enqueue_shielded_on_disconnect",
            key=identity.session_prefix(), status=status.value,
            hint="client disconnected; enqueue continues in background",
        )
        raise


def _build_request_params(request: Request, req: ChatCompletionRequest, *,
                          forwarded_tools: list | None = None) -> RequestParams:
    """T4(ADR-0018 §4.4 C3): 从请求构造 RequestParams + tools schema 入 __msg__ 池。

    requested_model = agent 请求的原始 model（Turn.model 存路由后实际用的）。
    tools schema 全文经 Memory Hub __msg__ 池 hash 去重存储，Turn 只存 tools_hash。
    Memory Hub 不可用时 tools_hash 留空（不阻塞热路径）。

    🔴 `forwarded_tools` = **实际转发上游的那份**（工具面增补之后）。
    2026-08-26 取证发现：`/v1/chat/completions` 上 `augment_tools` 排在本函数
    **之前**（注释明写"Hub 的 tools_hash 记录的是实际转发形态"），而
    `/v1/messages` 与 `/v1/responses` 上排在**之后**，且增补写进的是
    `extra_kwargs["tools"]`——与本函数读的 `req.tools` 是**两个对象**，
    光调换顺序都救不回来。后果：那两个协议的 tools_hash 记的是**增补前**形态，
    "模型实际看到哪些工具"在 Claude Code / Codex 上**查不出来**
    （差点据此得出"CC 从没拿到过工具面"的错误结论）。
    顺带统一了记录格式：`extra_kwargs["tools"]` 已由 `parse_*` 归一为 OpenAI 形态。
    """
    rp = RequestParams(
        requested_model=req.model or "",
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        tool_choice=req.tool_choice,
    )
    tools = req.tools if forwarded_tools is None else forwarded_tools
    if tools:
        hub: MemoryHub | None = request.app.state.hub
        if hub is not None:
            try:
                rp.tools_hash = hub.store_tools_schema(tools)
            except Exception as e:
                logger.warning("tools_schema_store_failed", error=str(e))
    return rp


def _build_response_meta_from_capture(result: CaptureResult) -> ResponseMeta:
    """T4(C1/C2/C6): 从 CaptureResult 构造 ResponseMeta。"""
    return ResponseMeta(
        reasoning_text=result.reasoning_text,
        usage=dict(result.usage),
        finish_reason=result.finish_reason,
        ms_first_chunk=result.ms_first_chunk,
    )


def _message_as_dict(message: Any) -> dict:
    """上游 message 对象 → 拦截协议要的 plain dict（**唯一实现点**，三条非流式路径共用）。

    真上游（Router/LiteLLM）给的是 pydantic Message，走 `model_dump()`；
    Mapping 原样拷贝；其余对象（测试 mock 的 `SimpleNamespace`、SDK 的轻量对象）
    按 `__dict__` 递归转——此前三处各写一份 `model_dump() if hasattr else dict(m)`，
    2026-09-02 四模块翻默认后 `/v1/responses` 非流式测试用 `SimpleNamespace` 首次
    走到拦截路径 ⇒ `dict(SimpleNamespace)` TypeError。同一件事三份字面量，收成一处。
    """
    def _plain(o: Any) -> Any:
        if hasattr(o, "model_dump"):
            return o.model_dump()
        if isinstance(o, dict):
            return {k: _plain(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_plain(x) for x in o]
        if hasattr(o, "__dict__") and not isinstance(o, type):
            return {k: _plain(v) for k, v in vars(o).items() if not k.startswith("_")}
        return o
    out = _plain(message)
    return out if isinstance(out, dict) else {"content": out}


def _loop_reply(response: Any) -> dict:
    """内循环 `call_llm` 的返回形态：assistant message dict + 归一 `usage`。

    🔴 MQ-P9（2026-09-02）：此前五处 `_loop_llm` 都只返回 `message.model_dump()`，
    而 usage 挂在 response 上不在 message 上 ⇒ `LoopRound.usage` 恒空、
    `_loop_cost(...).tokens` 恒 0、入 Hub 的 aux 轮拿不到 token 支出。
    五处收成一个函数——同一件事五份字面量正是"改一处漏四处"的形状。
    `innerloop` 取走 usage 后会把它从外发消息里剥掉（wire 形态里没有这个字段）。
    """
    m2 = response.choices[0].message
    out = _message_as_dict(m2)
    try:
        usage = getattr(response, "usage", None)
        if usage is not None:
            out["usage"] = _extract_usage_dict(usage)
    except Exception:  # noqa: BLE001 —— usage 只是入账附件，取不到不阻断内循环
        pass
    return out


def _build_response_meta_from_response(response: Any) -> ResponseMeta:
    """T4(C1/C2/C6): 从非流式上游响应构造 ResponseMeta。

    提取 reasoning_content（C1）、usage（C2）、finish_reason（C6）。
    任一缺失留默认值（非流式无首字延迟，ms_first_chunk=0）。
    """
    meta = ResponseMeta()
    try:
        choice = response.choices[0]
        message = choice.message
        rc = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
        if rc:
            # ADR-0020 T4: 某些模型/Router 解析下 reasoning_content 可能是 list -> 强制 str
            meta.reasoning_text = "".join(str(x) for x in rc) if isinstance(rc, list) else str(rc)
        fr = getattr(choice, "finish_reason", None)
        if fr:
            meta.finish_reason = ",".join(str(x) for x in fr) if isinstance(fr, list) else str(fr)
    except (AttributeError, IndexError, TypeError) as e:
        logger.debug("response_meta_parse_failed", error=str(e))
    try:
        usage = getattr(response, "usage", None)
        if usage is not None:
            meta.usage = _extract_usage_dict(usage)
    except Exception:
        pass
    return meta


def _extract_usage_dict(usage: Any) -> dict:
    """从上游 usage 对象提取 token 计数（非流式复用）。"""
    try:
        if hasattr(usage, "model_dump"):
            data = usage.model_dump()
        elif isinstance(usage, dict):
            data = usage
        else:
            data = dict(usage)
        return {
            "prompt": int(data.get("prompt_tokens", 0)),
            "completion": int(data.get("completion_tokens", 0)),
            "total": int(data.get("total_tokens", 0)),
        }
    except Exception:
        return {}


def _inject_top_k() -> int:
    """注入条目数（默认 `flags.MAX_PREFETCH_K`=15，`BLADEX_INJECT_TOPK` 可调）。
    S1（2026-09-03）后主动检索路径已退役，本值只透传给 InjectionSource.top_k 与 config show。"""
    from bladex_core.flags import MAX_PREFETCH_K
    try:
        v = int(os.environ.get("BLADEX_INJECT_TOPK", "") or MAX_PREFETCH_K)
    except ValueError:
        return MAX_PREFETCH_K
    return max(1, v)


def _build_decision_meta(route: ModelRoute, *, cap_triggered: bool = False,
                         cap_summary_hash: str = "", cap_kept_recent_n: int = 0,
                         inject_position: str = "",
                         injected_fact_ids: list[str] | None = None,
                         injected_matter_ids: list[str] | None = None,
                         injected_fact_keys: dict[str, str] | None = None) -> DecisionMeta:
    """T4(C4): 从路由决策 + CAP 信息构造 DecisionMeta。

    ADR-0028 E2.3：追加 injected_fact_ids —— 本轮注入命中的 Memory Index 条目 id 随 turn 落 Memory Hub，
    consolidator 消费时聚合成 ref_count（proxy 侧 Memory Index 只读，写权不变）。
    """
    return DecisionMeta(
        route={"source": route.source, "tier": route.tier, "reason": route.reason},
        cap={
            "triggered": cap_triggered,
            "summary_hash": cap_summary_hash,
            "kept_recent_n": cap_kept_recent_n,
        },
        inject_position=inject_position,
        injected_fact_ids=list(injected_fact_ids or []),
        injected_fact_keys=dict(injected_fact_keys or {}),
        # M3-1：Matter 卡命中留痕（三时钟的相关钟此前对 Matter 完全没有信号源）
        injected_matter_ids=list(injected_matter_ids or []),
    )


def _store_raw_request(request: Request, body: dict) -> str:
    """T4(C5): /v1/messages 原始请求体按 hash 入 __msg__ 池，返回 raw_request_ref。

    parse_anthropic_request 归一化有损（cache_control 丢弃等），存原文供回溯定位。
    Memory Hub 不可用时返回空（不阻塞热路径）。
    """
    if not body:
        return ""
    hub: MemoryHub | None = request.app.state.hub
    if hub is None:
        return ""
    try:
        return hub.store_raw_request(body)
    except Exception as e:
        logger.warning("raw_request_store_failed", error=str(e))
        return ""


def _resolve_task_units(
    messages: list[dict],
    units: list[TaskUnit] | None,
) -> list[TaskUnit]:
    """拿到本轮任务单元：优先用热路径已算好的，缺失才现算（ADR-0024 T1）。

    ~~`_infer_turn_metadata` 已退役~~——它是 `build_task_units` 的劣化版重复实现
    （ADR-0024 D1「任务结构算三遍、留零遍」的第二遍）。现在只有一套切分算法。

    `units` 由 `do_inject*` 经 `cap_info["units"]` 传入（索引基 = 未剥离的 messages
    = Memory Hub 存的 original_messages）。为 None 只发生在不经注入路径的调用（少数错误
    分支、旧测试），此时就地补算，口径完全一致。
    """
    if units:
        return units
    return build_task_units(messages)


def _extract_tool_events_from_response(message: Any) -> list:
    """T6: 从非流式响应的 message 中提取 tool_calls → ToolEvent 列表。

    与流式路径（capture.py 聚合 tool_calls）统一，避免提炼管线两边都取造成重复。
    """
    from bladex_proxy.models import ToolEvent

    events: list[ToolEvent] = []
    tool_calls = getattr(message, "tool_calls", None)
    if not tool_calls:
        return events

    for tc in tool_calls:
        try:
            func = getattr(tc, "function", None)
            tool_name = getattr(func, "name", "") if func else ""
            args_str = getattr(func, "arguments", "") if func else ""
            try:
                import json as _json
                args = _json.loads(args_str) if args_str else None
            except (json.JSONDecodeError, ValueError):
                args = args_str
            events.append(ToolEvent(
                tool_name=tool_name, arguments=args, direction="call",
            ))
        except Exception as e:
            logger.warning("non_stream_tool_event_extract_failed", error=str(e))

    return events


def _extract_query(messages: list[dict]) -> str:
    """末条 user 消息文本（进理解层之前的原始 query）。

    实现只有一份，在 `bladex_core.query_understanding.last_user_text`——
    离线仪器要复现生产的检索 query，三份近似实现里任何一份漂移都会让读数
    悄悄换参照系。
    """
    from bladex_core.query_understanding import last_user_text

    return last_user_text(messages)


def _build_kwargs(req: ChatCompletionRequest) -> dict:
    return req.to_router_kwargs()


def _shim_processed_response(response: Any, processed: dict,
                             finish_reason: str | None = None) -> Any:
    """把**处置后**的 OpenAI 形态消息套回 `format_*_response` 认识的形状。

    两个 formatter 只用 `getattr` 读 `choices[0].message.{content,tool_calls}`、
    `choices[0].finish_reason` 与 `response.usage`——所以给一个同形状的轻量替身
    就够，**不去就地改 litellm 的响应对象**（那是别人的可变状态，改了会连带
    影响 `_build_response_meta_from_response` 与入库口径）。

    `processed` 里的 tool_calls 是 dict，formatter 按属性读，故逐层转 namespace。
    """
    from types import SimpleNamespace

    def _tc(d: dict) -> Any:
        fn = d.get("function") or {}
        return SimpleNamespace(
            id=d.get("id", ""), type=d.get("type", "function"),
            function=SimpleNamespace(name=fn.get("name", ""),
                                     arguments=fn.get("arguments", "")))

    msg = SimpleNamespace(
        content=processed.get("content"),
        tool_calls=[_tc(t) for t in processed.get("tool_calls") or []] or None,
        role=processed.get("role", "assistant"))
    orig_fr = ""
    try:
        orig_fr = response.choices[0].finish_reason or ""
    except (AttributeError, IndexError):
        pass
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg,
                                 finish_reason=finish_reason or orig_fr or "stop")],
        usage=getattr(response, "usage", None))


async def _intercept_non_stream(request: Request, response: Any, *, model: str,
                                route: Any, failover: Any, messages: list[dict],
                                identity: Identity, allowed_exposure: str,
                                extra_kwargs: dict, health: Any, on_used: Any) -> Any:
    """非流式拦截处置，三协议共用。返回给 formatter 用的响应（或原对象）。

    🔴 V-P5（2026-08-27 勘察）：`/v1/messages` 与 `/v1/responses` 的非流式路径
    此前**完全没有拦截接线**——而 `augment_tools` 在流式/非流式分支**之前**执行，
    所以这两条路径**照发 `bladex_*` 工具面却不拦截**：模型一调，调用直接透传给
    agent，而 agent 没有这个工具的处理器。

    chat 协议早有这段（`_handle_non_stream` 里的 V-P5a）——**同一件事只在一个
    端点上做对了**，与 MQ-A18 / MQ-L23 同型，本仓今天第三次。
    """
    agency = request.app.state.agency
    if not agency.interception_on:
        return response
    try:
        message = response.choices[0].message
    except (AttributeError, IndexError):
        return response
    msg_dict = _message_as_dict(message)

    async def _loop_llm(msgs: list[dict]) -> dict:
        r2 = await call_model(model, msgs, stream=False, route=route,
                              failover=failover, health=health,
                              on_success=on_used, **extra_kwargs)
        return _loop_reply(r2)

    processed, _transcript, mode = await agency.process_message(
        msg_dict, upstream_messages=messages,
        session_prefix=identity.session_prefix(),
        allowed_exposure=allowed_exposure, call_llm=_loop_llm,
        session_id=identity.session_id, agent_id=identity.agent_id,
        project_id=identity.project_id)
    if mode == "none":
        return response                     # 零改动通道：逐字原样
    fr = None
    if mode == "pure_bladex":
        # 调用全被拦下 ⇒ 终止语义必须跟着改，否则 agent 收到
        # 「说要调工具却一个都没有」并重试（chat 侧 2026-08-25 实测过 4 轮）。
        fr = "tool_calls" if processed.get("tool_calls") else "stop"
    logger.info("agency_nonstream_intercepted", mode=mode,
                agent=identity.agent_id,
                calls_left=len(processed.get("tool_calls") or []))
    return _shim_processed_response(response, processed, fr)
