"""叠加注入 — 标记包裹、删旧注新（T6，ADR-0008 §6.5）。

原则：
  - 只在 agent 发来的 messages 上 **追加**，绝不删改正文。
  - 注入内容用 <bladex-memory>…</bladex-memory> 包裹。
  - 每轮注入前先删掉上一轮自己打标记的那段，再注入最新的（防累积）。

T3（M-proxy-1.5）：默认注入位置改为**最后一条 user 消息之前**（独立 system 消息），
保住前面长而稳定的历史前缀的 prompt cache 命中。
"合并进首 system"降级为兼容开关（config inject_merge_system=True）。

2026-09-03 S1（瘦身批）：**主动注入检索路径整体退役**——`ProxyPrefetcher` / 三平面
`build_planes` / 本模块内的画像卡 / hop expand / soft scoring 消费点 / 注入命中回写
（`_capture_injected_*`）连同 `BLADEX_INJECT_INDEX_RECALL` / `INJECT_PLANES` **两个开关**一起删除。
`HOP_EXPAND` / `SOFT_SCORING` / `PROFILE_CARDS` 三个开关**保留**：消费点不在注入路径——前两个门
`MemoryIndex.search`（memory_search 工具面走它），后者门写侧画像捕获（USER.md/AGENT.md
投影上游）；删它们 = 改工具面检索行为，不在拍板范围（HANDOFF-20260903 §2 偏差 1）。
判据：08-30 拍板后 121 轮 `inject_plane_counters=0`、
注入路径 `index_search_done=0`；记忆改由模型经工具面（memory_search 工具）按需取
（`agency.py` 直走 `MemoryIndex.search`，不经本模块）。注入面只剩 **硬规则**；
`inject_done` 形态逐字不变（`planes=False`）。回滚 = git 历史（拍板 a：不留 A/B 开关）。
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any

import structlog
from bladex_core.fact import Fact
from bladex_core.task_unit import build_task_units

from bladex_proxy import metrics as metrics_mod
from bladex_proxy.config import ProxyConfig
from bladex_proxy.modules import module_enabled
from bladex_proxy.pruning import apply_frozen_plan, prune_cold_context

if True:  # runtime import
    # SUMMARY_OPEN/CLOSE 的唯一定义在 assembly.py（S3 2026-09-03 收掉本文件那份重复
    # 定义——两个家迟早分叉；上一轮的上下文摘要块剥离仍靠它们，ADR-0016 review fix）。
    from bladex_proxy.assembly import (
        SUMMARY_CLOSE,
        SUMMARY_OPEN,
        ContextAssembler,
        assembly_enabled_for,
        estimate_context_chars,
    )

logger = structlog.get_logger()

MEMORY_OPEN = "<bladex-memory>"
MEMORY_CLOSE = "</bladex-memory>"

_SUMMARY_BLOCK_RE = re.compile(
    rf"\s*{re.escape(SUMMARY_OPEN)}.*?{re.escape(SUMMARY_CLOSE)}\s*",
    re.DOTALL,
)

_MEMORY_BLOCK_RE = re.compile(
    rf"\s*{re.escape(MEMORY_OPEN)}.*?{re.escape(MEMORY_CLOSE)}\s*",
    re.DOTALL,
)


# 内容块类型 → 所需能力（ADR-0013 §3.1 ③ 多模态过滤）。请求含某类块 → requires 对应能力。
# 候选模型的 capabilities 必须是 requires 的超集才能选（能力硬约束）。
_CONTENT_TYPE_CAPABILITY: dict[str, str] = {
    "image_url": "vision",
    "image": "vision",
    "video_url": "vision",
    "video": "vision",
    "input_audio": "audio",
    "file": "file",
}


def detect_required_capabilities(
    messages: list[dict],
    output_modalities: list[str] | None = None,
) -> list[str]:
    """从 messages 的 content 块检测所需能力 + 输出模态信号（TTS 等）。

    输入侧：含 image/video 块 → ["vision"]；含 input_audio → ["audio"]；含 file → ["file"]。
    输出侧：output_modalities 含 "audio"（TTS 请求的 modalities: ["text","audio"]）→ 加 "audio"。
    多模态混合则并集。纯文本 → []（不约束）。
    """
    required: set[str] = set()
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict):
                cap = _CONTENT_TYPE_CAPABILITY.get(part.get("type", ""))
                if cap:
                    required.add(cap)
    # 输出模态（TTS 等）：modalities: ["text", "audio"] → 需要 audio 能力的模型
    if output_modalities:
        required.update(output_modalities)
    return sorted(required)


# ── ADR-0028 E4：Prompt Cache TTL 感知 ──────────────────────────────────────


def _should_understand(inject_on: bool, auxiliary: bool, agent_id: str) -> bool:
    """理解层要不要跑——同步 / 异步两个入口的**唯一判据**（F0.4，2026-09-06）。

    S1 后 `_understand` 的唯一消费者是冷启相关性裁剪 `_prune_if_cold`，而它只在
    `assembly_enabled_for(agent_id)`（`BLADEX_ASSEMBLY_AGENTS`，默认只 claude-code）之后才跑
    ⇒ 非 CC 每轮白算一次理解层 + 一条 `query_understood` 日志。判据加上装配门：
    非 CC 轮 `query_understood` 0 条，CC 轮不变。
    """
    return inject_on and not auxiliary and assembly_enabled_for(agent_id)


def _understand(messages: list[dict], query: str, source: Any,  # noqa: ANN401
                session_id: str) -> str:
    """E6.1 Query 理解层：剥信封 + 短指代扩写 + 截断。

    S1 后它的唯一消费者是冷启相关性裁剪（`_prune_if_cold` 用它 embed）——
    词法通道 / Project 卡那两个消费者已随主动注入检索路径删除。

    两种坏 query 的真实形态：
      ① **信封当 query**——claude-code 末条 user 消息 p50 = 70,261 字符，
         其中 98.21% 是 agent 内部协议信封。拿它去 embed，向量指的是信封的语义。
      ② **短指代**——「这个怎么办」「继续」，8 个字符不到，召回等同随机。

    返回用于检索的 query（失败时原样返回，绝不打死注入主路径）。
    """
    try:
        # 单元切分 + open 单元取文 + 理解层，整条委托给 `retrieval_query`——
        # 离线仪器（eval_consumption 的相关性轴）必须复现**同一条路**，
        # 各写一份就是"筛子和被测系统各算各的"（MQ-S4 形态）。
        from bladex_core.query_understanding import retrieval_query

        q, identifiers = retrieval_query(messages, query)
        if q != query:
            logger.info("query_understood", raw_len=len(query), used_len=len(q),
                        identifiers=len(identifiers))
        return q
    except Exception as e:  # noqa: BLE001 —— 理解层失败退回原 query
        logger.warning("query_understanding_failed", error=str(e))
        return query


def _cache_registry():  # noqa: ANN202
    from bladex_proxy.cache_state import registry
    return registry()


def _ttl_enabled() -> bool:
    """TTL 感知与冷启裁剪的开关族（flags 收编，默认开，`=0` 回退全量转发）。"""
    from bladex_core.flags import flag_enabled
    return flag_enabled("BLADEX_CACHE_TTL_AWARE")


def _model_cache_ttl_s(model: str) -> int:
    """查该模型的 Prompt Cache TTL（routing.toml `[[models]].cache_ttl_s`）。

    读不到配置（未配路由 / 测试环境）→ 用默认 300s，与 ModelCandidate 默认一致。
    """
    from bladex_core.routing import ModelCandidate

    default = ModelCandidate.model_fields["cache_ttl_s"].default
    if not model:
        return default
    try:
        from bladex_proxy.routing_config import load_routing_config
        cfg = load_routing_config()
        for m in getattr(cfg, "models", []):
            if m.name == model:
                return int(getattr(m, "cache_ttl_s", default))
    except Exception:  # noqa: BLE001 —— 配置读不到不该影响注入主路径
        pass
    return default


def _cache_warm(session_id: str, prefix_changed: bool) -> bool:
    """本会话上游 prompt cache 是否温热（ADR-0028 §2）。

    判定用的模型 = 本会话**上一次实际转发**的模型。在会话粘性生效的前提下
    它就是本轮将要用的模型；万一路由真换了模型，`note_forward` 会在转发后
    丢弃冻结计划，下一轮自然按冷处理。
    """
    if not _ttl_enabled() or not session_id:
        return False
    st = _cache_registry().get(session_id)
    if st is None or not st.model:
        return False
    return _cache_registry().is_warm(
        session_id, st.model, _model_cache_ttl_s(st.model),
        prefix_changed=prefix_changed,
    )


def _prune_if_cold(
    messages: list[dict],
    source: Any,  # noqa: ANN401
    query: str,
    q_vec: list[float] | None,
    session_id: str,
) -> tuple[list[dict], dict | None]:
    """冷启相关性裁剪（E4.3）。任何前置条件不满足 → 原样返回。"""
    if not _ttl_enabled() or not messages:
        return messages, None
    index = getattr(source, "_index_for_hits", None)
    if index is None:
        return messages, None
    embed_passage = getattr(index, "embed_passage", None)
    if embed_passage is None:
        return messages, None
    if q_vec is None:
        try:
            q_vec = index.embed_query_vec(query)
        except Exception as e:  # noqa: BLE001
            logger.warning("prune_query_embed_failed", error=str(e))
            return messages, None
    state = _cache_registry().ensure(session_id) if session_id else None
    try:
        return prune_cold_context(
            messages, q_vec=q_vec, embed_passage=embed_passage, state=state,
        )
    except Exception as e:  # noqa: BLE001 —— 裁剪失败绝不打死请求
        logger.warning("prune_failed", error=str(e))
        return messages, None


class InjectionSource:
    """注入源：**只注硬规则**（2026-09-03 S1 起）。

    `index` 仍收——冷启相关性裁剪（`_prune_if_cold`，ADR-0028 E4.3）要用它 embed。
    `last_facts` / `last_facts_wide` / `last_exposure_filtered` 是 server 侧的既有
    读点（路由 facts / L2 摘要复用 / T7 指标），S1 后恒为空——与 08-30 拍板后
    `_index_recall_on()=False` 时的 live 形态逐字相同。
    """

    def __init__(
        self,
        hard_rules: list[str] | None = None,
        index: Any | None = None,  # noqa: ANN401  # MemoryIndex，可选（只供冷启裁剪 embed）
        top_k: int = 10,
        min_relevance: float | None = None,
        retrieval_top_k: int | None = None,
        max_chars: int | None = None,
    ) -> None:
        self._hard_rules = hard_rules or _DEFAULT_HARD_RULES
        self._top_k = top_k
        self._retrieval_top_k = retrieval_top_k or (top_k * 2)
        self._index_for_hits = index
        logger.info("injection_source_hard_rules_only",
                    reason="index_recall_retired" if index is not None else "no_index")

    @property
    def last_facts(self) -> list[Fact]:
        """最近一次召回的相关 Fact。S1 后恒空（主动检索路径已退役）。"""
        return []

    @property
    def last_facts_wide(self) -> list[Fact]:
        """最近一次宽检索的全部 Fact。S1 后恒空。"""
        return []

    @property
    def last_exposure_filtered(self) -> int:
        """最近一次被 exposure 守卫过滤掉的 Fact 数。S1 后恒 0。"""
        return 0

    def build_facts(
        self,
        query: str,
        config: ProxyConfig,
        user_id: str | None = None,
        visibility: list[str] | None = None,
        allowed_exposure: str = "public",
    ) -> list[str]:
        """返回注入文本行列表：硬规则（`[MUST/NEVER] …`）。"""
        rules = config.effective_hard_rules or self._hard_rules
        return [f"[MUST/NEVER] {rule}" for rule in rules]


_DEFAULT_HARD_RULES = [
    "NEVER use emojis in responses.",
    "MUST respond in the same language as the user's message.",
]


def strip_previous_injection(messages: list[dict]) -> list[dict]:
    """删掉上一轮注入的 <bladex-memory> 块（绝不碰 agent 正文）。

    T5: 只从 system 角色消息中剥离（我们只往 system 注，就只从 system 删）。
    非 system 消息中出现标记 → 告警但不删（防误伤正文）。
    支持 str content 和 list content（多模态）。
    """
    cleaned: list[dict] = []
    for msg in messages:
        msg = dict(msg)
        role = msg.get("role", "")
        content = msg.get("content")

        if role == "system":
            # system 消息：剥离注入块 + CAP 摘要块（ADR-0016 review fix）
            if isinstance(content, str) and (MEMORY_OPEN in content or SUMMARY_OPEN in content):
                content = _MEMORY_BLOCK_RE.sub("\n", content)
                content = _SUMMARY_BLOCK_RE.sub("\n", content).strip()
                if not content:
                    continue
                msg["content"] = content
            elif isinstance(content, list) and _list_content_has_marker(content):
                msg["content"] = _strip_from_list_content(content)
        else:
            # 非 system 消息：本函数按设计不删（绝不碰 agent 正文）。
            # 🔴 2026-08-25 口径统一：**回流剥离归 `splice.strip_inbound_echoes`**
            # （它按标记精确剥 user/tool/assistant 里我们自己的块，agent 正文零接触）。
            # 本函数只在**拦截模块关闭**时才是唯一防线，故降为 debug 级——
            # 开着 interception 时这条 warning 是纯噪声（live 实测每轮 4–5 条），
            # 两条路径各管一段但口径一致：谁都不碰 agent 自己写的字。
            if ((isinstance(content, str) and MEMORY_OPEN in content)
                    or (isinstance(content, list) and _list_content_has_marker(content))):
                logger.debug("marker_in_non_system_message", role=role,
                             hint="echo handled by splice.strip_inbound_echoes "
                                  "when interception is on")

        cleaned.append(msg)
    return cleaned


def _list_content_has_marker(content: list) -> bool:
    """检查多模态 list content 是否含注入标记。"""
    for part in content:
        if isinstance(part, dict):
            text = part.get("text", "")
            if isinstance(text, str) and MEMORY_OPEN in text:
                return True
    return False


def _strip_from_list_content(content: list) -> list:
    """从多模态 list content 中剥离注入块。"""
    result: list = []
    for part in content:
        if isinstance(part, dict):
            text = part.get("text", "")
            if isinstance(text, str) and MEMORY_OPEN in text:
                text = _MEMORY_BLOCK_RE.sub("\n", text).strip()
                if text:
                    result.append({**part, "text": text})
            else:
                result.append(part)
        else:
            result.append(part)
    return result


def inject_memory(
    messages: list[dict],
    facts: list[str],
    max_chars: int = 1800,
    merge_system: bool = False,
) -> tuple[list[dict], str]:
    """叠加注入：把 facts 包进标记，插入 messages。

    T3 默认策略（merge_system=False）：
      在**最后一条 user 消息之前**插入独立 system 消息。
      前面的历史前缀逐字不变 → 上游 prompt cache 命中。
      记忆贴近 query → 注意力权重更高。

    兼容策略（merge_system=True）：
      合进首个 system 消息末尾（旧行为，供拒绝多 system 消息的上游使用）。
    """
    if not facts:
        return messages, ""

    body_lines = [f"- {f}" for f in facts]
    body = "\n".join(body_lines)
    if len(body) > max_chars:
        body = body[:max_chars].rsplit("\n", 1)[0] + "\n- …（已截断）"

    injected_text = f"{MEMORY_OPEN}\n{body}\n{MEMORY_CLOSE}"
    new_messages = list(messages)

    if merge_system:
        return _inject_merge_first_system(new_messages, injected_text)
    else:
        return _inject_before_last_user(new_messages, injected_text)


def _inject_before_last_user(
    messages: list[dict],
    injected_text: str,
) -> tuple[list[dict], str]:
    """T3 默认：在最后一条 user 消息前插入独立 system 消息。"""
    # 找最后一条 user 消息的位置
    last_user_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            last_user_idx = i
            break

    injection_msg = {"role": "system", "content": injected_text}

    if last_user_idx >= 0:
        # 在最后一条 user 前插入
        new_messages = (
            messages[:last_user_idx]
            + [injection_msg]
            + messages[last_user_idx:]
        )
    else:
        # 没有 user 消息 → 在末尾追加
        new_messages = messages + [injection_msg]

    return new_messages, injected_text


def _inject_merge_first_system(
    messages: list[dict],
    injected_text: str,
) -> tuple[list[dict], str]:
    """兼容模式：合进首个 system 消息末尾（旧行为）。"""
    for i, msg in enumerate(messages):
        if msg.get("role") == "system":
            existing = msg.get("content", "")
            if isinstance(existing, str):
                messages[i] = {**msg, "content": existing.rstrip() + "\n\n" + injected_text}
            else:
                messages[i] = {**msg, "content": str(existing) + "\n\n" + injected_text}
            return messages, injected_text

    # 没有 system 消息 → 在开头插入
    messages.insert(0, {"role": "system", "content": injected_text})
    return messages, injected_text


def do_inject(
    messages: list[dict],
    query: str,
    config: ProxyConfig,
    source: InjectionSource | None = None,
    auxiliary: bool = False,
    user_id: str | None = None,
    assembler: ContextAssembler | None = None,
    prefix_changed: bool = False,
    visibility: list[str] | None = None,
    allowed_exposure: str = "public",
    session_id: str = "",
    agent_id: str = "",
) -> tuple[list[dict], str, float, list[Fact], dict]:
    """完整注入流程：删旧 → 召回 → 注入。返回 (messages, 注入文本, 耗时 ms, 路由用 facts)。

    P8: 第四个返回值是本轮召回的 Fact 列表，随请求上下文传递给路由，
    避免读 inject_source.last_facts 这个跨请求可变状态。
    """
    t0 = time.perf_counter()
    source = source or InjectionSource()
    route_facts: list[Fact] = []
    wide_facts: list[Fact] = []

    cleaned = strip_previous_injection(messages)
    # V-A3：inject 模块开关（ADR-0032 §2.1；默认开=现状）。关 = 注入平面整体停
    # （prefetch/三平面/硬规则/理解层/澄清都不做），但 strip 卫生、装配（assembly
    # 模块自己的开关管）、task_units 写侧认知照常——关掉注入不许连带废掉
    # 记忆写入质量（与 ADR-0024 §4.3 装配解耦同一条理由）。
    inject_on = module_enabled("inject")
    # ADR-0028 E6.1：检索用的 query 先过理解层（原 query 不改，只影响召回）
    if _should_understand(inject_on, auxiliary, agent_id):
        query = _understand(messages, query, source, session_id)

    if not inject_on:
        facts = []
        wide_facts = []
        injected_ids: list[str] = []
        injected_matter_ids: list[str] = []
        logger.info("inject_module_off")
    elif auxiliary:
        rules = config.effective_hard_rules or _DEFAULT_HARD_RULES
        facts = [f"[MUST/NEVER] {r}" for r in rules]
        logger.info("inject_auxiliary", facts_count=len(facts))
        injected_ids = []
        injected_matter_ids = []   # aux 轮只注硬规则，没钉卡 = 无 Matter 命中
    else:
        try:
            # S1（2026-09-03）：注入面只剩硬规则；三平面 / 主动检索路径已退役。
            facts = source.build_facts(query, config, user_id=user_id, visibility=visibility, allowed_exposure=allowed_exposure)
            route_facts = list(source.last_facts)
            wide_facts = list(source.last_facts_wide)
            injected_ids = []
            injected_matter_ids = []
        except Exception as e:
            logger.warning("inject_source_failed", error=str(e))
            rules = config.effective_hard_rules or _DEFAULT_HARD_RULES
            facts = [f"[MUST/NEVER] {r}" for r in rules]
            injected_ids = []
            injected_matter_ids = []

        elapsed_ms = (time.perf_counter() - t0) * 1000
        if elapsed_ms > config.hotpath_budget_ms:
            logger.warning("inject_over_budget", elapsed_ms=elapsed_ms, budget_ms=config.hotpath_budget_ms)
            facts = [f for f in facts if "[MUST/NEVER]" in f] if facts else [
                f"[MUST/NEVER] {r}" for r in (config.effective_hard_rules or _DEFAULT_HARD_RULES)
            ]
            injected_ids = []      # 降级为只注硬规则 = 无 Memory Index 条目命中
            injected_matter_ids = []   # 同理：没钉卡就没有 Matter 命中

    # CAP: 上下文感知压缩（ADR-0016），在注入前压缩旧历史
    cleaned, cap_info = _apply_cap(
        cleaned, assembler, wide_facts, prefix_changed, session_id=session_id,
        source=source, query=query, agent_id=agent_id,
    )

    # ADR-0024 T1：单元化是**认知**、降解是**改写**，两者解耦——
    # 装配关闭（assembler=None / enabled=false）时仍然产出单元，
    # 否则关掉装配会连带废掉记忆写入质量（ADR-0024 §4.3）。
    # 索引口径：对**未剥离**的 messages 计算 = Memory Hub 存的 original_messages，
    # 故 TaskUnit 的 index 直接可用于回查正文（T0 修正 D：只存引用不存正文）。
    cap_info["units"] = build_task_units(messages)
    # 注入命中反馈环（injected_fact_ids / injected_fact_keys → ref_count → importance，
    # injected_matter_ids → Matter.last_hit_at）随 S1 **永久断线**：生产者
    # `_capture_injected_*` 已删，键保留（decision_meta schema 不动）、恒空。
    # 0.3.0 重建检索时须连这条环一起重建（MQ 台账索引域登记）。
    cap_info["injected_fact_ids"] = list(injected_ids)
    cap_info["injected_fact_keys"] = {}
    cap_info["injected_matter_ids"] = list(injected_matter_ids)

    # U9 C 层澄清已随 S4（2026-09-03）删除：env 直读开关、默认关、零 live 读数。
    # `ReconstructionRecord.clarify_text` 字段保留（schema 不动），恒为空。
    clarify_text = ""

    # U2/U9：reconstruction 记录（A/B 层降解决策 + B 指纹 + C 澄清 → 落 Memory Hub 审计，
    # Memory Index 蒸馏永不读它——G3/I5，见 test_index_never_consumes_reconstruction）。
    _recon = None
    _prune = cap_info.get("prune") or {}
    _pruned_units = list(_prune.get("pruned") or [])
    if cap_info.get("degrade_plan") or clarify_text or cap_info.get("b_layer") or _pruned_units:
        _recon = {
            "layer": ("A" + ("B" if cap_info.get("b_layer") else "")
                      + ("C" if clarify_text else "") + ("P" if _pruned_units else "")),
            "degrade_plan": {int(k): v for k, v in (cap_info.get("degrade_plan") or {}).items()},
            "fingerprint": {"terms": list(cap_info.get("b_fingerprint") or [])},
            "clarify_text": clarify_text,
        }
        if _pruned_units:
            # ADR-0028 §3 约束4：裁剪决策整体落 reconstruction 审计，可回放可解释。
            _recon["prune"] = {
                "pruned": _pruned_units,
                "scores": _prune.get("scores", {}),
                "target": _prune.get("target", 0),
                "kept_chars": _prune.get("kept_chars", 0),
                "chars_before": _prune.get("chars_before", 0),
            }
    cap_info["reconstruction"] = _recon

    new_messages, injected_text = inject_memory(
        cleaned, facts, config.inject_max_chars,
        merge_system=config.inject_merge_system,
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info("inject_done", facts_count=len(facts), elapsed_ms=round(elapsed_ms, 2),
                merge_system=config.inject_merge_system, auxiliary=auxiliary,
                user_id=user_id or "", planes=False)
    return new_messages, injected_text, elapsed_ms, route_facts, cap_info


def _apply_cap(
    messages: list[dict],
    assembler: ContextAssembler | None,
    wide_facts: list[Fact],
    prefix_changed: bool,
    session_id: str = "",
    *,
    source: Any = None,  # noqa: ANN401  InjectionSource（取 embedder，冷启裁剪用）
    query: str = "",
    q_vec: list[float] | None = None,
    agent_id: str = "",  # MQ-CA3：只为 assembly_done 的归属维度透传，不参与任何判定
) -> tuple[list[dict], dict]:
    """上下文装配（ADR-0019）+ Prompt Cache TTL 感知（ADR-0028 §2/§3 / E4）。

    L1 证据降解（closed 单元巨型 tool 结果 → 占位摘录）常开；
    L2 单元摘要（条数/体积触发 → 最老单元换 Memory Index 摘要）承接原 CAP 语义。

    ADR-0028 E4 在其上加一道 **cache 温热门**：

      - **温热**（同模型 + 未过 TTL + agent 没自己压缩过）→ 只跑 L1（内容纯函数、
        前缀稳定）+ 原样复用上一轮冻结的 L2/裁剪计划，**禁止新增**任何降解/裁剪；
      - **冷**（模型换了 / TTL 过期 / 首轮）→ 装配正常跑 + E4.3 相关性裁剪，
        跑完把计划冻结起来给后续温热轮复用。

    不触发时原样返回（走叠加模式）。assembler=None 或关闭也原样返回。
    T4(ADR-0018 §4.4 C4): cap_info 供 Turn.decision_meta 记录
    （triggered/summary_hash/kept_recent_n + ADR-0019 新增 evidence_degraded/units_dropped/chars_*）。
    """
    cap_info: dict[str, Any] = {
        "triggered": False, "summary_hash": "", "kept_recent_n": 0,
        "evidence_degraded": 0, "units_dropped": 0,
        "chars_before": 0, "chars_after": 0,
        # ADR-0024 T1：由调用方在 _apply_cap 之后填（索引基 = 未剥离的 messages）。
        # 放在这里保证 key 恒存在，下游 .get("units", []) 不必兜 None。
        "units": [],
        # ADR-0028 E4：本轮 cache 判定与裁剪决策（落 Memory Hub reconstruction 审计）。
        "cache_warm": False, "prune": None,
    }
    if assembler is None or not assembler.enabled:
        return messages, cap_info
    if not assembly_enabled_for(agent_id):
        # 2026-09-03 拍板 e（MQ-CA6）：装配只对 `BLADEX_ASSEMBLY_AGENTS` 清单内的
        # agent 开；其余走与 `assembler=None` **逐字相同**的原样返回。仪器不关：
        # `assembly_done` 照打（evidence_degraded=0、chars 前后相等），验收判据
        # "非 CC evidence_degraded=0 ∧ chars_after/chars_before ≥ 0.95" 才有读数。
        _n = estimate_context_chars(messages)
        logger.info("assembly_done", evidence_degraded=0, units_dropped=0,
                    chars_before=_n, chars_after=_n,
                    msgs_before=len(messages), msgs_after=len(messages),
                    session_id=session_id, agent_id=agent_id,
                    skipped="agent_not_enabled")
        return messages, cap_info

    warm = _cache_warm(session_id, prefix_changed)
    cap_info["cache_warm"] = warm

    try:
        assembled, info = assembler.assemble(
            messages, wide_facts, prefix_changed=prefix_changed, session_id=session_id,
            allow_l2=not warm, agent_id=agent_id,
        )
    except TypeError:
        # 旧签名的 assembler（测试替身）——退回不带新参数
        try:
            assembled, info = assembler.assemble(
                messages, wide_facts, prefix_changed=prefix_changed, session_id=session_id,
            )
        except TypeError:
            assembled, info = assembler.assemble(
                messages, wide_facts, prefix_changed=prefix_changed)

    if warm:
        # 温热：原样复用冻结计划（不新增任何降解/裁剪）
        plan = _cache_registry().frozen_plan(session_id)
        assembled, touched = apply_frozen_plan(assembled, plan)
        if touched:
            info = dict(info)
            info["changed"] = True
            info["units_dropped"] = info.get("units_dropped", 0) + touched
        logger.info("cache_warm_full_forward", session_id=session_id[:16],
                    frozen_units=touched)
    else:
        # 冷启：允许相关性裁剪（cache 反正要重建，此时裁剪只赚不赔）
        assembled, prune_decision = _prune_if_cold(assembled, source, query, q_vec, session_id)
        cap_info["prune"] = prune_decision
        if prune_decision and prune_decision.get("pruned"):
            info = dict(info)
            info["changed"] = True
        _cache_registry().freeze_plan(session_id, {
            "l2_summary_msg": info.get("l2_summary_msg"),
            "l2_dropped_unit_keys": list(info.get("l2_dropped_unit_keys") or []),
            "pruned_unit_keys": list((prune_decision or {}).get("pruned") or []),
        })
    cap_info["evidence_degraded"] = info.get("evidence_degraded", 0)
    cap_info["units_dropped"] = info.get("units_dropped", 0)
    cap_info["chars_before"] = info.get("chars_before", 0)
    cap_info["chars_after"] = info.get("chars_after", 0)
    # U2/U9：A/B 层降解决策 + B 层指纹（reconstruction 落 Memory Hub 的原料）
    cap_info["degrade_plan"] = info.get("degrade_plan", {}) or {}
    cap_info["b_fingerprint"] = info.get("b_fingerprint", []) or []
    cap_info["b_layer"] = bool(info.get("b_layer", False))
    if not info.get("changed"):
        return messages, cap_info
    cap_info["triggered"] = True
    # 从装配结果反推 summary_hash + kept_recent_n（摘要消息含 SUMMARY_OPEN 标记）
    import hashlib
    summary_idx = -1
    for i, m in enumerate(assembled):
        c = m.get("content", "")
        if isinstance(c, str) and SUMMARY_OPEN in c:
            summary_idx = i
            cap_info["summary_hash"] = hashlib.sha256(c.encode()).hexdigest()[:16]
            break
    if summary_idx >= 0:
        cap_info["kept_recent_n"] = len(assembled) - summary_idx - 1
    return assembled, cap_info


async def do_inject_async(
    messages: list[dict],
    query: str,
    config: ProxyConfig,
    source: InjectionSource | None = None,
    auxiliary: bool = False,
    user_id: str | None = None,
    assembler: ContextAssembler | None = None,
    prefix_changed: bool = False,
    visibility: list[str] | None = None,
    allowed_exposure: str = "public",
    session_id: str = "",
    agent_id: str = "",
) -> tuple[list[dict], str, float, list[Fact], dict]:
    """异步注入流程：检索进线程池 + 真超时降级（F3，ADR-0008 §6.2）。

    P8: 第四个返回值是本轮召回的 Fact 列表。facts 在 executor
    同线程内抓取（build_facts 返回后立即读 last_facts），避免
    跨请求读 inject_source.last_facts 可变共享状态。

    S1（2026-09-03）：三平面注入已退役，注入面只剩硬规则；超时/异常降级路径不变。
    """
    t0 = time.perf_counter()
    source = source or InjectionSource()
    route_facts: list[Fact] = []
    wide_facts: list[Fact] = []
    facts: list[str] = []
    cleaned = strip_previous_injection(messages)
    # V-A3：inject 模块开关（与同步 do_inject 同一道门——两个入口一个判据，
    # 语义见 do_inject 内注释；测试 test_module_inject_gate.py 两个入口都钉）。
    inject_on = module_enabled("inject")
    # ADR-0028 E6.1：检索用的 query 先过理解层（原 query 不改，只影响召回）
    if _should_understand(inject_on, auxiliary, agent_id):
        query = _understand(messages, query, source, session_id)

    if not inject_on:
        facts = []
        wide_facts = []
        injected_ids: list[str] = []
        injected_matter_ids: list[str] = []
        logger.info("inject_module_off")
    elif auxiliary:
        rules = config.effective_hard_rules or _DEFAULT_HARD_RULES
        facts = [f"[MUST/NEVER] {r}" for r in rules]
        logger.info("inject_auxiliary", facts_count=len(facts))
        wide_facts = []
        injected_ids = []
        injected_matter_ids = []   # aux 轮只注硬规则，没钉卡 = 无 Matter 命中
    else:
        try:
            loop = asyncio.get_running_loop()

            def _build_and_capture() -> tuple[list[str], list[Fact], list[Fact]]:
                # S1（2026-09-03）：注入面只剩硬规则；仍走 executor + 真超时，
                # 保住 F3 的降级路径形态（`inject_timeout` 仪器不变）。
                lines = source.build_facts(query, config, user_id, visibility, allowed_exposure)
                return (lines, list(source.last_facts), list(source.last_facts_wide))

            facts, route_facts, wide_facts = await asyncio.wait_for(
                loop.run_in_executor(None, _build_and_capture),
                timeout=config.hotpath_budget_ms / 1000.0,
            )
            injected_ids = []
            injected_matter_ids = []
        except TimeoutError:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.warning("inject_timeout", elapsed_ms=round(elapsed_ms, 2),
                           budget_ms=config.hotpath_budget_ms)
            metrics_mod.record_degradation(metrics_mod.INJECT_TIMEOUT)
            rules = config.effective_hard_rules or _DEFAULT_HARD_RULES
            facts = [f"[MUST/NEVER] {r}" for r in rules]
            wide_facts = []
            injected_ids = []      # 只注硬规则 = 无 Memory Index 条目命中
            injected_matter_ids = []
        except Exception as e:
            logger.warning("inject_source_failed", error=str(e))
            metrics_mod.record_degradation(metrics_mod.INJECT_FAILED)
            rules = config.effective_hard_rules or _DEFAULT_HARD_RULES
            facts = [f"[MUST/NEVER] {r}" for r in rules]
            wide_facts = []
            injected_ids = []
            injected_matter_ids = []

    # CAP: 上下文感知压缩（ADR-0016），在注入前压缩旧历史
    cleaned, cap_info = _apply_cap(
        cleaned, assembler, wide_facts, prefix_changed, session_id=session_id,
        source=source, query=query, agent_id=agent_id,
    )

    # ADR-0024 T1：单元化是**认知**、降解是**改写**，两者解耦——
    # 装配关闭（assembler=None / enabled=false）时仍然产出单元，
    # 否则关掉装配会连带废掉记忆写入质量（ADR-0024 §4.3）。
    # 索引口径：对**未剥离**的 messages 计算 = Memory Hub 存的 original_messages，
    # 故 TaskUnit 的 index 直接可用于回查正文（T0 修正 D：只存引用不存正文）。
    cap_info["units"] = build_task_units(messages)
    # 注入命中反馈环（injected_fact_ids / injected_fact_keys → ref_count → importance，
    # injected_matter_ids → Matter.last_hit_at）随 S1 **永久断线**：生产者
    # `_capture_injected_*` 已删，键保留（decision_meta schema 不动）、恒空。
    # 0.3.0 重建检索时须连这条环一起重建（MQ 台账索引域登记）。
    cap_info["injected_fact_ids"] = list(injected_ids)
    cap_info["injected_fact_keys"] = {}
    cap_info["injected_matter_ids"] = list(injected_matter_ids)

    # U9 C 层澄清已随 S4（2026-09-03）删除：env 直读开关、默认关、零 live 读数。
    # `ReconstructionRecord.clarify_text` 字段保留（schema 不动），恒为空。
    clarify_text = ""

    # U2/U9：reconstruction 记录（A/B 层降解决策 + B 指纹 + C 澄清 → 落 Memory Hub 审计，
    # Memory Index 蒸馏永不读它——G3/I5，见 test_index_never_consumes_reconstruction）。
    _recon = None
    _prune = cap_info.get("prune") or {}
    _pruned_units = list(_prune.get("pruned") or [])
    if cap_info.get("degrade_plan") or clarify_text or cap_info.get("b_layer") or _pruned_units:
        _recon = {
            "layer": ("A" + ("B" if cap_info.get("b_layer") else "")
                      + ("C" if clarify_text else "") + ("P" if _pruned_units else "")),
            "degrade_plan": {int(k): v for k, v in (cap_info.get("degrade_plan") or {}).items()},
            "fingerprint": {"terms": list(cap_info.get("b_fingerprint") or [])},
            "clarify_text": clarify_text,
        }
        if _pruned_units:
            # ADR-0028 §3 约束4：裁剪决策整体落 reconstruction 审计，可回放可解释。
            _recon["prune"] = {
                "pruned": _pruned_units,
                "scores": _prune.get("scores", {}),
                "target": _prune.get("target", 0),
                "kept_chars": _prune.get("kept_chars", 0),
                "chars_before": _prune.get("chars_before", 0),
            }
    cap_info["reconstruction"] = _recon

    new_messages, injected_text = inject_memory(
        cleaned, facts, config.inject_max_chars,
        merge_system=config.inject_merge_system,
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info("inject_done", facts_count=len(facts), elapsed_ms=round(elapsed_ms, 2),
                merge_system=config.inject_merge_system, auxiliary=auxiliary,
                user_id=user_id or "", planes=False)
    return new_messages, injected_text, elapsed_ms, route_facts, cap_info
