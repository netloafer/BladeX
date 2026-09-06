"""身份分层解析 — 用户/agent/会话/轮次（ADR-0008 §6.4）。

分层策略（从最可靠往下退）：
  1. 显式信号：独立 API Key / X-Agent-ID header
  2. 被动指纹：system prompt 模式 + tool 签名（无需 agent 配合）
  3. 透传 metadata：User-Agent
  4. 推断兜底：时间窗口

会话 id：显式带就用；不带按"对话前缀连续性"推断；失败退到"API Key + 时间窗口"。
轮次：按消息条数推（元数据，不再当 Memory Hub 主键，见 T1）。

T2（M-proxy-1.5）：指纹跳过首条 system 消息（防动态时间戳断链），
加尾部接续判断（新请求是已知会话 messages 的尾部超集 → 同会话）。
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from typing import Any

import structlog

from bladex_core.envelope import ENVELOPE_MAX_OFFSET
from bladex_core.flags import flag_enabled

from bladex_proxy.agent_bucket import (
    VENDOR_SESSION_SUFFIXES,
    bucket_unknown_agent,
    extract_vendor_ids,
    redact_headers,
)
from bladex_proxy.agent_rules import AgentFingerprintRule as _ExternalRule
from bladex_proxy.agent_rules import load_agent_rules
from bladex_proxy.agent_registry import _agent_registry
from bladex_proxy.innerloop import INNER_LOOP_AUX_SOURCE as _INNER_LOOP_AUX_SOURCE
from bladex_proxy.innerloop import INNER_LOOP_MARKER as _INNER_LOOP_MARKER
from bladex_proxy.models import AgentSource, ChatCompletionRequest, Identity, SessionIdSource

# Agent 自带的 profile 标记（从 system prompt 中提取，无需用户配置）
# Hermes 在 system prompt 中写 "Active Hermes profile: accept/default/..."
# 这个标记由 agent 软件自己生成，每轮请求都会带，是配置文件最可靠的标识
_HERMES_PROFILE_RE = re.compile(r"Active Hermes profile: (\S+)", re.IGNORECASE)

# 通用 profile 标记模式表（可扩展其它 agent）
_PROFILE_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("hermes", _HERMES_PROFILE_RE),
]

logger = structlog.get_logger()

# 时间窗口分组：同一 API Key 在此秒数内的请求算同一会话（兜底用）
_TIME_WINDOW_SECONDS = 300  # 5 分钟


class _FpEpochWindow:
    """fp: 指纹会话桶的时间窗切分（MQ-S9 修值 / G12.2-pre；窗口 4h = 2026-08-20 拍板）。

    病：前缀指纹是无状态哈希——agent 固定开场白让不同日子的会话算出同一个 fp:
    （实测一个桶 8 天 / 474 轮 / 47 卡；重建后 max 跨度 20.47 天，
    `docs/benchmarks/session-gaps-mqs9-20260820.txt`）。
    修：同 (user, agent, fp) 桶内相邻请求 gap 超过 `BLADEX_SESSION_FP_WINDOW_S`
    → 之后的轮次挂 `fp:<hash>.e<yyyymmddHH>` 后缀。后缀 = 新纪元起点小时（UTC），
    **无计数器状态**——两个纪元不可能同小时开始（切换本身要求 gap > 窗口 ≥ 1h），
    所以不会撞后缀，也就不需要跨重启持久化的序号。

    🔴 四条边界（依据见执行卡 G12.2-pre 拍板记录）：
    1. **首纪元不带后缀**（保持 `fp:<hash>` 原形）——存量数据零漂移；进程重启
       状态清空后退回原形 = 最坏回到旧撞桶行为，严格不劣于现状。
    2. 只作用于**新入轮次**（Memory Hub key 写入即冻结，不改历史、不占重放窗口）。
    3. 只碰 fp: 桶——同一份读数里显式 session 对照段零长尾；切显式 session
       是替 agent 做主（红线 6 邻区）。
    4. 尾部接续（TAIL_CONTINUATION）轮必须 `touch()` 续命——长会话全程走接续、
       不再过指纹路径，不续命的话"6 小时连续工作后开新对话"会被误判成超窗。
    """

    _MAX_STATE = 4096  # fp 桶数量级实测 ~40/三周，这个帽只防病态膨胀

    def __init__(self) -> None:
        # (user_id, agent_id, fp 基名) -> (当前纪元后缀，""=首纪元, last_seen 秒)
        self._state: dict[tuple[str, str, str], tuple[str, float]] = {}

    @staticmethod
    def _window_s() -> float:
        from bladex_core.flags import flag_number
        return flag_number("BLADEX_SESSION_FP_WINDOW_S")

    @staticmethod
    def _base(session_id: str) -> str:
        return session_id.split(".e", 1)[0]  # fp 哈希是 hex，不含 ".e"

    def apply(self, user_id: str, agent_id: str, session_id: str,
              now: float | None = None) -> str:
        """指纹路径出口调用：返回（可能带纪元后缀的）session_id。"""
        window = self._window_s()
        if window <= 0 or not session_id.startswith("fp:"):
            return session_id
        now = time.time() if now is None else now
        key = (user_id, agent_id, self._base(session_id))
        prev = self._state.get(key)
        if prev is None:
            self._evict_if_needed()
            self._state[key] = ("", now)
            return key[2]
        suffix, last_seen = prev
        gap = now - last_seen
        if gap > window:
            suffix = ".e" + time.strftime("%Y%m%d%H", time.gmtime(now))
            logger.info("session_fp_window_split", user_id=user_id, agent_id=agent_id,
                        fp=key[2], suffix=suffix, gap_s=round(gap))
        self._state[key] = (suffix, now)
        return key[2] + suffix

    def touch(self, user_id: str, agent_id: str, session_id: str,
              now: float | None = None) -> None:
        """尾部接续路径调用：只续 last_seen，不改纪元（见类注边界 4）。"""
        if not session_id.startswith("fp:"):
            return
        now = time.time() if now is None else now
        key = (user_id, agent_id, self._base(session_id))
        prev = self._state.get(key)
        suffix = prev[0] if prev is not None else session_id[len(key[2]):]
        self._state[key] = (suffix, now)

    def _evict_if_needed(self) -> None:
        if len(self._state) < self._MAX_STATE:
            return
        oldest = sorted(self._state.items(), key=lambda kv: kv[1][1])[:256]
        for k, _ in oldest:
            del self._state[k]


_fp_epochs = _FpEpochWindow()

# 尾部接续比对时取多少条非 system 消息做前缀匹配
_PREFIX_MATCH_COUNT = 4

# 缓存每个 (user_id, agent_id) 最近多少个会话的前缀
_MAX_CACHED_SESSIONS_PER_KEY = 50


# ── 身份常量（ADR-0021 §2.3，2026-08-16 修订）────────────────────────────────

#: 个人模式（无 `identity.toml`）的唯一身份。**常量，不含 key hash、不随凭证变。**
#:
#: 语义：没有 identity.toml = 这是一台个人自部署实例 = 只有一个人。换 key、换设备、
#: 轮换凭证都不该改变"我是谁"。`Fact.scope` 相应派生为 `personal:local`
#: （`consolidation_proxy.py` 从 user_id 拼，无需单独改）。
#:
#: 🔴 改这个值 = 全库身份漂移，历史 Fact 的 scope 会与新流量分家。真要改必须配
#: 全量重建时的折叠映射，参见 docs/reviews/adr0021-migration-audit-20260816.md §5.4.4。
LOCAL_USER_ID = "local"

#: 无 API Key 请求的身份。**任何模式下都不并入 LOCAL_USER_ID 或任何 principal。**
#:
#: 它不是"另一个用户"，是 **auth 开关正确性的探测器**（2026-08-16 用户拍板）：
#: 正常配置下这个值永远不该出现，一旦出现就说明 auth 关了或 key 没带上。
#: 自动并入会把这个唯一的信号消解掉——那之后配置错了也永远发现不了。
#: 配套告警：`identity_no_api_key`（warning 级，本文件末尾）。
ANONYMOUS_USER_ID = "anonymous"


# ── T2(ADR-0018 R2): auxiliary 识别 ──
# Hermes 的异步委派回执 / 网页摘要子任务 / 上下文压缩 handoff 以 **user 角色**进入，
# system prompt 与主会话相同 -> 仅靠 system prompt 关键词（_AGENT_RULES 的 auxiliary）
# 漏判（实测 465/507 轮 auxiliary=0，但 ≥31 轮是子代理/委派流量）。
# classify_auxiliary 补 user 消息模板前缀这道，Memory Hub 存初判、Memory Index 重建用本函数重判
# （不信任冻结的 turn.auxiliary）。

#: 信封标记允许出现的**最深位置**（字符）。超过它 ⇒ 这一轮是在谈论信封，不是信封。
#: 标定过程与双峰分布见 `classify_auxiliary` 里的注释（MQ-A19）。
#: 🔴 改这个值前先重跑那份分布——它是**标定常数**，不是随手取的圆整数。
#:
#: 2026-08-28：定义搬到 `bladex_core.envelope`（envelope 的整条剥除也需要这道门，
#: 见那边的注释）。这里只做别名，**不许写第二个字面量** —— 同一个参数在两条
#: 调用路径上各有一个默认值 = 缺陷，与哪个值更合理无关（刚性原则 12）。
_ENVELOPE_MAX_OFFSET = ENVELOPE_MAX_OFFSET

# user 消息里的信封标记（最早命中且靠近开头 ⇒ auxiliary）。大小写不敏感。
# 🔴 匹配是**任意位置的子串 + 位置门**，不是 `startswith`——实测真信封常带小前言
# （`[Request interrupted by user]`、`<local-command-caveat>` 等，最深 278）。
# 位置门见 `_ENVELOPE_MAX_OFFSET`。
_AUX_USER_PATTERNS: list[tuple[str, str]] = [
    ("async_delegation_batch", "[ASYNC DELEGATION BATCH COMPLETE"),
    ("web_content_summary", "Please process this web content"),
    ("background_fanout", "A background fan-out of"),
    ("context_compaction_handoff", "[CONTEXT COMPACTION - REFERENCE ONLY]"),
    ("task_list_preserved", "[Your active task list was preserved across context compression]"),
    ("summarization_checkpoint", "You are a summarization agent creating a context checkpoint"),
    # ── ADR-0024 T3：Claude Code 内部信封（归档 Memory Hub 全量实测，1990 轮）──
    # 真实用户轮 1242 (62%) / transcript 信封 456 (23%) / slash·通知信封 292 (15%)，
    # 三类信封此前 classify_auxiliary 命中 **0** —— 38% 的 turn 被当成真实用户轮，
    # 既污染归属，又让蒸馏对着信封做无用功。
    # 注：安全检查子 agent 的 transcript 里**包含**用户原话，但那些原话在各自的
    # 真实用户轮里独立存在（1242 轮），标 aux 不丢信息。
    ("claude_code_safety_transcript", "<transcript>"),
    ("claude_code_slash_command", "<command-name>"),
    ("claude_code_local_command", "<local-command-stdout>"),
    ("claude_code_task_notification", "<task-notification>"),
    # ── U3（ADR-0024 遗留卡）：Hermes 侧内部调用指纹 ──
    # 这三类 hermes 内部调用此前 classify_auxiliary 未命中，既污染归属又让蒸馏做无用功
    # （consolidator new_facts=0 反复出现）。都归 DISTILL_ONLY（见下）：跳蒸馏，不动路由。
    ("hermes_memory_save", "Review the conversation above and consider saving to memory"),
    ("hermes_empty_response_retry", "You just executed tool calls but returned an empty response"),
    # ── MS-2 信封普查发现（2026-08-06，真实 Memory Hub 全量）──
    # 输出被截断后的续写提示：实测 364 次，此前无任何指纹 → 被当成真实用户轮，
    # 既污染归属又让蒸馏对着一句系统提示做无用功。
    # 归 DISTILL_ONLY：user 那侧没内容，但 assistant 续写的是真实工作，产出照收。
    ("truncated_response_continue", "[System: Your previous response was truncated"),
    # ── MQ-S12（2026-08-18）：三条脚手架模板，实测在**污染归属**（不只是白跑蒸馏）──
    # 这些轮的 user 侧是脚手架、内容与任务无关，但 assistant 侧在做真实工作 →
    # 蒸出的 fact 带着上一件事的上下文却落到当前卡上，形成跨话题错边。实证形态：
    #   `local-llm-inference-macos` ⇔ `bladex 正式 CLI 注册方案`
    #   `GLM5.2 报告投资人招募信息核查修正` ⇔ `泰山啤酒…优先债权人识别调查`
    # 三条同归 DISTILL_ONLY（user 侧无内容、assistant 侧照收，与 G4.1/MS-2 口径一致）。
    #
    # 🔴 `hermes_skill_library_update` 不得只取前缀 "Review the conversation above and"
    # ——那与 `hermes_memory_save` 共享前缀，取短了会把两条并成一条、失去可区分性。
    # 该规则 G4.1 在 2026-08-15 就定稿（live 取样 60/60 逐字相同），至今未落地。
    ("hermes_skill_library_update", "Review the conversation above and update the skill library"),
    # 🔴 MQ-S45 修复（2026-08-28）：原判据是 "…agent history **added since**"，
    # 而 live 里至少三种从句：`added since the last checkpoint` /
    # `added since your last approval assessment` / **`whose request action you
    # are assessing`**。第三种落空 ⇒ 判 REAL ⇒ 整段父 agent 转录当用户意图蒸馏。
    # 与 MS-2 的 em dash（811 轮压缩交接整段漏判）**同型第三例**：
    # 指纹按一次采样写死，agent 换个变体就复发。故改用**共同前缀**一次盖住。
    # 与 `envelope._WHOLE_MESSAGE_MARKERS` 逐字一致（对账测试守）。
    ("codex_history_injection", "The following is the Codex agent history"),
    ("session_resume_recap", "The user stepped away and is coming back"),
    # 注：转录形态 "User: …\n\nAssistant: …" 用结构化匹配（见 classify_auxiliary），
    # 不进本表——"User:" 做子串前缀太泛，会误伤真实用户消息。
]

# ── 「不蒸馏」与「可降档」是两回事（2026-07-28 路由回归修复）──
#
# `auxiliary` 这个标记有两个消费方，语义需求并不相同：
#   蒸馏侧（consolidation）：这一轮内容没有记忆价值 → 跳过
#   路由侧（ADR-0013 T9）  ：这一轮是廉价内部调用   → 强制便宜档
#
# T3 加的 Claude Code 信封指纹只满足前者：`<command-name>` / `<transcript>` 这些
# 出现在**末条 user 消息**里，但那一轮模型要做的往往是真实工作（slash 命令回显之后
# 紧接着就是用户的任务）。把整轮打到 weak 档 = 用户拿 lite 模型干正事。
#
# 实测（2026-07-28 Claude Code 真实流量）：`source=auxiliary` 82 次、
# `source=requested` 仅 1 次，全部落到 doubao-seed-2.0-lite——用户配置的 GLM-5.2
# 被彻底旁路。
#
# 故：本集合内的规则**只用于蒸馏跳过，不参与路由降档**。
# Hermes 原生的 checkpoint / 压缩 handoff 等仍享受 T9 的便宜档收益。
DISTILL_ONLY_AUX_RULES: frozenset[str] = frozenset({
    "claude_code_safety_transcript",
    "claude_code_slash_command",
    "claude_code_local_command",
    "claude_code_task_notification",
    # U3：hermes 三类内部调用——"没有记忆价值"（跳蒸馏），但**不**代表"可用便宜模型"。
    # memory_save 是轻量分类调用、empty_retry/transcript 后续往往是真实工作，一律不降档
    # （保守，避免 T3 式"用户配置模型被旁路"回归）。语义拆分见本文件顶部注释。
    "hermes_memory_save",
    "hermes_empty_response_retry",
    "hermes_transcript_replay",
    # MS-2：续写提示的 user 那侧没内容，但 assistant 续的是真实工作 → 产出照收、不降档。
    "truncated_response_continue",
    # MQ-S12：三条脚手架模板，同口径（user 侧脚手架、assistant 侧真实工作）。
    "hermes_skill_library_update",
    "codex_history_injection",
    "session_resume_recap",
})


#: 每条 `DISTILL_ONLY_AUX_RULES` 规则 → `(envelope kind, 是否整条剥空)`。
#:
#: ## 为什么必须有这张表（2026-08-28，live 事故驱动）
#:
#: `DISTILL_ONLY` 的语义是「**user 侧是信封**、assistant 侧照收」。rebuild 因此
#: 不整轮丢，而是把 user 侧**交给 `bladex_core.envelope` 剥离**
#: （`memory_index.py` 那段注释的原话）。但那是一次交接——交出去之后，
#: 没有任何东西保证接活的那边有对应的剥离模式。
#:
#: live 读数：`codex:guardian` 600 轮里 **595 轮判 scaffold**（指纹没问题），
#: 字符留存率 **99.6%** —— 整段父 agent transcript 原样进 `user_messages`，
#: 蒸出 2299 条 `origin:user_direct`（占该 agent 全部 fact 的 88%）。
#: 六条规则的剥离是**空头支票**：集合里有、envelope 里没有。
#:
#: `memory_index.py:6534` 的注释记着上一次同型事故（"集合建好了、语义写清了、
#: 没人读它"）。这是**第二次复发**，换了个形状：读了、把活交出去、下一棒是空的。
#: 故把交接契约**显式写下来并机制化对账**（ADR-0030 H1 生产者/消费者对账 gate）。
#:
#: ## 第二个字段的含义
#:
#: `True`  = 剥完必须为空（整条即信封，没有任何一部分属于用户意图）
#: `False` = 剥壳但**保留有效载荷**——目前只有 hermes 转录壳：
#:           `User: <真实问句>\n\nAssistant: …` 的第一段就是用户真正问的那句话，
#:           整条丢是错的（MS-2 实测"帮我核实 3000 万共益债公告"因此从未入库）。
#:
#: 🔴 加新 `DISTILL_ONLY` 规则时必须同时加这里的落点**和**测试里的 live 形态样本，
#: 否则 `test_distill_only_envelope_contract.py` 红。这正是要守的东西：
#: 规则与落点必须成对出现，"以后再补 envelope" = 又一张空头支票。
DISTILL_ONLY_ENVELOPE_CONTRACT: dict[str, tuple[str, bool]] = {
    # Claude Code 四条：成对标签，`_ENVELOPE_PATTERNS` 里逐条对应
    "claude_code_safety_transcript": ("transcript", True),
    "claude_code_slash_command": ("slash_command", True),
    "claude_code_local_command": ("local_command", True),
    "claude_code_task_notification": ("task_notification", True),
    # hermes 转录壳：唯一**保留有效载荷**的一条（见上）
    "hermes_transcript_replay": ("hermes_transcript", False),
    # 以下六条 2026-08-28 才有落点。此前全是空头支票。
    "hermes_memory_save": ("hermes_memory_save", True),
    "hermes_empty_response_retry": ("hermes_empty_response_retry", True),
    "hermes_skill_library_update": ("hermes_skill_library_update", True),
    "truncated_response_continue": ("truncated_response_continue", True),
    "codex_history_injection": ("codex_history_injection", True),
    "session_resume_recap": ("recap_scaffold", True),
}


def is_cheap_tier_auxiliary(aux_source: str) -> bool:
    """该 auxiliary 规则是否应触发路由降档（ADR-0013 T9）。

    空 aux_source（未命中 auxiliary）→ False。
    命中 `DISTILL_ONLY_AUX_RULES` → False（只跳蒸馏，不降档）。
    其余 auxiliary 规则 → True（真·内部廉价调用）。
    """
    if not aux_source:
        return False
    return aux_source not in DISTILL_ONLY_AUX_RULES


#: **同一件事的续做**——user 那侧是脚手架/续写提示/恢复提示，assistant 侧接着做的
#: 是**上一轮那件正事**。这类轮次该带账本面（它正在推进任务，账本就是它的工作状态）。
#:
#: 🔴 这是 `auxiliary` 的**第三套语义**，与另外两套刻意不同集：
#:   蒸馏问「有没有记忆价值」→ `DISTILL_ONLY_AUX_RULES`
#:   路由问「能不能用便宜模型」→ `is_cheap_tier_auxiliary`
#:   账本问「这一轮该不该有工作状态」→ 本集合
#: `is_isolated_subcall` 的注释早就写明"账本面是第三个消费方……同一个旋钮控三件事，
#: 是那次事故的形状"，却只为孤立子调用一个子情形另开了判据，其余仍读裸
#: `identity.auxiliary` ⇒ 2026-07-28 事故的**第三次复发**（MQ-L20）。
#:
#: 入选判据是**结构性的**：这一轮的 user 侧不是新任务、而是让 assistant
#: **接着上一轮**（续写 / 重试 / 恢复 / 历史回灌）。不确定的一律不入——
#: 漏给账本面只是少一轮工作状态，错给则可能开出噪声账本（jydesignhk 五卡前车之鉴）。
CONTINUATION_AUX_RULES: frozenset[str] = frozenset({
    "truncated_response_continue",     # 输出被截断，assistant 续写同一段
    "hermes_empty_response_retry",     # 空响应重试，重做的是同一件事
    "codex_history_injection",         # 历史回灌后接着干
    "session_resume_recap",            # 用户离开又回来，继续原任务
})


def is_ledgerless_auxiliary(aux_source: str) -> bool:
    """该 auxiliary 规则是否应**挡住账本/工具面**（MQ-L20）。

    空 aux_source → False（不挡）。
    命中 `CONTINUATION_AUX_RULES` → False（同一件事的续做，该带账本）。
    其余 auxiliary 规则 → True（真·独立内部调用，不给账本面）。
    """
    if not aux_source:
        return False
    return aux_source not in CONTINUATION_AUX_RULES


def brings_no_tools(tools: list | None) -> bool:
    """这一轮 agent **自己一个工具都没带**（账本面 gating，MQ-L21）。

    🔴 判据的立论：**一个不带任何工具的请求，要的是一段文本，不是"完成一件任务"。**
    给它账本面既没价值（它不会正确使用），又制造伤害（它会建账本、占掉激活位）。

    实证来源（2026-08-27 Codex 病例）：用户只发了一句话，BladeX 却收到四个 prompt
    ——另外两个是 Codex 客户端自己的附属请求（IDE 建议生成 / UI 标题生成），
    它们**原本零工具**：

        gpt-5.6-terra（建议生成）  工具 4 = bladex 4 / 自带 0
        gpt-5.6-luna （标题生成）  工具 4 = bladex 4 / 自带 0
        gpt-5.5      （用户任务）  工具 17 = bladex 4 / 自带 13

    是我们塞给它 4 个 `bladex_*` + "FIRST STEP, every turn: call
    bladex_ledger_switch"，**它才有了建账本的能力**，然后照做了。
    结果：标题生成那本账本占住激活位，用户真实任务再进来时对照 Goal 发现
    "不是同一件事"，只能新开 —— **模型判断全对，错在我们给了它不该给的东西**。

    与 `is_isolated_subcall` 同构：**结构性、零启发式**，问的都是
    "这一轮该不该有账本"。两者互补——那条看消息结构，这条看工具面。

    **为什么不用 user 文本前缀匹配**（初版方案）：脆弱。Codex 改一版模板就失效，
    且要逐个 agent 逐个功能维护；本判据与 agent 无关、不随改版失效
    （刚性原则 10：从 BladeX 侧适配，不要求 agent 带任何标记）。

    代价（全库分层抽样每 agent 120 轮，2026-08-27）：零自带工具占比
    hermes:default 0% / Pi 0% / dsh 1% / codex 4.2% / hermes:accept 6.7% /
    claude-code 15%；**逐条核过没有一条是真实用户轮次**（全是 transcript
    安全检查、转录回放、上下文压缩、标题生成），且多数已被既有 aux 规则盖住
    —— 本判据抓的是**规则表的残留**，是兜底不是替代。

    🔴 已知代价（诚实记账）：一个**极简 OpenAI 兼容客户端**（不带工具、只发文本）
    将永远拿不到账本面。实测语料里为 0，但语料只有 6 个 agent，不能当普遍结论。
    故由 `BLADEX_LEDGER_REQUIRE_TOOLS` 控制（默认开，置 0 完整回滚）。

    `bladex_*` 不计入——那是我们自己注入的，拿它当"agent 带了工具"的证据
    会让判据自我满足。
    """
    from bladex_proxy.toolface import BLADEX_TOOL_PREFIX  # noqa: PLC0415 —— 循环导入

    if not flag_enabled("BLADEX_LEDGER_REQUIRE_TOOLS"):
        return False
    for td in tools or []:
        fn = (td or {}).get("function") if isinstance(td, dict) else None
        name = str((fn or td or {}).get("name", "")) if isinstance(td, dict) else ""
        if name and not name.startswith(BLADEX_TOOL_PREFIX):
            return False
    return True


#: 账本域 aux 判据的四个子条件名（`agency_toolface_decision` 的 `aux_reason` 取值，
#: 与 `server._apply_agency_surfaces` 里的判定顺序一致；MQ-L34）。
AUX_REASON_SOURCE = "aux_source"          # is_ledgerless_auxiliary(aux_source)
AUX_REASON_SUBAGENT = "subagent"          # identity.subagent
AUX_REASON_NO_TOOLS = "no_tools"          # brings_no_tools
AUX_REASON_ISOLATED = "isolated_subcall"  # is_isolated_subcall

#: 🔴 只挡**账本族**、记忆族照注的子条件（MQ-L34，2026-09-02 Jason 拍板）。
#:
#: 四个子条件的立论全是"这一轮该不该有**账本**"（MQ-L8/L20/L21 各自 docstring），
#: 而它们合成的 `ledgerless_aux` 此前原样传给 `augment_tools(auxiliary=…)`，
#: 把两个工具族一起关掉——与 08-30「记忆族无条件注入」拍板相悖：一个不带工具的
#: 聊天类请求（08-31 hermes:accept 18 轮 `reason=aux`）连 `bladex_memory_search`
#: 都拿不到，而注入面砍到只剩硬规则后工具面是它取记忆的**唯一**通道。
#: `no_tools` / `isolated_subcall` 是结构判据（零工具 = 要一段文本；无 system 单 user
#: = 孤立子调用），它们证明的是"不该建账本"，不是"不需要记忆"。
#: `aux_source`（真·内部调用：压缩检查点 / 标题生成信封）与 `subagent` 本批**不动**
#: ——前者是 agent 自己的协议轮，后者的记忆需求没有读数，等 `aux_reason` 的
#: live 分布再议（仪器先行）。
#: 已知代价：Codex 标题/建议生成这类零工具附属请求会拿到记忆族工具，可能多一次
#: `bladex_memory_search`（成本走 MQ-P9 的 inner_loop aux 轮，可量）。
LEDGER_ONLY_AUX_REASONS: frozenset[str] = frozenset(
    {AUX_REASON_NO_TOOLS, AUX_REASON_ISOLATED})


def is_isolated_subcall(messages: list[dict]) -> bool:
    """这一轮是不是 agent 的**孤立内部子调用**（账本面 gating 专用判据，MQ-L8）。

    🔴 **为什么另开一个判据而不是扩 `auxiliary`**：`auxiliary` 已经被蒸馏与路由两个
    消费方共用过一次并出过事（2026-07-28：T3 加的 CC 信封指纹让用户配置的 GLM-5.2
    被整个旁路，修法是拆出 `DISTILL_ONLY_AUX_RULES` / `is_cheap_tier_auxiliary`）。
    账本面是**第三个**消费方，问的是"这一轮该不该有账本"——与"有没有记忆价值"
    和"能不能用便宜模型"都不是一回事。同一个旋钮控三件事，是那次事故的形状。

    **判据（结构性，零启发式）**：没有 system 消息 **且** 只有一条 user 消息。
    这是"一次无上下文的孤立调用"的形状——任何 agent 的**真实用户轮**都带着它自己的
    system prompt（我们正是靠它做指纹识别）。

    实测来源（jydesignhk 病例）：Hermes 的视觉子调用走 asyncopenai 客户端，
    `msgs=1`、`roles=['user']`、`content=[text, image_url]`、无 system ⇒ 指纹失败落
    UA 桶 `unknown-9eb9e3a9`、aux 零命中 ⇒ 账本面全开 ⇒ **一个用户请求开出 5 张账本**，
    还把注入块的 top-5 候选列表挤满噪声。10/10 轮全是这个形状（8 条带图、2 条纯文本）。

    代价评估：一个极简 OpenAI 兼容客户端的**首轮**（无 system、单条 user）会被判为
    孤立子调用而拿不到账本面；第二轮起消息数 ≥3 就不再命中。这个代价远小于
    "一个请求 5 张噪声账本"，且**不影响蒸馏与路由**（本判据不进 `auxiliary`）。
    """
    non_empty = [m for m in (messages or []) if m.get("role")]
    if any(m.get("role") == "system" for m in non_empty):
        return False
    return len(non_empty) == 1 and non_empty[0].get("role") == "user"


def classify_turn_disposition(messages: list[dict]) -> tuple[str, str]:
    """一轮在 **rebuild 视角**下的处置三分，返回 `(turn_class, aux_rule)`。

    `turn_class` ∈ `bladex_core.task_goal.TURN_CLASSES`：

        dropped   aux ∧ 规则 ∉ DISTILL_ONLY_AUX_RULES —— rebuild 整轮丢
                  （Hermes 原生委派回执 / 压缩检查点这类从头到尾的内部通信）
        scaffold  aux ∧ 规则 ∈ DISTILL_ONLY_AUX_RULES —— user 侧是信封、
                  assistant 侧照收，因而**能产 fact、能开卡**
        real      其余

    这个三分**在 rebuild 里已经存在**（`is_aux and _aux_rule not in
    DISTILL_ONLY_AUX_RULES` 那一行），G12.3 需要同一个判据，故收成一处：
    第二个消费者出现时抄一份字面量正是 MQ-S4「筛子和被测系统各算各的」的形态。
    仪器（`scripts/probe_first_turn_source.py`）也从这里读。

    🔴 判据是**重判**（`classify_auxiliary(messages)`），不读冻结的
    `turn.identity.auxiliary`——那个字段是请求时的初判，会漏 user 角色的
    委派/子任务（ADR-0018 R2 立的规矩）。
    """
    from bladex_core.task_goal import TURN_DROPPED, TURN_REAL, TURN_SCAFFOLD

    is_aux, rule = classify_auxiliary(messages)
    if not is_aux:
        return TURN_REAL, rule
    return (TURN_SCAFFOLD if rule in DISTILL_ONLY_AUX_RULES else TURN_DROPPED), rule

# system prompt 关键词（有独立 system prompt 的 auxiliary：MoA reference 等）。
# 与 _AGENT_RULES 里 auxiliary=True 的规则关键词对齐，统一走 classify_auxiliary。
_AUX_SYSTEM_KEYWORDS: list[str] = [
    "reference advisor in a mixture of agents",
    "summarization agent creating a context checkpoint",
    "task list was preserved across context compression",
    # ── Hermes 会话标题生成（2026-08-20 真实流量发现）──
    #
    # 形态：system = "You name chat sessions. Given the user's opening message,
    # write a title that lets them find this conversation again in a list."，
    # user = **用户的真实提问原文**、无 tools。
    #
    # 🔴 指纹必须落在 system 侧：user 那条是真实用户内容，拿它做判据会误伤。
    # 这也是它进 `_AUX_SYSTEM_KEYWORDS` 而不是 `_AUX_USER_PATTERNS` 的原因。
    #
    # 此前**完全没有指纹**：`aux_source='' auxiliary=False`，被当成真实用户轮进蒸馏，
    # 而它零记忆价值（内容就是"给这段对话起个标题"）。全库回扫历史仅 1 轮，量小，
    # 但会随每个新会话持续产生。
    #
    # 归**完整 aux**（不进 `DISTILL_ONLY_AUX_RULES`）：它既没有记忆价值，也确实是
    # 廉价内部调用，降档合理——与同在本表的 MoA reference / 压缩检查点同口径。
    "you name chat sessions",
    # ── Claude Code 内部子调用（2026-08-25 V-P5b live 流量发现）──
    #
    # 形态与上面 hermes 的会话标题同构：system 里有专属身份句，user 是
    # `<session>用户原话</session>` + 机器指令。**指纹同样必须落 system 侧**。
    #
    # 实测后果（不加指纹时）：`aux=False aux_source=''` ⇒ 拿到工具面与账本首步
    # 指令 ⇒ **子调用照着指令建了一本账本并抢走 session 绑定**，主任务的账本成孤儿
    # （ldg-66d0 vs ldg-2cfd 病例，V-L6d 复核）。
    "you are naming a coding session",
    "suggest what the user might naturally type next",
    # ── Codex 内部子调用（2026-08-25 V-P5b live 流量发现）──
    #
    # 形态：`fp:0238680d50a8` 那条 session 与主任务并行进来，system =
    # "You are a helpful assistant. You will be presented with a user…"，
    # 要求返回结构化 JSON（`{"title":…,"description":…}`）——即会话标题/摘要子调用。
    #
    # 未识别的后果与 CC 同构且更糟：它拿到工具面 ⇒ 先调 `bladex_ledger_switch` ⇒
    # 内循环把工具结果喂回 ⇒ 它返回的 JSON 被我们包成 message 事件发出 ⇒
    # **子调用期待的 schema 对不上**。指纹同样落 system 侧。
    "generate 0 to 3 hyperpersonalized suggestions",
    "you will be presented with a user's message and you need to generate a title",
]


#: 破折号归一（MS-2 信封普查发现，2026-08-06）。
#: 🔴 实测：`context_compaction_handoff` 的指纹写的是 ASCII 连字符
#: `[CONTEXT COMPACTION - REFERENCE ONLY]`，而真实流量发的是 em dash
#: `[CONTEXT COMPACTION — REFERENCE ONLY]`——**811 轮压缩交接从未命中**，
#: 全部被当成真实用户轮：污染归属 + 蒸馏对着一句系统提示做无用功。
#: 一个字符的差别，指纹表看起来是对的、日志里也没有任何异常，只有把真实数据
#: 按形态聚类才看得见（这正是普查流程存在的理由）。
#: 归一而不是改那一行：agent 换个 Unicode 变体就复发，治形状不治个案。
_DASH_VARIANTS = str.maketrans({"—": "-", "–": "-", "−": "-", "―": "-"})


def _normalize_dashes(text: str) -> str:
    """把各种 Unicode 破折号归一成 ASCII 连字符，供指纹匹配用。"""
    return text.translate(_DASH_VARIANTS)


def classify_auxiliary(messages: list[dict]) -> tuple[bool, str]:
    """判断一轮消息是否为 auxiliary（agent 内部通信/任务回执/子代理委派）。

    返回 (is_aux, rule_name)。rule_name 空 = 未命中。

    两条道（R2）：
      1. **末条** user 消息内容命中模板前缀（主道：Hermes 异步委派回执/网页摘要子任务
         以 user 角色进入，system prompt 同主会话 -> 只能从 user 内容判）。只看末条 =
         本轮新消息；历史中的回执不影响本轮判定（N2 修复：原扫全历史致回执之后所有轮
         误判 aux、新增真实消息漏蒸馏）。
      2. system prompt 命中 auxiliary 关键词（MoA reference 等有独立 system prompt 的）。

    纯函数，供 identity（请求时初判）与 rebuild_from_hub（重判，不信冻结的 turn.auxiliary）
    共用。旧 Turn（无 aux_source 字段）兼容：重判只看消息内容，不读 turn.auxiliary。
    """
    # 0. BladeX 自己写的内循环 aux 轮（MQ-P9，2026-09-02）：首条 system 以
    #    `INNER_LOOP_MARKER` 开头。判据落在**首条且 system**——真实 agent 请求的
    #    system prompt 不会以这串开头，而 user 消息里**谈论**这串（本仓文档被 CC
    #    塞进 user 消息）不会命中（MQ-A19 自指陷阱的形状，位置判据是它的解药）。
    #    规则名 = `INNER_LOOP_AUX_SOURCE`，不进 `DISTILL_ONLY_AUX_RULES` ⇒ rebuild 整轮丢。
    if messages:
        _m0 = messages[0]
        _c0 = _m0.get("content") if isinstance(_m0, dict) else None
        if (_m0.get("role") == "system" and isinstance(_c0, str)
                and _c0.startswith(_INNER_LOOP_MARKER)):
            return True, _INNER_LOOP_AUX_SOURCE

    # 1. 末条 user 消息模板前缀（N2 修复：只看本轮新消息，不扫全历史）
    # Hermes 委派回执（[ASYNC DELEGATION BATCH COMPLETE...]）进入会话历史后会留在
    # 后续每轮的 messages 里；扫全历史会把回执之后所有轮误判 aux，导致这些轮里
    # 新增的真实 user 消息永不蒸馏。末条 user 命中 = 本轮是 aux 回执轮；历史中的
    # 回执不影响本轮判定。
    last_user_content: str | None = None
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "") for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
        if isinstance(content, str):
            last_user_content = content  # 不断覆盖，保留最后一条 user
    if last_user_content:
        content_lower = _normalize_dashes(last_user_content.lower())
        # 🔴 **最早命中 + 必须靠近开头**（MQ-A19，2026-08-27 事故驱动）。
        #
        # 此前是"按表序第一条命中的子串"，两个毛病：
        #
        # ① **自指陷阱**：变量叫 `prefix`、注释写"模板前缀"，实现却是 `in`（任意位置）。
        #    于是**一条谈论信封的消息会被判成信封**。live 事故：Claude Code 每轮把
        #    `<system-reminder>` + CLAUDE.md 全文塞进 user 消息，而 CLAUDE.md 里
        #    ADR-0024 T3 那段话写着 "`<transcript>` 安全检查子 agent 转录一项就占
        #    439/507"——`<transcript>` 出现在第 **33917** 字符 ⇒ 113/114 轮全判 aux
        #    ⇒ 工具面零注入、账本零建、蒸馏全跳。**我们自己写的规则描述，
        #    把读这份文档的 agent 判成了内部调用。**
        #
        # ② **表序决定标签**：`task_list_preserved` 排在 `summarization_checkpoint`
        #    之前，于是"开头就是 summarization prompt、深处才有 task-list 标记"的轮次
        #    被贴成前者。改最早命中后标签自动归位。
        #
        # 阈值标定（全库 5605 条末条 user，2026-08-27）：判定命中的位置**双峰**——
        #    offset 0：913 ｜ 1–200：34 ｜ 201–2000：25 ｜ 2001–20000：4 ｜ >20000：468
        # 中间地带 29 条**全部人工核过是真信封**（`[Request interrupted by user]` /
        # `<local-command-caveat>` 之类的小前言，最深 278；那 4 条 >2000 的在改成
        # 最早命中后会由 offset=0 的 `summarization_checkpoint` 接手）。
        # >20000 那 468 条**全部是自指**（CC 读 CLAUDE.md / 我们讨论自己的规则）。
        # 取 2000 = 真信封实测上限 278 的 7 倍，自指实测下限的 1/10。
        best_rule, best_off = "", -1
        for rule_name, marker in _AUX_USER_PATTERNS:
            off = content_lower.find(_normalize_dashes(marker.lower()))
            if off < 0:
                continue
            if best_off < 0 or off < best_off:
                best_rule, best_off = rule_name, off
        if best_rule and best_off <= _ENVELOPE_MAX_OFFSET:
            return True, best_rule
        if best_rule:
            # 命中了，但埋得太深 ⇒ 这一轮是在**谈论**信封，不是信封本身。
            # 可 grep：自指陷阱再犯时要能一眼看见，而不是又靠"整个 agent 零账本"倒推。
            logger.info("aux_envelope_marker_too_deep", rule=best_rule,
                        offset=best_off, length=len(last_user_content))
        # U3：hermes 转录形态 "User: …\n\nAssistant: …" —— 结构化匹配（"User:" 开头
        # + 含 "\n\nAssistant:"），比子串前缀 "User:" 严格，避免误伤真实用户消息。
        stripped = last_user_content.lstrip()
        if stripped.startswith("User:") and "\n\nAssistant:" in stripped:
            return True, "hermes_transcript_replay"

    # 2. system prompt 关键词
    system_text = _extract_system_text(messages)
    if system_text:
        system_lower = system_text.lower()
        for kw in _AUX_SYSTEM_KEYWORDS:
            if kw.lower() in system_lower:
                return True, f"system:{kw[:40]}"

    return False, ""


# ── 被动 agent 指纹规则（G11.10：规则表已外置）──
#
# 规则定义与加载搬到 `bladex_proxy.agent_rules`：预置随包发布
# （`bladex_proxy/agent_rules.toml`），用户可在 `<部署根>/config/agent_rules.toml`
# 增补或覆盖，dashboard 认领（G11.11）写回的也是那个文件。
#
# 为什么外置：此前加一个 agent 要改这段 Python 字面量并发版本，用户无从参与——
# 而"遇到新 agent 还是麻烦"正是本组要解决的问题。
#
# `AgentFingerprintRule` 从这里 re-export，老引用（含测试）不用改。
AgentFingerprintRule = _ExternalRule


def _rules() -> list[AgentFingerprintRule]:
    """当前生效的规则表（惰性加载 + 进程内缓存）。

    🔴 不要改成模块级常量：那会让 import 本模块就去读盘。2026-08-04 踩过同型的坑
    （consolidator 模块级 `_load_env_file()` 让任何 import 都污染 os.environ）。
    """
    return load_agent_rules()


def _fingerprint_agent(
    messages: list[dict],
    tools: list[dict] | None,
    headers: dict[str, str] | None = None,
) -> tuple[str | None, str, bool]:
    """从请求内容被动识别 agent（无需 agent 配合）。

    返回 (agent_id | None, trigger_str, is_auxiliary)。
    trigger_str 记录命中的信号来源，供 AgentRegistry 审计。
    is_auxiliary: T6 — 是否为 auxiliary 调用（MoA reference 等）。

    信号优先级：
      1. system prompt 关键词匹配（最可靠，首条 system 几乎不会变）
      2. tool 签名匹配（工具名是 agent 生态特有，不跨 agent 共享）
      3. header 模式匹配（G11.10，**最后一道**，理由见下）
      4. 返回 None，交给上层同源继承 / 分桶

    **header 维排在最后**：UA 会随版本换 product token、可被用户改、可缺失
    （四条实证见 `agent_bucket` 模块 docstring）。前两道是内容派生信号，不随对方
    配置漂移，可靠性更高。header 维只是给"确实自报家门且形态稳定"的 agent 一条
    捷径——它命中不了的（如 Hermes 主路径 UA 是 SDK 默认值）本就该走前两道。

    注意：auxiliary 规则（如 MoA reference）放在规则表最前，
    优先于 base agent 规则命中——它们 system prompt 更特征鲜明。
    """
    # ── 1. system prompt 匹配 ──
    system_text = _extract_system_text(messages)
    if system_text:
        system_lower = system_text.lower()
        for rule in _rules():
            for kw in rule.system_prompt_keywords:
                if kw.lower() in system_lower:
                    logger.info("agent_fingerprint_system_match",
                                agent_id=rule.agent_id, keyword=kw,
                                auxiliary=rule.auxiliary)
                    return _resolve_profile(rule, system_text), f"system_prompt:{kw}", rule.auxiliary

    # ── 2. tool 签名匹配 ──
    tool_names = _extract_tool_names(messages, tools)
    if tool_names:
        for rule in _rules():
            if not rule.tool_signatures:
                continue
            matches = tool_names & set(rule.tool_signatures)
            if len(matches) >= rule.min_tool_match:
                logger.info("agent_fingerprint_tool_match",
                            agent_id=rule.agent_id, matched_tools=sorted(matches),
                            auxiliary=rule.auxiliary)
                return _resolve_profile(rule, system_text), f"tool:{','.join(sorted(matches))}", rule.auxiliary

    # ── 3. header 模式匹配（G11.10）──
    if headers:
        for rule in _rules():
            for hp in rule.header_patterns:
                if hp.matches(headers):
                    logger.info("agent_fingerprint_header_match",
                                agent_id=rule.agent_id, header=hp.name,
                                auxiliary=rule.auxiliary)
                    return (
                        _resolve_profile(rule, system_text),
                        f"header:{hp.name}",
                        rule.auxiliary,
                    )

    return None, "", False


def _resolve_profile(rule: AgentFingerprintRule, system_text: str) -> str:
    """从 system prompt 中提取 agent 自带的 profile 标识。

    不用 hash（太容易被动态内容影响），而是直接提取 agent 自己写的 profile 名字。
    例：Hermes 在 system prompt 里写 "Active Hermes profile: accept" → agent_id = "hermes:accept"

    无 profile 标记（如内部子请求、或 agent 不支持 profile 概念）→ 返回 base agent_id，
    由会话粘性从主请求继承完整 agent_id。
    """
    if not rule.profile_aware:
        return rule.agent_id

    if not system_text:
        return rule.agent_id

    # 尝试匹配已知 agent 的 profile 标记
    for pattern_agent, pattern in _PROFILE_PATTERNS:
        if pattern_agent != rule.agent_id:
            continue
        match = pattern.search(system_text)
        if match:
            profile_name = match.group(1).rstrip(".")
            return f"{rule.agent_id}:{profile_name}"

    # 无标记 → base agent_id（会话粘性会继承主请求的 profile 级 agent_id）
    return rule.agent_id


def resolve_subagent(agent_id: str, headers: dict[str, str] | None) -> str:
    """agent **在协议里自报**的子代理名；不是子代理返回 ""（② / MQ-A12）。

    实测来源（2026-08-26 普查）：Codex 的 guardian 子代理 12 轮全带
    `x-openai-subagent: guardian`，同时 UA / `originator` 与主路径**逐字相同**
    ——传输层完全一样，唯一的区别就是这个 header。不读它的后果已经发生过：
    12 轮落 `unknown-9d9bdd3e`、10 条 fact 写进幻影命名空间。

    产出走**既有的 profile 命名设施**（`hermes:accept` / `codex:guardian`），
    不另造概念：对下游而言它就是同一个 agent 的一个已命名侧面。

    🔴 只查**解析出的那个 agent 自己声明的** header，不做全表扫描——
    否则 A 家的 header 会给 B 家的轮次加后缀。
    """
    if not agent_id or not headers:
        return ""
    base = agent_id.split(":")[0]
    lower = {str(k).lower(): str(v) for k, v in headers.items()}
    for rule in _rules():
        if rule.agent_id != base or not rule.subagent_headers:
            continue
        for name in rule.subagent_headers:
            val = (lower.get(name) or "").strip()
            if val:
                # 值进 agent_id ⇒ 会成为记忆命名空间的一段，按 profile 同款收窄字符集
                safe = re.sub(r"[^A-Za-z0-9_.-]", "-", val)[:32]
                return safe
    return ""


#: 承载"系统指令"的角色。**`developer` 必须在内**（2026-08-19 Pi 实测）。
#:
#: 🔴 OpenAI 为 reasoning 系模型引入了 `developer` 角色取代 `system`，Pi 等较新的
#: agent 直接用它发系统提示。只认 `system` 的后果不是"少一点信息"，而是
#: **system prompt 关键词这条主力识别信号对这类 agent 完全失效**——
#: Pi 的首条消息写着 "You are an expert coding assistant operating inside pi,
#: a coding agent harness"，辨识度极高，而指纹匹配一个字都没看到。
#: 实测表现：Pi 只能靠 tool 签名分桶，规则建议器也推不出关键词。
_SYSTEM_ROLES = ("system", "developer")


def _extract_system_text(messages: list[dict]) -> str:
    """提取所有系统指令消息的文本（拼接，用于指纹匹配）。"""
    parts: list[str] = []
    for msg in messages:
        if msg.get("role") not in _SYSTEM_ROLES:
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(part.get("text", ""))
    return "\n".join(parts)


def _extract_tool_names(
    messages: list[dict],
    tools: list[dict] | None,
) -> set[str]:
    """从请求中提取所有出现的 tool name。

    来源：
      1. 请求体的 tools 字段（OpenAI function calling 定义）
      2. messages 里 assistant 的 tool_calls（历史调用记录）
      3. messages 里 role=tool 的 name 字段（工具返回）
    """
    names: set[str] = set()

    # 请求体 tools 字段
    if tools:
        for tool_def in tools:
            func = tool_def.get("function", {}) if isinstance(tool_def, dict) else {}
            name = func.get("name", "")
            if name:
                names.add(name)

    # messages 里的 tool 调用记录
    for msg in messages:
        role = msg.get("role", "")
        if role == "assistant":
            for tc in msg.get("tool_calls", []):
                func = tc.get("function", {}) if isinstance(tc, dict) else {}
                name = func.get("name", "")
                if name:
                    names.add(name)
        elif role == "tool":
            name = msg.get("name", "")
            if name:
                names.add(name)

    return names


@dataclass
class _SessionEntry:
    """缓存的已知会话条目，用于尾部接续判断。"""

    session_id: str
    # 前 N 条非 system 消息的 (role, content_hash) 列表
    prefix_fingerprint: list[tuple[str, str]]
    # 该会话上次见到时的非 system 消息总数
    msg_count: int


@dataclass
class SessionCache:
    """进程内会话缓存：记录最近见过的会话前缀，供尾部接续判断。

    不持久化——proxy 进程重启后重建。对单用户本地场景足够。
    """

    _entries: dict[str, list[_SessionEntry]] = field(default_factory=dict)

    def _key(self, user_id: str, agent_id: str) -> str:
        return f"{user_id}/{agent_id}"

    def find_match(
        self,
        user_id: str,
        agent_id: str,
        non_sys_msgs: list[dict],
    ) -> str | None:
        """查找尾部接续匹配：新消息的前缀是否与某已知会话一致。

        判断逻辑：取新请求前 _PREFIX_MATCH_COUNT 条非 system 消息的
        (role, content_hash)，与缓存中会话的前缀比对。
        若完全一致，且新请求消息数 >= 缓存中的消息数（尾部超集），判为同会话。
        """
        k = self._key(user_id, agent_id)
        entries = self._entries.get(k, [])
        if not entries or len(non_sys_msgs) < 2:
            return None

        new_prefix = _fingerprint_msgs(non_sys_msgs[:_PREFIX_MATCH_COUNT])

        for entry in reversed(entries):  # 最近匹配优先
            if (
                entry.prefix_fingerprint == new_prefix
                and len(non_sys_msgs) >= entry.msg_count
            ):
                # 尾部接续：前缀一致且消息数只增不减
                entry.msg_count = len(non_sys_msgs)
                return entry.session_id

        return None

    def remember(
        self,
        user_id: str,
        agent_id: str,
        session_id: str,
        non_sys_msgs: list[dict],
    ) -> None:
        """记住一个新会话的前缀指纹。"""
        k = self._key(user_id, agent_id)
        entries = self._entries.setdefault(k, [])
        prefix = _fingerprint_msgs(non_sys_msgs[:_PREFIX_MATCH_COUNT])
        entries.append(_SessionEntry(
            session_id=session_id,
            prefix_fingerprint=prefix,
            msg_count=len(non_sys_msgs),
        ))
        # 防止无限增长
        if len(entries) > _MAX_CACHED_SESSIONS_PER_KEY:
            entries.pop(0)

    def clear(self) -> None:
        self._entries.clear()


# 进程级单例（proxy 生命周期内复用）
_session_cache = SessionCache()


def _fingerprint_msgs(msgs: list[dict]) -> list[tuple[str, str]]:
    """取消息列表的 (role, content_hash) 指纹。"""
    result: list[tuple[str, str]] = []
    for msg in msgs:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, str):
            content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        elif isinstance(content, list):
            # 多模态：把所有 text part 拼起来 hash
            texts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    texts.append(part.get("text", ""))
            content_hash = hashlib.sha256("|".join(texts).encode()).hexdigest()[:16]
        else:
            content_hash = hashlib.sha256(str(content).encode()).hexdigest()[:16]
        result.append((role, content_hash))
    return result


def resolve_identity(
    headers: dict[str, str],
    request: ChatCompletionRequest,
    registry: Any | None = None,
) -> tuple[Identity, AgentSource]:
    """从请求头 + 请求体解析 Identity。

    分层：
      user_id: 见下（ADR-0021 §2.3，2026-08-16 修订）
      agent_id: X-Agent-ID > 被动指纹(system prompt + tools) > 同源继承 > `unknown-<hash8>`
      session_id: X-Session-ID / `x-{vendor}-session-id`（同级，G11.3）
                  > 尾部接续 > 前缀 hash > user+time_window

    **agent_id 末位不再是裸 "unknown"（G11.9，2026-08-18）**：所有未识别 agent 共用
    一个 `unknown` 等于把互不相干的客户端合进同一个记忆命名空间——是换了名字的
    误合并。改为按分桶键各自成桶 `unknown-<hash8>`，宁分勿合，用户在 dashboard
    认领后走 `AGENT_CLAIM` 收敛（可回滚），历史数据经重建重放改归属。

    **User-Agent 不再用于命名**（同上）：它只作分桶信号之一。理由见
    `agent_bucket` 模块 docstring。
      turn_index: 用户轮 = role==user 消息数（`ledger_runtime.user_turn_index`，E0.2；
                  修前 len(messages)//2 是请求数，工具循环里每个调用都算一轮）

    **user_id 三分支**：

      无 api_key                → `ANONYMOUS_USER_ID`（任何模式都不并入，是 auth 探测器）
      有 identity.toml（企业）  → key -> principal -> teams -> org，解析出可见集合 + 敏感度；
                                  多 key 归一同一 principal
      无 identity.toml（个人）  → `LOCAL_USER_ID` 常量，**与凭证解耦**

    最后一支是 2026-08-16 的修订：原为 `user_id = key hash8`，即把凭证当身份，
    换 key 就静默劈开记忆（实测 live 出现 4 个 user_id 实为一人）。两个常量的
    docstring 写明了各自的语义与改动代价。
    """
    # ── user_id（ADR-0021 §2.3，2026-08-16 修订）──
    api_key = _extract_api_key(headers)
    fallback_uid = _hash_short(api_key) if api_key else ANONYMOUS_USER_ID
    user_id = fallback_uid
    resolved = None
    if registry is not None and not getattr(registry, "empty", True):
        # 企业模式（有 identity.toml）：key -> principal。本支未改动。
        resolved = registry.resolve(api_key, fallback_user_id=fallback_uid)
        if resolved is not None:
            user_id = resolved.principal_id
    elif api_key:
        # 🔴 个人模式（无 identity.toml）= 单一本地身份，与凭证完全解耦。
        #
        # 原实现在这里落 `key hash8`，即**把凭证当身份**——换一把 key / 换台设备
        # 就是换个人，记忆静默劈成两半。2026-08-16 审计实测：live 出现 4 个 user_id
        # 实为同一人，一件泰山啤酒尽调被切开，而 P2 散点召回按 user_id 过滤 →
        # 两半互相召不回。该分裂自 ADR-0021 上线起潜伏，无任何告警。
        #
        # 判据是失败模式不对称：原默认的失败是"静默分裂 + 已兑现"，
        # 新默认的失败需要"把自部署实例给别人用且不配 identity.toml"这种主动误配置
        # （且 ADR-0027「非回环 + auth 关」强警告已承接该类）。
        # 完整论证与实证见 ADR-0021 §2.3 修订记录 +
        # docs/reviews/adr0021-migration-audit-20260816.md §5.4。
        user_id = LOCAL_USER_ID
    # 无 api_key 时落 ANONYMOUS_USER_ID —— 两种模式都不并入（见常量 docstring）。
    #
    # ── G11.3：`x-{vendor}-user-id` **有意不接进 user_id** ──
    # 卡面把它与 session-id 并列写作"显式 user 信号"，落地时否掉，两条理由：
    #
    # 1. **租户边界**。user_id 是 P2 可见性过滤的判据（scope / visibility 都按它
    #    取），而 header 未经任何认证 —— 采纳它等于"谁都能改一个 header 去读别人
    #    的记忆"。这属刚性原则里的**硬约束（安全边界，fail-closed 不可绕）**，
    #    不是可以权衡的策略项；企业模式下它还会直接盖掉 identity.toml 解析出的
    #    principal，那是把 ADR-0021 的凭证层整个绕开。
    # 2. **与 2026-08-16 那次修订同形**。个人模式的 user_id 已刻意与凭证解耦
    #    （LOCAL_USER_ID），接一个外部可变的值回来当身份，就是把 `key hash8` 的
    #    静默分裂换个来源重演一遍（当时实测 4 个 user_id 实为一人）。
    #
    # session-id 没有这个问题：它只切分同一个 user 内部的会话粒度，切错了是会话
    # 边界不准（可自愈），不跨越任何隔离边界。**两个后缀的风险等级不同，卡面把
    # 它们并列写是笼统了。**
    #
    # 值本身没丢：`redact_headers` 快照随 Turn 入 Memory Hub，dashboard 认领
    # （G11.11）看得到。按 ADR-0026 铁律三——没有署名消费者的字段不生产——到此为止。
    # 守卫：`test_g113_vendor_session.py::test_vendor_user_id_never_touches_user_id`

    # ── agent_id（四层：显式 > 被动指纹 > 会话粘性 > User-Agent > unknown）──
    explicit_agent = headers.get("x-agent-id") or headers.get("X-Agent-ID")
    agent_source: AgentSource
    agent_trigger = ""

    # G11.9：分桶键先算——粘性继承要用它当同源判据，未识别时也要用它命名。
    # 只读 header 名 + UA product token（去版本）+ 必要时的 tool 集合，不读易变的值。
    bucket = bucket_unknown_agent(
        headers.get("user-agent", ""),
        headers,
        _extract_tool_names(request.messages, request.tools),
    )
    redacted_headers = redact_headers(headers)

    is_auxiliary = False
    if explicit_agent:
        agent_id = explicit_agent
        agent_source = AgentSource.EXPLICIT
        agent_trigger = "header:X-Agent-ID"
    else:
        # 被动指纹：从请求内容识别（system prompt + tool 签名）
        fingerprinted, fp_trigger, is_auxiliary = _fingerprint_agent(
            request.messages, request.tools, headers
        )
        if fingerprinted:
            agent_id = fingerprinted
            agent_source = AgentSource.FINGERPRINT
            agent_trigger = fp_trigger
        else:
            is_auxiliary = False
            # 同源继承（G11.5）：同 user **且同分桶键**最近识别过的 agent。
            # 加同桶判据前，本分支会把任何陌生客户端粘成"最后说话的那个 agent"
            # ——2026-08-18 dsh 被署名成 hermes:default 即由此而来。
            # 同源判据用 origin_key（传输身份，不含 tool 签名）——子请求的工具集
            # 常与父请求不同，用含工具的 bucket_id 会匹配不上（实测被剧本逼出）。
            sticky = _agent_registry.lookup_sticky(user_id, bucket.origin_key)
            if sticky:
                agent_id = sticky
                agent_source = AgentSource.STICKY
                agent_trigger = "sticky"
            else:
                # G11.9：不再共用一个 `unknown`——那是"换了名字的误合并"。
                # 每个未识别客户端按分桶键各自成桶，宁分勿合；用户在 dashboard
                # 认领后走 AGENT_CLAIM 收敛（可回滚），历史数据经重建重放改归属。
                agent_id = bucket.bucket_id
                agent_source = AgentSource.FALLBACK
                agent_trigger = f"bucket:{bucket.basis}"
                # ③ 漂移检测（2026-08-26）：这到底是"新客户端"还是"老朋友升级后
                # 内容维失效"？三条连续性证据，其中**时序接替 vs 并存**是区分
                # "升级"与"子代理"的那条——搞反了两边都错（合并子代理=误合并，
                # 给升级后的自己起新名字=身份分裂）。输出证据，不输出结论。
                _drift: dict = {}
                try:
                    from bladex_proxy.agent_drift import detect, suggest_claim
                    _ev = detect(
                        basis=bucket.basis,
                        known_agent_ids=_agent_registry.known_agent_ids(),
                        last_seen=_agent_registry.last_seen_by_agent(),
                        bucket_first_seen=time.time(),
                        # 🔴 会话接续证据在这里**取不到**：`non_sys_msgs` 要到
                        # 下面的 session 解析段才算出来，而 agent 解析在它之前。
                        # 不为了凑一条证据把两段的顺序对调——那会让 session 解析
                        # 依赖 agent_id 之外的东西，耦合更深。
                        # 三条证据里这条留空 ⇒ 单靠传输重叠达不到 ≥2 的建议门槛，
                        # 需要"接替"一起成立，正好是保守方向（宁可不建议，不可误建议）。
                        session_match="",
                    )
                    if _ev.looks_like:
                        _drift = _ev.as_dict()
                        _sug = suggest_claim(bucket.bucket_id, _ev, redacted_headers)
                        if _sug is not None:
                            _drift["suggestion"] = _sug.as_dict()
                except Exception as e:  # noqa: BLE001 —— 检测失败不该挡住这一轮
                    logger.debug("agent_drift_detect_failed", error=str(e))
                if _agent_registry.note_unrecognized(
                    user_id, bucket.bucket_id, bucket.basis, redacted_headers,
                    drift=_drift or None,
                ):
                    # G11.6：首次出现才 warning，并带上完整脱敏 header 快照
                    # ——这就是"待认领登记"，dashboard 认领界面消费它。
                    if _drift.get("looks_like"):
                        # 🔴 改口：不是"未识别 agent"，是"已知 agent 的内容维失效"。
                        # 泛泛的告警让 Codex 那次潜伏了一整天（12 轮 / 10 条 fact）。
                        logger.warning(
                            "agent_signal_drift",
                            user_id=user_id,
                            bucket_id=bucket.bucket_id,
                            basis=bucket.basis,
                            looks_like=_drift["looks_like"],
                            evidence=_drift,
                            hint="known agent's content signals stopped matching "
                                 "(usually a version upgrade), not a new client; "
                                 "claim it to merge, or add A/B-tier header rules",
                        )
                    else:
                        logger.warning(
                            "agent_unrecognized",
                            user_id=user_id,
                            bucket_id=bucket.bucket_id,
                            basis=bucket.basis,
                            headers=redacted_headers,
                            hint="未识别 agent；dashboard 认领后写入规则库并改归属历史数据",
                        )
                else:
                    logger.debug(
                        "agent_unrecognized_repeat",
                        user_id=user_id, bucket_id=bucket.bucket_id,
                    )

    # ── profile 级区分 ──
    # profile_aware 的 agent（如 hermes）已在 _resolve_profile 中自动区分配置文件。
    # X-Agent-Profile header 作为可选的显式覆盖（优先级最高）。
    explicit_profile = headers.get("x-agent-profile") or headers.get("X-Agent-Profile")
    # G11.9: 未识别 agent 现在叫 `unknown-<hash8>`（不再是裸 "unknown"），
    # 判据同步改成前缀匹配——否则给未识别客户端拼 profile 后缀会污染桶 id。
    if explicit_profile and not agent_id.startswith("unknown"):
        # 显式 profile 覆盖：如果 agent_id 已有 profile 后缀，替换它
        base = agent_id.split(":")[0] if ":" in agent_id else agent_id
        agent_id = f"{base}:{explicit_profile}"
        agent_trigger += f"+profile:{explicit_profile}"

    # ② 子代理（2026-08-26）：agent 自报 `x-openai-subagent: guardian` 之类。
    # 排在 profile 之后——两者共用 `base:suffix` 位，显式 profile 是用户意图、优先。
    is_subagent = False
    if not agent_id.startswith("unknown") and not explicit_profile:
        _sub = resolve_subagent(agent_id, headers)
        if _sub:
            is_subagent = True
            base = agent_id.split(":")[0]
            agent_id = f"{base}:{_sub}"
            agent_trigger += f"+subagent:{_sub}"

    # 注册识别（供后续子请求的同源继承查询）。**带上分桶键**——G11.5 的同源判据
    # 就是它，不带等于退回"谁最后说话下一个陌生人就是谁"的旧行为。
    #
    # STICKY 也 register：原因是刷新时间戳（旧的 120s 窗口时代），窗口已在 ADR-0021
    # 修正中移除、G11.5 改用同源判据，但继续 register 仍有意义——它让"最近一次同桶
    # 记录"保持在列表尾部附近，减少 lookup 的回溯深度。
    #
    # AgentSource.USER_AGENT 保留在这个集合里：live 代码不再产出该来源（G11.9 后
    # UA 只做分桶不做命名），但**历史 Turn 里存着这个枚举值**，重放时仍会构造出
    # 带该来源的 Identity。删枚举 = 历史数据 Pydantic 校验失败。
    #
    # 🔴 FALLBACK（未识别桶）**有意不注册**：注册了它就会成为后续同 origin 请求的
    # 继承目标，而"传输身份相同的两个陌生 agent"恰恰是我们分不开的那一类——
    # 让它们各自按 tool 签名成桶（宁分勿合），好过合进先来的那个桶（误合并）。
    # 代价：未识别 agent 的子请求（工具集不同）会另开一个桶，认领时一并合并即可。
    if agent_source in (AgentSource.EXPLICIT, AgentSource.FINGERPRINT, AgentSource.USER_AGENT, AgentSource.STICKY):
        _agent_registry.register(user_id, agent_id, agent_trigger, bucket.origin_key)

    # ── 非系统消息（用于指纹和尾部接续）──
    non_sys_msgs = _extract_non_system_msgs(request.messages)

    # ── session_id ──
    # 两个来源同级（G11.3）：我方约定的 `X-Session-ID`，与厂商前缀
    # `x-{vendor}-session-id`。后者是刚性原则 10 的直接兑现——dsh 本来就在发
    # `x-deepseek-harness-session-id`，此前被整条忽略，于是它的会话只能退到
    # time_window 去猜（实测 `session_id_source=time_window`）。
    #
    # 我方 header 优先只为**平局时确定**，不代表更可信：两个都在时得有个定序，
    # 挑我们自己文档里那个。
    session_id_header = ""
    explicit_session = headers.get("x-session-id") or headers.get("X-Session-ID")
    if explicit_session:
        session_id_header = "x-session-id"
    else:
        vendor_session_ids = extract_vendor_ids(headers, VENDOR_SESSION_SUFFIXES)
        if vendor_session_ids:
            if len(vendor_session_ids) > 1:
                # 多家同时发。取排序首位（与 header 顺序无关），但必须可 grep：
                # 静默挑一个 = 以后没人能解释"这轮的 session 为什么来自它"。
                logger.warning(
                    "session_header_multi_vendor",
                    user_id=user_id,
                    agent_id=agent_id,
                    chosen=vendor_session_ids[0][1],
                    candidates=[name for _v, name, _val in vendor_session_ids],
                )
            _vendor, session_id_header, explicit_session = vendor_session_ids[0]

    if explicit_session:
        explicit_session, reject_reason = _screen_external_session_id(explicit_session)
        if reject_reason:
            # 退回推断（尾部接续 / 前缀指纹 / 时间窗），不是报错——外部 header 不合规
            # 不该让这一轮失败。但**必须留可 grep 的告警**，否则就是静默旁路。
            logger.warning(
                "session_header_rejected",
                user_id=user_id,
                agent_id=agent_id,
                header=session_id_header,
                reason=reject_reason,
            )
            explicit_session = ""
            session_id_header = ""

    # T4(ADR-0018 §4.2 R7): 记录 session_id 来源，tw: 会话跳过 prefix_changed/CAP 冲突检测
    session_id_source = SessionIdSource.TIME_WINDOW
    if explicit_session:
        session_id = explicit_session
        session_id_source = SessionIdSource.EXPLICIT
        _session_cache.remember(user_id, agent_id, session_id, non_sys_msgs)
    else:
        # 先试尾部接续（最皮实）
        matched = _session_cache.find_match(user_id, agent_id, non_sys_msgs)
        if matched:
            session_id = matched
            session_id_source = SessionIdSource.TAIL_CONTINUATION
            # MQ-S9：接续轮续命时间窗（长会话全程走接续，不续命会被误判超窗）
            _fp_epochs.touch(user_id, agent_id, session_id)
            logger.info("session_tail_match", user_id=user_id, agent_id=agent_id,
                       session_id=session_id)
        else:
            # 退到前缀指纹（跳过 system 消息）
            session_id = _infer_session_from_prefix(non_sys_msgs, user_id, agent_id)
            if session_id.startswith("fp:"):
                # MQ-S9 / G12.2-pre：同指纹超窗 → 纪元后缀，终结跨天撞桶
                session_id = _fp_epochs.apply(user_id, agent_id, session_id)
                session_id_source = SessionIdSource.FINGERPRINT
                _session_cache.remember(user_id, agent_id, session_id, non_sys_msgs)
            else:
                session_id_source = SessionIdSource.TIME_WINDOW

    # ── turn_index（元数据，不再当 Memory Hub 主键）──
    # E0.2（MQ-L48）：单位 = 用户轮，单一实现点在 bladex_core.ledger_runtime。
    from bladex_core.ledger_runtime import user_turn_index
    turn_index = user_turn_index(request.messages)

    # T2(ADR-0018 R2): classify_auxiliary 补 user 消息模板前缀这道
    # （_fingerprint_agent 只看 system prompt，漏掉以 user 角色进入的委派/子任务）。
    # 指纹 auxiliary 与 classify_auxiliary 取或；aux_source 记命中规则名供审计 + rebuild 重判。
    classified_aux, aux_rule = classify_auxiliary(request.messages)
    if classified_aux:
        is_auxiliary = True
        aux_source = aux_rule
    elif is_auxiliary:
        aux_source = f"fp:{agent_trigger}" if agent_trigger else "fp"
    else:
        aux_source = ""

    identity = Identity(
        user_id=user_id,
        agent_id=agent_id,
        session_id=session_id,
        turn_index=turn_index,
        auxiliary=is_auxiliary,
        aux_source=aux_source,
        session_id_source=session_id_source,
        subagent=is_subagent,
        visibility=list(resolved.visibility) if resolved is not None else [],
        principal_sensitivity=resolved.principal_sensitivity if resolved is not None else "",
        team_sensitivities=list(resolved.team_sensitivities) if resolved is not None else [],
        team_ids=[s[5:] for s in resolved.visibility if s.startswith("team:")] if resolved is not None else [],
        # G11.1 / MQ-A5：脱敏 header 快照随 Turn 入 Memory Hub。
        # 署名消费者（ADR-0026 铁律三）：dashboard 认领界面（G11.11）展示"这个桶凭
        # 什么分出来的"、未识别登记（G11.6）、人工排查新 agent 的接入形态。
        # 留存而非事前标定——本机没有那么多 agent 可测，各家 header 又随版本漂移。
        request_headers=redacted_headers,
        agent_bucket_basis=bucket.basis,
    )

    if not api_key:
        logger.warning("identity_no_api_key", agent_id=agent_id)
    if session_id.startswith("tw:"):
        logger.info("identity_time_window_fallback", user_id=user_id, agent_id=agent_id)

    logger.info(
        "identity_resolved",
        user_id=user_id,
        agent_id=agent_id,
        session_id=session_id,
        turn_index=turn_index,
        agent_source=agent_source.value,
        agent_trigger=agent_trigger,
        auxiliary=is_auxiliary,
        aux_source=aux_source,
        session_id_source=session_id_source.value,
        # G11.3：EXPLICIT 这一档现在有两种来源（我方 header / 厂商前缀）。
        # 枚举有意不拆（卡面判据就是 `session_id_source=explicit`，且它是 Turn
        # schema 里的持久值），来源改记在日志字段上——排查时要能答出"哪个 header"。
        session_id_header=session_id_header,
    )
    return identity, agent_source


#: 外部 header 提供的 session id 的留存上限。
#:
#: 它整条进 Memory Hub key（``principal/agent/session/entry_id``），所以约束不是
#: "够用就行"而是"不能破坏 key"。200 与 `redact_headers` 的单值上限同档。
_MAX_EXTERNAL_SESSION_ID_CHARS = 200


def _screen_external_session_id(value: str) -> tuple[str, str]:
    """校验外部 header 给的 session id（G11.3）。

    为什么必须校验：``session_id`` 会**原样**拼进 Memory Hub key
    （``principal/agent/session/entry_id``）。值里带 ``/`` 就多切出一段，
    `scan_agent_ids`、`session_prefix()` 这些按段解析的读侧全部错位。

    在此之前 ``X-Session-ID`` 也没有校验；本函数**同时管住两条路**，不给新开的
    厂商通配路径单独立一套标准（两套标准迟早漂）。对存量零影响：live Memory
    Hub 4372 条 turn key **全部恰好三个 ``/``**，即今天没有任何 session_id
    含斜杠——校验不会改变任何已发生的分组。

    :param value: header 原值。
    :returns: ``(清洗后的值, 拒绝原因)``；通过时原因为空串，拒绝时值为空串。
        拒绝原因是**闭集** ``empty`` / ``contains_slash`` / ``control_chars`` /
        ``too_long``，日志按它聚合。
    """
    cleaned = value.strip()
    if not cleaned:
        return "", "empty"
    if "/" in cleaned:
        return "", "contains_slash"
    if any(ch.isspace() or ord(ch) < 0x20 or ord(ch) == 0x7F for ch in cleaned):
        # 换行会把一行结构化日志劈成两行；空白与控制字符在 key 里也不可读。
        return "", "control_chars"
    if len(cleaned) > _MAX_EXTERNAL_SESSION_ID_CHARS:
        return "", "too_long"
    return cleaned, ""


def _extract_non_system_msgs(messages: list[dict]) -> list[dict]:
    """提取非 system 消息（跳过首条及所有 system 消息）。

    T2: 首条 system 常含动态时间戳/环境信息，纳入指纹会导致逐轮误判为新会话。
    """
    return [msg for msg in messages if msg.get("role") != "system"]


def _extract_api_key(headers: dict[str, str]) -> str | None:
    """从 Authorization header 提取 API Key（Bearer xxx）。"""
    auth = headers.get("authorization") or headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return None


def _hash_short(value: str) -> str:
    """脱敏 hash：取 SHA256 前 8 位。"""
    return hashlib.sha256(value.encode()).hexdigest()[:8]


# `_guess_agent_from_ua` 已于 2026-08-18（G11.9）删除，**不要加回来**。
#
# 它做的是"用 UA 给 agent 命名"，而 UA 不能当身份权威来源——四条实证见
# `agent_bucket` 模块 docstring（UA 会随版本换 product token、用户可配、可缺失、
# 一个 agent 多个 UA 且主路径上不是自己的名字）。把 UA 当身份 = 与 2026-08-16
# 修掉的 `user_id = key hash8` 同形。
#
# 而且它当时已经是**死代码**：排在会话粘性之后，而粘性在 user_id 变成常量后恒命中。
# UA 的正确位置是 `agent_bucket.bucket_unknown_agent()` 里的分桶信号（不命名）。


def _infer_session_from_prefix(
    non_sys_msgs: list[dict],
    user_id: str,
    agent_id: str,
) -> str:
    """从前缀连续性推断 session_id。

    T2: 只取非 system 消息的前两条做指纹（跳过动态 system 消息）。
    若消息太少（<2 条），退到时间窗口分组。
    """
    if len(non_sys_msgs) >= 2:
        prefix_parts = []
        for msg in non_sys_msgs[:2]:
            content = msg.get("content", "")
            if isinstance(content, str):
                prefix_parts.append(content[:100])
            elif isinstance(content, list):
                texts = [
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                ]
                prefix_parts.append(" ".join(texts)[:100])
        if prefix_parts:
            fingerprint = hashlib.sha256(
                "|".join(prefix_parts).encode()
            ).hexdigest()[:12]
            return f"fp:{fingerprint}"

    # 兜底：API Key + 时间窗口
    window = int(time.time()) // _TIME_WINDOW_SECONDS
    return f"tw:{user_id}:{agent_id}:{window}"
