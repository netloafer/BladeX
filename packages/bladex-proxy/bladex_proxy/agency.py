"""AgencyRuntime —— 工具面/拦截协议/账本的运行时装配与 server 接线面（批二启用卡 V-P5a）。

server.py 只拿到三个薄调用点（inbound 准备 / tools 增补 / 响应处置），全部先查
`modules.module_enabled`——**四个模块默认关，本文件在默认形态下零行为差异**
（三红线之 3；关着时每个入口第一行短路返回）。

# 组成

- ToolFace 真实 handler：memory_search 接 MemoryIndex（结果过 exposure 门，
  ADR-0032 §3.2-7）；ledger_* 接账本池 + ActivationTable，写路径全部落 Hub
  管理事件（池 = 事件重放的投影，重启 `_load_pool` 重建）。
- SpliceLedger + strip/splice：入站先剥回流（D6/D7）再拼接（V-P4）。
- 内循环：`run_inner_loop`，call_llm 由 server 侧以当轮路由偏应用注入。

# 🔴 接线纪律

- Hub 是唯一真相：账本事件经 `append_admin_event`（`AdminEvent.matter_id` 留空，
  MQ-A7 护栏）；拼接台账经 `Turn.splice_records` 入 Hub（V-P4 已落地）。
  内循环轮次经 `loop_ledger`（`bladex_proxy.loopledger.InnerLoopLedger`）缓冲、
  `server._enqueue_turn` 随主轮 drain 后以 aux 轮入 Hub（MQ-P9，2026-09-02 落地；
  `SpliceRecord` 仍只在**混合路径**产生，两者分工：拼接台账 vs 支出凭证）。
  数"模型调了多少次 bladex 工具"读 `Turn.tool_events`（拦截前抽取，两条路都在；
  MQ-N8）——aux 轮自身的 `tool_events` 是第 2 轮起的调用，尺子按 `auxiliary` 分档。
- 拦截决不制造空响应/悬空调用（hermes 重试指纹 / D2 报错形态）。
- 每请求 `RecentRequests` 去重：codex 超时整请求重发 ≥6 次实测——重发不得
  重触发内循环（返回上次结果的合成文本，宁可重复内容不重复花钱）。
"""

from __future__ import annotations

import asyncio
import re
import time
from collections import Counter
from typing import Any

import structlog
from bladex_core.ledger import (
    ACTOR_MODEL,
    EVENT_LEDGER_CREATE,
    EVENT_LEDGER_SWITCH,
    EVENT_LEDGER_UPDATE,
    Ledger,
    LedgerEntry,
    LedgerError,
    add_entry,
    children_of,
    iso_ms,
    new_ledger,
    render_ledger_md,
    replay_ledger_events,
)
from bladex_core.ledger_runtime import (
    ActivationTable,
    activation_scope,
    apply_tool_update,
    bind_matter,
    candidate_units,
    ledger_anchor_matter_id,
    relevant_ledgers,
    staleness_marker,
)
from bladex_core.flags import flag_number
from bladex_core.sensitivity import exposure_allows

from bladex_proxy.innerloop import InnerLoopResult, run_inner_loop
from bladex_proxy.loopledger import InnerLoopLedger
from bladex_proxy.interception import (
    MODE_MIXED,
    MODE_NONE,
    MODE_PURE,
    classify_message,
    strip_bladex_calls,
    ensure_call_ids,
)
from bladex_proxy.models import AdminEventType
from bladex_proxy.modules import module_enabled
from bladex_proxy.splice import (
    RecentRequests,
    SpliceLedger,
    SpliceRecord,
    anchor_key,
    request_fingerprint,
    splice_into_messages,
    strip_inbound_echoes,
)
from bladex_proxy.toolface import (
    NO_TOOLFACE_AGENT_BASES,
    ToolFace,
    inject_tools,
    ledger_tools_allowed,
)

logger = structlog.get_logger()


#: 账本注入块标记（入站剥离认它——CC 会把注入历史回流，D6 同理，splice 侧已covered）。
LEDGER_BLOCK_OPEN = "<bladex-ledger>"
LEDGER_BLOCK_CLOSE = "</bladex-ledger>"

#: V-P6：BladeX 自我介绍块（稳定层，ADR-0032 §3.1）。
ABOUT_BLOCK_OPEN = "<bladex-about>"
ABOUT_BLOCK_CLOSE = "</bladex-about>"

#: 全部注入标记的**单一清单**。
#:
#: 🔴 每个标记都必须在 `bladex_core.envelope._ENVELOPE_PATTERNS` 里有剥离落点，
#: 否则我们注入的内容会被 agent 回传、再被自己蒸成 fact（自反馈闭环，
#: ADR-0025 §6 担心过、而它**已经在真实数据里发生**：24 条 user 消息回带
#: `<bladex-memory>`）。2026-08-28 普查：三个标记只有一个有落点——
#: `<bladex-ledger>`（每轮注、体量最大）与 `<bladex-context-summary>` 全裸。
#:
#: 这与 MQ-S46 是**同一个交接缺陷**：生产方和处置方之间没有对账。故收成一张表，
#: 由 `test_injection_markers_have_strip_landings` 机制化守住——
#: 加标记不加落点直接红，不给"以后再补"留缝。
INJECTION_MARKERS: tuple[str, ...] = (
    "<bladex-memory>",
    LEDGER_BLOCK_OPEN,
    "<bladex-context-summary>",
    ABOUT_BLOCK_OPEN,
)

#: 首步指令（Jason 2026-08-25 拍板：goal 对照是模型每轮的第一动作——没有这一步，
#: 账本就没有用。指令随激活账本一起注入动态层；英文外壳 ADR-0027 §5.1）。
#: 🔴 2026-08-30 改写（Jason live 反馈，MQ-L29 第一案）：**"删"要与"写"绑成
#: 一个动作**，不能是独立的最后一句催促。
#:
#: 原文末句是 "Keep Next current — drop steps you have finished." —— 三个毛病：
#:   ① 它是**独立的最后一句**，最弱的位置；模型执行到"写 verified"就觉得完事了；
#:   ② **只说了 Next，没说 Open**，而 live 上 Open 同样留着；
#:   ③ 最要害：写 verified 与删对应条目**在语义上是同一件事的两面**
#:      （"这条从待办变成了已确认"），却被写成两句分开的话。
#: 改法不是加一句更强的催促，而是给一条**可自查的规则**：
#: "An entry that is now in Verified does not belong in Next or Open."
#: 模型每轮对照 Goal 时可以顺手对照它。
#:
#: 🔴 同一条规则**同时写进 `bladex_ledger_update` 的 description**（Jason：
#: "避免不一致，很多时候都是更新时候的遗忘"）—— 首步指令是每轮的提醒，
#: 工具描述是**调用现场**的契约，两处都在才覆盖得住。一致性由
#: `test_ledger_cleanup_rule_stated_in_both_places` 机械守住。
#: 本条只解 MQ-L29 的**段内条目卫生**那一半；"完结判定"（谁来 close 账本）
#: 仍归 0.2.0（Jason 同日确认）。
_FIRST_STEP_WITH_LEDGER = (
    "FIRST STEP, every turn: compare the user's request with the Goal above. "
    "If it is a DIFFERENT task, call bladex_ledger_switch (ledger_id='' creates a "
    "new ledger; give a short title). If it matches, proceed — and record real "
    "progress with bladex_ledger_update: add what you established to verified "
    "(with a ref) AND remove the next/open entry it resolves, in the same turn. "
    "An entry that is now in Verified does not belong in Next or Open. "
    "Put what is still unresolved in open, and the concrete steps ahead in next. "
    "Prefer switching to a listed matching ledger over creating a new one."
)
#: 子任务条目的标签（父子两侧都用同一个，便于 grep 与后续统计回写率）。
SUBTASK_TAG = "[sub-task]"

#: 「这段 user 文本像是机器写的」的形状——**只用于告警，不用于任何判定**。
#: 2026-08-27 Codex 病例的两条实测形态各贡献一条；`You are …` 那条覆盖面最广
#: （客户端把子任务的 system prompt 塞进 user 消息，是各家通用做法）。
_MACHINE_TEXT_MARKS: tuple[tuple[str, str], ...] = (
    ("assistant_persona", "you are a "),      # "You are a helpful assistant…"
    ("md_overview", "# overview"),
    ("agent_persona", "you are an agent"),
)


def _machine_text_mark(text: str) -> str:
    """user 文本疑似机器模板 ⇒ 返回形态名；否则 ""。**只在开头 200 字符内看**。

    位置门与 MQ-A19 同理：正文里**提到** "you are a helpful assistant"
    的真实用户消息不该被标记（这正是那次自指陷阱的教训）。
    """
    head = (text or "")[:200].lower()
    for name, mark in _MACHINE_TEXT_MARKS:
        if mark in head:
            return name
    return ""

_FIRST_STEP_NO_LEDGER = (
    "No active task ledger for this session. FIRST STEP: if one of the ledgers "
    "listed below already covers this request, call bladex_ledger_switch with its "
    "id to continue it. Otherwise call bladex_ledger_switch with ledger_id='', a "
    "short title, and optional initial core/open/next entries (following the "
    "template below). The Goal is set from the user's own words — you do not write it. "
    "Prefer switching to a listed matching ledger over creating a new one."
)

#: MQ-L38 相关性段标题（2026-09-04）。与 "Other recent ledgers" 并列：那段答"最近在干什么"，
#: 这段答"以前干过这件事没有"。无命中整段不出现（块逐字等于修前）。
_MATCHING_LEDGERS_HEADER = (
    "Ledgers that look like the SAME task as the current request (read it with "
    "bladex_ledger_read, then bladex_ledger_switch to it - do NOT create a new ledger "
    "if one matches):"
)
#: 建新本时若本轮候选里有 ≥ 此分的项 ⇒ `ledger_created_despite_match` 告警（重复建本的直接计数）。
_DESPITE_MATCH_SCORE = 0.5


def load_ledger_template(path: str = "") -> str:
    """读账本模板文件。**唯一来源 = `config/ledger-template.md`**
    （`BLADEX_LEDGER_TEMPLATE` 可指别处）；缺失/空 → 结构性兜底并告警可 grep。

    🔴 Jason 2026-08-25 拍板两条：①不许事出多头（core 侧不再存第二份模板正文）；
    ②**启动时预加载一次**，不在每个请求里读文件——`AgencyRuntime` 构建时取值
    存 `self.template`，改模板重启生效（与其它配置同语义）。
    """
    import os
    p = (path or os.environ.get("BLADEX_LEDGER_TEMPLATE", "").strip())
    if not p:
        # 🔴 **按部署根解析，不用相对路径**（2026-08-25 live 回归：首版写
        # `config/ledger-template.md` 相对 cwd，proxy 的工作目录不一定是仓库根 ⇒
        # 每次启动都静默回落到结构性兜底，那份精心写的模板从未被注入过。
        # 这正是 deployment.find_root 存在的理由——ADR-0027 §3 同型问题的既有解）。
        from bladex_proxy import deployment
        root = deployment.find_root() or os.getcwd()
        p = os.path.join(root, "config", "ledger-template.md")
    try:
        with open(p, encoding="utf-8") as f:
            text = f.read()
        if text.strip():
            return text
        logger.warning("ledger_template_empty", path=p)
    except OSError as e:
        logger.warning("ledger_template_unreadable", path=p, error=str(e))
    from bladex_core.ledger import FALLBACK_TEMPLATE_MD
    return FALLBACK_TEMPLATE_MD


#: V-P6：自我介绍的三份来源，按此顺序拼接。缺文件跳过并告警（不阻断）。
#: `SKILLS.md` 现为占位（"No BladeX skills are available yet."）——**照注**：
#: 明确告诉模型"没有 skills"，好过让它去猜有没有。真有 skills 时换内容即可。
_SYSTEM_NOTE_FILES: tuple[str, ...] = ("AGENT.md", "TOOLS.md", "SKILLS.md")


def load_system_notes(dir_path: str = "") -> str:
    """读 `config/system/{AGENT,TOOLS,SKILLS}.md` 并拼成自我介绍正文（V-P6）。

    ## 来源与路径

    **唯一来源 = `<部署根>/config/system/`**（`BLADEX_SYSTEM_NOTES_DIR` 可指别处）。
    🔴 按**部署根**解析，不用相对 cwd 的路径——`load_ledger_template` 的
    docstring 记着这个教训：首版写相对路径，proxy 的工作目录不一定是仓库根，
    于是那份精心写的模板从未被注入过，只有一条 warning。同型问题不再犯第二次。

    与模板同语义：**启动时预加载一次**存 `self.about`，改文件重启生效。

    ## 缺失处置

    三份全缺 → 返回 `""`，调用方据此**整块不注**（不是注一个空壳）。
    部分缺 → 注已有的并告警。不设结构性兜底：自我介绍写错了不如不写——
    模板缺失还能靠 `FALLBACK_TEMPLATE_MD` 保住五段结构，而一段错的自我介绍
    会让模型对 BladeX 的能力边界产生错误预期，代价方向相反。
    """
    import os
    d = (dir_path or os.environ.get("BLADEX_SYSTEM_NOTES_DIR", "").strip())
    if not d:
        from bladex_proxy import deployment
        root = deployment.find_root() or os.getcwd()
        d = os.path.join(root, "config", "system")
    parts: list[str] = []
    missing: list[str] = []
    for name in _SYSTEM_NOTE_FILES:
        p = os.path.join(d, name)
        try:
            with open(p, encoding="utf-8") as f:
                text = f.read().strip()
        except OSError:
            missing.append(name)
            continue
        if text:
            parts.append(text)
        else:
            missing.append(name)
    if missing:
        logger.warning("system_notes_missing", dir=d, files=missing,
                       loaded=len(parts))
    if not parts:
        logger.warning("system_notes_empty", dir=d)
        return ""
    return "\n\n---\n\n".join(parts)


def load_agents_roster(max_rows: int = 15) -> str:
    """读 flash 树的 AGENTS.md 名册（V-F3；缺失 = 返回 ""，整段不注）。

    来源 = flash daemon 物化的机器文件（registry 缓存的确定性字段）。
    个人模式 principal 固定 "local"；企业形态的 per-principal 名册注入
    归后续（稳定层是进程级冻结，与 per-request principal 天然不合）。
    `max_rows` 封顶（ADR-0032 §3.1「一行一个封顶」）——名册是注意力预算的
    一部分，agent 多到溢出时裁最旧的（表按 first_seen 升序，尾部最新，保尾）。
    """
    import os as _os

    from bladex_core.flash_tree import agents_roster_path
    from bladex_proxy.config import resolve_flash_path
    path = agents_roster_path(resolve_flash_path(), "local")
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read().strip()
    except OSError:
        return ""
    lines = [ln for ln in text.splitlines() if "<!--" not in ln]
    header, rows = [], []
    for ln in lines:
        (rows if ln.startswith("|") else header).append(ln)
    if len(rows) > 2 + max_rows:          # 表头 2 行 + 数据行封顶，保尾（最新）
        rows = rows[:2] + rows[-max_rows:]
    out = "\n".join([*header, *rows]).strip()
    return out if rows else ""


#: Goal 取值的长度上限（超长时截断并留标记——Goal 的价值是"用户一眼认出自己说的话"，
#: 一屏以上的机器文本进 Goal 只会毁掉这个价值。与已删的老渲染器 GOAL_INJECT_MAX_CHARS=600 同量级）。
_GOAL_MAX_CHARS = 600

#: Goal 专用：整条以成对包装标签开头（`<session>…</session>`、`<user_message>…` 等）
#: ⇒ 取标签内容，丢弃标签后的机器指令（CC 的 `<session>` 形态，2026-08-25 live）。
#: 标签名限字母数字下划线短横，避免误吃真实的 HTML/代码片段（`<div>` 这类也只在
#: **整条以它开头**时才命中，且内容会作为 Goal 让用户一眼看见——错了看得出来）。
_WRAPPER_TAG_RE = re.compile(
    r"^<(?P<tag>[A-Za-z][\w-]{2,30})>\s*(?P<inner>.*?)\s*</(?P=tag)>",
    re.DOTALL)

#: Goal 专用：**裸前缀句**形态的 agent 包装（既不是已知信封、也不是 `<tag>` 包裹，
#: 所以上面两道都漏）。2026-08-26 live 实证：jydesignhk 那 5 本账本的 Goal 全部以
#: `Fully describe and explain everything about this image, then answer the
#: following question:` 开头，用户真正说的话被压在 90 字符之后——5 个 Goal 的前缀
#: 逐字相同，区分度全在尾部。
#:
#: 只在**整条开头**匹配，且只剥这一句（后面照原样留给提炼）。Goal 会摆在用户
#: 眼前，剥错了看得出来——这也是敢用正则而不是模型判的原因。
#: 🔴 有意不动 `bladex_core.envelope.strip_envelopes`：它被蒸馏消费，改它要 bump
#: 台账版本并影响重建等价（同上一条注释的理由）。
_BARE_PREFIX_RES = (
    re.compile(r"^\s*Fully describe and explain everything about this image[^\n:]*:\s*",
               re.IGNORECASE),
)


def _last_user_text(messages: list[dict], *, for_goal: bool = False) -> str:
    """末条 user 正文。`for_goal=True` 时**剥 agent 信封**并截断。

    🔴 2026-08-25 live 病例：CC 的 user 消息带 `<session>…</session>` 包装 +
    后续机器指令，原样取进 Goal ⇒ Goal 成了机器文本，"用户可核对"当场失效
    （ADR-0032 §4.3 goal「取」不是「炼」的前提是取到的是**用户说的那句**）。
    envelope 剥离是 V-R1 实证过的 CC 信封形态，现成复用（零 LLM、确定性）。
    """
    raw = ""
    for m in reversed(messages or []):
        if m.get("role") == "user":
            c = m.get("content")
            raw = (" ".join(str(b.get("text", "")) for b in c if isinstance(b, dict))
                   if isinstance(c, list) else str(c or ""))
            break
    if not for_goal or not raw:
        return raw
    try:
        from bladex_core.envelope import strip_envelopes
        cleaned, marks = strip_envelopes(raw)
        # 🔴 `strip_envelopes` 只认已知信封种类（它被蒸馏消费，改它要 bump 台账版本
        # 并影响重建等价 ⇒ 不为 Goal 动它）。Goal 侧再加一条**通用包装标签解包**：
        # 整条即 `<tag>…</tag>`（可后接指令）时取标签内容——CC 的 `<session>` 属此形态。
        for pat in _BARE_PREFIX_RES:
            stripped, n = pat.subn("", cleaned, count=1)
            if n:
                cleaned = stripped
                marks = [*marks, f"bare_prefix:{pat.pattern[:24]}"]
        m = _WRAPPER_TAG_RE.match(cleaned.strip())
        if m:
            cleaned = m.group("inner")
            marks = [*marks, f"wrapper:{m.group('tag')}"]
        if marks:
            logger.info("agency_goal_envelope_stripped", marks=marks[:4],
                        before=len(raw), after=len(cleaned))
        raw = cleaned.strip() or raw.strip()
    except Exception as e:  # noqa: BLE001 —— 剥离失败退回原文，不阻断建本
        logger.warning("agency_goal_envelope_failed", error=str(e))
    if len(raw) > _GOAL_MAX_CHARS:
        raw = raw[:_GOAL_MAX_CHARS] + "…[bladex-goal-truncated]"
    return raw


def _call_names(calls: list[dict]) -> str:
    """被剥离的 bladex 工具名，逗号连接（与 `_loop_cost` 的 `tools=` 同格式、同 8 个上限）。

    2026-09-02 补（n1b 报告 §5-1）：`*_mixed_stripped` 此前只记 `removed=N` 不记名字，
    日志侧算不出混合路径里 `bladex_memory_search` 的次数（N1a 记忆搜索"1 次"只是下界）。
    """
    names = [str((tc.get("function") or {}).get("name") or "") for tc in calls]
    return ",".join(n for n in names[:8] if n)


def _loop_cost(result: Any) -> dict:
    """内循环耗时归因（观测缺口 MQ-L5 #1）——日志侧读数；Hub 侧入账见 `loopledger`。

    历史：`LoopRound` 的 `tool_ms` / `llm_ms` / `usage` 曾一个都不输出
    （2026-08-25 那六次 88–135 秒的内循环只能翻相邻日志行猜钱花在哪）。
    🔴 2026-09-02 又发现 `tokens` **恒为 0**：`usage` 挂在上游 response 上而不在
    message 上，server 侧 `_loop_llm` 返回的是 `message.model_dump()` ⇒ 从来没有
    usage；且这里读的键 `total` 与 OpenAI 的 `total_tokens` 也不同名。两处一起修：
    `_loop_llm` 经 `server._loop_reply` 把 usage 归一后挂上，这里按归一口径读。
    """
    from bladex_proxy.loopledger import normalize_usage

    rounds = getattr(result, "rounds", None) or []
    tool_ms = sum(getattr(r, "tool_ms", 0.0) for r in rounds)
    llm_ms = sum(getattr(r, "llm_ms", 0.0) for r in rounds)
    tokens = sum(normalize_usage(getattr(r, "usage", None)).get("total", 0)
                 for r in rounds)
    tools: list[str] = []
    for r in rounds:
        tools.extend(getattr(r, "tool_names", None) or [])
    return {"tool_ms": round(tool_ms), "llm_ms": round(llm_ms),
            "tokens": tokens, "tools": ",".join(tools[:8])}


#: 用户点名账本的 title 最短长度（E0.3 / MQ-L41）：短于此的标题（"任务"、"调研"）
#: 出现在用户话里不算点名——那是日常词，不是指代。
USER_NAMED_TITLE_MIN_CHARS = 6


def _user_named_ledger(user_text: str, ledger: Any) -> tuple[bool, str]:
    """本轮用户原话是否点名了这本账本（按 id 或 ≥6 字符的 title 词面命中）。
    返回 (命中, "id" | "title" | "")。纯函数；`ledger` 为 None ⇒ 不命中。"""
    if ledger is None or not user_text:
        return (False, "")
    lid = str(getattr(ledger, "ledger_id", "") or "")
    if lid and lid in user_text:
        return (True, "id")
    title = str(getattr(ledger, "title", "") or "").strip()
    if len(title) >= USER_NAMED_TITLE_MIN_CHARS and title in user_text:
        return (True, "title")
    return (False, "")


def tool_context(*, session_id: str, agent_id: str, project_id: str,
                 upstream_messages: list[dict], turn_index: int = 0) -> dict:
    """工具面调用上下文——**唯一实现点**（三条拦截路径共用：非流式 / chat 流式 / 协议流式）。

    🔴 MQ-L40（2026-09-04）：此前三处各写一份字面量，两条流式路径**漏传 `turn_index`**
    ⇒ `context.get("turn_index", 0)` 恒 0 ⇒ `request_switch` 的防抖差值恒 0 < 3 ⇒
    一个 scope 上**只有进程内第一次切换能落定**，之后所有"切到已有账本"全被防抖拒绝
    （live：09-04 五次 `agency_switch_debounced`，全部是相隔 4–13 轮的正当切换；历史每个
    proxy 进程恰好 1 条 `ledger_switch` 事件）。唯一能过的路是 `ledger_id=''` 新建
    （`newly_created` 豁免）——这正是 09-03 泰山重复建本的**第二个**成因（第一个是
    候选面，MQ-L38）。`turn_index` 缺省时按 identity 同一公式从 `upstream_messages`
    派生，不依赖调用方记得传——公式 = `ledger_runtime.user_turn_index`（用户轮，E0.2；
    修前 `len(messages)//2` 是请求数，工具循环里 3 轮防抖 ≈ 3 个 bash 调用，MQ-L48）。
    """
    from bladex_core.ledger_runtime import user_turn_index  # noqa: PLC0415
    ti = int(turn_index) if turn_index else user_turn_index(upstream_messages)
    return {"session_id": session_id, "agent_id": agent_id,
            "turn_index": ti, "project_id": project_id,
            "user_query": _last_user_text(upstream_messages, for_goal=True)}


class AgencyRuntime:
    """每 proxy 进程一个实例（lifespan 构建，挂 app.state.agency）。"""

    def __init__(self, *, index: Any = None, hub: Any = None,
                 flash_push: Any = None) -> None:
        self._index = index
        self._hub = hub
        self._flash_push = flash_push
        #: 账本模板：启动时预加载一次（拍板：不在每请求读文件；改模板重启生效）。
        self.template = load_ledger_template()
        #: 强制注入账本面的档位集合（BLADEX_LEDGER_TIERS，启动冻结与模板同语义）。
        #: 消费者两个：工具面（inject_tools 内部）与块注入 gate（MQ-L28）——
        #: 同一真相源，防"同一开关两个消费点只接一个"复发。
        from bladex_proxy.toolface import resolve_ledger_tiers
        self._ledger_tiers = resolve_ledger_tiers()
        #: V-P6 自我介绍：同款预加载。空串 = 三份文件都缺 ⇒ 整块不注。
        self.about = load_system_notes()
        # V-F3：AGENTS.md 名册（flash daemon 物化的机器文件）。与 about 同一
        # 冻结语义（启动预加载，改了重启生效）、同一门控（随 about 块注入）。
        self.agents_roster = load_agents_roster()
        self.splice = SpliceLedger(
            max_age_turns=int(flag_number("BLADEX_SPLICE_MAX_AGE_TURNS")))
        #: 已尝试过 Hub 懒恢复的会话（V-P4 硬点 1）——见 `_restore_splice`。
        self._splice_restored: set[str] = set()
        #: MQ-P9：内循环轮次待入账缓冲（§3.2 硬点 6）。三个内循环调用点 `record()`，
        #: `server._enqueue_turn` 在主轮入库时 `drain()`——与 `splice` 同一接线形态。
        self.loop_ledger = InnerLoopLedger()
        #: MQ-L26：最近一次 `augment_tools` 的账本面决定，供 `_enqueue_turn` 落 Turn。
        #: `None` = 本进程还没处理过请求（不是"没给"——三态语义见 `Turn.ledger_face`）。
        self.last_ledger_face: bool | None = None
        self.activation = ActivationTable()
        self.pool: dict[str, Ledger] = {}
        self._last_candidates: list = []   # MQ-L38：本轮相关性候选（建本告警用）
        #: V-L4 陈旧兜底：ledger_id → **这本账本上过了多少轮而没被更新**。
        #:
        #: 为什么是进程内计数而不是从 Hub 派生：从 Hub 算准确（点时激活账本 ∧
        #: ts > updated_at），但那是一次全库 `scan_meta`——热路径吃不起。
        #:
        #: 🔴 **重启归零 = 只会漏报，不会误报**，这个方向是刻意选的：对一本刚更新过
        #: 的账本喊"陈旧"会让模型去怀疑正确的状态，比不喊更有害。
        self._turns_since_update: Counter[str] = Counter()
        #: ledger_id → 上次计入陈旧计数的 (session_id, user_turn)（E0.2，同用户轮内不重计）
        self._last_seen_user_turn: dict[str, tuple[str, int]] = {}
        self.recent = RecentRequests()
        self.toolface = ToolFace()
        self.toolface.register("bladex_memory_search", self._h_memory_search)
        self.toolface.register("bladex_ledger_read", self._h_ledger_read)
        self.toolface.register("bladex_ledger_update", self._h_ledger_update)
        self.toolface.register("bladex_ledger_switch", self._h_ledger_switch)
        self._load_pool()

    # ── 开关（每请求 env 读，µs 级；默认全关 = 零行为差异）──
    @property
    def interception_on(self) -> bool:
        return module_enabled("interception")

    @property
    def toolface_on(self) -> bool:
        return module_enabled("toolface")

    @property
    def ledger_on(self) -> bool:
        return module_enabled("ledger")

    # ── 池装载（Hub 投影）──
    def _load_pool(self) -> None:
        if self._hub is None:
            return
        try:
            # 2026-09-01：三个消费方共用唯一装载入口（判据/墓碑/ts 各一处）。
            from bladex_proxy.ledger_events import load_ledger_events
            events = load_ledger_events(self._hub)
            if events:
                from bladex_core.ledger import binding_sessions, binding_switch_turns
                self.pool, bindings = replay_ledger_events(events)
                self.activation.restore(bindings, sessions=binding_sessions(events),
                                        switch_turns=binding_switch_turns(events))
                logger.info("agency_pool_loaded", ledgers=len(self.pool),
                            bindings=len(bindings))
        except Exception as e:  # noqa: BLE001 —— 投影装载失败 = 优雅降级空池
            logger.warning("agency_pool_load_failed", error=str(e))

    def _emit(self, etype: AdminEventType, payload: dict) -> None:
        """账本事件入 Hub（matter_id 留空 = MQ-A7 护栏）+ push Flash 维护进程。"""
        if self._hub is not None:
            try:
                self._hub.append_admin_event(etype, "", **payload)
            except Exception as e:  # noqa: BLE001
                logger.warning("agency_event_append_failed", type=str(etype),
                               error=str(e))
        if self._flash_push is not None:
            try:
                self._flash_push()
            except Exception:  # noqa: BLE001,S110
                pass

    #: 每会话只尝试恢复一次——没有拼接条目的会话是绝大多数，
    #: 不加这个标记就会**每一轮**都去扫一次 Hub（热路径，200ms 预算）。
    _SPLICE_RESTORE_SCAN_TURNS = 20

    def _restore_splice(self, session_prefix: str) -> list[SpliceRecord]:
        """从 Hub 重建本会话的拼接表（懒恢复，每会话一次）。

        为什么是懒的而不是启动时全量：启动时不知道哪些会话会回来，
        扫全库既慢又白做；而拼接条目本就允许老化退出，恢复晚一步无害。

        失败 = 优雅降级（ADR-0032 §3.2 硬点 1：拼接状态丢失时会话仍合法，
        agent 侧历史本就自洽，LLM 只是失忆那次查询）——所以这里吞异常。
        """
        if session_prefix in self._splice_restored:
            return []
        self._splice_restored.add(session_prefix)
        if self._hub is None:
            return []
        try:
            turns = list(self._hub.scan_prefix(session_prefix))
        except Exception as e:  # noqa: BLE001 —— 恢复失败 = 降级，不阻断这一轮
            logger.warning("agency_splice_restore_failed",
                           session=session_prefix, error=str(e))
            return []
        tail = turns[-self._SPLICE_RESTORE_SCAN_TURNS:]
        recovered: list[SpliceRecord] = []
        for pos, (_key, turn) in enumerate(tail):
            for rec in getattr(turn, "splice_records", None) or []:
                # age 按"它之后还有几轮"重算——原值是写入那一刻的 0，
                # 直接沿用会让老化窗口从恢复时刻重新计时（等于永不老化）。
                rec = rec.model_copy(update={"age_turns": len(tail) - 1 - pos})
                recovered.append(rec)
        # 同锚后写覆盖先写（与内存表 append 后 `splice_into_messages`
        # 逐条应用的语义一致；这里先去重，避免同一锚拼两次）。
        by_anchor: dict[str, SpliceRecord] = {}
        for rec in recovered:
            by_anchor[rec.anchor] = rec
        out = list(by_anchor.values())
        if out:
            self.splice.restore(out)
            logger.info("agency_splice_restored", session=session_prefix,
                        records=len(out), scanned_turns=len(tail))
        return out

    # ── server 接线面 ① inbound 准备 ──
    def prepare_inbound(self, messages: list[dict], session_prefix: str) -> list[dict]:
        if not self.interception_on:
            return messages
        out, stripped = strip_inbound_echoes(messages)
        records = self.splice.records(session_prefix)
        if not records:
            # V-P4 硬点 1：进程内表是 Hub 的**投影**，不是唯一副本。
            # 内存里没有 ⇒ 可能是 proxy 重启过，从 Hub 懒恢复一次。
            records = self._restore_splice(session_prefix)
        out, applied, dropped = splice_into_messages(out, records)
        self.splice.tick_and_prune(session_prefix)
        if stripped or applied or dropped:
            logger.info("agency_inbound_prepared", stripped=stripped,
                        spliced=applied, anchors_dropped=dropped)
        return out

    # ── server 接线面 ② tools 增补 ──
    def augment_tools(self, tools: list[dict] | None, *, agent_id: str,
                      auxiliary: bool, tier: str, fmt: str = "openai",
                      stream: bool | None = None,
                      session_id: str = "", project_id: str = "",
                      ledgerless: bool | None = None, aux_reason: str = "",
                      ) -> tuple[list[dict] | None, bool]:
        """工具面增补。两个 aux 旋钮（MQ-L34，2026-09-02）：

        - `auxiliary`：挡**两族**（真·内部调用，记忆族也不给）；
        - `ledgerless`：只挡**账本族**（零工具 / 孤立子调用这类"不该建账本"的
          结构判据，记忆族照注）。`None` = 沿用 `auxiliary`（旧调用方兼容）。
        - `aux_reason`：命中的子条件名（`identity.AUX_REASON_*`），只进日志——
          此前 `reason=aux` 四个子条件一个字不分，08-31 hermes:accept 18 轮读不出
          是哪一个（MQ-L5「二义的观测字段等于没有观测」同族）。
        """
        if not self.toolface_on:
            return tools, False
        if ledgerless is None:
            ledgerless = auxiliary
        # 2026-08-30 按工具族拆：`tier` 只 gate **账本族**，记忆族无条件注入。
        # 🔴 账本族的第二个放行口（同日补，MQ-L7 补完）：**本会话绑着激活账本**
        # 时照给——否则模型看得见账本却切不走（判据与块注入 gate 同源，
        # 见 `session_owns_active_ledger` 的 docstring）。
        # 🔴 `ledgerless` 压过两个放行口（MQ-L34）：零工具轮即使绑着账本也不给
        # 账本工具——它本来就是 MQ-L21 病例里"建噪声账本"的那一类请求。
        _owns = self.session_owns_active_ledger(agent_id, project_id, session_id)
        _ledger_ok = ((ledger_tools_allowed(tier, self._ledger_tiers) or _owns)
                      and not ledgerless)
        new_tools, injected = inject_tools(tools, agent_id=agent_id,
                                           auxiliary=auxiliary, model_tier=tier,
                                           ledger_allowed=_ledger_ok, fmt=fmt)
        # 🔴 观测缺口 MQ-L5 #2：注没注**此前无任何日志** ⇒ 一轮零账本活动时
        # 分不清是"没给工具"还是"给了模型没用"——2026-08-26 分析那 6 轮
        # Hermes 就卡在这个二义上，只能靠读代码倒推 tier。
        # 每请求一行、字段就四个，换回"这个问题永远能直接回答"。
        # 🔴 `stream`：V-P5③（2026-08-27）。此前日志里**没有任何**区分入站流式/
        # 非流式的信号——于是"非流式路径没接拦截"这个缺口发现之后，
        # 想量它有没有被真实踩到都做不到（`route_calling stream=False` 是
        # 内循环自己的上游调用，不是入站形态，当天差点据此误判）。
        # 挂在这条而不是新开一条：它三个端点都打、且正是排查时会 grep 的那条。
        logger.info("agency_toolface_decision", injected=injected,
                    agent=agent_id, tier=tier, auxiliary=auxiliary,
                    stream=stream,
                    # MQ-L34：子条件名单独一列（注了也要记——零工具轮现在注记忆族，
                    # `reason` 为空，子条件只能从这里读）；`ledgerless` 记账本族
                    # 被结构判据挡住这件事本身。
                    aux_reason=aux_reason, ledgerless=ledgerless,
                    # 🔴 账本族这一轮凭什么放行：档位在集合内，还是"本会话绑着
                    # 账本"这个新口子？不分开记，live 上就只能靠读代码倒推
                    # （MQ-L5 #2 同款理由——那次就是因为分不清多花了一轮）。
                    ledger_face=_ledger_ok, ledger_by_session=_owns,
                    # 🔴 末档原为 `agent_excluded_or_already_present` —— 两个
                    # 完全不同的原因共用一句文案，排查时分不清（2026-08-28 查
                    # dsh 时因此多走了一轮：以为是档位问题，实际它在排除名单里）。
                    # 二义的观测字段等于没有观测（MQ-L5 同族）。
                    # 🔴 2026-08-30 拆 weak 门后删掉了 `weak_tier` 这一档：
                    # 它不但成了死分支，还会**给出错误的标签**——分支顺序是
                    # aux → weak_tier → agent_excluded，于是一个 weak 档的 dsh
                    # 会被报成 `weak_tier` 而真因是 `agent_excluded`。
                    # 那正是上面那段注释警告过的"二义的观测字段等于没有观测"，
                    # 只是这次是**过期**而不是含混。
                    reason=("" if injected else
                            (("aux" + (f":{aux_reason}" if aux_reason else ""))
                             if auxiliary else
                             "agent_excluded"
                             if (agent_id or "").split(":")[0]
                             in NO_TOOLFACE_AGENT_BASES else
                             "already_present")))
        # 🔴 `ledger_face` 记的是**账本面**的决策，不是"工具面注了任何东西"
        # （2026-08-30 族拆分后两者不再等价：weak 档只拿到记忆工具）。
        # 字段名宽于语义正是 MQ-L28 连带修正记下的那个仪器陷阱，不许复发。
        # MQ-L26：把这一轮的决定记下来，供 `_enqueue_turn` 落进 `Turn.ledger_face`。
        # 记在 runtime 上而不是改 `_enqueue_turn` 签名——那有 12 个调用点，
        # 与 splice_records 同款做法。每请求恰好覆写一次（三端点各调一次 augment）。
        self.last_ledger_face = injected and _ledger_ok
        return (new_tools if injected else tools), injected

    # ── server 接线面 ②a 自我介绍（稳定层，V-P6 / ADR-0032 §3.1）──

    def insert_system_notes(self, messages: list[dict], *,
                            toolface_injected: bool) -> list[dict]:
        """把 BladeX 自我介绍插进**前缀**（稳定层）。返回新数组，不改原数组。

        ## 位置：前缀，与账本块（末尾）**刻意相反**

        稳定层的内容每轮逐字相同，放在前缀 ⇒ 它之后的历史全部落在同一个
        cache 前缀里；放末尾则每轮都在变动区之后，白白让上游多算一遍。
        动态层（账本块）反过来——它本来就每轮重算，追加末尾对 cache 无损
        且注意力位置最好（MQ-L10）。**两层的最优位置不同，这不是不一致。**

        插在最后一条 **agent 自己的** `system` 之后：agent 的 system prompt 仍是
        第一位的（刚性原则 10——我们是接入方，不抢开场），而我们的说明先于全部
        对话历史。没有 system 消息时插在最前。

        🔴 **必须跳过 BladeX 自己的注入块**（判据 = `INJECTION_MARKERS`）。
        `ledger_injection_message` 返回的 role **也是 `system`**，而账本块按
        MQ-L10 追加在数组**真正的末尾**——只找"最后一条 system"会找到它，
        把自我介绍插到它后面，账本块就此被挤离末尾。那正是 MQ-L10 修掉的病：
        账本块携带的指令是"FIRST STEP, every turn"，被埋在后面就等于没有。
        （首版就是这么写的，2026-08-28 gate 当场红——`pos=16 total=17`。）

        跳过还有第二个理由：`<bladex-memory>` 每轮内容都不同，插在它**之后**
        会让稳定层落进变动区，prompt cache 的收益归零——而那正是选前缀的全部理由。

        ## 门控：与工具面同步

        `toolface_injected=False` ⇒ **不注**。AGENT.md/TOOLS.md 通篇在讲
        "call bladex_ledger_switch / update"——没有工具面就是**有令无器**，
        纯注意力税。这正是 MQ-L7 拆开账本正文与首步指令的同一条理由，
        也是三红线之一「模型不调用时零行为差异」的直接推论。

        ## 去重

        agent 可能把上一轮的注入当历史回传（CC/Codex 实测会）。已存在标记
        ⇒ 直接返回原数组，不叠第二份。
        """
        if not self.about or not toolface_injected:
            return messages
        for m in messages:
            c = m.get("content")
            if isinstance(c, str) and ABOUT_BLOCK_OPEN in c:
                logger.debug("agency_about_already_present")
                return messages
        body = self.about
        if self.agents_roster:
            # 名册并入同一稳定块：仍是一条 system、同一冻结前缀（多开一条消息
            # 只会多一个 cache 边界）。一行一个 agent，封顶见 load_agents_roster。
            body = f"{body}\n\n{self.agents_roster}"
        block = {"role": "system",
                 "content": f"{ABOUT_BLOCK_OPEN}\n{body}\n{ABOUT_BLOCK_CLOSE}"}
        pos = 0
        for i, m in enumerate(messages):
            if m.get("role") != "system":
                continue
            c = m.get("content")
            if isinstance(c, str) and any(mk in c for mk in INJECTION_MARKERS):
                continue          # BladeX 自己的注入块，不是 agent 的 system
            pos = i + 1
        out = [*messages[:pos], block, *messages[pos:]]
        logger.info("agency_about_injected", pos=pos, total=len(out),
                    chars=len(block["content"]))
        return out

    # ── server 接线面 ②b 账本块注入（动态层）──
    def scope_of(self, agent_id: str, project_id: str = "") -> str:
        """本请求的激活作用域键。见 `ledger_runtime.activation_scope` 的立论。"""
        return activation_scope(agent_id, project_id)

    def ledger_injection_message(self, agent_id: str, *, project_id: str = "",
                                 with_instruction: bool = True,
                                 user_text: str = "") -> dict | None:
        """账本块。ledger/toolface 任一关 → None（零差异）。

        🔴 **C 方案（Jason 2026-08-26 拍板，MQ-L7）：正文与首步指令分开**。
        `with_instruction=False` 时只注**只读正文**（当前账本的 Goal/Core/Verified…），
        不注"请调用 bladex_ledger_switch/update"那段、不注模板、不注 id 列表。

        为什么要拆：两个 gating 的执行位置不同——账本块在**路由之前**注（它的体积要
        喂给路由规模层），工具面在**路由之后**按 tier 注。于是 weak 档出现
        "**有令无器**"：注入块白纸黑字写着 "call bladex_ledger_switch"，而 tools 数组里
        根本没有这个工具（实测 213 个非 aux 轮里 6 轮命中，约 500 tokens/轮 纯注意力税；
        n=4 样本未观察到幻觉调用）。这与 aux gating 的理由**自相矛盾**——
        aux 正是因为"有令无器纯属注意力税"才不注的。

        拆开之后语义变清楚：**看得见 ≠ 改得了**。weak 档保留只读账本
        （模型仍能看到 Goal/Core 从而聚焦），只是不被要求去改它。
        """
        if not (self.ledger_on and self.toolface_on):
            return None
        led = self._active_ledger(self.scope_of(agent_id, project_id))
        lines = [LEDGER_BLOCK_OPEN]
        if led is None:
            if not with_instruction:
                # 无账本 + 无工具 = **无器则无令**，整块不注（C 方案要点）。
                return None
            # 🔴 无账本分支**也要给列表**（V-L6e 复核发现）：不给列表却要求"没有就建"，
            # 模型即使想并入既有账本也无从判断——这正是 CC 那次同一任务开两本的一半成因。
            lines += [_FIRST_STEP_NO_LEDGER, "", self.template.strip()]
        else:
            note = self._coauthor_note(led, agent_id)
            header = ("Active task ledger (your working state):"
                      if not note else
                      "Active task ledger (your working state; " + note + "):")
            lines += [header, "", render_ledger_md(led).strip()]
            # V-L4 陈旧兜底：只陈述"这本账本上过了 N 轮没更新"这个事实。
            # 🔴 **不受 `with_instruction` 门控的反面**——它带一句"用
            # bladex_ledger_update 记下来"，没有工具面时就是有令无器（C 方案的
            # 教训，MQ-L7）。故与指令同门控。
            if with_instruction:
                mark = staleness_marker(
                    turns_since_update=self._turns_since_update[led.ledger_id])
                if mark:
                    lines += [mark]
                    logger.info("agency_ledger_stale_marked",
                                ledger=led.ledger_id,
                                turns=self._turns_since_update[led.ledger_id])
                lines += ["", _FIRST_STEP_WITH_LEDGER]
        # V-L6e：父子链可见（Jason 拍板"子任务可以建账本，但要与父账本信息同步"）。
        # 子侧的 parent 行由 render_ledger_md 渲染；这里补**父侧看得见子**——
        # 派生而非双写（单一方向存储，`children_of`）。子列表是**只读信息**
        # （父任务的进展全貌），故不受 with_instruction 门控。
        if led is not None:
            kids = children_of(self.pool, led.ledger_id)
            if kids:
                lines += ["", "Sub-ledgers spawned from this task "
                              "(switch by id to work in one):"]
                lines += [f"- {k.ledger_id}: {k.title or k.ledger_id} [{k.status}]"
                          for k in kids[:5]]
        # LEDGERS top-5 标题行（拍板 #6：只注标题，供切换判断）。
        # 🔴 只在带指令时注：它的唯一用途是"要不要 switch 过去"，没有 switch 工具
        # 时是纯噪声（而且实测这份列表会被子调用产生的噪声账本挤满，MQ-L8）。
        if with_instruction:
            # MQ-L38（2026-09-04）：相关性位——按当前用户话在**全池**找"像同一件事"的本。
            # recency top-5 只能回答"最近在干什么"；08-25 的泰山旧本排第 9，用户原话逐字
            # 重提时模型面前没有"切回"这个选项，只能新建。两段并列，top-5 一字不动。
            matches = relevant_ledgers(self.pool, user_text,
                                       exclude_id=led.ledger_id if led else "")
            self._last_candidates = matches
            matched_ids = {m.ledger_id for m in matches}
            if matches:
                lines += ["", _MATCHING_LEDGERS_HEADER]
                lines += [f"- {m.ledger_id}: {m.title} [{m.score:.2f}]" for m in matches]
            # 🔴 排序键是数值不是 ISO 串（MQ-L46）：池里混入不同偏移时字符串序会错，
            # 而这里就是 recency top-5——排错比时间显示错更贵。
            others = [(iso_ms(l.updated_at or l.created_at), lid, l.title or lid)
                      for lid, l in self.pool.items()
                      if (led is None or lid != led.ledger_id)
                      and lid not in matched_ids]   # 命中项只在相关段出现一次
            if others:
                others.sort(reverse=True)
                lines += ["", "Other recent ledgers (read one with "
                              "bladex_ledger_read before switching):"]
                lines += [f"- {lid}: {title}" for _, lid, title in others[:5]]
            logger.info("agency_ledger_candidates",
                        relevant=[m.ledger_id for m in matches],
                        top_score=(matches[0].score if matches else 0.0),
                        recent=min(len(others), 5),
                        query_units=len(candidate_units(user_text)))
        lines.append(LEDGER_BLOCK_CLOSE)
        return {"role": "system", "content": "\n".join(lines)}

    def insert_ledger_block(self, messages: list[dict], agent_id: str,
                            *, project_id: str = "", auxiliary: bool = False,
                            with_instruction: bool = True,
                            tier: str = "", session_id: str = "") -> list[dict]:
        """把账本块追加到**消息数组真正的末尾**。

        🔴 2026-08-26 修正（MQ-L10）：此前插在"最后一条 `role=user` 之后"，
        原意是"紧跟用户消息 = 上下文最末 = 注意力最高"。**在长工具循环里这两个位置
        不是同一个**：jydesignhk 那次主会话 27 轮里最后一条 user 恒在下标 167 不动，
        工具循环把 assistant/tool 对全追加在它之后 ⇒ 账本块**从第 17 条深处开始、
        每轮再沉 2 条，到第 27 轮已被埋在 65 条之下**。它携带的指令是
        "FIRST STEP, **every turn**: compare … with the Goal above"——而模型实际
        阅读位置在它 65 条之后。同一请求的子调用 `msgs=1`（块就在末尾）**开了 5 张
        账本**，主会话 17 轮**零调用**：机制在不该生效处 100% 生效、该生效处 0%。

        与 ADR-0019 同族——那次证伪的是"对话条数老化"，这次证伪的是
        **"末条 user ≈ 上下文末尾"**（对长工具循环的 agent 系统性失效）。

        追加到末尾对上游 prefix cache 是**友好**的：块在最末，它之前的前缀逐字不变；
        块本身每轮重算，但它本来就是每轮重算的（动态层）。

        aux 轮不注（有令无器纯属注意力税——与工具面 gating 同一条理由）。

        🔴 MQ-L28 绑定会话 gate（2026-08-29 Jason 拍板，Pi 天气循环事故）：
        集合外档位（无工具面）只在「激活账本是**本会话**绑的」时注只读正文——
        C 方案的连续性场景（同会话"继续"被路由到弱档）保住；**继承自其它会话
        的绑定不注**：弱模型读到旧任务 goal 会照办、又没有工具去 switch/关闭，
        只能无限循环旧任务（live 实证：doubao-lite 把昨天完结的"北京天气查询"
        账本当指令重新执行）。来源未知（老事件无 session）按继承保守处理。
        强档不变——有工具 + 首步指令，错配由模型自行裁决（设计如此）。"""
        if auxiliary:
            return messages
        # 🔴 首步指令的判据（2026-08-30 拆 weak 工具面门时独立出来）。
        #
        # 此前调用方直接传 `with_instruction=tf_injected`（工具面注没注），两件事
        # 等价。weak 档工具面门拆掉之后那个等价**会顺手把首步指令也给到 weak**
        # ——那等于改了账本注入策略，而 Jason 同日明确"账本注入策略保持不变"。
        # 故在这里合成，判据两个都要：
        #   ① `tier ∈ LEDGER_TIERS`  —— 账本策略（集合外档位不给指令，语义不变）
        #   ② 调用方传进来的工具面结果 —— **MQ-L7 的"不许有令无器"不变式**：
        #      指令说的是"调用 bladex_ledger_switch"，工具没注就是发空头支票。
        #      拆门后仍有一条路径会"tier 在集合内但工具面没注"：dsh 系 agent
        #      （`NO_TOOLFACE_AGENT_BASES`，它只有 run_code 可直呼）。
        # 拆开后的关系从"等价"变成"蕴含"：有令 ⇒ 必有器；有器不一定有令
        # （weak 档就是"有器无令"——安全，工具在那儿，用不用归模型）。
        # 放在生产点而不是调用点：调用方少传一个条件就静默退化，这类漏接
        # 2026-08-29 刚踩过（三条流式入口漏传 ctx）。
        with_instruction = bool(with_instruction) and ledger_tools_allowed(
            tier, self._ledger_tiers)
        self._maybe_migrate_binding(agent_id, project_id)
        _scope0 = self.scope_of(agent_id, project_id)
        if not ledger_tools_allowed(tier, self._ledger_tiers):
            if not self.session_owns_active_ledger(agent_id, project_id, session_id):
                bound_sess = self.activation.last_session(_scope0)
                if self._active_ledger(_scope0) is not None:
                    logger.info("agency_ledger_block_gated",
                                reason=("inherited_binding" if bound_sess
                                        else "unknown_binding_origin"),
                                tier=tier, agent=agent_id,
                                bound_session=(bound_sess or "")[:24],
                                session=(session_id or "")[:24])
                return messages
        # V-L4：**先记这一轮，再渲染**——计数是"到这一轮为止在这本账本上过了多少轮"。
        # 记在这里而不是 `ledger_injection_message` 里，是因为这是每请求恰好一次的
        # 位置（aux 已在上面返回，三个入站端点各调一次）；渲染函数会被测试反复调用，
        # 把副作用放进去会让计数随调用次数漂移。
        led0 = self._active_ledger(self.scope_of(agent_id, project_id))
        if led0 is not None:
            # E0.2（MQ-L48）：按**用户轮**计——同一用户轮内的工具往返（每个工具调用
            # 一次请求）不重复计。判据 = (session, user_turn) 与上次所见不同；用
            # `!=` 而非 `>`：同 session 压缩会让 user_turn 回落，那也是新的一轮。
            from bladex_core.ledger_runtime import user_turn_index  # noqa: PLC0415
            _seen = (session_id, user_turn_index(messages))
            if self._last_seen_user_turn.get(led0.ledger_id) != _seen:
                self._last_seen_user_turn[led0.ledger_id] = _seen
                self._turns_since_update[led0.ledger_id] += 1
        msg = self.ledger_injection_message(
            agent_id, project_id=project_id, with_instruction=with_instruction,
            user_text=_last_user_text(messages, for_goal=True))
        if msg is None:
            return messages
        out = [*messages, msg]
        # 🔴 观测缺口 MQ-L5 #3：此前这一步**一条日志都没有** ⇒ live 上无法证明
        # 账本正文真的进了请求、更无法证明它在哪个位置（Hub 存的是注入**前**的
        # `original_messages`）。MQ-L10 那次就是因为看不见，才让"块被埋 65 条"
        # 潜伏到靠翻消息结构才发现。`after=0` 是该条的**判据本身**。
        led = self._active_ledger(self.scope_of(agent_id, project_id))
        logger.info("agency_ledger_block_injected",
                    ledger=led.ledger_id if led else "",
                    with_instruction=with_instruction,
                    pos=len(out) - 1, total=len(out), after=0,
                    chars=len(msg.get("content") or ""))
        return out

    # ── server 接线面 ③ 非流式响应处置 ──
    async def process_message(
        self, message: dict, *, upstream_messages: list[dict],
        session_prefix: str, allowed_exposure: str, call_llm: Any,
        session_id: str, agent_id: str, turn_index: int = 0,
        project_id: str = "",
    ) -> tuple[dict, list[dict], str]:
        """分流处置一条 assistant 消息。返回 (给 agent 的消息, 内循环 transcript, mode)。"""
        if not self.interception_on:
            return message, [], MODE_NONE
        disp = classify_message(message)
        if disp.mode == MODE_NONE:
            return message, [], MODE_NONE
        ctx = tool_context(session_id=session_id, agent_id=agent_id,
                           project_id=project_id, upstream_messages=upstream_messages,
                           turn_index=turn_index)
        if disp.mode == MODE_PURE:
            fp = request_fingerprint(upstream_messages)
            if self.recent.seen(fp):
                # 重发（codex 超时形态）：不重跑内循环、不重花 LLM。
                logger.warning("agency_resend_dedup", session=session_id)
                return ({"role": "assistant",
                         "content": "[BladeX] duplicate request detected; "
                                    "previous memory lookup already ran."},
                        [], MODE_PURE)
            result: InnerLoopResult = await run_inner_loop(
                messages=upstream_messages, initial_calls=disp.bladex_calls,
                call_llm=call_llm,
                dispatch=lambda n, a: self.toolface.dispatch(
                    n, a, allowed_exposure=allowed_exposure, context=ctx),
            )
            self.loop_ledger.record(session_prefix, result)
            # 🔴 `agent` / `final_tool_calls` 是 MQ-L23 读数器的判据字段（2026-09-02
            #: 补齐）：08-31 给两条流式路径补过同款，**非流式这条漏了** —— 09-02 构造
            #: 流量首验时读数器对着一次真实内循环报"本次没有内循环"。三条路都对、
            #: 只有两条被量，第二次（MQ-P3 同族）。
            # MQ-L45：内循环的最终回复可能是 MIXED（模型先 read、再同轮 switch+自己的工具）
            # ⇒ 必须剥离后再交给 agent，否则 `bladex_*` 会被 agent 当自己的工具执行。
            final, leaked, extra = await self._strip_execute_splice(
                result.final_message, ctx=ctx, session_prefix=session_prefix,
                allowed_exposure=allowed_exposure)
            if leaked:
                logger.info("agency_inner_loop_final_stripped",
                            agent=agent_id, count=len(leaked), names=_call_names(leaked))
            logger.info("agency_inner_loop_done", rounds=len(result.rounds),
                        agent=agent_id,
                        final_tool_calls=len(final.get("tool_calls") or []),
                        final_bladex_stripped=len(leaked),
                        degraded=result.degraded, reason=result.degrade_reason,
                        elapsed_s=round(result.elapsed_s, 1),
                        **_loop_cost(result))
            return final, [*result.transcript, *extra], MODE_PURE
        # MIXED：执行己方 → 剥离 → 记拼接 → 下发
        stripped, removed, transcript = await self._strip_execute_splice(
            message, ctx=ctx, session_prefix=session_prefix,
            allowed_exposure=allowed_exposure)
        logger.info("agency_mixed_stripped", removed=len(removed),
                    kept=len(disp.agent_calls), names=_call_names(removed))
        return stripped, transcript, MODE_MIXED

    async def _strip_execute_splice(
        self, message: dict, *, ctx: dict, session_prefix: str,
        allowed_exposure: str,
    ) -> tuple[dict, list[dict], list[dict]]:
        """剥离 `bladex_*` 调用 → 就地执行 → 记拼接。返回 (给 agent 的消息, 被剥的调用, transcript)。

        🔴 **唯一实现点**（MQ-L45，2026-09-05）：此前只有 `process_message` 的 MIXED 分支
        做这件事，而**内循环的最终回复若是 MIXED，三条调用方一条都没剥**——
        `run_inner_loop` 的注释写着「mixed 的剥离-拼接归调用方」，三个调用方都没接。
        后果是 `bladex_*` 调用被原样转发给 agent，agent 报 `Tool ... not found`
        （Hub 取证：09-05 12:12 与 13:29 各一次，agent 因此判定"BladeX 工具不可用"
        并转去翻文件系统）。收成一个 helper 就是为了让"谁来剥"不再有第二个答案。

        `removed` 为空时**逐字返回原消息**（`strip_bladex_calls` 的行为），
        故三条路径可以无条件调用，不必各自先判 mode——少一个判据就少一处分叉。
        """
        stripped, removed = strip_bladex_calls(message)
        if not removed:
            return message, [], []
        # MQ-P8：**永不外发空 id**——铸完再锚，agent 回传即同 id
        # （下游 anthropic/responses formatter 的兜底合成因此不再触发=不再分叉）。
        ensure_call_ids(stripped.get("tool_calls") or [])
        results: list[dict] = []
        for tc in removed:
            fn = tc.get("function") or {}
            text = await self.toolface.dispatch(
                fn.get("name", ""), fn.get("arguments", "{}"),
                allowed_exposure=allowed_exposure, context=ctx)
            results.append({"role": "tool", "tool_call_id": tc.get("id", ""),
                            "content": text})
        rec = SpliceRecord(session_prefix=session_prefix,
                           anchor=anchor_key(stripped), calls=removed,
                           results=results, created_ms=int(time.time() * 1000))
        self.splice.add(rec)
        transcript = [{"role": "assistant", "content": None, "tool_calls": removed},
                      *results]
        return stripped, removed, transcript

    # ── ToolFace handlers ────────────────────────────────────────────────────

    @staticmethod
    def _fact_visible(f: Any, allowed_exposure: str) -> bool:
        """§3.2-7：工具结果也是注入——敏感条目按本请求目的地过滤（四模式同一道门）。"""
        ceiling = getattr(f, "exposure_ceiling", "") or "public"
        return exposure_allows(ceiling, allowed_exposure)

    async def _h_memory_search(self, args: dict, *, allowed_exposure: str,
                               context: dict) -> str:
        """四模式检索（V-I3，ADR-0032 §6.4）：query 语义检索（原行为）/
        keyword 精确关联（跨 fact/matter，V-I2 规范化键）/ matter_id 域内取 /
        fact_id 单条取。优先级 fact_id > matter_id > keyword > query。
        不可见与不存在**同文案**——不向低 exposure 目的地泄露条目存在性。"""
        if self._index is None:
            return "Error: memory index unavailable."
        from bladex_core.topic_keys import canonical_entity_key
        fact_id = str(args.get("fact_id", "")).strip()
        matter_id = str(args.get("matter_id", "")).strip()
        keyword = str(args.get("keyword", "")).strip()
        query = str(args.get("query", "")).strip()
        top_k = min(int(args.get("top_k", 5) or 5), 15)

        if fact_id:
            f = self._index.get_fact(fact_id)
            if f is None or not self._fact_visible(f, allowed_exposure):
                return f"No fact {fact_id!r}."
            keys = sorted({canonical_entity_key(str(e))
                           for e in (getattr(f, "entities", None) or [])
                           if canonical_entity_key(str(e))})
            return (f"Fact {fact_id}:\n- {f.content}\n"
                    f"(keywords: {', '.join(keys) or '-'})")

        if matter_id:
            m = self._index.get_matter(matter_id)
            if m is None:
                return f"No matter {matter_id!r}."
            facts = self._index.get_facts_for_matter(matter_id, top_k=top_k * 2)
            lines = [f"- {f.content}" for f in facts
                     if self._fact_visible(f, allowed_exposure)][:top_k]
            tk = m.topic_keys or {}
            keys = sorted(tk, key=lambda k: -tk[k])[:8]
            status = getattr(m.status, "value", str(m.status))
            return (f"Matter {m.matter_id} {m.title!r} (status={status}):\n"
                    f"keywords: {', '.join(keys) or '-'}\n"
                    + ("\n".join(lines) or "(no visible facts)"))

        if keyword:
            matters, facts = self._index.keyword_lookup(keyword)
            vis = [f for f in facts if self._fact_visible(f, allowed_exposure)]
            parts = [f"- (matter {m.matter_id}) {m.title}" for m in matters[:5]]
            parts += [f"- {f.content}" for f in vis[:top_k]]
            if not parts:
                return f"No memory found for keyword {keyword!r}."
            return f"Keyword {keyword!r}:\n" + "\n".join(parts)

        if not query:
            return ("Error: pass exactly one of query (semantic), "
                    "keyword (exact), matter_id, or fact_id.")
        facts = self._index.search(query, top_k=top_k * 2,
                                   session_id=context.get("session_id", ""))
        out = []
        for f in facts:
            if not self._fact_visible(f, allowed_exposure):
                continue
            out.append(f"- {f.content}")
            if len(out) >= top_k:
                break
        if not out:
            return f"No memory found for {query!r}."
        return f"Memory hits for {query!r}:\n" + "\n".join(out)

    def _maybe_migrate_binding(self, agent_id: str, project_id: str) -> None:
        """惰性作用域迁移（2026-08-29 Jason 拍板，MQ-L30 收尾）。

        规则（窄）：项目键无绑定 ∧ 同 agent Global 键有绑定 ∧ 本请求有项目
        信号 ⇒ 把绑定搬到项目键。立论：项目键时代之前的 Global 绑定本就是
        "项目无关混用"，搬到当前项目 = 维持键改造前的连续性；无项目信号的
        agent（Pi/hermes）查的就是 Global，本函数对它们是空操作。
        实测依据：自发自愈 0/2（模型信历史不信注入）、codex desktop 会话
        指纹跨天不换（"新会话自愈"不可用）；且卡死期间每轮注入"无账本请新建"
        有诱发重复建本的风险（doctor duplicate_title 前科）。
        """
        if not project_id or not self.ledger_on:
            return
        dst = activation_scope(agent_id, project_id)
        if self.activation.active(dst):
            return
        src = activation_scope(agent_id, "")
        lid = self.activation.migrate(src, dst)
        if not lid:
            return
        logger.warning("agency_scope_binding_migrated", agent=agent_id,
                       project=project_id, ledger=lid)
        self._emit(AdminEventType.LEDGER_SCOPE_MIGRATE,
                   {"agent_id": agent_id, "from_project": "",
                    "to_project": project_id, "ledger_id": lid})

    def session_owns_active_ledger(self, agent_id: str, project_id: str,
                                   session_id: str) -> bool:
        """**本会话**是否绑着一本激活账本（MQ-L28 判据，单一定义）。

        两个消费者读它，**不各自拼一遍**：
          ① 块注入 gate（`insert_ledger_block`）——集合外档位只给本会话绑的正文；
          ② 账本工具族放行（`augment_tools`）——见下面那条为什么。

        🔴 **为什么工具族也要看它**（2026-08-30 Jason 提，MQ-L7 的补完）：
        C 方案原本是"正文给、指令不给"，用**拿掉指令**来消除"有令无器"。
        实测下来那不够——模型看得见账本、发现任务不对，**却切不走**
        （没有 `bladex_ledger_switch`），只能照着旧 Goal 继续做。这正是
        MQ-L28 那个 Pi 天气循环事故的第二半，当时被记成"弱模型没有工具去
        更新/关闭/切换"，此后一直没人补上。
        补的是**能力**不是催促：工具给，首步指令仍按档位（见
        `insert_ledger_block`）——一次只动一个变量，弱模型误调用的风险
        还没有读数（MQ-L6 那次 43% 调用率是在**有指令**的 all 配置下测的）。

        判据三项缺一不可：有绑定来源会话 / 本轮有 session_id / 两者相等。
        来源未知（老事件无 session、未恢复）按继承保守处理 ⇒ False。
        """
        if not session_id:
            return False
        scope = self.scope_of(agent_id, project_id)
        bound = self.activation.last_session(scope)
        if not bound or bound != session_id:
            return False
        return self._active_ledger(scope) is not None

    def _active_ledger(self, scope: str) -> Ledger | None:
        lid = self.activation.active(scope)
        return self.pool.get(lid) if lid else None

    def _coauthor_note(self, led: Ledger, agent_id: str) -> str:
        """并发可见（2026-08-29 拍板②）：另一写者在窗口内动过这本账本 ⇒ 头部
        一行提示。硬保护是 rev 乐观锁；这行只把并发翻到模型眼前，让它读后再删。
        """
        w = led.last_writer
        if not w or w == agent_id:
            return ""
        window = flag_number("BLADEX_LEDGER_COAUTHOR_WINDOW_S")
        if window <= 0 or not led.updated_at:
            return ""
        try:
            import datetime as _dt
            ts = _dt.datetime.fromisoformat(led.updated_at)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=_dt.timezone.utc)
            age_s = (_dt.datetime.now(_dt.timezone.utc) - ts).total_seconds()
        except ValueError:
            return ""
        if age_s < 0 or age_s > window:
            return ""
        return (f"also being updated by {w} {max(1, int(age_s // 60))} min ago"
                " — re-read before removing entries")

    def _ledger_for(self, context: dict) -> Ledger | None:
        agent_id = context.get("agent_id", "")
        project_id = context.get("project_id", "")
        self._maybe_migrate_binding(agent_id, project_id)
        return self._active_ledger(self.scope_of(agent_id, project_id))

    def _mark_subtask_on_parent(self, parent_id: str, child: Ledger, *,
                                updated_at: str, writer: str = "") -> None:
        """父账本 Next 里留一条带标签的子任务条目（Jason 2026-08-26 拍板）。

        父子**两侧都留痕**才叫"不失联"：子侧的回写义务在它自己的 Next，
        父侧的等待项在这里。父侧条目的 `ref` = 子账本 id ⇒ 完成回写时能对上号。
        这是 BladeX 的**搬运**（把子任务的存在记到父账本上），不是判断——
        判断（这算不算子任务、做完了没有）始终归模型（三红线）。
        """
        parent = self.pool.get(parent_id)
        if parent is None:
            return
        try:
            new = add_entry(
                parent, "next",
                LedgerEntry(text=f"{SUBTASK_TAG} awaiting sub-task "
                                 f"{child.ledger_id}: {child.title or 'untitled'}",
                            source="model", ref=child.ledger_id),
                actor=ACTOR_MODEL, updated_at=updated_at)
        except Exception as e:  # noqa: BLE001 —— 父账本没有 next 段等结构差异不致命
            logger.warning("agency_subtask_parent_mark_failed",
                           parent=parent_id, child=child.ledger_id, error=str(e))
            return
        new = new.model_copy(update={"rev": parent.rev + 1,
                                     "last_writer": writer})
        self.pool[parent_id] = new
        self._emit(AdminEventType.LEDGER_UPDATE, {"ledger": new.model_dump()})

    async def _h_ledger_read(self, args: dict, *, allowed_exposure: str,
                             context: dict) -> str:
        """读账本：缺省=激活账本；给 `ledger_id` 则读那一本（MQ-L9）。

        🔴 为什么要能按 id 读（Jason 2026-08-26 指出）：注入块给的是
        `Other recent ledgers … switch by id if one matches` + **只有标题**的 id 列表，
        而此前本工具 `parameters: {"properties": {}}` —— 只能读激活账本。
        模型要么**盲切**（切了才看得见内容，切错还要再切回来、还撞防抖），要么放弃。
        实测 13 条 `ledger_switch` 里**没有一条**是"从列表里挑中既有账本"的结果——
        **那份列表从来没被用起来过，因为用不了**。
        """
        if not self.ledger_on:
            return "Error: ledger module is disabled."
        wanted = str(args.get("ledger_id", "")).strip()
        if wanted:
            led = self.pool.get(wanted)
            if led is None:
                known = ", ".join(sorted(self.pool)) or "(none)"
                return f"Error: unknown ledger {wanted!r}. Known: {known}"
            return render_ledger_md(led)
        led = self._ledger_for(context)
        if led is None:
            return ("No active ledger. Use bladex_ledger_switch with ledger_id='' "
                    "to create one, or pass ledger_id to read a specific ledger.")
        return render_ledger_md(led)

    async def _h_ledger_update(self, args: dict, *, allowed_exposure: str,
                               context: dict) -> str:
        if not self.ledger_on:
            return "Error: ledger module is disabled."
        led = self._ledger_for(context)
        if led is None:
            return "Error: no active ledger; switch/create one first."
        # rev 乐观锁（2026-08-29）：模型基于旧视图的 remove 会错删、双方互不知情
        # 的写会冲突。带 rev 且落后 ⇒ 拒写 + 返回最新正文让模型合并重试（内循环
        # 一轮内完成，不到达 agent）；不带 rev = 旧调用面，照旧不拦。
        want = args.get("rev")
        if isinstance(want, int) and want != led.rev:
            logger.warning("agency_ledger_rev_conflict", ledger=led.ledger_id,
                           want=want, current=led.rev,
                           writer=str(context.get("agent_id", "")),
                           last_writer=led.last_writer)
            return ("Error: ledger changed since you read it (your rev "
                    f"{want}, current {led.rev}). Latest content below — "
                    "merge your change and retry with the current rev.\n\n"
                    + render_ledger_md(led))
        try:
            import datetime as _dt
            now_iso = _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")
            # updated_at 必须随每次更新走——首个 live 任务实测漏传，账本时间恒为
            # 创建时刻，陈旧兜底（L4）会被它骗成"永远新鲜"。
            new, confirmation = apply_tool_update(
                led, args, updated_at=now_iso,
                # Goal 改动的锚：本轮用户原话。给的是**剥过信封的**那份——
                # 模型引用的是它看得见的用户正文，不是 agent 的机器包装。
                user_text=str(context.get("user_query", "")))
        except LedgerError as e:
            return f"Error: {e}"
        if new.goal_revisions != led.goal_revisions:
            # Goal 改动单独留痕：`goal_revisions` 是给人看的计数，这条是给审计看的
            # 明细（改前/改后/依据）。MQ-L17 教训：看不见的机制等于没有。
            logger.warning("agency_ledger_goal_revised", ledger=new.ledger_id,
                           revision=new.goal_revisions,
                           before=led.goal[:120], after=new.goal[:120],
                           quote=str(args.get("goal_change_quote", ""))[:120])
        new = new.model_copy(update={
            "rev": led.rev + 1,
            "last_writer": str(context.get("agent_id", ""))})
        self.pool[new.ledger_id] = new
        self._turns_since_update[new.ledger_id] = 0   # V-L4：更新即归零
        self._emit(AdminEventType.LEDGER_UPDATE, {"ledger": new.model_dump()})
        # MQ-L38/L40 仪器（C6c，2026-09-04）：候选段说"这句话像任务 X"（top ≥ 0.5），
        # 模型却把产物写进了另一本 ⇒ 错归属的直接计数。与 `ledger_created_despite_match`
        # 成对：候选命中后模型只有三种动作——切到 X（对）/ 新建（那条抓）/ 留在当前本上写
        # （这条抓）。09-04 09:10 病例：泰山句 → 候选 c9ab 1.0 / d50e 0.85，update 落进 v5 本
        # （`bladex_ledger_update` 恒写激活本，无 ledger_id 参数），日志零痕迹。只记不拦——
        # 用户在 A 任务里插一句 B 也会响，那是该响的：读数目标是逐条可解释，不是 0。
        _best = max(self._last_candidates, key=lambda m: m.score, default=None)
        if (_best is not None and _best.score >= _DESPITE_MATCH_SCORE
                and new.ledger_id not in {m.ledger_id for m in self._last_candidates}):
            logger.warning("ledger_update_off_candidate",
                           active=new.ledger_id, candidate=_best.ledger_id,
                           score=_best.score, section=str(args.get("section", "")),
                           agent=str(context.get("agent_id", "")))
        return confirmation

    async def _h_ledger_switch(self, args: dict, *, allowed_exposure: str,
                               context: dict) -> str:
        if not self.ledger_on:
            return "Error: ledger module is disabled."
        session_id = context.get("session_id", "")
        agent_id = context.get("agent_id", "")
        project_id = context.get("project_id", "")
        scope = self.scope_of(agent_id, project_id)
        target = str(args.get("ledger_id", "")).strip()
        created_now = not target          # 本次调用是否新建（防抖豁免判据）
        if not target:
            # 🔴 2026-08-26 拍板（Jason）：Goal 由**模型提炼**，不再原样引用用户原话。
            # 证伪来自 live：jydesignhk 那 5 本账本的 Goal 全都以 agent 自己的
            # `Fully describe and explain everything about this image…` 开头，
            # 用户真正说的话被压在 90 字符之后、5 个 Goal 前缀逐字相同。
            # **原文 ≠ 用户意图**：prompt 习惯各异，啰嗦与机器包装都会把后续轮次带偏。
            #
            # 用户原话仍然存下来（`goal_verbatim`），但**只存不注**——原话本来就在
            # 消息主体里，再注一遍是白花 token；它的用途是本地检索分析 + dashboard 核对。
            #
            # 模型没给 goal ⇒ 回落原话（向后兼容，且"不给就没有 Goal"更糟）。
            # 段结构来自模板（用户改 config/ledger-template.md = 改新账本结构）。
            import datetime as _dt

            from bladex_core.ledger import template_section_order  # noqa: PLC0415

            now_iso = _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")
            # 🔴 2026-08-26 拍板（MQ-L1）：parent **由模型显式给**，不再"把当时
            # 激活的那本无条件记成父"。旧行为把一串**平行**任务串成假父子链——
            # 实测同一网站的 5 个页面评审被链成 8a54→51a8→d7b9→863b 四层，
            # `parent_ledger_id` 既不表达层级也不表达顺序，是纯噪声且随注入传播。
            # 子任务是 LLM 的决策（三红线：BladeX 不做作者），所以它得自己说。
            _parent = str(args.get("parent_ledger_id", "")).strip()
            if _parent and _parent not in self.pool:
                return (f"Error: unknown parent_ledger_id {_parent!r}. "
                        "Omit it if this is not a sub-task.")
            _verbatim = str(context.get("user_query", "")).strip()
            _refined = str(args.get("goal", "")).strip()
            led = new_ledger(title=str(args.get("title", "")).strip() or "New task",
                             goal=_refined or _verbatim,
                             goal_source="model" if _refined else "user",
                             goal_verbatim=_verbatim, created_at=now_iso,
                             parent_ledger_id=_parent,
                             section_order=template_section_order(self.template))
            if not _refined:
                # 可 grep：模型没提炼 ⇒ 回落原话。这条读数决定要不要加强 prompt。
                logger.info("agency_ledger_goal_not_refined",
                            ledger=led.ledger_id, chars=len(_verbatim))
            # MQ-L21 辅助：这本账本是不是从**机器文本**建起来的？
            # 2026-08-27 病例全靠 Jason 肉眼翻账本列表发现，排查花了三小时。
            # 主判据（零自带工具）挡住了当时那两条形态，但**下一个新形态仍会漏**
            # ——客户端的内部功能是会变的。这条不阻止创建（怕误伤），
            # 只让它在日志里可 grep（MQ-L17：没人看得见的问题等于不存在）。
            if (_mark := _machine_text_mark(_verbatim)):
                logger.warning("agency_ledger_created_from_machine_text",
                               ledger=led.ledger_id, agent=agent_id, mark=_mark,
                               title=led.title[:60], head=_verbatim[:80])
            # 模型可带初始条目（它是这几段的作者；Goal 除外）。
            for section in ("core", "open", "next"):
                for text in args.get(section) or []:
                    text = str(text).strip()
                    if text and section in led.section_order:
                        led = add_entry(led, section,
                                        LedgerEntry(text=text, source="model"),
                                        actor=ACTOR_MODEL, updated_at=now_iso)
            if _parent:
                # 子账本的**回写义务**写进它自己的 Next（Jason 2026-08-26 拍板）：
                # 义务由账本自己的注入面每轮携带，不靠模型记性。父侧的对偶条目
                # 由 `_mark_subtask_on_parent` 落在父账本 Next 上。
                led = add_entry(
                    led, "next",
                    LedgerEntry(text=f"{SUBTASK_TAG} on completion: report status and "
                                     f"results back to parent ledger {_parent} "
                                     f"(bladex_ledger_switch to it, then "
                                     f"bladex_ledger_update).",
                                source="model", ref=_parent),
                    actor=ACTOR_MODEL, updated_at=now_iso)
                self._mark_subtask_on_parent(_parent, led, updated_at=now_iso,
                                             writer=agent_id)
                logger.info("agency_ledger_child_created", child=led.ledger_id,
                            parent=_parent, session=session_id, explicit=True)
            # ── V-L5 · 账本↔Matter 绑定（ADR-0032 §4.5）──
            # 建账本这一刻就把锚写死：id 由 ledger_id **确定性派生**，无需与
            # Memory Index 协调（它算出的是同一个 id）。这正是"边界记录"的
            # 形态——锚在任何 fact 被归属之前就存在。
            # 🔴 写在 proxy 侧而非 consolidator：增量 consolidation 用 secondary
            # 模式开 Memory Hub，**写不了管理事件**（ADR-0020 T1）。
            led = bind_matter(led, ledger_anchor_matter_id(led.ledger_id),
                              updated_at=now_iso)
            # MQ-L38 仪器：本轮候选段里有高分同题本却仍建新本 ⇒ 可 grep 告警
            # （"同题重复建本"的直接计数；只记不拦——切不切归模型，三红线）。
            _best = max(self._last_candidates, key=lambda m: m.score, default=None)
            if _best is not None and _best.score >= _DESPITE_MATCH_SCORE:
                logger.warning("ledger_created_despite_match",
                               candidate=_best.ledger_id, score=_best.score,
                               created=led.ledger_id, agent=agent_id)
            self.pool[led.ledger_id] = led
            self._turns_since_update[led.ledger_id] = 0   # V-L4：新账本不陈旧
            self._emit(AdminEventType.LEDGER_CREATE, {"ledger": led.model_dump()})
            target = led.ledger_id
        elif target not in self.pool:
            known = ", ".join(sorted(self.pool)) or "(none)"
            return f"Error: unknown ledger {target!r}. Known: {known}"
        # E0.3（MQ-L41）：`user_requested` 的**唯一生产者**。判据来自本轮用户原话
        # （`context["user_query"]`，已剥信封）是否点名目标账本——按 id 或 title
        # （≥6 字符，防 "任务" 这类短词误命中）。**不接受模型自报**（schema 无该参数，
        # 保持）：豁免若由模型说了算，防抖形同虚设。
        user_requested = False
        if not created_now:
            user_requested, _by = _user_named_ledger(
                str(context.get("user_query", "")), self.pool.get(target))
            if user_requested:
                logger.info("agency_switch_user_requested", ledger=target, by=_by,
                            scope=scope, session=session_id)
        verdict = self.activation.request_switch(
            scope, target, agent_id=agent_id, session_id=session_id,
            project_id=project_id,
            turn_index=int(context.get("turn_index", 0)),
            newly_created=created_now, user_requested=user_requested)
        if not verdict.allowed:
            if verdict.reason == "noop":
                return f"Ledger {target} is already active."
            logger.warning("agency_switch_debounced", scope=scope, session=session_id,
                           target=target, turn_index=int(context.get("turn_index", 0)),
                           last_switch_turn=self.activation.last_switch_turn(scope),
                           last_switch_session=self.activation.last_switch_turn_session(scope),
                           debounce=self.activation.debounce_turns)
            # C6c（2026-09-04）：旧回文 "debounced (switched too recently). Stay on the
            # current ledger" 被模型读成"已经切过了"（09:10 病例：把"switch 回同题账本 ✅"
            # 写成 Verified，实际 5 次全拒），且"stay on the current ledger"直接把它引向
            # 错本去写。改成明确否定态 + 说清还差几轮；不改判定。
            _cur = self._active_ledger(scope)
            _cur_id = _cur.ledger_id if _cur else ""
            _cur_title = (_cur.title or "") if _cur else ""
            _last = self.activation.last_switch_turn(scope)
            _ti = int(context.get("turn_index", 0))
            _wait = (max(0, self.activation.debounce_turns - (_ti - _last))
                     if _last is not None else self.activation.debounce_turns)
            return (f"NOT switched (debounce): the active ledger is still {_cur_id} "
                    f"({_cur_title!r}). Do NOT record the new task's progress into it "
                    "and do not report the switch as done. Answer the user without "
                    f"ledger updates this turn; switching to {target} will be allowed "
                    f"after {_wait} more turn(s).")
        # scope 争用告警：project 信号未落地前，同一 agent 的**并发**会话共用一个
        # scope（两个 CC 窗口开在不同仓库 ⇒ 互相切走对方的账本）。**不静默**——
        # 刚性原则 9 的"硬约束可以推翻静态配置，但必须留可 grep 的告警"同款。
        #
        # 🔴 判据必须带"**并发**"（2026-08-26 live 误报）：首版只判"换了 session"，
        # 结果 Pi 的两轮**先后**会话（一轮 `tw:` 时间窗兜底、一轮 `fp:` 指纹）
        # 也触发了告警——同一个 agent 连续工作换个 session_id 是常态，不是争用。
        # 真争用的形状是"**几乎同时**从两个 session 切"，所以加时间窗。
        # 误报的代价不是吵：真争用发生时会被淹在噪声里。
        _contention_window_s = 300.0
        if (verdict.prev_session and session_id
                and verdict.prev_session != session_id
                and 0 <= verdict.prev_age_s <= _contention_window_s):
            logger.warning("agency_activation_scope_contention", scope=scope,
                           prev_session=verdict.prev_session, session=session_id,
                           prev_age_s=round(verdict.prev_age_s, 1), target=target,
                           hint="same (agent, project) scope switched from another "
                                "session; project signal extraction is V-F2")
        # ③ 事件载荷补 `ledger_id`（2026-08-25 live：switch 事件的 ledger_id 为空，
        # 重放靠 to_ledger_id 能恢复但审计读起来是空的——两个字段同源同值，冗余无害）。
        self._emit(AdminEventType.LEDGER_SWITCH,
                   {**verdict.event_payload, "ledger_id": target})
        return f"Active ledger is now {target}."


# ── 流式拦截（chat 协议；V-P5a 只做 chat，anthropic/responses 归 V-P5b）────────



async def intercept_chat_stream(
    sse_iter: Any, *, agency: AgencyRuntime, capture_result: Any,
    upstream_messages: list[dict], session_prefix: str, allowed_exposure: str,
    call_llm: Any, session_id: str, agent_id: str, project_id: str = "",
):
    """包装 capture_stream 的 SSE 流：拦 bladex_* 调用、透传其余。

    行为矩阵（与非流式 process_message 同语义）：
      无 bladex 调用   → 逐字透传（含终止 chunk 与 [DONE]）
      mixed           → 剥离转发；流末执行己方工具 + 记拼接；终止 chunk 原样放行
      纯 bladex       → 全部吞下；流末跑内循环（期间发 SSE 注释 keepalive，
                        D4 实测五 agent 全部接受），把最终消息合成为 chunk 发出
                        ——绝不给 agent 一个空流（hermes 重试指纹）。
    """
    import asyncio
    import json as _json

    from bladex_proxy.interception import ChatStreamStripper

    stripper = ChatStreamStripper()
    held_terminal: list[str] = []      # finish chunk 与 [DONE]，流末按分流决定
    #: 🔴 两个信号必须分开（2026-08-25 live 事故）：`finish_reason=tool_calls`
    #: 的合法性只取决于**转发出去的调用数**，与有没有正文无关。首版用一个
    #: `forwarded_substance`（正文或调用任一为真）判 mixed，于是"有正文 +
    #: 调用全被拦"那轮把 `finish_reason=tool_calls` 原样放行，agent 收到
    #: 「indicated a tool call but none was included」并重试（实测 4 轮）。
    forwarded_calls = False           # 有 agent 自己的调用被转发出去
    forwarded_text = False            # 有正文被转发出去
    last_meta = {"id": "bladex-intercept", "model": ""}
    _last_out = [time.monotonic()]   # V-P3：上次给客户端发字节的时刻

    def _reser(chunk: dict) -> str:
        return f"data: {_json.dumps(chunk, ensure_ascii=False)}\n\n"

    async for sse in sse_iter:
        data = sse.strip()
        if not data.startswith("data: "):
            yield sse
            continue
        payload = data[len("data: "):]
        if payload == "[DONE]":
            held_terminal.append(sse)
            continue
        try:
            chunk = _json.loads(payload)
        except ValueError:
            yield sse
            continue
        last_meta["id"] = chunk.get("id", last_meta["id"])
        last_meta["model"] = chunk.get("model", last_meta["model"])
        choices = chunk.get("choices") or []
        finish = choices[0].get("finish_reason") if choices else None
        delta = (choices[0].get("delta") or {}) if choices else {}
        if delta.get("content"):
            forwarded_text = True
        if finish:
            held_terminal.append(sse)     # 终止 chunk 押后：分流未定
            continue
        _emitted = False
        for out in stripper.feed(chunk):
            if (out.get("choices") or [{}])[0].get("delta", {}).get("tool_calls"):
                forwarded_calls = True
            _emitted = True
            _last_out[0] = time.monotonic()
            yield _reser(out)
        # 剥空 ⇒ 这个 chunk 对客户端不可见。久无输出就发一行 SSE 注释（V-P3）。
        if not _emitted and _idle_too_long(_last_out):
            _last_out[0] = time.monotonic()
            yield _SSE_KEEPALIVE.decode()

    removed = stripper.removed_calls
    if not removed:
        for sse in held_terminal:
            yield sse
        return

    ctx = tool_context(session_id=session_id, agent_id=agent_id,
                       project_id=project_id, upstream_messages=upstream_messages)
    # 🔴 2026-08-26：判据只看 `forwarded_calls`，**不看 forwarded_text**
    # （与 `classify_message` 同一修正；病例见那里的 docstring）。
    # 正文已经流给 agent 了，但"说了一段前言"不代表这一轮结束——模型明说
    # "先归档，再给结论"，得让它拿着工具结果继续。已流出的正文不丢：作为
    # `initial_content` 进内循环的 assistant 消息，模型看得到自己刚说过什么，
    # 续写才连贯；用户侧则是同一条 SSE 流里前言之后接上结论。
    if forwarded_calls:
        # MIXED：agent 侧仍有自己的调用要执行，流后补拼接。
        # 🔴 终止 chunk 必须**按转发出去的内容重写** finish_reason：
        # 上游给的是 `tool_calls`（它算上了被我们拦掉的 bladex 调用），
        # 而 agent 只看到剩下的部分——转发调用为 0 时必须改成 `stop`，
        # 否则 agent 判"说了有工具调用却没有"并重试（live 实测 4 轮）。
        for sse in held_terminal:
            if not forwarded_calls and sse.startswith("data: ") and "[DONE]" not in sse:
                try:
                    _c = _json.loads(sse[len("data: "):])
                    _ch = (_c.get("choices") or [{}])[0]
                    if _ch.get("finish_reason") == "tool_calls":
                        _ch["finish_reason"] = "stop"
                        logger.info("agency_finish_reason_rewritten",
                                    session=session_id, removed=len(removed))
                        yield _reser(_c)
                        continue
                except ValueError:
                    pass
            yield sse
        # MQ-P8：tool_calls 取剥离器的**已发射形态**（含铸 id），不取 capture——
        # capture 看到的是上游原始 delta（deepseek 实测可无 id），agent 回传的
        # 是 wire 上的形态；两者不同源 = 锚哈希级永不相等（拼接恢复 0%）。
        stripped_assistant = {"role": "assistant",
                              "content": capture_result.full_text or "",
                              "tool_calls": list(stripper.forwarded_calls)}
        if not stripped_assistant["tool_calls"]:
            stripped_assistant.pop("tool_calls")
        results = []
        for tc in removed:
            fn = tc.get("function") or {}
            text = await agency.toolface.dispatch(
                fn.get("name", ""), fn.get("arguments", "{}"),
                allowed_exposure=allowed_exposure, context=ctx)
            results.append({"role": "tool", "tool_call_id": tc.get("id", ""),
                            "content": text})
        agency.splice.add(SpliceRecord(
            session_prefix=session_prefix, anchor=anchor_key(stripped_assistant),
            calls=removed, results=results, created_ms=int(time.time() * 1000)))
        logger.info("agency_stream_mixed_stripped", removed=len(removed),
                    names=_call_names(removed))
        return

    # PURE：内循环 + keepalive，最终消息合成为 chunk。
    logger.info("agency_stream_pure_loop", calls=len(removed),
                preamble_chars=len(capture_result.full_text or ""))
    _q: asyncio.Queue = asyncio.Queue()
    loop_task = asyncio.ensure_future(run_inner_loop(
        messages=upstream_messages, initial_calls=removed, call_llm=call_llm,
        initial_content=capture_result.full_text or "", on_progress=_q.put,
        dispatch=lambda n, a: agency.toolface.dispatch(
            n, a, allowed_exposure=allowed_exposure, context=ctx)))
    # V-P3 硬点 8：播报即 keepalive；没有播报的间隙才发注释行。
    _chunks = 0
    _narrator = _ChatNarrator() if _narrate_on() else None
    async for _out in _narrate_until_done(loop_task, _q, narrator=_narrator,
                                          keepalive=": bladex-keepalive\n\n"):
        _chunks += 1
        _last_out[0] = time.monotonic()
        yield _out
    if _chunks:
        logger.info("agency_stream_narrated", phase="inner_loop", chunks=_chunks)
    result: InnerLoopResult = loop_task.result()
    agency.loop_ledger.record(session_prefix, result)
    # 🔴 MQ-L45（2026-09-05）：内循环的最终回复可能是 MIXED（模型先 read、再同轮
    # switch + 自己的工具）。此前这里直接 `final.get("tool_calls")` 全量转发，
    # `bladex_*` 因此被 agent 当自己的工具执行 ⇒ `Tool bladex_ledger_switch not found`
    # （Hub 取证 09-05 12:12 / 13:29；agent 由此判定"BladeX 工具不可用"并转去翻文件系统）。
    # 剥离 + 就地执行 + 记拼接，与首条回复的 MIXED 处置**同一实现点**。
    final, _leaked, _extra = await agency._strip_execute_splice(
        result.final_message, ctx=ctx, session_prefix=session_prefix,
        allowed_exposure=allowed_exposure)
    text = str(final.get("content") or "")
    base = {"id": last_meta["id"], "object": "chat.completion.chunk",
            "created": int(time.time()), "model": last_meta["model"]}
    yield _reser({**base, "choices": [{"index": 0, "delta": {"content": text},
                                       "finish_reason": None}]})
    # 剥离之后剩下的才是 agent 自己的调用——以聚合形态一次性发出（少见但合法）。
    agent_calls = [tc for tc in final.get("tool_calls") or []]
    if agent_calls:
        yield _reser({**base, "choices": [{"index": 0,
                                           "delta": {"tool_calls": [
                                               {**tc, "index": i}
                                               for i, tc in enumerate(agent_calls)]},
                                           "finish_reason": None}]})
    yield _reser({**base, "choices": [{"index": 0, "delta": {},
                                       "finish_reason": "tool_calls" if agent_calls
                                       else "stop"}]})
    yield "data: [DONE]\n\n"
    # 🔴 `agent` / `final_tool_calls` 是 MQ-L23 判据字段（2026-08-31 补齐）。
    #: 转发本身 chat 路径一直是对的（MQ-L23 修的是另外两条协议），但**读数器
    #: 只认 `agency_protocol_inner_loop_done`** ⇒ 走 chat 的内循环
    #: （08-30 实测 10 次）在签收里一次都没被量过。缺的不是能力是**可观测性**，
    #: 与 MQ-P3「每个零件都对，装起来漏一环没人发现」同族：
    #: 这次是「三条路都对，只有两条被量了」。
    # MQ-L45：`final_bladex_stripped` / `final_call_names` 是本条的判据字段——
    # 修前 chat 路只记数量不记名字，16 次 `final_tool_calls>0` 里哪几次是泄漏读不出来。
    if _leaked:
        logger.info("agency_inner_loop_final_stripped",
                    agent=agent_id, count=len(_leaked), names=_call_names(_leaked))
    logger.info("agency_stream_loop_done", rounds=len(result.rounds),
                agent=agent_id, final_tool_calls=len(agent_calls),
                final_bladex_stripped=len(_leaked),
                final_call_names=_call_names(agent_calls),
                final_text_len=len(text),
                degraded=result.degraded, elapsed_s=round(result.elapsed_s, 1),
                **_loop_cost(result))


# ── V-P3：内循环 keepalive（ADR-0032 §3.2）───────────────────────────────────
#
# 🔴 2026-08-27 live 事故：Codex 首轮就调 `bladex_ledger_switch` + `bladex_memory_search`
# ⇒ 纯 bladex ⇒ 走内循环。而**两段都是静默的**：
#   ① 上游流的 603 个 chunk 全是 bladex_* 调用的 delta，被 stripper 剥掉不转发
#      —— 19.4 秒零字节；
#   ② 内循环本身 3.7 秒，生成器挂在 `await` 上什么也不 yield。
# 合计 23 秒无输出，`ms_total=25300` ⇒ Codex CLI 判超时断开
# （日志 `enqueue_shielded_on_disconnect`）。**内循环跑成功了，人没等到。**
#
# 为什么用 **SSE 注释行**而不是协议心跳事件（`ping` / `response.in_progress`）：
# 注释行在三个协议里都是合法且**语义惰性**的——任何符合规范的客户端读到它只会
# 重置空闲计时器，不会进状态机。心跳事件则各协议一套、且有污染客户端状态的风险。
# 一行注释买到"不静默"，代价是零。
#
# 为什么之前没撞到：Hermes 多数轮是混合调用（剥离-拼接，客户端一直有输出）；
# CC 此前 100% 被误判 aux（MQ-A19）根本拿不到工具面，走不到内循环。
# Codex 是第一个"首轮纯 bladex + 长上游流"的组合。MQ-A19 修完后 CC 也会走这条路。

_SSE_KEEPALIVE = b": bladex-keepalive\n\n"


# ── 硬点 8：拦截播报（ADR-0032 §3.2 #8；实测矩阵 survey §5c）────────────────
#
# 内循环期间不只发惰性字节，而是**播报 BladeX 在做什么**（"calling
# bladex_memory_search…"），走 reasoning/thinking 流。D7 实测各 agent 的收益分档：
#   codex 过程可见+收尾折叠（最理想）｜ Pi / dsh 全量渲染｜
#   CC 折叠成 "Thought for Xs" 计时条（半值）｜ hermes 不渲染（退化为 keepalive，零值）
#
# 🔴 **gate：入站剥离先行**（已由 `splice.strip_inbound_echoes` 满足）。
# CC/Pi/dsh 会把播报**回带**（messages 协议 thinking 块与 DeepSeek 系
# reasoning_content 本就要求客户端回传）。播报文本一律以 `NARRATE_MARK` 开头，
# 入站按标记剥净 ⇒ **播报只存在于 BladeX↔agent 之间，LLM 与真上游永远看不到**
# （三红线之 3：模型不调用时零行为差异）。
# 无剥离不许开播报——所以下面这三个发射器都不带"要不要剥"的开关，
# 剥离是 `strip_inbound_echoes` 的无条件行为，不是可选项。


#: 🔴 播报的正确形态**仓库里早就有**：`scripts/agent_probe_upstream.py`
#: （V-R1 D7 的 mock 上游，2026-08-25 对五个 agent 实测过）。
#: 初版我只抄了事件名、把序列和字段全丢了——发的是**不属于任何 item 的裸 delta**：
#:
#:   我写的： {"type": "response.reasoning_summary_text.delta", "delta": …}
#:   验过的： output_item.added(reasoning item) → delta(item_id/output_index/
#:            summary_index 齐全) → output_item.done
#:
#: 同一天第三次栽在"没先找现成的"上。D7 实测读数（survey §5c）：
#:   codex 过程实时可见+收尾折叠（最理想）｜ Pi/dsh 全量渲染 ｜
#:   CC 折叠成计时条 ｜ hermes 不渲染（自然退化为 keepalive）
#:
#: 播报是**有状态**的：开一次 item/block、多次 delta、收一次尾。
#: 故用类而不是纯函数——裸函数发不出"开/收"，那正是初版坏掉的根因。


class _Narrator:
    """协议对应的播报发射器。`open()/delta()/close()` 三段，调用方按序发。

    `index` 由调用方给（避开上游已用的编号），`closed` 幂等——
    内循环可能异常退出，收尾必须能安全重复调用。
    """

    def __init__(self, index: int = 90) -> None:
        self.index = index
        self._opened = False
        self._closed = False

    def open(self) -> list:      # noqa: D102 —— 子类实现
        return []

    def delta(self, text: str) -> list:  # noqa: D102
        return []

    def close(self) -> list:     # noqa: D102
        return []

    def items_used(self) -> int:
        """本轮播报**实际占用**了几个 output item 编号（没开口就是 0）。

        🔴 后续合成消息必须跳过这些编号。2026-08-27 实测：播报与合成都用
        `max_index + 1` ⇒ 官方解析器在 `output_text.delta` 处
        `assert output.type == "message"` 失败（那个下标上坐着 reasoning item）
        ⇒ 断连。与 08-25 `output_index=99` 事故同型：**编号是契约，不是装饰**。
        """
        return 1 if self._opened else 0

    def snapshot_item(self) -> dict | None:
        """交给 `response.completed` 快照的 item（没开口 ⇒ None）。

        流里发过的 item 必须在终止快照里出现，否则客户端对账失败——
        08-25 Codex `client disconnected` 就是这条。
        """
        return None


def _ev(name: str, obj: dict) -> bytes:
    import json as _json
    return (f"event: {name}\ndata: " + _json.dumps(obj, ensure_ascii=False)
            + "\n\n").encode()


class _ResponsesNarrator(_Narrator):
    """responses：reasoning item + summary delta（codex 的理想形态）。"""

    _ITEM_ID = "rs_bladex"

    def open(self):
        self._opened = True
        return [_ev("response.output_item.added", {
            "type": "response.output_item.added", "output_index": self.index,
            "item": {"type": "reasoning", "id": self._ITEM_ID, "summary": []}})]

    def delta(self, text: str):
        if not self._opened:
            return []
        return [_ev("response.reasoning_summary_text.delta", {
            "type": "response.reasoning_summary_text.delta",
            "item_id": self._ITEM_ID, "output_index": self.index,
            "summary_index": 0, "delta": text + "\n"})]

    def close(self):
        if not self._opened or self._closed:
            return []
        self._closed = True
        return [_ev("response.output_item.done", {
            "type": "response.output_item.done", "output_index": self.index,
            "item": self._item()})]

    def _item(self) -> dict:
        return {"type": "reasoning", "id": self._ITEM_ID,
                "summary": [{"type": "summary_text", "text": "BladeX memory work"}]}

    def snapshot_item(self) -> dict | None:
        return self._item() if self._opened else None


class _AnthropicNarrator(_Narrator):
    """messages：thinking block（CC 折叠成计时条，仍可感知）。"""

    def open(self):
        self._opened = True
        return [_ev("content_block_start", {
            "type": "content_block_start", "index": self.index,
            "content_block": {"type": "thinking", "thinking": ""}})]

    def delta(self, text: str):
        if not self._opened:
            return []
        return [_ev("content_block_delta", {
            "type": "content_block_delta", "index": self.index,
            "delta": {"type": "thinking_delta", "thinking": text + "\n"}})]

    def close(self):
        if not self._opened or self._closed:
            return []
        self._closed = True
        return [_ev("content_block_stop", {
            "type": "content_block_stop", "index": self.index})]


class _ChatNarrator(_Narrator):
    """chat/completions：reasoning_content delta（DeepSeek 系形态）。

    OpenAI chunk 无 item 概念，open/close 为空——**这正是我初版误以为
    三协议都这样的来源**。
    """

    def delta(self, text: str):
        import json as _json
        self._opened = True
        return ["data: " + _json.dumps(
            {"id": "bladex-narrate", "object": "chat.completion.chunk",
             "choices": [{"index": 0, "delta": {"reasoning_content": text + "\n"},
                          "finish_reason": None}]},
            ensure_ascii=False) + "\n\n"]


def _idle_too_long(last_out: list) -> bool:
    """距上次给客户端发字节是否已超过心跳间隔。`last_out` 是单元素可变槽。"""
    interval = _keepalive_interval_s()
    return interval > 0 and (time.monotonic() - last_out[0]) >= interval


def _narrate_on() -> bool:
    """播报开关。关掉 = 只发 SSE 注释行心跳（2026-08-27 已跑过真实流量的形态）。"""
    from bladex_core.flags import flag_enabled
    return flag_enabled("BLADEX_NARRATE_INTERCEPT")


def _keepalive_interval_s() -> float:
    """心跳间隔（秒）。0 = 关（回归通道）。"""
    return flag_number("BLADEX_STREAM_KEEPALIVE_S")


async def _narrate_until_done(task: Any, q: Any, *, narrator: Any, keepalive: Any):
    """驱动内循环，边等边产出播报；久无播报时补心跳。异步生成器，调用方 `async for`。

    🔴 **播报是有状态的**：先 `open()` 声明 item/block，之后 delta 才有归属，
    结束必须 `close()`。初版发裸 delta（不属于任何 item），协议上是非法的——
    正确形态见 `scripts/agent_probe_upstream.py`（V-R1 D7 实测过的 mock 上游）。

    `narrator=None` ⇒ 只发心跳（`BLADEX_NARRATE_INTERCEPT=0` 的形态，
    那一版当天跑过真实流量）。

    三条道同时等：**有播报** / **循环结束** / **一拍超时**。播报即 keepalive
    （ADR-0032 §3.2 #8"兼作 keepalive"），心跳只在两次播报间隔过长时兜底。
    """
    interval = _keepalive_interval_s()
    opened = False
    while True:
        getter = asyncio.ensure_future(q.get())
        done, _ = await asyncio.wait({task, getter}, timeout=interval or None,
                                     return_when=asyncio.FIRST_COMPLETED)
        if getter in done:
            text = getter.result()
            if narrator is not None:
                if not opened:
                    opened = True
                    for chunk in narrator.open():
                        yield chunk
                for chunk in narrator.delta(text):
                    yield chunk
            elif keepalive is not None:
                yield keepalive          # 播报关：退回心跳
            continue
        getter.cancel()
        if task in done:
            while not q.empty():          # 收尾：还没取走的播报
                text = q.get_nowait()
                if narrator is not None and opened:
                    for chunk in narrator.delta(text):
                        yield chunk
                elif narrator is None and keepalive is not None:
                    yield keepalive
            # 🔴 收尾在**正常返回路径**上，不能放 `finally`：
            # 消费方 `aclose()`（客户端断连）会在 yield 点抛 `GeneratorExit`，
            # 而 `finally` 里再 yield 就是 `RuntimeError: async generator
            # ignored GeneratorExit` —— 初版就是这么写的，测试当场抓出来。
            # 而且那种情况**本来也不需要收尾**：客户端都断了，发给谁？
            if narrator is not None and opened:
                for chunk in narrator.close():
                    yield chunk
            return
        if keepalive is not None:
            yield keepalive


# ── V-P5b：anthropic / responses 流式拦截（复用 V-P2 的两个剥离器）──────────
#
# 两协议的生成器都产出 `event: X\ndata: {json}\n\n` 的 bytes，故一套 parse-feed-
# reserialize 通吃；差别只在剥离器与"终止事件"的判据，用参数注入。
#
# 🔴 与 chat 端点的语义差（不是简化，是协议事实）：
# - anthropic/responses 的终止事件（message_delta / response.completed）携带
#   stop_reason，同样要按**转发出去的调用数**改写（chat 端点 live 事故同型）；
# - 纯 bladex 调用（剥完空流）在这两个协议里同样不可发空流，走内循环合成。
#   V-P5b 首版：内循环仅在 chat 端点启用（call_llm 需按协议构造），
#   两协议先只做**剥离-拼接**（mixed），纯 bladex 走"合成文本事件"降级——
#   这保证不会给 agent 一个空流，代价是那轮不做多轮内循环（记 V-P5c）。


async def _intercept_protocol_stream(
    sse_iter: Any, *, agency: AgencyRuntime, stripper: Any, capture_result: Any,
    session_prefix: str, allowed_exposure: str, session_id: str, agent_id: str,
    upstream_messages: list[dict], terminal_types: tuple[str, ...],
    project_id: str = "",
    rewrite_stop: Any, synth_text_events: Any, narrate: Any, call_llm: Any = None,
    synth_tool_call_events: Any = None,
):
    """anthropic / responses 共用的流式拦截。`stripper` = 对应协议的剥离器。"""
    import json as _json

    held: list[bytes] = []
    forwarded_calls = False
    forwarded_text = False
    _last_out = [time.monotonic()]   # V-P3：上次给客户端发字节的时刻
    #: 合成 item 的落点与 index（responses 协议：与终止快照同源，见 _synth_* 注释）
    synth_items: list = []
    max_index = -1

    async for raw in sse_iter:
        chunk = raw if isinstance(raw, bytes) else str(raw).encode()
        text = chunk.decode("utf-8", "replace")
        etype = ""
        payload: dict = {}
        for line in text.splitlines():
            if line.startswith("event: "):
                etype = line[len("event: "):].strip()
            elif line.startswith("data: "):
                try:
                    payload = _json.loads(line[len("data: "):])
                except ValueError:
                    payload = {}
        if not etype or not payload:
            yield chunk
            continue
        if etype in terminal_types:
            held.append(chunk)          # 终止事件押后：分流未定
            continue
        _emitted = False
        for out in stripper.feed({**payload, "type": payload.get("type", etype)}):
            if isinstance(out.get("output_index"), int):
                max_index = max(max_index, out["output_index"])
            t = out.get("type", "")
            if "text" in t or t.endswith("output_text.delta"):
                forwarded_text = True
            if "tool_use" in _json.dumps(out) or "function_call" in t:
                forwarded_calls = True
            _emitted = True
            _last_out[0] = time.monotonic()
            yield f"event: {t or etype}\ndata: {_json.dumps(out, ensure_ascii=False)}\n\n".encode()
        # 剥空 ⇒ 客户端看不到这个 chunk。久无输出就发心跳（V-P3；live 事故里
        # 603 个 chunk 全被剥掉 = 19.4 秒零字节）。
        if not _emitted and _idle_too_long(_last_out):
            _last_out[0] = time.monotonic()
            yield _SSE_KEEPALIVE

    removed = stripper.removed_calls
    if not removed:
        for h in held:
            yield h
        return

    ctx = tool_context(session_id=session_id, agent_id=agent_id,
                       project_id=project_id, upstream_messages=upstream_messages)

    # 同上：正文不参与"这一轮完没完"的判定（2026-08-26 修正）。
    if not forwarded_calls and call_llm is not None:
        # ── V-P5c：纯 bladex 走**真正的内循环**（CC live 实证：首轮就调 bladex
        # 工具是主路径，合成降级会把模型的思路打断在第一步——它再也走不到
        # `bladex_ledger_switch`，账本机制在 CC 上永远起不来）。
        # V-P3：内循环期间生成器本来挂在 await 上什么也不 yield ⇒ 客户端静默
        # （live 事故：+3.7 秒）。改成边等边 yield 心跳。
        # V-P3 硬点 8：内循环期间播报 BladeX 在做什么（走 reasoning/thinking 流），
        # 没有播报的间隙用心跳兜底。两者都必须**边等边出**，否则等于没有。
        _q: asyncio.Queue = asyncio.Queue()
        _task = asyncio.ensure_future(run_inner_loop(
            messages=upstream_messages, initial_calls=removed, call_llm=call_llm,
            on_progress=_q.put,
            dispatch=lambda n, a: agency.toolface.dispatch(
                n, a, allowed_exposure=allowed_exposure, context=ctx)))
        _beats = 0
        # index 避开上游已用的编号（max_index 一路在跟踪）。
        _narrator = narrate(max_index + 1) if _narrate_on() else None
        async for _out in _narrate_until_done(_task, _q, narrator=_narrator,
                                              keepalive=_SSE_KEEPALIVE):
            _beats += 1
            _last_out[0] = time.monotonic()
            yield _out
        if _beats:
            logger.info("agency_stream_narrated", phase="inner_loop", chunks=_beats)
        result = _task.result()
        agency.loop_ledger.record(session_prefix, result)
        # 🔴 MQ-L45（2026-09-05）：下面那段注释里"先量再修"的那件事，现在修了——
        # 内循环的最终回复若是 MIXED，`bladex_*` 必须先剥离执行，不能随 agent 调用一起转发。
        final, _leaked, _extra = await agency._strip_execute_splice(
            result.final_message, ctx=ctx, session_prefix=session_prefix,
            allowed_exposure=allowed_exposure)
        if _leaked:
            logger.info("agency_inner_loop_final_stripped",
                        agent=agent_id, count=len(_leaked), names=_call_names(_leaked))
        text = str(final.get("content") or "")
        # 🔴 记账缺口（同批修）：内循环产出的文本模型看得见 ⇒ 必须进 capture，
        # 否则 Hub 那轮 response_text 为空 =「model-visible ⟺ logged」破了。
        try:
            capture_result.full_text = (capture_result.full_text or "") + text
        except Exception:  # noqa: BLE001,S110
            pass
        # 🔴 播报若开了 item，它占掉一个编号，合成消息必须往后让。
        #: 并把播报 item 送进终止快照——流里发过的 item 不在 `response.completed`
        #: 的 output 里，客户端对账失败即断连（08-25 事故）。
        _used = _narrator.items_used() if _narrator else 0
        if _narrator is not None:
            _snap = _narrator.snapshot_item()
            if _snap is not None and synth_items is not None:
                synth_items.append(_snap)
        for ev in synth_text_events([{"content": text}], raw=True,
                                    index=max_index + 1 + _used, sink=synth_items):
            yield ev
        # 🔴 判据埋点（2026-08-27 MQ-L23）：内循环的最终回复若带 agent 自己的
        #: tool_calls，我们**只转发了 content**，调用被丢弃 ⇒ 客户端看到一个
        #: 无工具调用的助手消息 = 这回合已完成 ⇒ 收工。
        #: ~~`innerloop` 的契约明写「mixed 的剥离-拼接归调用方」，而这条路径没做。
        #: 先量再修~~ —— **2026-09-05 MQ-L45 已修**：`final` 在上面已过
        #: `_strip_execute_splice`，到这里只剩 agent 自己的调用。留着这段是为了记住
        #: 教训：「先量再修」当时把仪器装在了**这条**路上，而实际漏的是 chat 流式那条
        #: （那里连调用名都不记）。09-05 靠 agent 侧的 "Tool not found" 才浮出来——
        #: 仪器装错了路径，等于没装。
        _final_tcs = final.get("tool_calls") or []
        if _final_tcs and synth_tool_call_events is not None:
            for ev in synth_tool_call_events(
                    _final_tcs, index=max_index + 2 + _used, sink=synth_items):
                yield ev
            logger.info("agency_inner_loop_final_calls_forwarded",
                        agent=agent_id, count=len(_final_tcs),
                        names=[(t.get("function") or {}).get("name", "?")
                               for t in _final_tcs][:6])
        elif _final_tcs:
            logger.warning("agency_inner_loop_final_calls_dropped",
                           agent=agent_id, count=len(_final_tcs))
        logger.info("agency_protocol_inner_loop_done", removed=len(removed),
                    agent=agent_id, rounds=len(result.rounds),
                    degraded=result.degraded, final_tool_calls=len(_final_tcs),
                    final_text_len=len(text),
                    elapsed_s=round(result.elapsed_s, 1), **_loop_cost(result))
        # 🔴 转发了 agent 调用 ⇒ 终止事件必须保留 "还有工具要执行" 的语义
        #: （anthropic 保 `stop_reason=tool_use`，responses 保快照里的
        #: function_call item）。写死 False 会把它改成"回合结束"。
        for h in held:
            yield rewrite_stop(h, bool(_final_tcs), synth_items, stripper=stripper)
        return

    # 执行己方工具（mixed：剥离-拼接；无 call_llm 时的纯 bladex：合成降级）
    results = []
    for tc in removed:
        fn = tc.get("function") or {}
        out_text = await agency.toolface.dispatch(
            fn.get("name", ""), fn.get("arguments", "{}"),
            allowed_exposure=allowed_exposure, context=ctx)
        results.append({"role": "tool", "tool_call_id": tc.get("id", ""),
                        "content": out_text})

    if not forwarded_calls:
        # 无 call_llm 的兜底：绝不发空流——把结果合成为文本事件。
        for ev in synth_text_events(results, index=max_index + 1,
                                    sink=synth_items):
            yield ev
        logger.info("agency_protocol_pure_synthesized", removed=len(removed),
                    agent=agent_id)
    else:
        # MQ-P8：同 chat 侧——tool_calls 取剥离器已发射形态（含铸 id）。
        stripped_assistant = {"role": "assistant",
                              "content": capture_result.full_text or "",
                              "tool_calls": list(stripper.forwarded_calls)}
        if not stripped_assistant["tool_calls"]:
            stripped_assistant.pop("tool_calls")
        agency.splice.add(SpliceRecord(
            session_prefix=session_prefix, anchor=anchor_key(stripped_assistant),
            calls=removed, results=results, created_ms=int(time.time() * 1000)))
        logger.info("agency_protocol_mixed_stripped", removed=len(removed),
                    agent=agent_id, names=_call_names(removed))

    for h in held:
        yield rewrite_stop(h, forwarded_calls, synth_items, stripper=stripper)


def _rewrite_stop_anthropic(chunk: bytes, forwarded_calls: bool,
                            synth_items: list | None = None, stripper: Any = None) -> bytes:
    """message_delta 的 stop_reason：转发调用为 0 时 tool_use → end_turn。"""
    import json as _json
    if forwarded_calls:
        return chunk
    text = chunk.decode("utf-8", "replace")
    if "tool_use" not in text:
        return chunk
    try:
        head, _, data = text.partition("data: ")
        obj = _json.loads(data)
    except ValueError:
        return chunk
    d = obj.get("delta") or {}
    if d.get("stop_reason") == "tool_use":
        d["stop_reason"] = "end_turn"
        obj["delta"] = d
        logger.info("agency_stop_reason_rewritten", protocol="anthropic")
        return (head + "data: " + _json.dumps(obj, ensure_ascii=False) + "\n\n").encode()
    return chunk


def _rewrite_stop_responses(chunk: bytes, forwarded_calls: bool,
                            synth_items: list | None = None,
                            stripper: Any = None) -> bytes:
    """responses 的 `response.completed` 携带**整份 output 快照**——里面还有
    被我们拦掉的 function_call item（2026-08-25 测试当场抓到的泄漏）。
    终止事件必须同样过滤：剔除 bladex_ 开头的 item 并重排 output_index。"""
    import json as _json
    text = chunk.decode("utf-8", "replace")
    _has_minted = bool(getattr(stripper, "_minted_items", None))
    if "bladex_" not in text and not synth_items and not _has_minted:
        return chunk
    try:
        head, _, data = text.partition("data: ")
        obj = _json.loads(data)
    except ValueError:
        return chunk
    resp = obj.get("response") or {}
    items = resp.get("output") or []
    kept = [it for it in items
            if not str((it or {}).get("name", "")).startswith("bladex_")]
    dropped = len(items) - len(kept)
    # 🔴 合成 item 必须补进快照（否则流内有、快照无 ⇒ Codex 对账失败断连）。
    added = 0
    for it in synth_items or []:
        if not any((k or {}).get("id") == it.get("id") for k in kept):
            kept.append(it)
            added += 1
    # MQ-P8：流内 function_call item 带的是铸的 call_id，快照必须同源
    # （流/快照分叉 = Codex 对账断连，08-25 事故同族）。
    stamped = (stripper.stamp_snapshot_items(kept)
               if hasattr(stripper, "stamp_snapshot_items") else 0)
    if not dropped and not added and not stamped:
        return chunk
    resp["output"] = kept
    obj["response"] = resp
    logger.info("agency_responses_snapshot_filtered", dropped=dropped, added=added,
                stamped=stamped)
    return (head + "data: " + _json.dumps(obj, ensure_ascii=False) + "\n\n").encode()


#: 🔴 MQ-L23（2026-08-27）：内循环的最终回复可能带 **agent 自己的** tool_calls
#: （`innerloop` 契约明写「mixed 的剥离-拼接归调用方」）。chat 协议做了
#: （`intercept_chat_stream` 里 agent_calls 那段），**responses / anthropic 没做**
#: ⇒ 调用被丢弃 ⇒ 客户端收到一个无工具调用的助手消息 = 这回合完成 ⇒ 收工。
#: 症状：Codex 每次内循环之后零后续请求，2026-08-27 一天五次。
#: 与 MQ-A18 同型：**同一件事只在一个端点上做对了**。
#: 形态照抄 `scripts/agent_probe_upstream.py:355-369`（真协议那份）。


def _synth_responses_tool_call_events(tool_calls: list[dict], *, index: int = 0,
                                      sink: list | None = None):
    """agent tool_calls → Responses `function_call` 事件（每个调用占一个 item）。"""
    import json as _json

    def ev(name, obj):
        return f"event: {name}\ndata: {_json.dumps(obj, ensure_ascii=False)}\n\n".encode()

    for off, tc in enumerate(tool_calls):
        fn = tc.get("function") or {}
        call_id = str(tc.get("id") or f"call_bladex_{off}")
        item_id = f"fc_bladex_{off}"
        args = str(fn.get("arguments") or "{}")
        item = {"type": "function_call", "id": item_id, "call_id": call_id,
                "name": str(fn.get("name") or ""), "arguments": args,
                "status": "completed"}
        if sink is not None:
            sink.append(item)
        idx = index + off
        yield ev("response.output_item.added", {
            "type": "response.output_item.added", "output_index": idx,
            "item": {**item, "arguments": "", "status": "in_progress"}})
        yield ev("response.function_call_arguments.delta", {
            "type": "response.function_call_arguments.delta",
            "item_id": item_id, "output_index": idx, "delta": args})
        yield ev("response.function_call_arguments.done", {
            "type": "response.function_call_arguments.done",
            "item_id": item_id, "output_index": idx, "arguments": args})
        yield ev("response.output_item.done", {
            "type": "response.output_item.done", "output_index": idx, "item": item})


def _synth_anthropic_tool_call_events(tool_calls: list[dict], *, index: int = 1,
                                      sink: list | None = None):
    """agent tool_calls → Anthropic `tool_use` content blocks。

    `index` 从 1 起（0 被合成文本块占）。参数化而非硬编码，因为这里
    **块序号确实要连续**——与文本块那处不同，见该函数注释。
    """
    import json as _json

    def ev(name, obj):
        return f"event: {name}\ndata: {_json.dumps(obj, ensure_ascii=False)}\n\n".encode()

    for off, tc in enumerate(tool_calls):
        fn = tc.get("function") or {}
        idx = index + off
        block = {"type": "tool_use", "id": str(tc.get("id") or f"toolu_bladex_{off}"),
                 "name": str(fn.get("name") or ""), "input": {}}
        if sink is not None:
            sink.append(block)
        yield ev("content_block_start", {"type": "content_block_start",
                                         "index": idx, "content_block": block})
        yield ev("content_block_delta", {
            "type": "content_block_delta", "index": idx,
            "delta": {"type": "input_json_delta",
                      "partial_json": str(fn.get("arguments") or "{}")}})
        yield ev("content_block_stop", {"type": "content_block_stop", "index": idx})


def _synth_anthropic_text_events(results: list[dict], *, raw: bool = False,
                                 index: int = 0, sink: list | None = None):
    import json as _json
    import uuid as _uuid
    text = (str(results[0].get("content", "")) if raw and results
            else "[BladeX] " + " ".join(str(r.get("content", ""))[:400] for r in results))
    mid = f"msg_{_uuid.uuid4().hex[:12]}"

    def ev(name, obj):
        return f"event: {name}\ndata: {_json.dumps(obj, ensure_ascii=False)}\n\n".encode()

    # `index` 参数在本协议**故意不用**（不是漏了）：Anthropic 官方累加器
    # `accumulate_event` 对 `content_block_start` 做的是 `content.append(...)`，
    # 按到达顺序排，不按 index 定位——播报@0 之后合成再发 0 也会正确落到第 1 块。
    # 2026-08-27 用真 SDK 累加器实测三种形态全部通过。
    # 🔴 与 Responses 相反：那边 `content_part.added` 用 `output.content[i]` 定位，
    #    编号错一位就 IndexError。**同一件事在两个协议里的判据不同，别互相套用。**
    yield ev("content_block_start", {"type": "content_block_start", "index": 0,
                                     "content_block": {"type": "text", "text": ""}})
    yield ev("content_block_delta", {"type": "content_block_delta", "index": 0,
                                     "delta": {"type": "text_delta", "text": text}})
    yield ev("content_block_stop", {"type": "content_block_stop", "index": 0})
    _ = mid


#: 合成 message item 与终止快照**必须同源**（2026-08-25 Codex live 事故：
#: 合成事件用 `output_index=99` 占位、且 `response.completed` 的 output 快照
#: 里没有这条 item ⇒ Codex 拿流内 item 与最终快照对账、对不上就断开连接
#: `client disconnected`）。用模块级容器把这一轮合成的 item 交给 rewrite_stop。
_SYNTH_ITEM_ID = "msg_bladex"


def _synth_responses_text_events(results: list[dict], *, raw: bool = False,
                                 index: int = 0, sink: list | None = None):
    import json as _json
    text = (str(results[0].get("content", "")) if raw and results
            else "[BladeX] " + " ".join(str(r.get("content", ""))[:400] for r in results))

    def ev(name, obj):
        return f"event: {name}\ndata: {_json.dumps(obj, ensure_ascii=False)}\n\n".encode()

    item = {"type": "message", "id": _SYNTH_ITEM_ID, "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}]}
    if sink is not None:
        sink.append(item)          # 交给 rewrite_stop 补进 response.completed 快照
    part_done = {"type": "output_text", "text": text, "annotations": []}
    # 🔴 `content_part.added` 不是可选装饰——它是 delta 的**前置条件**。
    #: 官方解析器 `accumulate_event`：content_part.added 做
    #: `output.content.append(part)`，output_text.delta 做
    #: `output.content[content_index]`。少了前者，delta 落在空 list 上
    #: ⇒ IndexError ⇒ 流解析崩 ⇒ 客户端断连。
    #: 2026-08-27 实测（openai SDK 真解析器喂生产字节）：缺 → IndexError，
    #: 补 → 通过。四次「内循环之后零后续请求」的根因就是这一条。
    #: 形态照抄 `scripts/agent_probe_upstream.py:383-399`（Codex 已验通过那份），
    #: 含 `status="in_progress"`——`added` 时这条 item 还没写完。
    yield ev("response.output_item.added", {"type": "response.output_item.added",
                                            "output_index": index,
                                            "item": {**item, "status": "in_progress",
                                                     "content": []}})
    yield ev("response.content_part.added", {"type": "response.content_part.added",
                                             "item_id": _SYNTH_ITEM_ID,
                                             "output_index": index, "content_index": 0,
                                             "part": {"type": "output_text", "text": "",
                                                      "annotations": []}})
    yield ev("response.output_text.delta", {"type": "response.output_text.delta",
                                            "item_id": _SYNTH_ITEM_ID,
                                            "output_index": index, "content_index": 0,
                                            "delta": text})
    yield ev("response.content_part.done", {"type": "response.content_part.done",
                                            "item_id": _SYNTH_ITEM_ID,
                                            "output_index": index, "content_index": 0,
                                            "part": part_done})
    yield ev("response.output_item.done", {"type": "response.output_item.done",
                                           "output_index": index, "item": item})


def intercept_anthropic_stream(sse_iter, **kw):
    from bladex_proxy.interception import AnthropicStreamStripper
    return _intercept_protocol_stream(
        sse_iter, stripper=AnthropicStreamStripper(),
        terminal_types=("message_delta", "message_stop"),
        rewrite_stop=_rewrite_stop_anthropic,
        synth_text_events=_synth_anthropic_text_events,
        synth_tool_call_events=_synth_anthropic_tool_call_events,
        narrate=_AnthropicNarrator, **kw)


def intercept_responses_stream(sse_iter, **kw):
    from bladex_proxy.interception import ResponsesStreamStripper
    return _intercept_protocol_stream(
        sse_iter, stripper=ResponsesStreamStripper(),
        terminal_types=("response.completed", "response.incomplete"),
        rewrite_stop=_rewrite_stop_responses,
        synth_text_events=_synth_responses_text_events,
        synth_tool_call_events=_synth_responses_tool_call_events,
        narrate=_ResponsesNarrator, **kw)


def build_agency(app_state: Any) -> AgencyRuntime:
    """lifespan 装配：吃 app.state 上已有的 index/ledger（Hub），失败不阻断启动。"""
    return AgencyRuntime(index=getattr(app_state, "index", None),
                         hub=getattr(app_state, "hub", None))
