"""`AgencyRuntime` 类本体 + 注入块常量 + `tool_context`（工具面调用上下文唯一实现点）。

09-06 F0.1 自 `agency.py` 拆出，零行为（逐字搬家）。ToolFace handlers 在 `handlers.py`
（mixin），三条流式拦截在 `streams.py`，文件型输入在 `notes.py`；门面 `bladex_proxy.agency`。
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import Any

import structlog
from bladex_core.flags import flag_enabled, flag_number
from bladex_core.ledger import (
    SECTION_TITLES,
    STATUS_CLOSED,
    Ledger,
    children_of,
    iso_ms,
    render_ledger_md,
    replay_ledger_events,
)
from bladex_core.ledger_runtime import (
    ActivationTable,
    activation_scope,
    candidate_units,
    relevant_ledgers,
    staleness_marker,
)

from bladex_proxy.agency.handlers import ToolFaceHandlersMixin
from bladex_proxy.agency.notes import (
    _last_user_text,
    load_ledger_template,
    load_system_notes,
    render_system_notes,
)
from bladex_proxy.innerloop import InnerLoopResult, run_inner_loop
from bladex_proxy.interception import (
    MODE_MIXED,
    MODE_NONE,
    MODE_PURE,
    classify_message,
    ensure_call_ids,
    strip_bladex_calls,
)
from bladex_proxy.loopledger import InnerLoopLedger
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
from bladex_proxy.surface_budget import (
    FULL_PLAN,
    LEDGER_BLOCK_TRIM_BUCKETS,
    StablePlan,
    StableSizes,
    SurfaceBudget,
    dynamic_trim_step,
    log_trim,
    plan_stable,
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
#:
#: 🔴 2026-09-07 F-B4（MQ-L49 ④）：`in the same turn` → `in ONE bladex_ledger_update
#: call (entries=[…])`。同一轮**两次**调用与**一次**调用差一个完整的模型来回——
#: 纯内循环每轮把整段上下文再发上游，46 次里 4 次撞 6 轮上限、单次 44–78 万 tok，
#: 六轮的四次全是 `bladex_ledger_update ×5-6`。F-B1 把批量能力做出来了，
#: 这一句是**告诉模型该用**的那半——有器无令与有令无器一样白搭（MQ-L7 的镜像）。
_FIRST_STEP_WITH_LEDGER = (
    "FIRST STEP, every turn: compare the user's request with the Goal above. "
    "If it is a DIFFERENT task, call bladex_ledger_switch (ledger_id='' creates a "
    "new ledger; give a short title). If it matches, proceed — and record real "
    "progress with bladex_ledger_update: add what you established to verified "
    "(with a ref) AND remove the next/open entry it resolves, in ONE "
    "bladex_ledger_update call (entries=[…]). "
    "An entry that is now in Verified does not belong in Next or Open. "
    "Put what is still unresolved in open, and the concrete steps ahead in next. "
    "Prefer switching to a listed matching ledger over creating a new one."
)

#: G16.1（MQ-A56）：**工具面**的指令 = 只留 (b) 记账半句，去掉 (a) 比对/切换半句。
#:
#: 🔴 为什么用户面不拆、只新写工具面这一条：原文的 (b) 半句是
#: "If it matches, proceed — and record real progress with…"，**单独站不住**，
#: 逐字切开做不到。于是**用户面继续用 `_FIRST_STEP_WITH_LEDGER` 一字不改**
#: ⇒ 任何行为变化都只能归因到工具面，判据不被"两边同时改了措辞"污染。
#:
#: 去掉 (a) 的依据（E 档实测 + 跨 agent 复核）：三种入站协议都把工具结果归一成
#: `role="tool"`，工具往返里 `_last_user_text` 取回的是**同一条原始用户话、逐字不变**
#: ⇒ "compare the user's request with the Goal" 的两个输入都没变，答案不可能与上轮不同；
#: 而它同时在说 "call bladex_ledger_switch"，**任务中途切账本永远是错的**。
#: 记账 0 段首 / 26 段中（CC + hermes）⇒ (b) 恰恰兑现在这一面，必须留。
#: 🔴 2026-09-09 F2 之后改文案（单变量）：**语气改回强祈使**。
#: 首版写成 "You are continuing the task above — **As you make real progress**, record it…"，
#: 去掉切换半句的同时**把催促的强度一起去掉了**，而原文开头是
#: "FIRST STEP, **every turn**" 这样一句强祈使。F2 实测记账 6 次 / 19 edits，
#: 对 E 档的 11 次 / 49 edits **降 45%**（G16.3 判据 3 红线未过）。
#: 其余读数（耗时 112 vs 121 分、到第一次大输出同为第 4 分钟、ms/chunk 11.2 vs 10.7）
#: 两档几乎一致 ⇒ 变的只有记账 ⇒ 怀疑对象锁定在这六十个字上。
#: **本次只动语气，不动内容**：切换半句仍然不给工具面。
_RECORD_ONLY_WITH_LEDGER = (
    "EVERY TURN: if you have made real progress since the last turn, record it NOW "
    "with bladex_ledger_update: add what you established to verified (with a ref) "
    "AND remove the next/open entry it resolves, in ONE bladex_ledger_update call "
    "(entries=[…]). "
    "An entry that is now in Verified does not belong in Next or Open. "
    "Put what is still unresolved in open, and the concrete steps ahead in next. "
    "You are continuing the task above — do NOT switch or create ledgers here."
)

#: 🔴 MQ-L71（2026-09-12）：`core` 那半句原文是
#: "a short title, and **optional** initial core/open/next entries"。
#: **`optional` 在 ADR-0032 与 `config/ledger-template.md` 里都找不到依据**：
#:   · ADR §4.1：`Core=长期关键事实/世界模型，pinned`（结构性必备段）；
#:   · 模板（权威、用户可编辑）：*"When the ledger is created this holds
#:     **what the USER already told you**"*；
#:   · ADR §4.3 的"可带 core/open/next"是**区别于 Goal**（Goal 由模型提炼、仅用户可改），
#:     括号里写的是"**一次调用建满**"。
#: ⇒ 是实现侧自己把规格放宽了，而且**没说 Core 建本时该放什么**，只让模型"看下面的模板"。
#: 本次按模板语义改写：点名 core 的建本时内容 = 用户已经说过的硬约束。
#: 🔴 **本条只做规格符合性，不声称带来记账改善**：同批实测（MQ-L72）同 agent 同模型
#: 同指令文本三跑 `bladex_ledger_update` 调用 **44 / 15 / 4** —— 任何单跑的
#: "改完变多了"都落在噪声里。行为侧要验必须每臂 n≥3 比率。
#: 🔴 **同批的 MQ-L72（空段带回模板说明）已撤回、只留登记**：2026-09-12 的 handoff
#: 实测 B 从一本 **零 Core** 的账本接手，四项 N4 判据与有 4 条 Core 的 #1 **全部持平**
#: ⇒ 危害未证，而 ADR-0029 的规矩是**加注入者举证**。详见 MQ-L72「撤回」节。
_FIRST_STEP_NO_LEDGER = (
    "No active task ledger for this session. FIRST STEP: if one of the ledgers "
    "listed below already covers this request, call bladex_ledger_switch with its "
    "id to continue it. Otherwise call bladex_ledger_switch with ledger_id='', a "
    "short title, and initial core entries — the hard constraints and taboos the "
    "user has ALREADY stated (plus open/next if you have them), following the "
    "template below. The Goal is set from the user's own words — you do not write it. "
    "Prefer switching to a listed matching ledger over creating a new one."
)

#: MQ-L38 相关性段标题（2026-09-04）。与 "Other recent ledgers" 并列：那段答"最近在干什么"，
#: 这段答"以前干过这件事没有"。无命中整段不出现（块逐字等于修前）。
_MATCHING_LEDGERS_HEADER = (
    "Ledgers that look like the SAME task as the current request (read it with "
    "bladex_ledger_read, then bladex_ledger_switch to it - do NOT create a new ledger "
    "if one matches):"
)

#: 批 Y · Y1d（2026-09-28，拍板 (30)，MQ-L93）：项目键落空 ∧ 同 agent Global 有绑定 ⇒
#: 那本**置顶进 recency 段**并带这个标注。它取代的是 08-29 的惰性作用域迁移
#: （`_maybe_migrate_binding`：把 Global 绑定静默搬到项目键）——T1 §7a 实测全库 4 次迁移
#: 3 次跨项目（BladeX 账本进 g3 bench / teamai-cli 分析本进 vllm / bug-fix 本进 bladex-r4），
#: 立论「Global 绑定本就是项目无关混用」不成立，且它替模型改写归属而无举证（刚性原则 15）。
#: 08-29 立规则的两个理由**不许带回来**：① 模型要看得见旧本、一次 switch 接上（置顶保证
#: 它不会被 recency top-5 挤出去——live 病例里 Global 旧本常排第 6 以后）；② 不许诱导重复建本
#: （`_FIRST_STEP_NO_LEDGER` 先指候选、再说新建，措辞不改）。切不切归模型（三红线）。
_GLOBAL_BOUND_SUFFIX = " (your active ledger outside any project - switch to it if this is the same task)"

#: 批 HO（MQ-L99，拍板 (43)）：相关性段里**已关**那一行的行尾标注——只陈述事实（已关、谁、何时、
#: 切过去会重开），**不劝切**：切不切归模型（三红线）。没有已关命中 ⇒ 不出现，块逐字等于修前。
#: 它取代了 E4（MQ-L98）的「Ledgers you closed in this session recently」小节：那一节只对关它的
#: 那个会话列出（`closed_by_session == session_id`），交接换会话、换 agent 必然不满足；
#: L98 原病例（09-28 任务 2 返工原话）在相关性段上按内容命中原本（0.48 / shared 17）⇒
#: 并成一条规则、删掉同会话分支（`report-batch-HO-20261015.md` §2）。
def _closed_match_suffix(led: Any) -> str:
    who = f" by {led.closed_by_agent}" if getattr(led, "closed_by_agent", "") else ""
    at = (led.closed_at or "")
    when = f" {at[:10]} {at[11:16]} UTC" if len(at) >= 16 else ""
    return f" [closed{who}{when}; switching to it reopens it]"


#: 账本块的构成桶（F-A1 / MQ-A35）。前六个来自渲染好的账本 MD（`header` = 标题 +
#: 元数据行 + 块标记），后六个是 BladeX 在 MD 之外追加的段。
#:
#: 🔴 **闭集只是默认值不是全集**：`Ledger.section_order` 可增段（0.2.0 的 `lessons`），
#: 新段自动按 key 得到自己的桶（`_split_ledger_md` 的 `titles.get(t) or t.lower()`）——
#: 故消费方按 key 读、不按位置读（feedback_closed_set_from_definition）。
LEDGER_BLOCK_BUCKETS: tuple[str, ...] = (
    "header", "goal", "core", "verified", "open", "next",
    "stale", "instruction", "children", "candidates", "recent", "template",
)


@dataclass(slots=True)
class LedgerBlockBreakdown:
    """一次账本块注入的段级构成（**仪器**，不改行为）。

    分母纪律（feedback_instrument_reference_frame）：`chars` 的**分母是这一块本身**
    ——`sum(chars.values()) == len(block_content)` 逐字成立（`_build_ledger_block`
    按行累加，无 md 侧后验切分的漂移）。跨轮占比要另找分母（`bladex_added_chars`
    或整段 prompt），本对象不替消费方选。

    `ledger_id=""` ∧ `rev=-1` = 这一轮没有激活账本（`template` 桶非零）。
    """

    ledger_id: str = ""
    rev: int = -1
    chars: dict[str, int] = field(default_factory=lambda: dict.fromkeys(LEDGER_BLOCK_BUCKETS, 0))
    #: 🔴 MQ-A43（09-07 首验修）：Next+Open 的词面单元集，**在这里算**。
    #:
    #: F-A2 首版在 `_enqueue_turn` 里从进程内池现算——而那已经是**这一轮结束后**的池。
    #: 模型若在本轮 `bladex_ledger_update` 写了 Next，尺子就拿"它刚写进去的东西"去和
    #: "它写这些东西时的参数"求交，U ⊆ V 恒成立（live 实录：`hits=20 next_units=20`
    #: `rev=12 rev_now=13`，100% 是自指不是读数）。
    #: 收在 breakdown 上 = 与段级尺同一个生产点，拿到的天然是**模型看到的那一版**，
    #: 不必按 (id, rev) 回 Hub 重放。**不进 `LedgerAnchor`**：它是集合、只喂日志，
    #: 入 schema 就要承诺重建语义，而这把尺子的口径还在改。
    pending_units: frozenset[str] = frozenset()
    #: 🔴 MQ-A47（09-07 首次取基线时发现）：`pending_units` 的**稀有子集**。
    #:
    #: 全天 124 条可用读数里 **63 条（51%）命中的是 `documents,hermes`** —— Next/Open 写了
    #: 文件路径、首动作又在同一目录下操作文件，两个通用 token 必撞，**与"读没读账本"无关**。
    #: `hits>0 = 94%` 因此是被底噪撑起来的假读数。
    #:
    #: 判据复用 `relevant_ledgers` 已经为**同一个问题**写过的那条（MQ-L38「稀有锚」）：
    #: df ≤ max(2, N//10)。一个机制不是两个——那边是"这句话像不像同一件事"，
    #: 这边是"这个动作有没有引用待办"，共享的正是"通用词零信息"这个前提。
    #:
    #: 🔴 MQ-A51（09-08 复核实测）：**光靠 df 不成立**。31 份账本里
    #: `df(documents)=1`、`df(hermes)=3`，阈值≈3 ⇒ 底噪按 df 就是"稀有的"，
    #: 一条没剔掉；被剔掉的反而是「报告/更新/结果」这类真通用词。
    #: 前提"路径 token 是通用词"只在池内多本谈**不同**项目时成立，本池主题集中。
    #: ⇒ 现在是**两条判据并联**：df 稀有 ∧ 非路径类
    #: （`ledger_runtime.path_like_units`，按类别而不是按词频剔）。
    pending_rare: frozenset[str] = frozenset()

    def total(self) -> int:
        return sum(self.chars.values())


def _split_ledger_md(md: str, section_order: list[str]) -> list[tuple[str, str]]:
    """渲染好的账本 MD → `[(段键, 该段全文)]`。

    🔴 无损（原则 13）：`"\\n".join(chunk for _, chunk in out) == md` 逐字成立——
    切分只在 `## ` 行的边界上分组，不丢也不改一个字符。第一段（标题 + `- ledger:`
    等元数据行）归 `header`；未登记的段按 key 小写自成一桶而不是并进 header
    （并进去 = 新增段静默不可见，正是"闭集从定义读"要防的形态）。
    """
    titles = {SECTION_TITLES.get(k, k.capitalize()): k for k in section_order}
    groups: list[tuple[str, list[str]]] = [("header", [])]
    for line in md.split("\n"):
        if line.startswith("## "):
            t = line[3:].strip()
            groups.append((titles.get(t) or t.lower(), []))
        groups[-1][1].append(line)
    return [(k, "\n".join(ls)) for k, ls in groups]


def _pending_text(led: Any) -> str:
    """一本账本的 Next+Open 正文（`_build_ledger_block` 与 df 语料**同一个口径**，
    抄第二遍就会分叉）。"""
    return " ".join(e.text for sec in ("next", "open") for e in led.entries(sec))


def _pending_corpus(pool: dict[str, Ledger]) -> list[str]:
    """df 语料 = 池内**Next/Open 非空**的那些本的待办正文。

    🔴 MQ-A51（09-08）：分母必须是被测对象。小池保护原先判 `len(pool) < 2`，
    而 df 是按"待办正文"算的——31 份账本里空 Next/Open 的那些**一个词都不贡献**，
    却照样把 `len(pool)` 撑到过关（同批第六次"分母不是被测对象"，
    feedback_instrument_reference_frame）。
    """
    out: list[str] = []
    for led in pool.values():
        try:
            text = _pending_text(led)
        except AttributeError:
            continue          # 不是账本对象（测试替身 / 脏池）——不贡献 df，也不炸
        if text.strip():
            out.append(text)
    return out


def rare_pending_units(pool: dict[str, Ledger], units: frozenset[str], *,
                       pending_text: str) -> frozenset[str]:
    """`units` 里**够格当锚**的那些：池内稀有 **且** 不是路径带进来的（MQ-A47/A51）。

    两条判据**并联，都过才算 rare**——它们剔的是两类不同的底噪：

    - **df 稀有**（A47）：`df ≤ max(2, N//10)`，逐字取自
      `ledger_runtime.relevant_ledgers` 的 `rare_df`（同一个"通用词零信息"的前提，
      不另立一个数）。剔的是「报告 / 更新 / 结果」这类池内人人都有的词。
    - **非路径类**（A51）：`path_like_units(pending_text)`。剔的是 `documents` /
      `hermes` 这类**寻址片段**——它们 df 低（本池主题集中，路径只出现在少数几本
      里），df 那一条永远抓不到它们，这正是 A47 判据落空的原因。

    `pending_text` = 这本账本的 Next+Open 原文，**必传**：路径类判据要看原文
    （单元集已经把 `~/Documents/x.md` 打散成 token，从 token 反推不出它是路径）。
    做成必传关键字而不是可选——漏传就静默退回"只有 df"，那是把缺陷藏起来（原则 12）。

    df 语料里 Next/Open 非空的本 < 2 ⇒ df 分辨不出稀有，**只做路径类剔除后原样返回**
    （不猜；宁可留底噪也不假装筛过）。
    """
    if not units:
        return units
    from bladex_core.ledger_runtime import path_like_units  # noqa: PLC0415
    kept = units - path_like_units(pending_text)
    corpus = _pending_corpus(pool)
    if len(corpus) < 2:
        return kept
    df: dict[str, int] = {}
    for text in corpus:
        for u in candidate_units(text):
            df[u] = df.get(u, 0) + 1
    rare_df = max(2, len(corpus) // 10)
    return frozenset(u for u in kept if df.get(u, 0) <= rare_df)


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
    # MQ-L54：`rounds=6` 有两种成因（六件不同的事 / 同一条错撞六次），此前在日志里
    # 长得一模一样。`errors` 是总量、`fail_streak` 是"连着撞同一个"的最长次数——
    # 两轴独立：错三次但各不相同（正常自愈的形状）与连撞三次是两件事。
    # 三条日志（`agency_stream_loop_done` / `agency_protocol_inner_loop_done` /
    # `agency_inner_loop_done`）都 `**_loop_cost(result)`，加在这里三处同时带。
    return {"tool_ms": round(tool_ms), "llm_ms": round(llm_ms),
            "tokens": tokens, "tools": ",".join(tools[:8]),
            "errors": sum(int(getattr(r, "errors", 0) or 0) for r in rounds),
            "fail_streak": int(getattr(result, "fail_streak", 0) or 0)}


def tool_context(*, session_id: str, agent_id: str, project_id: str,
                 upstream_messages: list[dict], turn_index: int = 0) -> dict:
    """工具面调用上下文——**唯一实现点**（三条拦截路径共用：非流式 / chat 流式 / 协议流式）。

    🔴 MQ-L40（2026-09-04）：此前三处各写一份字面量，两条流式路径**漏传 `turn_index`**
    ⇒ 当时的切换防抖（批 Y1c 已删，MQ-L91）差值恒 0 ⇒ 一个 scope 上只有进程内第一次
    切换能落定——09-03 泰山重复建本的第二个成因。`turn_index` 缺省时按 identity 同一公式
    从 `upstream_messages` 派生，不依赖调用方记得传——公式 = `ledger_runtime.user_turn_index`
    （轮次数；修前 `len(messages)//2` 是请求数，MQ-L48）。Y1c 起它只进 `LEDGER_SWITCH`
    载荷与 `agency_switch_evidence` 观测，不参与任何判定。

    🔴 `user_query` 是**剥过信封**的本轮用户文本，全部"按用户原话"的判定只许读这一格：
    Goal 改动的引文锚、`_user_named_ledger`、切换举证 `verify_switch_evidence`（Y1c 判据 5）。
    """
    from bladex_core.ledger_runtime import user_turn_index  # noqa: PLC0415
    ti = int(turn_index) if turn_index else user_turn_index(upstream_messages)
    return {"session_id": session_id, "agent_id": agent_id,
            "turn_index": ti, "project_id": project_id,
            "user_query": _last_user_text(upstream_messages, for_goal=True)}


@dataclass
class _SessionStable:
    """一个会话的稳定层定格记录（批 Z · Z2.4 / MQ-A58 + Z2.3 稳定层计划）。

    `plan` 首轮按窗口算一次后冻结；`anchor` = 定格时 about 块**前一条**消息的 (role, 内容哈希)，
    之后各轮按它复位（agent 新追加的 system 一律排在其后）；`about_hash` 用来判内容有没有
    合法地变（账本面翻转 / TOOLS.md 升级），变了打一行日志并重新定格，不算漂移。
    """

    plan: StablePlan
    anchor: tuple[str, str] | None = None
    anchored: bool = False
    about_hash: str = ""
    ledger_sections: bool | None = None
    notes_hash: str = ""


#: 每进程记住多少个会话的定格（LRU）。超出的老会话再来 = 重新定格一次（只多冷启一次 cache）。
_SESSION_STABLE_MAX = 4096


def _msg_fingerprint(m: dict) -> tuple[str, str]:
    """消息的 (role, 内容哈希)——about 块的内容锚点。"""
    c = m.get("content")
    try:
        raw = c if isinstance(c, str) else json.dumps(c, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        raw = str(c)
    return (str(m.get("role", "")), hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16])


def _legacy_about_pos(messages: list[dict]) -> int:
    """定格前的落点规则：最后一条 **agent 自己的** system 之后（跳过 BladeX 注入块）。"""
    pos = 0
    for i, m in enumerate(messages):
        if m.get("role") != "system":
            continue
        c = m.get("content")
        if isinstance(c, str) and any(mk in c for mk in INJECTION_MARKERS):
            continue          # BladeX 自己的注入块，不是 agent 的 system
        pos = i + 1
    return pos

def _anchored_about_pos(messages: list[dict], anchor: tuple[str, str] | None) -> int | None:
    """按内容锚点复位：锚点消息之后。锚点找不到 ⇒ None（调用方回落 + 记日志）。

    锚点为 None（定格时数组头部没有 agent system）⇒ 0，除非现在头部出现了 agent 的
    system（不抢开场，刚性原则 10）⇒ None 让调用方重新定格。
    """
    if anchor is None:
        if messages and messages[0].get("role") == "system":
            c = messages[0].get("content")
            if not (isinstance(c, str) and any(mk in c for mk in INJECTION_MARKERS)):
                return None
        return 0
    for i, m in enumerate(messages):
        if _msg_fingerprint(m) == anchor:
            return i + 1
    return None


class AgencyRuntime(ToolFaceHandlersMixin):
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
        # （V-F3 的 AGENTS.md 名册段已于批 F4 退出稳定层——拍板 (16)，ADR-0034 §3.1：
        #  名册没有注入以外的消费方，行为画像改由模型经 `bladex_flash_read` 按需取；
        #  该工具 10-17 F5 判决退出注册，画像只留 dashboard / `/admin/behavior`。）
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
        #: F-A1 / MQ-A35：最近一次账本块注入的段级构成，供 `_apply_agency_surfaces`
        #: 组 `Turn.ledger_anchor`。`None` = 这一轮没注块（aux / 被 gate / 模块关）。
        #: 与 `last_ledger_face` 同款接线理由：`_enqueue_turn` 有 12 个调用点。
        self.last_ledger_breakdown: LedgerBlockBreakdown | None = None
        #: F-A1：最近一次自我介绍块的字符数（批 F4 起 = 整块正文长度；此前还要
        #: 扣掉名册那一段 `last_roster_chars`，该读数随名册退役）。每请求恰好覆写一次。
        self.last_about_chars: int = 0
        #: 批 Z · Z2.3 / Z2.4：session_id → 稳定层定格记录（LRU，`_SESSION_STABLE_MAX`）。
        self._session_stable: OrderedDict[str, _SessionStable] = OrderedDict()
        #: 最近一次 `stable_plan` 的结果（每请求覆写；`_apply_agency_surfaces` 同步链内读）。
        self.last_stable_plan: StablePlan = FULL_PLAN
        self.activation = ActivationTable()
        self.pool: dict[str, Ledger] = {}
        self._last_candidates: list = []   # MQ-L38：本轮相关性候选（建本告警用）
        #: MQ-L69：上面那份候选**这一轮有没有真渲染给模型**。
        #: `_last_candidates` 跨轮残留，单看它分不出"看了仍建"与"没给看"。
        self._candidates_shown: bool = False
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
        #: (激活作用域, session_id) → ((user_turn_index, 消息条数), 上次判定的 user 面?)
        #: G16.0/G16.2（MQ-A56）。🔴 与上一行**故意分开**：那个键是 `ledger_id`，
        #: 而注入门控在**无账本时也要判**。带上消息条数是为了把**重试**与**工具往返**
        #: 分开（前者形状完全相同 ⇒ 沿用上次判定，后者 uti 不变但变长 ⇒ 工具面）。
        #: 重启归零 ⇒ 重启后第一个请求判 `face=user`，只会多注一次，不会漏注。
        self._last_seen_face_turn: dict[tuple[str, str], tuple[tuple[int, int], bool]] = {}
        self.recent = RecentRequests()
        self.toolface = ToolFace()
        self.toolface.register("bladex_memory_search", self._h_memory_search)
        self.toolface.register("bladex_ledger_read", self._h_ledger_read)
        self.toolface.register("bladex_ledger_update", self._h_ledger_update)
        self.toolface.register("bladex_ledger_switch", self._h_ledger_switch)
        self.toolface.register("bladex_ledger_close", self._h_ledger_close)
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
                from bladex_core.ledger import binding_sessions
                self.pool, bindings = replay_ledger_events(events)
                # Y1c：不再恢复轮位（原 `switch_turns=` 只喂切换防抖基点，防抖已删）。
                self.activation.restore(bindings, sessions=binding_sessions(events))
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
                      stable_plan: StablePlan | None = None,
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
        # 🔴 MQ-L70（2026-09-12）：模块门 `ledger` 必须同时挡**工具面**。此前只有
        # `insert_ledger_block` 读 `self.ledger_on`，这里没并 ⇒ `BLADEX_MODULE_LEDGER=0`
        # 只挡正文块、账本三工具照注（live：`-105234.log` module_ledger=False 却
        # ledger_face=True 35/37 轮）。后果不是"多给了工具"这么简单：那一窗被当成
        # "账本关"对照臂，读出来的全是假读数。关账本必须是**一个**开关就关干净。
        # 批 Z · Z2.3：稳定层计划（会话首轮按窗口冻结）压在两族之上——被裁的族整会话不注。
        _plan = stable_plan or FULL_PLAN
        _ledger_ok = (self.ledger_on
                      and (ledger_tools_allowed(tier, self._ledger_tiers) or _owns)
                      and not ledgerless
                      and _plan.keep_ledger_family)
        new_tools, injected = inject_tools(tools, agent_id=agent_id,
                                           auxiliary=auxiliary, model_tier=tier,
                                           ledger_allowed=_ledger_ok, fmt=fmt,
                                           memory_allowed=_plan.keep_memory_family)
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
                    # Z2.3：这一会话的稳定层被预算裁掉了哪些（空 = 没裁 / 没配窗口）
                    stable_trimmed=list(_plan.trimmed),
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

    # ── 批 Z · Z2.3 / Z2.4：稳定层预算计划 + 会话定格 ──

    def stable_sizes(self, fmt: str = "openai") -> StableSizes:
        """稳定层各块字符数：两族工具 schema + 自我介绍（含/不含账本段，含标记行与换行）。"""
        from bladex_proxy.toolface import FAMILY_LEDGER, FAMILY_MEMORY, family_schema_chars
        fam = family_schema_chars(fmt)

        def _about(ledger: bool) -> int:
            body = render_system_notes(self.about, ledger=ledger) if self.about else ""
            return len(f"{ABOUT_BLOCK_OPEN}\n{body}\n{ABOUT_BLOCK_CLOSE}") if body else 0

        return StableSizes(memory_tools=fam.get(FAMILY_MEMORY, 0),
                           ledger_tools=fam.get(FAMILY_LEDGER, 0),
                           about_full=_about(True), about_no_ledger=_about(False))

    def stable_plan(self, session_id: str, budget: SurfaceBudget | None,
                    fmt: str = "openai") -> StablePlan:
        """会话的稳定层计划：首轮按预算算一次并冻结，之后同会话直接返回冻结值。

        没有 session_id（拿不到会话标识）⇒ 每轮按当轮预算算、不冻结——这时定格本就无从谈起。
        没配窗口（budget=None）⇒ `FULL_PLAN`，同样冻结（会话中途换到配了窗口的模型也不再裁稳定层，
        与「稳定层只在首轮决定一次」同一条纪律）。
        """
        rec = self._session_stable.get(session_id) if session_id else None
        if rec is not None:
            self._session_stable.move_to_end(session_id)
            self.last_stable_plan = rec.plan
            return rec.plan
        sizes = self.stable_sizes(fmt)
        plan = plan_stable(budget, sizes)
        if plan.trimmed:
            estimate = sizes.memory_tools + sizes.ledger_tools + sizes.about_full
            per_step = {"ledger_family": sizes.ledger_tools + (sizes.about_full - sizes.about_no_ledger),
                        "about": sizes.about_no_ledger, "memory_family": sizes.memory_tools}
            for step in plan.trimmed:
                log_trim(step, chars=per_step.get(step, 0), budget=plan.budget_chars,
                         estimate=estimate, session=session_id, layer="stable")
        if session_id:
            self._session_stable[session_id] = _SessionStable(plan=plan)
            while len(self._session_stable) > _SESSION_STABLE_MAX:
                self._session_stable.popitem(last=False)
        self.last_stable_plan = plan
        return plan

    def about_block_chars(self, *, ledger_face: bool | None, stable_plan: StablePlan | None,
                          toolface_injected: bool) -> int:
        """这一轮若注自我介绍会占多少字符（Z2.3 算动态层剩余预算用；不注 ⇒ 0）。"""
        _plan = stable_plan or FULL_PLAN
        if not self.about or not toolface_injected or not _plan.keep_about:
            return 0
        _ledger = (self.ledger_on if ledger_face is None else bool(ledger_face)) and _plan.keep_ledger_family
        body = render_system_notes(self.about, ledger=_ledger)
        return len(f"{ABOUT_BLOCK_OPEN}\n{body}\n{ABOUT_BLOCK_CLOSE}") if body else 0

    # ── server 接线面 ②a 自我介绍（稳定层，V-P6 / ADR-0032 §3.1）──

    def insert_system_notes(self, messages: list[dict], *,
                            toolface_injected: bool,
                            ledger_face: bool | None = None,
                            session_id: str = "",
                            stable_plan: StablePlan | None = None) -> list[dict]:
        """把 BladeX 自我介绍插进**前缀**（稳定层）。返回新数组，不改原数组。

        ## 首轮定格（批 Z · Z2.4，MQ-A58，2026-09-24）

        ADR-0032 §3.1 写"稳定层会话首轮定格、逐字节稳定"，此前实现却每轮按「最后一条 agent
        `system` 之后」重新定位，而 Claude Code 会往数组**末尾**持续追加 `role=system` 的
        system-reminder ⇒ 落点在一个会话里漂 15 次（Z2.0 读数：67 次注入变了 32 次）。
        现在：`session_id` 非空时，首轮按老规则算定位置后记下**内容锚点**（块前一条消息的
        (role, 哈希)），之后各轮按锚点复位——agent 新追加的 system 一律排在它之后；锚点找不到
        （agent 压缩了历史）⇒ 回落老规则重新定格 + `agency_about_anchor_lost`。选这一方案而不是
        "注进 system 前缀本体（数组最前）"：后者与刚性原则 10 冲突（agent 的 system prompt
        仍是第一位的，我们不抢开场），且改变了 hermes / codex 这些本就不漂的 agent 的形态。
        内容**合法地**还会变的两种情形——账本面翻转、proxy 升级改了 TOOLS.md——各打一行
        `agency_about_refrozen reason=…` 并重新定格，不算漂移。修它的理由是实现要做到文档
        说的那件事（刚性原则 12），**不是性能**（09-09 读数已证伪"漂移让会话变慢"）。

        `stable_plan`（Z2.3）：首轮按窗口冻结的稳定层计划——`keep_about=False` ⇒ 整会话不注；
        `keep_ledger_family=False` ⇒ 账本段随账本工具一起剥掉（与 `augment_tools` 同一份计划）。

        ## 按工具族装卸（MQ-L77，2026-09-18）

        `ledger_face`：这一轮**账本族工具**注没注（`augment_tools` 的 `_ledger_ok`，
        由调用方从 `last_ledger_face` 同步传入）。`False` ⇒ 正文里所有
        `<!-- bladex:ledger -->` 段整段剥掉——模块关 / 档位不在集合 / 零工具轮时
        说明书不能教模型去调一个工具列表里没有的 `bladex_ledger_switch`（live n4：
        codex 第 2 轮照着说明书发了，`agency_protocol_mixed_stripped removed=1`，
        白费一轮）。`None` 沿用 `self.ledger_on`（模块门）——只给旧调用方兜底。
        稳定层因此在「账本面翻转」时变一次；同一会话内档位与绑定不变，前缀不动。

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
        # F-A1：与 `last_ledger_breakdown` 同款——先清空再按路径填，防上一轮读数串台。
        self.last_about_chars = 0
        if not self.about or not toolface_injected:
            return messages
        _plan = stable_plan or FULL_PLAN
        if not _plan.keep_about:
            return messages       # Z2.3：首轮按预算裁掉，整会话不注（定格那一轮已打 agency_surface_trimmed）
        for m in messages:
            c = m.get("content")
            if isinstance(c, str) and ABOUT_BLOCK_OPEN in c:
                logger.debug("agency_about_already_present")
                return messages
        _ledger = ((self.ledger_on if ledger_face is None else bool(ledger_face))
                   and _plan.keep_ledger_family)
        body = render_system_notes(self.about, ledger=_ledger)
        if not body:
            return messages
        block = {"role": "system",
                 "content": f"{ABOUT_BLOCK_OPEN}\n{body}\n{ABOUT_BLOCK_CLOSE}"}
        about_hash = hashlib.sha256(block["content"].encode("utf-8")).hexdigest()[:16]
        notes_hash = hashlib.sha256(self.about.encode("utf-8")).hexdigest()[:16]
        rec = self._session_stable.get(session_id) if session_id else None
        freeze = "none"
        if rec is None or not rec.anchored:
            pos = _legacy_about_pos(messages)
            if session_id:
                if rec is None:
                    rec = _SessionStable(plan=_plan)
                    self._session_stable[session_id] = rec
                    while len(self._session_stable) > _SESSION_STABLE_MAX:
                        self._session_stable.popitem(last=False)
                rec.anchored = True
                rec.anchor = _msg_fingerprint(messages[pos - 1]) if pos > 0 else None
                rec.about_hash, rec.ledger_sections, rec.notes_hash = about_hash, _ledger, notes_hash
                freeze = "frozen"
        else:
            self._session_stable.move_to_end(session_id)
            _anchored = _anchored_about_pos(messages, rec.anchor)
            if _anchored is None:
                pos = _legacy_about_pos(messages)
                rec.anchor = _msg_fingerprint(messages[pos - 1]) if pos > 0 else None
                logger.info("agency_about_anchor_lost", session=session_id[:24], pos=pos,
                            total=len(messages) + 1)
                freeze = "anchor_lost"
            else:
                pos = _anchored
                freeze = "anchored"
            if about_hash != rec.about_hash:
                why = ("ledger_face" if rec.ledger_sections != _ledger
                       else "notes_changed" if rec.notes_hash != notes_hash else "unknown")
                logger.info("agency_about_refrozen", session=session_id[:24], reason=why,
                            old_hash=rec.about_hash, new_hash=about_hash,
                            ledger_sections=_ledger)
                rec.about_hash, rec.ledger_sections, rec.notes_hash = about_hash, _ledger, notes_hash
        out = [*messages[:pos], block, *messages[pos:]]
        self.last_about_chars = len(block["content"])
        # 🔴 `roster_chars` 字段自批 F4（2026-09-23）起**退役**：名册退出稳定层，
        # 此前 `chars` = about + roster（live ≈960），之后 `chars` 只剩 about。
        # 跨 F4 比较 `chars` 时这一跳是名册，不是 about 正文变了。
        logger.info("agency_about_injected", pos=pos, total=len(out),
                    chars=len(block["content"]),
                    # MQ-CA13 ①（批 R · R1）：内容哈希。`about_same` 此前拿 (chars, roster_chars)
                    # 代理"逐字相同"，字数相同不等于内容相同——这一列让它成为真证据。
                    about_hash=about_hash,
                    ledger_sections=_ledger,
                    # Z2.4：这一轮落点怎么来的——frozen（首轮定格）/ anchored（按锚点复位）/
                    # anchor_lost（锚点丢了，重新定格）/ none（没有 session_id，不定格）
                    freeze=freeze)
        return out

    # ── server 接线面 ②b 账本块注入（动态层）──
    def scope_of(self, agent_id: str, project_id: str = "") -> str:
        """本请求的激活作用域键。见 `ledger_runtime.activation_scope` 的立论。"""
        return activation_scope(agent_id, project_id)

    def ledger_injection_message(self, agent_id: str, *, project_id: str = "",
                                 with_instruction: bool = True,
                                 user_text: str = "",
                                 face: str = "user") -> dict | None:
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

        🔴 F-A1（2026-09-07）：本函数保留为**只返消息**的旧名——既有调用点与测试
        逐字不变。段级构成尺在 `_build_ledger_block`，本函数是它的一层壳。
        """
        msg, _ = self._build_ledger_block(
            agent_id, project_id=project_id,
            with_instruction=with_instruction, user_text=user_text, face=face)
        return msg

    def _build_ledger_block(self, agent_id: str, *, project_id: str = "",
                            with_instruction: bool = True,
                            user_text: str = "",
                            face: str = "user",
                            omit_buckets: tuple[str, ...] = (),
                            quiet: bool = False,
                            ) -> tuple[dict | None, LedgerBlockBreakdown]:
        """账本块 + 段级构成（F-A1 / MQ-A35）。正文逐字等同修前——只多了记账。

        为什么把尺子放在**生产点**而不是事后切分注入好的字符串：块里有六段是
        BladeX 自己拼的（stale / instruction / children / candidates / recent /
        template），它们在 MD 里没有 `## ` 标题，事后切分只能靠猜前缀——那正是
        "仪器骗人：参照系脱钩"的第一类。这里每追加一行就记一次，
        `sum(chars.values()) == len(content)` 是构造出来的恒等式，不是巧合。
        """
        bd = LedgerBlockBreakdown()
        # G16.2（MQ-A56）：开关关掉 ⇒ `_user_face` 恒真 ⇒ **逐字回到 09-09 之前**。
        # 默认值真身在 `bladex_core.flags`，与 `.env.example` 由测试对账（刚性原则 12）。
        _user_face = (face == "user") or not flag_enabled("BLADEX_LEDGER_FACE_SPLIT")
        # 🔴 MQ-L69（2026-09-11）：**每次建块都先归假**，放在两处 early return 之前。
        # `_last_candidates` 只在下面 `with_instruction and _user_face` 那一支赋值、
        # **从不在轮间重置** ⇒ 工具面轮沿用上一个用户面轮的候选。于是
        # `ledger_created_despite_match` 有二义：「模型看着候选仍新建」（该告警）
        # 与「这一轮压根没给它看候选」（不该算在模型头上）**日志上长得一模一样**。
        # 本标志把两者分开；它不改变任何注入正文，只是让告警可以被复算。
        self._candidates_shown = False
        if not (self.ledger_on and self.toolface_on):
            return None, bd
        led = self._active_ledger(self.scope_of(agent_id, project_id))
        lines: list[str] = []

        def _put(bucket: str, *new: str) -> None:
            """追加若干行并计入桶。每行按 `len+1`（行 + `"\\n"` 分隔符）计，
            末尾统一从 `header` 扣回 1 —— 于是桶和逐字等于 `"\\n".join(lines)` 的长度。"""
            lines.extend(new)
            bd.chars[bucket] = bd.chars.get(bucket, 0) + sum(len(s) + 1 for s in new)

        _put("header", LEDGER_BLOCK_OPEN)
        if led is None:
            if not with_instruction:
                # 无账本 + 无工具 = **无器则无令**，整块不注（C 方案要点）。
                return None, LedgerBlockBreakdown()
            # 🔴 无账本分支**也要给列表**（V-L6e 复核发现）：不给列表却要求"没有就建"，
            # 模型即使想并入既有账本也无从判断——这正是 CC 那次同一任务开两本的一半成因。
            _put("instruction", _FIRST_STEP_NO_LEDGER, "")
            _put("template", self.template.strip())
        else:
            bd.ledger_id = led.ledger_id
            bd.rev = led.rev
            # MQ-A43：Next+Open 的单元集取**此刻**这一版（= 模型即将看到的那一版）。
            # 与候选段的 `candidate_units(user_text)` 同一把分词、同一张停表，
            # 两个读数因此可以并排看。
            _pending = _pending_text(led)
            bd.pending_units = frozenset(candidate_units(_pending))
            # MQ-A47 + A51：稀有子集 = df 稀有 ∧ 非路径类
            # （`documents`/`hermes` 靠 df 剔不掉——本池里它们 df 反而低）。
            bd.pending_rare = rare_pending_units(self.pool, bd.pending_units,
                                                 pending_text=_pending)
            note = self._coauthor_note(led, agent_id)
            header = ("Active task ledger (your working state):"
                      if not note else
                      "Active task ledger (your working state; " + note + "):")
            _put("header", header, "")
            for key, chunk in _split_ledger_md(render_ledger_md(led).strip(),
                                               led.section_order):
                _put(key, chunk)
            # V-L4 陈旧兜底：只陈述"这本账本上过了 N 轮没更新"这个事实。
            # 🔴 **不受 `with_instruction` 门控的反面**——它带一句"用
            # bladex_ledger_update 记下来"，没有工具面时就是有令无器（C 方案的
            # 教训，MQ-L7）。故与指令同门控。
            if with_instruction:
                # 🔴 陈旧标记归 (b)「你该记账了」⇒ **两个面都给**（G16.1）。
                mark = staleness_marker(
                    turns_since_update=self._turns_since_update[led.ledger_id])
                if mark:
                    _put("stale", mark)
                    if not quiet:
                        logger.info("agency_ledger_stale_marked",
                                    ledger=led.ledger_id,
                                    turns=self._turns_since_update[led.ledger_id])
                # G16.2（MQ-A56）：用户面 = 原文一字不改；工具面 = 只留记账半句。
                _put("instruction", "",
                     _FIRST_STEP_WITH_LEDGER if _user_face else _RECORD_ONLY_WITH_LEDGER)
        # V-L6e：父子链可见（Jason 拍板"子任务可以建账本，但要与父账本信息同步"）。
        # 子侧的 parent 行由 render_ledger_md 渲染；这里补**父侧看得见子**——
        # 派生而非双写（单一方向存储，`children_of`）。子列表是**只读信息**
        # （父任务的进展全貌），故不受 with_instruction 门控。
        if led is not None:
            kids = children_of(self.pool, led.ledger_id)
            if kids:
                _put("children", "", "Sub-ledgers spawned from this task "
                                     "(switch by id to work in one):")
                _put("children",
                     *[f"- {k.ledger_id}: {k.title or k.ledger_id} [{k.status}]"
                       for k in kids[:5]])
        # LEDGERS top-5 标题行（拍板 #6：只注标题，供切换判断）。
        # 🔴 只在带指令时注：它的唯一用途是"要不要 switch 过去"，没有 switch 工具
        # 时是纯噪声（而且实测这份列表会被子调用产生的噪声账本挤满，MQ-L8）。
        # ~~🔴 G16.2：候选/LEDGERS 列表随 (a) 走，**只给用户面**~~ —— 它的唯一用途是
        # "要不要 switch 过去"，工具面既不该切、也没有新的用户话可匹配
        # （`user_text` 在工具往返里逐字不变），是纯噪声（MQ-L8：还会被噪声本挤满）。
        # 🔴 **上面这一句自 2026-09-20 起只对「有激活账本」成立**（MQ-L79，见下）：
        # 它把"候选列表"和"切换号召"绑成了一格，而**无激活账本时前者不是噪声，
        # 是唯一出口**。历史推理保留在此（它对有激活本那一格仍然正确），
        # 修订后的完整三格表在紧接着的注释里。
        # 🔴 MQ-L79（2026-09-20，ADR-0032 §3.1「修订二之补正」）：候选段与「中途切换
        # 号召」此前绑在同一格，而两者的适用条件**相反**——
        #
        #   |            | 用户面 | 工具面·有激活本 | 工具面·无激活本 |
        #   | 切换号召+模板 |  注   |      不注      |      不注      |
        #   | 相关性候选段  |  注   |      不注      |   🔴 **注**    |
        #   | recency top-5|  注   |      不注      |      不注      |
        #
        # 有激活本时候选列表确实是噪声（任务中途切账本永远是错的，G16 结论不动）；
        # **没有激活本时它是模型唯一能看见的出口**。live 病例：首轮之后连续 12 个
        # `face=tool` 请求是 `ledger=(空) candidates=0 recent=0 instruction=528
        # template=1323` —— 一边催记账/建本，一边不给激活本、不给候选、不给入口，
        # 模型只剩 `bladex_memory_search`，搜了 21 次、烧掉 13 轮。
        # 这是 MQ-L7「只给正文不给工具」的镜像：**给了工具不给对象**。
        _no_active_ledger = led is None
        # Z2.3 第一步 "candidates"：预算放不下时把两个候选桶整体去掉（重建一次，quiet）。
        _omit_candidates = bool(set(LEDGER_BLOCK_TRIM_BUCKETS) & set(omit_buckets))
        if with_instruction and (_user_face or _no_active_ledger):
          if not _omit_candidates:   # 守卫 test_mq_l79 钉死上一行原文，故多包一层（缩进 2 格是刻意的）
              # MQ-L38（2026-09-04）：相关性位——按当前用户话在**全池**找"像同一件事"的本。
              # recency top-5 只能回答"最近在干什么"；08-25 的泰山旧本排第 9，用户原话逐字
              # 重提时模型面前没有"切回"这个选项，只能新建。两段并列，top-5 一字不动。
              matches = relevant_ledgers(self.pool, user_text,
                                         exclude_id=led.ledger_id if led else "")
              self._last_candidates = matches
              # 判据是「**渲染到了正文里**」，不是「算出来了」：空候选段不入块，
              # 模型看到的和没算过完全一样（下面 `if matches:` 才是真正的渲染点）。
              self._candidates_shown = bool(matches)
              matched_ids = {m.ledger_id for m in matches}
              if matches:
                  _put("candidates", "", _MATCHING_LEDGERS_HEADER)
                  # 批 HO（MQ-L99）：已关的本按内容命中时照列，行尾标明已关（见 `_closed_match_suffix`）。
                  _put("candidates",
                       *[f"- {m.ledger_id}: {m.title} [{m.score:.2f}]"
                         + (_closed_match_suffix(self.pool[m.ledger_id]) if m.closed else "")
                         for m in matches])
              # 🔴 排序键是数值不是 ISO 串（MQ-L46）：池里混入不同偏移时字符串序会错，
              # 而这里就是 recency top-5——排错比时间显示错更贵。
              # 🔴 MQ-L79：recency 段**只给用户面** —— 它答的是"最近在干什么"，
              # 工具往返里确实一字不变，放进去就是 MQ-L8「被噪声本挤满」的原路。
              others = [(iso_ms(lg.updated_at or lg.created_at), lid, lg.title or lid)
                        for lid, lg in self.pool.items()
                        if (led is None or lid != led.ledger_id)
                        and lid not in matched_ids    # 命中项只在相关段出现一次
                        # 批 Z · Z3：已关的本不进 recency 候选。🔴 批 HO（MQ-L99）起与相关性段
                        # **不再同一判据**：那一段按内容命中时照列已关本；这一段没有内容依据，
                        # Z3「不替人翻案」的理由在这里仍成立。
                        and lg.status != STATUS_CLOSED]
              # 批 Y · Y1d：项目键落空时，同 agent Global 绑着的那本置顶（见 `_GLOBAL_BOUND_SUFFIX`）。
              # 只在「本请求有项目信号 ∧ 项目键无激活本」这一格生效；无项目信号的 agent
              # （Pi / hermes 查的就是 Global）走不到这里，块逐字等于修前。
              _global_bound = ""
              if led is None and project_id:
                  _gl = self._active_ledger(activation_scope(agent_id, ""))
                  if (_gl is not None and _gl.status != STATUS_CLOSED
                          and _gl.ledger_id not in matched_ids):
                      _global_bound = _gl.ledger_id
              if others and _user_face:
                  others.sort(reverse=True)
                  rows = [(lid, title) for _, lid, title in others if lid != _global_bound][:5]
                  if _global_bound:
                      rows = [(_global_bound, (self.pool[_global_bound].title or _global_bound)
                               + _GLOBAL_BOUND_SUFFIX)] + rows[:4]
                  _put("recent", "", "Other recent ledgers (read one with "
                                     "bladex_ledger_read before switching):")
                  _put("recent", *[f"- {lid}: {title}" for lid, title in rows])
              # 🔴 G16.0：补 `agent` —— 本事件此前无 agent 字段，导致混合日志里
              # 按时序分段无效（两 agent 请求交错），codex 的段内位置至今 NO-DATA。
              # 🔴 MQ-L79：补 `face` —— 三格表的三个格子在日志里必须可分，
              # 否则"工具面无激活本那一格放了多少候选"这个读数没有分母。
              if not quiet:
                  logger.info("agency_ledger_candidates", agent=agent_id,
                              face="user" if _user_face else "tool",
                              relevant=[m.ledger_id for m in matches],
                              # 批 HO（MQ-L99）：其中哪些是已关的本（读数 = 交接/返工找回的分子）
                              relevant_closed=[m.ledger_id for m in matches if m.closed],
                              top_score=(matches[0].score if matches else 0.0),
                              recent=(min(len(others), 5) if _user_face else 0),
                              # Y1d：置顶那本的来源可指认（原则 13 层 4）——值来自
                              # `activation_scope(agent, "")` 的绑定，不是 recency 排序。
                              global_bound=_global_bound,
                              candidate_chars=bd.chars.get("candidates", 0),
                              query_units=len(candidate_units(user_text)))
        _put("header", LEDGER_BLOCK_CLOSE)
        # 末行没有分隔符：把 `_put` 统一多计的那 1 个字符扣回 `header`
        # （块必以 `LEDGER_BLOCK_CLOSE` 收尾，故这一扣恒落在 header 上且不会为负）。
        bd.chars["header"] -= 1
        content = "\n".join(lines)
        # 🔴 恒等式的**自检**（feedback_instrument_reference_frame：仪器要能证明自己
        # 没脱钩）。**不 assert**——高危 1「仪器不许改行为」：一把尺子对不上账不该
        # 把这一轮的请求打挂。对不上就喊，读数照给（差多少也一并报，好定位）。
        if bd.total() != len(content):
            logger.warning("agency_ledger_breakdown_mismatch",
                           ledger=bd.ledger_id, buckets=bd.total(),
                           content=len(content), delta=bd.total() - len(content))
        return {"role": "system", "content": content}, bd

    def insert_ledger_block(self, messages: list[dict], agent_id: str,
                            *, project_id: str = "", auxiliary: bool = False,
                            with_instruction: bool = True,
                            tier: str = "", session_id: str = "",
                            budget_chars: int | None = None) -> list[dict]:
        """把账本块追加到**消息数组真正的末尾**。

        `budget_chars`（批 Z · Z2.3）：动态层这一轮还剩多少预算（None = 没配窗口，不裁）。
        块放不下 ⇒ 先去候选段（`LEDGER_BLOCK_TRIM_BUCKETS`），还放不下 ⇒ 整块不注；
        每次裁剪打一行 `agency_surface_trimmed`。顺序与理由见 `surface_budget.TRIM_ORDER`。

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

        🔴 **2026-09-15 更正**：此处原先写着"追加到末尾对上游 prefix cache 是**友好**的"。
        那句话**没有测量支撑，且已被实测证伪** —— 按刚性原则 12，断言性 docstring 是缺陷的气味，
        所以先把话改对，不等成因分离的结论。**结构事实**仍然成立：本轮请求里块之前的前缀逐字不变。
        但"结构不变 ⇒ 缓存友好"是一步**未验证的推断**，因为跨轮可复用的只是"上一轮请求去掉块"
        的那个真前缀 —— 上游肯不肯按真前缀部分命中，是它的实现细节，不是我们能从位置推出来的。
        **实测（`docs/benchmarks/swebench-ledger-20260913.md` §10/§10b，复核方逐格复算）**：
        ARK glm-5.3-flash 上账本开臂缓存命中 73.4% / 76.2%（零缓存轮 97 / 22）vs 关臂 97.6%，
        每轮未缓存输入 6–10×；**同一行为在 deepseek-v4-flash 上没有这个效应**（两臂 89.8% vs 91.1%）。
        成因未分离（块截断前缀 / 轮间隔变长被驱逐 / 前置 about 块逐轮变，三者修法不同）——
        判据与判决表见 `docs/planning/task-ca12-cause-separation-20260915.md`，处置归 MQ-CA12 / MQ-L74。
        **在那份分离做完之前，不得据本段改注入位置。**

        aux 轮不注（有令无器纯属注意力税——与工具面 gating 同一条理由）。

        🔴 MQ-L28 绑定会话 gate（2026-08-29 Jason 拍板，Pi 天气循环事故）：
        集合外档位（无工具面）只在「激活账本是**本会话**绑的」时注只读正文——
        C 方案的连续性场景（同会话"继续"被路由到弱档）保住；**继承自其它会话
        的绑定不注**：弱模型读到旧任务 goal 会照办、又没有工具去 switch/关闭，
        只能无限循环旧任务（live 实证：doubao-lite 把昨天完结的"北京天气查询"
        账本当指令重新执行）。来源未知（老事件无 session）按继承保守处理。
        强档不变——有工具 + 首步指令，错配由模型自行裁决（设计如此）。"""
        # 🔴 F-A1：**每请求恰好覆写一次**（与 `last_ledger_face` 同形态）。先清空
        # 再按路径填 —— 若只在成功路径赋值，aux/被 gate 的那一轮就会**沿用上一轮的
        # 读数**，把 A 轮的构成算到 B 轮头上。那是 feedback_instrument_reference_frame
        # 里最贵的一类（参照系脱钩），且静默。
        self.last_ledger_breakdown = None

        def _gated(reason: str, bound_sess: str = "") -> list[dict]:
            # 🔴 MQ-L84（批 Q3A，2026-09-29）：**每一道「不注」的门**都经这里留痕、带门名。
            # MQ-L83 只补了门 ②；门 ①（aux）与门 ③（模块关 / 无器则无令）两跳都静默。
            # 唯一生产点（L83 守卫数的就是本函数里这一行）。零行为：返回原数组。
            logger.info("agency_ledger_block_gated",
                        reason=reason,
                        tier=tier, agent=agent_id,
                        bound_session=(bound_sess or "")[:24],
                        session=(session_id or "")[:24])
            return messages

        if auxiliary:
            return _gated("auxiliary")
        # 🔴 首步指令的判据（2026-08-30 拆 weak 工具面门时独立出来）。
        #
        # 此前调用方直接传 `with_instruction=tf_injected`（工具面注没注），两件事
        # 等价。weak 档工具面门拆掉之后那个等价**会顺手把首步指令也给到 weak**
        # ——那等于改了账本注入策略，而 Jason 同日明确"账本注入策略保持不变"。
        # 故在这里合成，判据两个都要：
        #   ① `tier ∈ LEDGER_TIERS`  —— 账本策略（集合外档位不给指令，语义不变）
        #   ② 调用方传进来的工具面结果 —— **MQ-L7 的"不许有令无器"不变式**：
        #      指令说的是"调用 bladex_ledger_switch"，工具没注就是发空头支票。
        #      拆门后仍有一条路径会"tier 在集合内但工具面没注"：
        #      `NO_TOOLFACE_AGENT_BASES` 里的 agent。
        #      （该名单 2026-09-11 已清空——原来唯一的成员 dsh 依据过期，见 MQ-A66；
        #        名单本身保留，所以这条不变式仍然要守。）
        # 拆开后的关系从"等价"变成"蕴含"：有令 ⇒ 必有器；有器不一定有令
        # （weak 档就是"有器无令"——安全，工具在那儿，用不用归模型）。
        # 放在生产点而不是调用点：调用方少传一个条件就静默退化，这类漏接
        # 2026-08-29 刚踩过（三条流式入口漏传 ctx）。
        with_instruction = bool(with_instruction) and ledger_tools_allowed(
            tier, self._ledger_tiers)
        _scope0 = self.scope_of(agent_id, project_id)
        if not ledger_tools_allowed(tier, self._ledger_tiers):
            if not self.session_owns_active_ledger(agent_id, project_id, session_id):
                bound_sess = self.activation.last_session(_scope0)
                # 🔴 MQ-L83（批 W W3，2026-09-22）：**两条分支都要留痕**。
                # 此前这条日志只在"有激活账本"时打，而「**没有**激活账本」正是
                # MQ-L79 认定的「模型最需要候选面」那一格 ⇒ 连"我没给"都不记。
                # live 后果：窗口内 codex 116 轮末条 user 剥完为空、0 轮判 aux，
                # 而 `agency_goal_envelope_stripped` 在 codex 上一条都没有
                # ⇒ 只剩本门能解释，可它不留痕 ⇒「codex 为什么不触发」在存量
                # 日志里**不可判定**。与 MQ-L80 同一条理由（刚性原则 13：
                # 静默丢数据比丢数据更糟），只是那次丢的是失败事件、这次是跳过事件。
                # **零行为**：只加一条 info，门控逻辑与返回值一行不动。
                if self._active_ledger(_scope0) is None:
                    reason = "tier_not_allowed_no_ledger"
                elif bound_sess:
                    reason = "inherited_binding"
                else:
                    reason = "unknown_binding_origin"
                return _gated(reason, bound_sess)
        # V-L4：**先记这一轮，再渲染**——计数是"到这一轮为止在这本账本上过了多少轮"。
        # 记在这里而不是 `ledger_injection_message` 里，是因为这是每请求恰好一次的
        # 位置（aux 已在上面返回，三个入站端点各调一次）；渲染函数会被测试反复调用，
        # 把副作用放进去会让计数随调用次数漂移。
        from bladex_core.ledger_runtime import user_turn_index  # noqa: PLC0415
        # 🔴 G16.0（MQ-A56 / 父条目 MQ-L48，2026-09-09）：**只观测，不判决**。
        # 本批不改 `with_instruction` 的取值，只把两个候选判别器的结果打进日志，
        # 让 live 流量替我们在四个 agent 上对账，再由 G16.2 接门控。
        #
        # 候选 B（本行采用）= `user_turn_index` 与上次所见不同 —— MQ-L48/E0.2 已建立
        #   的参照系。键取 **(激活作用域, session_id)** 而不是 V-L4 那处的 `ledger_id`：
        #   无账本分支（`_FIRST_STEP_NO_LEDGER`）也要判，那时根本没有 id；
        #   也不用裸 `session_id`——它是指纹撞桶的重灾区，只能当候选来源不能当判据。
        # 候选 A（否决，仍打进日志留证）= "末条消息 role == 'user'"。看着更简、无状态，
        #   **实测在真实门控条件下恒为假**：E 档 88 个注块轮里它 0 次触发，
        #   因为 claude-code 在消息尾部追加 `role=system` 的 system-reminder
        #   （88 轮里 20 次末条是 system，**连那个真实用户轮也是**）。
        #   ⇒ `face_last_role` 保留为**反面证据**，不是备选实现。
        # 🔴 **重试沿用上次判定**（2026-09-09，被 `test_no_match_renders_exactly_the_old_block`
        # 抓出来的设计缺陷）：工具往返是 "uti 不变 **且消息变长**"，而**重试**是
        # "uti 与消息数组**完全相同**"。只按 uti 判会把重试也当工具往返 ⇒ 一个真实
        # 用户轮被 185 秒超时打断后重发（MQ-A57 实测 11 次 / 单会话），重试就丢掉 (a)。
        # 于是记 `(uti, len(messages))` 并连上次的 face 一起存：完全相同 ⇒ 原样沿用。
        _uti = user_turn_index(messages)
        _face_key = (self.scope_of(agent_id, project_id), session_id or "")
        _shape = (_uti, len(messages or ()))
        _prev = self._last_seen_face_turn.get(_face_key)
        if _prev is not None and _prev[0] == _shape:
            _new_user_turn = _prev[1]          # 逐字重试：沿用，不翻面
        else:
            _new_user_turn = (_prev is None) or (_prev[0][0] != _uti)
        self._last_seen_face_turn[_face_key] = (_shape, _new_user_turn)
        _last_role = ""
        if messages:
            _m = messages[-1]
            _last_role = str((_m.get("role") if isinstance(_m, dict)
                              else getattr(_m, "role", "")) or "")

        led0 = self._active_ledger(self.scope_of(agent_id, project_id))
        if led0 is not None:
            # E0.2（MQ-L48）：按**用户轮**计——同一用户轮内的工具往返（每个工具调用
            # 一次请求）不重复计。判据 = (session, user_turn) 与上次所见不同；用
            # `!=` 而非 `>`：同 session 压缩会让 user_turn 回落，那也是新的一轮。
            _seen = (session_id, user_turn_index(messages))
            if self._last_seen_user_turn.get(led0.ledger_id) != _seen:
                self._last_seen_user_turn[led0.ledger_id] = _seen
                self._turns_since_update[led0.ledger_id] += 1
        _face = "user" if _new_user_turn else "tool"
        _utext = _last_user_text(messages, for_goal=True)
        msg, breakdown = self._build_ledger_block(
            agent_id, project_id=project_id, with_instruction=with_instruction,
            user_text=_utext, face=_face)
        if msg is None:
            # 门 ③：`_build_ledger_block` 只在两处返回 None——模块关，或无激活账本且不给指令。
            return _gated("module_off" if not (self.ledger_on and self.toolface_on)
                          else "no_ledger_no_instruction")
        _step = dynamic_trim_step(len(msg["content"]), breakdown.chars, budget_chars)
        if _step == "candidates":
            log_trim("candidates",
                     chars=sum(int(breakdown.chars.get(b, 0) or 0) for b in LEDGER_BLOCK_TRIM_BUCKETS),
                     budget=budget_chars, estimate=len(msg["content"]), session=session_id)
            msg, breakdown = self._build_ledger_block(
                agent_id, project_id=project_id, with_instruction=with_instruction,
                user_text=_utext, face=_face, omit_buckets=LEDGER_BLOCK_TRIM_BUCKETS, quiet=True)
            if msg is None:
                return messages
        elif _step == "ledger_block":
            log_trim("ledger_block", chars=len(msg["content"]), budget=budget_chars,
                     estimate=len(msg["content"]), session=session_id)
            self._last_candidates = []
            self._candidates_shown = False
            return messages
        self.last_ledger_breakdown = breakdown
        out = [*messages, msg]
        # 🔴 观测缺口 MQ-L5 #3：此前这一步**一条日志都没有** ⇒ live 上无法证明
        # 账本正文真的进了请求、更无法证明它在哪个位置（Hub 存的是注入**前**的
        # `original_messages`）。MQ-L10 那次就是因为看不见，才让"块被埋 65 条"
        # 潜伏到靠翻消息结构才发现。`after=0` 是该条的**判据本身**。
        led = self._active_ledger(self.scope_of(agent_id, project_id))
        # 🔴 F-A1（MQ-A35）：`rev` 与 `sections` 是 handoff bench §4.2 与 A34 ①②
        # 裁剪对象的**唯一数据源**。`chars=` 一个总数只能回答"大不大"，回答不了
        # "大在哪一段"——而裁剪要动的正是某一段（MQ-L5「二义的观测字段等于没有观测」
        # 的量级版）。`(ledger, rev)` 一对让任一轮的读数能在 Hub 事件流里重放。
        logger.info("agency_ledger_block_injected",
                    ledger=led.ledger_id if led else "",
                    rev=breakdown.rev,
                    with_instruction=with_instruction,
                    # G16.0 观测（零行为改变）：`face` 是**将来**要接门控的判别器，
                    # `face_last_role` / `uti` 是它的原料与反面证据。判据见 G16.3：
                    # `face=user` 次数应 ≈ 用户轮数，跨 agent 比值应 ≈ 1。
                    face="user" if _new_user_turn else "tool",
                    face_last_role=_last_role, uti=_uti,
                    # G16.2：这一轮实际注的是哪一版指令。`switch` = 含 (a) 比对/切换
                    # 半句的完整原文；`record` = 只留 (b) 记账半句的工具面版；
                    # `none` = 没注指令（无工具面 / 只读正文）。判据 4「工具轮 switch = 0」
                    # 要的分母就是这一列。
                    instruction_kind=("none" if not with_instruction else
                                      "switch" if (_new_user_turn
                                                   or not flag_enabled("BLADEX_LEDGER_FACE_SPLIT"))
                                      else "record"),
                    pos=len(out) - 1, total=len(out), after=0,
                    chars=len(msg.get("content") or ""),
                    # MQ-CA13 ①（批 R · R1，2026-09-18）：账本块**内容哈希**。此前只有
                    # chars / rev / sections，`probe_cache_gap.block_same` 是拿它们代理"块逐字
                    # 不变"——字数与段长相同不等于内容相同（同 rev 下候选/recent 段也会变）。
                    # 没有这一列，"块真没变"始终是代理证据。sha256[:16]，与 Hub
                    # `LedgerAnchor.added_hash` 同款截断。
                    block_hash=hashlib.sha256((msg.get("content") or "").encode("utf-8")).hexdigest()[:16],
                    sections=json.dumps(breakdown.chars, separators=(",", ":")))
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
