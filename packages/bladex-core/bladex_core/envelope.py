"""Envelope — agent 内部协议信封的剥离（ADR-0024 T3）。

## 为什么需要它

T0 立卡时认为 claude-code 的记忆盲区是"用户消息太长"（p50 70K 字符、84.6% 被
`_MAX_CONTENT_CHARS=500` 整条丢弃），处置方向定为"结构化剥离 + 语义分段"。

T3 用归档 Memory Hub 全量复核，推翻了这个判断：

    claude-code >500 字符的 user 消息，80,751,734 字符里只有 1.79% 是真实用户意图，
    其余 98.21% 是 **agent 内部协议信封**——安全检查子 agent 的会话转录、
    slash 命令回显、后台任务通知、环境注入块。
    剥掉信封后 **91% 的消息本来就落在现有 20–500 蒸馏窗口内**。

即：**盲区的绝大部分不是"长消息"，是信封。**不需要提高阈值，不需要复杂的语义
分段——只需要把信封拆掉。

## 三层处置（本模块负责第 2 层）

    1. 整轮 auxiliary 判定   末条 user 是纯信封 → 整轮跳过（identity.classify_auxiliary）
    2. **消息级信封剥离**    本模块：剥掉信封，留下真实意图
    3. 剩余超长分段          剥离后仍 >max 的少数（实测 9%）按空行边界切段

## 硬约束

**只在蒸馏路径使用，绝不用于转发给上游的正文**（ADR-0008 §6.5 / ADR-0025 I1：
不改写 agent 正文）。转发什么由装配层决定，本模块只决定"记什么"。

特别地，`<bladex-memory>` 也在剥离表里：真实数据中有 **24 条 user 消息回带了
BladeX 自己的注入块**（`strip_previous_injection` 限 system 角色以防误伤正文，
claude-code 会把含注入的历史当 user 内容回传）。不剥它 = 自己注入的记忆被蒸馏回
Memory Index = ADR-0025 §6 担心的自反馈闭环，而它**已经在真实数据里发生**。

## agent 适配原则

规则按 agent 行为增长，不要求任何 agent 改协议（CLAUDE.md 刚性原则 9）。
hermes:default 实测 0 命中、残留 100%——本模块对聊天式 agent 零影响、零回归。
"""

from __future__ import annotations

import re

# 剥离表：(kind, 正则)。全部为成对标签块，非贪婪跨行匹配。
# 顺序无关（各自独立成对），但保持稳定以便日志可读。
_ENVELOPE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # Claude Code 安全检查子 agent：整段会话转录 + "响应必须以 <block> 开头"。
    # 实测 439/507 条超长消息含此块，是信封体积的绝对主力。
    ("transcript", re.compile(r"<transcript>.*?</transcript>", re.S | re.I)),
    # 环境注入（CLAUDE.md 内容、目录状态、待办提醒等）
    ("system_reminder", re.compile(r"<system-reminder>.*?</system-reminder>", re.S | re.I)),
    ("user_claude_md", re.compile(r"<user_claude_md>.*?</user_claude_md>", re.S | re.I)),
    # 后台任务完成通知（含 task-id / output-file / status / summary 子标签）
    ("task_notification", re.compile(r"<task-notification>.*?</task-notification>", re.S | re.I)),
    # slash 命令回显与本地命令输出
    ("slash_command", re.compile(r"<command-(name|message|args)>.*?</command-\1>", re.S | re.I)),
    ("local_command", re.compile(
        r"<local-command-(stdout|stderr|caveat)>.*?</local-command-\1>", re.S | re.I)),
    # ── BladeX 自己的注入块回声 —— 不剥则形成自反馈闭环（见模块 docstring）──
    #
    # 🔴 **注入面每加一个标记，这里就必须加一条落点**，否则我们注入的内容会被
    # agent 回传、被自己蒸成 fact。2026-08-28 普查发现三个标记只盖住了一个：
    # `<bladex-ledger>`（账本块，每轮注、体量最大）与 `<bladex-context-summary>`
    # 从来没有落点。这与 MQ-S46 是同一个交接缺陷的第三、四例
    # （判定/生产方与处置方之间没有对账），故一并机制化：
    # `test_injection_markers_have_strip_landings` 对着 `agency.INJECTION_MARKERS`
    # 逐个验，加标记不加落点直接红。
    ("bladex_memory_echo", re.compile(r"<bladex-memory>.*?</bladex-memory>", re.S | re.I)),
    ("bladex_ledger_echo", re.compile(r"<bladex-ledger>.*?</bladex-ledger>", re.S | re.I)),
    ("bladex_context_summary_echo",
     re.compile(r"<bladex-context-summary>.*?</bladex-context-summary>", re.S | re.I)),
    # V-P6：BladeX 自我介绍（AGENT/TOOLS/SKILLS.md 稳定层）
    ("bladex_about_echo", re.compile(r"<bladex-about>.*?</bladex-about>", re.S | re.I)),
    # ── MS-2：codex 环境注入块（工作目录/沙盒策略/审批模式等，每轮重复且无记忆价值）──
    ("codex_environment_context",
     re.compile(r"<environment_context>.*?</environment_context>", re.S | re.I)),
    # ── MQ-S46 ③（2026-08-28）：codex 注入的 AGENTS.md 全文 ──
    #
    # 形态：`# AGENTS.md instructions for <repo>\n<INSTRUCTIONS>…仓库协作文档全文…`
    # live（`--tag-census`，codex:guardian 600 轮）：**600 开 / 600 闭**——每轮恰好
    # 一次、成对闭合、剥离后残留 **11865 字符**，是 codex_history_injection 修好后
    # 唯一还站着的缺口（其余"未覆盖"标签经 v2 仪器复核全是假阳性）。
    #
    # 为什么该剥：它是**仓库文件**，不是用户说的话；每轮重复注入；真要它的内容
    # 该走 file_ref 通道，而不是当用户意图蒸成 fact。
    #
    # 🔴 成对剥离而非整条剥除：`<INSTRUCTIONS>` 标签名通用，万一用户自己写了这个
    # 标签，成对剥离只吃掉标签内、标签外的真实意图照留。整条剥除没有这个余地。
    # 剥完前面还剩标题行（≈50 字符）——高于 `_MIN_CONTENT_CHARS=20`，会进蒸馏，
    # 但那是一条低价值路径串、且判重会把 600 次压成 1 条，代价小于把正则写贪。
    ("codex_agents_md", re.compile(r"<INSTRUCTIONS>.*?</INSTRUCTIONS>", re.S | re.I)),
]

# 未闭合的信封开标签（流被截断 / agent 只发了半个块）。命中即从该处截断到末尾——
# 保守：宁可少蒸一点，也不要把半个信封当成用户意图。
_UNCLOSED_OPENERS: tuple[str, ...] = (
    "<transcript>", "<system-reminder>", "<task-notification>", "<user_claude_md>",
    # MS-2：codex 的环境块实测存在被截断的形态（只发了开标签）
    "<environment_context>",
)

#: 信封标记允许出现的**最深位置**（字符）。超过它 ⇒ 这一轮是在谈论信封，不是信封。
#:
#: 🔴 **单一真相源**：`bladex_proxy.identity._ENVELOPE_MAX_OFFSET` 从这里 import。
#: 标定过程与双峰分布见 `identity.classify_auxiliary` 的注释（MQ-A19）——
#: 改这个值前先重跑那份分布，它是**标定常数**，不是随手取的圆整数。
#:
#: 常量落在 core 而不是 proxy：envelope 是更底层的纯函数层，identity 依赖它、
#: 反之不成立。两边各写一个 2000 就是「同一个参数两条路径两个默认值」
#: （CLAUDE.md 刚性原则 12 的形状，`--concurrency` 那次的教训）。
ENVELOPE_MAX_OFFSET = 2000

# ── MS-2：整条即脚手架的消息（cowork / codex 的"生成建议"类内部提示）──
#
# 这些不是用户说的话，是宿主 app 自己塞进 user 角色的任务指令。它们**长度达标、
# 语义自洽**，所以既躲过长度过滤又能蒸出像模像样的"偏好"——库里那批
# `## Goal` / `Preference order for skills` 的假 preference 就是这么来的（M0-3 在清）。
#
# 与上面的成对标签不同，这类没有边界标记，只能靠稳定前缀认。命中即**整条剥除**
# （不是剥掉一段）：这类消息里没有任何一部分属于用户意图。
# 判据保守：短语要足够长且带产品口吻，避免误伤真实用户消息；
# **且必须出现在前 `ENVELOPE_MAX_OFFSET` 字符内**（见该常量）。
#
# 🔴 每条 `DISTILL_ONLY_AUX_RULES` 规则都必须在这里（或成对标签表里）有落点，
# 否则「user 侧交给 envelope 剥离」是空头支票。对账由
# `identity.DISTILL_ONLY_ENVELOPE_CONTRACT` + `test_distill_only_envelope_contract.py`
# 机制化守住（ADR-0030 H1 生产者/消费者对账）。
_WHOLE_MESSAGE_MARKERS: tuple[tuple[str, str], ...] = (
    ("suggestion_scaffold", "generate 0 to 3 hyperpersonalized suggestions"),
    ("suggestion_scaffold", "get an understanding of the user's intent and goals"),
    # 2026-08-28 对齐 `identity._AUX_USER_PATTERNS` 的 `session_resume_recap`：
    # 那边是 "The user stepped away and is coming back"（无 ". recap"）。两边不一致
    # ⇒ identity 判 scaffold、envelope 不剥 ⇒ 又一张空头支票。收窄的一方要向
    # 宽的一方对齐，因为**判 aux 的那侧已经认定它是信封了**。
    ("recap_scaffold", "the user stepped away and is coming back"),
    # ── 2026-08-28：DISTILL_ONLY 六条空头支票的落点（见模块 docstring「对账」）──
    #
    # 这六条此前**只有指纹、没有剥离模式**：`classify_turn_disposition` 判 scaffold
    # （"user 侧是信封"），rebuild 于是不整轮丢、把 user 侧交给本模块——而本模块
    # 里根本没有它们。live 读数：codex:guardian 600 轮里 595 轮判 scaffold，
    # 字符留存率 **99.6%**，整段父 agent transcript 原样进 `user_messages`，
    # 蒸出 2299 条 `origin:user_direct`（占该 agent 全部 fact 的 88%）。
    #
    # 这是 `memory_index.py` 那段注释记录的「机制写完接线断」的**第二次复发**：
    # 上次是"集合建好了没人读"，这次是"读了、把活交出去、接活的这边是空的"。
    #
    # 🔴 codex 用**共同前缀**而非那两句完整从句。MQ-S45 的原话：
    # 指纹按一次采样写死，agent 换个变体就复发（em dash 是第二例，从句是第三例）。
    # live 已见三种从句（`added since` / `added since your last approval assessment` /
    # `whose request action you are assessing`），共同前缀盖住全部。
    ("codex_history_injection", "the following is the codex agent history"),
    # 以下四条逐字取自 `identity._AUX_USER_PATTERNS`，两处必须一致（对账测试守）。
    ("hermes_memory_save", "review the conversation above and consider saving to memory"),
    ("hermes_skill_library_update",
     "review the conversation above and update the skill library"),
    ("hermes_empty_response_retry",
     "you just executed tool calls but returned an empty response"),
    ("truncated_response_continue", "[system: your previous response was truncated"),
)

# ── MS-2：hermes 会话转录壳 ──
#
# 形态：`User: <真实问句>\n\nAssistant: <回答>\n\nUser: …`
# 这是 hermes 把历史会话包成一条 user 消息回放。**整条丢是错的**——第一段
# `User:` 后面就是用户真正问的那句话（实测"帮我核实 3000 万共益债公告"整条被
# miss，泰山那条线的起点因此从未入库）。
#
# 保守判据（两个条件同时满足才认）：以 `User:` 起头 **且** 含 `\n\nAssistant:`。
# 只要一个条件就动手会误伤真实消息——"User: 是我们数据库里的一张表，帮我加索引"
# 就是合法的用户输入。
_TRANSCRIPT_ASSISTANT = "\n\nAssistant:"
_TRANSCRIPT_USER_PREFIX = "User:"

# 分段：默认按空行切；单段上限与蒸馏窗口对齐由调用方传入。
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")

#: `strip_envelopes` 可能返回的**全部** kind。对账测试用它校验
#: `identity.DISTILL_ONLY_ENVELOPE_CONTRACT` 里的 value 不是拼错的字符串。
#:
#: 派生而非手写：手写一份就会漏掉新加的条目，而漏掉的表现是「对账测试绿、
#: 实际没落点」——尺子和被测系统各算各的（MQ-S4 的形态）。
ENVELOPE_KINDS: frozenset[str] = frozenset(
    [k for k, _ in _ENVELOPE_PATTERNS]
    + [k for k, _ in _WHOLE_MESSAGE_MARKERS]
    + ["unclosed_envelope", "hermes_transcript"]
)


def strip_envelopes(text: str) -> tuple[str, list[str]]:
    """剥掉 agent 内部协议信封，返回 (真实意图文本, 剥掉的信封种类)。

    纯函数、确定性、零 LLM —— 与 ADR-0019 L1 同性质，保证重建等价性。
    未命中任何信封时原样返回（`kinds` 为空），对聊天式 agent 零影响。
    """
    if not text:
        return "", []

    kinds: list[str] = []

    # MS-2 ①：整条即脚手架 —— 先判，命中就没有"剩下的部分"可谈了
    #
    # 🔴 位置门（2026-08-28 随六条新 marker 一起加）：标记必须落在前
    # `ENVELOPE_MAX_OFFSET` 字符内。没有它，**一条讨论这些指纹的消息**
    # （评审记录、本文件的 diff、任务卡引文）会被整条剥除——MQ-A19 就是为
    # 这个自指陷阱立的，`identity.classify_auxiliary` 那边早已有门。
    # 六条新 marker 让陷阱面积翻倍，所以门必须同时补上，不能"以后再说"。
    lowered_all = text.lower()
    for kind, marker in _WHOLE_MESSAGE_MARKERS:
        at = lowered_all.find(marker)
        if 0 <= at <= ENVELOPE_MAX_OFFSET:
            return "", [kind]

    for kind, pattern in _ENVELOPE_PATTERNS:
        text, n = pattern.subn("", text)
        if n:
            kinds.append(kind)

    # 未闭合开标签：从第一个出现处截断
    lowered = text.lower()
    cut = min(
        (lowered.index(op) for op in _UNCLOSED_OPENERS if op in lowered),
        default=-1,
    )
    if cut >= 0:
        text = text[:cut]
        kinds.append("unclosed_envelope")

    # MS-2 ②：hermes 转录壳 —— 剥壳**保留首段真实问句**（不是整条丢）
    unwrapped = _unwrap_transcript(text)
    if unwrapped is not None:
        text = unwrapped
        kinds.append("hermes_transcript")

    return text.strip(), kinds


def _unwrap_transcript(text: str) -> str | None:
    """`User: <问句>\\n\\nAssistant: …` → `<问句>`；不是转录形态返回 None。

    只取**第一段** User 内容：后面的轮次在各自的真实用户轮里独立存在，
    重复蒸馏只会制造判重压力（与 claude-code transcript 同一条推理）。
    """
    stripped = text.lstrip()
    if not stripped.startswith(_TRANSCRIPT_USER_PREFIX):
        return None
    if _TRANSCRIPT_ASSISTANT not in stripped:
        return None
    body = stripped[len(_TRANSCRIPT_USER_PREFIX):]
    return body.split(_TRANSCRIPT_ASSISTANT, 1)[0].strip()


def is_pure_envelope(text: str) -> bool:
    """整条消息都是信封（剥完为空）—— 无记忆价值，调用方应整条跳过。"""
    clean, kinds = strip_envelopes(text)
    return bool(kinds) and not clean


def segment_long_text(
    text: str,
    max_chars: int,
    max_segments: int = 6,
) -> list[str]:
    """把剥离后仍超长的文本按空行边界切段（实测只影响 9% 的超长消息）。

    - 优先在空行处断开（保语义完整），单段超限则硬切。
    - `max_segments` 封顶：超长粘贴（实测残留最大 28 万字符）不该产出上百个候选
      去打爆蒸馏预算；取前 N 段——用户意图通常在开头。
    - 确定性：同一输入恒得同一切分。
    """
    if len(text) <= max_chars:
        return [text] if text else []

    segments: list[str] = []
    buf = ""
    for para in _PARAGRAPH_SPLIT.split(text):
        para = para.strip()
        if not para:
            continue
        # 单段本身超限 → 硬切
        if len(para) > max_chars:
            if buf:
                segments.append(buf)
                buf = ""
            for i in range(0, len(para), max_chars):
                segments.append(para[i:i + max_chars])
                if len(segments) >= max_segments:
                    return segments[:max_segments]
            continue
        if len(buf) + len(para) + 2 > max_chars:
            if buf:
                segments.append(buf)
            buf = para
        else:
            buf = f"{buf}\n\n{para}" if buf else para
        if len(segments) >= max_segments:
            return segments[:max_segments]

    if buf:
        segments.append(buf)
    return segments[:max_segments]


def prepare_distill_inputs(
    text: str,
    min_chars: int,
    max_chars: int,
    max_segments: int = 6,
) -> tuple[list[str], list[str]]:
    """蒸馏输入预处理总入口：剥信封 → 判过短 → 必要时分段。

    返回 `(候选文本列表, 剥掉的信封种类)`。空列表 = 本条无蒸馏价值。

    替代原先 `consolidation_proxy` 里那句
    `if not (_MIN <= len(content) <= _MAX): continue` 的**整条丢弃**——
    正是它让 claude-code 84.6% 的 user 消息从未进入蒸馏（提取率 7.5%）。
    """
    clean, kinds = strip_envelopes(text)
    if len(clean) < min_chars:
        return [], kinds
    if len(clean) <= max_chars:
        return [clean], kinds
    segs = [s for s in segment_long_text(clean, max_chars, max_segments)
            if len(s) >= min_chars]
    return segs, kinds
