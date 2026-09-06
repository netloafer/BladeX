"""bladex_* 工具面 —— LLM 可主动调用的记忆/账本能力（ADR-0032 §1/§3.2；批二卡 V-P1）。

# 🔴 三红线（ADR-0032 §1，工具面是它们的第一执行点）

1. 工具只读写记忆/账本对象——本模块**不得**注册任何产生 agent 领域动作的工具
   （不改文件、不执行命令、不产生计划外任务）。
2. Goal 仅用户可改——`bladex_ledger_update` 对 goal 段直接拒绝（复用 V-L1 写保护）。
3. 模型不调用时零行为差异——模块开关 `BLADEX_MODULE_TOOLFACE`（默认关）+
   注入是纯追加，撤掉即回到被动注入形态。

# 实测约束（agent-compat-survey-20260825）

- D3：tools 数组会话内逐字节稳定 ⇒ 注入**追加到末尾、不重排**（`inject_tools`）。
- aux 轮不注（CC 66.7% 轮是内部子调用，注了是注意力税+误调用风险）。
- ~~weak 档模型不注（工具调用不可靠）~~ → **2026-08-30 拆除**：砍掉 Memory Index
  主动注入后，工具面成了取记忆的唯一通道，对 weak 关着 = 弱档彻底没有记忆。
  改为**按工具族**：记忆族无条件注入，账本族仍受 `BLADEX_LEDGER_TIERS` 管。
- dsh 只有 `run_code` 可直呼 ⇒ 首版对 dsh 不注（`NO_TOOLFACE_AGENT_BASES`），
  其专项注入形态另立卡。
- 工具描述刻意短：注入的每个字都花注意力预算（ADR-0029），描述是给模型的不是文档。

结果侧敏感度：执行结果也是注入（ADR-0032 §3.2-7），`dispatch` 强制传
`allowed_exposure`，handler 契约要求按它过滤——**过滤在 handler 内做**
（数据在那儿），本模块把参数钉进签名让漏传成为类型错误。
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

import structlog

logger = structlog.get_logger()

BLADEX_TOOL_PREFIX = "bladex_"

#: 首版不注工具面的 agent base（dsh：工具只能从 run_code 程序内调，直呼形态无效）。
NO_TOOLFACE_AGENT_BASES: tuple[str, ...] = ("dsh",)

#: 模型档位的闭集（与 `bladex_core.routing._TIER_ORDER` 同源，别在别处另抄一份）。
LEDGER_TIER_NAMES: tuple[str, ...] = ("weak", "medium", "strong")

#: 🔴 `auto` 的语义 —— **单一真相源**（Jason 2026-08-26 拍板）。
#:
#: 它会随测量结果变（今天是"除 weak 外"，若实测 weak 档能可靠调工具就会变成全部），
#: 所以定义只能有一处，文档由 `.env.example` 与它对账（`flags.py` 那套同款）。
#: 改它的语义时**必须在 `.env.example` 注释里写明改了什么、依据是哪次读数**——
#: 否则用户升级后行为变了却不知道为什么。
#:
#: 🔴 当前依据（2026-08-30 更正——这段此前一直写着"尚无读数支撑"，而读数
#: **08-26 就有了**，注释没跟上，今天还误导了一次改动）：
#: ADR-0032 原文「weak 档模型工具调用不可靠」**已被实测证伪**（MQ-L6：
#: Pi 恒判 weak，配 all 实跑——调用率 43%（判据 ≥30%）、参数正确率 100%
#: （≥90%）、误调用 0，三项判据事前写死、事后未改）。
#: auto 仍保持 {medium, strong} 是 **Jason 2026-08-26 的拍板**（0.1.0 跑通优先，
#: 优化放 0.2.0）—— 是**已知的保守默认**，不是未验证的默认。两者的区别要写清楚，
#: 否则下一个人会重跑同一个实验。
AUTO_LEDGER_TIERS: tuple[str, ...] = ("medium", "strong")

#: 🔴 `NO_TOOLFACE_TIERS` 已删除（2026-08-30 按工具族拆）。
#: 它表达的概念——"不注**工具面**的档位"——不再成立：档位管的是**账本族**，
#: 记忆族无条件注入，"整个工具面被档位关掉"这件事没有了。
#: `AUTO_LEDGER_TIERS` 保留：它仍然是账本族（工具 + 首步指令）的判据。
#: 不留兼容别名：无生产消费者的常量就是僵尸，`bladex status` 还会把它打出来
#: 误导观感——KEYWORD_CHANNEL 那次已经付过学费（见 test_flag_defaults 头注）。


def resolve_ledger_tiers(raw: str = "") -> frozenset[str]:
    """解析 `BLADEX_LEDGER_TIERS` → **强制注入账本面的档位集合**。

    三种写法是**同一个机制**的具名预设，不是三条代码路径：

        auto            -> {medium, strong}   （当前默认，见 AUTO_LEDGER_TIERS）
        all             -> {weak, medium, strong}
        none            -> {}                 （只保留"有账本就延续"的只读行为）
        "medium,strong" -> {medium, strong}   （与 auto 等价）
        "strong"        -> {strong}

    **集合内** = 账本族强制注入（账本三工具 + 首步指令，模型可创建/切换/更新）；
    **集合外** = 有账本就注只读正文、没账本什么都不注（C 方案的既有语义）。

    🔴 **它只管账本族**（2026-08-30 修订）。记忆族 `bladex_memory_search`
    不受本集合影响——它现在是模型取记忆的唯一通道，按档位关掉它就不是
    "这件事需不需要账本"的策略问题了。本旋钮的立卡理由（"用户决定哪些档位
    拿到**账本面**"）本来说的也只是账本。
    🔴 账本族有**两个**放行口，别只记住档位这一个：
      ① 档位在集合内（本函数）；
      ② **本会话绑着激活账本**（`agency.session_owns_active_ledger`）——
         档位在集合外时也给账本工具，否则模型看得见账本却切不走
         （MQ-L7 的补完，2026-08-30）。
    "有令无器"（MQ-L7）仍不会复发：首步指令只走 ①，而 ① ⊆ (① ∪ ②)，
    有令必有器；②只补器不补令，是**有器无令**——安全。

    写错的档位名**响亮失败**（误配置不许静默降级成"什么都不注"）。
    """
    import os
    raw = (raw or os.environ.get("BLADEX_LEDGER_TIERS", "") or "auto").strip().lower()
    if raw == "auto":
        return frozenset(AUTO_LEDGER_TIERS)
    if raw == "all":
        return frozenset(LEDGER_TIER_NAMES)
    if raw == "none":
        return frozenset()
    tiers = {t.strip() for t in raw.split(",") if t.strip()}
    bad = tiers - set(LEDGER_TIER_NAMES)
    if bad:
        raise ValueError(
            f"BLADEX_LEDGER_TIERS 含未知档位 {sorted(bad)}；"
            f"可用：{list(LEDGER_TIER_NAMES)} 或 auto/all/none")
    return frozenset(tiers)


def is_bladex_tool(name: str) -> bool:
    return bool(name) and name.startswith(BLADEX_TOOL_PREFIX)


# ── 工具 schema（OpenAI function 格式为源，Anthropic 格式派生）─────────────────

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "bladex_memory_search",
            # 🔴 2026-08-30 重写（配合"Memory Index 不再主动注入"）。三处改动，
            # 每处都对着一个实测问题：
            # ① 「ONLY way / not in your context」—— 原文 "Use when past context
            #    would change your answer" 预设模型知道有 past context 存在。
            #    砍掉注入后模型**没有**被动供给了，得先告诉它这条通道的地位。
            # ② 具体触发场景取代抽象条件 —— 原文那句等于"需要时就用"，
            #    实测调用率 4.0%（250 轮给了工具只有 10 轮调，其中记忆搜索 2 次）。
            #    那个数被"模型已被喂饱"污染过，但描述本身确实没给可判定的时机。
            # ③ 「Facts state when they were recorded」—— 天气病例的直接教训：
            #    观测型事实没有 valid_until（MQ-S50 邻接，修复归 0.3.0），
            #    正文里带着日期，得让模型自己比对，别把旧观测当现行。
            # 长度从 ~45 词涨到 ~85 词：它在**稳定层**、会话首轮冻结进前缀、
            # 享 prompt cache，边际成本低；而它现在是记忆的唯一入口，值这些字。
            "description": "Read the user's long-term memory (facts and matter "
                           "cards) that BladeX keeps across all their sessions and "
                           "agents. This tool is the ONLY way to reach it — nothing "
                           "from earlier sessions is in your context. Call it before "
                           "answering when the user refers to past work, a decision, "
                           "a name, or 'like last time'; when they ask about their "
                           "own setup or preferences; or when you would otherwise "
                           "say you don't know or ask them to repeat something. "
                           "Facts state when they were recorded — check the date "
                           "before treating an old observation as current. "
                           "Pass exactly one of: query (semantic), keyword (exact "
                           "term: a name, model, file, tool), matter_id, or fact_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string",
                              "description": "semantic search: what to look for"},
                    "keyword": {"type": "string",
                                "description": "exact term (a name, tool, model, "
                                               "file); returns matters and facts "
                                               "that share it"},
                    "matter_id": {"type": "string",
                                  "description": "list a matter's facts (m-...)"},
                    "fact_id": {"type": "string",
                                "description": "fetch one fact with its keywords"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 15,
                              "default": 5},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bladex_ledger_read",
            "description": "Read a task ledger (Goal/Core/Verified/Open/Next). "
                           "Omit ledger_id for the active one; pass an id from the "
                           "listed ledgers to inspect it BEFORE switching to it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ledger_id": {
                        "type": "string",
                        "description": "ledger to read ('' or omitted = the active one)",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bladex_ledger_update",
            # 🔴 2026-08-30 补"完成 = 两次编辑"规则（Jason live 反馈，MQ-L29 第一案）。
            # 实测形态：任务做完了，结果写进 Verified，**Next/Open 里的对应条目
            # 还留着** ⇒ 下一个读到账本的模型（尤其弱档只读）把已完成的 Next
            # 当待办再执行（与 MQ-L28 天气循环同根）。
            # 🔴 写在**顶层 description** 而不是 `section` 枚举里：Jason 的判断是
            # "很多时候都是更新时候的遗忘"——遗忘发生在**决定调这个工具**的那一刻，
            # 顶层描述是模型考虑用不用它时读的第一段。同一条规则在 schema 内
            # **只说一次**（说两遍是注意力税），跨面的一致性由首步指令那一侧承担
            # （`agency._FIRST_STEP_WITH_LEDGER`，守卫见 test_toolface）。
            "description": "Update the active task ledger: add or remove an entry "
                           "in core/verified/open/next. Finishing something is TWO "
                           "edits, not one: add it to verified AND remove the "
                           "next/open entry it resolves — an entry that is in "
                           "Verified does not belong in Next or Open. The Goal may "
                           "only be changed when the user asks for it in this turn "
                           "— see goal / goal_change_quote.",
            "parameters": {
                "type": "object",
                "properties": {
                    # 段语义与 switch 那侧保持一致（MQ-L19：只写段名 = 让模型猜）。
                    "section": {
                        "type": "string",
                        "enum": ["core", "verified", "open", "next"],
                        "description": "core = facts true for the whole task "
                                       "(context a successor would need, not steps); "
                                       "verified = what you have actually "
                                       "established, with a ref; "
                                       "open = unresolved questions and decisions; "
                                       "next = the concrete steps ahead.",
                    },
                    "op": {"type": "string", "enum": ["add", "remove"]},
                    "text": {"type": "string",
                             "description": "entry text (for add)"},
                    "index": {"type": "integer",
                              "description": "entry position (for remove)"},
                    "ref": {"type": "string",
                            "description": "evidence ref (tool result / test id)"},
                    "rev": {"type": "integer",
                            "description": "the rev you last read (ledger "
                                           "header); include it — another agent "
                                           "may be updating this ledger too"},
                    # 🔴 Goal 改动通道（Jason 2026-08-26 拍板）：用户又发了 prompt、
                    # 模型判断这不是新任务而是**调整了原目标**时走这里。没有用户请求
                    # 就不许改——`goal_change_quote` 必须真的出现在本轮 user 消息里。
                    "goal": {
                        "type": "string",
                        "description": "ONLY when the user's message in THIS turn "
                                       "changes what the task is trying to achieve "
                                       "(and it is still the same task, so you are "
                                       "not switching ledgers): the rewritten Goal. "
                                       "Never revise the Goal on your own judgement "
                                       "mid-task. Requires goal_change_quote. "
                                       "After the Goal changes, re-read core and "
                                       "next: entries written for the old goal may "
                                       "now contradict it — remove those.",
                    },
                    "goal_change_quote": {
                        "type": "string",
                        "description": "the words from the user's message in THIS "
                                       "turn that ask for the change, copied "
                                       "exactly. Required with goal.",
                    },
                },
                # section/op 有意**不列 required**：只改 Goal 的调用两者都没有。
                # 校验在 `apply_tool_update` 里做（"要么给 section+op，要么给 goal"），
                # 报错文本直接说清楚——schema 表达不了这种二选一。
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bladex_ledger_switch",
            "description": "Switch this session to another ledger, or create one "
                           "for a new task. Use only when the user clearly changed "
                           "tasks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ledger_id": {"type": "string",
                                  "description": "target ledger id ('' = create new)"},
                    "title": {"type": "string",
                              "description": "title for a new ledger"},
                    # 🔴 Goal 由**你**提炼（Jason 2026-08-26 拍板，取代"原样引用用户
                    # 原话"）：用户 prompt 习惯各异，啰嗦与 agent 自己的机器包装都会
                    # 把后续轮次带偏——live 实证 jydesignhk 那 5 本账本的 Goal 全都以
                    # `Fully describe and explain everything about this image…` 开头。
                    # 提炼**只在建账本这一刻**做；之后不许自行改（见 update 的 goal 参数）。
                    "goal": {
                        "type": "string",
                        "description": "for a new ledger: the task goal in one or "
                                       "two short sentences, written so that "
                                       "'done' is checkable. State what must be "
                                       "true when the task is complete. Drop "
                                       "boilerplate and restate the user's intent "
                                       "— do not copy their message verbatim.",
                    },
                    # 🔴 段语义必须写进 description（MQ-L19，2026-08-27 拍板）。
                    # 原来只写 "initial Core entries" = 什么都没说，模型只能按段名
                    # **猜**。实证：同一个模型（deepseek-v4-flash / medium）在两本
                    # 账本上一对一错——Hermes 那本 core 填的是上下文常量（对），
                    # Pi 那本填成了待办清单（错）。**不是模型能力差异，是我们没说清楚。**
                    # 一个依赖模型猜的机制，对弱模型必然失效（Jason：不能强模型
                    # 跑得通、弱模型就不管）。
                    # 🔴 建账本这一刻 core 只能装"用户已经说了的"——模板写的
                    # "Pin only what you have VERIFIED in this session" 在第一轮
                    # 无法满足（什么都还没验证），却又提供了 core 参数邀请它填。
                    # 这个张力此前无人收口，模型只好拿待办去填。故在参数侧限定范围：
                    # 建本时 = 用户给的约束与偏好；验证出来的常量后续经 update 补。
                    "core": {
                        "type": "array", "items": {"type": "string"},
                        "description": "new ledger only. Constraints and preferences "
                                       "the USER has already stated, plus givens that "
                                       "hold for the WHOLE task — what a successor "
                                       "would need in order not to get it wrong. "
                                       "NOT a to-do list: steps go in 'next'. "
                                       "If an entry stops being true once the task is "
                                       "done, it does not belong here. Leave empty if "
                                       "the user gave no constraints; facts you "
                                       "establish later go in via bladex_ledger_update.",
                    },
                    "open": {
                        "type": "array", "items": {"type": "string"},
                        "description": "new ledger only. Questions you cannot answer "
                                       "yet and unresolved decisions — things that "
                                       "block or fork the work.",
                    },
                    "next": {
                        "type": "array", "items": {"type": "string"},
                        "description": "new ledger only. The concrete steps you are "
                                       "about to take, in order. These go stale as "
                                       "you work — that is expected; keep them "
                                       "current with bladex_ledger_update.",
                    },
                    # 🔴 parent 由**模型显式给**（Jason 2026-08-26 拍板，MQ-L1）：
                    # 此前 BladeX 把"当时激活的那本"无条件当父，把一串**平行**任务
                    # 串成假父子链（同一网站 5 个页面评审 → 四层链）。谁是子任务是
                    # LLM 的决策，BladeX 不替它判（三红线：不做作者）。
                    "parent_ledger_id": {
                        "type": "string",
                        "description": "ONLY when this new ledger is a sub-task of "
                                       "an existing one: that ledger's id. Omit for "
                                       "a new independent task, even if another "
                                       "ledger is currently active.",
                    },
                },
                "required": ["ledger_id"],
            },
        },
    },
]

TOOL_NAMES: tuple[str, ...] = tuple(t["function"]["name"] for t in TOOL_SCHEMAS)

#: 工具族（2026-08-30 Jason 拍板"按工具族拆"）。
#:
#: 砍掉 Memory Index 主动注入之后，工具面里装着两族取舍完全不同的东西：
#:   memory —— `bladex_memory_search` 是模型取记忆的**唯一**通道，
#:             对谁关掉就是让谁彻底没有记忆 ⇒ **始终注入**；
#:   ledger —— 账本读/改/切 + 首步指令，是"这件事需不需要账本"的策略，
#:             用户可配（`BLADEX_LEDGER_TIERS`，默认 auto 不变）。
#: 原来一刀切按档位注入，把这两族绑在一起——而 `BLADEX_LEDGER_TIERS` 的立卡
#: 理由（"用户决定哪些档位拿到**账本面**"）从头到尾说的就只是账本。
#:
#: 🔴 族由**名字前缀确定性派生**，不手工维护第二份清单（本仓两次栽在"闭集按
#: 记忆默写"）。新增 `bladex_ledger_*` 自动进账本族；落不进已知族的
#: **保守当账本族**（受门管），并由 `test_toolface` 的守卫直接判红——
#: 未知桶必须看得见，不能默默按记忆族放行。
FAMILY_MEMORY = "memory"
FAMILY_LEDGER = "ledger"


def tool_family(name: str) -> str:
    if name.startswith("bladex_memory_"):
        return FAMILY_MEMORY
    if name.startswith("bladex_ledger_"):
        return FAMILY_LEDGER
    return "unknown"


def ledger_tools_allowed(model_tier: str = "",
                         tiers: frozenset[str] | None = None) -> bool:
    """账本族（工具 + 首步指令）在这个档位上放不放行。

    🔴 **单一判据，三个消费者**：`inject_tools`（注不注账本工具）、
    `agency.augment_tools`（`ledger_face` 记什么）、`agency.insert_ledger_block`
    （给不给首步指令 + MQ-L28 gate）都调它，不各自写一遍
    `tier in resolve_ledger_tiers()`——两处各拼一份、改一处忘另一处，
    正是"有令无器"（MQ-L7）复发的路径。

    🔴 `tiers` 显式传入是为了**不吞掉两种生命周期的差异**：模块函数
    `inject_tools` 每次读 live env；`AgencyRuntime` 的档位集合是**启动冻结**的
    （与账本模板同语义："改配置重启生效"）。改动前两边就已经一个 live 一个
    frozen，只是生产上 env 不变所以从未发作——把来源做成参数，让调用方各自
    说明它用的是哪一份，比让谓词偷偷替它们选一个诚实。

    空 tier（老调用方 / 未路由）视为放行：拿不到档位不等于用户选择了不要。
    """
    allowed = resolve_ledger_tiers() if tiers is None else tiers
    return (not model_tier) or model_tier in allowed


def anthropic_tool_schemas() -> list[dict[str, Any]]:
    """OpenAI function 格式 → Anthropic 格式（name/description/input_schema）。"""
    return [{"name": t["function"]["name"],
             "description": t["function"]["description"],
             "input_schema": t["function"]["parameters"]}
            for t in TOOL_SCHEMAS]


# ── 注入（D3：追加末尾、不重排；会话首轮定格由调用方的稳定层机制保证）──────────

def inject_tools(
    existing_tools: list[dict] | None,
    *,
    agent_id: str,
    auxiliary: bool,
    model_tier: str = "",
    ledger_allowed: bool | None = None,
    fmt: str = "openai",
) -> tuple[list[dict], bool]:
    """把 bladex_* 工具定义追加进请求的 tools 数组。返回 (新列表, 是否注入)。

    不注的条件（任一满足）：aux 轮 / dsh 系 agent / 已注过（幂等）。

    🔴 **weak 档门已拆除**（2026-08-30 Jason 拍板，配合"Memory Index 不再主动
    注入"）：注入面砍到只剩硬规则之后，工具面成了模型取记忆的**唯一**通道，
    对 weak 档关着就等于让弱档 agent 彻底没有记忆。原来那道门的理由是
    `agent-compat-survey-20260825` 实测"weak 档工具调用不可靠"——不可靠是
    **调用质量**问题，而关掉它变成了**能力有无**问题，两者代价不对称。
    连带（正向）：MQ-L28 那个事故的病因原文是「弱模型读得到任务、**没有工具去
    更新/关闭/切换**」，拆门后第二半不再成立；但该 gate **本批保留观察**
    （Jason 同日拍板）——弱模型误调用是换了形态的同一风险，先用真实读数说话。

    🔴 `model_tier` 现在**只 gate 账本族**（见 `tool_family` / `ledger_tools_allowed`）。
    记忆族不受它管——那正是"按工具族拆"这个拍板的全部内容。
    （过程留痕：本次改动中我一度把这个参数整个删掉，理由是"工具面不再看档位"；
    那个前提被 `BLADEX_LEDGER_TIERS=none`（把账本面整个关掉）证伪——
    删掉它会让用户配的 `none` 静默失效，即刚性原则 9 的反面。）
    🔴 `existing_tools` 原顺序逐字保留——agent 自带工具的相对顺序动一位，
    上游 prompt cache 与 agent 侧 tools_hash 假设同时破。
    """
    base = (agent_id or "").split(":")[0]
    tools = list(existing_tools or [])
    if auxiliary or base in NO_TOOLFACE_AGENT_BASES:
        return tools, False
    ours = anthropic_tool_schemas() if fmt == "anthropic" else TOOL_SCHEMAS
    existing_names = set()
    for t in tools:
        fn = t.get("function") if isinstance(t, dict) else None
        existing_names.add((fn or t).get("name", "") if isinstance(fn or t, dict) else "")

    # 族过滤：记忆族无条件放行；账本族看档位；未知族**保守当账本族**。
    # 🔴 `ledger_allowed` 显式传入时**压过**档位判断：调用方（`AgencyRuntime`）
    # 知道一件本函数不可能知道的事——"本会话是不是正绑着一本激活账本"。
    # 那种情况下即使档位在集合外也要给账本工具，否则模型看得见账本却切不走
    # （MQ-L7 的补完，判据见 `agency.session_owns_active_ledger`）。
    # 不把 activation 传进这个纯函数：它是模块级的、无状态的，塞运行时对象
    # 会让它跟着 AgencyRuntime 的生命周期走。
    ledger_ok = (ledger_tools_allowed(model_tier) if ledger_allowed is None
                 else bool(ledger_allowed))

    def _keep(name: str) -> bool:
        return tool_family(name) == FAMILY_MEMORY or ledger_ok

    to_add = [t for t in ours
              if (nm := (t.get("function", t)).get("name")) not in existing_names
              and _keep(nm)]
    if not to_add:
        return tools, False
    return tools + to_add, True


# ── 执行分发（handler 注册表；真实 handler 接线随启用卡）──────────────────────

#: handler 契约：async (arguments: dict, *, allowed_exposure: str, context: dict)
#: -> str（给模型看的结果文本；**敏感条目必须已按 allowed_exposure 过滤**）。
ToolHandler = Callable[..., Awaitable[str]]


class ToolFace:
    """工具面执行器：只认 bladex_* 命名空间，未注册的 bladex 工具回结构化错误
    （与各 agent 的 unknown-tool 行为同形态——D2 实测五家都这么做，我们对模型
    也该这么做，而不是抛异常打断内循环）。"""

    def __init__(self) -> None:
        self._handlers: dict[str, ToolHandler] = {}

    def register(self, name: str, handler: ToolHandler) -> None:
        if not is_bladex_tool(name):
            # 红线 1 的机械执行点：非 bladex 命名空间的工具不许进注册表。
            raise ValueError(f"toolface only accepts {BLADEX_TOOL_PREFIX}* tools, got {name!r}")
        self._handlers[name] = handler

    @property
    def registered(self) -> tuple[str, ...]:
        return tuple(sorted(self._handlers))

    async def dispatch(self, name: str, arguments: str | dict,
                       *, allowed_exposure: str, context: dict | None = None) -> str:
        if not is_bladex_tool(name):
            raise ValueError(f"not a bladex tool: {name!r}")
        handler = self._handlers.get(name)
        if handler is None:
            logger.warning("toolface_unknown_tool", tool=name)
            return f"Error: unknown BladeX tool {name!r}. Available: {', '.join(self.registered)}"
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments or "{}")
            except json.JSONDecodeError as e:
                return f"Error: invalid JSON arguments for {name}: {e}"
        try:
            return await handler(arguments or {}, allowed_exposure=allowed_exposure,
                                 context=context or {})
        except Exception as e:  # noqa: BLE001 —— 单次工具失败不炸内循环
            logger.warning("toolface_handler_failed", tool=name, error=str(e))
            return f"Error: BladeX tool {name} failed: {e}"
