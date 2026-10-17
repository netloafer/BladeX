"""五段任务账本（Ledger）—— ADR-0032 §4；批一卡 V-L1。

账本 = agent 侧模型为一件任务维护的结构化工作状态（Goal/Core/Verified/Open/Next）。
**作者是 agent 侧的模型，BladeX 只保管/搬运/归还**（ADR-0032 三红线之 1）。

# 🔴 红线（不许"好心修正"）

1. **Goal 只有用户能改**（三红线之 2）：写路径按 actor 校验，
   `actor ∈ GOAL_WRITE_ACTORS` 之外一律拒绝并抛 `GoalWriteViolation`——
   模型能改 Goal 就自我拆台（ADR-0031 拍板继承）。
2. **文件是投影，不是存储**（ADR-0032 §2.3）：真相在 Memory Hub 的管理事件里，
   池内 MD 文件可由 `replay_ledger_events` 重放重建。用户直编 LEDGER 文件由维护
   进程捕获为**采纳**（2026-08-29 语义修订：用户版本即时生效于注入面，
   Hub 随会话收敛）后**再**重渲染——文件先行只是 UX，
   等价性锚在事件流上。
3. **渲染是纯函数**：同输入同输出，产物里不出现墙上时钟（`flash.py` 红线 2 同款）。
   时间一律用数据自带的 `created_at` / `updated_at`。
   **时区纪律（MQ-L46，2026-09-05）**：存储/比较层一律 UTC（`+00:00`，秒精度）；
   呈现层（MD / 注入块 / CLI）经 `local_iso` 转运行环境本地时区；比较一律走
   `iso_ms` 数值、**不比 ISO 字符串**——池里一旦混入不同偏移，字符串序就错，
   而候选面 top-5 正是靠这个序（比时间显示错 8 小时更贵）。`parse_ledger_md`
   把文件里任何偏移的串归一回 UTC，render→parse→render 逐字往返。
4. **段名是数据不是硬编码**（拍板 #8 连带义务）：`Ledger.section_order` 携带段结构，
   0.2.0 若并入 `lessons` 段 = 数据加一项，核心代码零改动。渲染/解析对未知段
   **原样保留**（forward-compat）。
5. **`ledger_id` 与标题/内容解耦**：内容无关的稳定 id（MQ-S39 标题任务化教训——
   标题会漂，身份不能跟着漂）。

设计约束：叶子模块，零 IO 依赖（落盘复用 `flash.write_if_changed`，路径函数在此、
写动作归维护进程）。被 consolidator / proxy / 脚本三方 import，日志用 stdlib logging。
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from .flash import LOCAL_SUFFIX, SCOPE_PERSONAL, safe_component, scope_dir

logger = logging.getLogger(__name__)

# ── 段结构（拍板 #8：五段先行，schema 留增段能力）─────────────────────────────

SECTION_GOAL = "goal"
SECTION_CORE = "core"
SECTION_VERIFIED = "verified"
SECTION_OPEN = "open"
SECTION_NEXT = "next"

#: 默认段序。**这是默认值不是闭集**——`Ledger.section_order` 才是每本账本的真实段结构。
DEFAULT_SECTION_ORDER: tuple[str, ...] = (
    SECTION_GOAL, SECTION_CORE, SECTION_VERIFIED, SECTION_OPEN, SECTION_NEXT,
)

#: 段标题（渲染用）。未知段的标题 = key 首字母大写（`_title_of`），
#: 所以加段不需要动这张表——表里只放想要**非默认标题**的段。
SECTION_TITLES: dict[str, str] = {
    SECTION_GOAL: "Goal",
    SECTION_CORE: "Core",
    SECTION_VERIFIED: "Verified",
    SECTION_OPEN: "Open",
    SECTION_NEXT: "Next",
}


def _title_of(key: str) -> str:
    return SECTION_TITLES.get(key, key.capitalize())


# ── 来源标记（ADR-0032 §4.1：每条带来源）───────────────────────────────────────

ENTRY_SOURCE_USER = "user"        # 用户说的 / 用户直编写入的
ENTRY_SOURCE_MODEL = "model"      # agent 侧模型经 bladex 工具写入的（账本的正常作者）
ENTRY_SOURCE_TOOL = "tool"        # 工具/测试结果佐证（Verified 的理想来源）
ENTRY_SOURCE_IMPORT = "import"    # BladeX 从历史补入（跨会话/跨 agent 归还）
ENTRY_SOURCES: tuple[str, ...] = (
    ENTRY_SOURCE_USER, ENTRY_SOURCE_MODEL, ENTRY_SOURCE_TOOL, ENTRY_SOURCE_IMPORT,
)

#: 手写行（无标记）默认算用户的——冒充作者只能往"归给用户"以外的方向错一定不行，
#: 而用户直编文件里新增的裸行确实就是用户写的（老渲染器 `slot_source_of` 曾用的同一保守原则，
#: 只是此处保守方向相反：账本文件的裸行来源恰是用户直编这一条路）。
DEFAULT_ENTRY_SOURCE = ENTRY_SOURCE_USER

# ── 写路径 actor（红线 1）────────────────────────────────────────────────────

ACTOR_USER = "user"
ACTOR_DASHBOARD = "dashboard"
ACTOR_MODEL = "model"
ACTOR_MAINTENANCE = "maintenance"

#: Goal 段的合法写者。**模型与维护进程都不在内**——维护进程也不行：
#: "自检时顺手修个 Goal"与模型改 Goal 是同一种越权，只是换了执行者。
GOAL_WRITE_ACTORS: tuple[str, ...] = (ACTOR_USER, ACTOR_DASHBOARD)


class LedgerError(ValueError):
    """账本域错误基类。"""


class GoalWriteViolation(LedgerError):
    """非用户 actor 试图写 Goal。调用方必须落可 grep 日志（`ledger_goal_write_denied`）。"""


# ── Domain 对象 ─────────────────────────────────────────────────────────────


class LedgerEntry(BaseModel):
    """段内一条。`source ∈ ENTRY_SOURCES`；`ref` = 证据指向（工具结果 / turn key / 测试名）。"""

    text: str
    source: str = DEFAULT_ENTRY_SOURCE
    ref: str = ""
    #: 写下这一条的 agent 完整 id（`claude-code` / `hermes:default` / …）。
    #: 🔴 **缺省空 = 历史条目或用户直编**，不是"未知的某个 agent"——
    #: 空与非空是两种语义，不许用占位符填平（分母纪律：缺数不当读数）。
    #:
    #: 为什么在**条目**这一级而不是账本级：账本级已经有 `last_writer`，它回答
    #: "谁最后动过这本"；轴 C（跨 agent 交接）要问的是"**这一条**是谁写的"——
    #: 一本账本上 CC 写的 Verified 与 Hermes 写的 Next 并存正是交接的常态形状，
    #: 账本级那一个字段把它们压成了同一个答案。
    writer: str = ""


class Ledger(BaseModel):
    """一本账本。`ledger_id` 是身份（内容无关，红线 5）；`matter_id` 是外部边界锚点
    （ADR-0032 §4.5，绑定动作属 V-L5，本 schema 只留位）。"""

    ledger_id: str
    title: str = ""
    status: str = "active"                 # active | closed
    matter_id: str = ""
    #: 父账本（V-L6e，Jason 2026-08-25 拍板"子任务可以建账本，但要与父账本信息同步，
    #: 否则两个账本失联"）。🔴 **只存这一个方向**：父侧的子列表由池派生
    #: （`children_of`）——两向都存就要双写同步，那是"事出多头"的经典形态。
    parent_ledger_id: str = ""
    created_at: str = ""                   # ISO，数据自带（红线 3）
    updated_at: str = ""
    #: 乐观并发版本号（2026-08-29 Jason 拍板「两 agent 同激活一本」批）：每次写
    #: 成功 +1；模型在 bladex_ledger_update 带它读到的 rev，落后即拒写并返回最新
    #: 正文——合并者是 agent 侧模型（作者边界），BladeX 只保证**不静默覆盖**。
    rev: int = 0
    #: 最近写者（agent 完整 id / "user"）。注入面并发标注的判据，不参与身份。
    last_writer: str = ""
    #: 被哪本账本取代（批 Z · Z3，README 承诺的 `supersede` 原语）。只存这一个方向——
    #: 与 `parent_ledger_id` 同理，反向（"它取代了谁"）由池派生，两向都存就要双写同步。
    #: 只由 `ledger_supersede` 事件写（`close_ledger`），`reopen_ledger` 清空。
    superseded_by: str = ""
    #: 批 E · E4（MQ-L98，2026-09-29）：**谁在何时关的**——加法字段，老事件缺省为空。
    #: 只由 `close_ledger` 写、`reopen_ledger` 清，重放与运行时同一变换。
    #: 🔴 批 HO（MQ-L99，2026-10-15）改了它的消费方：E4 时「已关账本只对关它的那个会话
    #: 列出」（`closed_by_session == session_id`），交接（换会话、换 agent）必然不满足；
    #: 现在已关账本按**内容**进相关性段（任何会话，`ledger_runtime.relevant_ledgers`），
    #: `closed_at` 定时间窗，`closed_by_agent` 只用于渲染「谁关的」，`closed_by_session`
    #: 不再参与候选判定（仍随事件重放，审计用）。
    closed_by_session: str = ""
    closed_by_agent: str = ""              # 批 HO：关闭事件载荷的写者 agent（CLI / admin 为空）
    closed_at: str = ""                    # ISO（关闭事件的时刻）
    # ── Goal：单块文本 ──
    #
    # 🔴 2026-08-26 拍板（Jason）：从「取」改为「**由 agent 侧模型提炼**」。
    # 原立论"原样引用、用户可核对"证伪于 live——jydesignhk 那 5 本账本的 Goal
    # 全部以 agent 自己的图片分析包装开头（`Fully describe and explain everything
    # about this image, then answer the following question:`），用户真正说的话被压在
    # 90 个字符之后，5 个 Goal 的前缀逐字相同、区分度全在尾部。**原文不等于用户意图**：
    # 用户 prompt 习惯各异，啰嗦与机器包装都会把模型带偏。
    #
    # 红线**没有松动**，只是把「谁执笔」与「谁能改」拆开了——红线防的一直是
    # "模型中途把做不到的事改成做得到的事然后宣告完成"，那是**中途篡改**。
    # 开局执笔一次不构成这个威胁；中途改动仍须锚定到用户原话（`revise_goal`）。
    goal: str = ""
    goal_source: str = ""                  # 空 = 未取到；user=原话 / model=提炼 / model_revised=经用户请求改过
    goal_origin: str = ""                  # 原文出处（turn/ledger key），审计可回查
    #: 触发建账本的**用户原话**。🔴 **只存不注**——原话本来就在消息主体里，
    #: 再注一遍是白花 token（Jason 2026-08-26）。它的用途是**我们本地检索与分析**
    #: （提炼跑偏时能对照原文），以及 dashboard 上给用户核对。
    #: `render_ledger_md` 有意不渲染它；守卫见 `test_goal_verbatim_never_rendered`。
    goal_verbatim: str = ""
    #: Goal 被改过几次。0 或 1 是正常任务的形态；数字大 = 值得人看一眼。
    #: 🔴 存在的理由是**可发现性**：`revise_goal` 挡不住"用户说了'继续'、模型硬引用
    #: '继续'来改目标"这种形态，真正的保护是它改了就留痕、留痕就能数。
    #: （MQ-L17 的教训：没人看得见的机制等于没有。）
    goal_revisions: int = 0
    # ── 其余段：条目列表，键 = 段名 ──
    sections: dict[str, list[LedgerEntry]] = Field(default_factory=dict)
    #: 段结构（红线 4）。含 "goal"——它也是段，只是渲染/写保护特殊。
    section_order: list[str] = Field(default_factory=lambda: list(DEFAULT_SECTION_ORDER))

    def entries(self, section: str) -> list[LedgerEntry]:
        return self.sections.get(section, [])


# ── 时间：存 UTC / 比数值 / 显本地（红线 3 时区纪律）────────────────────────


def _parse_iso(ts: str) -> datetime | None:
    s = (ts or "").strip()
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)   # naive 串按 UTC 解（与 iso_ms 同判）
    return dt


def iso_ms(ts: str) -> float:
    """ISO 时间串 → epoch 毫秒；空 / 不合形返回 0.0（**缺数不是 0 时刻**）。

    调用方必须把 0.0 当"取不出"处理、不得当"1970 年"参与大小比较——
    否则任何缺 `created_at` 的账本都会被判成"比一切都早"。
    任何偏移（`Z` / `+00:00` / `+08:00`）都按真实时刻折算——这是排序键，
    不许退回比字符串。
    """
    dt = _parse_iso(ts)
    return dt.timestamp() * 1000.0 if dt is not None else 0.0


def normalize_iso_utc(ts: str) -> str:
    """任意偏移的 ISO 串 → UTC `+00:00` 秒精度（存储/比较层的唯一形态）。
    空串 / 不合形**原样返回**（不伪造时间，让上层的 0.0 语义接手）。"""
    dt = _parse_iso(ts)
    return dt.astimezone(UTC).isoformat(timespec="seconds") if dt is not None else ts


def local_iso(ts: str) -> str:
    """UTC 串 → 运行环境本地时区、带偏移（如 `2026-09-05T12:58:56+08:00`）。
    呈现层专用：MD `created:`/`updated:` 行、注入块、CLI 时间列。
    空串 / 不合形原样返回。时区取 `datetime.astimezone()` 的系统本地（随 `TZ`）。"""
    dt = _parse_iso(ts)
    return dt.astimezone().isoformat(timespec="seconds") if dt is not None else ts


def new_ledger_id() -> str:
    """`ldg-<hex12>`。与标题/内容/时间无关（红线 5）。"""
    return f"ldg-{uuid.uuid4().hex[:12]}"


def children_of(pool: dict[str, Ledger], parent_id: str) -> list[Ledger]:
    """父账本的子账本（从池派生，单一方向存储的读侧；按 updated_at 降序）。"""
    kids = [led for led in pool.values() if led.parent_ledger_id == parent_id and parent_id]
    kids.sort(key=lambda led: iso_ms(led.updated_at or led.created_at), reverse=True)
    return kids


def new_ledger(*, ledger_id: str = "", title: str = "", goal: str = "",
               goal_source: str = "", goal_origin: str = "",
               goal_verbatim: str = "",
               created_at: str = "", parent_ledger_id: str = "",
               section_order: list[str] | None = None) -> Ledger:
    """建一本空账本（五段就位；`section_order` 可由模板覆盖——增段数据面）。

    `goal` 只在创建时允许由任意来源带入（模型提炼即走这里）——创建那一刻
    还没有"中途改目标"可言。创建后的修改走两道门：用户/dashboard 走
    `update_goal`，模型走 `revise_goal`（必须锚定到本轮用户原话）。
    `goal_verbatim` 是触发创建的用户原话，**只存不注**。"""
    order = list(section_order or DEFAULT_SECTION_ORDER)
    if SECTION_GOAL not in order:
        order.insert(0, SECTION_GOAL)      # goal 段不许被模板删掉（红线 1 的载体）
    return Ledger(
        ledger_id=ledger_id or new_ledger_id(), title=title,
        goal=goal, goal_source=goal_source, goal_origin=goal_origin,
        goal_verbatim=goal_verbatim,
        parent_ledger_id=parent_ledger_id,
        created_at=created_at, updated_at=created_at,
        sections={k: [] for k in order if k != SECTION_GOAL},
        section_order=order,
    )


# ── 写路径（红线 1 的执行点）─────────────────────────────────────────────────


def update_goal(ledger: Ledger, text: str, *, actor: str, updated_at: str = "") -> Ledger:
    """改 Goal。**唯一**允许的写者是用户（含 dashboard 代理的用户操作）。"""
    if actor not in GOAL_WRITE_ACTORS:
        # 可 grep：静默拒绝 = 模型以为写成功了，比拒绝本身更糟。
        logger.warning("ledger_goal_write_denied ledger=%s actor=%s",
                       ledger.ledger_id, actor)
        raise GoalWriteViolation(
            f"goal is user-owned; actor {actor!r} may not write it "
            f"(allowed: {', '.join(GOAL_WRITE_ACTORS)})")
    return ledger.model_copy(update={
        "goal": text, "goal_source": ACTOR_USER,
        "goal_revisions": ledger.goal_revisions + 1,
        "updated_at": updated_at or ledger.updated_at,
    })


#: `revise_goal` 引用核对时忽略的字符（空白与常见标点——模型复述时几乎必然
#: 改动这些，卡死在标点上只会逼它去猜格式，不会提高安全性）。
_QUOTE_NOISE = str.maketrans("", "", " \t\r\n　，,。.：:；;、!！?？\"'“”‘’()（）")


def _quote_key(s: str) -> str:
    return (s or "").translate(_QUOTE_NOISE).lower()


def revise_goal(ledger: Ledger, text: str, *, quote: str, user_text: str,
                updated_at: str = "") -> Ledger:
    """模型改 Goal——**只有在能锚定到本轮用户原话时**才允许（Jason 2026-08-26 拍板）。

    场景：用户又发了一条 prompt，模型判断这不是新任务、而是**调整了原目标**。
    这时该改 Goal 而不是新开账本。没有用户请求，模型不得在任务过程中改 Goal。

    🔴 **这道门锚定责任，并不能证明"用户真的要求了"**——我们无法核实模型的断言。
    它做到的是：`quote` 必须**确实出现在本轮 user 消息里**，凭空编造的依据当场被打回。
    用户说"继续"、模型硬引用"继续"来改目标，这种形态**过得去**。
    真正的保护是第二层：每次改都 `goal_revisions += 1` 并留管理事件，
    **改了就能被数出来**（正常任务 0 或 1 次）。诚实记在这里，别把它当成防住了。

    `goal_source` 记为 `model_revised`，与创建时的 `model`（提炼）区分开——
    "开局执笔"与"中途改动"是两件安全性质不同的事，读数上必须分得开。
    """
    text = (text or "").strip()
    if not text:
        raise GoalWriteViolation("goal revision requires non-empty text")
    if not _quote_key(quote):
        logger.warning("ledger_goal_revise_denied ledger=%s reason=no_quote",
                       ledger.ledger_id)
        raise GoalWriteViolation(
            "changing the Goal requires goal_change_quote: the words from the "
            "user's message in THIS turn that ask for the change")
    if _quote_key(quote) not in _quote_key(user_text):
        logger.warning("ledger_goal_revise_denied ledger=%s reason=quote_not_found "
                       "quote=%r", ledger.ledger_id, quote[:60])
        raise GoalWriteViolation(
            "goal_change_quote was not found in the user's message this turn; "
            "the Goal may only change when the user asks for it")
    return ledger.model_copy(update={
        "goal": text, "goal_source": "model_revised",
        "goal_revisions": ledger.goal_revisions + 1,
        "updated_at": updated_at or ledger.updated_at,
    })


def add_entry(ledger: Ledger, section: str, entry: LedgerEntry, *,
              actor: str, updated_at: str = "") -> Ledger:
    """向段追加一条。Goal 不是条目段（走 `update_goal`）；未知段允许——
    但必须已在 `section_order` 里（先增段再写，防手滑拼错段名静默建段）。"""
    if section == SECTION_GOAL:
        raise LedgerError("goal is a text block, not an entry section; use update_goal")
    if section not in ledger.section_order:
        raise LedgerError(f"unknown section {section!r}; extend section_order first")
    sections = {k: list(v) for k, v in ledger.sections.items()}
    sections.setdefault(section, []).append(entry)
    return ledger.model_copy(update={
        "sections": sections, "updated_at": updated_at or ledger.updated_at,
    })


def remove_entry(ledger: Ledger, section: str, index: int, *,
                 updated_at: str = "") -> tuple[Ledger, LedgerEntry]:
    """按位删一条（Next 完成项出列的原语）。返回 (新账本, 被删条目)。"""
    items = list(ledger.entries(section))
    if not 0 <= index < len(items):
        raise LedgerError(f"no entry #{index} in section {section!r}")
    removed = items.pop(index)
    sections = {k: list(v) for k, v in ledger.sections.items()}
    sections[section] = items
    return (ledger.model_copy(update={
        "sections": sections, "updated_at": updated_at or ledger.updated_at,
    }), removed)


def add_section(ledger: Ledger, key: str, *, position: int | None = None) -> Ledger:
    """增段（红线 4 的数据面）：0.2.0 加 `lessons` = 调这一下，核心零改动。"""
    k = key.strip().lower()
    if not k or not re.fullmatch(r"[a-z][a-z0-9_-]*", k):
        raise LedgerError(f"bad section key {key!r}")
    if k in ledger.section_order:
        return ledger
    order = list(ledger.section_order)
    order.insert(len(order) if position is None else position, k)
    sections = {**ledger.sections, k: list(ledger.sections.get(k, []))}
    return ledger.model_copy(update={"section_order": order, "sections": sections})


# ── 生命周期：close / supersede / reopen（批 Z · Z3，MQ-L39 / MQ-L86）──────────
#
# 🔴 三个原语**只动 `status` / `superseded_by`**，一字不碰 Goal 与各段条目——
# 关本是账本对象操作（三红线之 1 之内），不是内容写入，更不是改目标（红线 1：
# Goal 仅用户可改）。守卫 `test_ledger_lifecycle.py::test_close_never_touches_goal`。

STATUS_ACTIVE = "active"
STATUS_CLOSED = "closed"

#: 生命周期事件的 `origin` 闭集（原则 13 层 4：值要能指认来源）。
#: `model` 只由工具 handler 写；admin 端点只收 `user` / `cli`（不许冒充模型）。
LIFECYCLE_ORIGIN_MODEL = "model"
LIFECYCLE_ORIGIN_USER = "user"
LIFECYCLE_ORIGIN_CLI = "cli"
LIFECYCLE_ORIGINS: tuple[str, ...] = (
    LIFECYCLE_ORIGIN_MODEL, LIFECYCLE_ORIGIN_USER, LIFECYCLE_ORIGIN_CLI,
)


def ledger_is_done(ledger: Ledger | dict | None) -> bool:
    """账本**自述完结**：Open 与 Next 都空、Verified 非空（批 Z · Z3.2 / Y1b 的机判口径）。

    与 Y0 探针 `done_at` 同一判据（`20260923-pre-t1` 的 10 条机判就是按它分的）。
    段不存在按空算（模板可删段）；**没有 Verified 就不算做完**——空账本不是完结的账本。
    接受 `Ledger` 或其 dump（事件载荷），两种形态一处判。
    """
    if ledger is None:
        return False
    sec = ledger.get("sections") if isinstance(ledger, dict) else ledger.sections
    sec = sec or {}
    return (not sec.get(SECTION_OPEN) and not sec.get(SECTION_NEXT)
            and bool(sec.get(SECTION_VERIFIED)))


def close_ledger(ledger: Ledger, *, superseded_by: str = "",
                 closed_by_session: str = "", closed_at: str = "",
                 closed_by_agent: str = "") -> Ledger:
    """关本（`superseded_by` 非空 = supersede）。重放与运行时共用这一处变换。

    `closed_by_session` / `closed_by_agent` / `closed_at`（E4 MQ-L98 / 批 HO MQ-L99）：
    关闭事件载荷里的写者会话、写者 agent 与事件时刻；admin / CLI 关的本没有会话与 agent
    ⇒ 留空。是否回到模型候选**不看谁关的**（MQ-L99：按内容匹配，任何会话）。
    """
    return ledger.model_copy(update={
        "status": STATUS_CLOSED,
        "superseded_by": superseded_by or ledger.superseded_by,
        "closed_by_session": closed_by_session,
        "closed_by_agent": closed_by_agent,
        "closed_at": closed_at,
    })


def reopen_ledger(ledger: Ledger) -> Ledger:
    """重开：只恢复状态（**不激活**——激活只有 switch 一条路）。

    取代指针一并清空：重开意味着它又是一件在办的事，留着 "superseded by X"
    会让读者以为它已被取代。取代的历史仍在 Hub 事件流里（`ledger_supersede`）。
    """
    return ledger.model_copy(update={"status": STATUS_ACTIVE, "superseded_by": "",
                                     "closed_by_session": "", "closed_by_agent": "",
                                     "closed_at": ""})


# ── 渲染（纯函数）────────────────────────────────────────────────────────────

_MACHINE_COMMENT = ("<!-- BladeX Ledger · authored by the agent, kept by BladeX. "
                    "You may edit this file; BladeX records your edits. -->")
_EMPTY_MARK = "_(empty)_"
#: Goal 段禁改提示（渲染进文件，用户可见；英文 = ADR-0027 §5.1 外壳纪律）。
_GOAL_NOTE = "<sub>(the Goal section is yours — agents cannot change it)</sub>"


def _render_entry(e: LedgerEntry) -> str:
    """`- 正文  <sub>[source · ref · @writer]</sub>`。

    `@` 前缀让写者在词面上与 `source`/`ref` 分得开，也让 `_ENTRY_RE` 不必靠位置
    猜（老行只有两段，新行三段，中间那段可有可无）。writer 空则整段不渲染——
    历史条目与用户直编的裸行逐字保持原样（往返等价的前提）。
    """
    tail = f" · {e.ref}" if e.ref else ""
    who = f" · @{e.writer}" if e.writer else ""
    return f"- {e.text}  <sub>[{e.source}{tail}{who}]</sub>"


def render_ledger_md(ledger: Ledger) -> str:
    """账本 → MD（用户读的那份，也是池内落盘形态）。逐字节确定（红线 3）。"""
    lines = [f"# {ledger.title or ledger.ledger_id}", "", _MACHINE_COMMENT, "",
             f"- ledger: `{ledger.ledger_id}`",
             f"- status: {ledger.status}",
             # rev 必须渲染——模型要在 update 里回带它，读不到就用不了乐观锁
             f"- rev: {ledger.rev}"]
    if ledger.matter_id:
        lines.append(f"- matter: `{ledger.matter_id}`")
    if ledger.parent_ledger_id:
        # V-L6e：子账本永远指得回父账本（失联是 Jason 点名要避免的形态）
        lines.append(f"- parent ledger: `{ledger.parent_ledger_id}`")
    if ledger.superseded_by:
        # Z3：取代链的读侧——读到一本被取代的账本，下一跳就在这一行
        lines.append(f"- superseded by: `{ledger.superseded_by}`")
    # 呈现层：本地时区（用户在 CST 读到的 `updated` 曾比墙钟早 8 小时，MQ-L46）
    if ledger.created_at:
        lines.append(f"- created: {local_iso(ledger.created_at)}")
    if ledger.updated_at:
        lines.append(f"- updated: {local_iso(ledger.updated_at)}")
    for key in ledger.section_order:
        lines += ["", f"## {_title_of(key)}", ""]
        if key == SECTION_GOAL:
            if ledger.goal:
                lines.append(ledger.goal)
                meta = []
                if ledger.goal_source:
                    meta.append(f"source: `{ledger.goal_source}`")
                if ledger.goal_origin:
                    meta.append(f"origin: `{ledger.goal_origin}`")
                if meta:
                    lines += ["", f"<sub>({' · '.join(meta)})</sub>"]
            else:
                lines.append(_EMPTY_MARK)
            lines += ["", _GOAL_NOTE]
            continue
        items = ledger.entries(key)
        lines += [_render_entry(e) for e in items] if items else [_EMPTY_MARK]
    return "\n".join(lines) + "\n"


def render_template(*, title: str = "New task") -> str:
    """预置模板（`LEDGER.md`，ADR-0032 §4.1）：空五段 + 引导语，供用户手起一本。"""
    tpl = new_ledger(ledger_id="ldg-template", title=title)
    tpl = tpl.model_copy(update={
        "goal": "(Write the end goal here: what done looks like, hard constraints, "
                "acceptance criteria. One paragraph, verifiable.)",
        "goal_source": ACTOR_USER,
    })
    return render_ledger_md(tpl)


#: 🔴 **模板只有一个来源：`config/ledger-template.md`**（Jason 2026-08-25 拍板
#: "不能事出多头改两个地方"）。本模块**不再内置模板正文**——只保留一个
#: 结构性兜底：文件缺失/不可读时用它，且内容刻意只有段骨架 + 一句指路，
#: 让"模板文件没了"这件事在注入面上可见，而不是被一份漂亮的副本悄悄掩盖。
FALLBACK_TEMPLATE_MD = """\
# Task Ledger Template (fallback)

<!-- config/ledger-template.md is missing or unreadable; using the built-in
     structural fallback. Restore that file to customize sections. -->

## Goal

## Core

## Verified

## Open

## Next
"""


def template_section_order(template_md: str) -> list[str]:
    """从模板 md 解析段结构（用户在模板里加一段 = 新账本自动带上，增段的数据面）。
    解析不出任何段时回退默认五段——坏模板不该让建本失败。"""
    parsed = parse_ledger_md(template_md)
    order = [s for s in parsed.section_order if re.fullmatch(r"[a-z][a-z0-9_-]*", s)]
    return order if order else list(DEFAULT_SECTION_ORDER)


# ── 解析（MD → Ledger，与渲染往返等价；容忍手编）────────────────────────────

_META_RE = re.compile(
    r"^- (ledger|status|matter|parent ledger|superseded by|created|updated): `?([^`]*)`?\s*$")
#: 条目行。四个捕获组：正文 / source / ref / writer。
#: 🔴 **向后兼容是硬要求**：池里存量全是两段形态（`[model · ref]` 或 `[user]`），
#: ref 段与 writer 段都可缺席，且 writer 段永远带 `@` 前缀——ref 组的 `(?!@)`
#: 前瞻就是为此：没有它，`[model · @cc]` 会把 `@cc` 读成 ref，writer 恒空
#: （一个"看起来能解析"的静默错，往返等价还会照样成立）。
_ENTRY_RE = re.compile(
    r"^- (.*?)"
    r"(?:\s+<sub>\[([a-z]+)(?: · (?!@)(.*?))?(?: · @([^\]]*))?\]</sub>)?\s*$")
_GOAL_META_RE = re.compile(r"^<sub>\((?:source: `([^`]*)`)?(?: · )?(?:origin: `([^`]*)`)?\)</sub>$")


def parse_ledger_md(text: str, *, base: Ledger | None = None) -> Ledger:
    """解析池内/用户直编的账本文件。三条容忍：裸条目行来源按 `user`；
    未知 `## 段` 原样收进 `section_order`（红线 4）；空段标记与空行忽略。

    🔴 **MD 不携带的字段必须从 `base` 带回来**（`goal_verbatim` / `goal_revisions`）。
    `goal_verbatim` 是有意不渲染的——`render_ledger_md` 的产物就是注入块，
    渲染它等于把用户原话又注一遍（Jason 2026-08-26：原话在消息主体里已经有了）。
    代价是 MD 往返会丢它；不给 `base` 的调用方**会静默把它清空**，
    而用户直编捕获正是"拿解析结果当完整状态发事件"的形态。
    """
    title = ""
    meta: dict[str, str] = {}
    order: list[str] = []
    sections: dict[str, list[LedgerEntry]] = {}
    goal_lines: list[str] = []
    goal_source = goal_origin = ""
    current: str | None = None

    for raw in text.splitlines():
        line = raw.rstrip()
        if line.startswith("# ") and not title and current is None:
            title = line[2:].strip()
            continue
        if line.startswith("<!--") or line == _GOAL_NOTE:
            continue
        if line.startswith("## "):
            key = line[3:].strip().lower()
            current = key
            if key not in order:
                order.append(key)
                if key != SECTION_GOAL:
                    sections.setdefault(key, [])
            continue
        if current is None:
            m = _META_RE.match(line)
            if m:
                meta[m.group(1)] = m.group(2).strip()
            continue
        if not line or line == _EMPTY_MARK:
            continue
        if current == SECTION_GOAL:
            gm = _GOAL_META_RE.match(line)
            if gm:
                goal_source = gm.group(1) or ""
                goal_origin = gm.group(2) or ""
            else:
                goal_lines.append(line)
            continue
        em = _ENTRY_RE.match(line)
        if em and line.startswith("- "):
            src = em.group(2) or DEFAULT_ENTRY_SOURCE
            if src not in ENTRY_SOURCES:
                src = DEFAULT_ENTRY_SOURCE
            sections.setdefault(current, []).append(
                LedgerEntry(text=em.group(1), source=src, ref=em.group(3) or "",
                            writer=em.group(4) or ""))

    ledger_id = meta.get("ledger", "")
    goal_text = "\n".join(goal_lines).strip()
    ledger = Ledger(
        ledger_id=ledger_id or new_ledger_id(),
        title="" if title == ledger_id else title,
        status=meta.get("status", "active"),
        matter_id=meta.get("matter", ""),
        parent_ledger_id=meta.get("parent ledger", ""),
        superseded_by=meta.get("superseded by", ""),
        # 文件里是本地偏移串（render 产物）或用户手写的任意偏移 ⇒ 归一回 UTC，
        # 池里不许混偏移（红线 3 时区纪律）
        created_at=normalize_iso_utc(meta.get("created", "")),
        updated_at=normalize_iso_utc(meta.get("updated", "")),
        goal=goal_text, goal_source=goal_source, goal_origin=goal_origin,
        # MD 不携带的两个字段：有 base 就带回来，没有就是空（见 docstring 红线）
        goal_verbatim=base.goal_verbatim if base is not None else "",
        goal_revisions=base.goal_revisions if base is not None else 0,
        sections=sections,
        section_order=order or list(DEFAULT_SECTION_ORDER),
    )
    return ledger


# ── 池化路径（账本投影落点；真相在 Hub 事件流，ADR-0034 §3.2）──────────────────

LEDGERS_DIR = "ledgers"


def ledgers_dir(root: str, principal: str, *, scope: str = SCOPE_PERSONAL) -> str:
    """`<flash-root>/<principal>/<scope>/ledgers/`——账本池。scope 维度与 flash 同构
    （敏感度/多租户纪律免费继承）；个人模式恒 personal。"""
    return f"{scope_dir(root, principal, scope)}/{LEDGERS_DIR}"


def ledger_shard(ledger_id: str) -> str:
    """两级哈希分桶路径段，固定 `66/d0`（Jason 2026-08-28 拍板：一套方案，
    不设深度旋钮——个人场景目录惰性创建零成本，企业规模 65536 桶直接受益；
    每任务一本、经年累月上万文件，单层目录会拖死文件系统遍历）。

    id 形如 `ldg-<hex12>`（`new_ledger_id`，内容无关稳定 id）——第一级取 hex
    第 1–2 位、第二级取第 3–4 位，**从 id 自身确定性派生**，不引入第二个哈希
    （同一 id 永远同一桶，重建等价免费）。非标准形态（模板/手造 id）只留
    字母数字、短则补 0。
    """
    body = ledger_id[4:] if ledger_id.startswith("ldg-") else ledger_id
    comp = ("".join(c for c in body.lower() if c.isalnum()) + "0000")[:4]
    return f"{comp[0:2]}/{comp[2:4]}"


def ledger_md_path(root: str, principal: str, ledger_id: str,
                   *, scope: str = SCOPE_PERSONAL) -> str:
    """账本投影文件：`ledgers/<桶>/<id>.md`（两级分桶见 `ledger_shard`）。
    也是用户直编的落点（ADR-0034 §3.1 ①，Flash 唯一不可替代的职责）。
    🔴 签名里没有 agent/project/session——账本身份独立于访问路径。"""
    return (f"{ledgers_dir(root, principal, scope=scope)}/"
            f"{ledger_shard(ledger_id)}/{safe_component(ledger_id)}.md")


def ledger_local_path(root: str, principal: str, ledger_id: str,
                      *, scope: str = SCOPE_PERSONAL) -> str:
    """用户覆盖文件（`flash` 同款语义：永不被机器覆盖）。"""
    base = ledger_md_path(root, principal, ledger_id, scope=scope)
    return base[: -len(".md")] + LOCAL_SUFFIX


# ── 管理事件重放（红线 2：池 = Hub 事件的投影）───────────────────────────────

#: 事件类型串——与 `bladex_proxy.models.AdminEventType` 新增值逐字一致
#: （core 不 import proxy；一致性由 proxy 侧测试对账）。
#: 🔴 载荷纪律（MQ-A7）：`AdminEvent.matter_id` **留空**，一切走 payload。
EVENT_LEDGER_CREATE = "ledger_create"
EVENT_LEDGER_UPDATE = "ledger_update"        # 状态式：payload = 完整 dump（含模型经工具的改）
EVENT_LEDGER_SWITCH = "ledger_switch"        # payload = {session_id, agent_id, from_ledger_id, to_ledger_id}
#: 惰性作用域迁移（2026-08-29，MQ-L30 收尾拍板）：项目键 miss ∧ Global 有
#: 绑定 ∧ 本请求有项目信号 ⇒ 搬。payload = {agent_id, from_project,
#: to_project, ledger_id}。自愈实测证伪（自发 0/2）后立的机制解。
EVENT_LEDGER_SCOPE_MIGRATE = "ledger_scope_migrate"
EVENT_LEDGER_USER_EDIT = "ledger_user_edit"  # payload = {ledger: dump}——直编捕获后的完整状态
#: 生命周期（批 Z · Z3）。**增量式**载荷（与 create/update 的整本 dump 不同）：
#: 关 / 重开只动 status，一本整本 dump 会把"谁改了正文"与"谁关了本"混成一件事。
#: payload = {ledger_id, origin ∈ LIFECYCLE_ORIGINS, reason, agent_id, project_id,
#:            session_id, turn_index}（后四个只有 model 来源有）；
#: supersede 另带 `superseded_by`。
#: 🔴 close / supersede 在时间轴上 = 该本**当时激活着的每个 scope** 切到空
#: （`switch_timeline`）；在绑定上 = 这些 scope 解除（`replay_ledger_events`）。
#: reopen 只动池里的 status，**不碰时间轴与绑定**（重开 ≠ 激活）。
EVENT_LEDGER_CLOSE = "ledger_close"
EVENT_LEDGER_REOPEN = "ledger_reopen"
EVENT_LEDGER_SUPERSEDE = "ledger_supersede"
LEDGER_LIFECYCLE_EVENTS = frozenset({
    EVENT_LEDGER_CLOSE, EVENT_LEDGER_REOPEN, EVENT_LEDGER_SUPERSEDE,
})
#: 释放激活的那两类（close 与 supersede 在时间轴 / 绑定上同形）。
_RELEASING_EVENTS = frozenset({EVENT_LEDGER_CLOSE, EVENT_LEDGER_SUPERSEDE})

#: 账本事件全集（§3.2b 单一判据：agency 的池装载与 flash 维护进程的 Hub 源
#: 都按这一份过滤——两边各抄一份迟早分叉）。
LEDGER_EVENT_TYPES = frozenset({
    EVENT_LEDGER_CREATE, EVENT_LEDGER_UPDATE,
    EVENT_LEDGER_SWITCH, EVENT_LEDGER_USER_EDIT,
    EVENT_LEDGER_SCOPE_MIGRATE,
    *LEDGER_LIFECYCLE_EVENTS,
})


def apply_ledger_tombstones(
    events: list[dict], tombstoned: set[str], *,
    tombstone_ts_ms: dict[str, float] | None = None,
) -> list[dict]:
    """滤掉被墓碑账本的事件（纯函数）。返回新列表，不改入参。

    墓碑语义（ADR-0012 §3.6「重建不得复活已删数据」）在账本上的落地。
    三个消费方（`agency` 池装载 / 重建锚层 / flash 物化）都经
    `replay_ledger_events`+`switch_timeline`，**在这里滤一次，三处自动跟随**。

    ## 🔴 为什么 switch 事件不能直接丢弃

    时间轴是「点时查找」结构。把"切到被墓碑账本"整条丢掉，**前一本账本的激活
    窗口会向后延伸**，把本属于被墓碑账本的那段轮次吃进去——换了个受害者，
    问题没解决（这正是 MQ-L15「按终态给历史轮次归属 = 机制制造的误合并」的
    同一形状，方向相反）。

    故改写为 **`to_ledger_id=""` = 切到无账本**：该 scope 从那一刻起没有激活账本，
    归属走 D1/五层兜底。`replay_ledger_events` 与 `switch_timeline` 都认这个语义
    （见各自实现）；`agency` 侧永不产出空 `to_ledger_id`，故新语义不与存量撞车。

    其余四类事件（create/update/user_edit/scope_migrate）只喂账本池、不进时间轴，
    直接丢弃即可。

    ## 🔴 2026-09-18（批 S · S2.1，MQ-I19）：墓碑**从它发生的那一刻起**生效，不改写历史

    上面"把 switch 改写成切到空"是把墓碑**倒填到切换那一刻**：账本活着的那段
    窗口从时间轴上整段消失，窗口内的轮次在重建/滞后蒸馏时查不到激活账本，
    产出落到 DPL 内容卡上（批 S 实测：live 库 232 条 fact 的源轮次发生在账本
    存活期内、却在墓碑之后才入库 ⇒ 全部落内容卡；Q3 真跑 S3「fact 换卡长回来」
    同一成因）。而拍板 0.3.0-(2) 的语义是**墓碑账本 ≠ 清派生记忆**：任务收摊，
    知识不作废——知识的外部边界（账本的激活窗口）也不该跟着作废。

    故改为：**已知墓碑时刻**（`tombstone_ts_ms[lid]`）时，切入该账本的 switch
    **原样保留**，并在墓碑时刻**合成一条「切到空」**（`to_ledger_id=""`，与既有
    语义同形）——只对墓碑那一刻仍激活着该账本的 scope 合成，早已切走的 scope
    不动（否则会把后一本账本的窗口切断，同型于 MQ-L15）。合成事件按 ts 插回
    序列，`replay_ledger_events`（last-write-wins）与 `switch_timeline`（点时）
    都因此得到正确答案：墓碑之前 = 该账本在场；墓碑之后 = 无账本。

    墓碑时刻**未知**（调用方没传 / 该 id 不在表里）⇒ 回落旧行为（改写切入为空）
    ——缺数不当作"从未在场"以外的任何强信号。

    三个消费方的行为差异：agency 池 / flash 物化只看池（create/update 仍丢弃，
    **零变化**）；激活绑定终态与此前一致（墓碑那一刻仍激活 ⇒ 合成的空切换
    把它 pop 掉）；**只有重建锚层**看到了多出来的窗口——这正是本条要修的对象。
    """
    if not tombstoned:
        return list(events)
    ts_map = tombstone_ts_ms or {}
    out: list[dict] = []
    kept_switch_scopes: dict[str, set[str]] = {}   # lid → 保留了切入事件的 scope 集合
    for ev in events:
        etype = ev.get("type", "")
        payload = ev.get("payload") or {}
        if etype in (EVENT_LEDGER_CREATE, EVENT_LEDGER_UPDATE, EVENT_LEDGER_USER_EDIT):
            lid = (payload.get("ledger") or {}).get("ledger_id", "")
            if lid in tombstoned:
                continue
        elif etype in LEDGER_LIFECYCLE_EVENTS:
            # 批 Z · Z3：墓碑**前**的关本是时间轴事件（它比墓碑更早结束了窗口），原样保留——
            # 丢掉它会让窗口越过关本时刻一直延到墓碑（换受害者，MQ-L15 同型）。
            # 墓碑时刻未知 / 关本晚于墓碑 ⇒ 丢弃（切入已被改写成切到空，它无事可做）。
            lid = payload.get("ledger_id", "")
            if lid in tombstoned:
                t_tomb = ts_map.get(lid)
                if not (t_tomb and _event_ts_ms(ev) < t_tomb):
                    continue
        elif etype == EVENT_LEDGER_SCOPE_MIGRATE:
            lid = payload.get("ledger_id", "")
            if lid in tombstoned:
                # 🔴 批 Y · Y1a（2026-09-23）：迁移现在也是时间轴事件（见 `switch_timeline`）。
                # 墓碑前的迁移与墓碑前的切入同理**原样保留**，dst scope 记进待合成集合——
                # 丢掉它会让 Global 窗口越过迁移时刻一直延到墓碑、项目键上的那段窗口整段消失
                # （换了个受害者，MQ-L15 同型）。墓碑时刻未知 / 迁移晚于墓碑 ⇒ 丢弃（旧行为）。
                t_tomb = ts_map.get(lid)
                agent = payload.get("agent_id", "")
                if not (t_tomb and agent and _event_ts_ms(ev) < t_tomb):
                    continue
                from .ledger_runtime import activation_scope  # 循环导入：延迟到调用点
                kept_switch_scopes.setdefault(lid, set()).add(
                    activation_scope(agent, payload.get("to_project", "")))
        elif etype == EVENT_LEDGER_SWITCH:
            to = payload.get("to_ledger_id", "")
            if to in tombstoned:
                t_tomb = ts_map.get(to)
                agent = payload.get("agent_id", "")
                if t_tomb and agent and _event_ts_ms(ev) < t_tomb:
                    from .ledger_runtime import activation_scope  # 循环导入：延迟到调用点
                    kept_switch_scopes.setdefault(to, set()).add(
                        activation_scope(agent, payload.get("project_id", "")))
                else:
                    ev = {**ev, "payload": {**payload, "to_ledger_id": ""}}
        out.append(ev)
    if not kept_switch_scopes:
        return out
    # 合成「墓碑时刻切到空」：只对那一刻仍激活着该账本的 scope。
    timeline = switch_timeline(out)
    synthesized: list[dict] = []
    for lid, scopes in kept_switch_scopes.items():
        t_tomb = ts_map[lid]
        for scope in sorted(scopes):
            if ledger_active_at(timeline, scope, t_tomb) != lid:
                continue
            from .ledger_runtime import split_scope
            agent, project = split_scope(scope)
            # ts 用 datetime：`_event_ts_ms` 对裸数值按量级猜秒/毫秒，datetime 无歧义。
            from datetime import UTC as _UTC
            from datetime import datetime as _dt
            synthesized.append({
                "type": EVENT_LEDGER_SWITCH, "ts": _dt.fromtimestamp(t_tomb / 1000.0, _UTC),
                "payload": {"to_ledger_id": "", "agent_id": agent,
                            "project_id": "" if project in ("", "Global") else project,
                            "reason": "ledger_tombstone", "from_ledger_id": lid},
            })
    if not synthesized:
        return out
    # 按 ts 插回：last-write-wins 的重放靠顺序，合成事件必须排在它之前的真实事件之后、
    # 之后的真实事件之前（否则墓碑之后切到另一本账本的绑定会被它 pop 掉）。
    import bisect
    ts_list = [_event_ts_ms(e) for e in out]
    for sev in sorted(synthesized, key=_event_ts_ms):
        t = _event_ts_ms(sev)
        idx = bisect.bisect_right(ts_list, t)
        out.insert(idx, sev)
        ts_list.insert(idx, t)
    return out


def ledger_activation_windows(
    timeline: dict[str, list[tuple[float, str]]], ledger_id: str, *,
    until_ms: float | None = None,
) -> dict[str, list[tuple[float, float | None]]]:
    """某本账本在各 scope 上的**激活窗口** `{scope: [(start_ms, end_ms|None), …]}`。

    窗口 = 从切入它到下一次切换（含合成的「切到空」）；最后一段若没有后续切换，
    `end_ms=None`（仍激活）或 `until_ms`（调用方给的截止，如"现在"）。
    消费方：`bladex ledger tombstone --with-memory`（拍板 0.3.0-(2)）按窗口给源轮次打
    TURN 墓碑；仪器（R1b / D1-B）按窗口取 fact。**与 `ledger_active_at` 同一条时间轴**，
    不另起判据。
    """
    out: dict[str, list[tuple[float, float | None]]] = {}
    for scope, seq in timeline.items():
        windows: list[tuple[float, float | None]] = []
        start: float | None = None
        for ts, lid in seq:
            if lid == ledger_id and start is None:
                start = ts
            elif lid != ledger_id and start is not None:
                windows.append((start, ts))
                start = None
        if start is not None:
            windows.append((start, until_ms))
        if windows:
            out[scope] = windows
    return out


def replay_ledger_events(
    events: list[dict],
) -> tuple[dict[str, Ledger], dict[str, str]]:
    """事件流 → (账本池, 激活绑定 **scope→ledger_id**)。

    🔴 2026-08-26 拍板：绑定键从 `session_id` 改 **`activation_scope(agent, project)`**
    （MQ-L11：session 是最短命的锚，agent 每次压缩/新开 context window 就换一个，
    账本连续性随之断裂）。**零数据迁移**——`LEDGER_SWITCH` 事件的 payload 从第一天
    就带 `agent_id`，此前只是被这个函数丢掉了；历史事件没有 `project_id`
    ⇒ 落 `Global`，与"信号未落地时全是 Global"的现状一致。

    状态式重放（last-write-wins，按事件顺序）：`ledger_create/update/user_edit` 的
    payload 都带完整 dump，重放 = 覆盖——**故意不做 diff 重放**，等价性最简形态，
    与 FACT_IMPORT"payload=实体完整 dump"同一先例。事件顺序即 Hub 追加序，
    调用方按 journal 序传入，本函数不重排。
    """
    pool: dict[str, Ledger] = {}
    bindings: dict[str, str] = {}
    for ev in events:
        etype = ev.get("type", "")
        payload = ev.get("payload") or {}
        if etype in (EVENT_LEDGER_CREATE, EVENT_LEDGER_UPDATE, EVENT_LEDGER_USER_EDIT):
            dump = payload.get("ledger") or {}
            try:
                ledger = Ledger.model_validate(dump)
            except Exception as e:  # noqa: BLE001 —— 单条坏事件不炸整个重放
                logger.warning("ledger_replay_bad_event type=%s error=%s", etype, e)
                continue
            pool[ledger.ledger_id] = ledger
        elif etype == EVENT_LEDGER_SWITCH:
            to = payload.get("to_ledger_id", "")
            agent = payload.get("agent_id", "")
            if not to and agent:
                # 🔴 空 to = **切到无账本**（`apply_ledger_tombstones` 的产物）。
                # 必须 pop 而不是"跳过"：跳过会让上一条绑定留下来，
                # 等于让前一本账本继续对这个 scope 生效——正是墓碑要消除的那件事。
                from .ledger_runtime import activation_scope
                bindings.pop(activation_scope(agent, payload.get("project_id", "")), None)
            elif to and agent:
                from .ledger_runtime import activation_scope  # 循环导入：延迟到调用点
                bindings[activation_scope(agent, payload.get("project_id", ""))] = to
            elif to:
                # agent_id 缺失（理论上不该发生；防御性保留 session 键，
                # 让这条绑定至少不消失，且键形态不同不会与 scope 键撞车）。
                sid = payload.get("session_id", "")
                if sid:
                    bindings[sid] = to
        elif etype == EVENT_LEDGER_SCOPE_MIGRATE:
            agent = payload.get("agent_id", "")
            lid = payload.get("ledger_id", "")
            if agent and lid:
                from .ledger_runtime import activation_scope
                src = activation_scope(agent, payload.get("from_project", ""))
                dst = activation_scope(agent, payload.get("to_project", ""))
                # 搬 = src 清除 + dst 设值（重放等价于 runtime migrate）
                if bindings.get(src) == lid:
                    bindings.pop(src, None)
                bindings[dst] = lid
        elif etype in LEDGER_LIFECYCLE_EVENTS:
            # 批 Z · Z3：与运行时 `AgencyRuntime.close_ledger / reopen_ledger` 同一变换
            # （`close_ledger` / `reopen_ledger`），重放 = 运行时。
            lid = payload.get("ledger_id", "")
            led = pool.get(lid)
            if led is None:
                logger.warning("ledger_replay_lifecycle_unknown type=%s ledger=%s", etype, lid)
                continue
            if etype == EVENT_LEDGER_REOPEN:
                pool[lid] = reopen_ledger(led)
                continue
            # E4（MQ-L98）/ 批 HO（MQ-L99）：谁关的 = 载荷写者会话与 agent（工具关本才有；admin / CLI 为空），
            # 何时关的 = 事件时刻——与运行时 `lifecycle_close` 写进池里的值同源。
            _closed_ms = _event_ts_ms(ev)
            pool[lid] = close_ledger(
                led,
                superseded_by=(payload.get("superseded_by", "")
                               if etype == EVENT_LEDGER_SUPERSEDE else ""),
                closed_by_session=str(payload.get("session_id", "") or ""),
                closed_by_agent=str(payload.get("agent_id", "") or ""),
                closed_at=(datetime.fromtimestamp(_closed_ms / 1000.0, UTC).isoformat(timespec="seconds")
                           if _closed_ms > 0 else ""))
            # 关本 = 解除它在每个 scope 上的激活（与 `ActivationTable.forget_ledger` 同语义：
            # 该 scope 变成无激活账本，**不回退**到上一本——MQ-L15 同型）。
            for scope in [s for s, b in bindings.items() if b == lid]:
                bindings.pop(scope, None)
    return (pool, bindings)


def binding_sessions(events: list[dict]) -> dict[str, str]:
    """事件流 → **scope → 落定该绑定的 session_id**（与 `replay_ledger_events`
    的 bindings 同序 last-wins）。

    消费者 = 注入面的档位 gate（MQ-L28，2026-08-29 Pi 天气循环事故）：
    弱档只在「激活账本是本会话绑的」时注只读正文——判据需要绑定的来源会话，
    而 `ActivationTable.restore` 此前只恢复 scope→ledger_id，来源随重启丢失。
    🔴 不违反 2026-08-26「session 不参与**切换**判定」拍板——切换判定不碰这里，
    这是注入面的独立消费者。
    """
    out: dict[str, str] = {}
    bound: dict[str, str] = {}   # scope → ledger_id（只为关本时找得到该解除哪些 scope）
    for ev in events:
        etype = ev.get("type", "")
        if etype in _RELEASING_EVENTS:
            # 批 Z · Z3：关本解除绑定 ⇒ 来源会话随之作废（与 `replay_ledger_events` 同步）。
            lid = (ev.get("payload") or {}).get("ledger_id", "")
            for scope in [s for s, b in bound.items() if b == lid]:
                bound.pop(scope, None)
                out.pop(scope, None)
            continue
        if etype == EVENT_LEDGER_SCOPE_MIGRATE:
            payload = ev.get("payload") or {}
            agent = payload.get("agent_id", "")
            if agent:
                from .ledger_runtime import activation_scope
                # 搬走后 src 来源作废；dst 来源**有意不设**——迁移不是本会话
                # 的切换动作，MQ-L28 gate 对未知来源按继承保守处理（weak 不注）。
                src = activation_scope(agent, payload.get("from_project", ""))
                out.pop(src, None)
                lid = bound.pop(src, "") or payload.get("ledger_id", "")
                if lid:
                    bound[activation_scope(agent, payload.get("to_project", ""))] = lid
            continue
        if etype != EVENT_LEDGER_SWITCH:
            continue
        payload = ev.get("payload") or {}
        to = payload.get("to_ledger_id", "")
        agent = payload.get("agent_id", "")
        sid = payload.get("session_id", "")
        if agent:
            from .ledger_runtime import activation_scope
            scope = activation_scope(agent, payload.get("project_id", ""))
            if to:
                bound[scope] = to
            else:
                bound.pop(scope, None)
            if to and sid:
                out[scope] = sid
    return out


def _event_ts_ms(ev: dict) -> float:
    """事件的 wall-clock 毫秒；缺失/不合形返回 0.0（= 时间轴开端，永远在最前）。

    接受 `datetime`、epoch 秒（<1e11）、epoch 毫秒三种形态——调用方直接把
    `AdminEvent.ts` 塞进来即可，不必先转。
    """
    raw = ev.get("ts")
    if raw is None:
        return 0.0
    if hasattr(raw, "timestamp"):
        return float(raw.timestamp()) * 1000.0
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return v * 1000.0 if v < 1e11 else v


@dataclass
class DoneSpan:
    """一段「账本自述完结」区间（批 Z · Z3.2 / Y1b）：`(start_ms, end_ms]` 内锚层视为窗口暂停。

    - **开**：某次写入（create / update / user_edit，**正文有变化**）之后该本 `ledger_is_done`；
    - **关**：之后某次写入让它不再是完结形态（Open 或 Next 重新非空），或**有人显式切回它**
      （`ledger_switch` 的 `to` = 它、`from` ≠ 它）。`end_ms=None` = 至今未恢复。
    - `writers` = 区间内写过这本的**用户轮** `(session_id, turn_index)`：这些用户轮里的轮次
      照常锚（模型在这一轮里还在给它记账，完结写入所在的那一轮更是它的收尾）。
      只有载荷带写者的事件（本批起的 `ledger_update`）才进得来；历史事件没有 ⇒ 空集 ⇒ 不豁免。
    """

    start_ms: float
    end_ms: float | None = None
    writers: set[tuple[str, int]] = field(default_factory=set)


class SwitchTimeline(dict):
    """`switch_timeline` 的返回值：`{scope: [(ts_ms, ledger_id), …]}`，**dict 语义逐字不变**
    （`ledger_active_at` / `ledger_activation_windows` / 所有按 dict 读的仪器照旧），
    另挂 `done_spans: {ledger_id: [DoneSpan, …]}` 供唯一消费点
    `ledger_runtime.turn_active_ledger` 判「完结即停锚」。

    🔴 为什么挂在时间轴上而不是给 `turn_active_ledger` 加一个必传参数：锚层消费方有
    重建锚层 / 墓碑窗口 / 多支探针，漏传一处就是"同一轮两个答案"（原则 13 层 2）。
    挂在 `switch_timeline` 的产物上，拿到时间轴就拿到了停锚——**没有第二条取法**。
    手拼的普通 dict（如修前口径的探针）没有它 ⇒ 不停锚，正是"修前"的语义。
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.done_spans: dict[str, list[DoneSpan]] = {}


def _writer_turn(payload: dict) -> tuple[str, int] | None:
    """事件载荷里的写者用户轮 `(session_id, turn_index)`；缺任一 ⇒ None（缺数不猜）。"""
    sid = str(payload.get("session_id", "") or "")
    ti = payload.get("turn_index")
    if not sid or isinstance(ti, bool) or not isinstance(ti, int):
        return None
    return (sid, ti)


def _done_spans(events: list[dict]) -> dict[str, list[DoneSpan]]:
    """事件流 → 每本账本的完结区间（纯函数；判据见 `DoneSpan`）。

    只认**正文有变化**的写入：`bind-legacy-matters` 这类维护写（整本 dump、段不变）
    不是"模型说做完了"，不许借它重新开一段停锚。按 ts 排序处理（`load_ledger_events`
    会把墓碑账本的池事件补在最前面，顺序不可信）。
    """
    items: list[tuple[float, int, str, str, dict]] = []
    for i, ev in enumerate(events):
        etype = ev.get("type", "")
        payload = ev.get("payload") or {}
        if etype in (EVENT_LEDGER_CREATE, EVENT_LEDGER_UPDATE, EVENT_LEDGER_USER_EDIT):
            lid = (payload.get("ledger") or {}).get("ledger_id", "")
            if lid:
                items.append((_event_ts_ms(ev), i, "write", lid, payload))
        elif etype == EVENT_LEDGER_SWITCH:
            to = payload.get("to_ledger_id", "")
            if to and to != payload.get("from_ledger_id", ""):
                items.append((_event_ts_ms(ev), i, "switch_in", to, payload))
    items.sort(key=lambda it: (it[0], it[1]))
    spans: dict[str, list[DoneSpan]] = {}
    open_: dict[str, DoneSpan] = {}
    last_sections: dict[str, object] = {}
    for ts, _i, kind, lid, payload in items:
        span = open_.get(lid)
        if kind == "switch_in":
            if span is not None:      # 显式切回 ⇒ 恢复
                span.end_ms = ts
                del open_[lid]
            continue
        dump = payload.get("ledger") or {}
        sections = dump.get("sections") or {}
        if lid in last_sections and last_sections[lid] == sections:
            continue                  # 正文没变：不是一次写入
        last_sections[lid] = sections
        if ledger_is_done(dump):
            if span is None:
                span = DoneSpan(start_ms=ts)
                open_[lid] = span
                spans.setdefault(lid, []).append(span)
            writer = _writer_turn(payload)
            if writer is not None:
                span.writers.add(writer)
        elif span is not None:        # Open / Next 重新非空 ⇒ 恢复
            span.end_ms = ts
            del open_[lid]
    return spans


def switch_timeline(events: list[dict]) -> SwitchTimeline:
    """事件流 → **按 scope 的切换时间轴** `{scope: [(ts_ms, ledger_id), …]}`（升序）。

    🔴 为什么需要它（MQ-L15，2026-08-26 实测立）：`replay_ledger_events` 是
    last-write-wins，只给**终态**绑定。live 上 16 次 switch 压成 5 条终态——
    按终态给历史轮次归属，等于把一个 agent 先后做的**几件不同的事**合进
    同一个 Matter。**那是机制制造的误合并，第一红线。**

    故归属侧必须做**点时查找**（"这一轮发生时，激活的是哪本账本"），
    而不是"这个 agent 现在激活的是哪本"。

    **有意不改 `replay_ledger_events` 的返回**：它的终态语义对"当前激活是谁"
    （注入侧）是对的，两个消费方要的东西本就不同；合并成一个函数会让
    注入侧被迫处理它不需要的时间轴。事件顺序按调用方传入序，本函数只按
    `ts` 排序，不重排来源。

    ## 「建本即激活」也是一个窗口起点（批 T0，2026-09-22）

    建本这个动作本身就是"该 scope 从此刻起在这本账本上"。当前实现里
    `ledger_create` 之后紧跟着一条切到新本的 switch（全新 id 不可能 noop；
    批 Y1c 删防抖前它靠 `newly_created` 豁免才不被拦），所以两个起点**今天恒等**；但那是实现巧合，
    不是结构保证——`agency_switch_unknown_ledger`（MQ-L80）这类形态一旦让
    switch 发不出来，账本就只剩一条 create 事件、窗口整段消失（墓碑/清仓
    因此打 0 条而报成功，正是 MQ-I22 的同族）。故把 CREATE 也纳入起点。

    🔴 **只认载荷里带 scope 的 create**：`ledger_create` 的载荷历史上只有
    `{"ledger": {...}}`，**没有 agent_id / project_id**（`last_writer` 也 85/85 为空，
    MQ-L85）——没有 scope 就定不了在哪条轴上开窗，硬猜就是原则 13 层 4 要防的
    "值不来自它声称的来源"。故：老事件（存量 85/85）一律跳过，读数上
    **零变化**；生产侧自本批起在 create 载荷补 `agent_id` / `project_id`
    （`agency/handlers.py`），新事件才有第二个起点。

    ## 作用域迁移也是时间轴事件（批 Y · Y1a，2026-09-23）

    `ledger_scope_migrate`（曾由 `agency/handlers._maybe_migrate_binding` 产生，
    **批 Y · Y1d 2026-09-28 停产**，拍板 (30) / MQ-L93：全库 4 次迁移 3 次跨项目；
    Hub 里既有的事件是当时运行时的真实行为，**本段重放一字不动**）在运行时把
    `agent@Global` 的绑定**搬**到 `agent@project`（当时的 `ActivationTable.migrate`：src pop、
    dst 设值）。此前本函数不认它 ⇒ 时间轴上 Global 那本一直激活到下一次 Global 切换、
    项目键在第一次显式切换前**什么都没有**——而锚层终态兜底又把项目键的整段历史
    （含迁移之前）塞给迁移来的那本（MQ-L15 反向形状）。现在重放成两条：
    迁移时刻 src 切到空、dst 切到该账本 ⇒ **迁移之前的轮次查 Global、之后查项目键**。

    ## 关本与完结停锚（批 Z · Z3，2026-09-24）

    - `ledger_close` / `ledger_supersede`：在该本**那一刻激活着的每个 scope** 上插一条
      「切到空」（与墓碑合成的空切换同形）——关本之后的轮次不再锚到它，绑定也随之解除
      （`replay_ledger_events` 同步）。`ledger_reopen` 不进时间轴：重开不是激活。
    - 完结停锚（Y1b）**不改时间轴条目**——绑定没有解除（那是 close 的事），只是锚层在
      完结区间里不认它。区间挂在返回值的 `done_spans` 上（`SwitchTimeline`），唯一的
      消费点是 `ledger_runtime.turn_active_ledger`。
    """
    from .ledger_runtime import activation_scope  # 循环导入：延迟到调用点

    timeline: dict[str, list[tuple[float, str]]] = {}
    for ev in events:
        etype = ev.get("type", "")
        payload = ev.get("payload") or {}
        if etype == EVENT_LEDGER_SCOPE_MIGRATE:
            agent = payload.get("agent_id", "")
            lid = payload.get("ledger_id", "")
            if not agent or not lid:
                continue
            ts = _event_ts_ms(ev)
            timeline.setdefault(activation_scope(agent, payload.get("from_project", "")),
                                []).append((ts, ""))
            timeline.setdefault(activation_scope(agent, payload.get("to_project", "")),
                                []).append((ts, lid))
            continue
        if etype == EVENT_LEDGER_CREATE:
            # 建本即激活：scope 只认载荷里显式带的那一对；缺 agent ⇒ 跳过（见 docstring）。
            agent = payload.get("agent_id", "")
            lid = (payload.get("ledger") or {}).get("ledger_id", "")
            if not agent or not lid:
                continue
            scope = activation_scope(agent, payload.get("project_id", ""))
            timeline.setdefault(scope, []).append((_event_ts_ms(ev), lid))
            continue
        if etype != EVENT_LEDGER_SWITCH:
            continue
        to = payload.get("to_ledger_id", "")
        agent = payload.get("agent_id", "")
        # 🔴 **空 `to` 要保留**（2026-09-01）：它是 `apply_ledger_tombstones` 写的
        # 「切到无账本」标记。此前这里 `if not to: continue` 会把它丢掉，
        # 于是前一本账本的窗口向后延伸、吃掉本属于被墓碑账本的那段轮次
        # ——墓碑白打。只有 `agent` 缺失才跳过（那条事件定不了 scope）。
        if not agent:
            continue
        scope = activation_scope(agent, payload.get("project_id", ""))
        timeline.setdefault(scope, []).append((_event_ts_ms(ev), to))
    for seq in timeline.values():
        seq.sort(key=lambda it: it[0])
    # ── 批 Z · Z3：关本 = 该本**当时激活着的每个 scope** 切到空 ──
    # 按 ts 顺序处理：后一次关本要看到前一次插进去的空切换。插在同刻条目之后
    # （`bisect_right`）——同一毫秒里先切入、再关，结果是关。
    import bisect
    closes = sorted(
        ((_event_ts_ms(ev), i, (ev.get("payload") or {}).get("ledger_id", ""))
         for i, ev in enumerate(events) if ev.get("type", "") in _RELEASING_EVENTS),
        key=lambda it: (it[0], it[1]))
    for ts, _i, lid in closes:
        if not lid:
            continue
        for seq in timeline.values():
            idx = bisect.bisect_right([it[0] for it in seq], ts)
            if idx > 0 and seq[idx - 1][1] == lid:
                seq.insert(idx, (ts, ""))
    out = SwitchTimeline(timeline)
    # 完结停锚默认关（`BLADEX_LEDGER_DONE_PAUSE`，读数见 flags 注释）：关着 ⇒ 区间为空 ⇒
    # `turn_active_ledger` 逐字等于 Z3 之前。开关读在这里（产出区间的唯一一处），
    # 消费点不必各自再判一遍。
    from .flags import flag_enabled
    out.done_spans = _done_spans(events) if flag_enabled("BLADEX_LEDGER_DONE_PAUSE") else {}
    return out


def ledger_active_at(timeline: dict[str, list[tuple[float, str]]],
                     scope: str, ts_ms: float) -> str:
    """`ts_ms` 那一刻 `scope` 上激活的账本；该刻之前无切换则返回 ""。

    🔴 **早于第一次切换的轮次返回空**，不回落到"第一本账本"——那一段本来就
    没有账本，归属该走 D1/DPL 兜底。回落会把建账本之前的历史强行塞进
    锚定 Matter（同一个误合并形状，方向相反）。
    """
    import bisect

    seq = timeline.get(scope) or []
    if not seq:
        return ""
    idx = bisect.bisect_right([it[0] for it in seq], float(ts_ms))
    return seq[idx - 1][1] if idx > 0 else ""
