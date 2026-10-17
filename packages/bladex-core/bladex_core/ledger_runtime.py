"""账本运行时 —— 激活/切换举证/工具更新映射/陈旧兜底/Matter 锚点（批二卡 V-L2–L5）。

叶子模块：只依赖 `ledger`（V-L1）与 `flags`。零 IO——绑定表与轮计数由调用方持有
（Hub 投影语义与 `splice.SpliceLedger` 同构），本模块只做纯判定与纯变换。

# 各段职责

- **L2 激活**：每任务一账本、每会话绑一激活账本；切换=管理事件+**举证**
  （批 Y · Y1c，MQ-L91，2026-09-24：原「N 轮内重复切换拒绝」的轮数防抖已删——
  它是 ADR-0031「默认延续、要证明是新的事才切换」的错误翻译，证据从**内容**
  换成了**轮数计数**，上线一个月没有一例被确认拦对。举证 = 模型摘录本轮用户原话，
  由 `verify_switch_evidence` 对照剥信封后的用户文本校验；**观测阶段只记不拦**，
  拦截阶段要有读数才开，见 `task-ledger-switch-debounce-rethink-20260924.md` §5）。
- **L3 更新通道**：toolface `bladex_ledger_update` 参数 → V-L1 条目原语。
  actor 恒 `model`（工具面只有模型在调）⇒ Goal 写保护自动生效（V-L1 红线 1）。
  带 `ref` 的条目来源标 `tool`（有证据），否则 `model`。
- **L4 陈旧兜底**：账本落后 N 轮即注入陈旧标记——错误的 Next 比不注入更有害
  （批一卡 V-L4 立论），标记让 LLM 自行折价。
- **L5 Matter 锚点**：有账本的流量归属走锚（`anchor_decision`），D1/DPL 降兜底
  （ADR-0032 §9.2 / §10-6 双轨拍板）；一本账本原则上锚一个 Matter，
  第二个出现必须告警人工复核（拍板 #7 连带、误合并=0 红线侧）。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from .flags import flag_number
from .ledger import (
    ACTOR_MODEL,
    ENTRY_SOURCE_MODEL,
    ENTRY_SOURCE_TOOL,
    SECTION_GOAL,
    STATUS_CLOSED,
    Ledger,
    LedgerEntry,
    LedgerError,
    add_entry,
    remove_entry,
    revise_goal,
)

# iso_ms 定义已搬到 ledger.py（渲染/排序同用）；这里保留导出名给既有调用方。
# `as iso_ms` 冗余别名 = 显式再导出，ruff F401 不再当未用 import 删掉（批 O 自动修曾删过一次，
# test_ledger.py / test_ledger_gate.py / probe_cont_head_layer.py 三处调用方当场红）。
from .ledger import iso_ms as iso_ms

# 数值默认经 flags 单一真相源（BLADEX_LEDGER_STALE_TURNS）。


# ── L2 · 激活与切换举证 ─────────────────────────────────────────────────────


@dataclass
class SwitchVerdict:
    allowed: bool
    #: 唯一的拒绝理由是 `noop`（同本重复激活，幂等）——批 Y1c 起**没有**启发式拒绝。
    reason: str = ""                # noop
    event_payload: dict = field(default_factory=dict)   # LEDGER_SWITCH 的 payload
    #: 这个 scope 上一次落定切换所属的 session（空=首次）。仅供调用方判"并发争用"
    #: 并告警——**不参与判定**，见 `activation_scope` 里 project 信号未落地那段。
    prev_session: str = ""
    #: 距上次切换的秒数（`-1` = 首次）。争用判据要的是"**并发**"，不是"换了 session"。
    prev_age_s: float = -1.0
    #: 🔴 **只供观测、不参与判定**（`agency_switch_evidence` 的形态字段，Y1c）：
    #: 防抖删掉后「一轮里 A→B→A」这类快速切换会放行——不拦，但要看得见。
    #: 读数来自进程内轨迹 `ActivationTable._switch_trace`，**不从 Hub 恢复**
    #: （重启后首次切换这几格为空，不做跨口径相减——调度复核第 3 条）。
    #: `switch_in_turn` = 本 session 本轮第几次落定切换（1 起）；
    #: `delta_turns` = 距上次落定切换的轮次数（仅同 session，否则 None）；
    #: `back_to_prev` = 切回上一次切换离开的那本（A→B→A）。
    switch_in_turn: int = 1
    delta_turns: int | None = None
    back_to_prev: bool = False


@dataclass(frozen=True)
class _SwitchTrace:
    """一个 scope 上最近一次落定切换的**观测轨迹**（Y1c）。只进告警，不进判定。"""
    session_id: str
    turn_index: int
    from_ledger_id: str
    to_ledger_id: str
    n_in_turn: int


#: 无项目信号时的 project 段（Hermes 这类没有项目概念的 agent 恒落这里）。
ACTIVATION_PROJECT_GLOBAL = "Global"

#: scope 键的分隔符：用 US（unit separator）而非 `:`——agent_id 本身带冒号
#: （`hermes:default`），用冒号拼会产生歧义键。
_SCOPE_SEP = "\x1f"


def activation_scope(agent_id: str, project_id: str = "") -> str:
    """激活账本的绑定作用域键 = **(agent, project)**（Jason 2026-08-26 拍板）。

    🔴 **为什么不是 session**（MQ-L11，jydesignhk 病例）：账本原本按 session 绑定，
    而 session 是所有候选锚里**最短命**的——agent 每次压缩 / 新开 context window
    就换一个。实测：Hermes 在 `[CONTEXT COMPACTION — REFERENCE ONLY] … handoff from
    a previous context window` 之后拿到新指纹 `fp:d53cbb3bb15f`，之前两张账本绑在
    `fp:955be74ada9f` 上 ⇒ **交接的那一刻账本连续性断掉**，而跨会话交接正是北极星
    要接住的场景。改 (agent, project) 后同一病例会走"有账本分支"，模型拿到的是
    Goal 对照而不是冷启动。

    **为什么带 project**：光按 agent 会让同一 agent 的并发会话抢同一本
    （两个 Claude Code 窗口开在不同仓库 ⇒ 互相把对方账本切走）。project 段用
    ADR-0032 §5 四级模型（User→Agent→**Project**→Session）里那一级，由
    `bladex_proxy.project_identity.resolve_project` 判定（生产唯一实现点；原
    `flash_tree.resolve_project_identity` 从未接进生产，批 F4 随树删除）；
    没有信号的 agent（Hermes 形态）落 Global，
    此时**等价于 per-agent**——正是拍板要的效果。

    ✅ project 信号 2026-08-29 已接通：识别四档（git-remote → git-root → path →
    Global，`bladex_proxy.project_identity`）live，`identity.project_id` 已接进
    注入面（insert_ledger_block）与工具面（process_message→ctx）两个消费点
    （守卫 `test_ledger_rev_lock.py::test_server_threads_project_id`）。
    历史事件无 project 维 ⇒ 重放落 Global 键，零迁移。
    """
    agent = (agent_id or "unknown").strip()
    project = (project_id or "").strip() or ACTIVATION_PROJECT_GLOBAL
    return f"{agent}{_SCOPE_SEP}{project}"


def split_scope(scope: str) -> tuple[str, str]:
    """`activation_scope()` 的逆——`(agent, project)`；不合形则 `(scope, Global)`。

    只读视图（dashboard / CLI）要分别显示两段，但**分隔符是实现细节**——
    调用方自己 `split` 就把它复制成了第二份真相。给出这个函数，
    改分隔符时只需要改这一处。
    """
    agent, sep, project = (scope or "").partition(_SCOPE_SEP)
    if not sep:
        return (agent, ACTIVATION_PROJECT_GLOBAL)
    return (agent, project or ACTIVATION_PROJECT_GLOBAL)


def user_turn_index(messages: list | None) -> int:
    """轮次参照系 = **轮次数**，按 ADR-0033 §2.2 规则计（`task_unit.turn_starts`）。

    🔴 **单一实现点**——`identity.resolve_identity`（`Identity.turn_index`）与
    `agency.tool_context` 都调这里。修前两处各写 `len(messages)//2`（"一问一答算一轮"），
    那是**请求数**：编程 agent（CC/Codex/Pi）每个工具调用是一次独立请求、消息 +2
    ⇒ 当时的切换防抖（3 轮，Y1c 已删）在工具循环里 ≈ 3 个 bash 调用，
    `STALE_TURNS=20` ≈ 一次 grep+读+改+测（什么都没做错就被喊"陈旧"）。
    同一用户轮内的工具往返不动这个数；tool 结果消息（`role == "tool"`）不算。

    🔴 **口径切换（批 U · U1.1，2026-09-23；ADR-0033 §2.2「一套体系一个轮次定义」）**：
    MQ-L48（批 E E0.2，09-06）起按 `role == "user"` 计数；本批起**算法与
    `build_task_units` 同源**（都调 `turn_starts`）⇒ 工具循环中的用户插话、连续 user
    不再计轮——陈旧判定随之变宽松，与 L48「同一用户轮内工具往返不计」的本意一致。
    2026-09-23 之前产生的 `turn_index` 读数（Hub `LEDGER_SWITCH.turn_index` /
    `Identity.turn_index`）是 **role=user 计数口径**，不回填；跨这个日期的差值不可比。
    守卫：`test_turn_rule_single_source.py`（改回按 role=user 计数 ⇒ 红）。

    接受 dict 或带 `.role` 的对象；None/空 ⇒ 0。
    """
    from .task_unit import turn_starts

    return len(turn_starts(messages))


class ActivationTable:
    """(agent, project) → 激活账本 的绑定表（Hub `LEDGER_SWITCH` 事件的投影；
    `bladex_core.ledger.replay_ledger_events` 的 bindings 输出可直接 `restore`）。

    键由 `activation_scope()` 生成——**不要手拼**，分隔符是实现细节。
    """

    def __init__(self) -> None:
        self._active: dict[str, str] = {}
        #: scope → 最近一次落定切换所属的 session（并发争用告警 + MQ-L28 tier gate 用）
        self._last_switch_session: dict[str, str] = {}
        #: scope → 最近一次落定切换的**墙钟**（只为算"多久以前"——turn 跨会话不可比）。
        self._last_switch_at: dict[str, float] = {}
        #: scope → 最近一次落定切换的观测轨迹（Y1c，只进 `agency_switch_evidence`）。
        #: 🔴 **有意不进 `restore`**：Hub `LEDGER_SWITCH.turn_index` 存量跨两种轮次口径
        #: （09-23 批 U · U1.1 前为 role=user 计数），重启后拿它相减就是调度复核第 3 条
        #: 那个跨口径差值。进程内的值全是同一口径，观测读数只在进程内成立。
        self._switch_trace: dict[str, _SwitchTrace] = {}

    def restore(self, bindings: dict[str, str],
                sessions: dict[str, str] | None = None) -> None:
        """恢复绑定（+可选的绑定来源会话 `ledger.binding_sessions`——`LEDGER_SWITCH`
        事件的投影）。

        sessions 缺省 = 来源未知——注入面 tier gate 对未知来源按"继承"保守处理
        （MQ-L28：误注旧任务给弱模型的代价 > 弱档少注一轮上下文）。

        批 Y1c（2026-09-24）起**不再恢复任何轮位**：此前的 `switch_turns=`
        （`ledger.binding_switch_turns`）唯一消费方是切换防抖的基点，防抖已删，一并删除。"""
        self._active.update(bindings)
        if sessions:
            self._last_switch_session.update(sessions)

    def active(self, scope: str) -> str:
        return self._active.get(scope, "")

    def last_session(self, scope: str) -> str:
        """上一次在这个 scope 上落定切换的 session（争用告警用）。"""
        return self._last_switch_session.get(scope, "")

    def forget_ledger(self, ledger_id: str) -> list[str]:
        """把某本账本从**所有** scope 的激活位上摘掉。返回受影响的 scope 列表。

        2026-09-01 账本墓碑用（`task-ledger-tombstone-cleanup-20260901.md`）：
        墓碑后 Hub 侧已经不再重放它，但**活着的进程内表还留着**——不摘掉的话
        本次进程寿命内它仍会被注入、仍能被 switch 到，要等重启才收敛。

        🔴 摘掉是"该 scope 变成无激活账本"，**不是**回退到上一本：
        与 `apply_ledger_tombstones` 把 switch 改写成"切到空"同一语义。
        回退到上一本 = 让前一本吃掉本属于被墓碑账本的窗口（MQ-L15 同型）。

        观测轨迹与来源会话一并清除：它们描述的是"最近一次切到它"，
        账本没了这些记录就是悬空的。
        """
        hit = [s for s, lid in self._active.items() if lid == ledger_id]
        for s in hit:
            self._active.pop(s, None)
            self._switch_trace.pop(s, None)
            self._last_switch_session.pop(s, None)
            self._last_switch_at.pop(s, None)
        return hit

    def snapshot(self) -> dict[str, str]:
        """当前全部绑定 `{scope: ledger_id}` 的拷贝（只读视图用；改它不影响表）。"""
        return dict(self._active)

    def request_switch(self, scope: str, to_ledger_id: str, *,
                       agent_id: str, turn_index: int,
                       session_id: str = "",
                       project_id: str = "") -> SwitchVerdict:
        """判定一次切换请求。允许 ⇒ 应用到表并给出管理事件 payload（写 Hub 归调用方）。

        `scope` 由 `activation_scope(agent_id, project_id)` 生成；`session_id` **不是绑定键**
        （2026-08-26 拍板的核心变化），它进事件载荷（审计/溯源）。

        唯一的拒绝 = **noop**：同本重复激活不出事件（幂等，防事件流被重发灌水）。

        🔴 **批 Y · Y1c（MQ-L91，2026-09-24）删除轮数防抖**（原「同 session 距上次落定
        切换不足 N 轮 ⇒ 拒」+ `newly_created` / `user_requested` 两条豁免）。病根：ADR-0031
        「默认延续，要证明它是新的事才切换」在 v5 被翻译成了"要等 3 轮才能切换"——证据从
        **内容**换成了**轮数计数**。它豁免新建（拦不住真正的碎片 = 重复建本），却一直在拦
        正当切换：09-23 Hermes 飞书，距上次切换 13 小时的换题被拒两次，第三次只因轮数攒够
        才落定；上线一个月五个自身缺陷，零例被确认拦对（刚性原则 15）。
        「默认延续」改由**举证**承接：见 `verify_switch_evidence`，观测阶段只记不拦。

        `turn_index`（轮次数，`user_turn_index`）仍进事件载荷——它是 Hub 的原始数据
        （刚性原则 13 无损），只是不再被任何判定消费；观测轨迹（`SwitchVerdict` 的
        `switch_in_turn` / `delta_turns` / `back_to_prev`）只在同 session 内相减。
        """
        current = self._active.get(scope, "")
        if to_ledger_id == current:
            return SwitchVerdict(allowed=False, reason="noop")
        trace = self._switch_trace.get(scope)
        same_session = bool(trace is not None and session_id
                            and trace.session_id == session_id)
        n_in_turn = (trace.n_in_turn + 1
                     if same_session and trace.turn_index == int(turn_index) else 1)
        delta_turns = int(turn_index) - trace.turn_index if same_session else None
        back_to_prev = bool(trace is not None and trace.to_ledger_id == current
                            and trace.from_ledger_id == to_ledger_id
                            and trace.from_ledger_id != "")
        self._active[scope] = to_ledger_id
        self._switch_trace[scope] = _SwitchTrace(
            session_id=session_id, turn_index=int(turn_index),
            from_ledger_id=current, to_ledger_id=to_ledger_id, n_in_turn=n_in_turn)
        prev_session = self._last_switch_session.get(scope, "")
        prev_at = self._last_switch_at.get(scope)
        prev_age = -1.0 if prev_at is None else max(0.0, time.time() - prev_at)
        if session_id:
            self._last_switch_session[scope] = session_id
        self._last_switch_at[scope] = time.time()
        return SwitchVerdict(allowed=True, event_payload={
            "session_id": session_id, "agent_id": agent_id,
            "project_id": project_id or ACTIVATION_PROJECT_GLOBAL,
            "from_ledger_id": current, "to_ledger_id": to_ledger_id,
            # E0.2 起落定时的轮次进事件（Hub 原始数据，原则 13 无损）。Y1c 起无判定消费方。
            "turn_index": int(turn_index),
        }, prev_session=prev_session, prev_age_s=prev_age,
            switch_in_turn=n_in_turn, delta_turns=delta_turns,
            back_to_prev=back_to_prev)


# ── L2 · 切换举证（批 Y · Y1c，MQ-L91）────────────────────────────────────────

#: 举证来源闭集（`LEDGER_SWITCH.evidence_source` / `agency_switch_evidence.source`）。
#: - `user_text`       模型给了 `evidence`，按本轮用户原话校验（唯一需要校验的一种）；
#: - `new_ledger`      新建——证据是模型提炼的 Goal（建本本身是显式决定）；
#: - `child_writeback` 子账本回写父本——证据是子本 Next 里 `ref=<parent>` 那条义务；
#: - ``（空）          没有任何证据（观测阶段照常切换，`evidence_ok=None`）。
EVIDENCE_SOURCES: tuple[str, ...] = ("user_text", "new_ledger", "child_writeback", "")

#: 摘录的最小字数（归一化后）。1 个字几乎必然是任意用户文本的子串，不构成证据。
EVIDENCE_MIN_CHARS = 2


def _norm_evidence(text: str) -> str:
    # 空白折叠 + casefold：模型摘录时换行/多空格/大小写的出入不算编造。
    return " ".join(str(text or "").split()).casefold()


def verify_switch_evidence(evidence: str, user_text: str) -> bool | None:
    """`evidence` 是否是本轮用户原话的片段。**单一实现点**（Y1c 判据 5）。

    返回 None = 没给证据（与"给了但对不上"的 False 分开——观测阶段两者都照常切换，
    但它们回答的是两个问题：模型愿不愿举证 / 举的证是不是真的）。

    🔴 `user_text` 必须是**剥过信封**的本轮用户文本，调用方只许传
    `tool_context()["user_query"]`（`_last_user_text(for_goal=True)`，与 Goal 改动的
    引文锚同源）——**值要能回溯到它声称的来源**（刚性原则 13 层 4）。传模型自己给的
    任何参数进来，校验就退化成"模型自报"。守卫：`test_switch_evidence.py`。
    首尾引号/空白不计；归一化后不足 `EVIDENCE_MIN_CHARS` 字 ⇒ False（太短不成证据）。
    """
    ev = _norm_evidence(evidence).strip("\"'“”‘’「」『』`")
    if not ev:
        return None
    if len(ev) < EVIDENCE_MIN_CHARS:
        return False
    return ev in _norm_evidence(user_text)


# ── L3 · 工具更新通道 ───────────────────────────────────────────────────────


#: 条目改动的 op 闭集。**单一定义**——schema 的 enum 与本模块的分支由
#: `test_f_b_batch_update.py::test_schema_ops_match_runtime` 对账（枚举闭集要从
#: 定义读，feedback_closed_set_from_definition）。
UPDATE_OPS: tuple[str, ...] = ("add", "remove")

#: `match` 报歧义时列出的候选条数上限。多了对模型没用——它要的是"再写长一点"
#: 这个信号，不是一份清单（每个字都花注意力预算，ADR-0029）。
_MATCH_CANDIDATES_SHOWN = 3


def resolve_match(entries: list[LedgerEntry], match: str) -> int:
    """`match`（条目全文或唯一前缀）→ 段内下标。MQ-L49 ②。

    🔴 **为什么要有它**：`index` 要求模型先数一遍注入块里的条目位置，而它读到的
    那份可能已经被另一个 agent 改过（rev 锁挡住的正是这一刀）。按内容删则与位置
    无关——"删掉我刚写完的那条 Next"这件事，模型说得出内容，说不准位置。

    🔴 **两种歧义分开报**：多条**全文相等** ⇒ 让模型改用 index（列文本没有信息量，
    它们一模一样）；多条**前缀命中** ⇒ 列前 3 条文本，模型据此把 match 写长。
    零命中也是错误、不是无操作——静默不删会让模型以为删掉了，下一步就把它
    写进 Verified（MQ-L29 同族形态）。
    """
    key = (match or "").strip()
    if not key:
        raise LedgerError("match must be non-empty")
    exact = [i for i, e in enumerate(entries) if e.text.strip() == key]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise LedgerError(
            f"match {key!r} is not unique: entries {exact} have identical text; "
            "pass index to say which one")
    pref = [i for i, e in enumerate(entries) if e.text.strip().startswith(key)]
    if len(pref) == 1:
        return pref[0]
    if not pref:
        raise LedgerError(
            f"no entry matching {key!r}; match takes the exact text of the entry "
            "you mean, or a unique prefix of it")
    shown = ", ".join(repr(entries[i].text[:60])
                      for i in pref[:_MATCH_CANDIDATES_SHOWN])
    raise LedgerError(
        f"match {key!r} matches {len(pref)} entries ({shown}); "
        "make it longer, or pass index")


def _match_sections(ledger: Ledger, match: str) -> list[tuple[str, list[int]]]:
    """`match` 在**哪些段**里有候选（MQ-L53：`remove` 不给 `section` 时用）。

    段内精确优先于前缀，是 `resolve_match` 的既有规则；跨段**沿用同一条**——
    某段有精确命中时，只前缀命中的段一律出局。规则一份，不是两份。

    🔴 **跳过 `goal`**：Goal 仅用户可改（ADR-0032 三红线之一）。段级 enum 里本来
    就没有 goal，全账本搜索是新开的一条路，得把同一道门再关一次。
    """
    key = (match or "").strip()
    exact: list[tuple[str, list[int]]] = []
    prefix: list[tuple[str, list[int]]] = []
    for sec in ledger.section_order:
        if sec == SECTION_GOAL:
            continue
        entries = ledger.entries(sec)
        hit = [i for i, e in enumerate(entries) if e.text.strip() == key]
        if hit:
            exact.append((sec, hit))
            continue
        hit = [i for i, e in enumerate(entries) if e.text.strip().startswith(key)]
        if hit:
            prefix.append((sec, hit))
    return exact or prefix


def _resolve_section_for_match(ledger: Ledger, match: str) -> str:
    """`remove` 没给 `section` 时，定位到唯一那一段（MQ-L53）。

    三种结果，与 `resolve_match` 的三种失败**同族**（都要么删对、要么说清楚为什么
    删不了；静默不删是最坏的一种——模型会以为删掉了，下一步就把它写进 Verified）：

    - 恰好一段命中 ⇒ 返回段名，段**内**的歧义交回 `resolve_match`（它的三条报错
      一字不动）；
    - 跨段命中多条 ⇒ 报错带**段名 + 候选文本**（模型要的是"再写长一点或补 section"
      这个信号）；
    - 零命中 ⇒ 报错。
    """
    key = (match or "").strip()
    if not key:
        raise LedgerError("match must be non-empty")
    hits = _match_sections(ledger, key)
    if len(hits) == 1:
        return hits[0][0]
    if not hits:
        raise LedgerError(
            f"no entry matching {key!r} in any section; match takes the exact "
            "text of the entry you mean, or a unique prefix of it")
    shown = ", ".join(
        f"{sec}: {ledger.entries(sec)[idxs[0]].text[:60]!r}"
        for sec, idxs in hits[:_MATCH_CANDIDATES_SHOWN])
    raise LedgerError(
        f"match {key!r} matches entries in {len(hits)} sections ({shown}); "
        "pass section to say which one, or make match longer")


def _apply_one(ledger: Ledger, spec: dict, *, updated_at: str,
               writer: str = "") -> tuple[Ledger, str]:
    """一条条目改动 → (新账本, 确认文本)。**单条形态与批量共用这一个实现点**
    （两份实现改一处忘一处，是本仓反复付过学费的形状）。

    纯变换：不落盘、不发事件、失败以 `LedgerError` 上抛，调用方据此整批回滚
    （返回的是新对象，抛错时调用方手上那份原封不动）。
    """
    section = str(spec.get("section", "")).strip().lower()
    op = str(spec.get("op", "")).strip().lower()
    if op == "add":
        text = str(spec.get("text", "")).strip()
        if not text:
            raise LedgerError("add requires non-empty text")
        ref = str(spec.get("ref", "")).strip()
        entry = LedgerEntry(text=text,
                            source=ENTRY_SOURCE_TOOL if ref else ENTRY_SOURCE_MODEL,
                            ref=ref, writer=writer)
        new = add_entry(ledger, section, entry, actor=ACTOR_MODEL,
                        updated_at=updated_at)
        return new, f"added to {section}: {text}"
    if op == "remove":
        match = str(spec.get("match", "")).strip()
        if match:
            # MQ-L53：没给 `section` ⇒ 全账本按 `match` 搜（goal 除外）。
            # `match` 的契约本就是"全文或唯一前缀"，段名是冗余——两个互不相干的
            # agent 各自漏掉它，那是接口的问题（刚性原则 10）。
            if not section:
                section = _resolve_section_for_match(ledger, match)
            idx = resolve_match(ledger.entries(section), match)
        else:
            idx = spec.get("index")
            if not isinstance(idx, int):
                raise LedgerError(
                    "remove requires match (the entry text or a unique prefix "
                    f"of it) or an integer index — got keys {sorted(spec) or '(none)'}")
            if not section:
                # `index` 是**段内**位置，没有段就没有意义 —— 这一条不放宽。
                raise LedgerError(
                    "remove by index requires section (index is a position "
                    "within one section); or pass match instead")
        new, removed = remove_entry(ledger, section, idx, updated_at=updated_at)
        return new, f"removed from {section}: {removed.text}"
    raise LedgerError(f"unknown op {op!r} ({'|'.join(UPDATE_OPS)})")


#: 模型最可能写错的键名 → 我们认的那个。**只用来给提示，不做自动纠正**——
#: 替模型改参数就是替它做决定（三红线：BladeX 不做作者），而且静默纠正会让
#: 同一个错在下一个 agent 上再犯一次、永远不被发现。
_KEY_ALIASES: dict[str, str] = {
    "sec": "section", "segment": "section", "field": "section",
    "operation": "op", "action": "op", "type": "op",
    "content": "text", "value": "text", "entry": "text",
    "evidence": "ref", "source": "ref",
}


def _missing_hint(spec: dict) -> str:
    """给"缺 section/op"的报错补一句**可操作**的提示。

    两种最常见的写错法各给一句：① 键名近似（`operation` / `Section` 大小写）；
    ② 整条为空（模型用 `{}` 表示"就这些了"）。认不出就不猜——多说一句错的
    比不说更糟（它会照着错提示改第二遍）。
    """
    if not spec:
        return ". This entry is empty — drop it from the list instead of sending {}"
    lower = {str(k).lower(): k for k in spec}
    hints: list[str] = []
    for want in ("section", "op"):
        if want in spec:
            continue
        if want in lower:
            hints.append(f"you wrote {lower[want]!r}, it must be {want!r} (lowercase)")
            continue
        for alias, target in _KEY_ALIASES.items():
            if target == want and alias in lower:
                hints.append(f"{lower[alias]!r} is not {want!r}")
                break
    return (". " + "; ".join(hints)) if hints else ""


def _validate_spec(pos: int, spec: dict, ledger: Ledger) -> None:
    """批量里一条的**与账本状态无关**的校验（段名 / op / 必填字段）。

    先跑一遍再动手，让"第 4 条写错段名"在第 1 条落地之前就报出来。
    与状态有关的校验（index 范围、match 唯一命中）留给 `_apply_one`——它们要看
    的是**这一条执行时**的账本，而不是批次开始时的（批内先 add 后按 match 删
    同一条是合法形态）。两者都抛 `LedgerError`，整批语义相同。
    """
    where = f"entries[{pos}]"
    if not isinstance(spec, dict):
        raise LedgerError(
            f"{where} must be an object with section and op, got {type(spec).__name__}")
    section = str(spec.get("section", "")).strip().lower()
    op = str(spec.get("op", "")).strip().lower()
    # 🔴 MQ-L53（2026-09-08）：`section` 只在**推不出**它的时候才是必填。
    # `op=remove` + `match` 已经唯一确定条目（match 的契约就是"全文或唯一前缀"），
    # 再要一个段名是我们强加的冗余——两个互不相干的 agent 各自写出逐字相同的
    # `{op:"remove", match:"…"}`，那是接口的问题不是模型的问题（刚性原则 10）。
    match_given = bool(str(spec.get("match", "")).strip())
    section_optional = (op == "remove" and match_given)
    if not op or (not section and not section_optional):
        # 🔴 **回显收到了什么**（2026-09-08 事故驱动，MQ-L52）：原文只说"缺 section
        # 和 op"，模型看不出它写的与 BladeX 读到的差在哪 ⇒ 原样重发。live 实测
        # 连撞 6 次、rounds 撞上限、62 万 token、零写入。键名足以暴露真实病因
        # （大小写 / 拼写 / 整条为空），而**只回显键名不回显值**——值可能很长，
        # 且错误文本本身也是注入（ADR-0032 §3.2-7）。
        raise LedgerError(
            f"{where} needs both section and op — got keys {sorted(spec) or '(none)'}"
            f"{_missing_hint(spec)}")
    if op not in UPDATE_OPS:
        raise LedgerError(f"{where}: unknown op {op!r} ({'|'.join(UPDATE_OPS)})")
    if section and section not in ledger.section_order:
        raise LedgerError(
            f"{where}: unknown section {section!r}; "
            f"this ledger has {', '.join(ledger.section_order)}")
    if op == "add" and not str(spec.get("text", "")).strip():
        raise LedgerError(f"{where}: add requires non-empty text")
    if op == "remove" and not match_given and not isinstance(spec.get("index"), int):
        # 🔴 **第二类也回显 keys**（MQ-L53 第二半）：第一类之所以能被判成"schema 错"，
        # 正是因为报错回显了键名；这一类此前没有回显 ⇒ 模型当时写了什么键
        # （很可能是 `text`——remove 语境下"删这条文本"是最自然的写法，而
        # `_KEY_ALIASES` 里 `content/value/entry → text` 已在、独缺 `text → match`）
        # 在日志里**不可判定**。同一条判据不能只用在一半上。
        # **只回显不纠正**：把 `text` 当 `match` 用 = 替模型改参数（三红线：
        # BladeX 不做作者），而且静默纠正会让同一个错在下一个 agent 上再犯一次。
        raise LedgerError(
            f"{where}: remove requires match or index — got keys "
            f"{sorted(spec) or '(none)'}")


def update_edit_ops(args: dict) -> list[str]:
    """本次 `bladex_ledger_update` 调用的 op 序列（观测口径**单一定义**）。

    goal 调用 ⇒ `["goal"]`；批量 ⇒ 逐条 op；单条 ⇒ 一项。认不出的形态给
    `["?"]`——空列表会让"没读到"与"零改动"在读数里长成一样（分母纪律）。
    消费者：`agency_ledger_update_applied` 的 `n_edits=` / `ops=`（F-B1 判据）。
    """
    if str(args.get("goal", "")).strip():
        return ["goal"]
    raw = args.get("entries")
    if isinstance(raw, list) and raw:
        return [str(e.get("op", "?")).strip().lower() if isinstance(e, dict) else "?"
                for e in raw]
    op = str(args.get("op", "")).strip().lower()
    return [op or "?"]


def update_is_add_only(args: dict) -> bool:
    """这次调用是不是**只有 add**（F-B2 的 rev 判据，MQ-L49 ③）。

    🔴 立论：**add 可交换**。两个 agent 各加一条，谁先谁后结果都是两条都在——
    没有"基于旧视图错删"可言，而那正是乐观锁存在的理由（`remove` 按旧 index/
    内容删错东西；`goal` 是用户所有物）。live 读数：8 次 `agency_ledger_rev_conflict`
    **8/8 是同一个 agent 自撞**（并行 tool_calls 各带同一个 rev），跨 agent 真冲突 0
    ——被拦下的全是本该放行的 add。
    保守方向：认不出的形态返回 False（= 照旧判 rev）。
    """
    if str(args.get("goal", "")).strip():
        return False
    ops = update_edit_ops(args)
    return bool(ops) and all(op == "add" for op in ops)


def apply_tool_update(ledger: Ledger, args: dict, *, updated_at: str = "",
                      user_text: str = "",
                      writer: str = "") -> tuple[Ledger, str]:
    """toolface `bladex_ledger_update` 参数 → 账本变换。返回 (新账本, 给模型的确认文本)。

    条目段的 actor 恒 `model`；`goal` **不是条目段**，走 `revise_goal` 的锚定门
    （必须给出本轮用户原话中的依据）。`section` enum 里没有 goal，这里是第二道门
    ——两道门覆盖"模型伪造参数绕 schema"的形态。
    失败以 LedgerError 上抛，toolface.dispatch 的容错壳会把它变成给模型的 Error 文本。

    # 三种形态（MQ-L49 ①，2026-09-07）

    - `goal` + `goal_change_quote` —— Goal 改动，与条目改动互斥。
    - `entries: [{section, op, text|match|ref|index}, …]` —— **批量**，一次调用多条。
    - `section` + `op` + … —— 单条老形态，**逐字不动**（0.1.0 五 agent 的 live
      记录全是它；批量是新增参数不是替换）。

    `writer` = 写这一批条目的 agent 完整 id（F-B3，轴 C 数据面）。缺省空 ⇒ 条目
    不带写者标记，与历史形态逐字相同——调用方漏传只丢观测，不改账本语义。

    🔴 **批量是原子的**：任一条失败 ⇒ 整批不落、确认文本不产出、`rev` 不动
    （调用方在本函数返回后才写池，抛错时它手上那份原封不动）。半本账本比不落
    更糟——模型看不出哪几条成功了，只能重发，于是第二次把成功的那几条又写一遍。
    """
    goal_text = str(args.get("goal", "")).strip()
    raw_entries = args.get("entries")
    if goal_text:
        # Goal 改动与条目改动是两件事，一次调用只做一件——混在一起时
        # "哪一半成功了"会说不清，而 Goal 改动必须可审计。
        if isinstance(raw_entries, list) and raw_entries:
            # 🔴 响亮失败，不静默丢：老形态下 section/op 被 goal 悄悄忽略是既有
            # 行为（不改），但 entries 是这次新加的面，静默吞掉一整批条目
            # = 原则 13 的反面（丢数据要么不丢，要么留显式标记）。
            raise LedgerError(
                "a goal change and entry edits must be separate calls; "
                "send the entries in their own bladex_ledger_update call")
        new = revise_goal(ledger, goal_text,
                          quote=str(args.get("goal_change_quote", "")),
                          user_text=user_text, updated_at=updated_at)
        return new, f"goal updated (revision {new.goal_revisions}): {goal_text}"
    if raw_entries is not None and not isinstance(raw_entries, list):
        raise LedgerError("entries must be a list of {section, op, …} objects")
    if isinstance(raw_entries, list) and raw_entries:
        total = len(raw_entries)

        def _fail(pos: int, err: Exception) -> LedgerError:
            """整批失败的**统一包装**。

            🔴 2026-09-08 事故（MQ-L52）：预校验那一路此前**直接抛原始消息**
            （模型只看到 `entries[5] needs both section and op`），没有经过这层
            ——于是它不知道**整批都没落**，也不知道该重发整批。live 后果：
            hermes 连撞 6 次、`rounds=6 degraded=True`、62 万 token、零写入。
            两条失败路径（预校验 / 应用）现在共用这一个出口，措辞不会再分叉。
            """
            return LedgerError(
                f"edit {pos + 1} of {total} failed ({err}); NOTHING was written "
                "— the other edits were fine. Resend the WHOLE batch with just "
                "this one fixed, or drop it and resend the rest.")

        for pos, spec in enumerate(raw_entries):
            try:
                _validate_spec(pos, spec, ledger)
            except LedgerError as e:
                raise _fail(pos, e) from e
        new = ledger
        notes: list[str] = []
        for pos, spec in enumerate(raw_entries):
            try:
                new, note = _apply_one(new, spec, updated_at=updated_at,
                                       writer=writer)
            except LedgerError as e:
                raise _fail(pos, e) from e
            notes.append(note)
        return new, "\n".join(notes)
    section = str(args.get("section", "")).strip().lower()
    op = str(args.get("op", "")).strip().lower()
    # MQ-L53：单条形态与批量**同一条放宽**——`op=remove` + `match` 时 section 可选。
    # 两条路各判各的就是两套契约，模型在哪条路上写对全靠运气（同族：MQ-A18/P7）。
    section_optional = (op == "remove" and bool(str(args.get("match", "")).strip()))
    if not op or (not section and not section_optional):
        raise LedgerError(
            "give either section+op (to change an entry), entries (several at "
            "once) or goal+goal_change_quote "
            "(only when the user changed the goal this turn)")
    return _apply_one(ledger, args, updated_at=updated_at, writer=writer)


# ── L4 · 陈旧兜底 ───────────────────────────────────────────────────────────


def staleness_marker(*, turns_since_update: int,
                     threshold_turns: int | None = None) -> str:
    """注入时附在账本正文尾部的陈旧标记；不陈旧返回 ""。英文外壳（ADR-0027 §5.1）。

    🔴 **入参是"这本账本上过了多少轮"，不是两个 turn 计数之差**（2026-08-26 改）。
    旧签名 `(ledger_turn, current_turn)` 邀请调用方去减 `turn_index`——而
    `turn_index` 是 `len(messages)//2`，**跨会话不单调**（MQ-L13 同款陷阱：
    换 scope 之后拿它算"多久以前"会得出负数）。而账本天生跨会话，
    正是这个减法最容易出错的地方。改成单一计数后，喂错时钟这件事在类型上就做不到了。

    为什么用轮次而不是墙钟：账本放着两小时没动，可能只是**没干活**，不是陈旧。
    "在这本账本上又干了 N 轮却没更新过"才是要提醒的那件事。

    只陈述事实 + 一句提示，**不替模型写内容**（三红线：BladeX 不做作者）。
    """
    th = (int(flag_number("BLADEX_LEDGER_STALE_TURNS"))
          if threshold_turns is None else threshold_turns)
    n = int(turns_since_update)
    if th <= 0 or n < th:
        return ""
    return (f"\n<sub>⚠ {n} turns on this ledger since it was last updated — "
            "Verified/Open/Next may be out of date. If you have made real "
            "progress, record it with bladex_ledger_update.</sub>")


# ── L5 · Matter 锚点 ────────────────────────────────────────────────────────


@dataclass
class AnchorDecision:
    matter_id: str = ""          # 归属目标（空 = 无锚，走 D1/DPL 兜底）
    create_matter: bool = False  # 账本无锚 ⇒ 建新 Matter 并绑定
    warn_multi: bool = False     # 第二个 Matter 出现 ⇒ 告警 + 人工复核（不自动改绑）


def anchor_decision(ledger: Ledger | None, *, proposed_matter_id: str = "") -> AnchorDecision:
    """有账本流量的归属判定（主路径；无账本时调用方直接走 D1/DPL 兜底）。

    - 账本已绑 Matter ⇒ 归属它；若提案方又给出**另一个** Matter ⇒ `warn_multi`
      （误合并=0 侧的保守：告警人工复核，绝不静默换绑/合并）。
    - 账本未绑 ⇒ `create_matter`（账本给 Matter 提供外部边界，Matter 从此
      "边界记录"而非"内容推断"——ADR-0032 §4.5 的机制本体）。
    """
    if ledger is None:
        return AnchorDecision()
    if ledger.matter_id:
        return AnchorDecision(
            matter_id=ledger.matter_id,
            warn_multi=bool(proposed_matter_id
                            and proposed_matter_id != ledger.matter_id))
    return AnchorDecision(create_matter=True)


def ledger_anchor_aliases(title: str) -> list[str]:
    """账本锚卡的**原生键**（L2/L3 匹配面）= **只有账本标题**。

    ## 🔴 为什么不派生 Goal（2026-09-02 实测回退，读数驱动）

    首版写的是 `[title] + derive_topic_keys(topic=goal, ...)`，**用错了函数**：
    该函数的契约是「entities（模型点名的实体）→ topic/**标题**分词」，`topic`
    期望标题或短语；喂一整段用户 Goal 原话进去，它只能按标点把句子切碎。实测：

        评估 v5 架构开发进展 → ['基于此前对', '架构的分析,评估当前', '对照',
                              'adr', ',给出简报。', '评估', ...]
        Qwen3.8 Flash Next  → ['调研', 'qwen3', 'flash', 'next', 'moe', '模型', ...]
        Codex MCP Ledger    → ['docs', 'planning', 'codex', 'mcp', 'ledger', '开发']

    `adr` / `qwen3` / `模型` / `codex` 在本库里遍地都是 ⇒ L2/L3 匹配面被撑开，
    一轮全量重建把 ADR-0024 的开发过程吸进「v5 架构」卡、把 qwen3.8:27b 与
    ornith-1.5 的对比测试吸进「Qwen3.8 Flash Next」卡（人工判读，2026-09-02）。

    🔴 **对照组是个偶然，值得记**：泰山啤酒卡同样带 aliases 却几乎不误吸
    （+104 条，人工判读只 2–3 条不属于）——因为它的 Goal 用**全角逗号 U+FF0C**，
    而 `_SPLIT` 只含半角 `,`，整句没被切开，实际等于「只有标题」。
    **一个正确的结果由标点决定，说明机制本身不成立。**

    ⇒ 回到与内容卡同一口径：内容卡是 `[title] + entities`，其中 entities 是
    **模型点名的实体**；锚卡没有 entities，那就只有 title。
    从 Goal 抽实体（而非分词）是另一件事，需要 NER 或蒸馏，排 0.3.0。

    ## 但 aliases 本身没被否——只是不该多给

    归属层序是 L1 manual → L2 显式 → L3 别名 → 才轮到 L4，L2/L3 匹配的就是
    「标题 + aliases」。锚卡此前一个都没种 ⇒ 匹配面为空、结构上不可能赢：
    v5 那件事的 18 条边里 17 条在锚卡已在场 11 分钟时仍被内容卡截胡
    （`docs/benchmarks/dup-card-origin-20260901.md`）。**给标题就够纠正它。**

    ## 只在建卡时算一次，绝不随成员累积

    MS-16 已经为"累积并集"付过代价（归属雪球，永安←泰山 25 条错边）。
    本函数只接受 title，**结构上不可能滚雪球**——这是它签名如此之窄的原因。
    """
    t = (title or "").strip()
    return [t] if t else []


def ledger_anchor_matter_id(ledger_id: str) -> str:
    """账本锚定的 Matter 的**确定性 id**：`m-` + sha256("ledger:"+lid)[:12]。

    id 由 ledger_id 派生 ⇒ 全量重建重放同一批账本事件必得同一 Matter id
    （重建等价；与 D1 `new_matter_id(ledger_key=…)` 同一设计）。"""
    import hashlib
    if not ledger_id:
        return ""
    return "m-" + hashlib.sha256(f"ledger:{ledger_id}".encode()).hexdigest()[:12]


# ── MQ-L38（2026-09-04）：切换候选面的相关性位 ──────────────────────────────
#
# 病例：候选列表只有 `updated_at` top-5，08-25 的「泰山啤酒」旧本排第 9，用户原话
# 逐字重提时模型面前没有"切回"这个选项，只能新建（池 27 本 ⇒ 任意时刻 22 本不可达）。
# 本段回答"以前干过这件事没有"；recency top-5 回答"最近在干什么"——两段并列，谁也不替谁。
#
# 词面匹配，零 LLM / 零向量 / 确定性：CJK 字符 2-gram ∪ 拉丁 token，按池内 idf 加权。
# 不用 e5：热路径每轮跑、池只有几十本、标题+Goal 两行字，词面就够；向量在这里是杀鸡用牛刀，
# 而且不可解释（命中为什么命中要能在日志里对账）。

#: 对话功能词——出现在几乎每句用户话里、对"是不是同一件事"零判别力。写死不做配置：
#: 它是尺子的一部分，改它 = 改判据，要走测试。（泰山旧本 Goal 是 08-26 前的逐字原话，
#: 带「小黑，帮我…一下」——不停这些词，任何以「小黑」开头的请求都会命中它。）
_CANDIDATE_STOP: frozenset[str] = frozenset({
    "小黑", "帮我", "一下", "看看", "那个", "这个", "梳理", "分析", "评估", "检查",
    "最新", "进展", "问题", "项目", "开发", "我们", "请你", "一个", "怎么", "什么",
    "还有", "继续", "调研", "情况", "一次", "再梳", "理一", "新进", "有没", "没有",
    "新的", "变化", "的最", "理泰", "看有", "现在", "目前", "后来", "样了", "怎样",
    "的那", "帮忙", "一些", "关于", "需要", "可以", "如何", "为什", "是否", "然后",
})
_CJK_RUN = None   # 惰性编译（模块导入不付 re 成本）
_LATIN_TOK = None


def candidate_units(text: str) -> set[str]:
    """文本 → 词面单元集合：CJK 2-gram ∪ 拉丁 token（≥3 字符，小写），减停表。"""
    global _CJK_RUN, _LATIN_TOK
    import re
    if _CJK_RUN is None:
        _CJK_RUN = re.compile(r"[一-鿿]+")
        _LATIN_TOK = re.compile(r"[a-z0-9][a-z0-9_\-\.]{2,}")
    s = (text or "").lower()
    out: set[str] = set(_LATIN_TOK.findall(s))
    for run in _CJK_RUN.findall(s):
        out.update(run[i:i + 2] for i in range(len(run) - 1))
    return out - _CANDIDATE_STOP


def _is_latin_unit(u: str) -> bool:
    return bool(u) and ord(u[0]) < 0x2E80


# ── MQ-A51（2026-09-08）：路径类单元 ─────────────────────────────────────────
#
# 🔴 **为什么 df 解不了这件事**（A47 的前提翻了，复核实测 09-07）：稀有锚假设
# "路径 token 是通用词、df 高"。但 31 份账本里 `df(documents)=1`、`df(hermes)=3`，
# 阈值 `max(2, N//10)`≈3 ⇒ 它们**按 df 就是稀有的**，一条没被剔掉；被剔掉的反而是
# 「报告/更新/结果/结论」这类真通用词。前提只在"池内多本谈不同项目"时成立，
# 而本池主题高度集中。⇒ 底噪的本质是**类别**（文件路径的寻址片段），不是词频。
#
# 判据与 `scripts/probe_ledger_files.py` **同一个定义点**（`looks_like_path` 从那里
# 下沉到这里，探针改 import）——两份判据迟早分叉，那是本仓反复付过学费的形状。

#: 末段必须带扩展名。`/Users/jasonye` 与 `…/site-packages` 因此出局。
#: 代价 = 无扩展名的真文件（`Makefile`）漏掉——与"宁可漏不可错"的既定口径一致：
#: 假阳性污染判据，漏报只是少一条证据。`\w` 在 str 模式下含 CJK，故中文文件名照收。
_LOOKS_LIKE_FILE = None
#: 切片分隔符：**不含 `.` 与 `-`**（它们是文件名的一部分），含中英标点与空白。
_PATH_SPLIT = None


def looks_like_path(value: object) -> bool:
    """值像不像一个**文件**路径（纯函数，判别力对照钉在单测里）。

    🔴 2026-09-08 从 `scripts/probe_ledger_files.py` 下沉到本模块（MQ-A51）：
    热路径的稀有锚要用同一把判据，而探针在 `scripts/` 下、生产不可 import。
    复制第二份 = 两份判据迟早分叉（本仓反复付过学费的形状）。探针改 import 本函数。
    """
    global _LOOKS_LIKE_FILE
    import re
    if _LOOKS_LIKE_FILE is None:
        _LOOKS_LIKE_FILE = re.compile(r"^[\w.~-]+\.[a-z0-9]{1,5}$")
    if not isinstance(value, str):
        return False
    v = value.strip()
    if not v or len(v) > 512 or "\n" in v or v.endswith("/"):
        return False
    return bool(_LOOKS_LIKE_FILE.match(v.rsplit("/", 1)[-1]))


def path_like_units(text: str) -> frozenset[str]:
    """原文里**由文件路径带进来**的词面单元（MQ-A51）。

    两步，每一步都对着一个实测形态：

    ① **哪些片段算路径**：含 `/` ∧ `looks_like_path`（末段带扩展名）∧
       **至少有一个不带扩展名的路径段**。最后那一条不是修饰——
       `detect_track.py/classic_hough.py/viz_trajectory.py`（模型用 `/` 连写的
       文件名清单，live 病例里真实存在）前两条判据全过，而它一个目录都没有。
       把它当路径 = 把 `classic_hough.py` 这种最有判别力的词当底噪剔掉，
       比不剔更糟。

    ② **片段里剔什么**：只剔**拉丁 token**（`documents` / `hermes` /
       `20260907.md`），**保留 CJK 2-gram**。理由：ASCII 段是寻址
       （目录树 + 命名习惯，同一个 agent 每轮都长一样），文件名里的中文是内容
       （`台球视觉调研` 是这件事本身）。整片剔掉会把 `台球`/`调研` 一起剔走。

    🔴 **已知代价，方向是保守的**：`packages/…/ledger_runtime.py` 这种确实有判别力
    的路径，它的拉丁 token 也不再算锚。`hits_rare` 因此是**下界**——对"引用了待办"
    这把代理尺来说，少算比多算诚实（多算 = 宣称了没发生的注意力）。
    """
    global _PATH_SPLIT
    import re
    if _PATH_SPLIT is None:
        _PATH_SPLIT = re.compile(
            r"[\s,;:()\[\]{}<>\"'`|，、；：（）【】《》「」“”‘’。…]+")
    out: set[str] = set()
    for frag in _PATH_SPLIT.split(text or ""):
        frag = frag.strip()
        if "/" not in frag or not looks_like_path(frag):
            continue
        segs = [s for s in frag.split("/") if s]
        if all(looks_like_path(s) for s in segs):
            continue    # 全是文件名 ⇒ 这是清单不是路径（见 ① 的 live 病例）
        out.update(u for u in candidate_units(frag) if _is_latin_unit(u))
    return frozenset(out)


#: 批 HO（MQ-L99）：已关账本进相关性段的时间窗——关闭时刻距本轮 ≤ 此值才列。
#: 值沿用 E4（MQ-L98）「本会话近期关闭」小节的 24 h（那一节已并入相关性段，见
#: `agency/runtime.py`）；30 天回放读数：「回到那件事」4 条全在关后 1 h 内，
#: 24 h 之外的新增条目全是「新的同类事 / 无关」（`report-batch-HO-20261015.md` §3）。
CLOSED_MATCH_WINDOW_S = 24 * 3600.0


@dataclass(frozen=True)
class LedgerMatch:
    ledger_id: str
    title: str
    score: float
    shared: int
    #: 批 HO（MQ-L99）：命中的是一本**已关**的账本。渲染方据此标 `[closed …]`；
    #: 切过去 = 重开（`_h_ledger_switch` 的既有路径）。
    closed: bool = False


def relevant_ledgers(pool: dict[str, Ledger], user_text: str, *,
                     exclude_id: str = "", top_n: int = 3,
                     min_score: float = 0.3, min_shared: int = 2,
                     require_rare_anchor: bool = True,
                     include_closed: bool = True,
                     closed_window_s: float | None = CLOSED_MATCH_WINDOW_S,
                     include_superseded: bool = False,
                     now_ms: float | None = None) -> list[LedgerMatch]:
    """按当前用户话找"像同一件事"的账本（MQ-L38 相关性位）。

    - 账本侧单元 = title + goal；idf 按池内文档频次（`log((N+1)/(df+1))`）。
    - score = Σidf(交集) / Σidf(query 单元 ∩ 池词表)——**只对池里见过的单元归一**：
      query 里池中从未出现的词（「怎么样了」）对"是不是同一件事"零信息，不该稀释分母
      （否则 "freetoken 怎么样了" 只剩 0.27）。
    - **稀有锚**（`require_rare_anchor`）：交集里至少一个单元 df ≤ max(2, 10% N)。
      这是判别力的来源：「评估 BladeX v5 架构」与三本卡共享 BladeX/架构/评估，
      归一后 score 0.47，但这些词池里人人都有——没有一个"只有这件事才有"的词，
      就不是同一件事（沙盒回放实测，去掉这条即误召）。
    - 拉丁 token 交集**按 2 计**：`freetoken` 这类唯一标识一个就够定一件事；
      纯 CJK 仍要 ≥2 个 2-gram（单个 2-gram 太容易撞）。
    - 🔴 **`closed` 账本按内容命中时照列，带 `closed=True`**（批 HO，MQ-L99，2026-10-15；
      推翻批 Z · Z3 的「closed 不列」）。Z3 的理由是「close 是有人明确说了结束，再推到模型
      面前等于替它翻案」——这对**没有内容依据**的 recency 段仍成立（那一段照旧排除 closed，
      见 `agency/runtime.py`），对**这一段**不成立：这里列出一本已关的本，依据是**当前用户
      这句话**与它的 title + goal 对得上，翻案的是用户，不是我们。病例：10-15 交接，交出方
      把「交给别人收尾」当做完关了本，接手方首句与原本 score 0.83 / shared 9，却因这一行
      排除只能新建（MQ-L98 的同会话小节对换会话、换 agent 必然无效）。门槛与 open 同一套
      （30 天回放读数定，见 `report-batch-HO-20261015.md`）。切不切归模型；切过去 = 重开。
      放开有两个收紧，**都由回放读数定**（收紧前 31 条新增里「无关」21 条）：
      ① **被取代的不列**（`superseded_by` 非空）——那件事已有明确的继任本，继任本若对得上
      会作为 open 本自己出现；回放里被取代本新增 13 条，0 条该切回它；
      ② **关闭超过 `CLOSED_MATCH_WINDOW_S`（24 h）的不列**，`closed_at` 缺失（老事件）同样不列
      ——「回到那件事」全在关后 1 h 内。收紧后 10 条：回到 3 / 新的同类事 4 / 无关 3。
      `include_closed=False` / `closed_window_s=None` / `include_superseded=True` 只给对照臂
      （探针 / 阴性对照）用；`now_ms` 缺省取当前时刻（探针回放传该轮时刻）。
      idf / 稀有锚仍按**全池**算——关掉一本不改变其它账本的得分（Z3 判据 ② 的阴性对照）。
      排除激活本。
    - 返回按 score 降序前 `top_n`；query 无有效单元 ⇒ 空（不猜）。
    """
    import math
    q = candidate_units(user_text)
    if not q or not pool:
        return []
    _now = time.time() * 1000.0 if now_ms is None else now_ms

    def closed_listable(led: Ledger) -> bool:
        if not include_closed:
            return False
        if led.superseded_by and not include_superseded:
            return False
        if closed_window_s is None:
            return True
        if not led.closed_at:
            return False
        return 0 <= _now - iso_ms(led.closed_at) <= closed_window_s * 1000.0

    docs: list[tuple[str, str, set[str]]] = []
    df: dict[str, int] = {}
    for lid, led in pool.items():
        units = candidate_units(f"{led.title or ''} {led.goal or ''}")
        docs.append((lid, led.title or lid, units))
        for u in units:
            df[u] = df.get(u, 0) + 1
    n = len(docs)
    rare_df = max(2, n // 10)

    def idf(u: str) -> float:
        return math.log((n + 1) / (df.get(u, 0) + 1))

    q_seen = {u for u in q if u in df}
    denom = sum(idf(u) for u in q_seen)
    if denom <= 0:
        return []
    out: list[LedgerMatch] = []
    for lid, title, units in docs:
        closed = pool[lid].status == STATUS_CLOSED
        if lid == exclude_id or (closed and not closed_listable(pool[lid])):
            continue
        shared_units = q_seen & units
        if not shared_units:
            continue
        if require_rare_anchor and not any(df[u] <= rare_df for u in shared_units):
            continue
        shared = sum(2 if _is_latin_unit(u) else 1 for u in shared_units)
        score = sum(idf(u) for u in shared_units) / denom
        if shared >= min_shared and score >= min_score:
            out.append(LedgerMatch(lid, title, round(score, 2), shared, closed))
    out.sort(key=lambda m: (-m.score, -m.shared, m.ledger_id))
    return out[:top_n]


def session_of_turn_key(turn_key: str) -> str:
    """Hub turn key（principal/agent/session/entry）→ session 段；不合形返回 ""。"""
    parts = (turn_key or "").split("/")
    return parts[2] if len(parts) >= 4 else ""


def agent_of_turn_key(turn_key: str) -> str:
    """Hub turn key → agent 段；不合形返回 ""。

    2026-08-26 实测：turn key 的 agent 段与 `LEDGER_SWITCH.agent_id` 逐字同形
    （`hermes:default` / `claude-code` / `codex` / `Pi` / `unknown-9eb9e3a9`），
    故可直接拼 `activation_scope` 查绑定——**这是 MQ-L14 的修法依据，
    改 agent 命名口径时必须连这里一起想**。
    """
    parts = (turn_key or "").split("/")
    return parts[1] if len(parts) >= 4 else ""


def turn_key_ts_ms(turn_key: str) -> float:
    """Hub turn key 尾段 entry_id 的 stream 毫秒；不合形返回 0.0。

    与 D1 `_d1_ms` 同一取法（created_at 会被重建压扁，stream ms 不会）。
    """
    tail = (turn_key or "").rsplit("/", 1)[-1].split("-", 1)[0]
    return float(tail) if tail.isdigit() else 0.0


def project_scope_since(timeline: dict[str, list[tuple[float, str]]]) -> float:
    """时间轴上**第一条落在非 Global scope 上的激活事件**的时刻；没有 ⇒ `inf`。

    这是"运行时开始按 (agent, project) 分键查绑定"的最早可证时刻——证据来自运行时
    自己写下的事件（switch / migrate / 带 scope 的 create），不是部署日期常量。
    """
    since = float("inf")
    for scope, seq in timeline.items():
        if seq and split_scope(scope)[1] != ACTIVATION_PROJECT_GLOBAL:
            since = min(since, seq[0][0])
    return since


def turn_active_ledger(timeline: dict[str, list[tuple[float, str]]], turn_key: str,
                       project_id: str, *, bindings: dict[str, str] | None = None,
                       turn_index: int | Callable[[], int | None] | None = None,
                       apply_done_pause: bool = True) -> str:
    """**这一轮发生时激活的是哪本账本**——全仓唯一实现点（批 Y · Y1a，MQ-L88）。

    `turn_key` 定 agent 与时刻（stream ms），`project_id` = 该轮自己的
    `identity.project_id`（调用方从 Hub 行取：`scan_meta_snapshot` 行或 `Turn.identity`，
    同一字段）；老轮次没有 ⇒ 空 ⇒ `activation_scope` 落 Global，与历史事件同键。

    🔴 **为什么必须只有一处**（原则 13 层 2）：此前两个消费方各算一遍——
    `ledger_memory` 的窗口查表（MQ-I22，批 T0 修成带 project）与重建锚层
    `memory_index._la_ledger_at`（写死 `activation_scope(agent, "")`，docstring 还写着
    "project 段在 V-F1 落地前恒 Global"——project 信号 08-29 就接通了）。
    同一个错在两个消费方上各活了一遍，第二遍藏了三周、在第一红线上贡献 59 条错边。
    守卫 `test_turn_ledger_single_impl.py`：生产代码里除本函数外，任何同时取
    turn key 身份（`agent_of_turn_key` / `turn_key_ts_ms`）又查 scope/时间轴
    （`activation_scope` / `ledger_active_at` / `ledger_activation_windows`）的函数都红。

    `bindings`（可选）= 终态绑定兜底，只在**时间轴上这个 scope 压根没有条目**时用
    （无 ts 的历史事件形态）；scope 在轴上而该刻早于第一次切换 ⇒ 空，不回落
    （否则会把"建账本之前"的历史塞进锚定 Matter，MQ-L15 反向形状）。

    🔴 **project 维的生效时刻 = 事件流里第一条带真 project 的激活事件**
    （`project_scope_since`；原则 13 层 4：值是不是来自它声称的来源）。
    `identity.project_id` 在 08-29 先于"绑定按 project 分键"接通：live 上 codex 的轮次
    04:28 起就带 `p-10ad1f61d0c2`，而运行时 09:23 仍在 `codex@Global` 上切账本，
    第一条带 project 的 switch 要到 10:12:49。那段轮次的 project_id 是真的，
    但**运行时查绑定用的不是它**——按它查会把 54 轮从它们当时真正所在的账本上摘掉
    （Y1a 修前/修后逐轮差异 `unanchored|project` 桶）。故该时刻之前一律按 Global 查。

    🔴 **完结即停锚（批 Z · Z3.2 / Y1b，拍板 (27)）**：查到的账本若在这一刻处于
    「自述完结」区间（`ledger.DoneSpan`：某次写入后 Open 与 Next 都空、Verified 非空，
    至今没有写入让它重新有待办、也没有人显式切回它）⇒ **返回空**，本轮落回 D1/DPL
    兜底——Y0「做完不关」的机判形态（W15：07:50 自述完结后 6 小时的 ADR 问答被锚进来）。
    只影响**锚定**：live 绑定不解除（那是 close 的事），热路径上不替模型做任何决定。
    判据零标定（不用"账本陈旧度"——Y0 标为未标定候选）。
    区间里**写过这本账本的用户轮**照常锚（`DoneSpan.writers`：完结写入所在的那一轮就是
    它的收尾）——判它要本轮的 `turn_index`（`Turn.identity.turn_index`，与写入时
    `tool_context` 同一公式）。可传值或零参可调用（只在命中区间且会话对得上时才取，
    调用方不必为绝大多数轮次付一次 Hub 读）；不传 ⇒ 不豁免。
    区间挂在 `switch_timeline` 的产物上（`SwitchTimeline.done_spans`）——拿到时间轴就拿到
    停锚，没有第二条取法；手拼的普通 dict 没有它 ⇒ 不停锚（= 修前口径）。

    `apply_done_pause=False` = **标签口径**：只给 DPL 准入闸（`ledger_gate` 闸②）用。停锚只摘掉
    ANCHOR，不摘掉"这一轮在谁的激活窗口里"——被停锚的轮次仍不许挂到**别的**账本锚卡上
    （否则判据 ⑤ 说的"换受害者"：摘掉标签，闸② 放行，内容相似就能把它挂进别人的锚卡），
    挂回自己这本的锚卡则交给内容（CONT「延续 ∧ 无新主体」）决定。
    """
    from .ledger import ledger_active_at

    scope = turn_activation_scope(timeline, turn_key, project_id)
    if not scope:
        return ""
    lid = ledger_active_at(timeline, scope, turn_key_ts_ms(turn_key))
    if not lid and bindings is not None and scope not in timeline:
        lid = bindings.get(scope, "")
    if (lid and apply_done_pause
            and ledger_done_paused(timeline, lid, turn_key, turn_index=turn_index)):
        return ""
    return lid


def ledger_done_paused(timeline: dict[str, list[tuple[float, str]]], ledger_id: str,
                       turn_key: str, *,
                       turn_index: int | Callable[[], int | None] | None = None) -> bool:
    """`turn_key` 这一轮发生时 `ledger_id` 是否处于完结停锚区间（且本轮不是它的写者用户轮）。

    `turn_active_ledger` 的后半段，单独给出只为仪器能数"被停锚的轮次"而不必自己再判一遍。
    """
    spans = getattr(timeline, "done_spans", None) or {}
    t = turn_key_ts_ms(turn_key)
    for sp in spans.get(ledger_id, ()):
        if not (sp.start_ms < t and (sp.end_ms is None or t <= sp.end_ms)):
            continue
        session = session_of_turn_key(turn_key)
        if session and any(s == session for s, _ in sp.writers):
            ti = turn_index() if callable(turn_index) else turn_index
            if ti is not None and (session, int(ti)) in sp.writers:
                return False
        return True
    return False


def turn_activation_scope(timeline: dict[str, list[tuple[float, str]]], turn_key: str,
                          project_id: str) -> str:
    """这一轮发生时运行时查绑定用的 **scope**；turn key 不合形 ⇒ ""。

    `turn_active_ledger` 的前半段，单独给出是为了仪器（窗口 / 波及面）能拿到"锚层查的是哪条轴"
    而不必自己再拼一遍——拼一遍就是第二个实现点（守卫同一条）。project 维生效时刻见
    `turn_active_ledger` docstring。
    """
    agent = agent_of_turn_key(turn_key)
    if not agent:
        return ""
    if project_id and turn_key_ts_ms(turn_key) < project_scope_since(timeline):
        project_id = ""
    return activation_scope(agent, project_id)


# `iso_ms` 见 bladex_core.ledger（从本模块 import 再导出，调用方路径不变）。


#: `ledger_gate` 放行。
GATE_PASS = ""
#: 闸②：这一轮明确属于**别的**账本。
GATE_OTHER_LEDGER = "other_ledger"
#: 闸①：这一轮早于账本创建时刻。
GATE_BEFORE_CREATED = "before_created"


def ledger_gate(*, turn_ledger_id: str, turn_ms: float,
                target_ledger_id: str, target_created_ms: float) -> str:
    """DPL 边落到**账本锚卡**的准入。返回 `GATE_PASS`（放行）或拒绝原因。

    ADR-0032 §4.5 的归属准入顺序 **账本标定 → 时间 → 内容聚焦** 里的前两段。
    内容相似（L2/L3/L4 的匹配面、CONT 的延续判定）只在本函数放行后才有发言权。

    两条闸都是**因果不变量**，不是相似度阈值——没有可调参数，也不该有：

    - **闸②标签**（`turn_ledger_id` 指向别的账本 ⇒ 拒）。账本是外部任务边界；
      一轮明确归属 A 账本，它的产出不该落到 B 账本的锚卡上。这条最硬、零歧义。
    - **闸①时间**（`turn_ms < target_created_ms` ⇒ 拒）。一件事的记录不可能
      早于这件事被开立。**主题相同 ≠ 同一件事**——账本的创建时刻定义了"这件事"
      从哪里开始。

    🔴 三处"缺数不当违规"（与 ADR-0012 §3.6 同族：重建不得让历史归属一次性失效）：

    1. `target_ledger_id` 空 ⇒ 目标不是锚卡，本函数完全不表态（放行）。
       普通内容卡的归属不受账本机制影响，**零回归**。
    2. `turn_ledger_id` 空（历史轮无标签 / 该刻早于第一次切换）⇒ 闸②不表态，
       落到闸①用时间判。这批正是 08-28 `Turn.ledger_face` 上线前的存量。
    3. 任一时间取不出（0.0）⇒ 闸①不表态。见 `iso_ms` 的 0.0 语义。

    标签命中时**不再过闸①**：轮次不可能标到一个还不存在的账本上，真出现了也是
    标签更权威（"账本标定优先"）。这不是漏判，是优先级。

    实证（2026-09-02，第四次全量重建后的 11 张锚卡逐边交叉）：闸①单独就拦下
    281 条 DPL 边，全部落在 Jason 人工判为脏的三张卡上（评估 v5 105 / 泰山啤酒 99
    / Qwen3.8 64），五张干净卡（Codex ledger health 100 边、苹果 44、三星 21、
    jydesignhk 17、北京天气 9）**一条不动**，ANCHOR 层违规 **0 条**。

    对照前一天被否掉的修法：按 `ledger_face` 三态翻 D5 档，同档里既有最脏的
    泰山（44/119）也有四张全干净的卡 ⇒ 零分辨力，动手会为修 44 条打掉 143 条
    正确锚定（MQ-L31）。**同样是"看起来该动手"的读数，差别在有没有做同档反例交叉。**
    """
    if not target_ledger_id:
        return GATE_PASS
    if turn_ledger_id:
        return (GATE_PASS if turn_ledger_id == target_ledger_id
                else GATE_OTHER_LEDGER)
    if turn_ms and target_created_ms and turn_ms < target_created_ms:
        return GATE_BEFORE_CREATED
    return GATE_PASS


def bind_matter(ledger: Ledger, matter_id: str, *, updated_at: str = "") -> Ledger:
    """账本 ↔ Matter 绑定（一次性；改绑必须走人工复核路径，不提供静默换绑）。"""
    if not matter_id:
        raise LedgerError("matter_id required")
    if ledger.matter_id and ledger.matter_id != matter_id:
        raise LedgerError(
            f"ledger {ledger.ledger_id} already anchored to {ledger.matter_id}; "
            "re-binding requires manual review (misattribution red line)")
    return ledger.model_copy(update={"matter_id": matter_id,
                                     "updated_at": updated_at or ledger.updated_at})
