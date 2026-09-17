"""ToolFace handlers（`bladex_memory_search` / `bladex_ledger_{read,update,switch}`）与账本写路径的辅助方法。

09-06 F0.1 自 `agency.py` 拆出，零行为：方法体逐字搬家，以 **mixin** 挂回 `AgencyRuntime`
（`class AgencyRuntime(ToolFaceHandlersMixin)`）。选 mixin 不选模块函数的理由：
① `inspect.getsource(AgencyRuntime._h_ledger_switch)` 这类既有断言经 MRO 仍能取到真身；
② handler 大量读写 `self.pool / self.activation / self._emit`，模块函数要把 self 显式传一遍，
   等于改每个调用点（违反"搬家不重写"）。
mixin 本身不带状态；实例属性全部由 `AgencyRuntime.__init__` 建立。
"""

from __future__ import annotations

from typing import Any

import structlog
from bladex_core.ledger import (
    ACTOR_MODEL, Ledger, LedgerEntry, LedgerError, add_entry, new_ledger, render_ledger_md
)
from bladex_core.ledger_runtime import (
    activation_scope, apply_tool_update, bind_matter, ledger_anchor_matter_id,
    update_edit_ops, update_is_add_only
)
from bladex_core.flags import flag_number
from bladex_core.sensitivity import exposure_allows

from bladex_proxy.models import AdminEventType
from bladex_proxy.agency.notes import _machine_text_mark

logger = structlog.get_logger()


#: 子任务条目的标签（父子两侧都用同一个，便于 grep 与后续统计回写率）。
SUBTASK_TAG = "[sub-task]"

#: 建新本时若本轮候选里有 ≥ 此分的项 ⇒ `ledger_created_despite_match` 告警（重复建本的直接计数）。
_DESPITE_MATCH_SCORE = 0.5

#: 用户点名账本的 title 最短长度（E0.3 / MQ-L41）：短于此的标题（"任务"、"调研"）
#: 出现在用户话里不算点名——那是日常词，不是指代。
USER_NAMED_TITLE_MIN_CHARS = 6


#: 日志里单个自由文本字段的长度上限（错误消息等）。**截断必须带显式标记**。
_LOG_TEXT_MAX = 200

#: 原则 13 的显式截断标记（与 Hub 无损纪律同一套记法）。
TRUNCATED_MARK = "…<bladex:truncated>"


def _clip(text: str, limit: int = _LOG_TEXT_MAX) -> str:
    """长文本进日志前截断，**带标记**（CLAUDE.md 刚性原则 13）。

    🔴 这一条是 09-08 复核自查出来的：首版写的是 `str(e)[:200]` —— **静默截断**，
    而 MQ-A22 那次事故（header 快照 200 字符静默截断把 `x-codex-turn-metadata`
    的 JSON 切在半截，"codex 有没有 project 字段"从存量数据永远不可判定）
    **就是同一个数字上的同一个错**。原则 13 只允许两类有损处理：凭证脱敏，
    和**带显式标记的截断**——"静默丢数据比丢数据更糟"。
    live 实证：09-08 01:38:53 那条 `error=` 恰好断在 "or drop it and r"，
    读日志的人无从判断后面还有没有内容。
    """
    s = text or ""
    return s if len(s) <= limit else s[:limit] + TRUNCATED_MARK


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


class ToolFaceHandlersMixin:
    """`AgencyRuntime` 的 ToolFace handler 面（见模块 docstring）。"""


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
                            source="model", ref=child.ledger_id, writer=writer),
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
        # 🔴 MQ-L75（2026-09-12 批 I，定性 L73 时发现）：这两条拒绝此前**零日志**——
        # 与 MQ-L52 同族（成功路径有 `applied`、LedgerError 有 `rejected`，唯独这两条
        # 只返回字符串）。没有它们，`到达 handler = 落库 + 合法拒绝` 这条恒等式对不上时
        # 分不清是静默丢失还是"没本可写"。分母与 applied/rejected 同一个。
        if not self.ledger_on:
            logger.warning("agency_ledger_update_refused", reason="module_off",
                           agent=str(context.get("agent_id", "")))
            return "Error: ledger module is disabled."
        led = self._ledger_for(context)
        if led is None:
            logger.warning("agency_ledger_update_refused", reason="no_active_ledger",
                           agent=str(context.get("agent_id", "")),
                           scope=self.scope_of(str(context.get("agent_id", "")),
                                               str(context.get("project_id", ""))))
            return "Error: no active ledger; switch/create one first."
        # rev 乐观锁（2026-08-29）：模型基于旧视图的 remove 会错删、双方互不知情
        # 的写会冲突。带 rev 且落后 ⇒ 拒写 + 返回最新正文让模型合并重试（内循环
        # 一轮内完成，不到达 agent）；不带 rev = 旧调用面，照旧不拦。
        #
        # 🔴 **只对 remove / goal 判**（F-B2，MQ-L49 ③，2026-09-07）：add 可交换
        # ——两个 agent 各加一条，谁先谁后结果都是两条都在，没有"按旧视图错删"
        # 可言，而那正是这把锁存在的理由。live 读数坐实拦错了人：8 次
        # `agency_ledger_rev_conflict` **8/8 是同一 agent 自撞**（并行 tool_calls
        # 各带同一个 rev），跨 agent 真冲突 0 —— 被拦的全是本该放行的 add，
        # 而每一次拒写都换模型再发一轮全上下文（那正是 L49 的成本曲线）。
        want = args.get("rev")
        if isinstance(want, int) and want != led.rev:
            _writer = str(context.get("agent_id", ""))
            if update_is_add_only(args):
                # 可 grep：放行不是"没发生冲突"，是"这类冲突不成立"。
                # 数它 = 这条判据放行了多少次（分子），与 rev_conflict 同一分母。
                logger.info("agency_ledger_rev_stale_add_accepted",
                            ledger=led.ledger_id, want=want, current=led.rev,
                            writer=_writer, last_writer=led.last_writer,
                            n_edits=len(update_edit_ops(args)))
            else:
                # `self_conflict`：自撞（同一 agent 的并行调用）vs 跨 agent 真冲突。
                # 两者代价不同——前者是我们把成本曲线做出来的，后者是账本该挡的。
                # 分桶写在**生产侧**，探针只读（仪器不重算生产判据）。
                logger.warning("agency_ledger_rev_conflict", ledger=led.ledger_id,
                               want=want, current=led.rev,
                               writer=_writer,
                               last_writer=led.last_writer,
                               self_conflict=bool(_writer)
                               and _writer == led.last_writer,
                               ops=",".join(update_edit_ops(args)))
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
                user_text=str(context.get("user_query", "")),
                # F-B3：条目级写者（轴 C）。与 `last_writer` 同源同值，
                # 但它落在**条目**上——一本账本上 CC 的 Verified 与 Hermes 的
                # Next 并存正是交接的常态，账本级那一个字段把它们压成了一个答案。
                writer=str(context.get("agent_id", "")))
        except LedgerError as e:
            # 🔴 2026-09-08 事故驱动补埋点（MQ-L52）：这条路径此前**零日志**——
            # hermes 在 01:13 连撞 6 次 `entries[5] needs both section and op`，
            # `rounds=6 degraded=True tokens=621603`，而 proxy 日志一片空白，
            # 全靠 Jason 在 agent 终端肉眼看见。成功路径有
            # `agency_ledger_update_applied`，失败路径没有 ⇒ "模型反复撞同一个错"
            # 这件事在读数上不存在（MQ-L17：没人看得见的机制等于没有）。
            # 分母与 applied 同一个：到达 handler 且过了 rev 门的 update 调用。
            logger.warning("agency_ledger_update_rejected",
                           ledger=led.ledger_id, agent=str(context.get("agent_id", "")),
                           n_edits=len(update_edit_ops(args)),
                           ops=",".join(update_edit_ops(args)),
                           error=_clip(str(e)))
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
        # 🔴 批量整批**一次** rev+1、**一条** LEDGER_UPDATE 事件（状态式重放：
        # payload 是整本 dump，逐条发事件既冗余又会让 rev 与事件数对不上）。
        self._emit(AdminEventType.LEDGER_UPDATE, {"ledger": new.model_dump()})
        # MQ-L49 判据的读数点：`n_edits≥2` 占比 = 批量形态被用起来了没有。
        # 分母挂"落地成功的 update 调用"（被 rev 锁拒的、报错的都不到这里）。
        _ops = update_edit_ops(args)
        logger.info("agency_ledger_update_applied", ledger=new.ledger_id,
                    n_edits=len(_ops), ops=",".join(_ops), rev=new.rev,
                    agent=str(context.get("agent_id", "")))
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
                           agent=str(context.get("agent_id", "")),
                           # 成对告警共用同一条二义（MQ-L69），同格一起补。
                           candidates_shown=self._candidates_shown)
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
                        # F-B3：建本初始条目同样带写者——建本的那个 agent 就是
                        # 它们的作者，漏填会让一本账本"开头几条没人写过"。
                        led = add_entry(led, section,
                                        LedgerEntry(text=text, source="model",
                                                    writer=agent_id),
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
                                source="model", ref=_parent, writer=agent_id),
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
                               created=led.ledger_id, agent=agent_id,
                               # 🔴 MQ-L69：没有这一格，本告警有二义 ——
                               # `False` = 这一轮**根本没给模型看候选**（工具面轮，
                               # 候选段只进用户面），那就不是"模型无视了提示"。
                               # 09-11 CC 那次告警前 60 秒零条 `agency_ledger_candidates`，
                               # 当时无法判定属于哪一种。
                               candidates_shown=self._candidates_shown)
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
