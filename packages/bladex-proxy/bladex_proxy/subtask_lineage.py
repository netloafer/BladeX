"""批 BA2（MQ-A94）：子任务归属——**一处规则、一个实现点**，不按 agent 写两套判据（刚性原则 10）。

三条规则（结算稿 §2.1，拍板 (28)(32)-a）：

R1 **子会话项目继承**：子任务自报 cwd 优先 > 唯一母候选的项目 > Global。
   母候选 = 子任务请求的 user 消息**逐字包含**母会话 spawn tool_call 的参数原文（prompt / goal /
   message，≥ `MIN_SPAWN_TEXT` 字符）。包含关系是内容信号，不是时序推断（取证稿 §5.1）。
   候选不唯一 ⇒ **不选**（N6，Jason 10-28：挂到任一母会话都算误接，碰巧挂对也是猜的）。
R2 **续段项目粘滞**：本轮完全没有 cwd 声明 ∧ 同一会话此前有项目 ⇒ 沿用最近一次声明的项目。
   与批 E 否决的「会话级粘滞」差在两处：会话 id 已由 BA1 判定锚定；请求只要带了 cwd 声明
   （哪怕不可用 / 非项目目录）就不粘——用户换目录时不留旧键。`tw:` 时间窗桶不是一段对话，不参与。
R3 **子本自动挂母本**：`parent_ledger_id` = 派出时的激活本，口径 = **响应内 tool_call 顺序上、
   spawn tool_call 位置处的激活本**（拍板 (32)-a，S9 先派后切 / S10 先切后派）。派出时无激活本
   ⇒ 不挂（S6）。模型显式给了 parent 时不覆盖。

状态都在进程内，重启由 `rebuild_lineage` 从 Hub 轮次恢复（新字段随 Turn 入库）。
账本系统本身按 (agent, project) 作用域、不分 user，这里的会话表同口径；母候选另按 user 过滤。
"""

from __future__ import annotations

import json
import time
from collections import OrderedDict, deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import structlog

logger = structlog.get_logger()

#: 派生子任务的工具 → 参数里承载任务原文的字段（全库普查见结算稿 §3）。
#: 新 agent 的工具不在表里 ⇒ 子任务找不到母候选 ⇒ `subtask_mother_none`，安全方向（不挂）。
SPAWN_TOOLS: frozenset[str] = frozenset({
    "subagent", "subagent_fork", "delegate_task", "Agent", "Task", "ns-multi_agent_v1-spawn_agent"})
_SPAWN_TEXT_FIELDS = ("prompt", "goal", "message")
#: 太短的参数（"go on"）谈不上逐字包含的证据。
MIN_SPAWN_TEXT = 30
#: 母候选有效期：晚于这个时长还没发出首轮的子任务不认（宁不挂）。
SPAWN_TTL_S = 24 * 3600.0
_MAX_SPAWNS = 512
_MAX_SESSIONS = 4096
_LEDGER_STATE_TOOLS = frozenset({"bladex_ledger_switch", "bladex_ledger_close"})


@dataclass(frozen=True)
class SpawnRecord:
    user_id: str
    agent_id: str
    session_id: str
    project_id: str
    project_name: str
    project_source: str
    project_root: str
    ledger_id: str
    text: str
    ts: float


@dataclass
class SessionLineage:
    project_id: str = ""
    project_name: str = ""
    project_source: str = ""
    project_root: str = ""
    parent_ledger_id: str = ""
    mother_checked: bool = False
    mother: SpawnRecord | None = None


@dataclass(frozen=True)
class LedgerOp:
    """一次 switch / close 工具调用前后的激活本（失败 / 幂等也记：要和 tool_events 逐条对位）。"""

    before: str
    after: str


def _base_agent(agent_id: str) -> str:
    return (agent_id or "").split(":")[0]


def _get(obj: Any, name: str) -> Any:
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def _call_args(te: Any) -> dict:
    raw = _get(te, "arguments")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return {}
    return raw if isinstance(raw, dict) else {}


def spawn_texts(args: dict) -> list[str]:
    """spawn 参数里的任务原文。hermes 批量形态 `tasks: [{goal}]` 同样算。"""
    out = [str(args[f]) for f in _SPAWN_TEXT_FIELDS if isinstance(args.get(f), str)]
    for item in args.get("tasks") or []:
        if isinstance(item, dict):
            out.extend(str(item[f]) for f in _SPAWN_TEXT_FIELDS if isinstance(item.get(f), str))
    return [t for t in out if len(t) >= MIN_SPAWN_TEXT]


def active_at_spawn(ops: list[LedgerOp], ledger_calls_after_spawn: int, active_now: str) -> str:
    """spawn tool_call 位置上的激活本（拍板 (32)-a）。

    `ops` = 本轮按执行序的 switch / close 调用；`ledger_calls_after_spawn` = 同一响应里排在 spawn
    **之后**的这类调用数。前面的（含内循环里更早响应的）都已生效，后面的都还没有。
    """
    if not ops:
        return active_now
    k = len(ops) - ledger_calls_after_spawn
    return ops[k - 1].after if k >= 1 else ops[0].before


def _user_texts(messages: list[dict]) -> list[str]:
    from bladex_proxy.identity import _msg_text  # noqa: PLC0415 —— 单一解析点

    return [_msg_text(m) for m in messages or [] if m.get("role") == "user"]


class SubtaskLineage:
    def __init__(self) -> None:
        self._spawns: deque[SpawnRecord] = deque(maxlen=_MAX_SPAWNS)
        self._sessions: OrderedDict[tuple[str, str], SessionLineage] = OrderedDict()

    def clear(self) -> None:
        self._spawns.clear()
        self._sessions.clear()

    def _state(self, agent_id: str, session_id: str) -> SessionLineage:
        key = (agent_id, session_id)
        st = self._sessions.get(key)
        if st is None:
            st = self._sessions[key] = SessionLineage()
            if len(self._sessions) > _MAX_SESSIONS:
                self._sessions.popitem(last=False)
        else:
            self._sessions.move_to_end(key)
        return st

    # ── 母会话一侧 ──
    def note_spawns(self, identity: Any, tool_events: Iterable[Any], *,
                    ledger_ops: list[LedgerOp], active_now: str,
                    now: float | None = None) -> list[dict]:
        """本轮的 spawn tool_call 登记为母候选；返回入 Turn 的 `spawn_dispatches`。"""
        calls = [te for te in tool_events or [] if (_get(te, "direction") or "call") == "call"]
        out: list[dict] = []
        for i, te in enumerate(calls):
            if str(_get(te, "tool_name") or "") not in SPAWN_TOOLS:
                continue
            after = sum(1 for c in calls[i + 1:]
                        if str(_get(c, "tool_name") or "") in _LEDGER_STATE_TOOLS)
            ledger_id = active_at_spawn(ledger_ops, after, active_now)
            texts = spawn_texts(_call_args(te))
            out.append({"tool_call_id": str(_get(te, "tool_call_id") or ""),
                        "tool_name": str(_get(te, "tool_name") or ""),
                        "ledger_id": ledger_id})
            self._register(identity, texts, ledger_id, time.time() if now is None else now)
            logger.info("ledger_spawn_recorded", agent=identity.agent_id,
                        session=identity.session_id, tool=_get(te, "tool_name"),
                        ledger=ledger_id, calls_after=after, ops=len(ledger_ops),
                        text_chars=[len(t) for t in texts])
        return out

    def _register(self, identity: Any, texts: list[str], ledger_id: str, ts: float) -> None:
        for text in texts:
            self._spawns.append(SpawnRecord(
                user_id=identity.user_id, agent_id=identity.agent_id,
                session_id=identity.session_id, project_id=identity.project_id or "",
                project_name=identity.project_name or "",
                project_source=identity.project_source or "",
                project_root=identity.project_root or "",
                ledger_id=ledger_id, text=text, ts=ts))

    def find_mothers(self, identity: Any, messages: list[dict],
                     now: float | None = None) -> list[SpawnRecord]:
        """逐字包含判据；按 (母会话, 激活本, 项目) 去重。**同一母会话同一本派两次相同任务仍算一个**。"""
        now = time.time() if now is None else now
        base = _base_agent(identity.agent_id)
        texts = _user_texts(messages)
        hits: dict[tuple[str, str, str, str], SpawnRecord] = {}
        for rec in self._spawns:
            if (rec.user_id != identity.user_id or _base_agent(rec.agent_id) != base
                    or now - rec.ts > SPAWN_TTL_S or now < rec.ts
                    or (rec.agent_id, rec.session_id) == (identity.agent_id, identity.session_id)):
                continue
            if any(rec.text in t for t in texts):
                hits[(rec.agent_id, rec.session_id, rec.ledger_id, rec.project_id)] = rec
        return list(hits.values())

    # ── 子会话 / 续段一侧 ──
    def apply(self, identity: Any, project: Any, messages: list[dict], *, is_subtask: bool,
              now: float | None = None) -> None:
        """每轮一次（`_prepare_round`，在项目识别之后、用项目键之前）：就地补 `identity` 的项目与母本。"""
        st = self._state(identity.agent_id, identity.session_id)
        if is_subtask and not st.mother_checked:
            st.mother_checked = True
            self._resolve_mother(identity, st, messages, now)
        identity.parent_ledger_id = st.parent_ledger_id

        if not identity.project_id and not getattr(project, "cwd_declared", False):
            self._inherit(identity, st)
        # `tw:` 是 5 分钟时间窗桶，不是一段对话（回放集 N4）——既不记也不粘。
        if identity.project_id and not identity.session_id.startswith("tw:"):
            st.project_id = identity.project_id
            st.project_name = identity.project_name
            st.project_source = identity.project_source
            st.project_root = identity.project_root

    def _resolve_mother(self, identity: Any, st: SessionLineage, messages: list[dict],
                        now: float | None) -> None:
        mothers = self.find_mothers(identity, messages, now)
        if len(mothers) > 1:
            logger.info("subtask_mother_ambiguous", agent=identity.agent_id,
                        session=identity.session_id, candidates=len(mothers),
                        mothers=sorted(f"{m.agent_id}/{m.session_id}" for m in mothers)[:5])
            return
        if not mothers:
            logger.info("subtask_mother_none", agent=identity.agent_id, session=identity.session_id)
            return
        m = mothers[0]
        st.mother = m
        st.parent_ledger_id = m.ledger_id
        logger.info("subtask_mother_matched", rule="contains_spawn_text", agent=identity.agent_id,
                    session=identity.session_id, mother_agent=m.agent_id, mother_session=m.session_id,
                    parent_ledger=m.ledger_id, mother_project=m.project_id, text_chars=len(m.text))

    def _inherit(self, identity: Any, st: SessionLineage) -> None:
        if st.project_id:
            rule = "session_sticky"
            pid, name, source, root = (st.project_id, st.project_name,
                                       st.project_source, st.project_root)
        elif st.mother is not None and st.mother.project_id:
            rule = "mother_unique"
            m = st.mother
            pid, name, source, root = m.project_id, m.project_name, m.project_source, m.project_root
        else:
            return
        identity.project_id = pid
        identity.project_name = name
        identity.project_source = source
        identity.project_root = root
        identity.project_inherited = rule
        logger.info("subtask_project_inherited", rule=rule, agent=identity.agent_id,
                    session=identity.session_id, project=pid)

    def parent_for(self, agent_id: str, session_id: str) -> str:
        st = self._sessions.get((agent_id, session_id))
        return st.parent_ledger_id if st is not None else ""

    # ── 重启恢复 ──
    def rebuild_lineage(self, records: Iterable[Any]) -> int:
        """`SessionReplayRecord` 流（按时序）→ 恢复会话表与母候选。**不覆盖已有状态**：重放期间
        新来的请求已经建立的条目更新。返回恢复的母候选数。"""
        n = 0
        fresh: dict[tuple[str, str], SessionLineage] = {}
        spawns: list[SpawnRecord] = []
        for r in sorted(records, key=lambda x: x.ts):
            if r.parent_ledger_id or (r.project_id and not r.session_id.startswith("tw:")):
                st = fresh.setdefault((r.agent_id, r.session_id), SessionLineage())
                if r.parent_ledger_id:
                    st.parent_ledger_id = r.parent_ledger_id
                    st.mother_checked = True
                if r.project_id and not r.session_id.startswith("tw:"):
                    st.project_id, st.project_name = r.project_id, r.project_name
                    st.project_source, st.project_root = r.project_source, r.project_root
            by_id = {d.get("tool_call_id"): d for d in r.spawn_dispatches or []}
            for te in r.tool_events or []:
                d = by_id.get(str(_get(te, "tool_call_id") or ""))
                if d is None or (_get(te, "direction") or "call") != "call":
                    continue
                for text in spawn_texts(_call_args(te)):
                    spawns.append(SpawnRecord(
                        user_id=r.user_id, agent_id=r.agent_id, session_id=r.session_id,
                        project_id=r.project_id, project_name=r.project_name,
                        project_source=r.project_source, project_root=r.project_root,
                        ledger_id=str(d.get("ledger_id") or ""), text=text, ts=r.ts))
                    n += 1
        for key, st in fresh.items():
            self._sessions.setdefault(key, st)
        live = list(self._spawns)
        self._spawns.clear()
        self._spawns.extend(spawns[-_MAX_SPAWNS:])
        self._spawns.extend(live)
        return n


#: 进程级单例（同 `identity._session_cache`）。
lineage = SubtaskLineage()
