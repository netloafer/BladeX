"""内循环轮次入 Hub（ADR-0032 §3.2 硬点 6；MQ-P9，2026-09-02 落地）。

# 为什么需要它

内循环（`innerloop.run_inner_loop`）每一轮都是**一次真实的上游 LLM 调用**
（`route_calling` +1），但它不是 agent 发来的请求，`server._enqueue_turn`
只在 agent 请求收尾时写一条 Turn。于是 08-30 实测 `route_calling` 82 vs
`turn_enqueued` 56：差 26 = 十次内循环 `rounds=` 之和逐条吻合，**31.7% 的上游
调用在 Hub 里不存在**——MQ-W 调用预算对账出现看不见的支出。

# 形态：aux 轮，走既有 P0 管线，不加新存储路径

每轮一条 `Turn`：
- `identity` = 触发轮身份的副本，`auxiliary=True, aux_source="inner_loop"`
  ⇒ 同一 `session_prefix`（发起方 = agent / session 直接落在 key 上）；
- `request_messages` = `[system 标记, assistant(tool_calls), tool 结果…]`
  ——首条 system 以 `INNER_LOOP_MARKER` 开头，`identity.classify_auxiliary`
  据此**从消息本身重判** aux（ADR-0018 R2：重建不信冻结字段）；
- `response_text` = 本轮 LLM 回复正文；`response_meta.usage` = 本轮 token；
- `tool_events` = 本轮调用的 bladex 工具（call + result）；
- `roundtrip` = 轮序；`ms_total` = tool_ms + llm_ms。

`"inner_loop"` **不进** `DISTILL_ONLY_AUX_RULES` ⇒ rebuild 三分为 `dropped`，
整轮不蒸馏：内循环的工具往返没有记忆价值，它是支出凭证。

# 接线（与 `SpliceLedger.new_in_turn` 同款，理由也相同）

三个内循环调用点把 `InnerLoopResult` 交给 `InnerLoopLedger.record()`（进程内
缓冲，按 session_prefix 分桶）；`server._enqueue_turn` 在主轮入库时
`drain()` 同会话缓冲并逐条 enqueue。从 agency 取而不是给 `_enqueue_turn`
加参数：它有七八个调用点，加必传参数等于让每个调用点都记得传——漏一个
就是静默不落库。

对账口径：`route_calling ≈ turn_enqueued + Σ inner_loop_turn_enqueued.rounds`。
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import structlog

from bladex_proxy.innerloop import INNER_LOOP_AUX_SOURCE, INNER_LOOP_MARKER, InnerLoopResult
from bladex_proxy.models import Identity, ResponseMeta, ToolEvent, Turn, TurnStatus

logger = structlog.get_logger()

#: 缓冲最多保留多少个会话的待入账轮次（主轮永不入库的会话——断连且 shield
#: 也没救回——不能无限占内存；超出按最旧淘汰并告警，可 grep）。
_MAX_SESSIONS = 512


@dataclass
class PendingRound:
    """一轮待入账的内循环记录（`LoopRound` 的入账视图 + 触发上下文）。"""

    round_index: int
    rounds_total: int
    tool_names: list[str] = field(default_factory=list)
    tool_ms: float = 0.0
    llm_ms: float = 0.0
    usage: dict = field(default_factory=dict)
    messages: list[dict] = field(default_factory=list)
    reply: dict = field(default_factory=dict)
    degraded: bool = False
    degrade_reason: str = ""
    recorded_ms: int = 0


class InnerLoopLedger:
    """进程内待入账缓冲：`record()` 由内循环调用点写，`drain()` 由 `_enqueue_turn` 读。"""

    def __init__(self, max_sessions: int = _MAX_SESSIONS) -> None:
        self._by_session: OrderedDict[str, list[PendingRound]] = OrderedDict()
        self._max_sessions = max_sessions

    def record(self, session_prefix: str, result: InnerLoopResult) -> int:
        """把一次内循环的全部轮次登记到该会话的待入账桶。返回登记条数。"""
        rounds = list(getattr(result, "rounds", None) or [])
        if not rounds:
            return 0
        now_ms = int(time.time() * 1000)
        bucket = self._by_session.setdefault(session_prefix, [])
        for r in rounds:
            bucket.append(PendingRound(
                round_index=int(getattr(r, "round_index", 0) or 0),
                rounds_total=len(rounds),
                tool_names=list(getattr(r, "tool_names", None) or []),
                tool_ms=float(getattr(r, "tool_ms", 0.0) or 0.0),
                llm_ms=float(getattr(r, "llm_ms", 0.0) or 0.0),
                usage=dict(getattr(r, "usage", None) or {}),
                messages=list(getattr(r, "messages", None) or []),
                reply=dict(getattr(r, "reply", None) or {}),
                degraded=bool(getattr(result, "degraded", False)),
                degrade_reason=str(getattr(result, "degrade_reason", "") or ""),
                recorded_ms=now_ms,
            ))
        self._by_session.move_to_end(session_prefix)
        while len(self._by_session) > self._max_sessions:
            evicted, lost = self._by_session.popitem(last=False)
            logger.warning("inner_loop_ledger_evicted", session_prefix=evicted,
                           rounds_lost=len(lost))
        return len(rounds)

    def drain(self, session_prefix: str) -> list[PendingRound]:
        """取走并清空该会话的待入账轮次（无则空列表）。"""
        return self._by_session.pop(session_prefix, [])

    def pending_sessions(self) -> int:
        return len(self._by_session)


def normalize_usage(usage: dict | None) -> dict[str, int]:
    """把 OpenAI 形态 `*_tokens` 或已归一的 `{prompt, completion, total}` 收成后者。"""
    if not usage:
        return {}
    out: dict[str, int] = {}
    for dst, srcs in (("prompt", ("prompt", "prompt_tokens", "input_tokens")),
                      ("completion", ("completion", "completion_tokens", "output_tokens")),
                      ("total", ("total", "total_tokens"))):
        for s in srcs:
            v = usage.get(s)
            if v is not None:
                try:
                    out[dst] = int(v)
                except (TypeError, ValueError):
                    continue
                break
    if "total" not in out and ("prompt" in out or "completion" in out):
        out["total"] = out.get("prompt", 0) + out.get("completion", 0)
    return out


def _tool_events(messages: list[dict]) -> list[ToolEvent]:
    events: list[ToolEvent] = []
    for m in messages:
        role = m.get("role")
        if role == "assistant":
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                events.append(ToolEvent(tool_name=str(fn.get("name", "") or ""),
                                        arguments=fn.get("arguments"),
                                        direction="call",
                                        tool_call_id=str(tc.get("id", "") or "")))
        elif role == "tool":
            events.append(ToolEvent(result=m.get("content"), direction="result",
                                    tool_call_id=str(m.get("tool_call_id", "") or "")))
    return events


def _reply_text(reply: dict) -> str:
    content = reply.get("content")
    if isinstance(content, list):
        return " ".join(str(p.get("text", "")) for p in content
                        if isinstance(p, dict) and p.get("type") == "text")
    return str(content or "")


def build_inner_loop_turn(identity: Identity, model: str, pending: PendingRound,
                          *, trigger_ts: Any = None) -> Turn:
    """把一轮待入账记录构造成 aux `Turn`（不入库，纯构造；入库由调用方走管线）。"""
    aux_identity = identity.model_copy(update={
        "auxiliary": True, "aux_source": INNER_LOOP_AUX_SOURCE})
    tools = ",".join(pending.tool_names) or "-"
    marker = {
        "role": "system",
        "content": (f"{INNER_LOOP_MARKER} round {pending.round_index}/{pending.rounds_total}; "
                    f"tools={tools}; trigger_ts={trigger_ts or ''}; "
                    f"degraded={str(pending.degraded).lower()}"
                    + (f"({pending.degrade_reason})" if pending.degrade_reason else "")),
    }
    reply = pending.reply or {}
    finish = "tool_calls" if reply.get("tool_calls") else ("stop" if reply else "")
    return Turn(
        identity=aux_identity, model=model,
        request_messages=[marker, *pending.messages],
        response_text=_reply_text(reply),
        tool_events=_tool_events(pending.messages),
        status=TurnStatus.OK,
        roundtrip=pending.round_index,
        auxiliary=True, aux_source=INNER_LOOP_AUX_SOURCE,
        session_id_source=identity.session_id_source,
        sensitivity=identity.sensitivity,
        response_meta=ResponseMeta(usage=normalize_usage(pending.usage),
                                   finish_reason=finish),
        ms_total=pending.tool_ms + pending.llm_ms,
    )
