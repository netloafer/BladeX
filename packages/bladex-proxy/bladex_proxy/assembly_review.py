"""装配两级关联复核 + 判定持久化（ADR-0033 §2.4 / §2.5 / §2.7，批 U · U1.3，2026-09-23）。

## 它做什么

触发门（`inject._assembly_trigger`，§2.3）判冷且轮次数够时，对每个**已闭合、带工具结果**的轮次单元
回答一个问题：**它和"现在在做的事"还相关吗？** 相关 ⇒ `kept`；无关 ⇒ `trimmed`（U1.4 才真的裁）。

- 基准（§2.4）：`score(u) = max(sim(u, request), sim(u, goal))`。request = 当前（open）轮次内全部
  user 原文；goal = 本会话激活账本的 Goal + Core（没绑账本 ⇒ 只用 request）。
- 被比对表示 = 轮次内 user 原文 + conclusion 正文（无 conclusion 用最后一条 assistant 正文）；
  **不 embed 证据**（e5 上限 512 token，证据动辄数万字符）。user 原文先剥 agent 信封
  （`envelope.strip_envelopes`——CC 的 user 消息 p50 7 万字符里 98% 是信封，不剥就是在比信封）。
- 两级（§2.5），各自可关：

  | `REVIEW_EMBED` | `REVIEW_MODEL` | 行为 |
  |---|---|---|
  | 开 | 开 | `score ≥ HI` 相关 / `≤ LO` 无关 / 灰区一次打包交轻量模型 |
  | 开 | 关 | 只 embedding；灰区**一律保留** |
  | 关 | 开 | 全部候选过轻量模型 |
  | 关 | 关 | **只观测不裁**：`reviewed=0`，不产任何 trimmed |

- 复用（§2.7）：单元向量按 `(session, unit_key)` 永久缓存；goal 向量按 `(ledger_id, rev)`；
  判定按 `(session, unit_key, anchor_key)`。goal 锚点的判定跨轮次复用；模型判过的单元只要
  goal 没变（或仍在灰区）就不再送模型——**第二个冷轮起模型只看新进灰区的单元**（判据④）。
- 预算（§2.9）：冷轮总预算 `BLADEX_ASSEMBLY_COLD_BUDGET_MS`（默认 5000；embed 500 / 模型 4000）。
  超时 ⇒ 灰区保留 + `assembly_review_timeout`。**热轮不进这里**（200 ms 热路径预算不动）。

## 持久化

每条判定落 `AssemblyVerdictRecord`（`models.py`），随 `ReconstructionRecord.verdicts` 写穿 Memory Hub
（与 reconstruction 一起落，Hub 是真相源）。原则 13：字段无损（不截断），来源可指认——
`source`（embed / model / frozen）+ `anchor_key` + 模型名 + 首次判定所在的锚点。
进程重启后内存缓存为空：`hydrate()` 从 Hub 最近一轮的 reconstruction 读回判定。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import structlog
from bladex_core.task_unit import TaskUnit, text_of

logger = structlog.get_logger()

VERDICT_KEPT = "kept"
VERDICT_TRIMMED = "trimmed"
SOURCE_EMBED = "embed"
SOURCE_MODEL = "model"
SOURCE_FROZEN = "frozen"

#: 被比对表示的上限（字符）。e5 输入上限 512 token，中英混排约 2000 字符；超出部分 embed 时本就被截断，
#: 这里显式截是为了让"被比对的是哪一段"可复现（同一单元每次 embed 同一段文本）。
REPRESENTATION_MAX_CHARS = 2000
#: 送模型的每单元摘录上限（字符）。模型判定只需要"这件事是什么 + 结论是什么"。
MODEL_UNIT_MAX_CHARS = 600
#: 会话状态表容量（与 `cache_state._MAX_SESSIONS` 同量级）
_MAX_SESSIONS = 500
_MAX_GOAL_VECS = 256


# ── 配置 ─────────────────────────────────────────────────────────────────────


@dataclass
class ReviewConfig:
    """两级复核的开关与阈值（默认值真身 `bladex_core/flags.py`；HI / LO / 预算**待 U2 标定**）。"""

    embed: bool = True
    model: str = ""                  # 空 = 模型复核关；非空 = Router 网关上的模型名
    hi: float = 0.82
    lo: float = 0.72
    cold_budget_ms: int = 5000
    embed_budget_ms: int = 500
    model_budget_ms: int = 4000

    @classmethod
    def from_env(cls) -> ReviewConfig:
        from bladex_core.flags import flag_enabled, flag_number, flag_text

        total = int(flag_number("BLADEX_ASSEMBLY_COLD_BUDGET_MS"))
        # 子预算按 §2.9 的 500 / 4000 : 5000 比例随总预算缩放
        return cls(
            embed=flag_enabled("BLADEX_ASSEMBLY_REVIEW_EMBED"),
            model=flag_text("BLADEX_ASSEMBLY_REVIEW_MODEL").strip(),
            hi=float(flag_number("BLADEX_ASSEMBLY_SCORE_HI")),
            lo=float(flag_number("BLADEX_ASSEMBLY_SCORE_LO")),
            cold_budget_ms=total,
            embed_budget_ms=max(1, total // 10),
            model_budget_ms=max(1, total * 4 // 5),
        )


@dataclass(frozen=True)
class GoalAnchor:
    """本会话激活账本的锚点：`(ledger_id, rev)` 定版本，`text` = Goal + Core 正文。"""

    ledger_id: str
    rev: int
    text: str

    @property
    def key(self) -> str:
        return f"goal:{self.ledger_id}@{self.rev}"


# ── 判定与结果 ────────────────────────────────────────────────────────────────


@dataclass
class Verdict:
    """一个单元在某个锚点下的判定。`anchor_key` 与 `source` 是它的来源（原则 13 层 4）。"""

    unit_key: str
    verdict: str                       # kept | trimmed
    source: str                        # embed | model | frozen
    anchor_key: str
    score: float | None = None
    score_request: float | None = None
    score_goal: float | None = None
    model: str = ""
    reason: str = ""                   # hi / lo / gray_kept / model / model_missing / timeout_kept …
    decided_anchor_key: str = ""       # source=frozen 时：原判定所在锚点（可回溯到首判那一轮）

    def as_record(self) -> dict[str, Any]:
        return {
            "unit_key": self.unit_key, "verdict": self.verdict, "source": self.source,
            "anchor_key": self.anchor_key, "score": self.score,
            "score_request": self.score_request, "score_goal": self.score_goal,
            "model": self.model, "reason": self.reason,
            "decided_anchor_key": self.decided_anchor_key or self.anchor_key,
        }


@dataclass
class ReviewResult:
    verdicts: dict[str, Verdict] = field(default_factory=dict)
    candidates: int = 0
    reviewed: int = 0                  # 本轮新产生判定的单元数（不含 frozen 复用）
    reused: int = 0
    gray: int = 0
    model_calls: int = 0
    model_units: int = 0               # 送进模型的单元数（判据④ 的分子）
    timeout: str = ""                  # "" | embed | model
    mode: str = ""                     # embed+model | embed | model | observe
    elapsed_ms: float = 0.0

    @property
    def trimmed_keys(self) -> list[str]:
        return [k for k, v in self.verdicts.items() if v.verdict == VERDICT_TRIMMED]


# ── 会话内缓存（§2.7）────────────────────────────────────────────────────────


@dataclass
class _SessionReview:
    unit_vecs: dict[str, list[float]] = field(default_factory=dict)       # unit_key -> vec（永不失效）
    request_vecs: dict[str, list[float]] = field(default_factory=dict)    # request anchor_key -> vec
    verdicts: dict[tuple[str, str], Verdict] = field(default_factory=dict)  # (unit_key, anchor_key)
    model_latest: dict[str, Verdict] = field(default_factory=dict)        # unit_key -> 最近一次模型判定
    last_gray: set[str] = field(default_factory=set)
    hydrated: bool = False


class ReviewCache:
    """进程内复核缓存（LRU 会话表，与 `cache_state.registry()` 同族同上限）。"""

    def __init__(self, max_sessions: int = _MAX_SESSIONS) -> None:
        self._sessions: OrderedDict[str, _SessionReview] = OrderedDict()
        self._goal_vecs: OrderedDict[tuple[str, int], list[float]] = OrderedDict()
        self._max = max_sessions

    def session(self, session_id: str) -> _SessionReview:
        st = self._sessions.get(session_id)
        if st is None:
            st = _SessionReview()
            self._sessions[session_id] = st
            while len(self._sessions) > self._max:
                self._sessions.popitem(last=False)
        else:
            self._sessions.move_to_end(session_id)
        return st

    def goal_vec(self, goal: GoalAnchor) -> list[float] | None:
        v = self._goal_vecs.get((goal.ledger_id, goal.rev))
        if v is not None:
            self._goal_vecs.move_to_end((goal.ledger_id, goal.rev))
        return v

    def put_goal_vec(self, goal: GoalAnchor, vec: list[float]) -> None:
        self._goal_vecs[(goal.ledger_id, goal.rev)] = vec
        while len(self._goal_vecs) > _MAX_GOAL_VECS:
            self._goal_vecs.popitem(last=False)

    def hydrate(self, session_id: str, records: list[dict[str, Any]]) -> int:
        """从 Hub 读回的 `AssemblyVerdictRecord`（dict 形态）恢复判定（进程重启后首个冷轮用）。"""
        st = self.session(session_id)
        if st.hydrated:
            return 0
        st.hydrated = True
        n = 0
        for r in records or ():
            try:
                v = Verdict(unit_key=str(r["unit_key"]), verdict=str(r["verdict"]),
                            source=str(r.get("source", "")), anchor_key=str(r.get("anchor_key", "")),
                            score=r.get("score"), score_request=r.get("score_request"),
                            score_goal=r.get("score_goal"), model=str(r.get("model", "")),
                            reason=str(r.get("reason", "")),
                            decided_anchor_key=str(r.get("decided_anchor_key", "")))
            except (KeyError, TypeError):
                continue
            key_anchor = v.decided_anchor_key or v.anchor_key
            st.verdicts.setdefault((v.unit_key, key_anchor), v)
            if v.source == SOURCE_MODEL or (v.source == SOURCE_FROZEN and v.model):
                st.model_latest.setdefault(v.unit_key, v)
            n += 1
        return n

    def clear(self) -> None:
        self._sessions.clear()
        self._goal_vecs.clear()


_CACHE = ReviewCache()


def cache() -> ReviewCache:
    return _CACHE


# ── 纯函数 ────────────────────────────────────────────────────────────────────


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _strip(text: str) -> str:
    from bladex_core.envelope import strip_envelopes

    stripped, _kinds = strip_envelopes(text)
    return stripped.strip()


def unit_representation(unit: TaskUnit, messages: list[dict]) -> str:
    """被比对表示（§2.4）：轮次内 user 原文（剥信封）+ conclusion 正文；无 conclusion 用最后一条 assistant 正文。"""
    users: list[str] = []
    last_assistant = ""
    end = min(unit.msg_end, len(messages) - 1)
    for i in range(max(unit.msg_start, 0), end + 1):
        m = messages[i]
        role = m.get("role")
        if role == "user":
            t = _strip(text_of(m.get("content")))
            if t:
                users.append(t)
        elif role == "assistant":
            t = text_of(m.get("content")).strip()
            if t:
                last_assistant = t
    concl = ""
    if 0 <= unit.conclusion_index < len(messages):
        concl = text_of(messages[unit.conclusion_index].get("content")).strip()
    concl = concl or last_assistant
    return ("\n".join(users) + ("\n" + concl if concl else ""))[:REPRESENTATION_MAX_CHARS]


def request_text(units: list[TaskUnit], messages: list[dict]) -> str:
    """request 锚点原文 = 当前（open）轮次内全部 user 原文拼接（剥信封）。"""
    if not units:
        return ""
    u = units[-1]
    parts: list[str] = []
    end = min(u.msg_end, len(messages) - 1)
    for i in range(max(u.msg_start, 0), end + 1):
        if messages[i].get("role") == "user":
            t = _strip(text_of(messages[i].get("content")))
            if t:
                parts.append(t)
    return "\n".join(parts)[:REPRESENTATION_MAX_CHARS]


def request_anchor_key(text: str) -> str:
    return "req:" + hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()[:16]


def review_candidates(units: list[TaskUnit]) -> list[TaskUnit]:
    """进复核的单元：已闭合 ∧ 带工具结果（§2.4：没有 tool 结果的单元无可裁）。open 单元永不进。"""
    return [u for u in units if u.closed and u.evidence_count > 0]


def classify(score: float, hi: float, lo: float) -> str:
    """embedding 一级：`kept` / `trimmed` / `gray`。"""
    if score >= hi:
        return VERDICT_KEPT
    if score <= lo:
        return VERDICT_TRIMMED
    return "gray"


# ── 轻量模型（二级）────────────────────────────────────────────────────────────

_PROMPT_GRAY = (
    "You are auditing the context of a coding/chat agent session. Below are EARLIER task units "
    "(each = what the user asked in that turn + the agent's conclusion). An embedding check found "
    "their relevance to the CURRENT work borderline. For each unit decide whether its tool outputs "
    "may still be needed for the current request or the task goal. When unsure, answer relevant=true.\n"
    'Reply with JSON only: {"units": [{"id": "<id>", "relevant": true|false}]}'
)
_PROMPT_FULL = (
    "You are auditing the context of a coding/chat agent session. Below are ALL earlier task units "
    "(each = what the user asked in that turn + the agent's conclusion). For each unit decide whether "
    "its tool outputs may still be needed for the current request or the task goal. "
    "When unsure, answer relevant=true.\n"
    'Reply with JSON only: {"units": [{"id": "<id>", "relevant": true|false}]}'
)

Judge = Callable[[str, str, list[dict[str, str]], str], Awaitable[dict[str, bool]]]


def build_judge_prompt(request: str, goal: str, items: list[dict[str, str]], *, gray: bool) -> list[dict]:
    """两种输入形态（灰区 / 全量）各一份 system prompt（§2.5）；items = [{id, text}]。"""
    body = {"current_request": request, "task_goal": goal,
            "units": [{"id": it["id"], "text": it["text"][:MODEL_UNIT_MAX_CHARS]} for it in items]}
    return [{"role": "system", "content": _PROMPT_GRAY if gray else _PROMPT_FULL},
            {"role": "user", "content": json.dumps(body, ensure_ascii=False)}]


def parse_judge_reply(text: str) -> dict[str, bool]:
    """模型回复 → `{id: relevant}`。解析不了的 / 缺的单元由调用方按保留处理（保守方向）。"""
    s = (text or "").strip()
    if s.startswith("```"):
        s = s.strip("`")
        s = s[s.find("{"):] if "{" in s else s
    try:
        start, end = s.index("{"), s.rindex("}") + 1
        data = json.loads(s[start:end])
    except (ValueError, json.JSONDecodeError):
        return {}
    out: dict[str, bool] = {}
    for it in data.get("units", []) if isinstance(data, dict) else []:
        if isinstance(it, dict) and "id" in it and isinstance(it.get("relevant"), bool):
            out[str(it["id"])] = it["relevant"]
    return out


async def router_judge(model: str, request: str, items: list[dict[str, str]], goal: str,
                       *, gray: bool = True) -> dict[str, bool]:
    """经 Router 网关（`router_sdk.acompletion`）一次打包判定；返回 `{unit_key: relevant}`。"""
    from bladex_proxy import router_sdk

    kwargs: dict[str, Any] = {"model": model,
                              "messages": build_judge_prompt(request, goal, items, gray=gray),
                              "temperature": 0, "max_tokens": 32 + 24 * len(items), "stream": False}
    try:
        from bladex_proxy.routing_config import load_routing_config

        for m in getattr(load_routing_config(), "models", []):
            if m.name == model:
                if getattr(m, "api_base", ""):
                    kwargs["api_base"] = m.api_base
                if getattr(m, "api_key", ""):
                    kwargs["api_key"] = m.api_key
                break
    except Exception as e:  # noqa: BLE001 —— 路由配置读不到就用 SDK 默认凭证
        logger.debug("assembly_review_routing_config_unavailable", error=str(e))
    resp = await router_sdk.acompletion(**kwargs)
    return parse_judge_reply(resp.choices[0].message.content or "")


# ── 主流程 ────────────────────────────────────────────────────────────────────


async def _run_blocking(fn: Callable[[], Any], timeout_s: float) -> Any:  # noqa: ANN401
    loop = asyncio.get_running_loop()
    return await asyncio.wait_for(loop.run_in_executor(None, fn), timeout=timeout_s)


async def review_units(
    messages: list[dict],
    units: list[TaskUnit],
    *,
    session_id: str,
    config: ReviewConfig,
    goal: GoalAnchor | None = None,
    embedder: Any = None,  # noqa: ANN401  MemoryIndex 形态：embed_passage(list) / embed_query_vec(str)
    judge: Callable[..., Awaitable[dict[str, bool]]] | None = None,
    review_cache: ReviewCache | None = None,
) -> ReviewResult:
    """一次冷轮复核（纯判定，不改消息）。返回每个候选单元的判定（含复用）。"""
    t0 = time.perf_counter()
    rc = review_cache or cache()
    res = ReviewResult()
    cands = review_candidates(units)
    res.candidates = len(cands)
    use_embed = bool(config.embed) and embedder is not None
    use_model = bool(config.model)
    res.mode = ("embed+model" if use_embed and use_model else "embed" if use_embed
                else "model" if use_model else "observe")
    if not cands or res.mode == "observe":
        res.elapsed_ms = (time.perf_counter() - t0) * 1000
        return res

    st = rc.session(session_id)
    req_text = request_text(units, messages)
    req_key = request_anchor_key(req_text)
    anchor_key = req_key + (("|" + goal.key) if goal else "")

    # ── 复用（§2.7）：同锚点判过 / goal 锚点判过的 keep / goal 未变时模型判过 ──
    fresh: list[TaskUnit] = []
    for u in cands:
        v = st.verdicts.get((u.unit_key, anchor_key))
        if v is None and goal is not None:
            gv = st.verdicts.get((u.unit_key, goal.key))
            if gv is not None:
                v = gv
            else:
                mv = st.model_latest.get(u.unit_key)
                if mv is not None and goal.key in (mv.decided_anchor_key or mv.anchor_key):
                    v = mv
        if v is not None:
            res.verdicts[u.unit_key] = Verdict(
                unit_key=u.unit_key, verdict=v.verdict, source=SOURCE_FROZEN, anchor_key=anchor_key,
                score=v.score, score_request=v.score_request, score_goal=v.score_goal,
                model=v.model, reason=v.reason, decided_anchor_key=v.decided_anchor_key or v.anchor_key)
            res.reused += 1
        else:
            fresh.append(u)

    gray: list[tuple[TaskUnit, float | None, float | None, float | None]] = []
    if use_embed and fresh:
        scored = await _embed_scores(fresh, messages, st, rc, req_text, req_key, goal, embedder, config)
        if scored is None:
            res.timeout = "embed"
            logger.warning("assembly_review_timeout", stage="embed", session_id=session_id[:16],
                           budget_ms=config.embed_budget_ms, units=len(fresh))
            gray = [(u, None, None, None) for u in fresh]
        else:
            for u in fresh:
                s_req, s_goal = scored[u.unit_key]
                score = max(s_req, s_goal) if s_goal is not None else s_req
                cls = classify(score, config.hi, config.lo)
                if cls == "gray":
                    gray.append((u, score, s_req, s_goal))
                    continue
                v = Verdict(unit_key=u.unit_key, verdict=cls, source=SOURCE_EMBED,
                            anchor_key=(goal.key if (cls == VERDICT_KEPT and goal is not None
                                                     and s_goal is not None and s_goal >= config.hi)
                                        else anchor_key),
                            score=round(score, 4), score_request=round(s_req, 4),
                            score_goal=None if s_goal is None else round(s_goal, 4),
                            reason="hi" if cls == VERDICT_KEPT else "lo")
                st.verdicts[(u.unit_key, v.anchor_key)] = v
                res.verdicts[u.unit_key] = v
                res.reviewed += 1
    elif fresh:
        gray = [(u, None, None, None) for u in fresh]   # embed 关：全部候选交模型（或保留）
    res.gray = len(gray)

    # ── 二级：灰区（embed 开）/ 全量（embed 关）交轻量模型 ──
    # 送模型 = 新进灰区 ∧ 未按当前 goal 锚点判过（§2.7）。有 goal 时"按 goal 判过"已在上面的复用里
    # 处理（goal rev 变 ⇒ 锚点变 ⇒ 重判）；没有 goal 时 request 锚点每轮都变，"上一冷轮已在灰区且
    # 模型判过"的单元不算新进灰区 ⇒ 沿用那次模型判定。
    to_model: list[tuple[TaskUnit, float | None, float | None, float | None]] = []
    for g in gray:
        u = g[0]
        prev = st.model_latest.get(u.unit_key)
        if use_model and goal is None and prev is not None and u.unit_key in st.last_gray:
            _keep(res, st, g, anchor_key, SOURCE_FROZEN, prev.verdict, prev.reason, prev.model,
                  decided=prev.decided_anchor_key or prev.anchor_key, reviewed=False)
            res.reused += 1
        elif use_model:
            to_model.append(g)
        else:
            _keep(res, st, g, anchor_key, SOURCE_EMBED, VERDICT_KEPT, "gray_kept", "")
    st.last_gray = {g[0].unit_key for g in gray}

    if to_model:
        items = [{"id": u.unit_key, "text": unit_representation(u, messages)} for u, *_ in to_model]
        goal_text = goal.text if goal else ""
        budget_s = max(0.001, min(config.model_budget_ms,
                                  config.cold_budget_ms - (time.perf_counter() - t0) * 1000) / 1000)
        judged: dict[str, bool] | None
        res.model_calls += 1
        res.model_units += len(items)
        try:
            j = judge or (lambda r, it, gtext, gray_mode: router_judge(config.model, r, it, gtext,
                                                                      gray=gray_mode))
            judged = await asyncio.wait_for(j(req_text, items, goal_text, use_embed), timeout=budget_s)
        except TimeoutError:
            judged = None
            res.timeout = res.timeout or "model"
            logger.warning("assembly_review_timeout", stage="model", session_id=session_id[:16],
                           budget_ms=round(budget_s * 1000), units=len(items), model=config.model)
        except Exception as e:  # noqa: BLE001 —— 复核失败绝不打死请求；灰区保留
            judged = None
            from bladex_proxy.router_sdk import error_text

            logger.warning("assembly_review_model_failed", error=error_text(e), model=config.model,
                           units=len(items))
        for g in to_model:
            u = g[0]
            if judged is None:
                _keep(res, st, g, anchor_key, SOURCE_MODEL, VERDICT_KEPT,
                      "timeout_kept" if res.timeout == "model" else "model_failed_kept", config.model,
                      remember=False)
            elif u.unit_key not in judged:
                _keep(res, st, g, anchor_key, SOURCE_MODEL, VERDICT_KEPT, "model_missing", config.model)
            else:
                verdict = VERDICT_KEPT if judged[u.unit_key] else VERDICT_TRIMMED
                _keep(res, st, g, anchor_key, SOURCE_MODEL, verdict, "model", config.model)

    res.elapsed_ms = (time.perf_counter() - t0) * 1000
    return res


def _keep(res: ReviewResult, st: _SessionReview,
          g: tuple[TaskUnit, float | None, float | None, float | None], anchor_key: str,
          source: str, verdict: str, reason: str, model: str, *, decided: str = "",
          reviewed: bool = True, remember: bool = True) -> None:
    u, score, s_req, s_goal = g
    v = Verdict(unit_key=u.unit_key, verdict=verdict, source=source, anchor_key=anchor_key,
                score=None if score is None else round(score, 4),
                score_request=None if s_req is None else round(s_req, 4),
                score_goal=None if s_goal is None else round(s_goal, 4),
                model=model, reason=reason, decided_anchor_key=decided)
    res.verdicts[u.unit_key] = v
    if reviewed:
        res.reviewed += 1
    if remember and source != SOURCE_FROZEN:
        st.verdicts[(u.unit_key, anchor_key)] = v
        if source == SOURCE_MODEL:
            st.model_latest[u.unit_key] = v


async def _embed_scores(
    fresh: list[TaskUnit], messages: list[dict], st: _SessionReview, rc: ReviewCache,
    req_text: str, req_key: str, goal: GoalAnchor | None, embedder: Any,  # noqa: ANN401
    config: ReviewConfig,
) -> dict[str, tuple[float, float | None]] | None:
    """单元 / request / goal 三类向量（缺的才算，embed 子预算内），返回 `{unit_key: (s_req, s_goal)}`。

    超时或 embedder 失败 ⇒ None（调用方把 fresh 全部当灰区）。
    """
    todo = [u for u in fresh if u.unit_key not in st.unit_vecs]
    need_req = req_key not in st.request_vecs
    need_goal = goal is not None and rc.goal_vec(goal) is None

    def _work() -> tuple[list[list[float]], list[float] | None, list[float] | None]:
        vecs = embedder.embed_passage([unit_representation(u, messages) for u in todo]) if todo else []
        rv = embedder.embed_query_vec(req_text) if need_req else None
        gv = embedder.embed_query_vec(goal.text) if (need_goal and goal is not None) else None
        return vecs, rv, gv

    if todo or need_req or need_goal:
        try:
            vecs, rv, gv = await _run_blocking(_work, config.embed_budget_ms / 1000)
        except TimeoutError:
            return None
        except Exception as e:  # noqa: BLE001
            logger.warning("assembly_review_embed_failed", error=str(e))
            return None
        for u, v in zip(todo, vecs, strict=False):
            st.unit_vecs[u.unit_key] = list(v)
        if need_req:
            if rv is None:
                return None
            st.request_vecs = {req_key: list(rv)}      # 只留当前轮次的 request 向量
        if need_goal and goal is not None and gv is not None:
            rc.put_goal_vec(goal, list(gv))
    req_vec = st.request_vecs.get(req_key) or []
    goal_vec = rc.goal_vec(goal) if goal is not None else None
    out: dict[str, tuple[float, float | None]] = {}
    for u in fresh:
        uv = st.unit_vecs.get(u.unit_key, [])
        out[u.unit_key] = (_cosine(uv, req_vec), None if goal_vec is None else _cosine(uv, goal_vec))
    return out
