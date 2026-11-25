"""矛盾标记：只标不删（批 BB · 步骤 2a，拍板 (56)(59)(60)(101)(102)，README ③ 的最小实现）。

本模块是纯函数 + 协议：**不 import bladex_proxy、不碰网络、不碰存储**。上游（判定模型）、Hub 事件出口、
Index 派生台账都由调用方注入——步骤 2b 才把 consolidator / 端点 / CLI 接上来。设计出处：
`docs/planning/report-batch-BB-20261103.md` §6；读数出处：`report-batch-BB-step2-0-20261114.md`。

要点（每条都有守卫，见 `test_bb_*.py`）：
- **只标不删**：本模块任何路径都不写墓碑、不改 fact、不取代；唯一的写出口是 `ContradictionSink.append_flag`
  （2b 落成 `audience_set` + `kind` 载荷的 Hub 事件）和 `JudgmentLedger` / `PendingStore`（Index 派生）。
- **载荷绝不带 `fact_id` / `audience` 键**：v0.3.0 与 HEAD 的 `audience_set` 重放只认这两个键，没有它们就是空操作，
  所以新载荷对旧版本无害。解析只在本模块一处（`parse_payload`）。
- **Hub 不存正文**：`reason` 只存 `reason_sha`，全文进 Index 派生台账。
- **重建零判定**：全量重建期（`rebuilding=True`）判定调用恒为 0；标记集合来自 Hub 事件重放（`replay`）。
- **关着零副作用**：`BLADEX_CONTRADICTION_ENABLED` 关 ⇒ 判定 0、Hub 事件 0、Index 写入 0。
- **说不清 = 不标**：只有「矛盾」才标给用户。
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from bladex_core.fact import Fact
from bladex_core.flags import flag_enabled, flag_number

logger = logging.getLogger(__name__)

# ── 常量与闭集 ────────────────────────────────────────────────────────────

KIND_FLAG = "contradiction_flag"
KIND_VERDICT = "contradiction_verdict"
#: `audience_set` 事件载荷 `kind` 的闭集（同一事件类型承载两种语义是 MS-17 的形态，故闭集 + 单一解析点）。
KINDS = frozenset({KIND_FLAG, KIND_VERDICT})
SCHEMA = 1
DETECTOR = "consistency"

VERDICT_NOT_SAME = "not_same_thing"
VERDICT_CONFIRMED = "confirmed"
#: 用户裁定闭集。
USER_VERDICTS = frozenset({VERDICT_NOT_SAME, VERDICT_CONFIRMED})
#: 裁定来源闭集——**不接受 `model`**（模型不能替用户裁定，ADR-0032 红线同族；原则 13 层 4）。
VERDICT_ORIGINS = frozenset({"cli", "dashboard"})

#: 载荷里禁止出现的键：`audience_set` 重放只认它们，带上就会真的改 fact。
FORBIDDEN_PAYLOAD_KEYS = frozenset({"fact_id", "audience"})

#: 判定提示词：以文件为唯一出处，按 sha 校验载入（步骤 2-0 的 `judge-prompt-v3.txt`）。
PROMPT_FILE = Path(__file__).with_name("prompts") / "contradiction_judge_v3.txt"
PROMPT_SHA256 = "596c18755b4093e8b07299ecd9cf19b9c8f42ba4df135bda0cfb579b7ec6f192"
PROMPT_SHA16 = PROMPT_SHA256[:16]

ENABLED_FLAG = "BLADEX_CONTRADICTION_ENABLED"
MAX_PER_FACT_FLAG = "BLADEX_CONTRADICTION_MAX_JUDGMENTS_PER_FACT"
DAILY_TOKENS_FLAG = "BLADEX_CONTRADICTION_DAILY_INPUT_TOKENS"

#: 解析失败后最多再试一次（共 2 次尝试）；仍失败就不再判，也不进否定台账。
MAX_PARSE_ATTEMPTS = 2

# 候选方案 S4a ∪ top-3 的参数（拍板 (102)；取值依据 `report-batch-BB-20261103.md` §5）。
TOP_K = 3
RELATED_WINDOW = 50
RELATED_CAP = 20
MIN_SHARED_ENTITIES = 2
COS_FLOOR = 0.75


class PromptShaMismatch(RuntimeError):
    """提示词文件与登记的 sha 不一致：判定口径变了却没重新过读数——拒绝载入。"""


# ── 小工具 ────────────────────────────────────────────────────────────────


def sha16(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def content_sha(fact: Fact) -> str:
    """裁定 / 标记绑的内容指纹：任一侧被 `fact_correct` 改了文字后，旧裁定不再覆盖。"""
    return sha16(fact.content)


def pair_key(a_id: str, b_id: str) -> str:
    """无序对的规范键（排序后 `a|b`）；Hub 事件的 `target_key`。"""
    lo, hi = sorted((a_id, b_id))
    return f"{lo}|{hi}"


def _norm(s: str | None) -> str:
    import unicodedata

    return re.sub(r"[\s（）()·:_\-]", "", unicodedata.normalize("NFKC", s or "").casefold())


def _ts(f: Fact) -> str:
    return f.t_valid.strftime("%Y-%m-%dT%H:%M:%S.%fZ") if f.t_valid else ""


# ── 提示词 ────────────────────────────────────────────────────────────────

_PROMPT_CACHE: dict[str, str] = {}


def load_prompt(path: Path | None = None) -> str:
    """载入判定提示词并按 sha 校验。文件被改而 `PROMPT_SHA256` 没更新 ⇒ `PromptShaMismatch`。"""
    p = path or PROMPT_FILE
    key = str(p)
    if key in _PROMPT_CACHE:
        return _PROMPT_CACHE[key]
    raw = p.read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != PROMPT_SHA256:
        raise PromptShaMismatch(f"{p.name}: sha256 {got[:16]} != 登记的 {PROMPT_SHA16}")
    text = raw.decode("utf-8")
    _PROMPT_CACHE[key] = text
    return text


def render_prompt(new: Fact, old: Fact, template: str | None = None) -> str:
    """A = 新 fact，B = 被比对的旧 fact（与步骤 2-0 的 `jobs_for` 同向）。"""
    return (template or load_prompt()).format(
        ta=_ts(new), sa=new.scope, suba=new.subject or "", atta=new.attribute or "", ca=new.content,
        tb=_ts(old), sb=old.scope, subb=old.subject or "", attb=old.attribute or "", cb=old.content,
    )


# ── 判定输出解析 ──────────────────────────────────────────────────────────

_VERDICTS = ("矛盾", "不矛盾", "说不清")


@dataclass(frozen=True)
class Parsed:
    verdict: str | None  # 「矛盾」「不矛盾」「说不清」；解析失败 ⇒ None
    reason: str

    @property
    def flagged(self) -> bool:
        """说不清映射为不标：只有「矛盾」才标给用户。"""
        return self.verdict == "矛盾"


def parse_verdict(text: str) -> Parsed:
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        obj = json.loads(m.group(0)) if m else {}
        verdict = obj["verdict"]
        if verdict not in _VERDICTS:
            raise ValueError(verdict)
        return Parsed(verdict, str(obj.get("reason", "")))
    except (ValueError, KeyError, TypeError):
        return Parsed(None, (text or "")[:300])


# ── 候选预筛（S4a ∪ top-3，纯函数）────────────────────────────────────────


@dataclass(frozen=True)
class Candidate:
    fact: Fact
    cos: float
    rank: int  # 1 起，在传入的邻居序列里的名次
    via: str   # "top" | "related" | "both"


def select_candidates(
    fact: Fact,
    neighbors: Sequence[tuple[Fact, float]],
    *,
    topk: int = TOP_K,
    window: int = RELATED_WINDOW,
    cap: int = RELATED_CAP,
    min_shared: int = MIN_SHARED_ENTITIES,
    cos_floor: float = COS_FLOOR,
) -> list[Candidate]:
    """S4a ∪ top-k。`neighbors` = 比新 fact 旧的、现行的、非 file_ref 的 e5 邻居（调用方已过滤）。

    - top-k：按余弦最近的 k 条，不看相关性（兜住键分裂）。
    - S4a：e5 前 `window` 条里、余弦 ≥ `cos_floor`、且「同键 ∪ 同 subject ∪ 共享 ≥ `min_shared` 个 entities」，至多 `cap` 条。
    返回 top 在前、S4a 在后的去重序列。
    """
    ranked = sorted(((f, c) for f, c in neighbors if f.id != fact.id), key=lambda x: -x[1])
    key = (_norm(fact.subject), _norm(fact.attribute))
    mine = {_norm(x) for x in fact.entities} - {""}

    def related(f: Fact) -> bool:
        subj = _norm(f.subject)
        same_key = bool(key[0] and key[1]) and (subj, _norm(f.attribute)) == key
        same_subject = bool(key[0]) and subj == key[0]
        shared = len(mine & ({_norm(x) for x in f.entities} - {""}))
        return same_key or same_subject or shared >= min_shared

    top_ids = [f.id for f, _ in ranked[:topk]]
    s4a_ids: list[str] = []
    for f, c in ranked[:window]:
        if c < cos_floor:
            break
        if related(f):
            s4a_ids.append(f.id)
            if len(s4a_ids) >= cap:
                break
    order = list(dict.fromkeys(top_ids + s4a_ids))
    by_id = {f.id: (i + 1, f, c) for i, (f, c) in enumerate(ranked)}
    out: list[Candidate] = []
    for fid in order:
        rank, f, c = by_id[fid]
        via = "both" if fid in top_ids and fid in s4a_ids else "top" if fid in top_ids else "related"
        out.append(Candidate(f, float(c), rank, via))
    return out


# ── 事件载荷（构造 + 唯一解析点）──────────────────────────────────────────


def _check_clean(payload: Mapping[str, Any]) -> None:
    bad = FORBIDDEN_PAYLOAD_KEYS & set(payload)
    if bad:
        raise ValueError(f"矛盾载荷不得带 {sorted(bad)}：audience_set 重放会据此改 fact")


def build_flag_payload(
    new: Fact, old: Fact, *, judge_model: str, reason: str, rank: int, cos: float,
) -> dict[str, Any]:
    """标记载荷。`pair` 按 id 排序，`content_sha` 与之对齐；`reason` 只落 `reason_sha`。"""
    ids = sorted([(new.id, content_sha(new)), (old.id, content_sha(old))])
    payload: dict[str, Any] = {
        "kind": KIND_FLAG, "schema": SCHEMA,
        "pair": [ids[0][0], ids[1][0]], "content_sha": [ids[0][1], ids[1][1]],
        "judge_model": judge_model, "prompt_sha": PROMPT_SHA16, "reason_sha": sha16(reason),
        "rank": int(rank), "cos": round(float(cos), 4), "detector": DETECTOR,
    }
    _check_clean(payload)
    return payload


def build_verdict_payload(
    pair: tuple[str, str], shas: tuple[str, str], *, verdict: str, origin: str, note: str = "",
) -> dict[str, Any]:
    """裁定载荷。`origin` 闭集（不接受 `model`）；`pair` / `shas` 须已按 id 排序对齐。"""
    if verdict not in USER_VERDICTS:
        raise ValueError(f"verdict {verdict!r} 不在 {sorted(USER_VERDICTS)}")
    if origin not in VERDICT_ORIGINS:
        raise ValueError(f"origin {origin!r} 不在 {sorted(VERDICT_ORIGINS)}")
    if list(pair) != sorted(pair):
        raise ValueError("pair 须按 id 排序")
    payload: dict[str, Any] = {
        "kind": KIND_VERDICT, "schema": SCHEMA, "pair": list(pair), "content_sha": list(shas),
        "verdict": verdict, "origin": origin, "note": note,
    }
    _check_clean(payload)
    return payload


@dataclass(frozen=True)
class FlagRecord:
    pair: tuple[str, str]
    shas: tuple[str, str]
    judge_model: str
    prompt_sha: str
    reason_sha: str
    rank: int
    cos: float

    @property
    def key(self) -> str:
        return f"{self.pair[0]}|{self.pair[1]}"


@dataclass(frozen=True)
class VerdictRecord:
    pair: tuple[str, str]
    shas: tuple[str, str]
    verdict: str
    origin: str
    note: str

    @property
    def key(self) -> str:
        return f"{self.pair[0]}|{self.pair[1]}"


def _pair_shas(p: Mapping[str, Any]) -> tuple[tuple[str, str], tuple[str, str]] | None:
    pair, shas = p.get("pair"), p.get("content_sha")
    if (isinstance(pair, list) and isinstance(shas, list) and len(pair) == 2 and len(shas) == 2
            and all(isinstance(x, str) and x for x in [*pair, *shas]) and pair == sorted(pair)):
        return (pair[0], pair[1]), (shas[0], shas[1])
    return None


def parse_payload(payload: Mapping[str, Any]) -> FlagRecord | VerdictRecord | None:
    """**唯一解析点**：`audience_set` 的 payload → 标记 / 裁定记录；不是本模块的载荷、或不合规 ⇒ None。

    带 `fact_id` / `audience` 的一律拒收（那是人设 audience 补标的载荷，不归这里）。
    """
    kind = payload.get("kind")
    if kind not in KINDS or payload.get("schema") != SCHEMA or FORBIDDEN_PAYLOAD_KEYS & set(payload):
        return None
    ps = _pair_shas(payload)
    if ps is None:
        return None
    if kind == KIND_FLAG:
        return FlagRecord(ps[0], ps[1], str(payload.get("judge_model", "")), str(payload.get("prompt_sha", "")),
                          str(payload.get("reason_sha", "")), int(payload.get("rank", 0) or 0),
                          float(payload.get("cos", 0.0) or 0.0))
    verdict, origin = payload.get("verdict"), payload.get("origin")
    if verdict not in USER_VERDICTS or origin not in VERDICT_ORIGINS:
        return None
    return VerdictRecord(ps[0], ps[1], str(verdict), str(origin), str(payload.get("note", "")))


# ── 重放（Hub 事件 → 状态），重建期不调模型 ────────────────────────────────


@dataclass
class ContradictionState:
    flags: dict[str, FlagRecord] = field(default_factory=dict)
    verdicts: dict[str, VerdictRecord] = field(default_factory=dict)

    def snapshot(self) -> dict[str, Any]:
        """逐字可比的快照（D8：重建前后标记集合逐字相等）。"""
        return {
            "flags": {k: vars(v) for k, v in sorted(self.flags.items())},
            "verdicts": {k: vars(v) for k, v in sorted(self.verdicts.items())},
        }


def apply_event(state: ContradictionState, event: Any) -> bool:  # noqa: ANN401 —— AdminEvent 鸭子类型，core 不 import proxy
    """把一条管理事件折进状态；返回是否被本模块认领。按 `kind` 分派的**唯一位置**。"""
    et = getattr(event.event_type, "value", event.event_type)
    if et != "audience_set":
        return False
    rec = parse_payload(event.payload)
    if isinstance(rec, FlagRecord):
        state.flags[rec.key] = rec
        return True
    if isinstance(rec, VerdictRecord):
        state.verdicts[rec.key] = rec  # 事件按序到达，最新一条为准
        return True
    return False


def replay(events: Iterable[Any]) -> ContradictionState:
    state = ContradictionState()
    for ev in events:
        apply_event(state, ev)
    return state


class SettledPairs:
    """已裁定的对（单一读点 `settled_pairs()`）。裁定绑 `content_sha`：任一侧文字变了就不再覆盖。"""

    def __init__(self, verdicts: Mapping[str, VerdictRecord]) -> None:
        self._v = dict(verdicts)

    def get(self, a_id: str, b_id: str) -> VerdictRecord | None:
        return self._v.get(pair_key(a_id, b_id))

    def covers(self, a: Fact, b: Fact) -> bool:
        rec = self.get(a.id, b.id)
        if rec is None:
            return False
        sha = {a.id: content_sha(a), b.id: content_sha(b)}
        return tuple(sha[i] for i in rec.pair) == rec.shas

    def __len__(self) -> int:
        return len(self._v)


def settled_pairs(source: ContradictionState | Iterable[Any]) -> SettledPairs:
    state = source if isinstance(source, ContradictionState) else replay(source)
    return SettledPairs(state.verdicts)


# ── 成本：JudgeBudget（单一实现点）────────────────────────────────────────


class Admission(str, Enum):
    ALLOW = "allow"
    FACT_CAP = "fact_cap"
    DAILY_CAP = "daily_cap"


class BudgetStore(Protocol):
    """日计数 / 每 fact 计数的落点。2b 落 Index meta（重启不清零）；这里只要接口。"""

    def day_input(self, day: str) -> int: ...
    def add_day_input(self, day: str, n: int) -> None: ...
    def fact_calls(self, fact_id: str) -> int: ...
    def add_fact_call(self, fact_id: str) -> None: ...


class MemoryBudgetStore:
    def __init__(self) -> None:
        self._day: dict[str, int] = {}
        self._fact: dict[str, int] = {}

    def day_input(self, day: str) -> int:
        return self._day.get(day, 0)

    def add_day_input(self, day: str, n: int) -> None:
        self._day[day] = self._day.get(day, 0) + n

    def fact_calls(self, fact_id: str) -> int:
        return self._fact.get(fact_id, 0)

    def add_fact_call(self, fact_id: str) -> None:
        self._fact[fact_id] = self._fact.get(fact_id, 0) + 1


def _utc_day() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


class JudgeBudget:
    """每条新 fact 判定次数上限 + 绝对日上限（input token，取响应 usage）。

    日上限按「已花 ≥ 上限 ⇒ 拒绝下一次」判，所以最多超出一次调用的量。默认值在 `flags.py`。
    """

    def __init__(
        self, *, per_fact: int | None = None, daily_input_tokens: int | None = None,
        store: BudgetStore | None = None, today: Callable[[], str] = _utc_day,
    ) -> None:
        self.per_fact = int(per_fact if per_fact is not None else flag_number(MAX_PER_FACT_FLAG))
        self.daily_input_tokens = int(
            daily_input_tokens if daily_input_tokens is not None else flag_number(DAILY_TOKENS_FLAG))
        self.store: BudgetStore = store or MemoryBudgetStore()
        self._today = today

    def admit(self, fact_id: str) -> Admission:
        if self.store.fact_calls(fact_id) >= self.per_fact:
            return Admission.FACT_CAP
        if self.store.day_input(self._today()) >= self.daily_input_tokens:
            return Admission.DAILY_CAP
        return Admission.ALLOW

    def charge(self, fact_id: str, input_tokens: int) -> None:
        self.store.add_fact_call(fact_id)
        self.store.add_day_input(self._today(), max(0, int(input_tokens)))


# ── 判定器编排（上游 / Hub / Index 全部注入）───────────────────────────────


class JudgeUpstreamError(RuntimeError):
    """上游调用失败（超时 / 配额 / 5xx）。适配器把供应商异常翻成它；本模块据此停判、留 pending。"""


class SinkUnavailable(RuntimeError):
    """Hub 事件出口不可达（2b：proxy 不通）。停判、留 pending。"""


@dataclass(frozen=True)
class JudgeReply:
    text: str
    input_tokens: int
    output_tokens: int
    model: str


JudgeFn = Callable[[str], JudgeReply]


class ContradictionSink(Protocol):
    """Hub 出口。2b 实现 = `POST /admin/contradictions` → `append_admin_event(AUDIENCE_SET, "", pair_key, **payload)`。"""

    def append_flag(self, pair_key: str, payload: dict[str, Any]) -> None: ...


class JudgmentLedger(Protocol):
    """Index 派生台账（`contradiction_judged/<pair>`）：全部判定（含否定）留痕，同一对不重判；全文理由在这里。"""

    def is_judged(self, pair_key: str, shas: tuple[str, str]) -> bool: ...
    def record(self, pair_key: str, shas: tuple[str, str], record: dict[str, Any]) -> None: ...
    def parse_failures(self, pair_key: str, shas: tuple[str, str]) -> int: ...
    def record_parse_failure(self, pair_key: str, shas: tuple[str, str]) -> None: ...


class PendingStore(Protocol):
    """未判完的 fact（超限 / 上游或出口不可达 / 解析失败待重试）。2b 落 Index meta。"""

    def add(self, fact_id: str) -> None: ...
    def remove(self, fact_id: str) -> None: ...
    def ids(self) -> list[str]: ...


@dataclass
class ProcessResult:
    fact_id: str
    skipped: str = ""            # "disabled" | "rebuild" | ""
    candidates: int = 0
    judged: int = 0              # 实际调用上游的次数
    flagged: list[str] = field(default_factory=list)       # 标出的 pair_key
    settled_skips: int = 0
    cached_skips: int = 0
    parse_failed: int = 0
    truncated: int = 0           # 被每条 fact 次数上限截掉的候选数
    budget_exhausted: bool = False
    pending: bool = False


class ContradictionDetector:
    """新 fact → 预筛 → 逐对判定 → 标记。**不写墓碑、不改 fact**（D5）。"""

    def __init__(
        self, *, judge: JudgeFn, budget: JudgeBudget, ledger: JudgmentLedger, sink: ContradictionSink,
        pending: PendingStore, settled: Callable[[], SettledPairs],
        enabled: Callable[[], bool] | None = None,
    ) -> None:
        self._judge, self._budget, self._ledger = judge, budget, ledger
        self._sink, self._pending, self._settled = sink, pending, settled
        self._enabled = enabled or (lambda: flag_enabled(ENABLED_FLAG))

    def process(
        self, fact: Fact, neighbors: Sequence[tuple[Fact, float]], *, rebuilding: bool = False,
    ) -> ProcessResult:
        res = ProcessResult(fact.id)
        if rebuilding:  # D8：全量重建期不调模型，标记来自 Hub 重放
            res.skipped = "rebuild"
            return res
        if not self._enabled():  # 关着：零判定、零 Hub 事件、零 Index 写入
            res.skipped = "disabled"
            return res
        cands = select_candidates(fact, neighbors)
        res.candidates = len(cands)
        settled = self._settled()
        template = load_prompt()
        stop = False
        for i, cand in enumerate(cands):
            old = cand.fact
            pk = pair_key(fact.id, old.id)
            shas = tuple(h for _, h in sorted([(fact.id, content_sha(fact)), (old.id, content_sha(old))]))
            if settled.covers(fact, old):
                res.settled_skips += 1
                continue
            if self._ledger.is_judged(pk, shas):  # type: ignore[arg-type]
                res.cached_skips += 1
                continue
            if self._ledger.parse_failures(pk, shas) >= MAX_PARSE_ATTEMPTS:  # type: ignore[arg-type]
                continue
            adm = self._budget.admit(fact.id)
            if adm is Admission.FACT_CAP:
                res.truncated = len(cands) - i
                break
            if adm is Admission.DAILY_CAP:
                res.budget_exhausted = True
                res.pending = True
                break
            try:
                reply = self._judge(render_prompt(fact, old, template))
            except JudgeUpstreamError as e:
                logger.warning("contradiction_judge_upstream_error fact=%s error=%s", fact.id, e)
                res.pending = True
                break
            self._budget.charge(fact.id, reply.input_tokens)
            res.judged += 1
            parsed = parse_verdict(reply.text)
            if parsed.verdict is None:
                logger.warning("contradiction_judge_parse_failed fact=%s pair=%s", fact.id, pk)
                self._ledger.record_parse_failure(pk, shas)  # type: ignore[arg-type]
                res.parse_failed += 1
                res.pending = True  # 下一 pass 重试一次；不进否定台账
                continue
            if parsed.flagged:
                payload = build_flag_payload(fact, old, judge_model=reply.model, reason=parsed.reason,
                                             rank=cand.rank, cos=cand.cos)
                try:
                    self._sink.append_flag(pk, payload)
                except SinkUnavailable as e:
                    logger.warning("contradiction_sink_unavailable fact=%s error=%s", fact.id, e)
                    res.pending = True
                    stop = True
                if stop:
                    break
                res.flagged.append(pk)
            self._ledger.record(pk, shas, {  # type: ignore[arg-type]
                "verdict": parsed.verdict, "reason": parsed.reason, "judge_model": reply.model,
                "prompt_sha": PROMPT_SHA16, "input_tokens": reply.input_tokens,
                "output_tokens": reply.output_tokens, "rank": cand.rank, "cos": round(cand.cos, 4),
                "ts": datetime.now(UTC).isoformat(),
            })
        if res.pending:
            self._pending.add(fact.id)
            if res.budget_exhausted:
                logger.warning("contradiction_budget_exhausted fact=%s pending=%d", fact.id, len(self._pending.ids()))
        else:
            self._pending.remove(fact.id)
        return res

    def resume(
        self, load: Callable[[str], tuple[Fact, Sequence[tuple[Fact, float]]] | None], *, rebuilding: bool = False,
    ) -> list[ProcessResult]:
        """续判 pending（次日 / 下一 pass）。`load(fact_id)` 取回 fact 与邻居；取不到（fact 已删）⇒ 摘出 pending。"""
        out: list[ProcessResult] = []
        for fid in list(self._pending.ids()):
            got = load(fid)
            if got is None:
                self._pending.remove(fid)
                continue
            r = self.process(got[0], got[1], rebuilding=rebuilding)
            out.append(r)
            if r.budget_exhausted:
                break
        return out
