"""行为画像口径（Flash 重定位 F1a，方案 `flash-repositioning-plan-20260920.md` §1 / §2 S4·S5）。

本模块只放**纯函数口径**：输入 Memory Hub `scan_meta` 那一档的 turn 元数据
（key / ts，不要全量 `get()`——补遗 G1），输出确定性聚合结果。**不接线**
（daemon 写 `behavior.json` 是 F2 的事），**不读墙上时钟**（`now` 一律入参），
同输入跑两次逐字一致（守卫 `tests/test_behavior.py`）。

两个口径：

- :func:`last_seen_by_agent` —— S4，`last_seen` 生产者。今天 `USER.md` 15/15 是
  "最近 ?"：值本该来自 Index profile 的 `last_seen` 字段，而那条生产链断了
  （刚性原则 13 层 3 的教科书形态）。新口径直接从 Hub turn 元数据聚合。
- :func:`dedupe_sessions` —— S5，会话去重**过渡口径**（详见该函数 docstring）。

调用契约（给 F2 接线方）：``now`` 由调用方给一次 tz-aware 的 UTC 时刻；agent 准入
（`browsing_admittance` 判据① + bench 名单，拍板 (3)/G4）由调用方折成 ``eligible``
谓词传入——本模块只做两条硬排除：`unknown-*` 指纹桶与不合形状的 key。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from statistics import median

from pydantic import BaseModel, ConfigDict, Field

#: 名册准入同口径（`flash_daemon.registry_agents_source._admit`）：`unknown` / `unknown-<fp>`
#: 是待认领指纹桶，不进任何行为画像输出。
UNKNOWN_AGENT_PREFIX = "unknown"

#: session_id 前缀 ⇒ `session_id_source`（`identity._infer_session_from_prefix`：
#: `fp:<sha12>` = 前缀指纹；`tw:<user>:<agent>:<window>` = 时间窗兜底；其余为显式 / 尾部接续沿用）。
SESSION_SOURCE_FP = "fp"
SESSION_SOURCE_TW = "tw"
SESSION_SOURCE_EXPLICIT = "explicit"

#: 时间窗兜底的窗口宽度（秒）。**与 `bladex_proxy.identity._TIME_WINDOW_SECONDS` 同值**；
#: core 不反向 import proxy，两处由测试对账（`test_behavior.py::test_tw_window_matches_identity`）。
TW_WINDOW_S = 300


class TurnMeta(BaseModel):
    """一轮的元数据切片——`scan_meta` 一档能给的全部：key 三段身份 + ts。

    `key` 结构 = ``principal/agent/session/<entry_id>``（ADR-0026 §2.3 / U2）。
    """

    model_config = ConfigDict(frozen=True)

    principal: str
    agent_id: str
    session_id: str
    ts: datetime
    #: F2 补的元数据档字段（`MemoryHub.scan_meta_snapshot` 给；老 `scan_meta` 调用方
    #: 不带 ⇒ 全默认）。都是 Turn 顶层 / identity 里**已经存在**的值，不是新生产点。
    project_id: str = ""
    project_name: str = ""
    project_root: str = ""
    auxiliary: bool = False
    subagent: bool = False
    #: Hub key 原文（增量状态按它判"见过没有"；F1a 调用方不给 ⇒ 空）。
    key: str = ""

    @classmethod
    def from_scan(cls, key: str, ts_iso: str, **extra: object) -> TurnMeta | None:
        """从 `MemoryHub.scan_meta()` 的一条 ``(key, ts_iso)`` 构造；形状不合返回 None。

        ts 缺失时退到 key 第四段的 Redis entry_id 毫秒（`<ms>-<seq>`，与
        `flash_daemon.hub_tree_source` 同法）；两者都没有 ⇒ None（没有时间的轮
        进不了任何时间聚合，静默补 0 会伪造 1970 年的 last_seen）。

        ``extra``：`project_id` / `project_name` / `project_root` / `auxiliary` /
        `subagent`（`scan_meta_snapshot` 那一档才有；不给 = 默认）。
        """
        parts = key.split("/")
        if len(parts) < 4 or not parts[1] or not parts[2]:
            return None
        ts = _parse_ts(ts_iso)
        if ts is None:
            try:
                ms = int(parts[3].split("-")[0])
            except ValueError:
                return None
            if ms <= 0:
                return None
            ts = datetime.fromtimestamp(ms / 1000, tz=UTC)
        return cls(principal=parts[0], agent_id=parts[1], session_id=parts[2], ts=ts,
                   key=key,
                   project_id=str(extra.get("project_id") or ""),
                   project_name=str(extra.get("project_name") or ""),
                   project_root=str(extra.get("project_root") or ""),
                   auxiliary=bool(extra.get("auxiliary", False)),
                   subagent=bool(extra.get("subagent", False)))


def _parse_ts(ts_iso: str) -> datetime | None:
    if not ts_iso:
        return None
    try:
        ts = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC)


def _require_aware(now: datetime) -> datetime:
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware (pure function: caller supplies the clock)")
    return now.astimezone(UTC)


def agent_base(agent_id: str) -> str:
    """agent base 口径：`agent_id.split(":")[0]`（补遗 G3，与 E7.4 工具统计同口径）。

    `hermes` / `hermes:default` / `hermes:accept` ⇒ ``hermes``。
    """
    return (agent_id or "").split(":")[0]


def is_unknown_bucket(agent_id: str) -> bool:
    """`unknown` / `unknown-<fp>` 待认领指纹桶（与名册准入同口径）。"""
    return (agent_id or "").startswith(UNKNOWN_AGENT_PREFIX)


def session_source(session_id: str) -> str:
    """按 session_id 前缀判 `session_id_source`（fp / tw / explicit）。"""
    if session_id.startswith("fp:"):
        return SESSION_SOURCE_FP
    if session_id.startswith("tw:"):
        return SESSION_SOURCE_TW
    return SESSION_SOURCE_EXPLICIT


# ───────────────────────────── S4 · last_seen ─────────────────────────────


class AgentLastSeen(BaseModel):
    """一个 agent base 的最近活跃读数。"""

    model_config = ConfigDict(frozen=True)

    base: str
    last_seen: datetime
    first_seen: datetime
    age_s: int
    turns: int


def last_seen_by_agent(
    metas: Iterable[TurnMeta],
    *,
    now: datetime,
    eligible: Callable[[str], bool] | None = None,
) -> dict[str, AgentLastSeen]:
    """S4：按 agent base 聚合 `last_seen`（= 该 base 全部 turn 的最大 ts）。

    **分母（非空率 100% 的分母）= 有 turn 读数的 agent base。** 输出 key 集合
    ⊆ 输入里出现过的 agent base；规则库声明但零流量的 agent（如 `cursor` /
    `openclaw`）**不会出现**——本函数不接受任何"声明清单"入参，所以也没有
    把声明当活跃的通道（MQ-F1「稳定层每轮注入一张错表」的防线换了载体，
    MQ-F6 残留一并钉住）。

    - ``now``：tz-aware 时刻，调用方给；只用于算 ``age_s``（下限 0，时钟偏斜
      不会产出负年龄）。函数内不读墙上时钟。
    - ``eligible``：对**完整 agent_id** 的准入谓词（调用方折入判据① 与 bench
      名单）；None = 只做硬排除（`unknown-*`）。
    - 确定性：输出按 base 排序的 dict；同输入两次逐字一致。
    """
    now_utc = _require_aware(now)
    acc: dict[str, tuple[datetime, datetime, int]] = {}
    for m in metas:
        if is_unknown_bucket(m.agent_id):
            continue
        if eligible is not None and not eligible(m.agent_id):
            continue
        base = agent_base(m.agent_id)
        if not base:
            continue
        lo, hi, n = acc.get(base, (m.ts, m.ts, 0))
        acc[base] = (min(lo, m.ts), max(hi, m.ts), n + 1)
    out: dict[str, AgentLastSeen] = {}
    for base in sorted(acc):
        lo, hi, n = acc[base]
        age = int((now_utc - hi).total_seconds())
        out[base] = AgentLastSeen(base=base, last_seen=hi, first_seen=lo,
                                  age_s=max(0, age), turns=n)
    return out


# ───────────────────────────── S5 · 会话去重 ─────────────────────────────


class SessionSpan(BaseModel):
    """一个会话的时间跨度（去重前 = 一个 session_id；去重后可能并入 `tw:` 头）。"""

    model_config = ConfigDict(frozen=True)

    principal: str
    agent_id: str
    session_id: str
    source: str
    first_ts: datetime
    last_ts: datetime
    turns: int
    #: 并入本会话的 `tw:` 头 session_id（空 = 没并过）
    merged_heads: tuple[str, ...] = ()


class SessionDedup(BaseModel):
    """`dedupe_sessions` 的结果：去重后的会话 + 并了谁 + 拦下了谁（可 grep 的读数）。"""

    model_config = ConfigDict(frozen=True)

    sessions: tuple[SessionSpan, ...]
    #: (tw 头 session_id, 并入的 fp session_id)
    merged: tuple[tuple[str, str], ...]
    #: (tw session_id, 原因)：有紧邻候选但**没**并——ambiguous_fp / ambiguous_head / head_too_long
    held: tuple[tuple[str, str], ...]


HELD_AMBIGUOUS_FP = "ambiguous_fp"
HELD_AMBIGUOUS_HEAD = "ambiguous_head"
HELD_HEAD_TOO_LONG = "head_too_long"


def dedupe_sessions(
    metas: Iterable[TurnMeta],
    *,
    window_s: int = TW_WINDOW_S,
    max_head_turns: int = 1,
) -> SessionDedup:
    """S5：会话去重——🔴 **过渡口径**，不是会话身份的定义。

    **病灶**（MQ-S9 地基 / MQ-L13 首个病例 / 批 S2 §问题 2 的 SESSIONS.md 双行 /
    本批登记 **MQ-A73**）：同一次会话的**首轮**只有 1 条非 system 消息，
    `identity._infer_session_from_prefix` 拿不到两条做指纹 ⇒ 落 `tw:` 时间窗兜底；
    **第二轮起**有了 [user, assistant] ⇒ 走 `fp:` 指纹。于是一次会话被切成
    "`tw:` 头（1 轮）+ `fp:` 身"两个 session_id。**真修归 identity 域**（fp 首次
    落定时回收同 agent 上一窗口的 tw 桶），本函数只在 behavior 侧把两行合成一行，
    identity 修好那天本函数应退化为恒等（`merged` 空）。

    **口径**（每一条都往"宁漏勿错"收——误合并 = 0 是本仓第一红线）：
    同 principal ∧ 同**完整** agent_id（`hermes` 与 `hermes:default` **不**互认）∧
    `tw:` 段轮数 ≤ ``max_head_turns``（默认 1：桶里只落过一个"首条消息"）∧
    `tw:` 段末 ts ≤ `fp:` 段首 ts ≤ 末 ts + ``window_s``（头在前、身紧随其后；
    tw 落在 fp **中间**的不算——那是长会话里的单消息子调用，与首轮病灶不同源）∧
    该 tw 只有**一个**这样的 fp 候选 ∧ 该 fp 只被**一个** tw 认领。
    任一条不满足 ⇒ 两行都保留，并记进 ``held``（原因可 grep）。

    与交接稿"首末 ts 重叠"的措辞差：live 病例（hermes:default，2026-09-19）的
    tw 头 09:24:58、fp 身首轮 09:25:22，**相接不相交**——字面"重叠"对病灶本身
    是 0 命中，命中的反而是"tw 落在 fp 中间"那类（live 50 例），而那类正是
    吞并风险所在（另一次会话的孤立首条消息会被长会话吃掉）。故写成"相接"。

    `explicit` 来源的 session_id 原样保留，不参与任何合并。输出确定性排序。
    """
    if window_s < 0 or max_head_turns < 1:
        raise ValueError("window_s must be >= 0 and max_head_turns >= 1")
    spans: dict[tuple[str, str, str], tuple[datetime, datetime, int]] = {}
    for m in metas:
        if is_unknown_bucket(m.agent_id):
            continue
        k = (m.principal, m.agent_id, m.session_id)
        lo, hi, n = spans.get(k, (m.ts, m.ts, 0))
        spans[k] = (min(lo, m.ts), max(hi, m.ts), n + 1)

    heads: list[tuple[str, str, str]] = []
    bodies_by_agent: dict[tuple[str, str], list[tuple[str, str, str]]] = {}
    for k in sorted(spans):
        src = session_source(k[2])
        if src == SESSION_SOURCE_TW:
            heads.append(k)
        elif src == SESSION_SOURCE_FP:
            bodies_by_agent.setdefault((k[0], k[1]), []).append(k)

    proposals: dict[tuple[str, str, str], tuple[str, str, str]] = {}   # head -> body
    held: list[tuple[str, str]] = []
    for h in heads:
        h_lo, h_hi, h_n = spans[h]
        latest_start = h_hi + timedelta(seconds=window_s)
        cands = [
            b for b in bodies_by_agent.get((h[0], h[1]), ())
            if h_hi <= spans[b][0] <= latest_start
        ]
        if not cands:
            continue
        if h_n > max_head_turns:
            held.append((h[2], HELD_HEAD_TOO_LONG))
            continue
        if len(cands) > 1:
            held.append((h[2], HELD_AMBIGUOUS_FP))
            continue
        proposals[h] = cands[0]

    claims: dict[tuple[str, str, str], list[tuple[str, str, str]]] = {}
    for h, b in proposals.items():
        claims.setdefault(b, []).append(h)
    merged: dict[tuple[str, str, str], tuple[str, str, str]] = {}
    for b, hs in claims.items():
        if len(hs) == 1:
            merged[hs[0]] = b
        else:
            held.extend((h[2], HELD_AMBIGUOUS_HEAD) for h in hs)

    heads_of: dict[tuple[str, str, str], list[tuple[str, str, str]]] = {}
    for h, b in merged.items():
        heads_of.setdefault(b, []).append(h)

    out: list[SessionSpan] = []
    for k in sorted(spans):
        if k in merged:
            continue
        lo, hi, n = spans[k]
        head_ids: list[str] = []
        for h in heads_of.get(k, ()):
            h_lo, h_hi, h_n = spans[h]
            lo, hi, n = min(lo, h_lo), max(hi, h_hi), n + h_n
            head_ids.append(h[2])
        out.append(SessionSpan(principal=k[0], agent_id=k[1], session_id=k[2],
                               source=session_source(k[2]), first_ts=lo, last_ts=hi,
                               turns=n, merged_heads=tuple(sorted(head_ids))))
    out.sort(key=lambda s: (s.principal, s.agent_id, s.first_ts, s.session_id))
    return SessionDedup(
        sessions=tuple(out),
        merged=tuple(sorted((h[2], b[2]) for h, b in merged.items())),
        held=tuple(sorted(set(held))),
    )


# ───────────────────────── F2 · 行为画像聚合（`behavior.json`）─────────────────────────
#
# 方案 `flash-repositioning-plan-20260920.md` §1 的 schema + 拍板 (14)(15) 的两个视图。
# 🔴 读数纪律：`behavior.json` 是新尺子的**被测对象**——F3 之前不引它的数字作结论。
#
# 纯函数契约（判据 ①）：`now` 入参、本地时区偏移入参、不读文件、不 `get()`；
# 同输入两次 `to_json()` 逐字一致（列表全部显式排序、dict 按 key 排序序列化）。

#: 拍板 (3) 已确认的 bench 项目（按 `project_name` 认；与 `scripts/probe_memory_redundancy.py`
#: 的 `DEFAULT_BENCH_PROJECTS` 同一份——那边 import 这里）。批 T 确认名单前**只有这九个**是
#: `bench=true`；其余非 Global 项目一律 `unknown`（补遗 G4：unknown 视同 bench，不进模型卡）。
#: `r4-ladder` / `g19-rerun-01` 是批 T 的**候选**补条，Jason 未逐个确认 ⇒ 仍 unknown。
BENCH_PROJECT_NAMES_CONFIRMED: tuple[str, ...] = (
    "x1-cc", "x3-cc", "x7-cc-dsv4", "x8-ledger-on", "x9-ledger-off",
    "x10-off-dsv4", "hb-4", "s1-cc", "h4-p28",
)

BENCH_TRUE = "true"
BENCH_FALSE = "false"
BENCH_UNKNOWN = "unknown"

#: 模型卡上限（拍板 (13)：≤1.5K 字符；守卫 `test_behavior_aggregate.py`）。
MODEL_VIEW_MAX_CHARS = 1500

#: ADR-0034 §3.4 名册/树轨道的滞后上界（300s 节流 + 一次门铃）。`behavior.json`
#: 走的就是那条 300s 轨道，`flash_lag_s` 超它打 warning（MQ-F4）。
FLASH_LAG_BOUND_S = 360

_DAY_S = 86400


def bench_state(project_name: str, project_id: str, *,
                bench: Collection[str], confirmed: Collection[str] = ()) -> str:
    """项目 bench 三态（补遗 G4）：名单命中 ⇒ true；确认非 bench 名单命中 ⇒ false；
    否则 unknown。名单按 `project_name` **或** `project_id` 认（批 T 的名单是目录名，
    G19 那条补的是 id）。"""
    keys = {project_name, project_id} - {""}
    if keys & set(bench):
        return BENCH_TRUE
    if keys & set(confirmed):
        return BENCH_FALSE
    return BENCH_UNKNOWN


class TurnCounts(BaseModel):
    model_config = ConfigDict(frozen=True)
    all: int = 0
    d7: int = 0
    d30: int = 0
    aux: int = 0
    #: 落在 `bench=unknown` 项目里的轮（**计入** all/d7/d30；批 T 确认前这段是读者要知道的曝险）
    in_unknown_projects: int = 0


class SessionStats(BaseModel):
    model_config = ConfigDict(frozen=True)
    count_all: int = 0
    count_30d: int = 0
    median_turns_per_session: float = 0.0
    median_session_minutes: float = 0.0
    #: S5 过渡口径的读数（`dedupe_sessions`）：并了几对 / 拦了几条
    merged: int = 0
    held: int = 0
    #: 🔴 F1a 残留：单轮 `tw:` 且无紧邻 `fp:` 的会话——从元数据**判不出**是不是"头"。
    #: `count_*` 把它们**按独立会话计**；这个数就是"含 N 条未判"的 N（不许当 0）。
    undetermined_tw_heads: int = 0


class ToolStat(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    calls: int = 0
    ok: int = 0
    err: int = 0
    #: ok/(ok+err)；两者皆 0 ⇒ None（不是 0%）。值来自 Index `profile/tool/<base>/<tool>`
    #: 计数（E7.4），**不重算**（补遗 G1）。
    ok_rate: float | None = None


class LedgerStats(BaseModel):
    """一个 agent base 的账本读数。

    🔴 `created` 的口径是「**首个写者**是这个 agent 的本数」，不是「create 事件的作者」——
    live 实测 **85/85 条 `ledger_create` 的 `last_writer` 为空**（`new_ledger()` 建出来时
    还没人写过），按它归属会得到恒 0（原则 13 层 3 的形态：字段在、生产者没接上）。
    首个写者从事件流按时间取（update / user_edit 的 `last_writer` ∪ switch 的 `agent_id`），
    是这条链上**唯一有生产点**的归属信号。
    """

    model_config = ConfigDict(frozen=True)
    created: int = 0
    active: int = 0
    closed: int = 0
    tombstoned: int = 0
    switches_30d: int = 0
    #: 只算已完结（closed / 墓碑）的本；没有 ⇒ None
    median_lifetime_h: float | None = None
    median_verified_per_ledger: float | None = None


class AgentBehavior(BaseModel):
    model_config = ConfigDict(frozen=True)
    base: str
    #: 完整 agent_id 变体（dashboard 明细；模型卡不放，补遗 G3）
    variants: list[str] = Field(default_factory=list)
    first_seen: str = ""
    last_seen: str = ""
    age_s: int = 0
    turns: TurnCounts = TurnCounts()
    sessions: SessionStats = SessionStats()
    #: 24 桶，本地时区（`tz_offset_s` 入参）；只给 dashboard（拍板 (15)）
    active_hours: list[int] = Field(default_factory=lambda: [0] * 24)
    tools_top: list[ToolStat] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    ledger: LedgerStats = LedgerStats()


class AgentTurns(BaseModel):
    model_config = ConfigDict(frozen=True)
    base: str
    turns: int


class ProjectBehavior(BaseModel):
    model_config = ConfigDict(frozen=True)
    project_id: str
    name: str = ""
    root_path_hint: str = ""
    first_seen: str = ""
    last_seen: str = ""
    turns: int = 0
    agents: list[AgentTurns] = Field(default_factory=list)
    ledgers: list[str] = Field(default_factory=list)
    bench: str = BENCH_UNKNOWN


class Handoff(BaseModel):
    model_config = ConfigDict(frozen=True)
    ledger_id: str
    from_agent: str
    to_agent: str
    at: str


class HabitRef(BaseModel):
    """`habits_ref` 的输入项：Index profile lane 的一条 fact（调用方取，本模块不读 Index）。"""
    model_config = ConfigDict(frozen=True)
    fact_id: str
    strength: int = 1
    importance: float = 1.0
    text: str = ""


class HabitsRef(BaseModel):
    model_config = ConfigDict(frozen=True)
    total: int = 0
    fact_ids: list[str] = Field(default_factory=list)
    #: 拍板 (14)：overview 只带按 strength 排的前 3 条正文（截断），深挖走 memory_search
    overview: list[str] = Field(default_factory=list)


class HardRulesRef(BaseModel):
    model_config = ConfigDict(frozen=True)
    count: int = 0
    hash: str = ""


class BehaviorMeta(BaseModel):
    model_config = ConfigDict(frozen=True)
    generated_at: str
    #: 最后一条纳入的 turn ts（= 新鲜度读数，`flash_lag_s` 由 daemon 按补遗 G7 算）
    hub_watermark: str = ""
    tz_offset_s: int = 0
    turns_seen: int = 0
    turns_eligible: int = 0
    turns_bench: int = 0
    agents_excluded: dict[str, int] = Field(default_factory=dict)
    projects_by_bench: dict[str, int] = Field(default_factory=dict)
    sessions_undetermined_total: int = 0
    #: 字段说明（人读；含 F2.5 那句"sessions 计数含 N 条未判"）
    notes: dict[str, str] = Field(default_factory=dict)


class Behavior(BaseModel):
    model_config = ConfigDict(frozen=True)
    meta: BehaviorMeta
    agents: list[AgentBehavior] = Field(default_factory=list)
    projects: list[ProjectBehavior] = Field(default_factory=list)
    handoffs: list[Handoff] = Field(default_factory=list)
    habits_ref: HabitsRef = HabitsRef()
    hard_rules_ref: HardRulesRef = HardRulesRef()

    # ── 序列化（逐字一致的判据载体）──
    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), ensure_ascii=False,
                          indent=1, sort_keys=True) + "\n"

    def dashboard_view(self) -> dict:
        """全字段（活跃时段 / 工具失败率 / variants 只在这里，拍板 (15)）。"""
        return self.model_dump(mode="json")

    def model_view(self, *, max_chars: int = MODEL_VIEW_MAX_CHARS) -> dict:
        """模型卡（拍板 (13)(14)(15) + 补遗 G3/G4）：agents 摘要 / `bench=false` 的
        projects / handoffs / habits_ref（overview 前 3 条）。**≤ max_chars**：超了按
        `trimmed` 里记的顺序裁（先裁 overview 正文、再裁列表长度），裁了什么写在返回值里。
        """
        agents = [{
            "agent": a.base, "last_seen": a.last_seen[:16], "turns_30d": a.turns.d30,
            "projects": list(a.projects),
            "ledgers": {"created": a.ledger.created, "active": a.ledger.active},
        } for a in self.agents]
        projects = [{
            "id": p.project_id, "name": p.name,
            "agents": [x.base for x in p.agents], "ledgers": len(p.ledgers),
        } for p in self.projects if p.bench == BENCH_FALSE]
        handoffs = [{"ledger": h.ledger_id, "from": h.from_agent, "to": h.to_agent,
                     "at": h.at[:16]} for h in self.handoffs]
        habits = {"total": self.habits_ref.total, "fact_ids": list(self.habits_ref.fact_ids),
                  "overview": list(self.habits_ref.overview)}
        view: dict = {"hub_watermark": self.meta.hub_watermark[:19],
                      "agents": agents, "projects": projects, "handoffs": handoffs,
                      "habits_ref": habits,
                      "note": ("projects lists confirmed non-bench only; "
                               f"sessions counts include {self.meta.sessions_undetermined_total} "
                               "undetermined tw heads")}
        # 🔴 `trimmed` 从一开始就挂进 view（同一个 list 对象，后面 append 即时反映）：
        # 否则每次量的是**不含 trimmed 的**串，而返回值含它 —— 守卫按返回值量，
        # 就会在梯子走完之后再超出去（实测 1503 > 1500，尺子量的不是被交付的那个东西）。
        trimmed: list[str] = []
        view["trimmed"] = trimmed
        steps: list[tuple[str, Callable[[], None]]] = [
            ("overview_60", lambda: habits.__setitem__(
                "overview", [t[:60] for t in habits["overview"]])),
            ("handoffs_3", lambda: view.__setitem__("handoffs", view["handoffs"][:3])),
            ("projects_5", lambda: view.__setitem__("projects", view["projects"][:5])),
            ("agents_8", lambda: view.__setitem__("agents", view["agents"][:8])),
            ("fact_ids_3", lambda: habits.__setitem__("fact_ids", habits["fact_ids"][:3])),
            ("overview_drop", lambda: habits.__setitem__("overview", [])),
            ("agent_projects_3", lambda: [a.__setitem__("projects", a["projects"][:3])
                                          for a in view["agents"]]),
            ("note_drop", lambda: view.pop("note", None)),
            ("agents_5", lambda: view.__setitem__("agents", view["agents"][:5])),
            ("projects_3", lambda: view.__setitem__("projects", view["projects"][:3])),
            ("handoffs_1", lambda: view.__setitem__("handoffs", view["handoffs"][:1])),
            ("fact_ids_drop", lambda: habits.__setitem__("fact_ids", [])),
            ("agent_projects_drop", lambda: [a.__setitem__("projects", [])
                                             for a in view["agents"]]),
            ("handoffs_drop", lambda: view.__setitem__("handoffs", [])),
            ("projects_drop", lambda: view.__setitem__("projects", [])),
        ]
        for name, step in steps:
            if len(model_view_json(view)) <= max_chars:
                break
            step()
            trimmed.append(name)
        # 兜底：梯子走完仍超 ⇒ 逐个丢末尾 agent（列表按 base 排序）。
        # 🔴 这一步保证的是**上限恒成立**，不是"通常成立"——守卫按上限断言，
        # 而喂进来的 agent 数没有上界（它是 live 数据）。
        while len(model_view_json(view)) > max_chars and len(view["agents"]) > 1:
            view["agents"] = view["agents"][:-1]
            if trimmed[-1:] != ["agents_shrunk"]:
                trimmed.append("agents_shrunk")
        if not trimmed:
            view.pop("trimmed", None)
        return view


def model_view_json(view: dict) -> str:
    """模型卡的字节形态（紧凑 JSON；≤1.5K 的判据按这个串量）。"""
    return json.dumps(view, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


# ── 内部：ledger 事件的规整 ──


def _event_ts(ev: Mapping) -> datetime | None:
    raw = ev.get("ts")
    if isinstance(raw, datetime):
        return raw.astimezone(UTC) if raw.tzinfo else raw.replace(tzinfo=UTC)
    if isinstance(raw, int | float):
        v = float(raw)
        return datetime.fromtimestamp(v / 1000 if v > 1e11 else v, tz=UTC)
    if isinstance(raw, str):
        return _parse_ts(raw)
    return None


def _iso(dt: datetime | None) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ") if dt else ""


def _writer_base(writer: str) -> str:
    """账本 `last_writer` → agent base；"user" / 空 / unknown 桶不算 agent 写者。"""
    w = (writer or "").strip()
    if not w or w == "user" or is_unknown_bucket(w):
        return ""
    return agent_base(w)


def _ledger_facets(ledger_events: Iterable[Mapping], tombstoned: Mapping[str, float],
                   *, now: datetime) -> tuple[dict, dict[str, list[str]], list[Handoff],
                                              dict[str, int]]:
    """事件流 → (每本账本的 facet, project→ledgers, handoffs, 每 base 30d 切换数)。"""
    per: dict[str, dict] = {}
    proj_ledgers: dict[str, list[str]] = {}
    writers: dict[str, list[tuple[datetime, str]]] = {}
    switches_30d: dict[str, int] = {}
    cutoff30 = now - timedelta(days=30)
    # 🔴 按事件时刻折叠（last-write-wins 的"last"必须按**时间**定，不能按调用方给的
    # 序列顺序定）——否则同一批事件换个顺序喂进来，`verified` / `status` 会得到不同
    # 的终态，判据①（同输入逐字一致）就退化成"同顺序才一致"。稳定排序：ts 相同
    # 保持输入序（= Hub 追加序，与 `replay_ledger_events` 的口径一致）；缺 ts 的排最前。
    _epoch = datetime.fromtimestamp(0, tz=UTC)
    ordered = sorted(enumerate(ledger_events),
                     key=lambda p: (_event_ts(p[1]) or _epoch, p[0]))
    for _idx, ev in ordered:
        etype = str(ev.get("type", ""))
        pl = ev.get("payload") or {}
        ts = _event_ts(ev)
        if etype in ("ledger_create", "ledger_update", "ledger_user_edit"):
            dump = pl.get("ledger") or {}
            lid = str(dump.get("ledger_id") or "")
            if not lid:
                continue
            f = per.setdefault(lid, {"creator": "", "created": None, "last": None,
                                     "status": "active", "verified": 0})
            wb = _writer_base(str(dump.get("last_writer") or ""))
            if etype == "ledger_create":
                f["created"] = f["created"] or (_parse_ts(str(dump.get("created_at") or ""))
                                                or ts)
            f["last"] = max(x for x in (f["last"], ts) if x) if (f["last"] or ts) else None
            f["status"] = str(dump.get("status") or "active")
            f["verified"] = len((dump.get("sections") or {}).get("verified") or [])
            if wb and ts:
                writers.setdefault(lid, []).append((ts, wb))
        elif etype == "ledger_switch":
            lid = str(pl.get("to_ledger_id") or pl.get("ledger_id") or "")
            ab = agent_base(str(pl.get("agent_id") or ""))
            pid = str(pl.get("project_id") or "")
            if lid and pid:
                lst = proj_ledgers.setdefault(pid, [])
                if lid not in lst:
                    lst.append(lid)
            if lid and ab and ts:
                writers.setdefault(lid, []).append((ts, ab))
            if ab and ts and ts >= cutoff30 and not is_unknown_bucket(str(pl.get("agent_id") or "")):
                switches_30d[ab] = switches_30d.get(ab, 0) + 1
    handoffs: list[Handoff] = []
    for lid in sorted(writers):
        seq = sorted(writers[lid], key=lambda t: (t[0], t[1]))
        # 归属 = **首个写者**（见 `LedgerStats` docstring：create 事件没有作者）
        if seq and lid in per and not per[lid]["creator"]:
            per[lid]["creator"] = seq[0][1]
        prev = ""
        for ts, wb in seq:
            if prev and wb != prev:
                handoffs.append(Handoff(ledger_id=lid, from_agent=prev, to_agent=wb,
                                        at=_iso(ts)))
            prev = wb
    handoffs.sort(key=lambda h: (h.at, h.ledger_id, h.from_agent, h.to_agent))
    for lid, f in per.items():
        f["tombstoned_ms"] = float(tombstoned.get(lid, 0.0)) if lid in tombstoned else None
    return per, {k: sorted(v) for k, v in proj_ledgers.items()}, handoffs, switches_30d


def _ledger_stats_for(base: str, facets: Mapping[str, dict], switches_30d: Mapping[str, int],
                      *, now: datetime) -> LedgerStats:
    mine = [f for f in facets.values() if f["creator"] == base]
    lifetimes: list[float] = []
    created = active = closed = tomb = 0
    verified: list[int] = []
    for f in mine:
        created += 1
        verified.append(int(f["verified"]))
        if f["tombstoned_ms"] is not None:
            tomb += 1
            end = datetime.fromtimestamp(f["tombstoned_ms"] / 1000, tz=UTC) if f["tombstoned_ms"] else f["last"]
        elif f["status"] == "closed":
            closed += 1
            end = f["last"]
        else:
            active += 1
            end = None
        if end and f["created"]:
            lifetimes.append(max(0.0, (end - f["created"]).total_seconds() / 3600))
    return LedgerStats(
        created=created, active=active, closed=closed, tombstoned=tomb,
        switches_30d=int(switches_30d.get(base, 0)),
        median_lifetime_h=round(median(lifetimes), 2) if lifetimes else None,
        median_verified_per_ledger=round(float(median(verified)), 2) if verified else None,
    )


def aggregate(
    metas: Iterable[TurnMeta],
    *,
    now: datetime,
    ledger_events: Iterable[Mapping] = (),
    tombstoned_ledgers: Mapping[str, float] | None = None,
    tool_stats: Mapping[str, Sequence[Mapping]] | None = None,
    habits: Sequence[HabitRef] = (),
    hard_rules: Sequence[str] = (),
    eligible: Callable[[str], bool] | None = None,
    bench_projects: Collection[str] = BENCH_PROJECT_NAMES_CONFIRMED,
    confirmed_projects: Collection[str] = (),
    tz_offset_s: int = 0,
    hub_watermark: datetime | None = None,
    tools_top_n: int = 5,
    habits_top_n: int = 10,
) -> Behavior:
    """Hub 元数据 + 账本事件 + Index 工具统计 → `Behavior`（纯函数，方案 §1 / §4）。

    - ``metas``：`TurnMeta` 序列（`scan_meta_snapshot` 一档；不 `get()`——补遗 G1）。
    - ``now``：tz-aware；`age_s` / 7d / 30d 窗口 / 账本寿命都按它算。**不读墙上时钟。**
    - ``eligible``：对完整 agent_id 的准入谓词（`browsing_admittance` 判据①）；
      本函数只做硬排除 `unknown-*`。🔴 **没有"声明清单"入参**：`agents` 段 ⊆ 有 turn 的
      base，零流量 agent 没有通道进来（MQ-F1 防线，F1a 同款）。
    - ``bench_projects`` / ``confirmed_projects``：三态判据（`bench_state`）。
      **`bench=true` 项目的轮不进 `agents[].turns` / `active_hours` / `sessions`**；
      `unknown` 项目的轮计入，但另计 `turns.in_unknown_projects`。
    - ``tool_stats``：`{base: [{name, calls, ok, err}]}`（Index E7.4 原样，不重算）。
    - ``tz_offset_s``：本地时区偏移（存 UTC 算本地，MQ-L46）。
    - ``hub_watermark``：调用方给（daemon 的增量状态）；None ⇒ 取 metas 最大 ts。
    """
    now_utc = _require_aware(now)
    tombstoned_ledgers = tombstoned_ledgers or {}
    tool_stats = tool_stats or {}
    rows = [m for m in metas]
    rows.sort(key=lambda m: (m.ts, m.key, m.agent_id, m.session_id))

    excluded = {"unknown_bucket": 0, "ineligible": 0}
    kept: list[TurnMeta] = []
    for m in rows:
        if is_unknown_bucket(m.agent_id):
            excluded["unknown_bucket"] += 1
            continue
        if eligible is not None and not eligible(m.agent_id):
            excluded["ineligible"] += 1
            continue
        if not agent_base(m.agent_id):
            excluded["ineligible"] += 1
            continue
        kept.append(m)

    # ── projects（先定三态，agents 段要按它隔离）──
    proj: dict[str, dict] = {}
    for m in kept:
        if not m.project_id:
            continue
        p = proj.setdefault(m.project_id, {"name": "", "root": "", "first": m.ts,
                                           "last": m.ts, "turns": 0, "agents": {}})
        p["name"] = m.project_name or p["name"]
        p["root"] = m.project_root or p["root"]
        p["first"], p["last"] = min(p["first"], m.ts), max(p["last"], m.ts)
        p["turns"] += 1
        b = agent_base(m.agent_id)
        p["agents"][b] = p["agents"].get(b, 0) + 1
    bench_of = {pid: bench_state(p["name"], pid, bench=bench_projects,
                                 confirmed=confirmed_projects) for pid, p in proj.items()}

    facets, proj_ledgers, handoffs, switches_30d = _ledger_facets(
        ledger_events, tombstoned_ledgers, now=now_utc)

    projects = [ProjectBehavior(
        project_id=pid, name=p["name"], root_path_hint=p["root"],
        first_seen=_iso(p["first"]), last_seen=_iso(p["last"]), turns=p["turns"],
        agents=[AgentTurns(base=b, turns=n) for b, n in sorted(p["agents"].items())],
        ledgers=proj_ledgers.get(pid, []), bench=bench_of[pid],
    ) for pid, p in sorted(proj.items())]

    # ── agents（bench=true 的轮剔除；unknown 计入并另记）──
    non_bench = [m for m in kept if bench_of.get(m.project_id, BENCH_FALSE) != BENCH_TRUE]
    turns_bench = len(kept) - len(non_bench)
    seen = last_seen_by_agent(non_bench, now=now_utc)     # F1a 口径，不另写
    dedup = dedupe_sessions(non_bench)                     # window_s 用 F1a 默认，不另写
    held_ids = {sid for sid, _r in dedup.held}
    cut7, cut30 = now_utc - timedelta(days=7), now_utc - timedelta(days=30)
    agents: list[AgentBehavior] = []
    for base, ls in seen.items():
        mine = [m for m in non_bench if agent_base(m.agent_id) == base]
        hours = [0] * 24
        for m in mine:
            hours[int(((m.ts.timestamp() + tz_offset_s) % _DAY_S) // 3600)] += 1
        sess = [s for s in dedup.sessions if agent_base(s.agent_id) == base]
        s_turns = [s.turns for s in sess]
        s_mins = [(s.last_ts - s.first_ts).total_seconds() / 60 for s in sess]
        undetermined = sum(1 for s in sess if s.source == SESSION_SOURCE_TW and s.turns == 1
                           and not s.merged_heads and s.session_id not in held_ids)
        merged_n = sum(len(s.merged_heads) for s in sess)
        held_n = sum(1 for sid, _r in dedup.held
                     if any(s.session_id == sid for s in sess))
        stats = []
        for rec in list(tool_stats.get(base, ()))[:tools_top_n]:
            ok, err = int(rec.get("ok", 0)), int(rec.get("err", 0))
            stats.append(ToolStat(name=str(rec.get("name", "")), calls=int(rec.get("calls", 0)),
                                  ok=ok, err=err,
                                  ok_rate=round(ok / (ok + err), 4) if ok + err else None))
        agents.append(AgentBehavior(
            base=base, variants=sorted({m.agent_id for m in mine}),
            first_seen=_iso(ls.first_seen), last_seen=_iso(ls.last_seen), age_s=ls.age_s,
            turns=TurnCounts(all=len(mine), d7=sum(1 for m in mine if m.ts >= cut7),
                             d30=sum(1 for m in mine if m.ts >= cut30),
                             aux=sum(1 for m in mine if m.auxiliary),
                             in_unknown_projects=sum(1 for m in mine if m.project_id
                                                     and bench_of.get(m.project_id) == BENCH_UNKNOWN)),
            sessions=SessionStats(
                count_all=len(sess), count_30d=sum(1 for s in sess if s.last_ts >= cut30),
                median_turns_per_session=round(float(median(s_turns)), 2) if s_turns else 0.0,
                median_session_minutes=round(float(median(s_mins)), 2) if s_mins else 0.0,
                merged=merged_n, held=held_n, undetermined_tw_heads=undetermined),
            active_hours=hours, tools_top=stats,
            projects=sorted({m.project_id for m in mine if m.project_id}),
            ledger=_ledger_stats_for(base, facets, switches_30d, now=now_utc),
        ))

    hab = sorted(habits, key=lambda h: (-h.strength, -h.importance, h.fact_id))
    habits_ref = HabitsRef(total=len(hab), fact_ids=[h.fact_id for h in hab[:habits_top_n]],
                           overview=[h.text.strip().replace("\n", " ")[:120]
                                     for h in hab[:3] if h.text.strip()])
    rules_blob = "\n".join(hard_rules)
    hard_ref = HardRulesRef(count=len(hard_rules),
                            hash=hashlib.sha256(rules_blob.encode("utf-8")).hexdigest()[:16]
                            if hard_rules else "")
    wm = hub_watermark or (max((m.ts for m in rows), default=None))
    undetermined_total = sum(a.sessions.undetermined_tw_heads for a in agents)
    by_bench = {k: sum(1 for v in bench_of.values() if v == k)
                for k in (BENCH_TRUE, BENCH_FALSE, BENCH_UNKNOWN)}
    meta = BehaviorMeta(
        generated_at=_iso(now_utc), hub_watermark=_iso(wm), tz_offset_s=tz_offset_s,
        turns_seen=len(rows), turns_eligible=len(kept), turns_bench=turns_bench,
        agents_excluded=excluded, projects_by_bench=by_bench,
        sessions_undetermined_total=undetermined_total,
        notes={
            "sessions": (f"counts include {undetermined_total} single-turn tw sessions with no "
                         "adjacent fp body within the merge window; from scan_meta metadata "
                         "alone they cannot be told apart from session heads (F1a S5 residual, "
                         "MQ-A73). Batch F2 resolved the live population by reading those turns "
                         "only (61 targeted gets, <0.05s): all are single-message first "
                         "requests, 42/61 are auxiliary sub-calls, and only 2 non-aux ones have "
                         "a following fp session within an hour -- see report-batch-F2. "
                         "Counted as separate sessions here, never as 0."),
            "bench": ("projects[].bench: true = confirmed list; unknown = not yet confirmed "
                      "(treated as bench for the model card); agents[].turns exclude "
                      "bench=true turns only, unknown turns are counted and reported in "
                      "turns.in_unknown_projects"),
            "tools_top": "counts read from Index profile/tool/<base>/<tool>, not recomputed",
            "active_hours": "local time (tz_offset_s applied to UTC); dashboard only",
        },
    )
    return Behavior(meta=meta, agents=agents, projects=projects, handoffs=handoffs,
                    habits_ref=habits_ref, hard_rules_ref=hard_ref)
