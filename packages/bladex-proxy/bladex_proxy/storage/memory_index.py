"""Memory Index 派生层 — 事实存储 + LanceDB 向量索引（ADR-0009 §7，护城河底座）。

Memory Index 从 Memory Hub 增量派生：
  - 后台 consolidation worker 从 Memory Hub 读 Turns → 调 bladex-core 提炼 Fact
  - Fact 存 Memory Index（metadata + LanceDB 向量）
  - Memory Index 可从 Memory Hub 增量重建（水位游标 + 重建期只降级不停摆）

方向永远 Pipeline → Memory Hub → Memory Index（ADR-0009）。Memory Index 坏了能从 Memory Hub 重建。
本任务只建"写派生"，不接注入闭环（相关注入是 M-proxy-3）。
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import os
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import msgpack
import structlog
from bladex_core.attribution import (
    UNASSIGNED_MATTER_ID,
    AttributionPipeline,
    AttributionSource,
    LinkJudge,
    _native_match_keys,
    _normalize,
    new_matter_id,
)
from bladex_core.consolidation_proxy import (
    EmbedderProtocol,
    ProxyConsolidator,
    embed_passage_compat,
    embed_query_compat,
)
from bladex_core.distillation import DistillerProtocol, DistillOutput
from bladex_core.fact import ConversationTurn, Fact, ItemKind, Provenance, fact_lane
from bladex_core.flags import flag_enabled, flag_number
from bladex_core.matter import (
    _MAX_MATTER_ENTITIES,
    EdgeProvenance,
    EdgeRelation,
    EdgeTargetType,
    Matter,
    MatterAggregateView,
    MatterEdge,
    MatterOrigin,
    MatterStatus,
)
from rocksdict import Rdict

from bladex_proxy.models import (
    AdminEvent,
    DistillRecord,
    JudgmentRecord,
    TASKSTATE_VERDICTS,
    TaskStateJudgment,
)
from bladex_proxy.storage.memory_hub import MemoryHub

if TYPE_CHECKING:
    from bladex_core.attribution import JudgmentJournalProtocol

logger = structlog.get_logger()

# Memory Index metadata RocksDB key 前缀
_FACT_PREFIX = "fact/"
# 🔴 下面两个 key 字面量里的 "p2" 是历史拼写，**不要跟着 ADR-0026 改名**——
# 它们已落盘在用户的 Memory Index meta 库里，改字面量 = 游标全部读不到 = 全量重蒸。
# 层名叫 Memory Index，磁盘上的 key 仍是 p2_*，两者刻意脱钩。
# 遗留水位游标 key（字典序单游标方案已废弃，仅保留用于清理）
_INDEX_WATERMARK_KEY = "__meta__/p2_watermark"
# 消费游标：consumed-key 集合前缀（替代字典序水位游标，review 20260707）
_CONSUMED_PREFIX = "__meta__/p2_consumed/"

# 蒸馏中断守卫（2026-08-05 事故驱动，见 rebuild_from_hub 里的调用点）：
# 一轮 rebuild 里蒸馏调用失败率 >= 此比例 → 判为上游中断，本轮不消费、不写 Memory Index。
# 0 = 关闭守卫（退回旧行为：失败也照常消费）。默认 0.5 = 一半以上调用失败。
_DISTILL_OUTAGE_ABORT_RATIO_DEFAULT = 0.5
_DISTILL_OUTAGE_ABORT_RATIO_ENV = "BLADEX_DISTILL_OUTAGE_ABORT_RATIO"

# MS-1（复核 D4）：蒸馏失败的**单轮**重试上限。
# 守卫（上面那条）管的是"整批中断"，本项管的是"零星失败"——后者此前一律照常
# 标记消费，那些轮次的事实就永久丢了（台账不缓存失败 → 重试本是零成本的）。
# 加上限是为了防坏输入永久滞留：某条 turn 每次都解析失败会让它永远重放。
_DISTILL_RETRY_PREFIX = "__meta__/p2_distill_retry/"
_DISTILL_RETRY_MAX_DEFAULT = 3
_DISTILL_RETRY_MAX_ENV = "BLADEX_DISTILL_RETRY_MAX"


def _distill_retry_max() -> int:
    """单轮蒸馏失败重试上限（env 覆盖；非法值退回默认，0 = 不重试即旧行为）。"""
    raw = os.environ.get(_DISTILL_RETRY_MAX_ENV)
    if raw is None or not raw.strip():
        return _DISTILL_RETRY_MAX_DEFAULT
    try:
        val = int(raw)
    except ValueError:
        logger.warning("index_distill_retry_max_invalid", raw=raw,
                       fallback=_DISTILL_RETRY_MAX_DEFAULT)
        return _DISTILL_RETRY_MAX_DEFAULT
    return max(0, val)


def _distill_outage_abort_ratio() -> float:
    """守卫阈值（env 覆盖；非法值退回默认，不让配置笔误静默关掉守卫）。"""
    raw = os.environ.get(_DISTILL_OUTAGE_ABORT_RATIO_ENV)
    if raw is None or not raw.strip():
        return _DISTILL_OUTAGE_ABORT_RATIO_DEFAULT
    try:
        val = float(raw)
    except ValueError:
        logger.warning("index_distill_outage_ratio_invalid", raw=raw,
                       fallback=_DISTILL_OUTAGE_ABORT_RATIO_DEFAULT)
        return _DISTILL_OUTAGE_ABORT_RATIO_DEFAULT
    if val < 0 or val > 1:
        logger.warning("index_distill_outage_ratio_out_of_range", raw=raw,
                       fallback=_DISTILL_OUTAGE_ABORT_RATIO_DEFAULT)
        return _DISTILL_OUTAGE_ABORT_RATIO_DEFAULT
    return val


def _distiller_counters(consolidator: Any) -> dict[str, int] | None:
    """取 consolidator 的蒸馏计数器快照；拿不到返回 None（测试桩/无蒸馏器）。"""
    fn = getattr(consolidator, "distiller_stats", None)
    if fn is None:
        return None
    try:
        return fn()
    except Exception:  # noqa: BLE001 —— 观测面绝不影响主体
        return None


def _adjudicator_counters(adjudicator: Any) -> dict[str, int] | None:
    """取裁决器计数器快照；拿不到返回 None（未配裁决器 / 测试桩）。

    与 `_distiller_counters` 同型——**观测面拿不到就报"无从判断"，不冒充正常**。
    """
    fn = getattr(adjudicator, "stats", None)
    if fn is None:
        return None
    try:
        return fn()
    except Exception:  # noqa: BLE001 —— 观测面绝不影响主体
        return None


def _distill_delta(
    before: dict[str, int] | None, after: dict[str, int] | None
) -> tuple[int, int] | None:
    """本轮的 (调用数, 无产出数)。信息不全 → None = 无从判断，调用方按旧行为走。

    「无产出」= `fails`（调用抛错）+ `parse_fails`（拿到响应但不是合法 JSON）。
    两者对本函数的关心点**后果完全相同**：这一轮被标记消费、零事实、且都不写台账
    （`distillation.py` 只缓存解析成功的结果）——也就是同样地"白烧"。
    2026-08-05 实测撞上后者：模型正常开写 JSON 但被 max_tokens 掐断，
    `distill_parse_failed` 不计入 `fails`，守卫因此拦不住批量坏 JSON 的情形。
    """
    if not before or not after:
        return None
    calls = int(after.get("calls", 0)) - int(before.get("calls", 0))
    fails = (int(after.get("fails", 0)) - int(before.get("fails", 0))
             + int(after.get("parse_fails", 0)) - int(before.get("parse_fails", 0)))
    if calls <= 0 or fails < 0:
        # 计数器没动（全部命中台账）或被重置 → 没有可判定的失败信号。
        return None
    return calls, fails


# ADR-0012: Matter 卡 + 归属边的 metadata key 前缀
_MATTER_PREFIX = "matter/"
_EDGE_PREFIX = "edge/"
# U10（ADR-0026 R5）：画像存储——profile/user/<uid>（规则文件观察）+
# profile/agent/<base>（工具计数器）。确定性从 Memory Hub 重放聚合，clear() 一并清（重建等价）。
_PROFILE_PREFIX = "profile/"
# ADR-0028 E7.3：规则文件正文副本（每 agent 一份，hash 变才更新）。
_RULEFILE_PREFIX = "rulefile/"
# ADR-0028 E7.2：Project 实体（主题轴最上层 Project → Matter → TaskUnit）。
_PROJECT_PREFIX = "project/"
# ADR-0028 E7.1：文件内容索引的 hash 去重记录（同 hash 不重复调 LLM）。
_FILEHASH_PREFIX = "filehash/"

# 技术词表（USER.md「编程偏好」的判别依据；宁漏勿误）
# 注：这里的 "litellm" 是**用户可能写出的词**，不是我们的产品措辞——它要匹配的是
# 用户原文。2026-08-05 把产品面统一改叫 Router 时**有意保留**这一条：改了就认不出
# 用户说的那个词了。产品面的改名在 `router_sdk.py`。
_TECH_ENTITY_WORDS: frozenset[str] = frozenset({
    "python", "typescript", "javascript", "go", "rust", "java", "sql",
    "pytest", "git", "docker", "lancedb", "rocksdb", "redis", "fastapi",
    "litellm", "sqlite", "bladex", "kubernetes", "terraform",
})


def _is_tech_entity(entity: str) -> bool:
    return str(entity).strip().lower() in _TECH_ENTITY_WORDS


# Flash 落盘帮手（`write_if_changed`）在 `bladex_core.flash`；本模块自 2026-09-03 S5
# 起不再写 Flash（老渲染器已删，唯一写者是 flash_daemon）。


def _utc_now_iso() -> str:
    from datetime import UTC as _UTC
    from datetime import datetime as _dtt
    return _dtt.now(_UTC).isoformat()

# U4B-B3（ADR-0026 §4.3）：从 tool 事件确定性提取文件指针（路径 + content_hash，零 LLM）。
# ADR-0028 E1.5：白名单收紧——只认明确指向"被操作文件"的参数键；
# 删掉泛化的 "path"/"file" 键与"值形似路径就抓"的兜底分支
# （`search_files{path:"/Users/alice"}` 事故来源）。
_PATH_ARG_KEYS = ("file_path", "target_file", "filename", "notebook_path")
# 检索类工具的路径参数语义是"在哪找"，不是"操作了什么" —— 整个工具事件跳过。
_FILE_REF_TOOL_BLACKLIST = frozenset({
    "search_files", "glob", "grep", "list_dir", "ls",
})
_PATH_RE = None  # 惰性编译


def _extract_file_refs(tool_events: list) -> list[dict[str, str]]:
    """从 Turn.tool_events 提取 [{path, content_hash}]。确定性、零 LLM。

    ADR-0028 E1.5 白名单规则（三条全部满足才产出候选）：
      1. 参数键 ∈ _PATH_ARG_KEYS；
      2. 值匹配 _PATH_RE（必须带扩展名——目录天然出局）；
      3. tool_name ∉ _FILE_REF_TOOL_BLACKLIST（检索类工具整体跳过）。

    content_hash：result 为字符串时取 sha256[:16]（供 hash 变更失效检测，U5 生命周期）。
    """
    import re

    global _PATH_RE
    if _PATH_RE is None:
        _PATH_RE = re.compile(r"^/?(?:[\w.\-]+/)+[\w.\-]+\.[A-Za-z0-9]{1,8}$")

    refs: list[dict[str, str]] = []
    seen: set[str] = set()
    for te in tool_events or []:
        tool_name = str(getattr(te, "tool_name", "") or "").strip().lower()
        if tool_name in _FILE_REF_TOOL_BLACKLIST:
            continue
        args = getattr(te, "arguments", None)
        result = getattr(te, "result", None)
        path = ""
        if isinstance(args, dict):
            for k in _PATH_ARG_KEYS:
                v = args.get(k)
                if isinstance(v, str) and _PATH_RE.match(v.strip()):
                    path = v.strip()
                    break
        if not path or path in seen:
            continue
        seen.add(path)
        chash = ""
        if isinstance(result, str) and result:
            chash = hashlib.sha256(result.encode("utf-8", "ignore")).hexdigest()[:16]
        refs.append({"path": path, "content_hash": chash})
    return refs


# T2: 质心加权策略参数（留实测，ADR-0012 §3.5 manual 强锚点）
_DEFAULT_MANUAL_CENTROID_WEIGHT = 3.0  # manual 边权重（强锚点）
_DEFAULT_AUTO_CENTROID_WEIGHT = 1.0    # auto 边权重
_MAX_SUMMARY_CHARS = 500               # Matter 摘要最大长度
_MAX_SUMMARY_FACTS = 5                 # 摘要取最近 N 条 fact
# 复核 §3.4：matters 向量表按 matter_id 去重时的召回放大倍数
# （存量脏行下要多取一些才能凑够 k 个**不同**的 Matter）。
_MATTER_DEDUP_EXPAND = 5
_DEFAULT_DIGESTION_THRESHOLD = 0.85    # 未归属池消化阈值（ADR-0014 L0 重标；= semantic_threshold，同话题聚拢）
_DEFAULT_DIGESTION_MIN_CLUSTER = 2     # 最小成簇数（少于则留池）

# A1/T3: secondary 追新节流（秒级，不每次 search 都 catch-up，避免抖动）
_CATCHUP_THROTTLE_S = 1.0

# embedding 后端可选化（2026-07-26）：向量空间 model_id 不变量 key。
# query 与 fact 必须同模型同空间（ADR-0009）；换 embedding 后端/模型 = 全量重嵌
# （scripts/reembed_index.py），不允许混写。
_EMBED_MODEL_KEY = "__meta__/embed_model_id"
# ADR-0028 E2.4：生命周期重算 job 的上次执行时间戳（节流用）。
_LIFECYCLE_LAST_RUN_KEY = "__meta__/lifecycle_last_run"
# 重算后 |Δimportance| 超过此值才写回（省写放大）。
_LIFECYCLE_MIN_DELTA = 0.01

# ADR-0020 T1.3: 派生台账前缀（从 Memory Hub 迁 Memory Index meta，consolidator 独占 Memory Index 写锁 -> 无抢锁、写入不失败）
_DISTILL_LEDGER_PREFIX = "distill/"
#: 已蒸话语 memo（2026-08-08 欠采修复）。**纯内容维度**，与 distill/ 台账分开：
#: 后者的 key 故意含 context_digest（同一句话在不同脉络该蒸出不同产物），
#: 拿它判"这句话蒸过没有"会次次 miss。这里只问内容，不问脉络。
_DISCOURSE_MEMO_PREFIX = "seen_discourse/"
_JUDGMENT_LEDGER_PREFIX = "judgment/"
#: G12.2 R5 新开/延续判定台账（ADR-0031 §14.1）。🔴 独立前缀，**不是** `judgment/`
#: 的子空间（"judgment_taskstate/" 不以 "judgment/" 开头，scan_judgments 天然扫不到
#: ——MS-17 命名空间共用事故的根治形态，verdict 白名单只是第二道保险）。
_TASKSTATE_JUDGMENT_PREFIX = "judgment_taskstate/"
_INDEX_JOURNAL_SEQ = 0
"""进程级单调序号（judgment 台账 seq，ADR-0020 T1.3）。"""


def _continuation_ref_text(turn: Any) -> str:  # noqa: ANN401 -- Turn（避免循环导入）
    """GM-1 的「上一轮」参照文本 = 该轮 user 文本 + assistant 回复。

    🔴 **必须含 assistant 回复**：「把 P1 详细说明」里的 P1 是上一轮**回复里**提出的，
    只比 user↔user 会把追问误判成新任务（MQ-S13，实测 8→6 / 10→6 / 7→3 三例同向）。
    """
    from bladex_core.query_understanding import last_user_text

    return f"{last_user_text(turn.request_messages)}\n{turn.response_text or ''}"


def _prev_ref(
    prev_key: str,
    cache: dict[str, tuple[str, bool]],
    ledger: MemoryHub,
) -> tuple[str, bool]:
    """取上一轮的 (参照文本, 是否 aux)。返回 `("", True)` = 不可用，按 aux 处理。

    本 pass 内处理过的轮已在 `cache` 里（免费）；**批边界外的前一轮付一次 `get()`**
    ——这条路径不能省：consolidator 每 60s 一批，而人说话的间隔通常大于 60s，
    绝大多数相邻轮天生落在不同批里。只对判为 TAIL_CONTINUATION 的轮触发，
    量级与 rebuild 本身每轮一次 `get()` 同阶。
    """
    hit = cache.get(prev_key)
    if hit is not None:
        return hit
    try:
        prev_turn = ledger.get(prev_key)
    except Exception as e:  # noqa: BLE001 -- 取不到前一轮只是少一个候选，不该中断重建
        logger.warning("index_attrib_cont_prev_get_failed", key=prev_key, error=str(e))
        prev_turn = None
    if prev_turn is None:
        cache[prev_key] = ("", True)      # fail-closed：拿不到就不做延续判定
        return cache[prev_key]
    from bladex_proxy.identity import classify_auxiliary

    is_aux, _rule = classify_auxiliary(prev_turn.request_messages)
    cache[prev_key] = ("" if is_aux else _continuation_ref_text(prev_turn), is_aux)
    return cache[prev_key]


def _goal_context(ledger: MemoryHub, ledger_key: str) -> tuple[str, str] | None:
    """取某一轮的 `(turn_class, 末条 user 原文)`，供 G12.3 goal 捕获。

    只在**新开一件事**时调用（live 85 张卡 vs 5103 轮），故一次 `get()` 的成本
    可接受——与 `_prev_ref` 同款懒取：**不预先把全库正文装进内存**
    （首轮 user 消息实测 max 240KB，全量装载是 O(库) 的内存）。

    取不到返回 None ⇒ 调用方记 `no_turn_context` 并留空 goal（fail-closed：
    宁可空着，也不拿一个来路不明的文本当用户目标）。
    """
    from bladex_core.query_understanding import last_user_text
    from bladex_proxy.identity import classify_turn_disposition

    try:
        turn = ledger.get(ledger_key)
    except Exception as e:  # noqa: BLE001 -- 取不到首轮只是少一个 goal，不该中断重建
        logger.warning("goal_ctx_get_failed", key=ledger_key, error=str(e))
        return None
    if turn is None:
        return None
    turn_class, _rule = classify_turn_disposition(turn.request_messages)
    return turn_class, last_user_text(turn.request_messages)


def _capture_matter_goal(matter: Matter, ledger_key: str, goal_ctx: Any = None) -> str:
    """G12.3 **唯一写入点**：新开卡那一刻取 goal。返回分档 reason（空 = 取到了）。

    🔴 判据 D1/D2：只在 `task_goal` 为空时写；写完之后**任何路径都不得改写**
    （模型能改目标 ⇒ 防漂移自我拆台）。这里的 `if matter.task_goal` 是给
    "将来有人把本函数挪到别处调用"留的防线，不是当前调用点需要。
    """
    from bladex_core.task_goal import (
        GOAL_ABSENT_NO_CONTEXT,
        GOAL_ABSENT_NO_KEY,
        GoalCapture,
        capture_first_turn_goal,
    )

    if matter.task_goal:
        return ""
    lk = ledger_key or ""
    if not lk:
        cap = GoalCapture(reason=GOAL_ABSENT_NO_KEY)
    else:
        ctx = goal_ctx(lk) if goal_ctx is not None else None
        if ctx is None:
            cap = GoalCapture(reason=GOAL_ABSENT_NO_CONTEXT, ledger_key=lk)
        else:
            _cls, _text = ctx
            cap = capture_first_turn_goal(
                ledger_key=lk, user_text=_text, turn_class=_cls)
    matter.task_goal = cap.text
    matter.task_goal_source = cap.source
    matter.task_goal_reason = cap.reason
    matter.task_goal_ledger_key = cap.ledger_key
    return cap.reason


def _turn_matches_filter(
    key_str: str,
    ts_iso: str,
    *,
    since: str = "",
    until: str = "",
    agents: set[str] | None = None,
    exclude_agents: set[str] | None = None,
) -> bool:
    """定向重建过滤（2026-07-28）：按 agent + 时间窗匹配一条 Memory Hub Turn。

    - agent 取自 Memory Hub key（``user/agent/session/entry_id`` 第二段），精确匹配。
    - since/until 与 turn.ts 的 ISO 字符串比较：since 含端（``ts >= since``）；
      until 为前缀含语义（``ts[:len(until)] <= until``），date-only 如
      ``"2026-07-24"`` 表示"含 07-24 全天"。
    - 设了时间过滤但 ts 缺失 → 不匹配（保守跳过，不误伤脏数据）。
    """
    parts = key_str.split("/")
    agent = parts[1] if len(parts) >= 2 else ""
    if agents and agent not in agents:
        return False
    if exclude_agents and agent in exclude_agents:
        return False
    if since or until:
        if not ts_iso:
            return False
        if since and ts_iso < since:
            return False
        if until and ts_iso[: len(until)] > until:
            return False
    return True


def _ledger_source_text_hash(source_text: str, context_digest: str = "") -> str:
    """蒸馏台账 key 的 hash（sha256[:16]）。

    M1-3（拍板 1）：`context_digest` 非空时 key = `sha256(digest + "\\x00" + 正文)`。

    为什么上下文要进 key：v4 带上下文蒸馏之后，**同一句话在不同脉络下该蒸出不同产物**
    （"继续" / "还是不行" / "按上面那个改"）。v3 的 key 只 hash 正文，
    这类消息在任何语境里都命中同一条缓存，复用出来的必然是别的语境的产物。

    分隔符用 `\\x00`：它不可能出现在正文里，因此 `(digest, text)` 到 key 的映射
    是单射的——拼接歧义（`"ab"+"c"` 与 `"a"+"bc"` 撞同一个 key）在这里不会发生。

    空 digest 走**旧算法逐字不变**：v3 台账继续命中，不因本次改版整代作废。
    """
    if not context_digest:
        return hashlib.sha256(source_text.encode()).hexdigest()[:16]
    raw = f"{context_digest}\x00{source_text}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _ledger_entry_id() -> str:
    """judgment 台账 seq（进程级单调，{unix_ms:013d}-{seq:08d}）。"""
    global _INDEX_JOURNAL_SEQ
    _INDEX_JOURNAL_SEQ += 1
    ts_ms = int(time.time() * 1000)
    return f"{ts_ms:013d}-{_INDEX_JOURNAL_SEQ:08d}"


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """cosine 相似度（未归属池消化用，T3）。"""
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _dedupe_aliases(title: str, entities: list[str]) -> list[str]:
    """构造新 Matter 初始 aliases（标题 + 实体，保序去重）。

    拍板 D（2026-08-14）：实体种子过毒词——此前 opencli/Chrome/'16%股份转让'/
    '张开利62.4' 这类工具名/数字碎片原样入 alias 成为 L3 全串匹配键
    （matter-dedup-keywords 卡实测）。标题永远保留。
    """
    from bladex_core.topic_keys import is_valid_topic_key, normalize_key

    out: list[str] = []
    seen: set[str] = set()
    for x in [title, *entities]:
        if not x or x in seen:
            continue
        if x != title and not is_valid_topic_key(normalize_key(x)):
            continue
        seen.add(x)
        out.append(x)
    return out


class FastEmbedAdapter:
    """fastembed 适配器 — multilingual-e5-large 本地推理（ADR-0008 §6.5）。

    可被 mock 替换（单测不跑真模型）。
    """

    def __init__(
        self,
        model_name: str = "",
        cache_dir: str | Path | None = None,
    ) -> None:
        # 默认模型统一由 embedding.DEFAULT_LOCAL_MODEL 定义（避免两处写死不一致）
        if not model_name:
            from bladex_proxy.embedding import DEFAULT_LOCAL_MODEL
            model_name = DEFAULT_LOCAL_MODEL
        self._model_name = model_name
        self._cache_dir = str(cache_dir) if cache_dir else None
        self._model: Any = None
        self._failed: bool = False
        self._fail_reason: str = ""

    @property
    def available(self) -> bool:
        """模型是否可用（加载成功或尚未尝试）。"""
        return self._model is not None or not self._failed

    def _ensure_model(self) -> None:
        if self._model is None:
            if self._failed:
                raise RuntimeError(
                    f"embedder model unavailable (previous load failed): {self._fail_reason}"
                )
            try:
                from fastembed import TextEmbedding
                kwargs: dict[str, Any] = {"model_name": self._model_name}
                if self._cache_dir:
                    kwargs["cache_dir"] = self._cache_dir
                # 注：模型在 fastembed 白名单内但缓存缺失时，这里会**阻塞下载**
                # （huggingface_hub，尊重 HF_ENDPOINT 镜像）；不在白名单直接抛。
                self._model = TextEmbedding(**kwargs)
            except Exception as e:
                self._failed = True
                self._fail_reason = str(e)
                hint = "Memory Index consolidation disabled until model is available"
                # 白名单外模型给可操作提示（如误配 e5-small——fastembed 无此模型）
                try:
                    from fastembed import TextEmbedding as _TE
                    supported = [m["model"] for m in _TE.list_supported_models()]
                    if self._model_name not in supported:
                        multi = [m for m in supported
                                 if "multilingual" in m.lower() or m.endswith("-zh")
                                 or "e5" in m.lower()]
                        hint = (f"model not in fastembed supported list; "
                                f"multilingual options: {multi}")
                except Exception:  # noqa: BLE001 — 提示尽力而为
                    pass
                logger.error("embedder_model_load_failed", error=str(e),
                             model=self._model_name, hint=hint)
                raise RuntimeError(f"embedder model load failed: {e} ({hint})") from e

    @property
    def _is_e5(self) -> bool:
        """e5 系模型需要 query:/passage: 前缀约定；其它白名单模型（MiniLM/mpnet/
        jina/bge）无前缀约定，原文直嵌。非 e5 模型 = 新向量空间（换模型本就要
        reembed_index），无历史兼容负担。"""
        return "e5" in self._model_name.lower()

    def embed(self, texts: list[str]) -> list[list[float]]:
        """把文本列表嵌入成向量列表（passage 语义，历史行为）。"""
        self._ensure_model()
        # e5 模型需要在文本前加 "query: " 或 "passage: " 前缀；非 e5 原文直嵌
        prefixed = [f"passage: {t}" for t in texts] if self._is_e5 else list(texts)
        return [list(e) for e in self._model.embed(prefixed)]

    def embed_passage(self, texts: list[str]) -> list[list[float]]:
        """passage 语义嵌入 = embed()（前缀 adapter 内部处理）。"""
        return self.embed(texts)

    def embed_query(self, texts: list[str]) -> list[list[float]]:
        """query 语义嵌入。

        e5 历史 quirk（刻意保持，勿"修"）：老调用方手拼 "query: {q}" 再进 embed()，
        实际嵌的是 "passage: query: {q}" 双前缀。novelty/digestion 等全部阈值
        都在此行为上标定——改前缀 = 全库 query/passage 相对分布漂移 = 阈值失效。
        因此 e5 逐字复刻："passage: query: {t}"；非 e5 模型原文直嵌。
        """
        self._ensure_model()
        prefixed = [f"passage: query: {t}" for t in texts] if self._is_e5 else list(texts)
        return [list(e) for e in self._model.embed(prefixed)]

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_identity(self) -> str:
        """向量空间身份标识（Memory Index model_id 不变量用）。"""
        return f"local:{self._model_name}"


class _LanceDBNoveltyChecker:
    """LanceDB 向量检索做 novelty 检查（G3 + 方案 B 实体感知）。

    替代全量加载 facts 到内存做 brute-force cosine。
    对每个候选 embedding 查 LanceDB top-k，cosine >= novelty_threshold -> 候选重复项。

    方案 B 实体感知：纯 cosine 对"同义不同实体"误判（e5: cos('北京天气','上海天气')
    =0.9652>0.95，不同实体被判重复丢弃）。升级：cosine 召回 top-k 候选后比实体集合，
    若任一高相似候选的实体集合与当前实体 Jaccard >= entity_overlap_threshold -> 判重；
    实体有差异 -> 仍新颖。entities 缺失（候选或表中）退化为纯 cosine（向后兼容）。
    """

    def __init__(
        self,
        table: Any,
        novelty_threshold: float,
        *,
        entity_aware: bool = True,
        entity_overlap_threshold: float = 0.6,
        topk: int = 5,
    ) -> None:
        self._table = table
        self._threshold = novelty_threshold
        self._entity_aware = entity_aware
        self._entity_overlap_threshold = entity_overlap_threshold
        self._topk = max(1, topk)
        # 探测表是否有 entities 列（旧表没有 -> 退化纯 cosine）
        self._has_entities_col = self._detect_entities_column()

    def _detect_entities_column(self) -> bool:
        """探测 facts 表是否有 entities 列（兼容旧 schema）。"""
        try:
            schema = self._table.schema
            return "entities" in schema.names
        except Exception:
            return False

    @staticmethod
    def _entity_jaccard(a: list[str] | None, b: list[str] | None) -> float:
        """两实体集合的 Jaccard 相似度（大小写归一化）。空集约定：返回 -1（表示无法判定，调用方退化纯 cosine）。"""
        sa = {x.strip().lower() for x in (a or []) if x and x.strip()}
        sb = {x.strip().lower() for x in (b or []) if x and x.strip()}
        if not sa or not sb:
            return -1.0
        inter = len(sa & sb)
        union = len(sa | sb)
        return inter / union if union else 0.0

    def neighbors(self, embedding: list[float], *, k: int = 10,
                  min_similarity: float = 0.90) -> list[tuple[dict, float]]:
        """M2-2：取 `min_similarity` 以上的近邻（裁决输入包的来源）。

        与 `is_novel` 分开是因为两者问的问题不同：
        判重问"有没有一模一样的"（阈值 0.98，只拦重发），
        裁决问"有没有可能冲突的"（阈值 0.90，要把对手方端上桌）。
        3.9 实测 0.90 触发率 62%、平均 4.5 条邻居 —— 正好一个裁决包。

        MS-3：检索前刷新表句柄，否则**跨批次写入的条目看不见**
        （07-25 遗留：LanceDB 无 reload，写进程内同样存在）。
        """
        try:
            self._refresh()
            rows = (self._table.search(embedding)
                    .distance_type("cosine").limit(max(1, k)).to_list())
        except Exception:  # noqa: BLE001 —— 拿不到邻居 = 无冲突可判，退化成 ADD
            return []
        out: list[tuple[dict, float]] = []
        for r in rows:
            sim = max(0.0, 1.0 - r.get("_distance", 0.0))
            if sim >= min_similarity:
                out.append((r, sim))
        return out

    def _refresh(self) -> None:
        """MS-3：刷新 LanceDB 读句柄（跨批次新鲜度）。

        `_LanceDBNoveltyChecker` 拿到的是**构造时**的表句柄快照。同一次 rebuild 里
        pass N 写进去的条目，pass N+1 的判重/裁决检索看不见——于是同一条事实
        在两个批次里各入库一次（Hermes 定时任务对的复现场景，E6 前半）。
        句柄没有 reopen 能力时静默跳过（旧 lancedb / 测试替身），行为不变。
        """
        fn = getattr(self._table, "checkout_latest", None)
        if fn is None:
            return
        try:
            fn()
        except Exception:  # noqa: BLE001 —— 刷新失败不该让判重整条挂掉
            pass

    def is_novel(self, embedding: list[float], entities: list[str] | None = None,
                 content: str = "") -> bool:
        try:
            # MS-3：判重同样要看得见跨批次写入（不刷新 = 同批第二次必判新颖）
            self._refresh()
            # MS-3：**恒取 top-k，不再退化到 top-1**。
            # 纯 cosine 判定确实只需要 top-1（它就是最大相似度），但同一批结果
            # 还喂着 E6.5 的标识符前置放行——它比对的是"高相似已存条目的标识符集合"。
            # limit=1 时那个集合只有一条，于是候选的标识符只要不在**最相似那一条**
            # 里出现就放行，哪怕它明明在第 3 条里。放宽 limit 让放行判据变**严**，
            # 正是 M0-4 收窄之后该走的方向。
            limit = max(1, self._topk)
            results = (self._table.search(embedding)
                       .distance_type("cosine").limit(limit).to_list())
            if not results:
                return True
            # ADR-0028 E6.5：标识符前置放行——候选带着高相似条目**没有**的标识符
            # → 直接判新颖。理由是实测：新旧探针 token 的 cosine = 0.9890，
            # 而 entities 是 LLM 给的（它心情好才带上 token），
            # 标识符抽取是确定性的，比"指望模型这次记得写"可靠。
            if content:
                from bladex_core.distill_fidelity import identifier_novelty_override
                near = [
                    str(r.get("content", "") or "") for r in results
                    if max(0.0, 1.0 - r.get("_distance", 0.0)) >= self._threshold
                ]
                if near and identifier_novelty_override(content, near):
                    return True
            # 无实体感知条件 -> 纯 cosine top-1 判定（原行为）
            if not (self._entity_aware and self._has_entities_col and entities):
                distance = results[0].get("_distance", 0.0)
                similarity = max(0.0, 1.0 - distance)
                return similarity < self._threshold
            # 实体感知：遍历 top-k 高相似候选，任一实体高度重叠 -> 判重
            for row in results:
                distance = row.get("_distance", 0.0)
                similarity = max(0.0, 1.0 - distance)
                if similarity < self._threshold:
                    continue  # 相似度不够，跳过
                row_entities = row.get("entities")
                # LanceDB list<string> 列返回 list 或 None
                if isinstance(row_entities, str):
                    row_entities = [row_entities]
                jac = _LanceDBNoveltyChecker._entity_jaccard(entities, row_entities)
                if jac < 0:
                    continue  # 任一方无实体，无法判定实体维度，看下一个候选
                if jac >= self._entity_overlap_threshold:
                    return False  # 实体高度重叠 + cosine 高 -> 判重
            # 所有高相似候选实体都有差异（或都无实体）-> 新颖
            return True
        except Exception:
            return True  # 出错时保守写入


def build_agent_claim_map(ledger: MemoryHub) -> dict[str, str]:
    """扫 AGENT_CLAIM 管理事件，构造 `旧 agent_id -> 现 agent_id` 读侧映射（G11.11）。

    与 ADR-0021 的 `legacy_map`（legacy user_id -> principal）同构，**但走独立的
    事件类型与独立的映射表**——两者不共用命名空间，撤销一方不会碰到另一方。

    🔴 **读侧映射，不改 Memory Hub**：`agent_id` 是 ledger key 的组成部分
    （`principal/agent/session/entry`），而 ledger 是追加式总账。改归属改的是
    派生层的解释，不是事实本身；日志里已经形成的 agent_id 同理不追改
    （append-only 的事实记录，改它才是伪造历史）。

    链式解析：事件按 journal 顺序应用，``a→b`` 之后再来 ``b→c``，则 ``a`` 最终
    映射到 ``c``。这让"认领后改名""改名后再合并"这类连续操作自然收敛，
    也让**撤销**只需再发一条反向 AGENT_CLAIM。
    """
    mapping: dict[str, str] = {}
    for _key, event in ledger.scan_admin_events():
        if event.event_type != "agent_claim":
            continue
        to_id = str(event.payload.get("to_agent_id", "") or "")
        if not to_id:
            continue
        for from_id in event.payload.get("from_agent_ids", []) or []:
            from_id = str(from_id)
            if not from_id or from_id == to_id:
                continue
            mapping[from_id] = to_id
        # 已指向 from_id 的旧映射要跟着走（链式收敛）
        for src, dst in list(mapping.items()):
            if dst in (event.payload.get("from_agent_ids") or []) and dst != to_id:
                mapping[src] = to_id
        # 🔴 目标名字**重新启用**：删掉以它为 key 的旧映射。
        #
        # 场景（Jason 2026-08-19 的两步操作）：先 `Pi → Pi-old`（老记忆改名让位），
        # 再 `unknown-b1d8a6f4 → Pi`（新桶用回这个名字）。此时 `Pi` 不再是"已被
        # 取代的旧名字"，而是现行名字。留着 `Pi → Pi-old` 会有两处后果：
        #   ① dashboard 把 `Pi` 当成已改名走的条目、整行不显示；
        #   ② 更严重——等 `Pi` 有了规则、真流量以 `local/Pi/...` 落库，
        #      重建时会把**新数据错划给 `Pi-old`**。
        mapping.pop(to_id, None)
    if mapping:
        logger.info("index_agent_claim_map_built", pairs=len(mapping))
    return mapping


class MemoryIndex:
    """Memory Index 派生层：事实存储 + LanceDB 向量索引。

    事实 metadata 存 RocksDB（与 Memory Hub 同级但独立实例），
    向量存 LanceDB（语义检索）。
    """

    def __init__(
        self,
        path: str | Path,
        embedder: EmbedderProtocol | None = None,
        *,
        read_only: bool = False,
        novelty_threshold: float | None = None,
        semantic_threshold: float = 0.85,
        digestion_threshold: float = _DEFAULT_DIGESTION_THRESHOLD,
        distiller: DistillerProtocol | None = None,
        catchup_throttle_s: float = _CATCHUP_THROTTLE_S,
        link_judge: LinkJudge | None = None,
        adjudicator: Any | None = None,
        top_candidates: int = 5,
        sensitivity_config: Any | None = None,
        entity_aware_novelty: bool = True,
        entity_overlap_threshold: float = 0.6,
        novelty_topk: int = 5,
        embed_model_id: str | None = None,
        entity_aware_rerank: bool = True,
        entity_rerank_alpha: float = 0.7,
        entity_rerank_expand_k: int = 20,
    ) -> None:
        self._path = Path(path)
        self._read_only = read_only
        self._catchup_throttle_s = catchup_throttle_s
        # ③ 上轮 rebuild 是否因 max_turns 截断（还有积压 -> 调用方应立即续跑不 sleep）
        self._last_rebuild_backlog = False
        # MQ-R1：上次 facts 压缩的时刻（节流用，见 _COMPACT_MIN_INTERVAL_S）
        self._last_fact_compact_monotonic = 0.0
        # MQ-I15 ②：上次版本 vacuum 的时刻（节流理由同上——清理窗口内判据仍超阈值）
        self._last_version_vacuum_monotonic = 0.0
        if not read_only:
            self._path.mkdir(parents=True, exist_ok=True)
        self._embedder = embedder
        # M4-2：漏斗指标接收器（consolidator 进程注入；proxy/测试不传 → NullFunnel）。
        # 用 `attach_funnel()` 后置注入而不是构造参数——MemoryIndex 的构造点有十几处，
        # 加一个必填参数等于逼所有调用方都改一遍，而它们绝大多数不关心观测。
        from bladex_core.funnel import NULL_FUNNEL

        self._funnel: Any = NULL_FUNNEL
        # embedding 后端可选化：向量空间身份（"local:<model>" / "api:<model>" / probe 结果）。
        # None = 不校验（单测/legacy 路径）。
        self._embed_model_id = embed_model_id
        # ADR-0014 L0: 阈值配置驱动（novelty 仍用于杀重发；semantic/digestion 保留配置位，
        # ADR-0018 后归属不再用 cosine 红线，仅 digestion 兜底/不变式校验用）
        # M2-2：默认值收编 flags（此前写死 0.95，于是 `BLADEX_NOVELTY_THRESHOLD`
        # 加了也没人读——机制与配置脱节的同一种腐化）。显式传参仍优先。
        self._novelty_threshold = (
            flag_number("BLADEX_NOVELTY_THRESHOLD")
            if novelty_threshold is None else novelty_threshold)
        self._semantic_threshold = semantic_threshold
        self._digestion_threshold = digestion_threshold
        # G12.3：goal 捕获分档计数（captured + GOAL_ABSENT_REASONS 各一档）。
        # 这不是内部统计——它就是"有多少卡是被机器文本开出来的"那把尺子的读数面，
        # 每轮 rebuild 末尾整段打出来（0 也打：恒为 0 与没在跑必须可区分）。
        self._goal_counters: Counter[str] = Counter()
        # 实体感知 novelty（方案 B）：cosine 召回 top-k 后比实体集合，实体有差异仍判新颖。
        self._entity_aware_novelty = entity_aware_novelty
        self._entity_overlap_threshold = entity_overlap_threshold
        self._novelty_topk = novelty_topk
        # 实体感知检索重排序（ADR-0024 T8a）：cosine 召回 k 后按实体重叠重排序
        self._entity_aware_rerank = entity_aware_rerank
        self._entity_rerank_alpha = entity_rerank_alpha
        self._entity_rerank_expand_k = entity_rerank_expand_k
        # _has_entities_col 的缓存（schema 在表生命周期内不变，§8.5）
        self._has_entities_col_cached: bool | None = None
        # T3b：实体->关联 fact 数（热门实体衰减用），TTL 缓存。
        # 计数只喂衰减项，容忍陈旧（幅度是平方衰减里的一个分母）。
        self._entity_freq_cache: dict[str, int] | None = None
        self._entity_freq_cached_at: float = 0.0
        # MS-13：本 run 的重建标识（run 起始时间戳）。judgment 台账逐条盖章，
        # 读侧据此把"当前 run 的判决"与跨重建累积的历史残渣分开。
        # full rebuild 入口会刷新（一次全量重建 = 一个新 rebuild_id）。
        from datetime import UTC as _UTC
        from datetime import datetime as _dtm
        self._rebuild_id: str = _dtm.now(_UTC).strftime("%Y%m%dT%H%M%S")
        # ANN 索引维护的最近一次尝试时刻（T8b B17：与"有没有新 turn"解耦）
        self._ann_last_maintain_ts: float = 0.0
        self._distiller = distiller
        # ADR-0018 §3.2: L4 链接裁决器 + top-k 召回（注入后归属走五层链接）
        self._link_judge = link_judge
        # M2-2：写入时裁决器（None = 回落 supersede + 冲突检测的旧形态）
        self._adjudicator = adjudicator
        self._top_candidates = top_candidates
        # ADR-0021 section 3.3c: Fact 血统继承用（consolidator 据此设 exposure_ceiling）。
        self._sensitivity_config = sensitivity_config
        self._meta_db: Rdict | None = None
        # G4: manual edge 反向索引缓存（target_key -> MatterEdge），惰性构建
        self._manual_edge_cache: dict[str, MatterEdge] | None = None
        self._lancedb: Any = None
        self._table: Any = None
        self._matter_table: Any = None  # ADR-0012: LanceDB matters 表
        # A1/T3: secondary 追新节流时间戳（monotonic）
        self._last_catchup_monotonic: float = 0.0
        # ADR-0028 E6.2：词法第二路（SQLite FTS5 / trigram）。惰性打开；
        # 打不开就整体禁用（degraded 不崩，dense 主路径不受影响）。
        self._fts: Any = None
        self._fts_tried = False
        # V-I3：keyword 倒排缓存（canonical key -> fact_ids；facts 行数为失效戳）
        self._kw_map: dict[str, list[str]] | None = None
        self._kw_map_stamp: int = -1
        # ADR-0028 E7.1：文件内容索引（独立 LanceDB 表，与 facts 分开）
        self._files_tbl: Any = None
        # 模型产出文件的本地副本（内容寻址；只有 origin=produced 用）
        self._blobs: Any = None

    def open(self) -> None:
        """打开 Memory Index 存储（RocksDB + LanceDB）。

        read_only=True 时 meta_db 以 **secondary** 模式打开（A1/T3 修复）：
        read_only 是打开时刻的静态快照，永远看不到 consolidator 进程之后写入的数据
        -> 记忆闭环随 proxy 进程寿命静默退化。secondary 模式 + try_catch_up_with_primary
        让 proxy 读端能追新 consolidator 的写入（秒级节流）。
        LanceDB 读句柄不自动看到新写，search 前 checkout_latest() 刷新读视图。
        """
        if self._meta_db is not None:
            return
        import rocksdict
        if self._read_only:
            meta_path = self._path / "meta_rocksdb"
            if not meta_path.exists():
                # 空库（consolidator 从未跑过）-- meta_db 保持 None；
                # search/get_fact 会惰性重试打开（A1/T3：全新安装不再终生空库）
                self._meta_db = None
            else:
                opts = rocksdict.Options()
                # A1/T3: secondary 模式 + try_catch_up_with_primary -> 能看到 consolidator
                # 进程之后写入（read_only 是静态快照，永远看不到新写 -> 记忆闭环退化）
                secondary_path = str(meta_path) + "_secondary"
                self._meta_db = rocksdict.Rdict(
                    str(meta_path), opts, None,
                    rocksdict.AccessType.secondary(secondary_path),
                )
        else:
            self._meta_db = Rdict(str(self._path / "meta_rocksdb"))
        try:
            import lancedb
            self._lancedb = lancedb.connect(str(self._path / "lancedb"))
        except Exception as e:
            logger.warning("index_lancedb_unavailable", error=str(e),
                           hint="Memory Index will run without vector search")
            self._lancedb = None

        # F1: 读路径也要打开已存在的 facts 表（否则 search 恒返回空）。
        # 表不存在时保持 _table=None（空库，search 返回 [] 是合理行为）。
        if self._lancedb is not None:
            try:
                self._table = self._lancedb.open_table("facts")
                logger.info("index_facts_table_opened", rows=self._table.count_rows())
                # 方案 B：旧表无 entities 列 -> 加列 + 从 metadata 回填（仅可写实例执行）
                if not self._read_only:
                    self._migrate_entities_column()
            except Exception:
                # 表还不存在（consolidator 还没跑过）
                self._table = None
            # ADR-0012: Matter 表（含质心向量）
            try:
                self._matter_table = self._lancedb.open_table("matters")
                logger.info("index_matters_table_opened", rows=self._matter_table.count_rows())
            except Exception:
                self._matter_table = None
        else:
            self._matter_table = None

        # embedding 后端可选化：向量空间 model_id 不变量校验（换模型必须先重嵌）
        self._check_embed_model_invariant()

        logger.info("index_opened", path=str(self._path),
                    lancedb=self._lancedb is not None,
                    table=self._table is not None,
                    secondary=self._read_only)

    def _write_embed_stamp_sidecar(self, model_id: str) -> None:
        """把向量空间身份同时写成纯文本侧车 `<index>/embed_model_id`。

        RocksDB 里的章仍是权威；侧车只为让启动前的探测（embedding._stored_local_model）
        **不必打开 RocksDB** —— 用自造 Options 开 live 库会改写 rocksdict 的
        `rocksdict-config.json` 从而让库打不开（2026-07-26 事故）。写失败不致命。
        """
        try:
            (self._path / "embed_model_id").write_text(model_id, encoding="utf-8")
        except Exception as e:  # noqa: BLE001 — 侧车是便利设施，失败不影响主流程
            logger.debug("index_embed_stamp_sidecar_write_failed", error=str(e))

    def _check_embed_model_invariant(self) -> None:
        """校验 Memory Index 库内向量空间与当前 embedder 是否同源。

        - 写模式：库未标记 -> 盖章当前 model_id；不一致 -> **拒绝启动**
          （RuntimeError，提示跑 scripts/reembed_index.py），防止混写污染向量空间。
        - 只读模式（proxy）：不一致 -> 禁用向量检索（embedder 置 None，
          search 降级返回空 -> 注入退硬规则），proxy 本体不拒启。
        """
        if self._embed_model_id is None or self._meta_db is None:
            return
        if not self._embed_model_id:
            # 🔴 2026-08-25 live 病例 6：**"还不知道"不等于"不一致"**。
            # IPC adapter 在握手前 `model_identity` 是空串，被当成"另一个模型"
            # ⇒ 只读端 `_embedder=None` ⇒ 整个 proxy 进程终生禁用向量检索
            # （6/6 轮 `facts_count=0`，用户侧只表现为"记忆全没了"）。
            # 空身份 = 无法判定，**不作负面判定**；写侧的盖章仍是最终防线
            # （只读端拿错向量空间的后果是召回变差，不是污染库）。
            logger.warning("index_embed_model_unknown", stored_check="skipped",
                           hint="embedder reported an empty model identity; vector "
                                "space invariant NOT evaluated (search stays enabled)")
            return
        try:
            raw = self._meta_db.get(_EMBED_MODEL_KEY.encode())
        except Exception:
            raw = None
        stored = raw.decode() if isinstance(raw, bytes) else raw
        if stored is None:
            if not self._read_only:
                self._meta_db[_EMBED_MODEL_KEY.encode()] = self._embed_model_id.encode()
                self._write_embed_stamp_sidecar(self._embed_model_id)
                logger.info("index_embed_model_stamped", model_id=self._embed_model_id)
            return
        if stored == self._embed_model_id:
            # 侧车缺失（老库/手工恢复）时补齐——供启动前免开库探测用
            if not self._read_only:
                self._write_embed_stamp_sidecar(stored)
            return
        if self._read_only:
            logger.error(
                "EMBED_MODEL_MISMATCH_READONLY",
                stored=stored, current=self._embed_model_id,
                hint="vector search disabled (inject falls back to hard rules); "
                     "run scripts/reembed_index.py or restore BLADEX_EMBED_* config",
            )
            self._embedder = None
            return
        raise RuntimeError(
            f"Memory Index embed model mismatch: store={stored!r} current={self._embed_model_id!r}. "
            f"Switching embedding backend/model requires full re-embed: "
            f"run `python scripts/reembed_index.py` (or revert BLADEX_EMBED_* config)."
        )

    def _migrate_entities_column(self) -> None:
        """方案 B 存量迁移：facts 表加 entities 列并从 metadata 回填。

        旧表无 entities 列 -> add_columns 加列 -> merge_insert 按 id 回填每个 fact 的
        entities（来自 metadata RocksDB）。幂等：已有列则跳过。失败不致命（novelty
        check 会降级纯 cosine）。
        """
        if self._table is None:
            return
        try:
            col_names = self._table.schema.names
        except Exception as e:
            logger.warning("index_migrate_entities_schema_read_failed", error=str(e))
            return
        if "entities" in col_names:
            return  # 已迁移
        logger.info("index_migrate_entities_start", hint="adding entities column to facts table")
        import pyarrow as pa
        try:
            # 加列（pa.field 形式；DataFusion SQL 不支持 list<string> 字面量）
            self._table.add_columns([pa.field("entities", pa.list_(pa.string()))])
            # lancedb 0.33 无 reload；重新 open_table 刷新 schema 句柄
            self._table = self._lancedb.open_table("facts")
        except Exception as e:
            logger.warning("index_migrate_entities_add_column_failed",
                           error=str(e),
                           hint="novelty check will degrade to pure cosine")
            return
        # 回填：从 metadata 读所有 fact 的 entities，按 id merge_insert
        try:
            facts = self.all_facts()
            if not facts:
                logger.info("index_migrate_entities_done", backfilled=0, reason="no_facts")
                return
            ids = [f.id for f in facts]
            ents = [list(f.entities or []) for f in facts]
            patch = pa.table({"id": ids, "entities": ents})
            self._table.merge_insert("id").when_matched_update_all().execute(patch)
            self._table = self._lancedb.open_table("facts")  # 刷新数据句柄
            # schema 已变（加了 entities 列），重置探测缓存
            self._has_entities_col_cached = True
            backfilled = sum(1 for e in ents if e)
            logger.info("index_migrate_entities_done",
                        total=len(facts), backfilled=backfilled)
        except Exception as e:
            logger.warning("index_migrate_entities_backfill_failed",
                           error=str(e),
                           hint="column added but backfill incomplete; "
                                "new facts will populate entities on insert")

    def close(self) -> None:
        if self._meta_db is not None:
            self._meta_db.close()
            self._meta_db = None
        self._matter_table = None
        if self._fts is not None:
            self._fts.close()
            self._fts = None
            self._fts_tried = False
        logger.info("index_closed")

    # ── A1/T3: 读端追新（secondary catch-up + LanceDB checkout_latest + 惰性重开）──

    def _ensure_meta_db(self) -> None:
        """meta_db None 时惰性重试打开（全新安装 consolidator 首轮写入后可见）。"""
        if self._meta_db is not None or not self._read_only:
            return
        meta_path = self._path / "meta_rocksdb"
        if not meta_path.exists():
            return
        try:
            import rocksdict
            opts = rocksdict.Options()
            secondary_path = str(meta_path) + "_secondary"
            self._meta_db = rocksdict.Rdict(
                str(meta_path), opts, None,
                rocksdict.AccessType.secondary(secondary_path),
            )
            logger.info("index_meta_db_lazy_opened", path=str(meta_path))
        except Exception as e:
            logger.warning("index_meta_db_lazy_open_failed", error=str(e))

    def attach_funnel(self, funnel: Any) -> None:
        """接上漏斗指标接收器（M4-2）。consolidator 进程在 open() 后调一次。"""
        self._funnel = funnel

    def library_kpis(self) -> dict[str, Any]:
        """MS-8：库存健康度快照（gauge 源）。

        与漏斗（流量计数）互补：漏斗答"这一轮发生了什么"，KPI 答"库现在什么样"。
        M5-3 总验收拿这组数字当对比分母（对比旧库 508 条 / 30% 噪声）。
        只读聚合，跑在空闲轮，失败不抛。
        """
        from collections import Counter

        facts = self.all_facts()
        matters = self.all_matters()
        kinds = Counter(getattr(f.item_kind, "value", str(f.item_kind)) for f in facts)
        prov = Counter(getattr(f, "provenance", "") or "unknown" for f in facts)
        st = Counter(getattr(m.status, "value", str(m.status)) for m in matters)
        invalidated = sum(1 for f in facts if getattr(f, "t_invalid", None) is not None)

        # file_ref 在**向量平面**的残留（应恒 0——E1.3 之后它不该进向量表）。
        # 用向量表 id 与 meta 对账，而不是数 meta 里的 file_ref 总数：
        # 后者本来就该 >0（fact 本体保留），只是不参与向量召回。
        file_ref_in_vec = zombies = -1
        try:
            self._lazy_open_facts_table()
            if self._table is not None:
                vec_ids = set(self._table.to_arrow().column("id").to_pylist())
                by_id = {f.id: f for f in facts}
                file_ref_in_vec = sum(
                    1 for i in vec_ids
                    if i in by_id and by_id[i].item_kind == ItemKind.FILE_REF)
                zombies = sum(1 for i in vec_ids if i not in by_id)
        except Exception as e:  # noqa: BLE001 —— 观测面绝不影响主体
            logger.debug("index_library_kpi_vector_scan_failed", error=str(e))

        provisional = st.get("provisional", 0)
        established = st.get("active", 0) + st.get("dormant", 0) + st.get("closed", 0)
        total_m = provisional + established
        return {
            "facts_by_kind": dict(kinds),
            "provenance": dict(prov),
            "matters_by_status": dict(st),
            "invalidated": invalidated,
            "promotion_rate": (established / total_m) if total_m else 0.0,
            "file_ref_in_vectors": file_ref_in_vec,
            "zombie_vectors": zombies,
        }

    def publish_library_kpis(self) -> None:
        """把 `library_kpis()` 推给漏斗接收器（MS-8）。空闲轮调用。"""
        from bladex_core import funnel as _f

        try:
            k = self.library_kpis()
        except Exception as e:  # noqa: BLE001
            logger.warning("index_library_kpi_failed", error=str(e))
            return
        for kind, n in k["facts_by_kind"].items():
            self._funnel.set_gauge(_f.LIB_FACTS, n, labels={"item_kind": kind})
        for prov, n in k["provenance"].items():
            self._funnel.set_gauge(_f.LIB_PROVENANCE, n, labels={"provenance": prov})
        for status, n in k["matters_by_status"].items():
            self._funnel.set_gauge(_f.LIB_MATTERS, n, labels={"status": status})
        self._funnel.set_gauge(_f.LIB_INVALIDATED, k["invalidated"])
        self._funnel.set_gauge(_f.LIB_PROMOTION_RATE, k["promotion_rate"])
        self._funnel.set_gauge(_f.LIB_FILE_REF_IN_VECTORS, k["file_ref_in_vectors"])
        self._funnel.set_gauge(_f.LIB_ZOMBIE_VECTORS, k["zombie_vectors"])

    def _lazy_open_facts_table(self) -> None:
        """_table None 时惰性重试打开 facts 表（consolidator 首轮写入后可见）。"""
        if self._table is not None or self._lancedb is None:
            return
        try:
            self._table = self._lancedb.open_table("facts")
            logger.info("index_facts_table_lazy_opened", rows=self._table.count_rows())
        except Exception:
            pass  # 表还不存在，下次再试

    def _lazy_open_matter_table(self) -> None:
        """_matter_table None 时惰性重试打开 matters 表。"""
        if self._matter_table is not None or self._lancedb is None:
            return
        try:
            self._matter_table = self._lancedb.open_table("matters")
            logger.info("index_matters_table_lazy_opened", rows=self._matter_table.count_rows())
        except Exception:
            pass

    def _try_catch_up(self) -> None:
        """secondary 模式下追新 consolidator 写入（秒级节流，不每次调）。

        🔴 MQ-I16：primary 实例（``read_only=False``，consolidator 主写端）**不得**调
        ``try_catch_up_with_primary()``——RocksDB 对 primary 直接抛
        ``Not implemented: Supported only by secondary instance``，此前每小时 ~1,400 条
        ``index_catch_up_failed`` 全是这条假失败，把 proxy 侧真正的追新失败淹掉。
        """
        if self._meta_db is None or not self._read_only:
            return
        now = time.monotonic()
        if now - self._last_catchup_monotonic < self._catchup_throttle_s:
            return
        self._last_catchup_monotonic = now
        try:
            self._meta_db.try_catch_up_with_primary()
        except Exception as e:
            logger.debug("index_catch_up_failed", error=str(e))

    def _refresh_lance_read(self) -> None:
        """LanceDB 读视图刷新：checkout_latest（让 search 看到 consolidator 新写）。"""
        for tbl in (self._table, self._matter_table):
            if tbl is None:
                continue
            try:
                tbl.checkout_latest()
            except Exception as e:
                logger.debug("index_lance_checkout_latest_failed", error=str(e))

    @property
    def meta_db(self) -> Rdict | None:
        if self._meta_db is None and not self._read_only:
            self.open()
        return self._meta_db

    # ── ADR-0020 T1.3: 派生台账读写（从 Memory Hub 迁 Memory Index meta，consolidator 独占写，无抢锁）──

    def append_distill(
        self, source_text: str, distill_model: str, prompt_ver: str,
        facts: list, matter_proposals: list, context_digest: str = "",
    ) -> str:
        """写蒸馏台账（write-if-absent，ADR-0018 §4.1，ADR-0020 迁 Memory Index）。

        M1-3：`context_digest` 非空时并入 key（拍板 1）。
        """
        h = _ledger_source_text_hash(source_text, context_digest)
        key = f"{_DISTILL_LEDGER_PREFIX}{h}/{distill_model}/{prompt_ver}"
        db = self.meta_db
        if db is None:
            return key
        if db.get(key.encode()) is None:
            rec = DistillRecord(
                source_text_hash=h, distill_model=distill_model, prompt_ver=prompt_ver,
                facts=list(facts), matter_proposals=list(matter_proposals),
            )
            db[key.encode()] = msgpack.packb(rec.model_dump(mode="json"), use_bin_type=True)
        return key

    def get_distill(
        self, source_text: str, distill_model: str, prompt_ver: str,
        context_digest: str = "",
    ) -> DistillRecord | None:
        """读蒸馏台账（命中复用，ADR-0020 迁 Memory Index）。

        M1-3：`context_digest` 非空时并入 key（拍板 1）。
        """
        h = _ledger_source_text_hash(source_text, context_digest)
        key = f"{_DISTILL_LEDGER_PREFIX}{h}/{distill_model}/{prompt_ver}"
        db = self.meta_db
        if db is None:
            return None
        raw = db.get(key.encode())
        if raw is None:
            return None
        return DistillRecord.model_validate(msgpack.unpackb(raw, raw=False))

    def discourse_seen(self, text: str, prompt_ver: str) -> bool:
        """这条话语在**任何**轮次里被蒸过吗（纯内容判定）。

        修的是 `_collect_candidates_v4` 原来「只蒸 discourse[-1]」的欠采：
        那假设每条话语都当过某轮的最后一条，全库实测 32% 从未当过。
        """
        db = self.meta_db
        if db is None:
            return False
        h = hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]
        return db.get(f"{_DISCOURSE_MEMO_PREFIX}{h}/{prompt_ver}".encode()) is not None

    def mark_discourse(self, text: str, prompt_ver: str, ledger_key: str = "") -> None:
        """标记这条话语已蒸。值存首次蒸它的轮次，供审计"它是在哪被吃进去的"。

        prompt_ver 进 key：换 prompt = 该重蒸一遍，与 distill/ 台账同款分代语义。
        """
        db = self.meta_db
        if db is None:
            return
        h = hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]
        db[f"{_DISCOURSE_MEMO_PREFIX}{h}/{prompt_ver}".encode()] = (
            ledger_key or "1").encode()

    def append_judgment(
        self, fact_id: str, candidates: list[dict], verdict: str, judge_model: str,
        *, summary_rewrite: str = "", reason: str = "",
        considered: list[str] | None = None,
    ) -> str:
        """追加 L4 裁决台账（ADR-0018 §4.1，ADR-0020 迁 Memory Index）。

        `considered` = 裁决时篮子里的 fact_id（2026-08-31）。与 `candidates`
        （= targets）严格分开，理由见 `JudgmentRecord.considered` 的注释。
        """
        seq = _ledger_entry_id()
        key = f"{_JUDGMENT_LEDGER_PREFIX}{fact_id}/{seq}"
        db = self.meta_db
        if db is None:
            return key
        rec = JudgmentRecord(
            fact_id=fact_id, candidates=candidates, verdict=verdict,
            judge_model=judge_model, summary_rewrite=summary_rewrite, reason=reason,
            considered=list(considered or []),
            rebuild_id=self._rebuild_id,   # MS-13：判决归属哪一轮 run
        )
        db[key.encode()] = msgpack.packb(rec.model_dump(mode="json"), use_bin_type=True)
        return key

    def _replay_supersede_judgments(self, present: set[str]) -> int:
        """全量重建时按台账恢复取代关系。返回重放条数（ADR-0012 §3.6 修订项）。

        决策在 core（`plan_supersede_replay`，四条安全阀 + 先到先得语义），
        这里只负责**一次扫表**与落盘 —— 与 `supersede_plan` 同一分工。

        🔴 **一次扫表，不按 fact 逐条查**：`scan_judgments(fact_id)` 的实现是
        全表 `db.items()` 遍历后按前缀过滤，逐条查 = O(n²)（24308 条判决 ×
        每 fact 一次）。这也是探针慢的原因，别把它复制进热路径。
        """
        from bladex_core.adjudication import plan_supersede_replay

        rows: list[tuple[str, str, list[str], Any]] = []
        for key, rec in self.scan_judgments():          # 一次全表
            if not (getattr(rec, "verdict", "") or "").startswith("consolidation:update"):
                continue
            targets = [c.get("fact_id", "") for c in (rec.candidates or [])
                       if isinstance(c, dict) and c.get("fact_id")]
            if not targets:
                continue
            rows.append((key.rsplit("/", 1)[-1], rec.fact_id, targets, rec.ts))

        plan, skipped = plan_supersede_replay(rows, present)
        applied = 0
        for tid, (cand_id, ts) in plan.items():
            old = self.get_fact(tid)
            if old is None:
                skipped["target_absent"] = skipped.get("target_absent", 0) + 1
                continue
            old.t_invalid = ts
            old.superseded_by = cand_id
            old.updated_at = ts
            old.embedding = None                        # 只 upsert meta，不重写向量
            try:
                self.add_fact(old)
                applied += 1
            except Exception as e:  # noqa: BLE001
                logger.warning("index_supersede_replay_apply_failed",
                               fact_id=tid, error=str(e))
        logger.info("index_rebuild_supersede_replayed",
                    scanned=len(rows), applied=applied, skipped=skipped,
                    hint="restored supersede links from the judgment ledger "
                         "(zero LLM). Disable with BLADEX_SUPERSEDE_REPLAY_ENABLED=0")
        return applied

    def scan_judgments(
        self, fact_id: str | None = None,
    ) -> list[tuple[str, JudgmentRecord]]:
        """扫描裁决台账（按 fact_id 前缀或全部，ADR-0020 迁 Memory Index）。"""
        db = self.meta_db
        if db is None:
            return []
        prefix = f"{_JUDGMENT_LEDGER_PREFIX}{fact_id}/" if fact_id else _JUDGMENT_LEDGER_PREFIX
        out: list[tuple[str, JudgmentRecord]] = []
        for key_raw, value_raw in db.items():
            key_str = key_raw.decode() if isinstance(key_raw, bytes) else str(key_raw)
            if not key_str.startswith(prefix):
                continue
            out.append(
                (key_str, JudgmentRecord.model_validate(msgpack.unpackb(value_raw, raw=False)))
            )
        return out

    # ── G12.2 R5 新开/延续判定台账（ADR-0031 §14.1，决策表 v2 §6）────────────

    def append_taskstate_judgment(self, rec: TaskStateJudgment) -> str:
        """追加一条新开/延续判定（append-only；写侧 = consolidator 独占，同 L4 台账）。

        每条 R5 判定一条（≈ 每事一条 + 少量否定判决，不是每轮一条）；
        rebuild_id 由本 index 的当前 run 盖章（MS-13 纪律，调用方不用管）。
        """
        if rec.verdict not in TASKSTATE_VERDICTS:
            raise ValueError(
                f"taskstate verdict 必须是 {TASKSTATE_VERDICTS} 之一，收到 {rec.verdict!r}"
                " —— 封闭枚举，别把 L4 的 link:/none/uncertain 写进来（MS-17）")
        seq = _ledger_entry_id()
        key = f"{_TASKSTATE_JUDGMENT_PREFIX}{rec.turn_key}/{seq}"
        db = self.meta_db
        if db is None:
            return key
        if not rec.rebuild_id:
            rec = rec.model_copy(update={"rebuild_id": self._rebuild_id})
        db[key.encode()] = msgpack.packb(rec.model_dump(mode="json"), use_bin_type=True)
        return key

    def get_taskstate_judgment(self, turn_key: str) -> TaskStateJudgment | None:
        """读某轮**最新**（最大 seq）的新开/延续判定。无 → None。

        🔴 重放语义 = **台账优先**：调用方（G12.2 R5）先问这里，有记录读记录、
        **不重判**——这就是"改旋钮后重放，已有事 matter_id 零漂移"守卫的机制半边
        （另半边 = 旋钮值变更按约束 B 攒窗口）。
        """
        prefix = f"{_TASKSTATE_JUDGMENT_PREFIX}{turn_key}/"
        db = self.meta_db
        if db is None:
            return None
        latest_key: str | None = None
        latest_raw = None
        for key_raw, value_raw in db.items():
            key_str = key_raw.decode() if isinstance(key_raw, bytes) else str(key_raw)
            if not key_str.startswith(prefix):
                continue
            if latest_key is None or key_str > latest_key:
                latest_key = key_str
                latest_raw = value_raw
        if latest_raw is None:
            return None
        return TaskStateJudgment.model_validate(msgpack.unpackb(latest_raw, raw=False))

    def scan_taskstate_judgments(self) -> list[tuple[str, TaskStateJudgment]]:
        """全量扫描（审计/探针用；误延续归因读 anchor + window_snapshot 就靠它）。"""
        db = self.meta_db
        if db is None:
            return []
        out: list[tuple[str, TaskStateJudgment]] = []
        for key_raw, value_raw in db.items():
            key_str = key_raw.decode() if isinstance(key_raw, bytes) else str(key_raw)
            if not key_str.startswith(_TASKSTATE_JUDGMENT_PREFIX):
                continue
            out.append((key_str,
                        TaskStateJudgment.model_validate(msgpack.unpackb(value_raw, raw=False))))
        return out

    def add_fact(self, fact: Fact) -> None:
        """存一条 Fact（metadata + 向量）。"""
        # 存 metadata（不含 embedding）
        meta = fact.model_dump(mode="json", exclude={"embedding"})
        key = f"{_FACT_PREFIX}{fact.id}"
        self.meta_db[key.encode()] = msgpack.packb(meta, use_bin_type=True)

        # 存向量到 LanceDB
        # ADR-0028 E1.3（F2 止血）：file_ref 退出向量检索平面——只写 meta，不写 LanceDB。
        # fact 本体保留（Matter 归属 / supersede 取代键 / hash 失效）全部不受影响，
        # 只是不再参与散点向量召回（裸路径侵占召回位）。正确形态见 E7.1 文件内容索引。
        if fact.item_kind == ItemKind.FILE_REF:
            logger.debug("index_fact_vector_skipped_file_ref", fact_id=fact.id)
        elif fact.embedding is not None and self._lancedb is not None:
            self._add_to_lancedb(fact)

        # ADR-0028 E6.2：词法倒排同步（单写者 = consolidator；file_ref 不入、
        # t_invalid 非空则删——与向量平面同边界）。
        if not self._read_only:
            fts = self._ensure_fts()
            if fts is not None:
                fts.upsert(fact)

        logger.debug("index_fact_added", fact_id=fact.id,
                      content_len=len(fact.content))

    def _ensure_table(self, dim: int) -> Any:
        """确保 LanceDB table 存在（惰性创建）。"""
        if self._table is not None:
            return self._table

        import pyarrow as pa

        schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
            pa.field("content", pa.string()),
            pa.field("category", pa.string()),
            pa.field("source_session", pa.string()),
            pa.field("source_user_id", pa.string()),
            # 方案 B：实体列供 novelty check 实体感知去重（Jaccard 比对）
            pa.field("entities", pa.list_(pa.string())),
        ])

        try:
            self._table = self._lancedb.create_table(
                "facts", schema=schema, mode="create",
            )
            # 新建表的 schema 一定含 entities 列，重置探测缓存
            self._has_entities_col_cached = True
        except Exception:
            # table 已存在（可能是旧表无 entities 列），不预设缓存，留给 _has_entities_col 探测
            self._table = self._lancedb.open_table("facts")

        return self._table

    def _add_to_lancedb(self, fact: Fact) -> None:
        """把 fact 的向量加入 LanceDB。"""
        import pyarrow as pa

        dim = len(fact.embedding)
        table = self._ensure_table(dim)

        data = pa.table({
            "id": [fact.id],
            "vector": [fact.embedding],
            "content": [fact.content],
            "category": [fact.category],
            "source_session": [fact.source_session],
            "source_user_id": [fact.source_user_id],
            "entities": [list(fact.entities or [])],
        })
        table.add(data)

    def _rerank_by_entities(
        self,
        query: str,
        facts: list[Fact],
        top_k: int,
    ) -> list[Fact]:
        """实体感知重排（ADR-0024 T8a）。

        口径对齐：Fact.entities 是 consolidator 蒸馏时由 LLM 抽出的判别性命名实体
        （"泰山啤酒"、"世界杯"、"glm-5.2"），不是机械分词碎片。query 侧热路径没有预算
        再跑一次 LLM 抽实体，所以不做 fact-vs-fact 那样的集合运算（Jaccard）。

        解法（子串包含，query 无需抽实体）：对每个候选 fact，检查它的 entities 是否作为
        子串出现在 query 文本里。命中 = 该 fact 谈论的实体正是用户在问的东西，给融合加分。

            score = alpha * cosine + (1 - alpha) * hit_ratio
            hit_ratio = 命中的 entity 数 / 该 fact 的 entity 总数（[0,1]）

        子串包含天然中英文通吃：
          - 中文 query "泰山啤酒破产案有3000万共益债的公告吗" 包含 entity "泰山啤酒"
          - 英文 query "how to optimize BladeX Memory Index" 包含 entity "BladeX"
        不引入 NER/LLM，与 fact 侧 entities 同口径（直接用 fact.entities 原值，query 侧
        不做任何抽取/转换），延迟 = O(候选数 * 平均 entity 数 * 字符串查找)，可忽略。

        退化约定：
          - 候选都无 entity -> 命中分全 0 -> 等价纯 cosine 原序（稳定排序保证不抖动）
          - 开关关闭 -> 调用方不进入本方法
        """
        if not facts:
            return facts
        q_lower = query.lower()
        for fact in facts:
            ents = fact.entities or []
            if not ents:
                # 该 fact 无 entity，命中分 0，融合退化为纯 cosine（乘 alpha）
                fact._entity_hit = 0.0
                continue
            hit = 0
            for e in ents:
                if not e:
                    continue
                if e.lower() in q_lower:
                    hit += 1
            fact._entity_hit = hit / len(ents)
        alpha = self._entity_rerank_alpha
        # 融合：alpha 主导 cosine，(1-alpha) 给实体命中作修正
        for fact in facts:
            fact._score = alpha * getattr(fact, "_cosine_score", fact._score) + \
                          (1.0 - alpha) * fact._entity_hit
        # 稳定排序：融合分降序，分相同时保持 cosine 召回原序（不抖动）
        facts.sort(key=lambda f: f._score, reverse=True)
        return facts[:top_k]

    def _has_entities_col(self) -> bool:
        """检测 LanceDB 表是否已有 entities 列（存量迁移后才有）。

        schema 在表生命周期内不变，缓存一次避免每次 search 都查（§8.5）。
        """
        if self._has_entities_col_cached is not None:
            return self._has_entities_col_cached
        try:
            if self._table is None:
                self._has_entities_col_cached = False
            else:
                self._has_entities_col_cached = "entities" in self._table.schema.names
        except Exception:
            self._has_entities_col_cached = False
        return self._has_entities_col_cached

    def search(
        self,
        query: str,
        top_k: int = 10,
        user_id: str | None = None,
        visibility: list[str] | None = None,
        q_vec: list[float] | None = None,
        nprobes: int | None = None,
        identifiers: list[str] | None = None,
        session_id: str = "",
        explain: bool = False,
    ) -> list[Fact]:
        """语义检索：query → 向量 → LanceDB top-k → Fact 列表。

        user_id 非空时只检索该 user 的 Fact（跨 session、不串号）。

        q_vec 非空时**跳过 embedding**，直接用调用方算好的向量（热路径复用，
        2026-07-28 事故修复）：一次注入原本要 embed 两次（Matter 召回一次、
        Fact 召回一次），e5-large 单次 ~70ms，导致 e2e 151–157ms 稳定超
        150ms 预算 → 每轮静默降级成只注硬规则、记忆几乎从不生效。

        `explain=True`（S4，G9.3）：给**每条参与打分的候选**挂 `fact._explain`
        分数分解（`{rrf, base, importance_term, recency_term, audience_bonus,
        discount, final, entity_key, entity_hit, dropped_at}`），并把整张表放进 `self.last_explain`
        ——包括**被淘汰的候选**（`dropped_at=fold|quota|mmr`），那才是归因要看的。
        默认 False，输出逐字不变（守卫 `test_g93_explain.py`）。
        """
        # A1/T3: 惰性重试 + 追新（consolidator 写入后不重启 proxy 即可见）
        self._lazy_open_facts_table()
        self._ensure_meta_db()
        if self._embedder is None or self._lancedb is None or self._table is None:
            logger.warning("index_search_unavailable", hint="embedder or lancedb not initialized")
            # ADR-0027 §3.3：Memory Index 不可用 = 散点召回整段失效，用户侧只表现为"没记住"
            from ..metrics import INDEX_UNAVAILABLE, record_degradation
            record_degradation(INDEX_UNAVAILABLE)
            return []
        self._try_catch_up()
        self._refresh_lance_read()

        # 嵌入 query（前缀责任在 adapter；旧 mock 走 compat 降级，行为逐字不变）
        # q_vec 已由调用方算好则复用，省掉本轮第二次 embedding。
        if q_vec is None:
            q_vec = embed_query_compat(self._embedder, [query])[0]

        pids = self._visibility_pids(visibility, user_id)
        # 显式空可见集合（仅 team/org、无 personal--本卡无对应 fact）-> fail-closed
        if pids is not None and not pids:
            logger.info("index_search_empty_visibility", query_len=len(query))
            return []

        # ADR-0024 T8a：实体感知重排序时，扩大召回 k 再重排（如果开启）
        expand_k = top_k
        if self._entity_aware_rerank and self._has_entities_col():
            expand_k = self._entity_rerank_expand_k
            # 保证 expand_k >= top_k，否则浪费展开
            if expand_k < top_k:
                expand_k = top_k

        search_builder = self._table.search(q_vec).distance_type("cosine")
        search_builder = self._apply_ann_to_search(search_builder, nprobes, "facts")
        if pids:
            in_list = ",".join("'" + p.replace("'", "''") + "'" for p in pids)
            cond = f"source_user_id IN ({in_list})"
            # U6（ADR-0026 §6.1）：BLADEX_BOUNDARY_FILTER=1 时 **prefilter**——先按 scope 过滤
            # 再向量搜，解决多用户/多主题共表的「后过滤淹没」（某 scope 的 fact 挤不进全局
            # top-k → 散点召回恒空，2026-08-02 基线实测 results=0 达 95%）。默认关 = 现状后过滤
            # （回归逐字一致）；prefilter kwarg 若当前 LanceDB 版本不支持则自动回退后过滤、不崩。
            if os.environ.get("BLADEX_BOUNDARY_FILTER", "") == "1":
                try:
                    search_builder = search_builder.where(cond, prefilter=True)
                except TypeError:
                    logger.warning("index_prefilter_unsupported_fallback_postfilter")
                    search_builder = search_builder.where(cond)
            else:
                search_builder = search_builder.where(cond)
        results = search_builder.limit(expand_k).to_list()

        # 从 metadata 还原完整 Fact，附带 _distance 供相关性阈值过滤（F2）
        # U5.5（ADR-0026 §5 机制5）：低 importance 退出召回（遗忘=退出召回，不删数据）。
        # BLADEX_MIN_IMPORTANCE>0 时生效；默认 0=不过滤（零回归）。
        # ADR-0028 E2.1：默认值收编 flags.py（默认 0.2，遗忘=退出召回不删数据）。
        _min_imp = flag_number("BLADEX_MIN_IMPORTANCE")
        # U6 有限跳（ADR-0026 §6.4）：SUPERSEDES 反向替换——dense 命中了已被取代的
        # 旧条（current-only 会滤掉）时，把取代它的新条补进候选（BLADEX_HOP_EXPAND=1）。
        # 默认开（ADR-0027 §5.4）；BLADEX_HOP_EXPAND=0 / BLADEX_SOFT_SCORING=0 回滚。
        _hop_enabled = flag_enabled("BLADEX_HOP_EXPAND")
        _soft_enabled = flag_enabled("BLADEX_SOFT_SCORING")
        _supersede_swaps: list[str] = []
        facts: list[Fact] = []
        for row in results:
            fact_id = row.get("id", "")
            fact = self.get_fact(fact_id)
            if fact:
                # U6/U5.2（ADR-0026 §6.1 轴C current-only）：被取代/失效的条目退出召回
                # （t_invalid 非空）。数据仍在 meta/Memory Hub，as-of 可见。存量 fact t_invalid=None
                # → 不过滤任何东西（回归安全，无需开关）。
                if getattr(fact, "t_invalid", None) is not None:
                    if _hop_enabled and getattr(fact, "superseded_by", ""):
                        _supersede_swaps.append(fact.superseded_by)
                    continue
                # U5.5：低 importance 退出召回资格（gated）。
                if _min_imp > 0 and getattr(fact, "importance", 1.0) < _min_imp:
                    continue
                # LanceDB cosine 度量返回 _distance = 1 - cosine_similarity
                # 换算成相似度分数：similarity = 1 - distance
                distance = row.get("_distance", 0.0)
                fact._score = max(0.0, 1.0 - distance)
                # 保留原始 cosine 分数供融合重排（T8a）
                fact._cosine_score = fact._score
                facts.append(fact)

        # ADR-0024 T8a：实体感知重排序（cosine + entity 命中融合）
        # 见 _rerank_by_entities（子串包含，query 无需抽实体，中英文通吃）。
        # 融合重排（E6.3）开启时跳过 T8a，避免双重重排。
        reranked = False
        if (self._entity_aware_rerank and self._has_entities_col() and facts
                and not _soft_enabled):
            facts = self._rerank_by_entities(query, facts, top_k)
            reranked = True

        # ── ADR-0028 E6.2/E6.3：词法第二路 + RRF 融合重排 ─────────────────
        # 取代"全表 substring 关键词通道 + 分值加权软打分"：
        #   ① 词法走 SQLite FTS5 真倒排（BM25），标识符按 phrase 精确匹配
        #      —— `bxe3663a38` vs `bx3a984bcd` 的 cosine 是 0.9890，dense 分不开；
        #   ② 融合用**秩**不用分值（RRF）—— 库内 cosine 噪声底 0.88–0.93、
        #      top-15 极差仅 0.024–0.060，分值加权那套没有一路信号能对抗噪声。
        kw_added = 0
        soft_scored = False
        if _soft_enabled:
            facts, fuse_info = self._fuse_rerank(
                query, facts, top_k, pids, identifiers=identifiers,
                session_id=session_id, explain=explain,
            )
            soft_scored = True
            kw_added = fuse_info.get("channels", {}).get("lexical", 0)
            if explain:
                # 整张表（含被淘汰候选）留在实例上——返回值只有幸存者，
                # 而"为什么没进来"恰恰是归因要问的那一半。
                self.last_explain = fuse_info.get("explain", {})
                for _f in facts:
                    _f._explain = self.last_explain.get(_f.id)

        # U6 全量（ADR-0026 §6.4）：有限跳——边界内 1 跳扩展（同 Matter 兄弟 /
        # SUPERSEDES 反向替换 / PART_OF 父脉络），预算 = top_k//2（gated）。
        hop_added = 0
        if _hop_enabled and (facts or _supersede_swaps):
            facts, hop_added = self._hop_expand(
                facts, top_k, pids, _supersede_swaps, _min_imp,
            )

        # 2026-07-29 复核补：重排此前在生产日志里零痕迹（209 次 index_search_done
        # 无法回答"这轮是否重排、命中了几个实体"）。entity_hits = 本轮 top-k 里
        # entity 命中率 > 0 的 fact 数，可 grep。
        entity_hits = (
            sum(1 for f in facts if getattr(f, "_entity_hit", 0.0) > 0.0)
            if (reranked or soft_scored) else 0
        )
        logger.info("index_search_done", query_len=len(query),
                     results=len(facts), reranked=reranked,
                     expand_k=expand_k, entity_hits=entity_hits, kw_added=kw_added,
                     soft_scored=soft_scored, hop_added=hop_added)
        return facts

    # T3b：全库实体关联计数的 TTL（秒）。热路径不能每次 search 扫库，
    # 计数漂移慢（衰减项分母），10 分钟粒度足够。
    _ENTITY_FREQ_TTL_S = 600.0

    def _entity_freq(self) -> dict[str, int]:
        """实体 -> 关联 fact 数（热门实体衰减的 N，T3b）。

        取数路径（任务卡"FTS entities 列或 meta 聚合，实现取便宜者"）：
        LanceDB facts 表 entities 列**列扫**——比 meta 全量 msgpack 解码便宜
        一个量级；FTS 的 entities 列是空格拼接文本（多词实体会被拆散），
        解析歧义，不取。计数含 t_invalid 残留行：它只作衰减分母，
        高频判定不受影响。失败/无表 -> 空表 = 不衰减（graceful）。
        """
        import time as _time
        now = _time.monotonic()
        if (self._entity_freq_cache is not None
                and now - self._entity_freq_cached_at < self._ENTITY_FREQ_TTL_S):
            return self._entity_freq_cache
        freq: dict[str, int] = {}
        try:
            if self._table is not None and self._has_entities_col():
                try:
                    # 列投影：不把 1024 维向量整表拉进内存
                    tbl = self._table.to_lance().to_table(columns=["entities"])
                except Exception:  # noqa: BLE001 —— 版本差异，回退整表
                    tbl = self._table.to_arrow()
                col = tbl.column("entities") if "entities" in tbl.schema.names else None
                if col is not None:
                    for ents in col.to_pylist():
                        if not ents:
                            continue
                        if isinstance(ents, str):
                            ents = [ents]
                        for e in ents:
                            if e:
                                k = str(e).strip().lower()
                                freq[k] = freq.get(k, 0) + 1
        except Exception as e:  # noqa: BLE001
            logger.warning("index_entity_freq_failed", error=str(e))
        self._entity_freq_cache = freq
        self._entity_freq_cached_at = now
        return freq

    def _fuse_rerank(
        self, query: str, dense_facts: list[Fact], top_k: int,
        pids: list[str] | None, *, identifiers: list[str] | None = None,
        session_id: str = "", explain: bool = False,
    ) -> tuple[list[Fact], dict[str, Any]]:
        """E6.3 融合重排（取代分值加权的软打分五信号）。

        四路召回进 RRF：
          - **dense**：本函数入参（LanceDB 向量 top-expand_k）；
          - **lexical**：SQLite FTS5 / BM25（E6.2），标识符按 phrase 精确匹配；
          - **session**：同 source_session 的近 7 日 fact（E6.4 层级召回的一路）；
          - **entity**（T3b，2026-08-10）：fact.entities 子串命中 query
            （T8a 匹配口径迁移；热门实体衰减进通道内排序键）。
            08-04 soft 默认开以来 T8a 被跳过而融合无 entity 通道，
            实体信号实际退出了排序——本路把它接回。

        融合后按 `bladex_core.fusion` 走：修正项（importance/recency）→ 同义折叠
        → 类型配额 → MMR。分值不可比的问题由"用秩不用分值"正面解掉，
        `min_relevance` 绝对阈值随之退役。
        """
        from datetime import UTC as _UTC
        from datetime import datetime as _dtt

        from bladex_core.fusion import Channel, build_entity_channel, fuse_and_rerank

        facts_by_id: dict[str, Fact] = {f.id: f for f in dense_facts}
        channels = [Channel("dense", [f.id for f in dense_facts])]

        # ② 词法路（FTS5 BM25）
        lexical_ids: list[str] = []
        fts = self._ensure_fts()
        if fts is not None and fts.available:
            for fid, _rank in fts.search(query, identifiers=identifiers):
                f = facts_by_id.get(fid) or self.get_fact(fid)
                if f is None or not self._admissible(f, pids):
                    continue
                facts_by_id.setdefault(fid, f)
                lexical_ids.append(fid)
        if lexical_ids:
            channels.append(Channel("lexical", lexical_ids))

        # ③ session 路（E6.4：同会话近 7 日，权重同 dense）
        session_ids: list[str] = []
        if session_id:
            cutoff = _dtt.now(_UTC).timestamp() - 7 * 86400
            for f in dense_facts:
                if f.source_session == session_id:
                    session_ids.append(f.id)
            for f in facts_by_id.values():
                if f.id in session_ids or f.source_session != session_id:
                    continue
                created = getattr(f, "created_at", None)
                try:
                    _c = created if created.tzinfo else created.replace(tzinfo=_UTC)
                    if _c.timestamp() >= cutoff:
                        session_ids.append(f.id)
                except (TypeError, AttributeError):
                    continue
        if session_ids:
            channels.append(Channel("session", session_ids))

        # ④ entity 路（T3b）：候选池（dense∪lexical∪session）内子串命中，
        # 热门实体衰减用全库关联计数（TTL 缓存，见 _entity_freq）。
        # query 无实体命中 -> build 返回 None -> 三通道行为与改造前逐字一致。
        from bladex_core.flags import flag_number as _fn
        entity_ch = build_entity_channel(
            query, facts_by_id,
            entity_freq=self._entity_freq(),
            coef=_fn("BLADEX_ENTITY_DECAY_COEF"),
        )
        if entity_ch is not None:
            channels.append(entity_ch)

        # 向量（同义折叠 + MMR 用）：只取已在候选集里的，零额外 embedding。
        # MQ-R6：**一次**批量取，不是每个候选一次——后者是 40 次全表物化 / 207ms。
        vectors: dict[str, list[float]] = {
            fid: list(v) for fid, v in self._get_fact_vectors(facts_by_id).items()
        }

        now = _dtt.now(_UTC)
        ages: dict[str, float] = {}
        for fid, f in facts_by_id.items():
            created = getattr(f, "t_observed", None) or getattr(f, "created_at", None)
            try:
                _c = created if created.tzinfo else created.replace(tzinfo=_UTC)
                ages[fid] = max(0.0, (now - _c).total_seconds() / 86400.0)
            except (TypeError, AttributeError):
                ages[fid] = 0.0

        # T3-A：profile_obs 限额参与 + 排序折价（flag 接线，与 entity coef 同款；
        # QUOTA=0 即回退整类禁注的旧行为）。
        out, info = fuse_and_rerank(
            facts_by_id, channels, top_k, vectors=vectors, ages_days=ages,
            max_profile_obs=int(_fn("BLADEX_QUOTA_PROFILE_OBS")),
            profile_obs_discount=_fn("BLADEX_PROFILE_OBS_DISCOUNT"),
            # 2026-08-20：同义折叠是注入丢记忆的主因（e5 前 5 名流失里 80.7% 死在
            # 折叠，`rank` 那档 = 0）。阈值从 fusion 常量 0.92 提到 flag 默认 0.97，
            # 依据见 `docs/benchmarks/channel-votes-20260820.md` §11（曲线拐点 +
            # 人工签收分界两条独立证据）。接线方式与 T3-A 同款：调用方读 flag，
            # fusion 保持纯函数。回滚 = `BLADEX_FOLD_COSINE=0.92`。
            fold_cosine=_fn("BLADEX_FOLD_COSINE"),
            explain=explain,
        )
        logger.info("index_search_fused", channels=info["channels"],
                    scored=info["scored"], folded=info["folded"], kept=len(out),
                    fold_cosine=info.get("fold_cosine"))
        return out, info


    def backfill_fts(self, *, force: bool = False) -> tuple[int, int]:
        """把**存量** fact 补进词法索引（ADR-0028 E6.2 配套）。

        为什么需要它：`add_fact` 只在**写新 fact 时**同步 FTS，而 E6.2 是给已经跑了
        很久的库加的新通道——历史 fact 不会再被 add 一次，于是它们对词法通道
        永远不存在。实测 live 库：135 条应入索引的 fact，FTS 里只有 93 条。
        词法路是专治"dense 对标识符是瞎的"（`bxe3663a38` vs `bx3a984bcd` cosine 0.9890），
        少 42 条就是这 42 条永远只能靠 dense 撞运气。

        幂等：`upsert` 是先删后插；`force=False` 时索引已齐则直接返回（不空跑）。
        边界与写侧一致：file_ref 不入、t_invalid 非空不入。
        返回 (回填前行数, 回填后行数)。

        M0-5：判"齐不齐"要看**两张表**——影子二元组表是后加的，存量库里
        `CREATE TABLE IF NOT EXISTS` 会建出一张空表，只看 trigram 行数会误判成
        "已齐"，于是二字中文词永远查不到。
        """
        fts = self._ensure_fts()
        if fts is None or not fts.available or self._read_only:
            return 0, 0
        eligible = [
            f for f in self.all_facts()
            if f.item_kind != ItemKind.FILE_REF
            and getattr(f, "t_invalid", None) is None
        ]
        before = fts.count()
        bigram_before = fts.bigram_count() if hasattr(fts, "bigram_count") else before
        if not force and before >= len(eligible) and bigram_before >= before:
            return before, before
        for fact in eligible:
            fts.upsert(fact)
        after = fts.count()
        logger.info("index_fts_backfilled", before=before, after=after,
                    bigram_before=bigram_before,
                    bigram_after=fts.bigram_count() if hasattr(fts, "bigram_count") else -1,
                    eligible=len(eligible))
        return before, after

    def rebuild_fts(self) -> tuple[int, int]:
        """删表重建词法索引（M0-5 的 drop+rebuild 入口）。

        分词方案变更后必须走这条路：增量 `upsert` 只会补上新写的 fact，
        存量行仍是旧口径。返回 (重建后 trigram 行数, 重建后 bigram 行数)。
        """
        fts = self._ensure_fts()
        if fts is None or not fts.available or self._read_only:
            return 0, 0
        if not fts.drop_and_recreate():
            return 0, 0
        self.backfill_fts(force=True)
        n_tri, n_bi = fts.count(), fts.bigram_count()
        logger.info("index_fts_rebuilt", trigram_rows=n_tri, bigram_rows=n_bi)
        return n_tri, n_bi

    def _ensure_fts(self) -> Any:
        """惰性打开词法索引（ADR-0028 E6.2）。失败只试一次，之后恒返回 None。"""
        if self._fts is not None or self._fts_tried:
            return self._fts
        self._fts_tried = True
        try:
            from bladex_proxy.storage.fts_index import FtsIndex
            idx = FtsIndex(self._path, read_only=self._read_only)
            if idx.open():
                self._fts = idx
        except Exception as e:  # noqa: BLE001
            logger.warning("fts_init_failed", error=str(e))
        return self._fts

    def _admissible(self, fact: Fact, pids: list[str] | None) -> bool:
        """词法/层级通道补进来的候选是否过硬边界（与 dense 主路径同口径）。"""
        if pids and fact.source_user_id not in pids:
            return False
        if getattr(fact, "t_invalid", None) is not None:
            return False
        min_imp = flag_number("BLADEX_MIN_IMPORTANCE")
        return not (min_imp > 0 and getattr(fact, "importance", 1.0) < min_imp)

    def _hop_expand(
        self,
        facts: list[Fact],
        top_k: int,
        pids: list[str] | None,
        supersede_swaps: list[str],
        min_importance: float,
    ) -> tuple[list[Fact], int]:
        """U6 有限跳（ADR-0026 §6.4）：边界内 1 跳扩展，预算 top_k//2。

        三条跳边（全部只在已命中结果的邻域内，不做全图游走）：
          ① SUPERSEDES 反向替换：dense 命中了被取代旧条 → 补进取代它的新条。
          ② 同 Matter 兄弟：种子 fact 的 matter_id 域内 top fact（manual 优先 +
             weight/confidence 降序，get_facts_for_matter 既有口径）。
          ③ PART_OF 父脉络：种子 Matter 的 PART_OF 父 Matter 域内 top fact。

        边界不破：pids（可见性）/current-only/min-importance 与主检索同口径。
        跳入的 fact 打 f._hop 标记（supersede/sibling/part_of），追加在尾部不
        挤占 dense 排序（分数低于现有最低分，稳定）。
        """
        budget = max(1, top_k // 2)
        added = 0
        have = {f.id for f in facts}
        pid_set = set(pids) if pids else None
        floor_score = min((getattr(f, "_score", 0.0) for f in facts), default=0.0)

        def _admit(f: Fact | None, hop_kind: str) -> bool:
            nonlocal added
            if f is None or f.id in have or added >= budget:
                return False
            if getattr(f, "t_invalid", None) is not None:
                return False
            if pid_set is not None and f.source_user_id not in pid_set:
                return False
            if min_importance > 0 and getattr(f, "importance", 1.0) < min_importance:
                return False
            f._hop = hop_kind
            f._score = floor_score * 0.9  # 追加尾部，不挤占 dense 排序
            facts.append(f)
            have.add(f.id)
            added += 1
            return True

        # ① SUPERSEDES 反向替换
        for new_id in supersede_swaps:
            _admit(self.get_fact(new_id), "supersede")

        # ②③ 种子 = 前 3 个带 matter_id 的命中
        seed_matters: list[str] = []
        for f in facts[:top_k]:
            mid = getattr(f, "matter_id", "") or ""
            if mid and mid not in seed_matters:
                seed_matters.append(mid)
            if len(seed_matters) >= 3:
                break
        # ADR-0021 §2.4 复核（2026-08-16）：下面两处 `get_facts_for_matter` **有意不传**
        # user_id/visibility —— `_admit` 闭包已做 pid 检查（本函数上方
        # `if pid_set is not None and f.source_user_id not in pid_set`），**不会泄漏**。
        # 已知次要缺口：`top_k=3/2` 的截断发生在 `_admit` 之前，混合 user 的卡上
        # 可能少召回几条兄弟。个人模式（单 user）不可能触发；企业形态要修的话，
        # 应给 `get_facts_for_matter` 加一个内部 `pids` 直通参数，而不是在这里
        # 把 pid_set 反向拼回 `personal:` 字符串。单独立卡，不在本步范围。
        # MQ-R6（2026-08-18）：种子 Matter 的边**一次扫描全取回**。
        # 改造前这里每轮扫 6 遍全库（3 个种子 × 直接调 get_edges + get_facts_for_matter
        # 内部再调一次），每遍 28ms = 168ms = `_hop_expand` 的全部耗时。
        # 成本在"走过 26608 个 key"本身（探针实测：全库 key 迭代占 91%），
        # 所以唯一有效的省法就是少扫几遍。
        edges_by_matter = self.get_edges_for_matters(seed_matters)
        for mid in seed_matters:
            if added >= budget:
                break
            seed_edges = edges_by_matter.get(mid, [])
            # ② 同 Matter 兄弟
            try:
                for sib in self.get_facts_for_matter(mid, top_k=3, edges=seed_edges):
                    _admit(sib, "sibling")
            except Exception as e:  # noqa: BLE001 —— 跳失败不影响主结果
                logger.warning("index_hop_sibling_failed", matter_id=mid, error=str(e))
            # ③ PART_OF 父脉络（父卡的边不在本次批量里——父卡不是种子，
            #    且这一路命中率低，多扫那一遍不值得为它把批量做成两轮）
            try:
                for edge in seed_edges:
                    if (edge.relation == EdgeRelation.PART_OF
                            and edge.target_type == EdgeTargetType.MATTER):
                        for pf in self.get_facts_for_matter(edge.target_key, top_k=2):
                            _admit(pf, "part_of")
            except Exception as e:  # noqa: BLE001
                logger.warning("index_hop_partof_failed", matter_id=mid, error=str(e))

        if added:
            logger.info("index_search_hop_expanded", added=added, budget=budget,
                         seeds=len(seed_matters), swaps=len(supersede_swaps))
        return facts, added

    def _visibility_pids(
        self,
        visibility: list[str] | None,
        user_id: str | None,
    ) -> list[str] | None:
        """从可见集合提取 personal principal ids（LanceDB source_user_id in-list 过滤用）。

        返回 None = 不过滤（legacy user_id=None 全量，向后兼容）。
        返回 list（可能空）= 用 ``source_user_id IN (...)`` 过滤；空 list = fail-closed。
        ``personal:<pid>`` -> pid；team:/org: scope 本卡无对应 fact（提升是 E2），不计入。

        🔴 三种"空"必须分开（2026-07-28 真实流量事故修复）：

        ============================  ==================  ==========================
        visibility                    行为                语义
        ============================  ==================  ==========================
        ``[]`` / ``None``             回落 user_id        个人模式**未标注**
        ``["team:x"]``（无 personal） ``[]`` → fail-closed 显式只可见 team/org
        ``["personal:abc"]``          ``["abc"]``         企业模式正常
        ============================  ==================  ==========================

        修复前写的是 ``if visibility is not None``——个人模式下 ``Identity.visibility``
        默认是**空列表**（`Field(default_factory=list)`，非 None），于是走进第一分支
        得到 ``[]``，被下游判成"显式空可见集合"触发 fail-closed，
        **Fact 散点召回 100% 返回空**（实测 13/13 轮 `index_search_empty_visibility`，
        `index_search_done` 一次未出现），注入全靠 Matter 两级召回兜着。
        ADR-0021 §2.4 的原意是"空 = 个人模式现状回落，走 user_id 兼容路径"，
        代码把"没标注"误当成了"声明可见集合为空"。自 ADR-0021 T1（2026-07-22）起潜伏。
        """
        if visibility:
            return [s[len("personal:"):] for s in visibility if s.startswith("personal:")]
        if user_id:
            return [user_id]
        return None

    def _matter_visible(
        self,
        matter: Matter,
        pids: list[str] | None,
        visibility: list[str] | None,
    ) -> bool:
        """Matter 是否对请求者可见（ADR-0021 §2.4 补平，2026-08-16）。

        🔴 **不新写三态判定**：`pids` 由 `_visibility_pids` 算好传进来，本函数只做
        scope 比对。2026-07-28 的事故（`if visibility is not None` 把个人模式的空列表
        当成"显式空集合"，Fact 散点召回 fail-closed 锁死 13/13 轮、潜伏 6 天）就是
        同一套语义被写了两遍的代价 —— 这里只允许有一个实现。

        比对规则：

        ==========================  ==========================================
        matter.scope                行为
        ==========================  ==========================================
        ``""``（空）                可见。存量 120/120 都是空（写侧此前从未落地），
                                    过滤掉等于把整个 ②平面清零。**这是迁移兼容的
                                    有意 fail-open**：企业部署要靠 Matter 侧隔离，
                                    必须先跑一次全量重建把 scope 回填。
        ``personal:<pid>``          pid ∈ pids
        ``team:x`` / ``org:y``      整串 ∈ visibility（本轮无此形态，留通路）
        ==========================  ==========================================

        `pids is None` = 不过滤（legacy 全量路径），与 Fact 侧同义。
        """
        scope = (getattr(matter, "scope", "") or "").strip()
        if not scope or pids is None:
            return True
        if scope.startswith("personal:"):
            return scope[len("personal:"):] in pids
        return scope in (visibility or [])

    # ── U10（ADR-0026 R5）：画像卡——被动捕获 + 工具计数器 + 渲染 ──

    def _profile_get(self, key: str) -> dict:
        raw = self.meta_db.get(key.encode()) if self.meta_db is not None else None
        if raw is None:
            return {}
        try:
            return msgpack.unpackb(raw, raw=False)
        except Exception:  # noqa: BLE001
            return {}

    def _profile_put(self, key: str, data: dict) -> None:
        self.meta_db[key.encode()] = msgpack.packb(data, use_bin_type=True)

    def update_profiles_from_turn(self, turn: Any, uid: str) -> None:
        """U10：从一条 Memory Hub Turn 确定性更新画像存储（rebuild 侧调用，gated）。

        - 工具计数器：tool_events(direction=call).tool_name → profile/agent/<base>。
        - 规则文件被动捕获：system/user 消息中的 CLAUDE.md/AGENTS.md 等 →
          profile/user/<uid>（name → hash + 摘录；hash 变化才更新，版本化）。
        计数可交换（order-insensitive）→ 全量重放两次逐字节一致（G6）。
        """
        from bladex_core.profile import detect_rule_files

        # 工具计数器（audience 隔离粒度 = agent base）
        base = (turn.identity.agent_id or "").split(":")[0] or "unknown"
        calls = [te.tool_name for te in turn.tool_events
                 if getattr(te, "direction", "") == "call" and getattr(te, "tool_name", "")]
        if calls:
            key = f"{_PROFILE_PREFIX}agent/{base}"
            prof = self._profile_get(key)
            tools: dict[str, int] = dict(prof.get("tools", {}))
            for name in calls:
                tools[name] = int(tools.get(name, 0)) + 1
            prof["tools"] = tools
            prof["version"] = int(prof.get("version", 0)) + 1
            self._profile_put(key, prof)

        # ADR-0028 E7.4：工具习惯补"使用结果"维度（成功率 + 失败模式）。
        # 只有"用过什么工具"是不够的——真正有用的是"这个工具在这个 agent 手里
        # 好不好使"。判定是确定性启发式（result 前 200 字符命中错误词表）。
        self._update_tool_results(base, turn)

        # 规则文件被动捕获（profile_obs 语义，不产出为条目、不进散点注入）
        texts: list[str] = []
        for msg in turn.request_messages:
            if msg.get("role") not in ("system", "user"):
                continue
            c = msg.get("content", "")
            if isinstance(c, str):
                texts.append(c)
            elif isinstance(c, list):
                texts.extend(p.get("text", "") for p in c
                             if isinstance(p, dict) and p.get("type") == "text")
        hits: list[tuple[str, str, str]] = []
        for t in texts:
            hits.extend(detect_rule_files(t))
        if hits:
            key = f"{_PROFILE_PREFIX}user/{uid}"
            prof = self._profile_get(key)
            rules: dict[str, dict] = dict(prof.get("rules", {}))
            changed = False
            for name, h, excerpt in hits:
                old = rules.get(name)
                if old is not None and old.get("hash") == h:
                    continue
                rules[name] = {"hash": h, "excerpt": excerpt}
                changed = True
            if changed:
                prof["rules"] = rules
                prof["version"] = int(prof.get("version", 0)) + 1
                self._profile_put(key, prof)
                logger.info("index_profile_rules_updated", uid=uid,
                             rules=sorted(rules.keys()))

        # ADR-0028 E7.3：规则文件**正文**副本（此前信封剥离时整段丢弃）。
        # hash 变才更新；每个 agent 一份副本（同一份 CLAUDE.md 在不同 agent 眼里
        # 可能是不同版本）。
        self._capture_rule_file_bodies(base, texts)




    def _index_files_from_turn(self, turn: Any, uid: str) -> int:
        """ADR-0028 E7.1：从一条 Turn 的读写类 tool 事件建文件内容索引。

        同 content_hash 已索引过 → 跳过（不重复调 LLM，也不重复写向量）。
        无蒸馏器时只录元数据（name/ext/mime_class），不做摘要。
        """
        from bladex_core.file_index import (
            PATH_ARG_KEYS,
            FileOrigin,
            make_entry,
            worth_indexing,
        )

        if self._embedder is None:
            return 0
        # call 事件带路径、result 事件带正文 —— 需要把两侧配起来。
        #
        # 🔴 2026-08-05 真实 Memory Hub 复核：**不能只靠 tool_call_id**。实测捕获侧
        # call 事件的 `tool_call_id` 是空的（314 条 call 全空），cid 只出现在
        # result 侧；只按 cid 配对 → 一条都配不上 → 整个 E7.1 通道恒空转。
        # 故按两级配对：① cid 命中优先；② 退回**同 tool_name 按出现顺序**配对
        # （确定性：同一份 messages 永远配出同一组，重建等价性不破）。
        by_cid: dict[str, tuple[str, str]] = {}          # cid -> (tool_name, path)
        by_name: dict[str, list[str]] = defaultdict(list)  # tool_name -> [path, ...]
        for te in getattr(turn, "tool_events", None) or []:
            if getattr(te, "direction", "") != "call":
                continue
            args = getattr(te, "arguments", None)
            if not isinstance(args, dict):
                continue
            name = (getattr(te, "tool_name", "") or "").lower()
            for k in PATH_ARG_KEYS:
                v = args.get(k)
                if isinstance(v, str) and v.strip():
                    cid = getattr(te, "tool_call_id", "") or ""
                    if cid:
                        by_cid[cid] = (name, v.strip())
                    by_name[name].append(v.strip())
                    break

        indexed = 0
        for te in getattr(turn, "tool_events", None) or []:
            if getattr(te, "direction", "") == "call":
                continue
            cid = getattr(te, "tool_call_id", "") or ""
            tool_name, path = by_cid.get(cid, ("", ""))
            if not path:
                # ② 顺序配对：本 result 的工具名 → 该工具尚未用掉的第一个路径
                tool_name = (getattr(te, "tool_name", "") or "").lower()
                queue = by_name.get(tool_name)
                path = queue.pop(0) if queue else ""
            if not path:
                continue
            result = getattr(te, "result", None)
            content = result if isinstance(result, str) else ""
            if not worth_indexing(tool_name, path, content):
                continue
            entry = make_entry(
                path, content,
                agent_id=turn.identity.agent_id, user_id=uid,
                mtime=turn.ts.isoformat() if getattr(turn, "ts", None) else "",
                origin=FileOrigin.READ,   # 第一类：会话里读到的，正文磁盘上本来就有
            )
            if self._file_hash_seen(entry.file_id, entry.content_hash):
                continue
            if self._distiller is not None and entry.mime_class not in ("image", "av"):
                fn = getattr(self._distiller, "summarize_file", None)
                if fn is not None:
                    entry.summary, entry.keywords = fn(path, content)
            try:
                entry.vector = embed_passage_compat(
                    self._embedder, [entry.embed_text()])[0]
            except Exception as e:  # noqa: BLE001
                logger.warning("index_file_embed_failed", path=path, error=str(e))
                continue
            self.upsert_file_index(entry)
            self._file_hash_put(entry.file_id, entry.content_hash)
            indexed += 1
        return indexed


    def _index_pasted_document(self, doc: dict) -> bool:
        """M1-2 文档路：把一段粘贴物建成文件内容索引条目（E7.1 复用）。

        与 `_index_files_from_turn`（工具读到的文件）的区别只在**来源**：
        这一份是用户直接粘进对话的，磁盘上不一定有对应文件，故 path 用
        `paste://<content_hash>` 形态。同 hash 已索引过 → 跳过（不重复调 LLM）。

        为什么不产 assertion/preference（附录 D.4）：文档内容不是用户断言。
        把一份 12K 的规则文件切成六句去走"用户陈述蒸馏"，每一段都在错误的
        语义契约下被处理——那正是失效链 A/B 的成因。
        """
        from bladex_core.file_index import FileOrigin, make_entry

        if self._embedder is None:
            return False
        content = doc.get("content") or ""
        path = doc.get("path") or ""
        if not content or not path:
            return False
        entry = make_entry(
            path, content,
            agent_id=doc.get("agent_id", ""), user_id=doc.get("user_id", ""),
            mtime=doc.get("ts", ""), origin=FileOrigin.READ,
        )
        if self._file_hash_seen(entry.file_id, entry.content_hash):
            return False
        if self._distiller is not None:
            fn = getattr(self._distiller, "summarize_file", None)
            if fn is not None:
                entry.summary, entry.keywords = fn(path, content)
        entry.vector = embed_passage_compat(self._embedder, [entry.embed_text()])[0]
        self.upsert_file_index(entry)
        self._file_hash_put(entry.file_id, entry.content_hash)
        logger.info("index_pasted_document", path=path, chars=len(content),
                    has_summary=bool(entry.summary))
        return True

    def _blob_store(self):  # noqa: ANN202
        """内容寻址副本存储（惰性建；只有 PRODUCED 用得上）。"""
        if getattr(self, "_blobs", None) is None:
            from bladex_proxy.storage.blob_store import BlobStore
            self._blobs = BlobStore(self._path, read_only=self._read_only)
        return self._blobs

    def _index_produced_files_from_turn(self, turn: Any, uid: str) -> int:
        """第二类（2026-08-05 拍板）：**模型在回复正文里产出的文件**。

        与第一类（工具读到的）的两点不同：
          1. 索引标记 `origin=produced`，检索/展示时可区分"这是它写给我的"；
          2. 正文按 sha256 **落盘存副本**，索引用 `blob_ref` 关联 ——
             因为这份内容只存在于那一次回复里，不存就永远找不回来了。

        同 content_hash 已索引过 → 跳过（不重复调 LLM、不重复写 blob）。
        """
        from bladex_core.file_index import (
            FileOrigin,
            extract_produced_files,
            make_entry,
        )

        if self._embedder is None:
            return 0
        text = getattr(turn, "response_text", "") or ""
        produced = extract_produced_files(text)
        if not produced:
            return 0

        blobs = self._blob_store()
        indexed = 0
        for path, body in produced:
            entry = make_entry(
                path, body,
                agent_id=turn.identity.agent_id, user_id=uid,
                mtime=turn.ts.isoformat() if getattr(turn, "ts", None) else "",
                origin=FileOrigin.PRODUCED,
            )
            if self._file_hash_seen(entry.file_id, entry.content_hash):
                continue
            entry.blob_ref = blobs.put(body)
            if self._distiller is not None:
                fn = getattr(self._distiller, "summarize_file", None)
                if fn is not None:
                    entry.summary, entry.keywords = fn(path, body)
            try:
                entry.vector = embed_passage_compat(
                    self._embedder, [entry.embed_text()])[0]
            except Exception as e:  # noqa: BLE001
                logger.warning("index_produced_embed_failed", path=path, error=str(e))
                continue
            self.upsert_file_index(entry)
            self._file_hash_put(entry.file_id, entry.content_hash)
            indexed += 1
        if indexed:
            logger.info("index_produced_files", count=indexed,
                        blobs=blobs.count())
        return indexed

    def get_file_entry(self, file_id: str) -> dict | None:
        """按 file_id 读一条文件索引（含 origin / blob_ref）。"""
        table = getattr(self, "_files_tbl", None)
        if table is None and self._lancedb is not None:
            try:
                table = self._lancedb.open_table(self._FILES_TABLE)
                self._files_tbl = table
            except Exception:  # noqa: BLE001
                return None
        if table is None:
            return None
        try:
            for row in table.to_arrow().to_pylist():
                if row.get("file_id") == file_id:
                    return row
        except Exception as e:  # noqa: BLE001
            logger.warning("index_file_entry_read_failed", error=str(e))
        return None

    def get_file_body(self, file_id: str) -> str | None:
        """回溯模型产出的那份正文（只有 PRODUCED 有副本）。"""
        row = self.get_file_entry(file_id)
        if not row:
            return None
        return self._blob_store().get(row.get("blob_ref") or "")

    def list_file_entries(self) -> list[dict]:
        """全部文件索引条目（不含向量）——admin 只读面（/admin/files）数据源。

        全表 to_arrow 扫描：files 表规模 = 会话里出现过的文件数（小表），
        与 get_file_entry 同一读取路径；没建过表 = 无文件索引 → 空列表。
        """
        table = getattr(self, "_files_tbl", None)
        if table is None and self._lancedb is not None:
            try:
                table = self._lancedb.open_table(self._FILES_TABLE)
                self._files_tbl = table
            except Exception:  # noqa: BLE001 —— 没建过表 = 无文件索引
                return []
        if table is None:
            return []
        try:
            rows = table.to_arrow().to_pylist()
        except Exception as e:  # noqa: BLE001
            logger.warning("index_file_list_failed", error=str(e))
            return []
        for r in rows:
            r.pop("vector", None)
        return rows

    def delete_file_index(self, file_id: str, *, drop_blob: bool = True) -> bool:
        """删一条文件索引；副本没人再引用时一并删（forget 级联，拍板项）。

        内容寻址意味着多条条目可能共享同一个 blob，所以删之前先数引用。
        """
        if self._read_only:
            return False
        row = self.get_file_entry(file_id)
        if row is None:
            return False
        blob_ref = row.get("blob_ref") or ""
        try:
            safe = file_id.replace("'", "''")
            self._files_tbl.delete(f"file_id = '{safe}'")
        except Exception as e:  # noqa: BLE001
            logger.warning("index_file_delete_failed", file_id=file_id, error=str(e))
            return False
        # meta 侧的 hash 去重记录也要清，否则同内容再来时会被当成"已索引"而跳过
        try:
            del self.meta_db[f"{_FILEHASH_PREFIX}{file_id}".encode()]
        except Exception:  # noqa: BLE001 —— 没有就算了
            pass
        if drop_blob and blob_ref:
            still_used = any(
                r.get("blob_ref") == blob_ref
                for r in (self._files_tbl.to_arrow().to_pylist() or [])
            )
            if not still_used:
                self._blob_store().delete(blob_ref)
        logger.info("index_file_deleted", file_id=file_id, had_blob=bool(blob_ref))
        return True

    def _file_hash_seen(self, file_id: str, content_hash: str) -> bool:
        rec = self._profile_get(f"{_FILEHASH_PREFIX}{file_id}")
        return rec.get("content_hash") == content_hash

    def _file_hash_put(self, file_id: str, content_hash: str) -> None:
        self._profile_put(f"{_FILEHASH_PREFIX}{file_id}",
                          {"content_hash": content_hash, "updated_at": _utc_now_iso()})

    def maintain_projects(self) -> int:
        """ADR-0028 E7.2：从规则文件副本识别 Project（识别①）+ 挂 PART_OF 边 + 重算字段。

        空闲轮调用（与 ANN / lifecycle 同址）。识别②（tool 路径高频前缀）由
        `discover_project_from_paths` 在需要时补充；识别③是 CLI 手动。
        返回 upsert 的 Project 数。
        """
        from bladex_core.project import Project, name_from_root, project_id_for

        self._ensure_meta_db()
        if self.meta_db is None or self._read_only:
            return 0
        roots: set[str] = set()
        for f in self.all_facts():
            if f.item_kind != ItemKind.FILE_REF:
                continue
            path = (getattr(f, "subject", "") or "").replace("\\", "/")
            name = path.rsplit("/", 1)[-1].lower()
            if name in ("claude.md", "agents.md", "agent.md", ".cursorrules"):
                root = path.rsplit("/", 1)[0]
                if root:
                    roots.add(root)
        n = 0
        for root in sorted(roots):
            pid = project_id_for(root)
            proj = self.get_project(pid) or Project(
                project_id=pid, root_path=root, name=name_from_root(root))
            self.upsert_project(proj)
            self.link_matters_to_project(pid, root)
            self.refresh_project_fields(pid)
            n += 1
        if n:
            logger.info("index_projects_maintained", projects=n)
        return n

    def discover_project_from_paths(self, paths: list[str]) -> str:
        """ADR-0028 E7.2 识别②：同 session 内 tool 路径的高频公共前缀（≥5 次、深度 ≥2）。

        返回新建/命中的 project_id（识别不出返回空串——识别不出就不识别，不猜）。
        """
        from bladex_core.project import (
            Project,
            infer_root_from_paths,
            name_from_root,
            project_id_for,
        )

        root = infer_root_from_paths(paths)
        if not root or self._read_only:
            return ""
        pid = project_id_for(root)
        if self.get_project(pid) is None:
            self.upsert_project(Project(project_id=pid, root_path=root,
                                        name=name_from_root(root)))
        return pid

    # ── ADR-0028 E7.2：Project 实体（主题轴最上层）──

    def upsert_project(self, project: Any) -> None:  # noqa: ANN401  Project
        """写一条 Project（确定性 id，重建稳定）。"""
        project.version = int(getattr(project, "version", 0)) + 1
        self.meta_db[f"{_PROJECT_PREFIX}{project.project_id}".encode()] = msgpack.packb(
            project.model_dump(mode="json"), use_bin_type=True)
        logger.info("index_project_upserted", project_id=project.project_id,
                    name=project.name)

    def get_project(self, project_id: str):  # noqa: ANN201  Project | None
        from bladex_core.project import Project

        self._ensure_meta_db()
        if self.meta_db is None:
            return None
        raw = self.meta_db.get(f"{_PROJECT_PREFIX}{project_id}".encode())
        if raw is None:
            return None
        try:
            return Project.model_validate(msgpack.unpackb(raw, raw=False))
        except Exception as e:  # noqa: BLE001
            logger.warning("index_project_load_failed", project_id=project_id, error=str(e))
            return None

    def all_projects(self) -> list:
        from bladex_core.project import Project

        self._ensure_meta_db()
        if self.meta_db is None:
            return []
        out = []
        for key_raw, val_raw in self.meta_db.items():
            k = key_raw.decode() if isinstance(key_raw, bytes) else str(key_raw)
            if not k.startswith(_PROJECT_PREFIX):
                continue
            try:
                out.append(Project.model_validate(msgpack.unpackb(val_raw, raw=False)))
            except Exception:  # noqa: BLE001
                continue
        out.sort(key=lambda p: p.project_id)
        return out

    def link_matters_to_project(self, project_id: str, root_path: str) -> int:
        """`Matter --PART_OF--> Project`：成员 fact 路径落在 root_path 下 ≥50% 即挂边。

        路径来源 = 成员 fact 里 item_kind=file_ref 的 subject（E1.5 白名单产出的
        真实被操作文件路径）。返回新挂的边数。
        """
        from bladex_core.project import matter_belongs_to_project

        linked = 0
        for matter in self.all_matters():
            if matter.matter_id == UNASSIGNED_MATTER_ID:
                continue
            paths: list[str] = []
            for edge in self.get_edges(matter.matter_id):
                if edge.target_type != EdgeTargetType.FACT:
                    continue
                f = self.get_fact(edge.target_key)
                if f is not None and f.item_kind == ItemKind.FILE_REF:
                    paths.append(getattr(f, "subject", "") or "")
            if not matter_belongs_to_project(paths, root_path):
                continue
            edge = MatterEdge(
                matter_id=matter.matter_id,
                target_type=EdgeTargetType.MATTER,
                target_key=project_id,
                relation=EdgeRelation.PART_OF,
                provenance=EdgeProvenance.AUTO,
            )
            self.add_edge(edge)
            linked += 1
        if linked:
            logger.info("index_project_matters_linked", project_id=project_id, linked=linked)
        return linked

    def refresh_project_fields(self, project_id: str) -> None:
        """确定性重算 description / progress_note / languages（不 LLM）。"""
        from bladex_core.project import compose_description, compose_progress, languages_of

        proj = self.get_project(project_id)
        if proj is None:
            return
        titles: list[str] = []
        summaries: list[str] = []
        paths: list[str] = []
        for matter in self.all_matters():
            if not any(
                e.relation == EdgeRelation.PART_OF and e.target_key == project_id
                for e in self.get_edges(matter.matter_id)
            ):
                continue
            titles.append(matter.title)
            summaries.append(matter.summary)
            for edge in self.get_edges(matter.matter_id):
                if edge.target_type != EdgeTargetType.FACT:
                    continue
                f = self.get_fact(edge.target_key)
                if f is not None and f.item_kind == ItemKind.FILE_REF:
                    paths.append(getattr(f, "subject", "") or "")
        proj.description = compose_description(titles)
        proj.progress_note = compose_progress(summaries)
        if paths:
            proj.languages = languages_of(paths)
        self.upsert_project(proj)

    def project_card_for(self, text: str) -> list[str]:
        """E7.2 ①平面 Project 卡：请求文本里出现某 Project 的 root_path 前缀就注。

        确定性匹配（不做模糊），命中多个取 root_path 最长（最具体）的那个。
        """
        from bladex_core.project import render_project_card

        if not text:
            return []
        best = None
        for proj in self.all_projects():
            if proj.root_path and proj.root_path in text:
                if best is None or len(proj.root_path) > len(best.root_path):
                    best = proj
        if best is None:
            return []
        active = sum(
            1 for m in self.all_matters()
            if m.status == MatterStatus.ACTIVE
            and any(e.relation == EdgeRelation.PART_OF and e.target_key == best.project_id
                    for e in self.get_edges(m.matter_id))
        )
        return render_project_card(best, active_matters=active)

    # ── ADR-0028 E7.1：文件内容索引 ──

    _FILES_TABLE = "files"

    def _ensure_files_table(self, dim: int):  # noqa: ANN202
        """惰性建 files 表（与 facts 表分开——文件是**独立通道**，不挤占散点召回位）。"""
        if getattr(self, "_files_tbl", None) is not None:
            return self._files_tbl
        if self._lancedb is None:
            return None
        import pyarrow as pa

        schema = pa.schema([
            pa.field("file_id", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
            pa.field("path", pa.string()),
            pa.field("name", pa.string()),
            pa.field("ext", pa.string()),
            pa.field("mime_class", pa.string()),
            pa.field("summary", pa.string()),
            pa.field("keywords", pa.string()),
            pa.field("content_hash", pa.string()),
            pa.field("agent_id", pa.string()),
            pa.field("user_id", pa.string()),
            # 两类来源的索引标记（mentioned/read = 会话侧；produced = 模型产出）
            pa.field("origin", pa.string()),
            # PRODUCED 专有：本地内容寻址副本 id（空 = 无副本）
            pa.field("blob_ref", pa.string()),
        ])
        try:
            self._files_tbl = self._lancedb.create_table(
                self._FILES_TABLE, schema=schema, mode="create")
        except Exception:  # noqa: BLE001 —— 已存在则打开
            try:
                self._files_tbl = self._lancedb.open_table(self._FILES_TABLE)
                self._migrate_files_columns()
            except Exception as e:  # noqa: BLE001
                logger.warning("index_files_table_unavailable", error=str(e))
                self._files_tbl = None
        return self._files_tbl

    def _migrate_files_columns(self) -> None:
        """存量迁移：旧 files 表没有 origin/blob_ref 列 → 加列并回填默认值。

        与 `_migrate_entities_column` 同款（幂等；失败不致命——只是新列缺省）。
        存量 55 条全部来自工具读取，故 origin 默认回填 `read`。
        """
        if self._read_only or self._files_tbl is None:
            return
        try:
            names = set(self._files_tbl.schema.names)
        except Exception:  # noqa: BLE001
            return
        missing = {"origin": "'read'", "blob_ref": "''"} 
        todo = {k: v for k, v in missing.items() if k not in names}
        if not todo:
            return
        try:
            self._files_tbl.add_columns(todo)
            logger.info("index_files_columns_migrated", added=sorted(todo))
        except Exception as e:  # noqa: BLE001
            logger.warning("index_files_migrate_failed", error=str(e))

    def upsert_file_index(self, entry: Any) -> None:  # noqa: ANN401  FileEntry
        """写一条文件索引（同 path 新 hash → 覆盖更新）。

        files 表**不是记忆总账**，无 as-of 义务（历史在 Memory Hub）——所以这里是覆盖，
        不是 Fact 那套非破坏取代。
        """
        if entry.vector is None or self._lancedb is None:
            return
        table = self._ensure_files_table(len(entry.vector))
        if table is None:
            return
        import pyarrow as pa

        try:
            safe = entry.file_id.replace("'", "''")
            table.delete(f"file_id = '{safe}'")
        except Exception:  # noqa: BLE001 —— 首次写入无旧行
            pass
        table.add(pa.table({
            "file_id": [entry.file_id], "vector": [entry.vector],
            "path": [entry.path], "name": [entry.name], "ext": [entry.ext],
            "mime_class": [entry.mime_class], "summary": [entry.summary],
            "keywords": [" ".join(entry.keywords)],
            "content_hash": [entry.content_hash],
            "agent_id": [entry.agent_id], "user_id": [entry.user_id],
            "origin": [getattr(entry.origin, "value", str(entry.origin))],
            "blob_ref": [entry.blob_ref],
        }))
        logger.info("index_file_indexed", path=entry.path, mime_class=entry.mime_class)

    def search_files(self, q_vec: list[float], k: int = 3,
                     user_id: str | None = None) -> list[dict]:
        """文件独立通道检索（E7.1）。返回 [{file_id, path, name, summary, ...}]。"""
        if self._lancedb is None:
            return []
        table = getattr(self, "_files_tbl", None)
        if table is None:
            try:
                table = self._lancedb.open_table(self._FILES_TABLE)
                self._files_tbl = table
            except Exception:  # noqa: BLE001 —— 没建过表 = 无文件索引
                return []
        try:
            builder = table.search(q_vec).distance_type("cosine")
            if user_id:
                builder = builder.where(f"user_id = '{user_id}'")
            return builder.limit(k).to_list()
        except Exception as e:  # noqa: BLE001
            logger.warning("index_file_search_failed", error=str(e))
            return []

    # ── ADR-0028 E7.3/E7.4：规则文件副本 + 工具结果维度 + 画像文件 ──

    #: 工具结果判定词表（确定性启发式，任务卡 E7.4 写死）
    _ERR_MARKERS: tuple[str, ...] = (
        "error", "exception", "traceback", "失败", "错误", "denied", "not found",
    )
    _ERR_SCAN_CHARS = 200

    @classmethod
    def _classify_tool_result(cls, result: Any) -> tuple[bool, str]:
        """(是否失败, 失败类别)。判据 = result 前 200 字符命中错误词表（大小写不敏感）。"""
        if result is None:
            return False, ""
        text = result if isinstance(result, str) else str(result)
        head = text[:cls._ERR_SCAN_CHARS].lower()
        for marker in cls._ERR_MARKERS:
            if marker in head:
                return True, marker
        return False, ""

    def _update_tool_results(self, agent_base: str, turn: Any) -> None:
        """E7.4：profile/tool/<base>/<tool> = {calls, ok, err, last_err_class}。

        计数可交换（order-insensitive）→ 全量重放两次逐字节一致（G6）。
        """
        pairs: dict[str, str] = {}      # tool_call_id -> tool_name
        for te in getattr(turn, "tool_events", None) or []:
            name = getattr(te, "tool_name", "") or ""
            cid = getattr(te, "tool_call_id", "") or ""
            if getattr(te, "direction", "") == "call" and name:
                pairs[cid or name] = name
        for te in getattr(turn, "tool_events", None) or []:
            if getattr(te, "direction", "") == "call":
                continue
            cid = getattr(te, "tool_call_id", "") or ""
            name = getattr(te, "tool_name", "") or pairs.get(cid, "")
            if not name:
                continue
            failed, cls_ = self._classify_tool_result(getattr(te, "result", None))
            key = f"{_PROFILE_PREFIX}tool/{agent_base}/{name}"
            rec = self._profile_get(key)
            rec["calls"] = int(rec.get("calls", 0)) + 1
            if failed:
                rec["err"] = int(rec.get("err", 0)) + 1
                rec["last_err_class"] = cls_
            else:
                rec["ok"] = int(rec.get("ok", 0)) + 1
            rec["updated_at"] = _utc_now_iso()
            self._profile_put(key, rec)

    def get_tool_stats(self, agent_base: str, top: int = 5) -> list[dict]:
        """读该 agent 的工具统计（按调用数降序 top-N）。"""
        self._ensure_meta_db()
        if self.meta_db is None:
            return []
        prefix = f"{_PROFILE_PREFIX}tool/{agent_base}/"
        out: list[dict] = []
        for key_raw, val_raw in self.meta_db.items():
            k = key_raw.decode() if isinstance(key_raw, bytes) else str(key_raw)
            if not k.startswith(prefix):
                continue
            try:
                rec = msgpack.unpackb(val_raw, raw=False)
            except Exception:  # noqa: BLE001
                continue
            rec["name"] = k[len(prefix):]
            out.append(rec)
        out.sort(key=lambda r: (-int(r.get("calls", 0)), r.get("name", "")))
        return out[:top]

    def _capture_rule_file_bodies(self, agent_base: str, texts: list[str]) -> None:
        """E7.3：把规则文件正文存成副本（rulefile/<agent_base>/<name>）。"""
        from bladex_core.rulefile import extract_rule_files

        for text in texts:
            for name, body, chash in extract_rule_files(text):
                key = f"{_RULEFILE_PREFIX}{agent_base}/{name}"
                old = self._profile_get(key)
                if old.get("content_hash") == chash:
                    continue        # hash 没变 = 同一份，不重复写
                self._profile_put(key, {
                    "content": body, "content_hash": chash,
                    "updated_at": _utc_now_iso(),
                })
                logger.info("index_rulefile_captured", agent=agent_base, name=name,
                            chars=len(body))

    def get_rule_files(self, agent_base: str) -> list[str]:
        """该 agent 已捕获的规则文件名（按名排序，确定性）。"""
        self._ensure_meta_db()
        if self.meta_db is None:
            return []
        prefix = f"{_RULEFILE_PREFIX}{agent_base}/"
        names = [
            (key_raw.decode() if isinstance(key_raw, bytes) else str(key_raw))[len(prefix):]
            for key_raw in self.meta_db.keys()
            if (key_raw.decode() if isinstance(key_raw, bytes) else str(key_raw)).startswith(prefix)
        ]
        return sorted(names)

    def get_rule_file(self, agent_base: str, name: str) -> dict:
        """单个规则文件副本（content / content_hash / updated_at；无则空 dict）。"""
        self._ensure_meta_db()
        if self.meta_db is None:
            return {}
        return self._profile_get(f"{_RULEFILE_PREFIX}{agent_base}/{name}")

    def _meta_key_bases(self, prefix: str) -> list[str]:
        """扫 meta_db 前缀，取剩余段的第一节（agent base）——admin 只读枚举用。"""
        self._ensure_meta_db()
        if self.meta_db is None:
            return []
        bases: set[str] = set()
        for key_raw in self.meta_db.keys():
            k = key_raw.decode() if isinstance(key_raw, bytes) else str(key_raw)
            if k.startswith(prefix):
                bases.add(k[len(prefix):].split("/")[0])
        return sorted(b for b in bases if b)

    def list_tool_agents(self) -> list[str]:
        """有工具统计（profile/tool/<base>/…）的 agent base 列表。"""
        return self._meta_key_bases(f"{_PROFILE_PREFIX}tool/")

    def list_profile_agents(self) -> list[str]:
        """有画像痕迹的 agent base 并集（习惯文件 + 工具统计 + 规则文件副本）。"""
        out = set(self._meta_key_bases(f"{_PROFILE_PREFIX}agent_md/"))
        out.update(self.list_tool_agents())
        out.update(self._meta_key_bases(_RULEFILE_PREFIX))
        return sorted(out)

    def refresh_profile_docs(self) -> int:
        """E7.3 接线：为库里出现过的每个 user / agent 生成画像文件。

        空闲轮调用（与 ANN / lifecycle / project / FTS 回填同址）。确定性渲染，
        重复调用产出一致；返回写出的文件数。
        """
        if self._read_only:
            return 0
        users, agents = set(), set()
        for f in self.all_facts():
            if f.source_user_id:
                users.add(f.source_user_id)
            base = (getattr(f, "agent_id", "") or "").split(":")[0]
            if base:
                agents.add(base)
        n = 0
        for uid in sorted(users):
            self.build_user_md(uid)
            n += 1
        for base in sorted(agents):
            for uid in sorted(users):
                self.build_agent_md(base, user_id=uid)
            n += 1
        if n:
            logger.info("index_profile_docs_refreshed", users=len(users), agents=len(agents))
        return n

    # （G12.4/G12.5 ADR-0031 时代的 Flash 老渲染器 `refresh_flash_docs` /
    #  `_recent_matter_pins` / `_write_flash_for_principal` / `_build_matter_flash` 已于
    #  2026-09-03 S5 删除：`BLADEX_FLASH_RENDER` 默认关、live 未设、ADR-0031 已废；
    #  Flash 树的唯一写者是 `bladex_proxy.flash_daemon`（ADR-0032 §5）。）

    def build_user_md(self, user_id: str) -> str:
        """E7.3：渲染并存 USER.md（profile/user_md）。确定性，可重复调。"""
        from datetime import UTC as _UTC
        from datetime import datetime as _dtt
        from datetime import timedelta as _td

        from bladex_core.rulefile import render_user_md

        facts = [f for f in self.all_facts() if f.source_user_id == user_id]
        agents: Counter[str] = Counter()
        recent: Counter[str] = Counter()
        cutoff = _dtt.now(_UTC) - _td(days=7)
        for f in facts:
            aid = getattr(f, "agent_id", "") or ""
            if not aid:
                continue
            agents[aid] += 1
            created = getattr(f, "created_at", None)
            try:
                c = created if created.tzinfo else created.replace(tzinfo=_UTC)
                if c >= cutoff:
                    recent[aid] += 1
            except (TypeError, AttributeError):
                pass

        prefs = [
            f for f in facts
            if f.item_kind == ItemKind.PREFERENCE and getattr(f, "t_invalid", None) is None
        ]
        prefs.sort(key=lambda f: (-float(getattr(f, "importance", 0.0) or 0.0), f.id))
        pref_lines = [f.content for f in prefs[:10]]
        coding = [
            f.content for f in prefs[:10]
            if any(_is_tech_entity(e) for e in (f.entities or []))
        ]

        md = render_user_md(
            user_id=user_id,
            agents=[{"agent_id": a, "turns": n} for a, n in agents.most_common()],
            recent_activity=[{"agent_id": a, "turns": n} for a, n in recent.most_common()],
            preferences=pref_lines,
            coding_preferences=coding,
            key_count=len(agents),
        )
        self._profile_put(f"{_PROFILE_PREFIX}user_md", {"content": md,
                                                        "updated_at": _utc_now_iso()})
        return md

    def build_agent_md(self, agent_base: str, user_id: str = "") -> str:
        """E7.3：渲染并存 per-agent 习惯文件（profile/agent_md/<base>）。"""
        from bladex_core.rulefile import render_agent_md

        own = [
            f.content for f in self.all_facts()
            if (getattr(f, "audience", "all") or "all") == f"agent:{agent_base}"
            and (not user_id or f.source_user_id == user_id)
        ][:5]
        md = render_agent_md(
            agent_base=agent_base,
            tools=self.get_tool_stats(agent_base),
            own_facts=own,
            rule_files=self.get_rule_files(agent_base),
        )
        self._profile_put(f"{_PROFILE_PREFIX}agent_md/{agent_base}",
                          {"content": md, "updated_at": _utc_now_iso()})
        return md

    def get_profile_md(self, user_id: str | None, agent_base: str = "") -> tuple[str, str]:
        """读已生成的 USER.md / agent 习惯文件（读端零成本，没有就返回空）。"""
        self._ensure_meta_db()
        if self.meta_db is None:
            return "", ""
        user_md = self._profile_get(f"{_PROFILE_PREFIX}user_md").get("content", "")
        agent_md = ""
        if agent_base:
            agent_md = self._profile_get(
                f"{_PROFILE_PREFIX}agent_md/{agent_base}").get("content", "")
        return user_md, agent_md

    # 画像卡渲染细节
    _PROFILE_TOP_TOOLS = 5
    _PROFILE_MAX_RULES = 3

    def get_profile_cards(self, user_id: str | None, agent_id: str = "") -> list[list[str]]:
        """U7 ①平面消费：渲染 User 画像卡 + Agent 习惯卡（audience 隔离）。

        Agent 习惯卡只发给对应 agent（按 base 匹配）——codex 的工具习惯
        不出现在 hermes 的注入里（U10 验收）。确定性渲染（同版本同文本）。
        """
        self._ensure_meta_db()
        if self.meta_db is None:
            return []
        cards: list[list[str]] = []
        base = agent_id.split(":")[0] if agent_id else ""

        # ── ADR-0028 E7.3：画像卡 render v2 ──
        # 旧渲染是 "[User 画像] 使用规则文件 CLAUDE.md：<一段路径串联的乱码>"，
        # 而且**每轮都在注**（trace 报告实测那两行逐轮出现）。
        # v2 改为从 USER.md / agent 习惯文件里节选**完整句子**（≤5 行）。
        from bladex_core.rulefile import render_profile_card

        user_md, agent_md = self.get_profile_md(user_id, base)
        v2_lines = render_profile_card(user_md, agent_md)
        if v2_lines:
            cards.append(v2_lines)
            return cards

        # 回退（画像文件尚未生成——全新库/首轮）：保留 v1 渲染，不至于一片空白。
        if user_id:
            prof = self._profile_get(f"{_PROFILE_PREFIX}user/{user_id}")
            rules = prof.get("rules", {})
            if rules:
                lines = []
                for name in sorted(rules.keys())[:self._PROFILE_MAX_RULES]:
                    excerpt = rules[name].get("excerpt", "")
                    suffix = f"：{excerpt}" if excerpt else ""
                    lines.append(f"[User 画像] 使用规则文件 {name}{suffix}")
                cards.append(lines)
        if base:
            # E7.4：工具习惯带上"用得顺不顺"（成功率 + 常见失败），不只是"用过几次"
            stats = self.get_tool_stats(base, top=self._PROFILE_TOP_TOOLS)
            if stats:
                parts = []
                for t in stats:
                    calls = int(t.get("calls", 0))
                    ok = int(t.get("ok", 0))
                    rate = (ok / calls * 100) if calls else 0.0
                    parts.append(f"{t['name']}({calls}, ok {rate:.0f}%)")
                cards.append([f"[Agent 习惯: {base}] 常用工具: {', '.join(parts)}"])
            else:
                prof = self._profile_get(f"{_PROFILE_PREFIX}agent/{base}")
                tools = prof.get("tools", {})
                if tools:
                    top = sorted(tools.items(), key=lambda kv: (-kv[1], kv[0]))
                    top = top[:self._PROFILE_TOP_TOOLS]
                    joined = ", ".join(f"{n}({c})" for n, c in top)
                    cards.append([f"[Agent 习惯: {base}] 常用工具: {joined}"])
        return cards

    def _apply_matter_injection_hits(self, hit_ts: Mapping[str, str]) -> int:
        """M3-1：把 Matter 卡的注入命中回写成 `Matter.last_hit_at`（相关钟）。

        三个钟里这一个此前**完全没有信号源**——`injected_fact_ids` 只记散点条目，
        Matter 卡注入零留痕，连计数都没有（5.9 对账表里那个 ✗）。
        于是 2×2 矩阵的"久无命中"那一列恒为真，dormant 判定要么不敢开、要么开了误杀。

        时间取**turn 时间**不取 `now`：重放两次必须逐字节一致（G6）。
        **不 bump `version`**：version 是 U7 ③平面 changed-since 的判据，
        被"有人看了一眼"推高会让每轮都判"有变更"，把增量注入变成全量重注。
        """
        if not hit_ts or self._read_only:
            return 0
        from datetime import UTC as _UTC
        from datetime import datetime as _dtt

        updated = 0
        orphan = 0     # MQ-S38 同款：卡侧的悬空 id 也要数出来（此前静默 continue）
        for mid, ts_iso in hit_ts.items():
            m = self.get_matter(mid)
            if m is None:
                orphan += 1
                continue
            try:
                ts = _dtt.fromisoformat(ts_iso) if ts_iso else None
            except ValueError:
                ts = None
            if ts is None:
                continue
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=_UTC)
            prev = getattr(m, "last_hit_at", None)
            if prev is not None:
                _p = prev if prev.tzinfo else prev.replace(tzinfo=_UTC)
                if _p >= ts:
                    continue      # 只前进不后退（乱序重放也稳定）
            m.last_hit_at = ts
            try:
                self.add_matter(m)
                updated += 1
            except Exception as e:  # noqa: BLE001
                logger.warning("index_matter_hit_write_failed", matter_id=mid, error=str(e))
        if updated:
            logger.info("index_matter_hits_written", updated=updated)
        if orphan:
            # `matter_id` 自 G12.1 起取首轮 ledger key（不再是标题 hash），
            # 但**全量重建仍会让当前所有 matter_id 失效一次**（MQ-S30 已登记）。
            logger.warning("index_matter_hit_orphan_ids", orphan=orphan,
                           resolved=updated,
                           hint="matter_id 在重建/合并后可能换代（MQ-S30/S38）")
        self.last_matter_hit_orphans = orphan
        return updated

    def _apply_injection_hits(self, hits: Mapping[str, int],
                              hit_ts: Mapping[str, str] | None = None,
                              keys: Mapping[str, str] | None = None) -> int:
        """ADR-0028 E2.3：注入命中聚合回写——ref_count += n 并重算 importance。

        与被它取代的 `record_injection_hits` 的关键差别（也是这条卡存在的理由）：

            旧路径由 **proxy** 在注入后直接调，而 proxy 的 Memory Index 是 read_only
            （写权只有 consolidator，ADR-0009 §7）→ 恒 no-op。
            于是 importance 的引用增益项 `1 + 0.25·ln(1+ref_count)` 有消费者、
            没有生产者，ref_count 全库恒 0。

        新路径：命中 id 随 `Turn.decision_meta.injected_fact_ids` 落 Memory Hub，
        consolidator 在 rebuild 消费该 turn 时聚合调用本方法。
        幂等性由 consumed 标记天然保证（每个 turn 只被消费一次）。

        返回实际更新条数。
        """
        if not hits or self._read_only:
            return 0
        from datetime import UTC as _UTC
        from datetime import datetime as _dtt

        from bladex_core.importance import compute_importance
        updated = 0
        # 🔴 MQ-S38（2026-08-20）：`fact is None` 此前是**静默 continue**——
        # 而 `fact_id = sha256(ledger_key + ":" + content)[:12]`（内容派生）意味着
        # **蒸馏 prompt 一换版（v6→v7 那种），全库正文重蒸 ⇒ 全库 id 换代**，
        # 历史 turn 里记的 `injected_fact_ids` 整批指空。实测 385/402 = 95.8% 悬空。
        # 后果：ref_count 的引用增益、相关钟 `last_hit_at`、dormant 判定的历史信号
        # **在每次换版时被清零，且没有任何痕迹**——消费者分不清"没被命中过"
        # 与"命中记录丢了"。这正是本仓付过五次的那个形态（FUNNEL_HIT / open_issues /
        # ref_count 恒 0 / 五开关默认关 / L0 没接线）。
        # 本次只做**可见**（零行为改动）：数出来、报出来。id 迁移/重映射属修法，
        # 涉及重建等价性，按范围冻结走 0.2 候选、待 Jason 拍。
        orphan_ids = 0
        orphan_hits = 0
        rescued_ids = 0        # 按取代键接续成功
        ambiguous_ids = 0      # 有键但活着的同键条目 >1 —— 不猜，计数
        keys = keys or {}
        # 取代键 → 活着的 fact（只建一次；仅在真有悬空 id 且带键时才付这次扫描）
        _by_key: dict[str, list[Fact]] = {}
        _key_index_built = False
        now = _dtt.now(_UTC)
        for fid, n in hits.items():
            if not fid or fid.startswith("hard_") or n <= 0:
                continue
            fact = self.get_fact(fid)
            if fact is None:
                # ── MQ-S38 接续：id 指空时按**取代键**找当前世代的同一条 ──
                sk = keys.get(fid, "")
                if sk:
                    if not _key_index_built:
                        from bladex_core.supersede import supersede_key_str
                        for _f in self.all_facts():
                            # 只认**当前有效**的条目：已被取代的那条接了也白接
                            # （它不参与召回，importance/相关钟对它没有意义）。
                            if getattr(_f, "t_invalid", None) is not None:
                                continue
                            _k = supersede_key_str(_f)
                            if _k:
                                _by_key.setdefault(_k, []).append(_f)
                        _key_index_built = True
                    cands = _by_key.get(sk, [])
                    if len(cands) == 1:
                        fact = cands[0]
                        rescued_ids += 1
                    elif len(cands) > 1:
                        # 🔴 同键多条 live 是**已知存量状态**（MQ-S33：92 键 / 217 条），
                        # 说明"同键新压旧"没执行到位。此时接给谁都是猜——
                        # 猜错的方向是把命中记到错误条目上，比丢了更坏（污染排序信号）。
                        ambiguous_ids += 1
                        orphan_ids += 1
                        orphan_hits += int(n)
                        continue
                if fact is None:
                    orphan_ids += 1
                    orphan_hits += int(n)
                    continue
            fact.ref_count = int(getattr(fact, "ref_count", 0) or 0) + int(n)
            age_s = 0.0
            t_obs = getattr(fact, "t_observed", None) or fact.created_at
            try:
                _t = t_obs if t_obs.tzinfo else t_obs.replace(tzinfo=_UTC)
                age_s = max(0.0, (now - _t).total_seconds())
            except (TypeError, AttributeError):
                pass
            fact.importance = compute_importance(
                fact.item_kind.value if hasattr(fact.item_kind, "value") else str(fact.item_kind),
                strength=fact.strength, ref_count=fact.ref_count, age_seconds=age_s,
                rating=int(getattr(fact, "importance_rating", 0) or 0),
            )
            # M3-1：相关钟。取**这一轮的时间**不取 now —— 重放两次要逐字节一致（G6）。
            _hts = (hit_ts or {}).get(fid, "")
            if _hts:
                try:
                    _t2 = _dtt.fromisoformat(_hts)
                    if _t2.tzinfo is None:
                        _t2 = _t2.replace(tzinfo=_UTC)
                    _prev = getattr(fact, "last_hit_at", None)
                    _prev = (_prev if _prev is None or _prev.tzinfo
                             else _prev.replace(tzinfo=_UTC))
                    if _prev is None or _t2 > _prev:
                        fact.last_hit_at = _t2   # 只前进不后退
                except ValueError:
                    pass
            fact.updated_at = now
            fact.embedding = None  # 只 upsert meta
            try:
                self.add_fact(fact)
                updated += 1
            except Exception as e:  # noqa: BLE001
                logger.warning("inject_refcount_update_failed", fact_id=fid, error=str(e))
        if updated:
            logger.info("inject_refcount_written", updated=updated,
                        hits=sum(hits.values()))
        if rescued_ids:
            logger.info("inject_refcount_rescued_by_key", rescued=rescued_ids,
                        hint="id 换代后按取代键接续（MQ-S38）")
        if orphan_ids:
            # warning 级：这是**数据丢失事件**（历史命中信号进不了库），
            # 与 ADR-0017 把 `enqueue_skipped_no_redis` 从 debug 提到 warning 同款理由。
            _total = orphan_ids + updated
            logger.warning(
                "inject_refcount_orphan_ids",
                orphan_ids=orphan_ids, orphan_hits=orphan_hits,
                ambiguous_ids=ambiguous_ids, rescued_ids=rescued_ids,
                resolved=updated, total_ids=_total,
                orphan_pct=round(100.0 * orphan_ids / _total, 1) if _total else 0.0,
                hint="fact_id 内容派生：蒸馏 prompt 换版会让全库 id 换代。"
                     "带取代键的可接续；无键（lesson/procedure）或同键多条"
                     "live（MQ-S33）的仍会丢（MQ-S38）",
            )
        self.last_injection_orphans = (orphan_ids, orphan_hits)
        self.last_injection_rescued = rescued_ids
        self.last_injection_ambiguous = ambiguous_ids
        return updated

    def get_fact(self, fact_id: str) -> Fact | None:
        """读取一条 Fact（从 metadata RocksDB）。"""
        # A1/T3: 惰性重试 + 追新
        self._ensure_meta_db()
        self._try_catch_up()
        if self.meta_db is None:
            return None
        key = f"{_FACT_PREFIX}{fact_id}"
        raw = self.meta_db.get(key.encode())
        if raw is None:
            return None
        data = msgpack.unpackb(raw, raw=False)
        return Fact.model_validate(data)

    def all_facts(self) -> list[Fact]:
        """读取所有 Fact（用于 novelty 检测 + 重建）。"""
        if self.meta_db is None:
            return []
        facts: list[Fact] = []
        for key_raw, value_raw in self.meta_db.items():
            key_str = key_raw.decode() if isinstance(key_raw, bytes) else str(key_raw)
            if not key_str.startswith(_FACT_PREFIX):
                continue
            data = msgpack.unpackb(value_raw, raw=False)
            facts.append(Fact.model_validate(data))
        return facts

    def fact_count(self) -> int:
        """Fact 总数（调试/测试 + `/admin/status` 面板）。

        MQ-R3（2026-08-17）：这里**必须**和 `get_fact` 一样先追新。proxy 以 secondary
        模式打开 RocksDB，不 `try_catch_up_with_primary()` 就只见上次 catch-up 时刻的
        快照——症状是"数字看着正常，只是不动"（实测 status 报 1775 而同刻真实 1924）。
        """
        # MQ-R3: 惰性重试 + 追新（同 get_fact:2816）
        self._ensure_meta_db()
        self._try_catch_up()
        if self.meta_db is None:
            return 0
        return sum(1 for k in self.meta_db.keys()
                   if (k.decode() if isinstance(k, bytes) else str(k)).startswith(_FACT_PREFIX))

    def clear(self) -> None:
        """清空 Memory Index（用于重建）。"""
        # 清 metadata（含 fact / matter / edge / 消费游标 / 遗留水位）
        # 🔴 有意不清（ADR-0018 §4.1 收录原则：不可确定性重算的记录）：
        #   distill/ ｜ judgment/ ｜ judgment_taskstate/（G12.2 新开判定——清了它，
        #   重放的首轮集合就随旋钮漂，matter_id 跟着漂 = §0.1b 重演）｜ __meta__/*
        # 想"顺手清干净"之前先读 ADR-0031 §14.1。
        keys_to_delete = []
        for key_raw in self.meta_db.keys():
            key_str = key_raw.decode() if isinstance(key_raw, bytes) else str(key_raw)
            if (key_str.startswith(_FACT_PREFIX)
                    or key_str.startswith(_MATTER_PREFIX)
                    or key_str.startswith(_EDGE_PREFIX)
                    or key_str.startswith(_CONSUMED_PREFIX)
                    or key_str.startswith(_DISTILL_RETRY_PREFIX)
                    or key_str.startswith(_PROFILE_PREFIX)
                    or key_str == _INDEX_WATERMARK_KEY):
                keys_to_delete.append(key_raw)
        for k in keys_to_delete:
            del self.meta_db[k]

        # ADR-0028 E6.2：词法倒排随库清（rebuild 时重建）
        fts = self._ensure_fts()
        if fts is not None:
            fts.clear()

        # 清 LanceDB（facts + matters）
        if self._lancedb is not None:
            try:
                self._lancedb.drop_table("facts")
            except Exception as e:
                logger.debug("index_drop_facts_skip", error=str(e))
            try:
                self._lancedb.drop_table("matters")
            except Exception as e:
                logger.debug("index_drop_matters_skip", error=str(e))
            self._table = None
            self._matter_table = None

        self._invalidate_manual_edge_cache()
        logger.info("index_cleared", deleted_keys=len(keys_to_delete))

    # ── ADR-0012 §3.1/3.2: Matter 卡 + 归属边存储 ──

    def add_matter(self, matter: Matter) -> None:
        """存一个 Matter 卡（metadata + 质心向量到 LanceDB）。"""
        meta = matter.model_dump(mode="json", exclude={"embedding"})
        key = f"{_MATTER_PREFIX}{matter.matter_id}"
        self.meta_db[key.encode()] = msgpack.packb(meta, use_bin_type=True)

        if matter.centroid is not None and self._lancedb is not None:
            self._add_matter_to_lancedb(matter)

        logger.debug("index_matter_added", matter_id=matter.matter_id,
                      title_len=len(matter.title))

    def _ensure_matter_table(self, dim: int) -> Any:
        """确保 LanceDB matters 表存在（惰性创建）。"""
        if self._matter_table is not None:
            return self._matter_table

        import pyarrow as pa

        schema = pa.schema([
            pa.field("matter_id", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
            pa.field("title", pa.string()),
            pa.field("summary", pa.string()),
            pa.field("status", pa.string()),
            pa.field("origin", pa.string()),
        ])

        try:
            self._matter_table = self._lancedb.create_table(
                "matters", schema=schema, mode="create",
            )
        except Exception:
            self._matter_table = self._lancedb.open_table("matters")

        return self._matter_table


    # ── ANN 索引（T8b）──────────────────────────────────────────────
    # 索引不在 _ensure_table / _ensure_matter_table 中自动创建--空表上建索引
    # 无意义。索引在数据量足够后由 consolidator 调 rebuild_ann_indexes() 创建。
    # 查询侧：若 BLADEX_ANN_BYPASS=1 则跳过索引走暴力扫描（回滚开关）。

    _ANN_TABLE_NAMES = ("facts", "matters")
    # 索引维护最小间隔：consolidator 60s 一轮，别每轮都去 list_indices
    _ANN_MAINTAIN_INTERVAL_S = 300.0

    @staticmethod
    def _compute_ann_params(nrows: int) -> dict:
        """根据行数计算索引参数（IVF_FLAT + 经验法则）。

        IVF 分区数 ≈ 4√N（取 2 的幂，上界 256）。
        N < 256 行不建索引（全表扫描更快）。

        Returns dict with keys: num_partitions, index_type；或 {} 表示 N < 256。

        **不再返回 nprobes**（2026-07-29）：原 "num_partitions/10" 推荐值经真实 e5
        向量实测反而不如 LanceDB 默认（实测表见 `_ann_nprobes_override`），留着只会
        诱导下一个人再去"把它接上"。要调 nprobes 走 `BLADEX_ANN_NPROBES`。
        """
        if nrows < 256:
            return {}
        raw = int(4 * (nrows ** 0.5))
        np_val = 1
        while np_val * 2 <= raw:
            np_val *= 2
        np_val = max(8, min(256, np_val))
        return {
            "num_partitions": np_val,
            "index_type": "IVF_FLAT",
        }

    def _ann_bypass_enabled(self) -> bool:
        """BLADEX_ANN_BYPASS=1 时跳过索引，走暴力全量扫描。"""
        return os.environ.get("BLADEX_ANN_BYPASS", "") == "1"

    def _ann_nprobes_override(self) -> int | None:
        """`BLADEX_ANN_NPROBES` 显式指定 nprobes；未设 = 用 LanceDB 默认（B18）。

        **为什么不自动套用"4√N/10"推荐值**（2026-07-29 实测驱动，`scripts/t8b_ann_benchmark.py`）：

        | 档位 | 推荐值 | recall@10 均值 / 最差 | LanceDB 默认 | 延迟差 |
        |---|---|---|---|---|
        | 1119（**真实 e5 向量**） | 12 | **0.950 / 0.70** | **0.973 / 0.80** | 0.12ms |
        | 5000（自举） | 25 | 1.000 / 1.00 | 0.997 / 0.90 | 0.10ms |
        | 10000（自举） | 25 | 0.993 / 0.80 | 0.993 / 0.80 | ~0 |

        推荐值在**唯一一档真实数据**上反而更差，其余档位打平，延迟收益在噪声级别。
        既然没有可测的收益，就不该在读路径上悄悄改变检索行为——所以这个启发式
        **退出自动路径**（已从 `_compute_ann_params` 的返回值里删除），
        只保留一个显式旋钮：需要调的人自己 `BLADEX_ANN_NPROBES=25`。

        这也是对卡里 B18「要么被用上、要么从返回值里删掉」的回答：**删掉**。
        """
        raw = os.environ.get("BLADEX_ANN_NPROBES", "").strip()
        if not raw:
            return None
        try:
            val = int(raw)
        except ValueError:
            logger.warning("index_ann_nprobes_invalid", value=raw)
            return None
        return val if val >= 1 else None

    def _get_lance_table(self, table_name: str) -> Any:
        """获取已打开的 LanceDB table，未打开则惰性打开。"""
        if table_name == "facts":
            self._lazy_open_facts_table()
            return self._table
        elif table_name == "matters":
            self._lazy_open_matter_table()
            return self._matter_table
        return None

    def ensure_ann_index(self, table_name: str, force: bool = False) -> bool:
        """对指定表创建 ANN 索引。

        幂等：已有索引且 force=False 时跳过。
        force=True 时先 drop 旧索引再重建（用于数据量变化后调整 num_partitions）。
        返回 True 表示索引已就绪，False 表示数据量太小或表不存在。
        """
        table = self._get_lance_table(table_name)
        if table is None:
            return False

        nrows = table.count_rows()
        params = self._compute_ann_params(nrows)
        if not params:
            logger.info("index_ann_skip_too_small", table=table_name, nrows=nrows)
            return False

        existing = list(table.list_indices())
        if existing and not force:
            logger.info("index_ann_already_exists", table=table_name,
                        indices=[str(e) for e in existing])
            return True

        # force=True: drop old index first
        if existing and force:
            for idx in existing:
                idx_name = idx.name if hasattr(idx, "name") else str(idx)
                try:
                    table.drop_index(idx_name)
                    logger.info("index_ann_index_dropped", table=table_name, name=idx_name)
                except Exception as e:
                    logger.warning("index_ann_drop_failed", table=table_name,
                                   name=idx_name, error=str(e))

        table.create_index(
            metric="cosine",
            num_partitions=params["num_partitions"],
            index_type=params["index_type"],
        )
        logger.info("index_ann_index_created", table=table_name,
                    num_partitions=params["num_partitions"],
                    nrows=nrows)
        return True

    def maintain_ann_indexes(self, min_interval_s: float | None = None) -> bool:
        """限流版索引维护：距上次尝试不足 min_interval_s 则跳过（T8b B17）。

        2026-07-29 复核发现：`rebuild_ann_indexes()` 只挂在 `rebuild_from_hub` **尾部**，
        而"无新 turn"分支在中途 `return 0` —— 空闲期永远到不了那一行。
        真实后果：consolidator 12:51 重启后跑了一串 `index_rebuild_no_new_turns`，
        `index_ann_*` 事件 **0 条**，看上去像"接线没生效"，实际是够不着。

        索引维护本就该与"这一轮有没有新 turn"解耦——它取决于**库有多大**，不取决于
        这 60 秒里有没有人说话。故拆出本方法，两条路径都调，用时间窗限流
        （默认 `_ANN_MAINTAIN_INTERVAL_S`）避免每 60s 空转 list_indices。
        """
        interval = (
            self._ANN_MAINTAIN_INTERVAL_S if min_interval_s is None else min_interval_s
        )
        now = time.monotonic()
        if self._ann_last_maintain_ts and (now - self._ann_last_maintain_ts) < interval:
            return False
        self._ann_last_maintain_ts = now
        self.rebuild_ann_indexes()
        return True

    def recompute_lifecycle(self, *, force: bool = False) -> int:
        """ADR-0028 E2.4：生命周期重算 job（"睡眠巩固"的最小实现）。

        为什么需要它：`compute_importance` 里的时间半衰减项 `0.5^(age/half_life)`
        只在**写这条 fact 的那一刻**被算过一次。没有周期性重算，"旧记忆自然降权"
        这件事在库里从来没有真的发生过——一条一年前的 fact 的 importance
        与它刚写入时逐字相同。

        算法（全量，N≈10³–10⁵，O(N) 可接受）：
            importance = base(kind)
                       × (1 + 0.30·ln(strength))
                       × (1 + 0.25·ln(1 + ref_count))
                       × 0.5^(age_days / 30)
        即 U5.4 既有公式原样，只是让 age 项按**当前时间**重新求值。
        |Δ| > 0.01 才写回（省写放大）。

        节流：`__meta__/lifecycle_last_run` 时间戳 + `BLADEX_LIFECYCLE_INTERVAL_S`
        （默认 6h）。force=True 忽略节流（供测试/手动触发）。

        读侧遗忘：低分条目退出召回由 search 的 `BLADEX_MIN_IMPORTANCE` 门槛负责
        ——**不做物理删除**，meta/Memory Hub 全保留、as-of 可查（G6 重建等价性）。

        返回实际写回条数；-1 表示因节流未执行。
        """
        if self._read_only or self.meta_db is None:
            return 0

        import time as _time

        from bladex_core.flags import flag_number
        from bladex_core.importance import compute_importance

        interval = flag_number("BLADEX_LIFECYCLE_INTERVAL_S")
        now_ts = _time.time()
        if not force:
            raw = self.meta_db.get(_LIFECYCLE_LAST_RUN_KEY.encode())
            if raw is not None:
                try:
                    last = float(msgpack.unpackb(raw, raw=False))
                except Exception:  # noqa: BLE001 —— 坏值当没跑过
                    last = 0.0
                if interval > 0 and (now_ts - last) < interval:
                    return -1

        from datetime import UTC as _UTC
        from datetime import datetime as _dtt

        now = _dtt.now(_UTC)
        updated = 0
        scanned = 0
        for fact in self.all_facts():
            scanned += 1
            t_obs = getattr(fact, "t_observed", None) or fact.created_at
            age_s = 0.0
            try:
                _t = t_obs if t_obs.tzinfo else t_obs.replace(tzinfo=_UTC)
                age_s = max(0.0, (now - _t).total_seconds())
            except (TypeError, AttributeError):
                pass
            new_imp = compute_importance(
                fact.item_kind.value if hasattr(fact.item_kind, "value") else str(fact.item_kind),
                strength=fact.strength,
                ref_count=int(getattr(fact, "ref_count", 0) or 0),
                age_seconds=age_s,
            )
            if abs(new_imp - float(fact.importance)) <= _LIFECYCLE_MIN_DELTA:
                continue
            fact.importance = new_imp
            fact.updated_at = now
            fact.embedding = None  # 只 upsert meta，不重复写向量
            try:
                self.add_fact(fact)
                updated += 1
            except Exception as e:  # noqa: BLE001
                logger.warning("index_lifecycle_update_failed", fact_id=fact.id, error=str(e))

        # M3-3：时效到期 —— `valid_until` 已过的 fact 置 `t_invalid`（退出 current-only）。
        # **数据不删**、as-of 可查（G6）。t_invalid 取 `valid_until` 本身而不是 now：
        # "它是什么时候失效的"是个客观事实，取 now 会让重放两次得到不同的库。
        expired = self._expire_by_valid_until(now)

        # M3-2：Matter 三时钟判定（只判不动 → 判定结果在这里落库）
        clocks = self.recompute_matter_clocks(now=now)

        self.meta_db[_LIFECYCLE_LAST_RUN_KEY.encode()] = msgpack.packb(
            now_ts, use_bin_type=True)
        logger.info("index_lifecycle_recomputed", scanned=scanned, updated=updated,
                    interval_s=interval, expired=expired,
                    dormant=clocks.get("applied", 0),
                    clock_observe_only=clocks.get("observe_only", False))
        return updated

    def _expire_by_valid_until(self, now) -> int:
        """M3-3：把已到期的 fact 置 `t_invalid`。返回条数。

        豁免表照旧生效：preference/hard_rule/lesson 即便带了 `valid_until`
        也不驱逐 —— 蒸馏偶尔会给偏好类猜一个到期点，那多半是幻觉，
        而"静默丢一条用户偏好"是 G3 的原教训。
        """
        from bladex_core.clocks import exempt_from_clocks, is_expired

        n = 0
        for fact in self.all_facts():
            if fact.t_invalid is not None or fact.valid_until is None:
                continue
            kind = (fact.item_kind.value if hasattr(fact.item_kind, "value")
                    else str(fact.item_kind))
            if exempt_from_clocks(kind):
                continue
            if not is_expired(fact.valid_until, now):
                continue
            fact.t_invalid = fact.valid_until   # 客观时点，不取 now（重放等价）
            fact.updated_at = now
            fact.embedding = None
            try:
                self.add_fact(fact)
                n += 1
            except Exception as e:  # noqa: BLE001
                logger.warning("index_expire_failed", fact_id=fact.id, error=str(e))
        if n:
            logger.info("index_facts_expired", expired=n)
        return n

    def recompute_matter_clocks(self, *, now=None, dry_run: bool = False) -> dict:
        """M3-2：Matter 三时钟判定。**只判不删**（dormant = 退出常驻注入、保留可召回）。

        `dry_run=True` 只返回清单不落库 —— 卡里要求的"live 副本干跑，
        人工确认将被 dormant 的清单是否合理"就走这条路。

        观测期护栏（5.9 修正 3）：`last_hit_at` 覆盖率不足时**只 log 不动状态**。
        M3-1 刚上线时覆盖率必然是 0，M5 全量重建又会清零 ——
        "上线后一段时间只 log"是设计使然，不是没生效。
        """
        from datetime import UTC as _UTC
        from datetime import datetime as _dtt

        from bladex_core.clocks import decide_matter_clocks

        now = now or _dtt.now(_UTC)
        matters = [m for m in self.all_matters()
                   if m.matter_id != UNASSIGNED_MATTER_ID]
        # MS-4②：带 completed 语义信号的卡 → 更新钟阈值减半（加速证据，非判据）
        signals = {
            m.matter_id for m in matters
            if any((e.event or "").lower() in ("completed", "accepted")
                   for e in (m.lifecycle or []))
        }
        report = decide_matter_clocks(matters, now=now, completion_signals=signals)

        applied = 0
        if not dry_run and not report.observe_only and not self._read_only:
            by_id = {m.matter_id: m for m in matters}
            for d in report.to_apply():
                m = by_id.get(d.matter_id)
                if m is None or m.status == d.target_status:
                    continue
                m.status = d.target_status
                try:
                    self.add_matter(m)
                    applied += 1
                except Exception as e:  # noqa: BLE001
                    logger.warning("index_clock_apply_failed",
                                   matter_id=d.matter_id, error=str(e))
        from collections import Counter as _QC
        logger.info("index_matter_clocks",
                    scanned=report.scanned, coverage=round(report.coverage, 3),
                    observe_only=report.observe_only, applied=applied,
                    quadrants=dict(_QC(d.quadrant for d in report.decisions)),
                    completion_signals=len(signals), dry_run=dry_run)
        return {
            "scanned": report.scanned, "coverage": report.coverage,
            "observe_only": report.observe_only, "applied": applied,
            "decisions": report.decisions,
        }

    def rebuild_ann_indexes(self, force: bool = False) -> dict[str, bool]:
        """重建所有表的 ANN 索引。

        force=False（默认）：幂等，已有索引则跳过。
        force=True：先 drop 再重建（用于数据量显著变化后调整参数）。
        由 consolidator 在 bulk insert 后调用。
        """
        results: dict[str, bool] = {}
        for name in self._ANN_TABLE_NAMES:
            try:
                results[name] = self.ensure_ann_index(name, force=force)
            except Exception as e:
                logger.warning("index_ann_rebuild_failed", table=name, error=str(e))
                results[name] = False
        return results

    def drop_ann_index(self, table_name: str) -> bool:
        """删除指定表的 ANN 索引。

        使用 LanceDB 0.33 的 drop_index(name) 真正删除索引文件，
        之后查询自动回退到暴力全量扫描。
        """
        table = self._get_lance_table(table_name)
        if table is None:
            return False
        existing = list(table.list_indices())
        if not existing:
            return False
        for idx in existing:
            idx_name = idx.name if hasattr(idx, "name") else str(idx)
            try:
                table.drop_index(idx_name)
                logger.info("index_ann_index_dropped", table=table_name, name=idx_name)
            except Exception as e:
                logger.warning("index_ann_drop_failed", table=table_name,
                               name=idx_name, error=str(e))
        return True

    def get_ann_index_status(self) -> dict[str, dict]:
        """返回所有表的 ANN 索引状态。"""
        status: dict[str, dict] = {}
        for name in self._ANN_TABLE_NAMES:
            table = self._get_lance_table(name)
            if table is None:
                status[name] = {"has_index": False, "nrows": 0, "error": "table not open"}
                continue
            try:
                nrows = table.count_rows()
                indices = list(table.list_indices())
                has_index = len(indices) > 0
                params = self._compute_ann_params(nrows)
                status[name] = {
                    "has_index": has_index,
                    "nrows": nrows,
                    "index_count": len(indices),
                    "index_info": [str(e) for e in indices],
                    "recommended_params": params,
                    "bypass": self._ann_bypass_enabled(),
                }
            except Exception as e:
                status[name] = {"has_index": False, "nrows": 0, "error": str(e)}
        return status

    def _apply_ann_to_search(
        self, search_builder: Any, nprobes: int | None = None,
        table_name: str | None = None,
    ) -> Any:
        """在 search builder 上应用 ANN 设置。

        优先级：`BLADEX_ANN_BYPASS=1`（回滚，走暴力扫描）> 调用方显式 nprobes >
        `BLADEX_ANN_NPROBES` 环境覆盖 > 不设（LanceDB 默认，见 `_ann_nprobes_override`）。

        `table_name` 仅用于日志定位，不参与取值——nprobes 不再按表规模自动推导，
        理由见 `_ann_nprobes_override` 的实测表。
        """
        if self._ann_bypass_enabled():
            return search_builder.bypass_vector_index()
        if nprobes is None:
            nprobes = self._ann_nprobes_override()
        if nprobes is not None:
            return search_builder.nprobes(nprobes)
        return search_builder

    def _add_matter_to_lancedb(self, matter: Matter) -> None:
        """把 Matter 的质心向量写入 LanceDB（**upsert**：先删同 id 旧行再加）。

        🔴 2026-08-05 复核 §3.4 修复：本方法原来只 `add` 不删，而 rebuild 中
        participants / lifecycle / 质心每次更新都会调 `add_matter` —— 真实库实测
        **matters.lance 214 行 vs meta 里只有 20 个 Matter**（10.7 倍冗余）。
        后果不只是体积：`search_matters` top-k 会被同一个 Matter 的多份陈旧行占满，
        k=3 实际可能只召回 1 个 Matter（两级召回的覆盖面被自己吃掉了）。
        """
        import pyarrow as pa
        dim = len(matter.centroid)
        table = self._ensure_matter_table(dim)

        try:
            safe = matter.matter_id.replace("'", "''")
            table.delete(f"matter_id = '{safe}'")
        except Exception as e:  # noqa: BLE001 —— 首次写入无旧行；删失败退回旧的只追加行为
            logger.debug("index_matter_lancedb_delete_skip",
                         matter_id=matter.matter_id, error=str(e))

        data = pa.table({
            "matter_id": [matter.matter_id],
            "vector": [matter.centroid],
            "title": [matter.title],
            "summary": [matter.summary],
            "status": [matter.status.value],
            "origin": [matter.origin.value],
        })
        table.add(data)

    def get_matter(self, matter_id: str) -> Matter | None:
        """读取一个 Matter 卡。"""
        # A1/T3: 惰性重试 + 追新
        self._ensure_meta_db()
        self._try_catch_up()
        if self.meta_db is None:
            return None
        key = f"{_MATTER_PREFIX}{matter_id}"
        raw = self.meta_db.get(key.encode())
        if raw is None:
            return None
        data = msgpack.unpackb(raw, raw=False)
        return Matter.model_validate(data)

    # ── V-I3：keyword 跨 fact/matter 关联（ADR-0032 §6.3/6.4）──────────────

    def keyword_lookup(self, raw_key: str, *, top_facts: int = 30,
                       ) -> tuple[list[Matter], list[Fact]]:
        """按规范化 keyword 取关联的 Matter 与 Fact（工具面后端，非注入热路径）。

        key 空间 = V-I2 的 `canonical_entity_key`——fact.entities 与
        matter.topic_keys 用**同一把尺**（分隔符变体/CJK 粘连在此折叠，
        `qwen3.8:27b` vs `qwen3.8-27B` 那类病例不再分家）。
        失效的 fact（t_invalid 非空）不返回；exposure 过滤归调用方
        （工具结果也是注入，ADR-0032 §3.2-7——守卫在 agency 侧统一做）。
        """
        from bladex_core.topic_keys import canonical_entity_key
        key = canonical_entity_key(raw_key or "")
        if not key:
            return [], []
        matters = [m for m in self.all_matters()
                   if any(canonical_entity_key(str(k)) == key
                          for k in (m.topic_keys or {}))]
        facts: list[Fact] = []
        for fid in self._keyword_fact_map().get(key, []):
            f = self.get_fact(fid)
            if f is not None and not getattr(f, "t_invalid", ""):
                facts.append(f)
            if len(facts) >= top_facts:
                break
        return matters, facts

    def _keyword_fact_map(self) -> dict[str, list[str]]:
        """canonical keyword -> fact_ids 倒排（惰性建）。

        失效戳 = facts 表行数：consolidator 增量写入后行数变即重建。工具路径的
        延迟预算是秒级（内循环里跑），几千行的全列扫描是几百 ms——**不接进
        注入热路径**（那边有 800ms 预算与降级语义，别把这张表塞进去）。
        行数不变但内容原地更新的窗口里可能读到旧表——工具面语义可接受
        （下一轮写入即自愈），不为它加代价更高的版本轴。
        """
        self._lazy_open_facts_table()
        if self._table is None:
            return {}
        try:
            stamp = int(self._table.count_rows())
        except Exception as e:  # noqa: BLE001 —— 表暂不可读 = 用旧缓存
            logger.debug("index_keyword_map_stamp_failed", error=str(e))
            return self._kw_map or {}
        if self._kw_map is not None and stamp == self._kw_map_stamp:
            return self._kw_map
        from bladex_core.topic_keys import canonical_entity_key
        try:
            tbl = self._table.to_arrow()
            ids = tbl.column("id").to_pylist()
            ents = tbl.column("entities").to_pylist()
        except Exception as e:  # noqa: BLE001
            logger.warning("index_keyword_map_build_failed", error=str(e))
            return self._kw_map or {}
        m: dict[str, list[str]] = {}
        for fid, es in zip(ids, ents):
            for e in es or []:
                k = canonical_entity_key(str(e))
                if k:
                    m.setdefault(k, []).append(str(fid))
        self._kw_map, self._kw_map_stamp = m, stamp
        logger.info("index_keyword_map_built", keys=len(m), facts=len(ids))
        return m

    def all_matters(self) -> list[Matter]:
        """读取所有 Matter 卡。"""
        if self.meta_db is None:
            return []
        matters: list[Matter] = []
        for key_raw, value_raw in self.meta_db.items():
            key_str = key_raw.decode() if isinstance(key_raw, bytes) else str(key_raw)
            if not key_str.startswith(_MATTER_PREFIX):
                continue
            data = msgpack.unpackb(value_raw, raw=False)
            matters.append(Matter.model_validate(data))
        return matters

    def _find_matter_by_native_key(self, normalized_key: str) -> Matter | None:
        """按**规范化标题**找一张已存在的卡（ADR-0031 §3.4 解耦后的同名汇入）。

        以前"同标题汇入同一张卡"是 `matter_id = sha256(标题)` 的副产品——
        算出同一个 id 就自然接上了。id 与标题解耦之后，这个**行为**要显式保留，
        否则解耦会顺手把它删掉（那是碎片化的反方向，代价比解耦本身还大）。

        只认 `title` 与 `aliases`（= `_native_match_keys`，与 L3 同一套匹配面），
        不认累积 `entities`——MS-16 那条雪球机理（错边混入外题实体 → 后续同题
        fact 以 conf 0.9 持续链入）在这里同样成立。

        O(n) 全表扫（当前库 ~1000 张卡），只在消化路每组调一次，不在热路径。
        """
        if not normalized_key:
            return None
        for m in self.all_matters():
            if m.status == MatterStatus.CLOSED:
                continue
            if normalized_key in _native_match_keys(m):
                return m
        return None

    def update_matter(self, matter: Matter) -> None:
        """更新 Matter 卡（覆写 metadata，可选更新向量）。"""
        from datetime import UTC, datetime
        matter.updated_at = datetime.now(UTC)
        meta = matter.model_dump(mode="json", exclude={"embedding"})
        key = f"{_MATTER_PREFIX}{matter.matter_id}"
        self.meta_db[key.encode()] = msgpack.packb(meta, use_bin_type=True)
        logger.debug("index_matter_updated", matter_id=matter.matter_id,
                      status=matter.status.value)


    #: 行数 / Matter 数超过此倍数才触发存量压缩（真实库实测 10.7 倍）
    _MATTER_COMPACT_RATIO = 2.0

    def maybe_compact_matter_vectors(self) -> bool:
        """冗余明显（行数 > Matter 数 × 2）时才压缩——避免每个空闲轮都重写整表。"""
        self._lazy_open_matter_table()
        if self._read_only or self._matter_table is None:
            return False
        try:
            rows = self._matter_table.count_rows()
        except Exception:  # noqa: BLE001
            return False
        n = len([m for m in self.all_matters() if m.centroid])
        if n == 0 or rows <= n * self._MATTER_COMPACT_RATIO:
            return False
        self.compact_matter_vectors()
        return True

    def compact_matter_vectors(self) -> tuple[int, int]:
        """复核 §3.4 存量清理：matters 向量表按 matter_id 压缩成"每卡一行"。

        写侧已改 upsert（不再堆积），但存量还在——真实库实测 214 行 / 20 个 Matter。
        本方法整表重写：每个仍存在于 meta 的 Matter 保留一行（用 meta 里的**当前**
        质心与标题，而不是某份陈旧向量行），meta 里已不存在的 Matter 行直接丢弃。

        返回 (压缩前行数, 压缩后行数)。read_only 或表不存在 → (0, 0)。
        """
        self._lazy_open_matter_table()
        if self._read_only or self._lancedb is None or self._matter_table is None:
            return 0, 0
        try:
            before = self._matter_table.count_rows()
        except Exception as e:  # noqa: BLE001
            logger.warning("index_matter_compact_count_failed", error=str(e))
            return 0, 0

        matters = [m for m in self.all_matters() if m.centroid]
        if not matters:
            return before, before

        import pyarrow as pa

        dim = len(matters[0].centroid)
        try:
            self._lancedb.drop_table("matters")
        except Exception as e:  # noqa: BLE001
            logger.warning("index_matter_compact_drop_failed", error=str(e))
            return before, before
        self._matter_table = None
        table = self._ensure_matter_table(dim)
        table.add(pa.table({
            "matter_id": [m.matter_id for m in matters],
            "vector": [m.centroid for m in matters],
            "title": [m.title for m in matters],
            "summary": [m.summary for m in matters],
            "status": [m.status.value for m in matters],
            "origin": [m.origin.value for m in matters],
        }))
        after = table.count_rows()
        logger.info("index_matter_vectors_compacted", before=before, after=after)
        return before, after

    # ── M0-7（复核 I3 / H6 / G6）：僵尸向量清理 + facts 向量表压缩 ─────────────

    #: facts 向量表的碎片比（行数 / meta 中有向量的 fact 数）超过此值才压缩
    _FACTS_COMPACT_RATIO = 2.0

    def reconcile_vector_meta(self) -> tuple[int, int]:
        """向量表与 meta 对账：删掉**没有 meta 的僵尸向量行**（I3）。

        为什么会有僵尸：LanceDB 侧是"先加行、后写 meta"，中途失败/进程被杀就留下
        只有向量没有事实的孤儿行；墓碑级联删 meta 时若 `self._table` 尚未打开
        （惰性），那次删除也会漏掉向量行。实测 live 库 49 条僵尸——它们照常参与
        top-k 检索，随后 `get_fact` 返回 None 被丢弃，于是**白占召回位**：
        40 条随机查询的 top-15 里有 14% 是这种空位。

        返回 (扫描到的向量行数, 删除的僵尸行数)。read_only / 无表 → (0, 0)。
        """
        self._lazy_open_facts_table()
        if self._read_only or self._table is None or self.meta_db is None:
            return 0, 0
        try:
            rows = self._table.to_arrow().column("id").to_pylist()
        except Exception as e:  # noqa: BLE001 —— 卫生作业失败不该影响主管线
            logger.warning("index_vector_reconcile_scan_failed", error=str(e))
            return 0, 0
        zombies = []
        seen: set[str] = set()
        for fid in rows:
            if not fid or fid in seen:
                continue
            seen.add(fid)
            if self.meta_db.get(f"{_FACT_PREFIX}{fid}".encode()) is None:
                zombies.append(fid)
        if not zombies:
            logger.info("index_vector_reconcile_clean", scanned=len(rows))
            return len(rows), 0
        removed = 0
        for fid in zombies:
            try:
                self._table.delete(f"id = '{fid.replace(chr(39), chr(39) * 2)}'")
                removed += 1
            except Exception as e:  # noqa: BLE001
                logger.debug("index_vector_reconcile_delete_skip", fact_id=fid, error=str(e))
        logger.warning("index_vector_zombies_removed",
                       scanned=len(rows), removed=removed)
        return len(rows), removed

    #: 压缩时的版本清理窗口（秒）。**三个候选值里选中间那个，两头都试过/查过**：
    #:   0（`scripts/compact_index.py` 用的）——立刻删掉全部旧版本，所以那个脚本
    #:     要求先停 proxy 与 consolidator：会抽掉在线只读进程正握着的版本；
    #:   LanceDB 默认（天级）——旧文件长期留在 `data/`，而判据数的就是那里的文件，
    #:     于是维护**永远满足不了自己的退出条件**（gate 抓到的首版形态）；
    #:   60s——proxy 每次 search 前都 `checkout_latest`（`_refresh_lance_read`），
    #:     单次读持有一个版本的时间以毫秒计，60 秒是数量级以上的余量。
    _COMPACT_CLEANUP_S = 60.0
    #: 两次 facts 压缩的最小间隔（秒）。**必须大于 `_COMPACT_CLEANUP_S`**：
    #: 清理窗口内旧文件还在盘上、判据仍然超阈值，没有这条节流就会在窗口内空压若干次。
    _COMPACT_MIN_INTERVAL_S = 120.0

    def fact_vector_fragments(self) -> int:
        """facts 向量表的碎片（datafile）数；目录不存在返回 -1。

        数的是磁盘上的 `facts.lance/data/`——**live 那个 1619 就是这么量的**，
        阈值标定与它同源。不用 LanceDB 的 `to_lance().get_fragments()`
        （"当前版本有几个 fragment"，语义更准）是因为**本机没装 pylance**：
        2026-08-17 gate 实测 `The lance library is required ... pip install pylance`，
        整条路不通。同一原因让 `_entity_freq` 的列投影一直在静默回落到全表
        `to_arrow()`——那是 MQ-R6 的一条线索，另账。

        代价（必须知道）：压缩刚做完时，合并出的新文件是净增，旧版本的文件要等
        `_COMPACT_CLEANUP_S` 过去、**下一次**压缩才被清掉。所以本读数在一次压缩后
        **不会立刻降**，两次之后才降到位。别把第一次的读数读成"压缩没生效"——
        看日志里的 `index_fact_vectors_compacted` 才是"这次做了没做"。
        """
        data_dir = self._path / "lancedb" / "facts.lance" / "data"
        try:
            return len(os.listdir(data_dir))
        except OSError:
            return -1

    # ── MQ-I15 ②（2026-09-18）：LanceDB 版本 vacuum，三表统一 ───────────────────

    #: 受版本 vacuum 管的三张 LanceDB 表（表名 = `<name>.lance` 目录名）
    _LANCE_TABLES: tuple[str, ...] = ("facts", "files", "matters")

    def lance_version_count(self, table: str) -> int:
        """`<table>.lance/_versions/` 里的 manifest 数；目录不存在返回 -1。

        与 `fact_vector_fragments` 同一口径：量的是**盘上**的东西——MQ-I15 的读数
        （files 7,670 版本 / 242MB）就是这么量的，判据挂在被测对象上。
        """
        d = self._path / "lancedb" / f"{table}.lance" / "_versions"
        try:
            return len(os.listdir(d))
        except OSError:
            return -1

    def maybe_vacuum_lance_versions(self) -> list[str]:
        """任一表版本数 > `BLADEX_INDEX_MAX_VERSIONS` ⇒ 对**该表** optimize + 清旧版本。

        🔴 MQ-I15 的机制侧：此前 `compact_fact_vectors` 只管 facts（且只在碎片超阈值时
        触发），`compact_matter_vectors` 只按行冗余触发、且 drop+重建**不清版本**，
        files 表没有任何维护 ⇒ 三表的旧版本永存，磁盘单调上涨（771MB 里 ~600MB 是
        死版本；一次性清到 305MB 后 4 天 matters 又长回 827 版本）。

        清理窗口 `_COMPACT_CLEANUP_S`（60s）与节流 `_COMPACT_MIN_INTERVAL_S` 同 facts
        压缩：proxy 只读进程每次 search 前 `checkout_latest`，单次读持有版本以毫秒计。
        返回本次 vacuum 过的表名（空 = 没到阈值 / 节流中 / 只读实例）。
        """
        from datetime import timedelta

        if self._read_only or self._lancedb is None:
            return []
        max_versions = int(flag_number("BLADEX_INDEX_MAX_VERSIONS"))
        if max_versions <= 0:
            return []
        over = {t: n for t in self._LANCE_TABLES
                if (n := self.lance_version_count(t)) > max_versions}
        if not over:
            return []
        elapsed = time.monotonic() - self._last_version_vacuum_monotonic
        if elapsed < self._COMPACT_MIN_INTERVAL_S:
            logger.debug("index_version_vacuum_throttled", tables=over,
                         threshold=max_versions, elapsed_s=round(elapsed, 1))
            return []
        self._last_version_vacuum_monotonic = time.monotonic()
        done: list[str] = []
        for name, before in over.items():
            try:
                tbl = self._lancedb.open_table(name)
            except Exception as e:  # noqa: BLE001 —— 表还没建（如 files 从未写过）
                logger.debug("index_version_vacuum_open_failed", table=name, error=str(e))
                continue
            fn = getattr(tbl, "optimize", None)
            if fn is None:
                continue
            try:
                fn(cleanup_older_than=timedelta(seconds=self._COMPACT_CLEANUP_S))
            except Exception as e:  # noqa: BLE001
                logger.warning("index_version_vacuum_failed", table=name, error=str(e))
                continue
            done.append(name)
            logger.info("index_lance_versions_vacuumed", table=name,
                        versions_before=before, versions_after=self.lance_version_count(name),
                        threshold=max_versions)
        # 表句柄跟着刷新——vacuum 后旧句柄仍指向被清理的版本
        if "facts" in done:
            self._table = None
            self._lazy_open_facts_table()
        if "matters" in done:
            self._matter_table = None
            self._lazy_open_matter_table()
        if "files" in done:
            self._files_tbl = None
        return done

    def maybe_compact_fact_vectors(self) -> bool:
        """需要时压缩 facts 向量表（H6/G6 + MQ-R1）。

        两条触发轴，命中任一即压：

        1. **碎片数 > `BLADEX_INDEX_MAX_FRAGMENTS`**（MQ-R1，2026-08-17 新增，主轴）；
        2. 行数 > fact 数 × `_FACTS_COMPACT_RATIO`（旧轴，抓的是僵尸行/重复行）。

        🔴 主轴带一条**节流**（`_COMPACT_MIN_INTERVAL_S`），它不是"防抖"这种含糊
        理由，而是判据本身的性质决定的：碎片数数的是盘上的文件，而刚压出来的旧版本
        文件要过 `_COMPACT_CLEANUP_S` 才被下一次压缩清掉——窗口内判据仍然超阈值。
        没有节流，本方法会在这个窗口里反复空压（gate 抓到的首版形态：压完 6 -> 7，
        每个 pass 整表重写一次，**永远满足不了自己的退出条件**）。
        收敛过程是两步：第一次压缩合并出新文件、旧文件还在；第二次（≥120s 后）
        把上一代清掉，读数落到个位数、退出触发。

        🔴 为什么必须加第一条：MS-14（08-09）记过同一形态、手工压了一次、
        「自维护待开发窗」没做，于是 08-16 一次全量重建又把碎片一把堆回来——
        1619 条 fact = **1619 个 datafile**，dense 检索 6462ms，每轮撞 250ms 热路径
        deadline 降级只注硬规则，**注入面就这样空转了一个月**（2952 轮仅 10 轮有注入）。
        旧的第 2 条触发不了，是因为**它量的是另一根轴**：一行一碎片时
        `rows == fact_count`，比值恒为 1，永远够不着 ×2 门槛。
        判据必须挂在被测对象上——这正是"参照系脱钩"的一个实例。

        阈值标定（08-17 两点线性 + 一次外推验证，`scripts/time_inject_path.py`）：

            压缩后    2 碎片 / 1619 facts  ->  search   414ms
            当晚    182 碎片 / 1799 facts  ->  search  1067ms
            斜率 = (1067 − 414) / 180 = 3.63ms / 碎片；固定底 ≈ 410ms（MQ-R6 另查）
            外推校验：1619 × 3.63 ≈ 5.9s，当日实测 6.4s ✅

        由此，按 800ms 热路径预算：余量 = 800 − 410 − 22（embed + matter）= 368ms
        -> 约 **100 个碎片**；按实测 1.5 facts/轮 ≈ 60 轮。**阈值随预算变**——
        预算调回 250ms 时（MQ-R6 解决后）这个数要跟着重算。
        """
        self._lazy_open_facts_table()
        if self._read_only or self._table is None:
            return False

        # 轴 1（主）：碎片数
        max_fragments = int(flag_number("BLADEX_INDEX_MAX_FRAGMENTS"))
        fragments = self.fact_vector_fragments()
        if max_fragments > 0 and fragments > max_fragments:
            elapsed = time.monotonic() - self._last_fact_compact_monotonic
            if elapsed < self._COMPACT_MIN_INTERVAL_S:
                logger.debug("index_fact_compact_throttled",
                             fragments=fragments, threshold=max_fragments,
                             elapsed_s=round(elapsed, 1),
                             min_interval_s=self._COMPACT_MIN_INTERVAL_S)
                return False
            self._last_fact_compact_monotonic = time.monotonic()
            logger.info("index_fact_compact_triggered", axis="fragments",
                        fragments=fragments, threshold=max_fragments)
            self.compact_fact_vectors()
            return True

        # 轴 2（旧）：僵尸行 / 重复行。
        # 分母用 meta 里的 fact 总数：embedding 不落 meta（只在 LanceDB），
        # 而每条 fact 至多对应一行向量 —— 行数远超 fact 数就是碎片/重复。
        try:
            rows = self._table.count_rows()
        except Exception:  # noqa: BLE001
            return False
        n = self.fact_count()
        if n == 0 or rows <= n * self._FACTS_COMPACT_RATIO:
            return False
        logger.info("index_fact_compact_triggered", axis="rows", rows=rows, facts=n)
        self.compact_fact_vectors()
        return True

    def compact_fact_vectors(self, cleanup_older_than_s: float | None = None) -> tuple[int, int]:
        """facts 向量表整表重写成"每条 fact 一行"（M0-7）。

        与 matters 表同一套路：先对账去僵尸，再用 LanceDB 自带的
        `compact_files`/`optimize` 合并碎片。**不重建表**——facts 的向量不在 meta 里
        （meta 只存 Fact 元数据，embedding 字段不落盘），重建就等于丢向量。

        `cleanup_older_than_s`：版本清理窗口，缺省 `_COMPACT_CLEANUP_S`（60s）。
        与 `scripts/compact_index.py` 的差别在这里——那个脚本传 0（删光旧版本），
        所以它要求先停 proxy 与 consolidator；本方法跑在 consolidator 进程里、
        proxy 正只读同一张表，抽掉它脚下的版本就是把在线检索打断。
        60s 的余量见常量注释。测试传 0 以在一次调用内看到文件数下降。

        返回 (压缩前行数, 压缩后行数)。
        """
        from datetime import timedelta

        keep_s = (self._COMPACT_CLEANUP_S if cleanup_older_than_s is None
                  else cleanup_older_than_s)
        self._lazy_open_facts_table()
        if self._read_only or self._table is None:
            return 0, 0
        try:
            before = self._table.count_rows()
        except Exception as e:  # noqa: BLE001
            logger.warning("index_fact_compact_count_failed", error=str(e))
            return 0, 0
        # ① 先删僵尸（不然压缩只是把垃圾整理得更紧凑）
        self.reconcile_vector_meta()
        # ② 合并碎片 + 清理旧版本。三档依次退：带清理窗口的 optimize（想要的）→
        #    裸 optimize（老版本不认这个 kwarg；只合并不清理，判据靠节流兜住）→
        #    compact_files（更老的 API）。
        _attempts: list[tuple[str, dict[str, Any]]] = [
            ("optimize", {"cleanup_older_than": timedelta(seconds=keep_s)}),
            ("optimize", {}),
            ("compact_files", {}),
        ]
        for name, kwargs in _attempts:
            fn = getattr(self._table, name, None)
            if fn is None:
                continue
            try:
                fn(**kwargs)
                break
            except Exception as e:  # noqa: BLE001 —— 版本差异，能跑哪个算哪个
                logger.debug("index_fact_compact_call_failed", method=name,
                             kwargs=list(kwargs), error=str(e))
        try:
            self._table = self._lancedb.open_table("facts")
            after = self._table.count_rows()
        except Exception as e:  # noqa: BLE001
            logger.warning("index_fact_compact_reopen_failed", error=str(e))
            return before, before
        logger.info("index_fact_vectors_compacted", before=before, after=after)
        return before, after

    def search_matters(
        self, query_vec: list[float], k: int = 5,
        nprobes: int | None = None,
        user_id: str | None = None,
        visibility: list[str] | None = None,
    ) -> list[Matter]:
        """按质心向量检索 Matter（LanceDB top-k cosine）。

        `user_id` / `visibility`（ADR-0021 §2.4 补平，2026-08-16）：与 `search()` 同义。
        缺省不传 = 不过滤（向后兼容，个人模式下本就是恒真的 no-op）。
        补这两个参数之前，②平面是**唯一不设防的检索平面** —— Fact 侧建了门，
        Matter 侧连门框都没有。
        """
        # A1/T3: 惰性重试 + 追新
        self._lazy_open_matter_table()
        self._ensure_meta_db()
        self._try_catch_up()
        if self._lancedb is None or self._matter_table is None:
            return []

        search_builder = self._matter_table.search(query_vec)
        search_builder = search_builder.distance_type("cosine")
        search_builder = self._apply_ann_to_search(search_builder, nprobes, "matters")
        # 🔴 复核 §3.4：存量库里同一个 Matter 有多份陈旧行（写侧只追加不删，已在
        # `_add_matter_to_lancedb` 修好，但存量还在）。读侧多取一些再按 matter_id
        # 去重，否则 k=3 可能被同一个 Matter 的三份旧行占满 —— 召回覆盖面归零。
        results = search_builder.limit(max(k * _MATTER_DEDUP_EXPAND, k)).to_list()

        # ADR-0021 §2.4 补平：可见性过滤（此前本方法**签名里都没有这两个参数**）
        pids = self._visibility_pids(visibility, user_id)

        matters: list[Matter] = []
        seen: set[str] = set()
        dup_rows = 0
        scope_filtered = 0
        for row in results:
            matter_id = row.get("matter_id", "")
            if not matter_id or matter_id in seen:
                dup_rows += 1
                continue
            seen.add(matter_id)
            matter = self.get_matter(matter_id)
            if matter:
                if not self._matter_visible(matter, pids, visibility):
                    scope_filtered += 1
                    continue
                distance = row.get("_distance", 0.0)
                matter.embedding = [max(0.0, 1.0 - distance)]
                matters.append(matter)
            if len(matters) >= k:
                break

        logger.info("index_matter_search_done", results=len(matters),
                    dup_rows_skipped=dup_rows, scope_filtered=scope_filtered)
        return matters

    def embed_query_vec(self, query: str) -> list[float] | None:
        """把 query 编码成检索向量，供一轮内多路召回复用（热路径修复 2026-07-28）。

        返回 None = embedder 不可用，调用方应退回"各自 embed"的旧路径
        （由 search / search_matters_by_query 内部处理，行为不变）。
        """
        if self._embedder is None:
            return None
        try:
            return embed_query_compat(self._embedder, [query])[0]
        except Exception as e:  # noqa: BLE001 -- 编码失败不该炸热路径
            logger.warning("index_embed_query_failed", error=str(e))
            return None

    def search_matters_by_query(
        self, query: str, k: int = 5, q_vec: list[float] | None = None,
        user_id: str | None = None,
        visibility: list[str] | None = None,
    ) -> list[Matter]:
        """按文本 query 检索 Matter（内部 embed query → search_matters）。

        供 ProxyPrefetcher 两级召回用（ADR-0012 §4）。
        q_vec 非空则跳过 embedding（与 `search` 同一复用机制）。
        `user_id` / `visibility` 原样透传给 `search_matters`（ADR-0021 §2.4 补平）。
        """
        # A1/T3: 惰性重试 + 追新
        self._lazy_open_matter_table()
        self._ensure_meta_db()
        self._try_catch_up()
        self._refresh_lance_read()
        if self._embedder is None or self._lancedb is None or self._matter_table is None:
            return []
        try:
            if q_vec is None:
                q_vec = embed_query_compat(self._embedder, [query])[0]
            return self.search_matters(q_vec, k, user_id=user_id, visibility=visibility)
        except Exception as e:
            logger.warning("index_matter_search_by_query_failed", error=str(e))
            return []

    def get_facts_for_matter(
        self, matter_id: str, top_k: int = 5,
        user_id: str | None = None,
        visibility: list[str] | None = None,
        edges: list[MatterEdge] | None = None,
    ) -> list[Fact]:
        """获取某 Matter 域内的 top-k Fact（从归属边聚合）。

        供 ProxyPrefetcher 两级召回用（ADR-0012 §4）。

        ADR-0024 T8a（A-2）：原先 ``fact_ids[:top_k]`` 是边扫描顺序的无序截断，
        浪费了边上的 provenance/weight/confidence。现改为先排序后取：
          1. manual 边优先（用户明确归属 > 管线推断，ADR-0012 §3.4 manual 压 auto）
          2. 同 provenance 内按 weight 降序、再按 confidence 降序
        排序键确定，重建等价性保持。

        `user_id` / `visibility`（ADR-0021 §2.4 补平，2026-08-16）：**逐条按 fact 过滤**，
        复用 Fact 侧的 `_admissible`。仅按 Matter 可见还不够——一张可见的卡上仍可能
        挂着他人的 fact（归属侧 `search_matters` 长期无过滤，跨 user 归属是可能的）。
        🔴 **过滤在取 top_k 之前**：先滤后截，否则不可见的 fact 会占掉配额，
        表现为"卡明明有内容却召回不到"。

        `edges`：调用方已经取过这张卡的边时直接传进来，省一次全库扫描
        （MQ-R6：`get_edges` 每次 28ms，热路径上 `_hop_expand` 一轮要 6 次）。
        缺省 None = 自己取，行为与改造前逐字一致。
        """
        if edges is None:
            edges = self.get_edges(matter_id)
        fact_edges = [
            e for e in edges
            if e.target_type == EdgeTargetType.FACT
        ]
        # manual(=1) 排在 auto(=0) 前；weight/confidence 降序
        fact_edges.sort(
            key=lambda e: (
                1 if e.provenance == EdgeProvenance.MANUAL else 0,
                e.weight,
                e.confidence,
            ),
            reverse=True,
        )
        pids = self._visibility_pids(visibility, user_id)
        facts: list[Fact] = []
        dropped = 0
        for e in fact_edges:            # 先滤后截：不可见的不许占 top_k 配额
            if len(facts) >= top_k:
                break
            f = self.get_fact(e.target_key)
            if not f:
                continue
            # 🔴 只做**可见性**判定，不复用 `_admissible`。
            # `_admissible` 还会滤 `t_invalid`（被取代）与 `min_importance` ——
            # 那两条该不该在 ②平面生效是**另一个问题**（我倾向该，但需单独验证）。
            # 混进本步会打破"个人模式零回归"（个人模式下可见性恒真、是 no-op，
            # 但那两条不是），也会让 G7 重建的 delta 归因不清。单独立卡。
            if pids and (f.source_user_id or "") not in pids:
                dropped += 1
                continue
            facts.append(f)
        if dropped:
            logger.debug("index_matter_facts_scope_filtered",
                         matter_id=matter_id, dropped=dropped)
        return facts

    def add_edge(self, edge: MatterEdge) -> None:
        """存一条归属边。

        edge_id 确定性 = hash(matter_id/target_type/target_key)（T12d 重建等价性：
        full rebuild x2 -> 边集合逐条相等）。同 (matter,target) 重加幂等覆盖。
        """
        if not edge.edge_id:
            import hashlib
            raw = f"{edge.matter_id}/{edge.target_type.value}/{edge.target_key}"
            edge.edge_id = f"e-{hashlib.sha256(raw.encode()).hexdigest()[:12]}"
        meta = edge.model_dump(mode="json")
        key = f"{_EDGE_PREFIX}{edge.edge_id}"
        self.meta_db[key.encode()] = msgpack.packb(meta, use_bin_type=True)
        if edge.provenance == EdgeProvenance.MANUAL:
            self._invalidate_manual_edge_cache()
        logger.debug("index_edge_added", edge_id=edge.edge_id,
                      matter_id=edge.matter_id,
                      target_type=edge.target_type.value,
                      provenance=edge.provenance.value)

    def get_edges(self, matter_id: str) -> list[MatterEdge]:
        """读取某 Matter 的所有归属边。

        🔴 热路径成本（MQ-R6 二层段内计时，2026-08-17 实测）：**29.5ms / 次**，
        而 `_hop_expand` 一轮检索要调 6 次（3 个种子 Matter × 直接调 +
        `get_facts_for_matter` 内部再调）= **177ms**，正是 `_hop_expand` 的全部耗时、
        约 410ms 固定底的一半。

        本方法**必须**全扫 meta 是因为边的 key 是 `edge/{edge_id}`，
        matter_id 不在 key 里 —— 按 Matter 前缀扫不了。改 key 布局或加二级索引
        是真正的解，但那要动归属段的存储形态，另账（见台账 MQ-R6）。

        这里摘掉了一部分：**先看 msgpack 解出来的 `matter_id`，对不上就不
        `model_validate`**。⚠️ **实测只省 1.5ms（29.5 → 28.0），远小于预期**——
        我原以为"约 1900 条边逐条过 Pydantic"是大头，读数否掉了这个归因
        （当晚第四次同型失误：读数缩到函数级，我在函数里挑最像的当结论）。
        改动保留（语义逐字不变、白省一点），但**不要把它当成 `get_edges` 的解**。
        真正的开销在哪一层，由 `scripts/probe_meta_scan.py` 的读数说了算：
        它把全库 key 迭代 / 全库 (key,value) 迭代 / 逐条 get / 解码 / 校验分开称。
        当前最可疑的是 `items()` 会把**每一条 value 都拉起来**（含 1924 条 fact 的
        大 msgpack blob），但那仍是待测项，不是结论。
        """
        # A1/T3: 惰性重试 + 追新
        self._ensure_meta_db()
        self._try_catch_up()
        if self.meta_db is None:
            return []
        edges: list[MatterEdge] = []
        for key_raw, value_raw in self.meta_db.items():
            key_str = key_raw.decode() if isinstance(key_raw, bytes) else str(key_raw)
            if not key_str.startswith(_EDGE_PREFIX):
                continue
            data = msgpack.unpackb(value_raw, raw=False)
            # 先按 dict 里的字段筛，命中了才付 Pydantic 校验的钱。
            # 字段缺失（历史/脏数据）时退回校验后再判，语义与改造前逐字一致。
            if isinstance(data, dict) and "matter_id" in data:
                if data["matter_id"] != matter_id:
                    continue
                edges.append(MatterEdge.model_validate(data))
                continue
            edge = MatterEdge.model_validate(data)
            if edge.matter_id == matter_id:
                edges.append(edge)
        return edges

    def get_edges_for_matters(self, matter_ids: Iterable[str]) -> dict[str, list[MatterEdge]]:
        """一次扫描取回多张 Matter 的边（MQ-R6，2026-08-18）。

        为什么要有它——**探针读数，不是推断**（`scripts/probe_meta_scan.py`）：

            meta RocksDB 总 key 26608：judgment 12176 + distill 6772 +
            __meta__ 3071 + edge 1924 + fact 1924 + ...
            keys_only 全库迭代 22.8ms   ← 占 get_edges 那 25ms 的 **91%**
            取 value +2.2ms / 解码 1.6ms / Pydantic 校验 2.5ms —— 加起来不到 25%

        也就是说 `get_edges` 的成本**几乎全部是"走过 26608 个 key"这件事本身**，
        而其中 **71% 是 judgment/distill 派生台账**，与边毫无关系。
        减少解码、减少校验都没用（实测各省 1.5ms / 0.1ms，两次都试过了）；
        唯一有效的是**少扫几遍**。

        `_hop_expand` 原本一轮检索扫 6 遍（3 个种子 Matter × 直接调 +
        `get_facts_for_matter` 内部再调一次）= 168ms，正是它的全部耗时。
        本方法把这 6 遍合成 1 遍。

        更彻底的解是让边按 Matter 可前缀扫描（key 改成 `edge/{matter_id}/{edge_id}`
        或建二级索引），那要动归属段的存储布局，是 G2 的资源，另账。
        本仓当前也没有范围迭代的先例（全是 `keys()/items()/get()`），
        不在这一步引入未经验证的 API 用法。
        """
        wanted = set(matter_ids)
        out: dict[str, list[MatterEdge]] = {mid: [] for mid in wanted}
        if not wanted:
            return out
        self._ensure_meta_db()
        self._try_catch_up()
        if self.meta_db is None:
            return out
        for key_raw, value_raw in self.meta_db.items():
            key_str = key_raw.decode() if isinstance(key_raw, bytes) else str(key_raw)
            if not key_str.startswith(_EDGE_PREFIX):
                continue
            data = msgpack.unpackb(value_raw, raw=False)
            if isinstance(data, dict) and "matter_id" in data:
                mid = data["matter_id"]
                if mid not in wanted:
                    continue
                out[mid].append(MatterEdge.model_validate(data))
                continue
            edge = MatterEdge.model_validate(data)
            if edge.matter_id in wanted:
                out[edge.matter_id].append(edge)
        return out

    def get_matter_aggregate_view(self, matter_id: str) -> MatterAggregateView | None:
        """从归属边即时聚合 Matter 的聚合视图（不落独立存储）。"""
        matter = self.get_matter(matter_id)
        if matter is None:
            return None

        edges = self.get_edges(matter_id)
        session_keys: list[str] = []
        fact_ids: list[str] = []
        manual_count = 0

        for edge in edges:
            if edge.target_type == EdgeTargetType.SESSION:
                session_keys.append(edge.target_key)
            elif edge.target_type == EdgeTargetType.FACT:
                fact_ids.append(edge.target_key)
            if edge.provenance == EdgeProvenance.MANUAL:
                manual_count += 1

        return MatterAggregateView(
            matter_id=matter_id,
            title=matter.title,
            summary=matter.summary,
            status=matter.status,
            session_keys=session_keys,
            fact_ids=fact_ids,
            edge_count=len(edges),
            manual_edge_count=manual_count,
        )

    def update_matter_centroid(
        self,
        matter_id: str,
        new_vec: list[float],
        *,
        provenance: EdgeProvenance = EdgeProvenance.AUTO,
        manual_weight: float = _DEFAULT_MANUAL_CENTROID_WEIGHT,
        auto_weight: float = _DEFAULT_AUTO_CENTROID_WEIGHT,
    ) -> None:
        """滚动更新 Matter 质心（加权策略 + 边数对数衰减，ADR-0012 §3.5 + ADR-0018 §3.4）。

        manual 边权重 > auto 边；ADR-0018 新增边数对数衰减：新成员权重 ∝ 1/log(2+边数)，
        抑制大 Matter 质心趋向语料均值的正反馈黑洞。质心仅用于 e5 召回排序，不参与红线判定。
        权重比留实测、可配。
        """
        matter = self.get_matter(matter_id)
        if matter is None:
            return

        base = manual_weight if provenance == EdgeProvenance.MANUAL else auto_weight
        # ADR-0018 §3.4: 边数对数衰减（边越多，新成员对质心影响越小）。
        # 首成员（edge_count <= 1）不衰减--衰减只对已有成员的 Matter 生效。
        edge_count = sum(
            1 for e in self.get_edges(matter_id)
            if e.target_type == EdgeTargetType.FACT
        )
        if edge_count <= 1:
            decay = 1.0
        else:
            decay = 1.0 / math.log(2 + edge_count)
        w = base * decay

        if matter.centroid is None or len(matter.centroid) == 0:
            matter.centroid = list(new_vec)
            matter.centroid_weight = w
        elif matter.centroid_weight == 0.0:
            # 质心存在但无权重记录（外部创建）→ 视为 weight=1.0
            total = 1.0 + w
            matter.centroid = [
                (old * 1.0 + new * w) / total
                for old, new in zip(matter.centroid, new_vec, strict=True)
            ]
            matter.centroid_weight = total
        else:
            total = matter.centroid_weight + w
            matter.centroid = [
                (old * matter.centroid_weight + new * w) / total
                for old, new in zip(matter.centroid, new_vec, strict=True)
            ]
            matter.centroid_weight = total

        self.update_matter(matter)
        logger.debug("index_matter_centroid_updated", matter_id=matter_id,
                      provenance=provenance.value, weight=round(w, 4),
                      total_weight=round(matter.centroid_weight, 4),
                      edge_count=edge_count)

    def recompute_centroid(self, matter_id: str) -> None:
        """从所有归属边重算质心（T2 §3.5 manual 反哺）。

        手动指定/摘除后调用，重算质心 → 后续 auto 归属用新质心判。
        需要 LanceDB 获取 fact 向量（metadata 不存 embedding）。
        """
        matter = self.get_matter(matter_id)
        if matter is None:
            return

        edges = self.get_edges(matter_id)
        weighted_vecs: list[tuple[list[float], float]] = []

        for edge in edges:
            if edge.target_type != EdgeTargetType.FACT:
                continue
            vec = self._get_fact_vector(edge.target_key)
            if vec is None:
                continue
            w = (_DEFAULT_MANUAL_CENTROID_WEIGHT
                 if edge.provenance == EdgeProvenance.MANUAL
                 else _DEFAULT_AUTO_CENTROID_WEIGHT)
            weighted_vecs.append((vec, w))

        if not weighted_vecs:
            return

        dim = len(weighted_vecs[0][0])
        total_weight = sum(w for _, w in weighted_vecs)
        centroid = [0.0] * dim
        for vec, w in weighted_vecs:
            for i in range(dim):
                centroid[i] += vec[i] * w
        matter.centroid = [c / total_weight for c in centroid]
        matter.centroid_weight = total_weight
        self.update_matter(matter)
        logger.debug("index_matter_centroid_recomputed", matter_id=matter_id,
                      edges=len(weighted_vecs), total_weight=total_weight)

    def _get_fact_vectors(self, fact_ids: Iterable[str]) -> dict[str, list[float]]:
        """批量取 fact 向量：**一次** `to_arrow()`，不是每个 id 一次。

        🔴 MQ-R6（2026-08-17 段内计时定位）：`to_arrow()` 会把整张 facts 表
        （1799 行 × 1024 维向量）物化一遍。热路径上 `_fuse_rerank` 原本对
        **每个候选**调一次单条版本——40 个候选 = **40 次全表物化**，
        实测 `_fuse_rerank` p50 **207ms**，占 `search` 的 53%，
        是那个"约 410ms 固定底"的一半。
        它在段内计时里一直藏着：既不走 `to_list()`（LanceDB 查询本体只 3 次 / 9ms），
        也不走 `get_fact()`（159 次 / 3.8ms）——两个头号嫌疑都被读数排除之后，
        剩下的时间只能在这里。
        """
        ids = list(dict.fromkeys(fact_ids))
        if self._table is None or not ids:
            return {}
        try:
            import pyarrow as pa
            import pyarrow.compute as pc
            arrow_table = self._table.to_arrow()
            filtered = arrow_table.filter(pc.is_in(arrow_table["id"], value_set=pa.array(ids)))
            out: dict[str, list[float]] = {}
            for row in filtered.to_pylist():
                vec = row.get("vector")
                if vec is not None:
                    out[row["id"]] = vec
            return out
        except Exception as e:  # noqa: BLE001
            logger.debug("index_fact_vectors_lookup_failed", n_ids=len(ids), error=str(e))
            return {}

    def _get_fact_vector(self, fact_id: str) -> list[float] | None:
        """从 LanceDB 获取单条 fact 向量（metadata 不存 embedding）。

        委托批量版本——**同一件事只留一个实现**。单条调用的代价与改造前一致
        （仍是一次全表物化）；在循环里调它依然贵，热路径请直接用 `_get_fact_vectors`。
        """
        return self._get_fact_vectors([fact_id]).get(fact_id)

    def update_matter_summary(self, matter_id: str) -> None:
        """滚动更新 Matter 摘要（T2 ADR-0012 §3.1）。

        从归属 facts 的内容生成滚动摘要，随归入内容更新。
        取最近 N 条 fact 内容片段拼接。
        """
        matter = self.get_matter(matter_id)
        if matter is None:
            return

        edges = self.get_edges(matter_id)
        fact_ids = [e.target_key for e in edges if e.target_type == EdgeTargetType.FACT]

        if not fact_ids:
            return

        snippets: list[str] = []
        for fid in reversed(fact_ids):  # 最近的在前
            f = self.get_fact(fid)
            # 🔴 2026-08-05 实测修正：file_ref 成员不进摘要。
            # Matter 摘要要回答"这件事是什么"，而 file_ref 的 content 是
            # `文件 /Users/.../x.md` —— 真实库里 m-26bc3c841e08 的摘要因此变成了
            # 一串路径。E1.1 防的是 L4 rewrite 污染，保底拼接这条路同样会被污染。
            if f and f.content and f.item_kind != ItemKind.FILE_REF:
                snippets.append(f.content[:100])
            if len(snippets) >= _MAX_SUMMARY_FACTS:
                break

        if snippets:
            matter.summary = " | ".join(snippets)[:_MAX_SUMMARY_CHARS]
            # ADR-0018 §3.4: 保底拼接标记（L4 重写时覆盖为 llm_judge）
            matter.summary_source = "concat"
            self.update_matter(matter)
            logger.debug("index_matter_summary_updated", matter_id=matter_id,
                         summary_len=len(matter.summary))

    def digest_unassigned_pool(
        self,
        *,
        digestion_threshold: float = _DEFAULT_DIGESTION_THRESHOLD,
        min_cluster: int = _DEFAULT_DIGESTION_MIN_CLUSTER,
        max_pool_size: int = 200,
        goal_ctx: Any = None,
    ) -> int:
        """周期性消化未归属池（ADR-0018 §3.3 重写：按 proposal 标题规范化分组）。

        取代旧的 complete-linkage cosine 聚类（黑洞源，ADR-0018 §2）：
        按 proposal 标题规范化分组 -> 组内 >=2（不同 logical_turn）-> 诞生 provisional
        Matter（标题=提案标题）-> 立即转正（>=2 成员）。散乱/无 proposal 的留池。
        v1 简化：组内 LLM 确认留实测（proposal 标题已是 LLM 产物，作簇信号）。
        返回新创建的 Matter 数。
        """
        edges = self.get_edges(UNASSIGNED_MATTER_ID)
        fact_ids = [e.target_key for e in edges if e.target_type == EdgeTargetType.FACT]

        if len(fact_ids) < min_cluster:
            return 0

        # 加载 facts + embeddings（从 LanceDB 获取向量）
        embeddable: list[Fact] = []
        for fid in fact_ids:
            f = self.get_fact(fid)
            if f is None:
                continue
            vec = self._get_fact_vector(fid)
            if vec is not None:
                f.embedding = vec
            embeddable.append(f)

        if len(embeddable) < min_cluster:
            return 0

        # G5: 池大小上限
        if len(embeddable) > max_pool_size:
            logger.warning("index_unassigned_pool_truncated",
                           total=len(embeddable), processing=max_pool_size)
            embeddable = embeddable[:max_pool_size]

        # 按 proposal 标题规范化分组（无 proposal 的留池）
        groups: dict[str, list[Fact]] = defaultdict(list)
        title_of: dict[str, str] = {}
        for f in embeddable:
            title = next((t for t in f.proposal_titles if t.strip()), "")
            if not title:
                continue
            key = _normalize(title)
            groups[key].append(f)
            title_of[key] = title

        new_matters = 0
        clustered: set[str] = set()
        for key, group in groups.items():
            if len(group) < min_cluster:
                continue
            # 不同 logical_turn 才成簇（同一轮的多条 fact 不算独立证据）
            turns = {f.logical_turn for f in group if f.logical_turn is not None}
            if turns and len(turns) < min_cluster:
                continue

            title = title_of[key]
            # ── ADR-0031 §3.4：id 与标题解耦 ───────────────────────────────
            # 旧写法 `matter_id = _deterministic_matter_id(title)` 同时干了两件事：
            # ①生成 id ②顺带实现"同标题汇入同一张卡"（因为同标题算出同 id，
            # 于是 `get_matter(...) is not None` 就把它接上了）。
            # 解耦后 ②不能跟着 ①一起消失 —— 所以这里**显式查一次同名卡**。
            # 行为不变，只是不再依赖"标题即身份"这个巧合。
            existing = self._find_matter_by_native_key(key)
            seed_ledger_key = ""
            if existing is not None:
                matter_id = existing.matter_id
            else:
                # 新开：身份取种子 fact 的首轮 ledger key（组内按 logical_turn
                # 取最早那条，保证同一组重放两次拿到同一个 id）。
                seed = min(group, key=lambda f: (f.logical_turn if f.logical_turn
                                                 is not None else 1 << 30, f.id))
                seed_ledger_key = getattr(seed, "source_ledger_key", "") or ""
                matter_id = new_matter_id(ledger_key=seed_ledger_key, title=title)
            if self.get_matter(matter_id) is None:
                from bladex_core.topic_keys import derive_topic_keys as _dtk

                seed_vec = group[0].embedding
                matter = Matter(
                    matter_id=matter_id, title=title,
                    status=MatterStatus.PROVISIONAL, origin=MatterOrigin.AUTO,
                    # ADR-0021 §2.4 补平：继承创始 fact 的 scope（此前恒空）。
                    # group[0] 就是种子（centroid/aliases/topic_keys 都取它）。
                    scope=getattr(group[0], "scope", "") or "",
                    aliases=_dedupe_aliases(title, group[0].entities),
                    centroid=list(seed_vec) if seed_vec else None,
                    centroid_weight=1.0 if seed_vec else 0.0,
                    # 拍板 B：消化路新开同样种主题键（与 L5 新开同语义）
                    topic_keys=dict.fromkeys(_dtk(
                        getattr(group[0], "keywords", None),
                        getattr(group[0], "topic", "") or "",
                        group[0].proposal_titles, group[0].entities), 1),
                )
                # G12.3：消化路新开同样取 goal——首轮 = **种子 fact 那一轮**
                # （与 matter_id 的派生源同一条 key；用 group[0] 会让 id 与 goal
                # 指向两条不同的轮次）。`existing` 非空时 seed 未定义、key 为空 ⇒
                # 记 no_ledger_key（那条路本就不该进这个分支）。
                _reason = _capture_matter_goal(matter, seed_ledger_key, goal_ctx)
                self._goal_counters[_reason or "captured"] += 1
                self.add_matter(matter)

            # 转移归属边：从 __unassigned__ 摘除 -> 挂到新 Matter
            for cf in group:
                self.detach_edges_for_target(cf.id, EdgeTargetType.FACT)
                self.add_edge(MatterEdge(
                    matter_id=matter_id,
                    target_type=EdgeTargetType.FACT,
                    target_key=cf.id,
                    provenance=EdgeProvenance.AUTO,
                    decision={"layer": "digest", "verdict": f"link:{matter_id}"},
                ))
                clustered.add(cf.id)
                self._accumulate_matter_metadata(matter_id, cf)
                if cf.id != group[0].id and cf.embedding is not None:
                    self.update_matter_centroid(
                        matter_id, cf.embedding, provenance=EdgeProvenance.AUTO,
                    )

            self.update_matter_summary(matter_id)
            # >=2 成员 -> 立即转正
            self._maybe_promote_matter(matter_id)
            new_matters += 1
            logger.info("index_unassigned_digested", matter_id=matter_id, facts=len(group))

        if new_matters > 0:
            logger.info("index_unassigned_digestion_done", new_matters=new_matters,
                        total_facts=len(fact_ids), clustered=len(clustered))
        return new_matters

    # ── ADR-0018 §3.3/§3.4: provisional / 元数据累积 / 摘要重写 / 跨批一致性 ──

    def _accumulate_matter_metadata(self, matter_id: str, fact: Fact) -> None:
        """累积 Matter.entities（成员 entities 并集，封顶）——仅作展示/观察。

        MS-16（T4a，2026-08-10 审计）：**aliases 不再随成员 proposal_titles
        自动累积**。复现（scripts/audit_l2l3_gate_replay.py）证实这条累积
        通道是 L2 吸尘器的雪球源——一条错边混入后，它的提案标题成为
        aliases 里的合法匹配键，同题后续 fact 全部经 L2/L3 精确命中链入
        （README matter 145 边里 134 条 L2、aliases 累到 95 条）。
        aliases 语义回归「创建时原生键 + manual 追加」（rename / 手动
        enrich 不受影响，ADR-0012 手动映射主权保留）。
        entities 仍累积（保序去重 + 封顶保证重建确定），但已整体退出
        L2 门控与 L3 匹配面（bladex_core.attribution._native_match_keys）。
        """
        if matter_id == UNASSIGNED_MATTER_ID:
            return
        matter = self.get_matter(matter_id)
        if matter is None:
            return
        changed = False
        for e in fact.entities:
            if e and e not in matter.entities:
                matter.entities.append(e)
                changed = True
        if len(matter.entities) > _MAX_MATTER_ENTITIES:
            # FIFO 淘汰超限（留实测换频次淘汰）
            matter.entities = matter.entities[-_MAX_MATTER_ENTITIES:]
            changed = True
        # 拍板 A（2026-08-14）：成员 fact 主题键计票进 Matter.topic_keys。
        # 与 MS-16 停掉的 aliases 累积不同：① 有界（top-30 计票裁剪）；
        # ② 消费侧（L3 子层/合并检测）必须**多键重叠**才命中，单键进表
        # 不构成吸尘器。keywords 缺失（存量）走确定性派生兜底。
        from bladex_core.topic_keys import accumulate_topic_keys, derive_topic_keys

        fkeys = derive_topic_keys(
            getattr(fact, "keywords", None), getattr(fact, "topic", "") or "",
            fact.proposal_titles, fact.entities)
        if fkeys:
            new_tk = accumulate_topic_keys(
                getattr(matter, "topic_keys", None) or {}, fkeys)
            if new_tk != (getattr(matter, "topic_keys", None) or {}):
                matter.topic_keys = new_tk
                changed = True
        if changed:
            self.update_matter(matter)

    def _maybe_promote_matter(self, matter_id: str) -> bool:
        """provisional -> established(ACTIVE)：>=2 不同 logical_turn 成员 或 manual 触碰。

        ADR-0018 §3.3。manual 边与 established 状态是重判不动点。
        返回是否转正。
        """
        matter = self.get_matter(matter_id)
        if matter is None or matter.status != MatterStatus.PROVISIONAL:
            return False

        edges = self.get_edges(matter_id)
        # manual 触碰 -> 立即转正（用户认领即确认）
        if any(e.provenance == EdgeProvenance.MANUAL for e in edges):
            matter.status = MatterStatus.ACTIVE
            self.update_matter(matter)
            logger.info("index_matter_promoted matter_id=%s reason=manual", matter_id)
            return True

        member_facts = [
            self.get_fact(e.target_key) for e in edges
            if e.target_type == EdgeTargetType.FACT
        ]
        member_facts = [f for f in member_facts if f is not None]
        if len(member_facts) < 2:
            return False

        turns = {f.logical_turn for f in member_facts if f.logical_turn is not None}
        # >=2 不同 logical_turn；或无 logical_turn 信息但成员 >=2（fallback）
        if len(turns) >= 2 or len(turns) == 0:
            matter.status = MatterStatus.ACTIVE
            self.update_matter(matter)
            logger.info("index_matter_promoted matter_id=%s reason=members n=%d turns=%d",
                        matter_id, len(member_facts), len(turns))
            return True
        return False

    def _summary_rewrite_allowed(self, matter_id: str) -> bool:
        """ADR-0028 E1.1：L4 summary_rewrite 是否允许作用于该 Matter。

        确定性规则（非阈值）：只有 PROVISIONAL（胚芽态）Matter 允许被裁决重写摘要。
        established（ACTIVE/DORMANT/CLOSED）一律拒绝——它们的摘要由
        update_matter_summary() 从成员 fact 确定性聚合。
        Matter 不存在时按"新建 provisional"处理，放行（保 rewrite 对新 Matter 的现行为）。
        """
        if matter_id == UNASSIGNED_MATTER_ID:
            return False
        matter = self.get_matter(matter_id)
        if matter is None:
            return True
        return matter.status == MatterStatus.PROVISIONAL

    def _apply_summary_rewrite(self, matter_id: str, summary: str) -> None:
        """应用 L4 裁决顺带重写的 Matter 摘要（ADR-0018 §3.4）。"""
        if matter_id == UNASSIGNED_MATTER_ID or not summary:
            return
        matter = self.get_matter(matter_id)
        if matter is None:
            return
        matter.summary = summary[:_MAX_SUMMARY_CHARS]
        matter.summary_source = "llm_judge"
        self.update_matter(matter)

    def rejudge_pending(
        self,
        pipeline: AttributionPipeline,
        *,
        now: datetime | None = None,
    ) -> int:
        """T9(ADR-0018 §3.5)：周期重判未归属池中 pending_judgment 的 fact。

        对 unassigned 池里 pending 的 fact 重走 L3->L4（用最新 Matter 登记簿）。
        manual 边与 established 状态为不动点（不被移动）。恢复后的 LLM 补收早期判不了的。
        返回重新归属（移出未归属池）的 fact 数。
        """
        from datetime import UTC
        from datetime import datetime as _dt
        if now is None:
            now = _dt.now(UTC)

        edges = self.get_edges(UNASSIGNED_MATTER_ID)
        pending_facts: list[Fact] = []
        for e in edges:
            if e.target_type != EdgeTargetType.FACT:
                continue
            # 只重判标记 pending 的（uncertain / judge 曾不可用）
            if not (e.decision or {}).get("pending"):
                continue
            f = self.get_fact(e.target_key)
            if f is not None:
                vec = self._get_fact_vector(e.target_key)
                if vec is not None:
                    f.embedding = vec
                pending_facts.append(f)

        if not pending_facts:
            return 0

        # 每条 fact 召回候选（最新登记簿）
        candidates_by_fact: dict[str, list[Matter]] = {}
        for f in pending_facts:
            if f.embedding is not None and self._matter_table is not None:
                cands = self.search_matters(f.embedding, k=pipeline.top_candidates)
            else:
                cands = [m for m in self.all_matters()
                         if m.matter_id != UNASSIGNED_MATTER_ID]
            candidates_by_fact[f.id] = cands

        decisions = pipeline.attribute_batch(
            pending_facts, candidates_by_fact=candidates_by_fact, now=now,
        )

        reassigned = 0
        for f, dec in zip(pending_facts, decisions, strict=True):
            # 仍未解决（UNASSIGNED）-> 留池，不动
            if dec.source == AttributionSource.UNASSIGNED:
                continue
            # 解决了 -> 从未归属池摘除，挂到目标 Matter
            self.detach_edges_for_target(f.id, EdgeTargetType.FACT)
            self._apply_decision(f, dec, pipeline, now)
            reassigned += 1

        if reassigned > 0:
            logger.info("index_rejudge_reassigned count=%d pending=%d",
                        reassigned, len(pending_facts))
        return reassigned

    def _apply_decision(
        self,
        fact: Fact,
        decision: Any,
        pipeline: AttributionPipeline,
        now: datetime,
        goal_ctx: Any = None,
    ) -> None:
        """应用一条归属决策：创建 provisional / 加边 / 元数据 / 质心 / 摘要 / 转正。

        rebuild_from_hub 与 rejudge_pending 共用。
        MANUAL 决策的边已由管理事件重放创建，这里不重复加边。

        `goal_ctx`（G12.3）：`ledger_key -> (turn_class, 末条 user 原文)` 的懒取
        回调，只有 rebuild 传（rejudge_pending 拿不到那一轮 ⇒ goal 记
        `no_turn_context` 留空，**不猜**）。
        """
        matter_id = decision.matter_id
        if decision.is_new_matter and matter_id != UNASSIGNED_MATTER_ID:
            if self.get_matter(matter_id) is None:
                # ADR-0021 §2.4 补平：Matter.scope 继承创始 fact（此前恒空）
                matter = pipeline.create_matter_for_decision(
                    decision, centroid=fact.embedding,
                    scope=getattr(fact, "scope", "") or "")
                # G12.3 唯一写入点：新开卡这一刻取 goal（之后任何路径不得改写）
                _reason = _capture_matter_goal(
                    matter, getattr(fact, "source_ledger_key", "") or "", goal_ctx)
                self._goal_counters[_reason or "captured"] += 1
                self.add_matter(matter)

        # MANUAL：边已由 _replay_admin_events 创建，不重复
        if decision.source != AttributionSource.MANUAL:
            edge = pipeline.create_edge_for_decision(decision, fact)
            self.add_edge(edge)

        if matter_id == UNASSIGNED_MATTER_ID:
            return

        # U4/U6（ADR-0026 §4.2）：归属结果直接落 fact.matter_id 字段——
        # 消费者：U6 有限跳（同 Matter 兄弟）+ U7 注入②平面（当前 Matter 卡）+
        # 轴B 边界过滤。embedding 置 None = 只 upsert meta，不重复写向量。
        if fact.matter_id != matter_id:
            fact.matter_id = matter_id
            _emb = fact.embedding
            fact.embedding = None
            try:
                self.add_fact(fact)
            finally:
                fact.embedding = _emb

        self._accumulate_matter_metadata(matter_id, fact)

        # L4 摘要重写优先；否则保底拼接
        # ADR-0028 E1.1（F1 止血）：summary_rewrite 只对 PROVISIONAL Matter 生效。
        # established（active/dormant/closed）Matter 的 summary 唯一更新路径 =
        # update_matter_summary()——防"探针句一轮改写掉整卡摘要"。
        if decision.summary_rewrite and decision.source == AttributionSource.LLM_LINK:
            if self._summary_rewrite_allowed(matter_id):
                self._apply_summary_rewrite(matter_id, decision.summary_rewrite)
            else:
                logger.warning(
                    "matter_summary_rewrite_rejected matter_id=%s fact_id=%s reason=established",
                    matter_id, fact.id,
                )
                self.update_matter_summary(matter_id)
        else:
            self.update_matter_summary(matter_id)

        # 质心滚动更新（边数对数衰减）
        if fact.embedding is not None:
            self.update_matter_centroid(
                matter_id, fact.embedding, provenance=EdgeProvenance.AUTO,
            )

        # 唤醒 dormant / 转正 provisional
        matter = self.get_matter(matter_id)
        if matter is not None:
            pipeline.reactivate_on_hit(matter)
            self._maybe_promote_matter(matter_id)

    # ── ADR-0012 §3.5: 手动映射操作（manual 压过 auto）──

    def get_manual_edge_for_target(
        self, target_key: str, target_type: EdgeTargetType,
    ) -> MatterEdge | None:
        """查找某 target 是否已有 manual 归属边（供自动管线跳过用）。

        G4: 用内存缓存避免每条 fact 全扫 edges。缓存惰性构建，写操作失效。
        """
        if self.meta_db is None:
            return None
        if self._manual_edge_cache is None:
            self._build_manual_edge_cache()
        return self._manual_edge_cache.get(f"{target_type.value}:{target_key}")

    def _build_manual_edge_cache(self) -> None:
        """扫描全量 edges 构建 manual edge 反向索引（G4）。"""
        cache: dict[str, MatterEdge] = {}
        for key_raw, value_raw in self.meta_db.items():
            key_str = key_raw.decode() if isinstance(key_raw, bytes) else str(key_raw)
            if not key_str.startswith(_EDGE_PREFIX):
                continue
            data = msgpack.unpackb(value_raw, raw=False)
            edge = MatterEdge.model_validate(data)
            if edge.provenance == EdgeProvenance.MANUAL:
                cache[f"{edge.target_type.value}:{edge.target_key}"] = edge
        self._manual_edge_cache = cache

    def _invalidate_manual_edge_cache(self) -> None:
        """写操作后失效缓存（G4）。"""
        self._manual_edge_cache = None

    def assign_manual(
        self,
        matter_id: str,
        target_type: EdgeTargetType,
        target_key: str,
        weight: float = 2.0,
    ) -> MatterEdge:
        """手动指定归属（manual 压过 auto，ADR-0012 §3.5）。

        创建 provenance=MANUAL 的边，自动管线不得改动/移除。
        manual 边的 weight 高于 auto（作为质心强锚点）。
        """
        edge = MatterEdge(
            matter_id=matter_id,
            target_type=target_type,
            target_key=target_key,
            weight=weight,
            provenance=EdgeProvenance.MANUAL,
        )
        self.add_edge(edge)
        # T2: manual 反哺 — 手动指定后重算质心，后续 auto 归属用新质心判
        if target_type == EdgeTargetType.FACT:
            self.recompute_centroid(matter_id)
            self.update_matter_summary(matter_id)
        logger.info("index_manual_assign", matter_id=matter_id,
                     target_key=target_key, target_type=target_type.value)
        return edge

    def detach_edge(self, edge_id: str) -> bool:
        """手动摘除归属边。"""
        if self.meta_db is None:
            return False
        key = f"{_EDGE_PREFIX}{edge_id}"
        raw = self.meta_db.get(key.encode())
        if raw is None:
            return False
        # T2: 捕获 matter_id 以便摘除后重算质心
        edge = MatterEdge.model_validate(msgpack.unpackb(raw, raw=False))
        matter_id = edge.matter_id
        del self.meta_db[key.encode()]
        self._invalidate_manual_edge_cache()
        # T2: manual 反哺 — 摘除后重算质心
        self.recompute_centroid(matter_id)
        logger.info("index_edge_detached", edge_id=edge_id)
        return True

    def split_matter(
        self,
        source_id: str,
        new_matter_id: str,
        new_title: str,
        edge_ids: list[str],
    ) -> tuple[Matter, int]:
        """拆分 Matter：创建新 Matter 并把指定归属边转移过去（ADR-0012 T4）。

        与 merge 的对称操作——merge 是合、split 是拆。
        返回 (新 Matter, 转移边数)。source Matter 不删除。

        `scope`（ADR-0021 §2.4 补平）：**从源 Matter 继承**。拆分不改变可见范围——
        把一张卡拆成两张，两半仍属同一个可见集合；若拆出来的落成空 scope，
        它在读侧过滤下会变成"谁都看不见"或"谁都看得见"（取决于回落语义），两者都错。
        """
        source = self.get_matter(source_id)
        new_matter = Matter(
            matter_id=new_matter_id,
            title=new_title,
            scope=getattr(source, "scope", "") or "" if source is not None else "",
            origin=MatterOrigin.MANUAL,
        )
        self.add_matter(new_matter)

        moved = 0
        for eid in edge_ids:
            key = f"{_EDGE_PREFIX}{eid}"
            raw = self.meta_db.get(key.encode()) if self.meta_db is not None else None
            if raw is None:
                logger.warning("index_split_edge_not_found", edge_id=eid, source_id=source_id)
                continue
            edge = MatterEdge.model_validate(msgpack.unpackb(raw, raw=False))
            edge.matter_id = new_matter_id
            self.meta_db[key.encode()] = msgpack.packb(
                edge.model_dump(mode="json"), use_bin_type=True,
            )
            moved += 1

        logger.info("index_matter_split", source_id=source_id,
                     new_matter_id=new_matter_id, moved_edges=moved)
        return new_matter, moved

    def detach_edges_for_target(
        self,
        target_key: str,
        target_type: EdgeTargetType,
    ) -> int:
        """摘除指向某 target 的所有归属边（detach by target）。

        返回摘除的边数（通常为 0 或 1）。
        """
        if self.meta_db is None:
            return 0
        removed = 0
        keys_to_delete = []
        for key_raw, value_raw in self.meta_db.items():
            key_str = key_raw.decode() if isinstance(key_raw, bytes) else str(key_raw)
            if not key_str.startswith(_EDGE_PREFIX):
                continue
            data = msgpack.unpackb(value_raw, raw=False)
            edge = MatterEdge.model_validate(data)
            if edge.target_key == target_key and edge.target_type == target_type:
                keys_to_delete.append(key_raw)
                removed += 1
        for k in keys_to_delete:
            del self.meta_db[k]
        if removed > 0:
            logger.info("index_edges_detached_for_target",
                         target_key=target_key, count=removed)
        return removed

    def merge_matters(self, source_id: str, target_id: str) -> int:
        """合并 Matter：source 全部并入 target（manual 主权操作，管理事件驱动）。

        2026-08-14 完备性升级（matter-dedup-keywords 卡拍板 C）——此前只搬边 +
        平均质心，四个消费面漏掉：
          ① fact.matter_id 改写（U6 有限跳 / U7 ②平面 / 轴B 边界都读它——
             不改写 = 合并后成员散点照旧按旧 Matter 过滤，合并名存实亡）；
          ② source 标题/aliases 并入 target aliases（L3 原生键——下次 LLM 又用
             source 的措辞出 proposal 时能全串命中 target，不再开新的）；
          ③ topic_keys / entities 计票合并（L3 主题键子层的对照面）；
          ④ source 关闭 + 双方 lifecycle 记录（closed 退出 L2-L4 候选与钉卡）。
        返回转移的边数。重放语义：source 不存在（重建时 L3 子层已拦住没开出来）
        = 幂等 no-op。
        """
        if self.meta_db is None:
            return 0
        if (not source_id or not target_id or source_id == target_id
                or UNASSIGNED_MATTER_ID in (source_id, target_id)):
            logger.warning("index_matters_merge_rejected source=%s target=%s",
                           source_id, target_id)
            return 0
        source = self.get_matter(source_id)
        target = self.get_matter(target_id)
        if target is None or source is None:
            # 重放幂等：重建时 source 可能根本没被开出来（主题键子层拦截成功）
            logger.info("index_matters_merge_noop", source_id=source_id,
                        target_id=target_id,
                        reason="source_missing" if source is None else "target_missing")
            return 0
        if source.status == MatterStatus.CLOSED:
            # 幂等护栏：已合并过（closed）再应用一次 = no-op——同一事件可能被
            # proxy 端直接应用 + 重放各执行一次，计票类字段不许重复累加。
            logger.info("index_matters_merge_noop", source_id=source_id,
                        target_id=target_id, reason="source_already_closed")
            return 0

        edges = self.get_edges(source_id)
        moved = 0
        for edge in edges:
            edge.matter_id = target_id
            # 重新存储（新 key 或覆盖）
            meta = edge.model_dump(mode="json")
            key = f"{_EDGE_PREFIX}{edge.edge_id}"
            self.meta_db[key.encode()] = msgpack.packb(meta, use_bin_type=True)
            moved += 1
            # ① fact.matter_id 改写（meta upsert，不动向量）
            if edge.target_type == EdgeTargetType.FACT:
                f = self.get_fact(edge.target_key)
                if f is not None and f.matter_id == source_id:
                    f.matter_id = target_id
                    _emb = f.embedding
                    f.embedding = None
                    try:
                        self.add_fact(f)
                    finally:
                        f.embedding = _emb

        # ② source 措辞进 target 原生键（cap 40，保序去重）
        for a in [source.title, *source.aliases]:
            if a and a not in target.aliases and len(target.aliases) < 40:
                target.aliases.append(a)
        # ③ topic_keys 计票合并 + entities 并集（沿用各自上限）
        from bladex_core.topic_keys import accumulate_topic_keys

        src_tk = getattr(source, "topic_keys", None) or {}
        if src_tk:
            merged_tk = dict(getattr(target, "topic_keys", None) or {})
            for k, n in src_tk.items():
                merged_tk[k] = merged_tk.get(k, 0) + int(n)
            target.topic_keys = accumulate_topic_keys(merged_tk, [])
        for e in source.entities:
            if e and e not in target.entities:
                target.entities.append(e)
        if len(target.entities) > _MAX_MATTER_ENTITIES:
            target.entities = target.entities[-_MAX_MATTER_ENTITIES:]

        # 合并质心（简单均值）
        if source.centroid and target.centroid:
            target.centroid = [(a + b) / 2.0 for a, b in zip(
                source.centroid, target.centroid, strict=True)]

        # G12.3 判据 D3：**goal 不随合并搬家**。goal 是"这件事首轮定的身份"，
        # target 的首轮才是这件事的首轮；把 source 的 goal 搬过来 = 模型侧的动作
        # 改了用户的目标（D2 反面）。但不许**静默**丢——留一条可 grep 的告警，
        # 让"合并把一个目标陈述吃掉了"这件事能被发现（同 MQ-RT3 教训：
        # 机制正确而不可见，下次仍然靠人撞见）。
        if getattr(source, "task_goal", "") and not getattr(target, "task_goal", ""):
            logger.warning("matter_merge_goal_dropped", source_id=source_id,
                           target_id=target_id,
                           source_goal_len=len(source.task_goal))
        # ④ lifecycle + source 关闭（closed 退出 L2-L4 候选与②平面钉卡）
        target.record_lifecycle("merged", f"absorbed {source_id} ({source.title[:40]})")
        self.update_matter(target)
        source.status = MatterStatus.CLOSED
        source.record_lifecycle("merged_into", target_id)
        self.update_matter(source)

        logger.info("index_matters_merged", source_id=source_id,
                     target_id=target_id, moved_edges=moved)
        return moved

    def matter_merge_candidates(
        self, *, max_pairs: int = 50,
    ) -> list[dict[str, Any]]:
        """合并提案检测（拍板 C）：主题键高重叠的 Matter 对（**只检测不合并**）。

        误合并率=0 是第一红线——本方法只产出提案，合并永远走
        `/admin/matters/{id}/merge`（manual 主权 + 管理事件入账）。
        建议方向：成员少的并入成员多的（target = 边多者）。
        """
        from bladex_core.topic_keys import (
            effective_topic_keys,
            is_strong_topic_match,
            topic_key_overlap,
        )

        candidates_all = [
            m for m in self.all_matters()
            if m.matter_id != UNASSIGNED_MATTER_ID
            and m.status != MatterStatus.CLOSED
        ]
        # 存量 Matter（旧代码建的）topic_keys 空 → title+aliases+entities 派生兜底
        # （只读时算不落库；2026-08-14 上线即踩：泰安四开对自身检不出来）
        eff_keys = {
            m.matter_id: effective_topic_keys(
                getattr(m, "topic_keys", None), m.title, m.aliases, m.entities)
            for m in candidates_all
        }
        linkable = [m for m in candidates_all if eff_keys[m.matter_id]]
        edge_counts = {m.matter_id: len(self.get_edges(m.matter_id)) for m in linkable}
        pairs: list[dict[str, Any]] = []
        for i, a in enumerate(linkable):
            for b in linkable[i + 1:]:
                ka = set(eff_keys[a.matter_id])
                kb = eff_keys[b.matter_id]
                if not is_strong_topic_match(ka, kb):
                    continue
                inter, jac = topic_key_overlap(ka, kb)
                small, big = sorted((a, b), key=lambda m: (edge_counts[m.matter_id], m.matter_id))
                pairs.append({
                    "source_matter_id": small.matter_id, "source_title": small.title,
                    "target_matter_id": big.matter_id, "target_title": big.title,
                    "overlap": inter, "jaccard": round(jac, 3),
                    "shared_keys": sorted(ka & set(kb))[:8],
                    "source_edges": edge_counts[small.matter_id],
                    "target_edges": edge_counts[big.matter_id],
                })
        pairs.sort(key=lambda p: (-p["overlap"], -p["jaccard"],
                                  p["source_matter_id"]))
        return pairs[:max_pairs]

    def build_matter_key_index(self) -> tuple[dict[str, set[str]], int]:
        """主题键 → 拥有该键的 Matter 集合（GM-2 按键反查候选通路的地基）。

        返回 `(倒排表, 可链接 Matter 总数)`。第二个值是**判别力门槛的分母**——
        门槛必须随库规模走，不能是一个绝对数（同一个"出现在 5 张卡上"在 20 张卡的库
        里是泛词、在 5000 张卡的库里是强标识符）。

        存量 Matter（topic_keys 空）走 `effective_topic_keys` 的派生兜底，与
        `matter_merge_candidates` 同一口径——两处对"这张卡有哪些键"必须一致，
        否则离线检出的重复卡在线检不出来（2026-08-14 上线即踩过：泰安四开对自身检不出）。
        """
        from bladex_core.topic_keys import effective_topic_keys

        inverted: dict[str, set[str]] = defaultdict(set)
        total = 0
        for m in self.all_matters():
            if m.matter_id == UNASSIGNED_MATTER_ID or m.status == MatterStatus.CLOSED:
                continue
            total += 1
            for k in effective_topic_keys(
                getattr(m, "topic_keys", None), m.title, m.aliases, m.entities,
            ):
                inverted[k].add(m.matter_id)
        return inverted, total

    def key_based_matter_candidates(
        self,
        fact_keys: list[str] | set[str],
        inverted: dict[str, set[str]],
        total_matters: int,
        *,
        max_candidates: int = 3,
    ) -> list[tuple[str, int, list[str]]]:
        """按主题键反查候选 Matter（GM-2）。返回 `[(matter_id, 共享键数, 共享键)]`。

        🔴 **两道硬门，缺一不可**——它们是把 MS-16「单泛词吸尘器」挡在候选侧的全部依仗：

        1. **判别力门**：键出现在超过 `ratio × 库内可链接卡数` 张卡上 → 丢弃。
           阴性对照实测：泛词 `bladex` 一个键就命中 **57 张卡**，不设门 = 把吸尘器
           从合并侧原样搬到候选侧。MQ-D11 的分布支持这道门代价很小：903 个键里
           732 个只出现在 1 张卡上。
        2. **单键永不放行**：至少 2 个判别性键共享才算候选。与 `topic_keys` 模块
           既定纪律一致（`is_strong_topic_match` 同源），也是 T4a 那次误合并的修法。

        本方法**只产候选**，判定仍由 L1–L5 走完（GM 纪律 1）。
        """
        from bladex_core.topic_keys import clean_topic_keys

        ratio = flag_number("BLADEX_ATTRIB_KEY_DF_RATIO")
        # 分母随库走 + 下限 2：库很小时 ratio 会算出 0/1，把"恰好两张卡共享"
        # 这条本路径唯一想抓的形态也一并挡掉。
        max_df = max(2, int(total_matters * ratio))

        shared: dict[str, list[str]] = defaultdict(list)
        for k in clean_topic_keys(list(fact_keys), max_keys=30):
            owners = inverted.get(k)
            if not owners or len(owners) > max_df:
                continue          # 判别力门：泛词直接丢
            for mid in owners:
                shared[mid].append(k)

        hits = [
            (mid, len(keys), sorted(keys))
            for mid, keys in shared.items()
            if len(keys) >= 2        # 🔴 单键永不放行
        ]
        hits.sort(key=lambda t: (-t[1], t[0]))
        return hits[:max_candidates]

    def close_matter(self, matter_id: str) -> bool:
        """关闭 Matter（status → closed）。"""
        matter = self.get_matter(matter_id)
        if matter is None:
            return False
        matter.status = MatterStatus.CLOSED
        self.update_matter(matter)
        logger.info("index_matter_closed", matter_id=matter_id)
        return True

    def rename_matter(self, matter_id: str, title: str) -> bool:
        """重命名 Matter。"""
        matter = self.get_matter(matter_id)
        if matter is None:
            return False
        matter.title = title
        self.update_matter(matter)
        logger.info("index_matter_renamed", matter_id=matter_id, title_len=len(title))
        return True

    def remove_edges_for_turn(self, turn_key: str) -> int:
        """删除与某 turn 关联的所有归属边（级联清除用，T6 调用）。"""
        if self.meta_db is None:
            return 0
        removed = 0
        keys_to_delete = []
        for key_raw, value_raw in self.meta_db.items():
            key_str = key_raw.decode() if isinstance(key_raw, bytes) else str(key_raw)
            if not key_str.startswith(_EDGE_PREFIX):
                continue
            data = msgpack.unpackb(value_raw, raw=False)
            edge = MatterEdge.model_validate(data)
            if turn_key in edge.target_key:
                keys_to_delete.append(key_raw)
                removed += 1
        for k in keys_to_delete:
            del self.meta_db[k]
        if removed > 0:
            logger.info("index_edges_removed_for_turn", turn_key=turn_key, count=removed)
        return removed

    # ── ADR-0012 §3.6: 级联清除 ──

    def cascade_delete_turn(self, turn_key: str) -> int:
        """删除 tombstoned turn 在 Memory Index 的派生物（ADR-0012 §3.6 级联清除）。

        移除：该 turn 提炼出的 Fact（metadata + LanceDB 向量）+ 关联归属边。
        返回删除的条目数。
        """
        removed = 0

        # 找到 source_ledger_key 匹配的 facts
        facts_to_delete: list[str] = []
        for fact in self.all_facts():
            if fact.source_ledger_key == turn_key:
                facts_to_delete.append(fact.id)

        # 从 metadata 删除
        for fid in facts_to_delete:
            key = f"{_FACT_PREFIX}{fid}"
            if self.meta_db.get(key.encode()) is not None:
                del self.meta_db[key.encode()]
                removed += 1

        # 从 LanceDB 删除向量
        if self._table is not None and facts_to_delete:
            for fid in facts_to_delete:
                try:
                    safe_id = fid.replace("'", "''")
                    self._table.delete(f"id = '{safe_id}'")
                except Exception as e:
                    logger.debug("index_lancedb_delete_skip", fact_id=fid, error=str(e))

        # 删除关联归属边（session 边 + 已删 fact 的边）
        removed += self.remove_edges_for_turn(turn_key)
        for fid in facts_to_delete:
            removed += self._remove_edges_for_target_key(fid)

        logger.info("index_cascade_delete_done", turn_key=turn_key,
                     facts_deleted=len(facts_to_delete), total_removed=removed)
        return removed

    def delete_fact(self, fact_id: str) -> bool:
        """删除单条 Fact（meta + LanceDB 向量 + 指向它的归属边）。

        Beta T11 `bladex memory forget` 的 Memory Index 级联（ADR-0012 §3.6：调用方须先写
        Memory Hub 墓碑——fact.id 是确定性 hash（ledger_key+content），rebuild 侧按 FACT 墓碑
        过滤保证不复活）。返回 fact 是否存在过。
        """
        if self.meta_db is None:
            return False
        key = f"{_FACT_PREFIX}{fact_id}"
        # 先读出来：file_ref 的级联要用到它的 subject（删掉之后就读不到了）
        fact = self.get_fact(fact_id)
        existed = self.meta_db.get(key.encode()) is not None
        if existed:
            del self.meta_db[key.encode()]
        if self._table is not None:
            try:
                safe_id = fact_id.replace("'", "''")
                self._table.delete(f"id = '{safe_id}'")
            except Exception as e:  # noqa: BLE001
                logger.debug("index_lancedb_delete_skip", fact_id=fact_id, error=str(e))
        fts = self._ensure_fts()
        if fts is not None:
            fts.delete(fact_id)
        # 拍板项：forget 一条 file_ref → 同路径的文件索引与本地副本一并删
        #（"删除即删除"，不留影子副本；ADR-0012 墓碑语义）
        if existed and fact is not None and fact.item_kind == ItemKind.FILE_REF:
            from bladex_core.file_index import file_id_for
            path = (getattr(fact, "subject", "") or "").strip()
            if path:
                self.delete_file_index(file_id_for(path))
        edges_removed = self._remove_edges_for_target_key(fact_id)
        logger.info("index_fact_deleted", fact_id=fact_id, existed=existed,
                    edges_removed=edges_removed)
        return existed

    def _remove_edges_for_target_key(self, target_key: str) -> int:
        """删除 target_key 精确匹配的归属边（按 fact_id 删边）。"""
        if self.meta_db is None:
            return 0
        removed = 0
        keys_to_delete = []
        for key_raw, value_raw in self.meta_db.items():
            key_str = key_raw.decode() if isinstance(key_raw, bytes) else str(key_raw)
            if not key_str.startswith(_EDGE_PREFIX):
                continue
            data = msgpack.unpackb(value_raw, raw=False)
            edge = MatterEdge.model_validate(data)
            if edge.target_key == target_key:
                keys_to_delete.append(key_raw)
                removed += 1
        for k in keys_to_delete:
            del self.meta_db[k]
        return removed

    # ── 消费游标（consumed-key 集合，替代字典序水位游标）──

    def _mark_consumed(self, ledger_key: str) -> None:
        """标记一条 Memory Hub key 已被消费（增量重建去重）。

        替代旧的字典序单游标水位——Memory Hub key 形如 user/agent/session/entry_id，
        字典序 ≠ 插入时间序，单游标会被高字典序脏 key 顶到最大值后永久跳过
        真实数据（review 20260707）。consumed-key 集合每条 key 独立标记，互不影响。
        """
        self.meta_db[f"{_CONSUMED_PREFIX}{ledger_key}".encode()] = b"1"

    def is_consumed(self, ledger_key: str) -> bool:
        """检查一条 Memory Hub key 是否已被消费。"""
        if self.meta_db is None:
            return False
        return self.meta_db.get(f"{_CONSUMED_PREFIX}{ledger_key}".encode()) is not None

    # ── MS-1（复核 D4）：蒸馏失败轮的重试计数 ─────────────────────────────────

    def _distill_retry_count(self, ledger_key: str) -> int:
        if self.meta_db is None:
            return 0
        raw = self.meta_db.get(f"{_DISTILL_RETRY_PREFIX}{ledger_key}".encode())
        if raw is None:
            return 0
        try:
            return int(raw.decode() if isinstance(raw, bytes) else str(raw))
        except (ValueError, AttributeError):
            return 0

    def _bump_distill_retry(self, ledger_key: str) -> int:
        """失败重试计数 +1 并返回新值（持久化——进程重启不清零）。"""
        if self.meta_db is None:
            return 0
        n = self._distill_retry_count(ledger_key) + 1
        self.meta_db[f"{_DISTILL_RETRY_PREFIX}{ledger_key}".encode()] = str(n).encode()
        return n

    def _clear_distill_retry(self, ledger_key: str) -> None:
        """成功消费后清掉计数（下次再失败重新从 1 起算）。"""
        if self.meta_db is None:
            return
        k = f"{_DISTILL_RETRY_PREFIX}{ledger_key}".encode()
        if self.meta_db.get(k) is not None:
            del self.meta_db[k]

    def consumed_count(self) -> int:
        """已消费的 Memory Hub key 数量（调试/测试 + `/admin/status` 的 lag 减数）。

        MQ-R3（2026-08-17）：同 `fact_count`，secondary 视图必须先追新，否则
        "消费停摆"是假象——2026-08-17 为此误判两次、查了三小时。
        """
        # MQ-R3: 惰性重试 + 追新（同 get_fact:2816）
        self._ensure_meta_db()
        self._try_catch_up()
        if self.meta_db is None:
            return 0
        return sum(
            1 for k in self.meta_db.keys()
            if (k.decode() if isinstance(k, bytes) else str(k)).startswith(_CONSUMED_PREFIX)
        )

    @property
    def last_rebuild_backlog(self) -> bool:
        """上轮 rebuild 是否被 max_turns 截断（True = Memory Hub 里还有未消费的轮次）。

        ③（2026-07-28）：consolidator 据此决定"立即续跑"还是 sleep(interval)。
        积压时空转 60s/批 = 6793 轮要多睡 3.8 小时。
        """
        return self._last_rebuild_backlog

    def unconsume_matching(
        self,
        ledger: MemoryHub,
        *,
        since: str = "",
        until: str = "",
        agents: set[str] | None = None,
        exclude_agents: set[str] | None = None,
    ) -> int:
        """删除匹配 turn 的 consumed 标记，让增量重建重新拾起（定向重建，2026-07-28）。

        与 full rebuild（清库全量重放）不同：不动现有 Memory Index 数据。之后的增量
        rebuild 会按 max_turns 分批重放这些 turn——novelty（实体感知）判重
        跳过已提取过的内容、只补回缺失的 Fact；distill/judgment 台账命中的
        轮次不重新调 LLM。适合"修了提取缺陷后补历史窗口"的场景。

        必须至少给一个过滤条件（防误删全部标记 = 意外全量重放）。
        返回删除的标记数。
        """
        if not (since or until or agents or exclude_agents):
            raise ValueError("unconsume_matching 至少需要一个过滤条件"
                             "（since/until/agents/exclude_agents）")
        if self.meta_db is None:
            return 0
        removed = 0
        # ① 惰性扫描：只需 key + ts，不必付完整反序列化
        for key_str, ts_iso in ledger.scan_meta(""):
            if not _turn_matches_filter(
                key_str, ts_iso,
                since=since, until=until,
                agents=agents, exclude_agents=exclude_agents,
            ):
                continue
            ck = f"{_CONSUMED_PREFIX}{key_str}".encode()
            if self.meta_db.get(ck) is not None:
                del self.meta_db[ck]
                removed += 1
        logger.info("index_unconsumed_for_replay", removed=removed,
                    since=since or None, until=until or None,
                    agents=sorted(agents) if agents else None,
                    exclude_agents=sorted(exclude_agents) if exclude_agents else None)
        return removed

    def _cleanup_legacy_watermark(self) -> None:
        """一次性清理遗留的字典序水位游标（迁移用）。

        旧水位方案已废弃，首次增量重建时删除遗留 key，
        避免 poisoned 水位残留（review 20260707）。
        """
        if self.meta_db is None:
            return
        if self.meta_db.get(_INDEX_WATERMARK_KEY.encode()) is not None:
            del self.meta_db[_INDEX_WATERMARK_KEY.encode()]
            logger.info("index_legacy_watermark_removed")

    # ── 从 Memory Hub 增量重建 ──

    # ── 管理事件应用（full 重放与增量重放共用同一实现）───────────────────────

    _ADMIN_EVENTS_WATERMARK_KEY = b"__meta__/admin_events_replay_watermark"

    def _replay_new_admin_events(self, ledger: MemoryHub) -> int:
        """增量重放**新增**管理事件（2026-08-14，matter-dedup-keywords 卡）。

        缺口：`_replay_admin_events` 只在 full rebuild（clear_db）时调用——proxy
        端管理操作拿不到写锁时回 "deferred: 下次 rebuild 生效"，而正常运行永远
        没有下一次 full rebuild → deferred 的 merge/close/rename 无限期悬置
        （用户实测：三条 MATTER_MERGE 落账后 Matter 纹丝不动）。

        修法 = 水位游标：entry_id `{unix_ms:013d}-{seq:08d}` 定宽、字典序即时间序，
        每个增量 pass 重放 key > 水位的事件后推进水位。**每 pass 必调、放在
        turn 扫描之前**——"无新 turn 提前 return" 的分支也必须够得到
        （ADR-0024 T8b B17 的教训：挂尾部的维护永远等不到空闲轮）。
        应用实现与 full 重放共用（`_apply_admin_event`），语义不分叉。
        """
        if self.meta_db is None:
            return 0
        raw = self.meta_db.get(self._ADMIN_EVENTS_WATERMARK_KEY)
        watermark = raw.decode() if isinstance(raw, bytes) else (raw or "")
        replayed = 0
        last_key = ""
        for key_str, event in ledger.scan_admin_events():
            entry_id = key_str.split("/", 1)[-1]
            if watermark and entry_id <= watermark:
                continue
            if event.event_type == "scope_promote":
                if self._apply_scope_promote_event(event):
                    replayed += 1
            elif self._apply_admin_event(event):
                replayed += 1
            last_key = max(last_key, entry_id)
        if last_key:
            self._set_admin_events_watermark(last_key)
        if replayed > 0:
            logger.info("index_admin_events_incremental_replayed", count=replayed)
        return replayed

    def _set_admin_events_watermark(self, entry_id: str) -> None:
        if self.meta_db is not None and entry_id:
            self.meta_db[self._ADMIN_EVENTS_WATERMARK_KEY] = entry_id.encode()

    def _replay_admin_events(self, ledger: MemoryHub) -> int:
        """重放 Memory Hub 中的管理事件（ADR-0012 §3.5/§3.6）。

        按写入顺序重放：matter_create → assign → merge → close → rename。
        用于全量重建时恢复用户手动映射（Memory Index 重建不丢用户劳动）。
        重放完把增量水位推到最后一条——防下一个增量 pass 重复应用
        （merge 等操作有幂等护栏，但计票类字段重复应用会脏）。
        """
        replayed = 0
        last_entry = ""
        for key_str, event in ledger.scan_admin_events():
            last_entry = max(last_entry, key_str.split("/", 1)[-1])
            if event.event_type == "scope_promote":
                continue  # scope 由 _replay_scope_promotions 单独重放（既有分工）
            if self._apply_admin_event(event):
                replayed += 1
        if last_entry:
            self._set_admin_events_watermark(last_entry)
        if replayed > 0:
            logger.info("index_admin_events_replayed", count=replayed)
        return replayed

    def _apply_admin_event(self, event: AdminEvent) -> bool:
        """应用单条管理事件（scope_promote 除外）。返回是否成功计数。"""
        replayed = 0
        if event is not None:
            try:
                if event.event_type == "audience_set":
                    # MQ-L27 修法③：人设事实 audience 补标（payload 带 fact_id，
                    # matter_id 留空——MQ-A7 护栏）。幂等：重复应用同值无害。
                    fid = event.payload.get("fact_id", "")
                    aud = event.payload.get("audience", "")
                    f = self.get_fact(fid) if fid else None
                    if f is None or not aud:
                        return False
                    f.audience = aud
                    from datetime import UTC as _UTC_AUD
                    f.updated_at = datetime.now(_UTC_AUD)
                    self.add_fact(f)
                    return True
                if event.event_type == "matter_create":
                    title = event.payload.get("title", "")
                    summary = event.payload.get("summary", "")
                    # ADR-0021 §2.4 补平：手动建卡的 scope 从 payload 取。
                    # 历史事件没有该字段 -> 空 -> 走读侧的回落语义（与个人模式一致），
                    # **不能**在这里瞎猜一个 scope：管理事件是 append-only 的事实记录，
                    # 重放时补一个当时不存在的值 = 伪造历史（重建等价性依赖逐字重放）。
                    matter = Matter(
                        matter_id=event.matter_id,
                        title=title, summary=summary,
                        scope=event.payload.get("scope", "") or "",
                        origin=MatterOrigin.MANUAL,
                    )
                    self.add_matter(matter)
                    replayed += 1

                elif event.event_type == "matter_assign":
                    tt_str = event.payload.get("target_type", "session")
                    tt = EdgeTargetType.SESSION if tt_str == "session" else EdgeTargetType.FACT
                    self.assign_manual(event.matter_id, tt, event.target_key)
                    replayed += 1

                elif event.event_type == "matter_merge":
                    source_id = event.payload.get("source_matter_id", event.target_key)
                    self.merge_matters(source_id, event.matter_id)
                    replayed += 1

                elif event.event_type == "matter_detach":
                    edge_id = event.payload.get("edge_id", "")
                    target_key = event.payload.get("target_key", "") or event.target_key
                    if edge_id:
                        self.detach_edge(edge_id)
                    elif target_key:
                        tt_str = event.payload.get("target_type", "session")
                        tt = EdgeTargetType.SESSION if tt_str == "session" else EdgeTargetType.FACT
                        self.detach_edges_for_target(target_key, tt)
                    replayed += 1

                elif event.event_type == "matter_split":
                    new_matter_id = event.payload.get("new_matter_id", "")
                    new_title = event.payload.get("new_title", "")
                    edge_ids = event.payload.get("edge_ids", [])
                    if new_matter_id:
                        self.split_matter(
                            event.matter_id, new_matter_id, new_title, edge_ids,
                        )
                        replayed += 1

                elif event.event_type == "matter_close":
                    self.close_matter(event.matter_id)
                    replayed += 1

                elif event.event_type == "matter_rename":
                    title = event.payload.get("title", "")
                    self.rename_matter(event.matter_id, title)
                    replayed += 1

                # ── Beta T12 快照导入实体重放（幂等：按 id upsert）──
                # payload = {"entity": <dump>}（展开会与位置参数撞 matter_id 等键名）
                elif event.event_type == "fact_import":
                    fact = Fact.model_validate(event.payload.get("entity", event.payload))
                    fact.embedding = None
                    if self._embedder is not None:
                        try:
                            fact.embedding = embed_passage_compat(
                                self._embedder, [fact.content])[0]
                        except Exception as ee:  # noqa: BLE001 —— 无向量仍入 meta
                            logger.debug("index_import_replay_embed_skip", error=str(ee))
                    self.add_fact(fact)
                    replayed += 1

                elif event.event_type == "matter_import":
                    matter = Matter.model_validate(event.payload.get("entity", event.payload))
                    matter.centroid = None
                    matter.embedding = None
                    self.add_matter(matter)
                    replayed += 1

                elif event.event_type == "edge_import":
                    edge = MatterEdge.model_validate(event.payload.get("entity", event.payload))
                    self.add_edge(edge)
                    replayed += 1
            except Exception as e:
                logger.warning("index_admin_event_replay_failed",
                               event_type=event.event_type.value,
                               matter_id=event.matter_id, error=str(e))
        return replayed > 0

    def _apply_scope_promote_event(self, event: AdminEvent) -> bool:
        """应用单条 SCOPE_PROMOTE（与 _replay_scope_promotions 同实现，供增量重放）。"""
        try:
            target_type = event.payload.get("target_type", "fact")
            new_scope = event.payload.get("new_scope", "")
            if not new_scope:
                return False
            target_key = event.matter_id
            if target_type == "matter":
                m = self.get_matter(target_key)
                if m is None:
                    return False
                m.scope = new_scope
                self.update_matter(m)
                return True
            f = self.get_fact(target_key)
            if f is None:
                return False
            f.scope = new_scope
            from datetime import UTC as _UTC_SCOPE
            f.updated_at = datetime.now(_UTC_SCOPE)
            self.add_fact(f)
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("index_scope_promote_replay_failed",
                           target_key=event.matter_id, error=str(e))
            return False

    def build_agent_claim_map(self, ledger: MemoryHub) -> dict[str, str]:
        """见模块级 :func:`build_agent_claim_map`（薄委托，保持既有调用点不变）。"""
        return build_agent_claim_map(ledger)

    def _replay_scope_promotions(self, ledger: MemoryHub) -> int:
        """重放 Memory Hub 中的 SCOPE_PROMOTE 管理事件（ADR-0021 section 2.4）。

        把 Fact/Matter 的 scope 提升到 team/org（rebuild 后重放，提升不丢失）。
        matter_id 字段复用为 target_key；payload={target_type, new_scope}。
        """
        replayed = 0
        for _key, event in ledger.scan_admin_events():
            if event.event_type != "scope_promote":
                continue
            try:
                target_type = event.payload.get("target_type", "fact")
                new_scope = event.payload.get("new_scope", "")
                if not new_scope:
                    continue
                target_key = event.matter_id
                if target_type == "matter":
                    m = self.get_matter(target_key)
                    if m is not None:
                        m.scope = new_scope
                        self.update_matter(m)
                        replayed += 1
                else:
                    f = self.get_fact(target_key)
                    if f is not None:
                        f.scope = new_scope
                        # ADR-0027 §5.3：scope 提升是变更，导出增量必须能看见
                        # （否则目标端仍按旧 scope 呈现）。
                        from datetime import UTC as _UTC_SCOPE
                        f.updated_at = datetime.now(_UTC_SCOPE)
                        self.add_fact(f)
                        replayed += 1
            except Exception as e:
                logger.warning("index_scope_promote_replay_failed",
                               target_key=event.matter_id, error=str(e))
        if replayed > 0:
            logger.info("index_scope_promotions_replayed", count=replayed)
        return replayed

    def reattribute_by_claims(self, ledger: MemoryHub, *,
                              dry_run: bool = False) -> dict[str, int]:
        """按 `AGENT_CLAIM` 映射**只改归属、不重蒸**（⑤，2026-08-26）。

        🔴 **为什么必须单开这条路**（本次验证的产出）：`AGENT_CLAIM` 的读侧映射只在
        `rebuild_from_hub` 构造 `ConversationTurn` 时生效，也就是**只有重新蒸馏
        才会写出新归属**。而定向重放（`--unconsume --agents …`）**不清库、保留
        novelty**，纠正版与库内旧 Fact 内容逐字相同 ⇒ 判重命中 ⇒
        `consolidation_proxy` 里直接 `continue` 丢弃（`verdict="vector_kill"`）
        ⇒ **归属没改成，而且不报错**。静默无效比报错更糟。

        剩下唯一可行的路是全量重建（`clear_db=True` 关掉 novelty），代价是 2.5 小时
        + 全库重蒸的 LLM 费用——**为了改一个字段把整个派生层重算一遍**，
        与这件事的性质完全不成比例：`AGENT_CLAIM` 改的是"派生层的解释"
        （`build_agent_claim_map` 的原话），而解释的更新不需要重新蒸馏。

        所以本方法是一次 **UPDATE**：不碰 content、不碰向量、不调 LLM。
        走 `get/mutate/add` 三步（与 `_replay_scope_promotions` 同款 upsert 路径）。

        连带修的更一般的问题：`Fact.agent_id` 此前**只在创建时写入一次、此后永不更新**，
        任何"事后改归属"的机制都会撞上这堵墙，不只是 AGENT_CLAIM。

        :param dry_run: 只统计不落盘（先看看会改多少，再决定要不要真改）。
        :returns: `{"facts": n, "matters": n, "pairs": 映射条目数}`
        """
        from datetime import UTC as _UTC_RA

        claim_map = build_agent_claim_map(ledger)
        out = {"facts": 0, "matters": 0, "pairs": len(claim_map)}
        if not claim_map:
            logger.info("index_reattribute_noop", reason="no agent_claim events")
            return out

        for fact in self.all_facts():
            new_id = claim_map.get(fact.agent_id or "")
            if not new_id or new_id == fact.agent_id:
                continue
            out["facts"] += 1
            if dry_run:
                continue
            fact.agent_id = new_id
            # 归属变更是**变更**，导出增量必须能看见（ADR-0027 §5.3 同款理由）。
            fact.updated_at = datetime.now(_UTC_RA)
            self.add_fact(fact)

        # 🔴 Matter 卡的 `participants` 也按 agent_id 聚合（U4 `touch_participant`）。
        # 只改 Fact 会让 Matter 卡上继续挂着旧名字——而 Matter 卡是**整卡注入**的，
        # 旧名字会每轮出现在模型眼前。
        for matter in self.all_matters():
            hit = False
            for pt in matter.participants or []:
                new_id = claim_map.get(pt.agent_id or "")
                if new_id and new_id != pt.agent_id:
                    hit = True
                    if not dry_run:
                        pt.agent_id = new_id
            if not hit:
                continue
            out["matters"] += 1
            if dry_run:
                continue
            # 同一 Matter 上"旧名 + 新名"两条 participant 要合并，否则计数被拆成两份
            merged: dict[str, Any] = {}
            for pt in matter.participants or []:
                cur = merged.get(pt.agent_id)
                if cur is None:
                    merged[pt.agent_id] = pt
                    continue
                cur.turns += pt.turns
                cur.first_seen = min(cur.first_seen, pt.first_seen)
                cur.last_seen = max(cur.last_seen, pt.last_seen)
            matter.participants = list(merged.values())
            self.update_matter(matter)

        logger.info("index_reattributed", dry_run=dry_run, **out)
        return out

    def rebuild_from_hub(
        self, ledger: MemoryHub, *, full: bool = False, max_turns: int = 30,
        legacy_map: dict[str, str] | None = None,
        fold_user_id: str = "",
        since: str = "", until: str = "",
        agents: set[str] | None = None,
        exclude_agents: set[str] | None = None,
        distill_concurrency: int | None = None,
        progress_cb: Any | None = None,
    ) -> int:
        """从 Memory Hub 增量/全量重建 Memory Index（ADR-0009 §7 + ADR-0012 §3.6）。

        full=True: 重放会话数据 → 归属判定 → 成功后才清空旧 Memory Index + 重放管理事件。
        full=False: 增量重建（consumed-key 集合之后的新数据）。

        定向过滤（since/until/agents/exclude_agents，2026-07-28）：只处理匹配的
        turn，不匹配的既不蒸馏也不标记消费（保持未消费，日后可再处理）。用于
        "只重放某窗口/某 agent" 与 "永久排除压测流量"。**full=True 带过滤时不清库**
        ——清库+子集重放 = 其余数据静默丢失，与"重建等价性"冲突（ADR-0012 §3.6）；
        带过滤的 full 退化为"忽略 consumed 标记的定向重放"。

        安全重建（review 20260707）：full=True 时不在开头 clear()，而是先
        consolidate 成功后再 clear+store。若 consolidation 失败，旧 Memory Index 数据
        完整保留，避免"drop 在前、重建失败 = 数据全丢"。

        公式（ADR-0012 §3.6）：重放会话数据 + 重放管理事件 − 墓碑覆盖项。
        scan_prefix 已跳过墓碑覆盖的 turn。
        重建期只降级不停摆（consolidation 失败不阻塞）。
        返回新提炼的 Fact 数。
        """
        self._last_rebuild_backlog = False
        if self._embedder is None:
            logger.warning("index_rebuild_skip", reason="no_embedder")
            return 0

        # MS-13：全量重建 = 新的一轮 run，刷新 rebuild_id（增量 pass 沿用进程 run id
        # ——复验要区分的是"16:33 那次全量"与"12:44 的历史 run"，进程粒度足够）。
        if full:
            from datetime import UTC as _UTC
            from datetime import datetime as _dtm
            self._rebuild_id = _dtm.now(_UTC).strftime("%Y%m%dT%H%M%S")
            logger.info("index_rebuild_run_id", rebuild_id=self._rebuild_id)

        # G4: 失效 manual edge 缓存（replay_admin_events 可能改了 manual edges）
        self._invalidate_manual_edge_cache()
        # 一次性清理遗留水位游标（字典序单游标方案已废弃，review 20260707）
        self._cleanup_legacy_watermark()

        # 定向过滤（2026-07-28）：带过滤的 full 只"忽略 consumed 标记"，不清库
        # （清库+子集重放 = 其余数据静默丢失）。clear_db 才是真正的全量语义。
        has_filter = bool(since or until or agents or exclude_agents)
        clear_db = full and not has_filter
        if full and has_filter:
            logger.info("index_rebuild_filtered_replay_no_clear",
                        since=since or None, until=until or None,
                        agents=sorted(agents) if agents else None,
                        exclude_agents=sorted(exclude_agents) if exclude_agents else None)

        # 增量重放新增管理事件（2026-08-14）：proxy 端 deferred 的 merge/close/
        # rename 在这里兑现。🔴 必须在 turn 扫描**之前**——"无新 turn 提前 return"
        # 分支也要够得到（B17 教训）。clear_db 路径跳过：清库后 full 重放会应用
        # 全量事件并推进水位，先应用一遍纯属浪费。
        if not clear_db:
            try:
                self._replay_new_admin_events(ledger)
            except Exception as e:  # noqa: BLE001 —— 管理事件重放失败不阻塞蒸馏主线
                logger.warning("index_admin_events_incremental_failed", error=str(e))

        # G11.11：AGENT_CLAIM 读侧映射（认领 / 改名 / 合并后，历史 turn 的归属跟着走）。
        # 在 turn 扫描之前构造一次；扫描失败不阻塞蒸馏主线（映射为空 = 保持原 agent_id）。
        try:
            agent_claim_map = self.build_agent_claim_map(ledger)
        except Exception as e:  # noqa: BLE001 —— 映射构造失败不该拖垮重建
            logger.warning("index_agent_claim_map_failed", error=str(e))
            agent_claim_map = {}

        # G3: LanceDB 可用时用向量检索做 novelty 检查，不全量加载 facts
        # clear_db=True 时不用 novelty_checker -- clear() 在 consolidation 之后执行，
        # LanceDB 里还是旧数据，会把所有候选误判为重复（G3 回归修复）。
        # 定向重放（full+过滤）不清库 -> 保留 novelty，只补缺失 Fact 不重复堆积。
        #
        # 🔴 消费方唯一性（M5-2 判据③修复，2026-08-09）：这个 None **只影响判重**。
        # 此前裁决的邻居查询（`_neighbors_for`）复用同一个对象，于是全量重建时
        # 跨批邻居被整段关掉——同簇不同轮的矛盾对永远进不了同一个裁决包
        # （08-09 实测 1178 候选仅 139 真实判决）。裁决邻居现在走 consolidator
        # 内部的 `InRunNeighborSource`（本轮内存集 + 增量时的库内视图），
        # 与判重语义彻底拆开。别再把这个对象接回裁决路径。
        novelty_checker = (None if clear_db
                           else (_LanceDBNoveltyChecker(
                               self._table, self._novelty_threshold,
                               entity_aware=self._entity_aware_novelty,
                               entity_overlap_threshold=self._entity_overlap_threshold,
                               topk=self._novelty_topk,
                           )
                                 if self._table is not None else None))
        # 蒸馏并发。progress_cb 供 sync 任务把进度回写控制通道（None = 零行为变化）。
        #
        # 🔴 2026-08-22：这里曾是同一个参数的**第三个默认值**（就地
        # `os.environ.get(..., "1")` = 串行；另两个是 `bladex sync` 的 4 与守护进程的 1）。
        # 三条路三个默认值，没人说得出生产在跑哪个——MQ-W1 积压 720 轮的成因之一。
        # 现在统一读 `flags.BLADEX_DISTILL_CONCURRENCY`（唯一真相源，与 .env.example 对账）。
        # 🔴 用模块级 `flag_number`（第 45 行），**不要在这里写局部 import**：
        # 函数体内任何一处 `from … import flag_number` 都会把这个名字变成整个函数的
        # 局部变量，而这段在 `if` 分支里 —— 调用方传了值就跳过 import，函数后面
        # 那几处 `flag_number(...)` 当场 UnboundLocalError（2026-08-22 实际踩到，
        # gate 三条 sync 用例全红；同文件 1771 行的 `as _fn` 别名是前人绕开它的痕迹）。
        if distill_concurrency is None:
            distill_concurrency = max(1, int(flag_number("BLADEX_DISTILL_CONCURRENCY")))
        consolidator = ProxyConsolidator(
            embedder=self._embedder,
            novelty_threshold=self._novelty_threshold,
            distiller=self._distiller,
            novelty_checker=novelty_checker,
            sensitivity_config=self._sensitivity_config,
            entity_aware=self._entity_aware_novelty,
            entity_overlap_threshold=self._entity_overlap_threshold,
            distill_concurrency=distill_concurrency,
            progress_cb=progress_cb,
            funnel=self._funnel,          # M4-2：写入侧漏斗埋点
            adjudicator=self._adjudicator,  # M2-2：写入时裁决
        )

        def _prog(**ev: Any) -> None:
            if progress_cb is None:
                return
            try:
                progress_cb(ev)
            except Exception:  # noqa: BLE001 —— 进度观测绝不影响重建本体
                pass
        # clear_db=True 用空 existing（干净重算）
        # 否则 + LanceDB -> 空（novelty 由向量检索处理）
        # 否则无 LanceDB -> 全量 facts（brute-force novelty）
        if clear_db or novelty_checker is not None:
            existing = []
        else:
            existing = self.all_facts()

        # 从 Memory Hub 读未消费的 turns（consumed-key 集合去重，替代字典序水位游标）
        turns: list[ConversationTurn] = []
        new_consumed_keys: list[str] = []
        # T1: ledger_key → user 消息文本（供归属管线续接语/点名检测）
        ledger_key_to_user_msg: dict[str, str] = {}
        # U4B-B2: ledger_key → turn 时间戳 iso（lifecycle 事件用 turn 时间，保重建等价性 G6）
        ledger_key_to_ts: dict[str, str] = {}
        # MQ-L26: ledger_key → 这一轮**给没给账本面**（三态，见 `Turn.ledger_face`）。
        # 归属循环里只有 `fact.source_ledger_key`、拿不到 Turn，故在扫描阶段建映射
        # （与上面两个 dict 同款）。缺键 = 该 turn 没有这个字段 ⇒ 按旧行为锚定。
        ledger_key_to_face: dict[str, bool | None] = {}
        # ── GM-1（MQ-S13）：对话延续 → 上一轮的 ledger_key ──
        # 任务身份的载体是**延续关系**，不是标题相似度（三轮读数否掉了键匹配路线：
        # 开卡那一刻种子键达标率只有 3.8%）。信号 `Turn.session_id_source ==
        # TAIL_CONTINUATION` 一直在库里，归属层从不读它。
        # 🔴 只用来**往候选池里加卡**，判定仍交 L4（见 continuation.py 模块 docstring）。
        # 扫描已按 `(ts, key)` 全局排序，故"同 session 的上一轮"= 该 session 上一次
        # 在本循环里出现的 key。
        cont_prev_key: dict[str, str] = {}          # 本轮 key → 上一轮 key
        cont_new_subjects: dict[str, int] = {}      # 本轮 key → 新主体信号数（读数用）
        # 上一轮 key → (user+assistant 参照文本, 是否 aux)。本 pass 内的轮免费填；
        # 批边界外的前一轮付一次 `ledger.get()`（见 `_prev_ref` / `_prev_of`）。
        _prev_ref_cache: dict[str, tuple[str, bool]] = {}
        _cont_enabled = flag_enabled("BLADEX_ATTRIB_CONTINUATION")
        _cont_new_subject_max = int(flag_number("BLADEX_ATTRIB_CONT_NEW_SUBJECT_MAX"))
        # ── G12.2 D1（ADR-0031 §3；算法 = 决策表 v2）──
        # 依赖 GM-1 的扫描段（_prev_of 会话序 + _prev_ref 参照缓存），故与其同门。
        _d1_enabled = flag_enabled("BLADEX_TASK_STATE_D1") and _cont_enabled
        _d1_scan: dict[str, dict] = {}   # ledger_key -> 轮级 D1 输入（本 pass 的轮）

        skipped_auxiliary = 0
        skipped_filtered = 0
        backlog = False
        # ADR-0028 E2.3：注入命中计数（fact_id -> 本轮聚合次数）。
        # 生产者 = proxy 的 decision_meta.injected_fact_ids（随 turn 落 Memory Hub）；
        # 消费者 = 本函数尾部的 ref_count 回写 + importance 重算。
        # 幂等性由 consumed 标记天然保证：每个 turn 只被消费一次。
        injection_hits: Counter[str] = Counter()
        # MQ-S38：fact_id → 当时的取代键（换版后 id 指空时据此接续）
        injection_keys: dict[str, str] = {}
        # M3-1：命中时间（取 turn 时间，保重放等价）+ 会话内去重集合 + Matter 命中
        hit_ts: dict[str, str] = {}
        matter_hit_ts: dict[str, str] = {}
        hit_seen: set[tuple[str, str]] = set()
        if full:
            max_turns = 0  # 全量重建不限（重放所有 turn）

        # ① 惰性扫描（2026-07-28）：先只按 (key, ts) 筛，命中的才付完整反序列化。
        # 实测 6891 turns/1.2GB：全量 scan_prefix > 5min，scan_meta ~2.5s。
        # consolidator 每轮只取 ≤max_turns 条，却曾为全库付代价（O(n²) 重建）。
        # 🔴 MS-11：**按 `(ts, key)` 排序后再消费**（复核 20260807 发现一）。
        #
        # `scan_meta` 走 RocksDB 的 key 序，而 key = `principal/agent/session/entry_id`
        # —— entry_id 带时间戳前缀、**session 内有序**，但 session 段是指纹哈希，
        # **跨 session 的时间顺序被打乱**。实证：修正宣告（08-05、`fp:2aaa`）的 key
        # 字典序排在旧断言（08-03、`fp:7bbb`）**之前**。
        #
        # 为什么这会出事：写入时裁决的 prompt 假设"候选是更新的那条"。
        # 这个假设在**增量**下由到达顺序隐式成立（新 turn 天然后到），
        # 但全量重建/积压重放没有这个保证 —— 旧断言以"NEW item"身份进入裁决，
        # 修正宣告成了它的邻居，于是**旧值把新值 t_invalid**，取代链方向整个反转。
        # 恰好毁掉 M5-2 验收③要验证的东西，而且确定性没破（每次重建结果一致）、
        # 只是**语义确定地错** —— 最难发现的那一类。
        #
        # 排序键 `(ts, key)`：ts 相同时用 key 兜底，保证重放两次逐位一致（G6）。
        # 内存代价：~7k 条 (str, str) 元组，可忽略（实测全库 6891 turns）。
        _scan_batch: list[tuple[str, str]] = []
        # GM-1：会话内相邻关系必须按**全部**轮次算，不能只按本批次。
        # 🔴 首版把"上一轮"记在一个 pass 内的字典里，于是**跨批次的延续对全部丢失**
        # ——consolidator 每 60s 一批，而人说话的间隔通常大于 60s，绝大多数相邻轮
        # 天生落在不同批里。gate 上三条阴性对照连着红，日志里 `cont_pairs=0` 才把它
        # 抖出来：机制"接上了"但对生产流量几乎恒不触发（本仓第八例接线断）。
        # 修法零额外 I/O：在这趟已有的全量 `scan_meta` 里顺手记下会话内顺序，
        # 已消费的轮也要记（它正是要找的那个前驱）。过滤掉的轮不记——与定向重放
        # "不匹配的既不蒸馏也不标记消费"同语义。
        _sess_keys: dict[str, list[str]] = defaultdict(list)
        for _k, _t in ledger.scan_meta(""):
            if has_filter and not _turn_matches_filter(
                _k, _t, since=since, until=until,
                agents=agents, exclude_agents=exclude_agents,
            ):
                skipped_filtered += 1
                continue
            if _cont_enabled:
                _sess_keys[_k.rsplit("/", 1)[0]].append(_k)
            if not full and self.is_consumed(_k):
                continue
            _scan_batch.append((_t, _k))

        # 会话前缀内 key 字典序 == 时间序（entry_id = `{unix_ms}-{seq}`，session 内有序；
        # 跨 session 才乱，而这里从不跨 session 比较）。
        _prev_of: dict[str, str] = {}
        for _keys in _sess_keys.values():
            _keys.sort()
            for _a, _b in zip(_keys, _keys[1:], strict=False):
                _prev_of[_b] = _a
        _scan_batch.sort()
        logger.info("index_rebuild_scan_sorted", candidates=len(_scan_batch),
                    skipped_filtered=skipped_filtered)

        for ts_iso, key_str in _scan_batch:
            # 通过筛选 -> 才付完整加载（Pydantic 校验 + content_ref 还原）
            turn = ledger.get(key_str)
            if turn is None:
                logger.warning("index_rebuild_turn_missing", key=key_str)
                continue

            # ADR-0028 E2.3 + M3-1：注入命中计数（在 aux 判定之前收集——aux 轮只注硬规则、
            # injected_fact_ids 天然为空，放这里是为了"每个被消费的 turn 都被统计到"）。
            #
            # 🔴 M3-1 会话内去重（复核 X1/G5 的"机会归一"护栏）：同一会话内同一条目
            # 只计**首次**。不去重的话，一个工具循环里连注 30 轮的条目会拿到 30 次命中，
            # 而另一条在别的会话被真正用上一次的只有 1 次 —— 相关钟会把
            # "碰巧落在长会话里"读成"更有相关性"，富者愈富（G5）。
            _dm = getattr(turn, "decision_meta", None)
            _sid = turn.identity.session_id
            _ts_iso = turn.ts.isoformat() if turn.ts else ""
            _keys_of_turn = getattr(_dm, "injected_fact_keys", None) or {}
            for _fid in (getattr(_dm, "injected_fact_ids", None) or []):
                if not _fid:
                    continue
                _k = (_sid, str(_fid))
                if _k in hit_seen:
                    continue
                hit_seen.add(_k)
                injection_hits[str(_fid)] += 1
                # MQ-S38：随手记下这条 id 当时的稳定身份（取代键）。
                # 历史 turn 没有这个字段 → 不记 → 回写侧退回"只按 id"（现状）。
                _sk = _keys_of_turn.get(str(_fid), "")
                if _sk:
                    injection_keys[str(_fid)] = _sk
                # 相关钟取**这一轮的时间**，不取 now：重放两次必须逐字节一致（G6）。
                if _ts_iso > hit_ts.get(str(_fid), ""):
                    hit_ts[str(_fid)] = _ts_iso
            # M3-1：Matter 卡命中（三时钟里此前唯一没有信号源的那个）
            for _mid in (getattr(_dm, "injected_matter_ids", None) or []):
                if not _mid:
                    continue
                _k = (_sid, f"m:{_mid}")
                if _k in hit_seen:
                    continue
                hit_seen.add(_k)
                if _ts_iso > matter_hit_ts.get(str(_mid), ""):
                    matter_hit_ts[str(_mid)] = _ts_iso

            # T2(ADR-0018 R2): 不信任冻结的 turn.auxiliary，用最新规则重判
            # （identity 初判可能漏 user 角色的委派/子任务；rebuild 用 classify_auxiliary 补）
            from bladex_core.task_goal import TURN_DROPPED, TURN_REAL
            from bladex_proxy.identity import classify_turn_disposition

            # G12.3：三分收进 `classify_turn_disposition`（原地展开的那一行判据
            # 现在有第二个消费者——goal 捕获；两处各抄一份就是 MQ-S4 的形态）。
            # 语义逐字不变：dropped = 原 `is_aux and rule not in DISTILL_ONLY_AUX_RULES`。
            _turn_class, _aux_rule = classify_turn_disposition(turn.request_messages)
            is_aux = _turn_class != TURN_REAL
            # 🔴 MS-2：整轮丢**只对 Hermes 原生 aux 规则**（委派回执、压缩检查点这类
            # 从头到尾都是内部通信的轮次）。信封类指纹（`DISTILL_ONLY_AUX_RULES`）说的是
            # "用户那侧这条消息是信封"，**不代表这一轮 assistant 没做实事**——整轮丢会把
            # 结论/工作产出一起丢掉（实测 claude-code 123 轮里 120 轮被跳过）。
            # 这些轮次照常往下走：user 侧交给 envelope 剥离，assistant 侧照常收。
            #
            # 这条判断是 `DISTILL_ONLY_AUX_RULES` 的**第二个消费方**。此前它只有路由侧
            # 一个消费者，于是 2026-07-29 记录的"信封指纹改为交给 envelope 剥离"从未生效
            # ——集合建好了、语义写清了、没人读它。又一例"机制写完接线断"。
            # GM-1：本轮的参照文本进缓存，供**下一轮**（如果它在同一批里）免费取用。
            # aux 轮也记，但标记 aux —— 与探针口径一致（相邻关系按全部轮次算，
            # 任一端是 aux 则该对不参与判定）：脚手架轮不产生任务身份，拿它当参照会把
            # 无关的前后卡算成"该继承"，MQ-S12 那三条模板正是这个形态。
            if _cont_enabled:
                _prev_ref_cache[key_str] = (
                    "" if is_aux else _continuation_ref_text(turn), is_aux)
            if _turn_class == TURN_DROPPED:
                skipped_auxiliary += 1
                new_consumed_keys.append(key_str)
                continue

            # 归一成 ConversationTurn
            user_messages = []
            for msg in turn.request_messages:
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        user_messages.append(content)
                    elif isinstance(content, list):
                        texts = [
                            p.get("text", "") for p in content
                            if isinstance(p, dict) and p.get("type") == "text"
                        ]
                        if texts:
                            user_messages.append(" ".join(texts))

            # ADR-0019 P2：结论判定——status=ok、有正文、本轮无工具调用（direction=call）
            # = 逻辑轮的 final answer。中间往返/失败轮不蒸结论。
            _has_call = any(
                getattr(te, "direction", "") == "call" for te in turn.tool_events
            )
            _conclusion = (
                turn.response_text
                if (turn.status.value == "ok"
                    and turn.response_text.strip()
                    and not _has_call)
                else ""
            )
            # 🔴 M0-9（复核 D1）：工作产出通道接线。
            # `ConversationTurn.assistant_progress` 与它的消费端
            # （`_collect_progress_candidates`）2026-07-29 就写好了，但**归一处从来
            # 没给它赋过值** —— 字段恒为空串，于是那条通道自建成起一条 fact 都没产出过。
            # 语义与结论通道互斥（见 fact.py:44-52）：无 tool call = 结论（最终回答），
            # 有 tool call = 产出（模型在调工具前写的说明/分析/发现）。
            # 编程 agent 全程泡在工具循环里（实测 codex 37 轮中 35 轮带 tool call），
            # 对它们而言这条才是主通道。
            _progress = (
                turn.response_text
                if (turn.status.value == "ok"
                    and turn.response_text.strip()
                    and _has_call)
                else ""
            )

            # ADR-0021 section 2.5: legacy user_id(hash8) -> principal 读侧映射。
            # legacy_map 来自 identity.toml（consolidator 传）。
            #
            # 🔴 fold_user_id（ADR-0021 §2.3 修订，2026-08-16）：**个人模式下把一切
            # user_id 折叠成同一个身份**。个人模式（无 identity.toml）本就没有 user 维度
            # ——存量 turn 里那些 key hash8 是"把凭证当身份"时代的产物，不是不同的人。
            # 不折叠的话，改完身份语义后新流量落 `local`、存量仍是 hash8，检索按
            # visibility 过滤 -> 当场制造一个新的两分裂（正是本次修订要消除的东西）。
            # 比 legacy_map 简单：不需要事先知道有哪些历史 user_id，也不需要 identity.toml。
            raw_uid = turn.identity.user_id
            if fold_user_id:
                mapped_uid = fold_user_id
            else:
                mapped_uid = legacy_map.get(raw_uid, raw_uid) if legacy_map else raw_uid
            # ADR-0024 §4.5 / U4：本轮当前（open）单元 unit_key（供蒸出 facts 继承 → L0）。
            _task_units = getattr(turn, "task_units", None) or []
            _unit_key = _task_units[-1].unit_key if _task_units else ""
            conv_turn = ConversationTurn(
                session_id=turn.identity.session_id,
                user_id=mapped_uid,
                user_messages=user_messages,
                assistant_response=turn.response_text,
                assistant_conclusion=_conclusion,
                assistant_progress=_progress,   # M0-9（D1）：此前恒空，通道从未通电
                logical_turn=turn.logical_turn,
                roundtrip=turn.roundtrip,
                ledger_key=key_str,
                timestamp=turn.ts.isoformat() if turn.ts else "",
                sensitivity=getattr(turn, "sensitivity", "normal"),
                unit_key=_unit_key,
                # G11.11：AGENT_CLAIM 读侧映射（认领 / 改名 / 合并后历史数据跟着走）。
                # 与 user_id 的 legacy_map 同构但独立表，撤销一方不影响另一方。
                agent_id=agent_claim_map.get(
                    turn.identity.agent_id, turn.identity.agent_id
                ) if agent_claim_map else turn.identity.agent_id,
                tool_file_refs=_extract_file_refs(turn.tool_events),
            )
            turns.append(conv_turn)
            if len(turns) % 100 == 0:
                _prog(phase="scan", done=len(turns), total=0)
            if user_messages:
                ledger_key_to_user_msg[key_str] = "\n".join(user_messages)
            ledger_key_to_ts[key_str] = conv_turn.timestamp
            ledger_key_to_face[key_str] = getattr(turn, "ledger_face", None)

            # ── GM-1：判「延续 ∧ 无新主体」，登记上一轮 key（不做任何归属决定）──
            if _cont_enabled and not is_aux:
                from bladex_core.continuation import is_task_continuation
                from bladex_core.query_understanding import last_user_text

                # Turn 与 Identity 两处都带这个字段（server.py:2187 从 identity 复制
                # 到 Turn）。取 Turn 级为主、Identity 兜底：存量 turn 若只有其中一处，
                # 判定不该因为读错位置而静默失效。
                _src = (getattr(turn, "session_id_source", "")
                        or getattr(turn.identity, "session_id_source", "") or "")
                _prev_k = _prev_of.get(key_str, "")
                if _prev_k and "tail" in str(_src).lower():
                    _pref, _prev_aux = _prev_ref(
                        _prev_k, _prev_ref_cache, ledger)
                    if not _prev_aux:
                        _is_cont, _n_new = is_task_continuation(
                            session_id_source=_src,
                            current_text=last_user_text(turn.request_messages),
                            reference_text=_pref,
                            new_subject_max=_cont_new_subject_max,
                        )
                        if _is_cont:
                            cont_prev_key[key_str] = _prev_k
                            cont_new_subjects[key_str] = _n_new

            # ── G12.2 D1：收集轮级判定输入（决策引擎在归属段消费）──
            if _d1_enabled:
                from bladex_core.query_understanding import last_user_text as _lut
                _d1_intent = _lut(turn.request_messages)
                _d1_scan[key_str] = {
                    "agent": conv_turn.agent_id,
                    "session": turn.identity.session_id,
                    "src": (getattr(turn, "session_id_source", "")
                            or getattr(turn.identity, "session_id_source", "") or ""),
                    # aux 四不作用于**归属层**（表 §2；蒸馏/session 层是另两层，别混）
                    "aux": bool(getattr(turn.identity, "auxiliary", False)),
                    "intent": _d1_intent,
                    # 锚参照文本 = 本轮 user+assistant（S1 门② 参照集必须含回复——
                    # "把 P1 详细说明"的 P1 是上一轮回复里提出的）。截断防巨轮撑爆内存。
                    "ref": (_d1_intent + "\n" + (turn.response_text or ""))[:4000],
                    "unit": _unit_key,
                }
            # ADR-0028 E7.1：文件内容索引（读写类工具的 result 正文 → 摘要 + 关键词）
            if flag_enabled("BLADEX_FILE_INDEX"):
                try:
                    self._index_files_from_turn(turn, mapped_uid)          # 第一类：读到的
                    self._index_produced_files_from_turn(turn, mapped_uid)  # 第二类：产出的
                except Exception as e:  # noqa: BLE001 —— 文件索引失败不影响主管线
                    logger.warning("index_file_index_failed", key=key_str, error=str(e))

            # U10（BLADEX_PROFILE_CARDS=1）：画像被动捕获 + 工具计数器（确定性聚合）
            if flag_enabled("BLADEX_PROFILE_CARDS"):  # 默认开，ADR-0027 §5.4
                try:
                    self.update_profiles_from_turn(turn, mapped_uid)
                except Exception as e:  # noqa: BLE001 —— 画像失败不影响主管线
                    logger.warning("index_profile_update_failed", key=key_str, error=str(e))
            new_consumed_keys.append(key_str)

            # ADR-0020 后续：限流--每轮 rebuild 最多处理 max_turns 个非 aux 轮，
            # 避免积压一次性密集调用 LLM 触发上游限流（553 轮阻塞 1.8h 教训）。
            # 剩余新轮下轮 rebuild 处理（consolidator 60s 一轮）。
            if max_turns > 0 and len(turns) >= max_turns:
                backlog = True  # ③ 还有积压：调用方应立即续跑，不要 sleep(interval)
                logger.info("index_rebuild_max_turns_reached",
                            max_turns=max_turns, consumed_so_far=len(new_consumed_keys))
                break

        # 无新非 auxiliary 轮次：标记 auxiliary 已消费后返回
        if not turns:
            # clear_db=True 且无可处理 turn：Memory Index 应与 Memory Hub 一致（清空）
            # 带过滤的 full 不清库——过滤掉的数据不该被静默丢弃
            if clear_db:
                self.clear()
                self._replay_admin_events(ledger)
                logger.info("index_full_rebuild_cleared_no_turns")
            if new_consumed_keys:
                for k in new_consumed_keys:
                    self._mark_consumed(k)
                logger.info("index_rebuild_only_auxiliary",
                            consumed=len(new_consumed_keys),
                            skipped_auxiliary=skipped_auxiliary,
                            skipped_filtered=skipped_filtered)
            else:
                logger.info("index_rebuild_no_new_turns",
                            skipped_filtered=skipped_filtered)
            # T8b B17：索引维护与"这轮有没有新 turn"解耦——空闲期也要够得着。
            try:
                self.maintain_ann_indexes()
            except Exception as e:  # noqa: BLE001 -- 索引问题不该影响 rebuild 语义
                logger.warning("index_ann_maintain_error", error=str(e))
            # ADR-0028 E2.4：生命周期重算（"睡眠巩固"）——空闲轮做，自带 6h 节流。
            # 与 ANN 维护同址同理由：它取决于**时间过去了多久**，不取决于这 60 秒
            # 里有没有人说话。失败只告警，绝不影响 rebuild 语义。
            try:
                self.recompute_lifecycle()
            except Exception as e:  # noqa: BLE001
                logger.warning("index_lifecycle_recompute_error", error=str(e))
            # ADR-0028 E7.2：Project 识别与字段重算（确定性，空闲轮做）
            try:
                self.maintain_projects()
            except Exception as e:  # noqa: BLE001
                logger.warning("index_project_maintain_error", error=str(e))
            # 复核 §3.4：matters 向量表存量压缩（冗余明显时才做，避免每轮重写表）
            try:
                self.maybe_compact_matter_vectors()
            except Exception as e:  # noqa: BLE001
                logger.warning("index_matter_compact_error", error=str(e))
            # M0-7（I3/H6/G6）：僵尸向量对账 + facts 向量表压缩。
            # 与上面两项同址同节流理由——它们都是"库的卫生"，取决于库积累了多久，
            # 不取决于这 60 秒有没有新流量。僵尸向量会白占 top-k 召回位（实测 14%）。
            try:
                self.reconcile_vector_meta()
            except Exception as e:  # noqa: BLE001
                logger.warning("index_vector_reconcile_error", error=str(e))
            try:
                self.maybe_compact_fact_vectors()
            except Exception as e:  # noqa: BLE001
                logger.warning("index_fact_compact_error", error=str(e))
            # MQ-I15 ②：三表版本 vacuum（磁盘轴；碎片轴管不到 files/matters）
            try:
                self.maybe_vacuum_lance_versions()
            except Exception as e:  # noqa: BLE001
                logger.warning("index_version_vacuum_error", error=str(e))
            # MS-8：库存健康度刷新（与卫生作业同址——都取决于库积累了多久）
            try:
                self.publish_library_kpis()
            except Exception as e:  # noqa: BLE001
                logger.warning("index_library_kpi_error", error=str(e))
            # ADR-0028 E6.2 配套：存量 fact 补进词法索引（缺了才补，补齐即空转）
            try:
                self.backfill_fts()
            except Exception as e:  # noqa: BLE001
                logger.warning("index_fts_backfill_error", error=str(e))
            # ADR-0028 E7.3：生成 USER.md / per-agent 习惯文件。
            # 🔴 2026-08-05 实测补接线：方法写好了但**没有调用点**，导致
            # `profile/user_md` 恒为空 → ①平面画像卡永远回落 v1 渲染
            # （那正是 E7.3 要取代的"路径串联乱码"）。又一例"机制写完但从不触发"。
            try:
                self.refresh_profile_docs()
            except Exception as e:  # noqa: BLE001
                logger.warning("index_profile_docs_error", error=str(e))
            return 0

        # Consolidate
        logger.info("index_rebuild_consolidating",
                    turns=len(turns), skipped_auxiliary=skipped_auxiliary,
                    existing_facts=len(existing))
        distill_before = _distiller_counters(consolidator)
        adj_before = _adjudicator_counters(self._adjudicator)
        try:
            new_facts = consolidator.consolidate_turns(turns, existing_facts=existing)
        except Exception as e:
            logger.warning("index_rebuild_consolidation_failed", error=str(e))
            # 不标记消费 → 下轮重试；full=True 时旧数据未清，完整保留。
            # backlog 保持 False：失败要退避，不能立即空转重试。
            return 0

        # 蒸馏中断守卫（2026-08-05 事故驱动）：蒸馏失败是**降级不抛**的——
        # `LLMDistiller.distill` 逮到异常就返回空 DistillOutput，于是"上游断了"
        # 与"这批本来就没事实"在本函数看来完全一样，而下面会无条件把这批 key
        # 标记成已消费。上游 DNS 断掉那次的实测形态：连续多轮
        # `consumed=30 new_facts=0`，30 turn/分钟把积压全部烧成零事实，只能靠人工
        # `--unconsume` 捞回来，日志里没有任何一行提示发生了这件事。
        # 所以这里按本轮失败率决定：超阈值 = 判定为上游中断，**不消费、不清库**，
        # 退避到下轮重试（台账不缓存失败，重试是零成本的）。
        delta = _distill_delta(distill_before, _distiller_counters(consolidator))
        if delta is not None and delta[1] > 0:
            calls, fails = delta
            ratio = fails / calls if calls else 0.0
            threshold = _distill_outage_abort_ratio()
            # MS-1：本批失败的 turn，按"上游活着没有"分成两拨（详见
            # `ProxyConsolidator._record_distill_failure`）。
            _failed_keys = set(getattr(consolidator, "failed_ledger_keys", None) or set())
            _parse_failed = set(
                getattr(consolidator, "parse_failed_ledger_keys", None) or set())
            _retry_max = _distill_retry_max()

            def _tick_retries(keys: set[str]) -> tuple[set[str], set[str]]:
                """给这些 turn 记一次重试，返回 (缓期集合, 超限集合)。"""
                defer, exhausted = set(), set()
                for k in keys:
                    if _retry_max <= 0 or self._bump_distill_retry(k) > _retry_max:
                        exhausted.add(k)
                    else:
                        defer.add(k)
                return defer, exhausted

            if threshold > 0 and ratio >= threshold:
                # 整批中断：本轮不写事实、不消费。
                # **但队列不许因此永久停摆**（2026-08-06 用户拍板）——
                # 若失败全是 `parse`（上游明明活着，是这些输入本身用不了），
                # 照样记重试次数，超限的那些直接放行消费，让积压能往前走。
                # `call` 失败（连不上/超时/配额）一次都不记：那是上游的问题，
                # 记了就等于断线 N 分钟后开始把好数据当坏数据丢掉（2026-08-05 事故）。
                _defer, _exhausted = _tick_retries(_parse_failed)
                if _exhausted:
                    for _k in sorted(_exhausted):
                        self._mark_consumed(_k)
                        self._clear_distill_retry(_k)
                        logger.warning(
                            "index_rebuild_distill_retry_exhausted",
                            ledger_key=_k, retry_max=_retry_max, branch="outage",
                            hint="upstream responded but this turn's output was never "
                                 "usable; consuming it so the backlog can move on")
                logger.warning(
                    "index_rebuild_distill_outage",
                    calls=calls, fails=fails, fail_ratio=round(ratio, 3),
                    threshold=threshold, turns=len(turns), consumed=0,
                    hint="most distill calls produced nothing (upstream unreachable, out of "
                         "quota, or unparseable output); this pass consumed nothing and wrote "
                         "nothing -- it will be retried. Check upstream connectivity with "
                         "`bladex doctor`, and grep the log for distill_failed / "
                         "distill_parse_failed to tell the two apart. "
                         "Disable this guard with BLADEX_DISTILL_OUTAGE_ABORT_RATIO=0",
                )
                # backlog 保持 False：中断期要退避，不能立即空转重试
                #（空转正是把积压烧光的加速器）。
                return 0
            # 未达阈值 = 零星失败。MS-1（复核 D4）：**失败的那些 turn 不标记消费**，
            # 下轮重试。台账不缓存失败，所以重试是零成本的；此前它们照常被消费，
            # 那一轮的事实就永久丢了（"降级不停摆"降的是可用性，不该降数据）。
            #
            # 这个分支里**两种失败都计次**：批里既然有调用成功，上游就是活着的，
            # 那么无论 call 还是 parse 失败都可以归到这条 turn 头上。
            # 超过 BLADEX_DISTILL_RETRY_MAX 的照常消费，队列不因个别坏轮停摆。
            _defer, _exhausted = _tick_retries(_failed_keys)
            if _defer:
                new_consumed_keys = [k for k in new_consumed_keys if k not in _defer]
            for _k in _exhausted:
                self._clear_distill_retry(_k)
            logger.warning(
                "index_rebuild_distill_partial_failure",
                calls=calls, fails=fails, fail_ratio=round(ratio, 3),
                threshold=threshold,
                deferred_turns=len(_defer), retry_exhausted=len(_exhausted),
                parse_failed=len(_parse_failed), retry_max=_retry_max,
                hint="some distill calls failed or returned unparseable output. Those turns "
                     "are NOT marked consumed and will be retried next pass (failures are "
                     "never cached, so retrying costs nothing). Turns that exceeded "
                     "BLADEX_DISTILL_RETRY_MAX are consumed anyway -- grep "
                     "index_rebuild_distill_retry_exhausted for them.",
            )
            for _k in sorted(_exhausted):
                logger.warning("index_rebuild_distill_retry_exhausted",
                               ledger_key=_k, retry_max=_retry_max, branch="partial")

        # ── 裁决中断守卫（MQ-S32 病 B，2026-08-31）─────────────────────────
        # 蒸馏守卫的孪生兄弟，理由一字不差：**失败是降级不抛的**。
        # 蒸馏失败返回空结果，裁决失败返回一整块 ADD——后者更隐蔽，因为
        # ADD 是个合法判决，下游看不出这一批根本没被判过，消费标记照打，
        # 当场制造同键并存（08-31 全库读数 `adjudicator_error` 46 条）。
        # 决策在 core（`adjudicate_outage`），这里只落盘 —— 与 supersede_plan 同分工。
        from bladex_core.adjudication import adjudicate_outage
        _abort, _items, _degraded, _ratio = adjudicate_outage(
            adj_before, _adjudicator_counters(self._adjudicator),
            flag_number("BLADEX_ADJUDICATE_OUTAGE_ABORT_RATIO"),
        )
        if _abort:
            logger.warning(
                "index_rebuild_adjudicate_outage",
                items=_items, degraded=_degraded, degraded_ratio=round(_ratio, 3),
                threshold=flag_number("BLADEX_ADJUDICATE_OUTAGE_ABORT_RATIO"),
                turns=len(turns), consumed=0,
                hint="most adjudication candidates were degraded to ADD (upstream "
                     "unreachable, out of quota, or unparseable verdict JSON). A blanket "
                     "ADD looks exactly like a real verdict downstream, so this pass "
                     "consumed nothing and wrote nothing -- it will be retried. Check "
                     "upstream with `bladex doctor`, and grep for adjudicate_failed / "
                     "adjudicate_parse_failed to tell the two apart. Disable this guard "
                     "with BLADEX_ADJUDICATE_OUTAGE_ABORT_RATIO=0",
            )
            # 与蒸馏中断同一处置：不写、不消费、退避（backlog 保持 False）。
            return 0
        if _degraded:
            # 未达阈值 = 零星降级。**不静默**：每一条降级都意味着这条候选没被真判过，
            # 它与同键兄弟的取代关系就此错过（重建也不会补——判决只剩审计价值，
            # 见 MQ-S32 第五环）。这里只报数，不改消费语义（蒸馏侧那套 per-turn
            # 重试计数在裁决侧没有对应物：降级是**按候选**发生的，不按 turn）。
            logger.warning("index_rebuild_adjudicate_partial_degrade",
                           items=_items, degraded=_degraded,
                           degraded_ratio=round(_ratio, 3),
                           hint="these candidates were written as ADD without ever being "
                                "judged; same-key siblings will coexist (MQ-S32 病 B)")

        # Beta T11（ADR-0012 §3.6）：FACT 级墓碑——重建不得复活被 forget 的 Fact。
        # fact.id 确定性（sha(ledger_key+content)），跨 rebuild 稳定 → 按 id 过滤即可。
        try:
            fact_tombs = {
                t.target_key for _, t in ledger.scan_tombstones()
                if str(getattr(t.target_type, "value", t.target_type)) == "fact"
            }
        except Exception as e:  # noqa: BLE001 —— 墓碑读取失败不中断重建
            logger.warning("index_rebuild_fact_tombstone_scan_failed", error=str(e))
            fact_tombs = set()
        if fact_tombs:
            before = len(new_facts)
            new_facts = [f for f in new_facts if f.id not in fact_tombs]
            if len(new_facts) != before:
                logger.info("index_rebuild_fact_tombstones_filtered",
                            filtered=before - len(new_facts))

        # M1-2 文档路（拍板 3 / 附录 D.4）：分流判为「粘贴物/文档」的 user 消息
        # 不产 assertion/preference，改建**文件内容索引**（summary + keywords）。
        # core 只排队（agent 中立，不知道 LanceDB/files 表的存在），proxy 落盘——
        # 与 `supersede_plan` 同一个模式。失败只告警：文档索引不该拖垮主管线。
        for _doc in (getattr(consolidator, "document_entries", None) or []):
            try:
                self._index_pasted_document(_doc)
            except Exception as e:  # noqa: BLE001
                logger.warning("index_document_failed",
                               path=_doc.get("path", ""), error=str(e))

        # 安全重建：consolidation 成功后才清空旧 Memory Index（review 20260707）
        if clear_db:
            self.clear()
            self._replay_admin_events(ledger)
            logger.info("index_full_rebuild_cleared_after_consolidation",
                        new_facts=len(new_facts))

        # 存入 Memory Index + 五层链接归属（ADR-0018 §3.2，异步管线）
        # （T10 R5 跨批矛盾校验 `_check_consistency_vector` 已于 2026-09-03 S6 删除：
        #  返回值被丢弃、零消费者，每条新 fact 白付一次向量检索；MQ-S51 改判机制退役。）

        # 裁决台账：ADR-0020 T1.3 迁 Memory Index meta（consolidator 独占写，写入不失败；
        # 旧 Memory Hub 实现因 read_only Memory Hub 写失败 100%）。无 link_judge 则无 L4、无台账。
        judgment_journal: JudgmentJournalProtocol | None = None
        if self._link_judge is not None:
            from bladex_proxy.linking import IndexJudgmentJournal
            judgment_journal = IndexJudgmentJournal(
                self, judge_model=getattr(self._link_judge, "model_name", ""),
            )

        pipeline = AttributionPipeline(
            link_judge=self._link_judge,
            judgment_journal=judgment_journal,
            top_candidates=self._top_candidates,
            # 台账缓存判决的目标存在性谓词（孤边根因，任务卡 T3）。
            # 只在这一处注入：core 侧默认 None = 不校验，其它调用方零回归。
            matter_exists=lambda mid: self.get_matter(mid) is not None,
        )
        top_k = pipeline.top_candidates

        # 无 LanceDB 时缓存全量 Matter 列表一次（避免每 fact 全表重读）
        base_matters = (
            [m for m in self.all_matters() if m.matter_id != UNASSIGNED_MATTER_ID]
            if self._matter_table is None else []
        )

        # 先存所有 fact（metadata + LanceDB）
        added_facts: list[Fact] = []
        for fact in new_facts:
            try:
                self.add_fact(fact)
                added_facts.append(fact)
            except Exception as e:
                logger.warning("index_fact_add_failed", fact_id=fact.id, error=str(e))

        # ── 重放取代判决（MQ-S32 病 B 第五环，2026-08-31）───────────────────
        # 位置有讲究：必须在 add_fact 之后（targets 要先存在）、在本轮裁决应用
        # 之前（本轮新判决应当能覆盖历史判决）。**只在全量重建时跑**——增量 pass
        # 库里本就带着取代关系，重放等于重复应用（与 `_replay_admin_events` 同理由）。
        if clear_db and flag_enabled("BLADEX_SUPERSEDE_REPLAY_ENABLED"):
            try:
                self._replay_supersede_judgments({f.id for f in added_facts})
            except Exception as e:  # noqa: BLE001 —— 重放失败不该拖垮重建
                logger.warning("index_supersede_replay_failed", error=str(e))

        # ADR-0026 §5 机制2/3 / U5.2+U5.3：应用取代/合并计划（gated —— consolidator 未开
        # BLADEX_SUPERSEDE_ENABLED 时 supersede_plan=None，此块跳过=零回归）。
        # 旧条的 strength / t_invalid+superseded_by 已由 planner 改在对象上，这里 upsert 其 meta。
        # ── M2-2：应用裁决产生的旧条改动 + 写 judgment 台账（kind=consolidation）──
        # core 决策、proxy 落盘（与 supersede_plan 同一分工）。
        # 台账让 rebuild 重放**零 LLM**：同一批候选 + 同一组邻居 → 直接读结论。
        for _rec in (getattr(consolidator, "adjudication_records", None) or []):
            for _old in _rec.get("changed") or []:
                _old.embedding = None      # 只 upsert meta，不重复写向量
                try:
                    self.add_fact(_old)
                except Exception as e:  # noqa: BLE001
                    logger.warning("index_adjudicate_apply_failed",
                                   fact_id=_old.id, error=str(e))
            try:
                self.append_judgment(
                    _rec["candidate_id"],
                    [{"fact_id": t} for t in _rec.get("targets", [])],
                    f"consolidation:{_rec['op']}",
                    getattr(self._adjudicator, "model_name", ""),
                    reason=_rec.get("reason", ""),
                    considered=_rec.get("considered") or [],
                )
            except Exception as e:  # noqa: BLE001 —— 台账写失败不该丢已应用的裁决
                logger.warning("index_adjudicate_journal_failed",
                               fact_id=_rec.get("candidate_id", ""), error=str(e))
        if getattr(consolidator, "adjudication_records", None):
            from collections import Counter as _AC
            logger.info("index_adjudicate_applied",
                        ops=dict(_AC(r["op"] for r in consolidator.adjudication_records)))

        _plan = getattr(consolidator, "supersede_plan", None)
        if _plan is not None and (_plan.bump or _plan.invalidate):
            for old_fact, _s in _plan.bump:
                old_fact.embedding = None  # 只更新 meta，不重复写向量
                try:
                    self.add_fact(old_fact)
                except Exception as e:
                    logger.warning("index_supersede_bump_failed", fact_id=old_fact.id, error=str(e))
            for old_fact, _nid in _plan.invalidate:
                old_fact.embedding = None
                try:
                    self.add_fact(old_fact)
                except Exception as e:
                    logger.warning("index_supersede_invalidate_failed", fact_id=old_fact.id, error=str(e))
            logger.info("index_supersede_applied", bumped=len(_plan.bump),
                        invalidated=len(_plan.invalidate))

        # 顺序归属（mid-batch 可见：本 pass 新建的 Matter 立即成为后续 fact 的候选，
        # 使 L4 在 full rebuild 也能跑 + judgment 台账可重放，ADR-0018 §4.1）。
        # L4 逐条裁决（attribute 内部 1-item batch）+ 台账缓存；首 pass 付 N 次调用，
        # 后续 rebuild 命中=0。批量能力（attribute_batch）保留供它用，rebuild 取顺序保可见性。
        #
        # （L0「同单元同属」顺序循环折入版（MQ-I1，2026-08-15）已于 2026-09-03 S4 连
        #  `BLADEX_ATTRIB_L0` / `ATTRIB_L0_SPAN_H` 一起删除：默认关、从未翻默认、零 live
        #  读数；MQ-V6 的准入门教训（unit_key 不是可信输入，98 轮撞同一 key）留在 git 历史。）
        from datetime import UTC
        from datetime import datetime as _dt

        from bladex_core.attribution import AttributionDecision
        now = _dt.now(UTC)
        new_matters_this_pass: list[Matter] = []
        attributed = 0
        # MQ-P14：归属逐条 catch-all 吞掉的异常数与类型分布（见循环尾部的
        # `index_attribution_failed`）。进汇总行是因为 `attributed=0` 本身
        # 无法区分"没东西可归"与"每一条都抛异常"。
        _attrib_failed = 0
        _attrib_failed_types: Counter[str] = Counter()
        # U4B-B2（ADR-0026 铁律3）：会话事件折叠进 Matter 卡 lifecycle。
        # 同 (matter, ledger_key, event) 只记一次（同轮多 fact 归同 Matter 不重复记）。
        _lifecycle_enabled = flag_enabled("BLADEX_MATTER_LIFECYCLE")  # ADR-0028 E2.1 默认开
        _lifecycle_recorded: set[tuple[str, str, str]] = set()
        _lifecycle_folded = 0
        # MS-5 ①：ledger_key -> {matter_id: 票数}。同轮多 Matter 时取主归属。
        _matter_votes: dict[str, Counter[str]] = defaultdict(Counter)
        # MS-4：ledger_key -> (event, detail)，蒸馏产出的任务事件分类。
        # 规则匹配（detect_session_events）降为**无蒸馏产物时的兜底**：
        # 蒸馏看得懂"把任务卡打回重做"是 reworked，而词表只能靠字面命中，
        # 且 detail 取的是 130 字符原文（H2 那个塞卡问题的来源）。
        _distilled_events: dict[str, tuple[str, str]] = dict(
            getattr(consolidator, "turn_events", None) or {})

        # ── GM-1 候选通路：ledger_key → 该轮 fact 落定的 matter_id ──
        # 本 pass 内边判边填；跨 pass 的上一轮（consolidator 60s 一批，延续对可能被批
        # 边界切开）走一次性懒加载的存量索引。
        _turn_matters: dict[str, set[str]] = defaultdict(set)
        _existing_turn_matters: dict[str, set[str]] | None = None
        # 三个数分开记，因为它们回答三个不同问题——合成一个数就分不清
        # "机制没触发"和"触发了但向量本来就召回了"（后者是好事，前者是接线断）。
        _cont_supplied = 0           # 延续通路**提出**的卡数（去重前）
        _cont_candidates_added = 0   # 去重后真进候选池的次数
        _cont_adopted = 0            # 最终判归到延续独有候选上的次数

        # ── GM-2 按键反查候选通路（MQ-S11 的召回半边）──
        # 只有向量一条召回通路，在话题稳定、语料充足时够用（同一张卡实测跨 6 session
        # / 3 agent 持续吸附成员）；它补的是**向量散开的那一段**——同一对象不同动作
        # （进度评估／进展评估／问题报告／风险复核）各轮内容差异大时向量会散。
        # 🔴 判别力门在 `key_based_matter_candidates` 里，不在这里；这里只负责接线。
        _key_enabled = flag_enabled("BLADEX_ATTRIB_KEY_RECALL")
        _key_inverted: dict[str, set[str]] = {}
        _key_total_matters = 0
        if _key_enabled:
            try:
                _key_inverted, _key_total_matters = self.build_matter_key_index()
            except Exception as e:  # noqa: BLE001 —— 候选通路失败不得影响归属本体
                logger.warning("index_attrib_key_index_failed", error=str(e))
                _key_enabled = False
        _key_candidates_added = 0
        _key_adopted = 0

        def _make_room(cands: list[Matter], vec_n: int) -> int:
            """给一个确定性候选腾位子：只裁**向量段的末位**，返回新的向量段长度。

            🔴 不能裁列表尾部——尾部是 `new_matters_this_pass`，裁它等于把
            mid-batch 可见性（ADR-0018 §4.1，让 L4 在全量重建也看得见本 pass 刚建的卡）
            悄悄削掉。向量段空了就宁可略微超预算：候选多一个的代价是一次裁决，
            丢掉 mid-batch 可见性的代价是重复开卡——正是本专项要治的病。
            """
            if len(cands) >= top_k and vec_n > 0:
                cands.pop(vec_n - 1)
                return vec_n - 1
            return vec_n

        def _prev_turn_matters(prev_key: str) -> set[str]:
            """上一轮 fact 所属的 Matter（本 pass 内优先，miss 才付存量扫描）。"""
            nonlocal _existing_turn_matters
            hit = _turn_matters.get(prev_key)
            if hit:
                return hit
            if _existing_turn_matters is None:
                _existing_turn_matters = defaultdict(set)
                for _f in self.all_facts():
                    _lk = getattr(_f, "source_ledger_key", "") or ""
                    _mid = getattr(_f, "matter_id", "") or ""
                    if _lk and _mid and _mid != UNASSIGNED_MATTER_ID:
                        _existing_turn_matters[_lk].add(_mid)
            return _existing_turn_matters.get(prev_key, set())

        _lane_split_enabled = flag_enabled("BLADEX_LANE_HARD_SPLIT")
        _lane_skipped = 0

        # ── G12.3：goal 捕获的懒取上下文（只在新开卡时付一次 ledger.get()）──
        # 本 pass 的计数清零：读数按 pass 报，不跨 pass 累加（否则 rebuild 一多，
        # "有多少卡是被机器文本开出来的"就变成一个只增不减、无参照系的数）。
        self._goal_counters.clear()
        _goal_ctx_cache: dict[str, tuple[str, str] | None] = {}

        def _goal_ctx(lk: str) -> tuple[str, str] | None:
            if lk not in _goal_ctx_cache:
                _goal_ctx_cache[lk] = _goal_context(ledger, lk)
            return _goal_ctx_cache[lk]

        # ── V-L6 账本锚层状态（ADR-0032 §4.5：归属主路径；D1/五层降兜底）──
        # 绑定与账本标题来自 Hub 管理事件重放（投影，重建等价）；
        # 锚定 Matter 的 id 由 ledger_id 确定性派生（重放同批事件必得同 id）。
        from bladex_core.flags import flag_enabled as _la_flag
        _la_enabled = _la_flag("BLADEX_LEDGER_ANCHOR")
        _la_bind: dict[str, str] = {}
        _la_timeline: dict[str, list[tuple[float, str]]] = {}
        _la_pool: dict[str, Any] = {}
        _la_titles: dict[str, str] = {}
        _la_matter_by_ledger: dict[str, str] = {}
        _la_created: set[str] = set()
        _la_counters: Counter[str] = Counter()
        if _la_enabled:
            try:
                from bladex_core.ledger import replay_ledger_events as _la_replay
                from bladex_core.ledger import switch_timeline as _la_tl

                # 🔴 2026-09-01：装载收进唯一入口 `load_ledger_events`。
                # 此前这里用**前缀匹配**自行判定账本事件，而 agency/flash_daemon
                # 用的是 core 的 `LEDGER_EVENT_TYPES`（两处注释都写着"各抄一份
                # 迟早分叉"）——重建路径就是那个没跟上的第三处。顺带接入账本墓碑。
                #
                # ⚠️ 别在注释里复现那个前缀匹配的字面量：守卫按字面量扫全树，
                # 一句解释会被它当成新的违规抓住（同日 env 名那次同型）。
                from bladex_proxy.ledger_events import load_ledger_events
                _la_events = load_ledger_events(ledger)
                _la_pool, _la_bind = _la_replay(_la_events)
                _la_timeline = _la_tl(_la_events)
                _la_titles = {lid: (l.title or lid) for lid, l in _la_pool.items()}
            except Exception as e:  # noqa: BLE001 —— 锚层装载失败=降级走 D1/五层
                logger.warning("ledger_anchor_load_failed", error=str(e))
                _la_enabled = False
            for _m0 in self.all_matters():
                _lid0 = getattr(_m0, "ledger_id", "") or ""
                if _lid0:
                    _la_matter_by_ledger[_lid0] = _m0.matter_id
            if _la_bind:
                logger.info("ledger_anchor_ready", bindings=len(_la_bind),
                            ledgers=len(_la_titles),
                            anchored_matters=len(_la_matter_by_ledger))

        def _la_anchor_aliases(lid: str) -> list[str]:
            """锚卡原生键 = 只有账本标题（2026-09-02 回退 Goal 派生，见 core 侧 docstring）。"""
            from bladex_core.ledger_runtime import ledger_anchor_aliases
            return ledger_anchor_aliases(_la_titles.get(lid, lid))

        def _la_ensure_matter(lid: str, *, proposed: str = "") -> str:
            """账本 → 锚定 Matter（无则建）。**建卡的唯一实现点。**

            预物化（`_la_premateralize`）与按需路径（`_la_matter_for`）都走这里：
            同一件事两条建卡路径就是刚性原则 12 说的那种缺陷，与哪条更合理无关。
            """
            from bladex_core.ledger_runtime import (
                anchor_decision,
                ledger_anchor_matter_id,
            )
            mid = proposed
            # 账本自己记的 matter_id 是**权威边界**（proxy 侧建账本时写入）。
            # index 侧算出的与它不一致 ⇒ 告警 + 用账本的，**绝不静默换绑/合并**
            # （ADR-0032 §4.5 "特别存疑可生成多个 Matter，但必须告警人工复核"；
            # 不一致的真实来源是人工 merge 把锚定 Matter 并走了）。
            _dec = anchor_decision(_la_pool.get(lid), proposed_matter_id=mid)
            if _dec.warn_multi:
                _la_counters["multi_matter"] += 1
                logger.warning("ledger_anchor_multi_matter", ledger=lid,
                               ledger_matter=_dec.matter_id, index_matter=mid,
                               action="kept_ledger_binding")
            # 权威顺序：账本记的 > index 现存的 > 确定性派生的
            mid = _dec.matter_id or mid or ledger_anchor_matter_id(lid)
            if self.get_matter(mid) is None:
                _aliases = _la_anchor_aliases(lid)
                _mt = Matter(matter_id=mid, title=_la_titles.get(lid, lid),
                             status=MatterStatus.ACTIVE, origin=MatterOrigin.AUTO,
                             ledger_id=lid, aliases=_aliases)
                self.add_matter(_mt)
                new_matters_this_pass.append(_mt)
                _la_created.add(mid)
                logger.info("ledger_anchor_matter_created", matter=mid, ledger=lid,
                            title=_la_titles.get(lid, lid), aliases=len(_aliases))
            _la_matter_by_ledger[lid] = mid
            return mid

        def _la_prematerialize() -> int:
            """把账本池里**所有**账本的锚卡在归属开始前建出来。

            🔴 为什么（任务卡 §0 缺口 1）：账本边界是**外部已知**的——池子与标题
            在 `ledger_anchor_ready` 那一刻（实测 17:59:54）就装载完毕，而首条
            fact 17:59:49 已经开始入库。惰性建卡让锚卡比同题内容卡晚 15.6 秒
            出生，那段空窗里 DPL 只能另开一张。数据在手却等第一条 fact 撞上，
            没有任何理由。

            幂等：`_la_ensure_matter` 内部 `get_matter is None` 才建，重复调用
            无副作用；id 仍由 `ledger_anchor_matter_id` 确定性派生 ⇒ 重建等价不变。
            """
            n = 0
            for lid in sorted(_la_pool):
                before = len(_la_created)
                _la_ensure_matter(lid, proposed=_la_matter_by_ledger.get(lid, ""))
                n += len(_la_created) - before
            return n

        def _la_ledger_at(lk: str) -> str:
            """这一轮**发生时**激活的账本（点时查找；MQ-L15）。

            🔴 键是 `activation_scope(agent, project)`，**不是 session**
            （MQ-L14：MQ-L11 把绑定键从 session 改成 scope，这个消费方没跟上，
            于是锚层对 100% 的轮次返回空、静默落回 D1，却照常打
            `ledger_anchor_ready` —— 一个报告自己就绪的死机制）。
            project 段在 V-F1 落地前恒 `Global`，与 switch 事件现状一致。
            """
            from bladex_core.ledger import ledger_active_at
            from bladex_core.ledger_runtime import (
                activation_scope,
                agent_of_turn_key,
                turn_key_ts_ms,
            )
            agent = agent_of_turn_key(lk)
            if not agent:
                return ""
            scope = activation_scope(agent, "")
            lid = ledger_active_at(_la_timeline, scope, turn_key_ts_ms(lk))
            if lid:
                return lid
            # 时间轴给不出（无 ts 的历史事件 / 该刻早于第一次切换）：
            # 只在**时间轴上这个 scope 压根没有条目**时退回终态绑定，
            # 否则会把"建账本之前"的历史塞进锚定 Matter（MQ-L15 反向形状）。
            if scope in _la_timeline:
                return ""
            return _la_bind.get(scope, "")

        _lg_enabled = flag_enabled("BLADEX_LEDGER_GATE")

        def _lg_reject(lk: str, matter: Matter | None) -> str:
            """DPL 边落**账本锚卡**的准入。放行返回 ""，否则是可 grep 的原因。

            ADR-0032 §4.5 归属准入顺序的前两段（账本标定 → 时间 → 内容聚焦）。
            判定本体是纯函数 `ledger_runtime.ledger_gate`；这里只做取值。

            🔴 **复用 `_la_ledger_at`，不另起一套**（Jason 2026-09-02：
            "重建应该是回放 live 的过程，不能另起一套机制"）。锚层判"这一轮
            属于哪本账本"已经有唯一实现，DPL 侧再写一个同义判据就是 MQ-A18/P7
            那种"同一件事在两个地方各做一次、中间无对账"的形状。

            对非锚卡（`ledger_id` 空）恒放行 ⇒ 普通内容卡零回归。
            """
            if not _lg_enabled or matter is None:
                return ""
            _tgt = getattr(matter, "ledger_id", "") or ""
            if not _tgt:
                return ""
            from bladex_core.ledger_runtime import (
                iso_ms,
                ledger_gate,
                turn_key_ts_ms,
            )
            _tl = _la_pool.get(_tgt)
            r = ledger_gate(
                turn_ledger_id=_la_ledger_at(lk),
                turn_ms=turn_key_ts_ms(lk),
                target_ledger_id=_tgt,
                target_created_ms=iso_ms(getattr(_tl, "created_at", "") if _tl else ""),
            )
            if r:
                _la_counters[f"gate_{r}"] += 1
            return r

        # 第三个闸点：**L4 裁决目标**（台账缓存命中 / 现场裁决都走它）。
        # 与 CONT 直挂同型 —— 它也绕过候选池：闸把锚卡从候选里摘掉，
        # 缓存判决又照着 `result.matter_id` 写了回来（第五次全量重建实测漏 1 条）。
        # 后置赋值而非构造参数：pipeline 在 7119 行就构造了，那时账本池与
        # 时间轴还没装载（`_lg_reject` 依赖它们）。语义见 `target_admissible`。
        if _lg_enabled:
            pipeline.target_admissible = (
                lambda _f, _mid: _lg_reject(
                    getattr(_f, "source_ledger_key", "") or "",
                    self.get_matter(_mid)))

        def _la_matter_for(lk: str) -> str:
            """turn key → 点时激活账本 → 锚定 Matter（无则确定性新建）。

            🔴 **MQ-L26**：这一轮明确**没拿到账本面**时不锚（返回 ""）。

            账本面 gating 与本函数此前用两套判据、中间无对账：一轮没拿到账本面
            （模型看不到账本、也没有 `bladex_ledger_switch` 工具），它的产出照样
            被锚到当时激活的旧账本上。live 病例见
            `docs/reviews/t5-matter-signoff-20260828.md` §2.1——`[3]` 那张卡混入
            10 条 BladeX 全局状态，全是零工具「建议生成」轮（`reason=aux`）。

            **锚定一条模型从来没机会表态的产出，等于替它做了归属判断**，
            直接违反 ADR-0032 的作者边界。不锚 ⇒ 落回 DPL 五层兜底
            （provisional 留池、≥2 成员才转正），比"无条件挂到激活账本"保守得多。

            `None`（历史轮无此字段）按**旧行为**锚定：缺数不等于"明确没给"，
            否则全量重建会让历史锚定一次性失效（ADR-0012 §3.6）。
            """
            if ledger_key_to_face.get(lk) is False:
                _la_counters["no_ledger_face"] += 1
                return ""
            lid = _la_ledger_at(lk)
            if not lid:
                return ""
            mid = _la_matter_by_ledger.get(lid, "")
            if mid:
                _m = self.get_matter(mid)
                if not (_m is not None and _m.status in (MatterStatus.ACTIVE,
                                                         MatterStatus.PROVISIONAL)):
                    mid = ""
            return _la_ensure_matter(lid, proposed=mid)

        # 🔴 预物化：归属开始**之前**把所有账本锚卡建出来（任务卡 T1）。
        # 放在这里而不是更早，是因为它依赖上面几个闭包；放在这里而不是更晚，
        # 是因为下面的 D1 与主归属循环从这一刻起就要能把锚卡当候选看见。
        if _la_enabled and _la_pool:
            _la_pre_n = _la_prematerialize()
            logger.info("ledger_anchor_prematerialized",
                        ledgers=len(_la_pool), created=_la_pre_n,
                        already_present=len(_la_pool) - _la_pre_n)

        # ── G12.2 D1 状态构建（决策表 v2；红线 7 论证见 task_state.py 头注）──
        _d1_active = _d1_enabled and bool(_d1_scan)
        if _d1_active:
            from bladex_core.attribution import new_matter_id as _d1_new_id
            from bladex_core.task_state import (
                CONT_LAYER,
                D1_KNOB_VER,
                ActiveWindow,
                AnchorTracker,
                decide_turn,
                unique_unit_hit,
            )

            def _d1_ms(lk: str) -> int:
                # 时间轴 = ledger key 尾段 stream ms（created_at 会被重建压扁）
                tail = (lk or "").rsplit("/", 1)[-1].split("-", 1)[0]
                return int(tail) if tail.isdigit() else 0

            _d1_anchors = AnchorTracker()
            _d1_window = ActiveWindow()
            _d1_unit_index: dict[str, set[str]] = {}
            _d1_hist_votes: dict[str, Counter[str]] = defaultdict(Counter)
            # 存量库一遍扫：窗口活跃时钟 + 跨批次锚的轮→卡投票 + unit 倒排。
            # 成本与 GM-1 的 `_existing_turn_matters` 懒加载同级（O(all_facts) 一次）。
            # 🔴 只在**增量**模式做：full 重建时存量 = 即将被清的旧世界，读它当种子
            # 会把旧 matter_id 引进锚/窗口/unit 倒排（08-20 离线对照实测
            # `R4_missing=17` 就是被 get_matter 拦下的那部分）。full 从头按序重放，
            # in-pass 状态（finalize 逐轮积累）本来就是完整的。
            for _f0 in ([] if full else self.all_facts()):
                _lk0 = getattr(_f0, "source_ledger_key", "") or ""
                _mid0 = getattr(_f0, "matter_id", "") or ""
                if not _lk0 or not _mid0 or _mid0 == UNASSIGNED_MATTER_ID:
                    continue
                _d1_hist_votes[_lk0][_mid0] += 1
                _d1_window.touch(_mid0, _d1_ms(_lk0))
                _uk0 = getattr(_f0, "unit_key", "") or ""
                if _uk0:
                    _d1_unit_index.setdefault(_uk0, set()).add(_mid0)
            _d1_decisions: dict[str, dict | None] = {}
            _d1_r5_pending: dict[str, dict] = {}
            _d1_counters: Counter[str] = Counter()

            def _d1_matter_alive(mid: str) -> bool:
                _m = self.get_matter(mid)
                return _m is not None and _m.status in ("active", "provisional")

            def _d1_turn_votes(lk: str) -> Counter:
                c: Counter[str] = Counter()
                c.update(_d1_hist_votes.get(lk) or {})
                c.update(_matter_votes.get(lk) or {})
                return c

            def _d1_anchor_of(lk: str) -> tuple[str, str]:
                """跨批次锚：沿 _prev_of 链回走、跳过 aux 轮（≤5 步）。"""
                pk = _prev_of.get(lk, "")
                steps = 0
                while pk and steps < 5:
                    _rt, _raux = _prev_ref(pk, _prev_ref_cache, ledger)
                    if not _raux:
                        v = _d1_turn_votes(pk)
                        return (v.most_common(1)[0][0] if v else ""), _rt
                    pk = _prev_of.get(pk, "")
                    steps += 1
                return "", ""

            def _d1_decide(lk: str) -> dict | None:
                info = _d1_scan.get(lk)
                if info is None:
                    return None
                # 🔴 台账优先（ADR-0031 §14.1）：continue_to 记录直接照办（防旋钮
                # 标定后重放改判）；new/degraded_new 记录落回管线——L5 的 id 由
                # 同一条 ledger key 派生（G12.1），管线重放天然复现同 id。
                rec = self.get_taskstate_judgment(lk)
                if rec is not None:
                    if (rec.verdict == "continue_to" and rec.matter_id
                            and self.get_matter(rec.matter_id) is not None):
                        return {"kind": "direct", "matter_id": rec.matter_id,
                                "branch": "LEDGER", "anchor": rec.anchor_matter_id,
                                "window": list(rec.window_snapshot)}
                    return {"kind": "fallthrough", "branch": "LEDGER_NEW"}
                anchor_mid, anchor_ref = _d1_anchors.get(info["agent"], info["session"])
                if not anchor_mid:
                    anchor_mid, anchor_ref = _d1_anchor_of(lk)
                alive = bool(anchor_mid) and _d1_matter_alive(anchor_mid)
                snap = _d1_window.snapshot()
                d = decide_turn(
                    anchor_matter_id=anchor_mid, anchor_alive=alive,
                    anchor_reference_text=anchor_ref, window=snap,
                    session_id_source=info["src"], intent_text=info["intent"],
                    unit_hit_matter=unique_unit_hit(info["unit"], _d1_unit_index),
                    auxiliary=info["aux"], new_subject_max=_cont_new_subject_max)
                if d.verdict == "continue_to":
                    if self.get_matter(d.matter_id) is not None:
                        return {"kind": "direct", "matter_id": d.matter_id,
                                "branch": d.branch, "anchor": anchor_mid, "window": snap}
                    logger.warning("d1_target_missing", key=lk, matter_id=d.matter_id)
                    return {"kind": "fallthrough", "branch": d.branch + "_missing"}
                if d.branch == "AUX":
                    return {"kind": "fallthrough", "branch": "AUX"}
                # R5：本轮 facts 走既有五层管线兜底（ADR-0018 §12），落定后写台账
                _d1_r5_pending[lk] = {"anchor": anchor_mid, "alive": alive,
                                      "window": snap, "signals": dict(d.signals)}
                return {"kind": "fallthrough", "branch": "R5"}

            def _d1_finalize_turn(lk: str) -> None:
                """轮收尾：更新锚/窗口/unit 倒排；R5 轮写台账（否定判决也记——
                只记 new 的话旋钮一标定重放就漂首轮集合，决策表 v2 §6）。"""
                info = _d1_scan.get(lk)
                if info is None:
                    return
                votes = _d1_turn_votes(lk)
                mid = votes.most_common(1)[0][0] if votes else ""
                if mid:
                    _d1_anchors.update(info["agent"], info["session"], mid,
                                       info["ref"], auxiliary=info["aux"])
                    _d1_window.touch(mid, _d1_ms(lk), auxiliary=info["aux"])
                    if info["unit"]:
                        _d1_unit_index.setdefault(info["unit"], set()).add(mid)
                pend = _d1_r5_pending.pop(lk, None)
                if pend is not None and mid:
                    is_new = (mid == _d1_new_id(ledger_key=lk))
                    try:
                        self.append_taskstate_judgment(TaskStateJudgment(
                            turn_key=lk, unit_key=info["unit"],
                            verdict="new" if is_new else "continue_to",
                            matter_id=mid,
                            anchor_matter_id=pend["anchor"],
                            anchor_alive=pend["alive"],
                            window_snapshot=pend["window"],
                            signals=pend["signals"],
                            judge_model=getattr(self._link_judge, "model_name", "") or "",
                            knob_ver=D1_KNOB_VER))
                        _d1_counters["ledger_written"] += 1
                    except Exception as e:  # noqa: BLE001 —— 台账写失败不阻塞归属
                        logger.warning("d1_ledger_write_failed", key=lk, error=str(e))

        _d1_cur_turn = ""

        for fact in added_facts:
            try:
                manual_edge = self.get_manual_edge_for_target(fact.id, EdgeTargetType.FACT)
                manual_matter_id = manual_edge.matter_id if manual_edge else None

                # ── MQ-S27 硬分流（2026-08-20 拍板；G12.2 决策表 v2 §7 前置边界）──
                # lane:profile / lane:tool（含 PROFILE_OBS kind）不进 Matter 归属：
                # 不建边、matter_id 恒空、不参与本轮投票与延续链——profile 走画像
                # 原料通道，tool 类留 KB 检索。manual 手动映射压过本分流（红线 6：
                # 用户显式指定的归属永远算数）。
                # 🔴 仪器口径连带：未归属池读数（MQ-I7）此后必须把 lane 分流单列，
                # 否则会把这批读成"归属管线变差了"（仪器参照系教训）。
                if (_lane_split_enabled and manual_edge is None
                        and fact_lane(fact.tags, getattr(fact, "item_kind", ""))
                        in ("profile", "tool")):
                    _lane_skipped += 1
                    continue

                # ── V-L6 账本锚层（ADR-0032 §4.5：主路径，压过 D1 与五层；
                # manual 仍最高——用户显式指定永远算数）──
                if _la_enabled and manual_edge is None:
                    _la_mid = _la_matter_for(fact.source_ledger_key)
                    if _la_mid:
                        decision = AttributionDecision(
                            fact_id=fact.id, matter_id=_la_mid,
                            confidence=0.95,
                            source=AttributionSource.LEDGER_ANCHOR,
                            is_new_matter=_la_mid in _la_created,
                            signal="ledger_anchor",
                            decision={"layer": "ANCHOR", "branch": "LEDGER",
                                      "ts": now.isoformat()},
                        )
                        self._apply_decision(fact, decision, pipeline, now,
                                             goal_ctx=_goal_ctx)
                        _turn_matters[fact.source_ledger_key].add(_la_mid)
                        _matter_votes[fact.source_ledger_key][_la_mid] += 1
                        _la_counters["anchored"] += 1
                        attributed += 1
                        _prog(phase="attribute", done=attributed,
                              total=len(added_facts))
                        continue

                # ── G12.2 D1 直挂（R1/R2/R4 零 LLM；R5/AUX/台账-new 落回下方管线）──
                # 红线 7：延续是时序局部的、合并是全局的——"默认延续"不违反
                # 误合并=0 红线；不要把这里改回"默认走管线"，那是 80% 碎片率的来源。
                if _d1_active:
                    _lk_f = fact.source_ledger_key
                    if _lk_f != _d1_cur_turn:
                        if _d1_cur_turn:
                            _d1_finalize_turn(_d1_cur_turn)  # 先收上一轮，锚才是新的
                        _d1_cur_turn = _lk_f
                    if _lk_f not in _d1_decisions:
                        _d1_decisions[_lk_f] = _d1_decide(_lk_f)
                        _d1d0 = _d1_decisions[_lk_f]
                        if _d1d0 is not None:
                            _d1_counters[_d1d0["branch"]] += 1
                    _d1d = _d1_decisions[_lk_f]
                    # 账本准入闸（ADR-0032 §4.5）：CONT 直挂**绕过候选池**，
                    # 所以必须在这里单独过闸，不能只过滤候选。拒绝 ⇒ 不 continue，
                    # 落回下方五层管线（那里的候选已过同一道闸）。
                    #
                    # 🔴 只在 `kind == "direct"` 分支取 `matter_id`：`_d1_decide`
                    # 的 fallthrough 变体（LEDGER_NEW / *_missing / AUX / R5）
                    # **没有这个键**。首版无条件读它 ⇒ KeyError 被
                    # `index_attribution_failed` 兜成一行 warning，归属整链空转，
                    # 11 个不相干的用例集体转红而栈全被吞掉
                    # （"失败长得像正常"；判据=先从 `_d1_decide` 的 return 读全变体）。
                    _lg_cont = ""
                    if _d1d is not None and _d1d.get("kind") == "direct":
                        _lg_mid = _d1d.get("matter_id", "")
                        _lg_cont = _lg_reject(_lk_f, self.get_matter(_lg_mid))
                        if _lg_cont:
                            _la_counters["gate_cont_blocked"] += 1
                            logger.debug("index_ledger_gate_cont", fact_id=fact.id,
                                         matter=_lg_mid, reason=_lg_cont)
                    if (_d1d is not None and _d1d["kind"] == "direct"
                            and manual_edge is None and not _lg_cont):
                        decision = AttributionDecision(
                            fact_id=fact.id, matter_id=_d1d["matter_id"],
                            confidence=0.9, source=AttributionSource.CONTINUATION,
                            is_new_matter=False, signal=f"d1:{_d1d['branch']}",
                            decision={"layer": CONT_LAYER, "branch": _d1d["branch"],
                                      "verdict": "continue_to",
                                      "anchor": _d1d["anchor"],
                                      "window": _d1d["window"],
                                      "ts": now.isoformat()},
                        )
                        self._apply_decision(fact, decision, pipeline, now,
                                             goal_ctx=_goal_ctx)
                        # ⚠️ 与循环尾部共享簿记的复制段（改那边要同步这里）：
                        _turn_matters[_lk_f].add(decision.matter_id)
                        _matter_votes[_lk_f][decision.matter_id] += 1
                        _d1_counters["direct_facts"] += 1
                        attributed += 1
                        _prog(phase="attribute", done=attributed, total=len(added_facts))
                        continue

                if self._matter_table is not None and fact.embedding is not None:
                    candidates = self.search_matters(fact.embedding, k=top_k)
                else:
                    candidates = list(base_matters)
                # 向量段的长度：GM-1/GM-2 的确定性候选超预算时只能从**这一段**里让位
                # （见下方 `_make_room`）。
                _vec_n = len(candidates)
                # 加上本 pass 新建的 Matter（mid-batch 可见）
                candidates.extend(new_matters_this_pass)

                # ── GM-1：延续信号进候选（MQ-S13）──
                # 「延续 ∧ 无新主体」→ 把**上一轮 fact 所属的 Matter** 加进候选池。
                # 🔴 只加候选，判定仍由 L1–L5 走完（L4 判 same 才继承）。这样把
                # "参照集吞噬"那类假收益的代价从**误合并**降级为**多送一次裁决**。
                _cont_ids: set[str] = set()
                _prev_key = cont_prev_key.get(fact.source_ledger_key, "")
                if _prev_key:
                    _have = {m.matter_id for m in candidates}
                    for _mid in _prev_turn_matters(_prev_key):
                        _cont_supplied += 1
                        if _mid in _have:
                            continue   # 向量本来就召回了 —— 好事，不是"没触发"
                        _m = self.get_matter(_mid)
                        if _m is None or not pipeline.is_linkable(_m):
                            continue
                        _vec_n = _make_room(candidates, _vec_n)
                        candidates.append(_m)
                        _cont_ids.add(_mid)
                        _cont_candidates_added += 1

                # ── GM-2：按键反查进候选 ──
                _key_ids: set[str] = set()
                if _key_enabled:
                    from bladex_core.topic_keys import derive_topic_keys as _dtk2
                    _fkeys = _dtk2(
                        getattr(fact, "keywords", None),
                        getattr(fact, "topic", "") or "",
                        fact.proposal_titles, fact.entities,
                    )
                    _have2 = {m.matter_id for m in candidates}
                    for _mid, _nshared, _skeys in self.key_based_matter_candidates(
                        _fkeys, _key_inverted, _key_total_matters,
                    ):
                        if _mid in _have2:
                            continue
                        _m = self.get_matter(_mid)
                        if _m is None or not pipeline.is_linkable(_m):
                            continue
                        _vec_n = _make_room(candidates, _vec_n)
                        candidates.append(_m)
                        _key_ids.add(_mid)
                        _key_candidates_added += 1
                        logger.debug("index_attrib_key_candidate",
                                     fact_id=fact.id, matter_id=_mid,
                                     shared=_nshared, keys=_skeys[:5])

                # ── 账本准入闸（ADR-0032 §4.5）──
                # 🔴 **单一过滤点**：向量召回 / 本 pass 新建 / GM-1 延续 / GM-2 按键
                # 四路候选在这里汇合，过完闸才轮到内容判定（L2/L3/L4 的匹配面）。
                # 放在四路各自的 `is_linkable` 里 = 同一件事写四遍（刚性原则 12）。
                # manual 不过闸——用户显式指定的归属永远算数（ADR-0012 红线 6）。
                if _lg_enabled and not manual_matter_id:
                    _cand_before = len(candidates)
                    candidates = [
                        _c for _c in candidates
                        if not _lg_reject(fact.source_ledger_key, _c)
                    ]
                    _dropped = _cand_before - len(candidates)
                    if _dropped:
                        _la_counters["gate_cand_dropped"] += _dropped

                decision = pipeline.attribute(
                    fact, candidates,
                    manual_matter_id=manual_matter_id,
                    user_message=ledger_key_to_user_msg.get(fact.source_ledger_key),
                    now=now,
                )

                # GM-1 溯源：本条最终落到的卡是不是延续通路送进去的候选。
                # 记进 `decision.decision` 便于 GM-5 前后对照与 judgment 台账审计
                # （"这张卡是怎么被看见的"与"L4 判了什么"是两件事，必须分开可查）。
                if _cont_ids and decision.matter_id in _cont_ids:
                    decision.decision["candidate_source"] = "continuation"
                    decision.decision["continuation_prev_key"] = _prev_key
                    _cont_adopted += 1
                elif _key_ids and decision.matter_id in _key_ids:
                    decision.decision["candidate_source"] = "key_recall"
                    _key_adopted += 1
                if decision.matter_id and decision.matter_id != UNASSIGNED_MATTER_ID:
                    _turn_matters[fact.source_ledger_key].add(decision.matter_id)

                self._apply_decision(fact, decision, pipeline, now, goal_ctx=_goal_ctx)
                # U4B B4（ADR-0026 §4.1）：Matter 卡 participants populate——「这卡之前谁开发的」
                # （交接原语，U7 ②平面消费）。确定性从 fact.agent_id 聚合。gated 默认关=零回归。
                if (os.environ.get("BLADEX_MATTER_PARTICIPANTS", "") == "1"
                        and decision.matter_id and decision.matter_id != UNASSIGNED_MATTER_ID
                        and getattr(fact, "agent_id", "")):
                    _m = self.get_matter(decision.matter_id)
                    if _m is not None:
                        _m.touch_participant(fact.agent_id, ts=now)
                        self.add_matter(_m)
                # MS-5 ①：事件折叠**不再在这里逐 fact 做**。
                # 旧形态是每条 fact 各自把本轮事件折进它自己归属到的 Matter，
                # 于是同一轮的 facts 落到两张卡时，事件被折进**两张**——
                # H2 那条"BladeX 指派消息进泰山 Matter"就是这么来的。
                # 改为先投票、循环结束后按**主归属**折一次（见下方 _fold_session_events）。
                if decision.matter_id and decision.matter_id != UNASSIGNED_MATTER_ID:
                    _matter_votes[fact.source_ledger_key][decision.matter_id] += 1
                if decision.is_new_matter and decision.matter_id != UNASSIGNED_MATTER_ID:
                    existing_ids = {m.matter_id for m in new_matters_this_pass}
                    if decision.matter_id not in existing_ids:
                        m = self.get_matter(decision.matter_id)
                        if m is not None:
                            new_matters_this_pass.append(m)
                attributed += 1
                _prog(phase="attribute", done=attributed, total=len(added_facts))
            except Exception as e:  # noqa: BLE001
                # 🔴 `error_type` + 计数是 2026-09-02 加的（MQ-P14）：这个 catch-all
                # 让**编程错误**长得和"某条 fact 数据不好"一模一样。当天一个
                # `KeyError('matter_id')` 被它兜成逐条 warning，归属整链空转，
                # 11 个不相干用例集体转红而栈全被吞掉——`index_rebuild_done` 里
                # `attributed=0` 与"本来就没东西可归"也分不开。
                # 类型名足以当场分流：KeyError/AttributeError/TypeError = 我们的 bug。
                _attrib_failed += 1
                _attrib_failed_types[type(e).__name__] += 1
                logger.warning("index_attribution_failed", fact_id=fact.id,
                               error_type=type(e).__name__, error=str(e))

        # 激活态证据（纪律：只在特定流量形态才激活的机制，上线必须拿得到读数——
        # 边界/分层配额那次 −22 分就是死在"从未真实运行过"上）。
        # 计数器行本身含关键字，grep 要带 `=[1-9]`（仪器问题 #12）。
        # MQ-S27 激活态（grep 带 `=[1-9]`，仪器问题 #12）：分流数为 0 也打——
        # "恒为 0"与"没在跑"必须可区分（FUNNEL_HIT 潜伏的原因）。
        if _lane_split_enabled:
            logger.info("index_lane_hard_split", lane_skipped=_lane_skipped,
                        facts=len(added_facts))
        # V-L5 账本锚层：锚挂 fact 数 + 新建锚定 Matter 数（0 也打——"没触发"要可见）。
        # 🔴 `bindings` 是**绑定表大小**，不是命中数 —— MQ-L14 里它一直是 5，
        # 而真实命中是 0，光看它会以为锚层在工作。判据只能是 `anchored`。
        if _la_enabled:
            logger.info("index_ledger_anchor",
                        anchored=int(_la_counters.get("anchored", 0)),
                        matters_created=len(_la_created),
                        multi_matter=int(_la_counters.get("multi_matter", 0)),
                        # 台账缓存判决点名的 Matter 已不在库、因而落 L5 的次数
                        # （任务卡 T3）。它是孤边的**替代读数**：修法生效后
                        # 这个数应当≈原先的孤边增量，而孤边不再增长。
                        stale_judgment_targets=pipeline.stale_judgment_targets,
                        # MQ-L26：明确没拿到账本面而**不锚**的次数。这是本次修法
                        # 的直接读数——它涨 ⇒ 那些轮的产出落回 DPL 兜底，
                        # 恒 0 ⇒ 要么 live 没这类轮，要么字段没接上（两者要分开查：
                        # 前者看 `agency_toolface_decision injected=False` 的量）。
                        no_ledger_face=int(_la_counters.get("no_ledger_face", 0)),
                        # 账本准入闸（ADR-0032 §4.5）四个读数，分两轴看：
                        # **拒因**（哪道闸开的火）gate_other_ledger / gate_before_created；
                        # **落点**（在哪拦下的）gate_cont_blocked（CONT 直挂，绕候选池）
                        # / gate_cand_dropped（候选过滤，管 L2/L3/L4）。
                        # 🔴 两轴必须分开报：只看拒因分不出"CONT 还在错挂"与
                        # "候选面还在漏"，而这两个的修法不同（前者改窗口判定、
                        # 后者改匹配面）。恒 0 先证明桶可达——本闸只在锚层开着
                        # 且库里有 ledger_id 非空的 Matter 时才可能非零。
                        gate_enabled=_lg_enabled,
                        gate_other_ledger=int(_la_counters.get("gate_other_ledger", 0)),
                        gate_before_created=int(_la_counters.get("gate_before_created", 0)),
                        gate_cont_blocked=int(_la_counters.get("gate_cont_blocked", 0)),
                        gate_cand_dropped=int(_la_counters.get("gate_cand_dropped", 0)),
                        # 第三个落点：L4 裁决目标（缓存 + 现场都过这道）。
                        # 恒 0 要分两种查：谓词没接上（看 `_lg_enabled`）
                        # vs 真的没开火。第五次重建前它不存在，漏了 1 条。
                        gate_l4_blocked=dict(pipeline.blocked_l4_targets),
                        bindings=len(_la_bind),
                        scopes_on_timeline=len(_la_timeline))
        # G12.2 D1 激活态：分支分布 + 直挂 fact 数 + 台账写入数（0 也打，同上理由）。
        if _d1_active:
            if _d1_cur_turn:
                _d1_finalize_turn(_d1_cur_turn)   # 收最后一轮（含其 R5 台账）
            logger.info("index_d1_decisions",
                        facts=len(added_facts),
                        **{str(k): int(v) for k, v in sorted(_d1_counters.items())})
        # GM-1 激活态：`cont_pairs`=判为延续的轮数、`cont_candidates`=真进候选池的次数、
        # `cont_adopted`=L4 最终采纳的次数。三个数分开报，因为它们回答三个不同问题：
        # 信号有没有、有没有接上、接上之后判定层认不认。**只报覆盖，不报对错**——
        # 归属正确性只能由 `probe_attribution_quality.py` + 人工签收来判（GM 纪律 3）。
        if _cont_enabled:
            logger.info("index_attrib_continuation",
                        cont_pairs=len(cont_prev_key),
                        cont_supplied=_cont_supplied,
                        cont_candidates=_cont_candidates_added,
                        cont_adopted=_cont_adopted,
                        new_subject_max=_cont_new_subject_max,
                        facts=len(added_facts))
        # GM-2 激活态：`key_total_matters` 是判别力门的分母，必须打出来——
        # 门槛 = ratio × 它，读日志的人不看见分母就无法判断门是紧了还是松了。
        if _key_enabled:
            logger.info("index_attrib_key_recall",
                        key_candidates=_key_candidates_added,
                        key_adopted=_key_adopted,
                        key_total_matters=_key_total_matters,
                        key_index_size=len(_key_inverted),
                        facts=len(added_facts))

        # ── MS-5 ① + MS-4 ①：会话事件按**主归属**折叠，一轮一次 ──
        #
        # 两处修正合在这一段：
        #   MS-5 ①：折叠目标不再是"每条 fact 各自的 Matter"，而是该轮 facts 经归属
        #           管线落定的**多数派** Matter。旧形态下同一轮的 facts 落到两张卡时，
        #           事件被折进两张——H2「BladeX 指派消息进泰山 Matter」正是此形态。
        #           无 fact 产出的轮次**不折叠**（没有归属依据，宁可不记）。
        #   MS-4 ①：事件来源优先取**蒸馏产出**（`DistillFact.event`），
        #           规则词表（`detect_session_events`）降为无蒸馏产物时的兜底。
        #           蒸馏看得懂语义，且 detail 是蒸馏后的单句而非 130 字符原文
        #           （H2 的塞卡问题随之消失）。
        if _lifecycle_enabled and _matter_votes:
            from bladex_core.session_events import detect_session_events

            for _lk, _votes in _matter_votes.items():
                if not _votes:
                    continue
                # 主归属：票数最高；平票时按 matter_id 排序取定（保重放确定性）
                _mid = sorted(_votes.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
                _events: list[tuple[str, str]] = []
                _distilled = _distilled_events.get(_lk)
                if _distilled and _distilled[0] and _distilled[0] != "none":
                    _events = [_distilled]
                else:
                    _src = ledger_key_to_user_msg.get(_lk, "")
                    if _src:
                        _events = detect_session_events(_src)
                if not _events:
                    continue
                _lm = self.get_matter(_mid)
                if _lm is None:
                    continue
                _ts_iso = ledger_key_to_ts.get(_lk, "")
                try:
                    _ev_ts = _dt.fromisoformat(_ts_iso) if _ts_iso else None
                except ValueError:
                    _ev_ts = None
                _changed = False
                for _ev, _detail in _events:
                    _dk = (_mid, _lk, _ev)
                    if _dk in _lifecycle_recorded:
                        continue
                    _lifecycle_recorded.add(_dk)
                    # M0-10：持久化去重——跨 rebuild 重放同一事件返回 False
                    # （既不追加也不 bump version），据此不计数、不落盘。
                    if not _lm.record_lifecycle(_ev, _detail, ts=_ev_ts):
                        continue
                    _lifecycle_folded += 1
                    _changed = True
                if _changed:
                    self.add_matter(_lm)

        # 标记本轮所有处理的 key 为已消费（含 auxiliary）。
        # MS-1：蒸馏失败且未超重试上限的 turn 已在上面从这个列表里摘掉。
        for k in new_consumed_keys:
            self._mark_consumed(k)
            self._clear_distill_retry(k)   # 成功消费 → 计数归零

        # T3: 周期消化未归属池（成簇的新开 Matter，散乱的留池）
        try:
            digested = self.digest_unassigned_pool(
                digestion_threshold=self._digestion_threshold, goal_ctx=_goal_ctx)
            if digested > 0:
                logger.info("index_unassigned_digested_after_rebuild", new_matters=digested)
        except Exception as e:
            logger.warning("index_unassigned_digestion_failed", error=str(e))

        # G12.3 goal 捕获激活态：`captured` + 各分档 reason（**0 也打**：恒为 0 与
        # 没在跑必须可区分）。🔴 位置**必须在消化路之后**——消化路也开卡也取 goal，
        # 打在它前面就是尺子读早了（这一版本身就是从"读数参照系"那批教训里长出来的）。
        # 🔴 这不是内部统计，是一把尺子：`scaffold_turn` / `empty_after_strip` 的量
        # = "有多少卡是被机器文本开出来的"，即准入缺陷（MQ-S44/S45 那一族）的规模。
        # 动态键 `goal_<reason>`：闭集在 `task_goal.GOAL_ABSENT_REASONS`，
        # 这里不重抄字面量（也顺带绕开 H1 gate 的 schema 字段名赋值点判定）。
        logger.info("index_goal_capture",
                    new_matters=sum(self._goal_counters.values()),
                    **{f"goal_{k}": int(v) for k, v in sorted(self._goal_counters.items())})

        # ADR-0021 section 2.4: 重放 scope 提升管理事件（personal->team/org，rebuild 不丢失）。
        try:
            self._replay_scope_promotions(ledger)
        except Exception as e:
            logger.warning("index_scope_promote_replay_failed_top", error=str(e))

        # ADR-0028 E2.3：ref_count 聚合回写 + importance 重算。
        # 放在归属之后：此时本轮新 fact 已入库，命中的旧 fact 也仍在。
        _refcount_updated = self._apply_injection_hits(injection_hits, hit_ts, injection_keys)
        # M3-1：Matter 相关钟回写（与 fact 侧同一支笔）
        _matter_hits_written = self._apply_matter_injection_hits(matter_hit_ts)

        self._last_rebuild_backlog = backlog
        _prog(phase="done", done=len(turns), total=len(turns),
              new_facts=len(new_facts), attributed=attributed)
        _orphan_ids, _orphan_hits = getattr(self, "last_injection_orphans", (0, 0))
        logger.info("index_rebuild_done", turns_processed=len(turns),
                     # MQ-P14：归属被 catch-all 吞掉多少条、都是什么异常。
                     # 🔴 非零且类型是 KeyError/AttributeError/TypeError ⇒ **是 bug
                     # 不是脏数据**，别去查数据。恒 0 才是正常态。
                     attrib_failed=_attrib_failed,
                     attrib_failed_types=dict(_attrib_failed_types),
                     # MQ-S38：命中回写里有多少 id 指空（0 = 这一轮没丢信号）。
                     # 进汇总行的理由：只在 `_apply_injection_hits` 里 warning，
                     # 得有人恰好去翻那一行；漏斗式的"这次重建丢了多少"应当一眼可见。
                     injection_orphan_ids=_orphan_ids,
                     injection_orphan_hits=_orphan_hits,
                     injection_rescued=getattr(self, "last_injection_rescued", 0),
                     injection_ambiguous=getattr(self, "last_injection_ambiguous", 0),
                     matter_hit_orphans=getattr(self, "last_matter_hit_orphans", 0),
                     new_facts=len(new_facts), attributed=attributed,
                     skipped_auxiliary=skipped_auxiliary,
                     skipped_filtered=skipped_filtered,
                     consumed=len(new_consumed_keys), full=full,
                     cleared=clear_db, backlog=backlog,
                     lifecycle_folded=_lifecycle_folded,
                     refcount_updated=_refcount_updated,
                     matter_hits_written=_matter_hits_written)
        # T8b: rebuild ANN indexes after bulk insert.（2026-07-29 三轮修复后重新接线；
        # 二轮那次"摘除接线止血"的注释已随之删除，别再照着它判断当前状态。）
        # Must never break the Memory Index write pipeline -- wrap in try/except so index
        # failures are logged but do not interrupt rebuild_from_hub.
        try:
            self.rebuild_ann_indexes()
        except Exception as e:
            logger.warning("index_ann_rebuild_error", error=str(e))

        # MQ-R1（2026-08-17）：碎片自维护挂在 **pass 收尾**，不是只挂空闲轮。
        # 这是 T8b B17 那个坑的同一形态：卫生作业只写在 `if not turns:` 分支里，
        # 而碎片恰恰是**有新 turn 时**堆出来的——积压排空期一路写、一路碎，
        # 空闲轮永远等不到（08-17 实测：消费 27 轮 → 碎片 2 → 43，1:1 回涨，
        # 手工压缩撑不过一次排空）。触发条件在 `maybe_compact_fact_vectors` 里，
        # 未超阈值即空转，所以放在每个 pass 收尾不心疼。
        try:
            self.maybe_compact_fact_vectors()
        except Exception as e:  # noqa: BLE001 —— 卫生作业绝不影响 rebuild 语义
            logger.warning("index_fact_compact_error", error=str(e))

        return len(new_facts)


class IndexDistillJournal:
    """蒸馏台账的 MemoryIndex 适配（ADR-0020 T1.3，实现 DistillJournalProtocol）。

    从 Memory Hub 迁 Memory Index meta：consolidator 独占 Memory Index 写锁，台账写入无障碍（Memory Hub 旧实现因
    consolidator read_only Memory Hub 而写失败 100%，ADR-0020 根因）。
    index 延迟绑定--distill_journal 在 LLMDistiller 之前创建、index 在之后创建，bind 注入。
    """

    def __init__(self, index: MemoryIndex | None = None) -> None:
        self._index = index

    def bind(self, index: MemoryIndex) -> None:
        self._index = index

    def get_distill(
        self, source_text: str, distill_model: str, prompt_ver: str,
        context_digest: str = "",
    ) -> DistillOutput | None:
        if self._index is None:
            return None
        rec = self._index.get_distill(source_text, distill_model, prompt_ver,
                                      context_digest)
        if rec is None:
            return None
        return DistillOutput(
            facts=list(rec.facts),
            matter_proposals=list(rec.matter_proposals),
            model_name=rec.distill_model,
        )

    def put_distill(
        self, source_text: str, distill_model: str, prompt_ver: str,
        output: DistillOutput, context_digest: str = "",
    ) -> str:
        if self._index is None:
            return ""
        return self._index.append_distill(
            source_text, distill_model, prompt_ver,
            output.facts, output.matter_proposals, context_digest,
        )

    # ── 已蒸话语 memo（2026-08-08）──
    # core 的 `_collect_candidates_v4` 经 distiller 探测这两个方法；
    # 缺失即整体降级回旧行为，所以老部署/测试替身不受影响。

    def discourse_seen(self, text: str, prompt_ver: str) -> bool:
        if self._index is None:
            return False
        return self._index.discourse_seen(text, prompt_ver)

    def mark_discourse(self, text: str, prompt_ver: str, ledger_key: str = "") -> None:
        if self._index is None:
            return
        self._index.mark_discourse(text, prompt_ver, ledger_key)


class IndexConsolidationWorker:
    """后台 worker：周期性从 Memory Hub 增量重建 Memory Index（ADR-0009 §7）。

    启动后定时调 MemoryIndex.rebuild_from_hub()，
    consolidation 失败只降级不停摆（ADR-0009）。
    """

    def __init__(
        self,
        index: MemoryIndex,
        ledger: MemoryHub,
        interval_s: float = 30.0,
    ) -> None:
        self._index = index
        self._hub = ledger
        self._interval_s = interval_s
        self._running = False
        self._task: Any = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._index.open()
        self._task = asyncio.create_task(self._run())
        logger.info("index_worker_started", interval_s=self._interval_s)

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("index_worker_stopped")

    async def _run(self) -> None:
        """周期性从 Memory Hub 增量重建 Memory Index。

        consolidation 是同步 CPU 密集操作，必须跑在线程池里，
        不能阻塞事件循环（否则 Redis 读超时、proxy 响应变慢）。
        """
        while self._running:
            try:
                await asyncio.sleep(self._interval_s)
                if not self._running:
                    break

                # embedder 不可用时跳过（不每轮重试下载）
                embedder = getattr(self._index, "_embedder", None)
                if embedder is not None and hasattr(embedder, "available"):
                    if not embedder.available:
                        continue

                # 在线程池里跑同步 consolidation，不阻塞事件循环
                loop = asyncio.get_running_loop()
                new_facts = await loop.run_in_executor(
                    None, self._index.rebuild_from_hub, self._hub,
                )
                if new_facts > 0:
                    logger.info("index_worker_consolidated", new_facts=new_facts)
            except asyncio.CancelledError:
                break
            except RuntimeError as e:
                # embedder 不可用 — 不每轮刷日志
                logger.debug("index_worker_embedder_unavailable", error=str(e))
                await asyncio.sleep(self._interval_s)
            except Exception as e:
                logger.warning("index_worker_error", error=str(e))
                await asyncio.sleep(self._interval_s)
