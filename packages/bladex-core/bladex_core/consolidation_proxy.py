"""ProxyConsolidator — 从 Memory Hub 对话轮次提炼原子 Fact（v4 proxy 架构）。

设计原则（继承 v3 consolidation 的验证结论）：
  - 只 embed user 消息（事实来自用户陈述，assistant 是回应）
  - 跳过过短（< _MIN_CONTENT_CHARS）和过长（> _MAX_CONTENT_CHARS）消息
  - 用 embedding novelty 去重（cosine 相似度 >= _NOVELTY_THRESHOLD → 跳过）
  - 无 embedder 时跳过（宁缺毋滥）
  - 按逻辑用户轮聚合（T2 的 logical_turn 字段）

与 v3 consolidation.py 的区别：
  - v3 从 Hermes state.db 读消息、写 BladeXStore（holographic）
  - v4 从 Memory Hub Turns 读、返回 Fact 列表（agent 中立，存哪里由 proxy 决定）
  - core 不依赖 proxy/agent，只输出 Fact 对象
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections import Counter
from typing import TYPE_CHECKING, Any, Protocol

from bladex_core.distillation import DistillerProtocol
from bladex_core.envelope import (
    prepare_distill_inputs,
    segment_long_text,
    strip_envelopes,
)
from bladex_core.flags import flag_enabled, flag_number
from bladex_core.fact import ConversationTurn, Fact, ItemKind, Provenance
from bladex_core.funnel import FUNNEL_DEDUP, FUNNEL_STORED
from bladex_core.importance import compute_importance
from bladex_core.supersede import SupersedeMergePlan, plan_supersede_merge

# 旧 kind（preference|event|task|decision|general）→ 新 item_kind 六类兜底映射（U4）。
# 蒸馏器显式产出 item_kind 时优先用它；空时按旧 kind 推断，仍无则 assertion（最通用）。
_KIND_TO_ITEM: dict[str, ItemKind] = {
    "preference": ItemKind.PREFERENCE,
    "decision": ItemKind.ASSERTION,
    "task": ItemKind.ASSERTION,
    "event": ItemKind.ASSERTION,
    "general": ItemKind.ASSERTION,
}


def _resolve_item_kind(candidate: dict[str, Any]) -> ItemKind:
    """从候选解析 item_kind：蒸馏器显式值优先，否则按旧 kind 兜底映射。"""
    raw = (candidate.get("item_kind") or "").strip().lower()
    try:
        return ItemKind(raw)
    except ValueError:
        return _KIND_TO_ITEM.get((candidate.get("kind") or "").strip().lower(), ItemKind.ASSERTION)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _checker_is_novel(checker: Any, emb: list[float], entities: list[str],
                      content: str) -> bool:
    """调 novelty checker，带 content（E6.5 标识符前置放行）。

    旧 checker（测试替身 / 外部实现）没有 content 参数 → TypeError 退回旧签名，
    行为逐字不变（零改造兼容，与 IndexFactRetriever 的降级同一套路）。
    """
    try:
        return checker.is_novel(emb, entities=entities, content=content)
    except TypeError:
        return checker.is_novel(emb, entities=entities)

# 候选消息长度过滤
_MAX_CONTENT_CHARS = 500
_MIN_CONTENT_CHARS = 20

# ADR-0019 P2：结论蒸馏的最短长度（太短的"好的/已完成"没有信息量）。
# 无上限——蒸馏器内部截断输入；结论只在有 LLM 蒸馏器时处理（透传=跳过）。
_MIN_CONCLUSION_CHARS = 40

# 工作产出通道的最短长度（2026-07-29）。比结论略高——调工具前的说明里
# "好的，我来看看" 这类过场话更多，门槛抬一点滤掉噪音。
_MIN_PROGRESS_CHARS = 60

# M1-2：assistant 正文的**分段**粒度（不是截断——截断丢的是后半段的结论，D2/案例5）。
# 比 user 侧的 500 大：assistant 半边是"做了什么"的载体，
# D3 实测这一侧是欠采的那一头（8K 深度分析 → 截断到 2000 + 上限 3 条）。
_ASSISTANT_SEGMENT_CHARS = 2000

# novelty 阈值：与已有 facts 最高 cosine 相似度 >= 此值 → 近乎完全重复，跳过。
#
# M2-2：默认值收编 `flags.MEMORY_NUMERIC_DEFAULTS`，语义降为**只拦完全重发**（0.98）。
# 此前 0.95 是让 novelty 独自决定"这条该不该入库"，而 e5 在本库的噪声底就是
# 0.816–0.93 —— 它注定要么漏网要么误杀。那个判断现在归裁决器（看文本不看阈值）。
# 不变式：novelty_threshold > semantic_threshold（否则 Matter 锁死单例）。
def _default_novelty_threshold() -> float:
    return flag_number("BLADEX_NOVELTY_THRESHOLD")


_DEFAULT_NOVELTY_THRESHOLD = 0.98


def _parse_valid_until(raw: Any) -> Any:
    """蒸馏产出的 `valid_until`（ISO8601 字符串）→ datetime | None。

    解析不出一律 None（= 无到期）。**宁可当成不过期，也不要猜一个日期**：
    猜错会让一条仍然成立的记忆被 M3-3 提前置 t_invalid，
    那是静默丢记忆，比"该过期的没过期"严重得多。
    """
    from datetime import UTC, datetime

    text = str(raw or "").strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            dt = datetime.strptime(text, fmt)
            return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
        except ValueError:
            continue
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
    except ValueError:
        logger.debug("distill_valid_until_unparsed raw=%.40r", text)
        return None


def _deterministic_fact_id(ledger_key: str, content: str) -> str:
    """生成确定性 fact_id（G2 修复）。

    基于 source_ledger_key + content 的 SHA256 hash，保证同一 Memory Hub turn 的同一内容
    在每次 rebuild 时生成相同 fact_id。这样手动 fact 级归属映射
    （admin event 里 target_key = fact_id）在 full rebuild 后不会丢失。

    不同 ledger_key 或不同 content -> 不同 id（唯一性）。
    """
    raw = f"{ledger_key}:{content}"
    return f"fact_{hashlib.sha256(raw.encode()).hexdigest()[:12]}"


class EmbedderProtocol(Protocol):
    """嵌入接口（可 mock，单测不跑真模型）。

    embed() = passage 语义（历史行为：adapter 内部加模型所需前缀）。
    embed_query/embed_passage 是可选扩展方法（embedding 后端可选化，2026-07-26）：
    前缀责任收进 adapter，调用方不再手拼 "query: "/"passage: "。
    旧 mock/第三方实现只有 embed() 时，走 embed_query_compat/embed_passage_compat
    降级路径（行为与历史逐字一致）。
    """

    def embed(self, texts: list[str]) -> list[list[float]]:
        """把文本列表嵌入成向量列表。"""
        ...


def embed_query_compat(embedder: Any, texts: list[str]) -> list[list[float]]:
    """query 语义嵌入，带旧接口降级。

    - adapter 实现了 embed_query -> 直接调（前缀由 adapter 决定）。
    - 只有 embed()（旧 mock / 第三方）-> 保持历史调用方行为：手拼 "query: " 前缀
      再交 embed()（历史上 FastEmbedAdapter.embed 会再加 "passage: "，
      形成 "passage: query: ..." —— 阈值在此行为上标定，逐字保持，不"修"）。
    """
    fn = getattr(embedder, "embed_query", None)
    if fn is not None:
        return fn(texts)
    return embedder.embed([f"query: {t}" for t in texts])


def embed_passage_compat(embedder: Any, texts: list[str]) -> list[list[float]]:
    """passage 语义嵌入，带旧接口降级（embed() 历史语义就是 passage）。"""
    fn = getattr(embedder, "embed_passage", None)
    if fn is not None:
        return fn(texts)
    return embedder.embed(texts)
class NoveltyChecker(Protocol):
    """novelty 检查接口（G3：可接 LanceDB 向量检索替代全量 brute-force）。

    🔴 消费方唯一性（2026-08-09，M5-2 判据③修复）：本接口**只服务判重**。
    裁决的邻居查询曾复用同一个对象（`_neighbors_for` 直接 getattr .neighbors），
    于是全量重建 `novelty_checker=None`（对判重正确——库马上 clear）把裁决的
    跨批邻居也整段关掉——同一对象、两个消费方、语义相反，与 2026-07-28
    `auxiliary` 事故同型。裁决邻居现在走 `InRunNeighborSource`，两者不再共享语义。
    """

    def is_novel(self, embedding: list[float], entities: list[str] | None = None) -> bool:
        """返回 True 表示无已有 fact 与该 embedding 近似重复。

        方案 B：entities 提供时启用实体感知去重（cosine 高但实体有差异仍判新颖）。
        """
        ...


class InRunNeighborSource:
    """裁决专用邻居源（M5-2 判据③修复，2026-08-09 拍板：修法 2 + 两模式统一）。

    两个视图合一：

    - **本轮内存集**：`add()` 收录本轮所有转入 pending 的候选。它们要到
      `_adjudicate` 循环②才进 `running`，而邻居全部在循环①算完——即任何
      "有邻居的候选"对它之后的所有候选整体不可见（第二个洞）。密集簇里
      第一条进 running 后其余全部转 pending 消失，这正是「第19条」簇
      [0]×[4] cosine 0.9637 却互不可见的完整机制。
    - **库内视图**：包装 novelty_checker 的 `.neighbors`（增量时 = LanceDB；
      全量重建 checker=None = 无库内视图——旧库马上要 clear，**不得**泄漏进裁决）。

    形状说明：阈值**先**筛、再取 top-k（内存侧）。LanceDB 侧是 limit(k) 先于
    阈值筛（HANDOFF 顺带发现 #3，密集簇里够格邻居可能先被 top-k 砍掉），
    那是 checker 内部的形状，本类不改它、只在合并后按相似度重排取 top-k。

    悬空 target 边界（已知，可接受）：后见候选可取代一条尚未裁决的 pending
    邻居，而它随后被 NOOP 丢弃。落盘侧 `apply_verdict` 按 `by_id` 查不到即跳过
    （目标本来就没进库）；judgment 台账里会留悬空 id，读侧脚本需容忍。
    """

    def __init__(self, checker: Any | None = None) -> None:
        self._checker = checker
        self._facts: list[Fact] = []

    def add(self, fact: Fact) -> None:
        """收录一条本轮已算过邻居的候选（无 embedding 的不收，查不到）。"""
        if fact.embedding:
            self._facts.append(fact)

    def discard(self, fact_id: str) -> None:
        """裁决判丢弃（NOOP 不入库）后摘除——防同实例复用时幽灵邻居跨批存活。

        注意这**救不了**本批内已装好的裁决包（邻居全在循环①算完），
        那部分是文档化的悬空 target 边界；这里只保证跨批不再看见它。
        """
        self._facts = [f for f in self._facts if f.id != fact_id]

    def neighbors(
        self, embedding: list[float], *, k: int = 10, min_similarity: float = 0.90,
    ) -> list[tuple[Fact, float]]:
        """取 `min_similarity` 以上的近邻，内存集 + 库内视图合并去重，按相似度取 top-k。"""
        from bladex_core.adjudication import _cosine  # noqa: PLC0415 —— 防循环导入

        out: list[tuple[Fact, float]] = []
        for f in self._facts:
            sim = _cosine(embedding, f.embedding)
            if sim >= min_similarity:
                out.append((f, sim))

        fn = getattr(self._checker, "neighbors", None)
        if fn is not None:
            try:
                for row, sim in fn(embedding, k=k, min_similarity=min_similarity):
                    fid = str(row.get("fact_id") or row.get("id") or "")
                    if not fid or any(f.id == fid for f, _ in out):
                        continue
                    out.append((Fact(
                        id=fid, content=str(row.get("content", "") or ""),
                        subject=str(row.get("subject", "") or ""),
                        attribute=str(row.get("attribute", "") or ""),
                    ), sim))
            except Exception as e:  # noqa: BLE001 —— 库内视图拿不到 ≠ 内存集也作废
                logger.warning("neighbor_source_store_view_failed err=%s", e)

        out.sort(key=lambda p: -p[1])
        return out[: max(1, int(k))]


class ProxyConsolidator:
    """从 Memory Hub 对话轮次提炼原子 Fact。

    使用方式：
      consolidator = ProxyConsolidator(embedder=FastEmbedAdapter())
      facts = consolidator.consolidate_turns(turns, existing_facts=[])
      # facts 是新提炼的 Fact 列表，交给 Memory Index 存储层
    """

    def __init__(
        self,
        embedder: EmbedderProtocol | None = None,
        *,
        novelty_threshold: float = _DEFAULT_NOVELTY_THRESHOLD,
        distiller: DistillerProtocol | None = None,
        novelty_checker: NoveltyChecker | None = None,
        sensitivity_config: Any | None = None,
        entity_aware: bool = True,
        entity_overlap_threshold: float = 0.6,
        distill_concurrency: int = 1,
        progress_cb: Any | None = None,
        funnel: Any | None = None,
        adjudicator: Any | None = None,
        neighbor_source: Any | None = None,
    ) -> None:
        self._embedder = embedder
        self._novelty_threshold = novelty_threshold
        self._distiller = distiller
        self._novelty_checker = novelty_checker
        # M2-2：写入时裁决器（None = 回落旧形态 supersede + 冲突检测，回滚通道）
        self._adjudicator = adjudicator
        # M5-2 判据③（2026-08-09 拍板：修法 2，两模式统一）：裁决邻居源与判重
        # 对象**拆开**。默认构造 = 本轮内存集 + （若有 checker）库内视图；
        # 全量重建 novelty_checker=None 时自动退化为纯内存集——正确语义：
        # 旧库不进裁决，但本轮已产出的候选必须互相可见。
        self._neighbor_source = (
            neighbor_source if neighbor_source is not None
            else InRunNeighborSource(checker=novelty_checker))
        # 方案 B：实体感知 novelty（within-batch brute-force 路径）
        self._entity_aware = entity_aware
        self._entity_overlap_threshold = entity_overlap_threshold
        # ADR-0021 section 3.3c: Fact 血统继承--exposure_ceiling 从来源 Turn.sensitivity 映射。
        # None/disabled -> 全 public（个人模式零回归）。
        self._sensitivity_config = sensitivity_config
        # 2026-08-03（sync CLI）：蒸馏并发（默认 1 = 串行零回归）。并发只做**预蒸馏**：
        # 唯一文本 → DistillOutput 映射先并发算好，候选装配仍按原顺序串行消费映射
        # → 产出顺序/内容与串行逐字一致（G6 重建等价性不受并发影响）。
        self._distill_concurrency = max(1, int(distill_concurrency or 1))
        # 进度回调（sync CLI 终端进度/日志）：cb({"phase","done","total",...})，
        # 异常吞掉（观测面绝不影响主体）。None = 零行为变化。
        self._progress_cb = progress_cb
        # M4-2：漏斗埋点。None → NullFunnel（调用点不必到处判空，
        # 散落的判空正是"某处漏埋就静默少一段计数"的温床）。
        from bladex_core.funnel import NULL_FUNNEL

        self._funnel = funnel if funnel is not None else NULL_FUNNEL
        # MS-1（复核 D4）：本批蒸馏失败的 turn（ledger_key 集合）。
        # 在 `_collect_candidates` 开头重置；调用方读它来决定哪些 turn 不打消费标记。
        self.failed_ledger_keys: set[str] = set()
        # 其中"响应拿到了但没法用"（parse 失败）的那部分。
        # 这一分是队列不停摆的关键：parse 失败 = 上游活着 + 这条输入有问题，
        # 重放多少次都一样 → 该计次、超限放行；call 失败 = 上游的问题 → 不计次。
        self.parse_failed_ledger_keys: set[str] = set()

    def distiller_stats(self) -> dict[str, int] | None:
        """蒸馏器的累计计数器快照（calls / fails / parse_fails / ledger_hits）。

        为什么要公开这个（2026-08-05 事故）：`LLMDistiller.distill` 失败时**返回空
        DistillOutput 而不抛**（降级不停摆，见 distillation.py），所以调用方看到的
        "这批没提炼出事实" 与 "上游断了一条都没成" 长得一模一样。上游 DNS 断掉那次，
        consolidator 以 30 turn/轮的速度把积压全部标记成已消费、`new_facts=0`，
        日志里没有一行说这些轮次是白烧的。调用方要能区分两者，就必须看得见失败计数。

        返回 None = 没有蒸馏器或它不提供 stats()（测试桩），调用方按"无信息"处理。
        """
        distiller = self._distiller
        if distiller is None:
            return None
        stats_fn = getattr(distiller, "stats", None)
        if stats_fn is None:
            return None
        try:
            raw = stats_fn()
        except Exception:  # noqa: BLE001 —— 观测面绝不影响主体
            return None
        if not isinstance(raw, dict):
            return None
        return {k: int(v) for k, v in raw.items() if isinstance(v, int)}

    def _count_distill(self, output: Any) -> None:
        """M4-2：给一次蒸馏产出记一笔漏斗计数。

        `empty` 与 `*_fail` 必须分开 —— "这轮真没事实" 与 "上游断了" 在降级实现里
        长得一模一样，那正是 2026-08-05 事故里唯一缺的那条信息（与 MS-1 同源）。
        """
        from bladex_core.funnel import FUNNEL_DISTILL

        kind = str(getattr(output, "failure_kind", "") or "")
        if getattr(output, "failed", False):
            outcome = f"{kind or 'unknown'}_fail"
        elif getattr(output, "facts", None):
            outcome = "ok"
        else:
            outcome = "empty"
        self._funnel.inc(FUNNEL_DISTILL, labels={"outcome": outcome})

    def _record_distill_failure(self, ledger_key: str, output: Any) -> None:
        """记一次蒸馏失败（MS-1）。按 `failure_kind` 分两个集合。

        为什么要分：调用方对两者的处置**相反**——
        `parse` 失败（响应拿到了、内容用不了）说明上游活着、问题在这条输入上，
        必须计次并在超限后放行，否则一条永久坏的 turn 会把整条队列堵死；
        `call` 失败（连不上/超时/认证/配额）是上游的问题，计次就等于
        "断线 N 分钟后开始把积压当坏数据丢掉"。
        """
        if not ledger_key:
            return
        self.failed_ledger_keys.add(ledger_key)
        if str(getattr(output, "failure_kind", "") or "") == "parse":
            self.parse_failed_ledger_keys.add(ledger_key)

    def _report(self, **ev: Any) -> None:
        if self._progress_cb is None:
            return
        try:
            self._progress_cb(ev)
        except Exception:  # noqa: BLE001 —— 进度回调失败不影响 consolidation
            pass

    def _bulk_distill(
        self,
        user_texts: list[str],
        conclusion_texts: list[str],
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        """并发预蒸馏：唯一文本 → DistillOutput（G6：装配顺序不变，只是先算）。

        返回 (user_map, conclusion_map)；并发 <=1 或无蒸馏器 → None（走原串行路径）。
        台账写入经 IndexDistillJournal → RocksDB（线程安全）；同文本重复 miss 由
        write-if-absent 兜住。conclusion/progress 通道共用 distill_conclusion。
        """
        if self._distiller is None or self._distill_concurrency <= 1:
            return None
        from concurrent.futures import ThreadPoolExecutor, as_completed

        uniq_user = list(dict.fromkeys(user_texts))
        distill_conc_fn = getattr(self._distiller, "distill_conclusion", None)
        uniq_concl = list(dict.fromkeys(conclusion_texts)) if distill_conc_fn else []
        tasks: list[tuple[str, str]] = [("user", t) for t in uniq_user]
        tasks += [("conclusion", t) for t in uniq_concl]
        if not tasks:
            return {}, {}

        user_map: dict[str, Any] = {}
        concl_map: dict[str, Any] = {}

        def _work(kind: str, text: str) -> tuple[str, str, Any]:
            if kind == "user":
                return kind, text, self._distiller.distill(text)
            return kind, text, distill_conc_fn(text)

        total = len(tasks)
        done = 0
        logger.info("bulk_distill_start total=%d concurrency=%d "
                    "(user=%d conclusion=%d)",
                    total, self._distill_concurrency, len(uniq_user), len(uniq_concl))
        with ThreadPoolExecutor(max_workers=self._distill_concurrency) as ex:
            futs = [ex.submit(_work, k, t) for k, t in tasks]
            for fut in as_completed(futs):
                kind, text, out = fut.result()
                (user_map if kind == "user" else concl_map)[text] = out
                done += 1
                self._report(phase="distill", done=done, total=total)
                if done % 50 == 0 or done == total:
                    logger.info("bulk_distill_progress %d/%d", done, total)
        return user_map, concl_map

    def _ceiling_for(self, level: str) -> str:
        """ADR-0021 section 3.3c：来源 Turn 敏感等级 -> Fact.exposure_ceiling（血统继承）。

        sensitivity_config None/disabled -> 全 public（个人模式零回归）。
        enabled 时 levels[level] 映射；未知等级 fail-closed 取最严（strictest_exposure）。
        """
        from bladex_core.sensitivity import EXPOSURE_PUBLIC, strictest_exposure
        cfg = self._sensitivity_config
        if cfg is None or not getattr(cfg, "enabled", False):
            return EXPOSURE_PUBLIC
        levels = getattr(cfg, "levels", {}) or {}
        if level in levels:
            return levels[level]
        return strictest_exposure(levels)

    def consolidate_turns(
        self,
        turns: list[ConversationTurn],
        existing_facts: list[Fact] | None = None,
    ) -> list[Fact]:
        """从一批对话轮次提炼 Fact。

        参数：
          turns: Memory Hub 中的对话轮次（归一成 ConversationTurn）
          existing_facts: Memory Index 中已有的 facts（用于 novelty 去重）

        返回：新提炼的 Fact 列表（不含已有的）。
        无 embedder 时返回空列表（宁缺毋滥）。
        """
        if self._embedder is None:
            logger.info("consolidate_skip reason=no_embedder turns=%d", len(turns))
            return []

        # 收集候选消息（user 消息，过滤过长/过短）
        candidates = self._collect_candidates(turns)
        if not candidates:
            logger.info("consolidate_skip reason=no_candidates turns=%d", len(turns))
            return []

        # 准备已有 facts 的嵌入矩阵（G3: 有 novelty_checker 时不需全量加载）
        existing_embeddings: list[list[float]] = []
        # 方案 B：实体集合与嵌入矩阵并行（within-batch 实体感知去重用）
        existing_entities: list[list[str]] = []
        # ADR-0028 E6.5：标识符前置放行用的已存文本（与 embeddings 同序）
        existing_texts: list[str] = []
        if self._novelty_checker is None and existing_facts:
            existing_embeddings = [f.embedding for f in existing_facts if f.embedding]
            existing_entities = [list(f.entities or []) for f in existing_facts if f.embedding]
            existing_texts = [f.content or "" for f in existing_facts if f.embedding]
            if len(existing_embeddings) < len(existing_facts):
                # 部分 facts 缺嵌入 -> 重新嵌入
                missing = [f.content for f in existing_facts if not f.embedding]
                if missing:
                    reembedded = self._embedder.embed(missing)
                    for f, emb in zip(
                        [f for f in existing_facts if not f.embedding],
                        reembedded,
                        strict=True,
                    ):
                        f.embedding = emb
                        existing_embeddings.append(emb)
                        existing_entities.append(list(f.entities or []))
                        existing_texts.append(f.content or "")

        # 嵌入候选消息（2026-08-31：分块 + 进度，见 `_embed_chunked`）
        candidate_texts = [c["content"] for c in candidates]
        candidate_embs = self._embed_chunked(candidate_texts, phase="candidates")

        # novelty 检测（G3: 有 novelty_checker 用向量检索，无则 brute-force）
        # 🔴 进度 + **速率**（2026-08-31）：全量重建时 `novelty_checker=None` ⇒ 走
        # brute-force，`_is_novel` 每条要线性扫 `existing_embeddings`，而后者随本批
        # 增长 ⇒ 这一段**理论上是 O(n²)**。速率逐条下降就是它，稳定就不是它。
        # 在没有读数之前不预设结论——这正是两次盲跑没能回答的问题。
        import time as _t_nov
        _nov_t0 = _t_nov.perf_counter()
        _nov_seen = 0
        new_facts: list[Fact] = []
        for candidate, emb in zip(candidates, candidate_embs, strict=True):
            _nov_seen += 1
            if _nov_seen % 2000 == 0:
                _el = _t_nov.perf_counter() - _nov_t0
                logger.info(
                    "consolidate_novelty_progress %d/%d elapsed_s=%.1f rate=%.1f/s "
                    "kept=%d pool=%d",
                    _nov_seen, len(candidates), _el, _nov_seen / max(_el, 1e-6),
                    len(new_facts), len(existing_embeddings),
                )
            # within-batch 去重（始终需要，防同批重复）
            if not self._is_novel(emb, existing_embeddings,
                                  entities=candidate.get("entities", []),
                                  existing_entities=existing_entities,
                                  content=candidate["content"],
                                  existing_texts=existing_texts):
                self._funnel.inc(FUNNEL_DEDUP, labels={"verdict": "cosine_kill"})
                logger.debug(
                    "consolidate_skip_duplicate_batch content=%.50r",
                    candidate["content"],
                )
                continue
            # 已有 facts 去重
            if self._novelty_checker is not None:
                if not _checker_is_novel(self._novelty_checker, emb,
                                         candidate.get("entities", []),
                                         candidate["content"]):
                    self._funnel.inc(FUNNEL_DEDUP, labels={"verdict": "vector_kill"})
                    logger.debug(
                        "consolidate_skip_duplicate_vector content=%.50r",
                        candidate["content"],
                    )
                    continue
            self._funnel.inc(FUNNEL_DEDUP, labels={"verdict": "novel"})
            fact = Fact(
                id=_deterministic_fact_id(candidate["ledger_key"], candidate["content"]),
                content=candidate["content"],
                category="general",
                tags=candidate.get("tags", "origin:consolidation"),
                source_session=candidate["session_id"],
                source_user_id=candidate["user_id"],
                source_ledger_key=candidate["ledger_key"],
                logical_turn=candidate["logical_turn"],
                embedding=emb,
                kind=candidate.get("kind", "general"),
                entities=candidate.get("entities", []),
                proposal_titles=candidate.get("proposal_titles", []),
                # ADR-0021 section 2.4/3.3c: scope 默认 personal；exposure_ceiling 继承来源 Turn 敏感等级。
                scope=f"personal:{candidate['user_id']}",
                exposure_ceiling=self._ceiling_for(candidate.get("sensitivity", "normal")),
                # ── ADR-0024 §4.5 + ADR-0026 §4.2 / U4：单元键 + 条目类型 + 取代键槽位 ──
                unit_key=candidate.get("unit_key", ""),
                agent_id=candidate.get("agent_id", ""),
                item_kind=_resolve_item_kind(candidate),
                subject=(candidate.get("subject") or "").strip(),
                attribute=(candidate.get("attribute") or "").strip(),
                # 三段式 T2：轮级 topic（meta 层字段；LanceDB 不加列）
                topic=(candidate.get("topic") or "").strip(),
                # 拍板 A：轮级 keywords（meta 层字段；归属主题键原料）
                keywords=list(candidate.get("keywords") or []),
                # ── M1 新字段（v3 路径下全部落默认值，行为不变）──
                provenance=(candidate.get("provenance") or ""),
                importance_rating=int(candidate.get("importance_rating") or 0),
                valid_until=_parse_valid_until(candidate.get("valid_until")),
                # M2：修正自声明必须活到裁决那一步。少这一行，corrects 就在
                # 候选 dict → Fact 之间静默蒸发，裁决器拿到的包永远没有它
                # ——F4 那条"最高精度信号无人使用"会在新代码里原样复发。
                corrects=(candidate.get("corrects") or "").strip(),
            )
            # 双时态：`t_observed` = **摄入**时间（= created_at，重建时就是重建那一刻）；
            # `t_valid` = **业务有效起点** = 这条事实是在哪一轮被说出来的。
            #
            # 🔴 MS-11：`t_valid` 此前也被设成 created_at，于是全量重建后
            # 全库 fact 的两个时间戳都≈重建时刻、**彼此几乎相同** ——
            # 裁决的时序规则拿它们比就等于没比（"候选永远更旧 → 永不 UPDATE"）。
            # 取轮次时间才是这个字段被定义出来的意思，也让重放后的库
            # 保有真实的时间结构（as-of 查询同样依赖它）。
            # 该字段此前**只被写、从无消费方**，所以改它没有回归面。
            fact.t_observed = fact.created_at
            fact.t_valid = _parse_valid_until(candidate.get("turn_ts")) or fact.created_at
            # ADR-0026 §5.4 / U5.4：importance 公式（kind 基线 × 强度 × 引用 × 半衰减）。
            # 新产出 strength=1、ref_count=0、age=0 → importance = kind 基线。
            # 引用计数由 U7 注入命中回写；生命周期巩固（U5.5）会重算。
            # M1-5：rating>0 时取代 kind 基线（内容内在分）；0 = 未评分退回基线。
            fact.importance = compute_importance(
                fact.item_kind.value, strength=fact.strength,
                rating=fact.importance_rating)
            # L2: 记录蒸馏元数据（原文 + 模型名，供审计/对比）
            fact.source_text = candidate.get("source_text", "")
            fact.distill_model = candidate.get("distill_model", "")
            new_facts.append(fact)
            self._funnel.inc(FUNNEL_STORED,
                             labels={"item_kind": fact.item_kind.value})
            # 加入已有列表，防止后续候选重复。
            # 🔴 M0-1（复核 E1）：三个平行数组**必须同步增长**。
            # E6.5（08-05 落地）只补了前两个，于是批内第二条同文候选做标识符前置放行时，
            # 它的标识符（路径/日期）不在 existing_texts 的标识符集合里 → 判"新颖"→
            # cosine 判重整段被跳过 → 完全同文也无限入库
            # （实锤：`文件 …/p2_derived.py` 同文 ×6，创建时间集中在一分钟内）。
            existing_embeddings.append(emb)
            existing_entities.append(list(fact.entities or []))
            existing_texts.append(fact.content)

        # ── M2-2：写入时裁决（模块 2/3 的统一解，复核第 2 层）──
        #
        # 🔴 **吸收退役**（不是并存）：
        #   - `SemanticConsistencyChecker` 调用点移除。它的输出本就无人消费（F1），
        #     两条极性规则实为"任意两数不同"检测器（F2），反义词典就是验收剧本本身（F3）。
        #     2026-09-03 S6：模块连同这里的回滚通道消费点一起删除（MQ-S51 改判"机制
        #     退役"，真修复 = ADR-0006 归 0.3.0）。裁决关闭时**不再有矛盾校验退路**。
        #   - `plan_supersede_merge` 降为裁决前的**确定性快路径**（精确取代键命中直接
        #     出裁决、不调 LLM），不再独自决定入不入库。
        self.adjudication_records = []
        if new_facts and flag_enabled("BLADEX_ADJUDICATE_ENABLED") and self._adjudicator:
            new_facts = self._adjudicate(new_facts, list(existing_facts or []))
        else:
            # 关掉裁决 → 回落旧形态（supersede 独自决定）。回滚通道自 M2 起无 live
            # 读数；S6 后它不再带矛盾校验（那一段返回值本就被丢弃）。
            self.supersede_plan = None
            if new_facts and flag_enabled("BLADEX_SUPERSEDE_ENABLED"):
                plan = plan_supersede_merge(new_facts, list(existing_facts or []))
                self.supersede_plan = plan
                new_facts = plan.add
                if plan.bump or plan.invalidate:
                    logger.info(
                        "consolidate_supersede bump=%d invalidate=%d add=%d",
                        len(plan.bump), len(plan.invalidate), len(plan.add),
                    )

        logger.info(
            "consolidate_done candidates=%d new_facts=%d existing=%d distiller=%s",
            len(candidates), len(new_facts), len(existing_embeddings),
            self._distiller.model_name if self._distiller else "none",
        )
        return new_facts

    # ══════════════════════════════════════════════════════════════════════
    # M2-2：写入时裁决
    # ══════════════════════════════════════════════════════════════════════

    def _neighbors_for(self, fact: Fact, running: list[Fact]) -> list[tuple[Fact, float]]:
        """取一条候选的近邻（批内已应用 + 本轮 pending + 库内）。

        三个视图（M5-2 判据③修复后）：
          - **批内 running**：本批已应用（fast-path / no-neighbor / 循环②已裁决）
            的条目，还没写进向量库，只在内存里。
          - **本轮 pending**：转入 pending 的候选（循环②之前不在 running 里，
            此前对后续候选整体不可见——第二个洞）——由 neighbor_source 内存集覆盖。
          - **库内**：LanceDB 已有条目（增量时；全量重建无此视图，旧库不进裁决）
            ——由 neighbor_source 的 checker 包装覆盖。MS-3 句柄刷新仍在 checker 侧。

        🔴 不要在这里直接摸 `self._novelty_checker`：那是判重的对象，
        全量重建时为 None 是**判重的**正确语义，裁决借用它就是判据③的根因。
        """
        from bladex_core.adjudication import _cosine  # noqa: PLC0415

        min_sim = flag_number("BLADEX_ADJUDICATE_MIN_SIM")
        out: list[tuple[Fact, float]] = []
        if not fact.embedding:
            return out

        # 批内（已应用）
        for other in running:
            if other.id == fact.id or not other.embedding:
                continue
            sim = _cosine(fact.embedding, other.embedding)
            if sim >= min_sim:
                out.append((other, sim))

        # 本轮 pending + 库内（neighbor_source 合并视图）
        try:
            k = int(flag_number("BLADEX_ADJUDICATE_TOPK"))
            for nb, sim in self._neighbor_source.neighbors(
                    fact.embedding, k=k, min_similarity=min_sim):
                if nb.id == fact.id or any(f.id == nb.id for f, _ in out):
                    continue
                out.append((nb, sim))
        except Exception as e:  # noqa: BLE001 —— 邻居拿不到 = 无冲突可判，退化 ADD
            logger.warning("adjudicate_neighbors_failed err=%s", e)
        return out

    def _adjudicate(self, new_facts: list[Fact], existing: list[Fact]) -> list[Fact]:
        """对一批新 fact 逐条裁决，返回真正要入库的那些。

        旧条的改动（`t_invalid` / `superseded_by` / `strength`）落在**对象上**，
        并暴露 `self.adjudication_records` 供 caller 持久化 + 写 judgment 台账。
        与 supersede_plan 同一个分工：core 决策，proxy 落盘。
        """
        from datetime import UTC, datetime

        from bladex_core.adjudication import (
            AdjudicationOp,
            _observed_iso,
            apply_verdict,
            build_input,
            fast_path_verdict,
        )
        from bladex_core.funnel import FUNNEL_ADJUDICATE

        now = datetime.now(UTC)
        by_id: dict[str, Fact] = {f.id: f for f in existing}
        running: list[Fact] = list(existing)   # 批内可见（同批矛盾要能互相看见）
        kept: list[Fact] = []
        pending: list[tuple[Fact, Any]] = []   # 需要 LLM 裁决的 (fact, input)
        records: list[dict[str, Any]] = []
        #: candidate_id -> 裁决时篮子里的 fact_id（归因证据，随判决入台账）。
        #: 🔴 篮子只在循环①算得到，而 LLM 判决要到循环②才回来 —— 不存下来就丢了。
        baskets: dict[str, list[str]] = {}

        # ① 快路径：精确取代键命中，零 LLM
        # 🔴 进度 + 速率 + **邻居篮子大小**（2026-08-31，同上）：`_neighbors_for`
        # 逐条扫 `running`（:672）与 `neighbor_source._facts`（:244），两者都随本批
        # 增长 ⇒ 理论上 O(n²)。`nb_avg` 同时是另一条线索——它若很大，说明
        # `_neighbors_for` 第一段（**无上限**地收 sim≥min_sim 的全部，不是 top-k）
        # 才是主要代价，那是另一条修法。
        import time as _t_adj
        _adj_t0 = _t_adj.perf_counter()
        _adj_seen = 0
        _adj_nb = 0
        for fact in new_facts:
            _adj_seen += 1
            neighbors = self._neighbors_for(fact, running)
            _adj_nb += len(neighbors)
            if _adj_seen % 500 == 0:
                _el = _t_adj.perf_counter() - _adj_t0
                logger.info(
                    "consolidate_adjudicate_progress %d/%d elapsed_s=%.1f rate=%.1f/s "
                    "running=%d pending=%d nb_avg=%.1f",
                    _adj_seen, len(new_facts), _el, _adj_seen / max(_el, 1e-6),
                    len(running), len(pending), _adj_nb / _adj_seen,
                )
            basket = [n.id for n, _ in neighbors]
            fp = fast_path_verdict(fact, [n for n, _ in neighbors])
            if fp is not None:
                self._apply_one(fp, fact, by_id, now, kept, running, records,
                                considered=basket)
                self._funnel.inc(FUNNEL_ADJUDICATE,
                                 labels={"verdict": f"{fp.op.value}_fast"})
                continue
            if not neighbors:
                # 没有近邻 = 没有冲突可判，直接入库（不花 LLM 钱）。
                # 🔴 这条路径此前**不写台账**（没有 verdict 对象），于是它在台账里
                # 与"判了 ADD"完全无法区分。2026-08-31 补记一条 `add_no_neighbor`
                # 判决：篮子为空是**归因证据**（"当时真的没人可比"），不是无事发生。
                kept.append(fact)
                running.append(fact)
                by_id[fact.id] = fact
                records.append({
                    "candidate_id": fact.id, "op": "add", "targets": [],
                    "reason": "no_neighbor", "fallback": False,
                    "changed": [], "considered": [],
                })
                self._funnel.inc(FUNNEL_ADJUDICATE, labels={"verdict": "add_no_neighbor"})
                continue
            # M5-2 判据③：转入 pending 的候选立即进邻居源内存集——它要到循环②
            # 才进 running，不喂这里的话对后续所有候选整体不可见（密集簇变黑洞）。
            # fast-path / no-neighbor 的不喂：它们已进 running，喂了徒增重复。
            self._neighbor_source.add(fact)
            baskets[fact.id] = basket
            pending.append((fact, build_input(
                fact, neighbors, top_k=int(flag_number("BLADEX_ADJUDICATE_TOPK")),
                observed_at=_observed_iso(fact))))

        # ② 其余打包交裁决器（一次调用判多条）
        if pending:
            try:
                verdicts = self._adjudicator.adjudicate([pkg for _, pkg in pending])
            except Exception as e:  # noqa: BLE001 —— 宁多勿丢：判不出就全 ADD
                logger.warning("adjudicate_call_failed err=%s -> all add", e)
                verdicts = []
            by_cand = {v.candidate_id: v for v in verdicts}
            for fact, _pkg in pending:
                v = by_cand.get(fact.id)
                if v is None:
                    from bladex_core.adjudication import AdjudicationVerdict
                    v = AdjudicationVerdict(candidate_id=fact.id,
                                            op=AdjudicationOp.ADD,
                                            reason="no_verdict", fallback=True)
                self._apply_one(v, fact, by_id, now, kept, running, records,
                                considered=baskets.get(fact.id, []))
                label = v.op.value + ("_fallback" if v.fallback else "")
                self._funnel.inc(FUNNEL_ADJUDICATE, labels={"verdict": label})

        self.adjudication_records = records
        _ops = Counter(r["op"] for r in records)
        logger.info("consolidate_adjudicated candidates=%d kept=%d ops=%s llm_calls=%d",
                    len(new_facts), len(kept), dict(_ops), 1 if pending else 0)
        return kept

    def _apply_one(self, verdict, fact, by_id, now, kept, running, records,
                   *, considered: list[str] | None = None) -> None:
        """应用一条裁决 + 记账（供 caller 持久化与写台账）。

        `considered` = 裁决时篮子里的 fact_id，随判决入台账（2026-08-31）。
        它是**归因证据**：没有它，"兄弟当时不在篮子里"与"在篮子里仍判 ADD"
        在台账上长得一模一样（08-31 读数 44.3% 卡在这一点）。
        """
        from bladex_core.adjudication import apply_verdict

        keep, changed = apply_verdict(verdict, fact, by_id, now=now)
        if keep:
            kept.append(fact)
            running.append(fact)
            by_id[fact.id] = fact
        else:
            # M5-2 判据③：NOOP 丢弃的候选从邻居源摘除（同实例跨批不再可见）
            self._neighbor_source.discard(fact.id)
        records.append({
            "candidate_id": fact.id,
            "op": verdict.op.value,
            "targets": list(verdict.target_fact_ids),
            "reason": verdict.reason,
            "fallback": bool(verdict.fallback),
            # 被改动的旧条：caller 要 upsert 它们的 meta
            "changed": changed,
            "considered": list(considered or []),
        })

    # ══════════════════════════════════════════════════════════════════════
    # M1-2：v4 候选装配（两级路由 = 先分流、话语路再统一调用）
    # ══════════════════════════════════════════════════════════════════════

    def _turn_channel(self, turn: ConversationTurn) -> tuple[str, str]:
        """取这一轮的 assistant 正文与它的通道名（已剥信封）。

        结论与产出**互斥**（fact.py:44-52）：无 tool call = 结论（最终回答），
        有 tool call = 产出（模型调工具前写的说明/分析/发现）。
        归一层（rebuild）已按此规则二选一赋值，这里只需按非空取。
        """
        concl = strip_envelopes(turn.assistant_conclusion or "")[0].strip()
        if concl and len(concl) >= _MIN_CONCLUSION_CHARS:
            return concl, Provenance.CONCLUSION.value
        prog = strip_envelopes(turn.assistant_progress or "")[0].strip()
        if prog and len(prog) >= _MIN_PROGRESS_CHARS:
            return prog, Provenance.PROGRESS.value
        return "", ""

    def _collect_candidates_v4(
        self, turns: list[ConversationTurn]
    ) -> list[dict[str, Any]]:
        """v4 装配：分流（话语/规则文件/文档）→ 话语路一轮一次统一调用。

        与 v3 的三条独立流水线（user / conclusion / progress 各自蒸各自的）相比，
        这里一轮只调一次、user 与 assistant 同包 —— 模型这次**同时看见**
        "问了什么"和"做了什么"。v3 里这两半互不知情，产出因此大量是
        「用户询问 XXX」而丢掉真正的结论。

        通道信息不丢：合流后由产出侧标 `provenance`（M1-1 的 `source` 槽位），
        并**同时**写 `tags="origin:<通道>"`（Q4：`origin:progress` 是待验证项 A3
        的判据，不能在它被验证之前拆掉）。
        """
        from bladex_core.distill_routing import InputRoute, classify_input
        from bladex_core.distillation import DistillTurnInput, build_context_digest

        candidates: list[dict[str, Any]] = []
        self.document_entries = []
        # MS-4：ledger_key -> (event, detail)。蒸馏产出的任务事件分类，
        # 供 rebuild 折叠进 Matter 卡 lifecycle（**优先于**规则匹配兜底）。
        # 与 document_entries 同一分工：core 决策、proxy 落盘——挂在 Fact 上会
        # 连带持久化进 meta，给一个纯瞬态值污染 schema。
        self.turn_events = {}
        # T-A（2026-08-13，HANDOFF-20260813 §2）：v4/v5 主路径接上并发。
        # 此前 `--concurrency` 只接在 v3 兜底的 `_bulk_distill`（生产不走），
        # v4 一直逐 payload 串行调 LLM —— 全量重建 6.2h / bench 排空撞穿
        # drain_timeout 全由此来。修法照 v3 同款两阶段：
        #   Pass1 构造全部 payload（纯 CPU，memo 一次性查完）
        #   Pass2 去重后并发蒸馏成 map（`_bulk_distill_turn_payloads`）
        #   Pass3 按**原顺序**装配只查表 → candidates 顺序不变 ⇒ fact id 不变（G6）。
        # 并发<=1 时不走两阶段：构造与蒸馏仍逐轮交错，memo 中途标记语义**逐字不变**
        # （零回归，与 v3 `_bulk_distill` 返回 None 走原路径同款纪律）。
        two_phase = self._distiller is not None and self._distill_concurrency > 1
        plan: list[tuple[ConversationTurn, str, list[Any]]] = []
        # 分流分布对外可读（观测面）：附录 D.4 断言"真·超长话语 <5%"，
        # 这个计数器是它在真实语料上的读数。放属性而不是只打日志——
        # 日志在库代码里走 stdlib logging，在 proxy 进程根本不落盘（ADR-0020 T2 同款盲区）。
        self.route_counts: Counter[str] = Counter()
        route_counts = self.route_counts
        _env_kinds: Counter[str] = Counter()

        for turn in turns:
            # ① 剥信封 + 分流：这一轮的 user 消息里，哪些是话语、哪些是文档
            discourse: list[str] = []
            for msg in turn.user_messages:
                clean, kinds = strip_envelopes(msg)
                _env_kinds.update(kinds)
                clean = clean.strip()
                if not clean:
                    continue
                route = classify_input(clean, discourse_max_chars=_MAX_CONTENT_CHARS)
                route_counts[route.value] += 1
                if route is InputRoute.DISCOURSE:
                    discourse.append(clean)
                elif route is InputRoute.RULEFILE:
                    # 🔴 **只分流，不产条目**。规则文件正文的捕获早已归 E7.3
                    # （`update_profiles_from_turn` 读**未剥离**的 request_messages
                    # 调 `_capture_rule_file_bodies`，按 agent + hash 存副本）。
                    # 这条路的价值在于**把它从话语路上拦下来**——失效链 A 的根修是
                    # "别让规则文件走用户陈述蒸馏"，不是"再造一条 profile_obs 事实"。
                    # 产一条摘要式条目只是重复 E7.3 的活，还给库添一条低价值噪声
                    # （X3：每发现一个补偿机制，先问生产端能不能不产生）。
                    logger.debug("distill_route_rulefile ledger_key=%s chars=%d",
                                 turn.ledger_key, len(clean))
                else:
                    self._queue_document(turn, clean)

            # ② 本轮的"新内容" = 最后一条话语；之前的构成上下文。
            # user_messages 带着完整历史（agent 每轮回传），历史条目在它们**自己那轮**
            # 已经被蒸过——再蒸一遍只是给判重制造压力，还要按历史长度重复付 LLM 费用。
            # v3 对每条历史消息都蒸一次，正是调用量的大头。
            #
            # 🔴 **但那个前提 2026-08-08 被真实数据证伪**（32% 从未当过最后一条）。
            #
            # 「历史条目在它们自己那轮已经被蒸过」要求**每一轮都独立入库并被处理**。
            # 真实 Hub 里四种情形让中间消息永远轮不到当最后一条：
            # 会话中断后 agent 一次带回 N 条历史 / 上下文压缩后重启 /
            # agent 首次接入就带着既有对话 / 那一轮 failed 或被 aux 整轮过滤。
            #
            # 全库实测（`measure_discourse_undersampling.py`，2537 轮）：
            # distinct 话语 347 条，**111 条（32.0%）从未当过 `discourse[-1]`**，
            # `hermes:c7047465` 更是 10/10 全丢。而丢的不是寒暄，是
            # 「AMD Helios 会不会威胁英伟达」「深度调查 unabyssapp 与 BladeX 对比」
            # 这类驱动过整条研究线的实质提问 —— 每一条都是真实的记忆损失，
            # 且**没有任何报错**，这是它潜伏至今的原因。
            #
            # 修法：意图不变（别重复蒸历史），但取的量从「最后一条」改成
            # **「从未蒸过的」** —— 「最后一条」≠「新出现的」，这是原实现的错。
            # 判重靠**纯内容维度**的 memo（`_discourse_seen`），不能用蒸馏台账：
            # 台账 key 故意含 context_digest（同一句话在不同脉络下该蒸出不同产物），
            # 拿它当 memo 会次次 miss。
            new_text = discourse[-1] if discourse else ""
            prior = discourse[:-1]
            context = build_context_digest(prior)
            # 补采：prior 里从未蒸过的那些。各自带**自己位置之前**的上下文，
            # 与它们"当时作为新消息"应有的脉络一致。
            missed: list[tuple[str, str]] = []   # (text, context_digest)
            if prior and self._discourse_memo_available():
                for i, text in enumerate(prior):
                    if self._discourse_seen(text) is False:   # 明确没蒸过才补
                        missed.append((text, build_context_digest(prior[:i])))
            # 注：`build_context_digest` 支持 Matter 卡 summary 优先，但这里传不了——
            # 归属发生在蒸馏**之后**（先有 fact 才能归属），此刻还不知道属于哪张卡。
            # 更要紧的是：跨批次可见的 Matter 会让同一轮在不同 rebuild 批次里算出不同
            # digest → 台账 key 漂移 → "复跑零现蒸"破功。故 digest 只取该轮自带的历史。

            assistant_text, channel = self._turn_channel(turn)
            # 🔴 主 payload 也要查 memo，否则「同一条末尾消息跨多轮重复出现」
            # （工具循环的多个 roundtrip 共享同一句用户提问）会被反复蒸。
            # 只在**两侧都无新内容**时跳过：assistant 正文逐轮不同，
            # 它有内容就仍要蒸（M1 的通道合一价值在于模型同时看见问与答）。
            if (new_text and assistant_text == ""
                    and self._discourse_seen(new_text) is True):  # 明确蒸过才跳
                new_text = ""
            if not new_text and not assistant_text and not missed:
                continue

            # ③ 分段：话语与 assistant 正文各自按预算切（超长的少数情形）
            user_segs = segment_long_text(new_text, _MAX_CONTENT_CHARS) if new_text else []
            asst_segs = (segment_long_text(assistant_text, _ASSISTANT_SEGMENT_CHARS)
                         if assistant_text else [])

            payloads: list[DistillTurnInput] = []
            if user_segs or asst_segs:      # 两侧都空（只剩补采）时不发空 payload
                payloads.append(DistillTurnInput(
                    context_digest=context,
                    turn_time=turn.timestamp,
                    user_text=user_segs[0] if user_segs else "",
                    assistant_text=asst_segs[0] if asst_segs else "",
                    assistant_role=channel,
                ))
            # 补采：每条从未蒸过的历史话语单独走一次（user-only）。
            # 放在主 payload **之后** —— 主 payload 承载 M1 的「通道合一」
            # （模型同时看见问了什么 + 做了什么），那是这一版的核心价值，不能被稀释。
            for text, ctx in missed:
                for seg in segment_long_text(text, _MAX_CONTENT_CHARS):
                    payloads.append(DistillTurnInput(
                        context_digest=ctx, turn_time=turn.timestamp, user_text=seg))

            # 余下的段各自单独走一次（结论/产出长文本分段，D2/D3 的欠采侧）
            for seg in user_segs[1:]:
                payloads.append(DistillTurnInput(
                    context_digest=context, turn_time=turn.timestamp, user_text=seg))
            for seg in asst_segs[1:]:
                payloads.append(DistillTurnInput(
                    context_digest=context, turn_time=turn.timestamp,
                    assistant_text=seg, assistant_role=channel))

            if not payloads:
                continue
            if two_phase:
                # Pass1 只收集，不蒸——蒸馏推迟到 Pass2 并发批量做。
                plan.append((turn, channel, payloads))
                continue
            for payload in payloads:
                out = self._distill_turn(payload)
                self._consume_distill_v4(turn, payload, out, channel, candidates)

        if two_phase and plan:
            # Pass2：唯一 payload 并发蒸馏成 map。
            result_map = self._bulk_distill_turn_payloads(
                [p for _t, _c, ps in plan for p in ps])
            # Pass3：按原顺序装配只查表（G6：candidates 顺序与串行逐字一致）。
            for turn, channel, payloads in plan:
                for payload in payloads:
                    out = result_map.get(
                        (payload.context_digest, payload.cache_text()))
                    if out is None:
                        # 防御兜底：map 缺项直调（不应发生；发生了也与串行同形）。
                        logger.warning(
                            "bulk_distill_turns_map_miss ledger_key=%s len=%d",
                            turn.ledger_key, len(payload.cache_text()))
                        out = self._distill_turn(payload)
                    self._consume_distill_v4(turn, payload, out, channel, candidates)

        if _env_kinds:
            logger.info("consolidate_envelope_stripped kinds=%s", dict(_env_kinds))
        logger.info("consolidate_routes %s docs=%d",
                    dict(route_counts), len(self.document_entries))
        self._count_candidate_channels(candidates)
        return candidates

    def _embed_chunked(self, texts: list[str], *, phase: str) -> list[list[float]]:
        """分块嵌入 + 每块报进度。**行为等价**：同样的文本、同样的顺序、同样的向量。

        🔴 2026-08-31 事故驱动（全量重建两次卡在同一处，MQ-A28/A29）。
        原来是 `self._embedder.embed(candidate_texts)` **一次性整批**，代价三条：

        1. **IPC 后端结构上跑不通**：客户端超时写死 120s、整批不分块，
           一次请求装下 6 万段话语必然超时（第一次重建 12:56→12:58 正好 120s）。
        2. **没有进度**：`consolidate_routes` 之后到 `consolidate_adjudicated` 之前
           一行日志都没有，而 `--full` 连心跳都关着（`consolidator.py:464`）
           ⇒ **"在跑"与"卡死"在输出上完全不可区分**，第二次重建就死在这个不可区分上。
        3. **打不断**：第二次 `^C` 无反应（原因至今**未定位**——`^C` 打不断纯 Python
           循环这条与读码结论矛盾，两个假设都被推翻了，见 MQ-A29）。分块至少
           保证每块之间有一个确定的字节码窗口。

        **速率是本函数最值钱的产出**：每块都报 `rate`（条/秒）。
        速率**稳定** ⇒ 复杂度线性，只是慢，等就行；
        速率**逐块下降** ⇒ 有随规模增长的项（二次），等下去没有意义。
        这是"卡死还是在跑"之外，第二个必须能分辨的问题。
        """
        import time as _time

        if not texts:
            return []
        size = max(1, int(flag_number("BLADEX_EMBED_BATCH")))
        out: list[list[float]] = []
        total = len(texts)
        t0 = _time.perf_counter()
        for start in range(0, total, size):
            chunk = texts[start:start + size]
            t_chunk = _time.perf_counter()
            out.extend(self._embedder.embed(chunk))
            now = _time.perf_counter()
            logger.info(
                "consolidate_embed_progress phase=%s %d/%d chunk_s=%.1f "
                "elapsed_s=%.1f rate=%.1f/s eta_s=%.0f",
                phase, len(out), total, now - t_chunk, now - t0,
                len(out) / max(now - t0, 1e-6),
                (total - len(out)) / max(len(out) / max(now - t0, 1e-6), 1e-6),
            )
        return out

    def _distill_turn(self, payload: Any) -> Any:
        """调 v4 统一蒸馏（失败降级不抛，与 v3 同款语义）。"""
        return self._distiller.distill_turn(payload)

    def _consume_distill_v4(
        self,
        turn: ConversationTurn,
        payload: Any,
        out: Any,
        channel: str,
        candidates: list[dict[str, Any]],
    ) -> None:
        """一条 payload 蒸馏产出的装配消费（串行 / 两阶段共用同一段代码）。

        T-A：把它从串行循环里抽出来，是并发/串行 **fact 产出逐字一致**的结构保证
        ——两条路径消费的是同一个函数，差异只剩"蒸馏何时发生"。
        """
        self._count_distill(out)
        if getattr(out, "failed", False):
            self._record_distill_failure(turn.ledger_key, out)
            return
        # MS-4：轮级事件（一轮一个；同一轮多次调用取首个非空）
        for _df in out.facts:
            if _df.event and turn.ledger_key not in self.turn_events:
                self.turn_events[turn.ledger_key] = (
                    _df.event, _df.content[:120])
                break
        # 🔴 **成功才标记**。失败就标 = 一次上游抖动让这条话语永久跳过，
        # 而症状是"库里少了点东西"、没有任何报错 —— 正是本次要根治的形态。
        if payload.user_text:
            self._mark_discourse(payload.user_text)
        candidates.extend(
            self._facts_to_candidates(turn, out, payload, channel))

    def _bulk_distill_turn_payloads(
        self, payloads: list[Any],
    ) -> dict[tuple[str, str], Any]:
        """T-A Pass2：v4/v5 payload 并发预蒸馏，(context_digest, cache_text) → 产出。

        去重 key **有意不含 turn_time**（HANDOFF-20260813 §2）：同文同上下文的重复
        话语（工具循环 roundtrip 共享同一句提问 / 跨轮补采）在串行路径本会被
        `_discourse_seen` memo 中途标记跳过；两阶段下 memo 在 Pass1 一次性查完、
        看不见本次 run 的中间标记，若 key 再带 turn_time，这类重复会放大成
        每轮一次 LLM 调用。收敛成一次（取首个出现的 payload 去蒸），产出的少量
        重复候选由 台账 write-if-absent + 批内 novelty 判重兜住 ——
        与 v3 `_bulk_distill` 的既有行为一致。

        失败语义与串行同形：worker 内 `distill_turn` 降级不抛（LLMDistiller 契约）；
        真抛了（测试桩/编程错误）则 `fut.result()` 原样上抛，与串行一致。
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        uniq: dict[tuple[str, str], Any] = {}
        for p in payloads:
            uniq.setdefault((p.context_digest, p.cache_text()), p)
        if not uniq:
            return {}
        total = len(uniq)
        done = 0
        logger.info(
            "bulk_distill_turns_start uniq=%d payloads=%d concurrency=%d",
            total, len(payloads), self._distill_concurrency)
        result: dict[tuple[str, str], Any] = {}
        with ThreadPoolExecutor(max_workers=self._distill_concurrency) as ex:
            futs = {ex.submit(self._distill_turn, p): k for k, p in uniq.items()}
            for fut in as_completed(futs):
                result[futs[fut]] = fut.result()
                done += 1
                self._report(phase="distill", done=done, total=total)
                if done % 50 == 0 or done == total:
                    logger.info("bulk_distill_turns_progress %d/%d", done, total)
        return result

    # ── 已蒸话语 memo（2026-08-08，欠采修复）──────────────────────────────
    #
    # 存储归 proxy（P2 meta），core 只经 distiller 这条既有注入面访问 ——
    # 与 `_ledger_get` 同款能力探测：实现方没有这两个方法就整体降级回
    # 「只蒸最后一条」的旧行为，**逐字不变**（测试替身/旧部署不受影响）。

    def _discourse_memo_available(self) -> bool:
        return callable(getattr(self._distiller, "discourse_seen", None)) and \
            callable(getattr(self._distiller, "mark_discourse", None))

    def _discourse_seen(self, text: str) -> bool | None:
        """True=蒸过 / False=没蒸过 / **None=不知道**（无能力或读失败）。

        🔴 三态不是过度设计，是被一个真实 bug 逼出来的：最初用两态、失败返回
        True（"当作蒸过"），结果 memo 故障时主 payload 把**最后一条也跳了** ——
        比旧行为更差。而两条消费路径对"不知道"的正确处置恰好相反：

            补采路径：不知道 → **不补**（宁可少蒸，等于旧行为）
            主 payload：不知道 → **照蒸**（旧行为就是无条件蒸最后一条）

        用 True/False 两态没法同时满足这两条，只能是三态。
        """
        fn = getattr(self._distiller, "discourse_seen", None)
        if not callable(fn):
            return None
        try:
            return bool(fn(text))
        except Exception:  # noqa: BLE001 —— memo 故障不该阻断蒸馏
            logger.warning("discourse_memo_read_failed len=%d", len(text or ""))
            return None

    def _mark_discourse(self, text: str) -> None:
        fn = getattr(self._distiller, "mark_discourse", None)
        if not callable(fn):
            return
        try:
            fn(text)
        except Exception:  # noqa: BLE001
            logger.warning("discourse_memo_write_failed len=%d", len(text or ""))

    def _facts_to_candidates(
        self, turn: ConversationTurn, out: Any, payload: Any, channel: str,
    ) -> list[dict[str, Any]]:
        """DistillOutput → 候选 dict 列表（provenance 与 tags 双写，Q4）。"""
        effective_model = ("passthrough" if out.model_name == "passthrough"
                           else self._distiller.model_name)
        proposal_titles = [p.title for p in out.matter_proposals if p.title]
        result: list[dict[str, Any]] = []
        for dfact in out.facts:
            content = dfact.content.strip()
            if len(content) < 3:
                continue
            # provenance：产出侧标了就用它；没标则按这条 payload 的形态推断
            # （只有 assistant 正文 → assistant 通道；否则用户直述）。
            prov = dfact.provenance or (
                channel if (payload.assistant_text and not payload.user_text)
                else Provenance.USER_DIRECT.value)
            result.append({
                "content": content,
                "source_text": payload.cache_text(),
                "context_digest": payload.context_digest,
                "unit_key": turn.unit_key,
                "agent_id": turn.agent_id,
                "item_kind": dfact.item_kind,
                "subject": dfact.subject,
                "attribute": dfact.attribute,
                "distill_model": effective_model,
                "kind": dfact.kind,
                "entities": list(dfact.entities),
                "proposal_titles": proposal_titles,
                # Q4：provenance 是新判据，tags 是待验证项 A3 的**现有**判据。
                # 双写过渡——不能在一条待验证项被验证之前把它的判据拆掉。
                "provenance": prov,
                "tags": _merge_tags(prov, getattr(dfact, "tags", "")),
                # 三段式 T2：轮级 topic 继承式写进该轮每条 candidate（一轮一 topic）。
                # 读 dfact.topic（台账可穿越）而非 out.topic（命中路径会丢）。
                "topic": getattr(dfact, "topic", "") or "",
                # 拍板 A（2026-08-14）：轮级 keywords 同款继承（台账可穿越；
                # 历史台账 = [] 由 topic_keys.derive_topic_keys 派生兜底）。
                "keywords": list(getattr(dfact, "keywords", []) or []),
                "importance_rating": dfact.importance,
                "valid_until": dfact.valid_until,
                "corrects": dfact.corrects,
                "event": dfact.event,
                "session_id": turn.session_id,
                "user_id": turn.user_id,
                "ledger_key": turn.ledger_key,
                "logical_turn": turn.logical_turn,
                "sensitivity": turn.sensitivity,
                # MS-11：这一轮**发生**的时间（不是被摄入的时间）
                "turn_ts": turn.timestamp,
            })
        return result

    def _queue_document(self, turn: ConversationTurn, text: str) -> None:
        """文档路（E7.1）：排队交给调用方建**文件内容索引**，不产条目。

        为什么不在这里直接建索引：core 不该知道 LanceDB / files 表的存在
        （agent 中立）。与 `supersede_plan` 同一个模式——core 决策、proxy 持久化。

        为什么不产 assertion/preference：文档内容不是用户断言。
        把文档切成六句假话语去蒸，正是失效链 A/B 的成因。
        """
        import hashlib

        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        self.document_entries.append({
            "path": f"paste://{digest}",
            "content": text,
            "agent_id": turn.agent_id,
            "user_id": turn.user_id,
            "ledger_key": turn.ledger_key,
            "ts": turn.timestamp,
        })

    def _count_candidate_channels(self, candidates: list[dict[str, Any]]) -> None:
        """M4-2 漏斗第一段：候选按通道计数。"""
        from collections import Counter as _C

        from bladex_core.funnel import FUNNEL_CANDIDATES

        for _tag, _n in _C(
            _channel_of(c.get("tags", ""))
            for c in candidates
        ).items():
            self._funnel.inc(FUNNEL_CANDIDATES, _n, labels={"channel": _tag})

    def _collect_candidates(
        self, turns: list[ConversationTurn]
    ) -> list[dict[str, Any]]:
        """从 turns 收集候选 user 消息（信封剥离 + 必要时分段）。

        有 distiller 时：先做蒸馏输入预处理，再蒸馏成原子事实。
        蒸馏后每条事实独立成为候选（一条消息可产出 0-N 条事实）。
        无 distiller 时：透传原文（向后兼容，测试用）。

        ADR-0024 T3：预处理从"整条丢弃"换成 `prepare_distill_inputs`——
        剥掉 agent 内部协议信封，剥完再判长度，仍超长才分段。归档 Memory Hub 全量实测：
        claude-code 超长 user 消息 98.21% 的字符是信封，剥掉后 91% 落回
        20–500 窗口。旧逻辑那句 `if not (_MIN <= len <= _MAX): continue`
        正是 claude-code 提取率只有 7.5% 的直接原因。
        """
        # MS-1（复核 D4）：本轮蒸馏**失败**的 turn 的 ledger_key。
        # 调用方（memory_index.rebuild）据此把这些 turn 排除在消费标记之外，
        # 下轮重试（失败不入台账，重试零成本）。
        # 每次 _collect_candidates 重置——它是"本批"的结论，不是累计量。
        self.failed_ledger_keys: set[str] = set()
        self.parse_failed_ledger_keys: set[str] = set()
        self.document_entries: list[dict[str, Any]] = []

        # M1-2：蒸馏器支持 v4 统一调用 + 开关开着 → 走两级路由；
        # 否则回落 v3 三条独立流水线（旧 mock / 第三方实现零改造兼容，行为逐字不变）。
        if (flag_enabled("BLADEX_DISTILL_V4")
                and self._distiller is not None
                and getattr(self._distiller, "distill_turn", None) is not None):
            return self._collect_candidates_v4(turns)

        candidates: list[dict[str, Any]] = []
        # 预处理一遍拿总数（进度日志用）；纯字符串操作，成本可忽略
        _prepared: list[tuple[ConversationTurn, str]] = []
        _env_kinds: Counter[str] = Counter()
        _dropped_pure_envelope = 0
        for t in turns:
            for m in t.user_messages:
                texts, kinds = prepare_distill_inputs(
                    m, _MIN_CONTENT_CHARS, _MAX_CONTENT_CHARS,
                )
                _env_kinds.update(kinds)
                if kinds and not texts:
                    _dropped_pure_envelope += 1
                _prepared.extend((t, x) for x in texts)
        if _env_kinds:
            logger.info(
                "consolidate_envelope_stripped kinds=%s pure_envelope_dropped=%d",
                dict(_env_kinds), _dropped_pure_envelope,
            )
        _total_msgs = len(_prepared)
        _distilled_count = 0

        # 2026-08-03（sync CLI）：并发预蒸馏——唯一文本先并发算好，下面的装配循环
        # 顺序不变、只查表（G6 确定性保持）。并发=1/无蒸馏器 → maps=None 走原路径。
        # ADR-0028 E1.4：预蒸馏的文本必须与 _collect_*_candidates 用的**同一形态**
        # （剥信封后），否则 ① 查表恒 miss、② 白花 LLM 钱蒸一遍信封正文。
        _concl_texts: list[str] = []
        for t in turns:
            _c = strip_envelopes(t.assistant_conclusion or "")[0].strip()
            if _c and len(_c) >= _MIN_CONCLUSION_CHARS:
                _concl_texts.append(_c)
            _p = strip_envelopes(t.assistant_progress or "")[0].strip()
            if _p and len(_p) >= _MIN_PROGRESS_CHARS:
                _concl_texts.append(_p)
        _maps = self._bulk_distill([c for _, c in _prepared], _concl_texts)
        _user_map = _maps[0] if _maps is not None else None
        self._concl_map = _maps[1] if _maps is not None else None

        for turn, content in _prepared:
            if self._distiller is not None:
                _distilled_count += 1
                if _total_msgs > 5 and (_distilled_count % 10 == 1 or _distilled_count == _total_msgs):
                    logger.info(
                        "consolidate_distill_progress %d/%d turns=%d",
                        _distilled_count, _total_msgs, len(turns),
                    )
                if _user_map is None:
                    self._report(phase="distill", done=_distilled_count, total=_total_msgs)
                if _user_map is not None and content in _user_map:
                    output = _user_map[content]
                else:
                    output = self._distiller.distill(content)
                self._count_distill(output)
                if getattr(output, "failed", False):
                    # MS-1：这一轮的事实是被上游/解析吃掉的，不是"真没有"。
                    self._record_distill_failure(turn.ledger_key, output)
                effective_model = (
                    "passthrough" if output.model_name == "passthrough"
                    else self._distiller.model_name
                )
                # ADR-0018 §3.1: 一条消息的多条 facts 共享该消息的 proposals
                proposal_titles = [p.title for p in output.matter_proposals if p.title]
                for dfact in output.facts:
                    if len(dfact.content.strip()) < 3:
                        continue
                    candidates.append({
                        "content": dfact.content.strip(),
                        "source_text": content,
                        "unit_key": turn.unit_key,
                        "agent_id": turn.agent_id,
                        "item_kind": dfact.item_kind,
                        "subject": dfact.subject,
                        "attribute": dfact.attribute,
                        "distill_model": effective_model,
                        "kind": dfact.kind,
                        "entities": list(dfact.entities),
                        "proposal_titles": proposal_titles,
                        # 🔴 2026-08-20 修：本（v3 回落）路径此前不带 dfact.tags，
                        # lane 标在 BLADEX_DISTILL_V4=0 时静默消失 → lane 硬分流
                        # （MQ-S27）跟着失效。v4 路径的 _to_candidates 一直带
                        # （_merge_tags），两条路径对同一字段必须一致——
                        # 与"离线/在线口径不一致"（08-14 泰安四开自检不出）同型。
                        "tags": _merge_tags("", getattr(dfact, "tags", "")),
                        "session_id": turn.session_id,
                        "user_id": turn.user_id,
                        "ledger_key": turn.ledger_key,
                        "logical_turn": turn.logical_turn,
                        "sensitivity": turn.sensitivity,
                    })
            else:
                candidates.append({
                    "content": content,
                    "source_text": "",
                    "unit_key": turn.unit_key,
                    "agent_id": turn.agent_id,
                    "distill_model": "",
                    "session_id": turn.session_id,
                    "user_id": turn.user_id,
                    "ledger_key": turn.ledger_key,
                    "logical_turn": turn.logical_turn,
                    "sensitivity": turn.sensitivity,
                })

        # ADR-0019 P2：结论蒸馏——任务的 final assistant 回答里沉淀着
        # 被装配器降解掉的 tool 证据的关键信息（结论即蒸馏的守恒闭环）。
        # 🔴 必须独立遍历 turns，不能挂在上面的 _prepared 循环里：
        #   ① 一轮只该收一次结论（挂 _prepared 会按 user 消息数重复收）；
        #   ② user 消息全是纯信封的轮次不在 _prepared 里，但它**仍有结论**——
        #      结论通道扛着 66% 的 Fact 产出（T0 实测），漏掉就是把修好的一条腿
        #      换成打断另一条（ADR-0024 §4.5「结论通道只可加不可替」）。
        for turn in turns:
            candidates.extend(self._collect_conclusion_candidates(turn))
            candidates.extend(self._collect_progress_candidates(turn))
            # 🔴 拍板 3：file_ref 旧通道**已退役**（`_collect_file_ref_candidates` 已删；
            # 僵尸开关 `BLADEX_FILE_REF_ENABLED` 于 2026-09-03 S4 一并删除）。裸路径是零信息量的——实测
            # 30 条随机 query 的 top-10 被它们占 20.7%，相似度 mean 0.880
            # （"对任何问句都中等相似"的典型形状）。正确的可检索形态是索引**内容**，
            # 由 E7.1 文件内容索引接管（summary + keywords + 独立通道配额）。

        # M4-2：候选按通道计数。漏斗的第一段——它为 0 而 turns 非 0，
        # 说明问题在装配（信封/长度/分流），不在蒸馏。
        from bladex_core.funnel import FUNNEL_CANDIDATES
        from collections import Counter as _C

        for _tag, _n in _C(
            _channel_of(c.get("tags", ""))
            for c in candidates
        ).items():
            self._funnel.inc(FUNNEL_CANDIDATES, _n, labels={"channel": _tag})
        return candidates

    def _collect_progress_candidates(
        self, turn: ConversationTurn,
    ) -> list[dict[str, Any]]:
        """从工作产出（assistant_progress）蒸馏候选（2026-07-29 新增通道）。

        结论通道要求"本轮无 tool call"，编程 agent 几乎永远不满足
        （实测 codex 37 轮里 35 轮带 tool call）。这条通道收的是模型在**调工具之前**
        写的那段说明——"我发现 X 不存在，改用 Y"、"根因是 Z" 这类过程发现。

        与结论通道共用蒸馏器与 prompt（同样是"面向结果/产出/决定"的抽取），
        但 tags 标 `origin:progress` 以便审计与后续按通道调权。
        """
        # ADR-0028 E1.4（F5 止血）：长度判定**之前**先剥信封（含 <bladex-memory> 回声），
        # 否则"注入块回声 + 几句话"会因总长度达标而把自己注入的记忆蒸馏回 Memory Index（自反馈闭环）。
        progress, _envelope_kinds = strip_envelopes(turn.assistant_progress or "")
        progress = progress.strip()
        if not progress or len(progress) < _MIN_PROGRESS_CHARS:
            return []
        if self._distiller is None:
            return []
        distill_fn = getattr(self._distiller, "distill_conclusion", None)
        if distill_fn is None:
            return []

        _cm = getattr(self, "_concl_map", None)
        if _cm is not None and progress in _cm:
            output = _cm[progress]
        else:
            output = distill_fn(progress)
        self._count_distill(output)
        if getattr(output, "failed", False):
            # MS-1：结论/产出通道的失败同样要计入（它扛着 66% 的 Fact 产出，
            # 这条通道失败而 turn 照常被消费 = 那一轮的产出永久丢失）。
            self._record_distill_failure(turn.ledger_key, output)
        if not output.facts:
            return []
        effective_model = (
            "passthrough" if output.model_name == "passthrough"
            else self._distiller.model_name
        )
        proposal_titles = [p.title for p in output.matter_proposals if p.title]
        out: list[dict[str, Any]] = []
        for dfact in output.facts:
            if len(dfact.content.strip()) < 3:
                continue
            out.append({
                "content": dfact.content.strip(),
                "source_text": progress,
                "unit_key": turn.unit_key,
                "agent_id": turn.agent_id,
                "item_kind": dfact.item_kind,
                "subject": dfact.subject,
                "attribute": dfact.attribute,
                "distill_model": effective_model,
                "kind": dfact.kind,
                "entities": list(dfact.entities),
                "proposal_titles": proposal_titles,
                "tags": "origin:progress",
                "session_id": turn.session_id,
                "user_id": turn.user_id,
                "ledger_key": turn.ledger_key,
                "logical_turn": turn.logical_turn,
                "sensitivity": turn.sensitivity,
            })
        if out:
            logger.info(
                "progress_distilled facts=%d ledger_key=%s progress_len=%d",
                len(out), turn.ledger_key, len(progress),
            )
        return out

    def _collect_conclusion_candidates(
        self, turn: ConversationTurn,
    ) -> list[dict[str, Any]]:
        """从任务结论（assistant_conclusion）蒸馏候选（ADR-0019 P2）。

        条件：有结论文本 + 有 LLM 蒸馏器且实现 distill_conclusion（可选能力，
        getattr 探测——旧 mock 零改造兼容）。透传蒸馏器返回空（不污染）。
        """
        # ADR-0028 E1.4（F5 止血）：长度判定**之前**先剥信封（同 _collect_progress_candidates）。
        conclusion, _envelope_kinds = strip_envelopes(turn.assistant_conclusion or "")
        conclusion = conclusion.strip()
        if not conclusion or len(conclusion) < _MIN_CONCLUSION_CHARS:
            return []
        if self._distiller is None:
            return []
        distill_fn = getattr(self._distiller, "distill_conclusion", None)
        if distill_fn is None:
            logger.debug(
                "conclusion_distill_skip reason=distiller_lacks_capability model=%s",
                self._distiller.model_name,
            )
            return []

        _cm = getattr(self, "_concl_map", None)
        if _cm is not None and conclusion in _cm:
            output = _cm[conclusion]
        else:
            output = distill_fn(conclusion)
        self._count_distill(output)
        if getattr(output, "failed", False):
            # MS-1：同 progress 通道 —— 结论蒸馏失败也算这一轮失败。
            self._record_distill_failure(turn.ledger_key, output)
        if not output.facts:
            return []
        effective_model = (
            "passthrough" if output.model_name == "passthrough"
            else self._distiller.model_name
        )
        proposal_titles = [p.title for p in output.matter_proposals if p.title]
        out: list[dict[str, Any]] = []
        for dfact in output.facts:
            if len(dfact.content.strip()) < 3:
                continue
            out.append({
                "content": dfact.content.strip(),
                "source_text": conclusion,
                "unit_key": turn.unit_key,
                "agent_id": turn.agent_id,
                "item_kind": dfact.item_kind,
                "subject": dfact.subject,
                "attribute": dfact.attribute,
                "distill_model": effective_model,
                "kind": dfact.kind,
                "entities": list(dfact.entities),
                "proposal_titles": proposal_titles,
                "tags": "origin:conclusion",
                "session_id": turn.session_id,
                "user_id": turn.user_id,
                "ledger_key": turn.ledger_key,
                "logical_turn": turn.logical_turn,
                "sensitivity": turn.sensitivity,
            })
        if out:
            logger.info(
                "conclusion_distilled facts=%d ledger_key=%s conclusion_len=%d",
                len(out), turn.ledger_key, len(conclusion),
            )
        return out

    def _is_novel(
        self,
        candidate_emb: list[float],
        existing_embeddings: list[list[float]],
        entities: list[str] | None = None,
        existing_entities: list[list[str]] | None = None,
        content: str = "",
        existing_texts: list[str] | None = None,
    ) -> bool:
        """candidate 与已有 facts 最高 cosine 相似度 < 阈值 -> 新颖。

        冷启动（existing 为空）-> True。

        方案 B 实体感知：cosine >= 阈值时，遍历高相似已有 fact，若任一实体集合
        Jaccard >= entity_overlap_threshold -> 判重；实体有差异 -> 仍新颖。
        entities / existing_entities 缺失（任一为空）-> 退化为纯 cosine（原行为）。
        existing_entities 必须与 existing_embeddings 同序同长。
        """
        if not existing_embeddings:
            return True

        # ADR-0028 E6.5：标识符前置放行（within-batch 同款口径，见 checker 侧注释）
        if content and existing_texts:
            from bladex_core.distill_fidelity import identifier_novelty_override
            if identifier_novelty_override(content, existing_texts):
                return True

        try:
            import numpy as np

            q = np.asarray(candidate_emb, dtype=np.float32)
            q_norm = q / (float(np.linalg.norm(q)) or 1.0)

            mat = np.stack([np.asarray(e, dtype=np.float32) for e in existing_embeddings])
            # 归一化已有向量
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            mat_norm = mat / norms

            sims = mat_norm @ q_norm
            max_sim = float(sims.max())
            if max_sim < self._novelty_threshold:
                return True  # 相似度不够，新颖

            # cosine 高 -> 实体感知判定
            if not (self._entity_aware and entities and existing_entities):
                return False  # 无实体感知条件 -> 纯 cosine 判重
            cand_ents = {x.strip().lower() for x in entities if x and x.strip()}
            if not cand_ents:
                return False  # 候选无实体，退化纯 cosine
            # 遍历所有高相似已有 fact，任一实体高度重叠 -> 判重
            for idx, sim in enumerate(sims):
                if float(sim) < self._novelty_threshold:
                    continue
                if idx >= len(existing_entities):
                    break
                ex_ents = {x.strip().lower() for x in (existing_entities[idx] or []) if x and x.strip()}
                if not ex_ents:
                    continue  # 已有 fact 无实体，无法判定，看下一个
                jac = len(cand_ents & ex_ents) / len(cand_ents | ex_ents)
                if jac >= self._entity_overlap_threshold:
                    return False  # 实体高度重叠 + cosine 高 -> 判重
            # 所有高相似候选实体都有差异 -> 新颖
            return True
        except Exception as e:
            logger.warning("consolidate_novelty_error err=%s", e)
            return True  # 出错时保守写入


def _merge_tags(prov: str, extra: str) -> str:
    """拼 tags：`origin:<通道>` **恒在首位**，其余标记追加在后。

    首位约定是硬要求 —— `_channel_of` 之外仍有按位置读 tags 的历史代码，
    而 2026-08-08 之前 tags 里只可能有一个 token，谁都没想过会有第二个。
    """
    head = f"origin:{prov}" if prov else "origin:consolidation"
    rest = [t.strip() for t in (extra or "").split(",")
            if t.strip() and not t.strip().startswith("origin:")]
    return ",".join([head, *rest])


def _channel_of(tags: str) -> str:
    """从 tags 里取通道名。

    旧写法是 `tags.split(":")[-1]` —— 单 token 时侥幸正确，
    一旦 tags 变成 `origin:conclusion,lang:drift` 就会返回 `drift`，
    把漏斗标签悄悄写歪（不报错、不崩，只是数字从此没意义）。
    """
    for tok in (tags or "").split(","):
        tok = tok.strip()
        if tok.startswith("origin:"):
            return tok.split(":", 1)[1] or "consolidation"
    return "consolidation"
