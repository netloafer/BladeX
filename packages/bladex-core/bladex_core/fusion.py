"""检索融合与重排（ADR-0028 E6.3）—— 纯函数，agent 中立，零 LLM。

## 为什么换成 RRF

现行软打分是**分值加权**：`0.50·dense + 0.20·kw + 0.15·ent + 0.10·imp + 0.05·rec`。
真实库上它不工作，原因是分值不可比：

    库内 cosine 噪声底就是 **0.88–0.93**，top-15 内极差只有 0.024–0.060；
    而 importance 的差分（file_ref 0.6 vs assertion 0.7，×0.10 权重 ÷1.5 归一）
    ≈ **0.007** —— 三路信号没有一路能对抗 ±0.05 的 cosine 噪声。
    `min_relevance=0.3` 这个绝对阈值同样形同虚设（噪声底比它高得多）。

**RRF 用秩不用分值**（Cormack et al., SIGIR 2009）：

    rrf(f) = Σ_channel 1/(60 + rank_channel(f))

"某条在 dense 里排第 3、在词法里排第 1"是可比的；"cosine 0.912 vs 0.907"不是。
分值不可比的问题就此正面解掉，`min_relevance` 绝对阈值随之退役。

## 五步（顺序即语义）

  ① RRF 融合（dense / lexical / session / entity 各一路）
  ② 修正项：+ w_imp·importance + w_rec·recency（Generative Agents 三信号形制，
     Park et al. UIST 2023）—— **修正**不是主排序，权重初值 0.15/0.10，
     `[标定]` 由 E5 网格搜索给出，人不手调。
  ③ 同义折叠：同取代键 / 高 cosine+实体重叠 → 只留高分（跨语言双份在此折叠）
  ④ 类型配额：file_ref ≤1、profile_obs=0、lesson/procedure 保底各 1
  ⑤ MMR 多样性（Carbonell & Goldstein, SIGIR 1998），λ=0.7

**明确不做（v1）**：cross-encoder 重排（bge-reranker）不进热路径。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .flags import flag_enabled
from .topic_keys import canonical_entity_key as _canon_raw


def _canon(s: str) -> str:
    """V-I2 实体规范化的回滚通道：`BLADEX_ENTITY_CANON=0` 退回旧口径
    （strip+lower），供单变量 A/B 与出问题时止血。默认开（flags.py 单一真相源）。"""
    if flag_enabled("BLADEX_ENTITY_CANON"):
        return _canon_raw(s)
    return (s or "").strip().lower()

# RRF 常数（Cormack et al. 2009 的推荐值，勿"顺手调小试试"——它就是个平滑项）
RRF_K = 60

# ② 修正项权重 [标定：E5 网格搜索脚本给出，人不手调]
W_IMPORTANCE = 0.15
W_RECENCY = 0.10
RECENCY_HALFLIFE_DAYS = 30.0

# ③ 同义折叠阈值
FOLD_COSINE = 0.92
FOLD_ENTITY_JACCARD = 0.6

# ④ 类型配额（top-k 内）
QUOTA_MAX_FILE_REF = 1
QUOTA_MAX_PROFILE_OBS = 0
QUOTA_MIN_LESSON = 1
QUOTA_MIN_PROCEDURE = 1

# ⑤ MMR
MMR_LAMBDA = 0.7

# entity 通道（T3b，三段式卡 2026-08-10）：热门实体衰减系数。
# 取 mem0 的 memory_count_weight = 1/(1+coef·(N-1)²) 起步（scoring.py），
# N = 该实体关联的 fact 数——'bladex' 这类全项目高频实体关联数百条 fact，
# 对单条的加权趋零，防热门实体霸榜。可由 BLADEX_ENTITY_DECAY_COEF 覆盖。
ENTITY_DECAY_COEF = 0.001

# 入通道门槛：排序键（Σ decay/|ents|）低于此值的 fact **不进通道**。
# 为什么衰减必须同时管资格而不只管通道内排序（首跑评估集实测教训）：
# '泰山啤酒' 全库关联 119 条，55 条候选全部入通道——衰减把它们排得再靠后，
# RRF 仍给每条 1/(60+r) 的融合票，等于用一坨低精度候选重洗 dense 的
# 精细排序（taishan-000 mrr 1.0→0.5；与 bigram 被评估集否掉同一形态，
# 见 flags.BLADEX_FTS_BIGRAM_LIMIT 注释）。mem0 语义里热门实体 boost 趋零
# ≈ 没有 boost；RRF 的等价物 = 不入通道。0.1 ⇔ 单实体 fact 在 N≈96 时出局
# （decay(96)≈0.104），[标定] 起步值，随 E5 网格搜索重标。
ENTITY_MIN_CHANNEL_KEY = 0.1


@dataclass
class Channel:
    """一路召回：有序的 fact id 列表（越靠前越相关）。"""

    name: str
    ranked_ids: list[str] = field(default_factory=list)
    weight: float = 1.0


def entity_decay(n_linked: int, coef: float = ENTITY_DECAY_COEF) -> float:
    """热门实体衰减：1/(1+coef·(N-1)²)。N<=1 不衰减。"""
    if n_linked <= 1:
        return 1.0
    return 1.0 / (1.0 + coef * (n_linked - 1) ** 2)


def build_entity_channel(
    query: str,
    facts_by_id: dict[str, Any],
    *,
    entity_freq: dict[str, int] | None = None,
    coef: float = ENTITY_DECAY_COEF,
    name: str = "entity",
) -> Channel | None:
    """④ entity 第四通道（T3b）：fact.entities 与 query 的子串命中。

    匹配逻辑**迁移复用** T8a `_rerank_by_entities` 的口径：fact 侧 entities
    是蒸馏时 LLM 抽出的判别性命名实体，query 侧不抽实体不花 LLM，直接子串
    包含（中英文通吃）。自 08-04 `BLADEX_SOFT_SCORING` 默认开后 T8a 被显式
    跳过而 RRF 无 entity 通道——实体信号实际退出了检索排序，本函数把它
    以「第四路秩」的形态接回融合。

    通道内排序（RRF 只吃秩）：命中强度降序，热门实体衰减进排序键——
        key(f) = Σ_{e∈命中实体} entity_decay(N_e) / |f.entities|
    N_e = 实体 e 关联的 fact 数（entity_freq，调用方从库聚合；缺省 = 不衰减）。

    🔴 明确**不抄** mem0 的加法分值融合（(semantic+bm25+boost)/max_possible）：
    弃分值用秩的理由写死在 memory_index.py `_fuse_rerank` 一节——库内 cosine
    噪声底 0.88–0.93，分值信号扛不住噪声，RRF 骨架不动。

    副作用：给命中的 fact 设 `_entity_hit`（命中率 [0,1]），供
    `index_search_done` 的 entity_hits 观测字段使用（与 T8a 同名同义）。
    返回 None = query 无任何实体命中 → 不加通道，三通道行为与改造前逐字一致。
    """
    if not query or not facts_by_id:
        return None
    # V-I2：两侧同尺——实体与 query 都过 canonical_entity_key（分隔符变体折叠），
    # 修 `qwen3.8:27b` vs `qwen3.8-27B` 全灭病例（08-20 注入归因）。
    # 已规范的字符串折叠是恒等式 ⇒ 原本命中的照旧命中（只增不减）。
    q_canon = _canon(query)
    freq = entity_freq or {}
    scored: list[tuple[float, str]] = []
    for fid, f in facts_by_id.items():
        ents = [e for e in (getattr(f, "entities", None) or []) if e]
        if not ents:
            continue
        hits = [e for e in ents if _canon(e) in q_canon]
        if not hits:
            continue
        # freq 的键可能是调用方按旧口径（strip().lower()）聚合的——双键查找兜底，
        # 聚合侧迁移到 canonical 键后旧键自然退役。
        key = sum(entity_decay(
                      freq.get(_canon(e),
                               freq.get(e.strip().lower(), 1)), coef)
                  for e in hits) / len(ents)
        if key < ENTITY_MIN_CHANNEL_KEY:
            continue          # 热门实体的弱证据不进通道（见常量注释）
        f._entity_hit = len(hits) / len(ents)
        # S4（G9.3）：把**衰减后的排序键**留在对象上，供 explain 分解读取。
        # 不留这一项，"这条为什么排在这"就只能靠仪器在外面把 entity_decay 重算一遍
        # ——外挂重算 = 又一把会和生产脱钩的尺子（本仓已为此付过多次）。
        f._entity_key = key
        scored.append((key, fid))
    if not scored:
        return None
    scored.sort(key=lambda t: (-t[0], t[1]))     # 强度降序，同分按 id 确定性
    # 权重与其余通道一致（1.0）。评估集上试过 0.5：recall@10 +0.016 但丢掉
    # recall@5(−0.016)/matter-hit(−0.063) 增益、mrr 不回收——不占优，
    # 不引入第二个手调旋钮；通道权重标定统一归 E5 网格搜索（MS-9）。
    return Channel(name, [fid for _, fid in scored])


def rrf_fuse(channels: list[Channel]) -> dict[str, float]:
    """RRF：rrf(f) = Σ_channel weight / (RRF_K + rank)。rank 从 1 起。"""
    scores: dict[str, float] = {}
    for ch in channels:
        for i, fid in enumerate(ch.ranked_ids):
            scores[fid] = scores.get(fid, 0.0) + ch.weight / (RRF_K + i + 1)
    return scores


def _normalize(scores: dict[str, float]) -> dict[str, float]:
    """归一到 [0,1]，供修正项按同一标度相加。"""
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi <= lo:
        return dict.fromkeys(scores, 1.0)
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def _recency(age_days: float, half_life: float = RECENCY_HALFLIFE_DAYS) -> float:
    return 0.5 ** (max(0.0, age_days) / half_life) if half_life > 0 else 1.0


def _kind(fact: Any) -> str:  # noqa: ANN401
    ik = getattr(fact, "item_kind", "")
    return getattr(ik, "value", str(ik))


def _entity_jaccard(a: list[str], b: list[str]) -> float:
    # V-I2：Jaccard 也按规范键比——`qwen3.8:27b` 与 `qwen3.8-27B` 是同一实体，
    # 折叠判定（③）不该把它们当两个实体而漏折同义双份。
    sa = {_canon(str(x)) for x in (a or []) if x}
    sb = {_canon(str(x)) for x in (b or []) if x}
    sa.discard("")
    sb.discard("")
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _cosine(a: list[float] | None, b: list[float] | None) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _supersede_key(fact: Any) -> tuple[str, str, str] | None:  # noqa: ANN401
    from bladex_core.supersede import supersede_key
    try:
        return supersede_key(fact)
    except Exception:  # noqa: BLE001
        return None


#: `explain` 里 `dropped_at` 的**闭集**（`None` = 活到最后）。顺序即流水线顺序。
#: 🔴 仪器一律从这里读，不在自己文件里另抄一份字面量——本仓栽过两次
#: 「闭集按记忆默写」（`lane` 三值漏一个 ⇒ 447 条边静默消失）。
#: `quota_type`（类型上限挡下）/ `quota_floor`（被 lesson·procedure 保底挤掉）/
#: `rank_cut`（只是名次落在 2×top_k 截断线之外）三者**同出一段却是三种机制**，
#: 混成一个 `quota` 会把「名次不够」读成「配额杀」——那正是仪器此前在外面
#: 自己拆的那一刀，现在由生产给出。
DROPPED_AT_VALUES: tuple[str, ...] = ("fold", "quota_type", "quota_floor", "rank_cut", "mmr")


def fold_synonyms(
    ranked: list[Any],
    *,
    vectors: dict[str, list[float]] | None = None,
    cos_threshold: float = FOLD_COSINE,
    pairs: dict[str, tuple[str, float, float, str]] | None = None,
) -> tuple[list[Any], int]:
    """③ 同义折叠：同取代键 或 (cosine≥阈值 ∧ 实体 Jaccard≥0.6) → 只留高分那条。

    **展示层去重，不动库**——被折叠的那条仍在 Memory Index 里，as-of / 审计照常可见。
    跨语言双份（ZH/EN 同一事实各蒸一遍，实测 cosine 0.8639 < novelty 阈值 0.95
    所以两条都进了库）正是在这里被收拢。

    ── `cos_threshold` 为什么要能从外面传（2026-08-20，读数卡
    `docs/benchmarks/channel-votes-20260820.md`）────────────────────────────

    仪器把「e5 排进前 5 却没进注入」的候选按出局阶段拆开，**80.7% 死在本函数**
    （`rank` 那一档 = 0，通道投票根本不决定去留）。根因是本模块自相矛盾：

        模块开头写着「库内 cosine 噪声底就是 0.88–0.93 …… 绝对阈值形同虚设」，
        RRF 整套就是为**弃用绝对 cosine 阈值**（用秩不用分）而设计的；
        而 `FOLD_COSINE = 0.92` **正埋在它自己声明的那个噪声底里**。

    人工签收 10 组折叠：真重复 cos **0.972 / 0.979**，误折 cos **0.926–0.964**
    （误折形态 = 把「资源状态 vs 方法教训」「详案 vs 概括」「原因 vs 结论」判成同一条）。
    阈值敏感度扫描（150 条真实 query）：

        0.92 流失 29.1% ／ 0.95 → 24.0% ／ **0.97 → 18.1%** ／ 0.98 → 15.1%

    0.97 是曲线拐点与签收分界的交点，故 flag 默认取 0.97。

    🔴 **本函数的缺省仍是 `FOLD_COSINE`（0.92）= 改造前行为逐字不变**——
    生产值由调用方从 `BLADEX_FOLD_COSINE` 接线（与 T3-A `max_profile_obs` 同款：
    fusion 保持纯函数、不 import flags）。回滚 = 设该 env 回 0.92。

    另注：`FOLD_ENTITY_JACCARD = 0.6` 这道「第二条件」实测**几乎无判别力**——
    本库以少数实体为中心（虎彩集团/泰山啤酒/企信宝/bladex/qwen3.8:27b），同主题
    fact 的实体集合天然重合（10 组签收的 jaccard：1.00/0.60/0.60/0.75/1.00/1.00/
    0.50/1.00/0.60/0.67）。它留着是零成本，但**不要再拿它当「第二道保险」来论证**。
    """
    vectors = vectors or {}
    kept: list[Any] = []
    folded = 0
    for f in ranked:                       # ranked 已按分降序 → 先到者即高分者
        dup = False
        fk = _supersede_key(f)
        for k in kept:
            cos = jac = 0.0
            why = ""
            if fk is not None and fk == _supersede_key(k):
                dup, why = True, "supersede_key"
            else:
                # 🔴 短路顺序与改造前**逐字一致**（先 cosine，够了才算 jaccard）：
                # 热路径上每条候选要与全部 kept 比一遍，把 jaccard 提前算
                # 就是给注入预算加一笔没人要求的开销。
                cos = _cosine(vectors.get(f.id), vectors.get(k.id))
                if cos >= cos_threshold:
                    jac = _entity_jaccard(getattr(f, "entities", []),
                                          getattr(k, "entities", []))
                    if jac >= FOLD_ENTITY_JACCARD:
                        dup, why = True, "cosine"
            if dup:
                if pairs is not None:
                    # 折叠**配对**只有这里知道（返回值只有计数）。不落下来，
                    # 仪器就只能在外面按同样的谓词回找一遍——而那份外挂
                    # 一直在用模块常量 0.92，生产早已是 flag 的 0.97。
                    pairs[f.id] = (k.id, round(cos, 6), round(jac, 4), why)
                break
        if dup:
            folded += 1
            continue
        kept.append(f)
    return kept, folded


def apply_type_quota(
    ranked: list[Any], top_k: int, *,
    max_profile_obs: int = QUOTA_MAX_PROFILE_OBS,
    reasons: dict[str, str] | None = None,
) -> list[Any]:
    """④ 类型配额：file_ref ≤1、profile_obs ≤max_profile_obs、lesson/procedure 保底各 1。

    保底那两条是在修"记住了发生什么、没记住为什么错"的**展示面**：
    lesson（根因/纠正）与 procedure（怎么做有效）在纯相关性排序里
    经常被一堆 assertion 挤掉，而它们恰恰是同型错误第三次躲过的那部分记忆。

    T3-A（2026-08-15）：profile_obs 从整类禁（=0）改为**限额参与**。T1 归因
    11/18 错题的时间锚（"加入/开始/购买 于 DATE"）蒸成 profile_obs 后在这里
    被整类丢弃 → temporal 问题结构性不可达；而①平面画像卡实测不聚合其内容，
    两头无消费者=死数据。调用方从 flag `BLADEX_QUOTA_PROFILE_OBS` 取值传入
    （与 entity coef 同款接线，fusion 保持纯函数）；0 = 旧行为逐字不变。
    """
    out: list[Any] = []
    counts: dict[str, int] = {}
    deferred: list[Any] = []
    capped: set[str] = set()          # 被类型上限挡下的（只在 reasons 开时记）
    evicted: set[str] = set()         # 被保底挤掉的

    for f in ranked:
        k = _kind(f)
        if k == "profile_obs" and counts.get("profile_obs", 0) >= max_profile_obs:
            capped.add(f.id)
            continue                                   # 超限的画像原料不注入
        if k == "file_ref" and counts.get("file_ref", 0) >= QUOTA_MAX_FILE_REF:
            capped.add(f.id)
            continue
        if len(out) >= top_k:
            deferred.append(f)
            continue
        out.append(f)
        counts[k] = counts.get(k, 0) + 1

    # 保底：top_k 内没有 lesson/procedure 但召回集里有 → 挤掉末位的 assertion
    for kind, floor in (("lesson", QUOTA_MIN_LESSON), ("procedure", QUOTA_MIN_PROCEDURE)):
        if counts.get(kind, 0) >= floor:
            continue
        cand = next((f for f in deferred if _kind(f) == kind), None)
        if cand is None:
            continue
        victim = next((f for f in reversed(out) if _kind(f) == "assertion"), None)
        if victim is None:
            continue
        out[out.index(victim)] = cand
        evicted.add(victim.id)
        deferred.remove(cand)
        counts[kind] = counts.get(kind, 0) + 1

    if reasons is not None:
        # 🔴 本函数一段里干**三件事**：类型上限、保底顶替、以及 `len(out) >= top_k`
        # 那条截断线。三者混成一个 "quota" 会把「名次不够」记成「配额杀」，
        # 而两者的修法完全相反（前者调融合/召回宽度，后者调配额）。
        kept_ids = {f.id for f in out}
        for f in ranked:
            if f.id in kept_ids:
                continue
            reasons[f.id] = ("quota_type" if f.id in capped
                             else "quota_floor" if f.id in evicted
                             else "rank_cut")
    return out


def mmr_select(
    ranked: list[Any],
    scores: dict[str, float],
    top_k: int,
    *,
    vectors: dict[str, list[float]] | None = None,
    lam: float = MMR_LAMBDA,
) -> list[Any]:
    """⑤ MMR 多样性（Carbonell & Goldstein 1998）：

        迭代选 argmax  λ·score(f) − (1−λ)·max_{s∈已选} cos(f, s)

    没有向量时退化为按分排序（不做假多样性）。
    """
    vectors = vectors or {}
    if not ranked or not vectors:
        return ranked[:top_k]
    pool = list(ranked)
    chosen: list[Any] = []
    while pool and len(chosen) < top_k:
        best, best_val = None, -1e9
        for f in pool:
            sim = max(
                (_cosine(vectors.get(f.id), vectors.get(c.id)) for c in chosen),
                default=0.0,
            )
            val = lam * scores.get(f.id, 0.0) - (1 - lam) * sim
            if val > best_val:
                best, best_val = f, val
        chosen.append(best)
        pool.remove(best)
    return chosen


def fuse_and_rerank(
    facts_by_id: dict[str, Any],
    channels: list[Channel],
    top_k: int,
    *,
    vectors: dict[str, list[float]] | None = None,
    ages_days: dict[str, float] | None = None,
    audience_bonus_ids: set[str] | None = None,
    w_importance: float = W_IMPORTANCE,
    w_recency: float = W_RECENCY,
    w_audience: float = 0.1,
    max_profile_obs: int = QUOTA_MAX_PROFILE_OBS,
    profile_obs_discount: float = 1.0,
    fold_cosine: float = FOLD_COSINE,
    explain: bool = False,
) -> tuple[list[Any], dict[str, Any]]:
    """E6.3 全流程：RRF → 修正项 → 同义折叠 → 类型配额 → MMR。

    T3-A：`max_profile_obs`/`profile_obs_discount` 由调用方从 flag 接线
    （BLADEX_QUOTA_PROFILE_OBS / BLADEX_PROFILE_OBS_DISCOUNT）。折价乘在
    融合总分上：画像原料参与召回但不挤掉同等相关的 assertion/lesson；
    默认参数（0 配额 / 1.0 折价）= 改造前行为逐字不变。

    返回 (facts, info)。info 供日志与测试观察：
        {rrf_top, folded, channels, scored}

    `explain=True`（S4，G9.3）额外产出 `info["explain"]`：每个进入打分的候选一份
        {rrf, base, importance_term, recency_term, audience_bonus,
         discount, final, entity_key, entity_hit, dropped_at}
    死在折叠的那些**另加四项**（只有 `fold_synonyms` 内部知道的配对信息）：
        {folded_by, fold_cos, fold_jaccard, fold_why}

    **为什么要有它**：`dropped_at` 记的是候选**死在哪一段**，取值是闭集
    `DROPPED_AT_VALUES`（`None` = 活到最后）。2026-08-20 那次"注入丢记忆"的归因
    （72.6% 死在同义折叠、rank 只占 0.8%）是靠仪器在外面把整条流水线重算一遍
    得出的——**外挂重算的尺子迟早与生产脱钩**（本仓已为此付过多次）。
    分解由生产自己产出，仪器只做呈现。

    `explain=False` 时**逐字不改变任何输出**（守卫见 `test_g93_explain.py`）。
    """
    ages_days = ages_days or {}
    audience_bonus_ids = audience_bonus_ids or set()

    rrf = rrf_fuse(channels)
    norm = _normalize(rrf)

    scores: dict[str, float] = {}
    breakdown: dict[str, dict[str, Any]] = {}
    for fid, base in norm.items():
        f = facts_by_id.get(fid)
        if f is None:
            continue
        imp = min(max(float(getattr(f, "importance", 0.0) or 0.0), 0.0) / 1.5, 1.0)
        rec = _recency(ages_days.get(fid, 0.0))
        aud = 1.0 if fid in audience_bonus_ids else 0.0
        s = base + w_importance * imp + w_recency * rec + w_audience * aud
        discount = 1.0
        if profile_obs_discount != 1.0 and _kind(f) == "profile_obs":
            discount = profile_obs_discount
            s *= discount
        scores[fid] = s
        if explain:
            # 🔴 键名带 `_term` / `_bonus` 后缀**不是排版洁癖**：H1 对账 gate 把
            # **任意 dict 字面量的字符串键**当作同名 schema 字段的赋值点
            # （`check_producer_consumer._UseVisitor.visit_Dict`，为的是抓
            # `{"open_issues": …}` 那类构造）。裸用 `"audience"` 会让
            # `Fact.audience`（真实零生产者、已在基线里）**凭一个无关的 dict 键
            # 复活**，gate 当场红（2026-08-20 实测）。
            # 后缀同时更准确：这里存的是**加权后的贡献值**，不是字段原值。
            breakdown[fid] = {
                "rrf": round(rrf.get(fid, 0.0), 6),
                "base": round(base, 6),
                "importance_term": round(w_importance * imp, 6),
                "recency_term": round(w_recency * rec, 6),
                "audience_bonus": round(w_audience * aud, 6),
                "discount": discount,
                "final": round(s, 6),
                # entity 通道的**衰减后**排序键与命中率（build_entity_channel 落的）
                "entity_key": round(float(getattr(f, "_entity_key", 0.0) or 0.0), 6),
                "entity_hit": round(float(getattr(f, "_entity_hit", 0.0) or 0.0), 6),
                "dropped_at": None,      # 下面按各段幸存集合回填
            }

    ordered = sorted(
        (facts_by_id[i] for i in scores),
        key=lambda f: (-scores[f.id], f.id),      # 同分按 id：确定性排序
    )
    for f in ordered:
        f._score = scores[f.id]
        f._rrf = rrf.get(f.id, 0.0)

    fold_pairs: dict[str, tuple[str, float, float, str]] | None = {} if explain else None
    quota_reasons: dict[str, str] | None = {} if explain else None
    folded_list, folded = fold_synonyms(ordered, vectors=vectors,
                                        cos_threshold=fold_cosine,
                                        pairs=fold_pairs)
    quota_list = apply_type_quota(folded_list, max(top_k * 2, top_k),
                                  max_profile_obs=max_profile_obs,
                                  reasons=quota_reasons)
    final = mmr_select(quota_list, scores, top_k, vectors=vectors)

    if explain:
        # 出局阶段 = 逐段幸存集合的差集。顺序即流水线顺序，**最后一段先判**，
        # 免得一条既没进 quota 也没进 mmr 的候选被记成"死在 mmr"。
        after_fold = {f.id for f in folded_list}
        after_quota = {f.id for f in quota_list}
        after_mmr = {f.id for f in final}
        for fid, row in breakdown.items():
            if fid not in after_fold:
                row["dropped_at"] = "fold"
                # 折叠掉它的是谁、cos/jaccard 各多少、按哪条谓词——这三样
                # 只有 `fold_synonyms` 内部知道，落在这里，仪器就不必回找。
                _p = (fold_pairs or {}).get(fid)
                if _p:
                    row["folded_by"], row["fold_cos"], row["fold_jaccard"], row["fold_why"] = _p
            elif fid not in after_quota:
                row["dropped_at"] = (quota_reasons or {}).get(fid, "rank_cut")
            elif fid not in after_mmr:
                row["dropped_at"] = "mmr"

    info = {
        "channels": {ch.name: len(ch.ranked_ids) for ch in channels},
        "scored": len(scores),
        "folded": folded,
        # 生效阈值进 info：`index_search_fused` 打出来后，日志里能直接看出这轮
        # 用的是 0.92 还是 0.97 —— 否则"改了没生效"只能靠翻配置猜（已踩过多次）。
        "fold_cosine": fold_cosine,
        "rrf_top": round(max(rrf.values()), 5) if rrf else 0.0,
    }
    if explain:
        info["explain"] = breakdown
    return final, info
