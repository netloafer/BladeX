"""归属判定管线 - 五层链接（DPL：Distill-Propose-Link，ADR-0018 §3.2，护城河）。

归属从"向量空间找邻居"（cosine 阈值）重构为"提案与登记簿的链接匹配"：

  L1 manual   手动归属（admin event）              conf 1.0
  L2 explicit 对话点名/续接语（ExplicitSignalDetector）conf 0.95
  L3 alias-link proposal 标题/entities ↔ Matter 原生键(title+aliases) 规范化匹配  conf 0.9
                （MS-16：累积 Matter.entities 退出匹配面——并集是雪球源）
  L4 LLM-link e5 召回 top-k(≤5) 候选 -> 便宜档 LLM 裁决（批量+台账缓存）  conf 0.8
  L5 兜底     none->从 proposal 新开 provisional；uncertain/无 judge->留池

原则不变含义升级（ADR-0018 §3.2）：
  - 宁分勿合：模糊由 LLM 显式 uncertain -> 留池，绝不"分数够就合"。
  - e5 无红线：召回 top-k 只排序不设阈值；cosine 绝对值不出现在任何红线判定。

降级（ADR-0018 §3.7）：LLM 不可用 -> L1-L3 照常（纯规则），L4/新开暂停，
fact 全部留池 + pending_judgment，恢复后重判补收。绝不回退 cosine 阈值合并。

agent 中立：只定义协议 + 纯逻辑，LLM 实现在 proxy 侧（LLMLinkJudge）。
只跑异步管线（consolidation 阶段），热路径只读不判。
"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from datetime import UTC, datetime
from enum import Enum
from collections import Counter
from collections.abc import Callable
from typing import Protocol

from pydantic import BaseModel, Field

from bladex_core.fact import Fact
from bladex_core.matter import (
    EdgeProvenance,
    EdgeTargetType,
    Matter,
    MatterEdge,
    MatterOrigin,
    MatterStatus,
)
from bladex_core.topic_keys import (
    derive_topic_keys,
    is_strong_topic_match,
    is_valid_topic_key,
    normalize_key,
    topic_key_overlap,
)

logger = logging.getLogger(__name__)

# 未归属池的保留 matter_id
UNASSIGNED_MATTER_ID = "__unassigned__"

# 默认参数（均留实测，不拍脑袋；ADR-0018 §8 留实测参数表）
_DEFAULT_DORMANT_TIMEOUT_S = 7 * 24 * 3600  # 7 天无活动 -> dormant
_DEFAULT_TOP_CANDIDATES = 5          # L4 e5 召回 top-k（起点 5，ADR-0018 §8）
_DEFAULT_L3_ENTITY_INTERSECTION = 2  # L3 实体交集门槛（起点 2，ADR-0018 §8）
_EXPLICIT_CONFIDENCE = 0.95          # L2 显式信号置信度
_ALIAS_CONFIDENCE = 0.9              # L3 alias-link 置信度
_LLM_LINK_CONFIDENCE = 0.8           # L4 LLM-link 基础置信度（×judge 自评）
_MIN_PROVISIONAL_TITLE_CHARS = 3     # proposal 标题过短 -> 不新开 provisional

# 保留旧常量名（cosine 红线已退役 ADR-0018 §2，但 config 不变式/M-proxy-5 脚本仍引用）。
# 不再用于任何归属判定，仅维持向后兼容。
_DEFAULT_SEMANTIC_THRESHOLD = 0.85

# MS-16（T4a-1）：L2 门控弱命中（子串相含/实体命中）需要的 fact 侧独立信号数。
# 单个泛词（'bladex' 这类全项目高频 token）子串命中不足以放行——
# 08-10 审计：L2=466 里 27 substr + 16 exact_entity 走的就是单泛词路径。
_GATE_MIN_WEAK_HITS = 2


class AttributionSource(str, Enum):
    """归属决策的信号来源（ADR-0018 §3.2 五层 + 账本锚 + 延续）。

    （L0 `SAME_UNIT` 同单元同属已于 2026-09-03 S4 连函数删除：默认关、从未翻默认、
    零 live 读数，且 MQ-V6 实测碎片化瓶颈与任务单元无关；素材在 git 历史。）
    """

    # V-L6（ADR-0032 §4.5）：账本锚——session 绑定的账本锚定 Matter，归属主路径。
    # 生产者 = memory_index 账本锚层（BLADEX_LEDGER_ANCHOR）；边 decision.layer="ANCHOR"。
    LEDGER_ANCHOR = "ledger_anchor"
    # G12.2 D1（ADR-0031 §3）：延续判定直挂（R1/R2/R4 零 LLM；决策表 v2）。
    # 生产者 = memory_index 归属循环的 D1 直挂路径；边 decision.layer = "CONT"。
    CONTINUATION = "continuation"
    MANUAL = "manual"          # L1 手动指定
    EXPLICIT = "explicit"      # L2 对话点名/续接语
    ALIAS = "alias"            # L3 proposal↔aliases 规范化匹配
    LLM_LINK = "llm_link"      # L4 LLM 裁决链接
    PROVISIONAL = "provisional"  # L5 从 proposal 新开 provisional Matter
    UNASSIGNED = "unassigned"  # L5 挂未归属池


class AttributionDecision(BaseModel):
    """归属判定结果（ADR-0018 §3.2/§3.5）。"""

    fact_id: str
    matter_id: str           # 目标 Matter ID（UNASSIGNED_MATTER_ID 表示未归属）
    confidence: float        # 置信度 [0, 1]
    source: AttributionSource
    is_new_matter: bool = False  # 是否需要新建（provisional）Matter
    new_matter_title: str = ""    # 新建 Matter 标题（proposal 标题）
    new_matter_aliases: list[str] = Field(default_factory=list)  # 新建 Matter 初始 aliases
    # 拍板 A/B（2026-08-14）：新建 Matter 的主题键种子（创始 fact 派生，计数=1）。
    # 有种子，同批/次轮的兄弟 proposal 才能被 L3 主题键子层拦住不再开第二个。
    new_matter_topic_keys: list[str] = Field(default_factory=list)
    pending_judgment: bool = False  # 留池标记（uncertain / judge 不可用），T9 重判用
    summary_rewrite: str = ""  # L4 顺带重写的摘要（caller 应用到 Matter，ADR-0018 §3.4）
    signal: str = ""  # 可解释性：命中信号来源
    # ADR-0018 §3.5: 决策记录 {layer, candidates, verdict, judge_model, ts}
    decision: dict = Field(default_factory=dict)

    model_config = {"extra": "allow"}


# ── ADR-0018 §3.2 L4: 链接裁决协议（agent 中立，proxy 侧 LLMLinkJudge 实现）──


class LinkVerdict(str, Enum):
    """L4 裁决三值（ADR-0018 §3.2）。"""

    LINK = "link"            # link:<matter_id> -> 链到某候选
    NONE = "none"            # 不属于任何候选 -> 新开 provisional
    UNCERTAIN = "uncertain"  # 不确定 -> 留池


class LinkJudgeCandidate(BaseModel):
    """给 LLM 的候选 Matter 登记卡（ADR-0018 §3.2）。"""

    matter_id: str
    title: str = ""
    aliases: list[str] = Field(default_factory=list)
    summary: str = ""
    recent_members: list[str] = Field(default_factory=list)  # 最近 3 条成员 fact（v1 留空，summary 兜底）

    model_config = {"extra": "allow"}


class LinkJudgeItem(BaseModel):
    """一条待裁决的 fact + 其候选卡（批量裁决的单位）。

    topic（T4b，2026-08-10）：该 fact 的轮级主题——**只作证据输入**进裁决
    prompt 的 fact 描述（一行），不改判定门槛/verdict 语义/宁分勿合。
    """

    fact_id: str
    content: str
    entities: list[str] = Field(default_factory=list)
    proposal_titles: list[str] = Field(default_factory=list)
    topic: str = ""
    candidates: list[LinkJudgeCandidate] = Field(default_factory=list)

    model_config = {"extra": "allow"}


class LinkJudgeResult(BaseModel):
    """一条 L4 裁决结果。"""

    fact_id: str
    verdict: LinkVerdict = LinkVerdict.UNCERTAIN
    matter_id: str = ""        # verdict=LINK 时填（候选 matter_id）
    summary_rewrite: str = ""  # 顺带重写的摘要（ADR-0018 §3.4）
    reason: str = ""

    model_config = {"extra": "allow"}


class LinkJudge(Protocol):
    """L4 链接裁决接口（agent 中立，ADR-0018 §3.2/§3.6）。

    proxy 侧提供 Router 实现（LLMLinkJudge），复用蒸馏同一模型（少一个行为面）。
    同步调用（consolidator 是后台批处理进程，非热路径，与 LLMDistiller 同步一致）。
    批量：一次调用裁决多条 fact（ADR-0018 §3.2 "可批量"）。
    """

    @property
    def model_name(self) -> str:
        """裁决模型名（记入 decision.judge_model + judgment 台账）。"""
        ...

    def judge_links(self, items: list[LinkJudgeItem]) -> list[LinkJudgeResult]:
        """批量链接裁决。LLM 不可用/超时/解析失败 -> 全部 uncertain（降级留池，不抛）。"""
        ...


class JudgmentJournalProtocol(Protocol):
    """裁决台账接口（ADR-0018 §4.1，可 mock）。

    L4 裁决前查台账命中复用（零 LLM 成本），miss 才调 LLM、结果写台账。
    实现在 proxy 侧（HubJudgmentJournal 包装 MemoryHub），consolidator 注入。
    """

    def get_latest_judgment(self, fact_id: str) -> LinkJudgeResult | None:
        """命中返回最新裁决（fact_id 下最新非墓碑 seq），miss/被墓碑返回 None。"""
        ...

    def put_judgment(
        self, fact_id: str, candidates: list[dict], result: LinkJudgeResult,
    ) -> None:
        """追加一条裁决台账（judgment/{fact_id}/{seq}）。写失败由实现降级 log，不抛。"""
        ...


# ── L2: 显式信号检测（对话点名 / 续接语，保留现有实现）──


class ExplicitSignalMatch(BaseModel):
    """显式信号匹配结果（可解释）。"""

    matter_id: str
    matched_text: str       # 命中的文本片段
    signal_type: str        # "title_mention"（点名）| "continuation"（续接语+关键词）
    pattern: str = ""       # 命中的续接语模式（signal_type="continuation" 时）
    title: str = ""         # 命中的 Matter 标题

    model_config = {"extra": "allow"}


# 默认续接语模式（中英文覆盖，留实测）
_DEFAULT_CONTINUATION_PATTERNS: list[str] = [
    # 中文续接/指代
    r"继续",
    r"接着",
    r"回到",
    r"还是那个",
    r"之前那个",
    r"上次那个",
    r"昨天那个",
    r"接着昨天",
    r"继续上次",
    # English continuation / anaphora
    r"continue\b",
    r"still working on",
    r"still on",
    r"regarding\b",
    r"back to\b",
    r"picking up",
    r"follow.?up",
    r"as we discussed",
    r"that .+ (?:issue|task|thing|matter)",
    r"the .+ thing",
]

# 停用词（关键词提取时过滤，避免 "the"/"的" 等噪声词触发命中）
_STOP_WORDS: set[str] = {
    # English
    "the", "a", "an", "is", "are", "was", "were", "on", "in", "at",
    "to", "for", "of", "with", "and", "or", "but", "not", "this",
    "thing", "issue", "task", "matter", "it", "we", "you",
    # ADR-0018 真机回溯：'vs' 当关键词致阿根廷 Matter 吞跨主题 fact
    "vs",
    # Chinese
    "的", "了", "在", "是", "和", "与", "给", "把", "被", "让",
    "那个", "这个", "上次", "之前", "昨天", "今天",
}


class ExplicitSignalDetector:
    """显式信号检测器（L2，ADR-0012 §3.3 最高优先信号，ADR-0018 保留）。

    在 L3 alias-link 之前判定，信号分两层：
      1. 点名（title_mention）：user 消息直接包含 Matter 标题 -> 最强显式信号。
      2. 续接语 + 标题关键词（continuation）：user 消息含续接语模式
         且包含某 Matter 标题的关键词 -> 强合并许可。

    守宁分勿合：只在明确命中时触发，模糊指代不算命中。
    只在异步管线判（热路径不判）。
    """

    def __init__(
        self,
        *,
        continuation_patterns: list[str] | None = None,
        min_title_len: int = 4,
        min_keyword_len: int = 2,
    ) -> None:
        self._patterns = continuation_patterns or _DEFAULT_CONTINUATION_PATTERNS
        self._compiled = [re.compile(p, re.IGNORECASE) for p in self._patterns]
        self._min_title_len = min_title_len
        self._min_keyword_len = min_keyword_len

    def detect(
        self,
        fact: Fact,
        user_message: str,
        candidate_matters: list[Matter],
    ) -> ExplicitSignalMatch | None:
        """检测显式信号，返回匹配结果或 None。

        ADR-0018 真机回溯修复（B）：L2 显式信号是 turn 级（user 消息点名/续接
        某 Matter），但归属是 fact 级。一轮多主题时，turn 命中 Matter X 不能
        把该轮所有 fact 都链给 X。故每条候选命中都过 ``_fact_relevant`` 负向
        门控：fact 无主题信号（entities/proposal_titles 皆空）时信任 turn 信号
        放行；有主题信号但与该 Matter 无重叠则跳过（防 DNS fact 被链给阿根廷
        Matter）。门控未通过的候选不立即返回 None，继续找下一个相容候选。
        """
        if not user_message or not candidate_matters:
            return None

        msg_lower = user_message.lower()

        # 1. 点名：user 消息直接包含 Matter 标题（最强显式信号）
        for matter in candidate_matters:
            if matter.matter_id == UNASSIGNED_MATTER_ID:
                continue
            title = matter.title.strip()
            if len(title) < self._min_title_len:
                continue
            if title.lower() in msg_lower and self._fact_relevant(fact, matter):
                return ExplicitSignalMatch(
                    matter_id=matter.matter_id,
                    matched_text=title,
                    signal_type="title_mention",
                    title=title,
                )  # _fact_relevant：MS-16 统一原生键门控（见方法 docstring）

        # 2. 续接语 + 标题关键词
        matched_pattern = next(
            (p.pattern for p in self._compiled if p.search(user_message)),
            "",
        )
        if not matched_pattern:
            return None

        for matter in candidate_matters:
            if matter.matter_id == UNASSIGNED_MATTER_ID:
                continue
            title = matter.title.strip()
            if len(title) < self._min_title_len:
                continue
            for kw in self._extract_keywords(title):
                # MS-16：门控统一走原生键（title+aliases），accumulated entities
                # 已整体退出匹配面（ADR-0020 的 include_* 开关随之退役）。
                if self._kw_in_msg(kw, msg_lower) and self._fact_relevant(
                    fact, matter,
                ):
                    return ExplicitSignalMatch(
                        matter_id=matter.matter_id,
                        matched_text=kw,
                        signal_type="continuation",
                        pattern=matched_pattern,
                        title=title,
                    )

        return None

    def _kw_in_msg(self, kw: str, msg_lower: str) -> bool:
        """关键词是否出现在 user 消息中。

        ASCII 关键词用词边界匹配（``'Open'`` 不命中 ``'opencode'``）；CJK 用子串。
        """
        if not kw:
            return False
        kw_l = kw.lower()
        if kw.isascii():
            return re.search(r"\b" + re.escape(kw_l) + r"\b", msg_lower) is not None
        return kw_l in msg_lower

    def _fact_relevant(self, fact: Fact, matter: Matter) -> bool:
        """B 负向门控（MS-16 再门控，2026-08-10）：fact 与 Matter 主题是否相容。

        08-10 审计复现（scripts/audit_l2l3_gate_replay.py）：live 库 L2=466 里
        415 条走 `proposal_title == 累积 alias` 精确命中——雪球源是
        `_accumulate_matter_metadata` 把每个成员 fact 的 proposal_titles 无界
        追加进 aliases（README matter 累到 95 条），一条错边进来，它的提案
        标题就成了后续同题 fact 的合法匹配键；另有 43 条走单个泛词
        （'bladex'）的子串/实体命中。ADR-0020 修复只把 entities 请出了
        continuation 门控，aliases 这条累积通道漏掉了。

        新判定（配合 memory_index 停止 aliases 自动累积——aliases 回到
        「创建时原生键 + manual 追加」语义）：

        1. fact 无主题信号（entities / proposal_titles 皆空）-> 信任 turn
           显式信号，放行（generic 放行语义不变，复现中仅 8/466）。
        2. proposal_title 与原生键**全串规范化相等** -> 单票放行（整题相等
           是最强特异信号）。
        3. 弱命中（子串相含 / 实体命中原生键）：需 >= _GATE_MIN_WEAK_HITS 个
           **fact 侧独立信号**各自命中原生键——单个全项目高频 token
           （'bladex' ⊂ 标题）不再足以放行。
        """
        fact_titles_n = {_normalize(t) for t in fact.proposal_titles if t}
        fact_entities_n = {_normalize(e) for e in fact.entities if e}
        if not fact_titles_n and not fact_entities_n:
            return True

        native = _native_match_keys(matter)

        # 2) 提案标题全串相等（最强特异，单票放行）
        if fact_titles_n & native:
            return True

        # 3) 弱命中计票：fact 侧信号（entity / proposal_title）命中任一原生键
        #    （精确或子串相含）记 1 票；计票前做**子串包含去重**——命中信号 s
        #    若真包含另一命中信号 t，s 的证据被 t 覆盖不重复计
        #    （entity 'bladex' + 含 'bladex' 的 proposal 是同一 token 的两次
        #    出现，非独立证据）。'阿根廷'+'佛得角' 两实体各含于标题 = 2 票
        #    放行；'bladex' 单泛词无论出现几处只 1 票，阻断。
        hit_sigs = [
            sig for sig in sorted(fact_entities_n | fact_titles_n)
            if any(sig in mt or mt in sig for mt in native)
        ]
        independent = [
            s for s in hit_sigs
            if not any(t != s and t in s for t in hit_sigs)
        ]
        return len(independent) >= _GATE_MIN_WEAK_HITS

    def _extract_keywords(self, title: str) -> list[str]:
        """从标题提取关键词（过滤停用词 + 过短词 + 纯 ASCII 短片段）。

        ADR-0018 真机回溯修复（A）：``'阿根廷 vs 佛得角…'`` 原提取出 ``'vs'`` 当
        关键词，致任何含 "vs" 的消息命中阿根廷 Matter。纯 ASCII 短片段（vs/Memory Hub/v1）
        噪声大、过滤；CJK 2 字（修复/赔率）保留。整标题也入关键词（更特异，续接
        语路径可命中完整标题）。ASCII 关键词的匹配走词边界（见 ``_kw_in_msg``）。
        """
        words = re.split(r"[\s/_,\-\.()（）]+", title)
        keywords: list[str] = []
        for w in words:
            w = w.strip()
            if len(w) < self._min_keyword_len:
                continue
            if w.lower() in _STOP_WORDS:
                continue
            # 纯 ASCII 短片段（vs/Memory Hub/v1/to）噪声大，过滤
            if w.isascii() and len(w) < 3:
                continue
            keywords.append(w)
        # 整标题作为关键词（更特异；纯 CJK 无分隔符标题亦由此覆盖）
        full = title.strip()
        if len(full) >= self._min_title_len and full not in keywords:
            keywords.append(full)
        return keywords


# ── L3: 规范化工具 ──


def _normalize(s: str) -> str:
    """规范化字符串（L3 alias-link 用，ADR-0018 §3.2）。

    NFKC 全角->半角 + casefold + collapse whitespace。
    """
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.casefold().strip()
    s = " ".join(s.split())
    return s


def _native_match_keys(matter: Matter) -> set[str]:
    """Matter 的原生匹配键（MS-16）：title + aliases，规范化。

    aliases 语义已回归「创建时原生键（标题 + 首批成员实体/提案标题）+
    manual 追加」——`_accumulate_matter_metadata` 不再把成员 proposal_titles
    自动灌入（雪球源）。累积的 `Matter.entities` **不在**匹配面里
    （保留字段仅作展示/L4 观察）。

    注意：live 库在 T5 全量重建前，存量 matter 的 aliases 仍带历史累积污染；
    重建后 aliases 即原生形态，无需数据迁移（T5 一次兑现）。
    """
    keys: set[str] = set()
    if matter.title:
        keys.add(_normalize(matter.title))
    for a in matter.aliases:
        if a:
            keys.add(_normalize(a))
    return keys


def _deterministic_matter_id(title: str) -> str:
    """⚠️ **历史** id 规则：从 proposal 标题派生（T12d 重建等价性）。

    存量 Matter 的身份由它生成，**不要删**——重建旧数据、比对旧 id、
    以及 `new_matter_id` 在没有 ledger key 时的回落都要用它。
    新开的事走 `new_matter_id()`（ADR-0031 §3.4）。

    它当年解决的问题（"同一 proposal 标题 -> 同一 matter_id，多条同 proposal 的
    fact 汇入同一 provisional Matter"）**没有被放弃**，只是不再靠"标题即身份"
    这个巧合实现，改为在创建前显式查一次同名卡（见 `memory_index` 消化路）。
    """
    h = hashlib.sha256(_normalize(title).encode()).hexdigest()[:12]
    return f"m-{h}"


def new_matter_id(*, ledger_key: str, title: str = "") -> str:
    """一件事的身份 = **它首轮那条 Memory Hub key** 的哈希（ADR-0031 §3.4）。

    ## 为什么换（这是范式转换的技术前提，不是优化）

    旧规则 `sha256(_normalize(title))` 让 **Matter 的身份就是它的标题字符串**，
    于是"改标题"和"换一件事"在系统里是同一个动作：

        一轮只看得见一个动作 → 蒸馏产出动作句 → 动作句当身份哈希
        → 换个动词 = 一张新卡 → 碎片率 ≈80%（MQ-S17，14 张人工签收成 3 张）

    ADR-0031 的"首轮定身份 + 后续判延续"要求**改名不改身份**，不解耦就落不了地。

    ## 代价已经兑现过一次，记在这里防止再来

    2026-08-19 那次全量重建（`logs/rebuild-v7-20260819.log`）就是"M6 改了标题、
    而 id 还绑在标题上"的后果：`index_admin_events_replayed count=46`，其中
    **44 条 matter_merge 全部 `noop reason=source_missing`、成功合并 0**——
    历史上所有手动合并被静默丢弃。本函数消除的正是这个结构冲突。

    ## 重建等价性（必须写在这里，否则后人会以为破了 ADR-0009）

    首轮那一条 Turn 是**确定的、append-only 的、跨 rebuild 稳定的**：
    重放同样的 Memory Hub → 得到同样的首轮 → 同样的 id。**等价性保住。**
    相比之下旧规则的"稳定"依赖于蒸馏每次都产出同一个标题，而那是个 LLM 输出。

    ## 迁移：存量 id 不改

    存量 Matter 的 id 保持原样（改了等于全库失联）。本函数只在**新开一件事**时
    调用；没有 ledger key 的场景（合成 fact / 存量回放 / 测试构造）回落旧规则，
    保证任何调用方都拿得到一个确定性 id。
    """
    key = (ledger_key or "").strip()
    if not key:
        return _deterministic_matter_id(title)
    h = hashlib.sha256(key.encode()).hexdigest()[:12]
    return f"m-{h}"


class AttributionPipeline:
    """五层链接归属管线（ADR-0018 §3.2）。

    纯逻辑：输入 Fact + 候选 Matters，输出 AttributionDecision。
    不直接写存储--调用方（IndexConsolidationWorker）负责应用决策。
    L4 裁决经注入的 LinkJudge（proxy 侧 LLM 实现）+ JudgmentLedger（台账缓存）。
    """

    def __init__(
        self,
        *,
        link_judge: LinkJudge | None = None,
        judgment_journal: JudgmentJournalProtocol | None = None,
        explicit_detector: ExplicitSignalDetector | None = None,
        top_candidates: int = _DEFAULT_TOP_CANDIDATES,
        l3_entity_intersection: int = _DEFAULT_L3_ENTITY_INTERSECTION,
        dormant_timeout_s: float = _DEFAULT_DORMANT_TIMEOUT_S,
        # 台账缓存判决的目标是否仍然存在（None = 不校验 = 现行为，其它调用方零回归）。
        # 生产注入点在 memory_index；见 `_l4` 里的 `ledger_target_missing` 分支。
        matter_exists: Callable[[str], bool] | None = None,
        # 兼容旧调用（cosine 阈值已退役，ADR-0018 §2；接受但忽略）
        semantic_threshold: float | None = None,
    ) -> None:
        self._link_judge = link_judge
        self._judgment_ledger = judgment_journal
        self._explicit_detector = explicit_detector or ExplicitSignalDetector()
        self._top_candidates = top_candidates
        self._l3_entity_intersection = l3_entity_intersection
        self._dormant_timeout_s = dormant_timeout_s
        self._matter_exists = matter_exists
        #: 台账命中但目标已不在库的计数（孤边的直接来源，读数面）
        self.stale_judgment_targets = 0

        #: L4 LINK 裁决目标的**准入谓词**：`(fact, matter_id) -> 拒绝原因`（`""`=放行）。
        #:
        #: 🔴 为什么必须有这一道（2026-09-02，账本准入闸的第三个漏）：两条 L4 路径
        #: （台账缓存命中 / 现场裁决）都把 `result.matter_id` 直接落边，**从不校验
        #: 它还在不在 `linkable` 里**。于是调用方在候选池上做的任何过滤——例如
        #: ADR-0032 §4.5.1 的账本准入闸——都被 L4 绕过：闸把锚卡从候选里摘掉，
        #: 缓存判决又把它写了回来。实测第五次全量重建 4182 条归属漏 1 条
        #: （`from_ledger=True`，轮次早于账本创建 77.2 小时）。
        #: **凡是绕过候选池的落边路径，都要单独过一次闸**（与 CONT 直挂同型）。
        #:
        #: 公开属性而非构造参数：谓词依赖调用方在本类构造**之后**才装配好的状态
        #: （`memory_index` 的 `_lg_reject` 依赖账本池与切换时间轴），与
        #: `_consistency_checker` 的后置注入同款。`None` = 不校验（旧行为，零回归）。
        self.target_admissible: Callable[[Fact, str], str] | None = None
        #: 被上述谓词拦下的 L4 目标数，按拒因分。恒 0 = 谓词没接上或没开火，
        #: 两者要分开查（前者看调用方有没有赋值）。
        self.blocked_l4_targets: Counter[str] = Counter()
        # semantic_threshold 保留参数位以兼容旧脚本/测试构造，逻辑中不再使用
        self._deprecated_semantic_threshold = semantic_threshold

    @property
    def top_candidates(self) -> int:
        """L4 e5 召回 top-k（供 worker 读取）。"""
        return self._top_candidates

    # ── 单条归属（委托批量，L4 单项裁决）──

    def attribute(
        self,
        fact: Fact,
        candidate_matters: list[Matter],
        *,
        manual_matter_id: str | None = None,
        user_message: str | None = None,
        now: datetime | None = None,
    ) -> AttributionDecision:
        """判定一条 Fact 归属（L1->L2->L3->L4->L5）。

        candidate_matters: 该 fact 的 e5 召回候选（含本 pass 新建 Matter）。
        L4 经 link_judge 裁决（有台账则命中复用）。
        """
        decisions = self.attribute_batch(
            [fact], candidate_matters,
            manual_matter_ids={fact.id: manual_matter_id} if manual_matter_id else None,
            user_messages={fact.id: user_message} if user_message else None,
            now=now,
        )
        return decisions[0]

    def attribute_batch(
        self,
        facts: list[Fact],
        candidate_matters: list[Matter] | None = None,
        *,
        candidates_by_fact: dict[str, list[Matter]] | None = None,
        manual_matter_ids: dict[str, str] | None = None,
        user_messages: dict[str, str] | None = None,
        now: datetime | None = None,
    ) -> list[AttributionDecision]:
        """批量归属：逐条 L1–L3 + L4 合并一次裁决（ADR-0018 §3.2 可批量）。

        2026-09-03 S4：原先这里前置 L0「同单元同属」（同 `unit_key` 只让代表走判定、
        成员继承 `source=SAME_UNIT`）。L0 默认关、从未翻默认、零 live 读数，且 MQ-V6
        实测碎片化瓶颈在候选生成/判定两层、与任务单元无关——连函数删除，本方法
        退化为 `_attribute_primary` 的公开名（每条 fact 各自独立走管线）。
        `unit_key` 字段与 schema 保留（轮内切分单位，ADR-0024 降级后的用途）。
        """
        return self._attribute_primary(
            facts, candidate_matters,
            candidates_by_fact=candidates_by_fact,
            manual_matter_ids=manual_matter_ids,
            user_messages=user_messages, now=now,
        )

    def _attribute_primary(
        self,
        facts: list[Fact],
        candidate_matters: list[Matter] | None = None,
        *,
        candidates_by_fact: dict[str, list[Matter]] | None = None,
        manual_matter_ids: dict[str, str] | None = None,
        user_messages: dict[str, str] | None = None,
        now: datetime | None = None,
    ) -> list[AttributionDecision]:
        """L1-L5 批量判定（L1-L3 逐条；L4 合并一次裁决调用，ADR-0018 §3.2 可批量）。

        参数：
          candidate_matters: 共享候选池（candidates_by_fact 未提供时所有 fact 共用）。
          candidates_by_fact: {fact_id: 该 fact 的 e5 召回候选}（优先，rebuild 用）。
          manual_matter_ids: {fact_id: matter_id} 已有手动归属（L1 最高优先）。
          user_messages: {fact_id: user_message} 本轮 user 消息（L2 显式信号用）。
        """
        if now is None:
            now = datetime.now(UTC)
        manual = manual_matter_ids or {}
        messages = user_messages or {}

        # L1-L3 逐条判定，未解决的进 L4 待裁决队列
        decisions: dict[str, AttributionDecision] = {}
        l4_pending: list[tuple[Fact, list[Matter]]] = []  # (fact, its candidates)

        for fact in facts:
            cands = (
                candidates_by_fact.get(fact.id, [])
                if candidates_by_fact is not None
                else (candidate_matters or [])
            )
            dec = self._l1_l3(fact, cands, manual.get(fact.id), messages.get(fact.id), now)
            if dec is not None:
                decisions[fact.id] = dec
            else:
                # L1-L3 未命中 -> L4
                l4_pending.append((fact, cands))

        # L4：台账命中复用 + miss 批量裁决（一次 LLM 调用）
        l4_to_judge: list[tuple[int, Fact, list[Matter]]] = []  # (orig_index, fact, cands)
        for idx, (fact, cands) in enumerate(l4_pending):
            # 无候选 -> 直接 L5（不送 judge，ADR-0018 §3.2 "召回不到候选=直接走 L5"）
            linkable = [c for c in cands if self._is_linkable(c)]
            if not linkable:
                decisions[fact.id] = self._l5(fact, None, judge_ran=False, now=now)
                continue
            # 台账命中复用（LINK -> LLM_LINK；NONE/UNCERTAIN -> 走 L5，与现场裁决同路径）
            if self._judgment_ledger is not None:
                cached = self._safe_get_judgment(fact.id)
                if cached is not None:
                    if (cached.verdict == LinkVerdict.LINK
                            and self._stale_target(cached.matter_id, linkable)):
                        # 🔴 台账命中、但它点名的 Matter 这一轮**不存在**了。
                        # 此前直接照写（`_apply_l4_verdict` 收下 candidates 却不用），
                        # 于是边指向一个不存在的卡 = **孤边**：卡不在库所以任何按
                        # matter 聚合的读数都看不见它，边还在所以成员计数照算。
                        # 2026-09-01 实测 259 条真孤边 / 50 个 id，**全部 L4**
                        # （清理前 250 条 ⇒ 存量问题，非墓碑引入）。
                        #
                        # 落 L5 而不是回退调 LLM：全量重建的零 LLM 是硬约束
                        # （`ledger_hits=3286 / calls=0`），把缓存失效变成 LLM 调用
                        # 会让重建代价随存量线性上涨。L5 = 有 proposal 就新开
                        # provisional（≥2 成员才转正），比挂到鬼卡上保守。
                        self.stale_judgment_targets += 1
                        decisions[fact.id] = self._l5(
                            fact, None, judge_ran=True, now=now)
                        continue
                    if cached.verdict == LinkVerdict.LINK:
                        # 准入谓词（见 `target_admissible`）：缓存判决绕过候选池，
                        # 调用方的过滤只有在这里再问一次才管得到它。
                        _rej = self._l4_reject(fact, cached.matter_id)
                        if _rej:
                            decisions[fact.id] = self._l5(
                                fact, None, judge_ran=True, now=now)
                        else:
                            decisions[fact.id] = self._apply_l4_verdict(
                                fact, linkable, cached, now, from_ledger=True)
                    else:
                        decisions[fact.id] = self._l5(
                            fact, cached, judge_ran=True, now=now)
                    continue
            l4_to_judge.append((idx, fact, linkable))

        if l4_to_judge:
            l4_results = self._run_l4_batch(l4_to_judge)
            for (_idx, fact, linkable), result in l4_results:
                if result.verdict == LinkVerdict.LINK:
                    # 现场裁决同样过闸 —— **纵深防御，不是主防线**。
                    # 主防线是 `_run_l4_batch` 里的候选成员校验
                    # （`link_id_not_in_candidates` ⇒ 降级 UNCERTAIN）：候选池已被
                    # 调用方过滤过，被摘掉的卡模型点名了也会在那里被拦下。
                    # ⚠️ 我一度断言"这一半也在漏、两半生产里都活着"——**没验证就说了**，
                    # 读码后不成立。留这一句是让不变量落在本地，而不是依赖 80 行外
                    # 另一个方法里的检查；它在当前代码下**不会开火**。
                    if self._l4_reject(fact, result.matter_id):
                        decisions[fact.id] = self._l5(
                            fact, None, judge_ran=True, now=now)
                    else:
                        decisions[fact.id] = self._apply_l4_verdict(
                            fact, linkable, result, now, from_ledger=False)
                else:
                    # NONE / UNCERTAIN -> L5
                    decisions[fact.id] = self._l5(fact, result, judge_ran=True, now=now)

        # 保持输入顺序返回
        return [decisions[f.id] for f in facts]

    # ── L1-L3 ──

    def _l1_l3(
        self,
        fact: Fact,
        candidates: list[Matter],
        manual_matter_id: str | None,
        user_message: str | None,
        now: datetime,
    ) -> AttributionDecision | None:
        """L1 manual -> L2 explicit -> L3 alias-link。未命中返回 None（进 L4）。"""
        # L1 manual（最高，ADR-0012 §3.5 manual 压过 auto）
        if manual_matter_id is not None:
            return AttributionDecision(
                fact_id=fact.id, matter_id=manual_matter_id, confidence=1.0,
                source=AttributionSource.MANUAL, signal="manual",
                decision={"layer": "L1", "verdict": f"link:{manual_matter_id}", "ts": now.isoformat()},
            )

        linkable = [c for c in candidates if self._is_linkable(c)]

        # L2 explicit（对话点名/续接语）
        if user_message and linkable:
            match = self._explicit_detector.detect(fact, user_message, linkable)
            if match is not None:
                return AttributionDecision(
                    fact_id=fact.id, matter_id=match.matter_id,
                    confidence=_EXPLICIT_CONFIDENCE, source=AttributionSource.EXPLICIT,
                    signal=f"explicit:{match.signal_type}:{match.matched_text}",
                    decision={"layer": "L2", "signal": match.signal_type,
                              "verdict": f"link:{match.matter_id}", "ts": now.isoformat()},
                )

        # L3 alias-link（proposal 标题/entities ↔ Matter.aliases 规范化匹配）
        if linkable:
            hit = self._alias_link(fact, linkable)
            if hit is not None:
                matter, reason = hit
                return AttributionDecision(
                    fact_id=fact.id, matter_id=matter.matter_id,
                    confidence=_ALIAS_CONFIDENCE, source=AttributionSource.ALIAS,
                    signal=f"alias:{reason}",
                    decision={"layer": "L3", "verdict": f"link:{matter.matter_id}",
                              "matched": reason, "ts": now.isoformat()},
                )

        return None  # 进 L4

    def _is_linkable(self, matter: Matter) -> bool:
        """候选是否可被 L2-L4 链接（排除未归属池与已关闭）。"""
        return (matter.matter_id != UNASSIGNED_MATTER_ID
                and matter.status != MatterStatus.CLOSED)

    def is_linkable(self, matter: Matter) -> bool:
        """`_is_linkable` 的公开面，供候选生成方预筛（GM-1/GM-2 的确定性候选通路）。

        候选生成在 `memory_index` 侧，判定在本管线——两边对"什么算可链接候选"必须
        用同一把尺子，否则确定性通路会把不可链接的 Matter 塞进池子白占 top-k 名额。
        """
        return self._is_linkable(matter)

    def _alias_link(
        self, fact: Fact, candidates: list[Matter],
    ) -> tuple[Matter, str] | None:
        """L3: proposal 标题 ↔ 原生键规范化相等，或 entities ∩ 原生键 ≥门槛。

        MS-16（T4a-2，08-10 审计）：实体交集的对照面从累积 `matter.entities`
        并集换成原生键（title + aliases = 标题派生 + 首批成员 + manual）。
        雪球机理：一条错边混入外题实体后，`_accumulate_matter_metadata` 把它
        并进 matter.entities，该实体后续所有 fact 以 conf=0.9 持续 L3 链入
        恶性累积（永安←泰山 25 条边即此形态）。两方案（原生键 / 并集+同题
        门槛）取实现代价小者=原生键：首批成员实体本就在创建时 aliases 里
        （`new_matter_aliases = [title]+entities`），零 schema 迁移；而
        "同题门槛"需要一个可靠同题信号——坏门控正是因为缺它，循环依赖。
        """
        proposal_titles_n = {_normalize(t) for t in fact.proposal_titles if t}
        # T4b：fact.topic 加入标题对照面（与 Matter title/aliases 全串相等比对）。
        # 只作证据输入——门槛（全串相等 / entity 交集 >=_l3_entity_intersection）
        # 与 first-match-wins 零改动；topic 为空（存量 fact / v5 前台账）= 现状。
        topic_n = _normalize(getattr(fact, "topic", "") or "")
        if topic_n:
            proposal_titles_n = proposal_titles_n | {topic_n}
        fact_entities_n = {_normalize(e) for e in fact.entities if e}

        for matter in candidates:
            native = _native_match_keys(matter)
            # 标题规范化相等
            if proposal_titles_n and (proposal_titles_n & native):
                return matter, "title_alias_match"
            # 实体 ∩ 原生键 ≥门槛
            if fact_entities_n and self._l3_entity_intersection > 0:
                if len(fact_entities_n & native) >= self._l3_entity_intersection:
                    return matter, "entity_intersection"

        # ── L3 主题键子层（2026-08-14 拍板 A/B，matter-dedup-keywords 卡）──
        # 全串/实体交集都 miss 后，用**名词性主题键的多键重叠**兜同事儿：
        # 任务句标题动词一换（核实/查询/确认/查明）全串就 miss，而主题键
        # （泰安仁信/泰山啤酒/增资扩股）跨轮稳定——四开事故的机制解。
        # 高精度门槛（≥3 键或 ≥2 键且 Jaccard≥0.6，见 topic_keys 模块）+
        # 取重叠最大者（不搞 first-match-wins，防兄弟并存时挂错个）。
        # fact 无 keywords（存量台账）走确定性派生兜底；matter 无 topic_keys
        # （存量 Matter，未经新代码累积）→ 子层对它天然 inert = 零回归。
        fact_keys = derive_topic_keys(
            getattr(fact, "keywords", None),
            getattr(fact, "topic", "") or "",
            fact.proposal_titles, fact.entities)
        if fact_keys:
            best: tuple[int, float, Matter] | None = None
            for matter in candidates:
                # 🔴 matter 侧**只认累积的 topic_keys**（成员 keywords 计票），
                # 不做 entities/title 派生兜底——派生会把 MS-16 杀掉的
                # "污染 entities 匹配面"复活（永安←泰山：错边混入的外题实体
                # 派生成键 → 同题 fact 以 conf 0.9 自动链入 = 雪球回归，
                # `test_l3_accumulated_entities_no_longer_match` 当场抓获）。
                # 派生兜底只允许进**人审提案**（matter_merge_candidates），
                # 不允许进自动链接。存量 Matter 因此对子层 inert = 已知限制，
                # 由手动 merge + 全量重建收敛。
                mkeys = getattr(matter, "topic_keys", None) or {}
                if not mkeys:
                    continue
                inter, jac = topic_key_overlap(fact_keys, mkeys)
                if not is_strong_topic_match(fact_keys, mkeys):
                    continue
                if best is None or (inter, jac) > (best[0], best[1]):
                    best = (inter, jac, matter)
            if best is not None:
                logger.info(
                    "l3_topic_keys_overlap fact=%s matter=%s overlap=%d jaccard=%.2f",
                    fact.id, best[2].matter_id, best[0], best[1])
                return best[2], "topic_keys_overlap"
        return None

    # ── L4 ──

    def _run_l4_batch(
        self,
        items: list[tuple[int, Fact, list[Matter]]],
    ) -> list[tuple[tuple[int, Fact, list[Matter]], LinkJudgeResult]]:
        """L4 批量裁决：一次 LLM 调用裁决多条 fact + 写台账。"""
        if self._link_judge is None:
            # judge 不可用 -> 全部 uncertain（降级留池，ADR-0018 §3.7）
            return [(tup, LinkJudgeResult(fact_id=tup[1].id, verdict=LinkVerdict.UNCERTAIN,
                                          reason="judge_unavailable"))
                    for tup in items]

        judge_items = [
            LinkJudgeItem(
                fact_id=fact.id, content=fact.content,
                entities=list(fact.entities),
                proposal_titles=list(fact.proposal_titles),
                topic=getattr(fact, "topic", "") or "",   # T4b：证据输入
                candidates=[self._candidate_card(c) for c in linkable],
            )
            for _, fact, linkable in items
        ]

        try:
            results = self._link_judge.judge_links(judge_items)
        except Exception as e:  # noqa: BLE001
            logger.warning("l4_judge_batch_failed err=%s -> all uncertain", e)
            results = [LinkJudgeResult(fact_id=it.fact_id, verdict=LinkVerdict.UNCERTAIN,
                                       reason=f"judge_error:{e}") for it in judge_items]

        # 结果数对齐保护
        if len(results) != len(judge_items):
            logger.warning("l4_judge_result_count_mismatch expect=%d got=%d",
                           len(judge_items), len(results))
            by_id = {r.fact_id: r for r in results}
            results = [by_id.get(it.fact_id, LinkJudgeResult(
                fact_id=it.fact_id, verdict=LinkVerdict.UNCERTAIN, reason="missing"))
                for it in judge_items]

        # 写台账 + 校验 LINK 的 matter_id 在候选内
        out: list[tuple[tuple[int, Fact, list[Matter]], LinkJudgeResult]] = []
        for (tup, result) in zip(items, results, strict=True):
            _, fact, linkable = tup
            cand_ids = {c.matter_id for c in linkable}
            if result.verdict == LinkVerdict.LINK and result.matter_id not in cand_ids:
                # LLM 幻觉了不存在的 matter_id -> 降级 uncertain（守宁分勿合）
                logger.warning("l4_judge_link_id_not_in_candidates fact=%s id=%s -> uncertain",
                               fact.id, result.matter_id)
                result = result.model_copy(update={
                    "verdict": LinkVerdict.UNCERTAIN, "reason": "link_id_not_in_candidates"})

            if self._judgment_ledger is not None:
                cand_dicts = [c.model_dump(mode="json") for c in (
                    self._candidate_card(c) for c in linkable)]
                self._safe_put_judgment(fact.id, cand_dicts, result)
            out.append((tup, result))
        return out

    def _candidate_card(self, matter: Matter) -> LinkJudgeCandidate:
        """构造给 LLM 的候选登记卡（v1: recent_members 留空，summary 兜底，留实测补成员 fact）。"""
        return LinkJudgeCandidate(
            matter_id=matter.matter_id, title=matter.title,
            aliases=list(matter.aliases), summary=matter.summary,
        )

    def _apply_l4_verdict(
        self, fact: Fact, candidates: list[Matter], result: LinkJudgeResult,
        now: datetime, *, from_ledger: bool,
    ) -> AttributionDecision:
        """应用 L4 LINK 裁决（台账命中或刚裁决）。"""
        return AttributionDecision(
            fact_id=fact.id, matter_id=result.matter_id,
            confidence=_LLM_LINK_CONFIDENCE, source=AttributionSource.LLM_LINK,
            summary_rewrite=result.summary_rewrite,
            signal=f"llm_link:{'ledger' if from_ledger else 'judge'}",
            decision={"layer": "L4", "verdict": f"link:{result.matter_id}",
                      "judge_model": getattr(self._link_judge, "model_name", ""),
                      "from_ledger": from_ledger, "reason": result.reason,
                      "ts": now.isoformat()},
        )

    def _l4_reject(self, fact: Fact, matter_id: str) -> str:
        """L4 LINK 目标过一次调用方的准入谓词。放行返回 ""，否则是可 grep 的拒因。

        与 `_stale_target` 分工：那条问「卡还在不在」（存在性），本条问
        「这条 fact 能不能挂到这张卡上」（准入）。**被闸摘出候选池的锚卡
        仍然存在**，所以存在性校验对这个漏完全无效——两条都得有。

        谓词未接（`target_admissible is None`）⇒ 恒放行，其它调用方零回归。
        谓词自己抛异常 ⇒ 按放行处理并留日志：仪器坏了不该改变归属结果
        （与 `_stale_target` 同一取舍）。
        """
        if not matter_id or matter_id == UNASSIGNED_MATTER_ID:
            return ""
        if self.target_admissible is None:
            return ""
        try:
            reason = self.target_admissible(fact, matter_id) or ""
        except Exception as e:  # noqa: BLE001 —— 谓词失败不改变归属
            # 🔴 `%`-style：本模块用 **stdlib logging**（库随宿主，CLAUDE.md 日志约定），
            # 传 kwargs 会 `TypeError: Logger._log() got an unexpected keyword argument`
            # —— 首版按 structlog 风格写，于是这条"兜底"自己会抛，把
            # "谓词坏了不改归属"变成"谓词坏了整轮崩"。同文件其余 7 处都是 %-style。
            logger.warning("l4_admissible_probe_failed fact=%s matter=%s err=%s",
                           fact.id, matter_id, e)
            return ""
        if reason:
            self.blocked_l4_targets[reason] += 1
        return reason

    def _stale_target(self, matter_id: str, linkable: list[Matter]) -> bool:
        """台账缓存判决点名的 Matter 是不是已经不在了。

        两级判定，**先看手上的、再问外面**：
        ① 它就在本轮召回的候选里 ⇒ 一定还在，直接放行（零成本，覆盖绝大多数）；
        ② 否则问注入进来的存在性谓词。没有谓词（`None`）就**放行**——
           保持现行为，让其它调用方（测试 / 评测回放 / 嵌入式）零回归。

        🔴 不用"不在候选里就算失效"：候选是 top-k 召回，卡还在但这轮没被召回
        是常态，那样判会把大量正常复用打成失效、把重建推回 LLM。
        """
        if not matter_id or matter_id == UNASSIGNED_MATTER_ID:
            return False
        if any(c.matter_id == matter_id for c in linkable):
            return False
        if self._matter_exists is None:
            return False
        try:
            return not self._matter_exists(matter_id)
        except Exception as e:  # noqa: BLE001 —— 谓词自身出错不该改变归属结果
            logger.warning("matter_exists_probe_failed matter=%s err=%s", matter_id, e)
            return False

    def _safe_get_judgment(self, fact_id: str) -> LinkJudgeResult | None:
        try:
            return self._judgment_ledger.get_latest_judgment(fact_id)  # type: ignore[union-attr]
        except Exception as e:  # noqa: BLE001
            logger.warning("judgment_ledger_get_failed fact=%s err=%s", fact_id, e)
            return None

    def _safe_put_judgment(
        self, fact_id: str, candidates: list[dict], result: LinkJudgeResult,
    ) -> None:
        try:
            self._judgment_ledger.put_judgment(fact_id, candidates, result)  # type: ignore[union-attr]
        except Exception as e:  # noqa: BLE001
            logger.warning("judgment_journal_put_failed fact=%s err=%s", fact_id, e)

    # ── L5 ──

    def _l5(
        self,
        fact: Fact,
        l4_result: LinkJudgeResult | None,
        *,
        judge_ran: bool,
        now: datetime,
    ) -> AttributionDecision:
        """L5 兜底：none->provisional 新开；uncertain/无 judge->留池（ADR-0018 §3.2/§3.7）。

        规则：
          - judge 不可用（judge_ran=False 且无 l4_result）-> 留池 + pending（降级）
          - verdict UNCERTAIN -> 留池 + pending
          - verdict NONE 或无候选 -> 有 proposal 则新开 provisional，否则留池
        """
        verdict = l4_result.verdict if l4_result is not None else None

        # 降级：judge 不可用（l4_result 为 None 且本应裁决）或 UNCERTAIN -> 留池
        if l4_result is not None and verdict == LinkVerdict.UNCERTAIN:
            return self._unassigned(fact, now, pending=True,
                                    reason="l4_uncertain",
                                    judge_model=getattr(self._link_judge, "model_name", ""))
        if l4_result is None and not judge_ran:
            # judge 不可用（_run_l4_batch 已对无候选走 L5，此处 judge_ran=False 仅当无候选）
            # 无候选 + 有 proposal -> 仍可新开 provisional（ADR §3.2 "召回不到候选=直接走 L5"）
            pass

        # NONE 或无候选：有 proposal -> 新开 provisional；无 proposal -> 留池
        title = next((t for t in fact.proposal_titles if t.strip()), "")
        if len(title.strip()) >= _MIN_PROVISIONAL_TITLE_CHARS:
            # ADR-0031 §3.4：身份取**这件事首轮那条 ledger key**。走到 L5 = L1–L4
            # 都没链上 = 这条 fact 开的是新的事，所以"它所在的这一轮"就是首轮。
            # 同标题的汇入不靠 id 巧合了，靠 L3 `title_alias_match`（全串相等）——
            # 前提是那张姐妹卡在候选里，而那正是 G12.2 活跃档案窗口要保证的事。
            _lk = getattr(fact, "source_ledger_key", "") or ""
            if not _lk:
                # 降级必须可 grep：没有 ledger key 就退回旧的标题哈希，
                # 也就是**这条 fact 的 G12.1 没生效**。生产候选一定带 ledger key
                # （fact id 本身就是 `_deterministic_fact_id(ledger_key, content)` 派生的），
                # 所以这条日志一旦成规模出现，说明有一条通路没把它传下来
                # ——那种情况下解耦会静默变成 no-op（"机制存在但生产不走"的第六例）。
                logger.info("matter_id_fallback_no_ledger_key fact=%s title=%s",
                            fact.id, title[:40])
            matter_id = new_matter_id(ledger_key=_lk, title=title)
            # 拍板 D（2026-08-14）：alias 种子过毒词——此前 entities 原样入 alias，
            # opencli/Chrome/'16%股份转让'/'张开利62.4' 全成了 L3 全串匹配键
            # （既是误合并隐患也是噪声）。标题永远保留。
            clean_entities = [
                e for e in fact.entities
                if e and is_valid_topic_key(normalize_key(e))
            ]
            aliases = _dedupe([title] + clean_entities)
            return AttributionDecision(
                fact_id=fact.id, matter_id=matter_id, confidence=0.0,
                source=AttributionSource.PROVISIONAL, is_new_matter=True,
                new_matter_title=title, new_matter_aliases=aliases,
                # 拍板 B：主题键种子（创始 fact 派生）——没有它，下一条动词
                # 变体 proposal 照样开第二个 Matter（本卡四开事故的机理）。
                new_matter_topic_keys=derive_topic_keys(
                    getattr(fact, "keywords", None),
                    getattr(fact, "topic", "") or "",
                    fact.proposal_titles, fact.entities),
                signal="provisional:new_from_proposal",
                decision={"layer": "L5", "verdict": "none->provisional",
                          "proposal_title": title, "ts": now.isoformat()},
            )

        return self._unassigned(fact, now, pending=False, reason="no_proposal",
                                judge_model=getattr(self._link_judge, "model_name", ""))

    def _unassigned(
        self, fact: Fact, now: datetime, *, pending: bool, reason: str,
        judge_model: str = "",
    ) -> AttributionDecision:
        return AttributionDecision(
            fact_id=fact.id, matter_id=UNASSIGNED_MATTER_ID, confidence=0.0,
            source=AttributionSource.UNASSIGNED, pending_judgment=pending,
            signal=f"unassigned:{reason}",
            decision={"layer": "L5", "verdict": "unassigned", "reason": reason,
                      "pending": pending, "judge_model": judge_model,
                      "ts": now.isoformat()},
        )

    # ── 生命周期（provisional 不自动降级）──

    def update_lifecycles(
        self,
        matters: list[Matter],
        *,
        now: datetime | None = None,
    ) -> list[tuple[str, MatterStatus, MatterStatus]]:
        """检查所有 Matter 的生命周期状态（ADR-0012 §3.4）。

        返回变更列表：(matter_id, 旧状态, 新状态)。
        provisional/closed 不自动降级（provisional 等待成员，closed 只手动）。
        """
        if now is None:
            now = datetime.now(UTC)

        changes: list[tuple[str, MatterStatus, MatterStatus]] = []
        for matter in matters:
            if matter.status != MatterStatus.ACTIVE:
                continue  # provisional/dormant/closed 不自动降级

            elapsed = (now - matter.updated_at).total_seconds()
            if elapsed > self._dormant_timeout_s:
                old_status = matter.status
                matter.status = MatterStatus.DORMANT
                changes.append((matter.matter_id, old_status, MatterStatus.DORMANT))
                logger.info(
                    "matter_dormant_downgrade matter_id=%s elapsed_s=%.0f",
                    matter.matter_id, elapsed,
                )

        return changes

    def reactivate_on_hit(self, matter: Matter) -> bool:
        """dormant Matter 被归属命中时唤醒（ADR-0012 §3.4）。返回是否变更。"""
        if matter.status == MatterStatus.DORMANT:
            matter.status = MatterStatus.ACTIVE
            matter.updated_at = datetime.now(UTC)
            logger.info("matter_reactivated matter_id=%s", matter.matter_id)
            return True
        return False

    # ── Matter / Edge 构造（caller 应用决策时调用）──

    def create_matter_for_decision(
        self,
        decision: AttributionDecision,
        centroid: list[float] | None = None,
        scope: str = "",
    ) -> Matter:
        """为 PROVISIONAL 决策创建 Matter 对象（caller 负责存储）。

        新 Matter 从 proposal 诞生（标题=提案标题），初始 provisional，
        aliases = 标题 + 实体（L3 匹配键）。matter_id 确定性（T12d 重建等价）。

        `scope`（ADR-0021 §2.4 补平，2026-08-16）：**取创始 fact 的 scope**。
        此前四个 Matter 构造点全都不传，live 120/120 恒空 —— 字段有定义、读侧按
        "已迁移"设计，写侧从未落地（ADR-0026 铁律三的镜像：无生产者的字段被当作可用）。
        🔴 **后续成员 fact 即使 scope 更宽也不在这里提升**：提升 = 扩大可见面 =
        安全边界，只走 `SCOPE_PROMOTE` 显式管理操作（与 ADR-0021 §3.3 一致，fail-closed）。
        """
        # decision.matter_id 正常总是由 _l5 填好（已按 ADR-0031 §3.4 取 ledger key）；
        # 这里的回落只覆盖"手工构造 decision"的调用方，故只能退回标题规则。
        matter_id = decision.matter_id or _deterministic_matter_id(decision.new_matter_title)
        return Matter(
            matter_id=matter_id,
            scope=scope,
            title=decision.new_matter_title,
            status=MatterStatus.PROVISIONAL,
            centroid=centroid,
            centroid_weight=1.0 if centroid is not None else 0.0,
            aliases=list(decision.new_matter_aliases),
            entities=[],
            origin=MatterOrigin.AUTO,
            # 拍板 B：创始主题键种子（计数 1）——provisional 可被 L2-L4 链接
            # （_is_linkable 只排除 unassigned/closed），有种子后同批兄弟
            # proposal 走 L3 主题键子层挂进来，而不是再开一个。
            topic_keys=dict.fromkeys(getattr(decision, "new_matter_topic_keys", []) or [], 1),
        )

    def create_edge_for_decision(
        self,
        decision: AttributionDecision,
        fact: Fact,
    ) -> MatterEdge:
        """为归属决策创建归属边（带 decision 记录，ADR-0018 §3.5）。"""
        provenance = (EdgeProvenance.MANUAL
                      if decision.source == AttributionSource.MANUAL
                      else EdgeProvenance.AUTO)
        return MatterEdge(
            matter_id=decision.matter_id,
            target_type=EdgeTargetType.FACT,
            target_key=fact.id,
            provenance=provenance,
            confidence=decision.confidence,
            decision=dict(decision.decision),
        )


# ── 工具函数 ──


def _dedupe(items: list[str]) -> list[str]:
    """保序去重（aliases/entities 用，保证序列化顺序确定）。"""
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out
