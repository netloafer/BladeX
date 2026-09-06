"""主题键（topic keys）——Matter 去重与 Matter↔Fact 连接的名词性不变量。

立卡 `docs/planning/matter-dedup-keywords-20260814.md`（拍板 A/B/D）。

## 为什么需要这个模块（2026-08-14 实测）

同一件事（泰安仁信入股泰山啤酒）当天被开出 4 个 Matter——标题是**任务句**
（核实/查询/确认/查明 + 宾语），LLM 每轮换个动词，L3 全串规范化匹配永不相等；
而四者的**名词键**（泰安仁信/泰山啤酒/入股方式/增资扩股）完全重合。
蒸馏 v5 本来就产出轮级 keywords（≤10 名词性关键词），此前定为瞬态被丢弃
——本模块把它接为归属信号。

## 设计约束

- **叶子模块**：不 import bladex_core 其它模块（attribution/memory_index 单向依赖它）。
- **高精度优先**：主题键匹配服务于"同事儿识别"，误合并率=0 是第一红线
  （MS-16/T4a 拍板）——所以是多键重叠门槛（≥3 或 ≥2 且 Jaccard≥0.6），
  单键命中永不放行（防 'bladex' 单泛词吸尘器复发）。
- **毒词过滤**（拍板 D）：工具名/纯数字/百分比/日期碎片不得成为匹配键。
  实测污染样本：opencli / Chrome / 企信宝 / '16%股份转让' / '张开利62.4' /
  '2025.10审计负债率106.6' 全部进了 Matter aliases 当 L3 合法匹配键。
"""

from __future__ import annotations

import re
import unicodedata

# ── 过滤表 ──────────────────────────────────────────────────────────────────

# 通用工具/基建词：出现在几乎每个会话里，作为主题键零特异性（吸尘器种子）。
# 注意 企信宝/天眼查：它们是查询工具，任何尽调会话都会出现——主题是"查谁"，
# 不是"用什么查"。与 attribution._STOP_WORDS 分开维护：那边服务 L2 续接语
# 关键词提取（标题内词），这边服务主题键准入（跨轮不变量），语义不同。
TOOL_WORDS: set[str] = {
    "opencli", "openclaw", "chrome", "browser", "playwright",
    "web_search", "websearch", "curl", "wget", "bash", "python", "pytest",
    "git", "github", "docker", "redis", "rocksdb", "lancedb",
    "html", "http", "https", "json", "csv", "markdown", "pdf",
    "api", "url", "token", "tokens", "prompt", "dashboard", "proxy",
    "企信宝", "天眼查", "爱企查",
}

# 停用/泛词（主题键语境）：单独出现无判别力。
STOP_KEYS: set[str] = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "this", "that", "it", "we", "you", "task", "issue", "matter", "thing",
    "report", "update", "vs",
    "的", "了", "在", "是", "和", "与", "或", "并", "把", "被",
    "报告", "更新", "任务", "问题", "情况", "方式", "内容", "信息",
    "用户", "要求", "询问", "查询", "核实", "确认", "查明", "调查",
}

# 纯数字/标点/百分号/日期碎片（'16%'、'2022-11-21'、'2.0607%'）
_NUMERICISH = re.compile(r"^[\d\s.,:;%/+\-~—×xX*()（）]+$")

# 主题键长度边界：CJK ≥2 字；ASCII ≥3 字符（vs/v1/e.g 类噪声）
_MIN_CJK = 2
_MIN_ASCII = 3

_SPLIT = re.compile(r"[\s/_,\-\.()（）:：;；、|]+")


def normalize_key(s: str) -> str:
    """NFKC 全角→半角 + casefold + 空白折叠（与 attribution._normalize 同语义）。"""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.casefold().strip()
    return " ".join(s.split())


# ── 实体键规范化（V-I2，ADR-0032 §6-3 前置）────────────────────────────────
#
# 已知病例（2026-08-20 注入归因）：fact 实体 `qwen3.8:27b` 与 query 里的
# `qwen3.8-27B` 因冒号/连字符变体全灭 entity 通道 → RRF 少一票 → 该注入的
# 记忆掉出 top-k。分隔符在野外是可互换的（`:`/`-`/`_`/`/`/空格/`·`），
# 折成同一键；**`.` 有意保留**——`glm-5.2` 与 `glm-52` 是两个版本，
# 折掉点号就是误折叠（阴性对照钉死这条边界）。
#
# 🔴 entity 抽取与 topic_keys 共用这一套（V-I2 卡）：任何消费 `Fact.entities`
# 做相等/包含/Jaccard 的地方，两侧都必须先过 `canonical_entity_key`，
# 不许各自为政再长出一把不同的尺。

_SEP_VARIANTS = re.compile(r"[:_/·\s]+")


def canonical_entity_key(s: str) -> str:
    """实体名 → 规范键：`normalize_key` 之上折叠分隔符变体到 `-`。

    `qwen3.8:27b` ≡ `qwen3.8-27B` ≡ `qwen3.8_27b` → `qwen3.8-27b`；
    `glm-5.2` ≠ `glm-52`（点号保留）。整段文本也可用（query 侧同尺折叠）。
    """
    k = normalize_key(s)
    if not k:
        return ""
    k = _SEP_VARIANTS.sub("-", k)
    k = re.sub(r"-{2,}", "-", k).strip("-")
    return k


def _digit_ratio(s: str) -> float:
    if not s:
        return 0.0
    return sum(c.isdigit() for c in s) / len(s)


def is_valid_topic_key(key: str) -> bool:
    """一个（已规范化的）候选键能否当主题键。"""
    if not key:
        return False
    if key in STOP_KEYS or key in TOOL_WORDS:
        return False
    if _NUMERICISH.match(key):
        return False
    # 数字开头（'16%股份转让'、'2022-11-21'、'4.1'）或数字占比过高的混合碎片
    # （'张开利62.4'、'2025.10审计负债率106.6'）：数值是**事实内容**
    # （该在 fact.content 里），不是**事儿的名字**。
    if key[0].isdigit():
        return False
    if _digit_ratio(key) > 0.4:
        return False
    if key.isascii():
        # 按**字母数字数**计长（'e.g'/'p.s' 实际只有 2 个字母，是缩写噪声）
        return sum(c.isalnum() for c in key) >= _MIN_ASCII
    return len(key) >= _MIN_CJK


def clean_topic_keys(raw: list[str] | None, *, max_keys: int = 10) -> list[str]:
    """规范化 + 过滤 + 保序去重（上游给的顺序≈重要性排序）。"""
    out: list[str] = []
    seen: set[str] = set()
    for r in raw or []:
        k = normalize_key(r)
        if not is_valid_topic_key(k) or k in seen:
            continue
        seen.add(k)
        out.append(k)
        if len(out) >= max_keys:
            break
    return out


def derive_topic_keys(
    keywords: list[str] | None,
    topic: str = "",
    proposal_titles: list[str] | None = None,
    entities: list[str] | None = None,
    *,
    max_keys: int = 10,
) -> list[str]:
    """fact 侧主题键：优先蒸馏产出的 keywords；缺失（存量台账 / v5 前）时
    从 topic + proposal 标题分词 + entities **确定性派生**（零 LLM，G6 安全）。

    派生优先级：entities（模型点名的实体，最接近名词键）→ topic/标题分词。
    """
    cleaned = clean_topic_keys(keywords, max_keys=max_keys)
    if cleaned:
        return cleaned
    fallback: list[str] = list(entities or [])
    for text in [topic or "", *(proposal_titles or [])]:
        fallback.extend(p for p in _SPLIT.split(text) if p)
    return clean_topic_keys(fallback, max_keys=max_keys)


# ── Matter 侧聚合 ────────────────────────────────────────────────────────────

MAX_MATTER_TOPIC_KEYS = 30


def effective_topic_keys(
    topic_keys: dict[str, int] | None,
    title: str,
    aliases: list[str] | None = None,
    entities: list[str] | None = None,
) -> dict[str, int]:
    """Matter 侧生效主题键：已累积的 topic_keys 优先；**存量 Matter（旧代码建的，
    topic_keys 空）从 title+aliases+entities 确定性派生**（只读时算、不落库）。

    2026-08-14 上线即踩：促成本机制的那 4 个泰安 Matter 全是旧代码建的、
    topic_keys 全空——merge-candidates 对它们返回空（存量盲区）。aliases 里
    本就带创建时的实体种子，派生足以把它们检出来；毒词过滤照常生效。
    """
    if topic_keys:
        return dict(topic_keys)
    derived = derive_topic_keys(
        None, title or "", [title or "", *(aliases or [])], list(entities or []))
    return dict.fromkeys(derived, 1)


def accumulate_topic_keys(
    existing: dict[str, int], new_keys: list[str], *, cap: int = MAX_MATTER_TOPIC_KEYS,
) -> dict[str, int]:
    """成员 fact 的主题键计票进 Matter.topic_keys（返回新 dict，不改入参）。

    计数上限裁剪按 (count 降序, key 字典序) 确定性保留 top-cap——
    与 MS-16 停掉的 aliases 无界累积不同：① 有界；② 消费侧必须多键重叠
    才算命中，单键进表不构成吸尘器。
    """
    merged = dict(existing or {})
    for k in new_keys:
        if k:
            merged[k] = merged.get(k, 0) + 1
    if len(merged) > cap:
        kept = sorted(merged.items(), key=lambda kv: (-kv[1], kv[0]))[:cap]
        merged = dict(kept)
    return merged


# ── 匹配（高精度门槛）─────────────────────────────────────────────────────────

MIN_OVERLAP_STRONG = 3      # ≥3 个主题键相等 → 强命中
MIN_OVERLAP_JACCARD = 2     # 或 ≥2 个且覆盖小集合的
MIN_JACCARD = 0.6           # ≥60%


def topic_key_overlap(fact_keys: list[str] | set[str],
                      matter_keys: dict[str, int] | set[str]) -> tuple[int, float]:
    """返回 (相等键数, 相等数/较小集合)。空集 → (0, 0.0)。

    🔴 **只做精确相等，不做包含**（2026-08-14 当天实测回退）：曾加"长键互相
    包含"想救存量派生键（'泰安仁信' ⊂ '核实泰安仁信入股…'），live 库一跑
    检出 **300 对**——存量 entities 是路径/Redis key/命令名（'bladex' 藏在
    N 个长键里 = N 票），MS-16 的单泛词吸尘器借尸还魂。存量收敛不靠模糊匹配
    （垃圾键上调阈值是无底洞），靠：新数据 keywords 干净精确 + 手动 merge +
    全量重建时新代码从源头拦分裂。
    """
    a = set(fact_keys or ())
    b = set(matter_keys or ())
    if not a or not b:
        return 0, 0.0
    inter = len(a & b)
    return inter, inter / min(len(a), len(b))


def is_strong_topic_match(fact_keys: list[str] | set[str],
                          matter_keys: dict[str, int] | set[str]) -> bool:
    """同事儿判定（高精度）：≥3 键相等，或 ≥2 键且 Jaccard(小集)≥0.6。"""
    inter, jac = topic_key_overlap(fact_keys, matter_keys)
    if inter >= MIN_OVERLAP_STRONG:
        return True
    return inter >= MIN_OVERLAP_JACCARD and jac >= MIN_JACCARD
