"""蒸馏保真（ADR-0028 E6.5）—— 写入侧配套，检索侧的另一半。

E6.1–E6.4 把检索修好了，但**库里存的东西本身丢了信息**时，检索再准也捞不回来。
真实数据里三种丢法：

  1. **标识符被改写掉**：原文 `bxe3663a38`，蒸出来变成"用户的探针代码"——
     token 没了，词法通道（E6.2）失去唯一能分开近似 token 的抓手；
  2. **语言随机漂移**：同一轮的 user 通道蒸出中文、conclusion 通道蒸出英文，
     实测 ZH/EN 同义对 cosine **0.8639** < novelty 阈值 0.95 → 同一事实存两份；
  3. **novelty 误杀**：新旧探针 token 的 cosine **0.9890**，判重把新的那条吃掉。

三条处置（全部确定性，只有第 1 条在缺失时才多花一次 LLM）：

    ids_src = 标识符抽取(source_text)                     # 与 E6.1 同一函数
    ids_out = ∪ 每条 fact 的 标识符抽取(content) ∪ entities
    missing = ids_src − ids_out
    if missing: 重蒸一次（system prompt 追加"这些标识符必须逐字出现"）
    仍缺失 → 并入语义最近 fact 的 entities（确定性兜底）+ 告警

    语言钉死: CJK 占比>30% → zh 否则 en；产出主语言不符 → 重蒸一次；
             仍不符 → 接受 + tags 加 lang:drift（供 E6.3 ③ 折叠）

    novelty 前置放行: ids(候选) ⊄ ids(高相似已存条) → 直接判新颖
"""

from __future__ import annotations

import logging
import re

from bladex_core.identifiers import extract_identifiers, missing_identifiers

logger = logging.getLogger(__name__)

# ── M0-4（复核 E2 🔴）：前置放行的"可放行标识符"收窄 ───────────────────────
#
# E6.5 是为探针 token（`bxe3663a38` vs `bx3a984bcd`，cosine 0.9890）设计的，
# 但 `extract_identifiers` 的 version/path 规则会把**日期、金额、编号**一并抓出来
# ——而 assertion 类事实几乎条条带数字。于是同一事件的不同措辞变体各自提到不同的
# 日期子集 → 候选永远带着"旧条没有的标识符" → 永远放行 → **对带数字的事实，
# novelty 判重事实上已停用**（实锤：泰山招募 cluster 0.9738/0.9769/0.9694… 十余对
# 高相似事实互相漏网，每对的差异恰是 1月30日 vs 2月26日 vs 3月5日）。
#
# 收窄口径：**只有"混合字母数字的稀有 token"才配放行**——
#   ① 不含 CJK / 空白（`12.5亿`、`3.5亿元`、`1月30日`、`招募公告v2.0` 出局）；
#   ② 至少 2 个 ASCII 字母（`2026.02.26`、`3.1`、`0.1.0`、`12345` 出局；
#      `glm-5.2`、`bx3a984bcd`、`memory_index.py`、`BLADEX_MIN_IMPORTANCE` 留下）。
#
# 明确不做（本卡边界）：不动 0.95 novelty 阈值本体——那是 M2 写入时裁决落地时
# 统一降到 0.98 粗筛的事。这里只把被焊开的门重新关上。
_DISCRIMINATIVE_SHAPE = re.compile(r"^[A-Za-z0-9._\-/:@]+$")
_ASCII_LETTER = re.compile(r"[A-Za-z]")


def is_discriminative_identifier(token: str) -> bool:
    """该 token 是否"稀有到足以证明这是另一件事"（可用于 novelty 前置放行）。

    纯日期 / 纯金额 / 纯短数字 / 带 CJK 的混排一律判否——它们在近重复变体之间
    天然不同，用它们放行等于把判重门焊开。
    """
    tok = (token or "").strip()
    if len(tok) < 3:
        return False
    if not _DISCRIMINATIVE_SHAPE.match(tok):
        return False        # 含 CJK / 空白 / 其它符号 → 不是标识符形态
    return len(_ASCII_LETTER.findall(tok)) >= 2


def discriminative_identifiers(text: str) -> list[str]:
    """抽取文本里**可用于放行**的标识符（保序，已按 M0-4 口径收窄）。"""
    return [t for t in extract_identifiers(text) if is_discriminative_identifier(t)]

# 语言判定门槛：CJK 字符占比 > 30% 判 zh
_CJK_RATIO_ZH = 0.30

RETRY_HINT_IDENTIFIERS = (
    "The following identifiers MUST appear verbatim in facts or entities: {missing}"
)
RETRY_HINT_LANGUAGE = "Respond strictly in {lang}."

# 语言漂移标记（进 Fact.tags，供 E6.3 ③ 同义折叠与审计）
TAG_LANG_DRIFT = "lang:drift"


def detect_lang(text: str) -> str:
    """确定性语言判定：CJK 字符占比 > 30% → zh，否则 en。"""
    if not text:
        return "en"
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿")
    return "zh" if cjk / len(text) > _CJK_RATIO_ZH else "en"


def fact_texts(facts: list) -> list[str]:
    """一批蒸馏产出的可比较文本（content + entities）。"""
    out: list[str] = []
    for f in facts:
        out.append(getattr(f, "content", "") or "")
        out.extend(str(e) for e in (getattr(f, "entities", None) or []))
    return out


def check_identifiers(source_text: str, facts: list) -> list[str]:
    """原文里有、产出里没有的标识符（E6.5 第一条）。"""
    return missing_identifiers(source_text, fact_texts(facts))


def check_language(source_text: str, facts: list,
                   *, want_lang: str = "") -> tuple[str, str]:
    """返回 (期望语言, 实际主语言)。产出为空时实际语言 = 期望（不判漂移）。

    M1-4：`want_lang` 非空时期望语言取它（**配置量**），否则回落"跟随输入"（v3 行为）。

    跟随输入是 X2 的病灶本身：同一轮的 user 通道与 conclusion 通道输入语言可能不同
    （用户中文提问、模型英文作答），于是同一事实蒸出中英两份，
    ZH/EN 同义对 cosine 实测 0.8639 < 判重阈值 0.95 → 两份都入库，
    而中文 query 只捞得回其中一份。语言不该是输入的函数。
    """
    want = want_lang or detect_lang(source_text)
    blob = " ".join(getattr(f, "content", "") or "" for f in facts)
    return want, (detect_lang(blob) if blob.strip() else want)


def attach_missing_identifiers(facts: list, missing: list[str]) -> int:
    """确定性兜底：把仍然缺失的标识符并入**语义最近**那条 fact 的 entities。

    "语义最近"用确定性代理：与原标识符共享最长前缀的 content 那条；
    都不沾边就给第一条。返回实际并入的标识符数。

    为什么要兜底而不是丢弃：标识符是词法通道唯一能分开近似 token 的抓手
    （`bxe3663a38` vs `bx3a984bcd` 的 cosine 是 0.9890）。宁可挂在 entities 上
    略显生硬，也不能让它从库里消失。
    """
    if not facts or not missing:
        return 0
    added = 0
    for tok in missing:
        low = tok.lower()
        target = next(
            (f for f in facts if low[:4] and low[:4] in (getattr(f, "content", "") or "").lower()),
            facts[0],
        )
        ents = list(getattr(target, "entities", None) or [])
        if not any(str(e).lower() == low for e in ents):
            ents.append(tok)
            target.entities = ents
            added += 1
    if added:
        logger.warning("distill_identifier_lost missing=%s recovered_into_entities=%d",
                       missing, added)
    return added


def mark_language_drift(facts: list) -> None:
    """给整批产出打 `lang:drift` 标记（重蒸后仍不符时用）。"""
    for f in facts:
        tags = getattr(f, "tags", "") or ""
        if TAG_LANG_DRIFT not in tags:
            f.tags = f"{tags},{TAG_LANG_DRIFT}".strip(",")


def identifier_novelty_override(candidate_text: str, existing_texts: list[str]) -> bool:
    """novelty 前置放行（E6.5 第三条）：

        候选的标识符集合 ⊄ 已存条目的标识符集合 → **直接判新颖**。

    为什么不能只靠 cosine + entities：新旧探针 token 的 cosine 实测 0.9890，
    而 entities 是 LLM 给的——它心情好才带上 token。标识符抽取是确定性的，
    比"指望模型这次记得写"可靠。

    返回 True = 应判新颖（跳过 cosine 判重）。

    M0-4：两侧都只看**可放行标识符**（`is_discriminative_identifier`）。
    候选一个稀有 token 都没有 → 直接不放行，退回 cosine 判定。
    """
    cand = {t.lower() for t in discriminative_identifiers(candidate_text)}
    if not cand:
        return False
    seen: set[str] = set()
    for t in existing_texts:
        seen |= {x.lower() for x in discriminative_identifiers(t)}
    novel = cand - seen
    if novel:
        logger.debug("novelty_identifier_override tokens=%s", sorted(novel))
        return True
    return False
