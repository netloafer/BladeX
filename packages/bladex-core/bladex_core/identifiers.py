"""标识符抽取 —— 检索与蒸馏共用的**单一事实源**（ADR-0028 E6.1 / E6.5）。

## 为什么要单独一个模块

两处都需要"从一段文本里认出那些不能被语义近似的 token"：

  - **检索侧（E6.1/E6.2）**：`bxe3663a38` 与 `bx3a984bcd` 的 cosine 实测 **0.9890**
    ——dense 通道对标识符是瞎的，只能靠词法通道精确匹配。
  - **写入侧（E6.5）**：蒸馏后要校验"原文里的标识符有没有出现在 fact 里"，
    丢了就重蒸一次。

两边必须用**同一个函数**：一边认得出、另一边认不出，校验就成了摆设。

## 规则族（确定性、零 LLM）

    hex 串        \\b[a-f0-9]{6,}\\b            content hash / commit / 探针 token
    前缀 id       \\b[a-z]{2,4}[0-9a-f]{6,}\\b   fact_270d485c1a78 / m-06fef11a7845
    版本号        v?\\d+\\.\\d+[.\\w]*            0.1.0 / v4 / glm-5.2
    带扩展名路径尾段                              memory_index.py / routing.toml
    UPPER_SNAKE   [A-Z][A-Z0-9_]{3,}            BLADEX_MIN_IMPORTANCE / ROUTE_ENABLED
    反引号内容    `…`                            人手动标出来的那些

保序 + 去重（同一 token 只出一次），供词法查询与蒸馏校验直接使用。
"""

from __future__ import annotations

import re

# 顺序即优先级：先长后短，避免 `fact_270d485c1a78` 被 hex 规则先切成碎片。
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # 反引号内容（人手动标注的最高信号）
    ("backtick", re.compile(r"`([^`\n]{2,64})`")),
    # 带扩展名的路径尾段（.py/.toml/.md…）——取整段，供词法精确匹配
    ("path", re.compile(r"\b[\w.\-/]*[\w\-]\.[A-Za-z0-9]{1,8}\b")),
    # 前缀 id：字母前缀 + hex 尾（fact_xxx / m-xxx / bxe3663a38）
    ("prefixed_id", re.compile(r"\b[A-Za-z][\w\-]{0,12}[_\-]?[0-9a-f]{6,}\b")),
    # 纯 hex 串
    ("hex", re.compile(r"\b[a-f0-9]{6,}\b")),
    # 版本号
    ("version", re.compile(r"\bv?\d+\.\d+[.\w\-]*\b")),
    # UPPER_SNAKE 常量 / 错误码
    ("upper_snake", re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b")),
)

# 明显是普通词的短 hex（"decade"、"faced"…）会被 hex 规则误收；
# 要求至少含一个数字，把纯字母的英文单词挡掉。
_HEX_NEEDS_DIGIT = re.compile(r"\d")

_MAX_IDENTIFIERS = 32

# ── CJK 粘连修复（V-I2）：`\w` 在 Python 里含 CJK，"已下载qwen3.8" 会被 path/
# prefixed_id 规则整段吞下（实测 18.6% 的抽取带 CJK 粘连），词法通道随之
# 精确匹配不上。在 CJK↔ASCII 过渡处插空格再匹配——纯预处理，规则族不动。
_CJK_RANGE = "぀-ヿ㐀-䶿一-鿿가-힯"
_CJK_BOUNDARY = re.compile(
    f"(?<=[{_CJK_RANGE}])(?=[0-9A-Za-z])|(?<=[0-9A-Za-z])(?=[{_CJK_RANGE}])")


def split_cjk_boundary(text: str) -> str:
    """CJK 与 ASCII 字母数字的过渡处插入空格（幂等；无 CJK 时恒等）。"""
    return _CJK_BOUNDARY.sub(" ", text) if text else text


def extract_identifiers(text: str, *, max_items: int = _MAX_IDENTIFIERS) -> list[str]:
    """抽取文本里的标识符（保序去重，封顶 max_items）。

    纯函数、确定性、零 LLM —— 与 envelope/task_unit 同性质，保重建等价性。
    🔴 V-I2 起先做 CJK 边界切分——改的是"粘连修掉"，抽取规则族本身不动；
    历史台账/索引里已存的粘连 token 在下次重建/重嵌时自然收敛。
    """
    if not text:
        return []
    text = split_cjk_boundary(text)
    out: list[str] = []
    seen: set[str] = set()
    for kind, pat in _PATTERNS:
        for m in pat.finditer(text):
            tok = (m.group(1) if m.lastindex else m.group(0)).strip()
            if not tok or len(tok) < 3:
                continue
            if kind == "hex" and not _HEX_NEEDS_DIGIT.search(tok):
                continue        # 纯字母的"hex"多半只是个英文单词
            low = tok.lower()
            if low in seen:
                continue
            # 已被更长的标识符包含（`fact_270d485c1a78` vs `270d485c1a78`）→ 跳过
            if any(low in s and low != s for s in seen):
                continue
            seen.add(low)
            out.append(tok)
            if len(out) >= max_items:
                return out
    return out


def missing_identifiers(source_text: str, produced_texts: list[str]) -> list[str]:
    """E6.5 蒸馏保真：原文里有、产出里没有的标识符。

    大小写不敏感比较（蒸馏模型常改大小写，但 token 本身没丢就不算丢）。
    """
    src = extract_identifiers(source_text)
    if not src:
        return []
    blob = " ".join(produced_texts).lower()
    return [tok for tok in src if tok.lower() not in blob]
