"""L4 链接裁决 - LLM 裁决器 + Memory Hub 裁决台账（ADR-0018 §3.2/§3.6/§4.1）。

归属五层的 L4：e5 召回 top-k 候选 Matter 登记卡 -> 便宜档 LLM 裁决
（link:<id> | none | uncertain）。复用蒸馏同一模型（少一个行为面，§3.6）。

consolidator 是后台批处理进程（非热路径），走 Router 网关的同步调用（与 LLMDistiller 一致）。
批量：一次调用裁决同一轮 consolidation 的多条 fact（ADR-0018 §3.2 "可批量"）。
台账缓存（§4.1）：裁决前查 judgment/ 台账命中复用（零 LLM 成本），miss 才调 LLM + 写台账。

降级（§3.7）：LLM 不可用/超时/解析失败 -> 全部 uncertain（留池），绝不回退 cosine。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from bladex_core.attribution import (
    LinkJudgeItem,
    LinkJudgeResult,
    LinkVerdict,
)

from bladex_proxy import router_sdk
from bladex_proxy.storage.memory_hub import MemoryHub

logger = logging.getLogger(__name__)

# 裁决 prompt 版本（台账语义的一部分；prompt 改 -> 递增 -> 视为新裁决，旧台账仍可复用同 fact_id 最新）
LINK_PROMPT_VER = "v1-link-001"

_LINK_SYSTEM = """\
You are a memory attribution judge. For each FACT below, decide which existing MATTER it belongs to.

Output STRICT JSON: an array with one object per FACT, in the same order as given.
Each object: {"fact_id": "<id>", "verdict": "link:<matter_id>" | "none" | "uncertain", "summary": "<rewritten matter summary or empty>"}

Rules:
- verdict "link:<matter_id>": the fact clearly belongs to that candidate matter (use the matter_id exactly as listed). Only link when the fact is genuinely about the same thing/matter as the candidate.
- verdict "none": the fact does NOT belong to any candidate (it is about something else) -> a new matter should be created.
- verdict "uncertain": ambiguous or insufficient info -> leave for later review. When in doubt, prefer "uncertain" over "link" (never merge unrelated things).
- "summary": when verdict is link:<id>, optionally provide a concise updated summary for that matter (synthesizing the new fact). Empty string otherwise.
- Never invent a matter_id not listed in that fact's candidates. If none fits, use "none".
- Output in the same language as the fact content.
"""


class LLMLinkJudge:
    """L4 链接裁决器（实现 LinkJudge，ADR-0018 §3.2/§3.6）。

    用法：
      judge = LLMLinkJudge(model="openai/deepseek-v4-flash", api_base=..., api_key=...)
      results = judge.judge_links([item1, item2])  # 一次调用裁决多条
    """

    def __init__(
        self,
        model: str,
        api_base: str = "",
        api_key: str = "",
        *,
        temperature: float = 0.0,
        max_tokens: int = 768,
        timeout: float = 30.0,
        no_think: bool = True,
    ) -> None:
        self._model = model
        self._api_base = api_base
        self._api_key = api_key
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._no_think = no_think
        self._call_count = 0
        self._fail_count = 0

    @property
    def model_name(self) -> str:
        return self._model

    def judge_links(self, items: list[LinkJudgeItem]) -> list[LinkJudgeResult]:
        """批量链接裁决。失败/超时/解析失败 -> 全部 uncertain（降级，不抛）。"""
        if not items:
            return []

        prompt = _build_link_prompt(items)
        self._call_count += 1
        try:

            kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": _LINK_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                "temperature": self._temperature,
                "max_tokens": self._max_tokens,
                "timeout": self._timeout,
            }
            if self._api_base:
                kwargs["api_base"] = self._api_base
            if self._api_key:
                kwargs["api_key"] = self._api_key
            if self._no_think:
                kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

            response = router_sdk.completion(**kwargs)
            raw_text = response.choices[0].message.content or ""
            return _parse_link_json(raw_text, items)
        except Exception as e:  # noqa: BLE001
            self._fail_count += 1
            err = router_sdk.error_text(e)
            logger.warning("l4_judge_failed model=%s err=%s -> all uncertain",
                           self._model, err)
            return [_uncertain(it.fact_id, f"judge_error:{err}") for it in items]

    def stats(self) -> dict[str, Any]:
        return {
            "calls": self._call_count,
            "fails": self._fail_count,
            "model": self._model,
            "prompt_ver": LINK_PROMPT_VER,
        }


def _build_link_prompt(items: list[LinkJudgeItem]) -> str:
    """构造批量裁决 prompt（每个 fact block + 其候选登记卡）。"""
    lines: list[str] = []
    for i, it in enumerate(items):
        lines.append(f"### FACT {i}")
        lines.append(f"fact_id: {it.fact_id}")
        lines.append(f"content: {it.content[:500]}")
        # T4b（2026-08-10）：轮级 topic 作为证据一行进 fact 描述。
        # 只加证据不改判定语义/_LINK_SYSTEM，LINK_PROMPT_VER 不 bump
        # （judgment 台账按 fact_id/seq 键、不按 prompt_ver，旧裁决照常复用）。
        if getattr(it, "topic", ""):
            lines.append(f"topic: {it.topic[:80]}")
        if it.entities:
            lines.append(f"entities: {', '.join(it.entities[:10])}")
        if it.proposal_titles:
            lines.append(f"proposals: {', '.join(it.proposal_titles[:5])}")
        if it.candidates:
            lines.append("candidates:")
            for c in it.candidates:
                alias_str = ", ".join(c.aliases[:5]) if c.aliases else ""
                parts = [f"  - matter_id: {c.matter_id}", f"    title: {c.title}"]
                if alias_str:
                    parts.append(f"    aliases: {alias_str}")
                if c.summary:
                    parts.append(f"    summary: {c.summary[:200]}")
                if c.recent_members:
                    parts.append(f"    recent_members: {' | '.join(c.recent_members[:3])}")
                lines.extend(parts)
        else:
            lines.append("candidates: (none)")
        lines.append("")
    lines.append("### END")
    lines.append("Return the JSON array now.")
    return "\n".join(lines)


def _parse_link_json(raw_text: str, items: list[LinkJudgeItem]) -> list[LinkJudgeResult]:
    """解析 LLM JSON 输出 -> LinkJudgeResult 列表（与 items 对齐）。"""
    def _all_uncertain(reason: str) -> list[LinkJudgeResult]:
        return [_uncertain(it.fact_id, reason) for it in items]

    if not raw_text or not raw_text.strip():
        return _all_uncertain("empty_output")

    text = raw_text.strip()
    # 去 markdown ```json ... ``` 包裹
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2:
            text = "\n".join(lines[1:-1]) if lines[-1].startswith("```") else "\n".join(lines[1:])

    first = text.find("[")
    last = text.rfind("]")
    if first < 0 or last < 0 or last <= first:
        logger.warning("l4_judge_parse_failed no_array raw_len=%d head=%.80r", len(raw_text), raw_text)
        return _all_uncertain("parse_failed_no_array")

    try:
        data = json.loads(text[first:last + 1])
    except json.JSONDecodeError:
        logger.warning("l4_judge_parse_failed json_err raw_len=%d head=%.80r", len(raw_text), raw_text)
        return _all_uncertain("parse_failed_json")

    if not isinstance(data, list):
        return _all_uncertain("parse_failed_not_list")

    # 按 fact_id 对齐（LLM 可能乱序）
    by_id: dict[str, LinkJudgeResult] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        fid = str(item.get("fact_id", "")).strip()
        if not fid:
            continue
        verdict_raw = str(item.get("verdict", "uncertain")).strip().lower()
        result = _parse_verdict(fid, verdict_raw)
        result.summary_rewrite = str(item.get("summary", "") or "").strip()
        by_id[fid] = result

    # 与 items 对齐：缺的填 uncertain
    return [by_id.get(it.fact_id, _uncertain(it.fact_id, "missing_in_output")) for it in items]


def _parse_verdict(fact_id: str, verdict_raw: str) -> LinkJudgeResult:
    """解析 verdict 字符串 -> LinkJudgeResult。"""
    if verdict_raw.startswith("link:"):
        matter_id = verdict_raw[len("link:"):].strip()
        return LinkJudgeResult(
            fact_id=fact_id, verdict=LinkVerdict.LINK,
            matter_id=matter_id, reason="link",
        )
    if verdict_raw == "none":
        return LinkJudgeResult(fact_id=fact_id, verdict=LinkVerdict.NONE, reason="none")
    # uncertain 或无法识别 -> uncertain
    return LinkJudgeResult(
        fact_id=fact_id, verdict=LinkVerdict.UNCERTAIN,
        reason=verdict_raw or "uncertain",
    )


def _uncertain(fact_id: str, reason: str) -> LinkJudgeResult:
    return LinkJudgeResult(fact_id=fact_id, verdict=LinkVerdict.UNCERTAIN, reason=reason)


# ── Memory Hub 裁决台账（实现 JudgmentJournalProtocol，ADR-0018 §4.1）──


class HubJudgmentJournal:
    """裁决台账的 MemoryHub 适配（实现 JudgmentJournalProtocol）。

    读用传入的 Memory Hub（consolidator 通常 read_only）；写时由调用方（AttributionPipeline）
    try/except 降级--read_only Memory Hub 写会抛（增量 consolidation 场景），
    T11 full rebuild（consolidator 读写 Memory Hub）时写生效。
    """

    def __init__(self, journal: MemoryHub, *, judge_model: str = "") -> None:
        self._journal = journal
        self._judge_model = judge_model

    def get_latest_judgment(self, fact_id: str) -> LinkJudgeResult | None:
        """读 fact_id 下最新（最大 seq）的**L4 链接**裁决台账。无 -> None。

        非 L4 记录（M2 的 `consolidation:*` 等）跳过——命名空间共用，
        种类不共用（见 `_is_link_verdict`）。
        """
        latest_key: str | None = None
        latest_rec = None
        for key, rec in self._journal.scan_judgments(fact_id):
            if not _is_link_verdict(getattr(rec, "verdict", "")):
                continue
            # key = judgment/{fact_id}/{seq}，seq 单调递增 -> 字典序后者更新
            if latest_key is None or key > latest_key:
                latest_key = key
                latest_rec = rec
        if latest_rec is None:
            return None
        return _record_to_result(latest_rec)

    def put_judgment(
        self, fact_id: str, candidates: list[dict], result: LinkJudgeResult,
    ) -> None:
        """追加一条裁决台账。read_only Memory Hub 写会抛（调用方降级）。"""
        verdict_str = _result_to_verdict_str(result)
        self._journal.append_judgment(
            fact_id, candidates, verdict_str, self._judge_model,
            summary_rewrite=result.summary_rewrite, reason=result.reason,
        )


class IndexJudgmentJournal:
    """裁决台账的 MemoryIndex 适配（ADR-0020 T1.3，实现 JudgmentJournalProtocol）。

    从 Memory Hub 迁 Memory Index meta：consolidator 独占 Memory Index 写锁，台账写入无障碍（Memory Hub 旧实现因
    consolidator read_only Memory Hub 而写失败 100%，ADR-0020 根因）。
    """

    def __init__(self, index: Any, *, judge_model: str = "") -> None:
        self._index = index
        self._judge_model = judge_model

    def get_latest_judgment(self, fact_id: str) -> LinkJudgeResult | None:
        """读 fact_id 下最新（最大 seq）的**L4 链接**裁决台账。无 -> None。

        非 L4 记录（M2 的 `consolidation:*` 等）跳过——命名空间共用，
        种类不共用（见 `_is_link_verdict`；08-10 实测 796 条伪 uncertain
        的根因就在这里）。
        """
        latest_key: str | None = None
        latest_rec = None
        for key, rec in self._index.scan_judgments(fact_id):
            if not _is_link_verdict(getattr(rec, "verdict", "")):
                continue
            # key = judgment/{fact_id}/{seq}，seq 单调递增 -> 字典序后者更新
            if latest_key is None or key > latest_key:
                latest_key = key
                latest_rec = rec
        if latest_rec is None:
            return None
        return _record_to_result(latest_rec)

    def put_judgment(
        self, fact_id: str, candidates: list[dict], result: LinkJudgeResult,
    ) -> None:
        """追加一条裁决台账到 Memory Index meta（consolidator 独占写，不抛）。"""
        verdict_str = _result_to_verdict_str(result)
        self._index.append_judgment(
            fact_id, candidates, verdict_str, self._judge_model,
            summary_rewrite=result.summary_rewrite, reason=result.reason,
        )


def _is_link_verdict(verdict: str) -> bool:
    """这条台账记录是不是一条 **L4 链接裁决**（2026-08-11 回归修复）。

    🔴 背景：`judgment/{fact_id}/{seq}` 命名空间自 M2-2（2026-08-08 写入时裁决）
    起被**两个写入方共用**——L4 写 `link:<id>|none|uncertain`，M2 写
    `consolidation:add|update|noop`。而 `get_latest_judgment` 只取"最新 seq"、
    不区分种类，于是几乎每条 fact 在 L4 阶段都会命中**自己的 M2 记录**，
    `_record_to_result` 认不出那个 verdict 格式 → 降级 UNCERTAIN → 留池 +
    pending、**根本不送 judge**。

    实测（08-10 全量重建）：907 条落到 L4 的 fact 里 796 条走了这条伪 uncertain，
    未归属池 43%→77%，L4 真实参与 111 条。bug 自 08-08 存在，此前被 L2/L3
    吸尘器掩盖（走到 L4 的只有 26 条）；MS-16 收紧 L2/L3 后全面暴露。

    判据用**白名单**而不是"排除 consolidation:"：未来任何新写入方只要不用
    L4 的三种 verdict 形态，就自动不污染 L4 读侧（黑名单会漏下一个写入方）。
    零 schema 迁移——verdict 格式本就是 `_result_to_verdict_str` 定义的契约。
    """
    v = (verdict or "").strip().lower()
    return v.startswith("link:") or v in ("none", "uncertain")


def _record_to_result(rec: Any) -> LinkJudgeResult:
    """JudgmentRecord -> LinkJudgeResult（解析 verdict 字符串）。"""
    verdict_str = (rec.verdict or "").strip().lower()
    if verdict_str.startswith("link:"):
        matter_id = verdict_str[len("link:"):].strip()
        return LinkJudgeResult(
            fact_id=rec.fact_id, verdict=LinkVerdict.LINK,
            matter_id=matter_id, summary_rewrite=rec.summary_rewrite or "",
            reason=rec.reason or "ledger",
        )
    if verdict_str == "none":
        return LinkJudgeResult(
            fact_id=rec.fact_id, verdict=LinkVerdict.NONE,
            summary_rewrite=rec.summary_rewrite or "", reason=rec.reason or "ledger",
        )
    return LinkJudgeResult(
        fact_id=rec.fact_id, verdict=LinkVerdict.UNCERTAIN,
        summary_rewrite=rec.summary_rewrite or "", reason=rec.reason or "ledger",
    )


def _result_to_verdict_str(result: LinkJudgeResult) -> str:
    """LinkJudgeResult -> verdict 字符串（台账存储格式）。"""
    if result.verdict == LinkVerdict.LINK:
        return f"link:{result.matter_id}" if result.matter_id else "uncertain"
    if result.verdict == LinkVerdict.NONE:
        return "none"
    return "uncertain"
