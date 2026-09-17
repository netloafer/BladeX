"""写入时裁决的 LLM 实现（M2-1 / 复核第 2 层）。

模型 = **与蒸馏同一个**（拍板 4，DPL 先例）：少一个行为面、少一处配置、
少一次"到底是哪个模型判的"的排查。可手动改配置覆盖。

consolidator 是后台批处理进程（非热路径），走 Router 网关的同步调用
（与 `LLMDistiller` / `LLMLinkJudge` 一致）。

降级（宁多勿丢）：超时/失败/解析失败 → **全部 ADD**，并置 `fallback=True` 供计数。
理由与 L4 裁决相反：L4 判不出就 uncertain（留池，不归属），代价是"这条暂时没归属"；
这里判不出就 ADD，代价是"库里多一条重复" —— 而反过来（判不出就丢弃）
会**静默丢记忆**，那是不可接受的一侧。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from bladex_core.adjudication import (
    AdjudicationInput,
    AdjudicationOp,
    AdjudicationVerdict,
)
from bladex_core.flags import flag_number

from bladex_proxy import router_sdk

logger = logging.getLogger(__name__)

#: 裁决 prompt 版本（进 judgment 台账；prompt 改 → 递增 → 视为新裁决）
#:
#: v2（MS-11）：加入时序规则 + `observed_at` 字段。
#: **现在 bump 是零成本的** —— 裁决台账尚无存量（M2 落地至今未跑过真实重建）。
#: M5 全量重建之后再 bump 就要重烧全部裁决，所以这一次要趁早。
ADJ_VER = "v2-adj-001"

_ADJ_SYSTEM = """\
You are a memory write-time adjudicator. For each NEW item below you are given its
nearest existing NEIGHBORS from the memory library. Decide what should happen.

Output STRICT JSON: an array with one object per NEW item, in the same order.
Each object: {"id": "<id>", "op": "add" | "update" | "noop",
              "targets": ["<neighbor fact_id>", ...], "reason": "<short>"}

Each item carries `observed_at` — when it was recorded. Items are NOT necessarily
given to you in chronological order: this may be a replay of historical data.

The three operations:
- "add":    the NEW item states something the neighbors do not cover. Keep both.
            `targets` must be empty.
- "update": the NEW item CORRECTS or SUPERSEDES one or more neighbors — same subject,
            a different value that replaces theirs. List those neighbors in `targets`.
            The old ones are retained but marked no-longer-current, so be precise:
            only list neighbors the NEW item actually contradicts or replaces.
- "noop":   the NEW item restates something a neighbor already says (same meaning,
            different wording). Put that neighbor in `targets`. The NEW item is dropped.

🔴 TIME RULE (read this before choosing "update"):
- Compare `observed_at`. A memory can only be superseded by something recorded LATER.
- If the NEW item is OLDER than a neighbor it disagrees with, do NOT "update" that
  neighbor — the neighbor already reflects the newer state. Choose "add" (both are
  kept; the temporal ordering is preserved) or "noop" if they mean the same.
- The single exception: an explicit CORRECTION NOTE on the NEW item. A statement like
  "the earlier belief was wrong" is evidence about content, not about recency, so it
  justifies "update" regardless of timestamps.
- When `observed_at` is missing or equal on both sides, fall back to judging content
  alone and prefer "add" over "update".

Rules:
- 🔴 When the NEW item carries a CORRECTION note (an explicit statement that an earlier
  belief was wrong), trust it: that is direct evidence for "update", and the neighbors
  it contradicts belong in `targets`.
- Neighbors that are merely RELATED (same topic, compatible facts) are NOT targets.
  Compatible coexistence is "add", not "update".
- Do not invent fact_ids. Every id in `targets` must appear in that item's neighbor list.
- When genuinely unsure between add and update, choose "add" — keeping an extra item
  is recoverable, marking a still-true memory as superseded is not.
- Judge meaning, not wording overlap. High textual similarity with a different value
  (different date, number, name, status) is "update", not "noop".
"""


class LLMAdjudicator:
    """写入时裁决器（实现 `AdjudicatorProtocol`）。

    用法：
      adj = LLMAdjudicator(model="openai/deepseek-v4-flash", api_base=..., api_key=...)
      verdicts = adj.adjudicate([pkg1, pkg2])   # 一次调用裁决多条
    """

    def __init__(
        self,
        model: str,
        api_base: str = "",
        api_key: str = "",
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
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
        # 🔴 MQ-S32 病 B（2026-08-31）：**按候选条数**记降级面。
        # `_fail_count` 只数抛异常的调用，而 `_parse` 的两条降级路径
        # （`empty_output` / `parse_failed_no_array`）**不抛异常也不计数**——
        # 于是"整块降级成 ADD"在 `stats()` 里看不见。这正是本文件 `adjudicate`
        # docstring 自己写过的那句"失败与正常结果长得一模一样"，
        # 只不过上次修的是爆炸半径，没修可见性。
        # 判据取 `fallback=True`（`_all_add` 的唯一标记），**与降级来源无关**——
        # 新增第四条降级路径也会自动被数到（fail-closed 的计数口径）。
        self._item_count = 0
        self._degraded_items = 0

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def prompt_ver(self) -> str:
        return ADJ_VER

    def adjudicate(
        self, items: list[AdjudicationInput],
    ) -> list[AdjudicationVerdict]:
        """分块裁决 —— **单块失败只损失该块**。

        2026-08-08 全量重建的教训：此前这里把全部 pending 打包成**一次**调用。
        805 条候选 → 返回 JSON 被输出上限截断 → `_parse` 判失败 → 全量 fallback ADD。
        日志上只留下一行 `adjudicate_parse_failed`，而 `ops={'add': 805}` 看起来
        像"裁决跑了、判定就是全 ADD"——**失败与正常结果长得一模一样**，
        这才是它能瞒过一整轮 5 小时重建的原因。

        分块不是性能优化，是把爆炸半径从"整个库"压到"一块"。
        """
        if not items:
            return []
        chunk = max(1, int(flag_number("BLADEX_ADJUDICATE_CHUNK")))
        if len(items) <= chunk:
            return self._adjudicate_chunk(items)
        out: list[AdjudicationVerdict] = []
        for i in range(0, len(items), chunk):
            out.extend(self._adjudicate_chunk(items[i:i + chunk]))
        logger.info("adjudicate_chunked total=%d chunk=%d calls=%d",
                    len(items), chunk, (len(items) + chunk - 1) // chunk)
        return out

    def _adjudicate_chunk(
        self, items: list[AdjudicationInput],
    ) -> list[AdjudicationVerdict]:
        out = self._adjudicate_chunk_inner(items)
        # 单一记账点：三条降级路径（异常 / empty_output / parse_failed）都要经过它。
        self._item_count += len(items)
        self._degraded_items += sum(1 for v in out if getattr(v, "fallback", False))
        return out

    def _adjudicate_chunk_inner(
        self, items: list[AdjudicationInput],
    ) -> list[AdjudicationVerdict]:
        self._call_count += 1
        try:
            kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": _ADJ_SYSTEM},
                    {"role": "user", "content": _build_prompt(items)},
                ],
                "temperature": self._temperature,
                # 🔴 输出预算必须**随条数走**（2026-08-09 立）。
                #   固定 1024 是这个 bug 的真根因：一条判决 ~132 字符 ≈ 40 token，
                #   25 条就要 ~1000 token —— 正好卡死在上限上，34 块全被截断。
                #   分块只解决了"一次失败拖垮全部"，没解决"每次都失败"。
                #   配置值当**下限**，不是上限：用户调高有效，调低不至于自伤。
                "max_tokens": max(self._max_tokens, len(items) * 160 + 256),
                "timeout": self._timeout,
            }
            if self._api_base:
                kwargs["api_base"] = self._api_base
            if self._api_key:
                kwargs["api_key"] = self._api_key
            if self._no_think:
                kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

            resp = router_sdk.completion(**kwargs)
            raw = resp.choices[0].message.content or ""
            # 🔴 截断 ≠ 坏 JSON。此前两者都打 `json_err`，于是 2026-08-08 复盘时
            #   被读成"模型输出质量问题"，实际是**我们把它剪断的**。
            #   不同的失败必须长得不一样，否则等于没有信号。
            fin = ""
            try:
                fin = str(getattr(resp.choices[0], "finish_reason", "") or "")
            except Exception:  # noqa: BLE001 —— 诊断字段，拿不到不影响主体
                pass
            if fin == "length":
                logger.warning("adjudicate_output_truncated items=%d raw_len=%d "
                               "budget=%d -> 尝试打捞完整对象",
                               len(items), len(raw), max(self._max_tokens, len(items) * 160 + 256))
            return _parse(raw, items)
        except Exception as e:  # noqa: BLE001 —— 降级不抛（宁多勿丢）
            self._fail_count += 1
            logger.warning("adjudicate_failed model=%s err=%s -> all add",
                           self._model, router_sdk.error_text(e))
            return _all_add(items, f"adjudicator_error:{router_sdk.error_text(e)}")

    def stats(self) -> dict[str, Any]:
        return {"calls": self._call_count, "fails": self._fail_count,
                "items": self._item_count, "degraded_items": self._degraded_items,
                "model": self._model, "prompt_ver": ADJ_VER}


def _build_prompt(items: list[AdjudicationInput]) -> str:
    lines: list[str] = []
    for it in items:
        lines.append(f"### NEW {it.candidate_id}")
        lines.append(f"content: {it.content[:500]}")
        # MS-11：时间必须出现在 prompt 里。字段加了但不渲染 = 又一次"机制写完接线断"，
        # 而且这一次的症状是"裁决看起来在工作、方向却可能是反的"。
        lines.append(f"observed_at: {it.observed_at or 'unknown'}")
        if it.subject or it.attribute:
            lines.append(f"subject/attribute: {it.subject} / {it.attribute}")
        if it.entities:
            lines.append(f"entities: {', '.join(it.entities[:8])}")
        if it.corrects_hint:
            # 单独一行且措辞醒目：这是 F4 说的"最高精度信号"，
            # 混在别的字段里模型会当成普通元数据略过。
            lines.append(f"CORRECTION NOTE: {it.corrects_hint[:200]}")
        if it.neighbors:
            lines.append("neighbors:")
            for n in it.neighbors:
                extra = f" [{n.subject}/{n.attribute}]" if (n.subject or n.attribute) else ""
                lines.append(
                    f"  - {n.fact_id} (sim={n.similarity:.3f}, "
                    f"observed_at={n.observed_at or 'unknown'}){extra}: {n.content[:300]}")
        else:
            lines.append("neighbors: (none)")
        lines.append("")
    lines.append("### END")
    lines.append("Return the JSON array now.")
    return "\n".join(lines)


def _all_add(items: list[AdjudicationInput], reason: str) -> list[AdjudicationVerdict]:
    return [AdjudicationVerdict(candidate_id=it.candidate_id,
                                op=AdjudicationOp.ADD, reason=reason, fallback=True)
            for it in items]


def _salvage_objects(text: str) -> list[dict[str, Any]]:
    """从可能被截断的 JSON 数组文本里，逐个抠出**完整**的对象。

    `json.JSONDecoder.raw_decode` 会在遇到第一个不完整对象时抛错，
    此前的完整对象已经拿到手 —— 这正是我们要的。
    """
    dec = json.JSONDecoder()
    out: list[dict[str, Any]] = []
    i = 0
    n = len(text)
    while i < n:
        while i < n and text[i] in " \t\r\n,[":
            i += 1
        if i >= n or text[i] != "{":
            break
        try:
            obj, end = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            break                      # 截断处：到此为止
        if isinstance(obj, dict):
            out.append(obj)
        i = end
    return out


def _parse(raw: str, items: list[AdjudicationInput]) -> list[AdjudicationVerdict]:
    """解析 JSON 数组 → verdicts（与 items 对齐）。任何解析问题 → 全部降级 ADD。"""
    if not raw or not raw.strip():
        return _all_add(items, "empty_output")
    text = raw.strip()
    if text.startswith("```"):
        rows = text.splitlines()
        if len(rows) >= 2:
            text = "\n".join(rows[1:-1]) if rows[-1].startswith("```") else "\n".join(rows[1:])
    first = text.find("[")
    if first < 0:
        logger.warning("adjudicate_parse_failed no_array raw_len=%d head=%.80r",
                       len(raw), raw)
        return _all_add(items, "parse_failed_no_array")

    # 🔴 截断有**两条**失败路径，取决于内容里碰巧有没有 `]`（2026-08-09 立）：
    #   - 判决带 `"targets": []` -> rfind 找得到右括号 -> 切片后 json.loads 失败 = json_err
    #   - 判决不带 targets       -> 整段没有 `]`       -> 早关卡直接返回     = no_array
    #   线上 34 次全是前者，我的初版剧本造的是后者，于是打捞挂在 json_err 分支里
    #   **对线上有效、对剧本无效**——反过来也一样。两条都得能走到打捞。
    last = text.rfind("]")
    data: Any = None
    if last > first:
        try:
            data = json.loads(text[first:last + 1])
        except json.JSONDecodeError:
            data = None
    if data is None:
        # 打捞：逐个 decode 完整对象，截断处停下。
        #
        # 此前是**全或无**：数组不完整 -> 整块判失败 -> 25 条全降级 ADD。
        # 2026-08-09 那轮 34 块全被截断，可日志里明明能看到每块前 20 多条判决
        # 都是完整的 —— 它们被连坐丢掉了。截断只该损失最后那半条。
        data = _salvage_objects(text[first:])
        if data:
            logger.warning("adjudicate_parse_salvaged raw_len=%d recovered=%d/%d",
                           len(raw), len(data), len(items))
        else:
            logger.warning("adjudicate_parse_failed json_err raw_len=%d tail=%.60r",
                           len(raw), raw[-60:])
            return _all_add(items, "parse_failed_json")
    if not isinstance(data, list):
        return _all_add(items, "parse_failed_not_list")

    # 每条 item 的合法 target 集合——模型不许指认不在它邻居里的 id。
    allowed = {it.candidate_id: {n.fact_id for n in it.neighbors} for it in items}
    by_id: dict[str, AdjudicationVerdict] = {}
    for row in data:
        if not isinstance(row, dict):
            continue
        cid = str(row.get("id", "")).strip()
        if cid not in allowed:
            continue
        op_raw = str(row.get("op", "add")).strip().lower()
        try:
            op = AdjudicationOp(op_raw)
        except ValueError:
            op = AdjudicationOp.ADD
        targets = [str(t).strip() for t in (row.get("targets") or [])
                   if str(t).strip() in allowed[cid]]
        # op 说要动别人却没给出合法目标 → 退回 ADD。
        # 不这么做的话，一个幻觉 id 会让"取代"静默变成"什么都没发生"，
        # 而 verdict 记录里却写着 update —— 台账与实际行为不一致，最难查。
        if op is not AdjudicationOp.ADD and not targets:
            op, targets = AdjudicationOp.ADD, []
        by_id[cid] = AdjudicationVerdict(
            candidate_id=cid, op=op, target_fact_ids=targets,
            reason=str(row.get("reason", "") or "")[:200],
        )
    return [
        by_id.get(it.candidate_id,
                  AdjudicationVerdict(candidate_id=it.candidate_id,
                                      op=AdjudicationOp.ADD,
                                      reason="missing_in_output", fallback=True))
        for it in items
    ]
