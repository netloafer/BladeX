"""批 BB · 2b：矛盾检测的接线（consolidator 侧）——上游适配器、Hub 出口、pass 尾部入口。

`bladex_core.consistency` 是纯函数 + 协议；这里把它们接到真实的东西上：

| 协议 | 实现 |
|---|---|
| `JudgeFn` | `RouterJudge`：`router_sdk.completion(purpose="consistency_judge")` |
| `ContradictionSink` | `ProxySink`：`POST /admin/contradictions` 走 proxy 正门（consolidator 对 Hub 只读，ADR-0032 §2.3） |
| `JudgmentLedger` / `PendingStore` / `BudgetStore` | `storage.contradiction_store.IndexContradictionStore` |

默认关（`BLADEX_CONTRADICTION_ENABLED`）：关着时 `run_pass` 在第一行返回，不读 Index、不调模型、不发 HTTP。
**只标不删**：本模块不碰 fact、墓碑与取代（守卫 `test_bb_flag_never_deletes.py` 的静态扫描覆盖本文件）。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import structlog
from bladex_core import consistency as cs
from bladex_core.fact import ItemKind
from bladex_core.flags import flag_enabled

from bladex_proxy import router_sdk
from bladex_proxy.export_backflow import Requester
from bladex_proxy.storage.contradiction_store import IndexContradictionStore
from bladex_proxy.storage.memory_index import MemoryIndex

logger = structlog.get_logger()


class RouterJudge:
    """判定上游：Router 网关，purpose `consistency_judge`。供应商异常一律翻成 `JudgeUpstreamError`（文本过 `error_text`）。"""

    def __init__(self, *, model: str, api_base: str = "", api_key: str = "", timeout: float = 420.0) -> None:
        self._model, self._api_base, self._api_key, self._timeout = model, api_base, api_key, timeout

    def __call__(self, prompt: str) -> cs.JudgeReply:
        kwargs: dict[str, Any] = {
            "model": self._model, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": 4096, "timeout": self._timeout,
        }
        if self._api_base:
            kwargs["api_base"] = self._api_base
        if self._api_key:
            kwargs["api_key"] = self._api_key
        try:
            resp = router_sdk.completion(purpose="consistency_judge", **kwargs)
        except Exception as e:  # noqa: BLE001 —— 上游异常种类多；统一折成 JudgeUpstreamError，本模块据此停判留 pending
            raise cs.JudgeUpstreamError(router_sdk.error_text(e)) from e
        usage = getattr(resp, "usage", None)
        return cs.JudgeReply(
            text=resp.choices[0].message.content or "",
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            model=self._model,
        )


class ProxySink:
    """Hub 出口：`POST /admin/contradictions`。proxy 不通 / 5xx / 401 ⇒ `SinkUnavailable`（留 pending）。

    4xx（载荷被端点拒绝）是我们自己的 bug，不是暂时故障：记 `contradiction_flag_rejected` 错误日志后放行，
    不重试（重试同一份坏载荷只会无限空转）。
    """

    def __init__(self, requester: Requester) -> None:
        self._req = requester

    def append_flag(self, pair_key: str, payload: dict[str, Any]) -> None:
        try:
            status, body = self._req("POST", "/admin/contradictions", {"pair_key": pair_key, "payload": payload})
        except OSError as e:  # URLError / TimeoutError 都是 OSError 子类
            raise cs.SinkUnavailable(str(e)) from e
        if status in (200, 202):
            return
        if 400 <= status < 500 and status not in (401, 408, 429):
            logger.error("contradiction_flag_rejected", pair=pair_key, status=status, body=json.dumps(body)[:300])
            return
        raise cs.SinkUnavailable(f"HTTP {status}")


def build_detector(
    index: MemoryIndex, *, judge: cs.JudgeFn, sink: cs.ContradictionSink,
    enabled: Callable[[], bool] | None = None,
) -> cs.ContradictionDetector:
    store = IndexContradictionStore(index)
    return cs.ContradictionDetector(
        judge=judge, budget=cs.JudgeBudget(store=store), ledger=store, sink=sink, pending=store,
        settled=index.contradiction_settled, enabled=enabled,
    )


@dataclass
class PassSummary:
    skipped: str = ""
    new_facts: int = 0
    judged: int = 0
    flagged: int = 0
    pending: int = 0
    budget_exhausted: bool = False


def run_pass(
    index: MemoryIndex, detector: cs.ContradictionDetector, *, pass_started: datetime, rebuilding: bool = False,
) -> PassSummary:
    """consolidator 每个 pass 的尾部：对本轮新入库的 fact 逐条 `detector.process`，再续判 pending。

    - 关着 ⇒ 第一行返回（零读写）。全量重建 ⇒ `rebuilding=True`，零判定（标记来自 Hub 重放）。
    - 「新 fact」= `created_at` 晚于游标的现行、非 file_ref fact。**游标缺失时记为本 pass 开始时刻**——
      只处理本 pass 自己写入的，不回扫存量（拍板：不做存量回扫）。
    """
    out = PassSummary()
    if not flag_enabled(cs.ENABLED_FLAG):
        out.skipped = "disabled"
        return out
    if rebuilding:
        out.skipped = "rebuild"
        return out
    store = IndexContradictionStore(index)
    cursor = store.cursor()
    if not cursor:
        cursor = pass_started.astimezone(UTC).isoformat()
        store.set_cursor(cursor)
    fresh = sorted(
        (f for f in index.all_facts()
         if f.created_at.astimezone(UTC).isoformat() > cursor and not f.t_invalid and not f.superseded_by
         and f.item_kind != ItemKind.FILE_REF),
        key=lambda f: f.created_at)
    out.new_facts = len(fresh)

    def _fold(r: cs.ProcessResult) -> None:
        out.judged += r.judged
        out.flagged += len(r.flagged)
        out.budget_exhausted = out.budget_exhausted or r.budget_exhausted

    # 先续判上一轮留下的 pending（更老的先来），再处理本轮新 fact——本轮刚进 pending 的留到下一 pass 再试。
    def _load(fid: str) -> tuple[Any, Any] | None:
        fact = index.get_fact(fid)
        return None if fact is None else (fact, index.contradiction_neighbors(fact))

    for r in detector.resume(_load):
        _fold(r)
    for f in fresh:
        if out.budget_exhausted:
            store.add(f.id)  # 日上限已到：新 fact 不丢，进 pending 次日续
        else:
            _fold(detector.process(f, index.contradiction_neighbors(f)))
        store.set_cursor(f.created_at.astimezone(UTC).isoformat())
    out.pending = len(store.ids())
    return out
