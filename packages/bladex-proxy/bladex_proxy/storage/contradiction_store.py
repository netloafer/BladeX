"""批 BB · 2b：矛盾检测的持久化适配——把 `bladex_core.consistency` 的四个存储协议落进 Memory Index meta。

只有 consolidator（Index 单写者）会用这里的写方法。键前缀与它们在重建里的命运登记在 `memory_index.py`
（`_CONTRADICTION_*_PREFIX`，闭集守卫 `test_aa2_*::test_every_meta_prefix_is_classified`）：

| 协议 | 键 | 重建 |
|---|---|---|
| `JudgmentLedger` | `contradiction_judged/<pair>/<sha>-<sha>` = `{record, parse_failures}` | 保留（不重判） |
| `PendingStore` | `contradiction_state/pending/<fact_id>` | 保留 |
| `BudgetStore` | `contradiction_state/day/<YYYY-MM-DD>`、`contradiction_state/fact/<fact_id>` | 保留（重启不清零） |
| 新 fact 游标 | `contradiction_state/cursor` | 保留 |
"""

from __future__ import annotations

from typing import Any

from bladex_proxy.storage.memory_index import (
    _CONTRADICTION_JUDGED_PREFIX,
    _CONTRADICTION_STATE_PREFIX,
    MemoryIndex,
)

_PENDING = f"{_CONTRADICTION_STATE_PREFIX}pending/"
_DAY = f"{_CONTRADICTION_STATE_PREFIX}day/"
_FACT = f"{_CONTRADICTION_STATE_PREFIX}fact/"
_CURSOR = f"{_CONTRADICTION_STATE_PREFIX}cursor"


def _jkey(pair_key: str, shas: tuple[str, str]) -> str:
    return f"{_CONTRADICTION_JUDGED_PREFIX}{pair_key}/{shas[0]}-{shas[1]}"


class IndexContradictionStore:
    """`JudgmentLedger` + `PendingStore` + `BudgetStore` + 游标，同一个对象（共用一个 Index 句柄）。"""

    def __init__(self, index: MemoryIndex) -> None:
        self._ix = index

    # ── JudgmentLedger ──
    def is_judged(self, pair_key: str, shas: tuple[str, str]) -> bool:
        cur = self._ix.contradiction_kv_get(_jkey(pair_key, shas))
        return bool(cur and cur.get("record"))

    def record(self, pair_key: str, shas: tuple[str, str], record: dict[str, Any]) -> None:
        cur = self._ix.contradiction_kv_get(_jkey(pair_key, shas)) or {}
        cur["record"] = record
        self._ix.contradiction_kv_put(_jkey(pair_key, shas), cur)

    def parse_failures(self, pair_key: str, shas: tuple[str, str]) -> int:
        cur = self._ix.contradiction_kv_get(_jkey(pair_key, shas))
        return int((cur or {}).get("parse_failures", 0))

    def record_parse_failure(self, pair_key: str, shas: tuple[str, str]) -> None:
        cur = self._ix.contradiction_kv_get(_jkey(pair_key, shas)) or {}
        cur["parse_failures"] = int(cur.get("parse_failures", 0)) + 1
        self._ix.contradiction_kv_put(_jkey(pair_key, shas), cur)

    # ── PendingStore ──
    def add(self, fact_id: str) -> None:
        if self._ix.contradiction_kv_get(f"{_PENDING}{fact_id}") is None:
            self._ix.contradiction_kv_put(f"{_PENDING}{fact_id}", {"n": 1})

    def remove(self, fact_id: str) -> None:
        self._ix.contradiction_kv_delete(f"{_PENDING}{fact_id}")

    def ids(self) -> list[str]:
        return [k[len(_PENDING):] for k, _ in self._ix.contradiction_kv_scan(_PENDING)]

    # ── BudgetStore ──
    def day_input(self, day: str) -> int:
        return int((self._ix.contradiction_kv_get(f"{_DAY}{day}") or {}).get("n", 0))

    def add_day_input(self, day: str, n: int) -> None:
        self._ix.contradiction_kv_put(f"{_DAY}{day}", {"n": self.day_input(day) + int(n)})

    def fact_calls(self, fact_id: str) -> int:
        return int((self._ix.contradiction_kv_get(f"{_FACT}{fact_id}") or {}).get("n", 0))

    def add_fact_call(self, fact_id: str) -> None:
        self._ix.contradiction_kv_put(f"{_FACT}{fact_id}", {"n": self.fact_calls(fact_id) + 1})

    # ── 新 fact 游标（ISO 时间戳；缺失 = 还没开始）──
    def cursor(self) -> str:
        return str((self._ix.contradiction_kv_get(_CURSOR) or {}).get("created_at", ""))

    def set_cursor(self, created_at: str) -> None:
        self._ix.contradiction_kv_put(_CURSOR, {"created_at": created_at})
