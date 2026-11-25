"""批 BB · 2a 测试共用：假上游 / 假出口 / 假台账，以及「改一处源码再载入」的反向变异。"""

from __future__ import annotations

import sys
import types
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from bladex_core import consistency as real
from bladex_core.fact import Fact

CONSISTENCY_SRC = Path(real.__file__)


def load_mutant(*replacements: tuple[str, str]) -> types.ModuleType:
    """把 consistency.py 源码里的每个 old 恰好替换一处，编译成独立模块。替换不到 = 测试写坏了，直接抛。"""
    src = CONSISTENCY_SRC.read_text(encoding="utf-8")
    for old, new in replacements:
        assert src.count(old) == 1, f"变异锚点必须唯一出现一次：{old!r}（出现 {src.count(old)} 次）"
        src = src.replace(old, new)
    mod = types.ModuleType("bladex_core._consistency_mutant")
    mod.__file__ = str(CONSISTENCY_SRC)
    sys.modules[mod.__name__] = mod  # dataclass 解析注解要在 sys.modules 里找到本模块
    exec(compile(src, str(CONSISTENCY_SRC), "exec"), mod.__dict__)  # noqa: S102 —— 仅测试里的反向变异
    return mod


def mk_fact(fid: str, content: str, *, day: int = 1, subject: str = "s", attribute: str = "a",
            entities: list[str] | None = None) -> Fact:
    t = datetime(2026, 11, day, 8, 0, tzinfo=UTC) + timedelta(minutes=len(fid))
    return Fact(id=fid, content=content, subject=subject, attribute=attribute,
                entities=entities or ["x", "y"], t_valid=t, audience="all")


class FakeJudge:
    """假上游：按 prompt 里是否含某个词决定判定；记调用数。"""

    def __init__(self, verdict: str = "矛盾", *, tokens: int = 1000, text: str | None = None) -> None:
        self.verdict, self.tokens, self.text = verdict, tokens, text
        self.calls = 0

    def __call__(self, prompt: str) -> Any:
        self.calls += 1
        body = self.text if self.text is not None else f'{{"verdict": "{self.verdict}", "reason": "理由{self.calls}"}}'
        return real.JudgeReply(body, self.tokens, 800, "fake/judge")


class FakeSink:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def append_flag(self, pair_key: str, payload: dict[str, Any]) -> None:
        self.events.append((pair_key, payload))


class FakeLedger:
    def __init__(self) -> None:
        self.records: dict[tuple[str, tuple[str, str]], dict[str, Any]] = {}
        self.fails: dict[tuple[str, tuple[str, str]], int] = {}
        self.writes = 0

    def is_judged(self, pair_key: str, shas: tuple[str, str]) -> bool:
        return (pair_key, shas) in self.records

    def record(self, pair_key: str, shas: tuple[str, str], record: dict[str, Any]) -> None:
        self.writes += 1
        self.records[(pair_key, shas)] = record

    def parse_failures(self, pair_key: str, shas: tuple[str, str]) -> int:
        return self.fails.get((pair_key, shas), 0)

    def record_parse_failure(self, pair_key: str, shas: tuple[str, str]) -> None:
        self.writes += 1
        self.fails[(pair_key, shas)] = self.fails.get((pair_key, shas), 0) + 1


class FakePending:
    def __init__(self) -> None:
        self._ids: list[str] = []
        self.writes = 0

    def add(self, fact_id: str) -> None:
        self.writes += 1
        if fact_id not in self._ids:
            self._ids.append(fact_id)

    def remove(self, fact_id: str) -> None:
        if fact_id in self._ids:
            self.writes += 1
            self._ids.remove(fact_id)

    def ids(self) -> list[str]:
        return list(self._ids)


def neighbors_for(new: Fact, n: int, *, base_cos: float = 0.95) -> list[tuple[Fact, float]]:
    """n 个同 subject 的旧邻居，余弦递减。"""
    return [(mk_fact(f"old{i:02d}", f"旧内容{i}", day=0 + 1), round(base_cos - i * 0.005, 4)) for i in range(n)]


def make_detector(mod: Any = real, *, judge: Callable[[str], Any] | None = None, enabled: bool = True,
                  budget: Any = None, sink: FakeSink | None = None, ledger: FakeLedger | None = None,
                  pending: FakePending | None = None, settled: Callable[[], Any] | None = None) -> tuple[Any, dict[str, Any]]:
    parts = {
        "judge": judge or FakeJudge(), "sink": sink or FakeSink(), "ledger": ledger or FakeLedger(),
        "pending": pending or FakePending(),
    }
    det = mod.ContradictionDetector(
        judge=parts["judge"], budget=budget or mod.JudgeBudget(per_fact=6, daily_input_tokens=10**9),
        ledger=parts["ledger"], sink=parts["sink"], pending=parts["pending"],
        settled=settled or (lambda: mod.SettledPairs({})), enabled=lambda: enabled,
    )
    return det, parts
