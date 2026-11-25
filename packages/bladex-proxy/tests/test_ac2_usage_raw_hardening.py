"""批 AC2 步骤 3：`capture.usage_raw` 递归加深度上限与环检测（原则 13：截断要显式、可 grep）。

- `MagicMock` 形态的 usage（`model_dump()` 每次返回新 mock）不再递归到 RecursionError：
  返回带截断标记的结果、打 `capture_usage_raw_truncated`，单次 < 10 ms；
- 嵌套里某一支超深 / 成环 ⇒ 只在那个位置写标记，兄弟键原样保留（部分原文）；
- 真实形状（AC0 夹具，三种载体）逐字不变、不告警。
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from bladex_proxy import capture
from bladex_proxy.capture import (
    USAGE_RAW_MAX_DEPTH,
    USAGE_RAW_TRUNC_CYCLE,
    USAGE_RAW_TRUNC_DEPTH,
    USAGE_RAW_TRUNC_KEY,
    usage_norm,
    usage_raw,
)

FIX = Path(__file__).parent / "fixtures" / "ac_upstream_usage_shapes.json"


class _Log:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def warning(self, event: str, **kw) -> None:  # noqa: ANN003
        self.events.append((event, kw))

    def __getattr__(self, name: str):  # noqa: ANN204
        return lambda *a, **k: None


@pytest.fixture()
def log(monkeypatch) -> _Log:  # noqa: ANN001
    lg = _Log()
    monkeypatch.setattr(capture, "logger", lg)
    return lg


def _truncated(lg: _Log) -> list[dict]:
    return [kw for e, kw in lg.events if e == "capture_usage_raw_truncated"]


def test_magicmock_usage_is_cut_with_marker_fast(log: _Log) -> None:
    times = []
    for _ in range(5):
        m = MagicMock()  # 构造不计时：只量 usage_raw 本身
        t0 = time.perf_counter()
        out = usage_raw(m)
        times.append(time.perf_counter() - t0)
        assert out == {USAGE_RAW_TRUNC_KEY: USAGE_RAW_TRUNC_DEPTH}
    dt = sorted(times)[2]
    assert dt < 0.010, f"median {dt * 1000:.1f} ms of {[round(t * 1000, 1) for t in times]}"
    kw = _truncated(log)[0]
    assert len(_truncated(log)) == 5
    assert kw["reasons"] == ["depth"] and kw["carrier"] == "MagicMock"
    # 读方：截断的原文按 NO-DATA 读，不是 0
    assert usage_norm(out)["input"] is None


def test_magicmock_leaf_keeps_siblings(log: _Log) -> None:
    out = usage_raw({"prompt_tokens": 5, "completion_tokens": 1, "details": MagicMock()})
    assert out["prompt_tokens"] == 5 and out["completion_tokens"] == 1
    assert out["details"] == USAGE_RAW_TRUNC_DEPTH
    assert usage_norm(out)["input"] == 5
    assert len(_truncated(log)) == 1


def test_cycle_is_marked_in_place(log: _Log) -> None:
    d: dict = {"prompt_tokens": 3}
    d["self"] = d
    ns = SimpleNamespace(prompt_tokens=7)
    ns.back = ns
    assert usage_raw(d) == {"prompt_tokens": 3, "self": USAGE_RAW_TRUNC_CYCLE}
    assert usage_raw(ns) == {"prompt_tokens": 7, "back": USAGE_RAW_TRUNC_CYCLE}
    assert [kw["reasons"] for kw in _truncated(log)] == [["cycle"], ["cycle"]]


def test_shared_but_acyclic_object_is_not_a_cycle(log: _Log) -> None:
    shared = {"cached_tokens": 4}
    out = usage_raw({"a": shared, "b": shared})
    assert out == {"a": {"cached_tokens": 4}, "b": {"cached_tokens": 4}}
    assert _truncated(log) == []


def test_depth_limit_position(log: _Log) -> None:
    deep: dict = {"v": 1}
    for _ in range(USAGE_RAW_MAX_DEPTH + 3):
        deep = {"n": deep}
    out = usage_raw(deep)
    cur = out
    depth = 0
    while isinstance(cur, dict):
        cur = cur["n"]
        depth += 1
    assert cur == USAGE_RAW_TRUNC_DEPTH and depth == USAGE_RAW_MAX_DEPTH


def _carriers(raw: dict) -> list[object]:
    def _ns(v: object) -> object:
        if isinstance(v, dict):
            return SimpleNamespace(**{k: _ns(x) for k, x in v.items()})
        return v

    class _Pyd:
        def __init__(self, d: dict) -> None:
            self._d = d

        def model_dump(self) -> dict:
            return copy.deepcopy(self._d)

    return [copy.deepcopy(raw), _ns(raw), _Pyd(raw)]


def test_real_shapes_unchanged(log: _Log) -> None:
    cells = json.loads(FIX.read_text())["cells"]
    assert len(cells) >= 18
    for cell in cells:
        raw = cell["usage"]
        for carrier in _carriers(raw):
            assert usage_raw(carrier) == raw, (cell["model"], cell["protocol"], cell["stream"])
    assert _truncated(log) == []
