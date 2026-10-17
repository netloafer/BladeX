"""MQ-R10（批 Q3A）：`tool_results_enriched` 的 matched / unmatched 读数退役。

`enrich_tool_results` 的 `call_map` 取自**本轮响应**捕获的 call，而请求里的 tool_result
必然对应**上一轮**的 call ⇒ `matched` 结构性恒 0（两跑实测 44/44、18/18）。它误导过排查
（08-28 首轮把它读成"拼接台账一次没匹配上"）。一个恒为常数的读数比没有读数更坏。

退役 = 删掉这组计数与它唯一的产出（`tool_results_enriched` 日志行；全仓无消费方）。
回填行为不变（各分支照旧写 `ToolEvent`）。本守卫：该事件或这组字段再出现即红。
"""

from __future__ import annotations

import re
from pathlib import Path

from bladex_proxy.models import Identity, ToolEvent, Turn
from bladex_proxy.storage import pipeline_worker
from bladex_proxy.storage.pipeline_worker import enrich_tool_results
from structlog.testing import capture_logs


def _turn() -> Turn:
    return Turn(
        identity=Identity(user_id="u", agent_id="a", session_id="s"),
        request_messages=[
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "call_prev", "type": "function",
                             "function": {"name": "Bash", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_prev", "content": "ok"},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu_1",
                                          "content": "done"}]},
        ],
        tool_events=[ToolEvent(tool_name="Read", direction="call", tool_call_id="call_now")],
    )


def test_enrich_still_backfills_results_but_emits_no_matched_readout():
    turn = _turn()
    with capture_logs() as cap:
        enrich_tool_results(turn)
    results = {te.tool_call_id: te for te in turn.tool_events if te.direction == "result"}
    assert results["call_prev"].tool_name == "Bash" and results["call_prev"].result == "ok"
    assert results["tu_1"].result == "done"
    assert not [r for r in cap if r.get("event") == "tool_results_enriched"]
    assert not [r for r in cap if "matched" in r or "unmatched" in r]


def test_retired_readout_does_not_come_back_in_source():
    src = Path(pipeline_worker.__file__).read_text(encoding="utf-8")
    code = "\n".join(line.split("#", 1)[0] for line in src.splitlines())
    assert '"tool_results_enriched"' not in code
    assert not re.search(r"\b(un)?matched\s*=", code), "matched / unmatched 读数退役了（MQ-R10）"
