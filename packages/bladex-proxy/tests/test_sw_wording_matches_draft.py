"""批 SW · 步骤 3（拍板 (82)，MQ-L109 成因 ②）：运行时各槽位文案 = `wording_drafts.py` 的甲′，逐槽对账。

草案脚本（`docs/benchmarks/sw/wording_drafts.py`）是文案的出处，运行时是落点。以后谁只改了一边，
这条就红——不然两处会各自漂（刚性原则 12：同一件事在两处各有一个值 = 缺陷）。

S5（工具面 "do NOT switch or create ledgers here"）**不在甲′ 的改动里**，这里直接钉字面：
它是 G16「工具面 switch = 0」判据的载体，拍板 (80)-a 明确不动。
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from bladex_proxy.agency import runtime as R
from bladex_proxy.toolface import TOOL_FACE_SCHEMA_BUDGET_CHARS, TOOL_SCHEMAS

_REPO = Path(__file__).resolve().parents[3]
_SYSTEM_DIRS = (_REPO / "packages/bladex-proxy/bladex_proxy/assets/system", _REPO / "config/system")
_S5 = "You are continuing the task above — do NOT switch or create ledgers here."


def _drafts():
    path = _REPO / "docs/benchmarks/sw/wording_drafts.py"
    if not path.exists():       # 公开树 / 快照里没有 docs/ 时跳过，不当成红
        pytest.skip("docs/benchmarks/sw/wording_drafts.py 不在")
    spec = importlib.util.spec_from_file_location("sw_wording_drafts", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sw_wording_drafts"] = mod
    spec.loader.exec_module(mod)
    return mod


def _runtime_slots() -> dict[str, str]:
    tools = {t["function"]["name"]: t["function"]["description"] for t in TOOL_SCHEMAS}
    md = (_SYSTEM_DIRS[0] / "TOOLS.md").read_text()
    resume = md[md.index("- Resuming something"):md.index("- Already on the right ledger")].rstrip("\n")
    drift = md[md.index("- **Do not switch because the topic drifted.**"):
               md.index("- **Do not update every turn.**")].rstrip("\n")
    return {
        "S1_first_step_with_ledger": R._FIRST_STEP_WITH_LEDGER,
        "S2_first_step_no_ledger": R._FIRST_STEP_NO_LEDGER,
        "S3_matching_header": R._MATCHING_LEDGERS_HEADER,
        "S4_recent_header": R._RECENT_LEDGERS_HEADER,
        "S6_switch_description": tools["bladex_ledger_switch"],
        "S7_read_description": tools["bladex_ledger_read"],
        "S8_tools_md_resume": resume,
        "S9_tools_md_drift": drift,
    }


@pytest.mark.parametrize("slot", sorted(_runtime_slots()))
def test_runtime_slot_equals_draft_jiap(slot):
    d = _drafts()
    assert _runtime_slots()[slot] == d.JIAP[slot], f"{slot}: 运行时与 wording_drafts.JIAP 不一致"


def test_the_c_class_counterexample_is_in_s1():
    """甲′ 与甲的唯一区别（拍板 (80)-a）：S1 里那句 C 类反例。"""
    assert ("two separate investigations of different items — other models, companies or tools — "
            "are different tasks even when the kind of work is the same") in R._FIRST_STEP_WITH_LEDGER


def test_no_read_first_instruction_survives():
    """甲′ 删了所有「先 read 再 switch」（混合调用路径上与工具面 S5 互相打架）。"""
    for text in (R._MATCHING_LEDGERS_HEADER, R._RECENT_LEDGERS_HEADER, _runtime_slots()["S7_read_description"],
                 _runtime_slots()["S8_tools_md_resume"]):
        low = text.lower()
        assert "before switching" not in low and "read it first" not in low and "read one with" not in low


def test_s5_tool_face_sentence_untouched():
    assert R._RECORD_ONLY_WITH_LEDGER.endswith(_S5)
    assert "SAME" not in R._RECORD_ONLY_WITH_LEDGER


def test_tools_md_copies_stay_identical():
    """两份 TOOLS.md（包内资产 / 部署 config）同改，逐字相同。"""
    a, b = (d / "TOOLS.md" for d in _SYSTEM_DIRS)
    assert a.read_text() == b.read_text()


def test_tool_schema_budget_after_wording():
    now = len(json.dumps(TOOL_SCHEMAS, ensure_ascii=False))
    assert now <= TOOL_FACE_SCHEMA_BUDGET_CHARS
    # wording_drafts 甲′ 出数 7250；批 PW 步骤 2 叠加 D（parent_ledger_id 描述 +78）与 A（update 加 ledger_id，+197）= 7525。
    # 甲′ 的槽位（S1–S8）本批未动，只有工具面总长这个数要同步。
    assert now == 7525
