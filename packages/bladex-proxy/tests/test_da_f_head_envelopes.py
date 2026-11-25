"""批 DA · F（MQ-L129 / MQ-L130）：三种机器信封剥离 + 举证摘录空白归一化。

信封字面量逐字取自 CA 快照 11-17 的真实样本（Hermes 中断恢复 / Hermes 定时任务 / Pi 压缩续接）。
"""

from __future__ import annotations

import pytest
from bladex_core import ledger_runtime
from bladex_core.ledger_runtime import candidate_units, verify_switch_evidence
from bladex_proxy.agency import _clean_user_text, _last_user_text
from bladex_proxy.agency import notes as notes_mod

_WORDS = "这也是泰山啤酒重整调研的一部分。"
_INTERRUPT = ("[Context from the interrupted assistant response]\n"
              "[This response was interrupted by a user correction.]")
_CRON = ('[IMPORTANT: You are running as a scheduled cron job. DELIVERY: Your final response '
         'will be automatically delivered to the user — do NOT use send_message. SILENT: If there '
         'is genuinely nothing new to report, respond with exactly "[SILENT]" (nothing else) to '
         'suppress delivery. Never combine [SILENT] with content.]')
_CRON_TASK = "跟进微软开源模型 Mage-VL-Cowork 的社区反馈。\n\n任务：\n1. 用 web_search 查询"
_PI = ("The conversation history before this point was compacted into the following summary:"
       "\n\n<summary>\n## Goal\n评估 qwen2.5:3b\n</summary>")


def _user(text: str) -> dict:
    return {"role": "user", "content": text}


def test_hermes_interrupt_head_is_stripped_and_words_kept() -> None:
    cleaned, marks = _clean_user_text(f"{_INTERRUPT}\n\n{_WORDS}")
    assert cleaned == _WORDS
    assert "hermes_interrupt" in marks


def test_hermes_cron_strips_only_the_instruction_and_keeps_task() -> None:
    cleaned, marks = _clean_user_text(f"{_CRON}\n\n{_CRON_TASK}")
    assert cleaned == _CRON_TASK.strip()
    assert "[SILENT]" not in cleaned and "scheduled cron job" not in cleaned
    assert "hermes_cron" in marks


def test_pi_compaction_summary_is_whole_machine_text() -> None:
    cleaned, marks = _clean_user_text(_PI)
    assert cleaned == ""
    assert "pi_compaction_summary" in marks


@pytest.mark.parametrize("env", [_INTERRUPT, _CRON, _PI])
def test_envelope_only_message_is_empty_and_walks_back(env: str) -> None:
    assert _clean_user_text(env)[0] == ""
    got = _last_user_text([_user("先前那句真话"), _user(env)], for_goal=True)
    assert got == "先前那句真话"


def test_envelope_text_in_the_middle_is_not_stripped() -> None:
    # MQ-A19 自指陷阱：用户在讨论这些句子本身，不在消息开头 ⇒ 原样保留
    for env in (_INTERRUPT, _CRON, _PI):
        text = f"为什么会出现这一句：{env} 帮我查一下"
        cleaned, marks = _clean_user_text(text)
        assert cleaned == text.strip()
        assert not set(marks) & {"hermes_interrupt", "hermes_cron", "pi_compaction_summary"}


def test_interrupt_envelope_no_longer_leaks_machine_words_into_query() -> None:
    # L128 §4：117 字符信封当用户话 ⇒ 17 个单元里 12 个英文机器词
    got = _last_user_text([_user(f"{_INTERRUPT}\n\n{_WORDS}")], for_goal=True)
    units = candidate_units(got)
    assert not units & {"context", "assistant", "interrupted", "response"}


def test_evidence_ignores_cjk_latin_boundary_space() -> None:
    # MQ-L130：用户原话「如果摆在 iPhone」、摘录「如果摆在iPhone」
    assert verify_switch_evidence("站在personal agent角度，如果摆在iPhone",
                                  "站在personal agent角度，如果摆在 iPhone") is True


def test_evidence_still_rejects_fabricated_text() -> None:
    assert verify_switch_evidence("完全另一句话", "站在personal agent角度，如果摆在 iPhone") is False
    assert verify_switch_evidence("", "任意") is None


def test_norm_evidence_removes_all_whitespace() -> None:
    assert ledger_runtime._norm_evidence(" A b\n\tC ") == "abc"


def test_mutation_guards_exist() -> None:
    # 反向变异的锚：三种形态各自在表里；删掉任一条上面对应用例必红
    assert {k for k, _ in notes_mod._HEAD_ENVELOPE_RES} == {"hermes_interrupt", "hermes_cron"}
    assert "pi_compaction_summary" in {k for k, _ in notes_mod._MACHINE_ONLY_PREFIXES}
