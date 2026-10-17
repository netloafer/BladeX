"""MQ-S44（批 Q3A）：后台进程完成通知 `[IMPORTANT: Background process …]` 判 aux（DISTILL_ONLY）。

与批 E · E5 同一机制：`_AUX_USER_PATTERNS` 加前缀，只认消息开头（信封偏移内），正文深处引用不算。

🔴 归类偏离调度提示词（提示词写"完整 aux"）：Hub 11 例的 assistant 侧都在做实事（读训练
结果、出 1,076 字模型对比表、接着调工具）。完整 aux = 这些结论不进蒸馏 + 整轮降便宜档
（刚性原则 9 的 07-28 事故形状）。故归 DISTILL_ONLY：user 侧（通知）剥掉不进蒸馏、不给
账本面；assistant 侧照收、路由不降档。与 `claude_code_task_notification` 同口径，也是
`test_g123_goal_capture` 里钉该缺口时预先写下的修法（expected = TURN_SCAFFOLD）。

🔴 病例来源更正：调度提示词写的是「Claude Code 后台进程通知」，Hub 只读副本（2026-09-29）
实查——末条 user 以该前缀**开头**的 11 轮**全部是 `hermes:default`**（另 6 轮在更早的 user
消息里、6 轮在压缩检查点正文里），Claude Code 0 轮 ⇒ 规则名按真实来源叫 `hermes_background_process`。
修前 11 轮全判 REAL（`auxiliary=False`、`aux_source=""`）。
"""

from __future__ import annotations

import pytest
from bladex_core.envelope import strip_envelopes
from bladex_core.task_goal import TURN_REAL, TURN_SCAFFOLD
from bladex_proxy.identity import (
    CONTINUATION_AUX_RULES,
    DISTILL_ONLY_AUX_RULES,
    classify_auxiliary,
    classify_turn_disposition,
    is_cheap_tier_auxiliary,
    is_ledgerless_auxiliary,
    resolve_identity,
)
from bladex_proxy.models import ChatCompletionRequest

_RULE = "hermes_background_process"
_HERMES_SYS = "You are Hermes Agent, an intelligent AI assistant created by Nous Research."
_CC_SYS = ("You are Claude Code, Anthropic's official CLI for Claude.\n\n"
           "You are an interactive agent that helps users with software engineering tasks.")

# ── Hub 原文（hermes:default 09-08 / 09-10 两例，路径与命令脱敏、截短）──
_DONE = ("[IMPORTANT: Background process proc_8a79cd4b9df1 completed normally (exit code 0).\n"
         "Command: cd ~/work/demo && nohup .venv/bin/python train.py --epochs 50 > train_log.txt 2>&1 &\n"
         "Output:\n]")
_EXITED = ("[IMPORTANT: Background process proc_005266a5fd51 exited (exit code None).\n"
           "Command: cd ~/work/demo && exec .venv/bin/python resume.py > resume_log.txt 2>&1\n"
           "Output:\n]")
_TOOL = {"type": "function", "function": {"name": "terminal"}}


def _hermes(notice: str) -> list[dict]:
    return [{"role": "system", "content": _HERMES_SYS},
            {"role": "user", "content": "帮我把训练跑起来，跑完告诉我"},
            {"role": "assistant", "content": "已在后台启动训练。"},
            {"role": "user", "content": notice}]


@pytest.mark.parametrize("notice", [_DONE, _EXITED], ids=["completed", "exited"])
def test_background_process_notice_is_distill_only_aux(notice):
    msgs = _hermes(notice)
    is_aux, hit = classify_auxiliary(msgs)
    assert (is_aux, hit) == (True, _RULE)
    # user 侧不进蒸馏、不给账本面；路由不降便宜档（assistant 侧在做实事）
    assert hit in DISTILL_ONLY_AUX_RULES and hit not in CONTINUATION_AUX_RULES
    assert is_ledgerless_auxiliary(hit) is True and is_cheap_tier_auxiliary(hit) is False
    ident, _ = resolve_identity({"authorization": "Bearer k", "user-agent": "x"},
                                ChatCompletionRequest(model="m", messages=msgs, tools=[_TOOL]))
    assert ident.auxiliary is True and ident.aux_source == _RULE
    # Index 重建重判能认出：scaffold ⇒ user 侧交 envelope 剥空
    assert classify_turn_disposition(msgs) == (TURN_SCAFFOLD, _RULE)
    assert strip_envelopes(notice) == ("", [_RULE])


@pytest.mark.parametrize("sys_prompt", [_HERMES_SYS, _CC_SYS], ids=["hermes", "claude-code"])
def test_normal_turns_untouched(sys_prompt):
    msgs = [{"role": "system", "content": sys_prompt},
            {"role": "user", "content": "看一下训练日志，loss 为什么不降"}]
    assert classify_auxiliary(msgs) == (False, "")
    assert classify_turn_disposition(msgs) == (TURN_REAL, "")


def test_marker_deep_in_body_is_not_an_envelope():
    """用户在正文深处贴了一段通知原文来提问 ⇒ 不是内部调用。"""
    deep = "上次那个任务我看日志是这样的，" + "x" * 3000 + "\n" + _DONE + "\n为什么没有输出？"
    assert classify_auxiliary([{"role": "system", "content": _HERMES_SYS},
                               {"role": "user", "content": deep}]) == (False, "")
