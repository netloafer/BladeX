"""ADR-0033 §2.2 轮次规则 —— 一套体系一个轮次定义（批 U · U1.1，2026-09-23）。

钉五件事：
① **同源**：`ledger_runtime.user_turn_index` 与 `task_unit.build_task_units` 都调
   `task_unit.turn_starts`（源码 + 行为两道）；把 `user_turn_index` 改回按 role=user 计数
   ⇒ 插话用例红（反向变异 1）。
② **输入形态**：三个入站协议（chat / `/v1/messages` / `/v1/responses`）经各自的归一化
   后切出的轮次一致——`/v1/messages` 的 tool_result 原生落在 user 角色，若在归一化前切，
   CC 的轮次会碎（ADR-0033 §2.2 末段）。
③ 工具循环中的用户插话、连续 user 并入当前轮次；agent 停下后的 user 开新轮次。
④ **MQ-CA18**：轮次开头的连续 user 块里，身份原文取第一条非机器文本（codex 首轮
   `# AGENTS.md instructions` / `<environment_context>` 之后的真人那条）；全是机器文本 ⇒ 回落首条。
⑤ 口径切换的三句话（unit_key 新旧不互认 · 存量不迁移 · 重建时重放）落在
   `build_task_units` 的 docstring 里，不许被悄悄删掉。
"""

from __future__ import annotations

import ast
import inspect
import textwrap

from bladex_core import ledger_runtime, task_unit
from bladex_core.ledger_runtime import user_turn_index
from bladex_core.task_unit import build_task_units, make_unit_key, turn_starts

_SYS = {"role": "system", "content": "sys"}


def _u(t):
    return {"role": "user", "content": t}


def _a(t="", tools=False, cid="c"):
    m = {"role": "assistant", "content": t}
    if tools:
        m["tool_calls"] = [{"id": cid, "type": "function",
                            "function": {"name": "bash", "arguments": "{}"}}]
    return m


def _t(t="r", cid="c"):
    return {"role": "tool", "tool_call_id": cid, "content": t}


# 工具循环里用户插了一句话：旧口径（role=user 计数）= 3 轮，§2.2 = 2 轮
_INTERJECT = [_SYS, _u("q1"), _a(tools=True), _t(), _u("补充：只看 src/"),
              _a(tools=True), _t(), _a("done"), _u("q2")]
_CONSECUTIVE = [_SYS, _u("q1"), _u("q1 again"), _a("a1"), _u("q2")]
_QA = [_SYS, _u("q1"), _a("which one?"), _u("the second"), _a("ok")]
_FIXTURES = [[], [_SYS], _INTERJECT, _CONSECUTIVE, _QA,
             [_t("orphan"), _u("hi"), _t("r")],
             [_u("x"), {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c",
                                                      "content": "r"}]}, _a("ok"), _u("y")]]


# ── ① 同源 ────────────────────────────────────────────────────────────────


def test_user_turn_index_and_build_task_units_share_turn_starts_source():
    assert "turn_starts(" in inspect.getsource(ledger_runtime.user_turn_index), \
        "user_turn_index 必须调 task_unit.turn_starts（算法唯一，ADR-0033 §2.2）"
    assert "turn_starts(" in inspect.getsource(task_unit.build_task_units)
    # user_turn_index 的**代码**（去掉 docstring）里不许再自己数 role（改回按 role=user 计数 = 第二份算法）
    fn = ast.parse(textwrap.dedent(inspect.getsource(ledger_runtime.user_turn_index))).body[0]
    code = "\n".join(ast.unparse(stmt) for stmt in fn.body[1:])
    assert "role" not in code and "'user'" not in code, code


def test_user_turn_index_equals_unit_count_on_every_fixture():
    for msgs in _FIXTURES:
        n = len(turn_starts(msgs))
        assert user_turn_index(msgs) == n == len(build_task_units(msgs)), msgs


def test_interjection_in_tool_loop_is_not_a_new_turn():
    """反向变异 1 的判别用例：按 role=user 计数会得 3。"""
    assert user_turn_index(_INTERJECT) == 2
    units = build_task_units(_INTERJECT)
    assert [u.msg_start for u in units] == [1, 8]
    assert units[0].msg_end == 7 and units[0].evidence_count == 2
    assert units[0].conclusion_index == 7


def test_consecutive_users_and_agent_question():
    assert turn_starts(_CONSECUTIVE) == [1, 4]
    assert turn_starts(_QA) == [1, 3], "agent 反问、用户回答 = 新轮次（§2.2 明说不修的边界②）"


def test_objects_with_attributes_are_accepted():
    """`identity.resolve_identity` 传的是 pydantic 消息对象，不是 dict。"""
    from bladex_proxy.models import ChatCompletionRequest

    req = ChatCompletionRequest(messages=_INTERJECT)
    assert user_turn_index(req.messages) == user_turn_index(_INTERJECT) == 2


# ── ② 三协议输入形态 ──────────────────────────────────────────────────────


def test_three_protocols_normalize_before_splitting():
    """同一段对话（工具循环里插话）走三条入站归一化，轮次数都是 2。"""
    from bladex_proxy.anthropic import parse_anthropic_request
    from bladex_proxy.responses import parse_responses_request

    anthropic_body = {"model": "m", "system": "sys", "messages": [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "bash",
                                           "input": {}}]},
        # CC 的形态：tool_result 与用户插话同在一条 user 消息里
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "r"},
                                     {"type": "text", "text": "补充：只看 src/"}]},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t2", "name": "bash",
                                           "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t2", "content": "r"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
        {"role": "user", "content": "q2"},
    ]}
    responses_body = {"model": "m", "instructions": "sys", "input": [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "q1"}]},
        {"type": "function_call", "call_id": "t1", "name": "bash", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "t1", "output": "r"},
        {"type": "message", "role": "user", "content": [{"type": "input_text",
                                                         "text": "补充：只看 src/"}]},
        {"type": "function_call", "call_id": "t2", "name": "bash", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "t2", "output": "r"},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text",
                                                              "text": "done"}]},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "q2"}]},
    ]}
    chat_msgs = _INTERJECT
    a_msgs, _ = parse_anthropic_request(anthropic_body)
    r_msgs, _ = parse_responses_request(responses_body)
    for name, msgs in (("chat", chat_msgs), ("anthropic", a_msgs), ("responses", r_msgs)):
        assert not any(m.get("role") == "user" and task_unit._has_tool_result_block(m)
                       for m in msgs), f"{name}: tool_result 未归一化成 role=tool"
        assert user_turn_index(msgs) == 2, f"{name}: 轮次碎了 {turn_starts(msgs)}"
        units = build_task_units(msgs)
        assert [u.unit_key for u in units] == [make_unit_key("q1"), make_unit_key("q2")], name

    # 反面：若在归一化**之前**切（Anthropic 原始 messages），tool_result 会被当成 user
    raw = anthropic_body["messages"]
    naive = sum(1 for m in raw if m["role"] == "user")
    assert naive == 4 and user_turn_index(a_msgs) == 2


# ── ④ MQ-CA18 身份原文 ──────────────────────────────────────────────────


_AGENTS_MD = "# AGENTS.md instructions for /repo\n<INSTRUCTIONS>\n仓库协作文档全文\n</INSTRUCTIONS>"
_ENV = "<environment_context>\n  <cwd>/repo</cwd>\n</environment_context>"


def test_codex_first_turn_key_comes_from_human_message():
    human_a = "给 BladeX 加一个只读的管理端点"
    human_b = "修一下 ledger 的并发写"
    s1 = [_SYS, _u(_AGENTS_MD), _u(_ENV), _u(human_a), _a(tools=True), _t()]
    s2 = [_SYS, _u(_AGENTS_MD), _u(_ENV), _u(human_b), _a(tools=True), _t()]
    u1, u2 = build_task_units(s1)[0], build_task_units(s2)[0]
    assert u1.unit_key == make_unit_key(human_a) and u2.unit_key == make_unit_key(human_b)
    assert u1.unit_key != u2.unit_key, "同仓库两个会话的首轮身份不许撞车"
    assert u1.msg_start == 1 and u1.intent_index == 3, "轮次范围从首条 user 起，身份指向真人那条"
    assert turn_starts(s1) == [1], "身份例外不改切分"


def test_all_machine_block_falls_back_to_first_user():
    msgs = [_SYS, _u(_AGENTS_MD), _u(_ENV), _a("ok")]
    u = build_task_units(msgs)[0]
    assert u.intent_index == 1 and u.unit_key == make_unit_key(_AGENTS_MD)


def test_single_user_turn_key_unchanged():
    """开头块只有 1 条（CC / 聊天式 agent 的常态）⇒ 身份 = 首条原文，与 §2.2 原文逐字一致。"""
    text = "<system-reminder>x</system-reminder>\n真实问题"
    u = build_task_units([_SYS, _u(text), _a("ok")])[0]
    assert u.intent_index == 1 and u.unit_key == make_unit_key(text)


def test_later_interjection_never_becomes_identity():
    """只看开头块：工具循环里后到的插话不改身份（否则 key 会在轮次中途翻转）。"""
    before = build_task_units([_SYS, _u(_ENV), _a(tools=True), _t()])[0].unit_key
    after = build_task_units([_SYS, _u(_ENV), _a(tools=True), _t(), _u("真人插话")])[0].unit_key
    assert before == after == make_unit_key(_ENV)


# ── ⑤ 口径切换三句话 ──────────────────────────────────────────────────────


def test_switch_statement_stays_in_docstring():
    doc = task_unit.build_task_units.__doc__ or ""
    for phrase in ("不互认", "不迁移", "重放"):
        assert phrase in doc, f"build_task_units docstring 丢了口径切换的「{phrase}」"
