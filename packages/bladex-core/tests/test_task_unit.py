"""TaskUnit 验收剧本（ADR-0024 T1）。

覆盖：切分口径 / 跨轮身份稳定性（修正 A）/ 只存引用不存正文（修正 D）/
与退役实现的等价性 / 与 assembly._split_units 的一致性。
"""

from __future__ import annotations

import pytest

from bladex_core.task_unit import (
    TaskUnitStatus,
    build_task_units,
    derive_turn_metadata,
    make_unit_key,
    unit_indices,
)


def _msgs_tool_loop():
    """一个典型工具循环：2 个用户轮，第 1 轮含 2 次工具往返 + 结论。"""
    return [
        {"role": "system", "content": "You are helpful."},                       # 0
        {"role": "user", "content": "查一下北京天气"},                              # 1
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "function": {"name": "search", "arguments": "{}"}}]},   # 2
        {"role": "tool", "name": "search", "tool_call_id": "c1",
         "content": "x" * 3000},                                                  # 3
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c2", "function": {"name": "read", "arguments": "{}"}}]},     # 4
        {"role": "tool", "name": "read", "tool_call_id": "c2", "content": "y" * 50},  # 5
        {"role": "assistant", "content": "北京今天晴，25 度。"},                     # 6  ← 结论
        {"role": "user", "content": "那上海呢"},                                    # 7
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c3", "function": {"name": "search", "arguments": "{}"}}]},   # 8
        {"role": "tool", "name": "search", "tool_call_id": "c3", "content": "z" * 10},  # 9
    ]


# ── 切分口径 ──


def test_split_basic_shape():
    units = build_task_units(_msgs_tool_loop())
    assert len(units) == 2
    assert units[0].status is TaskUnitStatus.CLOSED
    assert units[1].status is TaskUnitStatus.OPEN
    assert units[0].msg_start == 1 and units[0].msg_end == 6
    assert units[1].msg_start == 7 and units[1].msg_end == 9


def test_system_never_enters_unit():
    """system 消息永远不属于任何单元（ADR-0019 口径）。"""
    msgs = [
        {"role": "system", "content": "s0"},
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "mid-system"},
        {"role": "assistant", "content": "yo"},
    ]
    units = build_task_units(msgs)
    assert len(units) == 1
    assert 2 not in unit_indices(units[0], msgs)   # 中间的 system 被跳过
    assert unit_indices(units[0], msgs) == [1, 3]


def test_messages_before_first_user_belong_to_no_unit():
    msgs = [
        {"role": "assistant", "content": "unsolicited"},
        {"role": "user", "content": "hi"},
    ]
    units = build_task_units(msgs)
    assert len(units) == 1
    assert units[0].msg_start == 1


def test_empty_and_no_user():
    assert build_task_units([]) == []
    assert build_task_units([{"role": "system", "content": "s"}]) == []


# ── 结论判定（与 ADR-0019 P2 同口径）──


def test_conclusion_is_last_assistant_without_tool_calls():
    units = build_task_units(_msgs_tool_loop())
    assert units[0].conclusion_index == 6
    assert units[0].conclusion_chars > 0
    # open 单元还没给出结论（末条是 tool）
    assert units[1].conclusion_index == -1


def test_assistant_with_tool_calls_is_not_conclusion():
    msgs = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "思考中", "tool_calls": [
            {"id": "c1", "function": {"name": "t", "arguments": "{}"}}]},
    ]
    units = build_task_units(msgs)
    assert units[0].conclusion_index == -1


# ── 证据聚合 ──


def test_evidence_aggregation():
    units = build_task_units(_msgs_tool_loop())
    assert units[0].evidence_count == 2
    assert units[0].evidence_chars == 3050
    assert units[0].tool_names == ["search", "read"]   # 去重保序
    assert units[1].evidence_count == 1


def test_tool_names_deduped():
    msgs = [{"role": "user", "content": "q"}] + [
        {"role": "tool", "name": "search", "content": "r"} for _ in range(5)
    ]
    assert build_task_units(msgs)[0].tool_names == ["search"]


# ── 修正 A：跨轮身份用 unit_key，位置索引会漂移 ──


def test_unit_key_stable_across_history_compaction():
    """agent 压缩历史 → 位置索引漂移，但 unit_key 不变（ADR-0024 T0 修正 A）。

    这是 prefix_changed 占真实流量 19–28% 时归属仍然稳定的依据。
    """
    full = _msgs_tool_loop()
    # 模拟 agent 自行压缩：把第一个单元换成一条摘要 system 消息
    compacted = [
        {"role": "system", "content": "You are helpful."},
        {"role": "system", "content": "[摘要] 之前问过北京天气"},
    ] + full[7:]

    u_full = build_task_units(full)
    u_comp = build_task_units(compacted)

    # 位置索引漂移了
    assert u_full[1].msg_start == 7
    assert u_comp[0].msg_start == 2
    assert u_full[1].unit_index == 1 and u_comp[0].unit_index == 0
    # 但身份键不变 —— 跨轮匹配的唯一依据
    assert u_full[1].unit_key == u_comp[0].unit_key


def test_unit_key_is_content_derived():
    assert make_unit_key("abc") == make_unit_key("abc")
    assert make_unit_key("abc") != make_unit_key("abd")
    assert len(make_unit_key("abc")) == 16


def test_unit_key_handles_multimodal_content():
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "看这张图"},
        {"type": "image_url", "image_url": {"url": "http://x/y.png"}},
    ]}]
    units = build_task_units(msgs)
    assert units[0].unit_key == make_unit_key("看这张图")
    assert units[0].intent_chars == len("看这张图")


# ── 修正 D：只存引用不存正文 ──


def test_no_raw_text_persisted():
    """claude-code 的 user 消息 p50 = 70K 字符——正文绝不能进 schema。"""
    msgs = [
        {"role": "user", "content": "长" * 70000},
        {"role": "assistant", "content": "结论" * 5000},
    ]
    dumped = build_task_units(msgs)[0].model_dump()
    blob = repr(dumped)
    assert "长长长" not in blob
    assert "结论结论" not in blob
    assert len(blob) < 1000            # 单元本身必须是常数级
    assert dumped["intent_chars"] == 70000     # 但统计量保留
    assert dumped["intent_index"] == 0         # 正文按索引回查


def test_indices_not_persisted_but_recomputable():
    msgs = _msgs_tool_loop()
    u = build_task_units(msgs)[0]
    assert "indices" not in u.model_dump()
    assert unit_indices(u, msgs) == [1, 2, 3, 4, 5, 6]


def test_unit_indices_tolerates_truncated_messages():
    """msg_end 超出列表长度时不越界（防御历史数据 / 截断场景）。"""
    msgs = _msgs_tool_loop()
    u = build_task_units(msgs)[1]
    assert unit_indices(u, msgs[:8]) == [7]


# ── 协议边界歧义（T0 实测 2/4251，兜底路径）──


def test_anthropic_tool_result_in_user_lowers_confidence():
    msgs = [
        {"role": "user", "content": "q"},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "c1", "content": "r"}]},
    ]
    units = build_task_units(msgs)
    assert units[0].boundary_confidence == 1.0
    assert units[1].boundary_confidence < 1.0


def test_normal_traffic_keeps_full_confidence():
    for u in build_task_units(_msgs_tool_loop()):
        assert u.boundary_confidence == 1.0


# ── 与退役实现的等价性（ADR-0024 T1 验收）──


def _legacy_infer_turn_metadata(messages):
    """退役前 `server._infer_turn_metadata` 的逐字副本，仅供等价性对照。"""
    user_count = 0
    tools_after_last_user = 0
    seen_last_user = False
    for msg in messages:
        role = msg.get("role", "")
        if role == "user":
            user_count += 1
            seen_last_user = True
            tools_after_last_user = 0
        elif role == "tool" and seen_last_user:
            tools_after_last_user += 1
    return user_count, tools_after_last_user


@pytest.mark.parametrize("msgs", [
    [],
    [{"role": "system", "content": "s"}],
    [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}],
    _msgs_tool_loop(),
    _msgs_tool_loop() + [{"role": "user", "content": "third"}],
    [{"role": "user", "content": f"u{i}"} for i in range(5)],
    [{"role": "tool", "name": "t", "content": "orphan"},
     {"role": "user", "content": "hi"},
     {"role": "tool", "name": "t", "content": "r"}],
])
def test_derive_turn_metadata_equivalent_to_retired_impl(msgs):
    """新旧实现在 (logical_turn, roundtrip) 上逐字等价 —— 退役的前提条件。"""
    assert derive_turn_metadata(build_task_units(msgs)) == _legacy_infer_turn_metadata(msgs)


# ── 与 assembly._split_units 的一致性（单一切分算法）──


def test_assembly_split_units_delegates_to_core():
    """assembly 不再自带切分算法，其 _Unit 视图必须与 TaskUnit 逐条一致。"""
    from bladex_proxy.assembly import AssemblyConfig, ContextAssembler

    msgs = _msgs_tool_loop()
    asm = ContextAssembler(AssemblyConfig())
    legacy_view = asm._split_units(msgs)
    core_units = build_task_units(msgs)

    assert len(legacy_view) == len(core_units)
    for lv, cu in zip(legacy_view, core_units, strict=True):
        assert lv.indices == unit_indices(cu, msgs)
        assert lv.closed == cu.closed


# ── 确定性（prompt cache / 重建等价性的前提）──


def test_build_is_pure_function():
    msgs = _msgs_tool_loop()
    a = [u.model_dump() for u in build_task_units(msgs)]
    b = [u.model_dump() for u in build_task_units(msgs)]
    assert a == b
