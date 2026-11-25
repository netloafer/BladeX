"""批 BC §2.3（MQ-P37）：`custom`（freeform）工具 ⇄ chat function 三向映射。

codex 的 `apply_patch` handler 只吃 `ToolPayload::Custom`，收到 `function_call` 形状就没有输出，
下一轮补 `aborted`（结算稿 §1.4）。判据：

- 声明：上游看到单字符串参数 `input` 的函数，名字走 P36 同一套，文法进 description；
- 响应：流式 added / done / 快照、非流式、内循环合成四条路径都还原成 `custom_tool_call`，`input` 逐字；
- 历史：回放的 `custom_tool_call` + output 配对成 assistant `tool_calls` + tool 消息；
- 阴性：无 custom 的请求转换结果不变；漏传集合退回 function_call。

端到端（真 LiteLLM + 假上游，含内循环与拦截混合轮）在 `test_bc_fake_upstream_e2e.py`。
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
import structlog.testing
from bladex_proxy.agency.streams import _synth_responses_tool_call_events
from bladex_proxy.capture import CaptureResult
from bladex_proxy.responses import (
    MAX_UPSTREAM_TOOL_NAME,
    custom_input_from_arguments,
    format_responses_response,
    parse_responses_request,
    responses_call_item_fields,
    responses_custom_tool_names,
    responses_stream_generator,
    responses_tool_name,
    tool_events_from_output,
)

PATCH = "*** Begin Patch\n*** Add File: a.txt\n+你好 \"引号\"\n*** End Patch\n"
GRAMMAR = 'start: begin_patch hunk+ end_patch\nbegin_patch: "*** Begin Patch" LF\n'
APPLY_PATCH = {"type": "custom", "name": "apply_patch", "description": "Edit files.",
               "format": {"type": "grammar", "syntax": "lark", "definition": GRAMMAR}}
PLAIN = {"type": "function", "name": "exec_command", "description": "Run.",
         "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}}}
NAMESPACED = {"type": "namespace", "name": "editors", "tools": [
    {"type": "custom", "name": "patch_it", "description": "ns custom."},
    {"type": "function", "name": "lint", "parameters": {"type": "object"}},
]}
#: DeepSeek 对单字符串参数工具的真实尾巴（结算稿 §2.3，3 / 13）：JSON 没闭合。
DSML_CLOSE = "</｜DSML｜invoke>\n</｜DSML｜tool_calls>"


def _upstream_tools(*tools: dict) -> list[dict]:
    return parse_responses_request({"tools": list(tools), "input": "hi"})[1]["tools"]


def _delta(content: str | None = None, tool_calls: list | None = None) -> SimpleNamespace:
    return SimpleNamespace(content=content, reasoning_content=None, tool_calls=tool_calls)


def _chunk(delta: SimpleNamespace, finish_reason: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)],
                           usage=None)


def _tool_delta(call_id: str | None, name: str | None, arguments: str,
                index: int = 0) -> SimpleNamespace:
    return SimpleNamespace(index=index, id=call_id,
                           function=SimpleNamespace(name=name, arguments=arguments))


class _Stream:
    def __init__(self, chunks: list[SimpleNamespace]) -> None:
        self._chunks = list(chunks)

    def __aiter__(self) -> _Stream:
        return self

    async def __anext__(self) -> SimpleNamespace:
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


async def _events(chunks: list[SimpleNamespace], custom_names: frozenset[str]) -> tuple[list[dict], CaptureResult]:
    result = CaptureResult()
    out: list[dict] = []
    async for raw in responses_stream_generator(_Stream(chunks), result, "m",
                                                custom_names=custom_names):
        out.append(json.loads(raw.decode().splitlines()[1][len("data: "):]))
    return out, result


_NAMES = frozenset({"apply_patch"})


def _patch_chunks(args: str | None = None, call_id: str | None = "call_p") -> list[SimpleNamespace]:
    args = json.dumps({"input": PATCH}) if args is None else args
    mid = len(args) // 2
    return [_chunk(_delta(tool_calls=[_tool_delta(call_id, "apply_patch", args[:mid])])),
            _chunk(_delta(tool_calls=[_tool_delta(None, None, args[mid:])])),
            _chunk(_delta(), "tool_calls")]


# ── ① 声明侧 ─────────────────────────────────────────────────────────────────


def test_declaration_becomes_single_string_function() -> None:
    fn = _upstream_tools(APPLY_PATCH)[0]["function"]
    assert fn["name"] == "apply_patch", "模型可见名不变，不新增前缀"
    params = fn["parameters"]
    assert params["properties"]["input"]["type"] == "string"
    assert params["required"] == ["input"]
    assert params["additionalProperties"] is False
    assert list(params["properties"]) == ["input"]
    assert fn["description"].startswith("Edit files.")
    assert GRAMMAR in fn["description"], "chat 没有文法字段，模型只能靠这段文字"
    assert "lark" in fn["description"]


def test_declaration_without_format_keeps_description() -> None:
    fn = _upstream_tools({"type": "custom", "name": "x", "description": "d"})[0]["function"]
    assert fn["description"] == "d"


def test_namespace_inner_custom_uses_p36_name() -> None:
    tools = _upstream_tools(NAMESPACED)
    by_name = {t["function"]["name"]: t["function"] for t in tools}
    assert set(by_name) == {"ns-editors-patch_it", "ns-editors-lint"}
    assert by_name["ns-editors-patch_it"]["parameters"]["required"] == ["input"]
    assert by_name["ns-editors-lint"]["parameters"] == {"type": "object"}


def test_custom_name_set_is_the_single_source() -> None:
    assert responses_custom_tool_names(
        {"tools": [APPLY_PATCH, PLAIN, NAMESPACED, "junk",
                   {"type": "custom"}]}) == frozenset({"apply_patch", "ns-editors-patch_it"})
    assert responses_custom_tool_names({}) == frozenset()
    assert responses_custom_tool_names({"tools": "x"}) == frozenset()


def test_plain_tools_unchanged_when_custom_present() -> None:
    assert _upstream_tools(PLAIN, APPLY_PATCH)[0] == {
        "type": "function",
        "function": {"name": "exec_command", "description": "Run.",
                     "parameters": PLAIN["parameters"]}}


def test_no_custom_request_converts_byte_identical() -> None:
    """回放判据的最小常驻版（全量读数见结算稿 §6）：无 custom、无 reasoning 的请求逐字不变。"""
    body = {"model": "m", "instructions": "sys", "input": [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
        {"type": "function_call", "name": "exec_command", "arguments": "{}", "call_id": "c1"},
        {"type": "function_call_output", "call_id": "c1", "output": "ok"},
    ], "tools": [PLAIN], "tool_choice": "auto", "parallel_tool_calls": True, "max_output_tokens": 9}
    messages, extra = parse_responses_request(body)
    assert messages == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "c1", "type": "function",
            "function": {"name": "exec_command", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
    ]
    assert extra == {"max_tokens": 9, "parallel_tool_calls": True, "tool_choice": "auto",
                     "tools": [{"type": "function", "function": {
                         "name": "exec_command", "description": "Run.",
                         "parameters": PLAIN["parameters"]}}]}


@pytest.mark.parametrize("ttype", ["web_search", "tool_search", "mcp"])
def test_server_side_tools_still_not_sent(ttype: str) -> None:
    from bladex_proxy import responses as mod
    mod._unsupported_tool_type_at.clear()
    with structlog.testing.capture_logs() as logs:
        _, extra = parse_responses_request({"input": "hi", "tools": [{"type": ttype, "name": "w"}]})
    assert "tools" not in extra
    assert [x["tool_type"] for x in logs if x["event"] == "responses_tool_type_unsupported"] == [ttype]


def test_custom_colliding_with_function_is_dropped_with_warning() -> None:
    with structlog.testing.capture_logs() as logs:
        tools = _upstream_tools({"type": "function", "name": "apply_patch"}, APPLY_PATCH)
    assert len(tools) == 1
    assert [x["source"] for x in logs if x["event"] == "responses_tool_name_collision"] == ["custom"]


# ── ② P38：普通函数名超长只告警，照发 ─────────────────────────────────────────


def test_p38_long_plain_name_warns_and_sends() -> None:
    name = "x" * (MAX_UPSTREAM_TOOL_NAME + 1)
    with structlog.testing.capture_logs() as logs:
        tools = _upstream_tools({"type": "function", "name": name})
    assert tools[0]["function"]["name"] == name
    hits = [x for x in logs if x["event"] == "responses_tool_name_too_long"]
    assert len(hits) == 1 and hits[0]["source"] == "function"


def test_p38_name_at_limit_is_silent() -> None:
    with structlog.testing.capture_logs() as logs:
        _upstream_tools({"type": "function", "name": "x" * MAX_UPSTREAM_TOOL_NAME})
    assert not [x for x in logs if x["event"] == "responses_tool_name_too_long"]


# ── ③ 参数解包（四种形态，后三种告警）─────────────────────────────────────────


def _unwrap(arguments: str) -> tuple[str, list[str]]:
    with structlog.testing.capture_logs() as logs:
        value = custom_input_from_arguments(arguments, "apply_patch")
    return value, [x["shape"] for x in logs if x["event"] == "responses_custom_args_unwrapped"]


def test_unwrap_input_key_is_silent() -> None:
    assert _unwrap(json.dumps({"input": PATCH})) == (PATCH, [])


def test_unwrap_literal_control_chars_in_json() -> None:
    assert _unwrap('{"input":"a\nb"}') == ("a\nb", [])


def test_unwrap_input_with_extra_keys() -> None:
    assert _unwrap(json.dumps({"input": PATCH, "x": 1})) == (PATCH, ["input_with_extra_keys"])


def test_unwrap_single_string_key_cx2_shape() -> None:
    assert _unwrap(json.dumps({"patch": PATCH})) == (PATCH, ["single_string_key"])


def test_unwrap_dsml_tail_real_shape_escaped_newline() -> None:
    args = '{"input":"*** Begin Patch\\n*** End Patch\\n' + DSML_CLOSE
    assert _unwrap(args) == ("*** Begin Patch\n*** End Patch", ["dsml_tail"])


def test_unwrap_dsml_tail_raw_newline() -> None:
    args = '{"input":"*** Begin Patch\n*** End Patch\n' + DSML_CLOSE
    assert _unwrap(args) == ("*** Begin Patch\n*** End Patch", ["dsml_tail"])


def test_unwrap_dsml_tail_ascii_bars() -> None:
    assert _unwrap('{"input":"body</|DSML|invoke>') == ("body", ["dsml_tail"])


def test_unwrap_dsml_tail_does_not_eat_real_trailing_n() -> None:
    """正文本身以字母 n 结尾：只有反斜杠 + n 才是转义换行，普通的 n 不是尾巴的一部分。"""
    assert _unwrap('{"input":"pattern' + DSML_CLOSE) == ("pattern", ["dsml_tail"])


def test_unwrap_dsml_inside_valid_json_is_content() -> None:
    value = f"文档里提到 {DSML_CLOSE} 这个串"
    assert _unwrap(json.dumps({"input": value}, ensure_ascii=False)) == (value, [])


@pytest.mark.parametrize("args", ["不是 JSON", '{"a":"x","b":"y"}', '{"input":5}', "[1]",
                                  '{"input":"未闭合'])
def test_unwrap_anything_else_is_raw_with_warning(args: str) -> None:
    assert _unwrap(args) == (args, ["raw"])


# ── ④ 响应侧：唯一还原点 ─────────────────────────────────────────────────────


def test_item_fields_custom_vs_function() -> None:
    args = json.dumps({"input": PATCH})
    assert responses_call_item_fields("apply_patch", args, _NAMES) == {
        "type": "custom_tool_call", "name": "apply_patch", "input": PATCH}
    assert responses_call_item_fields("apply_patch", args) == {
        "type": "function_call", "name": "apply_patch", "arguments": args}
    assert responses_call_item_fields("ns-editors-patch_it", args,
                                      frozenset({"ns-editors-patch_it"})) == {
        "type": "custom_tool_call", "name": "patch_it", "namespace": "editors", "input": PATCH}


def test_item_fields_open_item_does_not_unwrap_or_warn() -> None:
    with structlog.testing.capture_logs() as logs:
        assert responses_call_item_fields("apply_patch", None, _NAMES)["input"] == ""
    assert not logs


def _nonstream(tool_calls: list[SimpleNamespace], content: str | None = None) -> SimpleNamespace:
    message = SimpleNamespace(content=content, reasoning_content=None, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
                           usage=None)


def _tc(call_id: str, name: str, arguments: str) -> SimpleNamespace:
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


def test_nonstream_custom_item_and_tool_event() -> None:
    args = json.dumps({"input": PATCH})
    out = format_responses_response(_nonstream([_tc("call_p", "apply_patch", args)]), "m",
                                    custom_names=_NAMES)["output"]
    assert out == [{"type": "custom_tool_call", "id": out[0]["id"], "call_id": "call_p",
                    "name": "apply_patch", "input": PATCH, "status": "completed"}]
    events = tool_events_from_output(out)
    assert [(e.tool_name, e.arguments, e.tool_call_id, e.direction) for e in events] == [
        ("apply_patch", {"input": PATCH}, "call_p", "call")]


def test_nonstream_without_custom_set_falls_back_to_function_call() -> None:
    out = format_responses_response(_nonstream([_tc("c", "apply_patch", "{}")]), "m")["output"]
    assert out[0]["type"] == "function_call"


def test_tool_event_keeps_namespace_in_upstream_name() -> None:
    args = json.dumps({"input": "x"})
    out = format_responses_response(
        _nonstream([_tc("c", "ns-editors-patch_it", args)]), "m",
        custom_names=frozenset({"ns-editors-patch_it"}))["output"]
    assert tool_events_from_output(out)[0].tool_name == "ns-editors-patch_it"


@pytest.mark.asyncio
async def test_stream_added_done_snapshot_all_custom() -> None:
    events, result = await _events(_patch_chunks(), _NAMES)
    added = [e["item"] for e in events if e["type"] == "response.output_item.added"]
    done = [e["item"] for e in events if e["type"] == "response.output_item.done"]
    snapshot = [e for e in events if e["type"] == "response.completed"][0]["response"]["output"]
    assert [i["type"] for i in added + done + snapshot] == ["custom_tool_call"] * 3
    assert added[0]["input"] == "" and added[0]["status"] == "in_progress"
    assert done[0]["input"] == PATCH and done[0]["status"] == "completed"
    assert snapshot == done, "终止快照要和 item.done 逐字一致，否则 codex 对账分叉"
    assert done[0]["call_id"] == added[0]["call_id"] == "call_p"
    assert "arguments" not in done[0]
    assert not [e for e in events if e["type"] == "response.function_call_arguments.delta"], (
        "custom 不发 function_call_arguments.delta")
    assert [(t.tool_name, t.arguments, t.tool_call_id) for t in result.tool_events] == [
        ("apply_patch", {"input": PATCH}, "call_p")]


@pytest.mark.asyncio
async def test_stream_unwrap_warns_exactly_once() -> None:
    with structlog.testing.capture_logs() as logs:
        await _events(_patch_chunks(json.dumps({"patch": PATCH})), _NAMES)
    assert [x["shape"] for x in logs if x["event"] == "responses_custom_args_unwrapped"] == [
        "single_string_key"], "item.done 和快照只算一次"


@pytest.mark.asyncio
async def test_stream_dsml_salvaged_into_input() -> None:
    args = '{"input":"*** Begin Patch\\n*** End Patch\\n' + DSML_CLOSE
    events, _ = await _events(_patch_chunks(args), _NAMES)
    done = [e["item"] for e in events if e["type"] == "response.output_item.done"]
    assert done[0]["input"] == "*** Begin Patch\n*** End Patch"


@pytest.mark.asyncio
async def test_stream_mints_call_id_when_upstream_gives_none() -> None:
    events, _ = await _events(_patch_chunks(call_id=None), _NAMES)
    added = [e["item"] for e in events if e["type"] == "response.output_item.added"][0]
    snapshot = [e for e in events if e["type"] == "response.completed"][0]["response"]["output"][0]
    assert added["call_id"].startswith("call_")
    assert snapshot["call_id"] == added["call_id"]


@pytest.mark.asyncio
async def test_stream_mixed_function_and_custom_indexes() -> None:
    events, _ = await _events([
        _chunk(_delta("先说一句")),
        _chunk(_delta(tool_calls=[_tool_delta("c0", "exec_command", '{"cmd":"ls"}', 0),
                                  _tool_delta("c1", "apply_patch",
                                              json.dumps({"input": PATCH}), 1)])),
        _chunk(_delta(), "tool_calls"),
    ], _NAMES)
    added = [(e["output_index"], e["item"]["type"]) for e in events
             if e["type"] == "response.output_item.added"]
    assert added == [(0, "message"), (1, "function_call"), (2, "custom_tool_call")]
    deltas = [e for e in events if e["type"] == "response.function_call_arguments.delta"]
    assert [e["output_index"] for e in deltas] == [1]


@pytest.mark.asyncio
async def test_stream_and_nonstream_custom_items_identical() -> None:
    args = json.dumps({"input": PATCH})
    events, _ = await _events(_patch_chunks(args), _NAMES)
    stream_item = [e for e in events if e["type"] == "response.completed"][0]["response"]["output"][0]
    plain_item = format_responses_response(_nonstream([_tc("call_p", "apply_patch", args)]), "m",
                                           custom_names=_NAMES)["output"][0]
    drop_id = lambda item: {k: v for k, v in item.items() if k != "id"}  # noqa: E731
    assert drop_id(stream_item) == drop_id(plain_item)


# ── ⑤ 第四条回写路径：内循环合成 ──────────────────────────────────────────────


def _parse_sse(chunks: list[bytes]) -> list[dict]:
    return [json.loads(c.decode().split("data: ", 1)[1]) for c in chunks]


def test_synth_custom_call_events_and_sink() -> None:
    sink: list[dict] = []
    args = json.dumps({"input": PATCH})
    events = _parse_sse(list(_synth_responses_tool_call_events(
        [{"id": "call_p", "type": "function",
          "function": {"name": "apply_patch", "arguments": args}}],
        index=3, sink=sink, custom_names=_NAMES)))
    assert [e["type"] for e in events] == ["response.output_item.added", "response.output_item.done"]
    assert [e["output_index"] for e in events] == [3, 3]
    assert events[0]["item"]["type"] == "custom_tool_call"
    assert events[0]["item"]["input"] == ""
    assert events[1]["item"]["input"] == PATCH
    assert sink == [events[1]["item"]], "快照落点与流里同一个 item"


def test_synth_namespace_custom_restored() -> None:
    args = json.dumps({"input": "x"})
    events = _parse_sse(list(_synth_responses_tool_call_events(
        [{"id": "c", "function": {"name": "ns-editors-patch_it", "arguments": args}}],
        custom_names=frozenset({"ns-editors-patch_it"}))))
    assert events[-1]["item"]["name"] == "patch_it"
    assert events[-1]["item"]["namespace"] == "editors"


def test_synth_function_call_shape_unchanged_by_custom_set() -> None:
    plain = list(_synth_responses_tool_call_events(
        [{"id": "c", "function": {"name": "shell", "arguments": "{}"}}]))
    with_set = list(_synth_responses_tool_call_events(
        [{"id": "c", "function": {"name": "shell", "arguments": "{}"}}], custom_names=_NAMES))
    assert plain == with_set
    assert len(plain) == 4


# ── ⑥ 历史侧 ─────────────────────────────────────────────────────────────────


def _replay(**call: Any) -> list[dict]:
    body = {"input": [
        {"type": "message", "role": "user", "content": "改文件"},
        {"type": "custom_tool_call", "call_id": "call_p", "name": "apply_patch",
         "input": PATCH, "status": "completed", **call},
        {"type": "custom_tool_call_output", "call_id": "call_p", "output": "Success"},
    ]}
    return parse_responses_request(body)[0]


def test_history_pairs_call_and_output() -> None:
    _, assistant, tool = _replay()
    assert assistant["content"] == "", "MQ-P17：空串不是 None"
    call = assistant["tool_calls"][0]
    assert call["id"] == "call_p" == tool["tool_call_id"]
    assert call["function"]["name"] == "apply_patch"
    assert json.loads(call["function"]["arguments"]) == {"input": PATCH}
    assert tool == {"role": "tool", "tool_call_id": "call_p", "content": "Success"}


def test_history_arguments_deterministic_across_replays() -> None:
    first = _replay()[1]["tool_calls"][0]["function"]["arguments"]
    assert _replay()[1]["tool_calls"][0]["function"]["arguments"] == first
    assert "你好" in first, "不转义非 ASCII，与上游原文同形"


def test_history_namespace_custom_uses_p36_name() -> None:
    call = _replay(name="patch_it", namespace="editors")[1]["tool_calls"][0]
    assert call["function"]["name"] == responses_tool_name("patch_it", "editors")


def test_history_structured_output_flattened() -> None:
    body = {"input": [{"type": "custom_tool_call_output", "call_id": "c",
                       "output": [{"type": "input_text", "text": "ok"}]}]}
    assert parse_responses_request(body)[0] == [
        {"role": "tool", "tool_call_id": "c", "content": "ok"}]


def test_history_not_reported_as_unknown() -> None:
    with structlog.testing.capture_logs() as logs:
        _replay()
    assert not [x for x in logs if x["event"] == "responses_unknown_input_item_type"]


def test_round_trip_response_to_next_request() -> None:
    """响应侧发出去的 item 原样回放，能回到上游的函数调用，且名字在同一张工具表里。"""
    args = json.dumps({"input": PATCH})
    item = format_responses_response(_nonstream([_tc("call_p", "apply_patch", args)]), "m",
                                     custom_names=_NAMES)["output"][0]
    messages, extra = parse_responses_request({"input": [item], "tools": [APPLY_PATCH]})
    call = messages[0]["tool_calls"][0]["function"]
    assert call["name"] == extra["tools"][0]["function"]["name"]
    assert custom_input_from_arguments(call["arguments"]) == PATCH
