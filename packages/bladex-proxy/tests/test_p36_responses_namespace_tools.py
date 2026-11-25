"""MQ-P36：codex 的 `namespace` 工具族经 Responses→chat 转换后被压扁（批 P36，2026-10-28）。

## 这张表守的是什么

codex 把「子代理」（`multi_agent_v1`）与 MCP 工具按
`{"type":"namespace","name":<族名>,"tools":[内层…]}` 整块下发。修前
`responses.py` 只认「有 `name` 或 `type==function`」⇒ namespace 被当成**一个空参数
的普通函数**发给上游，内层工具全丢。模型只能调 `multi_agent_v1({})`，codex 回
`unsupported call: multi_agent_v1`，连试两次放弃（BA0 X1，10-27 23:46，
`raw_request_ref=93a12141881bfa3c`）。

## 夹具来源

`HUB_TOOLS` 逐字取自 Hub `__msg__/93a12141881bfa3c` 的 `tools`（只删了长描述，
`type` / `name` / `tools` 层级与 `parameters` schema 结构不动）。live 里
`mcp__cua_repl` 与 `mcp__node_repl` **都有** `js` / `js_reset` —— 这就是"重名不得
串"那条用例的真实来源，不是编的。

## 判据

每一条都对着提示词 §步骤 3 的表格一行；反向变异（去掉展开 / 不还原 / 历史不转换 /
映射丢掉族名 / 去掉未知 type 告警）必须各自打红。
"""

from __future__ import annotations

import json

import pytest
import structlog.testing
from bladex_proxy.agency.streams import _synth_responses_tool_call_events
from bladex_proxy.capture import CaptureResult
from bladex_proxy.responses import (
    MAX_UPSTREAM_TOOL_NAME,
    format_responses_response,
    parse_responses_request,
    responses_call_item_name_fields,
    responses_stream_generator,
    responses_tool_name,
    split_responses_tool_name,
)

# ── 夹具：Hub 原文的工具表（结构不动，描述截短）─────────────────────────────

_MULTI_AGENT_TOOLS = [
    {"type": "function", "name": "close_agent",
     "description": "Close an agent.",
     "parameters": {"type": "object", "properties": {
         "target": {"type": "string", "description": "Agent id."}}, "required": ["target"]}},
    {"type": "function", "name": "resume_agent",
     "description": "Resume an agent.",
     "parameters": {"type": "object", "properties": {
         "id": {"type": "string"}}, "required": ["id"]}},
    {"type": "function", "name": "send_input",
     "description": "Send input.",
     "parameters": {"type": "object", "properties": {
         "interrupt": {"type": "boolean"}, "input": {"type": "string"}}}},
    {"type": "function", "name": "spawn_agent",
     "description": "Spawn an agent.",
     "parameters": {"type": "object", "properties": {
         "fork_context": {"type": "boolean"},
         "goal": {"type": "string"},
         "subagentPrompt": {"type": "string"}},
         "required": ["goal", "subagentPrompt"]}},
    {"type": "function", "name": "wait_agent",
     "description": "Wait for agents.",
     "parameters": {"type": "object", "properties": {
         "targets": {"type": "array", "items": {"type": "string"}}}}},
]

#: Hub 原文里 `exec_command` 的参数 schema（普通 function 的"逐字相同"对照组）。
_EXEC_COMMAND_TOOL = {
    "type": "function", "name": "exec_command",
    "description": "Runs a command.",
    "parameters": {"type": "object", "properties": {
        "cmd": {"type": "string"}, "workdir": {"type": "string"},
        "yield_time_ms": {"type": "number"}}, "required": ["cmd"]},
}

#: BladeX 自己注入的工具面形态（`_apply_agency_surfaces` 产出，chat 结构）。
_BLADEX_TOOL = {
    "type": "function",
    "function": {"name": "bladex_memory_search",
                 "description": "Search memory.",
                 "parameters": {"type": "object", "properties": {
                     "query": {"type": "string"}}, "required": ["query"]}},
}

HUB_TOOLS = [
    _EXEC_COMMAND_TOOL,
    {"type": "function", "name": "view_image",
     "description": "View an image.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}},
    {"type": "namespace", "name": "multi_agent_v1",
     "description": "Tools for spawning and managing sub-agents.",
     "tools": _MULTI_AGENT_TOOLS},
    {"type": "namespace", "name": "mcp__cua_repl",
     "description": "UI automation through cua_repl using the initialized cua API",
     "tools": [
         {"type": "function", "name": "js",
          "parameters": {"type": "object", "properties": {"code": {"type": "string"}}}},
         {"type": "function", "name": "js_reset",
          "parameters": {"type": "object", "additionalProperties": False}},
     ]},
    {"type": "namespace", "name": "mcp__node_repl",
     "description": "Use `js` for `node_repl` execution with persistent state.",
     "tools": [
         {"type": "function", "name": "js",
          "parameters": {"type": "object", "properties": {"code": {"type": "string"}}}},
         {"type": "function", "name": "js_add_node_module_dir",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}},
         {"type": "function", "name": "js_reset",
          "parameters": {"type": "object", "additionalProperties": False}},
     ]},
    _BLADEX_TOOL,
]


def _upstream_tools(tools=None) -> list[dict]:
    _msgs, extra = parse_responses_request({"tools": tools if tools is not None else HUB_TOOLS,
                                            "input": "hi"})
    return extra.get("tools") or []


def _tool_by_name(tools: list[dict], name: str) -> dict:
    for t in tools:
        if t.get("function", {}).get("name") == name:
            return t
    raise AssertionError(f"上游工具表里没有 {name}：{[t['function']['name'] for t in tools]}")


# ── ① 请求侧：namespace 展开成内层函数（修前整块被压成一个空参数函数）────────


def test_request_side_expands_multi_agent_namespace() -> None:
    tools = _upstream_tools()
    names = [t["function"]["name"] for t in tools]
    # 空的 multi_agent_v1 不再存在（修前它就长这样：parameters={"type":"object"}）
    assert "multi_agent_v1" not in names, "namespace 仍被当成普通函数下发"
    for inner in _MULTI_AGENT_TOOLS:
        upstream = responses_tool_name(inner["name"], "multi_agent_v1")
        assert upstream in names, f"内层工具 {inner['name']} 没有展开"
        # 参数 schema 与原文逐字相同
        assert _tool_by_name(tools, upstream)["function"]["parameters"] == inner["parameters"]


def test_request_side_all_three_namespaces_expand() -> None:
    names = [t["function"]["name"] for t in _upstream_tools()]
    assert len(names) == 2 + 5 + 2 + 3 + 1  # exec_command/view_image + 三个 namespace + bladex
    assert "ns-mcp__cua_repl-js" in names
    assert "ns-mcp__node_repl-js" in names
    assert "ns-mcp__node_repl-js_add_node_module_dir" in names


def test_namespace_tool_description_preserved() -> None:
    tools = _upstream_tools()
    spawn = _tool_by_name(tools, "ns-multi_agent_v1-spawn_agent")
    assert spawn["function"]["description"] == "Spawn an agent."


# ── ② 响应侧（非流式）：上游名还原成 codex 认的 name + namespace ────────────


class _FakeFunction:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, call_id: str, name: str, arguments: str) -> None:
        self.id = call_id
        self.function = _FakeFunction(name, arguments)


class _FakeMessage:
    def __init__(self, tool_calls=None, content=None) -> None:
        self.content = content
        self.tool_calls = tool_calls or []
        self.reasoning_content = None


class _FakeChoice:
    def __init__(self, message: _FakeMessage, finish_reason: str = "tool_calls") -> None:
        self.message = message
        self.finish_reason = finish_reason


class _FakeUsage:
    prompt_tokens = 10
    completion_tokens = 5
    total_tokens = 15


class _FakeResponse:
    def __init__(self, message: _FakeMessage) -> None:
        self.choices = [_FakeChoice(message)]
        self.usage = _FakeUsage()


def test_response_side_nonstream_restores_namespace() -> None:
    upstream = responses_tool_name("spawn_agent", "multi_agent_v1")
    resp = format_responses_response(
        _FakeResponse(_FakeMessage([_FakeToolCall("call_1", upstream, '{"goal":"x"}')])),
        "gpt-5-codex")
    items = [i for i in resp["output"] if i["type"] == "function_call"]
    assert len(items) == 1
    item = items[0]
    # codex 侧：ResponseItem::FunctionCall{ name, namespace }（core/src/tools/router.rs:263
    # 用 ToolName::new(namespace, name) 查表）
    assert item["name"] == "spawn_agent"
    assert item["namespace"] == "multi_agent_v1"
    assert item["arguments"] == '{"goal":"x"}'
    assert item["call_id"] == "call_1"


def test_response_side_plain_function_has_no_namespace_key() -> None:
    """普通函数：修前逐字相同 ⇒ 不许凭空多出 `namespace` 键。"""
    resp = format_responses_response(
        _FakeResponse(_FakeMessage([_FakeToolCall("call_2", "exec_command", "{}")])),
        "gpt-5-codex")
    item = [i for i in resp["output"] if i["type"] == "function_call"][0]
    assert item == {"type": "function_call", "id": item["id"], "call_id": "call_2",
                    "name": "exec_command", "arguments": "{}", "status": "completed"}


# ── ③ 响应侧（流式）：逐 chunk 拼出的 item 与非流式逐字相同 ─────────────────


class _FakeDelta:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls
        self.reasoning_content = None


class _FakeStreamChoice:
    def __init__(self, delta, finish_reason=None):
        self.delta = delta
        self.finish_reason = finish_reason


class _FakeChunk:
    def __init__(self, delta, finish_reason=None, usage=None):
        self.choices = [_FakeStreamChoice(delta, finish_reason)]
        self.usage = usage


class _FakeToolCallDelta:
    def __init__(self, index: int, call_id: str = "", name: str = "", arguments: str = ""):
        self.index = index
        self.id = call_id
        self.function = _FakeFunction(name, arguments)


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        self._iter = iter(self._chunks)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration from None


async def _stream_events(chunks) -> list[dict]:
    result = CaptureResult()
    out: list[dict] = []
    async for raw in responses_stream_generator(_FakeStream(chunks), result, "gpt-5-codex"):
        text = raw.decode()
        etype = ""
        payload = {}
        for line in text.splitlines():
            if line.startswith("event: "):
                etype = line[len("event: "):]
            elif line.startswith("data: "):
                payload = json.loads(line[len("data: "):])
        out.append({**payload, "_event": etype})
    return out


def _completed_item(events: list[dict], item_type: str = "function_call") -> dict:
    done = [e for e in events if e["_event"] == "response.completed"]
    assert done, "没有 response.completed"
    return [i for i in done[-1]["response"]["output"] if i["type"] == item_type][0]


@pytest.mark.asyncio
async def test_response_side_stream_restores_namespace() -> None:
    upstream = responses_tool_name("spawn_agent", "multi_agent_v1")
    events = await _stream_events([
        _FakeChunk(_FakeDelta(tool_calls=[
            _FakeToolCallDelta(0, call_id="call_1", name=upstream,
                               arguments='{"goal":"x"}')])),
        _FakeChunk(_FakeDelta(), finish_reason="tool_calls"),
    ])
    # 流里的 item.added / item.done / 终止快照三处都必须是还原后的形态
    added = [e for e in events if e["_event"] == "response.output_item.added"
             and e["item"].get("type") == "function_call"]
    done = [e for e in events if e["_event"] == "response.output_item.done"
            and e["item"].get("type") == "function_call"]
    assert added and done
    for item in (added[0]["item"], done[0]["item"], _completed_item(events)):
        assert item["name"] == "spawn_agent", item
        assert item["namespace"] == "multi_agent_v1", item


@pytest.mark.asyncio
async def test_stream_and_nonstream_items_identical() -> None:
    """同一上游调用，流式拼出的 item 与非流式写回的 item 逐字相同（除 id）。"""
    upstream = responses_tool_name("spawn_agent", "multi_agent_v1")
    args = '{"goal":"x"}'
    nonstream = [i for i in format_responses_response(
        _FakeResponse(_FakeMessage([_FakeToolCall("call_1", upstream, args)])),
        "gpt-5-codex")["output"] if i["type"] == "function_call"][0]
    events = await _stream_events([
        _FakeChunk(_FakeDelta(tool_calls=[
            _FakeToolCallDelta(0, call_id="call_1", name=upstream, arguments=args)])),
        _FakeChunk(_FakeDelta(), finish_reason="tool_calls"),
    ])
    streamed = _completed_item(events)

    def _cmp(item: dict) -> dict:
        return {k: item[k] for k in ("type", "call_id", "name", "namespace",
                                     "arguments", "status")}

    assert _cmp(streamed) == _cmp(nonstream)


# ── ④ 历史侧：下一请求的 input 里带 namespace 的 function_call ───────────────


def test_history_side_converts_function_call_item() -> None:
    body = {
        "input": [
            {"type": "message", "role": "user",
             "content": [{"type": "input_text", "text": "派个子代理"}]},
            # codex 回放上一轮的调用（它记的是我们还原后的形态）
            {"type": "function_call", "name": "spawn_agent", "namespace": "multi_agent_v1",
             "arguments": '{"goal":"x"}', "call_id": "call_1"},
            {"type": "function_call_output", "call_id": "call_1", "output": "ok"},
        ],
    }
    messages, _extra = parse_responses_request(body)
    assistant = [m for m in messages if m.get("tool_calls")]
    assert len(assistant) == 1
    name = assistant[0]["tool_calls"][0]["function"]["name"]
    # 与请求侧工具表里的名字同一套（否则上游看到的历史调用名对不上工具表）
    assert name == responses_tool_name("spawn_agent", "multi_agent_v1")
    assert name in [t["function"]["name"] for t in _upstream_tools()]


def test_history_side_converts_function_call_block_in_assistant_message() -> None:
    body = {
        "input": [
            {"type": "message", "role": "assistant", "content": [
                {"type": "function_call", "name": "js_reset", "namespace": "mcp__node_repl",
                 "arguments": "{}", "call_id": "call_9"},
            ]},
        ],
    }
    messages, _extra = parse_responses_request(body)
    assistant = [m for m in messages if m.get("tool_calls")][0]
    assert assistant["tool_calls"][0]["function"]["name"] == "ns-mcp__node_repl-js_reset"


def test_history_side_plain_function_call_untouched() -> None:
    body = {"input": [{"type": "function_call", "name": "exec_command",
                       "arguments": "{}", "call_id": "c3"}]}
    messages, _extra = parse_responses_request(body)
    assistant = [m for m in messages if m.get("tool_calls")][0]
    assert assistant["tool_calls"][0]["function"]["name"] == "exec_command"


# ── ⑤ 重名：两个 namespace 都有 `js` / `js_reset`（live 原文）────────────────


def test_same_inner_name_across_namespaces_does_not_cross() -> None:
    names = [t["function"]["name"] for t in _upstream_tools()]
    assert "ns-mcp__cua_repl-js" in names and "ns-mcp__node_repl-js" in names
    assert split_responses_tool_name("ns-mcp__cua_repl-js") == ("mcp__cua_repl", "js")
    assert split_responses_tool_name("ns-mcp__node_repl-js") == ("mcp__node_repl", "js")
    assert split_responses_tool_name("ns-mcp__cua_repl-js_reset") == ("mcp__cua_repl", "js_reset")
    assert split_responses_tool_name("ns-mcp__node_repl-js_reset") == ("mcp__node_repl", "js_reset")

    resp = format_responses_response(
        _FakeResponse(_FakeMessage([
            _FakeToolCall("call_a", "ns-mcp__cua_repl-js", "{}"),
            _FakeToolCall("call_b", "ns-mcp__node_repl-js", "{}"),
        ])), "gpt-5-codex")
    got = [(i["call_id"], i["namespace"], i["name"])
           for i in resp["output"] if i["type"] == "function_call"]
    assert got == [("call_a", "mcp__cua_repl", "js"), ("call_b", "mcp__node_repl", "js")]


def test_mapping_roundtrip_is_injective() -> None:
    pairs = [("multi_agent_v1", "spawn_agent"), ("mcp__cua_repl", "js"),
             ("mcp__node_repl", "js"), ("mcp__cua_repl", "js_reset"),
             ("mcp__node_repl", "js_reset"), ("functions", "apply_patch")]
    encoded = {responses_tool_name(n, ns): (ns, n) for ns, n in pairs}
    assert len(encoded) == len(pairs)
    for up, pair in encoded.items():
        assert split_responses_tool_name(up) == pair


# ── ⑥ 未知 type：不再静默压成空参数函数 ────────────────────────────────────


def test_unknown_tool_type_warns_once_and_is_not_forwarded() -> None:
    # 批 BC（MQ-P37）起 `custom` 不再是「未知 type」（转函数下发，见 test_bc_custom_tool_mapping.py）。
    tools = [{"type": "web_search", "name": "web_search"}]
    with structlog.testing.capture_logs() as cap:
        first = _upstream_tools(tools)
        second = _upstream_tools(tools)
    assert first == [] and second == []
    events = [e for e in cap if e["event"] == "responses_tool_type_unsupported"]
    assert len(events) == 1, f"未知 type 告警没有按 type 去重限频：{events}"
    assert events[0]["tool_type"] == "web_search"
    assert events[0]["tool_name"] == "web_search"


def test_unknown_inner_type_warns() -> None:
    tools = [{"type": "namespace", "name": "multi_agent_v1", "tools": [
        {"type": "web_search", "name": "search"}]}]
    with structlog.testing.capture_logs() as cap:
        upstream = _upstream_tools(tools)
    assert upstream == []
    events = [e for e in cap if e["event"] == "responses_tool_type_unsupported"]
    assert len(events) == 1
    assert events[0]["tool_type"] == "namespace:web_search"
    assert events[0]["tool_name"] == "multi_agent_v1.search"


# ── ⑦ 普通 function 与 `bladex_*`：转换结果与修前逐字相同 ──────────────────


def test_plain_function_and_bladex_tool_identical_to_pre_fix() -> None:
    tools = _upstream_tools()
    assert _tool_by_name(tools, "exec_command") == {
        "type": "function",
        "function": {"name": "exec_command", "description": "Runs a command.",
                     "parameters": _EXEC_COMMAND_TOOL["parameters"]},
    }
    # bladex_* 是端点里以 chat 结构追加的（`_apply_agency_surfaces`）⇒ 原样透传
    assert _BLADEX_TOOL in tools


def test_plain_function_with_strict_preserved() -> None:
    tool = {"type": "function", "name": "f", "strict": True,
            "parameters": {"type": "object"}}
    assert _tool_by_name(_upstream_tools([tool]), "f") == {
        "type": "function",
        "function": {"name": "f", "description": "", "parameters": {"type": "object"},
                     "strict": True},
    }


def test_default_parameters_unchanged() -> None:
    tool = {"type": "function", "name": "g"}
    assert _tool_by_name(_upstream_tools([tool]), "g")["function"]["parameters"] == {
        "type": "object"}


# ── ⑧ 名字长度：合规才下发（超长丢工具 + 告警，不截断）────────────────────


def test_overlong_namespaced_name_dropped_with_warning() -> None:
    long_inner = "x" * (MAX_UPSTREAM_TOOL_NAME - len("ns-mcp__node_repl-") + 1)
    tools = [{"type": "namespace", "name": "mcp__node_repl",
              "tools": [{"type": "function", "name": long_inner,
                         "parameters": {"type": "object"}}]}]
    with structlog.testing.capture_logs() as cap:
        upstream = _upstream_tools(tools)
    assert upstream == []
    assert any(e["event"] == "responses_tool_name_too_long" for e in cap)


def test_all_forwarded_names_within_limit() -> None:
    names = [t["function"]["name"] for t in _upstream_tools()]
    assert all(len(n) <= MAX_UPSTREAM_TOOL_NAME for n in names), names


# ── ⑨ 第三条回写路径：内循环合成（BLADEX_MODULE_INTERCEPTION=1 时是主路径）───


def test_synth_path_restores_namespace() -> None:
    """X1 live 第一次调用就是从这条路径出去的（item id `fc_bladex_0`）。"""
    events = list(_synth_responses_tool_call_events([{
        "id": "call_1", "type": "function",
        "function": {"name": responses_tool_name("spawn_agent", "multi_agent_v1"),
                     "arguments": '{"goal":"x"}'},
    }]))
    assert len(events) == 4  # item.added + arguments.delta + arguments.done + item.done
    raw = events[-1].decode()
    payload = json.loads(raw.split("data: ", 1)[1])
    assert payload["item"]["name"] == "spawn_agent"
    assert payload["item"]["namespace"] == "multi_agent_v1"


def test_name_fields_helper_is_the_single_restore_point() -> None:
    """还原点只有一个：三处回写都调 `responses_call_item_name_fields`。"""
    assert responses_call_item_name_fields("ns-mcp__cua_repl-js") == {
        "name": "js", "namespace": "mcp__cua_repl"}
    assert responses_call_item_name_fields("exec_command") == {"name": "exec_command"}
