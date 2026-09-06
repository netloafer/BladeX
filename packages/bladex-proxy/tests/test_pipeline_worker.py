"""Pipeline worker 单元测试 — P1 tool_results 回填（ADR-0011 P1）。"""


from bladex_proxy.models import Identity, ToolEvent, Turn, TurnStatus
from bladex_proxy.storage.pipeline_worker import enrich_tool_results


def _make_turn(
    request_messages: list[dict] | None = None,
    tool_events: list[ToolEvent] | None = None,
) -> Turn:
    return Turn(
        identity=Identity(user_id="u1", agent_id="a1", session_id="s1"),
        model="test",
        request_messages=request_messages or [],
        response_text="",
        status=TurnStatus.OK,
        tool_events=tool_events or [],
    )


def test_enrich_matches_tool_results_to_calls():
    """tool_call_id 匹配：role=tool 消息回填到对应 tool_event.result。"""
    turn = _make_turn(
        request_messages=[
            {"role": "user", "content": "search for news"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_abc", "type": "function", "function": {"name": "web_search", "arguments": '{"query": "news"}'}},
            ]},
            {"role": "tool", "tool_call_id": "call_abc", "content": "Found 3 news articles about tech"},
        ],
        tool_events=[
            ToolEvent(tool_name="web_search", arguments={"query": "news"}, direction="call", tool_call_id="call_abc"),
        ],
    )

    enrich_tool_results(turn)

    assert turn.tool_events[0].result == "Found 3 news articles about tech"
    assert turn.tool_events[0].direction == "call"
    assert len(turn.tool_events) == 1  # matched, no extra event


def test_enrich_unmatched_result_creates_separate_event():
    """未匹配到 call 的 result（属于上一轮）记录为 direction=result 事件。"""
    turn = _make_turn(
        request_messages=[
            {"role": "tool", "tool_call_id": "call_xyz", "content": "File written successfully"},
        ],
        tool_events=[
            ToolEvent(tool_name="web_search", arguments={"query": "news"}, direction="call", tool_call_id="call_abc"),
        ],
    )

    enrich_tool_results(turn)

    # Original call event unchanged
    assert turn.tool_events[0].result is None
    # New result event added
    assert len(turn.tool_events) == 2
    assert turn.tool_events[1].direction == "result"
    assert turn.tool_events[1].result == "File written successfully"
    assert turn.tool_events[1].tool_call_id == "call_xyz"


def test_enrich_no_tool_messages_noop():
    """没有 role=tool 消息时不做任何操作。"""
    turn = _make_turn(
        request_messages=[
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ],
        tool_events=[
            ToolEvent(tool_name="web_search", direction="call", tool_call_id="call_1"),
        ],
    )

    enrich_tool_results(turn)

    assert len(turn.tool_events) == 1
    assert turn.tool_events[0].result is None


def test_enrich_empty_messages_noop():
    """空 request_messages 不报错。"""
    turn = _make_turn(request_messages=[], tool_events=[])

    enrich_tool_results(turn)

    assert len(turn.tool_events) == 0


def test_enrich_multiple_tool_results():
    """多个工具返回同时匹配多个 call。"""
    turn = _make_turn(
        request_messages=[
            {"role": "assistant", "tool_calls": [
                {"id": "call_1", "function": {"name": "read_file", "arguments": "{}"}},
                {"id": "call_2", "function": {"name": "web_search", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "file content here"},
            {"role": "tool", "tool_call_id": "call_2", "content": "search results"},
        ],
        tool_events=[
            ToolEvent(tool_name="read_file", direction="call", tool_call_id="call_1"),
            ToolEvent(tool_name="web_search", direction="call", tool_call_id="call_2"),
        ],
    )

    enrich_tool_results(turn)

    assert turn.tool_events[0].result == "file content here"
    assert turn.tool_events[1].result == "search results"
    assert len(turn.tool_events) == 2


def test_enrich_handles_list_content():
    """多模态 list content 的 tool 消息也能提取文本。"""
    turn = _make_turn(
        request_messages=[
            {"role": "tool", "tool_call_id": "call_1", "content": [
                {"type": "text", "text": "Part 1"},
                {"type": "text", "text": "Part 2"},
            ]},
        ],
        tool_events=[
            ToolEvent(tool_name="read_file", direction="call", tool_call_id="call_1"),
        ],
    )

    enrich_tool_results(turn)

    assert "Part 1" in turn.tool_events[0].result
    assert "Part 2" in turn.tool_events[0].result


# ── review 20260707: enrich_tool_results 去重 + tool_name 提取 ──


def test_enrich_dedup_skips_old_tool_results():
    """去重：只处理最后一条 assistant 之后的 role=tool 消息。

    模拟工具循环：assistant(call A) → tool(result A) → assistant(call B) → tool(result B)
    本轮 request_messages 包含全部历史，但只有 result B 是"新的"（在最后一条 assistant 之后）。
    result A 应被跳过（已在上一轮记录过）。
    """
    turn = _make_turn(
        request_messages=[
            {"role": "user", "content": "search and extract"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_A", "type": "function", "function": {"name": "web_search", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "call_A", "content": "search results A"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_B", "type": "function", "function": {"name": "web_extract", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "call_B", "content": "extracted content B"},
        ],
        tool_events=[
            # 本轮 response 捕获的 call（call_B）
            ToolEvent(tool_name="web_extract", direction="call", tool_call_id="call_B"),
        ],
    )

    enrich_tool_results(turn)

    # call_B 匹配到 result
    assert turn.tool_events[0].result == "extracted content B"
    # result A 在最后一条 assistant 之前 → 被跳过，不产生新 result 事件
    result_events = [te for te in turn.tool_events if te.direction == "result"]
    assert len(result_events) == 0


def test_enrich_dedup_no_assistant_processes_all():
    """无 assistant 消息时处理全部 role=tool（保守，不丢数据）。"""
    turn = _make_turn(
        request_messages=[
            {"role": "tool", "tool_call_id": "call_A", "content": "result A"},
            {"role": "tool", "tool_call_id": "call_B", "content": "result B"},
        ],
        tool_events=[],
    )

    enrich_tool_results(turn)

    result_events = [te for te in turn.tool_events if te.direction == "result"]
    assert len(result_events) == 2


def test_enrich_dedup_same_call_id_not_duplicated():
    """同轮内相同 tool_call_id 的 result 不重复添加。"""
    turn = _make_turn(
        request_messages=[
            {"role": "assistant", "content": "ok"},
            {"role": "tool", "tool_call_id": "call_X", "content": "result X"},
            {"role": "tool", "tool_call_id": "call_X", "content": "result X duplicate"},
        ],
        tool_events=[],
    )

    enrich_tool_results(turn)

    result_events = [te for te in turn.tool_events if te.direction == "result"]
    assert len(result_events) == 1
    assert result_events[0].result == "result X"


def test_enrich_tool_name_from_name_field():
    """tool_name 从 OpenAI 标准 name 字段提取。"""
    turn = _make_turn(
        request_messages=[
            {"role": "assistant", "content": "ok"},
            {"role": "tool", "tool_call_id": "call_A", "name": "read_file", "content": "file content"},
        ],
        tool_events=[],
    )

    enrich_tool_results(turn)

    result_events = [te for te in turn.tool_events if te.direction == "result"]
    assert len(result_events) == 1
    assert result_events[0].tool_name == "read_file"


def test_enrich_tool_name_from_content_source():
    """tool_name 从 content 的 source="..." 属性提取（Hermes 格式）。"""
    turn = _make_turn(
        request_messages=[
            {"role": "assistant", "content": "ok"},
            {"role": "tool", "tool_call_id": "call_A", "content": '<untrusted_tool_result source="web_search">Found results</untrusted_tool_result>'},
        ],
        tool_events=[],
    )

    enrich_tool_results(turn)

    result_events = [te for te in turn.tool_events if te.direction == "result"]
    assert len(result_events) == 1
    assert result_events[0].tool_name == "web_search"


def test_enrich_tool_name_fallback_to_call_lookup():
    """tool_name fallback：从当前轮 call events 按 tool_call_id 查。"""
    turn = _make_turn(
        request_messages=[
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_A", "function": {"name": "terminal", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "call_A", "content": "command output"},
        ],
        tool_events=[
            ToolEvent(tool_name="terminal", direction="call", tool_call_id="call_A"),
        ],
    )

    enrich_tool_results(turn)

    # call_A 匹配到 call → result 回填，不产生新 result 事件
    assert turn.tool_events[0].result == "command output"
    assert turn.tool_events[0].tool_name == "terminal"


def test_enrich_tool_name_empty_when_no_source():
    """无 name、无 source、无匹配 call → tool_name 为空（best effort）。"""
    turn = _make_turn(
        request_messages=[
            {"role": "assistant", "content": "ok"},
            {"role": "tool", "tool_call_id": "call_unknown", "content": "some result without source info"},
        ],
        tool_events=[],
    )

    enrich_tool_results(turn)

    result_events = [te for te in turn.tool_events if te.direction == "result"]
    assert len(result_events) == 1
    assert result_events[0].tool_name == ""


# ── T4(ADR-0018 §4.4 C7)验收④: Anthropic tool_result block 回填 ──


def test_enrich_anthropic_tool_result_matched():
    """T4 验收④(C7): Anthropic user 消息内 type=tool_result block 按 tool_use_id 回填 call event。

    Anthropic 工具返回是 user 消息里的 tool_result block（非 role=tool），
    parse_anthropic_request 归一后保留此 list 格式，enrich 必须识别并回填。
    """
    turn = _make_turn(
        request_messages=[
            {"role": "user", "content": [{"type": "text", "text": "search news"}]},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_abc", "name": "web_search", "input": {"query": "news"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_abc", "content": "Found 3 articles"},
            ]},
        ],
        tool_events=[
            ToolEvent(tool_name="web_search", arguments={"query": "news"}, direction="call", tool_call_id="toolu_abc"),
        ],
    )

    enrich_tool_results(turn)

    assert turn.tool_events[0].result == "Found 3 articles"
    assert len(turn.tool_events) == 1  # matched, no extra event


def test_enrich_anthropic_tool_result_unmatched_creates_event():
    """T4(C7): Anthropic 未匹配 tool_result（属上一轮）记为独立 result 事件，不丢失。"""
    turn = _make_turn(
        request_messages=[
            {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_xyz", "content": "prev round result"},
            ]},
        ],
        tool_events=[],
    )

    enrich_tool_results(turn)

    result_events = [te for te in turn.tool_events if te.direction == "result"]
    assert len(result_events) == 1
    assert result_events[0].result == "prev round result"
    assert result_events[0].tool_call_id == "toolu_xyz"


def test_enrich_anthropic_tool_result_list_content():
    """T4(C7): Anthropic tool_result content 为 list[{type:text}] 时拼接文本回填。"""
    turn = _make_turn(
        request_messages=[
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_l", "name": "read_file", "input": {}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_l", "content": [
                    {"type": "text", "text": "Part A"},
                    {"type": "text", "text": "Part B"},
                ]},
            ]},
        ],
        tool_events=[
            ToolEvent(tool_name="read_file", arguments={}, direction="call", tool_call_id="toolu_l"),
        ],
    )

    enrich_tool_results(turn)

    assert "Part A" in turn.tool_events[0].result
    assert "Part B" in turn.tool_events[0].result
    assert len(turn.tool_events) == 1


def test_enrich_no_tool_results_noop_anthropic_text_only():
    """T4(C7): 纯文本 Anthropic 请求（user content 仅 text block）不触发回填，早退。"""
    turn = _make_turn(
        request_messages=[
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
        ],
        tool_events=[],
    )

    enrich_tool_results(turn)

    assert len(turn.tool_events) == 0
