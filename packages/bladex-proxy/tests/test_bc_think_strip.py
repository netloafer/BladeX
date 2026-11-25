"""批 BC §2.2：孤立思考闭标签的结构化剥离 + reasoning item 回程（README ④「no thinking text leaks」）。

## 夹具

`fixtures/bc_think_marker_rows.json`：Hub 全库 26,907 轮里带标记的 43 轮的**片段**（42 轮要切，
1 轮是反引号里的合法字面量），由 `docs/benchmarks/bc/replay_bc.py --fixture` 生成，已去路径与邮箱。
标记字符在本文件里一律用 `capture.THINK_CLOSE` / `THINK_OPEN` 拼，不写字面量。

## 尺子不用被测函数

"代码区外还有没有标记"用本文件自己的 `_outside_code`（正则遮罩）判，不用 `split_inline_reasoning`
自证——否则去掉代码区遮罩的变异会被同一个错误带着一起绿（原则 13 层 4）。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import structlog.testing
from bladex_proxy.capture import (
    THINK_CLOSE,
    THINK_OPEN,
    CaptureResult,
    InlineThinkFilter,
    merge_inline_reasoning,
    split_inline_reasoning,
)
from bladex_proxy.responses import (
    format_responses_response,
    parse_responses_request,
    responses_stream_generator,
    responses_wants_reasoning_item,
    strip_response_inline_reasoning,
)

ROWS: list[dict[str, Any]] = json.loads(
    (Path(__file__).parent / "fixtures" / "bc_think_marker_rows.json").read_text("utf-8"))
_CUT_ROWS = [r for r in ROWS if r["expect_cut"]]
_CHUNK_SIZES = (1, 2, 3, 5, 7, 8, 9, 13, 64, 100_000)


def _outside_code(text: str) -> str:
    """遮掉围栏与同一行内的行内代码，留下代码区外的文本（独立于被测状态机的尺子）。"""
    text = re.sub(r"^ {0,3}(`{3,})[^\n]*\n.*?(?:^ {0,3}\1`*[ \t]*$|\Z)", "", text,
                  flags=re.DOTALL | re.MULTILINE)
    return re.sub(r"(`+)[^`\n]*?\1", "", text)


def _has_marker(text: str) -> bool:
    outside = _outside_code(text)
    return THINK_CLOSE in outside or THINK_OPEN in outside


def _chunks(text: str, size: int) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text), size)] or [""]


def _delta(content: str | None = None, reasoning: str | None = None,
           tool_calls: list | None = None) -> SimpleNamespace:
    return SimpleNamespace(content=content, reasoning_content=reasoning, tool_calls=tool_calls)


def _chunk(delta: SimpleNamespace, finish_reason: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)],
                           usage=None)


def _tool_delta(call_id: str, name: str, arguments: str, index: int = 0) -> SimpleNamespace:
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


async def _run_stream(chunks: list[SimpleNamespace], **kw: Any) -> tuple[list[dict], CaptureResult]:
    result = CaptureResult()
    events: list[dict] = []
    async for raw in responses_stream_generator(_Stream(chunks), result, "m", **kw):
        lines = raw.decode().splitlines()
        events.append(json.loads(lines[1][len("data: "):]))
    return events, result


def _deltas(events: list[dict]) -> str:
    return "".join(e["delta"] for e in events if e["type"] == "response.output_text.delta")


def _message_done(events: list[dict]) -> list[dict]:
    return [e["item"] for e in events if e["type"] == "response.output_item.done"
            and e["item"]["type"] == "message"]


def _completed(events: list[dict]) -> dict:
    return [e for e in events if e["type"] == "response.completed"][-1]["response"]


def _item_text(item: dict) -> str:
    return "".join(b["text"] for b in item["content"])


# ── ① 夹具：真实 43 轮 ──────────────────────────────────────────────────────


def test_fixture_is_the_registered_population() -> None:
    assert len(ROWS) == 43
    assert sum(r["expect_cut"] for r in ROWS) == 42
    assert {r["shape"] for r in ROWS} == {"tail", "middle", "code_or_paired"}


@pytest.mark.parametrize("row", ROWS, ids=lambda r: f"{r['family']}-{r['shape']}-{r['text'][:12]}")
def test_fixture_row_split(row: dict) -> None:
    prefix, clean, stats = split_inline_reasoning(row["text"])
    assert bool(stats.orphan_closes) is row["expect_cut"]
    assert prefix[-40:] == row["expect_prefix_tail"]
    assert clean[:40] == row["expect_clean_head"]
    assert not _has_marker(clean)
    if row["expect_cut"]:
        assert THINK_CLOSE not in prefix
    else:
        assert clean == row["text"], "代码区内的字面量要逐字保留"


@pytest.mark.parametrize("row", ROWS, ids=lambda r: f"{r['family']}-{r['shape']}-{r['text'][:12]}")
def test_split_is_idempotent(row: dict) -> None:
    _, clean, _ = split_inline_reasoning(row["text"])
    assert split_inline_reasoning(clean)[1] == clean
    assert split_inline_reasoning(clean)[2].orphan_closes == 0


# ── ② 规则：每条对应 §2.2-a ─────────────────────────────────────────────────


def test_no_marker_returns_text_verbatim() -> None:
    text = "  普通正文，前后空白都留着。\n\n"
    assert split_inline_reasoning(text)[:2] == ("", text)


def test_leading_close_only_removes_marker() -> None:
    prefix, clean, stats = split_inline_reasoning(f"{THINK_CLOSE}已完成。")
    assert (prefix, clean, stats.orphan_closes) == ("", "已完成。", 1)


def test_last_orphan_close_is_the_cut_point() -> None:
    text = f"想一\n\n{THINK_CLOSE}\n\n想二{THINK_CLOSE}\n\n{THINK_CLOSE}正文"
    prefix, clean, stats = split_inline_reasoning(text)
    assert stats.orphan_closes == 3
    assert clean == "正文"
    assert prefix == "想一\n\n\n\n想二"


@pytest.mark.parametrize("text", [
    f"默认先 `{THINK_OPEN}` 再回答，`{THINK_CLOSE}` 结束。",
    f"``双反引号 {THINK_CLOSE} 也算``。",
    f"```xml\n{THINK_OPEN}x{THINK_CLOSE}\n```\n后文。",
    f"~说明~\n   ```\n{THINK_CLOSE}\n   ```\n后文。",
], ids=["inline", "double-backtick", "fence", "indented-fence"])
def test_code_literal_kept_verbatim(text: str) -> None:
    prefix, clean, stats = split_inline_reasoning(text)
    assert (prefix, clean, stats.orphan_closes) == ("", text, 0)


def test_code_literal_kept_while_orphan_after_it_is_cut() -> None:
    text = f"提到 `{THINK_CLOSE}` 这个字面量。{THINK_CLOSE}正文 `{THINK_CLOSE}` 留。"
    prefix, clean, stats = split_inline_reasoning(text)
    assert stats.orphan_closes == 1
    assert prefix == f"提到 `{THINK_CLOSE}` 这个字面量。"
    assert clean == f"正文 `{THINK_CLOSE}` 留。"


def test_paired_open_outside_code_is_not_stripped_only_reported() -> None:
    text = f"{THINK_OPEN}配对的{THINK_CLOSE}正文"
    prefix, clean, stats = split_inline_reasoning(text)
    assert (prefix, clean) == ("", text)
    assert (stats.orphan_closes, stats.open_tags) == (0, 1)


def test_known_edge_lone_backtick_keeps_following_close() -> None:
    """结算稿 §10 明写的边界：同一行落单反引号后面的标记不剥（宁留勿误删）。"""
    text = f"这个 ` 落单了 {THINK_CLOSE}还在"
    assert split_inline_reasoning(text)[1] == text
    # 换行后行内代码状态复位，下一行的标记照常切
    prefix, clean, _ = split_inline_reasoning(f"这个 ` 落单\n想完了{THINK_CLOSE}正文")
    assert (prefix, clean) == ("这个 ` 落单\n想完了", "正文")


def test_known_edge_bare_close_in_sentence_is_cut() -> None:
    """已知漏网（全库 0 例，规则挡不住）：钉住行为，变了要知道。"""
    assert split_inline_reasoning(f"解释一下 {THINK_CLOSE} 这个标签")[1] == "这个标签"


def test_bare_prose_without_marker_not_stripped() -> None:
    text = "看起来很像思考的长散文。" * 100
    assert split_inline_reasoning(text)[:2] == ("", text)


def test_merge_keeps_both_sources() -> None:
    assert merge_inline_reasoning("结构化", "") == "结构化"
    assert merge_inline_reasoning("", "前缀") == "前缀"
    assert merge_inline_reasoning("结构化", "前缀") == "结构化\n前缀"


# ── ③ 流式与整段判定一致（同一文本，任意切法）────────────────────────────────


@pytest.mark.parametrize("size", _CHUNK_SIZES)
@pytest.mark.parametrize("row", ROWS, ids=lambda r: f"{r['family']}-{r['shape']}-{r['text'][:12]}")
def test_filter_segments_independent_of_chunking(row: dict, size: int) -> None:
    flt = InlineThinkFilter()
    segments: list[str | None] = []
    for piece in _chunks(row["text"], size):
        segments += flt.feed(piece)
    segments += flt.flush()
    whole = InlineThinkFilter()
    whole_segments = whole.feed(row["text"]) + whole.flush()
    assert flt.stats == whole.stats
    assert "".join(s for s in segments if s) == "".join(s for s in whole_segments if s)
    if row["expect_cut"]:
        assert "".join(s for s in segments if s) == row["text"].replace(THINK_CLOSE, "")
    else:
        assert "".join(s for s in segments if s) == row["text"]


@pytest.mark.asyncio
@pytest.mark.parametrize("size", _CHUNK_SIZES)
async def test_stream_done_snapshot_hub_agree_on_every_row(size: int) -> None:
    for row in ROWS:
        chunks = [_chunk(_delta(c)) for c in _chunks(row["text"], size)]
        events, result = await _run_stream(chunks + [_chunk(_delta(), "stop")])
        _, clean, _ = split_inline_reasoning(row["text"])
        assert result.full_text == clean
        assert _completed(events)["output_text"] == clean
        for item in _message_done(events) + [i for i in _completed(events)["output"]
                                             if i["type"] == "message"]:
            assert _item_text(item) == clean
        assert not _has_marker(_deltas(events))
        for e in events:
            if e["type"] == "response.content_part.done":
                assert e["part"]["text"] == clean
        assert result.done


# ── ④ 流式的分支 ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_marker_split_across_chunks_is_eaten() -> None:
    a, b = THINK_CLOSE[:3], THINK_CLOSE[3:]
    events, result = await _run_stream([
        _chunk(_delta(f"想一下{a}")), _chunk(_delta(f"{b}结论")), _chunk(_delta(), "stop")])
    assert THINK_CLOSE not in _deltas(events) and a not in _deltas(events)
    assert _deltas(events) == "想一下结论"
    assert result.full_text == "结论"
    assert result.reasoning_text == "想一下"


@pytest.mark.asyncio
async def test_partial_lookalike_is_released_not_lost() -> None:
    text = "比较 a </thin b，以及 <th 收尾 <"
    events, result = await _run_stream(
        [_chunk(_delta(c)) for c in _chunks(text, 4)] + [_chunk(_delta(), "stop")])
    assert _deltas(events) == text
    assert result.full_text == text


@pytest.mark.asyncio
async def test_prefix_streamed_with_no_body_still_has_item_in_snapshot() -> None:
    """`visible_text_sent` 分支：流里发过 message item，快照没有它 codex 就对账失败（08-25 同族）。"""
    events, result = await _run_stream([
        _chunk(_delta(f"全是思考{THINK_CLOSE}\n\n")), _chunk(_delta(), "stop")])
    assert _deltas(events) == "全是思考\n\n"
    assert result.full_text == ""
    assert result.reasoning_text == "全是思考"
    done = _message_done(events)
    assert len(done) == 1 and done[0]["content"] == []
    snapshot = [i for i in _completed(events)["output"] if i["type"] == "message"]
    assert len(snapshot) == 1 and snapshot[0]["content"] == []
    assert snapshot[0]["id"] == done[0]["id"]
    assert _completed(events)["output_text"] == ""


@pytest.mark.asyncio
async def test_no_message_item_when_only_marker_arrives() -> None:
    events, result = await _run_stream([_chunk(_delta(THINK_CLOSE)), _chunk(_delta(), "stop")])
    assert not _message_done(events)
    assert _completed(events)["output"] == []
    assert result.full_text == ""


@pytest.mark.asyncio
async def test_tool_call_flushes_held_tail_before_closing_message() -> None:
    """扣留的尾部要先放行，否则会被下面关掉的 message 吞掉。"""
    events, result = await _run_stream([
        _chunk(_delta("比较 a <")),
        _chunk(_delta(tool_calls=[_tool_delta("c1", "exec_command", "{}")])),
        _chunk(_delta(), "tool_calls"),
    ])
    types = [e["type"] for e in events]
    assert _deltas(events) == "比较 a <"
    last_delta = max(i for i, t in enumerate(types) if t == "response.output_text.delta")
    assert last_delta < types.index("response.content_part.done")
    assert _item_text(_message_done(events)[0]) == "比较 a <"
    assert result.full_text == "比较 a <"


@pytest.mark.asyncio
async def test_guard_log_and_strip_log_both_fire() -> None:
    with structlog.testing.capture_logs() as logs:
        await _run_stream([_chunk(_delta(f"想{THINK_CLOSE}答")), _chunk(_delta(), "stop")])
    names = [x["event"] for x in logs]
    assert "inline_think_leaked" in names, "上游原文的读数要留着"
    stripped = [x for x in logs if x["event"] == "inline_reasoning_stripped"]
    assert len(stripped) == 1
    assert (stripped[0]["markers"], stripped[0]["chars"], stripped[0]["path"]) == (1, 1, "responses")


@pytest.mark.asyncio
async def test_clean_text_has_no_strip_log() -> None:
    with structlog.testing.capture_logs() as logs:
        await _run_stream([_chunk(_delta("干净正文")), _chunk(_delta(), "stop")])
    assert not [x for x in logs if x["event"] in ("inline_reasoning_stripped",
                                                  "inline_think_open_tag_kept")]


@pytest.mark.asyncio
async def test_paired_open_tag_only_warns() -> None:
    text = f"{THINK_OPEN}x{THINK_CLOSE}正文"
    with structlog.testing.capture_logs() as logs:
        events, result = await _run_stream([_chunk(_delta(text)), _chunk(_delta(), "stop")])
    assert result.full_text == text == _deltas(events)
    assert [x["open_tags"] for x in logs if x["event"] == "inline_think_open_tag_kept"] == [1]


# ── ⑤ 非流式 ─────────────────────────────────────────────────────────────────


def _response(content: str | None, reasoning: str | None = None) -> SimpleNamespace:
    message = SimpleNamespace(content=content, reasoning_content=reasoning, tool_calls=[])
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")],
                           usage=None)


def test_nonstream_strip_rewrites_in_place_for_every_consumer() -> None:
    resp = _response(f"想{THINK_CLOSE}答", "结构化")
    with structlog.testing.capture_logs() as logs:
        strip_response_inline_reasoning(resp, "responses:non_stream")
    assert resp.choices[0].message.content == "答"
    assert resp.choices[0].message.reasoning_content == "结构化\n想"
    assert [x["path"] for x in logs if x["event"] == "inline_reasoning_stripped"] == [
        "responses:non_stream"]


def test_nonstream_meta_reads_merged_reasoning() -> None:
    from bladex_proxy.server.orchestration import _build_response_meta_from_response
    resp = _response(f"想{THINK_CLOSE}答")
    strip_response_inline_reasoning(resp, "responses:non_stream")
    assert _build_response_meta_from_response(resp).reasoning_text == "想"


def test_nonstream_clean_response_untouched() -> None:
    resp = _response("  干净  ", None)
    strip_response_inline_reasoning(resp, "x")
    assert resp.choices[0].message.content == "  干净  "
    assert resp.choices[0].message.reasoning_content is None


@pytest.mark.parametrize("bad", [SimpleNamespace(), SimpleNamespace(choices=[]),
                                 _response(None), _response("")])
def test_nonstream_strip_tolerates_unparseable_shapes(bad: SimpleNamespace) -> None:
    strip_response_inline_reasoning(bad, "x")


def test_format_cuts_too_for_paths_that_skip_the_endpoint_strip() -> None:
    """内循环的最终回复走 format，不经端点那次就地剥离：这里必须再切一次（幂等）。"""
    out = format_responses_response(_response(f"想{THINK_CLOSE}答"), "m")
    assert out["output_text"] == "答"
    assert out["output"][0]["content"][0]["text"] == "答"


@pytest.mark.parametrize("row", _CUT_ROWS[:6], ids=lambda r: r["text"][:12])
def test_nonstream_and_stream_agree(row: dict) -> None:
    import asyncio
    events, result = asyncio.run(_run_stream(
        [_chunk(_delta(row["text"])), _chunk(_delta(), "stop")]))
    resp = _response(row["text"])
    strip_response_inline_reasoning(resp, "x")
    formatted = format_responses_response(resp, "m")
    assert formatted["output_text"] == result.full_text == _completed(events)["output_text"]
    assert resp.choices[0].message.reasoning_content == result.reasoning_text


# ── ⑥ reasoning item 回程（§2.2-c，拍板 (48)-④）──────────────────────────────


@pytest.mark.parametrize(("reasoning", "want"), [
    (None, False), ("x", False), ({}, False), ({"effort": "high"}, False),
    ({"effort": "high", "summary": "none"}, False), ({"effort": "high", "summary": None}, False),
    ({"effort": "high", "summary": "detailed"}, True), ({"summary": "auto"}, True),
])
def test_wants_reasoning_item(reasoning: Any, want: bool) -> None:
    assert responses_wants_reasoning_item({"reasoning": reasoning}) is want


_REASONING_CHUNKS = [
    _chunk(_delta(reasoning="想")), _chunk(_delta(reasoning="一下")),
    _chunk(_delta("正文")),
    _chunk(_delta(tool_calls=[_tool_delta("c1", "exec_command", "{}")])),
    _chunk(_delta(), "tool_calls"),
]


@pytest.mark.asyncio
async def test_reasoning_item_event_order_and_dynamic_indexes() -> None:
    events, result = await _run_stream(list(_REASONING_CHUNKS), emit_reasoning=True)
    types = [e["type"] for e in events]
    expected_head = ["response.created", "response.output_item.added",
                     "response.reasoning_summary_part.added",
                     "response.reasoning_summary_text.delta",
                     "response.reasoning_summary_text.delta",
                     "response.reasoning_summary_text.done",
                     "response.reasoning_summary_part.done",
                     "response.output_item.done"]
    assert types[:len(expected_head)] == expected_head
    added = {e["item"]["type"]: e["output_index"] for e in events
             if e["type"] == "response.output_item.added"}
    assert added == {"reasoning": 0, "message": 1, "function_call": 2}
    for e in events:
        if e["type"] in ("response.output_text.delta", "response.content_part.added",
                         "response.content_part.done"):
            assert e["output_index"] == 1
    output = _completed(events)["output"]
    assert [i["type"] for i in output] == ["reasoning", "message", "function_call"]
    assert output[0]["summary"] == [{"type": "summary_text", "text": "想一下"}]
    assert output[0]["encrypted_content"] is None
    reasoning_done = [e["item"] for e in events if e["type"] == "response.output_item.done"
                      and e["item"]["type"] == "reasoning"]
    assert reasoning_done == [output[0]]
    assert result.reasoning_text == "想一下"


@pytest.mark.asyncio
async def test_no_summary_no_item_and_message_index_zero() -> None:
    events, result = await _run_stream(list(_REASONING_CHUNKS))
    assert not [e for e in events if "reasoning" in e["type"]]
    added = {e["item"]["type"]: e["output_index"] for e in events
             if e["type"] == "response.output_item.added"}
    assert added == {"message": 0, "function_call": 1}
    assert [i["type"] for i in _completed(events)["output"]] == ["message", "function_call"]
    assert result.reasoning_text == "想一下", "没要 item 也要进 Hub"


@pytest.mark.asyncio
async def test_late_reasoning_only_reaches_hub() -> None:
    events, result = await _run_stream([
        _chunk(_delta(reasoning="早")), _chunk(_delta("正文")),
        _chunk(_delta(reasoning="晚")), _chunk(_delta(), "stop")], emit_reasoning=True)
    assert _completed(events)["output"][0]["summary"][0]["text"] == "早"
    assert result.reasoning_text == "早晚"


@pytest.mark.asyncio
async def test_inline_prefix_never_backfills_reasoning_item() -> None:
    events, result = await _run_stream(
        [_chunk(_delta(f"想{THINK_CLOSE}答")), _chunk(_delta(), "stop")], emit_reasoning=True)
    assert [i["type"] for i in _completed(events)["output"]] == ["message"]
    assert result.reasoning_text == "想"


def test_nonstream_reasoning_item_only_when_asked() -> None:
    resp = _response("答", "想")
    assert [i["type"] for i in format_responses_response(resp, "m")["output"]] == ["message"]
    with_item = format_responses_response(resp, "m", emit_reasoning=True)["output"]
    assert [i["type"] for i in with_item] == ["reasoning", "message"]
    assert with_item[0]["summary"][0]["text"] == "想"


def test_replayed_reasoning_item_skipped_counted_not_unknown() -> None:
    body = {"input": [
        {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "想"}]},
        {"type": "message", "role": "user", "content": "hi"},
        {"type": "reasoning", "id": "rs_2", "summary": []},
    ]}
    with structlog.testing.capture_logs() as logs:
        messages, _ = parse_responses_request(body)
    assert messages == [{"role": "user", "content": "hi"}]
    assert [x["count"] for x in logs if x["event"] == "responses_reasoning_items_skipped"] == [2]
    assert not [x for x in logs if x["event"] == "responses_unknown_input_item_type"]
