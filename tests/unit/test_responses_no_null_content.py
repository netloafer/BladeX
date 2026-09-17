"""MQ-P17：`/v1/responses` 归一出来的消息里**不许有 `content is None`**（2026-09-08）。

## 为什么立这一条

ADR-0023 一直写着"`/v1/responses` 代码侧已就绪（pytest 71 全绿），**待真实 Codex 流量
签收**"。批 G 剧本③ 是第一次真实 Codex 流量，**第一个工具回合就挂**：

    responses_upstream_call_failed
      error='Router.BadRequestError: OpenAIException -
             The request failed because it is missing `messages.content` parameter'   × 53

`POST /v1/responses` 63 次里 200 只有 10 次，而那 10 次**全部发生在 Codex 第一次工具调用
之前**（`msg_count=6` / `msgs_before=4`）；第一个带 `function_call` 的回合起（`msg_count=9`）
53 次全 502。两边用的是**同一个上游模型** ⇒ 差异只能来自消息内容。

根因：`_normalize_input_item` 的顶层 `function_call` 分支与
`_normalize_assistant_message_content` 的空正文分支都产出 `content: None`。
OpenAI 官方 API **接受** assistant + tool_calls 且 `content: null`；
**Ark 的 OpenAI 兼容端点不接受**，要求键存在（空串即可）。

## 判据放在**产物**上，不放在分支上

老测试全绿而 live 全挂，正是因为它们断言的是"这个分支转出来长什么样"——
换一条没被列举的路径就照样静默。本文件只问一句话：**归一之后，有没有任何一条消息的
`content` 是 `None`**。新增分支自动被这条覆盖，不需要有人记得回来补断言。
（同族纪律：`feedback_closed_set_from_definition` —— 枚举闭集要从定义读。）
"""

from __future__ import annotations

from bladex_proxy.responses import parse_responses_request


def _null_content_messages(msgs: list[dict]) -> list[dict]:
    """归一产物里 `content is None` 的那些（判据单一定义点）。"""
    return [m for m in msgs if m.get("content", "__missing__") is None]


# ── ① live 病例回放：带 function_call 的完整一轮 ─────────────────────────────


def _codex_tool_round() -> dict:
    """Codex 第一个工具回合的形状：user 提问 → function_call → function_call_output。

    这正是 09-08 那 53 次 502 的输入形状（在此之前的纯文本轮全部 200）。
    """
    return {
        "model": "gpt-5-codex",
        "instructions": "You are Codex.",
        "input": [
            {"type": "message", "role": "user",
             "content": [{"type": "input_text", "text": "列一下仓库根目录"}]},
            {"type": "function_call", "call_id": "call_abc",
             "name": "shell", "arguments": "{\"command\":\"ls\"}"},
            {"type": "function_call_output", "call_id": "call_abc",
             "output": "packages\ndocs\nscripts"},
        ],
    }


def test_a_tool_round_produces_no_null_content():
    msgs, _ = parse_responses_request(_codex_tool_round())
    assert _null_content_messages(msgs) == [], (
        "上游（Ark OpenAI 兼容端点）会以 missing messages.content 拒收整条请求")


def test_the_function_call_message_keeps_its_tool_calls():
    """阳性对照：把 `None` 换成 `""` **不许**顺手把 tool_calls 弄丢。

    没有这一条，上一条也能靠"整条消息不生成"通过 —— 那是另一个更糟的缺陷
    （静默丢掉模型的工具调用）。
    """
    msgs, _ = parse_responses_request(_codex_tool_round())
    calls = [m for m in msgs if m.get("tool_calls")]
    assert len(calls) == 1
    assert calls[0]["role"] == "assistant"
    assert calls[0]["content"] == ""
    assert calls[0]["tool_calls"][0]["function"]["name"] == "shell"
    # tool 结果照旧独立成一条（同 Anthropic tool_result）
    tools = [m for m in msgs if m.get("role") == "tool"]
    assert len(tools) == 1 and tools[0]["tool_call_id"] == "call_abc"


# ── ② 第二处同族：assistant message 正文为空 ────────────────────────────────


def test_assistant_message_with_only_a_function_call_block_has_empty_content():
    """`function_call` 塞在 assistant message 的 content blocks 里（Codex 的另一种写法）。

    这是 `_normalize_assistant_message_content` 那一处 —— 与 ① 是同一个错的两个产生点，
    修一处漏一处正是 MQ-L53 / MQ-A52 那条"同一条判据只用在一半上"的形态。
    """
    msgs, _ = parse_responses_request({"input": [
        {"type": "message", "role": "assistant",
         "content": [{"type": "function_call", "call_id": "c1",
                      "name": "shell", "arguments": "{}"}]},
    ]})
    assert _null_content_messages(msgs) == []
    assert msgs[0]["content"] == "" and msgs[0]["tool_calls"]


def test_assistant_message_with_text_keeps_the_text():
    """阳性对照：有正文时原样保留（别把修法做成"content 一律置空"）。"""
    msgs, _ = parse_responses_request({"input": [
        {"type": "message", "role": "assistant",
         "content": [{"type": "output_text", "text": "好的，我先看目录"}]},
    ]})
    assert msgs[0]["content"] == "好的，我先看目录"


# ── ③ 全形态扫一遍（判据放在产物上的意义所在）──────────────────────────────


def test_no_null_content_across_every_input_item_shape():
    """把已知 item 形态混在一条请求里，整体只问一句：有没有 `content is None`。

    列举的是**输入形态**（来自 responses.py 的 docstring：message / function_call /
    function_call_output，以及 user content 里内嵌 function_call_output 的变体），
    断言的是**产物性质** —— 将来加了新形态，忘了改这个文件也会红。
    """
    msgs, _ = parse_responses_request({
        "instructions": "sys",
        "input": [
            "…" if False else {"type": "message", "role": "user", "content": "纯字符串正文"},
            {"type": "message", "role": "user",
             "content": [{"type": "input_text", "text": "块正文"}]},
            {"type": "message", "role": "user",
             "content": [{"type": "function_call_output", "call_id": "c9",
                          "output": "内嵌在 user content 里的工具结果"}]},
            {"type": "message", "role": "assistant", "content": []},
            {"type": "message", "role": "assistant",
             "content": [{"type": "output_text", "text": ""}]},
            {"type": "function_call", "call_id": "c1", "name": "shell", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": ""},
        ],
    })
    assert _null_content_messages(msgs) == []
    assert msgs, "整条请求被吃空了 —— 那不是修好，是换了一个缺陷"
