"""OpenAI Responses API (/v1/responses) 格式适配（ADR-0023）。

Codex CLI 原生走 OpenAI Responses API，格式与 /v1/chat/completions 不同：
  - instructions 是顶层字段（不在 input 数组里）
  - input 是 string 或 item 数组（message / function_call / function_call_output）
  - content block 类型用 input_text / output_text（非 text）
  - tools 是扁平结构（{type:"function", name, parameters}，非嵌套 function:{}）
  - SSE event 类型不同（response.created / response.output_text.delta / response.completed）

本模块负责：
  1. 解析 Responses 请求 -> 归一成内部 OpenAI chat 格式 messages（复用全流程）
  2. 流式：自己迭代上游 stream、按 Responses SSE 格式回传 + 拼完整回复
  3. 非流式：把 OpenAI chat 格式响应转成 Responses 格式

无状态假设（ADR-0023 §2.3）：Codex 每轮发完整 input，BladeX 当无状态 proxy 处理。
previous_response_id 非空由 server handler 拦截回 400（T6），不进本模块。
"""

from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import structlog

from bladex_proxy.capture import (
    CaptureResult,
    InlineThinkFilter,
    cache_read_tokens,
    merge_inline_reasoning,
    note_cache_reading,
    note_inline_reasoning_stripped,
    note_usage,
    record_first_chunk,
    round_fields,
    split_inline_reasoning,
    warn_bare_reasoning,
    warn_inline_think,
)
from bladex_proxy.models import ToolEvent

logger = structlog.get_logger()

# OpenAI chat finish_reason -> Responses status 映射
_STATUS_MAP: dict[str, str] = {
    "stop": "completed",
    "length": "incomplete",
    "tool_calls": "completed",
    "content_filter": "incomplete",
}


# ── P36（2026-10-28，MQ-P36）：Responses `namespace` 工具族的名字映射 ──────────
#
# codex 把「子代理」「MCP」这类工具族按 `{"type":"namespace","name":<族名>,
# "tools":[...]}` 整块下发（声明侧 = `codex-rs/tools/src/tool_spec.rs:25-26`
# `ToolSpec::Namespace`，结构 `codex-rs/tools/src/responses_api.rs:59-62`）；
# 调用回执的 `function_call` item 也带 `namespace` 字段（`codex-rs/protocol/src/models.rs:1124-1149`
# `ResponseItem::FunctionCall{ name, namespace, .. }`；codex 侧
# `codex-rs/core/src/tools/router.rs:267` 用 `ToolName::new(namespace, name)`
# 查注册表，查不到就回 `unsupported call: <name>`
# —— `codex-rs/core/src/tools/registry.rs:1088`）。
#
# chat 协议没有 namespace 概念 ⇒ 只能把族名编进函数名。本段是**唯一实现点**：
# 请求侧（展开声明）、历史侧（回放上一轮的 `function_call`）、响应侧（把上游的
# 调用名还原成 codex 认的形态）三处都只经 `responses_tool_name` /
# `split_responses_tool_name` 这一对。
#
# 编码 = `ns-<族名>-<内层名>`，两段各自 sanitize 到 `[A-Za-z0-9_]`：
#   · 上游字符集 / 长度：OpenAI OpenAPI `FunctionDefinition.name`「Must be a-z,
#     A-Z, 0-9, or contain underscores and dashes, with a maximum length of 64」
#     （`openai/openai-openapi` master `openapi.yaml` L42862-42863；ARK 走
#     OpenAI 兼容端点，同一约束）。
#   · `-` 只作分隔符：codex 侧族名与内层名本来就只含 `[A-Za-z0-9_]`
#     （`codex-rs/codex-mcp/src/mcp/mod.rs:603` `sanitize_responses_api_tool_name`），
#     两段内不可能出现 `-` ⇒ 在第一个 `-` 处切分是**单射**；跨 namespace 的
#     同名内层函数（live 里 `mcp__cua_repl` 与 `mcp__node_repl` 都有 `js` /
#     `js_reset`）因此不会串。
#   · `ns-` 前缀让「普通函数名」永远不会被误还原（没有前缀 = 原名返回），
#     普通 function 与 BladeX 自己注入的 `bladex_*` 走这条，转换逐字不变。
_NS_TOOL_PREFIX = "ns-"
_NS_TOOL_SEP = "-"
#: 上游 chat function name 上限（出处见上，含大小写/下划线/中划线）。
MAX_UPSTREAM_TOOL_NAME = 64

#: 名字里只允许 `[A-Za-z0-9_]`（与 codex 侧 sanitize 同一字符集）。
_NON_NAME_CHARS = re.compile(r"[^A-Za-z0-9_]")


def _sanitize_name_part(part: str) -> str:
    """一段名字 sanitize 到 `[A-Za-z0-9_]`（其它字符 -> `_`）。

    codex 侧本来就已经是这个字符集（MCP 侧有自己的 sanitizer，族名/内层名是
    Rust 常量或 sanitize 过的标识符），这里只是**不让脏字符把单射破坏掉**。
    """
    return _NON_NAME_CHARS.sub("_", part)


def responses_tool_name(name: str, namespace: str | None = None) -> str:
    """codex 侧 `(namespace, name)` -> 上游 chat function name（编码唯一实现点）。

    无 namespace（普通 function / `bladex_*`）⇒ **原名返回**，一个字符都不动。
    """
    if not namespace:
        return name
    return (f"{_NS_TOOL_PREFIX}{_sanitize_name_part(namespace)}"
            f"{_NS_TOOL_SEP}{_sanitize_name_part(name)}")


def split_responses_tool_name(upstream_name: str) -> tuple[str | None, str]:
    """上游 chat function name -> `(namespace, name)`（解码唯一实现点）。

    不是本模块编出来的形态（没有 `ns-` 前缀 / 前缀后面没有第二段）⇒
    `(None, 原名)`，交给调用方原样使用。
    """
    if not upstream_name.startswith(_NS_TOOL_PREFIX):
        return None, upstream_name
    rest = upstream_name[len(_NS_TOOL_PREFIX):]
    if _NS_TOOL_SEP not in rest:
        return None, upstream_name
    namespace, name = rest.split(_NS_TOOL_SEP, 1)
    if not namespace or not name:
        return None, upstream_name
    return namespace, name


def responses_call_item_name_fields(upstream_name: str) -> dict[str, Any]:
    """上游调用名 -> Responses 调用 item 的 `name` / `namespace` 两格。

    普通函数**不加 `namespace` 键**（修前逐字相同）。
    """
    namespace, name = split_responses_tool_name(upstream_name)
    if namespace is None:
        return {"name": name}
    return {"name": name, "namespace": namespace}


# ── 批 BC §2.3（MQ-P37）：`custom`（freeform）工具 ⇄ chat function ───────────────
#
# codex 的 `apply_patch` handler 只吃 `ToolPayload::Custom`（`core/src/tools/handlers/apply_patch.rs:375-379`）。
# chat 协议没有 custom ⇒ 声明侧转成「单个字符串参数 `input`」的函数，响应侧再还原成
# `custom_tool_call`，历史侧再转回 `tool_calls`。
# 名字走 P36 同一套（`responses_tool_name`），**不新增前缀**：模型看到的仍是 `apply_patch`，
# codex 的 instructions 按字面提这个名字。种类不编进名字，每个请求由
# `responses_custom_tool_names(body)` 算出集合；漏传 = 空集合 = 退回 function_call（修前行为）。
_CUSTOM_INPUT_KEY = "input"
#: DeepSeek 单字符串参数工具的尾巴：JSON 没闭合，后面跟着 DSML 结束标签（BC 探针 3 / 13，MQ-P42）。
_DSML_TAIL = re.compile(r"(?:\\n)?\s*</[｜|]DSML[｜|]")


def responses_custom_tool_names(body: dict[str, Any]) -> frozenset[str]:
    """请求里声明的 custom 工具的**上游名**集合（**唯一集合函数**，含 namespace 内层）。"""
    names: set[str] = set()
    tools = body.get("tools")
    for tool in tools if isinstance(tools, list) else []:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "custom" and tool.get("name"):
            names.add(responses_tool_name(str(tool["name"])))
        elif tool.get("type") == "namespace":
            namespace = str(tool.get("name") or "")
            for inner in tool.get("tools") or []:
                if isinstance(inner, dict) and inner.get("type") == "custom" and inner.get("name"):
                    names.add(responses_tool_name(str(inner["name"]), namespace))
    return frozenset(names)


def _salvage_dsml_arguments(arguments: str) -> str | None:
    """`{"input":"…\\n</｜DSML｜invoke>…`（JSON 未闭合）-> 抢救出字符串值；不是这个形态 -> None。"""
    match = _DSML_TAIL.search(arguments)
    if match is None:
        return None
    try:
        obj = json.loads(arguments[:match.start()].rstrip() + '"}', strict=False)
    except json.JSONDecodeError:
        return None
    value = obj.get(_CUSTOM_INPUT_KEY) if isinstance(obj, dict) else None
    return value if isinstance(value, str) else None


def custom_input_from_arguments(arguments: str, tool_name: str = "") -> str:
    """上游 chat 的 `arguments` -> custom 工具的 `input`（**唯一解包点**）。

    四种形态，每种有 live 依据（BC 结算稿 §2.3）；后三种都告警，不静默：
    ① `{"input":"…"}` · ② 单键字符串（`{"patch":"…"}`，CX2 真实形态）·
    ③ JSON 尾部被 DSML 截断 · ④ 其它：原文当 input，codex 回 `verification failed`，模型看得见。
    """
    def _warn(shape: str) -> None:
        logger.warning("responses_custom_args_unwrapped", shape=shape, tool_name=tool_name,
                       arguments_len=len(arguments))

    try:
        obj = json.loads(arguments, strict=False) if arguments else None
    except json.JSONDecodeError:
        obj = None
    if isinstance(obj, dict):
        value = obj.get(_CUSTOM_INPUT_KEY)
        if isinstance(value, str):
            if len(obj) > 1:
                _warn("input_with_extra_keys")
            return value
        if len(obj) == 1:
            only = next(iter(obj.values()))
            if isinstance(only, str):
                _warn("single_string_key")
                return only
    salvaged = _salvage_dsml_arguments(arguments)
    if salvaged is not None:
        _warn("dsml_tail")
        return salvaged
    _warn("raw")
    return arguments


def responses_call_item_fields(upstream_name: str, arguments: str | None,
                               custom_names: frozenset[str] = frozenset()) -> dict[str, Any]:
    """上游调用名 + 参数 -> 调用 item 的 `type` / `name` / `namespace` / 载荷（**唯一还原点**）。

    四条回写路径共用：流式（item.added / item.done / 终止快照）、非流式、内循环合成事件。
    `arguments=None` = item 刚开口、参数还没到（不解包，不告警）。
    """
    names = responses_call_item_name_fields(upstream_name)
    if upstream_name in custom_names:
        payload = "" if arguments is None else custom_input_from_arguments(arguments, upstream_name)
        return {"type": "custom_tool_call", **names, "input": payload}
    return {"type": "function_call", **names, "arguments": arguments or ""}


#: 未知工具 type 的告警限频（按 type 去重；同一 type 一分钟一条）。
_UNSUPPORTED_TOOL_TYPE_WINDOW_S = 60.0
_unsupported_tool_type_at: dict[str, float] = {}


def _warn_unsupported_tool_type(tool_type: str, tool_name: str) -> None:
    """未知工具 type 的可 grep 告警（MQ-P36 步骤 2.4：不再静默压成空参数函数）。"""
    now = time.monotonic()
    last = _unsupported_tool_type_at.get(tool_type)
    if last is not None and now - last < _UNSUPPORTED_TOOL_TYPE_WINDOW_S:
        return
    _unsupported_tool_type_at[tool_type] = now
    logger.warning(
        "responses_tool_type_unsupported",
        tool_type=tool_type,
        tool_name=tool_name,
        hint="not a `function`/`namespace` declaration; not forwarded (see MQ-P36)",
    )


def _warn_tool_name_dropped(reason: str, upstream_name: str, namespace: str, name: str) -> None:
    """namespace 内层工具被丢弃的可 grep 告警（超长 / 重名）。"""
    logger.warning(
        reason,
        upstream_name=upstream_name,
        namespace=namespace,
        inner_name=name,
        max_length=MAX_UPSTREAM_TOOL_NAME,
    )


def _extract_content_text(content: Any) -> str:
    """Responses input item 的 content 字段 -> 纯文本。

    content 可能是 str / list[input_text|output_text block] / 其他结构 -> 提取文本。
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for b in content:
            if isinstance(b, dict):
                # input_text / output_text 都取 text 字段
                if b.get("type") in ("input_text", "output_text", "text"):
                    parts.append(b.get("text", ""))
        return "\n".join(p for p in parts if p)
    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False)


def _normalize_input_item(item: dict[str, Any]) -> list[dict[str, Any]]:
    """单个 Responses input item -> OpenAI 消息列表（可能多条）。

    - message item (role=user): content blocks -> 可能拆出 tool 消息 + user 消息
      （function_call_output block -> 独立 tool 角色消息，同 Anthropic tool_result）
    - message item (role=assistant): content blocks -> assistant 消息（含 tool_calls）
      （function_call block -> tool_calls 字段）
    - 顶层 function_call item -> assistant 消息的 tool_calls
    - 顶层 function_call_output item -> 独立 tool 消息
    """
    itype = item.get("type")

    # 顶层 function_call item（不在 message 里，独立 item）
    if itype == "function_call":
        return [{
            "role": "assistant",
            # 🔴 MQ-P17（2026-09-08，Codex 首次真实流量当场撞到）：**空串不是 None**。
            # OpenAI 官方接受 assistant + tool_calls 且 `content: null`；**Ark 的 OpenAI
            # 兼容端点不接受**，报 `missing messages.content parameter`。live 读数：
            # `/v1/responses` 63 次里 200 只有 10 次，且那 10 次**全在 Codex 第一次工具
            # 调用之前** —— 第一个带 `function_call` 的回合起 53 次全 502，两边同一个上游
            # 模型 ⇒ 差异只能来自消息内容。
            # 不改成"删掉 content 键"：那是另一种形态的缺数，且 OpenAI 侧对
            # assistant+tool_calls 要求键存在。
            "content": "",
            "tool_calls": [{
                "id": item.get("call_id", "") or f"call_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {
                    # P36 历史侧：codex 回放上一轮的 `function_call` 时带 `namespace`
                    # （它记的就是我们从响应侧还原的形态）⇒ 必须用与请求侧同一套
                    # 上游名，否则上游看到的历史调用名与工具表对不上。
                    "name": responses_tool_name(
                        item.get("name", "") or "", item.get("namespace")),
                    "arguments": item.get("arguments", "") or "",
                },
            }],
        }]

    # 顶层 function_call_output item -> 独立 tool 消息
    if itype == "function_call_output":
        return [{
            "role": "tool",
            "tool_call_id": item.get("call_id", "") or "",
            "content": _extract_content_text(item.get("output")),
        }]

    # message item
    if itype == "message":
        role = item.get("role", "user")
        content = item.get("content")
        if isinstance(content, str):
            return [{"role": role, "content": content}]
        if isinstance(content, list):
            if role == "user":
                return _normalize_user_message_content(content)
            if role == "assistant":
                return [_normalize_assistant_message_content(content)]
            # system 等：拼纯文本
            return [{"role": role, "content": _extract_content_text(content)}]
        return [{"role": role, "content": content or ""}]

    # 批 BC §2.3 历史侧：`custom_tool_call` -> assistant `tool_calls`。序列化是确定性的，
    # 每轮回放出同一串 arguments，不破坏上游前缀缓存。
    if itype == "custom_tool_call":
        return [{
            "role": "assistant",
            "content": "",      # MQ-P17：空串不是 None
            "tool_calls": [{
                "id": item.get("call_id", "") or f"call_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {
                    "name": responses_tool_name(item.get("name", "") or "", item.get("namespace")),
                    "arguments": json.dumps({_CUSTOM_INPUT_KEY: item.get("input", "") or ""},
                                            ensure_ascii=False),
                },
            }],
        }]
    if itype == "custom_tool_call_output":
        return [{
            "role": "tool",
            "tool_call_id": item.get("call_id", "") or "",
            "content": _extract_content_text(item.get("output")),
        }]

    # 批 BC §2.2-c：我们回程发过的 reasoning item 下一轮会被回放。chat 协议没有对应物 ⇒
    # 不发上游，也不算「未知类型」（个数由 parse_responses_request 汇总打日志）。
    if itype == "reasoning":
        return []

    # 未知 item 类型：warning 跳过（不阻断，让请求继续）
    logger.warning("responses_unknown_input_item_type", item_type=itype, item_keys=list(item.keys()))
    return []


def _normalize_user_message_content(content: list) -> list[dict[str, Any]]:
    """user message 的 content blocks -> OpenAI 消息列表。

    - input_text block -> 合并为一条 user 消息（纯文本 str）
    - function_call_output block（Codex 偶尔塞在 user message content 里）-> 独立 tool 消息
    - image 等多模态 block -> 保留在 user 消息 list content
    """
    tool_msgs: list[dict[str, Any]] = []
    text_parts: list[str] = []
    other_blocks: list[dict] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        bt = block.get("type")
        if bt == "function_call_output":
            tool_msgs.append({
                "role": "tool",
                "tool_call_id": block.get("call_id", "") or "",
                "content": _extract_content_text(block.get("output")),
            })
        elif bt in ("input_text", "output_text", "text"):
            text_parts.append(block.get("text", ""))
        else:
            other_blocks.append(block)
    out: list[dict[str, Any]] = list(tool_msgs)
    if other_blocks:
        blocks = list(other_blocks)
        if text_parts:
            blocks.insert(0, {"type": "text", "text": "\n".join(text_parts)})
        out.append({"role": "user", "content": blocks})
    elif text_parts:
        out.append({"role": "user", "content": "\n".join(text_parts)})
    return out


def _normalize_assistant_message_content(content: list) -> dict[str, Any]:
    """assistant message 的 content blocks -> OpenAI assistant 消息。

    - output_text block -> content（拼接，空则**空串**——MQ-P17，不是 None）
    - function_call block（塞在 assistant content 里）-> tool_calls 字段
    """
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        bt = block.get("type")
        if bt in ("output_text", "input_text", "text"):
            text_parts.append(block.get("text", ""))
        elif bt == "function_call":
            tool_calls.append({
                "id": block.get("call_id", "") or f"call_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {
                    # P36 历史侧第二处（同顶层 item 分支，同一个映射函数）。
                    "name": responses_tool_name(
                        block.get("name", "") or "", block.get("namespace")),
                    "arguments": block.get("arguments", "") or "",
                },
            })
    msg: dict[str, Any] = {"role": "assistant"}
    # MQ-P17 同族第二处：正文为空时同样给 **空串**而不是 None（理由见上）。
    msg["content"] = "\n".join(text_parts)
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def _instruction_messages(body: dict[str, Any]) -> list[dict[str, Any]]:
    """顶层 `instructions` 归一出的消息（0 或 1 条 system）。"""
    instructions = body.get("instructions")
    if instructions:
        if isinstance(instructions, str):
            return [{"role": "system", "content": instructions}]
        if isinstance(instructions, list):
            text = _extract_content_text(instructions)
            if text:
                return [{"role": "system", "content": text}]
    return []


def empty_input_error(body: dict[str, Any], messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    """`input` 一条消息都没归一出来 ⇒ 400 错误体（MQ-P24）；否则 None。

    此前照常把 `messages=[]`（或只剩 instructions 那条 system）发给上游，用户读到的是
    上游那句 "messages … minimum length 1, but got an empty array"——而 BladeX 手里
    明明有全部信息（"这些 item type 我不认识"），只在日志里打了一条
    `responses_unknown_input_item_type` warning。**失败要响在最早能判定的地方。**

    判据只看 `input` 贡献的消息数（`messages` 减去 instructions 那条）：只有 system 的
    请求同样没有可答的对话。错误信息点名收到的 item type 清单。
    """
    if len(messages) > len(_instruction_messages(body)):
        return None
    raw = body.get("input")
    if isinstance(raw, list):
        types = [str(it.get("type")) if isinstance(it, dict) else type(it).__name__ for it in raw]
    else:
        types = [] if raw is None else [type(raw).__name__]
    detail = (f"received {len(types)} input item(s) of type(s) {sorted(set(types))}, "
              f"none of which was recognized" if types else "`input` is missing or empty")
    return {"error": {
        "type": "invalid_request_error",
        "param": "input",
        "message": (f"No message could be built from `input`: {detail}. BladeX understands "
                    f"`message`, `function_call` and `function_call_output` items (and plain "
                    f"strings); refusing instead of sending an empty conversation upstream."),
    }}


def _chat_function_from_responses_tool(tool: dict[str, Any], name: str) -> dict[str, Any]:
    """单个 Responses function 声明 -> chat function 声明（`name` 由调用方映射好）。

    结构与 P36 之前**逐字相同**（description 缺省 `""`、parameters 缺省
    `{"type": "object"}`、`strict` 原样带上）——普通 function 与 BladeX 自己注入的
    `bladex_*` 的转换结果不许因为本批改动而变。
    """
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters", {"type": "object"}),
            **({"strict": tool["strict"]} if "strict" in tool else {}),
        },
    }


def _chat_function_from_custom_tool(tool: dict[str, Any], name: str) -> dict[str, Any]:
    """custom（freeform）声明 -> 单字符串参数 `input` 的 chat function（批 BC §2.3）。

    文法全文并进 description：chat 协议没有文法字段，模型只能靠这段文字照着写。
    """
    description = str(tool.get("description") or "")
    fmt = tool.get("format")
    if isinstance(fmt, dict) and fmt.get("definition"):
        description = (f"{description}\n\nThe `input` string must follow this "
                       f"{fmt.get('syntax') or 'grammar'} grammar:\n{fmt['definition']}")
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {_CUSTOM_INPUT_KEY: {
                    "type": "string",
                    "description": "The entire raw input for this tool, as one string.",
                }},
                "required": [_CUSTOM_INPUT_KEY],
                "additionalProperties": False,
            },
        },
    }


def _expand_namespace_tool(tool: dict[str, Any],
                           seen: dict[str, str]) -> list[dict[str, Any]]:
    """`{"type":"namespace", name=族名, tools:[内层…]}` -> 逐个内层 function 声明。

    🔴 MQ-P36 的核心：修前这里被当成「一个带 `name` 的普通工具」，于是发给上游的
    是 `multi_agent_v1(parameters={"type":"object"})` —— 内层工具**全丢**，模型
    只能调 `multi_agent_v1({})`，codex 回 `unsupported call: multi_agent_v1`
    （BA0 X1，10-27 23:46，`raw_request_ref=93a12141881bfa3c`）。
    """
    namespace = str(tool.get("name") or "")
    out: list[dict[str, Any]] = []
    for inner in tool.get("tools") or []:
        if not isinstance(inner, dict):
            continue
        itype = inner.get("type")
        inner_name = str(inner.get("name") or "")
        if itype not in (None, "function", "custom") or not inner_name:
            _warn_unsupported_tool_type(f"namespace:{itype}", f"{namespace}.{inner_name}")
            continue
        upstream_name = responses_tool_name(inner_name, namespace)
        if len(upstream_name) > MAX_UPSTREAM_TOOL_NAME:
            # 超长就**不发**：截断会把「能还原」这个前提毁掉（步骤 2.1），而发超长名
            # 会被上游按字符集/长度规则拒（整轮 400，比少一个工具贵得多）。
            _warn_tool_name_dropped("responses_tool_name_too_long", upstream_name,
                                    namespace, inner_name)
            continue
        if upstream_name in seen:
            _warn_tool_name_dropped("responses_tool_name_collision", upstream_name,
                                    namespace, inner_name)
            continue
        seen[upstream_name] = f"{namespace}.{inner_name}"
        out.append(_chat_function_from_custom_tool(inner, upstream_name) if itype == "custom"
                   else _chat_function_from_responses_tool(inner, upstream_name))
    return out


def _chat_tools_from_responses_tools(responses_tools: list) -> list[dict[str, Any]]:
    """Responses 工具表 -> chat 工具表（P36：namespace 展开 + 未知 type 告警）。"""
    openai_tools: list[dict[str, Any]] = []
    #: 已下发的上游函数名 -> 来源（用于重名告警/去重）。
    seen: dict[str, str] = {}
    for tool in responses_tools:
        if not isinstance(tool, dict):
            continue
        ttype = tool.get("type")
        # 已经是嵌套结构（兼容客户端直接发 OpenAI 格式）：原样透传，P36 不碰。
        if ttype == "function" and isinstance(tool.get("function"), dict):
            openai_tools.append(tool)
            continue
        # 🟢 P36：namespace 族 -> 逐个内层 function（修前整块被压成一个空参数函数）。
        if ttype == "namespace":
            openai_tools.extend(_expand_namespace_tool(tool, seen))
            continue
        if ttype == "custom" and tool.get("name"):
            upstream_name = responses_tool_name(str(tool["name"]))
            if upstream_name in seen:
                logger.warning("responses_tool_name_collision", upstream_name=upstream_name,
                               source="custom", taken_by=seen[upstream_name])
                continue
            seen[upstream_name] = "custom"
            openai_tools.append(_chat_function_from_custom_tool(tool, upstream_name))
            continue
        if ttype in (None, "function") and (tool.get("name") or ttype == "function"):
            # 普通 function：`responses_tool_name` 无 namespace ⇒ **原名返回**。
            upstream_name = responses_tool_name(
                str(tool.get("name") or ""), tool.get("namespace"))
            # MQ-P38（拍板 (48)-③）：读数 0 例，上游声明阶段也不拒 ⇒ 不做短名映射，告警后照发。
            if len(upstream_name) > MAX_UPSTREAM_TOOL_NAME:
                logger.warning("responses_tool_name_too_long", upstream_name=upstream_name,
                               source="function", max_length=MAX_UPSTREAM_TOOL_NAME)
            if upstream_name in seen:
                # 普通函数重名：修前照样下发（客户端自己的问题），这里不改变行为，
                # 只让它可 grep；被 namespace 展开占用的名字则不能下发。
                logger.warning("responses_tool_name_collision", upstream_name=upstream_name,
                               source="function", taken_by=seen[upstream_name])
            seen.setdefault(upstream_name, "function")
            openai_tools.append(_chat_function_from_responses_tool(tool, upstream_name))
            continue
        # 🔴 步骤 2.4：function / namespace / custom 之外的 type（web_search / tool_search /
        # mcp …）**不再静默**。修前它们要么被压成空参数函数（模型只能传 `{}`，与 namespace
        # 那个坑同形），要么无 `name` 被静默跳过。
        # **处置 = 不下发**：服务端内置工具 chat 协议表达不了，
        # 压成 `{}` 只会换来一轮必然失败的调用；丢掉至少是"看得见的缺"
        # （告警带 type 与 name，按 type 限频）。依据见结算稿 §2。
        _warn_unsupported_tool_type(str(ttype), str(tool.get("name") or ""))
    return openai_tools


def parse_responses_request(body: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """把 OpenAI /v1/responses 请求归一成内部 OpenAI chat 格式。

    返回 (messages, extra_kwargs)：
      messages: OpenAI chat 格式消息列表（instructions -> 首条 system 消息）
      extra_kwargs: 透传给 Router 的额外参数（max_tokens, temperature, tools, ...）

    无状态假设：不处理 previous_response_id（由 handler 拦截）。
    """
    # instructions（顶层）-> 首条 system 消息（同 Anthropic system 字段处理）
    messages: list[dict[str, Any]] = _instruction_messages(body)

    # input: string -> 单条 user 消息；list -> 逐项归一
    input_field = body.get("input")
    if isinstance(input_field, str):
        messages.append({"role": "user", "content": input_field})
    elif isinstance(input_field, list):
        for item in input_field:
            if isinstance(item, dict):
                messages.extend(_normalize_input_item(item))
            elif isinstance(item, str):
                # input 数组里也可能直接放纯文本（OpenAI 允许）
                messages.append({"role": "user", "content": item})
        skipped = sum(1 for it in input_field
                      if isinstance(it, dict) and it.get("type") == "reasoning")
        if skipped:
            logger.info("responses_reasoning_items_skipped", count=skipped)

    # 透传给 Router 的额外参数
    extra: dict[str, Any] = {}
    # max_output_tokens -> max_tokens（OpenAI chat 参数名）
    if body.get("max_output_tokens") is not None:
        extra["max_tokens"] = body["max_output_tokens"]
    for key in ("temperature", "top_p", "metadata"):
        val = body.get(key)
        if val is not None:
            extra[key] = val
    # parallel_tool_calls / strict 等 tool 语义参数透传
    if body.get("parallel_tool_calls") is not None:
        extra["parallel_tool_calls"] = body["parallel_tool_calls"]

    # tools: Responses 扁平结构 -> OpenAI 嵌套 function calling 结构
    # Responses: [{type:"function", name, description, parameters}]
    #            [{type:"namespace", name, description, tools:[...]}]  ← P36
    #            [{type:"custom", name, description, format}]          ← P37
    # OpenAI:    [{type:"function", function:{name, description, parameters}}]
    responses_tools = body.get("tools")
    if responses_tools:
        openai_tools = _chat_tools_from_responses_tools(responses_tools)
        if openai_tools:
            extra["tools"] = openai_tools

    # tool_choice: Responses 与 OpenAI chat 基本一致（auto/required/{type:function,function:{name}})
    tc = body.get("tool_choice")
    if tc is not None:
        extra["tool_choice"] = tc

    # reasoning.effort -> 顶层 `reasoning_effort`，值逐字不映射（README ④，拍板 (48)-①）。
    # 批 BC 读数：codex 10,551 份请求全带 `reasoning.effort`；此前这个值每轮都设上、又每轮被
    # `drop_params` 丢掉（tap 实录出站没有它）。现在由端点标记、`route._do_call` 按路由后的模型
    # 带 `allowed_openai_params` 放行，配置 `reasoning = "off"` 或非 openai/* 前缀时才不发。
    # `summary` / `context` 在 chat 协议里没有对应物，不发上游：`summary` 决定回程是否给
    # reasoning item（`responses_wants_reasoning_item`），`context` 忽略并计数。
    reasoning = body.get("reasoning")
    if isinstance(reasoning, dict) and reasoning.get("effort"):
        extra["reasoning_effort"] = reasoning["effort"]
    if isinstance(reasoning, dict) and reasoning.get("context") is not None:
        logger.info("responses_reasoning_context_ignored")

    return messages, extra


def responses_wants_reasoning_item(body: dict[str, Any]) -> bool:
    """请求要不要 reasoning output item（**唯一判据**，拍板 (48)-④）。

    只有带 `reasoning.summary` 才发：live 的 `glm-5.3-flash` 目录 `default_reasoning_summary=none`，
    codex 不带这个键 ⇒ 行为与修前逐字相同。
    """
    reasoning = body.get("reasoning")
    if not isinstance(reasoning, dict):
        return False
    summary = reasoning.get("summary")
    return bool(summary) and summary != "none"


def _reasoning_item(item_id: str, text: str) -> dict[str, Any]:
    """reasoning output item（codex `protocol/src/models.rs:1016-1028`，`encrypted_content` 可空）。

    ⚠️ `summary` 里放的是原始思考全文，不是真摘要；chat 上游只给这个。
    """
    return {"type": "reasoning", "id": item_id,
            "summary": [{"type": "summary_text", "text": text}],
            "encrypted_content": None}


def strip_response_inline_reasoning(response: Any, path: str) -> None:
    """非流式响应就地剥孤立 `</think>`：`content` 换成干净正文，前缀并进 `reasoning_content`。

    批 BC §2.2：在一切下游（格式化 / 拦截 / ResponseMeta / 内循环）读它之前调一次，
    `done` / 快照 / Hub 就都是同一份。解析不了的响应形状不处理（交给下游原有的错误路径）。
    """
    try:
        message = response.choices[0].message
    except (AttributeError, IndexError, TypeError):
        return
    text = getattr(message, "content", None)
    if not isinstance(text, str) or not text:
        return
    prefix, clean, stats = split_inline_reasoning(text)
    rc = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None) or ""
    rc = "".join(str(x) for x in rc) if isinstance(rc, list) else str(rc)
    note_inline_reasoning_stripped(stats, prefix, clean, len(rc), path)
    if stats.orphan_closes:
        message.content = clean
        message.reasoning_content = merge_inline_reasoning(rc, prefix)


def format_responses_response(
    response: Any,
    model: str,
    *,
    emit_reasoning: bool = False,
    custom_names: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """把 OpenAI chat 格式非流式响应转成 Responses /v1/responses 格式。

    `emit_reasoning` = `responses_wants_reasoning_item(body)`；
    `custom_names` = `responses_custom_tool_names(body)`。
    """
    try:
        message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason or "stop"
    except (AttributeError, IndexError) as e:
        logger.error("responses_response_parse_failed", error=str(e))
        return _error_response(str(e))

    output: list[dict[str, Any]] = []
    output_text_parts: list[str] = []

    rc = getattr(message, "reasoning_content", None) or getattr(message, "reasoning_text", None)
    if rc and emit_reasoning:
        output.append(_reasoning_item(f"rs_{uuid.uuid4().hex[:24]}", str(rc)))

    # 文本内容 -> message item + output_text
    # 批 BC：这里再切一次（幂等）。内循环的最终回复也走这个函数，它不经端点那次就地剥离。
    _, text, _ = split_inline_reasoning(getattr(message, "content", None) or "")
    if text:
        output_text_parts.append(text)
        output.append({
            "type": "message",
            "id": f"msg_{uuid.uuid4().hex[:24]}",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text}],
        })

    # 工具调用 -> function_call / custom_tool_call items
    tool_calls = getattr(message, "tool_calls", None) or []
    for tc in tool_calls:
        func = getattr(tc, "function", None)
        name = getattr(func, "name", "") if func else ""
        args_str = getattr(func, "arguments", "") if func else ""
        output.append(_call_item(
            f"fc_{uuid.uuid4().hex[:24]}",
            getattr(tc, "id", "") or f"call_{uuid.uuid4().hex[:12]}",
            # P36 / P37 响应侧（非流式）：与流式路径同一个还原点。
            responses_call_item_fields(name, args_str, custom_names),
            "completed",
        ))

    status = _STATUS_MAP.get(finish_reason, "completed")

    # usage
    usage_raw = getattr(response, "usage", None)
    usage = {
        "input_tokens": getattr(usage_raw, "prompt_tokens", 0) if usage_raw else 0,
        "output_tokens": getattr(usage_raw, "completion_tokens", 0) if usage_raw else 0,
        "total_tokens": getattr(usage_raw, "total_tokens", 0) if usage_raw else 0,
    }

    resp_id = f"resp_{uuid.uuid4().hex[:24]}"
    return {
        "id": resp_id,
        "object": "response",
        "created_at": int(time.time()),
        "model": model,
        "status": status,
        "output": output,
        "output_text": "\n".join(output_text_parts),
        "usage": usage,
        # 无状态 proxy：不返 previous_response_id 关联字段
        "metadata": {},
    }


def _call_item(item_id: str, call_id: str, fields: dict[str, Any], status: str) -> dict[str, Any]:
    """调用 item（键序与 P36 之前相同：type / id / call_id / name… / 载荷 / status）。"""
    return {"type": fields["type"], "id": item_id, "call_id": call_id,
            **{k: v for k, v in fields.items() if k != "type"}, "status": status}


def tool_events_from_output(output: list[dict[str, Any]]) -> list[ToolEvent]:
    """从 Responses output items 提取 call 方向 ToolEvent（非流式入库用）。

    与流式路径（responses_stream_generator）保持一致：
      - arguments 解析成 dict（非流式 format 里是 JSON 字符串，需还原）
      - 带 tool_call_id（供跨轮 result 匹配，Bug A 同源修复延伸到非流式路径）
      - 🔴 P36：`tool_name` 记 **chat 侧上游名**（namespace 编在名字里），不是
        codex 侧的 `name`/`namespace` 两格——流式那条路径记的就是上游名
        （`acc["name"]`），两条路径必须同一个口径；而且上游名可无损还原出
        `(namespace, 内层名)`，只记内层名会把 namespace 丢掉（原则 13）。
    """
    events: list[ToolEvent] = []
    for item in output:
        if isinstance(item, dict) and item.get("type") == "custom_tool_call":
            events.append(ToolEvent(
                tool_name=responses_tool_name(item.get("name", "") or "", item.get("namespace")),
                arguments={_CUSTOM_INPUT_KEY: item.get("input", "")},
                direction="call",
                tool_call_id=item.get("call_id", "") or "",
            ))
            continue
        if not isinstance(item, dict) or item.get("type") != "function_call":
            continue
        raw_args = item.get("arguments")
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) and raw_args else raw_args
        except json.JSONDecodeError:
            args = raw_args
        events.append(ToolEvent(
            tool_name=responses_tool_name(item.get("name", "") or "", item.get("namespace")),
            arguments=args,
            direction="call",
            tool_call_id=item.get("call_id", "") or "",
        ))
    return events


def _error_response(error: str) -> dict[str, Any]:
    return {
        "id": f"resp_{uuid.uuid4().hex[:24]}",
        "object": "response",
        "created_at": int(time.time()),
        "status": "failed",
        "error": {"code": "internal_error", "message": error},
        "output": [],
        "output_text": "",
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        "metadata": {},
    }


async def responses_stream_generator(
    stream: Any,
    result: CaptureResult,
    model: str,
    *,
    emit_reasoning: bool = False,
    custom_names: frozenset[str] = frozenset(),
) -> AsyncIterator[bytes]:
    """迭代上游 stream，按 Responses SSE 格式回传 + 拼完整回复。

    事件序列（OpenAI Responses streaming）：
      response.created
      -> [emit_reasoning 且上游有思考：reasoning item 的 added / summary_part.added /
          summary_text.delta(×N) / summary_text.done / summary_part.done / item.done]
      -> response.output_item.added (message)
      -> response.content_part.added (output_text)
      -> response.output_text.delta (×N)
      -> response.content_part.done
      -> response.output_item.done (message)
      -> [function_call: response.output_item.added/done + response.function_call_arguments.delta]
      -> [custom_tool_call: response.output_item.added/done。参数是 JSON 片段、流到一半解不开，
          不发 `custom_tool_call_input.delta`（它只喂 codex 的进度 UI，可缺）]
      -> response.completed
    """
    resp_id = f"resp_{uuid.uuid4().hex[:24]}"
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    # 批 BC §2.2：`raw_text_parts` 是上游原文；出站 delta 过 `think_filter`（吃孤立 `</think>`），
    # `done` / 终止快照 / Hub 一律用 `_clean_text()`。已流出的思考前缀收不回（拍板 (48)-⑤）。
    raw_text_parts: list[str] = []
    think_filter = InlineThinkFilter()
    visible_text_sent = False

    def _clean_text() -> str:
        return split_inline_reasoning("".join(raw_text_parts))[1]

    def _text_item_content() -> list[dict[str, Any]]:
        clean = _clean_text()
        return [{"type": "output_text", "text": clean}] if clean else []

    reasoning_parts: list[str] = []
    tool_calls_acc: dict[int, dict[str, Any]] = {}
    t0 = time.perf_counter()
    text_part_open = False
    message_item_added = False

    def _reasoning_delta_events(text: str) -> list[bytes]:
        """上游思考增量 → reasoning item 事件（item 关过之后的增量只进 Hub，不再发）。"""
        nonlocal reasoning_item_open, reasoning_output_idx, next_output_idx
        if not emit_reasoning or reasoning_item_text is not None:
            return []
        events: list[bytes] = []
        if not reasoning_item_open:
            reasoning_item_open = True
            reasoning_output_idx = next_output_idx
            next_output_idx += 1
            events.append(_sse_event("response.output_item.added", {
                "type": "response.output_item.added",
                "output_index": reasoning_output_idx,
                "item": {"type": "reasoning", "id": reasoning_item_id, "summary": [],
                         "encrypted_content": None},
            }))
            events.append(_sse_event("response.reasoning_summary_part.added", {
                "type": "response.reasoning_summary_part.added",
                "item_id": reasoning_item_id, "output_index": reasoning_output_idx,
                "summary_index": 0, "part": {"type": "summary_text", "text": ""},
            }))
        reasoning_item_parts.append(text)
        events.append(_sse_event("response.reasoning_summary_text.delta", {
            "type": "response.reasoning_summary_text.delta",
            "item_id": reasoning_item_id, "output_index": reasoning_output_idx,
            "summary_index": 0, "delta": text,
        }))
        return events

    def _close_reasoning_item() -> list[bytes]:
        nonlocal reasoning_item_open, reasoning_item_text
        if not reasoning_item_open:
            return []
        reasoning_item_open = False
        reasoning_item_text = "".join(reasoning_item_parts)
        common = {"item_id": reasoning_item_id, "output_index": reasoning_output_idx,
                  "summary_index": 0}
        return [
            _sse_event("response.reasoning_summary_text.done", {
                "type": "response.reasoning_summary_text.done", **common,
                "text": reasoning_item_text}),
            _sse_event("response.reasoning_summary_part.done", {
                "type": "response.reasoning_summary_part.done", **common,
                "part": {"type": "summary_text", "text": reasoning_item_text}}),
            _sse_event("response.output_item.done", {
                "type": "response.output_item.done",
                "output_index": reasoning_output_idx,
                "item": _reasoning_item(reasoning_item_id, reasoning_item_text)}),
        ]

    def _text_events(segments: list[str | None]) -> list[bytes]:
        """filter 放行的文本段 → message item 开口事件 + `output_text.delta`。"""
        nonlocal visible_text_sent, message_item_added, text_part_open
        nonlocal next_output_idx, msg_output_idx, msg_output_allocated
        events: list[bytes] = []
        for segment in segments:
            if not segment:
                continue
            events.extend(_close_reasoning_item())
            if not msg_output_allocated:
                msg_output_idx = next_output_idx
                next_output_idx += 1
                msg_output_allocated = True
            if not message_item_added:
                events.append(_sse_event("response.output_item.added", {
                    "type": "response.output_item.added",
                    "output_index": msg_output_idx,
                    "item": {
                        "type": "message",
                        "id": msg_id,
                        "status": "in_progress",
                        "role": "assistant",
                        "content": [],
                    },
                }))
                message_item_added = True
            if not text_part_open:
                events.append(_sse_event("response.content_part.added", {
                    "type": "response.content_part.added",
                    "item_id": msg_id,
                    "output_index": msg_output_idx,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": ""},
                }))
                text_part_open = True
            visible_text_sent = True
            events.append(_sse_event("response.output_text.delta", {
                "type": "response.output_text.delta",
                "item_id": msg_id,
                "output_index": msg_output_idx,
                "content_index": 0,
                "delta": segment,
            }))
        return events

    # output_index 单调计数：按 item 开口的先后分配（批 BC：message 不再写死 0，前面可能有
    # reasoning item）。纯工具回复时 message 从不出现，首个 function_call 拿 index 0，
    # 不留空洞（与 response.completed 快照一致）。
    next_output_idx = 0
    msg_output_idx = 0
    msg_output_allocated = False
    # 批 BC §2.2-c：reasoning item 只开一次；开了之后到第一段正文 / 工具调用 / 流结束时关。
    reasoning_item_id = f"rs_{uuid.uuid4().hex[:24]}"
    reasoning_output_idx = -1
    reasoning_item_open = False
    reasoning_item_text: str | None = None      # 关闭时定格；None = 没发过
    reasoning_item_parts: list[str] = []
    status = "completed"
    input_tokens = 0
    output_tokens = 0
    # 🔴 MQ-P31：None ≠ 0（同 `capture.cache_read_tokens` 的口径）。
    # 上游不报是 None，报了且没命中是 0；混成 0 会让"这条路径测不了缓存"
    # 伪装成"缓存一直没命中"。
    cache_read: int | None = None
    created_at = int(time.time())

    # response.created
    yield _sse_event("response.created", {
        "type": "response.created",
        "response": {
            "id": resp_id,
            "object": "response",
            "created_at": created_at,
            "model": model,
            "status": "in_progress",
            "output": [],
        },
    })

    try:
        async for chunk in stream:
            result.chunk_count += 1
            # MQ-P19：首字延迟。**这条路径此前没有这一行**，而 codex 只走它。
            # 共用 `capture.record_first_chunk`（三条入站路径唯一实现点）。
            record_first_chunk(result, t0)
            try:
                choice = chunk.choices[0]
                delta = choice.delta

                # reasoning_content：恒累积进 result（入 Hub）；请求带 summary 才回程给 agent
                rc = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
                if rc:
                    reasoning_parts.append(str(rc))
                    for event in _reasoning_delta_events(str(rc)):
                        yield event

                # 文本 delta
                if delta.content:
                    raw_text_parts.append(delta.content)
                    for event in _text_events(think_filter.feed(delta.content)):
                        yield event

                # 工具调用 delta
                if delta.tool_calls:
                    # 扣留的尾部先放行，否则会被下面关掉的 message 吞掉
                    for event in _text_events(think_filter.flush()):
                        yield event
                    for event in _close_reasoning_item():
                        yield event
                    # 关闭当前文本 part / message item（如果有）
                    if text_part_open:
                        yield _sse_event("response.content_part.done", {
                            "type": "response.content_part.done",
                            "item_id": msg_id,
                            "output_index": msg_output_idx,
                            "content_index": 0,
                            "part": {"type": "output_text", "text": _clean_text()},
                        })
                        text_part_open = False
                    if message_item_added:
                        yield _sse_event("response.output_item.done", {
                            "type": "response.output_item.done",
                            "output_index": msg_output_idx,
                            "item": {
                                "type": "message",
                                "id": msg_id,
                                "status": "completed",
                                "role": "assistant",
                                "content": _text_item_content(),
                            },
                        })
                        message_item_added = False

                    for tc in delta.tool_calls:
                        idx = tc.index if tc.index is not None else 0
                        if idx not in tool_calls_acc:
                            # 单调计数：有正文时 message 已占 0（next=1）；纯工具时从 0 起
                            current_output_idx = next_output_idx
                            next_output_idx += 1
                            fc_id = f"fc_{uuid.uuid4().hex[:24]}"
                            tool_calls_acc[idx] = {
                                "output_idx": current_output_idx,
                                "fc_id": fc_id,
                                "call_id": "",
                                "name": "",
                                "arguments_parts": [],
                            }
                            acc = tool_calls_acc[idx]
                            if tc.id:
                                acc["call_id"] = tc.id
                            if tc.function and tc.function.name:
                                acc["name"] = tc.function.name
                            # custom item 不经拦截器的铸 id（它只认 function_call）⇒ 这里兜底
                            if acc["name"] in custom_names and not acc["call_id"]:
                                acc["call_id"] = f"call_{uuid.uuid4().hex[:12]}"
                            yield _sse_event("response.output_item.added", {
                                "type": "response.output_item.added",
                                "output_index": current_output_idx,
                                # P36 / P37 响应侧（流式 item.added）：唯一还原点。
                                "item": _call_item(
                                    fc_id, acc["call_id"],
                                    responses_call_item_fields(acc["name"], None, custom_names),
                                    "in_progress"),
                            })
                        else:
                            acc = tool_calls_acc[idx]
                            if tc.id:
                                acc["call_id"] = tc.id
                            if tc.function:
                                if tc.function.name:
                                    acc["name"] = tc.function.name

                        # 发 function_call_arguments.delta
                        if tc.function and tc.function.arguments:
                            acc["arguments_parts"].append(tc.function.arguments)
                            if acc["name"] in custom_names:
                                continue
                            yield _sse_event("response.function_call_arguments.delta", {
                                "type": "response.function_call_arguments.delta",
                                "item_id": acc["fc_id"],
                                "output_index": acc["output_idx"],
                                "delta": tc.function.arguments,
                            })

                # finish_reason
                if choice.finish_reason:
                    # 🔴 MQ-P22：落上游原值（不是映射后的 Responses status）。
                    # 此前无生产者 ⇒ codex 的流式轮 `finish_reason` 恒空
                    # ⇒ MQ-A34 截断告警对 codex 结构性失明（实测 393 轮里 7% 有值，
                    # 那 7% 全是非流式轮）。
                    result.finish_reason = str(choice.finish_reason)
                    status = _STATUS_MAP.get(choice.finish_reason, "completed")

            except (AttributeError, IndexError, KeyError) as e:
                logger.debug("responses_chunk_skip", error=str(e))
            # 批 AC：usage 收在 `choices[0]` 的 try **之外**——OpenAI 原生形态的收尾块是
            # `choices=[]` + usage，放在里面会被 IndexError 一并跳过（chat 的 `capture_stream`
            # 一直在外面；Router 目前给 choices=1 的空 delta，所以 live 还没踩到）。
            # 🔴 MQ-P21：`getattr(obj, k, 默认)` 在**字段存在且为 None** 时返回 None
            # （上游报一个空 usage 是常见形态）⇒ 这两个变量会变成 None ⇒
            # 收尾处 `input_tokens + output_tokens` **TypeError**，而那一行在
            # try 之外、在 `response.completed` 之前 ⇒ codex 收不到终止事件。
            # 与 anthropic.py 同一处修法（同一件事，两条路径都要做对）。
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage:
                _pt = getattr(chunk_usage, "prompt_tokens", None)
                if isinstance(_pt, int):
                    input_tokens = _pt
                _ct = getattr(chunk_usage, "completion_tokens", None)
                if isinstance(_ct, int):
                    output_tokens = _ct
                # MQ-P22：同上，此前这条路径不灌 `result.usage`。
                note_usage(result, chunk_usage)
                # MQ-P31：与 anthropic 路径同名同义、同一实现点。
                _cr = cache_read_tokens(chunk_usage)
                if _cr is not None:
                    cache_read = _cr

        result.done = True

    except Exception as e:
        result.error = str(e)
        logger.error("responses_stream_interrupted", error=str(e), chunks=result.chunk_count)

    # 收尾：扣留的尾部放行，再关闭未关闭的文本 part + message item
    for event in _text_events(think_filter.flush()):
        yield event
    for event in _close_reasoning_item():
        yield event
    if text_part_open:
        yield _sse_event("response.content_part.done", {
            "type": "response.content_part.done",
            "item_id": msg_id,
            "output_index": msg_output_idx,
            "content_index": 0,
            "part": {"type": "output_text", "text": _clean_text()},
        })
    if message_item_added:
        yield _sse_event("response.output_item.done", {
            "type": "response.output_item.done",
            "output_index": msg_output_idx,
            "item": {
                "type": "message",
                "id": msg_id,
                "status": "completed",
                "role": "assistant",
                "content": _text_item_content(),
            },
        })

    # 关闭工具调用 items。P36 / P37 响应侧（流式 item.done）：唯一还原点；只算一次，
    # 终止快照用同一份（否则 codex 对账分叉，解包告警也不会打两遍）。
    for idx in sorted(tool_calls_acc.keys()):
        acc = tool_calls_acc[idx]
        acc["final_item"] = _call_item(
            acc["fc_id"], acc["call_id"],
            responses_call_item_fields(acc["name"], "".join(acc["arguments_parts"]), custom_names),
            "completed")
        yield _sse_event("response.output_item.done", {
            "type": "response.output_item.done",
            "output_index": acc["output_idx"],
            "item": acc["final_item"],
        })

    # 填充 result（Hub 正文 = 干净正文；剥出来的前缀换字段进 reasoning，内容不丢）
    raw_full_text = "".join(raw_text_parts)
    inline_prefix, clean_text, inline_stats = split_inline_reasoning(raw_full_text)
    structured_reasoning = "".join(reasoning_parts)
    result.full_text = clean_text
    result.reasoning_text = merge_inline_reasoning(structured_reasoning, inline_prefix)
    result.ms = (time.perf_counter() - t0) * 1000

    for idx in sorted(tool_calls_acc.keys()):
        acc = tool_calls_acc[idx]
        args_str = "".join(acc["arguments_parts"])
        try:
            args = json.loads(args_str) if args_str else None
        except json.JSONDecodeError:
            args = args_str
        result.tool_events.append(ToolEvent(
            tool_name=acc["name"], arguments=args, direction="call",
            tool_call_id=acc["call_id"],
        ))

    logger.info(
        "responses_capture_done",
        chunks=result.chunk_count,
        text_len=len(result.full_text),
        tool_events=len(result.tool_events),
        elapsed_ms=round(result.ms, 2),
        # MQ-P19：与另两条路径同名同义（见 `capture.record_first_chunk`）。
        ms_first_chunk=round(result.ms_first_chunk, 2),
        # MQ-P18：结构化思考流走没走上来。`reasoning_len=0` 且下一行告警命中
        # ⇒ 上游把 reasoning **内联进了正文**，修法在响应侧剥离而不在请求侧转发。
        reasoning_len=len(result.reasoning_text),
        # 🔴 MQ-P31（2026-09-10 立，codex 双侧对照驱动）：这三格此前**一个都没有**，
        # 而 `capture_done`（chat）记 `usage`、`anthropic_capture_done` 记
        # `input_tokens_reported` / `cache_read_tokens` ⇒ **同一个读数三条路径三种拼法，
        # 其中一条压根没有**。代价：codex 经 BladeX 的 178 轮里 token 用量、上下文峰值、
        # 缓存命中率**全部不可测**，于是"BladeX 是否把上下文推过了 codex 的压缩阈值"
        # ——当前最大的开放问题——无法判定（裸跑侧 tap 有 usage，两侧比不了）。
        # ⚠️ 本批**只加不改**：另两条路径的字段名一个不动。改名会把正在跑的 bench
        #    序列断掉（尺子与被测对象不许分两次动）；三路字段统一归后续卡。
        # 🔴 `usage` 与这两格并列是故意的：`result.usage` 为空而 `input_tokens_reported`
        #    非零 ⇒ 是 `extract_usage` 的载体判断漏了，不是上游没报。两者分辨得开。
        usage=result.usage,
        input_tokens_reported=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read,
        done=result.done,
        error=result.error,
        # 批 R · R1：归属 / 工具结果体量 / 起止墙钟（三条路径同一实现点 `capture.round_fields`）。
        **round_fields(result),
    )
    # G17.0 守卫（MQ-P18）：量上游原文漏不漏（批 BC 起正文已剥，这里看的是原文）。
    # ~~`apply_patch` abort 的症状是 `</think>` 漏进正文~~ 批 BC 否证：工具参数 think 标记
    # 0 / 10,990 轮，abort 的根因是协议形状（ADR-0023 11-03 补记）。两处仍都查：
    # 工具参数那格是阴性读数，真出现了要看得见。
    note_cache_reading(result, cache_read)
    warn_inline_think(raw_full_text, "responses")
    note_inline_reasoning_stripped(inline_stats, inline_prefix, clean_text,
                                   len(structured_reasoning), "responses")
    warn_inline_think(
        "".join("".join(acc["arguments_parts"]) for acc in tool_calls_acc.values()),
        "responses:tool_args",
    )
    # MQ-P29 第二步（批 L）：认行为不认标记 —— 正文长 ∧ 同轮 reasoning 为 0。
    warn_bare_reasoning(result.full_text, len(result.reasoning_text), "responses")

    # 构建最终 output（用于 response.completed）
    final_output: list[dict[str, Any]] = []
    if reasoning_item_text is not None:
        final_output.append(_reasoning_item(reasoning_item_id, reasoning_item_text))
    # 判据是「流里发过」不是「干净正文非空」：思考前缀全流出、后面没正文时，流里已经有这个
    # message item，快照没有它 codex 就对账失败（08-25 断连同族）。
    if visible_text_sent:
        final_output.append({
            "type": "message",
            "id": msg_id,
            "status": "completed",
            "role": "assistant",
            "content": _text_item_content(),
        })
    for idx in sorted(tool_calls_acc.keys()):
        # P36 / P37 响应侧（终止快照）：必须与 item.done 逐字一致，否则 codex 对账分叉。
        final_output.append(tool_calls_acc[idx]["final_item"])

    # response.completed
    yield _sse_event("response.completed", {
        "type": "response.completed",
        "response": {
            "id": resp_id,
            "object": "response",
            "created_at": created_at,
            "model": model,
            "status": status,
            "output": final_output,
            "output_text": clean_text,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
            "metadata": {},
        },
    })


def _sse_event(event_type: str, data: dict[str, Any]) -> bytes:
    """格式化一个 Responses SSE event。"""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode()
