"""ADR-0023 T3: Codex 指纹核对。

验证 parse_responses_request 归一后的 messages 能被 resolve_identity 正确识别为 codex。
Codex CLI 走 /v1/responses，instructions 是顶层字段 -> parse 转成首条 system 消息。
"""
from __future__ import annotations

from bladex_proxy.identity import _extract_system_text, _extract_tool_names, _fingerprint_agent
from bladex_proxy.responses import parse_responses_request


# Codex CLI 真实 system prompt 开头（公开规范："You are Codex, a coding agent ..."）
CODEX_INSTRUCTIONS = (
    "You are Codex, a coding agent created by OpenAI. You help users with software "
    "engineering tasks by running shell commands and applying patches."
)
CODEX_TOOLS = [
    {"type": "function", "name": "shell", "description": "run shell", "parameters": {"type": "object"}},
    {"type": "function", "name": "apply_patch", "description": "apply patch", "parameters": {"type": "object"}},
    {"type": "function", "name": "update_plan", "description": "update plan", "parameters": {"type": "object"}},
]


def test_codex_instructions_normalize_to_system_message() -> None:
    """instructions 顶层字段 -> 首条 system 消息（identity 提取入口）。"""
    body = {
        "instructions": CODEX_INSTRUCTIONS,
        "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
        "tools": CODEX_TOOLS,
    }
    messages, extra = parse_responses_request(body)
    assert messages[0]["role"] == "system"
    assert "Codex" in messages[0]["content"]

    system_text = _extract_system_text(messages)
    assert "codex" in system_text.lower()


def test_codex_tools_normalize_to_openai_nested() -> None:
    """Responses 扁平 tools -> OpenAI 嵌套，_extract_tool_names 能提取。"""
    body = {"input": "hi", "tools": CODEX_TOOLS}
    _, extra = parse_responses_request(body)
    # extra["tools"] 是嵌套格式
    nested = extra["tools"]
    assert nested[0] == {"type": "function", "function": {"name": "shell", "description": "run shell", "parameters": {"type": "object"}}}

    names = _extract_tool_names([], nested)
    assert {"shell", "apply_patch", "update_plan"} <= names


def test_codex_fingerprint_matches_via_normalized_messages() -> None:
    """端到端：归一后的 messages + tools 命中 codex 指纹规则。"""
    body = {
        "instructions": CODEX_INSTRUCTIONS,
        "input": "list files",
        "tools": CODEX_TOOLS,
    }
    messages, extra = parse_responses_request(body)
    agent_id, trigger, _aux = _fingerprint_agent(messages, extra.get("tools"))
    assert agent_id == "codex"
    assert "system_prompt" in trigger


def test_codex_fingerprint_matches_tools_only() -> None:
    """无 system prompt 关键词时，纯工具签名也能命中（min_tool_match=1）。"""
    body = {
        "input": "hi",
        "tools": [{"type": "function", "name": "shell", "parameters": {}}],
    }
    messages, extra = parse_responses_request(body)
    agent_id, trigger, _aux = _fingerprint_agent(messages, extra.get("tools"))
    assert agent_id == "codex"
    assert "tool" in trigger
