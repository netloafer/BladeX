"""批 E · MQ-A82：codex 自报的内部轮次（标题 / 建议 / 压缩）按 `thread_source` 判 aux。

🔴 病例（2026-09-28，`docs/planning/report-batch-E-20260928.md` §E0）：Jason 让 codex 做
**一个**「0.3.0 复核」任务，Hub 里多了**三本**账本。第二本建在 Codex Desktop 的
**会话标题生成子调用**里——它带了 8 个自带工具，`brings_no_tools`（MQ-L21）挡不住；
`_machine_text_mark` 是内容启发式、设计上只告警不拦。建本后它抢走 scope 激活位，
主会话下一轮看到的是「生成标题」这本，候选段为空，只能再建第三本。

判据：codex 在 header `x-codex-turn-metadata`（JSON）里**自报**的 `thread_source`
∈ {thread_title, ambient_suggestions, system} ⇒ 完整 aux（不给账本族 / 记忆族工具、
蒸馏跳过）。`user` = 真实轮；`subagent` 不动；没见过的值只告警。

夹具 = Hub 原文（`local/codex/01a0e6d3-…/1790579054864-0` 与主会话 `01a0e6cf-…`），
id 类字段脱敏、长文本截短，**结构逐字保留**（header 键名、`client_metadata`、
`text.format`、`tools` 名单、input 三条的角色与顺序）。
"""

from __future__ import annotations

import json

import pytest
import structlog
from bladex_core.task_goal import TURN_DROPPED, TURN_REAL
from bladex_proxy.identity import (
    AUX_REASON_SOURCE,
    CODEX_INTERNAL_THREAD_SOURCES,
    CODEX_INTERNAL_TURN_RULE_PREFIX,
    LEDGER_ONLY_AUX_REASONS,
    brings_no_tools,
    classify_turn_disposition,
    codex_thread_source,
    is_cheap_tier_auxiliary,
    is_ledgerless_auxiliary,
    resolve_codex_internal_turn,
    resolve_identity,
)
from bladex_proxy.models import ChatCompletionRequest
from bladex_proxy.responses import parse_responses_request

_INSTALL = "00000000-0000-4000-8000-000000000001"
_TITLE_SID = "01a0e6d3-0000-4000-8000-000000000002"
_MAIN_SID = "01a0e6cf-0000-4000-8000-000000000003"

_CODEX_SYSTEM = ("You are Codex, a coding agent. You and the user share the same workspace "
                 "and collaborate to achieve the user's goals.")
_AGENTS_MD = "# AGENTS.md instructions for /Users/alice/dev/BladeX\n\n<INSTRUCTIONS>\n# BladeX — AI 协作开发指南\n…"
_TITLE_PROMPT = (
    "You are a helpful assistant. You will be presented with a user prompt, and your job is "
    "to provide a short title for a task that will be created from that prompt.\n"
    "The tasks typically have to do with coding-related tasks, for example requests for bug "
    "fixes or questions about a codebase. The title you generate will be shown in the UI to "
    "represent the prompt.\nGenerate a concise UI title (up to 36 characters) for this task.\n"
    "Fill the structured title field with plain text.\n…\n\n"
    "bladex 0.3.0已经推进一段时间了，复核一下版本进展，是否偏离了主线；是否有遗漏的功能；"
    "是否做了无用功，在一些不重要的功能上浪费了时间；是否存在其它的问题。")
_USER_PROMPT = ("bladex 0.3.0已经推进一段时间了，复核一下版本进展，是否偏离了主线；是否有遗漏的功能；"
                "是否做了无用功，在一些不重要的功能上浪费了时间；是否存在其它的问题。\n")

#: Desktop `thread_title` 子调用自带的 8 个工具（Hub 原文名单，逐字）。
_TITLE_TOOLS = ["exec_command", "write_stdin", "list_mcp_resources",
                "list_mcp_resource_templates", "read_mcp_resource", "request_user_input",
                "view_image", "mcp__node_repl"]
_MAIN_TOOLS = _TITLE_TOOLS[:7] + ["multi_agent_v1", "mcp__codex_app", "mcp__cua_repl",
                                  "mcp__node_repl", "get_goal", "create_goal", "update_goal"]


def _turn_metadata(sid: str, *, thread_source: str, turn_trigger: str | None) -> str:
    meta = {
        "installation_id": _INSTALL, "session_id": sid, "thread_id": sid,
        "agent_name": "/root", "turn_id": sid, "window_id": f"{sid}:0", "window_number": 0,
        "context_window_id": sid, "request_kind": "turn", "root_turn_id": sid,
        "thread_source": thread_source,
        "sandbox": "seatbelt", "sandbox_mode": "read-only", "auto_review_enabled": False,
        "node_repl_auto_review_required": False, "node_repl_disabled": False,
        "turn_started_at_unix_ms": 1790579031499,
    }
    if turn_trigger:
        meta["turn_trigger"] = turn_trigger
    return json.dumps(meta, separators=(",", ":"))


def _headers(sid: str, meta: str, originator: str = "Codex Desktop") -> dict[str, str]:
    return {
        "x-codex-beta-features": "remote_compaction_v2",
        "x-codex-window-id": f"{sid}:0",
        "x-codex-turn-metadata": meta,
        "x-client-request-id": sid, "session-id": sid, "thread-id": sid,
        "accept": "text/event-stream", "content-type": "application/json",
        "authorization": "Bearer test-key", "originator": originator,
        "user-agent": f"{originator}/0.154.0-alpha.6.2 (Mac OS 27.0.0; arm64)",
        "host": "127.0.0.1:38080", "x-agent-id": "", "x-session-id": "",
    }


def _tool(name: str) -> dict:
    return {"type": "function", "name": name, "description": name,
            "parameters": {"type": "object", "properties": {}}, "strict": False}


def _body(sid: str, meta: str, *, tools: list[str], user_text: str,
          json_schema: bool) -> dict:
    body = {
        "model": "glm-5.3-flash", "instructions": _CODEX_SYSTEM,
        "input": [
            {"type": "message", "role": "developer",
             "content": [{"type": "input_text", "text": "<skills_instructions>…</skills_instructions>"}]},
            {"type": "message", "role": "user",
             "content": [{"type": "input_text", "text": _AGENTS_MD}]},
            {"type": "message", "role": "user",
             "content": [{"type": "input_text", "text": user_text}]},
        ],
        "tools": [_tool(n) for n in tools], "tool_choice": "auto", "parallel_tool_calls": True,
        "reasoning": {"effort": "low"}, "store": False, "stream": True,
        "include": ["reasoning.encrypted_content"], "prompt_cache_key": sid,
        "client_metadata": {"x-codex-turn-metadata": meta},
    }
    if json_schema:
        body["text"] = {"format": {"type": "json_schema", "strict": True, "name": "codex_output_schema",
                                   "schema": {"type": "object",
                                              "properties": {"title": {"type": "string"},
                                                             "description": {"type": "string"}},
                                              "required": ["title", "description"],
                                              "additionalProperties": False}}}
    return body


def _req(body: dict) -> ChatCompletionRequest:
    messages, extra = parse_responses_request(body)
    return ChatCompletionRequest(
        model=body.get("model", ""), messages=messages, stream=True,
        **{k: v for k, v in extra.items() if k in ("tools", "tool_choice")})


@pytest.fixture
def title_call():
    """病例本体：Desktop `thread_title` 子调用（带 8 个自带工具 + json_schema）。"""
    meta = _turn_metadata(_TITLE_SID, thread_source="thread_title", turn_trigger="thread_title")
    return _headers(_TITLE_SID, meta), _body(_TITLE_SID, meta, tools=_TITLE_TOOLS,
                                             user_text=_TITLE_PROMPT, json_schema=True)


@pytest.fixture
def main_call():
    """同一用户任务的主会话首轮（`thread_source=user`）。"""
    meta = _turn_metadata(_MAIN_SID, thread_source="user", turn_trigger="composer")
    return _headers(_MAIN_SID, meta), _body(_MAIN_SID, meta, tools=_MAIN_TOOLS,
                                            user_text=_USER_PROMPT, json_schema=False)


class TestCase:
    def test_title_subcall_is_full_aux(self, title_call):
        headers, body = title_call
        req = _req(body)
        assert brings_no_tools(req.tools) is False, "前提：这条子调用带自带工具，MQ-L21 挡不住"
        ident, _src = resolve_identity(headers, req, body=body)
        assert ident.agent_id.split(":")[0] == "codex"
        assert ident.auxiliary is True
        assert ident.aux_source == f"{CODEX_INTERNAL_TURN_RULE_PREFIX}thread_title"
        # 账本域：aux_source 子条件 ⇒ 账本族 + 记忆族都不注（不在 LEDGER_ONLY 集合）
        assert is_ledgerless_auxiliary(ident.aux_source) is True
        assert AUX_REASON_SOURCE not in LEDGER_ONLY_AUX_REASONS
        # 路由：可走 aux 降档（静态策略仍压过它）
        assert is_cheap_tier_auxiliary(ident.aux_source) is True

    def test_main_turn_untouched(self, main_call):
        headers, body = main_call
        ident, _src = resolve_identity(headers, _req(body), body=body)
        assert ident.auxiliary is False and ident.aux_source == ""
        assert ident.session_id == _MAIN_SID

    @pytest.mark.parametrize("src", sorted(CODEX_INTERNAL_THREAD_SOURCES))
    def test_each_internal_source_is_ruled(self, src):
        meta = _turn_metadata(_TITLE_SID, thread_source=src, turn_trigger=None)
        assert resolve_codex_internal_turn("codex", {"x-codex-turn-metadata": meta}) == \
            f"{CODEX_INTERNAL_TURN_RULE_PREFIX}{src}"

    def test_reverse_mutation_rule_table_emptied(self, monkeypatch, title_call):
        """去掉识别规则 ⇒ 用例红：证明 aux 判定确实来自这张表，不是别的闸顺手挡的。"""
        import bladex_proxy.identity as ident_mod
        monkeypatch.setattr(ident_mod, "CODEX_INTERNAL_THREAD_SOURCES", frozenset())
        headers, body = title_call
        ident, _ = resolve_identity(headers, _req(body), body=body)
        assert ident.auxiliary is False and ident.aux_source == ""


class TestRuleShape:
    def test_only_codex_reads_the_header(self):
        meta = _turn_metadata(_TITLE_SID, thread_source="thread_title", turn_trigger=None)
        assert resolve_codex_internal_turn("hermes:default", {"x-codex-turn-metadata": meta}) == ""
        assert resolve_codex_internal_turn("", {"x-codex-turn-metadata": meta}) == ""
        assert resolve_codex_internal_turn("codex:guardian", {"x-codex-turn-metadata": meta}) != ""

    def test_subagent_source_is_left_alone(self):
        meta = _turn_metadata(_TITLE_SID, thread_source="subagent", turn_trigger=None)
        with structlog.testing.capture_logs() as logs:
            assert resolve_codex_internal_turn("codex", {"x-codex-turn-metadata": meta}) == ""
        assert not [e for e in logs if e["event"] == "identity_codex_thread_source_unknown"]

    def test_unknown_source_warns_but_does_not_rule(self):
        meta = _turn_metadata(_TITLE_SID, thread_source="some_new_kind", turn_trigger=None)
        with structlog.testing.capture_logs() as logs:
            assert resolve_codex_internal_turn("codex", {"x-codex-turn-metadata": meta}) == ""
        hits = [e for e in logs if e["event"] == "identity_codex_thread_source_unknown"]
        assert len(hits) == 1 and hits[0]["thread_source"] == "some_new_kind"

    def test_body_client_metadata_is_the_fallback(self):
        meta = _turn_metadata(_TITLE_SID, thread_source="ambient_suggestions", turn_trigger=None)
        body = {"client_metadata": {"x-codex-turn-metadata": meta}}
        assert codex_thread_source({}, body) == "ambient_suggestions"
        assert codex_thread_source({"X-Codex-Turn-Metadata": meta}, None) == "ambient_suggestions"

    @pytest.mark.parametrize("raw", ["", "not json", "[1,2]", '{"no_thread_source": 1}'])
    def test_malformed_or_missing_is_empty(self, raw):
        assert codex_thread_source({"x-codex-turn-metadata": raw}, None) == ""
        assert codex_thread_source(None, None) == ""


class TestThreeProtocolsUntouched:
    """主会话 / 非 codex 的正常请求逐字不受影响：规则表置空前后 Identity 完全相同。"""

    @staticmethod
    def _same_with_and_without_rule(monkeypatch, headers, req, body):
        import bladex_proxy.identity as ident_mod
        a, _ = resolve_identity(headers, req, body=body)
        monkeypatch.setattr(ident_mod, "CODEX_INTERNAL_THREAD_SOURCES", frozenset())
        b, _ = resolve_identity(headers, req, body=body)
        assert a.model_dump() == b.model_dump()
        assert a.auxiliary is False and a.aux_source == ""

    def test_responses_codex_main(self, monkeypatch, main_call):
        headers, body = main_call
        self._same_with_and_without_rule(monkeypatch, headers, _req(body), body)

    def test_chat_completions_other_agent(self, monkeypatch):
        body = {"model": "m", "messages": [
            {"role": "system", "content": "You are Hermes, an autonomous agent."},
            {"role": "user", "content": "小黑，看看这个仓库的测试为什么红"}],
            "tools": [{"type": "function", "function": {"name": "terminal"}}]}
        headers = {"authorization": "Bearer k", "user-agent": "python-httpx/0.27",
                   "x-agent-id": "", "x-session-id": ""}
        req = ChatCompletionRequest(model="m", messages=body["messages"], tools=body["tools"])
        self._same_with_and_without_rule(monkeypatch, headers, req, body)

    def test_anthropic_messages_claude_code(self, monkeypatch):
        from bladex_proxy.anthropic import parse_anthropic_request
        body = {"model": "m", "max_tokens": 8, "stream": True,
                "system": "You are Claude Code, Anthropic's official CLI for Claude.",
                "messages": [{"role": "user", "content": "看一下 git status"}],
                "tools": [{"name": "Bash", "input_schema": {"type": "object"}}]}
        messages, extra = parse_anthropic_request(body)
        req = ChatCompletionRequest(model="m", messages=messages,
                                    tools=extra.get("tools"))
        headers = {"authorization": "Bearer k", "user-agent": "claude-cli/2.0.0",
                   "x-agent-id": "", "x-session-id": ""}
        self._same_with_and_without_rule(monkeypatch, headers, req, body)


class TestRebuildRejudge:
    """Index 重建按 header 快照重算（不读冻结的 `identity.auxiliary`）。"""

    def test_dropped_from_header_snapshot(self, title_call):
        headers, body = title_call
        msgs, _ = parse_responses_request(body)
        cls, rule = classify_turn_disposition(msgs, agent_id="codex", headers=headers)
        assert cls == TURN_DROPPED and rule == f"{CODEX_INTERNAL_TURN_RULE_PREFIX}thread_title"

    def test_messages_only_call_is_unchanged(self, title_call):
        _headers_, body = title_call
        msgs, _ = parse_responses_request(body)
        assert classify_turn_disposition(msgs) == (TURN_REAL, "")

    def test_wired_into_index_rebuild(self):
        """两处重判调用点都把 header 快照传进去（漏一处 = 热路径判 aux、重建又当真轮）。"""
        from _source_probe import package_source
        src = package_source("storage")
        assert src.count("headers=turn.identity.request_headers") == 2
