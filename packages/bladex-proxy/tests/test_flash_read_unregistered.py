"""批 F5 守卫 · `bladex_flash_read` 退出注册（拍板 (45)，2026-10-17）。

F5 判决：预注册窗口 09-22 17:42 → 10-16，拿到工具面的主 lane 轮 1,712、调用 **0 次**，
低于预注册判据「调用率 ≥ 1% 且有消费」⇒ 模型工具面不再提供它，行为画像只留给人看
（`GET /admin/behavior` / dashboard，用例见 `test_admin_behavior_endpoint.py`）。

判据：
① 三协议（chat / messages / responses）各一条真实形状的请求，**转发给上游的 tools 数组**
  里没有 `bladex_flash_read`，且 bladex 段与 `TOOL_SCHEMAS` 逐字相同（不是只少了一个名字，
  而是其余工具一个字没动）；
② 运行时注册表、`TOOL_NAMES`、Anthropic 派生形态里都没有它；
③ 老会话缓存了工具清单、模型仍调这个名字 ⇒ 走「未知 bladex 工具」路径回结构化错误，
  端点照常 200（不是 500、不把调用漏给 agent）。

🔴 阴性对照（结算稿记录）：把 `toolface.TOOL_SCHEMAS` 的条目与 `runtime` 的注册加回 ⇒ ①② 红。
"""

from __future__ import annotations

import asyncio
import tempfile
from unittest.mock import patch

import pytest
from bladex_proxy.agency import AgencyRuntime
from bladex_proxy.config import ProxyConfig
from bladex_proxy.server import create_app
from bladex_proxy.toolface import TOOL_NAMES, TOOL_SCHEMAS, anthropic_tool_schemas
from fastapi.testclient import TestClient

RETIRED = "bladex_flash_read"

_SYS = "You are a test agent."
_OWN_TOOL = {"type": "function",
             "function": {"name": "exec_command", "description": "run a command",
                          "parameters": {"type": "object", "properties": {}}}}


def _config() -> ProxyConfig:
    tmpdir = tempfile.mkdtemp()
    return ProxyConfig(hard_rules=["MUST be polite"], upstream_model="openai/test",
                       upstream_api_key="sk-fake", rocksdb_path=f"{tmpdir}/rocksdb")


class _Msg:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self):
        d = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        return d


class _Resp:
    def __init__(self, msg: _Msg):
        self.choices = [type("C", (), {"message": msg, "finish_reason": None})()]
        self._msg = msg

    def model_dump(self):
        return {"id": "chatcmpl-x", "choices": [
            {"index": 0, "message": self._msg.model_dump(), "finish_reason": "stop"}]}


def _modules_on(monkeypatch) -> None:
    for n in ("TOOLFACE", "INTERCEPTION", "LEDGER"):
        monkeypatch.setenv(f"BLADEX_MODULE_{n}", "1")
    # 账本族也放行 ⇒ 全部 bladex 工具都在场，比对的是整张工具面
    monkeypatch.setenv("BLADEX_LEDGER_TIERS", "all")


def _post(client: TestClient, protocol: str, session: str):
    if protocol == "chat":
        return client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "stream": False, "tools": [_OWN_TOOL],
                  "messages": [{"role": "system", "content": _SYS},
                               {"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer sk", "X-Agent-ID": "test-agent",
                     "X-Session-ID": session})
    if protocol == "messages":
        return client.post(
            "/v1/messages",
            json={"model": "claude-x", "stream": False, "max_tokens": 100,
                  "system": _SYS,
                  "tools": [{"name": "Bash", "description": "run a command",
                             "input_schema": {"type": "object", "properties": {}}}],
                  "messages": [{"role": "user", "content": "hi"}]},
            headers={"x-api-key": "sk", "X-Agent-ID": "claude-code",
                     "X-Session-ID": session})
    return client.post(
        "/v1/responses",
        json={"model": "gpt-x", "stream": False, "instructions": _SYS,
              "tools": [{"type": "function", "name": "exec_command",
                         "description": "run a command",
                         "parameters": {"type": "object", "properties": {}}}],
              "input": "hi"},
        headers={"Authorization": "Bearer sk", "X-Agent-ID": "codex",
                 "X-Session-ID": session})


def _name(tool: dict) -> str:
    return str((tool.get("function") or tool).get("name", ""))


PROTOCOLS = ("chat", "messages", "responses")


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_forwarded_tool_face_has_no_flash_read(monkeypatch, protocol):
    _modules_on(monkeypatch)
    seen: list[list[dict]] = []

    async def fake(model, messages, stream, **kw):
        seen.append(list(kw.get("tools") or []))
        return _Resp(_Msg(content="ok"))

    with patch("bladex_proxy.route.router_sdk.acompletion", new=fake):
        with TestClient(create_app(_config())) as client:
            r = _post(client, protocol, f"s-f5-{protocol}")
    assert r.status_code == 200, r.text
    assert seen, "上游没被调用 ⇒ 这条判据空跑"
    forwarded = seen[0]
    names = [_name(t) for t in forwarded]
    assert RETIRED not in names
    # 工具面确实注了（否则"没有 flash_read"是空真）
    assert "bladex_memory_search" in names
    # 其余 bladex 工具与源头 schema 逐字相同、顺序不变（D3：追加末尾、不重排）
    ours = [t for t in forwarded if _name(t).startswith("bladex_")]
    assert ours == TOOL_SCHEMAS


def test_not_in_names_registry_or_anthropic_form():
    assert RETIRED not in TOOL_NAMES
    assert RETIRED not in AgencyRuntime(index=None).toolface.registered
    assert RETIRED not in {t["name"] for t in anthropic_tool_schemas()}
    # 运行时注册表与 schema 清单一一对应（防"schema 撤了、handler 还挂着"或反之）
    assert set(AgencyRuntime(index=None).toolface.registered) == set(TOOL_NAMES)


def test_stale_call_gets_unknown_tool_error_not_an_exception():
    out = asyncio.run(AgencyRuntime(index=None).toolface.dispatch(
        RETIRED, {"section": "overview"}, allowed_exposure="public",
        context={"agent_id": "probe"}))
    assert out.startswith("Error: unknown BladeX tool")
    assert RETIRED in out


def test_stale_call_through_the_endpoint_is_200(monkeypatch):
    """老会话里的模型仍调 `bladex_flash_read` ⇒ 内循环回错误文本、模型给最终答复；
    agent 看不到这个调用，端点不 500。"""
    _modules_on(monkeypatch)
    calls = {"n": 0}

    async def fake(model, messages, stream, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return _Resp(_Msg(tool_calls=[{"id": "b1", "type": "function",
                                           "function": {"name": RETIRED,
                                                        "arguments": "{}"}}]))
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        assert tool_msgs and "unknown BladeX tool" in str(tool_msgs[-1].get("content"))
        return _Resp(_Msg(content="final"))

    with patch("bladex_proxy.route.router_sdk.acompletion", new=fake):
        with TestClient(create_app(_config())) as client:
            r = _post(client, "chat", "s-f5-stale")
    assert r.status_code == 200, r.text
    msg = r.json()["choices"][0]["message"]
    assert msg["content"] == "final"
    assert not msg.get("tool_calls")
