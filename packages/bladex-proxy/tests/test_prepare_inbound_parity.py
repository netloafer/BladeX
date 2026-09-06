"""MQ-P7：入站准备（回流剥离+拼接恢复）三端点对齐。

此前 `agency.prepare_inbound` 只接在 chat 端点（V-P5a 接的，V-P5b 给两协议
开闸时工具面/账本块跟过去了、入站准备没跟）——MQ-P4 同型第四例。

判据取 A 的判据原文（工程规范 §3.2b）：回流样本用 `inject.MEMORY_OPEN` 真标记
构造——fixture 里放的是 BladeX 自己的注入块回声（第二必答问题：生产里这个位置
真实长什么样？CC 实测 119/120 会把注入块以 user 角色回传）。

断言是行为级：转发到上游的 user 角色消息里回声必须已被剥净。
（注入平面自己新加的 `<bladex-memory>` 块是 system 角色，不在断言范围。）
"""

from __future__ import annotations

import tempfile
from unittest.mock import patch

from bladex_proxy.config import ProxyConfig
from bladex_proxy.inject import MEMORY_CLOSE, MEMORY_OPEN
from bladex_proxy.server import create_app
from fastapi.testclient import TestClient


def _make_config() -> ProxyConfig:
    tmpdir = tempfile.mkdtemp()
    return ProxyConfig(
        hard_rules=["MUST be polite"],
        upstream_model="openai/test",
        upstream_api_key="sk-fake",
        rocksdb_path=f"{tmpdir}/rocksdb",
    )


class _Msg:
    def __init__(self, content="ok"):
        self.content = content
        self.tool_calls = None

    def model_dump(self):
        return {"role": "assistant", "content": self.content}


class _Resp:
    def __init__(self):
        m = _Msg()
        self.choices = [type("C", (), {"message": m, "finish_reason": "stop"})()]
        self._m = m

    def model_dump(self):
        return {"id": "chatcmpl-x", "choices": [
            {"index": 0, "message": self._m.model_dump(), "finish_reason": "stop"}]}


#: 回流样本 = 我们自己的注入块回声（真标记，非手写仿品）
_ECHO = f"{MEMORY_OPEN}\n- stale memory line\n{MEMORY_CLOSE}\n\n真正的问题在这里"

_HDRS = {"Authorization": "Bearer sk", "X-Agent-ID": "parity-agent",
         "X-Session-ID": "s-parity"}


def _forwarded_user_texts(seen: list[list[dict]]) -> list[str]:
    out = []
    for messages in seen:
        for m in messages:
            if m.get("role") != "user":
                continue
            c = m.get("content")
            if isinstance(c, str):
                out.append(c)
            elif isinstance(c, list):
                out.extend(str(p.get("text", "")) for p in c if isinstance(p, dict))
    return out


def _run(endpoint_post, monkeypatch):
    monkeypatch.setenv("BLADEX_MODULE_TOOLFACE", "1")
    monkeypatch.setenv("BLADEX_MODULE_INTERCEPTION", "1")
    app = create_app(_make_config())
    seen: list[list[dict]] = []

    async def fake(model, messages, stream, **kw):
        seen.append([dict(m) for m in messages])
        return _Resp()

    with patch("bladex_proxy.route.router_sdk.acompletion", new=fake):
        with TestClient(app) as client:
            r = endpoint_post(client)
    assert r.status_code == 200, r.text
    users = _forwarded_user_texts(seen)
    assert users, "上游必须收到过 user 消息（阳性对照：真正的问题仍在）"
    assert any("真正的问题在这里" in u for u in users), "剥离不许连正文一起吞"
    offenders = [u for u in users if MEMORY_OPEN in u]
    assert not offenders, (
        "回流的注入块以 user 角色到达上游 = 入站准备没接（MQ-P7）；"
        f"残留 {len(offenders)} 条")
    return seen


def test_chat_endpoint_strips_inbound_echo(monkeypatch):
    def post(client):
        return client.post("/v1/chat/completions", headers=_HDRS, json={
            "model": "gpt-4o", "stream": False,
            "messages": [{"role": "system", "content": "You are a test agent."},
                         {"role": "user", "content": _ECHO}]})
    _run(post, monkeypatch)


def test_anthropic_endpoint_strips_inbound_echo(monkeypatch):
    def post(client):
        return client.post("/v1/messages", headers=_HDRS, json={
            "model": "claude-x", "max_tokens": 100, "stream": False,
            "system": "You are a test agent.",
            "messages": [{"role": "user", "content": _ECHO}]})
    _run(post, monkeypatch)


def test_responses_endpoint_strips_inbound_echo(monkeypatch):
    def post(client):
        return client.post("/v1/responses", headers=_HDRS, json={
            "model": "gpt-4o", "stream": False,
            "instructions": "You are a test agent.",
            "input": _ECHO})
    _run(post, monkeypatch)
