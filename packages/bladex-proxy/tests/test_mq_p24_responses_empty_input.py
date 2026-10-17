"""MQ-P24（批 Q3A）：`/v1/responses` 的 `input` 一条消息都没归一出来 ⇒ 400 快返，不转发上游。

病例（2026-09-09 冒烟）：item 少写 `"type":"message"` ⇒ 走"未知 item 类型"分支返回 `[]` ⇒
messages 空 ⇒ 照常发给上游 ⇒ 用户读到上游那句 "messages … minimum length 1"。
BladeX 手里有全部信息却只打了一条 warning。修后：400、`param=input`、点名收到的 item type。
非空请求逐字不变（上游照常被调一次、收到的 messages 与修前同一个归一函数的产出相同）。
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import patch

import pytest
from bladex_proxy.config import ProxyConfig
from bladex_proxy.responses import empty_input_error, parse_responses_request
from bladex_proxy.server import create_app
from fastapi.testclient import TestClient


class _Msg:
    content = "ok"
    tool_calls = None

    def model_dump(self):
        return {"role": "assistant", "content": "ok"}


class _Resp:
    def __init__(self):
        self.choices = [type("C", (), {"message": _Msg(), "finish_reason": "stop"})()]
        self.usage = None

    def model_dump(self):
        return {"id": "chatcmpl-x", "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "ok"},
             "finish_reason": "stop"}]}


def _post(body: dict) -> tuple[int, dict, list[list[dict]]]:
    calls: list[list[dict]] = []

    async def fake(model, messages, stream, **kw):  # noqa: ANN001
        calls.append(messages)
        return _Resp()

    prev = os.environ.get("BLADEX_AGENT_REGISTRY_CACHE")
    os.environ["BLADEX_AGENT_REGISTRY_CACHE"] = f"{tempfile.mkdtemp()}/agent_registry.json"
    try:
        app = create_app(ProxyConfig(hard_rules=[], upstream_model="openai/test",
                                     upstream_api_key="sk-fake",
                                     rocksdb_path=f"{tempfile.mkdtemp()}/rocksdb"))
        with patch("bladex_proxy.route.router_sdk.acompletion", new=fake), TestClient(app) as c:
            r = c.post("/v1/responses", json={"model": "openai/test", "stream": False, **body},
                       headers={"Authorization": "Bearer sk", "X-Agent-ID": "p24-agent",
                                "X-Session-ID": "s-p24"})
    finally:
        if prev is None:
            os.environ.pop("BLADEX_AGENT_REGISTRY_CACHE", None)
        else:
            os.environ["BLADEX_AGENT_REGISTRY_CACHE"] = prev
    return r.status_code, r.json(), calls


# 冒烟原样：少写 "type":"message" 的 item；以及只有未知类型、空数组、带 instructions 的变体
_EMPTY = {
    "smoke_missing_type": {"input": [{"role": "user", "content": "hi"}]},
    "unknown_types": {"instructions": "You are a test agent.",
                      "input": [{"type": "reasoning", "summary": []},
                                {"type": "web_search_call", "id": "ws_1"}]},
    "empty_list": {"input": []},
}


@pytest.mark.parametrize("name", sorted(_EMPTY))
def test_empty_normalized_input_is_rejected_before_upstream(name):
    code, data, calls = _post(_EMPTY[name])
    assert code == 400, data
    assert calls == [], "空对话不许发给上游"
    err = data["error"]
    assert err["type"] == "invalid_request_error" and err["param"] == "input"
    assert "No message could be built from `input`" in err["message"]
    if name == "unknown_types":
        assert "['reasoning', 'web_search_call']" in err["message"], "要点名收到的 item type"
    if name == "smoke_missing_type":
        assert "['None']" in err["message"]


@pytest.mark.parametrize("body", [
    {"input": "hi"},
    {"instructions": "You are a test agent.",
     "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
               {"type": "reasoning", "summary": []}]},
], ids=["string_input", "message_plus_unknown"])
def test_non_empty_request_unchanged(body):
    code, data, calls = _post(body)
    assert code == 200, data
    assert len(calls) == 1
    expected, _extra = parse_responses_request({"model": "openai/test", "stream": False, **body})
    # 上游收到的对话以归一产出为骨架（注入面只会在其上追加），用户那条逐字在场
    user_msgs = [m for m in calls[0] if m.get("role") == "user"]
    assert any("hi" in str(m.get("content")) for m in user_msgs)
    assert empty_input_error(body, expected) is None
