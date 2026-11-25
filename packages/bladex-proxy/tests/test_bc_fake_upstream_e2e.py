"""批 BC：`/v1/responses` 端到端（真 LiteLLM + 本地假上游）。

丢参发生在 LiteLLM 内部（`drop_params`），mock 掉 `router_sdk.acompletion` 的用例看不见，
所以这里的出站断言读的是**假上游收到的 HTTP 请求体**。假上游只监听 127.0.0.1，不碰真实上游。

| 组 | 判据 |
|---|---|
| 请求侧 | 出站 `reasoning_effort` == 入站 `effort` 逐字；不带 `reasoning` 不多键；私有键不出站 |
| 阴性对照 | 不带私有键时 LiteLLM 仍丢；chat / messages 路径仍丢（MQ-P43 另案）|
| failover | 按路由后的模型判定：主 off 备没 off ⇒ 只有第二次带 |
| 上游错误 | 400 原样回 400 + 指向配置；5xx 仍 502 |
| 响应侧 | `</think>` 四种形态 × 流式 / 非流式：SSE、done、快照、Hub 代码区外标记 = 0，字面量保留 |
| 内循环 / 混合轮 | 最终回复剥离、custom 还原；`bladex_*` + `apply_patch` 同轮走剥离-拼接 |
"""

from __future__ import annotations

import json
import re
import threading
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
import structlog.testing
from bladex_proxy.capture import THINK_CLOSE, THINK_OPEN
from bladex_proxy.config import ProxyConfig
from bladex_proxy.models import Turn
from bladex_proxy.route import RESPONSES_REASONING_OFF_KWARG, ModelRoute, call_model
from bladex_proxy.server import create_app, endpoints_responses
from bladex_proxy.storage.memory_hub import MemoryHub
from fastapi.testclient import TestClient

PATCH = "*** Begin Patch\n*** Add File: a.txt\n+hi\n*** End Patch\n"
APPLY_PATCH = {"type": "custom", "name": "apply_patch", "description": "Edit files.",
               "format": {"type": "grammar", "syntax": "lark", "definition": "start: x"}}
HEADERS = {"Authorization": "Bearer sk", "X-Agent-ID": "bc-agent", "X-Session-ID": "s-bc"}


# ── 假上游 ───────────────────────────────────────────────────────────────────


@dataclass
class Reply:
    """假上游的一次回复。`pieces` = 流式时的正文切片（非流式拼起来）。"""
    pieces: list[str] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    reasoning: str = ""
    status: int = 200
    error: str = ""


def _tool_call(call_id: str, name: str, arguments: str) -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


class FakeUpstream:
    """OpenAI chat 形状的本地上游：录请求体，按脚本回复。"""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.script: list[Reply] = []
        self.behavior: Callable[[dict], Reply] | None = None
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a: Any) -> None:
                pass

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(body)
                reply = outer._next(body)
                if reply.status != 200:
                    kind = "invalid_request_error" if reply.status == 400 else "server_error"
                    self._send(reply.status, json.dumps({"error": {
                        "message": reply.error, "type": kind}}).encode())
                elif body.get("stream"):
                    self._stream(reply)
                else:
                    self._send(200, json.dumps(outer._completion(reply)).encode())

            def _send(self, status: int, data: bytes) -> None:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _stream(self, reply: Reply) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for chunk in outer._chunks(reply):
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        # poll_interval：`shutdown()` 要等 serve_forever 下一次轮询，默认 0.5 s，43 个用例白等 21 s（批 GT #3）
        threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.01},
                         daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def _next(self, body: dict) -> Reply:
        if self.behavior is not None:
            return self.behavior(body)
        return self.script.pop(0) if self.script else Reply(pieces=["ok"])

    @staticmethod
    def _completion(reply: Reply) -> dict:
        message: dict[str, Any] = {"role": "assistant", "content": "".join(reply.pieces) or None}
        if reply.reasoning:
            message["reasoning_content"] = reply.reasoning
        if reply.tool_calls:
            message["tool_calls"] = reply.tool_calls
        return {"id": "chatcmpl-x", "object": "chat.completion", "created": 1, "model": "m",
                "choices": [{"index": 0, "message": message,
                             "finish_reason": "tool_calls" if reply.tool_calls else "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}

    @staticmethod
    def _chunks(reply: Reply) -> Iterator[dict]:
        def chunk(delta: dict, finish: str | None = None) -> dict:
            return {"id": "chatcmpl-x", "object": "chat.completion.chunk", "created": 1, "model": "m",
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        if reply.reasoning:
            yield chunk({"role": "assistant", "reasoning_content": reply.reasoning})
        for piece in reply.pieces:
            yield chunk({"role": "assistant", "content": piece})
        for i, tc in enumerate(reply.tool_calls):
            yield chunk({"tool_calls": [{"index": i, **tc}]})
        yield chunk({}, "tool_calls" if reply.tool_calls else "stop")


@pytest.fixture
def upstream(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeUpstream]:
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    server = FakeUpstream()
    yield server
    server.close()


class Harness:
    """真 app + 真 LiteLLM + 假上游；入库换成「构造 Turn → 真 MemoryHub 读回」。"""

    def __init__(self, client: TestClient, hub: MemoryHub) -> None:
        self.client = client
        self.hub = hub
        self.turns: list[Turn] = []

    def post(self, body: dict, path: str = "/v1/responses") -> Any:
        return self.client.post(path, json={"model": "openai/test", **body}, headers=HEADERS)


@pytest.fixture
def app_factory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                upstream: FakeUpstream) -> Iterator[Callable[..., Harness]]:
    opened: list[tuple[TestClient, MemoryHub]] = []

    def _make(routing_toml: str = "", interception: bool = False) -> Harness:
        monkeypatch.setenv("BLADEX_AGENT_REGISTRY_CACHE", str(tmp_path / "agent_registry.json"))
        monkeypatch.setenv("BLADEX_MODULE_INTERCEPTION", "1" if interception else "0")
        monkeypatch.setenv("BLADEX_MODULE_TOOLFACE", "1" if interception else "0")
        monkeypatch.setenv("BLADEX_STREAM_KEEPALIVE_S", "0")
        monkeypatch.setenv("BLADEX_NARRATE_INTERCEPT", "0")
        routing = tmp_path / "routing.toml"
        routing.write_text(routing_toml, "utf-8")
        app = create_app(ProxyConfig(
            hard_rules=[], upstream_model="openai/test", upstream_api_key="sk-fake",
            upstream_api_base=upstream.url, rocksdb_path=str(tmp_path / "rocksdb"),
            routing_config_path=str(routing)))
        hub = MemoryHub(tmp_path / "bc_hub")
        hub.open()
        client = TestClient(app)
        client.__enter__()
        harness = Harness(client, hub)

        async def _enqueue(_request: Any, identity: Any, model: str, _messages: Any, _injected: str,
                           response_text: str, tool_events: list, status: Any, error: str,
                           *_a: Any, response_meta: Any = None, **_kw: Any) -> None:
            key = f"{identity.session_prefix()}{len(harness.turns):04d}-0"
            hub.put(key, Turn(identity=identity, model=model, response_text=response_text,
                              tool_events=tool_events, status=status, error=error,
                              **({"response_meta": response_meta} if response_meta else {})))
            harness.turns.append(hub.get(key))

        monkeypatch.setattr(endpoints_responses, "_enqueue_turn", _enqueue)
        monkeypatch.setattr(endpoints_responses, "_enqueue_turn_shielded", _enqueue)
        opened.append((client, hub))
        return harness

    yield _make
    for client, hub in opened:
        client.__exit__(None, None, None)
        hub.close()


def _sse(response: Any) -> list[dict]:
    events: list[dict] = []
    for block in response.text.split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: "):]))
    return events


def _outside_code(text: str) -> str:
    """独立于被测状态机的尺子：遮掉围栏与同一行内的行内代码。"""
    text = re.sub(r"^ {0,3}(`{3,})[^\n]*\n.*?(?:^ {0,3}\1`*[ \t]*$|\Z)", "", text,
                  flags=re.DOTALL | re.MULTILINE)
    return re.sub(r"(`+)[^`\n]*?\1", "", text)


def _no_marker(text: str) -> bool:
    outside = _outside_code(text)
    return THINK_CLOSE not in outside and THINK_OPEN not in outside


def _completed(events: list[dict]) -> dict:
    final = [e for e in events if e.get("type") == "response.completed"]
    assert len(final) == 1, [e.get("type") for e in events]
    return final[0]["response"]


def _message_texts(output: list[dict]) -> list[str]:
    return ["".join(b["text"] for b in i["content"]) for i in output if i["type"] == "message"]


# ── ① 请求侧：出站逐字 ────────────────────────────────────────────────────────


@pytest.mark.parametrize("effort", ["high", "low", "medium", "none", "max", "weird-custom"])
@pytest.mark.parametrize("stream", [True, False], ids=["stream", "nonstream"])
def test_effort_reaches_upstream_verbatim(app_factory: Any, upstream: FakeUpstream,
                                          effort: str, stream: bool) -> None:
    r = app_factory().post({"stream": stream, "input": "hi",
                            "reasoning": {"effort": effort, "summary": "detailed", "context": "x"}})
    assert r.status_code == 200, r.text
    sent = upstream.requests[-1]
    assert sent["reasoning_effort"] == effort
    assert not [k for k in sent if k.startswith("bladex") or k in (
        "allowed_openai_params", "reasoning", "summary", "context")], sorted(sent)


@pytest.mark.parametrize("stream", [True, False], ids=["stream", "nonstream"])
def test_no_reasoning_request_sends_no_new_keys(app_factory: Any, upstream: FakeUpstream,
                                                stream: bool) -> None:
    r = app_factory().post({"stream": stream, "input": "hi", "tools": [
        {"type": "function", "name": "exec_command", "parameters": {"type": "object"}}]})
    assert r.status_code == 200, r.text
    sent = upstream.requests[-1]
    expected = {"model", "messages", "tools"} | ({"stream", "stream_options"} if stream else set())
    assert set(sent) == expected


@pytest.mark.asyncio
async def test_litellm_drops_without_private_key_and_forwards_with_it(upstream: FakeUpstream) -> None:
    """阴性对照：私有键是唯一的放行口子。这条红 = 假上游根本量不到丢参。"""
    route = ModelRoute(model="openai/test", api_base=upstream.url, api_key="sk-fake")
    msgs = [{"role": "user", "content": "hi"}]
    await call_model("openai/test", msgs, stream=False, route=route, reasoning_effort="high")
    assert "reasoning_effort" not in upstream.requests[-1]
    await call_model("openai/test", msgs, stream=False, route=route, reasoning_effort="high",
                     **{RESPONSES_REASONING_OFF_KWARG: frozenset()})
    assert upstream.requests[-1]["reasoning_effort"] == "high"


def test_chat_path_still_dropped(app_factory: Any, upstream: FakeUpstream) -> None:
    h = app_factory()
    r = h.post({"messages": [{"role": "user", "content": "hi"}], "stream": False,
                "reasoning_effort": "high"}, path="/v1/chat/completions")
    assert r.status_code == 200, r.text
    assert upstream.requests, "请求要真的到上游，否则这条是空断言"
    assert "reasoning_effort" not in upstream.requests[-1]
    assert "allowed_openai_params" not in upstream.requests[-1]


def test_messages_path_not_affected(app_factory: Any, upstream: FakeUpstream) -> None:
    h = app_factory()
    r = h.post({"messages": [{"role": "user", "content": "hi"}], "stream": False,
                "max_tokens": 5, "reasoning": {"effort": "high"}}, path="/v1/messages")
    assert r.status_code == 200, r.text
    assert upstream.requests
    assert "reasoning_effort" not in upstream.requests[-1]


# ── ② 配置与不转发的告警 ──────────────────────────────────────────────────────

_OFF_TOML = """
[[models]]
name = "openai/test"
reasoning = "off"
"""


def test_off_model_not_sent_and_warns_every_round(app_factory: Any, upstream: FakeUpstream) -> None:
    h = app_factory(routing_toml=_OFF_TOML)
    with structlog.testing.capture_logs() as logs:
        h.post({"stream": False, "input": "hi", "reasoning": {"effort": "high"}})
        h.post({"stream": False, "input": "hi", "reasoning": {"effort": "low"}})
    assert all("reasoning_effort" not in sent for sent in upstream.requests)
    hits = [x for x in logs if x["event"] == "reasoning_not_forwarded"]
    assert [(x["effort"], x["reason"], x["model"]) for x in hits] == [
        ("high", "config", "openai/test"), ("low", "config", "openai/test")]
    assert "routing.toml" in hits[0]["hint"]


def test_off_config_only_affects_named_model(app_factory: Any, upstream: FakeUpstream) -> None:
    h = app_factory(routing_toml='[[models]]\nname = "openai/other"\nreasoning = "off"\n')
    h.post({"stream": False, "input": "hi", "reasoning": {"effort": "high"}})
    assert upstream.requests[-1]["reasoning_effort"] == "high"


@pytest.mark.asyncio
async def test_non_openai_prefix_treated_as_off_with_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    async def fake(**kw: Any) -> str:
        seen.update(kw)
        return "resp"

    monkeypatch.setattr("bladex_proxy.route.router_sdk.acompletion", fake)
    with structlog.testing.capture_logs() as logs:
        await call_model("anthropic/x", [{"role": "user", "content": "hi"}], stream=False,
                         reasoning_effort="high", **{RESPONSES_REASONING_OFF_KWARG: frozenset()})
    assert "reasoning_effort" not in seen and "allowed_openai_params" not in seen
    assert RESPONSES_REASONING_OFF_KWARG not in seen
    assert [x["reason"] for x in logs if x["event"] == "reasoning_not_forwarded"] == ["provider_prefix"]


def test_routing_config_rejects_unknown_reasoning_value(tmp_path: Path) -> None:
    from bladex_proxy.routing_config import RoutingConfig, RoutingConfigError
    path = tmp_path / "r.toml"
    path.write_text('[[models]]\nname = "openai/a"\nreasoning = "on"\n', "utf-8")
    with pytest.raises(RoutingConfigError, match="reasoning"):
        RoutingConfig.from_toml(str(path))


def test_routing_config_off_models(tmp_path: Path) -> None:
    from bladex_proxy.routing_config import RoutingConfig
    path = tmp_path / "r.toml"
    path.write_text('[[models]]\nname = "openai/a"\nreasoning = "off"\n'
                    '[[models]]\nname = "openai/b"\n', "utf-8")
    assert RoutingConfig.from_toml(str(path)).reasoning_off_models() == frozenset({"openai/a"})


# ── ③ failover：按路由后的模型判定 ────────────────────────────────────────────


def _failing_primary(body: dict) -> Reply:
    return (Reply(status=503, error="busy") if body["model"] == "a"
            else Reply(pieces=["ok"]))


@pytest.mark.asyncio
@pytest.mark.parametrize(("off", "expected"), [
    (frozenset({"openai/a"}), {"a": None, "b": "high"}),
    (frozenset({"openai/b"}), {"a": "high", "b": None}),
    (frozenset(), {"a": "high", "b": "high"}),
], ids=["primary-off", "backup-off", "none-off"])
async def test_failover_decides_per_attempt(upstream: FakeUpstream, off: frozenset[str],
                                            expected: dict[str, str | None]) -> None:
    upstream.behavior = _failing_primary
    primary = ModelRoute(model="openai/a", api_base=upstream.url, api_key="sk-fake")
    backup = ModelRoute(model="openai/b", api_base=upstream.url, api_key="sk-fake")
    resp = await call_model("openai/a", [{"role": "user", "content": "hi"}], stream=False,
                            route=primary, failover=[backup], reasoning_effort="high",
                            **{RESPONSES_REASONING_OFF_KWARG: off})
    assert resp.choices[0].message.content == "ok"
    assert {r["model"] for r in upstream.requests} == {"a", "b"}
    for sent in upstream.requests:
        assert sent.get("reasoning_effort") == expected[sent["model"]], sent["model"]


# ── ④ 上游错误 ───────────────────────────────────────────────────────────────

_NONE_REJECTED = "reasoning_effort `none` is not supported by this model"


@pytest.mark.parametrize("stream", [True, False], ids=["stream", "nonstream"])
def test_upstream_400_on_reasoning_passes_through_and_points_to_config(
        app_factory: Any, upstream: FakeUpstream, stream: bool) -> None:
    h = app_factory()
    upstream.script.append(Reply(status=400, error=_NONE_REJECTED))
    with structlog.testing.capture_logs() as logs:
        r = h.post({"stream": stream, "input": "hi", "reasoning": {"effort": "none"}})
    assert r.status_code == 400, r.text
    assert _NONE_REJECTED in r.json()["error"]["message"]
    assert r.json()["error"]["type"] == "invalid_request_error"
    hits = [x for x in logs if x["event"] == "responses_reasoning_rejected"]
    assert len(hits) == 1
    assert hits[0]["effort"] == "none" and hits[0]["model"] == "openai/test"
    assert "routing.toml" in hits[0]["hint"]
    assert len(upstream.requests) == 1, "不自动重试、不改值、不降档"
    assert h.turns[-1].error, "失败轮次仍入库"


def test_unrelated_400_passes_through_without_reasoning_alarm(app_factory: Any,
                                                              upstream: FakeUpstream) -> None:
    h = app_factory()
    upstream.script.append(Reply(status=400, error="context length exceeded"))
    with structlog.testing.capture_logs() as logs:
        r = h.post({"stream": False, "input": "hi", "reasoning": {"effort": "high"}})
    assert r.status_code == 400
    assert not [x for x in logs if x["event"] == "responses_reasoning_rejected"]


def test_400_without_forwarded_effort_has_no_alarm(app_factory: Any, upstream: FakeUpstream) -> None:
    h = app_factory()
    upstream.script.append(Reply(status=400, error="reasoning is broken but we never sent it"))
    with structlog.testing.capture_logs() as logs:
        r = h.post({"stream": False, "input": "hi"})
    assert r.status_code == 400
    assert not [x for x in logs if x["event"] == "responses_reasoning_rejected"]


def test_upstream_5xx_stays_502(app_factory: Any, upstream: FakeUpstream) -> None:
    h = app_factory()
    upstream.behavior = lambda _b: Reply(status=500, error="boom")
    r = h.post({"stream": False, "input": "hi", "reasoning": {"effort": "high"}})
    assert r.status_code == 502
    assert r.json()["error"]["type"] == "upstream_error"


# ── ⑤ 响应侧：四种形态 ───────────────────────────────────────────────────────

_FENCE = f"```xml\n{THINK_CLOSE}\n```"
SHAPES: dict[str, dict[str, Any]] = {
    "tail": {"pieces": ["思考散文。", THINK_CLOSE, "\n\n"], "clean": "", "prefix": "思考散文。"},
    "middle": {"pieces": ["想一下。", THINK_CLOSE, "正文结果。"], "clean": "正文结果。",
               "prefix": "想一下。"},
    "split": {"pieces": ["想一下。" + THINK_CLOSE[:3], THINK_CLOSE[3:6], THINK_CLOSE[6:] + "正文结果。"],
              "clean": "正文结果。", "prefix": "想一下。"},
    "code": {"pieces": ["默认先 `", THINK_OPEN, "` 再回答，`", THINK_CLOSE, "` 结束。\n", _FENCE],
             "clean": f"默认先 `{THINK_OPEN}` 再回答，`{THINK_CLOSE}` 结束。\n{_FENCE}", "prefix": ""},
}


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_injected_shapes_stream(app_factory: Any, upstream: FakeUpstream, shape: str) -> None:
    case = SHAPES[shape]
    h = app_factory()
    upstream.script.append(Reply(pieces=case["pieces"]))
    with structlog.testing.capture_logs() as logs:
        r = h.post({"stream": True, "input": "hi", "reasoning": {"effort": "high"}})
    assert r.status_code == 200, r.text
    events = _sse(r)
    deltas = "".join(e["delta"] for e in events if e.get("type") == "response.output_text.delta")
    done = [e["item"] for e in events if e.get("type") == "response.output_item.done"
            and e["item"]["type"] == "message"]
    parts = [e["part"]["text"] for e in events if e.get("type") == "response.content_part.done"]
    final = _completed(events)
    hub_turn = h.turns[-1]
    surfaces = {"deltas": deltas, "completed.output_text": final["output_text"],
                "hub": hub_turn.response_text,
                **{f"done[{i}]": t for i, t in enumerate(_message_texts(done))},
                **{f"part[{i}]": t for i, t in enumerate(parts)},
                **{f"snapshot[{i}]": t for i, t in enumerate(_message_texts(final["output"]))}}
    for where, text in surfaces.items():
        assert _no_marker(text), f"{where} 代码区外还有标记：{text!r}"
    assert final["output_text"] == hub_turn.response_text == case["clean"]
    assert _message_texts(done) == _message_texts(final["output"]) == [case["clean"]]
    stripped = [x for x in logs if x["event"] == "inline_reasoning_stripped"]
    if case["prefix"]:
        assert stripped, "注入组要证明真的走到了剥离，不是没碰上"
        assert hub_turn.response_meta.reasoning_text == case["prefix"]
    else:
        assert not stripped
        assert deltas == case["clean"], "代码区内字面量逐字保留"
        assert not hub_turn.response_meta.reasoning_text


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_injected_shapes_nonstream(app_factory: Any, upstream: FakeUpstream, shape: str) -> None:
    case = SHAPES[shape]
    h = app_factory()
    upstream.script.append(Reply(pieces=case["pieces"]))
    with structlog.testing.capture_logs() as logs:
        r = h.post({"stream": False, "input": "hi"})
    assert r.status_code == 200, r.text
    data = r.json()
    hub_turn = h.turns[-1]
    assert data["output_text"] == hub_turn.response_text == case["clean"]
    assert _message_texts(data["output"]) == ([case["clean"]] if case["clean"] else [])
    assert _no_marker(data["output_text"])
    stripped = [x for x in logs if x["event"] == "inline_reasoning_stripped"]
    if case["prefix"]:
        assert [x["path"] for x in stripped] == ["responses:non_stream"]
        assert hub_turn.response_meta.reasoning_text == case["prefix"]
    else:
        assert not stripped
        assert not hub_turn.response_meta.reasoning_text


def test_structured_and_inline_reasoning_merge_in_hub(app_factory: Any, upstream: FakeUpstream) -> None:
    h = app_factory()
    upstream.script.append(Reply(reasoning="结构化", pieces=["前缀", THINK_CLOSE, "答"]))
    h.post({"stream": False, "input": "hi"})
    assert h.turns[-1].response_meta.reasoning_text == "结构化\n前缀"


def test_reasoning_item_end_to_end_only_with_summary(app_factory: Any, upstream: FakeUpstream) -> None:
    h = app_factory()
    upstream.script.append(Reply(reasoning="想", pieces=["答"]))
    upstream.script.append(Reply(reasoning="想", pieces=["答"]))
    plain = _sse(h.post({"stream": True, "input": "hi", "reasoning": {"effort": "high"}}))
    with_item = _sse(h.post({"stream": True, "input": "hi",
                             "reasoning": {"effort": "high", "summary": "detailed"}}))
    assert [i["type"] for i in _completed(plain)["output"]] == ["message"]
    assert [i["type"] for i in _completed(with_item)["output"]] == ["reasoning", "message"]
    assert "reasoning_summary" not in json.dumps(upstream.requests[-1])


# ── ⑥ apply_patch 端到端 ─────────────────────────────────────────────────────


def _patch_call(call_id: str = "call_p") -> dict:
    return _tool_call(call_id, "apply_patch", json.dumps({"input": PATCH}))


def test_apply_patch_full_round_trip(app_factory: Any, upstream: FakeUpstream) -> None:
    h = app_factory()
    upstream.script.append(Reply(tool_calls=[_patch_call()]))
    first = _sse(h.post({"stream": True, "input": "改文件", "tools": [APPLY_PATCH]}))
    declared = upstream.requests[0]["tools"][0]["function"]
    assert declared["name"] == "apply_patch"
    assert declared["parameters"]["required"] == ["input"]
    item = _completed(first)["output"][0]
    assert (item["type"], item["call_id"], item["input"]) == ("custom_tool_call", "call_p", PATCH)
    assert h.turns[-1].tool_events[0].arguments == {"input": PATCH}

    h.post({"stream": True, "tools": [APPLY_PATCH], "input": [
        {"type": "message", "role": "user", "content": "改文件"}, item,
        {"type": "custom_tool_call_output", "call_id": "call_p", "output": "Success"}]})
    messages = upstream.requests[1]["messages"]
    assistant = [m for m in messages if m.get("tool_calls")]
    assert assistant[0]["tool_calls"][0]["id"] == "call_p"
    assert json.loads(assistant[0]["tool_calls"][0]["function"]["arguments"]) == {"input": PATCH}
    tool = [m for m in messages if m["role"] == "tool"]
    assert (tool[0]["tool_call_id"], tool[0]["content"]) == ("call_p", "Success")


def test_apply_patch_nonstream(app_factory: Any, upstream: FakeUpstream) -> None:
    h = app_factory()
    upstream.script.append(Reply(tool_calls=[_patch_call()]))
    data = h.post({"stream": False, "input": "改", "tools": [APPLY_PATCH]}).json()
    assert data["output"][0]["type"] == "custom_tool_call"
    assert data["output"][0]["input"] == PATCH
    assert h.turns[-1].tool_events[0].arguments == {"input": PATCH}


# ── ⑦ 内循环 ─────────────────────────────────────────────────────────────────


def _bladex_call(call_id: str = "call_b") -> dict:
    return _tool_call(call_id, "bladex_memory_search", json.dumps({"query": "x"}))


def test_inner_loop_final_reply_stripped_custom_restored_effort_kept(app_factory: Any,
                                                                     upstream: FakeUpstream) -> None:
    h = app_factory(interception=True)
    upstream.script.append(Reply(tool_calls=[_bladex_call()]))
    upstream.script.append(Reply(pieces=[f"想完了{THINK_CLOSE}最终答复"], tool_calls=[_patch_call()]))
    with structlog.testing.capture_logs() as logs:
        r = h.post({"stream": True, "input": "干活", "tools": [APPLY_PATCH],
                    "reasoning": {"effort": "low"}})
    assert r.status_code == 200, r.text
    assert len(upstream.requests) == 2, "纯 bladex 调用要走内循环"
    assert [s["reasoning_effort"] for s in upstream.requests] == ["low", "low"]
    assert not [k for s in upstream.requests for k in s if k.startswith("bladex")]
    events = _sse(r)
    assert THINK_CLOSE not in r.text, "这轮没有代码区，SSE 全文都不许有标记"
    final = _completed(events)
    types = [i["type"] for i in final["output"]]
    assert types == ["message", "custom_tool_call"], types
    assert _message_texts(final["output"]) == ["最终答复"]
    custom = final["output"][1]
    assert custom["input"] == PATCH and custom["call_id"] == "call_p"
    assert "inner_loop" in " ".join(
        x["path"] for x in logs if x["event"] == "inline_reasoning_stripped")
    assert h.turns[-1].response_text == "最终答复"


def test_nonstream_inner_loop_custom_restored(app_factory: Any, upstream: FakeUpstream) -> None:
    h = app_factory(interception=True)
    upstream.script.append(Reply(tool_calls=[_bladex_call()]))
    upstream.script.append(Reply(pieces=["最终"], tool_calls=[_patch_call()]))
    data = h.post({"stream": False, "input": "干活", "tools": [APPLY_PATCH]}).json()
    assert [i["type"] for i in data["output"]] == ["message", "custom_tool_call"]
    assert data["output"][1]["input"] == PATCH


# ── ⑧ 拦截混合轮：`bladex_*` + `apply_patch` 同一轮 ───────────────────────────


def test_mixed_round_forwards_custom_and_splices_on_replay(app_factory: Any,
                                                           upstream: FakeUpstream) -> None:
    h = app_factory(interception=True)
    upstream.script.append(Reply(tool_calls=[_bladex_call(), _patch_call()]))
    with structlog.testing.capture_logs() as logs:
        first = h.post({"stream": True, "input": "干活", "tools": [APPLY_PATCH]})
    assert len(upstream.requests) == 1, "有放行的 custom 调用就不该错走内循环"
    assert [x["removed"] for x in logs if x["event"] == "agency_protocol_mixed_stripped"] == [1]
    assert not [x for x in logs if x["event"] == "agency_protocol_inner_loop_done"]
    assert "bladex_memory_search" not in first.text.split("response.completed")[0].replace(
        "input", "")
    events = _sse(first)
    final = _completed(events)
    assert [i["type"] for i in final["output"]] == ["custom_tool_call"]
    item = final["output"][0]
    assert item["input"] == PATCH
    added = [e for e in events if e.get("type") == "response.output_item.added"]
    assert [e["output_index"] for e in added] == [0], "被拦 item 不占编号"

    upstream.script.append(Reply(pieces=["好了"]))
    with structlog.testing.capture_logs() as logs:
        h.post({"stream": True, "tools": [APPLY_PATCH], "input": [
            {"type": "message", "role": "user", "content": "干活"}, item,
            {"type": "custom_tool_call_output", "call_id": "call_p", "output": "Success"}]})
    calls = [tc for m in upstream.requests[1]["messages"] for tc in m.get("tool_calls") or []]
    assert {tc["function"]["name"] for tc in calls} == {"bladex_memory_search", "apply_patch"}, (
        "拼接锚含 custom 的 call_id：bladex 调用要拼回历史")
    tool_ids = Counter(m["tool_call_id"] for m in upstream.requests[1]["messages"]
                       if m["role"] == "tool")
    assert tool_ids == Counter({"call_b": 1, "call_p": 1})


def test_mixed_round_function_call_anchor_unchanged(app_factory: Any, upstream: FakeUpstream) -> None:
    """没有 custom 时拼接锚仍取剥离器（修前行为）。"""
    h = app_factory(interception=True)
    upstream.script.append(Reply(tool_calls=[_bladex_call(), _tool_call("call_e", "exec_command", "{}")]))
    tools = [{"type": "function", "name": "exec_command", "parameters": {"type": "object"}}]
    first = _sse(h.post({"stream": True, "input": "干活", "tools": tools}))
    item = _completed(first)["output"][0]
    assert item["type"] == "function_call"
    upstream.script.append(Reply(pieces=["好了"]))
    h.post({"stream": True, "tools": tools, "input": [
        {"type": "message", "role": "user", "content": "干活"}, item,
        {"type": "function_call_output", "call_id": "call_e", "output": "ok"}]})
    calls = [tc for m in upstream.requests[1]["messages"] for tc in m.get("tool_calls") or []]
    assert {tc["function"]["name"] for tc in calls} == {"bladex_memory_search", "exec_command"}
