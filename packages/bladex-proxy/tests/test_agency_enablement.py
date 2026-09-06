"""V-P5a 验收剧本：AgencyRuntime 接线端到端（chat 端点）。

覆盖：默认关零差异 / 非流式 pure 内循环 / 非流式 mixed 剥离+拼接 / tools 增补 /
入站回流剥离 / 流式 pure 合成 / 流式 mixed 剥离。开关经 monkeypatch env 打开。
"""

from __future__ import annotations

import json
import tempfile
from unittest.mock import patch

import pytest

from bladex_proxy.config import ProxyConfig
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


# ── 非流式 fakes ────────────────────────────────────────────────────────────

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
            {"index": 0, "message": self._msg.model_dump(),
             "finish_reason": "stop"}]}


def _tc(name, cid, args="{}"):
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": args}}


#: 🔴 每个请求都带 system —— 这是**真实 agent 的形状**（hermes/CC/Codex 全都发
#: system prompt，我们正是靠它做指纹识别）。2026-08-26 起「无 system + 单条 user」
#: 被判为**孤立子调用**、不给账本面（`identity.is_isolated_subcall`，MQ-L8：
#: Hermes 的视觉子调用就是这个形状，一个用户请求开出 5 张噪声账本）。
#: 测试若继续用无 system 的极简形状，测的就不是任何真实 agent 会发的请求。
_SYS = {"role": "system", "content": "You are a test agent."}


def _msgs(*users: str) -> list[dict]:
    return [_SYS, *({"role": "user", "content": u} for u in users)]


#: agent 自己带的工具面。🔴 **必须有**（MQ-L21，2026-08-27）：真实 agent
#: （hermes 35 个 / codex 13 个 / CC 30 个）每轮都带工具；不带工具的请求是
#: 客户端的**内部功能调用**（标题生成之类），现在不给账本面。
#: 此前这些用例一律不带 tools —— 测的是一个**真实不存在的形态**，
#: 新判据落地时它们集体变红，正是这个缺口的暴露。
_OWN_TOOL = {"type": "function",
             "function": {"name": "exec_command", "description": "run a command",
                          "parameters": {"type": "object", "properties": {}}}}


def _post(client, **extra):
    return client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "stream": False, "tools": [_OWN_TOOL],
              "messages": _msgs("hi"), **extra},
        headers={"Authorization": "Bearer sk", "X-Agent-ID": "test-agent",
                 "X-Session-ID": "s-agency"},
    )


def _on(monkeypatch, *names):
    for n in names:
        monkeypatch.setenv(f"BLADEX_MODULE_{n.upper()}", "1")


class TestDefaultOffZeroDiff:
    def test_bladex_call_passes_through_when_off(self, monkeypatch):
        monkeypatch.setenv("BLADEX_MODULE_INTERCEPTION", "0")  # 2026-09-02 翻默认后显式关
        app = create_app(_make_config())
        resp_msg = _Msg(content=None,
                        tool_calls=[_tc("bladex_memory_search", "b1")])

        async def fake(model, messages, stream, **kw):
            return _Resp(resp_msg)

        with patch("bladex_proxy.route.router_sdk.acompletion", new=fake):
            with TestClient(app) as client:
                r = _post(client)
        # 默认关：bladex 调用原样到达 agent（现状行为逐字不变）
        msg = r.json()["choices"][0]["message"]
        assert msg["tool_calls"][0]["function"]["name"] == "bladex_memory_search"


class TestNonStream:
    def test_pure_runs_inner_loop(self, monkeypatch):
        _on(monkeypatch, "toolface", "interception")
        app = create_app(_make_config())
        calls = {"n": 0}
        seen_msgs: list[list[dict]] = []

        async def fake(model, messages, stream, **kw):
            calls["n"] += 1
            seen_msgs.append(list(messages))
            if calls["n"] == 1:
                return _Resp(_Msg(tool_calls=[
                    _tc("bladex_memory_search", "b1", '{"query":"x"}')]))
            return _Resp(_Msg(content="基于记忆的最终回答"))

        with patch("bladex_proxy.route.router_sdk.acompletion", new=fake):
            with TestClient(app) as client:
                r = _post(client)
        msg = r.json()["choices"][0]["message"]
        assert msg["content"] == "基于记忆的最终回答"
        assert not msg.get("tool_calls")
        # 内循环第二次上游调用带了工具结果（index 不可用 → Error 文本也算结果）
        roles = [m.get("role") for m in seen_msgs[1]]
        assert "tool" in roles

    def test_mixed_strips_and_records_splice(self, monkeypatch):
        _on(monkeypatch, "toolface", "interception", "ledger")
        app = create_app(_make_config())

        async def fake(model, messages, stream, **kw):
            return _Resp(_Msg(content="", tool_calls=[
                _tc("web_search", "a1"),
                _tc("bladex_ledger_read", "b1")]))

        with patch("bladex_proxy.route.router_sdk.acompletion", new=fake):
            with TestClient(app) as client:
                r = _post(client)
                agency = client.app.state.agency
        msg = r.json()["choices"][0]["message"]
        names = [tc["function"]["name"] for tc in msg["tool_calls"]]
        assert names == ["web_search"]           # bladex 调用被剥
        recs = [rec for s in agency.splice._by_session.values() for rec in s]
        assert len(recs) == 1
        assert recs[0].calls[0]["function"]["name"] == "bladex_ledger_read"
        assert recs[0].results[0]["role"] == "tool"

    def test_toolface_appended_to_forwarded_tools(self, monkeypatch):
        _on(monkeypatch, "toolface")
        app = create_app(_make_config())
        captured_kw: dict = {}

        async def fake(model, messages, stream, **kw):
            captured_kw.update(kw)
            return _Resp(_Msg(content="ok"))

        agent_tools = [{"type": "function",
                        "function": {"name": "web_search", "parameters": {}}}]
        with patch("bladex_proxy.route.router_sdk.acompletion", new=fake):
            with TestClient(app) as client:
                _post(client, tools=agent_tools)
        names = [t["function"]["name"] for t in captured_kw.get("tools", [])]
        assert names[0] == "web_search"          # agent 自带前缀不动
        assert "bladex_memory_search" in names   # 追加在后

    def test_inbound_memory_echo_stripped(self, monkeypatch):
        _on(monkeypatch, "toolface", "interception")  # 依赖校验：interception 需 toolface
        app = create_app(_make_config())
        seen: list[dict] = []

        async def fake(model, messages, stream, **kw):
            seen.extend(messages)
            return _Resp(_Msg(content="ok"))

        with patch("bladex_proxy.route.router_sdk.acompletion", new=fake):
            with TestClient(app) as client:
                client.post(
                    "/v1/chat/completions",
                    json={"model": "gpt-4o", "stream": False, "messages": [
                        {"role": "user",
                         "content": "<bladex-memory>回流的旧注入"
                                    "</bladex-memory>真正的问题"}]},
                    headers={"Authorization": "Bearer sk",
                             "X-Agent-ID": "test-agent",
                             "X-Session-ID": "s-echo"})
        user_texts = [m["content"] for m in seen
                      if m.get("role") == "user" and isinstance(m.get("content"), str)]
        assert any("真正的问题" in t for t in user_texts)
        assert all("回流的旧注入" not in t for t in user_texts)


class TestLedgerFirstStep:
    """Jason 2026-08-25 拍板：goal 对照是模型每轮第一动作。"""

    def test_no_ledger_block_with_template_after_last_user(self, monkeypatch):
        _on(monkeypatch, "toolface", "interception", "ledger")
        app = create_app(_make_config())
        seen: list[dict] = []

        async def fake(model, messages, stream, **kw):
            seen.extend(messages)
            return _Resp(_Msg(content="ok"))

        with patch("bladex_proxy.route.router_sdk.acompletion", new=fake):
            with TestClient(app) as client:
                _post(client)
        idx = [i for i, m in enumerate(seen)
               if "<bladex-ledger>" in str(m.get("content", ""))]
        assert idx, "账本块未注入"
        block = str(seen[idx[0]]["content"])
        assert "FIRST STEP" in block and "bladex_ledger_switch" in block
        # 模板随块注入（教模型五段语义）
        assert "## Core" in block and "## Next" in block
        # 位置拍板（2026-08-25）：紧跟末条 user 之后，不被长上下文淹没
        last_user = max(i for i, m in enumerate(seen) if m.get("role") == "user")
        assert idx[0] == last_user + 1

    def test_template_env_override_and_custom_section(self, monkeypatch, tmp_path):
        """用户改模板 = 改新账本段结构（增段数据面）+ 注入内容跟随。"""
        _on(monkeypatch, "toolface", "interception", "ledger")
        tpl = tmp_path / "my-template.md"
        tpl.write_text(
            "# T\n\n## Goal\n\n_(user)_\n\n## Core\n\n_(c)_\n\n"
            "## Lessons\n\n_(why things failed)_\n\n## Next\n\n_(n)_\n",
            encoding="utf-8")
        monkeypatch.setenv("BLADEX_LEDGER_TEMPLATE", str(tpl))
        app = create_app(_make_config())
        calls = {"n": 0}
        seen: list[dict] = []

        async def fake(model, messages, stream, **kw):
            calls["n"] += 1
            seen.extend(messages)
            if calls["n"] == 1:
                return _Resp(_Msg(tool_calls=[_tc(
                    "bladex_ledger_switch", "b1", '{"ledger_id": ""}')]))
            return _Resp(_Msg(content="ok"))

        with patch("bladex_proxy.route.router_sdk.acompletion", new=fake):
            with TestClient(app) as client:
                client.post("/v1/chat/completions",
                            json={"model": "gpt-4o", "stream": False, "tools": [_OWN_TOOL], "messages": _msgs("任务丙")},
                            headers={"Authorization": "Bearer sk",
                                     "X-Agent-ID": "test-agent",
                                     "X-Session-ID": "s-tpl"})
                agency = client.app.state.agency
        blocks = [str(m.get("content", "")) for m in seen
                  if "<bladex-ledger>" in str(m.get("content", ""))]
        assert blocks and "Lessons" in blocks[0]          # 注入跟随自定义模板（启动预加载）
        led = list(agency.pool.values())[0]
        assert "lessons" in led.section_order             # 新账本带自定义段
        assert "verified" not in led.section_order        # 结构完全由模板定

    def test_switch_with_initial_entries(self, monkeypatch):
        _on(monkeypatch, "toolface", "interception", "ledger")
        app = create_app(_make_config())
        calls = {"n": 0}

        async def fake(model, messages, stream, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                return _Resp(_Msg(tool_calls=[_tc(
                    "bladex_ledger_switch", "b1",
                    '{"ledger_id": "", "title": "带初始条目",'
                    ' "core": ["repo 在 ~/dev/BladeX"],'
                    ' "next": ["先跑 gate_check", "再拆卡"]}')]))
            return _Resp(_Msg(content="ok"))

        with patch("bladex_proxy.route.router_sdk.acompletion", new=fake):
            with TestClient(app) as client:
                client.post("/v1/chat/completions",
                            json={"model": "gpt-4o", "stream": False, "tools": [_OWN_TOOL], "messages": _msgs("开始新任务")},
                            headers={"Authorization": "Bearer sk",
                                     "X-Agent-ID": "test-agent",
                                     "X-Session-ID": "s-init"})
                agency = client.app.state.agency
        led = list(agency.pool.values())[0]
        assert led.goal == "开始新任务"                     # goal 仍取用户原话
        assert [e.text for e in led.entries("next")] == ["先跑 gate_check", "再拆卡"]
        assert led.entries("core")[0].source == "model"    # 初始条目作者=模型

    def test_ledger_off_no_block(self, monkeypatch):
        _on(monkeypatch, "toolface", "interception")
        monkeypatch.setenv("BLADEX_MODULE_LEDGER", "0")  # 2026-09-02 翻默认后显式关
        app = create_app(_make_config())
        seen: list[dict] = []

        async def fake(model, messages, stream, **kw):
            seen.extend(messages)
            return _Resp(_Msg(content="ok"))

        with patch("bladex_proxy.route.router_sdk.acompletion", new=fake):
            with TestClient(app) as client:
                _post(client)
        assert not any("<bladex-ledger>" in str(m.get("content", "")) for m in seen)

    def test_switch_creates_ledger_goal_taken_from_user_query(self, monkeypatch):
        _on(monkeypatch, "toolface", "interception", "ledger")
        app = create_app(_make_config())
        calls = {"n": 0}

        async def fake(model, messages, stream, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                return _Resp(_Msg(tool_calls=[_tc(
                    "bladex_ledger_switch", "b1",
                    '{"ledger_id": "", "title": "修复路由"}')]))
            return _Resp(_Msg(content="账本已建，开始干活"))

        with patch("bladex_proxy.route.router_sdk.acompletion", new=fake):
            with TestClient(app) as client:
                client.post(
                    "/v1/chat/completions",
                    json={"model": "gpt-4o", "stream": False, "tools": [_OWN_TOOL], "messages": _msgs("帮我修复路由的回归问题")},
                    headers={"Authorization": "Bearer sk",
                             "X-Agent-ID": "test-agent",
                             "X-Session-ID": "s-goal"})
                agency = client.app.state.agency
                # goal「取」自用户原话
                leds = list(agency.pool.values())
                assert len(leds) == 1
                assert leds[0].goal == "帮我修复路由的回归问题"
                assert leds[0].goal_source == "user"
                assert leds[0].title == "修复路由"
                assert agency.activation.active(
                    agency.scope_of("test-agent")) == leds[0].ledger_id
                # 事件入 Hub（重建等价）
                evs = [e for _k, e in client.app.state.hub.scan_admin_events()
                       if str(getattr(e.event_type, "value", "")).startswith("ledger_")]
                assert {str(e.event_type.value) for e in evs} >= {
                    "ledger_create", "ledger_switch"}

    def test_new_ledger_is_bound_to_its_anchor_matter(self, monkeypatch):
        """V-L5：建账本这一刻就写死 Matter 锚（ADR-0032 §4.5）。

        🔴 绑定写在 proxy 侧而非 consolidator——增量 consolidation 用 secondary
        模式开 Memory Hub，写不了管理事件（ADR-0020 T1）。id 由 ledger_id
        确定性派生，所以两侧无需协调也必然一致。
        """
        from bladex_core.ledger_runtime import ledger_anchor_matter_id

        _on(monkeypatch, "toolface", "interception", "ledger")
        app = create_app(_make_config())
        calls = {"n": 0}

        async def fake(model, messages, stream, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                return _Resp(_Msg(tool_calls=[_tc(
                    "bladex_ledger_switch", "b1",
                    '{"ledger_id": "", "title": "锚点测试"}')]))
            return _Resp(_Msg(content="ok"))

        with patch("bladex_proxy.route.router_sdk.acompletion", new=fake):
            with TestClient(app) as client:
                client.post(
                    "/v1/chat/completions",
                    json={"model": "gpt-4o", "stream": False, "tools": [_OWN_TOOL], "messages": _msgs("干活")},
                    headers={"Authorization": "Bearer sk",
                             "X-Agent-ID": "test-agent",
                             "X-Session-ID": "s-anchor"})
                agency = client.app.state.agency
                led = next(iter(agency.pool.values()))
                assert led.matter_id == ledger_anchor_matter_id(led.ledger_id)
                # 事件载荷里也带着（重建等价：重放必得同一绑定）
                evs = [e for _k, e in client.app.state.hub.scan_admin_events()
                       if str(getattr(e.event_type, "value", "")) == "ledger_create"]
                assert evs and evs[-1].payload["ledger"]["matter_id"] == led.matter_id

    def test_goal_strips_agent_envelope(self, monkeypatch):
        """🔴 2026-08-25 live 回归：CC 的 `<session>…</session>` 信封不得进 Goal。"""
        _on(monkeypatch, "toolface", "interception", "ledger")
        app = create_app(_make_config())
        calls = {"n": 0}

        async def fake(model, messages, stream, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                return _Resp(_Msg(tool_calls=[_tc(
                    "bladex_ledger_switch", "b1", '{"ledger_id": ""}')]))
            return _Resp(_Msg(content="ok"))

        with patch("bladex_proxy.route.router_sdk.acompletion", new=fake):
            with TestClient(app) as client:
                client.post("/v1/chat/completions",
                            json={"model": "gpt-4o", "stream": False, "messages": [
                                {"role": "user",
                                 "content": "<session>\n详细评价一下新调整的V5架构\n"
                                            "</session>\n\nWrite the result to disk."}]},
                            headers={"Authorization": "Bearer sk",
                                     "X-Agent-ID": "claude-code",
                                     "X-Session-ID": "s-env"})
                agency = client.app.state.agency
        led = list(agency.pool.values())[0]
        assert "<session>" not in led.goal
        assert "详细评价一下新调整的V5架构" in led.goal

    def test_switch_event_payload_has_ledger_id(self, monkeypatch):
        _on(monkeypatch, "toolface", "interception", "ledger")
        app = create_app(_make_config())
        calls = {"n": 0}

        async def fake(model, messages, stream, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                return _Resp(_Msg(tool_calls=[_tc(
                    "bladex_ledger_switch", "b1", '{"ledger_id": ""}')]))
            return _Resp(_Msg(content="ok"))

        with patch("bladex_proxy.route.router_sdk.acompletion", new=fake):
            with TestClient(app) as client:
                _post(client)
                evs = [e for _k, e in client.app.state.hub.scan_admin_events()
                       if str(getattr(e.event_type, "value", "")) == "ledger_switch"]
        assert evs and evs[-1].payload.get("ledger_id"), "switch 事件缺 ledger_id"
        assert evs[-1].payload["ledger_id"] == evs[-1].payload["to_ledger_id"]

    def test_second_turn_block_contains_goal(self, monkeypatch):
        _on(monkeypatch, "toolface", "interception", "ledger")
        app = create_app(_make_config())
        calls = {"n": 0}
        seen_rounds: list[list[dict]] = []

        async def fake(model, messages, stream, **kw):
            calls["n"] += 1
            seen_rounds.append(list(messages))
            if calls["n"] == 1:
                return _Resp(_Msg(tool_calls=[_tc(
                    "bladex_ledger_switch", "b1", '{"ledger_id": ""}')]))
            return _Resp(_Msg(content="ok"))

        with patch("bladex_proxy.route.router_sdk.acompletion", new=fake):
            with TestClient(app) as client:
                hdr = {"Authorization": "Bearer sk", "X-Agent-ID": "test-agent",
                       "X-Session-ID": "s-goal2"}
                client.post("/v1/chat/completions",
                            json={"model": "gpt-4o", "stream": False, "tools": [_OWN_TOOL], "messages": _msgs("任务甲：清理索引")},
                            headers=hdr)
                client.post("/v1/chat/completions",
                            json={"model": "gpt-4o", "stream": False, "tools": [_OWN_TOOL], "messages": _msgs("继续")},
                            headers=hdr)
        # 第二轮注入块应含激活账本的 Goal 原文 + 首步对照指令
        last_round = seen_rounds[-1]
        blocks = [str(m.get("content", "")) for m in last_round
                  if "<bladex-ledger>" in str(m.get("content", ""))]
        assert blocks and "任务甲：清理索引" in blocks[0]
        assert "compare the user's request with the Goal" in blocks[0]


# ── 流式 fakes ──────────────────────────────────────────────────────────────

class _SDelta:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls
        self.reasoning_content = None


class _STc:
    def __init__(self, index, cid=None, name=None, args=None):
        self.index = index
        self.id = cid
        self.function = type("F", (), {"name": name, "arguments": args})()

    def dump(self):
        d = {"index": self.index}
        if self.id:
            d["id"] = self.id
        fn = {}
        if self.function.name:
            fn["name"] = self.function.name
        if self.function.arguments:
            fn["arguments"] = self.function.arguments
        if fn:
            d["function"] = fn
        d.setdefault("type", "function")
        return d


class _SChunk:
    def __init__(self, content=None, tool_calls=None, finish_reason=None):
        self.choices = [type("C", (), {
            "delta": _SDelta(content, tool_calls),
            "finish_reason": finish_reason})()]
        self.usage = None
        self._fr = finish_reason

    def model_dump(self):
        delta = {}
        d0 = self.choices[0].delta
        if d0.content is not None:
            delta["content"] = d0.content
        if d0.tool_calls:
            delta["tool_calls"] = [tc.dump() for tc in d0.tool_calls]
        return {"id": "chatcmpl-s", "object": "chat.completion.chunk",
                "model": "test",
                "choices": [{"index": 0, "delta": delta,
                             "finish_reason": self._fr}]}


class _SStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        self._it = iter(self._chunks)
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration from None


def _sse_events(text: str) -> list[dict]:
    out = []
    for line in text.splitlines():
        if line.startswith("data: ") and "[DONE]" not in line:
            out.append(json.loads(line[len("data: "):]))
    return out


class TestStream:
    def test_stream_pure_synthesizes_final(self, monkeypatch):
        _on(monkeypatch, "toolface", "interception")
        app = create_app(_make_config())
        calls = {"n": 0}

        async def fake(model, messages, stream, **kw):
            calls["n"] += 1
            if stream:
                return _SStream([
                    _SChunk(tool_calls=[_STc(0, "b1", "bladex_memory_search", '{"query":"x"}')]),
                    _SChunk(finish_reason="tool_calls"),
                ])
            return _Resp(_Msg(content="流式内循环后的答案"))

        with patch("bladex_proxy.route.router_sdk.acompletion", new=fake):
            with TestClient(app) as client:
                r = client.post(
                    "/v1/chat/completions",
                    json={"model": "gpt-4o", "stream": True, "tools": [_OWN_TOOL],
                          "messages": [{"role": "user", "content": "hi"}]},
                    headers={"Authorization": "Bearer sk",
                             "X-Agent-ID": "test-agent",
                             "X-Session-ID": "s-sp"})
        body = r.text
        events = _sse_events(body)
        # agent 看不到 bladex 调用
        assert not any(
            tc.get("function", {}).get("name", "").startswith("bladex_")
            for e in events
            for tc in (e.get("choices") or [{}])[0].get("delta", {}).get("tool_calls") or [])
        # 收到合成的最终文本 + stop 终止 + DONE
        texts = "".join((e.get("choices") or [{}])[0].get("delta", {}).get("content") or ""
                        for e in events)
        assert "流式内循环后的答案" in texts
        assert "[DONE]" in body

    def test_stream_text_plus_all_bladex_calls_rewrites_finish_reason(self, monkeypatch):
        """🔴 2026-08-25 live 事故回归：模型输出正文 + 只调 bladex 工具时，
        上游 finish_reason=tool_calls，但 agent 看到的调用数为 0 ⇒ 必须改写成
        stop，否则 hermes 报「indicated a tool call but none was included」并重试。"""
        _on(monkeypatch, "toolface", "interception", "ledger")
        app = create_app(_make_config())

        seen_inner: list[list[dict]] = []

        async def fake(model, messages, stream, **kw):
            if stream:
                return _SStream([
                    _SChunk(content="老板，先把证据归档到任务账本，再给结论。"),
                    _SChunk(tool_calls=[_STc(0, "b1", "bladex_ledger_update", "{}")]),
                    _SChunk(tool_calls=[_STc(1, "b2", "bladex_ledger_update", "{}")]),
                    _SChunk(finish_reason="tool_calls"),
                ])
            seen_inner.append(list(messages))     # 内循环的那次非流式往返
            return _Resp(_Msg(content="结论：图片全部原图直出，未做懒加载。"))

        with patch("bladex_proxy.route.router_sdk.acompletion", new=fake):
            with TestClient(app) as client:
                r = client.post(
                    "/v1/chat/completions",
                    json={"model": "gpt-4o", "stream": True, "tools": [_OWN_TOOL],
                          "messages": _msgs("进一步分析图片加载机制")},
                    headers={"Authorization": "Bearer sk",
                             "X-Agent-ID": "test-agent",
                             "X-Session-ID": "s-fr"})
        events = _sse_events(r.text)
        # agent 看不到任何 bladex 调用
        assert not any(
            tc.get("function", {}).get("name", "").startswith("bladex_")
            for e in events
            for tc in (e.get("choices") or [{}])[0].get("delta", {}).get("tool_calls") or [])
        texts = "".join((e.get("choices") or [{}])[0].get("delta", {}).get("content") or ""
                        for e in events)
        # 前言照常转发
        assert "先把证据归档到任务账本" in texts
        # 🔴 **本条的真判据**（2026-08-26 live："看起来卡死"）：模型说了"再给结论"，
        # 结论就必须真的来。旧行为判 MIXED ⇒ 只转发前言 + finish_reason=stop ⇒
        # 回合就此结束，用户看到"说要给结论然后没了"。
        assert "结论：图片全部原图直出" in texts, (
            "只转发了前言就收尾 ⇒ 回归到「看起来卡死」那个病例")
        assert texts.index("先把证据归档") < texts.index("结论：图片"), "顺序反了"
        # 🔴 已流出的前言要进内循环的 assistant 消息，否则模型看不到自己刚说过什么
        assert seen_inner, "内循环没跑"
        assistant_contents = [m.get("content") for m in seen_inner[0]
                              if m.get("role") == "assistant"]
        assert any(c and "先把证据归档" in str(c) for c in assistant_contents), \
            "前言没带进内循环 ⇒ 模型会重复或跑偏"
        # 终止 chunk 仍不能是 tool_calls（hermes 会报"说了有工具调用却没有"并重试）
        finals = [f for f in ((e.get("choices") or [{}])[0].get("finish_reason")
                              for e in events) if f]
        assert finals and finals[-1] == "stop", f"finish_reason 未改写: {finals}"

    def test_anthropic_stream_strips_bladex_tool_use(self, monkeypatch):
        """V-P5b：anthropic 端点（Claude Code）——bladex tool_use 块被剥、
        agent 自己的 tool_use 保留、stop_reason 按转发调用数改写。"""
        _on(monkeypatch, "toolface", "interception", "ledger")
        app = create_app(_make_config())

        async def fake(model, messages, stream, **kw):
            return _SStream([
                _SChunk(content="我先记一笔。"),
                _SChunk(tool_calls=[_STc(0, "b1", "bladex_ledger_update", "{}")]),
                _SChunk(finish_reason="tool_calls"),
            ])

        with patch("bladex_proxy.route.router_sdk.acompletion", new=fake):
            with TestClient(app) as client:
                r = client.post(
                    "/v1/messages",
                    json={"model": "claude-x", "stream": True, "max_tokens": 100,
                          "messages": [{"role": "user", "content": "hi"}]},
                    headers={"x-api-key": "sk", "X-Agent-ID": "claude-code",
                             "X-Session-ID": "s-anth"})
        body = r.text
        assert "bladex_ledger_update" not in body      # agent 侧看不到
        assert "我先记一笔。" in body                    # 正文照常
        assert '"stop_reason": "tool_use"' not in body  # 改写为 end_turn

    def test_anthropic_pure_bladex_runs_inner_loop(self, monkeypatch):
        """🔴 V-P5c（CC live 实证驱动）：纯 bladex 调用必须走多轮内循环，
        而不是合成降级——否则模型第一步被打断，永远走不到 ledger_switch。"""
        _on(monkeypatch, "toolface", "interception", "ledger")
        app = create_app(_make_config())
        calls = {"n": 0}

        async def fake(model, messages, stream, **kw):
            calls["n"] += 1
            if stream:      # 首轮：只调 bladex 工具（CC 的真实开场形态）
                return _SStream([
                    _SChunk(tool_calls=[_STc(0, "b1", "bladex_memory_search",
                                             '{"query":"x"}')]),
                    _SChunk(finish_reason="tool_calls"),
                ])
            # 内循环里的非流式回调：拿到工具结果后给最终答案
            return _Resp(_Msg(content="基于记忆的分析结论"))

        with patch("bladex_proxy.route.router_sdk.acompletion", new=fake):
            with TestClient(app) as client:
                r = client.post(
                    "/v1/messages",
                    json={"model": "claude-x", "stream": True, "max_tokens": 100,
                          "messages": [{"role": "user", "content": "分析一下"}]},
                    headers={"x-api-key": "sk", "X-Agent-ID": "claude-code",
                             "X-Session-ID": "s-loop"})
        body = r.text
        assert calls["n"] >= 2, "内循环没有发生（只调了一次上游）"
        assert "基于记忆的分析结论" in body        # 模型的最终回答到达 agent
        assert "bladex_memory_search" not in body  # 工具调用仍不可见
        assert "[BladeX]" not in body              # 不是合成降级文本

    def test_responses_stream_strips_bladex_function_call(self, monkeypatch):
        """V-P5b：responses 端点（Codex）——bladex function_call item 被剥。"""
        _on(monkeypatch, "toolface", "interception", "ledger")
        app = create_app(_make_config())

        async def fake(model, messages, stream, **kw):
            return _SStream([
                _SChunk(content="记一笔。"),
                _SChunk(tool_calls=[_STc(0, "b1", "bladex_ledger_update", "{}")]),
                _SChunk(finish_reason="tool_calls"),
            ])

        with patch("bladex_proxy.route.router_sdk.acompletion", new=fake):
            with TestClient(app) as client:
                r = client.post(
                    "/v1/responses",
                    json={"model": "gpt-x", "stream": True, "input": "hi"},
                    headers={"Authorization": "Bearer sk", "X-Agent-ID": "codex",
                             "X-Session-ID": "s-resp"})
        body = r.text
        assert "bladex_ledger_update" not in body
        assert "记一笔。" in body

    def test_responses_synth_item_lands_in_completed_snapshot(self):
        """🔴 2026-08-25 Codex live 事故回归：合成 message item 只出现在流里、
        没进 `response.completed` 的 output 快照 ⇒ Codex 对账失败 `client disconnected`。
        合成事件与终止快照必须同源，且 output_index 连续（不用 99 占位）。"""
        import json as _json

        from bladex_proxy.agency import (
            _rewrite_stop_responses,
            _synth_responses_text_events,
        )
        sink: list = []
        evs = list(_synth_responses_text_events([{"content": "最终答案"}], raw=True,
                                                index=2, sink=sink))
        idxs = [_json.loads(e.decode().split("data: ")[1])["output_index"] for e in evs]
        # 用调用方给的 index，全程一致（不用 99 占位）。
        # 🔴 断言写"编号一致"而不是"恰好三个事件"：2026-08-27 合成形态补了
        # `content_part.added/done`（缺它 ⇒ 官方解析器 IndexError ⇒ Codex 断连），
        # 事件数 3→5，而本测试要守的是**编号**这件事，与事件数无关。
        # 事件序列本身由 `test_synth_stream_shape.py` 按官方状态机钉死。
        assert set(idxs) == {2}
        assert len(idxs) >= 3
        assert sink and sink[0]["id"] == "msg_bladex"

        completed = (b"event: response.completed\ndata: " + _json.dumps({
            "type": "response.completed",
            "response": {"id": "r", "output": [
                {"type": "function_call", "name": "bladex_ledger_switch",
                 "call_id": "b1"},
                {"type": "message", "id": "m0"}]}}).encode() + b"\n\n")
        out = _rewrite_stop_responses(completed, False, sink)
        snap = _json.loads(out.decode().split("data: ")[1])["response"]["output"]
        ids = [i.get("id") or i.get("name") for i in snap]
        assert "msg_bladex" in ids, "合成 item 未进快照 ⇒ Codex 会断连"
        assert not any(str(i.get("name", "")).startswith("bladex_") for i in snap)

    def test_stream_mixed_strips_and_keeps_agent_call(self, monkeypatch):
        _on(monkeypatch, "toolface", "interception", "ledger")
        app = create_app(_make_config())

        async def fake(model, messages, stream, **kw):
            return _SStream([
                _SChunk(tool_calls=[_STc(0, "a1", "web_search", "{}")]),
                _SChunk(tool_calls=[_STc(1, "b1", "bladex_ledger_read", "{}")]),
                _SChunk(finish_reason="tool_calls"),
            ])

        with patch("bladex_proxy.route.router_sdk.acompletion", new=fake):
            with TestClient(app) as client:
                r = client.post(
                    "/v1/chat/completions",
                    json={"model": "gpt-4o", "stream": True, "tools": [_OWN_TOOL],
                          "messages": [{"role": "user", "content": "hi"}]},
                    headers={"Authorization": "Bearer sk",
                             "X-Agent-ID": "test-agent",
                             "X-Session-ID": "s-sm"})
                agency = client.app.state.agency
        events = _sse_events(r.text)
        names = [tc.get("function", {}).get("name")
                 for e in events
                 for tc in (e.get("choices") or [{}])[0].get("delta", {}).get("tool_calls") or []]
        assert "web_search" in names and not any(
            str(n).startswith("bladex_") for n in names if n)
        recs = [rec for s in agency.splice._by_session.values() for rec in s]
        assert recs and recs[0].calls[0]["function"]["name"] == "bladex_ledger_read"


class TestLedgerBoundaryFixes20260826:
    """2026-08-26 五项拍板的守卫（MQ-L7/L8/L9/L10/L11）。

    每条都对应一个 **live 病例**，不是假想的边界。病例出处见各自 docstring 与
    `docs/planning/memory-quality-problem-ledger-20260815.md` 账本域。
    """

    # ── MQ-L10：注入位置 ──
    def test_block_goes_to_the_very_end_not_after_last_user(self, monkeypatch):
        """🔴 jydesignhk 病例：工具循环里"末条 user 之后" ≠ "上下文末尾"。

        主会话 27 轮里最后一条 user 恒在下标 167 不动，账本块被埋在 17→65 条之下，
        17 轮**零** bladex 调用；同请求的子调用 `msgs=1`（块在末尾）**开了 5 张账本**。
        """
        _on(monkeypatch, "toolface", "interception", "ledger")
        app = create_app(_make_config())
        seen: list[dict] = []

        async def fake(model, messages, stream, **kw):
            seen.clear(); seen.extend(messages)
            return _Resp(_Msg(content="ok"))

        # 工具循环形态：末条 user 之后还有 assistant/tool 对
        msgs = [_SYS, {"role": "user", "content": "深度分析这个网站"}]
        for i in range(6):
            msgs += [{"role": "assistant", "content": None,
                      "tool_calls": [_tc("screenshot", f"t{i}")]},
                     {"role": "tool", "tool_call_id": f"t{i}", "content": "img"}]

        with patch("bladex_proxy.route.router_sdk.acompletion", new=fake):
            with TestClient(app) as client:
                client.post("/v1/chat/completions",
                            json={"model": "gpt-4o", "stream": False,
                                  "tools": [_OWN_TOOL], "messages": msgs},
                            headers={"Authorization": "Bearer sk",
                                     "X-Agent-ID": "test-agent",
                                     "X-Session-ID": "s-loop"})
        idx = [i for i, m in enumerate(seen)
               if "<bladex-ledger>" in str(m.get("content", ""))]
        assert idx, "账本块没注进去"
        assert idx[0] == len(seen) - 1, (
            f"账本块在下标 {idx[0]}/{len(seen) - 1}，之后还有 "
            f"{len(seen) - 1 - idx[0]} 条 —— 又被埋回工具循环里了")

    # ── MQ-L8：孤立子调用 ──
    def test_isolated_subcall_gets_no_ledger_face(self, monkeypatch):
        """🔴 一个用户请求开出 5 张账本的根因：Hermes 视觉子调用（无 system、
        单条 user、带图）指纹失败落 UA 桶、aux 零命中 ⇒ 账本面全开。"""
        _on(monkeypatch, "toolface", "interception", "ledger")
        app = create_app(_make_config())
        seen: list[dict] = []
        tools_seen: list = []

        async def fake(model, messages, stream, **kw):
            seen.clear(); seen.extend(messages)
            tools_seen.append(kw.get("tools"))
            return _Resp(_Msg(content="ok"))

        with patch("bladex_proxy.route.router_sdk.acompletion", new=fake):
            with TestClient(app) as client:
                client.post("/v1/chat/completions",
                            json={"model": "gpt-4o", "stream": False, "messages": [
                                {"role": "user", "content": [
                                    {"type": "text", "text": "描述这张图"},
                                    {"type": "image_url",
                                     "image_url": {"url": "data:image/png;base64,x"}},
                                ]}]},
                            headers={"Authorization": "Bearer sk",
                                     "X-Agent-ID": "test-agent"})
        assert not any("<bladex-ledger>" in str(m.get("content", "")) for m in seen)
        names = [(t.get("function") or t).get("name")
                 for t in (tools_seen[-1] or [])]
        # 🔴 2026-09-02 MQ-L34 改口径：断言收窄到**账本族**——原来写的是"任何 bladex_*
        # 都不许"，比 docstring 的立论（"它就能开账本"）宽了一族。族拆分后孤立子调用
        # 照拿记忆族（`bladex_memory_search` 是取记忆唯一通道），账本三工具仍不许出现。
        assert not any(str(n).startswith("bladex_ledger_") for n in names), \
            "孤立子调用拿到了账本工具 ⇒ 它就能开账本 ⇒ 噪声卡回归"
        assert "bladex_memory_search" in names, \
            "MQ-L34：孤立子调用只挡账本族，记忆族必须照注"

        # 🔴 阳性对照：同一个请求**加上 system**（= 真实 agent 的形状）必须照常拿到
        # 账本面。没有这一半，上面的断言在"工具面整个坏掉"时也会绿。
        with patch("bladex_proxy.route.router_sdk.acompletion", new=fake):
            with TestClient(app) as client:
                client.post("/v1/chat/completions",
                            json={"model": "gpt-4o", "stream": False,
                                  "tools": [_OWN_TOOL], "messages": [
                                _SYS,
                                {"role": "user", "content": [
                                    {"type": "text", "text": "描述这张图"},
                                    {"type": "image_url",
                                     "image_url": {"url": "data:image/png;base64,x"}},
                                ]}]},
                            headers={"Authorization": "Bearer sk",
                                     "X-Agent-ID": "test-agent"})
        ctrl = [(t.get("function") or t).get("name") for t in (tools_seen[-1] or [])]
        assert any(str(n).startswith("bladex_") for n in ctrl), \
            "阳性对照失败：带 system 的真实请求也没拿到工具面 ⇒ 上面那条断言无意义"
        assert any("<bladex-ledger>" in str(m.get("content", "")) for m in seen)

    # ── MQ-L11：激活 scope ──
    def test_active_ledger_survives_new_session_same_agent(self, monkeypatch):
        """🔴 Hermes 压缩交接后指纹变了，按 session 绑定时旧账本看不见 ⇒ 冷启动。
        改 (agent, project) 后**换个 session 仍拿到 Goal 对照**。"""
        _on(monkeypatch, "toolface", "interception", "ledger")
        app = create_app(_make_config())
        calls = {"n": 0}
        rounds: list[list[dict]] = []

        async def fake(model, messages, stream, **kw):
            calls["n"] += 1
            rounds.append(list(messages))
            if calls["n"] == 1:
                return _Resp(_Msg(tool_calls=[_tc(
                    "bladex_ledger_switch", "b1",
                    '{"ledger_id": "", "title": "泰山啤酒"}')]))
            return _Resp(_Msg(content="ok"))

        with patch("bladex_proxy.route.router_sdk.acompletion", new=fake):
            with TestClient(app) as client:
                hdr = {"Authorization": "Bearer sk", "X-Agent-ID": "test-agent"}
                client.post("/v1/chat/completions",
                            json={"model": "gpt-4o", "stream": False, "tools": [_OWN_TOOL],
                                  "messages": _msgs("梳理泰山啤酒破产重整")},
                            headers={**hdr, "X-Session-ID": "fp:before-compaction"})
                # 压缩交接 ⇒ 全新 session id
                client.post("/v1/chat/completions",
                            json={"model": "gpt-4o", "stream": False, "tools": [_OWN_TOOL],
                                  "messages": _msgs("[CONTEXT COMPACTION] 继续")},
                            headers={**hdr, "X-Session-ID": "fp:after-compaction"})
        block = [str(m.get("content", "")) for m in rounds[-1]
                 if "<bladex-ledger>" in str(m.get("content", ""))]
        assert block, "换 session 后账本块没注"
        assert "梳理泰山啤酒破产重整" in block[0], "换 session 后激活账本丢了（回归 MQ-L11）"
        assert "compare the user's request with the Goal" in block[0]

    # ── MQ-L7：C 方案，正文/指令分开 ──
    def test_weak_tier_gets_readonly_body_without_instruction(self, monkeypatch):
        """🔴 有令无器：weak 档注入块写着 "call bladex_ledger_switch"，
        而 tools 里根本没有这个工具。C 方案 = 正文照给（看得见），指令不给。"""
        from bladex_proxy.agency import AgencyRuntime
        _on(monkeypatch, "toolface", "ledger")
        ag = AgencyRuntime()
        led = _mk_ledger(ag, "test-agent", goal="修好路由")

        with_i = ag.insert_ledger_block([{"role": "user", "content": "hi"}],
                                        "test-agent", with_instruction=True)
        without = ag.insert_ledger_block([{"role": "user", "content": "hi"}],
                                         "test-agent", with_instruction=False)
        body_i, body_w = with_i[-1]["content"], without[-1]["content"]
        assert "修好路由" in body_i and "修好路由" in body_w      # 正文两边都在
        assert "bladex_ledger_switch" in body_i                  # 有工具才下指令
        assert "bladex_ledger_switch" not in body_w              # 无器则无令
        assert led.ledger_id                                     # 用一下，避免未用变量

    # ── 2026-08-30：工具面与首步指令解耦 ──
    @pytest.mark.parametrize(
        ("tier", "want_instruction"),
        [("weak", False), ("medium", True), ("strong", True), ("", True)],
    )
    def test_instruction_still_follows_the_tier_knob(self, monkeypatch,
                                                     tier, want_instruction):
        """🔴 拆 weak 工具面门后，**首步指令仍按 `AUTO_LEDGER_TIERS` 排除 weak**。

        为什么必须单独钉：拆门前调用方传的是 `with_instruction=tf_injected`
        （工具面注没注），两件事等价；拆门后 weak 的 `tf_injected` 变 True，
        那个等价会**顺手把首步指令也给到 weak** —— 等于改了账本注入策略，
        而 Jason 同日明确"账本注入策略保持不变"。判据合成挪进了
        `insert_ledger_block` 这个生产点（调用方少传一个条件就静默退化，
        2026-08-29 三条流式入口漏传 ctx 刚踩过）。

        `tier=""`（老调用方/未路由）视为在集合内——不能因为拿不到档位就
        静默关掉指令，与工具面那侧同一条哲学。

        🔴 前置必须**显式绑定来源会话**：`_mk_ledger` 走工具面建本时 dispatch
        context 里没有 session_id ⇒ `last_session` 为空 ⇒ MQ-L28 gate 按
        "来源未知"保守拦截，weak 档连正文都不注。初版没绑，于是断言"正文照给"
        撞在一个**正确行为**上——测试前提没搭对，不是生产错了。
        """
        from bladex_core.ledger_runtime import activation_scope
        from bladex_proxy.agency import AgencyRuntime
        _on(monkeypatch, "toolface", "ledger")
        monkeypatch.setenv("BLADEX_LEDGER_TIERS", "auto")
        ag = AgencyRuntime()
        led = _mk_ledger(ag, "test-agent", goal="修好路由")
        scope = activation_scope("test-agent", "")
        ag.activation.restore({scope: led.ledger_id}, sessions={scope: "sess-1"})
        out = ag.insert_ledger_block(
            [{"role": "user", "content": "hi"}], "test-agent",
            with_instruction=True, tier=tier, session_id="sess-1",
        )
        body = out[-1]["content"] if out and out[-1].get("role") else ""
        assert "修好路由" in body, f"tier={tier} 正文没注（C 方案要求正文照给）"
        assert ("bladex_ledger_switch" in body) is want_instruction, (
            f"tier={tier} 首步指令应为 {want_instruction}")
        assert led.ledger_id

    def test_instruction_never_without_tools(self, monkeypatch):
        """🔴 MQ-L7 不变式在解耦后仍成立：**有令 ⇒ 必有器**。

        拆门后关系从"等价"变"蕴含"。仍有一条路径会"tier 在集合内但工具面没注"
        —— dsh 系 agent（只有 run_code 可直呼）。调用方传 False，
        生产点不得把它变回 True。
        """
        from bladex_proxy.agency import AgencyRuntime
        _on(monkeypatch, "toolface", "ledger")
        ag = AgencyRuntime()
        _mk_ledger(ag, "test-agent", goal="修好路由")
        out = ag.insert_ledger_block(
            [{"role": "user", "content": "hi"}], "test-agent",
            with_instruction=False, tier="strong")
        assert "bladex_ledger_switch" not in out[-1]["content"]

    def test_no_ledger_and_no_tools_injects_nothing(self, monkeypatch):
        """无账本 + 无工具：整块不注（模板 1219 chars 是纯税）。"""
        from bladex_proxy.agency import AgencyRuntime
        _on(monkeypatch, "toolface", "ledger")
        ag = AgencyRuntime()
        out = ag.insert_ledger_block([{"role": "user", "content": "hi"}],
                                     "nobody", with_instruction=False)
        assert out == [{"role": "user", "content": "hi"}]

    # ── MQ-L1/④：parent 显式 + 双侧留痕 ──
    def test_parent_is_explicit_not_the_active_ledger(self, monkeypatch):
        """🔴 五个**平行**页面评审被串成四层假父子链：新建时把"当时激活的那本"
        无条件当父。parent 是 LLM 的决策，BladeX 不替它判。"""
        from bladex_proxy.agency import AgencyRuntime
        _on(monkeypatch, "toolface", "ledger")
        ag = AgencyRuntime()
        first = _mk_ledger(ag, "test-agent", goal="页面 A 评审")
        second = _mk_ledger(ag, "test-agent", goal="页面 B 评审")   # 不给 parent
        assert second.parent_ledger_id == "", "又把激活账本当父了（回归 MQ-L1）"
        assert first.ledger_id != second.ledger_id

    def test_explicit_parent_marks_both_sides(self, monkeypatch):
        """子任务协议：子侧 Next 带回写义务，父侧 Next 带等待项，两侧都带标签。"""
        import asyncio

        from bladex_proxy.agency import SUBTASK_TAG, AgencyRuntime
        _on(monkeypatch, "toolface", "ledger")
        ag = AgencyRuntime()
        parent = _mk_ledger(ag, "test-agent", goal="主任务")
        ctx = {"agent_id": "test-agent", "user_query": "顺手做个子任务", "turn_index": 5}
        asyncio.run(ag.toolface.dispatch(
            "bladex_ledger_switch",
            f'{{"ledger_id": "", "title": "子任务", '
            f'"parent_ledger_id": "{parent.ledger_id}"}}',
            allowed_exposure="public", context=ctx))
        child = next(l for l in ag.pool.values()
                     if l.parent_ledger_id == parent.ledger_id)
        assert any(SUBTASK_TAG in e.text and parent.ledger_id in e.text
                   for e in child.entries("next")), "子侧没写回写义务"
        parent_now = ag.pool[parent.ledger_id]
        assert any(SUBTASK_TAG in e.text and e.ref == child.ledger_id
                   for e in parent_now.entries("next")), "父侧没留等待项 ⇒ 失联"

    def test_unknown_parent_is_rejected(self, monkeypatch):
        import asyncio

        from bladex_proxy.agency import AgencyRuntime
        _on(monkeypatch, "toolface", "ledger")
        ag = AgencyRuntime()
        out = asyncio.run(ag.toolface.dispatch(
            "bladex_ledger_switch",
            '{"ledger_id": "", "title": "x", "parent_ledger_id": "ldg-nope"}',
            allowed_exposure="public",
            context={"agent_id": "a", "user_query": "q"}))
        assert "unknown parent_ledger_id" in out
        assert not ag.pool, "父账本不存在时不该把子账本建出来"

    # ── MQ-L9：按 id 读账本 ──
    def test_ledger_read_by_id(self, monkeypatch):
        """🔴 注入块让模型 `switch by id`，却只给标题、没有按 id 读的手段 ⇒ 只能盲切。
        实测 13 条 switch 里没有一条是"从列表挑中"的结果——列表从没被用起来过。"""
        import asyncio

        from bladex_proxy.agency import AgencyRuntime
        _on(monkeypatch, "toolface", "ledger")
        ag = AgencyRuntime()
        other = _mk_ledger(ag, "other-agent", goal="泰山啤酒破产重整")
        # 当前 agent 没有激活账本，但仍能读到别人那本（只读）
        out = asyncio.run(ag.toolface.dispatch(
            "bladex_ledger_read", f'{{"ledger_id": "{other.ledger_id}"}}',
            allowed_exposure="public", context={"agent_id": "test-agent"}))
        assert "泰山啤酒破产重整" in out
        miss = asyncio.run(ag.toolface.dispatch(
            "bladex_ledger_read", '{"ledger_id": "ldg-nope"}',
            allowed_exposure="public", context={"agent_id": "test-agent"}))
        assert "unknown ledger" in miss
        # 缺省仍是激活账本
        blank = asyncio.run(ag.toolface.dispatch(
            "bladex_ledger_read", "{}", allowed_exposure="public",
            context={"agent_id": "test-agent"}))
        assert "No active ledger" in blank

    def test_read_by_id_is_in_the_schema(self):
        """工具描述得让模型知道可以先 read 再 switch，否则修了也用不上。"""
        from bladex_proxy.toolface import TOOL_SCHEMAS
        read = next(t["function"] for t in TOOL_SCHEMAS
                    if t["function"]["name"] == "bladex_ledger_read")
        assert "ledger_id" in read["parameters"]["properties"]
        switch = next(t["function"] for t in TOOL_SCHEMAS
                      if t["function"]["name"] == "bladex_ledger_switch")
        assert "parent_ledger_id" in switch["parameters"]["properties"]


def _mk_ledger(ag, agent_id: str, *, goal: str):
    """经工具面建一本并激活（走真实路径，不手工塞 pool）。"""
    import asyncio
    asyncio.run(ag.toolface.dispatch(
        "bladex_ledger_switch", '{"ledger_id": "", "title": "t"}',
        allowed_exposure="public",
        context={"agent_id": agent_id, "user_query": goal, "turn_index": 0}))
    return ag.pool[ag.activation.active(ag.scope_of(agent_id))]


class TestObservabilityGaps20260826:
    """MQ-L5 三处观测缺口的守卫。

    🔴 为什么给日志写测试：这三条**不是锦上添花**，是三次归因失败的直接原因——
    ① 88–135 秒的内循环只能靠翻相邻日志行猜（`LoopRound` 算好了 tool_ms/llm_ms
    却不输出）；② 一轮零账本活动分不清"没给工具"还是"给了没用"；
    ③ 账本块被埋 65 条消息之下，潜伏到靠翻消息结构才发现（Hub 存的是注入**前**
    的 `original_messages`，线上根本看不见注入后的形态）。
    日志没测试 = 下次它悄悄不输出了也没人知道。
    """

    def test_loop_cost_breaks_down_tool_vs_llm(self):
        from bladex_proxy.agency import _loop_cost
        from bladex_proxy.innerloop import LoopRound

        class _R:
            rounds = [
                LoopRound(round_index=1, tool_ms=12.4, llm_ms=880.6,
                          tool_names=["bladex_ledger_read"], usage={"total": 150}),
                LoopRound(round_index=2, tool_ms=3.1, llm_ms=1200.0,
                          tool_names=["bladex_ledger_update"], usage={"total": 90}),
            ]

        c = _loop_cost(_R())
        assert c["tool_ms"] == 16 and c["llm_ms"] == 2081      # 钱花在哪，一眼可见
        assert c["tokens"] == 240
        assert c["tools"] == "bladex_ledger_read,bladex_ledger_update"

    def test_loop_cost_tolerates_missing_fields(self):
        """老 result 对象/降级路径不得把日志打挂——观测代码不许成为故障源。"""
        from bladex_proxy.agency import _loop_cost
        assert _loop_cost(object())["tool_ms"] == 0

    def test_toolface_decision_is_always_logged(self, monkeypatch):
        """注了要有日志，**没注更要有**——而且要说清为什么没注。

        🔴 用 `structlog.testing.capture_logs` 而非 pytest 的 `caplog`：proxy 侧走
        structlog，不经 stdlib handler ⇒ caplog 恒空（仓库里
        `test_personal_identity_semantics` 已踩过同一个坑）。
        """
        import structlog.testing

        from bladex_proxy.agency import AgencyRuntime
        _on(monkeypatch, "toolface", "ledger")
        ag = AgencyRuntime()
        with structlog.testing.capture_logs() as cap:
            ag.augment_tools(None, agent_id="hermes:default", auxiliary=False,
                             tier="medium")
            # 🔴 2026-08-30 换例：原来第二次用的是 `tier="weak"`（期望
            # reason=weak_tier）。拆掉 weak 门后 weak 会被正常注入，那一档
            # 已从 reason 枚举里删除——**换成 dsh 来覆盖"没注"的另一个真因**，
            # 而不是把断言改成期望它注了：这条测试守的是"三种决策都留痕、
            # 且 reason 分得清"，不是"weak 会怎样"。
            ag.augment_tools(None, agent_id="dsh", auxiliary=False,
                             tier="medium")
            ag.augment_tools(None, agent_id="hermes:default", auxiliary=True,
                             tier="strong")
        rows = [e for e in cap if e.get("event") == "agency_toolface_decision"]
        assert len(rows) == 3, "有一次决策没留痕"
        assert [r["injected"] for r in rows] == [True, False, False]
        assert [r["reason"] for r in rows] == ["", "agent_excluded", "aux"]

    def test_ledger_block_logs_its_position(self, monkeypatch):
        """`after=0` 就是 MQ-L10 的判据本身——线上必须能直接读到，
        不能再靠翻消息结构去推。"""
        import structlog.testing

        from bladex_proxy.agency import AgencyRuntime
        _on(monkeypatch, "toolface", "ledger")
        ag = AgencyRuntime()
        led = _mk_ledger(ag, "test-agent", goal="某任务")
        msgs = [_SYS, {"role": "user", "content": "hi"},
                {"role": "assistant", "content": None,
                 "tool_calls": [_tc("x", "t1")]},
                {"role": "tool", "tool_call_id": "t1", "content": "r"}]
        with structlog.testing.capture_logs() as cap:
            out = ag.insert_ledger_block(msgs, "test-agent")
        row = next(e for e in cap if e.get("event") == "agency_ledger_block_injected")
        assert row["after"] == 0, "块之后还有消息 ⇒ 又被埋回工具循环里了"
        assert row["pos"] == len(out) - 1 and row["total"] == len(out)
        assert row["ledger"] == led.ledger_id and row["with_instruction"] is True
