"""Server 端点单元测试（T8/T9，ISSUE-3/5 修复后）。"""

import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from bladex_proxy.config import ProxyConfig
from bladex_proxy.inject import MEMORY_OPEN
from bladex_proxy.models import TurnStatus
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


def test_health():
    app = create_app(_make_config())
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


def test_non_stream_inject_and_forward():
    """非流式：注入生效 + 回复透传 + 注入段出现在转发请求里。"""
    app = create_app(_make_config())

    fake_response = MagicMock()
    fake_response.choices = [MagicMock()]
    fake_response.choices[0].message.content = "Hello! I am polite."
    fake_response.model_dump.return_value = {
        "id": "chatcmpl-test",
        "choices": [{"message": {"role": "assistant", "content": "Hello! I am polite."}}],
    }

    captured_messages: list[dict] = []

    async def fake_acompletion(model, messages, stream, **kwargs):
        captured_messages.extend(messages)
        return fake_response

    with patch("bladex_proxy.route.router_sdk.acompletion", new=fake_acompletion):
        with TestClient(app) as client:
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Say hello"}], "stream": False},
                headers={"Authorization": "Bearer sk-test-key", "X-Agent-ID": "test-agent"},
            )

    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "Hello! I am polite."
    assert any(isinstance(m.get("content"), str) and MEMORY_OPEN in m["content"] for m in captured_messages)


def test_stream_inject_and_forward():
    """流式：注入生效 + SSE 回传。"""
    app = create_app(_make_config())

    from tests.test_capture import _FakeChunk, _FakeStream

    fake_stream = _FakeStream([_FakeChunk(content="Hello"), _FakeChunk(content="!")])

    captured_messages: list[dict] = []

    async def fake_acompletion(model, messages, stream, **kwargs):
        captured_messages.extend(messages)
        return fake_stream

    with patch("bladex_proxy.route.router_sdk.acompletion", new=fake_acompletion):
        with TestClient(app) as client:
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Say hello"}], "stream": True},
                headers={"Authorization": "Bearer sk-test-key", "X-Agent-ID": "test-agent"},
            )

    assert resp.status_code == 200
    body = resp.text
    assert "data: " in body
    assert "[DONE]" in body
    assert "Hello" in body
    assert any(isinstance(m.get("content"), str) and MEMORY_OPEN in m["content"] for m in captured_messages)


def test_multiturn_no_accumulation_via_api():
    """多轮通过 API 调用不累积注入段。"""
    app = create_app(_make_config())

    fake_response = MagicMock()
    fake_response.choices = [MagicMock()]
    fake_response.choices[0].message.content = "OK"
    fake_response.model_dump.return_value = {"choices": [{"message": {"role": "assistant", "content": "OK"}}]}

    async def fake_acompletion(model, messages, stream, **kwargs):
        bladex_count = sum(
            m.get("content", "").count(MEMORY_OPEN) for m in messages if isinstance(m.get("content"), str)
        )
        assert bladex_count <= 1, f"Expected <=1 bladex-memory block, got {bladex_count}"
        return fake_response

    with patch("bladex_proxy.route.router_sdk.acompletion", new=fake_acompletion):
        with TestClient(app) as client:
            client.post("/v1/chat/completions",
                json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}], "stream": False},
                headers={"Authorization": "Bearer sk-test", "X-Agent-ID": "a"})
            client.post("/v1/chat/completions",
                json={"model": "gpt-4o", "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "OK"},
                    {"role": "system", "content": f"{MEMORY_OPEN}\n- old\n</bladex-memory>"},
                    {"role": "user", "content": "Bye"},
                ], "stream": False},
                headers={"Authorization": "Bearer sk-test", "X-Agent-ID": "a"})


# 🔴 存储路径一律走 `tmp_path`，**不要写死 /tmp 下的固定目录**（2026-08-08 立）。
#
# 这三个用例原本写死 `/tmp/bladex_test_auth_r{,2,3}`，于是 RocksDB 状态跨运行累积。
# macOS 的 /tmp 定期清理会删旧文件却留下目录 —— 结果是 MANIFEST 还在、.sst 没了，
# 下一次 gate 报 `Corruption: ... 000061.sst: No such file or directory`，
# 看起来像存储层坏了，实际与被测代码毫无关系。
# 排查这种假红的成本远高于写 `tmp_path` 的成本。
def test_auth_rejected_returns_401(tmp_path):
    app = create_app(ProxyConfig(
        auth_enabled=True, client_keys_raw="bladex-valid-key||test",
        upstream_model="openai/test", upstream_api_key="sk-fake", rocksdb_path=str(tmp_path / "ledger"),
    ))
    with TestClient(app) as client:
        resp = client.post("/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
            headers={"Authorization": "Bearer wrong-key"})
        assert resp.status_code == 401
        resp = client.post("/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]})
        assert resp.status_code == 401


def test_auth_valid_key_passes(tmp_path):
    app = create_app(ProxyConfig(
        auth_enabled=True, client_keys_raw="bladex-valid-key||test-agent",
        upstream_model="openai/test", upstream_api_key="sk-fake", rocksdb_path=str(tmp_path / "ledger"),
    ))
    fake = MagicMock()
    fake.choices = [MagicMock()]
    fake.choices[0].message.content = "OK"
    fake.model_dump.return_value = {"choices": [{"message": {"content": "OK"}}]}

    async def fake_call(model, messages, stream, **kw):
        return fake

    with patch("bladex_proxy.route.router_sdk.acompletion", new=fake_call):
        with TestClient(app) as client:
            resp = client.post("/v1/chat/completions",
                json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
                headers={"Authorization": "Bearer bladex-valid-key"})
            assert resp.status_code == 200


def test_auth_disabled_no_key_needed(tmp_path):
    app = create_app(ProxyConfig(
        auth_enabled=False, upstream_model="openai/test", upstream_api_key="sk-fake",
        rocksdb_path=str(tmp_path / "ledger"),
    ))
    fake = MagicMock()
    fake.choices = [MagicMock()]
    fake.choices[0].message.content = "OK"
    fake.model_dump.return_value = {"choices": [{"message": {"content": "OK"}}]}

    async def fake_call(model, messages, stream, **kw):
        return fake

    with patch("bladex_proxy.route.router_sdk.acompletion", new=fake_call):
        with TestClient(app) as client:
            resp = client.post("/v1/chat/completions",
                json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]})
            assert resp.status_code == 200


def test_upstream_error_returns_502_and_stores_failed():
    """ISSUE-5: 上游出错时返回 502 + 存 FAILED 轮次。"""
    app = create_app(_make_config())

    async def failing_call(model, messages, stream, **kw):
        raise RuntimeError("upstream 500")

    captured_turns: list = []

    async def capture_enqueue(turn):
        captured_turns.append(turn)

    with patch("bladex_proxy.route.router_sdk.acompletion", new=failing_call):
        with TestClient(app) as client:
            # mock Pipeline enqueue + close
            mock_pipeline = AsyncMock()
            mock_pipeline.enqueue = capture_enqueue
            mock_pipeline.close = AsyncMock()
            app.state.pipeline = mock_pipeline

            resp = client.post("/v1/chat/completions",
                json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}], "stream": False},
                headers={"Authorization": "Bearer any", "X-Agent-ID": "test"})

    assert resp.status_code == 502
    assert len(captured_turns) == 1
    assert captured_turns[0].status == TurnStatus.FAILED
    assert "upstream 500" in captured_turns[0].error


# ── T6 新增 ──

def test_models_endpoint():
    """T6: GET /v1/models 返回 200 + 配置的上游模型。"""
    app = create_app(_make_config())
    with TestClient(app) as client:
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) >= 1
        # 模型名去掉 provider 前缀
        assert data["data"][0]["id"] == "test"  # openai/test → test


def test_non_stream_tool_events_captured():
    """T6: 非流式带工具调用时 Turn.tool_events 非空。"""
    app = create_app(_make_config())

    fake_response = MagicMock()
    fake_response.choices = [MagicMock()]
    fake_response.choices[0].message.content = "Let me search for that."
    # 模拟 tool_calls
    fake_tc = MagicMock()
    fake_tc.function.name = "web_search"
    fake_tc.function.arguments = '{"query": "weather today"}'
    fake_tc.id = "call_abc123"
    fake_response.choices[0].message.tool_calls = [fake_tc]
    fake_response.model_dump.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "Let me search."}}],
    }

    captured_turns: list = []

    async def capture_enqueue(turn):
        captured_turns.append(turn)

    async def fake_acompletion(model, messages, stream, **kwargs):
        return fake_response

    with patch("bladex_proxy.route.router_sdk.acompletion", new=fake_acompletion):
        with TestClient(app) as client:
            mock_pipeline = AsyncMock()
            mock_pipeline.enqueue = capture_enqueue
            mock_pipeline.close = AsyncMock()
            app.state.pipeline = mock_pipeline

            resp = client.post(
                "/v1/chat/completions",
                json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Search weather"}], "stream": False},
                headers={"Authorization": "Bearer sk-test", "X-Agent-ID": "test"},
            )

    assert resp.status_code == 200
    assert len(captured_turns) == 1
    assert len(captured_turns[0].tool_events) == 1
    assert captured_turns[0].tool_events[0].tool_name == "web_search"


# ── T2: roundtrip / logical_turn 推断测试 ──
# ADR-0024 T1: `_infer_turn_metadata` 已退役，改由 TaskUnit 派生。
# 本测试保留**逐字相同的期望值**，作为新旧实现的等价性 oracle。

def test_turn_roundtrip_logical_turn():
    """工具循环（一个用户轮多次往返）→ 同 logical_turn、roundtrip 递增。

    ADR-0024 T1 等价性验收：期望值与退役前的 `_infer_turn_metadata` 完全一致。
    """
    from bladex_core.task_unit import build_task_units, derive_turn_metadata

    def meta(msgs):
        return derive_turn_metadata(build_task_units(msgs))

    # 普通一轮：[system, user] → (1, 0)
    msgs1 = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
    ]
    lt, rt = meta(msgs1)
    assert lt == 1
    assert rt == 0

    # 工具循环第1次往返：[system, user, assistant(tc), tool] → (1, 1)
    msgs2 = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "function": {"name": "search", "arguments": "{}"}}]},
        {"role": "tool", "name": "search", "content": "result"},
    ]
    lt, rt = meta(msgs2)
    assert lt == 1
    assert rt == 1

    # 工具循环第2次往返：(1, 2)
    msgs3 = msgs2 + [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "2", "function": {"name": "read", "arguments": "{}"}}]},
        {"role": "tool", "name": "read", "content": "data"},
    ]
    lt, rt = meta(msgs3)
    assert lt == 1
    assert rt == 2

    # 新用户轮：(2, 0)
    msgs4 = msgs3 + [
        {"role": "assistant", "content": "Done"},
        {"role": "user", "content": "Thanks"},
    ]
    lt, rt = meta(msgs4)
    assert lt == 2
    assert rt == 0


# ── ADR-0012 T1: 删除端点 ──

def test_delete_endpoint_writes_tombstone_only():
    """T1: DELETE /admin/turns/{key} 写墓碑，原文仍在 Memory Hub，scan_prefix 不再返回。"""
    import tempfile
    from pathlib import Path

    from bladex_proxy.models import Identity, Turn, TurnStatus
    from bladex_proxy.storage.memory_hub import MemoryHub

    tmpdir = tempfile.mkdtemp()
    config = ProxyConfig(
        hard_rules=["MUST be polite"],
        upstream_model="openai/test",
        upstream_api_key="sk-fake",
        rocksdb_path=f"{tmpdir}/rocksdb",
    )
    app = create_app(config)

    # 先写一条 turn 到 Memory Hub（在 server 启动前，避免 RocksDB 锁冲突）
    ledger_pre = MemoryHub(Path(tmpdir) / "rocksdb")
    ledger_pre.open()
    identity = Identity(user_id="u1", agent_id="a1", session_id="s1", turn_index=0)
    turn = Turn(identity=identity, response_text="hello", status=TurnStatus.OK)
    turn_key = identity.storage_key("1719907200-0")
    ledger_pre.put(turn_key, turn)
    ledger_pre.close()

    with TestClient(app) as client:
        # 用 server 自己的 Memory Hub 实例验证（避免锁冲突）
        ledger: MemoryHub = app.state.hub

        # 删除前 scan 能找到
        assert len(list(ledger.scan_prefix("u1/"))) == 1

        # 调 DELETE 端点
        resp = client.delete(f"/admin/turns/{turn_key}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "deleted"
        assert body["turn_key"] == turn_key
        assert body["tombstone_key"].startswith("tombstone/")

        # 删除后 scan_prefix 不再返回该 turn
        results = list(ledger.scan_prefix("u1/"))
        assert len(results) == 0

        # 原文仍在 Memory Hub
        loaded = ledger.get(turn_key)
        assert loaded is not None
        assert loaded.response_text == "hello"

        # 墓碑可扫描到
        tombstones = list(ledger.scan_tombstones())
        assert len(tombstones) == 1
        assert tombstones[0][1].target_key == turn_key


def test_delete_nonexistent_turn_returns_404():
    """T1: 删除不存在的 turn 返回 404。"""
    import tempfile

    config = ProxyConfig(
        hard_rules=["MUST be polite"],
        upstream_model="openai/test",
        upstream_api_key="sk-fake",
        rocksdb_path=f"{tempfile.mkdtemp()}/rocksdb",
    )
    app = create_app(config)

    with TestClient(app) as client:
        resp = client.delete("/admin/turns/nonexistent/key/0")
        assert resp.status_code == 404


def test_delete_with_auth_enabled_rejects_invalid_key():
    """T1: auth 开启时无效 key 被拒。"""
    import tempfile

    config = ProxyConfig(
        auth_enabled=True,
        client_keys_raw="bladex-valid-key-123||test-label",
        hard_rules=["MUST be polite"],
        upstream_model="openai/test",
        upstream_api_key="sk-fake",
        rocksdb_path=f"{tempfile.mkdtemp()}/rocksdb",
    )
    app = create_app(config)

    with TestClient(app) as client:
        # 无 key → 401
        resp = client.delete("/admin/turns/some/key/0")
        assert resp.status_code == 401

        # 无效 key → 401
        resp = client.delete(
            "/admin/turns/some/key/0",
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert resp.status_code == 401


# ── T8：路由空转告警 + STRICT 拒绝启动 ──


def _make_config_routing(strict: bool) -> ProxyConfig:
    tmpdir = tempfile.mkdtemp()
    cfg = ProxyConfig(
        hard_rules=["MUST be polite"],
        upstream_model="openai/test",
        upstream_api_key="sk-fake",
        rocksdb_path=f"{tmpdir}/rocksdb",
    )
    cfg.route_enabled = True
    cfg.route_strict = strict
    cfg.routing_config_path = "/nonexistent/routing.toml"
    return cfg


def test_route_inactive_logs_banner(capsys):
    """route_enabled=true 但无 routing.toml + STRICT=false → 启动成功但打醒目告警。"""
    app = create_app(_make_config_routing(strict=False))
    with TestClient(app):
        pass
    out = capsys.readouterr().out
    assert "ROUTE_ENABLED_BUT_INACTIVE" in out


def test_route_strict_refuses_startup():
    """STRICT=true + 路由空转 → 拒绝启动（fail-fast）。"""
    import pytest as _pytest
    app = create_app(_make_config_routing(strict=True))
    with _pytest.raises(RuntimeError, match="BLADEX_ROUTE_STRICT"):
        with TestClient(app):
            pass



# ── T1/A2: 客户端断开时 shield 入库 ──

import asyncio  # noqa: E402

from bladex_proxy.models import Identity  # noqa: E402
from bladex_proxy.server import _enqueue_turn_shielded  # noqa: E402


class _FakeRequest:
    """最小 Request 替身，只暴露 app.state 给 _enqueue_turn。"""
    def __init__(self, app):
        self.app = app


@pytest.mark.asyncio
async def test_enqueue_shielded_survives_disconnect():
    """A2: 外层被 cancel（客户端断开）时，shield 让入库在后台跑完、Turn 不丢。

    重现 A2 缺陷：generator finally 里 await _enqueue_turn 被外层 cancel。
    修复前：CancelledError -> Turn 丢。修复后：shield 派发独立 task，断开后继续入库。
    """
    app = create_app(_make_config())
    captured_turns: list = []

    async def capture_enqueue(turn):
        # 模拟入库延迟，确保 cancel 发生在 await 期间
        await asyncio.sleep(0.05)
        captured_turns.append(turn)

    mock_pipeline = AsyncMock()
    mock_pipeline.enqueue = capture_enqueue
    mock_pipeline.close = AsyncMock()
    app.state.pipeline = mock_pipeline

    identity = Identity(user_id="u", agent_id="a", session_id="s")

    # 外层 task：模拟 generator finally 里的 shielded 入库
    task = asyncio.create_task(_enqueue_turn_shielded(
        _FakeRequest(app), identity, "m",
        [{"role": "user", "content": "disconnect-me"}],
        "", "long response that got cut", [], TurnStatus.OK, "",
        0.0, 0.0, 0.0,
    ))
    await asyncio.sleep(0.01)  # 让它进入 await
    task.cancel()  # 模拟客户端断开
    try:
        await task
    except asyncio.CancelledError:
        pass  # 外层被取消，符合预期

    # 内部入库 task 仍在 app.state.bg_tasks 里跑 -> 等它完成
    bg = app.state.bg_tasks
    if bg:
        await asyncio.gather(*list(bg), return_exceptions=True)

    assert len(captured_turns) == 1, \
        "Turn must be enqueued even when outer is cancelled (A2)"
    assert captured_turns[0].response_text == "long response that got cut"


@pytest.mark.asyncio
async def test_enqueue_shielded_normal_path_awaits():
    """A2: 正常路径（无 cancel）下 shield 等价于直接 await，Turn 入库后返回。"""
    app = create_app(_make_config())
    captured_turns: list = []

    async def capture_enqueue(turn):
        captured_turns.append(turn)

    mock_pipeline = AsyncMock()
    mock_pipeline.enqueue = capture_enqueue
    mock_pipeline.close = AsyncMock()
    app.state.pipeline = mock_pipeline

    identity = Identity(user_id="u", agent_id="a", session_id="s")
    await _enqueue_turn_shielded(
        _FakeRequest(app), identity, "m",
        [{"role": "user", "content": "normal"}],
        "", "reply", [], TurnStatus.OK, "", 0.0, 0.0, 0.0,
    )

    assert len(captured_turns) == 1
    assert captured_turns[0].response_text == "reply"
