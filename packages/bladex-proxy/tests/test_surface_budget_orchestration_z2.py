"""批 Z · Z2 编排级守卫（真 app + 真 AgencyRuntime + 假上游，三协议）。

① 没配 `context_window` ⇒ 发上游的请求体逐字 = 关掉预算机制时的形态（三协议各一条）；
④ 配了小窗口：只裁 BladeX 注入块，agent 正文与 max_tokens 逐字不变，硬规则块仍在，裁剪有日志；
⑦ 长会话：稳定层（about 哈希 + 工具列表）恒定，只有动态层被裁；
②/③ 应用层：动态选池绕开放不下的模型；静态钉死的放不下 ⇒ 不换 + 告警。
"""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import structlog.testing
from bladex_proxy.config import ProxyConfig
from bladex_proxy.server import create_app
from fastapi.testclient import TestClient

HARD_RULE = "MUST answer in Chinese"
OWN_TOOL = {"type": "function", "function": {"name": "agent_tool", "description": "agent's own",
                                             "parameters": {"type": "object", "properties": {}}}}


def _toml(tmp: Path, *, small_window: int, big_window: int = 0, pin_agent: str = "") -> Path:
    body = f'''
[[models]]
name = "openai/small"
api_base = "http://127.0.0.1:9/v1"
api_key_env = "ZT_KEY"
capabilities = ["text", "code"]
tier = "medium"
provider = "p1"
{f"context_window = {small_window}" if small_window else ""}

[[models]]
name = "openai/big"
api_base = "http://127.0.0.1:9/v1"
api_key_env = "ZT_KEY"
capabilities = ["text", "code"]
tier = "medium"
provider = "p2"
{f"context_window = {big_window}" if big_window else ""}

[upstream]
models = ["openai/small", "openai/big"]
'''
    if pin_agent:
        body += f'''
[strategies.agent]
enabled = true
[strategies.agent.map]
"{pin_agent}" = ["openai/small", "openai/big"]
'''
    p = tmp / "routing.toml"
    p.write_text(body, encoding="utf-8")
    return p


def _cfg(routing: Path) -> ProxyConfig:
    return ProxyConfig(hard_rules=[HARD_RULE], upstream_model="openai/small",
                       upstream_api_key="sk-fake", route_enabled=True,
                       routing_config_path=str(routing),
                       rocksdb_path=f"{tempfile.mkdtemp()}/rocksdb")


class _Usage:
    prompt_tokens, completion_tokens, total_tokens = 300, 4, 304

    def model_dump(self):
        return {"prompt_tokens": 300, "completion_tokens": 4, "total_tokens": 304}


class _Resp:
    def __init__(self):
        msg = type("M", (), {"content": "好的", "tool_calls": None,
                             "model_dump": lambda self: {"role": "assistant", "content": "好的"}})()
        self.choices = [type("C", (), {"message": msg, "finish_reason": "stop"})()]
        self.usage = _Usage()

    def model_dump(self):
        return {"id": "x", "choices": [{"index": 0, "message": {"role": "assistant", "content": "好的"},
                                        "finish_reason": "stop"}], "usage": self.usage.model_dump()}


def _capturing_upstream(seen: list[dict]):
    async def fake(model, messages, stream, **kw):
        seen.append({"model": model, "messages": json.loads(json.dumps(messages)),
                     "tools": json.loads(json.dumps(kw.get("tools"))), "max_tokens": kw.get("max_tokens")})
        return _Resp()
    return fake


def _post(client, proto: str, *, session: str, user: str, agent: str = "probe-z2"):
    hdr = {"X-Agent-ID": agent, "X-Session-ID": session}
    if proto == "chat":
        return client.post("/v1/chat/completions", json={
            "model": "gpt-4o", "stream": False, "max_tokens": 64, "tools": [OWN_TOOL],
            "messages": [{"role": "system", "content": "You are a test agent."},
                         {"role": "user", "content": user}]},
            headers={"Authorization": "Bearer sk", **hdr})
    if proto == "messages":
        return client.post("/v1/messages", json={
            "model": "claude-x", "stream": False, "max_tokens": 64, "system": "You are a test agent.",
            "tools": [{"name": "agent_tool", "description": "agent's own",
                       "input_schema": {"type": "object", "properties": {}}}],
            "messages": [{"role": "user", "content": user}]},
            headers={"x-api-key": "sk", **hdr})
    return client.post("/v1/responses", json={
        "model": "gpt-x", "stream": False, "max_output_tokens": 64,
        "instructions": "You are a test agent.", "input": user,
        "tools": [{"type": "function", "name": "agent_tool", "description": "agent's own",
                   "parameters": {"type": "object", "properties": {}}}]},
        headers={"Authorization": "Bearer sk", **hdr})


def _run(monkeypatch, tmp_path, *, small_window: int, big_window: int = 0, pin_agent: str = "",
         calls: list[tuple], patch_budget_off: bool = False):
    monkeypatch.setenv("ZT_KEY", "sk")
    for n in ("TOOLFACE", "LEDGER"):
        monkeypatch.setenv(f"BLADEX_MODULE_{n}", "1")
    app = create_app(_cfg(_toml(tmp_path, small_window=small_window, big_window=big_window,
                                pin_agent=pin_agent)))
    seen: list[dict] = []
    patches = [patch("bladex_proxy.route.router_sdk.acompletion", new=_capturing_upstream(seen))]
    if patch_budget_off:
        patches.append(patch("bladex_proxy.server.orchestration.surface_budget_for",
                             new=lambda **kw: None))
    with patches[0], (patches[1] if patch_budget_off else patch("builtins.len", new=len)):
        with structlog.testing.capture_logs() as cap:
            with TestClient(app) as client:
                for proto, session, user, *rest in calls:
                    r = _post(client, proto, session=session, user=user, agent=(rest[0] if rest else "probe-z2"))
                    assert r.status_code == 200, r.text
    return seen, cap


def _bladex_blocks(messages: list[dict]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {"about": [], "memory": [], "ledger": []}
    for i, m in enumerate(messages):
        c = str(m.get("content") or "")
        if "<bladex-about>" in c:
            out["about"].append(i)
        if "<bladex-memory>" in c:
            out["memory"].append(i)
        if "<bladex-ledger>" in c:
            out["ledger"].append(i)
    return out


def _agent_only(messages: list[dict]) -> list[dict]:
    return [m for m in messages if not any(mk in str(m.get("content") or "")
                                           for mk in ("<bladex-about>", "<bladex-memory>", "<bladex-ledger>"))]


# ── ① 没配窗口：三协议逐字零行为 ──

@pytest.mark.parametrize("proto", ["chat", "messages", "responses"])
def test_no_context_window_is_verbatim_zero_behaviour(monkeypatch, tmp_path, fresh_redis_stream,
                                                       proto):
    seen_on, cap = _run(monkeypatch, tmp_path, small_window=0, calls=[(proto, "s-1", "写一篇长文")])
    # 两个独立 app 做对照、不是重启语义：换一条 stream，免等 4 s 实例判活窗口（批 GS）
    monkeypatch.setenv("BLADEX_REDIS_STREAM", fresh_redis_stream())
    seen_off, _ = _run(monkeypatch, tmp_path, small_window=0, calls=[(proto, "s-1", "写一篇长文")],
                       patch_budget_off=True)
    assert seen_on[0] == seen_off[0], "没配窗口时请求体必须与机制不存在时逐字相同"
    body = seen_on[0]
    blocks = _bladex_blocks(body["messages"])
    assert blocks["about"] == [1], "about 在 agent 开场 system 之后（定格首轮 = 老规则）"
    assert blocks["memory"] and blocks["ledger"] == [len(body["messages"]) - 1]
    assert HARD_RULE in body["messages"][blocks["memory"][0]]["content"]
    names = [t.get("function", t).get("name") for t in body["tools"]]
    assert names[0] == "agent_tool" and {"bladex_memory_search", "bladex_ledger_switch"} <= set(names)
    assert body["max_tokens"] == 64
    assert not [e for e in cap if e.get("event") in ("agency_surface_trimmed", "agency_surface_budget")]


# ── ④ 小窗口：只裁 BladeX 注入块 ──

def test_small_window_trims_only_bladex_surfaces_and_logs(monkeypatch, tmp_path):
    # 4000 token × 2.5 − 2048 floor：≈ 4,880 字符预算，放不下 ~17K 的稳定层 ⇒ 稳定层裁到只剩记忆族
    user = "写一篇长文" * 3
    seen, cap = _run(monkeypatch, tmp_path, small_window=4000, calls=[("chat", "s-4", user)])
    body = seen[0]
    # agent 正文逐字不变、max_tokens 不变
    assert _agent_only(body["messages"]) == [{"role": "system", "content": "You are a test agent."},
                                             {"role": "user", "content": user}]
    assert body["max_tokens"] == 64
    # 硬规则永不裁
    blocks = _bladex_blocks(body["messages"])
    assert blocks["memory"] and HARD_RULE in body["messages"][blocks["memory"][0]]["content"]
    # 裁剪有日志，且顺序符合 TRIM_ORDER
    steps = [e["step"] for e in cap if e.get("event") == "agency_surface_trimmed"]
    assert steps[:2] == ["ledger_family", "about"], steps
    names = [t.get("function", t).get("name") for t in body["tools"]]
    assert "agent_tool" in names and not any(n.startswith("bladex_ledger_") for n in names)
    assert not blocks["about"]
    budget_rows = [e for e in cap if e.get("event") == "agency_surface_budget"]
    assert budget_rows and budget_rows[0]["context_window"] == 4000


# ── ⑦ 长会话：稳定层哈希恒定，只有动态层被裁 ──

def test_long_session_keeps_stable_layer_frozen_and_trims_only_dynamic(monkeypatch, tmp_path):
    # 首轮：40K 窗口，prompt 很小 ⇒ 全放得下；之后正文暴涨 ⇒ 预算变负，只许裁动态层
    huge = "历史很长" * 30_000        # 12 万字符 ≈ 48K token > 窗口
    # 两个模型都 40K：放不下时选池剔空 ⇒ 保留原池（不打死请求），仍落在 small 上——
    # 否则第二轮会被 Z2.2 正确地路由到没配窗口的 big，本条就测不到裁剪。
    seen, cap = _run(monkeypatch, tmp_path, small_window=40_000, big_window=40_000,
                     calls=[("chat", "s-7", "第一轮"), ("chat", "s-7", huge), ("chat", "s-7", huge + "再来")])
    assert len(seen) == 3
    def _about_hash(body):
        b = _bladex_blocks(body["messages"])
        return hashlib.sha256(body["messages"][b["about"][0]]["content"].encode()).hexdigest() if b["about"] else ""
    h = [_about_hash(b) for b in seen]
    assert h[0] and len(set(h)) == 1, "稳定层内容哈希在长会话里必须恒定"
    tools = [[t.get("function", t).get("name") for t in b["tools"]] for b in seen]
    assert tools[0] == tools[1] == tools[2]
    assert _bladex_blocks(seen[0]["messages"])["ledger"], "首轮账本块在（放得下）"
    assert not _bladex_blocks(seen[1]["messages"])["ledger"], "第二轮放不下 ⇒ 只裁动态层（账本块）"
    rows = [e for e in cap if e.get("event") == "agency_surface_trimmed"]
    assert rows and all(r["layer"] == "dynamic" for r in rows), rows
    assert all(_bladex_blocks(b["messages"])["memory"] for b in seen), "硬规则三轮都在"


# ── ②/③ 应用层：动态选池绕开、静态钉死不换 ──

def test_dynamic_route_skips_small_window_model_at_app_level(monkeypatch, tmp_path):
    huge = "长" * 20_000    # ≈ 8K token > small(4000)，big(1M) 放得下
    seen, cap = _run(monkeypatch, tmp_path, small_window=4000, big_window=1_000_000,
                     calls=[("chat", "s-2", huge)])
    assert seen[0]["model"] == "openai/big"
    row = next(e for e in cap if e.get("event") == "route_decision")
    assert row["context_window"] == 1_000_000 and row["prompt_tokens_est"] > 4000


def test_static_pin_is_kept_and_warned_at_app_level(monkeypatch, tmp_path, caplog):
    import logging
    huge = "长" * 20_000
    with caplog.at_level(logging.WARNING, logger="bladex_core.routing"):
        seen, cap = _run(monkeypatch, tmp_path, small_window=4000, big_window=1_000_000,
                         pin_agent="pinned", calls=[("chat", "s-3", huge, "pinned")])
    assert seen[0]["model"] == "openai/small", "静态钉死不许被窗口判断改写"
    assert "route_static_context_window_exceeded" in caplog.text
    assert "agent=pinned" in caplog.text and "model=openai/small" in caplog.text
    assert "routing.toml" in caplog.text
