"""批 AC · MQ-P34 / MQ-CA20 剩余 —— 上游 usage 原文无损入 Hub（六格）+ 非流式三路径登记 cache 读数。

## 为什么 `test_hub_lossless_guard` 不够

它查的是**切片**（`[:200]` 这类截断）。MQ-P34 的形态是**挑字段**：`extract_usage` 只留
`{prompt, completion, total}`，cache 读写 / 推理 token 被整键丢掉——没有切片，静态守卫看不见。
本文件对**六格**（chat / anthropic / responses × 流式 / 非流式）逐格断言：
**上游 usage 的每个键（含嵌套键）都原样出现在 Hub 那一轮的 `response_meta.usage_raw` 里**。

夹具 = `fixtures/ac_upstream_usage_shapes.json`：`scripts/probe_upstream_usage_shapes.py` 直连上游
录的真实形状（OpenAI 兼容 5 键；Anthropic 适配器 7 / 10 键，含 `cache_creation_input_tokens`）。

## 层 4 反问：上游换一个字段名、值照样好看，会不会照样全绿？

会全绿的写法是"断言 `usage_raw` 里有 `prompt_tokens`"——那只证明了我们**认识的**键还在。
这里断言的是**夹具里的每一个键路径**，并额外喂一个夹具之外、我们代码里从未出现过的键
（`TestUnknownKeySurvives`）：换名 / 新增的键若被挑掉，这里红。
另：断言比的是**值**逐个相等（不只是键在），且值来自夹具——上游给 0 我们存成别的数也会红。

## 三协议怎么喂

六个 handler（`_handle_{,anthropic_,responses_}{stream,non_stream}`）直接调，`call_model` 换成
回放夹具 usage 的桩，`_enqueue_turn*` 换成"构造 Turn → `MemoryHub.put` 到 tmpdir → `get` 回来"。
走的是生产的 handler 与生产的 Hub 序列化（msgpack），不经 Redis。
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

FIXTURE = Path(__file__).parent / "fixtures" / "ac_upstream_usage_shapes.json"
CELLS = [c for c in json.loads(FIXTURE.read_text())["cells"] if "usage" in c]


# ── 上游载体：三种都要认（pydantic / dict / 只有属性的对象）────────────────────

class _PUsage(BaseModel):
    """Router 给的是 pydantic（litellm `Usage`）；这里用 extra=allow 的同形替身，不 import 供应商 SDK。"""
    model_config = ConfigDict(extra="allow")


def _ns(obj: Any) -> Any:  # noqa: ANN401
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in obj.items()})
    return obj


def _carrier(raw: dict, kind: str) -> Any:  # noqa: ANN401
    if kind == "pydantic":
        return _PUsage(**raw)
    if kind == "dict":
        return json.loads(json.dumps(raw))
    return _ns(raw)


def _paths(obj: Any, prefix: str = "") -> dict[str, Any]:  # noqa: ANN401
    """嵌套 dict → {键路径: 叶子值}（夹具与 Hub 两侧同一把尺子）。"""
    out: dict[str, Any] = {}
    for k, v in obj.items():
        p = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_paths(v, p + "."))
        else:
            out[p] = v
    return out


# ── 上游桩 ───────────────────────────────────────────────────────────────────

def _nonstream_response(usage: Any) -> Any:  # noqa: ANN401
    msg = SimpleNamespace(role="assistant", content="ok", tool_calls=None)
    msg.model_dump = lambda: {"role": "assistant", "content": "ok"}  # noqa: E731
    resp = SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")], usage=usage)
    resp.model_dump = lambda: {"choices": [{"message": {"role": "assistant", "content": "ok"},  # noqa: E731
                                            "finish_reason": "stop"}]}
    return resp


class _Stream:
    """流：一个正文块 + 一个带 usage 的收尾块（`include_usage` 的形状）。

    收尾块两种形状都要认：`router` = Router 实测给的（choices=1、空 delta、usage 挂在上面；
    AC0 直连 18 次逐次核过）；`openai` = OpenAI 原生（`choices=[]` + usage）。
    """

    def __init__(self, usage: Any, tail: str = "router") -> None:  # noqa: ANN401
        delta = SimpleNamespace(content="ok", tool_calls=None, reasoning_content=None, role="assistant")
        first = SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason="stop", index=0)],
                                usage=None)
        first.model_dump = lambda: {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}  # noqa: E731
        if tail == "router":
            empty = SimpleNamespace(content=None, tool_calls=None, reasoning_content=None, role=None)
            last = SimpleNamespace(choices=[SimpleNamespace(delta=empty, finish_reason=None, index=0)],
                                   usage=usage)
        else:
            last = SimpleNamespace(choices=[], usage=usage)
        last.model_dump = lambda: {"choices": []}  # noqa: E731
        self._chunks = [first, last]

    def __aiter__(self) -> _Stream:
        return self

    async def __anext__(self) -> Any:  # noqa: ANN401
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


# ── 驱动六个 handler ─────────────────────────────────────────────────────────

class _OffAgency:
    interception_on = False


def _request() -> Any:  # noqa: ANN401
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
        agency=_OffAgency(), router=None, model_health=None, hub=None)))


def _identity(session: str = "ac-sess") -> Any:  # noqa: ANN401
    from bladex_proxy.models import Identity
    return Identity(user_id="u", agent_id="ac-agent", session_id=session)


def _drive(protocol: str, stream: bool, usage: Any, monkeypatch, hub, session: str = "ac-sess",  # noqa: ANN001, ANN201
           tail: str = "router"):
    """跑一格，返回 Hub 里读回来的那一轮 Turn。"""
    from bladex_proxy.models import ChatCompletionRequest, Turn
    from bladex_proxy.server import endpoints_anthropic as ea
    from bladex_proxy.server import endpoints_chat as ec
    from bladex_proxy.server import endpoints_responses as er

    mod = {"chat": ec, "anthropic": ea, "responses": er}[protocol]
    stored: list[Turn] = []

    async def _call_model(model, messages, stream=True, **_kw):  # noqa: ANN001, ANN202
        return _Stream(usage, tail) if stream else _nonstream_response(usage)

    async def _enqueue(request, identity, model, *_a, response_meta=None, **_kw):  # noqa: ANN001, ANN202
        key = f"{identity.session_prefix()}{len(stored):04d}-0"
        hub.put(key, Turn(identity=identity, model=model, response_meta=response_meta))
        stored.append(hub.get(key))

    monkeypatch.setattr(mod, "call_model", _call_model)
    monkeypatch.setattr(mod, "_enqueue_turn", _enqueue)
    if hasattr(mod, "_enqueue_turn_shielded"):
        monkeypatch.setattr(mod, "_enqueue_turn_shielded", _enqueue)

    ident = _identity(session)
    msgs = [{"role": "user", "content": "hi"}]
    common = dict(identity=ident, injected_text="", ms_identity=0.0, ms_inject=0.0, t_start=0.0)

    async def _run() -> None:
        if protocol == "chat":
            req = ChatCompletionRequest(model="m", messages=msgs, stream=stream)
            fn = ec._handle_stream if stream else ec._handle_non_stream
            resp = await fn(_request(), "m", None, [], msgs, req, **common)
        else:
            fn = {("anthropic", True): ea._handle_anthropic_stream,
                  ("anthropic", False): ea._handle_anthropic_non_stream,
                  ("responses", True): er._handle_responses_stream,
                  ("responses", False): er._handle_responses_non_stream}[(protocol, stream)]
            resp = await fn(_request(), "m", None, [], msgs, {}, **common)
        if stream:
            async for _ in resp.body_iterator:
                pass

    asyncio.run(_run())
    assert len(stored) == 1, f"{protocol}/{stream}: 应入库 1 轮，实际 {len(stored)}"
    return stored[0]


@pytest.fixture
def hub(tmp_path):  # noqa: ANN001, ANN201
    from bladex_proxy.storage.memory_hub import MemoryHub
    h = MemoryHub(tmp_path / "hub")
    h.open()
    yield h
    h.close()


def _cell_id(c: dict) -> str:
    return f"{c['model'].split('/')[0]}-{c['model'].split('/')[-1]}-{c['protocol']}-{'stream' if c['stream'] else 'nonstream'}"


# ── 六格守卫 ─────────────────────────────────────────────────────────────────

class TestSixCellsLossless:
    def test_fixture_covers_all_six_cells(self):
        cells = {(c["protocol"], c["stream"]) for c in CELLS}
        assert cells == {(p, s) for p in ("chat", "anthropic", "responses") for s in (True, False)}
        assert any("cache_creation_input_tokens" in c["usage"] for c in CELLS), \
            "夹具里要有 Anthropic 适配器形状（cache 写入键），否则守卫只覆盖了 OpenAI 五键"

    @pytest.mark.parametrize("carrier", ["pydantic", "dict", "namespace"])
    @pytest.mark.parametrize("cell", CELLS, ids=_cell_id)
    def test_every_upstream_key_reaches_hub(self, cell, carrier, monkeypatch, hub):
        turn = _drive(cell["protocol"], cell["stream"], _carrier(cell["usage"], carrier), monkeypatch, hub)
        want = _paths(cell["usage"])
        got = _paths(turn.response_meta.usage_raw)
        missing = sorted(set(want) - set(got))
        assert not missing, f"上游 usage 键没进 Hub：{missing}"
        changed = {p: (want[p], got[p]) for p in want if got[p] != want[p]}
        assert not changed, f"值被改了：{changed}"
        # 三键投影照旧（下游都读它）
        assert turn.response_meta.usage["prompt"] == cell["usage"]["prompt_tokens"]
        assert turn.response_meta.usage["completion"] == cell["usage"]["completion_tokens"]


class TestOpenAITailShape:
    """收尾块 `choices=[]`（OpenAI 原生）三条流式路径都要拿到 usage——批 AC 前 anthropic / responses
    把 usage 放在 `choices[0]` 的 try 里，这个形状会被 IndexError 整块跳过。"""

    @pytest.mark.parametrize("protocol", ["chat", "anthropic", "responses"])
    def test_empty_choices_tail_keeps_usage(self, protocol, monkeypatch, hub):
        raw = next(c["usage"] for c in CELLS if "cache_creation_input_tokens" in c["usage"])
        turn = _drive(protocol, True, _carrier(raw, "pydantic"), monkeypatch, hub, tail="openai")
        assert turn.response_meta.usage_raw == raw


class TestUnknownKeySurvives:
    """层 4：夹具之外、代码里从未出现过的键也必须原样入库（换名 / 新增字段不被挑掉）。"""

    @pytest.mark.parametrize("protocol", ["chat", "anthropic", "responses"])
    @pytest.mark.parametrize("stream", [True, False])
    def test_unseen_key_and_nested_key_survive(self, protocol, stream, monkeypatch, hub):
        raw = {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10,
               "zz_new_cache_hit_count": 5, "zz_details": {"zz_inner": 2, "zz_none": None}}
        turn = _drive(protocol, stream, _carrier(raw, "pydantic"), monkeypatch, hub)
        assert turn.response_meta.usage_raw == raw


class TestNoData:
    """上游没报 ⇒ 原文为空（NO-DATA），不是一份全 0 的 usage。"""

    @pytest.mark.parametrize("protocol", ["chat", "anthropic", "responses"])
    @pytest.mark.parametrize("stream", [True, False])
    def test_missing_usage_stays_empty(self, protocol, stream, monkeypatch, hub):
        turn = _drive(protocol, stream, None, monkeypatch, hub)
        assert turn.response_meta.usage_raw == {}

    def test_legacy_turn_without_field_reads_as_nodata(self):
        """批 AC 之前写入的轮次没有 `usage_raw` 键 ⇒ 读回是空，聚合按 NO-DATA 计，不当 0。"""
        from bladex_proxy.models import ResponseMeta
        from bladex_proxy.usage import turn_usage
        legacy = {"reasoning_text": "", "usage": {"prompt": 9, "completion": 2, "total": 11},
                  "finish_reason": "stop", "ms_first_chunk": 0.0}
        assert ResponseMeta.model_validate(legacy).usage_raw == {}
        u = turn_usage(legacy)
        assert u == {"input": 9, "output": 2, "cache_read": None}

    def test_rollback_reader_ignores_new_field(self):
        """回滚到 0.3.0：旧 `ResponseMeta`（无 `usage_raw`、pydantic 默认 extra=ignore）读新轮次不报错。"""
        from pydantic import Field

        class OldResponseMeta(BaseModel):          # 0.3.0 的字段集逐字照抄
            reasoning_text: str = ""
            usage: dict[str, int] = Field(default_factory=dict)
            finish_reason: str = ""
            ms_first_chunk: float = 0.0

        from bladex_proxy.models import ResponseMeta
        assert ResponseMeta.model_config.get("extra", "ignore") == "ignore"
        new = {"usage": {"prompt": 1, "completion": 1, "total": 2},
               "usage_raw": {"prompt_tokens": 1, "prompt_tokens_details": {"cached_tokens": 0}}}
        old = OldResponseMeta.model_validate(new)
        assert old.usage == {"prompt": 1, "completion": 1, "total": 2}


class TestInnerLoopRawReachesHub:
    """BladeX 自身调用里唯一入 Hub 的一类：内循环 aux 轮——原文同样无损。"""

    def test_loop_reply_carries_raw_and_turn_stores_it(self):
        from bladex_proxy.loopledger import PendingRound, build_inner_loop_turn
        from bladex_proxy.server.orchestration import _loop_reply
        raw = next(c["usage"] for c in CELLS if "cache_creation_input_tokens" in c["usage"])
        reply = _loop_reply(_nonstream_response(_PUsage(**raw)))
        assert reply["usage"] == raw
        turn = build_inner_loop_turn(_identity(), "m", PendingRound(
            round_index=1, rounds_total=1, usage=reply["usage"], reply={"content": "ok"}))
        assert turn.response_meta.usage_raw == raw
        assert turn.response_meta.usage == {"prompt": raw["prompt_tokens"],
                                            "completion": raw["completion_tokens"],
                                            "total": raw["total_tokens"]}


# ── MQ-CA20 剩余：非流式三路径登记 cache 读数 ─────────────────────────────────

class TestNonStreamCacheReading:
    """下一轮装配触发门读到的 cache 读数 = 上游回报值（流式三路径早已如此，非流式此前恒"无读数"）。"""

    @pytest.mark.parametrize("protocol", ["chat", "anthropic", "responses"])
    @pytest.mark.parametrize("cached", [0, 4096])
    def test_gate_reads_upstream_value(self, protocol, cached, monkeypatch, hub):
        from bladex_proxy.cache_state import registry
        session = f"ac-ns-{protocol}-{cached}"
        raw = {"prompt_tokens": 5000, "completion_tokens": 3, "total_tokens": 5003,
               "prompt_tokens_details": {"cached_tokens": cached}}
        _drive(protocol, False, _carrier(raw, "pydantic"), monkeypatch, hub, session=session)
        st = registry().get(session)
        assert st is not None and st.last_cache_read == cached, "非流式路径没登记 cache 读数"
        assert registry().cold_reason(session) == ("" if cached else "cache_read_zero")

    @pytest.mark.parametrize("protocol", ["chat", "anthropic", "responses"])
    def test_unreported_is_registered_as_none(self, protocol, monkeypatch, hub):
        from bladex_proxy.cache_state import registry
        session = f"ac-ns-none-{protocol}"
        _drive(protocol, False, _carrier({"prompt_tokens": 1, "completion_tokens": 1,
                                          "total_tokens": 2}, "pydantic"), monkeypatch, hub, session=session)
        st = registry().get(session)
        assert st is not None and st.last_cache_read_at > 0, "上游没报也要登记（None），不能沿用旧值"
        assert st.last_cache_read is None
        assert registry().cold_reason(session) == "no_reading"
