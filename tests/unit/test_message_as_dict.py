"""`server._message_as_dict`：上游 message → plain dict 的唯一实现点。

2026-09-02 四模块翻默认后，`/v1/responses` 非流式测试的 `SimpleNamespace` mock 首次走到
拦截路径，`dict(SimpleNamespace)` TypeError（gate 首跑红两条）。三种来源都要过：
pydantic（真上游）/ dict / 任意带 `__dict__` 的对象（递归到 tool_calls.function）。
"""

from __future__ import annotations

from types import SimpleNamespace

from bladex_proxy.server import _message_as_dict
from pydantic import BaseModel


class _Fn(BaseModel):
    name: str
    arguments: str


class _Call(BaseModel):
    id: str
    function: _Fn


class _Msg(BaseModel):
    content: str | None = None
    tool_calls: list[_Call] | None = None


def test_pydantic_message_uses_model_dump():
    m = _Msg(content=None, tool_calls=[_Call(id="c1", function=_Fn(name="bladex_memory_search", arguments="{}"))])
    d = _message_as_dict(m)
    assert d["tool_calls"][0]["function"]["name"] == "bladex_memory_search"


def test_namespace_message_is_converted_recursively():
    m = SimpleNamespace(content=None, tool_calls=[SimpleNamespace(
        id="call_1", function=SimpleNamespace(name="shell", arguments='{"cmd":"ls"}'))])
    d = _message_as_dict(m)
    assert d == {"content": None, "tool_calls": [
        {"id": "call_1", "function": {"name": "shell", "arguments": '{"cmd":"ls"}'}}]}


def test_dict_message_is_copied_not_aliased():
    src = {"role": "assistant", "content": "ok"}
    d = _message_as_dict(src)
    assert d == src and d is not src


def test_private_attrs_are_dropped():
    m = SimpleNamespace(content="x", _hidden=1)
    assert _message_as_dict(m) == {"content": "x"}
