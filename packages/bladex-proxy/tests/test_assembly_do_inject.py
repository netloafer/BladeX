"""`do_inject` × `ContextAssembler` 集成剧本（从 test_context_manager.py 拆出，2026-09-03 S3）。

CAP（ADR-0016 `context_manager.py`）随 S3 删除；L2 单元摘要随 MQ-CA9 清账（2026-09-18，
ADR-0033 §5-1）删除。本文件只剩三条阴性对照：**任何形态的会话经 do_inject 都不再出现摘要块**
——它们此前是"短会话不压缩 / 关闭不压缩 / 无装配器不压缩"，现在对长会话也成立。
"""

from __future__ import annotations

from bladex_proxy.assembly import SUMMARY_OPEN, AssemblyConfig, ContextAssembler
from bladex_proxy.config import ProxyConfig
from bladex_proxy.inject import do_inject


def _make_user(content: str = "Hello") -> dict:
    return {"role": "user", "content": content}


def _make_assistant(content: str = "Hi") -> dict:
    return {"role": "assistant", "content": content}


def _make_long_messages(count: int) -> list[dict]:
    """生成 count 条消息（交替 user/assistant，从 system 开始）。"""
    msgs: list[dict] = [{"role": "system", "content": "You are helpful."}]
    for i in range(count - 1):
        if i % 2 == 0:
            msgs.append(_make_user(f"User message {i}"))
        else:
            msgs.append(_make_assistant(f"Assistant reply {i}"))
    return msgs


def _no_summary(msgs: list[dict]) -> bool:
    return not any(SUMMARY_OPEN in str(m.get("content", "")) for m in msgs)


def test_short_session_no_compression():
    """短会话 -> 走叠加模式：消息数 = 原始 + 1（注入）。"""
    asm = ContextAssembler(AssemblyConfig(enabled=True))
    config = ProxyConfig(hard_rules=["MUST be polite"])
    messages = [
        {"role": "system", "content": "You are helpful."},
        _make_user("Hello"),
    ]
    new_messages, injected_text, _, _, _ = do_inject(
        messages, "Hello", config,
        assembler=asm,
    )
    assert len(new_messages) == len(messages) + 1
    assert SUMMARY_OPEN not in injected_text


def test_long_session_never_summarized():
    """长会话（40 条）+ 装配开 -> 不产摘要块、不丢消息（L2 已删的阴性对照）。"""
    asm = ContextAssembler(AssemblyConfig(enabled=True))
    config = ProxyConfig(hard_rules=["MUST be polite"])
    messages = _make_long_messages(40)
    new_messages, _, _, _, _ = do_inject(
        messages, "query", config,
        assembler=asm,
    )
    assert _no_summary(new_messages)
    assert len(new_messages) == len(messages) + 1
    for i in range(0, 39, 2):
        assert any(f"User message {i}" in str(m.get("content", "")) for m in new_messages)


def test_cap_disabled_no_compression():
    """装配关闭 -> 即使长会话也原样。"""
    asm = ContextAssembler(AssemblyConfig(enabled=False))
    config = ProxyConfig(hard_rules=["MUST be polite"])
    new_messages, _, _, _, _ = do_inject(
        _make_long_messages(30), "query", config,
        assembler=asm,
    )
    assert _no_summary(new_messages)


def test_no_context_manager_no_compression():
    """assembler=None -> 不压缩（向后兼容）。"""
    config = ProxyConfig(hard_rules=["MUST be polite"])
    new_messages, _, _, _, _ = do_inject(
        _make_long_messages(30), "query", config,
    )
    assert _no_summary(new_messages)
