"""`do_inject` × `ContextAssembler` 集成剧本（从 test_context_manager.py 拆出，2026-09-03 S3）。

CAP（ADR-0016 `context_manager.py`）随 S3 删除；本文件只保留走 ADR-0019 装配器的
五条 do_inject 剧本，逐字沿用（S3 的验收：装配行为逐字不变）。
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


# ── 短会话 + 装配关闭兼容性测试 ──

def test_short_session_no_compression():
    """短会话（< 阈值）-> 不触发 CAP，走叠加模式。"""
    asm = ContextAssembler(AssemblyConfig(enabled=True, msg_threshold=30))
    config = ProxyConfig(hard_rules=["MUST be polite"])
    messages = [
        {"role": "system", "content": "You are helpful."},
        _make_user("Hello"),
    ]
    new_messages, injected_text, _, _, _ = do_inject(
        messages, "Hello", config,
        assembler=asm,
    )
    # 短会话不压缩 -> 消息数 = 原始 + 1 (注入)
    assert len(new_messages) == len(messages) + 1
    # 没有摘要
    assert SUMMARY_OPEN not in injected_text


def test_cap_disabled_no_compression():
    """CAP 关闭 -> 即使超阈值也不压缩。"""
    asm = ContextAssembler(AssemblyConfig(enabled=False, msg_threshold=5))
    config = ProxyConfig(hard_rules=["MUST be polite"])
    messages = _make_long_messages(30)
    new_messages, _, _, _, _ = do_inject(
        messages, "query", config,
        assembler=asm,
    )
    # CAP 关闭 -> 没有摘要消息
    assert not any(SUMMARY_OPEN in str(m.get("content", "")) for m in new_messages)


def test_no_context_manager_no_compression():
    """assembler=None -> 不压缩（向后兼容）。"""
    config = ProxyConfig(hard_rules=["MUST be polite"])
    messages = _make_long_messages(30)
    new_messages, _, _, _, _ = do_inject(
        messages, "query", config,
        # 不传 assembler
    )
    assert not any(SUMMARY_OPEN in str(m.get("content", "")) for m in new_messages)


# ── do_inject 集成测试 ──

def test_do_inject_with_cap_compresses_long_session():
    """长会话 + CAP -> 压缩 + 注入。"""
    asm = ContextAssembler(AssemblyConfig(enabled=True, msg_threshold=10, preserve_units=2))
    config = ProxyConfig(hard_rules=["MUST be polite"])
    messages = _make_long_messages(40)
    original_count = len(messages)

    new_messages, injected_text, _, _, _ = do_inject(
        messages, "query", config,
        assembler=asm,
    )
    # 压缩后消息数应少于原始
    assert len(new_messages) < original_count
    # 有摘要
    assert any(SUMMARY_OPEN in str(m.get("content", "")) for m in new_messages)
    # 有注入（BladeX memory）
    from bladex_proxy.inject import MEMORY_OPEN
    assert any(MEMORY_OPEN in str(m.get("content", "")) for m in new_messages)


def test_do_inject_cap_preserves_recent_turns():
    """CAP 压缩后近期对话原文保留。"""
    asm = ContextAssembler(AssemblyConfig(enabled=True, msg_threshold=10, preserve_units=3))
    config = ProxyConfig(hard_rules=["MUST be polite"])
    # 构造有独特内容的近期对话
    messages = [
        {"role": "system", "content": "You are helpful."},
    ]
    for i in range(20):
        messages.append(_make_user(f"User question number {i}"))
        messages.append(_make_assistant(f"Assistant answer number {i}"))

    new_messages, _, _, _, _ = do_inject(
        messages, "latest query", config,
        assembler=asm,
    )

    # 最近 3 轮的 user 消息原文应在压缩后的 messages 中
    for i in range(17, 20):
        assert any(
            f"User question number {i}" in str(m.get("content", ""))
            for m in new_messages
        ), f"Recent user message {i} should be preserved"

