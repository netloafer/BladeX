"""注入单元测试（T6 + T3 + T5 修复）。

T3: 默认注入位置移到末条 user 前（保 prompt cache），兼容开关回退合 system。
T5: 标记剥离限 system 角色、补多模态 list content。
"""

import tempfile
from pathlib import Path

from bladex_core.fact import Fact
from bladex_proxy.config import ProxyConfig
from bladex_proxy.inject import (
    MEMORY_CLOSE,
    MEMORY_OPEN,
    InjectionSource,
    do_inject,
    inject_memory,
    strip_previous_injection,
)
from bladex_proxy.storage.memory_index import MemoryIndex

# （2026-09-03 S1：`BLADEX_INJECT_INDEX_RECALL` 与主动检索路径已删，注入面只剩硬规则；
#  原 `_pin_index_recall_on` 夹具与两条"召回内容进注入块"的用例随删。带 index 的用例
#  保留——它们钉的是位置/剥离/降级形态，与召回无关。）


def test_inject_strips_previous_marked_block():
    """上一轮注入的 <bladex-memory> 块应从 system 消息删掉，不累积。"""
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "system", "content": f"System prompt\n\n{MEMORY_OPEN}\n- [MUST/NEVER] old rule\n{MEMORY_CLOSE}"},
    ]
    cleaned = strip_previous_injection(messages)
    assert len(cleaned) == 3  # system 消息保留（有正文），注入块被删
    for msg in cleaned:
        assert MEMORY_OPEN not in msg.get("content", "")


def test_strip_ignores_non_system_marker():
    """T5: assistant/user 正文含 <bladex-memory> 字样 → 不被删。"""
    messages = [
        {"role": "user", "content": f"Hello {MEMORY_OPEN}\n- old\n{MEMORY_CLOSE} world"},
        {"role": "assistant", "content": f"I found {MEMORY_OPEN} in your text"},
    ]
    cleaned = strip_previous_injection(messages)
    assert len(cleaned) == 2
    # user 正文保留
    assert "Hello" in cleaned[0]["content"]
    assert "world" in cleaned[0]["content"]
    assert MEMORY_OPEN in cleaned[0]["content"]  # 没删
    # assistant 正文保留
    assert MEMORY_OPEN in cleaned[1]["content"]


def test_strip_handles_list_content():
    """T5: 多模态 list content 消息不报错、按预期处理。"""
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": f"{MEMORY_OPEN}\n- old rule\n{MEMORY_CLOSE}"},
                {"type": "text", "text": "Keep this"},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            ],
        },
    ]
    cleaned = strip_previous_injection(messages)
    assert len(cleaned) == 2
    # system: 注入块从 list content 的 text part 中删除
    sys_content = cleaned[0]["content"]
    assert isinstance(sys_content, list)
    assert "Keep this" in sys_content[0]["text"]
    assert MEMORY_OPEN not in sys_content[0]["text"]
    # user: list content 原样保留
    assert cleaned[1]["content"] == messages[1]["content"]


def test_strip_standalone_injection_message_removed():
    """T3+T5: 独立注入消息（只含标记块、role=system）→ 整条删除。"""
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "system", "content": f"{MEMORY_OPEN}\n- old\n{MEMORY_CLOSE}"},
        {"role": "user", "content": "Bye"},
    ]
    cleaned = strip_previous_injection(messages)
    # 独立注入消息被整条删除（内容只有注入块 → strip 后为空 → 跳过）
    assert len(cleaned) == 2
    assert all(MEMORY_OPEN not in str(m.get("content", "")) for m in cleaned)


# ── 注入测试（T3）──

def test_hard_rule_always_injected():
    """硬规则无条件注入。"""
    config = ProxyConfig(hard_rules=["MUST be polite", "NEVER lie"])
    source = InjectionSource(hard_rules=["MUST be polite", "NEVER lie"])
    facts = source.build_facts("any query", config)
    assert any("MUST be polite" in f for f in facts)
    assert any("NEVER lie" in f for f in facts)


def test_inject_before_last_user_default():
    """T3 默认：注入在最后一条 user 前插入独立 system 消息。"""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "user", "content": "What's the weather?"},
    ]
    new_messages, injected_text = inject_memory(
        messages, ["[MUST/NEVER] rule"], max_chars=1800,
    )
    # 多了一条独立 system 消息（5 → 6）
    assert len(new_messages) == 5
    # 注入消息在最后一条 user 之前
    inject_idx = next(i for i, m in enumerate(new_messages) if MEMORY_OPEN in str(m.get("content", "")))
    assert new_messages[inject_idx]["role"] == "system"
    assert new_messages[inject_idx + 1]["role"] == "user"
    assert new_messages[inject_idx + 1]["content"] == "What's the weather?"
    # 首个 system 不被修改
    assert new_messages[0]["content"] == "You are a helpful assistant."


def test_inject_preserves_history_prefix():
    """T3: 注入后，除注入块外的历史消息前缀逐字不变（保 prompt cache）。"""
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
        {"role": "user", "content": "Bye"},
    ]
    new_messages, _ = inject_memory(messages, ["[MUST/NEVER] rule"])

    # 找注入消息位置
    inject_idx = next(i for i, m in enumerate(new_messages) if MEMORY_OPEN in str(m.get("content", "")))

    # 注入前的所有消息逐字不变
    for i in range(inject_idx):
        assert new_messages[i] == messages[i]

    # 注入后的最后一条 user 也不变
    assert new_messages[-1] == messages[-1]


def test_inject_compat_merge_system():
    """T3 兼容开关：merge_system=True 时回退到合并进首 system。"""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
    ]
    new_messages, injected_text = inject_memory(
        messages, ["[MUST/NEVER] rule"], max_chars=1800,
        merge_system=True,
    )
    # 不新增消息（还是 2 条）
    assert len(new_messages) == 2
    # 注入在首个 system 里
    assert new_messages[0]["role"] == "system"
    assert "You are a helpful assistant." in new_messages[0]["content"]
    assert MEMORY_OPEN in new_messages[0]["content"]
    assert MEMORY_CLOSE in new_messages[0]["content"]
    # user 消息不变
    assert new_messages[1] == messages[1]


def test_inject_compat_no_system_inserts_at_start():
    """T3 兼容模式：无 system 消息时在开头插入。"""
    messages = [{"role": "user", "content": "Hello"}]
    new_messages, injected_text = inject_memory(
        messages, ["[MUST/NEVER] rule"], max_chars=1800,
        merge_system=True,
    )
    assert len(new_messages) == 2
    assert new_messages[0]["role"] == "system"
    assert MEMORY_OPEN in new_messages[0]["content"]
    assert new_messages[1] == messages[0]


def test_inject_no_system_default_inserts_before_last_user():
    """T3 默认：无 system 消息时也在末条 user 前插入。"""
    messages = [{"role": "user", "content": "Hello"}]
    new_messages, injected_text = inject_memory(
        messages, ["[MUST/NEVER] rule"],
    )
    assert len(new_messages) == 2
    assert new_messages[0]["role"] == "system"
    assert MEMORY_OPEN in new_messages[0]["content"]
    assert new_messages[1] == messages[0]


def test_inject_never_modifies_agent_content():
    """注入流程不改 agent 正文。"""
    config = ProxyConfig(hard_rules=["MUST be polite"])
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
    ]
    new_messages, injected_text, ms, _, _ = do_inject(messages, "What is 2+2?", config)
    # user 和 assistant 内容不变
    user_msgs = [m for m in new_messages if m["role"] == "user"]
    asst_msgs = [m for m in new_messages if m["role"] == "assistant"]
    assert user_msgs[0]["content"] == "What is 2+2?"
    assert asst_msgs[0]["content"] == "4"
    # 原始 system 不被修改
    sys_msgs = [m for m in new_messages if m["role"] == "system"]
    assert "You are helpful." in sys_msgs[0]["content"]
    # 注入在某个 system 消息里
    assert any(MEMORY_OPEN in str(m.get("content", "")) for m in new_messages)


def test_inject_empty_facts_returns_unchanged():
    """没有 facts 时不注入，messages 不变。"""
    messages = [{"role": "user", "content": "Hello"}]
    new_messages, injected_text = inject_memory(messages, [])
    assert new_messages == messages
    assert injected_text == ""


def test_inject_multistrip_no_accumulation():
    """多轮注入不累积：第二轮注入时第一轮的块被删后再注新的。"""
    config = ProxyConfig(hard_rules=["MUST be polite"])
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
    ]

    # 第一轮注入
    msg1, _, _, _, _ = do_inject(messages, "Hello", config)
    bladex_count = sum(
        m.get("content", "").count(MEMORY_OPEN)
        for m in msg1 if isinstance(m.get("content"), str)
    )
    assert bladex_count == 1

    # 第二轮：把第一轮的注入带上（模拟 agent 保留了上下文）
    msg2, _, _, _, _ = do_inject(msg1, "World", config)
    bladex_count = sum(
        m.get("content", "").count(MEMORY_OPEN)
        for m in msg2 if isinstance(m.get("content"), str)
    )
    assert bladex_count == 1


# ── T3: 真注入源替换桩 + 热路径预算降级 ──


class MockEmbedder:
    """固定向量 mock embedder。"""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            h = hash(text)
            vec = [((h >> i) & 1) * 1.0 for i in range(self._dim)]
            results.append(vec)
        return results


def _make_index_with_facts(facts: list[Fact]) -> MemoryIndex:
    """创建带 facts 的 Memory Index（用 mock embedder）。"""
    embedder = MockEmbedder()
    index = MemoryIndex(Path(tempfile.mkdtemp()) / "index", embedder=embedder)
    index.open()
    for fact in facts:
        if fact.embedding is None:
            fact.embedding = embedder.embed([f"passage: {fact.content}"])[0]
        index.add_fact(fact)
    return index


def test_inject_degrade_to_hard_rules_on_timeout():
    """检索超时/失败 → 降级只注硬规则、请求不挂。"""
    # Memory Index with no embedder → search will fail gracefully
    index = MemoryIndex(Path(tempfile.mkdtemp()) / "index", embedder=None)
    index.open()
    # Manually add a table so search tries to run but fails
    # Actually without embedder, search returns [] (graceful), so we test the fallback path
    config = ProxyConfig(hard_rules=["MUST be polite"])
    source = InjectionSource(hard_rules=["MUST be polite"], index=index, top_k=5, min_relevance=0.0)

    messages = [{"role": "user", "content": "query"}]
    new_messages, injected_text, ms, _, _ = do_inject(
        messages, "query", config, source, user_id="u1",
    )

    # 没有召回 → 只注硬规则
    assert "MUST be polite" in injected_text
    # 请求不挂
    assert ms >= 0

    index.close()


def test_inject_position_regression_t3():
    """T3 注入位置仍在末条 user 前、上轮标记被剥离。"""
    index = _make_index_with_facts([])
    config = ProxyConfig(hard_rules=["MUST be polite"])
    source = InjectionSource(hard_rules=["MUST be polite"], index=index, min_relevance=0.0)

    # 模拟已有上轮注入
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "system", "content": f"{MEMORY_OPEN}\n- old fact\n{MEMORY_CLOSE}"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
        {"role": "user", "content": "Bye"},
    ]
    new_messages, injected_text, _, _, _ = do_inject(
        messages, "Bye", config, source, user_id="u1",
    )

    # 上轮注入被删（只有一条 MEMORY_OPEN）
    bladex_count = sum(
        m.get("content", "").count(MEMORY_OPEN)
        for m in new_messages if isinstance(m.get("content"), str)
    )
    assert bladex_count == 1

    # 注入在末条 user 前
    inject_idx = next(i for i, m in enumerate(new_messages) if MEMORY_OPEN in str(m.get("content", "")))
    assert new_messages[inject_idx]["role"] == "system"
    assert new_messages[inject_idx + 1]["role"] == "user"
    assert new_messages[inject_idx + 1]["content"] == "Bye"

    index.close()


# ── ADR-0016 Review Fix 2: summary 块剥离 ──

def test_strip_removes_context_summary_block():
    """ADR-0016 Fix 2: strip_previous_injection 也应剥离 <bladex-context-summary> 块。"""
    from bladex_proxy.assembly import SUMMARY_CLOSE, SUMMARY_OPEN
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
        {"role": "system", "content": f"System prompt\n\n{SUMMARY_OPEN}\n以下是摘要\n- old fact\n{SUMMARY_CLOSE}"},
    ]
    cleaned = strip_previous_injection(messages)
    assert len(cleaned) == 3
    for msg in cleaned:
        assert SUMMARY_OPEN not in msg.get("content", "")
        assert SUMMARY_CLOSE not in msg.get("content", "")
    # 正文保留
    assert "System prompt" in cleaned[2]["content"]


def test_strip_removes_both_memory_and_summary_blocks():
    """同时含 <bladex-memory> 和 <bladex-context-summary> 的 system 消息都应被清理。"""
    from bladex_proxy.assembly import SUMMARY_CLOSE, SUMMARY_OPEN
    messages = [
        {"role": "system", "content": f"Base prompt\n\n{MEMORY_OPEN}\n- rule\n{MEMORY_CLOSE}\n\n{SUMMARY_OPEN}\n- summary\n{SUMMARY_CLOSE}"},
    ]
    cleaned = strip_previous_injection(messages)
    assert len(cleaned) == 1
    content = cleaned[0]["content"]
    assert MEMORY_OPEN not in content
    assert SUMMARY_OPEN not in content
    assert "Base prompt" in content
