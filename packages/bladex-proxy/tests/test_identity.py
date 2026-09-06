"""身份解析单元测试（T3）。"""

from bladex_proxy.identity import resolve_identity
from bladex_proxy.models import ChatCompletionRequest


def test_identity_from_api_key_and_agent_id():
    """显式 API Key + X-Agent-ID。"""
    headers = {
        "authorization": "Bearer sk-test-12345",
        "x-agent-id": "codex",
        "user-agent": "python-httpx/0.27",
    }
    req = ChatCompletionRequest(
        messages=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "What's up?"},
        ]
    )
    identity, _ = resolve_identity(headers, req)
    assert identity.user_id != "anonymous"  # 有 API Key
    assert identity.agent_id == "codex"
    assert identity.turn_index == 2  # 用户轮 = 2 条 role=user（E0.2；修前 len//2 = 1）


def test_identity_no_api_key_anonymous():
    """无 API Key → anonymous。"""
    headers = {"user-agent": "python-requests/2.31"}
    req = ChatCompletionRequest(messages=[{"role": "user", "content": "Hi"}])
    identity, _ = resolve_identity(headers, req)
    assert identity.user_id == "anonymous"


def test_identity_ua_no_longer_names_agent():
    """G11.9：User-Agent 不再用于**命名** agent，只作分桶信号。

    取代原 `test_identity_guess_agent_from_ua`（该测试断言 `ua=codex/1.0` →
    `agent_id="codex"`）。UA 不能当身份权威来源——会随版本换 product token、
    用户可配、可缺失、一个 agent 可有多个 UA。详见 agent_bucket 模块 docstring。
    """
    from bladex_proxy.agent_registry import _agent_registry
    _agent_registry.clear()
    from bladex_proxy.identity import _session_cache
    _session_cache.clear()

    headers = {"user-agent": "codex/1.0"}
    req = ChatCompletionRequest(messages=[{"role": "user", "content": "Hi"}])
    identity, _ = resolve_identity(headers, req)
    # 不再直接叫 "codex"，而是各自成桶待认领
    assert identity.agent_id != "codex"
    assert identity.agent_id.startswith("unknown-")
    # 但 UA 仍进了分桶判据（信号没丢，只是不用来命名）
    assert identity.agent_bucket_basis == "ua:codex"


def test_identity_session_id_explicit():
    """显式 X-Session-ID 优先。"""
    headers = {
        "authorization": "Bearer sk-test",
        "x-session-id": "my-session-42",
    }
    req = ChatCompletionRequest(messages=[{"role": "user", "content": "Hi"}])
    identity, _ = resolve_identity(headers, req)
    assert identity.session_id == "my-session-42"


def test_identity_session_id_prefix_infer():
    """无显式 session id，按前缀推断（>=2 条消息）。"""
    headers = {"authorization": "Bearer sk-test"}
    req = ChatCompletionRequest(messages=[
        {"role": "user", "content": "Tell me about cats"},
        {"role": "assistant", "content": "Cats are great"},
        {"role": "user", "content": "What about dogs?"},
    ])
    identity, _ = resolve_identity(headers, req)
    assert identity.session_id.startswith("fp:")


def test_identity_storage_key_uses_entry_id():
    """T1: storage_key 格式 = user/agent/session/{entry_id}，不再用 turn_index。"""
    from bladex_proxy.models import Identity
    identity = Identity(
        user_id="abc12345",
        agent_id="codex",
        session_id="sess1",
        turn_index=3,
    )
    key = identity.storage_key("1719907200-0")
    assert key == "abc12345/codex/sess1/1719907200-0"
    # turn_index 不出现在 key 里
    assert "000003" not in key


def test_identity_session_prefix():
    """T1: session_prefix 返回会话前缀（不含序号），用于 scan_prefix。"""
    from bladex_proxy.models import Identity
    identity = Identity(
        user_id="abc12345",
        agent_id="codex",
        session_id="sess1",
        turn_index=3,
    )
    prefix = identity.session_prefix()
    assert prefix == "abc12345/codex/sess1/"


def test_identity_same_person_across_keys_in_personal_mode():
    """个人模式：不同 API Key → **同一个** user_id。

    🔴 本条 2026-08-16 反转过（ADR-0021 §2.3 修订）。原断言是
    「不同 API Key → 不同 user_id」，即把凭证当身份。那不是隔离，是**记忆分裂**：
    同一个人换 key / 换设备就变成另一个人，两半记忆在 P2 散点召回里互相看不见。
    live 实测踩中——4 个 user_id 实为一人，一件泰山啤酒尽调被切成两半。

    原断言想保护的"不同用户不该串号"仍然成立，只是**在个人模式下 API Key 不再是
    区分用户的依据**（个人自部署 = 一个人）。真正的多用户隔离由 identity.toml
    的 principal 承担，见 `test_adr0021_identity.py` 与
    `test_personal_identity_semantics.py::test_enterprise_*`。
    """
    from bladex_proxy.identity import LOCAL_USER_ID

    headers1 = {"authorization": "Bearer sk-user1"}
    headers2 = {"authorization": "Bearer sk-user2"}
    req = ChatCompletionRequest(messages=[{"role": "user", "content": "Hi"}])
    id1, _ = resolve_identity(headers1, req)
    id2, _ = resolve_identity(headers2, req)
    assert id1.user_id == id2.user_id == LOCAL_USER_ID


# ── T2 新增：会话指纹加固 ──

def test_session_stable_across_dynamic_system():
    """T2: 首条 system 含变化时间戳时，同一会话连续多轮 → 同一 session_id。"""
    from bladex_proxy.identity import _session_cache
    _session_cache.clear()

    headers = {"authorization": "Bearer sk-test"}

    # 第一轮：system 含时间戳 T1
    req1 = ChatCompletionRequest(messages=[
        {"role": "system", "content": "Current time: 2026-07-02T10:00:00Z"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ])
    id1, _ = resolve_identity(headers, req1)

    # 第二轮：system 含时间戳 T2（变了），但 user/assistant 前缀一致 + 尾部追加
    req2 = ChatCompletionRequest(messages=[
        {"role": "system", "content": "Current time: 2026-07-02T10:05:00Z"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "user", "content": "What's up?"},
    ])
    id2, _ = resolve_identity(headers, req2)

    assert id1.session_id == id2.session_id, (
        f"Same session should have same id: {id1.session_id} vs {id2.session_id}"
    )


def test_session_tail_superset_same_session():
    """T2: 尾部追加一轮 → 判为同会话。"""
    from bladex_proxy.identity import _session_cache
    _session_cache.clear()

    headers = {"authorization": "Bearer sk-test"}

    req1 = ChatCompletionRequest(messages=[
        {"role": "user", "content": "Tell me about cats"},
        {"role": "assistant", "content": "Cats are great"},
    ])
    id1, _ = resolve_identity(headers, req1)

    # 第二轮：前缀一致，尾部追加
    req2 = ChatCompletionRequest(messages=[
        {"role": "user", "content": "Tell me about cats"},
        {"role": "assistant", "content": "Cats are great"},
        {"role": "user", "content": "What about dogs?"},
    ])
    id2, _ = resolve_identity(headers, req2)

    assert id1.session_id == id2.session_id


def test_session_different_prefix_new_session():
    """T2: 换了话题的新前缀 → 新会话。"""
    from bladex_proxy.identity import _session_cache
    _session_cache.clear()

    headers = {"authorization": "Bearer sk-test"}

    req1 = ChatCompletionRequest(messages=[
        {"role": "user", "content": "Tell me about cats"},
        {"role": "assistant", "content": "Cats are great"},
    ])
    id1, _ = resolve_identity(headers, req1)

    req2 = ChatCompletionRequest(messages=[
        {"role": "user", "content": "How to cook pasta"},
        {"role": "assistant", "content": "Boil water first"},
    ])
    id2, _ = resolve_identity(headers, req2)

    assert id1.session_id != id2.session_id


# ── 被动 agent 指纹识别 ──

def test_agent_fingerprint_hermes_system_prompt():
    """被动指纹：Hermes system prompt 含 'Hermes Agent' → 识别为 hermes:profile。"""
    from bladex_proxy.agent_registry import _agent_registry
    _agent_registry.clear()
    from bladex_proxy.identity import _session_cache
    _session_cache.clear()

    headers = {"authorization": "Bearer sk-test"}
    req = ChatCompletionRequest(messages=[
        {"role": "system", "content": "You are Hermes Agent, an intelligent AI assistant created by Nous Research. You are helpful."},
        {"role": "user", "content": "Hello"},
    ])
    identity, source = resolve_identity(headers, req)
    # 无 profile 标记 → base "hermes"
    assert identity.agent_id == "hermes"
    assert source.value == "fingerprint"


def test_agent_fingerprint_hermes_tool_signature():
    """被动指纹：无特征 system prompt，但 tool 签名含 skill_manage → 识别为 hermes。

    覆盖 Hermes 内部子请求（标题生成等）system prompt 变了的场景。
    system prompt 非 Hermes 特征 → profile_aware 无 system prompt 可 hash → 返回 base "hermes"。
    """
    from bladex_proxy.agent_registry import _agent_registry
    _agent_registry.clear()
    from bladex_proxy.identity import _session_cache
    _session_cache.clear()

    headers = {"authorization": "Bearer sk-test"}
    req = ChatCompletionRequest(
        messages=[
            {"role": "system", "content": "Generate a short title."},
            {"role": "user", "content": "Summarize"},
        ],
        tools=[
            {"type": "function", "function": {"name": "skill_manage", "parameters": {}}},
            {"type": "function", "function": {"name": "web_search", "parameters": {}}},
        ],
    )
    identity, _ = resolve_identity(headers, req)
    # tool 签名匹配，无 profile 标记 → base "hermes"
    assert identity.agent_id == "hermes"


def test_agent_fingerprint_hermes_from_tool_calls_history():
    """被动指纹：tools 字段没有，但 messages 历史里有 Hermes 工具调用记录。

    工具从 `session_search` 换成 `skill_manage`（G11.10/MQ-A6）：前者
    deepseek-harness 也有，留在 hermes 签名里会把 dsh 误判成 hermes。本用例验的是
    "从 tool_calls 历史里提取工具名"这条路径，换个 Hermes 专名不影响它的射程。
    """
    from bladex_proxy.agent_registry import _agent_registry
    _agent_registry.clear()
    from bladex_proxy.identity import _session_cache
    _session_cache.clear()

    headers = {"authorization": "Bearer sk-test"}
    req = ChatCompletionRequest(messages=[
        {"role": "system", "content": "Some generic prompt."},
        {"role": "user", "content": "Search the web"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "skill_manage", "arguments": "{}"}},
        ]},
    ])
    identity, _ = resolve_identity(headers, req)
    assert identity.agent_id == "hermes"


def test_agent_fingerprint_claude_code_system_prompt():
    """被动指纹：Claude Code system prompt → 识别为 claude-code。"""
    headers = {"authorization": "Bearer sk-test"}
    req = ChatCompletionRequest(messages=[
        {"role": "system", "content": "You are Claude Code, Anthropic's official CLI for Claude."},
        {"role": "user", "content": "Write a function"},
    ])
    identity, _ = resolve_identity(headers, req)
    assert identity.agent_id == "claude-code"


def test_agent_fingerprint_claude_code_tool_signature():
    """被动指纹：Claude Code 工具签名（PascalCase Read/Write/Edit/Bash）。"""
    headers = {"authorization": "Bearer sk-test"}
    req = ChatCompletionRequest(
        messages=[{"role": "user", "content": "Fix the bug"}],
        tools=[
            {"type": "function", "function": {"name": "Read", "parameters": {}}},
            {"type": "function", "function": {"name": "Write", "parameters": {}}},
            {"type": "function", "function": {"name": "Edit", "parameters": {}}},
        ],
    )
    identity, _ = resolve_identity(headers, req)
    assert identity.agent_id == "claude-code"


def test_agent_fingerprint_explicit_header_overrides():
    """显式 X-Agent-ID 优先于被动指纹（不走 profile hash）。"""
    from bladex_proxy.agent_registry import _agent_registry
    _agent_registry.clear()
    from bladex_proxy.identity import _session_cache
    _session_cache.clear()

    headers = {
        "authorization": "Bearer sk-test",
        "x-agent-id": "my-custom-agent",
    }
    req = ChatCompletionRequest(messages=[
        {"role": "system", "content": "You are Hermes Agent, created by Nous Research."},
        {"role": "user", "content": "Hello"},
    ])
    identity, source = resolve_identity(headers, req)
    assert identity.agent_id == "my-custom-agent"
    assert source.value == "explicit"


def test_agent_fingerprint_unknown_when_no_signal():
    """无任何识别信号 → `unknown-<hash8>`（G11.9 起不再是裸 "unknown"）。

    改动理由：所有未识别 agent 共用一个 `unknown` = 把互不相干的客户端合进同一个
    记忆命名空间，是换了名字的误合并。分桶后宁分勿合，用户认领即收敛。
    """
    from bladex_proxy.agent_registry import _agent_registry
    _agent_registry.clear()
    from bladex_proxy.identity import _session_cache
    _session_cache.clear()

    headers = {"authorization": "Bearer sk-test"}
    req = ChatCompletionRequest(messages=[
        {"role": "user", "content": "Hello"},
    ])
    identity, _ = resolve_identity(headers, req)
    assert identity.agent_id.startswith("unknown-")
    assert identity.agent_bucket_basis == "none"  # 三种信号全无


def test_user_agent_is_a_bucketing_signal_not_a_name():
    """G11.9：User-Agent 参与分桶，但不再直接当 agent 名。

    取代原 `test_agent_fingerprint_user_agent_still_works`（断言 ua=codex/1.0 →
    agent_id="codex"）。UA 不可作身份权威来源的四条实证见 agent_bucket 模块 docstring。
    """
    from bladex_proxy.agent_registry import _agent_registry
    _agent_registry.clear()
    from bladex_proxy.identity import _session_cache
    _session_cache.clear()

    headers = {
        "authorization": "Bearer sk-test",
        "user-agent": "codex/1.0",
    }
    req = ChatCompletionRequest(messages=[
        {"role": "user", "content": "Hello"},
    ])
    identity, _ = resolve_identity(headers, req)
    assert identity.agent_id.startswith("unknown-")
    assert identity.agent_bucket_basis == "ua:codex"   # 信号进了分桶
    assert identity.agent_id != "codex"                 # 但没有拿来命名


# ── profile 级区分（从 system prompt 自动提取，无需用户配置）──

def test_hermes_profile_extracted_from_system_prompt():
    """Hermes system prompt 含 'Active Hermes profile: accept' → agent_id = 'hermes:accept'。

    profile 名直接从 system prompt 提取，不需要用户配 header，不受动态内容影响。
    """
    from bladex_proxy.agent_registry import _agent_registry
    from bladex_proxy.identity import _session_cache

    _agent_registry.clear()
    _session_cache.clear()

    headers = {"authorization": "Bearer sk-test"}
    req = ChatCompletionRequest(messages=[
        {"role": "system", "content": "You are Hermes Agent, created by Nous Research.\nActive Hermes profile: accept. Other profiles live under ~/.hermes/profiles/"},
        {"role": "user", "content": "Hello"},
    ])
    identity, source = resolve_identity(headers, req)
    assert identity.agent_id == "hermes:accept"
    assert source.value == "fingerprint"


def test_hermes_different_profiles_different_agent_id():
    """同一 Hermes，不同 profile 名 → 不同 agent_id。"""
    from bladex_proxy.agent_registry import _agent_registry
    from bladex_proxy.identity import _session_cache

    _agent_registry.clear()
    _session_cache.clear()

    headers = {"authorization": "Bearer sk-test"}

    req_a = ChatCompletionRequest(messages=[
        {"role": "system", "content": "You are Hermes Agent.\nActive Hermes profile: accept."},
        {"role": "user", "content": "Hello"},
    ])
    id_a, _ = resolve_identity(headers, req_a)

    _session_cache.clear()

    req_b = ChatCompletionRequest(messages=[
        {"role": "system", "content": "You are Hermes Agent.\nActive Hermes profile: default."},
        {"role": "user", "content": "Hello"},
    ])
    id_b, _ = resolve_identity(headers, req_b)

    assert id_a.agent_id == "hermes:accept"
    assert id_b.agent_id == "hermes:default"
    assert id_a.agent_id != id_b.agent_id


def test_hermes_same_profile_stable_across_memory_changes():
    """同一 profile，MEMORY 内容不同 → agent_id 不变（profile 从标记提取，不受 MEMORY 影响）。"""
    from bladex_proxy.agent_registry import _agent_registry
    from bladex_proxy.identity import _session_cache

    _agent_registry.clear()
    _session_cache.clear()

    headers = {"authorization": "Bearer sk-test"}

    # 第一轮：MEMORY 有内容 A
    req1 = ChatCompletionRequest(messages=[
        {"role": "system", "content": "You are Hermes Agent.\nActive Hermes profile: accept.\n═══\nMEMORY [98%]\n═══\n记忆内容 A\n═══"},
        {"role": "user", "content": "Hello"},
    ])
    id1, _ = resolve_identity(headers, req1)

    _session_cache.clear()

    # 第二轮：MEMORY 内容变了
    req2 = ChatCompletionRequest(messages=[
        {"role": "system", "content": "You are Hermes Agent.\nActive Hermes profile: accept.\n═══\nMEMORY [89%]\n═══\n记忆内容 B\n记忆内容 C\n═══"},
        {"role": "user", "content": "Hello"},
    ])
    id2, _ = resolve_identity(headers, req2)

    assert id1.agent_id == id2.agent_id == "hermes:accept"


def test_hermes_same_profile_stable_across_injection_changes():
    """同一 profile，注入内容不同 → agent_id 不变。"""
    from bladex_proxy.agent_registry import _agent_registry
    from bladex_proxy.identity import _session_cache

    _agent_registry.clear()
    _session_cache.clear()

    headers = {"authorization": "Bearer sk-test"}
    base = "You are Hermes Agent.\nActive Hermes profile: accept."

    req1 = ChatCompletionRequest(messages=[
        {"role": "system", "content": base + "\n\n<bladex-memory>\n- rule A\n</bladex-memory>"},
        {"role": "user", "content": "Hello"},
    ])
    id1, _ = resolve_identity(headers, req1)

    _session_cache.clear()

    req2 = ChatCompletionRequest(messages=[
        {"role": "system", "content": base + "\n\n<bladex-memory>\n- rule B\n</bladex-memory>"},
        {"role": "user", "content": "Hi"},
    ])
    id2, _ = resolve_identity(headers, req2)

    assert id1.agent_id == id2.agent_id == "hermes:accept"


def test_hermes_no_profile_marker_returns_base():
    """system prompt 无 profile 标记 → 返回 base 'hermes'（子请求通过粘性继承）。"""
    from bladex_proxy.agent_registry import _agent_registry
    from bladex_proxy.identity import _session_cache

    _agent_registry.clear()
    _session_cache.clear()

    headers = {"authorization": "Bearer sk-test"}
    req = ChatCompletionRequest(messages=[
        {"role": "system", "content": "Generate a short title."},
        {"role": "user", "content": "Hello"},
    ], tools=[
        {"type": "function", "function": {"name": "skill_manage", "parameters": {}}},
    ])
    identity, _ = resolve_identity(headers, req)
    # tool 签名匹配 hermes，但无 profile 标记 → base "hermes"
    assert identity.agent_id == "hermes"


def test_agent_sticky_inherits_profile_agent_id():
    """会话粘性：主请求识别为 hermes:accept，子请求继承。"""
    from bladex_proxy.agent_registry import _agent_registry
    from bladex_proxy.identity import _session_cache

    _agent_registry.clear()
    _session_cache.clear()

    headers = {"authorization": "Bearer sk-test"}

    # 主请求：有 profile 标记
    req_main = ChatCompletionRequest(messages=[
        {"role": "system", "content": "You are Hermes Agent.\nActive Hermes profile: accept."},
        {"role": "user", "content": "Tell me a joke"},
    ])
    id_main, source_main = resolve_identity(headers, req_main)
    assert source_main.value == "fingerprint"
    assert id_main.agent_id == "hermes:accept"

    # 子请求：无 Hermes 特征 system prompt、无 tools 定义 → 指纹落空 → 会话粘性
    req_sub = ChatCompletionRequest(messages=[
        {"role": "system", "content": "Generate a short title."},
        {"role": "user", "content": "Tell me a joke"},
    ])
    id_sub, source_sub = resolve_identity(headers, req_sub)
    assert source_sub.value == "sticky"
    assert id_sub.agent_id == "hermes:accept"


# ── T9：auxiliary 指纹（上下文压缩 / 任务列表压缩）──


def test_aux_fingerprint_checkpoint():
    """Hermes 上下文压缩 system prompt → auxiliary=True。"""
    from bladex_proxy.identity import resolve_identity
    from bladex_proxy.models import ChatCompletionRequest
    req = ChatCompletionRequest(model="x", messages=[
        {"role": "system", "content": "You are a summarization agent creating a context checkpoint for the conversation."},
        {"role": "user", "content": "summarize the above"},
    ])
    ident, _ = resolve_identity({"user-agent": "test"}, req)
    assert ident.auxiliary is True
    assert ident.agent_id == "hermes"


def test_aux_fingerprint_task_list_compression():
    """Hermes 任务列表压缩 → auxiliary=True。"""
    from bladex_proxy.identity import resolve_identity
    from bladex_proxy.models import ChatCompletionRequest
    req = ChatCompletionRequest(model="x", messages=[
        {"role": "system", "content": "[Your active task list was preserved across context compression]"},
        {"role": "user", "content": "continue"},
    ])
    ident, _ = resolve_identity({"user-agent": "test"}, req)
    assert ident.auxiliary is True


def test_lookup_sticky_no_time_window():
    """ADR-0021: lookup_sticky 移除时间窗口，子任务继承最近 agent（不论多久前 register）。

    回归：原 120s 窗口假设"子任务秒级紧邻"不成立（Hermes 图片描述子任务可能
    在主请求后十几分钟才发），超窗口 -> unknown。移除窗口后超时也命中。
    """
    import time

    from bladex_proxy.agent_registry import AgentRecord, AgentRegistry
    reg = AgentRegistry()
    reg._records["u1"] = [AgentRecord(
        agent_id="hermes:accept", identified_at=time.time() - 3600,
    )]
    assert reg.lookup_sticky("u1") == "hermes:accept"  # 超原 120s 窗口仍命中
    assert reg.lookup_sticky("u2") is None  # 没识别过 -> None
