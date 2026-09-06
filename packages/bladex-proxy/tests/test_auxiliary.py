"""T6: auxiliary 调用识别（Hermes MoA 兼容）单元测试。

测试 MoA reference 指纹、auxiliary 注入策略、auxiliary Turn 标记、
consolidation 跳过 auxiliary 轮。
"""

import tempfile
from pathlib import Path

from bladex_proxy.config import ProxyConfig
from bladex_proxy.identity import resolve_identity
from bladex_proxy.inject import MEMORY_OPEN, do_inject
from bladex_proxy.models import ChatCompletionRequest, Identity, Turn, TurnStatus
from bladex_proxy.storage.memory_index import MemoryIndex
from bladex_proxy.storage.memory_hub import MemoryHub

# ── MoA reference 指纹 ──

def test_moa_reference_fingerprint():
    """模拟 MoA reference 请求（advisor system prompt、无 tools）→ 识别为 hermes + auxiliary。"""
    moa_system = (
        "You are a reference advisor in a Mixture of Agents (MoA) process. "
        "You are NOT the acting agent and you do NOT execute anything."
    )
    req = ChatCompletionRequest(
        messages=[
            {"role": "system", "content": moa_system},
            {"role": "user", "content": "What should I do next?"},
        ],
        tools=None,
    )
    headers = {"user-agent": "python-httpx/0.27"}
    identity, source = resolve_identity(headers, req)

    assert identity.agent_id == "hermes"
    assert identity.auxiliary is True, "MoA reference should be marked auxiliary"


def test_moa_reference_no_tools_no_auxiliary_mismatch():
    """正常 Hermes 请求（有 tool schema）不被误判为 auxiliary。"""
    req = ChatCompletionRequest(
        messages=[
            {"role": "system", "content": "You are a Hermes agent by Nous Research."},
            {"role": "user", "content": "Search for something"},
        ],
        tools=[
            {"type": "function", "function": {"name": "session_search"}},
        ],
    )
    headers = {"user-agent": "python-httpx/0.27"}
    identity, source = resolve_identity(headers, req)

    assert identity.agent_id.startswith("hermes")
    assert identity.auxiliary is False, "Normal Hermes request should not be auxiliary"


def test_auxiliary_inherits_session_stickiness():
    """auxiliary 调用继承主会话粘性（同一 user 的后续请求能查到 agent_id）。"""
    # 先注册一个正常 hermes 请求
    normal_req = ChatCompletionRequest(
        messages=[
            {"role": "system", "content": "You are a Hermes agent by Nous Research."},
            {"role": "user", "content": "Hello"},
        ],
        tools=[{"type": "function", "function": {"name": "skill_manage"}}],
    )
    headers = {"authorization": "Bearer test-key-123"}
    identity1, _ = resolve_identity(headers, normal_req)
    assert identity1.agent_id.startswith("hermes")
    assert identity1.auxiliary is False

    # 然后 MoA reference 请求（无 tools、不同 system prompt）
    moa_system = (
        "You are a reference advisor in a Mixture of Agents (MoA) process. "
        "You are NOT the acting agent."
    )
    ref_req = ChatCompletionRequest(
        messages=[
            {"role": "system", "content": moa_system},
            {"role": "user", "content": "Advise on next steps."},
        ],
        tools=None,
    )
    identity2, _ = resolve_identity(headers, ref_req)
    assert identity2.agent_id == "hermes"
    assert identity2.auxiliary is True


# ── auxiliary 注入策略 ──

def test_auxiliary_injection_policy():
    """auxiliary 轮次只注硬规则、不注记忆正文。"""
    config = ProxyConfig()
    config.hard_rules = ["NEVER use emojis.", "MUST be concise."]

    messages = [
        {"role": "system", "content": "You are a reference advisor in a Mixture of Agents."},
        {"role": "user", "content": "Advise me."},
    ]

    # auxiliary=True → 只注硬规则
    inj_msgs_aux, inj_text_aux, _, _, _ = do_inject(
        messages, "Advise me", config, auxiliary=True,
    )
    assert MEMORY_OPEN in inj_text_aux
    assert "NEVER use emojis" in inj_text_aux
    # 硬规则数 = 配置的 hard_rules 数（不召记忆事实）
    assert inj_text_aux.count("[MUST/NEVER]") == 2

    # auxiliary=False → 正常注入（桩只回硬规则，但路径不同）
    inj_msgs_normal, inj_text_normal, _, _, _ = do_inject(
        messages, "Advise me", config, auxiliary=False,
    )
    assert MEMORY_OPEN in inj_text_normal


# ── auxiliary Turn 标记 ──

def test_auxiliary_turn_marked():
    """auxiliary 轮次入库带 auxiliary=True 标记。"""
    identity = Identity(
        user_id="u1", agent_id="hermes", session_id="s1",
        auxiliary=True,
    )
    turn = Turn(
        identity=identity, model="test",
        request_messages=[{"role": "user", "content": "hi"}],
        response_text="ok",
        status=TurnStatus.OK,
        auxiliary=identity.auxiliary,
    )
    assert turn.auxiliary is True

    # 正常 Turn
    identity_normal = Identity(
        user_id="u1", agent_id="hermes", session_id="s1",
        auxiliary=False,
    )
    turn_normal = Turn(
        identity=identity_normal, model="test",
        request_messages=[{"role": "user", "content": "hi"}],
        response_text="ok",
        status=TurnStatus.OK,
        auxiliary=identity_normal.auxiliary,
    )
    assert turn_normal.auxiliary is False


# ── consolidation 跳过 auxiliary ──

def test_consolidation_skips_auxiliary():
    """Memory Index rebuild 从 Memory Hub 增量重建时跳过 auxiliary 轮次。"""
    class MockEmbedder:
        def embed(self, texts):
            return [[0.0] * 64 for _ in texts]

    with tempfile.TemporaryDirectory() as tmpdir:
        ledger = MemoryHub(Path(tmpdir) / "ledger")
        ledger.open()
        index = MemoryIndex(Path(tmpdir) / "index", embedder=MockEmbedder())
        index.open()

        # 写一条正常 Turn
        identity_normal = Identity(
            user_id="u1", agent_id="hermes", session_id="s1",
        )
        turn_normal = Turn(
            identity=identity_normal, model="m",
            request_messages=[
                {"role": "system", "content": "You are a Hermes agent."},
                {"role": "user", "content": "I am a developer building a memory system called BladeX."},
            ],
            response_text="OK",
            status=TurnStatus.OK,
            auxiliary=False,
        )
        ledger.put("u1/hermes/s1/100-0", turn_normal)

        # 写一条 auxiliary Turn（MoA reference）
        identity_aux = Identity(
            user_id="u1", agent_id="hermes", session_id="s1",
            auxiliary=True,
        )
        turn_aux = Turn(
            identity=identity_aux, model="m",
            request_messages=[
                {"role": "system", "content": "You are a reference advisor in a Mixture of Agents."},
                {"role": "user", "content": "I am a developer building a memory system called BladeX."},
            ],
            response_text="Advise...",
            status=TurnStatus.OK,
            auxiliary=True,
        )
        ledger.put("u1/hermes/s1/101-0", turn_aux)

        # 从 Memory Hub 重建 Memory Index
        index.rebuild_from_hub(ledger)

        # 只从正常轮提炼，auxiliary 轮被跳过
        # 正常轮的 user 消息会被提炼（>20 chars），auxiliary 轮的相同消息不会重复提炼
        facts = index.all_facts()
        assert len(facts) >= 1
        # auxiliary 轮的 ledger_key 不应出现在 fact 来源中
        aux_keys = [f for f in facts if f.source_ledger_key == "u1/hermes/s1/101-0"]
        assert len(aux_keys) == 0, "auxiliary turns should be skipped in consolidation"

        # auxiliary 轮的 key 应被标记已消费（不重复处理）
        assert index.is_consumed("u1/hermes/s1/101-0")

        index.close()
        ledger.close()


# ── T2(ADR-0018 R2): classify_auxiliary + rebuild 重判 ──

from bladex_proxy.identity import classify_auxiliary  # noqa: E402


def test_classify_auxiliary_user_message_templates():
    """每条 user 消息模板前缀命中即 auxiliary（覆盖真实 Memory Hub 漏网样本）。"""
    cases = [
        ("[ASYNC DELEGATION BATCH COMPLETE - deleg_abc123] result here", "async_delegation_batch"),
        ("Please process this web content and create a comprehensive markdown summary:", "web_content_summary"),
        ("A background fan-out of 1 subagent completed.", "background_fanout"),
        ("[CONTEXT COMPACTION - REFERENCE ONLY] Earlier turns were compacted.", "context_compaction_handoff"),
        ("[Your active task list was preserved across context compression]\n- [ ] do x", "task_list_preserved"),
        ("You are a summarization agent creating a context checkpoint. Treat...", "summarization_checkpoint"),
    ]
    for user_text, expected_rule in cases:
        messages = [
            {"role": "system", "content": "You are a Hermes agent by Nous Research."},
            {"role": "user", "content": user_text},
        ]
        is_aux, rule = classify_auxiliary(messages)
        assert is_aux is True, f"should be aux: {user_text[:40]!r}"
        assert rule == expected_rule, f"rule {rule} != {expected_rule}"


def test_classify_auxiliary_system_prompt_moa():
    """system prompt 命中 MoA reference 关键词 -> auxiliary（保留原 T6 路径）。"""
    messages = [
        {"role": "system", "content": "You are a reference advisor in a Mixture of Agents (MoA) process."},
        {"role": "user", "content": "What should I do next?"},
    ]
    is_aux, rule = classify_auxiliary(messages)
    assert is_aux is True
    assert rule.startswith("system:")


def test_classify_auxiliary_normal_user_message_no_false_positive():
    """正常用户消息 0 误伤（含世界杯、开发等真实话题）。"""
    normals = [
        "今天世界杯赛程怎么样？",
        "I am a developer building a memory system called BladeX.",
        "继续上次的破产案件分析",
        "帮我看一下这个代码有没有 bug",
        "search",  # 短消息
        "",
    ]
    for text in normals:
        messages = [
            {"role": "system", "content": "You are a Hermes agent by Nous Research."},
            {"role": "user", "content": text},
        ]
        is_aux, rule = classify_auxiliary(messages)
        assert is_aux is False, f"normal message falsely flagged aux: {text!r} (rule={rule})"


def test_classify_auxiliary_multimodal_user_content():
    """多模态 user 消息（content 为 list）也能命中模板。"""
    messages = [
        {"role": "system", "content": "You are a Hermes agent."},
        {"role": "user", "content": [
            {"type": "text", "text": "Please process this web content and summarize"},
            {"type": "image_url", "image_url": {"url": "data:..."}},
        ]},
    ]
    is_aux, rule = classify_auxiliary(messages)
    assert is_aux is True
    assert rule == "web_content_summary"


def test_classify_auxiliary_only_last_user_message_n2():
    """N2 修复：模板只看末条 user 消息，历史中的委派回执不误伤本轮真实消息。

    真实漏网样本：Hermes 委派回执留在历史里，之后用户发"帮我深入研究 RouteLLM"--
    原扫全历史会把这轮误判 aux -> 真实消息漏蒸馏。修复后末条是真实消息 -> aux=False。
    """
    messages = [
        {"role": "system", "content": "You are a Hermes agent by Nous Research."},
        {"role": "user", "content": "[ASYNC DELEGATION BATCH COMPLETE - deleg_x] subagent done"},
        {"role": "assistant", "content": "已收到委派回执。"},
        {"role": "user", "content": "帮我深入研究一下 RouteLLM/semantic-router 这些路由方案"},  # 末条 = 真实消息
    ]
    is_aux, rule = classify_auxiliary(messages)
    assert is_aux is False, "末条是真实消息时，历史中的回执不应误判本轮为 aux"
    assert rule == ""


def test_classify_auxiliary_last_user_is_receipt_still_aux():
    """对照：末条 user 本身就是回执 -> 仍判 aux（本轮是回执轮）。"""
    messages = [
        {"role": "system", "content": "You are a Hermes agent by Nous Research."},
        {"role": "user", "content": "帮我研究 RouteLLM"},  # 历史真实消息
        {"role": "assistant", "content": "好的。"},
        {"role": "user", "content": "[ASYNC DELEGATION BATCH COMPLETE - deleg_y] result"},  # 末条 = 回执
    ]
    is_aux, rule = classify_auxiliary(messages)
    assert is_aux is True
    assert rule == "async_delegation_batch"


def test_rebuild_rejudges_aux_does_not_trust_frozen_flag():
    """T2 核心：rebuild 用 classify_auxiliary 重判，不信冻结的 turn.auxiliary。

    构造一条 auxiliary=False（初判漏了）但 user 消息命中模板的 Turn，
    rebuild 应跳过它（重判为 aux）。
    """
    class MockEmbedder:
        def embed(self, texts):
            return [[0.0] * 64 for _ in texts]

    with tempfile.TemporaryDirectory() as tmpdir:
        ledger = MemoryHub(Path(tmpdir) / "ledger")
        ledger.open()
        index = MemoryIndex(Path(tmpdir) / "index", embedder=MockEmbedder())
        index.open()

        # auxiliary=False（初判漏）但 user 消息是异步委派回执 -> 重判应跳过
        identity = Identity(user_id="u1", agent_id="hermes", session_id="s1", auxiliary=False)
        turn = Turn(
            identity=identity, model="m",
            request_messages=[
                {"role": "system", "content": "You are a Hermes agent by Nous Research."},
                {"role": "user", "content": "[ASYNC DELEGATION BATCH COMPLETE - deleg_x] subagent computed 1+2+3=6"},
            ],
            response_text="ok",
            status=TurnStatus.OK,
            auxiliary=False,  # 冻结标记说不是 aux
        )
        ledger.put("u1/hermes/s1/200-0", turn)

        index.rebuild_from_hub(ledger)

        facts = index.all_facts()
        # 重判为 aux -> 不提炼
        assert len(facts) == 0, "frozen auxiliary=False must be overridden by re-judgment"
        assert index.is_consumed("u1/hermes/s1/200-0")

        index.close()
        ledger.close()


def test_rebuild_processes_normal_turn_with_aux_source_empty():
    """正常 Turn（aux_source 空）不被重判为 aux，正常提炼。"""
    class MockEmbedder:
        def embed(self, texts):
            return [[0.0] * 64 for _ in texts]

    with tempfile.TemporaryDirectory() as tmpdir:
        ledger = MemoryHub(Path(tmpdir) / "ledger")
        ledger.open()
        index = MemoryIndex(Path(tmpdir) / "index", embedder=MockEmbedder())
        index.open()

        identity = Identity(user_id="u1", agent_id="hermes", session_id="s1")
        turn = Turn(
            identity=identity, model="m",
            request_messages=[
                {"role": "system", "content": "You are a Hermes agent by Nous Research."},
                {"role": "user", "content": "用户最喜欢的本地嵌入模型是 multilingual-e5-large"},
            ],
            response_text="ok",
            status=TurnStatus.OK,
        )
        ledger.put("u1/hermes/s1/300-0", turn)

        index.rebuild_from_hub(ledger)
        facts = index.all_facts()
        assert len(facts) >= 1

        index.close()
        ledger.close()
