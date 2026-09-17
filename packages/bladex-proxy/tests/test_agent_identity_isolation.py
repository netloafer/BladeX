"""G11.5 + G11.9 集成验收 —— 跨 agent 不得互相署名。

这组剧本直接回放 2026-08-18 的事故：dsh 接入后 proxy 日志里 agent_id 只有
`hermes:default` 一个值、`unknown` 出现 0 次，dsh 的轮次被静默写进了 hermes 的
记忆命名空间。第一条测试就是那个事故的最小复现。

红线：**宁可 unknown，不可误署名**。误署名是不可逆的记忆污染（要重建才能洗），
未识别只是少一点归因。与「误合并=0」同构。
"""

from __future__ import annotations

import pytest
from bladex_proxy.agent_registry import _agent_registry
from bladex_proxy.identity import _session_cache, resolve_identity
from bladex_proxy.models import AgentSource, ChatCompletionRequest

HERMES_SYSTEM = (
    "You are a Hermes agent built by Nous Research.\nActive Hermes profile: default"
)

DSH_UA = "deepseek-harness/0.3.1 (+https://github.com/deepseek-ai/deepseek-harness)"


@pytest.fixture(autouse=True)
def _clean_state():
    """进程级单例必须逐用例清空，否则粘性会跨用例串味。"""
    _agent_registry.clear()
    _session_cache.clear()
    yield
    _agent_registry.clear()
    _session_cache.clear()


def _hermes_request(text: str = "帮我看下这个") -> tuple[dict, ChatCompletionRequest]:
    headers = {
        "authorization": "Bearer sk-shared-key",
        "user-agent": "OpenAI/Python 1.99.1",  # Hermes chat 主路径就是 SDK 默认 UA
    }
    req = ChatCompletionRequest(
        messages=[
            {"role": "system", "content": HERMES_SYSTEM},
            {"role": "user", "content": text},
        ],
        tools=[{"function": {"name": "session_search"}}],
    )
    return headers, req


def _dsh_request(text: str = "这个目录里有多少文件？") -> tuple[dict, ChatCompletionRequest]:
    """dsh 实发形态：自带 UA + 厂商前缀 header，共用同一把 client key。"""
    headers = {
        "authorization": "Bearer sk-shared-key",   # 事故现场即 label=hermes-default
        "user-agent": DSH_UA,
        "x-deepseek-harness-user-id": "anon-7f3a",
        "x-deepseek-harness-session-id": "sess-001",
    }
    req = ChatCompletionRequest(
        messages=[{"role": "user", "content": text}],
        tools=[{"function": {"name": "bash"}}, {"function": {"name": "read"}}],
    )
    return headers, req


# ── 事故复现 ────────────────────────────────────────────────────────────────


# ── 🔴 Pi 真实流量逼出来的回归（2026-08-19）─────────────────────────────────

#: Pi 实发 header，逐字抄自 live 日志（`agent_unrecognized` 那条的快照）。
#: 关键点：UA 是 **OpenAI JS SDK 的名字**，不是 Pi 自己的；厂商前缀全是
#: `x-stainless-*`（SDK 代码生成器，属基础设施）——两类传输信号都指向 SDK。
PI_HEADERS = {
    "authorization": "Bearer sk-shared-key",
    "user-agent": "OpenAI/JS 6.40.0",
    "x-stainless-lang": "js",
    "x-stainless-runtime": "node",
    "x-stainless-runtime-version": "v22.23.1",
    "x-stainless-package-version": "6.40.0",
    "x-stainless-os": "MacOS",
    "x-stainless-arch": "arm64",
}


def test_pi_after_hermes_is_not_signed_as_hermes() -> None:
    """🔴 两个都用 OpenAI SDK 的 agent，不得互相继承身份。

    **这条是 Pi 真实接入后复现出来的缺陷，不是假想**：
    `parse_ua_product` 只取第一个 product token，于是 `OpenAI/JS 6.40.0`（Pi）与
    `OpenAI/Python 1.99.1`（Hermes 主路径）双双塌缩成 `openai` → 传输身份相同
    （实测 origin 都是 `140bedbf`）→ Hermes 先说话，Pi 就被同源继承署名成
    `hermes:default`。区分信息本来就在 UA 里，是取 product token 时丢的。

    首次 live 验收之所以通过，只是因为那个窗口里 Hermes 恰好没说话——
    **顺序不同结论就不同的验收，等于没验收**。
    """
    h_headers, h_req = _hermes_request()
    hermes_identity, _ = resolve_identity(h_headers, h_req)
    assert hermes_identity.agent_id == "hermes:default"

    pi_req = ChatCompletionRequest(
        messages=[{"role": "user", "content": "这个目录里有多少文件？"}],
        tools=[{"function": {"name": "bash"}}, {"function": {"name": "read"}}],
    )
    pi_identity, pi_source = resolve_identity(dict(PI_HEADERS), pi_req)

    assert not pi_identity.agent_id.startswith("hermes"), (
        f"Pi 被署名成 {pi_identity.agent_id} —— 跨 agent 记忆污染回归"
    )
    assert pi_identity.agent_id.startswith("unknown-")
    assert pi_source is AgentSource.FALLBACK


def test_sdk_flavor_separates_transport_identity() -> None:
    """js SDK 与 python SDK 的传输身份必须不同（同源继承的判据）。"""
    from bladex_proxy.agent_bucket import bucket_unknown_agent

    pi = bucket_unknown_agent("OpenAI/JS 6.40.0", PI_HEADERS, {"bash", "read"})
    hermes = bucket_unknown_agent(
        "OpenAI/Python 1.99.1",
        {"x-stainless-lang": "python", "x-stainless-runtime": "CPython"},
        {"skill_manage"},
    )
    assert pi.origin_key != hermes.origin_key
    assert pi.basis.startswith("sdk:openai/js")


def test_sdk_flavor_survives_version_bumps() -> None:
    """SDK 升版本不得换传输身份（否则 agent 每升级一次就断一次继承）。"""
    from bladex_proxy.agent_bucket import bucket_unknown_agent

    keys = {
        bucket_unknown_agent(f"OpenAI/JS {v}", PI_HEADERS).origin_key
        for v in ("6.40.0", "6.41.2", "7.0.0")
    }
    assert len(keys) == 1


def test_same_sdk_different_tools_still_separate_buckets() -> None:
    """同一 SDK 下的两个陌生 agent 仍要靠工具集分开（SDK 风味区分度不够命名）。"""
    from bladex_proxy.agent_bucket import bucket_unknown_agent

    a = bucket_unknown_agent("OpenAI/JS 6.40.0", PI_HEADERS, {"bash", "read"})
    b = bucket_unknown_agent("OpenAI/JS 6.40.0", PI_HEADERS, {"draw", "render"})
    assert a.bucket_id != b.bucket_id
    assert a.origin_key == b.origin_key   # 传输层确实同源，这是对的


def _stranger_request() -> tuple[dict, ChatCompletionRequest]:
    """规则库里没有的陌生客户端 —— 用来验"未识别"那条路径。

    dsh 自 G11.10 起已进预置规则库、能被正常识别，不再适合当"未识别"的样本。
    """
    headers = {
        "authorization": "Bearer sk-shared-key",
        "user-agent": "some-new-agent/1.2.3",
    }
    return headers, ChatCompletionRequest(
        messages=[{"role": "user", "content": "hello"}]
    )


def test_dsh_after_hermes_is_not_signed_as_hermes() -> None:
    """🔴 2026-08-18 事故的最小复现：hermes 说完话后 dsh 请求不得继承 hermes。"""
    h_headers, h_req = _hermes_request()
    hermes_identity, hermes_source = resolve_identity(h_headers, h_req)
    assert hermes_identity.agent_id == "hermes:default"
    assert hermes_source is AgentSource.FINGERPRINT

    d_headers, d_req = _dsh_request()
    dsh_identity, _ = resolve_identity(d_headers, d_req)

    assert not dsh_identity.agent_id.startswith("hermes"), (
        f"dsh 被署名成 {dsh_identity.agent_id} —— 跨 agent 记忆污染回归"
    )
    # G11.10 之后 dsh 已进预置规则库，直接被指纹认出来（比落 unknown 桶更好的结果）
    assert dsh_identity.agent_id == "dsh"


def test_stranger_after_hermes_falls_into_its_own_bucket() -> None:
    """规则库没有的陌生 agent：不得继承 hermes，落自己的 unknown 桶。

    与上一条是同一个红线的两半——认识的走规则库，不认识的走分桶，
    两条路都不许借用别人的身份。
    """
    h_headers, h_req = _hermes_request()
    resolve_identity(h_headers, h_req)

    s_headers, s_req = _stranger_request()
    identity, source = resolve_identity(s_headers, s_req)

    assert not identity.agent_id.startswith("hermes")
    assert identity.agent_id.startswith("unknown-")
    assert source is AgentSource.FALLBACK


def test_stranger_bucket_is_stable_across_turns() -> None:
    """同一个未识别客户端多轮必须落同一个桶（稳定性是分桶的第一要求）。"""
    ids = set()
    for i in range(4):
        headers, req = _stranger_request()
        headers["x-some-new-agent-session-id"] = f"sess-{i}"
        identity, _ = resolve_identity(headers, req)
        ids.add(identity.agent_id)
    assert len(ids) == 1, f"同一 agent 落了多个桶: {ids}"


def test_two_unknown_agents_do_not_share_a_bucket() -> None:
    """两个不同的未识别 agent 不得共用一个命名空间（换名字的误合并）。"""
    a_headers, a_req = _stranger_request()
    a_identity, _ = resolve_identity(a_headers, a_req)

    other_headers = {
        "authorization": "Bearer sk-shared-key",
        "user-agent": "yet-another-agent/9.9",
    }
    other_req = ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}])
    other_identity, _ = resolve_identity(other_headers, other_req)

    assert a_identity.agent_id != other_identity.agent_id


# ── 不得回归：同源子请求仍要继承 ─────────────────────────────────────────────


def test_same_client_subrequest_still_inherits() -> None:
    """Hermes 无 system 指纹的内部子请求仍继承父 agent。

    不得回归 ADR-0021 修正解决的那个真实问题：子任务可能在主请求后十几分钟才发，
    退回时间窗口方案会让它变成未识别 → 误路由。同源判据用的是分桶键，不是时间。
    """
    h_headers, h_req = _hermes_request()
    parent, _ = resolve_identity(h_headers, h_req)
    assert parent.agent_id == "hermes:default"

    # 子请求：同一客户端（同 UA / 同 header 形态），但没有 system 指纹
    sub_headers = dict(h_headers)
    sub_req = ChatCompletionRequest(
        messages=[{"role": "user", "content": "给这段对话起个标题"}],
    )
    sub_identity, sub_source = resolve_identity(sub_headers, sub_req)

    assert sub_identity.agent_id == "hermes:default"
    assert sub_source is AgentSource.STICKY


def test_inheritance_is_not_time_limited() -> None:
    """同源继承不设时间上限（时间窗口方案已被 ADR-0021 实测否定）。"""
    h_headers, h_req = _hermes_request()
    resolve_identity(h_headers, h_req)

    # 直接改记录的时间戳，模拟"十几分钟后才发的子任务"
    for record in _agent_registry._records["local"]:
        record.identified_at -= 3600

    sub_req = ChatCompletionRequest(messages=[{"role": "user", "content": "标题"}])
    identity, source = resolve_identity(dict(h_headers), sub_req)
    assert identity.agent_id == "hermes:default"
    assert source is AgentSource.STICKY


# ── header 留存（G11.1 / MQ-A5）─────────────────────────────────────────────


def test_vendor_headers_reach_identity_and_are_retained() -> None:
    """厂商前缀 header 进得了门，并以脱敏形态留存。

    分桶判据即便在"已被规则库识别"的 agent 上也照算并留存——它是 dashboard 展示
    与后续认领的原料，不只服务于未识别路径。
    """
    headers, req = _dsh_request()
    identity, _ = resolve_identity(headers, req)

    assert "x-deepseek-harness-session-id" in identity.request_headers
    assert identity.agent_bucket_basis == "ua:deepseek-harness|vendor:deepseek-harness"


def test_credentials_never_reach_the_ledger() -> None:
    """🔴 负例：凭证不得出现在随 Turn 入库的任何字段里。"""
    headers, req = _dsh_request()
    headers["authorization"] = "Bearer sk-super-secret-value"
    identity, _ = resolve_identity(headers, req)

    assert "sk-super-secret-value" not in repr(identity.request_headers)
    assert identity.request_headers["authorization"] == "<redacted>"


def test_explicit_agent_id_still_wins() -> None:
    """显式 X-Agent-ID 优先级不变（刚性原则：配置压过推断）。"""
    headers, req = _dsh_request()
    headers["x-agent-id"] = "my-dsh"
    identity, source = resolve_identity(headers, req)
    assert identity.agent_id == "my-dsh"
    assert source is AgentSource.EXPLICIT


def test_profile_suffix_not_appended_to_bucket() -> None:
    """未识别桶不得被拼上 profile 后缀（会污染桶 id 的稳定性）。"""
    headers, req = _stranger_request()
    headers["x-agent-profile"] = "work"
    identity, _ = resolve_identity(headers, req)
    assert identity.agent_id.startswith("unknown-")
    assert ":" not in identity.agent_id
