"""G11.9 / G11.1 验收剧本 —— unknown-<hash> 分桶 + header 脱敏。

验收优先级按执行卡：**稳定性是第一验收项**，排在区分度之前。理由是失败模式不对
称——分不开只是多个 agent 挤一个桶（可合并），而不稳定是同一 agent 每轮换身份
（记忆碎成粉末，比现状更糟）。

真实样本出处（2026-08-18 调研，见台账接入域「UA 样本证据」）：
  dsh        直读 packages/llm/llm/src/attribution.ts + llm-deepseek/adapter.ts:283
  Hermes     直读 ~/.hermes/hermes-agent/hermes_cli/models.py:26
  其余       claude-code-hub 生产实测 UA 模式表
"""

from __future__ import annotations

import pytest

from bladex_proxy.agent_bucket import (
    GENERIC_CLIENT_TOKENS,
    BucketResult,
    bucket_unknown_agent,
    extract_vendor_segments,
    parse_ua_product,
    redact_headers,
)

# ── 真实样本 ────────────────────────────────────────────────────────────────

DSH_UA = "deepseek-harness/0.3.1 (+https://github.com/deepseek-ai/deepseek-harness)"
DSH_HEADERS = {
    "authorization": "Bearer sk-live-should-never-be-logged",
    "content-type": "application/json",
    "accept": "text/event-stream",
    "user-agent": DSH_UA,
    "x-deepseek-harness-user-id": "anon-7f3a",
    "x-deepseek-harness-session-id": "sess-20260818-001",
}


# ── 1. 稳定性（第一验收项）───────────────────────────────────────────────────


def test_same_agent_same_bucket_across_sessions() -> None:
    """换 session id 不得换桶 —— session id 每会话都变，进 hash 即每轮一个身份。"""
    buckets = set()
    for i in range(5):
        headers = dict(DSH_HEADERS, **{"x-deepseek-harness-session-id": f"sess-{i}"})
        buckets.add(bucket_unknown_agent(DSH_UA, headers).bucket_id)
    assert len(buckets) == 1, f"session 变动导致换桶: {buckets}"


def test_same_agent_same_bucket_across_versions() -> None:
    """升版本号不得换桶 —— 抗漂移的机制是"去版本"，不是维护名单。"""
    buckets = {
        bucket_unknown_agent(
            f"deepseek-harness/{ver} (+https://github.com/deepseek-ai/deepseek-harness)",
            DSH_HEADERS,
        ).bucket_id
        for ver in ("0.3.1", "0.4.0", "1.0.0-rc2", "2.11.7")
    }
    assert len(buckets) == 1, f"版本变动导致换桶: {buckets}"


def test_volatile_header_values_do_not_affect_bucket() -> None:
    """带 request-id / trace-id 这类每轮都变的值，桶必须不变（只读名不读值）。"""
    a = bucket_unknown_agent(DSH_UA, dict(DSH_HEADERS, **{"x-request-id": "req-1"}))
    b = bucket_unknown_agent(DSH_UA, dict(DSH_HEADERS, **{"x-request-id": "req-2"}))
    assert a.bucket_id == b.bucket_id


def test_header_order_does_not_affect_bucket() -> None:
    """header 顺序不得影响结果（vendor 段排序后再 hash）。"""
    reversed_headers = dict(reversed(list(DSH_HEADERS.items())))
    assert (
        bucket_unknown_agent(DSH_UA, DSH_HEADERS).bucket_id
        == bucket_unknown_agent(DSH_UA, reversed_headers).bucket_id
    )


# ── 2. 区分度 ───────────────────────────────────────────────────────────────


def test_distinct_agents_get_distinct_buckets() -> None:
    """dsh / claude-code / codex 三份真实 UA 各自成桶。"""
    ids = {
        bucket_unknown_agent(DSH_UA, DSH_HEADERS).bucket_id,
        bucket_unknown_agent("claude-code/2.0.37 (external, cli)", {}).bucket_id,
        bucket_unknown_agent("codex_cli_rs/0.9.2", {}).bucket_id,
        bucket_unknown_agent("gemini-cli/1.4.0", {}).bucket_id,
    }
    assert len(ids) == 4


def test_bucket_id_shape() -> None:
    result = bucket_unknown_agent(DSH_UA, DSH_HEADERS)
    assert result.bucket_id.startswith("unknown-")
    assert len(result.bucket_id) == len("unknown-") + 8


# ── 3. tool 签名补位（UA 无区分度时）─────────────────────────────────────────


def test_generic_sdk_ua_falls_back_to_tool_signature() -> None:
    """两个都走 openai-python 但工具集不同的请求，不得同桶。

    这是补位规则存在的理由：Hermes 主路径 UA 就是 SDK 默认值（models.py:26 的
    hermes-cli UA 只用于 /v1/models），光看 UA 分不开任何走裸 SDK 的 agent。
    """
    sdk_ua = "OpenAI/Python 1.99.1"
    a = bucket_unknown_agent(sdk_ua, {}, tool_names={"cordis_run", "ralph", "terminal_open"})
    b = bucket_unknown_agent(sdk_ua, {}, tool_names={"Read", "Write", "Bash"})
    assert a.bucket_id != b.bucket_id
    # 2026-08-19 起 basis 形如 `sdk:openai/python|tools:xxxx`：SDK 风味进了传输身份
    # （Pi 事故的修法），工具签名仍在，仍是把同 SDK 的两个陌生 agent 分开的那一项。
    assert "tools:" in a.basis and "tools:" in b.basis
    assert a.basis.startswith("sdk:openai/python")
    # 同源判据只到 SDK 风味为止，不含工具——子请求工具集会变
    assert a.origin_key == b.origin_key


def test_tool_signature_is_order_and_case_insensitive() -> None:
    sdk_ua = "python-httpx/0.27.0"
    a = bucket_unknown_agent(sdk_ua, {}, tool_names={"Read", "Write", "Bash"})
    b = bucket_unknown_agent(sdk_ua, {}, tool_names={"bash", "read", "write"})
    assert a.bucket_id == b.bucket_id


def test_tool_signature_not_used_when_ua_is_distinctive() -> None:
    """UA 有区分度时不看工具 —— 否则工具集一变（子 agent 带少量工具）就换桶。"""
    a = bucket_unknown_agent(DSH_UA, DSH_HEADERS, tool_names={"a", "b"})
    b = bucket_unknown_agent(DSH_UA, DSH_HEADERS, tool_names={"c"})
    assert a.bucket_id == b.bucket_id


def test_no_signal_at_all_still_deterministic() -> None:
    a = bucket_unknown_agent("", {})
    b = bucket_unknown_agent("", {})
    assert a.bucket_id == b.bucket_id
    assert a.basis == "none"


# ── 4. UA 解析 ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("ua", "expected"),
    [
        (DSH_UA, "deepseek-harness"),
        ("claude-code/2.0.37 (external, cli)", "claude-code"),
        ("claude-cli/2.0.37 (external, cli)", "claude-cli"),  # 同一 agent 的另一形态
        ("codex_cli_rs/0.9.2", "codex_cli_rs"),
        ("Codex Desktop/0.129.0-alpha.15", "codex"),
        ("gemini-cli/1.4.0", "gemini-cli"),
        ("opencode/0.3.0", "opencode"),
        ("OpenAI/Python 1.99.1", "openai"),
        ("python-httpx/0.27.0", "python-httpx"),
        ("", None),
        ("   ", None),
    ],
)
def test_parse_ua_product(ua: str, expected: str | None) -> None:
    assert parse_ua_product(ua) == expected


@pytest.mark.parametrize(
    "sdk_ua",
    [
        "OpenAI/Python 1.99.1",
        "python-httpx/0.27.0",
        "python-requests/2.32.3",
        "axios/1.7.2",
        "node-fetch/3.3.2",
        "undici/6.19.8",
        "Go-http-client/2.0",
        "curl/8.7.1",
        "litellm/1.44.0",
    ],
)
def test_generic_sdk_tokens_are_not_agent_names(sdk_ua: str) -> None:
    """SDK 名不得成为分桶信号 —— 否则所有走同一 SDK 的 agent 挤一个桶。"""
    token = parse_ua_product(sdk_ua)
    assert token in GENERIC_CLIENT_TOKENS
    # 且不带工具信号时，basis 里不出现 ua: 分量
    assert "ua:" not in bucket_unknown_agent(sdk_ua, {}).basis


# ── 5. 厂商前缀 header 提取 ─────────────────────────────────────────────────


def test_extract_vendor_from_real_dsh_headers() -> None:
    assert extract_vendor_segments(DSH_HEADERS) == ["deepseek-harness"]


def test_unknown_vendor_works_without_prior_knowledge() -> None:
    """不认识的厂商照样提取 —— 规则是通配的，不靠名单。"""
    assert extract_vendor_segments({"x-foo-bar-session-id": "s1"}) == ["foo-bar"]


@pytest.mark.parametrize(
    "header_name",
    [
        "x-agent-id",          # 我方约定的两段式
        "x-session-id",        # 我方约定
        "x-api-key",           # 凭证
        "x-request-id",        # 通用
        "x-forwarded-for",     # 基础设施
        "x-real-ip",
        "x-bladex-caller",     # 我方自有
        "x-stainless-lang",    # openai/anthropic SDK 代码生成器
        "content-type",        # 非 x- 前缀
    ],
)
def test_infra_and_own_headers_yield_no_vendor(header_name: str) -> None:
    assert extract_vendor_segments({header_name: "v"}) == []


def test_vendor_extraction_is_case_insensitive() -> None:
    assert extract_vendor_segments({"X-DeepSeek-Harness-Session-Id": "s"}) == [
        "deepseek-harness"
    ]


# ── 6. header 脱敏（G11.1）──────────────────────────────────────────────────


def test_credentials_never_survive_redaction() -> None:
    """负例：凭证值不得出现在脱敏结果的任何位置。"""
    secret = "sk-live-should-never-be-logged"
    out = redact_headers(DSH_HEADERS)
    assert secret not in repr(out)
    assert out["authorization"] == "<redacted>"


@pytest.mark.parametrize(
    "name",
    ["authorization", "Proxy-Authorization", "x-api-key", "api-key", "Cookie"],
)
def test_all_credential_headers_redacted(name: str) -> None:
    out = redact_headers({name: "secret-value"})
    assert "secret-value" not in repr(out)


def test_redaction_keeps_header_names() -> None:
    """键名必须保留 —— 键名本身就是识别信号，丢了这次留存就白做了。"""
    out = redact_headers(DSH_HEADERS)
    assert "x-deepseek-harness-session-id" in out
    assert extract_vendor_segments(out) == ["deepseek-harness"]


def test_redaction_truncates_long_values() -> None:
    """MQ-A22（2026-08-29）：200 静默截断把 x-codex-turn-metadata 的 JSON 切在
    半截——上限提到 4096 且截断必须留显式标记，不许静默丢数据。"""
    out = redact_headers({"x-huge": "a" * 5000, "x-fits": "b" * 4096})
    assert out["x-huge"] == "a" * 4096 + "…<bladex:truncated>"
    assert out["x-fits"] == "b" * 4096, "未超限不得带标记"


def test_redaction_lowercases_names() -> None:
    out = redact_headers({"User-Agent": DSH_UA, "Content-Type": "application/json"})
    assert set(out) == {"user-agent", "content-type"}


# ── 7. BucketResult 契约 ────────────────────────────────────────────────────


def test_bucket_result_exposes_basis_for_dashboard() -> None:
    """basis 要可读 —— dashboard 认领界面靠它告诉用户"这个桶是凭什么分出来的"。"""
    result = bucket_unknown_agent(DSH_UA, DSH_HEADERS)
    assert isinstance(result, BucketResult)
    assert result.basis == "ua:deepseek-harness|vendor:deepseek-harness"
