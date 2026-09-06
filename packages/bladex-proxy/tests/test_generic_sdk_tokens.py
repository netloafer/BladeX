"""通用 SDK 名单不许漏「同一 SDK 的异步客户端类名」（2026-08-26）。

🔴 病例：Hermes 的视觉子调用 UA 是 `AsyncOpenAI/Python 2.24.0`，
product token 归一成 `asyncopenai` —— 与 `openai` 是两个字符串，
于是它**逃过了 `GENERIC_CLIENT_TOKENS`**，`suggest_rule` 会给它一条 UA 规则。

危险度被同日的另一处改动放大了：合并路径已改为**自动采用**建议的 header 规则
（用户不再编辑），所以一点合并，`(?i)^asyncopenai\\b` 就会写进 hermes ——
**从此任何用 async-openai 的客户端都算 hermes**，正是"凭 SDK 名认 agent"
这个已经被否掉的判据借另一条路复活。
"""

from __future__ import annotations

from bladex_proxy.agent_bucket import GENERIC_CLIENT_TOKENS, parse_ua_product
from bladex_proxy.agent_rules import suggest_rule

#: 真实观测到的 UA（Hermes 视觉子调用）+ 同族形态。
SDK_UAS = [
    "AsyncOpenAI/Python 2.24.0",
    "OpenAI/Python 2.24.0",
    "OpenAI/JS 6.40.0",
]


class TestNoRuleForGenericSdkNames:
    def test_async_client_names_are_generic(self):
        """`AsyncOpenAI` 与 `OpenAI` 是同一个 SDK 的两个类名 ——
        判据（"标识 SDK 不是 agent"）对两者完全一样，名单不该只收一个。"""
        assert parse_ua_product("AsyncOpenAI/Python 2.24.0") in GENERIC_CLIENT_TOKENS

    def test_no_ua_rule_suggested_for_any_sdk_ua(self):
        for ua in SDK_UAS:
            r = suggest_rule("hermes", headers={"user-agent": ua}, existing=[])
            assert not r.get("header_patterns"), (
                f"{ua!r} 被建议成一条 UA 规则 ⇒ 认领后任何用这个 SDK 的客户端"
                f"都会被算成该 agent")

    def test_real_agent_names_still_get_rules(self):
        """🔴 阳性对照：不能为了堵漏把所有 UA 规则都关掉 ——
        专有自报名（A 档）仍必须产出规则，那是抗版本漂移的主力。"""
        for agent, ua in (("codex", "Codex Desktop/0.149.0-alpha.4.3 (Mac OS 26; arm64)"),
                          ("claude-code", "claude-cli/2.1.241 (external, cli)"),
                          ("dsh", "deepseek-harness/0.1.0-rc.5 (+https://github.com/x)")):
            r = suggest_rule(agent, headers={"user-agent": ua}, existing=[])
            assert r.get("header_patterns"), f"{agent}: 专有自报名反而没给规则"


class TestMergeOfAnAsyncSdkBucketWritesNoRule:
    """端到端：合并 Hermes 的视觉子调用桶时，**不该**顺手写一条 UA 规则。"""

    def test_hermes_subcall_bucket_yields_no_header_rule(self):
        r = suggest_rule("hermes",
                         headers={"user-agent": "AsyncOpenAI/Python 2.24.0",
                                  "x-agent-id": "", "x-session-id": ""},
                         existing=[])
        assert not r.get("header_patterns")
