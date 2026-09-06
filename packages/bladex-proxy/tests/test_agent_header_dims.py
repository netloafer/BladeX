"""① 预防层：preset 规则的 A/B 档 header 维（2026-08-26 五 agent 普查）。

🔴 事故：Codex 升级 0.147→0.149，**system prompt 措辞与工具集同时变了**，
两条内容维一起失效 ⇒ 落 `unknown-9d9bdd3e`，12 轮、10 条 fact 写进幻影命名空间。
而它一直在自报家门（`originator: Codex Desktop` 跨版本一字未动）。

本卡的判据不是"规则里有没有写 header"，是**"只剩 header 时还认不认得出来"**——
那才是升级发生时的真实条件。所有 header 样本取自 Hub 里的**真实流量快照**。
"""

from __future__ import annotations

from bladex_proxy.identity import _fingerprint_agent

#: 取自 Memory Hub `Identity.request_headers` 的真实快照（已脱敏，保留键名）。
REAL_HEADERS = {
    "codex": {
        "originator": "Codex Desktop",
        "user-agent": "Codex Desktop/0.149.0-alpha.4.3 (Mac OS 26.6.2; arm64)",
        "x-codex-window-id": "01a03893-43ec-7cc1-9c66-4df826a5c1e5:0",
        "x-codex-beta-features": "remote_compaction_v2",
        "x-agent-id": "", "x-session-id": "",
    },
    "claude-code": {
        "user-agent": "claude-cli/2.1.241 (external, cli)",
        "anthropic-beta": "claude-code-20250219,interleaved-thinking-2025-05-14",
        "anthropic-version": "2023-06-01",
        "x-app": "cli",
        "x-claude-code-session-id": "230ec9ae-6938-4f69-927d-369ea004a642",
    },
    "dsh": {
        "user-agent": "deepseek-harness/0.1.0-rc.5 "
                      "(+https://github.com/deepseek-ai/deepseek-harness)",
        "accept-language": "*", "sec-fetch-mode": "cors",
    },
}


class TestSurvivesContentSignalLoss:
    """🔴 核心：**内容维全部剥掉**（模拟版本升级改了措辞和工具名）后仍能识别。"""

    def test_codex_identified_by_headers_alone(self):
        agent, trigger, _aux = _fingerprint_agent([], None, REAL_HEADERS["codex"])
        assert agent == "codex", (
            "升级后 system prompt 与工具集都变了，只剩 header —— 认不出来就会"
            "重演 unknown-9d9bdd3e（12 轮 / 10 条 fact 进幻影命名空间）")
        assert "header" in trigger

    def test_claude_code_identified_by_headers_alone(self):
        agent, _t, _a = _fingerprint_agent([], None, REAL_HEADERS["claude-code"])
        assert agent == "claude-code"

    def test_dsh_identified_by_headers_alone(self):
        agent, _t, _a = _fingerprint_agent([], None, REAL_HEADERS["dsh"])
        assert agent == "dsh"


class TestNoFalsePositives:
    """🔴 阳性对照的反面：加了 header 维不能把别人也吸过来。
    误合并=0 在接入域的对应形态就是"别人的流量不许署我的名"。"""

    def test_hermes_not_captured_by_new_header_rules(self):
        """Hermes 零专有 header（UA 是纯 SDK 名）—— 新规则一条都不该命中它。"""
        hermes_headers = {"user-agent": "OpenAI/Python 2.24.0",
                          "x-agent-id": "", "x-session-id": ""}
        agent, _t, _a = _fingerprint_agent([], None, hermes_headers)
        assert agent is None, f"Hermes 被 header 维误吸成 {agent}"

    def test_pi_not_captured(self):
        """Pi 是 `OpenAI/JS`（通用 SDK 名）+ Node 运行时特征，同样不该命中。"""
        agent, _t, _a = _fingerprint_agent(
            [], None, {"user-agent": "OpenAI/JS 6.40.0",
                       "accept-language": "*", "sec-fetch-mode": "cors"})
        assert agent is None

    def test_bare_openai_sdk_not_captured(self):
        agent, _t, _a = _fingerprint_agent([], None, {"user-agent": "OpenAI/Python 2.24.0"})
        assert agent is None


class TestContentDimStillPrimary:
    """A/B 档是**补充**不是取代——Hermes/Pi 没有 A/B 档，内容维必须照常工作。"""

    def test_hermes_still_identified_by_system_prompt(self):
        msgs = [{"role": "system", "content": "You are a Hermes agent by Nous Research."}]
        agent, _t, _a = _fingerprint_agent(msgs, None, {"user-agent": "OpenAI/Python 2.24.0"})
        assert agent == "hermes"

    def test_codex_still_identified_by_content_without_headers(self):
        """老版本 Codex（没有 originator 的形态）不能因为加了 header 维就认不出。"""
        msgs = [{"role": "system", "content": "You are Codex, a coding agent."}]
        agent, _t, _a = _fingerprint_agent(msgs, None, {})
        assert agent == "codex"
