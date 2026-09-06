"""`suggest_rule` 的 UA 模式：**建议出来就该能用**（2026-08-26）。

🔴 病例：dashboard 合并对话框预填 `(?i)^codex/`，Jason 问"保持不动可以吗"——
不可以，**它匹配不上**。真实 UA 是 `Codex Desktop/0.149.0-alpha.4.3 (…)`，
产品名带空格；`parse_ua_product` 归一成首 token `codex`，而模板硬拼了斜杠：

    pattern = f"(?i)^{re.escape(product)}/"      # -> (?i)^codex/
    # codex 后面是**空格**不是 /  ⇒  匹配 False

单词产品名（`claude-cli/2.1.241`、`deepseek-harness/0.1.0`）恰好成立，所以这个
模板一直没露馅——Codex 是我们接的五个 agent 里**唯一的多词产品名**。

危害不是"少一条规则"，是**给了用户一条永远不命中的规则**：他会以为"下次自动
认出来"已经成立，而它不会，下一个版本照样落进新的 unknown 桶。
"""

from __future__ import annotations

import re

from bladex_proxy.agent_rules import suggest_rule

#: 取自 Memory Hub `Identity.request_headers` 的真实快照。
REAL_UA = {
    "codex": "Codex Desktop/0.149.0-alpha.4.3 (Mac OS 26.6.2; arm64) "
             "unknown (Codex Desktop; 26.818.61809)",
    "claude-code": "claude-cli/2.1.241 (external, cli)",
    "dsh": "deepseek-harness/0.1.0-rc.5 "
           "(+https://github.com/deepseek-ai/deepseek-harness)",
}
GENERIC_UA = {"hermes": "OpenAI/Python 2.24.0", "Pi": "OpenAI/JS 6.40.0"}


def _pattern(agent: str, ua: str) -> str:
    r = suggest_rule(agent, headers={"user-agent": ua}, existing=[])
    hp = (r.get("header_patterns") or [{}])[0]
    return hp.get("pattern", "")


class TestSuggestedPatternMatchesItsOwnSample:
    """判据只有一条：**建议器产出的规则，必须匹配它采样的那条 UA**。"""

    def test_all_real_agents(self):
        for agent, ua in REAL_UA.items():
            pat = _pattern(agent, ua)
            assert pat, f"{agent}: 没给出 UA 规则"
            assert re.search(pat, ua), f"{agent}: 规则 {pat!r} 匹配不上自己的 UA"

    def test_multiword_product_name_is_the_regression(self):
        """🔴 就是这一条挂过：产品名带空格时不能要求后面紧跟 `/`。"""
        pat = _pattern("codex", REAL_UA["codex"])
        assert "/" not in pat, f"又把斜杠硬拼进去了：{pat!r}"
        assert re.search(pat, REAL_UA["codex"])

    def test_still_matches_after_a_version_bump(self):
        """规则存在的意义就是扛版本升级——换个版本号照样要命中。"""
        pat = _pattern("codex", REAL_UA["codex"])
        assert re.search(pat, "Codex Desktop/0.200.0 (Mac OS 27; arm64)")


class TestNoUselessSuggestions:
    def test_generic_sdk_names_get_no_ua_rule(self):
        """Hermes / Pi 的 UA 是纯 SDK 名（无区分度）—— 不该给 UA 规则，
        给了就是"任何用 openai-python 的客户端都是 hermes"。"""
        for agent, ua in GENERIC_UA.items():
            assert _pattern(agent, ua) == "", f"{agent}: 给了通用 SDK 名一条 UA 规则"

    def test_self_check_drops_a_non_matching_pattern(self):
        """自证不过就**不给**，而不是给一条永远不命中的。
        （静默给错规则比不给更糟：用户以为识别已经修好了。）"""
        # 构造一个模板必然对不上的形态：产品名与 UA 开头不一致
        r = suggest_rule("x", headers={"user-agent": "(weird) NotAProduct"}, existing=[])
        pat = (r.get("header_patterns") or [{}])[0].get("pattern", "")
        assert not pat or re.search(pat, "(weird) NotAProduct")

    def test_word_boundary_does_not_swallow_a_longer_name(self):
        """`^codex\\b` 不许命中 `codexium/1.0` —— 边界要真的是边界。"""
        pat = _pattern("codex", REAL_UA["codex"])
        assert not re.search(pat, "codexium/1.0 (something)")
