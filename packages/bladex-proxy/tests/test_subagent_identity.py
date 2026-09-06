"""② 子代理：agent 自报的子代理名走 profile 命名设施（2026-08-26，MQ-A12）。

🔴 病例：Codex 的 guardian 子代理 12 轮全带 `x-openai-subagent: guardian`，
而 UA / `originator` 与主路径**逐字相同**——传输层完全一样，唯一的区别就是这个
header，而它此前被 `VENDOR_SUFFIXES` 封闭枚举整条丢掉。后果：12 轮落
`unknown-9d9bdd3e`、**10 条 fact 写进幻影命名空间**。

设计要点（这两天反复踩的那条）：**"这是谁"与"这该不该写账本"是两个判据**。
子代理**有身份、有记忆价值**（guardian 的安全审查结论要留），但**不是任务账本的作者**
——所以给名字（`Identity.agent_id`）的同时置 `Identity.subagent`，
账本面按后者关掉，蒸馏与路由**不受影响**（不进 `auxiliary`）。
"""

from __future__ import annotations

from bladex_proxy.identity import resolve_subagent

CODEX_MAIN = {
    "originator": "Codex Desktop",
    "user-agent": "Codex Desktop/0.149.0-alpha.4.3 (Mac OS 26.6.2; arm64)",
    "x-codex-window-id": "01a03893-43ec-7cc1-9c66-4df826a5c1e5:0",
}
CODEX_GUARDIAN = {**CODEX_MAIN, "x-openai-subagent": "guardian",
                  "x-codex-parent-thread-id": "01a03893-43ec-7cc1-9c66-4df826a5c1e5"}


class TestResolve:
    def test_guardian_declared(self):
        assert resolve_subagent("codex", CODEX_GUARDIAN) == "guardian"

    def test_main_path_has_no_subagent(self):
        """🔴 阳性对照：主路径不发这个 header ⇒ 不能给它加后缀，
        否则主 codex 的记忆会被切到一个新命名空间去。"""
        assert resolve_subagent("codex", CODEX_MAIN) == ""

    def test_only_the_owning_agent_rule_is_consulted(self):
        """A 家的 header 不许给 B 家的轮次加后缀 —— 全表扫描会制造跨 agent 污染。"""
        assert resolve_subagent("hermes:default", CODEX_GUARDIAN) == ""

    def test_value_is_sanitised_into_the_namespace(self):
        """值会成为记忆命名空间的一段，字符集按 profile 同款收窄。"""
        assert resolve_subagent("codex", {"x-openai-subagent": "a/b:c d"}) == "a-b-c-d"

    def test_empty_value_is_not_a_subagent(self):
        assert resolve_subagent("codex", {"x-openai-subagent": "  "}) == ""

    def test_no_headers(self):
        assert resolve_subagent("codex", None) == ""
        assert resolve_subagent("", CODEX_GUARDIAN) == ""


class TestWiring:
    def test_identity_carries_subagent_flag(self):
        from bladex_proxy.models import Identity
        assert Identity(user_id="u").subagent is False

    def test_ledger_face_is_off_for_subagents(self):
        """账本面三个门（aux / subagent / 孤立子调用）必须在单一判据里齐。

        V-A1 T2 编排函数化（2026-08-28）后判据只有 `_apply_agency_surfaces`
        里一份（原六处副本形态被守卫升级取代），三端点全部经它走
        （调用点数由 test_ledger_require_tools 的接线守卫钉住）。
        取源按路径读文本（MQ-V9）。"""
        from pathlib import Path
        src = (Path(__file__).resolve().parents[1]
               / "bladex_proxy" / "server.py").read_text(encoding="utf-8")
        assert src.count("identity.subagent") == 1, (
            "账本域判据必须只有 _apply_agency_surfaces 里那一份；"
            "多出来 = 有人在端点里重新内联（分叉复活），少了 = 子代理能开账本")

    def test_subagent_is_not_auxiliary(self):
        """🔴 不能顺手并进 `auxiliary`：那会连带跳蒸馏 + 路由降 weak 档，
        而 guardian 需要真实模型能力、它的结论也有记忆价值。
        （`auxiliary` 三个消费方共用一个旋钮，正是 2026-07-28 那次事故的形状。）"""
        from bladex_proxy.models import Identity
        i = Identity(user_id="u", subagent=True)
        assert i.auxiliary is False and i.aux_source == ""


class TestRuleSchema:
    def test_codex_preset_declares_subagent_header(self):
        from bladex_proxy.agent_rules import load_agent_rules
        codex = next(r for r in load_agent_rules() if r.agent_id == "codex")
        assert "x-openai-subagent" in codex.subagent_headers

    def test_round_trips_through_user_rule_render(self):
        from bladex_proxy.agent_rules import parse_rules, render_rule_toml
        toml = render_rule_toml({"name": "x", "agent_id": "x",
                                 "system_prompt_keywords": ["hello there"],
                                 "subagent_headers": ["x-foo-subagent"]})
        assert "subagent_headers" in toml
        assert parse_rules(toml)[0].subagent_headers == ["x-foo-subagent"]
