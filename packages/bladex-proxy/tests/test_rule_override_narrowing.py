"""认领写规则**不许缩小识别面**（2026-08-26 事故，A+B 两条修法）。

# 事故还原

dashboard 认领 codex，写了一条 `name="codex"` 的用户规则（只带一条 UA header 维）。
加载侧的语义是"用户表同名 ⇒ **覆盖**预置"，于是 codex 的识别面从

    2 关键词 + 3 工具签名 + 2 header 维 + subagent_headers

缩成 **一条 UA 规则**。后果不止"识别变脆"：`subagent_headers` 没了，
`codex:guardian` 那层命名当场失效（②整层白做）。

🔴 最难受的是**它是静默的**：dashboard 上 codex 仍显示 `rule`、一切正常，
只有把规则 dump 出来才看得见。

# 两条修法各管一半

- **A** 认领这条写入路径改成"补"（与同名预置逐维取并集）。
  覆盖语义本身不动 —— 刚性原则 9：用户显式写的必须能盖过我们替他做的推断。
  收窄的只是"认领"这个**系统代用户生成**的动作，它的意图是补一条，不是重定义。
- **B** 无论谁写的，只要覆盖导致某一维**从有到无**，加载时打可 grep 的告警。
  A 之后认领不再触发它，但用户手工编辑仍可能触发，而那时更需要看见。
"""

from __future__ import annotations

import structlog.testing
from bladex_proxy.agent_rules import (
    append_user_rule,
    load_agent_rules,
    merge_with_preset,
    parse_rules,
    reset_rules_cache,
)

CLAIM = {"name": "codex", "agent_id": "codex",
         "header_patterns": [{"name": "user-agent", "pattern": r"(?i)^codex\b"}]}


def _preset(agent_id: str):
    from bladex_proxy.agent_rules import PRESET_RULES_PATH
    return next(r for r in parse_rules(PRESET_RULES_PATH.read_text(encoding="utf-8"))
                if r.agent_id == agent_id)


class TestMergeWithPreset:
    def test_no_preset_dimension_is_lost(self):
        """🔴 病例本身：认领 codex 之后，预置的四维一个都不能少。"""
        m = merge_with_preset(CLAIM)
        p = _preset("codex")
        assert set(p.system_prompt_keywords) <= set(m["system_prompt_keywords"])
        assert set(p.tool_signatures) <= set(m["tool_signatures"])
        assert set(p.subagent_headers) <= set(m["subagent_headers"]), \
            "subagent_headers 丢了 ⇒ codex:guardian 那层命名失效"
        names = {h["name"] for h in m["header_patterns"]}
        assert {h.name for h in p.header_patterns} <= names

    def test_user_value_comes_first(self):
        """用户给的排在前——他的判断优先，预置只是补齐。"""
        m = merge_with_preset(CLAIM)
        assert m["header_patterns"][0]["name"] == "user-agent"

    def test_no_duplicates(self):
        """重复认领两次不该让规则里堆一堆重复项。"""
        once = merge_with_preset(CLAIM)
        twice = merge_with_preset(once)
        assert len(twice["header_patterns"]) == len(once["header_patterns"])
        assert len(twice["tool_signatures"]) == len(once["tool_signatures"])

    def test_unknown_agent_passes_through(self):
        """新 agent 没有同名预置 —— 原样返回，不该凭空长出别人的维度。"""
        r = {"name": "brand-new", "agent_id": "brand-new",
             "system_prompt_keywords": ["hello there friend"]}
        assert merge_with_preset(r) == r

    def test_merged_rule_is_writable(self, tmp_path):
        """并集要能过 render + parse 往返（否则写盘时才炸）。"""
        p = tmp_path / "agent_rules.toml"
        append_user_rule(merge_with_preset(CLAIM), path=p)
        got = next(r for r in parse_rules(p.read_text(encoding="utf-8"))
                   if r.agent_id == "codex")
        assert got.subagent_headers == ["x-openai-subagent"]
        assert len(got.header_patterns) == 3


class TestAppendMergesByDefault:
    def test_default_is_merge(self, tmp_path):
        p = tmp_path / "agent_rules.toml"
        append_user_rule(dict(CLAIM), path=p)
        got = next(r for r in parse_rules(p.read_text(encoding="utf-8"))
                   if r.agent_id == "codex")
        assert got.subagent_headers, "默认没合并 ⇒ 事故会复发"

    def test_opt_out_still_available(self, tmp_path):
        """手工路径若确实想覆盖预置，仍可以 —— 只是不再是默认。"""
        p = tmp_path / "agent_rules.toml"
        append_user_rule(dict(CLAIM), path=p, merge_preset=False)
        got = next(r for r in parse_rules(p.read_text(encoding="utf-8"))
                   if r.agent_id == "codex")
        assert not got.subagent_headers


class TestNarrowingWarning:
    def test_warns_and_names_the_lost_dimensions(self, tmp_path):
        """B：覆盖导致某一维从有到无 ⇒ 必须打告警，并**说清丢了哪几维**。"""
        p = tmp_path / "agent_rules.toml"
        p.write_text('[[rules]]\nname = "codex"\nagent_id = "codex"\n'
                     'header_patterns = [{ name = "user-agent", pattern = "x" }]\n',
                     encoding="utf-8")
        reset_rules_cache()
        with structlog.testing.capture_logs() as cap:
            load_agent_rules(user_path=p)
        rows = [e for e in cap
                if e.get("event") == "agent_rule_override_narrows_recognition"]
        assert rows, "识别面被缩小却没有任何告警 —— 这正是事故最难受的地方"
        lost = set(rows[0]["lost_dimensions"])
        assert {"system_prompt_keywords", "tool_signatures", "subagent_headers"} <= lost

    def test_merged_rule_does_not_warn(self, tmp_path):
        """阳性对照：A 修好之后，认领写出的规则不该再触发 B 的告警。
        没有这一半，上面那条可能只是在验证"永远告警"。"""
        p = tmp_path / "agent_rules.toml"
        append_user_rule(merge_with_preset(CLAIM), path=p)
        reset_rules_cache()
        with structlog.testing.capture_logs() as cap:
            load_agent_rules(user_path=p)
        assert not [e for e in cap
                    if e.get("event") == "agent_rule_override_narrows_recognition"]

    def test_unrelated_user_rule_does_not_warn(self, tmp_path):
        """没有同名预置的用户规则（如 Pi）不该被误报。"""
        p = tmp_path / "agent_rules.toml"
        p.write_text('[[rules]]\nname = "Pi"\nagent_id = "Pi"\n'
                     'system_prompt_keywords = ["operating inside pi"]\n',
                     encoding="utf-8")
        reset_rules_cache()
        with structlog.testing.capture_logs() as cap:
            load_agent_rules(user_path=p)
        assert not [e for e in cap
                    if e.get("event") == "agent_rule_override_narrows_recognition"]


class TestGenericUaWarning:
    """对称检查：**过宽的 UA 规则**（2026-08-26，同日第二次同型事故）。

    上一条管"识别面缩小"，这条管"识别面过宽"——两个方向都会坏事，且都静默。

    🔴 两次事故同一形状：`suggest_rule` 的缺陷让认领写出一条匹配**通用 SDK 名**
    的 UA 规则（第一次 `(?i)^codex/` 匹配不上，第二次 `(?i)^asyncopenai\\b` 过宽）。
    代码侧都修了，但**认领会把当时进程内的代码版本固化成磁盘上的配置**，
    而 `gate_check` 只跑测试不重启服务 —— 用户很自然在 gate_check 通过后立刻操作。

    所以检查放在**加载时**：无论那条规则当初怎么写进去的（认领生成 / 手工编辑 /
    从别处拷来），下次启动都会喊出来。
    """

    def _rules(self, toml_text: str):
        return parse_rules(toml_text, source="user")

    def test_flags_a_generic_sdk_ua_rule(self):
        from bladex_proxy.agent_rules import _generic_ua_rules
        rules = self._rules(
            '[[rules]]\nname = "hermes"\nagent_id = "hermes"\n'
            'header_patterns = [{ name = "user-agent", pattern = "(?i)^asyncopenai\\\\b" }]\n')
        hits = _generic_ua_rules(rules)
        assert hits and hits[0][0] == "hermes" and hits[0][1] == "asyncopenai"

    def test_does_not_flag_proprietary_product_names(self):
        """🔴 阳性对照：专有自报名（A 档）是抗版本漂移的主力，不能被误报。"""
        from bladex_proxy.agent_rules import _generic_ua_rules
        for agent, pat in (("codex", r"(?i)^codex\\b"),
                           ("claude-code", r"(?i)^claude\\-cli\\b"),
                           ("dsh", r"(?i)^deepseek\\-harness\\b")):
            rules = self._rules(
                f'[[rules]]\nname = "{agent}"\nagent_id = "{agent}"\n'
                f'header_patterns = [{{ name = "user-agent", pattern = "{pat}" }}]\n')
            assert not _generic_ua_rules(rules), f"{agent} 被误报成通用 SDK 名"

    def test_judged_by_running_the_regex_not_by_text(self):
        """判据是**让正则对着真实形态跑**，不是对模式串做文本匹配——
        写法千变万化（`^openai\\b` / `openai` / `(?:async)?openai`），
        只有真跑一遍才不依赖某一种写法。"""
        from bladex_proxy.agent_rules import _generic_ua_rules
        rules = self._rules(
            '[[rules]]\nname = "x"\nagent_id = "x"\n'
            'header_patterns = [{ name = "user-agent", pattern = "(?:async)?openai" }]\n')
        assert _generic_ua_rules(rules), "换个写法就漏 ⇒ 判据挂在文本形状上了"

    def test_other_header_dims_are_not_judged(self):
        """只看 user-agent 维：`x-codex-window-id` 之类的专有命名空间不参与。"""
        from bladex_proxy.agent_rules import _generic_ua_rules
        rules = self._rules(
            '[[rules]]\nname = "x"\nagent_id = "x"\n'
            'header_patterns = [{ name = "originator", pattern = "openai" }]\n')
        assert not _generic_ua_rules(rules)

    def test_warning_is_emitted_on_load(self, tmp_path):
        p = tmp_path / "agent_rules.toml"
        p.write_text('[[rules]]\nname = "hermes"\nagent_id = "hermes"\n'
                     'system_prompt_keywords = ["hermes agent"]\n'
                     'header_patterns = [{ name = "user-agent", pattern = "(?i)^asyncopenai\\\\b" }]\n',
                     encoding="utf-8")
        reset_rules_cache()
        with structlog.testing.capture_logs() as cap:
            load_agent_rules(user_path=p)
        rows = [e for e in cap if e.get("event") == "agent_rule_ua_matches_generic_sdk"]
        assert rows, "过宽的 UA 规则没有任何告警 —— 与上次事故同样静默"
        assert rows[0]["sdk_token"] == "asyncopenai"

    def test_clean_rules_emit_nothing(self, tmp_path):
        """阳性对照的另一半：正常规则不许刷屏。"""
        p = tmp_path / "agent_rules.toml"
        p.write_text('[[rules]]\nname = "Pi"\nagent_id = "Pi"\n'
                     'system_prompt_keywords = ["operating inside pi"]\n', encoding="utf-8")
        reset_rules_cache()
        with structlog.testing.capture_logs() as cap:
            load_agent_rules(user_path=p)
        assert not [e for e in cap if e.get("event") == "agent_rule_ua_matches_generic_sdk"]

    def test_preset_rules_are_clean(self):
        """自查：我们自己的预置表里不许有这种规则。"""
        from bladex_proxy.agent_rules import PRESET_RULES_PATH, _generic_ua_rules
        presets = parse_rules(PRESET_RULES_PATH.read_text(encoding="utf-8"))
        assert not _generic_ua_rules(presets)


class TestRegexRoundTrip:
    """🔴 `render_rule_toml` 把正则写坏了（2026-08-26 live，最基础的一条）。

    它把 pattern 原样插进 TOML **基本字符串**（双引号），而基本字符串会处理转义。
    正则里出现反斜杠是**常态不是例外**，于是：

        `(?i)^codex\\b`  → 写盘 → 读回 `(?i)^codex\\x08`（退格符！）—— **静默损坏**
        `\\d+`           → 写盘 → **整份规则文件解析失败**

    第一种最难受：规则从此永不命中，而 dashboard 上一切正常。
    live 实证：认领 hermes 写出的 `(?i)^asyncopenai\\b` 落盘后其实是
    `(?i)^asyncopenai\\x08`，一个客户端都匹配不上 —— 我为此还误判过一轮
    （以为它"过宽"，其实它谁都匹配不上）。

    修法：正则用 TOML **字面量字符串**（`'…'`，不处理任何转义），
    那正是为这种内容准备的语法。
    """

    PATTERNS = [
        r"(?i)^codex\b",            # 词边界：静默损坏的那一类
        r"\d+",                     # 非法转义：整份文件炸掉的那一类
        r"a\.b",
        r"(?i)^claude\-cli\b",
        "plain-no-backslash",
        "it's-got-a-quote",         # 退回基本字符串的分支
    ]

    def test_every_pattern_survives_write_and_read(self):
        from bladex_proxy.agent_rules import render_rule_toml
        for pat in self.PATTERNS:
            toml = render_rule_toml({"name": "x", "agent_id": "x",
                                     "header_patterns": [{"name": "ua", "pattern": pat}]})
            back = parse_rules(toml)[0].header_patterns[0].pattern
            assert back == pat, f"{pat!r} 往返后变成 {back!r}"

    def test_keywords_survive_too(self):
        """关键词也走同一个渲染函数——中文、制表符、引号都不许被改写。"""
        from bladex_proxy.agent_rules import render_rule_toml
        kws = ["带中文的关键词", "has\ttab", "you are codex"]
        toml = render_rule_toml({"name": "x", "agent_id": "x",
                                 "system_prompt_keywords": kws})
        assert parse_rules(toml)[0].system_prompt_keywords == kws

    def test_written_rule_actually_matches_live_traffic(self):
        """端到端判据：写盘再读回之后，规则要真的能命中它本来该命中的 UA。
        （只断言"字符串相等"不够——那是往返，不是**可用**。）"""
        import re

        from bladex_proxy.agent_rules import render_rule_toml
        toml = render_rule_toml({
            "name": "codex", "agent_id": "codex",
            "header_patterns": [{"name": "user-agent", "pattern": r"(?i)^codex\b"}]})
        pat = parse_rules(toml)[0].header_patterns[0].pattern
        assert re.search(pat, "Codex Desktop/0.149.0-alpha.4.3 (Mac OS 26; arm64)")

    def test_generic_ua_warning_now_fires(self):
        """连带确认：往返修好之后，**过宽 UA 规则的告警才可能生效**。
        在此之前正则被写坏、跑不出结果，那条检查形同虚设——
        两个 bug 叠在一起会互相掩护。"""
        from bladex_proxy.agent_rules import _generic_ua_rules, render_rule_toml
        toml = render_rule_toml({
            "name": "hermes", "agent_id": "hermes",
            "header_patterns": [{"name": "user-agent",
                                 "pattern": r"(?i)^asyncopenai\b"}]})
        assert _generic_ua_rules(parse_rules(toml, source="user"))
