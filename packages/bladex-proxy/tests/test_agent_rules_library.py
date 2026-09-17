"""G11.10 验收剧本 —— agent 规则样本库外置。

两条硬判据（执行卡）：
  ① **纯 header 规则不得成为唯一形态** —— 拿一条只有 header 维的规则去匹配
     Hermes 主路径请求（UA 是 OpenAI SDK 默认值）必须匹配失败，证明多信号是必需的；
  ② 现有 `_AGENT_RULES` 迁移后识别结果逐条等价（零回归）。
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from bladex_proxy.agent_rules import (
    PRESET_RULES_PATH,
    HeaderPattern,
    load_agent_rules,
    parse_rules,
    reset_rules_cache,
)
from bladex_proxy.identity import _fingerprint_agent, resolve_identity
from bladex_proxy.models import ChatCompletionRequest


@pytest.fixture(autouse=True)
def _clean():
    from bladex_proxy.agent_registry import _agent_registry
    from bladex_proxy.identity import _session_cache

    reset_rules_cache()
    _agent_registry.clear()
    _session_cache.clear()
    yield
    reset_rules_cache()
    _agent_registry.clear()
    _session_cache.clear()


# ── ① 纯 header 规则的射程限制（本组最要害的一条）────────────────────────────


def test_header_only_rule_cannot_match_hermes_main_path(tmp_path) -> None:
    """🔴 只有 header 维的规则匹配不上 Hermes —— 所以规则库必须是多信号的。

    Hermes 的 chat 主路径走 OpenAI SDK，`_HERMES_USER_AGENT`（hermes-cli/<ver>）
    只用在 /v1/models 与 pricing catalog 上。任何"把规则库简化成 header 正则"
    的改动都会在这条上翻车。
    """
    rules_file = tmp_path / "agent_rules.toml"
    rules_file.write_text(
        textwrap.dedent("""
            [[rules]]
            name = "hermes-header-only"
            agent_id = "hermes-by-header"
            header_patterns = [{ name = "user-agent", pattern = "(?i)hermes" }]
        """),
        encoding="utf-8",
    )
    rules = parse_rules(rules_file.read_text(encoding="utf-8"), source="user")
    (rule,) = rules

    # Hermes 主路径实际发的 header
    hermes_headers = {"user-agent": "OpenAI/Python 1.99.1"}
    assert not any(hp.matches(hermes_headers) for hp in rule.header_patterns)

    # 而 system prompt 这道能认出来（多信号的价值就在这里）
    agent_id, trigger, _ = _fingerprint_agent(
        [{"role": "system", "content": "You are a Hermes agent built by Nous Research."}],
        None,
        hermes_headers,
    )
    assert agent_id == "hermes"
    assert trigger.startswith("system_prompt:")


def test_volatile_header_uses_presence_not_value() -> None:
    """值每轮都变的 header 只能判"存在"，不能拿值做判据。"""
    hp = HeaderPattern(name="x-deepseek-harness-session-id", pattern="")
    assert hp.matches({"x-deepseek-harness-session-id": "sess-1"})
    assert hp.matches({"x-deepseek-harness-session-id": "sess-2"})
    assert not hp.matches({})


# ── ② 迁移零回归 ────────────────────────────────────────────────────────────


def _preset() -> list:
    return parse_rules(PRESET_RULES_PATH.read_text(encoding="utf-8"))


def test_preset_parses_and_keeps_aux_rules_first() -> None:
    rules = _preset()
    assert rules, "预置规则表为空 —— 打包漏了 agent_rules.toml？"
    hermes = [r for r in rules if r.agent_id == "hermes"]
    assert hermes[0].auxiliary and hermes[1].auxiliary, "auxiliary 规则必须排在 base 之前"
    assert not hermes[-1].auxiliary


def test_preset_preserves_hermes_profile_awareness() -> None:
    base = next(r for r in _preset() if r.name == "hermes")
    assert base.profile_aware is True


@pytest.mark.parametrize(
    ("system_text", "expected"),
    [
        ("You are a Hermes agent built by Nous Research.", "hermes"),
        ("You are Claude Code, Anthropic's CLI.", "claude-code"),
        ("You are codex, a coding agent.", "codex"),
        ("You are cursor, running in Cursor IDE.", "cursor"),
        ("This is openclaw speaking.", "openclaw"),
        ("You are an AI agent powered by DeepSeek Harness.", "dsh"),
        # MQ-A68：opencode 首次连 BladeX 即落 unknown-e0ca7fa5（兜底分桶从 UA
        # 读出了 "opencode" 这个名字，规则库里却没有它）。live wire 原文照抄。
        ("You are opencode, an interactive CLI tool that helps users with "
         "software engineering tasks.", "opencode"),
    ],
)
def test_system_prompt_recognition_unchanged(system_text: str, expected: str) -> None:
    agent_id, _, _ = _fingerprint_agent([{"role": "system", "content": system_text}], None)
    assert agent_id == expected


def test_hermes_profile_extraction_unchanged() -> None:
    agent_id, _, _ = _fingerprint_agent(
        [{
            "role": "system",
            "content": "You are a Hermes agent.\nActive Hermes profile: accept",
        }],
        None,
    )
    assert agent_id == "hermes:accept"


def test_moa_reference_still_marked_auxiliary() -> None:
    _, _, aux = _fingerprint_agent(
        [{"role": "system", "content": "You are a reference advisor in a mixture of agents."}],
        None,
    )
    assert aux is True


# ── MQ-A6：单个通用 tool 名不足以定 agent ───────────────────────────────────


def test_session_search_alone_no_longer_claims_hermes() -> None:
    """🔴 dsh 也有 `session_search`；单命中定 hermes 会造成跨 agent 记忆污染。"""
    agent_id, _, _ = _fingerprint_agent(
        [{"role": "user", "content": "hi"}],
        [{"function": {"name": "session_search"}}],
    )
    assert agent_id != "hermes"


@pytest.mark.parametrize(
    "tool", ["skill_manage", "skills_list", "skill_view", "web_extract"]
)
def test_hermes_exclusive_tool_alone_still_recognizes_hermes(tool: str) -> None:
    """收紧不得误伤：Hermes 真正独有的工具**单个命中**仍要认出来。

    这条钉住的是修法的层次——MQ-A6 要治的是 `session_search` 那个**通用名**，
    不是门槛。曾试过把 `min_tool_match` 提到 2，结果打挂三条既有识别用例：
    `skill_manage` 本就是 Hermes 专名，单命中理应算数。
    """
    agent_id, trigger, _ = _fingerprint_agent(
        [{"role": "user", "content": "hi"}],
        [{"function": {"name": tool}}],
    )
    assert agent_id == "hermes"
    assert trigger.startswith("tool:")


# ── 用户规则覆盖 ────────────────────────────────────────────────────────────


def test_user_rules_take_precedence(tmp_path) -> None:
    """刚性原则 9：用户配置压过预置（预置本质上是我们替用户做的推断）。"""
    user_file = tmp_path / "agent_rules.toml"
    user_file.write_text(
        textwrap.dedent("""
            [[rules]]
            name = "my-own-hermes"
            agent_id = "hermes-fork"
            system_prompt_keywords = ["hermes agent"]
        """),
        encoding="utf-8",
    )
    rules = load_agent_rules(user_path=user_file)
    assert rules[0].agent_id == "hermes-fork"
    assert rules[0].source == "user"
    # 预置仍在后面
    assert any(r.agent_id == "claude-code" for r in rules)


def test_same_name_user_rule_replaces_preset(tmp_path) -> None:
    """同名视为覆盖而非叠加 —— 否则改了一条另一条仍在前面命中，"改了没用"最难查。"""
    user_file = tmp_path / "agent_rules.toml"
    user_file.write_text(
        textwrap.dedent("""
            [[rules]]
            name = "claude-code"
            agent_id = "cc-custom"
            system_prompt_keywords = ["claude code"]
        """),
        encoding="utf-8",
    )
    rules = load_agent_rules(user_path=user_file)
    named = [r for r in rules if r.name == "claude-code"]
    assert len(named) == 1
    assert named[0].agent_id == "cc-custom"


def test_missing_user_file_is_fine(tmp_path) -> None:
    rules = load_agent_rules(user_path=tmp_path / "nope.toml")
    assert any(r.agent_id == "hermes" for r in rules)


# ── 规则建议（系统提议、用户改，而不是让用户手写）────────────────────────────


def test_suggest_picks_self_introduction_from_system_prompt() -> None:
    """自我介绍那行辨识度最高，优先取它。"""
    from bladex_proxy.agent_rules import suggest_rule

    rule = suggest_rule(
        "pi",
        system_excerpt="Some preamble line.\nYou are Pi, a coding agent.\nMore text.",
        tools=["pi_edit", "pi_run"],
    )
    assert rule["agent_id"] == "pi"
    assert rule["system_prompt_keywords"] == ["You are Pi, a coding agent"]
    assert rule["tool_signatures"] == ["pi_edit", "pi_run"]


def test_suggest_never_reuses_a_tool_name_another_agent_claims() -> None:
    """🔴 建议器不得再制造 MQ-A6 那种冲突：别家规则占用的工具名不进建议。

    `skill_manage` 在预置的 hermes 规则里，建议给新 agent 就等于让两条规则抢同一个
    信号——正是当年 `session_search` 把 dsh 误判成 hermes 的形态。

    **已知射程限制**：建议器只能看到"**有规则占用**"的名字。像 `session_search`
    这种通用叫法，因为 MQ-A6 修法已把它从 hermes 规则里移除，现在**不被任何规则占用**
    → 建议器不会滤掉它。它没有"这个名字在生态里很常见"的知识，也不该硬维护这样一张
    名单（那是被否掉过的路子）。兜底靠"建议 ≠ 决定"：预填进可编辑输入框，用户删得掉。
    """
    from bladex_proxy.agent_rules import suggest_rule

    rule = suggest_rule("newbie", tools=["skill_manage", "newbie_only"])
    assert rule.get("tool_signatures") == ["newbie_only"]


def test_suggest_drops_generic_tool_names_when_a_keyword_exists() -> None:
    """🔴 通用工具名不进建议——单个 `bash` 谁都有。

    Pi 实测（2026-08-19）：它的工具是 `bash`/`edit`/`read`/`write`，没有任何规则
    占用，但全是通用单词。配 `min_tool_match=1` 就等于"任何带 bash 的 agent 都是 Pi"
    ——`session_search` 把 dsh 误判成 hermes 的同一形状。
    规则是"任一信号命中即算"，所以有强关键词时加弱信号只会扩大误命中面。
    """
    from bladex_proxy.agent_rules import suggest_rule

    rule = suggest_rule(
        "pi",
        system_excerpt="You are an expert coding assistant operating inside pi.",
        tools=["bash", "edit", "read", "write"],
    )
    assert rule["system_prompt_keywords"]
    assert "tool_signatures" not in rule


def test_suggest_requires_several_generic_tools_when_they_are_the_only_signal() -> None:
    """没有关键词时工具是唯一信号，那就要求同时命中多个。"""
    from bladex_proxy.agent_rules import suggest_rule

    rule = suggest_rule("x", tools=["bash", "edit", "read", "write"])
    assert rule["tool_signatures"] == ["bash", "edit", "read", "write"]
    assert rule["min_tool_match"] == 2


def test_suggest_prefers_namespaced_tool_names() -> None:
    """带命名空间的是生态专名，单命中即可；判据是形状不是名单。"""
    from bladex_proxy.agent_rules import suggest_rule

    # 🔴 用**没被任何规则占用**的带命名空间名字：`cordis_run`/`terminal_open` 属
    # dsh 预置规则，会先被"别家已占用"那道滤掉，测不到形状这一层（本条初版就踩了）。
    rule = suggest_rule("newagent", tools=["bash", "read", "newagent_run", "newagent_open"])
    assert rule["tool_signatures"] == ["newagent_open", "newagent_run"]
    assert rule["min_tool_match"] == 1


def test_suggest_skips_keyword_that_collides_with_existing_rules() -> None:
    """与别家关键词互为子串的建议会互相误命中，要跳过。"""
    from bladex_proxy.agent_rules import suggest_rule

    rule = suggest_rule("fake", system_excerpt="You are a Hermes agent built by Nous Research.")
    assert "system_prompt_keywords" not in rule


def test_suggest_only_uses_ua_when_it_is_the_agents_own_name() -> None:
    """SDK 名不是 agent 名——UA 是通用 SDK 时不建议 header 规则。"""
    from bladex_proxy.agent_rules import suggest_rule

    sdk = suggest_rule("pi", headers={"user-agent": "OpenAI/JS 6.40.0"})
    assert "header_patterns" not in sdk

    own = suggest_rule("dsh2", headers={"user-agent": "deepseek-harness/0.4.0 (+url)"})
    assert own["header_patterns"][0]["name"] == "user-agent"


def test_suggested_rule_is_directly_usable(tmp_path) -> None:
    """建议出来的 dict 必须能直接喂给写盘函数（否则用户点确认会报错）。"""
    from bladex_proxy.agent_rules import append_user_rule, suggest_rule

    rule = suggest_rule("pi", system_excerpt="You are Pi, a coding agent.",
                        tools=["pi_edit"], headers={"user-agent": "pi/1.0"})
    target = tmp_path / "agent_rules.toml"
    append_user_rule(rule, path=target)
    assert any(r.agent_id == "pi" for r in load_agent_rules(user_path=target))


# ── 规则可编辑（追加即更新）──────────────────────────────────────────────────


def test_appending_same_name_updates_the_rule(tmp_path) -> None:
    """🔴 追加同名规则 = 更新它，而不是被旧的挡住。

    这条是"dashboard 能改规则"的地基：写盘走追加（不重写文件、不吃掉用户注释），
    那么"改一条规则"就是再追加一条同名的。若保留**先**写的那条，用户改完会发现
    **改了没用**——旧的仍排在前面命中，是最难查的一类问题。
    """
    from bladex_proxy.agent_rules import append_user_rule

    target = tmp_path / "agent_rules.toml"
    append_user_rule({"name": "x", "agent_id": "old-name",
                      "system_prompt_keywords": ["v1"]}, path=target)
    append_user_rule({"name": "x", "agent_id": "new-name",
                      "system_prompt_keywords": ["v2"]}, path=target)

    rules = [r for r in load_agent_rules(user_path=target) if r.name == "x"]
    assert len(rules) == 1, "同名规则没有折叠成一条"
    assert rules[0].agent_id == "new-name"
    assert rules[0].system_prompt_keywords == ["v2"]
    # 两次内容都还在文件里（追加语义），只是加载时后者生效
    assert target.read_text(encoding="utf-8").count("[[rules]]") == 2


def test_renaming_a_rule_backed_agent_needs_the_rule_too(tmp_path) -> None:
    """🔴 改名必须连规则一起改，否则新流量还是旧名字。

    实测复现（2026-08-19）：把 `hermes` 改名，只写 `AGENT_CLAIM`（那只重映射历史
    数据），规则里 `agent_id` 仍是 `hermes` → 下一轮 Hermes 请求解析出来还是
    `hermes`，**改名对新流量完全不生效**，而且会一直重新产出旧 agent_id。

    所以 dashboard 的改名对话必须预填现有规则并把 agent_id 换成新名字。
    """
    from bladex_proxy.agent_rules import (
        append_user_rule,
        effective_rules_for,
        rule_as_dict,
    )
    from bladex_proxy.identity import _fingerprint_agent

    target = tmp_path / "agent_rules.toml"
    # 取现有 hermes 规则 → 改 agent_id → 追加（这正是 UI 该做的事）
    base = next(r for r in effective_rules_for("hermes") if not r.auxiliary)
    renamed = rule_as_dict(base)
    renamed["agent_id"] = "hermes-work"
    append_user_rule(renamed, path=target)

    rules = load_agent_rules(user_path=target)
    hit = None
    for r in rules:
        if any(k.lower() in "you are a hermes agent built by nous research."
               for k in r.system_prompt_keywords):
            hit = r
            break
    assert hit is not None and hit.agent_id == "hermes-work", (
        "改了规则之后新流量仍会解析成旧名字"
    )
    assert _fingerprint_agent  # 引用一下，说明这条最终服务于识别路径


# ── 误配置必须响亮失败 ──────────────────────────────────────────────────────


def test_unknown_field_raises() -> None:
    """打错键名而规则悄悄不生效，是最难查的一类问题。"""
    with pytest.raises(ValueError, match="未知字段"):
        parse_rules('[[rules]]\nagent_id = "x"\nsystem_prompt_keyword = ["typo"]\n')


def test_missing_agent_id_raises() -> None:
    with pytest.raises(ValueError, match="agent_id"):
        parse_rules('[[rules]]\nname = "x"\n')


def test_bad_regex_raises() -> None:
    with pytest.raises(ValueError, match="正则非法"):
        parse_rules(
            '[[rules]]\nagent_id = "x"\n'
            'header_patterns = [{ name = "user-agent", pattern = "([" }]\n'
        )


# ── 🔴 测试隔离：不许读 live 的用户规则 ─────────────────────────────────────


def test_tests_never_read_the_live_user_rules_file() -> None:
    """🔴 规则加载必须指向隔离路径，不得读真实部署的 `config/agent_rules.toml`。

    实测炸过（2026-08-19 gate）：Jason 在 live dashboard 存了一条 `Pi` 规则，
    三个用例当场变红——两个因为 `Pi` 突然"已存在"、一个因为建议的关键词与那条真
    规则冲突被滤掉。**测试结果随用户当天存了什么而变**，那就不是测试了。

    与 conftest 里 `BLADEX_ROCKSDB_PATH` 等路径隔离同一个理由，只是这一处当初漏接。
    """
    import os

    from bladex_proxy.agent_rules import _discover_user_rules_path

    path = _discover_user_rules_path()
    assert path is not None, "BLADEX_AGENT_RULES_PATH 未设置——conftest 的隔离没生效"
    assert os.environ.get("BLADEX_AGENT_RULES_PATH"), "隔离 env 缺失"
    # 不得落在仓库工作树里（那就是 live 的那份）
    repo_config = Path.cwd() / "config" / "agent_rules.toml"
    assert path.resolve() != repo_config.resolve(), (
        f"测试在读 live 规则文件 {path}；结果会随用户当天存了什么规则而变"
    )


# ── import 无副作用 ─────────────────────────────────────────────────────────


def test_import_does_not_read_disk() -> None:
    """🔴 加载必须惰性：模块级读盘的坑 2026-08-04 踩过（consolidator 污染 os.environ）。"""
    import importlib

    import bladex_proxy.agent_rules as mod

    reset_rules_cache()
    importlib.reload(mod)
    assert mod._cache is None, "import/reload 后不应已有缓存 —— 说明有模块级读盘"


# ── 端到端：dsh 现在被规则库认出来，不再落 unknown 桶 ────────────────────────


def test_dsh_now_recognized_by_preset_rules() -> None:
    headers = {
        "authorization": "Bearer sk-shared",
        "user-agent": "deepseek-harness/0.3.1 (+https://github.com/deepseek-ai/deepseek-harness)",
        "x-deepseek-harness-session-id": "sess-001",
    }
    req = ChatCompletionRequest(
        messages=[{"role": "user", "content": "这个目录里有多少文件？"}],
        tools=[{"function": {"name": "cordis_run"}}],
    )
    identity, _ = resolve_identity(headers, req)
    assert identity.agent_id == "dsh"


def test_opencode_recognised_by_ua_alone() -> None:
    """🔴 MQ-A68：opencode 三个维度都专有，**任一维单独在场都要认得出**。

    2026-09-11 首次连接时它落 `unknown-e0ca7fa5`，而 `agent_trigger` 是
    `bucket:ua:opencode` —— **兜底分桶已经从 UA 里读出了名字，规则库里却没有**。
    "看得见却认不出"是接入域最贵的一类：`agent_source=fallback` 在日志里
    不刺眼，记忆与账本却已经挂到一个 unknown 命名空间上了。

    live wire 原文（Hub `local/unknown-e0ca7fa5/1789103464821-0`）：
        user-agent: opencode/1.18.30 ai-sdk/provider-utils/4.0.23 runtime/node.js/24
    """
    agent_id, trigger, _ = _fingerprint_agent(
        [], None, {"user-agent": "opencode/1.18.30 ai-sdk/provider-utils/4.0.23 "
                                 "runtime/node.js/24"})
    assert agent_id == "opencode", f"UA 单独在场应识别，实得 {agent_id} ({trigger})"


def test_opencode_rule_claims_no_generic_tool_names() -> None:
    """🔴 约束 ③ 的守卫：opencode 用的是 `bash`/`read`/`edit`/`glob`/`grep`
    这类**通用工具名**（live 实测），收进 `tool_signatures` 会把别的 agent
    误判成 opencode —— 那正是 MQ-A6「min_tool_match=1 把 dsh 误判成 hermes」
    的形态。它的 UA 与 system 自报名已经足够专有，工具维留空。
    """
    from bladex_proxy.agent_rules import load_agent_rules
    rule = next(r for r in load_agent_rules() if r.agent_id == "opencode")
    assert rule.tool_signatures == [], \
        "opencode 的工具名是通用名，不得进 tool_signatures"
