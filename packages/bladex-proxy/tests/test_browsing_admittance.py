"""浏览面准入判据①（`flash_daemon.browsing_admittance`）。

2026-08-29 拍板（v2，撤销当日早间的 base 聚合 v1）：**有独立 agent 声明才算 agent** ——
profile 变体（hermes:default/accept 各有 profile 与配置）= 独立 agent，分开展示；
子代理派生 id（codex:guardian）内容不进 Index，也不进 flash；
只有 auxiliary 声明的"agent"不算。记忆命名空间仍按完整 agent_id 隔离不动。

批 F4（2026-09-23）从 `test_flash_tree_materialize.py` 迁来：目录树与名册删除后，
这把谓词的唯一消费方是行为画像 `hub_behavior_source(eligible=...)`（`flash_daemon.main`），
谓词本身一字未改。原文件里依赖 `hub_tree_source` 的判据②用例随树一起删除。
"""

from __future__ import annotations


def _pred(bindings=()):
    from bladex_proxy.agent_rules import AgentFingerprintRule as R
    from bladex_proxy.flash_daemon import browsing_admittance
    rules = [
        R(agent_id="hermes", auxiliary=True,
          system_prompt_keywords=["checkpoint"]),   # aux 规则也挂 hermes
        R(agent_id="hermes", profile_aware=True),
        R(agent_id="codex", subagent_headers=["x-openai-subagent"]),
        R(agent_id="pi"),
        R(agent_id="titler", auxiliary=True),       # 只有 aux 声明的"agent"
    ]
    return browsing_admittance(rules, list(bindings))


def test_predicate_matrix():
    ok = _pred()
    assert ok("hermes")                    # 非 aux base 声明存在
    assert ok("hermes:default") and ok("hermes:accept"), \
        "profile 变体 = 独立 agent（各有 profile 与配置文件）"
    assert ok("codex")
    assert not ok("codex:guardian"), "子代理派生 id：内容不进 Index，不进 flash"
    assert not ok("titler"), "只有 auxiliary 声明 ≠ 独立 agent 声明"
    assert not ok("unknown-9d9bdd3e") and not ok("")


def test_explicit_header_declaration_admits():
    ok = _pred(bindings=[{"agent_id": "my-bot", "trigger": "header:X-Agent-ID"}])
    assert ok("my-bot"), "X-Agent-ID 显式声明 = 用户亲手起的名"
    assert not _pred()("my-bot")
