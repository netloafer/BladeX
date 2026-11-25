"""批 AR（MQ-A110）：认领建议与规则匹配剔除 BladeX 自己注入的 `bladex_*` 工具。"""

from __future__ import annotations

import structlog.testing
from bladex_proxy import identity as idm
from bladex_proxy.agent_rules import (
    load_agent_rules,
    parse_rules,
    reset_rules_cache,
    suggest_rule,
)

_OWN = ["bladex_ledger_close", "bladex_memory_search"]


def _tool(name: str) -> dict:
    return {"type": "function", "function": {"name": name, "parameters": {}}}


def test_suggest_drops_bladex_keeps_real_tools() -> None:
    rule = suggest_rule("X", tools=_OWN + ["cordis_run", "skill-manage"], existing=[])
    assert rule["tool_signatures"] == ["cordis_run", "skill-manage"]


def test_suggest_only_bladex_gives_no_tool_signatures() -> None:
    rule = suggest_rule("X", tools=_OWN, existing=[])
    assert "tool_signatures" not in rule
    assert "min_tool_match" not in rule


def test_handwritten_bladex_signature_never_matches_and_warns_once(tmp_path) -> None:
    p = tmp_path / "agent_rules.toml"
    p.write_text('[[rules]]\nname = "zz"\nagent_id = "zz"\n'
                 'tool_signatures = ["bladex_memory_search"]\nmin_tool_match = 1\n',
                 encoding="utf-8")
    with structlog.testing.capture_logs() as logs:
        rules = load_agent_rules(p)
    warns = [e for e in logs if e["event"] == "agent_rule_tool_signature_is_bladex_tool"]
    assert len(warns) == 1 and warns[0]["rule"] == "zz"
    assert warns[0]["tools"] == ["bladex_memory_search"]
    user_rule = next(r for r in rules if r.name == "zz")
    reset_rules_cache()

    orig = idm._rules
    idm._rules = lambda: [user_rule]  # type: ignore[assignment]
    try:
        agent, _, _ = idm._fingerprint_agent(
            [{"role": "user", "content": "hi"}], [_tool("bladex_memory_search")])
        assert agent is None
    finally:
        idm._rules = orig  # type: ignore[assignment]



def test_preset_rules_do_not_carry_bladex_tools() -> None:
    from bladex_proxy.agent_rules import PRESET_RULES_PATH
    from bladex_proxy.toolface import is_bladex_tool
    preset = parse_rules(PRESET_RULES_PATH.read_text(encoding="utf-8"))
    assert not [t for r in preset for t in r.tool_signatures if is_bladex_tool(t)]
