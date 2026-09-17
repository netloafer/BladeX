"""MQ-L58：routing.toml 的策略引用了不存在的模型名 ⇒ **拒绝启动**（2026-09-08）。

## 立卡经过

批 G 剧本③ 第一次实跑，`[strategies.agent.map]` 写着
`"codex" = ["anthropic/glm-5.3-flash"]`，而 `[[models]]` 表里当时没有这个名字。

启动时 `_validate_model_refs` **确实打了**结构化告警：

    [warning] routing_agent_unknown_model agent=codex
              hint='dropped from agent policy' missing=['anthropic/glm-5.3-flash']

一条不缺 —— 带 agent、带 missing、带 hint。**然后请求照常成功**：静态策略整个不命中，
路由退回 sticky ⇒ `doubao-seed-2.0-lite`（weak）⇒ 账本族被 gate ⇒ Codex 全程拿不到
账本块。现象是"跑起来了、只是效果不对"，而这条告警在 **10 个以上历史日志里连打了好几天**，
没人看。

⇒ **缺的从来不是告警，是严重性。** 那句 `hint='dropped from agent policy'` 还把静默旁路
描述成了正常行为（原则 12：把缺陷写进日志 ≠ 处理了缺陷；"警告性文案是缺陷的气味"）。

## 为什么是拒启动

配置错误**自包含、加载期即可判定** ⇒ 加载期失败（刚性原则：误配置失败要响亮）。
同仓先例：`IdentityRegistry.from_toml` 对 dangling ref 就是拒启动。同类问题两种处置，
本身就是"同一条判据只用在一半上"。
"""

from __future__ import annotations

import textwrap

import pytest
from bladex_proxy.routing_config import RoutingConfig, RoutingConfigError

_MODELS = """
[[models]]
name = "openai/real-a"
api_base = "https://example.invalid/v1"
capabilities = ["text"]
tier = "medium"

[[models]]
name = "openai/real-b"
api_base = "https://example.invalid/v1"
capabilities = ["text"]
tier = "weak"
"""


def _write(tmp_path, body: str):
    p = tmp_path / "routing.toml"
    p.write_text(textwrap.dedent(_MODELS + body), encoding="utf-8")
    return str(p)


# ── ① 三层静态策略逐层必炸（此前只查了 agent 一层）───────────────────────────


@pytest.mark.parametrize("where, body", [
    ("agent", '[strategies.agent.map]\n"codex" = ["openai/typo-here"]\n'),
    ("principal", '[strategies.principal.map]\n"p1" = ["openai/typo-here"]\n'),
    ("team", '[strategies.team.map]\n"t1" = ["openai/typo-here"]\n'),
])
def test_dangling_ref_in_any_static_strategy_refuses_startup(tmp_path, where, body):
    """team > principal > agent 三层是同一类判据，一个都不许漏。

    修前只查 agent 一层 —— 同一条判据只用在一部分上，是本仓反复付学费的形态
    （MQ-L53 第二类报错、MQ-A52 双侧不对称都是它）。
    """
    with pytest.raises(RoutingConfigError) as e:
        RoutingConfig.from_toml(_write(tmp_path, body))
    msg = str(e.value)
    assert "openai/typo-here" in msg, "报错必须点名那个写错的模型"
    assert where in msg, f"报错必须点名是哪一层策略（{where}）"


def test_dangling_ref_in_upstream_filter_and_multimodal_also_refuses(tmp_path):
    for body in (
        '[upstream]\nmodels = ["openai/typo-here"]\n',
        '[strategies.filter.candidates]\nweak = ["openai/typo-here"]\n',
        '[strategies.multimodal.map]\nvision = ["openai/typo-here"]\n',
    ):
        with pytest.raises(RoutingConfigError):
            RoutingConfig.from_toml(_write(tmp_path, body))


# ── ② 报错要能一遍改完 ──────────────────────────────────────────────────────


def test_every_dangling_ref_is_listed_at_once(tmp_path):
    """不是撞到第一条就抛：三处错要一次全列出来，用户一遍改完。

    分批报错 = 用户改一次重启一次，而每次重启都要等 embedding/Hub 起来。
    """
    cfg = ('[strategies.agent.map]\n"a1" = ["openai/typo-1"]\n'
           '"a2" = ["openai/real-a", "openai/typo-2"]\n'
           '[strategies.multimodal.map]\nvision = ["openai/typo-3"]\n')
    with pytest.raises(RoutingConfigError) as e:
        RoutingConfig.from_toml(_write(tmp_path, cfg))
    msg = str(e.value)
    for n in ("openai/typo-1", "openai/typo-2", "openai/typo-3"):
        assert n in msg, f"{n} 没被列出来 —— 用户要重启第二次才能发现它"


def test_the_error_lists_the_known_names_to_fix_by(tmp_path):
    """报错要指向配置（刚性原则 9 第三条）：把已登记的名字列出来，用户才知道该写什么。

    live 病例正是一个**形近**错误：配的是 `anthropic/glm-5.3-flash`，
    表里有 `anthropic/deepseek-v4-flash` 和 `openai/glm-5.3-flash` —— 列出来一眼可见。
    """
    with pytest.raises(RoutingConfigError) as e:
        RoutingConfig.from_toml(
            _write(tmp_path, '[strategies.agent.map]\n"a1" = ["openai/typo-1"]\n'))
    msg = str(e.value)
    assert "openai/real-a" in msg and "openai/real-b" in msg


# ── ③ 阳性对照：名字对得上时不许炸 ──────────────────────────────────────────


def test_valid_refs_load_fine(tmp_path):
    """没有这一条，① 全绿也可能只是因为"这个函数总是抛"。"""
    cfg = RoutingConfig.from_toml(_write(tmp_path, (
        '[upstream]\nmodels = ["openai/real-a", "openai/real-b"]\n'
        '[strategies.agent.map]\n"codex" = ["openai/real-a"]\n'
        '[strategies.filter.candidates]\nweak = ["openai/real-b"]\n')))
    assert {m.name for m in cfg.models} == {"openai/real-a", "openai/real-b"}
    assert cfg.strategies.agent.map["codex"] == ["openai/real-a"]


def test_empty_strategy_maps_load_fine(tmp_path):
    """空策略（现状默认）不受影响 —— 拒启动只针对**写了但写错**。"""
    assert RoutingConfig.from_toml(_write(tmp_path, "")).models


def test_the_repo_config_itself_loads(tmp_path):
    """仓库自带的 `config/routing.toml` 必须能过这一关。

    🔴 这一条兼作**回归闸**：以后谁往 routing.toml 里写错模型名，这里先红，
    而不是等到某个 agent 在 live 上被静默降级好几天才被发现（MQ-L58 的原始形态）。
    """
    import os
    p = os.path.join(os.path.dirname(__file__), "..", "..", "config", "routing.toml")
    if not os.path.isfile(p):
        pytest.skip("仓库配置不在预期位置")
    assert RoutingConfig.from_toml(p).models
