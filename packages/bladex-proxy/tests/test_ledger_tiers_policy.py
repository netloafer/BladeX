"""账本档位策略参数化（Jason 2026-08-26 拍板）。

# 为什么要有这个配置

`tier` 是"这个模型调工具可靠吗"的信号，被拿来当"这件事需不需要账本"的门。
实测代价：**五个 agent 里两个完全拿不到账本**——Pi 21/21 轮判 weak（结构性不可达），
dsh 在 agent 排除名单里。而"零改造接入任何 agent"是北极星。

在测出"weak 档到底可不可靠"之前，正确的做法不是替用户拍板，而是**把选择权交给他**
（刚性原则 9：配置文件里的静态策略压过一切推断）。

# 一个机制，三个具名预设

`auto` / `all` / 显式档位集合走的是**同一套代码**，不是三条路径——
auto 只是 `AUTO_LEDGER_TIERS` 这个有名字的默认值。
"""

from __future__ import annotations

import pytest
from bladex_proxy.toolface import (
    AUTO_LEDGER_TIERS,
    LEDGER_TIER_NAMES,
    inject_tools,
    resolve_ledger_tiers,
)


class TestResolve:
    @pytest.mark.parametrize(("raw", "expect"), [
        ("auto", {"medium", "strong"}),
        ("all", {"weak", "medium", "strong"}),
        ("none", set()),
        ("strong", {"strong"}),
        ("medium,strong", {"medium", "strong"}),
        ("  Weak , Strong ", {"weak", "strong"}),      # 大小写与空格不该绊倒用户
    ])
    def test_presets_and_explicit_sets(self, raw, expect):
        assert resolve_ledger_tiers(raw) == expect

    def test_unset_defaults_to_auto(self, monkeypatch):
        monkeypatch.delenv("BLADEX_LEDGER_TIERS", raising=False)
        assert resolve_ledger_tiers() == frozenset(AUTO_LEDGER_TIERS)

    def test_explicit_list_can_equal_auto(self):
        """auto 不是特例，是具名默认值 —— 它必须能被等价地显式写出来。"""
        assert resolve_ledger_tiers(",".join(AUTO_LEDGER_TIERS)) == resolve_ledger_tiers("auto")

    def test_bad_tier_fails_loudly(self):
        """🔴 误配置不许静默降级成"什么都不注"——那是最难查的一类。"""
        with pytest.raises(ValueError) as ei:
            resolve_ledger_tiers("strongg")
        assert "strongg" in str(ei.value) and "auto/all/none" in str(ei.value)


class TestSingleSourceOfTruth:
    # 🔴 `test_no_toolface_tiers_is_derived` 已删除（2026-08-30 拆 weak 门）：
    # `NO_TOOLFACE_TIERS` 这个常量随那道门一起删了——"不注工具面的档位"这个
    # 概念不再存在（工具面对所有档位注入）。删测试是因为**被测对象没了**，
    # 不是因为它红了；`AUTO_LEDGER_TIERS` 的单一真相源地位由下面两条继续守。

    def test_tier_names_match_the_router(self):
        """档位闭集要与路由那边同源，别在别处另抄一份字面量。"""
        from bladex_core.routing import _TIER_ORDER
        assert set(LEDGER_TIER_NAMES) == set(_TIER_ORDER)

    def test_env_example_documents_the_knob(self):
        """auto 的语义会随测量结果变 ⇒ 文档与代码必须对得上（flags.py 同款纪律）。"""
        from pathlib import Path
        root = Path(__file__).resolve().parents[3]
        txt = (root / "config" / ".env.example").read_text(encoding="utf-8")
        assert "BLADEX_LEDGER_TIERS" in txt
        # 依据从"假设"变成"读数"之后，这条断言也要跟着变 —— 否则它会钉住一句
        # 已经不成立的话（2026-08-26：实测 43% 调用率 / 100% 参数正确率，假设已证伪；
        # auto 出于 0.1.0 跑通优先仍保持保守，那是**拍板**不是**未验证**）。
        assert "调用率" in txt and "参数正确率" in txt, \
            "auto 的依据没写明实测读数 ⇒ 下一个人会重跑同一个实验"
        assert "0.2.0" in txt, "没写明何时重议 ⇒ 保守默认会变成永久默认"


class TestDrivesLedgerFamilyOnly:
    """🔴 这个配置驱动的是**账本族**（2026-08-30 按工具族拆之后的语义）。

    改动前它一刀切管整个工具面，于是 weak 档连 `bladex_memory_search` 都拿不到；
    砍掉 Memory Index 主动注入后那等于"弱档彻底没有记忆"。现在：
      · 记忆族 —— 无条件注入，不看这个旋钮；
      · 账本族 —— 仍受它管（含 `none` = 整个账本面关掉），默认 auto 不变。
    "有令无器"（MQ-L7）由同一个谓词 `ledger_tools_allowed` 保证。"""

    def _names(self, tools):
        return [(t.get("function") or t).get("name", "") for t in (tools or [])]

    @pytest.mark.parametrize("preset", ["auto", "all", "none", "strong"])
    @pytest.mark.parametrize("tier", LEDGER_TIER_NAMES)
    def test_memory_tool_always_injected(self, monkeypatch, preset, tier):
        """🔴 记忆族**无论怎么配、无论哪个档位**都在。

        全预设 × 全档位地跑，是因为这条是本次拍板的实质保证：
        只测 auto/weak 一格，回归可以从别的格子溜走。
        """
        monkeypatch.setenv("BLADEX_LEDGER_TIERS", preset)
        tools, injected = inject_tools(None, agent_id="Pi", auxiliary=False,
                                       model_tier=tier)
        assert injected, f"preset={preset} tier={tier}：工具面整个没注"
        assert "bladex_memory_search" in self._names(tools)

    def test_ledger_family_still_excluded_when_only_strong_configured(self, monkeypatch):
        """账本族仍按配置排除——旧断言 `not injected` 换成"账本工具不在"，
        因为记忆工具还在（整体 injected 仍为 True）。"""
        monkeypatch.setenv("BLADEX_LEDGER_TIERS", "strong")
        tools, injected = inject_tools(None, agent_id="hermes", auxiliary=False,
                                       model_tier="medium")
        assert injected
        names = self._names(tools)
        assert "bladex_memory_search" in names
        assert not [n for n in names if n.startswith("bladex_ledger_")]

    def test_none_turns_the_ledger_family_off(self, monkeypatch):
        """`none` 的语义保住了：账本面整个关掉。

        🔴 这一条是本次改动中**差点丢掉的东西**——一版实现让工具面完全不看
        旋钮，用户配 `none` 也照样拿到三个账本工具（静默旁路已有配置，
        刚性原则 9 的反面）。留着它守。
        """
        monkeypatch.setenv("BLADEX_LEDGER_TIERS", "none")
        for tier in LEDGER_TIER_NAMES:
            tools, _ = inject_tools(None, agent_id="hermes", auxiliary=False,
                                    model_tier=tier)
            names = self._names(tools)
            assert not [n for n in names if n.startswith("bladex_ledger_")], \
                f"none 档下 {tier} 仍注了账本工具"
            assert "bladex_memory_search" in names, \
                f"none 只该关账本面，不该连记忆一起关（tier={tier}）"


class TestOrthogonalGates:
    """配置只管 tier 这一轴，另外两道门不受它影响。"""

    def test_aux_still_excluded_under_all(self, monkeypatch):
        """aux 轮**整个工具面**都不注（含记忆族）——它是 agent 内部子调用，
        给了是注意力税 + 误调用风险。族拆分不改这一条。"""
        monkeypatch.setenv("BLADEX_LEDGER_TIERS", "all")
        _t, injected = inject_tools(None, agent_id="claude-code", auxiliary=True,
                                    model_tier="strong")
        assert not injected

    def test_agent_exclusion_beats_tiers_all(self, monkeypatch):
        """🔴 **agent 排除与档位策略正交，且 agent 排除压过 `TIERS=all`**。

        配成 all 也拿不到 —— 排除是"这个 agent 的机制接不住工具面"，
        不是"这个档位不配"，两者不在一个维度上。

        🔴 **2026-09-11 改写（MQ-A66）**：原名 `test_dsh_still_excluded_under_all`，
        写死 `agent_id="dsh"`，理由是「工具只能从 run_code 程序内调」。
        dsh 升级后 27 个工具全部直接声明、`run_code` 已不存在 ⇒ 依据失效、
        名单清空，这条跟着红。守的正交性一个字没变，**过期的是绑定的那个 agent**。
        改用合成 agent + monkeypatch 名单。
        """
        import bladex_proxy.toolface as _tf
        monkeypatch.setenv("BLADEX_LEDGER_TIERS", "all")
        monkeypatch.setattr(_tf, "NO_TOOLFACE_AGENT_BASES", ("excluded-probe",))
        _t, injected = inject_tools(None, agent_id="excluded-probe", auxiliary=False,
                                    model_tier="strong")
        assert not injected  # 按 agent 排除 = 整个工具面不注，两族都没有

    def test_missing_tier_is_treated_as_inside(self, monkeypatch):
        """拿不到档位（老调用方/未路由）时不能静默关掉账本面 ——
        "取不到值"不等于"用户选择了不要"。"""
        monkeypatch.setenv("BLADEX_LEDGER_TIERS", "auto")
        _t, injected = inject_tools(None, agent_id="hermes", auxiliary=False,
                                    model_tier="")
        assert injected


class TestContentionNeedsConcurrency:
    """scope 争用告警必须带"**并发**"（2026-08-26 live 误报）。

    首版判据是"换了 session"，结果 Pi 的两轮**先后**会话
    （一轮 `tw:` 时间窗兜底、一轮 `fp:` 指纹）也触发了告警——
    同一个 agent 连续工作换个 session_id 是常态，不是争用。

    误报的代价不是吵：**真争用发生时会被淹在噪声里**。
    """

    def _table(self):
        from bladex_core.ledger_runtime import ActivationTable
        return ActivationTable(debounce_turns=0)

    def test_first_switch_has_no_age(self):
        from bladex_core.ledger_runtime import activation_scope
        v = self._table().request_switch(activation_scope("Pi"), "a",
                                         agent_id="Pi", session_id="s1", turn_index=0)
        assert v.prev_session == "" and v.prev_age_s == -1.0

    def test_back_to_back_switch_is_contention(self):
        from bladex_core.ledger_runtime import activation_scope
        t, sc = self._table(), activation_scope("Pi")
        t.request_switch(sc, "a", agent_id="Pi", session_id="s1", turn_index=0)
        v = t.request_switch(sc, "b", agent_id="Pi", session_id="s2", turn_index=1)
        assert v.prev_session == "s1" and 0 <= v.prev_age_s <= 300

    def test_much_later_switch_is_not_contention(self):
        """先后会话（而非并发）不该报警 —— 正是 live 误报的那个形态。"""
        import time

        from bladex_core.ledger_runtime import activation_scope
        t, sc = self._table(), activation_scope("Pi")
        t.request_switch(sc, "a", agent_id="Pi", session_id="s1", turn_index=0)
        t._last_switch_at[sc] = time.time() - 1800
        v = t.request_switch(sc, "b", agent_id="Pi", session_id="s2", turn_index=1)
        assert v.prev_age_s > 300

    def test_agency_applies_the_time_window(self):
        import inspect

        from bladex_proxy import agency
        src = inspect.getsource(agency.AgencyRuntime._h_ledger_switch)
        assert "prev_age_s" in src and "_contention_window_s" in src

    def test_wall_clock_not_turn_index(self):
        """🔴 用墙钟不用 turn_index：换 scope 后 turn_index 跨会话不单调
        （它是 `len(messages)//2`），拿它算"多久以前"会得出负数。"""
        import inspect

        from bladex_core import ledger_runtime
        src = inspect.getsource(ledger_runtime.ActivationTable.request_switch)
        assert "time.time()" in src
