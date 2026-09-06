"""MQ-L21 · agent 自带零工具 ⇒ 不给账本面（ADR-0032 §4.3 边界）。

🔴 病例（2026-08-27，Jason 肉眼发现，排查三小时）：用户只发了**一句话**，
BladeX 却收到**四个 prompt** —— 另外两个是 Codex 客户端自己的附属请求
（IDE 建议生成 / UI 标题生成），用户看不见它们。

那两类请求**原本零工具**：

    gpt-5.6-terra（建议生成）  工具 4 = bladex 4 / 自带 0
    gpt-5.6-luna （标题生成）  工具 4 = bladex 4 / 自带 0
    gpt-5.5      （用户任务）  工具 17 = bladex 4 / 自带 13

是我们塞给它 4 个 `bladex_*` + "FIRST STEP, every turn: call
bladex_ledger_switch"，**它才有了建账本的能力**。结果标题生成那本占住激活位，
用户真实任务再进来时对照 Goal 发现"不是同一件事"，只能新开一本 ——
**模型判断全对，错在我们给了它不该给的东西。**

判据是结构性的（与 `is_isolated_subcall` 同构）：一个不带任何工具的请求，
要的是**一段文本**，不是"完成一件任务"。
"""

from __future__ import annotations

import pytest

_AGENT_TOOL = {"type": "function", "function": {"name": "exec_command"}}
_BLADEX_TOOL = {"type": "function", "function": {"name": "bladex_ledger_switch"}}
_ANTHROPIC_TOOL = {"name": "Bash", "input_schema": {}}


class TestBringsNoTools:
    @pytest.fixture(autouse=True)
    def _on(self, monkeypatch):
        monkeypatch.setenv("BLADEX_LEDGER_REQUIRE_TOOLS", "1")

    def test_empty_and_none_are_no_tools(self):
        from bladex_proxy.identity import brings_no_tools
        assert brings_no_tools(None) is True
        assert brings_no_tools([]) is True

    def test_only_bladex_tools_still_counts_as_no_tools(self):
        """🔴 病例本体：Codex 的标题生成请求最终带 4 个工具，全是我们注入的。

        拿 `bladex_*` 当"agent 带了工具"的证据，判据就会自我满足。
        """
        from bladex_proxy.identity import brings_no_tools
        assert brings_no_tools([_BLADEX_TOOL] * 4) is True

    def test_agent_own_tool_disables_the_gate(self):
        from bladex_proxy.identity import brings_no_tools
        assert brings_no_tools([_AGENT_TOOL]) is False
        assert brings_no_tools([_BLADEX_TOOL, _AGENT_TOOL]) is False

    def test_anthropic_flat_tool_shape(self):
        """messages 协议的工具是扁平 `{name, input_schema}`，不是嵌套 function。"""
        from bladex_proxy.identity import brings_no_tools
        assert brings_no_tools([_ANTHROPIC_TOOL]) is False

    def test_flag_off_is_a_full_rollback(self, monkeypatch):
        from bladex_proxy.identity import brings_no_tools
        monkeypatch.setenv("BLADEX_LEDGER_REQUIRE_TOOLS", "0")
        assert brings_no_tools([]) is False
        assert brings_no_tools([_BLADEX_TOOL]) is False


class TestWiredIntoEveryEndpoint:
    """账本域 aux 判据：单实现 + 三端点全接线。

    MQ-L20 的教训曾是"六处判据改一处漏五处"；V-A1 T2 编排函数化（2026-08-28）
    把六份副本收成 `_apply_agency_surfaces` 里的**一份**，本守卫随之升级：
    ① 判据只许存在这一份（防止有人在端点里再抄一份、重新分叉）；
    ② 三个端点都必须经它走（防止某端点绕开 = 机制只在部分协议生效）。
    取源按路径读文本（MQ-V9：inspect.getsource 会静默取错函数）。
    """

    @staticmethod
    def _src() -> str:
        from pathlib import Path
        return (Path(__file__).resolve().parents[1]
                / "bladex_proxy" / "server.py").read_text(encoding="utf-8")

    def test_single_gate_implementation(self):
        src = self._src()
        assert src.count("brings_no_tools(own_tools)") == 1, (
            "账本域 aux 判据必须只有 _apply_agency_surfaces 里那一份"
            "（再抄副本 = 回到 MQ-L20 改一处漏五处的形态）")
        assert src.count("brings_no_tools(_own_tools)") == 0, "旧六副本形态复活了"

    def test_all_three_endpoints_route_through_surfaces(self):
        src = self._src()
        # 只数**赋值形态的调用点**（`= _apply_agency_surfaces(`）——裸数函数名
        # 会把 def 行也算进去（写本断言时第一版就栽在这）。
        n = src.count("= _apply_agency_surfaces(")
        assert n == 3, (
            f"_apply_agency_surfaces 调用点 {n} 处，应为 3（chat/anthropic/responses）"
            "——少一处就有端点绕开账本域判据")

    def test_own_tools_captured_before_augment(self):
        """🔴 必须在 `augment_tools` **之前**取值——它会就地覆盖工具面。

        单实现后此序只需在 `_apply_agency_surfaces` 内部成立一次。
        """
        src = self._src().split("\n")
        cap = [i for i, l in enumerate(src) if "own_tools = list(tools_in or [])" in l]
        aug = [i for i, l in enumerate(src) if "agency.augment_tools(" in l]
        assert len(cap) == 1 and len(aug) == 1, (len(cap), len(aug))
        assert cap[0] < aug[0], "own_tools 取值晚于 augment_tools —— 拿到的是增补后形态"


class TestMachineTextAlarm:
    """辅助方案：让"内部调用建了账本"可被 grep（MQ-L17 的直接应用）。"""

    @pytest.mark.parametrize("text,expect", [
        ("You are a helpful assistant. You will be presented with a user prompt",
         "assistant_persona"),
        ("# Overview\n\nGenerate 0 to 3 hyperpersonalized suggestions", "md_overview"),
        ("You are an agent based on GPT-5.", "agent_persona"),
    ])
    def test_real_cases_are_flagged(self, text, expect):
        from bladex_proxy.agency import _machine_text_mark
        assert _machine_text_mark(text) == expect

    def test_normal_user_text_is_not_flagged(self):
        from bladex_proxy.agency import _machine_text_mark
        for t in ("按 docs/planning/task-codex-mcp-ledger-tools-20260827.md 任务卡，执行开发任务。",
                  "帮我把路由的回归问题修掉",
                  "小黑，帮我分析一下苹果这次新发的 Mac mini"):
            assert _machine_text_mark(t) == ""

    def test_mention_deep_in_the_text_is_not_flagged(self):
        """🔴 位置门，与 MQ-A19 同理：**提到**它的真实用户消息不该被标记。

        自指陷阱刚咬过一次（CLAUDE.md 里写着 `<transcript>`，读它的 agent
        被判成内部调用）。这条守住同型不再犯。
        """
        from bladex_proxy.agency import _machine_text_mark
        text = "帮我看看这段提示词。" + "背景说明。" * 60 + "You are a helpful assistant."
        assert text.index("You are a helpful") > 200
        assert _machine_text_mark(text) == ""

    def test_alarm_does_not_block_creation(self):
        """只告警不阻止——判据是启发式的，误伤的代价比漏报大。"""
        import inspect
        from bladex_proxy.agency import AgencyRuntime
        src = inspect.getsource(AgencyRuntime._h_ledger_switch)
        i = src.index("agency_ledger_created_from_machine_text")
        after = src[i:i + 400]
        assert "return" not in after.split("\n")[0]
        assert "logger.warning" in src[max(0, i - 120):i]
