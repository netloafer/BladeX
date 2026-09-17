"""F-B4（MQ-L49 ④）：批量形态的**提示引导**——三处模型面文本都得说这件事。

F-B1 把能力做出来了，本卡是"告诉模型该用"的那半。**有器无令与有令无器一样白搭**
（MQ-L7 的镜像：那次是注入块写着"call bladex_ledger_switch"而 tools 数组里没有它，
这次反过来——`entries` 参数在 schema 里，而没有任何一处告诉模型为什么该用它）。

三处各自的位置不可互相替代（与 `test_ledger_cleanup_rule_stated_in_both_places`
同一立论）：

- `bladex_ledger_update` 的顶层 description —— **调用现场**的契约；
- `_FIRST_STEP_WITH_LEDGER` —— 每轮的提醒，在模型还没决定调不调工具的时候；
- `TOOLS.md` / `AGENT.md` —— 稳定层，首轮冻结进前缀、享 prompt cache，讲得起
  "为什么"（一次调用 = 一个完整模型来回）。

判据取**语义关键词**不取整句 —— 取整句等于禁止 prompt 调优，而措辞本来就要随
读数调（MQ-L49 的判据是一周后的 `n_edits≥2` 占比，不是这里的字面）。
"""

from __future__ import annotations

import pathlib

import pytest

from bladex_proxy.toolface import TOOL_SCHEMAS

_ASSETS = pathlib.Path(__file__).resolve().parents[1] / "bladex_proxy" / "assets" / "system"
_CONFIG = pathlib.Path(__file__).resolve().parents[3] / "config" / "system"


def _text(name: str) -> str:
    return (_ASSETS / name).read_text(encoding="utf-8")


class TestBatchIsAskedForInAllThreePlaces:
    def test_tool_description_says_it(self):
        d = next(t["function"]["description"] for t in TOOL_SCHEMAS
                 if t["function"]["name"] == "bladex_ledger_update").lower()
        assert "entries" in d
        assert "batch" in d or "one call" in d

    def test_first_step_instruction_says_ONE_call(self):
        """🔴 原文是 "in the same turn" —— 同一轮**两次**调用满足它，而两次调用
        正是 MQ-L49 那条成本曲线（一次调用 = 一个完整模型来回）。"""
        from bladex_proxy.agency import _FIRST_STEP_WITH_LEDGER

        low = _FIRST_STEP_WITH_LEDGER.lower()
        assert "one bladex_ledger_update call" in low, \
            "首步指令没点名「一次调用」——「同一轮」不排除调两次"
        assert "entries" in low, "没点名 entries 参数，模型不知道用什么写法"

    def test_tools_md_cost_section_explains_why(self):
        cost = _text("TOOLS.md").split("## Cost", 1)[1].lower()
        assert "entries" in cost, "Cost 段没点名 entries"
        assert "match" in cost and "index" in cost, "没说优先 match 而不是 index"
        assert "same turn" in cost, \
            "没说「与你自己的工具调用同一轮发出」——独占一轮是白花一个模型来回"

    def test_agent_md_item_two_says_it_too(self):
        low = _text("AGENT.md").lower()
        assert "entries" in low, "AGENT.md 第 2 条没提批量"

    def test_cleanup_rule_survived_the_rewrite(self):
        """回归面：改首步指令时最容易顺手改掉的就是 MQ-L29 那条规则。
        （`test_toolface::test_ledger_cleanup_rule_stated_in_both_places` 是正主，
        这里重述一遍是因为本卡直接动了那段字符串。）"""
        from bladex_proxy.agency import _FIRST_STEP_WITH_LEDGER

        low = _FIRST_STEP_WITH_LEDGER.lower()
        for word in ("verified", "remove", "next", "open"):
            assert word in low, f"首步指令丢了 {word}"


class TestTwoCopiesStayVerbatim:
    """既有守卫 `test_model_facing_assets` 也管这件事；本卡两份文件都改了，
    在自己的测试里再钉一次——漏 copy 一份的后果是 pip 装出来的与仓库跑的不一样。"""

    @pytest.mark.parametrize("name", ["TOOLS.md", "AGENT.md"])
    def test_config_and_package_copies_are_identical(self, name: str):
        assert (_CONFIG / name).read_text(encoding="utf-8") == _text(name), \
            f"{name} 两副本分叉了：改一处必须同步另一处"


class TestErrorTableCoversTheNewShapes:
    """模型会撞上 F-B1 的新回文；错误表的作用就是让它知道下一步做什么。"""

    def test_batch_all_or_nothing_is_documented(self):
        low = _text("TOOLS.md").lower()
        assert "nothing was written" in low, \
            "批量整批失败的回文没进错误表 ⇒ 模型会只重发失败那一条，另几条就丢了"
        assert "whole" in low, "没说清要重发整批"

    def test_ambiguous_match_is_documented(self):
        assert "matches 2 entries" in _text("TOOLS.md"), \
            "match 歧义的回文没进错误表"
