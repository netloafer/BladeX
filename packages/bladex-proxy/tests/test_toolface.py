"""V-P1 验收剧本：注入 gating 矩阵 / 追加不重排 / 命名空间红线 / 分发容错。"""

from __future__ import annotations

import asyncio

import pytest
from bladex_proxy.toolface import (
    FAMILY_LEDGER,
    FAMILY_MEMORY,
    TOOL_NAMES,
    TOOL_SCHEMAS,
    ToolFace,
    anthropic_tool_schemas,
    inject_tools,
    is_bladex_tool,
    tool_family,
)

_AGENT_TOOLS = [
    {"type": "function", "function": {"name": "web_search", "parameters": {}}},
    {"type": "function", "function": {"name": "write_file", "parameters": {}}},
]


class TestInjectGating:
    def test_normal_injects_appended_at_end(self):
        tools, injected = inject_tools(_AGENT_TOOLS, agent_id="hermes:accept",
                                       auxiliary=False)
        assert injected
        # D3：agent 自带工具前缀逐字不变
        assert tools[:2] == _AGENT_TOOLS
        assert [t["function"]["name"] for t in tools[2:]] == list(TOOL_NAMES)

    def test_aux_turn_not_injected(self):
        tools, injected = inject_tools(_AGENT_TOOLS, agent_id="claude-code",
                                       auxiliary=True)
        assert not injected and tools == _AGENT_TOOLS

    def test_weak_tier_now_injected(self):
        """🔴 2026-08-30 行为反转：weak 档**照注**工具面。

        原断言是 `not injected`，理由是 ADR-0032「weak 档模型工具调用不可靠」
        （立卡时的假设，从未有读数支撑）。同日拍板砍掉 Memory Index 主动注入后，
        工具面成了模型取记忆的**唯一**通道——对 weak 关着就不再是"调用质量差"，
        而是"弱档 agent 彻底没有记忆"，两种代价不对称。
        账本首步指令仍按 `AUTO_LEDGER_TIERS` 排除 weak（账本策略不变），
        两者的解耦见 `agency.insert_ledger_block`。
        """
        tools, injected = inject_tools([], agent_id="hermes", auxiliary=False,
                                       model_tier="weak")
        assert injected
        names = [t["function"]["name"] for t in tools]
        assert "bladex_memory_search" in names          # 记忆族：无条件
        assert not [n for n in names if n.startswith("bladex_ledger_")]  # 账本族：auto 排除 weak


    def test_excluded_agent_gets_nothing_and_profile_suffix_follows_base(self, monkeypatch):
        """名单里的 agent 整个工具面都不注；`base:profile` 随 base 一起排除。

        🔴 **2026-09-11 改写（MQ-A66）**：原来写死 `agent_id="dsh"`，依据是
        「D2 结论：dsh 只有 run_code 可直呼」。dsh 升级后该依据失效、名单清空，
        这条测试跟着红了 —— 而它守的东西（**按 base 排除、profile 后缀随 base**）
        一个字都没变。**把机制测试绑在某个真实 agent 的能力判断上**，
        就是让机制的守卫随那个 agent 的版本一起过期。

        改用合成 agent + monkeypatch 名单：机制归机制，名单内容归配置。
        """
        import bladex_proxy.toolface as _tf
        monkeypatch.setattr(_tf, "NO_TOOLFACE_AGENT_BASES", ("excluded-probe",))
        _, injected = inject_tools([], agent_id="excluded-probe", auxiliary=False)
        assert not injected
        _, injected2 = inject_tools([], agent_id="excluded-probe:profile", auxiliary=False)
        assert not injected2, "profile 后缀必须随 base 一起排除"

    def test_idempotent(self):
        tools, _ = inject_tools(_AGENT_TOOLS, agent_id="codex", auxiliary=False)
        tools2, injected2 = inject_tools(tools, agent_id="codex", auxiliary=False)
        assert not injected2 and tools2 == tools

    def test_anthropic_format(self):
        tools, injected = inject_tools([], agent_id="claude-code", auxiliary=False,
                                       fmt="anthropic")
        assert injected
        assert all("input_schema" in t for t in tools)
        assert {t["name"] for t in tools} == set(TOOL_NAMES)

    def test_input_not_mutated(self):
        snapshot = [dict(t) for t in _AGENT_TOOLS]
        inject_tools(_AGENT_TOOLS, agent_id="codex", auxiliary=False)
        assert _AGENT_TOOLS == snapshot


class TestToolFamilies:
    """🔴 族划分本身的守卫（2026-08-30 按工具族拆）。"""

    def test_every_tool_maps_to_a_known_family(self):
        """闭集守卫：**新增工具必须落进已知族**。

        `tool_family` 对认不出来的名字返回 "unknown"，而 `inject_tools` 把
        unknown **保守当账本族**（受门管）。这条把"保守兜底"钉成"必须显式归族"——
        兜底是给运行期的安全网，不是给开发期的免责。本仓两次栽在"闭集按记忆
        默写"，所以判据从 `TOOL_NAMES` 遍历，不另抄一份清单。
        """
        unknown = [n for n in TOOL_NAMES if tool_family(n) == "unknown"]
        assert not unknown, (
            f"这些工具没归族：{unknown}。要么改名走 bladex_memory_/bladex_ledger_ 前缀，"
            "要么在 tool_family 里显式加一族并说明它该不该受档位门管。")

    def test_families_are_both_non_empty(self):
        """两族都得有人——某一族空了说明前缀约定被改坏了，而症状会是
        "某类工具悄悄不注了"，非常难查。"""
        fams = {tool_family(n) for n in TOOL_NAMES}
        assert {FAMILY_MEMORY, FAMILY_LEDGER} <= fams

    @pytest.mark.parametrize("tier", ["weak", "medium", "strong", ""])
    def test_memory_family_never_gated_by_tier(self, tier):
        tools, _ = inject_tools([], agent_id="hermes", auxiliary=False,
                                model_tier=tier)
        assert "bladex_memory_search" in [t["function"]["name"] for t in tools]


class TestSchemas:
    def test_all_names_namespaced(self):
        assert all(is_bladex_tool(n) for n in TOOL_NAMES)

    def test_goal_not_writable_via_update_schema(self):
        upd = next(t for t in TOOL_SCHEMAS
                   if t["function"]["name"] == "bladex_ledger_update")
        sections = upd["function"]["parameters"]["properties"]["section"]["enum"]
        assert "goal" not in sections  # 红线 2：schema 层就不给 goal 这个选项

    def test_anthropic_schema_parity(self):
        assert [t["name"] for t in anthropic_tool_schemas()] == list(TOOL_NAMES)

    def test_memory_search_description_states_it_is_the_only_channel(self):
        """🔴 砍掉主动注入后，这条描述承担的东西变了（2026-08-30）。

        模型不再有被动供给 —— 描述必须先说清这条通道的**地位**，再给可判定的
        触发时机。原文 "Use when past context would change your answer" 两样都
        没有：它预设模型知道有 past context 存在，且"需要时就用"不可判定。

        钉三件事，不钉措辞（措辞会随 prompt 调优变，钉死它等于禁止调优）：
          ① 说明这是唯一通道 / 早期会话不在上下文里；
          ② 四种检索模式都列出（少一种模型就用不到它）；
          ③ 提示事实带记录时间 —— 天气病例的直接教训（观测型事实无
             `valid_until`，MQ-S50，修复归 0.3.0），在此之前靠描述兜。
        """
        d = next(t["function"]["description"] for t in TOOL_SCHEMAS
                 if t["function"]["name"] == "bladex_memory_search")
        low = d.lower()
        # 判据是**性质**不是句子：说清"唯一通道"+"早期会话不在上下文里"这两件事，
        # 具体措辞随 prompt 调优可变。（初版把断言写成了一句我并没有写进描述的
        # 原文 —— 那是在测我记忆里的措辞，不是测性质。）
        assert "only" in low, "没说清它是唯一通道 —— 模型不知道自己有记忆可取"
        assert "earlier sessions" in low and "context" in low, \
            "没说清早期会话不在上下文里 —— 模型会以为自己已经看得到"
        for mode in ("query", "keyword", "matter_id", "fact_id"):
            assert mode in d, f"检索模式 {mode} 没在描述里列出"
        assert "recorded" in low or "date" in low, \
            "没提示事实带记录时间 —— 天气病例（把 08-28 的观测当今天）会复发"

    def test_ledger_cleanup_rule_stated_in_both_places(self):
        """🔴 "完成 = 两次编辑" 这条规则必须**两处都在**（Jason 2026-08-30）。

        live 形态（MQ-L29 段内条目卫生）：任务做完、结果写进 Verified，
        **Next/Open 里的对应条目还留着** ⇒ 下一个读到账本的模型把已完成的 Next
        当待办再执行。Jason 的判断是"很多时候都是**更新时候的遗忘**"，所以：

          · 首步指令（`_FIRST_STEP_WITH_LEDGER`）—— 每轮的提醒；
          · `bladex_ledger_update` description —— **调用现场**的契约。

        两处只写一处都覆盖不住：只写指令，模型调工具那一刻已经翻篇；
        只写工具描述，不调工具的轮次就没人提醒它该调。
        本条钉的是**两处都在**（Jason 原话"避免不一致"），措辞各自可调——
        判据取语义关键词，不取整句，否则禁止 prompt 调优。
        """
        from bladex_proxy.agency import _FIRST_STEP_WITH_LEDGER

        upd = next(t["function"]["description"] for t in TOOL_SCHEMAS
                   if t["function"]["name"] == "bladex_ledger_update")
        for name, text in (("bladex_ledger_update description", upd),
                           ("_FIRST_STEP_WITH_LEDGER", _FIRST_STEP_WITH_LEDGER)):
            low = text.lower()
            assert "verified" in low, f"{name}：没提 verified"
            assert "remove" in low, f"{name}：没说要删——只说写就是原来那个毛病"
            assert "next" in low and "open" in low, \
                f"{name}：Next/Open 没同时点名（原文只说了 Next，而 live 上 Open 也留着）"

    def test_tool_descriptions_stay_within_attention_budget(self):
        """🔴 工具描述**每个字都花注意力预算**（ADR-0029），不是文档。

        没有上限就会一路加下去 —— 本次把 memory_search 从 ~45 词加到 ~85 词
        是有理由的（它成了唯一入口），但那个理由不该变成"以后都可以加"。
        预算按整个工具面算：四个 schema 的 JSON 总长。当前约 5.3K 字符
        （≈1.3K token），上限取 8K，留出账本工具后续补参数的余量；
        真要超，先说明为什么这些字比它挤掉的上下文更值。
        """
        import json
        total = sum(len(json.dumps(t, ensure_ascii=False)) for t in TOOL_SCHEMAS)
        assert total <= 8000, (
            f"工具面 schema 总长 {total} 字符，超出注意力预算上限 8000。"
            "加字之前先问：它换回来的行为改进，抵得过它挤掉的上下文吗？")


class TestDispatch:
    def _face(self):
        face = ToolFace()

        async def echo(args, *, allowed_exposure, context):
            return f"ok:{args.get('query','')}:{allowed_exposure}"

        face.register("bladex_memory_search", echo)
        return face

    def test_dispatch_ok(self):
        face = self._face()
        out = asyncio.run(face.dispatch("bladex_memory_search",
                                        '{"query": "泰山"}',
                                        allowed_exposure="public"))
        assert out == "ok:泰山:public"

    def test_unknown_bladex_tool_returns_structured_error(self):
        face = self._face()
        out = asyncio.run(face.dispatch("bladex_nope", "{}",
                                        allowed_exposure="public"))
        assert out.startswith("Error: unknown BladeX tool")
        assert "bladex_memory_search" in out  # 与 hermes D2 同形态：附可用清单

    def test_bad_json_arguments(self):
        face = self._face()
        out = asyncio.run(face.dispatch("bladex_memory_search", "{bad",
                                        allowed_exposure="public"))
        assert out.startswith("Error: invalid JSON")

    def test_handler_exception_contained(self):
        face = ToolFace()

        async def boom(args, *, allowed_exposure, context):
            raise RuntimeError("db down")

        face.register("bladex_memory_search", boom)
        out = asyncio.run(face.dispatch("bladex_memory_search", "{}",
                                        allowed_exposure="public"))
        assert "failed: db down" in out

    def test_non_bladex_registration_refused(self):
        face = ToolFace()

        async def h(args, *, allowed_exposure, context):
            return ""

        with pytest.raises(ValueError):
            face.register("write_file", h)  # 红线 1：非命名空间工具不许进注册表
