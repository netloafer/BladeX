"""Dashboard / admin API 对齐五层调整（2026-08-26）。

三块新东西要在界面上成立，否则改了等于没改：
  ③④ 漂移证据与"疑似 X 升级"要看得见（否则用户看到的还是一个裸 hash）
  ⑤  认领之后必须能一键归位（否则用户以为改完了，实际存量记忆没动）
  ②  子代理是 `codex:guardian`，应折进 codex 而不是另起一行

🔴 字段契约要测：`/admin/agents` 里那句注释写着"字段可选 = 契约测不住 =
改名后页面静默渲染成空"——`drift` 是新字段，正好适用。
"""

from __future__ import annotations

import re
from pathlib import Path

DASH = (Path(__file__).resolve().parents[1]
        / "bladex_proxy" / "dashboard.html").read_text(encoding="utf-8")


def _rename_block() -> str:
    """Rename 对话框那一段（从它的标题锚到提交前）。

    🔴 不用 `}else{` 之类的结构锚：那种锚点随代码分支增减而漂，
    切错块之后测试会以"行为错"的面目失败，浪费一整轮排查。
    """
    i = DASH.index("title:`Rename ${from}`")
    return DASH[i:DASH.index("  v.to=target;", i)]


class TestApiContract:
    def test_unknown_buckets_always_carry_drift(self):
        """两个来源（进程内登记 / Memory Hub）都要给 `drift`，**形状一致**——
        前端不许对可选字段分支。"""
        import inspect

        from bladex_proxy import server
        src = inspect.getsource(server.create_app)
        # 两处构造 bucket 字典，各自都得有 drift
        assert src.count('"drift"') >= 2, "有一个来源漏了 drift ⇒ 前端要分支判断"

    def test_reattribute_endpoint_exists(self):
        import inspect

        from bladex_proxy import server
        src = inspect.getsource(server.create_app)
        assert '"/admin/agents/reattribute"' in src

    def test_claim_hint_no_longer_promises_automatic_rebuild(self):
        """🔴 原 hint 说"下一次 Memory Index 重建时改归属"——**对增量重建不成立**
        （纠正版被 novelty 判重丢弃）。留着它，用户会以为什么都不用做。"""
        import inspect

        from bladex_proxy import server
        src = inspect.getsource(server.create_app)
        assert "re-attributed on the next Memory Index rebuild" not in src
        assert "are NOT" in src and "reattribute" in src


class TestSyncControlAction:
    def test_submit_carries_action(self):
        from bladex_proxy.sync_control import SyncControl

        captured: dict = {}

        class _R:
            def hset(self, *a, **k): pass
            def expire(self, *a, **k): pass
            def set(self, *a, **k): pass
            def lpush(self, _k, v): captured["cmd"] = v

        import json
        SyncControl(_R()).submit(action="reattribute")
        assert json.loads(captured["cmd"])["action"] == "reattribute"

    def test_default_action_is_sync_zero_regression(self):
        """既有 `bladex sync` 不传 action —— 默认必须还是 sync。"""
        from bladex_proxy.sync_control import SyncControl

        captured: dict = {}

        class _R:
            def hset(self, *a, **k): pass
            def expire(self, *a, **k): pass
            def set(self, *a, **k): pass
            def lpush(self, _k, v): captured["cmd"] = v

        import json
        SyncControl(_R()).submit(full=True)
        assert json.loads(captured["cmd"])["action"] == "sync"

    def test_consolidator_branches_on_action(self):
        import inspect

        from bladex_proxy import consolidator
        src = inspect.getsource(consolidator.main)
        assert 'cmd.get("action")' in src and "reattribute_by_claims" in src


class TestDashboardRendering:
    def test_drift_evidence_is_rendered(self):
        assert "looks like" in DASH
        assert "transport overlap" in DASH
        assert "session continues" in DASH

    def test_the_discriminator_is_explained_to_the_user(self):
        """🔴 "接替 vs 并存"是区分"升级"与"子代理"的判据。界面上不解释这一条，
        用户就会把并行的子代理合并掉——那是误合并。"""
        assert re.search(r"took over.*coexists|coexists.*took over", DASH, re.S)
        assert "sub-agent" in DASH and "mis-attribution" in DASH

    def test_evidence_is_shown_not_acted_on(self):
        """BladeX 给证据、不给结论——误合并=0 是红线，"看起来像"不足以自动合并。"""
        assert "shown, not acted on" in DASH

    def test_claim_offers_reattribution_afterwards(self):
        assert "/admin/agents/reattribute" in DASH
        assert "Re-attribute existing memories?" in DASH

    def test_reattribution_prompt_explains_why_it_is_cheap(self):
        """用户会犹豫"这是不是又要跑两个半小时"——必须当场说清它不是重建。"""
        assert "no re-distillation" in DASH and "seconds not hours" in DASH

    def test_matter_cards_are_mentioned(self):
        """Matter 卡的 participants 也带 agent_id，而它是**整卡注入**的
        ——不提这一条，用户不知道为什么值得跑。"""
        assert "Matter card participants" in DASH


class TestMergeTargetIsPicked:
    """🔴 Jason 2026-08-26：合并进已知 agent 必须**选**，不能**打**。

    自由文本输入的失败模式很恶劣：打错一个字不会报错，它会被当成一个**新** agent 名
    ——那是身份分裂，而且悄无声息（认领成功、toast 也绿）。
    """

    def test_ask_supports_select(self):
        assert 'f.type==="select"' in DASH, "对话框不支持下拉 ⇒ 只能自由输入"

    def test_claim_dialog_lists_known_agents(self):
        assert '"Merge into "+a' in DASH
        assert "/admin/agents" in DASH and "known_agents" in DASH

    def test_new_agent_is_an_explicit_choice(self):
        """"新建"要是一个显式选项，不是"没选中"的副作用。"""
        assert "__new__" in DASH
        assert "New agent (type a name below)" in DASH

    def test_empty_target_is_refused(self):
        """选了"新建"却没填名字 ⇒ 明确报错，不许提交空目标。"""
        assert "Pick an agent to merge into, or type a new name" in DASH

    def test_drift_candidate_is_preselected(self):
        """有 ≥2 条证据指向某个已知 agent 时，下拉默认选中它 —— 一键路径。"""
        assert "knownList.includes(looks)" in DASH


class TestClaimAndRenameAreSeparateDialogs:
    """🔴 Jason 2026-08-26 第二次指出：第一版把合并下拉加在了**共用**的对话框上，
    于是已知 agent 的 **Rename** 里也冒出"Merge into XX"——用户点 Rename 是要改
    名字，不是要挑合并目标，操作逻辑不通。

    两个操作语义不同，界面必须分开（提交路径仍共用 —— `AGENT_CLAIM` 本就是
    "认领/改名/合并"一套语义）：
      未识别桶 → Claim：给身份，可合并进已知 agent，也可新建 ⇒ **下拉**
      已知 agent → Rename：改名 ⇒ **文本框**
    """

    def test_two_entry_points_exist(self):
        assert "async function agentClaim(from,drift)" in DASH
        assert "async function agentRename(from)" in DASH

    def test_known_agent_row_calls_rename_not_claim(self):
        """判据要**锚在 agent 行上**：页面里别处也有 Rename 按钮（Matter 卡就有一个），
        裸搜 `>Rename</button>` 会先匹配到它——第一版就栽在这上面，
        测试失败的原因不是被测行为错了，是判据不精确。"""
        assert "agentRename('${esc(a.agent_id)}')" in DASH, \
            "Known agents 行的 Rename 没挂到 agentRename 上"
        assert "agentClaim('${esc(a.agent_id)}')" not in DASH, \
            "Known agents 行还在调 agentClaim ⇒ 改名对话框里会冒出合并下拉"

    def test_unknown_bucket_row_calls_claim(self):
        import re
        row = re.search(r'>\$\{looks\?"Merge / Claim":"Claim"\}</button>', DASH)
        assert row, "未识别桶的按钮不见了"
        seg = DASH[max(0, row.start() - 200):row.end()]
        assert "agentClaim(" in seg

    def test_merge_dropdown_only_in_claim_branch(self):
        """下拉必须在 `isClaim` 分支里 —— 在共用层就会漏进 Rename。"""
        i_if = DASH.index("if(isClaim){")
        i_sel = DASH.index('label:"Merge into "+a')
        i_else = DASH.index("  }else{", i_if)
        assert i_if < i_sel < i_else, "合并下拉不在 claim 分支内"

    def test_rename_branch_has_no_merge_option(self):
        """判据锚在 **Rename 对话框自己**上。
        用 `}else{` 定位曾经可行，两步对话框之后合并分支里也有一个 `}else{`
        —— 切错了块，测试红的原因是判据不精确而不是行为错（今天第二次）。"""
        blk = _rename_block()
        assert "Merge into" not in blk

    def test_rename_refuses_empty_and_unchanged(self):
        assert "Name cannot be empty" in DASH
        assert "Name unchanged" in DASH


class TestSubagentsHiddenFromUI:
    """Jason 2026-08-26：`codex:guardian` 这类任务型子代理**界面不显示**
    ——名称随 agent 版本随时变，列出来只是噪声。轮次仍计入 base。"""

    def test_discriminator_comes_from_config_not_guessing(self):
        """判据是**规则里的两个标志**，不是"名字看起来像"：
        `subagent_headers` ⇒ 内部角色（隐藏）；`profile_aware` ⇒ 用户配置文件（显示）。"""
        from bladex_proxy.agent_rules import load_agent_rules
        rules = load_agent_rules()
        sub = {r.agent_id for r in rules if r.subagent_headers}
        prof = {r.agent_id for r in rules if r.profile_aware}
        assert sub and prof
        assert not (sub & prof), "两个标志重叠 ⇒ 判据不成立，得换判法"

    def test_server_filters_subagent_profiles(self):
        import inspect

        from bladex_proxy import server
        src = inspect.getsource(server.create_app)
        assert "_subagent_bases" in src
        assert "base not in _subagent_bases" in src

    def test_turns_still_counted_on_the_base(self):
        """隐藏的是名字，不是流量 —— guardian 的轮次仍要算进 codex。"""
        import inspect

        from bladex_proxy import server
        src = inspect.getsource(server.create_app)
        i_turns = src.index('entry["turns"] += count')
        i_filter = src.index("base not in _subagent_bases")
        assert i_turns < i_filter, "轮次累加被挡在过滤之后 ⇒ 子代理流量凭空消失"


class TestAdminErrorShape:
    """🔴 2026-08-26 live：点"Merge into codex"完全没反应。

    服务端**按设计工作**（`400 confirm_required`），断在前端取错误文本的那一行：

        throw new Error((data.error && data.error.message) || ("HTTP " + r.status));

    但 admin 端点走 `_admin_response`，产出的是
    `{"status": "confirm_required", "detail": "...", ...}` —— **没有 `error` 字段**。
    于是 `e.message` 恒为 `"HTTP 400"`，而前端判的是 `/confirm/i.test(e.message)`
    ⇒ 走 else 分支 ⇒ 弹一个三秒淡出的 `HTTP 400` 红条，看起来就是"没反应"。

    **这是个既有 bug**（所有 admin 侧 4xx 都退化成 "HTTP 4xx"），
    我的改动只是让它第一次被真的点到 —— 合并的二次确认路径**从来没跑通过**。
    """

    def test_error_message_falls_back_to_detail_and_status(self):
        assert "data.detail||data.status" in DASH, \
            "错误文本还是只认 data.error.message ⇒ admin 4xx 全退化成 HTTP 4xx"

    def test_structured_status_is_attached_to_the_error(self):
        assert "err.status=data.status" in DASH

    def test_confirm_branch_keys_on_status_not_prose(self):
        """判据挂结构化字段，不挂文案——文案会改、会翻译。"""
        assert 'e.status==="confirm_required"' in DASH

    def test_server_really_returns_that_status(self):
        """契约两头对齐：前端判的那个字符串，服务端得真的发。"""
        import inspect

        from bladex_proxy import server
        src = inspect.getsource(server.create_app)
        assert '"confirm_required"' in src

    def test_admin_response_has_no_error_field(self):
        """钉住前提：`_admin_response` 不产 `error` 字段。
        哪天它产了，上面那条回退链要重新想一遍。"""
        import inspect

        from bladex_proxy import server
        src = inspect.getsource(server._admin_response)
        assert '"status": status_text' in src and '"error"' not in src


class TestMergeDoesNotCarryContentRules:
    """🔴 Jason 2026-08-26：合并时预置的 keyword 会污染目标 agent。

    实测：guardian 子代理桶采出来的 keyword 是
    "You are judging one planned coding-agent action" —— 把它写成 `codex` 的规则，
    既匹配不上 codex 主路径（那是 guardian 的提示词），又给一个本来工作正常的
    agent 加了一条噪声规则。识别是"任一命中"，多一条错的就多一片误命中面。

    根因是 `suggest_rule` 从**来源桶自己的流量**采样——那份采样属于来源、
    不属于目标。两种情形的规则语义本来就不同：
      新建 → 目标没有规则，必须从零引导，内容维是主力
      合并 → 目标已有规则（正因如此才被识别出来）；漂移是**内容维失效**、
             header 仍稳定，所以唯一值得补的是 A/B 档 header 规则
    """

    def test_three_rule_modes_exist(self):
        """full=新建 / header=合并（自动采用建议） / none=不带规则。"""
        assert 'ruleMode="full"' in DASH
        assert 'ruleMode=hdrRule?"header":"none"' in DASH
        assert 'ruleMode==="header"' in DASH

    def test_content_dims_only_in_full_mode(self):
        """system_prompt_keywords / tool_signatures 只许出现在 full 分支里。"""
        i_full = DASH.index('if(ruleMode==="full")')
        i_hdr = DASH.index('else if(ruleMode==="header"')
        full_blk = DASH[i_full:i_hdr]
        hdr_blk = DASH[i_hdr:DASH.index("  let r;", i_hdr)]
        assert "system_prompt_keywords" in full_blk
        assert "system_prompt_keywords" not in hdr_blk, \
            "合并分支带上了内容维规则 ⇒ 会污染目标 agent"
        assert "tool_signatures" not in hdr_blk

    def test_merge_asks_the_user_nothing_about_rules(self):
        """🔴 Jason 2026-08-26 第二轮："如果是合并，不应该让我来动吧"。

        用户的决定是"这个桶是 codex"；规则长什么样系统比他清楚。让他编辑一条正则
        是让他做系统该做的事。所以合并路径**没有规则输入字段**——
        `suggest_rule` 自证过的 header 规则直接采用，披露放在二次确认那一屏。
        """
        assert "hdr_pattern" not in DASH, "合并路径还在让用户编辑正则"
        assert "hdr_name" not in DASH
        assert 'hdrRule=(sug.header_patterns||[])[0]' in DASH

    def test_confirm_screen_discloses_the_rule_change(self):
        """不让用户编辑 ≠ 偷偷改。二次确认必须同时说清两件事：
        合并了两个记忆命名空间 + 给目标 agent 加了一条识别规则。"""
        i = DASH.index('title:"Confirm merge"')
        blk = DASH[i:i + 1200]
        assert "joins two memory namespaces" in blk
        assert "a header rule will be added" in blk
        assert "hdrRule.name" in blk and "hdrRule.pattern" in blk

    def test_no_rule_case_is_also_disclosed(self):
        """通用 SDK 名（hermes/Pi）给不出 UA 规则——那也要说，
        否则用户以为识别修好了，下个版本照样落新桶。"""
        i = DASH.index('title:"Confirm merge"')
        assert "No recognition rule is added" in DASH[i:i + 1200]

    def test_rename_carries_no_rule_at_all(self):
        """改名不是识别问题——不该顺手改规则。"""
        blk = _rename_block()
        assert "ruleFields" not in blk and "keyword" not in blk

    def test_rule_name_uses_target_not_source(self):
        """规则的 agent_id 必须是**目标**。用 `v.to` 曾经可行，但两步对话框之后
        `v` 是第二步的答案、不再带 `to` —— 用 target 才是唯一真相。"""
        assert "{name:target,agent_id:target}" in DASH
        assert "name:v.to.trim(),agent_id:v.to.trim()" not in DASH


class TestNewNameDisabledWhenMerging:
    """Jason 2026-08-26：选了合并目标之后，"New agent name" 输入框要 disable。

    不禁用的话，用户可能一边选了 `Merge into codex`、一边还在输入框里打字，
    然后不确定哪个生效 —— 界面在同一时刻给出两个互斥的意图。

    实现是**声明式**的 `enableWhen:{field,equals}`：
      - 不注入裸 JS 字符串（那会让 `ask()` 变成执行任意代码的洞）；
      - 不写死 "target"/"to" 这对具体名字（下一个用到联动的对话框会各写一份）。

    额外收益（jsdom 实测）：**disabled 的字段不进 FormData** ——
    选了合并之后 `v.to` 直接不存在，调用方拿不到一个"用户其实没打算用"的值。
    """

    def test_ask_supports_declarative_enable_when(self):
        assert "f.enableWhen" in DASH
        assert "enableWhen.field" in DASH and "enableWhen.equals" in DASH

    def test_no_raw_js_injection_hook(self):
        """联动不许走"传一段 JS 字符串进来"那条路。"""
        assert "f.onchange" not in DASH and "new Function(" not in DASH

    def test_new_name_field_declares_the_link(self):
        i = DASH.index('{name:"to",label:"New agent name"')
        blk = DASH[i:i + 260]
        assert 'enableWhen:{field:"target",equals:"__new__"}' in blk

    def test_initial_state_respects_the_preselected_target(self):
        """有漂移证据时下拉预选了目标 agent —— 那么输入框**初次渲染就该是灰的**，
        不能等用户先动一次下拉才生效。"""
        assert "f.enableWhen&&!(f.enableWhen.equals===" in DASH

    def test_disabled_field_is_visually_dimmed(self):
        assert 'style.opacity=on?"":"0.45"' in DASH
