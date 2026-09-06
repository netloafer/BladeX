"""v7 九款合一（G9.1 M1–M9）——颗粒度 / 主体 / 准入分流。

**这个文件存在的理由**：MQ-S17 那个病（14 张卡逐张签收后收敛到 3 张、碎片率 ≈80%）
在 gate 上是**完全隐形的**——现有蒸馏测试测的是格式与字段，没有一条测"标题是不是
一件事的名字"。所以本轮除了 prompt 条款的锚词，还必须钉住两件**判别力**相关的事：

  ① 阴性对照进 prompt：`中国啤酒行业产能结构与利用率分析报告` 是 Jason 签收的
     **独立 matter**，不得被并进泰山重整。没有它，"标题都变粗了"会被读成"修好了"
     ——而"粗到把一切合成一张卡"是最容易达标的坏解（只测收敛 = 覆盖率，不是精度）。
  ② 🔴 高危词表**不许被做成词法判据**：已签收数据里同一个词一对一错
     （`评估` 在 `泰山啤酒资产与品牌价值评估` 里是错的、在 `qwen3.8:27b 本地部署评估`
     里是对的）。见 `test_high_risk_list_cannot_be_a_lexical_judge`。

条款正确与否只能靠真实模型验（离线 harness + 全量重建后的探针读数）；
本文件钉的是"条款还在、语义没被后来的改动悄悄抹掉"，锚词测试先例 = M1-4 / v6。

**判据出处**（每条都可追溯，不抄外部语料）：
  - 标题 WRONG/RIGHT 与阴性对照：`docs/benchmarks/matter-merge-labels-20260819.md`
    §A1/§A3/§A6/§A7 与「独立成立（不合）」，Jason 2026-08-19 逐张签收；
  - 主体归属 WRONG/RIGHT：台账 MQ-S20 四条同型样本 + MQ-S21 附「边界规则」；
  - 三条道：台账 MQ-S21（A 画像 11 / B 工具 7 / C 跨主体 9）。
"""

from __future__ import annotations

from bladex_proxy.distillation import (
    _DISTILL_CONCLUSION_SYSTEM,
    _DISTILL_SYSTEM,
    _DISTILL_TURN_SYSTEM,
    _TAG_LANE_PREFIX,
    _VALID_ITEM_KINDS,
    _VALID_LANES,
    TURN_PROMPT_VER,
    _parse_distill_json,
    build_turn_prompt,
)
from bladex_core.distillation import DistillTurnInput


# ── M1 单一时钟（原案"双日期"，实现形态见 TURN_PROMPT_VER 偏离 ①）─────────


def test_m1_slot_is_named_observed_not_time():
    """槽名必须说明"什么的时间"。

    `TIME:` 不说明这是谁的时间，而回放锚错（MQ-D2）正是把它当成了"现在"。
    """
    payload = DistillTurnInput(turn_time="2026-01-01T08:00:00+00:00",
                               user_text="上周三交的报告")
    body = build_turn_prompt(payload, "zh")
    assert "OBSERVED: 2026-01-01T08:00:00+00:00" in body
    assert "TIME:" not in body


def test_m1_forbids_the_distillation_run_date_as_a_clock():
    """点名禁止"蒸馏执行日"，不只是禁止"模型自己以为的今天"。

    回放/重建时这两者是同一个陷阱的两半：模型的先验 today、以及我们真的在哪天
    重跑。v6 只挡了前者。
    """
    assert "ONE CLOCK" in _DISTILL_TURN_SYSTEM
    assert "the distillation date" in _DISTILL_TURN_SYSTEM
    assert "it is the ONLY clock" in _DISTILL_TURN_SYSTEM


def test_m1_does_not_put_a_current_date_into_the_prompt_body():
    """🔴 正文里**不许**出现当前日期——它进台账 key。

    `_distill_with` hash 的就是 `build_turn_prompt` 的产物。塞进当前日期 =
    key 每天变 = 台账整代永不命中 + 两次重建产出不同（破 ADR-0018 重建等价性）。
    这条守的是"下一个照卡面字面实现双日期的人"。
    """
    import re

    payload = DistillTurnInput(turn_time="2026-01-01T08:00:00+00:00", user_text="x")
    body = build_turn_prompt(payload, "zh")
    # 正文里出现的日期**有且只有** OBSERVED 那一个（多一个就说明有第二个时钟，
    # 而第二个时钟只要是"运行时算出来的"，key 就随日期漂）
    assert re.findall(r"\d{4}-\d{2}-\d{2}", body) == ["2026-01-01"]
    # 纯函数：同一 payload 恒得同一文本（台账 key 稳定 + 重建等价的前提）
    assert body == build_turn_prompt(payload, "zh")


# ── M2 元提取禁令 ──────────────────────────────────────────────────────────


def test_m2_admission_rule_states_content_not_act():
    assert "EXTRACT THE CONTENT, NEVER THE ACT" in _DISTILL_TURN_SYSTEM


def test_m2_wrong_right_pairs_are_our_own_cases():
    """WRONG/RIGHT 用我们自己的病例（07-29「只记住问了什么」的实证形态）。

    抽象要求模型会绕过；成品句不会。两条 WRONG 都只记录"发生过一次对话"。
    """
    assert "用户要求按照 ADR-0020 任务卡开发" in _DISTILL_TURN_SYSTEM
    assert "用户询问蒸馏为什么零事实" in _DISTILL_TURN_SYSTEM
    # RIGHT 侧必须是可复用的结论，不是"用户问了 X"的改写
    assert "secondary 模式追新" in _DISTILL_TURN_SYSTEM
    assert "BLADEX_DISTILL_TIMEOUT_S" in _DISTILL_TURN_SYSTEM


# ── M3 专名保真 ────────────────────────────────────────────────────────────


def test_m3_identifiers_verbatim_with_our_examples():
    """标识符丢了就检索不回来（memory-retrieval-issues-20260805 实测）。"""
    assert "VERBATIM" in _DISTILL_TURN_SYSTEM
    for ident in ("_visibility_pids", "config/routing.toml", "BLADEX_EMBED_BACKEND"):
        assert ident in _DISTILL_TURN_SYSTEM, ident


def test_m3_forbids_generalising_qualifiers_away():
    """限定词泛化是"看起来没丢"的丢失——mem0 的 assistant manager 例同形态。"""
    assert "Qualifiers" in _DISTILL_TURN_SYSTEM
    assert "assistant manager" in _DISTILL_TURN_SYSTEM


# ── M4 转变捕获 ────────────────────────────────────────────────────────────


def test_m4_new_state_only_is_explicitly_wrong():
    """只写新态 Y 必须被点名为错——v6 只给了正例，模型照样只记新态。"""
    assert "new state only" in _DISTILL_TURN_SYSTEM
    assert "changed FROM X TO Y, because Z" in _DISTILL_TURN_SYSTEM


# ── M5 输出自检（不写数字配额）────────────────────────────────────────────


def test_m5_self_check_covers_topics_and_late_messages():
    assert "OUTPUT SELF-CHECK" in _DISTILL_TURN_SYSTEM
    assert "COVERAGE:" in _DISTILL_TURN_SYSTEM
    assert "LATER MESSAGES:" in _DISTILL_TURN_SYSTEM


def test_m5_does_not_smuggle_in_a_numeric_quota():
    """mem0 的 5–15 条/10+ 消息是**它的语料**标定，不是我们的读数。

    抄一个没标定过的配额，模型会去凑数——那是把"覆盖率"直接写成指令。
    """
    import re

    tail = _DISTILL_TURN_SYSTEM.split("OUTPUT SELF-CHECK", 1)[1]
    assert not re.search(r"\bat least \d+\b", tail)
    assert not re.search(r"\b\d+\s*-\s*\d+\s+(facts|items)\b", tail)


# ── M6 标题粒度（本组权重最高）────────────────────────────────────────────


def test_m6_title_rule_and_self_check_present():
    """自检判据必须是那句可执行的话，不是"要写得像个名字"。"""
    assert "TITLE RULE" in _DISTILL_TURN_SYSTEM
    assert "THE NAME OF THE MATTER" in _DISTILL_TURN_SYSTEM
    assert "would say it with a DIFFERENT verb" in _DISTILL_TURN_SYSTEM


def test_m6_high_risk_endings_listed():
    """卡面那十个动作词全在（作为**高危信号**，不是禁用词，见下一条）。"""
    for verb in ("核查", "核实", "排查", "修正", "评估",
                 "分析", "生成", "修复", "检查", "定位"):
        assert verb in _DISTILL_TURN_SYSTEM, verb


def test_high_risk_list_cannot_be_a_lexical_judge():
    """🔴 这条守的是"下一个想把高危词表做成词法 gate 的人"。

    已签收数据里**同一个词一对一错**，所以任何"标题以 X 结尾 = 坏"的规则都会
    误伤 Jason 签收过的正确标题：

        评估：`泰山啤酒资产与品牌价值评估`（WRONG，泰山重整的一步）
              `qwen3.8:27b 本地部署评估`（RIGHT，A3 签收的 Matter 名）
        排查：`p2 embedding 日志 auth_ok 排查`（RIGHT，A5 已合并组的名字）

    区别不是词，是"这个动作是整件事的目的、还是它里面的一步"——语义判定。
    这与 A1/A3 各否掉一条确定性修法是同一条纪律（确定性信号只进候选）。
    """
    assert "NOT as a banned word list" in _DISTILL_TURN_SYSTEM
    assert "泰山啤酒资产与品牌价值评估" in _DISTILL_TURN_SYSTEM
    assert "qwen3.8:27b 本地部署评估" in _DISTILL_TURN_SYSTEM
    assert "p2 embedding 日志 auth_ok 排查" in _DISTILL_TURN_SYSTEM
    # 两个正确标题都以高危词结尾 —— 词法规则在签收集上必然假阳性
    assert "qwen3.8:27b 本地部署评估".endswith("评估")
    assert "p2 embedding 日志 auth_ok 排查".endswith("排查")


def test_m6_wrong_examples_are_the_signed_off_fragments():
    """WRONG 全部取自 MQ-S17 逐张签收的真实碎片，不抄外部语料。"""
    for wrong in ("泰山啤酒股东结构及股权变动核查",
                  "泰山啤酒知识产权分析",
                  "泰山啤酒股权查封解除可行性评估",
                  "泰山啤酒仁信入股时间线核实",
                  "server.py:909 quiet 参数关闭"):
        assert wrong in _DISTILL_TURN_SYSTEM, wrong
    for right in ("泰山啤酒破产重整", "张开利整体布局梳理"):
        assert right in _DISTILL_TURN_SYSTEM, right


def test_m6b_minimal_noun_phrase_covers_modifier_drift():
    """🔴 M6 第一版的射程漏洞（2026-08-19 真实读数抓到，40 轮窗口）。

    原自检判据是「下一轮会不会换个**动词**说」。实测 v7 的碎片**换的不是动词，
    是修饰语**：同一件事（读 mem0/dsh 源码产出借鉴清单）产出 19 个不同标题——
    借鉴 / 借鉴意义 / 借鉴价值 × 分析 / 评估 / 研究 / 报告 / 建议，
    主体还在 bladex / bladex 项目 / bladex-dsh 之间漂。

    对照组同一跑里 `泰山啤酒破产重整` **7 次逐字稳定**——差别是它有**现成专名**。
    ⇒ 无专名时模型即兴造名，而"换动词"那条判据够不着即兴造名的漂移。

    新判据是**单轮可判**的（删掉评价性名词看还认不认得出这件事），
    不要求模型预测别的轮——原判据要求的恰恰是它看不到的东西。
    """
    assert "MINIMAL NOUN PHRASE" in _DISTILL_TURN_SYSTEM
    assert "SINGLE-TURN TEST" in _DISTILL_TURN_SYSTEM
    # 评价性名词表：描述"你与这件事的关系"，不是这件事
    for noun in ("意义", "价值", "情况", "问题", "报告", "建议"):
        assert noun in _DISTILL_TURN_SYSTEM, noun
    # 主体写法统一（bladex 项目 / bladex-dsh 这类变体是实测漂移源）
    assert "bladex-dsh" in _DISTILL_TURN_SYSTEM
    assert "bladex 借鉴 deepseek-harness" in _DISTILL_TURN_SYSTEM


def test_m6b_does_not_ban_action_nouns_that_can_be_the_goal():
    """边界：禁的是**叠加**与评价性名词，不是动作名词本身。

    `评估` / `分析` / `研究` 可能就是整件事的目的
    （`qwen3.8:27b 本地部署评估` 是签收过的正确标题），所以它们
    **不在**评价性名词表里——那一侧由 HIGH-RISK ENDINGS + 自检判据管。
    把它们也禁掉，就会误伤已签收的正确标题（同 MQ-S22 的形态）。
    """
    seg = _DISTILL_TURN_SYSTEM.split("MINIMAL NOUN PHRASE", 1)[1].split("WRONG", 1)[0]
    banned_line = [ln for ln in seg.splitlines() if "describe" in ln or "意义" in ln]
    joined = "\n".join(banned_line)
    for goal_noun in ("评估", "分析", "研究"):
        assert goal_noun not in joined, (
            f"{goal_noun} 被写进了评价性名词禁表——它可能是整件事的目的，"
            f"禁掉会误伤 `qwen3.8:27b 本地部署评估` 这类已签收标题"
        )
    assert "AT MOST one action noun" in _DISTILL_TURN_SYSTEM


def test_m6_negative_control_against_over_merging():
    """🔴 阴性对照：粗到把一切并成一张卡是最容易达标的坏解。

    `中国啤酒行业产能结构与利用率分析报告` 是 Jason 签收的独立 matter
    （"基于泰山重整的延伸分析，有一定独立性"），**不得**被折进泰山重整。
    只测收敛 = 覆盖率读数，测不出精度。
    """
    assert "DO NOT OVER-MERGE" in _DISTILL_TURN_SYSTEM
    assert "中国啤酒行业产能结构与利用率分析报告" in _DISTILL_TURN_SYSTEM
    assert "DIFFERENT goal" in _DISTILL_TURN_SYSTEM


# ── M8 主体归属 ────────────────────────────────────────────────────────────


def test_m8_subject_rule_present_with_both_sides_of_the_boundary():
    """一轮两个主体时，fact 讲谁就归谁（MQ-S20 四条同型样本）。"""
    assert "SUBJECT RULE" in _DISTILL_TURN_SYSTEM
    assert "TWO subjects" in _DISTILL_TURN_SYSTEM
    # Jason 口述的边界规则两侧都要在，只写一侧模型分不出线在哪
    assert "the CASE itself" in _DISTILL_TURN_SYSTEM
    assert "the INVESTOR side" in _DISTILL_TURN_SYSTEM


def test_m8_examples_cover_two_of_the_three_subject_pairs():
    """人 vs 案子（张开利 ↔ 泰山重整）+ 工具 vs 工具（dsh ↔ BladeX）。"""
    assert "差额计入资本公积" in _DISTILL_TURN_SYSTEM
    assert "deepseek-harness" in _DISTILL_TURN_SYSTEM


def test_m8_does_not_add_a_subject_field():
    """边界：M8 走**已有** `subject` 字段（supersede 键 + assembly 词项已在消费），
    不加新字段。卡面"改文本 vs 加字段"里选前者——加字段要拍板，且本轮只碰 prompt。
    """
    assert '"subject": "<the THING this item is about' in _DISTILL_TURN_SYSTEM


# ── M9 准入分流三条道 ─────────────────────────────────────────────────────


def test_m9_three_lanes_declared():
    assert "LANE RULE" in _DISTILL_TURN_SYSTEM
    assert _VALID_LANES == {"task", "profile", "tool"}
    for lane in ('"task"', '"profile"', '"tool"'):
        assert lane in _DISTILL_TURN_SYSTEM, lane


def test_m9_profile_and_tool_do_not_belong_to_a_matter():
    """MQ-S21 的主结论：18/27 的错不是"归错卡"，是"根本不该进任务卡"。"""
    assert "DO NOT BELONG TO ANY MATTER" in _DISTILL_TURN_SYSTEM
    assert '"matter_proposals" MUST be []' in _DISTILL_TURN_SYSTEM
    # WRONG 例取自 C-A / C-B 两组的真实条目
    assert "ook-Pro BladeX %" in _DISTILL_TURN_SYSTEM
    assert "opencli browser 没有 snapshot 子命令" in _DISTILL_TURN_SYSTEM


def test_m9_does_not_pre_decide_the_tool_lane_shape():
    """🔴 Jason 明确 B 类形态**本轮不拍**（PROCEDURE vs 根本不做成条目）。

    prompt 只要求标对 lane，不得把 tool 一路钉死到某个 item_kind——
    钉死了就等于替他拍了板，而"看新产出再定"正是本轮的顺序拍板。
    """
    lane = _DISTILL_TURN_SYSTEM.split("LANE RULE", 1)[1].split("SUBJECT RULE", 1)[0]
    tool_para = lane.split('For lane "tool"', 1)[1]
    # 中立指派：让模型自己在六类里挑，不指定某一类
    assert "pick whichever of the six kinds" in tool_para
    # 🔴 `procedure` 是 Jason 点名"语义更贴但本轮不拍"的那一类——不许出现在这里
    assert "procedure" not in tool_para, (
        "tool 一路被钉到了 procedure——形态待看新产出后拍板，prompt 不许提前拍"
    )
    # 唯一允许提到的是**去掉**现有默认（当前存量全是 assertion，Jason 判其形态错）
    assert 'do not default to "assertion"' in tool_para


def test_m9_lane_parsed_into_existing_tags_field():
    """行为测试（非锚词）：lane 落既有 `tags`，不新增 schema 字段。

    🔴 前缀是 `lane:`，不是 `channel:`——tags 里已有 `origin:<通道>`
    （`consolidation_proxy._channel_of` 读的就是它）。同名两次 = 下一个人
    会把两份读数混起来比。
    """
    raw = ('{"topic":"t","keywords":[],"facts":['
           '{"content":"用户的终端提示符是 ook-Pro BladeX %","lane":"profile",'
           '"item_kind":"profile_obs"}],"matter_proposals":[]}')
    out = _parse_distill_json(raw, "mock/m")
    assert out is not None
    assert out.facts[0].tags == f"{_TAG_LANE_PREFIX}profile"


def test_m9_unknown_or_missing_lane_writes_no_tag():
    """宁缺毋滥：模型没给 / 给了枚举外的值 -> 不写标记（不造假标签）。

    阴性对照的一半——如果什么都标上，后面按 lane 统计的读数会恒等于 100%。
    """
    for payload in ('{"content":"c"}', '{"content":"c","lane":"whatever"}',
                    '{"content":"c","lane":null}'):
        raw = f'{{"topic":"t","keywords":[],"facts":[{payload}],"matter_proposals":[]}}'
        out = _parse_distill_json(raw, "mock/m")
        assert out is not None and out.facts[0].tags == "", payload


def test_m9_does_not_extend_the_item_kind_enum():
    """边界：lane 与 item_kind 正交，六值封闭枚举一个都不动。"""
    assert _VALID_ITEM_KINDS == {
        "assertion", "preference", "procedure", "lesson", "file_ref", "profile_obs"}


# ── 版本纪律与边界 ────────────────────────────────────────────────────────


def test_version_bumped_exactly_once_this_card():
    """🔴 九款一次改齐、一次 bump。

    v6 的免费窗口已在 08-17 那次全量重建里支付掉，所以这次 bump 是**要付钱的**
    （约 1450 次真调用），Jason 2026-08-19 拍板照付。反过来说：本卡内再动一次
    prompt 就得再付一次，要改就停下来重排卡。
    不 bump 而改内容更不行——台账 key 含版本，旧条目会继续命中缓存，
    库里新旧两代产出混在一起且无法区分。

    🔴 串名当天改过一次（`v7-granularity-subject-001` → `v7-title-noun-001`，
    M6 补 `MINIMAL NOUN PHRASE` 之后）。**不是第二次 bump**：一卡一 bump 管的是
    钱，而这次没付过（全量重建未跑，预检走临时 index）。改名是因为
    `MemoryIndex.clear()` **不清 distill 台账**（跨重建保留 = ADR-0018 的设计），
    live 若用第一版 v7 蒸过任何一轮，全量重建就会命中旧产出、混两代。
    改名成本 0，不改的风险坐实 —— 不对称就按低风险那边走。
    """
    assert TURN_PROMPT_VER == "v7-title-noun-001"


def test_only_turn_prompt_touched():
    """v3 / conclusion 生产零流量（T1a 台账实测），本轮一个字不动。

    它们的文本 hash 由 `test_distill_three_segment.py` 冻结；这里钉的是
    "新条款没有溢出到它们身上"。
    """
    for other in (_DISTILL_SYSTEM, _DISTILL_CONCLUSION_SYSTEM):
        assert "TITLE RULE" not in other
        assert "LANE RULE" not in other
        assert "SUBJECT RULE" not in other
        assert "ONE CLOCK" not in other
