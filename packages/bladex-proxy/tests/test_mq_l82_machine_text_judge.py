"""MQ-L82（批 W W2，2026-09-22）：`agency_ledger_created_from_machine_text`
的判据此前**覆盖 0/6 个 live 形态** ⇒ 存量日志里它恒为 0 是**假绿**。

## 同一个读数，两种读法

- 旧读法："没有账本是从机器文本建起来的"。
- 实情：**判据认不出任何一种 live 形态**（`_machine_text_mark` 的三条指纹
  `you are a` / `# overview` / `you are an agent` 拿六个形态过一遍，6/6 全漏）。

与 MQ-L78「`after=0` 长得像成功」同族。本文件把**哪个读法对**定死：
台账那张六形态表直接当用例表，一条一个断言。

## 修法

主判据改为复用**取值侧**的 `_clean_user_text`（`_last_user_text(for_goal=True)`
用它判"这条里有没有用户说的话"）：剥完为空 ⇒ 机器文本。三条采样指纹降为补充项，
并从"开头 200 字符内 `in`"收紧为"剥完之后 `startswith`"。

## 判别力边界（别高估这份文件）

主判据的覆盖面 = `envelope` 生产表的覆盖面。生产表漏掉的新信封（MQ-L81 登记的
那一类"没被识别出来的信封"）本判据照样漏——这是**有意的**：一张表两个消费方，
补覆盖要补在生产表上，不是在告警侧再抄一份指纹（MQ-S45：按采样写死的指纹，
agent 换个变体就复发）。
"""
from __future__ import annotations

import pytest

# ── 台账六形态表（`memory-quality-problem-ledger` MQ-L82）────────────────────
#
# 第三列 = 修前 `_machine_text_mark` 的返回值，**全是 ''**。
SIX_LIVE_FORMS = [
    ("dsh_skills_dir",
     "<system-reminder>\n"
     "The following skills are available for use with the Skill tool:\n"
     "- mx-data: 金融数据查询工具。\n- dataviz: charts.\n"
     "</system-reminder>"),
    ("dsh_agents_md",
     "<system-reminder>\n"
     "Contents of /Users/you/dev/BladeX/CLAUDE.md (project instructions):\n\n"
     "# BladeX — AI 协作开发指南\n\n本文档是所有 AI 编码工具的唯一权威参考。\n"
     "</system-reminder>"),
    ("dsh_runtime_context",
     "Current runtime context. This snapshot supersedes any earlier statements "
     "about the sandbox, approvals, or network access.\n"
     "Sandbox: workspace-write\nApproval policy: on-request\nNetwork access: enabled"),
    ("codex_history",
     "The following is the Codex agent history added since your last approval "
     "assessment:\n<turn>assistant: running ls</turn>"),
    ("codex_agents_md",
     "# AGENTS.md instructions for /Users/you/dev/BladeX\n"
     "<INSTRUCTIONS>\n先读 ADR-0032，再动手。\n</INSTRUCTIONS>"),
    ("hermes_skill_library",
     "Review the conversation above and update the skill library."),
]


@pytest.mark.parametrize("name,text", SIX_LIVE_FORMS,
                         ids=[n for n, _ in SIX_LIVE_FORMS])
def test_all_six_live_forms_are_flagged(name, text):
    """判据①：台账六形态**逐条**判为机器文本（修前 0/6，现 6/6）。"""
    from bladex_proxy.agency import _machine_text_mark
    mark = _machine_text_mark(text)
    assert mark, f"{name}: 仍然认不出来（返回 {mark!r}）"


def test_flagged_form_names_are_specific():
    """形态名要能指认**是哪一种**机器文本，不能全是一个笼统标签。

    形态名进告警的 `mark=` 字段，排查时要靠它分流；六条全返回
    `"envelope_only"` 也能让上一条测试全绿，但那读数没有分辨力。
    """
    from bladex_proxy.agency import _machine_text_mark
    marks = {n: _machine_text_mark(t) for n, t in SIX_LIVE_FORMS}
    assert len(set(marks.values())) >= 4, marks
    assert marks["codex_agents_md"] == "codex_agents_md_header"
    assert marks["dsh_runtime_context"] == "dsh_runtime_context"
    assert marks["codex_history"] == "codex_history_injection"


# ── 判据②：误判防线（本卡唯一红线）────────────────────────────────────────
#
# 判宽了把真人提问当机器文本，后果比漏判重：漏判只是少一条告警，
# 误判会让"用户明确说的这件事"被当噪声。

REAL_USER_QUESTIONS = [
    # 短句
    "跑一下",
    "为什么 W1 会报错？",
    "ok，继续",
    # 中文长句
    "帮我看下 routing.py 里熔断过滤那段，静态命中时它会不会改写我配的模型？",
    "小黑，帮我分析一下这次的路由事故，重点是为什么告警只有 info 级",
    # 贴了代码片段的提问
    "这段为什么不生效：\n```python\nexcept Exception as e:\n    logger.warning('x')\n```\n"
    "是不是被上面那个 except 先接住了？",
    "```toml\n[strategies.request]\nenabled = false\n```\n这样写 claude-code 还生效吗",
    # 🔴 自指陷阱（MQ-A19）：**提到**模板句的真人提问
    "你觉得 'You are a helpful assistant' 这种 prompt 好不好？我想换一版",
    "客户端把 You are an agent 塞进 user 消息合理吗",
    # 提到信封标签本身的提问（本仓文档里逐字写着这些标签）
    "CLAUDE.md 里那段 <system-reminder> 的说明要不要改？",
]


@pytest.mark.parametrize("text", REAL_USER_QUESTIONS)
def test_real_user_questions_are_not_flagged(text):
    """判据②：一组真实用户提问（短句 / 中文 / 贴代码 / 提到模板句）⇒ **不判**。"""
    from bladex_proxy.agency import _machine_text_mark
    assert _machine_text_mark(text) == "", f"误判为机器文本：{text[:40]!r}"


def test_empty_text_is_not_flagged():
    """空 / 纯空白不是机器文本（建本时 verbatim 可能为空）。"""
    from bladex_proxy.agency import _machine_text_mark
    assert _machine_text_mark("") == ""
    assert _machine_text_mark("   \n  ") == ""


def test_known_boundary_message_starting_with_a_template_heading():
    """📌 **已知边界，非缺陷**：整条以模板头开头的真人消息会被判中。

    "# Overview …" 这类开头，只看前缀分不出"模板"还是"用户贴了一份文档"。
    留在这里是为了让这条边界有名有姓（而不是某天被当成新缺陷重新发现一次）：
    它**只影响一条告警**，不阻止建本（`test_alarm_does_not_block_creation` 钉着），
    代价可接受。要再收紧得引入"整条是不是文档"这种判断，不在本卡。
    """
    from bladex_proxy.agency import _machine_text_mark
    assert _machine_text_mark("# Overview 这个标题要不要改成中文") == "md_overview"


# ── 判据③：单一实现点（一张表两个消费方）──────────────────────────────────

def test_doctor_and_agency_share_one_judge():
    """doctor 侧与建本告警侧必须是**同一个函数**，不许各写一份采样指纹。

    `test_cli_ledger_doctor.py::test_reuses_agency_machine_text_mark` 已经钉了
    "doctor 必须 import 它"；这里补一条**行为侧**的对账：同一段文本，
    两个消费方的判定一致（import 对了、但传的探针不同样会分叉）。
    """
    from bladex_proxy.agency import _machine_text_mark
    from bladex_proxy.cli import ledger_cmds

    src = ledger_cmds.__file__
    with open(src, encoding="utf-8") as f:
        text = f.read()
    assert "from bladex_proxy.agency import _machine_text_mark" in text
    # doctor 的探针 = goal_verbatim（用户原话）→ 与建本侧同一个入参语义
    assert "goal_verbatim" in text
    for _name, form in SIX_LIVE_FORMS:
        assert _machine_text_mark(form), "doctor 与建本侧共用的判据漏了一个 live 形态"


# ── 判据④：阴性对照（改回旧的三条采样规则 ⇒ 判据① 红）─────────────────────

def test_negative_control_old_sampling_rules_miss_all_six():
    """判据④：把判据换回"三条指纹 + 开头 200 字符 `in`" ⇒ 六形态 **0/6**。

    这条不改生产代码，而是把旧判据**原样复刻**在测试里跑一遍六形态表——
    它同时是 MQ-L82 那张表的可执行版本：修前恒 0 的读数是怎么来的，一目了然。
    """
    old_marks = (("assistant_persona", "you are a "),
                 ("md_overview", "# overview"),
                 ("agent_persona", "you are an agent"))

    def old_judge(text: str) -> str:
        head = (text or "")[:200].lower()
        for name, mark in old_marks:
            if mark in head:
                return name
        return ""

    hits = {n: old_judge(t) for n, t in SIX_LIVE_FORMS if old_judge(t)}
    assert hits == {}, f"旧判据本应 6/6 全漏，实际命中 {hits}"
