"""MQ-L78 / 执行卡 G19.0 —— 取「用户说的那句话」的口径修正。

live 病例（`docs/reviews/dsh-ledger-unreachable-20260920.md`，session `fp:dd4bc3ca0ddd`）：
dsh 把注入上下文**以 user 角色单独发一条**排在提问之后，`_last_user_text` 取「最后一条
`role=user`」⇒ 剥完为空 ⇒ 回退原文 ⇒ 拿 1446 字符的 skills 目录当「用户这句话」去做账本召回。
目标账本相关性 0.80 排第一，实际召回到两本无关账本；模型 13 轮 / 21 次 `bladex_memory_search`
才自己撞见 id。

🔴 本文件的消息序列**逐字取自那一轮的 Hub 记录**（`local/dsh/fp:dd4bc3ca0ddd/1789835229139-0`），
只有第二条（6093 字符的 AGENTS.md 全文）为体积做了省略 —— 它整条是 `<system-reminder>`，
剥离结果与原文一致（都是空），不影响任何判据。

判据（交接单 §3 G19.0）：
  ① 本病例序列 ⇒ `query_units` 57 → 19，相关性第一名 = 目标账本；
  ③ 阴性对照：改回「最后一条 user」⇒ ① 必须红；
  ④ 回归：CC（提问 + 同条消息尾部 reminder）/ hermes（纯文本）取值逐字不变。

live 复算（本机 `.venv`，非沙盒）：修前 `query_units=57` /
`relevant=['ldg-8b18dd601d95','ldg-37696c99a521']` / `top=0.65` —— 与当时日志逐字一致；
修后 `query_units=19` / `ldg-497d5e2d542b` / `score=0.80` / `shared=6`。
读数与复算过程见 `docs/planning/report-batch-G19-20260920.md` §G19.0。
"""

from __future__ import annotations

import structlog.testing
from bladex_core.envelope import strip_envelopes
from bladex_core.ledger import Ledger
from bladex_core.ledger_runtime import candidate_units, relevant_ledgers
from bladex_proxy.agency import _clean_user_text, _last_user_text
from bladex_proxy.agency.notes import _BARE_PREFIX_RES, _GOAL_MAX_CHARS, _WRAPPER_TAG_RE

#: 用户真正说的那句话（42 字符）。
_USER_QUESTION = "之前新款Mac mini和Mac studio的研究怎么样了？能担当端侧算力平台吗？"

#: dsh 单发的第二条 user 消息：AGENTS.md 全文塞进 `<system-reminder>`（live 6093 字符，
#: 此处省略正文 —— 整条即信封，剥离结果与原文一致）。
_WORKSPACE_REMINDER = "<system-reminder>\nThe following workspace instructions may be relevant to your work. Use them as guidance when applicable. More specific instructions take precedence over broader ones. They do not override system, developer, or direct user instructions.\n\nInstr\n…（AGENTS.md 全文省略，剥离结果不变：整条即信封）…\n</system-reminder>"

#: 🔴 dsh 的运行时上下文快照：**没有任何信封标签**，`strip_envelopes` 原样返回。
#: 「往前找剥完非空的 user」会停在这条上 —— 这是 MQ-L81 的样本（live 485 次）。
_RUNTIME_CONTEXT = "Current runtime context. This snapshot supersedes earlier runtime-context snapshots.\n\nCurrent DSH file policy: workspace-write. Any available operation enforced by the DSH file sandbox may modify files under the session workspace: \"/Users/you/Documents/Code/vllm\". Some platform temporary areas may also be writable.\n\nApproval policy: ask. Operations that require approval may ask through the configured answerers; without an available answerer, the request fails closed."

#: live 里被当成「用户这句话」的那 1446 字符（截断到 600 后 `query_units=57`）。
_SKILLS_REMINDER = "<system-reminder>\nA skill is a reusable set of task-specific instructions. The following skills are available in this session:\n\n<available_skills>\n- `ci-fails-buildkite`: Fetch and diagnose vLLM Buildkite CI failure logs. Use when investigating failing CI jobs on a PR or build, when the user pastes a buildkite.com URL, or asks to fetch/diagnose CI logs.\n- `debug-ima`: Debug CUDA illegal memory access (IMA) errors in vLLM with CUDA core dumps and cuda-gdb.\n- `kernel-microbenchmark`: Build, debug, and interpret vLLM GPU kernel microbenchmarks for CUDA, Triton, and CuteDSL, including CUPTI timing, correctness checks, generated-code inspection, multi-GPU measurements, and SOL sanity checks.\n- `triton-kernel-writing`: Write or review Triton kernels for vLLM, with practical guidance for generated-code inspection, launch grids, indexing, specialization, tuning, and representative performance validation.\n</available_skills>\n\nIf the user names a skill, or the task clearly matches a skill's description, call the `skill` tool with the exact skill name before taking task actions. Load all applicable skills, then follow their full instructions. This catalog contains summaries only; do not infer or follow a skill's instructions until it has been loaded.\nA user may also invoke a skill directly; its <skill_content> block then appears in this conversation. Follow it, and do not call the `skill` tool again for that skill.\n</system-reminder>"

def _msgs() -> list[dict]:
    return [{"role": "system", "content": "you are dsh"},
            {"role": "user", "content": _USER_QUESTION},
            {"role": "user", "content": _WORKSPACE_REMINDER},
            {"role": "user", "content": _RUNTIME_CONTEXT},
            {"role": "user", "content": _SKILLS_REMINDER}]


def _old_last_user_text(messages: list[dict]) -> str:
    """修前口径，逐字取自 `82e5946` 的 `notes.py` —— **只作阴性对照，不进生产**。

    留在测试里而不是靠 git 历史：阴性对照要能在当前工作区一键复现（复核纪律
    「只认复核方自跑的 pytest」），而不是要求复核方去翻一个 commit。
    """
    raw = ""
    for m in reversed(messages or []):
        if m.get("role") == "user":
            c = m.get("content")
            raw = (" ".join(str(b.get("text", "")) for b in c if isinstance(b, dict))
                   if isinstance(c, list) else str(c or ""))
            break
    if not raw:
        return raw
    cleaned, _marks = strip_envelopes(raw)
    for pat in _BARE_PREFIX_RES:
        stripped, n = pat.subn("", cleaned, count=1)
        if n:
            cleaned = stripped
    m = _WRAPPER_TAG_RE.match(cleaned.strip())
    if m:
        cleaned = m.group("inner")
    raw = cleaned.strip() or raw.strip()
    if len(raw) > _GOAL_MAX_CHARS:
        raw = raw[:_GOAL_MAX_CHARS] + "…[bladex-goal-truncated]"
    return raw


def _pool() -> dict[str, Ledger]:
    """三本账本的合成池 —— 目标本 + 两本 live 里真被误召的本。

    🔴 score 的绝对值依赖池内 idf，合成池**不复现** live 的 0.80（那是 28 本池的读数，
    在结算稿里，由 Hub 只读复算）。这里钉的是**排序与命中**：目标本必须排第一。
    """
    return {
        "ldg-497d5e2d542b": Ledger(
            ledger_id="ldg-497d5e2d542b", agent_id="dsh",
            title="苹果新 Mac mini / Mac Studio 亮点与性价比分析",
            goal=("分析苹果最新发布的 Mac mini 与 Mac Studio：亮点、相对上一代的重大提升、"
                  "性价比评估，并结合老板本地 LLM 推理工具体系给出实用影响判断。")),
        "ldg-8b18dd601d95": Ledger(
            ledger_id="ldg-8b18dd601d95", agent_id="claude-code",
            title="MQ-W16：status 命令 Memory 失败提示按成因分开报",
            goal="让 bladex status 的 Memory 一行按成因分开报错，而不是一句 unavailable。"),
        "ldg-37696c99a521": Ledger(
            ledger_id="ldg-37696c99a521", agent_id="codex",
            title="4 independent bug-fix tasks (django, sympy, requests)",
            goal="Fix four independent upstream bugs with regression tests."),
    }


class TestCriterionOne:
    """判据 ①：本病例序列 ⇒ 取到用户原话、`query_units` 57 → 19、目标本排第一。"""

    def test_walks_back_to_the_users_own_words(self):
        got = _last_user_text(_msgs(), for_goal=True)
        assert got == _USER_QUESTION, got

    def test_query_units_57_to_19(self):
        before = len(candidate_units(_old_last_user_text(_msgs())))
        after = len(candidate_units(_last_user_text(_msgs(), for_goal=True)))
        assert (before, after) == (57, 19), (before, after)

    def test_target_ledger_ranks_first(self):
        got = relevant_ledgers(_pool(), _last_user_text(_msgs(), for_goal=True))
        assert got, "修后必须有命中"
        assert got[0].ledger_id == "ldg-497d5e2d542b", [m.ledger_id for m in got]

    def test_walkback_is_visible_in_the_log(self):
        """`back=` 是本缺陷的直接命中计数 —— 修前它恒为 0（根本不走这条路）。"""
        with structlog.testing.capture_logs() as cap:
            _last_user_text(_msgs(), for_goal=True)
        ev = next((e for e in cap if e.get("event") == "agency_goal_envelope_stripped"), None)
        assert ev is not None and ev.get("back") == 3, cap
        assert "dsh_runtime_context" in (ev.get("skipped") or []), ev


class TestNegativeControl:
    """判据 ③：改回「最后一条 user」⇒ ① 必须红。"""

    def test_old_rule_takes_machine_text(self):
        old = _old_last_user_text(_msgs())
        assert old != _USER_QUESTION
        assert old.startswith("<system-reminder>")

    def test_old_rule_recalls_the_wrong_ledgers(self):
        got = relevant_ledgers(_pool(), _old_last_user_text(_msgs()))
        assert "ldg-497d5e2d542b" not in [m.ledger_id for m in got], [m.ledger_id for m in got]

    def test_without_the_machine_prefix_landing_the_walkback_stops_one_short(self):
        """第二个阴性对照：**只**往前走、不认 dsh 的运行时上下文 ⇒ 停在它身上。

        修法有两个变量（往前走 / 认这条裸文本形态），拆开各钉一次 ——
        只做前者时 live 实测 `query_units=50`、召回 `ldg-5cc65426473e 0.62`，
        仍是噪声命中（结算稿 §G19.0「两个变量」表）。
        """
        texts = [m["content"] for m in _msgs() if m["role"] == "user"]
        naive = next((t for t in reversed(texts) if strip_envelopes(t)[0].strip()), "")
        assert naive.startswith("Current runtime context."), naive[:60]
        assert naive != _USER_QUESTION


class TestMachineOnlyPrefix:
    """`_MACHINE_ONLY_PREFIXES` 的判别力与自指门。"""

    def test_runtime_context_block_is_not_user_text(self):
        cleaned, marks = _clean_user_text(_RUNTIME_CONTEXT)
        assert cleaned == ""
        assert "dsh_runtime_context" in marks

    def test_mentioning_it_mid_sentence_is_still_user_text(self):
        """🔴 自指门（MQ-A19 的教训）：**整条以它开头**才算，`in` 会把讨论它的话吃掉。"""
        raw = ("dsh 每轮都发一条 Current runtime context. This snapshot supersedes "
               "earlier runtime-context snapshots. 的消息，这条是不是该剥掉？")
        cleaned, marks = _clean_user_text(raw)
        assert cleaned == raw
        assert "dsh_runtime_context" not in marks


class TestAllEnvelopesFallback:
    """全剥空 ⇒ 退回现状（末条原文）+ 留可 grep 的告警。"""

    def test_falls_back_and_warns(self):
        msgs = [{"role": "user", "content": "<system-reminder>a</system-reminder>"},
                {"role": "user", "content": _RUNTIME_CONTEXT}]
        with structlog.testing.capture_logs() as cap:
            got = _last_user_text(msgs, for_goal=True)
        assert got == _RUNTIME_CONTEXT.strip()[:_GOAL_MAX_CHARS]
        ev = next((e for e in cap if e.get("event") == "agency_user_text_all_envelopes"), None)
        assert ev is not None and ev.get("users") == 2, cap


class TestRegressionOtherAgents:
    """判据 ④：CC / hermes 既有轨迹逐字不变（提问与信封在**同一条**消息里）。"""

    def test_claude_code_question_with_trailing_reminder_unchanged(self):
        raw = "继续检查刚刚的问题。\n<system-reminder>CLAUDE.md 全文…</system-reminder>"
        msgs = [{"role": "user", "content": raw}]
        assert _last_user_text(msgs, for_goal=True) == "继续检查刚刚的问题。"
        assert (len(candidate_units(_last_user_text(msgs, for_goal=True)))
                == len(candidate_units(_old_last_user_text(msgs))))

    def test_hermes_plain_text_unchanged(self):
        msgs = [{"role": "user", "content": "帮我深度分析一下美光吧"},
                {"role": "assistant", "content": "好的"},
                {"role": "user", "content": "再看泰安仁信"}]
        assert _last_user_text(msgs, for_goal=True) == "再看泰安仁信"
        assert _last_user_text(msgs) == "再看泰安仁信"

    def test_cc_session_wrapper_unchanged(self):
        """V-R1 的 `<session>` 形态（2026-08-25 病例）不受本次改动影响。"""
        raw = "<session>把这段改成异步</session>\n请先读 CLAUDE.md 再动手。"
        assert _last_user_text([{"role": "user", "content": raw}],
                               for_goal=True) == "把这段改成异步"

    def test_multimodal_list_content_still_joins_text_blocks(self):
        """多模态 list content 的拼法**逐字不变**：非 text 块贡献一个空串，
        于是图片位置留下一个多余空格（`这张图  里的表格`）。这是修前就有的形状，
        本卡不动它 —— 钉在这里是为了「不变」有据可查。"""
        msgs = [{"role": "user", "content": [{"type": "text", "text": "这张图"},
                                             {"type": "image"},
                                             {"type": "text", "text": "里的表格"}]}]
        assert _last_user_text(msgs, for_goal=True) == "这张图  里的表格"

    def test_no_user_message_is_empty(self):
        assert _last_user_text([{"role": "system", "content": "x"}], for_goal=True) == ""
        assert _last_user_text([]) == ""

    def test_for_goal_false_is_byte_for_byte_the_old_behaviour(self):
        """非 Goal 口径**一个字都不动**：不剥、不往前走、不截断。"""
        assert _last_user_text(_msgs()) == _SKILLS_REMINDER


class TestSingleImplementationPoint:
    """🔴 召回侧与 Goal 侧共用同一个取法 —— 不许分叉出第二个（MQ-A18/P7 的形状）。"""

    def test_agency_package_has_exactly_one_taker(self):
        from tests._source_probe import package_source
        src = package_source("agency")
        assert src.count("def _last_user_text(") == 1, "取法只许有一个定义"
        assert src.count("def _clean_user_text(") == 1, "清洗口径只许有一个定义"
        assert src.count("_last_user_text(") >= 3, "两个调用点 + 一个定义"
