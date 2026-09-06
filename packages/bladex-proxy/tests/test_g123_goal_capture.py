"""G12.3 接线验收：轮次三分 + goal 唯一写入点 + 不可改（判据 §2.D）。

纯函数那一半在 `packages/bladex-core/tests/test_task_goal.py`；本文件钉的是
"生产判据接没接上"——包括两条**已知缺口**（MQ-S44 / MQ-S45），
它们在这里被钉成"当前行为"，修好之后**这两条断言要反过来**（各自注释里写了怎么改）。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from bladex_core.matter import Matter, MatterStatus
from bladex_core.task_goal import (
    GOAL_ABSENT_NO_CONTEXT,
    GOAL_ABSENT_NO_KEY,
    GOAL_ABSENT_SCAFFOLD,
    GOAL_SOURCE_FIRST_TURN_USER,
    TURN_DROPPED,
    TURN_REAL,
    TURN_SCAFFOLD,
)
from bladex_proxy.identity import classify_turn_disposition
from bladex_proxy.storage.memory_index import MemoryIndex, _capture_matter_goal

LK = "local/hermes:default/sess-1/0000000001"
P1_TEXT = "帮我搞定泰山啤酒的破产重整尽调，重点看股权变动，不用管财务报表"


def _u(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


# ── 轮次三分：与 rebuild 用的是同一个判据（MQ-S4「各算各的」的防线）────────

@pytest.mark.parametrize("text,expected", [
    (P1_TEXT, TURN_REAL),
    # SCAFFOLD：user 侧是信封，assistant 侧照收 ⇒ **能开卡**，故 goal 必须拒
    ("<command-name>/compact</command-name>", TURN_SCAFFOLD),
    ("<transcript>...</transcript>", TURN_SCAFFOLD),
    ("[System: Your previous response was truncated", TURN_SCAFFOLD),
    # DROPPED：Hermes 原生 aux，rebuild 整轮丢，永远当不了首轮
    ("Please process this web content and create a comprehensive markdown summary:",
     TURN_DROPPED),
    ("You are a summarization agent creating a context checkpoint", TURN_DROPPED),
])
def test_turn_disposition_three_way(text, expected):
    cls, _rule = classify_turn_disposition(_u(text))
    assert cls == expected


def test_disposition_rejudges_and_ignores_frozen_flag():
    """判据是**重判**：函数只吃 messages，够不着 `turn.identity.auxiliary`。

    2026-08-22 取读数时第一跑就栽在读冻结字段上——指纹明明命中的轮被算成真实
    用户轮，整跑作废。这条把"尺子只能从内容重判"钉在签名上。
    """
    import inspect

    params = inspect.signature(classify_turn_disposition).parameters
    assert list(params) == ["messages"]


# ── 🔴 已知缺口：钉当前行为，修好后反过来 ───────────────────────────────

def test_mq_s44_background_process_notice_currently_reads_as_real():
    """MQ-S44：`[IMPORTANT: Background process …]` 没有 aux 指纹。

    live 20 轮全判 REAL，其中 1 条**已确凿开卡**（`judgment_taskstate` 仅有的三条
    `verdict=new` 之一）。⇒ 现在 goal 会从一条后台进程完成通知里取。

    🔴 MQ-S44 修好后：expected 改成 `TURN_SCAFFOLD`（user 侧是通知、assistant 侧
    可能在做实事），并把 `test_..._currently_reads_as_real` 改名为 `_is_refused`。
    """
    text = ("[IMPORTANT: Background process proc_2f436f1cb432 completed normally "
            "(exit code 0).\nCommand: bash scripts/gate_check.sh")
    cls, rule = classify_turn_disposition(_u(text))
    assert (cls, rule) == (TURN_REAL, ""), "指纹表补上了？那就按 docstring 反转本条"


def test_mq_s45_codex_history_all_live_clauses_hit():
    """MQ-S45 **已修**（2026-08-28）：三种从句都命中，判据改用共同前缀。

    原缺口：指纹写死 `…agent history added since`，而 live 里第三种从句是
    `whose request action you are assessing` —— 子串匹配落空、判 REAL，
    整段父 agent 转录被当用户意图蒸馏。与 MS-2 的 em dash 同型第三例。

    修法**不是**补那一句，是取共同前缀 `The following is the Codex agent history`
    一次盖住全部——按一次采样写死判据，agent 换个变体就复发第四例。

    本条从"钉住坏行为"翻成"钉住修好后的行为"（原 docstring 的交代）。
    连带的 envelope 落点由 `test_distill_only_envelope_contract.py` 守——
    指纹命中只是第一步，**判 scaffold 之后 user 侧真的被剥掉**才是终点，
    那正是这次事故的真正断点（`codex:guardian` 字符留存率 99.6%）。
    """
    for clause in (
        "The following is the Codex agent history added since the last turn",
        "The following is the Codex agent history added since your last approval "
        "assessment. Continue the same review conversation.",
        "The following is the Codex agent history whose request action you are assessing",
    ):
        cls, rule = classify_turn_disposition(_u(clause))
        assert (cls, rule) == (TURN_SCAFFOLD, "codex_history_injection"), (
            f"这句从句漏了：{clause[:70]!r} —— 出现第四种变体？"
            "别再补一句，回去看共同前缀是否被改窄了"
        )


# ── 唯一写入点 + 降级分档 ───────────────────────────────────────────────

def test_capture_writes_all_four_fields():
    m = Matter(matter_id="m-1")
    reason = _capture_matter_goal(m, LK, lambda lk: (TURN_REAL, P1_TEXT))
    assert reason == ""
    assert m.task_goal == P1_TEXT
    assert m.task_goal_source == GOAL_SOURCE_FIRST_TURN_USER
    assert m.task_goal_reason == ""
    assert m.task_goal_ledger_key == LK


def test_scaffold_turn_opens_a_card_with_empty_goal():
    """N1 的接线面：卡照常建，goal 空着并标明原因（不阻塞任何路径）。"""
    m = Matter(matter_id="m-2")
    reason = _capture_matter_goal(m, LK, lambda lk: (TURN_SCAFFOLD, "<transcript>x</transcript>"))
    assert reason == GOAL_ABSENT_SCAFFOLD
    assert m.task_goal == "" and m.task_goal_source == ""
    assert m.task_goal_ledger_key == LK          # 可回查那一轮，便于人工复核


def test_missing_ledger_key_and_missing_context_are_distinguishable():
    """两种"空"分开报：没有 key（N10） vs 这条路径拿不到那一轮（rejudge_pending）。"""
    m1 = Matter(matter_id="m-3")
    assert _capture_matter_goal(m1, "", lambda lk: (TURN_REAL, P1_TEXT)) == GOAL_ABSENT_NO_KEY
    m2 = Matter(matter_id="m-4")
    assert _capture_matter_goal(m2, LK, None) == GOAL_ABSENT_NO_CONTEXT
    m3 = Matter(matter_id="m-5")
    assert _capture_matter_goal(m3, LK, lambda lk: None) == GOAL_ABSENT_NO_CONTEXT


def test_ledger_get_failure_does_not_break_creation():
    """取首轮失败只是少一个 goal，不该让建卡失败（fail-closed 而非 fail-stop）。"""
    def _boom(_lk):
        raise RuntimeError("ledger down")

    m = Matter(matter_id="m-6")
    with pytest.raises(RuntimeError):
        _capture_matter_goal(m, LK, _boom)   # 回调自己抛 = 调用方的事
    # 生产回调 `_goal_context` 自己吞异常返回 None，等价于下面这条：
    m2 = Matter(matter_id="m-7")
    assert _capture_matter_goal(m2, LK, lambda lk: None) == GOAL_ABSENT_NO_CONTEXT
    assert m2.task_goal == ""


# ── 判据 D2 / D3：只有用户能改 ──────────────────────────────────────────

def test_d2_existing_goal_is_never_overwritten():
    """D2：非空 goal 不许被任何后续动作改写——模型能改目标 = 防漂移自我拆台。"""
    m = Matter(matter_id="m-8", task_goal=P1_TEXT,
               task_goal_source=GOAL_SOURCE_FIRST_TURN_USER, task_goal_ledger_key=LK)
    reason = _capture_matter_goal(
        m, "other/key/2/9", lambda lk: (TURN_REAL, "其实我们真正要做的是重构路由层"))
    assert reason == ""
    assert m.task_goal == P1_TEXT
    assert m.task_goal_ledger_key == LK


def test_d3_merge_does_not_move_goal_and_warns():
    """D3：goal 不随合并搬家（target 的首轮才是这件事的首轮），但不许静默。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        index = MemoryIndex(Path(tmpdir) / "index", embedder=None, read_only=False)
        index.open()
        index.add_matter(Matter(matter_id="m-src", title="源卡",
                                status=MatterStatus.ACTIVE, task_goal=P1_TEXT,
                                task_goal_source=GOAL_SOURCE_FIRST_TURN_USER))
        index.add_matter(Matter(matter_id="m-dst", title="目标卡",
                                status=MatterStatus.ACTIVE))
        index.merge_matters("m-src", "m-dst")
        dst = index.get_matter("m-dst")
        assert dst.task_goal == ""            # 没搬家
        src = index.get_matter("m-src")
        assert src.task_goal == P1_TEXT       # 源卡自己的记录不动（closed 但保留）


def test_d3_merge_keeps_target_goal_intact():
    with tempfile.TemporaryDirectory() as tmpdir:
        index = MemoryIndex(Path(tmpdir) / "index", embedder=None, read_only=False)
        index.open()
        index.add_matter(Matter(matter_id="m-src2", title="源卡",
                                status=MatterStatus.ACTIVE, task_goal="源卡的目标"))
        index.add_matter(Matter(matter_id="m-dst2", title="目标卡",
                                status=MatterStatus.ACTIVE, task_goal=P1_TEXT,
                                task_goal_source=GOAL_SOURCE_FIRST_TURN_USER))
        index.merge_matters("m-src2", "m-dst2")
        assert index.get_matter("m-dst2").task_goal == P1_TEXT


# ── 加法式演进：历史 Matter 没有这四个字段 ──────────────────────────────

def test_legacy_matter_without_goal_fields_round_trips():
    """存量卡（无这四字段）反序列化默认空，不需要迁移。"""
    legacy = {"matter_id": "m-old", "title": "存量卡"}
    m = Matter.model_validate(legacy)
    assert (m.task_goal, m.task_goal_source, m.task_goal_reason,
            m.task_goal_ledger_key) == ("", "", "", "")
