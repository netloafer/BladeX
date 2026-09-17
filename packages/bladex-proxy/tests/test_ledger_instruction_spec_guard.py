"""MQ-L71 守卫：建本指令对 Core 的说法必须与模板/ADR 一致。

## 为什么需要机械守卫，而不是"改完就算了"

本条缺陷的形态原文写的是：**两边各说各话而没有任何守卫发现**。
`_FIRST_STEP_NO_LEDGER` 曾写 `"…and **optional** initial core/open/next entries"`，
而 `optional` 在 ADR-0032 §4.1（`Core=pinned`）与 `config/ledger-template.md`
（*"When the ledger is created this holds what the USER already told you"*）里
**都找不到依据** —— 是实现侧自己把规格放宽了。

改了措辞却不留守卫，`optional` 哪天被改回去照样没人发现 ——
那就是刚性原则 12 的形态（改实现但不留机械检查）。本表是那道检查。

🔴 **本表只钉规格一致性，不钉行为**。同 agent·同模型·同指令文本三跑
`bladex_ledger_update` 调用 **44 / 15 / 4**，任何"改完记账变多了"的断言都在噪声里。
"""

from __future__ import annotations

import re

from bladex_proxy.agency.runtime import (
    _FIRST_STEP_NO_LEDGER,
    _FIRST_STEP_WITH_LEDGER,
    _RECORD_ONLY_WITH_LEDGER,
)
from bladex_proxy.agency.notes import load_ledger_template


def _core_section(md: str) -> str:
    """模板里 `## Core` 那一段的正文（守卫的参照物 = 模板，不是常量抄一份）。"""
    m = re.search(r"^## Core\s*\n(.*?)(?=^## )", md, re.S | re.M)
    assert m, "模板里找不到 ## Core 段 —— 守卫的参照物没了，先修模板"
    return m.group(1)


# ── ① 建本指令：点名 core，且不得标为可选 ─────────────────────────────────


def test_create_instruction_names_core():
    """建本是 Core 一生**唯一**被要求填写的时刻，指令必须点到它。

    实测（2026-09-12 handoff #3 A 侧，改字前）：建本调用只带 goal，
    `g1/c0/v0/o0/n0`，其后 26 个主轮 Core 恒为 0。
    """
    assert "core" in _FIRST_STEP_NO_LEDGER.lower(), "建本指令没点名 core"


def test_create_instruction_does_not_call_core_optional():
    """🔴 回归钉：`optional` 在 ADR 与模板里都没有依据，不得回潮。"""
    txt = _FIRST_STEP_NO_LEDGER.lower()
    i = txt.find("core")
    window = txt[max(0, i - 80):i + 40]
    assert "optional" not in window, (
        "建本指令又把 core 说成 optional —— ADR-0032 §4.1 是 `Core=pinned`，"
        "模板是「建本时装用户已经告诉你的」，两处都没有这个依据")


def test_create_instruction_agrees_with_template_on_what_core_holds():
    """语义一致性：模板说 Core 建本时装**用户已经说过的**，指令也必须这么说。

    不做字符串相等（模板用户可编辑，措辞会变），只断言两边都把 Core 的
    建本内容系在"用户已陈述"这件事上。
    """
    tpl = _core_section(load_ledger_template()).lower()
    assert "user" in tpl and "already told you" in tpl, \
        "模板里 Core 的建本语义变了 —— 先确认是有意修订，再同步本守卫与指令"
    assert "already" in _FIRST_STEP_NO_LEDGER.lower(), \
        "建本指令没把 Core 的内容系在「用户已经说过的」上，与模板脱节"


# ── ② 🔴 反面：稳态指令**不**点名 core（这是已登记的结构缺口 MQ-L72）──────


def test_steady_state_instructions_still_omit_core_is_registered_not_fixed():
    """本条**不是要求**，是把 MQ-L72 的现状钉住，防止它被静默改掉而无人知道。

    实测：`verified` / `open` / `next` 在两条稳态指令里各被点名 3 次，
    **`core` 0 次** ⇒ Core 一生只有建本那一次机会。

    🔴 L72 的修法（空段带回模板说明）**已于 2026-09-12 撤回**：
    同日 handoff 实测 B 从**零 Core** 的账本接手，N4 四项与有 4 条 Core 的 #1 持平
    ⇒ 危害未证，而 ADR-0029 的规矩是加注入者举证。

    **若哪天这条断言变红**，说明有人给稳态指令加了 core —— 那不是坏事，
    但必须带着收益读数来（判据见 MQ-L72「什么读数出现才动手」），并同步改本表。
    """
    for name, txt in (("_FIRST_STEP_WITH_LEDGER", _FIRST_STEP_WITH_LEDGER),
                      ("_RECORD_ONLY_WITH_LEDGER", _RECORD_ONLY_WITH_LEDGER)):
        assert "core" not in txt.lower(), (
            f"{name} 现在点名了 core —— 现状变了。这需要收益读数支撑，"
            f"见 MQ-L72；确认后请同步本守卫")
