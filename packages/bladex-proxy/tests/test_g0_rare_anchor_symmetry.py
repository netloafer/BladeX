"""稀有锚的路径类剔除**双侧对称**（MQ-A52 ① / 批 G G0.2，2026-09-09）。

## 修的是什么

`hits_rare = pending_rare ∩ seen`。路径类剔除（A51，`path_like_units`）此前**只加在
`pending` 一侧**，`seen`（首动作正文 + 工具参数前 2000 字符）一侧原样。于是：

    账本侧："6项遗漏待Hermes补全：版本修正、CHANGELOG Unreleased变更"  ← 散文里的 hermes
    首动作："read_file(~/Documents/Hermes/CHANGELOG.md)"              ← 路径里的 hermes

账本侧按设计不剔（它确实不在任何路径里），首动作侧不剔（漏了），两边一撞就算命中。
live 09-08（`proxy-20260908-034258.log`）：`bucket=work ∧ usable=True` 22 行里
`hits_rare>0` 17 行，其中 **11 行的 `rare_sample` 就是 `hermes` 一个词** ——
77% 的"引用了待办"有 2/3 靠一个 agent 名撑着。

**同一条判据只用在一半上** —— 与 MQ-L53 第二类报错犯的是同一个错，同一周内两次。

## 为什么只作用在 rare 轴

`hits` / `sample` 是 A47 之前的旧口径，并列跑不作废（口径变更要能对照，
ADR-0029 §4）；路径类剔除本就是 rare 口径的一半（A51 与 df 并联）。
在旧轴上也剔 = 修前修后的基线不可比。

## 本卡**不做**的两件（A52 ②③，等 G-读② 的 A2 基线出来再拍）

② 参与者名从 `flash_agents_roster` 取闭集单独成类剔除；
③ `rare_units/next_units` 若仍≈1，说明 df 那一条在这类账本上根本不该参与。
"""

from __future__ import annotations

import types

import structlog.testing
from bladex_core.ledger import ACTOR_MODEL, LedgerEntry, add_entry, new_ledger
from bladex_core.ledger_runtime import candidate_units
from bladex_proxy.models import LedgerAnchor, ToolEvent
from bladex_proxy.server.orchestration import _ledger_next_referenced

_LID = "ldg-a52"
#: live 病例逐字（`ldg-9831b65622a6` 的两条待办，`hermes` 全在散文里、零路径）。
_OPEN = "6项遗漏待Hermes补全：版本修正、CHANGELOG Unreleased变更、MCP形态、工具清单"
_NEXT = "等待Hermes补全后再次核验"


def _pool():
    led = new_ledger(ledger_id=_LID, title="Hermes 文档补全", goal="补齐 6 项遗漏")
    led = add_entry(led, "open", LedgerEntry(text=_OPEN), actor=ACTOR_MODEL)
    return {_LID: add_entry(led, "next", LedgerEntry(text=_NEXT), actor=ACTOR_MODEL)}


def _run(*, first_action: tuple[str, dict] | None = None, response_text: str = ""):
    """跑一遍仪器，`pending_rare` 走**生产的** `rare_pending_units` 现算。

    不手写稀有集：手写就等于在测试里另立一份账本侧口径，两份判据迟早分叉
    （本仓反复付过学费的形状）。池内只有一本 ⇒ df 语料 <2 ⇒ 那一条按设计退化成
    "只做路径类剔除"，而这本账本的待办**没有路径** ⇒ `hermes` 在账本侧留下来,
    正是 live 的形状。
    """
    from bladex_proxy.agency.runtime import rare_pending_units
    pool = _pool()
    led = pool[_LID]
    pending_text = " ".join(e.text for sec in ("next", "open")
                            for e in led.entries(sec))
    units = frozenset(candidate_units(pending_text))
    rare = rare_pending_units(pool, units, pending_text=pending_text)
    assert "hermes" in rare, "前提坍了：账本侧的 hermes 本就该留着（散文，非路径）"

    events = ([ToolEvent(tool_name=first_action[0], direction="call",
                         arguments=first_action[1])] if first_action else [])
    req = types.SimpleNamespace(
        state=types.SimpleNamespace(bladex_pending_units=units,
                                    bladex_pending_rare=rare),
        app=types.SimpleNamespace(
            state=types.SimpleNamespace(
                agency=types.SimpleNamespace(pool=pool))))
    anchor = LedgerAnchor(ledger_id=_LID, rev=led.rev, sections={"next": 40})
    with structlog.testing.capture_logs() as cap:
        _ledger_next_referenced(req, anchor, response_text, events)
    recs = [e for e in cap if e["event"] == "agency_ledger_next_referenced"]
    assert len(recs) == 1
    return recs[0]


# ── ① 病例回放：账本侧散文 `hermes` × 首动作侧路径 `hermes` ⇒ 不计命中 ────────


def test_agent_name_from_a_path_argument_is_not_a_rare_hit():
    r = _run(first_action=("read_file",
                           {"file_path": "~/Documents/Hermes/CHANGELOG.md"}))
    assert r["bucket"] == "work" and r["usable"] is True, "分桶前提"
    assert r["hits_rare"] == 0, \
        "首动作里的 hermes 是路径带进来的寻址片段，对'引用了待办'零判别力"
    assert "hermes" not in r["rare_sample"]
    # 🔴 旧轴**必须**照旧命中：路径类剔除只作用在 rare 轴，两个口径并列可比。
    assert r["hits"] >= 1 and "hermes" in r["sample"], \
        "旧口径被一起改了 ⇒ 修前修后的基线不可比（ADR-0029 §4）"


def test_the_same_word_in_prose_still_counts():
    """阳性对照：首动作里 `Hermes` **不在路径里**时，它照样是命中。

    没有这一条，① 全绿也可能只是因为"hermes 被整体拉黑了"——那是另一个缺陷
    （手写词表，正是 MQ-S 那条要防的形态）。
    """
    r = _run(response_text="我先去问 Hermes 那 6 项遗漏各自缺什么")
    assert r["hits_rare"] >= 1 and "hermes" in r["rare_sample"]


def test_a_real_anchor_beside_a_path_argument_survives():
    """第二条阳性对照：真锚与底噪同处一个参数时，**只**剔路径带进来的那些。

    `path_like_units` 按片段切（空白/标点为界），所以同一个参数里
    `~/Documents/Hermes/CHANGELOG.md` 整片剔掉、`版本修正` 原样留下。
    没有这一条，① 的"剔"就可能是把整个首动作拉黑（那会让 `hits_rare` 恒为 0，
    尺子从此永远读 0 —— 仪器骗人的第一类）。
    """
    r = _run(first_action=("terminal",
                           {"command": "grep 版本修正 ~/Documents/Hermes/CHANGELOG.md"}))
    assert r["hits_rare"] >= 1, "非路径部分的锚被误剔了"
    assert "hermes" not in r["rare_sample"]


# ── ③ 已知漏口：裸目录路径（MQ-A54，2026-09-09 本卡实测发现，登记不改）────────


def test_bare_directory_path_still_leaks_the_agent_name():
    """🔴 **`ls ~/Documents/Hermes/` 这类裸目录路径漏网** —— 现状钉在这里。

    `looks_like_path` 要求末段带扩展名（`v.endswith("/")` 直接判 False），
    所以"目录、没有文件名"的片段整片不算路径 ⇒ `hermes` 照样进 `seen_rare`。

    为什么现在只登记不改（MQ-A54）：放宽末段判据会把 `/Users/jasonye` 这类
    真有判别力的东西一起吃掉，那正是 A51 当初收紧它的原因。**首动作侧比账本侧
    更容易出现裸目录**（`ls` / `cd` / `rg <dir>`），所以这是双侧对称之后才浮出来的
    新形态，不是 A51 的旧代价——处置要凭 G-读② 的 A2 基线里它占多少来拍。

    本条是**现状断言**：将来真改了，它会红，而不是静默改变基线口径。
    """
    r = _run(first_action=("terminal", {"command": "ls ~/Documents/Hermes/"}))
    assert r["hits_rare"] >= 1 and "hermes" in r["rare_sample"], \
        "行为变了 ⇒ MQ-A54 被处置了，请连同基线口径一起更新本条"


# ── ② 判别力：只在单侧剔（= 修前）⇒ ① 必红 ─────────────────────────────────


def test_discriminative_one_sided_removal_lets_the_agent_name_through(monkeypatch):
    """把首动作侧的剔除拿掉（`path_like_units` 恒空）⇒ `hermes` 又算命中。

    这一条证明**这个用例本身有区分度**：没有它，① 全绿也可能只是因为
    `~/Documents/Hermes/CHANGELOG.md` 压根没被判成路径
    （feedback_instrument_reference_frame：先证明那个桶可达）。

    🔴 打**消费方模块**（`bladex_core.ledger_runtime`，函数体内 import 的那个），
    F0 拆包后打门面不生效。账本侧的 `rare` 已在 `_run` 里算好停进 `request.state`,
    不受本次 patch 影响 —— 于是这里的形态精确等于修前的"只在账本侧剔"。
    """
    import bladex_core.ledger_runtime as lr
    monkeypatch.setattr(lr, "path_like_units", lambda _text: frozenset())
    r = _run(first_action=("read_file",
                           {"file_path": "~/Documents/Hermes/CHANGELOG.md"}))
    assert r["hits_rare"] >= 1 and "hermes" in r["rare_sample"], \
        "单侧剔除下必须放行 —— 否则本文件测的不是这条修法"
