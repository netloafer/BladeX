"""批 HO（MQ-L99，拍板 (43)）：已关账本按内容进相关性段——交接不再断。

病例（10-15 交接测试第 3、4 步）：codex 做到一半，用户说「先到这里，剩下的交给别人收尾」，
codex 把它当做完关了本；Claude Code 新会话首句描述那件事 ⇒ 相关性段为空，另建新本。
E4（MQ-L98）的「本会话近期关闭」只对关它的那个会话列出，换会话、换 agent 必然不满足。

判据（`prompt-batch-HO-20261015.md` §3）：
- **能接上**：agent X 建本 → 关本 → agent Y 新会话同项目说一句描述那件事的话 ⇒ Y 首轮在
  相关性段看得到它且标 closed；切过去即重开；Goal / Core 逐字不变；不新建。
- **渲染**：已关那一行行尾 `[closed by <agent> <日期 时刻> UTC; switching to it reopens it]`，
  open 行不带；措辞只陈述事实。
- **recency 段维持排除 closed**（没有内容依据，Z3 理由在那一段仍成立）。
- **去重**：E4 小节已并入，同一本只在相关性段出现一次。
- 收紧（由回放读数定）：被取代的不列、关闭超 24 h 的不列——core 侧用例在
  `bladex-core/tests/test_ledger_relevant_candidates.py`。
"""

from __future__ import annotations

import asyncio
import datetime as dt
import re

import pytest
from bladex_core.ledger import STATUS_ACTIVE, STATUS_CLOSED, new_ledger
from bladex_core.ledger_runtime import activation_scope
from bladex_proxy.agency import _MATCHING_LEDGERS_HEADER, AgencyRuntime

PROJECT = "p-10ad1f61d0c2"
X, Y = "codex", "claude-code"
S_X = "01a13e7c-e5e1-7262-baec-ddafaeead12d"
S_Y = "46f8b00b-98a7-4109-9528-7b829ae22123"
#: 10-15 第 3 步用户原话（截短）与第 4 步接手方首句，逐字取自 Hub。
START_TEXT = ("packages/bladex-proxy/tests/test_probe_member_blindspot.py 里有一条守卫，只看 "
              "pmc.CORPUS_CAP / pmc.SEED 这两个名字有没有出现：把脚本改成 pmc.CORPUS_CAP * 2，"
              "它照样绿。先只做第一段：写一条按行为判的测试，并确认改成 * 2 时它会红。")
STOP_TEXT = "先到这里，剩下的交给别人收尾。"
HANDOFF_TEXT = "codex 刚才在给盲区尺补一条按行为判的守卫，做到一半，你接着收尾"
TITLE = "probe_member_blindspot 行为守卫测试（第一段）"
CORE = ["守卫只看名字：pmc.CORPUS_CAP * 2 时仍绿", "先只做第一段，不提交"]


@pytest.fixture(autouse=True)
def _modules_on(monkeypatch):
    for m in ("TOOLFACE", "LEDGER", "INTERCEPTION"):
        monkeypatch.setenv(f"BLADEX_MODULE_{m}", "1")
    monkeypatch.setenv("BLADEX_LEDGER_TIERS", "strong,medium")


def _ctx(agent: str, session: str, user: str, turn_index: int = 1) -> dict:
    return {"session_id": session, "agent_id": agent, "project_id": PROJECT,
            "turn_index": turn_index, "user_query": user}


def _run(coro):
    return asyncio.run(coro)


def _switch(ag, args, **ctx):
    return _run(ag._h_ledger_switch(args, allowed_exposure="public", context=_ctx(**ctx)))


def _close(ag, args, **ctx):
    return _run(ag._h_ledger_close(args, allowed_exposure="public", context=_ctx(**ctx)))


def _block(ag, agent: str, session: str, text: str) -> str:
    msgs = [{"role": "system", "content": f"You are {agent}."},
            {"role": "user", "content": text}]
    out = ag.insert_ledger_block(msgs, agent, project_id=PROJECT, tier="strong",
                                 with_instruction=True, session_id=session)
    return out[-1]["content"]


def _noise(ag: AgencyRuntime, n: int = 6) -> None:
    """池里放几本无关的 open 本（idf 有分母；也把 recency 段填满）。"""
    for i, (t, g) in enumerate([
        ("MQ-A78 兼容修正提交", "陈旧 pidfile 升级兼容，status 只读"),
        ("build_public_tree 自检补 scripts 死链对账", "README 点名的 scripts 必须在白名单里"),
        ("拆分探针路径加载用例到独立测试文件", "u12 与 flash 行为轨迹的探针用例拆出"),
        ("泰山啤酒破产重整最新进展梳理", "梳理泰山啤酒破产重整的最新进展"),
        ("GitHub freetoken 项目调研", "调研 GitHub 上的 freetoken 项目"),
        ("北京今日天气查询", "查询北京今日天气"),
    ][:n]):
        lid = f"ldg-noise{i:07d}"
        ag.pool[lid] = new_ledger(ledger_id=lid, title=t, goal=g, goal_source="model",
                                  created_at=f"2026-10-0{i + 1}T00:00:00+00:00")


def _handoff_setup() -> tuple[AgencyRuntime, str]:
    """X 建本（带 Core）→ 关本（evidence = 交出方用户原话）。返回 (runtime, ledger_id)。"""
    ag = AgencyRuntime()
    ag._emit = lambda *_a, **_k: None  # noqa: SLF001
    _noise(ag)
    _switch(ag, {"ledger_id": "", "title": TITLE, "core": CORE},
            agent=X, session=S_X, user=START_TEXT)
    lid = ag.activation.active(activation_scope(X, PROJECT))
    assert lid and ag.pool[lid].title == TITLE
    assert f"Ledger {lid} closed." in _close(ag, {"evidence": "先到这里"},
                                             agent=X, session=S_X, user=STOP_TEXT, turn_index=2)
    assert ag.pool[lid].status == STATUS_CLOSED
    return ag, lid


# ── 能接上 ─────────────────────────────────────────────────────────────────

def test_handoff_receiver_sees_closed_ledger_and_switching_reopens_verbatim():
    ag, lid = _handoff_setup()
    goal_before, core_before = ag.pool[lid].goal, [e.text for e in ag.pool[lid].entries("core")]
    rev_before = ag.pool[lid].rev
    assert goal_before and core_before == CORE

    content = _block(ag, Y, S_Y, HANDOFF_TEXT)
    assert _MATCHING_LEDGERS_HEADER in content
    line = next(ln for ln in content.splitlines() if ln.startswith(f"- {lid}:"))
    assert "[closed by codex " in line and line.endswith("; switching to it reopens it]"), line

    n_before = len(ag.pool)
    out = _switch(ag, {"ledger_id": lid, "evidence": "你接着收尾"},
                  agent=Y, session=S_Y, user=HANDOFF_TEXT)
    assert "(It was closed; switching reopened it.)" in out
    led = ag.pool[lid]
    assert led.status == STATUS_ACTIVE and led.rev == rev_before
    assert led.goal == goal_before
    assert [e.text for e in led.entries("core")] == core_before
    assert ag.activation.active(activation_scope(Y, PROJECT)) == lid
    assert len(ag.pool) == n_before, "接手不该新建账本"


def test_closed_by_cli_is_listed_without_agent():
    ag, lid = _handoff_setup()
    _switch(ag, {"ledger_id": lid}, agent=X, session=S_X, user=START_TEXT, turn_index=3)  # 重开
    ag.lifecycle_close(lid, origin="cli", context=None)        # 再由 CLI 关（无会话、无 agent）
    line = next(ln for ln in _block(ag, Y, S_Y, HANDOFF_TEXT).splitlines()
                if ln.startswith(f"- {lid}:"))
    assert "[closed " in line and " by " not in line, line


# ── 渲染：一眼分清 open / closed，措辞不劝切 ────────────────────────────────

def test_open_rows_carry_no_closed_marker_and_wording_is_factual():
    ag, lid = _handoff_setup()
    # 再放一本内容相近的 open 本，让相关性段同时有 open 与 closed 两行
    ag.pool["ldg-open0000001"] = new_ledger(
        ledger_id="ldg-open0000001", title="probe_member_blindspot 行为守卫测试（第二段）",
        goal="把 probe_member_blindspot 盲区尺按行为判的守卫登记进 gate", goal_source="model",
        created_at="2026-10-15T08:00:00+00:00")
    rows = [ln for ln in _block(ag, Y, S_Y, HANDOFF_TEXT).splitlines()
            if re.match(r"^- ldg-\S+: .* \[\d\.\d\d\]", ln)]     # 相关性段的行（带分数）
    closed_rows = [r for r in rows if "[closed" in r]
    open_rows = [r for r in rows if "[closed" not in r]
    assert [r.split(":")[0] for r in closed_rows] == [f"- {lid}"]
    assert any(r.startswith("- ldg-open0000001:") for r in open_rows)
    marker = closed_rows[0][closed_rows[0].index("[closed"):]
    for word in ("should", "must", "do not", "instead", "recommended"):
        assert word not in marker.lower(), marker


def test_closed_ledger_appears_once_and_e4_section_is_gone():
    ag, lid = _handoff_setup()
    # 关本的那个会话自己回来（L98 的形状）：同样只在相关性段出现一次
    for agent, sess in ((X, S_X), (Y, S_Y)):
        content = _block(ag, agent, sess, HANDOFF_TEXT)
        assert content.count(f"- {lid}:") == 1, content
        assert "Ledgers you closed in this session recently" not in content
        bd = ag.last_ledger_breakdown
        assert bd is not None and "closed_recent" not in bd.chars


# ── recency 段维持排除 closed ──────────────────────────────────────────────

def test_recency_section_still_excludes_closed_without_content_match():
    ag, lid = _handoff_setup()
    content = _block(ag, Y, S_Y, "看一下 git status")         # 与已关本无内容关系
    assert lid not in content
    assert "Other recent ledgers" in content, "夹具前提：recency 段在"


def test_superseded_or_stale_closed_ledger_is_not_listed():
    ag, lid = _handoff_setup()
    old = (dt.datetime.now(dt.UTC) - dt.timedelta(hours=25)).isoformat(timespec="seconds")
    ag.pool[lid] = ag.pool[lid].model_copy(update={"closed_at": old})
    assert lid not in _block(ag, Y, S_Y, HANDOFF_TEXT)
    fresh = (dt.datetime.now(dt.UTC) - dt.timedelta(hours=23)).isoformat(timespec="seconds")
    ag.pool[lid] = ag.pool[lid].model_copy(update={"closed_at": fresh})
    assert lid in _block(ag, Y, S_Y, HANDOFF_TEXT)
    ag.pool[lid] = ag.pool[lid].model_copy(update={"superseded_by": "ldg-noise0000000"})
    assert lid not in _block(ag, Y, S_Y, HANDOFF_TEXT)


def test_candidate_log_names_closed_matches():
    import structlog.testing
    ag, lid = _handoff_setup()
    with structlog.testing.capture_logs() as cap:
        _block(ag, Y, S_Y, HANDOFF_TEXT)
    ev = next(e for e in cap if e["event"] == "agency_ledger_candidates")
    assert ev["relevant_closed"] == [lid] and lid in ev["relevant"]
