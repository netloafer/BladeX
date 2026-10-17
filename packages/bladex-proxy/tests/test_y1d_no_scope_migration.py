"""批 Y · Y1d（拍板 (30)，MQ-L93，2026-09-28）：停掉惰性作用域迁移。

删的是 `agency/handlers._maybe_migrate_binding`（08-29，MQ-L30 收尾）：「项目键无绑定 ∧
同 agent Global 有绑定 ∧ 本请求有项目信号 ⇒ 把 Global 绑定搬到项目键」。T1 §7a 实测全库
4 次迁移 **3 次跨项目**——它是一个替模型改写归属、又没有举证的启发式（刚性原则 15）。

判据（`prompt-batch-Y1d-20260926.md` §3）：
  ① 任何请求路径都不再写 `ledger_scope_migrate`：注入面（三协议唯一调用点
     `orchestration.insert_ledger_block`）+ 拦截内循环（工具 handlers 的 `_ledger_for`）；
     静态守卫：生产源码里不许再出现迁移调用 / 该事件的发射（阴性对照：加回 ⇒ 红）。
  ③ 项目键落空时，Global 那本**在候选里**（置顶进 recency 段、带标注），一次
     `bladex_ledger_switch` 接上；注入措辞不诱导新建（阴性对照：过滤掉 ⇒ 红）。
  ④ 无项目信号的 agent（Pi / hermes 查 Global）行为逐字不变。
  ⑤ 删干净（AST 级：无同名函数定义、无 `.migrate(` 调用）。
"""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path

import pytest
from bladex_core.ledger import new_ledger
from bladex_core.ledger_runtime import activation_scope
from bladex_proxy.agency import AgencyRuntime
from bladex_proxy.agency.runtime import _FIRST_STEP_NO_LEDGER, _GLOBAL_BOUND_SUFFIX

_ROOT = Path(__file__).resolve().parents[3]
_AGENT = "codex"
_PROJECT = "p-g3bench00001"
_OLD = "ldg-globalold001"
_MSGS = [{"role": "system", "content": "sys"},
         {"role": "user", "content": "帮我看看 g3 bench 的评测脚本报错"}]


@pytest.fixture(autouse=True)
def _modules_on(monkeypatch):
    monkeypatch.setenv("BLADEX_MODULE_LEDGER", "1")
    monkeypatch.setenv("BLADEX_MODULE_TOOLFACE", "1")
    monkeypatch.setenv("BLADEX_LEDGER_FACE_SPLIT", "1")


def _rt(*, global_bound: bool = True, newer: int = 6) -> tuple[AgencyRuntime, list[dict]]:
    """池：Global 上那本很老（recency 排第 newer+1，自然 top-5 挤不进去）+ newer 本无关新本。"""
    rt = AgencyRuntime(index=None, hub=None)
    events: list[dict] = []
    rt._emit = lambda t, p: events.append({"type": t.value, "payload": p})
    old = new_ledger(ledger_id=_OLD, title="BladeX 账本切换防抖复盘", goal="复盘账本切换防抖",
                     goal_source="user", created_at="2026-08-25T10:00:00+00:00")
    rt.pool[old.ledger_id] = old
    for i in range(newer):
        lg = new_ledger(ledger_id=f"ldg-newer00000{i}", title=f"无关任务 {i}", goal=f"无关 {i}",
                        goal_source="user", created_at=f"2026-09-2{i}T10:00:00+00:00")
        rt.pool[lg.ledger_id] = lg
    if global_bound:
        rt.activation.restore({activation_scope(_AGENT, ""): _OLD})
    return rt, events


def _block(rt: AgencyRuntime, project_id: str, msgs: list[dict] | None = None, **kw) -> str:
    base = [dict(m) for m in (msgs or _MSGS)]
    out = rt.insert_ledger_block([dict(m) for m in base], _AGENT, project_id=project_id,
                                 with_instruction=True, tier="strong", session_id="sess-1", **kw)
    return "\n".join(m["content"] for m in out if m not in base)


def _ctx(project_id: str) -> dict:
    return {"agent_id": _AGENT, "project_id": project_id, "session_id": "sess-1"}


# ── ① 任何请求路径都不再写 ledger_scope_migrate ────────────────────────────────

def test_injection_path_does_not_migrate():
    rt, events = _rt()
    _block(rt, _PROJECT)
    assert events == [], "注入面写了事件（改前这里是 ledger_scope_migrate）"
    assert rt.activation.active(activation_scope(_AGENT, "")) == _OLD, "Global 绑定被搬走"
    assert rt.activation.active(activation_scope(_AGENT, _PROJECT)) == "", "项目键被静默填上"


def test_inner_loop_tool_paths_do_not_migrate():
    """拦截内循环：读 / 写 / 切 三个账本工具的 `_ledger_for` 此前都先做迁移。"""
    rt, events = _rt()
    read = asyncio.run(rt._h_ledger_read({}, allowed_exposure="local", context=_ctx(_PROJECT)))
    upd = asyncio.run(rt._h_ledger_update(
        {"section": "verified", "op": "add", "text": "x"},
        allowed_exposure="local", context=_ctx(_PROJECT)))
    assert "ledger_scope_migrate" not in {e["type"] for e in events}
    assert rt.activation.active(activation_scope(_AGENT, "")) == _OLD
    assert rt.activation.active(activation_scope(_AGENT, _PROJECT)) == ""
    # 项目键落空 ⇒ 读/写拿不到激活本（不再"借"Global 那本），措辞不是空白
    assert not upd.startswith("added"), upd
    assert read, "read 应有回应（无激活本的提示），不是空串"


def test_single_injection_call_site_covers_three_protocols():
    """三协议（chat / messages / responses）共用 `orchestration.insert_ledger_block` 一个调用点
    ⇒ 上面那条用例覆盖三协议。调用点数变了这条先红，别让它悄悄变成"只修了一个端点"。"""
    src = "".join(p.read_text() for p in (_ROOT / "packages/bladex-proxy/bladex_proxy/server").glob("*.py"))
    assert src.count("insert_ledger_block(") == 1


def _defs_and_calls(paths: list[Path]) -> tuple[set[str], set[str], list[str]]:
    defs, calls, emits = set(), set(), []
    for p in paths:
        tree = ast.parse(p.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defs.add(node.name)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)
            elif isinstance(node, ast.Attribute) and node.attr == "LEDGER_SCOPE_MIGRATE":
                emits.append(f"{p.relative_to(_ROOT)}:{node.lineno}")
    return defs, calls, emits


def test_static_guard_no_migration_in_production_source():
    """⑤ 删干净 + ① 阴性对照：把迁移调用加回（`activation.migrate(` / 发射
    `AdminEventType.LEDGER_SCOPE_MIGRATE`）⇒ 红。允许的残留只有：`models.py` 枚举定义、
    `ledger.py` 重放分支的字符串常量、各处注释/docstring（AST 不看注释）。"""
    proxy = list((_ROOT / "packages/bladex-proxy/bladex_proxy").rglob("*.py"))
    core = list((_ROOT / "packages/bladex-core/bladex_core").rglob("*.py"))
    defs, calls, emits = _defs_and_calls(proxy + core)
    assert "_maybe_migrate_binding" not in defs
    assert "_maybe_migrate_binding" not in calls
    assert "migrate" not in calls, "生产源码里又出现了 `.migrate(` 调用"
    assert emits == [], f"生产源码引用了 AdminEventType.LEDGER_SCOPE_MIGRATE（发射点）：{emits}"


# ── ③ 项目键落空：Global 那本在候选里，一次 switch 接上；措辞不诱导新建 ────────

def test_global_bound_ledger_pinned_into_candidates_when_project_key_empty():
    rt, _ = _rt(newer=6)      # 自然 recency 排第 7 ⇒ 不置顶就看不见
    blk = _block(rt, _PROJECT)
    assert f"- {_OLD}: " in blk, "Global 那本不在候选里（08-29 立迁移规则的理由①被带回来了）"
    assert _GLOBAL_BOUND_SUFFIX in blk
    recent = blk[blk.index("Other recent ledgers"):]
    assert recent.splitlines()[1].startswith(f"- {_OLD}:"), "置顶失效"
    assert recent.count("\n- ") == 5, "recency 段仍是 5 行（置顶挤掉最旧一行，不加行）"


def test_one_switch_reconnects_the_global_ledger():
    rt, events = _rt()
    _block(rt, _PROJECT)
    out = asyncio.run(rt._h_ledger_switch({"ledger_id": _OLD}, allowed_exposure="local",
                                          context=_ctx(_PROJECT)))
    assert "Error" not in out, out
    assert rt.activation.active(activation_scope(_AGENT, _PROJECT)) == _OLD
    assert [e["type"] for e in events] == ["ledger_switch"], "接上靠的是模型的一次显式 switch，不是迁移"


def test_wording_points_to_candidates_before_creating():
    """不许出现「无账本，请新建」这类裸诱导：措辞先指候选、再说新建。"""
    rt, _ = _rt()
    blk = _block(rt, _PROJECT)
    assert _FIRST_STEP_NO_LEDGER in blk
    i_switch = _FIRST_STEP_NO_LEDGER.index("listed below already covers this request")
    i_create = _FIRST_STEP_NO_LEDGER.index("ledger_id=''")
    assert i_switch < i_create
    assert "Prefer switching to a listed matching ledger over creating a new one" in blk


def test_pin_is_skipped_when_global_ledger_is_closed():
    """已关的本不进候选（批 Z · Z3 判据同源）——置顶不绕过它。"""
    from bladex_core.ledger import STATUS_CLOSED
    rt, _ = _rt()
    rt.pool[_OLD] = rt.pool[_OLD].model_copy(update={"status": STATUS_CLOSED})
    blk = _block(rt, _PROJECT)
    assert _OLD not in blk


def test_pin_is_skipped_when_project_key_already_bound():
    """项目键已有激活本 ⇒ 有激活本那一格，候选/recency 不注（G16 结论不动），也不置顶。"""
    rt, _ = _rt()
    rt.pool["ldg-projown0001"] = new_ledger(ledger_id="ldg-projown0001", title="项目自己的本",
                                            goal="g", goal_source="user",
                                            created_at="2026-09-27T10:00:00+00:00")
    rt.activation.restore({activation_scope(_AGENT, _PROJECT): "ldg-projown0001"})
    blk = _block(rt, _PROJECT)
    assert "Active task ledger" in blk and "ldg-projown0001" in blk
    assert _GLOBAL_BOUND_SUFFIX not in blk


# ── ④ 无项目信号的 agent 行为不变 ─────────────────────────────────────────────

def test_no_project_signal_block_unchanged():
    """Pi / hermes 形态：查的就是 Global ⇒ 有激活本、不置顶、不带标注、绑定不动。
    与 `_rt` 同池但**不走**置顶分支的对照：把 project_id 传空，块必须逐字等于
    「项目键就绑着这本」时的块（两者都是"有激活本"那一格）。"""
    rt, events = _rt()
    blk_global = _block(rt, "")
    rt2, _ = _rt(global_bound=False)
    rt2.activation.restore({activation_scope(_AGENT, _PROJECT): _OLD})
    blk_project_bound = _block(rt2, _PROJECT)
    assert blk_global == blk_project_bound
    assert _GLOBAL_BOUND_SUFFIX not in blk_global
    assert "Active task ledger" in blk_global and _OLD in blk_global
    assert events == []
    assert rt.activation.active(activation_scope(_AGENT, "")) == _OLD
