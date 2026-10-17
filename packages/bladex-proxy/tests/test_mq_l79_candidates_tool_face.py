"""MQ-L79 / 执行卡 G19.1 —— 无激活账本时，候选段必须给到工具面。

ADR-0032 §3.1「修订二之补正」的三格表：

    |                      | 用户面 | 工具面·有激活本 | 工具面·无激活本 |
    | 切换号召 + 模板        |  注   |      不注      |      不注      |
    | **相关性候选段**       |  注   |      不注      |    🔴 **注**   |
    | LEDGERS recency top-5 |  注   |      不注      |      不注      |

live 病例（`docs/reviews/dsh-ledger-unreachable-20260920.md`）：首轮之后**连续 12 个**
`face=tool` 请求的账本块是 `ledger=(空) candidates=0 recent=0 instruction=528 template=1323`
——一边注 1851 字符催记账/建本，一边不给激活本、不给候选、不给入口。模型只剩
`bladex_memory_search`，搜了 21 次。这是 MQ-L7「只给正文不给工具」的**镜像**。

判据（交接单 §3 G19.1）：
  ① 本病例回放 ⇒ 首轮后的 `face=tool` 请求 `candidates>0`；
  ② 有激活本时 `candidates=0`（G16 结论不动：任务中途不换本）；
  ④ 阴性对照 = 去掉 `_no_active_ledger` ⇒ ① 红；
  ⑤ 体积必须带读数（ADR-0029：举证责任在加注入者）。
"""

from __future__ import annotations

import structlog.testing
from bladex_core.ledger import new_ledger
from bladex_core.ledger_runtime import activation_scope
from bladex_proxy.agency import AgencyRuntime

_AGENT = "test-agent"
_QUESTION = "之前新款Mac mini和Mac studio的研究怎么样了？能担当端侧算力平台吗？"


def _agency(monkeypatch) -> AgencyRuntime:
    for mod in ("LEDGER", "TOOLFACE", "INTERCEPTION"):
        monkeypatch.setenv(f"BLADEX_MODULE_{mod}", "1")
    monkeypatch.setenv("BLADEX_LEDGER_FACE_SPLIT", "1")
    a = AgencyRuntime()
    assert a.ledger_on and a.toolface_on, "机制没开 ⇒ 下面的阴性断言会假绿"
    for lid, title, goal in (
        ("ldg-497d5e2d542b", "苹果新 Mac mini / Mac Studio 亮点与性价比分析",
         "分析苹果最新发布的 Mac mini 与 Mac Studio：亮点、相对上一代的重大提升、性价比评估。"),
        ("ldg-8b18dd601d95", "MQ-W16：status 命令 Memory 失败提示按成因分开报",
         "让 bladex status 的 Memory 一行按成因分开报错。"),
        ("ldg-37696c99a521", "4 independent bug-fix tasks (django, sympy, requests)",
         "Fix four independent upstream bugs with regression tests."),
    ):
        a.pool[lid] = new_ledger(ledger_id=lid, title=title, goal=goal,
                                 goal_source="user", created_at="2026-08-27T10:00:00Z")
    return a


def _activate(a: AgencyRuntime, lid: str = "ldg-8b18dd601d95") -> None:
    a.activation.restore({activation_scope(_AGENT, ""): lid})


def _msgs(tool_calls: int) -> list[dict]:
    """一个用户轮 + `tool_calls` 次工具往返 —— 末条 role 随之变成 `tool`。

    这正是 live 病例的形状：`turn_index` 全程恒定（用户一句补话都没说），
    请求却一个接一个地来。
    """
    out: list[dict] = [{"role": "user", "content": _QUESTION}]
    for k in range(tool_calls):
        out.append({"role": "assistant", "content": None,
                    "tool_calls": [{"id": f"c{k}", "type": "function",
                                    "function": {"name": "bladex_memory_search",
                                                 "arguments": "{}"}}]})
        out.append({"role": "tool", "tool_call_id": f"c{k}", "content": "…"})
    return out


def _round(a: AgencyRuntime, tool_calls: int) -> tuple[str, dict]:
    """跑一轮真实注入路径，返回 (块正文, 段级构成)。"""
    out = a.insert_ledger_block(_msgs(tool_calls), agent_id=_AGENT,
                                with_instruction=True, session_id="s1")
    assert len(out) >= 2, "账本块没注进来 —— 断言会失去判别力"
    return str(out[-1].get("content") or ""), dict(a.last_ledger_breakdown.chars)


class TestCriterionOne:
    """① 无激活本 + 工具面 ⇒ 候选段照给。"""

    def test_tool_face_without_active_ledger_still_gets_candidates(self, monkeypatch):
        a = _agency(monkeypatch)
        first, _ = _round(a, 0)                       # 首轮：用户面
        assert "ldg-497d5e2d542b" in first
        for n in range(1, 13):                        # 病例里的那 12 个 face=tool 请求
            body, chars = _round(a, n)
            assert chars.get("candidates", 0) > 0, f"第 {n} 个工具面请求候选段为空"
            assert "ldg-497d5e2d542b" in body, f"第 {n} 个工具面请求看不到目标账本"

    def test_face_is_really_tool(self, monkeypatch):
        """前置断言：这些轮确实是工具面 —— 否则 ① 测的是用户面，整组假绿。"""
        a = _agency(monkeypatch)
        _round(a, 0)
        with structlog.testing.capture_logs() as cap:
            _round(a, 1)
        ev = next(e for e in cap if e.get("event") == "agency_ledger_block_injected")
        assert ev.get("face") == "tool" and ev.get("face_last_role") == "tool", ev

    def test_recency_stays_user_face_only(self, monkeypatch):
        """🔴 明确不做：recency top-5 不进工具面（MQ-L8「被噪声本挤满」的原路）。"""
        a = _agency(monkeypatch)
        first, chars_user = _round(a, 0)
        _body, chars_tool = _round(a, 1)
        assert chars_user.get("recent", 0) > 0, "用户面本来就该有 recency 段"
        assert chars_tool.get("recent", 0) == 0, chars_tool

    def test_candidates_log_carries_face(self, monkeypatch):
        """三格表的三个格子要在 live 日志里可分 —— 否则读数没有分母。"""
        a = _agency(monkeypatch)
        _round(a, 0)
        with structlog.testing.capture_logs() as cap:
            _round(a, 1)
        ev = next(e for e in cap if e.get("event") == "agency_ledger_candidates")
        assert ev.get("face") == "tool" and ev.get("recent") == 0, ev
        assert ev.get("relevant") == ["ldg-497d5e2d542b"], ev


class TestCriterionTwo:
    """② 有激活本 ⇒ 工具面仍然 `candidates=0`（G16 结论不动）。"""

    def test_tool_face_with_active_ledger_gets_nothing(self, monkeypatch):
        a = _agency(monkeypatch)
        _activate(a)
        _round(a, 0)
        for n in (1, 2, 3):
            _body, chars = _round(a, n)
            assert chars.get("candidates", 0) == 0, chars
            assert chars.get("recent", 0) == 0, chars

    def test_user_face_with_active_ledger_unchanged(self, monkeypatch):
        """用户面一字不改：有激活本时候选段照旧给（这一格修订二没动）。"""
        a = _agency(monkeypatch)
        _activate(a)
        body, chars = _round(a, 0)
        assert chars.get("candidates", 0) > 0 and chars.get("recent", 0) > 0
        assert "ldg-497d5e2d542b" in body


class TestNegativeControl:
    """④ 去掉 `_no_active_ledger` ⇒ ① 必须红。"""

    def test_old_gate_gives_the_live_shape(self, monkeypatch):
        """修前门控 `with_instruction and _user_face` 在工具面上就是病例那一行：
        `candidates=0 recent=0`，只剩 instruction + template。"""
        a = _agency(monkeypatch)
        _round(a, 0)
        with structlog.testing.capture_logs() as cap:
            _body, chars = _round(a, 1)
        ev = next(e for e in cap if e.get("event") == "agency_ledger_block_injected")
        user_face = ev.get("face") == "user"
        # 复算修前门控：`_user_face` 为假 ⇒ 那一支整段跳过 ⇒ 两段都是 0
        assert not user_face
        assert chars.get("candidates", 0) > 0, "修后应当有候选"
        assert chars.get("instruction", 0) > 0 and chars.get("template", 0) > 0

    def test_gate_source_names_both_conditions(self):
        """源码级阴性对照：门控必须同时带两个判别器 —— 删掉任一个本行即红。"""
        from tests._source_probe import package_source
        src = package_source("agency")
        assert "if with_instruction and (_user_face or _no_active_ledger):" in src
        assert "_no_active_ledger = led is None" in src


class TestVolume:
    """⑤ 体积带读数 ——「应该不大」不是证据（ADR-0029 举证责任在加注入者）。"""

    def test_candidate_section_is_under_400_chars(self, monkeypatch):
        a = _agency(monkeypatch)
        _round(a, 0)
        _body, chars = _round(a, 1)
        n = chars.get("candidates", 0)
        assert 0 < n < 400, f"相关性段 {n} 字符（live 实测 325 / ≤3 条）"

    def test_at_most_three_candidates(self, monkeypatch):
        a = _agency(monkeypatch)
        _round(a, 0)
        body, _chars = _round(a, 1)
        lines = [ln for ln in body.splitlines() if ln.startswith("- ldg-")]
        assert len(lines) <= 3, lines
