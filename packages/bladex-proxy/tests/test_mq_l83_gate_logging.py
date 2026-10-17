"""MQ-L83（批 W W3，2026-09-22）：账本块的档位 gate 在「**没有**激活账本」时
静默 return —— 最该说话的那一格没有读数。

`insert_ledger_block` 在 `_last_user_text` 之前有三道 return；门 ②（档位 ∉
`BLADEX_LEDGER_TIERS` ∧ 本会话没绑激活账本）的日志 `agency_ledger_block_gated`
**只在有激活账本时打**。而"没有激活账本"正是 MQ-L79 认定的「模型最需要候选面」
的那一格 ⇒ 同一件事的两面：一面是不给候选（L79，09-20 已修），另一面是
**连"我没给"都不记**（本条）。

live 后果：窗口内（`ts ≥ 2026-09-12`）codex 有 116 轮末条 user 剥完为空、
**0 轮判 aux**（门 ① 排除），而 `agency_goal_envelope_stripped` 在 codex 上
一条都没有 ⇒ 只剩门 ② 能解释，而它不留痕 ⇒「codex 为什么不触发」在存量日志里
**不可判定**，G19 Q1 只能给到"排除法 + 待抽检"。

本卡 = **零行为**：门控逻辑与返回值一行不动，只把那一格补上日志，
`reason` 分三档。
"""
from __future__ import annotations

import pytest
from bladex_core.ledger import Ledger
from bladex_core.ledger_runtime import activation_scope
from bladex_proxy.agency import AgencyRuntime
from structlog.testing import capture_logs

_MSGS = [{"role": "system", "content": "agent sys"},
         {"role": "user", "content": "在线吗？"}]


@pytest.fixture(autouse=True)
def _modules_on(monkeypatch):
    monkeypatch.setenv("BLADEX_MODULE_LEDGER", "1")
    monkeypatch.setenv("BLADEX_MODULE_TOOLFACE", "1")


def _rt(*, bound_session: str | None = "sess-A", with_ledger: bool = True):
    """带 / 不带激活账本的 runtime（其余与 `test_ledger_tier_session_gate` 同款）。"""
    rt = AgencyRuntime(index=None, hub=None)
    scope = activation_scope("codex", "")
    if not with_ledger:
        return rt
    led = Ledger(ledger_id="ldg-weather00001", title="北京今日天气查询",
                 goal="获取北京今天的实时天气信息")
    rt.pool[led.ledger_id] = led
    sessions = {scope: bound_session} if bound_session else None
    rt.activation.restore({scope: led.ledger_id}, sessions=sessions)
    return rt


def _run(rt, *, tier="weak", session_id="sess-B"):
    return rt.insert_ledger_block([dict(m) for m in _MSGS], "codex",
                                  tier=tier, session_id=session_id)


def _gated_rows(cap) -> list[dict]:
    """🔴 用 `structlog.testing.capture_logs` 而非 pytest 的 `caplog`：
    proxy 侧走 structlog，事件不经 stdlib logging（`caplog.records` 恒空，
    而日志照样打在 stdout 上 —— 这正是"假绿"最容易长出来的地方）。"""
    return [r for r in cap if r.get("event") == "agency_ledger_block_gated"]


def _reason(cap) -> str:
    rows = _gated_rows(cap)
    assert len(rows) == 1, f"期望恰好一条 gated 日志，实际 {len(rows)}"
    return rows[0]["reason"]


# ── 判据①：三档 reason 各一条 ─────────────────────────────────────────────

def test_reason_tier_not_allowed_no_ledger():
    """🔴 本卡新增的那一格：集合外档位 + **完全没有激活账本** ⇒ 也要留痕。

    修前这一格**静默 return**，是 MQ-L79「模型最需要候选面」的同一格。
    """
    with capture_logs() as cap:
        out = _run(_rt(with_ledger=False))
    assert out == _MSGS                      # 零行为：仍然不注
    assert _reason(cap) == "tier_not_allowed_no_ledger"


def test_reason_inherited_binding():
    """继承自其它会话的绑定（MQ-L28 的案发形态）⇒ 原有 reason 不变。"""
    with capture_logs() as cap:
        _run(_rt(bound_session="sess-A"), session_id="sess-B")
    assert _reason(cap) == "inherited_binding"


def test_reason_unknown_binding_origin():
    """来源未知（老事件无 session / 未恢复）⇒ 原有 reason 不变。"""
    with capture_logs() as cap:
        _run(_rt(bound_session=None), session_id="sess-B")
    assert _reason(cap) == "unknown_binding_origin"


def test_log_carries_tier_and_agent():
    """三档都要带 `tier` 与 `agent`——没有它们这条读数不可归属（MQ-V32 同族）。"""
    with capture_logs() as cap:
        _run(_rt(with_ledger=False), tier="weak")
    row = _gated_rows(cap)[0]
    assert row["tier"] == "weak" and row["agent"] == "codex"


# ── 判据②：零行为 ────────────────────────────────────────────────────────

@pytest.mark.parametrize("bound,with_ledger,tier,session,expect_injected", [
    ("sess-A", True, "weak", "sess-B", False),    # 继承绑定，集合外 ⇒ 不注
    (None, True, "weak", "sess-B", False),        # 来源未知 ⇒ 不注
    (None, False, "weak", "sess-B", False),       # 无激活账本 ⇒ 不注（本卡那一格）
    ("sess-A", True, "weak", "sess-A", True),     # 本会话绑的 ⇒ C 方案照注
    ("sess-A", True, "strong", "sess-B", True),   # 集合内档位 ⇒ 不受 gate 影响
    ("sess-A", True, "", "sess-B", True),         # tier 未传 ⇒ 老调用面不启 gate
])
def test_zero_behaviour_return_value_unchanged(bound, with_ledger, tier, session,
                                               expect_injected):
    """判据②：`insert_ledger_block` 的返回值在既有六种形态上逐字不变。

    这六格是 `test_ledger_tier_session_gate.py` 的全部形态 + 本卡新增的
    "无激活账本" 那一格。加日志不许改任何一格的注/不注。
    """
    rt = _rt(bound_session=bound, with_ledger=with_ledger)
    out = rt.insert_ledger_block([dict(m) for m in _MSGS], "codex",
                                 tier=tier, session_id=session)
    injected = len(out) > len(_MSGS)
    assert injected is expect_injected
    if not injected:
        assert out == _MSGS, "未注入时必须逐字返回原消息数组"


def test_no_gated_log_when_block_is_injected():
    """阴性对照：真注了块的轮次不该打 gated 日志（否则读数被稀释）。"""
    with capture_logs() as cap:
        _run(_rt(bound_session="sess-A"), tier="strong", session_id="sess-B")
    assert _gated_rows(cap) == []


# ── 判据③：阴性对照（删掉日志行 ⇒ 红）─────────────────────────────────────

def test_negative_control_gate_logs_exist_in_source():
    """判据③的静态一半：三档 reason 字面量必须在生产代码里各出现一次。

    行为侧的阴性对照由上面三条 `test_reason_*` 承担（删掉 `logger.info` 那几行
    ⇒ `_reason` 断言"恰好一条"当场红）。这条补的是"删掉其中**一档**"的形态：
    `reason` 收敛成两档时行为测试可能只红一条，字面量清单让它显性。
    """
    from bladex_proxy.agency import runtime

    with open(runtime.__file__, encoding="utf-8") as f:
        src = f.read()
    i = src.index("def insert_ledger_block")
    body = src[i:i + 6000]
    for reason in ("tier_not_allowed_no_ledger", "inherited_binding",
                   "unknown_binding_origin"):
        assert body.count(f'"{reason}"') == 1, reason
    assert body.count('logger.info("agency_ledger_block_gated"') == 1, (
        "gated 日志必须只有一个生产点（两处各打一次 = 计数翻倍，MQ-A18 同族）")
