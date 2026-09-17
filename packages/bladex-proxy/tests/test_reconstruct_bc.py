"""U9（ADR-0025 T3）：B 层关联降解。默认关（G10）。

覆盖：B 三档（full/excerpt/minimal）/ 判定冻结至单元边界（I4）/ 体积上限退档 /
degrade_plan 记录（U2）/ 关闭时与现状逐字一致 / reconstruction 落 cap_info。
（C 层澄清随 S4 2026-09-03 删除。）
"""

from __future__ import annotations

import pytest
from bladex_core.fact import Fact
from bladex_proxy.assembly import AssemblyConfig, ContextAssembler

# （原 `_pin_index_recall_on` 夹具随 2026-09-03 S1 删除：该开关与主动检索路径已不存在。）


BIG = 600  # > evidence_min_chars(500)


def _unit(user: str, tool_body: str) -> list[dict]:
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "search", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "search",
         "content": tool_body},
        {"role": "assistant", "content": "该单元结论。"},
    ]


def _messages() -> list[dict]:
    msgs: list[dict] = [{"role": "system", "content": "You are helpful."}]
    msgs += _unit("查一下 BladeX proxy 的 ANN 索引参数", "BladeX ANN 索引 recall 结果 " + "x" * BIG)
    msgs += _unit("帮我看 lancedb 的分区数", "lancedb 分区 " + "y" * BIG)
    msgs += _unit("今天天气怎么样", "天气晴 " + "z" * BIG)
    msgs += [{"role": "user", "content": "继续 ANN 的事"}]  # open unit
    return msgs


def _facts() -> list[Fact]:
    # 指纹项：BladeX/ANN/索引/recall → 单元1 强相关；lancedb → 单元2 弱相关；单元3 无关
    return [
        Fact(id="f1", content="c1", subject="BladeX", entities=["ANN", "索引", "recall"]),
        Fact(id="f2", content="c2", subject="", entities=["lancedb"]),
    ]


def _cfg(**kw) -> AssemblyConfig:
    base = dict(enabled=True, evidence_min_chars=500, evidence_excerpt_chars=200,
                msg_threshold=1000, budget_chars=0, reconstruct_b=True,
                b_strong_min_hits=3, b_full_cap_chars=50000)
    base.update(kw)
    return AssemblyConfig(**base)


def _tool_contents(msgs: list[dict]) -> list[str]:
    return [m["content"] for m in msgs if m.get("role") == "tool"]


def test_b_three_tiers():
    asm = ContextAssembler(_cfg())
    out, info = asm.assemble(_messages(), _facts(), session_id="s1")
    tools = _tool_contents(out)
    assert "recall 结果" in tools[0] and "x" * BIG in tools[0]      # 强相关：保全文
    assert tools[1].startswith("[bladex-archived-evidence")          # 弱相关：摘录
    assert "lancedb 分区" in tools[1]                                # 摘录带首段
    assert "unrelated]" in tools[2] and "天气晴" not in tools[2]     # 无关：一行元信息
    # degrade_plan 记录三档（U2：模型实际看到什么可复原）
    plan = info["degrade_plan"]
    assert set(plan.values()) == {"full", "excerpt", "minimal"}
    assert info["b_layer"] and info["b_fingerprint"]


def test_b_full_cap_falls_back_to_excerpt():
    asm = ContextAssembler(_cfg(b_full_cap_chars=300))  # 强相关单元超上限
    out, info = asm.assemble(_messages(), _facts(), session_id="s1")
    tools = _tool_contents(out)
    assert tools[0].startswith("[bladex-archived-evidence")  # 退回摘录
    assert info["degrade_plan"]


def test_b_freeze_until_unit_boundary():
    """I4：open unit 未变 → 换 facts 也复用冻结判定（工具循环内前缀稳定 R2）。"""
    asm = ContextAssembler(_cfg())
    msgs = _messages()
    out1, _ = asm.assemble(msgs, _facts(), session_id="s1")
    # 同一 open unit、facts 换成空（若重算，单元1 会降级为 minimal）
    out2, _ = asm.assemble(msgs, [], session_id="s1")
    assert _tool_contents(out1) == _tool_contents(out2)
    # 新单元边界（追加新 user）→ 重算：单元1 现在按空指纹判 minimal
    msgs2 = msgs + [{"role": "assistant", "content": "好"},
                    {"role": "user", "content": "换个话题"}]
    out3, _ = asm.assemble(msgs2, [], session_id="s1")
    assert "unrelated]" in _tool_contents(out3)[0]


def test_b_off_default_matches_status_quo():
    """B 关闭（默认）= 现状：closed tool 一律摘录，无关档不存在（下界=现状）。"""
    on = ContextAssembler(_cfg())
    off = ContextAssembler(_cfg(reconstruct_b=False))
    out_off, info_off = off.assemble(_messages(), _facts(), session_id="s1")
    tools = _tool_contents(out_off)
    assert all(t.startswith("[bladex-archived-evidence") for t in tools)
    assert "unrelated]" not in "".join(tools)
    assert not info_off["b_layer"] and info_off["b_fingerprint"] == []
    # 与 assemble 老签名（无 session_id）等价
    out_off2, _ = off.assemble(_messages(), _facts())
    assert out_off == out_off2
    _ = on  # silence


def test_from_env_defaults_off(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("BLADEX_RECONSTRUCT_B", raising=False)
    assert AssemblyConfig.from_env().reconstruct_b is False
    monkeypatch.setenv("BLADEX_RECONSTRUCT_B", "1")
    assert AssemblyConfig.from_env().reconstruct_b is True


# （C 层澄清剧本随 S4 2026-09-03 删除：`BLADEX_CLARIFY_ENABLED` env 直读、默认关、零 live 读数；
#  ReconstructionRecord.clarify_text 字段保留，恒为空。）


# ── 复盘补（2026-08-03）：B 层会话级 A/B（ADR-0025 T5 路线②）──


def test_b_ab_mode_deterministic_split(monkeypatch: pytest.MonkeyPatch):
    """ab 模式：按 session hash 确定性分组；同会话恒同组；无 session 保守关。"""
    monkeypatch.setenv("BLADEX_RECONSTRUCT_B", "ab")
    cfg = AssemblyConfig.from_env()
    assert cfg.reconstruct_b is False and cfg.reconstruct_b_ab is True
    asm = ContextAssembler(cfg)
    # 确定性：同 session 两次判定一致
    import hashlib
    sample = [f"s-{i}" for i in range(20)]
    for sid in sample:
        expect = hashlib.sha256(sid.encode()).digest()[0] % 2 == 0
        assert asm._b_enabled_for(sid) == expect
        assert asm._b_enabled_for(sid) == expect
    # 两组都存在（20 个固定样本里必然两组各有，若单侧说明分组逻辑坏了）
    groups = {asm._b_enabled_for(s) for s in sample}
    assert groups == {True, False}
    # 无 session → 关（保守）
    assert asm._b_enabled_for("") is False


def test_b_ab_group_on_behaves_like_b(monkeypatch: pytest.MonkeyPatch):
    """ab 模式分到 B_on 组的会话，行为与 reconstruct_b=True 一致。"""
    import hashlib
    sid_on = next(s for s in (f"s-{i}" for i in range(50))
                  if hashlib.sha256(s.encode()).digest()[0] % 2 == 0)
    sid_off = next(s for s in (f"s-{i}" for i in range(50))
                   if hashlib.sha256(s.encode()).digest()[0] % 2 == 1)
    cfg = _cfg(reconstruct_b=False, reconstruct_b_ab=True)
    asm = ContextAssembler(cfg)
    out_on, info_on = asm.assemble(_messages(), _facts(), session_id=sid_on)
    out_off, info_off = asm.assemble(_messages(), _facts(), session_id=sid_off)
    assert info_on["b_layer"] is True and "unrelated]" in "".join(_tool_contents(out_on))
    assert info_off["b_layer"] is False
    assert "unrelated]" not in "".join(_tool_contents(out_off))
