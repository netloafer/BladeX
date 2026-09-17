"""ADR-0028 E4：Prompt Cache TTL 感知的消息重构 + 冷启相关性裁剪。

事故即动机：长 session 跑了很久，用户突然问一个简单问题——现状把整段历史
（实测 147K / 221K 字符）原样塞给上游。裁剪显然该做，但**不能无条件做**：
温热期裁剪 = 打碎前缀 = 上游 prompt cache 全失效，省的 token 远不够 cache 损失。

所以本卡是一道门，不是一把刀：

    warm = 同模型 ∧ (now − last_forward_ts < 该模型 cache_ttl_s) ∧ ¬prefix_changed
    warm → 全量转发 + 冻结计划原样复用，**禁止新增**降解/裁剪
    cold → 装配正常跑 + 相关性裁剪，跑完冻结计划

三组验收（对应任务卡 E4 验收行）：
  ① 温热两连发：送出 messages 前缀 sha256 逐字节相同；
  ② 冷启事故场景回放：context_chars 降幅 ≥50%，tool 配对完整；
  ③ 同输入重放两次：裁剪决策逐字节一致（确定性）。
"""

from __future__ import annotations

import hashlib

import pytest
from bladex_proxy.cache_state import SessionCacheRegistry
from bladex_proxy.pruning import apply_frozen_plan, prune_cold_context

# ── SessionCacheRegistry 判定 ──────────────────────────────────────────


def test_warm_requires_same_model_and_fresh_ts():
    reg = SessionCacheRegistry()
    reg.note_forward("s1", "glm-5.2", now=1000.0)

    assert reg.is_warm("s1", "glm-5.2", 300, now=1100.0) is True     # 100s < 300s
    assert reg.is_warm("s1", "glm-5.2", 300, now=1400.0) is False    # 400s > 300s
    assert reg.is_warm("s1", "other-model", 300, now=1100.0) is False


def test_zero_ttl_model_is_always_cold():
    """`cache_ttl_s = 0` = 该模型无 cache → 每轮都按冷处理（每轮都可重新规划）。"""
    reg = SessionCacheRegistry()
    reg.note_forward("s1", "no-cache-model", now=1000.0)
    assert reg.is_warm("s1", "no-cache-model", 0, now=1000.1) is False


def test_prefix_changed_forces_cold():
    """agent 自己压缩过历史 → 上游 cache 本就失效 → 冷（可以重新规划）。"""
    reg = SessionCacheRegistry()
    reg.note_forward("s1", "m", now=1000.0)
    assert reg.is_warm("s1", "m", 300, prefix_changed=True, now=1001.0) is False


def test_first_turn_is_cold():
    reg = SessionCacheRegistry()
    assert reg.is_warm("brand-new", "m", 300) is False


def test_model_switch_drops_frozen_plan():
    """模型换了 → 上一轮的冻结计划对新模型的 cache 毫无意义，丢弃。"""
    reg = SessionCacheRegistry()
    reg.note_forward("s1", "m1", now=1000.0)
    reg.freeze_plan("s1", {"pruned_unit_keys": ["u1"]})
    assert reg.frozen_plan("s1")["pruned_unit_keys"] == ["u1"]

    reg.note_forward("s1", "m2", now=1010.0)
    assert reg.frozen_plan("s1") == {}


def test_registry_lru_capped():
    reg = SessionCacheRegistry(max_sessions=3)
    for i in range(5):
        reg.note_forward(f"s{i}", "m", now=1000.0 + i)
    assert reg.get("s0") is None and reg.get("s1") is None
    assert reg.get("s4") is not None


# ── 冷启相关性裁剪 ─────────────────────────────────────────────────────


def _vec(seed: float) -> list[float]:
    return [seed, 1.0 - seed, 0.0]


def _long_unit(topic: str, n: int = 3000) -> list[dict]:
    """一个 closed 单元：user + assistant(tool_call) + tool + assistant。"""
    return [
        {"role": "user", "content": f"请处理 {topic} 这件事"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": f"c-{topic}", "type": "function",
                         "function": {"name": "run", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": f"c-{topic}", "content": topic * n},
        {"role": "assistant", "content": f"{topic} 处理完了"},
    ]


def _accident_messages() -> list[dict]:
    """事故场景：长会话（6 个大单元）+ 末尾一个简单问题。"""
    msgs: list[dict] = [{"role": "system", "content": "You are helpful."}]
    for t in ("阿根廷", "开票", "报表", "迁移", "压测", "巡检"):
        msgs.extend(_long_unit(t))
    msgs.append({"role": "user", "content": "现在几点了"})
    return msgs


class _Embedder:
    """确定性 embedder：按 intent 文本给固定向量。

    "阿根廷/开票/报表/迁移" 与 query 正交（低分，可裁）；
    "压测" 与 query 同向（高分，`keep_floor` 保它）。
    """

    def __call__(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            out.append(_vec(1.0) if "压测" in t else _vec(0.0))
        return out


_Q_VEC = _vec(1.0)


def test_cold_prune_halves_accident_context():
    msgs = _accident_messages()
    before = sum(len(str(m.get("content", ""))) for m in msgs)

    out, decision = prune_cold_context(
        msgs, q_vec=_Q_VEC, embed_passage=_Embedder(), target_chars=20_000,
    )

    assert decision["pruned"], "事故场景必须裁到东西"
    after = sum(len(str(m.get("content", ""))) for m in out)
    assert after <= before * 0.5, f"降幅不足 50%：{before} → {after}"


def test_cold_prune_protects_system_open_and_recent_units():
    msgs = _accident_messages()
    out, decision = prune_cold_context(
        msgs, q_vec=_Q_VEC, embed_passage=_Embedder(), target_chars=1_000,
        protect_recent=3,
    )
    # system 永远在
    assert out[0] == {"role": "system", "content": "You are helpful."}
    # open 单元（末尾那句简单问题）原样保留
    assert out[-1] == {"role": "user", "content": "现在几点了"}
    # 最近 3 个单元没被裁（"迁移"/"压测"/"巡检" + open 那个）
    kept = "".join(str(m.get("content", "")) for m in out)
    assert "巡检 处理完了" in kept


def test_cold_prune_keeps_high_relevance_unit_even_over_budget():
    """score ≥ KEEP_FLOOR 的单元即使超预算也保留。"""
    msgs = _accident_messages()
    out, _ = prune_cold_context(
        msgs, q_vec=_Q_VEC, embed_passage=_Embedder(), target_chars=100,
        protect_recent=0, keep_floor=0.55,
    )
    kept = "".join(str(m.get("content", "")) for m in out)
    assert "压测" in kept                    # 高相关，保住
    assert "早前任务" in kept                 # 低相关的被换成占位


def test_cold_prune_never_splits_tool_pairing():
    """裁剪粒度 = TaskUnit：绝不切散 tool_call / tool_result 配对。"""
    msgs = _accident_messages()
    out, _ = prune_cold_context(
        msgs, q_vec=_Q_VEC, embed_passage=_Embedder(), target_chars=1_000,
        protect_recent=1,
    )
    call_ids = {
        tc["id"] for m in out for tc in (m.get("tool_calls") or [])
    }
    result_ids = {m.get("tool_call_id") for m in out if m.get("role") == "tool"}
    assert result_ids <= call_ids, "出现了没有对应 tool_call 的 tool 结果"
    assert call_ids <= result_ids, "出现了没有结果的 tool_call"


def test_cold_prune_is_deterministic():
    """同输入重放两次 → 裁剪决策逐字节一致（重放等价性）。"""
    a_out, a_dec = prune_cold_context(
        _accident_messages(), q_vec=_Q_VEC, embed_passage=_Embedder(),
        target_chars=20_000)
    b_out, b_dec = prune_cold_context(
        _accident_messages(), q_vec=_Q_VEC, embed_passage=_Embedder(),
        target_chars=20_000)
    assert a_dec == b_dec
    assert a_out == b_out


def test_cold_prune_noop_when_under_target():
    msgs = [{"role": "user", "content": "短会话"}]
    out, dec = prune_cold_context(msgs, q_vec=_Q_VEC, embed_passage=_Embedder())
    assert out is msgs and dec["pruned"] == [] and dec["reason"] == "under_target"


def test_cold_prune_noop_without_query_vector():
    msgs = _accident_messages()
    out, dec = prune_cold_context(msgs, q_vec=None, embed_passage=_Embedder(),
                                  target_chars=10)
    assert out is msgs and dec["reason"] == "no_query_vector"


def test_cold_prune_decision_recorded_for_audit():
    _out, dec = prune_cold_context(
        _accident_messages(), q_vec=_Q_VEC, embed_passage=_Embedder(),
        target_chars=20_000)
    assert set(dec) >= {"pruned", "scores", "target", "kept_chars", "chars_before"}
    assert all(k in dec["scores"] for k in dec["pruned"])


# ── 温热：冻结计划复用 → 前缀逐字节一致 ────────────────────────────────


def _prefix_hash(messages: list[dict]) -> str:
    blob = "\x00".join(f"{m.get('role')}:{m.get('content')}" for m in messages[:-1])
    return hashlib.sha256(blob.encode()).hexdigest()


def test_warm_frozen_plan_reproduces_identical_prefix():
    """温热两连发：复用冻结计划 → 送出前缀 sha256 相同。

    对照组（重新裁剪）会因为 q_vec 变了而给出不同决策 —— 那正是要避免的
    "每轮都重新想一遍，于是每轮都把 cache 打碎"。
    """
    msgs = _accident_messages()
    cold_out, dec = prune_cold_context(
        msgs, q_vec=_Q_VEC, embed_passage=_Embedder(), target_chars=20_000)
    plan = {"l2_summary_msg": None, "l2_dropped_unit_keys": [],
            "pruned_unit_keys": dec["pruned"]}

    # 下一轮：历史前缀不变，只追加了新的一问
    next_msgs = [*msgs, {"role": "assistant", "content": "现在是下午三点"},
                 {"role": "user", "content": "那明天呢"}]
    warm_out, touched = apply_frozen_plan(next_msgs, plan)

    assert touched == len(dec["pruned"])
    assert _prefix_hash(cold_out) == _prefix_hash(warm_out[:len(cold_out)])


def test_warm_frozen_plan_noop_when_empty():
    msgs = _accident_messages()
    out, touched = apply_frozen_plan(msgs, {})
    assert out is msgs and touched == 0


def test_warm_frozen_plan_survives_unknown_unit_keys():
    """计划里的 unit_key 在本轮找不到（历史被 agent 压缩过）→ 跳过，不崩。"""
    msgs = _accident_messages()
    out, touched = apply_frozen_plan(
        msgs, {"pruned_unit_keys": ["deadbeefdeadbeef"]})
    assert touched == 0 and out is msgs


# ── 配置：每模型 TTL ───────────────────────────────────────────────────


def test_model_candidate_has_cache_ttl_default_300():
    from bladex_core.routing import ModelCandidate

    assert ModelCandidate(model="m").cache_ttl_s == 300


def test_routing_config_model_carries_cache_ttl():
    from bladex_proxy.routing_config import ModelSpec

    assert ModelSpec(name="m").cache_ttl_s == 300
    assert ModelSpec(name="m", cache_ttl_s=0).cache_ttl_s == 0


@pytest.mark.parametrize("raw,expected", [("0", False), ("1", True)])
def test_ttl_awareness_flag_rollback(monkeypatch, raw, expected):
    from bladex_proxy.inject import _ttl_enabled

    monkeypatch.setenv("BLADEX_CACHE_TTL_AWARE", raw)
    assert _ttl_enabled() is expected
