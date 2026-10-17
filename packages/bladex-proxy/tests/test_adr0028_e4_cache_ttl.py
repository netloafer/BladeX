"""ADR-0028 E4：Prompt Cache TTL 感知的会话状态（`cache_state`）。

🔴 批 U · U1.4（ADR-0033）：本文件原有的冷启相关性裁剪 / 冻结计划用例已删（路径并入复核）；
TTL 判定 `is_warm` 仍在，只服务两个兜底消费方（`inject._cache_warm` 的读数 / 路由粘性 FAILOVER
释放），不再是装配的触发依据。下面的原始说明保留作历史：


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

import pytest
from bladex_proxy.cache_state import SessionCacheRegistry

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
    reg.freeze_plan("s1", {"verdicts": {"u1": "trimmed"}})
    assert reg.frozen_plan("s1")["verdicts"] == {"u1": "trimmed"}

    reg.note_forward("s1", "m2", now=1010.0)
    assert reg.frozen_plan("s1") == {}


def test_registry_lru_capped():
    reg = SessionCacheRegistry(max_sessions=3)
    for i in range(5):
        reg.note_forward(f"s{i}", "m", now=1000.0 + i)
    assert reg.get("s0") is None and reg.get("s1") is None
    assert reg.get("s4") is not None


# （E4.3 冷启相关性裁剪 `pruning.prune_cold_context` 与温热期 `apply_frozen_plan` 的用例随
#  批 U · U1.4 删除：该路径并入 ADR-0033 复核（`assembly_review`），冷轮判定冻结 / 热轮重放由
#  `test_u14_assembly_trim.py` 钉；触发门从 TTL 改为上一轮 `cache_read_tokens == 0`（`test_u12_trigger_gate.py`）。）


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
