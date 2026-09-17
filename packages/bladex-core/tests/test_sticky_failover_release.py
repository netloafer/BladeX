"""G14 / MQ-RT1 验收：failover 被动粘上的模型必须能自己放开。

2026-08-21 本机事故（`logs/proxy-20260821-112609.log`）：

    12:53:58  filter_judge_failed ... Connection error      ← ARK 端点抖动
    12:54:00  model_health_unhealthy  deepseek-v4-flash  cooldown_s=30
    12:54:00  route_failover  from=deepseek-v4-flash  to=openai/qwen3.6:35b-a3b
    13:47:25  route_decision  model=qwen3.6  source=sticky  ← 53 分钟后还粘着

熔断 30 秒就冷却了，但粘性表没有任何机制让会话切回去：`_try_sticky` 的判据里
只有"新档 <= 粘住档"，而 failover 落到的 medium 模型对之后判出的 weak/medium
恒成立。**一次 2 秒的上游抖动 = 整个会话永久漂移到降级路径。**

本文件钉三件事：
  1. origin 必须可区分（failover 被动 vs 选池主动），否则无法分别处置；
  2. failover 粘性在 cache 冷 / 超 max_hold 时释放，选池粘性**永不**被自动释放；
  3. `failover_max_hold_s=0` 逐字回到 G14 之前的行为（回滚通道）。

阴性对照贯穿全篇：修完之后"该粘的还粘着"必须同样成立——防止把防乒乓
（T2 的本意，每切一次模型 = 上游 prompt cache 全废）一起修没了。
"""

from __future__ import annotations

import time

import pytest
from bladex_core.routing import (
    FilterStrategyData,
    MemoryAwareRouter,
    ModelCandidate,
    RouteSource,
    ScaleStrategyData,
    StickyOrigin,
    StickyStrategyData,
)

SESSION = "fp:dcb2ffa12a77"  # 事故里那个真实会话指纹


def _models() -> dict[str, ModelCandidate]:
    """按事故当时的 routing.toml 形状：medium 档三个候选，其中两个是本地 ollama。"""
    return {
        "openai/deepseek-v4-flash": ModelCandidate(
            model="openai/deepseek-v4-flash", tier="medium",
            capabilities=["text", "code"], provider="ark"),
        "openai/qwen3.6:35b-a3b": ModelCandidate(
            model="openai/qwen3.6:35b-a3b", tier="medium",
            capabilities=["text", "code", "vision"], provider="ollama",
            exposure="local"),
        "openai/doubao-seed-2.0-lite": ModelCandidate(
            model="openai/doubao-seed-2.0-lite", tier="weak",
            capabilities=["text", "code"], provider="ark"),
    }


class _Warmth:
    """可控的 cache 温度（proxy 侧 SessionCacheRegistry 的替身）。"""

    def __init__(self, warm: bool = True) -> None:
        self.warm = warm
        self.calls: list[tuple[str, str]] = []

    def is_warm(self, session_id: str, model: str) -> bool:
        self.calls.append((session_id, model))
        return self.warm


def _router(warmth: _Warmth | None = None, max_hold: int = 300) -> MemoryAwareRouter:
    """裁判关掉 → filter 走 default_tier=weak，正是事故里"判 weak 却粘 medium"的形状。"""
    return MemoryAwareRouter(
        models=_models(),
        upstream=["openai/deepseek-v4-flash", "openai/doubao-seed-2.0-lite"],
        filter_strategy=FilterStrategyData(
            enabled=True,
            candidates={"weak": ["openai/doubao-seed-2.0-lite"],
                        "medium": ["openai/deepseek-v4-flash",
                                   "openai/qwen3.6:35b-a3b"]},
            default_tier="weak"),
        sticky=StickyStrategyData(enabled=True, failover_max_hold_s=max_hold),
        cache_warmth=warmth,
    )


# ── 前提断言：没修之前这个场景确实会粘住（否则后面全是假绿）────────────────


@pytest.mark.asyncio
async def test_premise_failover_sticky_would_hold_without_release() -> None:
    """前提：关掉自动释放（=G14 之前的行为）时，事故形状必须能复现。

    这条不是在测新功能，是在证明"被测的病"真的存在——否则下面那些
    "释放了"的断言可能只是因为压根没粘上。
    """
    r = _router(warmth=_Warmth(warm=False), max_hold=0)  # 0 = 旧行为
    r.update_sticky(SESSION, "openai/qwen3.6:35b-a3b",
                    origin=StickyOrigin.FAILOVER, agent_id="hermes:default")
    d = await r.route("随便问一句", session_id=SESSION, agent_id="hermes:default")
    assert d.source is RouteSource.STICKY
    assert d.model == "openai/qwen3.6:35b-a3b"  # ← 事故：cache 早冷了也照粘


# ── 1. origin 可区分 ────────────────────────────────────────────────────────


def test_origin_recorded_and_distinguishable() -> None:
    r = _router()
    r.update_sticky(SESSION, "openai/qwen3.6:35b-a3b",
                    origin=StickyOrigin.FAILOVER, agent_id="hermes:default")
    r.update_sticky("s2", "openai/deepseek-v4-flash",
                    origin=StickyOrigin.ROUTED, agent_id="hermes:accept")
    snap = {e["session_id"]: e for e in r.sticky_snapshot()}
    assert snap[SESSION]["origin"] == "failover"
    assert snap[SESSION]["agent_id"] == "hermes:default"
    assert snap["s2"]["origin"] == "routed"


def test_same_model_rewrite_keeps_origin_and_acquired_ts() -> None:
    """同模型续期不得刷新 acquired_ts。

    on_success 每轮都回写一次粘性；若每次都刷新时间戳，墙钟上限永远走不到头
    （held_s 每轮归零）——这个 bug 会让 max_hold 兜底静默失效。
    """
    r = _router()
    r.update_sticky(SESSION, "openai/qwen3.6:35b-a3b", origin=StickyOrigin.FAILOVER)
    first = r._sticky[SESSION].acquired_ts
    time.sleep(0.01)
    r.update_sticky(SESSION, "openai/qwen3.6:35b-a3b", origin=StickyOrigin.ROUTED)
    assert r._sticky[SESSION].acquired_ts == first          # 时间没被刷
    assert r._sticky[SESSION].origin is StickyOrigin.FAILOVER  # origin 也没被洗白


# ── 2. 释放：cache 冷（主触发器）────────────────────────────────────────────


@pytest.mark.asyncio
async def test_failover_sticky_released_when_cache_cold() -> None:
    warmth = _Warmth(warm=False)
    r = _router(warmth=warmth)
    r.update_sticky(SESSION, "openai/qwen3.6:35b-a3b", origin=StickyOrigin.FAILOVER)
    d = await r.route("随便问一句", session_id=SESSION)
    assert d.source is not RouteSource.STICKY
    assert d.model != "openai/qwen3.6:35b-a3b"      # 放回选池 → 回到 ARK
    assert SESSION not in r._sticky or r._sticky[SESSION].model != "openai/qwen3.6:35b-a3b"
    assert warmth.calls, "释放判定必须真的查过 cache 温度，而不是拍脑袋"


@pytest.mark.asyncio
async def test_failover_sticky_kept_while_cache_warm() -> None:
    """阴性对照：cache 还热时**不许**释放——释放要免费才做。"""
    r = _router(warmth=_Warmth(warm=True))
    r.update_sticky(SESSION, "openai/qwen3.6:35b-a3b", origin=StickyOrigin.FAILOVER)
    d = await r.route("随便问一句", session_id=SESSION)
    assert d.source is RouteSource.STICKY
    assert d.model == "openai/qwen3.6:35b-a3b"
    assert "origin=failover" in d.reason   # MQ-RT3：降级漂移必须在 reason 里看得见
    assert "held_s=" in d.reason


# ── 2b. 释放：墙钟兜底 ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_failover_sticky_released_after_max_hold_even_when_warm() -> None:
    """cache 一直热也不能无限期粘着——高频会话的兜底。"""
    r = _router(warmth=_Warmth(warm=True), max_hold=60)
    r.update_sticky(SESSION, "openai/qwen3.6:35b-a3b", origin=StickyOrigin.FAILOVER)
    r._sticky[SESSION].acquired_ts = time.time() - 61     # 持有超过上限
    d = await r.route("随便问一句", session_id=SESSION)
    assert d.source is not RouteSource.STICKY
    assert d.model != "openai/qwen3.6:35b-a3b"


@pytest.mark.asyncio
async def test_no_warmth_probe_falls_back_to_wall_clock() -> None:
    """未注入 CacheWarmth（如单测/老配置）时只剩墙钟兜底，不能变成"永不释放"。"""
    r = _router(warmth=None, max_hold=60)
    r.update_sticky(SESSION, "openai/qwen3.6:35b-a3b", origin=StickyOrigin.FAILOVER)
    r._sticky[SESSION].acquired_ts = time.time() - 61
    d = await r.route("随便问一句", session_id=SESSION)
    assert d.source is not RouteSource.STICKY


# ── 3. 选池粘性永不被自动释放（T2 本意不得被误伤）──────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("warm", [True, False])
async def test_routed_sticky_never_auto_released(warm: bool) -> None:
    """裁判/选池主动选出的粘性是设计意图（防乒乓保 cache），冷了也不放。

    两个温度都跑一遍：释放逻辑若漏判 origin，冷的那次会挂。
    """
    r = _router(warmth=_Warmth(warm=warm), max_hold=1)
    r.update_sticky(SESSION, "openai/qwen3.6:35b-a3b", origin=StickyOrigin.ROUTED)
    r._sticky[SESSION].acquired_ts = time.time() - 3600   # 持有一小时也不放
    d = await r.route("随便问一句", session_id=SESSION)
    assert d.source is RouteSource.STICKY
    assert d.model == "openai/qwen3.6:35b-a3b"


@pytest.mark.asyncio
async def test_upgrade_still_wins_over_failover_sticky() -> None:
    """升档语义不变：规模层判出更高档时照样切（与释放机制无关，防回归）。

    粘在 weak，规模 floor 抬到 medium → 必须升档，不能因为"cache 还热"就赖着。
    """
    r = MemoryAwareRouter(
        models=_models(),
        upstream=["openai/deepseek-v4-flash", "openai/doubao-seed-2.0-lite"],
        filter_strategy=FilterStrategyData(
            enabled=True,
            candidates={"weak": ["openai/doubao-seed-2.0-lite"],
                        "medium": ["openai/deepseek-v4-flash",
                                   "openai/qwen3.6:35b-a3b"]},
            default_tier="weak"),
        scale=ScaleStrategyData(enabled=True, floor_medium_chars=1000,
                                floor_strong_chars=0, judge_bypass_chars=1000),
        sticky=StickyStrategyData(enabled=True, failover_max_hold_s=300),
        cache_warmth=_Warmth(warm=True),
    )
    r.update_sticky(SESSION, "openai/doubao-seed-2.0-lite",
                    origin=StickyOrigin.FAILOVER)
    d = await r.route("x", session_id=SESSION, context_chars=5000)
    assert d.tier == "medium"
    assert d.model != "openai/doubao-seed-2.0-lite"


# ── 2c. 多轮续期（事故的真实形状：会话一直在说话）────────────────────────────


def _on_success(r: MemoryAwareRouter, primary: str, used: str) -> None:
    """复刻 `server._call_hooks._on_used` 的 origin 判定。

    它**每轮成功转发都会跑**，包括那些只是沿用粘性的轮次——正是本组两个坑的
    触发条件，所以仿真必须把它一起跑，只调 route() 测不出来。
    """
    origin = StickyOrigin.FAILOVER if primary and used != primary else StickyOrigin.ROUTED
    r.update_sticky(SESSION, used, origin=origin, agent_id="hermes:default")


@pytest.mark.asyncio
async def test_twenty_turns_of_renewal_do_not_launder_origin_or_reset_clock() -> None:
    """粘上之后连跑 20 轮，origin 不许被洗白、acquired_ts 不许被刷新。

    两个坑都在这条路径上（G14.1 完成记录）：
      坑一 acquired_ts 每轮归零 → 墙钟兜底永远走不到头；
      坑二 第 2 轮起 primary==used ⇒ 算出 ROUTED → FAILOVER 标记被抹掉
           → `_sticky_release_reason` 首行就 return None → 彻底不可释放。
    **cache 全程保持热**：这既是事故的真实形状（活跃会话），也让主触发器失效，
    从而把墙钟兜底单独暴露出来测——两个触发器同时死掉正是最危险的组合。
    """
    r = _router(warmth=_Warmth(warm=True), max_hold=300)

    d = await r.route("q", session_id=SESSION, agent_id="hermes:default")
    _on_success(r, primary=d.model, used="openai/qwen3.6:35b-a3b")  # 转发失败→failover
    t0 = r._sticky[SESSION].acquired_ts
    assert r._sticky[SESSION].origin is StickyOrigin.FAILOVER

    for _ in range(20):
        d = await r.route("q", session_id=SESSION, agent_id="hermes:default")
        assert d.model == "openai/qwen3.6:35b-a3b"     # 一直粘着（cache 热，不该释放）
        _on_success(r, primary=d.model, used=d.model)  # primary==used ⇒ 算出 ROUTED

    e = r._sticky[SESSION]
    assert e.origin is StickyOrigin.FAILOVER, "20 轮续期把 failover 标记洗白了"
    assert e.acquired_ts == t0, "20 轮续期把 acquired_ts 刷新了，墙钟兜底会永远走不到头"

    # 时钟推过 max_hold → 即使 cache 仍热也必须释放
    e.acquired_ts = time.time() - 301
    d = await r.route("q", session_id=SESSION, agent_id="hermes:default")
    assert d.source is not RouteSource.STICKY
    assert d.model != "openai/qwen3.6:35b-a3b"
    # 释放后按正常流水线重新粘上，origin 回到 ROUTED（新选择不是降级态）
    assert r._sticky[SESSION].origin is StickyOrigin.ROUTED


# ── 3b. 熔断挤掉首选 = 同一件事，也必须标 FAILOVER ─────────────────────────


class _Health:
    def __init__(self, down: set[str]) -> None:
        self.down = down

    def is_healthy(self, model: str) -> bool:
        return model not in self.down


@pytest.mark.asyncio
async def test_health_degraded_pick_is_marked_failover() -> None:
    """首选被熔断挤掉而选中的模型，粘性必须标 FAILOVER。

    否则释放机制在这条路径上被整个绕过：
      释放 → 重选池 → 首选仍在熔断 → 又选中同一个降级模型 → 却标成 ROUTED
      → 从此永不释放。等于修了一半，而且是更隐蔽的那一半
      （事故里 12:54 的下一轮走的正是这条路，不是 route_failover 那条）。
    """
    r = MemoryAwareRouter(
        models=_models(),
        upstream=["openai/deepseek-v4-flash", "openai/qwen3.6:35b-a3b"],
        filter_strategy=FilterStrategyData(
            enabled=True,
            candidates={"weak": ["openai/doubao-seed-2.0-lite"],
                        "medium": ["openai/deepseek-v4-flash",
                                   "openai/qwen3.6:35b-a3b"]},
            default_tier="medium"),
        sticky=StickyStrategyData(enabled=True, failover_max_hold_s=300),
        health=_Health({"openai/deepseek-v4-flash"}),   # ARK 那个还在熔断
        cache_warmth=_Warmth(warm=True),
    )
    d = await r.route("随便问一句", session_id=SESSION, agent_id="hermes:default")
    assert d.model == "openai/qwen3.6:35b-a3b"          # 首选熔断 → 落到本地
    assert r._sticky[SESSION].origin is StickyOrigin.FAILOVER


@pytest.mark.asyncio
async def test_healthy_pick_stays_routed() -> None:
    """阴性对照：没有熔断介入时照常是 ROUTED（别把所有选择都当降级）。"""
    r = MemoryAwareRouter(
        models=_models(),
        upstream=["openai/deepseek-v4-flash", "openai/qwen3.6:35b-a3b"],
        filter_strategy=FilterStrategyData(
            enabled=True,
            candidates={"medium": ["openai/deepseek-v4-flash",
                                   "openai/qwen3.6:35b-a3b"]},
            default_tier="medium"),
        sticky=StickyStrategyData(enabled=True, failover_max_hold_s=300),
        health=_Health(set()),
        cache_warmth=_Warmth(warm=True),
    )
    d = await r.route("随便问一句", session_id=SESSION)
    assert d.model == "openai/deepseek-v4-flash"
    assert r._sticky[SESSION].origin is StickyOrigin.ROUTED


# ── 4. 回滚通道：max_hold=0 = 逐字旧行为 ───────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("warm", [True, False])
async def test_max_hold_zero_disables_all_auto_release(warm: bool) -> None:
    r = _router(warmth=_Warmth(warm=warm), max_hold=0)
    r.update_sticky(SESSION, "openai/qwen3.6:35b-a3b", origin=StickyOrigin.FAILOVER)
    r._sticky[SESSION].acquired_ts = time.time() - 86400
    d = await r.route("随便问一句", session_id=SESSION)
    assert d.source is RouteSource.STICKY, "0 必须是完整的回滚通道，两个触发器都关"


# ── 5. 手动刷新（/admin/sticky/flush 的底层）───────────────────────────────


def test_flush_all() -> None:
    r = _router()
    r.update_sticky("s1", "openai/qwen3.6:35b-a3b", agent_id="hermes:default")
    r.update_sticky("s2", "openai/deepseek-v4-flash", agent_id="codex")
    assert r.flush_sticky() == 2
    assert r.sticky_snapshot() == []


def test_flush_by_agent_exact_and_base() -> None:
    """两级匹配与 [strategies.agent.map] 同规则：base 连 profile 一起清，精确只清一个。"""
    r = _router()
    r.update_sticky("s1", "openai/qwen3.6:35b-a3b", agent_id="hermes:default")
    r.update_sticky("s2", "openai/qwen3.6:35b-a3b", agent_id="hermes:accept")
    r.update_sticky("s3", "openai/deepseek-v4-flash", agent_id="codex")

    assert r.flush_sticky("hermes:default") == 1          # 精确
    left = {e["agent_id"] for e in r.sticky_snapshot()}
    assert left == {"hermes:accept", "codex"}

    assert r.flush_sticky("hermes") == 1                  # base 兜底
    assert {e["agent_id"] for e in r.sticky_snapshot()} == {"codex"}


def test_flush_unknown_agent_is_a_noop() -> None:
    r = _router()
    r.update_sticky("s1", "openai/qwen3.6:35b-a3b", agent_id="hermes:default")
    assert r.flush_sticky("nobody") == 0
    assert len(r.sticky_snapshot()) == 1
