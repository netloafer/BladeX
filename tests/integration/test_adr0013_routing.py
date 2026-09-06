"""Phase 1 路由规格卡六项验收剧本（A–F）。

用真实 routing.toml + build_router 过流水线。filter 裁判用 mock（不联网）。
A5 failover 用 monkeypatch 模拟 Router 网关 acompletion 的上游错误。
"""

from __future__ import annotations

import asyncio

from bladex_proxy import router_sdk
import pytest
from bladex_core.routing import JudgeResult, RouteSource
from bladex_proxy.config import ProxyConfig
from bladex_proxy.route import build_router, call_model, resolve_route

_TOML = '''
[[models]]
name = "pro"
api_key_env = "K"
capabilities = ["text", "vision", "code"]
tier = "strong"

[[models]]
name = "pro2"
api_key_env = "K"
capabilities = ["text", "vision"]
tier = "strong"

[[models]]
name = "flash"
api_key_env = "K"
capabilities = ["text", "code"]
tier = "medium"

[[models]]
name = "mini"
api_key_env = "K"
capabilities = ["text"]
tier = "weak"

[upstream]
models = ["pro", "flash", "mini"]

[strategies.agent]
enabled = true
[strategies.agent.map]
hermes = ["pro", "flash"]
visionagent = ["pro", "pro2"]
textonly = ["flash", "mini"]

[strategies.multimodal]
enabled = true
[strategies.multimodal.map]
vision = ["pro", "pro2"]

[strategies.filter]
enabled = false
judge_model = "mini"
default_tier = "weak"
max_prompt_chars = 2000
[strategies.filter.candidates]
weak = ["mini"]
medium = ["flash"]
strong = ["pro"]
'''


def _run(coro):
    return asyncio.run(coro)


class FakeJudge:
    def __init__(self, tier: str = "medium") -> None:
        self._tier = tier
        self.called = False

    async def judge(self, prompt: str) -> JudgeResult:
        self.called = True
        return JudgeResult(tier=self._tier, reason="fake")


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("K", "secret")
    p = tmp_path / "routing.toml"
    p.write_text(_TOML, encoding="utf-8")
    monkeypatch.setenv("BLADEX_ROUTING_CONFIG", str(p))
    c = ProxyConfig()
    c.route_enabled = True
    return c


def _route(cfg, query, **kw):
    router = build_router(cfg)
    assert router is not None
    return _run(resolve_route("client", cfg, router, query, **kw))


# ── A 全关=顺序 failover（策略全关配 upstream）──────────────────────────────


def test_a_all_off_sequential_failover(tmp_path, monkeypatch):
    """策略全关 → source=UPSTREAM_DEFAULT，primary=upstream[0]，failover 按序。"""
    monkeypatch.setenv("K", "secret")
    toml = '''
[[models]]
name = "pro"
api_key_env = "K"
tier = "strong"

[[models]]
name = "mini"
api_key_env = "K"
tier = "weak"

[upstream]
models = ["pro", "mini"]
'''
    p = tmp_path / "routing.toml"
    p.write_text(toml, encoding="utf-8")
    monkeypatch.setenv("BLADEX_ROUTING_CONFIG", str(p))
    c = ProxyConfig()
    c.route_enabled = True
    router = build_router(c)
    d = _run(router.route("hi"))
    assert d.source == RouteSource.UPSTREAM_DEFAULT
    assert d.model == "pro"
    fo = router.failover_candidates(d)
    assert [c2.model for c2 in fo] == ["mini"]


# ── B agent 命中收窄 ─────────────────────────────────────────────────────────


def test_b_agent_match_narrows(cfg):
    """agent hermes 命中 → 集合 [pro, flash]，primary 在集合内。"""
    router = build_router(cfg)
    d = _run(router.route("hi", agent_id="hermes"))
    assert d.source == RouteSource.AGENT
    assert d.model in ("pro", "flash")
    assert "hermes" in d.reason


def test_b_agent_absent_fallthrough(cfg):
    """未配 agent → fall-through 到 upstream。"""
    router = build_router(cfg)
    d = _run(router.route("hi", agent_id="unknown"))
    assert d.source == RouteSource.UPSTREAM_DEFAULT


# ── C 带图能力过滤 (+无合格快返) ──────────────────────────────────────────────


def test_c_capability_filter_selects_vision(cfg):
    """带图 + 池有 vision → 选 vision 模型。"""
    router = build_router(cfg)
    d = _run(router.route("看图", requires=["vision"]))
    assert d.model in ("pro", "pro2")  # vision 模型


def test_c_no_capable_fastfail(tmp_path, monkeypatch):
    """textonly 集合无 vision + multimodal 关 → NoCapableCandidateError（不静默降级）。"""
    monkeypatch.setenv("K", "secret")
    toml = """
[[models]]
name = "pro"
api_key_env = "K"
capabilities = ["text", "vision"]
tier = "strong"

[[models]]
name = "flash"
api_key_env = "K"
capabilities = ["text"]
tier = "medium"

[[models]]
name = "mini"
api_key_env = "K"
capabilities = ["text"]
tier = "weak"

[upstream]
models = ["pro", "flash", "mini"]

[strategies.agent]
enabled = true
[strategies.agent.map]
textonly = ["flash", "mini"]
"""
    p = tmp_path / "routing.toml"
    p.write_text(toml, encoding="utf-8")
    monkeypatch.setenv("BLADEX_ROUTING_CONFIG", str(p))
    c = ProxyConfig()
    c.route_enabled = True
    # ADR-0021 策略 > 能力过滤：agent 命中 textonly 池（无 vision）+ 带图 -> 用策略池不快返错
    model, route, _ = _route(c, "看图", agent_id="textonly", requires=["vision"])
    assert route.source == "agent"
    assert model in ("flash", "mini")  # 用策略池（即使无 vision，不切换）


def test_c_multimodal_preference_orders_vision_first(cfg):
    """multimodal 开 + vision map → 偏好映射内模型排前。"""
    router = build_router(cfg)
    d = _run(router.route("看图", requires=["vision"]))
    # pro 和 pro2 都有 vision，但 map 偏好 [pro, pro2] → pro 排前
    assert d.model == "pro"


# ── D 筛选器：小 prompt 判 / 大 prompt 跳 / 超时兜底 ──────────────────────────


def _cfg_with_filter(tmp_path, monkeypatch, max_chars=2000):
    """配置 filter 策略开的 routing.toml。"""
    monkeypatch.setenv("K", "secret")
    toml = '''
[[models]]
name = "pro"
api_key_env = "K"
capabilities = ["text", "vision"]
tier = "strong"

[[models]]
name = "flash"
api_key_env = "K"
capabilities = ["text"]
tier = "medium"

[[models]]
name = "mini"
api_key_env = "K"
capabilities = ["text"]
tier = "weak"

[upstream]
models = ["pro", "flash", "mini"]

[strategies.filter]
enabled = true
judge_model = "mini"
default_tier = "weak"
max_prompt_chars = ''' + str(max_chars) + '''
[strategies.filter.candidates]
weak = ["mini"]
medium = ["flash"]
strong = ["pro"]
'''
    p = tmp_path / "routing.toml"
    p.write_text(toml, encoding="utf-8")
    monkeypatch.setenv("BLADEX_ROUTING_CONFIG", str(p))
    c = ProxyConfig()
    c.route_enabled = True
    return c


def test_d_filter_judges_small_prompt(tmp_path, monkeypatch):
    """filter 开 + 小 prompt → 裁判判出的档对应模型。"""
    cfg = _cfg_with_filter(tmp_path, monkeypatch)
    router = build_router(cfg)
    router._judge = FakeJudge(tier="medium")
    d = _run(router.route("summarize this"))
    assert d.model == "flash"  # medium 档
    assert d.source == RouteSource.FILTER
    assert router._judge.called is True


def test_d_filter_skips_large_prompt(tmp_path, monkeypatch):
    """大 prompt → 跳过裁判走 default_tier（weak → mini）。"""
    cfg = _cfg_with_filter(tmp_path, monkeypatch, max_chars=100)
    router = build_router(cfg)
    judge = FakeJudge(tier="medium")
    router._judge = judge
    d = _run(router.route("x" * 200))
    assert d.model == "mini"  # default_tier=weak
    assert judge.called is False


def test_d_filter_timeout_fallback(tmp_path, monkeypatch):
    """裁判失败 → default_tier，请求不挂。"""
    cfg = _cfg_with_filter(tmp_path, monkeypatch)
    router = build_router(cfg)

    class FailingJudge:
        async def judge(self, prompt):
            raise RuntimeError("timeout")

    router._judge = FailingJudge()
    d = _run(router.route("hi"))
    assert d.model == "mini"  # default_tier=weak
    assert d.source == RouteSource.FILTER


# ── E 任意开关组合自洽 ──────────────────────────────────────────────────────


def test_e_agent_and_multimodal_combined(cfg):
    """agent 命中 + 带图 → 集合内选 vision 模型（层间无冲突）。"""
    router = build_router(cfg)
    d = _run(router.route("看图", agent_id="hermes", requires=["vision"]))
    assert d.model == "pro"  # hermes [pro,flash] 内仅 pro 有 vision
    assert d.source == RouteSource.AGENT


def test_e_agent_and_filter_mutually_exclusive(tmp_path, monkeypatch):
    """agent 和 filter 都开 → agent 命中时不跑裁判（显式配置压过启发式）。"""
    monkeypatch.setenv("K", "secret")
    toml = '''
[[models]]
name = "pro"
api_key_env = "K"
tier = "strong"

[[models]]
name = "mini"
api_key_env = "K"
tier = "weak"

[upstream]
models = ["pro", "mini"]

[strategies.agent]
enabled = true
[strategies.agent.map]
hermes = ["mini"]

[strategies.filter]
enabled = true
judge_model = "mini"
default_tier = "weak"
[strategies.filter.candidates]
weak = ["mini"]
strong = ["pro"]
'''
    p = tmp_path / "routing.toml"
    p.write_text(toml, encoding="utf-8")
    monkeypatch.setenv("BLADEX_ROUTING_CONFIG", str(p))
    c = ProxyConfig()
    c.route_enabled = True
    router = build_router(c)
    judge = FakeJudge(tier="strong")
    router._judge = judge
    # agent hermes 命中 → 走 agent（mini），不跑裁判
    d = _run(router.route("hi", agent_id="hermes"))
    assert d.model == "mini"
    assert d.source == RouteSource.AGENT
    assert judge.called is False  # agent 命中不跑裁判

    # 未知 agent → 走 filter（裁判判 strong → pro）
    d2 = _run(router.route("complex task", agent_id="unknown"))
    assert d2.source == RouteSource.FILTER
    assert judge.called is True


# ── F failover 保能力 ────────────────────────────────────────────────────────


class _UpstreamErr(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(f"upstream {code}")
        self.status_code = code


def test_f_failover_preserves_capability(cfg, monkeypatch):
    """带图请求 failover 只在 vision 备选间切（不退纯文本）。"""
    router = build_router(cfg)
    model, route, failover = _run(resolve_route(
        "c", cfg, router, "看图", agent_id="visionagent", requires=["vision"],
    ))
    vision_models = {"pro", "pro2"}
    assert model in vision_models
    assert all(r.model in vision_models for r in failover)
    assert failover  # 有备选

    calls: list[str] = []

    async def fake(**kw):
        calls.append(kw["model"])
        if kw["model"] == model:
            raise _UpstreamErr(503)
        return {"model": kw["model"], "ok": True}

    monkeypatch.setattr(router_sdk, "acompletion", fake)
    result = _run(call_model(model, [{"role": "user", "content": "看图"}], stream=False,
                             route=route, failover=failover))
    assert result["model"] in vision_models and result["model"] != model
    assert "flash" not in calls and "mini" not in calls  # 没退到纯文本


def test_f_failover_4xx_does_not_retry(cfg, monkeypatch):
    """4xx 不触发 failover。"""
    router = build_router(cfg)
    model, route, failover = _run(resolve_route(
        "c", cfg, router, "看图", agent_id="visionagent", requires=["vision"],
    ))

    async def fake4xx(**kw):
        raise _UpstreamErr(400)

    monkeypatch.setattr(router_sdk, "acompletion", fake4xx)
    with pytest.raises(_UpstreamErr):
        _run(call_model(model, [{"role": "user", "content": "看图"}], stream=False,
                        route=route, failover=failover))


# ── TTS（输出模态）：modalities: ["text", "audio"] → 走 audio 映射 ─────────────


def test_tts_routes_to_audio_mapped_model(tmp_path, monkeypatch):
    """TTS 请求（modalities 含 audio，无 input_audio 块）→ 走 multimodal.map[audio]。"""
    monkeypatch.setenv("K", "secret")
    toml = '''
[[models]]
name = "glm"
api_key_env = "K"
capabilities = ["text", "audio"]
tier = "medium"

[[models]]
name = "mini"
api_key_env = "K"
capabilities = ["text"]
tier = "weak"

[upstream]
models = ["glm", "mini"]

[strategies.multimodal]
enabled = true
[strategies.multimodal.map]
audio = ["glm"]
'''
    p = tmp_path / "routing.toml"
    p.write_text(toml, encoding="utf-8")
    monkeypatch.setenv("BLADEX_ROUTING_CONFIG", str(p))
    c = ProxyConfig()
    c.route_enabled = True
    router = build_router(c)

    # 模拟 TTS 请求：纯文本 messages + output_modalities=["audio"]
    from bladex_proxy.inject import detect_required_capabilities
    requires = detect_required_capabilities(
        [{"role": "user", "content": "请把这段话转成语音"}],
        output_modalities=["audio"],
    )
    assert "audio" in requires

    # RA9（routing-fix-plan-20260712 T5d）：multimodal 不再直选，
    # 走 upstream 池 + 能力过滤保证选到 audio 模型。
    d = _run(router.route("请把这段话转成语音", requires=requires))
    assert d.model == "glm"
    assert d.source == RouteSource.UPSTREAM_DEFAULT


def test_tts_with_filter_enabled_still_uses_multimodal(tmp_path, monkeypatch):
    """TTS + filter 都开（RA9 后）→ 裁判照跑；判出档无 audio 时 multimodal 映射兜底。"""
    monkeypatch.setenv("K", "secret")
    toml = '''
[[models]]
name = "glm"
api_key_env = "K"
capabilities = ["text", "audio"]
tier = "medium"

[[models]]
name = "mini"
api_key_env = "K"
capabilities = ["text"]
tier = "weak"

[upstream]
models = ["glm", "mini"]

[strategies.multimodal]
enabled = true
[strategies.multimodal.map]
audio = ["glm"]

[strategies.filter]
enabled = true
judge_model = "mini"
default_tier = "weak"
[strategies.filter.candidates]
weak = ["mini"]
strong = ["glm"]
'''
    p = tmp_path / "routing.toml"
    p.write_text(toml, encoding="utf-8")
    monkeypatch.setenv("BLADEX_ROUTING_CONFIG", str(p))
    c = ProxyConfig()
    c.route_enabled = True
    router = build_router(c)
    router._judge = FakeJudge(tier="weak")

    # RA9：裁判照跑（判 weak）→ weak 池 [mini] 无 audio → multimodal 映射兜底 glm
    d = _run(router.route("朗读这段文字", requires=["audio"]))
    assert d.model == "glm"
    assert d.source == RouteSource.CAPABILITY
    assert router._judge.called is True  # 不再绕过裁判（带音频请求也走档位判断）
