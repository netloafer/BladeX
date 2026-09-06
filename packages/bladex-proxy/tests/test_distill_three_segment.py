"""三段式蒸馏 v5（T1b/c/d，2026-08-10）：契约 / 兜底冻结 / MS-15。

覆盖任务卡 T1 验收六项：
  ① v5 契约解析（topic/keywords/facts 三段齐 + topic 广播到 facts）
  ② 缺 topic/keywords 键降级不失败
  ③ 时间锚定用 turn_time 不用系统时间（载荷渲染 + prompt 条款锚词）
  ④ 兜底路径冻结（v3/conclusion/filesum 的 PROMPT_VER 常量 + prompt 文本
     hash 钉死——动了当场红，那是第二次 bump）
  ⑤ MS-15 三场景（预算随输入走 / 截断打捞非空 / length 与 parse 分桶）
  ⑥ 语言钉死条款仍在 prompt（锚词断言）
"""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest
from bladex_proxy import distillation as distillation_mod
from bladex_proxy import router_sdk
from bladex_proxy.distillation import (
    CONCLUSION_PROMPT_VER,
    FILE_SUMMARY_PROMPT_VER,
    PROMPT_VER,
    TURN_PROMPT_VER,
    LLMDistiller,
    _effective_max_tokens,
    _parse_distill_json,
    _salvage_turn_json,
    build_turn_prompt,
)
from bladex_core.distillation import DistillTurnInput


def _resp(content: str, finish_reason: str = "stop"):
    msg = SimpleNamespace(content=content, reasoning_content=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason=finish_reason)])


_V5_FULL = json.dumps({
    "topic": "泰山啤酒生产线投资评估",
    "keywords": ["泰山啤酒", "生产线", "总投资", "10万吨"],
    "facts": [
        {"content": "新建10万吨泰山原浆啤酒生产线，2026年价格下总投资估算约3.8-5.1亿元",
         "item_kind": "assertion", "source": "assistant",
         "subject": "泰山啤酒生产线", "attribute": "投资额",
         "entities": ["泰山啤酒", "10万吨"], "importance": 7,
         "valid_until": "", "corrects": ""},
    ],
    "matter_proposals": [{"title": "泰山啤酒投资分析", "entities": ["泰山啤酒"]}],
    "event": "none",
}, ensure_ascii=False)


# ── ① v5 契约解析 ──────────────────────────────────────────────────────────


def test_v5_contract_three_segments():
    out = _parse_distill_json(_V5_FULL, "mock/m")
    assert out is not None
    assert out.topic == "泰山啤酒生产线投资评估"
    assert out.keywords == ["泰山啤酒", "生产线", "总投资", "10万吨"]
    assert len(out.facts) == 1
    # topic 广播到每条 fact（台账只存 facts，穿越靠它——与 event 同款）
    assert out.facts[0].topic == "泰山啤酒生产线投资评估"


def test_turn_prompt_ver_bumped_once():
    """一卡一 bump：版本串永远停在 `-001`，出现 `-002` = 同卡内 bump 了两次。

    🔴 2026-08-19 改判据（理由与 `test_m1_distill_v4.test_prompt_version_bumped_once`
    同款，详见那边的 docstring）：原文钉死 `== "v6-identity-time-001"`，
    守不住它自己那句话，只是让每次 bump 要改三个文件——v7 一改同时红三处。
    "当前值是多少"归当前版本那张卡的文件（`test_distill_v7_granularity.py`）。
    """
    assert TURN_PROMPT_VER.endswith("-001"), TURN_PROMPT_VER


# ── ② 缺键降级不失败 ──────────────────────────────────────────────────────


def test_missing_topic_keywords_not_a_failure():
    raw = json.dumps({"facts": [{"content": "用户偏好简洁回复",
                                 "item_kind": "preference",
                                 "subject": "回复", "attribute": "风格"}],
                      "matter_proposals": []}, ensure_ascii=False)
    out = _parse_distill_json(raw, "mock/m")
    assert out is not None and not out.failed
    assert out.topic == "" and out.keywords == []
    assert out.facts[0].topic == ""


def test_null_topic_keywords_safe():
    """M0-2 同款 null 安全：null ≠ 缺失，但两者都该折叠成空。"""
    raw = '{"topic": null, "keywords": null, "facts": [], "matter_proposals": []}'
    out = _parse_distill_json(raw, "mock/m")
    assert out is not None and out.topic == "" and out.keywords == []


# ── ③ 时间锚定：turn_time 是唯一时钟 ─────────────────────────────────────


def test_time_anchor_uses_turn_time_not_system_clock():
    """v7/M1 有意改名：槽名 `TIME:` → `OBSERVED:`（不是测试过时，是口径说清楚）。

    `TIME` 不说明"什么的时间"，而回放锚错（MQ-D2）正是把它当成了"现在"。
    随 v7 一次性 bump 付掉，槽名的完整语义与"禁止用蒸馏执行日"条款一起在
    `test_distill_v7_granularity.py`。
    """
    payload = DistillTurnInput(turn_time="2026-01-01T08:00:00+00:00",
                               user_text="下周三交报告")
    body = build_turn_prompt(payload, "zh")
    assert "OBSERVED: 2026-01-01T08:00:00+00:00" in body
    # prompt 条款：TIME 是唯一时钟，禁止用模型自己的"今天"
    sys_prompt = distillation_mod._DISTILL_TURN_SYSTEM
    assert "it is the ONLY clock" in sys_prompt
    assert "never use your own" in sys_prompt


# ── ④ 兜底路径冻结（hash 钉死；动了 = 第二次 bump，停下来重排卡）──────────


def test_fallback_prompt_versions_frozen():
    assert PROMPT_VER == "v3-items-001"
    assert CONCLUSION_PROMPT_VER == "v2-conclusion-items-001"
    assert FILE_SUMMARY_PROMPT_VER == "v1-filesum-001"


@pytest.mark.parametrize("attr,expect_hash", [
    ("_DISTILL_SYSTEM", "3d2e577b9fccea2c"),
    ("_DISTILL_CONCLUSION_SYSTEM", "511d1e574ec77f8f"),
    ("_FILE_SUMMARY_SYSTEM", "54ceecbcca98f36a"),
])
def test_fallback_prompt_text_frozen(attr: str, expect_hash: str):
    text = getattr(distillation_mod, attr)
    assert hashlib.sha256(text.encode()).hexdigest()[:16] == expect_hash, (
        f"{attr} 文本被改动——兜底路径 prompt 冻结（任务卡纪律 5：全卡只许"
        f" bump 一次 = T1 的 v5；要改这条 prompt 就得停下来重排卡）"
    )


def test_v3_path_budget_untouched(monkeypatch: pytest.MonkeyPatch):
    """v3 兜底行为逐字不变：预算 = 配置值（不随输入缩放）。"""
    calls: list[dict] = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return _resp('{"facts": [], "matter_proposals": []}')

    monkeypatch.setattr(router_sdk, "completion", fake_completion)
    LLMDistiller(model="mock/m", max_tokens=512).distill("一" * 1000)
    assert calls[0]["max_tokens"] == 512


# ── ⑤ MS-15：预算随输入 / 截断打捞 / 分桶 ─────────────────────────────────


def test_effective_max_tokens_scales_with_input():
    assert _effective_max_tokens(512, 100) == 537       # base + chars//4
    assert _effective_max_tokens(512, 4000) == 1512
    assert _effective_max_tokens(2048, 100) == 2048     # 配置值仍是下限（env 止血）
    assert _effective_max_tokens(512, 100_000) == 4096  # cap 封顶


def test_turn_path_budget_scales(monkeypatch: pytest.MonkeyPatch):
    calls: list[dict] = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return _resp(_V5_FULL)

    monkeypatch.setattr(router_sdk, "completion", fake_completion)
    d = LLMDistiller(model="mock/m", max_tokens=512)
    d.distill_turn(DistillTurnInput(user_text="内容" * 500))   # 1000 字符
    assert calls[0]["max_tokens"] > 512


def test_length_truncation_salvages_complete_objects():
    """截断只损失最后半条：topic/keywords/完整 facts 全部捞回。"""
    truncated = (
        '{"topic": "泰山啤酒投资", "keywords": ["泰山啤酒", "共益债"], "facts": ['
        '{"content": "泰山啤酒共益债规模3000万元", "item_kind": "assertion", '
        '"source": "assistant", "subject": "泰山啤酒", "attribute": "共益债"}, '
        '{"content": "这条被截断了一半, "item_k'
    )
    out = _salvage_turn_json(truncated, "mock/m")
    assert out is not None
    assert out.topic == "泰山啤酒投资"
    assert out.keywords == ["泰山啤酒", "共益债"]
    assert len(out.facts) == 1                       # 半条不入
    assert out.facts[0].content == "泰山啤酒共益债规模3000万元"
    assert out.facts[0].topic == "泰山啤酒投资"       # 广播语义与正常路径一致


def test_salvage_nothing_returns_none():
    assert _salvage_turn_json('{"topic": "", "keywords": [], "facts": [{"cont', "m") is None
    assert _salvage_turn_json("not json at all", "m") is None


def test_length_vs_parse_buckets_and_no_cache(monkeypatch: pytest.MonkeyPatch):
    """分桶：length 打捞成功计 length_salvaged；打捞产物不写台账。"""
    truncated = ('{"topic": "主题A", "keywords": ["k1"], "facts": ['
                 '{"content": "完整的一条事实内容", "item_kind": "assertion", '
                 '"source": "user", "subject": "s", "attribute": "a"}, {"content": "半')

    def fake_completion(**kwargs):
        return _resp(truncated, "length")

    puts: list[str] = []

    class Ledger:
        def get_distill(self, *a, **k):
            return None

        def put_distill(self, text, model, ver, output, **k):
            puts.append(ver)
            return "key"

    monkeypatch.setattr(router_sdk, "completion", fake_completion)
    d = LLMDistiller(model="mock/m", max_tokens=512, journal=Ledger())
    out = d.distill_turn(DistillTurnInput(user_text="足够长的用户输入内容示例"))
    assert not out.failed and out.topic == "主题A" and len(out.facts) == 1
    s = d.stats()
    assert s["length_salvaged"] == 1 and s["length_fails"] == 0 and s["parse_fails"] == 0
    assert puts == [], "打捞产物是部分产物，不许写台账（rebuild 该重蒸全量）"


def test_length_exhausted_bucket_distinct_from_parse(monkeypatch: pytest.MonkeyPatch):
    """length 截断且打捞为空 → length_fails；普通烂输出 → parse_fails。互斥。"""
    def truncated_garbage(**kwargs):
        return _resp('["not-a-dict-at-a', "length")

    monkeypatch.setattr(router_sdk, "completion", truncated_garbage)
    d = LLMDistiller(model="mock/m", max_tokens=512)
    out = d.distill_turn(DistillTurnInput(user_text="输入内容一"))
    assert out.failed and out.failure_kind == "parse"   # 队列语义与 parse 同
    assert d.stats()["length_fails"] == 1
    assert d.stats()["parse_fails"] == 0

    def plain_garbage(**kwargs):
        return _resp("我不是 JSON", "stop")

    monkeypatch.setattr(router_sdk, "completion", plain_garbage)
    d2 = LLMDistiller(model="mock/m", max_tokens=512)
    out2 = d2.distill_turn(DistillTurnInput(user_text="输入内容二"))
    assert out2.failed
    assert d2.stats()["parse_fails"] == 1
    assert d2.stats()["length_fails"] == 0


# ── ⑥ 语言钉死条款（M1-4）仍在 v5 prompt ─────────────────────────────────


def test_language_pinning_clause_survives_v5():
    sys_prompt = distillation_mod._DISTILL_TURN_SYSTEM
    assert distillation_mod._LANG_PLACEHOLDER in sys_prompt
    assert 'LANGUAGE: write every "content", "subject" and "attribute" in' in sys_prompt
    # keywords 段也要求跨语言归一到配置语言（T1b 契约）
    assert "normalize" in sys_prompt and "keywords" in sys_prompt


def test_turn_system_renders_output_lang(monkeypatch: pytest.MonkeyPatch):
    captured: list[str] = []

    def fake_completion(**kwargs):
        captured.append(kwargs["messages"][0]["content"])
        return _resp(_V5_FULL)

    monkeypatch.setattr(router_sdk, "completion", fake_completion)
    monkeypatch.setenv("BLADEX_DISTILL_LANG", "zh")
    LLMDistiller(model="mock/m").distill_turn(DistillTurnInput(user_text="内容"))
    assert "__OUTPUT_LANG__" not in captured[0]
    assert "Chinese" in captured[0]
