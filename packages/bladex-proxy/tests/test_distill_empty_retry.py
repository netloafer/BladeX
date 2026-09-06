"""蒸馏空响应重试（2026-08-03 实跑修复：84/106 次 parse_fail 全是 raw_len=0）。"""

from __future__ import annotations

from types import SimpleNamespace

from bladex_proxy import router_sdk
import pytest
from bladex_proxy.distillation import LLMDistiller


def _resp(content: str, finish_reason: str = "stop"):
    msg = SimpleNamespace(content=content, reasoning_content=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=finish_reason)])

_GOOD = '{"facts": [{"content": "用户偏好简洁回复", "item_kind": "preference", '\
        '"subject": "回复", "attribute": "风格", "entities": []}], "matter_proposals": []}'


def test_empty_content_retried_with_larger_budget(monkeypatch: pytest.MonkeyPatch):
    calls: list[dict] = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return _resp("", "length") if len(calls) == 1 else _resp(_GOOD)

    monkeypatch.setattr(router_sdk, "completion", fake_completion)
    out = LLMDistiller(model="mock/m", max_tokens=512).distill("回复请简洁一些，这是足够长的输入。")
    assert len(calls) == 2
    assert calls[1]["max_tokens"] == 2048  # max(512*4, 2048)
    assert out.facts and out.facts[0].item_kind == "preference"


def test_empty_twice_gives_up_not_cached(monkeypatch: pytest.MonkeyPatch):
    calls: list[dict] = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return _resp("")

    monkeypatch.setattr(router_sdk, "completion", fake_completion)
    d = LLMDistiller(model="mock/m")
    out = d.distill("重试两次仍为空的输入，足够长的输入内容。")
    assert len(calls) == 2 and not out.facts
    assert d.stats()["parse_fails"] == 1  # 记失败，不缓存（下轮可重蒸）


# ── 截断（2026-08-05）：同一个根因的另一种症状 ──────────────────────────────
#
# 08-03 那次修的判据是 `not raw_text.strip()`——只兜"全空"。实测 doctor --e2e 的探针轮：
# 模型正常开写 JSON、在 512 预算里被掐断，raw_len=451 **有正文**，于是走不进重试，
# 直接 distill_parse_failed，该轮零事实却仍被标记消费。reasoning 与正文共用预算是同一个
# 根因，症状一个空一个半截，判据必须都覆盖。
_TRUNCATED = '{\n  "facts": [\n    {\n      "content": "User\'s probe code is bx7a5d5e6b'


def test_truncated_json_is_retried_like_empty(monkeypatch: pytest.MonkeyPatch):
    calls: list[dict] = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return _resp(_TRUNCATED, "length") if len(calls) == 1 else _resp(_GOOD)

    monkeypatch.setattr(router_sdk, "completion", fake_completion)
    d = LLMDistiller(model="mock/m", max_tokens=512)
    out = d.distill("请记住我的探针代码，这是一条足够长的输入内容。")
    assert len(calls) == 2, "有正文但 finish_reason=length 也必须重试"
    assert calls[1]["max_tokens"] == 2048
    assert out.facts and d.stats()["parse_fails"] == 0


def test_complete_json_with_length_finish_is_not_retried(monkeypatch: pytest.MonkeyPatch):
    """finish_reason=length 但 JSON 恰好完整：解析得出来就别多花一次调用。

    这条守的是成本——重试判据放宽后，最容易顺手把"其实没坏"的轮次也重试一遍。
    """
    calls: list[dict] = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return _resp(_GOOD, "length")

    monkeypatch.setattr(router_sdk, "completion", fake_completion)
    out = LLMDistiller(model="mock/m").distill("恰好写完就到上限的足够长输入内容。")
    assert len(calls) == 1, "能解析就不该重试（finish_reason 不是唯一判据）"
    assert out.facts


def test_empty_retry_keeps_first_response_for_diagnosis(monkeypatch: pytest.MonkeyPatch,
                                                        caplog):
    """重试返回空时保留第一次的正文——否则日志里只剩 raw_len=0，线索被自己擦掉。"""
    import logging

    calls: list[dict] = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return _resp(_TRUNCATED, "length") if len(calls) == 1 else _resp("")

    monkeypatch.setattr(router_sdk, "completion", fake_completion)
    d = LLMDistiller(model="mock/m")
    with caplog.at_level(logging.WARNING):
        out = d.distill("重试返回空的足够长输入内容测试。")
    assert len(calls) == 2 and not out.facts
    assert d.stats()["parse_fails"] == 1  # 不缓存，下轮可重蒸
    msg = "\n".join(r.getMessage() for r in caplog.records)
    assert f"raw_len={len(_TRUNCATED)}" in msg, "别把有内容的第一次响应换成空的再报"


def test_parse_failure_logs_finish_reason_and_tail(monkeypatch: pytest.MonkeyPatch, caplog):
    """诊断缺口：截断的证据在尾巴上，此前只打了 raw_len + 前 80 字符。"""
    import logging

    def fake_completion(**kwargs):
        return _resp(_TRUNCATED, "stop")  # stop = 不触发重试，直接进解析失败

    monkeypatch.setattr(router_sdk, "completion", fake_completion)
    with caplog.at_level(logging.WARNING):
        LLMDistiller(model="mock/m").distill("解析失败诊断字段的足够长输入内容。")
    msg = "\n".join(r.getMessage() for r in caplog.records)
    assert "distill_parse_failed" in msg
    assert "finish_reason=stop" in msg
    assert "tail=" in msg


def test_max_tokens_env_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BLADEX_DISTILL_MAX_TOKENS", "1024")
    calls: list[dict] = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return _resp(_GOOD)

    monkeypatch.setattr(router_sdk, "completion", fake_completion)
    LLMDistiller(model="mock/m").distill("测试 env 覆盖 max_tokens 的足够长输入。")
    assert calls[0]["max_tokens"] == 1024


def test_timeout_env_override(monkeypatch: pytest.MonkeyPatch):
    """T-C（2026-08-14）：BLADEX_DISTILL_TIMEOUT_S 旋钮——bench 事故里一条慢
    payload 每 pass 都在硬编码 30s 超时，排空卡 91% 一小时（守卫与 gate 都对，
    缺的是旋钮）。

    MQ-R2（2026-08-17）**有意改掉了原来那句"默认仍是 30"**：写死的 30 与
    `max_tokens` 互不知情，正是同一形态咬第三次的原因。默认值改为由
    `default_distill_timeout_s(max_tokens)` 推出，显式 env 仍然最优先。
    """
    from bladex_proxy.distillation import default_distill_timeout_s

    calls: list[dict] = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return _resp(_GOOD)

    monkeypatch.setattr(router_sdk, "completion", fake_completion)
    monkeypatch.delenv("BLADEX_DISTILL_MAX_TOKENS", raising=False)
    monkeypatch.setenv("BLADEX_DISTILL_TIMEOUT_S", "120")
    LLMDistiller(model="mock/m").distill("测试 env 覆盖 timeout 的足够长输入。")
    assert calls[-1]["timeout"] == 120.0

    monkeypatch.delenv("BLADEX_DISTILL_TIMEOUT_S")
    LLMDistiller(model="mock/m").distill("测试默认 timeout 随预算走的足够长输入。")
    assert calls[-1]["timeout"] == default_distill_timeout_s(512)


def test_timeout_default_follows_max_tokens(monkeypatch: pytest.MonkeyPatch):
    """MQ-R2 核心：调大 `BLADEX_DISTILL_MAX_TOKENS`，超时默认值必须跟着变大。

    三次事故（08-14 bench 卡 91% / 08-16 重建 11.3% 静默零事实 / 08-17 排空停摆）
    都是"放大了输出预算、超时没跟上"。此前这条约束**没有任何地方在检查**，
    三次修法都只是在当次命令行临时带参数。
    """
    calls: list[dict] = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return _resp(_GOOD)

    monkeypatch.setattr(router_sdk, "completion", fake_completion)
    monkeypatch.delenv("BLADEX_DISTILL_TIMEOUT_S", raising=False)

    monkeypatch.setenv("BLADEX_DISTILL_MAX_TOKENS", "512")
    LLMDistiller(model="mock/m").distill("小预算档的足够长输入内容用于蒸馏。")
    small = calls[-1]["timeout"]

    monkeypatch.setenv("BLADEX_DISTILL_MAX_TOKENS", "2048")
    LLMDistiller(model="mock/m").distill("大预算档的足够长输入内容用于蒸馏。")
    big = calls[-1]["timeout"]

    assert big > small, "放大 max_tokens 后超时默认值必须跟着放大（MQ-R2）"
    # live 标定点：2048 配 180s 稳定（08-16 全量重建 + 08-17 排空）。
    assert big >= 180.0


def test_default_distill_timeout_formula():
    """公式本身（纯函数，钉住标定点与下限）。"""
    from bladex_proxy.distillation import default_distill_timeout_s

    assert default_distill_timeout_s(2048) >= 180.0   # live 实测点
    assert default_distill_timeout_s(0) == 30.0       # 下限 = 历史默认
    assert default_distill_timeout_s(4096) > default_distill_timeout_s(2048)


def test_env_example_pairs_the_two_coupled_knobs():
    """MQ-R2 第二层：`.env.example` 里两个参数必须**成对出现且写明耦合**。

    第一层（上面几条）保证代码里的默认值不再互不知情；本条保证**人**在读模板时
    也看得到这条约束——三次事故的共同点正是"改了一个、没人想到另一个"。
    """
    from pathlib import Path

    text = (Path(__file__).resolve().parents[3] / "config" / ".env.example").read_text(
        encoding="utf-8")
    assert "BLADEX_DISTILL_MAX_TOKENS" in text
    assert "BLADEX_DISTILL_TIMEOUT_S" in text
    # 两者必须挨着（同一段注释覆盖），不是散在文件两头各说各话
    gap = abs(text.index("BLADEX_DISTILL_TIMEOUT_S") - text.index("BLADEX_DISTILL_MAX_TOKENS"))
    assert gap < 400, "两个耦合参数在模板里离得太远，读的人不会把它们联系起来"
