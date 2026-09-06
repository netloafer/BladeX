"""LLMDistiller 台账缓存单元测试（T6.2，ADR-0018 §4.1）。

mock DistillLedger + mock Router 网关，验证命中复用 / miss 写台账 / 写失败降级。
"""

import pytest
from bladex_core.distillation import DistillFact, DistillOutput
from bladex_proxy.distillation import LLMDistiller


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _MockLedger:
    """mock 蒸馏台账（实现 DistillJournalProtocol）。"""

    def __init__(self, cached: DistillOutput | None = None) -> None:
        self._cached = cached
        self.get_calls = 0
        self.put_calls = 0
        self.last_output: DistillOutput | None = None

    def get_distill(self, source_text: str, distill_model: str, prompt_ver: str) -> DistillOutput | None:
        self.get_calls += 1
        return self._cached

    def put_distill(self, source_text: str, distill_model: str, prompt_ver: str, output: DistillOutput) -> str:
        self.put_calls += 1
        self.last_output = output
        return "distill/key"


def test_ledger_hit_skips_llm():
    """台账命中 -> 直接返回缓存，不调 LLM、不写台账（零 LLM 成本）。"""
    cached = DistillOutput(
        facts=[DistillFact(content="cached", kind="event")], model_name="m1",
    )
    journal = _MockLedger(cached=cached)
    distiller = LLMDistiller(model="m1", journal=journal)

    out = distiller.distill("some text")

    assert journal.get_calls == 1
    assert journal.put_calls == 0
    assert distiller.stats()["calls"] == 0          # 没调 LLM
    assert distiller.stats()["ledger_hits"] == 1
    assert len(out.facts) == 1
    assert out.facts[0].content == "cached"


def test_ledger_miss_calls_llm_and_writes(monkeypatch):
    """台账 miss -> 调 LLM + 写台账。"""
    router_sdk = pytest.importorskip("bladex_proxy.router_sdk")
    journal = _MockLedger(cached=None)  # miss
    distiller = LLMDistiller(model="m1", journal=journal)

    json_out = (
        '{"facts": [{"content": "用户喜欢 e5", "kind": "preference", "entities": ["e5"]}], '
        '"matter_proposals": [{"title": "嵌入选型", "entities": ["e5"]}]}'
    )
    monkeypatch.setattr(router_sdk, "completion", lambda **kw: _FakeResponse(json_out))

    # 输入用中文：本条测的是**台账**，不是语言一致性。ADR-0028 E6.5 起蒸馏产出
    # 与输入语言不符会触发一次重蒸（那条语义另有剧本覆盖），
    # 英文输入 + 中文产出会在这里多出一次调用，把台账断言搅浑。
    out = distiller.distill("用户说他喜欢 e5 这个嵌入模型")

    assert journal.get_calls == 1
    assert distiller.stats()["calls"] == 1
    assert journal.put_calls == 1                     # miss 后写台账
    assert journal.last_output is not None
    assert len(journal.last_output.facts) == 1
    assert len(out.facts) == 1
    assert out.facts[0].content == "用户喜欢 e5"
    assert len(out.matter_proposals) == 1


def test_journal_put_failure_degrades(monkeypatch):
    """台账写失败（read-only Memory Hub）-> 降级 log，不阻断蒸馏。"""
    router_sdk = pytest.importorskip("bladex_proxy.router_sdk")

    class _FailingLedger(_MockLedger):
        def put_distill(self, *a, **kw):  # type: ignore[override]
            raise RuntimeError("read-only Memory Hub cannot write")

    journal = _FailingLedger(cached=None)
    distiller = LLMDistiller(model="m1", journal=journal)

    json_out = '{"facts": [{"content": "x", "kind": "event"}], "matter_proposals": []}'
    monkeypatch.setattr(router_sdk, "completion", lambda **kw: _FakeResponse(json_out))

    out = distiller.distill("some text")  # 不应抛
    assert len(out.facts) == 1
    assert out.facts[0].content == "x"


def test_no_ledger_no_cache(monkeypatch):
    """无 journal -> 每次调 LLM（向后兼容，台账缓存关闭）。"""
    router_sdk = pytest.importorskip("bladex_proxy.router_sdk")
    distiller = LLMDistiller(model="m1")  # 无 journal

    json_out = '{"facts": [{"content": "x", "kind": "event"}], "matter_proposals": []}'
    monkeypatch.setattr(router_sdk, "completion", lambda **kw: _FakeResponse(json_out))

    out = distiller.distill("some text")
    assert distiller.stats()["calls"] == 1
    assert distiller.stats()["ledger_hits"] == 0
    assert len(out.facts) == 1


def test_ledger_hit_uses_truncated_text_as_key(monkeypatch):
    """台账 key 用截断后的 source_text（与 put 一致，超长输入命中正确）。"""
    journal = _MockLedger(cached=None)
    distiller = LLMDistiller(model="m1", journal=journal, max_tokens=64)
    router_sdk = pytest.importorskip("bladex_proxy.router_sdk")
    monkeypatch.setattr(
        router_sdk, "completion",
        lambda **kw: _FakeResponse('{"facts": [], "matter_proposals": []}'),
    )

    long_text = "x" * 3000  # 超 _MAX_INPUT_CHARS(2000)
    distiller.distill(long_text)

    # get_distill 收到的是截断后的文本（长度 2000）
    # _MockLedger 不记 source_text，改用记录版验证
    class _RecordingLedger(_MockLedger):
        def __init__(self):
            super().__init__(cached=None)
            self.seen_text = ""
        def get_distill(self, source_text, model, ver):
            self.seen_text = source_text
            return super().get_distill(source_text, model, ver)

    rec = _RecordingLedger()
    d2 = LLMDistiller(model="m1", journal=rec, max_tokens=64)
    monkeypatch.setattr(
        router_sdk, "completion",
        lambda **kw: _FakeResponse('{"facts": [], "matter_proposals": []}'),
    )
    d2.distill(long_text)
    assert len(rec.seen_text) == 2000  # 截断到 _MAX_INPUT_CHARS
