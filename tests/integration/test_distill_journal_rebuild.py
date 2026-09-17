"""T1 回归：distill 台账缓存空蒸馏 -- rebuild 复跑零现判（ADR-0018 遗留修复）。

复现 ADR-0018 遗留的 186 现判根因：空蒸馏（valid JSON 无 facts，即"真无事实"）原不写
台账（distillation.py 只缓存非空结果），rebuild 每轮对这些消息重蒸 = 现判。修复后空蒸馏
也入台账，第二遍 rebuild 全命中（含空）-> 0 LLM 调用。

mock Router 网关的 completion 计数，不调真 LLM；用真 IndexDistillJournal + MemoryIndex + MemoryHub
走完整 rebuild 路径（区别于 test_dpl_acceptance.test_g 用 CannedDistiller 不经台账）。
"""
from __future__ import annotations

from pathlib import Path

import pytest
from bladex_proxy.distillation import LLMDistiller
from bladex_proxy.models import Identity, Turn, TurnStatus
from bladex_proxy.storage.memory_hub import MemoryHub
from bladex_proxy.storage.memory_index import IndexDistillJournal, MemoryIndex


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _Embedder:
    """确定性 mock embedder（MemoryIndex rebuild 需要；同 test_dpl_acceptance 套路）。"""

    def __init__(self) -> None:
        self._dim = 64

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            body = t.replace("passage: ", "").replace("query: ", "")
            first = body.strip().split()[0] if body.strip() else ""
            h = abs(hash(first)) % self._dim
            v = [0.0] * self._dim
            v[h] = 0.9
            out.append(v)
        return out

    @property
    def available(self) -> bool:
        return True


# 4 条消息均在 consolidator 长度区间（20-500）内：2 条蒸馏为空（genuine empty），2 条非空。
_MESSAGES = [
    "今天天气不错适合出去散步没什么特别需要记住的事情",       # -> 真空
    "用户决定使用 LanceDB 作为派生层的向量存储引擎",      # -> 非空
    "好的我明白了这就去办一定完成没有任何问题放心吧",         # -> 真空
    "用户把嵌入模型从 e5 换成了 bge 以提升中文检索效果",      # -> 非空
]


def _make_hub(tmpdir: str) -> MemoryHub:
    ledger = MemoryHub(Path(tmpdir) / "ledger")
    ledger.open()
    for i, msg in enumerate(_MESSAGES):
        identity = Identity(user_id="u1", agent_id="a1", session_id="s1", turn_index=i)
        turn = Turn(
            identity=identity, model="test",
            request_messages=[{"role": "user", "content": msg}],
            response_text=f"resp {i}",  # < _MIN_CONCLUSION_CHARS，结论蒸馏跳过
            status=TurnStatus.OK, logical_turn=i,
        )
        ledger.put(identity.storage_key(f"171990720{i}-0"), turn)
    return ledger


def _make_distiller_counters(monkeypatch) -> dict[str, int]:
    """装 mock Router completion：按内容返回空/非空 JSON，计数 LLM 调用。返回计数器。"""
    router_sdk = pytest.importorskip("bladex_proxy.router_sdk")
    counter = {"n": 0}

    def fake_completion(**kw):
        counter["n"] += 1
        user_content = kw["messages"][-1]["content"]
        if "LanceDB" in user_content:
            return _FakeResponse(
                '{"facts": [{"content": "决定用LanceDB做派生层向量存储", "kind": "decision"}], '
                '"matter_proposals": []}'
            )
        if "e5" in user_content or "bge" in user_content:
            return _FakeResponse(
                '{"facts": [{"content": "嵌入模型从e5换成bge", "kind": "decision"}], '
                '"matter_proposals": []}'
            )
        # 真空：valid JSON 无 facts（修复前不缓存 -> rebuild 每轮重蒸）
        return _FakeResponse('{"facts": [], "matter_proposals": []}')

    monkeypatch.setattr(router_sdk, "completion", fake_completion)
    return counter


def test_distill_ledger_caches_empty_second_rebuild_zero_llm(monkeypatch, tmp_path):
    """rebuild 两遍：空蒸馏第一遍写台账，第二遍命中（含空）-> 0 LLM 调用。

    修复前：空蒸馏不写台账 -> 第二遍对 2 条空消息重蒸 = 2 现判（非 0）。
    修复后：空蒸馏入台账 -> 第二遍全命中 = 0。
    """
    counter = _make_distiller_counters(monkeypatch)
    ledger = _make_hub(str(tmp_path))
    journal = IndexDistillJournal()
    distiller = LLMDistiller(model="m1", journal=journal)
    index = MemoryIndex(Path(str(tmp_path)) / "index", embedder=_Embedder(), distiller=distiller)
    index.open()
    journal.bind(index)  # ADR-0020 T1.3: 台账绑 Memory Index meta

    # 第一遍 full rebuild：4 条消息全部蒸馏（4 LLM 调用，含 2 条真空）
    index.rebuild_from_hub(ledger, full=True)
    first_calls = counter["n"]
    assert first_calls == 4, f"第一遍应蒸馏 4 条消息（4 LLM 调用），got {first_calls}"

    # 第二遍 full rebuild：台账命中（含空蒸馏）-> 0 LLM 调用
    counter["n"] = 0
    index.rebuild_from_hub(ledger, full=True)
    second_calls = counter["n"]
    assert second_calls == 0, (
        f"第二遍 rebuild 应 0 LLM 调用（空蒸馏已入台账命中），got {second_calls}"
    )

    index.close()
    ledger.close()


def test_distill_ledger_parse_failure_not_cached_retries(monkeypatch, tmp_path):
    """解析失败（无合法 JSON）不缓存 -> rebuild 二次机会（保留记忆质量，不丢事实）。

    与空蒸馏相反：解析失败是 LLM 返回垃圾（瞬态），不应永久缓存为空。
    第二遍 rebuild 仍会重蒸（>0 调用），给 LLM 二次机会。
    """
    router_sdk = pytest.importorskip("bladex_proxy.router_sdk")
    counter = {"n": 0}

    def failing_completion(**kw):
        counter["n"] += 1
        return _FakeResponse("not json at all")  # 解析失败 -> None

    monkeypatch.setattr(router_sdk, "completion", failing_completion)

    ledger = _make_hub(str(tmp_path))
    journal = IndexDistillJournal()
    distiller = LLMDistiller(model="m1", journal=journal)
    index = MemoryIndex(Path(str(tmp_path)) / "index_b", embedder=_Embedder(), distiller=distiller)
    index.open()
    journal.bind(index)

    index.rebuild_from_hub(ledger, full=True)
    assert counter["n"] == 4, "第一遍应调 4 次（全部解析失败）"

    counter["n"] = 0
    index.rebuild_from_hub(ledger, full=True)
    # 解析失败不缓存 -> 第二遍重蒸（二次机会），仍是 4 调用（非 0）
    assert counter["n"] == 4, (
        f"解析失败不缓存，第二遍应重蒸给二次机会（4 调用），got {counter['n']}"
    )
    # parse_fail 计数可观测（每轮 4 次，两轮 >= 4）
    assert distiller.stats()["parse_fails"] >= 4

    index.close()
    ledger.close()
