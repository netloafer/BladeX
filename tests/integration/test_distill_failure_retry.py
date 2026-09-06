"""MS-1（复核 D4）：蒸馏失败轮不标记消费 —— 下轮重试。

## 这条链在修什么

`LLMDistiller.distill` 失败时**返回空 DistillOutput 而不抛**（降级不停摆）。
于是在 rebuild 看来，"上游断了一条都没成"与"这批本来就没事实"完全一样，
而消费标记是无条件打的 —— 那一轮的事实就永久丢了（台账不缓存失败，
所以它本来是零成本可重试的）。

守卫（`BLADEX_DISTILL_OUTAGE_ABORT_RATIO`）兜"整批中断"；本卡兜的是**零星失败**：
失败率在 (0, 阈值) 区间时，把失败的那些 turn 从消费列表里摘掉。

## 队列不许停摆（2026-08-06 用户拍板）

只做上面那条会留一个洞：积压尾部只剩几条**永久坏**的 turn 时，失败率恒为 1.0
→ 整批被守卫接管 → 谁也不前进。所以失败按"上游活着没有"分两种：

| `failure_kind` | 含义 | 处置 |
|---|---|---|
| `parse` | 响应拿到了、内容用不了 → **上游活着**，问题在这条输入上 | 计重试次数；超限**照常消费**（重放多少次都一样，不能堵死队列） |
| `call` | 连不上 / 超时 / 认证 / 配额 → 上游的问题 | **一次都不计**（计了就等于断线 N 分钟后把好数据当坏数据丢掉 = 2026-08-05 事故） |

走的是归一层集成测试纪律（conventions §5.1）：断言在真实 `rebuild_from_hub`
路径上，不复制逻辑到测试里。
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
        self.finish_reason = "stop"


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _Embedder:
    def __init__(self) -> None:
        self._dim = 64

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
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


# 10 条消息，其中 3 条（含 BAD）注定解析失败 = 30% 失败率，低于 0.5 守卫阈值
_GOOD = [f"用户决定把第 {i} 项配置改成新的默认值以便后续复用" for i in range(7)]
_BAD = [f"这条消息注定蒸馏失败编号 {i} 用于验证重试语义" for i in range(3)]
_MESSAGES = _GOOD + _BAD


def _make_hub(tmpdir: Path) -> MemoryHub:
    ledger = MemoryHub(tmpdir / "ledger")
    ledger.open()
    for i, msg in enumerate(_MESSAGES):
        ident = Identity(user_id="u1", agent_id="a1", session_id="s1", turn_index=i)
        ledger.put(ident.storage_key(f"17199072{i:02d}-0"), Turn(
            identity=ident, model="test",
            request_messages=[{"role": "user", "content": msg}],
            response_text="ok", status=TurnStatus.OK, logical_turn=i,
        ))
    return ledger


def _install_partial_failure(monkeypatch) -> dict[str, int]:
    router_sdk = pytest.importorskip("bladex_proxy.router_sdk")
    counter = {"calls": 0, "bad": 0}

    def fake_completion(**kw):
        counter["calls"] += 1
        content = kw["messages"][-1]["content"]
        if "注定蒸馏失败" in content:
            counter["bad"] += 1
            return _FakeResponse("not json at all")     # → parse fail → failed=True
        return _FakeResponse(
            '{"facts":[{"content":"一条被记住的决定","kind":"decision"}],'
            '"matter_proposals":[]}'
        )

    monkeypatch.setattr(router_sdk, "completion", fake_completion)
    return counter


def _build(tmp_path: Path, name: str):
    journal = IndexDistillJournal()
    distiller = LLMDistiller(model="m1", journal=journal)
    index = MemoryIndex(tmp_path / name, embedder=_Embedder(), distiller=distiller)
    index.open()
    journal.bind(index)
    return index


def test_failed_turns_are_not_consumed_and_retried(tmp_path, monkeypatch):
    """30% 失败率：失败的 3 轮不消费，下一轮 rebuild 重新出现在候选里。"""
    counter = _install_partial_failure(monkeypatch)
    ledger = _make_hub(tmp_path)
    index = _build(tmp_path, "idx")

    index.rebuild_from_hub(ledger)
    assert counter["bad"] == 3, "第一轮应有 3 次解析失败"
    consumed_after_first = index.consumed_count()
    assert consumed_after_first == 7, (
        f"只有成功的 7 轮该被消费，实际 {consumed_after_first} —— "
        "失败轮被消费掉了，它们的事实永久丢失（D4）"
    )

    # 第二轮：失败轮重新进候选（成功轮走 consumed 不重复）
    counter["calls"] = counter["bad"] = 0
    index.rebuild_from_hub(ledger)
    assert counter["bad"] == 3, "失败轮应在下一轮被重试"
    assert counter["calls"] == 3, (
        f"成功轮不得重复蒸馏，实际总调用 {counter['calls']}"
    )

    index.close()
    ledger.close()


def _append_good_turns(ledger: MemoryHub, batch: int, count: int = 4) -> None:
    """再灌若干条必定成功的新轮次（模拟流量持续进来）。

    两个约束决定了 count：
      ① 每一 pass 必须有成功调用 —— "上游是好的"这个前提要有证据，
         重试计数才允许推进（见 memory_index 里那段注释）；
      ② 失败率必须 < outage 阈值（0.5），否则整批被守卫接管。
         残留 3 条坏轮 → 至少要 4 条好轮才把 ratio 压到 3/7 < 0.5。
    """
    for i in range(count):
        n = batch * 100 + i
        ident = Identity(user_id="u1", agent_id="a1", session_id="s2", turn_index=n)
        ledger.put(ident.storage_key(f"1719908{n:03d}-0"), Turn(
            identity=ident, model="test",
            # 长度须过 _MIN_CONTENT_CHARS（20），否则根本不产生蒸馏调用。
            request_messages=[{"role": "user",
                               "content": f"用户又决定了第 {n} 件事情，需要长期记住并在后续复用"}],
            response_text="ok", status=TurnStatus.OK, logical_turn=n,
        ))


def test_retry_cap_eventually_consumes(tmp_path, monkeypatch):
    """永久失败的输入不得永远滞留：超过 BLADEX_DISTILL_RETRY_MAX 后照常消费。"""
    monkeypatch.setenv("BLADEX_DISTILL_RETRY_MAX", "2")
    _install_partial_failure(monkeypatch)
    ledger = _make_hub(tmp_path)
    index = _build(tmp_path, "idx_cap")

    index.rebuild_from_hub(ledger)              # retry=1 → 缓期；7 条成功轮消费
    assert index.consumed_count() == 7

    _append_good_turns(ledger, 1)
    index.rebuild_from_hub(ledger)              # retry=2 → 仍缓期；+4 好轮
    assert index.consumed_count() == 11

    _append_good_turns(ledger, 2)
    index.rebuild_from_hub(ledger)              # retry=3 > 2 → 3 条坏轮一并消费
    assert index.consumed_count() == 18, "超限后必须消费，否则坏输入永久滞留"

    index.close()
    ledger.close()


def test_queue_does_not_stall_when_only_bad_turns_remain(tmp_path, monkeypatch):
    """🔴 队列不许停摆：只剩永久坏轮时（失败率 1.0，走 outage 分支）也要能前进。

    这些失败是 `parse` 类——响应拿到了、内容用不了，说明**上游活着**，
    问题在这几条输入上，重放多少次都一样。所以照常计次、超限放行。
    """
    monkeypatch.setenv("BLADEX_DISTILL_RETRY_MAX", "2")
    _install_partial_failure(monkeypatch)
    ledger = _make_hub(tmp_path)
    index = _build(tmp_path, "idx_drain")

    index.rebuild_from_hub(ledger)      # 7 好轮消费；3 坏轮 retry=1
    assert index.consumed_count() == 7

    index.rebuild_from_hub(ledger)      # 只剩 3 坏轮 → ratio=1.0 → outage 分支，retry=2
    assert index.consumed_count() == 7
    index.rebuild_from_hub(ledger)      # retry=3 > 2 → 放行
    assert index.consumed_count() == 10, "永久坏轮必须能被排空，否则整条队列堵死"

    index.close()
    ledger.close()


def test_connection_outage_never_burns_the_backlog(tmp_path, monkeypatch):
    """🔴 反向红线：**连接类**失败（上游断了）一次都不许计重试。

    这是 2026-08-05 事故的形状——上游 DNS 断掉，consolidator 以 30 turn/分钟
    把积压全部标记成已消费且零事实。跑再多轮也不许有任何 turn 被消费。
    """
    router_sdk = pytest.importorskip("bladex_proxy.router_sdk")
    monkeypatch.setenv("BLADEX_DISTILL_RETRY_MAX", "1")

    def boom(**kw):
        raise ConnectionError("upstream unreachable (simulated DNS failure)")

    monkeypatch.setattr(router_sdk, "completion", boom)
    ledger = _make_hub(tmp_path)
    index = _build(tmp_path, "idx_outage_calls")

    for _ in range(5):
        index.rebuild_from_hub(ledger)
    assert index.consumed_count() == 0, "断线期间一条都不许消费（2026-08-05 事故红线）"
    for k, _ in ledger.scan_prefix(""):
        assert index._distill_retry_count(k) == 0, "call 失败不得计入重试次数"

    index.close()
    ledger.close()


def test_failure_kind_is_tagged_on_the_output():
    """两种失败在**返回值上**就分得开（下游据此决定计不计次）。"""
    from bladex_core.distillation import DistillOutput

    assert DistillOutput().failure_kind == ""
    assert DistillOutput(failed=True, failure_kind="parse").failure_kind == "parse"
    assert DistillOutput(failed=True, failure_kind="call").failure_kind == "call"


def test_retry_disabled_restores_old_behaviour(tmp_path, monkeypatch):
    """`BLADEX_DISTILL_RETRY_MAX=0` 回滚到旧行为（失败也照常消费）。"""
    monkeypatch.setenv("BLADEX_DISTILL_RETRY_MAX", "0")
    _install_partial_failure(monkeypatch)
    ledger = _make_hub(tmp_path)
    index = _build(tmp_path, "idx_off")

    index.rebuild_from_hub(ledger)
    assert index.consumed_count() == 10

    index.close()
    ledger.close()


def test_outage_guard_still_wins(tmp_path, monkeypatch):
    """整批中断（失败率 ≥ 阈值）语义不变：**一条都不消费**，不受本卡影响。"""
    router_sdk = pytest.importorskip("bladex_proxy.router_sdk")
    monkeypatch.setattr(
        router_sdk, "completion", lambda **kw: _FakeResponse("garbage"))
    ledger = _make_hub(tmp_path)
    index = _build(tmp_path, "idx_outage")

    index.rebuild_from_hub(ledger)
    assert index.consumed_count() == 0

    index.close()
    ledger.close()


def test_success_clears_retry_counter(tmp_path, monkeypatch):
    """失败→成功后计数归零（下次再失败重新从 1 起算，不吃历史账）。"""
    monkeypatch.setenv("BLADEX_DISTILL_RETRY_MAX", "2")
    _install_partial_failure(monkeypatch)
    ledger = _make_hub(tmp_path)
    index = _build(tmp_path, "idx_clear")

    index.rebuild_from_hub(ledger)
    bad_keys = [k for k, _ in ledger.scan_prefix("")][7:]
    assert all(index._distill_retry_count(k) == 1 for k in bad_keys)

    # 上游恢复：这次全部解析成功
    monkeypatch.setattr(
        router_sdk_module(),
        "completion",
        lambda **kw: _FakeResponse(
            '{"facts":[{"content":"恢复后的一条事实","kind":"decision"}],'
            '"matter_proposals":[]}'),
    )
    index.rebuild_from_hub(ledger)
    assert index.consumed_count() == 10
    assert all(index._distill_retry_count(k) == 0 for k in bad_keys)

    index.close()
    ledger.close()


def router_sdk_module():
    import bladex_proxy.router_sdk as m

    return m
