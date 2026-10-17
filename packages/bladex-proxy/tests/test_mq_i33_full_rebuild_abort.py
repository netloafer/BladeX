"""MQ-I33：full rebuild 被守卫中止时不许把“没做”报成“完成”。

守卫本身在 `rebuild_from_hub` 里是降级不抛的；空库 full rebuild 也会返回 0 条。
因此 done/aborted 只能看 `MemoryIndex.last_rebuild_abort_reason` 这个明确信号。
本文件同时钉住 main() 接线：机制存在不等于 CLI 会用它。
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import structlog.testing
from bladex_core.distillation import DistillOutput
from bladex_proxy import consolidator
from bladex_proxy.models import Identity, Turn, TurnStatus
from bladex_proxy.storage import memory_index as memory_index_module
from bladex_proxy.storage.memory_hub import MemoryHub
from bladex_proxy.storage.memory_index import MemoryIndex


class MockEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * 64
            vec[hash(text) % 64] = 1.0
            out.append(vec)
        return out

    @property
    def available(self) -> bool:
        return True


class StubDistiller:
    model_name = "stub/distiller"
    prompt_ver = "v1-test"

    def distill(self, user_message: str) -> DistillOutput:
        return DistillOutput(
            model_name=self.model_name,
            facts=[{"kind": "event", "item_kind": "assertion",
                    "content": f"stub fact from {user_message[:30]}"}],
        )

    def stats(self) -> dict[str, int]:
        return {"calls": 1, "fails": 0, "parse_fails": 0, "ledger_hits": 0}


class OutageDistiller(StubDistiller):
    """复刻 LLMDistiller 的降级形态：失败不抛、计数器可见。"""

    def distill(self, user_message: str) -> DistillOutput:
        self.calls += 1
        return DistillOutput(model_name=self.model_name)

    def __init__(self) -> None:
        self.calls = 0

    def stats(self) -> dict[str, int]:
        return {"calls": self.calls, "fails": self.calls,
                "parse_fails": 0, "ledger_hits": 0}


def _put_turn(ledger: MemoryHub) -> None:
    identity = Identity(
        user_id="u1", agent_id="codex", session_id="s1", turn_index=1,
    )
    turn = Turn(
        identity=identity,
        model="test",
        request_messages=[{"role": "user", "content": "please remember this detail"}],
        response_text="remembered",
        status=TurnStatus.OK,
        ts=datetime(2026, 9, 28, tzinfo=UTC),
    )
    ledger.put(identity.storage_key("20260928000000-0"), turn)


def _open_index(tmp_path: Path, *, embedder: Any = None,
                distiller: Any = None) -> MemoryIndex:
    index = MemoryIndex(
        tmp_path / "index", embedder=embedder, read_only=False,
        distiller=distiller,
    )
    index.open()
    return index


def _assert_full_cli_abort(index: MemoryIndex, *, reason: str) -> None:
    with structlog.testing.capture_logs() as logs:
        exit_code = consolidator._finish_full_rebuild(index)
    assert exit_code != 0
    events = {log["event"] for log in logs}
    assert "index_consolidator_full_rebuild_aborted" in events
    assert "index_consolidator_full_rebuild_done" not in events
    aborted = next(log for log in logs
                   if log["event"] == "index_consolidator_full_rebuild_aborted")
    assert aborted["reason"] == reason


@contextmanager
def _null_context() -> Iterator[None]:
    yield


class _FakeMemoryHub:
    def __init__(self, _path: str, **_kwargs: Any) -> None:
        pass

    def open(self) -> None:
        pass

    def catch_up(self) -> None:
        pass

    def close(self) -> None:
        pass


class _FakeIndex:
    def __init__(self, *, reason: str | None = None, raise_error: bool = False,
                 **_kwargs: Any) -> None:
        self.last_rebuild_abort_reason = reason
        self.last_rebuild_backlog = False
        self.last_rebuild_candidates = 0
        self._raise_error = raise_error
        self.closed = False

    def open(self) -> None:
        pass

    def attach_funnel(self, _funnel: Any) -> None:
        pass

    def rebuild_from_hub(self, _ledger: Any, **_kwargs: Any) -> int:
        if self._raise_error:
            raise RuntimeError("simulated rebuild failure")
        return 0

    def close(self) -> None:
        self.closed = True


class _FakeIdentityRegistry:
    empty = True

    def legacy_map(self) -> dict[str, str]:
        return {}


class _FakeRoutingConfig:
    def sensitivity_config(self) -> None:
        return None


class _FakeConfig:
    def __init__(self, tmp_path: Path) -> None:
        self.index_path = str(tmp_path / "index")
        self.rocksdb_path = str(tmp_path / "hub")
        self.distill_model = ""
        self.upstream_api_base = ""
        self.upstream_api_key = ""
        self.routing_config = _FakeRoutingConfig()
        self.routing_config_path = str(tmp_path / "routing.toml")
        self.identity_registry = _FakeIdentityRegistry()
        self.effective_hard_rules: list[str] = []
        self.entity_aware_novelty = True
        self.entity_overlap_threshold = 0.6
        self.novelty_topk = 5
        self.entity_aware_rerank = True
        self.entity_rerank_alpha = 0.7
        self.entity_rerank_expand_k = 20


def _run_main_with_fake_index(
    tmp_path: Path, monkeypatch: Any, *, reason: str | None,
    raise_error: bool = False,
) -> int:
    """把 main() 的 full 分支驱动到收尾，不碰真实 Hub/Index/模型。"""
    import sys

    index = _FakeIndex(reason=reason, raise_error=raise_error)
    monkeypatch.setattr(sys, "argv", ["bladex-consolidator", "--full"])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(consolidator.deployment, "find_root", lambda: str(tmp_path))

    import bladex_proxy.config as config_module
    import bladex_proxy.embedding as embedding_module
    import bladex_proxy.hard_rules as hard_rules_module
    import bladex_proxy.modules as modules_module
    import bladex_proxy.routing_config as routing_config_module
    import bladex_proxy.storage.memory_hub as memory_hub_module
    import bladex_proxy.storage.memory_index as memory_index_module
    from bladex_proxy import consolidator_metrics

    monkeypatch.setattr(config_module, "ProxyConfig", lambda: _FakeConfig(tmp_path))
    monkeypatch.setattr(memory_hub_module, "MemoryHub", _FakeMemoryHub)
    monkeypatch.setattr(memory_index_module, "MemoryIndex",
                        lambda *_args, **_kwargs: index)
    monkeypatch.setattr(embedding_module, "build_embedder",
                        lambda *_args, **_kwargs: MockEmbedder())
    monkeypatch.setattr(embedding_module, "embed_purpose",
                        lambda *_args, **_kwargs: _null_context())
    monkeypatch.setattr(embedding_module, "resolve_embed_settings",
                        lambda *_args, **_kwargs: SimpleNamespace(backend="local"))
    monkeypatch.setattr(embedding_module, "validate_embed_sensitivity",
                        lambda *_args, **_kwargs: None)
    monkeypatch.setattr(embedding_module, "effective_thresholds",
                        lambda *_args, **_kwargs: {"novelty": 0.9,
                                                    "semantic": 0.85,
                                                    "digestion": 0.8})
    monkeypatch.setattr(routing_config_module, "bootstrap_distill_model",
                        lambda *_args, **_kwargs: "")
    monkeypatch.setattr(hard_rules_module, "arm", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(modules_module, "module_enabled",
                        lambda *_args, **_kwargs: False)
    monkeypatch.setattr(consolidator_metrics, "start_metrics_server",
                        lambda *_args, **_kwargs: None)
    return consolidator.main()


def consolidator_metrics_module() -> Any:
    from bladex_proxy import consolidator_metrics
    return consolidator_metrics


def test_main_full_rebuild_guard_hit_aborts_with_nonzero_exit(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    """接线测试：main --full 必须消费 index 的明确中止信号。"""
    with structlog.testing.capture_logs() as logs:
        exit_code = _run_main_with_fake_index(
            tmp_path, monkeypatch, reason="adjudicate_outage",
        )
    assert exit_code != 0
    events = {log["event"] for log in logs}
    assert "index_consolidator_full_rebuild_aborted" in events
    assert "index_consolidator_full_rebuild_done" not in events
    aborted = next(log for log in logs
                   if log["event"] == "index_consolidator_full_rebuild_aborted")
    assert aborted["reason"] == "adjudicate_outage"


def test_main_full_rebuild_error_aborts_with_nonzero_exit(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    """rebuild 抛异常也不能落回 done；full 的 reason 必须是 error。"""
    with structlog.testing.capture_logs() as logs:
        exit_code = _run_main_with_fake_index(
            tmp_path, monkeypatch, reason=None, raise_error=True,
        )
    assert exit_code != 0
    events = {log["event"] for log in logs}
    assert "index_consolidator_error" in events
    assert "index_consolidator_full_rebuild_aborted" in events
    assert "index_consolidator_full_rebuild_done" not in events
    aborted = next(log for log in logs
                   if log["event"] == "index_consolidator_full_rebuild_aborted")
    assert aborted["reason"] == "error"


def test_full_rebuild_guard_hit_sets_explicit_abort_signal(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        "bladex_core.adjudication.adjudicate_outage",
        lambda *_args, **_kwargs: (True, 1, 1, 1.0),
    )
    ledger = MemoryHub(tmp_path / "hub")
    ledger.open()
    index = _open_index(tmp_path, embedder=MockEmbedder(), distiller=StubDistiller())
    try:
        _put_turn(ledger)
        assert index.rebuild_from_hub(ledger, full=True) == 0
        assert index.last_rebuild_abort_reason == "adjudicate_outage"
        assert index.consumed_count() == 0
        _assert_full_cli_abort(index, reason="adjudicate_outage")
    finally:
        index.close()
        ledger.close()


def test_full_rebuild_distill_outage_sets_abort_signal(tmp_path: Path) -> None:
    ledger = MemoryHub(tmp_path / "hub")
    ledger.open()
    index = _open_index(tmp_path, embedder=MockEmbedder(),
                        distiller=OutageDistiller())
    try:
        _put_turn(ledger)
        assert index.rebuild_from_hub(ledger, full=True) == 0
        assert index.last_rebuild_abort_reason == "distill_outage"
        assert index.consumed_count() == 0
        _assert_full_cli_abort(index, reason="distill_outage")
    finally:
        index.close()
        ledger.close()


def test_full_rebuild_consolidation_failed_sets_abort_signal(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    def _fail(self: Any, *_args: Any, **_kwargs: Any) -> int:
        raise RuntimeError("simulated consolidation failure")

    monkeypatch.setattr(memory_index_module.ProxyConsolidator,
                       "consolidate_turns", _fail)
    ledger = MemoryHub(tmp_path / "hub")
    ledger.open()
    index = _open_index(tmp_path, embedder=MockEmbedder(), distiller=StubDistiller())
    try:
        _put_turn(ledger)
        assert index.rebuild_from_hub(ledger, full=True) == 0
        assert index.last_rebuild_abort_reason == "consolidation_failed"
        assert index.consumed_count() == 0
        _assert_full_cli_abort(index, reason="consolidation_failed")
    finally:
        index.close()
        ledger.close()


def test_full_rebuild_no_embedder_sets_abort_signal(tmp_path: Path) -> None:
    ledger = MemoryHub(tmp_path / "hub")
    ledger.open()
    index = _open_index(tmp_path, embedder=None)
    try:
        assert index.rebuild_from_hub(ledger, full=True) == 0
        assert index.last_rebuild_abort_reason == "no_embedder"
        assert index.consumed_count() == 0
        _assert_full_cli_abort(index, reason="no_embedder")
    finally:
        index.close()
        ledger.close()


def test_full_rebuild_empty_hub_still_done_with_zero_exit(tmp_path: Path) -> None:
    ledger = MemoryHub(tmp_path / "hub")
    ledger.open()
    index = _open_index(tmp_path, embedder=MockEmbedder())
    try:
        assert index.rebuild_from_hub(ledger, full=True) == 0
        assert index.last_rebuild_abort_reason is None

        with structlog.testing.capture_logs() as logs:
            exit_code = consolidator._finish_full_rebuild(index)
        assert exit_code == 0
        events = {log["event"] for log in logs}
        assert "index_consolidator_full_rebuild_done" in events
        assert "index_consolidator_full_rebuild_aborted" not in events
    finally:
        index.close()
        ledger.close()
