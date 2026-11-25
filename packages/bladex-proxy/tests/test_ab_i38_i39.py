"""批 AB · MQ-I38（换名后读端自己重开）+ MQ-I39（full 同块就地重试 + 拍板 (42)-d 放行条件）。

| 用例 | 判据 |
|---|---|
| `test_reader_follows_swap` | 只读实例在换名后读到新库；`index_reader_reopened` 可 grep |
| `test_in_flight_reader_keeps_old_handle` | 替换后，手里拿着旧句柄的读仍然成功 |
| `test_reopen_failure_stays_on_old_index` | 打开失败：继续读旧库、告警、不立刻重试 |
| `test_no_swap_no_reopen` | inode 没变不重开（阴性对照） |
| `test_small_library_one_bad_block_completes` | 小库一块全坏：full 跑完；该块恰好调 1 + MAX 次；打 retry_exhausted |
| `test_all_blocks_bad_refuses_release` | > 3 块全坏（上游回垃圾）：不放行、中止、打 release_refused |
| `test_one_bad_block_among_good_is_released` | 有块成功：坏块放行，跑完 |
| `test_call_errors_are_not_retried_or_released` | 401 不是解析失败：不重试、不放行 |
| `test_network_error_does_not_take_this_path` | 503 走退避，不计入就地重试 |
| `test_incremental_keeps_cross_pass_counting` | 增量不就地重试 |
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest
import structlog.testing

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("_aa2_hub", _HERE / "test_aa2_rebuild_hub_only.py")
_aa2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_aa2)


# ── I38 ──────────────────────────────────────────────────────────────────────

def _library(path: Path, fact_id: str) -> None:
    from bladex_core.fact import Fact
    from bladex_proxy.storage.memory_index import MemoryIndex

    ns = _aa2._stubs()
    idx = MemoryIndex(path, embedder=ns["BigramEmbedder"]())
    idx.open()
    idx.add_fact(Fact(id=fact_id, content=f"内容 {fact_id}"))
    idx.close()


def _reader(path: Path) -> Any:
    from bladex_proxy.storage.memory_index import MemoryIndex

    ns = _aa2._stubs()
    ro = MemoryIndex(path, embedder=ns["BigramEmbedder"](), read_only=True)
    ro.open()
    ro._catchup_throttle_s = 0.0
    ro._try_catch_up()          # 记下当前 inode
    return ro


def _swap(prod: Path, new: Path) -> None:
    from bladex_proxy.storage import index_staging as stg

    stg.exchange(new, prod)


def _settle(ro: Any) -> None:
    import threading

    ro._try_catch_up()
    for t in threading.enumerate():
        if t.name == "index-reader-reopen":
            t.join(timeout=30)


@pytest.fixture()
def libs(tmp_path: Path) -> tuple[Path, Path]:
    prod, new = tmp_path / "prod", tmp_path / "new"
    _library(prod, "fact_old")
    _library(new, "fact_new")
    return prod, new


def test_reader_follows_swap(libs: tuple[Path, Path]) -> None:
    prod, new = libs
    ro = _reader(prod)
    assert ro.get_fact("fact_old") is not None
    _swap(prod, new)
    with structlog.testing.capture_logs() as logs:
        _settle(ro)
    assert [e for e in logs if e["event"] == "index_reader_reopened"]
    assert ro.get_fact("fact_new") is not None
    assert ro.get_fact("fact_old") is None
    ro.close()


def test_in_flight_reader_keeps_old_handle(libs: tuple[Path, Path]) -> None:
    prod, new = libs
    ro = _reader(prod)
    in_flight = ro.meta_db
    _swap(prod, new)
    _settle(ro)
    assert ro.meta_db is not in_flight
    assert in_flight.get(b"fact/fact_old") is not None, "旧句柄不许被替换动作关掉"
    ro.close()


def test_reopen_failure_stays_on_old_index(libs: tuple[Path, Path],
                                           monkeypatch: pytest.MonkeyPatch) -> None:
    from bladex_proxy.storage.memory_index import MemoryIndex

    prod, new = libs
    ro = _reader(prod)
    _swap(prod, new)

    def boom(self: Any) -> None:
        raise RuntimeError("cannot open")

    monkeypatch.setattr(MemoryIndex, "open", boom)
    with structlog.testing.capture_logs() as logs:
        _settle(ro)
    assert [e for e in logs if e["event"] == "index_reader_reopen_failed"]
    monkeypatch.undo()
    assert ro.get_fact("fact_old") is not None, "打开失败要留在旧库"
    assert ro._reopen_not_before > 0
    ro.close()


def test_no_swap_no_reopen(libs: tuple[Path, Path]) -> None:
    prod, _ = libs
    ro = _reader(prod)
    before = ro.meta_db
    with structlog.testing.capture_logs() as logs:
        _settle(ro)
        _settle(ro)
    assert ro.meta_db is before
    assert not [e for e in logs if e["event"] == "index_reader_reopened"]
    ro.close()


# ── I39 ──────────────────────────────────────────────────────────────────────

class _Stub(_aa2._stubs()["Stub"]):      # type: ignore[misc]
    """`verdict`: None = 桩的默认行为；"garbage" = 裁决一律回非 JSON；Exception = 抛。"""
    verdict: Any = None

    def completion(self, **kw: Any) -> Any:
        system = kw["messages"][0]["content"]
        if "write-time adjudicator" in system and self.verdict is not None:
            self.calls["adjudicate"] += 1
            if isinstance(self.verdict, Exception):
                raise self.verdict
            return _aa2._stubs()["_resp"]("sorry, no JSON today")
        return super().completion(**kw)


class _Status(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(f"status {code}")
        self.status_code = code


def _full(tmp: Path, hub: Path, stub: Any, monkeypatch: pytest.MonkeyPatch, *,
          chunk: int = 1, full: bool = True) -> tuple[Any, list[dict]]:
    from bladex_proxy import adjudicator as adj
    from bladex_proxy import router_sdk
    from bladex_proxy.storage.memory_hub import MemoryHub

    monkeypatch.setattr(router_sdk, "completion", stub.completion)
    clock = [0.0]
    monkeypatch.setattr(adj, "_monotonic", lambda: clock[0])
    monkeypatch.setattr(adj, "_sleep", lambda s: clock.__setitem__(0, clock[0] + s))
    monkeypatch.setenv("BLADEX_ADJUDICATE_CHUNK", str(chunk))
    monkeypatch.setenv("BLADEX_ADJUDICATE_OUTAGE_MAX_WAIT_S", "900")
    idx = _aa2._stubs()["build_index"](tmp / "idx", stub)
    h = MemoryHub(hub)
    h.open()
    with structlog.testing.capture_logs() as logs:
        idx.rebuild_from_hub(h, full=full, max_turns=30, distill_concurrency=4)
    h.close()
    return idx, logs


def _many_hub(path: Path, n: int = 9) -> Path:
    """n 轮相近话语：候选互为近邻，`chunk=1` 时块数 > 3。"""
    from datetime import UTC, datetime

    from bladex_proxy.models import Identity, Turn, TurnStatus
    from bladex_proxy.storage.memory_hub import MemoryHub

    hub = MemoryHub(path)
    hub.open()
    for i in range(n):
        ident = Identity(user_id="u1", agent_id="codex", session_id="s", turn_index=i)
        hub.put(ident.storage_key(f"17999100{i:02d}-0"), Turn(
            identity=ident, model="m", status=TurnStatus.OK, response_text="收到",
            request_messages=[{"role": "user", "content":
                               f"BladeX 全量重建的当前进度是第 {i} 阶段，负责人是 Jason。"}],
            ts=datetime(2026, 10, 22, 9, 0, i, tzinfo=UTC)))
    hub.close()
    return path


def _events(logs: list[dict], name: str) -> list[dict]:
    return [e for e in logs if e["event"] == name]


@pytest.fixture()
def hub(tmp_path: Path) -> Path:
    return _aa2._make_hub(tmp_path / "hub")


def test_small_library_one_bad_block_completes(tmp_path: Path, hub: Path,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    from bladex_core.flags import flag_number

    stub = _Stub()
    stub.verdict = "garbage"
    idx, logs = _full(tmp_path, hub, stub, monkeypatch, chunk=100)      # 全部候选一块
    assert idx.last_rebuild_abort_reason is None, "小库一块全坏不许永远重建不完"
    assert idx.fact_count() > 0
    assert _events(logs, "index_rebuild_adjudicate_retry_exhausted")
    assert stub.calls["adjudicate"] == 1 + int(flag_number("BLADEX_ADJUDICATE_RETRY_MAX"))
    idx.close()


def test_all_blocks_bad_refuses_release(tmp_path: Path,
                                        monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _Stub()
    stub.verdict = "garbage"
    many = _many_hub(tmp_path / "many")
    idx, logs = _full(tmp_path, many, stub, monkeypatch, chunk=1)       # > 3 块，全坏
    assert idx._adjudicator.block_outcome()[0] > 3
    assert idx.last_rebuild_abort_reason == "adjudicate_outage"
    assert _events(logs, "index_rebuild_adjudicate_release_refused")
    assert not _events(logs, "index_rebuild_adjudicate_retry_exhausted")
    idx.close()


def test_one_bad_block_among_good_is_released(tmp_path: Path, hub: Path,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _Stub()
    stub.garbage_marker = "第 2 阶段"
    idx, logs = _full(tmp_path, hub, stub, monkeypatch, chunk=1)
    assert idx.last_rebuild_abort_reason is None
    assert _events(logs, "index_rebuild_adjudicate_retry_exhausted")
    assert any("第 2 阶段" in f.content for f in idx.all_facts())
    idx.close()


def test_call_errors_are_not_retried_or_released(tmp_path: Path, hub: Path,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _Stub()
    stub.verdict = _Status(401)
    idx, logs = _full(tmp_path, hub, stub, monkeypatch, chunk=100)
    assert stub.calls["adjudicate"] == 1, "401 重试也没用"
    assert idx.last_rebuild_abort_reason == "adjudicate_outage", "401 不是坏输入，不放行"
    assert not _events(logs, "index_rebuild_adjudicate_retry_exhausted")
    idx.close()


def test_network_error_does_not_take_this_path(tmp_path: Path, hub: Path,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _Stub()
    stub.verdict = _Status(503)
    idx, logs = _full(tmp_path, hub, stub, monkeypatch, chunk=100)
    assert idx.last_rebuild_abort_reason == "adjudicate_upstream_unavailable"
    assert not _events(logs, "index_rebuild_adjudicate_retry_exhausted")
    idx.close()


def test_incremental_keeps_cross_pass_counting(tmp_path: Path, hub: Path,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _Stub()
    stub.verdict = "garbage"
    idx, logs = _full(tmp_path, hub, stub, monkeypatch, chunk=100, full=False)
    assert stub.calls["adjudicate"] == 1, "增量不就地重试"
    assert idx.last_rebuild_abort_reason == "adjudicate_outage"
    idx.close()
