"""Beta T13 export sync worker 验收（mock exporter）。

任务卡验收原文：mock exporter 单测覆盖增量/墓碑/重试/游标断点/exposure 过滤；
主管线延迟无回归（worker 在 rebuild 之后跑、异常永不外抛——由"目标炸了 worker
不炸"剧本钉住）。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from bladex_core.exporters import ExportBatch
from bladex_core.fact import Fact
from bladex_core.matter import Matter
from bladex_proxy.export_sync import (
    ExporterConfig,
    ExportStateStore,
    ExportSyncWorker,
    JsonlDirExporter,
    build_worker_from_config,
    load_export_config,
)
from bladex_proxy.models import TombstoneTargetType
from bladex_proxy.storage.memory_hub import MemoryHub
from bladex_proxy.storage.memory_index import MemoryIndex


class MockExporter:
    def __init__(self, name: str = "mock", exposure: str = "private",
                 fail: bool = False):
        self.name = name
        self.exposure = exposure
        self.fail = fail
        self.batches: list[ExportBatch] = []
        self.tombstones: list[tuple[str, str]] = []

    def sync_incremental(self, batch: ExportBatch, since_cursor: str) -> None:
        if self.fail:
            raise ConnectionError("target unreachable")
        self.batches.append(batch)

    def handle_tombstone(self, target_type: str, target_key: str) -> None:
        if self.fail:
            raise ConnectionError("target unreachable")
        self.tombstones.append((target_type, target_key))


@pytest.fixture()
def env(tmp_path: Path):
    index = MemoryIndex(tmp_path / "index", embedder=None, read_only=False)
    index.open()
    ledger = MemoryHub(tmp_path / "ledger")
    ledger.open()
    yield index, ledger, tmp_path
    index.close()
    ledger.close()


def _cfg(**kw) -> ExporterConfig:
    base = {"type": "mock", "name": "mock", "exposure": "private"}
    base.update(kw)
    return ExporterConfig(**base)


def _fact(fid: str, ts: datetime, ceiling: str = "public", scope: str = "") -> Fact:
    return Fact(id=fid, content=f"内容 {fid}", source_user_id="u1",
                created_at=ts, exposure_ceiling=ceiling, scope=scope)


def test_incremental_cursor_advance(env):
    """首轮全量 → 游标推进 → 第二轮只出新增。"""
    index, ledger, tmp = env
    t0 = datetime.now(UTC)
    index.add_fact(_fact("f1", t0))
    exp = MockExporter()
    worker = ExportSyncWorker(index, ledger, [(_cfg(), exp)], tmp / "state.json",
                              hard_rules=["MUST x"])
    r1 = worker.run_once()
    assert r1["mock"]["synced"] == 1
    assert exp.batches[0].hard_rules == ["MUST x"]  # 首轮带 hard rules

    r2 = worker.run_once()
    assert r2["mock"]["synced"] == 0  # 无新增

    index.add_fact(_fact("f2", t0 + timedelta(seconds=5)))
    r3 = worker.run_once()
    assert r3["mock"]["synced"] == 1
    assert [f.id for f in exp.batches[-1].facts] == ["f2"]


def test_cursor_persisted_across_worker_restart(env):
    """游标断点：新 worker 实例从状态文件续传，不重复导出。"""
    index, ledger, tmp = env
    index.add_fact(_fact("f1", datetime.now(UTC)))
    exp1 = MockExporter()
    ExportSyncWorker(index, ledger, [(_cfg(), exp1)], tmp / "state.json").run_once()
    assert len(exp1.batches) == 1

    exp2 = MockExporter()
    r = ExportSyncWorker(index, ledger, [(_cfg(), exp2)], tmp / "state.json").run_once()
    assert r["mock"]["synced"] == 0
    assert exp2.batches == []  # 断点续传：不重复


def test_tombstone_sync_once(env):
    index, ledger, tmp = env
    ledger.append_tombstone(target_type=TombstoneTargetType.FACT, target_key="f-gone")
    exp = MockExporter()
    worker = ExportSyncWorker(index, ledger, [(_cfg(), exp)], tmp / "state.json")
    r1 = worker.run_once()
    assert r1["mock"]["tombstones"] == 1
    assert exp.tombstones == [("fact", "f-gone")]
    r2 = worker.run_once()
    assert r2["mock"]["tombstones"] == 0  # done 集合去重


def test_failure_backoff_and_recovery(env):
    """目标不可达：退避不阻塞；恢复后游标不丢（失败轮不推进）。"""
    index, ledger, tmp = env
    index.add_fact(_fact("f1", datetime.now(UTC)))
    exp = MockExporter(fail=True)
    now = [1000.0]
    worker = ExportSyncWorker(index, ledger, [(_cfg(), exp)], tmp / "state.json",
                              now=lambda: now[0])
    r1 = worker.run_once()
    assert "error" in r1["mock"]
    st = worker.state.get("mock")
    assert st["failures"] == 1
    assert st["next_attempt"] > 1000.0
    assert st["cursor"] == ""  # 失败不推进游标

    # 退避窗口内：跳过
    r2 = worker.run_once()
    assert r2["mock"]["skipped"] == "backoff"

    # 目标恢复 + 时间过窗 → 数据补上
    exp.fail = False
    now[0] = 1000.0 + 3600 * 24
    r3 = worker.run_once()
    assert r3["mock"]["synced"] == 1
    assert worker.state.get("mock")["failures"] == 0


def test_exporter_crash_does_not_break_worker_or_siblings(env):
    """一个目标炸了：其它目标照常同步（主管线不阻塞语义）。"""
    index, ledger, tmp = env
    index.add_fact(_fact("f1", datetime.now(UTC)))
    bad, good = MockExporter("bad", fail=True), MockExporter("good")
    worker = ExportSyncWorker(
        index, ledger, [(_cfg(name="bad"), bad), (_cfg(name="good"), good)],
        tmp / "state.json")
    r = worker.run_once()  # 不抛
    assert "error" in r["bad"]
    assert r["good"]["synced"] == 1


def test_exposure_filter_fact_never_leaves(env):
    """敏感联动：ceiling=local 的 Fact 不出境到 private/public 目标。"""
    index, ledger, tmp = env
    t0 = datetime.now(UTC)
    index.add_fact(_fact("f-local", t0, ceiling="local"))
    index.add_fact(_fact("f-pub", t0, ceiling="public"))
    exp = MockExporter(exposure="private")
    worker = ExportSyncWorker(index, ledger, [(_cfg(exposure="private"), exp)],
                              tmp / "state.json")
    worker.run_once()
    ids = [f.id for f in exp.batches[0].facts]
    assert ids == ["f-pub"]

    # local 目标能收到全部（local 是最严目的地）
    exp2 = MockExporter("loc", exposure="local")
    worker2 = ExportSyncWorker(index, ledger, [(_cfg(name="loc", exposure="local"), exp2)],
                               tmp / "state2.json")
    worker2.run_once()
    assert {f.id for f in exp2.batches[0].facts} == {"f-local", "f-pub"}


def test_scope_filter(env):
    index, ledger, tmp = env
    t0 = datetime.now(UTC)
    index.add_fact(_fact("f-personal", t0, scope="personal:abc"))
    index.add_fact(_fact("f-team", t0, scope="team:x"))
    exp = MockExporter()
    worker = ExportSyncWorker(
        index, ledger, [(_cfg(scopes=["team:x"]), exp)], tmp / "state.json")
    worker.run_once()
    assert [f.id for f in exp.batches[0].facts] == ["f-team"]


def test_matters_and_edges_synced(env):
    index, ledger, tmp = env
    index.add_matter(Matter(matter_id="m1", title="事项"))
    exp = MockExporter()
    worker = ExportSyncWorker(index, ledger, [(_cfg(), exp)], tmp / "state.json")
    r = worker.run_once()
    assert r["mock"]["synced"] == 1
    assert exp.batches[0].matters[0].matter_id == "m1"


def test_load_export_config_and_builtin_jsonl(tmp_path: Path):
    cfg_file = tmp_path / "export.toml"
    cfg_file.write_text(
        '[[exporters]]\ntype = "jsonl"\nname = "out"\nexposure = "private"\n'
        f'[exporters.options]\ndir = "{tmp_path / "out"}"\n')
    configs = load_export_config(cfg_file)
    assert len(configs) == 1 and configs[0].type == "jsonl"

    # 不存在的文件 = 关闭
    assert load_export_config(tmp_path / "nope.toml") == []

    # 重名拒绝
    dup = tmp_path / "dup.toml"
    dup.write_text('[[exporters]]\ntype="jsonl"\nname="a"\n'
                   '[[exporters]]\ntype="jsonl"\nname="a"\n')
    with pytest.raises(ValueError, match="unique"):
        load_export_config(dup)

    # JsonlDirExporter 落盘闭环
    exp = JsonlDirExporter("out", "private", {"dir": str(tmp_path / "out")})
    exp.sync_incremental(ExportBatch(facts=[Fact(id="f1", content="x")]), "")
    exp.handle_tombstone("fact", "f1")
    assert (tmp_path / "out/entities.jsonl").exists()
    assert (tmp_path / "out/tombstones.jsonl").exists()


def test_build_worker_from_config_disabled_or_missing(env, tmp_path: Path):
    index, ledger, _ = env
    assert build_worker_from_config(index, ledger, tmp_path / "nope.toml") is None
    off = tmp_path / "off.toml"
    off.write_text('[[exporters]]\ntype="jsonl"\nname="a"\nenabled=false\n')
    assert build_worker_from_config(index, ledger, off) is None
    on = tmp_path / "on.toml"
    on.write_text(f'[[exporters]]\ntype="jsonl"\nname="a"\n'
                  f'[exporters.options]\ndir="{tmp_path / "o"}"\n')
    assert build_worker_from_config(index, ledger, on) is not None


def test_state_reset(tmp_path: Path):
    store = ExportStateStore(tmp_path / "s.json")
    st = store.get("a")
    st["cursor"] = "2026-08-03T00:00:00"
    store.save()
    store2 = ExportStateStore(tmp_path / "s.json")
    assert store2.get("a")["cursor"] == "2026-08-03T00:00:00"
    store2.reset("a")
    assert ExportStateStore(tmp_path / "s.json").get("a")["cursor"] == ""
