"""Memory Index 定向重建过滤测试（2026-07-28）。

覆盖 rebuild_from_hub 的 since/until/agents/exclude_agents 过滤 +
unconsume_matching 反消费 + "带过滤的 full 不清库" 安全语义。

背景：07-25 实体感知 novelty 修复上线后，07-13~07-24 窗口存量被纯 cosine
误丢的 Fact 需要补回；全量重建要重放 6793 turn（含 ~2100 条压测流量），
成本不可接受 -> 定向过滤。
"""

import tempfile
from datetime import UTC, datetime
from pathlib import Path

from bladex_proxy.models import Identity, Turn, TurnStatus
from bladex_proxy.storage.memory_hub import MemoryHub
from bladex_proxy.storage.memory_index import MemoryIndex, _turn_matches_filter


class MockEmbedder:
    """确定性 mock embedder（与 test_index_rebuild_t6 一致）。"""

    def embed(self, texts):
        results = []
        for t in texts:
            h = hash(t) % 100
            vec = [0.0] * 64
            vec[h % 64] = 1.0
            results.append(vec)
        return results

    @property
    def available(self):
        return True


def _put_turn(ledger: MemoryHub, *, agent: str, day: str, idx: int) -> str:
    """写一条 turn，ts 固定在 day 当天 12:00，返回 ledger key。"""
    identity = Identity(user_id="u1", agent_id=agent, session_id=f"s-{agent}", turn_index=idx)
    turn = Turn(
        identity=identity,
        model="test",
        request_messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user",
             "content": f"Message about topic {agent} {day} {idx}, long enough for consolidation."},
        ],
        response_text=f"Response {agent} {day} {idx}",
        status=TurnStatus.OK,
        ts=datetime.fromisoformat(f"{day}T12:00:00").replace(tzinfo=UTC),
    )
    key = identity.storage_key(f"17199{day.replace('-', '')}{idx}-0")
    ledger.put(key, turn)
    return key


def _make_hub(tmpdir: str) -> tuple[MemoryHub, dict[str, str]]:
    """三个 agent × 三个日期的 Memory Hub，返回 (ledger, {label: key})。"""
    ledger = MemoryHub(Path(tmpdir) / "rocksdb")
    ledger.open()
    keys = {}
    for i, (agent, day) in enumerate([
        ("claude-code", "2026-07-12"),
        ("claude-code", "2026-07-20"),
        ("hermes:default", "2026-07-20"),
        ("test-agent", "2026-07-20"),
        ("claude-code", "2026-07-30"),
    ]):
        keys[f"{agent}@{day}"] = _put_turn(ledger, agent=agent, day=day, idx=i)
    return ledger, keys


def _open_index(tmpdir: str) -> MemoryIndex:
    index = MemoryIndex(Path(tmpdir) / "index", embedder=MockEmbedder(), read_only=False)
    index.open()
    return index


# ── 纯函数：过滤谓词 ──

def test_filter_agent_include_and_exclude():
    key = "u1/claude-code/s1/0001"
    ts = "2026-07-20T12:00:00+00:00"
    assert _turn_matches_filter(key, ts, agents={"claude-code"})
    assert not _turn_matches_filter(key, ts, agents={"hermes:default"})
    assert not _turn_matches_filter(key, ts, exclude_agents={"claude-code"})
    assert _turn_matches_filter(key, ts, exclude_agents={"test-agent"})


def test_filter_until_is_inclusive_for_date_only():
    """until=2026-07-24 含 07-24 全天（前缀比较，不是字符串直比）。"""
    key = "u1/a/s/0001"
    assert _turn_matches_filter(key, "2026-07-24T23:59:59+00:00", until="2026-07-24")
    assert not _turn_matches_filter(key, "2026-07-25T00:00:01+00:00", until="2026-07-24")
    assert _turn_matches_filter(key, "2026-07-13T00:00:00+00:00", since="2026-07-13")
    assert not _turn_matches_filter(key, "2026-07-12T23:59:59+00:00", since="2026-07-13")


def test_filter_missing_ts_excluded_when_time_filter_set():
    """ts 缺失 + 设了时间过滤 -> 保守跳过；无时间过滤则不受影响。"""
    key = "u1/a/s/0001"
    assert not _turn_matches_filter(key, "", since="2026-07-13")
    assert _turn_matches_filter(key, "", agents={"a"})


# ── rebuild 集成 ──

def test_rebuild_time_window_filters_turns():
    """只重放窗口内的 turn，窗口外不产生 fact 也不被消费。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger, keys = _make_hub(tmpdir)
        index = _open_index(tmpdir)

        index.rebuild_from_hub(ledger, since="2026-07-20", until="2026-07-20")

        sources = {f.source_ledger_key for f in index.all_facts()}
        assert keys["claude-code@2026-07-12"] not in sources
        assert keys["claude-code@2026-07-30"] not in sources
        # 窗口外未消费 -> 日后放宽过滤仍能拾起
        assert not index.is_consumed(keys["claude-code@2026-07-12"])
        assert not index.is_consumed(keys["claude-code@2026-07-30"])
        assert index.is_consumed(keys["claude-code@2026-07-20"])

        index.close()
        ledger.close()


def test_rebuild_exclude_agents_skips_load_traffic():
    """排除的 agent（压测流量）不产生 fact、不被消费。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger, keys = _make_hub(tmpdir)
        index = _open_index(tmpdir)

        index.rebuild_from_hub(ledger, exclude_agents={"test-agent"})

        sources = {f.source_ledger_key for f in index.all_facts()}
        assert keys["test-agent@2026-07-20"] not in sources
        assert not index.is_consumed(keys["test-agent@2026-07-20"])
        assert index.is_consumed(keys["hermes:default@2026-07-20"])

        index.close()
        ledger.close()


def test_rebuild_agents_allowlist():
    """agents 白名单：只处理列出的 agent。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger, keys = _make_hub(tmpdir)
        index = _open_index(tmpdir)

        index.rebuild_from_hub(ledger, agents={"hermes:default"})

        for f in index.all_facts():
            assert f.source_ledger_key.split("/")[1] == "hermes:default"

        index.close()
        ledger.close()


def test_full_with_filter_does_not_clear_existing_index():
    """带过滤的 full 不清库——否则过滤掉的历史数据会被静默丢弃。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger, keys = _make_hub(tmpdir)
        index = _open_index(tmpdir)

        # 先全量建一次，拿到 07-12 的 fact
        index.rebuild_from_hub(ledger, full=True)
        before = {f.source_ledger_key for f in index.all_facts()}
        assert keys["claude-code@2026-07-12"] in before

        # 带过滤的 full 只重放 07-20，07-12 的 fact 必须还在
        index.rebuild_from_hub(ledger, full=True, since="2026-07-20", until="2026-07-20")
        after = {f.source_ledger_key for f in index.all_facts()}
        assert keys["claude-code@2026-07-12"] in after

        index.close()
        ledger.close()


def test_unconsume_matching_lets_incremental_replay_window():
    """反消费：删窗口内 consumed 标记后，增量 rebuild 重新拾起这些 turn。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger, keys = _make_hub(tmpdir)
        index = _open_index(tmpdir)

        index.rebuild_from_hub(ledger)
        assert index.is_consumed(keys["claude-code@2026-07-20"])
        assert index.is_consumed(keys["claude-code@2026-07-12"])

        removed = index.unconsume_matching(ledger, since="2026-07-20", until="2026-07-20")
        assert removed >= 1
        # 窗口内标记已删、窗口外保持
        assert not index.is_consumed(keys["claude-code@2026-07-20"])
        assert index.is_consumed(keys["claude-code@2026-07-12"])

        # 下轮增量重新消费该窗口
        index.rebuild_from_hub(ledger)
        assert index.is_consumed(keys["claude-code@2026-07-20"])

        index.close()
        ledger.close()


def test_unconsume_requires_filter():
    """无过滤条件的反消费 = 意外全量重放，必须报错拒绝。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger, _keys = _make_hub(tmpdir)
        index = _open_index(tmpdir)
        try:
            index.unconsume_matching(ledger)
        except ValueError as e:
            assert "过滤条件" in str(e)
        else:
            raise AssertionError("unconsume_matching 无过滤条件时应抛 ValueError")
        finally:
            index.close()
            ledger.close()


# ── ①③ 效率改造：惰性扫描 + backlog 信号 ──

def test_scan_meta_matches_scan_prefix_keys_and_ts():
    """惰性扫描与全量路径给出同一批 key + 同一 ts（等价性）。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger, _keys = _make_hub(tmpdir)

        meta = dict(ledger.scan_meta(""))
        full = {k: (t.ts.isoformat() if t.ts else "") for k, t in ledger.scan_prefix("")}

        assert set(meta) == set(full)
        for k in full:
            # scan_meta 取存储原样字符串，与 Turn.ts 序列化值同点
            assert meta[k][:19] == full[k][:19]

        ledger.close()


def test_scan_meta_skips_tombstoned():
    """惰性扫描同样跳过墓碑覆盖项（与 scan_prefix 语义一致）。"""
    from bladex_proxy.models import TombstoneTargetType
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger, keys = _make_hub(tmpdir)
        victim = keys["hermes:default@2026-07-20"]
        ledger.append_tombstone(TombstoneTargetType.TURN, victim)

        assert victim not in dict(ledger.scan_meta(""))

        ledger.close()


def test_backlog_flag_set_when_max_turns_truncates():
    """max_turns 截断 -> last_rebuild_backlog=True；追平后 False。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger, _keys = _make_hub(tmpdir)
        index = _open_index(tmpdir)

        index.rebuild_from_hub(ledger, max_turns=2)
        assert index.last_rebuild_backlog is True

        # 反复跑到追平
        for _ in range(10):
            if not index.last_rebuild_backlog:
                break
            index.rebuild_from_hub(ledger, max_turns=2)
        assert index.last_rebuild_backlog is False

        index.close()
        ledger.close()


def test_backlog_false_when_nothing_to_do():
    """无新数据时 backlog=False（consolidator 才会 sleep）。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger, _keys = _make_hub(tmpdir)
        index = _open_index(tmpdir)

        index.rebuild_from_hub(ledger)
        index.rebuild_from_hub(ledger)
        assert index.last_rebuild_backlog is False

        index.close()
        ledger.close()


def test_no_filter_rebuild_unchanged():
    """零回归：不带过滤参数时行为与既有增量重建一致（全部消费）。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger, keys = _make_hub(tmpdir)
        index = _open_index(tmpdir)

        index.rebuild_from_hub(ledger)
        for k in keys.values():
            assert index.is_consumed(k)

        index.close()
        ledger.close()
