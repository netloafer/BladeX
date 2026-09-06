"""存储锁冲突要给出可照做的提示，而不是甩 traceback（2026-08-05，C7 顺带）。

实测触发：常驻 consolidator 还活着时跑 `bladex-consolidator --unconsume`，得到

    Exception: IO error: While lock file: data/bladex_index/meta_rocksdb/LOCK:
    Resource temporarily unavailable

外加一整段 traceback。这不是 bug —— Memory Index 单写者是设计（ADR-0009 §7）—— 但**预期内的
失败必须告诉用户下一步敲什么**，否则得读源码才知道"先停另一个进程"。这与 ADR-0027 §3
的可观测性契约是同一条：灯要说真话，且话要能照做。

两种锁冲突的解法不同，所以提示分开：
- Memory Index 被另一个 consolidator 占（单写者）；
- Memory Hub read-write 被 proxy 占（只有 --full 会要这把锁，增量走 secondary 不需要）。
"""

from __future__ import annotations

import pytest
from bladex_proxy.consolidator import _lock_conflict_hint

# 真实报错原文（macOS，rocksdict 0.3.x）——照抄，别改写成"差不多的字符串"，
# 上游换措辞时这条才会红给我们看。
_REAL_INDEX_LOCK_ERROR = (
    "IO error: While lock file: data/bladex_index/meta_rocksdb/LOCK: "
    "Resource temporarily unavailable"
)


def test_real_index_lock_error_is_recognised():
    hint = _lock_conflict_hint(Exception(_REAL_INDEX_LOCK_ERROR), "Memory Index")
    assert hint is not None


def test_index_hint_names_the_command_to_run():
    """提示的价值在于"能照做"——必须点名停哪个进程、用哪条命令。"""
    hint = _lock_conflict_hint(Exception(_REAL_INDEX_LOCK_ERROR), "Memory Index")
    assert "run_consolidator.sh stop" in hint
    assert "bladex status" in hint
    assert "single writer" in hint  # 说明这是设计而非故障，省得用户去查 bug


def test_ledger_hint_points_at_the_proxy_not_the_consolidator():
    """Memory Hub read-write 锁的占用方是 proxy，解法完全不同——提示不能张冠李戴。"""
    hint = _lock_conflict_hint(
        Exception("IO error: While lock file: data/bladex_hub/LOCK: "
                  "Resource temporarily unavailable"), "Memory Hub")
    assert hint is not None
    assert "bladex stop" in hint
    assert "--full" in hint
    assert "secondary" in hint  # 顺带告诉用户增量模式根本不需要这把锁


@pytest.mark.parametrize("msg", [
    "IO error: While lock file: /x/LOCK: No locks available",
    "io error: while lock file: /x/LOCK: resource temporarily unavailable",
])
def test_other_lock_wordings_also_recognised(msg):
    assert _lock_conflict_hint(Exception(msg), "Memory Index") is not None


@pytest.mark.parametrize("msg", [
    # 🔴 向量空间不一致：自带指引（scripts/reembed_index.py），必须原样抛，不许被当成锁冲突吞掉
    "Memory Index embed model mismatch: sealed=local:e5-large current=local:bge-small; "
    "run scripts/reembed_index.py",
    "Corruption: CURRENT file does not end with newline",
    "No such file or directory",
])
def test_non_lock_errors_are_not_swallowed(msg):
    assert _lock_conflict_hint(Exception(msg), "Memory Index") is None
