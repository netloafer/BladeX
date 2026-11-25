"""批 AA · 重建可靠性（README「Coming next (0.4.0)」⑤：可中断 · 可续跑 · 断网不作废 · 不留半截索引）。

四个词各钉一组用例（反向变异见结算稿 §9）：

| 词 | 用例 | 判据 |
|---|---|---|
| 不留半截索引（MQ-I34） | `test_kill9_during_write_leaves_prod_byte_identical` / `test_self_check_failure_does_not_swap` | 生产目录树哈希不变（RocksDB 文本 LOG 除外） |
| 断网不作废（MQ-I35） | `test_outage_then_recovery_*` / `test_wait_cap_exceeded_*` | 退避后从同一块继续；blanket ADD = 0 |
| 可中断 + 可续跑 | `test_resume_after_kill_*` | 已完成块 LLM 调用 = 0；最终 fact 内容集合 = 不中断那一跑 |
| （A28）IPC | `test_ipc_*` | 单次请求 ≤ 块上限；超时可配 |

kill / 自检失败 / 等待超限三种情形**在子进程里跑**：生产目录的字节对比必须跨过一次真实的
进程生命周期（被 SIGKILL 的进程来不及做任何收尾，这才是"中止"的真实形态）。
基线哈希取在子进程**打开生产库之后**——RocksDB 每次打开都轮转 MANIFEST/WAL/OPTIONS，
那是打开动作本身的副作用，与重建无关；从打开到退出，生产目录一个字节都不许变。

存储一律在 tmp_path；不读 live 配置（子进程环境由本文件显式给出）。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

# ── 桩：子进程与进程内共用同一份源码（exec 进命名空间）───────────────────────
_STUBS_SRC = r'''
import hashlib, json, math, os, re, time
from types import SimpleNamespace

def _resp(text):
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=text), finish_reason="stop")])

class Down(Exception):
    """可编程上游故障：带 status_code=503 ⇒ `is_transient_upstream_error` 判为不可达。"""
    status_code = 503

class Stub:
    """桩上游：按 system prompt 分流。`fail_adj` = 接下来几次裁决调用抛 503（-1 = 永远）；
    `garbage_marker` = NEW 段内容含它的那块回非 JSON（确定性解析失败）。"""
    def __init__(self, rec=None):
        self.rec = rec or (lambda kind, extra="": None)
        self.fail_adj = 0
        self.garbage_marker = ""
        self.calls = {"distill": 0, "adjudicate": 0, "adjudicate_fail": 0, "link": 0}
        self.adj_hashes = []

    def completion(self, **kw):
        system = kw["messages"][0]["content"]
        user = kw["messages"][-1]["content"]
        if "write-time adjudicator" in system:
            if self.fail_adj:
                self.fail_adj -= 1 if self.fail_adj > 0 else 0
                self.calls["adjudicate_fail"] += 1
                self.rec("adjudicate_fail")
                raise Down("503 Service Unavailable (stub)")
            h = hashlib.sha1(user.encode()).hexdigest()[:12]
            time.sleep(float(os.environ.get("AA_ADJ_SLEEP", "0")))
            self.calls["adjudicate"] += 1
            self.adj_hashes.append(h)
            self.rec("adjudicate", h)
            if self.garbage_marker and any(
                    self.garbage_marker in c for c in
                    re.findall(r"^### NEW \S+\ncontent: (.*)$", user, flags=re.M)):
                return _resp("sorry, no JSON today")
            ids = re.findall(r"^### NEW (\S+)", user, flags=re.M)
            return _resp(json.dumps([{"id": i, "op": "add", "targets": [], "reason": "stub"}
                                     for i in ids]))
        if "attribution judge" in system:
            self.calls["link"] += 1
            self.rec("link")
            return _resp("[]")
        self.calls["distill"] += 1
        self.rec("distill")
        m = re.search(r"USER:\n(.*?)(?:\n\nASSISTANT|\Z)", user, flags=re.S)
        body = m.group(1) if m else ""
        facts = [{"content": s.strip(), "item_kind": "assertion", "lane": "task",
                  "source": "user", "subject": s.strip(), "attribute": "note",
                  "entities": [], "importance": 5, "valid_until": "", "corrects": ""}
                 for s in body.split("。") if len(s.strip()) >= 6]
        return _resp(json.dumps({"topic": "桩", "keywords": ["桩"], "facts": facts,
                                 "matter_proposals": [], "event": "none"},
                                ensure_ascii=False))

class BigramEmbedder:
    """确定性字符二元组向量：只差一个字的两句 cos ≈ 0.95（> 裁决邻居下限 0.90、< 判重 0.98）。"""
    model_name = "aa-bigram"
    model_identity = "test:aa-bigram"
    available = True
    def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * 128
            for i in range(len(t) - 1):
                v[int(hashlib.md5(t[i:i + 2].encode()).hexdigest()[:6], 16) % 128] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out

def build_index(path, stub, model="stub/m"):
    from bladex_proxy import router_sdk
    router_sdk.completion = stub.completion
    from bladex_proxy.adjudicator import LLMAdjudicator
    from bladex_proxy.distillation import LLMDistiller
    from bladex_proxy.storage.memory_index import IndexDistillJournal, MemoryIndex
    journal = IndexDistillJournal()
    index = MemoryIndex(path, embedder=BigramEmbedder(),
                        distiller=LLMDistiller(model=model, journal=journal),
                        adjudicator=LLMAdjudicator(model=model, journal=journal),
                        embed_model_id=BigramEmbedder.model_identity)
    index.open()
    journal.bind(index)
    return index
'''

_CHILD_MAIN = r'''
import sys, time
hub_dir, index_dir, log_path, go_path = sys.argv[1:5]
_fh = open(log_path, "a", buffering=1)
def rec(kind, extra=""):
    _fh.write(f"{kind}\t{extra}\n"); _fh.flush(); os.fsync(_fh.fileno())

stub = Stub(rec)
if os.environ.get("AA_DOWN_FOREVER"):
    stub.fail_adj = -1
from bladex_proxy import adjudicator as _adj
_clock = [0.0]                      # 假时钟：等待上限瞬间到达，不真睡
_adj._monotonic = lambda: _clock[0]
_adj._sleep = lambda s: _clock.__setitem__(0, _clock[0] + s)
from bladex_proxy.storage.memory_index import MemoryIndex
from bladex_proxy.storage.memory_hub import MemoryHub
_add_sleep = float(os.environ.get("AA_ADD_SLEEP", "0"))
_orig_add, _orig_clear = MemoryIndex.add_fact, MemoryIndex.clear
def _add(self, fact):
    rec("add_fact"); time.sleep(_add_sleep); return _orig_add(self, fact)
def _clear(self):
    rec("clear"); return _orig_clear(self)
MemoryIndex.add_fact, MemoryIndex.clear = _add, _clear
_orig_put = _adj.LLMAdjudicator._cache_put
def _put(self, items, verdicts):
    _orig_put(self, items, verdicts); rec("adj_cached", str(len(items)))
_adj.LLMAdjudicator._cache_put = _put
if os.environ.get("AA_BREAK_FTS"):
    from bladex_proxy.storage.fts_index import FtsIndex
    FtsIndex.upsert = lambda self, fact: None
index = build_index(index_dir, stub, os.environ.get("AA_MODEL", "stub/m"))
rec("opened")
while not os.path.exists(go_path):
    time.sleep(0.01)
hub = MemoryHub(hub_dir); hub.open()
n = index.rebuild_from_hub(hub, full=True, distill_concurrency=1)
print("RESULT " + json.dumps({"n": n, "abort": index.last_rebuild_abort_reason,
                              "adj": index._adjudicator.stats(), "calls": stub.calls}))
index.close(); hub.close()
'''

_REPO = Path(__file__).resolve().parents[3]


def _stubs() -> dict[str, Any]:
    ns: dict[str, Any] = {}
    exec(_STUBS_SRC, ns)  # noqa: S102 —— 本文件自带的桩源码（子进程同一份）
    return ns


# ── 夹具 ────────────────────────────────────────────────────────────────────

def _make_hub(path: Path, n: int = 8) -> Path:
    """n 轮：每轮两句——一句近重复（只差数字，进裁决包）、一句独有（无邻居）。

    assistant 正文 ≥ 40 字（结论通道非空）：否则第二次 full 会被库里遗留的
    `seen_discourse/` memo 整轮跳过（结算稿 §2 旁证，不在本批修）——那是另一个问题，
    不该混进本批"中断 / 续跑"的判据里。
    """
    from bladex_proxy.models import Identity, Turn, TurnStatus
    from bladex_proxy.storage.memory_hub import MemoryHub

    hub = MemoryHub(path)
    hub.open()
    for i in range(n):
        ident = Identity(user_id="u1", agent_id="codex", session_id="s-aa", turn_index=i)
        hub.put(ident.storage_key(f"17999000{i:02d}-0"), Turn(
            identity=ident, model="m", status=TurnStatus.OK,
            response_text=f"已记录第 {i} 阶段的进度与样本归档情况，后续按既定计划继续推进并在下一轮同步结果。",
            request_messages=[{"role": "user", "content":
                               f"BladeX 全量重建的当前进度是第 {i} 阶段，负责人是 Jason。"
                               f"第 {i} 轮另记一件独立的事：编号 Q{i * 37} 的样本已归档"}],
            ts=datetime(2026, 10, 22, 8, 0, i, tzinfo=UTC)))
    hub.close()
    return path


def _tree_hash(d: Path) -> str:
    """目录树内容哈希。只排除 RocksDB 的文本诊断日志（LOG / LOG.old.*）。"""
    h = hashlib.sha256()
    for p in sorted(x for x in d.rglob("*") if x.is_file()):
        if p.name.startswith("LOG"):
            continue
        h.update(str(p.relative_to(d)).encode())
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def _child_env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("BLADEX_")}
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_REPO / "packages" / "bladex-core"), str(_REPO / "packages" / "bladex-proxy"),
         *[p for p in sys.path if p]])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["BLADEX_ADJUDICATE_CHUNK"] = "1"     # 每条候选一块：中途 kill 有块可数
    env.update(extra)
    return env


def _log(path: Path) -> list[tuple[str, str]]:
    if not path.exists():
        return []
    return [tuple((ln.split("\t") + [""])[:2]) for ln in path.read_text().splitlines() if ln]


def _count(path: Path, kind: str) -> int:
    return sum(1 for k, _ in _log(path) if k == kind)


def _run_child(tmp: Path, hub: Path, idx: Path, *, tag: str,
               kill_when: Any = None, **env: str) -> dict[str, Any]:
    """跑一次子进程 full 重建。返回 {h0, h1, result, killed, log}。"""
    script = tmp / "aa_child.py"
    script.write_text(_STUBS_SRC + _CHILD_MAIN, encoding="utf-8")
    log, go = tmp / f"{tag}.log", tmp / f"{tag}.go"
    proc = subprocess.Popen(  # noqa: S603
        [sys.executable, str(script), str(hub), str(idx), str(log), str(go)],
        env=_child_env(**env), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    deadline = time.monotonic() + 120
    while _count(log, "opened") == 0:
        assert proc.poll() is None, proc.stdout.read()
        assert time.monotonic() < deadline, "child never opened the index"
        time.sleep(0.01)
    h0 = _tree_hash(idx)
    go.touch()
    killed = False
    if kill_when is not None:
        while proc.poll() is None:
            if kill_when(log):
                proc.send_signal(signal.SIGKILL)
                killed = True
                break
            assert time.monotonic() < deadline, "kill condition never met"
            time.sleep(0.005)
    out, _ = proc.communicate(timeout=120)
    result = None
    for ln in out.splitlines():
        if ln.startswith("RESULT "):
            result = json.loads(ln[len("RESULT "):])
    if not killed:
        assert proc.returncode == 0, out[-3000:]
    return {"h0": h0, "h1": _tree_hash(idx), "result": result, "killed": killed,
            "log": log, "out": out}


def _fact_contents(idx: Path) -> list[str]:
    from bladex_proxy.storage.memory_index import MemoryIndex

    ro = MemoryIndex(idx, embedder=None, read_only=True)
    ro.open()
    try:
        return sorted(f.content for f in ro.all_facts())
    finally:
        ro.close()


@pytest.fixture()
def hub(tmp_path: Path) -> Path:
    return _make_hub(tmp_path / "hub")


@pytest.fixture()
def built_prod(tmp_path: Path, hub: Path) -> Path:
    """一个已经完整重建过的"生产" Index（子进程跑完，像真实生产一样落过盘）。"""
    idx = tmp_path / "index"
    r = _run_child(tmp_path, hub, idx, tag="baseline")
    assert r["result"]["abort"] is None
    assert _fact_contents(idx), "基线应有 fact"
    for bak in idx.parent.glob(idx.name + ".bak-*"):   # 首次重建换下来的空库：夹具不需要它
        shutil.rmtree(bak)
    return idx


# ── 不留半截索引（MQ-I34）────────────────────────────────────────────────────

def test_kill9_during_write_leaves_prod_byte_identical(tmp_path: Path, hub: Path,
                                                       built_prod: Path) -> None:
    before = _fact_contents(built_prod)
    r = _run_child(tmp_path, hub, built_prod, tag="kill",
                   kill_when=lambda log: _count(log, "clear") >= 1
                   and _count(log, "add_fact") >= 3,
                   AA_ADD_SLEEP="0.05")
    assert r["killed"], "应在写入阶段被 SIGKILL"
    assert r["h1"] == r["h0"], "写入期 kill -9 后生产目录字节变了（半截索引）"
    assert _fact_contents(built_prod) == before
    assert (built_prod.parent / (built_prod.name + ".staging")).is_dir(), "暂存应留着待收割"


def test_self_check_failure_does_not_swap(tmp_path: Path, hub: Path,
                                          built_prod: Path) -> None:
    """FTS 写入被静默吞掉 ⇒ 三库对账不平 ⇒ 不换名、生产逐字节不变、非零信号。"""
    r = _run_child(tmp_path, hub, built_prod, tag="selfcheck", AA_BREAK_FTS="1")
    assert r["result"]["abort"] == "staging_self_check_failed"
    assert "index_rebuild_staging_self_check_failed" in r["out"]
    assert r["h1"] == r["h0"]
    assert not list(built_prod.parent.glob(built_prod.name + ".bak-*")), "自检失败不许换名"


def test_successful_full_rebuild_swaps_and_keeps_backup(tmp_path: Path, hub: Path,
                                                        built_prod: Path) -> None:
    before = _fact_contents(built_prod)
    r = _run_child(tmp_path, hub, built_prod, tag="swap")
    assert r["result"]["abort"] is None
    assert "index_full_rebuild_swapped" in r["out"]
    baks = list(built_prod.parent.glob(built_prod.name + ".bak-*"))
    assert len(baks) == 1 and _fact_contents(baks[0]) == before
    assert _fact_contents(built_prod) == before          # 同输入重建 ⇒ 同内容
    assert not (built_prod.parent / (built_prod.name + ".staging")).exists()


# ── 断网不作废（MQ-I35）──────────────────────────────────────────────────────

def test_wait_cap_exceeded_aborts_cleanly_without_blanket_add(
        tmp_path: Path, hub: Path, built_prod: Path) -> None:
    # 换个模型名：基线的蒸馏台账与判决缓存都按模型分代 ⇒ 这一跑必须真调上游
    r = _run_child(tmp_path, hub, built_prod, tag="cap", AA_DOWN_FOREVER="1",
                   AA_MODEL="stub/m2", BLADEX_ADJUDICATE_OUTAGE_MAX_WAIT_S="900")
    res = r["result"]
    assert res["abort"] == "adjudicate_upstream_unavailable"
    assert res["adj"]["degraded_items"] == 0, "断网期间不许产生任何 blanket ADD"
    assert res["adj"]["outage_wait_s"] >= 900
    assert "index_rebuild_adjudicate_upstream_unavailable" in r["out"]
    assert r["h1"] == r["h0"]
    assert not list(built_prod.parent.glob(built_prod.name + ".bak-*"))


def _inproc_rebuild(tmp: Path, hub: Path, name: str, stub: Any,
                    monkeypatch: pytest.MonkeyPatch) -> tuple[Any, list[str]]:
    from bladex_proxy import adjudicator as adj
    from bladex_proxy.storage.memory_hub import MemoryHub

    clock = [0.0]
    monkeypatch.setattr(adj, "_monotonic", lambda: clock[0])
    monkeypatch.setattr(adj, "_sleep", lambda s: clock.__setitem__(0, clock[0] + s))
    monkeypatch.setenv("BLADEX_ADJUDICATE_CHUNK", "1")
    ns = _stubs()
    index = ns["build_index"](tmp / name, stub)
    h = MemoryHub(hub)
    h.open()
    try:
        index.rebuild_from_hub(h, full=True, distill_concurrency=1)
    finally:
        h.close()
    contents = sorted(f.content for f in index.all_facts())
    return index, contents


def test_outage_then_recovery_resumes_same_chunk_with_zero_blanket_add(
        tmp_path: Path, hub: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from bladex_proxy import router_sdk

    ns = _stubs()
    clean = ns["Stub"]()
    monkeypatch.setattr(router_sdk, "completion", clean.completion)
    ref_index, ref = _inproc_rebuild(tmp_path, hub, "ref", clean, monkeypatch)
    ref_index.close()
    assert clean.calls["adjudicate"] >= 2, "夹具应产生多个裁决块"

    flaky = ns["Stub"]()
    flaky.fail_adj = 3                         # 连续 3 次 503，第 4 次恢复
    index, got = _inproc_rebuild(tmp_path, hub, "flaky", flaky, monkeypatch)
    st = index._adjudicator.stats()
    index.close()
    assert index.last_rebuild_abort_reason is None
    assert st["degraded_items"] == 0, "网络错误被当成了判决"
    assert st["outage_waits"] == 1 and flaky.calls["adjudicate_fail"] == 3
    # 恢复后重试的是**同一块**：第一个成功的裁决 prompt 与无故障那一跑的第一块相同
    assert flaky.adj_hashes == clean.adj_hashes
    assert got == ref


def _rebuild_with(tmp: Path, hub: Path, name: str, stub: Any, completion: Any,
                  monkeypatch: pytest.MonkeyPatch) -> Any:
    """用给定出口在 `tmp/name` 上跑一次 full（出口经 monkeypatch，teardown 自动复原）。"""
    from bladex_proxy import adjudicator as adj
    from bladex_proxy import router_sdk
    from bladex_proxy.storage.memory_hub import MemoryHub

    monkeypatch.setattr(router_sdk, "completion", completion)   # 先登记原值再让 build_index 改
    clock = [0.0]
    monkeypatch.setattr(adj, "_monotonic", lambda: clock[0])
    monkeypatch.setattr(adj, "_sleep", lambda s: clock.__setitem__(0, clock[0] + s))
    monkeypatch.setenv("BLADEX_ADJUDICATE_CHUNK", "1")
    idx = _stubs()["build_index"](tmp / name, stub)
    router_sdk.completion = completion
    h = MemoryHub(hub)
    h.open()
    try:
        idx.rebuild_from_hub(h, full=True, distill_concurrency=1)
    finally:
        h.close()
    return idx


def test_rate_limit_429_backs_off_like_outage(
        tmp_path: Path, hub: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """429（配额 / 限流）也是"上游暂时给不出答案"：退避、同一块继续、零降级。

    live 10-22 撞到的正是它（`RateLimitError … exceeded the 5-hour usage quota`）。
    """
    class _QuotaExceeded(Exception):
        """形同 litellm `RateLimitError`（status_code=429）。测试不直接 import 上游 SDK
        （`test_router_sdk::test_no_direct_vendor_import`）；判据本就只看 status_code。"""
        status_code = 429

    ns = _stubs()
    clean = ns["Stub"]()
    ref_idx = _rebuild_with(tmp_path, hub, "ref429", clean, clean.completion, monkeypatch)
    ref = sorted(f.content for f in ref_idx.all_facts())
    ref_idx.close()

    quota = ns["Stub"]()
    left = [3]

    def _completion(**kw: Any) -> Any:
        if "write-time adjudicator" in kw["messages"][0]["content"] and left[0]:
            left[0] -= 1
            raise _QuotaExceeded("RateLimitError: You have exceeded the 5-hour usage quota. "
                                 "It will reset at 19:00")
        return quota.completion(**kw)

    idx = _rebuild_with(tmp_path, hub, "quota", quota, _completion, monkeypatch)
    st = idx._adjudicator.stats()
    got = sorted(f.content for f in idx.all_facts())
    idx.close()
    assert left[0] == 0, "429 应被实际触发三次"
    assert idx.last_rebuild_abort_reason is None
    assert st["degraded_items"] == 0, "429 被当成了判决（降级 ADD）"
    assert st["outage_waits"] == 1
    assert quota.adj_hashes == clean.adj_hashes, "恢复后应从同一块继续"
    assert got == ref


def test_adjudication_cache_invalidated_by_prompt_version(
        tmp_path: Path, hub: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """同一输入：先落 adjcache/；`ADJ_VER` 一变 ⇒ 缓存 0 命中、裁决全部重调。"""
    from bladex_proxy import adjudicator as adj

    ns = _stubs()

    def _run(stub: Any) -> dict[str, Any]:
        idx = _rebuild_with(tmp_path, hub, "ver_index", stub, stub.completion, monkeypatch)
        st = idx._adjudicator.stats()
        idx.close()
        return st

    first = ns["Stub"]()
    _run(first)
    assert first.calls["adjudicate"] >= 2
    same = ns["Stub"]()
    st_same = _run(same)
    assert same.calls["adjudicate"] == 0 and st_same["cache_hits"] >= 2, "同版本应全部命中"
    monkeypatch.setattr(adj, "ADJ_VER", "v-aa2-test-bump")
    bumped = ns["Stub"]()
    st_bumped = _run(bumped)
    assert st_bumped["cache_hits"] == 0, "prompt 版本变了仍命中旧判决"
    assert bumped.calls["adjudicate"] == first.calls["adjudicate"]


def test_parse_failure_still_degrades_under_threshold(
        tmp_path: Path, hub: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """解析型失败不退避（不是上游的问题）：照旧降级 ADD，由比例阈值决定中止还是落盘。"""
    from bladex_proxy import router_sdk

    ns = _stubs()
    for ratio, expect_abort in (("0.99", None), ("0.01", "adjudicate_outage")):
        stub = ns["Stub"]()
        stub.garbage_marker = "第 3 阶段"
        monkeypatch.setattr(router_sdk, "completion", stub.completion)
        monkeypatch.setenv("BLADEX_ADJUDICATE_OUTAGE_ABORT_RATIO", ratio)
        # 批 AB · MQ-I39：默认会就地重试后放行（见 test_ab_i38_i39.py）；这里钉的是比例阈值本身
        monkeypatch.setenv("BLADEX_ADJUDICATE_RETRY_MAX", "0")
        index, _ = _inproc_rebuild(tmp_path, hub, f"parse-{ratio}", stub, monkeypatch)
        st = index._adjudicator.stats()
        index.close()
        assert st["degraded_items"] >= 1 and st["outage_waits"] == 0
        assert index.last_rebuild_abort_reason == expect_abort


# ── 可中断 + 可续跑 ──────────────────────────────────────────────────────────

def test_resume_after_kill_redoes_no_finished_llm_work(tmp_path: Path, hub: Path) -> None:
    """空库起跑 → 裁决第 2 块落缓存后 SIGKILL → 重跑：蒸馏 0 调用、已完成的裁决块 0 调用，
    最终 fact 内容集合 = 从同一起点不中断跑一次的结果。"""
    ref = _run_child(tmp_path, hub, tmp_path / "ref_index", tag="ref")
    ref_calls = ref["result"]["calls"]
    assert ref_calls["adjudicate"] >= 3, "夹具要有足够多的裁决块"

    idx = tmp_path / "resume_index"
    r1 = _run_child(tmp_path, hub, idx, tag="r1", AA_ADJ_SLEEP="0.2",
                    kill_when=lambda log: _count(log, "adj_cached") >= 2)
    assert r1["killed"]
    n_done = _count(r1["log"], "adj_cached")          # 落了缓存的块 = 已完成
    done = [h for k, h in _log(r1["log"]) if k == "adjudicate"][:n_done]
    assert 2 <= n_done < ref_calls["adjudicate"], "kill 应落在裁决阶段中途"

    r2 = _run_child(tmp_path, hub, idx, tag="r2")
    calls = r2["result"]["calls"]
    redone = {h for k, h in _log(r2["log"]) if k == "adjudicate"} & set(done)
    assert calls["distill"] == 0, "中断前已完成的蒸馏被重做"
    assert not redone, f"已完成的裁决块被重做：{redone}"
    assert calls["adjudicate"] == ref_calls["adjudicate"] - len(done)
    assert "index_rebuild_harvested" in r2["out"]
    assert _fact_contents(idx) == _fact_contents(tmp_path / "ref_index")


# ── A28：嵌入 IPC 分块 + 可配超时 ─────────────────────────────────────────────

class _StubEmbedServer:
    """TCP 桩：记录每次请求的条数；`delay_s` 模拟慢模型。"""

    def __init__(self, delay_s: float = 0.0) -> None:
        import asyncio

        self.sizes: list[int] = []
        self.delay_s = delay_s
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self.port = 0
        threading.Thread(target=self._run, daemon=True).start()
        self._ready.wait(10)

    def _run(self) -> None:
        import asyncio

        from bladex_proxy.embed_server import STREAM_LIMIT

        async def handle(reader: Any, writer: Any) -> None:
            line = await reader.readline()
            req = json.loads(line)
            self.sizes.append(len(req["texts"]))
            if self.delay_s:
                await asyncio.sleep(self.delay_s)
            writer.write((json.dumps({"vectors": [[1.0] for _ in req["texts"]],
                                      "model_identity": "test:stub"}) + "\n").encode())
            await writer.drain()
            writer.close()

        async def main() -> None:
            srv = await asyncio.start_server(handle, "127.0.0.1", 0, limit=STREAM_LIMIT)
            self.port = srv.sockets[0].getsockname()[1]
            self._ready.set()
            async with srv:
                await srv.serve_forever()

        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(main())


def test_ipc_rebuild_scale_batch_is_chunked(monkeypatch: pytest.MonkeyPatch) -> None:
    """T1 量级（39,069 段）一次 embed 调用：单次请求 ≤ 块上限，全部走 IPC、不回落。"""
    from bladex_proxy.embed_client import IPCEmbedAdapter

    monkeypatch.setenv("BLADEX_EMBED_IPC_CHUNK", "500")
    srv = _StubEmbedServer()
    ad = IPCEmbedAdapter("tcp", f"127.0.0.1:{srv.port}", local_factory=None)
    t0 = time.monotonic()
    vecs = ad.embed_passage([f"段落 {i}" for i in range(39_069)])
    assert len(vecs) == 39_069 and not ad.degraded
    assert max(srv.sizes) <= 500 and sum(srv.sizes) == 39_069
    assert len(srv.sizes) == -(-39_069 // 500)
    assert time.monotonic() - t0 < 60


def test_ipc_timeout_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    from bladex_proxy.embed_client import IPCEmbedAdapter

    class _Local:
        def embed_passage(self, texts: list[str]) -> list[list[float]]:
            return [[0.0] for _ in texts]

    srv = _StubEmbedServer(delay_s=0.3)
    monkeypatch.setenv("BLADEX_EMBED_IPC_TIMEOUT_S", "0.05")
    slow = IPCEmbedAdapter("tcp", f"127.0.0.1:{srv.port}", local_factory=_Local)
    assert slow._timeout_s == 0.05
    slow.embed_passage(["a"])
    assert slow.degraded, "0.05 s 超时应触发回落"
    monkeypatch.setenv("BLADEX_EMBED_IPC_TIMEOUT_S", "5")
    ok = IPCEmbedAdapter("tcp", f"127.0.0.1:{srv.port}", local_factory=_Local)
    assert ok.embed_passage(["a", "b"]) == [[1.0], [1.0]] and not ok.degraded


# ── 默认值单一真相源 ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("name,default", [
    ("BLADEX_ADJUDICATE_OUTAGE_MAX_WAIT_S", 21600.0),
    ("BLADEX_EMBED_IPC_CHUNK", 64.0),
    ("BLADEX_EMBED_IPC_TIMEOUT_S", 120.0),
])
def test_new_defaults_live_in_flags_and_env_example(name: str, default: float) -> None:
    import re

    from bladex_core.flags import MEMORY_NUMERIC_DEFAULTS

    assert MEMORY_NUMERIC_DEFAULTS[name] == default
    text = (_REPO / "config" / ".env.example").read_text(encoding="utf-8")
    m = re.search(rf'^\s*export\s+{name}="?([^"\n]*)"?\s*$', text, re.MULTILINE)
    assert m and float(m.group(1)) == default


def test_harvest_prefixes_match_index_ledger_prefixes() -> None:
    from bladex_proxy.storage import index_staging, memory_index

    assert index_staging.HARVEST_PREFIXES == (
        memory_index._DISTILL_LEDGER_PREFIX.encode(), memory_index._ADJ_CACHE_PREFIX.encode(),
        memory_index._EMBED_CACHE_PREFIX.encode())


# ── 原子交换与残留处置（单元）────────────────────────────────────────────────

def test_exchange_swaps_two_directories(tmp_path: Path) -> None:
    from bladex_proxy.storage.index_staging import exchange

    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "f").write_text("A")
    (b / "f").write_text("B")
    exchange(a, b)
    assert (a / "f").read_text() == "B" and (b / "f").read_text() == "A"


def test_exchange_refuses_cross_device(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from bladex_proxy.storage import index_staging

    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    real_stat = os.stat
    monkeypatch.setattr(index_staging.os, "stat", lambda p: (
        os.stat_result((*real_stat(p)[:2], 99, *real_stat(p)[3:]))
        if Path(p) == b else real_stat(p)))
    with pytest.raises(index_staging.AtomicSwapError, match="cross-device"):
        index_staging.exchange(a, b)


def test_leftover_without_marker_is_kept_as_backup(tmp_path: Path) -> None:
    """换名后、改 bak 前崩溃留下的旧生产：没有暂存标记 ⇒ 改名 bak，绝不删。"""
    from bladex_proxy.storage.index_staging import retire_leftovers, staging_path

    prod = tmp_path / "index"
    prod.mkdir()
    st = staging_path(prod)
    st.mkdir()
    (st / "data").write_text("old production")
    assert retire_leftovers(prod) == []
    baks = list(tmp_path.glob("index.bak-*"))
    assert len(baks) == 1 and (baks[0] / "data").read_text() == "old production"


# ── W22：LanceDB 新 API ──────────────────────────────────────────────────────

def test_ann_index_uses_new_lancedb_api_without_deprecation(tmp_path: Path) -> None:
    import lancedb
    import pyarrow as pa
    from bladex_proxy.storage.memory_index import MemoryIndex

    rows = 300
    t = lancedb.connect(str(tmp_path / "lancedb")).create_table("facts", data=pa.table({
        "id": [f"f{i}" for i in range(rows)],
        "vector": [[float((i * 7 + j) % 13) for j in range(16)] for i in range(rows)]}))
    idx = MemoryIndex(tmp_path, embedder=None)
    idx._get_lance_table = lambda name: t  # type: ignore[method-assign]
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert idx.ensure_ann_index("facts") is True
    info = t.list_indices()[0]
    assert info.index_type == "IvfFlat"
    assert info.index_details.get("metric_type") == "COSINE"


def test_no_deprecated_create_index_keyword_form_in_source() -> None:
    import re

    src = "\n".join(p.read_text(encoding="utf-8")
                    for p in (_REPO / "packages").glob("*/bladex_*/**/*.py"))
    assert not re.search(r"create_index\(\s*metric=", src)
