"""consolidator 空载退避（MQ-W17 修法①，批 X 2026-09-22）。

这组测试盯的不是"退避会不会省 CPU"——那要 live 读数（结算稿判据①）。
它盯的是**退避不会把消费延迟弄坏**（判据②红线），以及两条回落路径各有一把尺子
（判据④：不只证明"我写的那条路会红"，还要证明"我没写的那条路不会静默"）。

背景一句话：没有任何新 turn 的一轮，consolidator 仍跑满全套例行作业
（20,123 key 实测 ~74s/轮，周期 134s、占空比 54.1%）。成本 83% 在库卫生作业、
仅 8% 在全库扫描（MQ-W19 订正；最初归给"全量扫描"是错的）。
🔴 该缺陷自 v0.1.0 起一直存在，不是某一版的回归。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from bladex_proxy.consolidator import (
    IdleBackoff,
    _HubWatermark,
    resolve_idle_backoff,
    sleep_until_work,
)

_REPO = Path(__file__).resolve().parents[3]


def _backoff(**kw) -> IdleBackoff:
    opts = {"enabled": True, "after": 3, "max_s": 600.0}
    opts.update(kw)
    return IdleBackoff(60.0, **opts)


# ── 1. 状态机本体 ───────────────────────────────────────────────────────────


def test_idle_rounds_before_after_threshold_stay_at_base():
    """`after=3` ⇒ 前三轮空载仍按基准间隔，短暂静默不受影响。"""
    b = _backoff()
    assert [b.observe(0) for _ in range(3)] == [60.0, 60.0, 60.0]


def test_backoff_doubles_and_caps():
    b = _backoff()
    for _ in range(3):
        b.observe(0)
    assert b.observe(0) == 120.0
    assert b.observe(0) == 240.0
    assert b.observe(0) == 480.0
    assert b.observe(0) == 600.0   # 封顶
    assert b.observe(0) == 600.0   # 封顶之后不再涨


def test_candidates_reset_immediately_from_the_cap():
    """🔴 判据④(b) 变异点之一：`observe(candidates>0)` 里的 `reset()`。

    拆掉它 ⇒ 有候选也留在 600s ⇒ 下一轮消费最坏晚 10 分钟（判据② 失败）。
    """
    b = _backoff()
    for _ in range(9):
        b.observe(0)
    assert b.interval == 600.0
    assert b.observe(1) == 60.0
    assert b.idle_rounds == 0


def test_disabled_is_a_literal_no_op():
    """判据④(a) 的单元版：关掉 ⇒ 间隔恒等于 `--interval`，逐字旧行为。"""
    b = _backoff(enabled=False)
    assert [b.observe(0) for _ in range(12)] == [60.0] * 12
    assert b.interval == 60.0


def test_max_below_base_never_shortens_the_interval():
    """配置写反（MAX_S < interval）不该让它睡得比基准还短。"""
    b = _backoff(max_s=10.0)
    for _ in range(9):
        b.observe(0)
    assert b.interval == 60.0


def test_after_zero_backs_off_from_the_first_idle_round():
    b = _backoff(after=0)
    assert b.observe(0) == 120.0


# ── 2. 旋钮取值：唯一真相源 ─────────────────────────────────────────────────


def test_resolve_reads_the_flag_table(monkeypatch):
    for name in ("BLADEX_CONSOLIDATOR_IDLE_BACKOFF",
                 "BLADEX_CONSOLIDATOR_IDLE_AFTER",
                 "BLADEX_CONSOLIDATOR_IDLE_MAX_S"):
        monkeypatch.delenv(name, raising=False)
    b = resolve_idle_backoff(60.0)
    assert (b.enabled, b.after, b.max_s, b.base_s) == (True, 3, 1800.0, 60.0)


def test_resolve_honours_env(monkeypatch):
    monkeypatch.setenv("BLADEX_CONSOLIDATOR_IDLE_BACKOFF", "0")
    monkeypatch.setenv("BLADEX_CONSOLIDATOR_IDLE_AFTER", "1")
    monkeypatch.setenv("BLADEX_CONSOLIDATOR_IDLE_MAX_S", "30")
    b = resolve_idle_backoff(60.0)
    assert (b.enabled, b.after, b.max_s) == (False, 1, 60.0)


def test_no_second_default_anywhere():
    """🔴 刚性原则 12 第二条判据：同一参数在两条调用路径上各有一个默认值 = 缺陷。

    `--concurrency` 就是那个反面教材（argparse 0 / sync 4 / 就地 environ 1，
    代价是积压 720 轮不降）。这三个旋钮**只许**从 flags 取一次：
    既不加 argparse 参数，也不在任何地方就地 `os.environ.get`。
    """
    src = (_REPO / "packages/bladex-proxy/bladex_proxy/consolidator.py").read_text("utf-8")
    cli = (_REPO / "packages/bladex-proxy/bladex_proxy/cli/lifecycle.py").read_text("utf-8")
    for knob in ("BLADEX_CONSOLIDATOR_IDLE_BACKOFF",
                 "BLADEX_CONSOLIDATOR_IDLE_AFTER",
                 "BLADEX_CONSOLIDATOR_IDLE_MAX_S"):
        flag = knob.replace("BLADEX_CONSOLIDATOR_", "").lower()
        assert f'add_argument("--idle-{flag.replace("_", "-")}"' not in src
        assert f'environ.get("{knob}"' not in src
        assert f'environ.get("{knob}"' not in cli
    # 取值只能经 flags（`resolve_idle_backoff` 里那一处 import）
    assert src.count('flag_enabled("BLADEX_CONSOLIDATOR_IDLE_BACKOFF")') == 1
    assert src.count('flag_number("BLADEX_CONSOLIDATOR_IDLE_AFTER")') == 1
    assert src.count('flag_number("BLADEX_CONSOLIDATOR_IDLE_MAX_S")') == 1


# ── 3. 流量探针 ─────────────────────────────────────────────────────────────


class _FakeHub:
    """MemoryHub 替身：只需要 `seq()`（RocksDB WAL 序列号）。"""

    def __init__(self, seq: int = 100) -> None:
        self.cur = seq
        self.calls = 0

    def seq(self) -> int:
        self.calls += 1
        if self.cur is None:
            raise RuntimeError("db not open")
        return self.cur


def test_probe_sees_new_entries_only_after_the_watermark_moves():
    r = _FakeHub(100)
    probe = _HubWatermark(r)
    assert probe.has_new() is False
    r.cur = 101
    assert probe.has_new() is True
    probe.mark()
    assert probe.has_new() is False


def test_probe_swallows_hub_errors():
    """Hub 读不到序列号 ⇒ 按"没有新东西"算（退避照常，只是不提前唤醒）。"""
    r = _FakeHub(None)
    probe = _HubWatermark(r)
    assert probe.has_new() is False


def test_probe_watches_the_hub_not_the_p0_stream():
    """🔴 唤醒信号必须挂 Memory Hub，不能挂 P0（Redis stream）。

    2026-09-23 live 实测：P0 水位比 Hub 落库**早** —— 唤醒 01:54:54 跑在
    `hub.put` 01:54:57 前面，被唤醒的那一轮 `candidates=0`，`candidates=1`
    实际到 01:57:19（注入后 **146 s**），**比不退避的 56 s 还差**，正踩判据②红线。
    ⇒ 探针只许读 `hub.seq()`，不许回到 `xinfo_stream` / `last-generated-id`。
    """
    src = (_REPO / "packages/bladex-proxy/bladex_proxy/consolidator.py").read_text("utf-8")
    assert "xinfo_stream(" not in src        # 不许回到 P0 stream 水位
    assert "last-generated-id" not in src
    assert "traffic = _HubWatermark(ledger)" in src


def test_probe_never_regresses_on_a_lower_seq():
    """序列号只认**变大**：secondary 重开后 seq 可能回落，不该当成新流量。"""
    r = _FakeHub(100)
    probe = _HubWatermark(r)
    r.cur = 50
    assert probe.has_new() is False


# ── 4. 🔴 判据② 红线：退到上限后来了新流量 ─────────────────────────────────


def _fake_clock():
    t = {"now": 0.0}

    def now() -> float:
        return t["now"]

    def sleep(s: float) -> None:
        t["now"] += max(s, 0.001)

    return now, sleep, t


def test_wake_falls_back_to_base_interval():
    """🔴 判据④(b) 变异点之二：`sleep_until_work` 里的 `idle.reset()`。

    构造 = 退到 600s 上限、sleep 刚开始 5 秒就来了一轮真流量。
    要求两件事同时成立：① sleep 立刻结束（≤ 一个 1s 轮询节拍）；
    ② **下一轮的间隔回到基准 60s**。

    拆掉那一行 ⇒ ① 仍然成立（看上去"响应了新流量"），② 静默失败：
    下一次还睡 600s，消费延迟劣化十倍而日志上看不出来。
    这就是为什么这条断言必须钉 `interval`，不能只钉"提前返回"。
    """
    now, sleep, t = _fake_clock()
    b = _backoff()
    for _ in range(9):
        b.observe(0)
    assert b.interval == 600.0

    r = _FakeHub(100)
    probe = _HubWatermark(r)

    def poll_cmd() -> None:
        if now() >= 5.0:
            r.cur = 101      # 第 5 秒：一轮真流量落进 P0
        return None

    out = sleep_until_work(b.interval, poll_cmd=poll_cmd, traffic=probe,
                           idle=b, sleep=sleep, now=now)
    assert out is None
    assert t["now"] <= 7.0, "来了新流量却没在一个轮询节拍内醒来"
    assert b.interval == 60.0, "醒是醒了，但间隔没回落 —— 下一轮又要睡满退避上限"
    assert b.idle_rounds == 0


def test_sleep_without_traffic_sleeps_the_whole_interval():
    """没有新流量就老老实实睡满（否则退避等于没退）。"""
    now, sleep, t = _fake_clock()
    b = _backoff()
    probe = _HubWatermark(_FakeHub(100))
    assert sleep_until_work(600.0, traffic=probe, idle=b,
                            sleep=sleep, now=now) is None
    assert t["now"] >= 600.0
    assert b.interval == 60.0


def test_backoff_disabled_also_disables_the_wake():
    """🔴 判据④(a)：关掉开关 = **逐字**回到旧行为，唤醒这条新路也得一起关。

    只挡住退避、留着唤醒，等于"关了开关还是有新行为"——阴性对照就失去意义
    （本仓反复踩的假回滚形状：开关只挡了一半）。
    """
    now, sleep, t = _fake_clock()
    b = _backoff(enabled=False)
    r = _FakeHub(100)
    probe = _HubWatermark(r)

    def poll_cmd() -> None:
        r.cur = 101
        return None

    assert sleep_until_work(60.0, poll_cmd=poll_cmd, traffic=probe, idle=b,
                            sleep=sleep, now=now) is None
    assert t["now"] >= 60.0, "开关关着却被新流量提前唤醒 —— 这不是逐字旧行为"


def test_long_sleep_keeps_the_heartbeat_alive():
    """🔴 退避期间必须续跳，否则 `bladex status` 把活着的进程显示成"没心跳"。

    心跳 TTL = 300s（`sync_control._HEARTBEAT_TTL_S`），退避上限 1800s ⇒
    不续跳的话，一个 1800s 周期里有 1500s 读不到心跳。**省 CPU 不该以
    "看上去死了"为代价。** 拆掉 `_sleep_or_cmd` 的 `tick=` 这一路，本例必红。
    """
    from bladex_proxy.sync_control import _HEARTBEAT_TTL_S

    now, sleep, _ = _fake_clock()
    beats: list[float] = []
    assert sleep_until_work(1800.0, idle=_backoff(), tick=beats.append,
                            sleep=sleep, now=now) is None
    assert len(beats) >= 2
    gaps = [a - b for a, b in zip(beats, beats[1:], strict=False)]
    assert max(gaps) < _HEARTBEAT_TTL_S, (
        f"两次心跳最大间隔 {max(gaps)}s ≥ TTL {_HEARTBEAT_TTL_S}s —— 心跳会过期")


def test_sleep_tick_is_wired_from_the_loop():
    """接线守卫：`_sleep_or_cmd` 真的把 `tick=` 传下去（MQ-W13 同款）。"""
    src = (_REPO / "packages/bladex-proxy/bladex_proxy/consolidator.py").read_text("utf-8")
    assert "tick=_beat_while_sleeping," in src
    assert 'control.beat(state="idle", phase="sleeping", batch=batch_no,' in src


def test_manual_sync_command_still_wins():
    """老行为不变：控制通道有任务 ⇒ 立即返回该任务（比流量探针优先）。"""
    now, sleep, _ = _fake_clock()
    b = _backoff()
    job = {"job_id": "j1"}
    out = sleep_until_work(600.0, poll_cmd=lambda: job,
                           traffic=_HubWatermark(_FakeHub(100)),
                           idle=b, sleep=sleep, now=now)
    assert out is job


def test_probe_absent_degrades_to_plain_sleep():
    """Redis 建不起来（探针为 None）⇒ 退避照常，只是不提前醒。"""
    now, sleep, t = _fake_clock()
    b = _backoff()
    assert sleep_until_work(300.0, idle=b, sleep=sleep, now=now) is None
    assert t["now"] >= 300.0


# ── 5. 接线守卫（MQ-W13：「函数有测试」≠「接线有守卫」）────────────────────


def test_loop_feeds_candidates_not_new_facts():
    """空载信号必须取 `last_rebuild_candidates`。

    用 `new_facts == 0` 会把"扫到候选但判重/裁决全丢"的轮次误判成空载 ⇒
    真有流量时反而退避（静默失效方向）。
    """
    src = (_REPO / "packages/bladex-proxy/bladex_proxy/consolidator.py").read_text("utf-8")
    assert "_candidates = index.last_rebuild_candidates" in src
    assert "idle.observe(_candidates)" in src
    assert "idle.observe(new_facts)" not in src


def test_candidates_readout_is_read_only_in_memory_index():
    """🔴 判据③：`candidates > 0` 那一轮的行为逐字不变。

    机制解不是"跑一遍对比日志"，而是让读数**结构上不可能**参与判定：
    `_last_rebuild_candidates` 在 memory_index 里只许出现在
    初始化 / 每轮清零 / 扫描点赋值 三处，不许进任何比较或分支。
    """
    src = (_REPO / "packages/bladex-proxy/bladex_proxy/storage/memory_index.py"
           ).read_text("utf-8")
    uses = [ln.strip() for ln in src.splitlines() if "_last_rebuild_candidates" in ln]
    assert uses == [
        "self._last_rebuild_candidates = 0",
        "return self._last_rebuild_candidates",
        "self._last_rebuild_candidates = 0   # MQ-W17 读数，每轮先清零",
        "self._last_rebuild_candidates = len(_scan_batch)",
    ], f"读数被用在了赋值以外的地方：{uses}"


def test_watermark_is_marked_before_the_scan():
    """`traffic.mark()` 必须排在 `ledger.catch_up()` 之前。

    扫描要跑几十秒；先扫后 mark 会把扫描期间到达的 turn 一并吃掉，
    它们要多等一个完整的退避周期才被看见（判据② 的静默失效方向）。
    """
    src = (_REPO / "packages/bladex-proxy/bladex_proxy/consolidator.py").read_text("utf-8")
    assert src.index("traffic.mark()") < src.index("ledger.catch_up()  #")


def test_main_loop_sleeps_the_backed_off_interval():
    """🔴 **让这件事真的生效的唯一一行**：主循环必须睡 `idle.interval`，不是 `args.interval`。

    复核方 2026-09-23 的反向变异抓到的洞：把 `_sleep_or_cmd(idle.interval)` 换成
    `_sleep_or_cmd(args.interval)`，**28 条用例全绿** —— 退避状态机、唤醒、回落
    全都还在，只是**和主循环断开了**，功能等于没做。
    这是 MQ-W13「函数有测试 ≠ 接线有守卫」最纯粹的一例：
    机制的每个零件都有尺子，唯独"零件装上去了没有"没有。
    """
    src = (_REPO / "packages/bladex-proxy/bladex_proxy/consolidator.py").read_text("utf-8")
    assert "cmd = _sleep_or_cmd(idle.interval)" in src
    assert "_sleep_or_cmd(args.interval)" not in src


def test_manual_sync_resets_the_backoff():
    """手工 sync 跑完要回落基准（`_run_manual` 的 `finally: idle.reset()`）。

    拆掉它也是 28 全绿（复核方同批变异）。影响小于上一条——Hub 水位唤醒会兜住——
    但"手工消化完积压、紧接着来的流量撞在退避上"是个真实的形状，值得钉住。
    """
    src = (_REPO / "packages/bladex-proxy/bladex_proxy/consolidator.py").read_text("utf-8")
    manual = src[src.index("def _run_manual("):src.index("def _sleep_or_cmd(")]
    assert "finally:" in manual and "idle.reset()" in manual


def test_startup_line_prints_the_effective_knobs():
    """判据⑤：三个旋钮的**生效值**进启动行——live 上一眼可证接线到位。"""
    src = (_REPO / "packages/bladex-proxy/bladex_proxy/consolidator.py").read_text("utf-8")
    m = re.search(r'logger\.info\("index_consolidator_starting".*?\)\n\n',
                  src, re.DOTALL)
    assert m, "找不到 index_consolidator_starting 启动行"
    for field in ("idle_backoff=idle.enabled", "idle_after=idle.after",
                  "idle_max_s=idle.max_s"):
        assert field in m.group(0), f"启动行缺 {field}"


def test_both_launch_paths_reach_the_same_entry():
    """两条拉起路径（`bladex start` / `run_consolidator.sh`）同一个进程入口。

    旋钮走 env（`config/.env` → 进程环境 → flags），所以"接线到位"等价于
    "两条路都落在 `python -m bladex_proxy.consolidator` 上"——只要入口唯一，
    就不存在 `--concurrency` 那种"一条路传了、另一条没传"的缺口。
    """
    sh = (_REPO / "config/run_consolidator.sh").read_text("utf-8")
    assert "bladex consolidator" in sh and "bladex_proxy.cli consolidator" in sh
    life = (_REPO / "packages/bladex-proxy/bladex_proxy/cli/lifecycle.py").read_text("utf-8")
    assert '"-m", "bladex_proxy.consolidator"' in life


@pytest.mark.parametrize("knob", ["BLADEX_CONSOLIDATOR_IDLE_BACKOFF",
                                  "BLADEX_CONSOLIDATOR_IDLE_AFTER",
                                  "BLADEX_CONSOLIDATOR_IDLE_MAX_S"])
def test_knob_declared_in_env_example(knob):
    text = (_REPO / "config/.env.example").read_text("utf-8")
    assert re.search(rf"^export {knob}=", text, re.MULTILINE)
