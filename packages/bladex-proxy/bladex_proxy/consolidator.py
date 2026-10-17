"""Memory Index consolidation worker（ADR-0009 §7；ADR-0027 §4.2 起为包内入口）。

为什么独立进程：consolidation 是重操作（扫 RocksDB + embedding 推理），跑在 proxy
进程里会竞争 GIL 与磁盘 IO，影响热路径延迟；独立进程完全隔离，proxy 崩溃不影响
已积累的 Memory Index 数据。

为什么在包里（ADR-0027 §4.2）：此前唯一入口是仓库脚本 `scripts/run_index_consolidator.py`，
**不在 wheel 里**——`pip install` / `uv tool install` 的用户永远启动不了 Memory Index 提炼，
拿到的是一个"永远不产生任何记忆的记忆产品"。现在逻辑在包内，脚本转薄壳。

用法：
    bladex-consolidator                      # console script
    python -m bladex_proxy.consolidator      # 等价
    python scripts/run_index_consolidator.py    # 薄壳（仓库内保持兼容）

    # 定向重建：只重放某窗口/某 agent，不清库、不动其余 Memory Index 数据。
    # 两步——先反消费（删 consumed 标记），再跑增量重放：
    bladex-consolidator --unconsume --since 2026-07-13 --until 2026-07-24
    bladex-consolidator --once --since 2026-07-13 --until 2026-07-24

环境变量（同 proxy）：
    BLADEX_INDEX_PATH         Memory Index 存储路径（默认 data/bladex_index）
    BLADEX_ROCKSDB_PATH    Memory Hub 存储路径（默认 data/bladex_hub）
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import TYPE_CHECKING, Any

import structlog

from bladex_proxy import deployment
from bladex_proxy.storage.startup import open_hub_with_startup_window

if TYPE_CHECKING:
    from collections.abc import Callable

    from bladex_proxy.storage.memory_index import MemoryIndex

#: `_load_env_file()` 跑在 `_configure_logging()` 之前（它决定日志读什么配置），
#: 那时 structlog 还没配好。冲突先攒在这里，日志就绪后补一条结构化记录。
_pending_env_conflicts: list[deployment.EnvConflict] = []


# ── 存储锁冲突：把 rocksdict 的原始 IO error 翻成一句能照做的话 ────────────────
#
# 为什么值得单独处理（2026-08-05 实测）：`bladex-consolidator --unconsume` 在常驻
# consolidator 还活着时，甩出来的是
#     Exception: IO error: While lock file: data/bladex_index/meta_rocksdb/LOCK:
#     Resource temporarily unavailable
# 加一整段 traceback。这不是 bug——Memory Index 单写者是设计（ADR-0009 §7）——但用户看到的
# 是一串没有动作指向的堆栈，得读源码才知道"先停另一个进程"。发布形态上，凡是
# **预期内**的失败都必须给出下一步该敲什么。
_LOCK_MARKERS = (
    "lock file",                        # rocksdb 自己的措辞
    "resource temporarily unavailable",  # macOS / Linux flock errno
    "no locks available",                # 某些文件系统
)


def _lock_conflict_hint(exc: BaseException, store: str) -> str | None:
    """是存储锁冲突就返回可照做的提示，否则 None（调用方原样抛，别吞真错误）。

    `store` 取 "Memory Index" / "Memory Hub"——两者的占用方与解法不同：Memory Index 被另一个 consolidator 占，
    Memory Hub read-write 被 proxy 占（增量模式走 secondary 读，根本不需要锁）。
    """
    msg = str(exc).lower()
    if not any(m in msg for m in _LOCK_MARKERS):
        return None
    if store == "Memory Hub":
        return ("--full opens Memory Hub read-write, but the running proxy holds that lock. "
                "Stop the proxy first (`bladex stop`), run the full rebuild, then start it "
                "again. Incremental mode (without --full) reads Memory Hub as a secondary and needs "
                "no lock.")
    return ("another consolidator already holds the Memory Index write lock (single writer by design). "
            "Stop it first -- `bash config/run_consolidator.sh stop`, or `bladex stop` for the "
            "whole stack -- then re-run this command. `bladex status` shows what is running.")


def _finish_full_rebuild(index: MemoryIndex, *, reason: str | None = None) -> int:
    """按 rebuild 的明确中止信号收尾 full 模式，返回进程退出码。

    MQ-I33：`rebuild_from_hub` 的守卫是“降级不抛”的，空库 full rebuild 也合法地
    返回 0 条，所以这里只看 `last_rebuild_abort_reason`，不看事实数量。
    `reason` 只给 main 捕获到的异常路径显式覆盖用；正常路径必须来自 index。
    """
    if reason is None:
        reason = index.last_rebuild_abort_reason
    if reason is not None:
        logger.warning("index_consolidator_full_rebuild_aborted", reason=reason)
        return 1
    logger.info("index_consolidator_full_rebuild_done")
    return 0


def _load_env_file() -> None:
    """加载 config/.env 并把配置冲突报进日志（实现在 deployment.py，全仓库唯一）。

    2026-08-06 前这里是第三份各自为政的实现，而且是最弱的一份：**只认 cwd，
    连 `BLADEX_HOME` 都不认**，全局安装后 consolidator 永远读不到 `.env`。
    加上"不覆盖已设环境变量"的旧语义，当天真实事故 = 用户改了上游 base URL 与
    API key、重启 consolidator，进程仍读终端里的残留 export → 大量
    `distill_failed`（认证失败），提炼整段停摆而配置文件看上去完全正确。

    找不到部署根就跳过——env 也可能由 compose/systemd 直接注入。
    """
    root = deployment.find_root()
    if root is None:
        return
    conflicts = deployment.load_env_file(root)
    if not conflicts:
        return
    env_path = os.path.join(root, deployment.CONFIG_RELPATH)
    # 日志在 _configure_logging() 之前还没配好，用 print 到 stderr 保证看得见；
    # 之后再补一条结构化的（`config_env_shadowed` 是可 grep 的运营信号）。
    for line in deployment.format_conflicts(conflicts, env_path):
        print(line, file=sys.stderr)
    _pending_env_conflicts.extend(conflicts)


logger = structlog.get_logger()


def _configure_logging() -> None:
    """桥接 stdlib logging -> structlog 输出。

    distillation.py / consolidation_proxy.py 用 stdlib logging，不配置则它们的
    warning/error 不会出现在日志里。

    顺带接管上游模型 SDK 的日志（2026-08-05）：那一层此前在本进程里**完全没人管**——
    全局开关只写在 `route.py` 模块级，而 consolidator 根本不 import 它。实测一次运行
    的日志里 767 行是供应商自己的自言自语（含 100 行重复 + 指向别人 issue 页的求助链接）。
    `install_log_bridge()` 摘掉它自带的 handler、把署名换成 Router，INFO 级保留
    （"这轮发给了哪个模型"有运维价值，只是不该署别人的名）。
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] %(name)-28s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    from bladex_proxy import router_sdk

    router_sdk.install_log_bridge()


def _chdir_to_deployment_root() -> None:
    """切到部署根，让 `data/bladex_index`、`data/bladex_hub` 这些相对路径有唯一落点。

    consolidator 之前假设"总是从部署目录启动"（`os.getcwd()` 硬编码）。这个假设在
    `bladex start` 拉起它时成立，在用户手工 `bladex-consolidator` 时不成立——
    那时它会在别处新建一个空的 `data/`，看上去在跑、实际什么也不提炼。
    """
    root = deployment.find_root()
    if root is not None and root != os.getcwd():
        os.chdir(root)


def _log_env_conflicts() -> None:
    """日志就绪后补记配置冲突（`config_env_shadowed` 是可 grep 的运营信号）。"""
    for conflict in _pending_env_conflicts:
        logger.warning(
            "config_env_overridden" if conflict.applied else "config_env_shadowed",
            key=conflict.key,
            effective=deployment.mask(conflict.key, conflict.winner),
            ignored=deployment.mask(conflict.key, conflict.loser),
            source="config/.env" if conflict.applied else "environment",
        )
    _pending_env_conflicts.clear()


def resolve_concurrency(explicit: int) -> int:
    """蒸馏并发路数：显式 `--concurrency` 优先，否则读 flags（默认 4）。

    抽成纯函数是为了能被测试直接调——2026-08-22 之前这个回落**根本不存在**
    （`0 → None → 1`），而它埋在 `main()` 里，任何单测都够不着。
    取值依据与两次事故记录见 `bladex_core.flags` 的 `BLADEX_DISTILL_CONCURRENCY`。
    """
    if explicit >= 1:
        return explicit
    from bladex_core.flags import flag_number
    return max(1, int(flag_number("BLADEX_DISTILL_CONCURRENCY")))


# ── 空载退避（MQ-W17 修法①，批 X 2026-09-22）─────────────────────────────────
#
# 现象：`--interval 60` 的守护进程实测周期 **134s**，其中 ~74s 在干活、产出是
# 一行 `index_rebuild_scan_sorted candidates=0` ⇒ 占空比 54.1%、≈13h CPU/天
# （两把独立的尺子吻合：日志时间戳 74/134 ≈ 55%；`ps` 的 TIME/ELAPSED 两点差分
# 283/523 = 54.1%）。
#
# 🔴 成因（**2026-09-23 实测订正，见 MQ-W19**）：一轮"空载"里真正贵的**不是**扫描。
# 15 轮 × 2 个独立跑手拆时间线：全库 `scan_meta` 段只占 **5.6s（8%）**，
# **83% 在空闲轮的库卫生作业**——`maintain_projects` 那一段 **60s**：
# `link_matters_to_project` + `refresh_project_fields` 各遍历一遍全部 Matter、
# 每条 fact 边一次 `get_fact` 点读（`refresh` 里 `get_edges` 还调两次）
# ⇒ 每轮 3 趟全量边遍历，产出是 `projects=1`。其余卫生作业 6.7s（9%）。
# **早先把成因整个归给全量扫描是错的**（那是本批提示词与 MQ-W17 的原始判断，
# 由本批读数推翻）。
#
# 扫描本身确实也没用游标（`rebuild_from_hub` 进函数就 `scan_meta("")`，水位只用来
# **过滤**不用来**定位起点**；Hub key 按 `agent/session` 排序而水位是**时间**
# ⇒ 没有任何前缀能 seek 到水位——不是"忘了用游标"，是游标在这个键空间里用不上）。
# 但它只值 8% ⇒ **修法②③ 单独做几乎不解决本条**，排 0.4.0 前须按 MQ-W19 重估。
#
# 🔴 缺陷自 **v0.1.0** 起一直存在，**不是某一版引入的回归**：扫描点三个 tag 源码
# 逐字相同，`maintain_projects` 在空闲路径上的调用也在 v0.1.0 / v0.2.0 / HEAD 都在。
# 代价随记忆库增长而放大——上面两段循环都是"全部 Matter × 每条 fact 边"，
# 形状线性（**由代码结构 + 两个规模点判定：空库 7ms / 20,123 轮 72s；未做多点标定**）。
class IdleBackoff:
    """连续空载 ⇒ 间隔指数退避；一有候选立刻回落基准。

    状态**全在进程内**（不落盘、不进 Hub）——重启即回基准，故障面最小。

    🔴 红线是"消费延迟不劣化"，所以退避有两条独立的回落路径，缺一不可：
      ① `observe(candidates > 0)` —— 扫到候选，立刻回基准；
      ② `reset()` —— 退避期间 Memory Hub 水位动了（见 `_HubWatermark`），
         由 sleep 循环在 ≤1s 内打断并回基准。
    只有 ① 的话，退到上限（默认 1800s）时新 turn 最坏要等满那么久才被看见；
    只有 ② 的话，水位探针读不到时退避就永不回落。
    """

    def __init__(self, base_s: float, *, enabled: bool, after: int, max_s: float) -> None:
        self._base = max(0.0, float(base_s))
        self._enabled = bool(enabled)
        self._after = max(0, int(after))
        self._max = max(self._base, float(max_s))
        self._idle_rounds = 0
        self._interval = self._base

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def base_s(self) -> float:
        return self._base

    @property
    def after(self) -> int:
        return self._after

    @property
    def max_s(self) -> float:
        return self._max

    @property
    def interval(self) -> float:
        """下一次 sleep 该睡多久。关掉退避时恒等于 `--interval`（逐字旧行为）。"""
        return self._base if not self._enabled else self._interval

    @property
    def idle_rounds(self) -> int:
        return self._idle_rounds

    def observe(self, candidates: int) -> float:
        """吃一轮 rebuild 的候选数，返回下一次该睡多久。"""
        if candidates > 0:
            # 🔴 判据④(b) 盯的就是这一行：拆掉它 ⇒ 有流量也不回落 ⇒ 消费延迟劣化。
            self.reset()
            return self.interval
        self._idle_rounds += 1
        if self._enabled and self._idle_rounds > self._after:
            self._interval = min(self._max, max(self._base, self._interval) * 2)
        return self.interval

    def reset(self) -> None:
        """回到基准间隔（有候选 / 退避期间来了新流量 / 刚跑完手工 sync）。"""
        self._idle_rounds = 0
        self._interval = self._base


def resolve_idle_backoff(base_s: float) -> IdleBackoff:
    """三个旋钮的**唯一**取值点：全部读 `bladex_core.flags`。

    🔴 刻意**不加 argparse 参数**（刚性原则 12 第二条判据：同一参数在两条调用
    路径上各有一个默认值 = 缺陷）。`--concurrency` 就是那个反面教材——它有
    argparse 默认 `0`、`bladex sync` 默认 `4`、就地 `environ.get` 默认 `1`，
    三条路三个值，没人说得出生产在跑哪个，代价是积压 720 轮不降。
    这里只有一条路：`config/.env` → 进程 env → flags 表（与 `.env.example` 对账）。
    ⇒ `bladex start` 与 `run_consolidator.sh` 两条拉起路径同一个取值，接线不可能歪。
    """
    from bladex_core.flags import flag_enabled, flag_number
    return IdleBackoff(
        base_s,
        enabled=flag_enabled("BLADEX_CONSOLIDATOR_IDLE_BACKOFF"),
        after=int(flag_number("BLADEX_CONSOLIDATOR_IDLE_AFTER")),
        max_s=flag_number("BLADEX_CONSOLIDATOR_IDLE_MAX_S"),
    )


class _HubWatermark:
    """O(1) 回答"Memory Hub 里有没有我还没扫过的东西"——RocksDB 的 WAL 序列号。

    退避期间每秒问一次；序列号动了就结束 sleep 并把间隔打回基准。

    🔴 **为什么不是 P0（Redis stream）的水位——这一版是实测换掉的，不是设计出来的。**
    第一版读 P0 stream 的 XINFO 末条 id，在 20,123 key 的库上实测（2026-09-23）：
    注入 01:54:53 → 唤醒 **01:54:54（1 s，看着完美）** → 但 `hub.put` 直到 01:54:57 才落，
    于是被唤醒的那一轮扫描 `candidates=0`，要等下一个基准间隔才拾起，
    `candidates=1` 实际发生在 **01:57:19 = 注入后 146 s**，
    **比不退避的 56 s 还差** —— 正好踩在本批的红线（判据②）上。
    根因：P0 水位领先 Hub 落库，它回答的是"请求进来了"，不是"数据能被扫到了"。
    序列号回答的正是后者，没有这个时间差。
    🔴 五条单元用例 + 四处反向变异全绿也没拦住它 —— **这是只有 live 才照得出来的一类**。

    副作用是好的：不再依赖 Redis ⇒ 不需要 stream 名、不受 Redis 可用性影响、
    也没有"启动时 Redis 没起来探针就永久失效"那个坑。

    成本：`try_catch_up_with_primary() + latest_sequence_number()` 实测 **0.25 ms**，
    1800 s 退避里轮询 1800 次共 0.45 s CPU（占 0.025%），可忽略。
    任何异常都按"没有新东西"处理（退避照常，只是不提前唤醒）。
    """

    def __init__(self, hub: Any) -> None:  # noqa: ANN401 —— MemoryHub / 测试替身
        self._hub = hub
        self._seen = self._read()

    def _read(self) -> int:
        try:
            return int(self._hub.seq())
        except Exception:  # noqa: BLE001
            return 0

    def mark(self) -> None:
        """把当前水位记为"已知"。每轮 rebuild **开始前**调一次：
        扫描期间到达的 turn 于是留在水位之后，下一次 sleep 立刻感知得到。"""
        self._seen = self._read()

    def has_new(self) -> bool:
        cur = self._read()
        return cur > self._seen


def sleep_until_work(
    seconds: float,
    *,
    poll_cmd: Callable[[], dict | None] | None = None,
    traffic: _HubWatermark | None = None,
    idle: IdleBackoff | None = None,
    external: Any = None,
    external_round: Callable[[str], None] | None = None,
    tick: Callable[[float], None] | None = None,
    tick_every_s: float = 60.0,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.monotonic,
) -> dict | None:
    """可中断的轮间 sleep：1s 粒度轮询，有事就提前醒。

    两个中断源：
      ① 控制通道有手工 sync 任务 ⇒ 返回该任务（2026-08-03 起的老行为，不变）；
      ② MQ-W17：Memory Hub 有了没扫过的新数据 ⇒ **立刻把退避回落到基准**并结束 sleep。

    外加一个**不中断 sleep** 的唤醒源（Z4b）：
      ③ `external.has_change()`（Obsidian vault / Postgres 收件表有变化）⇒ 就地跑
         `external_round(src)`——即 `export_worker.run_once()` 那一轮回流 → 导出——
         然后**继续睡完剩下的时间**、退避阶梯不动。
         为什么不结束 sleep 去跑整轮：整轮先跑 rebuild，生产库空载一轮 ~72 s（MQ-W19），
         ≤5 s 的回流时延做不到，且每次编辑都白付一轮扫描。回流若真落了 Hub 事件，
         Hub 水位（②）下一跳就把整轮叫起来——重放与重导出仍走正常一轮。

    🔴 第二个中断源是本批红线（判据②）的承重墙：退避到上限（默认 1800s）后若只靠
    睡醒，新 turn 最坏要等满那么久 —— 那就是"为了省 CPU 把 MQ-W1 的积压
    请回来"。有了它，消费延迟反而**优于**退避前（不必等满一个 sleep 周期）。

    🔴 `tick`（每 `tick_every_s` 一次）不是可有可无的装饰：后台循环的心跳
    （`bladex:consolidator:heartbeat`）**TTL 只有 300s**，而退避上限是 1800s ——
    不在 sleep 里续跳，`bladex status` 会有 83% 的时间读不到心跳，把一个
    好好活着的进程显示成"没在跑"。省 CPU 不该以"看上去死了"为代价
    （与 MQ-W16 同族：状态行把两种成因合并成一句，指错方向）。

    `sleep` / `now` 可注入：阴性对照要在毫秒内跑完整个退避形状，真时钟做不到。
    """
    deadline = now() + seconds
    t0 = now()
    next_tick = now()
    ext_live = True
    while now() < deadline:
        if tick is not None and now() >= next_tick:
            next_tick = now() + max(1.0, tick_every_s)
            tick(deadline - now())
        if poll_cmd is not None:
            cmd = poll_cmd()
            if cmd:
                return cmd
        # 🔴 唤醒也归 `IDLE_BACKOFF` 管：开关关掉必须是**逐字**回到旧行为，
        # 否则判据④(a) 的阴性对照就有一条没被挡住的旁路——"关了开关还是有新行为"
        # 正是本仓反复踩的那类假回滚（开关只挡了一半）。
        _wake_armed = traffic is not None and (idle is None or idle.enabled)
        if _wake_armed and traffic.has_new():
            logger.info("index_consolidator_idle_wake",
                        slept_s=round(now() - t0, 1),
                        planned_s=round(seconds, 1),
                        interval_before_s=(round(idle.interval, 1) if idle else None))
            if idle is not None:
                # 🔴 判据④(b) 的反向变异拆的就是这一行：拆掉之后 sleep 仍会提前
                # 返回（看起来"响应了新流量"），但间隔留在上限 ⇒ 下一轮又睡满上限，
                # 消费延迟静默劣化。`test_wake_falls_back_to_base_interval` 盯它。
                idle.reset()
            return None
        # Z4b：同一把开关管（关退避 = 逐字旧行为，外部唤醒也一起关）。
        _ext_armed = (ext_live and external is not None and external_round is not None
                      and (idle is None or idle.enabled))
        if _ext_armed:
            src = external.has_change()
            if src:
                logger.info("index_consolidator_external_wake", source=src,
                            slept_s=round(now() - t0, 1), planned_s=round(seconds, 1),
                            interval_s=(round(idle.interval, 1) if idle else None))
                external_round(src)
                # 🔴 外部轮跑完基线就该对上；还报变化 = 基线没取上（探针坏了 / 导出每次都在写）。
                # 不解除的话每一拍都跑一轮回流 → 导出，退避形同虚设（反向变异 W2b 就是这个形状）。
                # 解除只到本次 sleep 结束，下一次 sleep 重新布防。
                if external.has_change():
                    ext_live = False
                    logger.warning("index_consolidator_external_wake_unsettled", source=src,
                                   hint="the probe still reports a change right after the "
                                        "export round -- early wake disarmed until the next sleep")
                # 不 `continue`：照常睡这一拍，下一拍 Hub 水位会叫起整轮（回流若落了事件）
        sleep(min(1.0, max(0.0, deadline - now())))
    return None


def main() -> int:
    # 🔴 env 加载与日志配置必须在 main() 里，**不能在模块导入时做**。
    # 这曾经是脚本，模块级 `_load_env_file()` 无害；变成包内模块后，任何
    # `import bladex_proxy.consolidator` 都会把真实 config/.env 灌进 os.environ——
    # 2026-08-04 实测：一个只是 import 它的测试，污染了同一进程后续全部 test_server
    # 用例（上游模型变成 live 值、auth 变成开启 → 一片 401）。
    # 导入必须无副作用；副作用归入口点。
    parser = argparse.ArgumentParser(description="BladeX Memory Index consolidation worker")
    parser.add_argument("--bladex-home", default="",
                        help="deployment root for this worker (passed by `bladex start`; "
                             "empty = BLADEX_HOME / config discovery)")
    parser.add_argument("--interval", type=float, default=60.0,
                        help="consolidation 轮询间隔（秒，默认 60）")
    parser.add_argument("--once", action="store_true",
                        help="跑一次后退出（用于 cron / 手动触发）")
    parser.add_argument("--max-turns", type=int, default=30,
                        help="每批最多处理的非 aux 轮次（默认 30；离线重放可给 200+，"
                             "在线保持默认防上游限流）")
    parser.add_argument("--full", action="store_true",
                        help="全量重建：清空 Memory Index → 重放 Memory Hub（ADR-0014 L0，用于阈值变更后重归属）；"
                             "带过滤参数时不清库，退化为定向重放")
    # 定向重建过滤（2026-07-28）：只处理匹配的 turn，不匹配的不消费也不蒸馏。
    parser.add_argument("--since", default="",
                        help="只处理 ts >= 该值的 turn（ISO 前缀，如 2026-07-13）")
    parser.add_argument("--until", default="",
                        help="只处理 ts <= 该值的 turn（ISO 前缀含端，如 2026-07-24 = 含当天）")
    parser.add_argument("--agents", default="",
                        help="只处理这些 agent 的 turn（逗号分隔，如 claude-code,hermes:default）")
    parser.add_argument("--exclude-agents", default="",
                        help="排除这些 agent 的 turn（逗号分隔，如压测流量 agent-0,agent-1,test-agent）")
    parser.add_argument("--reattribute", action="store_true",
                        help="按 AGENT_CLAIM 映射**只改归属、不重蒸**，然后退出（⑤）。"
                             "认领/改名/合并之后用它把存量 Fact 与 Matter 卡的 agent_id "
                             "改过来——这是一次 UPDATE，不碰内容/向量、不调 LLM，秒级完成。"
                             "🔴 不要用 `--unconsume --agents <桶>` 干这件事：那条路会"
                             "重新蒸馏出内容逐字相同的候选、被 novelty 判重丢弃"
                             "（verdict=vector_kill），归属改不成**而且不报错**。")
    parser.add_argument("--reattribute-dry-run", action="store_true",
                        help="配合 --reattribute：只统计会改多少，不落盘")
    parser.add_argument("--unconsume", action="store_true",
                        help="删除匹配 turn 的 consumed 标记后退出（不蒸馏）。"
                             "配合过滤参数用于'让某窗口被下轮增量重新拾起'；必须带至少一个过滤条件")
    parser.add_argument("--concurrency", type=int, default=0,
                        help="蒸馏并发路数（1=串行；0/省略 → BLADEX_DISTILL_CONCURRENCY，默认 4）。"
                             "历史：2026-08-13 之前 help 写'默认取 env'、实现却是 0→1 串行，"
                             "bench 全程串行蒸馏、排空超时、整跑数据作废——当时只改了 help 文案，"
                             "把缺陷记录成了规格。2026-08-22 同一个默认值打在生产上（CLI 起守护进程"
                             "只传 --interval），Memory Index 积压 720 轮不降。现已按 help 对齐实现："
                             "本进程读 env，`bladex` CLI 也显式传值。")
    parser.add_argument("--no-control", action="store_true",
                        help="关闭手工同步控制通道（默认常驻模式开启：CLI `bladex sync` 经 "
                             "Redis 下发任务、本进程执行并回写进度，见 bladex_proxy/sync_control.py）")
    args = parser.parse_args()
    # 🔴 MQ-A78 复核：--bladex-home 必须先钉住 BLADEX_HOME，再加载 .env / chdir。
    # 否则第一次部署根发现发生在旧 cwd，随后才改环境变量，配置和 PID/data 都会错根。
    if args.bladex_home:
        os.environ["BLADEX_HOME"] = os.path.abspath(os.path.expanduser(args.bladex_home))
    _load_env_file()
    _chdir_to_deployment_root()
    _configure_logging()
    _log_env_conflicts()

    def _split(v: str) -> set[str] | None:
        items = {s.strip() for s in v.split(",") if s.strip()}
        return items or None

    agents = _split(args.agents)
    exclude_agents = _split(args.exclude_agents)

    args.concurrency = resolve_concurrency(args.concurrency)

    has_filter = bool(args.since or args.until or agents or exclude_agents)
    if args.unconsume and not has_filter:
        parser.error("--unconsume 必须至少带一个过滤条件"
                     "（--since/--until/--agents/--exclude-agents），"
                     "否则等于清空所有消费标记 = 意外全量重放")

    from bladex_proxy.config import ProxyConfig
    from bladex_proxy.distillation import LLMDistiller
    from bladex_proxy.embedding import (
        build_embedder,
        effective_thresholds,
        embed_purpose,
        resolve_embed_settings,
        validate_embed_sensitivity,
    )
    from bladex_proxy.storage.memory_hub import MemoryHub
    from bladex_proxy.storage.memory_index import IndexDistillJournal, MemoryIndex

    cfg = ProxyConfig()

    # embedding 后端可选化（2026-07-26）：local（默认）/ api / proxy（共享 proxy 模型）。
    # 敏感层开 + api 档 -> strict 拒启动 / 告警+强制本地（与 proxy 同一 gate）。
    validate_embed_sensitivity(cfg)
    embed_settings = resolve_embed_settings(cfg)  # env > routing.toml [embedding] > 默认
    thresholds = effective_thresholds(cfg)  # 显式 env > per-model profile > 默认+告警

    # ADR-0021 section 2.5: legacy_ids -> principal 读侧映射（identity.toml 声明）。
    # empty registry（无 identity.toml）-> 空映射 = 现状（user_id 不变）。
    legacy_map = cfg.identity_registry.legacy_map()
    if legacy_map:
        logger.info("index_consolidator_legacy_map", entries=len(legacy_map))

    # ADR-0021 §2.3 修订（2026-08-16）：个人模式（无 identity.toml）折叠一切 user_id
    # 到单一本地身份。存量 turn 里的 key hash8 是"把凭证当身份"时代的产物，不是不同的人；
    # 不折叠则新流量落 `local`、存量留 hash8 -> 检索按 visibility 过滤当场两分裂。
    from bladex_proxy.identity import LOCAL_USER_ID  # 局部导入：模块级导入须无副作用
    fold_user_id = "" if not cfg.identity_registry.empty else LOCAL_USER_ID
    if fold_user_id:
        logger.info("index_consolidator_fold_user_id", fold_user_id=fold_user_id)

    # MQ-W17：空载退避。建在 starting 日志之前，为的是**把生效值打出来**——
    # 「函数有测试」≠「接线有守卫」（MQ-W13）：旋钮值只有出现在启动行里，
    # 才能在 live 上一眼证明两条拉起路径都真的传到位了。
    idle = resolve_idle_backoff(args.interval)

    logger.info("index_consolidator_starting",
                index_path=cfg.index_path, ledger_path=cfg.rocksdb_path,
                interval=args.interval, once=args.once, full=args.full,
                max_turns=args.max_turns,
                # MQ-W17 判据⑤：三个旋钮的生效值（不是配置里写了什么，是进程在用什么）
                idle_backoff=idle.enabled,
                idle_after=idle.after,
                idle_max_s=idle.max_s,
                # 2026-08-22：此前不打，于是"守护进程在串行蒸馏"这件事在日志里
                # 完全不可见——只能靠数 `Router completion()` 的间隔反推。
                distill_concurrency=args.concurrency,
                since=args.since or None, until=args.until or None,
                agents=sorted(agents) if agents else None,
                exclude_agents=sorted(exclude_agents) if exclude_agents else None,
                unconsume=args.unconsume,
                embed_backend=embed_settings.backend,
                novelty_threshold=thresholds["novelty"],
                semantic_threshold=thresholds["semantic"],
                digestion_threshold=thresholds["digestion"])

    # 打开 Memory Hub 和 Memory Index（读写）
    # ADR-0020 T1: 增量 consolidation 改 secondary 模式读 Memory Hub（try_catch_up_with_primary
    #   追新 proxy 写入，修 read_only 点时快照看不到新轮的停滞 bug）+ 派生台账迁 Memory Index meta
    #   （consolidator 独占 Memory Index 写锁，修 read_only Memory Hub 台账 100% 写失败）。
    # full rebuild 仍 read_write（维护操作，应在 proxy 停止时跑）。
    if args.full:
        ledger = MemoryHub(cfg.rocksdb_path, read_only=False)
        ledger_mode = "read_write"
    else:
        ledger = MemoryHub(cfg.rocksdb_path, secondary=True)
        ledger_mode = "secondary"
    try:
        open_hub_with_startup_window(ledger)
    except Exception as e:
        hint = _lock_conflict_hint(e, "Memory Hub")
        if hint is None:
            raise
        logger.error("hub_locked_by_another_process", path=cfg.rocksdb_path,
                     error=str(e), hint=hint)
        return 1
    logger.info("index_consolidator_ledger_mode", mode=ledger_mode, full=args.full)

    # Z1.2（MQ-W18）：硬规则热加载。本进程的硬规则消费方 = 导出 worker（Obsidian `Hard Rules.md`
    # / Postgres `hard_rules` 表），经 `lambda: cfg.effective_hard_rules` 每轮现取；规则一变
    # `hr_wake` 在退避 sleep 里叫起一轮导出（见下方 `external`）。
    from bladex_proxy import hard_rules as _hard_rules
    hr_wake = _hard_rules.ReloadWake()
    _hard_rules.arm("consolidator", hub=_hard_rules.hub_reader(ledger, catch_up=True),
                    initial=cfg.effective_hard_rules, on_change=hr_wake.notify)

    embedder = build_embedder(cfg, role="consolidator")

    # L2: 蒸馏器（ADR-0014 + ADR-0018 §3.1 结构化 + §3.6 自举 + §4.1 台账缓存）
    # 模型自举：env > routing [distill] > 模型池最弱档（写回 routing.toml）。
    # 台账缓存：read_only Memory Hub 读命中复用；写降级，T11 full rebuild 时写生效。
    from bladex_core.distillation import PassthroughDistiller

    from bladex_proxy.routing_config import bootstrap_distill_model
    distill_model = bootstrap_distill_model(
        cfg.distill_model, cfg.routing_config, cfg.routing_config_path,
    )
    distill_journal = IndexDistillJournal()  # ADR-0020 T1.3: 台账迁 Memory Index meta，延迟 bind(index)
    if distill_model:
        distiller = LLMDistiller(
            model=distill_model,
            api_base=cfg.upstream_api_base,
            api_key=cfg.upstream_api_key,
            journal=distill_journal,
        )
    else:
        distiller = PassthroughDistiller()
        logger.warning("index_distill_disabled_passthrough",
                       hint="no distill model (env/routing/pool all empty)")

    # ADR-0018 §3.2/§3.6: L4 链接裁决器（复用蒸馏同一模型，少一个行为面）
    link_judge = None
    if distill_model:
        from bladex_proxy.linking import LLMLinkJudge
        link_judge = LLMLinkJudge(
            model=distill_model,
            api_base=cfg.upstream_api_base,
            api_key=cfg.upstream_api_key,
        )
    # M2-2：写入时裁决器（复用蒸馏同一模型，拍板 4 —— 与 L4 同一条理由：
    # 少一个行为面、少一处配置、少一次"到底是哪个模型判的"的排查）。
    adjudicator = None
    if distill_model:
        from bladex_proxy.adjudicator import LLMAdjudicator
        adjudicator = LLMAdjudicator(
            model=distill_model,
            api_base=cfg.upstream_api_base,
            api_key=cfg.upstream_api_key,
        )
    logger.info("index_distiller_configured", model=distill_model or "passthrough",
                link_judge=distill_model or "none",
                adjudicator=distill_model or "none")

    # 检查 embedder 是否可用（触发首次加载 local 模型 / probe api / probe 远端 proxy）。
    # 必须在 MemoryIndex.open() 之前：model_identity（尤其 proxy 档远端身份）要先就绪，
    # 才能做 Memory Index 向量空间 model_id 不变量校验。
    try:
        # 共享模型档：run_proxy.sh 先拉 consolidator 再起 uvicorn，等 proxy 就绪
        # （proxy 侧还要加载/下载模型），避免启动竞态直接退出。
        if hasattr(embedder, "wait_ready"):
            embedder.wait_ready(timeout_s=float(os.environ.get(
                "BLADEX_EMBED_PROXY_WAIT_S", "180")))
        with embed_purpose("warmup"):
            embedder.embed(["warmup"])
    except Exception as e:
        logger.error("index_consolidator_embedder_unavailable", error=str(e),
                      backend=embed_settings.backend,
                      hint="local: install model / check disk; api: check key+model; "
                           "proxy: check BLADEX_EMBED_PROXY_URL is a running BladeX proxy")
        ledger.close()
        return 1

    index = MemoryIndex(
        cfg.index_path, embedder=embedder,
        novelty_threshold=thresholds["novelty"],
        semantic_threshold=thresholds["semantic"],
        digestion_threshold=thresholds["digestion"],
        distiller=distiller,
        link_judge=link_judge,
        adjudicator=adjudicator,
        sensitivity_config=cfg.routing_config.sensitivity_config(),
        entity_aware_novelty=cfg.entity_aware_novelty,
        entity_overlap_threshold=cfg.entity_overlap_threshold,
        novelty_topk=cfg.novelty_topk,
        embed_model_id=getattr(embedder, "model_identity", None),
        entity_aware_rerank=cfg.entity_aware_rerank,
        entity_rerank_alpha=cfg.entity_rerank_alpha,
        entity_rerank_expand_k=cfg.entity_rerank_expand_k,
    )
    try:
        # 向量空间不一致时 RuntimeError 拒启动（提示 scripts/reembed_index.py）——那条
        # 自带指引，原样抛；这里只翻译锁冲突。
        index.open()
    except Exception as e:
        hint = _lock_conflict_hint(e, "Memory Index")
        if hint is None:
            raise
        logger.error("index_locked_by_another_consolidator", path=cfg.index_path,
                     error=str(e), hint=hint)
        ledger.close()
        return 1
    distill_journal.bind(index)  # ADR-0020 T1.3: 台账迁 Memory Index，绑定 index（consolidator 独占写）

    logger.info("index_consolidator_ready",
                model=getattr(embedder, "model_name", "?"),
                backend=embed_settings.backend)

    # ⑤ 归属重映射（2026-08-26）：按 AGENT_CLAIM 只改归属、不重蒸，然后退出。
    # 认领/改名/合并之后走这条，秒级完成；它是 UPDATE 不是重建。
    if args.reattribute or args.reattribute_dry_run:
        try:
            ledger.catch_up()
        except Exception:  # noqa: BLE001 —— secondary 追新失败不阻断
            pass
        try:
            stats = index.reattribute_by_claims(
                ledger, dry_run=bool(args.reattribute_dry_run))
        finally:
            index.close()
            ledger.close()
        print(f"reattribute{' (dry-run)' if args.reattribute_dry_run else ''}: "
              f"facts={stats['facts']} matters={stats['matters']} "
              f"claim_pairs={stats['pairs']}")
        return 0

    # 定向反消费（2026-07-28）：删匹配 turn 的 consumed 标记后退出，
    # 下轮增量 rebuild 会重新拾起这批 turn（不清库、novelty 判重、台账命中不重调 LLM）。
    #
    # 🔴 **它不能用来改归属**（2026-08-26 验证）：重放会蒸出内容逐字相同的候选，
    # 被 novelty 判重丢弃（`verdict=vector_kill`，`consolidation_proxy` 里直接
    # `continue`）⇒ 归属改不成**而且不报错**。改归属走 `--reattribute`。
    if args.unconsume:
        _claimed = set()
        try:
            from bladex_proxy.storage.memory_index import build_agent_claim_map
            _claimed = set(build_agent_claim_map(ledger))
        except Exception:  # noqa: BLE001 —— 只为提示，失败不阻断
            pass
        if agents and (agents & _claimed):
            # 静默无效是本仓的高发病：宁可多一条刺眼的告警。
            logger.warning(
                "unconsume_will_not_change_attribution",
                agents=sorted(agents & _claimed),
                hint="these agents have AGENT_CLAIM mappings; replay re-distills "
                     "identical content and novelty drops it (verdict=vector_kill). "
                     "Use `--reattribute` to change attribution.")
        try:
            ledger.catch_up()
            removed = index.unconsume_matching(
                ledger, since=args.since, until=args.until,
                agents=agents, exclude_agents=exclude_agents,
            )
            logger.info("index_consolidator_unconsume_done", removed=removed,
                        hint="下轮增量 rebuild 会重放这些 turn")
        finally:
            index.close()
            ledger.close()
        return 0

    # ── 手工同步控制通道（2026-08-03，Beta T11 前置）──────────────────────────
    # 常驻模式下开启：`bladex sync run` 经 Redis 下发任务，本进程接单执行、
    # 逐步回写进度（CLI 轮询渲染 + 落日志）。Redis 不可达 → 通道关闭（只告警）。
    control = None
    if not args.once and not args.full and not args.no_control:
        try:
            import redis as _redis

            from bladex_proxy.sync_control import SyncControl
            _rc = _redis.Redis.from_url(cfg.redis_url, decode_responses=True,
                                        socket_connect_timeout=2)
            _rc.ping()
            control = SyncControl(_rc)
            logger.info("index_consolidator_sync_control_ready", redis=cfg.redis_url)
        except Exception as e:  # noqa: BLE001
            logger.warning("index_consolidator_sync_control_unavailable", error=str(e),
                           hint="`bladex sync` 将走 direct 模式或不可用")

    # MQ-W17：空载退避的唤醒信号。挂在 **Memory Hub** 上而不是 Redis——
    # 理由（实测换掉 P0 水位的那一版）见 `_HubWatermark` 的 docstring。
    # 常驻模式才需要：`--once` / `--full` 跑完就退，没有轮间 sleep。
    traffic = None
    if not args.once and not args.full:
        traffic = _HubWatermark(ledger)
        logger.info("index_consolidator_hub_watermark_ready", seq=ledger.seq())

    # ── 在线协同导出 worker（Beta T13/B3.2；无 config/export.toml = 关闭零回归）──
    export_worker = None
    from bladex_proxy.modules import module_enabled
    if not module_enabled("export"):
        # V-A3（F0.2）：模块门只挡 worker 构造；下面的 sync_control 心跳一行不动（B10 仪器消费）。
        logger.info("export_module_disabled", hint="BLADEX_MODULE_EXPORT=0")
    else:
        try:
            from bladex_proxy.cli import _admin_key
            from bladex_proxy.export_backflow import AdminBackflowClient, http_requester
            from bladex_proxy.export_sync import build_worker_from_config
            # Z4：回流经 proxy admin 正门落 Hub（本进程对 Hub 只读，不开写句柄）
            export_worker = build_worker_from_config(
                index, ledger, "config/export.toml",
                hard_rules=lambda: cfg.effective_hard_rules,
                backflow=AdminBackflowClient(http_requester(
                    f"http://127.0.0.1:{cfg.port}", _admin_key())))
            if export_worker is not None:
                logger.info("export_sync_worker_ready")
        except Exception as e:  # noqa: BLE001 —— 导出配置问题不拖垮 consolidator
            logger.warning("export_sync_worker_init_failed", error=str(e))

    # Z4b：退避 sleep 的第二个唤醒源（外部回流目标有变化）。与 Hub 水位同理，常驻模式才需要。
    # 没有开着回流的目标（无 export.toml / 全关 / inbox 关且无 Obsidian）⇒ None ⇒ 零探测。
    external = None
    if export_worker is not None and not args.once and not args.full:
        external = export_worker.change_probe()
        if external is not None:
            from bladex_core.flags import flag_number as _flag_number_z4b
            logger.info("index_consolidator_external_wake_ready",
                        pg_probe_s=_flag_number_z4b("BLADEX_BACKFLOW_PG_PROBE_S"))
        # Z1.2 ⑦：规则变了也算"外部有变化"——叠在 Z4b 探针前面（没有回流目标时单独生效）。
        hr_wake.inner = external
        external = hr_wake

    def _run_manual(cmd: dict) -> None:
        """执行一条手工同步任务（CLI 下发），进度回写控制通道。"""
        from bladex_proxy.sync_control import make_progress_writer
        job_id = cmd["job_id"]
        t0 = time.perf_counter()

        def _s(v: str) -> set[str] | None:
            items = {x for x in (v or []) if x}
            return items or None

        control.set_status(job_id, state="running", phase="start",
                           worker="consolidator")
        logger.info("index_sync_job_start", job_id=job_id, cmd=cmd)

        # ⑤ 归属重映射（2026-08-26）：按 AGENT_CLAIM 只改归属、不重蒸。
        # 🔴 走这条通道而不是让 dashboard 自己动手——Memory Index 的写权在
        # consolidator（proxy 侧是只读 secondary）。秒级完成，不进重建那条路。
        if str(cmd.get("action") or "sync") == "reattribute":
            try:
                ledger.catch_up()
                stats = index.reattribute_by_claims(ledger)
                control.set_status(
                    job_id, state="done", phase="reattribute",
                    facts=str(stats["facts"]), matters=str(stats["matters"]),
                    claim_pairs=str(stats["pairs"]),
                    elapsed_s=f"{time.perf_counter() - t0:.1f}")
                logger.info("index_reattribute_job_done", job_id=job_id, **stats)
            except Exception as e:  # noqa: BLE001 —— 单个 job 失败不该杀掉守护进程
                control.set_status(job_id, state="failed", phase="reattribute",
                                   error=str(e))
                logger.error("index_reattribute_job_failed", job_id=job_id, error=str(e))
            return

        try:
            ledger.catch_up()
            with embed_purpose(f"sync_job:{job_id}"):
                nf = index.rebuild_from_hub(
                    ledger, full=bool(cmd.get("full")), legacy_map=legacy_map,
                    fold_user_id=fold_user_id,
                    max_turns=int(cmd.get("max_turns") or 0),
                    since=cmd.get("since") or "", until=cmd.get("until") or "",
                    agents=_s(cmd.get("agents")),
                    exclude_agents=_s(cmd.get("exclude_agents")),
                    distill_concurrency=int(cmd.get("concurrency") or 0) or None,
                    progress_cb=make_progress_writer(control, job_id),
                )
            elapsed = round(time.perf_counter() - t0, 1)
            control.set_status(job_id, state="done", phase="done",
                               new_facts=nf, elapsed_s=elapsed,
                               backlog=str(index.last_rebuild_backlog))
            logger.info("index_sync_job_done", job_id=job_id, new_facts=nf,
                        elapsed_s=elapsed, distiller_stats=distiller.stats())
        except Exception as e:  # noqa: BLE001 —— 任务失败不拖垮常驻进程
            control.set_status(job_id, state="failed", error=str(e))
            logger.error("index_sync_job_failed", job_id=job_id, error=str(e))
        finally:
            # MQ-W17：手工 sync 是明确的"有人在用"，跑完回基准节奏——
            # 手工消化完积压紧接着来的流量不该撞在退避上。
            idle.reset()

    def _sleep_or_cmd(seconds: float) -> dict | None:
        """轮间 sleep（真实逻辑在模块级 `sleep_until_work`，便于注入时钟做阴性对照）。"""
        def _beat_while_sleeping(remaining: float) -> None:
            if control is None:
                return
            control.beat(state="idle", phase="sleeping", batch=batch_no,
                         interval_s=idle.interval,
                         sleep_remaining_s=round(remaining, 1))

        return sleep_until_work(
            seconds,
            poll_cmd=(control.poll_cmd if control is not None else None),
            traffic=traffic,
            idle=idle,
            external=external,
            external_round=_external_round,
            tick=_beat_while_sleeping,
        )

    def _external_round(source: str) -> None:
        """Z4b：外部目标有变化 ⇒ 就地跑一次回流 → 导出（同一个 `run_once`），再取基线。"""
        try:
            with embed_purpose("export_sync"):
                export_worker.run_once(force=True)
        except Exception as e:  # noqa: BLE001
            logger.warning("export_sync_worker_error", error=str(e), wake=source)
        finally:
            external.mark()

    # ── 后台循环心跳（2026-08-06）──────────────────────────────────────────
    # 手工 job 一直有进度回写，后台循环没有：队列恢复后 487 轮积压在消化，而
    # `bladex status` 的 lag 只在每轮提交时跳一次、轮内纹丝不动，和"卡死了"长得一样。
    # 心跳把轮内进度（第几批 / 蒸馏 71/305）送出来，让"慢"和"停"能被分开。
    heartbeat = None
    if control is not None:
        from bladex_proxy.sync_control import make_heartbeat_writer
        heartbeat = make_heartbeat_writer(control)
        control.beat(state="starting", phase="init", batch=0,
                     interval_s=args.interval, max_turns=args.max_turns)

    # M4-2：写入侧漏斗的**独立** metrics 端点（默认关，BLADEX_CONSOLIDATOR_METRICS_PORT）。
    # 两个进程各自暴露（2026-08-06 拍板）——哪半边没数一眼可见，
    # 不必经 P2 中转（中转层本身会成为一个可能坏掉且难以分辨的环节）。
    from bladex_proxy.consolidator_metrics import start_metrics_server
    from bladex_proxy.metrics import Metrics

    funnel_metrics = Metrics()
    index.attach_funnel(funnel_metrics)
    metrics_server = start_metrics_server(funnel_metrics)

    batch_no = 0
    try:
        while True:
            # 周期开始先看有没有手工任务（比常规增量优先）
            if control is not None:
                cmd = control.poll_cmd()
                if cmd:
                    _run_manual(cmd)
                    continue

            t0 = time.perf_counter()
            batch_no += 1
            if control is not None:
                control.beat(state="running", phase="scan", batch=batch_no,
                             done=0, total=0)
            # MQ-W17：**先**记水位再扫描——扫描要跑几十秒，期间到达的 turn
            # 必须留在水位之后，否则它会被这一次 mark 吃掉、要多等一个退避周期。
            if traffic is not None:
                traffic.mark()
            batch_error = False
            try:
                ledger.catch_up()  # ADR-0020 T1: secondary 追新 proxy 写入（非 secondary 空操作）
                with embed_purpose(f"rebuild:batch{batch_no}"):
                    new_facts = index.rebuild_from_hub(
                        ledger, full=args.full, legacy_map=legacy_map,
                        fold_user_id=fold_user_id,
                        max_turns=args.max_turns,
                        since=args.since, until=args.until,
                        agents=agents, exclude_agents=exclude_agents,
                        distill_concurrency=args.concurrency or None,
                        progress_cb=heartbeat,
                    )
                backlog = index.last_rebuild_backlog
                # MQ-W17：空载判据取**候选数**不取 new_facts——见
                # `MemoryIndex.last_rebuild_candidates` 的 docstring。
                _candidates = index.last_rebuild_candidates
                idle.observe(_candidates)
                elapsed = (time.perf_counter() - t0) * 1000
                if control is not None:
                    control.beat(state="idle" if not backlog else "running",
                                 phase="batch_done", batch=batch_no,
                                 last_new_facts=new_facts,
                                 last_elapsed_s=round(elapsed / 1000, 1),
                                 backlog=str(backlog))
                if new_facts > 0:
                    logger.info("index_consolidator_done",
                                new_facts=new_facts, elapsed_ms=round(elapsed, 1),
                                backlog=backlog,
                                distiller_stats=distiller.stats())
                else:
                    # MQ-W17：这一行从 debug 提到 info。空载轮此前在默认日志级别下
                    # **完全不可见**——134s 一轮、占空比 54% 的事实只能靠数
                    # `index_rebuild_scan_sorted` 的时间戳反推。退避生效与否
                    # 必须自带读数，否则又是一个"机制写完但没人看得见"。
                    logger.info("index_consolidator_idle", elapsed_ms=round(elapsed, 1),
                                backlog=backlog, candidates=_candidates,
                                idle_rounds=idle.idle_rounds,
                                next_interval_s=round(idle.interval, 1))
            except Exception as e:
                batch_error = True
                logger.error("index_consolidator_error", error=str(e))
                backlog = False  # 失败要退避，不立即空转重试
                if control is not None:
                    # 出错也要打心跳：不打的话读侧只看到进度停住，会当成"还在慢慢跑"
                    control.beat(state="error", phase="batch_failed",
                                 batch=batch_no, error=str(e)[:200])

            # T13：主管线之后跑导出（内部逐目标退避 + 永不外抛，不阻塞 rebuild）
            if export_worker is not None:
                try:
                    with embed_purpose("export_sync"):
                        export_worker.run_once()
                except Exception as e:  # noqa: BLE001
                    logger.warning("export_sync_worker_error", error=str(e))
                finally:
                    # 🔴 Z4b：基线在导出**写完之后**取——导出器自写不许唤醒自己
                    if external is not None:
                        external.mark()

            if args.full:
                return _finish_full_rebuild(
                    index, reason="error" if batch_error else None)
            if args.once:
                break
            # ③（2026-07-28）：有积压立即续跑，只在追平后才 sleep。
            # 旧行为每批固定睡 interval，6793 轮积压 = 多睡 ~3.8 小时。
            if backlog:
                continue
            if control is not None:
                control.beat(state="idle", phase="sleeping", batch=batch_no,
                             interval_s=idle.interval)
            # MQ-W17：退避关掉时 `idle.interval` 恒等于 `args.interval`（逐字旧行为）。
            cmd = _sleep_or_cmd(idle.interval)
            if cmd:
                _run_manual(cmd)
    except KeyboardInterrupt:
        logger.info("index_consolidator_stopped")
        if control is not None:
            control.beat(state="stopped", phase="stopped", batch=batch_no)
    finally:
        if metrics_server is not None:
            metrics_server.shutdown()
        index.close()
        ledger.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
