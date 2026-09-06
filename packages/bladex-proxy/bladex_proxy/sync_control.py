"""手工同步控制通道（2026-08-03，Beta T11 `bladex storage rebuild` 前置实现）。

场景：proxy 与 consolidator **正常运行中**，用户想立刻手工同步 Memory Hub→Memory Index（消化积压 /
全量重建），不停任何进程。Memory Index 写锁由 consolidator 独占 → 第二个进程不能直接写。
解法 = **CLI 当控制面、consolidator 当执行面**：

    CLI ── submit(job) ──► Redis(list bladex:sync:cmd) ──► consolidator 轮询接单
    CLI ◄─ 轮询 status ─── Redis(hash bladex:sync:job:{id}) ◄── 执行中持续回写进度

Redis 是现成依赖（Pipeline 就是它），不引入新组件。consolidator 不在跑时，CLI 退回
direct 模式自己执行（见 cli.py——Memory Hub 走 secondary 只读，proxy 无需停）。

协议（字段全 str，decode_responses=True）：
  cmd  : JSON {job_id, full, concurrency, max_turns, since, until, agents, exclude_agents}
  job  : hash {state: pending|running|done|failed, phase, done, total,
               new_facts, error, message, ts, ...}  （TTL 24h）
  last : bladex:sync:last = 最近一次 job_id（status 免记 id）
  hb   : bladex:consolidator:heartbeat = hash（下方"心跳"，TTL 5min）

心跳（2026-08-06 事故驱动）：手工 job 一直有进度回写，**后台常驻循环却没有**——
`_read_sync_job` 的 docstring 早写了"consolidator 无独立心跳机制（beta 已知限制）"。
后果是队列恢复后 487 轮积压在消化，而 `bladex status` 的 `-> Index N waiting` 只在
每轮提交时跳一次、轮内纹丝不动，看上去和"卡死了"一模一样，只能去 tail 日志才能
确认它在干活。心跳复用同一条 Redis 通道，写的是**后台循环**的轮内进度。
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

CMD_LIST = "bladex:sync:cmd"
JOB_PREFIX = "bladex:sync:job:"
LAST_JOB_KEY = "bladex:sync:last"
HEARTBEAT_KEY = "bladex:consolidator:heartbeat"
_JOB_TTL_S = 24 * 3600
#: 心跳 TTL。比 consolidator 默认 interval（60s）宽出几倍——进程死了心跳自己过期，
#: 读侧看不到就是看不到；但读侧**不能只靠 TTL 判活**，长轮次里两次写之间可能隔很久，
#: 所以心跳同时带 ts，由读侧算年龄自己判断（见 admin_read._read_consolidator_heartbeat）。
_HEARTBEAT_TTL_S = 300


class SyncControl:
    """控制通道薄封装。client = redis.Redis(decode_responses=True) 或测试替身。"""

    def __init__(self, client: Any) -> None:  # noqa: ANN401 —— 可注入测试替身
        self._r = client

    # ── CLI 侧 ──

    def submit(
        self,
        *,
        full: bool = False,
        concurrency: int = 1,
        max_turns: int = 0,
        since: str = "",
        until: str = "",
        agents: list[str] | None = None,
        exclude_agents: list[str] | None = None,
        action: str = "sync",
    ) -> str:
        """提交同步任务，返回 job_id。consolidator 侧 poll_cmd 接单。

        :param action: `"sync"`（默认，重建/增量）或 **`"reattribute"`**
            （2026-08-26：按 AGENT_CLAIM 只改归属、不重蒸）。
            🔴 归属重映射**必须走这条通道**：Memory Index 的写权在 consolidator
            （proxy 侧是只读 secondary），dashboard 不能自己动手。
        """
        job_id = f"sync-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        cmd = {
            "job_id": job_id, "full": bool(full),
            "concurrency": int(concurrency), "max_turns": int(max_turns),
            "since": since or "", "until": until or "",
            "agents": list(agents or []), "exclude_agents": list(exclude_agents or []),
            "action": action or "sync",
        }
        self.set_status(job_id, state="pending", phase="queued",
                        params=json.dumps(cmd, ensure_ascii=False))
        self._r.lpush(CMD_LIST, json.dumps(cmd, ensure_ascii=False))
        self._r.set(LAST_JOB_KEY, job_id, ex=_JOB_TTL_S)
        return job_id

    def get_status(self, job_id: str) -> dict[str, str]:
        return dict(self._r.hgetall(f"{JOB_PREFIX}{job_id}") or {})

    def last_job_id(self) -> str:
        return self._r.get(LAST_JOB_KEY) or ""

    # ── consolidator 侧 ──

    def poll_cmd(self) -> dict | None:
        """非阻塞取一条待执行命令；无则 None。畸形 JSON 丢弃（不让坏消息卡死队列）。"""
        raw = self._r.rpop(CMD_LIST)
        if not raw:
            return None
        try:
            cmd = json.loads(raw)
            if isinstance(cmd, dict) and cmd.get("job_id"):
                return cmd
        except (json.JSONDecodeError, TypeError):
            pass
        return None

    def set_status(self, job_id: str, **fields: Any) -> None:
        """回写任务状态（HSET + 续 TTL）。字段值统一转 str。"""
        key = f"{JOB_PREFIX}{job_id}"
        payload = {k: str(v) for k, v in fields.items() if v is not None}
        payload["ts"] = str(time.time())
        self._r.hset(key, mapping=payload)
        self._r.expire(key, _JOB_TTL_S)

    def beat(self, **fields: Any) -> None:
        """写一次后台循环心跳。**永不外抛**——观测挂了不能拖垮消化本体。

        用 HSET 覆盖式更新而非整键重写：一次心跳只带它知道的字段（比如蒸馏进度只有
        phase/done/total），上一轮写的 batch、last_new_facts 该留着。
        """
        try:
            payload = {k: str(v) for k, v in fields.items() if v is not None}
            payload["ts"] = str(time.time())
            self._r.hset(HEARTBEAT_KEY, mapping=payload)
            self._r.expire(HEARTBEAT_KEY, _HEARTBEAT_TTL_S)
        except Exception:  # noqa: BLE001
            pass

    def get_heartbeat(self) -> dict[str, str]:
        try:
            return dict(self._r.hgetall(HEARTBEAT_KEY) or {})
        except Exception:  # noqa: BLE001
            return {}


def make_heartbeat_writer(control: SyncControl, min_interval_s: float = 1.0):
    """把后台循环的 rebuild 进度节流写进心跳（事件形状与 job 进度同源）。

    节流是必需的：蒸馏进度回调每条 fact 都可能触发一次，逐条 HSET 会把 Redis 当日志用。
    但 `phase in (done, failed)` 不节流——收尾那一下丢了，心跳就永远停在中途的数字上，
    读侧会把"已经跑完"误报成"卡在 71/305"。
    """
    last = {"t": 0.0}

    def _cb(ev: dict) -> None:
        now = time.monotonic()
        if now - last["t"] < min_interval_s and ev.get("phase") not in ("done", "failed"):
            return
        last["t"] = now
        control.beat(state="running", **ev)

    return _cb


def make_progress_writer(control: SyncControl, job_id: str,
                         min_interval_s: float = 0.3):
    """把 rebuild 的 progress_cb 事件节流回写到 job status（终端展示由 CLI 轮询渲染）。"""
    last = {"t": 0.0}

    def _cb(ev: dict) -> None:
        now = time.monotonic()
        if now - last["t"] < min_interval_s and ev.get("phase") not in ("done", "failed"):
            return
        last["t"] = now
        try:
            control.set_status(job_id, state="running", **ev)
        except Exception:  # noqa: BLE001 —— 状态回写失败绝不影响重建本体
            pass

    return _cb
