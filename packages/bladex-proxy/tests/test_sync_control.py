"""sync CLI 控制通道协议（bladex_proxy/sync_control.py）。

FakeRedis 只实现协议用到的六个方法——测试跑真实协议路径，不 mock SyncControl 本身。
"""

from __future__ import annotations

import json
import time

from bladex_proxy.sync_control import (
    CMD_LIST,
    SyncControl,
    make_progress_writer,
)


class FakeRedis:
    def __init__(self) -> None:
        self.lists: dict[str, list[str]] = {}
        self.hashes: dict[str, dict[str, str]] = {}
        self.kv: dict[str, str] = {}
        self.ttl: dict[str, int] = {}

    def lpush(self, key, val):
        self.lists.setdefault(key, []).insert(0, val)

    def rpop(self, key):
        lst = self.lists.get(key) or []
        return lst.pop() if lst else None

    def hset(self, key, mapping):
        self.hashes.setdefault(key, {}).update(mapping)

    def hgetall(self, key):
        return dict(self.hashes.get(key, {}))

    def expire(self, key, ttl):
        self.ttl[key] = ttl

    def set(self, key, val, ex=None):
        self.kv[key] = val

    def get(self, key):
        return self.kv.get(key)


def test_submit_poll_status_roundtrip():
    c = SyncControl(FakeRedis())
    job_id = c.submit(full=True, concurrency=8, since="2026-08-01",
                      agents=["claude-code"])
    assert job_id and c.last_job_id() == job_id
    # 提交即有 pending 状态（CLI 等待接单的判定依据）
    assert c.get_status(job_id)["state"] == "pending"

    cmd = c.poll_cmd()
    assert cmd is not None
    assert cmd["job_id"] == job_id and cmd["full"] is True
    assert cmd["concurrency"] == 8 and cmd["agents"] == ["claude-code"]
    assert c.poll_cmd() is None  # 队列已空

    c.set_status(job_id, state="running", phase="distill", done=5, total=100)
    st = c.get_status(job_id)
    assert st["state"] == "running" and st["done"] == "5" and st["total"] == "100"
    c.set_status(job_id, state="done", new_facts=42)
    assert c.get_status(job_id)["state"] == "done"


def test_malformed_cmd_dropped_not_stuck():
    r = FakeRedis()
    c = SyncControl(r)
    r.lpush(CMD_LIST, "not-json{{{")
    assert c.poll_cmd() is None          # 坏消息被丢弃
    job = c.submit()
    assert c.poll_cmd()["job_id"] == job  # 队列没被卡死


def test_progress_writer_throttles_but_keeps_done():
    r = FakeRedis()
    c = SyncControl(r)
    job = c.submit()
    cb = make_progress_writer(c, job, min_interval_s=10.0)  # 大间隔 → 只放行首个
    cb({"phase": "distill", "done": 1, "total": 100})
    cb({"phase": "distill", "done": 2, "total": 100})   # 被节流
    st = c.get_status(job)
    assert st["done"] == "1"
    cb({"phase": "done", "done": 100, "total": 100})    # done 永不节流
    assert c.get_status(job)["phase"] == "done"


def test_progress_writer_survives_redis_failure():
    class Broken(FakeRedis):
        def hset(self, key, mapping):
            raise ConnectionError("redis down")

    c = SyncControl(Broken())
    cb = make_progress_writer(c, "j1", min_interval_s=0)
    cb({"phase": "distill", "done": 1})  # 不抛 —— 观测面失败不影响重建
    time.sleep(0)


def test_cmd_json_fields_are_plain_types():
    """协议字段必须 JSON 可序列化（跨进程边界）。"""
    r = FakeRedis()
    SyncControl(r).submit(exclude_agents=["a", "b"], until="2026-08-03")
    raw = r.lists[CMD_LIST][0]
    cmd = json.loads(raw)
    assert cmd["exclude_agents"] == ["a", "b"] and cmd["until"] == "2026-08-03"


# ── 后台循环心跳（2026-08-06：487 轮积压在消化，status 看着像卡死）──────────


def test_heartbeat_roundtrip_merges_fields():
    """心跳是覆盖式合并：一次只带它知道的字段，之前写的 batch 不能被抹掉。"""
    c = SyncControl(FakeRedis())
    c.beat(state="running", phase="scan", batch=3)
    c.beat(state="running", phase="distill", done=71, total=305)
    hb = c.get_heartbeat()
    assert hb["batch"] == "3"        # 上一次写的，没被这次覆盖掉
    assert hb["phase"] == "distill"
    assert hb["done"] == "71" and hb["total"] == "305"
    assert float(hb["ts"]) > 0


def test_heartbeat_never_raises_when_redis_is_broken():
    """观测面失败绝不能拖垮消化本体——心跳写不进去就当没写。"""
    class Broken(FakeRedis):
        def hset(self, key, mapping):
            raise ConnectionError("redis down")

        def hgetall(self, key):
            raise ConnectionError("redis down")

    c = SyncControl(Broken())
    c.beat(state="running", phase="distill", done=1)  # 不抛
    assert c.get_heartbeat() == {}


def test_heartbeat_writer_throttles_but_never_drops_the_final_beat():
    """收尾那一下丢了，心跳就永远停在中途的数字上，读侧会把跑完当成卡住。"""
    from bladex_proxy.sync_control import make_heartbeat_writer

    c = SyncControl(FakeRedis())
    cb = make_heartbeat_writer(c, min_interval_s=10.0)
    cb({"phase": "distill", "done": 1, "total": 100})
    cb({"phase": "distill", "done": 2, "total": 100})   # 被节流
    assert c.get_heartbeat()["done"] == "1"
    cb({"phase": "done", "done": 100, "total": 100})    # done 不节流
    assert c.get_heartbeat()["phase"] == "done"
