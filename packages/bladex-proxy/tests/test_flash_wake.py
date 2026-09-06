"""会话触发 push（Jason 2026-08-29 四点拍板）：
① Redis wake 取代定时轮询（生产=pipeline 落库后、消费=阻塞 XREAD、降级=轮询+告警）；
② 启动自检（骨架重建 + 停机期直编识别，baseline sha 区分「直编」vs「投影过期」）；
③/④ 预留低频轨道（钩子非 None 才调度，不造 flag 不空转）。
直编语义修订：捕获 = **采纳本地版本**（不写 Hub，随会话经模型工具面自然收敛），
日志留痕；未收敛期间限流重发采纳（proxy 重启自愈）。
"""

from __future__ import annotations

import asyncio
import os
import time

from bladex_core.ledger import (
    Ledger, LedgerEntry, ACTOR_MODEL, add_entry, ledger_md_path, render_ledger_md,
)
from bladex_proxy.flash_daemon import FlashDaemon
from bladex_proxy.flash_wake import WAKE_STREAM, WakeConsumer, notify_wake


# ── wake 通道 ──

class _FakeRedis:
    def __init__(self, batches=None, fail=False):
        self.batches = list(batches or [])
        self.fail = fail
        self.added = []

    def ping(self):
        if self.fail:
            raise ConnectionError("down")

    def xread(self, streams, block=0, count=0):
        if self.fail:
            raise ConnectionError("down")
        return self.batches.pop(0) if self.batches else []

    async def xadd(self, stream, fields, maxlen=None, approximate=False):
        if self.fail:
            raise ConnectionError("down")
        self.added.append((stream, fields, maxlen))


class TestWakeChannel:
    def _consumer(self, fake):
        c = WakeConsumer("redis://x")
        c._client = fake          # 绕过真连接：单测钉协议不钉网络
        return c

    def test_woken_advances_last_id(self):
        fake = _FakeRedis(batches=[[(WAKE_STREAM, [("1-1", {"kind": "turn"}),
                                                  ("1-2", {"kind": "turn"})])]])
        c = self._consumer(fake)
        assert c.wait(0.01) is True
        assert c._last_id == "1-2", "游标推进——同一门铃不重复吵醒"

    def test_clean_timeout_returns_false(self):
        assert self._consumer(_FakeRedis()).wait(0.01) is False

    def test_degraded_polling_returns_true_with_reset(self):
        c = self._consumer(_FakeRedis(fail=True))
        t0 = time.time()
        assert c.wait(0.05) is True, "降级 = 按轮询节奏干活，不是停摆"
        assert time.time() - t0 >= 0.05
        assert c._client is None, "掉线后清客户端 ⇒ 下轮懒重连（ADR-0017 形态）"

    def test_notify_is_fire_and_forget(self):
        ok, bad = _FakeRedis(), _FakeRedis(fail=True)
        asyncio.run(notify_wake(ok, "turn"))
        assert ok.added and ok.added[0][0] == WAKE_STREAM
        asyncio.run(notify_wake(bad, "turn"))   # 不抛——门铃故障不外溢
        asyncio.run(notify_wake(None, "turn"))  # 无 redis 直接跳过


# ── daemon：启动自检 + 采纳台账 + 树门控 ──

def _pool():
    led = Ledger(ledger_id="ldg-wake0000001", title="唤醒测试")
    return {led.ledger_id: led}


def _mk(tmp_path, holder, events):
    return FlashDaemon(
        root=str(tmp_path), principal="u1",
        ledger_source=lambda: (holder["pool"], {}),
        emit_admin_event=lambda t, p: events.append((t, p)))


class TestStartupCheck:
    def test_skeleton_and_untouched_projection_rebuilt(self, tmp_path):
        holder, events = {"pool": _pool()}, []
        d = _mk(tmp_path, holder, events)
        stats = d.startup_check()
        assert stats["written"] >= 1 and stats["offline_edits"] == 0
        assert os.path.isdir(os.path.join(str(tmp_path), "u1", "personal",
                                          "ledgers")), \
            "骨架建在池的真实路径（本卡梳理时抓到的第一版 bug：建成了 u1/ledgers）"

    def test_offline_edit_adopted_not_clobbered(self, tmp_path):
        """停机期直编：baseline sha 对不上 ⇒ 采纳用户版，不用旧投影冲掉。"""
        holder, events = {"pool": _pool()}, []
        d = _mk(tmp_path, holder, events)
        d.startup_check()
        path = ledger_md_path(str(tmp_path), "u1", "ldg-wake0000001")
        edited = open(path, encoding="utf-8").read().replace(
            "## Open\n\n_(empty)_", "## Open\n\n- 停机时手记的一行")
        open(path, "w", encoding="utf-8").write(edited)
        # 新实例 = 模拟 daemon 重启（持久 state 从盘上装回）
        d2 = _mk(tmp_path, holder, events)
        stats = d2.startup_check()
        assert stats["offline_edits"] == 1 and stats["user_edits"] == 1
        assert events and events[-1][0] == "ledger_user_edit"
        assert "停机时手记的一行" in open(path, encoding="utf-8").read()

    def test_stale_projection_is_not_misread_as_edit(self, tmp_path):
        """Hub 在停机期前进而用户没动文件 ⇒ 是过期投影，放心重写（不误采纳）。"""
        holder, events = {"pool": _pool()}, []
        d = _mk(tmp_path, holder, events)
        d.startup_check()
        led2 = add_entry(holder["pool"]["ldg-wake0000001"], "verified",
                         LedgerEntry(text="停机期模型写的", source="tool", ref="x"),
                         actor=ACTOR_MODEL)
        holder["pool"] = {"ldg-wake0000001": led2}
        d2 = _mk(tmp_path, holder, events)
        stats = d2.startup_check()
        assert stats["offline_edits"] == 0 and stats["user_edits"] == 0
        path = ledger_md_path(str(tmp_path), "u1", "ldg-wake0000001")
        assert "停机期模型写的" in open(path, encoding="utf-8").read()
        assert events == []


class TestAdoptionReemit:
    def test_pending_reemits_with_throttle(self, tmp_path):
        holder, events = {"pool": _pool()}, []
        d = _mk(tmp_path, holder, events)
        d.run_once()
        path = ledger_md_path(str(tmp_path), "u1", "ldg-wake0000001")
        edited = open(path, encoding="utf-8").read().replace(
            "## Open\n\n_(empty)_", "## Open\n\n- 手记")
        open(path, "w", encoding="utf-8").write(edited)
        d.run_once()                       # 捕获=采纳（第 1 次上报）
        assert len(events) == 1
        d.push()
        d.run_once()                       # 采纳后 30s 限流内 ⇒ 不重发
        assert len(events) == 1
        d._reemit_last = {k: 0.0 for k in d._reemit_last}   # 拨钟过限流窗
        d.push()
        d.run_once()                       # 未收敛 ⇒ 重发采纳（自愈通道）
        assert len(events) == 2

    def test_pending_survives_daemon_restart(self, tmp_path):
        holder, events = {"pool": _pool()}, []
        d = _mk(tmp_path, holder, events)
        d.run_once()
        path = ledger_md_path(str(tmp_path), "u1", "ldg-wake0000001")
        edited = open(path, encoding="utf-8").read().replace(
            "## Open\n\n_(empty)_", "## Open\n\n- 手记")
        open(path, "w", encoding="utf-8").write(edited)
        d.run_once()
        d2 = _mk(tmp_path, holder, events)     # 重启：adopted 台账装回
        d2.startup_check()
        assert "手记" in open(path, encoding="utf-8").read(), \
            "采纳未收敛 + 重启 ⇒ 零覆盖窗口必须还在"


class TestTreeTrafficGating:
    def test_tree_only_rescans_after_push(self, tmp_path):
        calls = []

        def tree_source():
            calls.append(1)
            return {"sessions": {}, "session_ledgers": {}, "agent_last_seen": {},
                    "agent_prompts": {}, "excluded": []}

        d = FlashDaemon(root=str(tmp_path), principal="u1",
                        ledger_source=lambda: ({}, {}),
                        emit_admin_event=lambda t, p: None,
                        tree_source=tree_source, tree_every_s=0.0)
        d.run_once()
        assert len(calls) == 1
        d.run_once()                    # 没有新流量 ⇒ 不重扫（wake 驱动的意义）
        assert len(calls) == 1
        assert d.tree_wait_s() is None
        d.push()
        assert d.tree_wait_s() == 0.0
        d.run_once()
        assert len(calls) == 2


class TestReservedHooks:
    def test_hooks_default_off(self, tmp_path):
        d = _mk(tmp_path, {"pool": {}}, [])
        assert d.upkeep_hook is None and d.extract_hook is None
        assert d.upkeep_every_s == 0.0 and d.extract_every_s == 3600.0
