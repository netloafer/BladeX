"""MQ-P16 验收：BladeX 自产的内循环 aux 轮不进 Hub 的"agent 压缩了没有"判据面。

病灶（09-02 复核 `0271f73` 发现）：`MemoryHub.put()` 对每条 Turn 都做
`_detect_prefix_change` + `_remember_prefix`，不分 aux。MQ-P9 的内循环轮
`request_messages = [system 标记, assistant(tool_calls), tool…]`，标记含轮序 /
trigger_ts ⇒ 每轮前缀都不同、msg_count 2–3：
① 每条 aux 轮必然"所有候选都对不上" ⇒ `hub_prefix_changed` warning × rounds
  （V-C1 刚清零的 warning 通道被我们自己重新灌满）；
② `_session_last` 每会话 4 槽 LRU，一次 rounds≥4 的内循环把主会话槽挤空 ⇒
  主会话下一轮假阳性；`_recover_session_cache` 取每会话最后 N 条 ⇒ 会话以内循环
  收尾则重启后主会话首请求假阳性 ⇒ `cache_state` 判冷、assembly L2 分支变化。
MQ-S47 续"短请求挤占槽位"的形态，这次是我们自己造的短请求。

修法**只跳这一种** `aux_source == INNER_LOOP_AUX_SOURCE`（写入照旧）。其它 aux 轮是
agent 真发来的历史，改它们归 MQ-S47 续（0.3.0）——本文件用 `aux_source="other"`
做阴性对照钉住这条边界：把跳过去掉、或换成"跳全部 aux"，都必须变红。

判别力：4 条 aux 轮（= 槽数）夹在主会话两轮之间；不跳 ⇒ 主会话第二轮必假阳性。
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import structlog
from bladex_proxy.innerloop import INNER_LOOP_AUX_SOURCE
from bladex_proxy.loopledger import PendingRound, build_inner_loop_turn
from bladex_proxy.models import Identity, Turn, TurnStatus
from bladex_proxy.storage.memory_hub import MemoryHub

IDENTITY = Identity(user_id="u1", agent_id="claude-code", session_id="s1")
PREFIX = IDENTITY.session_prefix()


def _msgs(n: int) -> list[dict]:
    return [{"role": "system", "content": "sys"}] + \
        [{"role": "user", "content": f"main-{i}"} for i in range(n - 1)]


def _main_turn(n: int) -> Turn:
    return Turn(identity=IDENTITY, model="m", request_messages=_msgs(n),
                response_text="r", status=TurnStatus.OK)


def _tc(name, cid):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": "{}"}}


def _aux_turn(i: int, *, aux_source: str = INNER_LOOP_AUX_SOURCE) -> Turn:
    pend = PendingRound(
        round_index=i, rounds_total=4, tool_names=["bladex_memory_search"],
        tool_ms=1.0, llm_ms=2.0, usage={"total": 5},
        messages=[{"role": "assistant", "content": None,
                   "tool_calls": [_tc("bladex_memory_search", f"c{i}")]},
                  {"role": "tool", "tool_call_id": f"c{i}", "content": f"hit-{i}"}],
        reply={"role": "assistant", "content": f"r{i}"})
    t = build_inner_loop_turn(IDENTITY, "m", pend, trigger_ts=f"T{i}")
    if aux_source != INNER_LOOP_AUX_SOURCE:
        # 阴性对照：同样形态的短轮，但**不是**我们自产的那一种
        t = t.model_copy(update={"aux_source": aux_source,
                                 "identity": t.identity.model_copy(
                                     update={"aux_source": aux_source})})
    return t


class _Hub:
    def __init__(self, td: str) -> None:
        self.path = Path(td) / "rocksdb"
        self.hub = MemoryHub(self.path)
        self.hub.open()
        self._n = 1719907200

    def put(self, turn: Turn) -> Turn:
        self._n += 1
        key = IDENTITY.storage_key(f"{self._n}-0")
        self.hub.put(key, turn)
        return self.hub.get(key)


def _interleave(h: _Hub, aux_source: str) -> tuple[list[dict], Turn, Turn]:
    """主会话 3 条 → 4 条 aux → 主会话 5 条（append-only 延续）。
    返回 (捕获日志, 第一主轮, 第二主轮)。

    🔴 槽位判据用 **prefix_hash** 不用 msg_count：aux 轮也是 3 条消息，按数量分不出
    （首版就是这么红的——`[3, 3, 3, 5]` 里的三个 3 是 aux 的）。"""
    with structlog.testing.capture_logs() as cap:
        first = h.put(_main_turn(3))
        for i in range(1, 5):
            h.put(_aux_turn(i, aux_source=aux_source))
        second = h.put(_main_turn(5))
    return cap, first, second


def _slot_hashes(hub: MemoryHub) -> set[str]:
    return {ph for ph, _c in hub._session_last.get(PREFIX, ())}


class TestPutSkipsSelfMadeRounds:
    def test_main_session_not_flagged_and_slots_clean(self):
        with tempfile.TemporaryDirectory() as td:
            h = _Hub(td)
            cap, first, second = _interleave(h, INNER_LOOP_AUX_SOURCE)
            assert second.prefix_changed is False, "主会话第二轮是 append-only 延续"
            warnings = [e for e in cap if e.get("event") == "hub_prefix_changed"]
            assert warnings == [], f"自产轮不该报 hub_prefix_changed：{warnings}"
            slots = list(h.hub._session_last[PREFIX])
            # 槽里只有主会话两条：数量恰 2，且含第一主轮的 hash（没被 aux 挤掉）
            assert len(slots) == 2 and first.prefix_hash in _slot_hashes(h.hub), slots
            # 写入照旧：4 条 aux 轮都在库里
            keys = [k for k, _ in h.hub.scan_meta(PREFIX)]
            assert len(keys) == 6
            h.hub.close()

    def test_negative_control_other_aux_is_still_detected(self):
        """阴性对照：同形态短轮但 aux_source="other" ⇒ 旧行为——每条报警、槽被挤空、
        主会话第二轮假阳性。**这条变绿 = 有人把跳过扩成了"全部 aux"**（MQ-S47 续范畴）。"""
        with tempfile.TemporaryDirectory() as td:
            h = _Hub(td)
            cap, first, second = _interleave(h, "other")
            warnings = [e for e in cap if e.get("event") == "hub_prefix_changed"]
            assert len(warnings) >= 4, "非自产 aux 轮仍参与检测（每条都对不上）"
            assert second.prefix_changed is True, "4 条短轮挤空了主会话槽 ⇒ 假阳性"
            assert first.prefix_hash not in _slot_hashes(h.hub), "第一主轮已被挤出槽"
            h.hub.close()


class TestRecoverSessionCacheSkipsSelfMadeRounds:
    def _write_then_reopen(self, aux_source: str) -> tuple[MemoryHub, Turn]:
        td = tempfile.mkdtemp()
        h = _Hub(td)
        first = h.put(_main_turn(3))
        for i in range(1, 5):
            h.put(_aux_turn(i, aux_source=aux_source))     # 会话以内循环收尾
        h.hub.close()
        hub2 = MemoryHub(h.path)
        hub2.open()                                        # 触发 _recover_session_cache
        return hub2, first

    def test_restart_after_inner_loop_keeps_main_prefix(self):
        hub2, first = self._write_then_reopen(INNER_LOOP_AUX_SOURCE)
        try:
            assert _slot_hashes(hub2) == {first.prefix_hash}
            # proxy 侧判据（cache_state / assembly L2 走的就是它）：主会话下一轮不是"变了"
            assert hub2.check_prefix_changed(PREFIX, _msgs(5)) is False
        finally:
            hub2.close()

    def test_negative_control_other_aux_pollutes_recovered_slots(self):
        hub2, first = self._write_then_reopen("other")
        try:
            assert first.prefix_hash not in _slot_hashes(hub2), "主轮被 4 条 aux 挤出恢复面"
            assert hub2.check_prefix_changed(PREFIX, _msgs(5)) is True
        finally:
            hub2.close()
