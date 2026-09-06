"""V-C1 验收剧本：会话前缀候选多槽（MQ-S47）。

病例（2026-08-28 两跑 live）：`hub_prefix_changed` 共 54 次告警，**53 次假阳性**，
真阳性 0；第二跑里它是日志中唯一的 warning（18/18，error=0）——
warning 通道被一个恒假的告警占满。

根因：`_session_last` 单槽 + `put()` 无条件覆盖，而 claude-code 在同一个
`session_prefix` 下混跑两路流量（主会话 msg_count 递增 / 3 条消息的小请求），
两路交替覆盖同一个槽，双方都拿对方当基准。

🔴 本文件的重心是 **A2**：修法不得为了消灭假阳性而把真阳性一起抹掉。
假阳性吵，假阴性静默——后者更坏。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from bladex_proxy.models import Identity, Turn, TurnStatus
from bladex_proxy.storage.memory_hub import MemoryHub


def _msgs(n: int, tag: str = "main") -> list[dict]:
    return [{"role": "user", "content": f"{tag}-{i}"} for i in range(n)]


def _turn(identity: Identity, messages: list[dict]) -> Turn:
    return Turn(identity=identity, model="test", request_messages=messages,
                response_text="r", status=TurnStatus.OK)


class _Hub:
    """临时目录里的 MemoryHub，带自增 entry_id。"""

    def __init__(self, tmpdir: str) -> None:
        self.hub = MemoryHub(Path(tmpdir) / "rocksdb")
        self.hub.open()
        self.identity = Identity(user_id="u1", agent_id="claude-code",
                                 session_id="s1", turn_index=0)
        self.prefix = self.identity.session_prefix()
        self._n = 1719907200

    def put(self, messages: list[dict]) -> bool:
        """写一轮，返回这一轮是否被判 prefix_changed。"""
        self._n += 1
        turn = _turn(self.identity, messages)
        changed = self.hub._detect_prefix_change(turn)
        self.hub.put(self.identity.storage_key(f"{self._n}-0"), turn)
        return changed

    def close(self) -> None:
        self.hub.close()


# ══════════════════════════════════════════════════════════════
# A1 阳性：主会话与小请求交替 —— 一次都不该报
# ══════════════════════════════════════════════════════════════

def test_interleaved_main_and_small_requests_report_once_then_go_quiet():
    """复刻 live 形态：主会话 append-only，中间插 3 条消息的小请求。

    改动前每次切换都报（本形态 8 次），**全是假阳性**。
    改动后只在小请求**首次**出现时报一次——那一刻池子里确实没有它的候选，
    单轮内它与"真压缩"不可区分（两者都是"所有候选都对不上"）。
    此后两路各占一槽，交替多久都不再报。

    🔴 这里**不能**靠"消息数骤降 ⇒ 是新路不是压缩"来消掉最后这一次：
    真压缩的信号形状与它一模一样，那样做只是把假阳性换成假阴性。
    单轮判不了的事就不要假装判得了——留一条告警比静默漏报好。
    """
    small = _msgs(3, "small")
    with tempfile.TemporaryDirectory() as td:
        h = _Hub(td)
        reported: list[str] = []

        h.put(_msgs(2))            # 会话首条，无候选
        for n in (4, 6, 8, 10):
            if h.put(small):
                reported.append(f"small@{n}")
            if h.put(_msgs(n)):
                reported.append(f"main@{n}")

        h.close()
        assert reported == ["small@4"], \
            f"只该在小请求首次出现时报一次，实际：{reported}"


def test_small_request_goes_quiet_after_first_occurrence():
    """小请求第二次起命中**它自己**建立的槽 —— 不再报。

    改动前每次都撞 `len(3) < prev_msg_count` 直接 return True，
    live 两跑里这一支贡献了 35 次告警中的大半。
    """
    small = _msgs(3, "small")
    with tempfile.TemporaryDirectory() as td:
        h = _Hub(td)
        h.put(_msgs(2))
        h.put(_msgs(20))
        assert h.put(small) is True, "首见：池子里确实没有它，报一次是诚实的"
        for _ in range(5):
            assert h.put(small) is False, "第二次起必须安静"
        h.close()


def test_main_session_never_reports_after_small_request_appears():
    """主会话不该被小请求带偏 —— 这是改动前假阳性的另一半（17 次）。"""
    small = _msgs(3, "small")
    with tempfile.TemporaryDirectory() as td:
        h = _Hub(td)
        h.put(_msgs(2))
        h.put(_msgs(20))
        h.put(small)                       # 首见，报（已在上一条测试覆盖）
        for n in (22, 24, 26, 28):
            assert h.put(_msgs(n)) is False, f"主会话 msgs={n} 被带偏了"
            h.put(small)
        h.close()


# ══════════════════════════════════════════════════════════════
# A2 阴性：真压缩仍然要报（这条是本卡的红线）
# ══════════════════════════════════════════════════════════════

def test_real_compaction_is_still_reported():
    """agent 真的改写了历史前缀 ⇒ 所有候选都对不上 ⇒ 必须报。"""
    with tempfile.TemporaryDirectory() as td:
        h = _Hub(td)
        h.put(_msgs(2))
        h.put(_msgs(10))
        h.put(_msgs(14))
        # 压缩：同样长度，但前缀内容整体换掉
        compacted = [{"role": "user", "content": f"compacted-{i}"} for i in range(14)]
        assert h.put(compacted) is True, \
            "真压缩必须报——修法不得为消灭假阳性而制造假阴性"
        h.close()


def test_real_compaction_reported_even_amid_small_request_traffic():
    """小请求把槽占着的时候，真压缩依然要能被检出。"""
    small = _msgs(3, "small")
    with tempfile.TemporaryDirectory() as td:
        h = _Hub(td)
        h.put(_msgs(2))
        for n in (6, 10, 14):
            h.put(small)
            h.put(_msgs(n))
        compacted = [{"role": "user", "content": f"compacted-{i}"} for i in range(14)]
        assert h.put(compacted) is True
        h.close()


# ══════════════════════════════════════════════════════════════
# A3 冷启动：无候选 → False（保守不误报，与改动前同语义）
# ══════════════════════════════════════════════════════════════

def test_first_turn_of_session_is_not_reported():
    with tempfile.TemporaryDirectory() as td:
        h = _Hub(td)
        assert h.put(_msgs(5)) is False
        h.close()


def test_check_prefix_changed_without_cache_is_false():
    with tempfile.TemporaryDirectory() as td:
        hub = MemoryHub(Path(td) / "rocksdb")
        hub.open()
        assert hub.check_prefix_changed("never/seen/before/", _msgs(3)) is False
        hub.close()


# ══════════════════════════════════════════════════════════════
# A4 槽位与去重
# ══════════════════════════════════════════════════════════════

def test_slots_are_capped_and_evict_oldest():
    with tempfile.TemporaryDirectory() as td:
        h = _Hub(td)
        for n in range(2, 20, 2):
            h.put(_msgs(n))
        slots = h.hub._session_last[h.prefix]
        assert len(slots) == h.hub._prefix_slots == 4
        h.close()


def test_identical_traffic_does_not_fill_all_slots():
    """🔴 同一路重复内容必须去重，否则它会占满全部槽位、把另一路挤掉

    ——那就退化回单槽了，只是退化得更隐蔽。
    """
    small = _msgs(3, "small")
    with tempfile.TemporaryDirectory() as td:
        h = _Hub(td)
        h.put(_msgs(2))
        for _ in range(10):
            h.put(small)
        slots = h.hub._session_last[h.prefix]
        assert len(slots) == 2, f"重复的小请求应只占一槽，实际 {list(slots)}"
        h.close()


def test_both_paths_keep_a_slot_under_long_interleaving():
    """长时间交替后，两路各自都还留着可匹配的候选。"""
    small = _msgs(3, "small")
    with tempfile.TemporaryDirectory() as td:
        h = _Hub(td)
        h.put(_msgs(2))
        for n in range(4, 30, 2):
            h.put(small)
            h.put(_msgs(n))
        # 两路都不报
        assert h.put(small) is False
        assert h.put(_msgs(30)) is False
        h.close()


# ══════════════════════════════════════════════════════════════
# 回滚通道：BLADEX_PREFIX_SLOTS=1 退回单槽行为
# ══════════════════════════════════════════════════════════════

def test_single_slot_reproduces_old_false_positive(monkeypatch):
    """槽数=1 时旧的假阳性重现 —— 证明这条回滚通道真的通到旧行为，

    也证明前面那些绿灯确实来自多槽而不是别的什么变化。
    """
    monkeypatch.setenv("BLADEX_PREFIX_SLOTS", "1")
    with tempfile.TemporaryDirectory() as td:
        h = _Hub(td)
        assert h.hub._prefix_slots == 1
        h.put(_msgs(2))
        h.put(_msgs(20))
        h.put(_msgs(3, "small"))          # 覆盖唯一的槽
        assert h.put(_msgs(22)) is True, "单槽下主会话被小请求带偏 = 旧的假阳性"
        h.close()


# ══════════════════════════════════════════════════════════════
# 两个入口共用同一判据（改前是两份重复实现）
# ══════════════════════════════════════════════════════════════

def test_two_entrypoints_agree():
    """`check_prefix_changed`（messages）与 `_detect_prefix_change`（Turn）
    对同一份输入必须给同一个答案。"""
    with tempfile.TemporaryDirectory() as td:
        h = _Hub(td)
        h.put(_msgs(2))
        h.put(_msgs(10))
        for candidate in (_msgs(12), _msgs(3, "small"),
                          [{"role": "user", "content": f"x-{i}"} for i in range(10)]):
            via_msgs = h.hub.check_prefix_changed(h.prefix, candidate)
            via_turn = h.hub._detect_prefix_change(_turn(h.identity, candidate))
            assert via_msgs == via_turn, f"两个入口不一致：{candidate[:1]}"
        h.close()
