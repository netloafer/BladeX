"""Memory Hub RocksDB 单元测试（T4）。

T1: key 改用 entry_id 作唯一后缀，重试不再覆盖。
"""

import tempfile
from pathlib import Path

from bladex_proxy.models import (
    AdminEventType,
    Identity,
    TombstoneSource,
    TombstoneTargetType,
    Turn,
    TurnStatus,
)
from bladex_proxy.storage.memory_hub import MemoryHub


def test_hub_put_get_roundtrip():
    """写进去读得出来。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = MemoryHub(Path(tmpdir) / "rocksdb")
        db.open()

        identity = Identity(user_id="user1", agent_id="codex", session_id="sess1", turn_index=0)
        turn = Turn(
            identity=identity,
            model="gpt-4o",
            request_messages=[{"role": "user", "content": "Hello"}],
            injected_memory="<bladex-memory>test</bladex-memory>",
            response_text="Hi there",
            status=TurnStatus.OK,
        )

        key = identity.storage_key("1719907200-0")
        db.put(key, turn)

        loaded = db.get(key)
        assert loaded is not None
        assert loaded.response_text == "Hi there"
        assert loaded.identity.user_id == "user1"
        assert loaded.status == TurnStatus.OK

        db.close()


def test_ledger_get_nonexistent():
    """不存在的 key 返回 None。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = MemoryHub(Path(tmpdir) / "rocksdb")
        db.open()
        assert db.get("nonexistent/key/0") is None
        db.close()


def test_ledger_scan_prefix():
    """按前缀扫描：同一会话的多个轮次顺序读出来。

    T1: entry_id 天然有序，scan_prefix 按会话前缀取回所有轮次。
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db = MemoryHub(Path(tmpdir) / "rocksdb")
        db.open()

        # 写 3 轮同一会话（不同 entry_id）+ 1 轮另一会话
        entry_ids = ["1719907200-0", "1719907201-0", "1719907202-0"]
        for i, eid in enumerate(entry_ids):
            identity = Identity(
                user_id="user1", agent_id="codex", session_id="sess1", turn_index=i
            )
            turn = Turn(
                identity=identity,
                response_text=f"reply-{i}",
                status=TurnStatus.OK,
            )
            db.put(identity.storage_key(eid), turn)

        identity2 = Identity(
            user_id="user1", agent_id="codex", session_id="sess2", turn_index=0
        )
        db.put(identity2.storage_key("1719907200-1"), Turn(
            identity=identity2, response_text="other-session"
        ))

        # 扫描 sess1
        results = list(db.scan_prefix("user1/codex/sess1/"))
        assert len(results) == 3
        # entry_id 有序 → 按时间排序
        for i, (key, turn) in enumerate(results):
            assert entry_ids[i] in key
            assert turn.response_text == f"reply-{i}"

        db.close()


def test_ledger_count():
    """计数正确。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = MemoryHub(Path(tmpdir) / "rocksdb")
        db.open()

        for i in range(5):
            identity = Identity(
                user_id="u", agent_id="a", session_id="s", turn_index=i
            )
            db.put(identity.storage_key(f"171990720{i}-0"), Turn(identity=identity))

        assert db.count() == 5
        db.close()


# ── T1 新增：追加式 key 不覆盖 ──

def test_ledger_key_no_overwrite_on_retry():
    """T1: 同一 Identity 连续两次入库（模拟重试）→ 两条记录，无覆盖。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = MemoryHub(Path(tmpdir) / "rocksdb")
        db.open()

        identity = Identity(
            user_id="user1", agent_id="codex", session_id="sess1", turn_index=2
        )

        # 第一次入库（重试前）
        turn1 = Turn(identity=identity, response_text="first attempt",
                     status=TurnStatus.FAILED, error="timeout")
        db.put(identity.storage_key("1719907200-0"), turn1)

        # 第二次入库（重试后，同 turn_index 但不同 entry_id）
        turn2 = Turn(identity=identity, response_text="second attempt",
                     status=TurnStatus.OK)
        db.put(identity.storage_key("1719907201-0"), turn2)

        # 两条都在
        results = list(db.scan_prefix(identity.session_prefix()))
        assert len(results) == 2

        # 各自的内容正确
        loaded1 = db.get(identity.storage_key("1719907200-0"))
        assert loaded1 is not None
        assert loaded1.response_text == "first attempt"
        assert loaded1.status == TurnStatus.FAILED

        loaded2 = db.get(identity.storage_key("1719907201-0"))
        assert loaded2 is not None
        assert loaded2.response_text == "second attempt"
        assert loaded2.status == TurnStatus.OK

        db.close()


def test_ledger_failed_turn_preserved():
    """T1: FAILED 轮次与其后成功轮次共存，FAILED 标记不被冲掉。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = MemoryHub(Path(tmpdir) / "rocksdb")
        db.open()

        identity = Identity(
            user_id="user1", agent_id="codex", session_id="sess1", turn_index=0
        )

        # FAILED 轮次
        failed_turn = Turn(identity=identity, response_text="",
                          status=TurnStatus.FAILED, error="upstream 500")
        db.put(identity.storage_key("1719907200-0"), failed_turn)

        # 成功轮次（同 turn_index，重试成功）
        ok_turn = Turn(identity=identity, response_text="Hello!",
                       status=TurnStatus.OK)
        db.put(identity.storage_key("1719907201-0"), ok_turn)

        # FAILED 仍在
        loaded_failed = db.get(identity.storage_key("1719907200-0"))
        assert loaded_failed is not None
        assert loaded_failed.status == TurnStatus.FAILED
        assert loaded_failed.error == "upstream 500"

        # 成功也在
        loaded_ok = db.get(identity.storage_key("1719907201-0"))
        assert loaded_ok is not None
        assert loaded_ok.status == TurnStatus.OK

        db.close()


# ── T4 新增：内容 hash 去重 ──

def test_ledger_dedup_storage_footprint():
    """T4: 100 轮会话入库后，Memory Hub 占用显著低于 O(n²)。

    对比：去重后 vs 假设不去重（每轮存全量）。
    验证：重复的 system 消息只存一份正文。
    """
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        db = MemoryHub(Path(tmpdir) / "rocksdb")
        db.open()

        identity = Identity(
            user_id="user1", agent_id="codex", session_id="sess1", turn_index=0
        )

        # 模拟 20 轮会话（用 20 轮验证趋势，不需要真跑 100 轮）
        system_msg = {"role": "system", "content": "You are a helpful assistant. " * 100}
        for i in range(20):
            msgs = [system_msg]
            for j in range(i + 1):
                msgs.append({"role": "user", "content": f"question {j}"})
                msgs.append({"role": "assistant", "content": f"answer {j}"})
            msgs.append({"role": "user", "content": f"latest question {i}"})

            turn = Turn(
                identity=identity,
                model="gpt-4o",
                request_messages=msgs,
                response_text=f"reply-{i}",
                status=TurnStatus.OK,
            )
            db.put(identity.storage_key(f"17199072{i:02d}-0"), turn)

        # 计算实际存储的 key 数量
        total_keys = sum(1 for _ in db.db.keys())
        # Turn 条目数
        turn_count = db.count()
        assert turn_count == 20

        # 内容存储条目数 = 不同消息内容数
        # system 消息只存 1 份（20 轮重复但 hash 相同）
        # 每轮的 user/assistant 消息各不相同，但跨轮重复的也只存一份
        # 第 j 轮的 "question j" / "answer j" 在后续轮次中重复出现
        content_keys = total_keys - turn_count

        # 如果不去重，20 轮的总消息实例数 = sum(1 + 2*(i+1) + 1 for i in range(20)) = 460
        # 去重后：1 system + 20 unique questions + 20 unique answers + 20 unique latest questions = 61
        # 所以 content_keys 应该远小于 460
        total_msg_instances = sum(1 + 2 * (i + 1) + 1 for i in range(20))
        assert content_keys < total_msg_instances * 0.2, (
            f"Expected <{total_msg_instances * 0.2:.0f} content keys (20% of {total_msg_instances}), "
            f"got {content_keys}"
        )

        db.close()


def test_ledger_reconstruct_full_messages():
    """T4: 从 Memory Hub 能完整重建任一轮的原始 messages（去重不丢信息）。"""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        db = MemoryHub(Path(tmpdir) / "rocksdb")
        db.open()

        identity = Identity(
            user_id="user1", agent_id="codex", session_id="sess1", turn_index=0
        )

        # 写 3 轮，每轮都有重复的 system 消息 + 增量新消息
        original_turns = []
        for i in range(3):
            msgs = [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": f"question {i}"},
            ]
            turn = Turn(
                identity=identity,
                model="gpt-4o",
                request_messages=msgs,
                response_text=f"reply-{i}",
                status=TurnStatus.OK,
            )
            original_turns.append((msgs, turn))
            db.put(identity.storage_key(f"171990720{i}-0"), turn)

        # 读回并验证完整重建
        for i, (original_msgs, _) in enumerate(original_turns):
            loaded = db.get(identity.storage_key(f"171990720{i}-0"))
            assert loaded is not None

            # request_messages 完整还原（无 content_ref 残留）
            for msg in loaded.request_messages:
                assert "content_ref" not in msg, f"content_ref not resolved in turn {i}"
                assert "content" in msg

            # 内容逐条匹配
            assert len(loaded.request_messages) == len(original_msgs)
            for orig, loaded_msg in zip(original_msgs, loaded.request_messages, strict=True):
                assert orig["role"] == loaded_msg["role"]
                assert orig["content"] == loaded_msg["content"]

        db.close()


def test_ledger_dedup_multimodal_content():
    """T4: 多模态 list content 也正确去重和还原。"""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        db = MemoryHub(Path(tmpdir) / "rocksdb")
        db.open()

        identity = Identity(
            user_id="user1", agent_id="codex", session_id="sess1", turn_index=0
        )

        multimodal_content = [
            {"type": "text", "text": "Describe this image"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}},
        ]

        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": multimodal_content},
        ]
        turn = Turn(
            identity=identity,
            request_messages=msgs,
            response_text="It's a cat",
            status=TurnStatus.OK,
        )
        db.put(identity.storage_key("1719907200-0"), turn)

        loaded = db.get(identity.storage_key("1719907200-0"))
        assert loaded is not None
        assert len(loaded.request_messages) == 2
        assert loaded.request_messages[1]["content"] == multimodal_content

        db.close()


def test_ledger_prefix_hash_set():
    """T4: 存入的 Turn 带 prefix_hash 字段（用于校验/重放）。"""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        db = MemoryHub(Path(tmpdir) / "rocksdb")
        db.open()

        identity = Identity(
            user_id="user1", agent_id="codex", session_id="sess1", turn_index=0
        )
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        turn = Turn(identity=identity, request_messages=msgs, status=TurnStatus.OK)
        db.put(identity.storage_key("1719907200-0"), turn)

        loaded = db.get(identity.storage_key("1719907200-0"))
        assert loaded is not None
        assert loaded.prefix_hash != ""
        assert len(loaded.prefix_hash) == 16

        db.close()


# ── T2: prefix_changed 测试 ──

def test_ledger_prefix_change_detected():
    """同会话正常追加 → prefix_changed=False；压缩/重建 → prefix_changed=True。"""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        ledger = MemoryHub(Path(tmpdir) / "rocksdb")
        ledger.open()

        identity = Identity(
            user_id="u1", agent_id="a1", session_id="s1", turn_index=0
        )

        # 第一轮：正常
        turn1 = Turn(
            identity=identity, model="m",
            request_messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ],
            response_text="Hi",
            status=TurnStatus.OK,
        )
        ledger.put("u1/a1/s1/100-0", turn1)
        loaded1 = ledger.get("u1/a1/s1/100-0")
        assert loaded1 is not None
        assert loaded1.prefix_changed is False

        # 第二轮：正常追加（前缀延续）
        turn2 = Turn(
            identity=identity, model="m",
            request_messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
                {"role": "user", "content": "How are you?"},
            ],
            response_text="Good",
            status=TurnStatus.OK,
        )
        ledger.put("u1/a1/s1/101-0", turn2)
        loaded2 = ledger.get("u1/a1/s1/101-0")
        assert loaded2 is not None
        assert loaded2.prefix_changed is False

        # 第三轮：前缀变了（压缩了历史，system 变了）
        turn3 = Turn(
            identity=identity, model="m",
            request_messages=[
                {"role": "system", "content": "You are a different system prompt."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "Good"},
                {"role": "user", "content": "Bye"},
            ],
            response_text="See you",
            status=TurnStatus.OK,
        )
        ledger.put("u1/a1/s1/102-0", turn3)
        loaded3 = ledger.get("u1/a1/s1/102-0")
        assert loaded3 is not None
        assert loaded3.prefix_changed is True

        ledger.close()


def test_tombstone_hides_turn_from_scan():
    """T1: 删除某 turn 后，Memory Hub 原条目仍在但 scan_prefix 不再返回。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = MemoryHub(Path(tmpdir) / "rocksdb")
        db.open()

        identity = Identity(
            user_id="user1", agent_id="codex", session_id="sess1", turn_index=0
        )
        turn = Turn(identity=identity, response_text="to be deleted", status=TurnStatus.OK)
        key = identity.storage_key("1719907200-0")
        db.put(key, turn)

        # 删除前：scan 能找到
        results = list(db.scan_prefix(identity.session_prefix()))
        assert len(results) == 1

        # 写墓碑
        tombstone_key = db.append_tombstone(
            target_type=TombstoneTargetType.TURN,
            target_key=key,
            source=TombstoneSource.USER,
        )
        assert tombstone_key.startswith("tombstone/")

        # 删除后：scan 不再返回该 turn
        results = list(db.scan_prefix(identity.session_prefix()))
        assert len(results) == 0

        # 原条目仍在 Memory Hub（get 能取到）
        loaded = db.get(key)
        assert loaded is not None
        assert loaded.response_text == "to be deleted"

        # 墓碑可扫描到
        tombstones = list(db.scan_tombstones())
        assert len(tombstones) == 1
        _, t = tombstones[0]
        assert t.target_type == TombstoneTargetType.TURN
        assert t.target_key == key
        assert t.source == TombstoneSource.USER

        db.close()


def test_tombstone_only_hides_targeted_turn():
    """T1: 墓碑只隐藏目标 turn，同会话其它 turn 不受影响。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = MemoryHub(Path(tmpdir) / "rocksdb")
        db.open()

        identity = Identity(
            user_id="user1", agent_id="codex", session_id="sess1", turn_index=0
        )

        # 写 3 轮
        keys = []
        for i in range(3):
            turn = Turn(identity=identity, response_text=f"reply-{i}", status=TurnStatus.OK)
            k = identity.storage_key(f"171990720{i}-0")
            db.put(k, turn)
            keys.append(k)

        # 删第 2 轮
        db.append_tombstone(TombstoneTargetType.TURN, keys[1])

        results = list(db.scan_prefix(identity.session_prefix()))
        assert len(results) == 2
        # 第 0 和第 2 轮仍在
        remaining_keys = [k for k, _ in results]
        assert keys[0] in remaining_keys
        assert keys[1] not in remaining_keys
        assert keys[2] in remaining_keys

        db.close()


def test_admin_event_journal_ordered_replay():
    """T1: 管理事件按写入顺序可完整重放。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = MemoryHub(Path(tmpdir) / "rocksdb")
        db.open()

        # 写一系列管理事件
        events_to_write = [
            (AdminEventType.MATTER_CREATE, "matter-1", ""),
            (AdminEventType.MATTER_ASSIGN, "matter-1", "user1/codex/sess1/100-0"),
            (AdminEventType.MATTER_RENAME, "matter-1", "", {"title": "新标题"}),
            (AdminEventType.MATTER_CLOSE, "matter-1", ""),
            (AdminEventType.MATTER_CREATE, "matter-2", ""),
            (AdminEventType.MATTER_MERGE, "matter-2", "matter-1", {"source": "matter-1"}),
        ]

        written_keys = []
        for event_type, matter_id, target_key, *payload in events_to_write:
            kw = payload[0] if payload else {}
            k = db.append_admin_event(event_type, matter_id, target_key, **kw)
            written_keys.append(k)

        # 扫描并验证顺序
        scanned = list(db.scan_admin_events())
        assert len(scanned) == 6

        # 验证按 entry_id 有序
        scanned_keys = [k for k, _ in scanned]
        assert scanned_keys == sorted(scanned_keys)

        # 验证内容
        for i, (_, event) in enumerate(scanned):
            expected_type, expected_matter, expected_target = events_to_write[i][:3]
            assert event.event_type == expected_type
            assert event.matter_id == expected_matter
            assert event.target_key == expected_target

        # 验证 payload
        assert scanned[2][1].payload == {"title": "新标题"}
        assert scanned[5][1].payload == {"source": "matter-1"}

        db.close()


def test_is_tombstoned():
    """T1: is_tombstoned 正确反映墓碑状态。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = MemoryHub(Path(tmpdir) / "rocksdb")
        db.open()

        key = "user1/codex/sess1/100-0"
        assert not db.is_tombstoned(key)

        db.append_tombstone(TombstoneTargetType.TURN, key)
        assert db.is_tombstoned(key)

        # 缓存失效后仍正确
        db.append_tombstone(TombstoneTargetType.TURN, "user1/codex/sess1/101-0")
        assert db.is_tombstoned(key)
        assert db.is_tombstoned("user1/codex/sess1/101-0")

        db.close()


def test_count_excludes_tombstones_and_admin_events():
    """T1: count() 不计墓碑记录与管理事件本身。

    MQ-R5（2026-08-17）**在此基础上收紧**：被墓碑覆盖的 turn 也不计。
    原来那句 `count() == 2`（墓碑掉一条仍数 2）是当时的正确行为，
    但它让 `/admin/status` 的 `lag_turns = count() − consumed_count()` 多出一个
    永不消失的常数——墓碑的 turn 永远不会被消费（`scan_meta` 按设计跳过），
    08-16 迁移掉的 500 条残渣于是让系统恒报"落后 500 轮"。
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db = MemoryHub(Path(tmpdir) / "rocksdb")
        db.open()

        identity = Identity(user_id="u", agent_id="a", session_id="s", turn_index=0)
        db.put(identity.storage_key("100-0"), Turn(identity=identity))
        db.put(identity.storage_key("101-0"), Turn(identity=identity))
        assert db.count() == 2, "阴性对照：还没墓碑时两条都算"

        db.append_tombstone(TombstoneTargetType.TURN, identity.storage_key("100-0"))
        db.append_admin_event(AdminEventType.MATTER_CREATE, "m1")

        # 墓碑记录 / 管理事件本身不计（T1 原意），被墓碑覆盖的 turn 也不计（MQ-R5）
        assert db.count() == 1
        assert len(list(db.scan_meta())) == 1, "与消费侧口径必须一致"

        db.close()


# ── G6/G8/G9 测试 ──

def test_g6_compact_tombstoned_deletes_original():
    """G6: compact_tombstoned 物理擦除被墓碑标记的 Turn 原文。"""
    import tempfile
    from pathlib import Path

    from bladex_proxy.models import Identity, TombstoneTargetType, Turn, TurnStatus

    with tempfile.TemporaryDirectory() as tmpdir:
        ledger = MemoryHub(Path(tmpdir) / "rocksdb")
        ledger.open()

        identity = Identity(user_id="u1", agent_id="a1", session_id="s1", turn_index=0)
        turn = Turn(
            identity=identity, model="test",
            request_messages=[{"role": "user", "content": "hello world"}],
            response_text="hi", status=TurnStatus.OK,
        )
        key = identity.storage_key("1719907200-0")
        ledger.put(key, turn)

        # 写墓碑
        ledger.append_tombstone(TombstoneTargetType.TURN, key)

        # 原文还在（scan_prefix 跳过墓碑覆盖的）
        assert ledger.get(key) is not None  # get 不检查墓碑

        # compact
        removed = ledger.compact_tombstoned()
        assert removed == 1

        # 原文物理删除
        assert ledger.get(key) is None

        # 墓碑记录还在
        tombstones = list(ledger.scan_tombstones())
        assert len(tombstones) == 1

        ledger.close()


def test_g8_session_cache_recovered_on_open():
    """G8: worker 重启后 _session_last 缓存从 RocksDB 恢复。"""
    import tempfile
    from pathlib import Path

    from bladex_proxy.models import Identity, Turn, TurnStatus

    with tempfile.TemporaryDirectory() as tmpdir:
        # 第一次打开，写入两条 turn
        ledger = MemoryHub(Path(tmpdir) / "rocksdb")
        ledger.open()
        identity = Identity(user_id="u1", agent_id="a1", session_id="s1", turn_index=0)
        turn1 = Turn(
            identity=identity, model="test",
            request_messages=[{"role": "user", "content": "msg1"}],
            response_text="r1", status=TurnStatus.OK,
        )
        turn2 = Turn(
            identity=identity, model="test",
            request_messages=[{"role": "user", "content": "msg1"}, {"role": "user", "content": "msg2"}],
            response_text="r2", status=TurnStatus.OK,
        )
        ledger.put(identity.storage_key("1719907200-0"), turn1)
        ledger.put(identity.storage_key("1719907201-0"), turn2)
        session_prefix = identity.session_prefix()
        assert session_prefix in ledger._session_last
        ledger.close()

        # 重新打开 -> 缓存应恢复
        p3b = MemoryHub(Path(tmpdir) / "rocksdb")
        p3b.open()
        assert session_prefix in p3b._session_last
        # V-C1（MQ-S47）：`_session_last` 的值从单个 tuple 改成了候选 deque
        # （同一 session 下多路流量各占一槽）。恢复语义不变——**最后一条**
        # turn 仍是最新候选，所以这里取 slots[-1]；改的是容器不是行为。
        slots = p3b._session_last[session_prefix]
        assert len(slots) == 2, "两条 turn 应各留一条候选（改前只留最后一条）"
        prefix_hash, msg_count = slots[-1]
        assert msg_count == 2  # turn2 有 2 条消息
        assert prefix_hash != ""

        # 恢复后 check_prefix_changed 能正常工作
        assert not p3b.check_prefix_changed(session_prefix, [
            {"role": "user", "content": "msg1"},
            {"role": "user", "content": "msg2"},
            {"role": "user", "content": "msg3"},
        ])  # 尾部追加 -> 前缀不变

        p3b.close()
