"""Memory Hub RocksDB 封装 — 完整总账，唯一真相源（T4，ADR-0009 §6）。

T1: key = user/agent/session/{entry_id}（追加式，不再覆盖）。
T4: 内容 hash 去重 — 同一 session 重复的 system/历史消息按 hash 存一份正文，
Turn 里存引用（content_ref），读出时还原。仅本轮新增消息存正文。
"""

from __future__ import annotations

import hashlib
from collections import deque
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import msgpack
import structlog
from bladex_core.distillation import DistillFact, DistillOutput, MatterProposal
from bladex_core.flags import flag_number
from rocksdict import Rdict

from bladex_proxy.models import (
    INNER_LOOP_AUX_SOURCE,
    AdminEvent,
    AdminEventType,
    DistillRecord,
    JudgmentRecord,
    Tombstone,
    TombstoneSource,
    TombstoneTargetType,
    Turn,
)

logger = structlog.get_logger()

# 内容存储 key 前缀（与 Turn key 区分）
_CONTENT_PREFIX = "__msg__/"


# ADR-0012: 墓碑与管理事件的 Memory Hub key 前缀（追加式记录，与 Turn key 区分）
_TOMBSTONE_PREFIX = "tombstone/"
_ADMIN_EVENT_PREFIX = "admin_event/"
# ADR-0018 §4.1: 派生台账前缀（append-only，LLM 判定输出的原始形态，不可确定性重算）
_DISTILL_PREFIX = "distill/"
_JUDGMENT_PREFIX = "judgment/"
_ANNOTATION_PREFIX = "annotation/"


class MemoryHub:
    """RocksDB 封装：put/get/scan + 内容 hash 去重。"""

    def __init__(self, path: str | Path, *, read_only: bool = False, secondary: bool = False,
                 secondary_dir: str | None = None) -> None:
        self._path = Path(path)
        self._read_only = read_only
        self._secondary = secondary
        # V-F2：secondary 副本目录可指定——flash 维护进程与 consolidator 各用
        # 各的（两个 secondary 共用一个副本目录会互踩 MANIFEST）。缺省 = 现状。
        self._secondary_dir = secondary_dir
        if not read_only and not secondary:
            self._path.mkdir(parents=True, exist_ok=True)
        self._db: Rdict | None = None
        # 进程内缓存：每个会话最近 N 条 Turn 的 (prefix_hash, msg_count) 候选。
        # 避免 _detect_prefix_change 每次写入全扫 RocksDB（O(n) → O(N)，N≤4）。
        #
        # V-C1 / MQ-S47：**曾经是单槽**，而 agent 会在同一个 session_prefix 下
        # 混跑多路流量（claude-code：主会话 + 3 条消息的小请求交替），
        # 单槽被两路互相覆盖 ⇒ 两跑 54 次告警里 53 次假阳性、真阳性 0。
        # 多槽 + 遍历匹配后：任一候选对得上就是"前缀未变"，全对不上才是真压缩。
        # N 的默认值与理由在 `bladex_core.flags`（`BLADEX_PREFIX_SLOTS`）。
        self._prefix_slots: int = max(1, int(flag_number("BLADEX_PREFIX_SLOTS")))
        self._session_last: dict[str, deque[tuple[str, int]]] = {}
        # ADR-0012: 墓碑缓存（target_key 集合），scan_prefix 用，写墓碑时失效
        self._tombstone_cache: set[str] | None = None

    def open(self) -> None:
        """打开（或重新打开）RocksDB。

        - read_only=True：只读**点时快照**（不取 LOCK，可与 proxy 共存），但看不到 proxy
          后续写入--consolidator 增量勿用此模式追新（ADR-0020 T1 根因）。
        - secondary=True：secondary 模式（不取 LOCK + try_catch_up_with_primary 追新），
          consolidator 增量用此模式看 proxy 新写入。
        - 两者皆否：read_write（取 LOCK，proxy / full rebuild 用）。
        """
        if self._db is not None:
            return
        import rocksdict
        if self._secondary:
            opts = rocksdict.Options()
            secondary_path = self._secondary_dir or (str(self._path) + "_secondary")
            self._db = rocksdict.Rdict(
                str(self._path), opts, None,
                rocksdict.AccessType.secondary(secondary_path),
            )
        elif self._read_only:
            opts = rocksdict.Options()
            self._db = rocksdict.Rdict(
                str(self._path), opts, None,
                rocksdict.AccessType.read_only(),
            )
        else:
            self._db = Rdict(str(self._path))
        # G8: 从 RocksDB 恢复 _session_last 缓存（worker 重启后不丢 prefix 检测）
        if not self._read_only and not self._secondary:
            self._recover_session_cache()
        logger.info("hub_rocksdb_opened", path=str(self._path),
                     read_only=self._read_only, secondary=self._secondary)

    def catch_up(self) -> None:
        """secondary 模式追新主进程写入（ADR-0020 T1）。

        consolidator 增量每轮 rebuild 前调一次，让 secondary 视图看到 proxy 新写入的 Turn。
        非 secondary 模式空操作。
        """
        if self._db is None or not self._secondary:
            return
        try:
            self._db.try_catch_up_with_primary()
        except Exception as e:
            logger.debug("hub_catch_up_failed", error=str(e))

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None
            logger.info("hub_rocksdb_closed")

    def _recover_session_cache(self) -> None:
        """从 RocksDB 恢复 _session_last 缓存（G8）。

        worker 重启后内存缓存清空，_detect_prefix_change 会误判首条为
        "会话首条"（prefix_changed=False）。本方法在 open() 时扫描全量
        Turn key，取每个 session_prefix 最后一条的 prefix_hash + msg_count。

        key 按 entry_id 天然有序，正向遍历时后者覆盖前者 -> 最后一条胜出。
        """
        for key_raw in self.db.keys():
            key_str = key_raw.decode() if isinstance(key_raw, bytes) else str(key_raw)
            if _is_internal_key(key_str):
                continue
            # key 格式: user/agent/session/entry_id -> session_prefix = user/agent/session/
            parts = key_str.rsplit("/", 1)
            if len(parts) < 2:
                continue
            session_prefix = parts[0] + "/"
            raw = self.db.get(key_raw)
            if raw is None:
                continue
            try:
                data = msgpack.unpackb(raw, raw=False)
                # MQ-P16：与 `put()` 同一判据——自产的内循环轮不进候选池，否则会话以
                # 内循环收尾时重启后主会话首请求必假阳性（`check_prefix_changed` 的
                # 槽就来自这里）。
                if data.get("aux_source") == INNER_LOOP_AUX_SOURCE:
                    continue
                prefix_hash = data.get("prefix_hash", "")
                msg_count = len(data.get("request_messages", []))
                if prefix_hash:
                    # V-C1：正向遍历 + append ⇒ 每个会话自然留下最后 N 条候选
                    # （改前是覆盖，只留最后一条）。多路流量的恢复面因此也变对：
                    # worker 重启后第一轮不再拿"碰巧最后写入的那一路"当唯一基准。
                    self._remember_prefix(session_prefix, prefix_hash, msg_count)
            except Exception:
                continue
        if self._session_last:
            logger.info("hub_session_cache_recovered",
                        sessions=len(self._session_last))

    @property
    def db(self) -> Rdict:
        if self._db is None:
            self.open()
        assert self._db is not None
        return self._db

    def put(self, key: str, turn: Turn) -> None:
        """写入一个完整 Turn（含 T4 内容去重 + T2 前缀变化检测）。

        去重逻辑：对 request_messages 里每条消息的 content 算 hash，
        已存过的 content 只存引用（content_ref），新增的存正文 + hash。
        前缀变化检测（T2）：对比本会话上一条 Turn，若前缀非延续 → 标记。
        """
        deduped_turn = self._dedup_messages(turn)
        # MQ-P16（2026-09-02）：BladeX **自产**的内循环 aux 轮不进"agent 压缩了没有"
        # 的判据面。它的 request_messages 是我们拼的 [system 标记, assistant, tool…]
        # （标记含轮序/trigger_ts ⇒ 每轮前缀都不同、msg_count 2–3），不是 agent 历史：
        # 参与检测 = 每条必报 `hub_prefix_changed`（V-C1 刚清零的 warning 通道每次
        # 内循环回来 rounds 条），且 rounds≥3 就把主会话的 4 槽 LRU 挤空 ⇒ 重启后主
        # 会话首请求假阳性、assembly L2 分支变化——MQ-S47 续的"短请求挤占槽位"，
        # 这次是我们自己造的短请求。**只跳这一种**：其它 aux 轮（CC 子调用等）是
        # agent 真发来的历史，改它们归 MQ-S47 续（0.3.0）。写入照旧。
        _self_made = turn.aux_source == INNER_LOOP_AUX_SOURCE
        deduped_turn.prefix_changed = (False if _self_made
                                       else self._detect_prefix_change(turn))
        data = deduped_turn.model_dump(mode="json")
        value = msgpack.packb(data, use_bin_type=True)
        self.db[key.encode()] = value
        if _self_made:
            logger.debug("hub_put", key=key, status=turn.status.value,
                         msg_count=len(turn.request_messages), inner_loop=True)
            return
        if deduped_turn.prefix_changed:
            # V-C1：带上当时的候选池状态。**单轮内分不清**"一路新流量首次出现"
            # 与"agent 真压缩了历史"——两者都是"所有候选都对不上"，而唯一能
            # 区分它们的判据（消息数骤降）真压缩也满足，用了就是拿假阳性换假阴性。
            # 分不清就不硬判，改为把判据材料打出来：`known_slots` 是当时池里的
            # 候选数，`slot_counts` 是各候选的 msg_count。首见新路的形态是
            # "本轮 msg_count 与任何候选都不衔接"，真压缩则通常紧跟在一条
            # 长度相近的候选之后 —— 事后看日志能分，实时不能。
            slots = self._session_last.get(turn.identity.session_prefix())
            logger.warning("hub_prefix_changed", key=key,
                           session=turn.identity.session_prefix(),
                           msg_count=len(turn.request_messages),
                           known_slots=len(slots) if slots else 0,
                           slot_counts=[c for _h, c in (slots or ())])
        # 缓存本会话最近 N 条 Turn 的前缀信息（供下次 _detect_prefix_change 遍历匹配）
        self._remember_prefix(
            turn.identity.session_prefix(),
            deduped_turn.prefix_hash,
            len(turn.request_messages),
        )
        logger.debug("hub_put", key=key, status=turn.status.value,
                     msg_count=len(turn.request_messages),
                     prefix_changed=deduped_turn.prefix_changed)

    def get(self, key: str) -> Turn | None:
        """读取一个 Turn（自动还原 content_ref → 完整 content）。"""
        raw = self.db.get(key.encode())
        if raw is None:
            return None
        data = msgpack.unpackb(raw, raw=False)
        turn = Turn.model_validate(data)
        self._resolve_messages(turn)
        return turn

    def scan_meta(self, prefix: str = "") -> Iterator[tuple[str, str]]:
        """惰性扫描：只返回 (key, ts_iso)，不做 Pydantic 校验、不解析 content_ref。

        性能（2026-07-28 实测，6891 turns / 1.2GB）：本方法约 2.5 秒；
        `scan_prefix` 全量路径（model_validate + _resolve_messages 逐条点查
        __msg__ 内容池）> 5 分钟。consolidator 每轮只挑 ≤max_turns 条重放，
        却要为全库付完整反序列化代价 —— 用本方法先筛（已消费 / 时间窗 / agent），
        命中的少数 key 再走 `get()` 拿完整 Turn。

        ts 取顶层字段原样字符串（写入即 ISO），缺失返回空串。
        与 scan_prefix 一致跳过内部 key 与墓碑覆盖项。
        """
        tombstoned = self._tombstoned_keys()
        for key_raw, value_raw in self.db.items():
            key_str = key_raw.decode() if isinstance(key_raw, bytes) else str(key_raw)
            if _is_internal_key(key_str):
                continue
            if not key_str.startswith(prefix):
                continue
            if key_str in tombstoned:
                continue
            try:
                data = msgpack.unpackb(value_raw, raw=False)
            except Exception as e:  # noqa: BLE001 -- 脏值不该中断整轮扫描
                logger.warning("hub_scan_meta_decode_failed", key=key_str, error=str(e))
                continue
            yield key_str, str(data.get("ts", "") or "")

    def scan_agent_ids(self) -> dict[str, tuple[str, int]]:
        """扫 key 里的 agent 段，返回 ``{agent_id: (代表 key, 轮次数)}``（G11.11）。

        **只解 key，一个 value 都不碰**——`agent_id` 本来就是 Memory Hub key 的
        第二段（`principal/agent/session/entry_id`），拿它不需要反序列化。比
        :meth:`scan_meta` 还便宜一档（那个至少要 msgpack.unpackb 顶层）。

        为什么需要它：dashboard 的 agent 管理页此前只读进程内登记，**proxy 一重启
        就空**——管理界面大多数时候没有东西可管，等于没有管理接口。更要命的是已经
        发生过的误署名（08-18 那批）在进程里根本不存在，没有任何入口去处理它。
        Memory Hub 是追加式总账，从它派生这个列表天然重启可存活。

        代表 key 供调用方按需 `get()` 一条完整 Turn 取判据串与脱敏 header 快照
        （每个 agent 只取一条，不为列表页付全量反序列化的代价）。

        :returns: agent_id → (该 agent 的**最后一条** key, 轮次数)。
        """
        found: dict[str, tuple[str, int]] = {}
        for key_raw in self.db.keys():
            key_str = key_raw.decode() if isinstance(key_raw, bytes) else str(key_raw)
            if _is_internal_key(key_str):
                continue
            parts = key_str.split("/")
            if len(parts) < 4:
                continue
            agent_id = parts[1]
            if not agent_id:
                continue
            _prev_key, count = found.get(agent_id, ("", 0))
            # 取最后一条当代表：越新的 header 快照越接近该 agent 当前的接入形态
            found[agent_id] = (key_str, count + 1)
        return found

    def get_tools(self, tools_hash: str) -> list[dict] | None:
        """按 hash 取回 tools schema（`store_tools` 的读侧，G11.11 详情页用）。

        Turn 只存 `request_params.tools_hash`，schema 全文在 `__msg__` 内容池里按
        hash 去重。不取回来，详情页就看不到这个 agent 有哪些工具——而工具名是
        辨识度第二高的信号（仅次于 system prompt）。
        """
        if not tools_hash:
            return None
        raw = self.db.get(f"{_CONTENT_PREFIX}{tools_hash}".encode())
        if raw is None:
            return None
        try:
            data = msgpack.unpackb(raw, raw=False)
        except Exception as e:  # noqa: BLE001 —— 脏值不该毁掉详情页
            logger.warning("hub_get_tools_decode_failed", hash=tools_hash, error=str(e))
            return None
        return data if isinstance(data, list) else None

    def keys_for_agent(self, agent_id: str) -> list[str]:
        """某个 agent 的全部 Memory Hub key（升序）。同样**只解 key**。

        供 agent 详情页取样用：先廉价拿到 key 列表，再只 `get()` 最后几条看内容，
        不为一个详情页付全库反序列化的代价。

        跨 principal：不限身份（没带 API Key 的客户端落 `ANONYMOUS_USER_ID`，
        只查 `local` 会让它在详情页里显示成"没有轮次"）。
        """
        out: list[str] = []
        for key_raw in self.db.keys():
            key_str = key_raw.decode() if isinstance(key_raw, bytes) else str(key_raw)
            if _is_internal_key(key_str):
                continue
            parts = key_str.split("/")
            if len(parts) >= 4 and parts[1] == agent_id:
                out.append(key_str)
        out.sort()
        return out

    def scan_prefix(self, prefix: str) -> Iterator[tuple[str, Turn]]:
        """按 key 前缀扫描（如 "user1/agent1/session1/" 返回该会话所有轮次）。

        T1: entry_id 天然有序，返回结果按时间排序。
        ADR-0012 §3.6: 跳过被墓碑覆盖的条目（原文仍在 Memory Hub 但不返回）。

        全量反序列化路径（校验 + content_ref 还原）。只需要 key/ts 做筛选时
        用 `scan_meta` —— 差约两个数量级（见其 docstring 实测）。
        """
        tombstoned = self._tombstoned_keys()
        for key_raw, value_raw in self.db.items():
            key_str = key_raw.decode() if isinstance(key_raw, bytes) else str(key_raw)
            # 跳过非 Turn key：内容存储 / meta / 墓碑 / 管理事件 / 派生台账
            if _is_internal_key(key_str):
                continue
            if key_str.startswith(prefix):
                if key_str in tombstoned:
                    continue
                data = msgpack.unpackb(value_raw, raw=False)
                turn = Turn.model_validate(data)
                self._resolve_messages(turn)
                yield key_str, turn

    def count(self) -> int:
        """可见 Turn 条目数（不含内容存储 / meta / 墓碑 / 管理事件，也**不含被墓碑覆盖的 turn**）。

        MQ-R5（2026-08-17）：本方法必须与 `scan_meta` / `scan_prefix` 共用同一套跳过规则
        （`_tombstoned_keys()`）——**同一个「有多少轮」的问题只允许一个答案**。
        此前只过 `_is_internal_key`，于是墓碑覆盖的 turn 计入被减数，而消费侧
        （`scan_meta` 按设计跳过墓碑）永远不会消费它们，
        `lag_turns = count() − consumed_count()` 成了一个**永不消失的常数**：
        08-16 ADR-0021 迁移掉的 500 条残渣，让 status 从那天起恒报"落后 500 轮"，
        而 consolidator 早已追平在空转。
        """
        tombstoned = self._tombstoned_keys()
        return sum(
            1 for k in self.db.keys()
            if not _is_internal_key(k)
            and (k.decode() if isinstance(k, bytes) else str(k)) not in tombstoned
        )

    # ── T2: 前缀变化检测 ──

    @staticmethod
    def _prefix_hash_of(messages: list[dict], n: int) -> str:
        """前 n 条消息的前缀 hash（与 `_dedup_messages` 的算法同源）。"""
        prefix_parts: list[str] = []
        for msg in messages[:n]:
            role = msg.get("role", "")
            content = msg.get("content")
            if content is None:
                prefix_parts.append(f"{role}:null")
            else:
                prefix_parts.append(f"{role}:{_hash_content(content)}")
        return hashlib.sha256("|".join(prefix_parts).encode()).hexdigest()[:16]

    def _prefix_changed_against_slots(
        self, session_prefix: str, messages: list[dict],
    ) -> bool:
        """本轮前缀是否与该会话的**任何**已知候选都对不上（V-C1 / MQ-S47）。

        返回 True 才是"agent 真的压缩/重建了历史"。

        为什么是"任一命中即算未变"而不是"跟最后一条比"：
        agent 在同一个 `session_prefix` 下混跑多路流量（claude-code 实测两路：
        主会话 msg_count 单调递增，与 3 条消息的小请求交替）。只留一个槽时两路
        互相覆盖，双方都拿对方当基准 ⇒ 双向假阳性（两跑 54 次告警，53 次如此，
        真阳性 0）。多槽之后：
          - 主会话 append-only ⇒ 必然命中自己那条历史槽
          - 小请求第二次来时命中它**自己**建立的槽
          - 真压缩 ⇒ 所有候选都对不上 ⇒ 报 True，且是真阳性

        无候选（会话首条 / worker 刚起）→ False：保守不误报，
        与改动前 `cached is None → False` 的语义一致。
        """
        slots = self._session_last.get(session_prefix)
        if not slots:
            return False  # 会话首条，无前缀可比

        for prev_hash, prev_count in slots:
            if not prev_hash:
                # 该候选无 prefix_hash，无法比较 —— 保守当作"没有异议"，
                # 与改动前单槽时 `not prev_prefix_hash → False` 同语义。
                return False
            if len(messages) < prev_count:
                continue  # 比这条候选短，换下一条候选比
            if self._prefix_hash_of(messages, prev_count) == prev_hash:
                return False  # 命中任一候选 = 前缀是它的延续
        return True

    def _remember_prefix(
        self, session_prefix: str, prefix_hash: str, msg_count: int,
    ) -> None:
        """把本轮前缀记进候选池（LRU，`_prefix_slots` 条封顶）。

        🔴 **先去重再追加**：同一路流量若每轮内容一致（如固定的 3 条消息小请求），
        不去重会让同一个 (hash, count) 占满全部槽位、把另一路挤出去——
        那就退化回单槽了，只是退化得更隐蔽。
        """
        slots = self._session_last.setdefault(
            session_prefix, deque(maxlen=self._prefix_slots))
        entry = (prefix_hash, msg_count)
        if entry in slots:
            slots.remove(entry)
        slots.append(entry)

    def _detect_prefix_change(self, turn: Turn) -> bool:
        """检测本轮前缀是否与该会话的已知候选都对不上（T2，ADR-0009 §6）。

        正常追加：本轮 messages 的前 N 条与某条候选完全一致（尾部超集）。
        前缀变化：agent 压缩/重建历史 → 所有候选都对不上 → 置 True。

        性能：进程内缓存，O(N) 且 N ≤ `_prefix_slots`（默认 4），不扫 RocksDB。
        """
        return self._prefix_changed_against_slots(
            turn.identity.session_prefix(), turn.request_messages)

    def check_prefix_changed(
        self,
        session_prefix: str,
        messages: list[dict],
    ) -> bool:
        """检查 incoming messages 的前缀是否与该会话的已知候选都对不上（ADR-0016 §3.5）。

        供装配 L2 冲突检测与 cache 状态判断用：agent 压缩过上下文 -> prefix 变了
        -> 不二次压缩。进程内缓存 O(N) 查（N ≤ `_prefix_slots`），不读 RocksDB。
        缓存未命中返回 False（保守不跳过）。

        V-C1：与 `_detect_prefix_change` **共用同一个匹配核心**——
        改前两处各写一遍同样的算法，是"同一判据两份实现"的形状
        （刚性原则 12 的近亲）。
        """
        return self._prefix_changed_against_slots(session_prefix, messages)

    # ── T4: 内容去重 / 还原 ──

    def _dedup_messages(self, turn: Turn) -> Turn:
        """对 Turn 的 request_messages 做内容 hash 去重。

        返回一个新的 Turn（不改原对象），其中重复的 content 替换为 content_ref。
        同时计算 prefix_hash 用于校验。
        """
        import copy

        deduped = copy.copy(turn)
        deduped.request_messages = []
        prefix_parts: list[str] = []

        for msg in turn.request_messages:
            msg_copy = dict(msg)
            role = msg_copy.get("role", "")
            content = msg_copy.get("content")

            if content is None:
                deduped.request_messages.append(msg_copy)
                prefix_parts.append(f"{role}:null")
                continue

            content_hash = _hash_content(content)
            prefix_parts.append(f"{role}:{content_hash}")

            # 检查内容是否已存储
            content_key = f"{_CONTENT_PREFIX}{content_hash}"
            if self.db.get(content_key.encode()) is not None:
                # 已存在 → 存引用
                msg_copy.pop("content", None)
                msg_copy["content_ref"] = content_hash
            else:
                # 新内容 → 存正文
                self.db[content_key.encode()] = msgpack.packb(
                    content, use_bin_type=True
                )

            deduped.request_messages.append(msg_copy)

        deduped.prefix_hash = hashlib.sha256(
            "|".join(prefix_parts).encode()
        ).hexdigest()[:16]

        return deduped

    def _resolve_messages(self, turn: Turn) -> None:
        """还原 content_ref → 完整 content（原地修改 turn.request_messages）。"""
        for msg in turn.request_messages:
            ref = msg.get("content_ref")
            if ref is None:
                continue
            content_key = f"{_CONTENT_PREFIX}{ref}"
            raw = self.db.get(content_key.encode())
            if raw is not None:
                msg.pop("content_ref", None)
                msg["content"] = msgpack.unpackb(raw, raw=False)
            else:
                logger.warning("hub_content_ref_missing", hash=ref,
                               hint="content store missing, content_ref left as-is")

    def _put_content(self, content_hash: str, content: Any) -> None:
        """直接存内容（供测试用）。"""
        self.db[f"{_CONTENT_PREFIX}{content_hash}".encode()] = msgpack.packb(
            content, use_bin_type=True
        )

    def store_tools_schema(self, tools: list[dict]) -> str:
        """T4(ADR-0018 §4.4 C3): tools schema 全文经 __msg__ 池 hash 去重存储。

        同一 tools schema 只存一份（hash 去重），Turn 记 tools_hash 引用。
        返回 tools_hash（空=无 tools）。idempotent：已存则跳过。
        """
        if not tools:
            return ""
        import json
        canonical = json.dumps(tools, sort_keys=True, ensure_ascii=False)
        tools_hash = hashlib.sha256(canonical.encode()).hexdigest()[:16]
        key = f"{_CONTENT_PREFIX}{tools_hash}"
        if self.db.get(key.encode()) is None:
            self.db[key.encode()] = msgpack.packb(tools, use_bin_type=True)
        return tools_hash

    def store_raw_request(self, body: dict) -> str:
        """T4(ADR-0018 §4.4 C5): /v1/messages 原始请求体按 hash 入 __msg__ 池。

        parse_anthropic_request 归一化有损（cache_control 丢弃等），存原文供回溯。
        返回 raw_request_ref（hash）。idempotent：已存则跳过。
        """
        if not body:
            return ""
        import json
        canonical = json.dumps(body, sort_keys=True, ensure_ascii=False)
        ref = hashlib.sha256(canonical.encode()).hexdigest()[:16]
        key = f"{_CONTENT_PREFIX}{ref}"
        if self.db.get(key.encode()) is None:
            self.db[key.encode()] = msgpack.packb(body, use_bin_type=True)
        return ref

    def _get_content(self, content_hash: str) -> Any | None:
        """直接取内容（供测试用）。"""
        raw = self.db.get(f"{_CONTENT_PREFIX}{content_hash}".encode())
        if raw is None:
            return None
        return msgpack.unpackb(raw, raw=False)

    # ── ADR-0012 §3.5/3.6: 墓碑 + 管理事件 ──

    def append_tombstone(
        self,
        target_type: TombstoneTargetType,
        target_key: str,
        source: TombstoneSource = TombstoneSource.USER,
    ) -> str:
        """向 Memory Hub 追加一条墓碑记录（ADR-0012 §3.6）。

        墓碑是追加式记录，不违反 Memory Hub 追加不变式。
        写入后 scan_prefix 不再返回被覆盖的条目（原文仍在 Memory Hub）。
        返回墓碑的 Memory Hub key。
        """
        entry_id = _generate_entry_id()
        key = f"{_TOMBSTONE_PREFIX}{entry_id}"
        tombstone = Tombstone(
            target_type=target_type, target_key=target_key, source=source,
        )
        self.db[key.encode()] = msgpack.packb(
            tombstone.model_dump(mode="json"), use_bin_type=True,
        )
        # 失效墓碑缓存
        self._tombstone_cache = None
        logger.info(
            "hub_tombstone_appended", key=key,
            target_type=target_type.value, target_key=target_key,
        )
        # ADR-0018 §4.1: turn->judgment 墓碑级联（删 turn 连带墓碑其专属裁决台账，
        # Memory Index 重建不复活）留 T7 实现--judgment key 含 fact_id = hash(ledger_key+content)，
        # 需 T6 蒸馏产出 content、T7 定义 fact_id 后才能从 turn 找到其 judgment。
        # distill 台账是内容寻址的共享资源（同 source_text 跨 turn 复用），不级联。
        return key

    def append_admin_event(
        self,
        event_type: AdminEventType,
        matter_id: str,
        target_key: str = "",
        **payload: Any,
    ) -> str:
        """向 Memory Hub 追加一条管理事件（ADR-0012 §3.5）。

        管理事件是 Matter 手动操作的 append-only journal。
        Memory Index 重建时重放管理事件，手动映射不丢失。
        返回管理事件的 Memory Hub key。
        """
        entry_id = _generate_entry_id()
        key = f"{_ADMIN_EVENT_PREFIX}{entry_id}"
        event = AdminEvent(
            event_type=event_type, matter_id=matter_id,
            target_key=target_key, payload=payload,
        )
        self.db[key.encode()] = msgpack.packb(
            event.model_dump(mode="json"), use_bin_type=True,
        )
        logger.info(
            "hub_admin_event_appended", key=key,
            event_type=event_type.value, matter_id=matter_id,
        )
        return key

    def scan_tombstones(self) -> Iterator[tuple[str, Tombstone]]:
        """扫描所有墓碑记录（按 entry_id 有序，ADR-0012 §3.6）。"""
        for key_raw, value_raw in self.db.items():
            key_str = key_raw.decode() if isinstance(key_raw, bytes) else str(key_raw)
            if key_str.startswith(_TOMBSTONE_PREFIX):
                data = msgpack.unpackb(value_raw, raw=False)
                yield key_str, Tombstone.model_validate(data)

    def scan_admin_events(self) -> Iterator[tuple[str, AdminEvent]]:
        """扫描所有管理事件（按 entry_id 有序，ADR-0012 §3.5）。"""
        for key_raw, value_raw in self.db.items():
            key_str = key_raw.decode() if isinstance(key_raw, bytes) else str(key_raw)
            if key_str.startswith(_ADMIN_EVENT_PREFIX):
                data = msgpack.unpackb(value_raw, raw=False)
                yield key_str, AdminEvent.model_validate(data)

    # ── ADR-0018 §4.1: 派生台账（distill / judgment）──

    def append_distill(
        self,
        source_text: str,
        distill_model: str,
        prompt_ver: str,
        facts: list[DistillFact],
        matter_proposals: list[MatterProposal],
    ) -> str:
        """写入一条蒸馏台账记录（ADR-0018 §4.1）。

        key = distill/{source_text_hash16}/{model}/{prompt_ver}（确定性）。
        write-if-absent：同 key 已存在则不覆盖（保留首次蒸馏结果，命中复用=零 LLM 成本）。
        返回台账 key。
        """
        h = _source_text_hash(source_text)
        key = f"{_DISTILL_PREFIX}{h}/{distill_model}/{prompt_ver}"
        if self.db.get(key.encode()) is None:
            record = DistillRecord(
                source_text_hash=h, distill_model=distill_model,
                prompt_ver=prompt_ver, facts=facts, matter_proposals=matter_proposals,
            )
            self.db[key.encode()] = msgpack.packb(
                record.model_dump(mode="json"), use_bin_type=True,
            )
            logger.debug("hub_distill_appended", key=key,
                         facts=len(facts), proposals=len(matter_proposals))
        return key

    def get_distill(
        self,
        source_text: str,
        distill_model: str,
        prompt_ver: str,
    ) -> DistillRecord | None:
        """读蒸馏台账（ADR-0018 §4.1）。

        命中语义：同 source_text + model + prompt_ver 才命中（任一变则 miss）。
        被墓碑的台账不返回（Memory Index 重建不复活）。
        """
        h = _source_text_hash(source_text)
        key = f"{_DISTILL_PREFIX}{h}/{distill_model}/{prompt_ver}"
        if self.is_tombstoned(key):
            return None
        raw = self.db.get(key.encode())
        if raw is None:
            return None
        return DistillRecord.model_validate(msgpack.unpackb(raw, raw=False))

    def append_judgment(
        self,
        fact_id: str,
        candidates: list[dict[str, Any]],
        verdict: str,
        judge_model: str,
        *,
        summary_rewrite: str = "",
        reason: str = "",
    ) -> str:
        """追加一条 L4 裁决台账（ADR-0018 §4.1/§3.5）。

        key = judgment/{fact_id}/{seq}（追加式，可重判，非确定性 key）。
        verdict: link:<matter_id> | none | uncertain。
        summary_rewrite: L4 顺带重写的 Matter 摘要（ADR-0018 §3.4）。
        返回台账 key。
        """
        seq = _generate_entry_id()
        key = f"{_JUDGMENT_PREFIX}{fact_id}/{seq}"
        record = JudgmentRecord(
            fact_id=fact_id, candidates=candidates,
            verdict=verdict, judge_model=judge_model,
            summary_rewrite=summary_rewrite, reason=reason,
        )
        self.db[key.encode()] = msgpack.packb(
            record.model_dump(mode="json"), use_bin_type=True,
        )
        logger.debug("hub_judgment_appended", key=key,
                     fact_id=fact_id, verdict=verdict)
        return key

    def scan_judgments(
        self, fact_id: str | None = None,
    ) -> Iterator[tuple[str, JudgmentRecord]]:
        """扫描裁决台账（按 fact_id 前缀，或全部；跳过墓碑，ADR-0018 §4.1）。"""
        tombstoned = self._tombstoned_keys()
        prefix = f"{_JUDGMENT_PREFIX}{fact_id}/" if fact_id else _JUDGMENT_PREFIX
        for key_raw, value_raw in self.db.items():
            key_str = key_raw.decode() if isinstance(key_raw, bytes) else str(key_raw)
            if not key_str.startswith(prefix):
                continue
            if key_str in tombstoned:
                continue
            data = msgpack.unpackb(value_raw, raw=False)
            yield key_str, JudgmentRecord.model_validate(data)

    def is_tombstoned(self, key: str) -> bool:
        """检查某条 Memory Hub key 是否已被墓碑覆盖（turn / 台账等任意类型 target）。"""
        return key in self._tombstoned_keys()

    def _tombstoned_keys(self) -> set[str]:
        """所有被墓碑覆盖的 Memory Hub key 集合（turn / fact / matter / edge / 台账，惰性缓存）。

        写墓碑时失效。scan_prefix 跳过被墓碑的 turn；台账 scan 跳过被墓碑的台账 key。
        """
        if self._tombstone_cache is None:
            self._tombstone_cache = {
                t.target_key for _, t in self.scan_tombstones()
            }
        return self._tombstone_cache

    # ── 物理擦除 stub（ADR-0012 §3.6，compaction 周期留实测）──

    def compact_tombstoned(self) -> int:
        """物理擦除被墓碑标记的原文（G6, ADR-0012 §3.6）。

        遍历 TURN 类型的墓碑，删除对应 Turn 原文（key + msgpack value）。
        墓碑记录本身保留（append-only 不变式 + 审计 trail）。
        __msg__ 内容存储不清理（共享去重，引用检查代价高，且体积小）。
        返回删除的条目数。
        """
        removed = 0
        for _, tombstone in self.scan_tombstones():
            if tombstone.target_type != TombstoneTargetType.TURN:
                continue
            target_key = tombstone.target_key
            if self.db.get(target_key.encode()) is not None:
                del self.db[target_key.encode()]
                removed += 1
        if removed > 0:
            logger.info("hub_compact_tombstoned", removed=removed)
        return removed


class HubDistillJournal:
    """蒸馏台账的 MemoryHub 适配（ADR-0018 §4.1，实现 DistillJournalProtocol）。

    读用传入的 Memory Hub（consolidator 通常 read_only）；写时由调用方（LLMDistiller）
    try/except 降级--read_only Memory Hub 写会抛（增量 consolidation 场景），
    T11 full rebuild（consolidator 读写 Memory Hub）时写生效。
    """

    def __init__(self, ledger: MemoryHub) -> None:
        self._hub = ledger

    def get_distill(
        self, source_text: str, distill_model: str, prompt_ver: str,
    ) -> DistillOutput | None:
        rec = self._hub.get_distill(source_text, distill_model, prompt_ver)
        if rec is None:
            return None
        return DistillOutput(
            facts=list(rec.facts),
            matter_proposals=list(rec.matter_proposals),
            model_name=rec.distill_model,
        )

    def put_distill(
        self, source_text: str, distill_model: str, prompt_ver: str,
        output: DistillOutput,
    ) -> str:
        return self._hub.append_distill(
            source_text, distill_model, prompt_ver,
            output.facts, output.matter_proposals,
        )


def _hash_content(content: Any) -> str:
    """对消息 content 算 hash（支持 str 和 list 多模态）。"""
    if isinstance(content, str):
        data = content.encode()
    elif isinstance(content, list):
        data = msgpack.packb(content, use_bin_type=True)
    else:
        data = str(content).encode()
    return hashlib.sha256(data).hexdigest()[:16]


def _source_text_hash(source_text: str) -> str:
    """蒸馏台账 key 的 source_text hash（ADR-0018 §4.1，sha256[:16]）。

    append_distill/get_distill 共用。T6 蒸馏接入时从 turn user 消息提取文本
    算同一 hash 作 key（提取工具随 T6 落地）。
    """
    return hashlib.sha256(source_text.encode()).hexdigest()[:16]


_entry_seq = 0
"""进程级单调递增序号，保证同毫秒内 entry_id 有序。"""


def _generate_entry_id() -> str:
    """生成唯一且有序的 entry_id（墓碑 / 管理事件用）。

    格式：{unix_ms:013d}-{seq:08d}
    毫秒时间戳保证大致有序，进程级单调序号保证同毫秒内严格有序。
    """
    global _entry_seq
    import time
    _entry_seq += 1
    ts_ms = int(time.time() * 1000)
    return f"{ts_ms:013d}-{_entry_seq:08d}"


def _is_internal_key(key_raw: bytes | str) -> bool:
    """判断一个 RocksDB key 是否为内部 key（非 Turn 条目）。"""
    key_str = key_raw.decode() if isinstance(key_raw, bytes) else str(key_raw)
    return (
        key_str.startswith("__")
        or key_str.startswith(_CONTENT_PREFIX)
        or key_str.startswith(_TOMBSTONE_PREFIX)
        or key_str.startswith(_ADMIN_EVENT_PREFIX)
        or key_str.startswith(_DISTILL_PREFIX)
        or key_str.startswith(_JUDGMENT_PREFIX)
        or key_str.startswith(_ANNOTATION_PREFIX)
    )


# ── V-A4 符号层收口（2026-08-28，Jason 拍板推翻批一「符号不强改」）────────────
# 类名已直接改为 MemoryHub，全仓引用同批更新，旧名（Ledger + RocksDB 拼合的那个类名）不再保留别名
# （pre-beta 无外部消费者）。守卫见 tests/test_hub_naming_guard.py。
