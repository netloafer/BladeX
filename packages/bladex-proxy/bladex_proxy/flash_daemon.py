"""Flash 维护进程 —— 单写者物化 + 用户直编捕获（ADR-0032 §5.2；批二卡 V-F2）。

独立后台进程（与 consolidator **不合并**，拍板 §10-5），账本池投影与
`behavior.json` 的**唯一写者**（用户直编与 dashboard 除外——那两路经本进程捕获为管理事件）。
批 F4（2026-09-23）起不再写四级目录树、`AGENTS.md` 名册与 USER/RULES/AGENT.md 投影
（ADR-0034 §3.1：零读者，行为画像 F2/F3 接替）。

# 🔴 红线

1. **文件是投影**：真相在 Hub 事件流。`run_once` 的物化 = 从注入的
   `ledger_source`（Hub 投影提供方）重渲染；删树重跑逐字一致。
2. **用户直编不被覆盖**：检测到池内文件与上次物化快照不一致 ⇒ 那是用户改的
   ⇒ parse + 经 `emit_admin_event` 回调**采纳**（2026-08-29 语义：只进 proxy
   内存池不写 Hub，随会话由模型工具面自然收敛；日志留痕供追溯）。
   **采纳 = 用户版进池 + 池版回投影，一步完成**（MQ-L47，2026-09-06）：proxy
   在响应里带回采纳后的池版（rev 已 +1），本进程**当轮**把它写回文件并更新快照
   ——红线的边界是"用户版进池**之前**不许覆盖"，进池之后写回的正是用户版本身
   （只多 rev/时间戳元数据）。旧 proxy 响应不带 `ledger` 时退回"挂起等收敛"路径
   （事件流回、渲染与文件一致才接管；重发有上限）。`*.local.md` 恒不碰（flash 红线 4）。
3. **零 LLM**：本进程全程不经 Router（批 F4 删除 V-F2b 的两个简介蒸馏器后成立，
   守卫 `test_flash_f4_retirement.py` 逐字扫本文件：Router 网关符号出现次数 = 0）。
4. 导入无副作用：env/配置读取全在 `main()` 入口（consolidator `_load_env_file`
   搬包事故的纪律）。

数据源与事件写侧都是注入依赖（`ledger_source` / `emit_admin_event`）——
Hub 接线归启用卡；本骨架在沙盒可全测。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections.abc import Callable
from datetime import UTC

import structlog
from bladex_core.flash import SCOPE_PERSONAL, behavior_path, write_if_changed
from bladex_core.ledger import Ledger, ledger_md_path, parse_ledger_md, render_ledger_md

from bladex_proxy.storage.startup import open_hub_with_startup_window

logger = structlog.get_logger()

#: Hub 投影提供方：() -> (账本池 {ledger_id: Ledger}, 会话绑定 {session_id: ledger_id})。
#: 典型实现 = 扫 Hub 管理事件 + `bladex_core.ledger.replay_ledger_events`。
LedgerSource = Callable[[], tuple[dict[str, Ledger], dict[str, str]]]
#: 管理事件写侧：(event_type: str, payload: dict) -> 响应 JSON | None。
#: 返回值若为 dict 且含 `ledger`（采纳后的池版），调用方当轮写回投影（MQ-L47）；
#: None / 无 `ledger` = 旧 proxy 或纯写侧，退回挂起等收敛路径。失败以抛出表达。
EmitEvent = Callable[[str, dict], dict | None]
#: 未收敛直编的重发上限（次）：超过即告警停发、pending 保留不覆盖（MQ-L47）。
REEMIT_CAP = 3
_REV_LINE_RE = re.compile(r"^- rev:\s*(\d+)\s*$", re.MULTILINE)


def rendered_rev(text: str) -> int:
    """读 MD 元数据行 `- rev: N`（`render_ledger_md` 产物）；缺失 ⇒ -1。

    `parse_ledger_md` 有意不带回 rev（用户不该能改乐观锁计数），但 daemon 判
    "磁盘这份是哪个 rev 渲染出来的"正需要它——只读不写，不经解析结果传播。
    """
    m = _REV_LINE_RE.search(text)
    return int(m.group(1)) if m else -1


class FlashDaemon:
    """单写者物化循环。线程/进程模型：进程内单实例、串行 `run_once`——
    多源写入经 push 收件箱（启用卡接 Hub 通知），物化永远只有这一支笔。"""

    def __init__(self, *, root: str, principal: str, ledger_source: LedgerSource,
                 emit_admin_event: EmitEvent, scope: str = SCOPE_PERSONAL,
                 behavior_every_s: float = 300.0,
                 tombstoned_source: Callable[[], set[str]] | None = None,
                 behavior_source: Callable[[], dict] | None = None,
                 tz_offset_s: int = 0,
                 bench_projects: tuple[str, ...] | None = None,
                 confirmed_projects: tuple[str, ...] = ()) -> None:
        self._root = root
        self._principal = principal
        self._scope = scope
        self._source = ledger_source
        #: MQ-F3：() -> 已墓碑账本 id 集合（Hub 真相）。投影删除**只认这个集合**——
        #: 「不在池里」不是删除判据（池装载降级为空时会误删活本），「被墓碑」才是。
        self._tombstoned_source = tombstoned_source
        self._emit = emit_admin_event
        #: path -> 上次物化的内容快照。文件≠快照 ⇒ 用户直编（红线 2 的判据：
        #: 与"文件≠当前渲染"不同——池数据更新也会造成后者，只有偏离**我们上次
        #: 写下的形态**才证明是别人动了文件）。
        self._snapshots: dict[str, str] = {}
        #: 已捕获、事件尚未经 Hub 流回的用户直编——物化对这些路径**跳过写**，
        #: 直到渲染内容与盘上一致（收敛）才恢复接管（红线 2 的零覆盖窗口）。
        self._pending_edits: set[str] = set()
        # 行为画像节流：`scan_meta_snapshot` 是全 Hub 元数据扫描，不适合 15s 一跑。
        # （批 F4 前这条 300s 轨道由目录树与画像共用；树已删，轨道只剩画像。）
        self._behavior_every_s = behavior_every_s
        self._behavior_last = 0.0
        self._dirty = True          # 收件箱有活（首轮恒真）
        self._behavior_dirty = True  # 画像=会话流量的投影，只在有新流量后重扫
        #: 持久小状态（root/.flash_state.json）：baseline = 每文件"我们上次写下/
        #: 采纳的内容" sha —— 跨重启区分「用户离线改过」vs「投影只是过期」；
        #: adopted = 已采纳、Hub 尚未收敛的直编 sha（重启后恢复零覆盖窗口 +
        #: 重发采纳）。丢了不产生正确性问题，只是无法识别停机期间的直编。
        self._state_path = os.path.join(root, ".flash_state.json")
        self._baseline_sha: dict[str, str] = {}
        self._adopted_sha: dict[str, str] = {}
        # ── F2 行为画像轨道（`behavior.json`，与树同一条 300s 节流轨道并行）──
        #: () -> 数据档 dict（`hub_behavior_source`）；None = 轨道不启用
        self._behavior_source = behavior_source
        self._tz_offset_s = int(tz_offset_s)
        from bladex_core.behavior import BENCH_PROJECT_NAMES_CONFIRMED
        self._bench_projects = tuple(bench_projects if bench_projects is not None
                                     else BENCH_PROJECT_NAMES_CONFIRMED)
        self._confirmed_projects = tuple(confirmed_projects)
        #: 增量状态（持久，`.flash_state.json["behavior"]`）：
        #:   tombstone_hash / watermark_ms / projects: [[pid,name,root],…] /
        #:   sessions: {"principal/agent/session": {entry_id: [ts_ms, flags, pidx]}}
        #: 丢了不产生正确性问题——下一轮全量 fold（判据①：增量 == 全量）。
        self._bh_state: dict = {"tombstone_hash": "", "watermark_ms": 0,
                                "projects": [], "sessions": {}}
        self._bh_keys: set[str] = set()      # 已 fold 的 turn key（进程内索引）
        #: 新鲜度读数（持久，`behavior_status`）：`flash_lag_s` 等，status / admin 读它
        self._bh_status: dict = {}
        #: 测试注入的时钟（None = 墙上时钟）；聚合本身是纯函数，时钟只在这一层
        self.behavior_clock: Callable[[], object] | None = None
        #: path -> 采纳后**池版的 rev**（新采纳路径，MQ-L47）。收敛判据 =
        #: Hub 投影里该本 rev ≥ 它（模型在采纳后写过 ⇒ 其 update 事件带的整本
        #: 含用户版）——不再逐字比对渲染（rev/时间戳元数据让逐字相等恒假）。
        #: 有此项的路径**不走重发通道**（proxy 已持有用户版；proxy 单独重启的
        #: 丢失由下次 daemon 启动的 `_reconcile_stale_adoption` 补采）。持久化。
        self._adopted_rev: dict[str, int] = {}
        self._reemit_last: dict[str, float] = {}
        #: path -> 已重发次数（进程内；≥ REEMIT_CAP 停发并告警一次）。
        self._reemit_count: dict[str, int] = {}
        self._load_state()
        #: 预留钩子（机制先立、规则后到；铁律"无署名消费者的字段不许生产"——
        #: 所以**不加配置 flag**，钩子非 None 才进调度）：
        #: upkeep = 账本自身低频维护（0.2.0）；extract = Flash→Index 提取（0.3.0，
        #: 拍板节奏约 1 小时）。
        self.upkeep_hook: Callable[[], None] | None = None
        self.upkeep_every_s = 0.0            # 0.2.0 落规则时定频（拍板：不能高频）
        self.extract_hook: Callable[[], None] | None = None
        self.extract_every_s = 3600.0        # 拍板节奏：约 1 小时

    def _load_state(self) -> None:
        try:
            with open(self._state_path, encoding="utf-8") as f:
                data = json.load(f)
            self._baseline_sha = dict(data.get("baseline") or {})
            self._adopted_sha = dict(data.get("adopted") or {})
            self._adopted_rev = {k: int(v) for k, v in
                                 (data.get("adopted_rev") or {}).items()}
            bh = data.get("behavior")
            if isinstance(bh, dict) and isinstance(bh.get("sessions"), dict):
                self._bh_state = {"tombstone_hash": str(bh.get("tombstone_hash", "")),
                                  "watermark_ms": int(bh.get("watermark_ms", 0) or 0),
                                  "projects": list(bh.get("projects") or []),
                                  "sessions": dict(bh["sessions"])}
                self._bh_keys = {f"{sk}/{e}" for sk, ents in self._bh_state["sessions"].items()
                                 for e in ents}
            st = data.get("behavior_status")
            if isinstance(st, dict):
                self._bh_status = dict(st)
        except (OSError, ValueError):
            pass               # 首次运行/状态损坏：从零开始（见字段注释）

    def _save_state(self) -> None:
        try:
            tmp = self._state_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"baseline": self._baseline_sha,
                           "adopted": self._adopted_sha,
                           "adopted_rev": self._adopted_rev,
                           "behavior": self._bh_state,
                           "behavior_status": self._bh_status}, f)
            os.replace(tmp, self._state_path)
        except OSError as e:
            logger.warning("flash_daemon_state_save_failed", error=str(e))

    @staticmethod
    def _sha(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def _base_of(self, ledger_id: str) -> Ledger | None:
        """池里当前那本，供 `parse_ledger_md` 带回 MD 不携带的字段。

        取不到就返回 None——解析照常进行，只是那几个字段会是空；
        比"因为读不到池就整条用户直编丢掉"要好。
        """
        try:
            pool, _ = self._source()
        except Exception as e:  # noqa: BLE001 —— 投影读失败不阻断直编捕获
            logger.warning("flash_daemon_base_lookup_failed",
                           ledger=ledger_id, error=str(e))
            return None
        return pool.get(ledger_id)

    # ── 收件箱（多源 push、单写者消费）──
    def push(self, kind: str = "turn") -> None:
        self._dirty = True
        self._behavior_dirty = True

    def startup_check(self) -> dict[str, int]:
        """启动自检（Jason 2026-08-29 拍板第二条）：骨架缺失即重建 + 识别
        停机期间的用户直编，然后跑一轮全量物化。

        直编识别靠持久 baseline：文件 sha == 上次写下/采纳的 sha ⇒ 只是投影
        （Hub 若前进了就放心重写）；sha 不同 ⇒ 停机期间被人改过 ⇒ 交给
        `_capture_user_edits` 走采纳。没有 baseline 记录的文件当投影处理
        （首次运行没有更好的判据，如实记录在案）。
        """
        from bladex_core.ledger import ledgers_dir
        d = ledgers_dir(self._root, self._principal, scope=self._scope)
        try:
            os.makedirs(d, exist_ok=True)
        except OSError as e:
            logger.warning("flash_daemon_skeleton_failed", dir=d, error=str(e))
        try:
            pool, _ = self._source()
        except Exception as e:  # noqa: BLE001 —— Hub 不在也要把骨架立起来
            logger.warning("flash_daemon_startup_source_failed", error=str(e))
            pool = {}
        offline = 0
        for ledger_id in pool:
            path = ledger_md_path(self._root, self._principal, ledger_id,
                                  scope=self._scope)
            base = self._baseline_sha.get(path)
            if base is None or not os.path.isfile(path):
                continue
            try:
                with open(path, encoding="utf-8") as f:
                    current = f.read()
            except OSError:
                continue
            if self._sha(current) == base:
                self._snapshots[path] = current       # 未被动过的投影
                if path in self._adopted_sha or path in self._adopted_rev:
                    self._reconcile_stale_adoption(path, current, pool.get(ledger_id))
            else:
                # 停机期间的直编：给个 ≠ 文件的快照哨兵，让捕获路径接手
                self._snapshots[path] = "<edited-while-daemon-down>"
                offline += 1
                logger.info("flash_daemon_offline_edit_detected", path=path)
        stats = self.run_once()
        stats["offline_edits"] = offline
        # 启动对账可能只改了采纳状态（补采/接管）而没写文件——`run_once` 的
        # "有写才存"会漏掉它，重启后又会对同一条再对账一次。
        self._save_state()
        return stats

    def _reconcile_stale_adoption(self, path: str, current: str,
                                  pool_led: Ledger | None) -> None:
        """启动时处理**存量**「已采纳、未收敛」条目（MQ-L47：修前它们跨重启存活，
        468d 连续 37 小时 pending、每轮重发旧版盖掉模型写入）。

        磁盘 sha == adopted sha ⇒ 用户采纳后没再动过文件。两种形态按 rev 分
        （采纳 rev = `_adopted_rev`（新路径）或 磁盘 rev + 1（旧路径推算））：
        - **池（Hub 投影）rev ≥ 采纳 rev** ⇒ Hub 已越过那次采纳（模型在采纳后写过，
          其 `ledger_update` 事件带的是"用户版 + 模型新写"整本）⇒ **池版胜**，不重发
          （重发 = 再盖一次模型写入，正是被修的缺陷），直接接管让物化把池版写回，
          磁盘 rev 追平池 rev。
        - **否则** ⇒ 采纳没进 Hub（proxy 重启丢了内存池、模型也没写过）⇒ 走一次
          新采纳路径（emit + 响应池版回写）；proxy 不在则保留 pending（旧路径的
          交给限流重发通道，有上限；新路径的等下次启动再补）。
        MQ-L51（09-06）后采纳当轮落 Hub ⇒ `pool.rev ≥ adopted_rev` 在重启后成立，
        `readopted` 只剩"采纳前 proxy 就没写成 Hub"一种来源；判据不改。
        """
        eid = os.path.splitext(os.path.basename(path))[0]
        disk_rev = rendered_rev(current)
        adopted_rev = self._adopted_rev.get(path, disk_rev + 1)
        if pool_led is not None and pool_led.rev >= adopted_rev:
            self._forget_adoption(path)
            # 快照 = 磁盘内容 ⇒ 物化时 snapshot == on_disk ⇒ 不是直编 ⇒ 池版覆盖。
            self._snapshots[path] = current
            logger.info("flash_daemon_stale_adoption_reconciled", ledger=eid,
                        mode="pool_wins", disk_rev=disk_rev, pool_rev=pool_led.rev,
                        adopted_rev=adopted_rev)
            return
        adopted = self._adopt(path, current, base=pool_led)
        if adopted is None:
            # proxy 不在（`bladex start` 里 daemon 比 proxy 早 1–2 秒起，09-06 两次重启
            # 都撞上）：恢复零覆盖窗口，并**降级为旧路径 pending**（`_adopted_sha`），
            # 让有上限的重发通道接手——新路径 pending 本身不重发，不降级就会卡到
            # 下一次 daemon 重启再撞一次同样的竞态（09-06 11:23 实录，E0.1c）。
            self._adopted_rev.pop(path, None)
            self._adopted_sha[path] = self._sha(current)
            self._pending_edits.add(path)
            return
        logger.info("flash_daemon_stale_adoption_reconciled", ledger=eid,
                    mode="readopted", disk_rev=disk_rev,
                    pool_rev=(pool_led.rev if pool_led is not None else -1),
                    adopted_rev=adopted_rev, converged=adopted)

    def _forget_adoption(self, path: str) -> None:
        """一次采纳的全部状态清零（收敛/接管时用）。"""
        self._pending_edits.discard(path)
        self._adopted_sha.pop(path, None)
        self._adopted_rev.pop(path, None)
        self._reemit_last.pop(path, None)
        self._reemit_count.pop(path, None)

    def _adopt(self, path: str, current: str, *, base: Ledger | None) -> bool | None:
        """把磁盘上的用户版发去采纳。返回 None=送达失败（调用方决定挂起/重试）；
        True=响应带池版且已当轮写回（收敛）；False=响应无 `ledger`（旧 proxy），
        已进 pending 等 Hub 收敛。解析失败也返回 None（调用方已各自处理日志）。"""
        try:
            ledger = parse_ledger_md(current, base=base)
        except Exception:  # noqa: BLE001
            return None
        try:
            resp = self._emit("ledger_user_edit", {"ledger": ledger.model_dump()})
        except Exception as e:  # noqa: BLE001 —— proxy 不在/网络抖动
            logger.warning("flash_daemon_emit_failed", ledger=ledger.ledger_id, error=str(e))
            return None
        return self._settle_adoption(path, current, ledger.ledger_id, resp)

    def _settle_adoption(self, path: str, current: str, ledger_id: str,
                         resp: dict | None) -> bool:
        """采纳送达后的落定。响应带 `ledger` ⇒ 池版当轮写回、快照/基线 = 池版，
        记 `_adopted_rev`（收敛改按 rev 判、**不重发**）；否则旧语义：快照 = 用户版
        + `_adopted_sha` 挂起等逐字收敛（限流重发，有上限）。

        两种形态都进 `_pending_edits`（= 物化跳过写，直到 Hub 收敛）：daemon 的
        `ledger_source` 是 **Hub 投影**，采纳只进 proxy 内存池不写 Hub，Hub 要等模型
        下一次 `ledger_update` 才带上用户版——这期间若放开物化，Hub 旧版会把刚写回
        的用户版再盖掉。
        MQ-L51（09-06）后 proxy 采纳当轮落 `LEDGER_USER_EDIT` 事件 ⇒ **Hub 当轮持有采纳版**，
        pending 在下一次 `ledger_source` 读到 rev ≥ 采纳 rev 时自然清；判据不改。"""
        pool_led: Ledger | None = None
        if isinstance(resp, dict) and resp.get("ledger"):
            try:
                pool_led = Ledger.model_validate(resp["ledger"])
            except Exception as e:  # noqa: BLE001 —— 响应形态坏：退回挂起路径
                logger.warning("flash_daemon_adoption_response_invalid",
                               ledger=ledger_id, error=str(e))
                pool_led = None
        if pool_led is not None:
            content = render_ledger_md(pool_led)
            write_if_changed(path, content)
            self._forget_adoption(path)
            self._snapshots[path] = content
            self._baseline_sha[path] = self._sha(content)
            self._adopted_rev[path] = pool_led.rev
            self._pending_edits.add(path)              # 等 Hub rev 追上，不重发
            logger.info("flash_daemon_user_edit_adopted", ledger=ledger_id,
                        path=path, rev=pool_led.rev, converged=True)
            return True
        # 旧 proxy：快照更新为用户版 + 挂起，物化跳过直到 Hub 逐字收敛。
        # 🔴 不清 `_reemit_count`——重发成功也计次，否则旧 proxy 下每 30s 一次永不停（原缺陷形态）。
        self._adopted_rev.pop(path, None)
        self._snapshots[path] = current
        self._pending_edits.add(path)
        self._baseline_sha[path] = self._sha(current)
        self._adopted_sha[path] = self._sha(current)
        # 限流基点 = 采纳时刻：本轮已上报，重发通道 30s 后才轮到
        self._reemit_last[path] = time.time()
        logger.info("flash_daemon_user_edit_adopted", ledger=ledger_id, path=path,
                    rev=-1, converged=False)
        return False

    def run_once(self) -> dict[str, int]:
        """一轮维护：先捕获用户直编，再物化。返回读数（写/编辑捕获/跳过）。"""
        edits = self._capture_user_edits()
        written = 0
        removed = 0
        if self._dirty or edits or self.pending_wait_s() is not None:
            pool, _bindings = self._source()
            written = self._materialize(pool)
            removed = self._purge_tombstoned(pool)
            self._dirty = False
        # 行为画像按节流独立于账本事件（轮次元数据来自全部流量，不只账本）。
        now = time.time()
        if (self._behavior_source is not None and self._behavior_dirty
                and now - self._behavior_last >= self._behavior_every_s):
            self._behavior_last = now
            self._behavior_dirty = False
            written += self._materialize_behavior()
        if written or edits or removed:
            self._save_state()
        return {"written": written, "user_edits": edits, "removed": removed}

    def pending_wait_s(self) -> float | None:
        """旧路径 pending（`_adopted_sha` 非空且未达重发上限）还差多久到下一次重发窗；
        None = 没有需要重发的（E0.1b，2026-09-06）。

        live 09-06 11:06：`bladex start` 里 daemon 比 proxy 早 1–2 秒起来，启动对账的
        补采撞上 Connection refused 落回 pending；而主循环是**唤醒驱动**的——没有新流量
        `run_once` 根本不跑，468d 的重发在没有请求进来之前永远等不到第二次。主循环据此
        把阻塞超时压到重发窗，上限（`REEMIT_CAP`）到了就回到零轮询。
        """
        due = [self._reemit_last.get(p, 0.0) for p in self._pending_edits
               if p in self._adopted_sha and self._reemit_count.get(p, 0) < REEMIT_CAP]
        if not due:
            return None
        return max(0.0, 30.0 - (time.time() - min(due)))

    def behavior_wait_s(self) -> float | None:
        """画像物化还差多久到窗（None = 没有待扫的流量）。主循环用它定阻塞时长。"""
        if self._behavior_source is None or not self._behavior_dirty:
            return None
        return max(0.0, self._behavior_every_s - (time.time() - self._behavior_last))

    def _capture_user_edits(self) -> int:
        captured = 0
        for path, snapshot in list(self._snapshots.items()):
            try:
                if not os.path.isfile(path):
                    continue
                with open(path, encoding="utf-8") as f:
                    current = f.read()
            except OSError as e:
                logger.warning("flash_daemon_read_failed", path=path, error=str(e))
                continue
            if current == snapshot:
                continue
            # 用户改了这份投影 ⇒ 捕获为事件；事件流回前不重写（红线 2）。
            expected_id = os.path.splitext(os.path.basename(path))[0]
            try:
                # 🔴 带 base：`goal_verbatim` / `goal_revisions` 不在 MD 里
                # （`render_ledger_md` 的产物就是注入块，渲染原话 = 又注一遍）。
                # 不带 base 解析出来的账本会把这两个字段清空，而这里正是
                # "拿解析结果当完整状态发事件"的形态 —— 那就是静默数据丢失。
                ledger = parse_ledger_md(current, base=self._base_of(expected_id))
            except Exception as e:  # noqa: BLE001 —— 手编坏格式不炸循环
                logger.warning("flash_daemon_user_edit_unparseable",
                               path=path, error=str(e))
                self._snapshots[path] = current   # 防每轮重报同一份坏文件
                self._baseline_sha[path] = self._sha(current)
                continue
            if ledger.ledger_id != expected_id:
                # `- ledger:` 元数据行被删/改坏 ⇒ 解析出的是随机新 id 的空账本，
                # 发出去就是给不存在的账本造事件。可 grep 告警 + 不发不覆盖。
                logger.warning("flash_daemon_user_edit_id_mismatch",
                               path=path, parsed=ledger.ledger_id,
                               expected=expected_id)
                self._snapshots[path] = current
                self._baseline_sha[path] = self._sha(current)
                continue
            try:
                resp = self._emit("ledger_user_edit", {"ledger": ledger.model_dump()})
            except Exception as e:  # noqa: BLE001 —— proxy 不在/网络抖动
                # 🔴 快照**不**更新 ⇒ 下轮重新捕获重发。用户直编绝不静默丢
                # （红线 2 的失败半边：捕获成功≠送达成功，送达才算数）。
                logger.warning("flash_daemon_emit_failed",
                               ledger=ledger.ledger_id, error=str(e))
                continue
            # 采纳落定（MQ-L47）：响应带池版 ⇒ 当轮写回、不挂起；否则旧挂起语义。
            # （2026-08-29 语义不变：采纳只进 proxy 内存池不写 Hub，
            # Hub 随会话由模型的 ledger_update 自然追平。）
            self._settle_adoption(path, current, ledger.ledger_id, resp)
            captured += 1
        return captured

    def _materialize(self, pool: dict[str, Ledger]) -> int:
        written = 0
        for ledger_id, ledger in sorted(pool.items()):
            path = ledger_md_path(self._root, self._principal, ledger_id,
                                  scope=self._scope)
            content = render_ledger_md(ledger)
            snapshot = self._snapshots.get(path)
            try:
                on_disk = None
                if os.path.isfile(path):
                    with open(path, encoding="utf-8") as f:
                        on_disk = f.read()
            except OSError:
                on_disk = None
            if path in self._pending_edits:
                adopted_rev = self._adopted_rev.get(path)
                by_rev = adopted_rev is not None and ledger.rev >= adopted_rev
                if content == on_disk or by_rev:
                    # Hub 已收敛（逐字相等 = 旧路径；rev 追上 = 新路径，MQ-L47），
                    # 恢复接管：池版写盘（by_rev 时它含用户版 + 模型新写）。
                    self._forget_adoption(path)
                    written += write_if_changed(path, content)
                    self._snapshots[path] = content
                    self._baseline_sha[path] = self._sha(content)
                    logger.info("flash_daemon_user_edit_converged", path=path,
                                by="rev" if by_rev else "verbatim",
                                pool_rev=ledger.rev)
                    continue
                if adopted_rev is None:
                    # 旧路径未收敛 ⇒ 绝不覆盖（红线 2）；限流重发采纳——proxy 若
                    # 中途重启，内存池的采纳会丢，重发是自愈通道（端点幂等）。
                    # MQ-L47：成功/失败都留日志，且有上限——修前成功无痕、失败
                    # `pass`，175 条重发在 daemon 侧零痕迹，把模型写入盖了两天。
                    self._reemit_pending(path, on_disk)
                continue            # 新路径：proxy 已持有用户版，等 Hub，不重发
            elif (snapshot is not None and on_disk is not None
                    and on_disk != snapshot and on_disk != content):
                # 本轮物化前刚出现、还没走过捕获的直编 ⇒ 同样不覆盖，下轮捕获。
                continue
            written += write_if_changed(path, content)
            self._snapshots[path] = content
            self._baseline_sha[path] = self._sha(content)
        return written

    def _purge_tombstoned(self, pool: dict[str, Ledger]) -> int:
        """MQ-F3（2026-09-18）：墓碑之后投影文件**必须不可读**——删除语义要兑现到文件面。

        删法（跑前登记；纪律「投影可删、用户内容宁留勿删」——与批 F4 删掉的
        目录回收 `_reclaim_dirs` 同一套）：
        - 判据 = **Hub 墓碑集合**（`tombstoned_source`），不是「不在池里」——池装载
          降级为空时后者会把活本全删；墓碑集合读不到（空集）就什么都不删。
        - 同时在池里 ⇒ 不删（防御：不该发生，发生了要可 grep）。
        - `.md` 投影 `unlink`；同名 `.local.md`（用户覆盖，永不被机器碰）**不动**。
        - 待收敛直编（`_pending_edits`）或盘上内容 ≠ 我们上次写下的（未捕获直编）
          ⇒ **不删**、告警 `flash_ledger_tombstoned_kept_user_edit`——误删用户内容比
          留一份陈旧文件更糟（红线 2）。
        - 删掉的清 snapshot/baseline 状态；空掉的分桶目录顺手收。
        每删一本记 `flash_ledger_projection_removed`。返回删除数。
        """
        if self._tombstoned_source is None:
            return 0
        try:
            tombstoned = set(self._tombstoned_source())
        except Exception as e:  # noqa: BLE001 —— 读不到墓碑 = 不删，不是拒服务
            logger.debug("flash_tombstone_source_failed", error=str(e))
            return 0
        removed = 0
        for ledger_id in sorted(tombstoned):
            if ledger_id in pool:
                logger.warning("flash_ledger_tombstoned_but_in_pool", ledger=ledger_id)
                continue
            path = ledger_md_path(self._root, self._principal, ledger_id,
                                  scope=self._scope)
            if not os.path.isfile(path):
                continue
            try:
                with open(path, encoding="utf-8") as f:
                    on_disk = f.read()
            except OSError:
                on_disk = None
            baseline = self._baseline_sha.get(path)
            snapshot = self._snapshots.get(path)
            user_touched = (path in self._pending_edits
                            or (on_disk is not None and snapshot is not None
                                and on_disk != snapshot)
                            or (on_disk is not None and snapshot is None
                                and baseline is not None
                                and self._sha(on_disk) != baseline))
            if user_touched:
                logger.warning("flash_ledger_tombstoned_kept_user_edit",
                               ledger=ledger_id, path=path)
                continue
            try:
                os.unlink(path)
            except OSError as e:
                logger.warning("flash_ledger_projection_remove_failed",
                               ledger=ledger_id, path=path, error=str(e))
                continue
            removed += 1
            self._snapshots.pop(path, None)
            self._baseline_sha.pop(path, None)
            self._adopted_sha.pop(path, None)
            self._adopted_rev.pop(path, None)
            logger.info("flash_ledger_projection_removed", ledger=ledger_id, path=path)
            # 分桶目录空了就收（两级：`ledgers/<a>/<b>/`）
            d = os.path.dirname(path)
            for _ in range(2):
                try:
                    os.rmdir(d)
                except OSError:
                    break
                d = os.path.dirname(d)
        return removed

    def _reemit_pending(self, path: str, on_disk: str | None) -> None:
        """挂起直编的限流重发（≥30s 一次，最多 `REEMIT_CAP` 次）。

        响应带池版 ⇒ 借 `_settle_adoption` 当轮收敛（旧 pending 被新 proxy 捞出）。
        达上限 ⇒ `warning flash_daemon_user_edit_reemit_capped` 恰一次并停发；
        pending 保留、文件不覆盖（红线 2）——人来看告警，机器不再无限重试。
        """
        if on_disk is None:
            return
        eid = os.path.splitext(os.path.basename(path))[0]
        n = self._reemit_count.get(path, 0)
        if n >= REEMIT_CAP:
            if n == REEMIT_CAP:
                self._reemit_count[path] = n + 1        # 告警只打一次
                logger.warning("flash_daemon_user_edit_reemit_capped", ledger=eid,
                               attempts=n, cap=REEMIT_CAP)
            return
        now_m = time.time()
        if now_m - self._reemit_last.get(path, 0.0) < 30.0:
            return
        self._reemit_last[path] = now_m
        self._reemit_count[path] = n + 1
        try:
            ledger = parse_ledger_md(on_disk, base=self._base_of(eid))
            resp = self._emit("ledger_user_edit", {"ledger": ledger.model_dump()})
        except Exception as e:  # noqa: BLE001 —— 下轮再试（计次）
            logger.warning("flash_daemon_reemit_failed", ledger=eid,
                           attempt=n + 1, error=str(e))
            return
        logger.info("flash_daemon_user_edit_reemitted", ledger=eid, attempt=n + 1)
        if isinstance(resp, dict) and resp.get("ledger"):
            self._settle_adoption(path, on_disk, eid, resp)

    # ── F2 · 行为画像轨道 ──

    def behavior_path(self) -> str:
        return behavior_path(self._root, self._principal, scope=self._scope)

    def reset_behavior_state(self) -> None:
        """丢弃增量状态（G2：墓碑集合变 / 状态损坏；测试用它做"删状态重跑"）。"""
        self._bh_state = {"tombstone_hash": "", "watermark_ms": 0, "projects": [], "sessions": {}}
        self._bh_keys = set()

    def _fold_behavior_rows(self, rows: list[tuple[str, dict]]) -> tuple[int, int]:
        """把快照里**没见过**的 turn 折进增量状态。返回 (新折入数, 本轮见到的最大 ts_ms)。"""
        from bladex_core.behavior import TurnMeta
        st = self._bh_state
        pidx: dict[str, int] = {p[0]: i for i, p in enumerate(st["projects"])}
        new_n = 0
        max_ms = 0
        for key, meta in rows:
            m = TurnMeta.from_scan(key, meta.get("ts", ""))
            if m is None:
                continue
            ts_ms = int(m.ts.timestamp() * 1000)
            max_ms = max(max_ms, ts_ms)
            if key in self._bh_keys:
                continue
            pid = str(meta.get("project_id") or "")
            if pid:
                if pid not in pidx:
                    pidx[pid] = len(st["projects"])
                    st["projects"].append([pid, "", ""])
                # 名字/根路径取最新非空（身份是 pid，名字只是投影）
                rec = st["projects"][pidx[pid]]
                rec[1] = str(meta.get("project_name") or "") or rec[1]
                rec[2] = str(meta.get("project_root") or "") or rec[2]
            flags = (1 if meta.get("auxiliary") else 0) | (2 if meta.get("subagent") else 0)
            sk, _, entry = key.rpartition("/")
            st["sessions"].setdefault(sk, {})[entry] = [ts_ms, flags, pidx.get(pid, -1)]
            self._bh_keys.add(key)
            new_n += 1
        return new_n, max_ms

    def _behavior_metas(self) -> list:
        """增量状态 → `TurnMeta` 列表（聚合的唯一输入形态；全量与增量在这里汇合）。"""
        from datetime import datetime

        from bladex_core.behavior import TurnMeta
        st = self._bh_state
        projects = st["projects"]
        out = []
        for sk, ents in st["sessions"].items():
            parts = sk.split("/")
            if len(parts) < 3:
                continue
            for entry, (ts_ms, flags, pi) in ents.items():
                pid, pname, proot = projects[pi] if 0 <= pi < len(projects) else ("", "", "")
                out.append(TurnMeta(principal=parts[0], agent_id=parts[1], session_id=parts[2],
                                    ts=datetime.fromtimestamp(ts_ms / 1000, tz=UTC),
                                    key=f"{sk}/{entry}", project_id=pid, project_name=pname,
                                    project_root=proot, auxiliary=bool(flags & 1),
                                    subagent=bool(flags & 2)))
        return out

    def _materialize_behavior(self, *, now: object | None = None) -> int:
        """`behavior.json`（方案 §1）：增量 fold + 纯函数聚合 + `flash_lag_s`（补遗 G7）。

        - 墓碑集合指纹变了 ⇒ 丢状态全量重算（G2；日志 `flash_behavior_state_reset`）；
        - lag = 本轮快照最大 turn ts − 上次写入时纳入的 watermark；无新轮 ⇒ 0；
          超 `FLASH_LAG_BOUND_S` 打 warning `flash_lag_exceeded`（MQ-F4）；
        - 零 LLM：本函数与数据源都不碰 Router。
        """
        from datetime import datetime

        from bladex_core.behavior import FLASH_LAG_BOUND_S, aggregate, model_view_json
        t0 = time.perf_counter()
        try:
            inputs = self._behavior_source() if self._behavior_source else {}
        except Exception as e:  # noqa: BLE001 —— 画像源失败不连累账本池
            logger.warning("flash_daemon_behavior_source_failed", error=str(e))
            return 0
        th = str(inputs.get("tombstone_hash", ""))
        if self._bh_state.get("tombstone_hash") and self._bh_state["tombstone_hash"] != th:
            logger.info("flash_behavior_state_reset", reason="tombstones_changed",
                        before=self._bh_state["tombstone_hash"], after=th)
            self.reset_behavior_state()
        self._bh_state["tombstone_hash"] = th
        prev_wm = int(self._bh_state.get("watermark_ms", 0) or 0)
        new_n, max_ms = self._fold_behavior_rows(list(inputs.get("rows") or []))
        lag_s = max(0, (max_ms - prev_wm) // 1000) if new_n and prev_wm else 0
        if not isinstance(now, datetime):
            now = self.behavior_clock() if self.behavior_clock else None
        now_dt = now if isinstance(now, datetime) else datetime.now(UTC)
        behavior = aggregate(
            self._behavior_metas(), now=now_dt,
            ledger_events=inputs.get("ledger_events") or (),
            tombstoned_ledgers=inputs.get("tombstoned_ledgers") or {},
            tool_stats=inputs.get("tool_stats") or {},
            habits=inputs.get("habits") or (),
            hard_rules=inputs.get("hard_rules") or (),
            eligible=inputs.get("eligible"),
            bench_projects=self._bench_projects, confirmed_projects=self._confirmed_projects,
            tz_offset_s=self._tz_offset_s,
            hub_watermark=(datetime.fromtimestamp(max_ms / 1000, tz=UTC) if max_ms else None),
        )
        path = self.behavior_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        written = write_if_changed(path, behavior.to_json())
        self._bh_state["watermark_ms"] = max(prev_wm, max_ms)
        self._bh_status = {
            "flash_lag_s": int(lag_s), "lag_bound_s": FLASH_LAG_BOUND_S,
            "lag_exceeded": bool(lag_s > FLASH_LAG_BOUND_S),
            "computed_at": now_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "hub_watermark": behavior.meta.hub_watermark,
            "turns_folded": int(new_n), "turns_total": len(self._bh_keys),
            "agents": len(behavior.agents), "projects": len(behavior.projects),
            "pass_ms": int((time.perf_counter() - t0) * 1000),
        }
        if lag_s > FLASH_LAG_BOUND_S:
            logger.warning("flash_lag_exceeded", flash_lag_s=int(lag_s),
                           bound_s=FLASH_LAG_BOUND_S, turns_folded=new_n)
        logger.info("flash_daemon_behavior_materialized", written=written, turns_folded=new_n,
                    turns_total=len(self._bh_keys), flash_lag_s=int(lag_s),
                    agents=len(behavior.agents), projects=len(behavior.projects),
                    model_view_chars=len(model_view_json(behavior.model_view())),
                    pass_ms=self._bh_status["pass_ms"])
        self._save_state()
        return written

def hub_ledger_source(hub: object) -> LedgerSource:
    """Hub 管理事件 → (账本池, scope 绑定)。secondary 模式每次先追新。

    事件过滤用 core 的 `LEDGER_EVENT_TYPES`（与 agency 池装载同一份判据，
    §3.2b——两边各抄一份迟早分叉）。
    """
    from bladex_core.ledger import replay_ledger_events

    from bladex_proxy.ledger_events import load_ledger_events

    def _source() -> tuple[dict[str, Ledger], dict[str, str]]:
        try:
            hub.catch_up()  # type: ignore[attr-defined]
        except Exception as e:  # noqa: BLE001 —— 追新失败用旧快照，下轮再追
            logger.debug("flash_daemon_catch_up_failed", error=str(e))
        # 2026-09-01：装载收进 `ledger_events.load_ledger_events`
        # （类型判据 + 账本墓碑 + ts，三处共用一份）。
        return replay_ledger_events(load_ledger_events(hub))

    return _source


def hub_tombstoned_source(hub: object) -> Callable[[], set[str]]:
    """MQ-F3：Hub 墓碑集合源（与 `load_ledger_events` 用同一份判据
    `ledger_events.tombstoned_ledger_ids`——两边各抄一份迟早分叉）。"""
    from bladex_proxy.ledger_events import tombstoned_ledger_ids

    def _source() -> set[str]:
        try:
            hub.catch_up()  # type: ignore[attr-defined]
        except Exception as e:  # noqa: BLE001
            logger.debug("flash_daemon_catch_up_failed", error=str(e))
        return tombstoned_ledger_ids(hub)

    return _source


def hub_behavior_source(hub: object, *, index_path: str | None = None,
                        eligible: Callable[[str], bool] | None = None,
                        hard_rules: Callable[[], list[str]] | None = None,
                        tools_top_n: int = 5) -> Callable[[], dict]:
    """Hub + Index → `behavior.json` 的数据档（F2；零 LLM、零 `hub.get()`——补遗 G1）。

    一趟 `scan_meta_snapshot`（turn 元数据 + 墓碑 + 管理事件同一趟），账本事件装载
    复用 `ledger_events.load_ledger_events`（快照冒充 hub），Index 只读工具统计
    （`profile/tool/<base>/<tool>`，E7.4 原样）与 profile lane 的 fact 引用（`habits_ref`）。
    Index 打不开（首装 / consolidator 未跑过）⇒ 那两段留空，不阻断。
    `habits` 按 `fact_count` 缓存：计数没变就不重读 11k 条 fact。
    """
    from bladex_core.behavior import HabitRef
    from bladex_core.fact import fact_lane

    from bladex_proxy.ledger_events import load_ledger_events, tombstoned_ledgers

    cache: dict = {"fact_count": -1, "habits": []}

    def _index_part() -> tuple[dict, list]:
        if not index_path:
            return {}, []
        from bladex_proxy.storage.memory_index import MemoryIndex
        try:
            idx = MemoryIndex(index_path, embedder=None, read_only=True)
            idx.open()
        except Exception as e:  # noqa: BLE001 —— Index 缺席 = 两段留空
            logger.debug("flash_behavior_index_unavailable", error=str(e))
            return {}, []
        try:
            stats = {b: idx.get_tool_stats(b, top=tools_top_n) for b in idx.list_tool_agents()}
            n = idx.fact_count()
            if n != cache["fact_count"]:
                habits = []
                for f in idx.all_facts():
                    if fact_lane(f.tags, f.item_kind) != "profile":
                        continue
                    if f.t_invalid is not None or f.superseded_by:
                        continue
                    habits.append(HabitRef(fact_id=f.id, strength=int(f.strength),
                                           importance=float(f.importance), text=f.content))
                cache["fact_count"], cache["habits"] = n, habits
            return stats, list(cache["habits"])
        except Exception as e:  # noqa: BLE001
            logger.warning("flash_behavior_index_read_failed", error=str(e))
            return {}, []
        finally:
            try:
                idx.close()
            except Exception:  # noqa: BLE001,S110
                pass

    def _source() -> dict:
        try:
            hub.catch_up()  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001,S110
            pass
        snap = hub.scan_meta_snapshot()  # type: ignore[attr-defined]
        events = load_ledger_events(snap, keep_tombstoned_pool=True)
        tool_stats, habits = _index_part()
        return {
            "rows": snap.rows,
            "tombstone_hash": snap.tombstone_hash(),
            "ledger_events": events,
            "tombstoned_ledgers": tombstoned_ledgers(snap),
            "tool_stats": tool_stats,
            "habits": habits,
            "hard_rules": list(hard_rules()) if hard_rules else [],
            "eligible": eligible,
        }

    return _source


def browsing_admittance(rules: list, bindings: list[dict]):
    """浏览面准入判据①（Jason 2026-08-29 拍板；批 F4 起唯一消费方 = 行为画像
    `hub_behavior_source(eligible=...)`，目录树与名册已删）：
    **有独立 agent 声明才算 agent**。

    - 规则库**非 auxiliary** 条目 = 独立声明（base 进；`profile_aware` 的
      `base:profile` 变体各自算独立 agent——各有自己的 profile 与配置文件）；
    - `subagent_headers` 派生的 `base:<值>`（如 codex:guardian）不进——规则库
      本就标记子代理"不是任务账本的作者"，其内容也不进 Index；
    - 显式 `X-Agent-ID` 声明（registry trigger=`header:X-Agent-ID` 原文）=
      用户亲手起的名，进；
    - unknown-* 未认领桶不进。

    判据②（真以该身份干过活 = 桶内有非 aux 轮）原在 `hub_tree_source` 抽样实现，
    随目录树一起删除（批 F4）；行为画像按轮计 `aux`，不另设判据②。
    记忆命名空间始终按完整 agent_id 隔离（身份层设计，不受浏览面影响）。
    """
    declared = {r.agent_id for r in rules if not r.auxiliary}
    profile_bases = {r.agent_id for r in rules
                     if r.profile_aware and not r.auxiliary}
    sub_bases = {r.agent_id for r in rules if r.subagent_headers}
    explicit = {str(b.get("agent_id", "")) for b in bindings
                if str(b.get("trigger", "")).startswith("header:X-Agent-ID")}

    def _ok(agent_id: str) -> bool:
        if not agent_id or agent_id.startswith("unknown"):
            return False
        if agent_id in explicit:
            return True
        base, _, suffix = agent_id.partition(":")
        if not suffix:
            return base in declared
        if base in sub_bases:
            # 同 base 若日后兼有 profile_aware，id 形态无法区分子代理与
            # profile——保守排除（宁缺勿混入子代理），届时需显式消歧。
            return False
        return base in profile_bases
    return _ok


def admin_emit(base_url: str, admin_key: str = "") -> EmitEvent:
    """管理事件写侧 = POST proxy 的 `/admin/ledgers/user-edit`（正门）。

    🔴 为什么不直接写 Hub：proxy 常驻持有 Hub 写锁（写权限矩阵，ADR-0032 §2.3
    ——Hub 唯一写者是 proxy，管理事件例外**也经 proxy 的门**进，与 CLI/dashboard
    同一条路）。失败**抛出**——调用方（`_capture_user_edits`）据此不更新快照、
    下轮重试；成功返回响应 JSON（含采纳后的池版 `ledger`，MQ-L47）；个人模式未配
    admin key 时不带 Authorization（require_admin_key 的回落语义，ADR-0027 §2.2）。
    """
    import json as _json
    import urllib.request

    def _emit(etype: str, payload: dict) -> dict | None:
        headers = {"Content-Type": "application/json"}
        if admin_key:
            headers["Authorization"] = f"Bearer {admin_key}"
        req = urllib.request.Request(
            base_url.rstrip("/") + "/admin/ledgers/user-edit",
            data=_json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310 —— 本机 proxy
            if resp.status != 200:
                raise RuntimeError(f"admin user-edit HTTP {resp.status}")
            _read = getattr(resp, "read", None)
            raw = _read() if callable(_read) else b""
        # MQ-L47：响应体（含采纳后的池版 `ledger`）交回调用方当轮写回；
        # 解析不了不算失败（采纳已 200），返回 None 让调用方走挂起路径。
        try:
            body = _json.loads(raw.decode("utf-8")) if raw else None
        except (ValueError, AttributeError):
            body = None
        return body if isinstance(body, dict) else None

    return _emit


#: proxy 就绪轮询间隔（秒）。不进 flags 表：它不是"待标定"的旋钮而是探针粒度，
#: 上限才是可配的那一个（`BLADEX_FLASH_WAIT_PROXY_S`）。
_PROXY_POLL_INTERVAL_S = 0.5


def _proxy_healthy(base_url: str, timeout: float = 2.0) -> bool:
    """proxy `/health` 探针（2xx = 就绪）。任何网络/解析异常一律读作"还没起"。"""
    import urllib.error
    import urllib.request

    req = urllib.request.Request(base_url.rstrip("/") + "/health")  # noqa: S310 —— 本机探针
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return 200 <= int(getattr(resp, "status", 0) or 0) < 300
    except (urllib.error.URLError, OSError, ValueError):
        return False


def wait_for_proxy(base_url: str, *, max_wait_s: float | None = None,
                   probe: Callable[[str], bool] | None = None,
                   sleep: Callable[[float], None] = time.sleep,
                   now: Callable[[], float] = time.monotonic) -> float | None:
    """等 proxy 就绪；返回等待秒数，超时返回 `None`（**不阻塞，照旧起**）。

    ## MQ-L50：为什么修 daemon 而不是改 cli 的起进程顺序

    `cli/lifecycle.py` 的顺序是 embed → consolidator → flash → proxy，于是
    `startup_check()` 的停机期直编补采 POST 撞 `Connection refused`
    （09-06 两次 + 09-07 三次 = 5/5 重启全撞），落回重发通道 30s 后才成。
    L51 修后它是**唯一**还会制造 pending 的来源。

    修在 daemon 侧：**对 proxy 就绪负责的是要 POST 的那一方**——与 `admin_emit`
    同一层（它同样只知道 `base_url`，不知道谁先起）。改 cli 顺序则要求
    "起进程的人记得 flash 依赖 proxy"，那是把一个可自证的依赖藏进调用顺序里。

    超时**不阻塞**：Flash 物化本身不需要 proxy（红线 1，真相在 Hub），
    只有直编采纳需要，而那一路本就有重发通道兜底。宁可晚采不可不起。
    """
    from bladex_core.flags import flag_number  # noqa: PLC0415 —— 入口读配置（红线 4）
    if max_wait_s is None:
        max_wait_s = flag_number("BLADEX_FLASH_WAIT_PROXY_S")
    _probe = probe or _proxy_healthy
    t0 = now()
    while True:
        if _probe(base_url):
            waited = round(now() - t0, 1)
            logger.info("flash_daemon_proxy_ready", waited_s=waited, proxy=base_url)
            return waited
        if now() - t0 >= max_wait_s:
            logger.warning("flash_daemon_proxy_not_ready",
                           waited_s=round(now() - t0, 1), proxy=base_url,
                           max_wait_s=max_wait_s)
            return None
        sleep(_PROXY_POLL_INTERVAL_S)


def main() -> int:  # pragma: no cover —— 进程入口；接线件各有单测
    """`bladex-flash` console entry。env/配置读取只在这里（红线 4）。"""
    import argparse

    # 与 consolidator 同一套入口自举（env 加载/切部署根都是入口副作用，
    # 复用而非再抄一份——那两个函数是纯入口帮手，import 无副作用）。
    ap = argparse.ArgumentParser(description="BladeX Memory Flash maintenance daemon")
    ap.add_argument("--bladex-home", default="",
                    help="deployment root for this daemon (passed by `bladex start`; "
                         "empty = BLADEX_HOME / config discovery)")
    ap.add_argument("--interval", type=float, default=60.0,
                    help="fallback poll interval (seconds) -- used ONLY when the "
                         "Redis wake stream is unavailable (degraded mode) and as "
                         "the block timeout upper bound. Normal operation is "
                         "wake-driven (zero polling). NOTE: cli._start_flash "
                         "passes this explicitly -- the two defaults must stay "
                         "equal (rigid rule 12; the consolidator --concurrency "
                         "incident)")
    ap.add_argument("--once", action="store_true", help="run one round and exit")
    ap.add_argument("--principal", default="local",
                    help="principal id (personal mode: 'local')")
    ap.add_argument("--proxy-url", default="",
                    help="proxy base url for admin writes (default http://127.0.0.1:<port>)")
    args = ap.parse_args()
    # 🔴 MQ-A78 复核：与 consolidator 同序——先钉 BLADEX_HOME，再加载 env / chdir。
    if args.bladex_home:
        os.environ["BLADEX_HOME"] = os.path.abspath(os.path.expanduser(args.bladex_home))
    from bladex_proxy.consolidator import _chdir_to_deployment_root, _load_env_file
    _load_env_file()
    _chdir_to_deployment_root()

    from bladex_proxy.modules import module_enabled
    if not module_enabled("flash"):
        print("bladex-flash: module 'flash' is disabled -- set BLADEX_MODULE_FLASH=1 "
              "(module registry is the single source of truth; refusing to run a "
              "daemon whose module is off).")
        return 2

    from bladex_proxy.cli import _admin_key
    from bladex_proxy.config import ProxyConfig, resolve_flash_path
    from bladex_proxy.storage.memory_hub import MemoryHub
    cfg = ProxyConfig()
    base_url = args.proxy_url or f"http://127.0.0.1:{cfg.port}"
    # 🔴 secondary 副本目录独立：与 consolidator 的 `<hub>_secondary` 分开，
    # 两个 secondary 共用一个副本目录会互踩 MANIFEST。
    hub = MemoryHub(cfg.rocksdb_path, secondary=True,
                    secondary_dir=str(cfg.rocksdb_path) + "_secondary_flash")
    open_hub_with_startup_window(hub)
    import json as _json
    from datetime import datetime as _dt
    from datetime import timedelta as _td

    from bladex_proxy.agent_registry import cache_path as _reg_cache
    from bladex_proxy.agent_rules import load_agent_rules
    try:
        with open(_reg_cache(), encoding="utf-8") as _f:
            _bindings = _json.load(_f).get("bindings") or []
    except OSError:
        _bindings = []
    _rules = load_agent_rules()
    _eligible = browsing_admittance(_rules, _bindings)

    daemon = FlashDaemon(root=resolve_flash_path(), principal=args.principal,
                         ledger_source=hub_ledger_source(hub),
                         tombstoned_source=hub_tombstoned_source(hub),
                         emit_admin_event=admin_emit(base_url, _admin_key()),
                         # F2：行为画像轨道（批 F4 起是本进程唯一的 300s 节流轨道）
                         behavior_source=hub_behavior_source(
                             hub, index_path=str(cfg.index_path), eligible=_eligible,
                             hard_rules=lambda: list(cfg.effective_hard_rules or [])),
                         # 存 UTC 算本地（MQ-L46）：偏移在入口取一次，daemon 内不读时钟设置
                         tz_offset_s=int((_dt.now().astimezone().utcoffset()
                                          or _td(0)).total_seconds()))
    # Z1.2（MQ-W18）：硬规则热加载。behavior_source 的 `cfg.effective_hard_rules` 每个画像
    # 周期现读 ⇒ arm 之后 `.env` / Hub `hard_rules_set` 的改动进下一次 behavior.json。
    # 画像轨道只在"有新流量"时才重扫（`_behavior_dirty`）——规则变了也要算一次"新东西"，
    # 否则没有对话时它永远进不了 behavior.json：on_change 标脏。
    from bladex_proxy import hard_rules as _hard_rules
    _hard_rules.arm("flash_daemon", hub=_hard_rules.hub_reader(hub, catch_up=True),
                    initial=cfg.effective_hard_rules,
                    on_change=lambda _rules: daemon.push("hard_rules"))
    logger.info("flash_daemon_started", root=resolve_flash_path(),
                principal=args.principal, fallback_poll_s=args.interval,
                proxy=base_url, wake="redis-stream")
    from bladex_proxy.flash_wake import WakeConsumer
    consumer = WakeConsumer(cfg.redis_url)
    hook_last: dict[str, float] = {}
    try:
        # 🔴 MQ-L50：先等 proxy 就绪再自检——`startup_check()` 的直编补采要 POST
        # `/admin/ledgers/user-edit`，而 cli 的起进程顺序把 flash 排在 proxy 之前。
        # 超时照旧起（见 `wait_for_proxy` docstring）。
        wait_for_proxy(base_url)
        # 启动自检（骨架重建 + 停机期直编识别）+ 首轮全量物化
        stats = daemon.startup_check()
        if any(stats.values()):
            logger.info("flash_daemon_startup", **stats)
        if args.once:
            return 0
        while True:
            # 阻塞等门铃；有待扫的画像就把超时压到画像到窗的时刻。
            timeout = args.interval
            tw = daemon.behavior_wait_s()
            if tw is not None:
                timeout = min(timeout, max(tw, 1.0))
            pw = daemon.pending_wait_s()          # E0.1b：有待重发的旧路径直编就别死等门铃
            if pw is not None:
                timeout = min(timeout, max(pw, 1.0))
            woken = consumer.wait(timeout)
            if woken:
                daemon.push()
            if woken or daemon.behavior_wait_s() == 0.0 or daemon.pending_wait_s() == 0.0:
                stats = daemon.run_once()
                if stats.get("written") or stats.get("user_edits"):
                    logger.info("flash_daemon_round", **stats)
            # 预留低频轨道（0.2.0 账本维护 / 0.3.0 Flash→Index 提取）：
            # 钩子非 None 且间隔 >0 才进调度——铁律"无署名消费者的字段
            # 不许生产"，规则未定就不造 flag、不空转。
            now = time.time()
            for name in ("upkeep", "extract"):
                hook = getattr(daemon, f"{name}_hook")
                every = getattr(daemon, f"{name}_every_s")
                if hook is None or every <= 0:
                    continue
                if now - hook_last.get(name, 0.0) >= every:
                    hook_last[name] = now
                    try:
                        hook()
                    except Exception as e:  # noqa: BLE001
                        logger.warning("flash_daemon_hook_failed",
                                       hook=name, error=str(e))
    finally:
        try:
            hub.close()
        except Exception:  # noqa: BLE001,S110
            pass


if __name__ == "__main__":
    raise SystemExit(main())
