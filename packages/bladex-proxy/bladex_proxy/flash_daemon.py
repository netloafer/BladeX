"""Flash 维护进程 —— 单写者物化 + 用户直编捕获（ADR-0032 §5.2；批二卡 V-F2）。

独立后台进程（与 consolidator **不合并**，拍板 §10-5），Flash 树与账本池的
**唯一写者**（用户直编与 dashboard 除外——那两路经本进程捕获为管理事件）。

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
3. **本卡零 LLM**：蒸馏简介（AGENTS.md 一句话介绍等）归 V-F2b 后置卡，
   先立调用预算再开（G15 纪律）。
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
from bladex_core.flash import SCOPE_PERSONAL, write_if_changed
from bladex_core.ledger import Ledger, iso_ms, ledger_md_path, parse_ledger_md, render_ledger_md

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
#: 名册数据源：() -> AgentRow 列表（V-F3；确定性字段，summary 归 V-F2b）。
AgentsSource = Callable[[], list]


class FlashDaemon:
    """单写者物化循环。线程/进程模型：进程内单实例、串行 `run_once`——
    多源写入经 push 收件箱（启用卡接 Hub 通知），物化永远只有这一支笔。"""

    def __init__(self, *, root: str, principal: str, ledger_source: LedgerSource,
                 emit_admin_event: EmitEvent, scope: str = SCOPE_PERSONAL,
                 agents_source: AgentsSource | None = None,
                 tree_source: Callable[[], dict] | None = None,
                 summarize: Callable[[str], str] | None = None,
                 summarize_project: Callable[[str], str] | None = None,
                 profiles_source: Callable[[], tuple] | None = None,
                 tree_every_s: float = 300.0,
                 tombstoned_source: Callable[[], set[str]] | None = None) -> None:
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
        self._agents_source = agents_source
        # V-F3 第二批：目录树数据源（sessions/session→ledger/agent last_seen +
        # 简介源文本）与蒸馏器（V-F2b；None = 不蒸，简介留空）。树物化节流
        # ——sessions 靠全 Hub 扫描导出，不适合 15s 一跑。
        self._tree_source = tree_source
        self._summarize = summarize
        #: 项目简介独立蒸馏器——不得复用 agent 口味的 NAME|ABOUT 提示词
        #: （2026-08-29 live：CLAUDE.md 被当 agent 人设蒸出
        #: 「BladeX Assistant | AI coding agent…」）。缺省 = 不蒸、留空。
        self._summarize_project = summarize_project
        #: () -> (user_md, rules_md, {agent_id: agent_md})——Index 读侧投影源。
        self._profiles_source = profiles_source
        self._tree_every_s = tree_every_s
        self._tree_last = 0.0
        self._summaries: dict[str, str] = {}      # agent -> 一句话（源 hash 缓存落盘）
        self._proj_summaries: dict[str, str] = {}  # pid -> 一句话（同款纪律）
        self._agent_names: dict[str, str] = {}     # agent -> 显示名（简介搭车产出）
        self._dirty = True          # 收件箱有活（首轮恒真）
        self._tree_dirty = True     # 树=会话流量的投影，只在有新流量后重扫
        #: 持久小状态（root/.flash_state.json）：baseline = 每文件"我们上次写下/
        #: 采纳的内容" sha —— 跨重启区分「用户离线改过」vs「投影只是过期」；
        #: adopted = 已采纳、Hub 尚未收敛的直编 sha（重启后恢复零覆盖窗口 +
        #: 重发采纳）。丢了不产生正确性问题，只是无法识别停机期间的直编。
        self._state_path = os.path.join(root, ".flash_state.json")
        self._baseline_sha: dict[str, str] = {}
        self._adopted_sha: dict[str, str] = {}
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
        except (OSError, ValueError):
            pass               # 首次运行/状态损坏：从零开始（见字段注释）

    def _save_state(self) -> None:
        try:
            tmp = self._state_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"baseline": self._baseline_sha,
                           "adopted": self._adopted_sha,
                           "adopted_rev": self._adopted_rev}, f)
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
        self._tree_dirty = True

    def startup_check(self) -> dict[str, int]:
        """启动自检（Jason 2026-08-29 拍板第二条）：骨架缺失即重建 + 识别
        停机期间的用户直编，然后跑一轮全量物化。

        直编识别靠持久 baseline：文件 sha == 上次写下/采纳的 sha ⇒ 只是投影
        （Hub 若前进了就放心重写）；sha 不同 ⇒ 停机期间被人改过 ⇒ 交给
        `_capture_user_edits` 走采纳。没有 baseline 记录的文件当投影处理
        （首次运行没有更好的判据，如实记录在案）。
        """
        from bladex_core.flash_tree import agents_roster_path
        from bladex_core.ledger import ledgers_dir
        for d in (os.path.dirname(agents_roster_path(self._root, self._principal,
                                                     scope=self._scope)),
                  ledgers_dir(self._root, self._principal, scope=self._scope)):
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
            pool, bindings = self._source()
            written = self._materialize(pool)
            removed = self._purge_tombstoned(pool)
            written += self._materialize_roster()
            self._dirty = False
        else:
            pool = bindings = None
        # 树物化按节流独立于账本事件（sessions 来自轮次流量，不只账本）。
        now = time.time()
        if (self._tree_source is not None and self._tree_dirty
                and now - self._tree_last >= self._tree_every_s):
            self._tree_last = now
            self._tree_dirty = False
            if pool is None:
                pool, bindings = self._source()
            written += self._materialize_tree(pool, bindings or {})
            # 树刷新带来 last_seen/简介的新值 ⇒ 名册重渲染（write_if_changed
            # 兜底：没变化就不写；不重跑的话增强列要等下一次账本事件才落盘）。
            written += self._materialize_roster()
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

    def tree_wait_s(self) -> float | None:
        """树物化还差多久到窗（None = 没有待扫的流量）。主循环用它定阻塞时长。"""
        if self._tree_source is None or not self._tree_dirty:
            return None
        return max(0.0, self._tree_every_s - (time.time() - self._tree_last))

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

        删法（跑前登记，与 `_reclaim_dirs` 同一套纪律「Flash 贵在精、投影可删、
        用户内容宁留勿删」）：
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

    #: 项目声明文件（简介蒸馏源，按此序取第一个存在的）。
    _PROJECT_DECL_FILES = ("CLAUDE.md", "AGENTS.md", "README.md")

    def _project_summary(self, pid: str, root: str) -> str:
        """PROJECTS.md 的 about 列：蒸馏项目声明文件头部的一句话。

        与 agent 简介（V-F2b）同一纪律：预算 = 每项目每源变更 1 次（sha 缓存
        落盘 `.project_summaries.json`）；JSON 形态输出判劫持缓存空值不重试；
        失败留空——空着比编一句强。root 读不到/无声明文件 ⇒ 空。
        """
        if not root or self._summarize_project is None:
            return self._proj_summaries.get(pid, "")
        src = ""
        for fn in self._PROJECT_DECL_FILES:
            try:
                fp = os.path.join(root, fn)
                if os.path.isfile(fp):
                    with open(fp, encoding="utf-8", errors="replace") as f:
                        src = f.read(4096)
                    break
            except OSError:
                continue
        if not src:
            return self._proj_summaries.get(pid, "")
        h = hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]
        cache_file = os.path.join(self._root, ".project_summaries.json")
        try:
            with open(cache_file, encoding="utf-8") as f:
                cache = json.load(f)
        except (OSError, ValueError):
            cache = {}
        ent = cache.get(pid) or {}
        if ent.get("sha") == h and ent.get("v") == 2:   # v2=项目口味提示词
            self._proj_summaries[pid] = ent.get("summary", "")
            return self._proj_summaries[pid]
        try:
            out = (self._summarize_project(src) or "").strip()
        except Exception as e:  # noqa: BLE001 —— 失败留空，源变更前不重试
            logger.warning("flash_daemon_project_summary_failed",
                           project=pid, error=str(e))
            out = ""
        if out.startswith("{") or out.startswith("["):
            logger.warning("flash_daemon_summary_hijacked", project=pid)
            out = ""      # 内嵌指令面同款防护（D7/名册简介先例）
        cache[pid] = {"sha": h, "summary": out, "v": 2}
        try:
            tmp = cache_file + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False)
            os.replace(tmp, cache_file)
        except OSError:
            pass
        self._proj_summaries[pid] = out
        return out

    def _materialize_profiles(self) -> int:
        """USER.md / RULES.md（scope 级）与各 agent 目录 AGENT.md 的**读侧投影**
        （ADR §5.1"从会话提取的 AGENT.md/USER.md/RULES.md"——真身与生产者都在
        Memory Index（E7.3 画像/习惯）+ 配置硬规则（RULES.md 与 prefetch 同源），Flash 只投影，零 LLM）。
        源不可用（Index 未接/读失败）⇒ 不写不删，既有投影原样保留。"""
        if self._profiles_source is None:
            return 0
        try:
            user_md, rules_md, agent_mds = self._profiles_source()
        except Exception as e:  # noqa: BLE001 —— 投影源失败不连累其它物化
            logger.warning("flash_daemon_profiles_source_failed", error=str(e))
            return 0
        from bladex_core.flash import scope_dir
        base = scope_dir(self._root, self._principal, self._scope)
        written = 0
        mark = "<!-- BladeX Memory Flash · projection of Memory Index -->\n\n"
        if user_md:
            written += write_if_changed(os.path.join(base, "USER.md"),
                                        mark + user_md)
        if rules_md:
            written += write_if_changed(os.path.join(base, "RULES.md"),
                                        mark + rules_md)
        from bladex_core.flash_tree import agent_dir
        for agent_id, md in (agent_mds or {}).items():
            if not md:
                continue
            adir = agent_dir(self._root, self._principal, agent_id,
                             scope=self._scope)
            if os.path.isdir(adir):    # 只投影给保鲜窗口内存在的 agent 目录
                written += write_if_changed(os.path.join(adir, "AGENT.md"),
                                            mark + md)
        return written

    def _materialize_roster(self) -> int:
        """AGENTS.md 名册（V-F3）。**不进 `_snapshots`**：名册是机器文件
        （`_MACHINE` 标记），不在直编捕获范围——注册进快照会让每次名册更新
        都被 `parse_ledger_md` 试啃一遍、逐轮告警（机器文件与账本文件是
        两类投影，捕获判据只属于后者）。"""
        if self._agents_source is None:
            return 0
        try:
            rows = self._agents_source()
        except Exception as e:  # noqa: BLE001 —— 名册源失败不阻断账本物化
            logger.warning("flash_daemon_agents_source_failed", error=str(e))
            return 0
        from dataclasses import replace as _dc_replace

        from bladex_core.flash_tree import agents_roster_path, render_agents_roster
        # V-F3 第二批：确定性补 last_seen（树数据源的 Hub 扫描）+ V-F2b 简介。
        # MQ-F1：行来源扩到 bindings ∪ 简介 ∪ 规则库声明后，判据②「真干过活」
        # 只能由树数据源回答（`agent_last_seen` 有它 = Hub 里有非 aux 轮；
        # `excluded` = 抽样全 aux 的内部功能桶）。规则库声明了但从没跑过流量的
        # agent（cursor/openclaw）不进名册。树还没扫过时不写——写一版没过判据②
        # 的名册就是把错表再注一轮（08-30 ~ 09-03 的事故形态）。
        if self._tree_source is not None and self._tree_cache is None:
            return 0
        tree = self._tree_cache or {}
        last_seen = tree.get("agent_last_seen") or {}
        _excluded = set(tree.get("excluded") or [])
        rows = [r for r in rows if r.agent_id not in _excluded]
        if self._tree_source is not None:
            rows = [r for r in rows if r.agent_id in last_seen]
        rows = [_dc_replace(r, name=r.name or self._agent_names.get(r.agent_id, ""),
                            last_seen=last_seen.get(r.agent_id, r.last_seen),
                            summary=self._summaries.get(r.agent_id, r.summary))
                for r in rows]
        path = agents_roster_path(self._root, self._principal, scope=self._scope)
        return write_if_changed(path, render_agents_roster(rows))

    _tree_cache: dict | None = None

    def _materialize_tree(self, pool: dict[str, Ledger], bindings: dict[str, str]) -> int:
        """agent/global/session 目录树 + 三张清单（V-F3 第二批，Jason 拍板：
        基础目录机制先立，账本可以为空）。全部 machine 文件：write_if_changed、
        不进直编捕获快照。"""
        try:
            tree = self._tree_source() if self._tree_source else {}
        except Exception as e:  # noqa: BLE001 —— 树源失败不连累账本池
            logger.warning("flash_daemon_tree_source_failed", error=str(e))
            return 0
        self._tree_cache = tree
        self._refresh_summaries(tree)
        from bladex_core.flash_tree import (
            LedgerRow,
            ProjectRow,
            SessionRow,
            ledgers_list_path,
            projects_roster_path,
            render_ledgers_list,
            render_projects_roster,
            render_sessions_roster,
            sessions_roster_path,
        )
        from bladex_core.ledger_runtime import activation_scope
        written = 0
        sessions: dict[str, list[dict]] = tree.get("sessions") or {}
        sess_ledgers: dict[tuple[str, str], list[str]] = tree.get("session_ledgers") or {}
        sess_project: dict = tree.get("session_project") or {}
        # ── 保鲜窗口（2026-08-29 Jason 拍板：Flash 贵在精——目录数量也是
        # 维护对象；默认值暂定按真实流量调，flags.py 单一真相源）──
        from bladex_core.flags import flag_number
        _sess_cap = int(flag_number("BLADEX_FLASH_SESSIONS_PER_PROJECT"))
        _proj_cap = int(flag_number("BLADEX_FLASH_PROJECTS_PER_AGENT"))
        _proj_days = flag_number("BLADEX_FLASH_PROJECT_ACTIVE_DAYS")
        _agent_days = flag_number("BLADEX_FLASH_AGENT_RETIRE_DAYS")

        def _age_days(last_at: str) -> float:
            from datetime import datetime
            try:
                dt = datetime.strptime(last_at, "%Y-%m-%d %H:%M").replace(
                    tzinfo=UTC)
                return (datetime.now(UTC) - dt).total_seconds() / 86400
            except ValueError:
                return 0.0          # 解析不了当新鲜——宁留勿删

        def _proj_dir(pid: str, pname: str) -> str:
            # 目录名 = 人读名-短hash（`BladeX-10ad1f`）：可读 + 同名项目不撞；
            # 身份永远是 project_id，目录名只是投影。
            from bladex_core.flash import safe_component
            return safe_component(f"{pname or 'proj'}-{pid[2:8]}")

        keep: dict[str, dict[str, set]] = {}     # agent_dir名 -> {proj_dir名: {sess...}}
        from bladex_core.flash import safe_component as _safec
        for agent_id, rows in sorted(sessions.items()):
            # agent 退休窗口：超期无流量不再物化（目录随回收走，名册行保留）
            if rows and _age_days(rows[0].get("last_at", "")) > _agent_days > 0:
                logger.info("flash_agent_retired", agent=agent_id)
                continue
            # 会话按 project 分组（无归属 = global——历史轮次没有 project_id）
            groups: dict[str, list[dict]] = {}
            pnames: dict[str, str] = {}
            for r in rows:
                pid, pname, proot = sess_project.get(
                    (agent_id, r["session_id"]), ("", "", ""))
                gdir = _proj_dir(pid, pname) if pid else "global"
                groups.setdefault(gdir, []).append(r)
                if pid:
                    pnames[gdir] = (pid, pname, proot)
            # project 活跃窗口 ∧ 数量上限（global 恒保留）；session 每组截前 N
            active = {g for g, rs in groups.items()
                      if g == "global"
                      or (_proj_days <= 0
                          or _age_days(rs[0].get("last_at", "")) <= _proj_days)}
            ranked = sorted((g for g in groups if g != "global" and g in active),
                            key=lambda g: _age_days(groups[g][0].get("last_at", "")))
            kept_projs = set(ranked[:_proj_cap] if _proj_cap > 0 else ranked)
            kept_projs |= ({"global"} if "global" in groups else set())
            groups = {g: rs[:_sess_cap] if _sess_cap > 0 else rs
                      for g, rs in groups.items() if g in kept_projs}
            pnames = {g: v for g, v in pnames.items() if g in kept_projs}
            keep[_safec(agent_id)] = {g: {r["session_id"] for r in rs}
                                      for g, rs in groups.items()}
            prows = [ProjectRow(project_id="global", name="Global",
                                source="fallback")]
            for g in sorted(pnames):
                pid, pname, proot = pnames[g]
                para = self._project_summary(pid, proot)
                one_liner = (para.split("。")[0].split(". ")[0])[:80] if para else ""
                grs = groups.get(g, [])
                prows.append(ProjectRow(
                    project_id=pid, name=pname or g, source="identity",
                    summary=one_liner,
                    created_at=min((r.get("first_at", "") for r in grs),
                                   key=iso_ms, default=""),
                    updated_at=max((r.get("last_at", "") for r in grs),
                                   key=iso_ms, default="")))
                if para:
                    # Project 级 PROJECT.md（ADR §5.1）：蒸馏段落全文，机器文件。
                    from bladex_core.flash_tree import project_md_path
                    written += write_if_changed(
                        project_md_path(self._root, self._principal, agent_id,
                                        g, scope=self._scope),
                        f"# {pname or g}\n\n<!-- BladeX Memory Flash · "
                        f"machine-rendered project brief -->\n\n{para}\n")
            written += write_if_changed(
                projects_roster_path(self._root, self._principal, agent_id,
                                     scope=self._scope),
                render_projects_roster(agent_id, prows))
            active = bindings.get(activation_scope(agent_id, ""), "")
            for gdir, grows in sorted(groups.items()):
                gtitle = pnames.get(gdir, ("", "Global"))[1] or "Global"
                srow = [SessionRow(session_id=r["session_id"],
                                   name=r.get("name", ""),
                                   first_at=r.get("first_at", ""),
                                   last_at=r.get("last_at", "")) for r in grows]
                written += write_if_changed(
                    sessions_roster_path(self._root, self._principal, agent_id,
                                         gdir, scope=self._scope),
                    render_sessions_roster(gtitle, srow))
                for r in grows:
                    lids = sess_ledgers.get((agent_id, r["session_id"]), [])
                    lrows = []
                    for lid in lids:
                        led = pool.get(lid)
                        lrows.append(LedgerRow(
                            ledger_id=lid,
                            title=getattr(led, "title", "") if led else "",
                            status=str(getattr(getattr(led, "status", ""), "value",
                                               getattr(led, "status", ""))) if led else "?",
                            created_at=getattr(led, "created_at", "") if led else "",
                            updated_at=getattr(led, "updated_at", "") if led else ""))
                    written += write_if_changed(
                        ledgers_list_path(self._root, self._principal, agent_id,
                                          gdir, r["session_id"], scope=self._scope),
                        render_ledgers_list(lrows, active_ledger_id=active))
        written += self._reclaim_dirs(keep)
        written += self._materialize_profiles()
        if written:
            logger.info("flash_daemon_tree_materialized", files=written,
                        agents=len(sessions))
        return written

    #: 回收豁免：scope 级的池与机器清单、名册。
    _RECLAIM_KEEP_TOP = frozenset({"ledgers", "AGENTS.md", "USER.md", "RULES.md"})

    def _reclaim_dirs(self, keep: dict[str, dict[str, set]]) -> int:
        """保鲜回收（Jason 拍板：Flash 贵在精）。删除窗口外的 agent/project/
        session 目录——文件是投影，历史在 Hub/Index 永远可查。

        安全三条：只动 scope 目录下的机器目录；**含用户手写 `*.local.md` 或
        待收敛直编的目录跳过**（宁留勿删，`flash_dir_skip_user_content` 可
        grep）；每次删除记 `flash_dir_reclaimed`。
        """
        import shutil

        from bladex_core.flash import scope_dir
        base = scope_dir(self._root, self._principal, self._scope)
        removed = 0

        def _has_user_content(path: str) -> bool:
            for dirpath, _dirs, files in os.walk(path):
                for fn in files:
                    full = os.path.join(dirpath, fn)
                    if fn.endswith(".local.md") or full in self._pending_edits:
                        return True
            return False

        def _rm(path: str, level: str) -> int:
            if _has_user_content(path):
                logger.warning("flash_dir_skip_user_content", path=path,
                               level=level)
                return 0
            try:
                shutil.rmtree(path)
                logger.info("flash_dir_reclaimed", path=path, level=level)
                return 1
            except OSError as e:
                logger.warning("flash_dir_reclaim_failed", path=path,
                               error=str(e))
                return 0

        try:
            top = os.listdir(base)
        except OSError:
            return 0
        for name in top:
            full = os.path.join(base, name)
            if name in self._RECLAIM_KEEP_TOP or not os.path.isdir(full):
                continue
            if name not in keep:
                removed += _rm(full, "agent")
                continue
            projs = keep[name]
            for pn in os.listdir(full):
                pfull = os.path.join(full, pn)
                if not os.path.isdir(pfull):
                    continue
                if pn not in projs:
                    removed += _rm(pfull, "project")
                    continue
                for sn in os.listdir(pfull):
                    sfull = os.path.join(pfull, sn)
                    if os.path.isdir(sfull) and sn not in projs[pn]:
                        removed += _rm(sfull, "session")
        return removed

    def _refresh_summaries(self, tree: dict) -> None:
        """V-F2b：一句话简介——**预算 = 每 agent 每源变更 1 次**，源 sha 缓存落盘。

        源 = 该 agent 最新 system prompt 摘录（树源提供，前 4KB）。失败留空、
        告警、不重试到下轮源变更（省调用不省诚实：空着比编一句强）。
        """
        if self._summarize is None:
            return
        import hashlib
        import json as _json
        cache_file = os.path.join(self._root, ".agent_summaries.json")
        try:
            with open(cache_file, encoding="utf-8") as f:
                cache = _json.load(f)
        except OSError:
            cache = {}
        changed = False
        for agent_id, src in (tree.get("agent_prompts") or {}).items():
            if not src:
                continue
            h = hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]
            hit = cache.get(agent_id)
            if hit and hit.get("hash") == h and "name" in hit:
                self._summaries[agent_id] = hit.get("summary", "")
                self._agent_names[agent_id] = hit.get("name", "")
                continue
            try:
                line = str(self._summarize(src)).strip().splitlines()[0][:120]
            except Exception as e:  # noqa: BLE001
                logger.warning("flash_daemon_summarize_failed",
                               agent=agent_id, error=str(e))
                continue
            if line[:1] in "{[":
                # 源文本的内嵌指令劫持了输出（live 病例：{"title": ...}）。
                # 缓存空值：同源不重试（重试只会再被劫持），源变更后自然重蒸。
                logger.warning("flash_daemon_summary_hijacked",
                               agent=agent_id, head=line[:40])
                line = ""
            # 双产出解析：'NAME | ABOUT'；不带分隔符 = 整行当 about、name 空
            # （弱模型格式服从性有限——解析失败降级，不重试不编造）。
            name = ""
            if "|" in line:
                name, _, line = line.partition("|")
                name, line = name.strip()[:40], line.strip()
            self._summaries[agent_id] = line
            self._agent_names[agent_id] = name
            cache[agent_id] = {"hash": h, "summary": line, "name": name}
            changed = True
            logger.info("flash_daemon_agent_summarized", agent=agent_id)
        if changed:
            try:
                os.makedirs(self._root, exist_ok=True)
                with open(cache_file, "w", encoding="utf-8") as f:
                    _json.dump(cache, f, ensure_ascii=False, indent=1)
            except OSError as e:
                logger.warning("flash_daemon_summary_cache_write_failed", error=str(e))


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


def registry_agents_source(cache_file: str, *, eligible=None,
                           summaries_file: str | None = None,
                           declared: tuple[str, ...] | list[str] = ()) -> AgentsSource:
    """名册行来源（V-F3；只读，确定性字段）。

    🔴 MQ-F1（2026-09-03）：行集合曾**只取 registry bindings**，而 `claude-code` /
    `codex` 靠 `X-Agent-ID` 直认、`dsh` 靠规则库指纹——都不进 bindings ⇒ live
    名册三行全是构造件/Pi，真实交接主体一个不在，稳定层每轮注入的是这张错表；
    简介层 `.agent_summaries.json` 却有 9 个 agent。**行来源 population 必须
    ⊇ 简介来源 population**，故行集合 = bindings ∪ `.agent_summaries.json` 的
    agent ∪ 规则库声明 agent（`declared`），再过同一把 `eligible`
    （`browsing_admittance` 判据①：有独立声明）。判据②「真干过活」在
    `FlashDaemon._materialize_roster` 用树数据源的 `agent_last_seen` 过。

    - `pending` 里的 `unknown-*` 指纹桶是待认领噪声，不进名册；
    - 同 agent 多 origin_key 去重，first_seen 取最早 `identified_at`；
      非 bindings 来源的行 first_seen 留空（名册诚实于数据源）；
    - `last_seen` 缓存里没有（运行期数据），留空——不用 identified_at 冒充
      "最后活跃"；`summary` 归 V-F2b（先立 LLM 预算）。
    """
    import json as _json
    from datetime import datetime

    from bladex_core.flash_tree import AgentRow

    def _load(path: str | None) -> dict:
        if not path:
            return {}
        try:
            with open(path, encoding="utf-8") as f:
                data = _json.load(f)
        except (OSError, ValueError):
            return {}          # 缓存还没建（首次运行）= 没有 agent，不是错误
        return data if isinstance(data, dict) else {}

    def _admit(aid: str) -> bool:
        if not aid or aid.startswith("unknown"):
            return False
        if eligible is not None and not eligible(aid):
            return False       # 判据①：子代理派生等无独立声明的 id 不进名册
        return True

    def _source() -> list:
        first: dict[str, float] = {}
        for b in _load(cache_file).get("bindings") or []:
            aid = str(b.get("agent_id") or "")
            if not _admit(aid):
                continue
            ts = float(b.get("identified_at") or 0.0)
            if aid not in first or (ts and ts < first[aid]):
                first[aid] = ts
        # 简介来源 ∪ 规则库声明：bindings 没有的 agent 补进来（first_seen 空）。
        extra = [str(a) for a in _load(summaries_file)] + [str(a) for a in declared]
        for aid in extra:
            if aid not in first and _admit(aid):
                first[aid] = 0.0
        rows = []
        # 有 identified_at 的按时间排，没有的排后并按 id 稳定排序（渲染确定性）。
        for aid in sorted(first, key=lambda a: (first[a] == 0.0, first[a], a)):
            seen = (datetime.fromtimestamp(first[aid], tz=UTC)
                    .strftime("%Y-%m-%d") if first[aid] else "")
            rows.append(AgentRow(agent_id=aid, first_seen=seen))
        return rows

    return _source


def browsing_admittance(rules: list, bindings: list[dict]):
    """浏览面（目录树/名册）准入判据①（Jason 2026-08-29 拍板）：
    **有独立 agent 声明才算 agent**。

    - 规则库**非 auxiliary** 条目 = 独立声明（base 进；`profile_aware` 的
      `base:profile` 变体各自算独立 agent——各有自己的 profile 与配置文件）；
    - `subagent_headers` 派生的 `base:<值>`（如 codex:guardian）不进——规则库
      本就标记子代理"不是任务账本的作者"，其内容也不进 Index；
    - 显式 `X-Agent-ID` 声明（registry trigger=`header:X-Agent-ID` 原文）=
      用户亲手起的名，进；
    - unknown-* 未认领桶不进。

    判据②（真以该身份干过活 = 桶内有非 aux 轮）在 `hub_tree_source` 抽样实现
    ——裸 `hermes` 桶（checkpoint/命名等内部功能流量）靠它挡。
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


def _msg_text(content) -> str:
    """消息 content 文本视图（str / 分块列表两种形态——与 project_identity
    同款归一；会话名摘录用）。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(x.get("text", "")) for x in content
                        if isinstance(x, dict))
    return ""


#: 判据②「真干过活」的抽样窗口：每 agent 取最近 N 轮，N 轮里**有一条**非 aux 即算干过活。
#: 2026-09-03 B0 live 首验：hermes:default 落榜——它的最近 5 轮全是 checkpoint/标题子调用
#: （aux），而它 176 轮真实流量就在后面。5 是 V-F2b 取简介源时定的（只要一条非 aux 的
#: system prompt），拿来判"干没干过活"太窄；放到 20，代价 = 每 agent 每树 pass 最多 20 次
#: `hub.get`（300s 一次）。仍是"≥1 条非 aux"，不加阈值。
TREE_AUX_SAMPLE_ROUNDS = 20


def hub_tree_source(hub: object, *, max_sessions_per_agent: int = 20,
                    eligible=None,
                    prompt_excerpt_chars: int = 4000,
                    aux_sample_rounds: int = TREE_AUX_SAMPLE_ROUNDS) -> Callable[[], dict]:
    """Hub → 目录树数据（V-F3 第二批）。

    一次 `scan_meta` 导出：每 agent 的 session 列表（**近 N 个**，ADR-0032 §5.2
    "按时间段/数量过滤，不做全量 dump"——7.6K 轮的库全量建 session 目录是树爆炸）、
    agent last_seen；一次 `scan_admin_events` 导出 session→ledger 触达映射
    （SWITCH 事件 payload 自带 session_id/agent_id）；每 agent 取最新一轮的
    system prompt 摘录（V-F2b 简介源，≤ #agents 次 get）。
    """
    from datetime import datetime

    def _iso(ms: int) -> str:
        return (datetime.fromtimestamp(ms / 1000, tz=UTC)
                .strftime("%Y-%m-%d %H:%M") if ms else "")

    def _source() -> dict:
        try:
            hub.catch_up()  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001,S110
            pass
        span: dict[tuple[str, str], list[int]] = {}
        latest_key: dict[str, list[tuple[int, str]]] = {}
        #: (agent, session) -> 该会话最新一轮的 key（project 归属取样用）
        sess_latest: dict[tuple[str, str], tuple[int, str]] = {}
        for key, _ts in hub.scan_meta():  # type: ignore[attr-defined]
            parts = key.split("/")
            if len(parts) < 4:
                continue
            agent, sess = parts[1], parts[2]
            if eligible is not None:
                if not eligible(agent):
                    continue      # 判据①：无独立声明（unknown/子代理派生）不建目录
            elif agent.startswith("unknown"):
                continue          # 谓词缺席时的最低限度排除（测试/降级路径）
            try:
                ms = int(parts[3].split("-")[0])
            except ValueError:
                continue
            sp = span.setdefault((agent, sess), [ms, ms])
            sp[0], sp[1] = min(sp[0], ms), max(sp[1], ms)
            if sess_latest.get((agent, sess), (0, ""))[0] < ms:
                sess_latest[(agent, sess)] = (ms, key)
            lk = latest_key.setdefault(agent, [])
            lk.append((ms, key))
            if len(lk) > aux_sample_rounds + 8:
                lk.sort(key=lambda t: -t[0])
                del lk[aux_sample_rounds:]
        sessions: dict[str, list[dict]] = {}
        agent_last: dict[str, str] = {}
        for (agent, sess), (lo, hi) in span.items():
            sessions.setdefault(agent, []).append(
                {"session_id": sess, "first_at": _iso(lo), "last_at": _iso(hi),
                 "_hi": hi})
        for agent, rows in sessions.items():
            rows.sort(key=lambda r: -r["_hi"])
            del rows[max_sessions_per_agent:]
            agent_last[agent] = rows[0]["last_at"][:10] if rows else ""
            for r in rows:
                r.pop("_hi", None)
        sess_ledgers: dict[tuple[str, str], list[str]] = {}
        for _k, ev in hub.scan_admin_events():  # type: ignore[attr-defined]
            etype = getattr(ev.event_type, "value", str(ev.event_type))
            if etype != "ledger_switch":
                continue
            pl = ev.payload or {}
            agent = str(pl.get("agent_id") or "")
            sess = str(pl.get("session_id") or "")
            lid = str(pl.get("to_ledger_id") or pl.get("ledger_id") or "")
            if agent and sess and lid:
                lids = sess_ledgers.setdefault((agent, sess), [])
                if lid not in lids:
                    lids.append(lid)
        # 项目归属（2026-08-29 树 project 层接线）：只对进窗口的会话取样
        # （每 agent ≤max_sessions 次 get，只在树 pass 发生——wake ∧ 300s 节流）。
        # 历史轮次无 project_id ⇒ 落 global（如实：识别是今天才上线的）。
        sess_project: dict[tuple[str, str], tuple[str, str]] = {}
        for agent, rows in sessions.items():
            for r in rows:
                mk = sess_latest.get((agent, r["session_id"]))
                if not mk:
                    continue
                try:
                    t = hub.get(mk[1])  # type: ignore[attr-defined]
                except Exception:  # noqa: BLE001
                    continue
                ident = getattr(t, "identity", None)
                pid = str(getattr(ident, "project_id", "") or "")
                pname = str(getattr(ident, "project_name", "") or "")
                proot = str(getattr(ident, "project_root", "") or "")
                if pid:
                    sess_project[(agent, r["session_id"])] = (pid, pname, proot)
                # 会话名 = 首条真实 user 消息的机械摘录（零 LLM——"确定性能做的
                # 不进 LLM"，ADR §5.2 原文纪律）。env/AGENTS.md 头等机器消息跳过。
                if not r.get("name"):
                    for m in (getattr(t, "request_messages", None) or []):
                        if m.get("role") != "user":
                            continue
                        txt = _msg_text(m.get("content")).strip()
                        if (not txt
                                or txt.startswith("# AGENTS.md instructions")
                                or txt.startswith("<environment_context")
                                or txt.startswith("<env>")):
                            continue
                        r["name"] = txt[:60].replace("\n", " ")
                        break
        # 名字按 pid 聚合：project_name 字段晚于 project_id 上线，老轮次有 id
        # 没名字（live 实证：目录渲染成 `proj-10ad1f`）——同一 pid 任何一轮
        # 带了名字，所有会话共用（身份是 pid，名字只是投影，取最新非空即可）。
        pid_names: dict[str, str] = {}
        pid_roots: dict[str, str] = {}
        for (_agent, _sess), (pid, pname, proot) in sess_project.items():
            if pname:
                pid_names[pid] = pname
            if proot:
                pid_roots[pid] = proot
        if pid_names or pid_roots:
            sess_project = {k: (pid, pname or pid_names.get(pid, ""),
                                proot or pid_roots.get(pid, ""))
                            for k, (pid, pname, proot) in sess_project.items()}
        prompts: dict[str, str] = {}
        excluded: set[str] = set()
        for agent, cand in latest_key.items():
            # 🔴 跳过 aux 轮（live 病例：hermes 的"最新一轮"是标题生成子调用，
            # 其 system prompt 要求输出 JSON，便宜模型服从了**源文本里的指令**，
            # 简介蒸出 {"title": ...}）——取最近若干轮里第一条非 aux 的。
            sampled = 0
            saw_non_aux = False
            for _ms, key in sorted(cand, key=lambda t: -t[0])[:aux_sample_rounds]:
                try:
                    t = hub.get(key)  # type: ignore[attr-defined]
                except Exception:  # noqa: BLE001
                    continue
                sampled += 1
                if getattr(t, "auxiliary", False):
                    continue
                saw_non_aux = True
                if agent not in prompts:
                    for m in (getattr(t, "request_messages", None) or []):
                        if (m.get("role") == "system"
                                and isinstance(m.get("content"), str)):
                            prompts[agent] = m["content"][:prompt_excerpt_chars]
                            break
            if sampled and not saw_non_aux:
                excluded.add(agent)   # 判据②：全是 aux 轮 = 内部功能桶（裸 hermes）
        if excluded:
            # 仪器：谁被判据②挡在名册/目录树外，日志可 grep（B0 首验 hermes:default 落榜
            # 时没有这一行，只能猜）。
            logger.info("flash_tree_agents_excluded_aux", agents=sorted(excluded),
                        sample_rounds=aux_sample_rounds)
        for agent in excluded:
            sessions.pop(agent, None)
            agent_last.pop(agent, None)
            prompts.pop(agent, None)
        return {"sessions": sessions, "session_ledgers": sess_ledgers,
                "agent_last_seen": agent_last, "agent_prompts": prompts,
                "session_project": sess_project,
                "excluded": sorted(excluded)}

    return _source


def router_summarizer() -> Callable[[str], str]:
    """V-F2b 蒸馏器：便宜档一句话简介（经 Router 网关；模型自举与 consolidator 同法）。

    预算（G15 纪律，立项即报）：**每 agent 每次 system prompt 变更 1 次调用**，
    源 sha 缓存命中 = 零调用；6 个 agent 的部署一次性 ~6 次，此后近零。
    """
    from bladex_proxy.config import ProxyConfig
    from bladex_proxy.routing_config import bootstrap_distill_model
    cfg = ProxyConfig()
    model = bootstrap_distill_model(cfg.distill_model, cfg.routing_config,
                                    cfg.routing_config_path)

    def _sum(src: str) -> str:
        if not model:
            raise RuntimeError("no distill model configured")
        from bladex_proxy import router_sdk
        resp = router_sdk.completion(
            model=model,
            messages=[{"role": "system",
                       "content": "From this AI agent's system prompt excerpt, "
                                  "output exactly one line in the format "
                                  "'NAME | ABOUT' where NAME is the agent's "
                                  "short display name (1-3 words) and ABOUT "
                                  "says what it is in max 15 words. "
                                  "No quotes, no preamble."},
                      {"role": "user", "content": src}],
            temperature=0, max_tokens=48, stream=False,
            api_base=cfg.upstream_api_base or None,
            api_key=cfg.upstream_api_key or None)
        return resp.choices[0].message.content or ""

    return _sum


def router_project_summarizer() -> Callable[[str], str]:
    """项目简介蒸馏器（V-F2c）：吃项目声明文件头部（CLAUDE.md/AGENTS.md/
    README.md），产出「这个项目是什么」的 1-2 句描述——与 agent 简介的
    NAME|ABOUT 提示词**必须分开**（复用会把项目蒸成 agent 人设，
    2026-08-29 live 实证）。预算同 V-F2b：每项目每源变更 1 次。
    """
    from bladex_proxy.config import ProxyConfig
    from bladex_proxy.routing_config import bootstrap_distill_model
    cfg = ProxyConfig()
    model = bootstrap_distill_model(cfg.distill_model, cfg.routing_config,
                                    cfg.routing_config_path)

    def _sum(src: str) -> str:
        if not model:
            raise RuntimeError("no distill model configured")
        from bladex_proxy import router_sdk
        resp = router_sdk.completion(
            model=model,
            messages=[{"role": "system",
                       "content": "The user message is the head of a software "
                                  "project's README or contributor guide. "
                                  "Describe what the PROJECT itself is in 1-2 "
                                  "plain sentences (max 40 words), in the "
                                  "document's own language. It is a project, "
                                  "not an agent or person. No quotes, no "
                                  "preamble, no markdown."},
                      {"role": "user", "content": src}],
            temperature=0, max_tokens=96, stream=False,
            api_base=cfg.upstream_api_base or None,
            api_key=cfg.upstream_api_key or None)
        return resp.choices[0].message.content or ""

    return _sum

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
    from bladex_proxy.consolidator import _chdir_to_deployment_root, _load_env_file
    _load_env_file()
    _chdir_to_deployment_root()

    ap = argparse.ArgumentParser(description="BladeX Memory Flash maintenance daemon")
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
    hub.open()
    from bladex_proxy.agent_registry import cache_path as _reg_cache
    try:
        summarize = router_summarizer()
        summarize_project = router_project_summarizer()
    except Exception as e:  # noqa: BLE001 —— 蒸馏器起不来只影响简介列
        logger.warning("flash_daemon_summarizer_unavailable", error=str(e))
        summarize = None
        summarize_project = None
    import json as _json

    from bladex_proxy.agent_rules import load_agent_rules
    try:
        with open(_reg_cache(), encoding="utf-8") as _f:
            _bindings = _json.load(_f).get("bindings") or []
    except OSError:
        _bindings = []
    _rules = load_agent_rules()
    _eligible = browsing_admittance(_rules, _bindings)
    # MQ-F1：名册行来源 = bindings ∪ 简介缓存 ∪ 规则库声明（非 aux），再过准入。
    _declared = sorted({r.agent_id for r in _rules if not r.auxiliary})
    _summaries_file = os.path.join(resolve_flash_path(), ".agent_summaries.json")

    # Index 读侧投影源（USER.md/RULES.md/AGENT.md——真身在 Index，零 LLM）。
    # Index 打不开（首装/consolidator 未跑过）⇒ 投影面整体留空，不阻断。
    def _profiles():
        from bladex_proxy.storage.memory_index import MemoryIndex
        idx = MemoryIndex(cfg.index_path, embedder=None, read_only=True)
        idx.open()
        try:
            user_md, _ = idx.get_profile_md(None)
            agent_mds = {}
            for ab in idx.list_profile_agents():
                _, amd = idx.get_profile_md(None, agent_base=ab)
                if amd:
                    agent_mds[ab] = amd
            # RULES.md 与 prefetch 同源（判据一致性）：live 硬规则真身在
            # 配置 effective_hard_rules，不是 Index fact——category=="hard_rule"
            # 的 fact 生产链路上不存在（2026-08-29 核实，只在测试里），
            # 按它扫 all_facts 是无生产者的死消费点。
            rules = [f"- [MUST/NEVER] {r}"
                     for r in (cfg.effective_hard_rules or [])]
            rules_md = ("# Rules\n\n" + "\n".join(rules) + "\n") if rules else ""
            return user_md, rules_md, agent_mds
        finally:
            idx.close()

    daemon = FlashDaemon(root=resolve_flash_path(), principal=args.principal,
                         ledger_source=hub_ledger_source(hub),
                         tombstoned_source=hub_tombstoned_source(hub),
                         emit_admin_event=admin_emit(base_url, _admin_key()),
                         agents_source=registry_agents_source(
                             str(_reg_cache()), eligible=_eligible,
                             summaries_file=_summaries_file, declared=_declared),
                         tree_source=hub_tree_source(hub, eligible=_eligible),
                         summarize=summarize,
                         summarize_project=summarize_project,
                         profiles_source=_profiles)
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
            # 阻塞等门铃；有待扫的树就把超时压到树到窗的时刻。
            timeout = args.interval
            tw = daemon.tree_wait_s()
            if tw is not None:
                timeout = min(timeout, max(tw, 1.0))
            pw = daemon.pending_wait_s()          # E0.1b：有待重发的旧路径直编就别死等门铃
            if pw is not None:
                timeout = min(timeout, max(pw, 1.0))
            woken = consumer.wait(timeout)
            if woken:
                daemon.push()
            if woken or daemon.tree_wait_s() == 0.0 or daemon.pending_wait_s() == 0.0:
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
