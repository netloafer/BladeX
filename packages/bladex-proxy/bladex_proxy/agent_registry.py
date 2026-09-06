"""Agent 注册表 — 会话粘性 + 可查询的 agent 识别记录。

解决的问题：Hermes 等 agent 会发内部子请求（标题生成、内容摘要等），
这些请求没有 agent 特征 system prompt / tool 定义，被动指纹识别不到。
但它们和主对话用同一个 API Key，时间上紧邻——
所以"同 user 最近识别过的 agent"可以把子请求归到正确的 agent。

两层用途：
  1. 会话粘性：指纹落空时，查同 user 最近 N 秒内识别过的 agent
  2. 数据记录：Memory Hub 里每条 Turn 带 agent_source，未来策略管理可按 agent 聚合

设计取舍：
  - 不持久化（进程重启重建）——单用户本地场景，首请求识别后即恢复
  - 多 agent 并发时取最近一个——子请求和主请求时间紧邻（秒级），
    用户切换 agent 是分钟级动作，时间窗口足以区分
  - 窗口可配（默认 120 秒），子请求通常在主请求后几秒内发出
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger()

# 会话粘性时间窗口：超过此秒数的"最近 agent"不再继承
_STICKY_WINDOW_SECONDS = 120

# 每个 user 最多记住多少个历史 agent 记录
_MAX_HISTORY_PER_USER = 20


@dataclass
class AgentRecord:
    """一条 agent 识别记录。"""

    agent_id: str
    identified_at: float  # time.time()
    # 触发来源的简要信息（调试用）
    trigger: str = ""  # 如 "system_prompt:hermes agent" / "tool:skill_manage"
    # G11.5: 本次识别的**传输身份键**（`bucket_unknown_agent().origin_key`：
    # UA product token + 厂商 vendor 段，**不含 tool 签名**）。
    # 同源继承要求它相同——见 lookup_sticky 的说明。
    origin_key: str = ""


@dataclass
class BucketSighting:
    """一个未识别分桶的登记条目（G11.11，供 dashboard 认领）。"""

    bucket_id: str
    #: 分桶判据串（如 "ua:deepseek-harness|vendor:deepseek-harness"）——
    #: 告诉用户"这个桶是凭什么分出来的"，是他判断"这是什么 agent"的主要依据。
    basis: str = ""
    #: 首次见到时的脱敏 header 快照（凭证已替换为 <redacted>）
    headers: dict[str, str] = field(default_factory=dict)
    first_seen: float = 0.0
    #: 见过多少轮（帮用户区分"跑了一整天的客户端"与"一次性探测"）
    count: int = 0
    #: ③/④ 漂移证据与认领建议（2026-08-26）。空 = 没有线索，就是个陌生客户端。
    #: 有值时 dashboard 直接显示"疑似 <agent> 升级"+ 证据 + 预填规则，一键认领。
    drift: dict = field(default_factory=dict)


@dataclass
class AgentRegistry:
    """进程内 agent 注册表。

    记录每个 user 最近被识别出的 agent，供会话粘性查询。
    """

    _records: dict[str, list[AgentRecord]] = field(default_factory=dict)
    #: G11.6/G11.11：待认领的未识别分桶，键 (user_id, bucket_id)。
    #:
    #: 🔴 **与 `_records` 分开存，不是偷懒**：未识别桶**有意不进 `_records`**
    #: （见 identity.py 里 FALLBACK 不 register 的说明——进了就会成为后续同源
    #: 请求的继承目标，把传输层同形的两个陌生 agent 合到一起）。但 dashboard 要
    #: 列它们供认领，所以另开一处只读登记：**列得出来，但粘不上**。
    _pending: dict[tuple[str, str], BucketSighting] = field(default_factory=dict)
    #: 已落盘的 (user, origin_key, agent) 三元组。register() 是热路径（每轮都调），
    #: **只有出现新绑定时才写盘**——否则缓存文件会随流量被反复重写。
    _persisted: set[tuple[str, str, str]] = field(default_factory=set)

    def last_seen_by_agent(self) -> dict[str, float]:
        """`agent_id -> 最近一次被识别的时间戳`（③ 时序接替判据的输入）。"""
        out: dict[str, float] = {}
        for records in self._records.values():
            for r in records:
                if r.agent_id and (r.agent_id not in out or r.identified_at > out[r.agent_id]):
                    out[r.agent_id] = r.identified_at
        return out

    def known_agent_ids(self) -> list[str]:
        """见过的真实 agent（排除 unknown 桶）——③ 传输重叠的比对面。"""
        return sorted({r.agent_id for records in self._records.values()
                       for r in records
                       if r.agent_id and not r.agent_id.startswith("unknown")})

    def note_unrecognized(
        self,
        user_id: str,
        bucket_id: str,
        basis: str = "",
        headers: dict[str, str] | None = None,
        drift: dict | None = None,
    ) -> bool:
        """登记一次未识别请求，返回**是否首次**见到这个分桶键。

        两个用途合一：
          - 首次才 warning（每轮都告警会把日志淹掉，MQ-A4 要的是"能被发现"不是刷屏）；
          - 登记待认领条目，供 `/admin/agents` 与 dashboard 的 Agents 页消费。

        进程内状态，重启后清空并重新告警一次——有意为之：重启后再报一次比永久静默好。
        代价是 dashboard 只列得出"本次启动以来见过的"未识别客户端；更早的要查
        `agent_unrecognized` 日志，或从 Memory Hub 的 `Identity.request_headers` 还原。
        """
        key = (user_id, bucket_id)
        entry = self._pending.get(key)
        if entry is not None:
            entry.count += 1
            # 证据可能后来才凑齐（比如"接替"要等老 agent 停下来才成立）——补写不覆盖
            if drift and not entry.drift:
                entry.drift = dict(drift)
                _autosave(self)
            return False
        self._pending[key] = BucketSighting(
            bucket_id=bucket_id, basis=basis,
            headers=dict(headers or {}), first_seen=time.time(), count=1,
            drift=dict(drift or {}),
        )
        _autosave(self)
        return True

    def pending_claims(self, user_id: str | None = None) -> list[BucketSighting]:
        """待认领的未识别分桶（按首次出现时间倒序）。

        :param user_id: ``None`` = 不限身份，返回全部。**dashboard 用 None**——
            未识别客户端未必带得上 API Key（没带 = `ANONYMOUS_USER_ID`，那是 auth
            开关的探测器身份，不是 `LOCAL_USER_ID`）。只列 local 会让"没配 key 的
            陌生客户端"在界面上彻底隐身，而那恰恰是最该被看见的一类。
        """
        items = [v for (u, _b), v in self._pending.items()
                 if user_id is None or u == user_id]
        return sorted(items, key=lambda s: s.first_seen, reverse=True)

    def drop_pending(self, bucket_id: str, user_id: str | None = None) -> None:
        """认领后从待认领列表移除（认领动作的落点在 journal，这里只是清 UI）。

        :param user_id: ``None`` = 跨身份删同名桶。与 `pending_claims(None)` 对称——
            列出来的时候不分身份，删的时候也不能分，否则认领完它还挂在界面上。
        """
        for key in [k for k in self._pending
                    if k[1] == bucket_id and (user_id is None or k[0] == user_id)]:
            self._pending.pop(key, None)

    def register(
        self,
        user_id: str,
        agent_id: str,
        trigger: str = "",
        origin_key: str = "",
    ) -> None:
        """记录一次成功的 agent 识别。

        :param origin_key: G11.5 —— 传输身份键（UA+vendor，不含 tool 签名），同源继承的判据。
        """
        records = self._records.setdefault(user_id, [])
        records.append(AgentRecord(
            agent_id=agent_id,
            identified_at=time.time(),
            trigger=trigger,
            origin_key=origin_key,
        ))
        # 防止无限增长
        if len(records) > _MAX_HISTORY_PER_USER:
            records.pop(0)
        if origin_key:
            triple = (user_id, origin_key, agent_id)
            if triple not in self._persisted:
                self._persisted.add(triple)
                _autosave(self)
        logger.debug("agent_registered", user_id=user_id, agent_id=agent_id,
                     trigger=trigger, origin_key=origin_key)

    def lookup_sticky(self, user_id: str, origin_key: str = "") -> str | None:
        """同源继承查询：返回同 user、**同传输身份**最近识别过的 agent。

        返回 None 表示没有可继承的记录。

        **它是"同源继承"，不是"识别"（G11.5，2026-08-18 重定位）。**

        事故背景：2026-08-18 dsh 接入后，日志里 agent_id 只有 `hermes:default`
        一个值、`unknown` 出现 0 次——dsh 的轮次被静默署名成 hermes 并写进了它的
        记忆命名空间。根因是本函数退化成了"谁最后说话，下一个陌生请求就是谁"：

        - ADR-0021 修正移除了 120s 窗口（理由见下，该理由本身仍成立）；
        - 2026-08-16 修订把个人模式 `user_id` 改成常量 `LOCAL_USER_ID`。

        两次单独看都对的改动叠在一起，使分桶键退化成**全机唯一一个桶**且无时间
        上限 → 粘性恒命中 → `identity.resolve_identity` 里排在其后的 User-Agent 与
        `unknown` 两条腿**永远走不到**（死代码）。

        修法不是把窗口加回来（那会重现下面那个真实问题），而是**补上同源判据**：
        只有传输身份（`origin_key`）相同才继承。子请求与父请求来自同一个客户端
        进程，UA 与厂商 header 一致，`origin_key` 天然相同；不同客户端则不同。

        🔴 **判据用 `origin_key` 而非 `bucket_id`**（被剧本逼出来的修正）：
        `bucket_id` 含 tool 签名，而工具集在同一客户端内部会变——主请求带全套
        工具、标题生成这类子请求一个都不带 → 子请求匹配不上父请求 → 退化成
        "每个子请求各自成桶"，正是下面那个 ADR-0021 问题的另一种形态。

        原 ADR-0021 修正的理由保留（**不要退回时间窗口方案**）：原假设"子任务和主
        请求秒级紧邻"不成立——实测 Hermes 图片描述 / 标题生成等子任务可能在主请求后
        十几分钟才发，超窗口 → `agent_id=unknown` → 子任务被当新请求误路由。

        遗留限制（同源判据解不掉，需 profile 级信号）：同 API key、**同客户端**的多
        profile（如 hermes accept/default）分桶键相同，子任务仍可能粘到最近 register
        的那个 profile。这属于 profile 级区分问题，与本函数的同源判据正交。

        :param user_id: 身份。
        :param origin_key: 本次请求的传输身份键；空串表示调用方未提供，
            此时**退回旧行为**（只按 user_id 取最近一条）以保持向后兼容。
        """
        records = self._records.get(user_id)
        if not records:
            return None

        if origin_key:
            for record in reversed(records):
                if record.origin_key == origin_key:
                    logger.info("agent_sticky_hit", user_id=user_id,
                                agent_id=record.agent_id,
                                origin_key=origin_key,
                                age_seconds=round(time.time() - record.identified_at, 1))
                    return record.agent_id
            # 有历史记录但没有同源的 —— 这正是 dsh 那种"另一个客户端"的情形。
            # 不继承，让上层落 unknown-<hash>（宁可 unknown，不可误署名）。
            logger.info("agent_sticky_miss_cross_origin", user_id=user_id,
                        origin_key=origin_key,
                        known_origins=sorted({r.origin_key for r in records if r.origin_key}))
            return None

        record = records[-1]
        logger.info("agent_sticky_hit", user_id=user_id,
                   agent_id=record.agent_id,
                   origin_key="",
                   age_seconds=round(time.time() - record.identified_at, 1))
        return record.agent_id

    def get_user_agents(self, user_id: str) -> list[str]:
        """查询某 user 用过哪些 agent（去重，供策略管理用）。"""
        records = self._records.get(user_id, [])
        seen: set[str] = set()
        result: list[str] = []
        for r in reversed(records):
            if r.agent_id not in seen and r.agent_id != "unknown":
                seen.add(r.agent_id)
                result.append(r.agent_id)
        return result

    def clear(self) -> None:
        self._records.clear()
        self._pending.clear()
        self._persisted.clear()


# 进程级单例
_agent_registry = AgentRegistry()


# ── 跨重启持久化（MQ-A10，2026-08-26）────────────────────────────────────────
#
# 🔴 **它是缓存，不是真相源** —— 所以按缓存来做（落盘 JSON，不进 Hub 事件流）。
# 归属的真相在每条 Turn 自己的 `agent_id`/`agent_source` 上，早已入 Hub；
# 本表只是"同源继承"这条推断路径的加速器，丢了只会退化成 unknown，不会丢数据。
# 落盘先例：`data/matter_judge_cache.json` 同款。
#
# 为什么非做不可（live 病例）：注册表原本是纯内存单例，`register()` 的记录和
# `note_unrecognized()` 的待认领登记**重启即空**。实测 2026-08-26 那次重启，
# Hermes 的内部子调用恰好是进程的**第一个**请求 ⇒ 同源继承无记录可继承
# （它的 `origin_key=0eaa8faf` 与 `hermes:default` **完全相同**，机制本该接住）
# ⇒ 落 `unknown-0eaa8faf`，并在那一刻被**永久**登记进待认领列表。
# 每重启一次就多一条噪声，而它本来就该被 G11.5 接住。
#
# 模块原 docstring 里"不持久化（进程重启重建）——单用户本地场景，首请求识别后即恢复"
# 与 `note_unrecognized` 里"重启后清空并重新告警一次——**有意为之**"这两句，
# 正是刚性原则 12 说的那种气味：把缺陷写成了规格。前者的前提"首请求会被识别"
# 恰恰不成立——**首请求可能就是那个识别不出来的子调用**。

import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

#: 缓存文件相对部署根的位置。
REGISTRY_CACHE_RELPATH = "data/agent_registry.json"

#: 落盘格式版本——字段变了就整份作废重建（缓存没有兼容义务）。
_CACHE_VERSION = 1


def _ttl_seconds() -> float:
    from bladex_core.flags import flag_number
    return max(0.0, flag_number("BLADEX_AGENT_REGISTRY_TTL_DAYS")) * 86400.0


def cache_path(root: str = "") -> Path:
    """缓存文件路径。**按部署根解析，不用相对 cwd**（2026-08-25 模板那次的教训）。

    🔴 `BLADEX_AGENT_REGISTRY_CACHE` 覆盖点**不是可选装饰**（2026-08-26 当场踩到）：
    凡是"按部署根发现"的新文件源，都必须同时接进测试隔离面，否则测试会读写**真实
    部署**的那一份——本次实测症状是 `agent_registry_cache_loaded pending=4`
    出现在一个全新用例里，跨用例互相污染、且按执行顺序抽风。
    与 `BLADEX_AGENT_RULES_PATH` 完全同型（那条的注释里写着同一句教训）。
    """
    # 优先级：**显式 root 参数 > env 覆盖 > 部署根**。
    # 显式参数比 env 更具体（调用方明确指定了），env 比"猜出来的部署根"更具体。
    if root:
        return Path(root) / REGISTRY_CACHE_RELPATH
    explicit = os.environ.get("BLADEX_AGENT_REGISTRY_CACHE", "").strip()
    if explicit:
        return Path(explicit)
    from bladex_proxy import deployment
    return Path(deployment.find_root() or os.getcwd()) / REGISTRY_CACHE_RELPATH


def snapshot(reg: AgentRegistry) -> dict:
    """可序列化快照。

    `_records` **按 (user, origin_key) 去重只留最近一条**：历史列表是给
    `get_user_agents` 用的运行期便利，跨重启没有保留价值，而同源继承只看最近。
    存全量会让文件随流量线性增长——缓存不该有这种性质。
    """
    latest: dict[tuple[str, str], AgentRecord] = {}
    for user_id, records in reg._records.items():
        for r in records:
            if not r.origin_key:
                continue          # 没有 origin_key 的记录对继承无用，不存
            latest[(user_id, r.origin_key)] = r
    return {
        "version": _CACHE_VERSION,
        "bindings": [
            {"user_id": u, "origin_key": ok, "agent_id": r.agent_id,
             "trigger": r.trigger, "identified_at": r.identified_at}
            for (u, ok), r in latest.items()
        ],
        "pending": [
            {"user_id": u, "bucket_id": s.bucket_id, "basis": s.basis,
             "headers": s.headers, "first_seen": s.first_seen, "count": s.count,
             "drift": s.drift}
            for (u, _b), s in reg._pending.items()
        ],
    }


def restore(reg: AgentRegistry, data: dict, *, now: float | None = None) -> tuple[int, int]:
    """从快照恢复；返回 (恢复的绑定数, 恢复的待认领数)。**过期的绑定直接丢弃**。

    过期只丢 `bindings`，不丢 `pending`：前者会**影响归属判定**（陈旧 = 可能误署名），
    后者只是一张给用户看的列表（陈旧 = 界面上多一行，代价不对等）。
    """
    if not isinstance(data, dict) or data.get("version") != _CACHE_VERSION:
        return (0, 0)
    t = time.time() if now is None else now
    ttl = _ttl_seconds()
    n_bind = 0
    for b in data.get("bindings") or []:
        try:
            at = float(b.get("identified_at") or 0.0)
            if ttl > 0 and (t - at) > ttl:
                continue
            reg._records.setdefault(str(b["user_id"]), []).append(AgentRecord(
                agent_id=str(b["agent_id"]), identified_at=at,
                trigger=str(b.get("trigger", "")), origin_key=str(b["origin_key"]),
            ))
            n_bind += 1
        except (KeyError, TypeError, ValueError):
            continue          # 单条坏记录不作废整份缓存
    n_pend = 0
    for s in data.get("pending") or []:
        try:
            key = (str(s["user_id"]), str(s["bucket_id"]))
            if key in reg._pending:
                continue
            reg._pending[key] = BucketSighting(
                bucket_id=str(s["bucket_id"]), basis=str(s.get("basis", "")),
                headers=dict(s.get("headers") or {}),
                first_seen=float(s.get("first_seen") or 0.0),
                count=int(s.get("count") or 0),
                drift=dict(s.get("drift") or {}),
            )
            n_pend += 1
        except (KeyError, TypeError, ValueError):
            continue
    return (n_bind, n_pend)


#: 🔴 自动落盘**由 `load_cache()` 武装**——也就是"只有真正把缓存接进来的进程才写它"。
#: 两个好处：① 副作用归入口点（server lifespan 调一次），import 与单测零文件写入；
#: ② 不会出现"某个测试进程悄悄把自己的假数据写进用户的 data/"（沙盒残渣那类事故）。
#: 存**解析后的路径**而不是 root：路径可能来自 env 覆盖，从它反推 root 会猜错。
_autosave_path: Path | None = None


def _autosave(reg: AgentRegistry) -> None:
    if _autosave_path is None:
        return
    _write_cache(reg, _autosave_path)


def load_cache(reg: AgentRegistry | None = None, *, root: str = "") -> tuple[int, int]:
    """启动时调用一次。**不在 import 时做**（2026-08-04 教训：导入必须无副作用）。"""
    global _autosave_path
    reg = _agent_registry if reg is None else reg
    p = cache_path(root)
    _autosave_path = p                              # 武装自动落盘
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.info("agent_registry_cache_absent", path=str(p),
                    hint="first run; the cache will be created on first new binding")
        return (0, 0)
    except (OSError, ValueError) as e:
        logger.warning("agent_registry_cache_unreadable", path=str(p), error=str(e))
        return (0, 0)
    n_bind, n_pend = restore(reg, data)
    # 恢复出来的绑定登记为"已落盘"，否则首个同源请求会立刻触发一次无谓重写
    for user_id, records in reg._records.items():
        for r in records:
            if r.origin_key:
                reg._persisted.add((user_id, r.origin_key, r.agent_id))
    logger.info("agent_registry_cache_loaded", path=str(p),
                bindings=n_bind, pending=n_pend)
    return (n_bind, n_pend)


def save_cache(reg: AgentRegistry | None = None, *, root: str = "") -> bool:
    """写缓存。失败**不致命**（缓存缺失只是退回旧行为），但要留可 grep 的告警。"""
    return _write_cache(_agent_registry if reg is None else reg, cache_path(root))


def _write_cache(reg: AgentRegistry, p: Path) -> bool:
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(snapshot(reg), ensure_ascii=False), encoding="utf-8")
        tmp.replace(p)          # 原子替换：读侧永远看不到半截文件
        return True
    except OSError as e:
        logger.warning("agent_registry_cache_write_failed", path=str(p), error=str(e))
        return False
