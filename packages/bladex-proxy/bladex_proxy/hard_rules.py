"""硬规则热加载 —— 解析与生效的**唯一实现点**（批 Z1 · Z1.2，MQ-W18，2026-09-23）。

## 为什么

README「Coming next (0.3.0)」承诺 Hot-reloadable hard rules。此前 `BLADEX_HARD_RULES`
只在进程启动时由 `ProxyConfig` 读一次，而消费它的有**三个进程**：proxy（注入面、
`/admin/hard_rules`）、consolidator（导出 worker 写 Obsidian `Hard Rules.md` / Postgres
`hard_rules` 表）、flash_daemon（`behavior.json` 的 `hard_rules_ref`）。改 `.env` 要重启全部三个；
Z4 起用户在 Obsidian 改的规则落成 Hub `hard_rules_set` 事件，却**没有任何进程读它**。

## 机制：mtime / Hub 轮询（不是显式命令）

每个进程起一个守护线程（`arm()`），每 `BLADEX_HARD_RULES_RELOAD_S` 秒问两个来源：
`config/.env` 的 mtime（变了才重新解析）+ Hub 最近一条 `hard_rules_set`（反向 seek，O(1)）。

取舍：显式命令（`bladex rules reload` + admin 端点）只能通知 proxy，另外两个进程要么再开
控制通道、要么靠信号——三条通知路径、漏一条就是"未兑现"。而第二个来源（Hub 事件）本来
就是异步到达的（consolidator 回流 → proxy admin 正门 → Hub），用户手里没有"该敲命令"的时机。
轮询让三个进程**各自**对同一对来源得出同一个结论，没有谁通知谁。代价：最坏
`BLADEX_HARD_RULES_RELOAD_S` 秒（默认 5）的生效延迟，外加每周期一次 stat + 一次反向 seek。

## 两个来源的优先级：后写者赢

比较「`.env` 里 `BLADEX_HARD_RULES` 这个**值**最近一次变化的时刻」与「Hub 最近一条
`hard_rules_set` 的事件时刻」，谁晚听谁。不选"Hub 永远压过 `.env`"：那样用户在 Obsidian
改过一次之后，再改 `.env` 永远不生效，而且没有任何提示。

"值变化的时刻"不能取 `.env` 的 mtime 本身——改别的键也会动 mtime，会把 Obsidian 的改动
悄悄盖回去。故按**值的指纹**记账（`data/hard_rules_env_seen.json`）：指纹没变就沿用记下的
时刻；指纹变了（运行中被改、或停机期间被改）才取当时的 mtime 作为变化时刻。三个进程看到的
是同一个文件、同一个 mtime ⇒ 算出同一个时刻，账本文件的并发写是幂等的。

## 坏输入：保留旧规则 + 响亮告警

硬规则每轮必注，**清空成零条是最坏的失败方式**。`.env` 那一行被清空 / 引号错配 / 文件读不了、
Hub 事件载荷不是非空字符串列表 ⇒ 一律保留当前生效的规则，打 WARNING
`hard_rules_reload_rejected`（带原因与来源），不降级到默认规则。

## `.env` 被环境变量遮蔽

`BLADEX_ENV_PRECEDENCE=env`（默认）且 shell 里另有 `BLADEX_HARD_RULES` 时，启动时环境变量
就赢了 `.env`；热加载沿用同一语义（改 `.env` 不生效），但会打 WARNING
`hard_rules_env_shadowed` 说明为什么——不许静默。Hub 来源不受影响。

## 生效面

`ProxyConfig.effective_hard_rules` 是三个进程读硬规则的**唯一咽喉**：本进程 `arm()` 过 ⇒
返回这里的当前值，否则逐字旧行为（测试与一次性脚本零变化）。每次生效变化各进程打一行
`hard_rules_reloaded`（`process` / `source=env|hub` / `old` / `new` 条数）。
Obsidian 页面解析仍是 `exporters.obsidian.parse_hard_rules_page` 一处；它的产物经 admin 正门
进 Hub 时已是规则列表，本模块只校验载荷，不再解析页面。
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import structlog

logger = structlog.get_logger()

ENV_KEY = "BLADEX_HARD_RULES"
RELOAD_FLAG = "BLADEX_HARD_RULES_RELOAD_S"
SEEN_FILENAME = "hard_rules_env_seen.json"

#: 与 `config._DEFAULT_HARD_RULES` / `inject._DEFAULT_HARD_RULES` 同值（`.env` 没声明时）。
DEFAULT_HARD_RULES = [
    "NEVER use emojis in responses.",
    "MUST respond in the same language as the user's message.",
]


def parse_hard_rules_env(raw: str) -> list[str]:
    """`BLADEX_HARD_RULES` 的值 → 规则列表（`||` 分隔，去首尾空白，丢空项）。全仓唯一解析。"""
    return [item.strip() for item in (raw or "").split("||") if item.strip()]


def validate_rules_payload(payload: Any) -> list[str] | None:
    """Hub `hard_rules_set` 载荷 → 规则列表；不是非空字符串列表 ⇒ None（坏载荷）。"""
    rules = payload.get("rules") if isinstance(payload, dict) else None
    if not isinstance(rules, list) or not all(isinstance(r, str) for r in rules):
        return None
    cleaned = [r.strip() for r in rules if r.strip()]
    return cleaned or None


@dataclass(frozen=True)
class HubRules:
    """Hub 最近一条 `hard_rules_set`（`at` = 事件时刻 epoch 秒）。"""

    key: str
    at: float
    payload: dict


def hub_reader(hub: Any, *, catch_up: bool = False) -> Callable[[], HubRules | None]:
    """把一个 MemoryHub 句柄包成本模块要的读函数。secondary 句柄传 `catch_up=True`。"""
    from bladex_proxy.models import AdminEventType

    def _read() -> HubRules | None:
        if catch_up:
            hub.catch_up()
        hit = hub.latest_admin_event(AdminEventType.HARD_RULES_SET)
        if hit is None:
            return None
        key, ev = hit
        ts = ev.ts if isinstance(ev.ts, datetime) else datetime.fromisoformat(str(ev.ts))
        return HubRules(key=key, at=ts.timestamp(), payload=dict(ev.payload or {}))

    return _read


def _fingerprint(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class HardRulesReloader:
    """一个进程的硬规则生效器：`refresh()` 做一次对账，`start()` 起守护线程周期对账。"""

    def __init__(self, *, process: str, env_path: str, seen_path: str,
                 hub: Callable[[], HubRules | None] | None,
                 initial: list[str],
                 on_change: Callable[[list[str]], None] | None = None) -> None:
        self.process = process
        self._on_change = on_change
        self._env_path = env_path
        self._seen_path = seen_path
        self._hub = hub
        self._lock = threading.Lock()
        self._rules: list[str] = list(initial)
        self.source = "env"
        self._env_mtime: float | None = None
        self._env_rules: list[str] | None = None
        self._env_at = 0.0
        self._hub_key: str | None = None
        self._hub_rules: list[str] | None = None
        self._hub_at = 0.0
        self._rejected: set[tuple[str, str]] = set()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._pinned = self._env_pinned()
        if self._pinned:
            logger.warning("hard_rules_env_shadowed", process=process, env_path=env_path,
                           hint=f"{ENV_KEY} is set in the process environment and differs from "
                                "config/.env; with BLADEX_ENV_PRECEDENCE=env the environment "
                                "wins, so edits to config/.env will NOT be hot-reloaded. "
                                "Unset it in your shell (Obsidian edits still apply).")

    # ── 读 ──

    @property
    def rules(self) -> list[str]:
        with self._lock:
            return list(self._rules)

    # ── 来源 ──

    def _env_pinned(self) -> bool:
        from bladex_proxy.deployment import _precedence, parse_env_file
        file_vals = parse_env_file(self._env_path)
        env_val = os.environ.get(ENV_KEY)
        if env_val is None or ENV_KEY not in file_vals:
            return False
        return _precedence(file_vals) == "env" and env_val != file_vals[ENV_KEY]

    def _reject(self, source: str, reason: str, **kw: Any) -> None:
        if (source, reason) in self._rejected:
            return  # 同一个坏输入每周期都会被看到——只喊一次，修好后再坏再喊
        self._rejected.add((source, reason))
        logger.warning("hard_rules_reload_rejected", process=self.process, source=source,
                       reason=reason, kept=len(self._rules),
                       hint="keeping the current hard rules (an empty rule set is never "
                            "applied); fix the input and it reloads automatically", **kw)

    def _read_seen(self) -> dict:
        try:
            with open(self._seen_path, encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _write_seen(self, fp: str, at: float) -> None:
        try:
            os.makedirs(os.path.dirname(self._seen_path) or ".", exist_ok=True)
            tmp = f"{self._seen_path}.{os.getpid()}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"fingerprint": fp, "changed_at": at}, f)
            os.replace(tmp, self._seen_path)
        except OSError as e:
            logger.warning("hard_rules_seen_write_failed", path=self._seen_path, error=str(e))

    def _poll_env(self) -> bool:
        """`.env` 变了 ⇒ 更新 env 候选；返回候选是否变化。"""
        if self._pinned:
            return False
        try:
            mtime = os.stat(self._env_path).st_mtime
        except OSError as e:
            if self._env_rules is not None:
                self._reject("env", "unreadable", path=self._env_path, error=str(e))
            return False
        if mtime == self._env_mtime:
            return False
        self._env_mtime = mtime
        from bladex_proxy.deployment import parse_env_file, unbalanced_quote_lines
        bad = [n for n, k in unbalanced_quote_lines(self._env_path) if k == ENV_KEY]
        if bad:
            self._reject("env", "unbalanced_quotes", path=self._env_path, line=bad[0])
            return False
        vals = parse_env_file(self._env_path)
        raw = vals.get(ENV_KEY)
        rules = parse_hard_rules_env(raw or "")
        if not rules:
            if self._env_rules is None:
                rules = list(DEFAULT_HARD_RULES)  # 首次对账：与启动时同一个默认（未声明 / 声明为空）
            else:
                self._reject("env", "empty", path=self._env_path)
                return False
        self._rejected = {r for r in self._rejected if r[0] != "env"}
        fp = _fingerprint(raw or "")
        seen = self._read_seen()
        if seen.get("fingerprint") == fp:
            at = float(seen.get("changed_at") or mtime)
        else:
            at = mtime
            self._write_seen(fp, at)
        changed = rules != self._env_rules
        self._env_rules, self._env_at = rules, at
        return changed

    def _poll_hub(self) -> bool:
        if self._hub is None:
            return False
        try:
            hit = self._hub()
        except Exception as e:  # noqa: BLE001 —— Hub 读失败：本周期当作没变化
            logger.debug("hard_rules_hub_read_failed", process=self.process, error=str(e))
            return False
        if hit is None or hit.key == self._hub_key:
            return False
        self._hub_key = hit.key
        rules = validate_rules_payload(hit.payload)
        if rules is None:
            self._reject("hub", "bad_payload", event_key=hit.key)
            return False
        self._rejected = {r for r in self._rejected if r[0] != "hub"}
        self._hub_rules, self._hub_at = rules, hit.at
        return True

    # ── 对账 ──

    def refresh(self) -> bool:
        """一次对账；生效规则变了 ⇒ 打 `hard_rules_reloaded` 并返回 True。"""
        env_changed = self._poll_env()
        hub_changed = self._poll_hub()
        if not (env_changed or hub_changed):
            return False
        cand: list[tuple[float, str, list[str]]] = []
        if self._env_rules is not None:
            cand.append((self._env_at, "env", self._env_rules))
        if self._hub_rules is not None:
            cand.append((self._hub_at, "hub", self._hub_rules))
        if not cand:
            return False
        at, source, rules = max(cand, key=lambda c: c[0])
        with self._lock:
            old = self._rules
            if rules == old and source == self.source:
                return False
            self._rules, self.source = list(rules), source
        if rules != old:
            logger.info("hard_rules_reloaded", process=self.process, source=source,
                        old=len(old), new=len(rules),
                        env_changed_at=round(self._env_at, 3) or None,
                        hub_event=self._hub_key, hub_at=round(self._hub_at, 3) or None)
            if self._on_change is not None:
                try:
                    self._on_change(list(rules))
                except Exception as e:  # noqa: BLE001 —— 回调失败不回滚生效
                    logger.warning("hard_rules_on_change_failed", process=self.process,
                                   error=str(e))
        return rules != old

    def start(self, interval_s: float) -> None:
        def _loop() -> None:
            while not self._stop.wait(interval_s):
                try:
                    self.refresh()
                except Exception as e:  # noqa: BLE001 —— 对账失败不许杀线程
                    logger.warning("hard_rules_refresh_failed", process=self.process,
                                   error=str(e))

        self._thread = threading.Thread(target=_loop, name="bladex-hard-rules", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()


class ReloadWake:
    """consolidator 的退避 sleep 唤醒源：生效规则变了 ⇒ `has_change()` 报 `"hard_rules"` 一次。

    与 Z4b 的 `ExternalChangeProbe` 同一接口（`has_change` / `mark`），叠在它前面：
    规则变了先报自己（consolidator 就地跑一轮导出，把 `Hard Rules.md` / Postgres 表重写成
    新规则），否则委托给外部探针。没有这一层，`.env` 的改动在空载退避下最长 30 分钟才进 vault。
    """

    def __init__(self, inner: Any = None) -> None:
        self.inner = inner  # 被委托的外部探针（Z4b `ExternalChangeProbe`），可后接
        self._event = threading.Event()

    def notify(self, _rules: list[str] | None = None) -> None:
        self._event.set()

    def has_change(self) -> str | None:
        if self._event.is_set():
            self._event.clear()
            return "hard_rules"
        return self.inner.has_change() if self.inner is not None else None

    def mark(self) -> None:
        if self.inner is not None:
            self.inner.mark()


# ── 进程级单例（ProxyConfig.effective_hard_rules 读它）──────────────────────────

_ACTIVE: HardRulesReloader | None = None


def live_rules() -> list[str] | None:
    """本进程 `arm()` 过 ⇒ 当前生效的硬规则；否则 None（调用方走旧行为）。"""
    active = _ACTIVE
    return active.rules if active is not None else None


def active() -> HardRulesReloader | None:
    return _ACTIVE


def arm(process: str, *, hub: Callable[[], HubRules | None] | None = None,
        initial: list[str] | None = None, root: str | None = None,
        interval_s: float | None = None,
        on_change: Callable[[list[str]], None] | None = None) -> HardRulesReloader | None:
    """在本进程启用硬规则热加载。`BLADEX_HARD_RULES_RELOAD_S=0` ⇒ 不启用（返回 None）。

    `on_change(rules)`：生效规则变化后回调（在对账线程里调，只做轻量的"标脏/唤醒"）。
    启用时先同步对账一次（Hub 里已有 Obsidian 改动 ⇒ 启动即生效并打 `hard_rules_reloaded`），
    再起守护线程。重复调用返回已有实例。
    """
    global _ACTIVE
    if _ACTIVE is not None:
        return _ACTIVE
    from bladex_core.flags import flag_number

    from bladex_proxy.deployment import CONFIG_RELPATH, deployment_root
    if interval_s is None:
        interval_s = float(flag_number(RELOAD_FLAG))
    if interval_s <= 0:
        logger.info("hard_rules_reload_disabled", process=process, flag=RELOAD_FLAG,
                    hint="hard rules are read once at startup; changes need a restart")
        return None
    base = root or deployment_root()
    env_path = os.path.join(base, CONFIG_RELPATH)
    seen_path = os.path.join(base, "data", SEEN_FILENAME)
    if initial is None:
        initial = parse_hard_rules_env(os.environ.get(ENV_KEY, "")) or list(DEFAULT_HARD_RULES)
    r = HardRulesReloader(process=process, env_path=env_path, seen_path=seen_path,
                          hub=hub, initial=initial, on_change=on_change)
    r.refresh()
    r.start(interval_s)
    _ACTIVE = r
    logger.info("hard_rules_reload_armed", process=process, interval_s=interval_s,
                source=r.source, count=len(r.rules), env_path=env_path,
                hub=hub is not None)
    return r


def disarm() -> None:
    """测试用：停线程、清单例。"""
    global _ACTIVE
    if _ACTIVE is not None:
        _ACTIVE.stop()
    _ACTIVE = None


def wait_until(pred: Callable[[], bool], timeout_s: float, step_s: float = 0.05) -> bool:
    """测试 / 探针用的小工具：轮询直到 `pred()` 为真或超时。"""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(step_s)
    return pred()
